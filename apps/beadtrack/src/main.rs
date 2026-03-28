use clap::Parser;
use imod_core::Point3f;
use imod_fft::cross_correlate_2d;
use imod_math::polynomial_fit;
use imod_model::{write_model, ImodContour, ImodModel, ImodObject};
use imod_mrc::MrcReader;
use imod_transforms::read_tilt_file;

/// Track fiducial gold beads through a tilt series by template matching.
///
/// Given a seed model with initial bead positions on one or more views,
/// tracks each bead across all views using cross-correlation with an
/// extracted bead template. Supports tilt-compensated search windows,
/// residual rejection, gap filling, and correlation quality scoring.
#[derive(Parser)]
#[command(name = "beadtrack", about = "Track fiducial beads through a tilt series")]
struct Args {
    /// Input tilt series (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Seed model with initial bead positions (.mod)
    #[arg(short = 's', long)]
    seed: String,

    /// Output tracked model (.mod)
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Bead diameter in pixels
    #[arg(short = 'd', long, default_value_t = 10.0)]
    bead_diameter: f32,

    /// Search radius in pixels
    #[arg(short = 'r', long, default_value_t = 20.0)]
    search_radius: f32,

    /// Maximum residual (pixels) from fitted trajectory before rejection
    #[arg(long, default_value_t = 5.0)]
    max_residual: f32,

    /// Maximum gap size (consecutive missing views) to attempt filling
    #[arg(long, default_value_t = 5)]
    max_gap: usize,

    /// Minimum mean correlation to keep a bead
    #[arg(long, default_value_t = 0.1)]
    min_correlation: f32,

    /// Polynomial order for smooth trajectory fitting (applied after tracking
    /// and gap filling). X and Y positions are fit separately as functions of
    /// view index. Set to 0 to disable trajectory smoothing.
    #[arg(long = "trajectory-order", default_value_t = 2)]
    trajectory_order: usize,

    /// Re-extract the bead template every N views from the current tracked
    /// position, adapting to changes in bead appearance at different tilt angles.
    /// Set to 0 to always use the seed template.
    #[arg(long = "template-update", default_value_t = 10)]
    template_update: usize,

    /// Maximum elongation factor (ratio of correlation peak width in X vs Y).
    /// Beads with elongation above this threshold are flagged as potentially
    /// incorrect and excluded from the output.
    #[arg(long = "max-elongation", default_value_t = 3.0)]
    max_elongation: f32,
}

/// Result of tracking a bead at a single view.
#[derive(Clone, Copy)]
struct TrackResult {
    x: f32,
    y: f32,
    correlation: f32,
    /// Elongation factor: ratio of correlation peak width in X vs Y.
    /// Values near 1.0 indicate a round peak; high values suggest tracking error.
    elongation: f32,
}

fn main() {
    let args = Args::parse();

    let tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let seed = imod_model::read_model(&args.seed).unwrap_or_else(|e| {
        eprintln!("Error reading seed model: {}", e);
        std::process::exit(1);
    });

    // Read all sections
    let mut sections: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        sections.push(reader.read_slice_f32(z).unwrap());
    }

    // Extract seed bead positions
    let box_size = (args.bead_diameter * 1.5).ceil() as usize | 1; // odd
    let search_r = args.search_radius as usize;
    let search_size = next_pow2(box_size + 2 * search_r);

    let mut bead_seeds: Vec<(f32, f32, usize)> = Vec::new(); // (x, y, view)
    for obj in &seed.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                let view = pt.z.round() as usize;
                if view < nz {
                    bead_seeds.push((pt.x, pt.y, view));
                }
            }
        }
    }

    eprintln!(
        "beadtrack: {} seed beads, {} views, box={}px, search={}px",
        bead_seeds.len(), nz, box_size, search_r
    );
    eprintln!(
        "  max_residual={:.1}, max_gap={}, min_correlation={:.2}",
        args.max_residual, args.max_gap, args.min_correlation
    );

    // Track each bead across all views
    let mut output_model = ImodModel {
        name: "tracked beads".into(),
        xmax: h.nx,
        ymax: h.ny,
        zmax: h.nz,
        ..Default::default()
    };

    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;

    for (bi, &(seed_x, seed_y, seed_view)) in bead_seeds.iter().enumerate() {
        let mut results: Vec<Option<TrackResult>> = vec![None; nz];
        results[seed_view] = Some(TrackResult { x: seed_x, y: seed_y, correlation: 1.0, elongation: 1.0 });

        // Extract template from seed view
        let mut template = extract_box(&sections[seed_view], nx, ny, seed_x, seed_y, box_size);

        // --- Phase 1: Bidirectional tracking with adaptive template update ---
        // Track forward from seed
        let mut cur_x = seed_x;
        let mut cur_y = seed_y;
        let mut views_since_update = 0usize;
        for v in (seed_view + 1)..nz {
            let tilt_angle = tilt_angles.get(v).copied().unwrap_or(0.0);
            if let Some(tr) = track_bead_tilt(
                &template, &sections[v], nx, ny, cur_x, cur_y,
                box_size, search_size, tilt_angle,
            ) {
                results[v] = Some(tr);
                cur_x = tr.x;
                cur_y = tr.y;

                // Adaptive template update
                views_since_update += 1;
                if args.template_update > 0 && views_since_update >= args.template_update {
                    template = extract_box(&sections[v], nx, ny, tr.x, tr.y, box_size);
                    views_since_update = 0;
                }
            } else {
                break;
            }
        }

        // Reset template for backward tracking
        template = extract_box(&sections[seed_view], nx, ny, seed_x, seed_y, box_size);

        // Track backward from seed
        cur_x = seed_x;
        cur_y = seed_y;
        views_since_update = 0;
        for v in (0..seed_view).rev() {
            let tilt_angle = tilt_angles.get(v).copied().unwrap_or(0.0);
            if let Some(tr) = track_bead_tilt(
                &template, &sections[v], nx, ny, cur_x, cur_y,
                box_size, search_size, tilt_angle,
            ) {
                results[v] = Some(tr);
                cur_x = tr.x;
                cur_y = tr.y;

                // Adaptive template update
                views_since_update += 1;
                if args.template_update > 0 && views_since_update >= args.template_update {
                    template = extract_box(&sections[v], nx, ny, tr.x, tr.y, box_size);
                    views_since_update = 0;
                }
            } else {
                break;
            }
        }

        // Reset template back to seed for gap filling and re-tracking
        template = extract_box(&sections[seed_view], nx, ny, seed_x, seed_y, box_size);

        // --- Phase 2: Gap filling ---
        fill_gaps(
            &mut results, &template, &sections, nx, ny,
            box_size, &tilt_angles, args.max_gap,
        );

        // --- Phase 3: Residual rejection and re-tracking ---
        reject_and_retrack(
            &mut results, &template, &sections, nx, ny,
            box_size, &tilt_angles, args.max_residual,
        );

        // --- Phase 4: Smooth trajectory fitting ---
        if args.trajectory_order > 0 {
            smooth_trajectory(&mut results, args.trajectory_order);
        }

        // --- Phase 5: Elongation filtering ---
        let mean_elongation = elongation_stats(&results);
        if mean_elongation > args.max_elongation {
            rejected_count += 1;
            if bi < 5 || bi == bead_seeds.len() - 1 {
                eprintln!(
                    "  bead {}: REJECTED mean_elongation={:.2} > {:.1}",
                    bi + 1, mean_elongation, args.max_elongation
                );
            }
            continue;
        }

        // --- Phase 6: Correlation quality scoring ---
        let (mean_corr, min_corr, tracked_count) = correlation_stats(&results);

        if mean_corr < args.min_correlation {
            rejected_count += 1;
            if bi < 5 || bi == bead_seeds.len() - 1 {
                eprintln!(
                    "  bead {}: REJECTED mean_corr={:.3} < {:.2} (tracked {}/{} views)",
                    bi + 1, mean_corr, args.min_correlation, tracked_count, nz
                );
            }
            continue;
        }

        accepted_count += 1;

        // Create model object for this bead
        let mut contour = ImodContour::default();
        for (v, res) in results.iter().enumerate() {
            if let Some(tr) = res {
                contour.points.push(Point3f { x: tr.x, y: tr.y, z: v as f32 });
            }
        }

        let mut obj = ImodObject {
            name: format!("bead{}", bi + 1),
            red: 0.0,
            green: 1.0,
            blue: 0.0,
            ..Default::default()
        };
        obj.contours.push(contour);
        output_model.objects.push(obj);

        if bi < 5 || bi == bead_seeds.len() - 1 {
            eprintln!(
                "  bead {}: tracked {}/{} views, corr mean={:.3} min={:.3}",
                bi + 1, tracked_count, nz, mean_corr, min_corr
            );
        }
    }

    write_model(&args.output, &output_model).unwrap();
    eprintln!(
        "beadtrack: wrote {} tracked beads to {} ({} rejected by correlation)",
        accepted_count, args.output, rejected_count
    );
}

/// Extract a square box from an image, centered at (cx, cy), with out-of-bounds
/// pixels filled with the local mean.
fn extract_box(image: &[f32], nx: usize, ny: usize, cx: f32, cy: f32, size: usize) -> Vec<f32> {
    let half = size / 2;
    let mut box_data = vec![0.0f32; size * size];
    let ix = cx.round() as isize;
    let iy = cy.round() as isize;

    // Compute mean for fill
    let n_samples = image.len().min(1000);
    let sum: f64 = image.iter().take(n_samples).map(|&v| v as f64).sum();
    let fill = (sum / n_samples as f64) as f32;

    for by in 0..size {
        for bx in 0..size {
            let sx = ix - half as isize + bx as isize;
            let sy = iy - half as isize + by as isize;
            if sx >= 0 && sx < nx as isize && sy >= 0 && sy < ny as isize {
                box_data[by * size + bx] = image[sy as usize * nx + sx as usize];
            } else {
                box_data[by * size + bx] = fill;
            }
        }
    }
    box_data
}

/// Extract a box stretched along X by the given factor using bilinear interpolation.
/// The output is always `size x size`, but samples from a wider X range.
fn extract_box_stretched(
    image: &[f32], nx: usize, ny: usize,
    cx: f32, cy: f32, size: usize, x_stretch: f32,
) -> Vec<f32> {
    let half = size as f32 / 2.0;
    let mut box_data = vec![0.0f32; size * size];

    // Compute mean for fill
    let n_samples = image.len().min(1000);
    let sum: f64 = image.iter().take(n_samples).map(|&v| v as f64).sum();
    let fill = (sum / n_samples as f64) as f32;

    for by in 0..size {
        for bx in 0..size {
            // Map output pixel to input coordinate, stretching X
            let src_x = cx + (bx as f32 - half) * x_stretch;
            let src_y = cy + (by as f32 - half);

            box_data[by * size + bx] = bilinear_sample(image, nx, ny, src_x, src_y, fill);
        }
    }
    box_data
}

/// Bilinear interpolation sample from an image.
fn bilinear_sample(image: &[f32], nx: usize, ny: usize, x: f32, y: f32, fill: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sample = |ix: isize, iy: isize| -> f32 {
        if ix >= 0 && ix < nx as isize && iy >= 0 && iy < ny as isize {
            image[iy as usize * nx + ix as usize]
        } else {
            fill
        }
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x0 + 1, y0);
    let v01 = sample(x0, y0 + 1);
    let v11 = sample(x0 + 1, y0 + 1);

    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}

/// Track a bead with tilt-compensated search window.
/// The search area is stretched by 1/cos(tilt_angle) along X to compensate for
/// foreshortening at high tilt angles.
fn track_bead_tilt(
    template: &[f32],
    image: &[f32],
    nx: usize,
    ny: usize,
    pred_x: f32,
    pred_y: f32,
    box_size: usize,
    fft_size: usize,
    tilt_angle_deg: f32,
) -> Option<TrackResult> {
    let tilt_rad = tilt_angle_deg.to_radians();
    let cos_tilt = tilt_rad.cos().abs().max(0.1); // clamp to avoid extreme stretch
    let x_stretch = 1.0 / cos_tilt;

    // Extract search area with tilt compensation (stretched along X)
    let search_box = extract_box_stretched(image, nx, ny, pred_x, pred_y, fft_size, x_stretch);

    // Pad template to fft_size (also stretched to match the search box geometry)
    let tmpl_stretched = stretch_template(template, box_size, fft_size, x_stretch);

    let cc = cross_correlate_2d(&search_box, &tmpl_stretched, fft_size, fft_size);

    // Normalize correlation to get a meaningful score
    let cc_max = cc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cc_min = cc.iter().cloned().fold(f32::INFINITY, f32::min);
    let cc_range = (cc_max - cc_min).max(1e-10);

    // Find peak
    let mut max_val = f32::NEG_INFINITY;
    let mut mx = 0usize;
    let mut my = 0usize;
    for y in 0..fft_size {
        for x in 0..fft_size {
            if cc[y * fft_size + x] > max_val {
                max_val = cc[y * fft_size + x];
                mx = x;
                my = y;
            }
        }
    }

    let norm_corr = (max_val - cc_min) / cc_range;

    // Compute elongation: measure the half-width of the correlation peak in X and Y
    // at half-maximum level, then take the ratio.
    let half_max = (max_val + cc_min) / 2.0;
    let mut width_x = 1.0f32;
    let mut width_y = 1.0f32;

    // Measure width in X direction
    for dx_step in 1..(fft_size / 2) {
        let px = (mx + dx_step) % fft_size;
        if cc[my * fft_size + px] < half_max {
            width_x = dx_step as f32;
            break;
        }
    }
    // Measure width in Y direction
    for dy_step in 1..(fft_size / 2) {
        let py = (my + dy_step) % fft_size;
        if cc[py * fft_size + mx] < half_max {
            width_y = dy_step as f32;
            break;
        }
    }
    let elongation = if width_y > 0.0 { (width_x / width_y).max(width_y / width_x) } else { 1.0 };

    // Convert peak to shift. The shift in X needs to be scaled back by x_stretch
    // because the search box was stretched.
    let dx_raw = if mx > fft_size / 2 { mx as f32 - fft_size as f32 } else { mx as f32 };
    let dy = if my > fft_size / 2 { my as f32 - fft_size as f32 } else { my as f32 };
    let dx = dx_raw * x_stretch;

    let new_x = pred_x + dx;
    let new_y = pred_y + dy;

    // Reject if out of bounds
    if new_x < 0.0 || new_x >= nx as f32 || new_y < 0.0 || new_y >= ny as f32 {
        return None;
    }

    Some(TrackResult { x: new_x, y: new_y, correlation: norm_corr, elongation })
}

/// Stretch a template along X and pad it into an fft_size buffer.
fn stretch_template(template: &[f32], box_size: usize, fft_size: usize, x_stretch: f32) -> Vec<f32> {
    let mut padded = vec![0.0f32; fft_size * fft_size];
    let offset = (fft_size - box_size) / 2;
    let half = box_size as f32 / 2.0;

    // Compute template mean for fill
    let tmpl_mean = template.iter().sum::<f32>() / template.len() as f32;

    for y in 0..box_size {
        for x in 0..box_size {
            // Map destination pixel back to source pixel in original template
            let src_x = half + (x as f32 - half) * x_stretch;
            let src_y = y as f32;

            let val = bilinear_sample_box(template, box_size, box_size, src_x, src_y, tmpl_mean);
            padded[(y + offset) * fft_size + (x + offset)] = val;
        }
    }
    padded
}

/// Bilinear sample from a box (not a full image).
fn bilinear_sample_box(data: &[f32], w: usize, h: usize, x: f32, y: f32, fill: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sample = |ix: isize, iy: isize| -> f32 {
        if ix >= 0 && ix < w as isize && iy >= 0 && iy < h as isize {
            data[iy as usize * w + ix as usize]
        } else {
            fill
        }
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x0 + 1, y0);
    let v01 = sample(x0, y0 + 1);
    let v11 = sample(x0 + 1, y0 + 1);

    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}

/// Fill gaps of up to `max_gap` consecutive missing views by interpolating
/// positions from neighbors and re-attempting tracking with a tight search.
fn fill_gaps(
    results: &mut Vec<Option<TrackResult>>,
    template: &[f32],
    sections: &[Vec<f32>],
    nx: usize,
    ny: usize,
    box_size: usize,
    tilt_angles: &[f32],
    max_gap: usize,
) {
    let nz = results.len();
    // Small search for gap filling: half the normal search radius
    let gap_search_r = box_size;
    let gap_fft_size = next_pow2(box_size + 2 * gap_search_r);

    // Find gaps
    let mut v = 0;
    while v < nz {
        if results[v].is_some() {
            v += 1;
            continue;
        }
        // Found start of a gap
        let gap_start = v;
        while v < nz && results[v].is_none() {
            v += 1;
        }
        let gap_end = v; // exclusive
        let gap_len = gap_end - gap_start;

        if gap_len > max_gap {
            continue;
        }

        // Find bounding tracked positions
        let before = if gap_start > 0 { results[gap_start - 1] } else { None };
        let after = if gap_end < nz { results[gap_end] } else { None };

        if before.is_none() && after.is_none() {
            continue;
        }

        // Interpolate positions for each gap view
        for g in gap_start..gap_end {
            let interp = interpolate_position(before, after, gap_start, gap_end, g);
            if let Some((ix, iy)) = interp {
                let tilt_angle = tilt_angles.get(g).copied().unwrap_or(0.0);
                if let Some(tr) = track_bead_tilt(
                    template, &sections[g], nx, ny, ix, iy,
                    box_size, gap_fft_size, tilt_angle,
                ) {
                    results[g] = Some(tr);
                }
            }
        }
    }
}

/// Interpolate a position between before (at gap_start-1) and after (at gap_end).
fn interpolate_position(
    before: Option<TrackResult>,
    after: Option<TrackResult>,
    gap_start: usize,
    gap_end: usize,
    target: usize,
) -> Option<(f32, f32)> {
    match (before, after) {
        (Some(b), Some(a)) => {
            // Linear interpolation
            let total = (gap_end - gap_start + 1) as f32;
            let t = (target - gap_start + 1) as f32 / total;
            Some((
                b.x + t * (a.x - b.x),
                b.y + t * (a.y - b.y),
            ))
        }
        (Some(b), None) => Some((b.x, b.y)),
        (None, Some(a)) => Some((a.x, a.y)),
        (None, None) => None,
    }
}

/// Fit a smooth trajectory and reject outliers, then re-track rejected views.
fn reject_and_retrack(
    results: &mut Vec<Option<TrackResult>>,
    template: &[f32],
    sections: &[Vec<f32>],
    nx: usize,
    ny: usize,
    box_size: usize,
    tilt_angles: &[f32],
    max_residual: f32,
) {
    let nz = results.len();

    // Collect tracked positions for fitting
    let mut views: Vec<f32> = Vec::new();
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for (v, res) in results.iter().enumerate() {
        if let Some(tr) = res {
            views.push(v as f32);
            xs.push(tr.x);
            ys.push(tr.y);
        }
    }

    if views.len() < 3 {
        return; // Not enough points to fit
    }

    // Fit quadratic: pos = a*v^2 + b*v + c
    let fit_x = fit_quadratic(&views, &xs);
    let fit_y = fit_quadratic(&views, &ys);

    // Tight search for re-tracking
    let retrack_search_r = box_size;
    let retrack_fft_size = next_pow2(box_size + 2 * retrack_search_r);

    // Check residuals and reject outliers
    for v in 0..nz {
        if let Some(tr) = results[v] {
            let vf = v as f32;
            let pred_x = fit_x.0 * vf * vf + fit_x.1 * vf + fit_x.2;
            let pred_y = fit_y.0 * vf * vf + fit_y.1 * vf + fit_y.2;
            let residual = ((tr.x - pred_x).powi(2) + (tr.y - pred_y).powi(2)).sqrt();

            if residual > max_residual {
                // Reject this position and re-track with tighter search centered on prediction
                let tilt_angle = tilt_angles.get(v).copied().unwrap_or(0.0);
                results[v] = track_bead_tilt(
                    template, &sections[v], nx, ny, pred_x, pred_y,
                    box_size, retrack_fft_size, tilt_angle,
                );
            }
        }
    }
}

/// Fit a quadratic y = a*x^2 + b*x + c using least squares.
/// Returns (a, b, c).
fn fit_quadratic(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let n = x.len() as f64;
    if n < 3.0 {
        // Fall back to linear or constant
        if n < 2.0 {
            return (0.0, 0.0, y.first().copied().unwrap_or(0.0));
        }
        let b = (y[1] - y[0]) / (x[1] - x[0]).max(1.0);
        let c = y[0] as f32 - b * x[0];
        return (0.0, b, c);
    }

    // Normal equations for quadratic fit
    let mut sx0 = 0.0f64; // sum(1)
    let mut sx1 = 0.0f64; // sum(x)
    let mut sx2 = 0.0f64; // sum(x^2)
    let mut sx3 = 0.0f64; // sum(x^3)
    let mut sx4 = 0.0f64; // sum(x^4)
    let mut sy0 = 0.0f64; // sum(y)
    let mut sy1 = 0.0f64; // sum(x*y)
    let mut sy2 = 0.0f64; // sum(x^2*y)

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xd = xi as f64;
        let yd = yi as f64;
        sx0 += 1.0;
        sx1 += xd;
        sx2 += xd * xd;
        sx3 += xd * xd * xd;
        sx4 += xd * xd * xd * xd;
        sy0 += yd;
        sy1 += xd * yd;
        sy2 += xd * xd * yd;
    }

    // Solve 3x3 system:
    // | sx4 sx3 sx2 | | a |   | sy2 |
    // | sx3 sx2 sx1 | | b | = | sy1 |
    // | sx2 sx1 sx0 | | c |   | sy0 |
    let det = sx4 * (sx2 * sx0 - sx1 * sx1)
            - sx3 * (sx3 * sx0 - sx1 * sx2)
            + sx2 * (sx3 * sx1 - sx2 * sx2);

    if det.abs() < 1e-12 {
        // Degenerate, fall back to mean
        return (0.0, 0.0, (sy0 / sx0) as f32);
    }

    let a = (sy2 * (sx2 * sx0 - sx1 * sx1)
           - sx3 * (sy1 * sx0 - sx1 * sy0)
           + sx2 * (sy1 * sx1 - sx2 * sy0)) / det;

    let b = (sx4 * (sy1 * sx0 - sx1 * sy0)
           - sy2 * (sx3 * sx0 - sx1 * sx2)
           + sx2 * (sx3 * sy0 - sy1 * sx2)) / det;

    let c = (sx4 * (sx2 * sy0 - sy1 * sx1)
           - sx3 * (sx3 * sy0 - sy1 * sx2)
           + sy2 * (sx3 * sx1 - sx2 * sx2)) / det;

    (a as f32, b as f32, c as f32)
}

/// Compute correlation statistics for a set of track results.
/// Returns (mean_correlation, min_correlation, tracked_count).
fn correlation_stats(results: &[Option<TrackResult>]) -> (f32, f32, usize) {
    let mut sum = 0.0f32;
    let mut min = f32::INFINITY;
    let mut count = 0usize;

    for res in results {
        if let Some(tr) = res {
            sum += tr.correlation;
            if tr.correlation < min {
                min = tr.correlation;
            }
            count += 1;
        }
    }

    if count == 0 {
        (0.0, 0.0, 0)
    } else {
        (sum / count as f32, min, count)
    }
}

/// Fit a smooth polynomial trajectory through all tracked positions and replace
/// the X/Y coordinates with the fitted values. Uses imod_math::polynomial_fit
/// for X and Y independently as functions of view index.
fn smooth_trajectory(results: &mut Vec<Option<TrackResult>>, order: usize) {
    // Collect tracked positions
    let mut views: Vec<f32> = Vec::new();
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for (v, res) in results.iter().enumerate() {
        if let Some(tr) = res {
            views.push(v as f32);
            xs.push(tr.x);
            ys.push(tr.y);
        }
    }

    let n = views.len();
    if n <= order {
        return; // Not enough points to fit
    }

    // Fit polynomial to X coordinates
    let fit_x = match polynomial_fit(&views, &xs, n, order) {
        Ok(fit) => fit,
        Err(_) => return, // Fall back to no smoothing on error
    };

    // Fit polynomial to Y coordinates
    let fit_y = match polynomial_fit(&views, &ys, n, order) {
        Ok(fit) => fit,
        Err(_) => return,
    };

    // Evaluate the fitted polynomial at each tracked view and replace coordinates
    for (v, res) in results.iter_mut().enumerate() {
        if let Some(tr) = res {
            let vf = v as f32;
            // Evaluate: intercept + slopes[0]*x + slopes[1]*x^2 + ...
            let mut fitted_x = fit_x.intercept;
            let mut fitted_y = fit_y.intercept;
            for (k, (&sx, &sy)) in fit_x.slopes.iter().zip(fit_y.slopes.iter()).enumerate() {
                let power = (k + 1) as i32;
                let vp = vf.powi(power);
                fitted_x += sx * vp;
                fitted_y += sy * vp;
            }
            tr.x = fitted_x;
            tr.y = fitted_y;
        }
    }
}

/// Compute the mean elongation factor across all tracked views.
/// Returns 1.0 if no views are tracked.
fn elongation_stats(results: &[Option<TrackResult>]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for res in results {
        if let Some(tr) = res {
            sum += tr.elongation;
            count += 1;
        }
    }
    if count == 0 { 1.0 } else { sum / count as f32 }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}
