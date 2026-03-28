use std::collections::VecDeque;

use clap::Parser;
use imod_core::MrcMode;
use imod_math::{mean_sd, min_max_mean};
use imod_model::read_model;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;

/// Erase X-ray and hot pixel artifacts from CCD camera images.
///
/// Detects pixels that deviate significantly from their neighbors and
/// replaces them with the local mean.
#[derive(Parser)]
#[command(name = "ccderaser", about = "Erase CCD artifacts (X-rays, hot pixels)")]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Peak criterion: number of SDs above local mean to flag a pixel
    #[arg(short = 'p', long, default_value_t = 8.0)]
    peak_criterion: f32,

    /// Difference criterion: pixels whose difference from neighbor mean
    /// exceeds this many SDs of the image are replaced
    #[arg(short = 'd', long, default_value_t = 6.0)]
    diff_criterion: f32,

    /// Maximum radius of connected artifact to erase (pixels)
    #[arg(short = 'r', long, default_value_t = 4)]
    max_radius: usize,

    /// Size of border to scan for extra peaks
    #[arg(short = 'b', long, default_value_t = 2)]
    border: usize,

    /// Find and erase all artifacts automatically
    #[arg(short = 'f', long, default_value_t = true)]
    find: bool,

    /// Diffraction spike criterion: number of SDs to detect linear chains of
    /// hot pixels (horizontal/vertical runs of 3+). 0.0 = disabled.
    #[arg(long, default_value_t = 0.0)]
    spike_criterion: f32,

    /// Path to an IMOD model file. If provided, erase regions marked by model
    /// contours instead of auto-detecting artifacts.
    #[arg(long)]
    model: Option<String>,

    /// Exclude pixels within this many pixels of image edges from detection.
    #[arg(long, default_value_t = 0)]
    edge_exclude: usize,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.add_label(&format!(
        "ccderaser: peak={:.1} diff={:.1} maxrad={}",
        args.peak_criterion, args.diff_criterion, args.max_radius
    ));

    // Load model if provided for model-based erasure
    let erase_model = args.model.as_ref().map(|path| {
        read_model(path).unwrap_or_else(|e| {
            eprintln!("Error loading model {}: {}", path, e);
            std::process::exit(1);
        })
    });

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    let mut total_replaced = 0usize;

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let mut slice = Slice::from_data(nx, ny, data);

        let replaced;

        if let Some(ref model) = erase_model {
            // Model-based erasure: erase regions marked by model contours on this Z
            replaced = erase_model_regions(&mut slice, model, z as f32);
        } else {
            // Auto-detection mode
            let (_img_mean, img_sd) = mean_sd(&slice.data);
            let diff_thresh = args.diff_criterion * img_sd;

            // Compute effective scan bounds considering both border and edge_exclude
            let margin = args.border.max(args.edge_exclude);
            let y_start = margin;
            let y_end = if ny > margin { ny - margin } else { 0 };
            let x_start = margin;
            let x_end = if nx > margin { nx - margin } else { 0 };

            // First pass: find all hot seed pixels
            let mut hot_seeds: Vec<(usize, usize)> = Vec::new();
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let val = slice.get(x, y);
                    let local_mean = local_neighbor_mean(&slice, x, y);
                    let diff = (val - local_mean).abs();
                    if diff > diff_thresh {
                        hot_seeds.push((x, y));
                    }
                }
            }

            // Grow each seed into a connected artifact region (flood-fill)
            let mut visited = vec![false; nx * ny];
            let mut region_count = 0usize;

            for &(sx, sy) in &hot_seeds {
                if visited[sy * nx + sx] {
                    continue;
                }

                // BFS flood-fill to grow the artifact region
                let region = grow_artifact_region(
                    &slice, sx, sy, diff_thresh, args.max_radius, &mut visited,
                );

                if !region.is_empty() {
                    // Compute fill value from ring around the region bounding box
                    let fill_val = compute_ring_fill(&slice, &region, 2);
                    for &(rx, ry) in &region {
                        slice.set(rx, ry, fill_val);
                    }
                    region_count += region.len();
                }
            }

            // Diffraction spike detection
            let mut spike_count = 0usize;
            if args.spike_criterion > 0.0 {
                let (_img_mean2, img_sd2) = mean_sd(&slice.data);
                let spike_thresh = args.spike_criterion * img_sd2;
                let img_mean2 = {
                    let sum: f64 = slice.data.iter().map(|&v| v as f64).sum();
                    (sum / slice.data.len() as f64) as f32
                };
                spike_count = detect_and_erase_spikes(
                    &mut slice, img_mean2, spike_thresh, margin, &visited,
                );
            }

            replaced = region_count + spike_count;
        }

        total_replaced += replaced;
        if replaced > 0 {
            eprintln!("  section {}: replaced {} pixels", z, replaced);
        }

        let (smin, smax, smean) = min_max_mean(&slice.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&slice.data).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
    eprintln!("ccderaser: replaced {} total pixels in {} sections", total_replaced, nz);
}

/// Compute the mean of the 8 neighbors around (x, y), excluding the center.
fn local_neighbor_mean(slice: &Slice, x: usize, y: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let px = x as i32 + dx;
            let py = y as i32 + dy;
            if px >= 0 && px < slice.nx as i32 && py >= 0 && py < slice.ny as i32 {
                sum += slice.get(px as usize, py as usize);
                count += 1;
            }
        }
    }
    if count > 0 { sum / count as f32 } else { slice.get(x, y) }
}

/// BFS flood-fill from a seed pixel to grow an artifact region.
/// A neighbor is added if it also exceeds the threshold relative to its own
/// local neighborhood mean. Growth is bounded by max_radius from the seed.
fn grow_artifact_region(
    slice: &Slice,
    seed_x: usize,
    seed_y: usize,
    thresh: f32,
    max_radius: usize,
    visited: &mut [bool],
) -> Vec<(usize, usize)> {
    let nx = slice.nx;
    let ny = slice.ny;
    let mut region = Vec::new();
    let mut queue = VecDeque::new();

    visited[seed_y * nx + seed_x] = true;
    queue.push_back((seed_x, seed_y));

    while let Some((cx, cy)) = queue.pop_front() {
        region.push((cx, cy));

        // Check all 8 neighbors
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx_ = cx as i32 + dx;
                let ny_ = cy as i32 + dy;
                if nx_ < 0 || nx_ >= nx as i32 || ny_ < 0 || ny_ >= ny as i32 {
                    continue;
                }
                let (ux, uy) = (nx_ as usize, ny_ as usize);

                // Check distance from seed
                let dist_x = (ux as i32 - seed_x as i32).unsigned_abs() as usize;
                let dist_y = (uy as i32 - seed_y as i32).unsigned_abs() as usize;
                if dist_x > max_radius || dist_y > max_radius {
                    continue;
                }

                if visited[uy * nx + ux] {
                    continue;
                }

                let val = slice.get(ux, uy);
                let local_mean = local_neighbor_mean(slice, ux, uy);
                let diff = (val - local_mean).abs();
                if diff > thresh {
                    visited[uy * nx + ux] = true;
                    queue.push_back((ux, uy));
                }
            }
        }
    }

    region
}

/// Compute a fill value from pixels in a ring around the bounding box of the region.
fn compute_ring_fill(slice: &Slice, region: &[(usize, usize)], padding: usize) -> f32 {
    if region.is_empty() {
        return 0.0;
    }

    let mut min_x = usize::MAX;
    let mut max_x = 0usize;
    let mut min_y = usize::MAX;
    let mut max_y = 0usize;
    for &(x, y) in region {
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }

    // Build a set of region pixels for fast lookup
    let nx = slice.nx;
    let ny = slice.ny;
    let r_min_x = min_x.saturating_sub(padding);
    let r_max_x = (max_x + padding).min(nx - 1);
    let r_min_y = min_y.saturating_sub(padding);
    let r_max_y = (max_y + padding).min(ny - 1);

    // Collect ring pixels (those in the padded box but not in the region)
    let mut region_set = std::collections::HashSet::new();
    for &(x, y) in region {
        region_set.insert((x, y));
    }

    let mut ring_sum = 0.0f32;
    let mut ring_count = 0u32;
    for y in r_min_y..=r_max_y {
        for x in r_min_x..=r_max_x {
            if !region_set.contains(&(x, y)) {
                ring_sum += slice.get(x, y);
                ring_count += 1;
            }
        }
    }

    if ring_count > 0 {
        ring_sum / ring_count as f32
    } else {
        // Fallback: use first region pixel value
        let (fx, fy) = region[0];
        slice.get(fx, fy)
    }
}

/// Detect and erase diffraction spikes: linear chains of hot pixels
/// (horizontal or vertical runs of 3+ that deviate from the mean).
fn detect_and_erase_spikes(
    slice: &mut Slice,
    img_mean: f32,
    spike_thresh: f32,
    margin: usize,
    already_erased: &[bool],
) -> usize {
    let nx = slice.nx;
    let ny = slice.ny;
    let mut erased = 0usize;

    let y_start = margin;
    let y_end = if ny > margin { ny - margin } else { 0 };
    let x_start = margin;
    let x_end = if nx > margin { nx - margin } else { 0 };

    // Detect horizontal spikes
    for y in y_start..y_end {
        let mut run_start = x_start;
        while run_start < x_end {
            if already_erased[y * nx + run_start] {
                run_start += 1;
                continue;
            }
            let val = slice.get(run_start, y);
            if (val - img_mean).abs() <= spike_thresh {
                run_start += 1;
                continue;
            }
            // Start of a potential horizontal run
            let mut run_end = run_start + 1;
            while run_end < x_end
                && !already_erased[y * nx + run_end]
                && (slice.get(run_end, y) - img_mean).abs() > spike_thresh
            {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            if run_len >= 3 {
                // This is a diffraction spike -- replace with mean of pixels
                // just above and below the run
                for x in run_start..run_end {
                    let mut fill_sum = 0.0f32;
                    let mut fill_count = 0u32;
                    if y > 0 {
                        fill_sum += slice.get(x, y - 1);
                        fill_count += 1;
                    }
                    if y + 1 < ny {
                        fill_sum += slice.get(x, y + 1);
                        fill_count += 1;
                    }
                    let fill = if fill_count > 0 { fill_sum / fill_count as f32 } else { img_mean };
                    slice.set(x, y, fill);
                    erased += 1;
                }
            }
            run_start = run_end;
        }
    }

    // Detect vertical spikes
    for x in x_start..x_end {
        let mut run_start = y_start;
        while run_start < y_end {
            if already_erased[run_start * nx + x] {
                run_start += 1;
                continue;
            }
            let val = slice.get(x, run_start);
            if (val - img_mean).abs() <= spike_thresh {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < y_end
                && !already_erased[run_end * nx + x]
                && (slice.get(x, run_end) - img_mean).abs() > spike_thresh
            {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            if run_len >= 3 {
                for y in run_start..run_end {
                    let mut fill_sum = 0.0f32;
                    let mut fill_count = 0u32;
                    if x > 0 {
                        fill_sum += slice.get(x - 1, y);
                        fill_count += 1;
                    }
                    if x + 1 < nx {
                        fill_sum += slice.get(x + 1, y);
                        fill_count += 1;
                    }
                    let fill = if fill_count > 0 { fill_sum / fill_count as f32 } else { img_mean };
                    slice.set(x, y, fill);
                    erased += 1;
                }
            }
            run_start = run_end;
        }
    }

    erased
}

/// Erase regions marked by model contours on the given Z slice.
/// Each closed contour defines a polygon; pixels inside are replaced with
/// the mean of pixels just outside the contour boundary.
fn erase_model_regions(slice: &mut Slice, model: &imod_model::ImodModel, z: f32) -> usize {
    let nx = slice.nx;
    let ny = slice.ny;
    let mut total_erased = 0usize;

    for obj in &model.objects {
        for cont in &obj.contours {
            // Gather points on this Z
            let pts: Vec<(f32, f32)> = cont
                .points
                .iter()
                .filter(|p| (p.z - z).abs() < 0.6)
                .map(|p| (p.x, p.y))
                .collect();

            if pts.len() < 3 {
                // Not enough points for a polygon; erase individual points
                for &(px, py) in &pts {
                    let ix = px.round() as usize;
                    let iy = py.round() as usize;
                    if ix < nx && iy < ny {
                        let fill = local_neighbor_mean(slice, ix, iy);
                        slice.set(ix, iy, fill);
                        total_erased += 1;
                    }
                }
                continue;
            }

            // Compute bounding box
            let min_x = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min).floor() as i32;
            let max_x = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max).ceil() as i32;
            let min_y = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor() as i32;
            let max_y = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil() as i32;

            let min_x = min_x.max(0) as usize;
            let max_x = (max_x as usize).min(nx - 1);
            let min_y = min_y.max(0) as usize;
            let max_y = (max_y as usize).min(ny - 1);

            // Collect inside pixels using ray-casting point-in-polygon test
            let mut inside_pixels = Vec::new();
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, &pts) {
                        inside_pixels.push((x, y));
                    }
                }
            }

            if inside_pixels.is_empty() {
                continue;
            }

            // Compute fill from ring around the region
            let fill = compute_ring_fill(slice, &inside_pixels, 2);
            for &(px, py) in &inside_pixels {
                slice.set(px, py, fill);
            }
            total_erased += inside_pixels.len();
        }
    }

    total_erased
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(px: f32, py: f32, polygon: &[(f32, f32)]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}
