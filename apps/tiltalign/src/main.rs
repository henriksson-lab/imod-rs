use clap::Parser;
use imod_model::read_model;
use imod_transforms::{write_tilt_file, write_xf_file, LinearTransform};

/// Solve for the alignment of a tilt series from fiducial bead positions.
///
/// Given an IMOD model file with tracked fiducial positions across a tilt series
/// and the tilt angles, solves for per-view translations, rotations, magnifications,
/// and tilt angle refinements that minimize the reprojection residual error.
///
/// Uses a projection model where for each bead at 3D position (X, Y, Z),
/// the projected position in view v is:
///   proj_x = mag_v * (cos(rot_v)*(X*cos(tilt_v) + Z*sin(tilt_v)) - sin(rot_v)*Y) + dx_v
///   proj_y = mag_v * (sin(rot_v)*(X*cos(tilt_v) + Z*sin(tilt_v)) + cos(rot_v)*Y) + dy_v
#[derive(Parser)]
#[command(name = "tiltalign", about = "Fiducial-based tilt series alignment")]
struct Args {
    /// Model file with fiducial positions (.mod or .fid)
    #[arg(short = 'm', long)]
    model: String,

    /// Tilt angle file (.tlt or .rawtlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Output transform file (.xf)
    #[arg(short = 'o', long)]
    output_xf: String,

    /// Output refined tilt angles (.tlt)
    #[arg(long)]
    output_tilt: Option<String>,

    /// Output 3D fiducial coordinates file
    #[arg(long)]
    output_xyz: Option<String>,

    /// Image X dimension (for centering transforms)
    #[arg(long, default_value_t = 0)]
    image_nx: i32,

    /// Image Y dimension
    #[arg(long, default_value_t = 0)]
    image_ny: i32,

    /// Solve for per-view rotation angle
    #[arg(long)]
    solve_rotation: bool,

    /// Solve for per-view magnification
    #[arg(long)]
    solve_mag: bool,

    /// Solve for tilt angle refinement
    #[arg(long)]
    solve_tilt: bool,

    /// Number of iterations
    #[arg(short = 'n', long, default_value_t = 20)]
    iterations: usize,

    /// Minimum number of views a bead must appear in to be used
    #[arg(long, default_value_t = 3)]
    min_views: usize,

    /// Reference view index (0-based) for fixing rotation and magnification
    #[arg(long)]
    ref_view: Option<usize>,
}

/// A fiducial track: observed (x, y) position at each view, or None if not visible.
type Track = Vec<Option<(f64, f64)>>;

/// 3D bead position.
#[derive(Clone, Copy, Debug)]
struct Bead3D {
    x: f64,
    y: f64,
    z: f64,
}

/// Per-view alignment parameters.
#[derive(Clone, Debug)]
struct ViewParams {
    dx: f64,
    dy: f64,
    rotation: f64, // radians
    mag: f64,
    tilt: f64, // radians
}

impl ViewParams {
    fn new(tilt_deg: f64) -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            rotation: 0.0,
            mag: 1.0,
            tilt: tilt_deg.to_radians(),
        }
    }
}

/// Project a 3D bead position through view parameters to get 2D position.
fn project(bead: &Bead3D, vp: &ViewParams) -> (f64, f64) {
    let cos_tilt = vp.tilt.cos();
    let sin_tilt = vp.tilt.sin();
    let cos_rot = vp.rotation.cos();
    let sin_rot = vp.rotation.sin();

    // The tilt-axis projection: X component foreshortened by cos(tilt), Z adds via sin(tilt)
    let xp = bead.x * cos_tilt + bead.z * sin_tilt;
    let yp = bead.y;

    // Apply in-plane rotation and magnification
    let proj_x = vp.mag * (cos_rot * xp - sin_rot * yp) + vp.dx;
    let proj_y = vp.mag * (sin_rot * xp + cos_rot * yp) + vp.dy;

    (proj_x, proj_y)
}

/// Estimate 3D bead positions from current alignment parameters via least-squares.
///
/// For each bead, we solve a linear system derived from the projection equations.
/// The projection for view v gives two equations:
///   obs_x_v = mag_v * (cos_rot_v * (X*cos_tilt_v + Z*sin_tilt_v) - sin_rot_v * Y) + dx_v
///   obs_y_v = mag_v * (sin_rot_v * (X*cos_tilt_v + Z*sin_tilt_v) + cos_rot_v * Y) + dy_v
///
/// Rearranging:
///   (obs_x_v - dx_v)/mag_v = cos_rot_v * cos_tilt_v * X - sin_rot_v * Y + cos_rot_v * sin_tilt_v * Z
///   (obs_y_v - dy_v)/mag_v = sin_rot_v * cos_tilt_v * X + cos_rot_v * Y + sin_rot_v * sin_tilt_v * Z
///
/// This is a linear system A * [X, Y, Z]^T = b, solved via normal equations.
fn estimate_beads(
    tracks: &[Track],
    params: &[ViewParams],
    n_views: usize,
) -> Vec<Bead3D> {
    let n_beads = tracks.len();
    let mut beads = Vec::with_capacity(n_beads);

    for track in tracks {
        // Build normal equations: A^T A x = A^T b, where A is (2*n_obs x 3), b is (2*n_obs)
        let mut ata = [0.0f64; 9]; // 3x3 row-major
        let mut atb = [0.0f64; 3];

        for vi in 0..n_views {
            if let Some((obs_x, obs_y)) = track[vi] {
                let vp = &params[vi];
                let cos_tilt = vp.tilt.cos();
                let sin_tilt = vp.tilt.sin();
                let cos_rot = vp.rotation.cos();
                let sin_rot = vp.rotation.sin();
                let inv_mag = 1.0 / vp.mag;

                // Residual after removing shifts and mag
                let rx = (obs_x - vp.dx) * inv_mag;
                let ry = (obs_y - vp.dy) * inv_mag;

                // Row for x equation: coefficients of [X, Y, Z]
                let ax0 = cos_rot * cos_tilt;
                let ax1 = -sin_rot;
                let ax2 = cos_rot * sin_tilt;

                // Row for y equation
                let ay0 = sin_rot * cos_tilt;
                let ay1 = cos_rot;
                let ay2 = sin_rot * sin_tilt;

                // Accumulate A^T A
                ata[0] += ax0 * ax0 + ay0 * ay0;
                ata[1] += ax0 * ax1 + ay0 * ay1;
                ata[2] += ax0 * ax2 + ay0 * ay2;
                ata[3] += ax1 * ax0 + ay1 * ay0;
                ata[4] += ax1 * ax1 + ay1 * ay1;
                ata[5] += ax1 * ax2 + ay1 * ay2;
                ata[6] += ax2 * ax0 + ay2 * ay0;
                ata[7] += ax2 * ax1 + ay2 * ay1;
                ata[8] += ax2 * ax2 + ay2 * ay2;

                // Accumulate A^T b
                atb[0] += ax0 * rx + ay0 * ry;
                atb[1] += ax1 * rx + ay1 * ry;
                atb[2] += ax2 * rx + ay2 * ry;
            }
        }

        // Solve 3x3 system using Cramer's rule (more robust for small systems than gaussj)
        let bead = solve_3x3(&ata, &atb);
        beads.push(bead);
    }

    beads
}

/// Solve a 3x3 linear system using Cramer's rule.
fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Bead3D {
    // a is row-major: a[row*3+col]
    let det = a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);

    if det.abs() < 1e-30 {
        return Bead3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    let inv = 1.0 / det;

    let x = (b[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (b[1] * a[8] - a[5] * b[2])
        + a[2] * (b[1] * a[7] - a[4] * b[2]))
        * inv;

    let y = (a[0] * (b[1] * a[8] - a[5] * b[2])
        - b[0] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * b[2] - b[1] * a[6]))
        * inv;

    let z = (a[0] * (a[4] * b[2] - b[1] * a[7])
        - a[1] * (a[3] * b[2] - b[1] * a[6])
        + b[0] * (a[3] * a[7] - a[4] * a[6]))
        * inv;

    Bead3D { x, y, z }
}

/// Update per-view alignment parameters by minimizing projection residuals.
///
/// For each view, given fixed 3D bead positions, we solve for the alignment
/// parameters that minimize sum of squared residuals. Translation is always
/// solved. Rotation, magnification, and tilt are solved when enabled.
///
/// The approach uses linearization around current parameter values:
/// for small perturbations (d_rot, d_mag, d_tilt, d_dx, d_dy), the projected
/// position changes approximately linearly. We set up a least-squares system
/// for these perturbations.
fn update_view_params(
    tracks: &[Track],
    beads: &[Bead3D],
    params: &mut [ViewParams],
    n_views: usize,
    solve_rotation: bool,
    solve_mag: bool,
    solve_tilt: bool,
    ref_view: usize,
) {
    for vi in 0..n_views {
        // Count observations in this view
        let obs: Vec<(usize, f64, f64)> = tracks
            .iter()
            .enumerate()
            .filter_map(|(bi, track)| track[vi].map(|(x, y)| (bi, x, y)))
            .collect();

        if obs.is_empty() {
            continue;
        }

        // Determine which parameters to solve for this view
        // Reference view has rotation=0 and mag=1 fixed
        let solve_rot_here = solve_rotation && vi != ref_view;
        let solve_mag_here = solve_mag && vi != ref_view;
        let solve_tilt_here = solve_tilt;

        // Number of unknowns: always dx, dy; optionally d_rot, d_mag, d_tilt
        let n_unknowns = 2
            + if solve_rot_here { 1 } else { 0 }
            + if solve_mag_here { 1 } else { 0 }
            + if solve_tilt_here { 1 } else { 0 };

        let vp = &params[vi];
        let cos_tilt = vp.tilt.cos();
        let sin_tilt = vp.tilt.sin();
        let cos_rot = vp.rotation.cos();
        let sin_rot = vp.rotation.sin();

        // Build normal equations: J^T J delta = J^T r
        // where J is the Jacobian of residuals w.r.t. parameters
        // and r is the residual vector
        let mut jtj = vec![0.0f64; n_unknowns * n_unknowns];
        let mut jtr = vec![0.0f64; n_unknowns];

        for &(bi, obs_x, obs_y) in &obs {
            let bead = &beads[bi];
            let (pred_x, pred_y) = project(bead, vp);
            let res_x = obs_x - pred_x;
            let res_y = obs_y - pred_y;

            // Jacobian row for this observation (2 rows: x and y)
            // Derivatives of proj_x and proj_y w.r.t. each parameter:
            // xp = bead.x * cos_tilt + bead.z * sin_tilt
            // proj_x = mag * (cos_rot * xp - sin_rot * Y) + dx
            // proj_y = mag * (sin_rot * xp + cos_rot * Y) + dy

            let xp = bead.x * cos_tilt + bead.z * sin_tilt;

            // d(proj_x)/d(dx) = 1, d(proj_y)/d(dx) = 0
            // d(proj_x)/d(dy) = 0, d(proj_y)/d(dy) = 1
            // d(proj_x)/d(rot) = mag * (-sin_rot * xp - cos_rot * Y)
            // d(proj_y)/d(rot) = mag * (cos_rot * xp - sin_rot * Y)
            // d(proj_x)/d(mag) = cos_rot * xp - sin_rot * Y
            // d(proj_y)/d(mag) = sin_rot * xp + cos_rot * Y
            // d(proj_x)/d(tilt) = mag * cos_rot * (-bead.x * sin_tilt + bead.z * cos_tilt)
            // d(proj_y)/d(tilt) = mag * sin_rot * (-bead.x * sin_tilt + bead.z * cos_tilt)

            let dxp_dtilt = -bead.x * sin_tilt + bead.z * cos_tilt;

            // Build Jacobian columns
            let mut jx = Vec::with_capacity(n_unknowns);
            let mut jy = Vec::with_capacity(n_unknowns);

            // dx
            jx.push(1.0);
            jy.push(0.0);
            // dy
            jx.push(0.0);
            jy.push(1.0);

            if solve_rot_here {
                jx.push(vp.mag * (-sin_rot * xp - cos_rot * bead.y));
                jy.push(vp.mag * (cos_rot * xp - sin_rot * bead.y));
            }
            if solve_mag_here {
                jx.push(cos_rot * xp - sin_rot * bead.y);
                jy.push(sin_rot * xp + cos_rot * bead.y);
            }
            if solve_tilt_here {
                jx.push(vp.mag * cos_rot * dxp_dtilt);
                jy.push(vp.mag * sin_rot * dxp_dtilt);
            }

            // Accumulate J^T J and J^T r
            for p in 0..n_unknowns {
                for q in 0..n_unknowns {
                    jtj[p * n_unknowns + q] += jx[p] * jx[q] + jy[p] * jy[q];
                }
                jtr[p] += jx[p] * res_x + jy[p] * res_y;
            }
        }

        // Solve normal equations using gaussj
        let n = n_unknowns;
        let mut a_flat: Vec<f32> = jtj.iter().map(|&v| v as f32).collect();
        let mut b_flat: Vec<f32> = jtr.iter().map(|&v| v as f32).collect();

        if imod_math::gaussj::gaussj(&mut a_flat, n, n, &mut b_flat, 1, 1).is_ok() {
            let delta: Vec<f64> = b_flat.iter().map(|&v| v as f64).collect();

            let vp = &mut params[vi];
            vp.dx += delta[0];
            vp.dy += delta[1];

            let mut idx = 2;
            if solve_rot_here {
                vp.rotation += delta[idx];
                idx += 1;
            }
            if solve_mag_here {
                vp.mag += delta[idx];
                // Clamp magnification to reasonable range
                vp.mag = vp.mag.clamp(0.5, 2.0);
                idx += 1;
            }
            if solve_tilt_here {
                vp.tilt += delta[idx];
            }
        }
    }
}

/// Compute RMS residual and per-view residuals.
fn compute_residuals(
    tracks: &[Track],
    beads: &[Bead3D],
    params: &[ViewParams],
    n_views: usize,
) -> (f64, Vec<f64>, Vec<usize>) {
    let mut total_sum_sq = 0.0f64;
    let mut total_count = 0usize;
    let mut view_sum_sq = vec![0.0f64; n_views];
    let mut view_count = vec![0usize; n_views];

    for (bi, track) in tracks.iter().enumerate() {
        for vi in 0..n_views {
            if let Some((obs_x, obs_y)) = track[vi] {
                let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                let dx = obs_x - pred_x;
                let dy = obs_y - pred_y;
                let sq = dx * dx + dy * dy;
                total_sum_sq += sq;
                total_count += 1;
                view_sum_sq[vi] += sq;
                view_count[vi] += 1;
            }
        }
    }

    let rms = if total_count > 0 {
        (total_sum_sq / total_count as f64).sqrt()
    } else {
        0.0
    };

    // Per-view RMS
    let view_rms: Vec<f64> = (0..n_views)
        .map(|vi| {
            if view_count[vi] > 0 {
                (view_sum_sq[vi] / view_count[vi] as f64).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    (rms, view_rms, view_count)
}

fn main() {
    let args = Args::parse();

    // Read fiducial model
    let model = read_model(&args.model).unwrap_or_else(|e| {
        eprintln!("Error reading model: {}", e);
        std::process::exit(1);
    });

    let tilt_angles = imod_transforms::read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let n_views = tilt_angles.len();
    eprintln!(
        "tiltalign: {} views, {} objects in model",
        n_views,
        model.objects.len()
    );

    // Extract fiducial tracks: each object is a bead tracked across views.
    // Convention: each contour point's z = view index.
    let mut tracks: Vec<Track> = Vec::new();

    for obj in &model.objects {
        let mut track: Track = vec![None; n_views];
        for cont in &obj.contours {
            for pt in &cont.points {
                let view = pt.z.round() as usize;
                if view < n_views {
                    track[view] = Some((pt.x as f64, pt.y as f64));
                }
            }
        }
        let visible: usize = track.iter().filter(|t| t.is_some()).count();
        if visible >= args.min_views {
            tracks.push(track);
        }
    }

    let n_beads = tracks.len();
    eprintln!("tiltalign: {} usable fiducial tracks", n_beads);

    if n_beads == 0 {
        eprintln!("Error: no usable fiducial tracks found");
        std::process::exit(1);
    }

    // Determine reference view (default: the view closest to 0 tilt)
    let ref_view = args.ref_view.unwrap_or_else(|| {
        tilt_angles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(n_views / 2)
    });
    eprintln!("tiltalign: reference view = {} (tilt = {:.1} deg)", ref_view, tilt_angles[ref_view]);

    // Initialize per-view parameters
    let mut params: Vec<ViewParams> = tilt_angles
        .iter()
        .map(|&t| ViewParams::new(t as f64))
        .collect();

    // Center the observations around the mean position in the reference view
    // to improve convergence
    let mut center_x = 0.0f64;
    let mut center_y = 0.0f64;
    let mut center_count = 0;
    for track in &tracks {
        if let Some((x, y)) = track[ref_view] {
            center_x += x;
            center_y += y;
            center_count += 1;
        }
    }
    if center_count > 0 {
        center_x /= center_count as f64;
        center_y /= center_count as f64;
    }

    // Use image center if provided, otherwise use bead centroid
    let cx = if args.image_nx > 0 {
        args.image_nx as f64 / 2.0
    } else {
        center_x
    };
    let cy = if args.image_ny > 0 {
        args.image_ny as f64 / 2.0
    } else {
        center_y
    };

    // Center the observations
    let tracks: Vec<Track> = tracks
        .into_iter()
        .map(|track| {
            track
                .into_iter()
                .map(|pos| pos.map(|(x, y)| (x - cx, y - cy)))
                .collect()
        })
        .collect();

    eprintln!(
        "tiltalign: solving for: translations{}{}{}",
        if args.solve_rotation { " + rotation" } else { "" },
        if args.solve_mag { " + magnification" } else { "" },
        if args.solve_tilt { " + tilt" } else { "" },
    );

    // Iterative refinement
    let mut prev_rms = f64::MAX;
    for iter in 0..args.iterations {
        // Step 1: Estimate 3D bead positions from current alignment parameters
        let beads = estimate_beads(&tracks, &params, n_views);

        // Step 2: Update alignment parameters to minimize residuals
        update_view_params(
            &tracks,
            &beads,
            &mut params,
            n_views,
            args.solve_rotation,
            args.solve_mag,
            args.solve_tilt,
            ref_view,
        );

        // Compute residuals with updated parameters and updated beads
        let beads = estimate_beads(&tracks, &params, n_views);
        let (rms, _view_rms, _view_count) = compute_residuals(&tracks, &beads, &params, n_views);

        eprintln!(
            "  iter {:3}: RMS residual = {:.4} pixels",
            iter + 1,
            rms
        );

        // Check convergence
        let change = (prev_rms - rms).abs();
        if iter > 2 && change < 1e-6 {
            eprintln!("  converged (delta RMS = {:.2e})", change);
            break;
        }
        prev_rms = rms;
    }

    // Final residual report
    let beads = estimate_beads(&tracks, &params, n_views);
    let (rms, view_rms, view_count) = compute_residuals(&tracks, &beads, &params, n_views);

    eprintln!("\nFinal RMS residual: {:.4} pixels", rms);
    eprintln!("\nPer-view residuals:");
    eprintln!("  {:>5}  {:>8}  {:>8}  {:>5}  {:>8}  {:>8}  {:>8}",
             "View", "Tilt", "RMS", "Npts", "Rot(deg)", "Mag", "dx,dy");
    for vi in 0..n_views {
        if view_count[vi] > 0 {
            eprintln!(
                "  {:5}  {:8.2}  {:8.4}  {:5}  {:8.3}  {:8.5}  {:8.2},{:8.2}",
                vi,
                params[vi].tilt.to_degrees(),
                view_rms[vi],
                view_count[vi],
                params[vi].rotation.to_degrees(),
                params[vi].mag,
                params[vi].dx,
                params[vi].dy,
            );
        }
    }

    // Build output transforms
    // The transform maps raw image coordinates (centered) to aligned coordinates:
    //   xf applies as: x' = a11*(x-cx) + a12*(y-cy) + dx + cx
    // where the rotation+mag part is the in-plane alignment and dx,dy are shifts.
    let transforms: Vec<LinearTransform> = params
        .iter()
        .map(|vp| {
            let c = vp.rotation.cos() * vp.mag;
            let s = vp.rotation.sin() * vp.mag;
            LinearTransform {
                a11: c as f32,
                a12: -s as f32,
                a21: s as f32,
                a22: c as f32,
                dx: vp.dx as f32,
                dy: vp.dy as f32,
            }
        })
        .collect();

    write_xf_file(&args.output_xf, &transforms).unwrap();
    eprintln!(
        "\ntiltalign: wrote {} transforms to {}",
        n_views, args.output_xf
    );

    // Write refined tilt angles if requested
    if let Some(ref tilt_path) = args.output_tilt {
        let refined_tilts: Vec<f32> = params.iter().map(|vp| vp.tilt.to_degrees() as f32).collect();
        write_tilt_file(tilt_path, &refined_tilts).unwrap();
        eprintln!("tiltalign: wrote {} tilt angles to {}", n_views, tilt_path);
    }

    // Write 3D fiducial coordinates if requested
    if let Some(ref xyz_path) = args.output_xyz {
        let mut file = std::fs::File::create(xyz_path).unwrap();
        use std::io::Write;
        for (bi, bead) in beads.iter().enumerate() {
            // Output in uncentered coordinates
            writeln!(
                file,
                "{:4}  {:12.4}  {:12.4}  {:12.4}",
                bi + 1,
                bead.x + cx,
                bead.y + cy,
                bead.z
            )
            .unwrap();
        }
        eprintln!(
            "tiltalign: wrote {} bead positions to {}",
            beads.len(),
            xyz_path
        );
    }
}
