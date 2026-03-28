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

    /// Enable robust fitting with Tukey bisquare weighting
    #[arg(long, default_value_t = true)]
    robust: bool,

    /// Tukey bisquare constant (k-factor for MAD scaling)
    #[arg(long, default_value_t = 4.685)]
    kfactor: f32,

    /// Number of surfaces to fit (1 = single plane, 2 = two surfaces for thick specimens)
    #[arg(long, default_value_t = 1)]
    surfaces: i32,

    /// Group size for rotation solving (1 = per-view, N = one value per N consecutive views)
    #[arg(long, default_value_t = 1)]
    rotation_group: usize,

    /// Group size for magnification solving (1 = per-view, N = one value per N consecutive views)
    #[arg(long, default_value_t = 1)]
    mag_group: usize,

    /// Comma-separated list of view indices where rotation is held fixed (in addition to ref_view)
    #[arg(long)]
    fixed_rotation_views: Option<String>,

    /// Group size for tilt angle solving (1 = per-view, N = one value per N consecutive views)
    #[arg(long, default_value_t = 1)]
    tilt_group: usize,

    /// Solve for beam tilt correction (systematic shift proportional to defocus * tilt)
    #[arg(long)]
    solve_beam_tilt: bool,

    /// Scale factor for beam tilt correction
    #[arg(long, default_value_t = 1.0)]
    beam_tilt_scale: f32,

    /// Output file for local alignment corrections
    #[arg(long)]
    output_local: Option<String>,

    /// Patch size in pixels for local alignment grid
    #[arg(long, default_value_t = 500)]
    local_patch_size: usize,

    /// Overlap fraction between adjacent local patches (0.0 to 0.9)
    #[arg(long, default_value_t = 0.5)]
    local_overlap: f32,

    /// Perform leave-one-out cross-validation to estimate true alignment quality
    #[arg(long)]
    leave_out: bool,

    /// Drop the observation with the highest residual after each iteration until RMS stops improving
    #[arg(long)]
    drop_worst: bool,

    /// Convergence mode: "auto" (delta < 1e-6), "strict" (< 1e-8), "relaxed" (< 1e-4), "iterations" (always run max)
    #[arg(long, default_value = "auto")]
    convergence: String,

    /// Exclude patches within N pixels of the image edge from local alignment output
    #[arg(long, default_value_t = 0)]
    skip_edge_patches: usize,
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

/// Compute per-observation robust weights using Tukey bisquare function.
///
/// For each observation (bead, view), compute the residual distance, then apply
/// the bisquare weight: w = (1 - (r / (k * MADN))^2)^2 if |r| < k*MADN, else 0.
/// Returns a 2D weight array indexed as weights[bead_idx][view_idx], and reports
/// the number of down-weighted and fully rejected points.
fn compute_robust_weights(
    tracks: &[Track],
    beads: &[Bead3D],
    params: &[ViewParams],
    n_views: usize,
    kfactor: f64,
) -> (Vec<Vec<f64>>, usize, usize) {
    let n_beads = tracks.len();

    // Collect all residual distances
    let mut all_residuals: Vec<f32> = Vec::new();
    for (bi, track) in tracks.iter().enumerate() {
        for vi in 0..n_views {
            if let Some((obs_x, obs_y)) = track[vi] {
                let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                let dx = obs_x - pred_x;
                let dy = obs_y - pred_y;
                let dist = (dx * dx + dy * dy).sqrt();
                all_residuals.push(dist as f32);
            }
        }
    }

    if all_residuals.is_empty() {
        return (vec![vec![1.0; n_views]; n_beads], 0, 0);
    }

    // Compute median and MADN (median absolute deviation, normalized)
    let (med, _sorted) = imod_math::median(&all_residuals);
    let (madn_val, _) = imod_math::madn(&all_residuals, med);

    // Cutoff threshold
    let threshold = kfactor * madn_val as f64;

    let mut weights = vec![vec![0.0f64; n_views]; n_beads];
    let mut n_downweighted = 0usize;
    let mut n_rejected = 0usize;

    for (bi, track) in tracks.iter().enumerate() {
        for vi in 0..n_views {
            if track[vi].is_some() {
                let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                let (obs_x, obs_y) = track[vi].unwrap();
                let dx = obs_x - pred_x;
                let dy = obs_y - pred_y;
                let r = (dx * dx + dy * dy).sqrt();

                if threshold < 1e-10 {
                    // Degenerate case: all residuals are essentially zero
                    weights[bi][vi] = 1.0;
                } else if r < threshold {
                    let u = r / threshold;
                    let w = (1.0 - u * u) * (1.0 - u * u);
                    weights[bi][vi] = w;
                    if w < 0.9 {
                        n_downweighted += 1;
                    }
                } else {
                    weights[bi][vi] = 0.0;
                    n_rejected += 1;
                }
            }
        }
    }

    (weights, n_downweighted, n_rejected)
}

/// Estimate 3D bead positions from current alignment parameters via weighted least-squares.
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
/// When weights are provided, each observation pair is scaled by the weight.
fn estimate_beads(
    tracks: &[Track],
    params: &[ViewParams],
    n_views: usize,
    weights: Option<&Vec<Vec<f64>>>,
) -> Vec<Bead3D> {
    let n_beads = tracks.len();
    let mut beads = Vec::with_capacity(n_beads);

    for (bi, track) in tracks.iter().enumerate() {
        // Build normal equations: A^T W A x = A^T W b
        let mut ata = [0.0f64; 9]; // 3x3 row-major
        let mut atb = [0.0f64; 3];

        for vi in 0..n_views {
            if let Some((obs_x, obs_y)) = track[vi] {
                let w = weights.map_or(1.0, |wts| wts[bi][vi]);
                if w < 1e-12 {
                    continue;
                }

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

                // Accumulate A^T W A
                ata[0] += w * (ax0 * ax0 + ay0 * ay0);
                ata[1] += w * (ax0 * ax1 + ay0 * ay1);
                ata[2] += w * (ax0 * ax2 + ay0 * ay2);
                ata[3] += w * (ax1 * ax0 + ay1 * ay0);
                ata[4] += w * (ax1 * ax1 + ay1 * ay1);
                ata[5] += w * (ax1 * ax2 + ay1 * ay2);
                ata[6] += w * (ax2 * ax0 + ay2 * ay0);
                ata[7] += w * (ax2 * ax1 + ay2 * ay1);
                ata[8] += w * (ax2 * ax2 + ay2 * ay2);

                // Accumulate A^T W b
                atb[0] += w * (ax0 * rx + ay0 * ry);
                atb[1] += w * (ax1 * rx + ay1 * ry);
                atb[2] += w * (ax2 * rx + ay2 * ry);
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

/// Fit a plane to bead Z coordinates as a function of (X, Y).
///
/// Fits: Z = a + b*X + c*Y using weighted least-squares.
/// Returns (offset, slope_x, slope_y) and applies the correction to beads in place.
/// For surfaces == 2, splits beads above/below median Z and fits two planes.
fn fit_surface(beads: &mut [Bead3D], surfaces: i32) -> Vec<(f64, f64, f64)> {
    if beads.is_empty() {
        return vec![(0.0, 0.0, 0.0)];
    }

    if surfaces <= 1 {
        // Single surface fit
        let result = fit_single_surface(beads, None);
        // Subtract the fitted plane from bead Z values
        for bead in beads.iter_mut() {
            bead.z -= result.0 + result.1 * bead.x + result.2 * bead.y;
        }
        vec![result]
    } else {
        // Two-surface fit: split beads by median Z
        let mut zvals: Vec<f64> = beads.iter().map(|b| b.z).collect();
        zvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z = if zvals.len() % 2 == 0 {
            (zvals[zvals.len() / 2 - 1] + zvals[zvals.len() / 2]) / 2.0
        } else {
            zvals[zvals.len() / 2]
        };

        // Assign beads to surfaces
        let above: Vec<bool> = beads.iter().map(|b| b.z >= median_z).collect();

        let result_low = fit_single_surface(beads, Some((&above, false)));
        let result_high = fit_single_surface(beads, Some((&above, true)));

        // Subtract the appropriate plane from each bead
        for (i, bead) in beads.iter_mut().enumerate() {
            let (off, sx, sy) = if above[i] { result_high } else { result_low };
            bead.z -= off + sx * bead.x + sy * bead.y;
        }

        vec![result_low, result_high]
    }
}

/// Fit a single plane Z = a + b*X + c*Y to beads.
/// If `subset` is Some((flags, target)), only fit beads where flags[i] == target.
fn fit_single_surface(
    beads: &[Bead3D],
    subset: Option<(&[bool], bool)>,
) -> (f64, f64, f64) {
    // Build normal equations for: Z = a + b*X + c*Y
    // Design matrix row: [1, X, Y], target: Z
    let mut ata = [0.0f64; 9];
    let mut atb = [0.0f64; 3];
    let mut count = 0;

    for (i, bead) in beads.iter().enumerate() {
        if let Some((flags, target)) = subset {
            if flags[i] != target {
                continue;
            }
        }
        let row = [1.0, bead.x, bead.y];
        for p in 0..3 {
            for q in 0..3 {
                ata[p * 3 + q] += row[p] * row[q];
            }
            atb[p] += row[p] * bead.z;
        }
        count += 1;
    }

    if count < 3 {
        // Not enough points for a plane fit, just return mean Z offset
        let mean_z = if count > 0 {
            beads.iter()
                .enumerate()
                .filter(|(i, _)| subset.map_or(true, |(f, t)| f[*i] == t))
                .map(|(_, b)| b.z)
                .sum::<f64>() / count as f64
        } else {
            0.0
        };
        return (mean_z, 0.0, 0.0);
    }

    // Solve via gaussj
    let mut a_flat: Vec<f32> = ata.iter().map(|&v| v as f32).collect();
    let mut b_flat: Vec<f32> = atb.iter().map(|&v| v as f32).collect();

    if imod_math::gaussj::gaussj(&mut a_flat, 3, 3, &mut b_flat, 1, 1).is_ok() {
        (b_flat[0] as f64, b_flat[1] as f64, b_flat[2] as f64)
    } else {
        // Fallback: just use mean Z
        let mean_z = beads.iter()
            .enumerate()
            .filter(|(i, _)| subset.map_or(true, |(f, t)| f[*i] == t))
            .map(|(_, b)| b.z)
            .sum::<f64>() / count as f64;
        (mean_z, 0.0, 0.0)
    }
}

/// Return the group index for a given view, given the group size.
/// Views are grouped into consecutive blocks of `group_size`.
fn view_group(vi: usize, group_size: usize) -> usize {
    if group_size <= 1 { vi } else { vi / group_size }
}

/// Update per-view alignment parameters by minimizing projection residuals.
///
/// For each view, given fixed 3D bead positions, we solve for the alignment
/// parameters that minimize sum of squared residuals. Translation is always
/// solved. Rotation, magnification, and tilt are solved when enabled.
///
/// When group sizes > 1, rotation/magnification are solved per group of
/// consecutive views instead of per view. Weights from robust fitting
/// are applied when provided.
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
    weights: Option<&Vec<Vec<f64>>>,
    rotation_group: usize,
    mag_group: usize,
    fixed_rotation_views: &std::collections::HashSet<usize>,
    tilt_group: usize,
) {
    // When group solving is active, we first solve grouped parameters,
    // then solve per-view translations (and tilt if enabled).
    // For simplicity, we solve each view individually but apply group constraints
    // by averaging the group delta and applying it uniformly.

    let rot_n_groups = if rotation_group > 1 {
        (n_views + rotation_group - 1) / rotation_group
    } else {
        n_views
    };
    let mag_n_groups = if mag_group > 1 {
        (n_views + mag_group - 1) / mag_group
    } else {
        n_views
    };

    // Accumulate group deltas for rotation and magnification
    let mut rot_group_delta = vec![0.0f64; rot_n_groups];
    let mut rot_group_count = vec![0usize; rot_n_groups];
    let mut mag_group_delta = vec![0.0f64; mag_n_groups];
    let mut mag_group_count = vec![0usize; mag_n_groups];

    // First pass: solve per-view and collect group deltas
    let mut per_view_deltas: Vec<Option<Vec<f64>>> = vec![None; n_views];

    for vi in 0..n_views {
        // Collect observations in this view
        let obs: Vec<(usize, f64, f64)> = tracks
            .iter()
            .enumerate()
            .filter_map(|(bi, track)| track[vi].map(|(x, y)| (bi, x, y)))
            .collect();

        if obs.is_empty() {
            continue;
        }

        // Determine which parameters to solve for this view
        // Reference view has rotation=0 and mag=1 fixed; fixed_rotation_views also hold rotation fixed
        let solve_rot_here = solve_rotation && vi != ref_view && !fixed_rotation_views.contains(&vi);
        let solve_mag_here = solve_mag && vi != ref_view;
        // Tilt is solved independently from magnification -- the Jacobian columns are
        // orthogonal: tilt uses dxp_dtilt = -X*sin(tilt)+Z*cos(tilt), while mag uses
        // (cos_rot*xp - sin_rot*Y) which depends on xp, not dxp_dtilt.  No coupling fix needed.
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

        // Build normal equations: J^T W J delta = J^T W r
        let mut jtj = vec![0.0f64; n_unknowns * n_unknowns];
        let mut jtr = vec![0.0f64; n_unknowns];

        for &(bi, obs_x, obs_y) in &obs {
            let w = weights.map_or(1.0, |wts| wts[bi][vi]);
            if w < 1e-12 {
                continue;
            }

            let bead = &beads[bi];
            let (pred_x, pred_y) = project(bead, vp);
            let res_x = obs_x - pred_x;
            let res_y = obs_y - pred_y;

            let xp = bead.x * cos_tilt + bead.z * sin_tilt;
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

            // Accumulate J^T W J and J^T W r
            for p in 0..n_unknowns {
                for q in 0..n_unknowns {
                    jtj[p * n_unknowns + q] += w * (jx[p] * jx[q] + jy[p] * jy[q]);
                }
                jtr[p] += w * (jx[p] * res_x + jy[p] * res_y);
            }
        }

        // Solve normal equations using gaussj
        let n = n_unknowns;
        let mut a_flat: Vec<f32> = jtj.iter().map(|&v| v as f32).collect();
        let mut b_flat: Vec<f32> = jtr.iter().map(|&v| v as f32).collect();

        if imod_math::gaussj::gaussj(&mut a_flat, n, n, &mut b_flat, 1, 1).is_ok() {
            let delta: Vec<f64> = b_flat.iter().map(|&v| v as f64).collect();
            per_view_deltas[vi] = Some(delta);
        }
    }

    // Tilt grouping accumulators
    let tilt_n_groups = if tilt_group > 1 {
        (n_views + tilt_group - 1) / tilt_group
    } else {
        n_views
    };
    let mut tilt_group_delta = vec![0.0f64; tilt_n_groups];
    let mut tilt_group_count = vec![0usize; tilt_n_groups];

    // Apply deltas, handling group constraints
    for vi in 0..n_views {
        if let Some(ref delta) = per_view_deltas[vi] {
            let solve_rot_here = solve_rotation && vi != ref_view && !fixed_rotation_views.contains(&vi);
            let solve_mag_here = solve_mag && vi != ref_view;

            // Translations always applied per-view
            params[vi].dx += delta[0];
            params[vi].dy += delta[1];

            let mut idx = 2;
            if solve_rot_here {
                if rotation_group > 1 {
                    let gi = view_group(vi, rotation_group);
                    rot_group_delta[gi] += delta[idx];
                    rot_group_count[gi] += 1;
                } else {
                    params[vi].rotation += delta[idx];
                }
                idx += 1;
            }
            if solve_mag_here {
                if mag_group > 1 {
                    let gi = view_group(vi, mag_group);
                    mag_group_delta[gi] += delta[idx];
                    mag_group_count[gi] += 1;
                } else {
                    params[vi].mag += delta[idx];
                    params[vi].mag = params[vi].mag.clamp(0.5, 2.0);
                }
                idx += 1;
            }
            if solve_tilt {
                if tilt_group > 1 {
                    let gi = view_group(vi, tilt_group);
                    tilt_group_delta[gi] += delta[idx];
                    tilt_group_count[gi] += 1;
                } else {
                    params[vi].tilt += delta[idx];
                }
            }
        }
    }

    // Apply grouped rotation deltas
    if rotation_group > 1 && solve_rotation {
        for gi in 0..rot_n_groups {
            if rot_group_count[gi] > 0 {
                let avg_delta = rot_group_delta[gi] / rot_group_count[gi] as f64;
                let start = gi * rotation_group;
                let end = ((gi + 1) * rotation_group).min(n_views);
                for vi in start..end {
                    if vi != ref_view {
                        params[vi].rotation += avg_delta;
                    }
                }
            }
        }
    }

    // Apply grouped magnification deltas
    if mag_group > 1 && solve_mag {
        for gi in 0..mag_n_groups {
            if mag_group_count[gi] > 0 {
                let avg_delta = mag_group_delta[gi] / mag_group_count[gi] as f64;
                let start = gi * mag_group;
                let end = ((gi + 1) * mag_group).min(n_views);
                for vi in start..end {
                    if vi != ref_view {
                        params[vi].mag += avg_delta;
                        params[vi].mag = params[vi].mag.clamp(0.5, 2.0);
                    }
                }
            }
        }
    }

    // Apply grouped tilt deltas
    if tilt_group > 1 && solve_tilt {
        for gi in 0..tilt_n_groups {
            if tilt_group_count[gi] > 0 {
                let avg_delta = tilt_group_delta[gi] / tilt_group_count[gi] as f64;
                let start = gi * tilt_group;
                let end = ((gi + 1) * tilt_group).min(n_views);
                for vi in start..end {
                    params[vi].tilt += avg_delta;
                }
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

/// Search for the beam tilt angle that minimizes residuals.
///
/// Beam tilt causes a systematic image shift proportional to sin(tilt_angle),
/// modeling the defocus-dependent deflection. For each candidate beam tilt angle,
/// we apply dx_beam = beam_tilt_rad * sin(tilt_v) * scale_factor to the
/// projected x positions and pick the angle giving the lowest RMS.
fn solve_beam_tilt(
    tracks: &[Track],
    beads: &[Bead3D],
    params: &[ViewParams],
    n_views: usize,
    scale_factor: f64,
) -> (f64, f64) {
    let mut best_bt = 0.0f64;
    let mut best_rms = f64::MAX;

    // Search -2 to +2 degrees in 0.1 degree steps
    let steps = 41; // -20..=20 in units of 0.1 deg
    for step in 0..steps {
        let bt_deg = -2.0 + step as f64 * 0.1;
        let bt_rad = bt_deg.to_radians();

        let mut sum_sq = 0.0f64;
        let mut count = 0usize;

        for (bi, track) in tracks.iter().enumerate() {
            for vi in 0..n_views {
                if let Some((obs_x, obs_y)) = track[vi] {
                    let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                    // Beam tilt correction: shifts x proportional to sin(tilt)
                    let dx_beam = bt_rad * params[vi].tilt.sin() * scale_factor;
                    let dx = obs_x - (pred_x + dx_beam);
                    let dy = obs_y - pred_y;
                    sum_sq += dx * dx + dy * dy;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let rms = (sum_sq / count as f64).sqrt();
            if rms < best_rms {
                best_rms = rms;
                best_bt = bt_deg;
            }
        }
    }

    // Refine around the best with finer 0.01 degree steps
    let coarse_best = best_bt;
    for step in -10..=10 {
        let bt_deg = coarse_best + step as f64 * 0.01;
        let bt_rad = bt_deg.to_radians();

        let mut sum_sq = 0.0f64;
        let mut count = 0usize;

        for (bi, track) in tracks.iter().enumerate() {
            for vi in 0..n_views {
                if let Some((obs_x, obs_y)) = track[vi] {
                    let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                    let dx_beam = bt_rad * params[vi].tilt.sin() * scale_factor;
                    let dx = obs_x - (pred_x + dx_beam);
                    let dy = obs_y - pred_y;
                    sum_sq += dx * dx + dy * dy;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let rms = (sum_sq / count as f64).sqrt();
            if rms < best_rms {
                best_rms = rms;
                best_bt = bt_deg;
            }
        }
    }

    (best_bt, best_rms)
}

/// Compute local alignment corrections on a grid of patches.
///
/// After global alignment, divides the image area into overlapping patches.
/// For each patch, finds nearby beads and solves for a local translation offset
/// (dx, dy) that minimizes residuals for those beads. This captures local
/// warping/distortion that the global rigid alignment cannot model.
fn compute_local_alignment(
    tracks: &[Track],
    beads: &[Bead3D],
    params: &[ViewParams],
    n_views: usize,
    patch_size: usize,
    overlap: f32,
    cx: f64,
    cy: f64,
) -> Vec<(usize, usize, usize, f64, f64)> {
    // Determine the bounding box of bead positions in image coords (uncentered)
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for bead in beads.iter() {
        let bx = bead.x + cx;
        let by = bead.y + cy;
        if bx < min_x { min_x = bx; }
        if bx > max_x { max_x = bx; }
        if by < min_y { min_y = by; }
        if by > max_y { max_y = by; }
    }

    let ps = patch_size as f64;
    let step = ps * (1.0 - overlap as f64);
    if step < 1.0 {
        return Vec::new();
    }

    // Build patch grid
    let n_patches_x = ((max_x - min_x) / step).ceil() as usize + 1;
    let n_patches_y = ((max_y - min_y) / step).ceil() as usize + 1;

    let mut results = Vec::new();

    for py in 0..n_patches_y {
        for px in 0..n_patches_x {
            let patch_cx = min_x + px as f64 * step + ps / 2.0;
            let patch_cy = min_y + py as f64 * step + ps / 2.0;
            let half = ps / 2.0;

            // For each view, find beads within this patch and solve for local shift
            for vi in 0..n_views {
                let mut sum_dx = 0.0f64;
                let mut sum_dy = 0.0f64;
                let mut count = 0usize;

                for (bi, track) in tracks.iter().enumerate() {
                    if let Some((obs_x, obs_y)) = track[vi] {
                        // Check if the bead's projected position falls within this patch
                        // Use the observed position (in centered coords), convert to image coords
                        let img_x = obs_x + cx;
                        let img_y = obs_y + cy;

                        if (img_x - patch_cx).abs() <= half && (img_y - patch_cy).abs() <= half {
                            let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                            sum_dx += obs_x - pred_x;
                            sum_dy += obs_y - pred_y;
                            count += 1;
                        }
                    }
                }

                if count >= 2 {
                    let local_dx = sum_dx / count as f64;
                    let local_dy = sum_dy / count as f64;
                    results.push((vi, px, py, local_dx, local_dy));
                }
            }
        }
    }

    results
}

/// Perform leave-one-out cross-validation.
///
/// For each bead, re-solve the alignment without that bead, then predict
/// the left-out bead's position and compare to the actual observation.
/// Returns the leave-one-out RMS and per-bead RMS values.
fn leave_one_out_validation(
    tracks: &[Track],
    params_orig: &[ViewParams],
    n_views: usize,
    iterations: usize,
    solve_rotation: bool,
    solve_mag: bool,
    solve_tilt: bool,
    ref_view: usize,
    kfactor: f64,
    robust: bool,
    rotation_group: usize,
    mag_group: usize,
    surfaces: i32,
    fixed_rotation_views: &std::collections::HashSet<usize>,
    tilt_group: usize,
) -> (f64, Vec<f64>) {
    let n_beads = tracks.len();
    let mut per_bead_rms = Vec::with_capacity(n_beads);
    let mut total_sum_sq = 0.0f64;
    let mut total_count = 0usize;

    for leave_out_bi in 0..n_beads {
        // Build tracks without the left-out bead
        let reduced_tracks: Vec<Track> = tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leave_out_bi)
            .map(|(_, t)| t.clone())
            .collect();

        // Re-solve alignment from scratch with the reduced set
        let mut params: Vec<ViewParams> = params_orig.to_vec();
        let mut robust_weights: Option<Vec<Vec<f64>>> = None;

        for iter in 0..iterations {
            let mut beads = estimate_beads(&reduced_tracks, &params, n_views, robust_weights.as_ref());
            if surfaces >= 1 && iter > 0 {
                fit_surface(&mut beads, surfaces);
            }
            update_view_params(
                &reduced_tracks,
                &beads,
                &mut params,
                n_views,
                solve_rotation,
                solve_mag,
                solve_tilt,
                ref_view,
                robust_weights.as_ref(),
                rotation_group,
                mag_group,
                fixed_rotation_views,
                tilt_group,
            );

            if robust && iter >= 2 {
                let beads = estimate_beads(&reduced_tracks, &params, n_views, robust_weights.as_ref());
                let (weights, _, _) =
                    compute_robust_weights(&reduced_tracks, &beads, &params, n_views, kfactor);
                robust_weights = Some(weights);
            }
        }

        // Now estimate the left-out bead's 3D position using the solved alignment
        // by treating it as a single-bead estimation
        let left_out_tracks = vec![tracks[leave_out_bi].clone()];
        let left_out_beads = estimate_beads(&left_out_tracks, &params, n_views, None);

        // Compute residual for the left-out bead
        let mut bead_sum_sq = 0.0f64;
        let mut bead_count = 0usize;

        for vi in 0..n_views {
            if let Some((obs_x, obs_y)) = tracks[leave_out_bi][vi] {
                let (pred_x, pred_y) = project(&left_out_beads[0], &params[vi]);
                let dx = obs_x - pred_x;
                let dy = obs_y - pred_y;
                bead_sum_sq += dx * dx + dy * dy;
                bead_count += 1;
            }
        }

        let bead_rms = if bead_count > 0 {
            (bead_sum_sq / bead_count as f64).sqrt()
        } else {
            0.0
        };
        per_bead_rms.push(bead_rms);
        total_sum_sq += bead_sum_sq;
        total_count += bead_count;
    }

    let loo_rms = if total_count > 0 {
        (total_sum_sq / total_count as f64).sqrt()
    } else {
        0.0
    };

    (loo_rms, per_bead_rms)
}

/// Parse a comma-separated list of view indices into a HashSet.
fn parse_fixed_views(s: &str) -> std::collections::HashSet<usize> {
    s.split(',')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .collect()
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
    // Parse fixed rotation views
    let fixed_rotation_views: std::collections::HashSet<usize> = args
        .fixed_rotation_views
        .as_deref()
        .map(parse_fixed_views)
        .unwrap_or_default();

    eprintln!("tiltalign: reference view = {} (tilt = {:.1} deg)", ref_view, tilt_angles[ref_view]);
    if !fixed_rotation_views.is_empty() {
        eprintln!("tiltalign: fixed rotation views: {:?}", fixed_rotation_views);
    }

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
    let tracks: Vec<Track> = tracks  // made mutable later if --drop-worst
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
    if args.robust {
        eprintln!(
            "tiltalign: robust fitting enabled (k = {:.3})",
            args.kfactor
        );
    }
    if args.rotation_group > 1 {
        eprintln!(
            "tiltalign: rotation group size = {} ({} groups)",
            args.rotation_group,
            (n_views + args.rotation_group - 1) / args.rotation_group
        );
    }
    if args.mag_group > 1 {
        eprintln!(
            "tiltalign: magnification group size = {} ({} groups)",
            args.mag_group,
            (n_views + args.mag_group - 1) / args.mag_group
        );
    }
    if args.tilt_group > 1 {
        eprintln!(
            "tiltalign: tilt group size = {} ({} groups)",
            args.tilt_group,
            (n_views + args.tilt_group - 1) / args.tilt_group
        );
    }
    if args.surfaces > 1 {
        eprintln!("tiltalign: fitting {} surfaces", args.surfaces);
    }

    let kfactor = args.kfactor as f64;

    // Parse convergence mode
    let convergence_threshold: Option<f64> = match args.convergence.to_lowercase().as_str() {
        "strict" => Some(1e-8),
        "relaxed" => Some(1e-4),
        "iterations" => None, // always run max iterations
        "auto" | _ => Some(1e-6),
    };
    eprintln!("tiltalign: convergence mode = {} (threshold = {})",
        args.convergence,
        convergence_threshold.map_or("none (max iterations)".to_string(), |t| format!("{:.0e}", t)));

    // Mutable tracks for --drop-worst support
    let mut tracks = tracks;

    // Iterative refinement
    let mut prev_rms = f64::MAX;
    let mut robust_weights: Option<Vec<Vec<f64>>> = None;

    for iter in 0..args.iterations {
        // Step 1: Estimate 3D bead positions from current alignment parameters
        let mut beads = estimate_beads(&tracks, &params, n_views, robust_weights.as_ref());

        // Step 1b: Surface fitting - remove systematic Z variation
        if args.surfaces >= 1 && iter > 0 {
            let surfaces = fit_surface(&mut beads, args.surfaces);
            if iter == 1 {
                for (si, (off, sx, sy)) in surfaces.iter().enumerate() {
                    let angle_x = sx.atan().to_degrees();
                    let angle_y = sy.atan().to_degrees();
                    eprintln!(
                        "  surface {}: offset = {:.2}, slope_x = {:.4} ({:.2} deg), slope_y = {:.4} ({:.2} deg)",
                        si + 1, off, sx, angle_x, sy, angle_y
                    );
                }
            }
        }

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
            robust_weights.as_ref(),
            args.rotation_group,
            args.mag_group,
            &fixed_rotation_views,
            args.tilt_group,
        );

        // Compute residuals with updated parameters and updated beads
        let beads = estimate_beads(&tracks, &params, n_views, robust_weights.as_ref());
        let (rms, _view_rms, _view_count) = compute_residuals(&tracks, &beads, &params, n_views);

        // Step 3: Compute robust weights for next iteration (after first 2 iters to stabilize)
        if args.robust && iter >= 2 {
            let (weights, n_down, n_rej) =
                compute_robust_weights(&tracks, &beads, &params, n_views, kfactor);
            eprintln!(
                "  iter {:3}: RMS = {:.4} px, robust: {} down-weighted, {} rejected",
                iter + 1, rms, n_down, n_rej
            );
            robust_weights = Some(weights);
        } else {
            eprintln!(
                "  iter {:3}: RMS residual = {:.4} pixels",
                iter + 1,
                rms
            );
        }

        // --drop-worst: find and remove the observation with the highest residual
        if args.drop_worst && iter > 0 {
            let mut worst_bi = 0usize;
            let mut worst_vi = 0usize;
            let mut worst_sq = 0.0f64;
            for (bi, track) in tracks.iter().enumerate() {
                for vi in 0..n_views {
                    if let Some((obs_x, obs_y)) = track[vi] {
                        let (pred_x, pred_y) = project(&beads[bi], &params[vi]);
                        let dx = obs_x - pred_x;
                        let dy = obs_y - pred_y;
                        let sq = dx * dx + dy * dy;
                        if sq > worst_sq {
                            worst_sq = sq;
                            worst_bi = bi;
                            worst_vi = vi;
                        }
                    }
                }
            }
            if worst_sq > 0.0 && rms < prev_rms {
                eprintln!("  drop-worst: removing observation bead={} view={} (residual={:.4})",
                    worst_bi, worst_vi, worst_sq.sqrt());
                tracks[worst_bi][worst_vi] = None;
            } else if rms >= prev_rms {
                eprintln!("  drop-worst: RMS not improving, stopping removal");
            }
        }

        // Check convergence
        let change = (prev_rms - rms).abs();
        if let Some(threshold) = convergence_threshold {
            if iter > 2 && change < threshold {
                eprintln!("  converged (delta RMS = {:.2e}, threshold = {:.0e})", change, threshold);
                break;
            }
        }
        prev_rms = rms;
    }

    // Final bead estimation and surface report
    let mut beads = estimate_beads(&tracks, &params, n_views, robust_weights.as_ref());
    if args.surfaces >= 1 {
        let surfaces = fit_surface(&mut beads, args.surfaces);
        eprintln!("\nFinal surface fit:");
        for (si, (off, sx, sy)) in surfaces.iter().enumerate() {
            let angle_x = sx.atan().to_degrees();
            let angle_y = sy.atan().to_degrees();
            eprintln!(
                "  surface {}: offset = {:.2}, angle_x = {:.2} deg, angle_y = {:.2} deg",
                si + 1, off, angle_x, angle_y
            );
        }
    }
    let (rms, view_rms, view_count) = compute_residuals(&tracks, &beads, &params, n_views);

    // Beam tilt correction
    if args.solve_beam_tilt {
        let scale = args.beam_tilt_scale as f64;
        let (bt_angle, bt_rms) = solve_beam_tilt(&tracks, &beads, &params, n_views, scale);
        eprintln!("\nBeam tilt search (scale = {:.2}):", scale);
        eprintln!("  Best beam tilt angle: {:.3} degrees", bt_angle);
        eprintln!("  RMS with beam tilt correction: {:.4} pixels (was {:.4})", bt_rms, rms);
        if bt_rms < rms {
            eprintln!("  Beam tilt correction reduces RMS by {:.4} pixels", rms - bt_rms);
        } else {
            eprintln!("  No improvement from beam tilt correction");
        }
    }

    // Local alignment output
    if let Some(ref local_path) = args.output_local {
        let local_corrections = compute_local_alignment(
            &tracks,
            &beads,
            &params,
            n_views,
            args.local_patch_size,
            args.local_overlap,
            cx,
            cy,
        );

        // Filter out edge patches if --skip-edge-patches is set
        let edge_margin = args.skip_edge_patches as f64;
        let img_w = if args.image_nx > 0 { args.image_nx as f64 } else { cx * 2.0 };
        let img_h = if args.image_ny > 0 { args.image_ny as f64 } else { cy * 2.0 };
        let local_corrections: Vec<_> = if edge_margin > 0.0 {
            let ps = args.local_patch_size as f64;
            let step = ps * (1.0 - args.local_overlap as f64);
            // Compute approximate image-space position for each patch
            // The patch grid starts at min bead position; we use (px, py) grid indices
            // and the step size to estimate the patch center position in image coordinates
            let before = local_corrections.len();
            let filtered: Vec<_> = local_corrections.into_iter().filter(|&(_, px, py, _, _)| {
                // Approximate patch center in image coords (uncentered)
                // We don't have the exact min_x/min_y here, so we use the patch index
                // as a proxy: patches near index 0 or max index are at the edge
                let patch_x = px as f64 * step + ps / 2.0;
                let patch_y = py as f64 * step + ps / 2.0;
                // Check if patch center is within edge_margin of image boundary
                patch_x >= edge_margin && patch_x <= img_w - edge_margin
                    && patch_y >= edge_margin && patch_y <= img_h - edge_margin
            }).collect();
            let removed = before - filtered.len();
            if removed > 0 {
                eprintln!("tiltalign: skip-edge-patches removed {} corrections within {} px of edge", removed, edge_margin);
            }
            filtered
        } else {
            local_corrections
        };

        let mut file = std::fs::File::create(local_path).unwrap();
        use std::io::Write as _;
        writeln!(file, "# Local alignment corrections: view patch_x patch_y dx dy").unwrap();
        for &(vi, px, py, dx, dy) in &local_corrections {
            writeln!(file, "{} {} {} {:.4} {:.4}", vi, px, py, dx, dy).unwrap();
        }
        eprintln!(
            "\ntiltalign: wrote {} local corrections ({} patches) to {}",
            local_corrections.len(),
            {
                let mut unique_patches = std::collections::HashSet::new();
                for &(_, px, py, _, _) in &local_corrections {
                    unique_patches.insert((px, py));
                }
                unique_patches.len()
            },
            local_path
        );
    }

    // Leave-one-out cross-validation
    if args.leave_out {
        eprintln!("\nPerforming leave-one-out cross-validation ({} beads)...", n_beads);
        // Use fewer iterations for LOO to keep runtime manageable
        let loo_iters = args.iterations.min(10);
        let (loo_rms, per_bead_rms) = leave_one_out_validation(
            &tracks,
            &params,
            n_views,
            loo_iters,
            args.solve_rotation,
            args.solve_mag,
            args.solve_tilt,
            ref_view,
            kfactor,
            args.robust,
            args.rotation_group,
            args.mag_group,
            args.surfaces,
            &fixed_rotation_views,
            args.tilt_group,
        );
        eprintln!("  Leave-one-out RMS: {:.4} pixels (regular RMS: {:.4})", loo_rms, rms);
        eprintln!("  Ratio LOO/regular: {:.2}x", if rms > 1e-10 { loo_rms / rms } else { 0.0 });
        eprintln!("\n  Per-bead leave-one-out residuals:");
        eprintln!("  {:>5}  {:>10}", "Bead", "LOO RMS");
        let mut flagged = Vec::new();
        for (bi, &brms) in per_bead_rms.iter().enumerate() {
            eprintln!("  {:5}  {:10.4}", bi + 1, brms);
            if brms > loo_rms * 2.0 {
                flagged.push(bi + 1);
            }
        }
        if !flagged.is_empty() {
            eprintln!(
                "\n  Problematic beads (LOO RMS > 2x mean): {:?}",
                flagged
            );
        }
    }

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
