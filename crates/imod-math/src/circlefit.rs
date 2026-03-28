//! Circle, sphere, and ellipse fitting routines.
//!
//! Translated from IMOD's `circlefit.c`.
//! Copyright (C) 2007 by Boulder Laboratory for 3-Dimensional Electron
//! Microscopy of Cells ("BL3DEMC") and the Regents of the University of Colorado.

use crate::amoeba::{amoeba, amoeba_init, dual_amoeba};

/// Degrees-to-radians conversion factor.
const RADIANS_PER_DEGREE: f32 = std::f32::consts::PI / 180.0;

/// 3x3 determinant (same as the private helper in stats.rs).
fn determ3(
    a11: f64, a12: f64, a13: f64,
    a21: f64, a22: f64, a23: f64,
    a31: f64, a32: f64, a33: f64,
) -> f64 {
    a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
}

/// Result of [`circle_through_3pts`].
#[derive(Debug, Clone, Copy)]
pub struct CircleResult {
    /// Radius of the circle.
    pub radius: f32,
    /// X coordinate of the center.
    pub xc: f32,
    /// Y coordinate of the center.
    pub yc: f32,
}

/// Computes the radius and center for a circle through three given points.
///
/// Returns the circle parameters, or an error if the points are nearly
/// collinear or the radius squared is negative.
///
/// # Errors
///
/// Returns an error string if the determinant is too small relative to the
/// coefficients (near-collinear points) or if the computed radius squared is
/// negative.
pub fn circle_through_3pts(
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    x3: f32, y3: f32,
) -> Result<CircleResult, &'static str> {
    let (x1, y1) = (x1 as f64, y1 as f64);
    let (x2, y2) = (x2 as f64, y2 as f64);
    let (x3, y3) = (x3 as f64, y3 as f64);

    let sq1 = x1 * x1 + y1 * y1;
    let sq2 = x2 * x2 + y2 * y2;
    let sq3 = x3 * x3 + y3 * y3;

    let a = determ3(x1, y1, 1.0, x2, y2, 1.0, x3, y3, 1.0);
    let d = -determ3(sq1, y1, 1.0, sq2, y2, 1.0, sq3, y3, 1.0);
    let e = determ3(sq1, x1, 1.0, sq2, x2, 1.0, sq3, x3, 1.0);
    let f = -determ3(sq1, x1, y1, sq2, x2, y2, sq3, x3, y3);

    let absa = a.abs();
    if absa < 1.0e-15 * d.abs() || absa < 1.0e-15 * e.abs() || absa < 1.0e-15 * f.abs() {
        return Err("points are nearly collinear");
    }

    let xc = -0.5 * d / a;
    let yc = -0.5 * e / a;
    let rsq = (d * d + e * e) / (4.0 * a * a) - f / a;
    if rsq < 0.0 {
        return Err("negative radius squared");
    }

    Ok(CircleResult {
        radius: rsq.sqrt() as f32,
        xc: xc as f32,
        yc: yc as f32,
    })
}

/// Result of [`fit_sphere`] or [`fit_sphere_wgt`].
#[derive(Debug, Clone, Copy)]
pub struct SphereFitResult {
    /// Fitted radius.
    pub radius: f32,
    /// X coordinate of center.
    pub xcen: f32,
    /// Y coordinate of center.
    pub ycen: f32,
    /// Z coordinate of center (meaningful only for sphere fits).
    pub zcen: f32,
    /// RMS error of the fit.
    pub rms_err: f32,
}

/// Fit a circle (2D) or sphere (3D) to a set of points using a simplex search.
///
/// If `zpt` is `Some`, a sphere fit is performed; otherwise a circle fit.
/// When `fit_radius` is true, the radius is fitted along with the center;
/// when false, the radius is held fixed at the input value.
///
/// The initial values of `radius`, `xcen`, `ycen`, and `zcen` (for sphere)
/// must be reasonable starting guesses.
///
/// Returns the fitted parameters and RMS error.
pub fn fit_sphere(
    xpt: &[f32],
    ypt: &[f32],
    zpt: Option<&[f32]>,
    radius: f32,
    xcen: f32,
    ycen: f32,
    zcen: f32,
    fit_radius: bool,
) -> SphereFitResult {
    fit_sphere_wgt(xpt, ypt, zpt, None, radius, xcen, ycen, zcen, fit_radius)
}

/// Fit a circle or sphere to a set of points with optional per-point weighting.
///
/// See [`fit_sphere`] for parameter descriptions. `weights` provides optional
/// per-point weights; if `None`, uniform weighting is used.
pub fn fit_sphere_wgt(
    xpt: &[f32],
    ypt: &[f32],
    zpt: Option<&[f32]>,
    weights: Option<&[f32]>,
    radius: f32,
    xcen: f32,
    ycen: f32,
    zcen: f32,
    fit_radius: bool,
) -> SphereFitResult {
    let num_pts = xpt.len();
    let fit_rad_offset: usize = if fit_radius { 1 } else { 0 };
    let is_sphere = zpt.is_some();

    let mut nvar = 2 + fit_rad_offset;
    if is_sphere {
        nvar += 1;
    }

    // Build initial parameter vector
    let mut a = vec![0.0f32; nvar];
    if fit_radius {
        a[0] = radius;
    }
    a[fit_rad_offset] = xcen;
    a[fit_rad_offset + 1] = ycen;
    if is_sphere {
        a[fit_rad_offset + 2] = zcen;
    }

    let fixed_radius = radius;

    // Build the error function closure
    let err_func = |params: &[f32]| -> f32 {
        let rad = if fit_radius { params[0] } else { fixed_radius };
        let xc = params[fit_rad_offset];
        let yc = params[fit_rad_offset + 1];

        let mut err: f64 = 0.0;
        if is_sphere {
            let zc = params[fit_rad_offset + 2];
            let zp = zpt.unwrap();
            for i in 0..num_pts {
                let dx = (xpt[i] - xc) as f64;
                let dy = (ypt[i] - yc) as f64;
                let dz = (zp[i] - zc) as f64;
                let delrad = (dx * dx + dy * dy + dz * dz).sqrt() - rad as f64;
                let w = weights.map_or(1.0, |ww| ww[i] as f64);
                err += delrad * delrad * w;
            }
        } else {
            for i in 0..num_pts {
                let dx = (xpt[i] - xc) as f64;
                let dy = (ypt[i] - yc) as f64;
                let delrad = (dx * dx + dy * dy).sqrt() - rad as f64;
                let w = weights.map_or(1.0, |ww| ww[i] as f64);
                err += delrad * delrad * w;
            }
        }
        (err / num_pts as f64) as f32
    };

    let da = vec![2.0f32; nvar];
    let delfac: f32 = 2.0;
    let ftol2: f32 = 5.0e-4;
    let ftol1: f32 = 1.0e-5;
    let ptol2: f32 = 0.1;
    let ptol1: f32 = 0.002;

    let mp = nvar + 1;

    let errmin = err_func(&a);
    if errmin > 0.0 {
        // First pass: coarse
        let mut p = vec![0.0f32; mp * nvar];
        let mut y = vec![0.0f32; mp];
        let ptol = amoeba_init(&mut p, &mut y, mp, nvar, delfac, ptol2, &a, &da, &err_func);
        let res = amoeba(&mut p, &mut y, mp, nvar, ftol2, &ptol, &err_func);
        for i in 0..nvar {
            a[i] = p[res.best_index + i * mp];
        }

        // Second pass: fine
        let mut p = vec![0.0f32; mp * nvar];
        let mut y = vec![0.0f32; mp];
        let ptol = amoeba_init(&mut p, &mut y, mp, nvar, delfac, ptol1, &a, &da, &err_func);
        let res = amoeba(&mut p, &mut y, mp, nvar, ftol1, &ptol, &err_func);
        for i in 0..nvar {
            a[i] = p[res.best_index + i * mp];
        }
    }

    let final_err = err_func(&a);

    let out_rad = if fit_radius { a[0] } else { fixed_radius };
    let out_xcen = a[fit_rad_offset];
    let out_ycen = a[fit_rad_offset + 1];
    let out_zcen = if is_sphere {
        a[fit_rad_offset + 2]
    } else {
        zcen
    };

    SphereFitResult {
        radius: out_rad,
        xcen: out_xcen,
        ycen: out_ycen,
        zcen: out_zcen,
        rms_err: final_err.sqrt(),
    }
}

// ---------------------------------------------------------------------------
// minimize1D - 1D search helper (translated from IMOD minimize1D.c)
// ---------------------------------------------------------------------------

/// State for the 1D minimizer.
struct Min1DState {
    /// Number of step-size halvings performed so far.
    num_cuts_done: i32,
    /// Bracket storage: positions[0..7] and values[7..14].
    brackets: [f32; 14],
}

/// Guides a 1D minimum search by walking then bisecting.
///
/// Returns `(ret_code, next_position)`. `ret_code` 0 means continue;
/// nonzero means done or error. The minimum position is `brackets[1]`
/// and the minimum value is `brackets[8]`.
fn minimize_1d(
    cur_position: f32,
    cur_value: f32,
    initial_step: f32,
    num_scan_steps: i32,
    state: &mut Min1DState,
) -> (i32, f32) {
    let walking: bool = state.num_cuts_done == 0;
    let mut step = initial_step;
    let next_position: f32;

    // Initialize
    if state.num_cuts_done < 0 {
        state.num_cuts_done = 0;
        state.brackets[0] = cur_position;
        state.brackets[1] = cur_position;
        state.brackets[2] = cur_position;
        state.brackets[8] = cur_value; // values[1]
        if num_scan_steps > 0 {
            state.brackets[13] = 2.0; // values[6] = direction
            next_position = cur_position + initial_step;
            state.brackets[5] = cur_position;
            state.brackets[6] = cur_position;
            state.brackets[12] = cur_value; // values[5]
            state.brackets[7] = cur_value - cur_value.abs(); // values[0]
        } else {
            state.brackets[13] = -1.0; // values[6] = direction
            next_position = cur_position - initial_step;
        }
        return (0, next_position);
    }

    let direction_f = state.brackets[13]; // values[6]
    let mut direction = direction_f.round() as i32;
    if direction < -1 || direction > 2 {
        return (1, 0.0);
    }

    // Compute current step size
    for _ in 0..state.num_cuts_done {
        step /= 2.0;
    }

    if direction == 2 {
        // INITIAL SCAN
        let step_num = ((cur_position - state.brackets[6]) / initial_step).round() as i32;
        if step_num <= 0 || step_num > num_scan_steps {
            return (1, 0.0);
        }

        // Roll positions
        state.brackets[3] = state.brackets[4];
        state.brackets[4] = state.brackets[5];
        state.brackets[5] = cur_position;
        state.brackets[10] = state.brackets[11]; // values[3] = values[4]
        state.brackets[11] = state.brackets[12]; // values[4] = values[5]
        state.brackets[12] = cur_value;           // values[5]

        if step_num > 1 && state.brackets[11] < state.brackets[8] {
            // values[4] < values[1]
            for i in 0..3 {
                state.brackets[i] = state.brackets[i + 3];
                state.brackets[i + 7] = state.brackets[i + 10];
            }
        }

        if step_num < num_scan_steps {
            return (0, cur_position + initial_step);
        }

        // Terminate scan
        if state.brackets[12] < state.brackets[8]   // values[5] < values[1]
            || state.brackets[10] < state.brackets[8] // values[3] < values[1]
            || state.brackets[9] < state.brackets[8]  // values[2] < values[1]
            || state.brackets[7] < state.brackets[8]  // values[0] < values[1]
        {
            return (2, 0.0);
        }

        // Cut step, set direction
        step /= 2.0;
        state.num_cuts_done += 1;
        direction = if state.brackets[9] > state.brackets[7] { -1 } else { 1 };
        // values[2] > values[0]
    } else {
        // WALKING OR CUTTING
        if cur_value > state.brackets[8] {
            // values[1]
            // New value is higher; replace bracket on same-direction side
            let side = (direction + 1) as usize;
            state.brackets[side] = cur_position;
            state.brackets[side + 7] = cur_value;

            if (!walking
                && (state.brackets[(1 - direction + 1) as usize] - state.brackets[1]).abs()
                    < 1.1 * (cur_position - state.brackets[1]).abs())
                || (walking && (state.brackets[1] - state.brackets[2]).abs() > 0.01 * step)
            {
                step /= 2.0;
                state.num_cuts_done += 1;
                direction = if state.brackets[9] > state.brackets[7] { -1 } else { 1 };
            } else {
                direction *= -1;
            }
        } else {
            // New minimum
            let other_side = (1 - direction + 1) as usize;
            state.brackets[other_side] = state.brackets[1];
            state.brackets[other_side + 7] = state.brackets[8];
            state.brackets[1] = cur_position;
            state.brackets[8] = cur_value;

            if !walking {
                step /= 2.0;
                state.num_cuts_done += 1;
                direction = if state.brackets[9] > state.brackets[7] { -1 } else { 1 };
            }
        }
    }

    next_position = state.brackets[1] + direction as f32 * step;
    state.brackets[13] = direction as f32;
    (0, next_position)
}

// ---------------------------------------------------------------------------
// Ellipse fitting
// ---------------------------------------------------------------------------

/// Result of [`fit_centered_ellipse`].
#[derive(Debug, Clone, Copy)]
pub struct EllipseFitResult {
    /// Semi-axis along X before rotation.
    pub xrad: f32,
    /// Semi-axis along Y before rotation.
    pub yrad: f32,
    /// Rotation angle counterclockwise, in degrees.
    pub theta: f32,
    /// RMS error (minimum distance from each point to the ellipse).
    pub rms_err: f32,
    /// Per-point errors (distances to the ellipse).
    pub converged: bool,
}

/// Fits an ellipse centered at the origin to a set of 2D points.
///
/// Returns the semi-axes `xrad` and `yrad` (before rotation), the rotation
/// angle `theta` in degrees (counterclockwise), the RMS error, and per-point
/// error distances in `errors_out` (must have length >= `xpt.len()`).
///
/// The error for each point is the minimum distance from the point to the
/// fitted ellipse. Parameters are found by simplex minimization.
///
/// The `converged` field is `false` if the simplex search did not move from
/// its initial rough estimate.
pub fn fit_centered_ellipse(
    xpt: &[f32],
    ypt: &[f32],
    errors_out: &mut [f32],
) -> EllipseFitResult {
    let num_pts = xpt.len();
    let pi_val = 180.0 * RADIANS_PER_DEGREE; // == PI

    // Compute polar coordinates
    let mut angles = vec![0.0f32; num_pts];
    let mut radii = vec![0.0f32; num_pts];
    let mut raw_min: f32 = f32::MAX;
    let mut raw_max: f32 = f32::MIN;
    for i in 0..num_pts {
        angles[i] = ypt[i].atan2(xpt[i]);
        radii[i] = (xpt[i] * xpt[i] + ypt[i] * ypt[i]).sqrt();
        if radii[i] < raw_min { raw_min = radii[i]; }
        if radii[i] > raw_max { raw_max = radii[i]; }
    }

    // Initialize: scan 90-degree range to find angle with maximum ratio of
    // distances at that angle vs angle+90
    let del_angle = 5.0 * RADIANS_PER_DEGREE;
    let start_angle = -90.0 * RADIANS_PER_DEGREE;
    let num_close = if num_pts < 8 { 1 } else if num_pts < 12 { 2 } else { 3 };

    let mut max_ratio: f32 = -1.0;
    let mut aa_init = [0.0f32; 3];

    for jnd in 0..18 {
        let angle_base = start_angle + jnd as f32 * del_angle;
        let mut mean_dist = [0.0f32; 2];

        for dir in 0..2 {
            let angle = angle_base + dir as f32 * pi_val / 2.0;
            let mut close_ang = [1.0e10f32; 3];
            let mut close_dist = [0.0f32; 3];

            for ind in 0..num_pts {
                let mut diff = angle - angles[ind];
                while diff >= pi_val / 2.0 { diff -= pi_val; }
                while diff < -pi_val / 2.0 { diff += pi_val; }

                for spot in 0..3 {
                    if diff.abs() < close_ang[spot] && num_close > spot {
                        // Shift entries down
                        let mut cli = num_close - 1;
                        while cli > spot {
                            close_ang[cli] = close_ang[cli - 1];
                            close_dist[cli] = close_dist[cli - 1];
                            cli -= 1;
                        }
                        close_ang[spot] = diff.abs();
                        close_dist[spot] = radii[ind];
                        break;
                    }
                }
            }

            mean_dist[dir] = 0.0;
            for spot in 0..num_close {
                mean_dist[dir] += close_dist[spot] / num_close as f32;
            }
        }

        let ratio = (mean_dist[0] / mean_dist[1]).max(mean_dist[1] / mean_dist[0]);
        if ratio > max_ratio {
            aa_init[2] = angle_base;
            max_ratio = ratio;
            aa_init[0] = mean_dist[0];
            aa_init[1] = mean_dist[1];
        }
    }

    let mut aa = aa_init;

    // Constrain fit ratio
    let mut max_fit_ratio = 2.0 * (aa[0] / aa[1]).max(aa[1] / aa[0]);
    let raw_ratio = 1.1 * (raw_max / raw_min);
    if raw_ratio > max_fit_ratio {
        max_fit_ratio = raw_ratio;
    }

    // Step sizes
    let da = [aa[0] / 10.0, aa[1] / 10.0, 5.0 * RADIANS_PER_DEGREE];

    // Build error function
    // We need interior mutability for last_error and errors_out filling,
    // so we use a Cell for last_error and track final_fit externally.
    let last_error_cell = std::cell::Cell::new(0.0f32);
    let final_fit_cell = std::cell::Cell::new(false);
    let errors_ptr = errors_out.as_mut_ptr();

    let ellipse_func = |params: &[f32]| -> f32 {
        let last_err = last_error_cell.get();

        if params[0] <= 0.0 || params[1] <= 0.0
            || params[0] < 1.0e-5 * params[1]
            || params[1] < 1.0e-5 * params[0]
        {
            return 10.0 * last_err;
        }

        let ratio = (params[0] / params[1]).max(params[1] / params[0]);
        if ratio > max_fit_ratio {
            return last_err * (1.0 + 5.0 * (ratio / max_fit_ratio - 1.0));
        }

        let is_final = final_fit_cell.get();
        let mut derr: f64 = 0.0;

        for ind in 0..num_pts {
            let xrot = radii[ind] * (angles[ind] - params[2]).cos();
            let yrot = radii[ind] * (angles[ind] - params[2]).sin();
            let pt_angle = yrot.atan2(xrot);
            let mut cur_pos = pt_angle;
            let initial_step = 4.0 * RADIANS_PER_DEGREE;

            let mut min_state = Min1DState {
                num_cuts_done: -1,
                brackets: [0.0; 14],
            };

            let cur_val = {
                let dx = xrot - params[0] * cur_pos.cos();
                let dy = yrot - params[1] * cur_pos.sin();
                dx * dx + dy * dy
            };

            let (ret, next) = minimize_1d(cur_pos, cur_val, initial_step, 0, &mut min_state);
            if ret != 0 {
                return 1.0e30;
            }
            cur_pos = next;

            loop {
                let dx = xrot - params[0] * cur_pos.cos();
                let dy = yrot - params[1] * cur_pos.sin();
                let cur_val = dx * dx + dy * dy;

                let (ret, next) = minimize_1d(cur_pos, cur_val, initial_step, 0, &mut min_state);
                if ret != 0 || (cur_pos - pt_angle).abs() > 90.0 * RADIANS_PER_DEGREE {
                    return 1.0e30;
                }
                if min_state.num_cuts_done > 10 {
                    break;
                }
                cur_pos = next;
            }

            derr += min_state.brackets[8] as f64; // minimum value
            if is_final {
                // SAFETY: errors_ptr points to errors_out which has length >= num_pts
                unsafe { *errors_ptr.add(ind) = min_state.brackets[8].sqrt(); }
            }
        }

        let err = (derr / num_pts as f64).sqrt() as f32;
        last_error_cell.set(err);
        err
    };

    // Initial evaluation
    let _init_err = ellipse_func(&aa);

    // Dual amoeba run
    let ptol_facs = [5.0e-4f32, 1.0e-5];
    let ftol_facs = [5.0e-4f32, 1.0e-5];
    let delfac = 2.0f32;
    let _result = dual_amoeba(3, delfac, ptol_facs, ftol_facs, &mut aa, &da, &ellipse_func);

    // Final fit to fill errors
    final_fit_cell.set(true);
    let rms_err = ellipse_func(&aa);

    // Check convergence
    let mut num_same = 0;
    for i in 0..3 {
        if (aa[i] - aa_init[i]).abs() < 2.0e-6 * aa_init[i].abs() {
            num_same += 1;
        }
    }

    EllipseFitResult {
        xrad: aa[0],
        yrad: aa[1],
        theta: aa[2] / RADIANS_PER_DEGREE,
        rms_err,
        converged: num_same != 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_through_known_points() {
        // Circle of radius 5 centered at (1, 2)
        // Three points on that circle at 0, 120, 240 degrees
        let cx = 1.0f32;
        let cy = 2.0f32;
        let r = 5.0f32;
        let angles = [0.0f64, 2.0 * std::f64::consts::PI / 3.0, 4.0 * std::f64::consts::PI / 3.0];
        let xs: Vec<f32> = angles.iter().map(|a| cx + r * a.cos() as f32).collect();
        let ys: Vec<f32> = angles.iter().map(|a| cy + r * a.sin() as f32).collect();

        let result = circle_through_3pts(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]).unwrap();
        assert!((result.radius - r).abs() < 0.01, "radius = {}", result.radius);
        assert!((result.xc - cx).abs() < 0.01, "xc = {}", result.xc);
        assert!((result.yc - cy).abs() < 0.01, "yc = {}", result.yc);
    }

    #[test]
    fn collinear_points_fail() {
        let result = circle_through_3pts(0.0, 0.0, 1.0, 1.0, 2.0, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn fit_circle_to_points() {
        // Generate points on a circle of radius 10 at origin
        let n = 20;
        let r = 10.0f32;
        let mut xpt = vec![0.0f32; n];
        let mut ypt = vec![0.0f32; n];
        for i in 0..n {
            let a = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            xpt[i] = r * a.cos();
            ypt[i] = r * a.sin();
        }

        let result = fit_sphere(&xpt, &ypt, None, r * 0.8, 1.0, 1.0, 0.0, true);
        assert!((result.radius - r).abs() < 0.5, "radius = {}", result.radius);
        assert!(result.xcen.abs() < 0.5, "xcen = {}", result.xcen);
        assert!(result.ycen.abs() < 0.5, "ycen = {}", result.ycen);
        assert!(result.rms_err < 1.0, "rms_err = {}", result.rms_err);
    }
}
