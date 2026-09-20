//! Translation of `IMOD/libcfshr/circlefit.c`.
//!
//! Only `fitCenteredEllipse` and its error routine are translated: the
//! `circleThrough3Pts`/`fitSphere`/`fitSphereWgt`/`enableRadiusFitting` family
//! (with the Fortran `fitcircle`/`fitcirclewgt` wrappers and the per-thread
//! `xpThrd`.. pointer tables of `circlefit.c:70`) is reached only from
//! `imodutil/imodcurvature.c`, outside this crate's scope, and was removed as
//! dead (see `DEAD_CODE.md`).

use crate::imod::libcfshr::amoeba::dual_amoeba;
use core::sync::atomic::{AtomicU32, Ordering};

const RADIANS_PER_DEGREE: f32 = 0.017_453_292_52;
const PI_VAL: f32 = 180. * RADIANS_PER_DEGREE;

/// C static `lastError` inside `ellipseFunc`, held as `f32` bits.
static LAST_ERROR: AtomicU32 = AtomicU32::new(0);

/// C `fitCenteredEllipse`.
#[allow(clippy::too_many_arguments)]
pub fn fit_centered_ellipse(
    xpt: &[f32],
    ypt: &[f32],
    num_pts: i32,
    xrad: &mut f32,
    yrad: &mut f32,
    theta: &mut f32,
    rms_err: &mut f32,
    work: &mut [f32],
) -> i32 {
    let mut num_close: usize;
    let mut raw_min = 1.0e30_f32;
    let mut raw_max = -1.0e30_f32;
    let ptol_facs = [5.0e-4_f32, 1.0e-5];
    let ftol_facs = [5.0e-4_f32, 1.0e-5];
    let delfac = 2.0_f32;
    let mut da = [1.0_f32, 1., 5. * RADIANS_PER_DEGREE];
    let mut yy = [0.0_f32; 4];
    let mut aa = [0.0_f32; 3];
    let mut aa_init = [0.0_f32; 3];

    // Set pointers to arrays and fill arrays with polar coordinates of values
    // sErrors = work; sAngles = work + numPts; sRadii = sAngles + numPts
    let n = num_pts as usize;
    let (s_errors, rest) = work.split_at_mut(n);
    let (s_angles, s_radii) = rest.split_at_mut(n);
    for ind in 0..n {
        s_angles[ind] = (ypt[ind] as f64).atan2(xpt[ind] as f64) as f32;
        s_radii[ind] = ((xpt[ind] * xpt[ind] + ypt[ind] * ypt[ind]) as f64).sqrt() as f32;
        if s_radii[ind] < raw_min {
            raw_min = s_radii[ind];
        }
        if s_radii[ind] > raw_max {
            raw_max = s_radii[ind];
        }
    }
    let s_num_pts = num_pts;

    // To initialize, scan 90 deg range, look for 3 closest angles to that angle and
    // to that angle plus 90 (or 2 closest if 8 to 11 points, or just closest)
    let del_angle = 5. * RADIANS_PER_DEGREE;
    let start_angle = -90. * RADIANS_PER_DEGREE;
    num_close = 3;
    if num_pts < 12 {
        num_close = if num_pts < 8 { 1 } else { 2 };
    }
    let mut max_ratio = -1.0_f32;
    for jnd in 0..18 {
        let angle_base = start_angle + jnd as f32 * del_angle;
        let mut angle = angle_base;
        let mut mean_dist = [0.0_f32; 2];
        for dir in 0..2 {
            let mut close_ang = [0.0_f32; 3];
            let mut close_dist = [0.0_f32; 3];
            for close in close_ang.iter_mut().take(num_close) {
                *close = 1.0e10;
            }
            close_dist[1] = 0.;
            close_dist[2] = 0.;
            for ind in 0..n {
                let mut diff = angle - s_angles[ind];
                while diff >= PI_VAL / 2. {
                    diff -= PI_VAL;
                }
                while diff < -PI_VAL / 2. {
                    diff += PI_VAL;
                }

                // Maintain angle differences and radial distances for selected # of
                // closest ones
                for spot in 0..3 {
                    if diff.abs() < close_ang[spot] && num_close > spot {
                        let mut cli = num_close - 1;
                        while cli > spot {
                            close_ang[cli] = close_ang[cli - 1];
                            close_dist[cli] = close_dist[cli - 1];
                            cli -= 1;
                        }
                        close_ang[spot] = diff.abs();
                        close_dist[spot] = s_radii[ind];
                        break;
                    }
                }
            }

            // Get mean distance at the direction
            mean_dist[dir] = 0.;
            for spot in 0..num_close {
                mean_dist[dir] += close_dist[spot] / num_close as f32;
            }
            angle += PI_VAL / 2.;
        }

        // Then get ratio of distances and keep track of angle with maximum ratio
        let ratio = (mean_dist[0] / mean_dist[1]).max(mean_dist[1] / mean_dist[0]);
        if ratio > max_ratio {
            aa_init[2] = angle_base;
            max_ratio = ratio;
            aa_init[0] = mean_dist[0];
            aa_init[1] = mean_dist[1];
        }
    }
    for ind in 0..3 {
        aa[ind] = aa_init[ind];
    }

    // Constrain fit to somewhere in the range of the initial ratio, and to not much
    // above the actual min/max
    let mut s_max_fit_ratio = 2. * (aa[0] / aa[1]).max(aa[1] / aa[0]);
    /* ACCUM_MAX(sMaxFitRatio, 1.1 * (rawMax / rawMin)) — the double literal
    multiplies the completed single-precision quotient. */
    let accum = 1.1 * (raw_max / raw_min) as f64;
    if accum > s_max_fit_ratio as f64 {
        s_max_fit_ratio = accum as f32;
    }

    // Set step sizes
    da[0] = aa[0] / 10.;
    da[1] = aa[1] / 10.;
    let mut num_iter: i32 = 0;

    ellipse_func(
        &aa,
        rms_err,
        s_errors,
        s_angles,
        s_radii,
        s_num_pts,
        0,
        s_max_fit_ratio,
    );

    {
        let mut func = |v: &[f32]| -> f32 {
            let mut error = 0.0_f32;
            ellipse_func(
                v,
                &mut error,
                s_errors,
                s_angles,
                s_radii,
                s_num_pts,
                0,
                s_max_fit_ratio,
            );
            error
        };
        dual_amoeba(
            &mut yy,
            3,
            delfac,
            &ptol_facs,
            &ftol_facs,
            &mut aa,
            &da,
            &mut func,
            &mut num_iter,
        );
    }

    // Final fit with error array filled
    ellipse_func(
        &aa,
        rms_err,
        s_errors,
        s_angles,
        s_radii,
        s_num_pts,
        1,
        s_max_fit_ratio,
    );

    // return values
    *xrad = aa[0];
    *yrad = aa[1];
    *theta = aa[2] / RADIANS_PER_DEGREE;

    // Find out if amoeba went nowhere
    let mut num_close = 0;
    for ind in 0..3 {
        if ((aa[ind] - aa_init[ind]) as f64) < 2.0e-6 * aa_init[ind] as f64 {
            num_close += 1;
        }
    }

    if num_close == 3 { 1 } else { 0 }
}

/// C static `ellipseFunc`.
#[allow(clippy::too_many_arguments)]
fn ellipse_func(
    aa: &[f32],
    error: &mut f32,
    s_errors: &mut [f32],
    s_angles: &[f32],
    s_radii: &[f32],
    s_num_pts: i32,
    s_final_fit: i32,
    s_max_fit_ratio: f32,
) {
    let mut derr = 0.0_f64;
    let last_error = f32::from_bits(LAST_ERROR.load(Ordering::Relaxed));

    if aa[0] <= 0.
        || aa[1] <= 0.
        || (aa[0] as f64) < 1.0e-5 * aa[1] as f64
        || (aa[1] as f64) < 1.0e-5 * aa[0] as f64
    {
        *error = (10. * last_error as f64) as f32;
        return;
    }

    let ratio = (aa[0] / aa[1]).max(aa[1] / aa[0]);
    if ratio > s_max_fit_ratio {
        /* `ratio / sMaxFitRatio` is a single-precision quotient; only the
        `- 1.` widens it. */
        *error = (last_error as f64 * (1. + 5. * ((ratio / s_max_fit_ratio) as f64 - 1.))) as f32;
        return;
    }

    // For each point, estimate the ray angle of the nearest point then find the
    // angle that minimizes distance
    for ind in 0..s_num_pts as usize {
        let xrot = (s_radii[ind] as f64 * ((s_angles[ind] - aa[2]) as f64).cos()) as f32;
        let yrot = (s_radii[ind] as f64 * ((s_angles[ind] - aa[2]) as f64).sin()) as f32;
        let pt_angle = (yrot as f64).atan2(xrot as f64) as f32;
        let mut cur_position = pt_angle;

        // Start with a fairly big step size because it may have to go a long way
        let initial_step = 4. * RADIANS_PER_DEGREE;
        let mut num_cuts_done: i32 = -1;
        let mut brackets = [0.0_f32; 14];
        let mut next_position = 0.0_f32;

        // It is monotonic so no need to scan
        loop {
            let dx = (xrot as f64 - aa[0] as f64 * (cur_position as f64).cos()) as f32;
            let dy = (yrot as f64 - aa[1] as f64 * (cur_position as f64).sin()) as f32;

            // Note that distance instead of squared distance was evaluated and found to
            // give more variable results
            let cur_value = dx * dx + dy * dy;
            let ret_val = crate::imod::libcfshr::minimize1d::minimize1d(
                cur_position,
                cur_value,
                initial_step,
                0,
                &mut num_cuts_done,
                &mut brackets,
                &mut next_position,
            );

            // Yes, a very elongated ellipse can run angle differences up very high
            if ret_val != 0
                || ((cur_position - pt_angle) as f64).abs() > 90. * RADIANS_PER_DEGREE as f64
            {
                *error = 1.0e30;
                return;
            }
            if num_cuts_done > 10 {
                break;
            }
            cur_position = next_position;
        }

        // Add and save error
        derr += brackets[8] as f64;
        if s_final_fit != 0 {
            s_errors[ind] = brackets[8].sqrt();
        }
    }
    *error = (derr / s_num_pts as f64).sqrt() as f32;
    LAST_ERROR.store((*error).to_bits(), Ordering::Relaxed);
}
