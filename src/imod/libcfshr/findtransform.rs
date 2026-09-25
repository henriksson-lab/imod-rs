//! Translation of `IMOD/libcfshr/findtransform.c`.
//!
//! The two Fortran wrappers, `findxf` (`findtransform.c:225`) and
//! `findxfrobustparams` (`:251`), are not translated: a translated Fortran
//! caller calls [`find_transform`] / [`find_xf_robust_params`] directly (see
//! `DEAD_CODE.md`).  `findxf`'s own error path, which prints
//! `ERROR: Findxf function - Allocating array for matrices` and exits, belongs
//! to such a caller when one is translated.

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use super::linearxforms::{xf_apply, xf_unit};
use super::regression::{mult_regress, robust_regress, stat_matrices};
use super::simplestat::sums_to_avg_sd;

/// `RADIANS_PER_DEGREE` (`b3dutil.h:68`), the truncated literal the source uses.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// C `static float sKfactor = 0.` (`findtransform.c:21`), as raw bits.
static S_KFACTOR: AtomicU32 = AtomicU32::new(0);
/// C `static float sMaxChange` (`findtransform.c:22`).
static S_MAX_CHANGE: AtomicU32 = AtomicU32::new(0);
/// C `static float sMaxOscill` (`findtransform.c:22`).
static S_MAX_OSCILL: AtomicU32 = AtomicU32::new(0);
/// C `static int sMaxIter` (`findtransform.c:23`).
static S_MAX_ITER: AtomicI32 = AtomicI32::new(0);
/// C `static int sMaxZeroWgt` (`findtransform.c:23`).
static S_MAX_ZERO_WGT: AtomicI32 = AtomicI32::new(0);

/// Original `findTransform` (`findtransform.c:59`).
///
/// `x_mat` is the C's `xMat`, with `m_size` columns per point, column index
/// fastest.  `xf` is the 2x3 transform (`f[3][2]` in C).  `dev_avg`, `dev_sd`,
/// `dev_max` and `ipnt_max` are written only when `if_dev` is non-zero, and
/// `ipnt_max` is also *read* when `if_dev` is negative, exactly as in the C.
///
/// The C's one `malloc` of `arrayTot` floats is carved into `sumX`, `xMean`,
/// `SD`, `ssMat`, `ssDev`, `dMat` and `rMat`; the same carving is done here on
/// one `Vec` with `split_at_mut`.  `robustRegress` and `multRegress` receive
/// `ssMat` as their work array and in C may run on into the arrays after it,
/// so they receive everything from `ssMat` to the end.  A failed allocation
/// still returns -1.
#[allow(clippy::too_many_arguments)]
pub fn find_transform(
    x_mat: &mut [f32],
    m_size: i32,
    icol_x: i32,
    num_points: i32,
    xcen: f32,
    y_cen: f32,
    if_trans: i32,
    if_rotrans: i32,
    mut if_dev: i32,
    xf: &mut [f32],
    dev_avg: &mut f32,
    dev_sd: &mut f32,
    dev_max: &mut f32,
    ipnt_max: &mut i32,
) -> i32 {
    let mut isol: i32;
    let mut ipnt: i32 = 0;
    let mut num_extra = 0;
    let mut c_vec = [0f32; 2];
    let mut b_mat = [0f32; 4];
    let ms = m_size as usize;
    let k_factor = f32::from_bits(S_KFACTOR.load(Ordering::Relaxed));

    let icol_y = icol_x + 1;
    let robust_tot = if k_factor != 0. { 2 * num_points } else { 0 };
    let array_tot = icol_y * 5
        + icol_y * icol_y
        + (if 3 * icol_y * icol_y > robust_tot {
            3 * icol_y * icol_y
        } else {
            robust_tot
        });
    if if_dev < 0 {
        if_dev = -if_dev;
        num_extra = if 0 > *ipnt_max { 0 } else { *ipnt_max };
    }
    // `B3DMALLOC(float, arrayTot)`: a negative count is a huge `size_t` and
    // `malloc` returns NULL.
    if array_tot < 0 {
        return -1;
    }
    let mut sum_x_buf: Vec<f32> = Vec::new();
    if sum_x_buf.try_reserve_exact(array_tot as usize).is_err() {
        return -1;
    }
    sum_x_buf.resize(array_tot as usize, 0.);
    let cy = icol_y as usize;
    let (sum_x, rest) = sum_x_buf.split_at_mut(cy);
    let (x_mean, rest) = rest.split_at_mut(cy);
    let (sd, ss_mat) = rest.split_at_mut(cy);

    xf_unit(xf, 1., 2);
    if if_rotrans == 0 && if_trans == 0 {
        //
        // move 4th or 5th column into 3rd for regression
        //
        if icol_x > 3 {
            for i in 0..num_points as usize {
                x_mat[i * ms + 2] = x_mat[i * ms + 3];
                x_mat[i * ms + 3] = x_mat[i * ms + 4];
            }
        }
        isol = 1;
        if k_factor != 0. {
            isol = robust_regress(
                x_mat,
                m_size,
                1,
                2,
                num_points,
                2,
                &mut b_mat,
                2,
                Some(&mut c_vec),
                x_mean,
                sd,
                ss_mat,
                k_factor,
                &mut ipnt,
                S_MAX_ITER.load(Ordering::Relaxed),
                S_MAX_ZERO_WGT.load(Ordering::Relaxed),
                f32::from_bits(S_MAX_CHANGE.load(Ordering::Relaxed)),
                f32::from_bits(S_MAX_OSCILL.load(Ordering::Relaxed)),
            );
        }
        if isol != 0 {
            // `findtransform.c:103-106`: the loop's own body, a `printf`, is
            // commented out, so the statement the `for` governs is the
            // `multRegress` call itself.  It runs `numPoints` times, and not
            // at all for `numPoints <= 0`, which leaves `isol` at 1.  Every
            // run is identical: `multRegress` with `wgtCol` 0 reads only
            // columns 0-3 of `xMat`, which it does not write, and writes every
            // element of `xMean`, `SD`, `bMat`, `cVec` and the work array it
            // later reads (`regression.c:198-300`).  So one call gives the
            // bit-identical result of `numPoints` calls, without the source's
            // accidental O(numPoints^2) cost.
            if num_points > 0 {
                isol = mult_regress(
                    &*x_mat,
                    m_size,
                    1,
                    2,
                    num_points,
                    2,
                    0,
                    &mut b_mat,
                    2,
                    Some(&mut c_vec),
                    x_mean,
                    sd,
                    ss_mat,
                );
            }
            for i in 0..num_points as usize {
                x_mat[i * ms + 4] = 1.;
            }
        }
        if isol != 0 {
            S_KFACTOR.store(0f32.to_bits(), Ordering::Relaxed);
            return isol;
        }
        for isol in 0..2usize {
            xf[isol] = b_mat[isol * 2];
            xf[2 + isol] = b_mat[isol * 2 + 1];
            xf[4 + isol] = c_vec[isol];
        }
        //
        // shift points back out for consistency in getting residuals
        //
        if icol_x > 3 {
            for i in 0..num_points as usize {
                if k_factor != 0. {
                    x_mat[i * ms + 5] = x_mat[i * ms + 4];
                }
                x_mat[i * ms + 4] = x_mat[i * ms + 3];
                x_mat[i * ms + 3] = x_mat[i * ms + 2];
            }
        }
    } else if if_rotrans == 0 {
        for isol in 0..2i32 {
            //
            // or, if getting translations only, find difference in
            // means for X or Y
            //
            let mut constant: f32 = 0.;
            for i in 0..num_points as usize {
                constant +=
                    x_mat[i * ms + (icol_x + isol - 1) as usize] - x_mat[i * ms + isol as usize];
            }
            constant /= num_points as f32;
            xf[(4 + isol) as usize] = constant;
        }
    } else {
        //
        // or find rotations and translations only by getting means
        // and sums of squares and cross-products of deviations
        //
        let (ss_mat, rest) = ss_mat.split_at_mut(cy * cy);
        let (ss_dev, rest) = rest.split_at_mut(cy * cy);
        let (d_mat, r_mat) = rest.split_at_mut(cy * cy);
        stat_matrices(
            x_mat, m_size, 1, icol_y, icol_y, num_points, sum_x, ss_mat, ss_dev, d_mat, r_mat,
            x_mean, sd, 1,
        );
        let icx = icol_x as usize;
        let theta = ((-(ss_dev[(icx - 1) * cy + 1] - ss_dev[icx * cy])
            / (ss_dev[(icx - 1) * cy] + ss_dev[icx * cy + 1])) as f64)
            .atan() as f32;
        let mut sin_theta = (theta as f64).sin() as f32;
        let mut cos_theta = (theta as f64).cos() as f32;
        if if_rotrans == 2 {
            let gmag = ((ss_dev[(icx - 1) * cy] + ss_dev[icx * cy + 1]) * cos_theta
                - (ss_dev[(icx - 1) * cy + 1] - ss_dev[icx * cy]) * sin_theta)
                / (ss_dev[0] + ss_dev[cy + 1]);
            sin_theta *= gmag;
            cos_theta *= gmag;
        }
        xf[0] = cos_theta;
        xf[2] = -sin_theta;
        xf[1] = sin_theta;
        xf[3] = cos_theta;
        xf[4] = x_mean[icx - 1] - x_mean[0] * cos_theta + x_mean[1] * sin_theta;
        xf[5] = x_mean[cy - 1] - x_mean[0] * sin_theta - x_mean[1] * cos_theta;
    }

    S_KFACTOR.store(0f32.to_bits(), Ordering::Relaxed);
    if if_dev == 0 {
        return 0;
    }
    //
    // compute mean and max deviation between points in adjacent
    // sections after the transformation is applied
    //
    let mut dev_sum: f32 = 0.;
    let mut dev_sum_sq: f32 = 0.;
    *dev_max = -1.;
    for ipnt in 0..(num_points + num_extra) {
        let row = ipnt as usize * ms;
        let (xx, yy) = xf_apply(xf, 0., 0., x_mat[row], x_mat[row + 1], 2);
        let x_dev = x_mat[row + (icol_x - 1) as usize] - xx;
        let y_dev = x_mat[row + (icol_y - 1) as usize] - yy;
        let dev_pnt = ((x_dev * x_dev + y_dev * y_dev) as f64).sqrt() as f32;
        x_mat[row + 9] = x_dev;
        x_mat[row + 10] = y_dev;
        x_mat[row + 12] = dev_pnt;
        if ipnt < num_points {
            dev_sum += dev_pnt;
            dev_sum_sq += dev_pnt * dev_pnt;
            if dev_pnt > *dev_max {
                *dev_max = dev_pnt;
                *ipnt_max = ipnt + 1;
            }
        }
        //
        // save true, not-centered coordinate and the angle of deviation if
        // ifdev > 1
        //
        if if_dev > 1 {
            x_mat[row + 7] = x_mat[row + (icol_x - 1) as usize] + xcen;
            x_mat[row + 8] = x_mat[row + (icol_y - 1) as usize] + y_cen;
            // `B3DABS(xDev) > 1.e-6`: the float is promoted for the compare.
            if (if x_dev >= 0. { x_dev } else { -x_dev }) as f64 > 1.0e-6
                && (if y_dev >= 0. { y_dev } else { -y_dev }) as f64 > 1.0e-6
            {
                x_mat[row + 11] = ((y_dev as f64).atan2(x_dev as f64) / RADIANS_PER_DEGREE) as f32;
            } else {
                x_mat[row + 11] = 0.;
            }
        }
    }
    sums_to_avg_sd(dev_sum, dev_sum_sq, num_points, dev_avg, dev_sd);
    0
}

/// Original `findXfRobustParams` (`findtransform.c:241`).
pub fn find_xf_robust_params(
    k_factor: f32,
    max_iter: i32,
    max_zero_wgt: i32,
    max_change: f32,
    max_oscill: f32,
) {
    S_KFACTOR.store(k_factor.to_bits(), Ordering::Relaxed);
    S_MAX_ITER.store(max_iter, Ordering::Relaxed);
    S_MAX_ZERO_WGT.store(max_zero_wgt, Ordering::Relaxed);
    S_MAX_CHANGE.store(max_change.to_bits(), Ordering::Relaxed);
    S_MAX_OSCILL.store(max_oscill.to_bits(), Ordering::Relaxed);
}
