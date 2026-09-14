//! Translation of `IMOD/libcfshr/findtransform.c`.
#![allow(dead_code)]

use super::linearxforms::{xf_apply, xf_unit};
use super::regression::{mult_regress, robust_regress, stat_matrices};
use std::cell::Cell;

/// Parameters installed for the next fit by `find_xf_robust_params`.
///
/// The original routine used process-global mutable values.  A fit and its
/// configuration are naturally one thread's operation, so thread-local owned
/// storage prevents one caller from changing another caller's fit.
#[derive(Clone, Copy, Default)]
struct FitControl {
    k_factor: f32,
    max_change: f32,
    max_oscill: f32,
    max_iter: i32,
    max_zero_wgt: i32,
}

thread_local! {
    static FIT_CONTROL: Cell<FitControl> = const { Cell::new(FitControl {
        k_factor: 0.,
        max_change: 0.,
        max_oscill: 0.,
        max_iter: 0,
        max_zero_wgt: 0,
    }) };
}

/// Original `findTransform` (`findtransform.c:59`).
pub unsafe fn find_transform(
    x_mat: *mut f32,
    m_size: i32,
    icol_x: i32,
    num_points: i32,
    xcen: f32,
    y_cen: f32,
    if_trans: i32,
    if_rotrans: i32,
    mut if_dev: i32,
    xf: *mut f32,
    dev_avg: *mut f32,
    dev_sd: *mut f32,
    dev_max: *mut f32,
    ipnt_max: *mut i32,
) -> i32 {
    unsafe {
        let control = FIT_CONTROL.with(Cell::get);
        let icol_y = icol_x + 1;
        let mut array_tot = icol_y * 5
            + icol_y * icol_y
            + (3 * icol_y * icol_y).max(if control.k_factor != 0. {
                2 * num_points
            } else {
                0
            });
        let mut num_extra = 0;
        if if_dev < 0 {
            if_dev = -if_dev;
            num_extra = 0.max(*ipnt_max);
        }
        if array_tot < 0 {
            return -1;
        }
        let mut array = vec![0_f32; array_tot as usize];
        // Window lengths within `array` for the slice-taking regression
        // routines; each runs to the end of the allocation, as the C pointers
        // effectively do.
        let len_from = |off: i32| (array_tot - off) as usize;
        let off_sum_x = 0;
        let off_x_mean = icol_y;
        let off_sd = 2 * icol_y;
        let off_ss_mat = 3 * icol_y;
        let off_ss_dev = off_ss_mat + icol_y * icol_y;
        let off_d_mat = off_ss_dev + icol_y * icol_y;
        let off_r_mat = off_d_mat + icol_y * icol_y;
        let x_mat_len = (num_points * m_size) as usize;
        let sum_x = array.as_mut_ptr();
        let x_mean = sum_x.add(icol_y as usize);
        let sd = x_mean.add(icol_y as usize);
        let ss_mat = sd.add(icol_y as usize);
        let ss_dev = ss_mat.add((icol_y * icol_y) as usize);
        let d_mat = ss_dev.add((icol_y * icol_y) as usize);
        let r_mat = d_mat.add((icol_y * icol_y) as usize);
        xf_unit(core::slice::from_raw_parts_mut(xf, 6), 1., 2);
        if if_rotrans == 0 && if_trans == 0 {
            if icol_x > 3 {
                for i in 0..num_points {
                    *x_mat.add((i * m_size + 2) as usize) = *x_mat.add((i * m_size + 3) as usize);
                    *x_mat.add((i * m_size + 3) as usize) = *x_mat.add((i * m_size + 4) as usize);
                }
            }
            let mut b_mat = [0_f32; 4];
            let mut c_vec = [0_f32; 2];
            let mut isol = 1;
            if control.k_factor != 0. {
                isol = robust_regress(
                    core::slice::from_raw_parts_mut(x_mat, x_mat_len),
                    m_size,
                    1,
                    2,
                    num_points,
                    2,
                    &mut b_mat,
                    2,
                    Some(&mut c_vec),
                    core::slice::from_raw_parts_mut(x_mean, len_from(off_x_mean)),
                    core::slice::from_raw_parts_mut(sd, len_from(off_sd)),
                    core::slice::from_raw_parts_mut(ss_mat, len_from(off_ss_mat)),
                    control.k_factor,
                    &mut *ipnt_max,
                    control.max_iter,
                    control.max_zero_wgt,
                    control.max_change,
                    control.max_oscill,
                );
            }
            if isol != 0 {
                isol = mult_regress(
                    core::slice::from_raw_parts(x_mat, x_mat_len),
                    m_size,
                    1,
                    2,
                    num_points,
                    2,
                    0,
                    &mut b_mat,
                    2,
                    Some(&mut c_vec),
                    core::slice::from_raw_parts_mut(x_mean, len_from(off_x_mean)),
                    core::slice::from_raw_parts_mut(sd, len_from(off_sd)),
                    core::slice::from_raw_parts_mut(ss_mat, len_from(off_ss_mat)),
                );
                for i in 0..num_points {
                    *x_mat.add((i * m_size + 4) as usize) = 1.;
                }
            }
            if isol != 0 {
                FIT_CONTROL.with(|state| {
                    state.set(FitControl {
                        k_factor: 0.,
                        ..state.get()
                    });
                });
                return isol;
            }
            for i in 0..2 {
                *xf.add(i as usize) = b_mat[(i * 2) as usize];
                *xf.add((2 + i) as usize) = b_mat[(i * 2 + 1) as usize];
                *xf.add((4 + i) as usize) = c_vec[i as usize];
            }
            if icol_x > 3 {
                for i in 0..num_points {
                    if control.k_factor != 0. {
                        *x_mat.add((i * m_size + 5) as usize) =
                            *x_mat.add((i * m_size + 4) as usize);
                    }
                    *x_mat.add((i * m_size + 4) as usize) = *x_mat.add((i * m_size + 3) as usize);
                    *x_mat.add((i * m_size + 3) as usize) = *x_mat.add((i * m_size + 2) as usize);
                }
            }
        } else if if_rotrans == 0 {
            for isol in 0..2 {
                let mut constant = 0.;
                for i in 0..num_points {
                    constant += *x_mat.add((i * m_size + icol_x + isol - 1) as usize)
                        - *x_mat.add((i * m_size + isol) as usize);
                }
                *xf.add((4 + isol) as usize) = constant / num_points as f32;
            }
        } else {
            stat_matrices(
                core::slice::from_raw_parts(x_mat, x_mat_len),
                m_size,
                1,
                icol_y,
                icol_y,
                num_points,
                core::slice::from_raw_parts_mut(sum_x, len_from(off_sum_x)),
                core::slice::from_raw_parts_mut(ss_mat, len_from(off_ss_mat)),
                core::slice::from_raw_parts_mut(ss_dev, len_from(off_ss_dev)),
                core::slice::from_raw_parts_mut(d_mat, len_from(off_d_mat)),
                core::slice::from_raw_parts_mut(r_mat, len_from(off_r_mat)),
                core::slice::from_raw_parts_mut(x_mean, len_from(off_x_mean)),
                core::slice::from_raw_parts_mut(sd, len_from(off_sd)),
                1,
            );
            let theta = -((*ss_dev.add(((icol_x - 1) * icol_y + 1) as usize)
                - *ss_dev.add((icol_x * icol_y) as usize))
                / (*ss_dev.add(((icol_x - 1) * icol_y) as usize)
                    + *ss_dev.add((icol_x * icol_y + 1) as usize)))
            .atan();
            let mut sin_theta = theta.sin();
            let mut cos_theta = theta.cos();
            if if_rotrans == 2 {
                let gmag = ((*ss_dev.add(((icol_x - 1) * icol_y) as usize)
                    + *ss_dev.add((icol_x * icol_y + 1) as usize))
                    * cos_theta
                    - (*ss_dev.add(((icol_x - 1) * icol_y + 1) as usize)
                        - *ss_dev.add((icol_x * icol_y) as usize))
                        * sin_theta)
                    / (*ss_dev.add((icol_y) as usize) + *ss_dev.add((icol_y + 1) as usize));
                sin_theta *= gmag;
                cos_theta *= gmag;
            }
            *xf = cos_theta;
            *xf.add(2) = -sin_theta;
            *xf.add(1) = sin_theta;
            *xf.add(3) = cos_theta;
            *xf.add(4) = *x_mean.add((icol_x - 1) as usize) - *x_mean * cos_theta
                + *x_mean.add(1) * sin_theta;
            *xf.add(5) = *x_mean.add((icol_y - 1) as usize)
                - *x_mean * sin_theta
                - *x_mean.add(1) * cos_theta;
        }
        FIT_CONTROL.with(|state| {
            state.set(FitControl {
                k_factor: 0.,
                ..state.get()
            });
        });
        if if_dev == 0 {
            return 0;
        }
        let mut dev_sum = 0.;
        let mut dev_sum_sq = 0.;
        *dev_max = -1.;
        for ipnt in 0..num_points + num_extra {
            let (xx, yy) = xf_apply(
                core::slice::from_raw_parts(xf, 6),
                0.,
                0.,
                *x_mat.add((ipnt * m_size) as usize),
                *x_mat.add((ipnt * m_size + 1) as usize),
                2,
            );
            let x_dev = *x_mat.add((ipnt * m_size + icol_x - 1) as usize) - xx;
            let y_dev = *x_mat.add((ipnt * m_size + icol_y - 1) as usize) - yy;
            let dev_pnt = (x_dev * x_dev + y_dev * y_dev).sqrt();
            *x_mat.add((ipnt * m_size + 9) as usize) = x_dev;
            *x_mat.add((ipnt * m_size + 10) as usize) = y_dev;
            *x_mat.add((ipnt * m_size + 12) as usize) = dev_pnt;
            if ipnt < num_points {
                dev_sum += dev_pnt;
                dev_sum_sq += dev_pnt * dev_pnt;
                if dev_pnt > *dev_max {
                    *dev_max = dev_pnt;
                    *ipnt_max = ipnt + 1;
                }
            }
            if if_dev > 1 {
                *x_mat.add((ipnt * m_size + 7) as usize) =
                    *x_mat.add((ipnt * m_size + icol_x - 1) as usize) + xcen;
                *x_mat.add((ipnt * m_size + 8) as usize) =
                    *x_mat.add((ipnt * m_size + icol_y - 1) as usize) + y_cen;
                *x_mat.add((ipnt * m_size + 11) as usize) =
                    if x_dev.abs() > 1.0e-6 && y_dev.abs() > 1.0e-6 {
                        y_dev.atan2(x_dev) / 0.017453292519943295
                    } else {
                        0.
                    };
            }
        }
        *dev_avg = 0.;
        *dev_sd = 0.;
        if num_points > 0 {
            *dev_avg = dev_sum / num_points as f32;
            if num_points > 1 {
                let den = (dev_sum_sq - num_points as f32 * *dev_avg * *dev_avg)
                    / (num_points as f32 - 1.);
                if den > 0. {
                    *dev_sd = den.sqrt();
                }
            }
        }
        0
    }
}

/// Original `findxf` (`findtransform.c:225`).
pub unsafe fn findxf(
    x_mat: *mut f32,
    m_size: *mut i32,
    icol_x: *mut i32,
    num_points: *mut i32,
    xcen: *mut f32,
    y_cen: *mut f32,
    if_trans: *mut i32,
    if_rotrans: *mut i32,
    if_dev: *mut i32,
    xf: *mut f32,
    dev_avg: *mut f32,
    dev_sd: *mut f32,
    dev_max: *mut f32,
    ipnt_max: *mut i32,
) {
    unsafe {
        if find_transform(
            x_mat,
            *m_size,
            *icol_x,
            *num_points,
            *xcen,
            *y_cen,
            *if_trans,
            *if_rotrans,
            *if_dev,
            xf,
            dev_avg,
            dev_sd,
            dev_max,
            ipnt_max,
        ) != 0
        {
            print!("ERROR: Findxf function - Allocating array for matrices\n");
            std::process::exit(1);
        }
    }
}

/// Original `findXfRobustParams` (`findtransform.c:242`).
pub unsafe fn find_xf_robust_params(
    k_factor: f32,
    max_iter: i32,
    max_zero_wgt: i32,
    max_change: f32,
    max_oscill: f32,
) {
    FIT_CONTROL.with(|state| {
        state.set(FitControl {
            k_factor,
            max_change,
            max_oscill,
            max_iter,
            max_zero_wgt,
        });
    });
}

/// Original `findxfrobustparams` (`findtransform.c:253`).
pub unsafe fn findxfrobustparams(
    kfactor: *mut f32,
    max_iter: *mut i32,
    max_zero_wgt: *mut i32,
    max_change: *mut f32,
    max_oscill: *mut f32,
) {
    unsafe {
        find_xf_robust_params(*kfactor, *max_iter, *max_zero_wgt, *max_change, *max_oscill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn translation_and_deviation_columns_match_source_layout() {
        unsafe {
            let mut x = vec![0_f32; 13 * 3];
            for i in 0..3 {
                x[i * 13] = i as f32;
                x[i * 13 + 1] = 2. * i as f32;
                x[i * 13 + 2] = i as f32 + 3.;
                x[i * 13 + 3] = 2. * i as f32 - 4.;
            }
            let mut xf = [0.; 6];
            let mut avg = 0.;
            let mut sd = 0.;
            let mut max = 0.;
            let mut at = 0;
            assert_eq!(
                find_transform(
                    x.as_mut_ptr(),
                    13,
                    3,
                    3,
                    0.,
                    0.,
                    1,
                    0,
                    2,
                    xf.as_mut_ptr(),
                    &mut avg,
                    &mut sd,
                    &mut max,
                    &mut at
                ),
                0
            );
            assert_eq!(xf, [1., 0., 0., 1., 3., -4.]);
            assert_eq!(avg, 0.);
            assert_eq!(max, 0.);
            assert_eq!(at, 1);
            assert_eq!(x[7], 3.);
            assert_eq!(x[8], -4.);
        }
    }

    #[test]
    fn general_affine_fit_uses_the_two_output_columns() {
        unsafe {
            let mut x = vec![0_f32; 5 * 4];
            let points = [(0., 0.), (1., 0.), (0., 1.), (2., -1.)];
            for (i, (px, py)) in points.into_iter().enumerate() {
                x[5 * i] = px;
                x[5 * i + 1] = py;
                x[5 * i + 2] = 2. * px + 3. * py + 4.;
                x[5 * i + 3] = -px + 0.5 * py - 2.;
            }
            let mut xf = [0.; 6];
            let mut avg = 0.;
            let mut sd = 0.;
            let mut max = 0.;
            let mut at = 0;
            assert_eq!(
                find_transform(
                    x.as_mut_ptr(),
                    5,
                    3,
                    4,
                    0.,
                    0.,
                    0,
                    0,
                    0,
                    xf.as_mut_ptr(),
                    &mut avg,
                    &mut sd,
                    &mut max,
                    &mut at
                ),
                0
            );
            for (got, expected) in xf.into_iter().zip([2., -1., 3., 0.5, 4., -2.]) {
                assert!((got - expected).abs() < 1.0e-5, "{got} != {expected}");
            }
        }
    }
}
