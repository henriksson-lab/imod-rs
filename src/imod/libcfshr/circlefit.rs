//! Translation of `IMOD/libcfshr/circlefit.c`.
#![allow(dead_code, static_mut_refs)]

use crate::imod::libcfshr::amoeba::{amoeba, amoeba_init, dual_amoeba};

const MAX_THREADS: usize = 64;
const MAXVAR: usize = 5;
const RADIANS_PER_DEGREE: f32 = 0.017_453_292_52;
const PI_VAL: f32 = 180. * RADIANS_PER_DEGREE;

static mut FIT_RADIUS: i32 = 1;
static mut FIXED_RADIUS: f32 = 0.;
static mut XP_THRD: [*const f32; MAX_THREADS] = [core::ptr::null(); MAX_THREADS];
static mut YP_THRD: [*const f32; MAX_THREADS] = [core::ptr::null(); MAX_THREADS];
static mut ZP_THRD: [*const f32; MAX_THREADS] = [core::ptr::null(); MAX_THREADS];
static mut WGT_THRD: [*const f32; MAX_THREADS] = [core::ptr::null(); MAX_THREADS];
static mut NUMPT_THRD: [i32; MAX_THREADS] = [0; MAX_THREADS];
static mut S_RADII: *mut f32 = core::ptr::null_mut();
static mut S_ANGLES: *mut f32 = core::ptr::null_mut();
static mut S_ERRORS: *mut f32 = core::ptr::null_mut();
static mut S_NUM_PTS: i32 = 0;
static mut S_FINAL_FIT: i32 = 0;
static mut S_MAX_FIT_RATIO: f32 = 0.;

/// C `circleThrough3Pts`.
pub unsafe fn circle_through_3_pts(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    rad: *mut f32,
    xc: *mut f32,
    yc: *mut f32,
) -> i32 {
    let sq1 = (x1 * x1 + y1 * y1) as f64;
    let sq2 = (x2 * x2 + y2 * y2) as f64;
    let sq3 = (x3 * x3 + y3 * y3) as f64;
    let det = |a1: f64, a2: f64, a3: f64, b1: f64, b2: f64, b3: f64, c1: f64, c2: f64, c3: f64| {
        a1 * b2 * c3 - a1 * b3 * c2 + a2 * b3 * c1 - a2 * b1 * c3 + a3 * b1 * c2 - a3 * b2 * c1
    };
    let a = det(
        x1 as f64, y1 as f64, 1., x2 as f64, y2 as f64, 1., x3 as f64, y3 as f64, 1.,
    );
    let d = -det(sq1, y1 as f64, 1., sq2, y2 as f64, 1., sq3, y3 as f64, 1.);
    let e = det(sq1, x1 as f64, 1., sq2, x2 as f64, 1., sq3, x3 as f64, 1.);
    let f = -det(
        sq1, x1 as f64, y1 as f64, sq2, x2 as f64, y2 as f64, sq3, x3 as f64, y3 as f64,
    );
    if a.abs() < 1e-15 * d.abs() || a.abs() < 1e-15 * e.abs() || a.abs() < 1e-15 * f.abs() {
        return 1;
    }
    unsafe {
        *xc = (-0.5 * d / a) as f32;
        *yc = (-0.5 * e / a) as f32;
    }
    let rsq = (d * d + e * e) / (4. * a * a) - f / a;
    if rsq < 0. {
        return 1;
    }
    unsafe {
        *rad = rsq.sqrt() as f32;
    }
    0
}

/// C `fitSphere`.
pub unsafe fn fit_sphere(
    xpt: *const f32,
    ypt: *const f32,
    zpt: *const f32,
    num_pts: i32,
    rad: *mut f32,
    xcen: *mut f32,
    ycen: *mut f32,
    zcen: *mut f32,
    rms_err: *mut f32,
) -> i32 {
    unsafe {
        fit_sphere_wgt(
            xpt,
            ypt,
            zpt,
            core::ptr::null(),
            num_pts,
            rad,
            xcen,
            ycen,
            zcen,
            rms_err,
        )
    }
}

/// C `fitSphereWgt`.
pub unsafe fn fit_sphere_wgt(
    xpt: *const f32,
    ypt: *const f32,
    zpt: *const f32,
    weights: *const f32,
    num_pts: i32,
    rad: *mut f32,
    xcen: *mut f32,
    ycen: *mut f32,
    zcen: *mut f32,
    rms_err: *mut f32,
) -> i32 {
    let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num() as usize;
    unsafe {
        XP_THRD[thrd] = xpt;
        YP_THRD[thrd] = ypt;
        NUMPT_THRD[thrd] = num_pts;
        FIXED_RADIUS = *rad;
        if !weights.is_null() {
            WGT_THRD[thrd] = weights;
        }
        let fit_radius = FIT_RADIUS as usize;
        let mut a = [0.; MAXVAR];
        if FIT_RADIUS != 0 {
            a[0] = *rad;
        }
        a[fit_radius] = *xcen;
        a[fit_radius + 1] = *ycen;
        let sphere = !zpt.is_null();
        if sphere {
            ZP_THRD[thrd] = zpt;
            a[fit_radius + 2] = *zcen;
        }
        let mut f = |v: &[f32]| -> f32 {
            let mut error = 0.;
            if sphere {
                if weights.is_null() {
                    sphere_err(v, &mut error);
                } else {
                    sphere_err_wgt(v, &mut error);
                }
            } else if weights.is_null() {
                circle_err(v, &mut error);
            } else {
                circle_err_wgt(v, &mut error);
            }
            error
        };
        let mut nvar = 2 + FIT_RADIUS as usize;
        if sphere {
            nvar += 1;
        }
        let da = [2., 2., 2., 2.];
        let mut pp = [0.; (MAXVAR + 1) * (MAXVAR + 1)];
        let mut yy = [0.; MAXVAR + 1];
        let mut ptol = [0.; MAXVAR];
        let mut iter = 0;
        let mut jmin = 0;
        let mut errmin = f(&a[..nvar]);
        if errmin > 0. {
            amoeba_init(
                &mut pp,
                &mut yy,
                MAXVAR + 1,
                nvar,
                2.,
                0.1,
                &a[..nvar],
                &da[..nvar],
                &mut f,
                &mut ptol[..nvar],
            );
            amoeba(
                &mut pp,
                &mut yy,
                MAXVAR + 1,
                nvar,
                5e-4,
                &mut f,
                &mut iter,
                &ptol[..nvar],
                &mut jmin,
            );
            for i in 0..nvar {
                a[i] = pp[jmin + i * (MAXVAR + 1)];
            }
            amoeba_init(
                &mut pp,
                &mut yy,
                MAXVAR + 1,
                nvar,
                2.,
                0.002,
                &a[..nvar],
                &da[..nvar],
                &mut f,
                &mut ptol[..nvar],
            );
            amoeba(
                &mut pp,
                &mut yy,
                MAXVAR + 1,
                nvar,
                1e-5,
                &mut f,
                &mut iter,
                &ptol[..nvar],
                &mut jmin,
            );
            for i in 0..nvar {
                a[i] = pp[jmin + i * (MAXVAR + 1)];
            }
            errmin = f(&a[..nvar]);
        }
        if FIT_RADIUS != 0 {
            *rad = a[0];
        }
        *xcen = a[fit_radius];
        *ycen = a[fit_radius + 1];
        if sphere {
            *zcen = a[fit_radius];
        }
        *rms_err = errmin.sqrt();
    }
    0
}

/// C `fitcircle`.
pub unsafe fn fitcircle(
    xpt: *const f32,
    ypt: *const f32,
    num_pts: *const i32,
    rad: *mut f32,
    xcen: *mut f32,
    ycen: *mut f32,
    rms_err: *mut f32,
) {
    unsafe {
        fit_sphere_wgt(
            xpt,
            ypt,
            core::ptr::null(),
            core::ptr::null(),
            *num_pts,
            rad,
            xcen,
            ycen,
            core::ptr::null_mut(),
            rms_err,
        );
    }
}

/// C `fitcirclewgt`.
pub unsafe fn fitcirclewgt(
    xpt: *const f32,
    ypt: *const f32,
    weights: *const f32,
    num_pts: *const i32,
    rad: *mut f32,
    xcen: *mut f32,
    ycen: *mut f32,
    rms_err: *mut f32,
) {
    unsafe {
        fit_sphere_wgt(
            xpt,
            ypt,
            core::ptr::null(),
            weights,
            *num_pts,
            rad,
            xcen,
            ycen,
            core::ptr::null_mut(),
            rms_err,
        );
    }
}

/// C `enableRadiusFitting`.
pub fn enable_radius_fitting(do_fit: i32) {
    unsafe {
        FIT_RADIUS = if do_fit != 0 { 1 } else { 0 };
    }
}

/// C `enableradiusfitting`.
pub unsafe fn enableradiusfitting(do_fit: *const i32) {
    unsafe {
        FIT_RADIUS = if *do_fit != 0 { 1 } else { 0 };
    }
}

/// C static `circleErr`.
fn circle_err(y: &[f32], error: &mut f32) {
    unsafe {
        let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num() as usize;
        let n = NUMPT_THRD[thrd] as usize;
        let rad = if FIT_RADIUS != 0 { y[0] } else { FIXED_RADIUS };
        let xcen = y[FIT_RADIUS as usize];
        let ycen = y[FIT_RADIUS as usize + 1];
        let mut err = 0_f64;
        for i in 0..n {
            let dx = *XP_THRD[thrd].add(i) - xcen;
            let dy = *YP_THRD[thrd].add(i) - ycen;
            let d = ((dx * dx + dy * dy) as f64).sqrt() - rad as f64;
            err += d * d;
        }
        *error = (err / n as f64) as f32;
    }
}

/// C static `circleErrWgt`.
fn circle_err_wgt(y: &[f32], error: &mut f32) {
    unsafe {
        let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num() as usize;
        let n = NUMPT_THRD[thrd] as usize;
        let rad = if FIT_RADIUS != 0 { y[0] } else { FIXED_RADIUS };
        let xcen = y[FIT_RADIUS as usize];
        let ycen = y[FIT_RADIUS as usize + 1];
        let mut err = 0_f64;
        for i in 0..n {
            let dx = *XP_THRD[thrd].add(i) - xcen;
            let dy = *YP_THRD[thrd].add(i) - ycen;
            let d = ((dx * dx + dy * dy) as f64).sqrt() - rad as f64;
            err += d * d * *WGT_THRD[thrd].add(i) as f64;
        }
        *error = (err / n as f64) as f32;
    }
}

/// C static `sphereErr`.
fn sphere_err(y: &[f32], error: &mut f32) {
    unsafe {
        let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num() as usize;
        let n = NUMPT_THRD[thrd] as usize;
        let rad = if FIT_RADIUS != 0 { y[0] } else { FIXED_RADIUS };
        let xcen = y[FIT_RADIUS as usize];
        let ycen = y[FIT_RADIUS as usize + 1];
        let zcen = y[FIT_RADIUS as usize + 2];
        let mut err = 0_f64;
        for i in 0..n {
            let dx = *XP_THRD[thrd].add(i) - xcen;
            let dy = *YP_THRD[thrd].add(i) - ycen;
            let dz = *ZP_THRD[thrd].add(i) - zcen;
            let d = ((dx * dx + dy * dy + dz * dz) as f64).sqrt() - rad as f64;
            err += d * d;
        }
        *error = (err / n as f64) as f32;
    }
}

/// C static `sphereErrWgt`.
fn sphere_err_wgt(y: &[f32], error: &mut f32) {
    unsafe {
        let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num() as usize;
        let n = NUMPT_THRD[thrd] as usize;
        let rad = if FIT_RADIUS != 0 { y[0] } else { FIXED_RADIUS };
        let xcen = y[FIT_RADIUS as usize];
        let ycen = y[FIT_RADIUS as usize + 1];
        let zcen = y[FIT_RADIUS as usize + 2];
        let mut err = 0_f64;
        for i in 0..n {
            let dx = *XP_THRD[thrd].add(i) - xcen;
            let dy = *YP_THRD[thrd].add(i) - ycen;
            let dz = *ZP_THRD[thrd].add(i) - zcen;
            let d = ((dx * dx + dy * dy + dz * dz) as f64).sqrt() - rad as f64;
            err += d * d * *WGT_THRD[thrd].add(i) as f64;
        }
        *error = (err / n as f64) as f32;
    }
}

/// C `fitCenteredEllipse`.
pub unsafe fn fit_centered_ellipse(
    xpt: *const f32,
    ypt: *const f32,
    num_pts: i32,
    xrad: *mut f32,
    yrad: *mut f32,
    theta: *mut f32,
    rms_err: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
        let n = num_pts as usize;
        S_ERRORS = work;
        S_ANGLES = work.add(n);
        S_RADII = S_ANGLES.add(n);
        S_NUM_PTS = num_pts;
        S_FINAL_FIT = 0;
        let mut raw_min = 1e30_f32;
        let mut raw_max = -1e30_f32;
        for i in 0..n {
            *S_ANGLES.add(i) = (*ypt.add(i)).atan2(*xpt.add(i));
            *S_RADII.add(i) = ((*xpt.add(i)).powi(2) + (*ypt.add(i)).powi(2)).sqrt();
            raw_min = raw_min.min(*S_RADII.add(i));
            raw_max = raw_max.max(*S_RADII.add(i));
        }
        let mut init = [0.; 3];
        let mut max_ratio = -1.;
        let num_close = if n < 8 {
            1
        } else if n < 12 {
            2
        } else {
            3
        };
        for j in 0..18 {
            let base = -90. * RADIANS_PER_DEGREE + j as f32 * 5. * RADIANS_PER_DEGREE;
            let mut means = [0.; 2];
            for dir in 0..2 {
                let angle = base + dir as f32 * PI_VAL / 2.;
                let mut close = [(1e10_f32, 0_f32); 3];
                for i in 0..n {
                    let mut diff = angle - *S_ANGLES.add(i);
                    while diff >= PI_VAL / 2. {
                        diff -= PI_VAL;
                    }
                    while diff < -PI_VAL / 2. {
                        diff += PI_VAL;
                    }
                    for spot in 0..num_close {
                        if diff.abs() < close[spot].0 {
                            for k in (spot + 1..num_close).rev() {
                                close[k] = close[k - 1];
                            }
                            close[spot] = (diff.abs(), *S_RADII.add(i));
                            break;
                        }
                    }
                }
                for item in close.iter().take(num_close) {
                    means[dir] += item.1 / num_close as f32;
                }
            }
            let ratio = (means[0] / means[1]).max(means[1] / means[0]);
            if ratio > max_ratio {
                max_ratio = ratio;
                init = [means[0], means[1], base];
            }
        }
        let mut aa = init;
        S_MAX_FIT_RATIO = 2. * (aa[0] / aa[1]).max(aa[1] / aa[0]);
        S_MAX_FIT_RATIO = S_MAX_FIT_RATIO.max(1.1 * raw_max / raw_min);
        let da = [aa[0] / 10., aa[1] / 10., 5. * RADIANS_PER_DEGREE];
        let mut yy = [0.; 4];
        let mut iterations = 0;
        let mut f = |v: &[f32]| {
            let mut e = 0.;
            ellipse_func(v, &mut e);
            e
        };
        let mut e = f(&aa);
        dual_amoeba(
            &mut yy,
            3,
            2.,
            &[5e-4, 1e-5],
            &[5e-4, 1e-5],
            &mut aa,
            &da,
            &mut f,
            &mut iterations,
        );
        S_FINAL_FIT = 1;
        e = f(&aa);
        *xrad = aa[0];
        *yrad = aa[1];
        *theta = aa[2] / RADIANS_PER_DEGREE;
        *rms_err = e;
        let mut close = 0;
        for i in 0..3 {
            if aa[i] - init[i] < 2e-6 * init[i] {
                close += 1;
            }
        }
        if close == 3 { 1 } else { 0 }
    }
}

/// C static `ellipseFunc`.
fn ellipse_func(aa: &[f32], error: &mut f32) {
    unsafe {
        static mut LAST_ERROR: f32 = 0.;
        if aa[0] <= 0. || aa[1] <= 0. || aa[0] < 1e-5 * aa[1] || aa[1] < 1e-5 * aa[0] {
            *error = 10. * LAST_ERROR;
            return;
        }
        let ratio = (aa[0] / aa[1]).max(aa[1] / aa[0]);
        if ratio > S_MAX_FIT_RATIO {
            *error = LAST_ERROR * (1. + 5. * (ratio / S_MAX_FIT_RATIO - 1.));
            return;
        }
        let mut derr = 0_f64;
        for i in 0..S_NUM_PTS as usize {
            let xrot = *S_RADII.add(i) * (*S_ANGLES.add(i) - aa[2]).cos();
            let yrot = *S_RADII.add(i) * (*S_ANGLES.add(i) - aa[2]).sin();
            let pt_angle = yrot.atan2(xrot);
            let mut position = pt_angle;
            let mut cuts = -1;
            let mut brackets = [0_f32; 14];
            loop {
                let dx = xrot - aa[0] * position.cos();
                let dy = yrot - aa[1] * position.sin();
                let value = dx * dx + dy * dy;
                let mut next = 0_f32;
                let ret = unsafe {
                    crate::imod::libcfshr::minimize1d::minimize1d(
                        position,
                        value,
                        4. * RADIANS_PER_DEGREE,
                        0,
                        &mut cuts,
                        brackets.as_mut_ptr(),
                        &mut next,
                    )
                };
                if ret != 0 || (position - pt_angle).abs() > 90. * RADIANS_PER_DEGREE {
                    *error = 1e30;
                    return;
                }
                if cuts > 10 {
                    break;
                }
                position = next;
            }
            derr += brackets[8] as f64;
            if S_FINAL_FIT != 0 {
                *S_ERRORS.add(i) = brackets[8].sqrt();
            }
        }
        *error = (derr / S_NUM_PTS as f64).sqrt() as f32;
        LAST_ERROR = *error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn circle_through_three_points_matches_geometric_circle() {
        let mut r = 0.;
        let mut x = 0.;
        let mut y = 0.;
        assert_eq!(
            unsafe { circle_through_3_pts(1., 0., 0., 1., -1., 0., &mut r, &mut x, &mut y) },
            0
        );
        assert_eq!((r, x, y), (1., 0., 0.));
    }
    #[test]
    fn fit_circle_recovers_exact_circle() {
        let x = [5., 3., 1., 3.];
        let y = [-1., 1., -1., -3.];
        let (mut r, mut xc, mut yc, mut e) = (1., 2., -1., 0.);
        unsafe {
            fit_sphere(
                x.as_ptr(),
                y.as_ptr(),
                core::ptr::null(),
                4,
                &mut r,
                &mut xc,
                &mut yc,
                core::ptr::null_mut(),
                &mut e,
            );
        }
        assert!(
            (r - 2.).abs() < 0.01 && (xc - 3.).abs() < 0.01 && (yc + 1.).abs() < 0.01 && e < 0.01,
            "r={r}, x={xc}, y={yc}, error={e}"
        );
    }
}
