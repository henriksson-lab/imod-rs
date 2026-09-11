//! Translation of `IMOD/libcfshr/simplestat.c`.
#![allow(dead_code)]

/// C `avgSD`.
pub unsafe fn avg_sd(x: *const f32, n: i32, avg: *mut f32, sd: *mut f32, sem: *mut f32) {
    unsafe {
        let mut sx = 0.0_f32;
        let mut sxsq = 0.0_f32;
        for i in 0..n {
            sx += *x.add(i as usize);
        }
        *avg = sx / n as f32;
        sx = 0.0;
        for i in 0..n {
            let d = *x.add(i as usize) - *avg;
            sx += d;
            sxsq += d * d;
        }
        let mut avnew = 0.0;
        sums_to_avg_sd(sx, sxsq, n, &mut avnew, sd);
        *avg += avnew;
        *sem = 0.0;
        if n > 0 {
            *sem = *sd / (n as f64).sqrt() as f32;
        }
    }
}

/// C Fortran wrapper `avgsd`.
pub unsafe fn avg_sd_fortran(
    x: *const f32,
    n: *const i32,
    avg: *mut f32,
    sd: *mut f32,
    sem: *mut f32,
) {
    unsafe { avg_sd(x, *n, avg, sd, sem) }
}

/// C `sumsToAvgSD`.
pub unsafe fn sums_to_avg_sd(sx: f32, sxsq: f32, n: i32, avg: *mut f32, sd: *mut f32) {
    unsafe {
        *avg = 0.0;
        *sd = 0.0;
        if n <= 0 {
            return;
        }
        *avg = sx / n as f32;
        if n > 1 {
            let den = (sxsq - n as f32 * *avg * *avg) as f64 / (n as f64 - 1.0);
            if den > 0.0 {
                *sd = den.sqrt() as f32;
            }
        }
    }
}

/// C Fortran wrapper `sums_to_avgsd`.
pub unsafe fn sums_to_avg_sd_fortran(
    sx: *const f32,
    sxsq: *const f32,
    n: *const i32,
    avg: *mut f32,
    sd: *mut f32,
) {
    unsafe { sums_to_avg_sd(*sx, *sxsq, *n, avg, sd) }
}

/// C `sumsToAvgSDdbl`.
pub unsafe fn sums_to_avg_sd_dbl(
    sx8: f64,
    sxsq8: f64,
    n1: i32,
    n2: i32,
    avg: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        let mut avg8 = 0.0;
        let mut sd8 = 0.0;
        sums_to_avg_sd_all_dbl(sx8, sxsq8, n1, n2, &mut avg8, &mut sd8);
        *avg = avg8 as f32;
        *sd = sd8 as f32;
    }
}

/// C Fortran wrapper `sums_to_avgsd8`.
pub unsafe fn sums_to_avg_sd_dbl_fortran(
    sx8: *const f64,
    sxsq8: *const f64,
    n1: *const i32,
    n2: *const i32,
    avg: *mut f32,
    sd: *mut f32,
) {
    unsafe { sums_to_avg_sd_dbl(*sx8, *sxsq8, *n1, *n2, avg, sd) }
}

/// C `sumsToAvgSDallDbl`.
pub unsafe fn sums_to_avg_sd_all_dbl(
    sx8: f64,
    sxsq8: f64,
    n1: i32,
    n2: i32,
    avg: *mut f64,
    sd: *mut f64,
) {
    unsafe {
        *avg = 0.0;
        *sd = 0.0;
        let dn = n1 as f64 * n2 as f64;
        if dn <= 0.0 {
            return;
        }
        let avg8 = sx8 / dn;
        *avg = avg8;
        if dn > 1.0 {
            let den = (sxsq8 - dn * avg8 * avg8) / (dn - 1.0);
            if den > 0.0 {
                *sd = den.sqrt();
            }
        }
    }
}

/// C Fortran wrapper `sumstoavgsdalldbl`.
pub unsafe fn sums_to_avg_sd_all_dbl_fortran(
    sx8: *const f64,
    sxsq8: *const f64,
    n1: *const i32,
    n2: *const i32,
    avg: *mut f64,
    sd: *mut f64,
) {
    unsafe { sums_to_avg_sd_all_dbl(*sx8, *sxsq8, *n1, *n2, avg, sd) }
}

/// C `arrayMinMaxMean`.
pub unsafe fn array_min_max_mean(
    array: *const f32,
    nx: i32,
    _ny: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe {
        let mut sum_dbl = 0.0;
        *dmin = 1.0e37;
        *dmax = -1.0e37;
        for iy in iy0..=iy1 {
            let mut sum_tmp = 0.0_f32;
            for ix in ix0..=ix1 {
                let den = *array.add((iy * nx + ix) as usize);
                sum_tmp += den;
                *dmin = (*dmin).min(den);
                *dmax = (*dmax).max(den);
            }
            sum_dbl += sum_tmp as f64;
        }
        *dmean = (sum_dbl / ((ix1 + 1 - ix0) as f64 * (iy1 + 1 - iy0) as f64)) as f32;
    }
}

/// C `imageSubareaMean`.
pub unsafe fn image_subarea_mean(
    array: *const core::ffi::c_void,
    data_type: i32,
    nx_dim: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
) -> f32 {
    unsafe {
        let mut sum_dbl = 0.0_f64;
        for iy in iy0..=iy1 {
            let mut sum_tmp = 0.0_f32;
            for ix in ix0..=ix1 {
                let ind = (iy * nx_dim + ix) as usize;
                sum_tmp += match data_type {
                    0 => *array.cast::<u8>().add(ind) as f32,
                    1 => *array.cast::<i16>().add(ind) as f32,
                    6 => *array.cast::<u16>().add(ind) as f32,
                    2 => *array.cast::<f32>().add(ind),
                    _ => return 0.0,
                };
            }
            sum_dbl += sum_tmp as f64;
        }
        (sum_dbl / ((ix1 + 1 - ix0) as f64 * (iy1 + 1 - iy0) as f64)) as f32
    }
}

/// C Fortran wrapper `iclden`.
pub unsafe fn array_min_max_mean_fortran(
    array: *const f32,
    nx: *const i32,
    ny: *const i32,
    ix0: *const i32,
    ix1: *const i32,
    iy0: *const i32,
    iy1: *const i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe {
        array_min_max_mean(
            array,
            *nx,
            *ny,
            *ix0 - 1,
            *ix1 - 1,
            *iy0 - 1,
            *iy1 - 1,
            dmin,
            dmax,
            dmean,
        )
    }
}

/// C `arrayMinMaxMeanSd`.
pub unsafe fn array_min_max_mean_sd(
    array: *const f32,
    nx: i32,
    ny: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dmin: *mut f32,
    dmax: *mut f32,
    sum_dbl: *mut f64,
    sum_sq_dbl: *mut f64,
    avg: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        let nx_area = ix1 + 1 - ix0;
        let ny_area = iy1 + 1 - iy0;
        *sum_dbl = 0.0;
        *sum_sq_dbl = 0.0;
        *dmin = 1.0e37;
        *dmax = -1.0e37;
        let x_div = 2.max(8.min(nx - 2));
        let y_div = 2.max(8.min(ny - 2));
        let mut rough_mean = 0.0_f64;
        let mut nsum = 0;
        for jy in 1..y_div {
            let iy = iy0 + jy * ny_area / y_div;
            for jx in 1..x_div {
                let ix = ix0 + jx * nx_area / x_div;
                rough_mean += *array.add((iy * nx + ix) as usize) as f64;
                nsum += 1;
            }
        }
        rough_mean /= nsum as f64;
        for iy in iy0..=iy1 {
            let mut sum_tmp = 0.0;
            let mut sum_tmp_sq = 0.0;
            for ix in ix0..=ix1 {
                // `den` is a `float` in the source, so subtracting the double
                // `roughMean` rounds back to single before it is accumulated,
                // and `den * den` is a single-precision product.
                let mut den = *array.add((iy * nx + ix) as usize);
                *dmin = (*dmin).min(den);
                *dmax = (*dmax).max(den);
                den = (den as f64 - rough_mean) as f32;
                sum_tmp += den as f64;
                sum_tmp_sq += (den * den) as f64;
            }
            *sum_dbl += sum_tmp;
            *sum_sq_dbl += sum_tmp_sq;
        }
        let mut avg8 = 0.0;
        let mut sd8 = 0.0;
        sums_to_avg_sd_all_dbl(*sum_dbl, *sum_sq_dbl, nx_area, ny_area, &mut avg8, &mut sd8);
        avg8 += rough_mean;
        // `nxArea * (nyArea * avg8)`: the parenthesised product is what the
        // source multiplies, and the two groupings do not round alike.
        *sum_dbl = nx_area as f64 * (ny_area as f64 * avg8);
        *sum_sq_dbl = (nx_area as f64 * ny_area as f64 - 1.0) * sd8 * sd8 + *sum_dbl * avg8;
        *avg = avg8 as f32;
        *sd = sd8 as f32;
    }
}

/// C Fortran wrapper `iclavgsd`.
pub unsafe fn array_min_max_mean_sd_fortran(
    array: *const f32,
    nx: *const i32,
    ny: *const i32,
    ix0: *const i32,
    ix1: *const i32,
    iy0: *const i32,
    iy1: *const i32,
    dmin: *mut f32,
    dmax: *mut f32,
    sum_dbl: *mut f64,
    sum_sq_dbl: *mut f64,
    avg: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        array_min_max_mean_sd(
            array,
            *nx,
            *ny,
            *ix0 - 1,
            *ix1 - 1,
            *iy0 - 1,
            *iy1 - 1,
            dmin,
            dmax,
            sum_dbl,
            sum_sq_dbl,
            avg,
            sd,
        )
    }
}

/// C `scaleArrayForMode`.
pub unsafe fn scale_array_for_mode(
    array: *mut f32,
    nx_dim: i32,
    mode: i32,
    nx1: i32,
    nx2: i32,
    ny1: i32,
    ny2: i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe {
        *dmax = if mode == 6 {
            65530.0
        } else if mode == 1 {
            32767.0
        } else {
            255.0
        };
        let mut dmin_in = 1.0e30_f32;
        let mut dmax_in = -1.0e30_f32;
        *dmean = 0.0;
        for iy in ny1..=ny2 {
            for ix in nx1..=nx2 {
                let val = *array.add((iy * nx_dim + ix) as usize);
                dmin_in = dmin_in.min(val);
                dmax_in = dmax_in.max(val);
            }
        }
        let sclfac = 0.99999 * *dmax / (dmax_in - dmin_in);
        for iy in ny1..=ny2 {
            let mut tsum = 0.0;
            for ix in nx1..=nx2 {
                let val = sclfac * (*array.add((iy * nx_dim + ix) as usize) - dmin_in);
                tsum += val;
                *array.add((iy * nx_dim + ix) as usize) = val;
            }
            *dmean += tsum;
        }
        *dmean /= (nx2 - nx1 + 1) as f32 * (ny2 - ny1 + 1) as f32;
        *dmin = 0.0;
    }
}

/// C Fortran wrapper `isetdn`.
pub unsafe fn scale_array_for_mode_fortran(
    array: *mut f32,
    nx_dim: *const i32,
    _ny_dim: *const i32,
    mode: *const i32,
    nx1: *const i32,
    nx2: *const i32,
    ny1: *const i32,
    ny2: *const i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe {
        scale_array_for_mode(
            array,
            *nx_dim,
            *mode,
            *nx1 - 1,
            *nx2 - 1,
            *ny1 - 1,
            *ny2 - 1,
            dmin,
            dmax,
            dmean,
        )
    }
}

/// C `lsFit`.
pub unsafe fn ls_fit(
    x: *const f32,
    y: *const f32,
    num: i32,
    slope: *mut f32,
    intcp: *mut f32,
    ro: *mut f32,
) {
    unsafe {
        let mut sa = 0.;
        let mut sb = 0.;
        let mut se = 0.;
        let mut ypred = 0.;
        let mut prederr = 0.;
        ls_fit_pred(
            x,
            y,
            num,
            slope,
            intcp,
            ro,
            &mut sa,
            &mut sb,
            &mut se,
            0.,
            &mut ypred,
            &mut prederr,
        );
    }
}
/// C Fortran wrapper `lsfit`.
pub unsafe fn ls_fit_fortran(
    x: *const f32,
    y: *const f32,
    num: *const i32,
    slope: *mut f32,
    intcp: *mut f32,
    ro: *mut f32,
) {
    unsafe { ls_fit(x, y, *num, slope, intcp, ro) }
}

/// C `lsFitPred`.
pub unsafe fn ls_fit_pred(
    x: *const f32,
    y: *const f32,
    n: i32,
    slope: *mut f32,
    bint: *mut f32,
    ro: *mut f32,
    sa: *mut f32,
    sb: *mut f32,
    se: *mut f32,
    xpred: f32,
    ypred: *mut f32,
    prederr: *mut f32,
) {
    unsafe {
        *slope = 1.;
        *bint = 0.;
        *ro = 0.;
        *sa = 0.;
        *sb = 0.;
        *se = 0.;
        *ypred = 0.;
        *prederr = 0.;
        if n < 2 {
            return;
        }
        let (mut sx, mut sy) = (0., 0.);
        for i in 0..n {
            sx += *x.add(i as usize) as f64;
            sy += *y.add(i as usize) as f64;
        }
        let xbar = sx / n as f64;
        let ybar = sy / n as f64;
        let (mut sxpsq, mut sxyp, mut sypsq) = (0., 0., 0.);
        for i in 0..n {
            let xp = *x.add(i as usize) as f64 - xbar;
            let yp = *y.add(i as usize) as f64 - ybar;
            sxpsq += xp * xp;
            sypsq += yp * yp;
            sxyp += xp * yp;
        }
        let d = n as f64 * sxpsq;
        let dslope = sxyp / sxpsq;
        *slope = dslope as f32;
        let dbint = (ybar * sxpsq - xbar * sxyp) / sxpsq;
        *bint = dbint as f32;
        let roden = (sxpsq * sypsq).sqrt();
        *ro = 1.;
        if roden != 0. && sxyp.abs() <= roden.abs() {
            *ro = (sxyp / roden) as f32;
        }
        let sxy = sxyp + n as f64 * xbar * ybar;
        let sysq = sypsq + n as f64 * ybar * ybar;
        let setmp = sysq - dbint * sy - dslope * sxy;
        if n > 2 && setmp > 0. {
            *se = (setmp / (n as f64 - 2.)).sqrt() as f32;
        }
        *sa = (*se as f64 * (1. / n as f64 + (sx * sx / n as f64) / d).sqrt()) as f32;
        *sb = (*se as f64 / (d / n as f64).sqrt()) as f32;
        *ypred = (dslope * xpred as f64 + dbint) as f32;
        *prederr = (*se as f64
            * (1. + 1. / n as f64 + n as f64 * (xpred as f64 - xbar) * (xpred as f64 - xbar) / d)
                .sqrt()) as f32;
    }
}
/// C Fortran wrapper `lsfitpred`.
pub unsafe fn ls_fit_pred_fortran(
    x: *const f32,
    y: *const f32,
    n: *const i32,
    slope: *mut f32,
    bint: *mut f32,
    ro: *mut f32,
    sa: *mut f32,
    sb: *mut f32,
    se: *mut f32,
    xpred: *const f32,
    ypred: *mut f32,
    prederr: *mut f32,
) {
    unsafe {
        ls_fit_pred(
            x, y, *n, slope, bint, ro, sa, sb, se, *xpred, ypred, prederr,
        )
    }
}
/// C Fortran wrapper `lsfits`.
pub unsafe fn ls_fit_standard_errors_fortran(
    x: *const f32,
    y: *const f32,
    n: *const i32,
    slope: *mut f32,
    bint: *mut f32,
    ro: *mut f32,
    sa: *mut f32,
    sb: *mut f32,
    se: *mut f32,
) {
    unsafe {
        let xpred = *x.add(1);
        let (mut ypred, mut prederr) = (0., 0.);
        ls_fit_pred(
            x,
            y,
            *n,
            slope,
            bint,
            ro,
            sa,
            sb,
            se,
            xpred,
            &mut ypred,
            &mut prederr,
        )
    }
}

/// C `lsFit2`.
pub unsafe fn ls_fit2(
    x1: *const f32,
    x2: *const f32,
    y: *const f32,
    n: i32,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
) {
    unsafe {
        let (mut ypred, mut prederr) = (0., 0.);
        ls_fit2_pred(x1, x2, y, n, a, b, c, 0., 0., &mut ypred, &mut prederr)
    }
}
/// C Fortran wrapper `lsfit2`.
pub unsafe fn ls_fit2_fortran(
    x1: *const f32,
    x2: *const f32,
    y: *const f32,
    n: *const i32,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
) {
    unsafe { ls_fit2(x1, x2, y, *n, a, b, c) }
}
/// C Fortran wrapper `lsfit2noc`.
pub unsafe fn ls_fit2_no_constant_fortran(
    x1: *const f32,
    x2: *const f32,
    y: *const f32,
    n: *const i32,
    a: *mut f32,
    b: *mut f32,
) {
    unsafe { ls_fit2(x1, x2, y, *n, a, b, core::ptr::null_mut()) }
}

/// C `lsFit2Pred`.
pub unsafe fn ls_fit2_pred(
    x1: *const f32,
    x2: *const f32,
    y: *const f32,
    n: i32,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    x1pred: f32,
    x2pred: f32,
    ypred: *mut f32,
    prederr: *mut f32,
) {
    unsafe {
        let (mut x1s, mut x2s, mut ys) = (0., 0., 0.);
        for i in 0..n {
            x1s += *x1.add(i as usize) as f64;
            x2s += *x2.add(i as usize) as f64;
            ys += *y.add(i as usize) as f64;
        }
        let (x1m, x2m, ym) = (x1s / n as f64, x2s / n as f64, ys / n as f64);
        let (mut x1sqs, mut x2sqs, mut x1x2s, mut x1ys, mut x2ys, mut ysqs) =
            (0., 0., 0., 0., 0., 0.);
        for i in 0..n {
            let (mut p1, mut p2, mut py) = (
                *x1.add(i as usize) as f64,
                *x2.add(i as usize) as f64,
                *y.add(i as usize) as f64,
            );
            if !c.is_null() {
                p1 -= x1m;
                p2 -= x2m;
                py -= ym;
            }
            x1sqs += p1 * p1;
            x2sqs += p2 * p2;
            x1ys += p1 * py;
            x2ys += p2 * py;
            x1x2s += p1 * p2;
            ysqs += py * py;
        }
        let denom = x1sqs * x2sqs - x1x2s * x1x2s;
        let anum = x1ys * x2sqs - x1x2s * x2ys;
        let bnum = x1sqs * x2ys - x1ys * x1x2s;
        *a = 0.;
        *b = 0.;
        if !c.is_null() {
            *c = ym as f32;
        }
        *ypred = ym as f32;
        *prederr = 0.;
        let absanum = anum.abs().max(bnum.abs());
        if denom.abs() < 1.0e-30 * absanum {
            return;
        }
        let dbla = anum / denom;
        let dblb = bnum / denom;
        *a = dbla as f32;
        *b = dblb as f32;
        let mut dblc = 0.;
        if !c.is_null() {
            dblc = ym - dbla * x1m - dblb * x2m;
            *c = dblc as f32;
        }
        *ypred = (dbla * x1pred as f64 + dblb * x2pred as f64 + dblc) as f32;
        let c11 = x2sqs / denom;
        let c22 = x1sqs / denom;
        let c12 = -x1x2s / denom;
        let devss = ysqs - dbla * x1ys - dblb * x2ys;
        let predsq = 1.
            + 1. / n as f64
            + c11 * (x1pred as f64 - x1m) * (x1pred as f64 - x1m)
            + c22 * (x2pred as f64 - x2m) * (x2pred as f64 - x2m)
            + 2. * c12 * (x1pred as f64 - x1m) * (x2pred as f64 - x2m);
        if n < 4 || predsq < 0. || devss < 0. {
            return;
        }
        *prederr = ((devss / (n as f64 - 3.)) * predsq).sqrt() as f32;
    }
}
/// C Fortran wrapper `lsfit2pred`.
pub unsafe fn ls_fit2_pred_fortran(
    x1: *const f32,
    x2: *const f32,
    y: *const f32,
    n: *const i32,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    x1pred: *const f32,
    x2pred: *const f32,
    ypred: *mut f32,
    prederr: *mut f32,
) {
    unsafe { ls_fit2_pred(x1, x2, y, *n, a, b, c, *x1pred, *x2pred, ypred, prederr) }
}

/// C `lsFit3`.
pub unsafe fn ls_fit3(
    x1: *const f32,
    x2: *const f32,
    x3: *const f32,
    y: *const f32,
    n: i32,
    a1: *mut f32,
    a2: *mut f32,
    a3: *mut f32,
    c: *mut f32,
) {
    unsafe {
        let (mut x1s, mut x2s, mut x3s, mut ys) = (0_f32, 0., 0., 0.);
        for i in 0..n {
            x1s += *x1.add(i as usize);
            x2s += *x2.add(i as usize);
            x3s += *x3.add(i as usize);
            ys += *y.add(i as usize);
        }
        let (x1m, x2m, x3m, ym) = (
            x1s / n as f32,
            x2s / n as f32,
            x3s / n as f32,
            ys / n as f32,
        );
        let (mut q1, mut q2, mut q3, mut q12, mut q13, mut q23, mut r1, mut r2, mut r3) =
            (0_f32, 0., 0., 0., 0., 0., 0., 0., 0.);
        for i in 0..n {
            let p1 = *x1.add(i as usize) - x1m;
            let p2 = *x2.add(i as usize) - x2m;
            let p3 = *x3.add(i as usize) - x3m;
            let py = *y.add(i as usize) - ym;
            q1 += p1 * p1;
            q2 += p2 * p2;
            q3 += p3 * p3;
            q12 += p1 * p2;
            q13 += p1 * p3;
            q23 += p2 * p3;
            r1 += p1 * py;
            r2 += p2 * py;
            r3 += p3 * py;
        }
        *a1 = 0.;
        *a2 = 0.;
        *a3 = 0.;
        *c = 0.;
        let den = (q1 * q2 * q3 - q1 * q23 * q23 + q12 * q23 * q13 - q12 * q12 * q3
            + q13 * q12 * q23
            - q13 * q2 * q13) as f64;
        let num1 = (r1 * q2 * q3 - r1 * q23 * q23 + q12 * q23 * r3 - q12 * r2 * q3 + q13 * r2 * q23
            - q13 * q2 * r3) as f64;
        let num2 = (q1 * r2 * q3 - q1 * q23 * r3 + r1 * q23 * q13 - r1 * q12 * q3 + q13 * q12 * r3
            - q13 * r2 * q13) as f64;
        let num3 = (q1 * q2 * r3 - q1 * r2 * q23 + q12 * r2 * q13 - q12 * q12 * r3 + r1 * q12 * q23
            - r1 * q2 * q13) as f64;
        let maxnum = num1.abs().max(num2.abs()).max(num3.abs());
        if den.abs() < 1.0e-30 * maxnum {
            return;
        }
        *a1 = (num1 / den) as f32;
        *a2 = (num2 / den) as f32;
        *a3 = (num3 / den) as f32;
        *c = ym - *a1 * x1m - *a2 * x2m - *a3 * x3m;
    }
}
/// C Fortran wrapper `lsfit3`.
pub unsafe fn ls_fit3_fortran(
    x1: *const f32,
    x2: *const f32,
    x3: *const f32,
    y: *const f32,
    n: *const i32,
    a1: *mut f32,
    a2: *mut f32,
    a3: *mut f32,
    c: *mut f32,
) {
    unsafe { ls_fit3(x1, x2, x3, y, *n, a1, a2, a3, c) }
}

/// C `eigenSort`.
pub unsafe fn eigen_sort(
    val: *mut f64,
    vec: *mut f64,
    n: i32,
    row_stride: i32,
    col_stride: i32,
    use_abs: i32,
) {
    unsafe {
        for i in 0..n - 1 {
            let mut imax = i;
            for j in i + 1..n {
                if (use_abs != 0 && (*val.add(j as usize)).abs() > (*val.add(imax as usize)).abs())
                    || (use_abs == 0 && *val.add(j as usize) >= *val.add(imax as usize))
                {
                    imax = j;
                }
            }
            if imax != i {
                let tmp = *val.add(i as usize);
                *val.add(i as usize) = *val.add(imax as usize);
                *val.add(imax as usize) = tmp;
                for k in 0..n {
                    let first = (k * row_stride + i * col_stride) as usize;
                    let second = (k * row_stride + imax * col_stride) as usize;
                    let tmp = *vec.add(first);
                    *vec.add(first) = *vec.add(second);
                    *vec.add(second) = tmp;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summaries_and_subareas_follow_source() {
        unsafe {
            let a = [1_f32, 2., 3., 4.];
            let (mut av, mut sd, mut sem) = (0., 0., 0.);
            avg_sd(a.as_ptr(), 4, &mut av, &mut sd, &mut sem);
            assert_eq!(av, 2.5);
            assert!((sd - 1.2909944).abs() < 1.0e-5);
            let (mut lo, mut hi, mut mean) = (0., 0., 0.);
            array_min_max_mean(a.as_ptr(), 2, 2, 0, 1, 0, 1, &mut lo, &mut hi, &mut mean);
            assert_eq!((lo, hi, mean), (1., 4., 2.5));
            assert_eq!(image_subarea_mean(a.as_ptr().cast(), 2, 2, 0, 1, 0, 1), 2.5);
        }
    }
    #[test]
    fn regressions_and_eigensort_follow_source() {
        unsafe {
            let x = [0_f32, 1., 2., 3.];
            let y = [1_f32, 3., 5., 7.];
            let (mut slope, mut b, mut ro) = (0., 0., 0.);
            ls_fit(x.as_ptr(), y.as_ptr(), 4, &mut slope, &mut b, &mut ro);
            assert!((slope - 2.).abs() < 1.0e-5);
            assert!((b - 1.).abs() < 1.0e-5);
            let x2 = [0_f32, 1., 0., 1., 2.];
            let x3 = [0_f32, 0., 1., 1., 2.];
            let yy = [1_f32, 3., 4., 6., 11.];
            let (mut aa, mut bb, mut cc) = (0., 0., 0.);
            ls_fit2(
                x2.as_ptr(),
                x3.as_ptr(),
                yy.as_ptr(),
                5,
                &mut aa,
                &mut bb,
                &mut cc,
            );
            assert!((aa - 2.).abs() < 1.0e-4);
            assert!((bb - 3.).abs() < 1.0e-4);
            assert!((cc - 1.).abs() < 1.0e-4);
            let mut val = [2_f64, -5., 3.];
            let mut vec = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
            eigen_sort(val.as_mut_ptr(), vec.as_mut_ptr(), 3, 1, 3, 1);
            assert_eq!(val, [-5., 3., 2.]);
        }
    }
}
