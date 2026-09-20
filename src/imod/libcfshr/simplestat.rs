//! Translation of `IMOD/libcfshr/simplestat.c`.

/// C `avgSD`.
pub fn avg_sd(x: &[f32], n: i32, avg: &mut f32, sd: &mut f32, sem: &mut f32) {
    let mut sx = 0.0_f32;
    let mut sxsq = 0.0_f32;
    for i in 0..n {
        sx += x[i as usize];
    }
    *avg = sx / n as f32;
    sx = 0.0;
    for i in 0..n {
        let d = x[i as usize] - *avg;
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

/// C `sumsToAvgSD`.
pub fn sums_to_avg_sd(sx: f32, sxsq: f32, n: i32, avg: &mut f32, sd: &mut f32) {
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

/// C `sumsToAvgSDdbl`.
pub fn sums_to_avg_sd_dbl(sx8: f64, sxsq8: f64, n1: i32, n2: i32, avg: &mut f32, sd: &mut f32) {
    let mut avg8 = 0.0;
    let mut sd8 = 0.0;
    sums_to_avg_sd_all_dbl(sx8, sxsq8, n1, n2, &mut avg8, &mut sd8);
    *avg = avg8 as f32;
    *sd = sd8 as f32;
}

/// C `sumsToAvgSDallDbl`.
pub fn sums_to_avg_sd_all_dbl(sx8: f64, sxsq8: f64, n1: i32, n2: i32, avg: &mut f64, sd: &mut f64) {
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

/// C `arrayMinMaxMean`.
pub fn array_min_max_mean(
    array: &[f32],
    nx: i32,
    _ny: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    dmean: &mut f32,
) {
    let mut sum_dbl = 0.0;
    *dmin = 1.0e37;
    *dmax = -1.0e37;
    for iy in iy0..=iy1 {
        let mut sum_tmp = 0.0_f32;
        for ix in ix0..=ix1 {
            let den = array[(iy * nx + ix) as usize];
            sum_tmp += den;
            // `B3DMIN`/`B3DMAX` are `a < b ? a : b`, which take the second
            // operand when the comparison is false; `f32::min`/`max` skip a
            // NaN instead.
            *dmin = if *dmin < den { *dmin } else { den };
            *dmax = if *dmax > den { *dmax } else { den };
        }
        sum_dbl += sum_tmp as f64;
    }
    *dmean = (sum_dbl / ((ix1 + 1 - ix0) as f64 * (iy1 + 1 - iy0) as f64)) as f32;
}

/// C Fortran wrapper `iclden`.
pub fn array_min_max_mean_fortran(
    array: &[f32],
    nx: &i32,
    ny: &i32,
    ix0: &i32,
    ix1: &i32,
    iy0: &i32,
    iy1: &i32,
    dmin: &mut f32,
    dmax: &mut f32,
    dmean: &mut f32,
) {
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

/// C `arrayMinMaxMeanSd`.
pub fn array_min_max_mean_sd(
    array: &[f32],
    nx: i32,
    ny: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    sum_dbl: &mut f64,
    sum_sq_dbl: &mut f64,
    avg: &mut f32,
    sd: &mut f32,
) {
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
            rough_mean += array[(iy * nx + ix) as usize] as f64;
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
            let mut den = array[(iy * nx + ix) as usize];
            *dmin = if *dmin < den { *dmin } else { den };
            *dmax = if *dmax > den { *dmax } else { den };
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
    // source multiplies, and the two groupings do not round alike.  The
    // returned sums are reconstructed from the mean and SD, not the sums
    // that were accumulated above (`simplestat.c:316-317`).
    *sum_dbl = nx_area as f64 * (ny_area as f64 * avg8);
    *sum_sq_dbl = (nx_area as f64 * ny_area as f64 - 1.0) * sd8 * sd8 + *sum_dbl * avg8;
    *avg = avg8 as f32;
    *sd = sd8 as f32;
}

/// C Fortran wrapper `iclavgsd`.
pub fn array_min_max_mean_sd_fortran(
    array: &[f32],
    nx: &i32,
    ny: &i32,
    ix0: &i32,
    ix1: &i32,
    iy0: &i32,
    iy1: &i32,
    dmin: &mut f32,
    dmax: &mut f32,
    sum_dbl: &mut f64,
    sum_sq_dbl: &mut f64,
    avg: &mut f32,
    sd: &mut f32,
) {
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

/// C `scaleArrayForMode`.
pub fn scale_array_for_mode(
    array: &mut [f32],
    nx_dim: i32,
    mode: i32,
    nx1: i32,
    nx2: i32,
    ny1: i32,
    ny2: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    dmean: &mut f32,
) {
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
            let val = array[(iy * nx_dim + ix) as usize];
            if val < dmin_in {
                dmin_in = val;
            }
            if val > dmax_in {
                dmax_in = val;
            }
        }
    }
    // `0.99999` is a double literal, so the source computes the whole
    // quotient in double and rounds once into the `float sclfac`.
    let sclfac = (0.99999 * *dmax as f64 / (dmax_in - dmin_in) as f64) as f32;
    for iy in ny1..=ny2 {
        let mut tsum = 0.0;
        for ix in nx1..=nx2 {
            let val = sclfac * (array[(iy * nx_dim + ix) as usize] - dmin_in);
            tsum += val;
            array[(iy * nx_dim + ix) as usize] = val;
        }
        *dmean += tsum;
    }
    // `(nx2 - nx1 + 1.)` promotes the extents to double, so the divide is
    // a double one and `*dmean` rounds back to float once.
    *dmean = (*dmean as f64 / (((nx2 - nx1) as f64 + 1.) * ((ny2 - ny1) as f64 + 1.))) as f32;
    *dmin = 0.0;
}

/// C `lsFit`.
pub fn ls_fit(x: &[f32], y: &[f32], num: i32, slope: &mut f32, intcp: &mut f32, ro: &mut f32) {
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

/// C `lsFitPred`.
pub fn ls_fit_pred(
    x: &[f32],
    y: &[f32],
    n: i32,
    slope: &mut f32,
    bint: &mut f32,
    ro: &mut f32,
    sa: &mut f32,
    sb: &mut f32,
    se: &mut f32,
    xpred: f32,
    ypred: &mut f32,
    prederr: &mut f32,
) {
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
        sx += x[i as usize] as f64;
        sy += y[i as usize] as f64;
    }
    let xbar = sx / n as f64;
    let ybar = sy / n as f64;
    let (mut sxpsq, mut sxyp, mut sypsq) = (0., 0., 0.);
    for i in 0..n {
        let xp = x[i as usize] as f64 - xbar;
        let yp = y[i as usize] as f64 - ybar;
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

/// C `lsFit2`.
pub fn ls_fit2(
    x1: &[f32],
    x2: &[f32],
    y: &[f32],
    n: i32,
    a: &mut f32,
    b: &mut f32,
    c: Option<&mut f32>,
) {
    let (mut ypred, mut prederr) = (0., 0.);
    ls_fit2_pred(x1, x2, y, n, a, b, c, 0., 0., &mut ypred, &mut prederr)
}

/// C `lsFit2Pred`.
pub fn ls_fit2_pred(
    x1: &[f32],
    x2: &[f32],
    y: &[f32],
    n: i32,
    a: &mut f32,
    b: &mut f32,
    mut c: Option<&mut f32>,
    x1pred: f32,
    x2pred: f32,
    ypred: &mut f32,
    prederr: &mut f32,
) {
    let (mut x1s, mut x2s, mut ys) = (0., 0., 0.);
    for i in 0..n {
        x1s += x1[i as usize] as f64;
        x2s += x2[i as usize] as f64;
        ys += y[i as usize] as f64;
    }
    let (x1m, x2m, ym) = (x1s / n as f64, x2s / n as f64, ys / n as f64);
    let (mut x1sqs, mut x2sqs, mut x1x2s, mut x1ys, mut x2ys, mut ysqs) = (0., 0., 0., 0., 0., 0.);
    for i in 0..n {
        let (mut p1, mut p2, mut py) = (
            x1[i as usize] as f64,
            x2[i as usize] as f64,
            y[i as usize] as f64,
        );
        if c.is_some() {
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
    if let Some(cval) = c.as_deref_mut() {
        *cval = ym as f32;
    }
    *ypred = ym as f32;
    *prederr = 0.;
    let mut absanum = if anum > 0. { anum } else { -anum };
    let absbnum = if bnum > 0. { bnum } else { -bnum };
    let absdenom = if denom > 0. { denom } else { -denom };
    if absanum < absbnum {
        absanum = absbnum;
    }
    if absdenom < 1.0e-30 * absanum {
        return;
    }
    let dbla = anum / denom;
    let dblb = bnum / denom;
    *a = dbla as f32;
    *b = dblb as f32;
    let mut dblc = 0.;
    if let Some(cval) = c.as_deref_mut() {
        dblc = ym - dbla * x1m - dblb * x2m;
        *cval = dblc as f32;
    }
    *ypred = (dbla * x1pred as f64 + dblb * x2pred as f64 + dblc) as f32;
    let c11 = x2sqs / denom;
    let c22 = x1sqs / denom;
    let c12 = -x1x2s / denom;
    let devss = ysqs - dbla * x1ys - dblb * x2ys;
    // `predsq` is a double, but the source casts the whole expression to
    // `(float)` before storing it (`simplestat.c:611`).
    let predsq = (1.
        + 1. / n as f64
        + c11 * (x1pred as f64 - x1m) * (x1pred as f64 - x1m)
        + c22 * (x2pred as f64 - x2m) * (x2pred as f64 - x2m)
        + 2. * c12 * (x1pred as f64 - x1m) * (x2pred as f64 - x2m)) as f32 as f64;
    if n < 4 || predsq < 0. || devss < 0. {
        return;
    }
    *prederr = ((devss / (n as f64 - 3.)) * predsq).sqrt() as f32;
}

/// C `lsFit3`.
pub fn ls_fit3(
    x1: &[f32],
    x2: &[f32],
    x3: &[f32],
    y: &[f32],
    n: i32,
    a1: &mut f32,
    a2: &mut f32,
    a3: &mut f32,
    c: &mut f32,
) {
    let (mut x1s, mut x2s, mut x3s, mut ys) = (0_f32, 0., 0., 0.);
    for i in 0..n {
        x1s += x1[i as usize];
        x2s += x2[i as usize];
        x3s += x3[i as usize];
        ys += y[i as usize];
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
        let p1 = x1[i as usize] - x1m;
        let p2 = x2[i as usize] - x2m;
        let p3 = x3[i as usize] - x3m;
        let py = y[i as usize] - ym;
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
    let den = (q1 * q2 * q3 - q1 * q23 * q23 + q12 * q23 * q13 - q12 * q12 * q3 + q13 * q12 * q23
        - q13 * q2 * q13) as f64;
    let num1 = (r1 * q2 * q3 - r1 * q23 * q23 + q12 * q23 * r3 - q12 * r2 * q3 + q13 * r2 * q23
        - q13 * q2 * r3) as f64;
    let num2 = (q1 * r2 * q3 - q1 * q23 * r3 + r1 * q23 * q13 - r1 * q12 * q3 + q13 * q12 * r3
        - q13 * r2 * q13) as f64;
    let num3 = (q1 * q2 * r3 - q1 * r2 * q23 + q12 * r2 * q13 - q12 * q12 * r3 + r1 * q12 * q23
        - r1 * q2 * q13) as f64;
    let mut maxnum = if num1.abs() > num2.abs() {
        num1.abs()
    } else {
        num2.abs()
    };
    maxnum = if maxnum > num3.abs() {
        maxnum
    } else {
        num3.abs()
    };
    if den.abs() < 1.0e-30 * maxnum {
        return;
    }
    *a1 = (num1 / den) as f32;
    *a2 = (num2 / den) as f32;
    *a3 = (num3 / den) as f32;
    *c = ym - *a1 * x1m - *a2 * x2m - *a3 * x3m;
}

/// C `eigenSort`.
pub fn eigen_sort(
    val: &mut [f64],
    vec: &mut [f64],
    n: i32,
    row_stride: i32,
    col_stride: i32,
    use_abs: i32,
) {
    for i in 0..n - 1 {
        let mut imax = i;
        for j in i + 1..n {
            if (use_abs != 0 && val[j as usize].abs() > val[imax as usize].abs())
                || (use_abs == 0 && val[j as usize] >= val[imax as usize])
            {
                imax = j;
            }
        }
        if imax != i {
            val.swap(i as usize, imax as usize);
            for k in 0..n {
                let first = (k * row_stride + i * col_stride) as usize;
                let second = (k * row_stride + imax * col_stride) as usize;
                vec.swap(first, second);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summaries_follow_source() {
        let a = [1_f32, 2., 3., 4.];
        let (mut av, mut sd, mut sem) = (0., 0., 0.);
        avg_sd(&a, 4, &mut av, &mut sd, &mut sem);
        assert_eq!(av, 2.5);
        assert!((sd - 1.2909944).abs() < 1.0e-5);
        let (mut lo, mut hi, mut mean) = (0., 0., 0.);
        array_min_max_mean(&a, 2, 2, 0, 1, 0, 1, &mut lo, &mut hi, &mut mean);
        assert_eq!((lo, hi, mean), (1., 4., 2.5));
    }
    #[test]
    fn regressions_and_eigensort_follow_source() {
        let x = [0_f32, 1., 2., 3.];
        let y = [1_f32, 3., 5., 7.];
        let (mut slope, mut b, mut ro) = (0., 0., 0.);
        ls_fit(&x, &y, 4, &mut slope, &mut b, &mut ro);
        assert!((slope - 2.).abs() < 1.0e-5);
        assert!((b - 1.).abs() < 1.0e-5);
        let x2 = [0_f32, 1., 0., 1., 2.];
        let x3 = [0_f32, 0., 1., 1., 2.];
        let yy = [1_f32, 3., 4., 6., 11.];
        let (mut aa, mut bb, mut cc) = (0., 0., 0.);
        ls_fit2(&x2, &x3, &yy, 5, &mut aa, &mut bb, Some(&mut cc));
        assert!((aa - 2.).abs() < 1.0e-4);
        assert!((bb - 3.).abs() < 1.0e-4);
        assert!((cc - 1.).abs() < 1.0e-4);
        let mut val = [2_f64, -5., 3.];
        let mut vec = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        eigen_sort(&mut val, &mut vec, 3, 1, 3, 1);
        assert_eq!(val, [-5., 3., 2.]);
    }
}
