// Translated from IMOD libcfshr/robuststat.c and libcfshr/simplestat.c
// Original author: David Mastronarde, University of Colorado

// ---------------------------------------------------------------------------
// Simple statistics (from simplestat.c)
// ---------------------------------------------------------------------------

/// Calculates the mean, standard deviation, and standard error of the mean
/// from the values in `x`.
///
/// Returns `(avg, sd, sem)`.
pub fn avg_sd(x: &[f32]) -> (f32, f32, f32) {
    let n = x.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let sx: f64 = x.iter().map(|&v| v as f64).sum();
    let avg = (sx / n as f64) as f32;
    let mut sx_dev = 0.0_f64;
    let mut sxsq_dev = 0.0_f64;
    for &v in x {
        let d = v as f64 - avg as f64;
        sx_dev += d;
        sxsq_dev += d * d;
    }
    let (avg_corr, sd) = sums_to_avg_sd(sx_dev as f32, sxsq_dev as f32, n);
    let avg = avg + avg_corr;
    let sem = if n > 0 {
        sd / (n as f32).sqrt()
    } else {
        0.0
    };
    (avg, sd, sem)
}

/// Computes a mean and standard deviation from the sum of values `sx`,
/// sum of squares `sxsq`, and number of values `n`.
/// It will not generate any division-by-zero errors.
///
/// Returns `(avg, sd)`.
pub fn sums_to_avg_sd(sx: f32, sxsq: f32, n: usize) -> (f32, f32) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let avg = sx / n as f32;
    let sd = if n > 1 {
        let den = (sxsq as f64 - n as f64 * avg as f64 * avg as f64) / (n as f64 - 1.0);
        if den > 0.0 {
            den.sqrt() as f32
        } else {
            0.0
        }
    } else {
        0.0
    };
    (avg, sd)
}

/// Computes a mean and standard deviation from the sum of values `sx8`,
/// sum of squares `sxsq8`, and number of values `n1 * n2`,
/// where the number of values can be greater than 2^31.
///
/// Returns `(avg, sd)` as `f64`.
pub fn sums_to_avg_sd_all_dbl(sx8: f64, sxsq8: f64, n1: i32, n2: i32) -> (f64, f64) {
    let dn = n1 as f64 * n2 as f64;
    if dn <= 0.0 {
        return (0.0, 0.0);
    }
    let avg = sx8 / dn;
    let sd = if dn > 1.0 {
        let den = (sxsq8 - dn * avg * avg) / (dn - 1.0);
        if den > 0.0 {
            den.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };
    (avg, sd)
}

/// Like [`sums_to_avg_sd_all_dbl`] but returns `(avg, sd)` as `f32`.
pub fn sums_to_avg_sd_dbl(sx8: f64, sxsq8: f64, n1: i32, n2: i32) -> (f32, f32) {
    let (avg, sd) = sums_to_avg_sd_all_dbl(sx8, sxsq8, n1, n2);
    (avg as f32, sd as f32)
}

/// Computes the minimum, maximum, and mean from data in `array`
/// (row-major, `nx` columns), for X indices `ix0..=ix1` and Y indices `iy0..=iy1`
/// (zero-based, inclusive).
///
/// Returns `(dmin, dmax, dmean)`.
pub fn array_min_max_mean(
    array: &[f32],
    nx: usize,
    ix0: usize,
    ix1: usize,
    iy0: usize,
    iy1: usize,
) -> (f32, f32, f32) {
    let mut dmin = f32::MAX;
    let mut dmax = f32::MIN;
    let mut sum_dbl = 0.0_f64;
    for iy in iy0..=iy1 {
        let mut sum_tmp = 0.0_f32;
        for ix in ix0..=ix1 {
            let den = array[iy * nx + ix];
            sum_tmp += den;
            if den < dmin {
                dmin = den;
            }
            if den > dmax {
                dmax = den;
            }
        }
        sum_dbl += sum_tmp as f64;
    }
    let dmean = (sum_dbl / ((ix1 + 1 - ix0) as f64 * (iy1 + 1 - iy0) as f64)) as f32;
    (dmin, dmax, dmean)
}

/// Result of [`array_min_max_mean_sd`].
pub struct ArrayMinMaxMeanSd {
    pub dmin: f32,
    pub dmax: f32,
    pub avg: f32,
    pub sd: f32,
    pub sum: f64,
    pub sum_sq: f64,
}

/// Computes the minimum, maximum, mean, and standard deviation from data in `array`
/// (row-major, `nx` columns), for X indices `ix0..=ix1` and Y indices `iy0..=iy1`
/// (zero-based, inclusive).  It also returns the sum and sum of squares.
///
/// It makes a rough estimate of the image mean and sums the deviations from that
/// estimate, then computes the SD from that sum and the actual mean.  This gives an
/// accurate SD value when the SD is much smaller than the mean.
pub fn array_min_max_mean_sd(
    array: &[f32],
    nx: usize,
    ny: usize,
    ix0: usize,
    ix1: usize,
    iy0: usize,
    iy1: usize,
) -> ArrayMinMaxMeanSd {
    let nx_area = ix1 + 1 - ix0;
    let ny_area = iy1 + 1 - iy0;
    let mut dmin = 1.0e37_f32;
    let mut dmax = -1.0e37_f32;

    // Rough mean from a grid of sample points
    let x_div = 2.max(8.min(nx.saturating_sub(2)));
    let y_div = 2.max(8.min(ny.saturating_sub(2)));
    let mut rough_mean = 0.0_f64;
    let mut nsum = 0usize;
    for jy in 1..y_div {
        let iy = iy0 + (jy * ny_area) / y_div;
        for jx in 1..x_div {
            let ix = ix0 + (jx * nx_area) / x_div;
            rough_mean += array[iy * nx + ix] as f64;
            nsum += 1;
        }
    }
    rough_mean /= nsum as f64;

    let mut sum_dbl = 0.0_f64;
    let mut sum_sq_dbl = 0.0_f64;
    for iy in iy0..=iy1 {
        let mut sum_tmp = 0.0_f64;
        let mut sum_tmp_sq = 0.0_f64;
        for ix in ix0..=ix1 {
            let den = array[iy * nx + ix];
            if den < dmin {
                dmin = den;
            }
            if den > dmax {
                dmax = den;
            }
            let dev = den as f64 - rough_mean;
            sum_tmp += dev;
            sum_tmp_sq += dev * dev;
        }
        sum_dbl += sum_tmp;
        sum_sq_dbl += sum_tmp_sq;
    }

    let (avg8, sd8) = sums_to_avg_sd_all_dbl(sum_dbl, sum_sq_dbl, nx_area as i32, ny_area as i32);
    let avg8 = avg8 + rough_mean;
    let n_total = nx_area as f64 * ny_area as f64;
    let sum_dbl = n_total * avg8;
    let sum_sq_dbl = (n_total - 1.0) * sd8 * sd8 + sum_dbl * avg8;

    ArrayMinMaxMeanSd {
        dmin,
        dmax,
        avg: avg8 as f32,
        sd: sd8 as f32,
        sum: sum_dbl,
        sum_sq: sum_sq_dbl,
    }
}

/// Scales densities in a subarea of `array` appropriately for the given `mode`
/// (0, 1, 6, or 2), and returns `(dmin, dmax, dmean)`.
/// Values will range from 0 to 255 for mode 0/2, 0 to 32767 for mode 1,
/// or 0 to 65530 for mode 6.
///
/// `nx_dim` is the X dimension of the array and the subarea is defined by
/// `nx1..=nx2`, `ny1..=ny2` (zero-based, inclusive).
pub fn scale_array_for_mode(
    array: &mut [f32],
    nx_dim: usize,
    mode: i32,
    nx1: usize,
    nx2: usize,
    ny1: usize,
    ny2: usize,
) -> (f32, f32, f32) {
    let out_max: f32 = match mode {
        6 => 65530.0,
        1 => 32767.0,
        _ => 255.0,
    };

    let mut dmin_in = 1.0e30_f32;
    let mut dmax_in = -1.0e30_f32;
    for iy in ny1..=ny2 {
        for ix in nx1..=nx2 {
            let val = array[iy * nx_dim + ix];
            if val < dmin_in {
                dmin_in = val;
            }
            if val > dmax_in {
                dmax_in = val;
            }
        }
    }

    let sclfac = 0.99999 * out_max / (dmax_in - dmin_in);
    let mut dmean = 0.0_f32;
    for iy in ny1..=ny2 {
        let mut tsum = 0.0_f32;
        for ix in nx1..=nx2 {
            let val = sclfac * (array[iy * nx_dim + ix] - dmin_in);
            tsum += val;
            array[iy * nx_dim + ix] = val;
        }
        dmean += tsum;
    }
    dmean /= (nx2 - nx1 + 1) as f32 * (ny2 - ny1 + 1) as f32;
    (0.0, out_max, dmean)
}

/// Result of [`ls_fit_pred`].
pub struct LsFitPredResult {
    pub slope: f32,
    pub intercept: f32,
    pub ro: f32,
    pub sa: f32,
    pub sb: f32,
    pub se: f32,
    pub ypred: f32,
    pub prederr: f32,
}

/// Fits a straight line to the points in `x` and `y` by the method of least squares,
/// returning slope, intercept, and correlation coefficient.
///
/// Returns `(slope, intercept, ro)`.
pub fn ls_fit(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let r = ls_fit_pred(x, y, 0.0);
    (r.slope, r.intercept, r.ro)
}

/// Fits a straight line to the points in `x` and `y` by the method of least squares.
///
/// Returns slope, intercept, correlation coefficient, standard errors of the
/// estimate (`se`), slope (`sb`), intercept (`sa`), and for the X value `xpred`,
/// the predicted value and its standard error.
pub fn ls_fit_pred(x: &[f32], y: &[f32], xpred: f32) -> LsFitPredResult {
    let n = x.len();
    let mut result = LsFitPredResult {
        slope: 1.0,
        intercept: 0.0,
        ro: 0.0,
        sa: 0.0,
        sb: 0.0,
        se: 0.0,
        ypred: 0.0,
        prederr: 0.0,
    };
    if n < 2 {
        return result;
    }
    let nf = n as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for i in 0..n {
        sx += x[i] as f64;
        sy += y[i] as f64;
    }
    let xbar = sx / nf;
    let ybar = sy / nf;
    let mut sxpsq = 0.0_f64;
    let mut sxyp = 0.0_f64;
    let mut sypsq = 0.0_f64;
    for i in 0..n {
        let xp = x[i] as f64 - xbar;
        let yp = y[i] as f64 - ybar;
        sxpsq += xp * xp;
        sypsq += yp * yp;
        sxyp += xp * yp;
    }
    let d = nf * sxpsq;
    let dslope = sxyp / sxpsq;
    result.slope = dslope as f32;
    let dbint = (ybar * sxpsq - xbar * sxyp) / sxpsq;
    result.intercept = dbint as f32;
    let roden = (sxpsq * sypsq).sqrt();
    result.ro = 1.0;
    if roden != 0.0 && sxyp.abs() <= roden.abs() {
        result.ro = (sxyp / roden) as f32;
    }
    let sxy = sxyp + nf * xbar * ybar;
    let sysq = sypsq + nf * ybar * ybar;
    let setmp = sysq - dbint * sy - dslope * sxy;
    if n > 2 && setmp > 0.0 {
        result.se = (setmp / (n as f64 - 2.0)).sqrt() as f32;
    }
    result.sa = (result.se as f64 * (1.0 / nf + (sx * sx / nf) / d).sqrt()) as f32;
    result.sb = (result.se as f64 / (d / nf).sqrt()) as f32;

    result.ypred = (dslope * xpred as f64 + dbint) as f32;
    result.prederr = (result.se as f64
        * (1.0 + 1.0 / nf + nf * (xpred as f64 - xbar) * (xpred as f64 - xbar) / d).sqrt())
        as f32;
    result
}

/// Result of [`ls_fit2_pred`].
pub struct LsFit2PredResult {
    pub a: f32,
    pub b: f32,
    pub c: Option<f32>,
    pub ypred: f32,
    pub prederr: f32,
}

/// Does a linear regression fit of `y` to `x1` and `x2`:
///   `y = a * x1 + b * x2 + c`
///
/// If `with_intercept` is true, fits with intercept `c`; otherwise fits
/// `y = a * x1 + b * x2`.
///
/// Returns coefficients `a`, `b`, optional intercept `c`.
pub fn ls_fit2(x1: &[f32], x2: &[f32], y: &[f32], with_intercept: bool) -> LsFit2PredResult {
    ls_fit2_pred(x1, x2, y, with_intercept, 0.0, 0.0)
}

/// Does a linear regression fit of `y` to `x1` and `x2`:
///   `y = a * x1 + b * x2 + c`
///
/// If `with_intercept` is true, fits with intercept `c`; otherwise fits
/// `y = a * x1 + b * x2`.
///
/// For one value of x1 and x2 given by `x1pred` and `x2pred`, returns
/// the predicted value and its standard error.
pub fn ls_fit2_pred(
    x1: &[f32],
    x2: &[f32],
    y: &[f32],
    with_intercept: bool,
    x1pred: f32,
    x2pred: f32,
) -> LsFit2PredResult {
    let n = x1.len();
    let nf = n as f64;

    let mut x1s = 0.0_f64;
    let mut x2s = 0.0_f64;
    let mut ys = 0.0_f64;
    for i in 0..n {
        x1s += x1[i] as f64;
        x2s += x2[i] as f64;
        ys += y[i] as f64;
    }
    let x1m = x1s / nf;
    let x2m = x2s / nf;
    let ym = ys / nf;

    let mut x1sqs = 0.0_f64;
    let mut x2sqs = 0.0_f64;
    let mut x1x2s = 0.0_f64;
    let mut x1ys = 0.0_f64;
    let mut x2ys = 0.0_f64;
    let mut ysqs = 0.0_f64;
    for i in 0..n {
        let (x1p, x2p, yp) = if with_intercept {
            (
                x1[i] as f64 - x1m,
                x2[i] as f64 - x2m,
                y[i] as f64 - ym,
            )
        } else {
            (x1[i] as f64, x2[i] as f64, y[i] as f64)
        };
        x1sqs += x1p * x1p;
        x2sqs += x2p * x2p;
        x1ys += x1p * yp;
        x2ys += x2p * yp;
        x1x2s += x1p * x2p;
        ysqs += yp * yp;
    }

    let denom = x1sqs * x2sqs - x1x2s * x1x2s;
    let anum = x1ys * x2sqs - x1x2s * x2ys;
    let bnum = x1sqs * x2ys - x1ys * x1x2s;

    let mut result = LsFit2PredResult {
        a: 0.0,
        b: 0.0,
        c: if with_intercept { Some(ym as f32) } else { None },
        ypred: ym as f32,
        prederr: 0.0,
    };

    let abs_max_num = anum.abs().max(bnum.abs());
    if denom.abs() < 1.0e-30 * abs_max_num {
        return result;
    }

    let dbla = anum / denom;
    let dblb = bnum / denom;
    result.a = dbla as f32;
    result.b = dblb as f32;
    let dblc = if with_intercept {
        let c = ym - dbla * x1m - dblb * x2m;
        result.c = Some(c as f32);
        c
    } else {
        0.0
    };
    result.ypred = (dbla * x1pred as f64 + dblb * x2pred as f64 + dblc) as f32;

    let c11 = x2sqs / denom;
    let c22 = x1sqs / denom;
    let c12 = -x1x2s / denom;
    let devss = ysqs - dbla * x1ys - dblb * x2ys;
    let predsq = 1.0
        + 1.0 / nf
        + c11 * (x1pred as f64 - x1m) * (x1pred as f64 - x1m)
        + c22 * (x2pred as f64 - x2m) * (x2pred as f64 - x2m)
        + 2.0 * c12 * (x1pred as f64 - x1m) * (x2pred as f64 - x2m);
    if n >= 4 && predsq >= 0.0 && devss >= 0.0 {
        result.prederr = ((devss / (n as f64 - 3.0)) * predsq).sqrt() as f32;
    }
    result
}

/// Does a linear regression fit of `y` to `x1`, `x2`, and `x3`:
///   `y = a1 * x1 + a2 * x2 + a3 * x3 + c`
///
/// Returns `(a1, a2, a3, c)`.
pub fn ls_fit3(x1: &[f32], x2: &[f32], x3: &[f32], y: &[f32]) -> (f32, f32, f32, f32) {
    let n = x1.len();
    let nf = n as f32;

    let mut x1s = 0.0_f32;
    let mut x2s = 0.0_f32;
    let mut x3s = 0.0_f32;
    let mut ys = 0.0_f32;
    for i in 0..n {
        x1s += x1[i];
        x2s += x2[i];
        x3s += x3[i];
        ys += y[i];
    }
    let x1m = x1s / nf;
    let x2m = x2s / nf;
    let x3m = x3s / nf;
    let ym = ys / nf;

    let mut x1sqs = 0.0_f32;
    let mut x2sqs = 0.0_f32;
    let mut x3sqs = 0.0_f32;
    let mut x1x2s = 0.0_f32;
    let mut x1x3s = 0.0_f32;
    let mut x2x3s = 0.0_f32;
    let mut x1ys = 0.0_f32;
    let mut x2ys = 0.0_f32;
    let mut x3ys = 0.0_f32;
    for i in 0..n {
        let x1p = x1[i] - x1m;
        let x2p = x2[i] - x2m;
        let x3p = x3[i] - x3m;
        let yp = y[i] - ym;
        x1sqs += x1p * x1p;
        x2sqs += x2p * x2p;
        x3sqs += x3p * x3p;
        x1ys += x1p * yp;
        x2ys += x2p * yp;
        x3ys += x3p * yp;
        x1x2s += x1p * x2p;
        x1x3s += x1p * x3p;
        x2x3s += x2p * x3p;
    }

    let den = determ3(
        x1sqs as f64, x1x2s as f64, x1x3s as f64,
        x1x2s as f64, x2sqs as f64, x2x3s as f64,
        x1x3s as f64, x2x3s as f64, x3sqs as f64,
    );
    let num1 = determ3(
        x1ys as f64, x1x2s as f64, x1x3s as f64,
        x2ys as f64, x2sqs as f64, x2x3s as f64,
        x3ys as f64, x2x3s as f64, x3sqs as f64,
    );
    let num2 = determ3(
        x1sqs as f64, x1ys as f64, x1x3s as f64,
        x1x2s as f64, x2ys as f64, x2x3s as f64,
        x1x3s as f64, x3ys as f64, x3sqs as f64,
    );
    let num3 = determ3(
        x1sqs as f64, x1x2s as f64, x1ys as f64,
        x1x2s as f64, x2sqs as f64, x2ys as f64,
        x1x3s as f64, x2x3s as f64, x3ys as f64,
    );

    let max_num = num1.abs().max(num2.abs()).max(num3.abs());
    if den.abs() < 1.0e-30 * max_num {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let a1 = (num1 / den) as f32;
    let a2 = (num2 / den) as f32;
    let a3 = (num3 / den) as f32;
    let c = ym - a1 * x1m - a2 * x2m - a3 * x3m;
    (a1, a2, a3, c)
}

/// Computes the determinant of a 3x3 matrix given by rows.
fn determ3(
    a11: f64, a12: f64, a13: f64,
    a21: f64, a22: f64, a23: f64,
    a31: f64, a32: f64, a33: f64,
) -> f64 {
    a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
}

/// Sorts eigenvalues in `val` into descending order and rearranges their eigenvectors
/// in `vec` so that they still correspond.  `n` is the number of dimensions,
/// `row_stride` is the index step between successive elements of an eigenvector,
/// and `col_stride` is the index step between successive eigenvectors.
/// Set `use_abs` to true to sort on the absolute value of the eigenvalues.
pub fn eigen_sort(
    val: &mut [f64],
    vec: &mut [f64],
    n: usize,
    row_stride: usize,
    col_stride: usize,
    use_abs: bool,
) {
    for i in 0..n.saturating_sub(1) {
        let mut imax = i;
        for j in (i + 1)..n {
            let greater = if use_abs {
                val[j].abs() > val[imax].abs()
            } else {
                val[j] >= val[imax]
            };
            if greater {
                imax = j;
            }
        }
        if imax != i {
            val.swap(i, imax);
            for k in 0..n {
                let idx_i = k * row_stride + i * col_stride;
                let idx_max = k * row_stride + imax * col_stride;
                vec.swap(idx_i, idx_max);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Robust statistics (from robuststat.c)
// ---------------------------------------------------------------------------

/// Sorts a mutable slice of `f32` in ascending order.
pub fn sort_floats(x: &mut [f32]) {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Sorts a mutable slice of `i32` in ascending order.
pub fn sort_ints(x: &mut [i32]) {
    x.sort();
}

/// Sorts indices into `x` so that the corresponding float values are in ascending order.
/// The indices in `index` refer directly into `x`.
pub fn sort_indexed_floats(x: &[f32], index: &mut [usize]) {
    index.sort_by(|&a, &b| {
        x[a]
            .partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Computes the median of the values in `x`, which must already be sorted.
pub fn median_of_sorted(x: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 != 0 {
        x[n / 2]
    } else {
        0.5 * (x[n / 2 - 1] + x[n / 2])
    }
}

/// Computes the median of the values in `x`.
/// Returns `(median, sorted_copy)`.
pub fn median(x: &[f32]) -> (f32, Vec<f32>) {
    let mut xsort = x.to_vec();
    sort_floats(&mut xsort);
    let med = median_of_sorted(&xsort);
    (med, xsort)
}

/// Computes the median of the values in `x` in linear time.
/// The input slice is rearranged (partially sorted).
pub fn fast_median_in_place(x: &mut [f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mut med = percentile_float((n + 1) / 2, x);
    if n % 2 == 0 {
        med = (med + percentile_float(n / 2 + 1, x)) / 2.0;
    }
    med
}

/// Computes the median of the values in `x` in linear time,
/// without modifying the input. Returns the median.
pub fn fast_median(x: &[f32]) -> f32 {
    let mut tmp = x.to_vec();
    fast_median_in_place(&mut tmp)
}

/// Computes the percentile of the sorted values in `x` indicated by `fraction`,
/// with interpolation between adjacent values.
pub fn percentile_of_sorted(x: &[f32], fraction: f32) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let real_ind = n as f32 * fraction - 0.5;
    let lower_ind = real_ind.floor() as i32;
    let f = real_ind - lower_ind as f32;
    if lower_ind < 0 {
        x[0]
    } else if lower_ind >= n as i32 - 1 {
        x[n - 1]
    } else {
        let li = lower_ind as usize;
        (1.0 - f) * x[li] + f * x[li + 1]
    }
}

/// Computes the normalized median absolute deviation from the median (MADN) for
/// the values in `x`, using the already-computed `median`.
///
/// The median absolute deviation is divided by 0.6745, which makes it estimate
/// sigma for a normal distribution.
///
/// Returns `(madn, sorted_absolute_deviations)`.
pub fn madn(x: &[f32], median: f32) -> (f32, Vec<f32>) {
    let mut tmp: Vec<f32> = x.iter().map(|&v| (v - median).abs()).collect();
    sort_floats(&mut tmp);
    let mad = median_of_sorted(&tmp);
    (mad / 0.6745, tmp)
}

/// Computes the normalized median absolute deviation from the median (MADN) for
/// the values in `x` in linear time, using the already-computed `median`.
///
/// The median absolute deviation is divided by 0.6745, which makes it estimate
/// sigma for a normal distribution.
pub fn fast_madn(x: &[f32], median: f32) -> f32 {
    let mut tmp: Vec<f32> = x.iter().map(|&v| (v - median).abs()).collect();
    let mad = fast_median_in_place(&mut tmp);
    mad / 0.6745
}

/// Selects outliers among the values in `x` by testing whether the absolute
/// deviation from the median is greater than the normalized median absolute
/// deviation by the criterion `kcrit`.
///
/// Returns a `Vec<f32>` with -1.0 or 1.0 for outliers in the negative or positive
/// direction from the median, 0.0 otherwise.
///
/// A typical value for `kcrit` is 2.24 (the square root of the 0.975 quantile of
/// a chi-square distribution with one degree of freedom).
pub fn mad_median_outliers(x: &[f32], kcrit: f32) -> Vec<f32> {
    let med = fast_median(x);
    let madn_val = fast_madn(x, med);
    x.iter()
        .map(|&v| {
            if madn_val != 0.0 && (v - med).abs() / madn_val > kcrit {
                if v > med {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            }
        })
        .collect()
}

/// Computes a trimmed mean of the already-sorted values in `x`, trimming off
/// the fraction `gamma` on each end of the distribution.
pub fn trimmed_mean_of_sorted(x: &[f32], gamma: f32) -> f32 {
    let n = x.len();
    let cut = (gamma * n as f32) as usize;
    let mut sum = 0.0_f64;
    for i in cut..(n - cut) {
        sum += x[i] as f64;
    }
    if sum != 0.0 {
        (sum / (n - 2 * cut) as f64) as f32
    } else {
        0.0
    }
}

/// Computes a trimmed mean of the values in `x`, trimming off the fraction
/// `gamma` on each end of the distribution.
///
/// Returns `(trimmed_mean, sorted_copy)`.
pub fn trimmed_mean(x: &[f32], gamma: f32) -> (f32, Vec<f32>) {
    let mut xsort = x.to_vec();
    sort_floats(&mut xsort);
    let tm = trimmed_mean_of_sorted(&xsort, gamma);
    (tm, xsort)
}

// ---------------------------------------------------------------------------
// Helper: linear-time k-th smallest (quickselect)
// ---------------------------------------------------------------------------

/// Returns the k-th smallest value (1-based) from `x`, partially rearranging it.
/// This is a quickselect algorithm equivalent to the C `percentileFloat`.
fn percentile_float(k: usize, x: &mut [f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    if k == 0 || k > x.len() {
        return x[0];
    }
    let target = k - 1; // convert to 0-based
    quickselect(x, target)
}

fn quickselect(arr: &mut [f32], k: usize) -> f32 {
    let mut lo = 0usize;
    let mut hi = arr.len() - 1;
    while lo < hi {
        let pivot = arr[(lo + hi) / 2];
        let mut i = lo;
        let mut j = hi;
        loop {
            while arr[i] < pivot {
                i += 1;
            }
            while arr[j] > pivot {
                j -= 1;
            }
            if i <= j {
                arr.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            if i > j {
                break;
            }
        }
        if j < k {
            lo = i;
        }
        if k < i {
            hi = if j > 0 { j } else { 0 };
        }
        if lo >= hi {
            break;
        }
    }
    arr[k]
}

// ---------------------------------------------------------------------------
// Re-export convenient aliases matching the old module's public API
// ---------------------------------------------------------------------------

/// Compute mean of a slice.
pub fn mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    (sum / data.len() as f64) as f32
}

/// Compute mean and standard deviation.
pub fn mean_sd(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n < 2 {
        return (mean(data), 0.0);
    }
    let n_f64 = n as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &x in data {
        let v = x as f64;
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n_f64;
    let variance = (sum_sq - sum * sum / n_f64) / (n_f64 - 1.0);
    (mean as f32, variance.max(0.0).sqrt() as f32)
}

/// Compute min, max, mean of a slice.
pub fn min_max_mean(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0_f64;
    for &x in data {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
    }
    (min, max, (sum / data.len() as f64) as f32)
}

/// Compute min, max, mean, and standard deviation.
pub fn min_max_mean_sd(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = data.len() as f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &x in data {
        let v = x as f64;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let sd = if data.len() > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    (min, max, mean as f32, sd as f32)
}

/// Robust statistics: median and normalized median absolute deviation (MADN).
/// MADN = MAD / 0.6745, which estimates the standard deviation for normal data.
pub fn robust_stat(data: &mut [f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    sort_floats(data);
    let med = median_of_sorted(data);
    let mut devs: Vec<f32> = data.iter().map(|&x| (x - med).abs()).collect();
    sort_floats(&mut devs);
    let mad = median_of_sorted(&devs);
    let madn_val = mad / 0.6745;
    (med, madn_val)
}

/// Sample mean and SD by reading every `sample_step`-th element.
/// Used for quick statistics on large images.
pub fn sample_mean_sd(data: &[f32], sample_step: usize) -> (f32, f32) {
    let step = sample_step.max(1);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0u64;
    let mut i = 0;
    while i < data.len() {
        let v = data[i] as f64;
        sum += v;
        sum_sq += v * v;
        count += 1;
        i += step;
    }
    if count < 2 {
        return (sum as f32, 0.0);
    }
    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq - sum * sum / n) / (n - 1.0);
    (mean as f32, variance.max(0.0).sqrt() as f32)
}

/// Linear regression: fit y = a + b*x.
/// Returns (intercept, slope, correlation_coefficient).
pub fn linear_regression(x: &[f32], y: &[f32]) -> Option<(f32, f32, f32)> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return None;
    }
    let n_f64 = n as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let xi = x[i] as f64;
        let yi = y[i] as f64;
        sx += xi;
        sy += yi;
        sxx += xi * xi;
        syy += yi * yi;
        sxy += xi * yi;
    }
    let denom = n_f64 * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return None;
    }
    let slope = (n_f64 * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n_f64;

    let var_x = sxx / n_f64 - (sx / n_f64).powi(2);
    let var_y = syy / n_f64 - (sy / n_f64).powi(2);
    let r = if var_x > 0.0 && var_y > 0.0 {
        let cov = sxy / n_f64 - (sx / n_f64) * (sy / n_f64);
        (cov / (var_x * var_y).sqrt()) as f32
    } else {
        0.0
    };

    Some((intercept as f32, slope as f32, r))
}
