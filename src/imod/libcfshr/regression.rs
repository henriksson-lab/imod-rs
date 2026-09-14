//! Translation of `IMOD/libcfshr/regression.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use std::io::Write;

use super::gaussj::gaussj;
use super::robuststat::{rs_madn, rs_median, rs_sort_floats};

/// Original `statMatrices` (`regression.c:66`).
#[allow(clippy::too_many_arguments)]
pub fn stat_matrices(
    x: &[f32],
    xsize: i32,
    col_fast: i32,
    m: i32,
    msize: i32,
    ndata: i32,
    sx: &mut [f32],
    ss: &mut [f32],
    ssd: &mut [f32],
    d: &mut [f32],
    r: &mut [f32],
    xm: &mut [f32],
    sd: &mut [f32],
    ifdisp: i32,
) {
    let col_stride = if col_fast != 0 { 1 } else { xsize };
    let row_stride = if col_fast != 0 { xsize } else { 1 };
    let fndata = ndata as f32;
    for i in 0..m {
        sx[i as usize] = 0.;
        for j in 0..m {
            ssd[(msize * i + j) as usize] = 0.;
            r[(msize * i + j) as usize] = 0.;
        }
    }
    if ifdisp >= 0 {
        for i in 0..m {
            for k in 0..ndata {
                sx[i as usize] += x[(k * row_stride + i * col_stride) as usize];
            }
            xm[i as usize] = sx[i as usize] / fndata;
        }
    } else {
        let mut wsum = 0.;
        for k in 0..ndata {
            wsum += x[(k * row_stride + m * col_stride) as usize];
        }
        for i in 0..m {
            for k in 0..ndata {
                sx[i as usize] += x[(k * row_stride + i * col_stride) as usize]
                    * x[(k * row_stride + m * col_stride) as usize];
            }
            xm[i as usize] = sx[i as usize] / wsum;
        }
    }
    for k in 0..ndata {
        let weight = if ifdisp < 0 {
            x[(k * row_stride + m * col_stride) as usize]
        } else {
            1.
        };
        for i in 0..m {
            for j in 0..m {
                ssd[(i * msize + j) as usize] += (x[(k * row_stride + i * col_stride) as usize]
                    - xm[i as usize])
                    * (x[(k * row_stride + j * col_stride) as usize] - xm[j as usize])
                    * weight;
            }
        }
    }
    for i in 0..m {
        // `(float)sqrt((double)(ssd[i * msize + i] / (fndata - 1.)))`.
        sd[i as usize] =
            ((ssd[(i * msize + i) as usize] as f64 / (fndata as f64 - 1.)).sqrt()) as f32;
        for j in 0..m {
            ss[(i * msize + j) as usize] =
                ssd[(i * msize + j) as usize] + sx[i as usize] * sx[j as usize] / fndata;
            ss[(j * msize + i) as usize] = ss[(i * msize + j) as usize];
            ssd[(j * msize + i) as usize] = ssd[(i * msize + j) as usize];
        }
    }
    if ifdisp == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..m {
            // `ssd[...] / (fndata - 1.)` is a double quotient because of
            // the double literal, rounded to float on store.
            d[(i * msize + j) as usize] =
                (ssd[(i * msize + j) as usize] as f64 / (fndata as f64 - 1.)) as f32;
            d[(j * msize + i) as usize] = d[(i * msize + j) as usize];
            let den = sd[i as usize] * sd[j as usize];
            r[(i * msize + j) as usize] = if den > 1.0e-30 {
                d[(i * msize + j) as usize] / den
            } else {
                1.
            };
            r[(j * msize + i) as usize] = r[(i * msize + j) as usize];
        }
    }
}

/// Original `statmatrices` (`regression.c:137`).
#[allow(clippy::too_many_arguments)]
pub fn statmatrices(
    x: &[f32],
    xsize: &i32,
    col_fast: &i32,
    m: &i32,
    msize: &i32,
    ndata: &i32,
    sx: &mut [f32],
    ss: &mut [f32],
    ssd: &mut [f32],
    d: &mut [f32],
    r: &mut [f32],
    xm: &mut [f32],
    sd: &mut [f32],
    ifdisp: &i32,
) {
    stat_matrices(
        x, *xsize, *col_fast, *m, *msize, *ndata, sx, ss, ssd, d, r, xm, sd, *ifdisp,
    )
}

/// Original `multRegress` (`regression.c:183`).
#[allow(clippy::too_many_arguments)]
pub fn mult_regress(
    x: &[f32],
    x_size: i32,
    col_fast: i32,
    num_inp_col: i32,
    num_data: i32,
    num_out_col: i32,
    wgt_col: i32,
    sol: &mut [f32],
    sol_size: i32,
    mut cons: Option<&mut [f32]>,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
) -> i32 {
    let col_stride = if col_fast != 0 { 1 } else { x_size };
    let row_stride = if col_fast != 0 { x_size } else { 1 };
    let mp = num_inp_col + num_out_col;
    let fndata = num_data as f32;
    if wgt_col > 0 && (wgt_col < mp || (col_fast == 0 && wgt_col >= x_size)) {
        return 1;
    }
    if wgt_col <= 0 {
        for i in 0..mp {
            let mut sum = 0f64;
            for k in 0..num_data {
                sum += x[(k * row_stride + i * col_stride) as usize] as f64;
            }
            x_mean[i as usize] = (sum / fndata as f64) as f32;
        }
    } else {
        let mut wsum = 0f64;
        for k in 0..num_data {
            wsum += x[(k * row_stride + wgt_col * col_stride) as usize] as f64;
        }
        for i in 0..mp {
            let mut sum = 0f64;
            for k in 0..num_data {
                sum += (x[(k * row_stride + i * col_stride) as usize]
                    * x[(k * row_stride + wgt_col * col_stride) as usize])
                    as f64;
            }
            x_mean[i as usize] = (sum / wsum) as f32;
        }
    }
    for i in 0..mp {
        for j in i..mp {
            if i >= num_inp_col && i != j {
                continue;
            }
            let mut sum = 0f64;
            for k in 0..num_data {
                let a = x[(k * row_stride + i * col_stride) as usize];
                let b = x[(k * row_stride + j * col_stride) as usize];
                // Every product here is single precision in the source --
                // `dsum` is the only double -- so the whole term is formed
                // in `f32` and only then widened.
                if cons.is_some() {
                    sum += if wgt_col <= 0 {
                        ((a - x_mean[i as usize]) * (b - x_mean[j as usize])) as f64
                    } else {
                        ((a - x_mean[i as usize])
                            * (b - x_mean[j as usize])
                            * x[(k * row_stride + wgt_col * col_stride) as usize])
                            as f64
                    };
                } else {
                    sum += if wgt_col <= 0 {
                        (a * b) as f64
                    } else {
                        (a * b * x[(k * row_stride + wgt_col * col_stride) as usize]) as f64
                    };
                }
            }
            work[(j * mp + i) as usize] = sum as f32;
        }
    }
    for i in 0..mp {
        // `(float)sqrt((double)(work[i * mp + i] / (fndata - 1.)))`.
        x_sd[i as usize] =
            ((work[(i * mp + i) as usize] as f64 / (fndata as f64 - 1.)).sqrt()) as f32;
    }
    if num_inp_col == 0 {
        if let Some(cons) = cons.as_deref_mut() {
            for j in 0..num_out_col {
                cons[j as usize] = x_mean[j as usize];
            }
        }
        return 0;
    }
    for i in 0..num_inp_col {
        for j in i..mp {
            let den = x_sd[i as usize] * x_sd[j as usize];
            if den < 1.0e-30 {
                work[(j * mp + i) as usize] = 1.;
            } else {
                // `den * (fndata - 1.)` is a double product, so the
                // division is done in double.
                work[(j * mp + i) as usize] = (work[(j * mp + i) as usize] as f64
                    / (den as f64 * (fndata as f64 - 1.)))
                    as f32;
            }
            if j < num_inp_col {
                work[(i * mp + j) as usize] = work[(j * mp + i) as usize];
            }
        }
    }
    for j in 0..num_out_col {
        for i in 0..num_inp_col {
            sol[(j + i * num_out_col) as usize] = work[((j + num_inp_col) * mp + i) as usize];
        }
    }
    // `work` is the mp-by-mp matrix and `gaussj` addresses the
    // num_inp_col-by-num_inp_col leading block of it; `sol` is addressed as
    // num_inp_col rows of num_out_col columns.
    if gaussj(work, num_inp_col, mp, sol, num_out_col, num_out_col) != 0 {
        return 3;
    }
    work[..(num_inp_col * num_out_col) as usize]
        .copy_from_slice(&sol[..(num_inp_col * num_out_col) as usize]);
    for j in 0..num_out_col {
        if let Some(cons) = cons.as_deref_mut() {
            cons[j as usize] = x_mean[(num_inp_col + j) as usize];
        }
        for i in 0..num_inp_col {
            sol[(i + sol_size * j) as usize] = if x_sd[i as usize] < 1.0e-30 {
                0.
            } else {
                work[(j + i * num_out_col) as usize] * x_sd[(num_inp_col + j) as usize]
                    / x_sd[i as usize]
            };
            if let Some(cons) = cons.as_deref_mut() {
                cons[j as usize] -= sol[(i + sol_size * j) as usize] * x_mean[i as usize];
            }
        }
    }
    0
}

/// Original `multregress` (`regression.c:317`).
#[allow(clippy::too_many_arguments)]
pub fn multregress(
    x: &[f32],
    x_size: &i32,
    col_fast: &i32,
    num_inp_col: &i32,
    num_data: &i32,
    num_out_col: &i32,
    wgt_col: &i32,
    sol: &mut [f32],
    sol_size: &i32,
    cons: Option<&mut [f32]>,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
) -> i32 {
    mult_regress(
        x,
        *x_size,
        *col_fast,
        *num_inp_col,
        *num_data,
        *num_out_col,
        *wgt_col - 1,
        sol,
        *sol_size,
        cons,
        x_mean,
        x_sd,
        work,
    )
}
/// Original `multregressnoc` (`regression.c:326`).
#[allow(clippy::too_many_arguments)]
pub fn multregressnoc(
    x: &[f32],
    x_size: &i32,
    col_fast: &i32,
    num_inp_col: &i32,
    num_data: &i32,
    num_out_col: &i32,
    wgt_col: &i32,
    sol: &mut [f32],
    sol_size: &i32,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
) -> i32 {
    mult_regress(
        x,
        *x_size,
        *col_fast,
        *num_inp_col,
        *num_data,
        *num_out_col,
        *wgt_col - 1,
        sol,
        *sol_size,
        None,
        x_mean,
        x_sd,
        work,
    )
}

/// Original `polynomialFit` (`regression.c:365`).
#[allow(clippy::too_many_arguments)]
pub fn polynomial_fit(
    x: &[f32],
    y: &[f32],
    ndata: i32,
    order: i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
) -> i32 {
    let wdim = order + 1;
    if order == 0 {
        return 1;
    }
    for i in 0..ndata {
        for j in 0..order {
            work[(i + j * ndata) as usize] = (x[i as usize] as f64).powf(j as f64 + 1.) as f32;
        }
        work[(i + order * ndata) as usize] = y[i as usize];
    }
    // `xMean = work + wdim * ndata; xSD = xMean + wdim; mwork = xSD + wdim`
    // (`regression.c:369-371`) -- three disjoint windows into `work`, which
    // is also the data matrix `multRegress` reads.
    let (data, rest) = work.split_at_mut((wdim * ndata) as usize);
    let (x_mean, rest) = rest.split_at_mut(wdim as usize);
    let (x_sd, mwork) = rest.split_at_mut(wdim as usize);
    mult_regress(
        data,
        ndata,
        0,
        order,
        ndata,
        1,
        0,
        slopes,
        ndata,
        Some(intcpt),
        x_mean,
        x_sd,
        mwork,
    )
}
/// Original `polynomialfit` (`regression.c:385`).
#[allow(clippy::too_many_arguments)]
pub fn polynomialfit(
    x: &[f32],
    y: &[f32],
    ndata: &i32,
    order: &i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
) -> i32 {
    polynomial_fit(x, y, *ndata, *order, slopes, intcpt, work)
}
/// Original `weightedPolyFit` (`regression.c:401`).
#[allow(clippy::too_many_arguments)]
pub fn weighted_poly_fit(
    x: &[f32],
    y: &[f32],
    weight: &[f32],
    ndata: i32,
    order: i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
) -> i32 {
    let wdim = order + 1;
    if order == 0 {
        return 1;
    }
    for i in 0..ndata {
        for j in 0..order {
            work[(i + j * ndata) as usize] = (x[i as usize] as f64).powf(j as f64 + 1.) as f32;
        }
        work[(i + order * ndata) as usize] = y[i as usize];
        work[(i + (order + 1) * ndata) as usize] = weight[i as usize];
    }
    // `xMean = work + (order + 2) * ndata; xSD = xMean + wdim;
    //  mwork = xSD + wdim` (`regression.c:405-407`).
    let (data, rest) = work.split_at_mut(((order + 2) * ndata) as usize);
    let (x_mean, rest) = rest.split_at_mut(wdim as usize);
    let (x_sd, mwork) = rest.split_at_mut(wdim as usize);
    mult_regress(
        data,
        ndata,
        0,
        order,
        ndata,
        1,
        order + 1,
        slopes,
        ndata,
        Some(intcpt),
        x_mean,
        x_sd,
        mwork,
    )
}
/// Original `weightedpolyfit` (`regression.c:422`).
#[allow(clippy::too_many_arguments)]
pub fn weightedpolyfit(
    x: &[f32],
    y: &[f32],
    weight: &[f32],
    ndata: &i32,
    order: &i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
) -> i32 {
    weighted_poly_fit(x, y, weight, *ndata, *order, slopes, intcpt, work)
}

/// Original `robustRegress` (`regression.c:504`).
#[allow(clippy::too_many_arguments)]
pub fn robust_regress(
    x: &mut [f32],
    x_size: i32,
    col_fast: i32,
    num_inp_col: i32,
    num_data: i32,
    num_out_col: i32,
    sol: &mut [f32],
    sol_size: i32,
    mut cons: Option<&mut [f32]>,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
    mut kfactor: f32,
    num_iter: &mut i32,
    mut max_iter: i32,
    max_zero_wgt: i32,
    max_change: f32,
    max_oscill: f32,
) -> i32 {
    let wgt_col = num_inp_col + num_out_col;
    let prev_col = wgt_col + 1;
    let col_stride = if col_fast != 0 { 1 } else { x_size };
    let row_stride = if col_fast != 0 { x_size } else { 1 };
    let mut split_last_time = 0;
    let mut keep_criterion = 0;
    let mut special_weights = 0;
    let min_non_zero_wgt: f32 = 0.02;
    let mut report = 0;
    if kfactor < 1. {
        kfactor = -kfactor;
        special_weights = 1;
    }
    if max_iter < 0 {
        report = 1;
        max_iter = -max_iter;
    }
    if col_fast != 0 && prev_col >= x_size {
        return 1;
    }
    for j in 0..num_data {
        x[(j * row_stride + wgt_col * col_stride) as usize] = 1.;
        x[(j * row_stride + prev_col * col_stride) as usize] = 1.;
    }
    if special_weights != 0 {
        // `B3DNINT` is `(int)floor(a + 0.5)`, not `round()`.
        let num1 = (work[0] as f64 + 0.5).floor() as i32;
        if num1 < 1 || num1 > num_data / 2 {
            return 2;
        }
        for i in 1..=num1 {
            let j = (work[i as usize] as f64 + 0.5).floor() as i32;
            if j < 0 || j >= num_data {
                return 2;
            }
            x[(j * row_stride + wgt_col * col_stride) as usize] = 0.;
            x[(j * row_stride + prev_col * col_stride) as usize] = 0.;
        }
    }
    let mut iter = 0;
    let mut rms_err = 0.;
    let mut wgt_rms_err = 0.;
    let mut criterion = 0.;
    while iter < max_iter {
        let ierr = mult_regress(
            x,
            x_size,
            col_fast,
            num_inp_col,
            num_data,
            num_out_col,
            wgt_col,
            sol,
            sol_size,
            cons.as_deref_mut(),
            x_mean,
            x_sd,
            work,
        );
        if ierr != 0 {
            return ierr;
        }
        rms_err = 0.;
        wgt_rms_err = 0.;
        for j in 0..num_data {
            if num_out_col == 1 {
                let mut colres = if let Some(c) = cons.as_deref() {
                    c[0]
                } else {
                    0.
                } - x[(j * row_stride + num_inp_col * col_stride) as usize];
                for i in 0..num_inp_col {
                    colres += x[(j * row_stride + i * col_stride) as usize] * sol[i as usize];
                }
                work[j as usize] = colres;
            } else {
                let mut ressum = 0.;
                for k in 0..num_out_col {
                    let mut colres = if let Some(c) = cons.as_deref() {
                        c[k as usize]
                    } else {
                        0.
                    } - x
                        [(j * row_stride + (num_inp_col + k) * col_stride) as usize];
                    for i in 0..num_inp_col {
                        colres += x[(j * row_stride + i * col_stride) as usize]
                            * sol[(i + k * sol_size) as usize];
                    }
                    ressum += colres * colres;
                }
                work[j as usize] = ressum.sqrt();
            }
            rms_err += work[j as usize] * work[j as usize];
            wgt_rms_err += work[j as usize]
                * work[j as usize]
                * x[(j * row_stride + wgt_col * col_stride) as usize];
        }
        rms_err = ((rms_err / num_data as f32) as f64).sqrt() as f32;
        wgt_rms_err = ((wgt_rms_err / num_data as f32) as f64).sqrt() as f32;
        let mut median = 0.;
        let mut madn = 0.;
        // `rsMedian(work, numData, &work[numData], &median)`: the sorting
        // scratch is the second half of the same `work` array.
        {
            let (res, sortbuf) = work.split_at_mut(num_data as usize);
            rs_median(res, num_data, sortbuf, &mut median);
            rs_madn(res, num_data, median, sortbuf, &mut madn);
        }
        if keep_criterion == 0 {
            criterion = kfactor * madn;
        }
        let mut num_out = 0;
        let mut num1 = 0;
        let mut num2 = 0;
        let mut num5 = 0;
        let mut diffsum = 0.;
        for j in 0..num_data {
            work[j as usize] -= median;
            if num_out_col > 1 {
                if work[j as usize] < 0. {
                    work[j as usize] = 0.;
                }
            } else if work[j as usize] < 0. {
                work[j as usize] = -work[j as usize];
            }
            if work[j as usize] > criterion {
                num_out += 1;
            }
        }
        if num_out > max_zero_wgt {
            let old_criterion = criterion;
            for j in 0..num_data {
                work[(num_data + j) as usize] = work[j as usize];
            }
            rs_sort_floats(&mut work[num_data as usize..], num_data);
            if max_zero_wgt > 0 {
                criterion = (work[(2 * num_data - max_zero_wgt) as usize]
                    + work[(2 * num_data - max_zero_wgt - 1) as usize])
                    / 2.;
            } else {
                // `work[2 * numData - 1] / sqrt(1. - sqrt(minNonZeroWgt))`
                // -- both square roots and the subtraction are done in
                // double, and so is the division.
                criterion = (work[(2 * num_data - 1) as usize] as f64
                    / (1. - (min_non_zero_wgt as f64).sqrt()).sqrt())
                    as f32;
            }
            if report != 0 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%d with zero weight, revising criterion from %g to %g\n",
                        &[
                            CArg::Int(num_out as i64),
                            CArg::Dbl(old_criterion as f64),
                            CArg::Dbl(criterion as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
            keep_criterion = 1;
        }
        num_out = 0;
        let mut diffmax = 0.;
        let mut prevmax = 0.;
        for j in 0..num_data {
            // `numOut` counts only the `work[j] > criterion` arm, not
            // every weight that happens to come out zero.
            let weight = if work[j as usize] > criterion {
                num_out += 1;
                0.
            } else if work[j as usize] <= 1.0e-6 * criterion {
                1.
            } else {
                let dev = work[j as usize] / criterion;
                (1. - dev * dev) * (1. - dev * dev)
            };
            let diff = (weight - x[(j * row_stride + wgt_col * col_stride) as usize]).abs();
            diffsum += diff;
            // `B3DMAX(diffmax, diff)` is `diffmax > diff ? diffmax : diff`,
            // which takes the second operand when the comparison is false.
            diffmax = if diffmax > diff { diffmax } else { diff };
            let prev = (weight - x[(j * row_stride + prev_col * col_stride) as usize]).abs();
            prevmax = if prevmax > prev { prevmax } else { prev };
            x[(j * row_stride + prev_col * col_stride) as usize] =
                x[(j * row_stride + wgt_col * col_stride) as usize];
            x[(j * row_stride + wgt_col * col_stride) as usize] = weight;
            if report != 0 {
                if weight > 0. && weight <= 0.1 {
                    num1 += 1;
                }
                if weight > 0.1 && weight <= 0.2 {
                    num2 += 1;
                }
                if weight < 0.5 {
                    num5 += 1;
                }
            }
        }
        if report != 0 {
            let mut out = ImodFile::Stdout;
            let _ = out.write_all(
                c_format(
                    "Iter %3d del mean %.4f max %.4f prev %.4f # 0, 0.1, 0.2, <0.5: %d %d %d \
                     %d\n",
                    &[
                        CArg::Int(iter as i64),
                        CArg::Dbl((diffsum / num_data as f32) as f64),
                        CArg::Dbl(diffmax as f64),
                        CArg::Dbl(prevmax as f64),
                        CArg::Int(num_out as i64),
                        CArg::Int(num1 as i64),
                        CArg::Int(num2 as i64),
                        CArg::Int(num5 as i64),
                    ],
                )
                .as_bytes(),
            );
            let _ = out.flush();
        }
        if split_last_time == 0
            && (diffmax < max_change || (diffmax < max_oscill && prevmax < max_change))
        {
            break;
        }
        if split_last_time == 0 && prevmax < max_change / 2. {
            split_last_time = 1;
            if report != 0 {
                let mut out = ImodFile::Stdout;
                let _ = out.write_all(
                    c_format(
                        "        Oscillation detected: averaging previous weights\n",
                        &[],
                    )
                    .as_bytes(),
                );
                let _ = out.flush();
            }
            for j in 0..num_data {
                let p = (j * row_stride + prev_col * col_stride) as usize;
                let w = (j * row_stride + wgt_col * col_stride) as usize;
                x[w] = (x[w] + x[p]) / 2.;
            }
        } else if split_last_time == 1 {
            split_last_time = 2;
        } else {
            split_last_time = 0;
        }
        iter += 1;
    }
    *num_iter = iter;
    work[0] = rms_err;
    work[1] = wgt_rms_err;
    if *num_iter < max_iter { 0 } else { -1 }
}

/// Original `robustregress` (`regression.c:695`).
#[allow(clippy::too_many_arguments)]
pub fn robustregress(
    x: &mut [f32],
    x_size: &i32,
    col_fast: &i32,
    num_inp_col: &i32,
    num_data: &i32,
    num_out_col: &i32,
    sol: &mut [f32],
    sol_size: &i32,
    cons: Option<&mut [f32]>,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
    kfactor: &f32,
    num_iter: &mut i32,
    max_iter: &i32,
    max_zero_wgt: &i32,
    max_change: &f32,
    max_oscillate: &f32,
) -> i32 {
    robust_regress(
        x,
        *x_size,
        *col_fast,
        *num_inp_col,
        *num_data,
        *num_out_col,
        sol,
        *sol_size,
        cons,
        x_mean,
        x_sd,
        work,
        *kfactor,
        num_iter,
        *max_iter,
        *max_zero_wgt,
        *max_change,
        *max_oscillate,
    )
}
/// Original `robustregressnoc` (`regression.c:706`).
#[allow(clippy::too_many_arguments)]
pub fn robustregressnoc(
    x: &mut [f32],
    x_size: &i32,
    col_fast: &i32,
    num_inp_col: &i32,
    num_data: &i32,
    num_out_col: &i32,
    sol: &mut [f32],
    sol_size: &i32,
    x_mean: &mut [f32],
    x_sd: &mut [f32],
    work: &mut [f32],
    kfactor: &f32,
    num_iter: &mut i32,
    max_iter: &i32,
    max_zero_wgt: &i32,
    max_change: &f32,
    max_oscillate: &f32,
) -> i32 {
    robust_regress(
        x,
        *x_size,
        *col_fast,
        *num_inp_col,
        *num_data,
        *num_out_col,
        sol,
        *sol_size,
        None,
        x_mean,
        x_sd,
        work,
        *kfactor,
        num_iter,
        *max_iter,
        *max_zero_wgt,
        *max_change,
        *max_oscillate,
    )
}

/// Original `robustPolyFit` (`regression.c:730`).
#[allow(clippy::too_many_arguments)]
pub fn robust_poly_fit(
    x: &[f32],
    y: &[f32],
    ndata: i32,
    order: i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
    kfactor: f32,
    num_iter: &mut i32,
    max_iter: i32,
    max_zero_wgt: i32,
) -> i32 {
    let max_change: f32 = 0.02;
    let max_oscill: f32 = 0.05;
    let wdim = order + 1;
    if order == 0 {
        return 1;
    }
    for i in 0..ndata {
        for j in 0..order {
            work[(i + j * ndata) as usize] = (x[i as usize] as f64).powf(j as f64 + 1.) as f32;
        }
        work[(i + order * ndata) as usize] = y[i as usize];
    }
    // `xMean = work + (order + 3) * ndata; xSD = xMean + wdim;
    //  mwork = xSD + wdim` (`regression.c:735-737`).
    let err = {
        let (data, rest) = work.split_at_mut(((order + 3) * ndata) as usize);
        let (x_mean, rest) = rest.split_at_mut(wdim as usize);
        let (x_sd, mwork) = rest.split_at_mut(wdim as usize);
        robust_regress(
            data,
            ndata,
            0,
            order,
            ndata,
            1,
            slopes,
            order,
            Some(intcpt),
            x_mean,
            x_sd,
            mwork,
            kfactor,
            num_iter,
            max_iter,
            max_zero_wgt,
            max_change,
            max_oscill,
        )
    };
    if err != 0 {
        return err;
    }
    for i in 0..ndata {
        work[i as usize] = work[(i + (order + 1) * ndata) as usize];
    }
    0
}
/// Original `robustpolyfit` (`regression.c:757`).
#[allow(clippy::too_many_arguments)]
pub fn robustpolyfit(
    x: &[f32],
    y: &[f32],
    ndata: &i32,
    order: &i32,
    slopes: &mut [f32],
    intcpt: &mut [f32],
    work: &mut [f32],
    kfactor: &f32,
    num_iter: &mut i32,
    max_iter: &i32,
    max_zero_wgt: &i32,
) -> i32 {
    robust_poly_fit(
        x,
        y,
        *ndata,
        *order,
        slopes,
        intcpt,
        work,
        *kfactor,
        num_iter,
        *max_iter,
        *max_zero_wgt,
    )
}

/// Original `robustPolySmooth` (`regression.c:779`).
#[allow(clippy::too_many_arguments)]
pub fn robust_poly_smooth(
    x: &[f32],
    y_in: &[f32],
    ndata: i32,
    order: i32,
    y_out: &mut [f32],
    num_fit: i32,
    min_fit: i32,
    work: &mut [f32],
    kfactor: f32,
    max_iter: i32,
    max_zero_wgt: i32,
    mut weights: Option<&mut [f32]>,
) -> i32 {
    let ordp1 = order + 1;
    if ndata <= 0 {
        y_out[0] = (ordp1
            + 2 * num_fit
            + (order + 3) * num_fit
            + 2 * ordp1
            + (ordp1 * ordp1).max(2 * num_fit))
        .max(ordp1 * (order + 3 + num_fit)) as f32;
        return 0;
    }
    if min_fit > ndata {
        return 1;
    }
    // `slopes = work; xfit = slopes + order; yfit = xfit + numFit;
    //  mwork = yfit + numFit` (`regression.c:787-790`).
    let (slopes, rest) = work.split_at_mut(order as usize);
    let (xfit, rest) = rest.split_at_mut(num_fit as usize);
    let (yfit, mwork) = rest.split_at_mut(num_fit as usize);
    for ind in 0..ndata {
        let mut fit_start = ind - num_fit / 2;
        let mut fit_end = fit_start + num_fit - 1;
        if fit_end >= ndata {
            fit_end = ndata - 1;
            fit_start = fit_start.min(ndata - min_fit);
        }
        if fit_start < 0 {
            fit_start = 0;
            fit_end = fit_end.max(min_fit - 1);
        }
        let this_fit = fit_end + 1 - fit_start;
        let mut xmean = 0.;
        let mut ymean = 0.;
        for j in 0..this_fit {
            xfit[j as usize] = x[(fit_start + j) as usize];
            yfit[j as usize] = y_in[(fit_start + j) as usize];
            xmean += xfit[j as usize];
            ymean += yfit[j as usize];
        }
        xmean /= this_fit as f32;
        ymean /= this_fit as f32;
        for j in 0..this_fit {
            xfit[j as usize] -= xmean;
            yfit[j as usize] -= ymean;
        }
        let mut intcpt = [0.0f32];
        let mut num_iter = 0;
        let mut err = robust_poly_fit(
            xfit,
            yfit,
            this_fit,
            order,
            slopes,
            &mut intcpt,
            mwork,
            kfactor,
            &mut num_iter,
            max_iter,
            max_zero_wgt,
        );
        if let Some(weights) = weights.as_deref_mut() {
            weights[ind as usize] = mwork[(ind - fit_start) as usize];
        }
        if err != 0 {
            // The source writes `weights[ind] = 1.` here without the NULL
            // check it made one line above (`regression.c:833`), so a NULL
            // `weights` with a failing fit is a null dereference there.
            if let Some(weights) = weights.as_deref_mut() {
                weights[ind as usize] = 1.;
            }
            err = polynomial_fit(xfit, yfit, this_fit, order, slopes, &mut intcpt, mwork);
        }
        if err != 0 {
            return err;
        }
        y_out[ind as usize] = intcpt[0] + ymean;
        for j in 0..order {
            // `slopes[j] * pow((double)(x[ind] - xmean), j + 1.)` is a
            // double product accumulated into the float `yOut[ind]`.
            y_out[ind as usize] = (y_out[ind as usize] as f64
                + slopes[j as usize] as f64
                    * ((x[ind as usize] - xmean) as f64).powf(j as f64 + 1.))
                as f32;
        }
    }
    0
}
/// Original `robustpolysmooth` (`regression.c:846`).
#[allow(clippy::too_many_arguments)]
pub fn robustpolysmooth(
    x: &[f32],
    y_in: &[f32],
    ndata: &i32,
    order: &i32,
    y_out: &mut [f32],
    num_fit: &i32,
    min_fit: &i32,
    work: &mut [f32],
    kfactor: &f32,
    max_iter: &i32,
    max_zero_wgt: &i32,
    weights: Option<&mut [f32]>,
) -> i32 {
    robust_poly_smooth(
        x,
        y_in,
        *ndata,
        *order,
        y_out,
        *num_fit,
        *min_fit,
        work,
        *kfactor,
        *max_iter,
        *max_zero_wgt,
        weights,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polynomial_fit_recovers_quadratic() {
        let x = [-2., -1., 0., 1., 2.];
        let y = [3., 0., 1., 6., 15.];
        let mut slopes = [0.; 2];
        let mut intcpt = [0.0f32];
        let mut work = [0.; 64];
        assert_eq!(
            polynomial_fit(&x, &y, 5, 2, &mut slopes, &mut intcpt, &mut work),
            0
        );
        assert!((intcpt[0] - 1.).abs() < 1.0e-5);
        assert!((slopes[0] - 3.).abs() < 1.0e-5);
        assert!((slopes[1] - 2.).abs() < 1.0e-5);
    }
    #[test]
    fn stat_matrices_column_major() {
        let x = [1., 2., 3., 2., 4., 6.];
        let mut sx = [0.; 2];
        let mut ss = [0.; 4];
        let mut ssd = [0.; 4];
        let mut d = [0.; 4];
        let mut r = [0.; 4];
        let mut xm = [0.; 2];
        let mut sd = [0.; 2];
        stat_matrices(
            &x, 3, 0, 2, 2, 3, &mut sx, &mut ss, &mut ssd, &mut d, &mut r, &mut xm, &mut sd, 1,
        );
        assert_eq!(xm, [2., 4.]);
        assert!((r[1] - 1.).abs() < 1.0e-6);
    }

    #[test]
    fn robust_regression_rejects_a_large_outlier() {
        {
            // Four columns (input, output, current and previous weights), each with ten rows.
            let mut data = [
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 3., 5., 7., 9., 11., 13., 15., 17.,
                100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0.,
            ];
            let mut sol = [0.; 1];
            let mut cons = [0.; 1];
            let mut mean = [0.; 2];
            let mut sd = [0.; 2];
            let mut work = [0.; 64];
            let mut iterations = 0;
            assert_eq!(
                robust_regress(
                    &mut data,
                    10,
                    0,
                    1,
                    10,
                    1,
                    &mut sol,
                    1,
                    Some(&mut cons),
                    &mut mean,
                    &mut sd,
                    &mut work,
                    4.68,
                    &mut iterations,
                    100,
                    1,
                    0.05,
                    0.05
                ),
                0
            );
            assert!(
                (sol[0] - 2.).abs() < 0.01,
                "slope {}, intercept {}, iterations {}, weight {}",
                sol[0],
                cons[0],
                iterations,
                data[29]
            );
            assert!((cons[0] - 1.).abs() < 0.02);
            assert_eq!(data[29], 0.);
        }
    }
}
