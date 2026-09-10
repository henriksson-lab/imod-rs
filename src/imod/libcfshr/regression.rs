//! Translation of `IMOD/libcfshr/regression.c`.
#![allow(dead_code)]

use super::gaussj::gaussj;
use super::robuststat::{rs_madn, rs_median, rs_sort_floats};

/// Original `statMatrices` (`regression.c:66`).
pub unsafe fn stat_matrices(
    x: *mut f32,
    xsize: i32,
    col_fast: i32,
    m: i32,
    msize: i32,
    ndata: i32,
    sx: *mut f32,
    ss: *mut f32,
    ssd: *mut f32,
    d: *mut f32,
    r: *mut f32,
    xm: *mut f32,
    sd: *mut f32,
    ifdisp: i32,
) {
    unsafe {
        let col_stride = if col_fast != 0 { 1 } else { xsize };
        let row_stride = if col_fast != 0 { xsize } else { 1 };
        let fndata = ndata as f32;
        for i in 0..m {
            *sx.add(i as usize) = 0.;
            for j in 0..m {
                *ssd.add((msize * i + j) as usize) = 0.;
                *r.add((msize * i + j) as usize) = 0.;
            }
        }
        if ifdisp >= 0 {
            for i in 0..m {
                for k in 0..ndata {
                    *sx.add(i as usize) += *x.add((k * row_stride + i * col_stride) as usize);
                }
                *xm.add(i as usize) = *sx.add(i as usize) / fndata;
            }
        } else {
            let mut wsum = 0.;
            for k in 0..ndata {
                wsum += *x.add((k * row_stride + m * col_stride) as usize);
            }
            for i in 0..m {
                for k in 0..ndata {
                    *sx.add(i as usize) += *x.add((k * row_stride + i * col_stride) as usize)
                        * *x.add((k * row_stride + m * col_stride) as usize);
                }
                *xm.add(i as usize) = *sx.add(i as usize) / wsum;
            }
        }
        for k in 0..ndata {
            let weight = if ifdisp < 0 {
                *x.add((k * row_stride + m * col_stride) as usize)
            } else {
                1.
            };
            for i in 0..m {
                for j in 0..m {
                    *ssd.add((i * msize + j) as usize) +=
                        (*x.add((k * row_stride + i * col_stride) as usize) - *xm.add(i as usize))
                            * (*x.add((k * row_stride + j * col_stride) as usize)
                                - *xm.add(j as usize))
                            * weight;
                }
            }
        }
        for i in 0..m {
            *sd.add(i as usize) = (*ssd.add((i * msize + i) as usize) / (fndata - 1.)).sqrt();
            for j in 0..m {
                *ss.add((i * msize + j) as usize) = *ssd.add((i * msize + j) as usize)
                    + *sx.add(i as usize) * *sx.add(j as usize) / fndata;
                *ss.add((j * msize + i) as usize) = *ss.add((i * msize + j) as usize);
                *ssd.add((j * msize + i) as usize) = *ssd.add((i * msize + j) as usize);
            }
        }
        if ifdisp == 0 {
            return;
        }
        for i in 0..m {
            for j in 0..m {
                *d.add((i * msize + j) as usize) =
                    *ssd.add((i * msize + j) as usize) / (fndata - 1.);
                *d.add((j * msize + i) as usize) = *d.add((i * msize + j) as usize);
                let den = *sd.add(i as usize) * *sd.add(j as usize);
                *r.add((i * msize + j) as usize) = if den > 1.0e-30 {
                    *d.add((i * msize + j) as usize) / den
                } else {
                    1.
                };
                *r.add((j * msize + i) as usize) = *r.add((i * msize + j) as usize);
            }
        }
    }
}

/// Original `statmatrices` (`regression.c:137`).
pub unsafe fn statmatrices(
    x: *mut f32,
    xsize: *mut i32,
    col_fast: *mut i32,
    m: *mut i32,
    msize: *mut i32,
    ndata: *mut i32,
    sx: *mut f32,
    ss: *mut f32,
    ssd: *mut f32,
    d: *mut f32,
    r: *mut f32,
    xm: *mut f32,
    sd: *mut f32,
    ifdisp: *mut i32,
) {
    unsafe {
        stat_matrices(
            x, *xsize, *col_fast, *m, *msize, *ndata, sx, ss, ssd, d, r, xm, sd, *ifdisp,
        )
    }
}

/// Original `multRegress` (`regression.c:183`).
pub unsafe fn mult_regress(
    x: *mut f32,
    x_size: i32,
    col_fast: i32,
    num_inp_col: i32,
    num_data: i32,
    num_out_col: i32,
    wgt_col: i32,
    sol: *mut f32,
    sol_size: i32,
    cons: *mut f32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
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
                    sum += *x.add((k * row_stride + i * col_stride) as usize) as f64;
                }
                *x_mean.add(i as usize) = (sum / fndata as f64) as f32;
            }
        } else {
            let mut wsum = 0f64;
            for k in 0..num_data {
                wsum += *x.add((k * row_stride + wgt_col * col_stride) as usize) as f64;
            }
            for i in 0..mp {
                let mut sum = 0f64;
                for k in 0..num_data {
                    sum += (*x.add((k * row_stride + i * col_stride) as usize)
                        * *x.add((k * row_stride + wgt_col * col_stride) as usize))
                        as f64;
                }
                *x_mean.add(i as usize) = (sum / wsum) as f32;
            }
        }
        for i in 0..mp {
            for j in i..mp {
                if i >= num_inp_col && i != j {
                    continue;
                }
                let mut sum = 0f64;
                for k in 0..num_data {
                    let a = *x.add((k * row_stride + i * col_stride) as usize);
                    let b = *x.add((k * row_stride + j * col_stride) as usize);
                    if !cons.is_null() {
                        sum += if wgt_col <= 0 {
                            (a - *x_mean.add(i as usize)) as f64
                                * (b - *x_mean.add(j as usize)) as f64
                        } else {
                            (a - *x_mean.add(i as usize)) as f64
                                * (b - *x_mean.add(j as usize)) as f64
                                * *x.add((k * row_stride + wgt_col * col_stride) as usize) as f64
                        };
                    } else {
                        sum += if wgt_col <= 0 {
                            (a * b) as f64
                        } else {
                            (a * b * *x.add((k * row_stride + wgt_col * col_stride) as usize))
                                as f64
                        };
                    }
                }
                *work.add((j * mp + i) as usize) = sum as f32;
            }
        }
        for i in 0..mp {
            *x_sd.add(i as usize) = (*work.add((i * mp + i) as usize) / (fndata - 1.)).sqrt();
        }
        if num_inp_col == 0 {
            if !cons.is_null() {
                for j in 0..num_out_col {
                    *cons.add(j as usize) = *x_mean.add(j as usize);
                }
            }
            return 0;
        }
        for i in 0..num_inp_col {
            for j in i..mp {
                let den = *x_sd.add(i as usize) * *x_sd.add(j as usize);
                if den < 1.0e-30 {
                    *work.add((j * mp + i) as usize) = 1.;
                } else {
                    *work.add((j * mp + i) as usize) /= den * (fndata - 1.);
                }
                if j < num_inp_col {
                    *work.add((i * mp + j) as usize) = *work.add((j * mp + i) as usize);
                }
            }
        }
        for j in 0..num_out_col {
            for i in 0..num_inp_col {
                *sol.add((j + i * num_out_col) as usize) =
                    *work.add(((j + num_inp_col) * mp + i) as usize);
            }
        }
        if gaussj(work, num_inp_col, mp, sol, num_out_col, num_out_col) != 0 {
            return 3;
        }
        core::ptr::copy_nonoverlapping(sol, work, (num_inp_col * num_out_col) as usize);
        for j in 0..num_out_col {
            if !cons.is_null() {
                *cons.add(j as usize) = *x_mean.add((num_inp_col + j) as usize);
            }
            for i in 0..num_inp_col {
                *sol.add((i + sol_size * j) as usize) = if *x_sd.add(i as usize) < 1.0e-30 {
                    0.
                } else {
                    *work.add((j + i * num_out_col) as usize)
                        * *x_sd.add((num_inp_col + j) as usize)
                        / *x_sd.add(i as usize)
                };
                if !cons.is_null() {
                    *cons.add(j as usize) -=
                        *sol.add((i + sol_size * j) as usize) * *x_mean.add(i as usize);
                }
            }
        }
        0
    }
}

/// Original `multregress` (`regression.c:317`).
pub unsafe fn multregress(
    x: *mut f32,
    x_size: *mut i32,
    col_fast: *mut i32,
    num_inp_col: *mut i32,
    num_data: *mut i32,
    num_out_col: *mut i32,
    wgt_col: *mut i32,
    sol: *mut f32,
    sol_size: *mut i32,
    cons: *mut f32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
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
}
/// Original `multregressnoc` (`regression.c:326`).
pub unsafe fn multregressnoc(
    x: *mut f32,
    x_size: *mut i32,
    col_fast: *mut i32,
    num_inp_col: *mut i32,
    num_data: *mut i32,
    num_out_col: *mut i32,
    wgt_col: *mut i32,
    sol: *mut f32,
    sol_size: *mut i32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
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
            core::ptr::null_mut(),
            x_mean,
            x_sd,
            work,
        )
    }
}

/// Original `polynomialFit` (`regression.c:365`).
pub unsafe fn polynomial_fit(
    x: *mut f32,
    y: *mut f32,
    ndata: i32,
    order: i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
        let wdim = order + 1;
        let x_mean = work.add((wdim * ndata) as usize);
        let x_sd = x_mean.add(wdim as usize);
        let mwork = x_sd.add(wdim as usize);
        if order == 0 {
            return 1;
        }
        for i in 0..ndata {
            for j in 0..order {
                *work.add((i + j * ndata) as usize) =
                    (*x.add(i as usize) as f64).powf(j as f64 + 1.) as f32;
            }
            *work.add((i + order * ndata) as usize) = *y.add(i as usize);
        }
        mult_regress(
            work, ndata, 0, order, ndata, 1, 0, slopes, ndata, intcpt, x_mean, x_sd, mwork,
        )
    }
}
/// Original `polynomialfit` (`regression.c:385`).
pub unsafe fn polynomialfit(
    x: *mut f32,
    y: *mut f32,
    ndata: *mut i32,
    order: *mut i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe { polynomial_fit(x, y, *ndata, *order, slopes, intcpt, work) }
}
/// Original `weightedPolyFit` (`regression.c:401`).
pub unsafe fn weighted_poly_fit(
    x: *mut f32,
    y: *mut f32,
    weight: *mut f32,
    ndata: i32,
    order: i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe {
        let wdim = order + 1;
        let x_mean = work.add(((order + 2) * ndata) as usize);
        let x_sd = x_mean.add(wdim as usize);
        let mwork = x_sd.add(wdim as usize);
        if order == 0 {
            return 1;
        }
        for i in 0..ndata {
            for j in 0..order {
                *work.add((i + j * ndata) as usize) =
                    (*x.add(i as usize) as f64).powf(j as f64 + 1.) as f32;
            }
            *work.add((i + order * ndata) as usize) = *y.add(i as usize);
            *work.add((i + (order + 1) * ndata) as usize) = *weight.add(i as usize);
        }
        mult_regress(
            work,
            ndata,
            0,
            order,
            ndata,
            1,
            order + 1,
            slopes,
            ndata,
            intcpt,
            x_mean,
            x_sd,
            mwork,
        )
    }
}
/// Original `weightedpolyfit` (`regression.c:422`).
pub unsafe fn weightedpolyfit(
    x: *mut f32,
    y: *mut f32,
    weight: *mut f32,
    ndata: *mut i32,
    order: *mut i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
) -> i32 {
    unsafe { weighted_poly_fit(x, y, weight, *ndata, *order, slopes, intcpt, work) }
}

/// Original `robustRegress` (`regression.c:504`).
pub unsafe fn robust_regress(
    x: *mut f32,
    x_size: i32,
    col_fast: i32,
    num_inp_col: i32,
    num_data: i32,
    num_out_col: i32,
    sol: *mut f32,
    sol_size: i32,
    cons: *mut f32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
    mut kfactor: f32,
    num_iter: *mut i32,
    mut max_iter: i32,
    max_zero_wgt: i32,
    max_change: f32,
    max_oscill: f32,
) -> i32 {
    unsafe {
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
            *x.add((j * row_stride + wgt_col * col_stride) as usize) = 1.;
            *x.add((j * row_stride + prev_col * col_stride) as usize) = 1.;
        }
        if special_weights != 0 {
            let num1 = (*work).round() as i32;
            if num1 < 1 || num1 > num_data / 2 {
                return 2;
            }
            for i in 1..=num1 {
                let j = (*work.add(i as usize)).round() as i32;
                if j < 0 || j >= num_data {
                    return 2;
                }
                *x.add((j * row_stride + wgt_col * col_stride) as usize) = 0.;
                *x.add((j * row_stride + prev_col * col_stride) as usize) = 0.;
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
                cons,
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
                    let mut colres = if cons.is_null() { 0. } else { *cons }
                        - *x.add((j * row_stride + num_inp_col * col_stride) as usize);
                    for i in 0..num_inp_col {
                        colres += *x.add((j * row_stride + i * col_stride) as usize)
                            * *sol.add(i as usize);
                    }
                    *work.add(j as usize) = colres;
                } else {
                    let mut ressum = 0.;
                    for k in 0..num_out_col {
                        let mut colres = if cons.is_null() {
                            0.
                        } else {
                            *cons.add(k as usize)
                        } - *x
                            .add((j * row_stride + (num_inp_col + k) * col_stride) as usize);
                        for i in 0..num_inp_col {
                            colres += *x.add((j * row_stride + i * col_stride) as usize)
                                * *sol.add((i + k * sol_size) as usize);
                        }
                        ressum += colres * colres;
                    }
                    *work.add(j as usize) = ressum.sqrt();
                }
                rms_err += *work.add(j as usize) * *work.add(j as usize);
                wgt_rms_err += *work.add(j as usize)
                    * *work.add(j as usize)
                    * *x.add((j * row_stride + wgt_col * col_stride) as usize);
            }
            rms_err = (rms_err / num_data as f32).sqrt();
            wgt_rms_err = (wgt_rms_err / num_data as f32).sqrt();
            let mut median = 0.;
            let mut madn = 0.;
            rs_median(work, num_data, work.add(num_data as usize), &mut median);
            rs_madn(
                work,
                num_data,
                median,
                work.add(num_data as usize),
                &mut madn,
            );
            if keep_criterion == 0 {
                criterion = kfactor * madn;
            }
            let mut num_out = 0;
            let mut num1 = 0;
            let mut num2 = 0;
            let mut num5 = 0;
            let mut diffsum = 0.;
            for j in 0..num_data {
                *work.add(j as usize) -= median;
                if num_out_col > 1 {
                    if *work.add(j as usize) < 0. {
                        *work.add(j as usize) = 0.;
                    }
                } else if *work.add(j as usize) < 0. {
                    *work.add(j as usize) = -*work.add(j as usize);
                }
                if *work.add(j as usize) > criterion {
                    num_out += 1;
                }
            }
            if num_out > max_zero_wgt {
                let old_criterion = criterion;
                for j in 0..num_data {
                    *work.add((num_data + j) as usize) = *work.add(j as usize);
                }
                rs_sort_floats(work.add(num_data as usize), num_data);
                if max_zero_wgt > 0 {
                    criterion = (*work.add((2 * num_data - max_zero_wgt) as usize)
                        + *work.add((2 * num_data - max_zero_wgt - 1) as usize))
                        / 2.;
                } else {
                    criterion = *work.add((2 * num_data - 1) as usize)
                        / (1. - min_non_zero_wgt.sqrt()).sqrt();
                }
                if report != 0 {
                    print!(
                        "{} with zero weight, revising criterion from {} to {}\n",
                        num_out, old_criterion, criterion
                    );
                }
                keep_criterion = 1;
            }
            num_out = 0;
            let mut diffmax = 0.;
            let mut prevmax = 0.;
            for j in 0..num_data {
                let weight = if *work.add(j as usize) > criterion {
                    0.
                } else if *work.add(j as usize) <= 1.0e-6 * criterion {
                    1.
                } else {
                    let dev = *work.add(j as usize) / criterion;
                    (1. - dev * dev) * (1. - dev * dev)
                };
                if weight == 0. {
                    num_out += 1;
                }
                let diff =
                    (weight - *x.add((j * row_stride + wgt_col * col_stride) as usize)).abs();
                diffsum += diff;
                if diff > diffmax {
                    diffmax = diff;
                }
                let prev =
                    (weight - *x.add((j * row_stride + prev_col * col_stride) as usize)).abs();
                if prev > prevmax {
                    prevmax = prev;
                }
                *x.add((j * row_stride + prev_col * col_stride) as usize) =
                    *x.add((j * row_stride + wgt_col * col_stride) as usize);
                *x.add((j * row_stride + wgt_col * col_stride) as usize) = weight;
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
                print!(
                    "Iter {:3} del mean {:.4} max {:.4} prev {:.4} # 0, 0.1, 0.2, <0.5: {} {} {} {}\n",
                    iter,
                    diffsum / num_data as f32,
                    diffmax,
                    prevmax,
                    num_out,
                    num1,
                    num2,
                    num5
                );
            }
            if split_last_time == 0
                && (diffmax < max_change || (diffmax < max_oscill && prevmax < max_change))
            {
                break;
            }
            if split_last_time == 0 && prevmax < max_change / 2. {
                split_last_time = 1;
                if report != 0 {
                    print!("        Oscillation detected: averaging previous weights\n");
                }
                for j in 0..num_data {
                    let p = (j * row_stride + prev_col * col_stride) as usize;
                    let w = (j * row_stride + wgt_col * col_stride) as usize;
                    *x.add(w) = (*x.add(w) + *x.add(p)) / 2.;
                }
            } else if split_last_time == 1 {
                split_last_time = 2;
            } else {
                split_last_time = 0;
            }
            iter += 1;
        }
        *num_iter = iter;
        *work = rms_err;
        *work.add(1) = wgt_rms_err;
        if *num_iter < max_iter { 0 } else { -1 }
    }
}

/// Original `robustregress` (`regression.c:695`).
pub unsafe fn robustregress(
    x: *mut f32,
    x_size: *mut i32,
    col_fast: *mut i32,
    num_inp_col: *mut i32,
    num_data: *mut i32,
    num_out_col: *mut i32,
    sol: *mut f32,
    sol_size: *mut i32,
    cons: *mut f32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
    kfactor: *mut f32,
    num_iter: *mut i32,
    max_iter: *mut i32,
    max_zero_wgt: *mut i32,
    max_change: *mut f32,
    max_oscillate: *mut f32,
) -> i32 {
    unsafe {
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
}
/// Original `robustregressnoc` (`regression.c:706`).
pub unsafe fn robustregressnoc(
    x: *mut f32,
    x_size: *mut i32,
    col_fast: *mut i32,
    num_inp_col: *mut i32,
    num_data: *mut i32,
    num_out_col: *mut i32,
    sol: *mut f32,
    sol_size: *mut i32,
    x_mean: *mut f32,
    x_sd: *mut f32,
    work: *mut f32,
    kfactor: *mut f32,
    num_iter: *mut i32,
    max_iter: *mut i32,
    max_zero_wgt: *mut i32,
    max_change: *mut f32,
    max_oscillate: *mut f32,
) -> i32 {
    unsafe {
        robust_regress(
            x,
            *x_size,
            *col_fast,
            *num_inp_col,
            *num_data,
            *num_out_col,
            sol,
            *sol_size,
            core::ptr::null_mut(),
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
}

/// Original `robustPolyFit` (`regression.c:730`).
pub unsafe fn robust_poly_fit(
    x: *mut f32,
    y: *mut f32,
    ndata: i32,
    order: i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
    kfactor: f32,
    num_iter: *mut i32,
    max_iter: i32,
    max_zero_wgt: i32,
) -> i32 {
    unsafe {
        let wdim = order + 1;
        let x_mean = work.add(((order + 3) * ndata) as usize);
        let x_sd = x_mean.add(wdim as usize);
        let mwork = x_sd.add(wdim as usize);
        if order == 0 {
            return 1;
        }
        for i in 0..ndata {
            for j in 0..order {
                *work.add((i + j * ndata) as usize) =
                    (*x.add(i as usize) as f64).powf(j as f64 + 1.) as f32;
            }
            *work.add((i + order * ndata) as usize) = *y.add(i as usize);
        }
        let err = robust_regress(
            work,
            ndata,
            0,
            order,
            ndata,
            1,
            slopes,
            order,
            intcpt,
            x_mean,
            x_sd,
            mwork,
            kfactor,
            num_iter,
            max_iter,
            max_zero_wgt,
            0.02,
            0.05,
        );
        if err != 0 {
            return err;
        }
        for i in 0..ndata {
            *work.add(i as usize) = *work.add((i + (order + 1) * ndata) as usize);
        }
        0
    }
}
/// Original `robustpolyfit` (`regression.c:757`).
pub unsafe fn robustpolyfit(
    x: *mut f32,
    y: *mut f32,
    ndata: *mut i32,
    order: *mut i32,
    slopes: *mut f32,
    intcpt: *mut f32,
    work: *mut f32,
    kfactor: *mut f32,
    num_iter: *mut i32,
    max_iter: *mut i32,
    max_zero_wgt: *mut i32,
) -> i32 {
    unsafe {
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
}

/// Original `robustPolySmooth` (`regression.c:779`).
pub unsafe fn robust_poly_smooth(
    x: *mut f32,
    y_in: *mut f32,
    ndata: i32,
    order: i32,
    y_out: *mut f32,
    num_fit: i32,
    min_fit: i32,
    work: *mut f32,
    kfactor: f32,
    max_iter: i32,
    max_zero_wgt: i32,
    weights: *mut f32,
) -> i32 {
    unsafe {
        let ordp1 = order + 1;
        let slopes = work;
        let xfit = slopes.add(order as usize);
        let yfit = xfit.add(num_fit as usize);
        let mwork = yfit.add(num_fit as usize);
        if ndata <= 0 {
            *y_out = (ordp1
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
                *xfit.add(j as usize) = *x.add((fit_start + j) as usize);
                *yfit.add(j as usize) = *y_in.add((fit_start + j) as usize);
                xmean += *xfit.add(j as usize);
                ymean += *yfit.add(j as usize);
            }
            xmean /= this_fit as f32;
            ymean /= this_fit as f32;
            for j in 0..this_fit {
                *xfit.add(j as usize) -= xmean;
                *yfit.add(j as usize) -= ymean;
            }
            let mut intcpt = 0.;
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
            if !weights.is_null() {
                *weights.add(ind as usize) = *mwork.add((ind - fit_start) as usize);
            }
            if err != 0 {
                if !weights.is_null() {
                    *weights.add(ind as usize) = 1.;
                }
                err = polynomial_fit(xfit, yfit, this_fit, order, slopes, &mut intcpt, mwork);
            }
            if err != 0 {
                return err;
            }
            *y_out.add(ind as usize) = intcpt + ymean;
            for j in 0..order {
                *y_out.add(ind as usize) +=
                    *slopes.add(j as usize) * (*x.add(ind as usize) - xmean).powf(j as f32 + 1.);
            }
        }
        0
    }
}
/// Original `robustpolysmooth` (`regression.c:846`).
pub unsafe fn robustpolysmooth(
    x: *mut f32,
    y_in: *mut f32,
    ndata: *mut i32,
    order: *mut i32,
    y_out: *mut f32,
    num_fit: *mut i32,
    min_fit: *mut i32,
    work: *mut f32,
    kfactor: *mut f32,
    max_iter: *mut i32,
    max_zero_wgt: *mut i32,
    weights: *mut f32,
) -> i32 {
    unsafe {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polynomial_fit_recovers_quadratic() {
        unsafe {
            let mut x = [-2., -1., 0., 1., 2.];
            let mut y = [3., 0., 1., 6., 15.];
            let mut slopes = [0.; 2];
            let mut intcpt = 0.;
            let mut work = [0.; 64];
            assert_eq!(
                polynomial_fit(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    5,
                    2,
                    slopes.as_mut_ptr(),
                    &mut intcpt,
                    work.as_mut_ptr()
                ),
                0
            );
            assert!((intcpt - 1.).abs() < 1.0e-5);
            assert!((slopes[0] - 3.).abs() < 1.0e-5);
            assert!((slopes[1] - 2.).abs() < 1.0e-5);
        }
    }
    #[test]
    fn stat_matrices_column_major() {
        unsafe {
            let mut x = [1., 2., 3., 2., 4., 6.];
            let mut sx = [0.; 2];
            let mut ss = [0.; 4];
            let mut ssd = [0.; 4];
            let mut d = [0.; 4];
            let mut r = [0.; 4];
            let mut xm = [0.; 2];
            let mut sd = [0.; 2];
            stat_matrices(
                x.as_mut_ptr(),
                3,
                0,
                2,
                2,
                3,
                sx.as_mut_ptr(),
                ss.as_mut_ptr(),
                ssd.as_mut_ptr(),
                d.as_mut_ptr(),
                r.as_mut_ptr(),
                xm.as_mut_ptr(),
                sd.as_mut_ptr(),
                1,
            );
            assert_eq!(xm, [2., 4.]);
            assert!((r[1] - 1.).abs() < 1.0e-6);
        }
    }

    #[test]
    fn robust_regression_rejects_a_large_outlier() {
        unsafe {
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
                    data.as_mut_ptr(),
                    10,
                    0,
                    1,
                    10,
                    1,
                    sol.as_mut_ptr(),
                    1,
                    cons.as_mut_ptr(),
                    mean.as_mut_ptr(),
                    sd.as_mut_ptr(),
                    work.as_mut_ptr(),
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
