//! Translation of `IMOD/flib/model/solve_wo_outliers.f90`.
//!
//! The module `d3multvars` is [`D3MultVars`]; it replaces a common for
//! communicating between `func` and `do3multr`, and lets `solve_wo_outliers`
//! initialize `do3multr` on its first call for reproducible searches.  As a
//! Fortran module it is program-wide state, so it is one process-global
//! value ([`D3MULTVARS`]); the `save`d local `var` of `do3multr` lives beside
//! it for the same reason.
//!
//! `xMat(matCols, *)` is column major: `xMat(j, i)` is
//! `x_mat[(j - 1) + (i - 1) * matCols]`.  `multRegress` is the libcfshr
//! Fortran wrapper `multregress` the gfortran symbol resolves to (its weight
//! column is shifted down by one), and `amoebaInit`/`amoeba` are the
//! `amoebainitfwrap`/`amoebafwrap` wrappers (1-based `jmin`).  None of this
//! goes through LAPACK.

use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::compat::gfortran_rt::maxss;
use crate::imod::flib::subrs::statsubs::statfuncs::erfcc;
use crate::imod::libcfshr::amat_to_rotmagstr::{amat_to_rotmag, rotmag_to_amat};
use crate::imod::libcfshr::amoeba::{amoebafwrap, amoebainitfwrap};
use crate::imod::libcfshr::regression::multregress;
use crate::imod::libcfshr::simplestat::sums_to_avg_sd;
use std::sync::Mutex;

/// Original module `d3multvars` (`solve_wo_outliers.f90:5`).
///
/// `aa(3,3)` is column major and the `a11`..`a33` equivalences name its
/// elements: `aIJ` is `aa[(I - 1) + (J - 1) * 3]`.  `sumSq(4,6)` likewise:
/// `sumSq(k, l)` is `sum_sq[(k - 1) + (l - 1) * 4]`.
pub struct D3MultVars {
    /// `sumSq(4,6)`.
    pub sum_sq: [f64; 24],
    /// `aa(3,3)`.
    pub aa: [f64; 9],
    /// `errMin`.
    pub err_min: f64,
    /// `nullAxis`.
    pub null_axis: i32,
    /// `magSign`.
    pub mag_sign: i32,
    /// `ifTrace`.
    pub if_trace: i32,
    /// `numTrials`.
    pub num_trials: i32,
    /// `firstTime`, `data /.true./`.
    pub first_time: bool,
    /// `do3multr`'s `save var` (`solve_wo_outliers.f90:195`): not a module
    /// variable, but a `SAVE`d local persists for the program's life exactly
    /// as the module does, so it is kept here.
    pub var: [f32; 6],
}

/// The one `d3multvars` module instance.
pub static D3MULTVARS: Mutex<D3MultVars> = Mutex::new(D3MultVars {
    sum_sq: [0.0; 24],
    aa: [0.0; 9],
    err_min: 0.0,
    null_axis: 0,
    mag_sign: 0,
    if_trace: 0,
    num_trials: 0,
    first_time: true,
    var: [0.0; 6],
});

/// `parameter (maxErr = 100000)` (`solve_wo_outliers.f90:48`).
const MAX_ERR: i32 = 100000;

/// Original `solve_wo_outliers` (`solve_wo_outliers.f90:43`).
///
/// SOLVE_WO_OUTLIERS solves for a fit between sets of positions in 3D and
/// eliminates outlying position-pairs from the solution.
///
/// XMAT is the data matrix, MATCOLS is its first dimension and must be at
/// least 17.  NUMDATA is the full amount of data.  NUMCOL is the number of
/// columns of independent variables.  ICOLFIXED specifies which column has
/// insufficient variance (is fixed) and should be negative to indicate
/// inversion in that dimension.  MAXDROP is the maximum number of points to
/// drop as outliers.  CRITPROB and ABSPROBCRIT are the criterion
/// probabilities for considering a point an outlier.  If the maximum
/// residual is below ELIMMIN nothing will be eliminated.  NUMDROP is number
/// of points dropped, point numbers returned in IDROP.  AMAT3D is the 3x3
/// matrix computed, DELXYZ has the displacements, CENMEANLOC is the mean
/// location in the dependent variables, DEVMEAN, DEVSD, DEVMAX are mean, SD,
/// and maximum deviations, IPNTMAX and DEVXYZMAX give the point number and
/// deviation in X, Y, Z at which the maximum occurred.
///
/// The routine copies original data to columns 8 to 14 (or numCol + 5 to
/// 2*(numCol + 4)), puts a cross index from the ordered data to the original
/// row in column 5 (or numCol + 2), returns a mean residual in column 4
/// (numCol + 1) for the ordered data, and puts residual vector components for
/// the ordered data in columns 15 to 17 (2*(numCol + 4) + 1, 2, 3).
#[allow(clippy::too_many_arguments)]
pub fn solve_wo_outliers(
    x_mat: &mut [f32],
    mat_cols: i32,
    num_data: i32,
    num_col: i32,
    icol_fixed: i32,
    max_drop: i32,
    crit_prob: f32,
    abs_prob_crit: f32,
    elim_min: f32,
    idrop: &mut [i32],
    num_drop: &mut i32,
    a_mat3d: &mut [f32],
    del_xyz: &mut [f32],
    cen_mean_loc: &mut [f32],
    dev_mean: &mut f32,
    dev_sd: &mut f32,
    dev_max: &mut f32,
    ipnt_max: &mut i32,
    dev_xyz_max: &mut [f32],
) {
    let mc = mat_cols as usize;
    // `xMat(j, i)` as a linear index.
    let xi = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * mc;
    let mut cen_tmp = [0.0_f32; 3];
    let mut index: Vec<i32>;
    let mut last_drop: i32;
    let mut itmp: i32;
    let mut num_keep: i32;
    let prob_per_point: f32;
    let abs_per_point: f32;
    let mut sigma_from_mean: f32;
    let mut sigma_from_sd: f32;
    let mut sigma: f32;
    let mut z: f32;
    let mut prob: f32;
    // Statement function `gprob(z) = 1. - 0.5 * erfcc(z / 1.414214)`.
    let gprob = |z: f32| -> f32 { 1.0_f32 - 0.5_f32 * erfcc(z / 1.414214_f32) };
    D3MULTVARS.lock().unwrap().first_time = true;
    //
    // get probability per single point from the overall criterion prob
    //
    prob_per_point = (1.0_f32 - crit_prob).powf(1.0_f32 / num_data as f32);
    abs_per_point = (1.0_f32 - abs_prob_crit).powf(1.0_f32 / num_data as f32);
    //
    // copy the data into columns 8-14 (for ncol = 3)
    //
    for i in 1..=num_data {
        for j in 1..=num_col + 4 {
            x_mat[xi(j + num_col + 4, i)] = x_mat[xi(j, i)];
        }
    }

    do3multr(
        x_mat,
        mat_cols,
        num_data,
        num_col,
        num_data,
        icol_fixed,
        a_mat3d,
        del_xyz,
        cen_mean_loc,
        dev_mean,
        dev_sd,
        dev_max,
        ipnt_max,
        dev_xyz_max,
    );
    *num_drop = 0;
    //
    // If returning right away, load cross-indexes into col 5
    if max_drop == 0 || *dev_max < elim_min {
        for i in 1..=num_data {
            x_mat[xi(num_col + 2, i)] = i as f32;
        }
        return;
    }
    //
    // order the residuals
    //
    last_drop = 0;
    if num_data > MAX_ERR {
        println!(" CANNOT FIND OUTLIERS: ARRAYS NOT LARGE ENOUGH");
        return;
    }
    //
    // Sort the residuals and keep index back to initial values
    //
    index = vec![0; num_data as usize];
    for i in 1..=num_data {
        index[(i - 1) as usize] = i;
    }
    // The residual column, gathered once: the comparisons read exactly the
    // values `xMat(numCol + 1, index(i))` the source reads (nothing is written
    // to that column during the sort).
    let resid: Vec<f32> = (1..=num_data).map(|k| x_mat[xi(num_col + 1, k)]).collect();
    for i in 1..=num_data - 1 {
        for j in i + 1..=num_data {
            if resid[(index[(i - 1) as usize] - 1) as usize]
                > resid[(index[(j - 1) as usize] - 1) as usize]
            {
                itmp = index[(i - 1) as usize];
                index[(i - 1) as usize] = index[(j - 1) as usize];
                index[(j - 1) as usize] = itmp;
            }
        }
    }
    //
    // load the data in this order
    //
    for i in 1..=num_data {
        for j in 1..=num_col + 4 {
            x_mat[xi(j, i)] = x_mat[xi(j + num_col + 4, index[(i - 1) as usize])];
        }
    }
    //
    // Drop successively more points: get mean and S.D. of the remaining
    // points and check how many of the points pass the criterion
    // for outliers.
    //
    for jdrop in 1..=max_drop + 1 {
        do3multr(
            x_mat,
            mat_cols,
            num_data,
            num_col,
            num_data - jdrop,
            icol_fixed,
            a_mat3d,
            del_xyz,
            &mut cen_tmp,
            dev_mean,
            dev_sd,
            dev_max,
            ipnt_max,
            dev_xyz_max,
        );
        //
        // estimate the sigma for the error distribution as the maximum of
        // the values implied by the mean and the SD of the deviations
        //
        sigma_from_mean = *dev_mean / (8.0_f32 / 3.14159_f32).sqrt();
        sigma_from_sd = *dev_sd / (3.0_f32 - 8.0_f32 / 3.14159_f32).sqrt();
        // `solve_wo_outliers.f90:131`: `maxss sigmaFromMean, sigmaFromSD`.
        sigma = maxss(sigma_from_mean, sigma_from_sd);

        num_keep = 0;
        for j in num_data - jdrop + 1..=num_data {
            if sigma < 0.1_f32 * x_mat[xi(num_col + 1, j)].abs() || sigma < 1.0e-5_f32 {
                z = 10.0_f32.copysign(x_mat[xi(num_col + 1, j)]);
            } else {
                z = x_mat[xi(num_col + 1, j)] / sigma;
            }
            prob = 2.0_f32 * (gprob(z) - 0.5_f32)
                - (2.0_f32 / 3.14159_f32).sqrt() * z * (-(z * z) / 2.0_f32).exp();
            if prob < prob_per_point {
                num_keep += 1;
            }
            if prob >= abs_per_point {
                *num_drop = max_drop.min((*num_drop).max(num_data + 1 - j));
            }
        }
        //
        // If all points are outliers, this is a candidate for a set to drop
        // When only the first point is kept, and all the rest of the points
        // were outliers on the previous round, then this is a safe place to
        // draw the line between good data and outliers.  In this case, set
        // ndrop; and at end take the biggest ndrop that fits these criteria
        //
        if num_keep == 0 {
            last_drop = jdrop;
        }
        if num_keep == 1 && last_drop == jdrop - 1 && last_drop > 0 {
            *num_drop = last_drop;
        }
    }
    //
    // when finish loop, need to redo with right amount of data and save
    // indices in column past the residuals
    //
    for i in 1..=*num_drop {
        idrop[(i - 1) as usize] = index[(num_data + i - *num_drop - 1) as usize];
    }
    do3multr(
        x_mat,
        mat_cols,
        num_data,
        num_col,
        num_data - *num_drop,
        icol_fixed,
        a_mat3d,
        del_xyz,
        &mut cen_tmp,
        dev_mean,
        dev_sd,
        dev_max,
        ipnt_max,
        dev_xyz_max,
    );
    *ipnt_max = index[(*ipnt_max - 1) as usize];
    for i in 1..=num_data {
        x_mat[xi(num_col + 2, i)] = index[(i - 1) as usize] as f32;
    }
}

/// Original `do3multr` (`solve_wo_outliers.f90:185`).
///
/// DO3MULTR does the three regressions to determine a transformation
/// matrix, or it does a search for a reduced set of parameters if ICOLFIXIN
/// is non-zero.  XMAT is the data matrix, MATCOLS is its first dimension,
/// NUMDATA is the full amount of data, NUMCOLIN is the number of columns of
/// independent variables, NUMFIT is the number of data points to use,
/// ICOLFIXIN specifies which column has insufficient variance (is fixed) and
/// should be negative to indicate inversion in that dimension, AMAT3D is the
/// 3x3 matrix computed, DELXYZ has the displacements, CENMEANLOC is the mean
/// location in the dependent variables, DEVMEAN, DEVSD, DEVMAX are mean, SD,
/// and maximum deviations, IPNTMAX and DEVXYZMAX give the point number and
/// deviation in X, Y, Z at which the maximum occurred.
///
/// `aMat3d(3,*)` is column major (`aMat3d(i, j)` is `a_mat3d[(i-1) + (j-1)*3]`).
/// The automatic arrays `xMeans`, `sd`, `b1`, `b3` are uninitialised in the
/// source and only read where written first; they start at zero here.
#[allow(clippy::too_many_arguments)]
pub fn do3multr(
    x_mat: &mut [f32],
    mat_cols: i32,
    num_data: i32,
    num_col_in: i32,
    num_fit: i32,
    icol_fix_in: i32,
    a_mat3d: &mut [f32],
    del_xyz: &mut [f32],
    cen_mean_loc: &mut [f32],
    dev_mean: &mut f32,
    dev_sd: &mut f32,
    dev_max: &mut f32,
    ipnt_max: &mut i32,
    dev_xyz_max: &mut [f32],
) {
    let mut guard = D3MULTVARS.lock().unwrap();
    let d3 = &mut *guard;
    let mc = mat_cols as usize;
    let xi = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * mc;
    let ai = |i: i32, j: i32| (i - 1) as usize + (j - 1) as usize * 3;
    let mut x_means = vec![0.0_f32; mc];
    let mut sd = vec![0.0_f32; mc];
    let mut work = [0.0_f32; 100];
    let mut b1 = vec![0.0_f32; mc];
    let mut b3 = vec![0.0_f32; mc * 3];
    let mut dev_xyz = [0.0_f32; 3];
    let icol_fixed: i32;
    let mut x_mean_save = [0.0_f32; 6];
    let mut num_col_do: i32;
    let mut k: i32;
    let mut km: i32;
    let mut j: i32;
    let mut dev_sum: f32;
    let mut dev_sq: f32;
    let mut dev_pnt: f32;
    let mut amat = [0.0_f32; 4];
    let func_err: f32;
    let mut sum: f64;
    let mut pp = [0.0_f32; 49];
    let mut yy = [0.0_f32; 7];
    let mut ptol = [0.0_f32; 6];
    let da: [f32; 6] = [2., 2., 0.02, 0.02, 2., 2.];
    let ptol1: f32;
    let ftol1: f32;
    let delfac: f32;
    let ftol2: f32;
    let ptol2: f32;
    let mut jmin: usize = 0;
    let mut iter: i32 = 0;
    let mut icol: i32;
    //
    // values for simplex fit
    //
    ptol2 = 5.0e-4;
    ftol2 = 5.0e-4;
    ptol1 = 1.0e-5;
    ftol1 = 1.0e-5;
    delfac = 2.;
    d3.err_min = 1.0e30_f32 as f64;
    d3.num_trials = 0;
    // set to 1 for results of each fit, 2 for trace of new minima, 3 for
    // full trace of simplex search
    d3.if_trace = 0;

    num_col_do = num_col_in;
    icol_fixed = icol_fix_in.abs();
    if icol_fixed == 0 {
        //
        // Simple fit, shove the data down to fill the empty spot
        for i in 1..=num_fit {
            for j in num_col_do + 1..=num_col_do + 3 {
                x_mat[xi(j, i)] = x_mat[xi(j + 1, i)];
            }
        }
        //
        // Do the fit and fill the matrix
        let _ = multregress(
            x_mat,
            &mat_cols,
            &1,
            &num_col_do,
            &num_fit,
            &3,
            &0,
            &mut b3,
            &mat_cols,
            Some(&mut del_xyz[..3]),
            &mut x_means,
            &mut sd,
            &mut work,
        );
        for ixyz in 1..=3 {
            for j in 1..=num_col_do {
                a_mat3d[ai(ixyz, j)] = b3[(j - 1) as usize + (ixyz - 1) as usize * mc];
            }
        }
        //
        // restore the data
        for i in 1..=num_fit {
            let mut j = num_col_do + 3;
            while j >= num_col_do + 1 {
                x_mat[xi(j + 1, i)] = x_mat[xi(j, i)];
                j -= 1;
            }
        }
    } else {
        d3.null_axis = icol_fixed;
        d3.mag_sign = if icol_fix_in >= 0 { 1 } else { -1 };
        //
        // if one column is fixed first get sums of squares and cross products
        //
        for icol in 1..=6 {
            j = icol;
            if icol > 3 {
                j = icol + 1;
            }
            sum = 0.;
            for i in 1..=num_fit {
                sum += x_mat[xi(j, i)] as f64;
            }
            x_mean_save[(icol - 1) as usize] = (sum / num_fit as f64) as f32;
        }
        for icol in 1..=6 {
            j = icol;
            if icol > 3 {
                j = icol + 1;
            }
            for ixyz in 1..=4 {
                k = ixyz;
                km = ixyz;
                if ixyz == 4 {
                    k = j;
                    km = icol;
                }
                sum = 0.;
                for i in 1..=num_fit {
                    sum += ((x_mat[xi(j, i)] - x_mean_save[(icol - 1) as usize])
                        * (x_mat[xi(k, i)] - x_mean_save[(km - 1) as usize]))
                        as f64;
                }
                d3.sum_sq[(ixyz - 1) as usize + (icol - 1) as usize * 4] = sum;
            }
        }
        //
        // now decrement the number of columns to do
        // subtract that column's independent var from its dependent var (FOR NO CURRENT
        // REASON) and pack the independent vars into the smaller number of columns
        //
        num_col_do = num_col_in - 1;
        for i in 1..=num_fit {
            x_mat[xi(num_col_in + 1 + icol_fixed, i)] =
                x_mat[xi(num_col_in + 1 + icol_fixed, i)] - x_mat[xi(icol_fixed, i)];
            x_mat[xi(num_col_in + 1, i)] = x_mat[xi(icol_fixed, i)];
            for j in icol_fixed..=num_col_do {
                x_mat[xi(j, i)] = x_mat[xi(j + 1, i)];
            }
        }
        //
        // do two multr's, moving the appropriate column of dependent
        // var data into the one past the packed independent vars
        //
        icol = 1;
        for ixyz in 1..=3 {
            if ixyz != icol_fixed {
                for i in 1..=num_fit {
                    x_mat[xi(num_col_do + 1, i)] = x_mat[xi(num_col_in + 1 + ixyz, i)];
                }
                let mut cons = [0.0_f32; 1];
                let _ = multregress(
                    x_mat,
                    &mat_cols,
                    &1,
                    &num_col_do,
                    &num_fit,
                    &1,
                    &0,
                    &mut b1,
                    &mat_cols,
                    Some(&mut cons),
                    &mut x_means,
                    &mut sd,
                    &mut work,
                );
                for j in 1..=num_col_do {
                    a_mat3d[ai(icol, j)] = b1[(j - 1) as usize];
                }
                del_xyz[(ixyz - 1) as usize] = cons[0];
                cen_mean_loc[(ixyz - 1) as usize] = x_means[num_col_do as usize];
                icol += 1;
            }
        }
        //
        // restore the data
        //
        for i in 1..=num_fit {
            let mut j = num_col_do;
            while j >= icol_fixed {
                x_mat[xi(j + 1, i)] = x_mat[xi(j, i)];
                j -= 1;
            }
            x_mat[xi(icol_fixed, i)] = x_mat[xi(num_col_in + 1, i)];
            x_mat[xi(num_col_in + 1 + icol_fixed, i)] =
                x_mat[xi(num_col_in + 1 + icol_fixed, i)] + x_mat[xi(icol_fixed, i)];
        }
        //
        // get starting variables for minimization from matrix the first time,
        // or just restart from previous run
        //
        // `amat(2,2)` column major: amat(1,1), amat(2,1), amat(1,2), amat(2,2).
        amat[0] = a_mat3d[ai(1, 1)];
        amat[2] = a_mat3d[ai(1, 2)];
        amat[1] = a_mat3d[ai(2, 1)];
        amat[3] = a_mat3d[ai(2, 2)];
        if icol_fixed == 2 {
            amat[2] = -a_mat3d[ai(1, 2)];
            amat[1] = -a_mat3d[ai(2, 1)];
        }

        if d3.first_time {
            let (v1, v2, v3, v4) = amat_to_rotmag(amat[0], amat[2], amat[1], amat[3]);
            d3.var[0] = v1;
            d3.var[1] = v2;
            d3.var[2] = v3;
            d3.var[3] = v4;
            d3.var[4] = 0.;
            d3.var[5] = 0.;
        }
        d3.first_time = false;
        let mut var = d3.var;
        {
            let mut funk = |sol_vec: &[f32]| -> f32 { func(sol_vec, d3) };
            amoebainitfwrap(
                &mut pp, &mut yy, 7, 6, delfac, ptol2, &var, &da, &mut funk, &mut ptol,
            );
            amoebafwrap(
                &mut pp, &mut yy, 7, 6, ftol2, &mut funk, &mut iter, &ptol, &mut jmin,
            );
            //
            // per Press et al. recommendation, just restart at current location
            //
            for i in 1..=6usize {
                var[i - 1] = pp[(jmin - 1) + (i - 1) * 7];
            }
            amoebainitfwrap(
                &mut pp, &mut yy, 7, 6, delfac, ptol1, &var, &da, &mut funk, &mut ptol,
            );
            amoebafwrap(
                &mut pp, &mut yy, 7, 6, ftol1, &mut funk, &mut iter, &ptol, &mut jmin,
            );
        }
        //
        // recover result, get aa matrix from best result, compute dxyz
        //
        for i in 1..=6usize {
            var[i - 1] = pp[(jmin - 1) + (i - 1) * 7];
        }
        d3.var = var;
        // `sqrt(max(0., yy(jmin) / numFit))` (`solve_wo_outliers.f90:368`):
        // `maxss yy(jmin) / numFit, 0.` in the reference object.
        func_err = maxss(yy[jmin - 1] / num_fit as f32, 0.0_f32).sqrt();
        if d3.if_trace != 0 {
            println!(
                "{:>5}{}{}{}{}{}{}{}",
                iter,
                format_f(func_err as f64, 10, 4),
                format_f(var[0] as f64, 9, 2),
                format_f(var[1] as f64, 9, 2),
                format_f(var[2] as f64, 9, 4),
                format_f(var[3] as f64, 9, 4),
                format_f(var[4] as f64, 9, 2),
                format_f(var[5] as f64, 9, 2)
            );
        }
        for i in 1..=3 {
            cen_mean_loc[(i - 1) as usize] = x_mean_save[(i + 3 - 1) as usize];
            del_xyz[(i - 1) as usize] = x_mean_save[(i + 3 - 1) as usize];
            for j in 1..=3 {
                a_mat3d[ai(i, j)] = d3.aa[ai(i, j)] as f32;
                del_xyz[(i - 1) as usize] -= a_mat3d[ai(i, j)] * x_mean_save[(j - 1) as usize];
            }
            if d3.if_trace != 0 {
                println!(
                    "{}{}{}{}",
                    format_f(a_mat3d[ai(i, 1)] as f64, 10, 6),
                    format_f(a_mat3d[ai(i, 2)] as f64, 10, 6),
                    format_f(a_mat3d[ai(i, 3)] as f64, 10, 6),
                    format_f(del_xyz[(i - 1) as usize] as f64, 10, 3)
                );
            }
        }
    }
    //
    // compute deviations for all the data (numData), keep track of max
    // for the ones in the fit (ndo)
    // return residual vector components in columns 15 to 17
    //
    dev_sum = 0.;
    *dev_max = -1.;
    dev_sq = 0.;
    let nci = num_col_in as usize;
    for ipnt in 1..=num_data {
        // Row `ipnt` of `xMat` (its `matCols` elements), `xMat(j, ipnt)` is
        // `row[j - 1]`.
        let row = &mut x_mat[(ipnt - 1) as usize * mc..ipnt as usize * mc];
        for ixyz in 1..=3usize {
            dev_xyz[ixyz - 1] = del_xyz[ixyz - 1] - row[nci + ixyz];
            for j in 1..=nci {
                dev_xyz[ixyz - 1] += a_mat3d[(ixyz - 1) + (j - 1) * 3] * row[j - 1];
            }
            row[2 * (nci + 4) + ixyz - 1] = dev_xyz[ixyz - 1];
        }
        dev_pnt =
            (dev_xyz[0] * dev_xyz[0] + dev_xyz[1] * dev_xyz[1] + dev_xyz[2] * dev_xyz[2]).sqrt();
        row[nci] = dev_pnt;
        if ipnt <= num_fit {
            dev_sum += dev_pnt;
            dev_sq += dev_pnt * dev_pnt;
            if dev_pnt > *dev_max {
                *dev_max = dev_pnt;
                *ipnt_max = ipnt;
                for i in 1..=3usize {
                    dev_xyz_max[i - 1] = dev_xyz[i - 1];
                }
            }
        }
    }
    sums_to_avg_sd(dev_sum, dev_sq, num_fit, dev_mean, dev_sd);
}

/// Original `func` (`solve_wo_outliers.f90:385`).
///
/// FUNC is called by the simplex routine to compute the sum squared error
/// for the solution vector in solVec.  The `real*8 all(2,2,3)` array and its
/// `b`, `c`, `d` equivalences are one column-major `[f64; 12]`:
/// `all(i, j, k)` is `all[(i-1) + (j-1)*2 + (k-1)*4]`, `b` is `k = 1`, `c`
/// is `k = 2`, `d` is `k = 3`.
pub fn func(sol_vec: &[f32], d3: &mut D3MultVars) -> f32 {
    let mut all = [0.0_f64; 12];
    let mut smag = [0.0_f64; 3];
    let alli = |i: usize, j: usize, k: usize| (i - 1) + (j - 1) * 2 + (k - 1) * 4;
    let cos_ang1: f64;
    let cos_ang2: f64;
    let sin_ang1: f64;
    let sin_ang2: f64;
    let dtor: f64;
    let ind5: usize;
    let ind6: usize;
    let mut func_err: f32;
    let mut star_out: char;
    //
    dtor = (3.1415926536_f32 / 180.0_f32) as f64;
    d3.num_trials += 1;
    cos_ang1 = (sol_vec[4] as f64 * dtor).cos();
    sin_ang1 = (sol_vec[4] as f64 * dtor).sin();
    cos_ang2 = (sol_vec[5] as f64 * dtor).cos();
    sin_ang2 = (sol_vec[5] as f64 * dtor).sin();
    //
    // set up the X, Y, and Z rotation matrices
    //
    if d3.null_axis == 1 {
        ind5 = 2;
        ind6 = 3;
    } else if d3.null_axis == 2 {
        ind5 = 1;
        ind6 = 3;
    } else {
        ind5 = 1;
        ind6 = 2;
    }
    let null = d3.null_axis as usize;
    //
    // `r4mat(2,2)` column major: r4mat(1,1), r4mat(2,1), r4mat(1,2), r4mat(2,2).
    let r4mat = rotmag_to_amat(sol_vec[0], sol_vec[1], sol_vec[2], sol_vec[3]);
    all[alli(1, 1, null)] = r4mat[0] as f64;
    all[alli(1, 2, null)] = r4mat[2] as f64;
    all[alli(2, 1, null)] = r4mat[1] as f64;
    all[alli(2, 2, null)] = r4mat[3] as f64;
    smag[null - 1] = (d3.mag_sign as f32 * sol_vec[2]) as f64;
    smag[ind5 - 1] = 1.;
    smag[ind6 - 1] = 1.;
    all[alli(1, 1, ind5)] = cos_ang1;
    all[alli(1, 2, ind5)] = -sin_ang1;
    all[alli(2, 1, ind5)] = sin_ang1;
    all[alli(2, 2, ind5)] = cos_ang1;
    all[alli(1, 1, ind6)] = cos_ang2;
    all[alli(1, 2, ind6)] = -sin_ang2;
    all[alli(2, 1, ind6)] = sin_ang2;
    all[alli(2, 2, ind6)] = cos_ang2;
    all[alli(1, 2, 2)] = -all[alli(1, 2, 2)];
    all[alli(2, 1, 2)] = -all[alli(2, 1, 2)];
    let b = |i: usize, j: usize| all[alli(i, j, 1)];
    let c = |i: usize, j: usize| all[alli(i, j, 2)];
    let d = |i: usize, j: usize| all[alli(i, j, 3)];
    //
    // get products for full a matrix.  If Y is the null axis, put it
    // first to avoid Z and X rotations playing off against each other
    // with a 90 degree rotation around Y
    //
    let aa = &mut d3.aa;
    // a11 = aa[0], a21 = aa[1], a31 = aa[2], a12 = aa[3], a22 = aa[4],
    // a32 = aa[5], a13 = aa[6], a23 = aa[7], a33 = aa[8].
    if d3.null_axis == 2 {
        //
        // Product C * B * D:
        //
        aa[0] = b(2, 1) * c(1, 2) * d(2, 1) + smag[0] * c(1, 1) * d(1, 1);
        aa[3] = b(2, 1) * c(1, 2) * d(2, 2) + smag[0] * c(1, 1) * d(1, 2);
        aa[6] = smag[2] * b(2, 2) * c(1, 2);
        aa[1] = smag[1] * b(1, 1) * d(2, 1);
        aa[4] = smag[1] * b(1, 1) * d(2, 2);
        aa[7] = smag[1] * b(1, 2) * smag[2];
        aa[2] = b(2, 1) * c(2, 2) * d(2, 1) + smag[0] * c(2, 1) * d(1, 1);
        aa[5] = b(2, 1) * c(2, 2) * d(2, 2) + smag[0] * c(2, 1) * d(1, 2);
        aa[8] = smag[2] * b(2, 2) * c(2, 2);
    } else {
        //
        // Product B * C * D:
        //
        aa[0] = smag[0] * c(1, 1) * d(1, 1);
        aa[3] = smag[0] * c(1, 1) * d(1, 2);
        aa[6] = smag[0] * c(1, 2) * smag[2];
        aa[1] = b(1, 2) * c(2, 1) * d(1, 1) + smag[1] * b(1, 1) * d(2, 1);
        aa[4] = b(1, 2) * c(2, 1) * d(1, 2) + smag[1] * b(1, 1) * d(2, 2);
        aa[7] = smag[2] * b(1, 2) * c(2, 2);
        aa[2] = b(2, 2) * c(2, 1) * d(1, 1) + smag[1] * b(2, 1) * d(2, 1);
        aa[5] = b(2, 2) * c(2, 1) * d(1, 2) + smag[1] * b(2, 1) * d(2, 2);
        aa[8] = smag[2] * b(2, 2) * c(2, 2);
    }
    let (a11, a21, a31, a12, a22, a32, a13, a23, a33) = (
        aa[0], aa[1], aa[2], aa[3], aa[4], aa[5], aa[6], aa[7], aa[8],
    );
    let s = |k: usize, l: usize| d3.sum_sq[(k - 1) + (l - 1) * 4];
    //
    // get error sum
    //
    func_err = ((a11 * a11 + a21 * a21 + a31 * a31) * s(1, 1)
        + 2. * (a11 * a12 + a21 * a22 + a31 * a32) * s(1, 2)
        + 2. * (a11 * a13 + a21 * a23 + a31 * a33) * s(1, 3)
        + (a12 * a12 + a22 * a22 + a32 * a32) * s(2, 2)
        + 2. * (a12 * a13 + a22 * a23 + a32 * a33) * s(2, 3)
        + (a13 * a13 + a23 * a23 + a33 * a33) * s(3, 3)
        - 2. * (a11 * s(1, 4) + a21 * s(1, 5) + a31 * s(1, 6))
        - 2. * (a12 * s(2, 4) + a22 * s(2, 5) + a32 * s(2, 6))
        - 2. * (a13 * s(3, 4) + a23 * s(3, 5) + a33 * s(3, 6))
        + s(4, 4)
        + s(4, 5)
        + s(4, 6)) as f32;

    if d3.if_trace > 1 {
        star_out = ' ';
        if (func_err as f64) < d3.err_min {
            star_out = '*';
            d3.err_min = func_err as f64;
        }
        if d3.if_trace > 2 || (func_err as f64) < d3.err_min {
            println!(
                " {}{:>4}{}{}{}{}{}{}{}",
                star_out,
                d3.num_trials,
                format_f(func_err as f64, 15, 5),
                format_f(sol_vec[0] as f64, 9, 3),
                format_f(sol_vec[1] as f64, 9, 3),
                format_f(sol_vec[2] as f64, 9, 5),
                format_f(sol_vec[3] as f64, 9, 5),
                format_f(sol_vec[4] as f64, 9, 3),
                format_f(sol_vec[5] as f64, 9, 3)
            );
        }
    }
    func_err
}
