//! Translation of `IMOD/raptor/optimization/SFMestimationWithBA.h` and
//! `SFMestimationWithBA.cpp`: structure from motion with bundle adjustment
//! (the in-plane rotation search, the weighted matrix factorization),
//! outlier analysis, and the fiducial-model writers over an `SFMdata`.
//!
//! OpenCV calls whose source and destination are the same matrix in C++
//! (`cvSub(a, b, a)`, `cvMul(w, u, u)`, `cvPow(r, r, 0.5)`, ...) take a copy
//! of the aliased source first; OpenCV's element loops read each source
//! element before writing it, so the result is the same.  A `CvMat` the
//! source allocates only for `cvGetCol`/`cvGetRow` to overwrite its header
//! (leaking the allocation) is the view itself here.
//!
//! `MakeDiag` and `norm` are not reached (DEAD_CODE.md); `norm` returns 0
//! whatever its argument.

use super::contour::Contour;
use super::estimation3d::{create_rand_mat, estimation_3d, rep_mat};
use super::estimation3ddata::Estimation3dData;
use super::prob_data::ProbData;
use super::sfm_data::SfmData;
use super::std_qp::{lp_solver, std_qp, zero_mat};
use crate::imod::c_sort::std_sort;
use crate::imod::cxx_stream::{cout, ostream_double};
use crate::imod::libcfshr::b3dutil::{RAND_MAX, exit, rand};
use crate::imod::raptor::main_classes::constants::{DELTA, MIN_WEIGHT, TOL, get_date};
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::point2d::Point2D;
use crate::imod::raptor::opencv::cxarithm::{cv_add, cv_add_s, cv_mul, cv_sub};
use crate::imod::raptor::opencv::cxconvert::cv_convert_scale;
use crate::imod::raptor::opencv::cxerror::{CV_ERR_MODE_LEAF, cv_set_err_mode};
use crate::imod::raptor::opencv::cxmathfuncs::cv_pow;
use crate::imod::raptor::opencv::cxmatmul::cv_mat_mul;
use crate::imod::raptor::opencv::cxmatrix::cv_transpose;
use crate::imod::raptor::opencv::cxnorm::{CV_L1, cv_norm};
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat, CvMatRef, CvScalar};
use crate::imod::raptor::suitesparse::cs_compress::cs_compress;
use crate::imod::raptor::suitesparse::cs_dropzeros::cs_dropzeros;
use crate::imod::raptor::suitesparse::cs_entry::cs_entry;
use crate::imod::raptor::suitesparse::cs_util::cs_spalloc;
use crate::imod::raptor::trajectory::trajectory::{free_trajectory_vector, write_imod_fid_model};

/// `Stack(CvMat* mat)` (`SFMestimationWithBA.cpp:19`).
pub fn stack(mat: &CvMat) -> CvMat {
    let x = mat.get_size().height;
    let y = mat.get_size().width;
    let mut result = CvMat::create(x * y, 1, CV_64FC1);
    for j in 0..y {
        for i in 0..x {
            result.set_real_2d(i + j * x, 0, mat.get_real_2d(i, j));
        }
    }
    result
}

/// `ConcatenateMatrix(CvMat* one, CvMat* two)` (`SFMestimationWithBA.cpp:47`).
pub fn concatenate_matrix(one: &CvMat, two: &CvMat) -> CvMat {
    let mut dst = CvMat::create(
        one.get_size().height,
        one.get_size().width + two.get_size().width,
        CV_64FC1,
    );
    for i in 0..one.get_size().height {
        for j in 0..one.get_size().width {
            dst.set_real_2d(i, j, one.get_real_2d(i, j));
        }
    }
    for i in 0..two.get_size().height {
        for j in 0..two.get_size().width {
            dst.set_real_2d(i, j + one.get_size().width, two.get_real_2d(i, j));
        }
    }
    dst
}

/// `ConcatenateMatrixDown(CvMat* one, CvMat* two)`
/// (`SFMestimationWithBA.cpp:68`).
pub fn concatenate_matrix_down(one: &CvMat, two: &CvMat) -> CvMat {
    let mut dst = CvMat::create(
        one.get_size().height + two.get_size().height,
        one.get_size().width,
        CV_64FC1,
    );
    for i in 0..one.get_size().height {
        for j in 0..one.get_size().width {
            dst.set_real_2d(i, j, one.get_real_2d(i, j));
        }
    }
    for i in 0..two.get_size().height {
        for j in 0..two.get_size().width {
            dst.set_real_2d(i + one.get_size().height, j, two.get_real_2d(i, j));
        }
    }
    dst
}

/// `OnesMat(int x, int y)` (`SFMestimationWithBA.cpp:102`).
pub fn ones_mat(x: i32, y: i32) -> CvMat {
    let mut result = CvMat::create(x, y, CV_64FC1);
    result.set(CvScalar::new(1.0, 0.0, 0.0, 0.0));
    result
}

/// The `estimation3D` call and `debugMode` report shared, verbatim, by the
/// source's four alpha searches is written out at each site.
///
/// `SFMestimationWithBA(SFMdata* sfm_, vector<double> tiltAngles, int W,
/// int H, float percentile, double* alphaFinal, int option, bool
/// debugMode)` (`SFMestimationWithBA.cpp:109`).  `sfm_` is deleted inside.
#[allow(clippy::too_many_arguments)]
pub fn sfm_estimation_with_ba(
    sfm_: SfmData,
    tilt_angles: &[f64],
    w: i32,
    h: i32,
    percentile: f32,
    alpha_final: &mut f64,
    option: i32,
    debug_mode: bool,
) -> SfmData {
    // to be able to debug with valgrind
    cv_set_err_mode(CV_ERR_MODE_LEAF);

    let t_ = sfm_.contour_x.get_num_frame();
    let m = sfm_.contour_x.get_num_traj();

    let mut sfm = sfm_.clone();
    let mut the_prob = ProbData::new();

    let mut mf = zero_mat(2 * t_, m);
    let mut wf = ones_mat(2 * t_, m);

    let mut b = CvMat::create(2 * t_, m, CV_64FC1); // helps creating constrains later
    let mut contourxrowtr = CvMat::create(1, m, CV_64FC1);
    let mut contouryrowtr = CvMat::create(1, m, CV_64FC1);

    let w_2 = 0.5 * w as f64;
    let h_2 = 0.5 * h as f64;

    for j in 0..t_ {
        cv_transpose(sfm_.contour_x.scores.get_col(j), &mut contourxrowtr);
        let src = contourxrowtr.clone();
        cv_convert_scale(src.view(), &mut contourxrowtr, 1.0, -w_2);

        for i in 0..m {
            let elem = contourxrowtr.get_real_2d(0, i);
            mf.set_real_2d(j + j, i, elem);
            if (elem + w_2).abs() < 0.1 {
                wf.set_real_2d(j + j, i, MIN_WEIGHT); // otherwise we leave it as a one
                b.set_real_2d(j + j, i, MIN_WEIGHT * elem);
            } else {
                b.set_real_2d(j + j, i, elem);
            }
        }

        cv_transpose(sfm_.contour_y.scores.get_col(j), &mut contouryrowtr);
        let src = contouryrowtr.clone();
        cv_convert_scale(src.view(), &mut contouryrowtr, 1.0, -h_2);

        for i in 0..m {
            let elem = contouryrowtr.get_real_2d(0, i);
            mf.set_real_2d(j + j + 1, i, contouryrowtr.get_real_2d(0, i));
            if (elem + h_2).abs() < 0.1 {
                wf.set_real_2d(j + j + 1, i, MIN_WEIGHT); // otherwise we leave it as a one
                b.set_real_2d(j + j + 1, i, MIN_WEIGHT * elem);
            } else {
                b.set_real_2d(j + j + 1, i, elem);
            }
        }
    }
    drop(contourxrowtr);
    drop(contouryrowtr);

    // generate Q matrix
    let mut tt = cs_spalloc(
        2 * t_ + 3 * m + 4 * t_ * m,
        2 * t_ + 3 * m + 4 * t_ * m,
        2 * t_ * m,
        1,
        1,
    );
    for i in 2 * t_ * m + 2 * t_ + 3 * m..(4 * t_ * m + 2 * t_ + 3 * m) {
        cs_entry(&mut tt, i, i, 1.0);
    }
    the_prob.q = cs_compress(&tt);
    drop(tt);

    // generate c matrix
    let mut tt = cs_spalloc(4 * t_ * m + 2 * t_ + 3 * m, 1, 2 * t_ * m, 1, 1);
    for i in 2 * t_ + 3 * m..(2 * t_ + 3 * m + 2 * t_ * m) {
        cs_entry(&mut tt, i, 0, DELTA);
    }
    the_prob.c = cs_compress(&tt);
    drop(tt);

    let stackb = stack(&b);
    let mut negb = CvMat::create(stackb.get_size().height, stackb.get_size().width, CV_64FC1);
    cv_convert_scale(stackb.view(), &mut negb, -1.0, 0.0);

    the_prob.buc = Some(concatenate_matrix_down(&negb, &stackb));

    drop(b);
    drop(stackb);

    // create part of matrix A
    let mut tt = cs_spalloc(4 * t_ * m, 2 * t_ + 3 * m + 4 * t_ * m, 24 * t_ * m, 1, 1);
    for j in 1..=m {
        let mut k = 2 * t_ * (j - 1);
        let mut r = 0;
        while k < 2 * t_ * j {
            cs_entry(&mut tt, k, r, -wf.get_real_2d(r, j - 1));
            k += 1;
            r += 1;
        }
        let mut k = 2 * t_ * (j - 1);
        let mut r = 3 * m + 2 * t_ + (2 * t_ * (j - 1));
        while k < 2 * t_ * j {
            cs_entry(&mut tt, k, r, -1.0);
            k += 1;
            r += 1;
        }
        let mut k = 2 * t_ * (j - 1);
        let mut r = 3 * m + 2 * t_ + 2 * t_ * m + 2 * t_ * (j - 1);
        while k < 2 * t_ * j {
            cs_entry(&mut tt, k, r, 1.0);
            k += 1;
            r += 1;
        }
        let mut k = 2 * t_ * m + 2 * t_ * (j - 1);
        let mut r = 0;
        while k < 2 * t_ * m + 2 * t_ * j {
            cs_entry(&mut tt, k, r, wf.get_real_2d(r, j - 1));
            k += 1;
            r += 1;
        }
        let mut k = 2 * t_ * m + 2 * t_ * (j - 1);
        let mut r = 3 * m + 2 * t_ + 2 * t_ * (j - 1);
        while k < 2 * t_ * m + 2 * t_ * j {
            cs_entry(&mut tt, k, r, -1.0);
            k += 1;
            r += 1;
        }
        let mut k = 2 * t_ * m + 2 * t_ * (j - 1);
        let mut r = 3 * m + 2 * t_ + 2 * t_ * m + 2 * t_ * (j - 1);
        while k < 2 * t_ * m + 2 * t_ * j {
            cs_entry(&mut tt, k, r, -1.0);
            k += 1;
            r += 1;
        }
    }

    if debug_mode {
        cout("Done generating first part of prob->a structure\n");
    }

    for j in 1..=m {
        let mut mm = 2 * t_ * (j - 1);
        let mut cc = 0;
        while mm < 2 * t_ * j {
            let aux_wf = wf.get_real_2d(cc, j - 1);
            for n in 3 * (j - 1) + 2 * t_..2 * t_ + 3 * j {
                cs_entry(&mut tt, mm, n, -aux_wf);
                cs_entry(&mut tt, mm + 2 * t_ * m, n, aux_wf);
            }
            mm += 1;
            cc += 1;
        }
    }

    // even if we add stuff to A later, it won't be in a new position
    the_prob.a = cs_compress(&tt);
    drop(tt);

    // remove zero entries from sparse matrix after weighting
    cs_dropzeros(the_prob.a.as_mut().unwrap());

    if debug_mode {
        cout("Done creating second part of structure prob-a\n");
    }

    let mut resid_mean_final: f64;

    if option == 0 {
        // setting up alpha
        let mut count_alpha_vec = 0;
        let mut alpha = zero_mat(1, 10);
        let mut i = -81.0f64;
        while i < 87.0 {
            alpha.set_real_2d(0, count_alpha_vec, i);
            count_alpha_vec += 1;
            i += 18.0;
        }

        // coarse search for alpha
        resid_mean_final = 1e+20;
        for k in 0..alpha.get_size().width {
            // good move. Otherwise strcuture will change for each call to estimation3D
            let mut copy_prob = ProbData::copy(&the_prob);
            let the_alpha = alpha.get_real_2d(0, k);
            let start = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let tresid_mean_struct = estimation_3d(
                &sfm_.contour_x,
                &sfm_.contour_y,
                the_alpha,
                tilt_angles,
                w,
                h,
                percentile as f64,
                &mut copy_prob,
                &wf,
                &mf,
            );
            let end = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            drop(copy_prob);
            if debug_mode {
                cout(&format!(
                    "Estimation 3D for alpha took {} secs\n",
                    ostream_double(end as f64 - start as f64)
                ));
                cout(&format!(
                    "Residual ={} for alpha={}\n",
                    ostream_double(tresid_mean_struct.resid_mean),
                    ostream_double(the_alpha)
                ));
                cout(&format!(
                    "Residual ={} for alpha={} with percentile={}\n",
                    ostream_double(tresid_mean_struct.resid_mean_perc),
                    ostream_double(the_alpha),
                    ostream_double(percentile as f64)
                ));
            }
            if tresid_mean_struct.resid_mean_perc < resid_mean_final {
                resid_mean_final = tresid_mean_struct.resid_mean_perc;
                *alpha_final = alpha.get_real_2d(0, k);
            }
        }
        if debug_mode {
            cout("First coarse search for alpha\n");
        }

        drop(alpha);
        // fine search for alpha
        let mut alpha = CvMat::create(1, 6, CV_64FC1);
        count_alpha_vec = 0;
        let mut i = (*alpha_final) - 9.0;
        while i <= (*alpha_final) + 10.0 {
            if (i - (*alpha_final)).abs() < 1e-3 {
                i += 3.0;
                continue; // we don't need to optimize again for the oarse value found in previous search
            }
            alpha.set_real_2d(0, count_alpha_vec, i);
            count_alpha_vec += 1;
            i += 3.0;
        }

        for k in 0..alpha.get_size().width {
            let mut copy_prob = ProbData::copy(&the_prob);
            let the_alpha = alpha.get_real_2d(0, k);
            let tresid_mean_struct = estimation_3d(
                &sfm_.contour_x,
                &sfm_.contour_y,
                the_alpha,
                tilt_angles,
                w,
                h,
                percentile as f64,
                &mut copy_prob,
                &wf,
                &mf,
            );
            drop(copy_prob);
            if debug_mode {
                cout(&format!(
                    "Residual ={} for alpha={}\n",
                    ostream_double(tresid_mean_struct.resid_mean),
                    ostream_double(the_alpha)
                ));
                cout(&format!(
                    "Residual ={} for alpha={} with percentile={}\n",
                    ostream_double(tresid_mean_struct.resid_mean_perc),
                    ostream_double(the_alpha),
                    ostream_double(percentile as f64)
                ));
            }
            if tresid_mean_struct.resid_mean_perc < resid_mean_final {
                resid_mean_final = tresid_mean_struct.resid_mean_perc;
                *alpha_final = alpha.get_real_2d(0, k);
            }
        }
        if debug_mode {
            cout("Fine search for alpha finished\n");
        }
        drop(alpha);

        // very fine search for alpha
        let mut alpha = CvMat::create(1, 2, CV_64FC1);
        alpha.set_real_2d(0, 0, (*alpha_final) - 1.0);
        alpha.set_real_2d(0, 1, (*alpha_final) + 1.0);
        for k in 0..2 {
            let mut copy_prob = ProbData::copy(&the_prob);
            let the_alpha = alpha.get_real_2d(0, k);
            let tresid_mean_struct = estimation_3d(
                &sfm_.contour_x,
                &sfm_.contour_y,
                the_alpha,
                tilt_angles,
                w,
                h,
                percentile as f64,
                &mut copy_prob,
                &wf,
                &mf,
            );
            drop(copy_prob);
            if debug_mode {
                cout(&format!(
                    "Residual ={} for alpha={}\n",
                    ostream_double(tresid_mean_struct.resid_mean),
                    ostream_double(the_alpha)
                ));
                cout(&format!(
                    "Residual ={} for alpha={} with percentile={}\n",
                    ostream_double(tresid_mean_struct.resid_mean_perc),
                    ostream_double(the_alpha),
                    ostream_double(percentile as f64)
                ));
            }
            if tresid_mean_struct.resid_mean_perc < resid_mean_final {
                resid_mean_final = tresid_mean_struct.resid_mean_perc;
                *alpha_final = alpha.get_real_2d(0, k);
            }
        }
        drop(alpha);
        if debug_mode {
            cout("Very fine search for alpha finished\n");
        }
    } else if option == 2 {
        // very fine search for alpha
        let mut alpha = CvMat::create(1, 5, CV_64FC1);
        alpha.set_real_2d(0, 0, (*alpha_final) - 0.5);
        alpha.set_real_2d(0, 1, (*alpha_final) - 0.25);
        alpha.set_real_2d(0, 2, *alpha_final);
        alpha.set_real_2d(0, 3, (*alpha_final) + 0.25);
        alpha.set_real_2d(0, 4, (*alpha_final) + 0.5);

        resid_mean_final = 1e20;
        for k in 0..alpha.get_size().width {
            let mut copy_prob = ProbData::copy(&the_prob);
            let the_alpha = alpha.get_real_2d(0, k);
            let tresid_mean_struct = estimation_3d(
                &sfm_.contour_x,
                &sfm_.contour_y,
                the_alpha,
                tilt_angles,
                w,
                h,
                percentile as f64,
                &mut copy_prob,
                &wf,
                &mf,
            );
            drop(copy_prob);
            if debug_mode {
                cout(&format!(
                    "Residual ={} for alpha={}\n",
                    ostream_double(tresid_mean_struct.resid_mean),
                    ostream_double(the_alpha)
                ));
                cout(&format!(
                    "Residual ={} for alpha={} with percentile={}\n",
                    ostream_double(tresid_mean_struct.resid_mean_perc),
                    ostream_double(the_alpha),
                    ostream_double(percentile as f64)
                ));
            }
            if tresid_mean_struct.resid_mean_perc < resid_mean_final {
                resid_mean_final = tresid_mean_struct.resid_mean_perc;
                *alpha_final = alpha.get_real_2d(0, k);
            }
        }
        drop(alpha);
    }
    let mut final_estimation3d: Option<Estimation3dData> = None;
    if option < 3 {
        let mut copy_prob = ProbData::copy(&the_prob);
        let fe = estimation_3d(
            &sfm_.contour_x,
            &sfm_.contour_y,
            *alpha_final,
            tilt_angles,
            w,
            h,
            percentile as f64,
            &mut copy_prob,
            &wf,
            &mf,
        );
        drop(copy_prob);

        if debug_mode {
            cout(&format!(
                "Residual ={} for alpha={}\n",
                ostream_double(fe.resid_mean),
                ostream_double(*alpha_final)
            ));
            cout(&format!(
                "Residual ={} for alpha={} with percentile={}\n",
                ostream_double(fe.resid_mean_perc),
                ostream_double(*alpha_final),
                ostream_double(percentile as f64)
            ));
        }
        // compute reprojection matrix
        let fg = fe.g.as_ref().unwrap();
        let fp = fe.p.as_ref().unwrap();
        let mut gp = zero_mat(fg.get_size().height, fp.get_size().width);
        cv_mat_mul(fg.view(), fp.view(), &mut gp);
        let mut mfhat = rep_mat(fe.t.as_ref().unwrap(), 1, m);
        let src = mfhat.clone();
        cv_add(gp.view(), src.view(), &mut mfhat);

        drop(gp);

        sfm.reproj_x = sfm_.contour_x.clone(); // just to allocate memory
        sfm.reproj_y = sfm_.contour_y.clone();

        let w_2 = w as f64 / 2.0;
        let h_2 = h as f64 / 2.0;
        for ii in 0..m {
            for jj in 0..t_ {
                sfm.reproj_x
                    .scores
                    .set_real_2d(ii, jj, mfhat.get_real_2d(jj + jj, ii) + w_2);
                sfm.reproj_y
                    .scores
                    .set_real_2d(ii, jj, mfhat.get_real_2d(jj + jj + 1, ii) + h_2);
            }
        }

        if fe.resid_mean_perc > 5.0 {
            cout(&format!(
                "WARNING: residual of percentile {} is above 5.\n",
                ostream_double(percentile as f64)
            ));
            cout(
                "That is an indication there is most likely a problem with pairwise correspondence or the marker template.\n",
            );
            cout(&format!("RAPTOR is unable to continue at {}\n", get_date()));
            exit(-1);
        }
        final_estimation3d = Some(fe);
    }
    let mut final_estimation3d = match final_estimation3d {
        Some(fe) => fe,
        None => Estimation3dData::new(),
    };

    drop(the_prob);
    cout("Finished block to estimate alpha (2D in-plane rotation)\n");

    // start factorization
    let mut mf_wf = CvMat::create(wf.get_size().height, wf.get_size().width, CV_64FC1);
    cv_mul(wf.view(), mf.view(), &mut mf_wf, 1.0);

    let mut vt: CvMat;
    let g: CvMat;
    let t: CvMat;

    let max_iter;
    let mu;
    let rand_ini;

    if option < 3 {
        max_iter = 5;
        mu = 100000.0f64;
        let fp = final_estimation3d.p.as_ref().unwrap();
        vt = CvMat::create(fp.cols, fp.rows, CV_64FC1);
        cv_transpose(fp.view(), &mut vt);
        // `G = finalEstimation3d->G; t = finalEstimation3d->t;` alias
        // matrices nothing writes to; copies here.
        g = final_estimation3d.g.as_ref().unwrap().clone_mat();
        t = final_estimation3d.t.as_ref().unwrap().clone_mat();
        rand_ini = 1;
    } else {
        max_iter = 30;
        mu = 0.0;
        g = create_rand_mat(2 * t_, 3, 0.01, -0.005);
        t = create_rand_mat(2 * t_, 1, 1.0, -0.5);
        vt = create_rand_mat(m, 3, 500.0, 0.0);
        rand_ini = 5;
    }

    // if option>=3 Ut will be created again randomly for each random Initialization
    let mut ut = concatenate_matrix(&g, &t);
    let mut ut_old: Option<CvMat> = Some(CvMat::create(
        ut.get_size().height,
        ut.get_size().width,
        CV_64FC1,
    ));
    cv_add_s(
        ut.view(),
        CvScalar::new(TOL * 10.0, 0.0, 0.0, 0.0),
        ut_old.as_mut().unwrap(),
    );

    let mut prob_u = ProbData::new();

    // ---------generate common probU->Q
    let mut tt = cs_spalloc(4 + 2 * m, 4 + 2 * m, m + 4, 1, 1);
    if mu > 0.0 {
        let _2mu = mu + mu;
        for ii in 0..3 {
            cs_entry(&mut tt, ii, ii, _2mu);
        }
        cs_entry(&mut tt, 3, 3, _2mu / 10.0);
    }
    for ii in m + 4..2 * m + 4 {
        cs_entry(&mut tt, ii, ii, 1.0);
    }
    prob_u.q = cs_compress(&tt);
    drop(tt);

    // ------------generate common probU->c
    let mut tt = cs_spalloc(4 + 2 * m, 1, m + 4, 1, 1);
    for i in 4..4 + m {
        cs_entry(&mut tt, i, 0, DELTA);
    }
    for i in 0..4 {
        cs_entry(&mut tt, i, 0, -1000.0); // to guarantee that it is there later
    }
    prob_u.c = cs_compress(&tt);
    drop(tt);

    let mut tt = cs_spalloc(2 * m, 4 + 2 * m, 12 * m, 1, 1);
    for i in 0..m {
        cs_entry(&mut tt, i, 4 + i, -1.0);
        cs_entry(&mut tt, i, 4 + m + i, 1.0);
        cs_entry(&mut tt, m + i, 4 + i, -1.0);
        cs_entry(&mut tt, m + i, 4 + m + i, -1.0);

        // to guarantee we have the space later
        cs_entry(&mut tt, i, 0, -1000.0);
        cs_entry(&mut tt, m + i, 0, -1000.0);
        cs_entry(&mut tt, i, 1, -1000.0);
        cs_entry(&mut tt, m + i, 1, -1000.0);
        cs_entry(&mut tt, i, 2, -1000.0);
        cs_entry(&mut tt, m + i, 2, -1000.0);
        cs_entry(&mut tt, i, 3, -1000.0);
        cs_entry(&mut tt, m + i, 3, -1000.0);
    }
    prob_u.a = cs_compress(&tt);
    drop(tt);

    // prepare matrices to solve for V
    let mut prob_v = ProbData::new();

    let mut tt = cs_spalloc(3 + 4 * t_, 3 + 4 * t_, 2 * t_, 1, 1);
    for i in 3 + 2 * t_..3 + 4 * t_ {
        cs_entry(&mut tt, i, i, 1.0);
    }
    prob_v.q = cs_compress(&tt);
    drop(tt);

    let mut tt = cs_spalloc(3 + 4 * t_, 1, 2 * t_, 1, 1);
    for i in 3..3 + 2 * t_ {
        cs_entry(&mut tt, i, 0, DELTA);
    }
    prob_v.c = cs_compress(&tt);
    drop(tt);

    let mut tt = cs_spalloc(4 * t_, 3 + 4 * t_, 20 * t_, 1, 1);
    for i in 0..2 * t_ {
        cs_entry(&mut tt, i, 3 + i, -1.0);
        cs_entry(&mut tt, i, 3 + 2 * t_ + i, 1.0);
        cs_entry(&mut tt, 2 * t_ + i, 3 + i, -1.0);
        cs_entry(&mut tt, 2 * t_ + i, 3 + 2 * t_ + i, -1.0);

        // to make sure those entries are found later
        cs_entry(&mut tt, i, 0, -1000.0);
        cs_entry(&mut tt, i, 1, -1000.0);
        cs_entry(&mut tt, i, 2, -1000.0);
        cs_entry(&mut tt, 2 * t_ + i, 0, -1000.0);
        cs_entry(&mut tt, 2 * t_ + i, 1, -1000.0);
        cs_entry(&mut tt, 2 * t_ + i, 2, -1000.0);
    }
    prob_v.a = cs_compress(&tt);
    drop(tt);

    // finished setting up matrices for factorization method

    let mut resid_mean_old = 1e11f64;

    let mut v = CvMat::create(m, 3, CV_64FC1);
    let mut vtt: Option<CvMat> = None;
    let mut utt: Option<CvMat> = None;
    for _kk in 1..=rand_ini {
        // random initialization
        if option >= 3 {
            final_estimation3d.resid_mean = 1e10;
            ut = create_rand_mat(2 * t_, 4, 0.01, -0.005);
        }
        let mut count = 0;

        let mut estimation3d_resid_mean_old = final_estimation3d.resid_mean + 10.0;
        while (cv_norm(ut.view(), ut_old.as_ref().map(|u| u.view()), CV_L1) > TOL)
            && (count < max_iter)
            && ((final_estimation3d.resid_mean - estimation3d_resid_mean_old).abs() > 1e-2)
        {
            estimation3d_resid_mean_old = final_estimation3d.resid_mean;
            count += 1;

            for ii in 0..m {
                // probV->buc
                let wfcol = wf.get_col(ii);
                let utcol = ut.get_col(3);
                let mf_wfcol = mf_wf.get_col(ii);

                let mut aux_buc1 = CvMat::create(wfcol.rows, wfcol.cols, CV_64FC1);
                let mut aux_buc2 = CvMat::create(wfcol.rows, wfcol.cols, CV_64FC1);

                cv_mul(wfcol, utcol, &mut aux_buc1, 1.0);
                let src = aux_buc1.clone();
                cv_sub(src.view(), mf_wfcol, &mut aux_buc1);
                cv_convert_scale(aux_buc1.view(), &mut aux_buc2, -1.0, 0.0);
                prob_v.buc = Some(concatenate_matrix_down(&aux_buc1, &aux_buc2));
                drop(aux_buc1);
                drop(aux_buc2);
                // probV.buc=[-MfWf(:,i)+Wf(:,i).*Ut(:,4); MfWf(:,i)-Wf(:,i).*Ut(:,4)];

                {
                    let a = prob_v.a.as_mut().unwrap();
                    let ax = a.x.as_mut().unwrap();
                    for jj in 0..3 {
                        for ss in a.p[jj as usize]..a.p[jj as usize + 1] {
                            let ss = ss as usize;
                            let aux_row = a.i[ss];
                            if aux_row < 2 * t_ {
                                ax[ss] =
                                    -(wf.get_real_2d(aux_row, ii) * ut.get_real_2d(aux_row, jj));
                            } else {
                                ax[ss] = wf.get_real_2d(aux_row - 2 * t_, ii)
                                    * ut.get_real_2d(aux_row - 2 * t_, jj);
                            }
                        }
                    }
                }

                // prob.x0 - generate feasible point
                let mut x0 = CvMat::create(3 + 4 * t_, 1, CV_64FC1);
                for k in 0..3 {
                    x0.set_real_2d(k, 0, 2.0 * (rand() as f64 / RAND_MAX as f64) - 1.0);
                }
                for k in 3..3 + 2 * t_ {
                    x0.set_real_2d(k, 0, 100.0);
                }
                {
                    let buc = prob_v.buc.as_ref().unwrap();
                    for k in 3 + 2 * t_..3 + 4 * t_ {
                        x0.set_real_2d(k, 0, buc.get_real_2d(k - 2 * t_ - 3, 0));
                    }
                }
                prob_v.x0 = Some(x0);

                let vstd = std_qp(
                    prob_v.q.as_ref().unwrap(),
                    prob_v.c.as_ref().unwrap(),
                    prob_v.a.as_ref().unwrap(),
                    prob_v.buc.as_ref().unwrap(),
                    prob_v.x0.as_ref().unwrap(),
                    m,
                    t_,
                    "probV",
                );
                let vans = vstd.answer_mat.as_ref().unwrap();

                let mut xxgreater1 = true;
                for k in 4..3 + 2 * t_ {
                    if vans.get_real_2d(k, 0) < -0.01 {
                        xxgreater1 = false;
                        break;
                    }
                }

                if xxgreater1 {
                    for j in 0..3 {
                        v.set_real_2d(ii, j, vans.get_real_2d(j, 0));
                    }
                } else {
                    for j in 0..3 {
                        v.set_real_2d(ii, j, vt.get_real_2d(ii, j));
                    }
                }
            }

            let mut vtr = CvMat::create(v.cols, v.rows, CV_64FC1);
            cv_transpose(v.view(), &mut vtr);
            let vtr_ones = concatenate_matrix_down(&vtr, &ones_mat(1, m));
            let mut ut_vtr_ones = CvMat::create(ut.rows, vtr_ones.cols, CV_64FC1);
            cv_mat_mul(ut.view(), vtr_ones.view(), &mut ut_vtr_ones);
            let src = ut_vtr_ones.clone();
            cv_mul(wf.view(), src.view(), &mut ut_vtr_ones, 1.0);
            let src = ut_vtr_ones.clone();
            cv_sub(mf_wf.view(), src.view(), &mut ut_vtr_ones);
            let mut r2 = CvMat::create(ut_vtr_ones.rows, ut_vtr_ones.cols, CV_64FC1);
            cv_mul(ut_vtr_ones.view(), ut_vtr_ones.view(), &mut r2, 1.0);
            drop(ut_vtr_ones);
            drop(vtr);
            drop(vtr_ones);
            // r2=(Wf.*Mf-Wf.*(Ut*[V';ones(1,M)])).^2;

            let mut r2replace = CvMat::create((r2.rows) / 2, r2.cols, CV_64FC1);
            let mut i = 0;
            while i < r2.get_size().height {
                for j in 0..r2.get_size().width {
                    r2replace.set_real_2d(
                        i / 2,
                        j,
                        (r2.get_real_2d(i, j) + r2.get_real_2d(i + 1, j)).sqrt(),
                    );
                }
                i += 2;
            }
            drop(r2);

            // r2=sqrt(r2(1:2:end,:)+r2(2:2:end,:));
            let mut mean_counter = 0.0f64;
            let mut counter = 0;
            for i in 0..r2replace.get_size().height {
                for j in 0..r2replace.get_size().width {
                    if wf.get_real_2d(i + i, j) > MIN_WEIGHT * 10.0 {
                        mean_counter += r2replace.get_real_2d(i, j);
                        counter += 1;
                    }
                }
            }
            drop(r2replace);
            let mut aa = mean_counter / counter as f64;
            // aa=mean(r2(find(r2~=0)));

            if aa <= (final_estimation3d.resid_mean + 0.1) {
                vt = v.clone_mat();
                final_estimation3d.resid_mean = aa;
                if debug_mode {
                    cout(&format!(
                        "iteration {} of the first matrix factorization; MSE(with outliers)={}\n",
                        count,
                        ostream_double(aa)
                    ));
                }
            } else {
                break;
            }

            let mut u = CvMat::create(2 * t_, 4, CV_64FC1);
            for jj in 0..2 * t_ {
                {
                    let c = prob_u.c.as_mut().unwrap();
                    let cx = c.x.as_mut().unwrap();
                    for ss in c.p[0]..c.p[1] {
                        let ss = ss as usize;
                        let aux_row = c.i[ss];
                        if aux_row < 3 {
                            cx[ss] = -2.0 * mu * g.get_real_2d(jj, aux_row);
                        } else if aux_row == 3 {
                            cx[ss] = -0.2 * mu * t.get_real_2d(jj, 0);
                        }
                    }
                }
                // probU.c(1:4)=[-2*mu*G(j,1:3)'; -2*mu*t(j)/10];

                let mf_wfrow = mf_wf.get_row(jj);

                let mut mf_wfrow_t = CvMat::create(
                    mf_wfrow.get_size().width,
                    mf_wfrow.get_size().height,
                    CV_64FC1,
                );
                let mut neg_mf_wfrow_t = CvMat::create(
                    mf_wfrow.get_size().width,
                    mf_wfrow.get_size().height,
                    CV_64FC1,
                );
                cv_transpose(mf_wfrow, &mut mf_wfrow_t);
                cv_convert_scale(mf_wfrow_t.view(), &mut neg_mf_wfrow_t, -1.0, 0.0);
                prob_u.buc = Some(concatenate_matrix_down(&neg_mf_wfrow_t, &mf_wfrow_t));
                drop(neg_mf_wfrow_t);
                drop(mf_wfrow_t);
                // probU.buc=[-MfWf(j,:)'; MfWf(j,:)'];

                {
                    let a = prob_u.a.as_mut().unwrap();
                    let ax = a.x.as_mut().unwrap();
                    for kk in 0..3 {
                        for ss in a.p[kk as usize]..a.p[kk as usize + 1] {
                            let ss = ss as usize;
                            let aux_row = a.i[ss];
                            if aux_row < m {
                                ax[ss] =
                                    -(wf.get_real_2d(jj, aux_row) * vt.get_real_2d(aux_row, kk));
                            } else {
                                ax[ss] = wf.get_real_2d(jj, aux_row - m)
                                    * vt.get_real_2d(aux_row - m, kk);
                            }
                        }
                    }
                    for ss in a.p[3]..a.p[4] {
                        let ss = ss as usize;
                        let aux_row = a.i[ss];
                        if aux_row < m {
                            ax[ss] = -(wf.get_real_2d(jj, aux_row));
                        } else {
                            ax[ss] = wf.get_real_2d(jj, aux_row - m);
                        }
                    }
                }
                // probU.a(1:M,1:4)=-repmat(Wf(j,:)',[1,4]).*[Vt ones(M,1)];
                // probU.a(M+1:2*M,1:4)=-probU.a(1:M,1:4);

                // fabricate feasible point
                let mut x0 = CvMat::create(4 + 2 * m, 1, CV_64FC1);
                for i in 0..3 {
                    x0.set_real_2d(
                        i,
                        0,
                        ut.get_real_2d(jj, i) + (0.01 * rand() as f64 / RAND_MAX as f64) - 0.005,
                    );
                }
                x0.set_real_2d(
                    3,
                    0,
                    ut.get_real_2d(jj, 3) + (10.0 * rand() as f64 / RAND_MAX as f64) - 5.0,
                );
                for i in 4..m + 4 {
                    x0.set_real_2d(i, 0, (10.0 * rand() as f64 / RAND_MAX as f64) + 1e3);
                }
                for i in 4 + m..4 + 2 * m {
                    x0.set_real_2d(i, 0, (2.0 * rand() as f64 / RAND_MAX as f64) - 1.0);
                }
                prob_u.x0 = Some(x0);

                let mut ustd = std_qp(
                    prob_u.q.as_ref().unwrap(),
                    prob_u.c.as_ref().unwrap(),
                    prob_u.a.as_ref().unwrap(),
                    prob_u.buc.as_ref().unwrap(),
                    prob_u.x0.as_ref().unwrap(),
                    m,
                    t_,
                    "probU",
                );

                if ustd.exit_flag == 2 {
                    {
                        let x0 = prob_u.x0.as_mut().unwrap();
                        for i in 4..4 + m {
                            let v0 = x0.get_real_2d(i, 0);
                            x0.set_real_2d(i, 0, v0 + 3.0 * (ustd.gap));
                        }
                    }
                    ustd = std_qp(
                        prob_u.q.as_ref().unwrap(),
                        prob_u.c.as_ref().unwrap(),
                        prob_u.a.as_ref().unwrap(),
                        prob_u.buc.as_ref().unwrap(),
                        prob_u.x0.as_ref().unwrap(),
                        m,
                        t_,
                        "probU",
                    );
                }
                let uans = ustd.answer_mat.as_ref().unwrap();

                let mut xxgreater1 = true;
                for k in 4..4 + m {
                    if uans.get_real_2d(k, 0) < -0.01 {
                        xxgreater1 = false;
                        break;
                    }
                }
                if xxgreater1 {
                    for j in 0..4 {
                        u.set_real_2d(jj, j, uans.get_real_2d(j, 0));
                    }
                } else {
                    for j in 0..4 {
                        u.set_real_2d(jj, j, ut.get_real_2d(jj, j));
                    }
                }
            }

            let mut vtr = CvMat::create(v.cols, v.rows, CV_64FC1);
            cv_transpose(v.view(), &mut vtr);
            let vtr_ones = concatenate_matrix_down(&vtr, &ones_mat(1, m));
            let mut ut_vtr_ones = CvMat::create(u.rows, vtr_ones.cols, CV_64FC1);
            cv_mat_mul(u.view(), vtr_ones.view(), &mut ut_vtr_ones);
            let src = ut_vtr_ones.clone();
            cv_mul(wf.view(), src.view(), &mut ut_vtr_ones, 1.0);
            let src = ut_vtr_ones.clone();
            cv_sub(mf_wf.view(), src.view(), &mut ut_vtr_ones);
            let mut r2 = CvMat::create(ut_vtr_ones.rows, ut_vtr_ones.cols, CV_64FC1);
            cv_mul(ut_vtr_ones.view(), ut_vtr_ones.view(), &mut r2, 1.0);
            drop(vtr);
            drop(vtr_ones);
            drop(ut_vtr_ones);

            let mut r2replace = CvMat::create((r2.rows) / 2, r2.cols, CV_64FC1);
            let mut i = 0;
            while i < r2.get_size().height {
                for j in 0..r2.get_size().width {
                    r2replace.set_real_2d(
                        i / 2,
                        j,
                        (r2.get_real_2d(i, j) + r2.get_real_2d(i + 1, j)).sqrt(),
                    );
                }
                i += 2;
            }
            drop(r2);

            let mut mean_counter2 = 0.0f64;
            let mut counter2 = 0;
            for i in 0..r2replace.get_size().height {
                for j in 0..r2replace.get_size().width {
                    if wf.get_real_2d(i + i, j) > MIN_WEIGHT * 10.0 {
                        mean_counter2 += r2replace.get_real_2d(i, j);
                        counter2 += 1;
                    }
                }
            }
            drop(r2replace);
            aa = mean_counter2 / counter2 as f64;

            if aa <= (final_estimation3d.resid_mean + 0.1) {
                ut_old = Some(ut.clone_mat());
                ut = u.clone_mat();
                drop(u);
                final_estimation3d.resid_mean = aa;
                if debug_mode {
                    cout(&format!(
                        "iteration {} of the second matrix factorization; MSE(with outliers)={}\n",
                        count,
                        ostream_double(aa)
                    ));
                }
            } else {
                drop(u);
                ut_old = None;
                // factorization finished
                break;
            }
        }
        if final_estimation3d.resid_mean < resid_mean_old {
            resid_mean_old = final_estimation3d.resid_mean;

            utt = Some(ut.clone_mat());
            vtt = Some(vt.clone_mat());
            // to speed-up the process
            if resid_mean_old < 1.0 {
                break;
            }
        }
    }

    final_estimation3d.resid_mean = resid_mean_old;
    drop(prob_v);
    drop(prob_u);
    drop(ut_old);

    cout(&format!(
        "Matrix factorization; MSE(with outliers)={}\n",
        ostream_double(final_estimation3d.resid_mean)
    ));

    let vtt = vtt.expect("SFMestimationWithBA: no factorization was kept (Vtt NULL)");
    let utt = utt.expect("SFMestimationWithBA: no factorization was kept (Utt NULL)");
    let mut vtt_t = CvMat::create(vtt.cols, vtt.rows, CV_64FC1);
    cv_transpose(vtt.view(), &mut vtt_t);
    let reproj = concatenate_matrix_down(&vtt_t, &ones_mat(1, m));
    drop(vtt_t);
    let mut mf_hat = CvMat::create(utt.rows, reproj.cols, CV_64FC1);
    cv_mat_mul(utt.view(), reproj.view(), &mut mf_hat);
    drop(reproj);

    sfm.reproj_x = sfm_.contour_x.clone(); // just to allocate memory
    sfm.reproj_y = sfm_.contour_y.clone();

    let mut pos = 0usize;
    for ii in 0..m {
        for jj in 0..t_ {
            sfm.reproj_x.scores.data[pos] = mf_hat.get_real_2d(jj + jj, ii) + w_2;
            sfm.reproj_y.scores.data[pos] = mf_hat.get_real_2d(jj + jj + 1, ii) + h_2;
            pos += 1;
        }
    }

    drop(sfm_);
    sfm
}

/// `residAnalysis(SFMdata* sfm, bool debugMode)`
/// (`SFMestimationWithBA.cpp:1205`): analysis of the contours to detect and
/// remove possible outliers.  `sfm` is freed inside.
pub fn resid_analysis(mut sfm: SfmData, debug_mode: bool) -> SfmData {
    // value of the residual to find teh kink in the resid curve (in pixels)
    let kink_thr = 3.0f64;
    // any value above maxResid is consider an Outlier automatically
    let max_resid = 20.0f64;

    // useful variables
    let t_ = sfm.contour_x.scores.cols;
    let m = sfm.contour_x.scores.rows;

    // compute residuals
    let mut resid = CvMat::create(m, t_, CV_64FC1);
    let mut resid_aux = CvMat::create(m, t_, CV_64FC1);
    cv_sub(
        sfm.contour_x.scores.view(),
        sfm.reproj_x.scores.view(),
        &mut resid,
    );
    let src = resid.clone();
    cv_mul(src.view(), src.view(), &mut resid, 1.0);
    cv_sub(
        sfm.contour_y.scores.view(),
        sfm.reproj_y.scores.view(),
        &mut resid_aux,
    );
    let src = resid_aux.clone();
    cv_mul(src.view(), src.view(), &mut resid_aux, 1.0);
    let src = resid.clone();
    cv_add(src.view(), resid_aux.view(), &mut resid);
    drop(resid_aux);
    let src = resid.clone();
    cv_pow(src.view(), &mut resid, 0.5);

    let mut total_num_outliers = 0;
    let mut aux_r: Vec<f64> = Vec::with_capacity(t_ as usize);
    // vector containing which contours need to be kept after detecting outliers
    let mut keep_contour: Vec<i32> = Vec::new();
    // analyze each trajectory
    for ii in 0..m {
        aux_r.clear();
        let mut pos = (ii * t_) as usize;
        for _jj in 0..t_ {
            // find positions where residual is too big and set them to
            // negative number so it won't affect the outlier detection in
            // case there are too many residuals
            if sfm.contour_x.scores.data[pos] > 1e-6 && resid.data[pos] < max_resid {
                aux_r.push(resid.data[pos]);
            }
            pos += 1;
        }

        // build linear programming to detect outliers
        let mut thr = 1e20f64;
        let l = aux_r.len() as i32;
        if l > 0 {
            std_sort(&mut aux_r, &mut |a: &f64, b: &f64| a < b);

            // create c
            let mut tt = cs_spalloc(l + 2, 1, l, 1, 1);
            for kk in 2..l + 2 {
                cs_entry(&mut tt, kk, 0, 1.0);
            }
            let c = cs_compress(&tt).unwrap();
            drop(tt);

            // create A
            let mut tt = cs_spalloc(l + l, l + 2, 6 * l, 1, 1);
            for kk in 0..l {
                cs_entry(&mut tt, kk, 0, (-kk - 1) as f64);
                cs_entry(&mut tt, kk + l, 0, (kk + 1) as f64);
                cs_entry(&mut tt, kk, 1, -1.0);
                cs_entry(&mut tt, kk + l, 1, 1.0);
                cs_entry(&mut tt, kk, kk + 2, -1.0);
                cs_entry(&mut tt, kk + l, kk + 2, -1.0);
            }
            let a = cs_compress(&tt).unwrap();
            drop(tt);

            // create b
            let mut b = CvMat::create(l + l, 1, CV_64FC1);
            for kk in 0..l as usize {
                b.data[kk] = -aux_r[kk];
                b.data[kk + l as usize] = aux_r[kk];
            }

            // create x0 as feasible point
            let mut xx = CvMat::create(2 + l, 1, CV_64FC1);
            xx.data[0] = 0.0;
            xx.data[1] = 0.0;
            for kk in 0..l as usize {
                xx.data[kk + 2] = aux_r[kk] + 1.0;
            }

            // solve robust linear regression using L1 norm to find outliers
            lp_solver(&c, &a, &b, &mut xx);

            // find threshold in residual
            for kk in 0..l as usize {
                if (aux_r[kk] - xx.data[0] * (kk + 1) as f64 - xx.data[1]) > kink_thr {
                    // `thr=min(thr,auxR[kk])`: std::min keeps `thr` unless
                    // `auxR[kk] < thr`
                    if aux_r[kk] < thr {
                        thr = aux_r[kk];
                    }
                }
            }
        }
        // mark all the outliers in this trajectory
        let mut num_outliers = 0;
        let mut pos = (ii * t_) as usize;
        for _jj in 0..t_ {
            if sfm.contour_x.scores.data[pos] > 1e-6
                && (resid.data[pos] > max_resid || resid.data[pos] >= thr)
            {
                // set contours to zero so they can be restimated using fillContours
                sfm.contour_x.scores.data[pos] = 0.0;
                sfm.contour_y.scores.data[pos] = 0.0;
                num_outliers += 1;
            }
            pos += 1;
        }
        if debug_mode {
            cout(&format!(
                "Detected {} outliers in trajectory {} with thr={}\n",
                num_outliers,
                ii,
                ostream_double(thr)
            ));
        }

        total_num_outliers += num_outliers;
        // if trajectory has 10% of outliers or more it should be eliminated
        if (num_outliers as f64) / (t_ as f64) < 0.1 {
            keep_contour.push(ii);
        }
    }

    cout(&format!("There are {} outliers\n", total_num_outliers));

    // redo sfm structure to delete trajectories with too many outliers
    if keep_contour.len() as i32 != m {
        let nk = keep_contour.len() as i32;
        let mut c_x = CvMat::create(nk, t_, CV_64FC1);
        let mut r_x = CvMat::create(nk, t_, CV_64FC1);
        let mut c_y = CvMat::create(nk, t_, CV_64FC1);
        let mut r_y = CvMat::create(nk, t_, CV_64FC1);
        let mut m_type: Vec<i32> = vec![0; keep_contour.len()];
        let mut pos = 0usize;
        for ii in 0..keep_contour.len() {
            let mut pos2 = (keep_contour[ii] * t_) as usize;
            for _jj in 0..t_ {
                c_x.data[pos] = sfm.contour_x.scores.data[pos2];
                r_x.data[pos] = sfm.reproj_x.scores.data[pos2];
                c_y.data[pos] = sfm.contour_y.scores.data[pos2];
                r_y.data[pos] = sfm.reproj_y.scores.data[pos2];
                pos2 += 1;
                pos += 1;
            }
            m_type[ii] = sfm.marker_type[keep_contour[ii] as usize];
        }

        drop(sfm);
        SfmData::with_reprojection(
            &Contour::from_mat(&c_x, 1),
            &Contour::from_mat(&c_y, 2),
            &Contour::from_mat(&r_x, 1),
            &Contour::from_mat(&r_y, 2),
            &m_type,
        )
    } else {
        sfm.clone()
    }
}

/// `decideTiltAlignOptions(SFMdata* sfm, int* tiltOption, int* rotOption,
/// int* magOption)` (`SFMestimationWithBA.cpp:1384`): decide if we have
/// enough markers to adjust all parameters or we need to fix some
/// parameters.
pub fn decide_tilt_align_options(
    sfm: &SfmData,
    tilt_option: &mut i32,
    rot_option: &mut i32,
    mag_option: &mut i32,
) {
    let m = sfm.contour_x.scores.rows;
    let t_ = sfm.contour_x.scores.cols;

    let mut min_num_markers = 10000;
    // we count number of markers per frame
    let data = &sfm.contour_x.scores.data;
    for jj in 0..t_ {
        // count number of nonzeros per frame
        let mut count = 0;
        let mut pos = jj;
        for _ii in 0..m {
            if data[pos as usize] > 1e-6 {
                count += 1;
            }
            pos += t_;
        }
        min_num_markers = min_num_markers.min(count);
    }
    match min_num_markers {
        1 | 2 | 3 => {
            // we fix everything except tilt angle
            *tilt_option = 5;
            *rot_option = 0;
            *mag_option = 0;
        }
        4 | 5 => {
            // we fix magnification option
            *tilt_option = 5;
            *rot_option = 3;
            *mag_option = 0;
        }
        _ => {
            // everything is grouped
            *tilt_option = 5;
            *rot_option = 3;
            *mag_option = 3;
        }
    }
}

/// `decideAlphaOption(SFMdata* sfm, int* optionAlpha, vector<frame>* frames)`
/// (`SFMestimationWithBA.cpp:1448`).  `sfm` is freed inside.  The source's
/// `list<int>` of columns to discard is sorted and made unique before use;
/// for `int`s any sort gives the same list.
pub fn decide_alpha_option(sfm: SfmData, option_alpha: &mut i32, frames: &mut [Frame]) -> SfmData {
    let m = sfm.contour_x.scores.rows;
    let t_ = sfm.contour_x.scores.cols;

    // map between frames and contours
    let mut discard_column: Vec<i32> = Vec::new();
    let mut map: Vec<i32> = vec![0; t_ as usize];
    let mut count = 0i32;
    for kk in 0..frames.len() {
        if !frames[kk].discard {
            map[count as usize] = kk as i32;
            count += 1;
        }
    }

    let mut flag_return = false;

    if *option_alpha == 1 || *option_alpha == 2 {
        flag_return = true;
    }

    // decide if we have enough markers to do just factorization without
    // regularization; we also decide if we have enough markers to continue

    // last condition
    let data = &sfm.contour_x.scores.data;
    for jj in 0..t_ {
        // count number of nonzeros per frame
        count = 0;
        let mut pos = jj;
        for _ii in 0..m {
            if data[pos as usize] > 1e-6 {
                count += 1;
            }
            pos += t_;
        }
        // we want to use optionAlpha=0 all the time (it is faster than factorization) UV
        if count < 100 && !flag_return {
            *option_alpha = 0;
            flag_return = true;
        }
        // check if we have a minimum of contours to continue
        if count <= 2 {
            cout(&format!(
                "WARNING: projection number {} (first projection is 0) contains only {}detected markers\n",
                jj, count
            ));
            cout("RAPTOR is ignoring projections on this side until the end of the tilt series\n");
            discard_column.push(jj);
        }
    }
    // second condition
    for jj in 0..t_ - 1 {
        // count number of nonzeros per pairs of frames
        count = 0;
        let mut pos = jj;
        for _ii in 0..m {
            if data[pos as usize] > 1e-6 && data[pos as usize + 1] > 1e-6 {
                count += 1;
            }
            pos += t_;
        }
        if count < 4 && !flag_return {
            *option_alpha = 0;
            flag_return = true;
        }
        // check if we have a minimum of contours to continue
        if count <= 1 {
            cout(&format!(
                "WARNING: projections number {}->{} (first projection is 0) contains only {}common detected markers\n",
                jj,
                jj + 1,
                count
            ));
            cout("RAPTOR is ignoring projections on this side until the end of the tilt series\n");
            discard_column.push(jj);
        }
    }

    // third condition
    if sfm.contour_x.calculate_nnz() < 3 * (8 * t_ + 3 * m) && !flag_return {
        *option_alpha = 0;
        flag_return = true;
    }

    // if the code made it here it means we can use just factorization
    // without regularization
    if *option_alpha == 4 && !flag_return {
        *option_alpha = 3;
    } else if !flag_return {
        *option_alpha = 4;
    }

    // decide if we have to erase some columns
    let new_sfm;
    if discard_column.is_empty() {
        new_sfm = sfm.clone();
    } else {
        std_sort(&mut discard_column, &mut |a: &i32, b: &i32| a < b);
        discard_column.dedup();
        let ll = discard_column.len() as i32;
        let mut mask: Vec<bool> = vec![false; t_ as usize];
        for kk in 0..t_ {
            if discard_column.binary_search(&kk).is_ok() {
                mask[kk as usize] = false; // remove this element
            } else {
                mask[kk as usize] = true;
            }
        }

        let mut c_x = CvMat::create(m, t_ - ll, CV_64FC1);
        let mut r_x = CvMat::create(m, t_ - ll, CV_64FC1);
        let mut c_y = CvMat::create(m, t_ - ll, CV_64FC1);
        let mut r_y = CvMat::create(m, t_ - ll, CV_64FC1);
        let mut pos = 0usize;
        let mut pos2 = 0usize;
        for _ii in 0..m {
            for jj in 0..t_ as usize {
                if mask[jj] {
                    c_x.data[pos] = sfm.contour_x.scores.data[pos2];
                    r_x.data[pos] = sfm.reproj_x.scores.data[pos2];
                    c_y.data[pos] = sfm.contour_y.scores.data[pos2];
                    r_y.data[pos] = sfm.reproj_y.scores.data[pos2];
                    pos += 1;
                }
                pos2 += 1;
            }
        }
        new_sfm = SfmData::with_reprojection(
            &Contour::from_mat(&c_x, 1),
            &Contour::from_mat(&c_y, 2),
            &Contour::from_mat(&r_x, 1),
            &Contour::from_mat(&r_y, 2),
            &sfm.marker_type,
        );
        // update frames information
        for iter in &discard_column {
            frames[map[*iter as usize] as usize].discard = true;
        }
    }

    drop(sfm);
    new_sfm
}

/// `writeIMODfidModel(SFMdata* sfm, int width, int height, int num_frames,
/// string basename, ostream& out, vector<frame>* frames)`
/// (`SFMestimationWithBA.cpp:1610`): create trajectory vector and then call
/// the original function.  `points` is the arena the temporary
/// trajectories are built in.
#[allow(clippy::too_many_arguments)]
pub fn write_imod_fid_model_sfm(
    sfm: &SfmData,
    width: i32,
    height: i32,
    num_frames: i32,
    basename: &str,
    out: &mut Vec<u8>,
    frames: &[Frame],
    points: &mut Vec<Point2D>,
) {
    let t = sfm
        .contour_x
        .contour2trajectory(&sfm.contour_x, &sfm.contour_y, frames, points);
    write_imod_fid_model(&t, width, height, num_frames, basename, out, points);
    // free memory appropiately
    free_trajectory_vector(&t, points);
}

/// `writeIMODfidModelReproj(SFMdata* sfm, int width, int height, int
/// num_frames, string basename, ostream& out, vector<frame>* frames)`
/// (`SFMestimationWithBA.cpp:1620`).
#[allow(clippy::too_many_arguments)]
pub fn write_imod_fid_model_reproj(
    sfm: &SfmData,
    width: i32,
    height: i32,
    num_frames: i32,
    basename: &str,
    out: &mut Vec<u8>,
    frames: &[Frame],
    points: &mut Vec<Point2D>,
) {
    let t = sfm
        .contour_x
        .contour2trajectory(&sfm.reproj_x, &sfm.reproj_y, frames, points);
    write_imod_fid_model(&t, width, height, num_frames, basename, out, points);
    // free memory appropiately
    free_trajectory_vector(&t, points);
}
