//! Translation of `IMOD/raptor/optimization/estimation3d.h` and
//! `estimation3d.cpp`: fit the 3D projection model for one in-plane
//! rotation `alpha`.

use super::contour::Contour;
use super::estimation3ddata::Estimation3dData;
use super::prob_data::ProbData;
use super::std_qp::{std_qp, zero_mat};
use crate::imod::c_sort::std_sort;
use crate::imod::libcfshr::b3dutil::{RAND_MAX, rand};
use crate::imod::raptor::main_classes::constants::{MIN_WEIGHT, PI};
use crate::imod::raptor::opencv::cxarithm::{cv_mul, cv_sub};
use crate::imod::raptor::opencv::cxmatmul::cv_mat_mul;
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat};
use crate::imod::raptor::suitesparse::cs_add::cs_add;
use crate::imod::raptor::suitesparse::cs_compress::cs_compress;
use crate::imod::raptor::suitesparse::cs_entry::cs_entry;
use crate::imod::raptor::suitesparse::cs_util::cs_spalloc;

/// `CreateRandMat(int x, int y, double scale, double shift)`
/// (`estimation3d.cpp:21`): one `rand()` per element, row by row.
pub fn create_rand_mat(x: i32, y: i32, scale: f64, shift: f64) -> CvMat {
    let mut result = CvMat::create(x, y, CV_64FC1);
    for i in 0..x {
        for j in 0..y {
            result.set_real_2d(i, j, scale * (rand() as f64 / RAND_MAX as f64) + shift);
        }
    }
    result
}

/// `repMat(CvMat* mat, int m, int n)` (`estimation3d.cpp:35`).
pub fn rep_mat(mat: &CvMat, m: i32, n: i32) -> CvMat {
    let mut result = CvMat::create(
        m * mat.get_size().height,
        n * mat.get_size().width,
        CV_64FC1,
    );
    for i in 0..m {
        for j in 0..n {
            for si in 0..mat.get_size().height {
                for sj in 0..mat.get_size().width {
                    result.set_real_2d(
                        (i * mat.get_size().height) + si,
                        (j * mat.get_size().width) + sj,
                        mat.get_real_2d(si, sj),
                    );
                }
            }
        }
    }
    result
}

/// `estimation3D(contour& contour_x, contour& contour_y, double alpha,
/// vector<double> tiltAngles, int W, int H, double percentile, probData*
/// prob, CvMat* Wf, CvMat* Mf)` (`estimation3d.cpp:55`).  `W`, `H` and
/// `contour_y` are not read by the source either.
///
/// Kept as native (BUGS.md): the last block of the starting point is
/// `buc + 2*(rand()/RAND_MAX) - 1`, an integer division that is 0 unless
/// `rand()` returns `RAND_MAX`; the `rand()` call is still made.
#[allow(clippy::too_many_arguments)]
pub fn estimation_3d(
    contour_x: &Contour,
    _contour_y: &Contour,
    alpha: f64,
    tilt_angles: &[f64],
    _w: i32,
    _h: i32,
    percentile: f64,
    prob: &mut ProbData,
    wf: &CvMat,
    mf: &CvMat,
) -> Estimation3dData {
    let t_ = contour_x.get_num_frame();
    let m = contour_x.get_num_traj();

    let mut the_estimate = Estimation3dData::with_size(t_, m);

    // create 3d rotation matrix
    let mut r = zero_mat(2 * t_, 3);
    let mut i = 0;
    while i < r.get_size().height {
        r.set_real_2d(i, 0, 1.0);
        i += 2;
    }
    let mut i = 1;
    while i < r.get_size().height {
        r.set_real_2d(
            i,
            1,
            (tilt_angles[((i - 1) / 2) as usize] * PI / 180.0).cos(),
        );
        i += 2;
    }
    let mut i = 1;
    while i < r.get_size().height {
        r.set_real_2d(
            i,
            2,
            (tilt_angles[((i - 1) / 2) as usize] * PI / 180.0).sin(),
        );
        i += 2;
    }

    let mut ralpha = zero_mat(2, 2);
    ralpha.set_real_2d(0, 0, (alpha * PI / 180.0).cos());
    ralpha.set_real_2d(0, 1, -(alpha * PI / 180.0).sin());
    ralpha.set_real_2d(1, 0, (alpha * PI / 180.0).sin());
    ralpha.set_real_2d(1, 1, (alpha * PI / 180.0).cos());

    {
        let g = the_estimate.g.as_mut().unwrap();
        for j in 0..t_ {
            let mut ralpha_rsub = zero_mat(2, 3);
            let rsub = r.get_rows(2 * j, 2 * j + 1 + 1, 1);
            cv_mat_mul(ralpha.view(), rsub, &mut ralpha_rsub);
            for kk in 0..3 {
                g.set_real_2d(2 * j, kk, ralpha_rsub.get_real_2d(0, kk));
                g.set_real_2d(2 * j + 1, kk, ralpha_rsub.get_real_2d(1, kk));
            }
        }
    }
    drop(r);
    drop(ralpha);

    // updating parts of prob.a that require G
    {
        let g = the_estimate.g.as_ref().unwrap();
        let a = prob.a.as_mut().unwrap();
        let ax = a.x.as_mut().unwrap();
        for j in 2 * t_..2 * t_ + 3 * m {
            let col_g = (j - 2 * t_) % 3;
            for ii in a.p[j as usize]..a.p[j as usize + 1] {
                let ii = ii as usize;
                ax[ii] *= g.get_real_2d(a.i[ii] % (2 * t_), col_g);
            }
        }
    }

    let mut x0 = create_rand_mat(4 * t_ * m + 2 * t_ + 3 * m, 1, 2.0, -1.0);
    // prob.x0=2*rand(4*T*M+2*T+3*M,1)-1;

    for i in 0..2 * t_ {
        let v = 10.0 * x0.get_real_2d(i, 0);
        x0.set_real_2d(i, 0, v);
    }

    for i in 2 * t_ + 3 * m..2 * t_ + 3 * m + 2 * t_ * m {
        x0.set_real_2d(i, 0, (rand() as f64 / RAND_MAX as f64) * 3.0 + 100.0);
    }

    {
        let buc = prob.buc.as_ref().unwrap();
        for i in 2 * t_ * m + 3 * m + 2 * t_..4 * t_ * m + 2 * t_ + 3 * m {
            let q = 2 * (rand() / RAND_MAX);
            x0.set_real_2d(
                i,
                0,
                buc.get_real_2d(i - (2 * t_ * m + 3 * m + 2 * t_), 0) + q as f64 - 1.0,
            );
        }
    }
    prob.x0 = Some(x0);

    // regularization on the translation parameter
    {
        let q = prob.q.as_ref().unwrap();
        let mut tt = cs_spalloc(q.m, q.n, 2 * t_, 1, 1);
        for kk in 0..2 * t_ {
            cs_entry(&mut tt, kk, kk, 1e-6);
        }
        let ttt = cs_compress(&tt).unwrap();
        let sum = cs_add(q, &ttt, 1.0, 1.0);
        prob.q = sum;
    }

    // [xx,itersQP,exit_flag]
    let the_result = std_qp(
        prob.q.as_ref().unwrap(),
        prob.c.as_ref().unwrap(),
        prob.a.as_ref().unwrap(),
        prob.buc.as_ref().unwrap(),
        prob.x0.as_ref().unwrap(),
        m,
        t_,
        "prob",
    );

    if the_result.num_itrs > 200 {
        the_estimate.resid_mean_perc = 1e6;
        the_estimate.resid_mean = 1e6;
        return the_estimate;
    }

    let answer = the_result.answer_mat.as_ref().unwrap();
    {
        let t = the_estimate.t.as_mut().unwrap();
        for i in 0..2 * t_ {
            t.set_real_2d(i, 0, answer.get_real_2d(i, 0));
        }
    }
    {
        let p = the_estimate.p.as_mut().unwrap();
        let mut i = 2 * t_;
        let mut pos_p = 0;
        while i < 2 * t_ + 3 * m {
            p.set_real_2d(0, pos_p, answer.get_real_2d(i, 0));
            p.set_real_2d(1, pos_p, answer.get_real_2d(i + 1, 0));
            p.set_real_2d(2, pos_p, answer.get_real_2d(i + 2, 0));
            i += 3;
            pos_p += 1;
        }
    }

    let g = the_estimate.g.as_ref().unwrap();
    let p = the_estimate.p.as_ref().unwrap();
    let mut gp = zero_mat(g.get_size().height, p.get_size().width);
    cv_mat_mul(g.view(), p.view(), &mut gp);
    let rept = rep_mat(the_estimate.t.as_ref().unwrap(), 1, m);

    let mut mf_gp = zero_mat(mf.get_size().height, mf.get_size().width);
    cv_sub(mf.view(), gp.view(), &mut mf_gp);
    let mut mf_gprept = zero_mat(mf_gp.get_size().height, mf_gp.get_size().width);
    cv_sub(mf_gp.view(), rept.view(), &mut mf_gprept);
    let mut resid = zero_mat(mf_gprept.get_size().height, mf_gprept.get_size().width);
    cv_mul(mf_gprept.view(), mf_gprept.view(), &mut resid, 1.0);
    // resid=(Mf-G*P-repmat(t,[1,M])).^2;

    drop(gp);
    drop(rept);
    drop(mf_gp);
    drop(mf_gprept);

    let mut resid_aux: Vec<f64> = Vec::new(); // we compute residual here
    let mut i = 0;
    while i < wf.get_size().height {
        for j in 0..wf.get_size().width {
            if wf.get_real_2d(i, j) > (MIN_WEIGHT * 1.1) {
                resid_aux.push((resid.get_real_2d(i, j) + resid.get_real_2d(i + 1, j)).sqrt());
            }
        }
        i += 2;
    }
    drop(resid);

    std_sort(&mut resid_aux, &mut |a: &f64, b: &f64| a < b);

    if resid_aux.is_empty() {
        the_estimate.resid_mean = 0.0;
        the_estimate.resid_mean_perc = 0.0;
    } else {
        let size = (resid_aux.len() as f64 * percentile).ceil() as i32;
        let mut sum = 0.0f64;
        for i in 0..size as usize {
            sum += resid_aux[i];
        }
        the_estimate.resid_mean_perc = sum / size as f64;
        for i in size as usize..resid_aux.len() {
            sum += resid_aux[i];
        }
        the_estimate.resid_mean = sum / resid_aux.len() as f64;
    }

    the_estimate
}
