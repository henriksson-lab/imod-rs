//! Translation of `IMOD/raptor/optimization/std_qp.h` and `std_qp.cpp`: the
//! log-barrier interior-point QP (`std_qp`) and LP (`LPsolver`) solvers and
//! their dense/sparse matrix utilities.
//!
//! OpenCV calls whose source and destination are the same matrix in C++
//! (`cvConvertScale(x, x, ..)`, `cvSub(b, w, w)`, `cvScaleAdd(dx, t2, x,
//! x)`) take a copy of the aliased source first; OpenCV's element loops read
//! each source element before writing it, so the result is the same.
//! The two `print` overloads (debug output behind comments) are not reached.

use super::stdqp_data::StdQpData;
use crate::imod::cxx_stream::cout;
use crate::imod::raptor::main_classes::constants::SPARSE_MATRIX_ZERO_VAL;
use crate::imod::raptor::opencv::cxarithm::{cv_add, cv_sub};
use crate::imod::raptor::opencv::cxconvert::cv_convert_scale;
use crate::imod::raptor::opencv::cxmatmul::{cv_dot_product, cv_scale_add};
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat, CvScalar};
use crate::imod::raptor::suitesparse::cs::Cs;
use crate::imod::raptor::suitesparse::cs_add::cs_add;
use crate::imod::raptor::suitesparse::cs_cholsol::cs_cholsol;
use crate::imod::raptor::suitesparse::cs_compress::cs_compress;
use crate::imod::raptor::suitesparse::cs_droptol::cs_droptol;
use crate::imod::raptor::suitesparse::cs_entry::cs_entry;
use crate::imod::raptor::suitesparse::cs_gaxpy::cs_gaxpy;
use crate::imod::raptor::suitesparse::cs_multiply::cs_multiply;
use crate::imod::raptor::suitesparse::cs_transpose::cs_transpose;
use crate::imod::raptor::suitesparse::cs_util::cs_spalloc;

/// `ZeroMat(int x, int y)` (`std_qp.cpp:20`).
pub fn zero_mat(x: i32, y: i32) -> CvMat {
    let mut result = CvMat::create(x, y, CV_64FC1);
    result.set_zero();
    result
}

/// `MinElem(CvMat* mat)` (`std_qp.cpp:28`).
pub fn min_elem(mat: &CvMat) -> f64 {
    let mut min = mat.get_real_2d(0, 0);
    for i in 0..mat.get_size().height {
        for j in 0..mat.get_size().width {
            if mat.get_real_2d(i, j) < min {
                min = mat.get_real_2d(i, j);
            }
        }
    }
    min
}

/// `Combinedx1dx2(CvMat* dx1, CvMat* dx2)` (`std_qp.cpp:42`).
pub fn combinedx1dx2(dx1: &CvMat, dx2: &CvMat) -> CvMat {
    assert!(dx1.get_size().width == dx2.get_size().width);
    let width = dx2.get_size().width;
    let height = dx1.get_size().height + dx2.get_size().height;
    let mut result = CvMat::create(height, width, CV_64FC1);
    for i in 0..dx1.get_size().height {
        for j in 0..dx1.get_size().width {
            result.set_real_2d(i, j, dx1.get_real_2d(i, j));
        }
    }
    for i in 0..dx2.get_size().height {
        for j in 0..dx2.get_size().width {
            result.set_real_2d(i + dx1.get_size().height, j, dx2.get_real_2d(i, j));
        }
    }
    result
}

/// `sumLog(CvMat* mat)` (`std_qp.cpp:67`).
pub fn sum_log(mat: &CvMat) -> f64 {
    assert!(mat.get_size().width == 1);
    let mut answer = 0.0f64;
    for i in 0..mat.get_size().height {
        for j in 0..mat.get_size().width {
            let mut elem = mat.get_real_2d(i, j);
            elem = elem.ln();
            answer += elem;
        }
    }
    answer
}

/// `std_qp(cs* _Q, cs* _c, cs* _A, CvMat* b, CvMat* x0, int M, int T, string
/// option)` (`std_qp.cpp:85`).
#[allow(clippy::too_many_arguments)]
pub fn std_qp(
    q_in: &Cs,
    c_in: &Cs,
    a_in: &Cs,
    b: &CvMat,
    x0: &CvMat,
    m: i32,
    t_: i32,
    option: &str,
) -> StdQpData {
    // make sure sparse matrices are column compress format
    let q_owned;
    let q: &Cs = if q_in.nz != -1 {
        q_owned = cs_compress(q_in).expect("std_qp: cs_compress Q");
        &q_owned
    } else {
        q_in
    };
    let c_owned;
    let c: &Cs = if c_in.nz != -1 {
        c_owned = cs_compress(c_in).expect("std_qp: cs_compress c");
        &c_owned
    } else {
        c_in
    };
    let a_owned;
    let a: &Cs = if a_in.nz != -1 {
        a_owned = cs_compress(a_in).expect("std_qp: cs_compress A");
        &a_owned
    } else {
        a_in
    };

    // initialize result struct
    let mut the_result = StdQpData::new();

    // initialize constants
    let mut answer_mat = x0.clone_mat();
    let alpha = 0.25f64;
    let beta = 0.5f64;
    let mut tol1 = 1e-3f64;
    let tol2 = 1e-3f64;
    let newton_maxiters = 20;
    let qp_maxiters = 200;
    let mut exit_flag = 0;
    let mu = 10.0f64;
    let mut t = 1.0 / mu;
    let n = answer_mat.get_size().height as f64;
    let mut iters = 1;

    let mut b_ax = b.clone_mat();
    let src = b_ax.clone();
    cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
    cs_gaxpy(a, &answer_mat.data, &mut b_ax.data);
    let src = b_ax.clone();
    cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);

    let min_value = min_elem(&b_ax);

    // check unfeasible starting point
    if min_value < 0.0 {
        if option != "probU" {
            cout(&format!(
                "WARNING: Infeasible starting point for QP solving {}\n",
                option
            ));
        }
        the_result.answer_mat = Some(answer_mat);
        the_result.exit_flag = 2;
        the_result.gap = -1.0 * min_value;
        return the_result;
    }

    while (n / t > tol1) && (exit_flag == 0) && (iters < qp_maxiters) {
        t *= mu;

        for _iter in 1..=newton_maxiters {
            iters += 1;
            the_result.num_itrs = iters;

            // compute gradient and Hessian
            let bh = b_ax.get_size().height;
            let mut tt = cs_spalloc(bh, bh, bh, 1, 1);
            for ii in 0..bh {
                cs_entry(&mut tt, ii, ii, 1.0 / (b_ax.get_real_2d(ii, 0)));
            }
            let diag_m = cs_compress(&tt).unwrap();
            drop(tt);

            let tt = cs_multiply(&diag_m, a).unwrap();
            drop(diag_m);
            let aux_op = cs_transpose(&tt, 1).unwrap();
            let aux_op2 = cs_multiply(&aux_op, &tt).unwrap();
            let mut h = cs_add(&aux_op2, q, 1.0, t).unwrap();
            drop(tt);
            drop(aux_op);
            drop(aux_op2);
            // drop small entries
            cs_droptol(&mut h, SPARSE_MATRIX_ZERO_VAL);

            let mut g = CvMat::create(h.m, 1, CV_64FC1);
            g.set_zero();
            cs_gaxpy(q, &answer_mat.data, &mut g.data);
            let aux_c = [1.0f64];
            cs_gaxpy(c, &aux_c, &mut g.data);
            for ii in 0..g.rows as usize {
                g.data[ii] *= t;
            }

            let mut b_ax_inv: Vec<f64> = vec![0.0; a.m as usize];
            for ii in 0..a.m {
                b_ax_inv[ii as usize] = 1.0 / b_ax.get_real_2d(ii, 0);
            }

            let tt = cs_transpose(a, 1).unwrap();
            cs_gaxpy(&tt, &b_ax_inv, &mut g.data);
            drop(tt);
            drop(b_ax_inv);

            let cc: Cs;
            let r: Cs;
            let d1: Vec<f64>;
            let d2: Vec<f64>;
            let d3: Vec<f64>;
            let g1: CvMat;
            let mut g2: CvMat;
            let dsize: i32;

            if option == "prob" {
                cc = get_sub_matrix(&h, 0, 2 * t_ + 3 * m - 1, 0, 2 * t_ + 3 * m - 1);
                r = get_sub_matrix(&h, 2 * t_ + 3 * m, h.m - 1, 0, 2 * t_ + 3 * m - 1);
                d1 = get_sub_matrix_diag(
                    &h,
                    2 * t_ + 3 * m,
                    2 * t_ + 3 * m + 2 * t_ * m - 1,
                    2 * t_ + 3 * m,
                    2 * t_ + 3 * m + 2 * t_ * m - 1,
                );
                d2 = get_sub_matrix_diag(
                    &h,
                    2 * t_ + 3 * m,
                    2 * t_ + 3 * m + 2 * t_ * m - 1,
                    2 * t_ + 3 * m + 2 * t_ * m,
                    h.n - 1,
                );
                d3 = get_sub_matrix_diag(
                    &h,
                    2 * t_ + 3 * m + 2 * t_ * m,
                    h.m - 1,
                    2 * t_ + 3 * m + 2 * t_ * m,
                    h.n - 1,
                );
                g1 = get_sub_matrix_dense(0, 2 * t_ + 3 * m - 1, 0, 0, &g);
                g2 = get_sub_matrix_dense(2 * t_ + 3 * m, g.get_size().height - 1, 0, 0, &g);
                dsize = 2 * t_ * m;
            } else if option == "probV" {
                cc = get_sub_matrix(&h, 0, 2, 0, 2);
                r = get_sub_matrix(&h, 3, h.m - 1, 0, 2);
                d1 = get_sub_matrix_diag(&h, 3, 2 * t_ + 2, 3, 2 * t_ + 2);
                d2 = get_sub_matrix_diag(&h, 3, 2 * t_ + 2, 3 + 2 * t_, h.n - 1);
                d3 = get_sub_matrix_diag(&h, 3 + 2 * t_, h.m - 1, 3 + 2 * t_, h.n - 1);
                g1 = get_sub_matrix_dense(0, 2, 0, 0, &g);
                g2 = get_sub_matrix_dense(3, g.get_size().height - 1, 0, 0, &g);
                tol1 = 1e-4;
                dsize = 2 * t_;
            } else if option == "probU" {
                cc = get_sub_matrix(&h, 0, 3, 0, 3);
                r = get_sub_matrix(&h, 4, h.m - 1, 0, 3);
                d1 = get_sub_matrix_diag(&h, 4, 4 + m - 1, 4, 4 + m - 1);
                d2 = get_sub_matrix_diag(&h, 4, 4 + m - 1, 4 + m, h.n - 1);
                d3 = get_sub_matrix_diag(&h, 4 + m, h.m - 1, 4 + m, h.n - 1);
                g1 = get_sub_matrix_dense(0, 3, 0, 0, &g);
                g2 = get_sub_matrix_dense(4, g.get_size().height - 1, 0, 0, &g);
                tol1 = 1e-4;
                dsize = m;
            } else {
                cout("Unknown option for std_qp");
                the_result.answer_mat = Some(answer_mat);
                return the_result;
            }

            let mut aux1: Vec<f64> = vec![0.0; dsize as usize];
            for ii in 0..dsize as usize {
                aux1[ii] = d1[ii] * d3[ii] - d2[ii] * d2[ii];
            }

            let mut tt = cs_spalloc(dsize + dsize, dsize + dsize, 4 * dsize, 1, 1);
            for ii in 0..dsize {
                let iu = ii as usize;
                cs_entry(&mut tt, ii, ii, d3[iu] / aux1[iu]);
                cs_entry(&mut tt, ii + dsize, ii + dsize, d1[iu] / aux1[iu]);
                cs_entry(&mut tt, ii, ii + dsize, -d2[iu] / aux1[iu]);
                cs_entry(&mut tt, ii + dsize, ii, -d2[iu] / aux1[iu]);
            }
            let dinv = cs_compress(&tt).unwrap();
            drop(tt);
            drop(d1);
            drop(d2);
            drop(d3);
            drop(aux1);

            let tt = cs_transpose(&r, 1).unwrap();
            let aux = cs_multiply(&tt, &dinv).unwrap();
            drop(tt);

            // aux=D1.*D3-D2.^2;
            // Dinv=[spdiags(D3./aux,0,l,l) -spdiags(D2./aux,0,l,l);-spdiags(D2./aux,0,l,l) spdiags(D1./aux,0,l,l)];
            // aux=R'*Dinv;
            // dx1=(C-aux*R)\(-g1+aux*g2);
            // dx2=-Dinv*(g2+R*dx1);
            let tt = cs_multiply(&aux, &r).unwrap();
            let dense_a = cs_add(&cc, &tt, 1.0, -1.0).unwrap();
            drop(tt);
            drop(cc);

            let mut dense_b = CvMat::create(aux.m, 1, CV_64FC1);
            dense_b.set_zero();
            cs_gaxpy(&aux, &g2.data, &mut dense_b.data);
            let src = dense_b.clone();
            cv_sub(src.view(), g1.view(), &mut dense_b);

            // solving system using sparse Cholesky decomposition; the flag
            // is not used
            cs_cholsol(1, &dense_a, &mut dense_b.data);
            let dx1 = dense_b.clone_mat();
            drop(aux);
            drop(dense_a);
            drop(dense_b);

            // dx2 = -Dinv*(g2+R*dx1)
            cs_gaxpy(&r, &dx1.data, &mut g2.data);
            let src = g2.clone();
            cv_convert_scale(src.view(), &mut g2, -1.0, 0.0);
            let mut dx2 = zero_mat(g2.rows, g2.cols);
            cs_gaxpy(&dinv, &g2.data, &mut dx2.data);

            drop(dinv);
            drop(r);
            drop(g2);
            drop(g1);

            // dx = [dx1;dx2]
            let dx = combinedx1dx2(&dx1, &dx2);
            drop(dx1);
            drop(dx2);

            let mut t2 = 1.0f64;

            // forcing A*x <= b
            let mut xdx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
            cv_add(answer_mat.view(), dx.view(), &mut xdx);
            let mut while_loop_test = sparse_dense_mult(a, &xdx);
            drop(xdx);
            let src = while_loop_test.clone();
            cv_sub(b.view(), src.view(), &mut while_loop_test);
            let mut min_value2 = min_elem(&while_loop_test);

            let mut adx = zero_mat(a.m, dx.cols);
            cs_gaxpy(a, &dx.data, &mut adx.data);
            while min_value2 <= 0.0 {
                t2 = beta * t2;
                if t2 < 1e-11 {
                    exit_flag = 1;
                    break;
                }
                let src = while_loop_test.clone();
                cv_scale_add(
                    adx.view(),
                    CvScalar::new((1.0 - beta) * t2 / beta, 0.0, 0.0, 0.0),
                    src.view(),
                    &mut while_loop_test,
                );
                min_value2 = min_elem(&while_loop_test);
            }
            drop(adx);
            drop(while_loop_test);

            // backtracking line search
            // 0.5*t*(x+t2*dx)'*Q*(x+t2*dx)
            let mut xt2dx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
            cv_scale_add(
                dx.view(),
                CvScalar::new(t2, 0.0, 0.0, 0.0),
                answer_mat.view(),
                &mut xt2dx,
            );
            let mut qxt2dx = zero_mat(q.m, 1);
            cs_gaxpy(q, &xt2dx.data, &mut qxt2dx.data);

            let mut thexdx_qxdx = 0.5 * t * cv_dot_product(qxt2dx.view(), xt2dx.view());
            drop(qxt2dx);

            // 0.5*t*x'*Q*x
            let qxt2dx = sparse_dense_mult(q, &answer_mat);
            let thet05x_qx = 0.5 * t * cv_dot_product(answer_mat.view(), qxt2dx.view());
            drop(qxt2dx);

            // t*c'*(x+t2*dx)
            let mut thetctxt2dx = t * (dot_product(&xt2dx, c));

            // t*c'*x
            let thetctx = t * (dot_product(&answer_mat, c));

            // b-A*(x+t2*dx)
            let mut b_axt2dx = b.clone_mat();
            let src = b_axt2dx.clone();
            cv_convert_scale(src.view(), &mut b_axt2dx, -1.0, 0.0);
            cs_gaxpy(a, &xt2dx.data, &mut b_axt2dx.data);
            let src = b_axt2dx.clone();
            cv_convert_scale(src.view(), &mut b_axt2dx, -1.0, 0.0);
            drop(xt2dx);
            // sum log
            let mut sl_b_axt2dx = sum_log(&b_axt2dx);
            let sl_b_ax = sum_log(&b_ax);
            drop(b_axt2dx);

            // alpha*t2*g'*dx
            let mut theat2g_tdx = alpha * t2 * cv_dot_product(g.view(), dx.view());

            // backtracking search
            while (thexdx_qxdx + thetctxt2dx - sl_b_axt2dx)
                > (thet05x_qx + thetctx - sl_b_ax + theat2g_tdx)
            {
                t2 *= beta;
                if t2 < 1e-11 {
                    exit_flag = 1;
                    break;
                }

                // check
                // 0.5*t*(x+t2*dx)'*Q*(x+t2*dx)
                let mut xt2dx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
                cv_scale_add(
                    dx.view(),
                    CvScalar::new(t2, 0.0, 0.0, 0.0),
                    answer_mat.view(),
                    &mut xt2dx,
                );
                let mut qxt2dx = zero_mat(q.m, xt2dx.cols);
                cs_gaxpy(q, &xt2dx.data, &mut qxt2dx.data);

                thexdx_qxdx = 0.5 * t * cv_dot_product(qxt2dx.view(), xt2dx.view());
                drop(qxt2dx);

                thetctxt2dx = t * (dot_product(&xt2dx, c));

                let src = xt2dx.clone();
                cv_convert_scale(src.view(), &mut xt2dx, -1.0, 0.0);
                let mut b_axt2dx = b.clone_mat();
                cs_gaxpy(a, &xt2dx.data, &mut b_axt2dx.data);

                // sum log
                sl_b_axt2dx = sum_log(&b_axt2dx);
                drop(b_axt2dx);
                drop(xt2dx);

                // alpha*t2*g'*dx
                theat2g_tdx = alpha * t2 * cv_dot_product(g.view(), dx.view());
            }

            let mut hdx = zero_mat(h.m, 1);
            cs_gaxpy(&h, &dx.data, &mut hdx.data);
            let lambda2 = cv_dot_product(dx.view(), hdx.view());
            drop(hdx);
            drop(h);
            drop(g);
            if (lambda2 < tol2) || (exit_flag == 1) {
                drop(dx);
                break;
            }

            let src = answer_mat.clone();
            cv_scale_add(
                dx.view(),
                CvScalar::new(t2, 0.0, 0.0, 0.0),
                src.view(),
                &mut answer_mat,
            );
            drop(dx);
            // update bAx
            b_ax = b.clone_mat();
            let src = b_ax.clone();
            cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
            cs_gaxpy(a, &answer_mat.data, &mut b_ax.data);
            let src = b_ax.clone();
            cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
        }
    }

    the_result.answer_mat = Some(answer_mat);
    the_result
}

/// `GetSubMatrix(const cs* A, int xs, int xe, int ys, int ye)`
/// (`std_qp.cpp:682`).
pub fn get_sub_matrix(a: &Cs, xs: i32, xe: i32, ys: i32, ye: i32) -> Cs {
    let xsize = xe - xs + 1; // +1 for inclusive bounds
    let ysize = ye - ys + 1;
    let mut t = cs_spalloc(xsize, ysize, a.nzmax, 1, 1);
    let ax = a.x.as_ref().expect("GetSubMatrix values");

    // cs sparse uses column compress data structure
    for j in ys..=ye {
        for row_p in a.p[j as usize]..a.p[j as usize + 1] {
            let aux_row = a.i[row_p as usize];
            if aux_row >= xs && aux_row <= xe {
                cs_entry(&mut t, aux_row - xs, j - ys, ax[row_p as usize]);
            }
        }
    }
    cs_compress(&t).unwrap()
}

/// `GetSubMatrixDiag(const cs* A, int xs, int xe, int ys, int ye)`
/// (`std_qp.cpp:704`): extract just the main diagonal.  Fixed in
/// translation (BUGS.md): the C++ `new double[xsize]` leaves an entry whose
/// diagonal element is absent from `A` uninitialised; it is 0 here.
pub fn get_sub_matrix_diag(a: &Cs, xs: i32, xe: i32, ys: i32, ye: i32) -> Vec<f64> {
    let xsize = xe - xs + 1; // +1 for inclusive bounds
    let ysize = ye - ys + 1;

    assert!(xsize == ysize);
    let mut result: Vec<f64> = vec![0.0; xsize as usize];
    let ax = a.x.as_ref().expect("GetSubMatrixDiag values");

    // cs sparse uses column compress data structure
    for j in ys..=ye {
        for row_p in a.p[j as usize]..a.p[j as usize + 1] {
            if a.i[row_p as usize] - xs == j - ys {
                result[(j - ys) as usize] = ax[row_p as usize];
            }
        }
    }
    result
}

/// `GetSubMatrix(int xs, int xe, int ys, int ye, CvMat* dense)`
/// (`std_qp.cpp:724`).
pub fn get_sub_matrix_dense(xs: i32, xe: i32, ys: i32, ye: i32, dense: &CvMat) -> CvMat {
    let xsize = xe - xs + 1; // +1 for inclusive bounds
    let ysize = ye - ys + 1;
    let mut result = CvMat::create(xsize, ysize, CV_64FC1);

    for i in xs..=xe {
        for j in ys..=ye {
            result.set_real_2d(i - xs, j - ys, dense.get_real_2d(i, j));
        }
    }
    result
}

/// `SparseDenseMult(cs* S, CvMat* dense)` (`std_qp.cpp:742`): sparse x dense
/// returns a dense matrix.
pub fn sparse_dense_mult(s: &Cs, dense: &CvMat) -> CvMat {
    assert!(s.n == dense.get_size().height);
    let mut result = zero_mat(s.m, dense.get_size().width);

    let n = s.n;
    let ap = &s.p;
    let ai = &s.i;
    let ax = s.x.as_ref().expect("SparseDenseMult values");
    let dense_data = &dense.data;
    let n_cols = result.cols as usize;
    let result_data = &mut result.data;
    for cc in 0..n_cols {
        for j in 0..n as usize {
            for p in ap[j]..ap[j + 1] {
                let p = p as usize;
                result_data[n_cols * ai[p] as usize + cc] += ax[p] * dense_data[n_cols * j + cc];
            }
        }
    }
    result
}

/// `dotProduct(CvMat* dense, cs* sparse)` (`std_qp.cpp:769`).
pub fn dot_product(dense: &CvMat, sparse: &Cs) -> f64 {
    let mut result = 0.0f64;
    assert!(dense.get_size().width == 1);
    assert!(sparse.n == 1);
    assert!(sparse.nz == -1); // sparse matrix has to be in compressed form
    let sx = sparse.x.as_ref().expect("dotProduct values");
    for jj in 0..sparse.p[1] as usize {
        result += dense.data[sparse.i[jj] as usize] * sx[jj];
    }
    result
}

/// `LPsolver(cs* _c, cs* _A, CvMat* b, CvMat* x0)` (`std_qp.cpp:830`):
/// solves LP optimization problems (min c'*x subject to Ax<=b) with the
/// log-barrier method and Newton steps; the result is returned in `x0`.
///
/// Fixed in translation (BUGS.md): the source ends with
/// `memcpy(x0->data.db, xHat->data.db, sizeof(CV_64FC1)*xHat->rows)`, where
/// `CV_64FC1` is an `int` constant, so only 4 bytes per row -- the first
/// half of the vector, and for a 3-row problem half of `x0[1]` -- are copied
/// back.  Every row is copied here.
pub fn lp_solver(c_in: &Cs, a_in: &Cs, b: &CvMat, x0: &mut CvMat) {
    // make sure sparse matrices are column compress format
    let c_owned;
    let c: &Cs = if c_in.nz != -1 {
        c_owned = cs_compress(c_in).expect("LPsolver: cs_compress c");
        &c_owned
    } else {
        c_in
    };
    let a_owned;
    let a: &Cs = if a_in.nz != -1 {
        a_owned = cs_compress(a_in).expect("LPsolver: cs_compress A");
        &a_owned
    } else {
        a_in
    };

    // final result returned. Initialize it
    let mut x_hat = x0.clone_mat();

    let alpha = 0.25f64;
    let beta = 0.5f64;
    let tol1 = 1e-3f64;
    let tol2 = 1e-3f64;
    let newton_maxiters = 20;
    let qp_maxiters = 200;
    let mut exit_flag = 0;
    let mu = 10.0f64;
    let mut t = 1.0 / mu;
    let n = x0.get_size().height as f64;
    let mut iters = 1;

    let mut b_ax = b.clone_mat();
    let src = b_ax.clone();
    cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
    cs_gaxpy(a, &x_hat.data, &mut b_ax.data);
    let src = b_ax.clone();
    cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);

    let min_value = min_elem(&b_ax);

    // check unfeasible starting point
    if min_value < 0.0 {
        cout("Infeasible starting point for LP\n");
    }

    while (n / t > tol1) && (exit_flag == 0) && (iters < qp_maxiters) {
        t *= mu;

        for _iter in 1..=newton_maxiters {
            iters += 1;

            // compute gradient and Hessian
            let bh = b_ax.get_size().height;
            let mut tt = cs_spalloc(bh, bh, bh, 1, 1);
            for ii in 0..bh {
                cs_entry(&mut tt, ii, ii, 1.0 / (b_ax.get_real_2d(ii, 0)));
            }
            let diag_m = cs_compress(&tt).unwrap();
            drop(tt);

            let tt = cs_multiply(&diag_m, a).unwrap();
            drop(diag_m);
            let aux_op = cs_transpose(&tt, 1).unwrap();
            let mut h = cs_multiply(&aux_op, &tt).unwrap();
            drop(tt);
            drop(aux_op);
            // drop small entries
            cs_droptol(&mut h, SPARSE_MATRIX_ZERO_VAL);

            let mut g = CvMat::create(h.m, 1, CV_64FC1);
            g.set_zero();
            let cx = c.x.as_ref().expect("LPsolver c values");
            for cc in c.p[0]..c.p[1] {
                g.data[c.i[cc as usize] as usize] = cx[cc as usize] * t;
            }

            let mut b_ax_inv: Vec<f64> = vec![0.0; a.m as usize];
            for ii in 0..a.m {
                b_ax_inv[ii as usize] = 1.0 / b_ax.get_real_2d(ii, 0);
            }

            let tt = cs_transpose(a, 1).unwrap();
            cs_gaxpy(&tt, &b_ax_inv, &mut g.data);
            drop(tt);
            drop(b_ax_inv);

            // dx = -H\g, computed using cholesky
            let mut dx = CvMat::create(g.rows, g.cols, CV_64FC1);
            cv_convert_scale(g.view(), &mut dx, -1.0, 0.0);
            cs_cholsol(1, &h, &mut dx.data);

            let mut t2 = 1.0f64;

            // forcing A*x <= b
            let mut xdx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
            cv_add(x_hat.view(), dx.view(), &mut xdx);
            let mut while_loop_test = sparse_dense_mult(a, &xdx);
            drop(xdx);
            let src = while_loop_test.clone();
            cv_sub(b.view(), src.view(), &mut while_loop_test);
            let mut min_value2 = min_elem(&while_loop_test);

            let mut adx = zero_mat(a.m, dx.cols);
            cs_gaxpy(a, &dx.data, &mut adx.data);
            while min_value2 <= 0.0 {
                t2 = beta * t2;
                if t2 < 1e-11 {
                    exit_flag = 1;
                    break;
                }
                let src = while_loop_test.clone();
                cv_scale_add(
                    adx.view(),
                    CvScalar::new((1.0 - beta) * t2 / beta, 0.0, 0.0, 0.0),
                    src.view(),
                    &mut while_loop_test,
                );
                min_value2 = min_elem(&while_loop_test);
            }
            drop(adx);
            drop(while_loop_test);

            // backtracking line search
            let mut xt2dx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
            cv_scale_add(
                dx.view(),
                CvScalar::new(t2, 0.0, 0.0, 0.0),
                x_hat.view(),
                &mut xt2dx,
            );

            // t*c'*(x+t2*dx)
            let mut thetctxt2dx = t * (dot_product(&xt2dx, c));

            // t*c'*x
            let thetctx = t * (dot_product(&x_hat, c));

            // b-A*(x+t2*dx)
            let mut b_axt2dx = b.clone_mat();
            let src = b_axt2dx.clone();
            cv_convert_scale(src.view(), &mut b_axt2dx, -1.0, 0.0);
            cs_gaxpy(a, &xt2dx.data, &mut b_axt2dx.data);
            let src = b_axt2dx.clone();
            cv_convert_scale(src.view(), &mut b_axt2dx, -1.0, 0.0);
            drop(xt2dx);
            // sum log
            let mut sl_b_axt2dx = sum_log(&b_axt2dx);
            let sl_b_ax = sum_log(&b_ax);
            drop(b_axt2dx);

            // alpha*t2*g'*dx
            let mut theat2g_tdx = alpha * t2 * cv_dot_product(g.view(), dx.view());

            // backtracking search
            while (thetctxt2dx - sl_b_axt2dx) > (thetctx - sl_b_ax + theat2g_tdx) {
                t2 *= beta;
                if t2 < 1e-11 {
                    exit_flag = 1;
                    break;
                }

                let mut xt2dx = CvMat::create(dx.get_size().height, dx.get_size().width, CV_64FC1);
                cv_scale_add(
                    dx.view(),
                    CvScalar::new(t2, 0.0, 0.0, 0.0),
                    x_hat.view(),
                    &mut xt2dx,
                );

                // t*c'*(x+t2*dx)
                thetctxt2dx = t * (dot_product(&xt2dx, c));

                let src = xt2dx.clone();
                cv_convert_scale(src.view(), &mut xt2dx, -1.0, 0.0);
                let mut b_axt2dx = b.clone_mat();
                cs_gaxpy(a, &xt2dx.data, &mut b_axt2dx.data);

                // sum log
                sl_b_axt2dx = sum_log(&b_axt2dx);
                drop(b_axt2dx);
                drop(xt2dx);
                // alpha*t2*g'*dx
                theat2g_tdx = alpha * t2 * cv_dot_product(g.view(), dx.view());
            }

            let mut hdx = zero_mat(h.m, 1);
            cs_gaxpy(&h, &dx.data, &mut hdx.data);
            let lambda2 = cv_dot_product(dx.view(), hdx.view());
            drop(hdx);
            drop(h);
            drop(g);
            if (lambda2 < tol2) || (exit_flag == 1) {
                drop(dx);
                break;
            }

            let src = x_hat.clone();
            cv_scale_add(
                dx.view(),
                CvScalar::new(t2, 0.0, 0.0, 0.0),
                src.view(),
                &mut x_hat,
            );
            drop(dx);
            // update bAx
            b_ax = b.clone_mat();
            let src = b_ax.clone();
            cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
            cs_gaxpy(a, &x_hat.data, &mut b_ax.data);
            let src = b_ax.clone();
            cv_convert_scale(src.view(), &mut b_ax, -1.0, 0.0);
        }
    }

    let rows = x_hat.rows as usize;
    x0.data[..rows].copy_from_slice(&x_hat.data[..rows]);
}
