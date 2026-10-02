//! Translation of `IMOD/raptor/opencv/cxmatmul.cpp` (the parts RAPTOR
//! reaches): `cvGEMM` (as `cvMatMul`) with its small-size fast paths and
//! `icvGEMMSingleMul_64f_C1R`, `cvScaleAdd` and `cvDotProduct` for
//! `CV_64FC1`.
//!
//! The destination never shares data with a source here (the Rust borrow of
//! `d` forbids it), so the C's `b != d`/`a != d` tests are always true and
//! its copy-through-a-temporary path is not reached.  RAPTOR's products
//! have an inner dimension of 2, 3 or 4, so `cvGEMM`'s blocked path
//! (`icvGEMMBlockMul`/`icvGEMMStore`, taken only when the inner dimension
//! exceeds 10 on large matrices) and the BLAS hook (null in this build) are
//! not reached.
//!
//! `icvInitGEMMTable` and `icvInitMulAddCTable` (dispatch tables) collapse
//! to the `CV_64FC1` kernels; `icvGEMM_CopyBlock` and `icvGEMM_TransposeBlock`
//! belong to the blocked path.

use super::cxerror::{CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `CV_GEMM_A_T`.
pub const CV_GEMM_A_T: i32 = 1;
/// `CV_GEMM_B_T`.
pub const CV_GEMM_B_T: i32 = 2;
/// `CV_GEMM_C_T`.
pub const CV_GEMM_C_T: i32 = 4;

/// `icvGEMMSingleMul_64f_C1R` (`ICV_DEF_GEMM_SINGLE_MUL( 64f_C1R, double,
/// double )`, `cxmatmul.cpp:116`).  Steps are in bytes as in the C; `c` is
/// `None` for the C's null `c_data`.
#[allow(clippy::too_many_arguments)]
fn icv_gemm_single_mul_64f_c1r(
    a_data: &[f64],
    a_step: usize,
    b_data: &[f64],
    b_step: usize,
    c_data: Option<&[f64]>,
    c_step: usize,
    d_data: &mut [f64],
    d_step: usize,
    a_size: CvSize,
    d_size: CvSize,
    alpha: f64,
    beta: f64,
    flags: i32,
) {
    let mut n = a_size.width as usize;
    let m = d_size.width as usize;
    let drows = d_size.height as usize;
    let a_step = a_step / 8;
    let b_step = b_step / 8;
    let c_step = c_step / 8;
    let d_step = d_step / 8;
    let mut a_step0 = a_step;
    let mut a_step1 = 1usize;
    let (c_step0, c_step1) = if c_data.is_none() {
        (0, 0)
    } else if (flags & CV_GEMM_C_T) == 0 {
        (c_step, 1)
    } else {
        (1, c_step)
    };
    let mut a_buf: Option<Vec<f64>> = None;

    if (flags & CV_GEMM_A_T) != 0 {
        std::mem::swap(&mut a_step0, &mut a_step1);
        n = a_size.height as usize;
        if a_step > 1 && n > 1 {
            a_buf = Some(vec![0.; n]);
        }
    }

    let cval = |off: usize| -> f64 { c_data.map_or(0., |c| c[off]) };

    if n == 1 {
        // external product
        let a_vals: Vec<f64> = if a_step > 1 {
            (0..drows).map(|k| a_data[a_step * k]).collect()
        } else {
            a_data[..drows].to_vec()
        };
        let b_vals: Vec<f64> = if b_step > 1 {
            (0..m).map(|j| b_data[j * b_step]).collect()
        } else {
            b_data[..m].to_vec()
        };

        let mut cc = 0usize;
        let mut dd = 0usize;
        for i in 0..drows {
            let al = a_vals[i] * alpha;
            let mut c = cc;
            let mut j = 0usize;
            while j + 2 <= m {
                let s0 = al * b_vals[j];
                let s1 = al * b_vals[j + 1];
                if c_data.is_none() {
                    d_data[dd + j] = s0;
                    d_data[dd + j + 1] = s1;
                } else {
                    d_data[dd + j] = s0 + cval(c) * beta;
                    d_data[dd + j + 1] = s1 + cval(c + c_step1) * beta;
                }
                j += 2;
                c += 2 * c_step1;
            }
            while j < m {
                let s0 = al * b_vals[j];
                if c_data.is_none() {
                    d_data[dd + j] = s0;
                } else {
                    d_data[dd + j] = s0 + cval(c) * beta;
                }
                j += 1;
                c += c_step1;
            }
            cc += c_step0;
            dd += d_step;
        }
    } else if (flags & CV_GEMM_B_T) != 0 {
        // A * Bt
        let mut aa = 0usize;
        let mut cc = 0usize;
        let mut dd = 0usize;
        for _ in 0..drows {
            let row: Vec<f64> = match &mut a_buf {
                Some(buf) => {
                    for k in 0..n {
                        buf[k] = a_data[aa + a_step1 * k];
                    }
                    buf.clone()
                }
                None => a_data[aa..aa + n].to_vec(),
            };
            let mut bb = 0usize;
            let mut c = cc;
            for j in 0..m {
                let (mut s0, mut s1, mut s2, mut s3) = (0f64, 0f64, 0f64, 0f64);
                let mut k = 0usize;
                while k + 4 <= n {
                    s0 += row[k] * b_data[bb + k];
                    s1 += row[k + 1] * b_data[bb + k + 1];
                    s2 += row[k + 2] * b_data[bb + k + 2];
                    s3 += row[k + 3] * b_data[bb + k + 3];
                    k += 4;
                }
                while k < n {
                    s0 += row[k] * b_data[bb + k];
                    k += 1;
                }
                s0 = (s0 + s1 + s2 + s3) * alpha;

                if c_data.is_none() {
                    d_data[dd + j] = s0;
                } else {
                    d_data[dd + j] = s0 + cval(c) * beta;
                }
                bb += b_step;
                c += c_step1;
            }
            aa += a_step0;
            cc += c_step0;
            dd += d_step;
        }
    } else if m * 8 <= 1600 {
        let mut aa = 0usize;
        let mut cc = 0usize;
        let mut dd = 0usize;
        for _ in 0..drows {
            let row: Vec<f64> = match &mut a_buf {
                Some(buf) => {
                    for k in 0..n {
                        buf[k] = a_data[aa + a_step1 * k];
                    }
                    buf.clone()
                }
                None => a_data[aa..aa + n].to_vec(),
            };
            let mut c = cc;
            let mut j = 0usize;
            while j + 4 <= m {
                let mut b = j;
                let (mut s0, mut s1, mut s2, mut s3) = (0f64, 0f64, 0f64, 0f64);

                for k in 0..n {
                    let a = row[k];
                    s0 += a * b_data[b];
                    s1 += a * b_data[b + 1];
                    s2 += a * b_data[b + 2];
                    s3 += a * b_data[b + 3];
                    b += b_step;
                }

                if c_data.is_none() {
                    d_data[dd + j] = s0 * alpha;
                    d_data[dd + j + 1] = s1 * alpha;
                    d_data[dd + j + 2] = s2 * alpha;
                    d_data[dd + j + 3] = s3 * alpha;
                } else {
                    s0 = s0 * alpha;
                    s1 = s1 * alpha;
                    s2 = s2 * alpha;
                    s3 = s3 * alpha;
                    d_data[dd + j] = s0 + cval(c) * beta;
                    d_data[dd + j + 1] = s1 + cval(c + c_step1) * beta;
                    d_data[dd + j + 2] = s2 + cval(c + c_step1 * 2) * beta;
                    d_data[dd + j + 3] = s3 + cval(c + c_step1 * 3) * beta;
                }
                j += 4;
                c += 4 * c_step1;
            }

            while j < m {
                let mut b = j;
                let mut s0 = 0f64;

                for k in 0..n {
                    s0 += row[k] * b_data[b];
                    b += b_step;
                }

                s0 = s0 * alpha;
                if c_data.is_none() {
                    d_data[dd + j] = s0;
                } else {
                    d_data[dd + j] = s0 + cval(c) * beta;
                }
                j += 1;
                c += c_step1;
            }
            aa += a_step0;
            cc += c_step0;
            dd += d_step;
        }
    } else {
        let mut d_buf = vec![0f64; m];
        let mut aa = 0usize;
        let mut cc = 0usize;
        let mut dd = 0usize;
        for _ in 0..drows {
            let row: Vec<f64> = match &mut a_buf {
                Some(buf) => {
                    for k in 0..n {
                        buf[k] = a_data[aa + a_step1 * k];
                    }
                    buf.clone()
                }
                None => a_data[aa..aa + n].to_vec(),
            };

            for j in 0..m {
                d_buf[j] = 0.;
            }

            let mut bb = 0usize;
            for k in 0..n {
                let al = row[k];
                let mut j = 0usize;
                while j + 4 <= m {
                    let mut t0 = d_buf[j] + b_data[bb + j] * al;
                    let mut t1 = d_buf[j + 1] + b_data[bb + j + 1] * al;
                    d_buf[j] = t0;
                    d_buf[j + 1] = t1;
                    t0 = d_buf[j + 2] + b_data[bb + j + 2] * al;
                    t1 = d_buf[j + 3] + b_data[bb + j + 3] * al;
                    d_buf[j + 2] = t0;
                    d_buf[j + 3] = t1;
                    j += 4;
                }
                while j < m {
                    d_buf[j] += b_data[bb + j] * al;
                    j += 1;
                }
                bb += b_step;
            }

            if c_data.is_none() {
                for j in 0..m {
                    d_data[dd + j] = d_buf[j] * alpha;
                }
            } else {
                let mut c = cc;
                for j in 0..m {
                    let t = d_buf[j] * alpha;
                    d_data[dd + j] = t + cval(c) * beta;
                    c += c_step1;
                }
            }
            aa += a_step0;
            cc += c_step0;
            dd += d_step;
        }
    }
}

/// `cvGEMM(A, B, alpha, C, beta, D, tABC)` (`cxmatmul.cpp:605`) for
/// `CV_64FC1`.
pub fn cv_gemm(
    a: CvMatRef<'_>,
    b: CvMatRef<'_>,
    alpha: f64,
    c: Option<CvMatRef<'_>>,
    beta: f64,
    d: &mut CvMat,
    t_abc: i32,
) {
    let mut flags = t_abc;
    let c = if beta == 0. { None } else { c };

    if let Some(c) = c {
        if ((flags & CV_GEMM_C_T) == 0 && (c.cols != d.cols || c.rows != d.rows))
            || ((flags & CV_GEMM_C_T) != 0 && (c.rows != d.cols || c.cols != d.rows))
        {
            cv_error(CV_STS_UNMATCHED_SIZES, "cvGEMM", "", "cxmatmul.cpp", 678);
        }
    }
    // `C = &stub3; C->data.ptr = 0; C->step = 0; C->type = CV_MAT_CONT_FLAG;`
    let (c_data, c_step): (Option<&[f64]>, usize) = match c {
        Some(c) => (Some(c.data), c.step as usize),
        None => (None, 0),
    };

    let a_size = CvSize {
        width: a.cols,
        height: a.rows,
    };
    let d_size = CvSize {
        width: d.cols,
        height: d.rows,
    };
    let len;

    match flags & (CV_GEMM_A_T | CV_GEMM_B_T) {
        0 => {
            len = b.rows;
            if a_size.width != len || b.cols != d_size.width || a_size.height != d_size.height {
                cv_error(CV_STS_UNMATCHED_SIZES, "cvGEMM", "", "cxmatmul.cpp", 705);
            }
        }
        1 => {
            len = b.rows;
            if a_size.height != len || b.cols != d_size.width || a_size.width != d_size.height {
                cv_error(CV_STS_UNMATCHED_SIZES, "cvGEMM", "", "cxmatmul.cpp", 712);
            }
        }
        2 => {
            len = b.cols;
            if a_size.width != len || b.rows != d_size.width || a_size.height != d_size.height {
                cv_error(CV_STS_UNMATCHED_SIZES, "cvGEMM", "", "cxmatmul.cpp", 719);
            }
        }
        _ => {
            len = b.cols;
            if a_size.height != len || b.rows != d_size.width || a_size.width != d_size.height {
                cv_error(CV_STS_UNMATCHED_SIZES, "cvGEMM", "", "cxmatmul.cpp", 726);
            }
        }
    }

    if flags == 0 && (2..=4).contains(&len) && (len == d_size.width || len == d_size.height) {
        let a_step = (a.step / 8) as usize;
        let b_step = (b.step / 8) as usize;
        let d_step = (d.step / 8) as usize;
        let mut c_step = c_step / 8;
        let zero = [0f64; 4];
        let cd: &[f64] = c_data.unwrap_or(&zero);
        let c_is_zero = c_data.is_none();
        let ad = a.data;
        let bd = b.data;
        let dh = d_size.height as usize;
        let dw = d_size.width as usize;
        let dd = &mut d.data;

        match len {
            2 => {
                if len == d_size.width {
                    let (mut ai, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dh {
                        let t0 = ad[ai] * bd[0] + ad[ai + 1] * bd[b_step];
                        let t1 = ad[ai] * bd[1] + ad[ai + 1] * bd[b_step + 1];
                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + 1] = t1 * alpha + cd[ci + 1] * beta;
                        di += d_step;
                        ai += a_step;
                        ci += c_step;
                    }
                } else {
                    let mut c_step0 = 1usize;
                    if c_is_zero {
                        c_step0 = 0;
                        c_step = 1;
                    }
                    let (mut bi, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dw {
                        let t0 = ad[0] * bd[bi] + ad[1] * bd[bi + b_step];
                        let t1 = ad[a_step] * bd[bi] + ad[a_step + 1] * bd[bi + b_step];
                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + d_step] = t1 * alpha + cd[ci + c_step] * beta;
                        di += 1;
                        bi += 1;
                        ci += c_step0;
                    }
                }
                return;
            }
            3 => {
                if len == d_size.width {
                    let (mut ai, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dh {
                        let t0 =
                            ad[ai] * bd[0] + ad[ai + 1] * bd[b_step] + ad[ai + 2] * bd[b_step * 2];
                        let t1 = ad[ai] * bd[1]
                            + ad[ai + 1] * bd[b_step + 1]
                            + ad[ai + 2] * bd[b_step * 2 + 1];
                        let t2 = ad[ai] * bd[2]
                            + ad[ai + 1] * bd[b_step + 2]
                            + ad[ai + 2] * bd[b_step * 2 + 2];
                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + 1] = t1 * alpha + cd[ci + 1] * beta;
                        dd[di + 2] = t2 * alpha + cd[ci + 2] * beta;
                        di += d_step;
                        ai += a_step;
                        ci += c_step;
                    }
                } else {
                    let mut c_step0 = 1usize;
                    if c_is_zero {
                        c_step0 = 0;
                        c_step = 1;
                    }
                    let (mut bi, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dw {
                        let t0 =
                            ad[0] * bd[bi] + ad[1] * bd[bi + b_step] + ad[2] * bd[bi + b_step * 2];
                        let t1 = ad[a_step] * bd[bi]
                            + ad[a_step + 1] * bd[bi + b_step]
                            + ad[a_step + 2] * bd[bi + b_step * 2];
                        let t2 = ad[a_step * 2] * bd[bi]
                            + ad[a_step * 2 + 1] * bd[bi + b_step]
                            + ad[a_step * 2 + 2] * bd[bi + b_step * 2];

                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + d_step] = t1 * alpha + cd[ci + c_step] * beta;
                        dd[di + d_step * 2] = t2 * alpha + cd[ci + c_step * 2] * beta;
                        di += 1;
                        bi += 1;
                        ci += c_step0;
                    }
                }
                return;
            }
            _ => {
                if len == d_size.width {
                    let (mut ai, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dh {
                        let t0 = ad[ai] * bd[0]
                            + ad[ai + 1] * bd[b_step]
                            + ad[ai + 2] * bd[b_step * 2]
                            + ad[ai + 3] * bd[b_step * 3];
                        let t1 = ad[ai] * bd[1]
                            + ad[ai + 1] * bd[b_step + 1]
                            + ad[ai + 2] * bd[b_step * 2 + 1]
                            + ad[ai + 3] * bd[b_step * 3 + 1];
                        let t2 = ad[ai] * bd[2]
                            + ad[ai + 1] * bd[b_step + 2]
                            + ad[ai + 2] * bd[b_step * 2 + 2]
                            + ad[ai + 3] * bd[b_step * 3 + 2];
                        let t3 = ad[ai] * bd[3]
                            + ad[ai + 1] * bd[b_step + 3]
                            + ad[ai + 2] * bd[b_step * 2 + 3]
                            + ad[ai + 3] * bd[b_step * 3 + 3];
                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + 1] = t1 * alpha + cd[ci + 1] * beta;
                        dd[di + 2] = t2 * alpha + cd[ci + 2] * beta;
                        dd[di + 3] = t3 * alpha + cd[ci + 3] * beta;
                        di += d_step;
                        ai += a_step;
                        ci += c_step;
                    }
                    return;
                } else if d_size.width <= 16 {
                    let mut c_step0 = 1usize;
                    if c_is_zero {
                        c_step0 = 0;
                        c_step = 1;
                    }
                    let (mut bi, mut ci, mut di) = (0usize, 0usize, 0usize);
                    for _ in 0..dw {
                        let t0 = ad[0] * bd[bi]
                            + ad[1] * bd[bi + b_step]
                            + ad[2] * bd[bi + b_step * 2]
                            + ad[3] * bd[bi + b_step * 3];
                        let t1 = ad[a_step] * bd[bi]
                            + ad[a_step + 1] * bd[bi + b_step]
                            + ad[a_step + 2] * bd[bi + b_step * 2]
                            + ad[a_step + 3] * bd[bi + b_step * 3];
                        let t2 = ad[a_step * 2] * bd[bi]
                            + ad[a_step * 2 + 1] * bd[bi + b_step]
                            + ad[a_step * 2 + 2] * bd[bi + b_step * 2]
                            + ad[a_step * 2 + 3] * bd[bi + b_step * 3];
                        let t3 = ad[a_step * 3] * bd[bi]
                            + ad[a_step * 3 + 1] * bd[bi + b_step]
                            + ad[a_step * 3 + 2] * bd[bi + b_step * 2]
                            + ad[a_step * 3 + 3] * bd[bi + b_step * 3];
                        dd[di] = t0 * alpha + cd[ci] * beta;
                        dd[di + d_step] = t1 * alpha + cd[ci + c_step] * beta;
                        dd[di + d_step * 2] = t2 * alpha + cd[ci + c_step * 2] * beta;
                        dd[di + d_step * 3] = t3 * alpha + cd[ci + c_step * 3] * beta;
                        di += 1;
                        bi += 1;
                        ci += c_step0;
                    }
                    return;
                }
                // else break: fall through to the general code
            }
        }
    }

    let mut b_step = b.step as usize;

    if (d_size.width == 1 || len == 1) && (flags & CV_GEMM_B_T) == 0 && cv_is_mat_cont(b.type_) {
        b_step = if d_size.width == 1 { 0 } else { 8 };
        flags |= CV_GEMM_B_T;
    }

    assert!(
        d_size.height <= 128 / 2
            || d_size.width <= 128 / 2
            || len <= 10
            || (d_size.width <= 128 && d_size.height <= 128 && len <= 128),
        "cvGEMM: the blocked product is not reached from RAPTOR"
    );
    let d_step = d.step as usize;
    icv_gemm_single_mul_64f_c1r(
        a.data,
        a.step as usize,
        b.data,
        b_step,
        c_data,
        c_step,
        &mut d.data,
        d_step,
        a_size,
        d_size,
        alpha,
        beta,
        flags,
    );
}

/// `cvMatMul(A, B, D)`: `cxcore.h` defines it as `cvGEMM(A, B, 1, NULL, 0,
/// D, 0)`.
pub fn cv_mat_mul(a: CvMatRef<'_>, b: CvMatRef<'_>, d: &mut CvMat) {
    cv_gemm(a, b, 1.0, None, 0.0, d, 0)
}

/// `cvScaleAdd(src1, scale, src2, dst)` (`cxmatmul.cpp:2214`) for
/// `CV_64FC1`: `src1*scale.val[0] + src2` per element (the inline route for
/// a small continuous array and `icvMulAddC_64f_C1R` compute the same
/// expression).
pub fn cv_scale_add(src1: CvMatRef<'_>, scale: CvScalar, src2: CvMatRef<'_>, dst: &mut CvMat) {
    if src1.rows != dst.rows
        || src1.cols != dst.cols
        || src2.rows != dst.rows
        || src2.cols != dst.cols
    {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvScaleAdd",
            "",
            "cxmatmul.cpp",
            2250,
        );
    }
    let s0 = scale.val[0];
    let cols = src1.cols as usize;
    let (s1, s2, sd) = (
        (src1.step / 8) as usize,
        (src2.step / 8) as usize,
        (dst.step / 8) as usize,
    );
    for r in 0..src1.rows as usize {
        let a = &src1.data[r * s1..r * s1 + cols];
        let b = &src2.data[r * s2..r * s2 + cols];
        let d = &mut dst.data[r * sd..r * sd + cols];
        for ((d, &a), &b) in d.iter_mut().zip(a).zip(b) {
            *d = a * s0 + b;
        }
    }
}

/// `icvDotProduct_64f_C1R` (`ICV_DEF_DOT_PROD_FUNC_2D( 64f, double,
/// double, double )`, `cxmatmul.cpp:3230`).  `step1`/`step2` in bytes.
fn icv_dot_product_64f_c1r(
    src1: &[f64],
    step1: usize,
    src2: &[f64],
    step2: usize,
    size: CvSize,
) -> f64 {
    let mut sum = 0f64;
    let step1 = step1 / 8;
    let step2 = step2 / 8;
    let width = size.width as usize;
    let (mut o1, mut o2) = (0usize, 0usize);

    for _ in 0..size.height {
        let mut i = 0usize;
        while i + 4 <= width {
            let mut t0 = src1[o1 + i] * src2[o2 + i];
            let mut t1 = src1[o1 + i + 1] * src2[o2 + i + 1];
            t0 += src1[o1 + i + 2] * src2[o2 + i + 2];
            t1 += src1[o1 + i + 3] * src2[o2 + i + 3];
            sum += t0 + t1;
            i += 4;
        }

        while i < width {
            sum += src1[o1 + i] * src2[o2 + i];
            i += 1;
        }
        o1 += step1;
        o2 += step2;
    }

    sum
}

/// `cvDotProduct(srcA, srcB)` (`cxmatmul.cpp:3271`) for `CV_64FC1`.  A
/// continuous pair of at most `CV_MAX_INLINE_MAT_OP_SIZE` elements is summed
/// inline from the last element to the first; otherwise
/// `icvDotProduct_64f_C1R` sums in groups of four.
pub fn cv_dot_product(src_a: CvMatRef<'_>, src_b: CvMatRef<'_>) -> f64 {
    if src_a.rows != src_b.rows || src_a.cols != src_b.cols {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvDotProduct",
            "",
            "cxmatmul.cpp",
            3326,
        );
    }

    let mut size = src_a.get_size();
    let mut step_a = src_a.step as usize;
    let mut step_b = src_b.step as usize;

    if cv_is_mat_cont(src_a.type_ & src_b.type_) {
        size.width *= size.height;

        if size.width <= super::cxconvert::CV_MAX_INLINE_MAT_OP_SIZE {
            let m_a = src_a.data;
            let m_b = src_b.data;
            let mut sum = 0f64;
            let mut w = size.width as usize;
            loop {
                sum += m_a[w - 1] * m_b[w - 1];
                w -= 1;
                if w == 0 {
                    break;
                }
            }
            return sum;
        }
        size.height = 1;
        step_a = 0;
        step_b = 0;
    }

    icv_dot_product_64f_c1r(src_a.data, step_a, src_b.data, step_b, size)
}
