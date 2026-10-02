//! Translation of `IMOD/raptor/opencv/cxmatrix.cpp` (the parts RAPTOR
//! reaches): `cvTranspose` of `CV_64FC1` into a separate matrix and
//! `cvSolve` with `CV_SVD`.

use super::cxerror::{CV_STS_BAD_ARG, CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxsvd::{CV_SVD_U_T, CV_SVD_V_T, cv_svbksb, cv_svd};
use super::cxtypes::*;

/// `CV_LU`.
pub const CV_LU: i32 = 0;
/// `CV_SVD`.
pub const CV_SVD: i32 = 1;
/// `CV_SVD_SYM`.
pub const CV_SVD_SYM: i32 = 2;

/// `cvTranspose(src, dst)` (`cxmatrix.cpp:441`) into a different matrix:
/// `icvTranspose_32s_C2R`, which moves each 8-byte element as a pair of
/// `int`s (a bit copy).  The in-place routes are not reached.
pub fn cv_transpose(src: CvMatRef<'_>, dst: &mut CvMat) {
    let size = src.get_size();
    if size.width != dst.rows || size.height != dst.cols {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvTranspose",
            "",
            "cxmatrix.cpp",
            490,
        );
    }

    for i in 0..size.height {
        for j in 0..size.width {
            let k = dst.index(j, i);
            dst.data[k] = src.elem(i, j);
        }
    }
}

/// `cvSolve(A, b, x, method)` (`cxmatrix.cpp:1122`).  RAPTOR solves with
/// `CV_SVD` only (its one `CV_LU` call is commented out), so the LU route is
/// not reached.  Returns the C's `result` (1).
pub fn cv_solve(a: CvMatRef<'_>, b: CvMatRef<'_>, x: &mut CvMat, method: i32) -> i32 {
    let result = 1;

    if method == CV_SVD || method == CV_SVD_SYM {
        let n = a.rows.min(a.cols);

        if method == CV_SVD_SYM && a.rows != a.cols {
            cv_error(
                -201,
                "cvSolve",
                "CV_SVD_SYM method is used for non-square matrix",
                "cxmatrix.cpp",
                1148,
            );
        }
        assert!(method == CV_SVD, "cvSolve: CV_SVD_SYM is not reached");

        let mut u = CvMat::create(n, a.rows, a.type_);
        let mut v = CvMat::create(n, a.cols, a.type_);
        let mut w = CvMat::create(n, 1, a.type_);
        cv_svd(a, &mut w, &mut u, &mut v, CV_SVD_U_T + CV_SVD_V_T);
        cv_svbksb(&w, &u, &v, Some(b), x, CV_SVD_U_T + CV_SVD_V_T);
        return result;
    } else if method != CV_LU {
        cv_error(
            CV_STS_BAD_ARG,
            "cvSolve",
            "Unknown inversion method",
            "cxmatrix.cpp",
            1158,
        );
    }

    panic!("cvSolve: the CV_LU route is not reached from RAPTOR");
}
