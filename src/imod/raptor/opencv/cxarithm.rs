//! Translation of `IMOD/raptor/opencv/cxarithm.cpp` (the parts RAPTOR
//! reaches): `cvAdd`, `cvSub`, `cvMul` and `cvAddS` on `CV_64FC1` arrays
//! without a mask.
//!
//! Each C entry point takes one of two routes to the same per-element
//! expression: an inline loop (last element first) for a small continuous
//! array, or its `icv*_64f_C1R` kernel row by row.  Elements do not depend
//! on each other and the destination never aliases a source here (a caller
//! that passes the same array twice copies the source first), so the order
//! in which they are written does not matter; the expression, in `double`,
//! is what is kept.

use super::cxerror::{CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `cvAdd(src1, src2, dst, NULL)` (`cxarithm.cpp:747`); kernel
/// `icvAdd_64f_C1R` (`ICV_DEF_BIN_ARI_OP_2D( CV_ADD, .. )`).
pub fn cv_add(src1: CvMatRef<'_>, src2: CvMatRef<'_>, dst: &mut CvMat) {
    if src1.rows != src2.rows
        || src1.cols != src2.cols
        || src1.rows != dst.rows
        || src1.cols != dst.cols
    {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvAdd", "", "cxarithm.cpp", 807);
    }
    for r in 0..src1.rows {
        for c in 0..src1.cols {
            let k = dst.index(r, c);
            dst.data[k] = src1.elem(r, c) + src2.elem(r, c);
        }
    }
}

/// `cvSub(src1, src2, dst, NULL)` (`cxarithm.cpp:271`); the C swaps the
/// sources "to comply with IPP" and computes `src2 - src1` of the swapped
/// pair, i.e. the caller's `src1 - src2` (`icvSub_64f_C1R`).
pub fn cv_sub(src1: CvMatRef<'_>, src2: CvMatRef<'_>, dst: &mut CvMat) {
    if src1.rows != src2.rows
        || src1.cols != src2.cols
        || src1.rows != dst.rows
        || src1.cols != dst.cols
    {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvSub", "", "cxarithm.cpp", 353);
    }
    for r in 0..src1.rows {
        for c in 0..src1.cols {
            let k = dst.index(r, c);
            dst.data[k] = src1.elem(r, c) - src2.elem(r, c);
        }
    }
}

/// `cvMul(src1, src2, dst, scale)` (`cxarithm.cpp:1309`).  The inline route
/// is taken only for `scale == 1` and computes `src1*src2`;
/// `icvMul_64f_C1R` (`cxarithm.cpp:1218`) computes `src1*src2` when
/// `fabs(scale - 1.) < DBL_EPSILON` and `scale*src1*src2` otherwise.
pub fn cv_mul(src1: CvMatRef<'_>, src2: CvMatRef<'_>, dst: &mut CvMat, scale: f64) {
    if src1.rows != src2.rows
        || src1.cols != src2.cols
        || src1.rows != dst.rows
        || src1.cols != dst.cols
    {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvMul", "", "cxarithm.cpp", 1386);
    }
    let unit = scale == 1. || (scale - 1.).abs() < f64::EPSILON;
    for r in 0..src1.rows {
        for c in 0..src1.cols {
            let k = dst.index(r, c);
            dst.data[k] = if unit {
                src1.elem(r, c) * src2.elem(r, c)
            } else {
                scale * src1.elem(r, c) * src2.elem(r, c)
            };
        }
    }
}

/// `cvAddS(src, value, dst, NULL)` (`cxarithm.cpp:996`): `value.val[0] +
/// src` per element (inline route, or `icvAddC_64f_C1R` with the scalar
/// expanded by `cvScalarToRawData`).
pub fn cv_add_s(src: CvMatRef<'_>, value: CvScalar, dst: &mut CvMat) {
    if src.rows != dst.rows || src.cols != dst.cols {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvAddS", "", "cxarithm.cpp", 1087);
    }
    for r in 0..src.rows {
        for c in 0..src.cols {
            let k = dst.index(r, c);
            dst.data[k] = value.val[0] + src.elem(r, c);
        }
    }
}
