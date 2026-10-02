//! Translation of `IMOD/raptor/opencv/cxnorm.cpp` (the parts RAPTOR
//! reaches): `cvNorm` of the difference of two `CV_64FC1` matrices without
//! a mask, which RAPTOR calls with `CV_L1` only.
//!
//! `icvInitNormTabs` (the dispatch tables) collapses to the one kernel.

use super::cxerror::{CV_STS_BAD_FLAG, CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `CV_C`.
pub const CV_C: i32 = 1;
/// `CV_L1`.
pub const CV_L1: i32 = 2;
/// `CV_L2`.
pub const CV_L2: i32 = 4;
/// `CV_RELATIVE`.
const CV_RELATIVE: i32 = 8;
/// `CV_DIFF`.
const CV_DIFF: i32 = 16;

/// `icvNormDiff_L1_64f_C1R` (`ICV_DEF_NORM_FUNC_ALL_L1( 64f, fabs, fabs,
/// NOHINT, NOHINT, double, double, double, INT_MAX )`,
/// `ICV_DEF_NORM_DIFF_NOHINT_FUNC_2D`): `norm += fabs(src1[x] - src2[x])`
/// in element order (the four-way unrolled body updates `norm` in the same
/// sequence).  `step1`/`step2` in elements.
fn icv_norm_diff_l1_64f_c1r(
    src1: &[f64],
    step1: usize,
    src2: &[f64],
    step2: usize,
    size: CvSize,
) -> f64 {
    let mut norm = 0f64;
    let (mut o1, mut o2) = (0usize, 0usize);
    for _ in 0..size.height {
        for x in 0..size.width as usize {
            let t0 = (src1[o1 + x] - src2[o2 + x]).abs();
            norm += t0;
        }
        o1 += step1;
        o2 += step2;
    }
    norm
}

/// `cvNorm(imgA, imgB, normType, NULL)` (`cxnorm.cpp:965`), the "light
/// variant" for two `CV_64FC1` matrices and `normType` `CV_L1` (the only
/// form RAPTOR calls): `mat1` is `imgB` and `mat2` is `imgA`, and the
/// result is `icvNormDiff_L1_64f_C1R( mat1, mat2 )`.
pub fn cv_norm(img_a: CvMatRef<'_>, img_b: Option<CvMatRef<'_>>, norm_type: i32) -> f64 {
    let img_b = img_b.expect("cvNorm: RAPTOR always passes two arrays");
    let mat1 = img_b;
    let mat2 = img_a;

    let mut norm_type = norm_type;
    let is_relative = (norm_type & CV_RELATIVE) != 0;
    norm_type &= !CV_RELATIVE;

    match norm_type {
        CV_C | CV_L1 | CV_L2 => norm_type = (norm_type & 7) >> 1,
        t if t == CV_C | CV_DIFF || t == CV_L1 | CV_DIFF || t == CV_L2 | CV_DIFF => {
            norm_type = (norm_type & 7) >> 1
        }
        _ => cv_error(CV_STS_BAD_FLAG, "cvNorm", "", "cxnorm.cpp", 1006),
    }
    assert!(
        norm_type == 1 && !is_relative,
        "cvNorm: RAPTOR reaches CV_L1 only"
    );

    if mat1.rows != mat2.rows || mat1.cols != mat2.cols {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvNorm", "", "cxnorm.cpp", 1016);
    }

    let mut size = mat1.get_size();
    let (mat1_step, mat2_step);
    if cv_is_mat_cont(mat1.type_ & mat2.type_) {
        size.width *= size.height;
        size.height = 1;
        mat1_step = 0;
        mat2_step = 0;
    } else {
        mat1_step = (mat1.step / 8) as usize;
        mat2_step = (mat2.step / 8) as usize;
    }

    icv_norm_diff_l1_64f_c1r(mat1.data, mat1_step, mat2.data, mat2_step, size)
}
