//! Translation of `IMOD/raptor/opencv/cxmathfuncs.cpp` (the parts RAPTOR
//! reaches): `cvPow` with `power == 0.5` on `CV_64FC1` (the
//! `icvSqrt_64f` route).

use super::cxerror::{CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `cvPow(src, dst, power)` (`cxmathfuncs.cpp:1698`).  RAPTOR calls it with
/// `power` 0.5 only: `cvRound(0.5)` is 0, not within `DBL_EPSILON` of 0.5, so
/// the power is fractional, and `fabs(fabs(power) - 0.5) < DBL_EPSILON`
/// selects `icvSqrt_64f` (`cxmathfuncs.cpp:247`), `dst[i] = sqrt(src[i])`
/// row by row.  The integer-power and `exp(log)` routes are not reached.
pub fn cv_pow(src: CvMatRef<'_>, dst: &mut CvMat, power: f64) {
    if src.rows != dst.rows || src.cols != dst.cols {
        cv_error(CV_STS_UNMATCHED_SIZES, "cvPow", "", "cxmathfuncs.cpp", 1732);
    }
    let ipower = cv_round(power);
    assert!(
        !((ipower as f64 - power).abs() < f64::EPSILON)
            && (power.abs() - 0.5).abs() < f64::EPSILON
            && power >= 0.,
        "cvPow: RAPTOR only reaches power 0.5"
    );
    for r in 0..src.rows {
        for c in 0..src.cols {
            let k = dst.index(r, c);
            dst.data[k] = src.elem(r, c).sqrt();
        }
    }
}
