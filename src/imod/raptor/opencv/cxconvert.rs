//! Translation of `IMOD/raptor/opencv/cxconvert.cpp` (the parts RAPTOR
//! reaches): `cvConvertScale` between `CV_64FC1` matrices.  The 32F<->64F
//! conversions `icvCrossCorr` makes are written out in `cvtemplmatch.rs`
//! where it calls them.

use super::cxerror::{CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `CV_MAX_INLINE_MAT_OP_SIZE` (`cxmisc.h:64`).
pub const CV_MAX_INLINE_MAT_OP_SIZE: i32 = 10;

/// `cvConvertScale(src, dst, scale, shift)` (`cxconvert.cpp:1495`); also
/// `cvScale(src, dst, scale)`, which `cxcore.h` defines as
/// `cvConvertScale(src, dst, scale, 0)`.  For `CV_64FC1` to `CV_64FC1`:
/// with `scale == 1 && shift == 0` it is `cvCopy`; a continuous array of at
/// most `CV_MAX_INLINE_MAT_OP_SIZE` elements is converted inline, last
/// element first; otherwise `icvCvtScaleTo_64f_C1R` converts row by row.
/// Every path computes `src*scale + shift` per element, in `double`.
pub fn cv_convert_scale(src: CvMatRef<'_>, dst: &mut CvMat, scale: f64, shift: f64) {
    let no_scale = scale == 1. && shift == 0.;

    if no_scale {
        // `cvCopy( src, dst )`; the types are equal.
        if src.rows != dst.rows || src.cols != dst.cols {
            cv_error(CV_STS_UNMATCHED_SIZES, "cvCopy", "", "cxcopy.cpp", 320);
        }
        for r in 0..src.rows {
            for c in 0..src.cols {
                let k = dst.index(r, c);
                dst.data[k] = src.elem(r, c);
            }
        }
        return;
    }

    if src.rows != dst.rows || src.cols != dst.cols {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvConvertScale",
            "",
            "cxconvert.cpp",
            1601,
        );
    }

    let mut size = src.get_size();
    if cv_is_mat_cont(src.type_ & dst.type_) {
        size.width *= size.height;
        size.height = 1;
    }

    if size.height == 1 && size.width <= CV_MAX_INLINE_MAT_OP_SIZE {
        // both continuous (or one row): element k of the flat run
        let mut w = size.width as usize;
        loop {
            dst.data[w - 1] = src.data[w - 1] * scale + shift;
            w -= 1;
            if w == 0 {
                break;
            }
        }
        return;
    }

    // icvCvtScaleTo_64f_C1R, `ICV_DEF_CVT_SCALE_CASE( double, double,
    // CV_NOP, CV_CAST_64F, scale, shift )`: `(a)*_src[i]+(b)`.
    for r in 0..src.rows {
        for c in 0..src.cols {
            let k = dst.index(r, c);
            dst.data[k] = scale * src.elem(r, c) + shift;
        }
    }
}
