//! Translation of `IMOD/raptor/opencv/cxcopy.cpp` (the parts RAPTOR
//! reaches): `cvSet` and `cvSetZero` on a `CV_64FC1` matrix without a mask.
//! `icvSetZero_8u_C1R` is a `memset` (every element becomes `+0.0`) and
//! `icvSet_8u_C1R` copies the scalar's raw `double` (`cvScalarToRawData`)
//! into every element; both are folded into these bodies.
//!
//! `cvCopy` of two `CV_64FC1` matrices is reached only through
//! `cvConvertScale` with a unit scale and zero shift, and is written out in
//! `cxconvert.rs`.

use super::cxtypes::*;

impl CvMat {
    /// `cvSetZero(arr)` (`cxcopy.cpp:672`).
    pub fn set_zero(&mut self) {
        let n = self.rows as usize * self.cols as usize;
        self.data[..n].fill(0.);
    }

    /// `cvSet(arr, value, NULL)` (`cxcopy.cpp:484`): an all-zero scalar is
    /// `cvZero`; otherwise every element becomes `value.val[0]` (the raw data
    /// of a scalar converted to `CV_64FC1`).
    pub fn set(&mut self, value: CvScalar) {
        if value.val[0] == 0. && value.val[1] == 0. && value.val[2] == 0. && value.val[3] == 0. {
            self.set_zero();
            return;
        }
        let n = self.rows as usize * self.cols as usize;
        self.data[..n].fill(value.val[0]);
    }
}
