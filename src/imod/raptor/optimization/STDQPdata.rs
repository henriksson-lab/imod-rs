//! Translation of `IMOD/raptor/optimization/STDQPdata.h` and `STDQPdata.cpp`.
//!
//! Fixed in translation (BUGS.md): the C++ constructor sets only
//! `answerMat`, and `std_qp` never stores `numItrs` on an infeasible start
//! nor `exit_flag`/`gap` on a normal return, while `estimation3D` reads
//! `numItrs` and `SFMestimationWithBA` reads `exit_flag` after every call --
//! reads of uninitialised heap memory.  They start at 0 here.

use crate::imod::raptor::opencv::cxtypes::CvMat;

/// `class STDQPdata`: the result of `std_qp`.  `clear()` and the
/// destructor are `Drop`.
#[derive(Clone, Debug, Default)]
pub struct StdQpData {
    pub answer_mat: Option<CvMat>,
    pub num_itrs: i32,
    pub exit_flag: i32,
    pub gap: f64,
}

impl StdQpData {
    /// `STDQPdata()` (`STDQPdata.cpp:15`).
    pub fn new() -> StdQpData {
        StdQpData {
            answer_mat: None,
            num_itrs: 0,
            exit_flag: 0,
            gap: 0.0,
        }
    }
}
