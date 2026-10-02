//! Translation of `IMOD/raptor/optimization/estimation3ddata.h` and
//! `estimation3ddata.cpp`.  The destructor is `Drop`.

use super::std_qp::zero_mat;
use crate::imod::raptor::opencv::cxtypes::CvMat;

/// `class estimation3ddata`: the projection model.
#[derive(Clone, Debug, Default)]
pub struct Estimation3dData {
    pub resid_mean_perc: f64,
    pub resid_mean: f64,
    pub g: Option<CvMat>,
    pub p: Option<CvMat>,
    pub t: Option<CvMat>,
}

impl Estimation3dData {
    /// `estimation3ddata()` (`estimation3ddata.cpp:17`).  The residuals the
    /// C++ leaves uninitialised start at 0; every reached path sets
    /// `residMean` before reading it.
    pub fn new() -> Estimation3dData {
        Estimation3dData::default()
    }

    /// `estimation3ddata(int T, int M)` (`estimation3ddata.cpp:24`).
    pub fn with_size(t: i32, m: i32) -> Estimation3dData {
        Estimation3dData {
            resid_mean_perc: 0.0,
            resid_mean: 0.0,
            g: Some(zero_mat(2 * t, 3)),
            t: Some(zero_mat(2 * t, 1)),
            p: Some(zero_mat(3, m)),
        }
    }
}
