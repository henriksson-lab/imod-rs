//! Translation of `IMOD/raptor/optimization/probData.h` and `probData.cpp`.
//! The destructor is `Drop`.

use crate::imod::raptor::opencv::cxtypes::CvMat;
use crate::imod::raptor::suitesparse::cs::Cs;
use crate::imod::raptor::suitesparse::cs_util::cs_spalloc;

/// `class probData`: the data of a QP problem.
#[derive(Debug, Default)]
pub struct ProbData {
    pub buc: Option<CvMat>,
    pub a: Option<Cs>,
    pub c: Option<Cs>,
    pub q: Option<Cs>,
    pub x0: Option<CvMat>,
}

impl ProbData {
    /// `probData()` (`probData.cpp:15`).
    pub fn new() -> ProbData {
        ProbData::default()
    }

    /// `probData(const probData& pD)` (`probData.cpp:24`), the copy
    /// constructor.
    pub fn copy(pd: &ProbData) -> ProbData {
        ProbData {
            buc: pd.buc.as_ref().map(|m| m.clone_mat()),
            a: pd.a.as_ref().map(ProbData::copy_cs),
            c: pd.c.as_ref().map(ProbData::copy_cs),
            q: pd.q.as_ref().map(ProbData::copy_cs),
            x0: pd.x0.as_ref().map(|m| m.clone_mat()),
        }
    }

    /// `probData::CopyCS(cs* A)` (`probData.h:28`).
    pub fn copy_cs(a: &Cs) -> Cs {
        let mut b = if a.nz == -1 {
            cs_spalloc(a.m, a.n, a.nzmax, 1, 0)
        } else {
            cs_spalloc(a.m, a.n, a.nzmax, 1, 1)
        };
        let ax = a.x.as_ref().expect("probData::CopyCS values");
        let bx = b.x.as_mut().unwrap();
        for ii in 0..a.nzmax as usize {
            b.i[ii] = a.i[ii];
            bx[ii] = ax[ii];
        }
        if a.nz == -1 {
            // compressed colum format
            for ii in 0..(a.n + 1) as usize {
                b.p[ii] = a.p[ii];
            }
        } else {
            for ii in 0..a.nzmax as usize {
                b.p[ii] = a.p[ii];
            }
        }
        b.nz = a.nz;
        b
    }
}
