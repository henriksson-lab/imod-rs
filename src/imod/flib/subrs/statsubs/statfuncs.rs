//! Translation of `IMOD/flib/subrs/statsubs/statfuncs.f`.
//!
//! Simple floating point wrappers for statistical functions in libcfshr, to
//! replace old routines based on Press et al.  Each calls the libcfshr
//! Fortran-callable wrapper the gfortran symbol resolves to (`incompbeta_`,
//! `dtvalue_`, `dfvalue_`, `errfunc_`).

use crate::imod::libcfshr::statfuncs::{dfvalue, dtvalue, errfunc, incompbeta};

/// Original `betai` (`statfuncs.f:4`).
pub fn betai(a: f32, b: f32, x: f32) -> f32 {
    let da: f64 = a as f64;
    let db: f64 = b as f64;
    let dx: f64 = x as f64;
    incompbeta(&da, &db, &dx) as f32
}

/// Original `tvalue` (`statfuncs.f:15`).
pub fn tvalue(signif: f32, ndf: i32) -> f32 {
    let dsig: f64 = signif as f64;
    dtvalue(&dsig, &ndf) as f32
}

/// Original `fvalue` (`statfuncs.f:25`).
pub fn fvalue(signif: f32, ndf1: i32, ndf2: i32) -> f32 {
    let dsig: f64 = signif as f64;
    dfvalue(&dsig, &ndf1, &ndf2) as f32
}

/// Original `erfcc` (`statfuncs.f:35`).  `1. - errFunc(dx)` is a `real*4`
/// constant widened exactly into a `real*8` subtraction, rounded to `real*4`
/// on the assignment.
pub fn erfcc(x: f32) -> f32 {
    let dx: f64 = x as f64;
    (1.0_f64 - errfunc(&dx)) as f32
}
