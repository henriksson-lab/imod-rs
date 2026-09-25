//! Translation of `IMOD/flib/subrs/statsubs/polyfit.f`.

use crate::imod::libcfshr::regression::polynomial_fit;

/// Original `polyfit` (`polyfit.f:6`).
///
/// POLYFIT uses multiple linear regression to fit a polynomial of order
/// IORDER to NFIT points whose (x,y) coordinates are in the arrays X
/// and Y It returns the coefficient of x**i in the array SLOP and a
/// constant term in BINT.  Y = BINTCP + sum ( SLOPES(i) * X**i )
///
/// `work` is the source's automatic array of `2*(iorder + 1) *
/// (iorder + 3 + nfit)` reals; `polynomialFit` stores every element it
/// reads, so the zero fill stands in for the uninitialised stack storage
/// without changing a result.
pub fn polyfit(
    x: &[f32],
    y: &[f32],
    nfit: i32,
    iorder: i32,
    slopes: &mut [f32],
    bintcp: &mut f32,
) -> i32 {
    let mut work = vec![0f32; (2 * (iorder + 1) * (iorder + 3 + nfit)).max(0) as usize];
    polynomial_fit(
        x,
        y,
        nfit,
        iorder,
        slopes,
        std::slice::from_mut(bintcp),
        &mut work,
    )
}

/// Original `localpolyfit` (`polyfit.f:15`).
pub fn localpolyfit(
    x: &[f32],
    y: &[f32],
    nfit: i32,
    iorder: i32,
    slopes: &mut [f32],
    bintcp: &mut f32,
) -> i32 {
    polyfit(x, y, nfit, iorder, slopes, bintcp)
}
