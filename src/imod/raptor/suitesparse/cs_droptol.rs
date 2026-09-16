//! Translation of `IMOD/raptor/suitesparse/cs_droptol.c`.
use super::Cs;
use super::cs_fkeep::cs_fkeep;

/// C static `cs_tol`.
fn cs_tol(_i: usize, _j: usize, aij: f64, tolerance: f64) -> bool {
    aij.abs() > tolerance
}

/// C `cs_droptol`: retains CSC entries whose absolute value exceeds `tolerance`.
pub fn cs_droptol(matrix: &mut Cs, tolerance: f64) -> Option<usize> {
    cs_fkeep(matrix, |i, j, value| cs_tol(i, j, value, tolerance))
}
