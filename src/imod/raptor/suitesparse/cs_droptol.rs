//! Translation of `IMOD/raptor/suitesparse/cs_droptol.c`.
use super::Cs;
use super::cs_fkeep::cs_fkeep;

/// C `cs_droptol`: retains CSC entries whose absolute value exceeds `tolerance`.
pub fn cs_droptol(matrix: &mut Cs, tolerance: f64) -> Option<usize> {
    cs_fkeep(matrix, |_, _, value| value.abs() > tolerance)
}
