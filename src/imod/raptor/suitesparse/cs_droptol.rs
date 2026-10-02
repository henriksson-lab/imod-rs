//! Translation of `IMOD/raptor/suitesparse/cs_droptol.c`.

use super::cs::Cs;
use super::cs_fkeep::cs_fkeep;

/// `cs_tol(i, j, aij, tol)` (static).
fn cs_tol(_i: i32, _j: i32, aij: f64, tol: &f64) -> i32 {
    (aij.abs() > *tol) as i32
}

/// `cs_droptol(A, tol)`: keep all large entries.
pub fn cs_droptol(a: &mut Cs, tol: f64) -> i32 {
    cs_fkeep(a, &|i, j, aij| cs_tol(i, j, aij, &tol))
}
