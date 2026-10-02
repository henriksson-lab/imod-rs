//! Translation of `IMOD/raptor/suitesparse/cs_dropzeros.c`.

use super::cs::Cs;
use super::cs_fkeep::cs_fkeep;

/// `cs_nonzero(i, j, aij, other)` (static).
fn cs_nonzero(_i: i32, _j: i32, aij: f64) -> i32 {
    (aij != 0.0) as i32
}

/// `cs_dropzeros(A)`: keep all nonzero entries.
pub fn cs_dropzeros(a: &mut Cs) -> i32 {
    cs_fkeep(a, &cs_nonzero)
}
