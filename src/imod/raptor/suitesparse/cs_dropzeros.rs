//! Translation of `IMOD/raptor/suitesparse/cs_dropzeros.c`.
use super::Cs;
use super::cs_fkeep::cs_fkeep;

/// C static `cs_nonzero`.
fn cs_nonzero(_i: usize, _j: usize, aij: f64) -> bool {
    aij != 0.0
}

/// C `cs_dropzeros`: retains all nonzero CSC entries.
pub fn cs_dropzeros(matrix: &mut Cs) -> Option<usize> {
    cs_fkeep(matrix, cs_nonzero)
}
