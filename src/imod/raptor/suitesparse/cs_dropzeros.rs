//! Translation of `IMOD/raptor/suitesparse/cs_dropzeros.c`.
use super::Cs;
use super::cs_fkeep::cs_fkeep;

/// C `cs_dropzeros`: retains all nonzero CSC entries.
pub fn cs_dropzeros(matrix: &mut Cs) -> Option<usize> {
    cs_fkeep(matrix, |_, _, value| value != 0.0)
}
