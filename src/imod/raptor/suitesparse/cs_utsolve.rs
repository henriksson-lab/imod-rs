//! Translation of `IMOD/raptor/suitesparse/cs_utsolve.c`.

use super::Cs;

/// C `cs_utsolve`: solves `U'*x=b` in place for a CSC upper-triangular matrix.
pub fn cs_utsolve(matrix: &Cs, values: &mut [f64]) -> bool {
    if !matrix.is_csc()
        || values.len() < matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return false;
    }
    for column in 0..matrix.columns {
        let diagonal = matrix.column_pointers[column + 1].checked_sub(1);
        let Some(diagonal) = diagonal else {
            return false;
        };
        if diagonal >= matrix.values.len() || matrix.values[diagonal] == 0.0 {
            return false;
        }
        for entry in matrix.column_pointers[column]..diagonal {
            let row = matrix.row_indices[entry];
            if row >= values.len() {
                return false;
            }
            values[column] -= matrix.values[entry] * values[row];
        }
        values[column] /= matrix.values[diagonal];
    }
    true
}
