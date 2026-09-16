//! Translation of `IMOD/raptor/suitesparse/cs_lsolve.c`.

use super::Cs;

/// C `cs_lsolve`: solves `L*x=b` in place for a CSC lower-triangular matrix.
pub fn cs_lsolve(matrix: &Cs, values: &mut [f64]) -> bool {
    if !matrix.is_csc()
        || values.len() < matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return false;
    }
    for column in 0..matrix.columns {
        let diagonal = matrix.column_pointers[column];
        if diagonal >= matrix.values.len() || matrix.values[diagonal] == 0.0 {
            return false;
        }
        values[column] /= matrix.values[diagonal];
        for entry in diagonal + 1..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row >= values.len() {
                return false;
            }
            values[row] -= matrix.values[entry] * values[column];
        }
    }
    true
}
