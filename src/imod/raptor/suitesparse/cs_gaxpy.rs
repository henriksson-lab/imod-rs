//! Translation of `IMOD/raptor/suitesparse/cs_gaxpy.c`.

use super::Cs;

/// C `cs_gaxpy`: accumulates `matrix * input` into `output`.
pub fn cs_gaxpy(matrix: &Cs, input: &[f64], output: &mut [f64]) -> bool {
    if !matrix.is_csc()
        || input.len() < matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return false;
    }
    for column in 0..matrix.columns {
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row >= output.len() {
                return false;
            }
            output[row] += matrix.values[entry] * input[column];
        }
    }
    true
}
