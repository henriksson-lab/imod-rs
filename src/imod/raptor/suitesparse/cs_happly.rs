//! Translation of `IMOD/raptor/suitesparse/cs_happly.c`.

use super::Cs;

/// C `cs_happly`: applies a CSC Householder vector column to `values`.
pub fn cs_happly(matrix: &Cs, column: usize, beta: f64, values: &mut [f64]) -> bool {
    if !matrix.is_csc() || column + 1 >= matrix.column_pointers.len() {
        return false;
    }
    let mut tau = 0.0;
    for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
        let row = matrix.row_indices[entry];
        if row >= values.len() || entry >= matrix.values.len() {
            return false;
        }
        tau += matrix.values[entry] * values[row];
    }
    tau *= beta;
    for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
        values[matrix.row_indices[entry]] -= matrix.values[entry] * tau;
    }
    true
}
