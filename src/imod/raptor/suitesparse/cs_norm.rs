//! Translation of `IMOD/raptor/suitesparse/cs_norm.c`.

use super::Cs;

/// C `cs_norm`: returns the largest absolute column sum, or `None` for non-CSC input.
pub fn cs_norm(matrix: &Cs) -> Option<f64> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let mut norm = 0.0_f64;
    for column in 0..matrix.columns {
        let mut sum = 0.0;
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            sum += matrix.values.get(entry)?.abs();
        }
        norm = norm.max(sum);
    }
    Some(norm)
}
