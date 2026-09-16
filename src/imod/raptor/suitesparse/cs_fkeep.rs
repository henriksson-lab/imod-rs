//! Translation of `IMOD/raptor/suitesparse/cs_fkeep.c`.

use super::Cs;

/// C `cs_fkeep`: removes CSC entries rejected by `keep`, returning the retained count.
pub fn cs_fkeep(matrix: &mut Cs, mut keep: impl FnMut(usize, usize, f64) -> bool) -> Option<usize> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let mut retained = 0;
    for column in 0..matrix.columns {
        let start = matrix.column_pointers[column];
        let end = matrix.column_pointers[column + 1];
        if end > matrix.row_indices.len() || end > matrix.values.len() {
            return None;
        }
        matrix.column_pointers[column] = retained;
        for entry in start..end {
            let row = matrix.row_indices[entry];
            let value = matrix.values[entry];
            if keep(row, column, value) {
                matrix.row_indices[retained] = row;
                matrix.values[retained] = value;
                retained += 1;
            }
        }
    }
    matrix.column_pointers[matrix.columns] = retained;
    matrix.row_indices.truncate(retained);
    matrix.values.truncate(retained);
    matrix.nzmax = retained;
    Some(retained)
}
