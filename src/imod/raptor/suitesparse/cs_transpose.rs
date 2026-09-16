//! Translation of `IMOD/raptor/suitesparse/cs_transpose.c`.
use super::Cs;

/// C `cs_transpose`: transposes a CSC matrix, retaining values when requested.
pub fn cs_transpose(matrix: &Cs, retain_values: bool) -> Option<Cs> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let entries = *matrix.column_pointers.get(matrix.columns)?;
    if entries > matrix.row_indices.len() || (retain_values && entries > matrix.values.len()) {
        return None;
    }
    let mut counts = vec![0_usize; matrix.rows];
    for &row in &matrix.row_indices[..entries] {
        *counts.get_mut(row)? += 1;
    }
    let mut pointers = vec![0; matrix.rows + 1];
    for column in 0..matrix.rows {
        pointers[column + 1] = pointers[column] + counts[column];
    }
    let mut insertion = pointers[..matrix.rows].to_vec();
    let mut rows = vec![0; entries];
    let mut values = vec![0.0; entries];
    for column in 0..matrix.columns {
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            let target = insertion[row];
            rows[target] = column;
            if retain_values {
                values[target] = matrix.values[entry];
            }
            insertion[row] += 1;
        }
    }
    Some(Cs {
        nzmax: entries,
        rows: matrix.columns,
        columns: matrix.rows,
        column_pointers: pointers,
        row_indices: rows,
        values,
        nz: -1,
    })
}
