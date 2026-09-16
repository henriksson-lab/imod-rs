//! Translation of `IMOD/raptor/suitesparse/cs_entry.c`.

use super::Cs;

/// C `cs_entry`: appends one `(row, column, value)` entry to a triplet matrix.
pub fn cs_entry(matrix: &mut Cs, row: usize, column: usize, value: f64) -> bool {
    if !matrix.is_triplet() {
        return false;
    }
    let entry = matrix.nz as usize;
    if entry >= matrix.nzmax
        || matrix.column_pointers.len() <= entry
        || matrix.row_indices.len() <= entry
        || matrix.values.len() <= entry
    {
        let capacity = matrix
            .nzmax
            .max(matrix.column_pointers.len())
            .max(matrix.row_indices.len())
            .max(matrix.values.len())
            .saturating_mul(2)
            .max(entry + 1)
            .max(1);
        matrix.column_pointers.resize(capacity, 0);
        matrix.row_indices.resize(capacity, 0);
        matrix.values.resize(capacity, 0.0);
        matrix.nzmax = capacity;
    }
    matrix.values[entry] = value;
    matrix.row_indices[entry] = row;
    matrix.column_pointers[entry] = column;
    matrix.nz += 1;
    matrix.rows = matrix.rows.max(row + 1);
    matrix.columns = matrix.columns.max(column + 1);
    true
}

#[cfg(test)]
mod tests {
    use super::cs_entry;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn entry_grows_triplet_storage_as_c_does() {
        let mut matrix = Cs {
            nzmax: 1,
            nz: 0,
            ..Cs::default()
        };
        assert!(cs_entry(&mut matrix, 2, 3, 4.5));
        assert!(cs_entry(&mut matrix, 0, 1, -2.0));
        assert_eq!(matrix.nz, 2);
        assert_eq!((matrix.rows, matrix.columns), (3, 4));
        assert_eq!(&matrix.row_indices[..2], &[2, 0]);
        assert_eq!(&matrix.column_pointers[..2], &[3, 1]);
    }
}
