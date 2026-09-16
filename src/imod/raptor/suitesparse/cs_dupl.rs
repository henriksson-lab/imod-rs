//! Translation of `IMOD/raptor/suitesparse/cs_dupl.c`.

use super::Cs;

/// C `cs_dupl`: combines repeated row entries within each CSC column in place.
pub fn cs_dupl(matrix: &mut Cs) -> bool {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return false;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() || entries > matrix.values.len() {
        return false;
    }
    let mut seen = vec![None; matrix.rows];
    let mut retained = 0;
    for column in 0..matrix.columns {
        let start = matrix.column_pointers[column];
        let end = matrix.column_pointers[column + 1];
        let column_start = retained;
        for entry in start..end {
            let row = matrix.row_indices[entry];
            if row >= matrix.rows {
                return false;
            }
            if let Some(previous) = seen[row].filter(|&index| index >= column_start) {
                matrix.values[previous] += matrix.values[entry];
            } else {
                seen[row] = Some(retained);
                matrix.row_indices[retained] = row;
                matrix.values[retained] = matrix.values[entry];
                retained += 1;
            }
        }
        matrix.column_pointers[column] = column_start;
    }
    matrix.column_pointers[matrix.columns] = retained;
    matrix.row_indices.truncate(retained);
    matrix.values.truncate(retained);
    matrix.nzmax = retained;
    true
}

#[cfg(test)]
mod tests {
    use super::cs_dupl;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn duplicate_rows_are_summed_only_within_their_column() {
        let mut matrix = Cs {
            nzmax: 5,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 3, 5],
            row_indices: vec![0, 1, 0, 1, 1],
            values: vec![2., 3., 4., 5., -1.],
            nz: -1,
        };
        assert!(cs_dupl(&mut matrix));
        assert_eq!(matrix.column_pointers, [0, 2, 3]);
        assert_eq!(matrix.row_indices, [0, 1, 1]);
        assert_eq!(matrix.values, [6., 3., 4.]);
    }
}
