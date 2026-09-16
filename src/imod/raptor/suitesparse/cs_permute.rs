//! Translation of `IMOD/raptor/suitesparse/cs_permute.c`.

use super::Cs;

/// C `cs_permute`: creates `A(p,q)` from a CSC matrix, accepting absent identity permutations.
pub fn cs_permute(
    matrix: &Cs,
    inverse_rows: Option<&[usize]>,
    columns: Option<&[usize]>,
    retain_values: bool,
) -> Option<Cs> {
    if !matrix.is_csc()
        || matrix.column_pointers.len() < matrix.columns + 1
        || inverse_rows.is_some_and(|items| items.len() < matrix.rows)
        || columns.is_some_and(|items| items.len() < matrix.columns)
    {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() || entries > matrix.values.len() {
        return None;
    }
    let mut pointers = Vec::with_capacity(matrix.columns + 1);
    let mut rows = Vec::with_capacity(entries);
    let mut values = Vec::with_capacity(entries);
    for destination in 0..matrix.columns {
        pointers.push(rows.len());
        let source = columns.map_or(destination, |items| items[destination]);
        if source >= matrix.columns {
            return None;
        }
        for entry in matrix.column_pointers[source]..matrix.column_pointers[source + 1] {
            let row = matrix.row_indices[entry];
            rows.push(inverse_rows.map_or(row, |items| items[row]));
            if retain_values {
                values.push(matrix.values[entry]);
            } else {
                values.push(0.0);
            }
        }
    }
    pointers.push(rows.len());
    Some(Cs {
        nzmax: rows.len(),
        rows: matrix.rows,
        columns: matrix.columns,
        column_pointers: pointers,
        row_indices: rows,
        values,
        nz: -1,
    })
}

#[cfg(test)]
mod tests {
    use super::cs_permute;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn permute_matches_csparse_column_then_row_mapping() {
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 3],
            row_indices: vec![1, 0, 1],
            values: vec![2., 4., 3.],
            nz: -1,
        };
        let result = cs_permute(&matrix, Some(&[1, 0]), Some(&[1, 0]), true).unwrap();
        assert_eq!(result.column_pointers, [0, 2, 3]);
        assert_eq!(result.row_indices, [1, 0, 0]);
        assert_eq!(result.values, [4., 3., 2.]);
    }
}
