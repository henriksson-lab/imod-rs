//! Translation of `IMOD/raptor/suitesparse/cs_symperm.c`.

use super::Cs;

/// C `cs_symperm`: returns `A(p,p)` from the stored upper triangle of a symmetric CSC matrix.
pub fn cs_symperm(
    matrix: &Cs,
    inverse_permutation: Option<&[usize]>,
    retain_values: bool,
) -> Option<Cs> {
    if !matrix.is_csc()
        || matrix.rows != matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
        || inverse_permutation.is_some_and(|items| items.len() < matrix.columns)
    {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() || entries > matrix.values.len() {
        return None;
    }
    let mut counts = vec![0_usize; matrix.columns];
    for column in 0..matrix.columns {
        let mapped_column = inverse_permutation.map_or(column, |items| items[column]);
        if mapped_column >= matrix.columns {
            return None;
        }
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row >= matrix.rows {
                return None;
            }
            if row > column {
                continue;
            }
            let mapped_row = inverse_permutation.map_or(row, |items| items[row]);
            if mapped_row >= matrix.rows {
                return None;
            }
            counts[mapped_row.max(mapped_column)] += 1;
        }
    }
    let mut pointers = vec![0; matrix.columns + 1];
    for column in 0..matrix.columns {
        pointers[column + 1] = pointers[column] + counts[column];
    }
    let mut insertion = pointers[..matrix.columns].to_vec();
    let mut rows = vec![0; entries];
    let mut values = vec![0.0; entries];
    for column in 0..matrix.columns {
        let mapped_column = inverse_permutation.map_or(column, |items| items[column]);
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row > column {
                continue;
            }
            let mapped_row = inverse_permutation.map_or(row, |items| items[row]);
            let target_column = mapped_row.max(mapped_column);
            let target = insertion[target_column];
            rows[target] = mapped_row.min(mapped_column);
            if retain_values {
                values[target] = matrix.values[entry];
            }
            insertion[target_column] += 1;
        }
    }
    let used = *pointers.last()?;
    rows.truncate(used);
    values.truncate(used);
    Some(Cs {
        nzmax: used,
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
    use super::cs_symperm;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn symmetric_permutation_only_reads_source_upper_triangle() {
        let matrix = Cs {
            nzmax: 4,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 0, 1],
            values: vec![1., 99., 2., 3.],
            nz: -1,
        };
        let result = cs_symperm(&matrix, Some(&[1, 0]), true).unwrap();
        assert_eq!(result.column_pointers, [0, 1, 3]);
        assert_eq!(result.row_indices, [0, 1, 0]);
        assert_eq!(result.values, [3., 1., 2.]);
    }
}
