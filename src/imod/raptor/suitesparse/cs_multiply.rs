//! Translation of `IMOD/raptor/suitesparse/cs_multiply.c`.

use super::Cs;

/// C `cs_multiply`: returns the CSC product `left * right`.
pub fn cs_multiply(left: &Cs, right: &Cs) -> Option<Cs> {
    if !left.is_csc()
        || !right.is_csc()
        || left.columns != right.rows
        || left.column_pointers.len() < left.columns + 1
        || right.column_pointers.len() < right.columns + 1
    {
        return None;
    }
    let left_entries = left.column_pointers[left.columns];
    let right_entries = right.column_pointers[right.columns];
    if left_entries > left.row_indices.len()
        || left_entries > left.values.len()
        || right_entries > right.row_indices.len()
        || right_entries > right.values.len()
    {
        return None;
    }
    let mut marks = vec![0_usize; left.rows];
    let mut workspace = vec![0.0; left.rows];
    let mut pointers = Vec::with_capacity(right.columns + 1);
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for column in 0..right.columns {
        pointers.push(rows.len());
        let mark = column + 1;
        for right_entry in right.column_pointers[column]..right.column_pointers[column + 1] {
            let inner = right.row_indices[right_entry];
            if inner >= left.columns {
                return None;
            }
            let multiplier = right.values[right_entry];
            for left_entry in left.column_pointers[inner]..left.column_pointers[inner + 1] {
                let row = left.row_indices[left_entry];
                if row >= left.rows {
                    return None;
                }
                if marks[row] != mark {
                    marks[row] = mark;
                    rows.push(row);
                    workspace[row] = left.values[left_entry] * multiplier;
                } else {
                    workspace[row] += left.values[left_entry] * multiplier;
                }
            }
        }
        for &row in &rows[*pointers.last()?..] {
            values.push(workspace[row]);
        }
    }
    pointers.push(rows.len());
    Some(Cs {
        nzmax: rows.len(),
        rows: left.rows,
        columns: right.columns,
        column_pointers: pointers,
        row_indices: rows,
        values,
        nz: -1,
    })
}

#[cfg(test)]
mod tests {
    use super::cs_multiply;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn multiply_matches_column_scatter_order() {
        let left = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![1., 2., 3.],
            nz: -1,
        };
        let right = Cs {
            nzmax: 2,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 2],
            row_indices: vec![0, 1],
            values: vec![4., 5.],
            nz: -1,
        };
        let product = cs_multiply(&left, &right).unwrap();
        assert_eq!(product.column_pointers, [0, 2]);
        assert_eq!(product.row_indices, [0, 1]);
        assert_eq!(product.values, [4., 23.]);
    }
}
