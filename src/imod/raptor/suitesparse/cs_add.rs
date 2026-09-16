//! Translation of `IMOD/raptor/suitesparse/cs_add.c`.

use super::Cs;

/// C `cs_add`: returns `alpha * left + beta * right` in CSC form.
pub fn cs_add(left: &Cs, right: &Cs, alpha: f64, beta: f64) -> Option<Cs> {
    if !left.is_csc()
        || !right.is_csc()
        || left.rows != right.rows
        || left.columns != right.columns
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
    let mut pointers = Vec::with_capacity(left.columns + 1);
    let mut rows = Vec::with_capacity(left_entries + right_entries);
    let mut values = Vec::with_capacity(left_entries + right_entries);
    for column in 0..left.columns {
        pointers.push(rows.len());
        let mark = column + 1;
        for (matrix, scale) in [(left, alpha), (right, beta)] {
            for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
                let row = matrix.row_indices[entry];
                if row >= matrix.rows {
                    return None;
                }
                if marks[row] != mark {
                    marks[row] = mark;
                    rows.push(row);
                    workspace[row] = scale * matrix.values[entry];
                } else {
                    workspace[row] += scale * matrix.values[entry];
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
        columns: left.columns,
        column_pointers: pointers,
        row_indices: rows,
        values,
        nz: -1,
    })
}

#[cfg(test)]
mod tests {
    use super::cs_add;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn addition_uses_csparse_scatter_order_and_sums_rows() {
        let a = Cs {
            nzmax: 2,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 2],
            row_indices: vec![0, 1],
            values: vec![2., 3.],
            nz: -1,
        };
        let b = Cs {
            nzmax: 2,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 2],
            row_indices: vec![1, 0],
            values: vec![5., 7.],
            nz: -1,
        };
        let sum = cs_add(&a, &b, 2., -1.).unwrap();
        assert_eq!(sum.row_indices, [0, 1]);
        assert_eq!(sum.values, [-3., 1.]);
    }
}
