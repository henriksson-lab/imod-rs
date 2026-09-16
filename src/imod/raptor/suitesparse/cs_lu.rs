//! Translation of `IMOD/raptor/suitesparse/cs_lu.c`.

use super::{Cs, Csn, Css, cs_spsolve::cs_spsolve};

/// C `cs_lu`: sparse LU factorization with threshold partial pivoting.
pub fn cs_lu(matrix: &Cs, symbolic: &Css, tolerance: f64) -> Option<Csn> {
    let n = matrix.columns;
    if !matrix.is_csc()
        || matrix.rows != n
        || matrix.column_pointers.len() < n + 1
        || symbolic.q.as_ref().is_some_and(|q| q.len() != n)
    {
        return None;
    }
    let entries = matrix.column_pointers[n];
    if entries > matrix.row_indices.len()
        || entries > matrix.values.len()
        || matrix
            .column_pointers
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        || matrix.row_indices[..entries].iter().any(|&row| row >= n)
    {
        return None;
    }
    let capacity = usize::try_from(symbolic.lnz.max(symbolic.unz) as u128).ok()?;
    let initial_capacity = capacity.max(n);
    let mut lower = Cs {
        nzmax: initial_capacity,
        rows: n,
        columns: n,
        column_pointers: vec![0; n + 1],
        row_indices: Vec::with_capacity(initial_capacity),
        values: Vec::with_capacity(initial_capacity),
        nz: -1,
    };
    let mut upper = lower.clone();
    let mut workspace = vec![0.0; n];
    let mut pattern = vec![0; 2 * n];
    let mut inverse_pivot = vec![None; n];

    for column in 0..n {
        let lower_count = lower.row_indices.len();
        let upper_count = upper.row_indices.len();
        lower.column_pointers[column] = lower_count;
        upper.column_pointers[column] = upper_count;
        // Later columns are empty during this incremental factorization.  The
        // owned representation retains monotonic CSC offsets while `cs_spsolve`
        // traverses only established factor columns.
        lower.column_pointers[column + 1..].fill(lower_count);
        upper.column_pointers[column + 1..].fill(upper_count);
        let source_column = symbolic.q.as_ref().map_or(column, |q| q[column]);
        if source_column >= n {
            return None;
        }
        let top = cs_spsolve(
            &lower,
            matrix,
            source_column,
            &mut pattern,
            &mut workspace,
            Some(&inverse_pivot),
            true,
        )?;

        let mut pivot_row = None;
        let mut maximum = -1.0;
        for &row in &pattern[top..n] {
            if inverse_pivot[row].is_none() {
                let magnitude = workspace[row].abs();
                if magnitude > maximum {
                    maximum = magnitude;
                    pivot_row = Some(row);
                }
            } else {
                upper.row_indices.push(inverse_pivot[row]?);
                upper.values.push(workspace[row]);
            }
        }
        let mut pivot_row = pivot_row.filter(|_| maximum > 0.0)?;
        if inverse_pivot[source_column].is_none()
            && workspace[source_column].abs() >= maximum * tolerance
        {
            pivot_row = source_column;
        }
        let pivot = workspace[pivot_row];
        upper.row_indices.push(column);
        upper.values.push(pivot);
        inverse_pivot[pivot_row] = Some(column);
        lower.row_indices.push(pivot_row);
        lower.values.push(1.0);
        for &row in &pattern[top..n] {
            if inverse_pivot[row].is_none() {
                lower.row_indices.push(row);
                lower.values.push(workspace[row] / pivot);
            }
            workspace[row] = 0.0;
        }
    }
    lower.column_pointers[n] = lower.row_indices.len();
    upper.column_pointers[n] = upper.row_indices.len();
    // Once all rows are pivoted, factor row indices use the final pivot order,
    // exactly as the source's final `Li[p] = pinv[Li[p]]` pass does.
    for row in &mut lower.row_indices {
        *row = inverse_pivot[*row]?;
    }
    lower.nzmax = lower.row_indices.len();
    upper.nzmax = upper.row_indices.len();
    Some(Csn {
        l: Some(lower),
        u: Some(upper),
        pinv: Some(inverse_pivot.into_iter().collect::<Option<Vec<_>>>()?),
        beta: None,
    })
}

#[cfg(test)]
mod tests {
    use super::cs_lu;
    use crate::imod::raptor::suitesparse::{Cs, cs_sqr::cs_sqr};

    #[test]
    fn lu_factorization_uses_partial_pivoting_and_reconstructs_matrix() {
        let matrix = Cs {
            nzmax: 4,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 0, 1],
            values: vec![0.0, 2.0, 1.0, 3.0],
            nz: -1,
        };
        let symbolic = cs_sqr(0, &matrix, false).unwrap();
        let numeric = cs_lu(&matrix, &symbolic, 1.0).unwrap();
        assert_eq!(numeric.pinv, Some(vec![1, 0]));
        let lower = numeric.l.unwrap();
        let upper = numeric.u.unwrap();
        assert_eq!(lower.column_pointers, [0, 2, 3]);
        assert_eq!(lower.row_indices, [0, 1, 1]);
        assert_eq!(lower.values, [1.0, 0.0, 1.0]);
        assert_eq!(upper.column_pointers, [0, 1, 3]);
        assert_eq!(upper.row_indices, [0, 0, 1]);
        assert_eq!(upper.values, [2.0, 3.0, 1.0]);
    }

    #[test]
    fn lu_rejects_a_singular_matrix() {
        let singular = Cs {
            nzmax: 2,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![1.0, 2.0],
            nz: -1,
        };
        assert_eq!(
            cs_lu(&singular, &cs_sqr(0, &singular, false).unwrap(), 1.0),
            None
        );
    }
}
