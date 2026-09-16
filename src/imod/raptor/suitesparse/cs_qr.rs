//! Translation of `IMOD/raptor/suitesparse/cs_qr.c`.

use super::{Cs, Csn, Css, cs_happly::cs_happly, cs_house::cs_house};

/// C `cs_qr`: sparse QR factorization `[V,beta,pinv,R] = qr(A)`.
pub fn cs_qr(matrix: &Cs, symbolic: &Css) -> Option<Csn> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() || entries > matrix.values.len() {
        return None;
    }
    let rows = matrix.rows;
    let columns = matrix.columns;
    let parent = symbolic.parent.as_ref()?;
    let inverse_rows = symbolic.pinv.as_ref()?;
    let leftmost = symbolic.leftmost.as_ref()?;
    if parent.len() != columns
        || inverse_rows.len() < rows
        || leftmost.len() != rows
        || symbolic.q.as_ref().is_some_and(|q| q.len() != columns)
        || symbolic.m2 < rows
    {
        return None;
    }
    let extended_rows = symbolic.m2;
    let mut marks = vec![-1_isize; extended_rows];
    let mut values = vec![0.0; extended_rows];
    let mut beta = vec![0.0; columns];
    let mut v_pointers = Vec::with_capacity(columns + 1);
    let mut v_rows = Vec::with_capacity(symbolic.lnz.max(0.0) as usize);
    let mut v_values = Vec::with_capacity(symbolic.lnz.max(0.0) as usize);
    let mut r_pointers = Vec::with_capacity(columns + 1);
    let mut r_rows = Vec::with_capacity(symbolic.unz.max(0.0) as usize);
    let mut r_values = Vec::with_capacity(symbolic.unz.max(0.0) as usize);

    for column in 0..columns {
        r_pointers.push(r_rows.len());
        v_pointers.push(v_rows.len());
        let v_start = v_rows.len();
        marks[column] = column as isize;
        v_rows.push(column);
        let source_column = symbolic.q.as_ref().map_or(column, |q| q[column]);
        if source_column >= columns {
            return None;
        }
        let mut pattern = Vec::new();
        for entry in
            matrix.column_pointers[source_column]..matrix.column_pointers[source_column + 1]
        {
            let row = matrix.row_indices[entry];
            if row >= rows {
                return None;
            }
            let mut node = leftmost[row];
            let mut path = Vec::new();
            for _ in 0..columns {
                if node >= columns {
                    return None;
                }
                if marks[node] == column as isize {
                    break;
                }
                path.push(node);
                marks[node] = column as isize;
                node = match parent[node] {
                    Some(ancestor) if ancestor < columns => ancestor,
                    _ => return None,
                };
            }
            if node >= columns || marks[node] != column as isize {
                return None;
            }
            pattern.extend(path);
            let permuted_row = inverse_rows[row];
            if permuted_row >= extended_rows {
                return None;
            }
            values[permuted_row] = matrix.values[entry];
            if permuted_row > column && marks[permuted_row] < column as isize {
                v_rows.push(permuted_row);
                marks[permuted_row] = column as isize;
            }
        }
        for node in pattern {
            let reflector = Cs {
                nzmax: v_rows.len(),
                rows: extended_rows,
                columns,
                column_pointers: {
                    let mut pointers = v_pointers.clone();
                    pointers.push(v_rows.len());
                    pointers
                },
                row_indices: v_rows.clone(),
                values: v_values.clone(),
                nz: -1,
            };
            if node >= beta.len() || !cs_happly(&reflector, node, beta[node], &mut values) {
                return None;
            }
            r_rows.push(node);
            r_values.push(values[node]);
            values[node] = 0.0;
            if parent[node] == Some(column) {
                let start = v_pointers[node];
                let end = if node + 1 < v_pointers.len() {
                    v_pointers[node + 1]
                } else {
                    v_rows.len()
                };
                let inherited_rows = v_rows[start..end].to_vec();
                for row in inherited_rows {
                    if marks[row] < column as isize {
                        v_rows.push(row);
                        marks[row] = column as isize;
                    }
                }
            }
        }
        for &row in &v_rows[v_start..] {
            v_values.push(values[row]);
            values[row] = 0.0;
        }
        r_rows.push(column);
        let (diagonal, column_beta) = cs_house(&mut v_values[v_start..])?;
        r_values.push(diagonal);
        beta[column] = column_beta;
    }
    r_pointers.push(r_rows.len());
    v_pointers.push(v_rows.len());
    Some(Csn {
        l: Some(Cs {
            nzmax: v_rows.len(),
            rows: extended_rows,
            columns,
            column_pointers: v_pointers,
            row_indices: v_rows,
            values: v_values,
            nz: -1,
        }),
        u: Some(Cs {
            nzmax: r_rows.len(),
            rows: extended_rows,
            columns,
            column_pointers: r_pointers,
            row_indices: r_rows,
            values: r_values,
            nz: -1,
        }),
        beta: Some(beta),
        ..Csn::default()
    })
}
