//! Translation of `IMOD/raptor/suitesparse/cs_sqr.c`.

use super::{
    Cs, Css, cs_amd::cs_amd, cs_counts::cs_counts, cs_etree::cs_etree, cs_permute::cs_permute,
    cs_post::cs_post,
};

/// C `cs_vcount`: computes QR row ordering and the nonzero count of V.
fn cs_vcount(matrix: &Cs, symbolic: &mut Css) -> Option<()> {
    let rows = matrix.rows;
    let columns = matrix.columns;
    let parent = symbolic.parent.as_ref()?;
    if parent.len() != columns || matrix.column_pointers.len() < columns + 1 {
        return None;
    }
    let mut inverse_rows = vec![None; rows + columns];
    let mut leftmost = vec![None; rows];
    for column in (0..columns).rev() {
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = *matrix.row_indices.get(entry)?;
            if row >= rows {
                return None;
            }
            leftmost[row] = Some(column);
        }
    }
    let mut next = vec![None; rows];
    let mut head = vec![None; columns];
    let mut tail = vec![None; columns];
    let mut queued = vec![0_usize; columns];
    for row in (0..rows).rev() {
        let Some(column) = leftmost[row] else {
            continue;
        };
        if queued[column] == 0 {
            tail[column] = Some(row);
        }
        next[row] = head[column];
        head[column] = Some(row);
        queued[column] += 1;
    }
    symbolic.lnz = 0.0;
    symbolic.m2 = rows;
    for column in 0..columns {
        let queue_length = queued[column];
        let row = head[column].unwrap_or_else(|| {
            let fictitious = symbolic.m2;
            symbolic.m2 += 1;
            fictitious
        });
        symbolic.lnz += 1.0;
        inverse_rows[row] = Some(column);
        if queue_length == 0 {
            continue;
        }
        queued[column] -= 1;
        if queued[column] == 0 {
            continue;
        }
        symbolic.lnz += queued[column] as f64;
        if let Some(ancestor) = parent[column] {
            if ancestor >= columns {
                return None;
            }
            if queued[ancestor] == 0 {
                tail[ancestor] = tail[column];
            }
            let tail_row = tail[column]?;
            next[tail_row] = head[ancestor];
            head[ancestor] = next[row];
            queued[ancestor] = queued[ancestor].checked_add(queued[column])?;
        }
    }
    let mut next_column = columns;
    for value in inverse_rows.iter_mut().take(rows) {
        if value.is_none() {
            *value = Some(next_column);
            next_column += 1;
        }
    }
    symbolic.pinv = Some(
        inverse_rows[..symbolic.m2]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()?,
    );
    symbolic.leftmost = Some(
        leftmost
            .into_iter()
            .map(|column| column.unwrap_or(columns))
            .collect(),
    );
    Some(())
}

/// C `cs_sqr`: symbolic ordering and analysis for QR or LU.
pub fn cs_sqr(order: i32, matrix: &Cs, qr: bool) -> Option<Css> {
    if !matrix.is_csc()
        || !(0..=3).contains(&order)
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() || entries > matrix.values.len() {
        return None;
    }
    let mut symbolic = Css {
        q: if order == 0 {
            None
        } else {
            Some(cs_amd(order, matrix)?)
        },
        ..Css::default()
    };
    if !qr {
        let estimate = entries.checked_mul(4)?.checked_add(matrix.columns)?;
        symbolic.lnz = estimate as f64;
        symbolic.unz = estimate as f64;
        return Some(symbolic);
    }
    let ordered = if let Some(permutation) = symbolic.q.as_deref() {
        cs_permute(matrix, None, Some(permutation), false)?
    } else {
        matrix.clone()
    };
    let parent = cs_etree(&ordered, true)?;
    let post = cs_post(&parent)?;
    let counts = cs_counts(&ordered, &parent, &post, true)?;
    symbolic.parent = Some(parent);
    symbolic.cp = Some(counts.clone());
    cs_vcount(&ordered, &mut symbolic)?;
    symbolic.unz = counts.into_iter().map(|count| count as f64).sum();
    (symbolic.lnz >= 0.0 && symbolic.unz >= 0.0).then_some(symbolic)
}

#[cfg(test)]
mod tests {
    use super::cs_sqr;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn square_qr_analysis_builds_row_and_column_symbolics() {
        let matrix = Cs {
            nzmax: 4,
            rows: 3,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 1, 2],
            values: vec![1.0; 4],
            nz: -1,
        };
        let symbolic = cs_sqr(0, &matrix, true).unwrap();
        assert_eq!(symbolic.q, None);
        assert_eq!(symbolic.m2, 3);
        assert_eq!(symbolic.pinv, Some(vec![0, 1, 2]));
        assert_eq!(symbolic.leftmost, Some(vec![0, 0, 1]));
        assert_eq!(symbolic.unz, 3.0);
    }
}
