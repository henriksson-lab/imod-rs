//! Translation of `IMOD/raptor/suitesparse/cs_spsolve.c`.

use super::{Cs, cs_reach::cs_reach};

/// C `cs_spsolve`: solves one sparse right-hand-side column through a
/// triangular CSC factor.
///
/// `inverse_pivot[row]` is the factor column for an original row; `None`
/// corresponds to CSparse's negative, not-yet-pivotal sentinel.  The return
/// is the first occupied entry of `pattern`, whose suffix through `n` holds
/// the reachability order used for the solve.
pub fn cs_spsolve(
    factor: &Cs,
    right_hand_side: &Cs,
    column: usize,
    pattern: &mut [usize],
    values: &mut [f64],
    inverse_pivot: Option<&[Option<usize>]>,
    lower: bool,
) -> Option<usize> {
    let n = factor.columns;
    if !factor.is_csc()
        || !right_hand_side.is_csc()
        || factor.rows != n
        || right_hand_side.rows != n
        || column >= right_hand_side.columns
        || pattern.len() < 2 * n
        || values.len() < n
        || factor.column_pointers.len() < n + 1
        || right_hand_side.column_pointers.len() < right_hand_side.columns + 1
        || inverse_pivot.is_some_and(|pivot| pivot.len() < n)
    {
        return None;
    }
    let factor_entries = factor.column_pointers[n];
    let rhs_entries = right_hand_side.column_pointers[right_hand_side.columns];
    if factor_entries > factor.row_indices.len()
        || factor_entries > factor.values.len()
        || rhs_entries > right_hand_side.row_indices.len()
        || rhs_entries > right_hand_side.values.len()
        || factor
            .column_pointers
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        || right_hand_side
            .column_pointers
            .windows(2)
            .any(|pair| pair[0] > pair[1])
    {
        return None;
    }

    let pivot = inverse_pivot
        .map(|items| {
            items
                .iter()
                .map(|entry| match entry {
                    Some(value) if *value < n => Ok(*value as isize),
                    Some(_) => Err(()),
                    None => Ok(-1),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .ok()?;
    let top = cs_reach(factor, right_hand_side, column, pattern, pivot.as_deref())?;
    for &node in &pattern[top..n] {
        *values.get_mut(node)? = 0.0;
    }
    for entry in
        right_hand_side.column_pointers[column]..right_hand_side.column_pointers[column + 1]
    {
        let row = right_hand_side.row_indices[entry];
        *values.get_mut(row)? = right_hand_side.values[entry];
    }
    for position in top..n {
        let row = pattern[position];
        let mapped_column = match inverse_pivot {
            Some(pivot) => pivot.get(row).copied().flatten(),
            None => Some(row),
        };
        let Some(mapped_column) = mapped_column else {
            continue;
        };
        let start = factor.column_pointers[mapped_column];
        let end = factor.column_pointers[mapped_column + 1];
        let diagonal = if lower { start } else { end.checked_sub(1)? };
        let divisor = *factor.values.get(diagonal)?;
        if divisor == 0.0 {
            return None;
        }
        values[row] /= divisor;
        let solved = values[row];
        let (from, to) = if lower {
            (start.checked_add(1)?, end)
        } else {
            (start, diagonal)
        };
        for entry in from..to {
            let target = factor.row_indices[entry];
            *values.get_mut(target)? -= factor.values[entry] * solved;
        }
    }
    Some(top)
}

#[cfg(test)]
mod tests {
    use super::cs_spsolve;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn sparse_lower_solve_preserves_csparse_reach_pattern() {
        let lower = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![2.0, 3.0, 4.0],
            nz: -1,
        };
        let rhs = Cs {
            nzmax: 1,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![2.0],
            nz: -1,
        };
        let mut pattern = vec![usize::MAX; 4];
        let mut values = vec![0.0; 2];
        assert_eq!(
            cs_spsolve(&lower, &rhs, 0, &mut pattern, &mut values, None, true),
            Some(0)
        );
        assert_eq!(&pattern[..2], &[0, 1]);
        assert_eq!(values, [1.0, -0.75]);
    }
}
