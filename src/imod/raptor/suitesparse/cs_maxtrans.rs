//! Translation of `IMOD/raptor/suitesparse/cs_maxtrans.c`.

use super::{Cs, cs_randperm::cs_randperm, cs_transpose::cs_transpose};

/// C `cs_augment`: finds an augmenting path from one column and extends the
/// row-to-column matching in place.
fn cs_augment(
    start_column: usize,
    matrix: &Cs,
    row_match: &mut [Option<usize>],
    cheap: &mut [usize],
    marks: &mut [Option<usize>],
    column_stack: &mut [usize],
    row_stack: &mut [usize],
    position_stack: &mut [usize],
    traversal: usize,
) -> Option<()> {
    let mut found = false;
    let mut head = 0_usize;
    column_stack[0] = start_column;
    loop {
        let column = column_stack[head];
        if column >= matrix.columns {
            return None;
        }
        if marks[column] != Some(traversal) {
            marks[column] = Some(traversal);
            let end = matrix.column_pointers[column + 1];
            let mut position = cheap[column];
            while position < end && !found {
                let row = *matrix.row_indices.get(position)?;
                if row >= row_match.len() {
                    return None;
                }
                found = row_match[row].is_none();
                if found {
                    row_stack[head] = row;
                }
                position += 1;
            }
            cheap[column] = position;
            if found {
                break;
            }
            position_stack[head] = matrix.column_pointers[column];
        }
        let end = matrix.column_pointers[column + 1];
        let mut position = position_stack[head];
        while position < end {
            let row = *matrix.row_indices.get(position)?;
            let Some(matched_column) = *row_match.get(row)? else {
                return None;
            };
            if marks.get(matched_column).copied()? == Some(traversal) {
                position += 1;
                continue;
            }
            position_stack[head] = position + 1;
            row_stack[head] = row;
            head += 1;
            if head >= column_stack.len() {
                return None;
            }
            column_stack[head] = matched_column;
            break;
        }
        if position == end {
            if head == 0 {
                return Some(());
            }
            head -= 1;
        }
    }
    for depth in (0..=head).rev() {
        row_match[row_stack[depth]] = Some(column_stack[depth]);
    }
    Some(())
}

/// C `cs_maxtrans`: computes a maximum transversal.
///
/// The returned pair is `(jmatch, imatch)`: `jmatch[row]` is its matched
/// column and `imatch[column]` is its matched row. `None` corresponds to C's
/// `-1`.
pub fn cs_maxtrans(matrix: &Cs, seed: i32) -> Option<(Vec<Option<usize>>, Vec<Option<usize>>)> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len()
        || matrix
            .column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return None;
    }
    let original_rows = matrix.rows;
    let original_columns = matrix.columns;
    let mut nonempty_rows = vec![false; original_rows];
    let mut nonempty_columns = 0;
    let mut diagonal_entries = 0;
    for column in 0..original_columns {
        nonempty_columns +=
            usize::from(matrix.column_pointers[column] < matrix.column_pointers[column + 1]);
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row >= original_rows {
                return None;
            }
            nonempty_rows[row] = true;
            diagonal_entries += usize::from(column == row);
        }
    }
    if diagonal_entries == original_rows.min(original_columns) {
        let mut row_match = vec![None; original_rows];
        let mut column_match = vec![None; original_columns];
        for index in 0..diagonal_entries {
            row_match[index] = Some(index);
            column_match[index] = Some(index);
        }
        return Some((row_match, column_match));
    }
    let nonempty_row_count = nonempty_rows.into_iter().filter(|&present| present).count();
    let transposed = nonempty_row_count < nonempty_columns;
    let working = if transposed {
        cs_transpose(matrix, false)?
    } else {
        matrix.clone()
    };
    let rows = working.rows;
    let columns = working.columns;
    if working.column_pointers.len() < columns + 1 {
        return None;
    }
    let mut row_match = vec![None; rows];
    let mut cheap = working.column_pointers[..columns].to_vec();
    let mut marks = vec![None; columns];
    let mut column_stack = vec![0; columns.max(1)];
    let mut row_stack = vec![0; columns.max(1)];
    let mut position_stack = vec![0; columns.max(1)];
    let permutation = cs_randperm(columns, seed);
    for traversal in 0..columns {
        let column = permutation
            .as_ref()
            .map_or(traversal, |order| order[traversal]);
        cs_augment(
            column,
            &working,
            &mut row_match,
            &mut cheap,
            &mut marks,
            &mut column_stack,
            &mut row_stack,
            &mut position_stack,
            traversal,
        )?;
    }
    let mut column_match = vec![None; columns];
    for (row, &column) in row_match.iter().enumerate() {
        if let Some(column) = column {
            column_match[column] = Some(row);
        }
    }
    if transposed {
        Some((column_match, row_match))
    } else {
        Some((row_match, column_match))
    }
}

#[cfg(test)]
mod tests {
    use super::cs_maxtrans;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn maximum_transversal_uses_augmenting_paths_for_a_square_pattern() {
        let matrix = Cs {
            nzmax: 5,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 2, 3, 5],
            row_indices: vec![0, 1, 1, 1, 2],
            values: vec![1.0; 5],
            nz: -1,
        };
        assert_eq!(
            cs_maxtrans(&matrix, 0),
            Some((
                vec![Some(0), Some(1), Some(2)],
                vec![Some(0), Some(1), Some(2)],
            ))
        );
    }

    #[test]
    fn maximum_transversal_swaps_match_views_after_the_transpose_path() {
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 3,
            column_pointers: vec![0, 1, 2, 3],
            row_indices: vec![0, 0, 1],
            values: vec![1.0; 3],
            nz: -1,
        };
        let (row_match, column_match) = cs_maxtrans(&matrix, -1).unwrap();
        assert_eq!(row_match.len(), 2);
        assert_eq!(column_match.len(), 3);
        assert_eq!(row_match.iter().flatten().count(), 2);
        assert_eq!(column_match.iter().flatten().count(), 2);
        for (row, column) in row_match.into_iter().enumerate() {
            assert_eq!(column_match[column.unwrap()], Some(row));
        }
    }
}
