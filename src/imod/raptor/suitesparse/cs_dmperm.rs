//! Translation of `IMOD/raptor/suitesparse/cs_dmperm.c`.

use super::{
    Cs, Csd, cs_fkeep::cs_fkeep, cs_maxtrans::cs_maxtrans, cs_permute::cs_permute,
    cs_pinv::cs_pinv, cs_scc::cs_scc, cs_transpose::cs_transpose,
};

/// C `cs_bfs`: breadth-first search for a coarse Dulmage--Mendelsohn set.
fn cs_bfs(
    matrix: &Cs,
    node_count: usize,
    rows: &mut [isize],
    columns: &mut [isize],
    queue: &mut [usize],
    row_match: &[Option<usize>],
    column_match: &[Option<usize>],
    mark: isize,
) -> Option<()> {
    if node_count != columns.len()
        || queue.len() < node_count
        || row_match.len() < node_count
        || column_match.len() < rows.len()
    {
        return None;
    }
    let mut tail = 0;
    for column in 0..node_count {
        if row_match[column].is_some() {
            continue;
        }
        columns[column] = 0;
        queue[tail] = column;
        tail += 1;
    }
    if tail == 0 {
        return Some(());
    }
    let transpose;
    let graph = if mark == 1 {
        matrix
    } else {
        transpose = cs_transpose(matrix, false)?;
        &transpose
    };
    if !graph.is_csc()
        || graph.columns != node_count
        || graph.column_pointers.len() < node_count + 1
        || graph.column_pointers[node_count] > graph.row_indices.len()
    {
        return None;
    }
    let mut head = 0;
    while head < tail {
        let column = queue[head];
        head += 1;
        for entry in graph.column_pointers[column]..graph.column_pointers[column + 1] {
            let row = graph.row_indices[entry];
            if row >= rows.len() || rows[row] >= 0 {
                continue;
            }
            rows[row] = mark;
            let next_column = column_match.get(row).copied().flatten()?;
            if next_column >= columns.len() {
                return None;
            }
            if columns[next_column] >= 0 {
                continue;
            }
            columns[next_column] = mark;
            if tail >= queue.len() {
                return None;
            }
            queue[tail] = next_column;
            tail += 1;
        }
    }
    Some(())
}

/// C `cs_matched`: collects one matched row/column coarse set.
fn cs_matched(
    column_count: usize,
    columns: &[isize],
    column_match: &[Option<usize>],
    rows_permutation: &mut [usize],
    columns_permutation: &mut [usize],
    coarse_columns: &mut [usize; 5],
    coarse_rows: &mut [usize; 5],
    set: usize,
    mark: isize,
) -> Option<()> {
    if set == 0 || set >= coarse_columns.len() || column_count > columns.len() {
        return None;
    }
    let mut column_position = coarse_columns[set];
    let mut row_position = coarse_rows[set - 1];
    for column in 0..column_count {
        if columns[column] != mark {
            continue;
        }
        let row = column_match.get(column).copied().flatten()?;
        if row_position >= rows_permutation.len() || column_position >= columns_permutation.len() {
            return None;
        }
        rows_permutation[row_position] = row;
        columns_permutation[column_position] = column;
        row_position += 1;
        column_position += 1;
    }
    coarse_columns[set + 1] = column_position;
    coarse_rows[set] = row_position;
    Some(())
}

/// C `cs_unmatched`: collects unmatched rows into a permutation vector.
fn cs_unmatched(
    row_count: usize,
    rows: &[isize],
    permutation: &mut [usize],
    coarse_rows: &mut [usize; 5],
    set: usize,
) -> Option<()> {
    if set + 1 >= coarse_rows.len() || row_count > rows.len() {
        return None;
    }
    let mut position = coarse_rows[set];
    for row in 0..row_count {
        if rows[row] != 0 {
            continue;
        }
        if position >= permutation.len() {
            return None;
        }
        permutation[position] = row;
        position += 1;
    }
    coarse_rows[set + 1] = position;
    Some(())
}

/// C `cs_rprune`: retains the R2 rows during fine decomposition.
fn cs_rprune(row: usize, coarse_rows: &[usize; 5]) -> bool {
    row >= coarse_rows[1] && row < coarse_rows[2]
}

/// C `cs_dmperm`: computes the coarse and fine Dulmage--Mendelsohn decomposition.
pub fn cs_dmperm(matrix: &Cs, seed: i32) -> Option<Csd> {
    if !matrix.is_csc()
        || matrix.column_pointers.len() < matrix.columns + 1
        || matrix.column_pointers[matrix.columns] > matrix.row_indices.len()
    {
        return None;
    }
    let m = matrix.rows;
    let n = matrix.columns;
    let mut result = Csd {
        p: vec![0; m],
        q: vec![0; n],
        r: vec![0; m + 6],
        s: vec![0; n + 6],
        nb: 0,
        rr: [0; 5],
        cc: [0; 5],
    };
    let (row_match, column_match) = cs_maxtrans(matrix, seed)?;
    if row_match.len() != m || column_match.len() != n {
        return None;
    }

    let mut row_marks = vec![-1; m];
    let mut column_marks = vec![-1; n];
    cs_bfs(
        matrix,
        n,
        &mut row_marks,
        &mut column_marks,
        &mut result.q,
        &column_match,
        &row_match,
        1,
    )?;
    cs_bfs(
        matrix,
        m,
        &mut column_marks,
        &mut row_marks,
        &mut result.p,
        &row_match,
        &column_match,
        3,
    )?;

    cs_unmatched(n, &column_marks, &mut result.q, &mut result.cc, 0)?;
    cs_matched(
        n,
        &column_marks,
        &column_match,
        &mut result.p,
        &mut result.q,
        &mut result.cc,
        &mut result.rr,
        1,
        1,
    )?;
    cs_matched(
        n,
        &column_marks,
        &column_match,
        &mut result.p,
        &mut result.q,
        &mut result.cc,
        &mut result.rr,
        2,
        -1,
    )?;
    cs_matched(
        n,
        &column_marks,
        &column_match,
        &mut result.p,
        &mut result.q,
        &mut result.cc,
        &mut result.rr,
        3,
        3,
    )?;
    cs_unmatched(m, &row_marks, &mut result.p, &mut result.rr, 3)?;

    let inverse_rows = cs_pinv(Some(&result.p))?;
    let mut fine_matrix = cs_permute(matrix, Some(&inverse_rows), Some(&result.q), false)?;
    let columns = result.cc[3] - result.cc[2];
    if result.cc[2] > 0 {
        fine_matrix
            .column_pointers
            .copy_within(result.cc[2]..=result.cc[3], 0);
    }
    fine_matrix.columns = columns;
    if result.rr[2] - result.rr[1] < m {
        cs_fkeep(&mut fine_matrix, |row, _, _| cs_rprune(row, &result.rr))?;
        if result.rr[1] > 0 {
            for row in &mut fine_matrix.row_indices {
                *row -= result.rr[1];
            }
        }
    }
    fine_matrix.rows = columns;
    let strongly_connected = cs_scc(&fine_matrix)?;

    for index in 0..columns {
        column_marks[index] = result.q[strongly_connected.p[index] + result.cc[2]] as isize;
    }
    for index in 0..columns {
        result.q[index + result.cc[2]] = column_marks[index] as usize;
    }
    for index in 0..columns {
        row_marks[index] = result.p[strongly_connected.p[index] + result.rr[1]] as isize;
    }
    for index in 0..columns {
        result.p[index + result.rr[1]] = row_marks[index] as usize;
    }

    let mut blocks = 0;
    result.r[0] = 0;
    result.s[0] = 0;
    if result.cc[2] > 0 {
        blocks += 1;
    }
    for block in 0..strongly_connected.nb {
        result.r[blocks] = strongly_connected.r[block] + result.rr[1];
        result.s[blocks] = strongly_connected.r[block] + result.cc[2];
        blocks += 1;
    }
    if result.rr[2] < m {
        result.r[blocks] = result.rr[2];
        result.s[blocks] = result.cc[3];
        blocks += 1;
    }
    result.r[blocks] = m;
    result.s[blocks] = n;
    result.nb = blocks;
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::cs_dmperm;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn dmperm_splits_diagonal_entries_into_fine_blocks() {
        let matrix = Cs {
            nzmax: 2,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![0, 1],
            values: vec![1.0; 2],
            nz: -1,
        };
        let result = cs_dmperm(&matrix, 0).unwrap();
        assert_eq!(result.p, [0, 1]);
        assert_eq!(result.q, [0, 1]);
        assert_eq!(result.nb, 2);
        assert_eq!(&result.r[..=result.nb], [0, 1, 2]);
        assert_eq!(&result.s[..=result.nb], [0, 1, 2]);
        assert_eq!(result.rr, [0, 0, 2, 2, 2]);
        assert_eq!(result.cc, [0, 0, 0, 2, 2]);
    }

    #[test]
    fn dmperm_keeps_a_rectangular_coarse_block() {
        let matrix = Cs {
            nzmax: 2,
            rows: 2,
            columns: 3,
            column_pointers: vec![0, 1, 2, 2],
            row_indices: vec![0, 1],
            values: vec![1.0; 2],
            nz: -1,
        };
        let result = cs_dmperm(&matrix, 0).unwrap();
        assert_eq!(result.p, [0, 1]);
        assert_eq!(result.q, [2, 0, 1]);
        assert_eq!(result.r[result.nb], 2);
        assert_eq!(result.s[result.nb], 3);
        assert_eq!(result.cc, [0, 1, 1, 3, 3]);
    }
}
