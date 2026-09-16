//! Translation of `IMOD/raptor/suitesparse/cs_counts.c`.

use super::{Cs, cs_leaf::cs_leaf, cs_transpose::cs_transpose};

/// C static `init_ata`.
///
/// `transpose` is the source `AT = A'`; `head` and `next` are the linked
/// lists laid out in C's workspace after the other four `n`-element regions.
fn init_ata(
    transpose: &Cs,
    post: &[usize],
    head: &mut [Option<usize>],
    next: &mut [Option<usize>],
) -> Option<()> {
    let rows = transpose.columns;
    let nodes = transpose.rows;
    if post.len() != nodes
        || head.len() < nodes + 1
        || next.len() < rows
        || transpose.column_pointers.len() < rows + 1
    {
        return None;
    }
    let mut inverse_post = vec![0; nodes];
    for (position, &node) in post.iter().enumerate() {
        if node >= nodes {
            return None;
        }
        inverse_post[node] = position;
    }
    for row in 0..rows {
        let mut minimum = nodes;
        let start = transpose.column_pointers[row];
        let end = transpose.column_pointers[row + 1];
        if start > end || end > transpose.row_indices.len() {
            return None;
        }
        for entry in start..end {
            let column = transpose.row_indices[entry];
            if column >= nodes {
                return None;
            }
            minimum = minimum.min(inverse_post[column]);
        }
        next[row] = head[minimum];
        head[minimum] = Some(row);
    }
    Some(())
}

/// C `cs_counts`: computes column counts for `LL' = A` or `LL' = A' A`.
///
/// Parent links use `None` for C's `-1`.  The input ordering is the
/// postorder returned by [`super::cs_post::cs_post`].
pub fn cs_counts(
    matrix: &Cs,
    parent: &[Option<usize>],
    post: &[usize],
    ata: bool,
) -> Option<Vec<usize>> {
    if !matrix.is_csc()
        || parent.len() != matrix.columns
        || post.len() != matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
        || parent.iter().flatten().any(|&node| node >= matrix.columns)
        || post.iter().any(|&node| node >= matrix.columns)
    {
        return None;
    }
    let mut seen_post = vec![false; matrix.columns];
    for &node in post {
        if seen_post[node] {
            return None;
        }
        seen_post[node] = true;
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
    let transpose = cs_transpose(matrix, false)?;
    let node_count = matrix.columns;
    let mut delta = vec![0_isize; node_count];
    let mut ancestor: Vec<usize> = (0..node_count).collect();
    let mut maxfirst = vec![None; node_count];
    let mut prevleaf = vec![None; node_count];
    let mut first = vec![None; node_count];

    for (position, &post_node) in post.iter().enumerate() {
        if first[post_node].is_none() {
            delta[post_node] = 1;
        }
        let mut node = Some(post_node);
        for _ in 0..node_count {
            let Some(current) = node else { break };
            if first[current].is_some() {
                break;
            }
            first[current] = Some(position);
            node = parent[current];
        }
    }
    if first.iter().any(Option::is_none) {
        return None;
    }

    let mut head = if ata {
        Some(vec![None; node_count + 1])
    } else {
        None
    };
    let mut next = if ata {
        Some(vec![None; matrix.rows])
    } else {
        None
    };
    if ata {
        init_ata(&transpose, post, head.as_mut()?, next.as_mut()?)?;
    }

    for (position, &node) in post.iter().enumerate() {
        if let Some(ancestor_node) = parent[node] {
            delta[ancestor_node] -= 1;
        }
        let mut source_column = if ata {
            head.as_ref()?[position]
        } else {
            Some(node)
        };
        while let Some(column) = source_column {
            if column >= transpose.columns {
                return None;
            }
            for entry in transpose.column_pointers[column]..transpose.column_pointers[column + 1] {
                let row_subtree = transpose.row_indices[entry];
                let mut jleaf = 0;
                let lca = cs_leaf(
                    row_subtree,
                    node,
                    &first,
                    &mut maxfirst,
                    &mut prevleaf,
                    &mut ancestor,
                    &mut jleaf,
                );
                if jleaf >= 1 {
                    delta[node] += 1;
                }
                if jleaf == 2 {
                    delta[lca?] -= 1;
                }
            }
            source_column = if ata { next.as_ref()?[column] } else { None };
        }
        if let Some(ancestor_node) = parent[node] {
            ancestor[node] = ancestor_node;
        }
    }
    for node in 0..node_count {
        if let Some(ancestor_node) = parent[node] {
            delta[ancestor_node] += delta[node];
        }
    }
    delta
        .into_iter()
        .map(|count| usize::try_from(count).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::cs_counts;
    use crate::imod::raptor::suitesparse::{Cs, cs_etree::cs_etree, cs_post::cs_post};

    #[test]
    fn counts_match_the_cholesky_column_pattern_for_a_symmetric_matrix() {
        let matrix = Cs {
            nzmax: 5,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 1, 3, 5],
            row_indices: vec![0, 0, 1, 1, 2],
            values: vec![1.0; 5],
            nz: -1,
        };
        let parent = cs_etree(&matrix, false).unwrap();
        let post = cs_post(&parent).unwrap();
        assert_eq!(parent, [Some(1), Some(2), None]);
        assert_eq!(post, [0, 1, 2]);
        assert_eq!(
            cs_counts(&matrix, &parent, &post, false),
            Some(vec![2, 2, 1])
        );
    }

    #[test]
    fn ata_counts_accept_a_rectangular_matrix() {
        let matrix = Cs {
            nzmax: 4,
            rows: 3,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 1, 2],
            values: vec![1.0; 4],
            nz: -1,
        };
        let parent = cs_etree(&matrix, true).unwrap();
        let post = cs_post(&parent).unwrap();
        assert_eq!(cs_counts(&matrix, &parent, &post, true), Some(vec![2, 1]));
    }
}
