//! Translation of `IMOD/raptor/suitesparse/cs_reach.c`.

use super::{Cs, cs_dfs::cs_dfs};

/// C `cs_reach`: finds nodes reachable from one CSC column through a graph.
///
/// Returns the first occupied index in `xi`; the reachable nodes occupy
/// `xi[top..graph.columns]`, exactly as in CSparse.  The original temporarily
/// marks `graph` through signed column-pointer values and restores it before
/// returning.  The owned Rust representation instead uses a local mark array,
/// yielding the same output without mutating graph storage.
pub fn cs_reach(
    graph: &Cs,
    right_hand_side: &Cs,
    column: usize,
    xi: &mut [usize],
    inverse_permutation: Option<&[isize]>,
) -> Option<usize> {
    let node_count = graph.columns;
    if !graph.is_csc()
        || !right_hand_side.is_csc()
        || column >= right_hand_side.columns
        || xi.len() < 2 * node_count
        || graph.column_pointers.len() < node_count + 1
        || right_hand_side.column_pointers.len() < right_hand_side.columns + 1
        || inverse_permutation.is_some_and(|permutation| permutation.len() < node_count)
    {
        return None;
    }
    let rhs_entries = right_hand_side.column_pointers[right_hand_side.columns];
    if rhs_entries > right_hand_side.row_indices.len()
        || right_hand_side
            .column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return None;
    }

    let mut top = node_count;
    let mut marked = vec![false; node_count];
    let mut stack_offsets = vec![0; node_count];
    for entry in
        right_hand_side.column_pointers[column]..right_hand_side.column_pointers[column + 1]
    {
        let node = right_hand_side.row_indices[entry];
        if node >= node_count {
            return None;
        }
        if !marked[node] {
            top = cs_dfs(
                node,
                graph,
                top,
                xi,
                &mut stack_offsets,
                inverse_permutation,
                &mut marked,
            )?;
        }
    }
    Some(top)
}

#[cfg(test)]
mod tests {
    use super::cs_reach;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn reach_uses_rhs_column_seeds_and_preserves_finishing_order() {
        let graph = Cs {
            nzmax: 4,
            rows: 4,
            columns: 4,
            column_pointers: vec![0, 2, 3, 4, 4],
            row_indices: vec![1, 2, 2, 3],
            values: vec![1.0; 4],
            nz: -1,
        };
        let rhs = Cs {
            nzmax: 2,
            rows: 4,
            columns: 1,
            column_pointers: vec![0, 2],
            row_indices: vec![0, 2],
            values: vec![1.0; 2],
            nz: -1,
        };
        let mut xi = vec![usize::MAX; 8];
        assert_eq!(cs_reach(&graph, &rhs, 0, &mut xi, None), Some(0));
        assert_eq!(&xi[..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn reach_honors_a_negative_inverse_permutation_entry() {
        let graph = Cs {
            nzmax: 2,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 1, 2, 2],
            row_indices: vec![1, 2],
            values: vec![1.0; 2],
            nz: -1,
        };
        let rhs = Cs {
            nzmax: 1,
            rows: 3,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![1.0],
            nz: -1,
        };
        let mut xi = vec![0; 6];
        assert_eq!(
            cs_reach(&graph, &rhs, 0, &mut xi, Some(&[-1, 1, 2])),
            Some(2)
        );
        assert_eq!(&xi[2..3], &[0]);
    }
}
