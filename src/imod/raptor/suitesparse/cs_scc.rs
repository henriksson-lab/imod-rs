//! Translation of `IMOD/raptor/suitesparse/cs_scc.c`.

use super::{Cs, Csd, cs_dfs::cs_dfs, cs_transpose::cs_transpose};

/// C `cs_scc`: finds the strongly connected components of a square CSC graph.
///
/// CSparse marks `matrix.column_pointers` in place during each DFS and restores
/// them before returning.  The owned Rust representation keeps its offsets
/// unsigned, so the equivalent marks live in temporary boolean vectors and
/// `matrix` is never changed.
pub fn cs_scc(matrix: &Cs) -> Option<Csd> {
    if !matrix.is_csc()
        || matrix.rows != matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
        || matrix
            .column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return None;
    }
    let node_count = matrix.columns;
    let entries = matrix.column_pointers[node_count];
    if entries > matrix.row_indices.len()
        || matrix.row_indices[..entries]
            .iter()
            .any(|&row| row >= node_count)
    {
        return None;
    }
    let transpose = cs_transpose(matrix, false)?;
    let mut finish_order = vec![0; node_count];
    let mut stack_offsets = vec![0; node_count];
    let mut marked = vec![false; node_count];
    let mut top = node_count;
    for node in 0..node_count {
        if !marked[node] {
            top = cs_dfs(
                node,
                matrix,
                top,
                &mut finish_order,
                &mut stack_offsets,
                None,
                &mut marked,
            )?;
        }
    }

    let mut permutation = vec![0; node_count];
    let mut blocks = vec![0; node_count + 6];
    marked.fill(false);
    top = node_count;
    let mut remaining_blocks = node_count;
    for index in 0..node_count {
        let node = finish_order[index];
        if marked[node] {
            continue;
        }
        blocks[remaining_blocks] = top;
        remaining_blocks -= 1;
        top = cs_dfs(
            node,
            &transpose,
            top,
            &mut permutation,
            &mut stack_offsets,
            None,
            &mut marked,
        )?;
    }
    blocks[remaining_blocks] = 0;
    let block_count = node_count - remaining_blocks;
    blocks.copy_within(remaining_blocks..=node_count, 0);
    blocks.truncate(block_count + 1);

    let mut block_of_node = vec![0; node_count];
    for block in 0..block_count {
        for index in blocks[block]..blocks[block + 1] {
            block_of_node[permutation[index]] = block;
        }
    }
    let mut next = blocks[..block_count].to_vec();
    for node in 0..node_count {
        let block = block_of_node[node];
        permutation[next[block]] = node;
        next[block] += 1;
    }

    Some(Csd {
        p: permutation,
        q: vec![0; node_count],
        r: blocks,
        s: vec![0; node_count + 6],
        nb: block_count,
        rr: [0; 5],
        cc: [0; 5],
    })
}

#[cfg(test)]
mod tests {
    use super::cs_scc;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn scc_groups_components_and_sorts_nodes_within_each_block() {
        // 0 <-> 1 -> 2 <-> 3.  CSparse's finish-time traversal visits the
        // downstream component first, then sorts node order within each block.
        let graph = Cs {
            nzmax: 5,
            rows: 4,
            columns: 4,
            column_pointers: vec![0, 1, 3, 4, 5],
            row_indices: vec![1, 0, 2, 3, 2],
            values: vec![1.0; 5],
            nz: -1,
        };
        let result = cs_scc(&graph).unwrap();
        assert_eq!(result.nb, 2);
        assert_eq!(result.r, [0, 2, 4]);
        assert_eq!(result.p, [2, 3, 0, 1]);
        assert_eq!(graph.column_pointers, [0, 1, 3, 4, 5]);
    }

    #[test]
    fn scc_rejects_non_square_or_malformed_graphs() {
        let non_square = Cs {
            nzmax: 0,
            rows: 1,
            columns: 2,
            column_pointers: vec![0, 0, 0],
            row_indices: vec![],
            values: vec![],
            nz: -1,
        };
        assert_eq!(cs_scc(&non_square), None);
    }

    #[test]
    fn scc_accepts_a_structural_matrix_without_values() {
        let graph = Cs {
            nzmax: 2,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![1, 0],
            values: vec![],
            nz: -1,
        };
        let result = cs_scc(&graph).unwrap();
        assert_eq!(result.nb, 1);
        assert_eq!(result.p, [0, 1]);
    }
}
