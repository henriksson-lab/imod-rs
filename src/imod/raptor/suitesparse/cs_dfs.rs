//! Translation of `IMOD/raptor/suitesparse/cs_dfs.c`.

use super::Cs;

/// C `cs_dfs`: depth-first search of a CSC graph.
///
/// The C implementation temporarily negates graph column pointers to mark
/// vertices.  `Cs` deliberately owns unsigned offsets, so the equivalent
/// transient mark array is explicit here and the graph remains unchanged.
pub fn cs_dfs(
    start: usize,
    graph: &Cs,
    mut top: usize,
    xi: &mut [usize],
    stack_offsets: &mut [usize],
    inverse_permutation: Option<&[isize]>,
    marked: &mut [bool],
) -> Option<usize> {
    let node_count = graph.columns;
    if !graph.is_csc()
        || start >= node_count
        || top > node_count
        || xi.len() < node_count
        || stack_offsets.len() < node_count
        || marked.len() < node_count
        || graph.column_pointers.len() < node_count + 1
        || inverse_permutation.is_some_and(|permutation| permutation.len() < node_count)
    {
        return None;
    }
    let entries = graph.column_pointers[node_count];
    if entries > graph.row_indices.len()
        || graph
            .column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return None;
    }

    let mut head = 0;
    xi[0] = start;
    loop {
        let node = xi[head];
        if node >= node_count {
            return None;
        }
        let mapped_node =
            inverse_permutation.map_or(node as isize, |permutation| permutation[node]);
        if !marked[node] {
            marked[node] = true;
            stack_offsets[head] = if mapped_node < 0 {
                0
            } else {
                let mapped_node = mapped_node as usize;
                if mapped_node >= node_count {
                    return None;
                }
                graph.column_pointers[mapped_node]
            };
        }

        let upper = if mapped_node < 0 {
            0
        } else {
            let mapped_node = mapped_node as usize;
            if mapped_node >= node_count {
                return None;
            }
            graph.column_pointers[mapped_node + 1]
        };
        if upper > entries || stack_offsets[head] > upper {
            return None;
        }
        let mut done = true;
        for entry in stack_offsets[head]..upper {
            let neighbor = graph.row_indices[entry];
            if neighbor >= node_count {
                return None;
            }
            if marked[neighbor] {
                continue;
            }
            stack_offsets[head] = entry;
            head += 1;
            if head >= node_count {
                return None;
            }
            xi[head] = neighbor;
            done = false;
            break;
        }
        if done {
            if top == 0 {
                return None;
            }
            top -= 1;
            xi[top] = node;
            if head == 0 {
                break;
            }
            head -= 1;
        }
    }
    Some(top)
}

#[cfg(test)]
mod tests {
    use super::cs_dfs;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn dfs_returns_nodes_in_csparse_finishing_order() {
        let graph = Cs {
            nzmax: 4,
            rows: 4,
            columns: 4,
            column_pointers: vec![0, 2, 3, 4, 4],
            row_indices: vec![1, 2, 2, 3],
            values: vec![1.0; 4],
            nz: -1,
        };
        let mut xi = vec![0; 4];
        let mut offsets = vec![0; 4];
        let mut marked = vec![false; 4];
        assert_eq!(
            cs_dfs(0, &graph, 4, &mut xi, &mut offsets, None, &mut marked),
            Some(0)
        );
        assert_eq!(xi, [0, 1, 2, 3]);
        assert_eq!(marked, [true, true, true, true]);
    }
}
