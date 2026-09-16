//! Translation of `IMOD/raptor/suitesparse/cs_leaf.c`.

/// C `cs_leaf`: considers `A(i,j)` in row-subtree `i` and returns the least
/// common ancestor of its previous and current leaves.
///
/// `None` is C's `-1`: either `j` is not a leaf or the input workspaces do
/// not describe the requested nodes.  On success, `jleaf` is set to one for
/// the first leaf or two for a subsequent leaf.
pub fn cs_leaf(
    row_subtree: usize,
    node: usize,
    first: &[Option<usize>],
    maxfirst: &mut [Option<usize>],
    prevleaf: &mut [Option<usize>],
    ancestor: &mut [usize],
    jleaf: &mut usize,
) -> Option<usize> {
    if row_subtree >= first.len()
        || node >= first.len()
        || row_subtree >= maxfirst.len()
        || row_subtree >= prevleaf.len()
        || row_subtree >= ancestor.len()
    {
        return None;
    }
    *jleaf = 0;
    let node_first = first[node]?;
    if row_subtree <= node || maxfirst[row_subtree].is_some_and(|seen| node_first <= seen) {
        return None;
    }
    maxfirst[row_subtree] = Some(node_first);
    let previous = prevleaf[row_subtree];
    prevleaf[row_subtree] = Some(node);
    let Some(previous) = previous else {
        *jleaf = 1;
        return Some(row_subtree);
    };
    *jleaf = 2;

    let mut lca = previous;
    for _ in 0..ancestor.len() {
        let parent = *ancestor.get(lca)?;
        if lca == parent {
            break;
        }
        lca = parent;
    }
    if ancestor.get(lca).copied()? != lca {
        return None;
    }
    let mut current = previous;
    for _ in 0..ancestor.len() {
        if current == lca {
            return Some(lca);
        }
        let parent = *ancestor.get(current)?;
        ancestor[current] = lca;
        current = parent;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::cs_leaf;

    #[test]
    fn leaf_tracks_first_and_subsequent_leaves_with_path_compression() {
        let first = [Some(0), Some(1), Some(2), Some(3)];
        let mut maxfirst = vec![None; 4];
        let mut prevleaf = vec![None; 4];
        let mut ancestor = vec![2, 2, 3, 3];
        let mut jleaf = 99;
        assert_eq!(
            cs_leaf(
                2,
                3,
                &first,
                &mut maxfirst,
                &mut prevleaf,
                &mut ancestor,
                &mut jleaf,
            ),
            None
        );
        assert_eq!(jleaf, 0);

        let first = [Some(1), Some(2), Some(3)];
        let mut maxfirst = vec![None; 3];
        let mut prevleaf = vec![None; 3];
        let mut ancestor = vec![1, 2, 2];
        assert_eq!(
            cs_leaf(
                2,
                0,
                &first,
                &mut maxfirst,
                &mut prevleaf,
                &mut ancestor,
                &mut jleaf,
            ),
            Some(2)
        );
        assert_eq!(jleaf, 1);
        assert_eq!(
            cs_leaf(
                2,
                1,
                &first,
                &mut maxfirst,
                &mut prevleaf,
                &mut ancestor,
                &mut jleaf,
            ),
            Some(2)
        );
        assert_eq!(jleaf, 2);
        assert_eq!(ancestor[0], 2);
    }
}
