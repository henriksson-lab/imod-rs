//! Translation of `IMOD/raptor/suitesparse/cs_tdfs.c`.

/// C `cs_tdfs`: depth-first postorder of a child-list tree rooted at `root`.
pub fn cs_tdfs(
    root: usize,
    start: usize,
    heads: &mut [Option<usize>],
    next: &[Option<usize>],
    postorder: &mut Vec<usize>,
) -> Option<usize> {
    if root >= heads.len() || start > postorder.len() {
        return None;
    }
    let mut stack = vec![root];
    let mut index = start;
    while let Some(node) = stack.last().copied() {
        if node >= heads.len() {
            return None;
        }
        if let Some(child) = heads[node] {
            heads[node] = *next.get(child)?;
            stack.push(child);
        } else {
            stack.pop();
            if index == postorder.len() {
                postorder.push(node);
            } else {
                *postorder.get_mut(index)? = node;
            }
            index += 1;
        }
    }
    Some(index)
}

#[cfg(test)]
mod tests {
    use super::cs_tdfs;
    #[test]
    fn tdfs_consumes_child_lists_in_source_order() {
        let mut heads = [Some(1), Some(2), None];
        let next = [None, None, None];
        let mut post = vec![];
        assert_eq!(cs_tdfs(0, 0, &mut heads, &next, &mut post), Some(3));
        assert_eq!(post, [2, 1, 0]);
    }
}
