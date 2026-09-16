//! Translation of `IMOD/raptor/suitesparse/cs_post.c`.

/// C `cs_post`: returns the postorder of a forest encoded by parent links.
///
/// A `None` parent denotes a root, corresponding to `-1` in CSparse's integer
/// representation.  The result has the same deterministic child traversal
/// order as the original linked-list workspace implementation.
pub fn cs_post(parent: &[Option<usize>]) -> Option<Vec<usize>> {
    let node_count = parent.len();
    if parent
        .iter()
        .flatten()
        .any(|&ancestor| ancestor >= node_count)
    {
        return None;
    }

    let mut head = vec![None; node_count];
    let mut next = vec![None; node_count];
    for node in (0..node_count).rev() {
        if let Some(ancestor) = parent[node] {
            next[node] = head[ancestor];
            head[ancestor] = Some(node);
        }
    }

    let mut post = Vec::with_capacity(node_count);
    let mut stack = Vec::new();
    for root in 0..node_count {
        if parent[root].is_some() {
            continue;
        }
        stack.push(root);
        while let Some(&node) = stack.last() {
            if let Some(child) = head[node] {
                head[node] = next[child];
                stack.push(child);
            } else {
                stack.pop();
                post.push(node);
            }
        }
    }
    Some(post)
}

#[cfg(test)]
mod tests {
    use super::cs_post;

    #[test]
    fn postorder_matches_csparse_linked_child_traversal() {
        // Children are visited in ascending node order because cs_post builds
        // each linked list by iterating nodes in reverse order.
        let parent = [Some(3), Some(3), Some(4), None, None, Some(4)];
        assert_eq!(cs_post(&parent), Some(vec![0, 1, 3, 2, 5, 4]));
    }

    #[test]
    fn postorder_rejects_parent_outside_the_forest() {
        assert_eq!(cs_post(&[Some(1), Some(2)]), None);
    }
}
