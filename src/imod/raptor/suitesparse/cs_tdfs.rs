//! Translation of `IMOD/raptor/suitesparse/cs_tdfs.c`.

/// `cs_tdfs(j, k, head, next, post, stack)`: depth-first search and
/// postorder of a tree rooted at node j.
pub fn cs_tdfs(
    j: i32,
    mut k: i32,
    head: &mut [i32],
    next: &[i32],
    post: &mut [i32],
    stack: &mut [i32],
) -> i32 {
    let mut top = 0i32;
    stack[0] = j; // place j on the stack
    while top >= 0 {
        let p = stack[top as usize]; // p = top of stack
        let i = head[p as usize]; // i = youngest child of p
        if i == -1 {
            top -= 1; // p has no unordered children left
            post[k as usize] = p; // node p is the kth postordered node
            k += 1;
        } else {
            head[p as usize] = next[i as usize]; // remove i from children of p
            top += 1;
            stack[top as usize] = i; // start dfs on child node i
        }
    }
    k
}
