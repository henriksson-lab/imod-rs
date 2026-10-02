//! Translation of `IMOD/raptor/suitesparse/cs_post.c`.

use super::cs_malloc::cs_malloc;
use super::cs_tdfs::cs_tdfs;
use super::cs_util::cs_idone;

/// `cs_post(parent, n)`: post order a forest.
pub fn cs_post(parent: Option<&[i32]>, n: i32) -> Option<Vec<i32>> {
    let parent = parent?;
    let mut k = 0i32;
    let mut post: Vec<i32> = cs_malloc(n);
    let nu = n as usize;
    // w = [head | next | stack], each n long
    let mut head: Vec<i32> = cs_malloc(n);
    let mut next: Vec<i32> = cs_malloc(n);
    let mut stack: Vec<i32> = cs_malloc(n);
    for j in 0..nu {
        head[j] = -1; // empty linked lists
    }
    let mut j = n - 1;
    while j >= 0 {
        // traverse nodes in reverse order
        let ju = j as usize;
        if parent[ju] != -1 {
            next[ju] = head[parent[ju] as usize]; // add j to list of its parent
            head[parent[ju] as usize] = j;
        }
        j -= 1;
    }
    for j in 0..n {
        if parent[j as usize] != -1 {
            continue; // skip j if it is not a root
        }
        k = cs_tdfs(j, k, &mut head, &next, &mut post, &mut stack);
    }
    cs_idone(post, 1)
}
