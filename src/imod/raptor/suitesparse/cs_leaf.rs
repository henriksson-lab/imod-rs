//! Translation of `IMOD/raptor/suitesparse/cs_leaf.c`.

/// `cs_leaf(i, j, first, maxfirst, prevleaf, ancestor, &jleaf)`: consider
/// A(i,j), node j in ith row subtree and return lca(jprev,j).
pub fn cs_leaf(
    i: i32,
    j: i32,
    first: &[i32],
    maxfirst: &mut [i32],
    prevleaf: &mut [i32],
    ancestor: &mut [i32],
    jleaf: &mut i32,
) -> i32 {
    *jleaf = 0;
    let iu = i as usize;
    if i <= j || first[j as usize] <= maxfirst[iu] {
        return -1; // j not a leaf
    }
    maxfirst[iu] = first[j as usize]; // update max first[j] seen so far
    let jprev = prevleaf[iu]; // jprev = previous leaf of ith subtree
    prevleaf[iu] = j;
    *jleaf = if jprev == -1 { 1 } else { 2 }; // j is first or subsequent leaf
    if *jleaf == 1 {
        return i; // if 1st leaf, q = root of ith subtree
    }
    let mut q = jprev;
    while q != ancestor[q as usize] {
        q = ancestor[q as usize];
    }
    let mut s = jprev;
    while s != q {
        let sparent = ancestor[s as usize]; // path compression
        ancestor[s as usize] = q;
        s = sparent;
    }
    q // q = least common ancester (jprev,j)
}
