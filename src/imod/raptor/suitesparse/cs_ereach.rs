//! Translation of `IMOD/raptor/suitesparse/cs_ereach.c`.

use super::cs::{Cs, cs_csc, cs_mark, cs_marked};

/// `cs_ereach(A, k, parent, s, w)`: find nonzero pattern of Cholesky
/// L(k,1:k-1) using etree and triu(A(:,k)).
pub fn cs_ereach(a: &Cs, k: i32, parent: &[i32], s: &mut [i32], w: &mut [i32]) -> i32 {
    if !cs_csc(a) {
        return -1;
    }
    let n = a.n;
    let mut top = n;
    let ap = &a.p;
    let ai = &a.i;
    cs_mark(w, k); // mark node k as visited
    for p in ap[k as usize]..ap[k as usize + 1] {
        let mut i = ai[p as usize]; // A(i,k) is nonzero
        if i > k {
            continue; // only use upper triangular part of A
        }
        let mut len = 0i32;
        while !cs_marked(w, i) {
            // traverse up etree
            s[len as usize] = i; // L(k,i) is nonzero
            len += 1;
            cs_mark(w, i); // mark i as visited
            i = parent[i as usize];
        }
        while len > 0 {
            top -= 1;
            len -= 1;
            s[top as usize] = s[len as usize]; // push path onto stack
        }
    }
    for p in top..n {
        cs_mark(w, s[p as usize]); // unmark all nodes
    }
    cs_mark(w, k); // unmark node k
    top // s [top..n-1] contains pattern of L(k,:)
}
