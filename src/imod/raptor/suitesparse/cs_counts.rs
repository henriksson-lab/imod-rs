//! Translation of `IMOD/raptor/suitesparse/cs_counts.c`.
//!
//! The workspace `w` is one array of `s` ints as in C; its sub-arrays
//! (`ancestor = w`, `maxfirst = w+n`, `prevleaf = w+2n`, `first = w+3n`,
//! and for `ata` `head = w+4n`, `next = w+5n+1`) are split off it.

use super::cs::{Cs, cs_csc, cs_min};
use super::cs_leaf::cs_leaf;
use super::cs_malloc::cs_malloc;
use super::cs_transpose::cs_transpose;
use super::cs_util::cs_idone;

/// `init_ata(AT, post, w, &head, &next)` (static): `w` is the whole
/// workspace; `head` and `next` are returned as offsets into it.
fn init_ata(at: &Cs, post: &[i32], w: &mut [i32], head: &mut usize, next: &mut usize) {
    let m = at.n;
    let n = at.m;
    let atp = &at.p;
    let ati = &at.i;
    *head = 4 * n as usize;
    *next = 5 * n as usize + 1;
    for k in 0..n {
        w[post[k as usize] as usize] = k; // invert post
    }
    for i in 0..m as usize {
        let mut k = n;
        for p in atp[i]..atp[i + 1] {
            k = cs_min(k, w[ati[p as usize] as usize]);
        }
        w[*next + i] = w[*head + k as usize]; // place row i in linked list k
        w[*head + k as usize] = i as i32;
    }
}

/// `cs_counts(A, parent, post, ata)`: column counts of LL'=A or LL'=A'A,
/// given parent & post ordering.
pub fn cs_counts(
    a: &Cs,
    parent: Option<&[i32]>,
    post: Option<&[i32]>,
    ata: i32,
) -> Option<Vec<i32>> {
    if !cs_csc(a) {
        return None;
    }
    let (parent, post) = match (parent, post) {
        (Some(parent), Some(post)) => (parent, post),
        _ => return None,
    };
    let mut jleaf = 0i32;
    let m = a.m;
    let n = a.n;
    let nu = n as usize;
    let s = 4 * n + if ata != 0 { n + m + 1 } else { 0 };
    let mut colcount: Vec<i32> = cs_malloc(n); // delta = colcount
    let mut w: Vec<i32> = cs_malloc(s);
    let at = cs_transpose(a, 0)?; // AT = A'
    let (mut head, mut next) = (0usize, 0usize);
    for k in 0..s as usize {
        w[k] = -1; // clear workspace w [0..s-1]
    }
    // first = w+3n
    for k in 0..nu {
        let mut j = post[k];
        colcount[j as usize] = if w[3 * nu + j as usize] == -1 { 1 } else { 0 }; // delta[j]=1 if j is a leaf
        while j != -1 && w[3 * nu + j as usize] == -1 {
            w[3 * nu + j as usize] = k as i32;
            j = parent[j as usize];
        }
    }
    let atp = &at.p;
    let ati = &at.i;
    if ata != 0 {
        init_ata(&at, post, &mut w, &mut head, &mut next);
    }
    for i in 0..nu {
        w[i] = i as i32; // each node in its own set (ancestor = w)
    }
    for k in 0..nu {
        let j = post[k]; // j is the kth node in postordered etree
        if parent[j as usize] != -1 {
            colcount[parent[j as usize] as usize] -= 1; // j is not a root
        }
        // J=j for LL'=A case: HEAD(k,j), NEXT(J)
        let mut jj = if ata != 0 { w[head + k] } else { j };
        while jj != -1 {
            for p in atp[jj as usize]..atp[jj as usize + 1] {
                let i = ati[p as usize];
                let (ancestor, rest) = w.split_at_mut(nu);
                let (maxfirst, rest) = rest.split_at_mut(nu);
                let (prevleaf, rest) = rest.split_at_mut(nu);
                let first = &rest[..nu];
                let q = cs_leaf(i, j, first, maxfirst, prevleaf, ancestor, &mut jleaf);
                if jleaf >= 1 {
                    colcount[j as usize] += 1; // A(i,j) is in skeleton
                }
                if jleaf == 2 {
                    colcount[q as usize] -= 1; // account for overlap in q
                }
            }
            jj = if ata != 0 { w[next + jj as usize] } else { -1 };
        }
        if parent[j as usize] != -1 {
            w[j as usize] = parent[j as usize];
        }
    }
    for j in 0..nu {
        // sum up delta's of each child
        if parent[j] != -1 {
            colcount[parent[j] as usize] += colcount[j];
        }
    }
    cs_idone(colcount, 1)
}
