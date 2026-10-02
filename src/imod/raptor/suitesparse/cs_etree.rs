//! Translation of `IMOD/raptor/suitesparse/cs_etree.c`.

use super::cs::{Cs, cs_csc};
use super::cs_malloc::cs_malloc;
use super::cs_util::cs_idone;

/// `cs_etree(A, ata)`: compute the etree of A (using triu(A), or A'A without
/// forming A'A).
pub fn cs_etree(a: &Cs, ata: i32) -> Option<Vec<i32>> {
    if !cs_csc(a) {
        return None;
    }
    let m = a.m;
    let n = a.n;
    let ap = &a.p;
    let ai = &a.i;
    let mut parent: Vec<i32> = cs_malloc(n);
    // w = [ancestor (n) | prev (m if ata)]
    let mut ancestor: Vec<i32> = cs_malloc(n);
    let mut prev: Vec<i32> = cs_malloc(if ata != 0 { m } else { 0 });
    if ata != 0 {
        for i in 0..m as usize {
            prev[i] = -1;
        }
    }
    for k in 0..n {
        let ku = k as usize;
        parent[ku] = -1; // node k has no parent yet
        ancestor[ku] = -1; // nor does k have an ancestor
        for p in ap[ku]..ap[ku + 1] {
            let p = p as usize;
            let mut i = if ata != 0 {
                prev[ai[p] as usize]
            } else {
                ai[p]
            };
            while i != -1 && i < k {
                // traverse from i to k
                let inext = ancestor[i as usize]; // inext = ancestor of i
                ancestor[i as usize] = k; // path compression
                if inext == -1 {
                    parent[i as usize] = k; // no anc., parent is k
                }
                i = inext;
            }
            if ata != 0 {
                prev[ai[p] as usize] = k;
            }
        }
    }
    cs_idone(parent, 1)
}
