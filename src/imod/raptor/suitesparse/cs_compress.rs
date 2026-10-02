//! Translation of `IMOD/raptor/suitesparse/cs_compress.c`.

use super::cs::{Cs, cs_triplet};
use super::cs_cumsum::cs_cumsum;
use super::cs_malloc::cs_calloc;
use super::cs_util::{cs_done, cs_spalloc};

/// `cs_compress(T)`: C = compressed-column form of a triplet matrix T.
pub fn cs_compress(t: &Cs) -> Option<Cs> {
    if !cs_triplet(t) {
        return None;
    }
    let m = t.m;
    let n = t.n;
    let ti = &t.i;
    let tj = &t.p;
    let tx = t.x.as_ref();
    let nz = t.nz;
    let mut c = cs_spalloc(m, n, nz, tx.is_some() as i32, 0);
    let mut w: Vec<i32> = cs_calloc(n);
    for k in 0..nz as usize {
        w[tj[k] as usize] += 1; // column counts
    }
    cs_cumsum(&mut c.p, &mut w, n); // column pointers
    for k in 0..nz as usize {
        let p = w[tj[k] as usize] as usize; // A(i,j) is the pth entry in C
        w[tj[k] as usize] += 1;
        c.i[p] = ti[k];
        if let (Some(cx), Some(tx)) = (c.x.as_mut(), tx) {
            cx[p] = tx[k];
        }
    }
    cs_done(c, 1)
}
