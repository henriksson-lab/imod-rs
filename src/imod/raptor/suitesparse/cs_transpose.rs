//! Translation of `IMOD/raptor/suitesparse/cs_transpose.c`.

use super::cs::{Cs, cs_csc};
use super::cs_cumsum::cs_cumsum;
use super::cs_malloc::cs_calloc;
use super::cs_util::{cs_done, cs_spalloc};

/// `cs_transpose(A, values)`: C = A'.
pub fn cs_transpose(a: &Cs, values: i32) -> Option<Cs> {
    if !cs_csc(a) {
        return None;
    }
    let m = a.m;
    let n = a.n;
    let ap = &a.p;
    let ai = &a.i;
    let ax = a.x.as_ref();
    let mut c = cs_spalloc(
        n,
        m,
        ap[n as usize],
        (values != 0 && ax.is_some()) as i32,
        0,
    );
    let mut w: Vec<i32> = cs_calloc(m);
    for p in 0..ap[n as usize] as usize {
        w[ai[p] as usize] += 1; // row counts
    }
    cs_cumsum(&mut c.p, &mut w, m); // row pointers
    for j in 0..n {
        for p in ap[j as usize]..ap[j as usize + 1] {
            let p = p as usize;
            let q = w[ai[p] as usize] as usize; // place A(i,j) as entry C(j,i)
            w[ai[p] as usize] += 1;
            c.i[q] = j;
            if let Some(cx) = c.x.as_mut() {
                cx[q] = ax.unwrap()[p];
            }
        }
    }
    cs_done(c, 1)
}
