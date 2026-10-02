//! Translation of `IMOD/raptor/suitesparse/cs_add.c`.

use super::cs::{Cs, cs_csc};
use super::cs_malloc::{cs_calloc, cs_malloc};
use super::cs_scatter::cs_scatter;
use super::cs_util::{cs_done, cs_spalloc, cs_sprealloc};

/// `cs_add(A, B, alpha, beta)`: C = alpha*A + beta*B.
pub fn cs_add(a: &Cs, b: &Cs, alpha: f64, beta: f64) -> Option<Cs> {
    let mut nz = 0i32;
    if !cs_csc(a) || !cs_csc(b) {
        return None;
    }
    if a.m != b.m || a.n != b.n {
        return None;
    }
    let m = a.m;
    let anz = a.p[a.n as usize];
    let n = b.n;
    let bnz = b.p[n as usize];
    let mut w: Vec<i32> = cs_calloc(m);
    let values = a.x.is_some() && b.x.is_some();
    let mut x: Option<Vec<f64>> = if values { Some(cs_malloc(m)) } else { None };
    let mut c = cs_spalloc(m, n, anz + bnz, values as i32, 0);
    for j in 0..n {
        c.p[j as usize] = nz; // column j of C starts here
        nz = cs_scatter(a, j, alpha, &mut w, x.as_deref_mut(), j + 1, &mut c.i, nz); // alpha*A(:,j)
        nz = cs_scatter(b, j, beta, &mut w, x.as_deref_mut(), j + 1, &mut c.i, nz); // beta*B(:,j)
        if values {
            let x = x.as_ref().unwrap();
            let cx = c.x.as_mut().unwrap();
            for p in c.p[j as usize]..nz {
                cx[p as usize] = x[c.i[p as usize] as usize];
            }
        }
    }
    c.p[n as usize] = nz; // finalize the last column of C
    cs_sprealloc(&mut c, 0); // remove extra space from C
    cs_done(c, 1)
}
