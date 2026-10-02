//! Translation of `IMOD/raptor/suitesparse/cs_multiply.c`.

use super::cs::{Cs, cs_csc};
use super::cs_malloc::{cs_calloc, cs_malloc};
use super::cs_scatter::cs_scatter;
use super::cs_util::{cs_done, cs_spalloc, cs_sprealloc};

/// `cs_multiply(A, B)`: C = A*B.
pub fn cs_multiply(a: &Cs, b: &Cs) -> Option<Cs> {
    let mut nz = 0i32;
    if !cs_csc(a) || !cs_csc(b) {
        return None;
    }
    if a.n != b.m {
        return None;
    }
    let m = a.m;
    let anz = a.p[a.n as usize];
    let n = b.n;
    let bp = &b.p;
    let bi = &b.i;
    let bx = b.x.as_ref();
    let bnz = bp[n as usize];
    let mut w: Vec<i32> = cs_calloc(m);
    let values = a.x.is_some() && bx.is_some();
    let mut x: Option<Vec<f64>> = if values { Some(cs_malloc(m)) } else { None };
    let mut c = cs_spalloc(m, n, anz + bnz, values as i32, 0);
    for j in 0..n {
        if nz + m > c.nzmax && {
            let nzmax = 2 * c.nzmax + m;
            cs_sprealloc(&mut c, nzmax) == 0
        } {
            return cs_done(c, 0); // out of memory
        }
        c.p[j as usize] = nz; // column j of C starts here
        for p in bp[j as usize]..bp[j as usize + 1] {
            let p = p as usize;
            nz = cs_scatter(
                a,
                bi[p],
                match bx {
                    Some(bx) => bx[p],
                    None => 1.0,
                },
                &mut w,
                x.as_deref_mut(),
                j + 1,
                &mut c.i,
                nz,
            );
        }
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
