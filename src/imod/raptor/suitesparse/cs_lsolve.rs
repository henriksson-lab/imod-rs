//! Translation of `IMOD/raptor/suitesparse/cs_lsolve.c`.

use super::cs::{Cs, cs_csc};

/// `cs_lsolve(L, x)`: solve Lx=b where x and b are dense.  x=b on input,
/// solution on output.
pub fn cs_lsolve(l: &Cs, x: &mut [f64]) -> i32 {
    if !cs_csc(l) {
        return 0;
    }
    let n = l.n;
    let lp = &l.p;
    let li = &l.i;
    let lx = l.x.as_ref().expect("cs_lsolve values");
    for j in 0..n as usize {
        x[j] /= lx[lp[j] as usize];
        for p in lp[j] + 1..lp[j + 1] {
            let p = p as usize;
            x[li[p] as usize] -= lx[p] * x[j];
        }
    }
    1
}
