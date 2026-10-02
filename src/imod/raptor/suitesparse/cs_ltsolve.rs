//! Translation of `IMOD/raptor/suitesparse/cs_ltsolve.c`.

use super::cs::{Cs, cs_csc};

/// `cs_ltsolve(L, x)`: solve L'x=b where x and b are dense.  x=b on input,
/// solution on output.
pub fn cs_ltsolve(l: &Cs, x: &mut [f64]) -> i32 {
    if !cs_csc(l) {
        return 0;
    }
    let n = l.n;
    let lp = &l.p;
    let li = &l.i;
    let lx = l.x.as_ref().expect("cs_ltsolve values");
    let mut j = n - 1;
    while j >= 0 {
        let ju = j as usize;
        for p in lp[ju] + 1..lp[ju + 1] {
            let p = p as usize;
            x[ju] -= lx[p] * x[li[p] as usize];
        }
        x[ju] /= lx[lp[ju] as usize];
        j -= 1;
    }
    1
}
