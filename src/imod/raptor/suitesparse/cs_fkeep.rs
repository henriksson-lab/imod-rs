//! Translation of `IMOD/raptor/suitesparse/cs_fkeep.c`.

use super::cs::{Cs, cs_csc};
use super::cs_util::cs_sprealloc;

/// `cs_fkeep(A, fkeep, other)`: drop entries for which fkeep(A(i,j)) is
/// false; return nz if OK, else -1.  The `other` argument is captured by
/// the `fkeep` closure.
pub fn cs_fkeep(a: &mut Cs, fkeep: &dyn Fn(i32, i32, f64) -> i32) -> i32 {
    let mut nz = 0i32;
    if !cs_csc(a) {
        return -1;
    }
    let n = a.n;
    for j in 0..n as usize {
        let mut p = a.p[j]; // get current location of col j
        a.p[j] = nz; // record new location of col j
        while p < a.p[j + 1] {
            let pu = p as usize;
            let aij = match a.x.as_ref() {
                Some(ax) => ax[pu],
                None => 1.0,
            };
            if fkeep(a.i[pu], j as i32, aij) != 0 {
                if let Some(ax) = a.x.as_mut() {
                    ax[nz as usize] = ax[pu]; // keep A(i,j)
                }
                a.i[nz as usize] = a.i[pu];
                nz += 1;
            }
            p += 1;
        }
    }
    a.p[n as usize] = nz; // finalize A
    cs_sprealloc(a, 0); // remove extra space from A
    nz
}
