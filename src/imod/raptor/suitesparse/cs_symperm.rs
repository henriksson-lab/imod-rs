//! Translation of `IMOD/raptor/suitesparse/cs_symperm.c`.

use super::cs::{Cs, cs_csc, cs_max, cs_min};
use super::cs_cumsum::cs_cumsum;
use super::cs_malloc::cs_calloc;
use super::cs_util::{cs_done, cs_spalloc};

/// `cs_symperm(A, pinv, values)`: C = A(p,p) where A and C are symmetric the
/// upper part stored; pinv not p.
pub fn cs_symperm(a: &Cs, pinv: Option<&[i32]>, values: i32) -> Option<Cs> {
    if !cs_csc(a) {
        return None;
    }
    let n = a.n;
    let ap = &a.p;
    let ai = &a.i;
    let ax = a.x.as_ref();
    let mut c = cs_spalloc(
        n,
        n,
        ap[n as usize],
        (values != 0 && ax.is_some()) as i32,
        0,
    );
    let mut w: Vec<i32> = cs_calloc(n);
    for j in 0..n {
        // count entries in each column of C
        let j2 = match pinv {
            Some(pinv) => pinv[j as usize],
            None => j,
        }; // column j of A is column j2 of C
        for p in ap[j as usize]..ap[j as usize + 1] {
            let i = ai[p as usize];
            if i > j {
                continue; // skip lower triangular part of A
            }
            let i2 = match pinv {
                Some(pinv) => pinv[i as usize],
                None => i,
            }; // row i of A is row i2 of C
            w[cs_max(i2, j2) as usize] += 1; // column count of C
        }
    }
    cs_cumsum(&mut c.p, &mut w, n); // compute column pointers of C
    for j in 0..n {
        let j2 = match pinv {
            Some(pinv) => pinv[j as usize],
            None => j,
        };
        for p in ap[j as usize]..ap[j as usize + 1] {
            let i = ai[p as usize];
            if i > j {
                continue; // skip lower triangular part of A
            }
            let i2 = match pinv {
                Some(pinv) => pinv[i as usize],
                None => i,
            };
            let q = w[cs_max(i2, j2) as usize] as usize;
            w[cs_max(i2, j2) as usize] += 1;
            c.i[q] = cs_min(i2, j2);
            if let Some(cx) = c.x.as_mut() {
                cx[q] = ax.unwrap()[p as usize];
            }
        }
    }
    cs_done(c, 1)
}
