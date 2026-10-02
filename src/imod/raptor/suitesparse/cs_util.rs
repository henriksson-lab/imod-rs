//! Translation of `IMOD/raptor/suitesparse/cs_util.c` (the parts RAPTOR
//! reaches).  `cs_spfree`, `cs_sfree` and `cs_nfree` are `Drop`.

use super::cs::{Cs, Csn, cs_csc, cs_triplet};
use super::cs_malloc::{cs_malloc, cs_realloc};

/// `cs_spalloc(m, n, nzmax, values, triplet)`: allocate a sparse matrix
/// (triplet form or compressed-column form).
pub fn cs_spalloc(m: i32, n: i32, nzmax: i32, values: i32, triplet: i32) -> Cs {
    let nzmax = if nzmax > 1 { nzmax } else { 1 };
    Cs {
        m,
        n,
        nzmax,
        nz: if triplet != 0 { 0 } else { -1 },
        p: cs_malloc(if triplet != 0 { nzmax } else { n + 1 }),
        i: cs_malloc(nzmax),
        x: if values != 0 {
            Some(cs_malloc(nzmax))
        } else {
            None
        },
    }
}

/// `cs_sprealloc(A, nzmax)`: change the max # of entries sparse matrix.
pub fn cs_sprealloc(a: &mut Cs, mut nzmax: i32) -> i32 {
    let mut oki = 0;
    let mut okj = 1;
    let mut okx = 1;
    if nzmax <= 0 {
        nzmax = if cs_csc(a) { a.p[a.n as usize] } else { a.nz };
    }
    cs_realloc(&mut a.i, nzmax, &mut oki);
    if cs_triplet(a) {
        cs_realloc(&mut a.p, nzmax, &mut okj);
    }
    if let Some(x) = a.x.as_mut() {
        cs_realloc(x, nzmax, &mut okx);
    }
    let ok = (oki != 0 && okj != 0 && okx != 0) as i32;
    if ok != 0 {
        a.nzmax = nzmax;
    }
    ok
}

/// `cs_done(C, w, x, ok)`: free workspace and return a sparse matrix result.
pub fn cs_done(c: Cs, ok: i32) -> Option<Cs> {
    if ok != 0 { Some(c) } else { None }
}

/// `cs_idone(p, C, w, ok)`: free workspace and return int array result.
pub fn cs_idone(p: Vec<i32>, ok: i32) -> Option<Vec<i32>> {
    if ok != 0 { Some(p) } else { None }
}

/// `cs_ndone(N, C, w, x, ok)`: free workspace and return a numeric
/// factorization.
pub fn cs_ndone(n: Csn, ok: i32) -> Option<Csn> {
    if ok != 0 { Some(n) } else { None }
}
