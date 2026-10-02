//! Translation of `IMOD/raptor/suitesparse/cs_cholsol.c`.

use super::cs::{Cs, cs_csc};
use super::cs_chol::cs_chol;
use super::cs_ipvec::cs_ipvec;
use super::cs_lsolve::cs_lsolve;
use super::cs_ltsolve::cs_ltsolve;
use super::cs_malloc::cs_malloc;
use super::cs_pvec::cs_pvec;
use super::cs_schol::cs_schol;

/// `cs_cholsol(order, A, b)`: x=A\b where A is symmetric positive definite;
/// b overwritten with solution.
pub fn cs_cholsol(order: i32, a: &Cs, b: &mut [f64]) -> i32 {
    if !cs_csc(a) {
        return 0;
    }
    let n = a.n;
    let s = cs_schol(order, a); // ordering and symbolic analysis
    let nn = match s.as_ref() {
        Some(s) => cs_chol(a, s), // numeric Cholesky factorization
        None => None,
    };
    let mut x: Vec<f64> = cs_malloc(n); // get workspace
    let ok = (s.is_some() && nn.is_some()) as i32;
    if ok != 0 {
        let s = s.as_ref().unwrap();
        let l = nn.as_ref().unwrap().l.as_ref().unwrap();
        cs_ipvec(s.pinv.as_deref(), b, &mut x, n); // x = P*b
        cs_lsolve(l, &mut x); // x = L\x
        cs_ltsolve(l, &mut x); // x = L'\x
        cs_pvec(s.pinv.as_deref(), &x, b, n); // b = P'*x
    }
    ok
}
