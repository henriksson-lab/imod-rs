//! Translation of `IMOD/raptor/suitesparse/cs_schol.c`.

use super::cs::{Cs, Css, cs_csc};
use super::cs_amd::cs_amd;
use super::cs_counts::cs_counts;
use super::cs_cumsum::cs_cumsum;
use super::cs_etree::cs_etree;
use super::cs_malloc::cs_malloc;
use super::cs_pinv::cs_pinv;
use super::cs_post::cs_post;
use super::cs_symperm::cs_symperm;

/// `cs_schol(order, A)`: ordering and symbolic analysis for a Cholesky
/// factorization.
pub fn cs_schol(order: i32, a: &Cs) -> Option<Css> {
    if !cs_csc(a) {
        return None;
    }
    let n = a.n;
    let mut s = Css::default();
    let p = cs_amd(order, a); // P = amd(A+A'), or natural
    s.pinv = cs_pinv(p.as_deref(), n); // find inverse permutation
    drop(p);
    if order != 0 && s.pinv.is_none() {
        return None;
    }
    let c = cs_symperm(a, s.pinv.as_deref(), 0)?; // C = spones(triu(A(P,P)))
    s.parent = cs_etree(&c, 0); // find etree of C
    let post = cs_post(s.parent.as_deref(), n); // postorder the etree
    let mut cc = cs_counts(&c, s.parent.as_deref(), post.as_deref(), 0)?; // find column counts of chol(C)
    drop(post);
    drop(c);
    let mut cp: Vec<i32> = cs_malloc(n + 1); // allocate result S->cp
    let lnz = cs_cumsum(&mut cp, &mut cc, n); // find column pointers for L
    s.cp = Some(cp);
    s.lnz = lnz;
    s.unz = lnz;
    drop(cc);
    if s.lnz >= 0.0 { Some(s) } else { None }
}
