//! Translation of `IMOD/raptor/suitesparse/cs_pinv.c`.

use super::cs_malloc::cs_malloc;

/// `cs_pinv(p, n)`: pinv = p', or p = pinv'.  p = NULL denotes identity.
pub fn cs_pinv(p: Option<&[i32]>, n: i32) -> Option<Vec<i32>> {
    let p = p?;
    let mut pinv: Vec<i32> = cs_malloc(n);
    for k in 0..n {
        pinv[p[k as usize] as usize] = k; // invert the permutation
    }
    Some(pinv)
}
