//! Translation of `IMOD/raptor/suitesparse/cs_ipvec.c`.

/// `cs_ipvec(p, b, x, n)`: x(p) = b, for dense vectors x and b; p=NULL
/// denotes identity.
pub fn cs_ipvec(p: Option<&[i32]>, b: &[f64], x: &mut [f64], n: i32) -> i32 {
    for k in 0..n as usize {
        x[match p {
            Some(p) => p[k] as usize,
            None => k,
        }] = b[k];
    }
    1
}
