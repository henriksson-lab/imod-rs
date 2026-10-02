//! Translation of `IMOD/raptor/suitesparse/cs_pvec.c`.

/// `cs_pvec(p, b, x, n)`: x = b(p), for dense vectors x and b; p=NULL
/// denotes identity.
pub fn cs_pvec(p: Option<&[i32]>, b: &[f64], x: &mut [f64], n: i32) -> i32 {
    for k in 0..n as usize {
        x[k] = b[match p {
            Some(p) => p[k] as usize,
            None => k,
        }];
    }
    1
}
