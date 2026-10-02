//! Translation of `IMOD/raptor/suitesparse/cs_cumsum.c`.

/// `cs_cumsum(p, c, n)`: p [0..n] = cumulative sum of c [0..n-1], and then
/// copy p [0..n-1] into c.
pub fn cs_cumsum(p: &mut [i32], c: &mut [i32], n: i32) -> f64 {
    let mut nz = 0i32;
    let mut nz2 = 0.0f64;
    for i in 0..n as usize {
        p[i] = nz;
        nz += c[i];
        nz2 += c[i] as f64; // also in double to avoid int overflow
        c[i] = p[i];
    }
    p[n as usize] = nz;
    nz2
}
