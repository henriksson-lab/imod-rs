//! Translation of `IMOD/flib/subrs/hvem/xfmult3d.f`.

/// Original `xfmult3d` (`xfmult3d.f:1`).
///
/// Multiplies the 3D transform `a1`, `d1` by `a2`, `d2` (applied second) to
/// give `a3`, `d3`.  The `real*4 a(3,3)` arguments are column major:
/// `a(i,j)` is `a[(i - 1) + (j - 1) * 3]`.  The source's implicit `real`
/// accumulations stay in single precision, in the source's order.
pub fn xfmult3d(a1: &[f32], d1: &[f32], a2: &[f32], d2: &[f32], a3: &mut [f32], d3: &mut [f32]) {
    for i in 1..=3usize {
        d3[i - 1] = d2[i - 1];
        for j in 1..=3usize {
            d3[i - 1] += a2[(i - 1) + (j - 1) * 3] * d1[j - 1];
            a3[(i - 1) + (j - 1) * 3] = 0.;
            for k in 1..=3usize {
                a3[(i - 1) + (j - 1) * 3] += a2[(i - 1) + (k - 1) * 3] * a1[(k - 1) + (j - 1) * 3];
            }
        }
    }
}
