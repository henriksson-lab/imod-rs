//! Translation of `IMOD/flib/subrs/hvem/xfcopy3d.f`.

/// Original `xfcopy3d` (`xfcopy3d.f:1`).
///
/// Copies the 3D transform `a1`, `d1` into `a2`, `d2`.  The `real*4 a(3,3)`
/// arguments are column major: `a(i,j)` is `a[(i - 1) + (j - 1) * 3]`.
pub fn xfcopy3d(a1: &[f32], d1: &[f32], a2: &mut [f32], d2: &mut [f32]) {
    for i in 1..=3usize {
        d2[i - 1] = d1[i - 1];
        for j in 1..=3usize {
            a2[(i - 1) + (j - 1) * 3] = a1[(i - 1) + (j - 1) * 3];
        }
    }
}
