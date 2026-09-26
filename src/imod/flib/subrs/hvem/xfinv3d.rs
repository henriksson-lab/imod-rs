//! Translation of `IMOD/flib/subrs/hvem/xfinv3d.f`.

use crate::imod::libcfshr::linearxforms::inv_matrix;

/// Original `xfinv3d` (`xfinv3d.f:1`).
///
/// Inverts the 3D transform `a`, `d` into `ainv`, `dinv`.  The matrices are
/// the column-major `real*4 (3,3)` arrays; `inv_matrix` is the libcfshr C
/// routine the Fortran call resolves to.
pub fn xfinv3d(a: &[f32], d: &[f32], ainv: &mut [f32], dinv: &mut [f32]) {
    let mut matrix = [0.0_f32; 9];
    matrix.copy_from_slice(&a[..9]);
    let mut inverse = [0.0_f32; 9];
    inv_matrix(&matrix, &mut inverse);
    ainv[..9].copy_from_slice(&inverse);
    for i in 1..=3usize {
        dinv[i - 1] = 0.;
        for j in 1..=3usize {
            dinv[i - 1] -= ainv[(i - 1) + (j - 1) * 3] * d[j - 1];
        }
    }
}
