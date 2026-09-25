//! Translation of `IMOD/flib/subrs/hvem/b3dxor.f`.

/// Original `b3dxor` (`b3dxor.f:10`).
///
/// B3DXOR returns the exclusive or of its two arguments
pub fn b3dxor(a: bool, b: bool) -> bool {
    (a && !b) || (b && !a)
}
