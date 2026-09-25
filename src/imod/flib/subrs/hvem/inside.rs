//! Translation of `IMOD/flib/subrs/hvem/inside.f`.

use crate::imod::libcfshr::insidecontour::inside_contour;

/// Original `inside` (`inside.f:11`).
///
/// Returns .true. if point [x], [y] is on or inside the polygon defined
/// by vertices in [xvert], [yvert], or .false. if point is outside.
/// [np] is the number of vertices in polygon
pub fn inside(xvert: &[f32], yvert: &[f32], np: i32, x: f32, y: f32) -> bool {
    inside_contour(xvert, yvert, np, x, y) != 0
}
