//! Translation of `IMOD/flib/subrs/hvem/indmap.f`.

/// Original `indmap` (`indmap.f:4`).
///
/// INDMAP returns IX if IX is between 1 and NX, but when IX is wrapped
/// below 1 or past NX, it adjusts it by NX to be between 1 and NX
pub fn indmap(ix: i32, nx: i32) -> i32 {
    let mut indmap = ix;
    if ix < 1 {
        indmap = ix + nx;
    }
    if ix > nx {
        indmap = ix - nx;
    }
    indmap
}
