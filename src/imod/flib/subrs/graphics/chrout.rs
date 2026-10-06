//! Translation of `IMOD/flib/subrs/graphics/chrout.f`.

use std::io::Write as _;

/// Original `chrout` (`chrout.f:4`): outputs the character with the given
/// code, with no line end.
pub fn chrout(int: i32) {
    let _ = std::io::stdout().write_all(&[int as u8]);
}
