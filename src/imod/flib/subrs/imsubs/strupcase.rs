//! Translation of `IMOD/flib/subrs/imsubs/strupcase.f`.

/// Original: `STRUPCASE` (`strupcase.f:11`).
///
/// Converts the lower case characters in `atbute` to upper case and returns
/// the result in `at2`.  Both are fixed-length Fortran `CHARACTER*(*)`
/// arguments, so they are byte buffers whose lengths are `LEN(AT2)` and
/// `LEN(ATBUTE)`: `at2 = ''` blank-fills the whole destination, and only
/// `min(LEN(ATBUTE), LEN(AT2))` characters are then copied.
pub fn strupcase(at2: &mut [u8], atbute: &[u8]) {
    at2.fill(b' ');
    for lcv in 1..=atbute.len().min(at2.len()) {
        if atbute[lcv - 1] >= b'a' && atbute[lcv - 1] <= b'z' {
            at2[lcv - 1] = atbute[lcv - 1] - 32;
        } else {
            at2[lcv - 1] = atbute[lcv - 1];
        }
    }
}
