//! Translation of `IMOD/flib/subrs/linetrack/b3dputs.c`.

use std::io::Write;

/// `b3dputs`: emit the supplied Fortran string bytes followed by a newline.
///
/// Rust callers provide a length-delimited slice, which is the native
/// representation of the source ABI's `(char *, fortStrLen_t)` pair.
pub fn b3dputs(string: &[u8]) {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(string);
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::b3dputs;

    #[test]
    fn accepts_non_utf8_fortran_bytes() {
        b3dputs(&[b'a', 0xff]);
    }
}
