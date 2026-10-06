//! Translation of `IMOD/flib/subrs/hvem/mypause.f`.

use crate::imod::flib::subrs::compat::gfortran_rt::read_line_stdin;
use std::io::Write as _;

/// Original `mypause` (`mypause.f:1`): prints the message and waits for a
/// line of input.
pub fn mypause(message: &[u8]) {
    // `101 format(' PAUSE: ',a,'   (return to continue)',$)`
    let mut out = std::io::stdout();
    let _ = out.write_all(b" PAUSE: ");
    let _ = out.write_all(message);
    let _ = out.write_all(b"   (return to continue)");
    // `read(5,'(a)')ichar` into a `character`
    let _ = read_line_stdin(1);
}
