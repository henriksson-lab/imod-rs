//! Translation of `IMOD/flib/subrs/graphics/flnam.f`.

use crate::imod::flib::subrs::compat::gfortran_rt::{len_trim, read_line_stdin};
use std::io::Write as _;

/// Original `flnam` (`flnam.f:4`): reads a file name into the
/// `character*len` variable (returned blank-padded to `len`), with an
/// optional prompt, adding `ext` when the name has no `.` (unless `ext` is
/// `0`).
pub fn flnam(len: usize, ifmes: i32, ext: &str) -> Vec<u8> {
    if ifmes != 0 {
        let mut out = std::io::stdout();
        if ext == "0" {
            let _ = out.write_all(b" File name: ");
        } else {
            let _ = write!(out, " File name (.{ext} assumed): ");
        }
    }
    let mut name = read_line_stdin(len);
    if ext == "0" || name.contains(&b'.') {
        return name;
    }
    // `name=trim(name)//'.'//trim(ext)`
    let mut joined = name[..len_trim(&name)].to_vec();
    joined.push(b'.');
    joined.extend_from_slice(ext.trim_end().as_bytes());
    joined.truncate(len);
    joined.resize(len, b' ');
    name = joined;
    name
}
