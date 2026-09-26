//! Translation of `IMOD/flib/subrs/hvem/int_iwrite.f`.

/// Original `int_iwrite` (`int_iwrite.f:4`).
///
/// INT_IWRITE writes the integer NUM left-justified into STRING and reports
/// the number of characters in NCHAR.  `string` is the `character*(*)`
/// argument: it is first blanked over its whole length (`string = ' '`),
/// then the non-blank characters of `write(temp,'(i12)') num` are stored
/// from its start.  `i12` of an `integer*4` always fits (at most 11
/// characters).
pub fn int_iwrite(string: &mut [u8], num: i32, nchar: &mut i32) {
    string.fill(b' ');
    let temp = format!("{:>12}", num);
    *nchar = 0;
    for &byte in temp.as_bytes() {
        if byte != b' ' {
            *nchar += 1;
            string[(*nchar - 1) as usize] = byte;
        }
    }
}
