use crate::imod::libcfshr::b3dutil::{c2f_string, fortran_string};
use crate::imod::libcfshr::parse_params::*;
pub type fortStrLen_t = i32;

/// `pipf2cstr` (`pip_fwrap.c:435`).  The Fortran `character*(*)` argument is
/// the byte slice itself, so the source's hidden `strSize` is the slice
/// length; the C string `f2cString` allocates is the owned `String` that
/// comes back, and every wrapper's `free` is its drop.  The one branch that
/// disappears is the `malloc` failure the source reports through
/// `PipSetError`, which has no Rust counterpart.
fn pipf2cstr(string: &[u8]) -> String {
    fortran_string(string)
}
/// `pipgetnonoptionarg` (`pip_fwrap.c:190`).  `arg` is the caller's fixed-width
/// `character*(*)` field and its length is the source's `stringSize`:
/// `c2fString` fills it and blank-pads the remainder, so it stays a
/// `&mut [u8]` of that width rather than becoming text.
pub fn pipgetnonoptionarg_(argNo: i32, arg: &mut [u8]) -> i32 {
    let mut argVec: Vec<u8> = Vec::new();
    let mut err: i32 = 0;
    match pip_get_non_option_arg(argNo - 1 as i32) {
        Ok(value) => argVec = value,
        Err(()) => err = -(1 as i32),
    }
    if err == 0 && c2f_string(&argVec, arg).is_err() {
        pip_set_error(b"Non-option argument too long for character variable");
        err = -(1 as i32);
    }
    return err;
}
/// `pipgetstring` (`pip_fwrap.c:206`).  `option` is the Fortran field whose
/// trailing blanks `pipf2cstr` trims; `string` is the fixed-width destination
/// `c2fString` blank-pads, and both slice lengths are the source's
/// `optionSize` and `stringSize`.
pub fn pipgetstring_(option: &[u8], string: &mut [u8]) -> i32 {
    let cStr = pipf2cstr(option);
    let mut strVec: Vec<u8> = Vec::new();
    let mut err = pip_get_string(cStr.as_bytes(), &mut strVec);
    if err == 0 && c2f_string(&strVec, string).is_err() {
        pip_set_error(b"In PipGetString, string is too long for character variable");
        err = -(1 as i32);
    }
    err
}
/// `pipgetinteger` (`pip_fwrap.c:227`).  `option`'s slice length is the
/// source's `optionSize`.
pub fn pipgetinteger_(option: &[u8], val: &mut i32) -> i32 {
    let cStr = pipf2cstr(option);
    pip_get_integer(cStr.as_bytes(), val)
}
/// `pipnumberofentries` (`pip_fwrap.c:418`).  `option`'s slice length is the
/// source's `optionSize`.
pub fn pipnumberofentries_(option: &[u8], numEntries: &mut i32) -> i32 {
    let cStr = pipf2cstr(option);
    pip_number_of_entries(cStr.as_bytes(), numEntries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pip_fortran_converter_trims_fortran_padding() {
        assert_eq!(pipf2cstr(b"option   "), "option");
        assert_eq!(pipf2cstr(b"   "), "");
        assert_eq!(pipf2cstr(b""), "");
    }
}
