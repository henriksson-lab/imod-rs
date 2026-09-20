use crate::imod::libcfshr::b3dutil::{c2f_string, fortran_string};
use crate::imod::libcfshr::parse_params::*;
pub type fortStrLen_t = i32;

/// `pipf2cstr` (`pip_fwrap.c:435`).  `f2cString` allocated a temporary C
/// string which every wrapper freed after its PIP call; an owned `String`
/// expresses the same temporary lifetime.  A null/negative input is the
/// Rust-visible counterpart of C allocation/conversion failure.
unsafe fn pipf2cstr(
    string: *const ::core::ffi::c_char,
    string_size: fortStrLen_t,
) -> Option<String> {
    if string.is_null() || string_size < 0 {
        pip_set_error(b"Memory error converting string from Fortran to C");
        return None;
    }
    Some(unsafe { fortran_string(string, string_size) })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetnonoptionarg_(
    mut argNo: *mut i32,
    mut arg: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let mut argVec: Vec<u8> = Vec::new();
    let mut err: i32 = 0;
    err = pip_get_non_option_arg(*argNo - 1 as i32, &mut argVec);
    argVec.push(0);
    let argPtr = argVec.as_ptr().cast::<::core::ffi::c_char>();
    if err == 0 && c2f_string(argPtr, arg, stringSize as i32) != 0 {
        pip_set_error(b"Non-option argument too long for character variable");
        err = -(1 as i32);
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetstring_(
    mut option: *mut ::core::ffi::c_char,
    mut string: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let option = fortran_string(option, optionSize);
    let mut strVec: Vec<u8> = Vec::new();
    let mut err = pip_get_string(option.as_bytes(), &mut strVec);
    strVec.push(0);
    strPtr = strVec.as_ptr().cast::<::core::ffi::c_char>().cast_mut();
    if err == 0 && c2f_string(strPtr, string, stringSize as i32) != 0 {
        pip_set_error(b"In pip_get_string, string is too long for character variable");
        err = -(1 as i32);
    }
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetinteger_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut i32,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_integer(option.as_bytes(), &mut *val)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnumberofentries_(
    mut option: *mut ::core::ffi::c_char,
    mut numEntries: *mut i32,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_number_of_entries(option.as_bytes(), &mut *numEntries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pip_fortran_converter_trims_fortran_padding() {
        let bytes = b"option   ";
        let converted = unsafe { pipf2cstr(bytes.as_ptr().cast(), bytes.len() as i32) };
        assert_eq!(converted.as_deref(), Some("option"));
        assert!(unsafe { pipf2cstr(core::ptr::null(), 4) }.is_none());
    }
}
