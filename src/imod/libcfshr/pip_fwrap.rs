use crate::imod::libcfshr::b3dutil::{c2f_string, fortran_string};
use crate::imod::libcfshr::parse_params::*;
unsafe extern "C" {
    fn exit(__status: i32) -> !;
}
pub type fortStrLen_t = i32;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
#[inline]
unsafe extern "C" fn __bswap_16(mut __bsx: __uint16_t) -> __uint16_t {
    return (__bsx as i32 >> 8 as i32 & 0xff as i32 | (__bsx as i32 & 0xff as i32) << 8 as i32)
        as __uint16_t;
}
#[inline]
unsafe extern "C" fn __bswap_32(mut __bsx: __uint32_t) -> __uint32_t {
    return (__bsx & 0xff000000 as __uint32_t) >> 24 as i32
        | (__bsx & 0xff0000 as __uint32_t) >> 8 as i32
        | (__bsx & 0xff00 as __uint32_t) << 8 as i32
        | (__bsx & 0xff as __uint32_t) << 24 as i32;
}
#[inline]
unsafe extern "C" fn __bswap_64(mut __bsx: __uint64_t) -> __uint64_t {
    return ((__bsx as ::core::ffi::c_ulonglong & 0xff00000000000000 as ::core::ffi::c_ulonglong)
        >> 56 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000000000 as ::core::ffi::c_ulonglong)
            >> 40 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000000000 as ::core::ffi::c_ulonglong)
            >> 24 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00000000 as ::core::ffi::c_ulonglong)
            >> 8 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000 as ::core::ffi::c_ulonglong) << 8 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000 as ::core::ffi::c_ulonglong) << 24 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00 as ::core::ffi::c_ulonglong) << 40 as i32
        | (__bsx as ::core::ffi::c_ulonglong & 0xff as ::core::ffi::c_ulonglong) << 56 as i32)
        as __uint64_t;
}
#[inline]
unsafe extern "C" fn __uint16_identity(mut __x: __uint16_t) -> __uint16_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint32_identity(mut __x: __uint32_t) -> __uint32_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint64_identity(mut __x: __uint64_t) -> __uint64_t {
    return __x;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipinitialize_(mut numOptions: *mut i32) -> i32 {
    return pip_initialize(*numOptions);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipexitonerrorfw_(
    mut useStdErr: *mut i32,
    mut prefix: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let prefix = fortran_string(prefix, stringSize);
    pip_exit_on_error(*useStdErr, prefix.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipexit_(mut val: *mut i32) {
    exit(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipallowcommadefaults_(mut val: *mut i32) {
    pip_allow_comma_defaults(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetmanpageoutput_(mut val: *mut i32) {
    pip_set_manpage_output(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipenableentryoutput_(mut val: *mut i32) {
    pip_enable_entry_output(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetspecialflags_(
    mut inCase: *mut i32,
    mut inDone: *mut i32,
    mut inStd: *mut i32,
    mut inLines: *mut i32,
    mut inAbbrevs: *mut i32,
) {
    pip_set_special_flags(*inCase, *inDone, *inStd, *inLines, *inAbbrevs);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadstdinifset_() -> i32 {
    return pip_read_stdin_if_set();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadoptionfile_(
    mut progName: *mut ::core::ffi::c_char,
    mut helpLevel: *mut i32,
    mut localDir: *mut i32,
    mut stringSize: i32,
) -> i32 {
    let prog_name = fortran_string(progName, stringSize);
    pip_read_option_file(prog_name.as_bytes(), *helpLevel, *localDir)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadprogdefaults_(
    mut option: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
) {
    let option = fortran_string(option, optionSize);
    pip_read_prog_defaults(option.as_bytes());
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipaddoption_(
    mut optionString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(optionString, stringSize);
    pip_add_option(option.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnextarg_(
    mut argString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let arg = fortran_string(argString, stringSize);
    pip_next_arg(arg.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnumberofargs_(mut numOptArgs: *mut i32, mut numNonOptArgs: *mut i32) {
    pip_number_of_args(&mut *numOptArgs, &mut *numNonOptArgs);
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
pub unsafe extern "C" fn pipgetfloat_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_float,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_float(option.as_bytes(), &mut *val)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgettwointegers_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut i32,
    mut val2: *mut i32,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_two_integers(option.as_bytes(), &mut *val1, &mut *val2)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgettwofloats_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_two_floats(option.as_bytes(), &mut *val1, &mut *val2)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetthreeintegers_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut i32,
    mut val2: *mut i32,
    mut val3: *mut i32,
    mut optionSize: i32,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_three_integers(option.as_bytes(), &mut *val1, &mut *val2, &mut *val3)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetthreefloats_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
    mut val3: *mut ::core::ffi::c_float,
    mut optionSize: i32,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_three_floats(option.as_bytes(), &mut *val1, &mut *val2, &mut *val3)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetboolean_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut i32,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_boolean(option.as_bytes(), &mut *val)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetintegerarray_(
    mut option: *mut ::core::ffi::c_char,
    mut array: *mut i32,
    mut numToGet: *mut i32,
    mut arraySize: *mut i32,
    mut optionSize: i32,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_integer_array(
        option.as_bytes(),
        ::core::slice::from_raw_parts_mut(array, (*arraySize).max(0) as usize),
        &mut *numToGet,
        *arraySize,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetfloatarray_(
    mut option: *mut ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_float,
    mut numToGet: *mut i32,
    mut arraySize: *mut i32,
    mut optionSize: i32,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_get_float_array(
        option.as_bytes(),
        ::core::slice::from_raw_parts_mut(array, (*arraySize).max(0) as usize),
        &mut *numToGet,
        *arraySize,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipprinthelp_(
    mut string: *mut ::core::ffi::c_char,
    mut useStdErr: *mut i32,
    mut inputFiles: *mut i32,
    mut outputFiles: *mut i32,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let string = fortran_string(string, stringSize);
    pip_print_help(string.as_bytes(), *useStdErr, *inputFiles, *outputFiles)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipprintentries_() {
    pip_print_entries();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgeterror_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let mut strVec: Vec<u8> = Vec::new();
    let mut err: i32 = 0;
    /* `PipGetError` always returns a string here -- the source's NULL test can
    only fire on a failed strdup -- so the error code it returns is ignored,
    exactly as `pipgeterror_` ignores it. */
    err = pip_get_error(&mut strVec);
    strVec.push(0);
    let strPtr = strVec.as_ptr().cast::<::core::ffi::c_char>();
    err = c2f_string(strPtr, errString, stringSize as i32);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipseterror_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let error = fortran_string(errString, stringSize);
    pip_set_error(error.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetusagestring_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> i32 {
    let usage = fortran_string(errString, stringSize);
    pip_set_usage_string(usage.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetlinkedoption_(
    mut option: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    pip_set_linked_option(option.as_bytes())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn piplinkedindex_(
    mut option: *mut ::core::ffi::c_char,
    mut index: *mut i32,
    mut optionSize: fortStrLen_t,
) -> i32 {
    let option = fortran_string(option, optionSize);
    let err = pip_linked_index(option.as_bytes(), &mut *index);
    *index += 1;
    err
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipdone_() {
    pip_done();
}
