use crate::imod::libcfshr::b3dutil::{c2f_string, f2c_string};
use crate::imod::libcfshr::parse_params::*;
unsafe extern "C" {
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn exit(__status: ::core::ffi::c_int) -> !;
}
pub type fortStrLen_t = ::core::ffi::c_int;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
#[inline]
unsafe extern "C" fn __bswap_16(mut __bsx: __uint16_t) -> __uint16_t {
    return (__bsx as ::core::ffi::c_int >> 8 as ::core::ffi::c_int & 0xff as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_int & 0xff as ::core::ffi::c_int) << 8 as ::core::ffi::c_int)
        as __uint16_t;
}
#[inline]
unsafe extern "C" fn __bswap_32(mut __bsx: __uint32_t) -> __uint32_t {
    return (__bsx & 0xff000000 as __uint32_t) >> 24 as ::core::ffi::c_int
        | (__bsx & 0xff0000 as __uint32_t) >> 8 as ::core::ffi::c_int
        | (__bsx & 0xff00 as __uint32_t) << 8 as ::core::ffi::c_int
        | (__bsx & 0xff as __uint32_t) << 24 as ::core::ffi::c_int;
}
#[inline]
unsafe extern "C" fn __bswap_64(mut __bsx: __uint64_t) -> __uint64_t {
    return ((__bsx as ::core::ffi::c_ulonglong & 0xff00000000000000 as ::core::ffi::c_ulonglong)
        >> 56 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000000000 as ::core::ffi::c_ulonglong)
            >> 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000000000 as ::core::ffi::c_ulonglong)
            >> 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00000000 as ::core::ffi::c_ulonglong)
            >> 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000 as ::core::ffi::c_ulonglong)
            << 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000 as ::core::ffi::c_ulonglong)
            << 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00 as ::core::ffi::c_ulonglong)
            << 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff as ::core::ffi::c_ulonglong)
            << 56 as ::core::ffi::c_int) as __uint64_t;
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
pub unsafe extern "C" fn pipinitialize_(
    mut numOptions: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return pip_initialize(*numOptions);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipexitonerrorfw_(
    mut useStdErr: *mut ::core::ffi::c_int,
    mut prefix: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(prefix, stringSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_exit_on_error(*useStdErr, cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipexit_(mut val: *mut ::core::ffi::c_int) {
    exit(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipallowcommadefaults_(mut val: *mut ::core::ffi::c_int) {
    pip_allow_comma_defaults(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetmanpageoutput_(mut val: *mut ::core::ffi::c_int) {
    pip_set_manpage_output(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipenableentryoutput_(mut val: *mut ::core::ffi::c_int) {
    pip_enable_entry_output(*val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetspecialflags_(
    mut inCase: *mut ::core::ffi::c_int,
    mut inDone: *mut ::core::ffi::c_int,
    mut inStd: *mut ::core::ffi::c_int,
    mut inLines: *mut ::core::ffi::c_int,
    mut inAbbrevs: *mut ::core::ffi::c_int,
) {
    pip_set_special_flags(*inCase, *inDone, *inStd, *inLines, *inAbbrevs);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadstdinifset_() -> ::core::ffi::c_int {
    return pip_read_stdin_if_set();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadoptionfile_(
    mut progName: *mut ::core::ffi::c_char,
    mut helpLevel: *mut ::core::ffi::c_int,
    mut localDir: *mut ::core::ffi::c_int,
    mut stringSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(progName, stringSize as fortStrLen_t);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_read_option_file(cStr, *helpLevel, *localDir);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipreadprogdefaults_(
    mut option: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
) {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        pip_set_error(
            b"Memory error in pipreadprogdefaults_\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return;
    }
    pip_read_prog_defaults(cStr);
    free(cStr as *mut ::core::ffi::c_void);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipaddoption_(
    mut optionString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(optionString, stringSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_add_option(cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnextarg_(
    mut argString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(argString, stringSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_next_arg(cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnumberofargs_(
    mut numOptArgs: *mut ::core::ffi::c_int,
    mut numNonOptArgs: *mut ::core::ffi::c_int,
) {
    pip_number_of_args(numOptArgs, numNonOptArgs);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetnonoptionarg_(
    mut argNo: *mut ::core::ffi::c_int,
    mut arg: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut argPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    err = pip_get_non_option_arg(*argNo - 1 as ::core::ffi::c_int, &raw mut argPtr);
    if err == 0 && c2f_string(argPtr, arg, stringSize as ::core::ffi::c_int) != 0 {
        pip_set_error(
            b"Non-option argument too long for character variable\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        err = -(1 as ::core::ffi::c_int);
    }
    if !argPtr.is_null() {
        free(argPtr as *mut ::core::ffi::c_void);
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetstring_(
    mut option: *mut ::core::ffi::c_char,
    mut string: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_string(cStr, &raw mut strPtr);
    if err == 0 && c2f_string(strPtr, string, stringSize as ::core::ffi::c_int) != 0 {
        pip_set_error(
            b"In pip_get_string, string is too long for character variable\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        err = -(1 as ::core::ffi::c_int);
    }
    if !strPtr.is_null() {
        free(strPtr as *mut ::core::ffi::c_void);
    }
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetinteger_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_int,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_integer(cStr, val);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetfloat_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_float,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_float(cStr, val);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgettwointegers_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_int,
    mut val2: *mut ::core::ffi::c_int,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_two_integers(cStr, val1, val2);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgettwofloats_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_two_floats(cStr, val1, val2);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetthreeintegers_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_int,
    mut val2: *mut ::core::ffi::c_int,
    mut val3: *mut ::core::ffi::c_int,
    mut optionSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize as fortStrLen_t);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_three_integers(cStr, val1, val2, val3);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetthreefloats_(
    mut option: *mut ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
    mut val3: *mut ::core::ffi::c_float,
    mut optionSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize as fortStrLen_t);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_three_floats(cStr, val1, val2, val3);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetboolean_(
    mut option: *mut ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_int,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_boolean(cStr, val);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetintegerarray_(
    mut option: *mut ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_int,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: *mut ::core::ffi::c_int,
    mut optionSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize as fortStrLen_t);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_integer_array(cStr, array, numToGet, *arraySize);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgetfloatarray_(
    mut option: *mut ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_float,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: *mut ::core::ffi::c_int,
    mut optionSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize as fortStrLen_t);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_get_float_array(cStr, array, numToGet, *arraySize);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipprinthelp_(
    mut string: *mut ::core::ffi::c_char,
    mut useStdErr: *mut ::core::ffi::c_int,
    mut inputFiles: *mut ::core::ffi::c_int,
    mut outputFiles: *mut ::core::ffi::c_int,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(string, stringSize);
    if cStr.is_null() {
        pip_set_error(b"Memory error in pipgethelp_\0" as *const u8 as *const ::core::ffi::c_char);
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_print_help(cStr, *useStdErr, *inputFiles, *outputFiles);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipprintentries_() {
    pip_print_entries();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipgeterror_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    err = pip_get_error(&raw mut strPtr);
    if strPtr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = c2f_string(strPtr, errString, stringSize as ::core::ffi::c_int);
    free(strPtr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipseterror_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(errString, stringSize);
    if cStr.is_null() {
        pip_set_error(b"Memory error in pipseterror_\0" as *const u8 as *const ::core::ffi::c_char);
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_set_error(cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetusagestring_(
    mut errString: *mut ::core::ffi::c_char,
    mut stringSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(errString, stringSize);
    if cStr.is_null() {
        pip_set_error(
            b"Memory error in pipsetusagestring_\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_set_usage_string(cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipsetlinkedoption_(
    mut option: *mut ::core::ffi::c_char,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        pip_set_error(
            b"Memory error in pipsetlinkedoption_\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_set_linked_option(cStr);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn piplinkedindex_(
    mut option: *mut ::core::ffi::c_char,
    mut index: *mut ::core::ffi::c_int,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_linked_index(cStr, index);
    *index += 1;
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipnumberofentries_(
    mut option: *mut ::core::ffi::c_char,
    mut numEntries: *mut ::core::ffi::c_int,
    mut optionSize: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    cStr = pipf2cstr(option, optionSize);
    if cStr.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    err = pip_number_of_entries(cStr, numEntries);
    free(cStr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pipdone_() {
    pip_done();
}
unsafe extern "C" fn pipf2cstr(
    mut str: *mut ::core::ffi::c_char,
    mut strSize: fortStrLen_t,
) -> *mut ::core::ffi::c_char {
    let mut newStr: *mut ::core::ffi::c_char = f2c_string(str, strSize as ::core::ffi::c_int);
    if newStr.is_null() {
        pip_set_error(
            b"Memory error converting string from Fortran to C\0" as *const u8
                as *const ::core::ffi::c_char,
        );
    }
    return newStr;
}
