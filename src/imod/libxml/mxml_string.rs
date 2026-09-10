#![allow(
    dead_code,
    non_snake_case,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn
)]
//! Mechanical direct baseline of `IMOD/libxml/mxml-string.c`.

unsafe extern "C" {
    fn vsnprintf(
        __s: *mut ::core::ffi::c_char,
        __maxlen: size_t,
        __format: *const ::core::ffi::c_char,
        __arg: ::core::ffi::VaList,
    ) -> ::core::ffi::c_int;
    fn calloc(__nmemb: size_t, __size: size_t) -> *mut ::core::ffi::c_void;
    fn strdup(__s: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
}
pub type __builtin_va_list = [__va_list_tag; 1];
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __va_list_tag {
    pub gp_offset: ::core::ffi::c_uint,
    pub fp_offset: ::core::ffi::c_uint,
    pub overflow_arg_area: *mut ::core::ffi::c_void,
    pub reg_save_area: *mut ::core::ffi::c_void,
}
pub type size_t = usize;
pub type va_list = __builtin_va_list;
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _mxml_strdupf(
    mut format: *const ::core::ffi::c_char,
    mut args: ...
) -> *mut ::core::ffi::c_char {
    let mut ap: ::core::ffi::VaListImpl;
    let mut s: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    ap = args.clone();
    s = _mxml_vstrdupf(format, ap.as_va_list());
    return s;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _mxml_vstrdupf(
    mut format: *const ::core::ffi::c_char,
    mut ap: ::core::ffi::VaList,
) -> *mut ::core::ffi::c_char {
    let mut bytes: ::core::ffi::c_int = 0;
    let mut buffer: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut temp: [::core::ffi::c_char; 256] = [0; 256];
    let mut apcopy: ::core::ffi::VaListImpl;
    apcopy = ap.clone();
    bytes = vsnprintf(
        &raw mut temp as *mut ::core::ffi::c_char,
        ::core::mem::size_of::<[::core::ffi::c_char; 256]>() as size_t,
        format,
        apcopy.as_va_list(),
    );
    if (bytes as usize) < ::core::mem::size_of::<[::core::ffi::c_char; 256]>() as usize {
        return strdup(&raw mut temp as *mut ::core::ffi::c_char);
    }
    buffer = calloc(1 as size_t, (bytes + 1 as ::core::ffi::c_int) as size_t)
        as *mut ::core::ffi::c_char;
    if !buffer.is_null() {
        vsnprintf(
            buffer,
            (bytes + 1 as ::core::ffi::c_int) as size_t,
            format,
            ap.as_va_list(),
        );
    }
    return buffer;
}
