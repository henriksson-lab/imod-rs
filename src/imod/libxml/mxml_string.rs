//! Translation of `IMOD/libxml/mxml-string.c`.
//!
//! `config.h` defines `HAVE_SNPRINTF`, `HAVE_STRDUP` and `HAVE_VSNPRINTF` on
//! every non-Windows build, so `_mxml_snprintf`, `_mxml_strdup` and
//! `_mxml_vsnprintf` are preprocessed out of this compilation unit and are not
//! part of the vendored library.  Only `_mxml_strdupf` and `_mxml_vstrdupf`
//! are compiled, and both are translated below.
//!
//! Stable Rust cannot *define* a C-variadic function, so `_mxml_strdupf` takes
//! its single variable argument explicitly.  That covers every call in the
//! vendored library: `_mxml_strdupf("![CDATA[%s]]", data)` in `mxml-node.c`
//! and `mxml-set.c`, and the `va_list` that `mxmlSetTextf` forwards.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use core::ffi::{c_char, c_int, c_void};

/// Matches C `__va_list_tag`, the element type of `va_list` in the
/// x86-64 System V ABI.  `_mxml_vstrdupf` needs it to express `va_copy`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaListTag {
    pub gp_offset: u32,
    pub fp_offset: u32,
    pub overflow_arg_area: *mut c_void,
    pub reg_save_area: *mut c_void,
}

unsafe extern "C" {
    fn vsnprintf(s: *mut c_char, maxlen: usize, format: *const c_char, arg: *mut c_void) -> c_int;
}

/// Matches C `_mxml_strdupf` (`mxml-string.c:89`).
///
/// The C function is `_mxml_strdupf(const char *format, ...)`; it starts a
/// `va_list` and hands it to `_mxml_vstrdupf`.  Here the one variable argument
/// is passed through to `snprintf` directly, which is the same formatting for
/// the single-argument formats the library uses.
pub unsafe fn _mxml_strdupf(format: *const c_char, arg: *mut c_void) -> *mut c_char {
    let mut temp: [c_char; 256] = [0; 256];
    let bytes: c_int = libc::snprintf(
        temp.as_mut_ptr(),
        core::mem::size_of::<[c_char; 256]>(),
        format,
        arg,
    );

    if (bytes as usize) < core::mem::size_of::<[c_char; 256]>() {
        return libc::strdup(temp.as_ptr());
    }

    let buffer = libc::calloc(1, (bytes + 1) as usize) as *mut c_char;
    if !buffer.is_null() {
        libc::snprintf(buffer, (bytes + 1) as usize, format, arg);
    }
    buffer
}

/// Matches C `_mxml_vstrdupf` (`mxml-string.c:425`).
pub unsafe fn _mxml_vstrdupf(format: *const c_char, ap: *mut c_void) -> *mut c_char {
    let bytes: c_int;
    let buffer: *mut c_char;
    let mut temp: [c_char; 256] = [0; 256];

    /* va_copy(apcopy, ap) -- va_list is an array of one __va_list_tag. */
    let mut apcopy: [VaListTag; 1] = [*(ap as *const VaListTag)];

    bytes = vsnprintf(
        temp.as_mut_ptr(),
        core::mem::size_of::<[c_char; 256]>(),
        format,
        apcopy.as_mut_ptr() as *mut c_void,
    );

    if (bytes as usize) < core::mem::size_of::<[c_char; 256]>() {
        return libc::strdup(temp.as_ptr());
    }

    buffer = libc::calloc(1, (bytes + 1) as usize) as *mut c_char;
    if !buffer.is_null() {
        vsnprintf(buffer, (bytes + 1) as usize, format, ap);
    }
    buffer
}
