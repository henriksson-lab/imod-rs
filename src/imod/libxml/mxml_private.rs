//! Translation of `IMOD/libxml/mxml-private.c` and `IMOD/libxml/mxml-private.h`.
//!
//! `config.h` defines `HAVE_PTHREAD_H` on every non-Windows build, so the
//! per-thread `pthread_key_t` variant of `_mxml_global()` is the one compiled
//! into the vendored library and the one translated here.  The `WIN32` and
//! no-threads variants are not part of this build.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// Matches C `_mxml_global_t` (`mxml-private.h:30`).
#[repr(C)]
pub struct MxmlGlobal {
    pub error_cb: MxmlErrorCb,
    pub num_entity_cbs: c_int,
    pub entity_cbs: [MxmlEntityCb; 100],
    pub wrap: c_int,
    pub custom_load_cb: MxmlCustomLoadCb,
    pub custom_save_cb: MxmlCustomSaveCb,
}

/// Matches C static `_mxml_key` (`mxml-private.c:106`).
static mut MXML_KEY: libc::pthread_key_t = !0;
/// Matches C static `_mxml_key_once` (`mxml-private.c:107`).
static mut MXML_KEY_ONCE: libc::pthread_once_t = libc::PTHREAD_ONCE_INIT;

/// Matches C `mxml_error` (`mxml-private.c:41`).
///
/// The C function is `mxml_error(const char *format, ...)` and runs the
/// arguments through `vsnprintf` into a 1024-byte buffer.  Stable Rust cannot
/// define a C-variadic function, so callers format their arguments into a
/// 1024-byte buffer of their own and pass the result here; running it back
/// through `snprintf("%s")` keeps the same 1023-character truncation.
pub unsafe fn mxml_error(format: *const c_char) {
    let mut s: [c_char; 1024] = [0; 1024];
    let global: *mut MxmlGlobal = mxml_global();

    if format.is_null() {
        return;
    }

    libc::snprintf(
        s.as_mut_ptr(),
        core::mem::size_of::<[c_char; 1024]>(),
        c"%s".as_ptr(),
        format,
    );

    if let Some(cb) = (*global).error_cb {
        cb(s.as_ptr());
    } else {
        libc::fprintf(stderr, c"mxml: %s\n".as_ptr(), s.as_ptr());
    }
}

/// Matches C `mxml_ignore_cb` (`mxml-private.c:71`).
pub unsafe extern "C" fn mxml_ignore_cb(_node: *mut MxmlNode) -> MxmlType {
    MXML_IGNORE
}

/// Matches C `mxml_integer_cb` (`mxml-private.c:85`).
pub unsafe extern "C" fn mxml_integer_cb(_node: *mut MxmlNode) -> MxmlType {
    MXML_INTEGER
}

/// Matches C `mxml_opaque_cb` (`mxml-private.c:99`).
pub unsafe extern "C" fn mxml_opaque_cb(_node: *mut MxmlNode) -> MxmlType {
    MXML_OPAQUE
}

/// Matches C `mxml_real_cb` (`mxml-private.c:113`).
pub unsafe extern "C" fn mxml_real_cb(_node: *mut MxmlNode) -> MxmlType {
    MXML_REAL
}

/// Matches C static `_mxml_destructor` (`mxml-private.c:119`).
pub unsafe extern "C" fn _mxml_destructor(g: *mut c_void) {
    libc::free(g);
}

/// Matches C static `_MXML_FINI`/`_mxml_fini` (`mxml-private.c:130`).
///
/// The C function carries `__attribute((destructor))` so the loader runs it at
/// unload.  Rust has no portable equivalent; the body is translated but is not
/// registered, which only affects process-exit cleanup of the per-thread block.
pub unsafe fn _mxml_fini() {
    let global: *mut MxmlGlobal;

    if MXML_KEY != !0 {
        global = libc::pthread_getspecific(MXML_KEY) as *mut MxmlGlobal;
        if !global.is_null() {
            _mxml_destructor(global as *mut c_void);
        }

        libc::pthread_key_delete(MXML_KEY);
        MXML_KEY = !0;
    }
}

/// Matches C `_mxml_global` (`mxml-private.c:151`).
pub unsafe fn mxml_global() -> *mut MxmlGlobal {
    let mut global: *mut MxmlGlobal;

    libc::pthread_once(&raw mut MXML_KEY_ONCE, _mxml_init);

    global = libc::pthread_getspecific(MXML_KEY) as *mut MxmlGlobal;
    if global.is_null() {
        global = libc::calloc(1, core::mem::size_of::<MxmlGlobal>()) as *mut MxmlGlobal;
        libc::pthread_setspecific(MXML_KEY, global as *const c_void);

        (*global).num_entity_cbs = 1;
        (*global).entity_cbs[0] = Some(mxml_entity_cb);
        (*global).wrap = 72;
    }

    global
}

/// Matches C static `_mxml_init` (`mxml-private.c:177`).
pub extern "C" fn _mxml_init() {
    unsafe {
        libc::pthread_key_create(&raw mut MXML_KEY, Some(_mxml_destructor));
    }
}
