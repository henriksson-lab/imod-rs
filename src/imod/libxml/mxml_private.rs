//! Translation of `IMOD/libxml/mxml-private.c` and `IMOD/libxml/mxml-private.h`.
//!
//! `config.h` defines `HAVE_PTHREAD_H` on every non-Windows build, so the
//! per-thread `pthread_key_t` variant of `_mxml_global()` is the one compiled
//! into the vendored library and the one translated here.  The `WIN32` and
//! no-threads variants are not part of this build.
//!
//! The `pthread_key_t` plus `pthread_once` plus `calloc` machinery is exactly
//! what Rust's `thread_local!` is: a lazily created, per-thread block with a
//! destructor.  `_mxml_key`, `_mxml_key_once`, `_mxml_init` and
//! `_mxml_destructor` therefore become the declaration of [`MXML_GLOBAL`] and
//! its `Drop`, and `_mxml_global` hands out the key rather than a pointer, so
//! a caller reaches the block with `mxml_global().with_borrow_mut(...)`.
#![allow(dead_code)]

use super::*;
use core::cell::RefCell;
use core::ffi::c_int;
use std::io::Write;
use std::thread::LocalKey;

/// Matches C `_mxml_global_t` (`mxml-private.h:30`).
pub struct MxmlGlobal {
    pub error_cb: MxmlErrorCb,
    pub num_entity_cbs: c_int,
    pub entity_cbs: [MxmlEntityCb; 100],
    pub wrap: c_int,
    pub custom_load_cb: MxmlCustomLoadCb,
    pub custom_save_cb: MxmlCustomSaveCb,
}

thread_local! {
    /// Matches the per-thread block C reaches through static `_mxml_key`
    /// (`mxml-private.c:106`).  The C `calloc`s it and then fills in the three
    /// fields at `mxml-private.c:170-172`; those are the initialiser here.
    static MXML_GLOBAL: RefCell<MxmlGlobal> = RefCell::new(MxmlGlobal {
        error_cb: None,
        num_entity_cbs: 1,
        entity_cbs: {
            let mut cbs: [MxmlEntityCb; 100] = [None; 100];
            cbs[0] = Some(mxml_entity_cb);
            cbs
        },
        wrap: 72,
        custom_load_cb: None,
        custom_save_cb: None,
    });
}

/// Matches C `mxml_error` (`mxml-private.c:41`).
///
/// The C function is `mxml_error(const char *format, ...)` and runs the
/// arguments through `vsnprintf` into a 1024-byte buffer.  Stable Rust cannot
/// define a C-variadic function, so callers format their arguments themselves
/// and pass the result here; truncating at 1023 bytes keeps the same
/// truncation the C buffer imposes.
pub fn mxml_error(format: &[u8]) {
    let error_cb: MxmlErrorCb = mxml_global().with_borrow(|global| global.error_cb);

    /*
     * Range check input...  (the C tests `format` for NULL, which a slice
     * cannot be.)
     */

    /*
     * Format the error message string...
     */

    let s: &[u8] = &format[..format.len().min(1023)];

    /*
     * And then display the error message...
     */

    if let Some(cb) = error_cb {
        cb(s);
    } else {
        let mut stderr = std::io::stderr();
        let _ = stderr.write_all(b"mxml: ");
        let _ = stderr.write_all(s);
        let _ = stderr.write_all(b"\n");
    }
}

/// Matches C `mxml_ignore_cb` (`mxml-private.c:71`).
pub fn mxml_ignore_cb(_arena: &MxmlArena, _node: Option<usize>) -> MxmlType {
    MXML_IGNORE
}

/// Matches C `mxml_integer_cb` (`mxml-private.c:85`).
pub fn mxml_integer_cb(_arena: &MxmlArena, _node: Option<usize>) -> MxmlType {
    MXML_INTEGER
}

/// Matches C `mxml_opaque_cb` (`mxml-private.c:99`).
pub fn mxml_opaque_cb(_arena: &MxmlArena, _node: Option<usize>) -> MxmlType {
    MXML_OPAQUE
}

/// Matches C `mxml_real_cb` (`mxml-private.c:113`).
pub fn mxml_real_cb(_arena: &MxmlArena, _node: Option<usize>) -> MxmlType {
    MXML_REAL
}

/// Matches C static `_mxml_destructor` (`mxml-private.c:119`).
///
/// The C body is `free(g)`, the `pthread_key_create` destructor for the
/// per-thread block.  A `thread_local!` runs the block's `Drop` at the same
/// point, so the function has nothing left to do; taking the block by value
/// keeps the ownership transfer the `free` performed.
pub fn _mxml_destructor(g: MxmlGlobal) {
    drop(g);
}

/// Matches C static `_MXML_FINI`/`_mxml_fini` (`mxml-private.c:130`).
///
/// The C function carries `__attribute((destructor))` so the loader runs it at
/// unload: it frees this thread's block and deletes the key.  Rust has no
/// portable equivalent and none is needed — the `thread_local!` block is
/// dropped when the thread ends — so the body resets the block to the state a
/// fresh `_mxml_global` would build, which is what a later call would see.
pub fn _mxml_fini() {
    mxml_global().with_borrow_mut(|global| {
        global.error_cb = None;
        global.num_entity_cbs = 1;
        global.entity_cbs = [None; 100];
        global.entity_cbs[0] = Some(mxml_entity_cb);
        global.wrap = 72;
        global.custom_load_cb = None;
        global.custom_save_cb = None;
    });
}

/// Matches C `_mxml_global` (`mxml-private.c:151`).
///
/// The C returns a pointer into the per-thread block; the key itself is
/// returned instead, so a caller writes `mxml_global().with_borrow_mut(|global|
/// ...)` where the C writes `global = _mxml_global(); global->wrap = ...`.
pub fn mxml_global() -> &'static LocalKey<RefCell<MxmlGlobal>> {
    &MXML_GLOBAL
}

/// Matches C static `_mxml_init` (`mxml-private.c:177`).
///
/// The C body is the one-time `pthread_key_create`; `thread_local!` creates
/// its key on first access, so there is nothing to do here.
pub fn _mxml_init() {}
