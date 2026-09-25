//! Translation of `IMOD/flib/subrs/imsubs/convert_vms.c`: byte-order and VMS
//! float conversion callable from Fortran.
//!
//! Which arms are live is decided by the generated `imodconfig.h`
//! (`/tmp/imod-reference-build/include/imodconfig.h:10`, written by
//! `IMOD/setup`), which defines `SWAP_IEEE_FLOATS` and neither
//! `OLD_F77FUNCAP` nor `TEST_DRIVER`.  So `convert_floats` is the IEEE
//! byte-swap copy of `convert_longs` (`convert_vms.c:77-94`), not the
//! VMS-to-big-endian arm under `#else` (`:98-127`); `convert_ufloats`
//! (`:131-158`, `#ifdef OLD_F77FUNCAP`) and the `main` test driver
//! (`:162-202`, `#ifdef TEST_DRIVER`) are not compiled and so not translated.
//! The `F77FUNCAP`/`G77__HACK` name mangling is an ABI concern that no longer
//! exists once the Fortran callers are Rust.
//!
//! The C takes `unsigned char *data` whatever the caller's element type, so
//! the translated routines take the bytes of the caller's array.

/// Original: `convert_shorts` (`convert_vms.c:44`).
pub fn convert_shorts(data: &mut [u8], amt: i32) {
    let ldata = (amt * 2) as usize;
    let mut ptr = 0usize;
    let mut tmp: u8;

    while ptr < ldata {
        tmp = data[ptr];
        data[ptr] = data[ptr + 1];
        data[ptr + 1] = tmp;
        ptr += 2;
    }
}

/// Original: `convert_longs` (`convert_vms.c:58`).
pub fn convert_longs(data: &mut [u8], amt: i32) {
    let ldata = (amt * 4) as usize;
    let mut ptr = 0usize;
    let mut tmp: u8;
    while ptr < ldata {
        tmp = data[ptr];
        data[ptr] = data[ptr + 3];
        data[ptr + 3] = tmp;
        ptr += 1;
        tmp = data[ptr];
        data[ptr] = data[ptr + 1];
        data[ptr + 1] = tmp;
        ptr += 3;
    }
}

/* IEEE: use a copy of convert_longs to swap the bytes for convert_floats */

/// Original: `convert_floats` (`convert_vms.c:79`), the `SWAP_IEEE_FLOATS`
/// arm.
pub fn convert_floats(data: &mut [u8], amt: i32) {
    let ldata = (amt * 4) as usize;
    let mut ptr = 0usize;
    let mut tmp: u8;
    while ptr < ldata {
        tmp = data[ptr];
        data[ptr] = data[ptr + 3];
        data[ptr + 3] = tmp;
        ptr += 1;
        tmp = data[ptr];
        data[ptr] = data[ptr + 1];
        data[ptr + 1] = tmp;
        ptr += 3;
    }
}
