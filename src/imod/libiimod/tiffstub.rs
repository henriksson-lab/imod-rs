//! Translation of `IMOD/libiimod/tiffstub.c`.
//!
//! This source unit is linked when IMOD is built without libtiff.  Its return
//! values are therefore part of the unavailable-TIFF contract, not allocation
//! or I/O fallbacks.
#![allow(dead_code, non_upper_case_globals, unused_variables)]

use crate::imod::libiimod::iitif::Tiff;
use core::ffi::{c_char, c_int, c_void};

pub type Tdir = u16;
pub type Ttag = u32;
pub type Tsample = u16;
pub type Tstrip = u32;
pub type Tsize = isize;
pub type Tmsize = isize;
pub type Toff = u64;
pub type Thandle = *mut c_void;
pub type Tdata = *mut c_void;

/// Opaque representation of the platform C `va_list` at this no-libtiff ABI
/// boundary.  No handler is ever invoked by this stub implementation.
pub type TiffWarningHandler =
    Option<unsafe extern "C" fn(*const c_char, *const c_char, *mut c_void)>;
pub type TiffErrorHandler = TiffWarningHandler;
pub type TiffReadWriteProc = Option<unsafe extern "C" fn(Thandle, *mut c_void, Tmsize) -> Tmsize>;
pub type TiffSeekProc = Option<unsafe extern "C" fn(Thandle, Toff, c_int) -> Toff>;
pub type TiffCloseProc = Option<unsafe extern "C" fn(Thandle) -> c_int>;
pub type TiffSizeProc = Option<unsafe extern "C" fn(Thandle) -> Toff>;
pub type TiffMapFileProc =
    Option<unsafe extern "C" fn(Thandle, *mut *mut c_void, *mut Toff) -> c_int>;
pub type TiffUnmapFileProc = Option<unsafe extern "C" fn(Thandle, *mut c_void, Toff)>;

/// C global `version` (`tiffstub.c:4`), the string `TIFFGetVersion` returns.
/// The bytes are held as bytes; only the pointer handed back across the
/// libtiff ABI is a C string, and its terminator is the last element.
pub static VERSION: [u8; 9] = *b"IMODSTUB\0";

/// C `TIFFSetDirectory` (`tiffstub.c:6`).
pub unsafe fn tiff_set_directory(d: *mut Tiff, t: Tdir) -> c_int {
    -1
}

/// C `TIFFReadDirectory` (`tiffstub.c:10`).
pub unsafe fn tiff_read_directory(d: *mut Tiff) -> c_int {
    0
}

/// C `TIFFWriteDirectory` (`tiffstub.c:14`).
pub unsafe fn tiff_write_directory(d: *mut Tiff) -> c_int {
    0
}

/// C `TIFFGetField` (`tiffstub.c:18`).  The C variadic tail is deliberately
/// absent because the stub never reads it.
pub unsafe fn tiff_get_field(d: *mut Tiff, t: Ttag) -> c_int {
    0
}

/// C `TIFFSetField` (`tiffstub.c:23`).  The C variadic tail is deliberately
/// absent because the stub never reads it.
pub unsafe fn tiff_set_field(d: *mut Tiff, t: Ttag) -> c_int {
    0
}

/// C `TIFFNumberOfStrips` (`tiffstub.c:28`).
pub unsafe fn tiff_number_of_strips(d: *mut Tiff) -> Tstrip {
    0
}

/// C `TIFFStripSize` (`tiffstub.c:33`).
pub unsafe fn tiff_strip_size(tif: *mut Tiff) -> Tsize {
    0
}

/// C `TIFFReadEncodedStrip` (`tiffstub.c:38`).
pub unsafe fn tiff_read_encoded_strip(t: *mut Tiff, s: Tstrip, d: Tdata, z: Tsize) -> Tsize {
    0
}

/// C `TIFFWriteEncodedStrip` (`tiffstub.c:43`).
pub unsafe fn tiff_write_encoded_strip(t: *mut Tiff, s: Tstrip, d: Tdata, z: Tsize) -> Tsize {
    0
}

/// C `TIFFTileSize` (`tiffstub.c:48`).
pub unsafe fn tiff_tile_size(tif: *mut Tiff) -> Tsize {
    0
}

/// C `TIFFReadEncodedTile` (`tiffstub.c:53`).
pub unsafe fn tiff_read_encoded_tile(t: *mut Tiff, s: Tstrip, d: Tdata, z: Tsize) -> Tsize {
    0
}

/// C `TIFFReadScanline` (`tiffstub.c:58`).
pub unsafe fn tiff_read_scanline(t: *mut Tiff, d: Tdata, u: u32, s: Tsample) -> c_int {
    0
}

/// C `TIFFOpen` (`tiffstub.c:63`).
pub unsafe fn tiff_open(a: *const c_char, b: *const c_char) -> *mut Tiff {
    core::ptr::null_mut()
}

/// C `TIFFClose` (`tiffstub.c:68`).
pub unsafe fn tiff_close(d: *mut Tiff) {}

/// C `TIFFSetWarningHandler` (`tiffstub.c:76`).
pub unsafe fn tiff_set_warning_handler(handler: TiffWarningHandler) -> TiffWarningHandler {
    None
}

/// C `TIFFSetErrorHandler` (`tiffstub.c:81`).
pub unsafe fn tiff_set_error_handler(handler: TiffErrorHandler) -> TiffErrorHandler {
    None
}

/// C `TIFFGetVersion` (`tiffstub.c:86`).
pub unsafe fn tiff_get_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// C `TIFFIsByteSwapped` (`tiffstub.c:91`).
pub unsafe fn tiff_is_byte_swapped(t: *mut Tiff) -> c_int {
    0
}

/// C `_TIFFmalloc` (`tiffstub.c:96`).
pub unsafe fn tiff_malloc(s: Tmsize) -> *mut c_void {
    core::ptr::null_mut()
}

/// C `_TIFFfree` (`tiffstub.c:98`).
pub unsafe fn tiff_free(p: *mut c_void) {}

/// C `TIFFClientOpen` (`tiffstub.c:100`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn tiff_client_open(
    name: *const c_char,
    name2: *const c_char,
    dum1: Thandle,
    dum2: TiffReadWriteProc,
    dum3: TiffReadWriteProc,
    dum4: TiffSeekProc,
    dum8: TiffCloseProc,
    dum5: TiffSizeProc,
    dum6: TiffMapFileProc,
    dum7: TiffUnmapFileProc,
) -> *mut Tiff {
    core::ptr::null_mut()
}

/// C `TIFFWriteEncodedTile` (`tiffstub.c:110`).
pub unsafe fn tiff_write_encoded_tile(
    tif: *mut Tiff,
    tile: u32,
    data: *mut c_void,
    cc: Tmsize,
) -> Tmsize {
    0
}

/// C `TIFFRawStripSize` (`tiffstub.c:115`).
pub unsafe fn tiff_raw_strip_size(tif: *mut Tiff, strip: Tstrip) -> Tsize {
    0
}

/// C `TIFFReadRawStrip` (`tiffstub.c:120`).
pub unsafe fn tiff_read_raw_strip(
    tif: *mut Tiff,
    strip: u32,
    buf: *mut c_void,
    size: Tmsize,
) -> Tmsize {
    0
}

/// C `TIFFWriteRawStrip` (`tiffstub.c:125`).
pub unsafe fn tiff_write_raw_strip(
    tif: *mut Tiff,
    strip: u32,
    data: *mut c_void,
    cc: Tmsize,
) -> Tmsize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_tiff_contract_matches_source() {
        unsafe {
            assert_eq!(tiff_set_directory(core::ptr::null_mut(), 0), -1);
            assert_eq!(tiff_read_directory(core::ptr::null_mut()), 0);
            assert!(tiff_open(core::ptr::null(), core::ptr::null()).is_null());
            assert_eq!(
                core::slice::from_raw_parts(tiff_get_version().cast::<u8>(), VERSION.len() - 1),
                b"IMODSTUB"
            );
        }
    }
}
