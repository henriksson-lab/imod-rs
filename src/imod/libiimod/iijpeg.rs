//! Translation scaffold for `IMOD/libiimod/iijpeg.c`.
//!
//! This unit deliberately retains the libjpeg C ABI as its boundary.  It must
//! not be replaced by a Rust image abstraction: IMOD stores a libjpeg
//! compression/decompression object in the explicit external image-backend
//! handle, uses libjpeg's
//! stdio source/destination managers, and relies on its error-manager ABI.
//! A generated, version-locked binding for `jpeglib.h` and a C-compatible
//! `setjmp`/`longjmp` error trampoline are still required to enable this
//! unit.  A Rust frame may not safely be crossed by `longjmp`, so, until that
//! native boundary is linked, the source entry points return IMOD's
//! `IIERR_NO_SUPPORT` rather than pretending an image crate is equivalent.
//!
//! Required external ABI: libjpeg / libjpeg-turbo (`jpeg_std_error`,
//! `jpeg_Create{Compress,Decompress}`, stdio source/destination, scanline,
//! abort, destroy, and error-manager structures) built against the same
//! `jpeglib.h` layout as the binding.
#![allow(dead_code, unused_variables)]

use crate::imod::libiimod::iimage::{IIERR_BAD_CALL, ImodImageFile};

/// IMOD `IIERR_NO_SUPPORT`.  JPEG needs the C error trampoline described in
/// this module's documentation before these entry points can enter libjpeg.
const IIERR_NO_SUPPORT: i32 = 4;

/// C `iiJPEGCheck` (`iijpeg.c:54`).
///
/// The direct libjpeg ABI implementation is pending its version-locked
/// `jpeglib.h` binding and C `setjmp` trampoline; see this module's header.
pub unsafe fn ii_jpeg_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    IIERR_NO_SUPPORT
}

/// C static `jpegDelete` (`iijpeg.c:141`).
unsafe fn jpeg_delete(in_file: *mut ImodImageFile) {
    // Destruction of an initialized libjpeg object is owned by the native
    // bridge.  There is no Rust allocation to release in the ABI-gated path.
}

/// C static `jpegReadSectionByte` (`iijpeg.c:155`).
unsafe fn jpeg_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    jpeg_read_section_any(in_file, buf, in_section, 1)
}

/// C static `jpegReadSectionUShort` (`iijpeg.c:160`).
unsafe fn jpeg_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    jpeg_read_section_any(in_file, buf, in_section, 3)
}

/// C static `jpegReadSectionFloat` (`iijpeg.c:165`).
unsafe fn jpeg_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    jpeg_read_section_any(in_file, buf, in_section, 2)
}

/// C static `jpegReadSection` (`iijpeg.c:170`).
unsafe fn jpeg_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    jpeg_read_section_any(in_file, buf, in_section, 0)
}

/// C static `jpegReadSectionAny` (`iijpeg.c:177`).
unsafe fn jpeg_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    type_: i32,
) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    IIERR_NO_SUPPORT
}

/// C `jpegOpenNew` (`iijpeg.c:322`).
pub unsafe fn jpeg_open_new(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    IIERR_NO_SUPPORT
}

/// C static `iiJpegWriteSection` (`iijpeg.c:342`).
unsafe fn ii_jpeg_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    ii_jpeg_write_section_any(in_file, buf, in_section, 0)
}

/// C static `iiJpegWriteSectionFloat` (`iijpeg.c:347`).
unsafe fn ii_jpeg_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    ii_jpeg_write_section_any(in_file, buf, in_section, 1)
}

/// C static `iiJpegWriteSectionAny` (`iijpeg.c:356`).
unsafe fn ii_jpeg_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    if_float: i32,
) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    IIERR_NO_SUPPORT
}

/// C `jpegWriteSection` (`iijpeg.c:435`).
pub unsafe fn jpeg_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    inverted: i32,
    resolution: i32,
    quality: i32,
) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    IIERR_NO_SUPPORT
}
