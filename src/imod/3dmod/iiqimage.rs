//! Translation of `IMOD/3dmod/iiqimage.cpp`.
//!
//! QImage remains the source decoder.  `iiqimage_qt.cpp` is the narrow direct
//! Qt ABI boundary; this module owns the source `ImodImageFile` setup and its
//! conversion loops.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::b3d_error;
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_NO_SUPPORT, IIERR_NOT_FORMAT, IIFILE_QIMAGE, IIFORMAT_LUMINANCE,
    IIFORMAT_RGB, IITYPE_UBYTE, ImodImageFile, ii_simple_fill_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, MRC_MODE_RGB, get_byte_map};
use core::ffi::{c_char, c_void};

/// The Qt C++ ABI of `iiqimage_qt.cpp`, compiled by `build.rs` and linked
/// statically.  This is the one boundary in this unit that keeps a C string:
/// `QImage::load` takes one, and the file name is handed over with a
/// terminator appended at the call site and nowhere else.
#[cfg(feature = "qt")]
unsafe extern "C" {
    fn iiqimage_open(filename: *const c_char) -> *mut c_void;
    fn iiqimage_delete(image: *mut c_void);
    fn iiqimage_is_null(image: *mut c_void) -> i32;
    fn iiqimage_width(image: *mut c_void) -> i32;
    fn iiqimage_height(image: *mut c_void) -> i32;
    fn iiqimage_depth(image: *mut c_void) -> i32;
    fn iiqimage_is_grayscale(image: *mut c_void) -> i32;
    fn iiqimage_color_count(image: *mut c_void) -> i32;
    fn iiqimage_color(image: *mut c_void, index: i32) -> u32;
    fn iiqimage_index_row(image: *mut c_void, y: i32, out: *mut u8, width: i32);
    fn iiqimage_rgb_row(image: *mut c_void, y: i32, out: *mut u8, width: i32);
}

#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_open(_filename: *const c_char) -> *mut c_void {
    core::ptr::null_mut()
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_delete(_image: *mut c_void) {}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_is_null(_image: *mut c_void) -> i32 {
    1
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_width(_image: *mut c_void) -> i32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_height(_image: *mut c_void) -> i32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_depth(_image: *mut c_void) -> i32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_is_grayscale(_image: *mut c_void) -> i32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_color_count(_image: *mut c_void) -> i32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_color(_image: *mut c_void, _index: i32) -> u32 {
    0
}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_index_row(_image: *mut c_void, _y: i32, _out: *mut u8, _width: i32) {}
#[cfg(not(feature = "qt"))]
unsafe fn iiqimage_rgb_row(_image: *mut c_void, _y: i32, _out: *mut u8, _width: i32) {}

/// Matches C `iiQImageCheck` (`iiqimage.cpp:35`).
pub unsafe extern "C" fn ii_q_image_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    // `iiqimage_qt.cpp` is the Qt C++ ABI and takes a C string; the
    // terminator is added here, at the boundary, and nowhere else.
    let Ok(name) =
        std::ffi::CString::new(unsafe { (*in_file).filename.as_deref().unwrap_or_default() })
    else {
        // A Rust filename can contain a NUL while a Qt C entry point cannot.
        // This is an unsupported source name, not a reason to abort a host
        // process through the image-file checker callback.
        return IIERR_NOT_FORMAT;
    };
    let image = unsafe { iiqimage_open(name.as_ptr().cast()) };
    if unsafe { iiqimage_is_null(image) } != 0 {
        unsafe { iiqimage_delete(image) };
        return IIERR_NOT_FORMAT;
    }
    if unsafe { iiqimage_depth(image) } < 8 {
        unsafe { iiqimage_delete(image) };
        unsafe {
            b3d_error(
                None,
                format_args!(
                    "{} is a recognized file type but data type is not supported\n",
                    (*in_file).filename.as_deref().unwrap_or("")
                ),
            );
        }
        return IIERR_NO_SUPPORT;
    }
    unsafe {
        (*in_file).fp = None;
        (*in_file).nx = iiqimage_width(image);
        (*in_file).ny = iiqimage_height(image);
        (*in_file).nz = 1;
        (*in_file).file = IIFILE_QIMAGE;
        (*in_file).type_ = IITYPE_UBYTE;
        (*in_file).amean = 128.0;
        (*in_file).amax = 255.0;
        (*in_file).smax = 255.0;
        if iiqimage_depth(image) == 8 && iiqimage_is_grayscale(image) != 0 {
            (*in_file).format = IIFORMAT_LUMINANCE;
            (*in_file).read_section_byte = Some(qimage_read_section_byte);
            (*in_file).read_section_float = Some(qimage_read_section_float);
            (*in_file).mode = MRC_MODE_BYTE;
            (*in_file).read_section = Some(qimage_read_section);
        } else {
            (*in_file).format = IIFORMAT_RGB;
            (*in_file).mode = MRC_MODE_RGB;
            (*in_file).read_section = Some(qimage_read_section);
            (*in_file).read_section_byte = Some(qimage_read_section_byte);
        }
        (*in_file).header_size = 8;
        (*in_file).header = image.cast();
        (*in_file).clean_up = Some(qimage_close);
        (*in_file).reopen = Some(qimage_reopen);
        (*in_file).close = Some(qimage_close);
        (*in_file).fill_mrc_header = Some(core::mem::transmute::<
            unsafe fn(*mut ImodImageFile, *mut crate::imod::libiimod::mrcfiles::MrcHeader) -> i32,
            unsafe extern "C" fn(
                *mut ImodImageFile,
                *mut crate::imod::libiimod::mrcfiles::MrcHeader,
            ) -> i32,
        >(ii_simple_fill_mrc_header));
    }
    0
}

/// Matches C `qimageReadSectionByte` (`iiqimage.cpp:94`).
pub unsafe extern "C" fn qimage_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let _ = in_section;
    let file = unsafe { &mut *in_file };
    let xmax = if file.urx < 0 { file.nx - 1 } else { file.urx };
    let ymax = if file.ury < 0 { file.ny - 1 } else { file.ury };
    let pixels = (xmax - file.llx + 1).max(0) * (ymax - file.lly + 1).max(0);
    read_section(
        file,
        unsafe { core::slice::from_raw_parts_mut(buf, pixels as usize) },
        1,
    )
}

/// Matches C `qimageReadSectionFloat` (`iiqimage.cpp:99`).
pub unsafe extern "C" fn qimage_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let _ = in_section;
    let file = unsafe { &mut *in_file };
    let xmax = if file.urx < 0 { file.nx - 1 } else { file.urx };
    let ymax = if file.ury < 0 { file.ny - 1 } else { file.ury };
    let pixels = (xmax - file.llx + 1).max(0) * (ymax - file.lly + 1).max(0);
    read_section(
        file,
        unsafe { core::slice::from_raw_parts_mut(buf, pixels as usize * 4) },
        2,
    )
}

/// Matches C `qimageReadSection` (`iiqimage.cpp:104`).
pub unsafe extern "C" fn qimage_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let _ = in_section;
    let file = unsafe { &mut *in_file };
    let xmax = if file.urx < 0 { file.nx - 1 } else { file.urx };
    let ymax = if file.ury < 0 { file.ny - 1 } else { file.ury };
    let pixels = (xmax - file.llx + 1).max(0) * (ymax - file.lly + 1).max(0);
    let bytes_per_pixel = if file.format == IIFORMAT_LUMINANCE {
        1
    } else {
        3
    };
    read_section(
        file,
        unsafe { core::slice::from_raw_parts_mut(buf, pixels as usize * bytes_per_pixel) },
        0,
    )
}

/// Matches C `qimageReopen` (`iiqimage.cpp:109`).
pub unsafe extern "C" fn qimage_reopen(in_file: *mut ImodImageFile) -> i32 {
    let Ok(name) =
        std::ffi::CString::new(unsafe { (*in_file).filename.as_deref().unwrap_or_default() })
    else {
        return 1;
    };
    let image = unsafe { iiqimage_open(name.as_ptr().cast()) };
    if unsafe { iiqimage_is_null(image) } != 0 {
        unsafe { iiqimage_delete(image) };
        return 1;
    }
    unsafe {
        (*in_file).header_size = 8;
        (*in_file).header = image.cast();
        // `iiqimage.cpp:110`: the QImage pointer cast to `FILE *` as this
        // file's identity, never used for I/O.
        (*in_file).fp = Some(crate::imod::libcfshr::b3dutil::ImodFile::Token(
            image as usize,
        ));
    }
    0
}

/// Matches C `qimageClose` (`iiqimage.cpp:123`).
pub unsafe extern "C" fn qimage_close(in_file: *mut ImodImageFile) {
    let image = unsafe { (*in_file).header.cast::<c_void>() };
    if !image.is_null() {
        unsafe { iiqimage_delete(image) };
    }
    unsafe {
        (*in_file).header = core::ptr::null_mut();
        (*in_file).fp = None;
    }
}

/// Matches file-local C `ReadSection` (`iiqimage.cpp:134`).
fn read_section(in_file: &mut ImodImageFile, buf: &mut [u8], byte: i32) -> i32 {
    let ysize = in_file.ny;
    if in_file.axis == 2 || (byte > 1 && in_file.format != IIFORMAT_LUMINANCE) {
        return -1;
    }
    if in_file.header.is_null() && unsafe { qimage_reopen(in_file) } != 0 {
        return -1;
    }
    let image = in_file.header.cast::<c_void>();
    let depth = unsafe { iiqimage_depth(image) };
    let direct_bytes = byte == 0 && depth == 8 && unsafe { iiqimage_is_grayscale(image) } != 0;
    let color_count = unsafe { iiqimage_color_count(image) };
    let mut colors = Vec::with_capacity(color_count.max(0) as usize);
    for index in 0..color_count {
        colors.push(unsafe { iiqimage_color(image, index) });
    }
    let xmin = in_file.llx;
    let ymin = in_file.lly;
    let xmax = if in_file.urx < 0 {
        in_file.nx - 1
    } else {
        in_file.urx
    };
    let ymax = if in_file.ury < 0 {
        in_file.ny - 1
    } else {
        in_file.ury
    };
    let mut map = [0_u8; 256];
    let mut map2 = [0_u8; 256];
    if byte != 0 || direct_bytes {
        if byte != 0 {
            let source_map = get_byte_map(in_file.slope, in_file.offset, 0, 255, 0);
            map2.copy_from_slice(unsafe { core::slice::from_raw_parts(source_map, 256) });
        }
        let maxind = (colors.len() as i32 - 1).max(0) as usize;
        for index in 0..256_usize {
            let pixel = if maxind > 0 {
                (colors[index.min(maxind)] >> 16) as u8
            } else {
                index as u8
            };
            map[index] = if byte != 0 {
                map2[pixel as usize]
            } else {
                pixel
            };
        }
    }
    let width = in_file.nx as usize;
    let mut indexes = vec![0_u8; width];
    let mut rgb = vec![0_u8; width * 3];
    let output_bytes = if byte != 0 && in_file.format != IIFORMAT_LUMINANCE {
        1
    } else if byte == 1 || direct_bytes {
        1
    } else if byte == 2 {
        core::mem::size_of::<f32>()
    } else {
        3
    };
    let pixels = (xmax - xmin + 1) as usize * (ymax - ymin + 1) as usize;
    if xmin < 0
        || ymin < 0
        || xmax >= in_file.nx
        || ymax >= in_file.ny
        || buf.len() < pixels * output_bytes
    {
        return -1;
    }
    let mut out = 0usize;
    for y in ymin..=ymax {
        if depth == 8 {
            unsafe { iiqimage_index_row(image, ysize - 1 - y, indexes.as_mut_ptr(), width as i32) };
        } else {
            unsafe { iiqimage_rgb_row(image, ysize - 1 - y, rgb.as_mut_ptr(), width as i32) };
        }
        if byte != 0 && in_file.format != IIFORMAT_LUMINANCE {
            for x in xmin..=xmax {
                let (r, g, b) = if depth == 8 {
                    let value = colors[indexes[x as usize] as usize];
                    ((value >> 16) as u8, (value >> 8) as u8, value as u8)
                } else {
                    let p = 3 * x as usize;
                    (rgb[p], rgb[p + 1], rgb[p + 2])
                };
                let gray = (0.3 * r as f32 + 0.59 * g as f32 + 0.11 * b as f32) as u8;
                buf[out] = map2[gray as usize];
                out += 1;
            }
        } else if byte == 1 || direct_bytes {
            for x in xmin..=xmax {
                buf[out] = map[indexes[x as usize] as usize];
                out += 1;
            }
        } else if byte == 2 {
            for x in xmin..=xmax {
                buf[out..][..4]
                    .copy_from_slice(&(map[indexes[x as usize] as usize] as f32).to_ne_bytes());
                out += core::mem::size_of::<f32>();
            }
        } else if depth == 8 {
            for x in xmin..=xmax {
                let value = colors[indexes[x as usize] as usize];
                buf[out..][..3].copy_from_slice(&[
                    (value >> 16) as u8,
                    (value >> 8) as u8,
                    value as u8,
                ]);
                out += 3;
            }
        } else {
            for x in xmin..=xmax {
                let p = 3 * x as usize;
                buf[out..][..3].copy_from_slice(&rgb[p..][..3]);
                out += 3;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{
        ii_q_image_check, iiqimage_index_row, qimage_close, qimage_read_section, qimage_reopen,
    };
    use crate::imod::libiimod::iimage::{IIFORMAT_LUMINANCE, ImodImageFile};

    #[test]
    fn qrgb_component_layout_matches_qt_qrgb() {
        let value = 0xff12_3456_u32;
        assert_eq!(
            ((value >> 16) as u8, (value >> 8) as u8, value as u8),
            (0x12, 0x34, 0x56)
        );
    }

    #[test]
    fn embedded_nul_name_is_rejected_at_the_qt_boundary() {
        let mut file = ImodImageFile::default();
        file.filename = Some("not-a-path\0.png".into());

        assert_eq!(unsafe { ii_q_image_check(&mut file) }, 1);
        assert_eq!(unsafe { qimage_reopen(&mut file) }, 1);
    }

    #[cfg(feature = "qt")]
    #[test]
    fn source_qimage_png_path_sets_up_and_reads_bottom_to_top() {
        let filename = "fixtures/mrc2tif-float-scaled.png";
        let mut file = ImodImageFile::default();
        file.filename = Some(filename.into());
        file.fp = crate::imod::libcfshr::b3dutil::ImodFile::open(filename, "rb");
        assert!(file.fp.is_some());
        assert_eq!(unsafe { ii_q_image_check(&mut file) }, 0);
        assert_eq!(file.format, IIFORMAT_LUMINANCE);
        assert_eq!(file.nz, 1);
        file.llx = 0;
        file.lly = 0;
        file.urx = -1;
        file.ury = -1;
        let mut read = vec![0_u8; (file.nx * file.ny) as usize];
        assert_eq!(
            unsafe { qimage_read_section(&mut file, read.as_mut_ptr(), 0) },
            0
        );
        let mut expected = vec![0_u8; read.len()];
        for y in 0..file.ny {
            unsafe {
                iiqimage_index_row(
                    file.header.cast(),
                    file.ny - 1 - y,
                    expected.as_mut_ptr().add((y * file.nx) as usize),
                    file.nx,
                );
            }
        }
        assert_eq!(read, expected);
        unsafe { qimage_close(&mut file) };
    }
}
