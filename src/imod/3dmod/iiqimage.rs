//! Translation of `IMOD/3dmod/iiqimage.cpp`.
//!
//! The original uses QImage as a generic raster decoder.  The Rust version
//! keeps the same image-file callbacks while owning decoded L8 or RGB8 pixels
//! directly through the `image` crate.
#![allow(dead_code, unused_variables)]

use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_NOT_FORMAT, IIFILE_QIMAGE, IIFORMAT_LUMINANCE, IIFORMAT_RGB,
    IITYPE_UBYTE, ImodImageFile, ii_simple_fill_mrc_header_callback,
};
use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, MRC_MODE_RGB, get_byte_map};

/// Matches C `iiQImageCheck` (`iiqimage.cpp:35`).
pub unsafe fn ii_q_image_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    if unsafe { qimage_reopen(in_file) } != 0 {
        return IIERR_NOT_FORMAT;
    }
    unsafe {
        let file = &mut *in_file;
        file.nz = 1;
        file.file = IIFILE_QIMAGE;
        file.type_ = IITYPE_UBYTE;
        file.amean = 128.0;
        file.amax = 255.0;
        file.smax = 255.0;
        if file.native_image_rgb {
            file.format = IIFORMAT_RGB;
            file.mode = MRC_MODE_RGB;
        } else {
            file.format = IIFORMAT_LUMINANCE;
            file.mode = MRC_MODE_BYTE;
            file.read_section_float = Some(qimage_read_section_float);
        }
        file.read_section = Some(qimage_read_section);
        file.read_section_byte = Some(qimage_read_section_byte);
        file.clean_up = Some(qimage_close);
        file.reopen = Some(qimage_reopen);
        file.close = Some(qimage_close);
        file.fill_mrc_header = Some(ii_simple_fill_mrc_header_callback);
    }
    0
}

/// Matches C `qimageReadSectionByte` (`iiqimage.cpp:94`).
pub unsafe fn qimage_read_section_byte(
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
pub unsafe fn qimage_read_section_float(
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
pub unsafe fn qimage_read_section(
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
pub unsafe fn qimage_reopen(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return 1;
    }
    let file = unsafe { &mut *in_file };
    let Some(filename) = file.filename.as_deref() else {
        return 1;
    };
    let Ok(reader) = image::ImageReader::open(filename) else {
        return 1;
    };
    let Ok(decoded) = reader.decode() else {
        return 1;
    };
    let (pixels, rgb) = if decoded.color().has_color() {
        (decoded.to_rgb8().into_raw(), true)
    } else {
        (decoded.to_luma8().into_raw(), false)
    };
    file.nx = decoded.width() as i32;
    file.ny = decoded.height() as i32;
    file.header_size = 8;
    file.native_image_pixels = Some(pixels);
    file.native_image_rgb = rgb;
    file.backend_handle = core::ptr::null_mut();
    file.fp = None;
    0
}

/// Matches C `qimageClose` (`iiqimage.cpp:123`).
pub unsafe fn qimage_close(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    let file = unsafe { &mut *in_file };
    file.native_image_pixels = None;
    file.native_image_rgb = false;
    file.backend_handle = core::ptr::null_mut();
    file.fp = None;
}

/// Matches file-local C `ReadSection` (`iiqimage.cpp:134`).
fn read_section(in_file: &mut ImodImageFile, buf: &mut [u8], byte: i32) -> i32 {
    if in_file.axis == 2 || (byte > 1 && in_file.format != IIFORMAT_LUMINANCE) {
        return -1;
    }
    if in_file.native_image_pixels.is_none() && unsafe { qimage_reopen(in_file) } != 0 {
        return -1;
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
    let output_bytes = if byte == 2 {
        4
    } else if byte != 0 || !in_file.native_image_rgb {
        1
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
    let map = get_byte_map(in_file.slope, in_file.offset, 0, 255, 0);
    let Some(source) = in_file.native_image_pixels.as_deref() else {
        return -1;
    };
    let source_bytes = if in_file.native_image_rgb { 3 } else { 1 };
    let mut out = 0;
    for y in ymin..=ymax {
        let row = (in_file.ny - 1 - y) as usize * in_file.nx as usize * source_bytes;
        for x in xmin..=xmax {
            let source_at = row + x as usize * source_bytes;
            if in_file.native_image_rgb {
                let rgb = &source[source_at..source_at + 3];
                if byte != 0 {
                    let gray =
                        (0.3 * rgb[0] as f32 + 0.59 * rgb[1] as f32 + 0.11 * rgb[2] as f32) as u8;
                    buf[out] = map[gray as usize];
                    out += 1;
                } else {
                    buf[out..out + 3].copy_from_slice(rgb);
                    out += 3;
                }
            } else {
                let value = source[source_at];
                if byte == 0 {
                    buf[out] = value;
                    out += 1;
                } else if byte == 1 {
                    buf[out] = map[value as usize];
                    out += 1;
                } else {
                    buf[out..out + 4].copy_from_slice(&(map[value as usize] as f32).to_ne_bytes());
                    out += 4;
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{ii_q_image_check, qimage_close, qimage_read_section, qimage_reopen};
    use crate::imod::libiimod::iimage::{IIFORMAT_LUMINANCE, ImodImageFile};

    #[test]
    fn embedded_nul_name_is_rejected_without_a_c_string_boundary() {
        let mut file = ImodImageFile::default();
        file.filename = Some("not-a-path\0.png".into());
        assert_eq!(unsafe { ii_q_image_check(&mut file) }, 1);
        assert_eq!(unsafe { qimage_reopen(&mut file) }, 1);
    }

    #[test]
    fn native_png_path_sets_up_and_reads_bottom_to_top() {
        let filename = "fixtures/mrc2tif-float-scaled.png";
        let mut file = ImodImageFile::default();
        file.filename = Some(filename.into());
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
        let source = file.native_image_pixels.as_ref().unwrap();
        let mut expected = Vec::with_capacity(read.len());
        for row in source.chunks_exact(file.nx as usize).rev() {
            expected.extend_from_slice(row);
        }
        assert_eq!(read, expected);
        unsafe { qimage_close(&mut file) };
    }
}
