//! Safe, owned JPEG backend for `IMOD/libiimod/iijpeg.c`.
#![allow(dead_code)]
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_JPEG, IIFORMAT_LUMINANCE,
    IIFORMAT_RGB, IISTATE_READY, IITYPE_UBYTE, ImodImageFile, ii_simple_fill_mrc_header_callback,
};
use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, MRC_MODE_RGB};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageFormat};
use std::io::{Read as _, Seek as _, SeekFrom};
const NOPROC: i32 = 0;
const BYTE: i32 = 1;
const FLOAT: i32 = 2;
const USHORT: i32 = 3;
fn count(f: &ImodImageFile) -> Option<usize> {
    usize::try_from(f.nx)
        .ok()?
        .checked_mul(usize::try_from(f.ny).ok()?)
}
fn jpeg_delete(f: &mut ImodImageFile) {
    f.native_image_pixels = None;
    f.native_image_rgb = false;
}
/// C `iiJPEGCheck`: decode once into crate-owned pixels instead of retaining libjpeg state.
pub fn ii_jpeg_check(f: &mut ImodImageFile) -> i32 {
    let Some(mut fp) = f.fp.clone() else {
        return IIERR_BAD_CALL;
    };
    if fp.seek(SeekFrom::Start(0)).is_err() {
        return IIERR_IO_ERROR;
    };
    let mut raw = vec![];
    if fp.read_to_end(&mut raw).is_err() || raw.len() < 3 {
        return IIERR_IO_ERROR;
    };
    if raw[..3] != [255, 216, 255] {
        return IIERR_NOT_FORMAT;
    };
    let Ok(im) = image::load_from_memory_with_format(&raw, ImageFormat::Jpeg) else {
        return IIERR_IO_ERROR;
    };
    let gray = matches!(
        im.color(),
        ColorType::L8 | ColorType::La8 | ColorType::L16 | ColorType::La16
    );
    let (w, h) = (im.width(), im.height());
    let Ok(nx) = i32::try_from(w) else {
        return IIERR_IO_ERROR;
    };
    let Ok(ny) = i32::try_from(h) else {
        return IIERR_IO_ERROR;
    };
    f.nx = nx;
    f.ny = ny;
    f.nz = 1;
    f.type_ = IITYPE_UBYTE;
    f.format = if gray {
        IIFORMAT_LUMINANCE
    } else {
        IIFORMAT_RGB
    };
    f.mode = if gray { MRC_MODE_BYTE } else { MRC_MODE_RGB };
    f.file = IIFILE_JPEG;
    f.amin = 0.;
    f.amax = 255.;
    f.amean = 128.;
    f.native_image_rgb = !gray;
    f.native_image_pixels = Some(if gray {
        im.into_luma8().into_raw()
    } else {
        im.into_rgb8().into_raw()
    });
    f.clean_up = Some(jpeg_delete_callback);
    f.fill_mrc_header = Some(ii_simple_fill_mrc_header_callback);
    f.read_section = Some(read_callback);
    f.read_section_byte = Some(read_byte_callback);
    f.read_section_ushort = Some(read_ushort_callback);
    f.read_section_float = Some(read_float_callback);
    0
}
/// C `jpegReadSectionAny`.
fn jpeg_read_section_any(f: &ImodImageFile, out: &mut [u8], z: i32, kind: i32) -> i32 {
    if z != 0 {
        return IIERR_BAD_CALL;
    }
    let Some(src) = f.native_image_pixels.as_deref() else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let c = if f.native_image_rgb { 3 } else { 1 };
    let b = match kind {
        FLOAT => 4,
        USHORT => 2,
        _ => 1,
    };
    let ch = if kind == NOPROC { c } else { 1 };
    if out.len() != n * ch * b || src.len() != n * c {
        return IIERR_BAD_CALL;
    }
    for oy in 0..f.ny as usize {
        let sy = f.ny as usize - 1 - oy;
        for x in 0..f.nx as usize {
            let si = (sy * f.nx as usize + x) * c;
            let lum = if c == 1 {
                src[si]
            } else {
                (0.299 * src[si] as f32 + 0.587 * src[si + 1] as f32 + 0.114 * src[si + 2] as f32)
                    .round() as u8
            };
            let i = oy * f.nx as usize + x;
            match kind {
                NOPROC => out[i * c..i * c + c].copy_from_slice(&src[si..si + c]),
                BYTE => out[i] = lum,
                USHORT => out[2 * i..2 * i + 2].copy_from_slice(&(lum as u16 * 257).to_ne_bytes()),
                FLOAT => out[4 * i..4 * i + 4].copy_from_slice(&(lum as f32).to_ne_bytes()),
                _ => return IIERR_BAD_CALL,
            }
        }
    }
    0
}

/// C `jpegReadSection`.
fn jpeg_read_section(f: &ImodImageFile, out: &mut [u8], z: i32) -> i32 {
    jpeg_read_section_any(f, out, z, NOPROC)
}

/// C `jpegReadSectionByte`.
fn jpeg_read_section_byte(f: &ImodImageFile, out: &mut [u8], z: i32) -> i32 {
    jpeg_read_section_any(f, out, z, BYTE)
}

/// C `jpegReadSectionUShort`.
fn jpeg_read_section_ushort(f: &ImodImageFile, out: &mut [u8], z: i32) -> i32 {
    jpeg_read_section_any(f, out, z, USHORT)
}

/// C `jpegReadSectionFloat`.
fn jpeg_read_section_float(f: &ImodImageFile, out: &mut [u8], z: i32) -> i32 {
    jpeg_read_section_any(f, out, z, FLOAT)
}
/// C `jpegOpenNew`.
pub fn jpeg_open_new(f: &mut ImodImageFile) -> i32 {
    let Some(name) = f.filename.as_deref() else {
        return IIERR_BAD_CALL;
    };
    let Some(fp) = ImodFile::open(name, "wb") else {
        return IIERR_IO_ERROR;
    };
    f.fp = Some(fp);
    f.state = IISTATE_READY;
    f.clean_up = Some(jpeg_delete_callback);
    f.fill_mrc_header = Some(ii_simple_fill_mrc_header_callback);
    f.write_section = Some(write_callback);
    f.write_section_float = Some(write_float_callback);
    // C treats any nonzero `lastWrittenZ` as the not-yet-written sentinel;
    // `-1` is the normal iimage new-file value.
    f.last_written_z = -1;
    0
}

pub(crate) unsafe fn jpeg_open_new_callback(p: *mut ImodImageFile) -> i32 {
    let Some(file) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_open_new(file)
}
/// `jpegWriteSection` (`iijpeg.c:435`).  This is the direct one-section
/// writer used after iimage has converted float data and selected quality.
/// The `image` encoder used by the Rust backend does not expose JFIF density
/// fields, so `resolution` is validated at this layer but the host encoder
/// cannot currently emit it.
pub fn jpeg_write_section(
    f: &mut ImodImageFile,
    input: &[u8],
    inverted: bool,
    resolution: i32,
    quality: i32,
) -> i32 {
    if f.write_section.is_none()
        || f.last_written_z == 0
        || !matches!(f.mode, MRC_MODE_BYTE | MRC_MODE_RGB)
    {
        return IIERR_BAD_CALL;
    }
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let channels = if f.mode == MRC_MODE_RGB { 3 } else { 1 };
    if input.len() != n * channels {
        return IIERR_BAD_CALL;
    }
    // Native accepts 0 for no JFIF density and otherwise writes a u16 value.
    let _resolution = resolution.clamp(0, u16::MAX as i32);
    let row = f.nx as usize * channels;
    let mut top_down = vec![0; input.len()];
    for output_y in 0..f.ny as usize {
        let source_y = if inverted {
            output_y
        } else {
            f.ny as usize - 1 - output_y
        };
        top_down[output_y * row..(output_y + 1) * row]
            .copy_from_slice(&input[source_y * row..(source_y + 1) * row]);
    }
    let Some(fp) = f.fp.as_mut() else {
        return IIERR_BAD_CALL;
    };
    if fp.seek(SeekFrom::Start(0)).is_err() {
        return IIERR_IO_ERROR;
    }
    let quality = if quality > 0 {
        quality.clamp(1, 100)
    } else {
        75
    } as u8;
    let color = if channels == 1 {
        ColorType::L8
    } else {
        ColorType::Rgb8
    };
    if JpegEncoder::new_with_quality(fp, quality)
        .encode(&top_down, f.nx as u32, f.ny as u32, color.into())
        .is_err()
    {
        return IIERR_IO_ERROR;
    }
    f.last_written_z = 0;
    0
}
/// C `iiJpegWriteSectionAny`.
fn ii_jpeg_write_section_any(f: &mut ImodImageFile, input: &[u8], z: i32, floats: bool) -> i32 {
    if z != 0
        || f.pad_left != 0
        || f.pad_right != 0
        || !((f.format == IIFORMAT_LUMINANCE && f.mode == MRC_MODE_BYTE)
            || (f.format == IIFORMAT_RGB && !floats))
        || f.llx != 0
        || f.lly != 0
        || (f.urx != -1 && f.urx != f.nx - 1)
        || (f.ury != -1 && f.ury != f.ny - 1)
    {
        return IIERR_BAD_CALL;
    }
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let c = if f.mode == MRC_MODE_RGB { 3 } else { 1 };
    if input.len() != n * c * if floats { 4 } else { 1 } {
        return IIERR_BAD_CALL;
    }
    let mut data = vec![0; n * c];
    if floats {
        for (i, v) in data.iter_mut().zip(input.chunks_exact(4)) {
            *i = f32::from_ne_bytes(v.try_into().unwrap())
                .round()
                .clamp(0., 255.) as u8
        }
    } else {
        data.copy_from_slice(input)
    }
    let q = std::env::var("IMOD_JPEG_QUALITY")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(75)
        .clamp(1, 100) as u8;
    jpeg_write_section(f, &data, false, 0, q as i32)
}

/// C `iiJpegWriteSection`.
fn ii_jpeg_write_section(f: &mut ImodImageFile, input: &[u8], z: i32) -> i32 {
    ii_jpeg_write_section_any(f, input, z, false)
}

/// C `iiJpegWriteSectionFloat`.
fn ii_jpeg_write_section_float(f: &mut ImodImageFile, input: &[u8], z: i32) -> i32 {
    ii_jpeg_write_section_any(f, input, z, true)
}
unsafe fn file(p: *mut ImodImageFile) -> Option<&'static mut ImodImageFile> {
    unsafe { p.as_mut() }
}
unsafe fn slice<'a>(p: *mut u8, n: usize) -> Option<&'a mut [u8]> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts_mut(p, n) })
    }
}
pub(crate) unsafe fn ii_jpeg_check_callback(p: *mut ImodImageFile) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    ii_jpeg_check(f)
}
unsafe fn jpeg_delete_callback(p: *mut ImodImageFile) {
    if let Some(f) = unsafe { file(p) } {
        jpeg_delete(f)
    }
}
unsafe fn read_cb(p: *mut ImodImageFile, b: *mut u8, z: i32, k: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let c = if k == NOPROC && f.native_image_rgb {
        3
    } else {
        1
    };
    let sz = match k {
        FLOAT => 4,
        USHORT => 2,
        _ => 1,
    };
    let Some(o) = (unsafe { slice(b, n * c * sz) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_read_section_any(f, o, z, k)
}
unsafe fn read_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let channels = if f.native_image_rgb { 3 } else { 1 };
    let Some(out) = (unsafe { slice(b, n * channels) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_read_section(f, out, z)
}
unsafe fn read_byte_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let Some(out) = (unsafe { slice(b, n) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_read_section_byte(f, out, z)
}
unsafe fn read_ushort_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let Some(out) = (unsafe { slice(b, n * 2) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_read_section_ushort(f, out, z)
}
unsafe fn read_float_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let Some(out) = (unsafe { slice(b, n * 4) }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_read_section_float(f, out, z)
}
unsafe fn write_cb(p: *mut ImodImageFile, b: *mut u8, z: i32, fl: bool) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let c = if f.mode == MRC_MODE_RGB { 3 } else { 1 };
    let Some(i) = (unsafe { slice(b, n * c * if fl { 4 } else { 1 }) }) else {
        return IIERR_BAD_CALL;
    };
    ii_jpeg_write_section_any(f, i, z, fl)
}
unsafe fn write_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let channels = if f.mode == MRC_MODE_RGB { 3 } else { 1 };
    let Some(input) = (unsafe { slice(b, n * channels) }) else {
        return IIERR_BAD_CALL;
    };
    ii_jpeg_write_section(f, input, z)
}
unsafe fn write_float_callback(p: *mut ImodImageFile, b: *mut u8, z: i32) -> i32 {
    let Some(f) = (unsafe { file(p) }) else {
        return IIERR_BAD_CALL;
    };
    let Some(n) = count(f) else {
        return IIERR_BAD_CALL;
    };
    let channels = if f.mode == MRC_MODE_RGB { 3 } else { 1 };
    let Some(input) = (unsafe { slice(b, n * channels * 4) }) else {
        return IIERR_BAD_CALL;
    };
    ii_jpeg_write_section_float(f, input, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_jpeg_roundtrip_preserves_dimensions_and_imod_row_order() {
        let file = ImodFile::tmpfile().unwrap();
        let mut writer = ImodImageFile::default();
        writer.fp = Some(file.clone());
        writer.nx = 2;
        writer.ny = 2;
        writer.nz = 1;
        writer.mode = MRC_MODE_BYTE;
        writer.format = IIFORMAT_LUMINANCE;
        writer.write_section = Some(write_callback);
        writer.last_written_z = -1;
        writer.urx = -1;
        writer.ury = -1;
        // IMOD order is bottom row then top row.
        let pixels = [0_u8, 0, 255, 255];
        assert_eq!(ii_jpeg_write_section(&mut writer, &pixels, 0), 0);

        let mut reader = ImodImageFile::default();
        reader.fp = Some(file);
        assert_eq!(ii_jpeg_check(&mut reader), 0);
        assert_eq!(
            (reader.nx, reader.ny, reader.nz, reader.mode),
            (2, 2, 1, MRC_MODE_BYTE)
        );
        let mut restored = [0_u8; 4];
        assert_eq!(jpeg_read_section(&reader, &mut restored, 0), 0);
        assert!(restored[..2].iter().all(|pixel| *pixel < 20));
        assert!(restored[2..].iter().all(|pixel| *pixel > 235));
    }

    #[test]
    fn direct_jpeg_writer_rejects_second_section_and_obeys_row_inversion() {
        let file = ImodFile::tmpfile().unwrap();
        let mut writer = ImodImageFile::default();
        writer.fp = Some(file.clone());
        writer.nx = 1;
        writer.ny = 2;
        writer.mode = MRC_MODE_BYTE;
        writer.write_section = Some(write_callback);
        writer.last_written_z = -1;
        writer.urx = -1;
        writer.ury = -1;
        assert_eq!(jpeg_write_section(&mut writer, &[0, 255], false, 0, 100), 0);
        assert_eq!(
            jpeg_write_section(&mut writer, &[0, 255], false, 0, 100),
            IIERR_BAD_CALL
        );
        let mut reader = ImodImageFile::default();
        reader.fp = Some(file);
        assert_eq!(ii_jpeg_check(&mut reader), 0);
        let mut pixels = [0; 2];
        assert_eq!(jpeg_read_section(&reader, &mut pixels, 0), 0);
        assert!(pixels[0] < 5 && pixels[1] > 250);
    }

    #[test]
    fn jpeg_read_converts_owned_luminance_to_requested_scalar_types() {
        let mut image = ImodImageFile::default();
        image.nx = 1;
        image.ny = 1;
        image.native_image_pixels = Some(vec![128]);
        let mut ushort = [0_u8; 2];
        let mut float = [0_u8; 4];
        assert_eq!(jpeg_read_section_ushort(&image, &mut ushort, 0), 0);
        assert_eq!(jpeg_read_section_float(&image, &mut float, 0), 0);
        assert_eq!(u16::from_ne_bytes(ushort), 32_896);
        assert_eq!(f32::from_ne_bytes(float), 128.);
    }
}
