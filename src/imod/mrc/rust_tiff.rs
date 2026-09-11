//! Experimental Rust `tiff`-crate reader for the legacy tif2mrc boundary.
//!
//! It decodes supported pages eagerly, then exposes the same malloc-owned
//! section buffers that `tiff.c` returns to tif2mrc.  The default path never
//! enters this module.

use crate::imod::mrc::tiff::TfInfo;

#[cfg(feature = "rust-tiff")]
mod implementation {
    use super::TfInfo;
    use crate::imod::libiimod::iimage::{
        IIFILE_TIFF, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_INT, IITYPE_SHORT, IITYPE_UBYTE,
        IITYPE_UINT, IITYPE_USHORT, ii_new,
    };
    use crate::imod::libiimod::iitif::{IICOMPRESSION_LZW, IICOMPRESSION_NONE, IICOMPRESSION_ZIP};
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    };
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::fs::File;
    use std::sync::{LazyLock, Mutex};
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;

    struct Image {
        data: Vec<u8>,
    }

    struct FileData {
        images: Vec<Image>,
    }

    static FILES: LazyLock<Mutex<HashMap<usize, FileData>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    fn bytes(result: DecodingResult) -> (Vec<u8>, i32, i32, i32) {
        match result {
            DecodingResult::U8(values) => (values, 8, IITYPE_UBYTE, MRC_MODE_BYTE),
            DecodingResult::I8(values) => (
                values.into_iter().map(|value| value as u8).collect(),
                8,
                IITYPE_BYTE,
                MRC_MODE_BYTE,
            ),
            DecodingResult::U16(values) => (
                values.into_iter().flat_map(u16::to_ne_bytes).collect(),
                16,
                IITYPE_USHORT,
                MRC_MODE_USHORT,
            ),
            DecodingResult::I16(values) => (
                values.into_iter().flat_map(i16::to_ne_bytes).collect(),
                16,
                IITYPE_SHORT,
                MRC_MODE_SHORT,
            ),
            DecodingResult::U32(values) => (
                values.into_iter().flat_map(u32::to_ne_bytes).collect(),
                32,
                IITYPE_UINT,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::I32(values) => (
                values.into_iter().flat_map(i32::to_ne_bytes).collect(),
                32,
                IITYPE_INT,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::F32(values) => (
                values.into_iter().flat_map(f32::to_ne_bytes).collect(),
                32,
                IITYPE_FLOAT,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::F16(_)
            | DecodingResult::F64(_)
            | DecodingResult::U64(_)
            | DecodingResult::I64(_) => (Vec::new(), 0, 0, 0),
        }
    }

    fn flip_rows(data: &mut [u8], width: usize, height: usize, bytes_per_pixel: usize) {
        let row = width * bytes_per_pixel;
        for y in 0..height / 2 {
            let other = height - y - 1;
            for index in 0..row {
                data.swap(y * row + index, other * row + index);
            }
        }
    }

    pub unsafe fn open_file(filename: *mut i8, tif: *mut TfInfo, any_tif_pixel: i32) -> i32 {
        if filename.is_null() || tif.is_null() {
            return 1;
        }
        let Ok(path) = unsafe { CStr::from_ptr(filename) }.to_str() else {
            return 1;
        };
        let Ok(file) = File::open(path) else {
            return 1;
        };
        let Ok(mut decoder) = Decoder::new(file) else {
            return 1;
        };
        let mut images = Vec::new();
        let mut first = None;
        loop {
            let Ok((width, height)) = decoder.dimensions() else {
                return 1;
            };
            let Ok(color) = decoder.colortype() else {
                return 1;
            };
            // IMOD's libtiff reader drops alpha samples and, for indexed
            // files, keeps the indices unless the caller explicitly expands
            // the palette.  Preserve those boundary semantics here.
            let rgb = matches!(color, ColorType::RGB(8) | ColorType::RGBA(8));
            let discard_alpha = matches!(color, ColorType::RGBA(8) | ColorType::GrayA(8));
            let palette = matches!(color, ColorType::Palette(8));
            let min_is_white = decoder
                .get_tag_unsigned::<u16>(Tag::PhotometricInterpretation)
                .is_ok_and(|value| value == 0);
            if !rgb
                && !palette
                && !matches!(color, ColorType::Gray(8 | 16 | 32) | ColorType::GrayA(8))
            {
                return 1;
            }
            let Ok(result) = decoder.read_image() else {
                return 1;
            };
            let (mut data, bits, type_, mode) = bytes(result);
            if data.is_empty() {
                return 1;
            }
            if discard_alpha {
                let source_samples = if rgb { 4 } else { 2 };
                let kept_samples = source_samples - 1;
                if bits != 8 || data.len() != width as usize * height as usize * source_samples {
                    return 1;
                }
                data = data
                    .chunks_exact(source_samples)
                    .flat_map(|pixel| pixel[..kept_samples].iter().copied())
                    .collect();
            }
            // The Rust TIFF decoder normalizes MINISWHITE samples, while
            // IMOD's existing libtiff path copies these samples unchanged.
            // Undo that normalization so this alternative reader has the
            // established tif2mrc observable behavior.
            if min_is_white && !rgb && !palette {
                match bits {
                    8 => data.iter_mut().for_each(|value| *value = 255 - *value),
                    16 => data.chunks_exact_mut(2).for_each(|value| {
                        let sample = u16::from_ne_bytes([value[0], value[1]]);
                        value.copy_from_slice(&(u16::MAX - sample).to_ne_bytes());
                    }),
                    _ => return 1,
                }
            }
            let samples = if rgb { 3 } else { 1 };
            let bytes_per_pixel = samples * (bits as usize / 8);
            if rgb && bits != 8 {
                return 1;
            }
            if let Some((old_width, old_height, old_bits, old_type, old_rgb)) = first {
                if (width, height, bits, type_, rgb)
                    != (old_width, old_height, old_bits, old_type, old_rgb)
                {
                    return 1;
                }
            } else {
                first = Some((width, height, bits, type_, rgb));
            }
            flip_rows(&mut data, width as usize, height as usize, bytes_per_pixel);
            images.push(Image { data });
            if !decoder.more_images() {
                break;
            }
            if decoder.next_image().is_err() {
                return 1;
            }
            let _ = mode;
        }
        let Some((width, height, bits, type_, rgb)) = first else {
            return 1;
        };
        let iifile = ii_new();
        if iifile.is_null() {
            return 1;
        }
        unsafe {
            (*iifile).file = IIFILE_TIFF;
            (*iifile).nx = width as i32;
            (*iifile).ny = height as i32;
            (*iifile).nz = images.len() as i32;
            (*iifile).type_ = type_;
            (*iifile).mode = if rgb { MRC_MODE_RGB } else { mode_for(type_) };
            (*iifile).any_tiff_pix_size = any_tif_pixel;
            (*tif).iifile = iifile;
            (*tif).fp = core::ptr::null_mut();
            (*tif).bits_per_sample = bits;
            (*tif).photometric_interpretation = if rgb { 2 } else { 1 };
            (*tif).directory[1].value = width as i32;
            (*tif).directory[2].value = height as i32;
            (*tif).width = width as i32;
            (*tif).length = height as i32;
        }
        FILES
            .lock()
            .unwrap()
            .insert(tif as usize, FileData { images });
        0
    }

    /// Writes non-tiled TIFF pages with none/LZW/ZIP compression and no
    /// IMOD-specific tags. The caller retains command-level selection,
    /// scaling, and file-lifetime behavior and uses the parity writer for all
    /// other cases.
    pub fn write_stack(
        filename: &str,
        width: i32,
        height: i32,
        mode: i32,
        compression: i32,
        quality: i32,
        images: &[Vec<u8>],
    ) -> Result<(), String> {
        use tiff::encoder::{Compression, DeflateLevel, TiffEncoder, colortype};

        if width <= 0 || height <= 0 || images.is_empty() {
            return Err("Rust TIFF writer requires at least one non-empty image".into());
        }
        let row_bytes = match mode {
            MRC_MODE_BYTE => width as usize,
            MRC_MODE_SHORT | MRC_MODE_USHORT => 2 * width as usize,
            MRC_MODE_FLOAT => 4 * width as usize,
            MRC_MODE_RGB => 3 * width as usize,
            _ => return Err(format!("Rust TIFF writer does not support MRC mode {mode}")),
        };
        let file = File::create(filename)
            .map_err(|error| format!("Rust TIFF writer could not create {filename}: {error}"))?;
        let compression = match compression {
            IICOMPRESSION_NONE => Compression::Uncompressed,
            IICOMPRESSION_LZW => Compression::Lzw,
            IICOMPRESSION_ZIP => Compression::Deflate(if quality <= 3 {
                DeflateLevel::Fast
            } else if quality >= 8 {
                DeflateLevel::Best
            } else {
                DeflateLevel::Balanced
            }),
            other => {
                return Err(format!(
                    "Rust TIFF writer does not support compression {other}"
                ));
            }
        };
        let mut encoder = TiffEncoder::new(file).map_err(|error| {
            format!("Rust TIFF writer could not initialize {filename}: {error}")
        })?;
        encoder = encoder.with_compression(compression);
        for image in images {
            if image.len() != row_bytes * height as usize {
                return Err("Rust TIFF writer received an invalid image buffer".into());
            }
            // mrc2tif's source/libtiff boundary writes MRC's bottom-up rows
            // as conventional top-down TIFF rows.
            let mut oriented = image.clone();
            flip_rows(
                &mut oriented,
                width as usize,
                height as usize,
                row_bytes / width as usize,
            );
            match mode {
                MRC_MODE_BYTE => {
                    encoder.write_image::<colortype::Gray8>(width as u32, height as u32, &oriented)
                }
                MRC_MODE_SHORT => {
                    let values = oriented
                        .chunks_exact(2)
                        .map(|value| i16::from_ne_bytes([value[0], value[1]]))
                        .collect::<Vec<_>>();
                    encoder.write_image::<colortype::GrayI16>(width as u32, height as u32, &values)
                }
                MRC_MODE_USHORT => {
                    let values = oriented
                        .chunks_exact(2)
                        .map(|value| u16::from_ne_bytes([value[0], value[1]]))
                        .collect::<Vec<_>>();
                    encoder.write_image::<colortype::Gray16>(width as u32, height as u32, &values)
                }
                MRC_MODE_FLOAT => {
                    let values = oriented
                        .chunks_exact(4)
                        .map(|value| f32::from_ne_bytes([value[0], value[1], value[2], value[3]]))
                        .collect::<Vec<_>>();
                    encoder.write_image::<colortype::Gray32Float>(
                        width as u32,
                        height as u32,
                        &values,
                    )
                }
                MRC_MODE_RGB => {
                    encoder.write_image::<colortype::RGB8>(width as u32, height as u32, &oriented)
                }
                _ => unreachable!(),
            }
            .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
        }
        Ok(())
    }

    /// Single-page pointer adapter for the current C-shaped mrc2tif buffer.
    pub unsafe fn write_image(
        filename: &str,
        width: i32,
        height: i32,
        mode: i32,
        compression: i32,
        quality: i32,
        data: *const u8,
        byte_len: usize,
    ) -> Result<(), String> {
        if data.is_null() {
            return Err("Rust TIFF writer requires a non-null image buffer".into());
        }
        let image = unsafe { std::slice::from_raw_parts(data, byte_len) }.to_vec();
        write_stack(
            filename,
            width,
            height,
            mode,
            compression,
            quality,
            &[image],
        )
    }

    fn mode_for(type_: i32) -> i32 {
        match type_ {
            IITYPE_SHORT => MRC_MODE_SHORT,
            IITYPE_USHORT => MRC_MODE_USHORT,
            IITYPE_FLOAT | IITYPE_INT | IITYPE_UINT => MRC_MODE_FLOAT,
            _ => MRC_MODE_BYTE,
        }
    }

    pub unsafe fn read_section(tif: *mut TfInfo, section: i32) -> Option<*mut u8> {
        let files = FILES.lock().unwrap();
        let data = files
            .get(&(tif as usize))?
            .images
            .get(section.max(0) as usize)?
            .data
            .as_slice();
        let output = unsafe { libc::malloc(data.len()).cast::<u8>() };
        if output.is_null() {
            return Some(core::ptr::null_mut());
        }
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), output, data.len()) };
        unsafe { (*tif).data = output };
        Some(output)
    }

    pub fn contains(tif: *mut TfInfo) -> bool {
        FILES.lock().unwrap().contains_key(&(tif as usize))
    }

    pub unsafe fn close_file(tif: *mut TfInfo) -> bool {
        FILES.lock().unwrap().remove(&(tif as usize)).is_some()
    }

    #[cfg(test)]
    mod tests {
        use super::{IICOMPRESSION_NONE, write_image, write_stack};
        use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, MRC_MODE_RGB, MRC_MODE_SHORT};
        use std::fs::File;
        use tiff::decoder::{Decoder, DecodingResult};

        #[test]
        fn writer_keeps_mrc_row_orientation_for_short_and_rgb_images() {
            let stem = format!("imod-rs-rust-tiff-writer-{}", std::process::id());
            let short_path = std::env::temp_dir().join(format!("{stem}-short.tif"));
            let rgb_path = std::env::temp_dir().join(format!("{stem}-rgb.tif"));
            let shorts = [1_i16, 2, 3, 4];
            unsafe {
                write_image(
                    short_path.to_str().unwrap(),
                    2,
                    2,
                    MRC_MODE_SHORT,
                    IICOMPRESSION_NONE,
                    -1,
                    shorts.as_ptr().cast(),
                    core::mem::size_of_val(&shorts),
                )
                .unwrap();
            }
            let mut decoder = Decoder::new(File::open(&short_path).unwrap()).unwrap();
            match decoder.read_image().unwrap() {
                DecodingResult::I16(values) => assert_eq!(values, vec![3, 4, 1, 2]),
                _ => panic!("Rust TIFF writer did not emit signed 16-bit grayscale"),
            }

            let rgb = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            unsafe {
                write_image(
                    rgb_path.to_str().unwrap(),
                    2,
                    2,
                    MRC_MODE_RGB,
                    IICOMPRESSION_NONE,
                    -1,
                    rgb.as_ptr(),
                    rgb.len(),
                )
                .unwrap();
            }
            let mut decoder = Decoder::new(File::open(&rgb_path).unwrap()).unwrap();
            match decoder.read_image().unwrap() {
                DecodingResult::U8(values) => {
                    assert_eq!(values, vec![7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]);
                }
                _ => panic!("Rust TIFF writer did not emit RGB8"),
            }
            std::fs::remove_file(short_path).unwrap();
            std::fs::remove_file(rgb_path).unwrap();
        }

        #[test]
        fn writer_chains_stack_pages_in_mrc_order() {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-rust-tiff-stack-{}.tif",
                std::process::id()
            ));
            write_stack(
                path.to_str().unwrap(),
                2,
                2,
                MRC_MODE_BYTE,
                IICOMPRESSION_NONE,
                -1,
                &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            )
            .unwrap();
            let mut decoder = Decoder::new(File::open(&path).unwrap()).unwrap();
            match decoder.read_image().unwrap() {
                DecodingResult::U8(values) => assert_eq!(values, vec![3, 4, 1, 2]),
                _ => panic!("Rust TIFF writer did not emit byte grayscale"),
            }
            assert!(decoder.more_images());
            decoder.next_image().unwrap();
            match decoder.read_image().unwrap() {
                DecodingResult::U8(values) => assert_eq!(values, vec![7, 8, 5, 6]),
                _ => panic!("Rust TIFF writer did not emit second byte grayscale page"),
            }
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(feature = "rust-tiff")]
pub use implementation::{close_file, contains, open_file, read_section};

#[cfg(feature = "rust-tiff")]
pub use implementation::{write_image, write_stack};

#[cfg(not(feature = "rust-tiff"))]
pub unsafe fn open_file(_filename: *mut i8, _tif: *mut TfInfo, _any_tif_pixel: i32) -> i32 {
    1
}

#[cfg(not(feature = "rust-tiff"))]
pub unsafe fn read_section(_tif: *mut TfInfo, _section: i32) -> Option<*mut u8> {
    None
}

#[cfg(not(feature = "rust-tiff"))]
pub fn contains(_tif: *mut TfInfo) -> bool {
    false
}

#[cfg(not(feature = "rust-tiff"))]
pub unsafe fn close_file(_tif: *mut TfInfo) -> bool {
    false
}

#[cfg(not(feature = "rust-tiff"))]
pub unsafe fn write_image(
    _filename: &str,
    _width: i32,
    _height: i32,
    _mode: i32,
    _compression: i32,
    _quality: i32,
    _data: *const u8,
    _byte_len: usize,
) -> Result<(), String> {
    Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
}

#[cfg(not(feature = "rust-tiff"))]
pub fn write_stack(
    _filename: &str,
    _width: i32,
    _height: i32,
    _mode: i32,
    _compression: i32,
    _quality: i32,
    _images: &[Vec<u8>],
) -> Result<(), String> {
    Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
}
