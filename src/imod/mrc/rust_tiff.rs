//! Experimental Rust `tiff`-crate reader for the legacy tif2mrc boundary.
//!
//! It decodes supported pages eagerly, then hands back the same owned section
//! buffers that `tiff.c` returns to tif2mrc.  Its writer takes one page at a
//! time.  The default path never enters this module.

use crate::imod::mrc::tiff::TfInfo;

#[cfg(feature = "rust-tiff")]
mod implementation {
    use super::TfInfo;
    use crate::imod::libiimod::iimage::{IIFILE_TIFF, ImageDataType, ImodImageFile};
    use crate::imod::libiimod::iitif::{IICOMPRESSION_LZW, IICOMPRESSION_NONE, IICOMPRESSION_ZIP};
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    };
    use std::fs::File;
    use std::io::Cursor;
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;

    fn bytes(result: DecodingResult) -> (Vec<u8>, i32, ImageDataType, i32) {
        match result {
            DecodingResult::U8(values) => (
                values,
                8,
                ImageDataType::UnsignedByte,
                MRC_MODE_BYTE,
            ),
            DecodingResult::I8(values) => (
                values.into_iter().map(|value| value as u8).collect(),
                8,
                ImageDataType::Byte,
                MRC_MODE_BYTE,
            ),
            DecodingResult::U16(values) => (
                {
                    // Sized once up front; the same `to_ne_bytes` sequence
                    // the former `flat_map(..).collect()` produced.
                    let mut data = Vec::with_capacity(values.len() * 2);
                    for value in values {
                        data.extend_from_slice(&value.to_ne_bytes());
                    }
                    data
                },
                16,
                ImageDataType::UnsignedShort,
                MRC_MODE_USHORT,
            ),
            DecodingResult::I16(values) => (
                {
                    // Sized once up front; the same `to_ne_bytes` sequence
                    // the former `flat_map(..).collect()` produced.
                    let mut data = Vec::with_capacity(values.len() * 2);
                    for value in values {
                        data.extend_from_slice(&value.to_ne_bytes());
                    }
                    data
                },
                16,
                ImageDataType::Short,
                MRC_MODE_SHORT,
            ),
            DecodingResult::U32(values) => (
                {
                    // Sized once up front; the same `to_ne_bytes` sequence
                    // the former `flat_map(..).collect()` produced.
                    let mut data = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        data.extend_from_slice(&value.to_ne_bytes());
                    }
                    data
                },
                32,
                ImageDataType::UnsignedInt,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::I32(values) => (
                {
                    // Sized once up front; the same `to_ne_bytes` sequence
                    // the former `flat_map(..).collect()` produced.
                    let mut data = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        data.extend_from_slice(&value.to_ne_bytes());
                    }
                    data
                },
                32,
                ImageDataType::Int,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::F32(values) => (
                {
                    // Sized once up front; the same `to_ne_bytes` sequence
                    // the former `flat_map(..).collect()` produced.
                    let mut data = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        data.extend_from_slice(&value.to_ne_bytes());
                    }
                    data
                },
                32,
                ImageDataType::Float,
                MRC_MODE_FLOAT,
            ),
            DecodingResult::F16(_)
            | DecodingResult::F64(_)
            | DecodingResult::U64(_)
            // Unsupported sample formats: the caller rejects these on the
            // empty buffer, so the reported type is never read.
            | DecodingResult::I64(_) => (Vec::new(), 0, ImageDataType::UnsignedByte, 0),
        }
    }

    fn flip_rows(data: &mut [u8], width: usize, height: usize, bytes_per_pixel: usize) {
        let row = width * bytes_per_pixel;
        for y in 0..height / 2 {
            let other = height - y - 1;
            // Exchanging the two rows as slices is the same set of byte swaps
            // the per-byte loop made -- `y < other` always, so the two never
            // name the same row -- and it is the idiom the legacy reader
            // already uses for this flip (`mrc/tiff.rs`, `tiff.c:224-236`).
            let (head, tail) = data.split_at_mut(other * row);
            head[y * row..y * row + row].swap_with_slice(&mut tail[..row]);
        }
    }

    pub fn open_file(filename: &[u8], tif: &mut TfInfo, any_tif_pixel: i32) -> i32 {
        let Ok(path) = std::str::from_utf8(filename) else {
            return 1;
        };
        let Ok(mut file_bytes) = std::fs::read(path) else {
            return 1;
        };
        // `tiff` deliberately exposes Palette in ColorType but refuses to
        // construct a readout for it because it does not expand ColorMap.
        // `tif2mrc`'s libtiff boundary does the opposite: it keeps the stored
        // one-byte palette indices (the source records photometric 1, so its
        // later color-map expansion is not entered).  Normalize only that
        // in-memory tag to MINISBLACK.  This leaves the crate responsible for
        // every strip/tile, compression, predictor, and endian decode; it
        // neither changes the input file nor interprets ColorMap values.
        if file_bytes.len() >= 8 && (&file_bytes[..2] == b"II" || &file_bytes[..2] == b"MM") {
            let little_endian = &file_bytes[..2] == b"II";
            let read_u16 = |data: &[u8], offset: usize| -> Option<u16> {
                let bytes: [u8; 2] = data.get(offset..offset + 2)?.try_into().ok()?;
                Some(if little_endian {
                    u16::from_le_bytes(bytes)
                } else {
                    u16::from_be_bytes(bytes)
                })
            };
            let read_u32 = |data: &[u8], offset: usize| -> Option<u32> {
                let bytes: [u8; 4] = data.get(offset..offset + 4)?.try_into().ok()?;
                Some(if little_endian {
                    u32::from_le_bytes(bytes)
                } else {
                    u32::from_be_bytes(bytes)
                })
            };
            let read_u64 = |data: &[u8], offset: usize| -> Option<u64> {
                let bytes: [u8; 8] = data.get(offset..offset + 8)?.try_into().ok()?;
                Some(if little_endian {
                    u64::from_le_bytes(bytes)
                } else {
                    u64::from_be_bytes(bytes)
                })
            };
            let Some(version) = read_u16(&file_bytes, 2) else {
                return 1;
            };
            let (mut ifd_offset, count_bytes, entry_bytes, next_bytes, value_offset) =
                if version == 42 {
                    let Some(offset) = read_u32(&file_bytes, 4) else {
                        return 1;
                    };
                    (offset as usize, 2_usize, 12_usize, 4_usize, 8_usize)
                } else if version == 43 {
                    if read_u16(&file_bytes, 4) != Some(8) || read_u16(&file_bytes, 6) != Some(0) {
                        return 1;
                    }
                    let Some(offset) =
                        read_u64(&file_bytes, 8).and_then(|value| usize::try_from(value).ok())
                    else {
                        return 1;
                    };
                    (offset, 8_usize, 20_usize, 8_usize, 12_usize)
                } else {
                    return 1;
                };
            let mut seen_ifds = std::collections::HashSet::new();
            while ifd_offset != 0 {
                if !seen_ifds.insert(ifd_offset) {
                    return 1;
                }
                let entry_count = if count_bytes == 2 {
                    let Some(value) = read_u16(&file_bytes, ifd_offset) else {
                        return 1;
                    };
                    value as usize
                } else {
                    let Some(value) = read_u64(&file_bytes, ifd_offset)
                        .and_then(|value| usize::try_from(value).ok())
                    else {
                        return 1;
                    };
                    value
                };
                let Some(entries_end) = ifd_offset
                    .checked_add(count_bytes)
                    .and_then(|value| value.checked_add(entry_count.checked_mul(entry_bytes)?))
                    .and_then(|value| value.checked_add(next_bytes))
                else {
                    return 1;
                };
                if entries_end > file_bytes.len() {
                    return 1;
                }
                for index in 0..entry_count {
                    let entry = ifd_offset + count_bytes + index * entry_bytes;
                    if read_u16(&file_bytes, entry) == Some(262)
                        && read_u16(&file_bytes, entry + 2) == Some(3)
                        && if count_bytes == 2 {
                            read_u32(&file_bytes, entry + 4) == Some(1)
                        } else {
                            read_u64(&file_bytes, entry + 4) == Some(1)
                        }
                        && read_u16(&file_bytes, entry + value_offset) == Some(3)
                    {
                        let value = if little_endian {
                            1_u16.to_le_bytes()
                        } else {
                            1_u16.to_be_bytes()
                        };
                        file_bytes[entry + value_offset..entry + value_offset + 2]
                            .copy_from_slice(&value);
                    }
                }
                let next_ifd = if next_bytes == 4 {
                    read_u32(&file_bytes, entries_end - next_bytes)
                        .and_then(|value| usize::try_from(value).ok())
                } else {
                    read_u64(&file_bytes, entries_end - next_bytes)
                        .and_then(|value| usize::try_from(value).ok())
                };
                let Some(next_ifd) = next_ifd else {
                    return 1;
                };
                ifd_offset = next_ifd;
            }
        }
        let Ok(mut decoder) = Decoder::new(Cursor::new(file_bytes)) else {
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
            // IMOD's libtiff reader drops alpha samples. Indexed palette
            // tags were normalized to their stored byte indices above.
            let rgb = matches!(color, ColorType::RGB(8) | ColorType::RGBA(8));
            let discard_alpha = matches!(color, ColorType::RGBA(8) | ColorType::GrayA(8));
            let palette = matches!(color, ColorType::Palette(8));
            // `iiTIFFCheck` explicitly accepts one-plane, unsigned 4-bit
            // grayscale and marks it `PACKED_4BIT_MODE`; `tiffReadSection`
            // then expands its two nibbles to two byte samples.  The Rust
            // crate correctly decodes the enclosing TIFF layout but retains
            // those nibbles packed in its U8 result, so do the one source
            // mandated expansion at this boundary.
            let packed_four_bit = matches!(color, ColorType::Gray(4));
            let planar_separate = decoder
                .get_tag_unsigned::<u16>(Tag::PlanarConfiguration)
                .is_ok_and(|value| value == 2);
            let fill_order_lsb = decoder
                .get_tag_unsigned::<u16>(Tag::FillOrder)
                .is_ok_and(|value| value == 2);
            let min_is_white = decoder
                .get_tag_unsigned::<u16>(Tag::PhotometricInterpretation)
                .is_ok_and(|value| value == 0);
            if !rgb
                && !palette
                && !packed_four_bit
                && !matches!(color, ColorType::Gray(8 | 16 | 32) | ColorType::GrayA(8))
            {
                return 1;
            }
            // `Decoder::read_image` intentionally reads only the first plane
            // of a planar-separate image.  `read_image_to_buffer` is the
            // crate's all-plane API; it returns R, G, B (and alpha, where
            // present) as consecutive planes, which IMOD's RGB path needs
            // interleaved before it drops alpha and flips rows.
            let mut result = DecodingResult::U8(Vec::new());
            let Ok(_) = decoder.read_image_to_buffer(&mut result) else {
                return 1;
            };
            let (mut data, bits, type_, mode) = bytes(result);
            if data.is_empty() {
                return 1;
            }
            if packed_four_bit {
                let pixels = width as usize * height as usize;
                if data.len() != pixels.div_ceil(2) {
                    return 1;
                }
                let packed = data;
                data = Vec::with_capacity(pixels);
                for byte in packed {
                    if fill_order_lsb {
                        // TIFFReadEncodedStrip normalizes FillOrder 2 by
                        // reversing each source byte.  IMOD then swaps its
                        // first/second nibble maps for LSB order
                        // (`iitif.c:1124-1131`), hence low then high after
                        // the per-byte reversal.
                        let byte = byte.reverse_bits();
                        data.push(byte & 15);
                        data.push(byte >> 4);
                    } else {
                        data.push(byte >> 4);
                        data.push(byte & 15);
                    }
                }
                data.truncate(pixels);
            }
            if planar_separate && rgb {
                let samples = if discard_alpha { 4 } else { 3 };
                let pixels = width as usize * height as usize;
                if bits != 8 || data.len() != pixels * samples {
                    return 1;
                }
                let planar_data = data;
                data = Vec::with_capacity(planar_data.len());
                for pixel in 0..pixels {
                    for sample in 0..samples {
                        data.push(planar_data[pixel + sample * pixels]);
                    }
                }
            }
            if discard_alpha {
                let source_samples = if rgb { 4 } else { 2 };
                let kept_samples = source_samples - 1;
                if bits != 8 || data.len() != width as usize * height as usize * source_samples {
                    return 1;
                }
                // Compact in place: pixel `p`'s kept samples move from
                // `p * source_samples` to `p * kept_samples`, never forward,
                // so each source is read before anything overwrites it.  The
                // result is the byte sequence the former `flat_map(..)
                // .collect()` built in a second allocation.
                let pixels = data.len() / source_samples;
                for pixel in 0..pixels {
                    let from = pixel * source_samples;
                    data.copy_within(from..from + kept_samples, pixel * kept_samples);
                }
                data.truncate(pixels * kept_samples);
            }
            // The Rust TIFF decoder normalizes MINISWHITE samples, while
            // IMOD's existing libtiff path copies these samples unchanged.
            // Undo that normalization so this alternative reader has the
            // established tif2mrc observable behavior.
            if min_is_white && !rgb && !palette {
                match bits {
                    8 if packed_four_bit => data.iter_mut().for_each(|value| *value = 15 - *value),
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
            images.push(data);
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
        let mut iifile = ImodImageFile::default();
        iifile.xscale = 1.0;
        iifile.yscale = 1.0;
        iifile.zscale = 1.0;
        iifile.slope = 1.0;
        iifile.smax = 255.0;
        iifile.axis = 3;
        iifile.urx = -1;
        iifile.ury = -1;
        iifile.urz = -1;
        iifile.rms = -1.0;
        iifile.last_written_z = -1;
        iifile.packed4bits = 0;
        iifile.half_floats = 0;
        iifile.adoc_index = -1;
        iifile.global_adoc_index = -1;
        iifile.hdf_compression = -1;
        iifile.tiff_compression = 1;
        iifile.file = IIFILE_TIFF;
        iifile.nx = width as i32;
        iifile.ny = height as i32;
        iifile.nz = images.len() as i32;
        iifile.type_ = type_;
        iifile.mode = if rgb { MRC_MODE_RGB } else { mode_for(type_) };
        iifile.any_tiff_pix_size = any_tif_pixel;
        {
            (*tif).iifile = Some(iifile);
            // `tiff.c:262` always leaves `tiff->fp` an open stream, and
            // `tif2mrc.c:242` copies it out unconditionally, so this backend
            // has to provide one even though it decodes from its own in-memory
            // copy.  Previously `core::ptr::null_mut()`, which stopped
            // compiling when `Tf_info.fp` became `Option<ImodFile>`; leaving
            // it `None` compiles and then panics in `tif2mrc`.
            (*tif).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(path, "rb");
            if (*tif).fp.is_none() {
                return 1;
            }
            (*tif).bits_per_sample = bits;
            (*tif).photometric_interpretation = if rgb { 2 } else { 1 };
            (*tif).directory[1].value = width as i32;
            (*tif).directory[2].value = height as i32;
            (*tif).width = width as i32;
            (*tif).length = height as i32;
        }
        tif.decoded_pages = images;
        0
    }

    /// Writes non-tiled TIFF pages with none/LZW/ZIP compression and no
    /// IMOD-specific tags, one page per [`StackWriter::write_page`] call, so
    /// `mrc2tif -s` can hand over each section as it is produced
    /// (`mrc2tif.cpp:598-607` writes each section to the open TIFF the same
    /// way) instead of holding the whole volume until the end.  The caller
    /// retains command-level selection, scaling, and file-lifetime behavior
    /// and uses the parity writer for all other cases.
    ///
    /// Nothing is validated or created until the first page, so a run that
    /// never delivers a page creates no file, and the first page runs the
    /// same checks in the same order (size, mode, create, compression,
    /// encoder header, buffer length) that the former whole-stack writer ran
    /// before its first page.  The encoder receives the identical sequence
    /// of `new_image`/`resolution`/`write_data` calls with identical
    /// arguments either way; only *when* they happen relative to reading the
    /// input moved, and `TiffEncoder` output is a function of that call
    /// sequence alone.
    pub struct StackWriter {
        filename: String,
        width: i32,
        height: i32,
        mode: i32,
        compression: i32,
        quality: i32,
        row_bytes: usize,
        encoder: Option<tiff::encoder::TiffEncoder<File>>,
        pages: usize,
        /// Top-down copy of the current page, reused across pages. One of
        /// these is in use for a given writer, chosen by `mode`.
        oriented_bytes: Vec<u8>,
        oriented_i16: Vec<i16>,
        oriented_u16: Vec<u16>,
        oriented_f32: Vec<f32>,
    }

    /// Creates a [`StackWriter`]; no I/O happens until the first page.
    pub fn stack_writer(
        filename: &str,
        width: i32,
        height: i32,
        mode: i32,
        compression: i32,
        quality: i32,
    ) -> StackWriter {
        StackWriter {
            filename: filename.to_string(),
            width,
            height,
            mode,
            compression,
            quality,
            row_bytes: 0,
            encoder: None,
            pages: 0,
            oriented_bytes: Vec::new(),
            oriented_i16: Vec::new(),
            oriented_u16: Vec::new(),
            oriented_f32: Vec::new(),
        }
    }

    impl StackWriter {
        /// Encodes one MRC-orientation (bottom-up) page.
        pub fn write_page(&mut self, image: &[u8], resolution: i32) -> Result<(), String> {
            use tiff::encoder::{Compression, DeflateLevel, Rational, TiffEncoder, colortype};
            use tiff::tags::ResolutionUnit;

            let (width, height, mode) = (self.width, self.height, self.mode);
            if self.encoder.is_none() {
                if width <= 0 || height <= 0 {
                    return Err("Rust TIFF writer requires at least one non-empty image".into());
                }
                self.row_bytes = match mode {
                    MRC_MODE_BYTE => width as usize,
                    MRC_MODE_SHORT | MRC_MODE_USHORT => 2 * width as usize,
                    MRC_MODE_FLOAT => 4 * width as usize,
                    MRC_MODE_RGB => 3 * width as usize,
                    _ => return Err(format!("Rust TIFF writer does not support MRC mode {mode}")),
                };
                let filename = &self.filename;
                let file = File::create(filename).map_err(|error| {
                    format!("Rust TIFF writer could not create {filename}: {error}")
                })?;
                let quality = self.quality;
                let compression = match self.compression {
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
                let encoder = TiffEncoder::new(file).map_err(|error| {
                    format!("Rust TIFF writer could not initialize {filename}: {error}")
                })?;
                self.encoder = Some(encoder.with_compression(compression));
            }
            let row_bytes = self.row_bytes;
            let encoder = self.encoder.as_mut().unwrap();
            if image.len() != row_bytes * height as usize {
                return Err("Rust TIFF writer received an invalid image buffer".into());
            }
            // mrc2tif's source/libtiff boundary writes MRC's bottom-up rows
            // as conventional top-down TIFF rows.  Row `y` of the page is
            // input row `height - 1 - y`: exactly what cloning the image and
            // exchanging rows `y` and `height - 1 - y` produced (the middle
            // row of an odd height stays put either way), built here in one
            // pass into a buffer kept across pages.  The typed arms decode
            // each element with the same `from_ne_bytes` the former
            // whole-page transcode used, in the same order.
            let rows = image.chunks_exact(row_bytes).rev();
            let unit = if resolution < 0 {
                ResolutionUnit::Inch
            } else {
                ResolutionUnit::Centimeter
            };
            let rational = Rational {
                n: resolution.unsigned_abs(),
                d: 1,
            };
            match mode {
                MRC_MODE_BYTE => {
                    self.oriented_bytes.clear();
                    rows.for_each(|row| self.oriented_bytes.extend_from_slice(row));
                    let mut page = encoder
                        .new_image::<colortype::Gray8>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(unit, rational);
                    }
                    page.write_data(&self.oriented_bytes)
                }
                MRC_MODE_SHORT => {
                    self.oriented_i16.clear();
                    rows.for_each(|row| {
                        self.oriented_i16.extend(
                            row.chunks_exact(2)
                                .map(|value| i16::from_ne_bytes([value[0], value[1]])),
                        )
                    });
                    let mut page = encoder
                        .new_image::<colortype::GrayI16>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(unit, rational);
                    }
                    page.write_data(&self.oriented_i16)
                }
                MRC_MODE_USHORT => {
                    self.oriented_u16.clear();
                    rows.for_each(|row| {
                        self.oriented_u16.extend(
                            row.chunks_exact(2)
                                .map(|value| u16::from_ne_bytes([value[0], value[1]])),
                        )
                    });
                    let mut page = encoder
                        .new_image::<colortype::Gray16>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(unit, rational);
                    }
                    page.write_data(&self.oriented_u16)
                }
                MRC_MODE_FLOAT => {
                    // `from_ne_bytes` is `from_bits`: a bit copy, no float
                    // arithmetic, NaN payloads included.
                    self.oriented_f32.clear();
                    rows.for_each(|row| {
                        self.oriented_f32.extend(row.chunks_exact(4).map(|value| {
                            f32::from_ne_bytes([value[0], value[1], value[2], value[3]])
                        }))
                    });
                    let mut page = encoder
                        .new_image::<colortype::Gray32Float>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(unit, rational);
                    }
                    page.write_data(&self.oriented_f32)
                }
                MRC_MODE_RGB => {
                    self.oriented_bytes.clear();
                    rows.for_each(|row| self.oriented_bytes.extend_from_slice(row));
                    let mut page = encoder
                        .new_image::<colortype::RGB8>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(unit, rational);
                    }
                    page.write_data(&self.oriented_bytes)
                }
                _ => unreachable!(),
            }
            .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
            self.pages += 1;
            Ok(())
        }

        /// Ends the stack.  A writer that received no page reports the same
        /// error the whole-stack writer gave an empty stack, and has created
        /// no file.  Dropping the encoder closes the file, as returning from
        /// the whole-stack writer did.
        pub fn finish(self) -> Result<(), String> {
            if self.pages == 0 {
                return Err("Rust TIFF writer requires at least one non-empty image".into());
            }
            Ok(())
        }
    }

    /// Single-page adapter over [`StackWriter`] for one mrc2tif slice.
    pub fn write_image(
        filename: &str,
        width: i32,
        height: i32,
        mode: i32,
        compression: i32,
        quality: i32,
        resolution: i32,
        data: &[u8],
    ) -> Result<(), String> {
        let mut writer = stack_writer(filename, width, height, mode, compression, quality);
        writer.write_page(data, resolution)?;
        writer.finish()
    }

    fn mode_for(type_: ImageDataType) -> i32 {
        match type_ {
            ImageDataType::Short => MRC_MODE_SHORT,
            ImageDataType::UnsignedShort => MRC_MODE_USHORT,
            ImageDataType::Float | ImageDataType::Int | ImageDataType::UnsignedInt => {
                MRC_MODE_FLOAT
            }
            ImageDataType::UnsignedByte | ImageDataType::Byte => MRC_MODE_BYTE,
        }
    }

    pub fn read_section(tif: &mut TfInfo, section: i32) -> Option<Vec<u8>> {
        tif.decoded_pages.get(section.max(0) as usize).cloned()
    }

    pub fn contains(tif: &mut TfInfo) -> bool {
        !tif.decoded_pages.is_empty()
    }

    pub fn close_file(tif: &mut TfInfo) -> bool {
        !std::mem::take(&mut tif.decoded_pages).is_empty()
    }

    #[cfg(test)]
    mod tests {
        use super::{IICOMPRESSION_NONE, stack_writer, write_image};
        use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, MRC_MODE_RGB, MRC_MODE_SHORT};
        use std::fs::File;
        use tiff::decoder::{Decoder, DecodingResult};
        use tiff::tags::Tag;

        #[test]
        fn writer_keeps_mrc_row_orientation_for_short_and_rgb_images() {
            let stem = format!("imod-rs-rust-tiff-writer-{}", std::process::id());
            let short_path = std::env::temp_dir().join(format!("{stem}-short.tif"));
            let rgb_path = std::env::temp_dir().join(format!("{stem}-rgb.tif"));
            let shorts = [1_i16, 2, 3, 4];
            let short_bytes: Vec<u8> = shorts
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            write_image(
                short_path.to_str().unwrap(),
                2,
                2,
                MRC_MODE_SHORT,
                IICOMPRESSION_NONE,
                -1,
                0,
                &short_bytes,
            )
            .unwrap();
            let mut decoder = Decoder::new(File::open(&short_path).unwrap()).unwrap();
            match decoder.read_image().unwrap() {
                DecodingResult::I16(values) => assert_eq!(values, vec![3, 4, 1, 2]),
                _ => panic!("Rust TIFF writer did not emit signed 16-bit grayscale"),
            }

            let rgb = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            write_image(
                rgb_path.to_str().unwrap(),
                2,
                2,
                MRC_MODE_RGB,
                IICOMPRESSION_NONE,
                -1,
                0,
                &rgb,
            )
            .unwrap();
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
            let mut writer = stack_writer(
                path.to_str().unwrap(),
                2,
                2,
                MRC_MODE_BYTE,
                IICOMPRESSION_NONE,
                -1,
            );
            writer.write_page(&[1, 2, 3, 4], -300).unwrap();
            writer.write_page(&[5, 6, 7, 8], 20_000_000).unwrap();
            writer.finish().unwrap();
            let mut decoder = Decoder::new(File::open(&path).unwrap()).unwrap();
            assert_eq!(
                decoder
                    .get_tag(Tag::ResolutionUnit)
                    .unwrap()
                    .into_u16()
                    .unwrap(),
                2
            );
            assert_eq!(decoder.get_tag_u32_vec(Tag::XResolution).unwrap(), [300, 1]);
            match decoder.read_image().unwrap() {
                DecodingResult::U8(values) => assert_eq!(values, vec![3, 4, 1, 2]),
                _ => panic!("Rust TIFF writer did not emit byte grayscale"),
            }
            assert!(decoder.more_images());
            decoder.next_image().unwrap();
            assert_eq!(
                decoder
                    .get_tag(Tag::ResolutionUnit)
                    .unwrap()
                    .into_u16()
                    .unwrap(),
                3
            );
            assert_eq!(
                decoder.get_tag_u32_vec(Tag::XResolution).unwrap(),
                [20_000_000, 1]
            );
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
pub use implementation::{StackWriter, stack_writer, write_image};

#[cfg(not(feature = "rust-tiff"))]
pub fn open_file(_filename: &[u8], _tif: &mut TfInfo, _any_tif_pixel: i32) -> i32 {
    1
}

#[cfg(not(feature = "rust-tiff"))]
pub fn read_section(_tif: &mut TfInfo, _section: i32) -> Option<Vec<u8>> {
    None
}

#[cfg(not(feature = "rust-tiff"))]
pub fn contains(_tif: &mut TfInfo) -> bool {
    false
}

#[cfg(not(feature = "rust-tiff"))]
pub fn close_file(_tif: &mut TfInfo) -> bool {
    false
}

#[cfg(not(feature = "rust-tiff"))]
pub fn write_image(
    _filename: &str,
    _width: i32,
    _height: i32,
    _mode: i32,
    _compression: i32,
    _quality: i32,
    _resolution: i32,
    _data: &[u8],
) -> Result<(), String> {
    Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
}

/// Without the `rust-tiff` feature every page fails, as the whole-stack
/// writer did.
#[cfg(not(feature = "rust-tiff"))]
pub struct StackWriter;

#[cfg(not(feature = "rust-tiff"))]
pub fn stack_writer(
    _filename: &str,
    _width: i32,
    _height: i32,
    _mode: i32,
    _compression: i32,
    _quality: i32,
) -> StackWriter {
    StackWriter
}

#[cfg(not(feature = "rust-tiff"))]
impl StackWriter {
    pub fn write_page(&mut self, _image: &[u8], _resolution: i32) -> Result<(), String> {
        Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
    }

    pub fn finish(self) -> Result<(), String> {
        Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
    }
}
