//! Experimental Rust `tiff`-crate reader for the legacy tif2mrc boundary.
//!
//! It decodes supported pages eagerly, then hands back the same owned section
//! buffers that `tiff.c` returns to tif2mrc.  The default path never enters
//! this module.

use crate::imod::mrc::tiff::TfInfo;

#[cfg(feature = "rust-tiff")]
mod implementation {
    use super::TfInfo;
    use crate::imod::libiimod::iimage::{
        IIFILE_TIFF, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_INT, IITYPE_SHORT, IITYPE_UBYTE,
        IITYPE_UINT, IITYPE_USHORT, ImodImageFile,
    };
    use crate::imod::libiimod::iitif::{IICOMPRESSION_LZW, IICOMPRESSION_NONE, IICOMPRESSION_ZIP};
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    };
    use std::fs::File;
    use std::io::Cursor;
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;

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
        unsafe {
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
        resolutions: &[i32],
    ) -> Result<(), String> {
        use tiff::encoder::{Compression, DeflateLevel, Rational, TiffEncoder, colortype};
        use tiff::tags::ResolutionUnit;

        if width <= 0 || height <= 0 || images.is_empty() || images.len() != resolutions.len() {
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
        for (image, &resolution) in images.iter().zip(resolutions) {
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
                    let mut page = encoder
                        .new_image::<colortype::Gray8>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(
                            if resolution < 0 {
                                ResolutionUnit::Inch
                            } else {
                                ResolutionUnit::Centimeter
                            },
                            Rational {
                                n: resolution.unsigned_abs(),
                                d: 1,
                            },
                        );
                    }
                    page.write_data(&oriented)
                }
                MRC_MODE_SHORT => {
                    let values = oriented
                        .chunks_exact(2)
                        .map(|value| i16::from_ne_bytes([value[0], value[1]]))
                        .collect::<Vec<_>>();
                    let mut page = encoder
                        .new_image::<colortype::GrayI16>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(
                            if resolution < 0 {
                                ResolutionUnit::Inch
                            } else {
                                ResolutionUnit::Centimeter
                            },
                            Rational {
                                n: resolution.unsigned_abs(),
                                d: 1,
                            },
                        );
                    }
                    page.write_data(&values)
                }
                MRC_MODE_USHORT => {
                    let values = oriented
                        .chunks_exact(2)
                        .map(|value| u16::from_ne_bytes([value[0], value[1]]))
                        .collect::<Vec<_>>();
                    let mut page = encoder
                        .new_image::<colortype::Gray16>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(
                            if resolution < 0 {
                                ResolutionUnit::Inch
                            } else {
                                ResolutionUnit::Centimeter
                            },
                            Rational {
                                n: resolution.unsigned_abs(),
                                d: 1,
                            },
                        );
                    }
                    page.write_data(&values)
                }
                MRC_MODE_FLOAT => {
                    let values = oriented
                        .chunks_exact(4)
                        .map(|value| f32::from_ne_bytes([value[0], value[1], value[2], value[3]]))
                        .collect::<Vec<_>>();
                    let mut page = encoder
                        .new_image::<colortype::Gray32Float>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(
                            if resolution < 0 {
                                ResolutionUnit::Inch
                            } else {
                                ResolutionUnit::Centimeter
                            },
                            Rational {
                                n: resolution.unsigned_abs(),
                                d: 1,
                            },
                        );
                    }
                    page.write_data(&values)
                }
                MRC_MODE_RGB => {
                    let mut page = encoder
                        .new_image::<colortype::RGB8>(width as u32, height as u32)
                        .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
                    if resolution != 0 {
                        page.resolution(
                            if resolution < 0 {
                                ResolutionUnit::Inch
                            } else {
                                ResolutionUnit::Centimeter
                            },
                            Rational {
                                n: resolution.unsigned_abs(),
                                d: 1,
                            },
                        );
                    }
                    page.write_data(&oriented)
                }
                _ => unreachable!(),
            }
            .map_err(|error| format!("Rust TIFF writer failed: {error}"))?;
        }
        Ok(())
    }

    /// Single-page adapter over [`write_stack`] for one mrc2tif slice.
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
        let image = data.to_vec();
        write_stack(
            filename,
            width,
            height,
            mode,
            compression,
            quality,
            &[image],
            &[resolution],
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
        use super::{IICOMPRESSION_NONE, write_image, write_stack};
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
            write_stack(
                path.to_str().unwrap(),
                2,
                2,
                MRC_MODE_BYTE,
                IICOMPRESSION_NONE,
                -1,
                &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
                &[-300, 20_000_000],
            )
            .unwrap();
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
pub use implementation::{write_image, write_stack};

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

#[cfg(not(feature = "rust-tiff"))]
pub fn write_stack(
    _filename: &str,
    _width: i32,
    _height: i32,
    _mode: i32,
    _compression: i32,
    _quality: i32,
    _images: &[Vec<u8>],
    _resolutions: &[i32],
) -> Result<(), String> {
    Err("Rust TIFF backend requires Cargo feature rust-tiff".into())
}
