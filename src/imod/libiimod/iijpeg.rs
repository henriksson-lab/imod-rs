//! Translation of `IMOD/libiimod/iijpeg.c`: JPEG-type `ImodImageFile`s.
//!
//! One Rust function per source function.  The source drives libjpeg; this
//! translation keeps **only the codec itself** behind a Rust crate boundary
//! (the deliberate Rust-native JPEG decision, `RUST_NATIVE_BACKENDS_PLAN.md`):
//!
//! - decoding: `zune-jpeg` stands in for `jpeg_read_header`,
//!   `jpeg_start_decompress` and `jpeg_read_scanlines`.  The crate decodes a
//!   whole image at once, so the scanlines are taken from its top-down output
//!   one at a time, in the order libjpeg hands them over, into the same
//!   one-line `tmpData` buffer the source reads into.  Everything around that
//!   - the file checks, the header setup, the load-info/sub-area/scaling/
//!   padding set-up through `iiMRCsetLoadInfo` and `iiInitReadSectionAny`, the
//!   bottom-up placement, and every conversion including RGB to gray
//!   (`iiProcessReadLine`, `mrcsec.c:740-782`) - is the translated source.
//! - encoding: the `image` crate's `JpegEncoder` stands in for
//!   `jpeg_create_compress` .. `jpeg_finish_compress`; the scanline order,
//!   quality and JFIF density follow `jpegWriteSection`.
//!
//! The libjpeg decoder and `zune-jpeg` are different IDCT/upsampling/colour
//! conversion implementations, so decoded pixel values can differ by small
//! amounts; `TOFIX.md` records the measured difference.
//!
//! Two libjpeg behaviours that are observable through IMOD are reproduced at
//! the boundary because they decide the exit status: libjpeg refuses an
//! `out_color_space` of `JCS_RGB` for CMYK/YCCK input at
//! `jpeg_start_decompress` (`JERR_CONVERSION_NOTIMPL`), and
//! `jpeg_finish_decompress` longjmps with `JERR_TOO_LITTLE_DATA` whenever the
//! reading loop stopped before the last scanline, which is what the source's
//! error block (`iijpeg.c:233-247`) turns into a normal return or an error.
//!
//! The source's static `sJerr` error manager and its `setjmp`/`longjmp` have
//! no counterpart: a crate error is returned, and the source's error block is
//! executed in place at each point where libjpeg could have longjmp'd.

use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_error};
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIERR_QUITTING, IIFILE_JPEG,
    IIFORMAT_LUMINANCE, IIFORMAT_RGB, IISTATE_READY, ImageDataType, ImodImageFile, LineProcData,
    MRSA_BYTE, MRSA_FLOAT, MRSA_NOPROC, MRSA_USHORT, ii_make_buffer_convert_if_float,
    ii_simple_fill_mrc_header, ii_simple_fill_mrc_header_callback,
};
use crate::imod::libiimod::iimrc::ii_mrc_set_load_info;
use crate::imod::libiimod::mrcfiles::{LoadInfo, MRC_MODE_BYTE, MRC_MODE_RGB, MrcHeader};
use crate::imod::libiimod::mrcsec::{ii_init_read_section_any, ii_process_read_line};
use image::ImageEncoder;
use image::codecs::jpeg::{JpegEncoder, PixelDensity, PixelDensityUnit};
use std::io::{Read as _, Seek as _, SeekFrom};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

/// `IIERR_MEMORY_ERR` (`iimage.h:73`).
const IIERR_MEMORY_ERR: i32 = 3;

/// The libjpeg header state `iiJPEGCheck` and `jpegReadSectionAny` get from
/// `jpeg_read_header`: rewind the file, read it, and parse the headers.  The
/// source keeps a `jpeg_decompress_struct` in `inFile->header` for this; the
/// crate decoder borrows the file bytes, so each call re-reads them, as the
/// source's `rewind` + `jpeg_stdio_src` + `jpeg_read_header` does.
///
/// `jpeg_read_header` equivalent: returns the file bytes, or the crate's
/// error text where libjpeg would have longjmp'd with a message.
fn jpeg_read_header(fp: &mut ImodFile) -> Result<Vec<u8>, String> {
    let mut raw = Vec::new();
    if let Err(error) = fp
        .seek(SeekFrom::Start(0))
        .and_then(|_| fp.read_to_end(&mut raw))
    {
        return Err(error.to_string());
    }
    Ok(raw)
}

/// Decoder options: libjpeg accepts dimensions up to `JPEG_MAX_DIMENSION`
/// (65500); the crate's default ceiling is 16384.
fn jpeg_decoder(raw: &[u8], out_color_space: ColorSpace) -> JpegDecoder<ZCursor<&[u8]>> {
    JpegDecoder::new_with_options(
        ZCursor::new(raw),
        DecoderOptions::default()
            .set_max_width(65535)
            .set_max_height(65535)
            .jpeg_set_out_colorspace(out_color_space),
    )
}

/// C `iiJPEGCheck` (`iijpeg.c:54`): check for and open a JPEG file.
pub fn ii_jpeg_check(in_file: &mut ImodImageFile) -> i32 {
    let mut buf = [0u8; 4];
    let Some(mut fp) = in_file.fp.clone() else {
        return IIERR_BAD_CALL;
    };

    /* Look for the magic numbers at start */
    let mut num_read = 0;
    if fp.seek(SeekFrom::Start(0)).is_ok() {
        while num_read < 4 {
            match fp.read(&mut buf[num_read..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => num_read += n,
            }
        }
    }
    if num_read < 4 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiJPEGCheck - Reading file {}\n",
                in_file.filename.as_deref().unwrap_or("")
            ),
        );
        return IIERR_IO_ERROR;
    }
    if buf[0] != 0xFF || buf[1] != 0xD8 || buf[2] != 0xFF {
        return IIERR_NOT_FORMAT;
    }

    /* Create object and read header to get properties.  The source's
    error block (`iijpeg.c:84-94`) is executed wherever libjpeg would
    have longjmp'd. */
    let header_error = |message: &str| {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("iiJPEGCheck: JPEG warning/error {}\n", message),
        );
        IIERR_IO_ERROR
    };
    let raw = match jpeg_read_header(&mut fp) {
        Ok(raw) => raw,
        Err(message) => return header_error(&message),
    };
    let mut cinfo = jpeg_decoder(&raw, ColorSpace::RGB);
    if let Err(error) = cinfo.decode_headers() {
        return header_error(&error.to_string());
    }
    let (Some(info), Some(jpeg_color_space)) = (cinfo.info(), cinfo.input_colorspace()) else {
        return header_error("headers not decoded");
    };

    in_file.nx = info.width as i32;
    in_file.ny = info.height as i32;
    in_file.nz = 1;
    in_file.type_ = ImageDataType::UnsignedByte;
    if jpeg_color_space == ColorSpace::Luma {
        in_file.format = IIFORMAT_LUMINANCE;
        in_file.mode = MRC_MODE_BYTE;
    } else if info.components > 2 {
        in_file.format = IIFORMAT_RGB;
        in_file.mode = MRC_MODE_RGB;
    } else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiJPEGCheck - JPEG colorspace not GRAYSCALE and # of components is {}\n",
                info.components
            ),
        );
        return IIERR_NOT_FORMAT;
    }

    /* Pixel size smaller than 400 nm cannot be encoded in the 16-bit integers of the JFIF
    so skip setting a pixel size */

    /* Set up the rest ofteh basic stuff and pointers for reading routines */
    in_file.amin = 0.;
    in_file.amax = 255.;
    in_file.amean = 128.;
    in_file.file = IIFILE_JPEG;
    in_file.clean_up = Some(jpeg_delete_callback);
    in_file.fill_mrc_header = Some(ii_simple_fill_mrc_header_callback);

    in_file.read_section = Some(jpeg_read_section);
    in_file.read_section_ushort = Some(jpeg_read_section_ushort);
    in_file.read_section_byte = Some(jpeg_read_section_byte);
    in_file.read_section_float = Some(jpeg_read_section_float);

    /* We need abort the object and redo header later because some callers might close
    and reopen the file */
    0
}

pub(crate) unsafe fn ii_jpeg_check_callback(in_file: *mut ImodImageFile) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    ii_jpeg_check(in_file)
}

/// C `jpegDelete` (`iijpeg.c:141`): when a file is being deleted, that is
/// when to destroy and free the (de)compression object.  The crate decoder
/// and encoder live only for the duration of one call, so there is nothing
/// left to destroy.
fn jpeg_delete(_in_file: &mut ImodImageFile) {}

unsafe fn jpeg_delete_callback(in_file: *mut ImodImageFile) {
    if let Some(in_file) = unsafe { in_file.as_mut() } {
        jpeg_delete(in_file)
    }
}

/// C `jpegReadSectionByte` (`iijpeg.c:156`).
unsafe fn jpeg_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { jpeg_read_section_any(in_file, buf, in_section, MRSA_BYTE) }
}

/// C `jpegReadSectionUShort` (`iijpeg.c:161`).
unsafe fn jpeg_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { jpeg_read_section_any(in_file, buf, in_section, MRSA_USHORT) }
}

/// C `jpegReadSectionFloat` (`iijpeg.c:166`).
unsafe fn jpeg_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { jpeg_read_section_any(in_file, buf, in_section, MRSA_FLOAT) }
}

/// C `jpegReadSection` (`iijpeg.c:171`).
unsafe fn jpeg_read_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    unsafe { jpeg_read_section_any(in_file, buf, in_section, MRSA_NOPROC) }
}

/// C `jpegReadSectionAny` (`iijpeg.c:179`): the main reading routine.
unsafe fn jpeg_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    type_: i32,
) -> i32 {
    let in_file = unsafe { &mut *in_file };
    let mut pix_size_buf: [i32; 4] = [0, 1, 4, 2];
    let mut hdata = MrcHeader::default();
    let mut d = LineProcData::default();
    let mut load_info = LoadInfo::default();
    let li = &mut load_info;
    let pad_left: i32;
    let pad_right: i32;
    let ny = in_file.ny;
    let mut y_end: i32;
    let mut err: i32 = -1;
    let mut jpeg_line: i32 = 0;
    let tval: i32;

    if in_file.read_section.is_none() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: jpegReadSectionAny - Trying to read from newly created JPEG file\n"
            ),
        );
        return IIERR_BAD_CALL;
    }

    if in_section != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: jpegReadSectionAny - Trying to read section {}; only section0 can be read from JPEG file\n",
                in_section
            ),
        );
        return IIERR_BAD_CALL;
    }

    let mut tmp_data: Vec<u8> = Vec::new();
    let tmp_size = in_file.nx as usize * if in_file.format == IIFORMAT_RGB { 3 } else { 1 };
    if tmp_data.try_reserve_exact(tmp_size).is_err() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: jpegReadSectionAny - Allocating line buffer for reading JPEG file\n"
            ),
        );
        return IIERR_MEMORY_ERR;
    }
    tmp_data.resize(tmp_size, 0);

    /* Initialize any variables used in the error block to make compiler happy */
    d.y_start = 0;

    /* The error block (`iijpeg.c:233-247`), run where libjpeg would longjmp.
    `jpeg_abort_decompress`, freeing `tmpData` and the map are drops here.

    `err` and `jpegLine` are non-volatile locals modified between `setjmp` and
    `longjmp`, so after the jump their values are indeterminate (C11
    7.13.2.1p3).  The reference binary (gcc -O2) reads them as they were at
    the `setjmp` call, `err = -1` and `jpegLine = 0`, so every sub-area that
    excludes row 0 fails with "too few scanlines" (`BUGS.md`, JPEG input).
    Fixed in translation (2026-09-26): the block receives the *live* `err`
    and `jpegLine`, which is what the test `jpegLine < d.yStart && !err` and
    the loop comment "it is safe to stop when the desired lines are obtained
    as the error is suppressed above" evidently intend -- stopping early
    after a clean read of the wanted lines is a normal return. */
    let error_block = |y_start: i32, mut err: i32, jpeg_line: i32, message: &str| -> i32 {
        if err != IIERR_QUITTING {
            err = if jpeg_line < y_start && err == 0 {
                0
            } else {
                IIERR_IO_ERROR
            };
            if err != 0 {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("jpegReadSectionAny: JPEG error - {}\n", message),
                );
            }
        }
        err
    };

    /* Reestablish the object, data source and header */
    let Some(mut fp) = in_file.fp.clone() else {
        return IIERR_BAD_CALL;
    };
    let raw = match jpeg_read_header(&mut fp) {
        Ok(raw) => raw,
        Err(message) => return error_block(d.y_start, err, jpeg_line, &message),
    };

    /* Set output properties if not grayscale */
    let out_color_space = if in_file.format == IIFORMAT_RGB {
        ColorSpace::RGB
    } else {
        ColorSpace::Luma
    };
    let mut cinfo = jpeg_decoder(&raw, out_color_space);
    if let Err(error) = cinfo.decode_headers() {
        return error_block(d.y_start, err, jpeg_line, &error.to_string());
    }

    /* Translate the information to a loadInfo to call common routines and set some
    type-dependent settings as in iimrc; setup MRC header too */
    ii_simple_fill_mrc_header(in_file, &mut hdata);
    ii_mrc_set_load_info(in_file, li);
    pad_left = 0.max(li.pad_left);
    pad_right = 0.max(li.pad_right);
    y_end = li.ymax;
    if type_ == MRSA_FLOAT || type_ == 0 {
        li.outmin = in_file.smin as i32;
        li.outmax = in_file.smax as i32;
    } else {
        li.outmin = 0;
        li.outmax = if type_ == MRSA_USHORT { 65535 } else { 255 };
    }

    /* Initialize members of data structure */
    d.type_ = type_;
    d.read_y = 0;
    d.cz = 0;
    d.swapped = 0;
    err = unsafe {
        ii_init_read_section_any(&hdata, li, buf, &mut d, &mut y_end, "jpegReadSectionAny")
    };
    if err != 0 {
        return err;
    }

    /* Fill in some missing pieces and adjust pointers/indexes for inversion
    No need to invert yStart/yEnd because they are used as limits for line numbers
    counting from the true bottom */
    d.x_dimension = d.xsize + pad_left + pad_right;
    d.need_data = 1;
    pix_size_buf[0] = d.pix_size;
    tval = (y_end - d.y_start) * d.x_dimension;
    d.pix_index = d.pix_index.wrapping_add(tval as u32);
    // `d.bufp`, `d.usbufp` and `d.fbufp` all advance by `tval` pixels of their
    // own type; the Rust `LineProcData` keeps the one byte offset they share.
    d.bufp_offset += (pix_size_buf[type_ as usize] * tval) as isize;
    d.delta_y_sign = -1;

    /* Loop from the top of the image; it is safe to stop when the desired lines are
    obtained as the error is suppressed above */
    jpeg_line = ny - 1;

    // `jpeg_start_decompress`: libjpeg cannot convert CMYK or YCCK to RGB
    // (`jinit_color_deconverter`, `JERR_CONVERSION_NOTIMPL`).
    if matches!(
        cinfo.input_colorspace(),
        Some(ColorSpace::CMYK | ColorSpace::YCCK)
    ) {
        return error_block(
            d.y_start,
            err,
            jpeg_line,
            "Unsupported color conversion request",
        );
    }
    let decoded = match cinfo.decode() {
        Ok(decoded) => decoded,
        Err(error) => return error_block(d.y_start, err, jpeg_line, &error.to_string()),
    };
    let mut output_scanline: usize = 0;
    err = 0;
    while jpeg_line >= 0 && jpeg_line >= d.y_start {
        // `jpeg_read_scanlines(cinfoPtr, &tmpData, 1)`
        let start = output_scanline * tmp_size;
        tmp_data.copy_from_slice(&decoded[start..start + tmp_size]);
        output_scanline += 1;
        if jpeg_line <= y_end {
            d.bdata = unsafe { tmp_data.as_mut_ptr().add((d.x_start * d.pix_size) as usize) };
            err = unsafe {
                ii_process_read_line(
                    &hdata,
                    li,
                    &mut d,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            if err != 0 {
                break;
            }
        }
        jpeg_line -= 1;
    }

    // `jpeg_finish_decompress` longjmps with `JERR_TOO_LITTLE_DATA` when fewer
    // than all the scanlines were read.
    if output_scanline < ny as usize {
        return error_block(
            d.y_start,
            err,
            jpeg_line,
            "Application transferred too few scanlines",
        );
    }
    err
}

/// C `jpegOpenNew` (`iijpeg.c:313`): open a new JPEG file: just setup
/// function pointers and allocate compression object.
pub fn jpeg_open_new(in_file: &mut ImodImageFile) -> i32 {
    in_file.state = IISTATE_READY;
    in_file.clean_up = Some(jpeg_delete_callback);
    in_file.fill_mrc_header = Some(ii_simple_fill_mrc_header_callback);
    in_file.write_section = Some(ii_jpeg_write_section);
    in_file.write_section_float = Some(ii_jpeg_write_section_float);
    in_file.fp = in_file
        .filename
        .as_deref()
        .and_then(|name| ImodFile::open(name, "wb"));
    if in_file.fp.is_none() {
        return IIERR_IO_ERROR;
    }
    // The compression object is created by the encoder inside
    // `jpegWriteSection`; there is nothing to allocate here.
    0
}

pub(crate) unsafe fn jpeg_open_new_callback(in_file: *mut ImodImageFile) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    jpeg_open_new(in_file)
}

/// C `iiJpegWriteSection` (`iijpeg.c:334`).
unsafe fn ii_jpeg_write_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    unsafe { ii_jpeg_write_section_any(in_file, buf, in_section, 0) }
}

/// C `iiJpegWriteSectionFloat` (`iijpeg.c:339`).
unsafe fn ii_jpeg_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { ii_jpeg_write_section_any(in_file, buf, in_section, 1) }
}

/// C `iiJpegWriteSectionAny` (`iijpeg.c:348`): wrapper that handles the
/// iimage calls in, checks for possible bad things in a generic call, and
/// sets quality and resolution from environment variables.
unsafe fn ii_jpeg_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    if_float: i32,
) -> i32 {
    let in_file = unsafe { &mut *in_file };
    let mut quality: i32 = -1;
    let mut resolution: i32 = 0;
    let mut inverted = false;
    let mut err: i32 = 0;
    // C `atoi`: leading blanks, an optional sign, then digits.
    let atoi = |text: &str| -> i32 {
        let text = text.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']);
        let (negative, digits) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let mut value: i32 = 0;
        for byte in digits.bytes().take_while(u8::is_ascii_digit) {
            value = value.wrapping_mul(10).wrapping_add((byte - b'0') as i32);
        }
        if negative {
            value.wrapping_neg()
        } else {
            value
        }
    };

    if in_section != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiJpegWriteSectionAny - Trying to write section {} to a JPEG file; only 0 is allowed\n",
                in_section
            ),
        );
        err = IIERR_BAD_CALL;
    }
    if err == 0 && (in_file.pad_left != 0 || in_file.pad_right != 0) {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiJpegWriteSectionAny - Cannot write from a subset of an array\n"),
        );
        err = -1;
    }
    if err == 0
        && !((in_file.format == IIFORMAT_LUMINANCE && in_file.mode == MRC_MODE_BYTE)
            || (in_file.format == IIFORMAT_RGB && if_float == 0))
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiJpegWriteSectionAny - {}\n",
                if in_file.format == IIFORMAT_RGB && if_float != 0 {
                    "Cannot write float data to an RGB JPEG file"
                } else {
                    "File mode must be byte or RGB to write to a JPEG file"
                }
            ),
        );
        err = -1;
    }
    if err == 0
        && (in_file.llx != 0
            || in_file.lly != 0
            || (in_file.urx != -1 && in_file.urx != in_file.nx - 1)
            || (in_file.ury != -1 && in_file.ury != in_file.ny - 1))
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiJpegWriteSectionAny - Can only write a whole section at once\n"),
        );
        err = -1;
    }

    /* Convert floats if necessary */
    let mut converted: Option<Vec<u8>> = None;
    if err == 0 && if_float != 0 {
        let count = in_file.nx as usize * in_file.ny as usize;
        let floats = unsafe { core::slice::from_raw_parts(buf.cast::<f32>(), count) };
        match ii_make_buffer_convert_if_float(
            in_file,
            Some(floats),
            &mut inverted,
            "iiJpegWriteSectionAny",
        ) {
            Ok(buffer) => converted = buffer,
            Err(()) => err = IIERR_MEMORY_ERR,
        }
    }

    /* Free the header, it has not been initialized yet, just created */
    if err != 0 {
        return err;
    }

    /* Get environment variable values */
    if let Ok(value) = std::env::var("IMOD_JPEG_QUALITY") {
        quality = atoi(&value);
        quality = 1.max(100.min(quality));
    }
    if let Ok(value) = std::env::var("IMOD_JPEG_RESOLUTION") {
        err = atoi(&value);
        if err > 65535 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "WARNING: iiJpegWriteSectionAny - IMOD_JPEG_RESOLUTION is set too high to be stored in 16-bit field of JPEG file\n"
                ),
            );
        } else {
            resolution = err;
        }
        resolution = 1.max(resolution);
    }

    /* Call through central routine */
    let channels = if in_file.mode == MRC_MODE_RGB { 3 } else { 1 };
    let use_buf: &[u8] = match converted.as_deref() {
        Some(buffer) => buffer,
        None => unsafe {
            core::slice::from_raw_parts(buf, in_file.nx as usize * in_file.ny as usize * channels)
        },
    };
    err = jpeg_write_section(in_file, use_buf, inverted, resolution, quality);
    if err == 0 {
        in_file.last_written_z = 0;
    }
    err
}

/// C `jpegWriteSection` (`iijpeg.c:435`).
///
/// Writes the section in `buf` to a JPEG file open on `in_file`, which must
/// have a mode of RGB or BYTE.  Set `inverted` if line order is already
/// inverted in Y (first line at top); `resolution` to a value up to 65535 in
/// dots per inch or 0 for none, and `quality` to a value from 1 to 100, or -1
/// for no setting (libjpeg's default is 75, and so is the encoder's here).
pub fn jpeg_write_section(
    in_file: &mut ImodImageFile,
    buf: &[u8],
    inverted: bool,
    resolution: i32,
    quality: i32,
) -> i32 {
    let mut err = 0;

    /* Check various bad things */
    if in_file.write_section.is_none() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: jpegWriteSection - Trying to write to an existing JPEG file\n"),
        );
        err = IIERR_BAD_CALL;
    }
    if err == 0 && in_file.last_written_z == 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: jpegWriteSection - Trying to write more than one section to a JPEG file\n"
            ),
        );
        err = IIERR_BAD_CALL;
    }
    if err == 0 && in_file.mode != MRC_MODE_BYTE && in_file.mode != MRC_MODE_RGB {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: jpegWriteSection - Mode for writing a JPEG file must be either byte or RGB; it is {}\n",
                in_file.mode
            ),
        );
        err = IIERR_BAD_CALL;
    }

    /* Free the header, it has not been initialized yet, just created */
    if err != 0 {
        return err;
    }

    /* Set up error handling */
    let Some(fp) = in_file.fp.as_mut() else {
        return IIERR_BAD_CALL;
    };
    let write_error = |message: String| {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("jpegWriteSection: JPEG error - {}\n", message),
        );
        IIERR_IO_ERROR
    };
    if let Err(error) = fp.seek(SeekFrom::Start(0)) {
        return write_error(error.to_string());
    }

    /* Create the compression object and set the basic properties as well as optional
    quality and resolution */
    let input_components: usize = if in_file.mode == MRC_MODE_RGB { 3 } else { 1 };
    let mut cinfo = if quality > 0 {
        JpegEncoder::new_with_quality(&mut *fp, 100.min(quality) as u8)
    } else {
        JpegEncoder::new(&mut *fp)
    };
    if resolution > 0 {
        cinfo.set_pixel_density(PixelDensity {
            density: (resolution as u16, resolution as u16),
            unit: PixelDensityUnit::Inches,
        });
    }

    /* Do the compression line by line.  The encoder takes the whole image, so
    the scanlines are gathered top-down in the order the source hands them to
    jpeg_write_scanlines. */
    let nx = in_file.nx as usize;
    let row = input_components * nx;
    let mut scanlines = Vec::with_capacity(row * in_file.ny as usize);
    let mut iy = in_file.ny - 1;
    while iy >= 0 {
        let use_y = if inverted { (in_file.ny - 1) - iy } else { iy } as usize;
        scanlines.extend_from_slice(&buf[input_components * use_y * nx..][..row]);
        iy -= 1;
    }
    if let Err(error) = cinfo.write_image(
        &scanlines,
        in_file.nx as u32,
        in_file.ny as u32,
        if input_components == 3 {
            image::ExtendedColorType::Rgb8
        } else {
            image::ExtendedColorType::L8
        },
    ) {
        return write_error(error.to_string());
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(file: &ImodFile, nx: i32, ny: i32) -> ImodImageFile {
        let mut writer = ImodImageFile::default();
        writer.fp = Some(file.clone());
        writer.nx = nx;
        writer.ny = ny;
        writer.nz = 1;
        writer.mode = MRC_MODE_BYTE;
        writer.format = IIFORMAT_LUMINANCE;
        writer.write_section = Some(ii_jpeg_write_section);
        writer.last_written_z = -1;
        writer.urx = -1;
        writer.ury = -1;
        writer
    }

    #[test]
    fn check_reads_the_header_and_rejects_non_jpeg() {
        let file = ImodFile::tmpfile().unwrap();
        let mut out = writer(&file, 2, 2);
        let mut pixels = [0_u8, 0, 255, 255];
        assert_eq!(
            unsafe { ii_jpeg_write_section(&mut out, pixels.as_mut_ptr(), 0) },
            0
        );
        let mut reader = ImodImageFile::default();
        reader.fp = Some(file);
        assert_eq!(ii_jpeg_check(&mut reader), 0);
        assert_eq!(
            (reader.nx, reader.ny, reader.nz, reader.mode, reader.format),
            (2, 2, 1, MRC_MODE_BYTE, IIFORMAT_LUMINANCE)
        );

        let other = ImodFile::tmpfile().unwrap();
        {
            use std::io::Write as _;
            let mut handle = other.clone();
            handle.write_all(b"MRC not a jpeg").unwrap();
            handle.flush().unwrap();
        }
        let mut reader = ImodImageFile::default();
        reader.fp = Some(other);
        assert_eq!(ii_jpeg_check(&mut reader), IIERR_NOT_FORMAT);
    }

    #[test]
    fn writer_rejects_a_second_section_and_nonzero_section() {
        let file = ImodFile::tmpfile().unwrap();
        let mut out = writer(&file, 1, 2);
        let mut pixels = [0_u8, 255];
        assert_eq!(
            unsafe { ii_jpeg_write_section(&mut out, pixels.as_mut_ptr(), 1) },
            IIERR_BAD_CALL
        );
        assert_eq!(jpeg_write_section(&mut out, &pixels, false, 0, 100), 0);
        out.last_written_z = 0;
        assert_eq!(
            jpeg_write_section(&mut out, &pixels, false, 0, 100),
            IIERR_BAD_CALL
        );
    }
}
