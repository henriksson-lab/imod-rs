//! Translation of `IMOD/qttools/mrc2tif/mrc2tif.cpp`.
//!
//! TIFF writing defaults to IMOD's libtiff/QImage boundary.  An explicitly
//! selected, default-off Rust encoder owns only the JPEG/PNG save operation;
//! source-visible image preparation and command behavior stay here.
#![allow(dead_code, unused_variables)]

use crate::imod::backends::{Mrc2TifEncoder, TiffBackend, mrc2tif_encoder, tiff_backend};
use crate::imod::libcfshr::autodoc::{
    ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_image_meta_info, adoc_lookup_by_name_value,
    adoc_open_image_metadata, adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{imod_prog_name, make_line_pointers};
use crate::imod::libcfshr::islice::{Islice, slice_get_val, slice_init, slice_put_val};
use crate::imod::libcfshr::samplemeansd::{sample_mean_sd, type_for_sample_mean};
use crate::imod::libiimod::iimage::{
    IIFILE_ADOC, IIFILE_TIFF, IIFORMAT_LUMINANCE, IIFORMAT_RGB, IITYPE_FLOAT, IITYPE_SHORT,
    IITYPE_UBYTE, IITYPE_USHORT, ImodImageFile, ii_close, ii_delete, ii_fclose, ii_fopen,
    ii_lookup_file_from_fp, ii_new,
};
use crate::imod::libiimod::iitif::{
    tiff_open_new, tiff_parallel_write, tiff_version, tiff_write_finish, tiff_write_section,
    tiff_write_setup, tiff_write_strip,
};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    MrcHeader, mrc_contrast_scaling, mrc_get_scale, mrc_head_read, mrc_init_li,
};
use crate::imod::libiimod::mrcsec::mrc_read_z;
use crate::imod::libiimod::mrcslice::{slice_mmm, slice_new_mode};
use crate::imod::mrc::tiff::tiff_write_image;
use core::ffi::c_char;
use std::ffi::{CStr, CString};

#[cfg(feature = "qt")]
unsafe extern "C" {
    /// C++/Qt shim for the source QImage constructor, resolution/color-table setup, and save.
    /// It returns 0 when `QImage::save` succeeds and nonzero on failure.
    fn mrc2tif_qimage_save(
        data: *mut u8,
        width: i32,
        height: i32,
        bytes_per_line: i32,
        rgb: i32,
        resolution: i32,
        filename: *const c_char,
        format: *const c_char,
        quality: i32,
    ) -> i32;
}

/// No QImage implementation is linked in non-Qt builds.  The source program itself
/// has a Qt build dependency; returning the failed `QImage::save` result here keeps
/// non-Qt binaries linkable without substituting a different JPEG/PNG encoder.
#[cfg(not(feature = "qt"))]
unsafe fn mrc2tif_qimage_save(
    data: *mut u8,
    width: i32,
    height: i32,
    bytes_per_line: i32,
    rgb: i32,
    resolution: i32,
    filename: *const c_char,
    format: *const c_char,
    quality: i32,
) -> i32 {
    let _ = (
        data,
        width,
        height,
        bytes_per_line,
        rgb,
        resolution,
        filename,
        format,
        quality,
    );
    1
}

/// Original: `usage` (`mrc2tif.cpp:28`).
fn usage(progname: &str) {
    // `mrc2tif.cpp:28-62`.  Every line goes through `printf` so it interleaves
    // with `imodCopyright`'s own `printf` in the source's order; a `println!`
    // here lands after the whole libc buffer when stdout is a pipe.
    let name = std::ffi::CString::new(progname).unwrap_or_default();
    unsafe {
        libc::printf(
            c"%s version %s \n".as_ptr(),
            name.as_ptr(),
            c"5.2.17".as_ptr(),
        );
        crate::imod::libcfshr::b3dutil::imod_copyright();
        libc::printf(
            c"%s [options] <mrc file> <tiff name/root>\n\n".as_ptr(),
            name.as_ptr(),
        );
        libc::printf(
            c" Without -s, a series of tiff files will be created with the\n prefix [tiff root name] and with the suffix nnn.tif, where nnn is the z number. \n  Options:\n"
                .as_ptr(),
        );
        libc::printf(
            c"    -s         Stack all images in the mrc file into a single tiff file\n".as_ptr(),
        );
        libc::printf(
            c"    -c val     Compress data; val can be lzw, zip, jpeg, or numbers defined\n\t\t in libtiff\n"
                .as_ptr(),
        );
        libc::printf(
            c"    -q #       Quality for jpeg compression (0-100) or for zip compression (1-9)\n"
                .as_ptr(),
        );
        libc::printf(c"    -S min,max Initial scaling limits for conversion to bytes\n".as_ptr());
        libc::printf(
            c"    -C b,w     Contrast black/white values for conversion to bytes\n".as_ptr(),
        );
        libc::printf(
            c"    -a mn,sd   Scale to mean and SD for conversion to bytes (0,0 for default)\n"
                .as_ptr(),
        );
        libc::printf(c"    -z min,max Starting and ending Z (from 0) to output\n".as_ptr());
        libc::printf(c"    -i #       Initial file number (default is starting Z)\n".as_ptr());
        libc::printf(c"    -j         Output jpeg file instead of tiff\n".as_ptr());
        libc::printf(c"    -p         Output png file instead of tiff\n".as_ptr());
        libc::printf(c"    -r #       Resolution setting in dots per inch\n".as_ptr());
        libc::printf(
            c"    -P         Use pixel spacing in MRC header for resolution setting\n".as_ptr(),
        );
        libc::printf(
            c"    -m         Use pixel spacings in mdoc file for resolution setting\n".as_ptr(),
        );
        libc::printf(
            c"    -T nx[,ny] Output data in tiles of size nx by ny (nx by nx if ny omitted)\n"
                .as_ptr(),
        );
        libc::printf(
            c"    -O #       Override default for parallel tiff compression on whole image\n"
                .as_ptr(),
        );
        libc::printf(
            c"    -t #       Criterion image size in megabytes for processing file in strips\n"
                .as_ptr(),
        );
        libc::printf(c"    -o         Write file with old IMOD code instead of libtiff\n".as_ptr());
    }
}

/// Original: `main` (`mrc2tif.cpp:65`).
pub fn mrc2tif() {
    unsafe {
        let args: Vec<String> = std::env::args().collect();
        let full_progname =
            CString::new(args.first().map(String::as_bytes).unwrap_or(b"mrc2tif")).unwrap();
        let progname_alloc = imod_prog_name(full_progname.as_ptr());
        let progname = CStr::from_ptr(progname_alloc).to_str().unwrap_or("mrc2tif");
        // `mrc2tif.cpp:116-117`: `sprintf(prefix, "\nERROR: %s - ", progname)`
        // then `setExitPrefix(prefix)`.  `PipSetError` (`parse_params.c:2049`)
        // prints the prefix with `"%s "` -- a second space -- and sends it to
        // *stdout*, so an `eprintln!` here is wrong on three counts.
        let exit_prefix = std::ffi::CString::new(format!("\nERROR: {progname} - ")).unwrap();
        crate::imod::libcfshr::parse_params::setExitPrefix(exit_prefix.as_ptr());
        // `mrc2tif.cpp:92`: `int doParallel = -1, didParallel = 0;` -- function
        // scope, because the write-error message reads it.
        let mut did_parallel = 0_i32;
        let encoder = match mrc2tif_encoder() {
            Ok(backend) => backend,
            Err(error) => {
                let message = std::ffi::CString::new(format!("{error}")).unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        };
        let mut iarg = 1usize;
        let mut stack = false;
        let mut oldcode = false;
        let mut compression = 1i32;
        let mut resolution = 0i32;
        let mut use_pixel = false;
        let mut use_mdoc = false;
        let mut make_jpg = false;
        let mut make_png = false;
        let mut convert = false;
        let mut auto_contrast = false;
        let mut auto_mean = 150f32;
        let mut auto_sd = 40f32;
        let mut smin = 0f32;
        let mut smax = 0f32;
        let mut black = 0i32;
        let mut white = 255i32;
        let mut zmin = -1i32;
        let mut zmax = -1i32;
        let mut initial_num = -1i32;
        let mut quality = -1i32;
        let mut tile_x = 0i32;
        let mut lines_per_chunk = 0i32;
        let mut do_chunks = false;
        let mut do_parallel = -1i32;
        let mut chunk_criterion = 400f32;
        while iarg + 1 < args.len() && args[iarg].starts_with('-') {
            let option = args[iarg].as_bytes().get(1).copied().unwrap_or_default() as char;
            match option {
                's' => stack = true,
                'o' => oldcode = true,
                'P' => use_pixel = true,
                'm' => use_mdoc = true,
                'p' => make_png = true,
                'j' => make_jpg = true,
                'h' => {
                    usage(progname);
                    libc::exit(1);
                }
                'c' => {
                    iarg += 1;
                    match args.get(iarg).map(String::as_str) {
                        // `zip` is IMOD's `IICOMPRESSION_ZIP` (Adobe Deflate,
                        // tag 8); 32946 is separately accepted as the
                        // non-Adobe Deflate value, exactly as in the C list.
                        Some("zip") => compression = 8,
                        Some("jpeg") => compression = 7,
                        Some("lzw") => compression = 5,
                        Some(v) => compression = v.parse().unwrap_or(0),
                        None => {
                            usage(progname);
                            libc::exit(1);
                        }
                    }
                    if ![1, 5, 7, 8, 32773, 32946].contains(&compression) {
                        {
                            let message = std::ffi::CString::new(format!(
                                "Compression value {compression} not allowed"
                            ))
                            .unwrap();
                            crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                            unreachable!()
                        }
                    }
                }
                'r' => {
                    iarg += 1;
                    resolution = -args.get(iarg).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
                }
                'S' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        libc::exit(1);
                    };
                    let value = CString::new(value.as_bytes()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f%*c%f".as_ptr(), &mut smin, &mut smax);
                    convert = true;
                }
                'C' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        libc::exit(1);
                    };
                    let value = CString::new(value.as_bytes()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%d%*c%d".as_ptr(), &mut black, &mut white);
                    convert = true;
                }
                'a' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        libc::exit(1);
                    };
                    let value = CString::new(value.as_bytes()).unwrap();
                    let (mut mean, mut sd) = (0.0_f32, 0.0_f32);
                    libc::sscanf(value.as_ptr(), c"%f%*c%f".as_ptr(), &mut mean, &mut sd);
                    if mean != 0. {
                        auto_mean = mean;
                    }
                    if sd != 0. {
                        auto_sd = sd;
                    }
                    auto_contrast = true;
                }
                'z' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        libc::exit(1);
                    };
                    let value = CString::new(value.as_bytes()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%d%*c%d".as_ptr(), &mut zmin, &mut zmax);
                }
                'i' => {
                    iarg += 1;
                    initial_num = args.get(iarg).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
                }
                'q' => {
                    iarg += 1;
                    quality = args.get(iarg).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
                }
                'T' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        libc::exit(1);
                    };
                    let value = CString::new(value.as_bytes()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &mut tile_x,
                        &mut lines_per_chunk,
                    );
                    if lines_per_chunk == 0 {
                        lines_per_chunk = tile_x;
                    }
                    do_chunks = true;
                }
                'O' => {
                    iarg += 1;
                    do_parallel = args.get(iarg).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
                }
                't' => {
                    iarg += 1;
                    chunk_criterion = args
                        .get(iarg)
                        .map(|v| v.parse().unwrap_or(0.0))
                        .unwrap_or(0.0);
                }
                _ => {}
            }
            iarg += 1;
        }
        if oldcode && compression != 1 {
            {
                let message = std::ffi::CString::new(format!(
                    "Compression not available with old writing code"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if oldcode && (resolution != 0 || use_pixel) {
            {
                let message = std::ffi::CString::new(format!(
                    "Resolution setting is not available with old writing code"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if oldcode && tile_x != 0 {
            {
                let message =
                    std::ffi::CString::new(format!("Tiling not available with old writing code"))
                        .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if oldcode && (make_jpg || make_png) {
            {
                let message = std::ffi::CString::new(format!(
                    "JPEG and PNG output not available with old writing code"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if (resolution != 0 && (use_pixel || use_mdoc)) || (use_pixel && use_mdoc) {
            {
                let message = std::ffi::CString::new(format!(
                    "You cannot enter more than one of -r, -m, and -P"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if use_mdoc && stack {
            {
                let message = std::ffi::CString::new(format!(
                    "You cannot use pixel spacings from an mdoc file when making a stack"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if make_jpg && make_png {
            {
                let message =
                    std::ffi::CString::new(format!("You cannot enter both -j and -p")).unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if (make_jpg || make_png) && (stack || do_chunks || compression != 1 || tile_x != 0) {
            {
                let message = std::ffi::CString::new(format!(
                    "You cannot enter -s, -c, or -T with JPEG and PNG output"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if (compression == 8 || make_png) && (quality < -1 || quality == 0 || quality > 9) {
            {
                let message = std::ffi::CString::new(format!(
                    "Quality for ZIP compression or PNG output must be between 1 and 9"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if convert && auto_contrast {
            {
                let message = std::ffi::CString::new(format!(
                    "You cannot enter -C or -S for scaling with -a for auto-contrasting"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if auto_contrast {
            convert = true;
        }
        if make_png && quality >= 0 {
            quality = 10 * (9 - quality);
        }
        let native_tiff_writer = if make_jpg || make_png {
            false
        } else {
            match tiff_backend() {
                Ok(TiffBackend::Parity) => false,
                Ok(TiffBackend::Rust) => true,
                Err(error) => {
                    let message = std::ffi::CString::new(format!("{error}")).unwrap();
                    crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                    unreachable!()
                }
            }
        };
        if native_tiff_writer && compression == 7 {
            // `tiff` 0.11.x can decode modern JPEG TIFF, but its encoder has
            // no JPEG compression variant.  Reject this source-accepted
            // request before opening the input, rather than emitting a TIFF
            // with a false compression tag or falling back to libtiff.
            {
                let message = std::ffi::CString::new(format!(
                    "Rust TIFF writer does not support JPEG compression; use the parity backend"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if native_tiff_writer && (oldcode || do_chunks || !matches!(compression, 1 | 5 | 8)) {
            {
                let message = std::ffi::CString::new(format!(
                "Rust TIFF writer currently supports only non-tiled images with no/LZW/ZIP compression and without old-writer options"
            )).unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        #[cfg(not(feature = "qt"))]
        if make_jpg || make_png {
            {
                let message = std::ffi::CString::new(format!(
                    "JPEG/PNG output requires the source QImage Qt boundary"
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if args.len().saturating_sub(iarg) != 2 {
            usage(progname);
            libc::exit(1);
        }
        let input = match CString::new(args[iarg].as_bytes()) {
            Ok(v) => v,
            Err(_) => {
                let message = std::ffi::CString::new(format!("input filename has NUL")).unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        };
        let fin = ii_fopen(input.as_ptr(), c"rb".as_ptr());
        if fin.is_null() {
            {
                let message =
                    std::ffi::CString::new(format!("Couldn't open {}", args[iarg])).unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        let mut hdata: MrcHeader = core::mem::zeroed();
        if mrc_head_read(fin, &mut hdata) != 0 {
            ii_fclose(fin);
            {
                let message =
                    std::ffi::CString::new(format!("Can't Read Input Header from {}", args[iarg]))
                        .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        if use_pixel || use_mdoc {
            let (xscale, _, _) = mrc_get_scale(&hdata);
            resolution = (1.0e8 / xscale) as i32;
        }
        let mut num_adoc_sect = 0;
        let mut sect_type = 0;
        if use_mdoc {
            let in_file = ii_lookup_file_from_fp(fin);
            if in_file.is_null() {
                ii_fclose(fin);
                {
                    let message = std::ffi::CString::new(format!(
                        "could not get image-file data for input stream"
                    ))
                    .unwrap();
                    crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                    unreachable!()
                }
            }
            let mut if_montage = 0;
            let mut adoc_ind = (*in_file).adoc_index;
            if adoc_ind < 0 {
                adoc_ind = adoc_open_image_metadata(
                    (*in_file).filename,
                    if (*in_file).file == IIFILE_ADOC { 0 } else { 1 },
                    &mut if_montage,
                    &mut num_adoc_sect,
                    &mut sect_type,
                );
            }
            if adoc_ind < 0
                || adoc_set_current(adoc_ind) != 0
                || adoc_get_image_meta_info(&mut if_montage, &mut num_adoc_sect, &mut sect_type)
                    != 0
            {
                ii_fclose(fin);
                {
                    let message =
                        std::ffi::CString::new(format!("could not open or select input metadata"))
                            .unwrap();
                    crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                    unreachable!()
                }
            }
        }
        if zmin == -1 && zmax == -1 {
            zmin = 0;
            zmax = hdata.nz - 1;
        }
        if zmin < 0 || zmax >= hdata.nz || zmin > zmax {
            ii_fclose(fin);
            {
                let message = std::ffi::CString::new(format!(
                    "zmin,zmax values are reversed or out of the range 0 to {}",
                    hdata.nz - 1
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                unreachable!()
            }
        }
        let (psize, type_, format) = match hdata.mode {
            MRC_MODE_BYTE => (1usize, IITYPE_UBYTE, IIFORMAT_LUMINANCE),
            MRC_MODE_SHORT => (2usize, IITYPE_SHORT, IIFORMAT_LUMINANCE),
            MRC_MODE_USHORT => (2usize, IITYPE_USHORT, IIFORMAT_LUMINANCE),
            MRC_MODE_FLOAT => (4usize, IITYPE_FLOAT, IIFORMAT_LUMINANCE),
            MRC_MODE_RGB => (3usize, IITYPE_UBYTE, IIFORMAT_RGB),
            _ => {
                ii_fclose(fin);
                {
                    let message =
                        std::ffi::CString::new(format!("Data mode {} not supported.", hdata.mode))
                            .unwrap();
                    crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                    unreachable!()
                }
            }
        };
        // `mrc2tif.cpp:356-358`: "Must convert to bytes if it is not color for
        // PNG/JPEG output".  This has to run *before* `mrcContrastScaling`
        // (`mrc2tif.cpp:376`), or a float image reaches the QImage writer with
        // scale 1 and offset 0 and is never scaled into byte range.
        if (make_jpg || make_png) && hdata.mode != MRC_MODE_BYTE && hdata.mode != MRC_MODE_RGB {
            convert = true;
        }
        let real_mode = crate::imod::libcfshr::islice::slice_mode_if_real(hdata.mode);
        let mut scale = 1f32;
        let mut offset = 0f32;
        if convert && !auto_contrast {
            (scale, offset) = mrc_contrast_scaling(&hdata, smin, smax, black, white, 1);
        }
        let output_count = zmax - zmin + 1;
        if !(make_jpg || make_png) && !native_tiff_writer {
            let mut minor = 0;
            let version = tiff_version(&mut minor);
            if version < 4 {
                let save_criterion = if version == 0 || oldcode {
                    2.146e9
                } else {
                    4.292e9
                };
                let image_bytes = hdata.nx as f64 * hdata.ny as f64 * psize as f64;
                if image_bytes > save_criterion
                    || (stack && output_count as f64 * image_bytes > save_criterion)
                {
                    ii_fclose(fin);
                    {
                        let message = std::ffi::CString::new(format!(
                            "TIFF {}.{} cannot save this {}",
                            version,
                            minor,
                            if stack { "stack" } else { "image" }
                        ))
                        .unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                        unreachable!()
                    }
                }
            }
            if version > 0
                && !oldcode
                && (do_parallel <= 0 || compression == 1)
                && !auto_contrast
                && hdata.nx as f64 * hdata.ny as f64 * psize as f64
                    > chunk_criterion as f64 * 1024. * 1024.
            {
                do_chunks = true;
            }
        }
        let mut filenum = if initial_num < 0 { zmin } else { initial_num };
        let digits = if output_count >= 10000 {
            5
        } else if output_count >= 1000 {
            4
        } else {
            3
        };
        let iifile = ii_new();
        if iifile.is_null() {
            ii_fclose(fin);
            libc::exit(1);
        }
        (*iifile).format = format;
        (*iifile).file = IIFILE_TIFF;
        (*iifile).type_ = type_;
        // C sets the output ImageFile type before tiffWriteSetup; the setup
        // chooses samples/bits from this field, not from the input header.
        if convert && real_mode > 0 {
            (*iifile).type_ = IITYPE_UBYTE;
        }
        (*iifile).nx = hdata.nx;
        (*iifile).ny = hdata.ny;
        (*iifile).nz = if stack { output_count } else { 1 };
        (*iifile).new_file = 1;
        let mut li: LoadInfo = core::mem::zeroed();
        mrc_init_li(Some(&mut li), None);
        mrc_init_li(Some(&mut li), Some(&hdata));
        let output_root = &args[iarg + 1];
        let extension = if make_jpg {
            "jpg"
        } else if make_png {
            "png"
        } else {
            "tif"
        };
        let mut old_fp: *mut libc::FILE = core::ptr::null_mut();
        let mut ifd_offset = 0u32;
        let mut data_offset = 0u32;
        let mut all_min = 1.0e30f32;
        let mut all_max = -1.0e30f32;
        let rust_output_mode = if convert && real_mode > 0 {
            MRC_MODE_BYTE
        } else {
            hdata.mode
        };
        let rust_output_pixel_size = if rust_output_mode == MRC_MODE_RGB {
            3
        } else if rust_output_mode == MRC_MODE_BYTE {
            1
        } else if rust_output_mode == MRC_MODE_SHORT || rust_output_mode == MRC_MODE_USHORT {
            2
        } else {
            4
        };
        let mut rust_tiff_stack = Vec::new();
        let mut rust_tiff_resolutions = Vec::new();
        if stack && !native_tiff_writer {
            let name = CString::new(output_root.as_bytes()).unwrap();
            let pn = CString::new(progname).unwrap();
            old_fp = open_either_way(
                iifile,
                name.as_ptr().cast_mut(),
                pn.as_ptr(),
                oldcode as i32,
            );
            if old_fp.is_null() && (*iifile).header.is_null() {
                ii_delete(iifile);
                ii_fclose(fin);
                libc::exit(1);
            }
        }
        let type_name = if make_jpg {
            "JPEG"
        } else if make_png {
            "PNG"
        } else {
            "TIFF"
        };
        print!("Writing {type_name} images. ");
        for z in zmin..=zmax {
            let mut slice_min = 1.0e30_f32;
            let mut slice_max = -1.0e30_f32;
            let name_text = if stack || hdata.nz == 1 {
                output_root.to_owned()
            } else {
                format!(
                    "{}.{:0width$}.{extension}",
                    output_root,
                    filenum,
                    width = digits
                )
            };
            if !stack && !(make_jpg || make_png) && !native_tiff_writer {
                if z > zmin {
                    libc::free((*iifile).filename.cast());
                    (*iifile).filename = core::ptr::null_mut();
                }
                filenum += 1;
                let name = match CString::new(name_text.as_bytes()) {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let pn = CString::new(progname).unwrap();
                old_fp = open_either_way(
                    iifile,
                    name.as_ptr().cast_mut(),
                    pn.as_ptr(),
                    oldcode as i32,
                );
                if old_fp.is_null() && (*iifile).header.is_null() {
                    // `mrc2tif.cpp:662`.
                    {
                        let m = std::ffi::CString::new(format!("Opening {name_text}")).unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                        unreachable!()
                    }
                }
            }
            if !stack && (make_jpg || make_png) {
                filenum += 1;
            }
            print!(".");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let mut use_resol = resolution;
            if use_mdoc && z < num_adoc_sect {
                let section_name = if sect_type == 2 {
                    c"Image".as_ptr()
                } else {
                    ADOC_ZVALUE_NAME.as_ptr()
                };
                let section_ind = if sect_type == 2 {
                    z
                } else {
                    adoc_lookup_by_name_value(section_name, z)
                };
                let mut sec_resol = 0.0;
                if section_ind >= 0
                    && adoc_get_float(
                        section_name,
                        section_ind,
                        c"PixelSpacing".as_ptr(),
                        &mut sec_resol,
                    ) == 0
                {
                    use_resol = (1.0e8 / sec_resol) as i32;
                }
            }
            // `tiffWriteSetup` is deliberately outside the chunk loop, as in the
            // C source: it creates one directory and each iteration contributes a
            // consecutive strip or tile to that directory.
            (*iifile).amin = hdata.amin;
            (*iifile).amax = hdata.amax;
            let mut num_chunks = 1;
            if do_chunks {
                if tiff_write_setup(
                    iifile,
                    compression,
                    0,
                    use_resol,
                    quality,
                    &mut lines_per_chunk,
                    &mut num_chunks,
                    &mut tile_x,
                ) != 0
                {
                    // `mrc2tif.cpp:483` ignores the return of `tiffWriteSetup`; this reports it.
                    {
                        let m = std::ffi::CString::new(format!("Setting up TIFF chunks")).unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                        unreachable!()
                    }
                }
                if tile_x > hdata.nx + 16 || lines_per_chunk > hdata.ny + 16 {
                    // `mrc2tif.cpp:486`.
                    {
                        let m = std::ffi::CString::new(format!("Entered tile size was too large"))
                            .unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                        unreachable!()
                    }
                }
            }
            let mut lines_done = 0;
            for chunk in 0..num_chunks {
                let nlines = if do_chunks {
                    lines_per_chunk.min(hdata.ny - lines_done)
                } else {
                    hdata.ny
                };
                // The source owns this as `malloc` storage.  `sliceNewMode`
                // frees and replaces it for converted real modes, so a Rust
                // Vec would be freed once by the C-shaped slice routine and
                // again by Vec's destructor.
                let buffer = libc::malloc(hdata.nx as usize * nlines as usize * psize).cast::<u8>();
                if buffer.is_null() {
                    // `mrc2tif.cpp:497`.
                    {
                        let m =
                            std::ffi::CString::new(format!("Failed to allocate memory for slice"))
                                .unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                        unreachable!()
                    }
                }
                if do_chunks {
                    li.ymin = hdata.ny - (lines_done + nlines);
                    li.ymax = li.ymin + nlines - 1;
                    lines_done += nlines;
                }
                if mrc_read_z(&mut hdata, &mut li, buffer, z) != 0 {
                    // `mrc2tif.cpp:509`; `exitError` exits, so the source frees nothing here.
                    {
                        let m = std::ffi::CString::new(format!("Reading section {z}")).unwrap();
                        crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                        unreachable!()
                    }
                }
                let mut slice: Islice = core::mem::zeroed();
                if slice_init(&mut slice, hdata.nx, nlines, hdata.mode, buffer.cast()) != 0 {
                    libc::free(buffer.cast());
                    break;
                }
                if auto_contrast {
                    let line_ptrs =
                        make_line_pointers(slice.data.b.cast(), hdata.nx, hdata.ny, psize as i32);
                    if line_ptrs.is_null() {
                        // `mrc2tif.cpp:516`.
                        {
                            let m = std::ffi::CString::new(format!(
                                "Allocating line pointers for autocontrasting"
                            ))
                            .unwrap();
                            crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                            unreachable!()
                        }
                    }
                    let sample =
                        (hdata.nx * hdata.ny).min(100_000) as f32 / (hdata.nx * hdata.ny) as f32;
                    let mut image_mean = 0.0;
                    let mut image_sd = 0.0;
                    let sample_error = sample_mean_sd(
                        line_ptrs,
                        type_for_sample_mean(hdata.mode),
                        hdata.nx,
                        hdata.ny,
                        sample,
                        0,
                        0,
                        hdata.nx,
                        hdata.ny,
                        &mut image_mean,
                        &mut image_sd,
                    );
                    libc::free(line_ptrs.cast());
                    if sample_error != 0 {
                        // `mrc2tif.cpp:521`.
                        {
                            let m = std::ffi::CString::new(format!(
                                "Error {sample_error} calling sampleMeanSD for autocontrasting"
                            ))
                            .unwrap();
                            crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                            unreachable!()
                        }
                    }
                    scale = auto_sd / image_sd;
                    offset = auto_mean - scale * image_mean;
                }
                if convert {
                    for y in 0..nlines {
                        for x in 0..hdata.nx {
                            let mut value = [0.; 4];
                            slice_get_val(&mut slice, x, y, &mut value);
                            for channel in 0..if psize == 3 { 3 } else { 1 } {
                                value[channel] = (value[channel] * scale + offset).clamp(0., 255.);
                            }
                            slice_put_val(&mut slice, x, y, value);
                        }
                    }
                    if real_mode > 0 && slice_new_mode(&mut slice, MRC_MODE_BYTE) != 0 {
                        break;
                    }
                }
                let write_buffer = slice.data.b;
                if !(make_jpg || make_png) {
                    slice_mmm(&mut slice);
                    slice_min = slice_min.min(slice.min);
                    slice_max = slice_max.max(slice.max);
                    all_min = all_min.min(slice.min);
                    all_max = all_max.max(slice.max);
                    if !do_chunks {
                        (*iifile).amin = slice_min;
                        (*iifile).amax = slice_max;
                    }
                }
                let mut rust_encoder_error = None;
                let write_error = if make_jpg || make_png {
                    let format = if make_jpg {
                        c"JPEG".as_ptr()
                    } else {
                        c"PNG".as_ptr()
                    };
                    let filename = CString::new(name_text.as_bytes()).unwrap();
                    let out_pixel_size = if psize == 3 { 3 } else { 1 };
                    let line_bytes = (hdata.nx * out_pixel_size + 3) & !3;
                    let mut qbuf = vec![0u8; line_bytes as usize * hdata.ny as usize];
                    for row in 0..hdata.ny as usize {
                        libc::memcpy(
                            qbuf.as_mut_ptr().add(row * line_bytes as usize).cast(),
                            write_buffer
                                .add(
                                    (hdata.ny as usize - 1 - row)
                                        * hdata.nx as usize
                                        * out_pixel_size as usize,
                                )
                                .cast(),
                            (hdata.nx * out_pixel_size) as usize,
                        );
                    }
                    match encoder {
                        Mrc2TifEncoder::Parity => mrc2tif_qimage_save(
                            qbuf.as_mut_ptr(),
                            hdata.nx,
                            hdata.ny,
                            line_bytes,
                            if psize == 3 { 1 } else { 0 },
                            use_resol,
                            filename.as_ptr(),
                            format,
                            quality,
                        ),
                        Mrc2TifEncoder::Rust => match super::rust_encoder::save(
                            &qbuf,
                            hdata.nx,
                            hdata.ny,
                            line_bytes,
                            psize == 3,
                            use_resol,
                            &name_text,
                            if make_jpg { "JPEG" } else { "PNG" },
                            quality,
                        ) {
                            Ok(()) => 0,
                            Err(error) => {
                                rust_encoder_error = Some(error);
                                1
                            }
                        },
                    }
                } else if native_tiff_writer {
                    let rust_bytes = hdata.nx as usize * nlines as usize * rust_output_pixel_size;
                    if stack {
                        rust_tiff_stack
                            .push(core::slice::from_raw_parts(write_buffer, rust_bytes).to_vec());
                        rust_tiff_resolutions.push(use_resol);
                        0
                    } else {
                        match crate::imod::mrc::rust_tiff::write_image(
                            &name_text,
                            hdata.nx,
                            nlines,
                            rust_output_mode,
                            compression,
                            quality,
                            use_resol,
                            write_buffer,
                            rust_bytes,
                        ) {
                            Ok(()) => 0,
                            Err(error) => {
                                rust_encoder_error = Some(error);
                                1
                            }
                        }
                    }
                } else if !old_fp.is_null() {
                    tiff_write_image(
                        old_fp,
                        hdata.nx,
                        hdata.ny,
                        hdata.mode,
                        write_buffer,
                        &mut ifd_offset,
                        &mut data_offset,
                        hdata.amin,
                        hdata.amax,
                    )
                } else if do_chunks {
                    tiff_write_strip(iifile, chunk, write_buffer.cast())
                } else if do_parallel != 0 {
                    tiff_parallel_write(
                        iifile,
                        write_buffer.cast(),
                        compression,
                        0,
                        use_resol,
                        quality,
                        &raw mut did_parallel,
                    )
                } else {
                    tiff_write_section(
                        iifile,
                        write_buffer.cast(),
                        compression,
                        0,
                        use_resol,
                        quality,
                    )
                };
                if real_mode > 0 && convert {
                    libc::free(write_buffer.cast());
                } else {
                    libc::free(buffer.cast());
                }
                if write_error != 0 {
                    // `mrc2tif.cpp:610-613`: one `exitError` with the section
                    // number and the file name, plus the parallel-compression
                    // hint when that path ran.  `exitError` exits without
                    // closing anything, so the source's cleanup here is none.
                    let message = std::ffi::CString::new(format!(
                        "Error ({write_error}) writing section {z} to {name_text}{}",
                        if did_parallel != 0 {
                            "; you could use option -O 0 to try again without parallelized compression"
                        } else {
                            ""
                        }
                    ))
                    .unwrap();
                    crate::imod::libcfshr::parse_params::exit_error(message.as_ptr());
                    unreachable!()
                }
            }
            (*iifile).amin = slice_min;
            (*iifile).amax = slice_max;
            if do_chunks {
                tiff_write_finish(iifile);
            }
            if !stack && !(make_jpg || make_png) && !native_tiff_writer {
                if !old_fp.is_null() {
                    libc::fclose(old_fp);
                } else {
                    ii_close(iifile);
                }
            }
        }
        if stack {
            (*iifile).amin = all_min;
            (*iifile).amax = all_max;
            if native_tiff_writer {
                if let Err(error) = crate::imod::mrc::rust_tiff::write_stack(
                    output_root,
                    hdata.nx,
                    hdata.ny,
                    rust_output_mode,
                    compression,
                    quality,
                    &rust_tiff_stack,
                    &rust_tiff_resolutions,
                ) {
                    // The Rust TIFF writer's own failure has no source
                    // counterpart; report it through the same `exitError`
                    // path so the prefix and stream match.
                    let m = std::ffi::CString::new(error).unwrap_or_default();
                    crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
                    unreachable!()
                }
            } else if !old_fp.is_null() {
                libc::fclose(old_fp);
            } else {
                ii_close(iifile);
            }
        }
        print!("\r\n");
        if tile_x != 0 {
            println!("Actual tile size = {tile_x} x {lines_per_chunk}");
        }
        ii_delete(iifile);
        ii_fclose(fin);
    }
}

/// Original: `openEitherWay` (`mrc2tif.cpp:651`).
unsafe fn open_either_way(
    iifile: *mut ImodImageFile,
    iname: *mut c_char,
    progname: *const c_char,
    oldcode: i32,
) -> *mut libc::FILE {
    static mut WARNED: i32 = 0;
    unsafe {
        if iifile.is_null() || iname.is_null() {
            return core::ptr::null_mut();
        }
        (*iifile).filename = libc::strdup(iname);
        if oldcode != 0 || tiff_open_new(iifile) != 0 {
            let fp = libc::fopen(iname, c"wb".as_ptr());
            if fp.is_null() {
                libc::perror(c"mrc2tif system message".as_ptr());
                // `mrc2tif.cpp:661-662`.
                let m = std::ffi::CString::new(format!(
                    "Opening {}",
                    CStr::from_ptr(iname).to_string_lossy()
                ))
                .unwrap();
                crate::imod::libcfshr::parse_params::exit_error(m.as_ptr());
            }
            if oldcode == 0 && WARNED == 0 {
                WARNED = 1;
                libc::printf(
                    c"\nWARNING: %s - Not writing with libtiff, compression not available".as_ptr(),
                    progname,
                );
            }
            fp
        } else {
            core::ptr::null_mut()
        }
    }
}
