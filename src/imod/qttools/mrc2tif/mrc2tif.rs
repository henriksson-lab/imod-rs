//! Translation of `IMOD/qttools/mrc2tif/mrc2tif.cpp`.
//!
//! TIFF writing defaults to IMOD's libtiff/QImage boundary.  An explicitly
//! selected, default-off Rust encoder owns only the JPEG/PNG save operation;
//! source-visible image preparation and command behavior stay here.
#![allow(dead_code, unused_variables)]

use std::cell::Cell;

use crate::imod::backends::{Mrc2TifEncoder, TiffBackend, mrc2tif_encoder, tiff_backend};
use crate::imod::libcfshr::autodoc::{
    ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_image_meta_info, adoc_lookup_by_name_value,
    adoc_open_image_metadata, adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes, imod_prog_name};
// `mrc2tif.cpp` scans its paired option values with the C library's `sscanf`.
// `clip/clip.rs` carries this tree's translation of that routine (NATIVE.md
// hazard 2: `str::parse` is not `sscanf` -- it rejects the partial parse the
// `%f%*c%f` pairs depend on), so it is used rather than written again here.
use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::islice::{slice_create, slice_get_val, slice_init, slice_put_val};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
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
use crate::imod::libiimod::mrcslice::slice_mmm;
use crate::imod::mrc::tiff::tiff_write_image;
use std::io::Write;

/// The one C boundary this unit keeps.  `build.rs` compiles
/// `mrc2tif_qimage.cpp` against system Qt5 and links it statically, so
/// `filename` and `format` are `const char *` on the C++ side and have to be
/// NUL-terminated here.  Nothing above this declaration carries a C string.
#[cfg(feature = "qt")]
unsafe extern "C" {
    /// C++/Qt shim for the source QImage constructor, resolution/color-table setup, and save.
    /// It returns 0 when `QImage::save` succeeds and nonzero on failure.
    #[link_name = "mrc2tif_qimage_save"]
    fn mrc2tif_qimage_save_ffi(
        data: *mut u8,
        width: i32,
        height: i32,
        bytes_per_line: i32,
        rgb: i32,
        resolution: i32,
        filename: *const core::ffi::c_char,
        format: *const core::ffi::c_char,
        quality: i32,
    ) -> i32;
}

/// No QImage implementation is linked in non-Qt builds.  The source program itself
/// has a Qt build dependency; returning the failed `QImage::save` result here keeps
/// non-Qt binaries linkable without substituting a different JPEG/PNG encoder.
#[cfg(feature = "qt")]
fn mrc2tif_qimage_save(
    data: &mut [u8],
    width: i32,
    height: i32,
    bytes_per_line: i32,
    rgb: i32,
    resolution: i32,
    filename: &str,
    format: &str,
    quality: i32,
) -> i32 {
    let mut filename = filename.as_bytes().to_vec();
    filename.push(0);
    let mut format = format.as_bytes().to_vec();
    format.push(0);
    unsafe {
        mrc2tif_qimage_save_ffi(
            data.as_mut_ptr(),
            width,
            height,
            bytes_per_line,
            rgb,
            resolution,
            filename.as_ptr().cast(),
            format.as_ptr().cast(),
            quality,
        )
    }
}

/// No QImage implementation is linked in non-Qt builds.  The source program itself
/// has a Qt build dependency; returning the failed `QImage::save` result here keeps
/// non-Qt binaries linkable without substituting a different JPEG/PNG encoder.
#[cfg(not(feature = "qt"))]
fn mrc2tif_qimage_save(
    data: &mut [u8],
    width: i32,
    height: i32,
    bytes_per_line: i32,
    rgb: i32,
    resolution: i32,
    filename: &str,
    format: &str,
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
fn usage(progname: &[u8]) {
    // `mrc2tif.cpp:28-62`.  Every line goes through `printf`, so it has to
    // reach the *C* stdout to interleave with `imodCopyright`'s own `printf`
    // in the source's order; a `println!` here lands after the whole libc
    // buffer when stdout is a pipe.
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(&c_format_bytes(
        "%s version %s \n",
        &[CArg::Bytes(progname), CArg::Str("5.2.17")],
    ));
    crate::imod::libcfshr::b3dutil::imod_copyright();
    let _ = out.write_all(&c_format_bytes(
        "%s [options] <mrc file> <tiff name/root>\n\n",
        &[CArg::Bytes(progname)],
    ));
    let _ = out.write_all(
        b" Without -s, a series of tiff files will be created with the\n prefix [tiff root name] and with the suffix nnn.tif, where nnn is the z number. \n  Options:\n",
    );
    let _ =
        out.write_all(b"    -s         Stack all images in the mrc file into a single tiff file\n");
    let _ = out.write_all(
        b"    -c val     Compress data; val can be lzw, zip, jpeg, or numbers defined\n\t\t in libtiff\n",
    );
    let _ = out.write_all(
        b"    -q #       Quality for jpeg compression (0-100) or for zip compression (1-9)\n",
    );
    let _ = out.write_all(b"    -S min,max Initial scaling limits for conversion to bytes\n");
    let _ = out.write_all(b"    -C b,w     Contrast black/white values for conversion to bytes\n");
    let _ = out.write_all(
        b"    -a mn,sd   Scale to mean and SD for conversion to bytes (0,0 for default)\n",
    );
    let _ = out.write_all(b"    -z min,max Starting and ending Z (from 0) to output\n");
    let _ = out.write_all(b"    -i #       Initial file number (default is starting Z)\n");
    let _ = out.write_all(b"    -j         Output jpeg file instead of tiff\n");
    let _ = out.write_all(b"    -p         Output png file instead of tiff\n");
    let _ = out.write_all(b"    -r #       Resolution setting in dots per inch\n");
    let _ =
        out.write_all(b"    -P         Use pixel spacing in MRC header for resolution setting\n");
    let _ =
        out.write_all(b"    -m         Use pixel spacings in mdoc file for resolution setting\n");
    let _ = out.write_all(
        b"    -T nx[,ny] Output data in tiles of size nx by ny (nx by nx if ny omitted)\n",
    );
    let _ = out.write_all(
        b"    -O #       Override default for parallel tiff compression on whole image\n",
    );
    let _ = out.write_all(
        b"    -t #       Criterion image size in megabytes for processing file in strips\n",
    );
    let _ = out.write_all(b"    -o         Write file with old IMOD code instead of libtiff\n");
}

/// Original: `main` (`mrc2tif.cpp:65`).
pub fn mrc2tif() {
    unsafe {
        let args: Vec<String> = std::env::args().collect();
        let progname_alloc = imod_prog_name(args.first().map(String::as_str).unwrap_or("mrc2tif"));
        let progname = progname_alloc.as_bytes();
        // `mrc2tif.cpp:116-117`: `sprintf(prefix, "\nERROR: %s - ", progname)`
        // then `setExitPrefix(prefix)`.  `PipSetError` (`parse_params.c:2049`)
        // prints the prefix with `"%s "` -- a second space -- and sends it to
        // *stdout*, so an `eprintln!` here is wrong on three counts.
        setExitPrefix(&c_format_bytes("\nERROR: %s - ", &[CArg::Bytes(progname)]));
        // `mrc2tif.cpp:92`: `int doParallel = -1, didParallel = 0;` -- function
        // scope, because the write-error message reads it.
        let mut did_parallel = 0_i32;
        let encoder = match mrc2tif_encoder() {
            Ok(backend) => backend,
            Err(error) => exit_error(error.as_bytes()),
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
                    let _ = ImodFile::Stdout.flush();
                    std::process::exit(1);
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
                            let _ = ImodFile::Stdout.flush();
                            std::process::exit(1);
                        }
                    }
                    if ![1, 5, 7, 8, 32773, 32946].contains(&compression) {
                        exit_error(&c_format_bytes(
                            "Compression value %d not allowed",
                            &[CArg::Int(compression as i64)],
                        ));
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
                        let _ = ImodFile::Stdout.flush();
                        std::process::exit(1);
                    };
                    // `mrc2tif.cpp:167`: `sscanf(argv[++iarg], "%f%*c%f", ...)`.
                    sscanf(
                        value,
                        "%f%*c%f",
                        &mut [ScanArg::Flt(&mut smin), ScanArg::Flt(&mut smax)],
                    );
                    convert = true;
                }
                'C' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        let _ = ImodFile::Stdout.flush();
                        std::process::exit(1);
                    };
                    sscanf(
                        value,
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut black), ScanArg::Int(&mut white)],
                    );
                    convert = true;
                }
                'a' => {
                    iarg += 1;
                    let Some(value) = args.get(iarg) else {
                        usage(progname);
                        let _ = ImodFile::Stdout.flush();
                        std::process::exit(1);
                    };
                    // `mrc2tif.cpp:177`: the source scans into `scale` and
                    // `offset`, which are otherwise the contrast-scaling
                    // locals, and copies each into `autoMean`/`autoSD` only
                    // when it is nonzero.
                    let (mut mean, mut sd) = (0.0_f32, 0.0_f32);
                    sscanf(
                        value,
                        "%f%*c%f",
                        &mut [ScanArg::Flt(&mut mean), ScanArg::Flt(&mut sd)],
                    );
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
                        let _ = ImodFile::Stdout.flush();
                        std::process::exit(1);
                    };
                    sscanf(
                        value,
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut zmin), ScanArg::Int(&mut zmax)],
                    );
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
                        let _ = ImodFile::Stdout.flush();
                        std::process::exit(1);
                    };
                    sscanf(
                        value,
                        "%d%*c%d",
                        &mut [
                            ScanArg::Int(&mut tile_x),
                            ScanArg::Int(&mut lines_per_chunk),
                        ],
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
            exit_error(b"Compression not available with old writing code");
        }
        if oldcode && (resolution != 0 || use_pixel) {
            exit_error(b"Resolution setting is not available with old writing code");
        }
        if oldcode && tile_x != 0 {
            exit_error(b"Tiling not available with old writing code");
        }
        if oldcode && (make_jpg || make_png) {
            exit_error(b"JPEG and PNG output not available with old writing code");
        }
        if (resolution != 0 && (use_pixel || use_mdoc)) || (use_pixel && use_mdoc) {
            exit_error(b"You cannot enter more than one of -r, -m, and -P");
        }
        if use_mdoc && stack {
            exit_error(b"You cannot use pixel spacings from an mdoc file when making a stack");
        }
        // `mrc2tif.cpp:240-241`: `makeQimage` and `typeInd`, which index the
        // extension and type-name tables below.
        let make_qimage = make_jpg || make_png;
        if make_qimage && (stack || do_chunks || compression != 1 || tile_x != 0) {
            exit_error(b"You cannot enter -s, -c, or -T with JPEG and PNG output");
        }
        if make_png && make_jpg {
            exit_error(b"You cannot enter both -j and -p");
        }
        if (compression == 8 || make_png) && (quality < -1 || quality == 0 || quality > 9) {
            exit_error(b"Quality for ZIP compression or PNG output must be between 1 and 9");
        }
        if convert && auto_contrast {
            exit_error(b"You cannot enter -C or -S for scaling with -a for auto-contrasting");
        }
        if auto_contrast {
            convert = true;
        }
        let native_tiff_writer = if make_qimage {
            false
        } else {
            match tiff_backend() {
                Ok(TiffBackend::Parity) => false,
                Ok(TiffBackend::Rust) => true,
                Err(error) => exit_error(error.as_bytes()),
            }
        };
        if native_tiff_writer && compression == 7 {
            // `tiff` 0.11.x can decode modern JPEG TIFF, but its encoder has
            // no JPEG compression variant.  Reject this source-accepted
            // request before opening the input, rather than emitting a TIFF
            // with a false compression tag or falling back to libtiff.
            exit_error(
                b"Rust TIFF writer does not support JPEG compression; use the parity backend",
            );
        }
        if native_tiff_writer && (oldcode || do_chunks || !matches!(compression, 1 | 5 | 8)) {
            exit_error(
                b"Rust TIFF writer currently supports only non-tiled images with no/LZW/ZIP compression and without old-writer options",
            );
        }
        #[cfg(not(feature = "qt"))]
        if make_qimage {
            exit_error(b"JPEG/PNG output requires the source QImage Qt boundary");
        }
        if args.len().saturating_sub(iarg) != 2 {
            usage(progname);
            let _ = ImodFile::Stdout.flush();
            std::process::exit(1);
        }
        if make_qimage {
            // `mrc2tif.cpp:270-271`: convert the quality for QImage use by
            // inverting and multiplying by 10.  The plugin-path block above it
            // is `QApplication::setLibraryPaths`, which has no counterpart on
            // this side of the boundary.
            if make_png && quality >= 0 {
                quality = 10 * (9 - quality);
            }
        }
        let mut fin = match ii_fopen(args[iarg].as_bytes(), "rb") {
            Some(file) => file,
            None => exit_error(&c_format_bytes(
                "Couldn't open %s",
                &[CArg::Bytes(args[iarg].as_bytes())],
            )),
        };
        let mut hdata = MrcHeader::default();
        if mrc_head_read(&mut fin, &mut hdata) != 0 {
            exit_error(&c_format_bytes(
                "Can't Read Input Header from %s",
                &[CArg::Bytes(args[iarg].as_bytes())],
            ));
        }
        // `mrc2tif.cpp:288`: `iarg++`, so from here `argv[iarg]` is the output
        // name or root.
        iarg += 1;
        if use_pixel || use_mdoc {
            let (xscale, _, _) = mrc_get_scale(&hdata);
            // `mrc2tif.cpp:291`: `resolution = 1.e8 / xscale`, with `xscale` a
            // `float` and `1.e8` a *double*, so the division is in double and
            // only the assignment to `int` narrows.  Dividing in f32 instead
            // lands one ulp away -- at 3.0 A/pixel that is 33333334 rather
            // than 33333333, which libtiff then stores as a different
            // XResolution rational (CLAUDE.md: widen after the narrow
            // operation, not before).
            resolution = (1.0e8 / xscale as f64) as i32;
        }
        let mut num_adoc_sect = 0;
        let mut sect_type = 0;
        if use_mdoc {
            let Some(in_file) = ii_lookup_file_from_fp(&fin) else {
                exit_error(b"Could not get general image file data correspoding to file pointer");
            };
            let mut if_montage = 0;
            let mut adoc_ind = (*in_file).adoc_index;
            if adoc_ind < 0 {
                let name = (*in_file).filename.clone().unwrap_or_default();
                adoc_ind = adoc_open_image_metadata(
                    name.as_bytes(),
                    if (*in_file).file == IIFILE_ADOC { 0 } else { 1 },
                    &mut if_montage,
                    &mut num_adoc_sect,
                    &mut sect_type,
                );
            }
            if adoc_ind < 0 {
                exit_error(&c_format_bytes(
                    "Could not find an mdoc file or autodoc information for input file %s",
                    &[CArg::Bytes(
                        (*in_file)
                            .filename
                            .as_deref()
                            .unwrap_or_default()
                            .as_bytes(),
                    )],
                ));
            }
            if adoc_set_current(adoc_ind) != 0
                || adoc_get_image_meta_info(&mut if_montage, &mut num_adoc_sect, &mut sect_type)
                    != 0
            {
                exit_error(b"Setting current autodoc or getting information about it");
            }
        }
        if zmin == -1 && zmax == -1 {
            zmin = 0;
            zmax = hdata.nz - 1;
        } else if zmin < 0 || zmax >= hdata.nz || zmin > zmax {
            exit_error(&c_format_bytes(
                "zmin,zmax values are reversed or out of the range 0 to %d\n",
                &[CArg::Int((hdata.nz - 1) as i64)],
            ));
        }
        let mut filenum = if initial_num < 0 { zmin } else { initial_num };
        let (psize, type_, format) = match hdata.mode {
            MRC_MODE_BYTE => (1usize, IITYPE_UBYTE, IIFORMAT_LUMINANCE),
            MRC_MODE_SHORT => (2usize, IITYPE_SHORT, IIFORMAT_LUMINANCE),
            MRC_MODE_USHORT => (2usize, IITYPE_USHORT, IIFORMAT_LUMINANCE),
            MRC_MODE_FLOAT => (4usize, IITYPE_FLOAT, IIFORMAT_LUMINANCE),
            MRC_MODE_RGB => (3usize, IITYPE_UBYTE, IIFORMAT_RGB),
            _ => exit_error(&c_format_bytes(
                "Data mode %d not supported.",
                &[CArg::Int(hdata.mode as i64)],
            )),
        };
        // `mrc2tif.cpp:356-358`: "Must convert to bytes if it is not color for
        // PNG/JPEG output".  This has to run *before* `mrcContrastScaling`
        // (`mrc2tif.cpp:376`), or a float image reaches the QImage writer with
        // scale 1 and offset 0 and is never scaled into byte range.
        if make_qimage && hdata.mode != MRC_MODE_BYTE && hdata.mode != MRC_MODE_RGB {
            convert = true;
        }
        let iifile = ii_new();
        if iifile.is_null() {
            let _ = ImodFile::Stdout.flush();
            std::process::exit(1);
        }
        (*iifile).format = format;
        (*iifile).file = IIFILE_TIFF;
        (*iifile).type_ = type_;
        (*iifile).amin = 0.;
        (*iifile).amax = 0.;
        (*iifile).new_file = 1;
        (*iifile).nx = hdata.nx;
        (*iifile).ny = hdata.ny;
        (*iifile).nz = 1;
        let mut li = LoadInfo::default();
        mrc_init_li(Some(&mut li), None);
        li.xmin = 0;
        li.xmax = hdata.nx - 1;
        li.ymin = 0;
        li.ymax = hdata.ny - 1;
        let dmin = hdata.amin;
        let dmax = hdata.amax;
        let real_mode = crate::imod::libcfshr::islice::slice_mode_if_real(hdata.mode);
        let mut out_psize = psize;
        let mut scale = 1f32;
        let mut offset = 0f32;
        if convert {
            if !auto_contrast {
                (scale, offset) = mrc_contrast_scaling(&hdata, smin, smax, black, white, 1);
            }
            // C sets the output ImageFile type before tiffWriteSetup; the
            // setup chooses samples/bits from this field, not from the input
            // header.
            if real_mode > 0 {
                (*iifile).type_ = IITYPE_UBYTE;
                out_psize = 1;
            }
        }
        let output_count = zmax - zmin + 1;
        let xysize = hdata.nx as f64 * hdata.ny as f64;
        if !make_qimage {
            let (version, minor) = tiff_version();
            if version < 4 {
                let save_criterion = if version == 0 || oldcode {
                    2.146e9
                } else {
                    4.292e9
                };
                if xysize * out_psize as f64 > save_criterion {
                    exit_error(&c_format_bytes(
                        "The image is too large in X/Y to save with TIFF version %d.%d",
                        &[CArg::Int(version as i64), CArg::Int(minor as i64)],
                    ));
                }
                if stack && output_count as f64 * xysize * out_psize as f64 > save_criterion {
                    exit_error(&c_format_bytes(
                        "The volume is too large to save in a stack with TIFF version %d.%d",
                        &[CArg::Int(version as i64), CArg::Int(minor as i64)],
                    ));
                }
            }
            if version > 0
                && !oldcode
                && (do_parallel <= 0 || compression == 1)
                && !auto_contrast
                && xysize * psize as f64 > chunk_criterion as f64 * 1024. * 1024.
            {
                do_chunks = true;
            }
        } else if make_jpg && (hdata.nx > 65535 || hdata.ny > 65535) {
            // `mrc2tif.cpp:410`.
            exit_error(b"The input image is too large in X or Y for JPEG output");
        }
        // `mrc2tif.cpp:414-421`: the QImage line stride, rounded up to a
        // 32-bit boundary, and the one-line swap buffer the inversion uses.
        let line_bytes = 4 * ((hdata.nx as usize * out_psize + 3) / 4);
        let output_root = args[iarg].clone();
        let extension: &str = if make_jpg {
            "jpg"
        } else if make_png {
            "png"
        } else {
            "tif"
        };
        let type_name: &str = if make_jpg {
            "JPEG"
        } else if make_png {
            "PNG"
        } else {
            "TIFF"
        };
        let mut old_fp: Option<ImodFile> = None;
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
        let mut rust_tiff_stack: Vec<Vec<u8>> = Vec::new();
        let mut rust_tiff_resolutions: Vec<i32> = Vec::new();
        if stack {
            // `mrc2tif.cpp:426-429`.
            (*iifile).nz = output_count;
        }
        if stack && !native_tiff_writer {
            old_fp = open_either_way(
                &mut *iifile,
                output_root.as_bytes(),
                progname,
                oldcode as i32,
            );
            if old_fp.is_none()
                && (*iifile).mrc_header.is_none()
                && (*iifile).backend_handle.is_null()
            {
                let _ = ImodFile::Stdout.flush();
                std::process::exit(1);
            }
        }
        let digits = if output_count >= 10000 {
            5
        } else if output_count >= 1000 {
            4
        } else {
            3
        };
        let mut out = ImodFile::Stdout;
        let _ = out.write_all(&c_format_bytes(
            "Writing %s images. ",
            &[CArg::Str(type_name)],
        ));
        for z in zmin..=zmax {
            let mut slice_min = 1.0e30_f32;
            let mut slice_max = -1.0e30_f32;
            let _ = out.write_all(b".");
            let _ = out.flush();
            let mut name_text = output_root.clone();
            if !stack {
                // `mrc2tif.cpp:451-453`: the numbered name is built, then
                // replaced by the bare root when the *input* has a single
                // section.  `filenum++` happens either way.
                name_text = String::from_utf8_lossy(&c_format_bytes(
                    "%s.%0*d.%s",
                    &[
                        CArg::Bytes(output_root.as_bytes()),
                        CArg::Star(digits),
                        CArg::Int(filenum as i64),
                        CArg::Str(extension),
                    ],
                ))
                .into_owned();
                filenum += 1;
                if hdata.nz == 1 {
                    name_text = output_root.clone();
                }
                if !make_qimage && !native_tiff_writer {
                    if z > zmin {
                        (*iifile).filename = None;
                    }
                    old_fp = open_either_way(
                        &mut *iifile,
                        name_text.as_bytes(),
                        progname,
                        oldcode as i32,
                    );
                    if old_fp.is_none()
                        && (*iifile).mrc_header.is_none()
                        && (*iifile).backend_handle.is_null()
                    {
                        // `mrc2tif.cpp:662`.
                        exit_error(&c_format_bytes(
                            "Opening %s",
                            &[CArg::Bytes(name_text.as_bytes())],
                        ));
                    }
                    ifd_offset = 0;
                    data_offset = 0;
                }
            }
            let mut use_resol = resolution;
            if use_mdoc && z < num_adoc_sect {
                // `mrc2tif.cpp:100`: `sectNames[3] = {ADOC_ZVALUE_NAME,
                // "Image", ADOC_ZVALUE_NAME}`, indexed by `sectType - 1`.
                let section_name: &[u8] = if sect_type == 2 {
                    b"Image"
                } else {
                    ADOC_ZVALUE_NAME
                };
                let section_ind = if sect_type == 2 {
                    z
                } else {
                    adoc_lookup_by_name_value(section_name, z)
                };
                let mut sec_resol = 0.0;
                if section_ind >= 0
                    && adoc_get_float(section_name, section_ind, b"PixelSpacing", &mut sec_resol)
                        == 0
                {
                    // `mrc2tif.cpp:471`, in double for the same reason as
                    // line 291 above.
                    use_resol = (1.0e8 / sec_resol as f64) as i32;
                }
            }
            // `tiffWriteSetup` is deliberately outside the chunk loop, as in the
            // C source: it creates one directory and each iteration contributes a
            // consecutive strip or tile to that directory.
            (*iifile).amin = dmin;
            (*iifile).amax = dmax;
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
                    exit_error(b"Setting up TIFF chunks");
                }
                if tile_x > hdata.nx + 16 || lines_per_chunk > hdata.ny + 16 {
                    // `mrc2tif.cpp:486`.
                    exit_error(b"Entered tile size was too large");
                }
            }
            let mut lines_done = 0;
            for chunk in 0..num_chunks {
                let nlines = if do_chunks {
                    lines_per_chunk.min(hdata.ny - lines_done)
                } else {
                    hdata.ny
                };
                let Some(buffer_len) = (hdata.nx as usize)
                    .checked_mul(nlines as usize)
                    .and_then(|pixels| pixels.checked_mul(psize))
                else {
                    exit_error(b"Failed to allocate memory for slice");
                };
                let mut buffer = Vec::new();
                if buffer.try_reserve_exact(buffer_len).is_err() {
                    // `mrc2tif.cpp:497`.
                    exit_error(b"Failed to allocate memory for slice");
                }
                buffer.resize(buffer_len, 0);
                if do_chunks {
                    li.ymin = hdata.ny - (lines_done + nlines);
                    li.ymax = li.ymin + nlines - 1;
                    lines_done += nlines;
                }
                if mrc_read_z(&mut hdata, &mut li, &mut buffer, z) != 0 {
                    // `mrc2tif.cpp:508-509`: `perror("mrc2tif ")` then
                    // `exitError`.  `exitError` exits, so the source frees
                    // nothing here.
                    // `perror("mrc2tif ")` writes `"mrc2tif : <strerror>\n"` to
                    // stderr.  Rust's `io::Error` Display appends
                    // ` (os error N)`, which the C library does not, so the
                    // suffix is trimmed back off.
                    let message = std::io::Error::last_os_error().to_string();
                    let message = message
                        .split(" (os error ")
                        .next()
                        .unwrap_or(message.as_str());
                    let _ = ImodFile::Stderr
                        .write_all(&c_format_bytes("mrc2tif : %s\n", &[CArg::Str(message)]));
                    exit_error(&c_format_bytes(
                        "Reading section %d",
                        &[CArg::Int(z as i64)],
                    ));
                }
                let Some(mut slice) = slice_create(hdata.nx, nlines, hdata.mode) else {
                    break;
                };
                slice.data.copy_from_slice(&buffer);
                if auto_contrast {
                    let sample =
                        (hdata.nx * hdata.ny).min(100_000) as f32 / (hdata.nx * hdata.ny) as f32;
                    let mut image_mean = 0.0;
                    let mut image_sd = 0.0;
                    // `makeLinePointers` becomes the line byte views the
                    // translated `sampleMeanSD` takes.
                    let bytes = buffer.as_slice();
                    let lines: Vec<&[u8]> = (0..nlines as usize)
                        .map(|index| &bytes[(hdata.nx as usize * index * psize)..])
                        .collect();
                    let sample_error = sample_mean_sd(
                        Some(&lines),
                        type_for_sample_mean(hdata.mode),
                        hdata.nx,
                        hdata.ny,
                        sample,
                        0,
                        0,
                        hdata.nx,
                        hdata.ny,
                        Some(&mut image_mean),
                        Some(&mut image_sd),
                    );
                    if sample_error != 0 {
                        // `mrc2tif.cpp:521`.
                        exit_error(&c_format_bytes(
                            "Error %d calling sampleMeanSD for autocontrasting",
                            &[CArg::Int(sample_error as i64)],
                        ));
                    }
                    scale = auto_sd / image_sd;
                    offset = auto_mean - scale * image_mean;
                }
                if convert {
                    for y in 0..nlines {
                        for x in 0..hdata.nx {
                            let mut value = [0.; 4];
                            slice_get_val(slice.as_mut(), x, y, &mut value);
                            for channel in 0..if psize == 3 { 3 } else { 1 } {
                                value[channel] = (value[channel] * scale + offset).clamp(0., 255.);
                            }
                            slice_put_val(slice.as_mut(), x, y, value);
                        }
                    }
                    if real_mode > 0 {
                        let mut converted = Vec::new();
                        if converted.try_reserve_exact(buffer_len).is_err() {
                            exit_error(&c_format_bytes(
                                "Converting slice %d to bytes",
                                &[CArg::Int(z as i64)],
                            ));
                        }
                        converted.resize(buffer_len / psize, 0);
                        for y in 0..nlines {
                            for x in 0..hdata.nx {
                                let mut value = [0.; 4];
                                slice_get_val(slice.as_mut(), x, y, &mut value);
                                converted[x as usize + y as usize * hdata.nx as usize] =
                                    value[0] as i32 as u8;
                            }
                        }
                        if slice_init(slice.as_mut(), hdata.nx, nlines, MRC_MODE_BYTE, converted)
                            != 0
                        {
                            exit_error(&c_format_bytes(
                                "Converting slice %d to bytes",
                                &[CArg::Int(z as i64)],
                            ));
                        }
                    }
                }
                if !make_qimage {
                    slice_mmm(slice.as_mut());
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
                let write_buffer = slice.data.as_mut_slice();
                let write_error = if make_qimage {
                    let out_pixel_size = if psize == 3 { 3 } else { 1 };
                    let mut qbuf = vec![0u8; line_bytes * hdata.ny as usize];
                    let row_bytes = hdata.nx as usize * out_pixel_size;
                    let source = &write_buffer[..row_bytes * hdata.ny as usize];
                    // `mrc2tif.cpp:570-583`: spread the rows to the aligned
                    // stride and invert the image, in one pass.
                    for row in 0..hdata.ny as usize {
                        let from = (hdata.ny as usize - 1 - row) * row_bytes;
                        qbuf[row * line_bytes..row * line_bytes + row_bytes]
                            .copy_from_slice(&source[from..from + row_bytes]);
                    }
                    match encoder {
                        Mrc2TifEncoder::Parity => mrc2tif_qimage_save(
                            &mut qbuf,
                            hdata.nx,
                            hdata.ny,
                            line_bytes as i32,
                            if psize == 3 { 1 } else { 0 },
                            use_resol,
                            &name_text,
                            type_name,
                            quality,
                        ),
                        Mrc2TifEncoder::Rust => match super::rust_encoder::save(
                            &qbuf,
                            hdata.nx,
                            hdata.ny,
                            line_bytes as i32,
                            psize == 3,
                            use_resol,
                            &name_text,
                            type_name,
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
                    let image = &write_buffer[..rust_bytes];
                    if stack {
                        rust_tiff_stack.push(image.to_vec());
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
                            image,
                        ) {
                            Ok(()) => 0,
                            Err(error) => {
                                rust_encoder_error = Some(error);
                                1
                            }
                        }
                    }
                } else if old_fp.is_some() {
                    tiff_write_image(
                        old_fp.as_mut().unwrap(),
                        hdata.nx,
                        hdata.ny,
                        hdata.mode,
                        write_buffer,
                        &mut ifd_offset,
                        &mut data_offset,
                        dmin,
                        dmax,
                    )
                } else if do_chunks {
                    tiff_write_strip(iifile, chunk, write_buffer.as_mut_ptr().cast())
                } else if do_parallel != 0 {
                    tiff_parallel_write(
                        iifile,
                        write_buffer.as_mut_ptr().cast(),
                        compression,
                        0,
                        use_resol,
                        quality,
                        &raw mut did_parallel,
                    )
                } else {
                    tiff_write_section(
                        &mut *iifile,
                        write_buffer,
                        compression,
                        0,
                        use_resol,
                        quality,
                    )
                };
                if write_error != 0 {
                    // `mrc2tif.cpp:610-613`: one `exitError` with the section
                    // number and the file name, plus the parallel-compression
                    // hint when that path ran.  `exitError` exits without
                    // closing anything, so the source's cleanup here is none.
                    if let Some(error) = rust_encoder_error {
                        let _ = ImodFile::Stderr.write_all(error.as_bytes());
                        let _ = ImodFile::Stderr.write_all(b"\n");
                    }
                    exit_error(&c_format_bytes(
                        "Error (%d) writing section %d to %s%s",
                        &[
                            CArg::Int(write_error as i64),
                            CArg::Int(z as i64),
                            CArg::Bytes(name_text.as_bytes()),
                            CArg::Str(if did_parallel != 0 {
                                "; you could use option -O 0 to try again without parallelized compression"
                            } else {
                                ""
                            }),
                        ],
                    ));
                }
            }
            (*iifile).amin = slice_min;
            (*iifile).amax = slice_max;
            if do_chunks {
                tiff_write_finish(iifile);
            }
            if !stack && !make_qimage && !native_tiff_writer {
                if old_fp.is_some() {
                    drop(old_fp.take());
                } else {
                    ii_close(iifile);
                }
            }
        }
        let _ = out.write_all(b"\r\n");
        (*iifile).amin = all_min;
        (*iifile).amax = all_max;
        if stack {
            if native_tiff_writer {
                if let Err(error) = crate::imod::mrc::rust_tiff::write_stack(
                    &output_root,
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
                    exit_error(error.as_bytes());
                }
            } else if old_fp.is_some() {
                drop(old_fp.take());
            } else {
                ii_close(iifile);
            }
        }
        ii_fclose(&mut fin);
        if tile_x != 0 {
            let _ = out.write_all(&c_format_bytes(
                "Actual tile size = %d x %d\n",
                &[CArg::Int(tile_x as i64), CArg::Int(lines_per_chunk as i64)],
            ));
        }
        ii_delete(iifile);
        let _ = out.flush();
        std::process::exit(0);
    }
}

/// Original: `openEitherWay` (`mrc2tif.cpp:651`).
fn open_either_way(
    iifile: &mut ImodImageFile,
    iname: &[u8],
    progname: &[u8],
    oldcode: i32,
) -> Option<ImodFile> {
    thread_local! {
        static WARNED: Cell<bool> = const { Cell::new(false) };
    }
    iifile.filename = Some(String::from_utf8_lossy(iname).into_owned());
    if oldcode != 0 || unsafe { tiff_open_new(iifile) } != 0 {
        let fp = ImodFile::open(&String::from_utf8_lossy(iname), "wb");
        if fp.is_none() {
            // `perror("mrc2tif system message")`; see the note at the
            // other `perror` site about the ` (os error N)` suffix.
            let message = std::io::Error::last_os_error().to_string();
            let message = message
                .split(" (os error ")
                .next()
                .unwrap_or(message.as_str());
            let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                "mrc2tif system message: %s\n",
                &[CArg::Str(message)],
            ));
            // `mrc2tif.cpp:661-662`.
            exit_error(&c_format_bytes("Opening %s", &[CArg::Bytes(iname)]));
        }
        if oldcode == 0 && !WARNED.with(Cell::get) {
            WARNED.with(|warned| warned.set(true));
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "\nWARNING: %s - Not writing with libtiff, compression not available",
                &[CArg::Bytes(progname)],
            ));
        }
        fp
    } else {
        None
    }
}
