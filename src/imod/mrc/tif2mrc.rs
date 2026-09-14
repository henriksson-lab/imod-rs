//! Translation of `IMOD/mrc/tif2mrc.c`.
//!
//! The old `b3dtiff.h` reader remains the default C-ABI path.  A deliberately
//! opt-in Rust decoder may be selected for its currently supported TIFF cases;
//! palette and TVIPS behavior continue to require the parity reader.
#![allow(dead_code, unused_variables, unused_assignments, unused_mut)]

use crate::imod::libcfshr::b3dutil::{
    b3d_fwrite, b3d_shift_bytes, imod_backup_file, imod_prog_name, mrc_big_seek,
    override_write_bytes, replace_file_arg_vec,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::iitif::{tiff_filter_warnings, tiff_set_mapping};
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    MRC_NLABELS, MrcHeader, mrc_head_label, mrc_head_new, mrc_head_write,
};
pub use crate::imod::mrc::tiff::{
    TfEntry, TfHeader, TfInfo, read_tiffentries, read_tiffheader, tiff_close_file, tiff_first_ifd,
    tiff_ifd_number, tiff_open_file, tiff_read_file, tiff_read_section,
};
use core::ffi::c_char;

unsafe extern "C" {
    static mut stdout: *mut libc::FILE;
}

/// `b3dtiff.h:68`.
const WIDTHINDEX: usize = 1;
/// `b3dtiff.h:69`.
const LENGTHINDEX: usize = 2;

const IITYPE_INT: i32 = 4;
const IITYPE_UINT: i32 = 5;
const IITYPE_USHORT: i32 = 3;
const IIFLAG_TVIPS_DATA: u32 = 2;
const IIFLAG_BYTES_SWAPPED: u32 = 4;

/// Original `usage` (`tif2mrc.c:32`).
unsafe fn usage(progname: *const c_char) -> ! {
    unsafe {
        // `tif2mrc.c:33`: `VERSION_NAME`, `__DATE__` and `__TIME__`.
        {
            use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
            use std::io::Write;
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "Tif2mrc Version %s %s %s\n",
                    &[
                        CArg::Str("5.2.17"),
                        CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_DATE),
                        CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_TIME),
                    ],
                )
                .as_bytes(),
            );
        }
        crate::imod::libcfshr::b3dutil::imod_copyright();
        libc::printf(
            c"Usage: %s [options] <tiff files...> <mrcfile>\n".as_ptr(),
            progname,
        );
        libc::printf(c"Options:\n".as_ptr());
        libc::printf(c"\t-g      Convert 24-bit RGB to 8-bit grayscale\n".as_ptr());
        libc::printf(
            c"\t-G      Convert 24-bit RGB to 8-bit grayscale with NTSC scaling\n".as_ptr(),
        );
        libc::printf(c"\t-u      Convert unsigned 16-bit values by subtracting 32768\n".as_ptr());
        libc::printf(c"\t-d      Convert unsigned 16-bit values by dividing by 2\n".as_ptr());
        libc::printf(
            c"\t-k      Keep unsigned 16-bit values; store in unsigned integer mode\n".as_ptr(),
        );
        libc::printf(
            c"\t-s      Store 16-bit as signed (mode 1) even if data are unsigned\n".as_ptr(),
        );
        libc::printf(c"\t-B #    Write bytes as unsigned (for # 0) or signed (for # 1)\n".as_ptr());
        libc::printf(c"\t-i      Invert order of sections in output stack\n".as_ptr());
        libc::printf(c"\t-p  #   Set pixel spacing in MRC header to given #\n".as_ptr());
        libc::printf(
            c"\t-P      Set pixel spacing in MRC header from resolution in TIFF file\n".as_ptr(),
        );
        libc::printf(
            c"\t-T file Output tilt angles from TVIPS input files to given file\n".as_ptr(),
        );
        libc::printf(c"\t-f      Read only first image of multi-page file\n".as_ptr());
        libc::printf(c"\t-o x,y  Set output file size in X and Y\n".as_ptr());
        libc::printf(c"\t-F  #   Set value to fill areas with no image data to given #\n".as_ptr());
        libc::printf(c"\t-b file Background subtract image in given file\n".as_ptr());
        libc::printf(
            c"\t-t #    Set criterion in megabytes for reading files in chunks\n".as_ptr(),
        );
        libc::printf(c"\t-m      Turn off file-to-memory mapping in libtiff\n".as_ptr());
        libc::exit(3)
    }
}

/// Original `manageMode` (`tif2mrc.c:634`).
unsafe fn manage_mode(
    tiff: *const TfInfo,
    keep_ushort: i32,
    force_signed: i32,
    makegray: i32,
    pix_size: *mut i32,
    mode: *mut i32,
) {
    unsafe {
        *pix_size = 1;
        if (*tiff).bits_per_sample == 16 {
            *mode = MRC_MODE_SHORT;
            *pix_size = 2;
        }
        if (*tiff).bits_per_sample == 32 {
            *mode = MRC_MODE_FLOAT;
            *pix_size = 4;
        }
        if *mode == MRC_MODE_SHORT
            && (keep_ushort != 0
                || (force_signed == 0
                    && !(*tiff).iifile.is_null()
                    && (*(*tiff).iifile).type_ == IITYPE_USHORT))
        {
            *mode = MRC_MODE_USHORT;
        }
        if (*tiff).photometric_interpretation / 2 == 1 && makegray == 0 {
            *mode = MRC_MODE_RGB;
            *pix_size = 3;
        }
    }
}

/// Original `convertrgb` (`tif2mrc.c:661`).
unsafe fn convertrgb(tifdata: *mut u8, xsize: i32, ysize: i32, ntsc: i32) {
    unsafe {
        let mut pixel: i32;
        let mut fpixel: f32;
        let mut input = tifdata;
        let mut output = tifdata;
        let xysize = xsize as usize * ysize as usize;
        if ntsc != 0 {
            for _ in 0..xysize {
                // `tif2mrc.c:672-674` accumulates through a `float fpixel`
                // while each term is computed in double (the weights are
                // double constants), so the running sum is rounded back to
                // float after every term.  Keeping this as one f32 expression
                // shifts occasional pixels by one count.
                fpixel = (*input as f64 * 0.3) as f32;
                input = input.add(1);
                fpixel = (fpixel as f64 + *input as f64 * 0.59) as f32;
                input = input.add(1);
                fpixel = (fpixel as f64 + *input as f64 * 0.11) as f32;
                input = input.add(1);
                *output = (fpixel + 0.5f32) as i32 as u8;
                output = output.add(1);
            }
        } else {
            for _ in 0..xysize {
                pixel = *input as i32;
                input = input.add(1);
                pixel += *input as i32;
                input = input.add(1);
                pixel += *input as i32;
                input = input.add(1);
                *output = (pixel / 3) as u8;
                output = output.add(1);
            }
        }
    }
}

/// Original `expandIndexToRGB` (`tif2mrc.c:689`).
unsafe fn expand_index_to_rgb(datap: *mut *mut u8, iifile: *mut ImodImageFile, section: i32) {
    unsafe {
        if iifile.is_null() || (*iifile).colormap.is_null() {
            exit_error(b"Colormap data not read in properly.\n");
            return;
        }
        let mut size = (*iifile).nx as usize * (*iifile).ny as usize;
        if (*iifile).ury >= 0 {
            size = (*iifile).nx as usize * ((*iifile).ury + 1 - (*iifile).lly) as usize;
        }
        let out = libc::malloc(3 * size).cast::<u8>();
        if out.is_null() {
            exit_error(b"Unable to allocate memory for expanding RGB data.\n");
            return;
        }
        let input = *datap;
        *datap = out;
        let map = (*iifile).colormap.add(768 * section as usize);
        for i in 0..size {
            let ind = *input.add(i) as usize;
            *out.add(3 * i) = *map.add(ind);
            *out.add(3 * i + 1) = *map.add(256 + ind);
            *out.add(3 * i + 2) = *map.add(512 + ind);
        }
    }
}

/// Original `convertLongToFloat` (`tif2mrc.c:716`).
unsafe fn convert_long_to_float(tifdata: *mut u8, iifile: *const ImodImageFile) {
    unsafe {
        if iifile.is_null() || ((*iifile).type_ != IITYPE_UINT && (*iifile).type_ != IITYPE_INT) {
            return;
        }
        let mut size = (*iifile).nx as usize * (*iifile).ny as usize;
        if (*iifile).ury >= 0 {
            size = (*iifile).nx as usize * ((*iifile).ury + 1 - (*iifile).lly) as usize;
        }
        if (*iifile).type_ == IITYPE_UINT {
            for i in 0..size {
                *tifdata.cast::<f32>().add(i) = *tifdata.cast::<u32>().add(i) as f32;
            }
        } else {
            for i in 0..size {
                *tifdata.cast::<f32>().add(i) = *tifdata.cast::<i32>().add(i) as f32;
            }
        }
    }
}

/// Original `minmaxmean` (`tif2mrc.c:738`).
unsafe fn minmaxmean(
    tifdata: *mut u8,
    mode: i32,
    unsign: i32,
    divide: i32,
    xsize: i32,
    ysize: i32,
    min: *mut f32,
    max: *mut f32,
) -> f32 {
    unsafe {
        let size = xsize as usize * ysize as usize;
        let mut mean = 0.0_f64;
        if !matches!(
            mode,
            MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT
        ) {
            return 0.0;
        }
        if mode == MRC_MODE_SHORT && unsign != 0 {
            for i in 0..size {
                let value = *tifdata.cast::<u16>().add(i);
                *tifdata.cast::<i16>().add(i) = if divide != 0 {
                    (value / 2) as i16
                } else {
                    (value as i32 - 32768) as i16
                };
            }
        }
        for i in 0..size {
            let value = match mode {
                MRC_MODE_BYTE => *tifdata.add(i) as f32,
                MRC_MODE_SHORT => *tifdata.cast::<i16>().add(i) as f32,
                MRC_MODE_USHORT => *tifdata.cast::<u16>().add(i) as f32,
                // C assigns this float through its `int pixel` local before
                // updating statistics, so preserve its truncating conversion.
                MRC_MODE_FLOAT => *tifdata.cast::<f32>().add(i) as i32 as f32,
                _ => unreachable!(),
            };
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
            mean += value as f64;
        }
        (mean / size as f64) as f32
    }
}

/// Original `manageTVIPSdata` (`tif2mrc.c:813`).
unsafe fn manage_tvipsdata(
    iifile: *const ImodImageFile,
    label: *mut c_char,
    tilt_angle: *mut f32,
) -> i32 {
    static mut LAST_AXIS: f32 = 0.0;
    static mut LAST_SPOT: i32 = 0;
    static mut LAST_BINNING: i32 = 0;
    static mut SAME_SPOT: i32 = -1;
    static mut SAME_BIN: i32 = -1;
    static mut SAME_AXIS: i32 = -1;
    unsafe {
        if iifile.is_null()
            || (*iifile).user_data.is_null()
            || (*iifile).user_count < 4260
            || (*iifile).user_flags & IIFLAG_TVIPS_DATA == 0
        {
            return 1;
        }
        let mut axis = core::ptr::read_unaligned((*iifile).user_data.add(3704).cast::<f32>());
        *tilt_angle = core::ptr::read_unaligned((*iifile).user_data.add(3564).cast::<f32>());
        let mut spot_float = core::ptr::read_unaligned((*iifile).user_data.add(3624).cast::<f32>());
        let mut binning = core::ptr::read_unaligned((*iifile).user_data.add(3944).cast::<i32>());
        if (*iifile).user_flags & IIFLAG_BYTES_SWAPPED != 0 {
            axis = f32::from_bits(axis.to_bits().swap_bytes());
            *tilt_angle = f32::from_bits((*tilt_angle).to_bits().swap_bytes());
            spot_float = f32::from_bits(spot_float.to_bits().swap_bytes());
            binning = binning.swap_bytes();
        }
        let spot = spot_float.round() as i32;
        if spot < 1 {
            SAME_SPOT = 0;
        }
        if binning < 1 {
            SAME_BIN = 0;
        }
        if SAME_SPOT < 0 {
            SAME_SPOT = 1;
        } else if SAME_SPOT > 0 && spot != LAST_SPOT {
            SAME_SPOT = 0;
        }
        if SAME_BIN < 0 {
            SAME_BIN = 1;
        } else if SAME_BIN > 0 && binning != LAST_BINNING {
            SAME_BIN = 0;
        }
        if SAME_AXIS < 0 {
            SAME_AXIS = 1;
        } else if SAME_AXIS > 0 && (axis - LAST_AXIS).abs() > 1.0e-5 {
            SAME_AXIS = 0;
        }
        LAST_AXIS = axis;
        LAST_SPOT = spot;
        LAST_BINNING = binning;
        axis -= 90.0;
        if axis < -180.0 {
            axis += 360.0;
        }
        if axis > 180.0 {
            axis -= 360.0;
        }
        if SAME_AXIS > 0 && SAME_SPOT > 0 && SAME_BIN > 0 {
            libc::snprintf(
                label,
                MRC_LABEL_SIZE,
                c"    Tilt axis angle = %.1f, binning = %d  spot = %d".as_ptr(),
                axis as f64,
                binning,
                spot,
            );
        } else if SAME_AXIS > 0 && SAME_BIN > 0 {
            libc::snprintf(
                label,
                MRC_LABEL_SIZE,
                c"    Tilt axis angle = %.1f, binning = %d".as_ptr(),
                axis as f64,
                binning,
            );
        } else if SAME_AXIS > 0 {
            libc::snprintf(
                label,
                MRC_LABEL_SIZE,
                c"    Tilt axis angle = %.1f".as_ptr(),
                axis as f64,
            );
        } else {
            *label = 0;
        }
        0
    }
}

/// Original `main` (`tif2mrc.c:58`).  It retains C argv semantics for binary wiring.
pub unsafe fn tif2mrc(mut argc: i32, argv: *mut *mut c_char) -> i32 {
    unsafe {
        let mut bgfp: crate::imod::libcfshr::b3dutil::ImodFile;
        let mut tiffp: crate::imod::libcfshr::b3dutil::ImodFile;
        let mut mrcfp: Option<crate::imod::libcfshr::b3dutil::ImodFile>;

        let mut tiff = TfInfo::default();
        let mut hdata = MrcHeader::default();

        let mut mode = 0_i32;
        let mut pix_size = 0_i32;
        let mut bgdata: *mut u8 = core::ptr::null_mut();
        let mut tifdata: *mut u8;
        let mut min: f32;
        let mut max: f32;
        let mut iarg: i32;
        let mut k: i32;
        let mut tmpdata: i32;
        let mut bg = 0_i32;
        let mut makegray = 0_i32;
        let mut fill_entered = 0_i32;
        let mut unsign = 0_i32;
        let mut divide = 0_i32;
        let mut keep_ushort = 0_i32;
        let mut force_signed = 0_i32;
        let mut read_first = 0_i32;
        let mut use_ntsc = 0_i32;
        let mut any_tif_pixel = 0_i32;
        let mut pixel_entered = 0_i32;
        let mut invert_stack = 0_i32;
        let mut xsize: i32;
        let mut ysize: i32;
        let mut iread: i32;
        let mut mrcxsize = 0_i32;
        let mut mrcysize = 0_i32;
        let mut user_fill = 0.0_f32;
        let mut fill_val = 0.0_f32;
        let mut mean: f32;
        let mut tmean: f32;
        let mut tilt_angle = 0.0_f32;
        let mut pixel_size = 1.0_f32;
        let mut y_pixel_size = 1.0_f32;
        let mut chunk_criterion = 100.0_f32;
        let mut bg_bits = 0_i32;
        let mut bgxsize = 0_i32;
        let mut bgysize = 0_i32;
        let mut xoffset: i32;
        let mut yoffset: i32;
        let mut xdo: usize;
        let mut ydo: i32;
        let first_file_ind: i32;
        let mut do_chunks: i32;
        let mut num_chunks: i32;
        let mut lines_per_chunk: i32;
        let mut nlines: i32;
        let mut lines_done: i32;
        let mut fill_ptr: *const u8;
        let mut byte_fill = [0_u8; 3];
        let mut short_fill: i16 = 0;
        let mut ushort_fill: u16 = 0;
        let mut label = [0_i8; MRC_LABEL_SIZE + 1];
        let mut tilt_file: *mut c_char = core::ptr::null_mut();
        let mut tiltfp: Option<crate::imod::libcfshr::b3dutil::ImodFile> = None;
        let openmode = c"rb".as_ptr().cast_mut();
        let mut bgfile: *mut c_char = core::ptr::null_mut();
        let progname_str =
            imod_prog_name(core::ffi::CStr::from_ptr(*argv).to_string_lossy().as_ref());
        // `usage` and the `sprintf` below still take the C string the source
        // passes; this keeps one copy alive for them.
        let progname_c = std::ffi::CString::new(progname_str.as_str()).unwrap_or_default();
        let progname = progname_c.as_ptr();
        let mut prefix = [0_i8; 100];
        // `tif2mrc.c:105-106`.
        libc::sprintf(prefix.as_mut_ptr(), c"\nERROR: %s - ".as_ptr(), progname);
        setExitPrefix(core::ffi::CStr::from_ptr(prefix.as_ptr()).to_bytes());
        // `exitError` is variadic in the source; the translated entry point takes a
        // single formatted string, so each call formats into this buffer first.
        let mut errmess = [0_i8; 512];

        xsize = 0;
        ysize = 0;
        mean = 0.;
        min = 100000.;
        max = -100000.;
        label[0] = 0x00;

        if argc < 3 {
            usage(progname);
        }

        iarg = 1;
        while iarg < argc - 1 {
            let arg = *argv.add(iarg as usize);
            if *arg == b'-' as c_char {
                match *arg.add(1) as u8 {
                    /* help */
                    b'h' => usage(progname),

                    /* convert rgb to gray scale */
                    b'g' => makegray = 1,

                    /* convert rgb to gray scale */
                    b'G' => {
                        makegray = 1;
                        use_ntsc = 1;
                    }

                    /* treat ints as unsigned */
                    b'u' => unsign = 1,

                    /* treat ints as unsigned and divide by 2*/
                    b'd' => divide = 1,

                    /* save as unsigned */
                    b'k' => keep_ushort = 1,

                    /* save unsigned as signed */
                    b's' => force_signed = 1,

                    /* Control signed nature of byte output */
                    b'B' => {
                        iarg += 1;
                        override_write_bytes(if libc::atof(*argv.add(iarg as usize)) != 0. {
                            1
                        } else {
                            0
                        });
                    }

                    /* Invert output stack */
                    b'i' => invert_stack = 1,

                    /* Insert pixel size in header */
                    b'p' => {
                        iarg += 1;
                        pixel_size = libc::atof(*argv.add(iarg as usize)) as f32;
                        if pixel_size <= 0. {
                            pixel_size = 1.;
                        } else {
                            pixel_entered = 1;
                        }
                        y_pixel_size = pixel_size;
                    }

                    /* Use resolution from TIFF file regardless of value */
                    b'P' => any_tif_pixel = 1,

                    /* read only first image */
                    b'f' => read_first = 1,

                    /* Define fill value */
                    b'F' => {
                        iarg += 1;
                        user_fill = libc::atof(*argv.add(iarg as usize)) as f32;
                        fill_entered = 1;
                    }

                    b'b' => {
                        iarg += 1;
                        bgfile = libc::strdup(*argv.add(iarg as usize));
                        bg = 1;
                    }

                    /* Set output size */
                    b'o' => {
                        iarg += 1;
                        libc::sscanf(
                            *argv.add(iarg as usize),
                            c"%d%*c%d".as_ptr(),
                            &mut mrcxsize as *mut i32,
                            &mut mrcysize as *mut i32,
                        );
                    }

                    b'm' => tiff_set_mapping(0),

                    b't' => {
                        iarg += 1;
                        chunk_criterion = libc::atof(*argv.add(iarg as usize)) as f32;
                    }

                    b'T' => {
                        iarg += 1;
                        tilt_file = libc::strdup(*argv.add(iarg as usize));
                    }

                    _ => {}
                }
            } else {
                break;
            }
            iarg += 1;
        }

        if (argc - 1) < (iarg + 1) {
            exit_error(b"Argument error: no output file specified");
        }
        let mut argvp = argv;
        let mut replaced = 0_i32;
        if replace_file_arg_vec(
            (&mut argvp as *mut *mut *mut c_char).cast::<*const *const c_char>(),
            &mut argc,
            &mut iarg,
            &mut replaced,
        ) != 0
        {
            libc::exit(1);
        }

        if divide + unsign + keep_ushort + force_signed > 1 {
            exit_error(b"You must select only one of -u, -d, -k, or -s.");
        }
        if divide != 0 {
            unsign = 1;
        }
        if unsign != 0 {
            force_signed = 1;
        }
        tiff_filter_warnings();
        chunk_criterion *= 1024. * 1024.;
        if pixel_entered != 0 && any_tif_pixel != 0 {
            exit_error(b"You cannot enter both -p and -P");
        }
        if !tilt_file.is_null() {
            imod_backup_file(
                core::ffi::CStr::from_ptr(tilt_file)
                    .to_string_lossy()
                    .as_ref(),
            );
            tiltfp = crate::imod::libcfshr::b3dutil::ImodFile::open(
                &core::ffi::CStr::from_ptr(tilt_file).to_string_lossy(),
                "w",
            );
            if tiltfp.is_none() {
                libc::snprintf(
                    errmess.as_mut_ptr(),
                    512,
                    c"Opening tilt angle file %s".as_ptr(),
                    tilt_file,
                );
                exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
            }
        }

        if iarg == (argc - 2) && read_first == 0 {
            /* check for multi-paged tiff file. */
            /* Open the TIFF file. */
            let tiff_pages: i32;

            if tiff_open_file(
                *argvp.add(iarg as usize),
                openmode,
                &mut tiff,
                any_tif_pixel,
            ) != 0
            {
                libc::snprintf(
                    errmess.as_mut_ptr(),
                    512,
                    c"Couldn't open %s.".as_ptr(),
                    *argvp.add(iarg as usize),
                );
                exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
            }

            tiffp = tiff.fp.clone().unwrap();
            if !tiff.iifile.is_null() {
                tiff_pages = (*tiff.iifile).nz;
            } else {
                tiff_pages = tiff_ifd_number(&mut tiffp);
            }
            if tiff_pages > 1 {
                libc::printf(c"Reading multi-paged TIFF file.\n".as_ptr());

                if bg != 0 {
                    exit_error(b"Background subtraction not supported for multi-paged images.");
                }

                if mrcxsize != 0 {
                    libc::printf(
                        c"Warning: output file size option ignored for multi-paged file\n".as_ptr(),
                    );
                }

                if tiff.iifile.is_null() {
                    read_tiffheader(&mut tiffp, &mut tiff.header);
                    crate::imod::libcfshr::b3dutil::b3d_rewind(&mut tiffp);
                    crate::imod::libcfshr::b3dutil::b3d_fread(
                        core::slice::from_raw_parts_mut(
                            (&mut tiff.header.byteorder as *mut i16).cast::<u8>(),
                            2,
                        ),
                        2,
                        1,
                        &mut tiffp,
                    );

                    tiff.header.first_ifd_offset = tiff_first_ifd(&mut tiffp) as i32;
                    crate::imod::libcfshr::b3dutil::b3d_rewind(&mut tiffp);
                    read_tiffentries(&mut tiffp, &mut tiff);
                }

                if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null()
                    && imod_backup_file(
                        core::ffi::CStr::from_ptr(*argvp.add((argc - 1) as usize))
                            .to_string_lossy()
                            .as_ref(),
                    ) != 0
                {
                    exit_error(b"Couldn't create backup file");
                }
                mrcfp = crate::imod::libcfshr::b3dutil::ImodFile::open(
                    &core::ffi::CStr::from_ptr(*argvp.add((argc - 1) as usize)).to_string_lossy(),
                    "wb",
                );
                if mrcfp.is_none() {
                    libc::perror(c"tif2mrc".as_ptr());
                    libc::snprintf(
                        errmess.as_mut_ptr(),
                        512,
                        c"Opening %s\n".as_ptr(),
                        *argvp.add((argc - 1) as usize),
                    );
                    exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
                }

                xsize = tiff.directory[WIDTHINDEX].value;
                ysize = tiff.directory[LENGTHINDEX].value;
                mrc_head_new(&mut hdata, xsize, ysize, tiff_pages, mode);
                mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);
                manage_mode(
                    &tiff,
                    keep_ushort,
                    force_signed,
                    makegray,
                    &mut pix_size,
                    &mut mode,
                );

                libc::printf(
                    c"Converting %d images size %d x %d\n".as_ptr(),
                    tiff_pages,
                    xsize,
                    ysize,
                );

                for section in 0..tiff_pages {
                    let in_section = if invert_stack != 0 {
                        tiff_pages - 1 - section
                    } else {
                        section
                    };

                    tifdata = tiff_read_section(&mut tiffp, &mut tiff, in_section);

                    if tifdata.is_null() {
                        libc::snprintf(
                            errmess.as_mut_ptr(),
                            512,
                            c"Failed to get image data for section %d".as_ptr(),
                            in_section,
                        );
                        exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
                    }

                    if tiff.photometric_interpretation == 3 {
                        expand_index_to_rgb(&mut tifdata, tiff.iifile, in_section);
                    }

                    /* convert RGB to gray scale */
                    if tiff.photometric_interpretation / 2 == 1 && makegray != 0 {
                        convertrgb(tifdata, xsize, ysize, use_ntsc);
                    }

                    /* Convert long ints to floats */
                    convert_long_to_float(tifdata, tiff.iifile);

                    mean += minmaxmean(
                        tifdata, mode, unsign, divide, xsize, ysize, &mut min, &mut max,
                    );

                    mrc_big_seek(
                        mrcfp.as_mut().unwrap(),
                        1024,
                        section * xsize,
                        ysize * pix_size,
                        libc::SEEK_SET,
                    );

                    if mode == 0 && hdata.bytes_signed != 0 {
                        b3d_shift_bytes(tifdata, tifdata.cast(), xsize, ysize, 1, 1);
                    }
                    b3d_fwrite(
                        core::slice::from_raw_parts(
                            tifdata.cast::<u8>(),
                            (pix_size * xsize) as usize * ysize as usize,
                        ),
                        (pix_size * xsize) as usize,
                        ysize as usize,
                        mrcfp.as_mut().unwrap(),
                    );

                    libc::free(tifdata.cast());
                }
                /* write more info to mrc header. 1/17/04 eliminate unneeded rewind */
                if !tiff.iifile.is_null() && pixel_entered == 0 {
                    pixel_size = (*tiff.iifile).xscale;
                    y_pixel_size = (*tiff.iifile).yscale;
                }
                hdata.nx = xsize;
                hdata.ny = ysize;
                hdata.mx = hdata.nx;
                hdata.my = hdata.ny;
                hdata.mz = hdata.nz;
                hdata.xlen = hdata.nx as f32 * pixel_size;
                hdata.ylen = hdata.ny as f32 * y_pixel_size;
                hdata.zlen = hdata.nz as f32 * pixel_size;
                if mode == MRC_MODE_RGB {
                    hdata.amax = 255.;
                    hdata.amean = 128.0;
                    hdata.amin = 0.;
                } else {
                    hdata.amax = max;
                    hdata.amean = mean / hdata.nz as f32;
                    hdata.amin = min;
                    libc::printf(
                        c"Min = %g, Max = %g, Mean = %g\n".as_ptr(),
                        min as f64,
                        max as f64,
                        hdata.amean as f64,
                    );
                }
                hdata.mode = mode;
                mrc_head_label(&mut hdata, b"tif2mrc: Converted to mrc format.");
                mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

                /* cleanup */
                drop(mrcfp.take());
                libc::exit(0);
            }
            tiff_close_file(&mut tiff);
        }

        /* read in bg file */
        if bg != 0 {
            if tiff_open_file(bgfile, openmode, &mut tiff, any_tif_pixel) != 0 {
                libc::snprintf(
                    errmess.as_mut_ptr(),
                    512,
                    c"Couldn't open %s.".as_ptr(),
                    bgfile,
                );
                exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
            }
            bgfp = tiff.fp.clone().unwrap();
            bgdata = tiff_read_file(&mut bgfp, &mut tiff);
            if bgdata.is_null() {
                libc::snprintf(errmess.as_mut_ptr(), 512, c"Reading %s.".as_ptr(), bgfile);
                exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
            }
            bg_bits = tiff.bits_per_sample;
            if (bg_bits != 8 && bg_bits != 16) || tiff.photometric_interpretation >= 2 {
                exit_error(b"Background file must be 8 or 16-bit grayscale");
            }

            bgxsize = tiff.directory[WIDTHINDEX].value;
            bgysize = tiff.directory[LENGTHINDEX].value;

            if bg_bits == 8 {
                max = 0.;
                min = 255.;

                for y in 0..bgysize {
                    for x in 0..bgxsize as usize {
                        tmpdata = *bgdata.add(x + (y * bgxsize) as usize) as i32;
                        if tmpdata as f32 > max {
                            max = tmpdata as f32;
                        }
                        if (tmpdata as f32) < min {
                            min = tmpdata as f32;
                        }
                    }
                }

                for y in 0..bgysize {
                    for x in 0..bgxsize as usize {
                        let at = x + (y * bgxsize) as usize;
                        *bgdata.add(at) = (max - *bgdata.add(at) as f32) as u8;
                    }
                }
            }

            tiff_close_file(&mut tiff);
        }

        /* Write out mrcheader */
        if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null()
            && imod_backup_file(
                core::ffi::CStr::from_ptr(*argvp.add((argc - 1) as usize))
                    .to_string_lossy()
                    .as_ref(),
            ) != 0
        {
            exit_error(b"Couldn't create backup file");
        }
        mrcfp = crate::imod::libcfshr::b3dutil::ImodFile::open(
            &core::ffi::CStr::from_ptr(*argvp.add((argc - 1) as usize)).to_string_lossy(),
            "wb",
        );
        if mrcfp.is_none() {
            libc::perror(c"tif2mrc".as_ptr());
            libc::snprintf(
                errmess.as_mut_ptr(),
                512,
                c"Opening %s".as_ptr(),
                *argvp.add((argc - 1) as usize),
            );
            exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
        }
        mrc_head_new(&mut hdata, xsize, ysize, argc - iarg - 1, mode);
        mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

        /* Loop through all the tiff files adding them to the MRC stack. */
        first_file_ind = iarg;
        while iarg < argc - 1 {
            iread = if invert_stack != 0 {
                first_file_ind + argc - 2 - iarg
            } else {
                iarg
            };

            /* Open the TIFF file. */
            if tiff_open_file(
                *argvp.add(iread as usize),
                openmode,
                &mut tiff,
                any_tif_pixel,
            ) != 0
            {
                libc::snprintf(
                    errmess.as_mut_ptr(),
                    512,
                    c"Couldn't open %s.".as_ptr(),
                    *argvp.add(iread as usize),
                );
                exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
            }
            libc::printf(
                c"Opening %s for input\n".as_ptr(),
                *argvp.add(iread as usize),
            );
            libc::fflush(stdout);
            tiffp = tiff.fp.clone().unwrap();
            k = manage_tvipsdata(tiff.iifile, label.as_mut_ptr(), &mut tilt_angle);
            if !tilt_file.is_null() {
                if k != 0 {
                    exit_error(b"There is no tilt angle value in this file");
                }
                use std::io::Write;
                let _ = tiltfp.as_mut().unwrap().write_all(
                    crate::imod::libcfshr::b3dutil::c_format(
                        "%7.2f\n",
                        &[crate::imod::libcfshr::b3dutil::CArg::Dbl(tilt_angle as f64)],
                    )
                    .as_bytes(),
                );
            }

            /* Decide whether to set up chunks */
            do_chunks = 0;
            num_chunks = 1;
            lines_per_chunk = 0;
            lines_done = 0;
            if !tiff.iifile.is_null() && bg == 0 {
                xsize = (*tiff.iifile).nx;
                ysize = (*tiff.iifile).ny;
                k = if tiff.photometric_interpretation == 2 {
                    3
                } else {
                    tiff.bits_per_sample / 8
                };
                if (mrcxsize == 0 || (xsize == mrcxsize && ysize == mrcysize))
                    && xsize as f64 * ysize as f64 * k as f64 > chunk_criterion as f64
                {
                    num_chunks = 1
                        + ((xsize as f64 * ysize as f64 * k as f64) / chunk_criterion as f64)
                            as i32;
                    do_chunks = 1;
                    lines_per_chunk = (ysize + num_chunks - 1) / num_chunks;
                    lines_done = 0;
                    libc::printf(
                        c"Reading file in %d chunks of %d lines\n".as_ptr(),
                        num_chunks,
                        lines_per_chunk,
                    );
                }
            }

            for chunk in 0..num_chunks {
                nlines = 0;
                if do_chunks != 0 {
                    nlines = if lines_per_chunk < ysize - lines_done {
                        lines_per_chunk
                    } else {
                        ysize - lines_done
                    };
                    (*tiff.iifile).lly = lines_done;
                    (*tiff.iifile).ury = lines_done + nlines - 1;
                    lines_done += nlines;
                }

                /* Read in tiff file */
                tifdata = tiff_read_file(&mut tiffp, &mut tiff);
                if tifdata.is_null() {
                    libc::snprintf(
                        errmess.as_mut_ptr(),
                        512,
                        c"Reading %s.".as_ptr(),
                        *argvp.add(iread as usize),
                    );
                    exit_error(core::ffi::CStr::from_ptr(errmess.as_ptr()).to_bytes());
                }

                xsize = tiff.directory[WIDTHINDEX].value;
                ysize = tiff.directory[LENGTHINDEX].value;
                if nlines == 0 {
                    nlines = ysize;
                }

                if chunk == 0 && first_file_ind == iarg {
                    if mrcxsize == 0 || mrcysize == 0 {
                        mrcxsize = xsize;
                        mrcysize = ysize;
                    }
                    manage_mode(
                        &tiff,
                        keep_ushort,
                        force_signed,
                        makegray,
                        &mut pix_size,
                        &mut mode,
                    );

                    /* Collect the pixel size the first time */
                    if !tiff.iifile.is_null() && pixel_entered == 0 {
                        pixel_size = (*tiff.iifile).xscale;
                        y_pixel_size = (*tiff.iifile).yscale;
                    }
                }

                if (tiff.bits_per_sample == 16 && mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT)
                    || (tiff.bits_per_sample == 32 && mode != MRC_MODE_FLOAT)
                    || (tiff.photometric_interpretation / 2 == 1
                        && makegray == 0
                        && mode != MRC_MODE_RGB)
                    || (((tiff.photometric_interpretation / 2 == 1 && makegray != 0)
                        || (tiff.photometric_interpretation / 2 == 0 && tiff.bits_per_sample == 8))
                        && mode != MRC_MODE_BYTE)
                {
                    exit_error(b"All files must have the same data type.");
                }

                if tiff.photometric_interpretation == 3 {
                    expand_index_to_rgb(&mut tifdata, tiff.iifile, 0);
                }

                /* convert RGB to gray scale */
                if tiff.photometric_interpretation / 2 == 1 && makegray != 0 {
                    convertrgb(tifdata, xsize, nlines, use_ntsc);
                }

                /* Convert long ints to floats */
                convert_long_to_float(tifdata, tiff.iifile);

                /* Correct for bg */
                if bg != 0 {
                    if (mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT && bg_bits == 16)
                        || (mode != MRC_MODE_BYTE && bg_bits == 8)
                    {
                        exit_error(
                            b"Background data must have  the same data type as the image files.",
                        );
                    }

                    xdo = if bgxsize < xsize { bgxsize } else { xsize } as usize;
                    ydo = if bgysize < ysize { bgysize } else { ysize };

                    if mode == MRC_MODE_BYTE {
                        for y in 0..ydo {
                            for x in 0..xdo {
                                let at = x + (y as usize * xdo);
                                tmpdata = *tifdata.add(at) as i32 + *bgdata.add(at) as i32;
                                if tmpdata > 255 {
                                    tmpdata = 255;
                                }
                                *tifdata.add(at) = tmpdata as u8;
                            }
                        }
                    } else {
                        let sptr = tifdata.cast::<i16>();
                        let bgshort = bgdata.cast::<i16>();
                        for y in 0..ydo {
                            for x in 0..xdo {
                                let at = x + (y as usize * xdo);
                                *sptr.add(at) = (*sptr.add(at)).wrapping_sub(*bgshort.add(at));
                            }
                        }
                    }
                }

                tmean = minmaxmean(
                    tifdata, mode, unsign, divide, xsize, nlines, &mut min, &mut max,
                );
                mean += (tmean * nlines as f32) / ysize as f32;

                if mode == 0 && hdata.bytes_signed != 0 {
                    b3d_shift_bytes(tifdata, tifdata.cast(), xsize, nlines, 1, 1);
                }

                if (xsize == mrcxsize) && (ysize == mrcysize) {
                    /* Write out mrc file */
                    b3d_fwrite(
                        core::slice::from_raw_parts(
                            tifdata.cast::<u8>(),
                            (pix_size * xsize) as usize * nlines as usize,
                        ),
                        (pix_size * xsize) as usize,
                        nlines as usize,
                        mrcfp.as_mut().unwrap(),
                    );
                } else {
                    libc::printf(
                        c"WARNING: tif2mrc - File %s not same size.\n".as_ptr(),
                        *argvp.add(iread as usize),
                    );

                    /* Unequal sizes: set the fill value and pointer */
                    fill_val = if fill_entered != 0 { user_fill } else { tmean };
                    fill_ptr = core::ptr::null();
                    match mode {
                        MRC_MODE_BYTE => {
                            byte_fill[0] = fill_val as u8;
                            if hdata.bytes_signed != 0 {
                                byte_fill[0] = ((fill_val as i32 - 128) & 255) as u8;
                            }
                            fill_ptr = byte_fill.as_ptr();
                        }
                        MRC_MODE_RGB => {
                            let value = if fill_entered != 0 {
                                user_fill as i32 as u8
                            } else {
                                128
                            };
                            byte_fill[0] = value;
                            byte_fill[1] = value;
                            byte_fill[2] = value;
                            fill_ptr = byte_fill.as_ptr();
                        }
                        MRC_MODE_SHORT => {
                            short_fill = fill_val as i16;
                            fill_ptr = (&short_fill as *const i16).cast();
                        }
                        MRC_MODE_USHORT => {
                            ushort_fill = fill_val as u16;
                            fill_ptr = (&ushort_fill as *const u16).cast();
                        }
                        MRC_MODE_FLOAT => {
                            fill_ptr = (&fill_val as *const f32).cast();
                        }
                        _ => {}
                    }

                    /* Output centered data */
                    yoffset = (ysize - mrcysize) / 2;
                    xoffset = (xsize - mrcxsize) / 2;
                    for y in 0..mrcysize {
                        if y + yoffset < 0 || y + yoffset >= ysize {
                            /* Do fill lines */
                            for _x in 0..mrcxsize {
                                b3d_fwrite(
                                    core::slice::from_raw_parts(
                                        fill_ptr.cast::<u8>(),
                                        pix_size as usize,
                                    ),
                                    pix_size as usize,
                                    1,
                                    mrcfp.as_mut().unwrap(),
                                );
                            }
                        } else {
                            /* Fill left edge if necessary, write data, fill right if needed */
                            k = xoffset;
                            while k < 0 {
                                b3d_fwrite(
                                    core::slice::from_raw_parts(
                                        fill_ptr.cast::<u8>(),
                                        pix_size as usize,
                                    ),
                                    pix_size as usize,
                                    1,
                                    mrcfp.as_mut().unwrap(),
                                );
                                k += 1;
                            }
                            xdo = (pix_size
                                * (if xoffset > 0 { xoffset } else { 0 } + (y + yoffset) * xsize))
                                as usize;
                            b3d_fwrite(
                                core::slice::from_raw_parts(
                                    tifdata.add(xdo).cast::<u8>(),
                                    pix_size as usize
                                        * (if xsize < mrcxsize { xsize } else { mrcxsize })
                                            as usize,
                                ),
                                pix_size as usize,
                                (if xsize < mrcxsize { xsize } else { mrcxsize }) as usize,
                                mrcfp.as_mut().unwrap(),
                            );
                            k = 0;
                            while k < mrcxsize - xsize + xoffset {
                                b3d_fwrite(
                                    core::slice::from_raw_parts(
                                        fill_ptr.cast::<u8>(),
                                        pix_size as usize,
                                    ),
                                    pix_size as usize,
                                    1,
                                    mrcfp.as_mut().unwrap(),
                                );
                                k += 1;
                            }
                        }
                    }
                }
                if !tifdata.is_null() {
                    libc::free(tifdata.cast());
                }
                tifdata = core::ptr::null_mut();
            }

            tiff_close_file(&mut tiff);
            iarg += 1;
        }

        /* write more info to mrc header. 1/17/04 eliminate unneeded rewind */
        hdata.nx = mrcxsize;
        hdata.ny = mrcysize;
        hdata.mx = hdata.nx;
        hdata.my = hdata.ny;
        hdata.mz = hdata.nz;
        hdata.xlen = hdata.nx as f32 * pixel_size;
        hdata.ylen = hdata.ny as f32 * y_pixel_size;
        hdata.zlen = hdata.nz as f32 * pixel_size;
        if mode == MRC_MODE_RGB {
            hdata.amax = 255.;
            hdata.amean = 128.0;
            hdata.amin = 0.;
        } else {
            hdata.amax = max;
            hdata.amean = mean / hdata.nz as f32;
            hdata.amin = min;
            libc::printf(
                c"Min = %g, Max = %g, Mean = %g\n".as_ptr(),
                min as f64,
                max as f64,
                hdata.amean as f64,
            );
        }
        hdata.mode = mode;
        mrc_head_label(&mut hdata, b"tif2mrc: Converted to MRC format.");
        if label[0] != 0x00 {
            k = libc::strlen(label.as_ptr()) as i32;
            while k < MRC_LABEL_SIZE as i32 {
                label[k as usize] = b' ' as i8;
                k += 1;
            }
            if hdata.nlabl < MRC_NLABELS as i32 {
                for index in 0..MRC_LABEL_SIZE {
                    hdata.labels[hdata.nlabl as usize][index] = label[index] as u8;
                }
                hdata.labels[hdata.nlabl as usize][MRC_LABEL_SIZE] = 0;
                hdata.nlabl += 1;
            }
        }
        mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

        /* cleanup */
        if mrcfp.is_some() {
            drop(mrcfp.take());
        }
        if !tilt_file.is_null() {
            drop(tiltfp.take());
        }
        libc::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converts_rgb_in_place() {
        let mut values = [30_u8, 60, 90, 9, 12, 15];
        unsafe { convertrgb(values.as_mut_ptr(), 2, 1, 0) };
        assert_eq!(&values[..2], &[60, 12]);
    }
    #[test]
    fn ntsc_weighting_rounds_between_terms_like_the_source() {
        // `tif2mrc.c:672-674` stores each partial sum back into a `float`, so
        // these triples come out one count above what a single f32 expression
        // gives (3, 10, 15, 12).  Verified against the native binary.
        let mut values = [0_u8, 5, 5, 0, 15, 15, 0, 19, 39, 0, 21, 1];
        unsafe { convertrgb(values.as_mut_ptr(), 4, 1, 1) };
        assert_eq!(&values[..4], &[4, 11, 16, 13]);
    }
    #[test]
    fn unsigned_short_conversion_and_stats() {
        let mut values = [0_u16, 65535];
        let mut min = 1.0e5;
        let mut max = -1.0e5;
        let mean = unsafe {
            minmaxmean(
                values.as_mut_ptr().cast(),
                MRC_MODE_SHORT,
                1,
                0,
                2,
                1,
                &mut min,
                &mut max,
            )
        };
        assert_eq!(values, [32768, 32767]);
        assert_eq!((min, max, mean), (-32768., 32767., -0.5));
    }

    #[test]
    fn unsigned_16_bit_tiff_selects_unsigned_short_mode() {
        unsafe {
            let mut image = ImodImageFile::default();
            image.type_ = IITYPE_USHORT;
            let mut tiff = TfInfo::default();
            tiff.bits_per_sample = 16;
            tiff.iifile = &raw mut image;
            let mut pixel_size = 0;
            let mut mode = 0;
            manage_mode(&tiff, 0, 0, 0, &mut pixel_size, &mut mode);
            assert_eq!((pixel_size, mode), (2, MRC_MODE_USHORT));
        }
    }

    #[test]
    fn float_statistics_follow_c_integer_pixel_conversion() {
        let mut values = [1.9_f32, -2.4_f32];
        let mut min = 100000.0;
        let mut max = -100000.0;
        let mean = unsafe {
            minmaxmean(
                values.as_mut_ptr().cast(),
                MRC_MODE_FLOAT,
                0,
                0,
                2,
                1,
                &mut min,
                &mut max,
            )
        };
        assert_eq!((min, max, mean), (-2.0, 1.0, -0.5));
    }

    #[test]
    fn rgb_statistics_leave_source_minimum_and_maximum_unchanged() {
        let mut values = [30_u8, 60, 90];
        let mut min = 100000.0;
        let mut max = -100000.0;
        let mean = unsafe {
            minmaxmean(
                values.as_mut_ptr(),
                MRC_MODE_RGB,
                0,
                0,
                1,
                1,
                &mut min,
                &mut max,
            )
        };
        assert_eq!((min, max, mean), (100000.0, -100000.0, 0.0));
    }
}
