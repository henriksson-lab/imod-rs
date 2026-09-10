//! Translation of `IMOD/mrc/tif2mrc.c`.
//!
//! The old `b3dtiff.h` reader remains at its C ABI.  This intentionally does
//! not substitute another TIFF decoder: palette and TVIPS behaviour must match IMOD.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{
    b3d_shift_bytes, imod_backup_file, imod_prog_name, mrc_big_seek, override_write_bytes,
    replace_file_arg_vec,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::iitif::{tiff_filter_warnings, tiff_set_mapping};
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    MRC_NLABELS, MrcHeader, mrc_head_label, mrc_head_new, mrc_head_write,
};
pub use crate::imod::mrc::tiff::{
    TfEntry, TfHeader, TfInfo, tiff_close_file, tiff_ifd_number, tiff_open_file, tiff_read_file,
    tiff_read_section,
};
use core::ffi::c_char;

const IITYPE_INT: i32 = 4;
const IITYPE_UINT: i32 = 5;
const IITYPE_USHORT: i32 = 3;
const IIFLAG_TVIPS_DATA: u32 = 2;
const IIFLAG_BYTES_SWAPPED: u32 = 4;

/// Original `usage` (`tif2mrc.c:32`).
unsafe fn usage(progname: *const c_char) -> ! {
    unsafe {
        libc::printf(c"Tif2mrc Version\n".as_ptr());
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
        libc::printf(c"\t-p #    Set pixel spacing in MRC header to given #\n".as_ptr());
        libc::printf(
            c"\t-P      Set pixel spacing in MRC header from resolution in TIFF file\n".as_ptr(),
        );
        libc::printf(
            c"\t-T file Output tilt angles from TVIPS input files to given file\n".as_ptr(),
        );
        libc::printf(c"\t-f      Read only first image of multi-page file\n".as_ptr());
        libc::printf(c"\t-o x,y  Set output file size in X and Y\n".as_ptr());
        libc::printf(c"\t-F #    Set value to fill areas with no image data to given #\n".as_ptr());
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
        let mut input = tifdata;
        let mut output = tifdata;
        for _ in 0..(xsize as usize * ysize as usize) {
            let value = if ntsc != 0 {
                (*input as f32 * 0.3
                    + *input.add(1) as f32 * 0.59
                    + *input.add(2) as f32 * 0.11
                    + 0.5) as u8
            } else {
                ((*input as i32 + *input.add(1) as i32 + *input.add(2) as i32) / 3) as u8
            };
            *output = value;
            input = input.add(3);
            output = output.add(1);
        }
    }
}

/// Original `expandIndexToRGB` (`tif2mrc.c:689`).
unsafe fn expand_index_to_rgb(datap: *mut *mut u8, iifile: *mut ImodImageFile, section: i32) {
    unsafe {
        if iifile.is_null() || (*iifile).colormap.is_null() {
            exit_error(c"Colormap data not read in properly.\n".as_ptr());
            return;
        }
        let mut size = (*iifile).nx as usize * (*iifile).ny as usize;
        if (*iifile).ury >= 0 {
            size = (*iifile).nx as usize * ((*iifile).ury + 1 - (*iifile).lly) as usize;
        }
        let out = libc::malloc(3 * size).cast::<u8>();
        if out.is_null() {
            exit_error(c"Unable to allocate memory for expanding RGB data.\n".as_ptr());
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
pub unsafe fn tif2mrc(argc: i32, argv: *mut *mut c_char) -> i32 {
    unsafe {
        if argc < 3 || argv.is_null() {
            if argv.is_null() {
                libc::exit(3);
            }
            usage(*argv);
        }
        let progname = imod_prog_name(*argv);
        let prefix = match std::ffi::CString::new(format!(
            "\nERROR: {} - ",
            core::ffi::CStr::from_ptr(progname).to_string_lossy(),
        )) {
            Ok(prefix) => prefix,
            Err(_) => return 1,
        };
        setExitPrefix(prefix.as_ptr());
        let mut makegray = 0;
        let mut use_ntsc = 0;
        let mut unsign = 0;
        let mut divide = 0;
        let mut keep_ushort = 0;
        let mut force_signed = 0;
        let mut invert_stack = 0;
        let mut read_first = 0;
        let mut pixel_size = 1.0_f32;
        let mut y_pixel_size = 1.0_f32;
        let mut pixel_entered = 0;
        let mut any_tif_pixel = 0;
        let mut fill_entered = 0;
        let mut user_fill = 0.0_f32;
        let mut mrcxsize = 0_i32;
        let mut mrcysize = 0_i32;
        let mut background = String::new();
        let mut tilt_file = String::new();
        let mut chunk_criterion = 100.0_f32;
        let mut names = Vec::<String>::new();
        let mut index = 1;
        while index < argc - 1 {
            let value = core::ffi::CStr::from_ptr(*argv.add(index as usize))
                .to_string_lossy()
                .into_owned();
            if !value.starts_with('-') || value == "-" {
                break;
            }
            match value.as_bytes().get(1).copied() {
                Some(b'g') => makegray = 1,
                Some(b'G') => {
                    makegray = 1;
                    use_ntsc = 1;
                }
                Some(b'u') => unsign = 1,
                Some(b'd') => divide = 1,
                Some(b'k') => keep_ushort = 1,
                Some(b's') => force_signed = 1,
                Some(b'i') => invert_stack = 1,
                Some(b'f') => read_first = 1,
                Some(b'P') => any_tif_pixel = 1,
                Some(b'p') | Some(b'F') | Some(b'o') | Some(b'b') | Some(b't') | Some(b'T')
                | Some(b'B') => {
                    index += 1;
                    if index >= argc - 1 {
                        usage(progname);
                    }
                    let option =
                        core::ffi::CStr::from_ptr(*argv.add(index as usize)).to_string_lossy();
                    match value.as_bytes()[1] {
                        b'p' => {
                            pixel_size = option.parse::<f32>().unwrap_or(1.0);
                            if pixel_size <= 0.0 {
                                pixel_size = 1.0;
                            } else {
                                pixel_entered = 1;
                            }
                            y_pixel_size = pixel_size;
                        }
                        b'F' => {
                            user_fill = option.parse::<f32>().unwrap_or(0.0);
                            fill_entered = 1;
                        }
                        b'o' => {
                            let mut pair =
                                option.split(|character| character == ',' || character == 'x');
                            mrcxsize = pair.next().and_then(|part| part.parse().ok()).unwrap_or(0);
                            mrcysize = pair.next().and_then(|part| part.parse().ok()).unwrap_or(0);
                        }
                        b'B' => {
                            override_write_bytes(if option.parse::<f32>().unwrap_or(0.0) != 0.0 {
                                1
                            } else {
                                0
                            });
                        }
                        b'b' => background = option.into_owned(),
                        b'T' => tilt_file = option.into_owned(),
                        b't' => chunk_criterion = option.parse().unwrap_or(100.0),
                        _ => {}
                    }
                }
                Some(b'm') => tiff_set_mapping(0),
                Some(b'h') => usage(progname),
                _ => {}
            }
            index += 1;
        }
        let mut arg_count = argc;
        let mut first_argument = index;
        let mut allocated = 0;
        let mut argument_vector = argv.cast::<*const c_char>() as *const *const c_char;
        if replace_file_arg_vec(
            &mut argument_vector,
            &mut arg_count,
            &mut first_argument,
            &mut allocated,
        ) != 0
        {
            return 1;
        }
        names.clear();
        for input_index in first_argument..arg_count - 1 {
            names.push(
                core::ffi::CStr::from_ptr(*argument_vector.add(input_index as usize))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if divide + unsign + keep_ushort + force_signed > 1 {
            exit_error(c"You must select only one of -u, -d, -k, or -s.\n".as_ptr());
        }
        if pixel_entered != 0 && any_tif_pixel != 0 {
            exit_error(c"You cannot enter both -p and -P\n".as_ptr());
        }
        if divide != 0 {
            unsign = 1;
        }
        if unsign != 0 {
            force_signed = 1;
        }
        tiff_filter_warnings();
        chunk_criterion *= 1024.0 * 1024.0;
        let output = core::ffi::CStr::from_ptr(*argument_vector.add((arg_count - 1) as usize))
            .to_string_lossy()
            .into_owned();
        if names.is_empty() {
            exit_error(c"Argument error: no output file specified\n".as_ptr());
        }
        let first_name = match std::ffi::CString::new(names[0].as_bytes()) {
            Ok(name) => name,
            Err(_) => return 1,
        };
        let mut first: TfInfo = core::mem::zeroed();
        if tiff_open_file(
            first_name.as_ptr().cast_mut(),
            c"rb".as_ptr().cast_mut(),
            &mut first,
            any_tif_pixel,
        ) != 0
        {
            exit_error(c"Couldn't open first TIFF file.\n".as_ptr());
        }
        if first.iifile.is_null() {
            let probe = tiff_read_file(first.fp, &mut first);
            if probe.is_null() {
                tiff_close_file(&mut first);
                exit_error(c"Couldn't read first TIFF file.\n".as_ptr());
            }
            libc::free(probe.cast());
        }
        let mut pix_size = 0;
        let mut mode = 0;
        manage_mode(
            &first,
            keep_ushort,
            force_signed,
            makegray,
            &mut pix_size,
            &mut mode,
        );
        let first_width = first.width;
        let first_height = first.length;
        if first_width < 1 || first_height < 1 {
            tiff_close_file(&mut first);
            exit_error(c"First TIFF file has invalid dimensions.\n".as_ptr());
        }
        if !first.iifile.is_null() && pixel_entered == 0 {
            pixel_size = (*first.iifile).xscale;
            y_pixel_size = (*first.iifile).yscale;
        }
        let first_pages = if read_first != 0 {
            1
        } else if !first.iifile.is_null() {
            (*first.iifile).nz
        } else {
            tiff_ifd_number(first.fp)
        };
        if names.len() == 1 && read_first == 0 && first_pages > 1 {
            libc::printf(c"Reading multi-paged TIFF file.\n".as_ptr());
            if !background.is_empty() {
                tiff_close_file(&mut first);
                exit_error(
                    c"Background subtraction not supported for multi-paged images.".as_ptr(),
                );
            }
            if mrcxsize != 0 {
                libc::printf(
                    c"Warning: output file size option ignored for multi-paged file\n".as_ptr(),
                );
            }
            libc::printf(
                c"Converting %d images size %d x %d\n".as_ptr(),
                first_pages,
                first_width,
                first_height,
            );
            let output_name = match std::ffi::CString::new(output.as_bytes()) {
                Ok(name) => name,
                Err(_) => {
                    tiff_close_file(&mut first);
                    return 1;
                }
            };
            if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null()
                && imod_backup_file(output_name.as_ptr()) != 0
            {
                tiff_close_file(&mut first);
                exit_error(c"Couldn't create backup file\n".as_ptr());
            }
            let mrcfp = libc::fopen(output_name.as_ptr(), c"wb".as_ptr());
            if mrcfp.is_null() {
                tiff_close_file(&mut first);
                return 1;
            }
            let mut hdata: MrcHeader = core::mem::zeroed();
            if mrc_head_new(&mut hdata, first_width, first_height, first_pages, mode) != 0
                || mrc_head_write(mrcfp, &mut hdata) != 0
            {
                libc::fclose(mrcfp);
                tiff_close_file(&mut first);
                return 1;
            }
            let mut min = 100000.0_f32;
            let mut max = -100000.0_f32;
            let mut mean = 0.0_f32;
            for section in 0..first_pages {
                let input_section = if invert_stack != 0 {
                    first_pages - 1 - section
                } else {
                    section
                };
                let mut data = tiff_read_section(first.fp, &mut first, input_section);
                if data.is_null() {
                    libc::fclose(mrcfp);
                    tiff_close_file(&mut first);
                    return 1;
                }
                if first.photometric_interpretation == 3 {
                    expand_index_to_rgb(&mut data, first.iifile, input_section);
                }
                if first.photometric_interpretation / 2 == 1 && makegray != 0 {
                    convertrgb(data, first_width, first_height, use_ntsc);
                }
                convert_long_to_float(data, first.iifile);
                mean += minmaxmean(
                    data,
                    mode,
                    unsign,
                    divide,
                    first_width,
                    first_height,
                    &mut min,
                    &mut max,
                );
                if mrc_big_seek(
                    mrcfp,
                    1024,
                    section * first_width,
                    first_height * pix_size,
                    libc::SEEK_SET,
                ) != 0
                {
                    libc::free(data.cast());
                    libc::fclose(mrcfp);
                    tiff_close_file(&mut first);
                    return 1;
                }
                if mode == MRC_MODE_BYTE && hdata.bytes_signed != 0 {
                    b3d_shift_bytes(data, data.cast(), first_width, first_height, 1, 1);
                }
                if libc::fwrite(
                    data.cast(),
                    (pix_size * first_width) as usize,
                    first_height as usize,
                    mrcfp,
                ) != first_height as usize
                {
                    libc::free(data.cast());
                    libc::fclose(mrcfp);
                    tiff_close_file(&mut first);
                    return 1;
                }
                libc::free(data.cast());
            }
            if !first.iifile.is_null() && pixel_entered == 0 {
                pixel_size = (*first.iifile).xscale;
                y_pixel_size = (*first.iifile).yscale;
            }
            hdata.mx = first_width;
            hdata.my = first_height;
            hdata.mz = first_pages;
            hdata.xlen = first_width as f32 * pixel_size;
            hdata.ylen = first_height as f32 * y_pixel_size;
            hdata.zlen = first_pages as f32 * pixel_size;
            if mode == MRC_MODE_RGB {
                hdata.amin = 0.0;
                hdata.amax = 255.0;
                hdata.amean = 128.0;
            } else {
                hdata.amin = min;
                hdata.amax = max;
                hdata.amean = mean / first_pages as f32;
                libc::printf(
                    c"Min = %g, Max = %g, Mean = %g\n".as_ptr(),
                    min as f64,
                    max as f64,
                    hdata.amean as f64,
                );
            }
            hdata.mode = mode;
            mrc_head_label(&mut hdata, b"tif2mrc: Converted to mrc format.");
            let result = mrc_head_write(mrcfp, &mut hdata);
            libc::fclose(mrcfp);
            tiff_close_file(&mut first);
            return if result == 0 { 0 } else { 1 };
        }
        tiff_close_file(&mut first);
        let sections = names.len() as i32;
        if sections < 1 {
            return 1;
        }
        if mrcxsize == 0 || mrcysize == 0 {
            mrcxsize = first_width;
            mrcysize = first_height;
        }
        let mut bgdata: *mut u8 = core::ptr::null_mut();
        let mut bg_bits = 0_i32;
        let mut bgxsize = 0_i32;
        let mut bgysize = 0_i32;
        if !background.is_empty() {
            let name = match std::ffi::CString::new(background.as_bytes()) {
                Ok(name) => name,
                Err(_) => return 1,
            };
            let mut tif: TfInfo = core::mem::zeroed();
            if tiff_open_file(
                name.as_ptr().cast_mut(),
                c"rb".as_ptr().cast_mut(),
                &mut tif,
                any_tif_pixel,
            ) != 0
            {
                return 1;
            }
            bgdata = tiff_read_file(tif.fp, &mut tif);
            bg_bits = tif.bits_per_sample;
            bgxsize = tif.width;
            bgysize = tif.length;
            let invalid = bgdata.is_null()
                || (bg_bits != 8 && bg_bits != 16)
                || tif.photometric_interpretation >= 2;
            tiff_close_file(&mut tif);
            if invalid {
                exit_error(c"Background file must be 8 or 16-bit grayscale\n".as_ptr());
            }
            if bg_bits == 8 {
                let mut bgmin = 255_u8;
                let mut bgmax = 0_u8;
                for y in 0..bgysize {
                    for x in 0..bgxsize {
                        let value = *bgdata.add((x + y * bgxsize) as usize);
                        bgmax = bgmax.max(value);
                        bgmin = bgmin.min(value);
                    }
                }
                let _ = bgmin;
                for y in 0..bgysize {
                    for x in 0..bgxsize {
                        let at = (x + y * bgxsize) as usize;
                        *bgdata.add(at) = bgmax - *bgdata.add(at);
                    }
                }
            }
        }
        let mut tiltfp: *mut libc::FILE = core::ptr::null_mut();
        if !tilt_file.is_empty() {
            let name = match std::ffi::CString::new(tilt_file.as_bytes()) {
                Ok(name) => name,
                Err(_) => {
                    libc::exit(1);
                }
            };
            if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null()
                && imod_backup_file(name.as_ptr()) != 0
            {
                exit_error(c"Opening tilt angle file\n".as_ptr());
            }
            tiltfp = libc::fopen(name.as_ptr(), c"w".as_ptr());
            if tiltfp.is_null() {
                exit_error(c"Opening tilt angle file\n".as_ptr());
            }
        }
        let output_name = match std::ffi::CString::new(output.as_bytes()) {
            Ok(name) => name,
            Err(_) => libc::exit(1),
        };
        if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null()
            && imod_backup_file(output_name.as_ptr()) != 0
        {
            exit_error(c"Couldn't create output backup file\n".as_ptr());
        }
        let mrcfp = libc::fopen(output_name.as_ptr(), c"wb".as_ptr());
        if mrcfp.is_null() {
            exit_error(c"Opening output MRC file\n".as_ptr());
        }
        let mut hdata: MrcHeader = core::mem::zeroed();
        if mrc_head_new(&mut hdata, mrcxsize, mrcysize, sections, mode) != 0 {
            libc::exit(1);
        }
        hdata.fp = mrcfp.cast();
        if mrc_head_write(mrcfp, &mut hdata) != 0 {
            libc::exit(1);
        }
        let mut min = 100000.0_f32;
        let mut max = -100000.0_f32;
        let mut mean = 0.0_f32;
        let mut out_section = 0_i32;
        let mut tvips_label = [0_i8; MRC_LABEL_SIZE + 1];
        for input_index in 0..names.len() {
            let selected = if invert_stack != 0 {
                names.len() - 1 - input_index
            } else {
                input_index
            };
            let name = match std::ffi::CString::new(names[selected].as_bytes()) {
                Ok(name) => name,
                Err(_) => {
                    libc::fclose(mrcfp);
                    return 1;
                }
            };
            let mut tif: TfInfo = core::mem::zeroed();
            if tiff_open_file(
                name.as_ptr().cast_mut(),
                c"rb".as_ptr().cast_mut(),
                &mut tif,
                any_tif_pixel,
            ) != 0
            {
                exit_error(c"Couldn't open input TIFF file.\n".as_ptr());
            }
            // `tif2mrc.c:380-381`: regular (non-multipage-fast-path) input
            // announces each opened TIFF before processing it.
            libc::printf(c"Opening %s for input\n".as_ptr(), name.as_ptr());
            libc::fflush(core::ptr::null_mut());
            if !tiltfp.is_null() {
                let mut tilt_angle = 0.0_f32;
                let mut label = [0_i8; MRC_LABEL_SIZE + 1];
                if manage_tvipsdata(tif.iifile, label.as_mut_ptr(), &mut tilt_angle) != 0 {
                    exit_error(c"There is no tilt angle value in this file\n".as_ptr());
                }
                if label[0] != 0 {
                    tvips_label = label;
                }
                libc::fprintf(tiltfp, c"%7.2f\n".as_ptr(), tilt_angle as f64);
            }
            let mut num_chunks = 1;
            let mut lines_per_chunk = tif.length;
            if !tif.iifile.is_null() && background.is_empty() {
                let bytes_per_pixel = if tif.photometric_interpretation == 2 {
                    3
                } else {
                    tif.bits_per_sample / 8
                };
                if (mrcxsize == 0 || (tif.width == mrcxsize && tif.length == mrcysize))
                    && tif.width as f32 * tif.length as f32 * bytes_per_pixel as f32
                        > chunk_criterion
                {
                    num_chunks = 1
                        + (tif.width as f32 * tif.length as f32 * bytes_per_pixel as f32
                            / chunk_criterion) as i32;
                    lines_per_chunk = (tif.length + num_chunks - 1) / num_chunks;
                }
            }
            let mut lines_done = 0;
            for chunk in 0..num_chunks {
                let mut nlines = tif.length;
                if num_chunks > 1 {
                    nlines = lines_per_chunk.min(tif.length - lines_done);
                    (*tif.iifile).lly = lines_done;
                    (*tif.iifile).ury = lines_done + nlines - 1;
                    lines_done += nlines;
                }
                let mut data = tiff_read_file(tif.fp, &mut tif);
                if data.is_null() {
                    exit_error(c"Reading TIFF image data\n".as_ptr());
                }
                let xsize = tif.width;
                let ysize = tif.length;
                if (tif.bits_per_sample == 16 && mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT)
                    || (tif.bits_per_sample == 32 && mode != MRC_MODE_FLOAT)
                    || (tif.photometric_interpretation / 2 == 1
                        && makegray == 0
                        && mode != MRC_MODE_RGB)
                    || (((tif.photometric_interpretation / 2 == 1 && makegray != 0)
                        || (tif.photometric_interpretation / 2 == 0 && tif.bits_per_sample == 8))
                        && mode != MRC_MODE_BYTE)
                {
                    exit_error(c"All files must have the same data type.\n".as_ptr());
                }
                if tif.photometric_interpretation == 3 {
                    expand_index_to_rgb(&mut data, tif.iifile, 0);
                }
                if tif.photometric_interpretation / 2 == 1 && makegray != 0 {
                    convertrgb(data, xsize, nlines, use_ntsc);
                }
                convert_long_to_float(data, tif.iifile);
                if !bgdata.is_null() {
                    if (mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT && bg_bits == 16)
                        || (mode != MRC_MODE_BYTE && bg_bits == 8)
                    {
                        exit_error(
                            c"Background data must have the same data type as the image files.\n"
                                .as_ptr(),
                        );
                    }
                    let xdo = bgxsize.min(xsize);
                    let ydo = bgysize.min(ysize);
                    if mode == MRC_MODE_BYTE {
                        for y in 0..ydo {
                            for x in 0..xdo {
                                let at = (x + y * xdo) as usize;
                                *data.add(at) =
                                    (*data.add(at) as i32 + *bgdata.add(at) as i32).min(255) as u8;
                            }
                        }
                    } else {
                        for y in 0..ydo {
                            for x in 0..xdo {
                                let at = (x + y * xdo) as usize;
                                *data.cast::<i16>().add(at) -= *bgdata.cast::<i16>().add(at);
                            }
                        }
                    }
                }
                let section_mean = minmaxmean(
                    data, mode, unsign, divide, xsize, nlines, &mut min, &mut max,
                );
                mean += section_mean * nlines as f32 / ysize as f32;
                if mode == MRC_MODE_BYTE && hdata.bytes_signed != 0 {
                    b3d_shift_bytes(data, data.cast(), xsize, nlines, 1, 1);
                }
                if xsize == mrcxsize && ysize == mrcysize {
                    if libc::fwrite(
                        data.cast(),
                        pix_size as usize,
                        (xsize * nlines) as usize,
                        mrcfp,
                    ) != (xsize * nlines) as usize
                    {
                        exit_error(c"Writing output MRC data\n".as_ptr());
                    }
                } else {
                    let fill = if fill_entered != 0 {
                        user_fill
                    } else {
                        section_mean
                    };
                    let fill_bytes: Vec<u8> = match mode {
                        MRC_MODE_BYTE => vec![if hdata.bytes_signed != 0 {
                            ((fill as i32 - 128) & 255) as u8
                        } else {
                            fill as u8
                        }],
                        MRC_MODE_SHORT => (fill as i16).to_ne_bytes().to_vec(),
                        MRC_MODE_USHORT => (fill as u16).to_ne_bytes().to_vec(),
                        MRC_MODE_FLOAT => fill.to_ne_bytes().to_vec(),
                        MRC_MODE_RGB => vec![
                            if fill_entered != 0 {
                                user_fill as u8
                            } else {
                                128
                            };
                            3
                        ],
                        _ => {
                            exit_error(c"Unsupported output MRC mode\n".as_ptr());
                            libc::exit(1);
                        }
                    };
                    let xoffset = (xsize - mrcxsize) / 2;
                    let yoffset = (ysize - mrcysize) / 2;
                    for y in 0..mrcysize {
                        for x in 0..mrcxsize {
                            if x + xoffset < 0
                                || x + xoffset >= xsize
                                || y + yoffset < 0
                                || y + yoffset >= ysize
                            {
                                if libc::fwrite(
                                    fill_bytes.as_ptr().cast(),
                                    pix_size as usize,
                                    1,
                                    mrcfp,
                                ) != 1
                                {
                                    exit_error(c"Writing padded output MRC data\n".as_ptr());
                                }
                            } else {
                                let offset = ((x + xoffset) + (y + yoffset) * xsize) as usize
                                    * pix_size as usize;
                                if libc::fwrite(
                                    data.add(offset).cast(),
                                    pix_size as usize,
                                    1,
                                    mrcfp,
                                ) != 1
                                {
                                    exit_error(c"Writing centered output MRC data\n".as_ptr());
                                }
                            }
                        }
                    }
                }
                libc::free(data.cast());
            }
            out_section += 1;
            tiff_close_file(&mut tif);
        }
        hdata.nx = mrcxsize;
        hdata.ny = mrcysize;
        hdata.mx = mrcxsize;
        hdata.my = mrcysize;
        hdata.mz = sections;
        hdata.xlen = mrcxsize as f32 * pixel_size;
        hdata.ylen = mrcysize as f32 * y_pixel_size;
        hdata.zlen = sections as f32 * pixel_size;
        hdata.mode = mode;
        if mode == MRC_MODE_RGB {
            hdata.amin = 0.;
            hdata.amax = 255.;
            hdata.amean = 128.;
        } else {
            hdata.amin = min;
            hdata.amax = max;
            hdata.amean = mean / sections as f32;
        }
        mrc_head_label(&mut hdata, b"tif2mrc: Converted to MRC format.");
        if tvips_label[0] != 0 && hdata.nlabl < MRC_NLABELS as i32 {
            for index in 0..MRC_LABEL_SIZE {
                hdata.labels[hdata.nlabl as usize][index] = tvips_label[index] as u8;
            }
            hdata.labels[hdata.nlabl as usize][MRC_LABEL_SIZE] = 0;
            hdata.nlabl += 1;
        }
        let result = mrc_head_write(mrcfp, &mut hdata);
        libc::fclose(mrcfp);
        if !bgdata.is_null() {
            libc::free(bgdata.cast());
        }
        if !tiltfp.is_null() {
            libc::fclose(tiltfp);
        }
        if result != 0 || out_section != sections {
            1
        } else {
            0
        }
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
            let mut image: ImodImageFile = core::mem::zeroed();
            image.type_ = IITYPE_USHORT;
            let mut tiff: TfInfo = core::mem::zeroed();
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
