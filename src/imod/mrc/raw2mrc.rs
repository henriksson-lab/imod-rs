//! Translation of `IMOD/mrc/raw2mrc.c`.

use std::io::Write as _;
use std::os::unix::ffi::OsStrExt as _;

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, SEEK_SET, b3d_fread, c_format_bytes, imod_backup_file, imod_prog_name,
    imod_usage_header, mrc_big_seek,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_non_option_arg,
    pip_get_string, pip_number_of_entries, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libiimod::iimage::{ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader,
    mrc_head_label, mrc_head_new, mrc_head_write, mrc_set_scale, mrc_write_slice,
};

/* input data types */
const DTYPE_CHAR: i32 = 1;
const DTYPE_BYTE: i32 = 1;
const DTYPE_UCHAR: i32 = 2;
const DTYPE_SHORT: i32 = 3;
const DTYPE_USHORT: i32 = 4;
const DTYPE_LONG: i32 = 5;
const DTYPE_ULONG: i32 = 6;
const DTYPE_FLOAT: i32 = 7;
const DTYPE_SBYTE: i32 = 10;
const DTYPE_DOUBLE: i32 = 11;
const DTYPE_LONG2SHORT: i32 = 15;
const DTYPE_ULONG2SHORT: i32 = 16;
const DTYPE_DOUBLE2SHORT: i32 = 17;

/* input image types */
const ITYPE_GREY: i32 = 1;
const ITYPE_COMPLEX: i32 = 2;
const ITYPE_RGB: i32 = 3;

/* compression */
const CTYPE_NONE: i32 = 0;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    // `PipPrintHelp` writes through Rust's stdout; the banner is on the C
    // stream, so hand it over before the help body follows it.
    let _ = ImodFile::Stdout.flush();
}

/// C `main` in `raw2mrc.c` (`raw2mrc.c:49`).
pub fn raw2mrc(arguments: &[String]) -> i32 {
    let mut hdata = MrcHeader::default();
    let mut i: i32 = 0;
    let mut zread: i32;
    let mut jread: i32;
    let mut num_in_by_opt: i32 = 0;
    let xysize: usize;
    let mut ind: usize;
    let mut intype: i32 = -1;
    let mut outtype: i32 = 0;
    let mut izsec: i32;
    let mut out_pixsize: i32 = -1;
    let compression: i32 = 0;
    let mut byteswap: i32 = 0;
    let mut x: i32 = 640;
    let mut y: i32 = 480;
    let mut z: i32 = 1;
    let nfiles: i32;
    let nsecs: i32;
    let mut flip: i32 = 0; /* flag whether to flip about X axis */
    let mut divide: i32 = 0; /* flag to divide unsigned short by 2 */
    let mut keep_ushort: i32 = 0; /* Flag to keep unsigned shorts */
    let mut hsize: i32 = 0; /* amount of data to skip */
    let mut convert: i32 = 0; /* Flag to convert long to short */
    let mut invert_stack: i32 = 0;
    let mut invert_files: i32 = 0;
    let mut zoffset: i32 = 0;
    let mut pixsize: i32 = 1;
    let mut pixel: f32;
    let mut pix_spacing: f32 = 0.;
    let mut z_pixel: f32 = 0.;
    let mut tmean: f64 = 0.0;
    let mut mean: f32 = 0.0;
    let mut min: f32 = 5e29;
    let mut max: f32 = -5e29;
    let mut allmax: f32 = -5e29;
    let mut allmin: f32 = 5e29;
    let mut val: i32;
    let mut num_opt_args: i32 = 0;
    let mut num_non_opt_args: i32 = 0;
    let mut input_files: Vec<Vec<u8>>;
    let mut temp_str: Vec<u8> = Vec::new();
    let mut out_file: Vec<u8> = Vec::new();
    // `char *progname = imodProgName(argv[0]);`
    let progname_owned = imod_prog_name(arguments.first().map_or("", String::as_str));
    let progname = progname_owned.as_bytes();

    // Fallbacks from    ../manpages/autodoc2man 2 1 raw2mrc
    let num_options: i32 = 18;
    let options: [&[u8]; 18] = [
        b"input:InputFile:FNM:",
        b"output:OutputFile:FN:",
        b"x:XSize:I:",
        b"y:YSize:I:",
        b"z:SectionsPerFile:I:",
        b"t:DataType:CH:",
        b"s:SwapBytes:B:",
        b"o:OffsetToImageData:I:",
        b"oz:SkipBetweenSections:I:",
        b"f:FlipInY:B:",
        b"i:InvertInZ:B:",
        b"iz:ReadInvertedInZ:B:",
        b"d:DivideBy2:B:",
        b"u:UnsignedOutput:B:",
        b"c:ConvertTo16BitInt:B:",
        b"p:PixelSpacing:F:",
        b"pz:ZPixelSpacing:F:",
        b"help:Usage:B:",
    ];

    /* Startup with fallback */
    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        num_options,
        progname,
        4,
        2,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );
    if pip_get_boolean(b"Usage", &mut i) == 0 {
        pip_print_help(progname, 0, 2, 1);
        return 0;
    }

    // get sizes, make sure they are there
    if pip_get_integer(b"XSize", &mut x) != 0 || pip_get_integer(b"YSize", &mut y) != 0 {
        exit_error(b"Image size in X and Y must be entered");
    }

    // Make sure an input file is entered somehow, get number by option
    pip_number_of_entries(b"InputFile", &mut num_in_by_opt);
    if num_in_by_opt == 0 && num_non_opt_args == 0 {
        exit_error(b"At least one input file must be entered");
    }

    // get output file from option, or last non-opt arg
    if pip_get_string(b"OutputFile", &mut out_file) != 0 {
        if num_non_opt_args == 0 || (num_in_by_opt == 0 && num_non_opt_args == 1) {
            exit_error(
                b"An output filename must be entered either with -input or as last \
                  non-option argument",
            );
        }
        pip_get_non_option_arg(num_non_opt_args - 1, &mut out_file);
        num_non_opt_args -= 1;
    }

    nfiles = num_in_by_opt + num_non_opt_args;
    // `inputFiles = B3DMALLOC(char *, nfiles);`
    input_files = Vec::new();
    if input_files.try_reserve_exact(nfiles as usize).is_err() {
        exit_error(b"Failed to allocate array for filenames");
    }

    // get the input files
    ind = 0;
    while ind < nfiles as usize {
        let mut name: Vec<u8> = Vec::new();
        if ind < num_in_by_opt as usize {
            pip_get_string(b"InputFile", &mut name);
        } else {
            pip_get_non_option_arg(ind as i32 - num_in_by_opt, &mut name);
        }
        input_files.push(name);
        ind += 1;
    }

    if pip_get_string(b"DataType", &mut temp_str) == 0 {
        intype = setintype(&temp_str, &mut pixsize, &mut outtype);
        // `free(tempStr);`
        drop(std::mem::take(&mut temp_str));
    }

    pip_get_integer(b"OffsetToImageData", &mut hsize);
    pip_get_integer(b"SkipBetweenSections", &mut zoffset);
    pip_get_integer(b"SectionsPerFile", &mut z);
    pip_get_boolean(b"SwapBytes", &mut byteswap);
    pip_get_boolean(b"FlipInY", &mut flip);
    // `ind = PipGetBoolean(...)` assigns an `int` return to the `size_t`;
    // only its zero-ness is tested below.
    ind = pip_get_boolean(b"ReadInvertedInZ", &mut invert_stack) as usize;
    if pip_get_boolean(b"InvertInZ", &mut invert_files) == 0 {
        if ind == 0 {
            exit_error(b"You cannot enter both -iz and -i");
        }
        invert_stack = invert_files;
    }

    pip_get_boolean(b"DivideBy2", &mut divide);
    pip_get_boolean(b"UnsignedOutput", &mut keep_ushort);
    pip_get_boolean(b"ConvertTo16BitInt", &mut convert);
    pip_get_float(b"PixelSpacing", &mut pix_spacing);
    pip_get_float(b"ZPixelSpacing", &mut z_pixel);
    pip_done();

    if convert != 0 {
        if intype < 0 || intype == DTYPE_LONG {
            intype = setintype(b"long2short", &mut pixsize, &mut outtype);
            out_pixsize = 2;
        } else if intype == DTYPE_ULONG {
            intype = setintype(b"ulong2short", &mut pixsize, &mut outtype);
            out_pixsize = 2;
        } else if intype == DTYPE_DOUBLE {
            intype = setintype(b"double2short", &mut pixsize, &mut outtype);
            out_pixsize = 2;
        } else {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "WARNING: %s - conversion option -c ignored with specified input type\n",
                &[CArg::Bytes(progname)],
            ));
        }
    }

    if divide != 0 && keep_ushort != 0 && (intype != DTYPE_DOUBLE2SHORT) {
        exit_error(b"You cannot divide by 2 and keep mode as unsigned");
    }

    if intype < 0 {
        intype = setintype(b"byte", &mut pixsize, &mut outtype);
    }
    if out_pixsize < 0 {
        out_pixsize = pixsize;
    }

    if keep_ushort != 0 {
        if intype == DTYPE_USHORT || intype == DTYPE_ULONG2SHORT || intype == DTYPE_DOUBLE2SHORT {
            outtype = MRC_MODE_USHORT;
        } else {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "WARNING: %s - option -u ignored with specified input type\n",
                &[CArg::Bytes(progname)],
            ));
        }
    }

    nsecs = z.wrapping_mul(nfiles);

    /* printf("nfiles = %d, argc = %d\n",nfiles, argc); */

    if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
        && imod_backup_file(&String::from_utf8_lossy(&out_file)) != 0
    {
        exit_error(b"Couldn't create backup file");
    }
    let Some(mut fout) = ii_fopen(&out_file, "wb") else {
        exit_error(&c_format_bytes(
            "Opening %s for output",
            &[CArg::Bytes(&out_file)],
        ));
    };

    // `xysize = (size_t)x * y;`
    xysize = (x as usize).wrapping_mul(y as i64 as usize);

    // `indata = (void *)malloc(pixsize * (xysize + (flip ? x : 0)));`
    let mut indata: Vec<u8> = Vec::new();
    let alloc_size = (pixsize as usize).wrapping_mul(xysize.wrapping_add(if flip != 0 {
        x as i64 as usize
    } else {
        0
    }));
    if indata.try_reserve_exact(alloc_size).is_err() {
        exit_error(b"Getting memory for data");
    }
    indata.resize(alloc_size, 0);

    /* 7/21/11: make header now so signed byte output is known */
    mrc_head_new(&mut hdata, x, y, nsecs, outtype);
    if pix_spacing > 0. {
        if z_pixel == 0. {
            z_pixel = pix_spacing;
        }
        mrc_set_scale(
            &mut hdata,
            pix_spacing as f64,
            pix_spacing as f64,
            z_pixel as f64,
        );
    }
    hdata.fp = Some(fout.clone());
    izsec = 0;

    let mut j: i32 = 0;
    while j < nfiles {
        jread = if invert_files != 0 {
            nfiles - (j + 1)
        } else {
            j
        };
        let Some(mut fin) = ImodFile::open(
            std::ffi::OsStr::from_bytes(&input_files[jread as usize]),
            "rb",
        ) else {
            exit_error(&c_format_bytes(
                "Opening %s for input",
                &[CArg::Bytes(&input_files[jread as usize])],
            ));
        };

        let mut k: i32 = 0;
        while k < z {
            tmean = 0.0;
            max = -5e29;
            min = 5e29;
            zread = if invert_stack != 0 { z - 1 - k } else { k };
            if mrc_big_seek(
                &mut fin,
                hsize.wrapping_add(zread.wrapping_mul(zoffset)),
                pixsize.wrapping_mul(x),
                y.wrapping_mul(zread),
                SEEK_SET,
            ) != 0
            {
                exit_error(&c_format_bytes(
                    "Seeking to data at section %d in file  %s",
                    &[
                        CArg::Int(zread as i64),
                        CArg::Bytes(&input_files[jread as usize]),
                    ],
                ));
            }
            if b3d_fread(&mut indata, pixsize as usize, xysize, &mut fin) != xysize {
                exit_error(&c_format_bytes(
                    "Reading data from file  %s",
                    &[CArg::Bytes(&input_files[jread as usize])],
                ));
            }
            if compression != 0 {
                rawdecompress(&mut indata, pixsize, xysize as i32, compression);
            }
            if byteswap != 0 {
                rawswap(&mut indata, pixsize, xysize as i32);
            }

            /* First do any needed conversions of the input types */
            // The source reinterprets the one buffer through typed pointers
            // (`usdata`, `sdata`, `ldata`, ...) and converts in place; here
            // each element is read and written back through its native-endian
            // bytes at the same offsets, so the same in-place narrowing holds.
            match intype {
                DTYPE_USHORT => {
                    if divide != 0 {
                        ind = 0;
                        while ind < xysize {
                            let us = u16::from_ne_bytes([indata[ind * 2], indata[ind * 2 + 1]]);
                            let s = ((us as i32) / 2) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    } else if keep_ushort == 0 {
                        ind = 0;
                        while ind < xysize {
                            let us = u16::from_ne_bytes([indata[ind * 2], indata[ind * 2 + 1]]);
                            let s = ((us as i32) - 32767) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    }
                }

                DTYPE_LONG => {
                    ind = 0;
                    while ind < xysize {
                        let l =
                            i32::from_ne_bytes(indata[ind * 4..ind * 4 + 4].try_into().unwrap());
                        let f = l as f32;
                        indata[ind * 4..ind * 4 + 4].copy_from_slice(&f.to_ne_bytes());
                        ind += 1;
                    }
                }

                DTYPE_DOUBLE => {
                    ind = 0;
                    while ind < xysize {
                        let d =
                            f64::from_ne_bytes(indata[ind * 8..ind * 8 + 8].try_into().unwrap());
                        let f = d as f32;
                        indata[ind * 4..ind * 4 + 4].copy_from_slice(&f.to_ne_bytes());
                        ind += 1;
                    }
                }

                DTYPE_ULONG => {
                    ind = 0;
                    while ind < xysize {
                        let ul =
                            u32::from_ne_bytes(indata[ind * 4..ind * 4 + 4].try_into().unwrap());
                        let f = ul as f32;
                        indata[ind * 4..ind * 4 + 4].copy_from_slice(&f.to_ne_bytes());
                        ind += 1;
                    }
                }

                DTYPE_ULONG2SHORT => {
                    if divide != 0 {
                        ind = 0;
                        while ind < xysize {
                            let ul = u32::from_ne_bytes(
                                indata[ind * 4..ind * 4 + 4].try_into().unwrap(),
                            );
                            let s = (ul / 2) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    } else if keep_ushort != 0 {
                        ind = 0;
                        while ind < xysize {
                            let ul = u32::from_ne_bytes(
                                indata[ind * 4..ind * 4 + 4].try_into().unwrap(),
                            );
                            let us = ul as u16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&us.to_ne_bytes());
                            ind += 1;
                        }
                    } else {
                        ind = 0;
                        while ind < xysize {
                            let ul = u32::from_ne_bytes(
                                indata[ind * 4..ind * 4 + 4].try_into().unwrap(),
                            );
                            // `(b3dInt16)(uldata[ind] - 32767)`: unsigned
                            // subtraction, then truncation to 16 bits.
                            let s = ul.wrapping_sub(32767) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    }
                }

                DTYPE_LONG2SHORT => {
                    if divide != 0 {
                        ind = 0;
                        while ind < xysize {
                            let l = i32::from_ne_bytes(
                                indata[ind * 4..ind * 4 + 4].try_into().unwrap(),
                            );
                            let s = (l / 2) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    } else {
                        ind = 0;
                        while ind < xysize {
                            let l = i32::from_ne_bytes(
                                indata[ind * 4..ind * 4 + 4].try_into().unwrap(),
                            );
                            let s = l as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    }
                }

                DTYPE_DOUBLE2SHORT => {
                    // A `double` to `short`/`unsigned short` conversion is
                    // undefined in C outside the target range; gcc on x86-64
                    // emits `cvttsd2si` to a 32-bit integer and keeps the low
                    // 16 bits, which `(d as i32) as i16` reproduces for every
                    // value inside the 32-bit range.  Rust's direct `as i16`
                    // would saturate instead.
                    if divide != 0 && keep_ushort != 0 {
                        ind = 0;
                        while ind < xysize {
                            let d = f64::from_ne_bytes(
                                indata[ind * 8..ind * 8 + 8].try_into().unwrap(),
                            );
                            let us = ((d / 2.) as i32) as u16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&us.to_ne_bytes());
                            ind += 1;
                        }
                    } else if keep_ushort != 0 {
                        ind = 0;
                        while ind < xysize {
                            let d = f64::from_ne_bytes(
                                indata[ind * 8..ind * 8 + 8].try_into().unwrap(),
                            );
                            let us = (d as i32) as u16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&us.to_ne_bytes());
                            ind += 1;
                        }
                    } else if divide != 0 {
                        ind = 0;
                        while ind < xysize {
                            let d = f64::from_ne_bytes(
                                indata[ind * 8..ind * 8 + 8].try_into().unwrap(),
                            );
                            let s = ((d / 2.) as i32) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    } else {
                        ind = 0;
                        while ind < xysize {
                            let d = f64::from_ne_bytes(
                                indata[ind * 8..ind * 8 + 8].try_into().unwrap(),
                            );
                            let s = (d as i32) as i16;
                            indata[ind * 2..ind * 2 + 2].copy_from_slice(&s.to_ne_bytes());
                            ind += 1;
                        }
                    }
                }

                /* DNM 12/27/01: fixed these two types on the PC by making the
                signed and unsigned types explicit */
                DTYPE_SBYTE => {
                    ind = 0;
                    while ind < xysize {
                        val = indata[ind] as i8 as i32;
                        val += 128;
                        indata[ind] = val as u8;
                        ind += 1;
                    }
                }

                _ => {}
            }

            /* Now compute min, mnax, mean based on output type */
            match outtype {
                MRC_MODE_SHORT => {
                    ind = 0;
                    while ind < xysize {
                        pixel = i16::from_ne_bytes([indata[ind * 2], indata[ind * 2 + 1]]) as f32;
                        tmean += pixel as f64;
                        min = if min < pixel { min } else { pixel };
                        max = if max > pixel { max } else { pixel };
                        ind += 1;
                    }
                }

                MRC_MODE_USHORT => {
                    ind = 0;
                    while ind < xysize {
                        pixel = u16::from_ne_bytes([indata[ind * 2], indata[ind * 2 + 1]]) as f32;
                        tmean += pixel as f64;
                        min = if min < pixel { min } else { pixel };
                        max = if max > pixel { max } else { pixel };
                        ind += 1;
                    }
                }

                MRC_MODE_FLOAT => {
                    ind = 0;
                    while ind < xysize {
                        pixel =
                            f32::from_ne_bytes(indata[ind * 4..ind * 4 + 4].try_into().unwrap());
                        tmean += pixel as f64;
                        min = if min < pixel { min } else { pixel };
                        max = if max > pixel { max } else { pixel };
                        ind += 1;
                    }
                }

                MRC_MODE_BYTE => {
                    ind = 0;
                    while ind < xysize {
                        pixel = indata[ind] as f32;
                        tmean += pixel as f64;
                        min = if min < pixel { min } else { pixel };
                        max = if max > pixel { max } else { pixel };
                        ind += 1;
                    }
                }

                _ => {}
            }

            if flip != 0 {
                let row = (out_pixsize as usize) * (x as usize);
                let scratch = (y as usize) * row;
                i = 0;
                while i < y / 2 {
                    let bdata = ((y - 1 - i) as usize) * row;
                    let bdata2 = (i as usize) * row;
                    indata.copy_within(bdata..bdata + row, scratch);
                    indata.copy_within(bdata2..bdata2 + row, bdata);
                    indata.copy_within(scratch..scratch + row, bdata2);
                    i += 1;
                }
            }
            hdata.amin = min;
            hdata.amax = max;
            if mrc_write_slice(&indata, &mut fout, &mut hdata, izsec, b'Z') != 0 {
                exit_error(b"Writing data to file");
            }
            izsec += 1;
            // `mean += (tmean / (float)xysize);` -- the quotient is double
            // and the sum is folded back into the `float` accumulator.
            mean = (mean as f64 + (tmean / (xysize as f32) as f64)) as f32;
            allmin = if min < allmin { min } else { allmin };
            allmax = if max > allmax { max } else { allmax };
            k += 1;
        }
        // `fclose(fin);`
        drop(fin);
        j += 1;
    }

    hdata.amin = allmin;
    hdata.amax = allmax;
    mean /= nsecs as f32;
    hdata.amean = mean;

    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "Min = %g, Max = %g, Mean = %g\n",
        &[
            CArg::Dbl(min as f64),
            CArg::Dbl(max as f64),
            CArg::Dbl(mean as f64),
        ],
    ));

    /* write out MRC header */
    /* DNM 11/5/02: change from raw writes of each element to calling
    library routines.  Added label.  1/17/04: eliminate unneeded rewind */
    mrc_head_label(&mut hdata, b"raw2mrc: Converted to mrc format.");
    mrc_head_write(&mut fout, &mut hdata);

    ii_fclose(&mut fout);

    0
}

/// C `rawdecompress` (`raw2mrc.c:424`).
fn rawdecompress(_indata: &mut [u8], _pixsize: i32, _xysize: i32, compression: i32) {
    if compression == 0 {
        return;
    }
}

/// C `rawswap` (`raw2mrc.c:434`).
fn rawswap(indata: &mut [u8], pixsize: i32, xysize: i32) {
    // The C declares `unsigned char u1, u2, u3, u4` at the top of the
    // function (`raw2mrc.c:436`) and assigns them inside each branch; the
    // bindings below are those, in the branch that uses them.
    let mut i: i32;

    if pixsize == 4 {
        i = 0;
        while i < xysize {
            let b = i as usize;
            /* DNM 12/18/01: changed i * 2 to i * 4 */
            let u1 = indata[b * 4];
            let u2 = indata[(b * 4) + 1];
            let u3 = indata[(b * 4) + 2];
            let u4 = indata[(b * 4) + 3];
            indata[b * 4] = u4;
            indata[(b * 4) + 1] = u3;
            indata[(b * 4) + 2] = u2;
            indata[(b * 4) + 3] = u1;
            i += 1;
        }
        return;
    }

    if pixsize == 2 {
        i = 0;
        while i < xysize {
            let b = i as usize;
            let u1 = indata[b * 2];
            let u2 = indata[(b * 2) + 1];
            indata[b * 2] = u2;
            indata[(b * 2) + 1] = u1;
            i += 1;
        }
    }
}

/// C `setintype` (`raw2mrc.c:468`).
fn setintype(stype: &[u8], size: &mut i32, otype: &mut i32) -> i32 {
    if stype == b"byte" {
        *size = 1;
        *otype = MRC_MODE_BYTE;
        return DTYPE_BYTE;
    }

    if stype == b"sbyte" {
        *size = 1;
        *otype = MRC_MODE_BYTE;
        return DTYPE_SBYTE;
    }

    if stype == b"rgb" {
        *size = 3;
        *otype = MRC_MODE_RGB;
        return DTYPE_BYTE;
    }

    if stype == b"short" {
        *size = 2;
        *otype = MRC_MODE_SHORT;
        return DTYPE_SHORT;
    }
    if stype == b"ushort" {
        *size = 2;
        *otype = MRC_MODE_SHORT;
        return DTYPE_USHORT;
    }

    /* DNM 12/27/01: long was missing from the quotes here */
    if stype == b"long" {
        *size = 4;
        *otype = MRC_MODE_FLOAT;
        return DTYPE_LONG;
    }
    if stype == b"ulong" {
        *size = 4;
        *otype = MRC_MODE_FLOAT;
        return DTYPE_ULONG;
    }

    if stype == b"long2short" {
        *size = 4;
        *otype = MRC_MODE_SHORT;
        return DTYPE_LONG2SHORT;
    }
    if stype == b"ulong2short" {
        *size = 4;
        *otype = MRC_MODE_SHORT;
        return DTYPE_ULONG2SHORT;
    }

    if stype == b"float" {
        *size = 4;
        *otype = MRC_MODE_FLOAT;
        return DTYPE_FLOAT;
    }

    if stype == b"double" {
        *size = 8;
        *otype = MRC_MODE_FLOAT;
        return DTYPE_DOUBLE;
    }

    if stype == b"double2short" {
        *size = 8;
        *otype = MRC_MODE_SHORT;
        return DTYPE_DOUBLE2SHORT;
    }

    -1
}
