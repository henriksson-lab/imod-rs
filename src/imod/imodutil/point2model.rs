//! Translation of `IMOD/imodutil/point2model.c`: converts a simple point file
//! to a model.

use std::io::Write;

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_rewind, c_format_bytes, exit, fgetline, imod_backup_file, imod_prog_name,
    imod_usage_header, program_args,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_three_floats, pip_get_three_integers, pip_number_of_entries,
    pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_get_scale, mrc_head_read};
use crate::imod::libimod::ilabel::{imod_label_item_add, imod_label_new};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, IMOD_UNIT_NM, IOBJ_STRSIZE, Ipoint, Iref_image, imod_new,
    imod_new_contour, imod_new_object, imod_set_index, imod_set_ref_image,
};
use crate::imod::libimod::imodel_files::imod_write_file;
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_MCOLOR, IMOD_OBJFLAG_TIME, IMOD_OBJFLAG_USE_VALUE, IOBJ_EX_LABEL_SIZE,
    IOBJ_SYMF_FILL,
};
use crate::imod::libimod::ipoint::{imod_point_append, imod_point_set_size};
use crate::imod::libimod::istore::{
    GEN_STORE_FLOAT, GEN_STORE_VALUE1, Istore, istore_find_add_min_max1, istore_insert,
    istore_lookup,
};

/// `iobj.h:76-79`.
const IOBJ_SYM_CIRCLE: u8 = 0;
const IOBJ_SYM_SQUARE: u8 = 2;
const IOBJ_SYM_TRIANGLE: u8 = 3;

/// `B3DNINT(a)`: `(int)floor((a) + 0.5)`, the `0.5` a double.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `printf` with the source's format.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// Rust-only adapter: `PipReadOrParseOptions` takes the usage-header callback
/// as `void (*)(const char *)`; `imodUsageHeader` is that function.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Original: `main` (`point2model.c:25`).
pub fn point2model() {
    let argv = program_args();
    let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
    let mut point: Ipoint;
    let mut store = Istore::default();
    let mut line = [0u8; 1024];
    let mut open = 0;
    let mut zsort = 0;
    let mut scat = 0;
    let mut num_per_cont = 0;
    let mut from_zero = 0;
    let mut z_from_zero = 0;
    let mut err = 0;
    let mut nvals: i32;
    let mut nread: i32;
    let mut ob: i32;
    let mut co: i32;
    let mut line_num: i32;
    let mut after: usize;
    let mut needcont: i32;
    let num_offset: i32;
    let mut linelen: i32;
    let mut num_pts = 0;
    let mut num_conts = 0;
    let mut num_objs = 0;
    let mut sphere = 0;
    let mut circle = 0;
    let mut width2d = 0;
    let mut thickness = 0;
    let mut sym_type = 0;
    let mut tst1 = 0f32;
    let mut tst2 = 0f32;
    let mut xx = 0f32;
    let mut yy = 0f32;
    let mut zz = 0f32;
    let mut value = 0f32;
    let mut point_size = 0f32;
    let mut cont_time = 0f32;
    let mut ftemp = 0f32;
    let mut z_offset = 0f32;
    let mut num_colors = 0;
    let mut num_names = 0;
    let mut label_column = 0;
    let mut has_values = 0;
    let mut has_sizes = 0;
    let mut has_times = 0;
    let mut label_size = 0;
    let mut skip_lines = 0;
    let mut obj_num_only = 0;
    let mut use_default_size = 0;
    let mut display_flags = 0;
    let mut cont_values = 0;
    let maxes_entered: i32;
    let mut xmax_in = 0;
    let mut ymax_in = 0;
    let mut zmax_in = 0;
    let mut red: Vec<i32> = Vec::new();
    let mut green: Vec<i32> = Vec::new();
    let mut blue: Vec<i32> = Vec::new();
    let direct_scale: i32;
    let mut xscale = 1f32;
    let mut yscale = 1f32;
    let mut zscale = 1f32;
    let mut xtrans = 0f32;
    let mut ytrans = 0f32;
    let mut ztrans = 0f32;
    let mut model_zscale = 1f32;
    let mut mod_pixel = 0f32;
    let mut names: Vec<Vec<u8>> = Vec::new();
    let mut label_ptr: Option<usize>;

    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let mut filename: Vec<u8> = Vec::new();
    let mut imagename: Vec<u8> = Vec::new();
    let mut hdata = MrcHeader::default();
    let mut fpimage: Option<ImodFile> = None;
    let mut num_opt_args = 0;
    let mut num_non_opt_args = 0;

    /* Fallbacks from    ../manpages/autodoc2man 2 1 point2model  */
    let num_options = 31;
    let options: [&[u8]; 31] = [
        b"input:InputFile:FN:",
        b"output:OutputFile:FN:",
        b"open:OpenContours:B:",
        b"scat:ScatteredPoints:B:",
        b"number:PointsPerContour:I:",
        b"planar:PlanarContours:B:",
        b"zero:NumberedFromZero:B:",
        b"skip:SkipLinesAtStart:I:",
        b"object:ObjectNumbersOnly:B:",
        b"zcoord:ZCoordinatesFromZero:B:",
        b"sizes:PointSizes:B:",
        b"default:SetAsDefaultSize:B:",
        b"values:ValuesInLastColumn:I:",
        b"times:TimesForContours:B:",
        b"labels:LabelsStartInColumn:I:",
        b"font:LabelFontSize:I:",
        b"flags:DisplayFlags:I:",
        b"circle:CircleSize:I:",
        b"symbol:SymbolType:I:",
        b"sphere:SphereRadius:I:",
        b"width:LineWidthIn2D:I:",
        b"thick:LineThicknessIn3D:I:",
        b"color:ColorOfObject:ITM:",
        b"name:NameOfObject:CHM:",
        b"image:ImageForCoordinates:FN:",
        b"volume:VolumeSizeXYZ:IT:",
        b"pixel:PixelSpacingOfImage:FT:",
        b"origin:OriginOfImage:FT:",
        b"modpix:ModelPixelSize:F:",
        b"zscale:ZScaleOfModel:F:",
        b"help:usage:B:",
    ];

    /* Startup with fallback */
    pip_read_or_parse_options(
        argv_bytes.len() as i32,
        &argv_bytes,
        &options,
        num_options,
        progname.as_bytes(),
        2,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );
    if pip_get_boolean(b"usage", &mut err) == 0 {
        pip_print_help(progname.as_bytes(), 0, 1, 1);
        exit(0);
    }

    /* Get input and output files */
    if pip_get_in_out_file(b"InputFile", 0, &mut filename) != 0 {
        exit_error(b"No input file specified");
    }
    let fname = String::from_utf8_lossy(&filename).into_owned();
    let Some(mut infp) = ImodFile::open(&fname, "r") else {
        exit_error_fmt!("Error opening input file %s", CArg::Str(&fname))
    };
    if pip_get_in_out_file(b"OutputFile", 1, &mut filename) != 0 {
        exit_error(b"No output file specified");
    }
    let filename = String::from_utf8_lossy(&filename).into_owned();

    if pip_get_string(b"ImageForCoordinates", &mut imagename) == 0 {
        let iname = String::from_utf8_lossy(&imagename).into_owned();
        fpimage = ImodFile::open(&iname, "rb");
        let Some(fp) = fpimage.as_mut() else {
            exit_error_fmt!(
                "Could not open image file for coordinates: %s",
                CArg::Str(&iname)
            )
        };
        if mrc_head_read(fp, &mut hdata) != 0 {
            exit_error_fmt!("Reading header from %s", CArg::Str(&iname));
        }
        (xscale, yscale, zscale) = mrc_get_scale(&hdata);
    }

    direct_scale =
        2 - pip_get_three_floats(
            b"PixelSpacingOfImage",
            &mut xscale,
            &mut yscale,
            &mut zscale,
        ) - pip_get_three_floats(b"OriginOfImage", &mut xtrans, &mut ytrans, &mut ztrans);
    if direct_scale != 0 && fpimage.is_some() {
        exit_error(b"You cannot use -image together with -pixel or -origin");
    }

    err = pip_get_float(b"ZScaleOfModel", &mut model_zscale);
    err = pip_get_float(b"ModelPixelSize", &mut mod_pixel);
    if mod_pixel <= 0. && xscale != 1.0f32 {
        mod_pixel = xscale / 10.0f32;
    }

    err = pip_get_integer(b"PointsPerContour", &mut num_per_cont);
    err = pip_get_integer(b"SphereRadius", &mut sphere);
    err = pip_get_integer(b"CircleSize", &mut circle);
    err = pip_get_integer(b"SymbolType", &mut sym_type);
    sym_type = b3dmax!(0, b3dmin!(5, sym_type));
    err = pip_get_integer(b"LineWidthIn2D", &mut width2d);
    err = pip_get_integer(b"LineThicknessIn3D", &mut thickness);
    err = pip_get_boolean(b"OpenContours", &mut open);
    err = pip_get_boolean(b"ScatteredPoints", &mut scat);
    err = pip_get_boolean(b"PlanarContours", &mut zsort);
    err = pip_get_integer(b"ValuesInLastColumn", &mut has_values);
    err = pip_get_integer(b"LabelsStartInColumn", &mut label_column);
    err = pip_get_integer(b"LabelFontSize", &mut label_size);
    err = pip_get_boolean(b"ZCoordinatesFromZero", &mut z_from_zero);
    err = pip_get_boolean(b"PointSizes", &mut has_sizes);
    err = pip_get_boolean(b"TimesForContours", &mut has_times);
    err = pip_get_integer(b"DisplayFlags", &mut display_flags);
    err = pip_get_boolean(b"SetAsDefaultSize", &mut use_default_size);
    err = pip_get_boolean(b"ObjectNumbersOnly", &mut obj_num_only);
    err = pip_get_integer(b"SkipLinesAtStart", &mut skip_lines);
    maxes_entered =
        1 - pip_get_three_integers(b"VolumeSizeXYZ", &mut xmax_in, &mut ymax_in, &mut zmax_in);
    if z_from_zero != 0 {
        z_offset = 0.5;
    }
    has_values = b3dmax!(-1, b3dmin!(1, has_values));
    if has_values < 0 {
        has_values = 1;
        cont_values = 1;
    }
    has_times = b3dmax!(0, b3dmin!(1, has_times));
    err = pip_get_boolean(b"NumberedFromZero", &mut from_zero);
    num_offset = 1 - from_zero;
    if num_per_cont < 0 {
        exit_error(b"Number of points per contour must be positive or zero");
    }
    if open + scat > 1 {
        exit_error(b"Only one of -open or -scat may be entered");
    }

    // Get colors
    err = pip_number_of_entries(b"ColorOfObject", &mut num_colors);
    if num_colors != 0 {
        red = vec![0; num_colors as usize];
        green = vec![0; num_colors as usize];
        blue = vec![0; num_colors as usize];
        for co in 0..num_colors as usize {
            err = pip_get_three_integers(
                b"ColorOfObject",
                &mut red[co],
                &mut green[co],
                &mut blue[co],
            );
        }
    }

    // Get names
    err = pip_number_of_entries(b"NameOfObject", &mut num_names);
    if num_names != 0 {
        names = vec![Vec::new(); num_names as usize];
        for co in 0..num_names as usize {
            err = pip_get_string(b"NameOfObject", &mut names[co]);
        }
    }
    let _ = err;

    pip_done();

    for _ in 0..skip_lines {
        if fgetline(&mut infp, &mut line, 1024) <= 0 {
            exit_error(b"Reading lines to skip at start of file");
        }
    }

    // Read first line of file, figure out how many values
    if fgetline(&mut infp, &mut line, 1024) <= 0 {
        exit_error(b"Reading beginning of file");
    }

    let text_of = |line: &[u8]| -> String {
        let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
        String::from_utf8_lossy(&line[..end]).into_owned()
    };
    nvals = sscanf(
        &text_of(&line),
        "%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f",
        &mut [
            ScanArg::Flt(&mut tst1),
            ScanArg::Flt(&mut tst2),
            ScanArg::Flt(&mut xx),
            ScanArg::Flt(&mut yy),
            ScanArg::Flt(&mut zz),
            ScanArg::Flt(&mut point_size),
            ScanArg::Flt(&mut cont_time),
            ScanArg::Flt(&mut value),
        ],
    );
    if label_column > 2 {
        nvals = b3dmin!(nvals, label_column);
    }
    nvals -= has_values + has_sizes + has_times;
    if (nvals < 3 && (has_values != 0 || has_sizes != 0 || has_times != 0)) || nvals < 2 {
        exit_error_fmt!(
            "There must be at least %d values on the first line",
            CArg::Int(if has_values != 0 {
                (3 + has_values + has_sizes + has_times) as i64
            } else {
                2
            })
        );
    }
    nvals = b3dmin!(nvals, 5);
    if num_per_cont != 0 && obj_num_only == 0 && nvals > 3 {
        exit_error(b"The point file has contour numbers and the -number option cannot be used");
    }
    if zsort != 0 && obj_num_only == 0 && nvals > 3 {
        exit_error(b"The point file has contour numbers and the -planar option cannot be used");
    }

    b3d_rewind(&mut infp);

    let Some(mut imod) = imod_new() else {
        exit_error(b"Failed to get model structure")
    };

    // Set the image reference scaling.  The source tests `fpimage` again
    // after closing it (`point2model.c:406`), so whether an image was entered
    // is kept separately.
    let image_entered = fpimage.is_some();
    if let Some(fp) = fpimage.take() {
        imod_set_ref_image(&mut imod, &hdata);
        drop(fp);
    } else if direct_scale != 0 {
        imod.ref_image = Some(Iref_image {
            ctrans: Ipoint {
                x: xtrans,
                y: ytrans,
                z: ztrans,
            },
            cscale: Ipoint {
                x: xscale,
                y: yscale,
                z: zscale,
            },
            oscale: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            orot: Ipoint::default(),
            crot: Ipoint::default(),
            otrans: Ipoint::default(),
        });
    }

    ob = 0;
    co = 0;
    line_num = 0;
    imod.xmax = 0;
    imod.ymax = 0;
    imod.zmax = 0;
    imod.zscale = model_zscale;
    if mod_pixel > 0. {
        imod.pixsize = mod_pixel;
        imod.units = IMOD_UNIT_NM;
    }
    store.type_ = GEN_STORE_VALUE1;
    store.flags = GEN_STORE_FLOAT << 2;

    // To do: error check contour and object #'s, and they are numbered from 1.
    loop {
        // get line, done on EOF, skip blank line
        linelen = fgetline(&mut infp, &mut line, 1024);
        if linelen < 0 && linelen > -3 {
            break;
        }
        if linelen == 0 {
            continue;
        }

        let text = text_of(&line);
        if nvals <= 3 {
            zz = 0.;
            nread = sscanf(
                &text,
                "%f%*c%f%*c%f%*c%f%*c%f%*c%f",
                &mut [
                    ScanArg::Flt(&mut xx),
                    ScanArg::Flt(&mut yy),
                    ScanArg::Flt(&mut zz),
                    ScanArg::Flt(&mut point_size),
                    ScanArg::Flt(&mut cont_time),
                    ScanArg::Flt(&mut value),
                ],
            );
        } else if nvals == 4 || obj_num_only != 0 {
            nread = sscanf(
                &text,
                "%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f",
                &mut [
                    ScanArg::Flt(&mut ftemp),
                    ScanArg::Flt(&mut xx),
                    ScanArg::Flt(&mut yy),
                    ScanArg::Flt(&mut zz),
                    ScanArg::Flt(&mut point_size),
                    ScanArg::Flt(&mut cont_time),
                    ScanArg::Flt(&mut value),
                ],
            );
            if obj_num_only != 0 {
                ob = b3dnint!(ftemp) - num_offset;
            } else {
                co = b3dnint!(ftemp) - num_offset;
            }
        } else {
            nread = sscanf(
                &text,
                "%f%*c%d%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f",
                &mut [
                    ScanArg::Flt(&mut ftemp),
                    ScanArg::Int(&mut co),
                    ScanArg::Flt(&mut xx),
                    ScanArg::Flt(&mut yy),
                    ScanArg::Flt(&mut zz),
                    ScanArg::Flt(&mut point_size),
                    ScanArg::Flt(&mut cont_time),
                    ScanArg::Flt(&mut value),
                ],
            );
            ob = b3dnint!(ftemp) - num_offset;
            co -= num_offset;
        }
        zz -= z_offset;
        line_num += 1;

        // `strtok_r(i ? NULL : line, " ,\t", &labelPtr)` repeated labelColumn - 1
        // times, as glibc implements it: skip leading delimiters, end the token
        // at the next delimiter, and leave the save pointer just past it (or at
        // the terminating NUL when the token ends the string).
        label_ptr = None;
        let bytes = text.as_bytes();
        if label_column > 2 {
            let is_delim = |b: u8| b == b' ' || b == b',' || b == b'\t';
            let mut save = 0usize;
            let mut found = true;
            for _ in 0..label_column - 1 {
                let mut s = save;
                while s < bytes.len() && is_delim(bytes[s]) {
                    s += 1;
                }
                if s >= bytes.len() {
                    found = false;
                    break;
                }
                let mut e = s;
                while e < bytes.len() && !is_delim(bytes[e]) {
                    e += 1;
                }
                save = if e < bytes.len() { e + 1 } else { e };
            }
            if found {
                let mut p = save;
                while p < bytes.len() && bytes[p] == b' ' {
                    p += 1;
                }
                label_ptr = Some(p);
            }
        }

        // Skip line with no values
        if nread <= 0 {
            if linelen < 0 {
                break;
            }
            continue;
        }
        if b3dmin!(5 + has_values + has_sizes + has_times, nread)
            != nvals + has_values + has_sizes + has_times
            && !((cont_values != 0 || has_times != 0) && nread == nvals + has_sizes)
        {
            exit_error_fmt!(
                "Every line should have %d entries; line %d has %d",
                CArg::Int(
                    (nvals + (if cont_values != 0 { 0 } else { has_values }) + has_sizes) as i64
                ),
                CArg::Int(line_num as i64),
                CArg::Int(nread as i64)
            );
        }

        if ob < 0 || co < 0 {
            exit_error_fmt!(
                "Illegal object or contour number (object %d, contour %d at line %d",
                CArg::Int((ob + num_offset) as i64),
                CArg::Int((co + num_offset) as i64),
                CArg::Int(line_num as i64)
            );
        }

        // Add objects if needed to get to the current object
        if ob >= imod.obj.len() as i32 {
            for i in imod.obj.len()..=ob as usize {
                if imod_new_object(&mut imod) != 0 {
                    exit_error(b"Failed to add object to model");
                }
                let obj = &mut imod.obj[i];
                if open != 0 {
                    obj.flags |= IMOD_OBJFLAG_OPEN;
                }
                if scat != 0 {
                    obj.flags |= IMOD_OBJFLAG_SCAT | IMOD_OBJFLAG_OPEN;
                }
                if has_times != 0 {
                    obj.flags |= IMOD_OBJFLAG_TIME;
                }
                if display_flags & 1 != 0 {
                    obj.flags |= IMOD_OBJFLAG_USE_VALUE;
                }
                if display_flags & 2 != 0 {
                    obj.flags |= IMOD_OBJFLAG_MCOLOR;
                }
                num_objs += 1;
                obj.pdrawsize = b3dmax!(0, sphere);
                if circle > 0 {
                    obj.symsize = circle as u8;
                    obj.symbol = IOBJ_SYM_CIRCLE;
                    if sym_type != 0 {
                        if sym_type % 3 != 0 {
                            obj.symbol = if (sym_type % 3) == 1 {
                                IOBJ_SYM_SQUARE
                            } else {
                                IOBJ_SYM_TRIANGLE
                            };
                        }
                        if sym_type > 2 {
                            obj.symflags |= IOBJ_SYMF_FILL as u8;
                        }
                    }
                }
                if width2d > 0 {
                    obj.linewidth2 = width2d as u8;
                }
                if thickness > 0 {
                    obj.linewidth = thickness as u8;
                }
                if (i as i32) < num_colors {
                    obj.red = (red[i] as f64 / 255.) as f32;
                    obj.green = (green[i] as f64 / 255.) as f32;
                    obj.blue = (blue[i] as f64 / 255.) as f32;
                }
                if (i as i32) < num_names {
                    // strncpy(name, names[i], IOBJ_STRSIZE - 1) zero-pads, then
                    // the last byte is set to 0.
                    let src = &names[i];
                    let n = src
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(src.len())
                        .min(IOBJ_STRSIZE - 1);
                    obj.name[..n].copy_from_slice(&src[..n]);
                    obj.name[n..IOBJ_STRSIZE - 1].fill(0);
                    obj.name[IOBJ_STRSIZE - 1] = 0x00;
                }
                if label_size > 0 {
                    obj.extra[IOBJ_EX_LABEL_SIZE] = label_size as u32;
                }
            }
        }

        // Determine if a contour is needed: either the contour number is too high
        // or the number limit is reached or there is a change in Z
        needcont = 0;
        let obu = ob as usize;
        if co >= imod.obj[obu].cont.len() as i32 {
            needcont = 1;
        } else if (num_per_cont != 0
            && imod.obj[obu].cont[co as usize].pts.len() as i32 >= num_per_cont)
            || (zsort != 0 && b3dnint!(imod.obj[obu].cont[co as usize].pts[0].z) != b3dnint!(zz))
        {
            co += 1;
            needcont = 1;
        }

        if needcont != 0 {
            imod_set_index(&mut imod, ob, -1, -1);
            for _ in imod.obj[obu].cont.len() as i32..=co {
                if imod_new_contour(&mut imod).is_err() {
                    exit_error(b"Failed to add contour to model");
                }
                num_conts += 1;
            }
            if has_times != 0 {
                if nread < nvals + has_sizes + has_values + 1 {
                    exit_error_fmt!(
                        "The first point for a contour must have a time entry; line %d has only %d entries",
                        CArg::Int(line_num as i64),
                        CArg::Int(nread as i64)
                    );
                }

                // Save the value, set time from first entry
                if has_sizes == 0 {
                    value = cont_time;
                    cont_time = point_size;
                }
                let last = imod.obj[obu].cont.len() - 1;
                imod.obj[obu].cont[last].time = b3dnint!(cont_time);

                // Put value back in second entry or first: leave this is if there were no times
                if has_sizes != 0 {
                    cont_time = value;
                } else {
                    point_size = value;
                }
            }
        }

        point = Ipoint {
            x: xx,
            y: yy,
            z: zz,
        };
        imod.xmax = b3dmax!(imod.xmax, b3dnint!(xx as f64 + 10.));
        imod.ymax = b3dmax!(imod.ymax, b3dnint!(yy as f64 + 10.));
        imod.zmax = b3dmax!(imod.zmax, b3dnint!(zz as f64 + 1.));
        let cou = co as usize;
        let obj = &mut imod.obj[obu];
        if imod_point_append(&mut obj.cont[cou], point) == 0 {
            exit_error(b"Failed to add point to contour");
        }
        if has_sizes != 0 {
            if use_default_size != 0
                && sphere > 0
                && ((point_size - sphere as f32) as f64).abs() < 1.0e-5
            {
                point_size = -1.;
            }
            let last = obj.cont[cou].pts.len() as i32 - 1;
            imod_point_set_size(&mut obj.cont[cou], last, point_size);
        }

        // Add label if any
        if let Some(p) = label_ptr {
            if obj.cont[cou].label.is_none() {
                obj.cont[cou].label = Some(imod_label_new());
            }
            let last = obj.cont[cou].pts.len() as i32 - 1;
            imod_label_item_add(
                obj.cont[cou].label.as_mut().unwrap(),
                Some(&bytes[p..]),
                last,
            );
        }

        num_pts += 1;

        // take care of value for contour or point, only add one per contour
        if has_values != 0 {
            // Take it from second entry, or first if no sizes
            if has_sizes == 0 {
                cont_time = point_size;
            }
            store.value.set_f(cont_time);
            let err;
            if cont_values != 0 && istore_lookup(&obj.store, co).0.is_none() {
                if nread < nvals + has_sizes + has_times + 1 {
                    exit_error_fmt!(
                        "The first point for a contour must have a value entry; line %d has only %d entries",
                        CArg::Int(line_num as i64),
                        CArg::Int(nread as i64)
                    );
                }
                store.index.set_i(co);
                err = istore_insert(&mut obj.store, store);
            } else {
                store.index.set_i(obj.cont[cou].pts.len() as i32 - 1);
                err = istore_insert(&mut obj.cont[cou].store, store);
            }
            if err != 0 {
                exit_error(b"Failed to add general value");
            }
        }
        if linelen < 0 {
            break;
        }
    }
    after = 0;
    let _ = after;

    // Get the object min/max values set up
    if has_values != 0 {
        for obj in imod.obj.iter_mut() {
            if !obj.cont.is_empty() && istore_find_add_min_max1(obj) != 0 {
                exit_error(b"Adding min/max values to object");
            }
        }
    }

    // Attach image file's size as the max values, or use entered values
    if image_entered {
        imod.xmax = hdata.nx;
        imod.ymax = hdata.ny;
        imod.zmax = hdata.nz;
    }
    if maxes_entered != 0 {
        imod.xmax = xmax_in;
        imod.ymax = ymax_in;
        imod.zmax = zmax_in;
    }

    drop(infp);
    if imod_backup_file(&filename) != 0 {
        printf!(
            "Warning: %s - Failed to make old version of %s be a backup file\n",
            CArg::Str(&progname),
            CArg::Str(&filename)
        );
    }

    let Some(mut file) = ImodFile::open(&filename, "wb") else {
        exit_error_fmt!("Opening new model file %s", CArg::Str(&filename))
    };
    if imod_write_file(&imod, &mut file).is_err() {
        exit_error_fmt!("Writing model file %s", CArg::Str(&filename));
    }
    drop(file);
    printf!(
        "Model created with %d objects, %d contours, %d points\n",
        CArg::Int(num_objs as i64),
        CArg::Int(num_conts as i64),
        CArg::Int(num_pts as i64)
    );
    exit(0);
}
