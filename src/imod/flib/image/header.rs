//! Translation of `IMOD/flib/image/header.f90`.
//!
//! `header.f90` is a Fortran main program, so its single executable program
//! unit maps directly to [`header`].  The lower-level Fortran `iiunit` calls
//! used by the original are represented by the already-translated native image
//! interface; no second unit-file registry is introduced here.
#![allow(dead_code, unused_variables)]

use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_number_of_sections,
    adoc_get_section_name, adoc_open_image_metadata,
};
use crate::imod::libcfshr::b3dutil::extra_is_nbytes_and_flags;
use crate::imod::libcfshr::extraheader::{
    get_extra_header_items, get_extra_header_value, get_fei_ext_head_angle_scale,
};
use crate::imod::libiimod::iimage::{
    ImodImageFile, ii_allow_multi_volume, ii_delete, ii_fill_mrc_header, ii_fopen_volume,
    ii_lookup_file_from_fp, ii_open,
};
use crate::imod::libiimod::iitif::{
    tiff_get_max_eer_super_res, tiff_set_eer_read_properties, tiff_set_string_tag_to_print,
};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_get_scale, mrc_read_extra_header};
use crate::imod::libiimod::unit_fileio::{
    ialbrief_, iiu_close, iiu_file_info, iiu_open, iiu_ret_num_volumes, iiu_volume_open,
    iiualtprint_,
};
use crate::imod::libiimod::unit_header::{
    iiu_ret_delta, iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_imod_flags,
    iiu_ret_labels, iiu_ret_mrc_version, iiu_ret_num_extended, iiu_ret_origin, iiu_ret_rms,
};
use std::ffi::{CStr, CString};
use std::io::{self, Write};

/// Original program `header` (`header.f90:9`).
///
/// This preserves the program's PIP option spellings and its distinction
/// between the terse machine-readable switches and ordinary header output.
/// The image dispatch layer owns file-format probing exactly as `imopen` did
/// in the Fortran source.
pub fn header() {
    let type_name = [
        "Tilt angles",
        "Piece coordinates",
        "Stage positions",
        "Magnifications",
        "Intensities",
        "Exposure doses",
    ];
    let extract_com = [
        "extracttilts",
        "extractpieces",
        "extracttilts -stage",
        "extracttilts -mag",
        "extracttilts -int",
        "extracttilts -exp",
    ];
    let brief_name = [
        "Tilts",
        "Piece coords",
        "Stage positions",
        "Mags",
        "Intensities",
        "Exposure Doses",
    ];
    let mut do_size = false;
    let mut do_mode = false;
    let mut do_min = false;
    let mut do_max = false;
    let mut do_mean = false;
    let mut do_pixel = false;
    let mut do_origin = false;
    let mut do_rms = false;
    let mut do_full_eer = false;
    let mut if_brief = 0_i32;
    let mut i_volume = -1_i32;
    let mut tag_to_print = 0_i32;
    let mut input_files = Vec::<String>::new();
    let mut args = std::env::args().skip(1).peekable();
    let mut pip_input = false;

    while let Some(argument) = args.next() {
        pip_input = true;
        let option = argument.trim_start_matches('-').to_ascii_lowercase();
        match option.as_str() {
            "size" => do_size = true,
            "mode" => do_mode = true,
            "minimum" | "min" => do_min = true,
            "maximum" | "max" => do_max = true,
            "mean" => do_mean = true,
            "rootsquaremean" | "rootmeansquare" | "rms" => do_rms = true,
            "pixelsize" | "pixel" => do_pixel = true,
            "origin" => do_origin = true,
            "fullsizeofeerfile" | "eer" => do_full_eer = true,
            "input" | "inputfile" => match args.next() {
                Some(value) => input_files.push(value),
                None => {
                    eprintln!("ERROR: HEADER - No value supplied for InputFile");
                    std::process::exit(3);
                }
            },
            "brief" => if_brief = 1,
            "vol" | "volumenumber" | "volume" => {
                match args.next().and_then(|value| value.parse::<i32>().ok()) {
                    Some(value) => i_volume = value,
                    None => {
                        eprintln!("ERROR: HEADER - Invalid value for VolumeNumber");
                        std::process::exit(3);
                    }
                }
            }
            "tag" | "tiffstringtagtoprint" => {
                match args.next().and_then(|value| value.parse::<i32>().ok()) {
                    Some(value) => tag_to_print = value,
                    None => {
                        eprintln!("ERROR: HEADER - Invalid value for TiffStringTagToPrint");
                        std::process::exit(3);
                    }
                }
            }
            "help" | "usage" => {
                println!("Usage: header [options] input_file ...");
                println!("  -size -mode -pixel -origin -minimum -maximum -mean -rms");
                println!("  -volume # -brief # -eer -tag # -input file");
                return;
            }
            _ if argument.starts_with('-') => {
                eprintln!("ERROR: HEADER - Unknown option: {argument}");
                std::process::exit(3);
            }
            _ => input_files.push(argument),
        }
    }

    if input_files.is_empty() {
        if pip_input {
            eprintln!("ERROR: HEADER - No input file specified");
            std::process::exit(3);
        }
        print!(" Name of input file: ");
        let _ = io::stdout().flush();
        let mut in_file = String::new();
        if io::stdin().read_line(&mut in_file).is_err() || in_file.trim().is_empty() {
            eprintln!("ERROR: HEADER - No input file specified");
            std::process::exit(3);
        }
        input_files.push(in_file.trim_end_matches(['\r', '\n']).to_owned());
    }

    if do_full_eer {
        tiff_set_eer_read_properties(tiff_get_max_eer_super_res(), 1, 0);
    }
    if tag_to_print != 0 {
        tiff_set_string_tag_to_print(tag_to_print);
    }
    let silent =
        do_size || do_mode || do_min || do_max || do_mean || do_rms || do_pixel || do_origin;
    unsafe {
        if silent {
            let mut value = 0;
            iiualtprint_(&mut value);
        }
        let mut value = if_brief * 2;
        ialbrief_(&mut value);
    }

    for in_file in input_files {
        let filename = match CString::new(in_file.as_bytes()) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("ERROR: HEADER - Input file contains a NUL byte");
                std::process::exit(3);
            }
        };
        // The source opens every image through iiunit before calling irdhdr,
        // including the machine-readable switches.  Keep that path separate
        // from the retained old native code below so no adapter supplies the
        // silent fields.
        if silent {
            unsafe {
                ii_allow_multi_volume(1);
                iiu_open(1, filename.as_ptr(), c"RO".as_ptr());
                let num_volumes = iiu_ret_num_volumes(1);
                let mut im_unit = 1;
                if i_volume > num_volumes.max(1) {
                    eprintln!(
                        "ERROR: HEADER - The volume number entered is higher than the number of volumes in the file"
                    );
                    iiu_close(1);
                    std::process::exit(3);
                }
                if num_volumes > 1 {
                    if i_volume < 0 {
                        println!(
                            "This is the header for the first of{:4} volumes, use -vol # to see others",
                            num_volumes
                        );
                    } else if i_volume > 1 {
                        im_unit = 11;
                        if iiu_volume_open(im_unit, 1, i_volume - 1) != 0 {
                            eprintln!("ERROR: HEADER - Opening additional volume in file");
                            iiu_close(1);
                            std::process::exit(3);
                        }
                    }
                }
                let mut nxyz = [0_i32; 3];
                let mut mxyz = [0_i32; 3];
                let mut mode = 0_i32;
                let mut dmin = 0.0_f32;
                let mut dmax = 0.0_f32;
                let mut dmean = 0.0_f32;
                irdhdr(
                    im_unit,
                    nxyz.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin,
                    &mut dmax,
                    &mut dmean,
                );
                let mut unused_file_size = 0;
                let mut unused_file_type = 0;
                let mut iflags = 0;
                iiu_file_info(
                    im_unit,
                    &mut unused_file_size,
                    &mut unused_file_type,
                    &mut iflags,
                );
                if do_size {
                    println!("{:8}{:8}{:8}", nxyz[0], nxyz[1], nxyz[2]);
                }
                if iflags & (1 << 8) != 0 {
                    mode = 12;
                }
                if do_mode {
                    println!("{:4}", mode);
                }
                if do_pixel {
                    let mut delta = [0.0_f32; 3];
                    iiu_ret_delta(im_unit, delta.as_mut_ptr());
                    println!("{:15.5}{:15.5}{:15.5}", delta[0], delta[1], delta[2]);
                }
                if do_origin {
                    let mut origin = [0.0_f32; 3];
                    iiu_ret_origin(im_unit, &mut origin[0], &mut origin[1], &mut origin[2]);
                    println!("{:15.5}{:15.5}{:15.5}", origin[0], origin[1], origin[2]);
                }
                if do_min {
                    println!("{:13.5}", dmin);
                }
                if do_max {
                    println!("{:13.5}", dmax);
                }
                if do_mean {
                    println!("{:13.5}", dmean);
                }
                if do_rms {
                    let mut imod_flags = 0;
                    let mut is_imod = 0;
                    let mut version = 0;
                    let mut rms = 0.0_f32;
                    iiu_ret_imod_flags(im_unit, &mut imod_flags, &mut is_imod);
                    iiu_ret_mrc_version(im_unit, &mut version);
                    iiu_ret_rms(im_unit, &mut rms);
                    let computed = if rms > 0.0
                        || (rms == 0.0 && (version > 0 || (is_imod != 0 && imod_flags & 8 != 0)))
                    {
                        ""
                    } else {
                        "(not computed)"
                    };
                    println!("{:13.5}{computed}", rms);
                }
                iiu_close(im_unit);
                if im_unit > 1 {
                    iiu_close(1);
                }
            }
            continue;
        }
        // Source routes the normal and brief presentation through `imopen` /
        // `irdhdr` on the iiunit registry.  Keep that presentation in the
        // directly translated lower source unit; only the machine-readable
        // switches use the native image view below after `iiuAltPrint(0)`.
        if !silent {
            let mut nxyz = [0_i32; 3];
            let mut mxyz = [0_i32; 3];
            let mut mode = 0_i32;
            let mut dmin = 0.0_f32;
            let mut dmax = 0.0_f32;
            let mut dmean = 0.0_f32;
            unsafe {
                ii_allow_multi_volume(1);
                iiu_open(1, filename.as_ptr(), c"RO".as_ptr());
                let num_volumes = iiu_ret_num_volumes(1);
                let mut im_unit = 1;
                if i_volume > num_volumes.max(1) {
                    eprintln!(
                        "ERROR: HEADER - The volume number entered is higher than the number of volumes in the file"
                    );
                    iiu_close(1);
                    std::process::exit(3);
                }
                if num_volumes > 1 {
                    if i_volume < 0 {
                        println!(
                            "This is the header for the first of{:4} volumes, use -vol # to see others",
                            num_volumes
                        );
                    } else if i_volume > 1 {
                        im_unit = 11;
                        if iiu_volume_open(im_unit, 1, i_volume - 1) != 0 {
                            eprintln!("ERROR: HEADER - Opening additional volume in file");
                            iiu_close(1);
                            std::process::exit(3);
                        }
                    }
                }
                irdhdr(
                    im_unit,
                    nxyz.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin,
                    &mut dmax,
                    &mut dmean,
                );
                let mut found_pixel = false;
                let mut found_axis_rot = false;
                // Source SerialEM extended-header inventory.
                let mut extra_bytes = 0;
                iiu_ret_num_extended(im_unit, &mut extra_bytes);
                if extra_bytes > 0 {
                    let mut words = vec![0_f32; extra_bytes as usize / 4 + 10];
                    if iiu_ret_extended_data(im_unit, &mut extra_bytes, words.as_mut_ptr().cast())
                        == 0
                    {
                        let mut num_int = 0;
                        let mut num_real = 0;
                        iiu_ret_extended_type(im_unit, &mut num_int, &mut num_real);
                        // Agard/old FEI type: header.f90 indexes the raw
                        // extended-header words directly after `numInt`.
                        if extra_is_nbytes_and_flags(num_int, num_real) == 0 && num_real >= 12 {
                            let tilt_axis = words[num_int as usize + 10];
                            if (-360.0..=360.0).contains(&tilt_axis) {
                                let mut tilt_axis = tilt_axis;
                                if tilt_axis < -180.0 {
                                    tilt_axis += 360.0;
                                }
                                if tilt_axis > 180.0 {
                                    tilt_axis -= 360.0;
                                }
                                let mut labels = [[0_u8; 81]; 10];
                                let mut num_labels = 0;
                                iiu_ret_labels(
                                    im_unit,
                                    labels.as_mut_ptr().cast(),
                                    &mut num_labels,
                                );
                                if labels[0][..4] == *b"Fei " {
                                    println!(
                                        "          Tilt axis rotation angle = {:7.1} (Corrected sign)",
                                        -tilt_axis
                                    );
                                } else {
                                    println!(
                                        "          Tilt axis rotation angle = {:7.1}",
                                        tilt_axis
                                    );
                                }
                                found_axis_rot = true;
                            }
                            let mut pixel = words[num_int as usize + 11];
                            if pixel > 0.05 && pixel < 100000.0 {
                                pixel /= 10.0;
                            } else {
                                pixel *= 1.0e9;
                            }
                            let mut delta = [0.0_f32; 3];
                            let mut imod_flags = 0;
                            let mut is_imod = 0;
                            iiu_ret_delta(im_unit, delta.as_mut_ptr());
                            iiu_ret_imod_flags(im_unit, &mut imod_flags, &mut is_imod);
                            if pixel > 0.005 && pixel < 10000.0 && imod_flags & 2 == 0 {
                                let mut binning = 0_i32;
                                for index in (0..3).rev() {
                                    binning = delta[index].round() as i32;
                                    if (delta[index] - binning as f32).abs() > 1.0e-6
                                        || binning <= 0
                                        || binning > 4
                                    {
                                        binning = 0;
                                        break;
                                    }
                                }
                                if binning == 1 {
                                    println!("          Pixel size in nanometers ={:11.4}", pixel);
                                } else if (2..5).contains(&binning) {
                                    println!(
                                        "          Pixel size in nanometers ={:11.4} (Assumed binning of{:2})",
                                        pixel * binning as f32,
                                        binning
                                    );
                                } else {
                                    println!(
                                        "          Original/extended header pixel size in nanometers ={:11.4}",
                                        pixel
                                    );
                                }
                                found_pixel = true;
                            }
                        }
                        // New FEI type.  The calls and raw offsets are the
                        // direct `getExtraHeaderValue` sequence in header.f90.
                        if num_int == -3 {
                            let mut byte_value = 0_u8;
                            let mut short_value = 0_i16;
                            let mut mask = 0_i32;
                            let mut tilt_axis = 0.0_f32;
                            let mut axis8 = 0.0_f64;
                            if get_extra_header_value(
                                words.as_mut_ptr().cast(),
                                8,
                                3,
                                &mut byte_value,
                                &mut short_value,
                                &mut mask,
                                &mut tilt_axis,
                                &mut axis8,
                            ) == 0
                                && get_extra_header_value(
                                    words.as_mut_ptr().cast(),
                                    140,
                                    4,
                                    &mut byte_value,
                                    &mut short_value,
                                    &mut mask,
                                    &mut tilt_axis,
                                    &mut axis8,
                                ) == 0
                                && mask & (1 << 12) != 0
                            {
                                tilt_axis = (axis8
                                    * get_fei_ext_head_angle_scale(words.as_mut_ptr().cast()))
                                    as f32;
                                if (-360.0..=360.0).contains(&tilt_axis) {
                                    if tilt_axis < -180.0 {
                                        tilt_axis += 360.0;
                                    }
                                    if tilt_axis > 180.0 {
                                        tilt_axis -= 360.0;
                                    }
                                    println!(
                                        "          Tilt axis rotation angle = {:7.1} (Corrected sign)",
                                        -tilt_axis
                                    );
                                    found_axis_rot = true;
                                }
                            }
                        }
                        if extra_is_nbytes_and_flags(num_int, num_real) != 0 {
                            if if_brief == 0 {
                                println!("\nExtended header from SerialEM contains:");
                            }
                            let mut brief_separator = "  Contains:";
                            let mut found = 0;
                            for index in 0..6 {
                                if (num_real / (1 << index)) % 2 != 0 {
                                    if if_brief == 0 {
                                        println!(
                                            "  {} - Extract with \"{}\"",
                                            type_name[index], extract_com[index]
                                        );
                                    } else {
                                        print!("{} {}", brief_separator, brief_name[index]);
                                        brief_separator = " -";
                                        found += 1;
                                    }
                                }
                            }
                            if found > 0 {
                                println!();
                            }
                        } else {
                            let mut tilts = vec![0.0_f32; nxyz[2] as usize + 9];
                            let mut iz_piece = (0..nxyz[2]).collect::<Vec<_>>();
                            let mut num_tilts = 0;
                            if get_extra_header_items(
                                words.as_mut_ptr().cast(),
                                extra_bytes,
                                num_int,
                                num_real,
                                nxyz[2],
                                1,
                                tilts.as_mut_ptr(),
                                tilts.as_mut_ptr(),
                                &mut num_tilts,
                                nxyz[2] + 9,
                                iz_piece.as_mut_ptr(),
                            ) > 0
                            {
                                println!(
                                    "Extended header has tilt angles - extract with \"extracttilts\""
                                );
                            }
                        }
                    }
                }
                // `header.f90` treats a title mentioning this angle as an
                // already-present axis value, without producing a second line.
                if !found_axis_rot {
                    let mut labels = [[0_u8; 81]; 10];
                    let mut num_labels = 0;
                    iiu_ret_labels(im_unit, labels.as_mut_ptr().cast(), &mut num_labels);
                    for label in labels.iter().take(num_labels as usize) {
                        if String::from_utf8_lossy(&label[..20]).contains("Tilt axis angle") {
                            found_axis_rot = true;
                            break;
                        }
                    }
                }
                if !found_pixel {
                    let mut delta = [0.0_f32; 3];
                    iiu_ret_delta(im_unit, delta.as_mut_ptr());
                    found_pixel = delta[0] != 1.0 || delta[1] != 1.0 || delta[2] != 1.0;
                }
                // Direct mdoc fallback in the original: global PixelSpacing,
                // then `T` title sections and (for TFS titles) ZValue 1's
                // RotationAngle.
                if !found_pixel || !found_axis_rot {
                    let mut montage = 0;
                    let mut num_sections = 0;
                    let mut section_type = 0;
                    if adoc_open_image_metadata(
                        filename.as_ptr(),
                        1,
                        &mut montage,
                        &mut num_sections,
                        &mut section_type,
                    ) >= 0
                    {
                        if !found_pixel {
                            let mut pixel = 0.0_f32;
                            if adoc_get_float(
                                ADOC_GLOBAL_NAME.as_ptr(),
                                0,
                                c"PixelSpacing".as_ptr(),
                                &mut pixel,
                            ) == 0
                            {
                                println!(
                                    "          Pixel size in nanometers ={:11.4}  , from mdoc",
                                    pixel / 10.0
                                );
                            }
                        }
                        if !found_axis_rot {
                            let num_labels = adoc_get_number_of_sections(c"T".as_ptr());
                            for index in 0..num_labels {
                                let mut title = core::ptr::null_mut();
                                if adoc_get_section_name(c"T".as_ptr(), index, &mut title) == 0 {
                                    let title_text = CStr::from_ptr(title).to_string_lossy();
                                    let fei_label = title_text.contains("TiltAxisAngle");
                                    if (fei_label || title_text.contains("Tilt axis angle"))
                                        && let Some((_, value)) = title_text.split_once('=')
                                        && let Ok(mut tilt_axis) = value.trim().parse::<f32>()
                                    {
                                        if fei_label {
                                            let mut rotation_angle = 0.0_f32;
                                            if adoc_get_float(
                                                ADOC_ZVALUE_NAME.as_ptr(),
                                                0,
                                                c"RotationAngle".as_ptr(),
                                                &mut rotation_angle,
                                            ) == 0
                                            {
                                                if (-(rotation_angle + 90.0) - tilt_axis).abs()
                                                    < 0.11
                                                {
                                                    println!(
                                                        "          Tilt axis rotation angle = {:7.1}  (from RotationAngle in mdoc)",
                                                        rotation_angle
                                                    );
                                                } else if ((rotation_angle - 90.0) - tilt_axis)
                                                    .abs()
                                                    < 0.11
                                                {
                                                    println!(
                                                        "          Tilt axis rotation angle = {:7.1}  (from mdoc)",
                                                        tilt_axis
                                                    );
                                                } else if (-(rotation_angle - 90.0) - tilt_axis)
                                                    .abs()
                                                    < 0.11
                                                {
                                                    tilt_axis = -tilt_axis;
                                                    println!(
                                                        "          Tilt axis rotation angle = {:7.1}  (corrected sign, from mdoc)",
                                                        tilt_axis
                                                    );
                                                }
                                            }
                                        } else {
                                            println!(
                                                "          Tilt axis rotation angle = {:7.1}  (from mdoc)",
                                                tilt_axis
                                            );
                                        }
                                        libc::free(title.cast());
                                        break;
                                    }
                                    libc::free(title.cast());
                                }
                            }
                        }
                    }
                }
                iiu_close(im_unit);
                if im_unit > 1 {
                    iiu_close(1);
                }
            }
            continue;
        }
        ii_allow_multi_volume(1);
        let root_image = unsafe { ii_open(filename.as_ptr(), c"rb".as_ptr()) };
        if root_image.is_null() {
            eprintln!("ERROR: HEADER - Opening input file: {in_file}");
            std::process::exit(3);
        }
        let mut image = root_image;
        let num_volumes = unsafe { (*root_image).num_volumes };
        if i_volume > num_volumes.max(1) {
            eprintln!(
                "ERROR: HEADER - The volume number entered is higher than the number of volumes in the file"
            );
            unsafe { ii_delete(root_image) };
            std::process::exit(3);
        }
        if num_volumes > 1 && i_volume < 0 {
            println!(
                "This is the header for the first of{:4} volumes, use -vol # to see others",
                num_volumes
            );
        } else if num_volumes > 1 && i_volume > 1 {
            let volume_fp = unsafe { ii_fopen_volume(root_image, i_volume - 1) };
            image = unsafe { ii_lookup_file_from_fp(volume_fp) };
            if image.is_null() {
                eprintln!("ERROR: HEADER - Opening additional volume in file");
                unsafe { ii_delete(root_image) };
                std::process::exit(3);
            }
        }

        // `irdhdr` gets this data from its unit table.  `iiFillMrcHeader`
        // supplies the same normalized MRC view for every supported format.
        let mut hdata = std::mem::MaybeUninit::<MrcHeader>::zeroed();
        if unsafe { ii_fill_mrc_header(image, hdata.as_mut_ptr()) } != 0 {
            eprintln!("ERROR: HEADER - Reading header from input file: {in_file}");
            unsafe {
                ii_delete(image);
                if image != root_image {
                    ii_delete(root_image);
                }
            };
            std::process::exit(3);
        }
        let mut hdata = unsafe { hdata.assume_init() };
        let image_data: &ImodImageFile = unsafe { &*image };
        let mut mode = image_data.mode;
        let (delta_x, delta_y, delta_z) = mrc_get_scale(&hdata);

        if silent {
            if (image_data.user_flags & (1_u32 << 8)) != 0 {
                mode = 12;
            }
            if do_size {
                println!("{:8}{:8}{:8}", image_data.nx, image_data.ny, image_data.nz);
            }
            if do_mode {
                println!("{:4}", mode);
            }
            if do_pixel {
                println!("{:15.5}{:15.5}{:15.5}", delta_x, delta_y, delta_z);
            }
            if do_origin {
                println!(
                    "{:15.5}{:15.5}{:15.5}",
                    image_data.xtrans, image_data.ytrans, image_data.ztrans
                );
            }
            if do_min {
                println!("{:13.5}", image_data.amin);
            }
            if do_max {
                println!("{:13.5}", image_data.amax);
            }
            if do_mean {
                println!("{:13.5}", image_data.amean);
            }
            if do_rms {
                let computed = if image_data.rms > 0.0
                    || (image_data.rms == 0.0
                        && (hdata.nversion > 0
                            || (hdata.imod_stamp != 0 && (hdata.imod_flags & 8) != 0)))
                {
                    ""
                } else {
                    "(not computed)"
                };
                println!("{:13.5}{computed}", image_data.rms);
            }
        } else if if_brief > 0 {
            println!(
                " Dimensions:{:7}{:7}{:7}   Pixel size:{:11.4}{:11.4}{:11.4}",
                image_data.nx, image_data.ny, image_data.nz, delta_x, delta_y, delta_z
            );
            println!(
                " Mode:{:3}               Min, max, mean:{:13.5}{:13.5}{:13.5}",
                mode, image_data.amin, image_data.amax, image_data.amean
            );
            if hdata.nlabl > 0 {
                println!("{}", String::from_utf8_lossy(&hdata.labels[0][..80]));
            }
            if hdata.nlabl > 1 {
                println!(
                    "{}",
                    String::from_utf8_lossy(
                        &hdata.labels[(hdata.nlabl - 1).clamp(0, 9) as usize][..80]
                    )
                );
            }
        } else {
            let mode_name = match mode {
                0 => "(byte)",
                1 => "(16-bit integer)",
                2 => "(32-bit float)",
                3 => "(complex integer)",
                4 => "(complex)",
                6 => "(unsigned 16-bit integer)",
                12 => "(16-bit float)",
                16 => "RGB color",
                _ => "(unknown)",
            };
            println!();
            println!(
                " Number of columns, rows, sections .....{:8}{:8}{:8}",
                image_data.nx, image_data.ny, image_data.nz
            );
            println!(
                " Map mode ..............................{:5}   {mode_name}",
                mode
            );
            println!(
                " Pixel spacing (Angstroms).............. {:11.4}{:11.4}{:11.4}",
                delta_x, delta_y, delta_z
            );
            println!(
                " Origin on x,y,z ....................... {:12.4}{:12.4}{:12.4}",
                image_data.xtrans, image_data.ytrans, image_data.ztrans
            );
            println!(
                " Minimum density ........................{:13.5}",
                image_data.amin
            );
            println!(
                " Maximum density ........................{:13.5}",
                image_data.amax
            );
            println!(
                " Mean density ...........................{:13.5}",
                image_data.amean
            );
            if image_data.rms > 0.0 || (image_data.rms == 0.0 && hdata.nversion > 0) {
                println!(
                    " RMS deviation from mean................{:13.5}",
                    image_data.rms
                );
            }
            println!(
                " tilt angles (original,current) ........{:6.1}{:6.1}{:6.1}{:6.1}{:6.1}{:6.1}",
                hdata.tiltangles[0],
                hdata.tiltangles[1],
                hdata.tiltangles[2],
                hdata.tiltangles[3],
                hdata.tiltangles[4],
                hdata.tiltangles[5]
            );
            println!(
                " Space group,# extra bytes,idtype,lens .{:9}{:9}{:9}{:9}",
                hdata.ispg, hdata.next, hdata.idtype, hdata.lens
            );
            println!();
            println!("{:5} Titles :", hdata.nlabl);
            for label in hdata.labels.iter().take(hdata.nlabl.clamp(0, 10) as usize) {
                println!("{}", String::from_utf8_lossy(&label[..80]));
            }

            if hdata.next > 0 {
                let mut ext_data = std::ptr::null_mut();
                if unsafe { mrc_read_extra_header(&mut hdata, &mut ext_data) } == 0
                    && !ext_data.is_null()
                {
                    let ext_words = unsafe {
                        std::slice::from_raw_parts(ext_data.cast::<f32>(), hdata.next as usize / 4)
                    };
                    if extra_is_nbytes_and_flags(hdata.nint as i32, hdata.nreal as i32) != 0 {
                        let mut contains = Vec::new();
                        for index in 0..6 {
                            if ((hdata.nreal as i32 >> index) & 1) != 0 {
                                if if_brief == 0 {
                                    println!(
                                        "  {} - Extract with \"{}\"",
                                        type_name[index], extract_com[index]
                                    );
                                } else {
                                    contains.push(brief_name[index]);
                                }
                            }
                        }
                        if !contains.is_empty() {
                            println!("  Contains: {}", contains.join(" - "));
                        }
                    } else if hdata.nreal >= 12
                        && ext_words.len() >= hdata.nint.max(0) as usize + 12
                    {
                        let mut tilt_axis = ext_words[hdata.nint.max(0) as usize + 10];
                        if (-360.0..=360.0).contains(&tilt_axis) {
                            if tilt_axis < -180.0 {
                                tilt_axis += 360.0;
                            }
                            if tilt_axis > 180.0 {
                                tilt_axis -= 360.0;
                            }
                            let fei = hdata.labels[0][..4] == *b"Fei ";
                            println!(
                                "          Tilt axis rotation angle = {:7.1}{}",
                                if fei { -tilt_axis } else { tilt_axis },
                                if fei { " (Corrected sign)" } else { "" }
                            );
                        }
                        let mut pixel = ext_words[hdata.nint.max(0) as usize + 11];
                        if pixel > 0.05 && pixel < 100000.0 {
                            pixel /= 10.0;
                        } else {
                            pixel *= 1.0e9;
                        }
                        if pixel > 0.005 && pixel < 10000.0 && (hdata.imod_flags & 2) == 0 {
                            println!(
                                "          Original/extended header pixel size in nanometers ={:11.4}",
                                pixel
                            );
                        }
                    }
                    unsafe { libc::free(ext_data.cast()) };
                }
            }
        }
        if if_brief > 0 && !silent {
            println!();
        }
        unsafe {
            ii_delete(image);
            if image != root_image {
                ii_delete(root_image);
            }
        };
    }
}
