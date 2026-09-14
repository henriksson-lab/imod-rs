//! Translation of `IMOD/flib/image/header.f90`.
//!
//! `header.f90` is a Fortran main program, so its single executable program
//! unit maps directly to [`header`].  The lower-level Fortran `iiunit` calls
//! used by the original are represented by the already-translated native image
//! interface; no second unit-file registry is introduced here.
#![allow(dead_code, unused_variables)]

use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_number_of_sections,
    adoc_get_section_name, adoc_open_image_metadata,
};
use crate::imod::libcfshr::b3dutil::extra_is_nbytes_and_flags;
use crate::imod::libcfshr::extraheader::{
    get_extra_header_items_fortran, get_extra_header_value, get_fei_ext_head_angle_scale,
};
use crate::imod::libcfshr::parse_params::pip_enable_entry_output;
use crate::imod::libcfshr::pip_fwrap::{
    pipgetinteger_, pipgetnonoptionarg_, pipgetstring_, pipnumberofentries_,
};
use crate::imod::libiimod::iimage::ii_allow_multi_volume;
use crate::imod::libiimod::iitif::{
    tiff_get_max_eer_super_res, tiff_set_eer_read_properties, tiff_set_string_tag_to_print,
};
use crate::imod::libiimod::unit_fileio::{
    ialbrief_, iiu_close, iiu_file_info, iiu_ret_num_volumes, iiu_volume_open, iiualtprint_,
};
use crate::imod::libiimod::unit_header::{
    iiu_ret_delta, iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_imod_flags,
    iiu_ret_labels, iiu_ret_mrc_version, iiu_ret_num_extended, iiu_ret_origin, iiu_ret_rms,
};
use std::io::{self, Write};

/// `parameter (ntypes = 6)` (`header.f90:12`).
const NTYPES: usize = 6;

/// `parameter (numOptions = 14)` (`header.f90:51`).
const NUM_OPTIONS: i32 = 14;

/// The `options(1)` fallback PIP table (`header.f90:54`), kept as the single
/// `@`-separated Fortran string that `PipReadOrParseOptions` splits.
const HEADER_OPTIONS: &str = "input:InputFile:FNM:@size:Size:B:@mode:Mode:B:@pixel:PixelSize:B:@\
origin:Origin:B:@minimum:Minimum:B:@maximum:Maximum:B:@mean:Mean:B:@\
rms:RootMeanSquare:B:@volume:VolumeNumber:I:@brief:Brief:B:@\
eer:FullSizeOfEERFile:B:@tag:TiffStringTagToPrint:I:@help:usage:B:";

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
    let mut num_file_in = 1_i32;
    let mut num_input_files = 0_i32;
    let mut if_brief = 0_i32;
    let mut i_volume = -1_i32;
    let mut tag_to_print = 0_i32;
    let mut num_opt_arg = 0_i32;
    let mut num_non_opt_arg = 0_i32;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    // But turn off the entry printing first!
    unsafe {
        pip_enable_entry_output(0);
    }
    pip_read_or_parse_options(
        &[HEADER_OPTIONS],
        NUM_OPTIONS,
        "header",
        "ERROR: HEADER - ",
        true,
        1,
        2,
        0,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    unsafe {
        if pip_input {
            pip_get_logical("Size", &mut do_size);
            pip_get_logical("Mode", &mut do_mode);
            pip_get_logical("Max", &mut do_max);
            pip_get_logical("Min", &mut do_min);
            pip_get_logical("Mean", &mut do_mean);
            pip_get_logical("RootMeanSquare", &mut do_rms);
            pip_get_logical("PixelSize", &mut do_pixel);
            pip_get_logical("Origin", &mut do_origin);
            let mut option = *b"Brief";
            pipgetinteger_(
                option.as_mut_ptr().cast(),
                &raw mut if_brief,
                option.len() as i32,
            );
            pip_get_logical("FullSizeOfEERFile", &mut do_full_eer);
            let mut option = *b"VolumeNumber";
            pipgetinteger_(
                option.as_mut_ptr().cast(),
                &raw mut i_volume,
                option.len() as i32,
            );
            let mut option = *b"TiffStringTagToPrint";
            if pipgetinteger_(
                option.as_mut_ptr().cast(),
                &raw mut tag_to_print,
                option.len() as i32,
            ) == 0
            {
                // `iiSetTiffTagToPrint` (`unit_fileio.c:941`) is declared
                // `int` but falls off its end without returning, so the
                // translated `iisettifftagtoprint_` traps on that source-level
                // UB; the Fortran only ever `call`s it, so its one statement,
                // `tiffSetStringTagToPrint(*tag)`, is invoked directly.
                tiff_set_string_tag_to_print(tag_to_print);
            }
            let mut option = *b"InputFile";
            pipnumberofentries_(
                option.as_mut_ptr().cast(),
                &raw mut num_input_files,
                option.len() as i32,
            );
            num_file_in = num_input_files + num_non_opt_arg;
            if num_file_in == 0 {
                exit_error("No input file specified");
            }
            if do_full_eer {
                tiff_set_eer_read_properties(tiff_get_max_eer_super_res(), 1, 0);
            }
        }
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

    // Fortran `Gw.d` edit descriptor, as used by every `g11.4`, `g13.5` and
    // `g15.5` field written by `header.f90` (FORMAT 102/104 and the
    // `write(*, '(3g15.5)')` / `write(*, '(g13.5,a)')` statements at lines
    // 153-168).  With no `Ee` part, a magnitude that rounds to `d` significant
    // digits within [0.1, 10**d) is written as `F(w-4).(d-k)` followed by four
    // blanks; anything else is written as `Ew.d` with the default scale factor.
    let g_edit = |value: f32, w: usize, d: i32| -> String {
        let magnitude = value.abs();
        let mut digits = String::new();
        let mut exponent = 1_i32;
        if magnitude != 0.0 {
            let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
            let (mantissa, power) = scientific.split_once('e').unwrap();
            digits = mantissa.replace('.', "");
            exponent = power.parse::<i32>().unwrap() + 1;
        }
        if (0..=d).contains(&exponent) {
            let mut text = format!("{:.*}", (d - exponent) as usize, value);
            if exponent == d {
                text.push('.');
            }
            format!("{:>1$}    ", text, w - 4)
        } else {
            format!(
                "{:>1$}",
                format!(
                    "{}0.{}E{}{:02}",
                    if value < 0.0 { "-" } else { "" },
                    digits,
                    if exponent < 0 { '-' } else { '+' },
                    exponent.abs()
                ),
                w
            )
        }
    };

    // `computed`, `briefSep`, `foundPixel` and `foundAxisRot` are declared once
    // for the whole program in `header.f90:29,30,72,73`, so their state carries
    // from one input file to the next inside the loop below.
    let mut computed = "(not computed)";
    let mut brief_sep = "  Contains:";
    let mut found_pixel = false;
    let mut found_axis_rot = false;

    for i in 1..=num_file_in {
        //
        // get the next filename
        //
        // `character*320 inFile` (`header.f90:19`), so the Fortran PIP wrappers
        // fill a fixed 320-character record and blank-pad it.
        let mut in_file_record = [b' '; 320];
        unsafe {
            if pip_input {
                if i <= num_input_files {
                    let mut option = *b"InputFile";
                    pipgetstring_(
                        option.as_mut_ptr().cast(),
                        in_file_record.as_mut_ptr().cast(),
                        option.len() as i32,
                        in_file_record.len() as i32,
                    );
                } else {
                    let mut arg_number = i - num_input_files;
                    pipgetnonoptionarg_(
                        &raw mut arg_number,
                        in_file_record.as_mut_ptr().cast(),
                        in_file_record.len() as i32,
                    );
                }
            } else {
                print!(" {}", "Name of input file: ");
                let _ = io::stdout().flush();
                let mut line = String::new();
                let _ = io::stdin().read_line(&mut line);
                let bytes = line.trim_end_matches(['\r', '\n']).as_bytes();
                let count = bytes.len().min(in_file_record.len());
                in_file_record[..count].copy_from_slice(&bytes[..count]);
            }
        }
        let mut length = in_file_record.len();
        while length > 0 && in_file_record[length - 1] == b' ' {
            length -= 1;
        }
        let in_file = String::from_utf8_lossy(&in_file_record[..length]).into_owned();
        unsafe {
            ii_allow_multi_volume(1);
            imopen(1, &in_file, "RO");
            let num_volumes = iiu_ret_num_volumes(1);
            let mut im_unit = 1;
            if i_volume > num_volumes.max(1) {
                exit_error(
                    "The volume number entered is higher than the number of volumes in the file",
                );
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
                        exit_error("Opening additional volume in file");
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
            if silent {
                let mut idum = 0;
                let mut ifile_type = 0;
                let mut iflags = 0;
                iiu_file_info(im_unit, &mut idum, &mut ifile_type, &mut iflags);
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
                    println!(
                        "{}{}{}",
                        g_edit(delta[0], 15, 5),
                        g_edit(delta[1], 15, 5),
                        g_edit(delta[2], 15, 5)
                    );
                }
                if do_origin {
                    let mut delta = [0.0_f32; 3];
                    iiu_ret_origin(im_unit, &mut delta);
                    println!(
                        "{}{}{}",
                        g_edit(delta[0], 15, 5),
                        g_edit(delta[1], 15, 5),
                        g_edit(delta[2], 15, 5)
                    );
                }
                if do_min {
                    println!("{}", g_edit(dmin, 13, 5));
                }
                if do_max {
                    println!("{}", g_edit(dmax, 13, 5));
                }
                if do_mean {
                    println!("{}", g_edit(dmean, 13, 5));
                }
                if do_rms {
                    let mut iflags = 0;
                    let mut if_imod = 0;
                    let mut j = 0;
                    let mut rms = 0.0_f32;
                    iiu_ret_imod_flags(im_unit, &mut iflags, &mut if_imod);
                    iiu_ret_mrc_version(im_unit, &mut j);
                    iiu_ret_rms(im_unit, &mut rms);
                    if rms > 0.0
                        || (rms == 0.0 && (j > 0 || (if_imod != 0 && iflags & (1 << 3) != 0)))
                    {
                        computed = "";
                    }
                    println!("{}{}", g_edit(rms, 13, 5), computed);
                }
            } else {
                let mut nbsym = 0;
                iiu_ret_num_extended(im_unit, &mut nbsym);
                if nbsym > 0 {
                    // `allocate(array(nbsym / 4 + 10), stat=ierr)` followed by
                    // `call memoryError(ierr, 'array for extended header')`
                    // (`header.f90:174`).
                    let mut array = Vec::<f32>::new();
                    let ierr = i32::from(array.try_reserve_exact(nbsym as usize / 4 + 10).is_err());
                    memory_error(ierr, "array for extended header");
                    array.resize(nbsym as usize / 4 + 10, 0.0);
                    iiu_ret_extended_data(im_unit, &mut nbsym, array.as_mut_ptr().cast());
                    let mut extended_type = [0; 2];
                    iiu_ret_extended_type(im_unit, &mut extended_type);
                    let [mut num_int, mut num_real] = extended_type;
                    if extra_is_nbytes_and_flags(num_int, num_real) == 0 && num_real >= 12 {
                        //
                        // Agard/old FEI type
                        let mut tiltaxis = array[(num_int + 10) as usize];
                        if (-360.0..=360.0).contains(&tiltaxis) {
                            if tiltaxis < -180.0 {
                                tiltaxis += 360.0;
                            }
                            if tiltaxis > 180.0 {
                                tiltaxis -= 360.0;
                            }
                            let mut labels = [[0_u8; 80]; 10];
                            let mut nlabel = 0;
                            iiu_ret_labels(im_unit, labels.as_mut_ptr().cast(), &mut nlabel);
                            if labels[0][..4] == *b"Fei " {
                                println!(
                                    "          Tilt axis rotation angle = {:7.1}{}",
                                    -tiltaxis, " (Corrected sign)"
                                );
                            } else {
                                println!("          Tilt axis rotation angle = {:7.1}", tiltaxis);
                            }
                            found_axis_rot = true;
                        }
                        //
                        // The pixel size is supposed to be in meters but UCSF frame file has it
                        // in Angstroms.  So see if A is reasonable and scale to nm, or scale m
                        // to nm
                        let mut pixel = array[(num_int + 11) as usize];
                        if array[(num_int + 11) as usize] > 0.05
                            && array[(num_int + 11) as usize] < 100000.0
                        {
                            pixel /= 10.0;
                        } else {
                            pixel *= 1.0e9;
                        }
                        let mut delta = [0.0_f32; 3];
                        let mut iflags = 0;
                        let mut if_imod = 0;
                        iiu_ret_delta(im_unit, delta.as_mut_ptr());
                        iiu_ret_imod_flags(im_unit, &mut iflags, &mut if_imod);
                        if pixel > 0.005 && pixel < 10000.0 && iflags & 2 == 0 {
                            let mut i_binning = 0_i32;
                            for j in (0..3).rev() {
                                i_binning = delta[j].round() as i32;
                                if (delta[j] - i_binning as f32).abs() > 1.0e-6
                                    || i_binning <= 0
                                    || i_binning > 4
                                {
                                    i_binning = 0;
                                    break;
                                }
                            }
                            if i_binning == 1 {
                                println!(
                                    "          Pixel size in nanometers ={}",
                                    g_edit(pixel, 11, 4)
                                );
                            } else if i_binning > 1 && i_binning < 5 {
                                println!(
                                    "          Pixel size in nanometers ={}{}{:2}{}",
                                    g_edit(pixel * i_binning as f32, 11, 4),
                                    " (Assumed binning of",
                                    i_binning,
                                    ")"
                                );
                            } else {
                                println!(
                                    "          Original/extended header pixel size in nanometers ={}",
                                    g_edit(pixel, 11, 4)
                                );
                            }
                            found_pixel = true;
                        }
                    }
                    //
                    // New FEI type
                    if num_int == -3 {
                        let mut byte_value = 0_u8;
                        let mut short_value = 0_i16;
                        let mut mask = 0_i32;
                        let mut j = 0_i32;
                        let mut tiltaxis = 0.0_f32;
                        let mut axis8 = 0.0_f64;
                        let array_bytes = unsafe {
                            core::slice::from_raw_parts(
                                array.as_ptr().cast::<u8>(),
                                core::mem::size_of_val(array.as_slice()),
                            )
                        };
                        if get_extra_header_value(
                            array_bytes,
                            8,
                            3,
                            &mut byte_value,
                            &mut short_value,
                            &mut mask,
                            &mut tiltaxis,
                            &mut axis8,
                        ) == 0
                            && get_extra_header_value(
                                array_bytes,
                                140,
                                4,
                                &mut byte_value,
                                &mut short_value,
                                &mut j,
                                &mut tiltaxis,
                                &mut axis8,
                            ) == 0
                            && mask & (1 << 12) != 0
                        {
                            tiltaxis = (axis8 * get_fei_ext_head_angle_scale(array_bytes)) as f32;
                            if (-360.0..=360.0).contains(&tiltaxis) {
                                if tiltaxis < -180.0 {
                                    tiltaxis += 360.0;
                                }
                                if tiltaxis > 180.0 {
                                    tiltaxis -= 360.0;
                                }
                                println!(
                                    "          Tilt axis rotation angle = {:7.1}{}",
                                    -tiltaxis, " (Corrected sign)"
                                );
                                found_axis_rot = true;
                            }
                        }
                    }
                    //
                    // SerialEM type
                    if extra_is_nbytes_and_flags(num_int, num_real) != 0 {
                        if if_brief == 0 {
                            println!();
                            println!("Extended header from SerialEM contains:");
                        }
                        num_int = 0;
                        for j in 0..NTYPES {
                            if (num_real / (1 << j)) % 2 != 0 {
                                if if_brief == 0 {
                                    // FORMAT 103 writes `typeName` from its full
                                    // `character*17` field, blank padded.
                                    println!(
                                        "  {:17} - Extract with \"{}\"",
                                        type_name[j], extract_com[j]
                                    );
                                } else {
                                    print!("{} {}", brief_sep, brief_name[j]);
                                    let _ = io::stdout().flush();
                                    brief_sep = " -";
                                    num_int += 1;
                                }
                            }
                        }
                        if num_int > 0 {
                            println!();
                        }
                    } else if nbsym > 0 {
                        let mut tilts = vec![0.0_f32; nxyz[2] as usize + 9];
                        let mut iz_piece = vec![0_i32; nxyz[2] as usize + 9];
                        for j in 0..nxyz[2] as usize {
                            iz_piece[j] = j as i32;
                        }
                        let mut ierr = 0;
                        let mut one = 1;
                        let mut max_vals = nxyz[2] + 9;
                        let mut nz = nxyz[2];
                        get_extra_header_items_fortran(
                            array.as_mut_ptr().cast(),
                            &mut nbsym,
                            &mut num_int,
                            &mut num_real,
                            &mut nz,
                            &mut one,
                            tilts.as_mut_ptr(),
                            tilts.as_mut_ptr(),
                            &mut ierr,
                            &mut max_vals,
                            iz_piece.as_mut_ptr(),
                        );
                        if ierr > 0 {
                            println!(
                                "Extended header has tilt angles - extract with \"extracttilts\""
                            );
                        }
                    }
                }
            }

            // If no axis rotation in extended header, look for it in labels
            if !found_axis_rot {
                let mut all_labels = [[0_u8; 80]; 10];
                let mut num_labels = 0;
                iiu_ret_labels(1, all_labels.as_mut_ptr().cast(), &mut num_labels);
                for j in 0..num_labels.clamp(0, 10) as usize {
                    let temp_label_str = String::from_utf8_lossy(&all_labels[j][..80]);
                    if temp_label_str.contains("Tilt axis angle") {
                        found_axis_rot = true;
                        break;
                    }
                }
            }

            // if no pixel in extended header,
            if !found_pixel {
                let mut delta = [0.0_f32; 3];
                iiu_ret_delta(1, delta.as_mut_ptr());
                found_pixel = delta[0] != 1.0 || delta[1] != 1.0 || delta[2] != 1.0;
            }

            // Look for mdoc file in either case
            if !found_pixel || !found_axis_rot {
                let mut montage = 0;
                let mut num_sect = 0;
                let mut i_type_adoc = 0;
                let ind_adoc = adoc_open_image_metadata(
                    in_file.as_bytes(),
                    1,
                    &mut montage,
                    &mut num_sect,
                    &mut i_type_adoc,
                );
                if ind_adoc >= 0 {
                    if !found_pixel {
                        // Etomo needed a comma before text, or no text.  Copytomocoms now
                        // wants "from mdoc" to know that pixel spacing was 1
                        let mut pixel = 0.0_f32;
                        if adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut pixel) == 0 {
                            println!(
                                "          Pixel size in nanometers ={}{}",
                                g_edit(pixel / 10.0, 11, 4),
                                "  , from mdoc"
                            );
                        }
                    }

                    if !found_axis_rot {
                        // Look in titles
                        let num_labels = adoc_get_number_of_sections(b"T");
                        for j in 0..num_labels {
                            let mut name = Vec::<u8>::new();
                            if adoc_get_section_name(b"T", j, &mut name) == 0 {
                                let temp_label_str = String::from_utf8_lossy(&name);
                                let fei_label = temp_label_str.contains("TiltAxisAngle");
                                if fei_label || temp_label_str.contains("Tilt axis angle") {
                                    if let Some((_, rest)) = temp_label_str.split_once('=') {
                                        // Fortran list-directed `read(extract, *)`
                                        // stops at the first value separator.
                                        let extract = rest.trim_start();
                                        let end = extract
                                            .find([',', ' ', '\t', '/'])
                                            .unwrap_or(extract.len());
                                        if let Ok(mut tilt_axis) = extract[..end].parse::<f32>() {
                                            // If it is from TFS software, make sure there is a
                                            // RotationAngle entry too and that values make some
                                            // kind of sense
                                            if fei_label {
                                                let mut rot_angle = 0.0_f32;
                                                if adoc_get_float(
                                                    ADOC_ZVALUE_NAME,
                                                    0,
                                                    b"RotationAngle",
                                                    &mut rot_angle,
                                                ) == 0
                                                {
                                                    // Do what alignframes does:
                                                    // The current wrong FEI implementation
                                                    if (-(rot_angle + 90.0) - tilt_axis).abs()
                                                        < 0.11
                                                    {
                                                        println!(
                                                            "          Tilt axis rotation angle = {:7.1}{}",
                                                            rot_angle,
                                                            "  (from RotationAngle in mdoc)"
                                                        );
                                                    // If they corrected it to match SerialEM
                                                    } else if ((rot_angle - 90.0) - tilt_axis).abs()
                                                        < 0.11
                                                    {
                                                        println!(
                                                            "          Tilt axis rotation angle = {:7.1}{}",
                                                            tilt_axis, "  (from mdoc)"
                                                        );
                                                    // If they sorta corrected it but kept it
                                                    // inverted as in TS file
                                                    } else if (-(rot_angle - 90.0) - tilt_axis)
                                                        .abs()
                                                        < 0.11
                                                    {
                                                        println!(
                                                            "          Tilt axis rotation angle = {:7.1}{}",
                                                            -tilt_axis,
                                                            "  (corrected sign, from mdoc)"
                                                        );
                                                        tilt_axis = -tilt_axis;
                                                        let _ = tilt_axis;
                                                    }
                                                }
                                            } else {
                                                println!(
                                                    "          Tilt axis rotation angle = {:7.1}{}",
                                                    tilt_axis, "  (from mdoc)"
                                                );
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if if_brief > 0 {
                println!();
            }

            iiu_close(im_unit);
            if im_unit > 1 {
                iiu_close(1);
            }
        }
    }
}
