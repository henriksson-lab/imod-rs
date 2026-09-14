//! Direct translation of `IMOD/flib/image/alterheader.f90`.
//!
//! All header access goes through the source program's iiunit interface; no
//! copied `MrcHeader` adapter is used.  The Fortran computed `GO TO` becomes a
//! single dispatch loop over the same label numbers.
#![allow(dead_code, unused_variables)]

use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, pip_read_or_parse_options};
use crate::imod::flib::subrs::hvem::rdlist::{parselist2, rdlist};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::b3dutil::{extra_is_nbytes_and_flags, override_invert_mrc_origin};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_integer, pip_get_non_option_arg, pip_get_string, pip_get_three_floats,
    pip_get_three_integers,
};
use crate::imod::libcfshr::simplestat::array_min_max_mean_sd_fortran;
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_brief, iiu_close, iiu_read_lines, iiu_set_position,
};
use crate::imod::libiimod::unit_header::*;
use std::io::Write;

/// `parameter (numOptions = 27)` (`alterheader.f90:46`).
const ALTERHEADER_NUM_OPTIONS: i32 = 27;

/// Source fallback PIP table (`alterheader.f90:48`), kept as the 27
/// `@`-separated entries of the Fortran `options(1)` string.
const ALTERHEADER_OPTIONS: &str = "org:Origin:FT:@cel:CellSize:FT:@del:PixelSize:FT:@map:MapIndexes:IT:@\
sam:SampleSize:IT:@tlt:TiltCurrent:FT:@firsttlt:TiltOriginal:FT:@\
rottlt:RotateTilt:FT:@mmm:MinMaxMean:B:@rms:RootMeanSquare:B:@\
fixpixel:FixPixel:B:@gridfix:FixGrid:B:@feipixel:FeiPixel:I:@\
extrafix:FixExtra:B:@modefix:FixMode:B:@invertorg:InvertOrigin:B:@\
toggleorg:ToggleOrigin:B:@setmmm:SetMinMaxMean:FT:@real:RealMode:B:@\
fft:ComplexMode:B:@4bit:Change4BitMode:I:@ispg:SpaceGroup:I:@\
title:TitleToAdd:CH:@position:PositionForTitle:I:@remove:RemoveTitles:LI:@\
copy:CopyFromImage:FN:@help:usage:B:";

/// Original Fortran `program alterheader` (`alterheader.f90:10`).
pub fn alterheader() {
    unsafe {
        const NFUNC: usize = 28;
        const IDIM: i32 = 2100;
        let param: [&str; NFUNC] = [
            "ORG",
            "CEL",
            "DAT",
            "DEL",
            "MAP",
            "SAM",
            "TLT",
            "TLT_ORIG",
            "TLT_ROT",
            "LAB",
            "MMM",
            "RMS",
            "FIXPIXEL",
            "FIXEXTRA",
            "FIXMODE",
            "SETMMM",
            "FEIPIXEL",
            "INVERTORG",
            "REAL",
            "FFT",
            "ISPG",
            "VOLSTACK",
            "4BIT",
            "TOGGLEORG",
            "FIXGRID",
            "START",
            "HELP",
            "DONE",
        ];
        // `go to(1, 2, 3, ...), iwhich` at `alterheader.f90:216`.
        let computed_goto: [i32; NFUNC] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 12, 13, 17, 18, 19, 20, 21, 22, 23, 29, 24, 25,
            26, 27, 14, 15,
        ];

        // Fortran `Gw.d` editing (F2008 10.7.5.2.2): a value whose decimal
        // exponent falls in 0..=d prints in F editing with (d - exponent)
        // fraction digits followed by four blanks; anything else uses E
        // editing.  `{:w.d}` is plain F editing and matches neither.
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
        // A Fortran `read(5, ...)` on unit 5 at end of file aborts the run;
        // this reproduces the termination without the gfortran backtrace.
        let read_record = || -> String {
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
                std::process::exit(2);
            }
            line.trim_end_matches(['\r', '\n']).to_owned()
        };
        // List-directed `read(5,*)` for reals: values are separated by blanks
        // or commas, a slash ends the list leaving the rest unchanged, and
        // reading continues onto further records until the list is filled.
        let read_reals = |values: &mut [f32]| {
            let mut index = 0;
            while index < values.len() {
                let record = read_record();
                let mut ended = false;
                for token in record.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
                    if token.starts_with('/') {
                        ended = true;
                        break;
                    }
                    if index >= values.len() {
                        break;
                    }
                    match token.parse::<f32>() {
                        Ok(value) => values[index] = value,
                        // gfortran reports "Bad real number in item N of list
                        // input" and terminates the program.
                        Err(_) => std::process::exit(2),
                    }
                    index += 1;
                }
                if ended {
                    return;
                }
            }
        };
        let read_integers = |values: &mut [i32]| {
            let mut index = 0;
            while index < values.len() {
                let record = read_record();
                let mut ended = false;
                for token in record.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
                    if token.starts_with('/') {
                        ended = true;
                        break;
                    }
                    if index >= values.len() {
                        break;
                    }
                    match token.parse::<i32>() {
                        Ok(value) => values[index] = value,
                        // gfortran reports "Bad integer for item N in list
                        // input" and terminates the program.
                        Err(_) => std::process::exit(2),
                    }
                    index += 1;
                }
                if ended {
                    return;
                }
            }
        };

        // The single list-directed `read(5,*) itype, lens, n1, n2, v1, v2` at
        // `alterheader.f90:299` takes four integers then two reals from one
        // token stream, so it cannot be split into two reads.
        let read_mixed = |ints: &mut [i32], reals: &mut [f32]| {
            let total = ints.len() + reals.len();
            let mut index = 0;
            while index < total {
                let record = read_record();
                let mut ended = false;
                for token in record.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
                    if token.starts_with('/') {
                        ended = true;
                        break;
                    }
                    if index >= total {
                        break;
                    }
                    if index < ints.len() {
                        match token.parse::<i32>() {
                            Ok(value) => ints[index] = value,
                            Err(_) => std::process::exit(2),
                        }
                    } else {
                        match token.parse::<f32>() {
                            Ok(value) => reals[index - ints.len()] = value,
                            Err(_) => std::process::exit(2),
                        }
                    }
                    index += 1;
                }
                if ended {
                    return;
                }
            }
        };

        let mut if_add_title = -1_i32;
        // `PipReadOrParseOptions(options, numOptions, 'alterheader',
        // 'ERROR: ALTERHEADER -', .true., 1, 1, 0, numOptArg, numNonOptArg)`
        // (`alterheader.f90:60`).
        let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
        pip_read_or_parse_options(
            &[ALTERHEADER_OPTIONS],
            ALTERHEADER_NUM_OPTIONS,
            "alterheader",
            "ERROR: ALTERHEADER -",
            true,
            1,
            1,
            0,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );

        let pip_input = num_opt_arg > 0;
        let mut copy_from = false;
        let mut in_file = String::new();
        let mut string = String::new();
        let mut ind_pip_opt = 0_i32;
        if pip_input {
            if num_non_opt_arg == 0 {
                exit_error("Image filename must be entered");
            }
            let mut name: Vec<u8> = Vec::new();
            pip_get_non_option_arg(0, &mut name);
            in_file = String::from_utf8_lossy(&name).into_owned();
            ind_pip_opt = 0;
            let mut copy_name: Vec<u8> = Vec::new();
            copy_from = pip_get_string(b"CopyFromImage", &mut copy_name) == 0;
            if copy_from {
                string = String::from_utf8_lossy(&copy_name).into_owned();
            }
            if copy_from && num_opt_arg > 1 {
                exit_error("No other options can be entered with -copy");
            }
        } else {
            in_file = match getinout(1) {
                Ok((input, _)) => input,
                Err(_) => std::process::exit(2),
            };
        }
        //
        iiu_alt_brief(0);
        imopen(2, &in_file, "OLD");
        let (mut nxyz, mut mxyz, mut nxyzst) = ([0_i32; 3], [0_i32; 3], [0_i32; 3]);
        let (mut mode, mut dmin, mut dmax, mut dmean) = (0_i32, 0.0_f32, 0.0_f32, 0.0_f32);
        irdhdr(
            2,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
        //
        // Get starting state of inversion of origin and set it to output the same way
        let (mut iold, mut iflags, mut if_imod) = (0_i32, 0_i32, 0_i32);
        iiu_ret_mrc_version(2, &mut iold);
        iiu_ret_imod_flags(2, &mut iflags, &mut if_imod);
        let mut invert_origin = 0_i32;
        if iold > 0 || (if_imod > 0 && iflags & 4 != 0) {
            invert_origin = 1;
        }
        override_invert_mrc_origin(invert_origin);

        let mut title = [[b' '; 80]; 10];
        let mut ntitle = 0_i32;
        let mut listdel = [0_i32; 1000];
        let mut ndel = 0_i32;
        let mut cell = [0.0_f32; 6];
        let mut delt = [0.0_f32; 3];
        let mut tilt = [0.0_f32; 3];
        let mut mcrs = [0_i32; 3];
        let (mut origx, mut origy, mut origz) = (0.0_f32, 0.0_f32, 0.0_f32);
        let (mut itype, mut lens, mut n1, mut n2, mut n3) = (0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
        let (mut v1, mut v2) = (0.0_f32, 0.0_f32);
        let mut rms = 0.0_f32;
        let mut if_ok = 0_i32;
        let mut iwhich = 0_i32;
        let mut last_func = String::new();

        let mut goto_label = if copy_from { 28 } else { 30 };
        if !copy_from && !pip_input {
            println!(" If you make a mistake, interrupt with Ctrl-C instead of exiting with DONE");
        }
        loop {
            if goto_label == 30 {
                // Label 30 (`alterheader.f90:92`)
                if pip_input {
                    if_ok = 0;
                    ind_pip_opt += 1;
                    goto_label = 0;
                    match ind_pip_opt {
                        1 => {
                            if pip_get_three_floats(b"Origin", &mut origx, &mut origy, &mut origz)
                                == 0
                            {
                                goto_label = 1;
                            }
                        }
                        2 => {
                            iiu_ret_cell(2, &mut cell);
                            if {
                                let [cell_0, cell_1, cell_2, ..] = &mut cell;
                                pip_get_three_floats(b"CellSize", cell_0, cell_1, cell_2)
                            } == 0
                            {
                                goto_label = 2;
                            }
                        }
                        3 => {
                            if {
                                let [delt_0, delt_1, delt_2, ..] = &mut delt;
                                pip_get_three_floats(b"PixelSize", delt_0, delt_1, delt_2)
                            } == 0
                            {
                                goto_label = 4;
                            }
                        }
                        4 => {
                            if {
                                let [mcrs_0, mcrs_1, mcrs_2, ..] = &mut mcrs;
                                pip_get_three_integers(b"MapIndexes", mcrs_0, mcrs_1, mcrs_2)
                            } == 0
                            {
                                goto_label = 5;
                            }
                        }
                        5 => {
                            if {
                                let [mxyz_0, mxyz_1, mxyz_2, ..] = &mut mxyz;
                                pip_get_three_integers(b"SampleSize", mxyz_0, mxyz_1, mxyz_2)
                            } == 0
                            {
                                goto_label = 6;
                            }
                        }
                        6 => {
                            if {
                                let [tilt_0, tilt_1, tilt_2, ..] = &mut tilt;
                                pip_get_three_floats(b"TiltCurrent", tilt_0, tilt_1, tilt_2)
                            } == 0
                            {
                                goto_label = 7;
                            }
                        }
                        7 => {
                            if {
                                let [tilt_0, tilt_1, tilt_2, ..] = &mut tilt;
                                pip_get_three_floats(b"TiltOriginal", tilt_0, tilt_1, tilt_2)
                            } == 0
                            {
                                goto_label = 8;
                            }
                        }
                        8 => {
                            if {
                                let [tilt_0, tilt_1, tilt_2, ..] = &mut tilt;
                                pip_get_three_floats(b"RotateTilt", tilt_0, tilt_1, tilt_2)
                            } == 0
                            {
                                goto_label = 9;
                            }
                        }
                        9 => {
                            pip_get_boolean(b"MinMaxMean", &mut if_ok);
                            iwhich = 11;
                            if if_ok > 0 {
                                goto_label = 11;
                            }
                        }
                        10 => {
                            pip_get_boolean(b"RootMeanSquare", &mut if_ok);
                            iwhich = 12;
                            if if_ok > 0 {
                                goto_label = 16;
                            }
                        }
                        11 => {
                            pip_get_boolean(b"FixPixel", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 12;
                            }
                        }
                        12 => {
                            if pip_get_integer(b"FeiPixel", &mut if_ok) == 0 {
                                if if_ok <= 0 {
                                    if_ok = -1;
                                }
                                goto_label = 19;
                            }
                        }
                        13 => {
                            pip_get_boolean(b"FixExtra", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 13;
                            }
                        }
                        14 => {
                            pip_get_boolean(b"FixMode", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 17;
                            }
                        }
                        15 => {
                            pip_get_boolean(b"InvertOrigin", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 20;
                            }
                        }
                        16 => {
                            if pip_get_three_floats(
                                b"SetMinMaxMean",
                                &mut dmin,
                                &mut dmax,
                                &mut dmean,
                            ) == 0
                            {
                                goto_label = 18;
                            }
                        }
                        17 => {
                            pip_get_boolean(b"RealMode", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 21;
                            }
                        }
                        18 => {
                            pip_get_boolean(b"ComplexMode", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 22;
                            }
                        }
                        19 => {
                            if pip_get_integer(b"SpaceGroup", &mut iflags) == 0 {
                                goto_label = 23;
                            }
                        }
                        20 => {
                            if pip_get_integer(b"VolumeStack", &mut iflags) == 0 {
                                goto_label = 29;
                            }
                        }
                        21 => {
                            if pip_get_integer(b"Change4BitMode", &mut iflags) == 0 {
                                goto_label = 24;
                            }
                        }
                        22 => {
                            pip_get_boolean(b"ToggleOrigin", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 25;
                            }
                        }
                        23 => {
                            pip_get_boolean(b"FixGrid", &mut if_ok);
                            if if_ok > 0 {
                                goto_label = 26;
                            }
                        }
                        24 => {
                            let mut list: Vec<u8> = Vec::new();
                            if pip_get_string(b"RemoveTitles", &mut list) == 0 {
                                string = String::from_utf8_lossy(&list).into_owned();
                                let _ = parselist2(&string, &mut listdel, &mut ndel, &mut 1000);
                                if ndel == 1 && listdel[0] <= 0 {
                                    exit_error("Title number to remove must be positive");
                                }
                                goto_label = 10;
                            }
                        }
                        25 => {
                            let mut j = 0_i32;
                            if pip_get_integer(b"PositionForTitle", &mut j) == 0 {
                                if j <= 0 {
                                    exit_error("Position for title to add must be positive");
                                }
                                ndel = 1;
                                listdel[0] = 1 - j;
                                if_add_title = -2;
                            }
                        }
                        26 => {
                            let mut text: Vec<u8> = Vec::new();
                            if pip_get_string(b"TitleToAdd", &mut text) == 0 {
                                // `PipGetString('TitleToAdd', string)` fills the
                                // shared `string` variable that label 10 reads
                                // the new label from.
                                string = String::from_utf8_lossy(&text).into_owned();
                                title[0] = [b' '; 80];
                                let bytes = string.as_bytes();
                                let len = bytes.len().min(80);
                                title[0][..len].copy_from_slice(&bytes[..len]);
                                if if_add_title < -1 {
                                    goto_label = 10;
                                } else {
                                    if_add_title = 1;
                                }
                            }
                        }
                        27 => goto_label = 15,
                        _ => {}
                    }
                    if goto_label == 0 {
                        goto_label = 30;
                        continue;
                    }
                } else {
                    println!(
                        " Options: org, cel, dat, del, map, sam, tlt, tlt_orig, tlt_rot, lab, mmm,"
                    );
                    println!(
                        " rms, fixpixel, feipixel, fixextra, fixmode, invertorg, setmmm, real, fft,"
                    );
                    println!(" ispg, volstack, 4bit, toggleorg, fixgrid, start, help, OR done");
                    print!(" Enter option: ");
                    let _ = std::io::stdout().flush();
                    let funcin = read_record();
                    let funcin: String = funcin.chars().take(20).collect();
                    //
                    // Exit if a -1 is received after feipixel, it means there was no pixel
                    // available
                    if funcin.trim_end() == "-1" && last_func.trim_end() == "FEIPIXEL" {
                        std::process::exit(0);
                    }
                    let funcup = funcin.to_ascii_uppercase();
                    iwhich = 0;
                    for (i, name) in param.iter().enumerate() {
                        if funcup.trim_end() == *name {
                            iwhich = i as i32 + 1;
                        }
                    }
                    last_func = funcup;
                    if iwhich >= 1 && iwhich <= NFUNC as i32 {
                        goto_label = computed_goto[iwhich as usize - 1];
                    } else {
                        println!(" Not a legal entry, try again");
                        goto_label = 30;
                        continue;
                    }
                }
            }
            match goto_label {
                //
                // ORIGIN
                //
                1 => {
                    if !pip_input {
                        let mut origin = [0.; 3];
                        iiu_ret_origin(2, &mut origin);
                        [origx, origy, origz] = origin;
                        println!(
                            " Alter origin.  The origin is the offset FROM the first point in the image"
                        );
                        println!(
                            " file TO the center of the coordinate system, expressed in true coordinates."
                        );
                        println!(
                            " Current x, y, z:{}{}{}",
                            g_edit(origx, 14, 5),
                            g_edit(origy, 14, 5),
                            g_edit(origz, 14, 5)
                        );
                        print!("New x, y, z: ");
                        let _ = std::io::stdout().flush();
                        let mut values = [origx, origy, origz];
                        read_reals(&mut values);
                        [origx, origy, origz] = values;
                    }
                    iiu_alt_origin(2, &[origx, origy, origz]);
                    goto_label = 30;
                }
                //
                // CELL
                //
                2 => {
                    if !pip_input {
                        iiu_ret_cell(2, &mut cell);
                        println!(" Alter cell.  Current size and angles:");
                        println!(
                            "{}{}{}{:9.3}{:9.3}{:9.3}",
                            g_edit(cell[0], 14, 5),
                            g_edit(cell[1], 14, 5),
                            g_edit(cell[2], 14, 5),
                            cell[3],
                            cell[4],
                            cell[5]
                        );
                        print!("New size and angles: ");
                        let _ = std::io::stdout().flush();
                        read_reals(&mut cell);
                    }
                    if cell[0] > 0.0 && cell[1] > 0.0 && cell[2] > 0.0 {
                        iiu_alt_cell(2, &cell);
                    } else {
                        if pip_input {
                            exit_error("The values for the cell entry must positive");
                        }
                        println!(" No good, cell(1-3) must be positive");
                    }
                    goto_label = 30;
                }
                //
                // data type etc
                //
                3 => {
                    iiu_ret_data_type(2, &mut itype, &mut lens, &mut n1, &mut n2, &mut v1, &mut v2);
                    println!(" Alter data type.  Current type, lens, n1, n2, v1, v2:");
                    println!("{itype:5}{lens:5}{n1:5}{n2:5}{v1:10.3}{v2:10.3}");
                    println!(
                        " Enter new type (0 regular serial sections, 1 tilt series, 2 serial stereo"
                    );
                    print!(" pairs, 3 averaged serial sections, 4 averaged serial stereo pairs): ");
                    let _ = std::io::stdout().flush();
                    let mut one = [itype];
                    read_integers(&mut one);
                    itype = one[0];
                    if itype < 0 || itype > 4 {
                        println!(" No good, type must be 0-4");
                        goto_label = 30;
                        continue;
                    }
                    if itype == 0 {
                        println!(" You do not need to change any of the other parameters");
                    } else if itype == 1 {
                        n2 = 0;
                        print!(" 1, 2, or 3 if the tilt is around the X, Y or Z axis: ");
                        let _ = std::io::stdout().flush();
                        let mut one = [n1];
                        read_integers(&mut one);
                        n1 = one[0];
                        if n1 <= 0 || n1 > 3 {
                            println!(" Value no good");
                            goto_label = 30;
                            continue;
                        }
                        print!(
                            " Increment in tilt angle between views, tilt angle of first view: "
                        );
                        let _ = std::io::stdout().flush();
                        let mut two = [v1, v2];
                        read_reals(&mut two);
                        [v1, v2] = two;
                    }
                    if itype >= 3 {
                        print!(" Number of original sections averaged into one section: ");
                        let _ = std::io::stdout().flush();
                        let mut one = [n1];
                        read_integers(&mut one);
                        n1 = one[0];
                        println!(
                            " Enter the spacing between the original section numbers contributing"
                        );
                        print!("    to successive sections in this file: ");
                        let _ = std::io::stdout().flush();
                        let mut one = [n2];
                        read_integers(&mut one);
                        n2 = one[0];
                    }
                    if itype == 2 || itype == 4 {
                        print!(" Tilt angles of left and right eye views: ");
                        let _ = std::io::stdout().flush();
                        let mut two = [v1, v2];
                        read_reals(&mut two);
                        [v1, v2] = two;
                    }
                    println!(" Proposed new type, lens, n1, n2, v1, v2:");
                    println!("{itype:5}{lens:5}{n1:5}{n2:5}{v1:10.3}{v2:10.3}");
                    print!(" Enter / to accept, or a new type, lens, n1, n2, v1, v2: ");
                    let _ = std::io::stdout().flush();
                    let mut ints = [itype, lens, n1, n2];
                    let mut reals = [v1, v2];
                    read_mixed(&mut ints, &mut reals);
                    [itype, lens, n1, n2] = ints;
                    [v1, v2] = reals;
                    if itype >= 0 && itype <= 4 {
                        iiu_alt_data_type(2, itype, lens, n1, n2, v1, v2);
                    } else {
                        println!(" No good, type must be 0-4");
                    }
                    goto_label = 30;
                }
                //
                // DELTA
                //
                4 => {
                    iiu_ret_sample(2, mxyz.as_mut_ptr());
                    iiu_ret_cell(2, &mut cell);
                    if !pip_input {
                        iiu_ret_delta(2, delt.as_mut_ptr());
                        println!(
                            " Alter delta - changes cell sizes to achieve desired pixel spacing"
                        );
                        println!(
                            " Current delta x, y, z:{}{}{}",
                            g_edit(delt[0], 14, 5),
                            g_edit(delt[1], 14, 5),
                            g_edit(delt[2], 14, 5)
                        );
                        print!("New delta x, y, z: ");
                        let _ = std::io::stdout().flush();
                        read_reals(&mut delt);
                    }
                    if delt[0] > 0.0 && delt[1] > 0.0 && delt[2] > 0.0 {
                        for i in 0..3 {
                            cell[i] = mxyz[i] as f32 * delt[i];
                        }
                        iiu_alt_cell(2, &cell);
                    } else {
                        if pip_input {
                            exit_error("The values for the delta entry must positive");
                        }
                        println!(" No good, must be positive");
                    }
                    goto_label = 30;
                }
                //
                // MAPPING
                //
                5 => {
                    if !pip_input {
                        iiu_ret_axis_map(2, &mut mcrs);
                        println!(
                            " Alter mapping.  Current mapping constants:{:3}{:3}{:3}",
                            mcrs[0], mcrs[1], mcrs[2]
                        );
                        print!("New constants: ");
                        let _ = std::io::stdout().flush();
                        read_integers(&mut mcrs);
                    }
                    n1 = 0;
                    n2 = 0;
                    n3 = 0;
                    for i in 0..3 {
                        if mcrs[i] == 1 {
                            n1 += 1;
                        }
                        if mcrs[i] == 2 {
                            n2 += 1;
                        }
                        if mcrs[i] == 3 {
                            n3 += 1;
                        }
                    }
                    if n1 == 1 && n2 == 1 && n3 == 1 {
                        iiu_alt_axis_map(2, &mcrs);
                    } else {
                        if pip_input {
                            exit_error(
                                "The values for the map entry must be a permutation of 1, 2, 3",
                            );
                        }
                        println!(" No good, must be a permutation of 1, 2, 3");
                    }
                    goto_label = 30;
                }
                //
                // SAMPLING
                //
                6 => {
                    if !pip_input {
                        iiu_ret_sample(2, mxyz.as_mut_ptr());
                        println!(
                            " Alter sampling (mxyz).  Current x, y, z:{:5}{:5}{:5}",
                            mxyz[0], mxyz[1], mxyz[2]
                        );
                        print!("New x, y, z: ");
                        let _ = std::io::stdout().flush();
                        read_integers(&mut mxyz);
                    }
                    if mxyz[0] > 0 && mxyz[1] > 0 && mxyz[2] > 0 {
                        iiu_alt_sample(2, mxyz.as_mut_ptr());
                    } else {
                        if pip_input {
                            exit_error("The values for the sample entry must be positive");
                        }
                        println!(" No good, must be positive");
                    }
                    goto_label = 30;
                }
                //
                // TILT - current angles
                //
                7 => {
                    if !pip_input {
                        iiu_ret_tilt(2, tilt.as_mut_ptr());
                        println!(
                            " Alter current tilt angles.  Current angles:{:6.1}{:6.1}{:6.1}",
                            tilt[0], tilt[1], tilt[2]
                        );
                        // FORMAT 117 is `6f6.1` with three items, so output
                        // stops at the fourth data edit descriptor and the
                        // 'New current angles: ' prompt is never reached.
                        read_reals(&mut tilt);
                    }
                    iiu_alt_tilt(2, tilt.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // TILT_ORIG - original angles
                //
                8 => {
                    if !pip_input {
                        iiu_ret_tilt_orig(2, tilt.as_mut_ptr());
                        println!(
                            " Alter original tilt angles.  Current angles:{:6.1}{:6.1}{:6.1}",
                            tilt[0], tilt[1], tilt[2]
                        );
                        // FORMAT 118 is `6f6.1` with three items; see above.
                        read_reals(&mut tilt);
                    }
                    iiu_alt_tilt_orig(2, tilt.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // TILT_ROT - rotate current angles
                //
                9 => {
                    if !pip_input {
                        println!(" Rotate current tilt angles.");
                        print!("Angles to rotate by: ");
                        let _ = std::io::stdout().flush();
                        read_reals(&mut tilt);
                    }
                    iiu_alt_tilt_rot(2, tilt.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // LAB - delete selected labels or add one
                //
                10 => {
                    iiu_ret_labels(2, title.as_mut_ptr().cast(), &raw mut ntitle);
                    if !pip_input {
                        println!(" Delete labels or add one label.  Current labels are:");
                        println!();
                        for i in 0..ntitle as usize {
                            println!("{:3} {}", i + 1, String::from_utf8_lossy(&title[i][..76]));
                        }
                        println!();
                        println!("To delete labels, enter numbers of labels to delete (ranges ok)");
                        println!(
                            "To add a label, enter 0 or the NEGATIVE of the label number to add it after"
                        );
                        let stdin = std::io::stdin();
                        let mut lock = stdin.lock();
                        let _ = rdlist(&mut lock, &mut listdel, &mut ndel);
                    }
                    let newtitle;
                    if ndel == 1 && listdel[0] <= 0 {
                        if ntitle >= 10 {
                            if pip_input {
                                if_add_title = 1;
                            } else {
                                println!(" You need to delete some labels before adding any");
                            }
                            goto_label = 30;
                            continue;
                        }
                        let ifdel = (-listdel[0]).min(ntitle);
                        let mut iold = ntitle;
                        while iold >= ifdel + 1 {
                            title[iold as usize] = title[iold as usize - 1];
                            iold -= 1;
                        }
                        if !pip_input {
                            println!(" Enter new label");
                            string = read_record();
                        }
                        let mut label = [b' '; 80];
                        let bytes = string.as_bytes();
                        let len = bytes.len().min(80);
                        label[..len].copy_from_slice(&bytes[..len]);
                        title[ifdel as usize] = label;
                        newtitle = ntitle + 1;
                    } else {
                        let mut count = 0_i32;
                        for iold in 0..ntitle as usize {
                            let mut ifdel = 0;
                            for id in 0..ndel as usize {
                                if iold as i32 + 1 == listdel[id] {
                                    ifdel = 1;
                                }
                            }
                            if ifdel == 0 {
                                title[count as usize] = title[iold];
                                count += 1;
                            }
                        }
                        newtitle = count;
                    }
                    if pip_input {
                        if_ok = 1;
                    } else {
                        if newtitle > 0 {
                            println!(" New label list would be:");
                            println!();
                            for i in 0..newtitle as usize {
                                println!(
                                    "{:3} {}",
                                    i + 1,
                                    String::from_utf8_lossy(&title[i][..76])
                                );
                            }
                        } else {
                            println!(" New label list would be empty");
                        }
                        println!();
                        print!(" 1 to confirm changing to this label list, 0 not to: ");
                        let _ = std::io::stdout().flush();
                        let mut one = [if_ok];
                        read_integers(&mut one);
                        if_ok = one[0];
                    }
                    if if_ok != 0 {
                        iiu_alt_labels(2, title.as_mut_ptr().cast(), newtitle);
                    }
                    goto_label = 30;
                }
                //
                // MMM - recompute min/max/mean
                //
                11 => {
                    let max_lines = IDIM * IDIM / nxyz[0];
                    let num_chunks = (nxyz[1] + max_lines - 1) / max_lines;
                    println!(" Recomputing min/max/mean of images - takes a while...");
                    iiu_set_position(2, 0, 0);
                    dmin = 1.0e10;
                    dmax = -1.0e10;
                    let mut tsum = 0.0_f64;
                    let mut sumsq = 0.0_f64;
                    let mut totn = 0.0_f64;
                    let mut array = vec![0.0_f32; (IDIM as usize) * (IDIM as usize)];
                    let mut failed = false;
                    'sections: for _iz in 1..=nxyz[2] {
                        for i_chunk in 1..=num_chunks {
                            let num_lines = max_lines.min(nxyz[1] - (i_chunk - 1) * max_lines);
                            if iiu_read_lines(2, array.as_mut_ptr().cast(), num_lines) != 0 {
                                failed = true;
                                break 'sections;
                            }
                            let (mut dmins, mut dmaxs, mut sums, mut sumsqs, mut sd) =
                                (0.0_f32, 0.0_f32, 0.0_f64, 0.0_f64, 0.0_f32);
                            array_min_max_mean_sd_fortran(
                                &array,
                                &nxyz[0],
                                &num_lines,
                                &1,
                                &nxyz[0],
                                &1,
                                &num_lines,
                                &mut dmins,
                                &mut dmaxs,
                                &mut sums,
                                &mut sumsqs,
                                &mut dmean,
                                &mut sd,
                            );
                            dmin = dmin.min(dmins);
                            dmax = dmax.max(dmaxs);
                            tsum += sums;
                            sumsq += sumsqs;
                            totn += f64::from(nxyz[0] * num_lines);
                        }
                    }
                    if failed {
                        println!();
                        println!(" ERROR: ALTERHEADER - reading file");
                        std::process::exit(1);
                    }
                    let dmeans = tsum / totn;
                    rms = ((sumsq - totn * dmeans * dmeans) / totn).sqrt() as f32;
                    dmean = dmeans as f32;
                    iiu_alt_rms(2, rms);
                    if iwhich == 12 {
                        println!(" New RMS value = {}", g_edit(rms, 13, 5));
                    }
                    goto_label = 30;
                }
                //
                // RMS - first inform of current RMS value
                //
                16 => {
                    iiu_ret_rms(2, &mut rms);
                    println!(" Current RMS value = {}", g_edit(rms, 13, 5));
                    goto_label = 11;
                }
                //
                // FIXPIXEL
                12 => {
                    println!(" Changing sample and cell sizes to match image size, ");
                    println!(" which will make pixel spacing be 1.0 1.0 1.0.");
                    iiu_ret_cell(2, &mut cell);
                    cell[0] = nxyz[0] as f32;
                    cell[1] = nxyz[1] as f32;
                    cell[2] = nxyz[2] as f32;
                    iiu_alt_cell(2, &cell);
                    iiu_alt_sample(2, nxyz.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // FIXGRID
                26 => {
                    println!(
                        " Changing sample size to match image size while preserving pixel spacing"
                    );
                    iiu_ret_cell(2, &mut cell);
                    iiu_ret_sample(2, mxyz.as_mut_ptr());
                    for i in 0..3 {
                        cell[i] = (cell[i] / mxyz[i] as f32) * nxyz[i] as f32;
                    }
                    iiu_alt_cell(2, &cell);
                    iiu_alt_sample(2, nxyz.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // FIXPIECES - Remove flag for piece coordinates from header
                //
                13 => {
                    println!(" Marking header as not containing any piece coordinates.");
                    println!(" This will make other extended header data inaccessible");
                    let mut extended_type = [0; 2];
                    iiu_ret_extended_type(2, &mut extended_type);
                    let [nbytex, mut iflag] = extended_type;
                    if (iflag / 2) % 2 > 0 {
                        iflag -= 2;
                    }
                    iiu_alt_extended_type(2, &[nbytex, iflag]);
                    goto_label = 30;
                }
                //
                // FIXMODE - change between 1 and 6
                //
                17 => {
                    if mode != 6 && mode != 1 {
                        let message = "Only mode 6 can be changed to mode 1, or 1 to 6";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }
                    //
                    mode = 7 - mode;
                    println!("\nChanging mode to{mode:2}");
                    iiu_alt_mode(2, mode);
                    if dmax > 32767.0 && mode == 1 {
                        println!(
                            "\nThe file maximum is{dmax:12.1} and numbers bigger than 32767 will not be"
                        );
                        println!(" represented correctly in this mode.");
                    }
                    if dmin < 0.0 && mode == 6 {
                        println!(
                            "\nThe file minimum is{dmin:12.1} and negative numbers will not be"
                        );
                        println!(" represented correctly in this mode.");
                    }
                    goto_label = 30;
                }
                //
                // SETMMM - set the min, max, mean
                18 => {
                    if !pip_input {
                        println!(
                            " Alter min/max/mean.  Current values:{}{}{}",
                            g_edit(dmin, 15, 5),
                            g_edit(dmax, 15, 5),
                            g_edit(dmean, 15, 5)
                        );
                        // FORMAT 218 is `6g15.5` with three items; output stops
                        // at the fourth descriptor before the prompt.
                        let mut values = [dmin, dmax, dmean];
                        read_reals(&mut values);
                        [dmin, dmax, dmean] = values;
                    }
                    goto_label = 30;
                }
                //
                // FEIPIXEL - use the pixel size in extra header to set pixel spacing
                19 => {
                    let mut nbsym = 0_i32;
                    iiu_ret_num_extended(2, &mut nbsym);
                    if nbsym <= 0 {
                        println!(" No extended header information in this file");
                        goto_label = 30;
                        continue;
                    }
                    if nbsym > IDIM * IDIM * 4 {
                        println!(" Extended header data too large for array");
                        goto_label = 30;
                        continue;
                    }
                    let mut extra = vec![0_i32; (nbsym as usize + 3) / 4];
                    if iiu_ret_extended_data(2, &raw mut nbsym, extra.as_mut_ptr()) != 0 {
                        println!(" Error reading extended header data");
                        goto_label = 30;
                        continue;
                    }
                    let mut extended_type = [0; 2];
                    iiu_ret_extended_type(2, &mut extended_type);
                    let [num_int, num_real] = extended_type;
                    if num_int < 0 || extra_is_nbytes_and_flags(num_int, num_real) != 0 {
                        println!(" The extended header is not in Agard/FEI format");
                        goto_label = 30;
                        continue;
                    }
                    if num_real < 12 {
                        println!(
                            " There is no pixel size in this extended header (too few values per section)"
                        );
                        goto_label = 30;
                        continue;
                    }
                    let pixel = *extra.as_ptr().add(num_int as usize + 11).cast::<f32>() * 1.0e10;
                    if pixel <= 0.0 {
                        println!(
                            " Pixel size in extended header is not a usable value:{}",
                            g_edit(pixel, 13, 6)
                        );
                        goto_label = 30;
                        continue;
                    }
                    iiu_ret_delta(2, delt.as_mut_ptr());
                    iiu_ret_imod_flags(2, &mut iflags, &mut if_imod);
                    let mut no_binning = iflags & 2 != 0;
                    let mut i_binning = [1_i32; 3];
                    for i in 0..3 {
                        i_binning[i] = delt[i].round() as i32;
                        if (delt[i] - i_binning[i] as f32).abs() > 1.0e-6
                            || i_binning[i] <= 0
                            || i_binning[i] > 4
                        {
                            no_binning = true;
                        }
                    }
                    if no_binning {
                        i_binning = [1; 3];
                        if iflags & 2 != 0 {
                            println!(
                                "\nThe pixel size has already been transferred to the standard pixel spacing"
                            );
                        }
                        println!(
                            "Pixel size in extended header is{} Angstroms",
                            g_edit(pixel, 11, 4)
                        );
                        if !pip_input {
                            println!();
                            print!(
                                "Enter 1 to set the pixel spacing to this value, 0 not to, -1 to abort: "
                            );
                            let _ = std::io::stdout().flush();
                            let mut one = [if_ok];
                            read_integers(&mut one);
                            if_ok = one[0];
                        }
                        if if_ok < 0 {
                            if iflags & 2 != 0 {
                                println!(
                                    "\nThe pixel size has already been transferred to the standard pixel spacing"
                                );
                            } else {
                                println!();
                                println!(
                                    "The existing regular pixel spacing did not correspond to a binning"
                                );
                            }
                            println!("The pixel size in the extended header is not being used");
                            std::process::exit(0);
                        }
                        i_binning = [1; 3];
                    } else {
                        println!(
                            "Pixel size in extended header is{} Angstroms",
                            g_edit(pixel, 11, 4)
                        );
                        if i_binning[0] > 1 {
                            println!(
                                "  but the data seem to have been binned by{:2}",
                                i_binning[0]
                            );
                        }
                        if !pip_input {
                            print!(
                                "Enter 1 or -1 to set the pixel spacing to{}, 0 not to: ",
                                g_edit(pixel * i_binning[0] as f32, 11, 4)
                            );
                            let _ = std::io::stdout().flush();
                            let mut one = [if_ok];
                            read_integers(&mut one);
                            if_ok = one[0];
                        }
                    }
                    if if_ok == 0 {
                        goto_label = 30;
                        continue;
                    }

                    iflags |= 2;
                    iiu_alt_imod_flags(2, iflags);
                    iiu_ret_sample(2, mxyz.as_mut_ptr());
                    iiu_ret_cell(2, &mut cell);
                    for i in 0..3 {
                        cell[i] = mxyz[i] as f32 * pixel * i_binning[i] as f32;
                    }
                    iiu_alt_cell(2, &cell);
                    goto_label = 30;
                }
                //
                // INVERTORG  - invert the sign of the origin
                20 => {
                    let mut origin = [0.; 3];
                    iiu_ret_origin(2, &mut origin);
                    [origx, origy, origz] = origin;
                    origx = -origx;
                    origy = -origy;
                    origz = -origz;
                    println!(
                        "Inverting sign of origin: new origin = {}{}{}",
                        g_edit(origx, 15, 6),
                        g_edit(origy, 15, 6),
                        g_edit(origz, 15, 6)
                    );
                    iiu_alt_origin(2, &[origx, origy, origz]);
                    goto_label = 30;
                }
                //
                // TOGGLEORG  - invert the sign of the origin saved to file
                25 => {
                    println!("Inverting sign of origin saved to file");
                    invert_origin = 1 - invert_origin;
                    override_invert_mrc_origin(invert_origin);
                    goto_label = 30;
                }
                //
                // REAL: make an FFT real
                21 => {
                    if mode != 4 {
                        let message = "Must be mode 4 to change file to real";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }
                    iiu_alt_mode(2, 2);
                    mode = 2;
                    nxyz[0] *= 2;
                    iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr());
                    println!("Changing mode to {mode:1} and X size to {:6}", nxyz[0]);
                    goto_label = 30;
                }
                //
                // FFT: restore a real file
                22 => {
                    if mode != 2 || nxyz[0] % 2 != 0 {
                        let message = "Must be mode 2 and NX must be even to change file to fft";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }
                    iiu_alt_mode(2, 4);
                    mode = 4;
                    nxyz[0] /= 2;
                    iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr());
                    println!("Changing mode to {mode:1} and X size to {:6}", nxyz[0]);
                    goto_label = 30;
                }
                //
                // ISPG
                23 => {
                    if !pip_input {
                        iiu_ret_space_group(2, &mut iflags);
                        println!(" Alter space group.  Current space group:{iflags:3}");
                        print!("New space group: ");
                        let _ = std::io::stdout().flush();
                        let mut one = [iflags];
                        read_integers(&mut one);
                        iflags = one[0];
                    }
                    if iflags < 0 {
                        let message = "Space group entry must be non-negative";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }
                    iiu_alt_space_group(2, iflags);
                    goto_label = 30;
                }
                //
                // VOLSTACK
                29 => {
                    if !pip_input {
                        print!(
                            " Number of sections per volume in stack, or 0 to revert from being volume stack: "
                        );
                        let _ = std::io::stdout().flush();
                        let mut one = [iflags];
                        read_integers(&mut one);
                        iflags = one[0];
                    }
                    if iflags < 0 {
                        let message = "Entry for volume stack must be non-negative";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }
                    iiu_ret_sample(2, mxyz.as_mut_ptr());
                    iiu_ret_cell(2, &mut cell);

                    // revert, restore MZ = NZ
                    if iflags == 0 {
                        cell[2] = (cell[2] * nxyz[2] as f32) / mxyz[2] as f32;
                        mxyz[2] = nxyz[2];
                        iiu_alt_sample(2, mxyz.as_mut_ptr());
                        iiu_alt_cell(2, &cell);
                        iiu_alt_space_group(2, 1);
                        goto_label = 30;
                        continue;
                    }

                    // Or set to given # of sections if it is OK
                    if nxyz[2] / iflags < 2 {
                        let message = "Too many sections per volume; there must be at least two volumes in stack";
                        if pip_input {
                            exit_error(message);
                        }
                        // `string` is `character*320`; `print *` writes it padded.
                        println!(" {message:<320}");
                        goto_label = 30;
                        continue;
                    }

                    if nxyz[2] % iflags > 0 {
                        println!(
                            "WARNING: The # of sections per volume does not divide evenly into the total # of sections"
                        );
                    }
                    cell[2] = (cell[2] * iflags as f32) / mxyz[2] as f32;
                    mxyz[2] = iflags;
                    iiu_alt_sample(2, mxyz.as_mut_ptr());
                    iiu_alt_cell(2, &cell);
                    iiu_alt_space_group(2, 401);
                    goto_label = 30;
                }
                //
                // 4BIT
                24 => {
                    if !pip_input {
                        println!("Enter 1 to change stored mode for 4-bit data to 101; ");
                        print!(
                            "        -1 to change it from 101 to 0; or 0 to leave mode unchanged: "
                        );
                        let _ = std::io::stdout().flush();
                        let mut one = [iflags];
                        read_integers(&mut one);
                        iflags = one[0];
                    }
                    if_ok = iiu_alt_4_bit_mode(2, iflags);
                    if if_ok < 0 && iflags != 0 {
                        println!(" No change was made; the file was already in the given mode");
                    }
                    if if_ok <= 0 {
                        goto_label = 30;
                        continue;
                    }
                    let message = match if_ok {
                        1 => "The file must be byte or 4-bit mode",
                        2 => "This change is not allowed for a file with signed bytes",
                        _ => "The file has an odd X size and cannot be changed to a byte file",
                    };
                    if pip_input {
                        exit_error(message);
                    }
                    // `string` is `character*320`; `print *` writes it padded.
                    println!(" {message:<320}");
                    goto_label = 30;
                }
                //
                // START
                27 => {
                    iiu_ret_size(2, nxyz.as_mut_ptr(), mxyz.as_mut_ptr(), nxyzst.as_mut_ptr());
                    println!(
                        "Current start coordinates in X, Y, Z are: {:7}{:7}{:7}",
                        nxyzst[0], nxyzst[1], nxyzst[2]
                    );
                    print!("Enter new start X, Y, Z or / to leave unchanged: ");
                    let _ = std::io::stdout().flush();
                    read_integers(&mut nxyzst);
                    iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr());
                    goto_label = 30;
                }
                //
                // COPY FROM IMAGE
                28 => {
                    imopen(3, &string, "RO");
                    let (mut nxyz2, mut mxyz2) = ([0_i32; 3], [0_i32; 3]);
                    let (mut ierr2, mut dmins, mut dmaxs, mut sd) =
                        (0_i32, 0.0_f32, 0.0_f32, 0.0_f32);
                    irdhdr(
                        3,
                        nxyz2.as_mut_ptr(),
                        mxyz2.as_mut_ptr(),
                        &raw mut ierr2,
                        &raw mut dmins,
                        &raw mut dmaxs,
                        &raw mut sd,
                    );
                    if nxyz2[0] != nxyz[0] || nxyz2[1] != nxyz[1] {
                        exit_error("The image to copy from must be the same size in X and Y");
                    }
                    let mut cell2 = [0.0_f32; 6];
                    iiu_ret_cell(3, &mut cell2);
                    iiu_ret_cell(2, &mut cell);
                    cell[0] = cell2[0];
                    cell[1] = cell2[1];
                    mxyz[0] = mxyz2[0];
                    mxyz[1] = mxyz2[1];
                    delt[2] = cell2[2] / mxyz2[2] as f32;
                    mxyz[2] =
                        1.max((mxyz2[2] as f32 * nxyz[2] as f32 / nxyz2[2] as f32).round() as i32);
                    cell[2] = mxyz[2] as f32 * delt[2];
                    iiu_alt_cell(2, &cell);
                    iiu_alt_sample(2, mxyz.as_mut_ptr());
                    iiu_ret_size(
                        3,
                        nxyz2.as_mut_ptr(),
                        mxyz2.as_mut_ptr(),
                        nxyzst.as_mut_ptr(),
                    );
                    iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr());
                    let mut origin = [0.; 3];
                    iiu_ret_origin(3, &mut origin);
                    iiu_alt_origin(2, &origin);
                    iiu_ret_tilt(3, tilt.as_mut_ptr());
                    iiu_alt_tilt(2, tilt.as_mut_ptr());
                    iiu_close(3);
                    goto_label = 15;
                }
                //
                // HELP
                14 => {
                    println!();
                    println!(" org = change x,y,z origin");
                    println!(" cel = change cell size");
                    println!(
                        " dat = change data type (for tilt series, stereo pairs, or averaged sections)"
                    );
                    println!(" del = change delta (pixel size) directly");
                    println!(" map = change x,y,z mapping to rows, columns, sections");
                    println!(" sam = change mxyz sampling");
                    println!(" tlt = change alpha, beta, gamma current tilt angles");
                    println!(" tlt_orig = change original tilt angles");
                    println!(" tlt_rot = rotate current tilt angles");
                    println!(" lab = delete selected labels");
                    println!(" mmm = fix min/max/mean and set RMS value by reading all images");
                    println!(" rms = set RMS value, does same actions as mmm");
                    println!(
                        " fixpixel = fix pixel spacing by setting cell and sample sizes to image size"
                    );
                    println!(" fixgrid = set sample size to image size and preserve pixel size");
                    println!(
                        " feipixel = set pixel spacing from pixel size in Agard/FEI extended header"
                    );
                    println!("            sample sizes to image size");
                    println!(
                        " fixextra = fix extra header so that file does not look like a montage"
                    );
                    println!(
                        " fixmode = change mode from 6 to 1 (unsigned to signed integer) or 1 to 6"
                    );
                    println!(" setmmm = set min, max, mean to entered values");
                    println!(" real = change mode 4 file to mode 2 and change X size");
                    println!(" fft = change compatible-sized mode 2 file back to mode 4");
                    println!(" invertorg = change sign of origins in file and internally in IMOD");
                    println!(
                        " toggleorg = change origin sign in file, keep it the same within IMOD"
                    );
                    println!(" help = type this again");
                    println!(" done = exit");
                    println!();
                    goto_label = 30;
                }
                //
                15 => {
                    iiu_write_header(
                        2,
                        title.as_mut_ptr().cast(),
                        if_add_title,
                        dmin,
                        dmax,
                        dmean,
                    );
                    iiu_close(2);
                    imopen(3, &in_file, "RO");
                    irdhdr(
                        3,
                        nxyz.as_mut_ptr(),
                        mxyz.as_mut_ptr(),
                        &raw mut mode,
                        &raw mut dmin,
                        &raw mut dmax,
                        &raw mut dmean,
                    );
                    iiu_close(3);
                    std::process::exit(0);
                }
                _ => {
                    goto_label = 30;
                }
            }
        }
    }
}
