//! Translation of `IMOD/pysrc/trimvol`.
//!
//! `trimvol` is a Python script with no function definitions: its whole body
//! is the module's top level, translated here statement by statement as
//! [`trimvol`].  It composes `densmatch`, `findcontrast`, `newstack` and
//! `clip` command lines and runs them; by owner decision (2026-09-24) those
//! four are this crate's own programs and run **in process** through
//! [`run_cmd_in_process`], which keeps `runcmd`'s contract (collected output
//! lines, `ERROR:` lines and exit status on failure) — see `CLAUDE.md`, "Our
//! own commands are called in process".
//!
//! Python values: `PipGetTwoFloats` returns Python floats (doubles) and the
//! script formats them with `str.format`, i.e. `repr`.  This crate's PIP
//! parses into `f32`, so a float entry is written as the shortest decimal
//! that reads back as that `f32`, then in Python's `repr` shape (`0.0`,
//! `255.0`) — the same text Python prints for any entry of up to seven
//! significant digits, and a text that the called program parses back to the
//! same `f32` it would have parsed from Python's.

use crate::imod::pysrc::batchruntomo::py_str_float;
use crate::imod::pysrc::imodpy::{
    add_imod_bin_ignore_sighup, exit_from_imod_error, fmtstr, get_mrc_size, print_pid, prnstr,
    run_cmd_in_process,
};
use crate::imod::pysrc::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_integer, pip_get_non_option_arg,
    pip_get_string, pip_get_two_floats, pip_get_two_integers, pip_print_help,
    pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`pysrc/trimvol:1-379`).  Returns the status of
/// its final `sys.exit(0)`; every error path exits the process itself, as
/// `exitError`/`exitFromImodError`/`sys.exit(1)` do.
pub fn trimvol(arguments: &[OsString]) -> i32 {
    let progname = "trimvol";
    let prefix = format!("ERROR: {progname} - ");

    // `str(float)` of a PIP float entry; see the module comment.
    let py_float = |value: f32| -> String {
        py_str_float(
            format!("{value}")
                .parse::<f64>()
                .unwrap_or(f64::from(value)),
        )
    };
    let s = |value: i32| value.to_string();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        // `sys.stdout.write(prefix + " IMOD_DIR is not defined!\n")`
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    // Initializations
    let mut adjust = String::from("-ori");
    let mut fliparg = String::new();
    let mut sxarg = String::new();
    let mut syarg = String::new();
    let mut secout = String::new();

    // Fallbacks from ../manpages/autodoc2man 3 1 trimvol
    let options: Vec<String> = [
        "x:XStartAndEnd:IP:",
        "y:YStartAndEnd:IP:",
        "z:ZStartAndEnd:IP:",
        "nx:XSize:I:",
        "ny:YSize:I:",
        "nz:ZSize:I:",
        "sz:ZFindStartAndEnd:IP:",
        "sx:XFindStartAndEnd:IP:",
        "sy:YFindStartAndEnd:IP:",
        "c:ContrastBlackWhite:IP:",
        "meansd:ScaleToMeanAndSD:FP:",
        "mm:IntegerMinMax:FP:",
        "mode:ModeToOutput:I:",
        "rx:RotateX:B:",
        "yz:FlipYZ:B:",
        "format:FormatOfOutputFile:CH:",
        "i:IndexCoordinates:B:",
        "f:FlippedCoordinates:B:",
        "old:OldFlippedCoordinates:I:",
        "k:KeepOrigin:B:",
        ":PID:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|entry| (*entry).to_owned())
    .collect();

    // Startup and get input and output files
    // Special case: give some good error messages for eliminated/renamed options
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    if argv.iter().any(|argument| argument == "-s") {
        prnstr(
            &(prefix.clone()
                + "The -s option has been eliminated; use -sz instead and add -f "
                + "if coordinates are from a volume loaded with flipping"),
            "\n",
            false,
        );
        return 1;
    }

    let (_opts, nonopts) = pip_read_or_parse_options(&argv, &options, progname, 2, 1, 1, None);
    if nonopts != 2 {
        prnstr(&(prefix.clone() + "wrong number of arguments"), "\n", false);
        pip_print_help(progname, 0, 1, 1);
        return 1;
    }

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    let input_file = pip_get_non_option_arg(0).unwrap_or_default();
    let output_file = pip_get_non_option_arg(1).unwrap_or_default();
    if !std::path::Path::new(&input_file).exists() {
        exit_error(&format!("Input file {input_file} does not exist"));
    }

    // Get scaling-related options
    let (mut black, mut white) =
        pip_get_two_integers("ContrastBlackWhite", (0, 0)).unwrap_or((0, 0));
    let contrast = 1 - pip_get_err_no();

    let (intmin, intmax) = pip_get_two_floats("IntegerMinMax", (0., 0.)).unwrap_or((0., 0.));
    let inmm = 1 - pip_get_err_no();

    let (slicest, slicend) = pip_get_two_integers("ZFindStartAndEnd", (0, 0)).unwrap_or((0, 0));
    let zslices = 1 - pip_get_err_no();
    let (mut xsmin, mut xsmax) =
        pip_get_two_integers("XFindStartAndEnd", (-1, -1)).unwrap_or((-1, -1));
    let (mut ysmin, mut ysmax) =
        pip_get_two_integers("YFindStartAndEnd", (-1, -1)).unwrap_or((-1, -1));

    let (targ_mean, targ_sd) = pip_get_two_floats("ScaleToMeanAndSD", (0., 0.)).unwrap_or((0., 0.));
    let meansd = 1 - pip_get_err_no();

    let mut mode = pip_get_integer("ModeToOutput", 0).unwrap_or(0);
    let if_mode = 1 - pip_get_err_no();
    let mut contout = String::new();
    let mut clip_mode = String::new();
    if if_mode != 0 {
        if contrast != 0 || (zslices != 0 && meansd == 0) {
            exit_error("You cannot enter -mode with -c, or when running findcontrast");
        }
        if inmm + meansd == 0 {
            contout = format!("-mode {mode}");
            clip_mode = format!("-m {mode}");
        }
    }

    if contrast + inmm + zslices > 1 || contrast + inmm + meansd > 1 {
        exit_error("You cannot enter -c and -mm with each other or with -sz or -meansd");
    }

    // Manage output format
    let oformat = pip_get_string("FormatOfOutputFile", "").unwrap_or_default();
    let allowed = ["HDF", "MRC", "TIFF", "TIF"];
    let mut newst_format = String::new();
    let mut clip_format = String::new();
    if !oformat.is_empty() {
        if !allowed.contains(&oformat.to_uppercase().as_str()) {
            exit_error(&format!("Output format {oformat} is not a valid format"));
        }
        newst_format = format!("-format {oformat}");
        clip_format = format!("-f {oformat}");
    }

    // Get size or coordinate limit options
    let mut xsize = pip_get_integer("XSize", 0).unwrap_or(0);
    let ifxsz = 1 - pip_get_err_no();
    let (mut xstart, mut xend) = pip_get_two_integers("XStartAndEnd", (0, 0)).unwrap_or((0, 0));
    let ifxse = 1 - pip_get_err_no();
    if ifxse + ifxsz > 1 {
        exit_error("You cannot enter both -x and -nx options");
    }

    let mut ysize = pip_get_integer("YSize", 0).unwrap_or(0);
    let mut ifysz = 1 - pip_get_err_no();
    let (mut ystart, mut yend) = pip_get_two_integers("YStartAndEnd", (0, 0)).unwrap_or((0, 0));
    let mut ifyse = 1 - pip_get_err_no();
    if ifyse + ifysz > 1 {
        exit_error("You cannot enter both -y and -ny options");
    }

    let mut zsize = pip_get_integer("ZSize", 0).unwrap_or(0);
    let mut ifzsz = 1 - pip_get_err_no();
    let (mut zstart, mut zend) = pip_get_two_integers("ZStartAndEnd", (0, 0)).unwrap_or((0, 0));
    let mut ifzse = 1 - pip_get_err_no();
    if ifzse + ifzsz > 1 {
        exit_error("You cannot enter both -z and -nz options");
    }

    // Flipping and rotation options
    let flipyz = pip_get_boolean("FlippedCoordinates", 0).unwrap_or(0);
    let old_flip = pip_get_integer("OldFlippedCoordinates", 0).unwrap_or(0);
    if flipyz != 0 {
        fliparg = String::from("-flip");
        // Python `//` floors; a negative entry is never > 0 either way.
        if old_flip.div_euclid(2) > 0 {
            fliparg += " -oldflip";
        }
    }

    // `doflip` starts as the boolean and becomes the clip command name.
    let doflip_entered = pip_get_boolean("FlipYZ", 0).unwrap_or(0);
    let dorot = pip_get_boolean("RotateX", 0).unwrap_or(0);
    if dorot != 0 && doflip_entered != 0 {
        exit_error("You cannot use both -yz and -rx options");
    }
    let mut doflip = String::new();
    if doflip_entered != 0 {
        doflip = String::from("flipyz");
    }
    if dorot != 0 {
        doflip = String::from("rotx");
    }

    if pip_get_boolean("KeepOrigin", 0).unwrap_or(0) != 0 {
        adjust = String::new();
    }
    let index = pip_get_boolean("IndexCoordinates", 0).unwrap_or(0);

    // Get file size
    let (nx, ny, nz) = match get_mrc_size(&input_file) {
        Ok(size) => size,
        Err(_) => exit_from_imod_error(progname),
    };

    // If flipped coordinated, swap appropriate entries
    if flipyz != 0 {
        std::mem::swap(&mut ysize, &mut zsize);
        std::mem::swap(&mut ifysz, &mut ifzsz);
        let stmp = (ystart, yend);
        // Python `%` takes the sign of the divisor: -1 % 2 is 1.
        if old_flip.rem_euclid(2) > 0 {
            (ystart, yend) = (zstart, zend);
        } else if index != 0 {
            (ystart, yend) = (ny - 1 - zend, ny - 1 - zstart);
        } else {
            (ystart, yend) = (ny + 1 - zend, ny + 1 - zstart);
        }

        (zstart, zend) = stmp;
        std::mem::swap(&mut ifyse, &mut ifzse);
    }

    // Check and set up the X coordinates
    let mut xoffset = 0;
    let mut yoffset = 0;
    if ifxsz != 0 {
        if xsize <= 0 || xsize > nx {
            exit_error(&fmtstr("Illegal X size in -nx {}", &[s(xsize)]));
        }
    } else {
        xsize = nx;
    }
    if ifxse != 0 {
        if index == 0 {
            xstart -= 1;
            xend -= 1;
        }
        if xend < 0 || xstart >= nx || xstart > xend {
            exit_error(&fmtstr(
                "X coordinates out of range for file in -x {},{}",
                &[s(xstart + index), s(xend + index)],
            ));
        }
        xsize = xend + 1 - xstart;
        xoffset = xstart + xsize.div_euclid(2) - nx.div_euclid(2);
    }

    // Check and set up Y coordinates
    let inlet = if flipyz != 0 { "z" } else { "y" };
    if ifysz != 0 {
        if ysize <= 0 || ysize > ny {
            exit_error(&fmtstr(
                "Illegal {} size in -n{} {}",
                &[inlet.to_owned(), inlet.to_owned(), s(ysize)],
            ));
        }
    } else {
        ysize = ny;
    }
    if ifyse != 0 {
        if index == 0 {
            ystart -= 1;
            yend -= 1;
        }
        if yend < 0 || ystart >= ny || ystart > yend {
            exit_error(&fmtstr(
                "{} coordinates out of range for file in -{} {},{}",
                &[
                    inlet.to_uppercase(),
                    inlet.to_owned(),
                    s(ystart + index),
                    s(yend + index),
                ],
            ));
        }
        ysize = yend + 1 - ystart;
        yoffset = ystart + ysize.div_euclid(2) - ny.div_euclid(2);
    }

    // Check and set up Z section list if either entry given
    let inlet = if flipyz != 0 { "y" } else { "z" };
    if ifzsz != 0 {
        if zsize <= 0 || zsize > nz {
            exit_error(&fmtstr(
                "Illegal {} size in -n{} {}",
                &[inlet.to_owned(), inlet.to_owned(), s(zsize)],
            ));
        }
        zstart = (nz - zsize).div_euclid(2);
        zend = zstart + zsize - 1;
        secout = fmtstr("-sec {}-{}", &[s(zstart), s(zend)]);
    }

    if ifzse != 0 {
        if index == 0 {
            zstart -= 1;
            zend -= 1;
        }
        if zend < 0 || zstart >= nz || zstart > zend {
            exit_error(&fmtstr(
                "{} coordinates out of range for file in -{} {},{}",
                &[
                    inlet.to_owned(),
                    inlet.to_owned(),
                    s(zstart + index),
                    s(zend + index),
                ],
            ));
        }
        secout = fmtstr("-sec {}-{}", &[s(zstart), s(zend)]);
        if zstart < 0 || zend >= nz {
            secout += " -blank";
        }
    }

    // Process the entries for X and Y limits in findcontrast
    if xsmin >= 0 && xsmax > 0 {
        if index == 0 {
            xsmin -= 1;
            xsmax -= 1;
        }
        sxarg = fmtstr("-xminmax {},{}", &[s(xsmin), s(xsmax)]);
    }

    if ysmin >= 0 && ysmax > 0 {
        if index == 0 {
            ysmin -= 1;
            ysmax -= 1;
        }
        syarg = fmtstr("-yminmax {},{}", &[s(ysmin), s(ysmax)]);
    }

    // Take care of converting other contrast entries to arguments
    if contrast != 0 {
        contout = fmtstr("-mode 0 -con {},{}", &[s(black), s(white)]);
    }
    if inmm != 0 {
        if if_mode == 0 {
            mode = 1;
        }
        contout = fmtstr(
            "-mode {} -sca {},{}",
            &[s(mode), py_float(intmin), py_float(intmax)],
        );
    }

    // Check entered slice limits for scaling depending on flipping
    let mut slicelim = nz;
    let mut ylim = ny;
    if !fliparg.is_empty() {
        slicelim = ny;
        ylim = nz;
    }
    if zslices != 0 && (slicest < 1 || slicend > slicelim || slicest > slicend) {
        exit_error(&fmtstr(
            "Slices out of range for file in -sz {},{}",
            &[s(slicest), s(slicend)],
        ));
    }

    let mut newstout = output_file.clone();
    if !doflip.is_empty() {
        newstout = format!("{input_file}.tmp.{}", std::process::id());
    }

    // If given target mean/SD, find the scaling factors
    if meansd != 0 {
        if sxarg.is_empty() {
            xsmin = (0.1 * f64::from(nx)) as i32;
            xsmax = (0.9 * f64::from(nx)) as i32;
        }
        if syarg.is_empty() {
            ysmin = (0.1 * f64::from(ylim)) as i32;
            ysmax = (0.9 * f64::from(ylim)) as i32;
        }
        let (mut zsmin, mut zsmax);
        if zslices != 0 {
            zsmin = slicest - 1;
            zsmax = slicend - 1;
        } else {
            zsmin = 0;
            zsmax = slicelim - 1;
        }
        if flipyz != 0 {
            if old_flip.div_euclid(2) > 0 {
                (ysmin, ysmax, zsmin, zsmax) = (zsmin, zsmax, ysmin, ysmax);
            } else {
                (ysmin, ysmax, zsmin, zsmax) = (ny - 1 - zsmax, ny - 1 - zsmin, ysmin, ysmax);
            }
        }

        let comlines = vec![
            format!("ScaledFile {input_file}"),
            fmtstr(
                "TargetMeanAndSD {},{}",
                &[py_float(targ_mean), py_float(targ_sd)],
            ),
            String::from("ReportOnly 1"),
            fmtstr("XMinAndMax {},{}", &[s(xsmin), s(xsmax)]),
            fmtstr("YMinAndMax {},{}", &[s(ysmin), s(ysmax)]),
            fmtstr("ZMinAndMax {},{}", &[s(zsmin), s(zsmax)]),
        ];
        prnstr(
            "Running densmatch to find scaling to mean and SD...",
            "\n",
            false,
        );
        let dens_lines = match run_cmd_in_process("densmatch -StandardInput", Some(&comlines), None)
        {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => exit_from_imod_error(progname),
        };

        for line in &dens_lines {
            prnstr(line.trim_end_matches(['\r', '\n']), "\n", false);
            if line.starts_with("Scale factors to") {
                let lsplit: Vec<&str> = line.split_whitespace().collect();
                // `try: ... except Exception: pass`
                if lsplit.len() >= 2 {
                    if let (Ok(multfac), Ok(sdfac)) = (
                        lsplit[lsplit.len() - 2].parse::<f64>(),
                        lsplit[lsplit.len() - 1].parse::<f64>(),
                    ) {
                        if if_mode == 0 {
                            mode = 0;
                        }
                        contout = fmtstr(
                            "-mode {} -multadd {},{}",
                            &[s(mode), py_str_float(multfac), py_str_float(sdfac)],
                        );
                    }
                }
            }
        }

        if contout.is_empty() {
            exit_error("Cannot find scaling information in output of densmatch");
        }

    // Or run findcontrast if Z slices were entered
    } else if zslices != 0 {
        slicelim = nz;
        if !fliparg.is_empty() {
            slicelim = ny;
        }
        if slicest < 1 || slicend > slicelim || slicest > slicend {
            exit_error(&fmtstr(
                "Slices out of range for file in -sz {},{}",
                &[s(slicest), s(slicend)],
            ));
        }
        prnstr(
            &format!("Determining byte scaling of {input_file}..."),
            "\n",
            false,
        );
        let findcom = fmtstr(
            "findcontrast -slice {},{} {} {} {} \"{}\"",
            &[
                s(slicest),
                s(slicend),
                fliparg.clone(),
                sxarg.clone(),
                syarg.clone(),
                input_file.clone(),
            ],
        );
        prnstr(&findcom, "\n", false);
        let findlines = match run_cmd_in_process(&findcom, None, None) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => exit_from_imod_error(progname),
        };

        // Get the black white while printing the lines
        let mut found_black: Option<i32> = None;
        for line in &findlines {
            prnstr(line.trim(), "\n", false);
            if found_black.is_none() && line.contains("Implied") {
                if let Some(ind) = line.find("are ").filter(|ind| *ind > 0) {
                    let bwsplit: Vec<&str> = line[ind + 3..].split_whitespace().collect();
                    if bwsplit.len() > 2 {
                        // Python `int()`: a non-integer token raises and the
                        // script ends with a traceback and status 1.
                        let parse = |token: &str| -> i32 {
                            token.parse::<i32>().unwrap_or_else(|_| {
                                eprintln!(
                                    "ValueError: invalid literal for int() with base 10: '{token}'"
                                );
                                std::process::exit(1)
                            })
                        };
                        found_black = Some(parse(bwsplit[0]));
                        white = parse(bwsplit[2]);
                    }
                }
            }
        }

        match found_black {
            Some(value) => black = value,
            None => exit_error("Findcontrast failed to return scaling values"),
        }
        contout = fmtstr("-mode 0 -con {},{}", &[s(black), s(white)]);
    }

    // compose and run the newstack command
    let newstcom = fmtstr(
        "newstack -siz {},{} -off {},{} {} {} {} {} \"{}\" \"{}\"",
        &[
            s(xsize),
            s(ysize),
            s(xoffset),
            s(yoffset),
            newst_format.clone(),
            adjust.clone(),
            contout.clone(),
            secout.clone(),
            input_file.clone(),
            newstout.clone(),
        ],
    );

    if run_cmd_in_process(&newstcom, None, Some("stdout")).is_err() {
        exit_from_imod_error(progname);
    }

    if zslices != 0 {
        prnstr(
            &fmtstr(
                "Contrast black/white levels determined from file were {},{}",
                &[s(black), s(white)],
            ),
            "\n",
            false,
        );
    }
    prnstr(" ", "\n", false);
    prnstr("The newstack command was:", "\n", false);
    prnstr(&newstcom, "\n", false);

    // Flip or rotate if requested
    if !doflip.is_empty() {
        prnstr(&format!("Running clip {doflip}"), "\n", false);
        if run_cmd_in_process(
            &fmtstr(
                "clip {} {} {} \"{}\" \"{}\"",
                &[
                    doflip.clone(),
                    clip_format.clone(),
                    clip_mode.clone(),
                    newstout.clone(),
                    output_file.clone(),
                ],
            ),
            None,
            None,
        )
        .is_err()
        {
            exit_from_imod_error(progname);
        }
        if std::fs::remove_file(&newstout).is_err() {
            prnstr(
                &format!("WARNING: error trying to delete temporary file {newstout}"),
                "\n",
                false,
            );
        }
    }

    0
}
