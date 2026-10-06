//! Translation of `IMOD/pysrc/squeezevol`: runs matchvol (or binvol) to squeeze
//! or expand a volume.  Authors: Tor Mohling and David Mastronarde.
//!
//! A Python command script with no functions; its top level is
//! [`squeezevol`].  `matchvol`, `alterheader` and `binvol` are this crate's
//! own programs; `runcmd` runs them in process (`imodpy::run_cmd`).  Python
//! values are carried with their types: the sizes are ints, the factors and
//! the header values floats (doubles) formatted with `str()`.

use super::imodpy::{
    MrcInfo, add_imod_bin_ignore_sighup, exit_from_imod_error, fmtstr, get_mrc, print_pid,
    prnstr, py_int_of_float, py_round, py_round_ndigits, py_str_float, run_cmd,
};
use super::pip::{
    exit_error, pip_done, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer,
    pip_get_non_option_arg, pip_get_string, pip_print_help, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`squeezevol:1-155`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn squeezevol(arguments: &[OsString]) -> i32 {
    let progname = "squeezevol";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Initializations
    let mut factor: f64 = 1.6;
    let (mut ix, mut iy, mut iz): (i32, i32, i32);
    let mut linear = String::new();
    let mut pixelxyz = [1.0_f64, 1.0, 1.0];
    let mut tempdir = String::new();

    // Fallbacks from ../manpages/autodoc2man 3 1 squeezevol
    let options: Vec<String> = [
        "f:factor:F:",
        "e:expand:F:",
        "x:xFactor:F:",
        "y:yFactor:F:",
        "z:zFactor:F:",
        "ix:ixSize:I:",
        "iy:iySize:I:",
        "iz:izSize:I:",
        "t:tempdir:FN:",
        "l:linear:B:",
        ":PID:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    //
    // Process command-line using PIP
    let (_opts, nonopts) = pip_read_or_parse_options(&argv, &options, progname, 2, 1, 1);
    if nonopts != 2 {
        prnstr(&format!("{prefix}wrong number of arguments"), "\n", false);
        pip_print_help(progname, 0, 1, 1);
        return done(1);
    }

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    let input_file = pip_get_non_option_arg(0).unwrap_or_default();
    let output_file = pip_get_non_option_arg(1).unwrap_or_default();
    if !Path::new(&input_file).exists() {
        exit_error(&format!("input file {input_file} does not exist"));
    }

    if let Ok(t) = pip_get_string("tempdir", "") {
        if !t.is_empty() {
            tempdir = format!("TemporaryDirectory {t}");
        }
    }
    if pip_get_boolean("linear", 0).unwrap_or(0) == 1 {
        linear = "InterpolationOrder 1".to_owned();
    }
    factor = pip_get_float("factor", factor).unwrap_or(factor);
    let if_factor = 1 - pip_get_err_no();
    let expand = pip_get_float("expand", 0.).unwrap_or(0.);
    if expand != 0. && if_factor != 0 {
        exit_error("You can enter either -factor or -expand but not both");
    }
    if expand != 0. {
        factor = 1. / expand;
    }
    let xs = pip_get_float("xFactor", factor).unwrap_or(factor);
    let ys = pip_get_float("yFactor", factor).unwrap_or(factor);
    let zs = pip_get_float("zFactor", factor).unwrap_or(factor);
    ix = pip_get_integer("ixSize", 0).unwrap_or(0);
    let ix_entered = 1 - pip_get_err_no();
    iy = pip_get_integer("iySize", 0).unwrap_or(0);
    let iy_entered = 1 - pip_get_err_no();
    iz = pip_get_integer("izSize", 0).unwrap_or(0);
    let iz_entered = 1 - pip_get_err_no();
    let mut legacy = pip_get_boolean("LegacyInterpolation", 0).unwrap_or(0);
    pip_done();

    if ((xs - py_round(xs)).abs() >= 0.001 || (ys - py_round(ys)).abs() >= 0.001)
        && (xs - ys).abs() >= 0.001
        && legacy == 0
    {
        legacy = 1;
        prnstr(
            "Using matchvol because reductions in X and Y are unequal and non-integer",
            "\n",
            false,
        );
    }

    if legacy == 0 && (xs < 1. || ys < 1. || zs < 1.) {
        legacy = 1;
        prnstr(
            "Using matchvol because the volume is being expanded",
            "\n",
            false,
        );
    }

    if legacy == 0 && ix_entered + iy_entered + iz_entered > 0 {
        legacy = 1;
        prnstr(
            "Using matchvol because an output size was entered",
            "\n",
            false,
        );
    }
    //
    let result = (|| -> Result<(), ()> {
        if legacy != 0 {
            let MrcInfo::All(tx, ty, tz, _mode, px, py, pz, mut origx, mut origy, mut origz, ..) =
                get_mrc(&input_file, true, false).map_err(|_| ())?
            else {
                return Err(());
            };
            pixelxyz = [px, py, pz];
            //
            if ix_entered == 0 {
                ix = tx;
            }
            if iy_entered == 0 {
                iy = ty;
            }
            if iz_entered == 0 {
                iz = tz;
            }
            let ox = py_int_of_float(ix as f64 / xs);
            let oy = py_int_of_float(iy as f64 / ys);
            let oz = py_int_of_float(iz as f64 / zs);
            let squeezex = 1. / xs;
            let squeezey = 1. / ys;
            let squeezez = 1. / zs;
            let pixelx = py_round_ndigits(xs * pixelxyz[0], 3);
            let pixely = py_round_ndigits(ys * pixelxyz[1], 3);
            let pixelz = py_round_ndigits(zs * pixelxyz[2], 3);
            origx += 0.5 * (ox as f64 * xs - tx as f64) * pixelxyz[0];
            origy += 0.5 * (oy as f64 * ys - ty as f64) * pixelxyz[1];
            origz += 0.5 * (oz as f64 * zs - tz as f64) * pixelxyz[2];
            //
            prnstr("Squeezing the volume with Matchvol...", "\n", false);
            let matchin = vec![
                format!("InputFile {input_file}"),
                format!("OutputFile {output_file}"),
                tempdir.clone(),
                linear.clone(),
                fmtstr(
                    "OutputSizeXYZ {} {} {}",
                    &[ox.to_string(), oy.to_string(), oz.to_string()],
                ),
                fmtstr(
                    "3DTransform {} 0 0 0 0 {} 0 0 0 0 {} 0",
                    &[
                        py_str_float(squeezex),
                        py_str_float(squeezey),
                        py_str_float(squeezez),
                    ],
                ),
            ];
            run_cmd(
                "matchvol -StandardInput",
                Some(&matchin),
                Some("stdout"),
                None,
                &[],
            )
            .map_err(|_| ())?;
            //
            prnstr(
                &fmtstr(
                    "Adjusting pixel spacing in header to {} {} {}",
                    &[
                        py_str_float(pixelx),
                        py_str_float(pixely),
                        py_str_float(pixelz),
                    ],
                ),
                "\n",
                false,
            );
            let alterin = vec![
                output_file.clone(),
                "del".to_owned(),
                fmtstr(
                    "{} {} {}",
                    &[
                        py_str_float(pixelx),
                        py_str_float(pixely),
                        py_str_float(pixelz),
                    ],
                ),
                "org".to_owned(),
                fmtstr(
                    "{} {} {}",
                    &[
                        py_str_float(origx),
                        py_str_float(origy),
                        py_str_float(origz),
                    ],
                ),
                "done".to_owned(),
            ];
            run_cmd("alterheader", Some(&alterin), None, None, &[]).map_err(|_| ())?;
        } else {
            prnstr("Squeezing the volume with Binvol...", "\n", false);
            run_cmd(
                &fmtstr(
                    "binvol -xbin {} -ybin {} -zbin {} -anti -1 \"{}\" \"{}\"",
                    &[
                        py_str_float(xs),
                        py_str_float(ys),
                        py_str_float(zs),
                        input_file.clone(),
                        output_file.clone(),
                    ],
                ),
                None,
                Some("stdout"),
                None,
                &[],
            )
            .map_err(|_| ())?;
        }
        Ok(())
    })();
    if result.is_err() {
        exit_from_imod_error(progname);
    }

    done(0)
}
