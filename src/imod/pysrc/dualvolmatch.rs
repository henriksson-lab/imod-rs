//! Translation of `IMOD/pysrc/dualvolmatch`.
//!
//! A Python command script: its two functions are [`cleanup`] and
//! [`clean_exit_error`], and its top level is [`dualvolmatch`], translated
//! statement by statement.  The module globals they read (`testMode`,
//! `cleanList`) are passed explicitly.
//!
//! The source's main `try` catches `ImodpyError`, `IndexError` and
//! `ValueError`; its body is a closure returning [`PyExc`] for those three,
//! and any other exception the source would leave uncaught (a `NameError`
//! on a value never assigned, a `ZeroDivisionError`) ends the process with
//! the traceback's last line on standard error and status 1.  Values are
//! Python floats (doubles); `'{}'.format` of one is its `repr`
//! ([`py_str_float`]); `round()` rounds half to even.

use super::batchruntomo::py_str_float;
use super::imodpy::{
    add_imod_bin_ignore_sighup, call_own_program, cleanup_files, dataset_filename,
    exit_from_imod_error, find_root_axis_and_extensions, get_mrc_size, get_naming_style,
    make_backup_file, prnstr, read_text_file, run_cmd, set_root_and_extension,
    standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_two_floats, pip_read_or_parse_options,
};
use crate::imod::flib::distort::xf2rotmagstr::{xf2rotmagstr_compute, xf2rotmagstr_line};
use crate::imod::flib::model::refinematch::{
    RefinematchResult, refinematch_recording, refinematch_residual_line,
};
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::parse_input_params::set_exit_prefix;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// Rust-only: the exceptions the source's main `try` catches.
#[derive(Debug)]
pub enum PyExc {
    /// `ImodpyError`
    Imodpy,
    /// `IndexError`
    Index,
    /// `ValueError`
    Value,
}

/// Matches `cleanup` (`dualvolmatch:12`).
pub fn cleanup(test_mode: i32, clean_list: &[String]) {
    if test_mode < 2 {
        cleanup_files(clean_list);
    }
}

/// Matches `cleanExitError` (`dualvolmatch:17`).
pub fn clean_exit_error(test_mode: i32, clean_list: &[String], message: &str) -> ! {
    cleanup(test_mode, clean_list);
    exit_error(message)
}

/// The script's top level (`dualvolmatch:22-437`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn dualvolmatch(arguments: &[OsString]) -> i32 {
    let progname = "dualvolmatch";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    // An exception the source does not catch
    let uncaught = |message: &str| -> ! {
        let _ = std::io::stdout().flush();
        eprintln!("Traceback (most recent call last):");
        eprintln!("{message}");
        std::process::exit(1)
    };

    // Fallbacks from ../manpages/autodoc2man 3 1 dualvolmatch
    let options: Vec<String> = [
        "name:RootName:CH:",
        "atob:MatchAtoB:B:",
        "binning:BinningToApply:I:",
        "tilt:TiltAngleMaxAndStep:FP:",
        "refine:RefineTiltAngles:I:",
        "center:CenterShiftLimit:F:",
        "maxresid:MaximumResidual:F:",
        "scan:ScanRotationMaxAndStep:FP:",
        "final:FinalOutputFile:FN:",
        "style:NamingStyle:I:",
        "test:TestMode:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    let max_tilt_shift = 2;
    let mut leave_opt = "";
    let max_drop_frac: f64 = 0.17;
    #[allow(clippy::excessive_precision)]
    let dtor: f64 = 0.01745329252;

    let (_com_ext, _dual_num, _setroot, type_ext, _stack_ext) =
        find_root_axis_and_extensions(-1, None);
    let (_name_style, type_ext) = match get_naming_style(type_ext.as_deref(), false, false) {
        Ok(result) => result,
        Err(message) => exit_error(&message),
    };
    let type_ext = type_ext.unwrap_or_default();

    // Get options
    let root_name = pip_get_in_out_file("RootName", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if root_name.is_empty() {
        exit_error("The root name of the dataset must be entered");
    }

    let if_ato_b = pip_get_boolean("MatchAtoB", 0).unwrap_or(0);
    let mut asrc = "a";
    let mut bsrc = "b";
    if if_ato_b != 0 {
        asrc = "b";
        bsrc = "a";
    }

    let root_a = format!("{root_name}{asrc}");
    let root_b = format!("{root_name}{bsrc}");
    set_root_and_extension(&root_b, &type_ext);
    let rec_name_a = dataset_filename(".rec", Some(&root_a), None);
    let rec_name_b = dataset_filename(".rec", None, None);
    for rec in [&rec_name_a, &rec_name_b] {
        if !Path::new(rec).exists() {
            exit_error(&format!("Tomogram does not exist: {rec}"));
        }
    }

    // Outside the `try`: a failure is an uncaught ImodpyError.
    // Fixed in translation (BUGS.md): `dualvolmatch:89` reads `recNameA` a
    // second time, so native's B size is A's (binning and both thickness
    // advisories ignore B); this reads the B tomogram.
    let (nxa, nya, nza) = match get_mrc_size(&rec_name_a) {
        Ok((x, y, z)) => (x as i64, y as i64, z as i64),
        Err(error) => uncaught(&format!("imodpy.ImodpyError: {error}")),
    };
    let (nxb, nyb, nzb) = match get_mrc_size(&rec_name_b) {
        Ok((x, y, z)) => (x as i64, y as i64, z as i64),
        Err(error) => uncaught(&format!("imodpy.ImodpyError: {error}")),
    };

    let mut binning = pip_get_integer("BinningToApply", -1).unwrap_or(-1) as i64;
    if binning <= 0 {
        let min_size = nxa.min(nza).min(nxb).min(nzb);
        binning = 1i64.max(4i64.min(min_size.div_euclid(512)));
    }

    let (scan_max, scan_interval) =
        pip_get_two_floats("ScanRotationMaxAndStep", (20., 4.)).unwrap_or((20., 4.));
    let (tilt_max, mut tilt_interval) =
        pip_get_two_floats("TiltAngleMaxAndStep", (4., 2.)).unwrap_or((4., 2.));
    if tilt_max < 0. || tilt_interval <= 0. {
        exit_error("Both maximum tilt angle and step size should be positive");
    }
    let mut num_steps = (2. * tilt_max / tilt_interval).round_ties_even() as i64 + 1;
    if num_steps < 3 {
        exit_error("The tilt interval must not be bigger than the maximum tilt angle");
    }

    let num_refine = pip_get_integer("RefineTiltAngles", 1).unwrap_or(1) as i64;
    if num_refine < 0 || num_refine > 5 {
        exit_error("Number of tilt angle refinements should be between 0 and 5");
    }

    let ub_max_resid = pip_get_float("MaximumResidual", 10.).unwrap_or(10.);
    let max_cen_shift = pip_get_float("CenterShiftLimit", 10.).unwrap_or(10.);
    let solve_file = pip_get_string("FinalOutputFile", "solve.xf").unwrap_or_default();
    let test_mode = pip_get_integer("TestMode", 0).unwrap_or(0);
    if test_mode > 2 {
        leave_opt = "-t";
    }

    // Make up all needed names and put on the cleanup list
    let rec_names = [rec_name_a.clone(), rec_name_b.clone()];
    let mut bin_recs = [
        dataset_filename("_bin.rec", Some(&root_a), None),
        dataset_filename("_bin.rec", None, None),
    ];
    if binning == 1 {
        bin_recs = rec_names.clone();
    }
    let bin_proj = [
        dataset_filename("_bin.proj", Some(&root_a), None),
        dataset_filename("_bin.proj", None, None),
    ];
    let best_xf_file = format!("{root_name}_dvmatch.xf");
    let init3d_file = format!("{root_name}_dvm3d.xf");
    let refine_file = format!("{root_name}_refine.xf");
    let patch_file = format!("{root_name}_dvmpatch.out");
    let one_patch_file = format!("{root_name}_dvmcenpat.out");
    let match_rec = dataset_filename("_bin.mat", None, None);
    let mut clean_list = vec![
        bin_proj[0].clone(),
        bin_proj[1].clone(),
        best_xf_file.clone(),
        init3d_file.clone(),
        refine_file.clone(),
        patch_file.clone(),
        one_patch_file.clone(),
        match_rec.clone(),
    ];
    if binning > 1 {
        clean_list.extend([bin_recs[0].clone(), bin_recs[1].clone()]);
    }
    let ind = clean_list.len();
    for i in 0..ind {
        let name = format!("{}~", clean_list[i]);
        clean_list.push(name);
    }

    // Set environment variable to produce files of the right type
    if !type_ext.is_empty() && standard_type_extensions().contains(&type_ext) {
        unsafe {
            std::env::set_var("MRC_OUTPUT_FORMAT", type_ext.to_uppercase());
        }
    }

    let result: Result<(), PyExc> = (|| {
        let run = |command: &str, input: Option<&[String]>| -> Result<Vec<String>, PyExc> {
            run_cmd(command, input, None, None, &[])
                .map(Option::unwrap_or_default)
                .map_err(|_| PyExc::Imodpy)
        };
        let float =
            |text: &str| -> Result<f64, PyExc> { text.parse::<f64>().map_err(|_| PyExc::Value) };
        let int =
            |text: &str| -> Result<i64, PyExc> { text.parse::<i64>().map_err(|_| PyExc::Value) };
        let at = |fields: &[&str], index: isize| -> Result<String, PyExc> {
            let index = if index < 0 {
                fields.len() as isize + index
            } else {
                index
            };
            if index < 0 || index as usize >= fields.len() {
                return Err(PyExc::Index);
            }
            Ok(fields[index as usize].to_owned())
        };

        // Start out with binning the volumes if needed
        if binning > 1 {
            for ind in 0..2 {
                prnstr(
                    &format!("Making binned by {binning} volume {}", bin_recs[ind]),
                    "\n",
                    true,
                );
                run(
                    &format!(
                        "binvol -bin {binning} \"{}\" \"{}\"",
                        rec_names[ind], bin_recs[ind]
                    ),
                    None,
                )?;
            }
        }

        // Set up for loop on projection and matching
        let mut ind_refine = 0;
        let mut ind_shift_ref = 0;
        let max_ref_shift = 2;
        let mut num_tilt_shift = 0;
        tilt_interval = 2. * tilt_max / (num_steps - 1) as f64;
        let mut tilt_start = [-tilt_max, -tilt_max];
        let mut best_tilt_a: Option<f64> = None;
        let mut best_tilt_b: Option<f64> = None;
        let mut interp_tilt_a: Option<f64> = None;
        let mut interp_tilt_b: Option<f64> = None;
        let mut num_mirror;
        let mut num_regular;
        loop {
            // Reproject at current tilt angles
            for ind in 0..2 {
                let start = tilt_start[ind];
                let end = tilt_start[ind] + (num_steps - 1) as f64 * tilt_interval;
                prnstr(
                    &format!(
                        "Making reprojection {}: {start:.1} to {end:.1} at {tilt_interval:.1} degrees",
                        bin_proj[ind]
                    ),
                    "\n",
                    false,
                );
                run(
                    &format!(
                        "xyzproj -axis Z -angles {},{},{} -mode 2 \"{}\" \"{}\"",
                        py_str_float(start),
                        py_str_float(end),
                        py_str_float(tilt_interval),
                        bin_recs[ind],
                        bin_proj[ind]
                    ),
                    None,
                )?;
            }

            // Find best match
            prnstr(
                "Running matchrotpairs to find best matching tilts",
                "\n",
                false,
            );
            let match_lines = run(
                &format!(
                    "matchrotpairs -near -scan {},{} -swap {leave_opt} \"{}\" \"{}\" \"{best_xf_file}\"",
                    py_str_float(scan_max),
                    py_str_float(scan_interval),
                    bin_proj[0],
                    bin_proj[1]
                ),
                None,
            )?;

            // Look for best pair, interpolated values, and if it is mirrored
            let mut end_of_range = true;
            let mut best_a: i64 = -1;
            let mut best_b: Option<i64> = None;
            num_mirror = 0;
            num_regular = 0;
            for line in &match_lines {
                if test_mode != 0 {
                    prnstr(line.trim_end(), "\n", false);
                }
                if line.contains("mirror") {
                    num_mirror += 1;
                } else if line.contains("regular") {
                    num_regular += 1;
                }
                if line.contains("Views in best pair") {
                    let lsplit = line.split_whitespace().collect::<Vec<_>>();
                    best_a = int(&at(&lsplit, -1)?)?;
                    let b = int(&at(&lsplit, -3)?)?;
                    best_b = Some(b);
                    let ta = tilt_start[0] + (best_a - 1) as f64 * tilt_interval;
                    let tb = tilt_start[1] + (b - 1) as f64 * tilt_interval;
                    best_tilt_a = Some(ta);
                    best_tilt_b = Some(tb);
                    prnstr(
                        &format!(
                            "Tilt angles of best pair: {} {ta:.1}  {} {tb:.1}",
                            asrc.to_uppercase(),
                            bsrc.to_uppercase()
                        ),
                        "\n",
                        false,
                    );
                } else if line.contains("Interpolated view") {
                    let lsplit = line.split_whitespace().collect::<Vec<_>>();
                    end_of_range = false;
                    let interp_a = float(&at(&lsplit, -1)?)?;
                    let interp_b = float(&at(&lsplit, -2)?)?;
                    let ta = tilt_start[0] + (interp_a - 1.) * tilt_interval;
                    let tb = tilt_start[1] + (interp_b - 1.) * tilt_interval;
                    interp_tilt_a = Some(ta);
                    interp_tilt_b = Some(tb);
                    prnstr(
                        &format!(
                            "Interpolated tilt angles: {} {ta:.1}  {} {tb:.1}",
                            asrc.to_uppercase(),
                            bsrc.to_uppercase()
                        ),
                        "\n",
                        false,
                    );
                } else if line.contains("Temporary files") {
                    prnstr(&format!("Matchrotpairs {}", line.trim()), "\n", false);
                }
            }

            if best_a <= 0 {
                clean_exit_error(
                    test_mode,
                    &clean_list,
                    "Cannot find best view pair in output of Matchrotpairs",
                );
            }
            let best_b = best_b.unwrap_or(0);

            if end_of_range {
                // If end of range flag set, make sure it not just failure to get interpolated
                // values, that it is not a refinement search, and that a shift is still OK
                if best_a > 1 && best_a < num_steps && best_b > 1 && best_b < num_steps {
                    clean_exit_error(
                        test_mode,
                        &clean_list,
                        "Cannot find interpolated view numbers in output of Matchrotpairs",
                    );
                }
                if ind_refine != 0 {
                    if ind_shift_ref >= max_ref_shift {
                        clean_exit_error(
                            test_mode,
                            &clean_list,
                            &format!(
                                "Search is at end of range in refinement step after shifting refinement {max_ref_shift} times; try setting TiltAngleMaxAndStep to more, smaller steps (e.g. 5,1)"
                            ),
                        );
                    }
                    ind_shift_ref += 1;
                    prnstr("Shifting search to center on latest best pair", "\n", false);
                }

                if num_tilt_shift >= max_tilt_shift {
                    clean_exit_error(
                        test_mode,
                        &clean_list,
                        "The search for best pairs has already been shifted the maximum number of times",
                    );
                }

                num_tilt_shift += 1;
            } else {
                // If not at end of range, stop if refinement steps are over, otherwise
                // make sure there are at least 5 steps now and cut the interval
                if ind_refine >= num_refine {
                    break;
                }

                num_steps = 7i64.max(num_steps);
                ind_refine += 1;
                tilt_interval /= 2.;
            }

            // Set up starting point for next round
            let (Some(ta), Some(tb)) = (best_tilt_a, best_tilt_b) else {
                uncaught("NameError: name 'bestTiltA' is not defined");
            };
            tilt_start[0] = ta - num_steps.div_euclid(2) as f64 * tilt_interval;
            tilt_start[1] = tb - num_steps.div_euclid(2) as f64 * tilt_interval;
        }

        // Search is over, get the transformation
        let xflines = read_text_file(
            &best_xf_file,
            Some("file with transformation between best tilts"),
            false,
            None,
        )
        .unwrap_or_default();
        if xflines.len() != 2 {
            clean_exit_error(
                test_mode,
                &clean_list,
                "The file with transformation between best tilts does not have 2 lines",
            );
        }
        let lsplit = xflines[1].split_whitespace().collect::<Vec<_>>();
        if lsplit.len() < 6 {
            clean_exit_error(
                test_mode,
                &clean_list,
                "Not enough values in transformation between best tilts",
            );
        }
        let l11 = float(lsplit[0])?;
        let l12 = float(lsplit[1])?;
        let l21 = float(lsplit[2])?;
        let l22 = float(lsplit[3])?;
        let ldx = float(lsplit[4])?;
        let ldy = float(lsplit[5])?;

        // Get the mean mag
        // Direct call (owner rule, 2026-09-26): xf2rotmagstr's transforms in
        // place of parsing its last output line (`dualvolmatch:295-302`).  The
        // script required that line to be transform 2's, printed it from its
        // fifth character, and took `float()` of its last word, the
        // `Mean mag={f7.4}` field: so the value is rounded as `f7.4` prints
        // it, and a field printed without a leading blank ran into `mag=` and
        // raised ValueError.
        let mag_command = format!("xf2rotmagstr {best_xf_file}");
        let mag_file = best_xf_file.clone();
        let mag_result =
            match call_own_program(&mag_command, &["xf2rotmagstr"], None, true, move || {
                set_exit_prefix("ERROR: XF2ROTMAGSTR -");
                xf2rotmagstr_compute(&mag_file)
            }) {
                Ok((Some(result), _)) => result,
                _ => return Err(PyExc::Imodpy),
            };
        // `magLines[-1]`: the last transform's line, or the warping-file note
        // when there are none.
        if mag_result.transforms.is_empty() && !mag_result.from_warp_file {
            return Err(PyExc::Index);
        }
        if mag_result.transforms.len() != 2 {
            clean_exit_error(
                test_mode,
                &clean_list,
                "Cannot find mean mag in output from xf2rotmagstr",
            );
        }
        let last = xf2rotmagstr_line(2, &mag_result.transforms[1]);
        prnstr("Transformation between best reprojections:", "\n", false);
        prnstr(last.chars().skip(4).collect::<String>().trim(), "\n", false);
        let mag_field = format_f(f64::from(mag_result.transforms[1].smag_mean), 7, 4);
        if !mag_field.starts_with(' ') {
            return Err(PyExc::Value);
        }
        let mut mag = float(mag_field.trim())?;
        if num_mirror > num_regular {
            mag = -mag;
            if l12 * l21 < 0. {
                prnstr(
                    &format!(
                        "WARNING: {progname} - Matchrotpairs indicated mirroring, but the transformation is not consistent with that"
                    ),
                    "\n",
                    false,
                );
            }
        }

        // Compute the transform
        let (Some(interp_tilt_a), Some(interp_tilt_b)) = (interp_tilt_a, interp_tilt_b) else {
            uncaught("NameError: name 'interpTiltA' is not defined");
        };
        let cos_a = (dtor * interp_tilt_a).cos();
        let sin_a = (dtor * interp_tilt_a).sin();
        let cos_b = (dtor * interp_tilt_b).cos();
        let sin_b = (dtor * interp_tilt_b).sin();
        let a11 = l11 * cos_b * cos_a + mag * sin_a * sin_b;
        let a21 = -l11 * cos_b * sin_a + mag * cos_a * sin_b;
        let a31 = l21 * cos_b;
        let a12 = -l11 * sin_b * cos_a + mag * sin_a * cos_b;
        let a22 = l11 * sin_b * sin_a + mag * cos_a * cos_b;
        let a32 = -l21 * sin_b;
        let a13 = l12 * cos_a;
        let a23 = -l12 * sin_a;
        let a33 = l22;
        let dx = ldx * cos_a;
        let dy = -ldx * sin_a;
        let dz = ldy;

        // Write it to file with '{:10.6f} {:10.6f} {:10.6f} {:10.3f}'
        let lines = vec![
            format!("{a11:10.6} {a12:10.6} {a13:10.6} {dx:10.3}"),
            format!("{a21:10.6} {a22:10.6} {a23:10.6} {dy:10.3}"),
            format!("{a31:10.6} {a32:10.6} {a33:10.6} {dz:10.3}"),
        ];
        let _ = write_text_file(&init3d_file, &lines, false);

        // Make binned matching volume
        let (nx_bin_a, ny_bin_a, nz_bin_a) =
            get_mrc_size(&bin_recs[0]).map_err(|_| PyExc::Imodpy)?;
        let (_nx_bin_b, ny_bin_b, _nz_bin_b) =
            get_mrc_size(&bin_recs[1]).map_err(|_| PyExc::Imodpy)?;
        let (nx_bin_a, ny_bin_a, nz_bin_a) = (nx_bin_a as i64, ny_bin_a as i64, nz_bin_a as i64);
        prnstr(
            &format!("Making initial matching binned volume {match_rec}"),
            "\n",
            false,
        );
        run(
            &format!(
                "matchvol -size {nx_bin_a},{},{nz_bin_a} -xffile {init3d_file} \"{}\" \"{match_rec}\"",
                ny_bin_a.max(ny_bin_b as i64),
                bin_recs[1]
            ),
            None,
        )?;

        // Set up and run the correlation search
        let xmin = nx_bin_a.div_euclid(5);
        let xmax = nx_bin_a - xmin;
        let zmin = nz_bin_a.div_euclid(5);
        let zmax = nz_bin_a - zmin;
        let range_x = xmax - xmin;
        let range_z = zmax - zmin;
        let mut patch_size = range_x
            .div_euclid(3)
            .min(range_z.div_euclid(3))
            .min(512i64.div_euclid(binning));
        if patch_size == 0 {
            uncaught("ZeroDivisionError: float division by zero");
        }
        let num_patch_x = 3i64.max(
            ((3. * range_x as f64 + patch_size as f64) / (2. * patch_size as f64)).round_ties_even()
                as i64,
        );
        let num_patch_z = 3i64.max(
            ((3. * range_z as f64 + patch_size as f64) / (2. * patch_size as f64)).round_ties_even()
                as i64,
        );
        let border = 24 + 12 * nx_bin_a.min(nz_bin_a).div_euclid(1000);
        let cscom_base = vec![
            format!("ReferenceFile {}", bin_recs[0]),
            format!("FileToAlign {match_rec}"),
            format!("XMinAndMax {xmin},{xmax}"),
            format!("YMinAndMax {},{ny_bin_a}", 1),
            format!("ZMinAndMax {zmin},{zmax}"),
            format!("BSourceOrSizeXYZ {}", bin_recs[1]),
            format!("BSourceTransform {init3d_file}"),
            format!("BSourceBorderXLoHi {border},{border}"),
            format!("BSourceBorderYZLoHi {border},{border}"),
            "FlipYZMessages".to_owned(),
        ];
        let mut cscom = cscom_base.clone();
        cscom.extend([
            format!("PatchSizeXYZ {patch_size},{},{patch_size}", ny_bin_a - 2),
            format!("NumberOfPatchesXYZ {num_patch_x},1,{num_patch_z}"),
            format!("OutputFile {patch_file}"),
        ]);

        prnstr(
            "Running corrsearch3d on large patches in binned volumes",
            "\n",
            false,
        );
        run("corrsearch3d -StandardInput", Some(&cscom))?;

        // Run refinematch for the transform
        let rfcom = vec![
            format!("PatchFile {patch_file}"),
            format!("OutputFile {refine_file}"),
            format!("VolumeOrSizeXYZ {}", bin_recs[0]),
            format!("InitialTransformFile {init3d_file}"),
            format!("ProductTransformFile {solve_file}"),
            format!("ScaleShiftByFactor {binning}"),
            format!("MeanResidualLimit {nx_bin_a}"),
            format!("MaxFractionToDrop {}", py_str_float(max_drop_frac)),
        ];
        prnstr(
            "Running refinematch to get refined transformation with Z shift",
            "\n",
            false,
        );
        // Direct call (owner rule, 2026-09-26): refinematch records the values
        // of its reports into a `RefinematchResult`, used here in place of
        // parsing its "center shift" and "Mean residual" lines
        // (`dualvolmatch:361-376`).  It still reads its options through PIP
        // from `rfcom`.
        let sink = std::sync::Arc::new(std::sync::Mutex::new(RefinematchResult::default()));
        let recorder = std::sync::Arc::clone(&sink);
        let ref_lines = call_own_program(
            "refinematch -StandardInput",
            &["refinematch", "-StandardInput"],
            Some(&rfcom),
            true,
            move || refinematch_recording(recorder),
        )
        .map(|(_, lines)| lines)
        .map_err(|_| PyExc::Imodpy)?;
        let result = sink.lock().expect("refinematch result").clone();
        if test_mode != 0 {
            for line in &ref_lines {
                prnstr(line.trim_end(), "\n", false);
            }
        }

        // The script read both values back from the printed text: the last
        // word of `Implied center shift is{f8.1}` and the third word of
        // ` Mean residual{f8.3},  maximum{f8.3}` with commas blanked.  So each
        // is rounded as that format prints it, and a residual field printed
        // without a leading blank ran into `residual`, leaving `maximum` as
        // the third word, which `float()` rejects (ValueError).  The center
        // shift line is printed after the residual lines.
        let mut cen_shift: Option<f64> = None;
        let mut mean_resid: Option<f64> = None;
        for &(dev_mean, dev_max) in &result.residuals {
            if test_mode == 0 {
                prnstr(&refinematch_residual_line(dev_mean, dev_max), "\n", false);
            }
            let field = format_f(f64::from(dev_mean), 8, 3);
            if !field.starts_with(' ') {
                return Err(PyExc::Value);
            }
            mean_resid = Some(float(field.trim())? * binning as f64);
        }
        if let Some(value) = result.center_shift {
            let shift = float(format_f(f64::from(value), 8, 1).trim())?;
            cen_shift = Some(shift);
            //
            // BRT is using 'implies a center' as a tag
            prnstr(
                &format!(
                    "The residual for the center patch implies a center shift in Z of {shift:.1}"
                ),
                "\n",
                false,
            );
        }

        let Some(cen_shift) = cen_shift else {
            uncaught("NameError: name 'cenShift' is not defined");
        };
        let round_cen = cen_shift.round_ties_even() as i64;
        let round_abs_cen = cen_shift.abs().round_ties_even() as i64;
        let Some(mean_resid) = mean_resid else {
            uncaught("NameError: name 'meanResid' is not defined");
        };
        if mean_resid > ub_max_resid {
            // If the residual is too big, just get the center shift from a bigger patch and
            // add it to the initial transform and make that be the output file
            patch_size = ((3 * patch_size).div_euclid(2))
                .min(range_x - 2)
                .min(range_z - 2);
            let mut cscom = cscom_base.clone();
            cscom.extend([
                format!("PatchSizeXYZ {patch_size},{},{patch_size}", ny_bin_a - 2),
                "NumberOfPatchesXYZ 1,1,1".to_owned(),
                format!("OutputFile {one_patch_file}"),
            ]);

            // BRT is using 'unbinned mean residual' and 'Falling back' as tags
            prnstr(
                &format!("The unbinned mean residual is {mean_resid:.2} which is above the limit"),
                "\n",
                false,
            );
            prnstr(
                "Falling back to the initial estimate of the 3D transformation",
                "\n",
                false,
            );
            prnstr(
                "Running corrsearch3d on one bigger patch to get shifts",
                "\n",
                false,
            );
            run("corrsearch3d -StandardInput", Some(&cscom))?;
            let pat_lines = read_text_file(
                &one_patch_file,
                Some("file with one center patch displacement"),
                false,
                None,
            )
            .unwrap_or_default();
            if pat_lines.len() < 2 {
                clean_exit_error(test_mode, &clean_list, "Too few lines in patch output file");
            }
            let lsplit = pat_lines[1].split_whitespace().collect::<Vec<_>>();
            let binf = binning as f64;
            let lines = vec![
                format!(
                    "{a11:10.6} {a12:10.6} {a13:10.6} {:10.3}",
                    binf * dx + float(&at(&lsplit, 3)?)?
                ),
                format!(
                    "{a21:10.6} {a22:10.6} {a23:10.6} {:10.3}",
                    binf * dy + float(&at(&lsplit, 4)?)?
                ),
                format!(
                    "{a31:10.6} {a32:10.6} {a33:10.6} {:10.3}",
                    binf * dz + float(&at(&lsplit, 5)?)?
                ),
            ];
            make_backup_file(&solve_file);
            let _ = write_text_file(&solve_file, &lines, false);

            // If there is much center shift in earlier run, warn that thickness should be bigger
            // perhaps by twice this shift
            // BRT is looking for 'may need to set thickness'
            if round_abs_cen >= 5 && nyb + 2 * round_abs_cen > nya {
                prnstr("", "\n", false);
                prnstr(
                    &format!(
                        "WARNING: dualvolmatch - You may need to set thickness of initial matching file to at least {}",
                        nyb + 2 * round_abs_cen
                    ),
                    "\n",
                    false,
                );
                prnstr(
                    "     (In Etomo, Initial match size for Matchvol1)",
                    "\n",
                    false,
                );
                prnstr("", "\n", false);
            }
        } else if cen_shift.abs() > max_cen_shift {
            prnstr(
                &format!("The unbinned mean residual is {mean_resid:.2}"),
                "\n",
                false,
            );

            // When using the solved transform, if the center shift is above limit, behave like
            // solvematch, using similar text, and advise on thickness too
            // BRT is looking for 'InitialShiftXYZ' and 'needs'
            prnstr("", "\n", false);
            prnstr(
                "   The center shift is bigger than the specified limit",
                "\n",
                false,
            );
            prnstr(
                &format!("   The InitialShiftXYZ for corrsearch3d needs to be 0 {round_cen} 0"),
                "\n",
                false,
            );
            prnstr(
                &format!("   In Etomo, set Patchcorr Initial shifts in X, Y, Z to 0 0 {round_cen}"),
                "\n",
                false,
            );
            if nyb > nya {
                prnstr(
                    &format!(
                        "   You should also set thickness of initial matching file to at least {nyb}"
                    ),
                    "\n",
                    false,
                );
                prnstr(
                    "     (In Etomo, Initial match size for Matchvol1)",
                    "\n",
                    false,
                );
            }

            // BRT is looking for 'CenterShiftLimit' and 'avoid stopping'
            prnstr(
                &format!(
                    "   To avoid stopping with this error, set CenterShiftLimit to {}",
                    round_abs_cen + 2
                ),
                "\n",
                false,
            );

            // BRT is looking for 'Initial shift needs'
            clean_exit_error(
                test_mode,
                &clean_list,
                "Initial shift needs to be set for patch correlation",
            );
        }

        cleanup(test_mode, &clean_list);
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    })();

    match result {
        Ok(()) => 0,
        Err(PyExc::Imodpy) => {
            cleanup(test_mode, &clean_list);
            exit_from_imod_error(progname)
        }
        Err(PyExc::Index) => clean_exit_error(
            test_mode,
            &clean_list,
            "Fewer than expected number of values on line in program output",
        ),
        Err(PyExc::Value) => clean_exit_error(
            test_mode,
            &clean_list,
            "Extracting a numerical value from program output",
        ),
    }
}
