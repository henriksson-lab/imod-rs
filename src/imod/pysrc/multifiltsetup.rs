//! Translation of `IMOD/pysrc/multifiltsetup`: sets up command files to make
//! several tomograms with different filtering (SIRT-like, exact object size,
//! Hamming-like or Gaussian radial filters).
//!
//! A Python command script with no functions; its top level is
//! [`multifiltsetup`].  `getmrcsize` runs our own `header` in process.
//! Python values carry their types: the filter values are floats formatted
//! with `str()` ([`py_str_float`]) or `'{:.3f}'`.

use super::imodpy::{
    FLOAT_VALUE, INT_VALUE, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup,
    clean_chunk_files, complete_and_check_com_file, exit_from_imod_error,
    find_root_axis_and_extensions, fmtstr, get_mrc_size, option_value, os_path_splitext,
    parse_list, prnstr, py_int_floordiv, py_round_ndigits, py_str_float, read_text_file,
    standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_get_err_no, pip_get_float, pip_get_float_array, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`multifiltsetup:1-270`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn multifiltsetup(arguments: &[OsString]) -> i32 {
    let progname = "multifiltsetup";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 multifiltsetup
    let options: Vec<String> = [
        "com:CommandFile:FN:",
        "fake:FakeSIRTiterations:LI:",
        "exact:ExactObjectSizes:LI:",
        "hamming:HammingLikeStarts:FA:",
        "cutoffs:GaussianCutoffs:FA:",
        "falloffs:GaussianFalloffs:FA:",
        "width:WidthInX:I:",
        "ysize:SizeInY:I:",
        "thick:ThicknessInZ:I:",
        "xshift:ShiftInX:F:",
        "yshift:ShiftInY:I:",
        "zshift:ShiftInDepth:F:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opt_args, _num_non_opt_args) =
        pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    // Get the com file name, derive a com root name and full com file name, check exists
    let comfile = pip_get_in_out_file("CommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (comfile, mut comroot) = complete_and_check_com_file(&comfile);

    // Get options
    let width = pip_get_integer("WidthInX", 0).unwrap_or(0);
    let ysize = pip_get_integer("SizeInY", 0).unwrap_or(0) as i64;
    let thickness = pip_get_integer("ThicknessInZ", 0).unwrap_or(0);
    let xshift = pip_get_float("ShiftInX", 0.).unwrap_or(0.);
    let yshift = pip_get_integer("ShiftInY", 0).unwrap_or(0) as i64;
    let mut zshift = pip_get_float("ShiftInDepth", 0.).unwrap_or(0.);
    let z_entered = 1 - pip_get_err_no();
    if ysize < 0 || width < 0 || thickness < 0 {
        exit_error("Entries for -width, -thick, and -ysize cannot be negative");
    }

    // Count up filtering options, replace None with empty arrays for Gaussian
    let iter_string = pip_get_string("FakeSIRTiterations", "").unwrap_or_default();
    let mut num_filt_opt = 1 - pip_get_err_no();
    let object_string = pip_get_string("ExactObjectSizes", "").unwrap_or_default();
    num_filt_opt += 1 - pip_get_err_no();
    let hammings = pip_get_float_array("HammingLikeStarts", 0).unwrap_or_default();
    num_filt_opt += 1 - pip_get_err_no();
    let mut cutoffs = pip_get_float_array("GaussianCutoffs", 0).unwrap_or_default();
    let mut falloffs = pip_get_float_array("GaussianFalloffs", 0).unwrap_or_default();

    if !falloffs.is_empty() || !cutoffs.is_empty() {
        num_filt_opt += 1;
    }

    if num_filt_opt == 0 {
        exit_error("One of the filtering options must be entered");
    }
    if num_filt_opt > 1 {
        exit_error("Only one of the filtering options can be entered");
    }

    // `min`/`max` of a Python list of floats
    let list_min = |values: &[f64]| values.iter().copied().fold(f64::INFINITY, f64::min);
    let list_max = |values: &[f64]| values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Process lists of iterations or object sizes
    let mut iter_list: Vec<i32> = Vec::new();
    let mut max_num = 0;
    let mut rec_prefix = "";
    let num_filts: usize;
    let mut num_falls = 0usize;
    if !iter_string.is_empty() || !object_string.is_empty() {
        let desc;
        if !iter_string.is_empty() {
            iter_list = parse_list(&iter_string).unwrap_or_default();
            desc = "iteration number ";
            rec_prefix = "slfi";
        } else {
            iter_list = parse_list(&object_string).unwrap_or_default();
            desc = "object size ";
            rec_prefix = "efos";
        }
        if iter_list.is_empty() {
            exit_error(&format!("Illegal entry for {desc}list"));
        }
        for &num in &iter_list {
            if num <= 0 || num > 9999 {
                exit_error(&format!("{desc}{num} is out of allowed range"));
            }
        }
        max_num = *iter_list.iter().max().unwrap();
        num_filts = iter_list.len();

    // Check hamming entries
    } else if !hammings.is_empty() {
        if list_min(&hammings) < 0.0 || list_max(&hammings) > 0.5 {
            exit_error("Hamming filter start frequencies must be between 0 and 0.5");
        }
        num_filts = hammings.len();

    // Check Gaussian entries and figure out total number of filters
    } else {
        if !cutoffs.is_empty() && (list_min(&cutoffs) < 0. || list_max(&cutoffs) > 0.5) {
            exit_error("Gaussian filter cutoff frequencies must be between 0 and 0.5");
        }
        if !falloffs.is_empty() && (list_min(&falloffs) < 0. || list_max(&falloffs) > 0.5) {
            exit_error("Gaussian filter falloff frequencies must be between 0 and 0.5");
        }

        let num_cuts = cutoffs.len();
        num_falls = falloffs.len();
        num_filts = num_cuts.max(num_falls);
        if num_cuts > 1 && num_falls > 1 {
            if num_cuts < num_falls {
                exit_error(
                    "You must enter either one cutoff or the same number of cutoffs as falloffs",
                );
            } else if num_cuts > num_falls {
                exit_error(
                    "You must enter either one falloff or the same number of falloffs as cutoffs",
                );
            }
        }
    }

    // read com file and get options from it
    let comlines =
        read_text_file(&comfile, Some("tilt command file"), false, None).unwrap_or_default();
    let alifile = match option_value(&comlines, "inputproj", STRING_VALUE, true, 0, None, None) {
        Some(OptionValue::String(value)) => value,
        _ => String::new(),
    };
    let mut binning: i64 = 1;
    if let Some(OptionValue::Integers(values)) =
        option_value(&comlines, "imagebinned", INT_VALUE, true, 0, None, None)
    {
        if !values.is_empty() {
            binning = values[0] as i64;
        }
    }

    // If no Z shift was entered, get the existing Z shift from the SHIFT entry if any
    if z_entered == 0 {
        if let Some(OptionValue::Floats(values)) =
            option_value(&comlines, "shift", FLOAT_VALUE, true, 0, None, None)
        {
            if !values.is_empty() {
                zshift = values[1];
            }
        }
    }

    // Get the RADIAL entry if only one Gaussian option entered
    let mut radial_arr: Vec<f64> = Vec::new();
    if (!cutoffs.is_empty() && falloffs.is_empty()) || (!falloffs.is_empty() && cutoffs.is_empty())
    {
        radial_arr = match option_value(&comlines, "radial", FLOAT_VALUE, true, 0, None, None) {
            Some(OptionValue::Floats(values)) if !values.is_empty() => values,
            _ => exit_error(
                "Cannot find RADIAL entry, needed for default value of falloff or cutoff",
            ),
        };
        if radial_arr.len() < 2 {
            exit_error("The RADIAL entry in the command file does not have a falloff value");
        }
        if cutoffs.is_empty() {
            cutoffs.push(radial_arr[0]);
        }
    }

    // set up complete arrays of falloffs and cutoffs
    // Cutoffs take the same value fif 0 or 1 entered
    // Falloffs have a constant ratio if 0 or 1 entered
    if !cutoffs.is_empty() || !falloffs.is_empty() {
        let fall_ratio = if falloffs.is_empty() {
            radial_arr[1] / radial_arr[0].max(0.01)
        } else {
            falloffs[0] / cutoffs[0].max(0.01)
        };
        for _ in cutoffs.len()..num_filts {
            let first = cutoffs[0];
            cutoffs.push(first);
        }
        for ind in num_falls..num_filts {
            falloffs.push(py_round_ndigits(fall_ratio * cutoffs[ind], 3));
        }
    }

    let (_alix, aliy, _aliz) = match get_mrc_size(&alifile) {
        Ok(size) => size,
        Err(_) => exit_from_imod_error(progname),
    };

    let (com_ext, dual_num, mut rootname, type_ext, _stack_ext) =
        find_root_axis_and_extensions(0, Some(&comfile));
    let mut type_ext = type_ext.unwrap_or_default();
    if type_ext.is_empty() || !standard_type_extensions().contains(&type_ext) {
        type_ext = "mrc".to_owned();
    }

    let mut axis_let = "";
    if dual_num > 0 {
        if comroot.ends_with('a') {
            axis_let = "a";
            comroot.pop();
        } else if comroot.ends_with('b') {
            axis_let = "b";
            comroot.pop();
        }
    }

    if dual_num < 0 {
        rootname = os_path_splitext(&alifile).0;
    }

    comroot += &format!("_mulfil{axis_let}");

    // Clean up existing coms and logs if any
    clean_chunk_files(&comroot, false);

    // Shifts are consistent
    let shift_entry = fmtstr("{} {}", &[py_str_float(xshift), py_str_float(zshift)]);

    // But slices need to decrease to shift the volume up in rotated view
    let mut slice_entry = String::new();
    if ysize > 0 || yshift != 0 {
        let ubsize = aliy as i64 * binning;
        let slice_start = py_int_floordiv(ubsize - ysize, 2) - yshift;
        let slice_end = slice_start + ysize - 1;
        if slice_start < 0 || slice_end >= ubsize {
            exit_error("The combination of Y size and Y shift is outside the range of slices");
        }
        slice_entry = fmtstr("{} {}", &[slice_start.to_string(), slice_end.to_string()]);
    }

    // Loop on the output files
    for ind in 0..num_filts {
        // Set name of com file and tomogram file
        let comname = format!("{comroot}-{:03}.{com_ext}", ind + 1);
        let outfile = if !iter_list.is_empty() {
            let mut ndec = 2;
            if max_num > 99 {
                ndec = 3;
            }
            if max_num > 999 {
                ndec = 4;
            }
            format!(
                "{rootname}{axis_let}_{rec_prefix}{:0ndec$}.{type_ext}",
                iter_list[ind]
            )
        } else if !hammings.is_empty() {
            format!("{rootname}{axis_let}_hlfs{:.3}.{type_ext}", hammings[ind])
        } else {
            format!(
                "{rootname}{axis_let}_gfc{:.3}-f{:.3}.{type_ext}",
                cutoffs[ind], falloffs[ind]
            )
        };

        let mut sedcom = vec![sed_modify("OutputFile", &outfile, '/')];
        sedcom.extend(sed_del_and_add("SHIFT", &shift_entry, "THICKNESS", '/'));

        // Add the main filtering option
        if !iter_string.is_empty() {
            sedcom.push("/ExactFilterSize/d".to_owned());
            sedcom.extend(sed_del_and_add(
                "FakeSIRTiterations",
                &iter_list[ind].to_string(),
                "THICKNESS",
                '/',
            ));
        } else if !object_string.is_empty() {
            sedcom.push("/FakeSIRTiterations/d".to_owned());
            sedcom.extend(sed_del_and_add(
                "ExactFilterSize",
                &iter_list[ind].to_string(),
                "THICKNESS",
                '/',
            ));
        } else if !hammings.is_empty() {
            sedcom.push("/RADIAL/d".to_owned());
            sedcom.extend(sed_del_and_add(
                "HammingLikeFilter",
                &py_str_float(hammings[ind]),
                "THICKNESS",
                '/',
            ));
        } else {
            sedcom.push("/HammingLikeFilter/d".to_owned());
            sedcom.extend(sed_del_and_add(
                "RADIAL",
                &fmtstr(
                    "{} {}",
                    &[py_str_float(cutoffs[ind]), py_str_float(falloffs[ind])],
                ),
                "THICKNESS",
                '/',
            ));
        }

        // Add the volume selection options
        if width > 0 {
            sedcom.extend(sed_del_and_add(
                "WIDTH",
                &width.to_string(),
                "THICKNESS",
                '/',
            ));
        } else {
            sedcom.push("/WIDTH/d".to_owned());
        }

        if thickness > 0 {
            sedcom.extend(sed_del_and_add(
                "THICKNESS",
                &thickness.to_string(),
                "OutputFile",
                '/',
            ));
        }

        if !slice_entry.is_empty() {
            sedcom.extend(sed_del_and_add("SLICE", &slice_entry, "THICKNESS", '/'));
        } else {
            sedcom.push("/SLICE/d".to_owned());
        }

        if let Err(message) = pysed(
            &sedcom,
            PysedSrc::Lines(&comlines),
            Some(&comname),
            false,
            '/',
            false,
        ) {
            exit_error(&message);
        }
    }

    // Add cleanup file
    let _ = write_text_file(
        &format!("{comroot}-finish.com"),
        &[format!(
            "$b3dremove -g {comroot}-[0-9][0-9][0-9]*.com* {comroot}-[0-9][0-9][0-9]*.log* {comroot}-finish.com "
        )],
        false,
    );

    prnstr(
        &fmtstr(
            "{} command files output and ready to run with",
            &[(num_filts + 1).to_string()],
        ),
        "\n",
        false,
    );
    prnstr(
        &format!("   processchunks machine_list {comroot}"),
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    0
}
