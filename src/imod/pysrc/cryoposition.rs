//! Translation of `IMOD/pysrc/cryoposition`: finds the positioning of a
//! cryo-tomogram (`findsection` on a volume with gold and other high
//! densities erased).
//!
//! The script's top level is [`cryoposition`]; its functions are
//! [`cleanup`] and [`get_threshold_from_clip_output`].  `newstack`, `tilt`,
//! `findbeads3d`, `boxstartend`, `clip`, `alterheader`, `imodauto`,
//! `ccderaser` and `findsection` are our own programs and run in process
//! through `imodpy::run_cmd`; the `findbeads3d` report and the `clip hist`
//! thresholds are still read from their printed text, as the script reads
//! them.

use super::imodpy::{
    BOOL_VALUE, FLOAT_VALUE, MrcInfo, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup,
    cleanup_files, dataset_filename, exit_from_imod_error, extract_program_entries,
    find_root_axis_and_extensions, get_err_strings, get_mrc, get_mrc_size, option_value, prnstr,
    py_float, py_int, py_round, py_str_float, read_text_file, run_cmd, set_output_format_if_needed,
    set_root_and_extension,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_integer_array, pip_get_string, pip_get_three_integers,
    pip_get_two_floats, pip_number_of_entries, pip_read_or_parse_options, python_uncaught,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// How the script's `try` block ended: `except ImodpyError`, or `except
/// (IndexError, ValueError)`.
enum Failure {
    Imodpy,
    Interpret,
}

/// `def cleanup()` (`cryoposition:12`).
fn cleanup(leave_temp: i32, clean_list: &[String]) {
    if leave_temp <= 0 {
        prnstr("Cleaning up temporary files", "\n", true);
        let mut list = clean_list.to_vec();
        for name in clean_list {
            list.push(format!("{name}~"));
        }
        cleanup_files(&list);
    }
}

/// `def getThresholdFromClipOutput(clipLines, key)` (`cryoposition:20`).
/// `Err` is the `ValueError` of a last field that is not a number.
fn get_threshold_from_clip_output(clip_lines: &[String], key: &str) -> Result<f64, Failure> {
    for line in clip_lines {
        if line.contains(key) {
            prnstr(line.trim(), "\n", false);
            let last = line.trim().split_whitespace().last().unwrap_or("");
            return py_float(last).ok_or(Failure::Interpret);
        }
    }
    exit_error("Could not find threshold value in clip output")
}

/// The script's top level (`cryoposition:29-559`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn cryoposition(arguments: &[OsString]) -> i32 {
    let progname = "cryoposition";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 cryoposition
    let options: Vec<String> = [
        "root:RootName:CH:",
        "thickness:ThicknessOfTomograms:I:",
        "find:FindBeadsInVolume:I:",
        "size:BeadSize:F:",
        "light:LightFeatures:B:",
        "binning:BinningToApply:I:",
        "erase:EraseFraction:F:",
        "high:HighSDCriterion:F:",
        "boost:BoostThickness:F:",
        "scales:ScalesToApply:IA:",
        "box:BoxSizeInXYZ:IT:",
        "spacing:SpacingOfBoxesInXYZ:IT:",
        "gpu:UseGPU:I:",
        "pitch:TomoPitchModel:FN:",
        "control:ControlValue:FPM:",
        "fsopt:FindSecOptions:CH:",
        "leave:LeaveTempFiles:I:",
        "use:UseTempFiles:I:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 2, 0, 0);
    // SAFETY: the script sets its own environment before running anything.
    unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };

    let bead_optimal = 5.0_f64;
    let bead_minimum = 4.2_f64;
    let bead_big_vol_min = 3.0_f64;
    let mut bead_maximum = 7.0_f64;
    let mut max_binning: i64 = 4;
    let vol_optimal = 650.0_f64;
    let mut x_oversize_frac = 1.18_f64;
    let mut y_oversize_frac = 1.06_f64;
    let mut find_avg_fallback = 0.5_f64;
    let mut find_store_fallback = 0.5_f64;
    let mut extra_hist_frac = 0.95_f64;
    let mut bead_vol_frac = 0.33_f64;
    let mut no_bead_diameter = 3.0_f64;
    let mut thresh_sum_factor = 0.05_f64;
    let mut max_bead_erase_frac = 0.01_f64;
    let mut min_bead_erase_frac = 0.0005_f64;
    // Good maxErase value for a reference volume
    let ref_erase_limit = 0.01_f64;
    // Fraction of voxels in beads for the reference volume
    let mut voxels_at_ref_limit = 6.7e-5_f64;

    let (com_ext, dual_num, _ds_root_name, type_ext, stack_ext) =
        find_root_axis_and_extensions(0, None);
    if dual_num < 0 || com_ext.is_empty() || type_ext.is_none() {
        exit_error(
            "Command files like tilt.com either are missing or have conflicting entries about critical information",
        );
    }
    let type_ext = type_ext.unwrap_or_default();

    // Get options
    let root_name = pip_get_in_out_file("RootName", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if root_name.is_empty() {
        exit_error("The root name of image files must be entered");
    }
    if dual_num > 0 && !root_name.ends_with(['a', 'b']) {
        exit_error(
            "Command files indicate this is a dual axis set but the entered root name does not end in a or b",
        );
    }
    let mut thickness = pip_get_integer("ThicknessOfTomograms", 0).unwrap_or(0) as i64;
    if thickness <= 0 {
        exit_error("A sample thickness must be entered");
    }
    let find_beads = pip_get_integer("FindBeadsInVolume", 0).unwrap_or(0);
    let mut bead_size = pip_get_float("BeadSize", 0.).unwrap_or(0.);
    let mut binning = pip_get_integer("Binning", 0).unwrap_or(0) as i64;
    let erase_frac = pip_get_float("EraseFraction", 0.002).unwrap_or(0.002);
    let high_sd_crit = pip_get_float("HighSDCriterion", 5.).unwrap_or(5.);
    let fs_opts = pip_get_string("FindSecOptions", "").unwrap_or_default();
    let leave_temp = pip_get_integer("LeaveTempFiles", 0).unwrap_or(0);
    let use_temp = pip_get_integer("UseTempFiles", 0).unwrap_or(0);
    let boost_thickness = pip_get_float("BoostThickness", 0.1).unwrap_or(0.1);
    let use_gpu = pip_get_integer("UseGPU", 0).unwrap_or(0);
    let gpu_entered = 1 - pip_get_err_no();

    // Get control values
    let num_control = pip_number_of_entries("ControlValue").unwrap_or(0);
    for _ in 0..num_control {
        let (con_type, con_val) = pip_get_two_floats("ControlValue", (0., 0.)).unwrap_or((0., 0.));
        let con_int = py_round(con_type) as i64;
        match con_int {
            1 => bead_maximum = con_val,
            2 => max_binning = con_val as i64,
            3 => x_oversize_frac = con_val,
            4 => y_oversize_frac = con_val,
            5 => find_avg_fallback = con_val,
            6 => find_store_fallback = con_val,
            7 => extra_hist_frac = con_val,
            8 => bead_vol_frac = con_val,
            9 => no_bead_diameter = con_val,
            10 => thresh_sum_factor = con_val,
            11 => min_bead_erase_frac = con_val,
            12 => max_bead_erase_frac = con_val,
            13 => voxels_at_ref_limit = con_val,
            _ => {}
        }
    }

    let mut com_suffix = format!(".{com_ext}");
    if dual_num > 0 {
        com_suffix = format!("{}.{com_ext}", &root_name[root_name.len() - 1..]);
    }

    let pitch_default = format!("tomopitch{}mod", &com_suffix[..com_suffix.len() - 3]);
    let pitch_model = pip_get_string("TomoPitchModel", &pitch_default).unwrap_or(pitch_default);
    let stack_name = format!("{root_name}.{stack_ext}");
    if !Path::new(&stack_name).exists() {
        exit_error(&format!("The raw stack {stack_name} does not exist"));
    }
    let (nx_stack, ny_stack, _nz_stack) = match get_mrc_size(&stack_name) {
        Ok((x, y, z)) => (x as i64, y as i64, z as i64),
        Err(error) => python_uncaught(&format!("imodpy.ImodpyError: {error}")),
    };
    let vol_size = ((nx_stack * ny_stack) as f64).sqrt();

    let mut light_beads = pip_get_boolean("LightFeatures", 0).unwrap_or(0) != 0;
    let light_entered = pip_get_err_no() == 0;

    // If doing beads, get size and whether beads are light
    if find_beads != 0 {
        let trackcom = format!("track{com_suffix}");
        let track_lines = read_text_file(&trackcom, None, false, None).unwrap_or_default();
        if bead_size <= 0. {
            // Fixed in translation (BUGS.md, `cryoposition`): a missing entry
            // is `None`, which the source compares with `<=` (a TypeError).
            bead_size = match option_value(
                &track_lines,
                "BeadDiameter",
                FLOAT_VALUE,
                false,
                1,
                None,
                None,
            ) {
                Some(OptionValue::Floats(values)) => values[0],
                _ => 0.,
            };
        }
        if bead_size <= 0. {
            // Fixed in translation (BUGS.md, `cryoposition`): the source names
            // an undefined `trackcom` here (a NameError).
            exit_error(&format!(
                "There is no positive BeadDiameter entry in {trackcom}; fix this or enter a size with -size"
            ));
        }
        if !light_entered {
            light_beads = matches!(
                option_value(&track_lines, "LightBeads", BOOL_VALUE, false, 0, None, None),
                Some(OptionValue::Boolean(true))
            );
        }

        // Get a binning based on bead size
        if binning <= 0 {
            binning = 1.max(py_round(bead_size / bead_optimal) as i64);
            if bead_size / binning as f64 > bead_maximum
                && bead_size / (binning + 1) as f64 > bead_minimum
            {
                binning += 1;
            }
            while bead_size / (binning as f64) < bead_minimum && binning > 1 {
                binning -= 1;
            }
            while vol_size / binning as f64 > 2. * vol_optimal
                && bead_size / (binning + 1) as f64 > bead_big_vol_min
            {
                binning += 1;
            }

            while binning > max_binning && bead_size / (binning - 1) as f64 <= bead_maximum {
                binning -= 1;
            }
        }

    // Or get a binning based on volume size
    } else if binning <= 0 {
        binning = max_binning.min(1.max(py_round(vol_size / vol_optimal) as i64));
    }

    // Now that binning is known, set up scales to apply in findsection
    let mut scales: Vec<i64> = match binning {
        1 => vec![3, 4, 6, 8],
        2 => vec![2, 3, 4],
        3 => vec![1, 2, 3],
        _ => vec![1, 2],
    };

    if let Some(scales_in) = pip_get_integer_array("ScalesToApply", 0).filter(|v| !v.is_empty()) {
        scales = scales_in;
    }
    let net_bin = scales[0] * binning;
    let (nxz_box, ny_box_default) = if net_bin == 1 {
        (48, 12)
    } else if net_bin == 2 {
        (32, 8)
    } else {
        (16, 4)
    };

    let (nx_box, ny_box, nz_box) = pip_get_three_integers(
        "BoxSizeInXYZ",
        (nxz_box, ny_box_default, nxz_box),
    )
    .unwrap_or((nxz_box, ny_box_default, nxz_box));
    let spacing_default = (
        1.max(nx_box.div_euclid(2)),
        1.max(ny_box.div_euclid(4)),
        1.max(nz_box.div_euclid(2)),
    );
    let (x_spacing, y_spacing, z_spacing) =
        pip_get_three_integers("SpacingOfBoxesInXYZ", spacing_default).unwrap_or(spacing_default);
    let mut polarity = -1;
    if light_beads {
        polarity = 1;
    }

    // Get a fallback bead diameter for erasure purposes and also adjust the volume fraction
    // to keep the minimum size up as beads get smaller
    if bead_size <= 0. {
        bead_size = no_bead_diameter;
    }
    let bead_binned = bead_size / binning as f64;
    let bead_vol = bead_binned.powi(3) * 3.1416 / 6.;
    let mut bvf = bead_vol_frac;
    if bead_binned < bead_optimal {
        bvf = 1.0_f64.min(bead_vol_frac * bead_optimal / bead_binned);
    }
    let min_for_thresh = 2.max(py_round(bvf * bead_vol) as i64);

    // Get the newstack com lines, make sure we can get transform file
    let newst_lines =
        read_text_file(&format!("newst{com_suffix}"), None, false, None).unwrap_or_default();
    let newst_lines = match extract_program_entries(&newst_lines, "newstack", "-Standard") {
        Some(lines) if !lines.is_empty() => lines,
        _ => exit_error(&format!(
            "The file newst{com_suffix} is in an older format and cannot be used"
        )),
    };
    let xf_file = match option_value(
        &newst_lines,
        "TransformFile",
        STRING_VALUE,
        false,
        0,
        None,
        None,
    ) {
        Some(OptionValue::String(value)) if !value.is_empty() => value,
        _ => exit_error(&format!(
            "Cannot find transform file name in newst{com_suffix}"
        )),
    };

    // Get transforms and set ali size, transposing if middle terms are bigger than outer
    let xf_lines = read_text_file(&xf_file, None, false, None).unwrap_or_default();
    let Some(line) = xf_lines.get(xf_lines.len() / 2) else {
        python_uncaught("IndexError: list index out of range")
    };
    let lsplit: Vec<&str> = line.split_whitespace().collect();
    let mut nx_full_ali = nx_stack;
    let mut ny_full_ali = ny_stack;
    let term = |index: usize| -> f64 {
        match lsplit.get(index).and_then(|text| py_float(text)) {
            Some(value) => value,
            None => exit_error(&format!("Trying to interpret transform in {xf_file}")),
        }
    };
    if term(0).abs() + term(3).abs() < term(1).abs() + term(2).abs() {
        ny_full_ali = nx_stack;
        nx_full_ali = ny_stack;
    }

    // Get the tilt lines then look up the X-axis tilt and increase thickness if needed
    let tilt_all_lines =
        read_text_file(&format!("tilt{com_suffix}"), None, false, None).unwrap_or_default();
    let tilt_lines = extract_program_entries(&tilt_all_lines, "tilt", "-Standard")
        .unwrap_or_else(|| python_uncaught("TypeError: 'NoneType' object is not iterable"));
    let xtilt = match option_value(&tilt_lines, "XAXISTILT", FLOAT_VALUE, false, 1, None, None) {
        Some(OptionValue::Floats(values)) => values[0],
        _ => 0.,
    };
    if xtilt != 0. {
        let mut extra_thick = py_round(thickness as f64 * (xtilt * 0.0174533).tan()) as i64;
        extra_thick += extra_thick.rem_euclid(2);
        if extra_thick as f64 > 0.01 * thickness as f64 {
            thickness += extra_thick;
            prnstr(
                &format!(
                    "Increasing thickness to {thickness} to compensate for the X-tilt in tilt{com_suffix}"
                ),
                "\n",
                false,
            );
        }
    }

    // Set up oversize size and the filenames
    let nx_full_over = (nx_full_ali as f64 * x_oversize_frac) as i64;
    let ny_full_over = (ny_full_ali as f64 * y_oversize_frac) as i64;
    let x_subset_start = (nx_full_ali - nx_full_over).div_euclid(2);
    let y_subset_start = (ny_full_ali - ny_full_over).div_euclid(2);
    set_root_and_extension(&root_name, &type_ext);
    let ali_name = dataset_filename("_cpos.ali", None, None);
    let full_rec_name = dataset_filename("_cpos.rec", None, None);
    let peak_model = format!("{root_name}_cpos.pkmod");
    let auto_model = format!("{root_name}_cposAuto.mod");
    let erase_rec = dataset_filename("_cposErase.rec", None, None);
    let thresh_rec = dataset_filename("_cposThresh.rec", None, None);
    let reproj_name = dataset_filename("_cpos.reproj", None, None);
    let box_stack = dataset_filename("_cposBox.st", None, None);
    let erase_ali = dataset_filename("_cposErase.ali", None, None);
    let mut clean_list = vec![
        ali_name.clone(),
        peak_model.clone(),
        auto_model.clone(),
        thresh_rec.clone(),
        reproj_name.clone(),
        box_stack.clone(),
        erase_ali.clone(),
    ];

    if leave_temp >= 0 || (-leave_temp).rem_euclid(2) == 0 {
        clean_list.push(full_rec_name.clone());
    }
    if leave_temp >= 0 || (-leave_temp).div_euclid(2) == 0 {
        clean_list.push(erase_rec.clone());
    }

    // Set environment variable to produce files of the right type
    set_output_format_if_needed(&type_ext, false);

    let s = |value: i64| value.to_string();
    let run = |command: &str, input: Option<&[String]>, outfile: Option<&str>| {
        run_cmd(command, input, outfile, None, &[]).map_err(|_| Failure::Imodpy)
    };
    let sed = |sedcom: &[String], lines: &[String]| -> Vec<String> {
        match pysed(sedcom, PysedSrc::Lines(lines), None, false, '/', false) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(message) => exit_error(&message),
        }
    };

    let result: Result<(), Failure> = (|| {
        // Oversized aligned stack
        let mut sedcom = vec![sed_modify("OutputFile", &ali_name, '/')];
        sedcom.extend(sed_del_and_add(
            "SizeToOutputInXandY",
            &format!(
                "{},{}",
                nx_full_over.div_euclid(binning),
                ny_full_over.div_euclid(binning)
            ),
            "OutputFile",
            '/',
        ));
        sedcom.extend(sed_del_and_add(
            "BinByFactor",
            &s(binning),
            "OutputFile",
            '/',
        ));
        sedcom.extend(sed_del_and_add("TaperAtFill", "1,0", "OutputFile", '/'));
        sedcom.extend(sed_del_and_add("AntialiasFilter", "-1", "OutputFile", '/'));
        let sedlines = sed(&sedcom, &newst_lines);
        let need_new_ali = !Path::new(&ali_name).exists();
        if use_temp < 1 || need_new_ali {
            prnstr(
                &format!("Building oversized aligned stack with binning = {binning}"),
                "\n",
                true,
            );
            run("newstack -StandardInput", Some(&sedlines), None)?;
        }

        // Oversized tomogram : First build base tilt sed command
        let mut tilt_base = sed_del_and_add("IMAGEBINNED", &s(binning), "OutputFile", '/');
        tilt_base.extend(sed_del_and_add("AdjustOrigin", "1", "OutputFile", '/'));
        tilt_base.extend([
            sed_modify("THICKNESS", &s(thickness), '/'),
            sed_modify(
                "SUBSETSTART",
                &format!("{x_subset_start} {y_subset_start}"),
                '/',
            ),
            sed_modify("XAXISTILT", "0.0", '/'),
        ]);

        // Get rid of a log value; modify scale value if log was there OR it was seemingly
        // not modified yet
        let floats = |option: &str| match option_value(
            &tilt_lines,
            option,
            FLOAT_VALUE,
            false,
            0,
            None,
            None,
        ) {
            Some(OptionValue::Floats(values)) => Some(values),
            _ => None,
        };
        let found_log = floats("LOG").is_some_and(|values| !values.is_empty());
        if found_log {
            tilt_base.push("/^ *LOG/d".to_owned());
        }
        let scale_arr = floats("SCALE").unwrap_or_default();
        if scale_arr.len() < 2 {
            exit_error(&format!(
                "Cannot verify or modify SCALE value in tilt{com_suffix} for linear scaling"
            ));
        }
        if found_log || scale_arr[1] > 3. {
            tilt_base.push(sed_modify(
                "SCALE",
                &format!("{} {:.3}", py_str_float(scale_arr[0]), scale_arr[1] / 5000.),
                '/',
            ));
        }

        if gpu_entered != 0 {
            tilt_base.extend(sed_del_and_add(
                "UseGPU",
                &use_gpu.to_string(),
                "OutputFile",
                '/',
            ));
        }

        // Get the rest of the command for oversized tomo
        let mut sedcom = tilt_base.clone();
        sedcom.extend([
            sed_modify("OutputFile", &full_rec_name, '/'),
            sed_modify("InputProjections", &ali_name, '/'),
            sed_modify("WIDTH", &s(nx_full_over), '/'),
            "/^ *SLICE/d".to_owned(),
        ]);
        let sedlines = sed(&sedcom, &tilt_lines);
        if use_temp < 2 || !Path::new(&full_rec_name).exists() {
            prnstr("Building oversized binned tomogram", "\n", true);
            run("tilt -StandardInput", Some(&sedlines), None)?;
        }

        let mut ind_lowest: i64 = -1;
        let mut ind_storing: i64 = -1;
        let mut num_stored: i64 = 0;
        let mut rec_threshold: Option<f64> = None;
        let need_reproj = use_temp < 5 || !Path::new(&reproj_name).exists();
        if find_beads != 0 && need_reproj {
            // Find beads in the subvolume corresponding to regular size reconstruction
            let beadcom = vec![
                format!("InputFile {full_rec_name}"),
                format!("OutputFile {peak_model}"),
                format!("BeadSize {}", py_str_float(bead_size)),
                "StorageThreshold -1".to_owned(),
                format!("BinningOfVolume {binning}"),
                format!("TiltFile {root_name}.tlt"),
                "YAxisElongated".to_owned(),
                format!(
                    "XMinAndMax {},{}",
                    (-x_subset_start).div_euclid(binning),
                    (nx_full_over + x_subset_start).div_euclid(binning)
                ),
                format!(
                    "ZMinAndMax {},{}",
                    (-y_subset_start).div_euclid(binning),
                    (ny_full_over + y_subset_start).div_euclid(binning)
                ),
                format!(
                    "FallbackThresholds {},{}",
                    py_str_float(find_avg_fallback),
                    py_str_float(find_store_fallback)
                ),
            ];

            prnstr("Finding beads in tomogram", "\n", true);
            let find_lines =
                run("findbeads3d -StandardInput", Some(&beadcom), None)?.unwrap_or_default();

            // Look for results and see if fallback storage is used, or nothing
            let mut elongation = 1.5_f64;
            for (ind, raw) in find_lines.iter().enumerate() {
                let line = raw.trim();
                if line.contains("lowest dip") {
                    ind_lowest = ind as i64;
                }
                if line.contains("Storing") && line.contains("peaks in model") {
                    ind_storing = ind as i64;
                    num_stored = line.split_whitespace().nth(1).and_then(py_int).unwrap_or(0);
                }
                if line.contains("using fallback storage threshold") {
                    prnstr(line, "\n", false);
                    ind_lowest = -1;
                }
                if line.contains("Elongation factor is") {
                    elongation = line
                        .split_whitespace()
                        .nth(3)
                        .and_then(py_float)
                        .ok_or(Failure::Interpret)?;
                }
            }

            if ind_lowest > 0 {
                prnstr(find_lines[ind_lowest as usize].trim(), "\n", false);
            }
            if ind_storing < 0 {
                prnstr(
                    "Bead-finding failed, falling back to erasing a small fraction of dense material",
                    "\n",
                    false,
                );
            } else {
                // Now if anything was stored, extract boxes
                // Make boxes even and
                prnstr(find_lines[ind_storing as usize].trim(), "\n", false);
                let radius = 0.5 * bead_size / binning as f64;
                let mut nxz_box_se = py_round(2. * radius + radius.max(6.)) as i64;
                let mut ny_box_se = py_round(elongation * (2. * radius + radius.max(6.))) as i64;
                nxz_box_se += nxz_box_se.rem_euclid(2);
                ny_box_se += ny_box_se.rem_euclid(2);
                let boxcom = vec![
                    format!("InputImageFile {full_rec_name}"),
                    format!("ModelFile {peak_model}"),
                    format!("OutputFile {box_stack}"),
                    format!("VolumeSizeXYZ {nxz_box_se},{ny_box_se},{nxz_box_se}"),
                ];
                if use_temp < 3 || !Path::new(&box_stack).exists() {
                    prnstr("Extracting boxed beads or densities", "\n", false);
                    run("boxstartend -StandardInput", Some(&boxcom), None)?;
                }

                // Look for threshold of extra counts on one size of peak in histogram
                prnstr("Analyzing histogram of boxed beads", "\n", false);
                match run_cmd(
                    &format!(
                        "clip hist -E {},{polarity} \"{box_stack}\"",
                        py_str_float(extra_hist_frac)
                    ),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(clip_lines) => {
                        rec_threshold = Some(get_threshold_from_clip_output(
                            &clip_lines.unwrap_or_default(),
                            "extra counts",
                        )?);
                    }
                    Err(_) => {
                        let err_strn = get_err_strings();
                        match err_strn.iter().find(|line| {
                            line.contains("fewer counts") || line.contains("too close to end")
                        }) {
                            Some(line) => {
                                prnstr(line.trim(), "\n", false);
                                prnstr(
                                    "Falling back to erasing a small fraction of dense material",
                                    "\n",
                                    false,
                                );
                            }
                            None => {
                                cleanup(leave_temp, &clean_list);
                                exit_from_imod_error(progname);
                            }
                        }
                    }
                }

                if let Some(threshold) = rec_threshold.filter(|value| *value != 0.) {
                    let full_rec_size =
                        (nx_full_over * ny_full_over * thickness) as f64 / binning as f64;
                    let mut max_erase = max_bead_erase_frac;
                    if ind_storing >= 0 && num_stored > 0 {
                        let voxel_frac = bead_vol * num_stored as f64 / full_rec_size;
                        let erase_lim_factor = ref_erase_limit / voxels_at_ref_limit;
                        prnstr(
                            &format!(
                                "Fraction of voxels in beads : {}",
                                crate::imod::libcfshr::b3dutil::c_format(
                                    "%g",
                                    &[crate::imod::libcfshr::b3dutil::CArg::Dbl(voxel_frac)]
                                )
                            ),
                            "\n",
                            false,
                        );
                        max_erase = voxel_frac * erase_lim_factor;
                        max_erase = max_bead_erase_frac.min(min_bead_erase_frac.max(max_erase));
                    }
                    prnstr(
                        &format!(
                            "Making sure that threshold does not select more than {max_erase:.4} of voxels"
                        ),
                        "\n",
                        false,
                    );
                    if light_beads {
                        max_erase = 1. - max_erase;
                    }
                    let clip_lines = run(
                        &format!(
                            "clip hist -t {} \"{full_rec_name}\"",
                            py_str_float(max_erase)
                        ),
                        None,
                        None,
                    )?
                    .unwrap_or_default();
                    let lim_threshold =
                        get_threshold_from_clip_output(&clip_lines, "Threshold value")?;
                    if (light_beads && lim_threshold > threshold)
                        || (!light_beads && lim_threshold < threshold)
                    {
                        prnstr(
                            "Using that threshold to limit number of selected pixels",
                            "\n",
                            false,
                        );
                        rec_threshold = Some(lim_threshold);
                    }
                }
            }
        }

        // If no beads, or no threshold was gotten that way, do histogram on whole volume
        if rec_threshold.is_none() && need_reproj {
            let mut frac = erase_frac;
            if light_beads {
                frac = 1. - frac;
            }
            prnstr(
                "Getting fallback threshold value from histogram of full volume",
                "\n",
                false,
            );
            let clip_lines = run(
                &format!("clip hist -t {} \"{full_rec_name}\"", py_str_float(frac)),
                None,
                None,
            )?
            .unwrap_or_default();
            rec_threshold = Some(get_threshold_from_clip_output(
                &clip_lines,
                "Threshold value",
            )?);
        }

        // Threshold the volume after determining good min and max values for it
        let full_header = |file: &str| match get_mrc(file, true, false) {
            Ok(MrcInfo::All(_, _, _, _, xp, yp, zp, xo, yo, zo, mn, mx, mean)) => {
                Ok([xp, yp, zp, xo, yo, zo, mn, mx, mean])
            }
            _ => Err(Failure::Imodpy),
        };
        let [_, _, _, _, _, _, tmin, tmax, tmean] = full_header(&full_rec_name)?;
        let mut low_for_thresh = 0.;
        let mut high_for_thresh = 0.;
        let threshold = rec_threshold.unwrap_or(0.);
        if light_beads && need_reproj {
            low_for_thresh = tmean;
            high_for_thresh = tmax.min(2. * threshold - tmean);
        } else if need_reproj {
            high_for_thresh = tmean;
            low_for_thresh = tmin.max(2. * threshold - tmean);
        }
        if use_temp < 4 || (need_reproj && !Path::new(&thresh_rec).exists()) {
            prnstr(
                &format!("Creating thresholded volume with minimum feature size {min_for_thresh}"),
                "\n",
                true,
            );
            run(
                &format!(
                    "clip thresh -t {} -M {min_for_thresh},{polarity} -l {} -h {} \"{full_rec_name}\" \"{thresh_rec}\"",
                    rec_threshold.map_or("None".to_owned(), py_str_float),
                    py_str_float(low_for_thresh),
                    py_str_float(high_for_thresh)
                ),
                None,
                None,
            )?;
        }

        if need_reproj {
            // Reproject thresholded volume
            let mut sedcom = tilt_base.clone();
            sedcom.extend([
                sed_modify("OutputFile", &reproj_name, '/'),
                sed_modify("InputProjections", &ali_name, '/'),
                sed_modify("WIDTH", &s(nx_full_over), '/'),
                "/^ *SLICE/d".to_owned(),
                "/^ *EXCLUDELIST/d".to_owned(),
            ]);
            sedcom.extend(sed_del_and_add(
                "RecFileToReproject",
                &thresh_rec,
                "OutputFile",
                '/',
            ));
            sedcom.extend(sed_del_and_add(
                "ThresholdedReproj",
                &format!(
                    "{} {polarity} {}",
                    py_str_float((low_for_thresh + high_for_thresh) / 2.),
                    py_str_float(thresh_sum_factor)
                ),
                "OutputFile",
                '/',
            ));
            let sedlines = sed(&sedcom, &tilt_lines);
            prnstr("Reprojecting thresholded volume", "\n", true);
            run("tilt -StandardInput", Some(&sedlines), None)?;

            // Fix the header in the reprojection to match the oversize ali
            let [x_pix, y_pix, z_pix, x_orig, y_orig, z_orig, ..] = full_header(&ali_name)?;
            run(
                &format!(
                    "alterheader -del {},{},{} -org {},{},{} \"{reproj_name}\"",
                    py_str_float(x_pix),
                    py_str_float(y_pix),
                    py_str_float(z_pix),
                    py_str_float(x_orig),
                    py_str_float(y_orig),
                    py_str_float(z_orig)
                ),
                None,
                None,
            )?;
        }

        // Make contours around thresholded density
        let mut opt = "-l";
        if light_beads {
            opt = "-h";
        }
        if use_temp < 6 || !Path::new(&auto_model).exists() {
            prnstr("Making contours around reprojected density", "\n", true);
            run(
                &format!("imodauto {opt} 128 -m 1 -f 3 -x \"{reproj_name}\" \"{auto_model}\""),
                None,
                None,
            )?;
        }

        // Erase from the aligned stack.  There is no need to make separate output file
        let ccdcom = vec![
            format!("InputFile {ali_name}"),
            format!("OutputFile {erase_ali}"),
            format!("ModelFile {auto_model}"),
            "BoundaryObjects 1".to_owned(),
            "PolynomialOrder 0".to_owned(),
        ];
        if use_temp < 7 || need_new_ali {
            prnstr(
                "Erasing high-density regions from aligned stack",
                "\n",
                true,
            );
            run("ccderaser -StandardInput", Some(&ccdcom), None)?;
        }

        // Make new tomogram, back down to regular size, from erased stack
        let mut sedcom = tilt_base.clone();
        sedcom.extend([
            sed_modify("OutputFile", &erase_rec, '/'),
            sed_modify("InputProjections", &erase_ali, '/'),
        ]);
        sedcom.extend(sed_del_and_add("WIDTH", &s(nx_full_ali), "SCALE", '/'));
        sedcom.extend(sed_del_and_add(
            "SLICE",
            &format!(
                "{} {}",
                -y_subset_start,
                ny_full_over + y_subset_start - binning
            ),
            "SCALE",
            '/',
        ));
        let sedlines = sed(&sedcom, &tilt_lines);
        // Fixed in translation (BUGS.md, `cryoposition`): with `-use 8` the
        // source checks `threshRec`, not the volume this step makes.
        if use_temp < 8 || !Path::new(&erase_rec).exists() {
            prnstr("Building tomogram from erased stack", "\n", true);
            run("tilt -StandardInput", Some(&sedlines), None)?;
        }

        // Find the section at last
        let mut findcom = vec![
            format!("TomogramFile {erase_rec}"),
            format!("HighSDboxCriterion {}", py_str_float(high_sd_crit)),
            format!("BoostHighSDThickness {}", py_str_float(boost_thickness)),
            format!("SizeOfBoxesInXYZ {nx_box},{ny_box},{nz_box}"),
            format!("SpacingInXYZ {x_spacing},{y_spacing},{z_spacing}"),
            format!("TomoPitchModel {pitch_model}"),
        ];
        for scale in &scales {
            findcom.push(format!("BinningInXYZ {scale},{scale},{scale}"));
        }
        if ((!need_reproj && Path::new(&box_stack).exists()) || ind_storing >= 0) && find_beads > 1
        {
            findcom.extend([
                format!("BeadModelFile {peak_model}"),
                format!("BeadDiameter {}", py_str_float(bead_size / binning as f64)),
            ]);
            if find_beads == 1 {
                findcom.push("ControlValue 29,0.".to_owned());
            }
        }
        prnstr(
            "Analyzing structure to find material to include",
            "\n",
            true,
        );
        run(
            &format!("findsection {fs_opts} -StandardInput"),
            Some(&findcom),
            Some("stdout"),
        )?;
        prnstr("Tomopitch model created", "\n", true);
        Ok(())
    })();

    match result {
        Ok(()) => {
            cleanup(leave_temp, &clean_list);
            done(0)
        }
        Err(Failure::Imodpy) => {
            cleanup(leave_temp, &clean_list);
            exit_from_imod_error(progname);
        }
        Err(Failure::Interpret) => {
            cleanup(leave_temp, &clean_list);
            exit_error("An error occurred interpreting program output");
        }
    }
}
