//! Translation of `IMOD/pysrc/reducefiltvol`: reduces a volume with binvol
//! and/or filters it with mtffilter, setting up chunks for parallel filtering
//! if mtffilter runs out of memory.
//!
//! A Python command script with no functions; its top level is
//! [`reducefiltvol`].  `binvol`, `mtffilter` and `header` (through `getmrc`)
//! are our own programs and run in process (`imodpy::run_cmd`); `chunksetup`
//! is a Python-script translation, which `run_cmd` runs as a child.  Python
//! values keep their types: the factors, filter parameters and pixel size
//! are floats written with `str()` ([`py_str_float`]); `Voltage` is an int
//! when entered or read, and the float default `0.` otherwise.

use super::imodpy::{
    FLOAT_VALUE, INT_VALUE, MrcInfo, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup,
    cleanup_files, exit_from_imod_error, find_root_axis_and_extensions, fmtstr, get_err_strings,
    get_mrc, option_value, os_path_splitext, prnstr, py_float, py_int, py_str_float,
    read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_two_floats, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`reducefiltvol:1-330`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn reducefiltvol(arguments: &[OsString]) -> i32 {
    let progname = "reducefiltvol";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 reducefiltvol
    let options: Vec<String> = [
        "input:InputFile:FN:",
        "output:OutputFile:FN:",
        "reduce:ReductionFactor:F:",
        "zfactor:ZReductionFactor:F:",
        "lowpass:LowPassRadiusSigma:FP:",
        "deconv:DeconvolutionStrength:F:",
        "snr:SNRFalloff:F:",
        "dchigh:HighPassNyquist:F:",
        "pixel:PixelSize:F:",
        "volt:Voltage:I:",
        "cs:SphericalAberration:F:",
        "defocus:DefocusInMicrons:F:",
        "mode:PhaseShift:I:",
        "setup:SetupChunksIfMemoryError:B:",
        "param:ParameterFile:PF:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    //
    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 3, 1, 1);

    // Input and output files
    let input_file = pip_get_in_out_file("InputFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if input_file.is_empty() {
        exit_error("An input file must be entered");
    }
    let output_file = pip_get_in_out_file("OutputFile", 1)
        .ok()
        .flatten()
        .unwrap_or_default();
    if output_file.is_empty() {
        exit_error("An output file must be entered");
    }

    if !Path::new(&input_file).exists() {
        exit_error(&format!("Input file {input_file} does not exist"));
    }

    // Reduction factors
    let factor = pip_get_float("ReductionFactor", 1.).unwrap_or(1.);
    let if_factor = 1 - pip_get_err_no();
    let z_factor = pip_get_float("ZReductionFactor", factor).unwrap_or(factor);
    let do_reduce = (if_factor != 0 || pip_get_err_no() == 0) && (factor != 1. || z_factor != 1.);
    if factor < 1. || z_factor < 1. {
        exit_error("Reduction factors must be greater than or equal to 1");
    }

    // Filtering main options
    let (radius, sigma) = pip_get_two_floats("LowPassRadiusSigma", (0., 0.)).unwrap_or((0., 0.));
    let do_gaussian = 1 - pip_get_err_no();
    let deconv_strength = pip_get_float("DeconvolutionStrength", 0.).unwrap_or(0.);
    let do_deconv = 1 - pip_get_err_no();
    if do_gaussian != 0 && do_deconv != 0 {
        exit_error("You cannot do both a Gaussian filter and a deconvolution filter");
    }

    let do_filter = do_gaussian != 0 || do_deconv != 0;
    let setup_chunks = pip_get_boolean("SetupChunksIfMemoryError", 0).unwrap_or(0);

    if !do_reduce && !do_filter {
        exit_error("There is no meaningful operation specified");
    }

    // Get all the optional entries, keep track if gotten
    let snr_falloff = pip_get_float("SNRFalloff", 0.).unwrap_or(0.);
    let if_snr = 1 - pip_get_err_no();
    let high_pass = pip_get_float("HighPassNyquist", 0.).unwrap_or(0.);
    let if_high_pass = 1 - pip_get_err_no();
    let mut defocus = pip_get_float("DefocusInMicrons", 0.).unwrap_or(0.);
    let _if_defocus = 1 - pip_get_err_no();
    let phase = pip_get_float("PhaseShift", 0.).unwrap_or(0.);
    let if_phase = 1 - pip_get_err_no();
    let mut pixel_size = pip_get_float("PixelSize", 0.).unwrap_or(0.);
    let mut if_pixel = 1 - pip_get_err_no();
    let mut spher_aber = pip_get_float("SphericalAberration", 0.).unwrap_or(0.);
    let mut if_cs = 1 - pip_get_err_no();
    // `PipGetInteger('Voltage', 0.)`: the int entered, or the float default;
    // the text `str()` gives is kept.
    let voltage_int = pip_get_integer("Voltage", 0).unwrap_or(0);
    let mut if_voltage = 1 - pip_get_err_no();
    let mut voltage_text = if if_voltage != 0 {
        voltage_int.to_string()
    } else {
        "0.0".to_owned()
    };
    let out_mode = pip_get_integer("ModeToOutput", 0).unwrap_or(0);
    let if_mode = 1 - pip_get_err_no();

    // Set up file names if intermediate file
    let mut red_output = output_file.clone();
    let mut filt_input = input_file.clone();
    if do_reduce && do_filter {
        let (out_root, out_ext) = os_path_splitext(&output_file);
        red_output = format!("{out_root}.filttemp{out_ext}");
        filt_input = red_output.clone();
    }

    // Now get ctfplotter com file if it is needed for anything in there
    if do_deconv != 0 && !(if_pixel != 0 && if_cs != 0 && if_voltage != 0) {
        let mut plotcom = "ctfplotter.com".to_owned();
        let (_com_ext, dual_num, root_name, _type_ext, _stack_ext) =
            find_root_axis_and_extensions(0, None);
        if dual_num == 2 {
            plotcom = "ctfplottera.com".to_owned();
        }

        let px_vol = match get_mrc(&input_file, false, false) {
            Ok(MrcInfo::Basic(_, _, _, _, px, _, _)) => px,
            Ok(_) => 0.,
            Err(_) => exit_from_imod_error(progname),
        };

        let mut mess = String::new();
        let mut vmess = String::new();
        let mut cs_mess = String::new();
        let mut ctf_lines: Vec<String> = Vec::new();
        if !Path::new(&plotcom).exists() {
            mess = format!("{plotcom} does not exist");
            vmess = mess.clone();
            cs_mess = mess.clone();
        } else {
            ctf_lines = read_text_file(&plotcom, None, false, None).unwrap_or_default();
        }

        // Find the pixel size and raw stack
        let mut raw_pixel = 0.0_f64;
        let mut stack_input = String::new();
        if if_pixel == 0 && mess.is_empty() {
            match option_value(&ctf_lines, "PixelSize", FLOAT_VALUE, false, 1, None, None) {
                Some(OptionValue::Floats(values)) if values[0] != 0. => {
                    raw_pixel = values[0];
                    stack_input = match option_value(
                        &ctf_lines,
                        "InputStack",
                        STRING_VALUE,
                        false,
                        0,
                        None,
                        None,
                    ) {
                        Some(OptionValue::String(value)) => value,
                        _ => String::new(),
                    };
                    if stack_input.is_empty() {
                        mess = format!("could not find name of raw stack in {plotcom}");
                    } else if !Path::new(&stack_input).exists() {
                        mess = format!("raw stack file {stack_input} not found");
                    }
                }
                _ => {
                    mess = format!("could not find raw stack pixel size in {plotcom}");
                }
            }
        }

        // Read the headers to determine binning and get final pixel size
        if if_pixel == 0 && mess.is_empty() {
            match get_mrc(&stack_input, false, false) {
                Ok(MrcInfo::Basic(_, _, _, _, px_raw, _, _)) => {
                    pixel_size = raw_pixel * px_vol / px_raw;
                    if do_reduce {
                        pixel_size *= factor;
                    }
                    if_pixel = 1;
                }
                _ => {
                    mess = format!("there was an error reading the header of {stack_input}");
                }
            }
        }

        if !mess.is_empty() && if_pixel == 0 {
            pixel_size = px_vol / 10.;
            prnstr(
                &format!(
                    "Assuming pixel size of {pixel_size:.3} nm in volume header is correct; {mess}"
                ),
                "\n",
                false,
            );
            if do_reduce {
                pixel_size *= factor;
            }
            if_pixel = 1;
        }

        // Collect voltage and/or Cs and just go on if not there
        if if_voltage == 0 && vmess.is_empty() {
            match option_value(&ctf_lines, "Voltage", INT_VALUE, false, 1, None, None) {
                Some(OptionValue::Integers(values)) if values[0] != 0 => {
                    voltage_text = values[0].to_string();
                    if_voltage = 1;
                }
                _ => {
                    vmess = format!("could not find value in {plotcom}");
                }
            }
        }

        if if_voltage == 0 && !vmess.is_empty() {
            prnstr(&format!("Assuming voltage is 300; {vmess}"), "\n", false);
        }

        if if_cs == 0 && cs_mess.is_empty() {
            match option_value(
                &ctf_lines,
                "SphericalAberration",
                FLOAT_VALUE,
                false,
                1,
                None,
                None,
            ) {
                Some(OptionValue::Floats(values)) if values[0] != 0. => {
                    spher_aber = values[0];
                    if_cs = 1;
                }
                _ => {
                    cs_mess = format!(" could not find value in {plotcom}");
                }
            }
        }

        if if_cs == 0 && !cs_mess.is_empty() {
            prnstr(
                &format!("Assuming spherical aberration is 2.7; {cs_mess}"),
                "\n",
                false,
            );
        }

        // For defocus, need a file to analyze
        if defocus == 0. {
            if root_name.is_empty() {
                exit_error("Defocus must be entered; could not determine root name of dataset");
            }
            let mut def_file = format!("{root_name}.defocus");
            if dual_num == 2 {
                def_file = format!("{root_name}a.defocus");
                if !Path::new(&def_file).exists() {
                    def_file = format!("{root_name}b.defocus");
                }
            }

            if !Path::new(&def_file).exists() {
                exit_error(&format!(
                    "Defocus must be entered; could not find {def_file}"
                ));
            }

            // Figure out what's there from the header
            let def_lines = read_text_file(&def_file, None, false, None).unwrap_or_default();
            let lsplit: Vec<&str> = def_lines
                .first()
                .map(|line| line.split_whitespace().collect())
                .unwrap_or_default();
            let mut astig = false;
            let mut start_line = 0;
            if lsplit.len() < 5 {
                exit_error(&format!(
                    "Defocus must be entered; the defocus file {def_file} has too few entries on first line"
                ));
            }
            if lsplit.len() == 6 {
                match py_int(lsplit[5]) {
                    Some(vers_num) => {
                        if vers_num > 2 {
                            match py_int(lsplit[0]) {
                                Some(flags) => {
                                    astig = flags.rem_euclid(2) != 0;
                                    start_line = 1;
                                }
                                None => exit_error(&format!(
                                    "Defocus must be entered; an error occurred converting a value to integer on first line of {def_file}"
                                )),
                            }
                        }
                    }
                    None => exit_error(&format!(
                        "Defocus must be entered; an error occurred converting a value to integer on first line of {def_file}"
                    )),
                }
            }

            if start_line > 0 && def_lines.len() < 2 {
                exit_error(&format!(
                    "Defocus must be entered; there is only a header line in {def_file}"
                ));
            }

            // Get the defocus at minimum angle
            let mut min_angle = 1000.0_f64;
            for line in &def_lines[start_line.min(def_lines.len())..] {
                let lsplit: Vec<&str> = line.split_whitespace().collect();
                let parsed = (|| -> Option<()> {
                    let low_angle = py_float(lsplit.get(2)?)?;
                    let high_angle = py_float(lsplit.get(3)?)?;
                    let angle = (low_angle + high_angle).abs() / 2.;
                    if angle < min_angle {
                        min_angle = angle;
                        defocus = py_float(lsplit.get(4)?)?;
                        if astig {
                            defocus = (defocus + py_float(lsplit.get(5)?)?) / 2.;
                        }
                        defocus /= 1000.;
                    }
                    Some(())
                })();
                if parsed.is_none() {
                    exit_error(&format!(
                        "Defocus must be entered; an error occurred trying to analyze {def_file}"
                    ));
                }
            }
        }
    }

    // Set up filter com lines before running in case of program error
    let mut comlines: Vec<String> = Vec::new();
    if do_filter {
        comlines = vec![
            format!("InputFile {filt_input}"),
            format!("OutputFile {output_file}"),
            "FilterIn3D 1".to_owned(),
        ];
        if if_mode != 0 {
            comlines.push(format!("ModeToOutput {out_mode}"));
        }
        if do_gaussian != 0 {
            comlines.push(fmtstr(
                "LowPassRadiusSigma {},{}",
                &[py_str_float(radius), py_str_float(sigma)],
            ));
        } else {
            comlines.extend([
                format!("DeconvolutionStrength {}", py_str_float(deconv_strength)),
                format!("PixelSize {}", py_str_float(pixel_size)),
                format!("Defocus {}", py_str_float(defocus)),
            ]);
            if if_snr != 0 {
                comlines.push(format!("SNRFalloff {}", py_str_float(snr_falloff)));
            }
            if if_high_pass != 0 {
                comlines.push(format!("HighPassNyquist {}", py_str_float(high_pass)));
            }
            if if_phase != 0 {
                comlines.push(format!("PhaseShift {}", py_str_float(phase)));
            }
            if if_voltage != 0 {
                comlines.push(format!("Voltage {voltage_text}"));
            }
            if if_cs != 0 {
                comlines.push(format!("SphericalAberration {}", py_str_float(spher_aber)));
            }
        }
    }

    // Run reduction
    // SAFETY: a single-threaded script setting its own environment, as the
    // source's `os.environ[...] =` does.
    unsafe { std::env::set_var("IMOD_BRIEF_HEADER", "1") };
    if do_reduce {
        let mut mode_opt = String::new();
        if if_mode != 0 {
            mode_opt = format!("-mode {out_mode}");
        }
        prnstr("Reducing the volume with Binvol...", "\n", false);
        if run_cmd(
            &fmtstr(
                "binvol -xbin {} -ybin {} -zbin {} -anti -1 {} \"{}\" \"{}\"",
                &[
                    py_str_float(factor),
                    py_str_float(factor),
                    py_str_float(z_factor),
                    mode_opt,
                    input_file.clone(),
                    red_output.clone(),
                ],
            ),
            None,
            Some("stdout"),
            None,
            &[],
        )
        .is_err()
        {
            exit_from_imod_error(progname);
        }
    }

    // Run filter
    if do_filter {
        prnstr("Filtering the volume with Mtffilter...", "\n", false);
        match run_cmd("mtffilter -StandardInput", Some(&comlines), None, None, &[]) {
            Ok(filt_lines) => {
                // Filter, print output, and clean up and exit
                for line in filt_lines.unwrap_or_default() {
                    prnstr(line.trim_end(), "\n", false);
                }
                if do_reduce {
                    cleanup_files(&[filt_input]);
                }
                return done(0);
            }
            Err(_) => {
                if setup_chunks != 0 {
                    // Check for memory error
                    let err_strings = get_err_strings();
                    match err_strings.iter().find(|line| line.contains("[MTF1]")) {
                        Some(line) => {
                            // The Python line keeps the ending `runcmd` read it
                            // with; `run_cmd` returns collected lines without one.
                            prnstr(&format!("Mtffilter exited with: {line}\n"), "\n", false);
                            prnstr(
                                "Setting up chunks for filtering the volume in parallel",
                                "\n",
                                true,
                            );
                        }
                        None => exit_from_imod_error(progname),
                    }
                } else {
                    exit_from_imod_error(progname);
                }
            }
        }

        // Have to make chunk files
        comlines[0] = "InputFile INPUTFILE".to_owned();
        comlines[1] = "OutputFile OUTPUTFILE".to_owned();
        comlines.insert(0, "$mtffilter -StandardInput".to_owned());
        let _ = write_text_file("rfvfilter.com", &comlines, false);
        if run_cmd(
            &fmtstr(
                "chunksetup -master rfvfilter.com \"{}\" \"{}\"",
                &[filt_input.clone(), output_file.clone()],
            ),
            None,
            Some("stdout"),
            None,
            &[],
        )
        .is_err()
        {
            exit_from_imod_error(progname);
        }

        // Add cleanup line to finish file, forgive errors
        if do_reduce {
            let finish = "rfvfilter-finish.com";
            match read_text_file(finish, None, true, None) {
                Err(message) => {
                    prnstr(
                        &format!(
                            "WARNING: You will have to remove {filt_input} when done; failed to read {finish} with error: {message}"
                        ),
                        "\n",
                        false,
                    );
                }
                Ok(mut fin_lines) => {
                    fin_lines.push(format!("$b3dremove {filt_input}"));
                    if write_text_file(finish, &fin_lines, true).is_err() {
                        prnstr(
                            &format!(
                                "WARNING: You will have to remove {filt_input} when done; failed to write {finish}"
                            ),
                            "\n",
                            false,
                        );
                    }
                }
            }
        }
    }

    done(0)
}
