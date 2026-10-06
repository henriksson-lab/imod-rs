//! Translation of `IMOD/pysrc/sorttiltframes`: sorts out filenames by tilt
//! angles embedded in their names.
//!
//! A Python command script with no functions; its top level is
//! [`sorttiltframes`].  `extracttilts` is our own program and runs in process
//! (`imodpy::run_cmd_in_process`, which keeps the line endings the
//! script's lines carry).  Its angles are still parsed from its printed text
//! (see `TODO.md`): with a metadata file, `extracttilts` prints a message
//! after the last blank line and the script, like native, fails on it.  The delimiter pattern is a Python regular expression
//! built from the user's delimiters, converted with [`py_regex`].

use super::imodpy::{
    add_imod_bin_ignore_sighup, exit_from_imod_error, fmtstr, prnstr, py_float, py_regex,
    py_str_float, read_text_file, run_cmd_in_process, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_non_option_arg,
    pip_get_string, pip_number_of_entries, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`sorttiltframes:1-232`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn sorttiltframes(arguments: &[OsString]) -> i32 {
    let progname = "sorttiltframes";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 sorttiltframes
    let options: Vec<String> = [
        "input:InputFile:FNM:",
        "listin:ListOfInputFiles:FN:",
        "stack:TiltSeriesFile:FN:",
        "angle:TiltAngleFile:FN:",
        "outlist:OutputFileList:FN:",
        "tilt:OutputTiltAngleFile:FN:",
        "reverse:ReverseOrder:B:",
        "unsorted:UnsortedOutput:B:",
        "delim:Delimiters:CH:",
        "fixed:FixedImageDose:F:",
        "dose:DoseOutputFile:FN:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // PIP startup and help
    let (_opts, num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 2, 0);

    // Read all options
    let out_file = pip_get_string("OutputFileList", "").unwrap_or_default();
    let list_file = pip_get_string("ListOfInputFiles", "").unwrap_or_default();
    let angle_file = pip_get_string("TiltAngleFile", "").unwrap_or_default();
    let tilt_series = pip_get_string("TiltSeriesFile", "").unwrap_or_default();
    let tilt_out_file = pip_get_string("OutputTiltAngleFile", "").unwrap_or_default();
    let delims = pip_get_string("Delimiters", "[]").unwrap_or_default();
    let reverse = pip_get_boolean("ReverseOrder", 0).unwrap_or(0);
    let rev_entered = 1 - pip_get_err_no();
    let unsorted = pip_get_boolean("UnsortedOutput", 0).unwrap_or(0);
    let fixed_dose = pip_get_float("FixedImageDose", 0.).unwrap_or(0.);
    let dose_output = pip_get_string("DoseOutputFile", "").unwrap_or_default();

    // Check for problems and conflicting options
    let delim_chars: Vec<char> = delims.chars().collect();
    if delim_chars.len() != 2 {
        exit_error("You must enter exactly two delimiting characters with -d");
    }

    if unsorted != 0 && rev_entered != 0 {
        exit_error("You cannot enter both -u for unsorted and -r for reverse-sorted");
    }

    if (fixed_dose > 0. && dose_output.is_empty()) || (fixed_dose <= 0. && !dose_output.is_empty())
    {
        exit_error("You must enter both -dose and -fixed if one is entered");
    }

    let mut angles: Vec<f64> = Vec::new();
    let mut out_list: Vec<String> = Vec::new();
    let mut dose_list: Vec<String> = Vec::new();
    let angle_tol = 0.1_f64;
    // Fixed in translation (BUGS.md, `sorttiltframes`): the source pastes the
    // delimiters into the pattern unescaped, so the default `[]` opens a
    // character class and a name with `[-3.00]` sorts as angle 3.00.  The
    // delimiters are matched literally here, as `re.escape` would.
    let delim_match = match regex::Regex::new(&format!(
        "{}{}{}",
        regex::escape(&delim_chars[0].to_string()),
        py_regex("\\-?[0-9][0-9]?\\.[0-9][0-9]?"),
        regex::escape(&delim_chars[1].to_string())
    )) {
        Ok(regex) => regex,
        Err(error) => exit_error(&error.to_string()),
    };

    let num_in_by_opt = pip_number_of_entries("InputFile").unwrap_or(0);
    let num_total_in = num_in_by_opt + num_non_opts;

    if !list_file.is_empty() && num_total_in != 0 {
        exit_error("You cannot enter a list file and filenames as command line arguments too");
    }

    if list_file.is_empty() && num_total_in < 2 {
        exit_error("You must enter at least two filenames on the command line or a list file");
    }

    // Get the filename from the list file or the non-option arguments
    let file_list: Vec<String> = if !list_file.is_empty() {
        read_text_file(&list_file, None, false, None).unwrap_or_default()
    } else {
        let mut list = Vec::new();
        for ind in 0..num_total_in {
            if ind < num_in_by_opt {
                list.push(pip_get_string("InputFile", "").unwrap_or_default());
            } else {
                list.push(pip_get_non_option_arg(ind - num_in_by_opt).unwrap_or_default());
            }
        }
        list
    };

    // If tilt angles to be matched, check for conflicts
    if !tilt_series.is_empty() || !angle_file.is_empty() {
        if rev_entered != 0 {
            exit_error(
                "You cannot enter a file with angles to match and also indicate the direction to sort",
            );
        }
        if unsorted != 0 {
            exit_error(
                "You cannot get unsorted output lists when entering a file with angles to match",
            );
        }

        if !tilt_series.is_empty() && !angle_file.is_empty() {
            exit_error("You cannot enter both \"-s tiltSeries\" and \"-a angleFile\"");
        }

        // Use extracttilts to get the tilt angles from the file
        let angle_text: Vec<String> = if !tilt_series.is_empty() {
            let extract_lines =
                match run_cmd_in_process(&format!("extracttilts \"{tilt_series}\""), None, None) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };

            // Find first blank line from the end
            match (0..extract_lines.len())
                .rev()
                .find(|&ind| extract_lines[ind].trim().is_empty())
            {
                Some(ind) => extract_lines[ind + 1..].to_vec(),
                None => {
                    exit_error("Cannot find blank line before angles in output from extracttilts")
                }
            }
        } else {
            // Or just read the tilt file
            read_text_file(&angle_file, None, false, None).unwrap_or_default()
        };

        // Convert the values
        for line in &angle_text {
            match py_float(line) {
                Some(value) => angles.push(value),
                None => exit_error(&format!("Converting angle value {line} to float")),
            }
        }

        if angles.is_empty() {
            exit_error("Angle list from file is empty");
        }
    }

    // Analyze the filenames for floating point number between delimiters
    let mut templist: Vec<String> = Vec::new();
    let mut file_angles: Vec<f64> = Vec::new();
    let mut num_toss = 0;
    let mut num_multiple = 0;
    let mut accum_dose: f64 = 0.;
    let mut accum_list: Vec<f64> = Vec::new();
    for item in &file_list {
        let ind_last = 0usize;
        let mut used = false;
        let mut angle = 0.0_f64;
        if item.trim().is_empty() {
            continue;
        }
        while delim_match.is_match(&item[ind_last..]) {
            if used {
                num_multiple += 1;
                used = false;
                break;
            }

            let mat = delim_match.find(&item[ind_last..]).unwrap();

            // Try converting the number and save filename and angle to list
            if let Some(value) = py_float(&item[mat.start() + 1..mat.end() - 1]) {
                angle = value;
                used = true;
                break;
            }
        }

        if used {
            file_angles.push(angle);
            templist.push(item.clone());
            if fixed_dose > 0. {
                accum_list.push(accum_dose);
                accum_dose += fixed_dose;
            }
        } else {
            num_toss += 1;
        }
    }

    if file_angles.is_empty() {
        exit_error(
            "None of the entered filenames have a number with decimal point between the delimiters",
        );
    }

    if num_multiple != 0 {
        prnstr(
            &fmtstr(
                "WARNING: {} - Multiple parts of the name look like a tilt angle in {} names",
                &[progname.to_owned(), num_multiple.to_string()],
            ),
            "\n",
            false,
        );
    }

    // Make list of unique angles and sort them if no angles entered
    if angles.is_empty() {
        for &tilt in &file_angles {
            if !angles.contains(&tilt) {
                angles.push(tilt);
            }
        }
        if unsorted == 0 {
            // `angles.sort(reverse = reverse)`: a stable sort; with reverse the
            // order is descending with equal items keeping their order.
            if reverse != 0 {
                angles.sort_by(|a, b| b.partial_cmp(a).unwrap());
            } else {
                angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
        }
    }

    // For each angle in list, find the file angles that match within tolerance and add to
    // output list
    let mut out_tilts: Vec<String> = Vec::new();
    for &angle in &angles {
        let mut num_found = 0;
        for ind in 0..file_angles.len() {
            if (angle - file_angles[ind]).abs() < angle_tol {
                num_found += 1;
                out_list.push(templist[ind].clone());
                // `fmtstr('{:8.2f}', angle)`
                out_tilts.push(format!("{angle:8.2}"));
                if fixed_dose != 0. {
                    dose_list.push(fmtstr(
                        "{}  {}",
                        &[py_str_float(accum_list[ind]), py_str_float(fixed_dose)],
                    ));
                }
            }
        }

        if num_found == 0 {
            exit_error(&fmtstr(
                "No files were found with an angle within {} of tilt angle {}",
                &[py_str_float(angle_tol), py_str_float(angle)],
            ));
        }
    }

    // Give warning outputs of unused or non-matching files
    if num_toss != 0 {
        prnstr(
            &fmtstr(
                "WARNING: {} - {} filenames did not have a number with decimal point between the delimiters",
                &[progname.to_owned(), num_toss.to_string()],
            ),
            "\n",
            false,
        );
    }

    if out_list.len() < file_angles.len() {
        prnstr(
            &fmtstr(
                "WARNING: {} - {} filenames did not have a number matching any tilt angle",
                &[
                    progname.to_owned(),
                    (file_angles.len() - out_list.len()).to_string(),
                ],
            ),
            "\n",
            false,
        );
    }

    // Output the list to file or terminal
    if !out_file.is_empty() {
        let _ = write_text_file(&out_file, &out_list, false);
    } else {
        for name in &out_list {
            prnstr(name, "\n", false);
        }
    }

    if !tilt_out_file.is_empty() {
        let _ = write_text_file(&tilt_out_file, &out_tilts, false);
        if out_file.is_empty() && num_toss != 0 {
            prnstr(
                &fmtstr(
                    "WARNING: {} - {} of the input filenames were omitted from the output lists",
                    &[progname.to_owned(), num_multiple.to_string()],
                ),
                "\n",
                false,
            );
        }
    }

    if !dose_output.is_empty() {
        let _ = write_text_file(&dose_output, &dose_list, false);
    }

    let _ = std::io::stdout().flush();
    0
}
