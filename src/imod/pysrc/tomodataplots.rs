//! Translation of `IMOD/pysrc/tomodataplots`: plots selected data from
//! tomogram processing (logs of `blendmont`, `tiltalign`, `clip stats`,
//! `ctfplotter`, `alignframes`, ...) through `onegenplot`.
//!
//! The script's top level is [`tomodataplots`]; its functions are
//! [`find_limiting_lines`], [`extract_bin_and_rad2`] and
//! [`add_to_alignframes_arrays`].  `onegenplot` is a Python-script
//! translation of this crate and runs as `imodpy::run_cmd` /
//! `bkgd_process` run it (as a child of our own binary); it in turn runs our
//! plot window program `genhstplt` (`flib/graphics/genhstplt.rs`).

use super::imodpy::{
    add_imod_bin_ignore_sighup, bkgd_process, cleanup_files, exit_from_imod_error, imod_temp_dir,
    print_pid, prnstr, py_float, py_int, py_regex, py_str_float, read_text_file, run_cmd,
    write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_in_out_file, pip_get_integer,
    pip_get_integer_array, pip_get_string, pip_get_two_integers, pip_number_of_entries,
    pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// One row of `typeTable` (`tomodataplots:61-93`): file type, columns,
/// ordinal flag, symbols, connect flag, keys.
type TypeRow = (
    usize,
    &'static [i64],
    bool,
    &'static [i64],
    i32,
    &'static [&'static str],
);

const TYPE_TABLE: [TypeRow; 22] = [
    (0, &[5], true, &[7], 1, &["X shift"]),
    (0, &[6], true, &[7], 1, &["Y shift"]),
    (0, &[5, 6], true, &[7, 9], 1, &["X shift", "Y shift"]),
    (1, &[1, 2], true, &[0, 0], 1, &["Mean error", "Max error"]),
    (2, &[1, 2], false, &[0], 0, &["Rotation"]),
    (2, &[1, 4, 7], false, &[0, 0], 1, &["Delta tilt", "Skew"]),
    (2, &[1, 5], false, &[0], 0, &["Mag"]),
    (2, &[1, 6], false, &[0], 0, &["X-Stretch (dmag)"]),
    (
        2,
        &[1, 2],
        false,
        &[9, 15],
        0,
        &["Mean residual", "(view is multiple of 5)"],
    ),
    (
        2,
        &[1, 2],
        false,
        &[9, 15],
        0,
        &["Local mean residual", "(view is multiple of 5)"],
    ),
    (3, &[1], true, &[7], 0, &["Minimum value"]),
    (3, &[2], true, &[7], 0, &["Maximum value"]),
    (
        3,
        &[1, 2],
        true,
        &[7, 9],
        0,
        &["Minimum value", "Maximum value"],
    ),
    (
        4,
        &[1, 2, 3, 4],
        false,
        &[5, 7, 9],
        0,
        &["X position", "Y position", "Z position"],
    ),
    (5, &[1, 2], false, &[9], 1, &["Defocus (microns)"]),
    (5, &[1, 2], false, &[9], 1, &["Astigmatism (um)"]),
    (5, &[1, 2], false, &[9], 1, &["Astig axis (deg)"]),
    (5, &[1, 2], false, &[9], 1, &["Phase shift (deg)"]),
    (5, &[1, 2], false, &[9], 1, &["Cut-on freq (1/nm)"]),
    (
        6,
        &[1, 2, 3],
        false,
        &[7, 9],
        1,
        &["Raw distance (pixels)", "Smoothed distance"],
    ),
    (
        6,
        &[1, 2, 3],
        false,
        &[7, 9],
        1,
        &[
            "Mean leave-out error (pixels)",
            "Mean weighted residual (pixels)",
        ],
    ),
    (6, &[1, 2], false, &[9], 1, &["Max of max weighted resids"]),
];

/// `tempNeeded` (`tomodataplots:95`).
const TEMP_NEEDED: [bool; 7] = [false, true, true, true, false, true, true];

/// `def findLimitingLines(lines, startText, endText, startLook = 0)`
/// (`tomodataplots:12`).
fn find_limiting_lines(
    lines: &[String],
    start_text: &str,
    end_text: &str,
    start_look: i64,
) -> (i64, i64) {
    let re_start = regex::Regex::new(&py_regex(start_text)).expect("start pattern");
    let re_end = regex::Regex::new(&py_regex(end_text)).expect("end pattern");
    let mut start_line: i64 = -1;
    for ind in start_look.max(0) as usize..lines.len() {
        if start_line < 0 && re_start.is_match(&lines[ind]) {
            start_line = ind as i64;
        }
        if start_line >= 0 && re_end.is_match(&lines[ind]) {
            return (start_line, ind as i64);
        }
    }
    (start_line, -1)
}

/// `def extractBinAndRad2(line)` (`tomodataplots:25`): extract binning and
/// filter value from a "Results" line of Alignframes log.
fn extract_bin_and_rad2(line: &str) -> (Option<String>, Option<String>) {
    let line = line.replace('=', " ");
    let lsplit: Vec<&str> = line.split_whitespace().collect();
    let mut bin = None;
    let mut rad2 = None;
    for ind in 0..lsplit.len().saturating_sub(1) {
        if lsplit[ind] == "bin" {
            bin = Some(lsplit[ind + 1].to_owned());
        }
        if lsplit[ind] == "rad2" {
            rad2 = Some(lsplit[ind + 1].to_owned());
        }
    }
    (bin, rad2)
}

/// A key of `setDict`: the string `'0'` or a `(bin, rad2)` tuple.
#[derive(Clone, Debug, PartialEq)]
enum SetKey {
    Zero,
    BinRad(Option<String>, Option<String>),
}

/// The module-level state of the alignframes analysis that
/// [`add_to_alignframes_arrays`] reads and updates.
struct AlignframesState {
    angles: Vec<Option<String>>,
    set_nums: Vec<String>,
    all_values: Vec<Vec<Option<String>>>,
    key: SetKey,
    set_dict: Vec<(SetKey, Vec<Option<String>>)>,
    best_bin: Option<String>,
    best_rad2: Option<String>,
    use_hybrid: bool,
    angle: Option<String>,
    set_num: String,
}

impl AlignframesState {
    /// `setDict[key] = values`
    fn store(&mut self, values: &[Option<String>]) {
        let key = self.key.clone();
        match self.set_dict.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = values.to_vec(),
            None => self.set_dict.push((key, values.to_vec())),
        }
    }
}

/// `def addToAlignframesArrays()` (`tomodataplots:41`): add the best set of
/// data to the array for plotting alignframes data.
fn add_to_alignframes_arrays(st: &mut AlignframesState) {
    // Use the original 0 key if it is still that, or use the best key
    if st.key != SetKey::Zero && st.best_bin.is_some() {
        if st.use_hybrid {
            st.key = SetKey::BinRad(st.best_bin.clone(), None);
        } else {
            st.key = SetKey::BinRad(st.best_bin.clone(), st.best_rad2.clone());
        }
    }

    if let Some((_, values)) = st.set_dict.iter().find(|(k, _)| *k == st.key) {
        let values = values.clone();
        st.angles.push(st.angle.clone());
        st.set_nums.push(st.set_num.clone());
        st.all_values.push(values);
    }
}

/// The script's top level (`tomodataplots:53-578`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn tomodataplots(arguments: &[OsString]) -> i32 {
    let progname = "tomodataplots";
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

    let mut temp_file = String::new();
    let default_colors = ["1,navy", "2,maroon", "3,darkgreen"];
    let mut axis_labels = [
        "View number",
        "View number",
        "View number",
        "View number",
        "Tilt Angle (degrees)",
        "Tilt Angle (degrees)",
        "Tilt Angle (degrees)",
    ]
    .map(str::to_owned);

    // Fallbacks from ../manpages/autodoc2man 3 1 tomodataplots
    let options: Vec<String> = [
        "input:InputFile:FN:",
        "type:TypeOfDataToPlot:IA:",
        "connect:ConnectWithLines:I:",
        "symbols:SymbolsForGroups:IA:",
        "hue:HueOfGroup:CHM:",
        "axis:XaxisLabel:CH:",
        "append:AppendToKey:CH:",
        "size:SizeOfPlot:IP:",
        "position:PositionOfPlot:IP:",
        "background:BackgroundProcess:B:",
        ":PID:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    // Get input file, make sure it exists
    let data_name = pip_get_in_out_file("InputFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if data_name.is_empty() {
        exit_error("The input file name must be entered");
    }
    if !Path::new(&data_name).exists() {
        exit_error(&format!("Input file {data_name} does not exist"));
    }

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    // Get type of file, and if it needs processing, read the lines and set up for temp file
    let data_type = pip_get_integer("TypeOfDataToPlot", 0).unwrap_or(0) as i64 - 1;
    if data_type < 0 || data_type >= TYPE_TABLE.len() as i64 {
        exit_error(&format!(
            "A type of data must be entered between 1 and {}",
            TYPE_TABLE.len()
        ));
    }
    let row = TYPE_TABLE[data_type as usize];
    let file_type = row.0;
    let mut plot_file = data_name.clone();
    let mut data_lines: Vec<String> = Vec::new();
    if TEMP_NEEDED[file_type] {
        temp_file = format!("{}/{progname}.{}", imod_temp_dir(), std::process::id());
        plot_file = temp_file.clone();
        data_lines = read_text_file(&data_name, None, false, None).unwrap_or_default();
    }

    // Get other options, fall back to defaults for connect
    let background = pip_get_boolean("BackgroundProcess", 0).unwrap_or(0);
    let mut connect = pip_get_integer("ConnectWithLines", 0).unwrap_or(0);
    connect = 0.max(1.min(connect));
    if pip_get_err_no() != 0 {
        connect = row.4;
    }
    let symbols_in: Vec<i64> = pip_get_integer_array("SymbolsForGroups", 0).unwrap_or_default();
    let num_colors = pip_number_of_entries("HueOfGroup").unwrap_or(0);
    let mut axis_label = pip_get_string("XaxisLabel", "").unwrap_or_default();
    let add_to_key = pip_get_string("AppendToKey", "").unwrap_or_default();

    let mut if_types = false;

    // Handle generic option setup into onegenplot
    let mut columns: Vec<i64> = row.1.to_vec();
    let mut symbols: Vec<i64> = row.3.to_vec();
    let mut keys: Vec<String> = row.5.iter().map(|key| (*key).to_owned()).collect();
    // `axisLabels[fileType]` is read here, before the alignframes branch can
    // change it
    if axis_label.is_empty() {
        axis_label = axis_labels[file_type].clone();
    }
    if file_type == 3 && !data_lines.is_empty() && data_lines[0].contains("iece") {
        axis_label = axis_label.replace("View number", "Piece number");
    }

    let mut comlines = vec![
        format!("InputDataFile {plot_file}"),
        format!("ConnectWithLines {connect}"),
        format!("XaxisLabel {axis_label}"),
    ];
    if row.2 {
        comlines.push("OrdinalsForXvalues".to_owned());
    }

    let index_error =
        || -> ! { super::pip::python_uncaught("IndexError: list index out of range") };

    // Handle specific file types
    // BLEND
    let mut out_lines: Vec<String> = Vec::new();
    if file_type == 1 {
        let err_match =
            regex::Regex::new(&py_regex("^.*mean&max.*after.*:")).expect("blend pattern");
        for l in &data_lines {
            if err_match.is_match(l) {
                out_lines.push(err_match.replace_all(l, "").into_owned());
            }
        }

        if out_lines.len() < 2 {
            symbols = vec![7, 9];
        }

    // ALIGN
    } else if file_type == 2 {
        let (start, mut end) = find_limiting_lines(&data_lines, "^ view.*deltilt", "^$", 0);
        if start < 0 || end < 0 {
            exit_error(&format!(
                "Could not find global solution table in {data_name}"
            ));
        }
        if !keys[0].contains("residual") {
            out_lines = data_lines[(start + 1) as usize..end as usize].to_vec();

            // Look for fixed value of second column
            if symbols.len() > 1 {
                let col = (columns[columns.len() - 1] - 1) as usize;
                let mut colval = String::new();
                let mut broke = false;
                for l in &out_lines {
                    let lsplit: Vec<&str> = l.split_whitespace().collect();
                    if lsplit.len() <= col {
                        exit_error(&format!(
                            "Not enough columns in solution table in {data_name}"
                        ));
                    }
                    if colval.is_empty() {
                        colval = lsplit[col].to_owned();
                    } else if colval != lsplit[col] {
                        broke = true;
                        break;
                    }
                }
                if !broke {
                    // ELSE ON FOR; peel off the last column
                    columns.pop();
                    symbols.pop();
                    keys.pop();
                }
            }
        } else {
            // Local or global mean residual: first find maximum view in global
            // `dataLines[end - 1]`: Python indexing, from the end when negative
            let last_index = if end >= 1 {
                end - 1
            } else {
                data_lines.len() as i64 - 1
            };
            let last_view = match data_lines
                .get(last_index.max(0) as usize)
                .and_then(|line| line.split_whitespace().next().and_then(py_int))
            {
                Some(value) => value,
                None => exit_error("Converting view number in align log"),
            };
            let mut err_sum = vec![0.0_f64; (last_view + 1).max(0) as usize];
            let mut num_in_sum = vec![0_i64; (last_view + 1).max(0) as usize];
            if_types = true;

            // Analyze each line, add residual to sum for view
            let mut start = start;
            loop {
                let mut local = "global";
                if keys[0].contains("Local mean") {
                    local = "local";
                    (start, end) = find_limiting_lines(&data_lines, "^ view.*deltilt", "^$", end);
                    if start < 0 || end < 0 {
                        break;
                    }
                }
                for l in &data_lines[(start + 1) as usize..end.max(start + 1) as usize] {
                    let lsplit: Vec<&str> = l.split_whitespace().collect();
                    let (Some(view), Some(resid)) = (
                        lsplit.first().and_then(|text| py_int(text)),
                        lsplit.get(7).and_then(|text| py_float(text)),
                    ) else {
                        exit_error(&format!("Analyzing {local} solution in {data_name}"))
                    };
                    if view > last_view {
                        exit_error("View number higher in local than global solution");
                    }
                    // A negative view counts from the end of the Python list
                    let at = if view < 0 {
                        view + err_sum.len() as i64
                    } else {
                        view
                    };
                    if at < 0 {
                        index_error();
                    }
                    err_sum[at as usize] += resid;
                    num_in_sum[at as usize] += 1;
                }

                if local != "local" {
                    break;
                }
            }

            // Make the output lines with means
            for view in 0..(last_view + 1).max(0) as usize {
                if num_in_sum[view] != 0 {
                    let mut group = 1;
                    if view % 5 == 0 {
                        group = 2;
                    }
                    out_lines.push(format!(
                        "{group}  {view}  {}",
                        py_str_float(err_sum[view] / num_in_sum[view] as f64)
                    ));
                }
            }
            if out_lines.is_empty() {
                exit_error(&format!("No local solutions found in {data_name}"));
            }
        }

    // CLIP STATS
    } else if file_type == 3 {
        let (start, end) = find_limiting_lines(&data_lines, "----", "all", 0);
        if start < 0 || end < 0 {
            exit_error(&format!(
                "Cannot find starting and ending lines for stats in {data_name}"
            ));
        }
        for l in &data_lines[(start + 1) as usize..end.max(start + 1) as usize] {
            let l = l.replace('*', " ");
            let lsplit: Vec<&str> = l.split_whitespace().collect();
            let parsplit: Vec<&str> = l.split(')').collect();
            let first = lsplit.get(1);
            let max = parsplit
                .get(1)
                .and_then(|part| part.split_whitespace().next());
            match (first, max) {
                (Some(min), Some(max)) => out_lines.push(format!("{min}  {max}")),
                _ => exit_error(&format!("Extracting min and max from lines in {data_name}")),
            }
        }
        if out_lines.is_empty() {
            exit_error(&format!("No min/max data found in {data_name}"));
        }

    // CTFPLOTTER
    } else if file_type == 5 {
        let Some(first_line) = data_lines.first() else {
            index_error()
        };
        let lsplit: Vec<&str> = first_line.split_whitespace().collect();
        let mut start_line = 0;
        let mut has_astig = false;
        let mut has_phase = false;
        let mut has_cuton = false;
        let mut angles: Vec<f64> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut min_axis = 1000.0_f64;
        let mut max_axis = -1000.0_f64;
        let mut min_abs_axis = 1000.0_f64;
        let mut num_plus = 0;
        let mut num_minus = 0;
        // `int()`/`float()` failures are ValueErrors the script's `except
        // IOError` does not catch: an uncaught exception
        let value_error = |text: &str| -> ! {
            super::pip::python_uncaught(&format!(
                "ValueError: could not convert string to number: '{text}'"
            ))
        };
        let int_at = |fields: &[&str], index: usize| -> i64 {
            match fields.get(index) {
                Some(text) => py_int(text).unwrap_or_else(|| value_error(text)),
                None => index_error(),
            }
        };
        let float_at = |fields: &[&str], index: usize| -> f64 {
            match fields.get(index) {
                Some(text) => py_float(text).unwrap_or_else(|| value_error(text)),
                None => index_error(),
            }
        };
        if lsplit.len() >= 6 {
            let version = int_at(&lsplit, 5);
            if version > 2 {
                start_line = 1;
                let flags = int_at(&lsplit, 0);
                has_astig = flags & 1 != 0;
                has_phase = flags & 4 != 0;
                has_cuton = flags & 32 != 0;
            }
        }
        if data_type > 17 {
            if !has_cuton {
                exit_error("There are no cut-on frequencies in this defocus file");
            }
        } else if data_type > 16 {
            if !has_phase {
                exit_error("There are no phase shift solutions in this defocus file");
            }
        } else if data_type > 14 && !has_astig {
            exit_error("There are no astigmatism solutions in this defocus file");
        }

        let mut phase_col = 5;
        let mut cuton_col = 6;
        if has_astig {
            phase_col = 7;
            cuton_col = 8;
        }

        for l in &data_lines[start_line.min(data_lines.len())..] {
            if l.trim().is_empty() {
                continue;
            }
            let lsplit: Vec<&str> = l.split_whitespace().collect();
            let angle = 0.5 * (float_at(&lsplit, 2) + float_at(&lsplit, 3));
            let value;
            if data_type < 17 {
                let mut defocus1 = float_at(&lsplit, 4) / 1000.;
                let mut chosen = defocus1;
                if has_astig {
                    let defocus2 = float_at(&lsplit, 5) / 1000.;
                    let axis = float_at(&lsplit, 6);
                    let mut astig = 0.0_f64;
                    if defocus2 != 0. {
                        astig = defocus1 - defocus2;
                        defocus1 = 0.5 * (defocus1 + defocus2);
                    }
                    if data_type == 14 {
                        chosen = defocus1;
                    } else {
                        if defocus2.abs() < 1.0e-6 {
                            continue;
                        }
                        if data_type == 15 {
                            chosen = astig;
                        } else {
                            chosen = axis;
                            // `min`/`max`: the first argument unless the
                            // second is smaller/larger
                            if min_axis < axis {
                            } else if axis < min_axis {
                                min_axis = axis;
                            }
                            if axis > max_axis {
                                max_axis = axis;
                            }
                            if axis.abs() < min_abs_axis {
                                min_abs_axis = axis.abs();
                            }
                            if axis >= 0. {
                                num_plus += 1;
                            } else {
                                num_minus += 1;
                            }
                        }
                    }
                }
                value = chosen;
            } else if data_type == 17 {
                value = float_at(&lsplit, phase_col);
            } else {
                value = float_at(&lsplit, cuton_col);
            }

            angles.push(angle);
            values.push(value);
        }

        // If the axis values are extreme on both sides of zero and none are near zero,
        // then adjust the minority sign values by 180
        if min_axis < -60. && max_axis > 60. && min_abs_axis > 45. {
            for value in values.iter_mut() {
                if *value < 0. && num_plus > num_minus {
                    *value += 180.;
                } else if *value > 0. && num_plus <= num_minus {
                    *value -= 180.;
                }
            }
        }

        for (angle, value) in angles.iter().zip(&values) {
            out_lines.push(format!("{} {}", py_str_float(*angle), py_str_float(*value)));
        }

    // ALIGNFRAMES
    } else if file_type == 6 {
        let mut st = AlignframesState {
            angles: Vec::new(),
            set_nums: Vec::new(),
            all_values: Vec::new(),
            key: SetKey::Zero,
            set_dict: Vec::new(),
            best_bin: None,
            best_rad2: None,
            use_hybrid: false,
            angle: None,
            set_num: String::new(),
        };
        let mut values: Vec<Option<String>> = vec![None; 5];
        let mut have_all_angles = true;
        let mut have_leave_out = false;
        let mut have_weighted = false;
        let mut got_set = false;
        for raw in &data_lines {
            // Convert , to space so splitting is easy
            let line = raw.replace(',', " ");

            // Look for valid option to use hybrid result
            if line.contains("UseHybrid") && !line.trim().starts_with('#') {
                let lsplit: Vec<&str> = line.split('=').collect();
                st.use_hybrid = lsplit.len() < 2
                    || match py_int(lsplit[1]) {
                        Some(value) => value != 0,
                        None => exit_error("Extracting values from Alignframes output file"),
                    };
            }

            // Set lines can be of 3 kinds
            if line.starts_with("Set ") || line.starts_with("File ") {
                // A line about the best binning/filter
                if line.contains(": Best") {
                    st.store(&values);
                    (st.best_bin, st.best_rad2) = extract_bin_and_rad2(&line);

                // A line NOT about dropped frames in FISE
                } else if !line.contains(": drop") {
                    if got_set {
                        // If there is a previous set, put current values in dictionary and
                        // add to data array
                        st.store(&values);
                        if st.angle.is_none() {
                            have_all_angles = false;
                        }
                        add_to_alignframes_arrays(&mut st);
                    }

                    // Initialize for a set, get the set number and hopefully the degrees
                    got_set = true;
                    st.set_dict = Vec::new();
                    st.key = SetKey::Zero;
                    st.best_bin = None;
                    values = vec![None; 5];
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    st.angle = None;
                    st.set_num = match lsplit.get(1) {
                        Some(text) => (*text).to_owned(),
                        None => index_error(),
                    };
                    if lsplit[lsplit.len() - 1].contains("deg") {
                        match lsplit.len().checked_sub(2) {
                            Some(index) => {
                                st.angle = Some(lsplit[index].replace('(', ""));
                            }
                            None => index_error(),
                        }
                    }
                }

            // For a "Results" line, process the previous data if any get new key
            } else if line.starts_with("Results with") || line.starts_with("Hybrid results") {
                st.store(&values);
                let (bin, rad2) = extract_bin_and_rad2(&line);
                st.key = SetKey::BinRad(bin, rad2);
                values = vec![None; 5];
            } else if got_set {
                // Once there is a set, process each line looking for the values
                let line = line.replace('=', " ");
                let lsplit: Vec<&str> = line.split_whitespace().collect();
                let last_ind = lsplit.len() as i64 - 1;
                let resid_line =
                    line.contains("esid") && line.contains("mean") && line.contains("max max");
                let lower = line.to_lowercase();
                if resid_line && (lower.contains(" wgtd") || lower.contains(" weighted")) {
                    have_weighted = true;
                }
                let next = |ind: usize| -> Option<String> {
                    match lsplit.get(ind + 1) {
                        Some(text) => Some((*text).to_owned()),
                        None => index_error(),
                    }
                };
                for ind in 0..lsplit.len() {
                    let i = ind as i64;
                    if resid_line {
                        if ind != 0
                            && lsplit[ind] == "mean"
                            && lsplit[ind - 1].contains("esid")
                            && i < last_ind
                        {
                            values[0] = next(ind);
                        }
                        if ind != 0
                            && lsplit[ind] == "max"
                            && lsplit[ind - 1] == "max"
                            && i < last_ind
                        {
                            values[1] = next(ind);
                        }
                        if lsplit[ind] == "l-o" && i < last_ind {
                            if i < last_ind - 1 && lsplit[ind + 1] == "err" {
                                values[4] = Some(lsplit[ind + 2].to_owned());
                                have_leave_out = true;
                            } else if lsplit[ind + 1] != "err" {
                                values[4] = Some(lsplit[ind + 1].to_owned());
                                have_leave_out = true;
                            }
                        }
                    }

                    if lsplit[ind] == "Dist" {
                        values[2] = next(ind);
                    }
                    if lsplit[ind].contains("smooth") || lsplit[ind].contains("smth") {
                        values[3] = next(ind);
                    }
                }
            }
        }

        // At end, add last set of data
        if got_set {
            st.store(&values);
            add_to_alignframes_arrays(&mut st);
        }

        // Load the data array whenever data exists for an angle
        let xvals: Vec<String> = if have_all_angles {
            st.angles
                .iter()
                .map(|angle| angle.clone().unwrap_or_default())
                .collect()
        } else {
            axis_labels[row.0] = "Set number".to_owned();
            st.set_nums.clone()
        };
        for ind in 0..xvals.len() {
            let v = &st.all_values[ind];
            if data_type == 19 && v[2].is_some() && v[3].is_some() {
                out_lines.push(format!(
                    "{} {} {}",
                    xvals[ind],
                    v[2].as_deref().unwrap_or_default(),
                    v[3].as_deref().unwrap_or_default()
                ));
            } else if data_type == 20 {
                if have_leave_out && v[0].is_some() && v[4].is_some() {
                    out_lines.push(format!(
                        "{} {} {}",
                        xvals[ind],
                        v[4].as_deref().unwrap_or_default(),
                        v[0].as_deref().unwrap_or_default()
                    ));
                } else if !have_leave_out && v[0].is_some() {
                    out_lines.push(format!(
                        "{} {}",
                        xvals[ind],
                        v[0].as_deref().unwrap_or_default()
                    ));
                }
            } else if data_type == 21 && v[1].is_some() {
                out_lines.push(format!(
                    "{} {}",
                    xvals[ind],
                    v[1].as_deref().unwrap_or_default()
                ));
            }
        }

        if data_type == 20 && !have_leave_out {
            columns = vec![1, 2];
            symbols = vec![symbols[1]];
            keys = vec![keys[1].clone()];
        }
        if data_type == 20 && !have_weighted {
            let last = keys.len() - 1;
            keys[last] = "Mean residual (pixels)".to_owned();
        }
    }

    if !temp_file.is_empty() {
        if out_lines.is_empty() {
            exit_error(&format!(
                "Did not find any lines of the selected data type in {data_name}"
            ));
        }
        let _ = write_text_file(&temp_file, &out_lines, false);
    }

    // Now that everything is set, make up columns and symbols input
    let mut colstr = "ColumnsToPlot ".to_owned();
    for (ind, column) in columns.iter().enumerate() {
        if ind != 0 {
            colstr.push(',');
        }
        colstr += &column.to_string();
    }
    comlines.push(colstr);
    if if_types {
        comlines.push("TypesToPlot 1,2".to_owned());
    }

    let mut symstr = "SymbolsForTypes ".to_owned();
    for (ind, symbol) in symbols.iter().enumerate() {
        let mut sym = *symbol;
        if !symbols_in.is_empty() && ind < symbols_in.len() {
            sym = symbols_in[ind];
        }
        if ind != 0 {
            symstr.push(',');
        }
        symstr += &sym.to_string();
    }
    comlines.push(symstr);

    // Append filename, time, or string to key
    if !add_to_key.is_empty() {
        let last = keys.len() - 1;
        if add_to_key == "@file" {
            let base = Path::new(&data_name)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            keys[last] += &format!(" {base}");
        } else if add_to_key == "@time" {
            keys[last] += &format!(" {}", chrono::Local::now().format("%H:%M:%S"));
        } else {
            keys[last] += &format!(" {add_to_key}");
        }
    }

    for key in &keys {
        comlines.push(format!("KeyLabels {key}"));
    }

    // Start with default colors, take in each entry and replace in default if present
    let mut colors: Vec<String> = if symbols.len() > 1 {
        default_colors
            .iter()
            .map(|color| (*color).to_owned())
            .collect()
    } else {
        Vec::new()
    };
    for _ in 0..num_colors {
        let new_color = pip_get_string("HueOfGroup", "").unwrap_or_default();
        let nc_first = new_color.split(',').next().unwrap_or_default().to_owned();
        match colors
            .iter()
            .position(|color| color.split(',').next().unwrap_or_default() == nc_first)
        {
            Some(index) => colors[index] = new_color,
            None => colors.push(new_color),
        }
    }
    for col in &colors {
        comlines.push(format!("HueOfGroup {col}"));
    }

    let (xsize, ysize) = pip_get_two_integers("SizeOfPlot", (0, 0)).unwrap_or((0, 0));
    if xsize > 0 && ysize > 0 {
        comlines.push(format!("SizeOfPlot {xsize},{ysize}"));
    }
    let (xpos, ypos) = pip_get_two_integers("PositionOfPlot", (0, 0)).unwrap_or((0, 0));
    if pip_get_err_no() == 0 {
        comlines.push(format!("PositionOfPlot {xpos},{ypos}"));
    }

    // Run onegenplot
    if background != 0 {
        // In the background: compose a command array and add the cleanup option
        let mut com_array: Vec<OsString> = vec!["onegenplot".into()];
        for line in &comlines {
            let mut lsplit = line.splitn(2, ' ');
            com_array.push(format!("-{}", lsplit.next().unwrap_or_default()).into());
            if let Some(rest) = lsplit.next() {
                com_array.push(rest.into());
            }
        }

        if let Err(error) = bkgd_process(&com_array, None, Some("stdout"), true, false) {
            exit_error(&format!("Cannot start onegenplot: {error}"));
        }
        return done(0);
    }

    // In foreground, run as usual command
    prnstr("Close graph window to exit", "\n", true);
    if run_cmd(
        "onegenplot -StandardInput",
        Some(&comlines),
        None,
        None,
        &[],
    )
    .is_err()
    {
        if !temp_file.is_empty() {
            cleanup_files(&[temp_file.clone()]);
        }
        exit_from_imod_error(progname);
    }

    if !temp_file.is_empty() {
        cleanup_files(&[temp_file]);
    }
    done(0)
}
