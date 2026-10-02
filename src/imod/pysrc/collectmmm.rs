//! Translation of `IMOD/pysrc/collectmmm`.
//!
//! A Python command script with no functions; its top level is
//! [`collectmmm`], translated statement by statement.  Values are Python
//! floats (doubles) and are formatted with `str` ([`py_str_float`]).

use super::imodpy::{
    add_imod_bin_ignore_sighup, exit_from_imod_error, prnstr, read_text_file, run_cmd,
};
use super::imodpy::{py_float, py_int, py_str_float};
use super::pip::{exit_error, set_exit_prefix};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`collectmmm:1-106`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn collectmmm(arguments: &[OsString]) -> i32 {
    let progname = "collectmmm";
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

    if argv.len() < 5 {
        prnstr(
            "
Usage: collectmmm tag rootname #_of_logs image_file [starting_#] [logdir]
   tag  =  a unique text string prefix to the min, max, mean, and pixel #
   rootname = root name of numbered log files, rootname-nnn.log
   #_of_logs = number of log files
   image_file = image file to correct the header of\x20
   starting_# = optional entry of number of first file, default is 1
   log_dir = path from working directory to directory where logs are",
            "\n",
            true,
        );
        return 1;
    }

    set_exit_prefix(prefix);
    let tag = &argv[1];
    let root = &argv[2];
    let imfile = &argv[4];
    let mut log_dir = ".".to_owned();
    // `int()` strips surrounding whitespace before converting
    let numlogs: i64;
    let mut startnum: i64 = 1;
    match py_int(&argv[3]) {
        Some(value) => numlogs = value,
        None => exit_error("Converting number of logs or number of first file to an integer"),
    }
    if argv.len() > 5 {
        match py_int(&argv[5]) {
            Some(value) => startnum = value,
            None => exit_error("Converting number of logs or number of first file to an integer"),
        }
    }
    if argv.len() > 6 {
        log_dir = argv[6].clone();
    }

    let mut allmin: f64 = 1.0e37;
    let mut allmax = -allmin;
    let mut allsum: f64 = 0.;
    let mut pixsum: f64 = 0.;
    for num in 0..numlogs.max(0) {
        let numrec = num + startnum;
        let numtext = format!("{numrec:03}");
        let thislog = format!("{log_dir}/{root}-{numtext}.log");
        let loglines = read_text_file(&thislog, None, false, None).unwrap_or_default();

        // get the line from the log file and insist it has 4 entries
        let mut vsplit: Option<Vec<String>> = None;
        for l in &loglines {
            if let Some(index) = l.find(tag.as_str()) {
                let valstr = &l[index + tag.len()..];
                let split: Vec<String> = valstr.split_whitespace().map(str::to_owned).collect();
                if split.len() != 4 {
                    exit_error(&format!(
                        "{thislog} does not contain 4 values after tag: {valstr}"
                    ));
                }
                vsplit = Some(split);
                break;
            }
        }
        let Some(vsplit) = vsplit else {
            exit_error(&format!("{thislog} does not contain a line with \"{tag}\""));
        };

        // Convert values to float
        let mut logmmm: Vec<f64> = Vec::new();
        for v in &vsplit {
            match py_float(v) {
                Some(value) => logmmm.push(value),
                None => exit_error(&format!("Converting {v} in {thislog} to a numeric value")),
            }
        }

        // Maintain min/max and sums: Python's `min`/`max` keep the first
        // argument unless the second compares less/greater
        if logmmm[0] < allmin {
            allmin = logmmm[0];
        }
        if logmmm[1] > allmax {
            allmax = logmmm[1];
        }
        allsum += logmmm[2] * logmmm[3];
        pixsum += logmmm[3];
    }

    // Compute final mean and set into header.
    // Fixed in translation (BUGS.md): a zero pixel sum (no logs, or zero
    // pixels) is an uncaught ZeroDivisionError in native (`collectmmm:98`);
    // there is nothing to set, so it is an ordinary error, exit 1.
    if pixsum == 0. {
        exit_error("The log files contain no pixels to compute a mean from");
    }
    let mean = allsum / pixsum;
    let altlines = vec![
        imfile.clone(),
        "setmmm".to_owned(),
        format!(
            "{},{},{}",
            py_str_float(allmin),
            py_str_float(allmax),
            py_str_float(mean)
        ),
        "done".to_owned(),
    ];
    if run_cmd("alterheader", Some(&altlines), Some("stdout"), None, &[]).is_err() {
        exit_from_imod_error(progname);
    }

    let _ = std::io::stdout().flush();
    0
}
