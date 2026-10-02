//! Translation of `IMOD/pysrc/findsirtdiffs`: collect difference statistics
//! from SIRT log files and combine them.
//!
//! A Python command script; `diffKey` is the sort key and the top level is
//! [`findsirtdiffs`], translated statement by statement.  Values are Python
//! ints and floats (doubles); `'{:15.3f}'` is C's `%15.3f`.

use super::imodpy::{add_imod_bin_ignore_sighup, glob_glob, prnstr, read_text_file};
use super::pip::{exit_error, set_exit_prefix};
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use std::ffi::OsString;
use std::io::Write as _;

/// One stored line: `[iteration, start, end, mean, sd]`.
type Diff = (i64, i64, i64, f64, f64);

/// Matches `diffKey` (`findsirtdiffs:11`).
fn diff_key(item: &Diff) -> i64 {
    item.0
}

/// The script's top level (`findsirtdiffs:14-91`).  Returns the status of
/// its final `sys.exit(0)`; error paths exit the process as `exitError`
/// does.
pub fn findsirtdiffs(arguments: &[OsString]) -> i32 {
    let progname = "findsirtdiffs";
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

    set_exit_prefix(prefix);
    if argv.len() < 2 {
        exit_error("Root name of log files must be entered");
    }

    // Read the log files, find the lines, and extract and store the numbers
    let logfiles = glob_glob(&format!("{}-[0-9]*.log", argv[1]));
    let mut diffs: Vec<Diff> = Vec::new();
    let mut lspl: Vec<String> = Vec::new();
    for log in &logfiles {
        let lines = read_text_file(log, None, false, None).unwrap_or_default();
        for l in &lines {
            if l.contains("diff rec mean") {
                lspl = l.split_whitespace().map(str::to_owned).collect();
                let int = |text: &str| super::imodpy::py_int(text);
                let float = |text: &str| super::imodpy::py_float(text);
                let parsed = (|| {
                    Some((
                        int(lspl.get(1)?.trim_end_matches(','))?,
                        int(lspl.get(3)?)?,
                        int(lspl.get(4)?.trim_end_matches(','))?,
                        float(lspl.get(8)?)?,
                        float(lspl.get(9)?)?,
                    ))
                })();
                match parsed {
                    Some(diff) => diffs.push(diff),
                    None => exit_error("Converting a line of difference statistics"),
                }
            }
        }
    }

    // Sort by iteration (`list.sort` is stable)
    diffs.sort_by_key(diff_key);

    let mut ind = 0usize;
    while ind < diffs.len() {
        // Loop through data, set up variables based on one entry for this iteration
        let mut start = diffs[ind].1;
        let mut end = diffs[ind].2;
        let mut mean = diffs[ind].3;
        let mut sd = diffs[ind].4;
        let itern = diffs[ind].0;
        let mut jlast = ind;
        if ind < diffs.len() - 1 && itern == diffs[ind + 1].0 {
            // But if there are multiple iterations, accumulate sums and get the overall
            // values
            let mut dsum: f64 = 0.;
            let mut dsumsq: f64 = 0.;
            let mut numsum: i64 = 0;
            for jnd in ind..diffs.len() {
                if diffs[jnd].0 != itern {
                    break;
                }
                let num = diffs[jnd].2 + 1 - diffs[jnd].1;
                numsum += num;
                dsum += num as f64 * diffs[jnd].3;
                dsumsq += (num - 1) as f64 * diffs[jnd].4 * diffs[jnd].4;
                start = start.min(diffs[jnd].1);
                end = end.max(diffs[jnd].2);
                jlast = jnd;
            }

            mean = dsum / numsum as f64;
            sd = (dsumsq / (numsum - 1) as f64).sqrt();
        }

        ind = jlast + 1;
        prnstr(
            &format!(
                "{} {:3}, {}{:6}{:6}, {} {} {}{}",
                lspl[0],
                itern,
                lspl[2],
                start,
                end,
                lspl[5],
                lspl[6],
                lspl[7],
                c_format("%15.3f%15.3f", &[CArg::Dbl(mean), CArg::Dbl(sd)])
            ),
            "\n",
            false,
        );
    }

    let _ = std::io::stdout().flush();
    0
}
