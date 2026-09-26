//! Translation of `IMOD/pysrc/alignlog`.
//!
//! A Python command script: its three functions are [`pyawk`], [`pygrep`] and
//! [`separator`], and its top level is [`alignlog`], translated statement by
//! statement.  The module global `output` is [`OUTPUT`].

use super::imodpy::{add_imod_bin_ignore_sighup, prnstr, read_text_file};
use super::pip::{
    exit_error, pip_enable_entry_output, pip_get_boolean, pip_get_non_option_arg,
    pip_read_or_parse_options,
};
use regex::Regex;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};

/// The module global `output` (`alignlog:11`).
static OUTPUT: AtomicI32 = AtomicI32::new(0);

/// Matches `pyawk` (`alignlog:14`): like doing awk from starting to ending line.
///
/// The source's defaults are `excludeEnd = False`, `skipMatch = None`,
/// `skipEmpty = False`, `separator = False`.
pub fn pyawk(
    lines: &[String],
    start: &str,
    end: &str,
    exclude_end: bool,
    skip_match: Option<&str>,
    skip_empty: bool,
    separator: bool,
) {
    let re_start = Regex::new(start).unwrap();
    let re_end = Regex::new(end).unwrap();
    let mut got_start = false;
    let mut got_any = false;
    for l in lines {
        if !got_start && re_start.is_match(l) {
            got_start = true;
            if separator && got_any {
                prnstr(" ", "\n", false);
            }
            got_any = true;
        }
        let mut putout = got_start;
        if got_start && re_end.is_match(l) {
            got_start = false;
            putout = !exclude_end;
        }

        if putout {
            if (!skip_empty || !l.trim_end().is_empty())
                && (skip_match.is_none_or(|skip| !l.contains(skip)))
            {
                prnstr(l.trim_end(), "\n", false);
            }
        }
    }
}

/// Matches `pygrep` (`alignlog:37`): like doing grep on a single simple string.
pub fn pygrep(lines: &[String], string: &str, skip_match: Option<&str>) {
    let parts = string.split('|').collect::<Vec<_>>();
    for l in lines {
        for strn in &parts {
            if l.contains(strn) && skip_match.is_none_or(|skip| !l.contains(skip)) {
                prnstr(l.trim_end(), "\n", false);
                break;
            }
        }
    }
}

/// Matches `separator` (`alignlog:47`): put out a separator if there has been
/// output already.
pub fn separator() {
    if OUTPUT.load(Ordering::SeqCst) != 0 {
        prnstr(" ", "\n", false);
        prnstr(
            "* * * * * * * * * * * * * * * * * * * * * * * * * * * *",
            "\n",
            false,
        );
        prnstr(" ", "\n", false);
    }
    OUTPUT.store(1, Ordering::SeqCst);
}

/// The script's top level (`alignlog:57-289`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn alignlog(arguments: &[OsString]) -> i32 {
    let progname = "alignlog";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 alignlog
    let options: Vec<String> = [
        ":m:B:", ":e:B:", ":s:B:", ":l:B:", ":c:B:", ":r:B:", ":a:B:", ":b:B:", ":w:B:", ":p:B",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    pip_enable_entry_output(0);
    let (num_opts, num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    let mut infile = "align.log".to_owned();
    if num_non_opts != 0 {
        infile = pip_get_non_option_arg(0).unwrap_or_default();
    }

    if infile == "a" || infile == "b" {
        infile = format!("align{infile}.log");
    }
    if !Path::new(&infile).exists() {
        exit_error(&format!("Alignment log file {infile} does not exist"));
    }
    let lines = read_text_file(&infile, None, false, None).unwrap_or_default();

    if num_opts == 0 {
        return 0;
    }

    // Get options that are needed for
    let error = pip_get_boolean("e", 0).unwrap_or(0) != 0;
    let solution = pip_get_boolean("s", 0).unwrap_or(0) != 0;
    let angle = pip_get_boolean("a", 0).unwrap_or(0) != 0;
    let weight = pip_get_boolean("w", 0).unwrap_or(0) != 0;
    let project = pip_get_boolean("p", 0).unwrap_or(0) != 0;

    // initialize variables as 1's and set up for searches
    // Sorry, replicating grep output status with these variables
    let mut newsurface = 1; // for angle
    let mut pipinput = 1; // for angle
    let mut newratios = 1; // for error
    let mut oneratio = 1; // for error
    let mut nobeamtilt = 1; // for solution
    let mut noprogentry = 1; // for angle
    let mut nolocals = 1; // for angle
    let mut no_cross_val = 1; // for error or weight
    let mut proj_skew_line = String::new();
    let mut rot_at_min_tilt_line = String::new();
    let mut no_cpp_version = 1;

    let total_rat = Regex::new("Ratio .*to total unknown").unwrap();
    let former_rat = Regex::new("Ratio .*formerly").unwrap();

    // find values of needed variables
    for l in &lines {
        if angle && newsurface != 0 && l.contains("SURFACE ANALYSIS") {
            newsurface = 0;
        }
        if angle && pipinput != 0 && l.contains("to do series") {
            pipinput = 0;
        }
        if error && newratios != 0 && total_rat.is_match(l) {
            newratios = 0;
        }
        if error && oneratio != 0 && former_rat.is_match(l) {
            oneratio = 0;
        }
        if solution && nobeamtilt != 0 && l.contains("olved beam tilt") {
            nobeamtilt = 0;
        }
        if angle && noprogentry != 0 && l.contains("ntries to program") {
            noprogentry = 0;
        }
        if (angle || weight || error || project) && nolocals != 0 && l.contains("Doing local area")
        {
            nolocals = 0;
        }
        if (angle || error || weight) && no_cross_val != 0 && l.contains("leave-out") {
            no_cross_val = 0;
        }
        if solution && l.contains("Projection skew is") {
            proj_skew_line = l.trim().to_owned();
        }
        if solution && l.contains("minimum tilt, rotation") {
            rot_at_min_tilt_line = l.trim().to_owned();
        }
        if no_cpp_version != 0 && l.contains("Opened NEW file") {
            no_cpp_version = 0;
        }
    }

    // Go through the arguments
    for optind in 1..(num_opts + 1) as usize {
        let opt = argv[optind].as_str();
        if opt == "-m" {
            separator();
            pyawk(
                &lines,
                "^ *Variable mappings",
                "^$",
                false,
                None,
                false,
                false,
            );
        }

        if opt == "-e" {
            separator();
            if newratios == 0 {
                pyawk(
                    &lines,
                    "^ *Final   F",
                    "^  Ratio",
                    false,
                    None,
                    false,
                    false,
                );
            } else if oneratio != 0 {
                pyawk(
                    &lines,
                    "^.*Final   F",
                    "^  Ratio of",
                    false,
                    Some("weight"),
                    true,
                    true,
                );
            } else {
                pyawk(
                    &lines,
                    "^.*Final   F",
                    "^  Ratio to",
                    false,
                    None,
                    false,
                    false,
                );
            }

            if no_cross_val != 0 {
                pygrep(&lines, "Residual error|Ratio of local", Some("weight"));
            } else {
                for l in &lines {
                    let mut l = l.as_str();
                    if (l.contains("Residual error") && !l.contains("weighted"))
                        || l.contains("Ratio of local")
                    {
                        if l.contains("Global") || l.contains("Ratio of local") {
                            prnstr(" ", "\n", false);
                        }
                        prnstr(l.trim_end(), "\n", false);
                    }
                    if l.contains("leave-out error") && !l.contains("   robust") {
                        if let Some(ind) = l.find("weighted").filter(|ind| *ind > 0) {
                            l = &l[..ind];
                        }
                        prnstr(l.trim_end(), "\n", false);
                        if nolocals == 0 && l.contains("Global") {
                            prnstr(" ", "\n", false);
                        }
                    }
                }
            }
        }

        if opt == "-s" {
            separator();
            if nobeamtilt == 0 {
                pygrep(&lines, "Beam tilt angle is", None);
                prnstr(" ", "\n", false);
            }
            if !proj_skew_line.is_empty() {
                prnstr(&proj_skew_line, "\n", false);
                prnstr(" ", "\n", false);
            }
            if !rot_at_min_tilt_line.is_empty() {
                prnstr(&rot_at_min_tilt_line, "\n", false);
                prnstr(" ", "\n", false);
            }
            pyawk(&lines, "^ view.*deltilt", "^$", false, None, false, false);
        }

        if opt == "-l" {
            separator();
            for l in &lines {
                if l.contains("Doing local area")
                    || l.contains("on bottom and")
                    || (l.contains("Residual error") && !l.contains("weighted"))
                {
                    prnstr(l.trim_end(), "\n", false);
                    if l.contains("Residual error mean") {
                        prnstr(" ", "\n", false);
                    }
                }
            }
        }

        if opt == "-c" {
            separator();
            pyawk(
                &lines,
                "^ *3-D point",
                "^ Midpoint",
                true,
                None,
                false,
                false,
            );
        }

        if opt == "-r" {
            separator();
            pyawk(
                &lines,
                "^ *Projection points",
                "^$",
                false,
                None,
                false,
                false,
            );
        }

        if opt == "-a" {
            separator();
            if newsurface == 1 {
                pyawk(
                    &lines,
                    "^ Fit to all",
                    "^ 1 to do",
                    true,
                    None,
                    false,
                    false,
                );
            } else if pipinput == 0 {
                pyawk(
                    &lines,
                    "^ SURFACE ANALYSIS",
                    "^ 1 to do",
                    true,
                    None,
                    false,
                    false,
                );
            } else if noprogentry == 1 || nolocals == 0 || no_cross_val == 0 {
                if no_cross_val == 0 {
                    pyawk(
                        &lines,
                        "^ SURFACE ANALYSIS",
                        "Running ",
                        true,
                        None,
                        false,
                        false,
                    );
                } else if no_cpp_version == 0 {
                    pyawk(
                        &lines,
                        "^ SURFACE ANALYSIS",
                        "Opened NEW file",
                        true,
                        None,
                        false,
                        false,
                    );
                } else {
                    pyawk(
                        &lines,
                        "^ SURFACE ANALYSIS",
                        "file opened",
                        true,
                        None,
                        false,
                        false,
                    );
                }
            } else {
                pyawk(
                    &lines,
                    "^ SURFACE ANALYSIS",
                    "ntries to program",
                    true,
                    None,
                    false,
                    false,
                );
            }
        }

        if opt == "-b" {
            separator();
            pygrep(&lines, " beam tilt =", None);
        }

        if opt == "-w" {
            separator();
            pyawk(
                &lines,
                "Starting robust",
                "are < .5",
                false,
                None,
                false,
                true,
            );
            if nolocals == 0 {
                prnstr(" ", "\n", false);
                pyawk(
                    &lines,
                    "Summary of robust",
                    "are < .5",
                    false,
                    None,
                    false,
                    false,
                );
                prnstr(" ", "\n", false);
            }

            if no_cross_val != 0 {
                pygrep(&lines, "Residual error weighted", None);
                if nolocals == 0 {
                    pygrep(&lines, "Weighted error ", None);
                }
            } else {
                let mut num_ben = 0;
                for l in &lines {
                    if l.contains("Residual error weighted") {
                        prnstr(l.trim_end(), "\n", false);
                    }
                    if l.contains("Weighted error local") {
                        prnstr(" ", "\n", false);
                        prnstr(l.trim_end(), "\n", false);
                    }
                    if l.contains("robust leave-out error") || l.contains("Benefit") {
                        prnstr(l.trim_end(), "\n", false);
                        if l.contains("Benefit") && num_ben == 0 && nolocals == 0 {
                            prnstr(" ", "\n", false);
                            num_ben = 1;
                        }
                    }
                }
            }
        }

        if opt == "-p" {
            separator();
            let mut ratio_line = String::new();
            let mut ratio_match = "Ratio of total";
            let mut cv_match = "Global";
            if nolocals == 0 {
                ratio_match = "Ratio of local";
                cv_match = "Local";
            }
            let mut cv_lines: Vec<String> = Vec::new();
            let mut err_lines: Vec<String> = Vec::new();
            for l in &lines {
                if ratio_line.is_empty() && l.contains(ratio_match) {
                    ratio_line = l.clone();
                }
                if l.contains(cv_match) && l.contains("leave-out") {
                    cv_lines.push(l.clone());
                }
                if !cv_lines.is_empty() && l.contains("Benefit from") {
                    cv_lines.push(l.clone());
                }
                if l.contains("Residual error") && !l.contains("Local area") {
                    err_lines.push(l.clone());
                }
                if nolocals == 0 && l.contains("Weighted error") {
                    err_lines.push(l.clone());
                }
            }

            let mut out_lines = vec![ratio_line];
            out_lines.extend(err_lines);
            out_lines.extend(cv_lines);
            for l in &out_lines {
                prnstr(l.trim_end(), "\n", false);
            }
        }
    }

    0
}
