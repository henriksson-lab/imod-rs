//! Translation of `IMOD/pysrc/tomocleanup`.
//!
//! A Python command script: its two functions are
//! [`find_files_add_to_list`] and [`warning`], and its top level is
//! [`tomocleanup`], translated statement by statement.  The module global
//! `removeList` is passed to [`find_files_add_to_list`] explicitly.

use super::imodpy::{
    OptionValue, add_imod_bin_ignore_sighup, cleanup_files, dataset_filename,
    find_root_axis_and_extensions, glob_glob, option_value, prnstr, read_text_file,
    set_root_and_extension, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_integer, pip_get_non_option_arg, pip_get_string,
    pip_number_of_entries, pip_print_help, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// Matches `findFilesAddToList` (`tomocleanup:13`): glob the given patterm and
/// add to removal list.
pub fn find_files_add_to_list(remove_list: &mut Vec<String>, pattern: &str) {
    let found = glob_glob(pattern);
    if !found.is_empty() {
        remove_list.extend(found);
    }
}

/// Matches `warning` (`tomocleanup:21`): print a warning.
pub fn warning(message: &str) {
    prnstr(&format!("WARNING: tomocleanup - {message}"), "\n", false);
}

/// The script's top level (`tomocleanup:25-310`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn tomocleanup(arguments: &[OsString]) -> i32 {
    let progname = "tomocleanup";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 tomocleanup
    let options: Vec<String> = [
        "dir:Directory:FNM:",
        "aligned:KeepAlignedStack:B:",
        "untrimmed:KeepUntrimmedRec:B:",
        "axis:KeepAxisRecs:B:",
        "sirt:KeepSIRTRecs:B:",
        "filter:KeepFilterTrials:B:",
        "trial:TrialRun:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (num_opts, num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);

    if num_opts + num_non_opts == 0 || pip_get_boolean("help", 0).unwrap_or(0) != 0 {
        pip_print_help(progname, 0, 0, 0);
        return 0;
    }

    let keep_ali = pip_get_boolean("KeepAlignedStack", 0).unwrap_or(0);
    let keep_untrim = pip_get_boolean("KeepUntrimmedRec", 0).unwrap_or(0);
    let keep_axes = pip_get_boolean("KeepAxisRecs", 0).unwrap_or(0);
    let keep_sirt = pip_get_boolean("KeepSIRTRecs", 0).unwrap_or(0);
    let keep_filter = pip_get_boolean("KeepFilterTrials", 0).unwrap_or(0);
    let trial_run = pip_get_integer("TrialRun", 0).unwrap_or(0);
    let num_dir_entry = pip_number_of_entries("Directory").unwrap_or(0);
    let mut dir_list: Vec<String> = Vec::new();
    if num_dir_entry != 0 {
        for _ind in 0..num_dir_entry {
            dir_list.push(pip_get_string("Directory", "").unwrap_or_default());
        }
    }

    if num_non_opts != 0 {
        for ind in 0..num_non_opts {
            dir_list.push(pip_get_non_option_arg(ind).unwrap_or_default());
        }
    }

    if dir_list.is_empty() {
        exit_error(
            &("You must enter at least one directory name; use \".\" for the current ".to_owned()
                + "directory"),
        );
    }

    let cur_dir = std::env::current_dir().unwrap_or_default();
    for tdir in &dir_list {
        if !Path::new(tdir).is_dir() {
            exit_error(&format!("{tdir} is not a directory"));
        }
        // `os.access(tdir, os.W_OK)`: the POSIX `access` call itself, which asks
        // with the real uid and gid, not a reading of the permission bits
        let writable = std::ffi::CString::new(tdir.as_bytes())
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
            .unwrap_or(false);
        if !writable {
            exit_error(&format!(
                "You do not have permission to remove files in the directory {tdir}"
            ));
        }
    }

    for tdir in &dir_list {
        let _ = std::env::set_current_dir(&cur_dir);
        let _ = std::env::set_current_dir(tdir);
        let mut axis_type: Option<String> = None;
        let etomo_files = glob_glob("*.edf");
        let mut etomo_file = String::new();
        let mut type_ext: Option<String> = None;
        let mut stack_ext: Option<String> = None;
        let mut setname: Option<String> = None;
        if etomo_files.len() > 1 {
            warning(&format!(
                "There is more than one .edf file in {tdir}; skipping that directory"
            ));
            continue;
        }
        if !etomo_files.is_empty() {
            etomo_file = etomo_files[0].clone();
            let etomo_lines = match read_text_file(&etomo_file, None, true, None) {
                Ok(lines) => lines,
                Err(message) => {
                    warning(&format!("{message}; skipping directory {tdir}"));
                    continue;
                }
            };
            setname = match option_value(
                &etomo_lines,
                "Setup.DatasetName",
                0,
                false,
                0,
                Some('='),
                None,
            ) {
                Some(OptionValue::String(value)) => Some(value),
                _ => None,
            };
            if setname.as_deref().is_none_or(str::is_empty) {
                warning(&format!(
                    "Cannot find dataset name in {etomo_file} in directory {tdir}; falling back to analyzing command files"
                ));
            } else {
                axis_type = match option_value(
                    &etomo_lines,
                    "Setup.AxisType",
                    0,
                    false,
                    0,
                    Some('='),
                    None,
                ) {
                    Some(OptionValue::String(value)) => Some(value),
                    _ => None,
                };
                if axis_type.as_deref().is_none_or(str::is_empty) {
                    warning(&format!(
                        "Cannot find axis type in {etomo_file} in directory {tdir}; falling back to analyzing command files"
                    ));
                }

                type_ext = Some(String::new());
                let etomo_style = match option_value(
                    &etomo_lines,
                    "Setup.ImageFile.ImageFilenameStyle",
                    0,
                    false,
                    0,
                    Some('='),
                    None,
                ) {
                    Some(OptionValue::String(value)) => Some(value),
                    _ => None,
                };
                if etomo_style.as_deref() == Some("MRC") {
                    type_ext = Some("mrc".to_owned());
                }
                if etomo_style.as_deref() == Some("HDF") {
                    type_ext = Some("hdf".to_owned());
                }

                stack_ext = Some("st".to_owned());
                let etomo_ext = match option_value(
                    &etomo_lines,
                    "Setup.Setup.OrigImageStackExt",
                    0,
                    false,
                    0,
                    Some('='),
                    None,
                ) {
                    Some(OptionValue::String(value)) => Some(value),
                    _ => None,
                };
                if let Some(ext) = etomo_ext.filter(|ext| !ext.is_empty()) {
                    stack_ext = Some(ext);
                }
            }
        }

        if axis_type.as_deref() == Some("Not Set") {
            axis_type = None;
        }

        if axis_type.as_deref().is_none_or(str::is_empty) {
            if etomo_file.is_empty() {
                warning(&format!(
                    "No .edf file found in {tdir}; falling back to analyzing command files"
                ));
            }

            let (_com_ext, dual_num, root, type_ext_found, stack_ext_found) =
                find_root_axis_and_extensions(0, None);
            if root.is_empty() {
                warning(&format!(
                    "Cannot find data set files in {tdir}; skipping that directory"
                ));
                continue;
            }
            if !etomo_file.is_empty()
                && setname.as_deref().is_some_and(|name| !name.is_empty())
                && setname.as_deref() != Some(root.as_str())
            {
                warning(&format!(
                    "The data set name from {etomo_file} conflicts with that found from data set files in {tdir}; skipping that directory"
                ));
                continue;
            }

            setname = Some(root);

            if dual_num == 2 {
                axis_type = Some("Dual Axis".to_owned());
            } else if dual_num >= 0 {
                axis_type = Some("Single Axis".to_owned());
            } else {
                warning(&format!(
                    "Cannot determine axis type from data set files in {tdir}; skipping that directory"
                ));
                continue;
            }

            match type_ext_found {
                None => {
                    if type_ext.is_none() {
                        warning(&format!(
                            "Cannot determine file name style from edf file or data set files in {tdir}; skipping that directory"
                        ));
                        continue;
                    }
                }
                Some(found) => {
                    if type_ext.is_some() && type_ext.as_deref() != Some(found.as_str()) {
                        warning(&format!(
                            "The file name style from {etomo_file} conflicts with the style found from data set files in {tdir}; skipping that directory"
                        ));
                        continue;
                    } else {
                        type_ext = Some(found);
                    }
                }
            }

            // `stackExtFound` is never None: findRootAxisAndExtensions returns '' for
            // an undetermined raw stack extension
            let stack_ext_found = stack_ext_found;
            if stack_ext.is_some() && stack_ext.as_deref() != Some(stack_ext_found.as_str()) {
                warning(&format!(
                    "The raw stack extension from {etomo_file} conflicts with that found from data set files in {tdir}; skipping that directory"
                ));
                continue;
            } else {
                stack_ext = Some(stack_ext_found);
            }
        }

        // Information is now adequate one way or the other, set up name style
        let setname = setname.unwrap_or_default();
        let type_ext = type_ext.unwrap_or_default();
        let stack_ext = stack_ext.unwrap_or_default();
        set_root_and_extension(&setname, &type_ext);
        let mut remove_list: Vec<String> = Vec::new();
        find_files_add_to_list(&mut remove_list, "*~");

        let trimmed = dataset_filename(".rec", None, None);
        let mut num_axes = 1;
        let mut untrimmed = dataset_filename("_full.rec", None, None);

        // If dual-axis, add sum.rec files
        if axis_type.as_deref() == Some("Dual Axis") {
            num_axes = 2;
            untrimmed = dataset_filename(".rec", Some("sum"), None);
            find_files_add_to_list(
                &mut remove_list,
                &dataset_filename("[0-9]*.rec", Some("sum"), None),
            );
        }

        // Remove untrimmed if not keeping and trimmed exists
        if keep_untrim == 0 && Path::new(&trimmed).exists() {
            find_files_add_to_list(&mut remove_list, &untrimmed);
        }

        // Loop on axes
        for axis in 0..num_axes {
            let mut setlet = "";
            if num_axes > 1 {
                setlet = "a";
                if axis != 0 {
                    setlet = "b";
                }
            }
            let recext = format!("{setlet}.rec");

            // Remove single axis file if not keeping and combine or final trim exists
            if num_axes > 1
                && keep_axes == 0
                && (Path::new(&trimmed).exists() || Path::new(&untrimmed).exists())
            {
                find_files_add_to_list(&mut remove_list, &dataset_filename(&recext, None, None));
            }

            // Sample files
            for base in ["mid", "top", "bot"] {
                find_files_add_to_list(
                    &mut remove_list,
                    &dataset_filename(&recext, Some(base), None),
                );
            }

            // Unused files from various steps
            find_files_add_to_list(
                &mut remove_list,
                &format!("{setname}{setlet}_fixed.{stack_ext}"),
            );
            for pref in ["_filt", "_ctfcorr", "_erase"] {
                find_files_add_to_list(&mut remove_list, &format!("{setname}{setlet}{pref}.ali"));
            }

            // Basic simple dataset named files
            let mut ext_list = vec![
                "bl",
                "preali",
                "dcst",
                "alilog10",
                "_sub.ali",
                "_sub.alilog10",
                "_3dfind.rec",
            ];
            if keep_ali == 0 {
                ext_list.push("ali");
            }
            for ext in ext_list {
                let mut sep = "";
                if !ext.contains('.') {
                    sep = ".";
                }
                find_files_add_to_list(
                    &mut remove_list,
                    &dataset_filename(&format!("{setlet}{sep}{ext}"), None, None),
                );
            }

            // diff files from SIRT, and set up the two prefixes
            let mut sirt_prefs = vec!["_full".to_owned(), "_sub".to_owned()];
            if num_axes > 1 {
                sirt_prefs = vec![setlet.to_owned(), format!("{setlet}_sub")];
            }

            for pref in &sirt_prefs {
                find_files_add_to_list(
                    &mut remove_list,
                    &dataset_filename(&format!("{pref}.diff"), None, None),
                );
            }

            // Other SIRT files with numbers
            let mut sirt_list = vec!["vsr"];
            if keep_sirt == 0 {
                sirt_list.extend(["srec", "strm", "sint"]);
                if Path::new(&trimmed).exists() || Path::new(&untrimmed).exists() {
                    find_files_add_to_list(
                        &mut remove_list,
                        &dataset_filename(&format!("{setlet}.slfrec"), None, None),
                    );
                }
            }

            for sirt in &sirt_list {
                for pref in &sirt_prefs {
                    let pattern = format!("{pref}.{sirt}[0-9][0-9]*");
                    find_files_add_to_list(
                        &mut remove_list,
                        &dataset_filename(&pattern, None, None),
                    );
                }
            }

            // Filter trial output
            if keep_filter == 0 {
                let mut multi_ext = type_ext.as_str();
                if multi_ext.is_empty() {
                    multi_ext = "mrc";
                }
                for pref in ["slfi", "efos", "hlfs0.", "gfc0."] {
                    find_files_add_to_list(
                        &mut remove_list,
                        &format!("{setname}{setlet}_{pref}[0-9][0-9]*.{multi_ext}"),
                    );
                }
            }

            // Leftover stuff from parallel runs
            let mut parallels = vec![
                format!("tilt{setlet}"),
                format!("ctfphaseflip{setlet}"),
                format!("tilt{setlet}_mulfil"),
            ];
            if axis != 0 {
                parallels.push("volcombine".to_owned());
            }

            for com in &parallels {
                for ext in ["log", "com", "pcm"] {
                    find_files_add_to_list(
                        &mut remove_list,
                        &format!("{com}-[0-9][0-9][0-9]*.{ext}"),
                    );
                    find_files_add_to_list(&mut remove_list, &format!("{com}-start.{ext}"));
                    find_files_add_to_list(&mut remove_list, &format!("{com}-finish.{ext}"));
                }
            }

            // Combine temp files
            if num_axes > 1 {
                find_files_add_to_list(
                    &mut remove_list,
                    &format!("{setname}{setlet}.rec.mat[0-9][0-9][0-9][0-9]*"),
                );
                find_files_add_to_list(
                    &mut remove_list,
                    &format!("{setname}{setlet}.rec.wrp[0-9][0-9][0-9][0-9]*"),
                );
            }
        }

        // List is done, now report or use it
        if remove_list.is_empty() {
            prnstr(&format!("Nothing to remove in {tdir}"), "\n", false);
            continue;
        }

        if trial_run != 0 {
            if trial_run == 1 || dir_list.len() > 1 {
                prnstr("", "\n", false);
                prnstr(&format!("Files to be removed in {tdir}:"), "\n", false);
            }

            let mut num_backup = 0;
            for name in &remove_list {
                if trial_run == 1 && name.ends_with('~') {
                    num_backup += 1;
                } else {
                    prnstr(name, "\n", false);
                }
            }
            if num_backup != 0 {
                prnstr(
                    &format!("{num_backup} backup files (ending in ~)"),
                    "\n",
                    false,
                );
            }
        } else {
            cleanup_files(&remove_list);
            let _ = write_text_file("cleanedUpFiles", &remove_list, true);
            prnstr(
                &format!("{} files removed from {tdir}", remove_list.len()),
                "\n",
                false,
            );
        }
    }

    0
}
