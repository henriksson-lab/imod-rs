//! Translation of `IMOD/pysrc/swaptomostacks`: exchanges the raw stack and
//! other files of a data set for processing with the same alignment.
//!
//! The script's two functions are [`convert_filename`] and
//! [`rename_files`]; its top level is [`swaptomostacks`].  The module
//! globals they read (`rootname`, `stackExt`) are passed in.  `header`
//! (through `getmrcsize`) runs in process; `excludeviews` is a Python-script
//! translation with process-global PIP state, so `imodpy::run_cmd` runs it
//! as a child.

use super::imodpy::{
    add_imod_bin_ignore_sighup, allowed_raw_stack_extensions, dataset_filename,
    default_naming_style, exit_from_imod_error, find_root_axis_and_extensions, fmtstr,
    get_mrc_size, prnstr, run_cmd, set_root_and_extension,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_in_out_file, pip_get_integer, pip_get_string,
    pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// `stTag = '/st/'` (`swaptomostacks:10`).
const ST_TAG: &str = "/st/";

/// `str(sys.exc_info()[1])` of the `OSError` that `os.rename(from, to)`
/// raised: `[Errno n] strerror: 'from' -> 'to'`.
fn rename_error_str(error: &std::io::Error, from: &str, to: &str) -> String {
    let text = error.to_string();
    match error.raw_os_error() {
        Some(errno) => {
            let strerror = text
                .strip_suffix(&format!(" (os error {errno})"))
                .unwrap_or(&text);
            format!("[Errno {errno}] {strerror}: '{from}' -> '{to}'")
        }
        None => text,
    }
}

/// Matches `convertFilename` (`swaptomostacks:13`): substitute stack
/// extension into stack-type suffix, or call datasetFilename.  `rootname`
/// and `stack_ext` are the script's globals.
pub fn convert_filename(suff: &str, root: Option<&str>, rootname: &str, stack_ext: &str) -> String {
    let root = match root {
        Some(root) if !root.is_empty() => root,
        _ => rootname,
    };
    if suff.contains("/st/") {
        return format!("{root}{}", suff.replace(ST_TAG, stack_ext));
    }
    dataset_filename(suff, Some(root), None)
}

/// Matches `renameFiles` (`swaptomostacks:22`): rename files with the given
/// suffixes if they exist.
pub fn rename_files(
    suffixes: &[String],
    root_from: &str,
    root_to: &str,
    rootname: &str,
    stack_ext: &str,
) {
    for suff in suffixes {
        let from_name = convert_filename(suff, Some(root_from), rootname, stack_ext);
        if Path::new(&from_name).exists() {
            let to_name = convert_filename(suff, Some(root_to), rootname, stack_ext);
            match std::fs::rename(&from_name, &to_name) {
                Ok(()) => prnstr(&format!("Renamed {from_name} -> {to_name}"), "\n", false),
                Err(error) => exit_error(&fmtstr(
                    "Renaming {} to {} gave the error {}",
                    &[
                        from_name.clone(),
                        to_name.clone(),
                        rename_error_str(&error, &from_name, &to_name),
                    ],
                )),
            }
        }
    }
}

/// The script's top level (`swaptomostacks:34-230`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn swaptomostacks(arguments: &[OsString]) -> i32 {
    let progname = "swaptomostacks";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 swaptomostacks
    let options: Vec<String> = [
        "from:FromRootname:CH:",
        "to:ToRootname:CH:",
        "root:SetRootname:CH:",
        "single:SingleAxisSet:B:",
        "excl:ExcludeType:I:",
        "check:CheckForExcludingViews:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // PIP startup and help
    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 2, 1, 1);

    // Get the options, which are almost all required
    let mut rootname = pip_get_string("SetRootname", "").unwrap_or_default();

    let from_root = pip_get_in_out_file("FromRootname", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if from_root.is_empty() {
        exit_error("You must enter the rootname to rename files from");
    }
    let to_root = pip_get_in_out_file("ToRootname", 1)
        .ok()
        .flatten()
        .unwrap_or_default();
    if to_root.is_empty() {
        exit_error("You must enter the rootname to rename files to");
    }
    let exclude = pip_get_integer("ExcludeType", 0).unwrap_or(0);
    let force_single = pip_get_boolean("SingleAxisSet", 0).unwrap_or(0);
    let check_sizes = pip_get_boolean("CheckForExcludingViews", 0).unwrap_or(0);

    let try_exts = allowed_raw_stack_extensions();

    let (_com_ext, mut dual_num, root, mut type_ext, mut stack_ext) =
        find_root_axis_and_extensions(0, None);
    let mut single = dual_num != 2;

    if rootname.is_empty() {
        if !root.is_empty() {
            if force_single != 0 && !single {
                exit_error(
                    "You must enter a root name including a or b for a dual-axis set with the \
                     -single option",
                );
            }
            rootname = root.clone();
        } else {
            exit_error("You must enter the data set root name; it cannot be determined");
        }
    }

    if type_ext.is_none() {
        let (_name_style, default_ext) = default_naming_style();
        type_ext = Some(default_ext);
        let mut warntx = "descriptive extension style".to_owned();
        if type_ext.as_deref().is_some_and(|ext| !ext.is_empty()) {
            warntx = format!("extension {}", type_ext.as_deref().unwrap());
        }
        prnstr(
            &format!("WARNING: {progname} - Cannot determine file naming style, assuming {warntx}"),
            "\n",
            false,
        );
    }

    // Just check the stacks if axis type and extension found
    if !stack_ext.is_empty() {
        if force_single != 0 {
            single = true;
            dual_num = 0;
        }
        if single && !Path::new(&format!("{rootname}.{stack_ext}")).exists() {
            exit_error(&format!(
                "Cannot find single stack named {rootname}.{stack_ext}"
            ));
        }
        if dual_num != 0 && !Path::new(&format!("{rootname}a.{stack_ext}")).exists() {
            exit_error(&format!(
                "Cannot find A stack named {rootname}a.{stack_ext}"
            ));
        }
        if dual_num != 0 && !Path::new(&format!("{rootname}b.{stack_ext}")).exists() {
            exit_error(&format!(
                "Cannot find B stack named {rootname}b.{stack_ext}"
            ));
        }
    } else {
        // Fall back to deducing single or dual set and standard extension
        for ext in &try_exts {
            stack_ext = ext.clone();
            dual_num = 0;
            single = Path::new(&format!("{rootname}.{stack_ext}")).exists();
            if force_single == 0 {
                if Path::new(&format!("{rootname}a.{stack_ext}")).exists() {
                    dual_num += 1;
                }
                if Path::new(&format!("{rootname}b.{stack_ext}")).exists() {
                    dual_num += 1;
                }
            }
            if single && dual_num != 0 {
                exit_error(&format!(
                    "Both single and dual-axis stacks exist with extension .{stack_ext}"
                ));
            }
            if dual_num == 1 {
                exit_error(&format!(
                    "Only one of the dual-axis stacks exists with extension .{stack_ext}"
                ));
            }
            if single || dual_num != 0 {
                break;
            }
        }

        if !single && dual_num == 0 {
            exit_error("Cannot find single or dual axis stacks with any standard extensions");
        }
    }

    set_root_and_extension(&rootname, type_ext.as_deref().unwrap_or(""));

    // Set up the suffixes to move; if you use a tuple you need ('',)
    let mut move_suffixes: Vec<String> = Vec::new();
    let mut set_lets: Vec<&str> = if single { vec![""] } else { vec!["a", "b"] };
    for lett in &set_lets {
        move_suffixes.extend([
            format!("{lett}.{ST_TAG}"),
            format!("{lett}_orig.{ST_TAG}"),
            format!("{lett}_xray.{ST_TAG}.gz"),
        ]);
    }

    // do a and b separately because of the rec variations
    if exclude < 2 {
        move_suffixes.push(".rec".to_owned());
        if single {
            move_suffixes.push("_full.rec".to_owned());
        } else {
            move_suffixes.push("a.rec".to_owned());
            move_suffixes.push("b.rec".to_owned());
        }
    }
    if exclude < 1 {
        if single {
            move_suffixes.push(".ali".to_owned());
        } else {
            move_suffixes.push("a.ali".to_owned());
            move_suffixes.push("b.ali".to_owned());
        }
    }

    // Check for conflicts in the TO names before renaming anything
    for suff in &move_suffixes {
        let from_file = convert_filename(suff, None, &rootname, &stack_ext);
        let to_file = convert_filename(suff, Some(&to_root), &rootname, &stack_ext);
        if Path::new(&from_file).exists() && Path::new(&to_file).exists() {
            exit_error(&format!("A file already exists with the name {to_file}"));
        }
    }

    // Now make sure the source stack(s) exist; allow them to be named st, mrc, hdf
    let mut rename_exts: [Option<String>; 2] = [None, None];
    for ext in &try_exts {
        if single {
            if Path::new(&format!("{from_root}.{ext}")).exists() && rename_exts[0].is_none() {
                rename_exts[0] = Some(ext.clone());
                break;
            }
        } else {
            if Path::new(&format!("{from_root}a.{ext}")).exists() && rename_exts[0].is_none() {
                rename_exts[0] = Some(ext.clone());
            }
            if Path::new(&format!("{from_root}b.{ext}")).exists() && rename_exts[1].is_none() {
                rename_exts[1] = Some(ext.clone());
            }
            if rename_exts[0].is_some() && rename_exts[1].is_some() {
                break;
            }
        }
    }

    if rename_exts[0].is_none() || (dual_num != 0 && rename_exts[1].is_none()) {
        exit_error(&format!(
            "No stacks exists with the \"from\" rootname {from_root}"
        ));
    }

    // Check for matching sizes and excluding views
    if check_sizes != 0 {
        set_lets = vec![""];
        if !single {
            set_lets = vec!["a", "b"];
        }
        for (lp, lett) in set_lets.iter().enumerate() {
            let main_stack = format!("{rootname}{lett}.{stack_ext}");
            // Set for every letter used: checked above.
            let rename_ext = rename_exts[lp].clone().unwrap_or_default();
            let alt_stack = format!("{from_root}{lett}.{rename_ext}");
            let attempt = (|| -> Result<(), ()> {
                let (_main_nx, _main_ny, main_nz) = get_mrc_size(&main_stack).map_err(|_| ())?;
                let (_alt_nx, _alt_ny, alt_nz) = get_mrc_size(&alt_stack).map_err(|_| ())?;
                if main_nz > alt_nz {
                    exit_error(&fmtstr(
                        "The current stack is bigger in Z ({}) than the stack being swapped \
                         in, {} ({})",
                        &[main_nz.to_string(), alt_stack.clone(), alt_nz.to_string()],
                    ));
                }
                if main_nz < alt_nz {
                    let cut_name = format!("{rootname}{lett}_cutviews0.info");
                    if !Path::new(&cut_name).exists() {
                        exit_error(&fmtstr(
                            "The current stack is smaller in Z ({}) than the stack being \
                             swapped in, {} ({}), but there is no info file for running \
                             Excludeviews",
                            &[main_nz.to_string(), alt_stack.clone(), alt_nz.to_string()],
                        ));
                    }
                    prnstr(
                        &format!("Running Excludeviews on stack being swapped in, {alt_stack}"),
                        "\n",
                        false,
                    );
                    run_cmd(
                        &fmtstr(
                            "excludeviews -alt \"{}\" \"{}\"",
                            &[alt_stack.clone(), main_stack.clone()],
                        ),
                        None,
                        Some("stdout"),
                        None,
                        &[],
                    )
                    .map_err(|_| ())?;
                }
                Ok(())
            })();
            if attempt.is_err() {
                exit_from_imod_error(progname);
            }
        }
    }

    // Rename current files
    rename_files(&move_suffixes, &rootname, &to_root, &rootname, &stack_ext);

    // Rename the source stacks to standard extension if needed
    for (ind, lett) in set_lets.iter().enumerate() {
        let rename_ext = rename_exts[ind].clone().unwrap_or_default();
        let cur_name = format!("{from_root}{lett}.{rename_ext}");
        let std_name = format!("{rootname}{lett}.{stack_ext}");
        if cur_name != std_name {
            match std::fs::rename(&cur_name, &std_name) {
                Ok(()) => prnstr(&format!("Renamed {cur_name} -> {std_name}"), "\n", false),
                Err(error) => exit_error(&fmtstr(
                    "Renaming {} to {} gave the error {}",
                    &[
                        cur_name.clone(),
                        std_name.clone(),
                        rename_error_str(&error, &cur_name, &std_name),
                    ],
                )),
            }
        }
    }

    // Rename the source files
    rename_files(&move_suffixes, &from_root, &rootname, &rootname, &stack_ext);

    done(0)
}
