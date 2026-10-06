//! Translation of `IMOD/pysrc/excludeviews`: removes views from a tilt
//! series reversibly, or restores the full stack.
//!
//! The script's functions are translated one for one; its top level is
//! [`excludeviews`].  The module globals `fileWithErr`, `logOut` and
//! `renamesDone` are the fields of [`Globals`].  `newstack`, `edmont`,
//! `extracttilts`, `montagesize` (through `getMontageSize`) and `header`
//! (through `getmrcsize`/`getImageFormat`) are our own programs and run in
//! process through `imodpy::run_cmd`/`header_in_process`.  The `-altstack`
//! branch runs `excludeviews` itself for each earlier iteration; that is a
//! Python-script translation with process-global PIP state, so `run_cmd`
//! runs it as a child.  `str(sys.exc_info()[1])` of an `OSError` is
//! reproduced by [`os_error_str`].

use super::imodpy::{
    add_imod_bin_ignore_sighup, cleanup_files, convert_to_integer, exit_from_imod_error, fmtstr,
    get_image_format, get_montage_size, get_mrc_size, glob_glob, make_backup_file,
    os_path_splitext, parse_list, print_pid, prnstr, py_float, py_int, read_text_file, run_cmd,
    write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_in_out_file, pip_get_string,
    pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's module globals `fileWithErr`, `logOut` and `renamesDone`.
#[derive(Default)]
pub struct Globals {
    pub file_with_err: String,
    pub log_out: Vec<String>,
    pub renames_done: Vec<(String, String)>,
}

/// `str(sys.exc_info()[1])` of the `OSError` that `os.remove(name)` or
/// `os.rename(name, name2)` raised: `[Errno n] strerror: 'name'`, with
/// ` -> 'name2'` for a rename.
fn os_error_str(error: &std::io::Error, name: &str, name2: Option<&str>) -> String {
    let text = error.to_string();
    match error.raw_os_error() {
        Some(errno) => {
            let strerror = text
                .strip_suffix(&format!(" (os error {errno})"))
                .unwrap_or(&text);
            match name2 {
                Some(name2) => format!("[Errno {errno}] {strerror}: '{name}' -> '{name2}'"),
                None => format!("[Errno {errno}] {strerror}: '{name}'"),
            }
        }
        None => text,
    }
}

/// Matches `listToRanges` (`excludeviews:17`).  Converts a list to a series
/// of ranges, using range output even for adjacent values.  Handles
/// positive and negative directions just as parselist does.
pub fn list_to_ranges(vals: &[i32]) -> String {
    /// Matches the nested `addRangeToLine` (`excludeviews:18`).
    fn add_range_to_line(line: &str, vals: &[i32], ind_start: usize, ind_end: usize) -> String {
        let mut line = line.to_owned();
        if !line.is_empty() {
            line += ",";
        }
        if ind_end > ind_start {
            format!("{line}{}-{}", vals[ind_start], vals[ind_end])
        } else {
            format!("{line}{}", vals[ind_start])
        }
    }

    let mut ind_start = 0usize;
    let mut line = String::new();
    let mut direc = 0i32;
    for ind in 1..vals.len() {
        if (direc >= 0 && vals[ind] == vals[ind - 1] + 1)
            || (direc <= 0 && vals[ind] == vals[ind - 1] - 1)
        {
            direc = vals[ind] - vals[ind - 1];
        } else {
            line = add_range_to_line(&line, vals, ind_start, ind - 1);
            ind_start = ind;
            direc = 0;
        }
    }

    // Fixed in translation (BUGS.md, `excludeviews`): native indexes
    // `vals[0]` of an empty list here and dies with an IndexError; the
    // callers never pass an empty list after the fixes below, and an empty
    // list gives an empty string.
    if vals.is_empty() {
        return line;
    }
    add_range_to_line(&line, vals, ind_start, vals.len() - 1)
}

/// Matches `removeFile` (`excludeviews:42`).  Removes a file, with a
/// message, saving the file being acted on; the `Err` is the `OSError` text.
pub fn remove_file(g: &mut Globals, name: &str) -> Result<(), String> {
    g.file_with_err = name.to_owned();
    prnstr(&format!("    {name}"), "\n", false);
    std::fs::remove_file(name).map_err(|e| os_error_str(&e, name, None))
}

/// Matches `renameFile` (`excludeviews:48`).
pub fn rename_file(
    g: &mut Globals,
    from_name: &str,
    to_name: &str,
    backup: bool,
) -> Result<(), String> {
    g.file_with_err = from_name.to_owned();
    prnstr(
        &fmtstr(
            "    {}  ->  {}",
            &[from_name.to_owned(), to_name.to_owned()],
        ),
        "\n",
        false,
    );

    // Windows python cannot rename to an existing file, so we need to back it up or delete
    // it explicitly
    if backup {
        make_backup_file(to_name);
    } else if Path::new(to_name).exists() {
        std::fs::remove_file(to_name).map_err(|e| os_error_str(&e, to_name, None))?;
    }
    std::fs::rename(from_name, to_name).map_err(|e| os_error_str(&e, from_name, Some(to_name)))
}

/// Matches `checkExists` (`excludeviews:63`).  Check if file exists and exit
/// with message if not.
pub fn check_exists(name: &str, descrip: &str) {
    if !Path::new(name).exists() {
        exit_error(&format!("{descrip}, {name}, does not exist"));
    }
}

/// Matches `revertRenames` (`excludeviews:69`).  Move all the files back
/// that were saved.
pub fn revert_renames(g: &mut Globals) -> i32 {
    let mut err = 0;
    prnstr("Trying to revert files to original state", "\n", false);
    let renames = g.renames_done.clone();
    for files in &renames {
        if let Err(exc) = rename_file(g, &files.1, &files.0, false) {
            let all_exists = Path::new(&files.1).exists();
            if all_exists && Path::new(&files.0).exists() {
                prnstr(
                    &fmtstr(
                        "ERROR: Could not remove new {} ({}) so could not rename {} back to {}",
                        &[files.0.clone(), exc, files.1.clone(), files.0.clone()],
                    ),
                    "\n",
                    false,
                );
                g.log_out
                    .push(format!("You need to remove the new {}", files.0));
                g.log_out.push(fmtstr(
                    "You need to rename {} to {}",
                    &[files.1.clone(), files.0.clone()],
                ));
                err = 1;
            } else if all_exists {
                prnstr(
                    &fmtstr(
                        "ERROR: Could not rename {} back to {} ({})",
                        &[files.1.clone(), files.0.clone(), exc],
                    ),
                    "\n",
                    false,
                );
                g.log_out.push(fmtstr(
                    "You need to rename {} to {}",
                    &[files.1.clone(), files.0.clone()],
                ));
                err = 1;
            }

            // Figure there is no problem if it somehow succeeded
        }
    }

    err
}

/// Matches `manageRename` (`excludeviews:98`).  Do a rename of original file
/// as safely as possible, reverting everything upon a failure and giving
/// error message for what was not done correctly in reversion.
pub fn manage_rename(
    g: &mut Globals,
    main_name: &str,
    all_name: &str,
    kept_name: &str,
    rename_to_all: bool,
    rename_kept: bool,
) -> i32 {
    // First rename existing file to all
    if rename_to_all {
        match rename_file(g, main_name, all_name, false) {
            Ok(()) => g
                .renames_done
                .push((main_name.to_owned(), all_name.to_owned())),
            Err(exc) => {
                prnstr(
                    &fmtstr(
                        "ERROR: Could not rename {} to {} ({})",
                        &[main_name.to_owned(), all_name.to_owned(), exc],
                    ),
                    "\n",
                    false,
                );
                return 1 + revert_renames(g);
            }
        }
    }

    // Then renaming the new file to the main name, trying to restore the old if it fails
    if rename_kept {
        if let Err(exc) = rename_file(g, kept_name, main_name, false) {
            prnstr(
                &fmtstr(
                    "ERROR: Could not rename {} to {} ({})",
                    &[kept_name.to_owned(), main_name.to_owned(), exc],
                ),
                "\n",
                false,
            );
            return 1 + revert_renames(g);
        }
    }

    0
}

/// Matches `removeAllFile` (`excludeviews:123`).  Remove a file with a
/// warning and explanation if it fails.
pub fn remove_all_file(main_name: &str, all_name: &str) -> i32 {
    prnstr(&format!("Removing {all_name}"), "\n", false);
    if let Err(e) = std::fs::remove_file(all_name) {
        prnstr(
            &fmtstr(
                "WARNING: could not remove {}, which is the original {} ({})",
                &[
                    all_name.to_owned(),
                    main_name.to_owned(),
                    os_error_str(&e, all_name, None),
                ],
            ),
            "\n",
            false,
        );
        return 1;
    }

    0
}

/// Matches `checkOneDoseType` (`excludeviews:137`).  Read either exposure
/// dose or prior record dose from the mdoc file and return the number of
/// entries; for exposure dose return 0 if any entry is 0.
pub fn check_one_dose_type(filename: &str, key_entry: &str) -> usize {
    let mut num_exp_doses = 0usize;
    let dose_name = format!("{filename}.dosetemp12345");
    let attempt = (|| -> Option<()> {
        run_cmd(
            &format!("extracttilts {key_entry} -mdoc \"{filename}\" \"{dose_name}\""),
            None,
            None,
            None,
            &[],
        )
        .ok()?;
        if let Ok(exp_lines) = read_text_file(&dose_name, None, true, None) {
            num_exp_doses = exp_lines.len();
            for line in &exp_lines {
                let dose = py_float(line)?;
                if key_entry == "-exp" && dose == 0. {
                    num_exp_doses = 0;
                    break;
                }
            }
        }
        Some(())
    })();
    if attempt.is_none() {
        num_exp_doses = 0;
    }

    cleanup_files(&[dose_name]);
    num_exp_doses
}

/// Matches `checkDoseInfoInMdoc` (`excludeviews:162`).  Check whether there
/// are doses but no prior record dose entries, in which case dose weighting
/// can get screwed up.
pub fn check_dose_info_in_mdoc(filename: &str) {
    let num_exp_doses = check_one_dose_type(filename, "-exp");
    let num_prior_doses = check_one_dose_type(filename, "-key PriorRecordDose");
    if num_exp_doses > 0 && num_prior_doses != num_exp_doses {
        prnstr(
            "WARNING: Prior Record dose values are missing or incomplete in metadata; the new \
             file should not be used for dose weighting",
            "\n",
            false,
        );
    }
}

/// Matches `getIterationNumber` (`excludeviews:172`).  Get the iteration
/// number for next operation on a stack, or last operation if restore set.
/// The number is the script's string.
pub fn get_iteration_number(root: &str, restore: bool) -> String {
    let mut info_list = glob_glob(&format!("{root}[1-9][0-9].info"));
    info_list.sort();
    let mut iter_num = "0".to_owned();
    if let Some(last) = info_list.last() {
        // `infoList[-1][-7:-5]`
        let chars: Vec<char> = last.chars().collect();
        iter_num = chars[chars.len() - 7..chars.len() - 5].iter().collect();
        if !restore {
            if iter_num == "99" {
                exit_error("This operation cannot be done more than 100 times in succession");
            }
            iter_num = (py_int(&iter_num).unwrap_or(0) + 1).to_string();
        }
    } else {
        let mut info_list = glob_glob(&format!("{root}[0-9].info"));
        info_list.sort();
        if let Some(last) = info_list.last() {
            // `infoList[-1][-6]`
            let chars: Vec<char> = last.chars().collect();
            iter_num = chars[chars.len() - 6].to_string();
            if !restore {
                iter_num = (py_int(&iter_num).unwrap_or(0) + 1).to_string();
            }
        }
    }

    iter_num
}

/// The script's top level (`excludeviews:192-710`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn excludeviews(arguments: &[OsString]) -> i32 {
    let progname = "excludeviews";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };
    let mut g = Globals::default();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 excludeviews
    let options: Vec<String> = [
        "stack:StackName:FN:",
        "views:ViewsToExclude:LI:",
        "montage:MontagedImages:B:",
        "delete:DeleteOldFiles:B:",
        "restore:RestoreFullStack:B:",
        "orig:OriginalStack:B:",
        ":PID:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 1);

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);
    let mut command = "newstack";
    let mut image_str = "";

    let stack_name = pip_get_in_out_file("StackName", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if stack_name.is_empty() {
        exit_error("The name of the image stack must be entered");
    }
    let alt_stack = pip_get_string("AlternateStackName", "").unwrap_or_default();
    let exclude_str = pip_get_string("ViewsToExclude", "").unwrap_or_default();
    let delete_old = pip_get_boolean("DeleteOldFiles", 0).unwrap_or(0);
    let keep_old = 1 - delete_old;
    let restore = pip_get_boolean("RestoreFullStack", 0).unwrap_or(0);
    let use_orig = pip_get_boolean("OriginalStack", 0).unwrap_or(0);
    let mut montage = pip_get_boolean("MontagedImages", 0).unwrap_or(0);
    let mont_entered = 1 - pip_get_err_no();
    if !alt_stack.is_empty()
        && (!exclude_str.is_empty()
            || delete_old != 0
            || restore != 0
            || use_orig != 0
            || mont_entered != 0)
    {
        exit_error("No other options besides -stack should be entered with -altstack");
    }

    // Split up name and figure out what iteration number to use for files
    let (rootname, stack_ext) = os_path_splitext(&stack_name);
    let tilt_ext = ".rawtlt";
    let mut cut_root = format!("{rootname}_cutviews");

    let iter_num = get_iteration_number(&cut_root, restore != 0);

    // Process alternate stack: get its next iteration number
    if !alt_stack.is_empty() {
        let (alt_root, _alt_ext) = os_path_splitext(&alt_stack);
        let alt_iter_num = get_iteration_number(&format!("{alt_root}_cutviews"), false);
        // Fixed in translation (BUGS.md, `excludeviews`): native compares
        // the two iteration numbers as strings, so '9' >= '10' and an
        // alternate stack nine iterations behind cannot catch up past 9;
        // they are compared as the numbers they are.
        if py_int(&alt_iter_num).unwrap_or(0) >= py_int(&iter_num).unwrap_or(0) {
            exit_error(&fmtstr(
                "The number for the next iteration with the main stack ({}) must be greater \
                 than that for the alternate stack ({})",
                &[iter_num.clone(), alt_iter_num.clone()],
            ));
        }

        // Loop from its next iter number to the last iteration on main stack
        let first_iter = py_int(&alt_iter_num).unwrap_or(0);
        for this_iter in first_iter..py_int(&iter_num).unwrap_or(0) {
            // Get the info file, view list and info line
            let info_file = format!("{cut_root}{this_iter}.info");
            if !Path::new(&info_file).exists() {
                exit_error(&format!(
                    "The exclusion info file {info_file} does not exist"
                ));
            }
            let info_lines = read_text_file(&info_file, Some("exclusion info file"), false, None)
                .unwrap_or_default();
            // Fixed in translation (BUGS.md, `excludeviews`): native takes
            // `infoLines[-1]` and `infoLines[0]` of an empty file and dies
            // with an IndexError; an empty file has too few entries.
            if info_lines.is_empty() {
                exit_error(&format!(
                    "There are not at least 4 entries on the first line in {info_file}"
                ));
            }
            let view_list = info_lines[info_lines.len() - 1].clone();
            let lsplit: Vec<&str> = info_lines[0].split_whitespace().collect();
            let mut mont_opt = "";
            let mut del_opt = "";

            // Get size and montage flag from info line
            if lsplit.len() < 4 {
                exit_error(&format!(
                    "There are not at least 4 entries on the first line in {info_file}"
                ));
            }
            let (num_at_start, mont) = match (py_int(lsplit[0]), py_int(lsplit[3])) {
                (Some(num_at_start), Some(mont)) => (num_at_start, mont),
                _ => exit_error(&format!(
                    "Converting a value in on the first line in {info_file} to an integer"
                )),
            };

            // Make sure the number it started with there matches the current alt stack size
            if this_iter == first_iter {
                match get_mrc_size(&alt_stack) {
                    Ok((_nx_alt, _ny_alt, nz_alt)) => {
                        if nz_alt as i64 != num_at_start {
                            exit_error(&format!(
                                "The starting stack size in {info_file} does not match the \
                                 current size of the alternate stack"
                            ));
                        }
                    }
                    Err(_) => exit_from_imod_error(progname),
                }
            }

            // Set up excludeviews command and run it
            let all_file = format!("{rootname}_allviews{this_iter}{stack_ext}");
            if !Path::new(&all_file).exists() {
                del_opt = "-del";
            }
            if mont != 0 {
                mont_opt = "-mont";
            }
            let exclude_com = fmtstr(
                "excludeviews -view {} {} {} \"{}\"",
                &[
                    view_list,
                    mont_opt.to_owned(),
                    del_opt.to_owned(),
                    alt_stack.clone(),
                ],
            );
            match run_cmd(&exclude_com, None, None, None, &[]) {
                Ok(excl_lines) => {
                    for line in excl_lines.unwrap_or_default() {
                        if line.contains("LOG:") {
                            break;
                        }
                        prnstr(line.trim(), "\n", false);
                    }
                }
                Err(_) => exit_from_imod_error(progname),
            }
        }

        return done(0);
    }

    // make up lots of names
    cut_root += &iter_num;
    let all_root = format!("{rootname}_allviews{iter_num}");
    let kept_root = format!("{rootname}_keptviews{iter_num}");
    let info_name = format!("{cut_root}.info");
    let all_stack = format!("{all_root}{stack_ext}");
    let cut_stack = format!("{cut_root}{stack_ext}");
    let kept_stack = format!("{kept_root}{stack_ext}");
    let all_tilt = format!("{all_root}{tilt_ext}");
    let cut_tilt = format!("{cut_root}{tilt_ext}");
    let kept_tilt = format!("{kept_root}{tilt_ext}");
    let raw_tilt = format!("{rootname}{tilt_ext}");
    let all_pclist = format!("{all_root}.pl");
    let cut_pclist = format!("{cut_root}.pl");
    let kept_pclist = format!("{kept_root}.pl");
    let piece_list = format!("{rootname}.pl");
    // `formatsForExt = {'MRC' : 'MRC', 'HDF' : 'HDF', 'TIF' : 'TIF', 'TIFF' : 'TIF'}`
    let formats_for_ext = |key: &str| -> Option<&'static str> {
        match key {
            "MRC" => Some("MRC"),
            "HDF" => Some("HDF"),
            "TIF" => Some("TIF"),
            "TIFF" => Some("TIF"),
            _ => None,
        }
    };

    let mut do_mdoc = 0i32;
    let mut do_rawtlt = 0i32;
    let mut do_pclist = 0i32;
    let mut use_stack_name = stack_name.clone();
    if restore != 0 && use_orig != 0 {
        use_stack_name = format!("{rootname}_orig{stack_ext}");
    }

    // If montage flag not entered, autodetect montage from header or mdoc, not pl
    if mont_entered == 0 {
        if get_montage_size(&use_stack_name, None).is_ok() {
            montage = 1;
        }
    }

    // Get the exclude list and do common checks on it
    let mut exclude_list: Vec<i32> = Vec::new();
    let mut exc_sort_str = String::new();
    if !exclude_str.is_empty() {
        // Fixed in translation (BUGS.md, `excludeviews`): `parselist`
        // returns None for a bad entry and native then dies sorting it with
        // an AttributeError; a list with no numbers dies in `listToRanges`
        // with an IndexError.
        exclude_list = match parse_list(&exclude_str) {
            Some(list) if !list.is_empty() => list,
            _ => exit_error(&format!(
                "Bad entry in the list of views to exclude: {exclude_str}"
            )),
        };
        exclude_list.sort();
        exc_sort_str = list_to_ranges(&exclude_list);
        for iz in &exclude_list {
            if exclude_list.iter().filter(|value| *value == iz).count() > 1 {
                exit_error(&format!(
                    "View number {iz} occurs more than once in the list"
                ));
            }
        }
    }

    // Set output format if current file type matches its extension
    let ext_format = stack_ext.chars().skip(1).collect::<String>().to_uppercase();
    if let Some(ext_mapped) = formats_for_ext(&ext_format) {
        if let Ok(cur_format) = get_image_format(&stack_name) {
            if let Some(cur_mapped) = formats_for_ext(&cur_format) {
                if cur_mapped == ext_mapped {
                    // SAFETY: a single-threaded script setting its own
                    // environment, as the source's `os.environ[...] =` does.
                    unsafe { std::env::set_var("IMOD_OUTPUT_FORMAT", cur_mapped) };
                }
            }
        }
    }

    if restore == 0 {
        if exclude_str.is_empty() {
            exit_error("The list of views to exclude must be entered when not restoring files");
        }
        if use_orig != 0 {
            exit_error("The -orig option is not used except when restoring the full stack");
        }

        check_exists(&stack_name, "The stack file");

        if Path::new(&raw_tilt).exists() {
            do_rawtlt = 1;
        }

        if montage != 0 && Path::new(&piece_list).exists() {
            do_pclist = 1;
        }

        let nz: i32;
        let hdf_input: bool;
        let kept_hdf: bool;
        let attempt = (|| -> Result<(i32, bool, bool), ()> {
            let hdf_input = get_image_format(&stack_name).map_err(|_| ())? == "HDF";
            if Path::new(&format!("{stack_name}.mdoc")).exists() || hdf_input {
                do_mdoc = 1;
                if !hdf_input {
                    check_dose_info_in_mdoc(&stack_name);
                }
            }
            let nz;
            if montage != 0 {
                image_str = "Image";
                command = "edmont";
                (_, _, nz) = get_montage_size(&stack_name, Some(&piece_list)).map_err(|_| ())?;
            } else {
                (_, _, nz) = get_mrc_size(&stack_name).map_err(|_| ())?;
            }

            let mut include_list: Vec<i32> = Vec::new();
            for iz in 1..nz + 1 {
                if !exclude_list.contains(&iz) {
                    include_list.push(iz);
                }
            }

            // Do further checks on the list
            for iz in &exclude_list {
                if *iz < 1 || *iz > nz {
                    exit_error(&format!("View number {iz} is out of range for stack"));
                }
                if exclude_list.len() as i32 == nz {
                    exit_error("The exclude list contains all views");
                }
            }

            // Sort the tilt angles into the two sets if doing them
            if do_rawtlt != 0 {
                let raw_tilt_list =
                    read_text_file(&raw_tilt, None, false, None).unwrap_or_default();
                let mut num_tilts = 0;
                let mut cut_tilt_list: Vec<String> = Vec::new();
                let mut kept_tilt_list: Vec<String> = Vec::new();
                for line in &raw_tilt_list {
                    if !line.trim().is_empty() {
                        num_tilts += 1;
                        if exclude_list.contains(&num_tilts) {
                            cut_tilt_list.push(line.clone());
                        } else {
                            kept_tilt_list.push(line.clone());
                        }
                    }
                }
                if num_tilts != nz {
                    exit_error(&format!("There is not a line in {raw_tilt} for each view"));
                }

                make_backup_file(&cut_tilt);
                let _ = write_text_file(&cut_tilt, &cut_tilt_list, false);
                make_backup_file(&kept_tilt);
                let _ = write_text_file(&kept_tilt, &kept_tilt_list, false);
            }

            // Copy the retained and excluded views
            exc_sort_str = list_to_ranges(&exclude_list);
            // Fixed in translation (see `list_to_ranges`): with every view
            // excluded the checks above have already exited.
            let include_str = list_to_ranges(&include_list);
            let mut common = vec![
                format!("{image_str}InputFile {stack_name}"),
                "NumberedFromOne".to_owned(),
            ];
            if montage != 0 {
                common.push("RenumberZFromZero 1".to_owned());
                if do_pclist != 0 {
                    common.push(format!("PieceListInput {piece_list}"));
                }
            }
            if do_mdoc != 0 {
                common.push("UseMdocFiles".to_owned());
            }
            let mut input = common.clone();
            input.extend([
                format!("SectionsToRead {include_str}"),
                format!("{image_str}OutputFile {kept_stack}"),
            ]);
            if do_pclist != 0 {
                input.push(format!("PieceListOutput {kept_pclist}"));
            }
            prnstr(
                &format!("Copying retained views to {kept_stack}"),
                "\n",
                false,
            );
            run_cmd(
                &format!("{command} -StandardInput"),
                Some(&input),
                None,
                None,
                &[],
            )
            .map_err(|_| ())?;

            let mut input = common;
            input.extend([
                format!("SectionsToRead {exc_sort_str}"),
                format!("{image_str}OutputFile {cut_stack}"),
            ]);
            if do_pclist != 0 {
                input.push(format!("PieceListOutput {cut_pclist}"));
            }
            prnstr(
                &format!("Copying excluded views to {cut_stack}"),
                "\n",
                false,
            );
            run_cmd(
                &format!("{command} -StandardInput"),
                Some(&input),
                None,
                None,
                &[],
            )
            .map_err(|_| ())?;
            let kept_hdf = get_image_format(&kept_stack).map_err(|_| ())? == "HDF";
            Ok((nz, hdf_input, kept_hdf))
        })();
        match attempt {
            Ok(values) => (nz, hdf_input, kept_hdf) = values,
            Err(()) => {
                prnstr(
                    "Operation failed; leaving original files as they were",
                    "\n",
                    false,
                );
                exit_from_imod_error(progname);
            }
        }

        // Do all the renames/removals
        g.log_out = Vec::new();
        g.renames_done = Vec::new();
        let mdoc_file = format!("{stack_name}.mdoc");
        let kept_mdoc = format!("{kept_stack}.mdoc");
        let move_mdoc = !hdf_input && Path::new(&mdoc_file).exists();
        prnstr("Renaming files:", "\n", false);
        let mut error = manage_rename(&mut g, &stack_name, &all_stack, &kept_stack, true, true);
        if do_rawtlt != 0 && error == 0 {
            error = manage_rename(&mut g, &raw_tilt, &all_tilt, &kept_tilt, true, true);
        }
        if do_mdoc != 0 && error == 0 {
            error = manage_rename(
                &mut g,
                &mdoc_file,
                &format!("{all_stack}.mdoc"),
                &kept_mdoc,
                move_mdoc,
                (!kept_hdf) && Path::new(&kept_mdoc).exists(),
            );
        }
        if do_pclist != 0 && error == 0 {
            error = manage_rename(&mut g, &piece_list, &all_pclist, &kept_pclist, true, true);
        }
        if error > 1 {
            prnstr(
                "LOG: You need to do these actions to make this dataset usable again:",
                "\n",
                false,
            );
            for line in &g.log_out {
                prnstr(line, "\n", false);
            }

            prnstr(" ", "\n", false);
            exit_error(
                "YOU CANNOT PROCEED WITH THIS DATASET WITHOUT TAKING THE ACTIONS IN THE LOG \
                 MESSAGES",
            );
        }

        if error != 0 {
            exit_error("Operation failed; no views were removed and original files were restored");
        }

        if keep_old == 0 {
            remove_all_file(&stack_name, &all_stack);
            if do_rawtlt != 0 {
                remove_all_file(&raw_tilt, &all_tilt);
            }
            if do_mdoc != 0 && move_mdoc {
                remove_all_file(&mdoc_file, &format!("{all_stack}.mdoc"));
            }
            if do_pclist != 0 {
                remove_all_file(&piece_list, &all_pclist);
            }
        }

        let _ = write_text_file(
            &info_name,
            &[
                fmtstr(
                    "{} {} {} {}",
                    &[
                        nz.to_string(),
                        do_rawtlt.to_string(),
                        do_mdoc.to_string(),
                        montage.to_string(),
                    ],
                ),
                exc_sort_str.clone(),
            ],
            false,
        );
        prnstr("LOG:", "\n", false);
        prnstr(
            "Operations successfully completed.  To restore full stack, enter:",
            "\n",
            false,
        );
        prnstr(&format!("   {progname} -restore {stack_name}"), "\n", false);
        return done(0);
    }

    // RESTORING A FULL STACK

    // First check if appropriate files exist
    check_exists(&cut_stack, "Stack with removed views");
    check_exists(&use_stack_name, "Stack with retained views");

    let mut kept_hdf = false;
    let mut cut_hdf = false;
    let mut kept_mdoc = false;
    let mut all_hdf = false;
    let attempt = (|| -> Result<(), ()> {
        kept_hdf = get_image_format(&use_stack_name).map_err(|_| ())? == "HDF";
        cut_hdf = get_image_format(&cut_stack).map_err(|_| ())? == "HDF";
        kept_mdoc = Path::new(&format!("{stack_name}.mdoc")).exists();
        let cut_mdoc = Path::new(&format!("{cut_stack}.mdoc")).exists();
        let kept_pl = Path::new(&piece_list).exists();
        let cut_pl = Path::new(&cut_pclist).exists();
        let mut nz_info = -1i32;

        // If the list was entered, try to figure out everything from it
        if !exclude_str.is_empty() {
            do_rawtlt = i32::from(Path::new(&raw_tilt).exists() && Path::new(&cut_tilt).exists());
            do_mdoc = i32::from((kept_hdf || kept_mdoc) && (cut_hdf || cut_mdoc));
            if do_mdoc == 0 && (kept_mdoc || cut_mdoc) {
                prnstr(
                    "WARNING: An .mdoc file exists for one of the two files being combined, \
                     not both",
                    "\n",
                    false,
                );
            }

        // Otherwise get info from the file and make sure it makes sense
        } else if Path::new(&info_name).exists() {
            let info_lines = read_text_file(&info_name, None, false, None).unwrap_or_default();
            if info_lines.len() < 2 {
                exit_error(&format!(
                    "The info file, {info_name}, has fewer than two lines"
                ));
            }
            exc_sort_str = info_lines[1].clone();
            // Fixed in translation (BUGS.md, `excludeviews`): a bad list
            // makes `parselist` return None, which native then iterates and
            // dies with a TypeError.
            exclude_list = match parse_list(&exc_sort_str) {
                Some(list) => list,
                None => exit_error(&format!(
                    "Bad entry in the list of views in the info file, {info_name}"
                )),
            };
            let lsplit: Vec<&str> = info_lines[0].split_whitespace().collect();
            if lsplit.len() < 4 {
                exit_error(&format!(
                    "The first line of the info file, {info_name}, has fewer than 4 numbers"
                ));
            }
            let descrip = "value on first line of info file";
            nz_info = convert_to_integer(lsplit[0], descrip);
            do_rawtlt = convert_to_integer(lsplit[1], descrip);
            do_mdoc = convert_to_integer(lsplit[2], descrip);
            montage = convert_to_integer(lsplit[3], descrip);
            if do_rawtlt != 0 && (!Path::new(&raw_tilt).exists() || !Path::new(&cut_tilt).exists())
            {
                exit_error(
                    "The info file shows that .rawtlt files were operated on, but a .rawtlt \
                     exists for only one  of the two files being combined",
                );
            }
            if do_mdoc != 0 && !((kept_hdf || kept_mdoc) && (cut_hdf || cut_mdoc)) {
                exit_error(
                    "The info file shows that metadata was operated on, but metadata exists \
                     for only one of the two files being combined",
                );
            }
        } else {
            exit_error(&format!(
                "There is no info file from the original run of {progname}; try running with \
                 an excluded view list"
            ));
        }

        do_pclist = i32::from(montage != 0 && kept_pl && cut_pl);
        if montage != 0 && do_pclist == 0 && (kept_pl || cut_pl) {
            prnstr(
                "WARNING: A piece list file exists for only one of the two stacks being \
                 combined; cannot make combined piece list file",
                "\n",
                false,
            );
        }

        // Get sizes, set strings for montage
        let (nz_cut, nz_kept);
        if montage != 0 {
            image_str = "Image";
            command = "edmont";
            (_, _, nz_cut) = get_montage_size(&cut_stack, Some(&cut_pclist)).map_err(|_| ())?;
            (_, _, nz_kept) =
                get_montage_size(&use_stack_name, Some(&piece_list)).map_err(|_| ())?;
        } else {
            (_, _, nz_cut) = get_mrc_size(&cut_stack).map_err(|_| ())?;
            (_, _, nz_kept) = get_mrc_size(&use_stack_name).map_err(|_| ())?;
        }

        let nz_all = nz_cut + nz_kept;
        if nz_info >= 0 && nz_info != nz_all {
            exit_error(&fmtstr(
                "The # of views listed in the info file ({}) does not match the total of \
                 excluded and retained views ({} + {} = {})",
                &[
                    nz_info.to_string(),
                    nz_cut.to_string(),
                    nz_kept.to_string(),
                    nz_all.to_string(),
                ],
            ));
        }

        // Final checks of the exclude list
        for iz in &exclude_list {
            if *iz < 1 || *iz > nz_all {
                exit_error(&format!(
                    "View number {iz} is out of range for combined stack"
                ));
            }
        }
        if exclude_list.len() as i32 != nz_cut {
            exit_error(&fmtstr(
                "The # of views in the exclude list, {}, does not match the # in the stack of \
                 cut views, {}",
                &[exclude_list.len().to_string(), nz_cut.to_string()],
            ));
        }

        let mut kept_tilt_list: Vec<String> = Vec::new();
        let mut cut_tilt_list: Vec<String> = Vec::new();
        let mut all_tilt_list: Vec<String> = Vec::new();
        if do_rawtlt != 0 {
            let kept_lines = read_text_file(&raw_tilt, None, false, None).unwrap_or_default();
            let cut_lines = read_text_file(&cut_tilt, None, false, None).unwrap_or_default();
            for line in kept_lines {
                if !line.is_empty() {
                    kept_tilt_list.push(line);
                }
            }
            for line in cut_lines {
                if !line.is_empty() {
                    cut_tilt_list.push(line);
                }
            }
            if kept_tilt_list.len() as i32 != nz_kept {
                exit_error(&fmtstr(
                    "The # of lines in {} ({}) does not match the # of retained views ({})",
                    &[
                        raw_tilt.clone(),
                        kept_tilt_list.len().to_string(),
                        nz_kept.to_string(),
                    ],
                ));
            }
            if cut_tilt_list.len() as i32 != nz_cut {
                exit_error(&fmtstr(
                    "The # of lines in {} ({}) does not match the # of excluded views ({})",
                    &[
                        cut_tilt.clone(),
                        cut_tilt_list.len().to_string(),
                        nz_cut.to_string(),
                    ],
                ));
            }
        }

        if do_mdoc != 0 && kept_mdoc && !kept_hdf && use_orig != 0 {
            if std::fs::copy(
                format!("{stack_name}.mdoc"),
                format!("{use_stack_name}.mdoc"),
            )
            .is_err()
            {
                exit_error("Copying .mdoc file to use with _orig stack");
            }
        }

        let mut kept_ind = 0i32;
        let mut cut_ind = 0i32;
        let mut last_excluded = 0i32;
        let exc_split: Vec<&str> = exc_sort_str.split(',').collect();
        let mut cut_old_views: Vec<Vec<i32>> = Vec::new();
        let mut input = vec![format!("{image_str}OutputFile {all_stack}")];
        if do_mdoc != 0 {
            input.push("UseMdocFiles".to_owned());
        }
        if montage != 0 {
            input.push("RenumberZFromZero 1".to_owned());
            if do_pclist != 0 {
                input.push(format!("PieceListOutput {all_pclist}"));
            }
        }

        for exc in &exc_split {
            // `parselist(exc)`; a group the whole list parsed in cannot fail
            // alone except as an empty group, which native also indexes
            // below and dies on.  Fixed in translation (BUGS.md,
            // `excludeviews`): such a group is the list error above.
            match parse_list(exc) {
                Some(group) if !group.is_empty() => cut_old_views.push(group),
                _ => exit_error(&format!(
                    "Bad entry in the list of views to exclude: {exc_sort_str}"
                )),
            }
        }

        // Loop on the excluded groups and fill in kept views before each one
        let num_exc_groups = cut_old_views.len();
        for group_ind in 0..num_exc_groups + 1 {
            let mut next_excluded = nz_all + 1;
            if group_ind < num_exc_groups {
                next_excluded = cut_old_views[group_ind][0];
            }
            let num_kept = next_excluded - (last_excluded + 1);
            if num_kept != 0 {
                input.extend([
                    format!("{image_str}InputFile {use_stack_name}"),
                    format!("SectionsToRead {kept_ind}"),
                ]);
                if num_kept > 1 {
                    let last = input.len() - 1;
                    input[last] += &format!("-{}", kept_ind + num_kept - 1);
                }
                if do_pclist != 0 {
                    input.push(format!("PieceListInput {piece_list}"));
                }
                if do_rawtlt != 0 {
                    all_tilt_list.extend(py_list_slice(
                        &kept_tilt_list,
                        kept_ind as i64,
                        (kept_ind + num_kept) as i64,
                    ));
                }
                kept_ind += num_kept;
            }

            // Then add the excluded views in that group
            if group_ind < num_exc_groups {
                let num_cut = cut_old_views[group_ind].len() as i32;
                input.extend([
                    format!("{image_str}InputFile {cut_stack}"),
                    format!("SectionsToRead {cut_ind}"),
                ]);
                if num_cut > 1 {
                    let last = input.len() - 1;
                    input[last] += &format!("-{}", cut_ind + num_cut - 1);
                }
                if do_pclist != 0 {
                    input.push(format!("PieceListInput {cut_pclist}"));
                }
                if do_rawtlt != 0 {
                    all_tilt_list.extend(py_list_slice(
                        &cut_tilt_list,
                        cut_ind as i64,
                        (cut_ind + num_cut) as i64,
                    ));
                }
                cut_ind += num_cut;
                last_excluded = *cut_old_views[group_ind].last().unwrap();
            }
        }

        if do_rawtlt != 0 {
            let _ = write_text_file(&all_tilt, &all_tilt_list, false);
        }
        prnstr(
            &format!("Recombining the stack into {all_stack}..."),
            "\n",
            false,
        );
        run_cmd(
            &format!("{command} -StandardInput"),
            Some(&input),
            None,
            None,
            &[],
        )
        .map_err(|_| ())?;
        all_hdf = get_image_format(&all_stack).map_err(|_| ())? == "HDF";
        Ok(())
    })();
    if attempt.is_err() {
        prnstr(
            "Operation failed; leaving original files as they were",
            "\n",
            false,
        );
        exit_from_imod_error(progname);
    }

    // Fixed in translation (BUGS.md, `excludeviews`): native assigns
    // `action` only once it starts removing (`-delete`) or renaming, so a
    // failure removing the temporary `_orig` mdoc with old files kept raised
    // a NameError in the error handler.  It starts as 'Removing '.
    let mut action = "Removing ";
    let attempt = (|| -> Result<(), String> {
        // Do all the renames/removals
        let mdoc_file = format!("{stack_name}.mdoc");
        if keep_old == 0 {
            action = "Removing ";
            prnstr(
                "Removing files for retained and excluded views...",
                "\n",
                false,
            );
            remove_file(&mut g, &cut_stack)?;
            if do_rawtlt != 0 {
                remove_file(&mut g, &cut_tilt)?;
            }
            if do_mdoc != 0 && !cut_hdf {
                remove_file(&mut g, &format!("{cut_stack}.mdoc"))?;
            }
            if do_mdoc != 0 && kept_mdoc && !kept_hdf {
                remove_file(&mut g, &mdoc_file)?;
            }
            if do_pclist != 0 {
                remove_file(&mut g, &cut_pclist)?;
            }
            if montage != 0 && do_pclist == 0 && Path::new(&piece_list).exists() {
                // `keptPL`, tested before any file was moved
                remove_file(&mut g, &piece_list)?;
            }
        }

        if do_mdoc != 0 && kept_mdoc && !kept_hdf && use_orig != 0 {
            if keep_old != 0 {
                prnstr("Removing temporary mdoc file", "\n", false);
            }
            remove_file(&mut g, &format!("{use_stack_name}.mdoc"))?;
        }

        action = "Renaming ";
        prnstr("Renaming files...", "\n", false);
        if keep_old != 0 {
            rename_file(&mut g, &stack_name, &kept_stack, false)?;
            if use_orig != 0 {
                rename_file(
                    &mut g,
                    &use_stack_name,
                    &format!("{rootname}_orig_keptviews{stack_ext}"),
                    false,
                )?;
            }
            if do_rawtlt != 0 {
                rename_file(&mut g, &raw_tilt, &kept_tilt, false)?;
            }
            if do_mdoc != 0 && kept_mdoc && !kept_hdf {
                rename_file(&mut g, &mdoc_file, &format!("{kept_stack}.mdoc"), false)?;
            }
            if do_pclist != 0 {
                rename_file(&mut g, &piece_list, &kept_pclist, false)?;
            }
        }

        rename_file(&mut g, &all_stack, &use_stack_name, false)?;
        if do_rawtlt != 0 {
            rename_file(&mut g, &all_tilt, &raw_tilt, false)?;
        }
        if do_mdoc != 0 && !all_hdf && Path::new(&format!("{all_stack}.mdoc")).exists() {
            rename_file(&mut g, &format!("{all_stack}.mdoc"), &mdoc_file, false)?;
        }
        if do_pclist != 0 {
            rename_file(&mut g, &all_pclist, &piece_list, false)?;
        }
        rename_file(&mut g, &info_name, &format!("{cut_root}_old.info"), false)?;
        Ok(())
    })();
    if let Err(exc) = attempt {
        exit_error(&format!("{action}{}: {exc}", g.file_with_err));
    }

    done(0)
}

/// Python's `list[start:end]` for non-negative bounds, clamped to the list.
fn py_list_slice(list: &[String], start: i64, end: i64) -> Vec<String> {
    let len = list.len() as i64;
    let start = start.clamp(0, len) as usize;
    let end = end.clamp(0, len) as usize;
    if start >= end {
        return Vec::new();
    }
    list[start..end].to_vec()
}
