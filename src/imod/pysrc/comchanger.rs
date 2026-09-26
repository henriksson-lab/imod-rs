//! Translation of `IMOD/pysrc/comchanger.py`.
use super::imodpy::read_text_file;
use super::pip::{exit_error, pip_get_string, pip_number_of_entries};
use super::pysed::{PysedSrc, pysed};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// Matches the Python change-list representation: all components of a
/// directive but the first, then the value.
pub type Change = Vec<String>;

/// Matches the module globals `fileEntries` and `oneOptEntries`
/// (`IMOD/pysrc/comchanger.py:131-132`): "there is no way to reset PIP".
static FILE_ENTRIES: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static ONE_OPT_ENTRIES: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Matches `modifyForChangeList` (`IMOD/pysrc/comchanger.py:17`).
///
/// `Err` carries the string `pysed` returns with `retErr`, which the source
/// passes back to its caller.
pub fn modify_for_change_list(
    comlines: &[String],
    com_root: &str,
    axis_let: &str,
    change_list: &[Change],
    return_on_err: bool,
) -> Result<Vec<String>, String> {
    // Analyze the lines for command blocks
    let mut proc_starts: Vec<(usize, String)> = Vec::new();
    let proc_match = regex::Regex::new(r"^ *\$ *(\w+).*$").unwrap();
    for ind in 0..comlines.len() {
        let line = &comlines[ind];
        if let Some(captures) = proc_match.captures(line) {
            let process = captures[1].to_owned();
            proc_starts.push((ind, process));
        }
    }
    let num_procs = proc_starts.len();
    proc_starts.push((comlines.len(), String::new()));

    // Start output with any lines up to the first command
    let mut outlines: Vec<String> = Vec::new();
    if proc_starts[0].0 != 0 {
        outlines.extend_from_slice(&comlines[0..proc_starts[0].0]);
    }

    // Loop on the processes, for each one, run through the change lines and compile the
    // list of relevant changes
    for ind in 0..num_procs {
        let proclines = &comlines[proc_starts[ind].0..proc_starts[ind + 1].0];
        let process = &proc_starts[ind].1;
        if process == "if" {
            outlines.extend_from_slice(proclines);
            continue;
        }

        let com_axis = format!("{com_root}{axis_let}");
        let mut proc_changes: Vec<&Change> = Vec::new();
        for change in change_list {
            if (change[0] == com_axis || change[0] == com_root) && change[1] == *process {
                let mut broke = false;
                for ichan in 0..proc_changes.len() {
                    if proc_changes[ichan][2] == change[2] {
                        // If single axis, ignore a change that is B-specific
                        if axis_let.is_empty() && change[0] == format!("{com_root}b") {
                            broke = true;
                            break;
                        }

                        // Otherwise, replace with a later entry unless previous one was
                        // axis-specific while this one is generic
                        if axis_let.is_empty()
                            || !(proc_changes[ichan][0] == com_axis && change[0] == com_root)
                        {
                            proc_changes[ichan] = change;
                        }
                        broke = true;
                        break;
                    }
                }
                if !broke {
                    // If the loop does not break by replacement or ignoring, add the change
                    proc_changes.push(change);
                }
            }
        }

        // build a sed command
        let mut sedcom: Vec<String> = Vec::new();
        if !proc_changes.is_empty() {
            for change in &proc_changes {
                // Modify an existing line or append a new one after the process command
                // if there is a value
                if !change[3].is_empty() {
                    if proclines.iter().any(|line| line.starts_with(&change[2])) {
                        sedcom.push(format!("|^{}|s|[ \t].*|\t{}|", change[2], change[3]));
                    } else {
                        sedcom.push(format!(
                            "|^ *$ *{}|a|{}\t{}|",
                            process, change[2], change[3]
                        ));
                    }
                } else {
                    // Or delete the line if there is no value
                    sedcom.push(format!("|^{}|d", change[2]));
                }
            }

            let tmplines = pysed(
                &sedcom,
                PysedSrc::Lines(proclines),
                None,
                false,
                '|',
                return_on_err,
            )?;
            outlines.extend(tmplines.unwrap_or_default());
        } else {
            outlines.extend_from_slice(proclines);
        }
    }

    Ok(outlines)
}

/// Matches `changeToAddToList` (`IMOD/pysrc/comchanger.py:106`).
///
/// Add one change to the list with the given prefix and number of components, which
/// can be negative to just skip ones without that number of components
/// Store a change as a list of all components but the first and the value, which
/// can be blank unless valNeed is true
pub fn change_to_add_to_list(
    change_list: &mut Vec<Change>,
    line: &str,
    change_file: &str,
    prefix: &str,
    num_components: i32,
    val_needed: bool,
) {
    let lsplit = line.split('=').collect::<Vec<_>>();
    let keysplit = lsplit[0].split('.').collect::<Vec<_>>();
    let mut errtext = "entry: ".to_owned();
    let allow_other_nums = num_components < 0;
    let mut num_components = num_components;
    if allow_other_nums {
        num_components *= -1;
    }
    if !change_file.is_empty() {
        errtext = format!("line from {change_file}: ");
    }
    if keysplit[0] == prefix {
        if lsplit.len() < 2 {
            exit_error(&format!("Missing = sign in {errtext}{line}"));
        }
        if val_needed && lsplit[1].trim().is_empty() {
            exit_error(&format!("Empty value in {errtext}{line}"));
        }
        if !(allow_other_nums && keysplit.len() as i32 != num_components) {
            if (keysplit.len() as i32) < num_components {
                exit_error(&format!("Not enough components in {errtext}{line}"));
            }
            let mut change: Change = Vec::new();
            for comp in &keysplit[1..num_components as usize] {
                change.push(comp.trim().to_owned());
            }
            change.push(lsplit[1].trim().to_owned());
            change_list.push(change);
        }
    }
}

/// Matches `processChangeOptions` (`IMOD/pysrc/comchanger.py:137`).
///
/// Process changes from two kinds of options, specified by fileOption for a file,
/// and oneChangeOpt for an entry of one change, and look for changes starting with prefix
/// numComponents can be negative to skip directives have different numbers of components
/// (the source's defaults are `numComponents = 4`, `valNeeded = False`).
pub fn process_change_options(
    file_option: &str,
    one_change_opt: &str,
    prefix: &str,
    num_components: i32,
    val_needed: bool,
) -> Vec<Change> {
    let mut change_list: Vec<Change> = Vec::new();
    let mut file_entries = FILE_ENTRIES.lock().expect("comchanger file entries");
    let mut one_opt_entries = ONE_OPT_ENTRIES.lock().expect("comchanger one entries");
    if !file_option.is_empty() && file_entries.is_empty() {
        let num_changers = pip_number_of_entries(file_option).unwrap_or(0);
        for _chan in 0..num_changers {
            let change_file = pip_get_string(file_option, "").unwrap_or_default();
            file_entries.push(change_file);
        }
    }

    if !one_change_opt.is_empty() && one_opt_entries.is_empty() {
        let num_changers = pip_number_of_entries(one_change_opt).unwrap_or(0);
        for _chan in 0..num_changers {
            let line = pip_get_string(one_change_opt, "").unwrap_or_default();
            one_opt_entries.push(line);
        }
    }

    if !file_option.is_empty() {
        let num_changers = file_entries.len();
        for chan in 0..num_changers {
            let change_file = file_entries[chan].clone();
            let Ok(change_lines) = read_text_file(&change_file, None, false, None) else {
                unreachable!("readTextFile exits on error without returnOnErr")
            };
            for line in &change_lines {
                change_to_add_to_list(
                    &mut change_list,
                    line,
                    &change_file,
                    prefix,
                    num_components,
                    val_needed,
                );
            }
        }
    }

    if !one_change_opt.is_empty() {
        let num_changers = one_opt_entries.len();
        for chan in 0..num_changers {
            let line = one_opt_entries[chan].clone();
            change_to_add_to_list(
                &mut change_list,
                &line,
                "",
                prefix,
                num_components,
                val_needed,
            );
        }
    }

    change_list
}

/// Matches `getSetupsetValue` (`IMOD/pysrc/comchanger.py:177`).
pub fn get_setupset_value(change_list: &[Change], prefix: &str, option: &str) -> Option<String> {
    let mut value = None;
    for change in change_list {
        if prefix == change[0] && option == change[1] {
            value = Some(change[2].clone());
        }
    }

    value
}

/// Matches `absTemplatePath` (`IMOD/pysrc/comchanger.py:188`).
pub fn abs_template_path(
    template: &str,
    index: i32,
    user_template_dir: Option<PathBuf>,
    error_name: &str,
) -> (Option<PathBuf>, i32, Option<PathBuf>, String) {
    let error = format!("{error_name} file {template} not found in expected location");
    let path = Path::new(template);
    if path.is_absolute() {
        return (Some(path.to_owned()), 0, user_template_dir, String::new());
    }
    if path.parent().is_some_and(|parent| parent != Path::new("")) {
        return (
            None,
            -1,
            user_template_dir,
            format!(
                "Directive for {error_name} name must be either an absolute path or just a filename, it is {template}"
            ),
        );
    }
    let mut updated_user_dir = user_template_dir;
    let found = match index {
        0 => std::env::var_os("IMOD_CALIB_DIR")
            .map(|root| PathBuf::from(root).join("ScopeTemplate").join(template)),
        1 => std::env::var_os("IMOD_CALIB_DIR")
            .map(|root| PathBuf::from(root).join("SystemTemplate").join(template))
            .or_else(|| {
                std::env::var_os("IMOD_DIR")
                    .map(|root| PathBuf::from(root).join("SystemTemplate").join(template))
            }),
        _ => {
            if updated_user_dir.is_none() {
                if let Some(home) = std::env::var_os("HOME") {
                    let home = PathBuf::from(home);
                    let default = home.join(".etomotemplate");
                    if default.exists() {
                        updated_user_dir = Some(default);
                    }
                    let preferences = home.join(".etomo");
                    if let Ok(lines) = std::fs::read_to_string(preferences) {
                        for line in lines.lines() {
                            if let Some((key, value)) = line.split_once('=') {
                                if key.trim() == "Defaults.UserTemplateDir"
                                    && !value.trim().is_empty()
                                {
                                    updated_user_dir = Some(PathBuf::from(value.trim()));
                                }
                            }
                        }
                    }
                }
            }
            updated_user_dir.clone().map(|root| root.join(template))
        }
    };
    match found.filter(|path| path.exists()) {
        Some(path) => (Some(path), 1, updated_user_dir, String::new()),
        None => (None, -2, updated_user_dir, error),
    }
}
