//! Translation of `IMOD/pysrc/comchanger.py`.
use super::pysed::pysed;
use std::path::{Path, PathBuf};

/// Matches the Python change-list representation.
pub type Change = Vec<String>;

/// Matches `modifyForChangeList` (`IMOD/pysrc/comchanger.py:17`).
pub fn modify_for_change_list(
    comlines: &[String],
    com_root: &str,
    axis_let: &str,
    changes: &[Change],
) -> Result<Vec<String>, String> {
    let mut starts = comlines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let text = line.trim_start();
            if !text.starts_with('$') {
                return None;
            }
            let process = text[1..].trim_start().split_whitespace().next()?.to_owned();
            Some((index, process))
        })
        .collect::<Vec<_>>();
    if starts.is_empty() {
        return Ok(comlines.to_vec());
    }
    starts.push((comlines.len(), String::new()));
    let mut output = comlines[..starts[0].0].to_vec();
    for index in 0..starts.len() - 1 {
        let (start, process) = &starts[index];
        let lines = &comlines[*start..starts[index + 1].0];
        if process == "if" {
            output.extend_from_slice(lines);
            continue;
        }
        let axis = format!("{com_root}{axis_let}");
        let mut selected: Vec<&Change> = Vec::new();
        for change in changes {
            if change.len() < 4
                || (change[0] != axis && change[0] != com_root)
                || change[1] != *process
            {
                continue;
            }
            if let Some(old) = selected.iter().position(|item| item[2] == change[2]) {
                if axis_let.is_empty() && change[0] == format!("{com_root}b") {
                    continue;
                }
                if axis_let.is_empty() || !(selected[old][0] == axis && change[0] == com_root) {
                    selected[old] = change;
                }
            } else {
                selected.push(change);
            }
        }
        let mut sed = Vec::new();
        for change in selected {
            if !change[3].is_empty() {
                if lines.iter().any(|line| line.starts_with(&change[2])) {
                    sed.push(format!("|^{}|s|[ \t].*|\t{}|", change[2], change[3]));
                } else {
                    sed.push(format!(
                        "|^ *\\$ *{}|a|{}\t{}|",
                        process, change[2], change[3]
                    ));
                }
            } else {
                sed.push(format!("|^{}|d", change[2]));
            }
        }
        if sed.is_empty() {
            output.extend_from_slice(lines);
        } else {
            output.extend(pysed(&sed, lines, false, '|')?);
        }
    }
    Ok(output)
}

/// Matches `changeToAddToList` (`IMOD/pysrc/comchanger.py:112`).
pub fn change_to_add_to_list(
    change_list: &mut Vec<Change>,
    line: &str,
    prefix: &str,
    num_components: i32,
    value_needed: bool,
) -> Result<(), String> {
    let split = line.splitn(2, '=').collect::<Vec<_>>();
    let keys = split[0].split('.').map(str::trim).collect::<Vec<_>>();
    let desired = num_components.unsigned_abs() as usize;
    if keys.first() != Some(&prefix) {
        return Ok(());
    }
    if split.len() < 2 {
        return Err(format!("Missing = sign in entry: {line}"));
    }
    if value_needed && split[1].trim().is_empty() {
        return Err(format!("Empty value in entry: {line}"));
    }
    if num_components < 0 && keys.len() != desired {
        return Ok(());
    }
    if keys.len() < desired {
        return Err(format!("Not enough components in entry: {line}"));
    }
    let mut change = keys[1..desired]
        .iter()
        .map(|item| (*item).to_owned())
        .collect::<Vec<_>>();
    change.push(split[1].trim().to_owned());
    change_list.push(change);
    Ok(())
}

/// Matches `processChangeOptions` (`IMOD/pysrc/comchanger.py:139`) after PIP input has been collected.
pub fn process_change_options(
    file_lines: &[Vec<String>],
    entries: &[String],
    prefix: &str,
    num_components: i32,
    value_needed: bool,
) -> Result<Vec<Change>, String> {
    let mut output = Vec::new();
    for lines in file_lines {
        for line in lines {
            change_to_add_to_list(&mut output, line, prefix, num_components, value_needed)?;
        }
    }
    for line in entries {
        change_to_add_to_list(&mut output, line, prefix, num_components, value_needed)?;
    }
    Ok(output)
}

/// Matches `getSetupsetValue` (`IMOD/pysrc/comchanger.py:177`).
pub fn get_setupset_value(change_list: &[Change], prefix: &str, option: &str) -> Option<String> {
    change_list
        .iter()
        .filter(|change| change.len() >= 3 && change[0] == prefix && change[1] == option)
        .map(|change| change[2].clone())
        .last()
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
