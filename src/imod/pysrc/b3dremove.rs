//! Translation of `IMOD/pysrc/b3dremove`.
//!
//! A Python command script with no functions; its top level is
//! [`b3dremove`], translated statement by statement.

use super::imodpy::glob_glob;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`b3dremove:1-89`).  Returns the status of its
/// `sys.exit`.
pub fn b3dremove(arguments: &[OsString]) -> i32 {
    let retry_wait = std::time::Duration::from_secs_f64(0.5);
    let max_trials = 10;
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    if argv.len() < 2 {
        print!(
            "Usage: b3dremove [-e] [-g] filenames OR -r directories
    Removes files with optional -g to glob on wild cards in the filenames, or
    removes directory trees with the -r option.  In either case -e makes it
    exit with an error if it fails to remove a file or directory.\n"
        );
        let _ = std::io::stdout().flush();
        return 0;
    }

    let mut num_start = 1;
    let mut doglob = false;
    let mut dotree = false;
    let mut fail_exit_val = 0;
    let mut fail_pref = "WARNING";
    while num_start < argv.len() {
        if argv[num_start] == "-g" {
            doglob = true;
            num_start += 1;
        } else if argv[num_start] == "-r" {
            dotree = true;
            num_start += 1;
        } else if argv[num_start] == "-e" {
            fail_exit_val = 1;
            num_start += 1;
            fail_pref = "ERROR";
        } else {
            break;
        }
    }

    if argv.len() <= num_start {
        return 0;
    }

    if doglob && dotree {
        print!("ERROR: b3dremove - You cannot enter both -r and -g\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    let mut files: Vec<Option<String>> = Vec::new();
    for arg in &argv[num_start..] {
        if doglob {
            files.extend(glob_glob(arg).into_iter().map(Some));
        } else if Path::new(arg).exists() {
            files.push(Some(arg.clone()));
        }
    }

    let mut num_to_del = files.len();
    let mut trial = 0;
    let mut exit_val = 0;
    let mut fail_mess = String::new();
    while num_to_del != 0 && trial < max_trials {
        trial += 1;
        for ind in 0..files.len() {
            let Some(onef) = files[ind].clone() else {
                continue;
            };
            if onef.is_empty() {
                continue;
            }
            let removed = if dotree {
                // `shutil.rmtree` refuses a symbolic link ("Cannot call rmtree
                // on a symbolic link"), where `remove_dir_all` would remove the
                // link itself
                match std::fs::symlink_metadata(&onef) {
                    Ok(meta) if meta.file_type().is_symlink() => false,
                    _ => std::fs::remove_dir_all(&onef).is_ok(),
                }
            } else {
                std::fs::remove_file(&onef).is_ok()
            };
            if removed {
                num_to_del -= 1;
                files[ind] = None;
                if trial > 1 {
                    print!("It took {trial} tries to remove {onef}\n");
                }
            } else if trial == max_trials {
                if !fail_mess.is_empty() {
                    fail_mess += ", ";
                }
                fail_mess += &onef;
            }
        }

        if num_to_del != 0 {
            std::thread::sleep(retry_wait);
        }
    }

    if !fail_mess.is_empty() {
        print!("{fail_pref}: b3dremove - Could not remove: {fail_mess}\n");
        exit_val = fail_exit_val;
    }

    let _ = std::io::stdout().flush();
    exit_val
}
