//! Translation of `IMOD/pysrc/imodkillgroup`: kills a whole process group
//! given the PID of any member, or the process tree below each PID with
//! `-t`.
//!
//! The script's top level is [`imodkillgroup`]; its functions are
//! [`process_status`] and [`kill_group`].  The Linux/macOS and Windows
//! (`psutil`) arms are translated; the Cygwin branches are not (Cygwin
//! Python is not a platform this crate builds for), and `-s` is refused
//! outside Cygwin by the script itself.  On Unix the process list comes from
//! the system `ps`, as the script runs it, and the group and signal calls are
//! the POSIX ones Python's `os` module wraps.  On Windows the `psutil` calls
//! are the Win32 ones `psutil` makes: a ToolHelp snapshot for the process
//! list and parents, `QueryFullProcessImageNameW` for `exe()`,
//! `NtSuspendProcess` for `suspend()` and `TerminateProcess` for `kill()`.

use super::imodpy::{add_imod_bin_ignore_sighup, fmtstr, get_err_strings, prnstr, py_int, run_cmd};
use super::pip::{exit_error, set_exit_prefix};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::Write as _;

/// A `{pid: value}` dict in insertion order, as Python keeps it.
type PidDict<V> = Vec<(i64, V)>;

/// `str(OSError)` for an errno: `[Errno n] strerror`.
fn os_error_text(errno: i32) -> String {
    let text = std::io::Error::from_raw_os_error(errno).to_string();
    let strerror = text
        .strip_suffix(&format!(" (os error {errno})"))
        .unwrap_or(&text)
        .to_owned();
    format!("[Errno {errno}] {strerror}")
}

/// `def processStatus(pid = None)` (`imodkillgroup:15`), the Windows arm
/// (`:14-39`): every process whose parent and executable can be read, with
/// `AccessDenied`/`NoSuchProcess` entries left out.  The `proc` object of the
/// source's triple is the PID itself, reopened when it is used.
#[cfg(windows)]
fn process_status(pid: Option<i64>) -> Result<PidDict<(i64, String)>, String> {
    let mut stat_dict: PidDict<(i64, String)> = Vec::new();
    for (proc_pid, ppid) in super::imodpy::psutil::process_list() {
        if proc_pid == 0 || pid.is_some_and(|pid| pid != 0 && pid != proc_pid) {
            continue;
        }
        if let Some(exe) = super::imodpy::psutil::process_exe(proc_pid) {
            match stat_dict.iter_mut().find(|(key, _)| *key == proc_pid) {
                Some(slot) => slot.1 = (ppid, exe),
                None => stat_dict.push((proc_pid, (ppid, exe))),
            }
        }
    }
    Ok(stat_dict)
}

/// `def processStatus(pid = None)` (`imodkillgroup:15`), the non-Windows,
/// non-Cygwin arm: a dict of PID to (parent PID, command) from `ps`, or the
/// first error string when `ps` fails.
#[cfg(not(windows))]
fn process_status(pid: Option<i64>) -> Result<PidDict<(i64, String)>, String> {
    let mut stat_dict: PidDict<(i64, String)> = Vec::new();
    let mut command = "ps -aeo pid,ppid,comm".to_owned();
    if let Some(pid) = pid.filter(|pid| *pid != 0) {
        command += &format!(" -p {pid}");
    }
    let pslines = match run_cmd(&command, None, None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => {
            let errs = get_err_strings();
            return Err(errs.first().cloned().unwrap_or_default());
        }
    };

    for line in pslines.iter().skip(1) {
        let lsplit: Vec<&str> = line.split_whitespace().collect();
        if lsplit.len() < 3 {
            continue;
        }
        let (Some(pid), Some(ppid)) = (py_int(lsplit[0]), py_int(lsplit[1])) else {
            continue;
        };
        let entry = (ppid, lsplit[lsplit.len() - 1].to_owned());
        match stat_dict.iter_mut().find(|(key, _)| *key == pid) {
            Some(slot) => slot.1 = entry,
            None => stat_dict.push((pid, entry)),
        }
    }

    Ok(stat_dict)
}

/// `def killGroup(groupid)` (`imodkillgroup:68`): an error message, or
/// `None` on success.
#[cfg(unix)]
fn kill_group(groupid: i64, kill_signal: i32) -> Option<String> {
    // SAFETY: `killpg` takes plain integers.
    if unsafe { libc::killpg(groupid as libc::pid_t, kill_signal) } != 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        return Some(format!(
            "imodkillgroup - Error killing processes with group ID {groupid}: {}",
            os_error_text(errno)
        ));
    }
    None
}

/// The script's top level (`imodkillgroup:77-379`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn imodkillgroup(arguments: &[OsString]) -> i32 {
    let progname = "imodkillgroup";
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

    set_exit_prefix(prefix.clone());

    let mut kill_tree = false;
    let mut pid_list: Vec<i64> = Vec::new();
    let mut verbose = false;
    let mut get_status = false;
    let mut use_term = false;
    // `signal.SIGKILL`/`SIGTERM`; Windows Python has no SIGKILL, and the
    // signal is used only on the non-Windows arms.
    let mut kill_signal: i32 = 9;
    let windows = cfg!(windows);

    for arg in argv.iter().skip(1) {
        if arg == "-t" {
            kill_tree = true;
        } else if arg == "-v" {
            verbose = true;
        } else if arg == "-s" {
            get_status = true;
        } else if arg == "-e" {
            use_term = true;
        } else {
            match py_int(arg) {
                Some(pid) => {
                    if !pid_list.contains(&pid) {
                        pid_list.push(pid);
                    }
                }
                None => exit_error(&format!("Unrecognized option or non-integer entry: {arg}")),
            }
        }
    }

    if pid_list.is_empty() {
        prnstr(
            "Usage: imodkillgroup [-t | -v | -s] PID [PID ...]'
    Kills whole process group given PID of any member, or just kills process
    tree below each PID if -t option is given.  Takes multiple PIDs from
    different groups.  Options:
      -t  Tree kill, implied with Windows Python; unreliable with Cygwin Python
      -s  Use initial and follow-up ps to verifying children are gone (Cygwin)
      -e  Use SIGTERM instead of SIGKILL to allow cleanup actions
      -v  Verbose output",
            "\n",
            false,
        );
        return done(0);
    }

    if get_status {
        exit_error("The -s option works only in Cygwin Python");
    }

    if use_term {
        kill_signal = 15;
    }

    let mut exit_val: i32 = 0;
    #[cfg(unix)]
    if !kill_tree && !windows {
        let num_pids = pid_list.len() as i32;
        let mut group_ids: BTreeMap<i64, i64> = BTreeMap::new();
        let mut pid_done: BTreeMap<i64, i32> = BTreeMap::new();
        for &pid_kill in &pid_list {
            pid_done.insert(pid_kill, 0);
            // SAFETY: `getpgid` takes a plain integer.
            let group = unsafe { libc::getpgid(pid_kill as libc::pid_t) };
            if group < 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                prnstr(
                    &format!(
                        "imodkillgroup - Error getting group ID for process ID {pid_kill}: {}",
                        os_error_text(errno)
                    ),
                    "\n",
                    false,
                );
                exit_val += 1;
                pid_done.insert(pid_kill, 1);
                continue;
            }
            group_ids.insert(pid_kill, group as i64);
            if verbose {
                prnstr(&format!("Group id: {group}"), "\n", false);
            }
        }

        if exit_val == num_pids {
            exit_error("No group ID(s) could be found");
        }

        // With neither Cygwin nor `-s`, the trial loop ends after its first
        // pass, before the follow-up `ps` and the loop's `else`.
        if verbose {
            prnstr("Trial 1 for killing groups", "\n", false);
        }
        for &pid_kill in &pid_list {
            if pid_done[&pid_kill] == 0 {
                if let Some(mess) = kill_group(group_ids[&pid_kill], kill_signal) {
                    exit_val += 1;
                    prnstr(&mess, "\n", false);
                } else {
                    pid_done.insert(pid_kill, 1);
                }
            }
        }

        return done(exit_val);
    }

    // For TREE KILL, Start an array of PID's for each level; a value of -1
    // marks a PID that is no longer there
    let mut pid_tree: Vec<PidDict<Option<i64>>> =
        vec![pid_list.iter().map(|pid| (*pid, None)).collect()];
    // On Windows a value of `Some(pid)` (any non-negative PID) stands for
    // the saved `psutil.Process` object
    let mut psdict: PidDict<(i64, String)> = Vec::new();
    for level in 0..100 {
        // Need a ps to get going for windows
        if level == 0 && windows {
            psdict = match process_status(None) {
                Ok(dict) => dict,
                Err(message) => exit_error(&format!("{prefix}{message}")),
            };
        }

        // Stop processes for PID's at the current level
        for (pid, value) in pid_tree[level].iter_mut() {
            let mut stop_proc = true;
            if windows {
                stop_proc = false;
                if psdict.iter().any(|(key, _)| key == pid) {
                    stop_proc = windows;
                    if verbose {
                        prnstr(&format!("Saving process object for PID {pid}"), "\n", false);
                    }
                    *value = Some(*pid);
                }
            }
            if !stop_proc {
                continue;
            }
            let mut stopstr = format!("Stopping PID {pid}");
            if level != 0 || windows {
                if let Some((_, (_, comm))) = psdict.iter().find(|(key, _)| key == pid) {
                    stopstr += &format!(": {comm}");
                }
            }
            if verbose {
                prnstr(&stopstr, "\n", false);
            }
            #[cfg(windows)]
            let stop_error = super::imodpy::psutil::suspend(*pid).err();
            // SAFETY: `kill` takes plain integers.
            #[cfg(unix)]
            let stop_error = if unsafe { libc::kill(*pid as libc::pid_t, libc::SIGSTOP) } != 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                Some(os_error_text(errno))
            } else {
                None
            };
            if let Some(error) = stop_error {
                prnstr(
                    &format!("imodkillgroup - Error occurred trying to stop {pid}: {error}"),
                    "\n",
                    false,
                );
            }
        }

        // Get a ps and first find out if each PID is still there
        psdict = match process_status(None) {
            Ok(dict) => dict,
            Err(message) => exit_error(&format!("{prefix}{message}")),
        };
        pid_tree.push(Vec::new());
        for (pid, value) in pid_tree[level].iter_mut() {
            if !psdict.iter().any(|(key, _)| key == pid) {
                prnstr(
                    &format!("imodkillgroup - PID {pid} is no longer in the process list"),
                    "\n",
                    false,
                );
                exit_val += 1;
                *value = Some(-1);
            }
        }

        // find children of these processes
        for (pid, (parent, comm)) in &psdict {
            if pid_tree[level]
                .iter()
                .any(|(key, value)| key == parent && (!windows || value.is_some()))
            {
                let next = &mut pid_tree[level + 1];
                if !next.iter().any(|(key, _)| key == pid) {
                    next.push((*pid, None));
                }
                if verbose {
                    prnstr(
                        &fmtstr(
                            "Adding child {} - {} of {} at level {}",
                            &[
                                pid.to_string(),
                                comm.clone(),
                                parent.to_string(),
                                level.to_string(),
                            ],
                        ),
                        "\n",
                        false,
                    );
                }
            }
        }

        if pid_tree[level + 1].is_empty() {
            break;
        }
    }

    // Kill all the processes from the bottom level up
    for level in (0..pid_tree.len()).rev() {
        for (pid, value) in &pid_tree[level] {
            if *value == Some(-1) || (!windows && value.is_some()) {
                continue;
            }
            if verbose && (!windows || value.is_some()) {
                prnstr(&format!("Killing PID {pid} at level {level}"), "\n", false);
            }

            #[cfg(windows)]
            let kill_error = match value {
                Some(proc) => super::imodpy::psutil::kill(*proc).err(),
                None => None,
            };
            // SAFETY: `kill` takes plain integers.
            #[cfg(unix)]
            let kill_error = if unsafe {
                (use_term && libc::kill(*pid as libc::pid_t, libc::SIGCONT) != 0)
                    || libc::kill(*pid as libc::pid_t, kill_signal) != 0
            } {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                Some(os_error_text(errno))
            } else {
                None
            };
            if let Some(error) = kill_error {
                prnstr(
                    &format!("imodkillgroup - Error occurred trying to kill {pid}: {error}"),
                    "\n",
                    false,
                );
                exit_val += 1;
            }
        }
    }

    done(exit_val)
}
