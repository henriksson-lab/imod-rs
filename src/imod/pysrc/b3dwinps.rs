//! Translation of `IMOD/pysrc/b3dwinps`: process status output on Windows,
//! in the layout of Cygwin's `ps`, for eTomo's `PsParam`.
//!
//! A Python command script with no functions; its top level is
//! [`b3dwinps`].  The Cygwin arm (`ps` through `runcmd`) is not translated:
//! Cygwin Python is not a platform this crate builds for.  The Windows arm's
//! `psutil` calls are `imodpy::psutil`'s Win32 equivalents; everywhere else
//! the script exits with its own "only on Windows" error.

use super::imodpy::prnstr;
use super::pip::{exit_error, set_exit_prefix};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`b3dwinps:1-108`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn b3dwinps(arguments: &[OsString]) -> i32 {
    let progname = "b3dwinps";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup minimal runtime environment
    if std::env::var_os("IMOD_DIR").is_none() {
        let mut out = std::io::stdout();
        let _ = out.write_all(format!("{prefix} IMOD_DIR is not defined!\n").as_bytes());
        let _ = out.flush();
        return 1;
    }

    set_exit_prefix(prefix);

    let windows = cfg!(windows);
    let mut one_pid: Option<i64> = None;

    if !windows {
        exit_error("This program produces process status output only on Windows");
    }

    if argv.len() > 1 {
        if argv[1] != "-p" || argv.len() != 3 {
            exit_error("Incorrect entry; use \"b3dwinps -p pid\" for output from one PID");
        }
        match super::imodpy::py_int(&argv[2]) {
            Some(pid) => one_pid = Some(pid),
            None => exit_error(&format!("Converting {} to an integer", argv[2])),
        }
    }

    // WINDOWS: get the process list (`psutil.Process(onePid)` or
    // `psutil.process_iter()`)
    #[cfg(windows)]
    {
        use super::imodpy::psutil;
        let proc_list: Vec<i64> = match one_pid.filter(|pid| *pid != 0) {
            Some(pid) => vec![pid],
            None => psutil::process_list()
                .into_iter()
                .map(|(pid, _)| pid)
                .collect(),
        };
        let parents = psutil::process_list();

        // Output it just like Cygwin
        prnstr(
            "      PID    PPID    PGID     WINPID   TTY     UID    STIME COMMAND",
            "\n",
            false,
        );
        for pid in proc_list {
            if pid == 0 {
                continue;
            }
            // Any exception skips the process, as the script's three
            // `except` clauses do
            let Some(ppid) = parents
                .iter()
                .find(|(p, _)| *p == pid)
                .map(|(_, ppid)| *ppid)
            else {
                continue;
            };
            let Some(exen) = psutil::process_exe(pid) else {
                continue;
            };
            let Some(username) = psutil::username(pid) else {
                continue;
            };
            let Some(create_time) = psutil::create_time(pid) else {
                continue;
            };
            // `os.path.basename` of `DOMAIN\user` (ntpath splits on `\`)
            let mut uid: String = username
                .rsplit(['\\', '/'])
                .next()
                .unwrap_or_default()
                .to_owned();
            if uid.chars().count() > 7 {
                uid = uid.chars().take(7).collect();
            }
            if uid.chars().count() < 7 {
                uid = format!("{uid:>7}");
            }
            // `datetime.fromtimestamp(createTime).strftime("%H:%M:%S")`, local time
            let stime = {
                use chrono::TimeZone as _;
                let seconds = create_time.floor() as i64;
                let nanos = ((create_time - create_time.floor()) * 1e9) as u32;
                chrono::Local
                    .timestamp_opt(seconds, nanos)
                    .single()
                    .map(|time| time.format("%H:%M:%S").to_string())
                    .unwrap_or_default()
            };
            // `fmtstr(' {:8d} {:7d}      -1 {:10d}  ?    {} {} {}', ...)`: Python's
            // `d` fields right-align, as Rust's integer formatting does
            prnstr(
                &format!(" {pid:8} {ppid:7}      -1 {pid:10}  ?    {uid} {stime} {exen}"),
                "\n",
                false,
            );
        }
    }
    #[cfg(not(windows))]
    let _ = one_pid;

    0
}
