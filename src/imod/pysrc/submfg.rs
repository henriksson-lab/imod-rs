//! Translation of `IMOD/pysrc/submfg`.
//!
//! The original is a Python command program, so its one top-level program
//! body is [`submfg`], translated statement by statement.  **Changed
//! (owner, 2026-09-26: no Python, no pipes):** each command file is run by
//! the in-process runner ([`crate::imod::comrun::run_com_file`]) instead of
//! `vmstopy` + `python -u` (or `vmstocsh` + `tcsh -ef` with -s), so no
//! `submtemp.<pid>` file is written and no `Python PID:` line is printed.
//! -s now means only "programs' standard error is not logged", the one
//! observable difference of the tcsh path the runner can keep; -n nices this
//! process once; -t appends `in N.NN sec` to the completion line (the
//! source's Windows form) instead of the shell's `time` report.
//!
//! The script's `try` over each command file catches `ImodpyError` (a failed
//! `runcmd`) and `KeyboardInterrupt`; `exitError` inside it is a
//! `SystemExit` and ends the program there, without the final
//! `cleanupFiles`.  A Ctrl-C is Python's `KeyboardInterrupt`, which `runcmd`
//! passes on after `passOnKeyInterrupt(True)`; the translation records the
//! signal and takes the `except KeyboardInterrupt: break` arm once the
//! interrupted command has returned.

use std::ffi::OsString;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use super::imodpy::{
    ImodpyError, add_imod_bin_ignore_sighup, convert_to_integer, get_err_strings, glob_glob,
    os_path_splitext, pass_on_key_interrupt, prnstr, read_text_file, set_run_error,
};
use super::pip::{exit_error, expand_arg_list, set_exit_prefix};
use super::vmstopy::VmstopyOptions;
use crate::imod::comrun::{ComOptions, run_com_file};

/// Original Python top-level program (`IMOD/pysrc/submfg:1`).
pub fn submfg(arguments: &[OsString]) -> i32 {
    let progname = "submfg";
    let prefix = format!("ERROR: {progname} - ");
    let sys_argv = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    //
    // Setup runtime environment
    if let Some(imod_dir) = std::env::var_os("IMOD_DIR") {
        let mut imod_dir = imod_dir.to_string_lossy().into_owned();
        if cfg!(target_os = "cygwin") {
            imod_dir = imod_dir.replace('\\', "/");
            let bytes = imod_dir.as_bytes();
            if bytes.len() < 3 {
                eprintln!("IndexError: string index out of range");
                return 1;
            }
            if bytes[1] == b':' && bytes[2] == b'/' {
                imod_dir = format!(
                    "/cygdrive/{}{}",
                    (bytes[0] as char).to_ascii_lowercase(),
                    &imod_dir[2..]
                );
            }
        }
        // `sys.path.insert(0, os.path.join(IMOD_DIR, 'pylib'))` and
        // `from imodpy import *` locate the Python library; here it is linked
        let _ = imod_dir;
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return 1;
    }

    //
    // load IMOD Libraries
    set_exit_prefix(prefix);

    let bell = "\x07";
    let mut message = format!(" finished successfully{bell}");
    if let Some(value) = std::env::var_os("SUBM_MESSAGE") {
        message = value.to_string_lossy().into_owned();
    }
    let mut log_type = 0;
    if let Some(value) = std::env::var_os("SUBM_LOG_TYPE") {
        log_type = convert_to_integer(
            &value.to_string_lossy(),
            "environment variable SUBM_LOG_TYPE",
        );
    }

    // Process arguments
    let lenarg = sys_argv.len();
    let mut argind = 1;
    if lenarg < 2 {
        prnstr(
            "subm or submfg will execute a series of command files in sequence
Usage:  submfg [options] command_file1 command_file2 ...
        Command files can have default extension .com or .pcm
        If the filename is comfile.com or comfile.pcm, you can enter\x20
                 comfile    comfile.  or  comfile.com or comfile.pcm
        submfg will execute the files in the foreground
        subm is an alias defined in the IMOD startup script to execute submfg
               in the background
        Set the environment variable SUBM_MESSAGE to modify the message upon
               completion
        Set the environment variable SUBM_LOG_TYPE to set a default log type
    Options:
        -t     Report the execution time
        -c     Continue with the next command file if one fails
        -s     Translate file with vmstocsh and run with tcsh
                 (default is to translate with vmstopy and run with python)
        -k     Keep backslashes instead of converting to forward slashes'
        -n #   Run niced with # as nice increment (range 1 to 19)
        -l #   Log type for numbered or time-stamped logs:
                  1 - 4 for sequential numbers with 1-4 digits
                 -1 for date-time stamps like Mar-01-195046.4
                 -2 for date-time stamps like 20120301-195121.9
                 -3 for date-time stamps like 2012-03-01T19:51:51.9",
            "\n",
            false,
        );
        return 0;
    }

    let mut nice = 0;
    let mut use_tcsh = false;
    let mut dotime = false;
    let mut cont_if_err = false;
    let mut keep_backslash = false;
    let windows = cfg!(windows);
    while argind < lenarg {
        let oarg = sys_argv[argind].as_str();
        if oarg.starts_with('-') {
            if oarg == "-t" {
                dotime = true;
            } else if oarg == "-c" {
                cont_if_err = true;
            } else if oarg == "-k" {
                keep_backslash = true;
            } else if oarg == "-s" {
                use_tcsh = true;
                if windows {
                    exit_error("You cannot run command files with tcsh from Windows Python");
                }
            } else if oarg == "-n" {
                argind += 1;
                if argind >= lenarg {
                    break;
                }
                nice = convert_to_integer(&sys_argv[argind], "\"nice\" value");
            } else if oarg == "-l" {
                argind += 1;
                if argind >= lenarg {
                    break;
                }
                log_type = convert_to_integer(&sys_argv[argind], "log type value");
            } else {
                exit_error(&format!("Unrecognized argument {oarg}"));
            }
            argind += 1;
        } else {
            break;
        }
    }

    if argind >= lenarg {
        exit_error("No command file was entered");
    }

    let (new_args, no_match_ind) = expand_arg_list(&arguments[argind..]);
    if no_match_ind >= 0 {
        exit_error(&format!(
            "No files match the entry: {}",
            sys_argv[argind + no_match_ind as usize]
        ));
    }

    pass_on_key_interrupt(true);

    // `-n`: the source nices each job (`imodNice` in the vmstopy script, or
    // `nice +n` in the tcsh file); the jobs run in this process now
    if nice != 0 {
        unsafe {
            libc::nice(nice as libc::c_int);
        }
    }

    // Python raises KeyboardInterrupt on SIGINT; `runcmd` passes it on.
    // While a command file runs in this process, the interrupt ends the
    // program the way native's `except KeyboardInterrupt: break` does once
    // the interrupted child has died: with the exit value so far (the
    // `cleanupFiles` there has no temporary file to remove any more).
    static KEY_INTERRUPT: AtomicBool = AtomicBool::new(false);
    static RUNNING: AtomicBool = AtomicBool::new(false);
    static EXIT_VAL: AtomicI32 = AtomicI32::new(0);
    extern "C" fn key_interrupt(_signal: libc::c_int) {
        KEY_INTERRUPT.store(true, Ordering::SeqCst);
        if RUNNING.load(Ordering::SeqCst) {
            unsafe { libc::_exit(EXIT_VAL.load(Ordering::SeqCst)) };
        }
    }
    unsafe {
        libc::signal(
            libc::SIGINT,
            key_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }

    // Loop over the command files
    let mut exit_val = 0;
    for argname in new_args {
        let argname = argname.to_string_lossy().into_owned();
        if KEY_INTERRUPT.load(Ordering::SeqCst) {
            break;
        }

        // Get the full command file name
        let (rootname, ext) = os_path_splitext(&argname);
        let comname;
        if ext.is_empty() || ext == "." {
            let com_exists = Path::new(&format!("{rootname}.com")).exists();
            let pcm_exists = Path::new(&format!("{rootname}.pcm")).exists();
            // (Native leaves the previous file's `submtemp.<pid>` behind at
            // these `exitError`s; there is no temporary file any more.)
            if com_exists && pcm_exists {
                exit_error(&format!(
                    "Both {rootname}.com and {rootname}.pcm exist; specify which"
                ));
            }
            if com_exists {
                comname = format!("{rootname}.com");
            } else if pcm_exists {
                comname = format!("{rootname}.pcm");
            } else {
                exit_error(&format!("Neither {rootname}.com nor {rootname}.pcm exists"));
            }
        } else {
            comname = argname.clone();
        }

        // Get the log file name
        let mut logname = format!("{rootname}.log");
        if log_type > 0 {
            log_type = std::cmp::min(4, log_type);
            let loglist = glob_glob(&format!("{logname}-[0-9]*"));
            let mut lognum: i64 = 1;
            for log in &loglist {
                let logspl = log.split('-').collect::<Vec<_>>();
                // `int(logspl[len(logspl) - 1])` inside a `try`
                if let Some(num) = super::imodpy::py_int(logspl[logspl.len() - 1]) {
                    lognum = std::cmp::max(num + 1, lognum);
                }
            }

            logname += &format!("-{:0width$}", lognum, width = log_type as usize);
        } else if log_type < 0 {
            let d = chrono::Local::now();
            let stamp;
            if log_type < -1 {
                // `d.isoformat()`: microseconds follow only when non-zero
                let mut iso = d.format("%Y-%m-%dT%H:%M:%S").to_string();
                let microsecond = d.timestamp_subsec_micros();
                if microsecond != 0 {
                    iso += &format!(".{microsecond:06}");
                }
                let mut iso_stamp = iso;
                if let Some(msind) = iso_stamp.find('.').filter(|&msind| msind > 0) {
                    iso_stamp.truncate(msind);
                }
                if log_type == -2 {
                    iso_stamp = iso_stamp
                        .replace('-', "")
                        .replace(':', "")
                        .replace('T', "-");
                }
                stamp = iso_stamp;
            } else {
                stamp = d.format("%b-%d-%H%M%S").to_string();
            }

            logname += &format!("-{stamp}.{}", d.timestamp_subsec_micros() / 100000);
        }

        let attempt: Result<(), ImodpyError> = 'attempt: {
            // Changed (owner, 2026-09-26: no Python, no pipes): the command
            // file is run by the in-process runner instead of being converted
            // to `comtmp` with `vmstopy` and run with `python -u` (or, with
            // -s, with `vmstocsh` and `tcsh -ef`).  -s keeps what separated
            // the tcsh path observably: programs' standard error is not
            // logged.  -k is `vmstopy -k`; -n is applied once, before the
            // loop; -t reports the time the way the source's Windows branch
            // does, since there is no `time` command to prefix.
            let options = ComOptions {
                log: Some(logname.clone().into()),
                vmstopy: VmstopyOptions {
                    keep_backslash,
                    ..VmstopyOptions::default()
                },
                stderr_to_log: !use_tcsh,
            };

            // Convert the command file: `vmstopy` failed before `Running ...`
            // was printed, with its message on stdout; the runner converts
            // it again, identically
            let converted = match std::fs::File::open(&comname) {
                Ok(com) if !Path::new(&comname).is_dir() => {
                    super::vmstopy::convert(com, &logname, &options.vmstopy, &mut Vec::new())
                }
                _ => Err(format!("Opening command file {comname}")),
            };
            if let Err(error) = converted {
                prnstr(&format!("ERROR: vmstopy - {error}"), "\n", false);
                break 'attempt Err(set_run_error(
                    vec![format!("vmstopy {comname} {logname}: exited with status 1")],
                    1,
                ));
            }

            // Run it
            if log_type != 0 {
                prnstr(
                    &format!("Running {comname} with log in {logname} ... "),
                    "",
                    false,
                );
            } else {
                prnstr(&format!("Running {comname} ... "), "", false);
            }
            let _ = std::io::stdout().flush();
            let start_time = std::time::Instant::now();

            RUNNING.store(true, Ordering::SeqCst);
            let result = run_com_file(Path::new(&comname), &options);
            RUNNING.store(false, Ordering::SeqCst);
            if result.status != 0 {
                if let Some(error) = &result.error {
                    prnstr(error, "\n", false);
                }
                break 'attempt Err(set_run_error(
                    vec![format!("{comname}: exited with status {}", result.status)],
                    result.status,
                ));
            }
            if dotime {
                prnstr(
                    &format!(
                        "{comname} {message}   in {} sec",
                        super::imodpy::py_fixed(start_time.elapsed().as_secs_f64(), 0, 2)
                    ),
                    "\n",
                    false,
                );
            } else {
                prnstr(&format!("{comname} {message}"), "\n", false);
            }
            Ok(())
        };

        if KEY_INTERRUPT.load(Ordering::SeqCst) {
            break;
        }
        if attempt.is_err() {
            prnstr(&format!("Error executing {comname}{bell}"), "\n", false);
            exit_val = 1;
            EXIT_VAL.store(exit_val, Ordering::SeqCst);

            // If log exists, find ERROR:, and if not do last few lines
            if Path::new(&logname).exists() {
                let Ok(loglines) =
                    read_text_file(&logname, Some(" log file to find ERROR"), false, None)
                else {
                    unreachable!("readTextFile exits on error");
                };
                let mut got_err = 0;
                for l in &loglines {
                    if l.contains("ERROR:") {
                        prnstr(l, "\n", false);
                        got_err = 1;
                    }
                }
                if got_err == 0 {
                    prnstr("   last lines of log:", "\n", false);
                    let ind = std::cmp::max(0, loglines.len() as isize - 4) as usize;
                    for l in &loglines[ind..] {
                        prnstr(l, "\n", false);
                    }
                }
            } else {
                // If no log file, dump any error strings from runcmd; each
                // still carries its line ending in the source
                let err_strings = get_err_strings();
                for l in err_strings {
                    prnstr(&format!("{l}\n"), "\n", false);
                }
            }

            // stop loop unless continuing from errors
            if !cont_if_err {
                break;
            }
        }
    }

    exit_val
}
