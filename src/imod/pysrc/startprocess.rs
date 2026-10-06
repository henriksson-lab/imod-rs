//! Translation of `IMOD/pysrc/startprocess`: runs a process in background with
//! redirection of output/error.
//!
//! A Python command script with no functions; its top level is
//! [`startprocess`].  `bkgdProcess` is `imodpy::bkgd_process`; the command
//! runs as a detached child, as in the source, because it must outlive this
//! process (it is how eTomo starts `processchunks` and command files).

use super::imodpy::{add_imod_bin_ignore_sighup, bkgd_process, prnstr};
use super::pip::{exit_error, set_exit_prefix};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`startprocess:1-84`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn startprocess(arguments: &[OsString]) -> i32 {
    let progname = "startprocess";
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

    set_exit_prefix(prefix);

    if argv.len() < 2 {
        prnstr(
            &format!(
                "Usage: {progname} [-o outfile] [-e errfile] [-a] [-d dir] command arguments...
   Runs a process in the background with optional redirection of the output
      outfile is a file for standard output
      errfile is a file for standard error, or 'None' for no redirection
      -a option will make it append to these files if they exist
      dir is a working directory to set before running the command; errfile
          and outfile are relative to this directory
   The default is no redirection of standard output, and redirection of
      standard error into standard output (use '-e None' to override)"
            ),
            "\n",
            false,
        );
        let _ = std::io::stdout().flush();
        return 0;
    }

    let mut argind = 1;
    let mut outfile: Option<String> = None;
    let mut errfile: Option<String> = Some("stdout".to_owned());
    let mut workdir: Option<String> = None;
    let mut do_append = false;
    while argind < argv.len() - 1 {
        if argv[argind] == "-o" {
            outfile = Some(argv[argind + 1].clone());
            argind += 2;
            continue;
        } else if argv[argind] == "-e" {
            errfile = Some(argv[argind + 1].clone());
            argind += 2;
            if errfile.as_deref() == Some("None") {
                errfile = None;
            }
            continue;
        }
        if argv[argind] == "-d" {
            workdir = Some(argv[argind + 1].clone());
            argind += 2;
            continue;
        }
        if argv[argind] == "-a" {
            do_append = true;
            argind += 1;
            continue;
        } else {
            break;
        }
    }

    if argind >= argv.len() {
        exit_error("No command was included");
    }

    // `if workdir:` -- an empty string is false.
    if let Some(dir) = workdir.as_deref().filter(|dir| !dir.is_empty()) {
        if std::env::set_current_dir(dir).is_err() {
            exit_error(&format!("Changing working directory to {dir}"));
        }
    }

    // `bkgdProcess` without `returnOnErr` calls `exitError` itself.
    let _ = bkgd_process(
        &arguments[argind..],
        outfile.as_deref(),
        errfile.as_deref(),
        false,
        do_append,
    );
    let _ = std::io::stdout().flush();
    0
}
