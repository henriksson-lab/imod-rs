//! Single-binary busybox-style launcher for every translated IMOD command.
//!
//! Upstream IMOD installs one executable per command.  This crate deliberately
//! deviates: it builds exactly one binary, `imod`, which dispatches to the
//! translated program unit.  Dispatch is on `basename(argv[0])` first, so an
//! IMOD-style install that symlinks (or hardlinks) `header`, `newstack`, … to
//! `imod` gives each program the same `argv` it has always had — same PIP
//! parsing, same `imodProgName`-derived error prefixes.
//!
//! When `argv[0]` does not name a command, `argv[1]` is taken as the
//! subcommand (`imod header -size f.mrc`).  That form dispatches in-process:
//! the launcher records the `argv` the command would have had as its own
//! executable — `["<bindir>/header", "-size", "f.mrc"]` — through
//! `b3dutil::set_program_args`, and every translated unit reads `argv` through
//! `b3dutil::program_args`/`program_args_os` rather than `std::env::args()`,
//! so it never sees the `imod` wrapper.  Nothing is re-exec'd: the command
//! pays process start-up and dynamic linking once.

use imod_rs::imod::commands::{COMMANDS, find};
use imod_rs::imod::libcfshr::b3dutil::set_program_args;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Runs the named command, or returns `false` when the name is not a command.
///
/// The table itself -- names, usage order and entry points -- lives in the
/// library (`imod_rs::imod::commands`), shared with `imodpy::run_cmd`'s
/// in-process runner.  An entry point returns (status 0) or ends in
/// `b3dutil::exit`, which outside the in-process runner is
/// `std::process::exit`.
///
/// **Signal disposition (process boundary).**  The Rust runtime starts every
/// binary with `SIGPIPE` ignored, so a write to a closed pipe returns `EPIPE`:
/// Rust's `print!` then panics (exit 101) and libc `printf` carries on.  A
/// C, C++ or Fortran program -- what every `in_process` command translates --
/// runs with the default disposition and is killed by the signal (a shell sees
/// status 141, `imod header f.mrc | head -1`).  So the default is restored here,
/// once, before the command runs.  The Python-script translations and the
/// other non-`in_process` commands keep `SIG_IGN`, which is what the Python
/// interpreter itself installs at start-up.  Children spawned through
/// `std::process::Command` are unaffected either way: std resets `SIGPIPE` to
/// the default in the child before `exec`, as Python's `subprocess` does.
fn dispatch(name: &str) -> bool {
    match find(name) {
        Some(command) => {
            if command.in_process {
                // SAFETY: `signal` with a valid signal number and `SIG_DFL` has
                // no memory-safety preconditions; it runs before the command
                // starts any thread.
                #[cfg(unix)]
                unsafe {
                    libc::signal(libc::SIGPIPE, libc::SIG_DFL)
                };
            }
            (command.entry)();
            true
        }
        None => false,
    }
}

/// Prints the command listing.  Used for no arguments, `-h`/`--help`, and an
/// unrecognised subcommand alike; the caller exits 1 in every case.
fn usage(launcher: &str) {
    eprintln!("Usage: {launcher} <command> [options ...]");
    eprintln!();
    eprintln!(
        "Every IMOD command translated by this crate is built into this single\n\
         binary.  Run one either as a subcommand, as above, or through a link\n\
         named after it (an IMOD-style install links each command name to this\n\
         binary, and the command then behaves exactly as its own executable)."
    );
    eprintln!();
    eprintln!("Commands:");
    for command in COMMANDS {
        eprintln!("  {}", command.name);
    }
}

fn main() {
    let argv0 = std::env::args_os().next().unwrap_or_default();

    // 1. `basename(argv[0])` names a command: run it with `argv` untouched.
    //    On Windows the executable's name ends in `.exe`, which is not part of
    //    the command name (`imodProgName` strips it the same way).
    let base_name = |path: &Path| {
        if cfg!(windows) {
            path.file_stem()
        } else {
            path.file_name()
        }
        .and_then(|n| n.to_str())
        .map(str::to_owned)
    };
    if let Some(base) = base_name(Path::new(&argv0))
        && dispatch(&base)
    {
        return;
    }

    let launcher = base_name(Path::new(&argv0)).unwrap_or_else(|| "imod".to_string());

    // 2. `argv[1]` names a command: run it in-process with the `argv` it would
    //    have had as its own binary.
    let arguments: Vec<OsString> = std::env::args_os().collect();
    let subcommand = match arguments.get(1).and_then(|a| a.to_str()) {
        Some(name) if find(name).is_some() => name.to_string(),
        _ => {
            usage(&launcher);
            std::process::exit(1);
        }
    };

    // The rewritten `argv[0]` keeps the directory of the running binary, so it
    // is the path a command link would have: `<bindir>/newstack`, not a bare
    // name.  That is what `imodProgName` and PIP's program name see.
    let new_argv0: PathBuf = match std::env::current_exe() {
        Ok(path) => path.with_file_name(&subcommand),
        Err(_) => PathBuf::from(&subcommand),
    };
    let mut argv: Vec<OsString> = Vec::with_capacity(arguments.len() - 1);
    argv.push(new_argv0.into_os_string());
    argv.extend(arguments.into_iter().skip(2));
    set_program_args(argv);
    dispatch(&subcommand);
}
