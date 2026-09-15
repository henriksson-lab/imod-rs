//! Shared test helpers.
//!
//! The crate builds a single `imod` binary that dispatches to every translated
//! command (see `src/bin/imod.rs`), so there is no longer a
//! `CARGO_BIN_EXE_<command>` per command.  These helpers give the tests the two
//! invocation forms the launcher supports.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

/// A `Command` running `<command>` through the `imod <command> …` subcommand
/// form.  The launcher re-execs itself with `argv[0]` rewritten, so the
/// translated program observes exactly the `argv` it would have as its own
/// executable.
pub fn imod_cmd(command: &str) -> Command {
    let mut assembled = Command::new(env!("CARGO_BIN_EXE_imod"));
    assembled.arg(command);
    // Backend choice is deliberately a child-process concern in integration
    // tests.  A developer's shell selection must not turn an ordinary parity
    // fixture into a different test; tests for a Rust backend set it explicitly
    // on the command they are exercising.
    assembled.env_remove("IMOD_RS_TIFF_BACKEND");
    assembled.env_remove("IMOD_RS_MRC2TIF_ENCODER");
    assembled.env_remove("IMOD_RS_FFT_BACKEND");
    assembled
}

/// Directory holding the per-command symlinks, qualified by process id the way
/// the rest of this suite's fixtures are, so two test binaries never share one.
pub fn command_link_directory() -> PathBuf {
    std::env::temp_dir().join(format!("imod-rs-command-links-{}", std::process::id()))
}

/// Path to a symlink named `<command>` pointing at the `imod` binary — the
/// second invocation form, and the one an IMOD-style install uses.
///
/// Creation tolerates the directory and the link already existing, so repeated
/// calls within a process are safe; `remove_command_links` cleans the tree up.
pub fn imod_link(command: &str) -> PathBuf {
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_imod"));
    let directory = command_link_directory();
    match std::fs::create_dir_all(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("create {}: {error}", directory.display()),
    }
    let link = directory.join(command);
    match std::os::unix::fs::symlink(&binary, &link) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("link {}: {error}", link.display()),
    }
    link
}

/// A `Command` running `<command>` through its symlink.
pub fn imod_link_cmd(command: &str) -> Command {
    let mut assembled = Command::new(imod_link(command));
    assembled.env_remove("IMOD_RS_TIFF_BACKEND");
    assembled.env_remove("IMOD_RS_MRC2TIF_ENCODER");
    assembled.env_remove("IMOD_RS_FFT_BACKEND");
    assembled
}

/// Removes the symlink directory this process created.
pub fn remove_command_links() {
    let _ = std::fs::remove_dir_all(command_link_directory());
}
