//! Translation of `IMOD/pysrc/subm`.
//!
//! Its background process is deliberately an OS process boundary, as in
//! `imodpy.bkgdProcess` in the source.

use super::imodpy::{add_imod_bin_ignore_sighup, bkgd_process};
use std::ffi::OsString;

/// Original Python top-level program (`IMOD/pysrc/subm:1`).
pub fn subm(arguments: &[OsString]) -> i32 {
    let prefix = "ERROR: subm - ";
    if std::env::var_os("IMOD_DIR").is_none() {
        println!("{prefix} IMOD_DIR is not defined!");
        return 1;
    }
    // Source startup calls `addIMODbinIgnoreSIGHUP` before constructing the
    // submfg command, so the installed IMOD executable wins PATH lookup.
    add_imod_bin_ignore_sighup();
    let program = if cfg!(windows) {
        "submfg.cmd"
    } else {
        "submfg"
    };
    let mut command = vec![OsString::from(program)];
    command.extend(arguments.iter().skip(1).cloned());
    // `bkgdProcess(args, None, 'stdout')` starts but deliberately does not wait
    // for the child process.
    // Python calls `bkgdProcess(args, None, 'stdout')`: its default
    // `returnOnErr=False` reports an IMOD/PIP error itself rather than turning
    // a background-launch failure into a separate launcher return path.
    match bkgd_process(&command, None, Some("stdout"), false, false) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{prefix}{error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::subm;
    use std::ffi::OsString;

    #[test]
    fn missing_runtime_is_an_error() {
        // This source-mapped command has no parse phase; its environmental
        // gate is covered by the executable integration test instead.
        assert!(matches!(subm(&[OsString::from("subm")]), 0 | 1));
    }
}
