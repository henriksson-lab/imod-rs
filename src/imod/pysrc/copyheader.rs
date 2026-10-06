//! Translation of `IMOD/pysrc/copyheader`: copies the standard and extended
//! header of an MRC file to another file.
//!
//! A Python command script with no functions; its top level is
//! [`copyheader`].  The `header` process the script ran is
//! `imodpy::header_in_process` (owner decision 2026-09-24), whose output
//! carries the `Space group,# extra bytes` line the script reads.

use super::imodpy::{
    add_imod_bin_ignore_sighup, exit_from_imod_error, get_image_format, header_in_process,
    make_backup_file, prnstr, py_int,
};
use super::pip::{exit_error, set_exit_prefix};
use std::ffi::OsString;
use std::io::{Read as _, Write as _};
use std::path::Path;

/// The script's top level (`copyheader:1-94`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn copyheader(arguments: &[OsString]) -> i32 {
    let progname = "copyheader";
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

    // Get file names, check existence
    if argv.len() < 3 {
        prnstr(
            "Usage: copyheader inputFile outputFile\n    Copies standard and extended header of inputFile to outputFile",
            "\n",
            false,
        );
        return done(0);
    }

    let in_name = argv[1].clone();
    let out_name = argv[2].clone();

    if !Path::new(&in_name).exists() {
        exit_error(&format!("Input file {in_name} does not exist"));
    }

    // Get a full header and make sure it is MRC, and find # of extra bytes
    let head_lines = (|| {
        if get_image_format(&in_name)? != "MRC" {
            exit_error("This is not an MRC file; cannot copy header");
        }
        header_in_process(&format!("header \"{in_name}\""), &in_name, false, None)
    })()
    .unwrap_or_else(|_| exit_from_imod_error(progname));

    let mut num_bytes: i64 = 1024;
    for line in &head_lines {
        if line.contains("extra bytes") {
            let mut num_start = line.find(".  ").map_or(-1, |index| index as i64) + 2;
            if num_start > 2 {
                let lsplit: Vec<&str> = line[num_start as usize..].split_whitespace().collect();
                match lsplit.get(1).and_then(|text| py_int(text)) {
                    Some(value) => num_bytes += value,
                    None => num_start = 0,
                }
            }
            if num_start < 3 {
                exit_error("Finding number of extra header bytes in output from header");
            }
        }
    }

    // `str(sys.exc_info()[1])` of an OSError
    let exc_text = |error: &std::io::Error, name: Option<&str>| -> String {
        let text = error.to_string();
        match error.raw_os_error() {
            Some(errno) => {
                let strerror = text
                    .strip_suffix(&format!(" (os error {errno})"))
                    .unwrap_or(&text)
                    .to_owned();
                match name {
                    Some(name) => format!("[Errno {errno}] {strerror}: '{name}'"),
                    None => format!("[Errno {errno}] {strerror}"),
                }
            }
            None => text,
        }
    };

    // Open and read the necessary bytes
    let mut action = "Opening";
    let header: Vec<u8> = (|| {
        let in_file = std::fs::File::open(&in_name).map_err(|e| exc_text(&e, Some(&in_name)))?;
        action = "Reading from";
        let mut header = Vec::new();
        in_file
            .take(num_bytes.max(0) as u64)
            .read_to_end(&mut header)
            .map_err(|e| exc_text(&e, None))?;
        Ok::<_, String>(header)
    })()
    .unwrap_or_else(|message| exit_error(&format!("{action} input file {in_name} - {message}")));

    // Backup output file, open, and write the bytes.  That's it.
    make_backup_file(&out_name);

    let mut action = "Opening";
    let written: Result<(), String> = (|| {
        let mut out_file =
            std::fs::File::create(&out_name).map_err(|e| exc_text(&e, Some(&out_name)))?;
        action = "Writing to";
        out_file
            .write_all(&header)
            .map_err(|e| exc_text(&e, None))?;
        Ok(())
    })();
    if let Err(message) = written {
        exit_error(&format!("{action} output file {out_name} - {message}"));
    }

    done(0)
}
