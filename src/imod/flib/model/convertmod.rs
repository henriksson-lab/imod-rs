//! Translation of `IMOD/flib/model/convertmod.f`.
#![allow(dead_code)]

use std::fs::OpenOptions;

use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::libimod::imodel_to::imod_to_wmod;

/// Original program: `convertmod` (`convertmod.f:1`).
pub fn convertmod() {
    // `setExitPrefix` affects errors emitted by the old C/Fortran runtime.
    // Rust command diagnostics carry the equivalent source prefix directly.
    let (oldfile, newfile) = match getinout(2) {
        Ok(files) if !files.0.is_empty() && !files.1.is_empty() => files,
        _ => {
            eprintln!("ERROR: CONVERTMOD - Error getting input/output file names");
            std::process::exit(1);
        }
    };
    let imod = match readw_or_imod(&oldfile) {
        Ok(imod) => imod,
        Err(()) => {
            // `convertmod.f:16` is a direct Fortran `print *`; the prefix
            // installed for library errors does not apply to this stdout line.
            println!(" Error reading mode file");
            std::process::exit(1);
        }
    };
    let mut output = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&newfile)
    {
        Ok(file) => file,
        Err(_) => {
            eprintln!("ERROR: CONVERTMOD - Error 11 writing to WIMP model file");
            std::process::exit(1);
        }
    };
    // `imodWriteAsWimp` invokes `imod_to_wmod` after its native model bridge
    // has been populated.  `readw_or_imod` already provides that native Imod.
    if let Err(error) = imod_to_wmod(&imod, &mut output, &newfile) {
        eprintln!("ERROR: CONVERTMOD - Error {error} writing to WIMP model file");
        std::process::exit(1);
    }
}
