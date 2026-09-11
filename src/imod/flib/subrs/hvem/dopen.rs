//! Translation of `IMOD/flib/subrs/hvem/dopen.f`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::imod_backup_file;
use std::ffi::CString;
use std::fs::File;

/// Source `common /hushcom/hush` with `data hush /.false./` (`dopen.f:32`).
/// A Fortran blank-common cell has no Rust analogue, so the flag is a module
/// static holding exactly the same one value that `DOPEN` and `dopenHush`
/// share.
static mut S_HUSH: bool = false;

/// Original `DOPEN` (`dopen.f:20`).
///
/// The Fortran unit table has no Rust analogue: `iunit` is retained as the
/// source's first argument and the connected file is returned to the caller
/// instead of being registered under that unit number.
pub fn dopen(iunit: i32, fname: &str, itype: &str, iform: &str) -> File {
    let _ = iunit;
    //
    // if 'NEW' and file exists, rename to file~
    // DNM 10/20/03: changed to call subroutine, and clean up old stuff
    //
    if itype == "NEW" || itype == "new" {
        let name = CString::new(fname.as_bytes()).unwrap_or_default();
        let ierr = unsafe { imod_backup_file(name.as_ptr()) };
        if ierr != 0 {
            println!("\nWARNING: DOPEN - Error attempting to rename existing file{fname}");
        }
    }
    //
    // check existence of old file to get a nice error message
    //
    if itype == "OLD" || itype == "old" || itype == "RO" || itype == "ro" {
        let exist = std::path::Path::new(fname).exists();
        if !exist {
            println!("\nERROR: DOPEN - FILE {fname} DOES NOT EXIST");
            std::process::exit(1);
        }
    }
    //
    // open old 'P' as FORMATTED
    //
    let mut format = "FORMATTED";
    if iform == "U" || iform == "u" {
        format = "UNFORMATTED";
    }
    //
    // Treat RO as OLD, leave RO in the message for log file stability
    //
    let opened = if itype == "RO" || itype == "ro" || itype == "OLD" || itype == "old" {
        File::open(fname)
    } else if itype == "SCRATCH" || itype == "scratch" {
        File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(fname)
    } else {
        File::create(fname)
    };
    let Ok(file) = opened else {
        println!("\nERROR: DOPEN - CANNOT OPEN FILE {fname}");
        std::process::exit(1);
    };
    //
    // NOW WRITE OUT FILE INFO
    //
    if unsafe { S_HUSH } {
        return file;
    }
    // `INQUIRE (FILE=FNAME,NAME=FULLNAM)` returns the name as it was given.
    let fullnam = fname;
    // FORMAT(/,1x,A,2X,A,'  file opened: ',A)
    println!("\n {format}  {itype}  file opened: {fullnam}");
    file
}

/// Original `dopenHush` (`dopen.f:78`).
pub fn dopen_hush(value: bool) {
    unsafe { S_HUSH = value };
}
