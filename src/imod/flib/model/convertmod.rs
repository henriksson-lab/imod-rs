//! Translation of `IMOD/flib/model/convertmod.f`.
use std::fs::OpenOptions;

use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::store_mod::store_mod;
use crate::imod::libimod::imodel_fwrap::imodwriteaswimp;

/// Original program: `convertmod` (`convertmod.f:1`).
pub fn convertmod() {
    // `setExitPrefix` affects errors emitted by the old C/Fortran runtime.
    // Rust command diagnostics carry the equivalent source prefix directly.
    let (oldfile, newfile) = match getinout(2) {
        Ok(files) if !files.0.is_empty() && !files.1.is_empty() => files,
        _ => {
            eprintln!("ERROR: CONVERTMOD - Error getting input/output file names");
            crate::imod::libcfshr::b3dutil::exit(1);
        }
    };
    // `use fortmodel` — the module arrays `readw_or_imod` fills.
    let mut fm = FortModel::default();
    // `if(.not.readw_or_imod(oldfile))`
    if !readw_or_imod(&oldfile, &mut fm) {
        // `convertmod.f:16` is a direct Fortran `print *`; the prefix
        // installed for library errors does not apply to this stdout line.
        println!(" Error reading mode file");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    //
    // `close(20)` is the end of `readw_or_imod`'s unit 20 there.
    // `ierr = imodWriteAsWimp(newfile)`: `writeimod` returns
    // `FWRAP_ERROR_NO_MODEL` (-5) when `openImodData` never opened a model,
    // which is exactly the WIMP branch of `readw_or_imod`.
    let ierr: i32 = imodwriteaswimp(&newfile);
    if ierr == -5 {
        // `open(20,file=newfile,status='new',form='formatted')` — `status=
        // 'new'` fails when the file already exists, and the source supplies
        // no `err=` branch, so gfortran terminates with status 2.  The
        // address-bearing backtrace it prints after these two lines is not
        // reproducible.
        let mut unit20 = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&newfile)
        {
            Ok(file) => file,
            Err(_) => {
                eprintln!("At line 23 of file convertmod.f (unit = 20)");
                eprintln!("Fortran runtime error: Cannot open file '{newfile}': File exists");
                crate::imod::libcfshr::b3dutil::exit(2);
            }
        };
        store_mod(&mut unit20, &newfile, &mut fm);
    } else if ierr != 0 {
        // `print *,'Error', ierr,' writing to WIMP model file'`
        println!("  Error{ierr:12} writing to WIMP model file");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    crate::imod::libcfshr::b3dutil::exit(0);
}
