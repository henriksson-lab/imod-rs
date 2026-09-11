//! Translation of `IMOD/flib/model/convertmod.f`.
#![allow(dead_code)]

use std::fs::OpenOptions;

use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::store_mod::store_mod;
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
    // `use fortmodel` — the module arrays `readw_or_imod` fills.
    let mut fm = FortModel::default();
    // `if(.not.readw_or_imod(oldfile))`
    let model = match readw_or_imod(&oldfile, &mut fm) {
        Ok(model) => model,
        Err(()) => {
            // `convertmod.f:16` is a direct Fortran `print *`; the prefix
            // installed for library errors does not apply to this stdout line.
            println!(" Error reading mode file");
            std::process::exit(1);
        }
    };
    // `close(20)` / `ierr = imodWriteAsWimp(newfile)`
    let ierr: i32 = match model {
        Some(mut imod) => {
            // `imodWriteAsWimp` runs `writeimod` before `imod_to_wmod`.
            // Reverse the `openImodData` reference-coordinate transform with
            // the same f32 matrix-operation order from
            // `imodel_fwrap.c:1567-1592`.
            if let Some(reference) = imod.ref_image {
                let scale_x = (1.0_f64 / reference.cscale.x as f64) as f32;
                let scale_y = (1.0_f64 / reference.cscale.y as f64) as f32;
                let scale_z = (1.0_f64 / reference.cscale.z as f64) as f32;
                for object in &mut imod.obj {
                    for contour in &mut object.cont {
                        for point in &mut contour.pts {
                            point.x = point.x * scale_x + reference.ctrans.x * scale_x;
                            point.y = point.y * scale_y + reference.ctrans.y * scale_y;
                            point.z = point.z * scale_z + reference.ctrans.z * scale_z;
                        }
                    }
                }
            }
            if imod.flags & crate::imod::libimod::imodel::IMODF_FLIPYZ != 0 {
                crate::imod::libimod::imodel::imod_flip_yz(&mut imod);
            }
            let mut output = match OpenOptions::new()
                .write(true)
                // `imodel_fwrap.c:1760` opens the destination with `"wb"`,
                // which creates it or truncates an existing WIMP file.
                .create(true)
                .truncate(true)
                .open(&newfile)
            {
                Ok(file) => file,
                // `sImod->file` is NULL and `imod_to_wmod` reports its write
                // failure code.
                Err(_) => {
                    println!("  Error{:12} writing to WIMP model file", 11);
                    std::process::exit(1);
                }
            };
            match imod_to_wmod(&imod, &mut output, &newfile) {
                Ok(()) => 0,
                Err(error) => error,
            }
        }
        // `writeimod` returns `FWRAP_ERROR_NO_MODEL` (`imodel_fwrap.c:31`)
        // when `openImodData` never opened a model, which is exactly the WIMP
        // text branch of `readw_or_imod`.
        None => -5,
    };
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
                std::process::exit(2);
            }
        };
        store_mod(&mut unit20, &newfile, &mut fm);
    } else if ierr != 0 {
        // `print *,'Error', ierr,' writing to WIMP model file'`
        println!("  Error{ierr:12} writing to WIMP model file");
        std::process::exit(1);
    }
    std::process::exit(0);
}
