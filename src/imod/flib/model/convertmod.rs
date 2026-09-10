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
    let mut imod = match readw_or_imod(&oldfile) {
        Ok(imod) => imod,
        Err(()) => {
            // `convertmod.f:16` is a direct Fortran `print *`; the prefix
            // installed for library errors does not apply to this stdout line.
            println!(" Error reading mode file");
            std::process::exit(1);
        }
    };
    // `imodWriteAsWimp` calls `writeimod` before `imod_to_wmod`.  Reverse
    // the `openImodData` reference-coordinate transform with the same f32
    // matrix-operation order from `imodel_fwrap.c:1567-1592`.
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
        // `imodel_fwrap.c:1762` opens the destination with `"wb"`, which
        // creates it or truncates an existing WIMP file.
        .create(true)
        .truncate(true)
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
