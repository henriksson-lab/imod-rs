//! Model-format dispatcher from `IMOD/flib/subrs/model/readw_or_imod.f`.

use std::path::Path;

use crate::imod::flib::subrs::model::read_mod::read_mod;
use crate::imod::libimod::imodel::{IMODF_FLIPYZ, Imod, imod_flip_yz};
use crate::imod::libimod::imodel_files::imod_read;

/// Original: `readw_or_imod` (`readw_or_imod.f:11`).
pub fn readw_or_imod(path: impl AsRef<Path>) -> Result<Imod, ()> {
    // `openImodData` is first in the source.  Its native replacement reads
    // the same binary V1.2 model container.  On failure the Fortran routine
    // falls through to its WIMP reader and eventually `read_mod`.
    match imod_read(path.as_ref()) {
        Ok(mut imod) => {
            // `openImodData` (`imodel_fwrap.c:468-491`) first puts a model
            // into the Fortran bridge's identity reference coordinates.  Its
            // matching `writeimod` path reverses this before `imod_to_wmod`;
            // retaining both f32 transforms is observable at decimal-format
            // boundaries in `convertmod` output.
            if imod.flags & IMODF_FLIPYZ != 0 {
                imod_flip_yz(&mut imod);
            }
            if let Some(reference) = imod.ref_image {
                for object in &mut imod.obj {
                    for contour in &mut object.cont {
                        for point in &mut contour.pts {
                            point.x = point.x * reference.cscale.x - reference.ctrans.x;
                            point.y = point.y * reference.cscale.y - reference.ctrans.y;
                            point.z = point.z * reference.cscale.z - reference.ctrans.z;
                        }
                    }
                }
            }
            Ok(imod)
        }
        Err(_) => read_mod(path),
    }
}
