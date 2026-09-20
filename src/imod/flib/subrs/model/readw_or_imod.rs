//! Model-format dispatcher from `IMOD/flib/subrs/model/readw_or_imod.f`.
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::imod::flib::subrs::model::fortmodel::{FortModel, allocate_fort_model};
use crate::imod::flib::subrs::model::read_mod::read_mod;
use crate::imod::libimod::imodel::{IMODF_FLIPYZ, Imod, imod_flip_yz};
use crate::imod::libimod::imodel_files::imod_read;

/// Original: `readw_or_imod` (`readw_or_imod.f:11`).
///
/// `Ok(Some(model))` is the `openImodData` branch, where the C model stays
/// open in the fortran wrapper's `sImod`; `Ok(None)` is the WIMP branch, where
/// only the `fortmodel` module arrays in `fm` are filled and `sImod` stays
/// null.  `convertmod` needs that distinction because `imodWriteAsWimp`
/// returns `FWRAP_ERROR_NO_MODEL` for the second case.
pub fn readw_or_imod(path: impl AsRef<Path>, fm: &mut FortModel) -> Result<Option<Imod>, ()> {
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
            // `getimod` also fills the `fortmodel` arrays from the open model.
            // Nothing in this branch reads them back before `writeimod`, so
            // the copy is not translated here.
            allocate_fort_model(fm);
            Ok(Some(imod))
        }
        Err(_) => {
            // `call allocateFortModel()` then `open(20,file=filename,
            // status='old',err=20)`.
            allocate_fort_model(fm);
            let file = match File::open(path.as_ref()) {
                Ok(file) => file,
                // `err=20` closes unit 20 and returns `.false.`
                Err(_) => return Err(()),
            };
            // The `qopen`/`qread` branch above label `10` recognises the old
            // *binary* WIMP container.  No `qopen` layer exists in this crate,
            // and every text WIMP file fails its `int2(1).ne.3` test on the
            // first record, so the source's `go to 10` fall-through to
            // `read_mod` is what is translated here.
            let mut unit20 = BufReader::new(file);
            if read_mod(&mut unit20, fm) {
                Ok(None)
            } else {
                Err(())
            }
        }
    }
}

/// Original: `getModelObjectRange` (`readw_or_imod.f:142`).
///
/// Once a WIMP model has been opened, this routine fills the model arrays with
/// contour data just for the objects ranging from `iobj_strt` to `iobj_end`.
/// Returns `false` for error.
///
/// Gap: the `getimodobjrange` entry point of `IMOD/libimod/imodel_fwrap.c`,
/// and the partial-mode model handle it reads, are not translated yet, so the
/// call cannot be issued and this unit reports the error the source reports
/// for a non-zero `ierr`.  No translated caller uses it.
pub fn get_model_object_range(iobj_strt: i32, iobj_end: i32, fm: &mut FortModel) -> bool {
    let _ = (iobj_strt, iobj_end);
    let ierr: i32 = -1;
    let get_model_object_range = ierr == 0;
    if ierr == 0 {
        complete_model_values(fm);
    }
    get_model_object_range
}

/// Original: `getModelObjectList` (`readw_or_imod.f:158`).
///
/// Once a WIMP model has been opened, this routine fills the model arrays with
/// contour data just for the list of `nin_list` objects in `iobj_list`.
/// Returns `false` for error.
///
/// Gap: as for `get_model_object_range`, the `getimodobjlist` entry point of
/// `IMOD/libimod/imodel_fwrap.c` is not translated yet.  No translated caller
/// uses it.
pub fn get_model_object_list(iobj_list: &[i32], nin_list: i32, fm: &mut FortModel) -> bool {
    let _ = (iobj_list, nin_list);
    let ierr: i32 = -1;
    let get_model_object_list = ierr == 0;
    if ierr == 0 {
        complete_model_values(fm);
    }
    get_model_object_list
}

/// Original: `completeModelValues` (`readw_or_imod.f:170`).
pub fn complete_model_values(fm: &mut FortModel) {
    let mut i: i32 = 1;
    while i <= fm.n_point {
        fm.object[i as usize - 1] = i;
        i += 1;
    }
    i = 1;
    while i <= fm.n_object {
        fm.ndx_order[i as usize - 1] = i;
        fm.obj_order[i as usize - 1] = i;
        i += 1;
    }
    fm.max_mod_obj = fm.n_object;
    fm.ntot_in_obj = fm.n_point;
    i = fm.max_mod_obj + 1;
    while i <= fm.max_obj_num {
        fm.npt_in_obj[i as usize - 1] = 0;
        i += 1;
    }
    fm.n_clabel = 0;
    fm.ibase_free = 0;
    if fm.n_object > 0 {
        fm.ibase_free =
            fm.ibase_obj[fm.n_object as usize - 1] + fm.npt_in_obj[fm.n_object as usize - 1];
    }
    fm.nin_order = fm.n_object;
}
