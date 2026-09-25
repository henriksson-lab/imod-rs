//! Translation of `IMOD/flib/subrs/model/write_wmod.f`.
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::imod_backup_file;
use crate::imod::libimod::imodel_fwrap::{putimod, writeimod};

/// Original: `write_wmod` (`write_wmod.f:13`).
///
/// Writes the current model as an IMOD model to the file `modelfile`.  If
/// `n_point` in the model arrays is negative, it will not put contour data in
/// the model arrays into the model before saving.  `write_wmod` itself does
/// not `use fortmodel`; `fm` is passed through to `putModelObjects`, which
/// does.
pub fn write_wmod(modelfile: &str, fm: &mut FortModel) {
    let mut ierr: i32;
    //
    ierr = imod_backup_file(modelfile);
    if ierr != 0 {
        // `write(6,*)` of two adjacent character items: gfortran's
        // list-directed output puts no separator between them.
        println!(" WARNING: write_wmod - Error attempting to rename existing model file");
    }

    put_model_objects(fm);
    ierr = writeimod(modelfile);
    if ierr != 0 {
        println!(" ERROR: write_wmod - writing output model file");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
}

/// Original: `putModelObjects` (`write_wmod.f:36`).
///
/// Puts contour data in the model arrays back into the model, unless
/// `n_point` is negative.  In partial mode, only the objects containing data
/// in the arrays will be modified in the IMOD model; otherwise any existing
/// objects in the IMOD model will be removed.
pub fn put_model_objects(fm: &mut FortModel) {
    let ierr: i32;
    //
    // Set n_point to negative to signal that objects are already put out
    //
    if fm.n_point >= 0 {
        ierr = putimod(
            &fm.ibase_obj,
            &fm.npt_in_obj,
            &fm.p_coord,
            &fm.object,
            &fm.obj_color,
            fm.n_point,
            fm.max_mod_obj,
        );
    } else {
        ierr = 0;
    }
    if ierr != 0 {
        println!(" ERROR: putModelObjects - putting objects back into IMOD model");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
}
