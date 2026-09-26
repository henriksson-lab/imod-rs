//! WIMP-style model-array bridge from `IMOD/libimod/fortmodel.c`.
//!
//! The arrays themselves are the translated `fortmodel.f90` common state,
//! represented by [`FortModel`].  This source unit supplies the C helpers
//! which connect that state to `imodel_fwrap` and keep its sparse object list
//! packed.  Keeping one owned Rust record avoids the C version's mutable
//! process-global pointers without changing the one-based array convention.

pub use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::imod_backup_file;
use crate::imod::libiimod::unit_header::{iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt};
use crate::imod::libimod::imodel_fwrap::{
    getimod, getimodhead, getimodscales, imodcountcontspoints, imodhasimageref, imodopenerror,
    openimoddata, putimageref, putimod, writeimod,
};

/// Original: `allocateFortModel` (`fortmodel.c:119`).
///
/// The common storage is shared with translated Fortran program units, so its
/// allocation implementation lives there; this named bridge is the C source
/// entry point and keeps the source-unit API complete.
pub fn allocate_fort_model(fm: &mut FortModel) {
    crate::imod::flib::subrs::model::fortmodel::allocate_fort_model(fm);
}

/// Original: `completeModelValues` (`fortmodel.c:246`).
pub fn complete_model_values(fm: &mut FortModel) {
    for point in 0..fm.n_point.max(0) as usize {
        fm.object[point] = point as i32 + 1;
    }
    for object in 0..fm.n_object.max(0) as usize {
        fm.ndx_order[object] = object as i32 + 1;
        fm.obj_order[object] = object as i32 + 1;
    }
    fm.max_mod_obj = fm.n_object;
    fm.ntot_in_obj = fm.n_point;
    for count in fm
        .npt_in_obj
        .iter_mut()
        .skip(fm.max_mod_obj.max(0) as usize)
    {
        *count = 0;
    }
    fm.ibase_free = if fm.n_object > 0 {
        let object = fm.n_object as usize - 1;
        fm.ibase_obj[object] + fm.npt_in_obj[object]
    } else {
        0
    };
    fm.nin_order = fm.n_object;
}

/// Original: `fortModObjToCont` (`fortmodel.c:264`).
///
/// Converts a WIMP-style object number in `iobj`, which is actually a contour
/// number within the whole model, numbered from 1, to an IMOD object and
/// contour number, also numbered from 1.  The source indexes
/// `fmodObj_color[iobj * 2 - 1]`, i.e. `obj_color(2, iobj)` — the IMOD-object
/// color code, `obj_color[iobj - 1][1]` here, not the always-1 first column.
pub fn fort_mod_obj_to_cont(
    iobj: i32,
    fmod_obj_color: &[[i32; 2]],
    imodobj: &mut i32,
    imodcont: &mut i32,
) {
    let icolor: i32 = fmod_obj_color[(iobj - 1) as usize][1];
    *imodobj = 256 - icolor;
    *imodcont = 0;
    let mut i = 1;
    while i <= iobj {
        if icolor == fmod_obj_color[(i - 1) as usize][1] {
            *imodcont += 1;
        }
        i += 1;
    }
}

/// Original: `fortObjectMover` (`fortmodel.c:289`).  Returns `true` for the
/// native `failed != 0` result.
///
/// Line for line with the C: no bounds checks of its own (an out-of-range
/// `iobj` is an index panic here where the C reads out of bounds), and the
/// `insufficient space` message when the repack does not help.
pub fn fort_object_mover(fm: &mut FortModel, iobj: i32) -> bool {
    //
    // just return if object already at top of list
    //
    let mut failed;
    if fm.nin_order > 0 {
        if fm.obj_order[(fm.nin_order - 1) as usize] == iobj {
            return false;
        }
    }
    //
    // if the order array is full or there's not enough space in the object
    // array, first try to repack arrays, then give up if still too full
    //
    let num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
    failed = fm.nin_order >= fm.max_obj_order || fm.ibase_free + num_in_obj + 4 >= fm.len_object;
    if failed {
        fort_object_packer(fm);
    }
    failed = fm.nin_order >= fm.max_obj_order || fm.ibase_free + num_in_obj + 4 >= fm.len_object;
    if failed {
        print!("fortObjectMover: insufficient space to make desired object current\n");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        return true;
    }
    //
    // if object is not null, move to top of OBJECT and adjust things;
    // otherwise assume nothing is set up, not even base pointer
    //
    if num_in_obj > 0 {
        let ibase = fm.ibase_obj[(iobj - 1) as usize];
        fm.ibase_obj[(iobj - 1) as usize] = fm.ibase_free;
        let mut ind = ibase + 1;
        while ind <= ibase + num_in_obj {
            fm.ibase_free += 1;
            fm.object[(fm.ibase_free - 1) as usize] = fm.object[(ind - 1) as usize];
            ind += 1;
        }
        fm.nin_order += 1; // add to end of order list
        fm.obj_order[(fm.nin_order - 1) as usize] = iobj;
        // clear earlier spot in list
        // An object the packer dropped from the order list has
        // `fmodNdx_order` 0, and the C then stores to `fmodObj_order[-1]`, a
        // write before the allocation (source-level UB, CLAUDE.md); that
        // store is skipped here rather than reproduced.
        let earlier = fm.ndx_order[(iobj - 1) as usize];
        if earlier > 0 {
            fm.obj_order[(earlier - 1) as usize] = 0;
        }
        fm.ndx_order[(iobj - 1) as usize] = fm.nin_order; // set up index to order list
    }
    false
}

/// Original: `fortObjectPacker` (`fortmodel.c:337`).
/// Repacks the fmodObject array so that all objects are contiguous.
pub fn fort_object_packer(fm: &mut FortModel) {
    //
    // recompute total entries in OBJECT, total # of objects, max object #
    //
    let ntot_save = fm.ntot_in_obj;
    let num_obj_save = fm.n_object;
    let max_save = fm.max_mod_obj;
    fm.ntot_in_obj = 0;
    fm.n_object = 0;
    for iobj in 1..=fm.max_obj_num {
        if fm.npt_in_obj[(iobj - 1) as usize] > 0 {
            fm.ntot_in_obj += fm.npt_in_obj[(iobj - 1) as usize];
            fm.n_object += 1;
            fm.max_mod_obj = iobj;
        }
    }
    if ntot_save != fm.ntot_in_obj || num_obj_save != fm.n_object || max_save < fm.max_mod_obj {
        print!(
            "fortObjectPacker mismatch: ntot {} {}, nobj {} {}, max {} {}\n",
            ntot_save, fm.ntot_in_obj, num_obj_save, fm.n_object, max_save, fm.max_mod_obj
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    //
    // if total entries in OBJECT is less than the current free pointer,
    // pack the array down
    //
    if fm.ntot_in_obj < fm.ibase_free {
        fm.ibase_free = 0;
        for iord in 1..=fm.nin_order {
            let iobj = fm.obj_order[(iord - 1) as usize];
            if iobj != 0 {
                let num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
                let ibase_obj = fm.ibase_obj[(iobj - 1) as usize];
                if num_in_obj != 0 {
                    if ibase_obj != fm.ibase_free {
                        // move pointers down if needed
                        let mut ibase_down = fm.ibase_free;
                        let mut ibase = 1 + ibase_obj;
                        while ibase <= num_in_obj + ibase_obj {
                            ibase_down += 1;
                            fm.object[(ibase_down - 1) as usize] = fm.object[(ibase - 1) as usize];
                            ibase += 1;
                        }
                    }
                    fm.ibase_obj[(iobj - 1) as usize] = fm.ibase_free; // in any case, reset these
                    fm.ibase_free += num_in_obj;
                } else {
                    // if fmodObject empty, clear spot
                    fm.obj_order[(iord - 1) as usize] = 0;
                }
            }
        }
    }
    //
    // repack the object order array if # of objects is < # in order array
    //
    if fm.n_object < fm.nin_order {
        let mut ind_order = 0;
        for iord in 1..=fm.nin_order {
            let iobj = fm.obj_order[(iord - 1) as usize];
            if iobj != 0 {
                if fm.npt_in_obj[(iobj - 1) as usize] != 0 {
                    ind_order += 1;
                    fm.obj_order[(ind_order - 1) as usize] = iobj;
                    fm.ndx_order[(iobj - 1) as usize] = ind_order;
                } else {
                    fm.ndx_order[(iobj - 1) as usize] = 0;
                }
            }
        }
        fm.nin_order = ind_order;
    }
}

/// Original: `readFortModel` (`fortmodel.c:170`).
pub fn read_fort_model(filename: &str, fm: &mut FortModel) -> Result<(), i32> {
    {
        let error = openimoddata(filename);
        if error != 0 {
            return Err(error);
        }
        let (mut total_contours, mut max_contours, mut total_points, mut max_points) = (0, 0, 0, 0);
        let error = imodcountcontspoints(
            &mut total_contours,
            &mut max_contours,
            &mut total_points,
            &mut max_points,
        );
        if error != 0 {
            return Err(error);
        }
        if fm.fm_max_obj_loaded > 0 {
            fm.fm_need_objects = fm.fm_max_obj_loaded
                * ((max_contours as f32 * fm.fm_boost_read_in_by) as i32)
                    .max(max_contours + fm.fm_inc_read_obj_by);
            fm.fm_need_points = fm.fm_max_obj_loaded
                * ((max_points as f32 * fm.fm_boost_read_in_by) as i32)
                    .max(max_points + fm.fm_inc_read_points_by);
        } else {
            fm.fm_need_objects = ((total_contours as f32 * fm.fm_boost_read_in_by) as i32)
                .max(total_contours + fm.fm_inc_read_obj_by);
            fm.fm_need_points = ((total_points as f32 * fm.fm_boost_read_in_by) as i32)
                .max(total_points + fm.fm_inc_read_points_by);
        }
        allocate_fort_model(fm);
        let error = getimod(
            &mut fm.ibase_obj,
            &mut fm.npt_in_obj,
            &mut fm.p_coord,
            &mut fm.obj_color,
            &mut fm.n_point,
            &mut fm.n_object,
            filename,
        );
        if error != 0 {
            return Err(error);
        }
    }
    complete_model_values(fm);
    Ok(())
}

/// Original: `putFortModObjects` (`fortmodel.c:229`).
pub fn put_fort_mod_objects(fm: &mut FortModel) -> Result<(), i32> {
    if fm.n_point < 0 {
        return Ok(());
    }
    let error = putimod(
        &fm.ibase_obj,
        &fm.npt_in_obj,
        &fm.p_coord,
        &fm.object,
        &fm.obj_color,
        fm.n_point,
        fm.max_mod_obj,
    );
    // `fortmodel.c:231-232`: the C exits here rather than returning the code.
    if error != 0 {
        crate::imod::libcfshr::parse_params::exit_error(b"putting objects back into IMOD model");
    }
    Ok(())
}

/// Original: `writeFortModel` (`fortmodel.c:204`).
///
/// The C returns nothing: a failed backup is a warning and a failed write an
/// `exitError` (`fortmodel.c:207-214`), so this always returns `Ok`.
pub fn write_fort_model(filename: &str, fm: &mut FortModel) -> Result<(), i32> {
    let ierr = imod_backup_file(filename);
    if ierr != 0 {
        use std::io::Write as _;
        let _ = crate::imod::libcfshr::b3dutil::ImodFile::Stdout
            .write_all(b"WARNING: writeFortModel - Error attempting to rename existing model file");
    }
    put_fort_mod_objects(fm)?;
    let error = writeimod(filename);
    if error != 0 {
        crate::imod::libcfshr::parse_params::exit_error(
            format!("Writing output model file {filename}").as_bytes(),
        );
    }
    Ok(())
}

/// Original: `fortModOpenError` (`fortmodel.c:201`).
/// The source fills a 320-byte blank-padded Fortran buffer and then trims it;
/// `imodopenerror` returns the message directly now, so the trim is the
/// identity.
pub fn fort_mod_open_error() -> String {
    imodopenerror()
}

/// Original: `scaleFortModel` (`fortmodel.c:452`).
pub fn scale_fort_model(fm: &mut FortModel, idir: i32) -> Result<(), i32> {
    let (mut xy, mut zscale, mut xoff, mut yoff, mut zoff, mut flip) = (0., 0., 0., 0., 0., 0);
    let (mut xs, mut ys, mut zs) = (0., 0., 0.);
    let err = getimodhead(
        &mut xy,
        &mut zscale,
        &mut xoff,
        &mut yoff,
        &mut zoff,
        &mut flip,
    );
    if err != 0 {
        return Err(err);
    }
    let err = getimodscales(&mut xs, &mut ys, &mut zs);
    if err != 0 {
        return Err(err);
    }
    for point in fm.p_coord.iter_mut().take(fm.n_point.max(0) as usize) {
        if idir == 0 {
            point[0] = (point[0] - xoff) / xs;
            point[1] = (point[1] - yoff) / ys;
            point[2] = (point[2] - zoff) / zs;
        } else {
            point[0] = xs * point[0] + xoff;
            point[1] = ys * point[1] + yoff;
            point[2] = zs * point[2] + zoff;
        }
    }
    Ok(())
}

/// Original: `scaleFortModToImage` (`fortmodel.c:493`).
pub fn scale_fort_mod_to_image(fm: &mut FortModel, unit: i32, idir: i32) -> Result<(), i32> {
    let (mut origin, mut delta, mut tilt) = ([0.; 3], [0.; 3], [0.; 3]);
    origin = iiu_ret_origin(unit);
    delta = iiu_ret_delta(unit);
    tilt = iiu_ret_tilt(unit);
    if idir == 0 && imodhasimageref() <= 0 {
        return Ok(());
    }
    if idir != 0 {
        let error = putimageref(&delta, &origin, &tilt);
        if error != 0 {
            return Err(error);
        }
    }
    for point in fm.p_coord.iter_mut().take(fm.n_point.max(0) as usize) {
        for axis in 0..3 {
            point[axis] = if idir == 0 {
                (point[axis] + origin[axis]) / delta[axis]
            } else {
                point[axis] * delta[axis] - origin[axis]
            };
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_model() -> FortModel {
        let mut fm = FortModel::default();
        fm.max_obj_num = 4;
        fm.max_pt = 12;
        fm.len_object = 16;
        fm.max_obj_order = 6;
        fm.object = vec![0; 16];
        fm.npt_in_obj = vec![0; 4];
        fm.ibase_obj = vec![0; 4];
        fm.obj_color = vec![[0; 2]; 4];
        fm.obj_order = vec![0; 6];
        fm.ndx_order = vec![0; 4];
        fm.p_coord = vec![[0.; 3]; 12];
        fm.pt_label = vec![0; 12];
        fm
    }

    #[test]
    fn mover_and_packer_preserve_sparse_object_entries() {
        let mut fm = small_model();
        fm.object[..5].copy_from_slice(&[11, 12, 21, 22, 23]);
        fm.npt_in_obj[..2].copy_from_slice(&[2, 3]);
        fm.ibase_obj[..2].copy_from_slice(&[0, 2]);
        fm.obj_order[..2].copy_from_slice(&[1, 2]);
        fm.ndx_order[..2].copy_from_slice(&[1, 2]);
        fm.nin_order = 2;
        fm.n_object = 2;
        fm.max_mod_obj = 2;
        fm.ntot_in_obj = 5;
        fm.ibase_free = 5;
        assert!(!fort_object_mover(&mut fm, 1));
        assert_eq!(&fm.object[5..7], &[11, 12]);
        fm.npt_in_obj[1] = 0;
        fort_object_packer(&mut fm);
        assert_eq!(fm.n_object, 1);
        assert_eq!(fm.nin_order, 1);
        assert_eq!(fm.obj_order[0], 1);
        assert_eq!(&fm.object[..2], &[11, 12]);
    }

    #[test]
    fn maps_wimp_object_number_to_model_object_and_contour() {
        let mut fm = small_model();
        fm.obj_color[0][1] = 254;
        fm.obj_color[1][1] = 253;
        fm.obj_color[2][1] = 254;
        let (mut obj, mut cont) = (0, 0);
        fort_mod_obj_to_cont(3, &fm.obj_color, &mut obj, &mut cont);
        assert_eq!((obj, cont), (2, 2));
    }
}
