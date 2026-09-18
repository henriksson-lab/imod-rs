//! WIMP-style model-array bridge from `IMOD/libimod/fortmodel.c`.
//!
//! The arrays themselves are the translated `fortmodel.f90` common state,
//! represented by [`FortModel`].  This source unit supplies the C helpers
//! which connect that state to `imodel_fwrap` and keep its sparse object list
//! packed.  Keeping one owned Rust record avoids the C version's mutable
//! process-global pointers without changing the one-based array convention.

use std::ffi::{CString, c_char};

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

/// Original: `fortModObjToCont` (`fortmodel.c:278`).
pub fn fort_mod_obj_to_cont(fm: &FortModel, iobj: i32) -> Option<(i32, i32)> {
    let index = usize::try_from(iobj.checked_sub(1)?).ok()?;
    let color = fm.obj_color.get(index)?[0];
    let contour = fm.obj_color[..=index]
        .iter()
        .filter(|entry| entry[0] == color)
        .count() as i32;
    Some((256 - color, contour))
}

/// Original: `fortObjectMover` (`fortmodel.c:307`).  Returns `true` for the
/// native `failed != 0` result.
pub fn fort_object_mover(fm: &mut FortModel, iobj: i32) -> bool {
    let Some(index) = iobj
        .checked_sub(1)
        .and_then(|value| usize::try_from(value).ok())
    else {
        return true;
    };
    if index >= fm.npt_in_obj.len() {
        return true;
    }
    if fm.nin_order > 0 && fm.obj_order[fm.nin_order as usize - 1] == iobj {
        return false;
    }
    let count = fm.npt_in_obj[index];
    if fm.nin_order >= fm.max_obj_order || fm.ibase_free + count + 4 >= fm.len_object {
        fort_object_packer(fm);
    }
    if fm.nin_order >= fm.max_obj_order || fm.ibase_free + count + 4 >= fm.len_object {
        return true;
    }
    if count <= 0 {
        return false;
    }
    let old_base = fm.ibase_obj[index] as usize;
    let new_base = fm.ibase_free as usize;
    let count = count as usize;
    if old_base + count > fm.object.len() || new_base + count > fm.object.len() {
        return true;
    }
    // `copy_within` deliberately preserves the native memmove semantics if a
    // caller moves an object whose old slot overlaps the free tail.
    fm.object.copy_within(old_base..old_base + count, new_base);
    fm.ibase_obj[index] = fm.ibase_free;
    fm.ibase_free += count as i32;
    fm.nin_order += 1;
    fm.obj_order[fm.nin_order as usize - 1] = iobj;
    let old_order = fm.ndx_order[index];
    if old_order > 0 && (old_order as usize) <= fm.obj_order.len() {
        fm.obj_order[old_order as usize - 1] = 0;
    }
    fm.ndx_order[index] = fm.nin_order;
    false
}

/// Original: `fortObjectPacker` (`fortmodel.c:367`).
pub fn fort_object_packer(fm: &mut FortModel) {
    let saved_total = fm.ntot_in_obj;
    let saved_objects = fm.n_object;
    let saved_max = fm.max_mod_obj;
    fm.ntot_in_obj = 0;
    fm.n_object = 0;
    fm.max_mod_obj = 0;
    for (index, &count) in fm.npt_in_obj.iter().enumerate() {
        if count > 0 {
            fm.ntot_in_obj += count;
            fm.n_object += 1;
            fm.max_mod_obj = index as i32 + 1;
        }
    }
    let _mismatch =
        saved_total != fm.ntot_in_obj || saved_objects != fm.n_object || saved_max < fm.max_mod_obj;
    if fm.ntot_in_obj < fm.ibase_free {
        let mut free = 0usize;
        for order in 0..fm.nin_order.max(0) as usize {
            let object = fm.obj_order[order];
            if object <= 0 {
                continue;
            }
            let index = object as usize - 1;
            let count = fm.npt_in_obj[index].max(0) as usize;
            if count == 0 {
                fm.obj_order[order] = 0;
                continue;
            }
            let base = fm.ibase_obj[index].max(0) as usize;
            if base != free {
                fm.object.copy_within(base..base + count, free);
            }
            fm.ibase_obj[index] = free as i32;
            free += count;
        }
        fm.ibase_free = free as i32;
    }
    if fm.n_object < fm.nin_order {
        let mut retained = 0usize;
        for order in 0..fm.nin_order.max(0) as usize {
            let object = fm.obj_order[order];
            if object <= 0 {
                continue;
            }
            let index = object as usize - 1;
            if fm.npt_in_obj[index] == 0 {
                fm.ndx_order[index] = 0;
                continue;
            }
            fm.obj_order[retained] = object;
            fm.ndx_order[index] = retained as i32 + 1;
            retained += 1;
        }
        fm.nin_order = retained as i32;
    }
}

/// Original: `readFortModel` (`fortmodel.c:170`).
pub fn read_fort_model(filename: &str, fm: &mut FortModel) -> Result<(), i32> {
    let filename = CString::new(filename).map_err(|_| -1)?;
    unsafe {
        let error = openimoddata(filename.as_ptr(), filename.as_bytes().len() as i32);
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
            fm.ibase_obj.as_mut_ptr(),
            fm.npt_in_obj.as_mut_ptr(),
            fm.p_coord.as_mut_ptr(),
            fm.obj_color.as_mut_ptr(),
            &mut fm.n_point,
            &mut fm.n_object,
            filename.as_ptr(),
            filename.as_bytes().len() as i32,
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
    let error = unsafe {
        putimod(
            fm.ibase_obj.as_ptr(),
            fm.npt_in_obj.as_ptr(),
            fm.p_coord.as_ptr(),
            fm.object.as_ptr(),
            fm.obj_color.as_ptr(),
            fm.n_point,
            fm.max_mod_obj,
        )
    };
    if error == 0 { Ok(()) } else { Err(error) }
}

/// Original: `writeFortModel` (`fortmodel.c:208`).
pub fn write_fort_model(filename: &str, fm: &mut FortModel) -> Result<(), i32> {
    let _ = imod_backup_file(filename);
    put_fort_mod_objects(fm)?;
    let name = CString::new(filename).map_err(|_| -1)?;
    let error = unsafe { writeimod(name.as_ptr(), name.as_bytes().len() as i32) };
    if error == 0 { Ok(()) } else { Err(error) }
}

/// Original: `fortModOpenError` (`fortmodel.c:201`).
pub fn fort_mod_open_error() -> String {
    let mut bytes = [b' ' as c_char; 320];
    unsafe { imodopenerror(bytes.as_mut_ptr(), bytes.len() as i32) };
    bytes
        .iter()
        .map(|&byte| byte as u8 as char)
        .collect::<String>()
        .trim_end()
        .to_owned()
}

/// Original: `scaleFortModel` (`fortmodel.c:452`).
pub fn scale_fort_model(fm: &mut FortModel, idir: i32) -> Result<(), i32> {
    let (mut xy, mut zscale, mut xoff, mut yoff, mut zoff, mut flip) = (0., 0., 0., 0., 0., 0);
    let (mut xs, mut ys, mut zs) = (0., 0., 0.);
    unsafe {
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
    iiu_ret_origin(unit, &mut origin);
    iiu_ret_delta(unit, &mut delta);
    iiu_ret_tilt(unit, &mut tilt);
    if idir == 0 && unsafe { imodhasimageref() } <= 0 {
        return Ok(());
    }
    if idir != 0 {
        let error = unsafe { putimageref(delta.as_ptr(), origin.as_ptr(), tilt.as_ptr()) };
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
        fm.obj_color[0][0] = 254;
        fm.obj_color[1][0] = 253;
        fm.obj_color[2][0] = 254;
        assert_eq!(fort_mod_obj_to_cont(&fm, 3), Some((2, 2)));
    }
}
