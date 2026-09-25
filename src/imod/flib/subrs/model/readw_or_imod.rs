//! Translation of `IMOD/flib/subrs/model/readw_or_imod.f`.
//!
//! The Fortran `use fortmodel` module arrays are the [`FortModel`] passed in;
//! the IMOD model that `openImodData` opens stays in `imodel_fwrap`'s own
//! state (`sImod`), exactly as in the source, so a later `writeimod`,
//! `imodWriteAsWimp` or `getimodhead` sees it.  Fortran unit 20 is the
//! `BufReader` that `read_mod` reads; the binary WIMP stream is a `blockio`
//! unit.
use std::fs::File;
use std::io::BufReader;

use crate::imod::flib::subrs::imsubs::blockio::{qclose, qopen, qread, qseek};
use crate::imod::flib::subrs::imsubs::convert_vms::{convert_longs, convert_shorts};
use crate::imod::flib::subrs::model::fortmodel::{FortModel, allocate_fort_model};
use crate::imod::flib::subrs::model::read_mod::read_mod;
use crate::imod::libimod::imodel_fwrap::{
    fromvmsfloats, getimod, getimodobjlist, getimodobjrange, imodcountcontspoints, openimoddata,
};

/// `lowbyte` from `include 'endian.inc'`, the file `IMOD/setup2:428` writes
/// for the build machine: `parameter (lowbyte=1,...)` on this little-endian
/// platform (`/tmp/imod-reference-build/include/endian.inc`).
const LOWBYTE: i32 = 1;

/// Original: `readw_or_imod` (`readw_or_imod.f:11`).
///
/// Reads an IMOD model or an old WIMP model from `filename` and returns the
/// model contours in the model arrays of `fm`.  Returns `false` for error.
///
/// Source-level UB kept as a panic rather than reproduced: in the binary
/// WIMP branch an object number `int4(1)` outside `1..=max_obj_num`, or a
/// point number that `mod(ipt,10000)` turns into 0, indexes outside the
/// Fortran arrays, which gfortran (no bounds checking) writes through into
/// neighbouring memory.
pub fn readw_or_imod(filename: &str, fm: &mut FortModel) -> bool {
    let mut int4 = [0u8; 16];
    let (mut num_pts_tot, mut max_num_pts, mut num_conts_tot, mut max_num_conts) = (0, 0, 0, 0);
    let mut ierr: i32;
    let mut istrm: i32 = 0;
    let mut ier: i32 = 0;
    let mut ninobj: i32;
    let mut ipt: i32;
    let mut flt = [0u8; 12];
    let mut int2 = [0u8; 52];
    let mut ptbyte = [0u8; 1];
    // `int2(1)`, `int4(k)` and `flt(k)` read the buffers in native order.
    let int2_1 = |b: &[u8; 52]| i16::from_ne_bytes([b[0], b[1]]);
    let int4_k = |b: &[u8; 16], k: usize| {
        i32::from_ne_bytes([b[4 * k - 4], b[4 * k - 3], b[4 * k - 2], b[4 * k - 1]])
    };
    let flt_k = |b: &[u8; 12], k: usize| {
        f32::from_ne_bytes([b[4 * k - 4], b[4 * k - 3], b[4 * k - 2], b[4 * k - 1]])
    };

    let mut readw_or_imod = false;
    ierr = openimoddata(filename);
    if ierr == 0 {
        if imodcountcontspoints(
            &mut num_conts_tot,
            &mut max_num_conts,
            &mut num_pts_tot,
            &mut max_num_pts,
        ) != 0
        {
            return readw_or_imod;
        }
        // `int(maxNumConts * fmBoostReadInBy)`: integer times real is a
        // real*4 product, truncated by `int`.
        if fm.fm_max_obj_loaded > 0 {
            fm.fm_need_objects = fm.fm_max_obj_loaded
                * ((max_num_conts as f32 * fm.fm_boost_read_in_by) as i32)
                    .max(max_num_conts + fm.fm_inc_read_obj_by);
            fm.fm_need_points = fm.fm_max_obj_loaded
                * ((max_num_pts as f32 * fm.fm_boost_read_in_by) as i32)
                    .max(max_num_pts + fm.fm_inc_read_points_by);
        } else {
            fm.fm_need_objects = ((num_conts_tot as f32 * fm.fm_boost_read_in_by) as i32)
                .max(num_conts_tot + fm.fm_inc_read_obj_by);
            fm.fm_need_points = ((num_pts_tot as f32 * fm.fm_boost_read_in_by) as i32)
                .max(num_pts_tot + fm.fm_inc_read_points_by);
        }
        allocate_fort_model(fm);
        //
        if getimod(
            &mut fm.ibase_obj,
            &mut fm.npt_in_obj,
            &mut fm.p_coord,
            &mut fm.obj_color,
            &mut fm.n_point,
            &mut fm.n_object,
            filename,
        ) != 0
        {
            return readw_or_imod;
        }

        complete_model_values(fm);
        readw_or_imod = true;
    } else {
        allocate_fort_model(fm);
        // `open(20,file=filename,status='old',err=20)`; `err=20` goes to the
        // `close(20)`, which for a unit that never opened does nothing.
        let mut unit20 = match File::open(filename) {
            Ok(file) => BufReader::new(file),
            Err(_) => return readw_or_imod,
        };
        qopen(&mut istrm, filename, "RO");
        qseek(istrm, 1, 1, 1, 1, 1);
        //
        // Every `go to 10` below is this label: `readw_or_imod=read_mod()`,
        // then `15 call qclose(istrm)` and `20 close(20)`.
        'label10: {
            qread(istrm, &mut int2, 52, &mut ier);
            if LOWBYTE == 2 {
                convert_shorts(&mut int2, 1);
            }
            if ier != 0 || int2_1(&int2) != 3 {
                break 'label10;
            }
            //
            qread(istrm, &mut int2[..2], 2, &mut ier);
            if LOWBYTE == 2 {
                convert_shorts(&mut int2, 1);
            }
            if ier != 0 || int2_1(&int2) != 3 {
                break 'label10;
            }
            //
            qread(istrm, &mut int4[..12], 12, &mut ierr);
            if ierr != 0 {
                break 'label10;
            }
            if LOWBYTE == 2 {
                convert_longs(&mut int4, 3);
            }
            //
            fm.n_point = int4_k(&int4, 2);
            fm.n_object = 0; //recount # of objects
            fm.ibase_free = 0;
            fm.ntot_in_obj = 0;
            fm.max_mod_obj = 0;
            for i in 1..=fm.max_obj_num {
                fm.npt_in_obj[i as usize - 1] = 0;
            }
            // `100 call qread(...)` ... `go to 100`
            loop {
                qread(istrm, &mut int2[..2], 2, &mut ier);
                if LOWBYTE == 2 {
                    convert_shorts(&mut int2, 1);
                }
                if ier != 0 || int2_1(&int2) != 3 {
                    break 'label10;
                }
                //
                qread(istrm, &mut int4, 16, &mut ier);
                if ier != 0 {
                    break 'label10;
                }
                if LOWBYTE == 2 {
                    convert_longs(&mut int4, 4);
                }
                //
                if int4_k(&int4, 1) == 0 {
                    break;
                }
                let i = int4_k(&int4, 1);
                ninobj = int4_k(&int4, 2);
                fm.n_object += 1;
                fm.obj_color[i as usize - 1][0] = int4_k(&int4, 3);
                fm.obj_color[i as usize - 1][1] = int4_k(&int4, 4);
                fm.obj_order[fm.n_object as usize - 1] = i;
                fm.ndx_order[i as usize - 1] = fm.n_object;
                fm.npt_in_obj[i as usize - 1] = ninobj;
                fm.ntot_in_obj += ninobj;
                fm.ibase_obj[i as usize - 1] = fm.ibase_free;
                fm.max_mod_obj = fm.max_mod_obj.max(i);
                for ii in 1..=ninobj {
                    qread(istrm, &mut int2[..2], 2, &mut ier);
                    if LOWBYTE == 2 {
                        convert_shorts(&mut int2, 1);
                    }
                    if ier != 0 || int2_1(&int2) != 3 {
                        break 'label10;
                    }
                    //
                    qread(istrm, &mut int4[..4], 4, &mut ier);
                    if ier != 0 {
                        break 'label10;
                    }
                    if LOWBYTE == 2 {
                        convert_longs(&mut int4, 1);
                    }
                    ipt = int4_k(&int4, 1);
                    //
                    qread(istrm, &mut flt, 12, &mut ier);
                    if ier != 0 {
                        break 'label10;
                    }
                    fromvmsfloats(&mut flt, 3);
                    // `if(lowbyte.eq.1.)`: integer against real, true here.
                    if LOWBYTE as f32 == 1. {
                        convert_longs(&mut flt, 3);
                    }
                    //
                    qread(istrm, &mut ptbyte, 1, &mut ier);
                    if ier != 0 {
                        break 'label10;
                    }
                    //
                    if ipt > 0 {
                        if ipt > fm.n_point {
                            ipt %= 10000; //in case of old models
                        }
                        fm.p_coord[ipt as usize - 1][0] = flt_k(&flt, 1);
                        fm.p_coord[ipt as usize - 1][1] = flt_k(&flt, 2);
                        fm.p_coord[ipt as usize - 1][2] = flt_k(&flt, 3);
                        fm.pt_label[ipt as usize - 1] = ptbyte[0] as i8;
                    }
                    fm.object[(ii + fm.ibase_free) as usize - 1] = ipt;
                }
                fm.ibase_free += ninobj;
            }
            fm.n_clabel = 0;
            fm.nin_order = fm.n_object;
            readw_or_imod = true;
            // `go to 15`
            qclose(istrm);
            return readw_or_imod;
        }
        //
        // `10 readw_or_imod=read_mod()`
        readw_or_imod = read_mod(&mut unit20, fm);
        // `15 call qclose(istrm)`; `20 close(20)` is the drop of `unit20`.
        qclose(istrm);
    }
    readw_or_imod
}

/// Original: `getModelObjectRange` (`readw_or_imod.f:142`).
///
/// Once a WIMP model has been opened, this routine fills the model arrays with
/// contour data just for the objects ranging from `iobj_strt` to `iobj_end`.
/// Returns `false` for error.
pub fn get_model_object_range(iobj_strt: i32, iobj_end: i32, fm: &mut FortModel) -> bool {
    let ierr = getimodobjrange(
        iobj_strt,
        iobj_end,
        &mut fm.ibase_obj,
        &mut fm.npt_in_obj,
        &mut fm.p_coord,
        &mut fm.obj_color,
        &mut fm.n_point,
        &mut fm.n_object,
    );
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
pub fn get_model_object_list(iobj_list: &[i32], nin_list: i32, fm: &mut FortModel) -> bool {
    let ierr = getimodobjlist(
        iobj_list,
        nin_list,
        &mut fm.ibase_obj,
        &mut fm.npt_in_obj,
        &mut fm.p_coord,
        &mut fm.obj_color,
        &mut fm.n_point,
        &mut fm.n_object,
    );
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
