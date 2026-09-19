//! Loading an IMOD model file from Fortran code, from
//! `IMOD/libimod/imodel_fwrap.c`.
//!
//! This is the Fortran-facing wrapper around one model context.  Its source
//! file-static state is held in a thread-local owned record while the exported
//! calls retain the Fortran calling convention (pointer arguments,
//! `fortStrLen_t` hidden string lengths, arrays of `[f32; 3]` / `[i32; 2]), as
//! `libcfshr/adoc_fwrap.rs` does for the autodoc wrapper.
//!
//! The two convenience pointers `sObj` and `sCont` that `checkAssignObject` and
//! `checkAssignObjCont` assign are held here as indices into the model's
//! object and contour `Vec`s, because the translated `Imod` owns those in
//! `Vec`s rather than as `malloc`ed arrays.
//!
//! Not translated here, with the reason in each case:
//!
//! * `getimodnesting` (`imodel_fwrap.c:2343`) — translated below, but note
//!   that its `Icont **scancont` array of aliasing pointers becomes a cloned
//!   `Vec<Icont>`, because `imodContourCheckNesting` is translated in
//!   `icont.rs` against `&mut [Icont]`.  The source never writes a scan contour
//!   back into the object either, so the observable result is the same.
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::c_char;

use super::icont::{
    ICONT_SCANLINE, Nesting, imod_contour_check_nesting, imod_contour_clear, imod_contour_copy,
    imod_contour_free_nests, imod_contour_free_z_tables, imod_contour_get_bbox,
    imod_contour_make_z_tables, imod_contour_nest_levels, imod_contours_delete_to_end,
};
use super::imat::{imod_mat_delete, imod_mat_new, imod_mat_scale, imod_mat_trans};
use super::imesh::{
    IMESH_FLAG_NMAG, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_END, IMOD_MESH_ENDPOLY, imesh_resol,
    imod_mesh_nearest_res, imod_mesh_poly_norm_factors,
};
use super::imodel::{
    IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, IMOD_STRSIZE, IMOD_UNIT_PIXEL, IMODF_FLIPYZ,
    IMODF_OTRANS_ORIGIN, IMODF_TILTOK, IMODF_Z_FROM_MINUSPT5, IOBJ_STRSIZE, Icont, Imod, Iobj,
    Ipoint, Iref_image, imod_delete, imod_delete_contour, imod_delete_list_of_conts, imod_flip_yz,
    imod_new, imod_new_object, imod_set_index, imod_transform,
};
use super::imodel_files::{imod_from_vms_floats, imod_read_file, imod_write_file};
use super::imodel_to::imod_to_wmod;
use super::iobj::{
    IMOD_OBJFLAG_PNT_ON_SEC, IMOD_OBJFLAG_THICK_CONT, IMOD_OBJFLAG_USE_VALUE, MATFLAGS2_CONSTANT,
    MATFLAGS2_SKIP_LOW, imod_object_add_contour, imod_object_sort_surf, imod_objects_delete,
    iobj_scat,
};
use super::ipoint::{imod_point_add, imod_point_append, imod_point_get_size, imod_point_set_size};
use super::istore::{
    Istore, StoreUnion, istore_find_add_min_max1, istore_get_min_max, istore_insert_change,
    istore_lookup,
};
use super::iview::{imod_objview_complete, imod_objviews_free};
use crate::imod::libcfshr::b3dutil::{ImodFile, fortran_string};

/// Original: `fortStrLen_t` (`imodel.h` via `b3dutil.h`).
pub type FortStrLenT = i32;

/* These values should match max_obj_num and max_pt in model.inc */
/// Original: `FWRAP_MAX_OBJECT` (`imodel_fwrap.c:21`).
pub const FWRAP_MAX_OBJECT: i32 = 1000000;
/// Original: `FWRAP_MAX_POINTS` (`imodel_fwrap.c:22`).
pub const FWRAP_MAX_POINTS: i32 = 20000000;
/// Original: `FWRAP_MAX_CLIP_PLANES` (`imodel_fwrap.c:23`).
pub const FWRAP_MAX_CLIP_PLANES: i32 = 100;

/// Original: `FWRAP_NOERROR` (`imodel_fwrap.c:25`).
pub const FWRAP_NOERROR: i32 = 0;
pub const FWRAP_ERROR_BAD_FILENAME: i32 = -1;
pub const FWRAP_ERROR_FILE_NOT_IMOD: i32 = -2;
pub const FWRAP_ERROR_FILE_TO_BIG: i32 = -3;
pub const FWRAP_ERROR_NO_OBJECTS: i32 = -4;
pub const FWRAP_ERROR_NO_MODEL: i32 = -5;
pub const FWRAP_ERROR_BAD_OBJNUM: i32 = -6;
pub const FWRAP_ERROR_MEMORY: i32 = -7;
pub const FWRAP_ERROR_NO_VALUE: i32 = -8;
pub const FWRAP_ERROR_STRING_LEN: i32 = -9;
pub const FWRAP_ERROR_FROM_CALL: i32 = -10;
pub const FWRAP_ERROR_OPENING_FILE: i32 = -11;
pub const FWRAP_ERROR_READING_FILE: i32 = -12;
pub const FWRAP_ERROR_NO_PIXEL_SIZE: i32 = -13;

/* Move this number when an error is added */
/// Original: `FWRAP_PAST_ERRORS` (`imodel_fwrap.c:41`).
pub const FWRAP_PAST_ERRORS: i32 = -13;

/// Original: `sErrorStrings` (`imodel_fwrap.c:43`).
static S_ERROR_STRINGS: [&str; 12] = [
    "FILE DOES NOT EXIST",
    "FILE IS NOT AN IMOD MODEL",
    "",
    "MODEL HAS NO OBJECTS",
    "",
    "",
    "FAILURE TO ALLOCATE MEMORY",
    "",
    "",
    "",
    "FILE EXISTS BUT CANNOT BE OPENED",
    "ERROR READING/PROCESSING MODEL FILE",
];

/// Original: `NO_VALUE_PUT` (`imodel_fwrap.c:48`).
pub const NO_VALUE_PUT: f32 = -9999.;

/// `GEN_STORE_FLOAT` (`istore.h:24`).
const GEN_STORE_FLOAT: u16 = 1;
/// `GEN_STORE_ONEPOINT` (`istore.h:32`).
const GEN_STORE_ONEPOINT: u16 = 1 << 7;
/// `GEN_STORE_VALUE1` (`istore.h:47`).
const GEN_STORE_VALUE1: i16 = 10;
/// `GEN_STORE_MINMAX1` (`istore.h:48`).
const GEN_STORE_MINMAX1: i16 = 11;

/// Original: `SizeStruct` (`imodel_fwrap.c:213`).
#[derive(Clone, Debug, Default)]
pub struct SizeStruct {
    pub ob: i32,
    pub co: i32,
    pub num: i32,
    pub sizes: Vec<f32>,
}

/// Original: `ValueStruct` (`imodel_fwrap.c:220`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ValueStruct {
    pub ob: i32,
    pub co: i32,
    pub pt: i32,
    pub value: f32,
}

/// Original: `NameStruct` (`imodel_fwrap.c:227`).
///
/// The Fortran-facing wrapper owns a decoded Rust name until it is applied to
/// the model during `putimod`.
#[derive(Clone, Debug)]
pub struct NameStruct {
    pub ob: i32,
    pub name: String,
}

/// Mutable model-construction state owned by one Fortran wrapper context.
///
/// This groups the C translation unit's file-static state so callers can be
/// migrated to an explicit context without retaining individual global slots.
/// The selected object and contour are indices because `Imod` owns the
/// corresponding collections.
#[derive(Default)]
struct FortranModelState {
    imod: Option<Imod>,
    partial_mode: i32,
    max_objects: i32,
    max_points: i32,
    flags_put: Vec<i32>,
    maxes_put: i32,
    xmax_put: i32,
    ymax_put: i32,
    zmax_put: i32,
    zscale_put: f32,
    rotation_put: Ipoint,
    sizes_put: Vec<SizeStruct>,
    values_put: Vec<ValueStruct>,
    max_values: i32,
    names_put: Vec<NameStruct>,
    last_open_error: i32,
    write_as_wimp: bool,
    obj: usize,
    cont: usize,
}

impl FortranModelState {
    fn new() -> Self {
        Self {
            max_objects: FWRAP_MAX_OBJECT,
            max_points: FWRAP_MAX_POINTS,
            zscale_put: NO_VALUE_PUT,
            rotation_put: Ipoint {
                x: NO_VALUE_PUT,
                y: NO_VALUE_PUT,
                z: NO_VALUE_PUT,
            },
            ..Self::default()
        }
    }
}

thread_local! {
    static FWRAP_STATE: RefCell<FortranModelState> = RefCell::new(FortranModelState::new());
}

/* These are not bit flags so they can range up to 255 */
pub const SCAT_SIZE_FLAG: i32 = 3;
pub const SYMBOL_SIZE_FLAG: i32 = 4;
pub const SYMBOL_TYPE_FLAG: i32 = 5;
pub const OBJECT_COLOR_FLAG: i32 = 6;
pub const USE_VALUE_FLAGS: i32 = 7;
pub const VAL_BLACKWHITE_FLAG: i32 = 8;
pub const PNT_ON_SEC_FLAG: i32 = 9;
pub const THICKEN_CONT_FLAG: i32 = 10;
pub const SYMBOL_FLAGS_FLAG: i32 = 11;
pub const SYMBOL_WIDTH_FLAG: i32 = 12;

/// Original: `FLAG_VALUE_SHIFT` (`imodel_fwrap.c:270`).
pub const FLAG_VALUE_SHIFT: i32 = 8;

/// Original: `OBJ_EMPTY` (`imodel_fwrap.c:1317`).
pub const OBJ_EMPTY: i32 = -2;
/// Original: `OBJ_HAS_DATA` (`imodel_fwrap.c:1318`).
pub const OBJ_HAS_DATA: i32 = -1;

/// Original: `Wmod_Colors` (`imodel_fwrap.c:1320`).
static WMOD_COLORS: [[f32; 3]; 9] = [
    [0.90, 0.82, 0.37], /* Dim Yellow  */
    [0.54, 0.51, 0.01], /* Olive Brown */
    [0.94, 0.49, 0.0],  /* Orange      */
    [1.00, 0.0, 0.0],   /* Red         */
    [0.0, 1.0, 0.0],    /* Green       */
    [0.0, 0.0, 1.0],    /* Blue        */
    [1.0, 1.0, 0.0],    /* Yellow      */
    [1.0, 0.0, 1.0],    /* Magenta     */
    [0.0, 1.0, 1.0],    /* Cyan        */
];

/* DNM: a common function to delete model and object flags */
/// Original: `deleteFimod` (`imodel_fwrap.c:273`).
unsafe fn delete_fimod(state: &mut FortranModelState) {
    if let Some(imod) = state.imod.as_mut() {
        imod_delete(imod);
    }
    state.imod = None;
    state.flags_put.clear();
    state.maxes_put = 0;
    state.zscale_put = NO_VALUE_PUT;
    state.rotation_put = Ipoint {
        x: NO_VALUE_PUT,
        y: NO_VALUE_PUT,
        z: NO_VALUE_PUT,
    };
    state.sizes_put.clear();
    state.values_put.clear();
    state.max_values = 0;
    state.names_put.clear();
}

/// Original: `checkAssignObject` (`imodel_fwrap.c:306`).
unsafe fn check_assign_object(state: &mut FortranModelState, ob: i32) -> i32 {
    let imod = match state.imod.as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if ob < 1 || ob > imod.obj.len() as i32 || imod.obj.is_empty() {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    state.obj = ob as usize - 1;
    FWRAP_NOERROR
}

/// Original: `checkAssignObjCont` (`imodel_fwrap.c:316`).
unsafe fn check_assign_obj_cont(state: &mut FortranModelState, ob: i32, co: i32) -> i32 {
    let err = check_assign_object(state, ob);
    if err != 0 {
        return err;
    }
    let imod = state.imod.as_ref().unwrap();
    if co < 1 || co > imod.obj[state.obj].cont.len() as i32 {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    state.cont = co as usize - 1;
    FWRAP_NOERROR
}

/// Original: `getMeshTrans` (`imodel_fwrap.c:327`).
unsafe fn get_mesh_trans(state: &FortranModelState, trans: &mut Ipoint) -> i32 {
    let imod = match state.imod.as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };

    trans.x = 0.;
    trans.y = 0.;
    trans.z = 0.;
    if let Some(iref) = imod.ref_image.as_ref() {
        if imod.flags & IMODF_OTRANS_ORIGIN != 0 {
            trans.x = (iref.otrans.x - iref.ctrans.x) / iref.cscale.x;
            trans.y = (iref.otrans.y - iref.ctrans.y) / iref.cscale.y;
            trans.z = (iref.otrans.z - iref.ctrans.z) / iref.cscale.z;
        }
    }
    0
}

/// Original: `imodarraylimits` (`imodel_fwrap.c:347`).
pub unsafe fn imodarraylimits(maxpt: i32, maxob: i32) {
    FWRAP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.max_objects = maxob;
        state.max_points = maxpt;
    });
}

/// Original: `imodpartialmode` (`imodel_fwrap.c:356`).
pub unsafe fn imodpartialmode(mode: i32) {
    FWRAP_STATE.with(|state| state.borrow_mut().partial_mode = mode);
}

/// Original: `getimod` (`imodel_fwrap.c:373`).
pub unsafe fn getimod(
    ibase: *mut i32,
    npt: *mut i32,
    coord: *mut [f32; 3],
    color: *mut [i32; 2],
    npoint: *mut i32,
    nobject: *mut i32,
    fname: *const c_char,
    fsize: FortStrLenT,
) -> i32 {
    let err = openimoddata(fname, fsize);

    if err != 0 {
        return err;
    }

    getopenedimod(ibase, npt, coord, color, npoint, nobject)
}

/// Original: `getopenedimod` (`imodel_fwrap.c:394`).
///
/// Source defect, translated as written: the `if (!sImod)` branch evaluates
/// `FWRAP_ERROR_NO_MODEL;` as a statement with no `return`, so a null model
/// falls through instead of reporting the error.
pub unsafe fn getopenedimod(
    ibase: *mut i32,
    npt: *mut i32,
    coord: *mut [f32; 3],
    color: *mut [i32; 2],
    npoint: *mut i32,
    nobject: *mut i32,
) -> i32 {
    let one = 1;
    *npoint = 0;
    *nobject = 0;
    let (has_model, partial_mode, objsize) = FWRAP_STATE.with(|state| {
        let state = state.borrow();
        (
            state.imod.is_some(),
            state.partial_mode,
            state.imod.as_ref().map_or(0, |imod| imod.obj.len() as i32),
        )
    });
    if !has_model {
        let _ = FWRAP_ERROR_NO_MODEL;
    }
    if partial_mode != 0 {
        return FWRAP_NOERROR;
    }
    getimodobjrange(one, objsize, ibase, npt, coord, color, npoint, nobject)
}

/// Original: `openimoddata` (`imodel_fwrap.c:412`).
pub unsafe fn openimoddata(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        state.last_open_error = FWRAP_ERROR_MEMORY;
        let mut model = match imod_new() {
            Some(model) => model,
            None => return FWRAP_ERROR_MEMORY,
        };

        let cfilename = fortran_string(fname, fsize);

        state.last_open_error = FWRAP_NOERROR;
        let path = cfilename;
        // `imodel_files.rs` now takes the shared `ImodFile` (NATIVE.md vocabulary
        // item 1); this wrapper stays `extern "C"` on the Fortran side but adapts
        // here.
        let fin = ImodFile::open(&path, "rb").ok_or(());
        if fin.is_err() {
            if std::fs::metadata(&path).is_err() {
                state.last_open_error = FWRAP_ERROR_BAD_FILENAME;
            } else {
                state.last_open_error = FWRAP_ERROR_OPENING_FILE;
            }
        }
        if state.last_open_error != 0 {
            return state.last_open_error;
        }
        let mut fin = fin.unwrap();

        let err = imod_read_file(&mut model, &mut fin);
        drop(fin);
        if let Err(code) = err {
            state.last_open_error = if code == -2 {
                FWRAP_ERROR_FILE_NOT_IMOD
            } else {
                FWRAP_ERROR_READING_FILE
            };
            return state.last_open_error;
        }

        if model.obj.is_empty() {
            state.last_open_error = FWRAP_ERROR_NO_OBJECTS;
            return FWRAP_ERROR_NO_OBJECTS;
        }

        /* DNM: need to delete model to avoid memory leak */
        delete_fimod(&mut state);

        state.imod = Some(model);
        /*
         *  Translate reference image coordinates to
         *  the identity matrix.
         *  DNM 11/5/98: rearranged to correspond to proper conventions
         */
        let imod = state.imod.as_mut().unwrap();
        let iref = imod.ref_image;

        if imod.flags & IMODF_FLIPYZ != 0 {
            imod_flip_yz(imod);
        }

        if let Some(iref) = iref {
            let mut mat = match imod_mat_new(3) {
                Some(mat) => mat,
                None => return FWRAP_ERROR_MEMORY,
            };

            imod_mat_scale(&mut mat, &iref.cscale);

            let pnt = Ipoint {
                x: -iref.ctrans.x,
                y: -iref.ctrans.y,
                z: -iref.ctrans.z,
            };
            imod_mat_trans(&mut mat, &pnt);

            /* DNM 11/5/98: no fortran code expects or wants tilt angles to be
            applied, so leave this out */

            imod_transform(Some(imod), Some(&mat));
            imod_mat_delete(&mut mat);
        }

        FWRAP_NOERROR
    })
}

/// Original: `imodopenerror` (`imodel_fwrap.c:500`).
pub unsafe fn imodopenerror(error: *mut c_char, errlen: FortStrLenT) {
    FWRAP_STATE.with(|state| unsafe {
        let last_open_error = state.borrow().last_open_error;
        let text = if last_open_error > -1 || last_open_error <= FWRAP_PAST_ERRORS {
            ""
        } else {
            S_ERROR_STRINGS[(-1 - last_open_error) as usize]
        };
        let limit = errlen.max(0) as usize;
        let mut index = 0usize;
        while index < text.len() && index < limit {
            *error.add(index) = text.as_bytes()[index] as c_char;
            index += 1;
        }
        while index < limit {
            *error.add(index) = b' ' as c_char;
            index += 1;
        }
    });
}

/// Original: `imodcountcontspoints` (`imodel_fwrap.c:513`).
pub unsafe fn imodcountcontspoints(
    num_conts_total: &mut i32,
    max_num_conts: &mut i32,
    num_pts_total: &mut i32,
    max_num_pts: &mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        *num_pts_total = 0;
        *max_num_pts = 0;
        *num_conts_total = 0;
        *max_num_conts = 0;

        for ob in 0..imod.obj.len() {
            let obj = &imod.obj[ob];
            let mut num_in_obj = 0;
            /* ACCUM_MAX (`b3dutil.h:40`) */
            *max_num_conts = if *max_num_conts > obj.cont.len() as i32 {
                *max_num_conts
            } else {
                obj.cont.len() as i32
            };
            *num_conts_total += obj.cont.len() as i32;
            for co in 0..obj.cont.len() {
                num_in_obj += obj.cont[co].pts.len() as i32;
            }
            *max_num_pts = if *max_num_pts > num_in_obj {
                *max_num_pts
            } else {
                num_in_obj
            };
            *num_pts_total += num_in_obj;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodobjlist` (`imodel_fwrap.c:551`).
pub unsafe fn getimodobjlist(
    obj_list: *const i32,
    nin_list: i32,
    ibase: *mut i32,
    npt: *mut i32,
    coord: *mut [f32; 3],
    color: *mut [i32; 2],
    npoint: *mut i32,
    nobject: *mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let mut ncontour = 0;
        let mut npoints = 0;
        let mut coord_index = 0usize;
        let mut ibase_val = 0;
        let mut coi = 0usize;

        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        if nin_list <= 0 {
            return FWRAP_ERROR_NO_OBJECTS;
        }

        /* Check object numbers and count contours and points */
        for ind in 0..nin_list as usize {
            let ob = *obj_list.add(ind) - 1;
            if ob < 0 || ob >= imod.obj.len() as i32 {
                return FWRAP_ERROR_BAD_OBJNUM;
            }
            let obj = &imod.obj[ob as usize];
            ncontour += obj.cont.len() as i32;
            for co in 0..obj.cont.len() {
                npoints += obj.cont[co].pts.len() as i32;
            }
        }

        if ncontour > state.max_objects {
            eprintln!("getimod: Too many contours in model for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        if npoints > state.max_points {
            eprintln!("getimod: Too many points in model for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        *npoint = npoints;
        *nobject = ncontour;

        for ind in 0..nin_list as usize {
            let ob = *obj_list.add(ind) - 1;
            let obj = &imod.obj[ob as usize];
            for co in 0..obj.cont.len() {
                let cont = &obj.cont[co];
                *ibase.add(coi) = ibase_val;
                ibase_val += cont.pts.len() as i32;
                *npt.add(coi) = cont.pts.len() as i32;
                (*color.add(coi))[0] = 1;
                (*color.add(coi))[1] = 255 - ob;

                for pt in 0..cont.pts.len() {
                    (*coord.add(coord_index))[0] = cont.pts[pt].x;
                    (*coord.add(coord_index))[1] = cont.pts[pt].y;
                    (*coord.add(coord_index))[2] = cont.pts[pt].z;
                    coord_index += 1;
                }
                coi += 1;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodobjrange` (`imodel_fwrap.c:628`).
pub unsafe fn getimodobjrange(
    obj_start: i32,
    obj_end: i32,
    ibase: *mut i32,
    npt: *mut i32,
    coord: *mut [f32; 3],
    color: *mut [i32; 2],
    npoint: *mut i32,
    nobject: *mut i32,
) -> i32 {
    let nin_list = obj_end + 1 - obj_start;

    if FWRAP_STATE.with(|state| state.borrow().imod.is_none()) {
        return FWRAP_ERROR_NO_MODEL;
    }
    if nin_list <= 0 {
        return FWRAP_ERROR_NO_OBJECTS;
    }
    let mut obj_list: Vec<i32> = vec![0; nin_list as usize];
    for i in 0..nin_list as usize {
        obj_list[i] = obj_start + i as i32;
    }
    getimodobjlist(
        obj_list.as_ptr(),
        nin_list,
        ibase,
        npt,
        coord,
        color,
        npoint,
        nobject,
    )
}

/// Original: `getimodscat` (`imodel_fwrap.c:660`).
pub unsafe fn getimodscat(
    ibase: *mut i32,
    npt: *mut i32,
    coord: *mut [f32; 3],
    color: *mut [i32; 2],
    npoint: *mut i32,
    maxobject: *mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let mut npoints = 0;
        let mut maxobj = 0;
        let mut coi = 0usize;
        let mut coord_index = 0usize;

        let model = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        if model.obj.is_empty() {
            return FWRAP_ERROR_NO_OBJECTS;
        }

        for ob in 0..model.obj.len() {
            let obj = &model.obj[ob];
            if iobj_scat(obj.flags) == 0 {
                continue;
            }
            maxobj = ob as i32 + 1;
            for co in 0..obj.cont.len() {
                npoints += obj.cont[co].pts.len() as i32;
            }
        }

        if maxobj > state.max_objects {
            eprintln!("getimodscat: Too many contours in model for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        if npoints > state.max_points {
            eprintln!("getimodscat: Too many points in model for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }
        *npoint = npoints;
        *maxobject = maxobj;

        for ob in 0..model.obj.len() {
            let obj = &model.obj[ob];

            let mut scont = Icont::default(); /* Create empty contour. */

            if iobj_scat(obj.flags) != 0 {
                for co in 0..obj.cont.len() {
                    let cont = &obj.cont[co];
                    for pt in 0..cont.pts.len() {
                        imod_point_append(&mut scont, cont.pts[pt]);
                    }
                }
            }

            *ibase.add(coi) = coord_index as i32;
            *npt.add(coi) = scont.pts.len() as i32;

            (*color.add(coi))[0] = 1;
            (*color.add(coi))[1] = 255 - ob as i32;

            for pt in 0..scont.pts.len() {
                (*coord.add(coord_index))[0] = scont.pts[pt].x;
                (*coord.add(coord_index))[1] = scont.pts[pt].y;
                (*coord.add(coord_index))[2] = scont.pts[pt].z;
                coord_index += 1;
            }
            coi += 1;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodmesh` (`imodel_fwrap.c:736`).
pub unsafe fn getimodmesh(
    objnum: i32,
    mut verts: *mut f32,
    mut index: *mut i32,
    limverts: &mut i32,
    limindex: &mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut resol = 0;
        let mut trans = Ipoint::default();

        if get_mesh_trans(&state, &mut trans) != 0 {
            return FWRAP_ERROR_NO_MODEL;
        }
        let imod = state.imod.as_mut().unwrap();
        imod.cindex.object = objnum - 1;
        let obj = &imod.obj[imod.cindex.object as usize];
        if obj.mesh.is_empty() {
            return -1;
        }

        let mesh = &obj.mesh;
        imod_mesh_nearest_res(mesh, obj.mesh.len() as i32, 0, &mut resol);

        let mut vsum = 0;
        let mut lsum = 0;
        for m in 0..obj.mesh.len() {
            if imesh_resol(mesh[m].flag) == resol {
                vsum += mesh[m].vert.len() as i32;
                lsum += mesh[m].list.len() as i32;
            }
        }
        if *limverts == 0 && *limindex == 0 {
            *limverts = vsum;
            *limindex = lsum;
            return FWRAP_NOERROR;
        }

        if vsum > *limverts {
            eprintln!("getimodmesh: Too many vertices in mesh for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        if lsum > *limindex {
            eprintln!("getimodmesh: Too many indices in mesh for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        for m in 0..obj.mesh.len() {
            if imesh_resol(mesh[m].flag) == resol {
                let mut i = 0;
                while i < mesh[m].vert.len() {
                    *verts = mesh[m].vert[i].x + trans.x;
                    verts = verts.add(1);
                    *verts = mesh[m].vert[i].y + trans.y;
                    verts = verts.add(1);
                    *verts = mesh[m].vert[i].z + trans.z;
                    verts = verts.add(1);
                    *verts = mesh[m].vert[i + 1].x;
                    verts = verts.add(1);
                    *verts = mesh[m].vert[i + 1].y;
                    verts = verts.add(1);
                    *verts = mesh[m].vert[i + 1].z;
                    verts = verts.add(1);
                    i += 2;
                }
                for i in 0..mesh[m].list.len() {
                    *index = mesh[m].list[i];
                    index = index.add(1);
                }
            }
        }

        FWRAP_NOERROR
    })
}

/// Original: `getimodverts` (`imodel_fwrap.c:808`).
pub unsafe fn getimodverts(
    objnum: i32,
    mut verts: *mut f32,
    index: *mut i32,
    limverts: i32,
    limindex: i32,
    nverts: &mut i32,
    nindex: &mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut resol = 0;
        let mut norm_add = 0;
        let mut vert_base = 0;
        let mut list_inc = 0;
        let mut trans = Ipoint::default();

        if get_mesh_trans(&state, &mut trans) != 0 {
            return FWRAP_ERROR_NO_MODEL;
        }
        let imod = state.imod.as_mut().unwrap();
        imod.cindex.object = objnum - 1;
        let obj = &imod.obj[imod.cindex.object as usize];
        if obj.mesh.is_empty() {
            return 1;
        }

        imod_mesh_nearest_res(&obj.mesh, obj.mesh.len() as i32, 0, &mut resol);

        let mut vsum = 0;
        for m in 0..obj.mesh.len() {
            if imesh_resol(obj.mesh[m].flag) == resol {
                vsum += obj.mesh[m].vert.len() as i32;
            }
        }

        if vsum / 2 > limverts {
            /* we don't want this message when running mtk */
            return FWRAP_ERROR_FILE_TO_BIG;
        }

        *nverts = 0;
        let mut j = 0usize;
        for m in 0..obj.mesh.len() {
            let mesh = &obj.mesh[m];
            if imesh_resol(mesh.flag) != resol {
                continue;
            }
            let mut mi = mesh.vert.len();
            let mut i = 0usize;
            while i < mi {
                *verts = mesh.vert[i].x + trans.x;
                verts = verts.add(1);
                *verts = mesh.vert[i].y + trans.y;
                verts = verts.add(1);
                *verts = mesh.vert[i].z + trans.z;
                verts = verts.add(1);
                i += 2;
            }
            *nverts += (mi / 2) as i32;

            mi = mesh.list.len();
            let mut i = 0usize;
            while i < mi && mesh.list[i] != IMOD_MESH_END {
                while i < mi
                    && mesh.list[i] != IMOD_MESH_END
                    && imod_mesh_poly_norm_factors(
                        mesh.list[i],
                        &mut list_inc,
                        &mut vert_base,
                        &mut norm_add,
                    ) == 0
                {
                    i += 1;
                }
                if i < mi
                    && imod_mesh_poly_norm_factors(
                        mesh.list[i],
                        &mut list_inc,
                        &mut vert_base,
                        &mut norm_add,
                    ) != 0
                {
                    *index.add(j) = IMOD_MESH_BGNPOLYNORM;
                    j += 1;
                    i += 1;
                    while i < mi && mesh.list[i] != IMOD_MESH_ENDPOLY {
                        if j as i32 + 6 > limindex {
                            println!("{i} {mi} {j} {limindex}");
                            eprintln!("getimodverts: Too many indices in mesh for Fortran program");
                            return FWRAP_ERROR_FILE_TO_BIG;
                        }
                        *index.add(j) = mesh.list[i + vert_base as usize] / 2;
                        j += 1;
                        *index.add(j) = mesh.list[i + (vert_base + list_inc) as usize] / 2;
                        j += 1;
                        *index.add(j) = mesh.list[i + (vert_base + 2 * list_inc) as usize] / 2;
                        j += 1;
                        i += 3 * list_inc as usize;
                    }

                    *index.add(j) = IMOD_MESH_ENDPOLY;
                    j += 1;
                    i += 1;
                }
            }
        }
        *index.add(j) = IMOD_MESH_END;
        j += 1;
        *nindex = j as i32;
        FWRAP_NOERROR
    })
}

/// Original: `getimodsizes` (`imodel_fwrap.c:895`).
pub unsafe fn getimodsizes(ob: i32, mut sizes: *mut f32, limsizes: i32, nsizes: &mut i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        *nsizes = 0;
        let co = check_assign_object(&mut state, ob);
        if co != 0 {
            return co;
        }
        let obj = &state.imod.as_ref().unwrap().obj[state.obj];
        for cont in &obj.cont {
            if *nsizes + cont.pts.len() as i32 > limsizes {
                eprintln!("getimodsizes: Model too large for Fortran program");
                return FWRAP_ERROR_FILE_TO_BIG;
            }
            for pt in 0..cont.pts.len() {
                *sizes = imod_point_get_size(obj, cont, pt as i32);
                sizes = sizes.add(1);
            }
            *nsizes += cont.pts.len() as i32;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getcontpointsizes` (`imodel_fwrap.c:923`).
pub unsafe fn getcontpointsizes(
    ob: i32,
    co: i32,
    mut sizes: *mut f32,
    limsizes: i32,
    nsizes: &mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        *nsizes = 0;
        let pt = check_assign_obj_cont(&mut state, ob, co);
        if pt != 0 {
            return pt;
        }
        let cont = &state.imod.as_ref().unwrap().obj[state.obj].cont[state.cont];
        if cont.sizes.is_empty() {
            return FWRAP_NOERROR;
        }
        if cont.pts.len() as i32 > limsizes {
            eprintln!("getcontpointsizes: Too many points for array");
            return FWRAP_ERROR_FILE_TO_BIG;
        }
        for size in &cont.sizes {
            *sizes = *size;
            sizes = sizes.add(1);
        }
        *nsizes = cont.pts.len() as i32;
        FWRAP_NOERROR
    })
}

/// Original: `putcontpointsizes` (`imodel_fwrap.c:947`).
pub unsafe fn putcontpointsizes(ob: i32, co: i32, sizes: *const f32, nsizes: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        /* Saves the sizes in a new element of the size structure array */
        let mut entry = SizeStruct {
            ob: ob - 1,
            co: co - 1,
            num: nsizes,
            sizes: vec![0f32; nsizes.max(0) as usize],
        };
        for i in 0..nsizes.max(0) as usize {
            entry.sizes[i] = *sizes.add(i);
        }
        state.borrow_mut().sizes_put.push(entry);
        FWRAP_NOERROR
    })
}

/// Original: `getimodtimes` (`imodel_fwrap.c:973`).
pub unsafe fn getimodtimes(times: *mut i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let mut coi = 0usize;
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        for obj in &imod.obj {
            for contour in &obj.cont {
                *times.add(coi) = contour.time;
                coi += 1;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodobjtimes` (`imodel_fwrap.c:994`).
pub unsafe fn getimodobjtimes(ob: i32, times: *mut i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let co = check_assign_object(&mut state, ob);
        if co != 0 {
            return co;
        }
        for (index, contour) in state.imod.as_ref().unwrap().obj[state.obj]
            .cont
            .iter()
            .enumerate()
        {
            *times.add(index) = contour.time;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodsurfaces` (`imodel_fwrap.c:1008`).
pub unsafe fn getimodsurfaces(surfs: *mut i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let mut coi = 0usize;
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        for obj in &imod.obj {
            for contour in &obj.cont {
                *surfs.add(coi) = contour.surf;
                coi += 1;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `getobjsurfaces` (`imodel_fwrap.c:1031`).
///
/// `contSave` is the source's saved copy of the contour array, restored over
/// `sObj->cont` after the sort has written surface numbers into the live one.
pub unsafe fn getobjsurfaces(ob: i32, sort_surfs: i32, surfs: *mut i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut retval = FWRAP_NOERROR;
        let co = check_assign_object(&mut state, ob);
        if co != 0 {
            return co;
        }
        let obj_index = state.obj;
        let obj = &mut state.imod.as_mut().unwrap().obj[obj_index];
        if sort_surfs > 0 && !obj.cont.is_empty() {
            /* Duplicate the contour array so that surfaces can be assigned in it */
            let mut cont_save: Vec<Icont> = Vec::with_capacity(obj.cont.len());
            for co in 0..obj.cont.len() {
                let mut copy = Icont::default();
                imod_contour_copy(&obj.cont[co], &mut copy);
                cont_save.push(copy);
            }

            /* Sort the surfaces.  If no meshes, return all zeros; otherwise return surfaces */
            let co = imod_object_sort_surf(obj);
            if co == 2 {
                retval = FWRAP_ERROR_MEMORY;
            } else if co == 1 {
                for co in 0..obj.cont.len() {
                    *surfs.add(co) = 0;
                }
            } else {
                for co in 0..obj.cont.len() {
                    *surfs.add(co) = obj.cont[co].surf;
                }
            }
            obj.cont = cont_save;
        } else {
            /* If not sorting surfaces, return existing values */
            for co in 0..obj.cont.len() {
                *surfs.add(co) = obj.cont[co].surf;
            }
        }
        retval
    })
}

/// Original: `getcontvalue` (`imodel_fwrap.c:1072`).
pub unsafe fn getcontvalue(ob: i32, co: i32, value: &mut f32) -> i32 {
    getpointvalue(ob, co, 0, value)
}

/// Original: `getpointvalue` (`imodel_fwrap.c:1083`).
pub unsafe fn getpointvalue(ob: i32, co: i32, pt: i32, value: &mut f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let i = check_assign_obj_cont(&mut state, ob, co);
        if i != 0 {
            return i;
        }
        let obj = &state.imod.as_ref().unwrap().obj[state.obj];
        let cont = &obj.cont[state.cont];
        let (store, index, after) = if pt < 1 {
            let (index, after) = istore_lookup(&obj.store, co - 1);
            (&obj.store, index, after)
        } else {
            if pt > cont.pts.len() as i32 {
                return FWRAP_ERROR_BAD_OBJNUM;
            }
            let (index, after) = istore_lookup(&cont.store, pt - 1);
            (&cont.store, index, after)
        };
        let index = match index {
            Some(index) => index,
            None => return FWRAP_ERROR_NO_VALUE,
        };
        for item in &store[index..after] {
            if item.type_ == GEN_STORE_VALUE1 {
                *value = item.value.f();
                return FWRAP_NOERROR;
            }
        }
        FWRAP_ERROR_NO_VALUE
    })
}

/// Original: `putcontvalue` (`imodel_fwrap.c:1114`).
pub unsafe fn putcontvalue(ob: i32, co: i32, value: f32) -> i32 {
    putpointvalue(ob, co, 0, value)
}

/// Original: `putpointvalue` (`imodel_fwrap.c:1124`).
pub unsafe fn putpointvalue(ob: i32, co: i32, pt: i32, value: f32) -> i32 {
    FWRAP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.values_put.len() as i32 >= state.max_values {
            state.max_values += 100;
        }
        state.values_put.push(ValueStruct {
            ob: ob - 1,
            co: co - 1,
            pt: pt - 1,
            value,
        });
        FWRAP_NOERROR
    })
}

/// Original: `clearimodobjstore` (`imodel_fwrap.c:1148`).
pub unsafe fn clearimodobjstore(ob: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let object = &mut state.imod.as_mut().unwrap().obj[obj_index];
        object.store.clear();
        FWRAP_NOERROR
    })
}

/// Original: `deleteimodmeshes` (`imodel_fwrap.c:1162`).
pub unsafe fn deleteimodmeshes(ob: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let object = &mut state.imod.as_mut().unwrap().obj[obj_index];
        object.mesh.clear();
        FWRAP_NOERROR
    })
}

/// Original: `deleteimodcont` (`imodel_fwrap.c:1178`).
pub unsafe fn deleteimodcont(ob: i32, co: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_obj_cont(&mut state, ob, co);
        if err != 0 {
            return err;
        }
        let imod = state.imod.as_mut().unwrap();
        imod_set_index(imod, ob - 1, co - 1, -1);
        if imod_delete_contour(imod, co - 1) < 0 {
            return FWRAP_ERROR_FROM_CALL;
        }
        FWRAP_NOERROR
    })
}

/// Original: `deletelistofconts` (`imodel_fwrap.c:1194`).
pub unsafe fn deletelistofconts(ob: i32, contours: *mut i32, num_cont: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let imod = state.imod.as_mut().unwrap();
        imod_set_index(imod, ob - 1, -1, -1);
        for ind in 0..num_cont as usize {
            *contours.add(ind) -= 1;
        }
        let list = std::slice::from_raw_parts(contours, num_cont.max(0) as usize).to_vec();
        let err = imod_delete_list_of_conts(imod, &list, num_cont);
        for ind in 0..num_cont as usize {
            *contours.add(ind) += 1;
        }
        if err < 0 {
            return FWRAP_ERROR_FROM_CALL;
        }
        FWRAP_NOERROR
    })
}

/// Original: `deleteimodpoint` (`imodel_fwrap.c:1214`).
pub unsafe fn deleteimodpoint(ob: i32, co: i32, pt: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_obj_cont(&mut state, ob, co);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let cont_index = state.cont;
        let contour = &mut state.imod.as_mut().unwrap().obj[obj_index].cont[cont_index];
        if pt < 1 || pt > contour.pts.len() as i32 {
            return FWRAP_ERROR_BAD_OBJNUM;
        }
        if crate::imod::libimod::ipoint::imod_point_delete(contour, pt - 1) < 0 {
            return FWRAP_ERROR_FROM_CALL;
        }
        FWRAP_NOERROR
    })
}

/// Original: `addimodpoint` (`imodel_fwrap.c:1236`).
pub unsafe fn addimodpoint(
    ob: i32,
    co: i32,
    pt_or_z: f32,
    if_by_z: i32,
    x: f32,
    y: f32,
    z: f32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        /* B3DNINT (`b3dutil.h:33`) */
        let mut pt = (if pt_or_z as f64 >= 0. {
            (pt_or_z as f64 + 0.5).floor()
        } else {
            (pt_or_z as f64 - 0.5).ceil()
        }) as i32
            - 1;
        let mut state = state.borrow_mut();
        let err = check_assign_obj_cont(&mut state, ob, co);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let cont_index = state.cont;
        let contour = &mut state.imod.as_mut().unwrap().obj[obj_index].cont[cont_index];
        if pt < 0 && if_by_z == 0 {
            return FWRAP_ERROR_BAD_OBJNUM;
        }
        let psize = contour.pts.len() as i32;
        if if_by_z != 0 {
            pt = 0;
            while pt < psize {
                if contour.pts[pt as usize].z >= pt_or_z {
                    break;
                }
                pt += 1;
            }
        }
        pt = if pt < psize { pt } else { psize };
        let point = Ipoint { x, y, z };
        if imod_point_add(contour, Some(point), pt) == 0 {
            return FWRAP_ERROR_FROM_CALL;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getobjvaluethresh` (`imodel_fwrap.c:1265`).
pub unsafe fn getobjvaluethresh(ob: i32, thresh: &mut f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut vmin = 0f32;
        let mut vmax = 0f32;
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj = &state.imod.as_ref().unwrap().obj[state.obj];
        if istore_get_min_max(
            &obj.store,
            obj.cont.len() as i32,
            GEN_STORE_MINMAX1,
            &mut vmin,
            &mut vmax,
        ) == 0
        {
            return FWRAP_ERROR_NO_VALUE;
        }
        *thresh = (obj.valblack as f64 * (vmax - vmin) as f64 / 255. + vmin as f64) as f32;
        FWRAP_NOERROR
    })
}

/// Original: `getobjskiplowvalues` (`imodel_fwrap.c:1281`).
pub unsafe fn getobjskiplowvalues(ob: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj = &state.imod.as_ref().unwrap().obj[state.obj];
        ((obj.flags & IMOD_OBJFLAG_USE_VALUE) != 0
            && (obj.matflags2 as u32 & MATFLAGS2_SKIP_LOW) != 0) as i32
    })
}

/// Original: `findaddminmax1value` (`imodel_fwrap.c:1294`).
pub unsafe fn findaddminmax1value(ob: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let object = &mut state.imod.as_mut().unwrap().obj[obj_index];
        if istore_find_add_min_max1(object) > 0 {
            return FWRAP_ERROR_FROM_CALL;
        }
        FWRAP_NOERROR
    })
}

/// Original: `putimod` (`imodel_fwrap.c:1334`).
pub unsafe fn putimod(
    ibase: *const i32,
    npt: *const i32,
    coord: *const [f32; 3],
    cindex: *const i32,
    color: *const [i32; 2],
    _npoint: i32,
    nobject: i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut nobj = 0usize;
        let names_put = state.names_put.clone();

        if state.imod.is_none() {
            state.imod = imod_new();
            state.imod.as_mut().unwrap().obj.clear();
        }

        let mut mincolor = 256 - state.imod.as_ref().unwrap().obj.len() as i32;

        /* Find minimum color, and maximum object #; get arrays */
        /* Skip empty contours unless we are in partial mode, where they are needed
        to signal that an existing object is now empty */
        for object in 0..nobject as usize {
            if *npt.add(object) == 0 && state.partial_mode == 0 {
                continue;
            }
            if mincolor > (*color.add(object))[1] {
                mincolor = (*color.add(object))[1];
            }
        }
        let maxobj = 256 - mincolor;

        let mut objlookup: Vec<i32> = vec![0; maxobj.max(0) as usize];
        let mut nsaved: Vec<i32> = vec![0; maxobj.max(0) as usize];

        /*
         * Mark obj lookup array to empty=-2 or used=-1 for new data passed in
         * or index >= 0 for an existing object.
         */
        for ob in 0..maxobj as usize {
            objlookup[ob] = OBJ_EMPTY;
            nsaved[ob] = 0;
        }

        /* Fill lookup table for existing objects; if in partial mode set nsaved
        to full count to keep contours from being deleted later */
        let objsize = state.imod.as_ref().unwrap().obj.len();
        for ob in 0..objsize {
            objlookup[((255 - mincolor) - ob as i32) as usize] = ob as i32;
            nobj += 1;
            if state.partial_mode != 0 {
                nsaved[ob] = state.imod.as_ref().unwrap().obj[ob].cont.len() as i32;
            }
        }

        /* For all non-empty contours passed back, mark an empty object as having
        data. */
        for object in 0..nobject as usize {
            if *npt.add(object) == 0 && state.partial_mode == 0 {
                continue;
            }
            let ob = ((*color.add(object))[1] - mincolor) as usize;
            if objlookup[ob] >= 0 {
                nsaved[objlookup[ob] as usize] = 0;
            }
            if *npt.add(object) == 0 {
                continue;
            }
            if objlookup[ob] == OBJ_EMPTY {
                objlookup[ob] = OBJ_HAS_DATA;
            }
        }

        /*  Do not remove old object data yet */

        /*
         * Create additional imod objects if needed.
         */
        for ob in (0..maxobj as usize).rev() {
            if objlookup[ob] != OBJ_HAS_DATA {
                continue;
            }
            let imod = state.imod.as_mut().unwrap();
            objlookup[ob] = imod.obj.len() as i32;
            imod_new_object(imod);
            let wimpno = ob as i32 + mincolor;

            /* Just use top 6 colors and let new color scheme hold after that */
            if wimpno >= 250 {
                imod.obj[nobj].red = WMOD_COLORS[(wimpno - 247) as usize][0];
                imod.obj[nobj].green = WMOD_COLORS[(wimpno - 247) as usize][1];
                imod.obj[nobj].blue = WMOD_COLORS[(wimpno - 247) as usize][2];
            }
            imod.obj[nobj].flags |= IMOD_OBJFLAG_OPEN;

            /* Find object name in list if any */
            let mut ci = 0;
            for name_put in &names_put {
                if name_put.ob == 255 - (ob as i32 + mincolor) {
                    let src = name_put.name.as_bytes();
                    if !src.is_empty() {
                        // `imodel_fwrap.c:1435` `strncpy(name, src, IOBJ_STRSIZE - 1)`.
                        // `Iobj::name` is `[u8; N]` now (NATIVE.md §3), so the C
                        // string is copied byte for byte with the same bound —
                        // *and* the rest of the field is zero-filled, which is
                        // what `strncpy` does and a bounded copy does not.
                        let count = src.len().min(IOBJ_STRSIZE - 1);
                        imod.obj[nobj].name[..count].copy_from_slice(&src[..count]);
                        imod.obj[nobj].name[count..IOBJ_STRSIZE - 1].fill(0);
                    }
                    imod.obj[nobj].name[IOBJ_STRSIZE - 1] = 0;
                    ci = 1;
                }
            }
            if ci == 0 {
                let name = format!("Fmod # {wimpno}");
                let bytes = name.as_bytes();
                let count = bytes.len().min(IOBJ_STRSIZE - 1);
                imod.obj[nobj].name[..count].copy_from_slice(&bytes[..count]);
                imod.obj[nobj].name[count] = 0;
            }
            nobj += 1;
        }

        /*
         * Copy all wimp objects to imod contours.
         */
        for object in 0..nobject as usize {
            /* Don't even look up if empty and out of range */
            if *npt.add(object) == 0
                && ((*color.add(object))[1] < mincolor || (*color.add(object))[1] > 255)
            {
                continue;
            }
            let ob = objlookup[((*color.add(object))[1] - mincolor) as usize];
            if ob < 0 {
                if *npt.add(object) != 0 {
                    eprintln!("putimod warning: bad object bounds {ob}");
                }
                continue;
            }
            let imod = state.imod.as_mut().unwrap();
            if ob > imod.obj.len() as i32 {
                eprintln!("putimod warning: bad object bounds {ob}");
                continue;
            }
            let ob = ob as usize;

            if nsaved[ob] < imod.obj[ob].cont.len() as i32 {
                /* Existing contour */
                let cont_index = nsaved[ob] as usize;
                let want = *npt.add(object) as usize;
                let cont = &mut imod.obj[ob].cont[cont_index];

                /* For empty contour, clear out existing data */
                if want == 0 && !cont.pts.is_empty() {
                    imod_contour_clear(cont);
                } else if want != cont.pts.len() {
                    cont.pts.resize(want, Ipoint::default());
                    if !cont.sizes.is_empty() {
                        cont.sizes.resize(want, 0.);
                    }
                }
                cont.pts.resize(want, Ipoint::default());
                for pt in 0..want {
                    let ci = *cindex.add(*ibase.add(object) as usize + pt) - 1;
                    cont.pts[pt].x = (*coord.add(ci as usize))[0];
                    cont.pts[pt].y = (*coord.add(ci as usize))[1];
                    cont.pts[pt].z = (*coord.add(ci as usize))[2];
                }
            } else {
                /* otherwise, if # saved = # of contours, add a new contour */
                let mut cont = Icont::default();
                let want = *npt.add(object) as usize;
                cont.pts = vec![Ipoint::default(); want];

                for pt in 0..want {
                    let ci = *cindex.add(*ibase.add(object) as usize + pt) - 1;
                    cont.pts[pt].x = (*coord.add(ci as usize))[0];
                    cont.pts[pt].y = (*coord.add(ci as usize))[1];
                    cont.pts[pt].z = (*coord.add(ci as usize))[2];
                }

                imod_object_add_contour(&mut imod.obj[ob], cont);
            }
            nsaved[ob] += 1;
        }

        /*
         *  Remove old contours that were not replaced
         */
        let imod = state.imod.as_mut().unwrap();
        for ob in 0..imod.obj.len() {
            if imod_contours_delete_to_end(&mut imod.obj[ob], nsaved[ob]) != 0 {
                return FWRAP_ERROR_MEMORY;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `writeimod` (`imodel_fwrap.c:1546`).
pub unsafe fn writeimod(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let filename = fortran_string(fname, fsize);
        if state.imod.is_none() {
            return FWRAP_ERROR_NO_MODEL;
        }
        let flags_put = state.flags_put.clone();
        let sizes_put = state.sizes_put.clone();
        let values_put = state.values_put.clone();
        let maxes_put = state.maxes_put;
        let (xmax_put, ymax_put, zmax_put) = (state.xmax_put, state.ymax_put, state.zmax_put);
        let zscale_put = state.zscale_put;
        let rotation_put = state.rotation_put;
        let write_as_wimp = std::mem::replace(&mut state.write_as_wimp, false);
        let path = filename;

        /*
         *  Translate coordinates to match reference image.
         */
        let imod = state.imod.as_mut().unwrap();
        if let Some(iref) = imod.ref_image {
            let mut mat = match imod_mat_new(3) {
                Some(mat) => mat,
                None => return FWRAP_ERROR_MEMORY,
            };

            imod_mat_trans(&mut mat, &iref.ctrans);

            let pnt = Ipoint {
                x: 1.0 / iref.cscale.x,
                y: 1.0 / iref.cscale.y,
                z: 1.0 / iref.cscale.z,
            };
            imod_mat_scale(&mut mat, &pnt);

            imod_transform(Some(imod), Some(&mat));
            imod_mat_delete(&mut mat);
        }

        /* Reflip data back if necessary */
        if imod.flags & IMODF_FLIPYZ != 0 {
            imod_flip_yz(imod);
        }

        /* Put out the flag data */
        let flagmask = (1 << FLAG_VALUE_SHIFT) - 1;
        let num_flags = flags_put.len() / 2;
        for i in 0..num_flags {
            let ob = flags_put[2 * i];
            let mut flag = flags_put[2 * i + 1];
            let value = flag >> FLAG_VALUE_SHIFT;
            flag &= flagmask;
            if ob >= 0 && ob < imod.obj.len() as i32 {
                /* Save some data in the current view if it has an object view for
                this object */
                let ob = ob as usize;
                let cview = imod.cview;
                let has_objview = cview > 0
                    && cview < imod.view.len() as i32
                    && imod.view[cview as usize].objview.len() > ob;

                match flag {
                    0 => {
                        imod.obj[ob].flags &= !IMOD_OBJFLAG_OPEN;
                        imod.obj[ob].flags &= !IMOD_OBJFLAG_SCAT;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            imod.view[cview as usize].objview[ob].flags = f;
                        }
                    }
                    1 => {
                        imod.obj[ob].flags |= IMOD_OBJFLAG_OPEN;
                        imod.obj[ob].flags &= !IMOD_OBJFLAG_SCAT;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            imod.view[cview as usize].objview[ob].flags = f;
                        }
                    }
                    2 => {
                        imod.obj[ob].flags &= !IMOD_OBJFLAG_OPEN;
                        imod.obj[ob].flags |= IMOD_OBJFLAG_SCAT;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            imod.view[cview as usize].objview[ob].flags = f;
                        }
                    }
                    SCAT_SIZE_FLAG => {
                        imod.obj[ob].pdrawsize = value;
                        if has_objview {
                            let v = imod.obj[ob].pdrawsize;
                            imod.view[cview as usize].objview[ob].pdrawsize = v;
                        }
                    }
                    SYMBOL_SIZE_FLAG => imod.obj[ob].symsize = value as u8,
                    SYMBOL_TYPE_FLAG => imod.obj[ob].symbol = value as u8,
                    SYMBOL_FLAGS_FLAG => imod.obj[ob].symflags = value as u8,
                    SYMBOL_WIDTH_FLAG => imod.obj[ob].linewidth2 = value as u8,
                    OBJECT_COLOR_FLAG => {
                        imod.obj[ob].red = ((value & 255) as f64 / 255.) as f32;
                        imod.obj[ob].green = (((value >> 8) & 255) as f64 / 255.) as f32;
                        imod.obj[ob].blue = (((value >> 16) & 255) as f64 / 255.) as f32;
                        if has_objview {
                            let (r, g, b) =
                                (imod.obj[ob].red, imod.obj[ob].green, imod.obj[ob].blue);
                            imod.view[cview as usize].objview[ob].red = r;
                            imod.view[cview as usize].objview[ob].green = g;
                            imod.view[cview as usize].objview[ob].blue = b;
                        }
                    }
                    USE_VALUE_FLAGS => {
                        imod.obj[ob].flags |= IMOD_OBJFLAG_USE_VALUE;
                        imod.obj[ob].matflags2 |= (MATFLAGS2_CONSTANT | MATFLAGS2_SKIP_LOW) as u8;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            let m = imod.obj[ob].matflags2;
                            imod.view[cview as usize].objview[ob].flags = f;
                            imod.view[cview as usize].objview[ob].matflags2 = m;
                        }
                    }
                    VAL_BLACKWHITE_FLAG => {
                        imod.obj[ob].valblack = (value & 255) as u8;
                        imod.obj[ob].valwhite = ((value >> 8) & 255) as u8;
                        if has_objview {
                            let (bl, wh) = (imod.obj[ob].valblack, imod.obj[ob].valwhite);
                            imod.view[cview as usize].objview[ob].valblack = bl;
                            imod.view[cview as usize].objview[ob].valwhite = wh;
                        }
                    }
                    PNT_ON_SEC_FLAG => {
                        imod.obj[ob].flags |= IMOD_OBJFLAG_PNT_ON_SEC;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            imod.view[cview as usize].objview[ob].flags = f;
                        }
                    }
                    THICKEN_CONT_FLAG => {
                        imod.obj[ob].flags |= IMOD_OBJFLAG_THICK_CONT;
                        if has_objview {
                            let f = imod.obj[ob].flags;
                            imod.view[cview as usize].objview[ob].flags = f;
                        }
                    }
                    _ => {}
                }
            }
        }

        if maxes_put != 0 {
            imod.xmax = xmax_put;
            imod.ymax = ymax_put;
            imod.zmax = zmax_put;
        }

        /* Save Zscale, save rotation in current view if there is one */
        if zscale_put != NO_VALUE_PUT {
            imod.zscale = zscale_put;
        }
        if imod.cview > 0 && rotation_put.x != NO_VALUE_PUT {
            let cview = imod.cview as usize;
            imod.view[cview].rot = rotation_put;
        }

        /* Put out the sizes into the contours */
        for entry in &sizes_put {
            /* Ignore illegal object or contour numbers */
            let put_ob = entry.ob;
            if put_ob < 0 || put_ob >= imod.obj.len() as i32 {
                continue;
            }
            let put_co = entry.co;
            if put_co < 0 || put_co >= imod.obj[put_ob as usize].cont.len() as i32 {
                continue;
            }

            /* Fill up to the limit on the number of point in array and contour */
            let cont = &mut imod.obj[put_ob as usize].cont[put_co as usize];
            let num = entry.num;
            let value = if num < cont.pts.len() as i32 {
                num
            } else {
                cont.pts.len() as i32
            };
            for j in 0..value as usize {
                let size = entry.sizes[j];
                if size >= 0. {
                    imod_point_set_size(cont, j as i32, size);
                }
            }
        }

        /* Put out general values */
        if !values_put.is_empty() {
            let mut store = Istore {
                type_: GEN_STORE_VALUE1,
                flags: (GEN_STORE_FLOAT << 2) | GEN_STORE_ONEPOINT,
                index: StoreUnion::from_i(0),
                value: StoreUnion::from_i(0),
            };
            for entry in values_put {
                /* Ignore illegal object or contour numbers */
                if entry.ob < 0 || entry.ob >= imod.obj.len() as i32 {
                    continue;
                }
                if entry.co < 0 || entry.co >= imod.obj[entry.ob as usize].cont.len() as i32 {
                    continue;
                }

                /* Insert value into contour or object stores */
                store.value = StoreUnion::from_f(entry.value);
                if entry.pt < 0 {
                    store.index = StoreUnion::from_i(entry.co);
                    istore_insert_change(&mut imod.obj[entry.ob as usize].store, store);
                } else {
                    let cont = &mut imod.obj[entry.ob as usize].cont[entry.co as usize];
                    if entry.pt > cont.pts.len() as i32 {
                        continue;
                    }
                    store.index = StoreUnion::from_i(entry.pt);
                    istore_insert_change(&mut cont.store, store);
                }
            }

            /* Now need to set min/maxes for all stores */
            for ob in 0..imod.obj.len() {
                istore_find_add_min_max1(&mut imod.obj[ob]);
            }
        }

        /* Complete the set of object views just before writing */
        imod_objview_complete(imod);
        let retcode;
        if write_as_wimp {
            let mut out = match ImodFile::open(&path, "wb") {
                Some(out) => out,
                None => {
                    return FWRAP_ERROR_OPENING_FILE;
                }
            };
            retcode = match imod_to_wmod(imod, &mut out, &path) {
                Ok(()) => 0,
                Err(code) => code,
            };
        } else {
            let mut out = match ImodFile::open(&path, "wb") {
                Some(out) => out,
                None => {
                    return FWRAP_ERROR_OPENING_FILE;
                }
            };
            retcode = match imod_write_file(imod, &mut out) {
                Ok(()) => 0,
                Err(code) => code,
            };
        }
        retcode
    })
}

/// Original: `imodwriteaswimp` (`imodel_fwrap.c:1774`).
pub unsafe fn imodwriteaswimp(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| state.borrow_mut().write_as_wimp = true);
    writeimod(fname, fsize)
}

/// Original: `putimodscat` (`imodel_fwrap.c:1784`).
pub unsafe fn putimodscat(ob: i32, verts: *const f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let co = check_assign_object(&mut state, ob);
        if co != 0 {
            return co;
        }
        let obj_index = state.obj;
        let object = &mut state.imod.as_mut().unwrap().obj[obj_index];
        if iobj_scat(object.flags) == 0 {
            return 2;
        }
        let mut v = 0usize;
        for contour in &mut object.cont {
            for point in &mut contour.pts {
                point.x = *verts.add(v);
                v += 1;
                point.y = *verts.add(v);
                v += 1;
                point.z = *verts.add(v);
                v += 1;
            }
        }
        0
    })
}

/// Original: `putimodmesh` (`imodel_fwrap.c:1808`).
pub unsafe fn putimodmesh(mut verts: *const f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut trans = Ipoint::default();
        if get_mesh_trans(&state, &mut trans) != 0 {
            return FWRAP_ERROR_NO_MODEL;
        }
        let imod = state.imod.as_mut().unwrap();
        let ob = imod.cindex.object as usize;

        imod.obj[ob].mesh[0].flag |= IMESH_FLAG_NMAG;
        let mi = imod.obj[ob].mesh[0].vert.len();

        let mut i = 0usize;
        while i < mi {
            imod.obj[ob].mesh[0].vert[i].x = *verts - trans.x;
            verts = verts.add(1);
            imod.obj[ob].mesh[0].vert[i].y = *verts - trans.y;
            verts = verts.add(1);
            imod.obj[ob].mesh[0].vert[i].z = *verts - trans.z;
            verts = verts.add(1);
            imod.obj[ob].mesh[0].vert[i + 1].x = *verts;
            verts = verts.add(1);
            imod.obj[ob].mesh[0].vert[i + 1].y = *verts;
            verts = verts.add(1);
            imod.obj[ob].mesh[0].vert[i + 1].z = *verts;
            verts = verts.add(1);
            i += 2;
        }
        FWRAP_NOERROR
    })
}

/// Original: `newimod` (`imodel_fwrap.c:1836`).
pub unsafe fn newimod() -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        delete_fimod(&mut state);
        state.imod = imod_new();
        let Some(imod) = state.imod.as_mut() else {
            return FWRAP_ERROR_NO_MODEL;
        };
        imod.obj.clear();
        FWRAP_NOERROR
    })
}

/// Original: `deleteimod` (`imodel_fwrap.c:1850`).
pub unsafe fn deleteimod() -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        if state.imod.is_none() {
            return FWRAP_ERROR_NO_MODEL;
        }
        delete_fimod(&mut state);
        FWRAP_NOERROR
    })
}

/// Original: `deleteiobj` (`imodel_fwrap.c:1861`).
pub unsafe fn deleteiobj() -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let Some(imod) = state.imod.as_mut() else {
            return FWRAP_ERROR_NO_MODEL;
        };
        imod_objviews_free(imod);
        imod_objects_delete(&mut imod.obj);
        imod.obj.clear();
        FWRAP_NOERROR
    })
}

/// Original: `getimodobjsize` (`imodel_fwrap.c:1878`).
pub unsafe fn getimodobjsize() -> i32 {
    FWRAP_STATE.with(|state| {
        state
            .borrow()
            .imod
            .as_ref()
            .map_or(-1, |imod| imod.obj.len() as i32)
    })
}

/// Original: `getimodheado` (`imodel_fwrap.c:1889`).
pub unsafe fn getimodheado(um: &mut f32, zscale: &mut f32) -> i32 {
    FWRAP_STATE.with(|state| {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        if imod.units == IMOD_UNIT_PIXEL {
            return FWRAP_ERROR_NO_PIXEL_SIZE;
        }
        let exval = imod.units + 6;
        *um = (imod.pixsize as f64 * 10.0f64.powf(exval as f64)) as f32;
        *zscale = imod.zscale;
        FWRAP_NOERROR
    })
}

/// Original: `getimodmaxes` (`imodel_fwrap.c:1908`).
pub unsafe fn getimodmaxes(xmax: &mut i32, ymax: &mut i32, zmax: &mut i32) -> i32 {
    FWRAP_STATE.with(|state| {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        *xmax = imod.xmax;
        *ymax = imod.ymax;
        *zmax = imod.zmax;
        FWRAP_NOERROR
    })
}

/// Original: `getzfromminuspt5` (`imodel_fwrap.c:1923`).
pub unsafe fn getzfromminuspt5() -> i32 {
    FWRAP_STATE.with(|state| {
        state
            .borrow()
            .imod
            .as_ref()
            .map_or(FWRAP_ERROR_NO_MODEL, |imod| {
                (imod.flags & IMODF_Z_FROM_MINUSPT5 != 0) as i32
            })
    })
}

/// Original: `getimodhead` (`imodel_fwrap.c:1935`).
///
/// Source defect, translated as written: when `refImage` is NULL the three
/// offsets are set to zero, but the `else if`/`else` chain leaves them
/// untouched in no case — the caller's values are always assigned.
pub unsafe fn getimodhead(
    um: &mut f32,
    zscale: &mut f32,
    xoffset: &mut f32,
    yoffset: &mut f32,
    zoffset: &mut f32,
    ifflip: &mut i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };

        let exval = imod.units + 6;
        *um = (imod.pixsize as f64 * 10.0f64.powf(exval as f64)) as f32;
        *zscale = imod.zscale;

        let iref = imod.ref_image;

        /* DNM 7/20/02: If image origin has been stored in otrans, return that
        because that is what is needed to get to full volume index coords */
        match iref {
            None => {
                *xoffset = 0.;
                *yoffset = 0.;
                *zoffset = 0.;
            }
            Some(iref) => {
                if imod.flags & IMODF_OTRANS_ORIGIN != 0 {
                    *xoffset = -iref.otrans.x;
                    *yoffset = -iref.otrans.y;
                    *zoffset = -iref.otrans.z;
                } else {
                    *xoffset = -iref.ctrans.x;
                    *yoffset = -iref.ctrans.y;
                    *zoffset = -iref.ctrans.z;
                }
            }
        }

        if imod.flags & IMODF_FLIPYZ != 0 {
            *ifflip = 1;
        } else {
            *ifflip = 0;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodscales` (`imodel_fwrap.c:1981`).
pub unsafe fn getimodscales(ximscale: &mut f32, yimscale: &mut f32, zimscale: &mut f32) -> i32 {
    FWRAP_STATE.with(|state| {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        match imod.ref_image {
            Some(iref) => {
                *ximscale = iref.cscale.x;
                *yimscale = iref.cscale.y;
                *zimscale = iref.cscale.z;
            }
            None => {
                *ximscale = 1.;
                *yimscale = 1.;
                *zimscale = 1.;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `putimageref` (`imodel_fwrap.c:2006`).
pub unsafe fn putimageref(delta: *const f32, origin: *const f32, tilt: *const f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let Some(imod) = state.imod.as_mut() else {
            return FWRAP_ERROR_NO_MODEL;
        };
        if imod.ref_image.is_none() {
            imod.ref_image = Some(Iref_image {
                oscale: Ipoint::default(),
                otrans: Ipoint::default(),
                orot: Ipoint::default(),
                cscale: Ipoint::default(),
                ctrans: Ipoint::default(),
                crot: Ipoint::default(),
            });
        }
        let iref = imod.ref_image.as_mut().unwrap();
        iref.cscale.x = *delta.add(0);
        iref.cscale.y = *delta.add(1);
        iref.cscale.z = *delta.add(2);
        iref.ctrans.x = *origin.add(0);
        iref.ctrans.y = *origin.add(1);
        iref.ctrans.z = *origin.add(2);
        iref.otrans = iref.ctrans;
        iref.crot.x = *tilt.add(0);
        iref.crot.y = *tilt.add(1);
        iref.crot.z = *tilt.add(2);
        imod.flags |= IMODF_OTRANS_ORIGIN | IMODF_TILTOK;
        FWRAP_NOERROR
    })
}

/// Original: `imodhasimageref` (`imodel_fwrap.c:2036`).
pub unsafe fn imodhasimageref() -> i32 {
    FWRAP_STATE.with(|state| {
        state
            .borrow()
            .imod
            .as_ref()
            .map_or(FWRAP_ERROR_NO_MODEL, |imod| imod.ref_image.is_some() as i32)
    })
}

/// Original: `getimodflags` (`imodel_fwrap.c:2048`).
pub unsafe fn getimodflags(flags: *mut i32, limflags: i32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        if imod.obj.len() as i32 > limflags {
            eprintln!("getimodflags: Too many objects for Fortran program");
            return FWRAP_ERROR_FILE_TO_BIG;
        }
        for i in 0..limflags as usize {
            *flags.add(i) = -1;
        }
        for i in 0..imod.obj.len() {
            if imod.obj[i].cont.is_empty() {
                continue;
            }
            *flags.add(i) = 0;
            if imod.obj[i].flags & IMOD_OBJFLAG_OPEN != 0 {
                *flags.add(i) = 1;
            }
            if imod.obj[i].flags & IMOD_OBJFLAG_SCAT != 0 {
                *flags.add(i) = 2;
            }
            if !imod.obj[i].mesh.is_empty() {
                *flags.add(i) += 4;
            }
        }
        FWRAP_NOERROR
    })
}

/// Original: `getmodelname` (`imodel_fwrap.c:2077`).
pub unsafe fn getmodelname(fname: *mut c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let state = state.borrow();
        let imod = match state.imod.as_ref() {
            Some(imod) => imod,
            None => return FWRAP_ERROR_NO_MODEL,
        };
        let limit = fsize.max(0) as usize;
        let mut index = 0usize;
        while index < imod.name.len() && index < limit && imod.name[index] != 0 {
            *fname.add(index) = imod.name[index] as c_char;
            index += 1;
        }
        if index < imod.name.len() && imod.name[index] != 0 {
            return FWRAP_ERROR_STRING_LEN;
        }
        while index < limit {
            *fname.add(index) = b' ' as c_char;
            index += 1;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodobjname` (`imodel_fwrap.c:2090`).
pub unsafe fn getimodobjname(ob: i32, fname: *mut c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let err = check_assign_object(&mut state, ob);
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let obj = &state.imod.as_ref().unwrap().obj[obj_index];
        let mut i = 0i32;
        while i < fsize && (i as usize) < IOBJ_STRSIZE && obj.name[i as usize] != 0 {
            *fname.add(i as usize) = obj.name[i as usize] as c_char;
            i += 1;
        }
        while i < fsize {
            *fname.add(i as usize) = 0x20;
            i += 1;
        }
        FWRAP_NOERROR
    })
}

/// Original: `getimodclip` (`imodel_fwrap.c:2108`).
pub unsafe fn getimodclip(objnum: i32, clip: *mut f32) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut iout = 0usize;
        let err = check_assign_object(&mut state, objnum);
        if err != 0 {
            return err;
        }

        let obj_index = state.obj;
        let imod = state.imod.as_mut().unwrap();
        imod.cindex.object = objnum - 1;
        /* DNM 9/19/04: modified for multiple clip planes */
        let obj = &imod.obj[obj_index];
        let mut nout = obj.clips.count as i32;
        if nout > FWRAP_MAX_CLIP_PLANES {
            nout = FWRAP_MAX_CLIP_PLANES;
        }

        for i in 0..nout as usize {
            *clip.add(iout) = obj.clips.normal[i].x;
            iout += 1;
            *clip.add(iout) = obj.clips.normal[i].y;
            iout += 1;
            *clip.add(iout) = obj.clips.normal[i].z / imod.zscale;
            iout += 1;
            *clip.add(iout) = (obj.clips.normal[i].x * obj.clips.point[i].x)
                + (obj.clips.normal[i].y * obj.clips.point[i].y)
                + (obj.clips.normal[i].z * obj.clips.point[i].z);
            *clip.add(iout) *= imod.pixsize;
            iout += 1;
        }
        nout
    })
}

/* DNM 6/8/01: change this to save the values until a model is written */
/// Original: `putimodmaxes` (`imodel_fwrap.c:2137`).
pub unsafe fn putimodmaxes(xmax: i32, ymax: i32, zmax: i32) -> i32 {
    FWRAP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.xmax_put = xmax;
        state.ymax_put = ymax;
        state.zmax_put = zmax;
        state.maxes_put = 1;
        FWRAP_NOERROR
    })
}

/// Original: `putimodflag` (`imodel_fwrap.c:2153`).
pub unsafe fn putimodflag(objnum: i32, flag: i32) {
    FWRAP_STATE.with(|state| state.borrow_mut().flags_put.extend([objnum - 1, flag]));
}

/// Original: `putimodzscale` (`imodel_fwrap.c:2174`).
pub unsafe fn putimodzscale(zscale: f32) {
    FWRAP_STATE.with(|state| state.borrow_mut().zscale_put = zscale);
}

/// Original: `putimodrotation` (`imodel_fwrap.c:2182`).
pub unsafe fn putimodrotation(xrot: f32, yrot: f32, zrot: f32) {
    FWRAP_STATE.with(|state| {
        state.borrow_mut().rotation_put = Ipoint {
            x: xrot,
            y: yrot,
            z: zrot,
        };
    });
}

/* DNM 12/3/01: added so that old wimp models can be converted */
/// Original: `fromvmsfloats` (`imodel_fwrap.c:2190`).
pub fn fromvmsfloats(data: &mut [u8], amt: i32) {
    imod_from_vms_floats(data, amt);
}

/// Original: `getscatsize` (`imodel_fwrap.c:2198`).
pub unsafe fn getscatsize(objnum: i32, size: &mut i32) -> i32 {
    FWRAP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let err = unsafe { check_assign_object(&mut state, objnum) };
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        *size = state.imod.as_ref().unwrap().obj[obj_index].pdrawsize;
        FWRAP_NOERROR
    })
}

/* DNM 5/15/02: put scattered point sizes out in flags with an offset */
/// Original: `putscatsize` (`imodel_fwrap.c:2211`).
pub unsafe fn putscatsize(objnum: i32, size: i32) {
    let flag = (size << FLAG_VALUE_SHIFT) + SCAT_SIZE_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putsymtype` (`imodel_fwrap.c:2220`).
pub unsafe fn putsymtype(objnum: i32, type_: i32) {
    let flag = (type_ << FLAG_VALUE_SHIFT) + SYMBOL_TYPE_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putsymflags` (`imodel_fwrap.c:2230`).
pub unsafe fn putsymflags(objnum: i32, flags: i32) {
    let flag = (flags << FLAG_VALUE_SHIFT) + SYMBOL_FLAGS_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putlinewidth` (`imodel_fwrap.c:2239`).
pub unsafe fn putlinewidth(objnum: i32, width: i32) {
    let flag = (width << FLAG_VALUE_SHIFT) + SYMBOL_WIDTH_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `getobjcolor` (`imodel_fwrap.c:2249`).
pub unsafe fn getobjcolor(objnum: i32, red: &mut i32, green: &mut i32, blue: &mut i32) -> i32 {
    FWRAP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let err = unsafe { check_assign_object(&mut state, objnum) };
        if err != 0 {
            return err;
        }
        let obj_index = state.obj;
        let obj = &state.imod.as_ref().unwrap().obj[obj_index];
        /* B3DNINT (`b3dutil.h:33`) */
        *red = b3d_nint_local(255. * obj.red as f64);
        *green = b3d_nint_local(255. * obj.green as f64);
        *blue = b3d_nint_local(255. * obj.blue as f64);
        FWRAP_NOERROR
    })
}

/// `B3DNINT` (`b3dutil.h:33`): `floor(x + 0.5)` for non-negative, `ceil(x - 0.5)`
/// otherwise.
fn b3d_nint_local(x: f64) -> i32 {
    (if x >= 0. {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }) as i32
}

/* DNM 4/1/2/05: Add object color. */
/// Original: `putobjcolor` (`imodel_fwrap.c:2266`).
pub unsafe fn putobjcolor(objnum: i32, red: i32, green: i32, blue: i32) {
    let r = 255.min(0.max(red));
    let g = 255.min(0.max(green));
    let b = 255.min(0.max(blue));
    let flag =
        ((r << FLAG_VALUE_SHIFT) | (g << (FLAG_VALUE_SHIFT + 8)) | (b << (FLAG_VALUE_SHIFT + 16)))
            + OBJECT_COLOR_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putsymsize` (`imodel_fwrap.c:2279`).
pub unsafe fn putsymsize(objnum: i32, size: i32) {
    let flag = (size << FLAG_VALUE_SHIFT) + SYMBOL_SIZE_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putvalblackwhite` (`imodel_fwrap.c:2289`).
pub unsafe fn putvalblackwhite(objnum: i32, black: i32, white: i32) {
    let flag =
        (black << FLAG_VALUE_SHIFT) + (white << (FLAG_VALUE_SHIFT + 8)) + VAL_BLACKWHITE_FLAG;
    putimodflag(objnum, flag);
}

/// Original: `putmodelname` (`imodel_fwrap.c:2299`).
pub unsafe fn putmodelname(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let Some(imod) = state.imod.as_mut() else {
            return FWRAP_ERROR_NO_MODEL;
        };
        let tmpstr = fortran_string(fname, fsize);
        let bytes = tmpstr.as_bytes();
        let count = bytes.len().min(IMOD_STRSIZE - 1);
        // `strncpy(sImod->name, tmpstr, IMOD_STRSIZE - 1)` does not stop at
        // the source string: it zero-**pads** the whole destination out to
        // `IMOD_STRSIZE - 1`.  Copying only `count` bytes leaves whatever the
        // previous model left in `name`, which `imodel_write` then emits — so
        // the written model carried heap residue where the source writes NULs.
        imod.name[..count].copy_from_slice(&bytes[..count]);
        imod.name[count..IMOD_STRSIZE - 1].fill(0);
        imod.name[IMOD_STRSIZE - 1] = 0;
        FWRAP_NOERROR
    })
}

/// Original: `putimodobjname` (`imodel_fwrap.c:2316`).
pub unsafe fn putimodobjname(objnum: i32, fname: *const c_char, fsize: FortStrLenT) {
    FWRAP_STATE.with(|state| {
        state.borrow_mut().names_put.push(NameStruct {
            ob: objnum - 1,
            name: fortran_string(fname, fsize),
        });
    });
}

/// Original: `getimodnesting` (`imodel_fwrap.c:2343`).
pub unsafe fn getimodnesting(
    ob: i32,
    in_only: i32,
    level: *mut i32,
    in_index: *mut i32,
    in_cont: *mut i32,
    out_index: *mut i32,
    out_cont: *mut i32,
    array_size: i32,
) -> i32 {
    FWRAP_STATE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let mut contz: Vec<i32> = Vec::new();
        let mut numatz: Vec<i32> = Vec::new();
        let mut contatz: Vec<Vec<i32>> = Vec::new();
        let mut zlist: Vec<i32> = Vec::new();
        let mut zmin = 0;
        let mut zmax = 0;
        let mut zlsize = 0;
        let mut nummax = 0;
        let mut numnests = 0;
        let mut numwarn = -1;

        let co = check_assign_object(&mut state, ob);
        if co != 0 {
            return co;
        }
        let obj_index = state.obj;
        let imod = state.imod.as_mut().unwrap();
        if array_size <= imod.obj[obj_index].cont.len() as i32 {
            return FWRAP_ERROR_FILE_TO_BIG;
        }
        if imod_contour_make_z_tables(
            &mut imod.obj[obj_index],
            1,
            0,
            &mut contz,
            &mut zlist,
            &mut numatz,
            &mut contatz,
            &mut zmin,
            &mut zmax,
            &mut zlsize,
            &mut nummax,
        ) != 0
        {
            return FWRAP_ERROR_MEMORY;
        }

        let contsize = imod.obj[obj_index].cont.len();

        /* Allocate lists of mins, max's, and scan contours */
        let mut pmin: Vec<Ipoint> = vec![Ipoint::default(); contsize];
        let mut pmax: Vec<Ipoint> = vec![Ipoint::default(); contsize];
        /* The C keeps `Icont *scancont[]` pointing into the object; the translated
        `imodContourCheckNesting` takes `&mut [Icont]`, so the contours are cloned
        here.  The source never writes a scan contour back into the object either. */
        let mut scancont: Vec<Icont> = imod.obj[obj_index].cont.clone();
        let mut nestind: Vec<i32> = vec![0; contsize];
        let mut nests: Vec<Nesting> = Vec::new();

        /* Get mins, maxes, set addresses into scan contour list */
        for co in 0..contsize {
            *level.add(co) = 0;
            nestind[co] = -1;
            if !imod.obj[obj_index].cont[co].pts.is_empty() {
                imod_contour_get_bbox(
                    Some(&imod.obj[obj_index].cont[co]),
                    &mut pmin[co],
                    &mut pmax[co],
                );
            }
        }

        /* Loop on contours to find inside and outside pairs */
        for indz in 0..(zmax + 1 - zmin).max(0) as usize {
            for kis in 0..(numatz[indz] - 1).max(0) as usize {
                let co = contatz[indz][kis];
                if imod.obj[obj_index].cont[co as usize].pts.is_empty() {
                    continue;
                }
                for lis in (kis + 1)..numatz[indz] as usize {
                    let eco = contatz[indz][lis];
                    if imod.obj[obj_index].cont[eco as usize].pts.is_empty() {
                        continue;
                    }

                    if imod_contour_check_nesting(
                        co,
                        eco,
                        &mut scancont,
                        &pmin,
                        &pmax,
                        &mut nests,
                        &mut nestind,
                        &mut numnests,
                        &mut numwarn,
                    ) != 0
                    {
                        return FWRAP_ERROR_MEMORY;
                    }
                }
            }
        }

        /* Analyze inside and outside contours to determine level */
        imod_contour_nest_levels(&mut nests, &nestind, numnests);

        let mut intot = 0usize;
        let mut outtot = 0usize;
        *in_index.add(0) = 1;
        if in_only == 0 {
            *out_index.add(0) = 1;
        }

        /* Fill arrays with the inside and outside lists and indexes to lists */
        for co in 0..contsize {
            if nestind[co] >= 0 {
                let nest = &nests[nestind[co] as usize];
                if array_size <= 0
                    || intot + nest.inside.len() >= array_size as usize
                    || outtot + nest.outside.len() >= array_size as usize
                {
                    return FWRAP_ERROR_FILE_TO_BIG;
                }
                for &inside in &nest.inside {
                    *in_cont.add(intot) = inside + 1;
                    intot += 1;
                }
                if in_only == 0 {
                    for &outside in &nest.outside {
                        *out_cont.add(outtot) = outside + 1;
                        outtot += 1;
                    }
                }
                *level.add(co) = nest.level;
            }
            *in_index.add(co + 1) = intot as i32 + 1;
            if in_only == 0 {
                *out_index.add(co + 1) = outtot as i32 + 1;
            }
        }

        /* clean up everything */
        imod_contour_free_nests(&mut nests, numnests);
        for co in 0..contsize {
            if !imod.obj[obj_index].cont[co].pts.is_empty()
                && (scancont[co].flags & ICONT_SCANLINE) != 0
            {
                scancont[co] = Icont::default();
            }
        }
        imod_contour_free_z_tables(
            &mut numatz,
            &mut contatz,
            &mut contz,
            &mut zlist,
            zmin,
            zmax,
        );

        FWRAP_NOERROR
    })
}
