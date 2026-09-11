//! Loading an IMOD model file from Fortran code, from
//! `IMOD/libimod/imodel_fwrap.c`.
//!
//! This is the Fortran-facing wrapper around a single file-static model, so it
//! keeps the source's module state as `static mut` items with the same names,
//! and keeps the Fortran calling convention (pointer arguments, `fortStrLen_t`
//! hidden string lengths, arrays of `[f32; 3]` / `[i32; 2]`), as
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

use std::ffi::{c_char, c_void};
use std::fs::File;

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
use crate::imod::libcfshr::b3dutil::{c2f_string, f2c_string};

unsafe extern "C" {
    static stderr: *mut libc::FILE;
}

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
/// `name` stays the `f2cString` allocation the source stores, so that
/// `deleteFimod` frees exactly what the source frees.
#[derive(Clone, Copy, Debug)]
pub struct NameStruct {
    pub ob: i32,
    pub name: *mut c_char,
}

/// Original: `sImod` (`imodel_fwrap.c:232`).
pub static mut S_IMOD: Option<Imod> = None;

/// Original: `sPartialMode` (`imodel_fwrap.c:234`).
pub static mut S_PARTIAL_MODE: i32 = 0;
/// Original: `sMaxObjects` (`imodel_fwrap.c:235`).
pub static mut S_MAX_OBJECTS: i32 = FWRAP_MAX_OBJECT;
/// Original: `sMaxPoints` (`imodel_fwrap.c:236`).
pub static mut S_MAX_POINTS: i32 = FWRAP_MAX_POINTS;
/// Original: `sNumFlagsPut` and `sFlagsPut` (`imodel_fwrap.c:237-238`), the
/// `malloc`ed pair array and its count.
pub static mut S_FLAGS_PUT: Vec<i32> = Vec::new();
/// Original: `sMaxesPut` (`imodel_fwrap.c:239`).
pub static mut S_MAXES_PUT: i32 = 0;
/// Original: `sXmaxPut`, `sYmaxPut`, `sZmaxPut` (`imodel_fwrap.c:240`).
pub static mut S_XMAX_PUT: i32 = 0;
pub static mut S_YMAX_PUT: i32 = 0;
pub static mut S_ZMAX_PUT: i32 = 0;
/// Original: `sZscalePut` (`imodel_fwrap.c:241`).
pub static mut S_ZSCALE_PUT: f32 = NO_VALUE_PUT;
/// Original: `sRotationPut` (`imodel_fwrap.c:242`).
pub static mut S_ROTATION_PUT: Ipoint = Ipoint {
    x: NO_VALUE_PUT,
    y: NO_VALUE_PUT,
    z: NO_VALUE_PUT,
};
/// Original: `sNumSizesPut` and `sSizesPut` (`imodel_fwrap.c:243-244`).
pub static mut S_SIZES_PUT: Vec<SizeStruct> = Vec::new();
/// Original: `sNumValuesPut` and `sValuesPut` (`imodel_fwrap.c:245,247`).
pub static mut S_VALUES_PUT: Vec<ValueStruct> = Vec::new();
/// Original: `sMaxValues` (`imodel_fwrap.c:246`).
pub static mut S_MAX_VALUES: i32 = 0;
/// Original: `sNamesPut` and `sNumNamesPut` (`imodel_fwrap.c:248-249`).
pub static mut S_NAMES_PUT: Vec<NameStruct> = Vec::new();
/// Original: `sLastOpenError` (`imodel_fwrap.c:250`).
pub static mut S_LAST_OPEN_ERROR: i32 = 0;
/// Original: `sWriteAsWimp` (`imodel_fwrap.c:251`).
pub static mut S_WRITE_AS_WIMP: i32 = 0;

/* Convenience pointers assigned by checkAssignObject... */
/// Original: `sObj` (`imodel_fwrap.c:255`), held as an object index.
pub static mut S_OBJ: usize = 0;
/// Original: `sCont` (`imodel_fwrap.c:256`), held as a contour index.
pub static mut S_CONT: usize = 0;

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
pub unsafe fn delete_fimod() {
    if let Some(imod) = (*(&raw mut S_IMOD)).as_mut() {
        imod_delete(imod);
    }
    *(&raw mut S_IMOD) = None;
    *(&raw mut S_FLAGS_PUT) = Vec::new();
    S_MAXES_PUT = 0;
    S_ZSCALE_PUT = NO_VALUE_PUT;
    S_ROTATION_PUT.x = NO_VALUE_PUT;
    S_ROTATION_PUT.y = NO_VALUE_PUT;
    S_ROTATION_PUT.z = NO_VALUE_PUT;
    *(&raw mut S_SIZES_PUT) = Vec::new();
    *(&raw mut S_VALUES_PUT) = Vec::new();
    S_MAX_VALUES = 0;
    for i in 0..(*(&raw const S_NAMES_PUT)).len() {
        let name = (&(*(&raw const S_NAMES_PUT)))[i].name;
        if !name.is_null() {
            libc::free(name.cast::<c_void>());
        }
    }
    *(&raw mut S_NAMES_PUT) = Vec::new();
}

/// Original: `checkAssignObject` (`imodel_fwrap.c:306`).
pub unsafe fn check_assign_object(ob: i32) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if ob < 1 || ob > imod.obj.len() as i32 || imod.obj.is_empty() {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    S_OBJ = ob as usize - 1;
    FWRAP_NOERROR
}

/// Original: `checkAssignObjCont` (`imodel_fwrap.c:316`).
pub unsafe fn check_assign_obj_cont(ob: i32, co: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    if co < 1 || co > imod.obj[S_OBJ].cont.len() as i32 {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    S_CONT = co as usize - 1;
    FWRAP_NOERROR
}

/// Original: `getMeshTrans` (`imodel_fwrap.c:327`).
pub unsafe fn get_mesh_trans(trans: &mut Ipoint) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
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
    S_MAX_OBJECTS = maxob;
    S_MAX_POINTS = maxpt;
}

/// Original: `imodpartialmode` (`imodel_fwrap.c:356`).
pub unsafe fn imodpartialmode(mode: i32) {
    S_PARTIAL_MODE = mode;
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
    if (*(&raw const S_IMOD)).is_none() {
        let _ = FWRAP_ERROR_NO_MODEL;
    }
    *npoint = 0;
    *nobject = 0;
    if S_PARTIAL_MODE != 0 {
        return FWRAP_NOERROR;
    }
    let objsize = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod.obj.len() as i32,
        None => 0,
    };
    getimodobjrange(one, objsize, ibase, npt, coord, color, npoint, nobject)
}

/// Original: `openimoddata` (`imodel_fwrap.c:412`).
pub unsafe fn openimoddata(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    S_LAST_OPEN_ERROR = FWRAP_ERROR_MEMORY;
    let mut model = match imod_new() {
        Some(model) => model,
        None => return FWRAP_ERROR_MEMORY,
    };

    let cfilename = f2c_string(fname, fsize);
    if cfilename.is_null() {
        return FWRAP_ERROR_MEMORY;
    }

    S_LAST_OPEN_ERROR = FWRAP_NOERROR;
    let path = std::ffi::CStr::from_ptr(cfilename)
        .to_string_lossy()
        .into_owned();
    let fin = File::open(&path);
    if fin.is_err() {
        if std::fs::metadata(&path).is_err() {
            S_LAST_OPEN_ERROR = FWRAP_ERROR_BAD_FILENAME;
        } else {
            S_LAST_OPEN_ERROR = FWRAP_ERROR_OPENING_FILE;
        }
    }
    libc::free(cfilename.cast::<c_void>());
    if S_LAST_OPEN_ERROR != 0 {
        return S_LAST_OPEN_ERROR;
    }
    let mut fin = fin.unwrap();

    let err = imod_read_file(&mut model, &mut fin);
    drop(fin);
    if let Err(code) = err {
        S_LAST_OPEN_ERROR = if code == -2 {
            FWRAP_ERROR_FILE_NOT_IMOD
        } else {
            FWRAP_ERROR_READING_FILE
        };
        return S_LAST_OPEN_ERROR;
    }

    if model.obj.is_empty() {
        S_LAST_OPEN_ERROR = FWRAP_ERROR_NO_OBJECTS;
        return FWRAP_ERROR_NO_OBJECTS;
    }

    /* DNM: need to delete model to avoid memory leak */
    delete_fimod();

    *(&raw mut S_IMOD) = Some(model);
    /*
     *  Translate reference image coordinates to
     *  the identity matrix.
     *  DNM 11/5/98: rearranged to correspond to proper conventions
     */
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
}

/// Original: `imodopenerror` (`imodel_fwrap.c:500`).
pub unsafe fn imodopenerror(error: *mut c_char, errlen: FortStrLenT) {
    if S_LAST_OPEN_ERROR > -1 || S_LAST_OPEN_ERROR <= FWRAP_PAST_ERRORS {
        c2f_string(c"".as_ptr(), error, errlen);
    } else {
        let text = S_ERROR_STRINGS[(-1 - S_LAST_OPEN_ERROR) as usize];
        let cstr = std::ffi::CString::new(text).unwrap();
        c2f_string(cstr.as_ptr(), error, errlen);
    }
}

/// Original: `imodcountcontspoints` (`imodel_fwrap.c:513`).
pub unsafe fn imodcountcontspoints(
    num_conts_total: &mut i32,
    max_num_conts: &mut i32,
    num_pts_total: &mut i32,
    max_num_pts: &mut i32,
) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
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
    let mut ncontour = 0;
    let mut npoints = 0;
    let mut coord_index = 0usize;
    let mut ibase_val = 0;
    let mut coi = 0usize;

    let imod = match (*(&raw const S_IMOD)).as_ref() {
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

    if ncontour > S_MAX_OBJECTS {
        libc::fprintf(
            stderr,
            c"getimod: Too many contours in model for Fortran program\n".as_ptr(),
        );
        return FWRAP_ERROR_FILE_TO_BIG;
    }

    if npoints > S_MAX_POINTS {
        libc::fprintf(
            stderr,
            c"getimod: Too many points in model for Fortran program\n".as_ptr(),
        );
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

    if (*(&raw const S_IMOD)).is_none() {
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
    let mut npoints = 0;
    let mut maxobj = 0;
    let mut coi = 0usize;
    let mut coord_index = 0usize;

    let model = match (*(&raw const S_IMOD)).as_ref() {
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

    if maxobj > S_MAX_OBJECTS {
        libc::fprintf(
            stderr,
            c"getimodscat: Too many contours in model for Fortran program\n".as_ptr(),
        );
        return FWRAP_ERROR_FILE_TO_BIG;
    }

    if npoints > S_MAX_POINTS {
        libc::fprintf(
            stderr,
            c"getimodscat: Too many points in model for Fortran program\n".as_ptr(),
        );
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
}

/// Original: `getimodmesh` (`imodel_fwrap.c:736`).
pub unsafe fn getimodmesh(
    objnum: i32,
    mut verts: *mut f32,
    mut index: *mut i32,
    limverts: &mut i32,
    limindex: &mut i32,
) -> i32 {
    let mut resol = 0;
    let mut trans = Ipoint::default();

    if get_mesh_trans(&mut trans) != 0 {
        return FWRAP_ERROR_NO_MODEL;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
        libc::fprintf(
            stderr,
            c"getimodmesh: Too many vertices in mesh for Fortran program\n".as_ptr(),
        );
        return FWRAP_ERROR_FILE_TO_BIG;
    }

    if lsum > *limindex {
        libc::fprintf(
            stderr,
            c"getimodmesh: Too many indices in mesh for Fortran program\n".as_ptr(),
        );
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
    let mut resol = 0;
    let mut norm_add = 0;
    let mut vert_base = 0;
    let mut list_inc = 0;
    let mut trans = Ipoint::default();

    if get_mesh_trans(&mut trans) != 0 {
        return FWRAP_ERROR_NO_MODEL;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
                        libc::printf(
                            c"%d %d %d %d\n".as_ptr(),
                            i as std::ffi::c_int,
                            mi as std::ffi::c_int,
                            j as std::ffi::c_int,
                            limindex as std::ffi::c_int,
                        );
                        libc::fprintf(
                            stderr,
                            c"getimodverts: Too many indices in mesh for Fortran program\n"
                                .as_ptr(),
                        );
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
}

/// Original: `getimodsizes` (`imodel_fwrap.c:895`).
pub unsafe fn getimodsizes(ob: i32, mut sizes: *mut f32, limsizes: i32, nsizes: &mut i32) -> i32 {
    *nsizes = 0;
    let co = check_assign_object(ob);
    if co != 0 {
        return co;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        if *nsizes + cont.pts.len() as i32 > limsizes {
            libc::fprintf(
                stderr,
                c"getimodsizes: Model too large for Fortran program\n".as_ptr(),
            );
            return FWRAP_ERROR_FILE_TO_BIG;
        }
        for pt in 0..cont.pts.len() {
            *sizes = imod_point_get_size(obj, cont, pt as i32);
            sizes = sizes.add(1);
        }
        *nsizes += cont.pts.len() as i32;
    }

    FWRAP_NOERROR
}

/// Original: `getcontpointsizes` (`imodel_fwrap.c:923`).
pub unsafe fn getcontpointsizes(
    ob: i32,
    co: i32,
    mut sizes: *mut f32,
    limsizes: i32,
    nsizes: &mut i32,
) -> i32 {
    *nsizes = 0;
    let pt = check_assign_obj_cont(ob, co);
    if pt != 0 {
        return pt;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let cont = &imod.obj[S_OBJ].cont[S_CONT];
    if cont.sizes.is_empty() {
        return FWRAP_NOERROR;
    }

    if cont.pts.len() as i32 > limsizes {
        libc::fprintf(
            stderr,
            c"getcontpointsizes: Too many points for array\n".as_ptr(),
        );
        return FWRAP_ERROR_FILE_TO_BIG;
    }
    for pt in 0..cont.pts.len() {
        *sizes = cont.sizes[pt];
        sizes = sizes.add(1);
    }
    *nsizes = cont.pts.len() as i32;
    FWRAP_NOERROR
}

/// Original: `putcontpointsizes` (`imodel_fwrap.c:947`).
pub unsafe fn putcontpointsizes(ob: i32, co: i32, sizes: *const f32, nsizes: i32) -> i32 {
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
    (*(&raw mut S_SIZES_PUT)).push(entry);
    FWRAP_NOERROR
}

/// Original: `getimodtimes` (`imodel_fwrap.c:973`).
pub unsafe fn getimodtimes(times: *mut i32) -> i32 {
    let mut coi = 0usize;

    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };

    for ob in 0..imod.obj.len() {
        let obj = &imod.obj[ob];
        for co in 0..obj.cont.len() {
            *times.add(coi) = obj.cont[co].time;
            coi += 1;
        }
    }

    FWRAP_NOERROR
}

/// Original: `getimodobjtimes` (`imodel_fwrap.c:994`).
pub unsafe fn getimodobjtimes(ob: i32, times: *mut i32) -> i32 {
    let co = check_assign_object(ob);
    if co != 0 {
        return co;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    for co in 0..obj.cont.len() {
        *times.add(co) = obj.cont[co].time;
    }
    FWRAP_NOERROR
}

/// Original: `getimodsurfaces` (`imodel_fwrap.c:1008`).
pub unsafe fn getimodsurfaces(surfs: *mut i32) -> i32 {
    let mut coi = 0usize;

    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };

    for ob in 0..imod.obj.len() {
        let obj = &imod.obj[ob];
        for co in 0..obj.cont.len() {
            *surfs.add(coi) = obj.cont[co].surf;
            coi += 1;
        }
    }

    FWRAP_NOERROR
}

/// Original: `getobjsurfaces` (`imodel_fwrap.c:1031`).
///
/// `contSave` is the source's saved copy of the contour array, restored over
/// `sObj->cont` after the sort has written surface numbers into the live one.
pub unsafe fn getobjsurfaces(ob: i32, sort_surfs: i32, surfs: *mut i32) -> i32 {
    let mut retval = FWRAP_NOERROR;

    let co = check_assign_object(ob);
    if co != 0 {
        return co;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    let obj = &mut imod.obj[S_OBJ];
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
}

/// Original: `getcontvalue` (`imodel_fwrap.c:1072`).
pub unsafe fn getcontvalue(ob: i32, co: i32, value: &mut f32) -> i32 {
    getpointvalue(ob, co, 0, value)
}

/// Original: `getpointvalue` (`imodel_fwrap.c:1083`).
pub unsafe fn getpointvalue(ob: i32, co: i32, pt: i32, value: &mut f32) -> i32 {
    let i = check_assign_obj_cont(ob, co);
    if i != 0 {
        return i;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    let cont = &obj.cont[S_CONT];
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
    for i in index..after {
        let item = &store[i];
        if item.type_ == GEN_STORE_VALUE1 {
            *value = item.value.f;
            return FWRAP_NOERROR;
        }
    }
    FWRAP_ERROR_NO_VALUE
}

/// Original: `putcontvalue` (`imodel_fwrap.c:1114`).
pub unsafe fn putcontvalue(ob: i32, co: i32, value: f32) -> i32 {
    putpointvalue(ob, co, 0, value)
}

/// Original: `putpointvalue` (`imodel_fwrap.c:1124`).
pub unsafe fn putpointvalue(ob: i32, co: i32, pt: i32, value: f32) -> i32 {
    let values = &mut *(&raw mut S_VALUES_PUT);
    if values.len() as i32 >= S_MAX_VALUES {
        S_MAX_VALUES += 100;
    }
    values.push(ValueStruct {
        ob: ob - 1,
        co: co - 1,
        pt: pt - 1,
        value,
    });
    FWRAP_NOERROR
}

/// Original: `clearimodobjstore` (`imodel_fwrap.c:1148`).
pub unsafe fn clearimodobjstore(ob: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    imod.obj[S_OBJ].store = Vec::new();
    FWRAP_NOERROR
}

/// Original: `deleteimodmeshes` (`imodel_fwrap.c:1162`).
pub unsafe fn deleteimodmeshes(ob: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    imod.obj[S_OBJ].mesh = Vec::new();
    FWRAP_NOERROR
}

/// Original: `deleteimodcont` (`imodel_fwrap.c:1178`).
pub unsafe fn deleteimodcont(ob: i32, co: i32) -> i32 {
    let err = check_assign_obj_cont(ob, co);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    imod_set_index(imod, ob - 1, co - 1, -1);
    if imod_delete_contour(imod, co - 1) < 0 {
        return FWRAP_ERROR_FROM_CALL;
    }
    FWRAP_NOERROR
}

/// Original: `deletelistofconts` (`imodel_fwrap.c:1194`).
pub unsafe fn deletelistofconts(ob: i32, contours: *mut i32, num_cont: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
}

/// Original: `deleteimodpoint` (`imodel_fwrap.c:1214`).
pub unsafe fn deleteimodpoint(ob: i32, co: i32, pt: i32) -> i32 {
    let err = check_assign_obj_cont(ob, co);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    let obj_index = S_OBJ;
    let cont_index = S_CONT;
    if pt < 1 || pt > imod.obj[obj_index].cont[cont_index].pts.len() as i32 {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    if crate::imod::libimod::ipoint::imod_point_delete(
        &mut imod.obj[obj_index].cont[cont_index],
        pt - 1,
    ) < 0
    {
        return FWRAP_ERROR_FROM_CALL;
    }
    FWRAP_NOERROR
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
    /* B3DNINT (`b3dutil.h:33`) */
    let mut pt = (if pt_or_z as f64 >= 0. {
        (pt_or_z as f64 + 0.5).floor()
    } else {
        (pt_or_z as f64 - 0.5).ceil()
    }) as i32
        - 1;
    let err = check_assign_obj_cont(ob, co);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    let obj_index = S_OBJ;
    let cont_index = S_CONT;
    if pt < 0 && if_by_z == 0 {
        return FWRAP_ERROR_BAD_OBJNUM;
    }
    let psize = imod.obj[obj_index].cont[cont_index].pts.len() as i32;
    if if_by_z != 0 {
        pt = 0;
        while pt < psize {
            if imod.obj[obj_index].cont[cont_index].pts[pt as usize].z >= pt_or_z {
                break;
            }
            pt += 1;
        }
    }
    pt = if pt < psize { pt } else { psize };
    let point = Ipoint { x, y, z };
    if imod_point_add(&mut imod.obj[obj_index].cont[cont_index], Some(point), pt) == 0 {
        return FWRAP_ERROR_FROM_CALL;
    }
    FWRAP_NOERROR
}

/// Original: `getobjvaluethresh` (`imodel_fwrap.c:1265`).
pub unsafe fn getobjvaluethresh(ob: i32, thresh: &mut f32) -> i32 {
    let mut vmin = 0f32;
    let mut vmax = 0f32;
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
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
}

/// Original: `getobjskiplowvalues` (`imodel_fwrap.c:1281`).
pub unsafe fn getobjskiplowvalues(ob: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    if (obj.flags & IMOD_OBJFLAG_USE_VALUE) != 0 && (obj.matflags2 as u32 & MATFLAGS2_SKIP_LOW) != 0
    {
        1
    } else {
        0
    }
}

/// Original: `findaddminmax1value` (`imodel_fwrap.c:1294`).
pub unsafe fn findaddminmax1value(ob: i32) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    let err = istore_find_add_min_max1(&mut imod.obj[S_OBJ]);
    if err > 0 {
        return FWRAP_ERROR_FROM_CALL;
    }
    FWRAP_NOERROR
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
    let mut nobj = 0usize;

    if (*(&raw const S_IMOD)).is_none() {
        *(&raw mut S_IMOD) = imod_new();
        (*(&raw mut S_IMOD)).as_mut().unwrap().obj = Vec::new();
    }

    let mut mincolor = 256 - (*(&raw const S_IMOD)).as_ref().unwrap().obj.len() as i32;

    /* Find minimum color, and maximum object #; get arrays */
    /* Skip empty contours unless we are in partial mode, where they are needed
    to signal that an existing object is now empty */
    for object in 0..nobject as usize {
        if *npt.add(object) == 0 && S_PARTIAL_MODE == 0 {
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
    let objsize = (*(&raw const S_IMOD)).as_ref().unwrap().obj.len();
    for ob in 0..objsize {
        objlookup[((255 - mincolor) - ob as i32) as usize] = ob as i32;
        nobj += 1;
        if S_PARTIAL_MODE != 0 {
            nsaved[ob] = (*(&raw const S_IMOD)).as_ref().unwrap().obj[ob].cont.len() as i32;
        }
    }

    /* For all non-empty contours passed back, mark an empty object as having
    data. */
    for object in 0..nobject as usize {
        if *npt.add(object) == 0 && S_PARTIAL_MODE == 0 {
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
        let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
        for pt in 0..(*(&raw const S_NAMES_PUT)).len() {
            if (&(*(&raw const S_NAMES_PUT)))[pt].ob == 255 - (ob as i32 + mincolor) {
                let src = (&(*(&raw const S_NAMES_PUT)))[pt].name;
                if !src.is_null() {
                    libc::strncpy(imod.obj[nobj].name.as_mut_ptr(), src, IOBJ_STRSIZE - 1);
                }
                imod.obj[nobj].name[IOBJ_STRSIZE - 1] = 0;
                ci = 1;
            }
        }
        if ci == 0 {
            libc::sprintf(
                imod.obj[nobj].name.as_mut_ptr(),
                c"Fmod # %d".as_ptr(),
                wimpno as std::ffi::c_int,
            );
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
                libc::fprintf(
                    stderr,
                    c"putimod warning: bad object bounds %d\n".as_ptr(),
                    ob as std::ffi::c_int,
                );
            }
            continue;
        }
        let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
        if ob > imod.obj.len() as i32 {
            libc::fprintf(
                stderr,
                c"putimod warning: bad object bounds %d\n".as_ptr(),
                ob as std::ffi::c_int,
            );
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
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    for ob in 0..imod.obj.len() {
        if imod_contours_delete_to_end(&mut imod.obj[ob], nsaved[ob]) != 0 {
            return FWRAP_ERROR_MEMORY;
        }
    }
    FWRAP_NOERROR
}

/// Original: `writeimod` (`imodel_fwrap.c:1546`).
pub unsafe fn writeimod(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    let filename = f2c_string(fname, fsize);
    if filename.is_null() {
        return FWRAP_ERROR_BAD_FILENAME;
    }
    if (*(&raw const S_IMOD)).is_none() {
        libc::free(filename.cast::<c_void>());
        return FWRAP_ERROR_NO_MODEL;
    }
    let path = std::ffi::CStr::from_ptr(filename)
        .to_string_lossy()
        .into_owned();

    /*
     *  Translate coordinates to match reference image.
     */
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
    let num_flags = (*(&raw const S_FLAGS_PUT)).len() / 2;
    for i in 0..num_flags {
        let ob = (&(*(&raw const S_FLAGS_PUT)))[2 * i];
        let mut flag = (&(*(&raw const S_FLAGS_PUT)))[2 * i + 1];
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
                        let (r, g, b) = (imod.obj[ob].red, imod.obj[ob].green, imod.obj[ob].blue);
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

    if S_MAXES_PUT != 0 {
        imod.xmax = S_XMAX_PUT;
        imod.ymax = S_YMAX_PUT;
        imod.zmax = S_ZMAX_PUT;
    }

    /* Save Zscale, save rotation in current view if there is one */
    if S_ZSCALE_PUT != NO_VALUE_PUT {
        imod.zscale = S_ZSCALE_PUT;
    }
    if imod.cview > 0 && S_ROTATION_PUT.x != NO_VALUE_PUT {
        let cview = imod.cview as usize;
        imod.view[cview].rot = S_ROTATION_PUT;
    }

    /* Put out the sizes into the contours */
    for i in 0..(*(&raw const S_SIZES_PUT)).len() {
        /* Ignore illegal object or contour numbers */
        let put_ob = (&(*(&raw const S_SIZES_PUT)))[i].ob;
        if put_ob < 0 || put_ob >= imod.obj.len() as i32 {
            continue;
        }
        let put_co = (&(*(&raw const S_SIZES_PUT)))[i].co;
        if put_co < 0 || put_co >= imod.obj[put_ob as usize].cont.len() as i32 {
            continue;
        }

        /* Fill up to the limit on the number of point in array and contour */
        let cont = &mut imod.obj[put_ob as usize].cont[put_co as usize];
        let num = (&(*(&raw const S_SIZES_PUT)))[i].num;
        let value = if num < cont.pts.len() as i32 {
            num
        } else {
            cont.pts.len() as i32
        };
        for j in 0..value as usize {
            let size = (&(*(&raw const S_SIZES_PUT)))[i].sizes[j];
            if size >= 0. {
                imod_point_set_size(cont, j as i32, size);
            }
        }
    }

    /* Put out general values */
    if !(*(&raw const S_VALUES_PUT)).is_empty() {
        let mut store = Istore {
            type_: GEN_STORE_VALUE1,
            flags: (GEN_STORE_FLOAT << 2) | GEN_STORE_ONEPOINT,
            index: StoreUnion { i: 0 },
            value: StoreUnion { i: 0 },
        };
        for i in 0..(*(&raw const S_VALUES_PUT)).len() {
            let entry = (&(*(&raw const S_VALUES_PUT)))[i];

            /* Ignore illegal object or contour numbers */
            if entry.ob < 0 || entry.ob >= imod.obj.len() as i32 {
                continue;
            }
            if entry.co < 0 || entry.co >= imod.obj[entry.ob as usize].cont.len() as i32 {
                continue;
            }

            /* Insert value into contour or object stores */
            store.value = StoreUnion { f: entry.value };
            if entry.pt < 0 {
                store.index = StoreUnion { i: entry.co };
                istore_insert_change(&mut imod.obj[entry.ob as usize].store, store);
            } else {
                let cont = &mut imod.obj[entry.ob as usize].cont[entry.co as usize];
                if entry.pt > cont.pts.len() as i32 {
                    continue;
                }
                store.index = StoreUnion { i: entry.pt };
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
    if S_WRITE_AS_WIMP != 0 {
        let mut out = match File::create(&path) {
            Ok(out) => out,
            Err(_) => {
                libc::free(filename.cast::<c_void>());
                return FWRAP_ERROR_OPENING_FILE;
            }
        };
        retcode = match imod_to_wmod(imod, &mut out, &path) {
            Ok(()) => 0,
            Err(code) => code,
        };
        S_WRITE_AS_WIMP = 0;
    } else {
        let mut out = match File::create(&path) {
            Ok(out) => out,
            Err(_) => {
                libc::free(filename.cast::<c_void>());
                return FWRAP_ERROR_OPENING_FILE;
            }
        };
        retcode = match imod_write_file(imod, &mut out) {
            Ok(()) => 0,
            Err(code) => code,
        };
    }
    libc::free(filename.cast::<c_void>());
    retcode
}

/// Original: `imodwriteaswimp` (`imodel_fwrap.c:1774`).
pub unsafe fn imodwriteaswimp(fname: *const c_char, fsize: FortStrLenT) -> i32 {
    S_WRITE_AS_WIMP = 1;
    writeimod(fname, fsize)
}

/// Original: `putimodscat` (`imodel_fwrap.c:1784`).
pub unsafe fn putimodscat(ob: i32, verts: *const f32) -> i32 {
    let co = check_assign_object(ob);
    if co != 0 {
        return co;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    if iobj_scat(imod.obj[S_OBJ].flags) == 0 {
        return 2;
    }

    let mut v = 0usize;
    let obj_index = S_OBJ;
    for co in 0..imod.obj[obj_index].cont.len() {
        let psize = imod.obj[obj_index].cont[co].pts.len();
        for pt in 0..psize {
            imod.obj[obj_index].cont[co].pts[pt].x = *verts.add(v);
            v += 1;
            imod.obj[obj_index].cont[co].pts[pt].y = *verts.add(v);
            v += 1;
            imod.obj[obj_index].cont[co].pts[pt].z = *verts.add(v);
            v += 1;
        }
    }
    0
}

/// Original: `putimodmesh` (`imodel_fwrap.c:1808`).
pub unsafe fn putimodmesh(mut verts: *const f32) -> i32 {
    let mut trans = Ipoint::default();

    if get_mesh_trans(&mut trans) != 0 {
        return FWRAP_ERROR_NO_MODEL;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
}

/// Original: `newimod` (`imodel_fwrap.c:1836`).
pub unsafe fn newimod() -> i32 {
    delete_fimod();
    *(&raw mut S_IMOD) = imod_new();
    if (*(&raw const S_IMOD)).is_none() {
        return FWRAP_ERROR_NO_MODEL;
    }
    (*(&raw mut S_IMOD)).as_mut().unwrap().obj = Vec::new();
    FWRAP_NOERROR
}

/// Original: `deleteimod` (`imodel_fwrap.c:1850`).
pub unsafe fn deleteimod() -> i32 {
    if (*(&raw const S_IMOD)).is_none() {
        return FWRAP_ERROR_NO_MODEL;
    }
    delete_fimod();
    FWRAP_NOERROR
}

/// Original: `deleteiobj` (`imodel_fwrap.c:1861`).
pub unsafe fn deleteiobj() -> i32 {
    if (*(&raw const S_IMOD)).is_none() {
        return FWRAP_ERROR_NO_MODEL;
    }

    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    imod_objviews_free(imod);
    imod_objects_delete(&mut imod.obj);
    imod.obj = Vec::new();

    FWRAP_NOERROR
}

/// Original: `getimodobjsize` (`imodel_fwrap.c:1878`).
pub unsafe fn getimodobjsize() -> i32 {
    match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod.obj.len() as i32,
        None => -1,
    }
}

/// Original: `getimodheado` (`imodel_fwrap.c:1889`).
pub unsafe fn getimodheado(um: &mut f32, zscale: &mut f32) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
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
}

/// Original: `getimodmaxes` (`imodel_fwrap.c:1908`).
pub unsafe fn getimodmaxes(xmax: &mut i32, ymax: &mut i32, zmax: &mut i32) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };

    *xmax = imod.xmax;
    *ymax = imod.ymax;
    *zmax = imod.zmax;
    FWRAP_NOERROR
}

/// Original: `getzfromminuspt5` (`imodel_fwrap.c:1923`).
pub unsafe fn getzfromminuspt5() -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if imod.flags & IMODF_Z_FROM_MINUSPT5 != 0 {
        1
    } else {
        0
    }
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
    let imod = match (*(&raw const S_IMOD)).as_ref() {
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
}

/// Original: `getimodscales` (`imodel_fwrap.c:1981`).
pub unsafe fn getimodscales(ximscale: &mut f32, yimscale: &mut f32, zimscale: &mut f32) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
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
}

/// Original: `putimageref` (`imodel_fwrap.c:2006`).
pub unsafe fn putimageref(delta: *const f32, origin: *const f32, tilt: *const f32) -> i32 {
    if (*(&raw const S_IMOD)).is_none() {
        return FWRAP_ERROR_NO_MODEL;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
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
}

/// Original: `imodhasimageref` (`imodel_fwrap.c:2036`).
pub unsafe fn imodhasimageref() -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if imod.ref_image.is_some() { 1 } else { 0 }
}

/// Original: `getimodflags` (`imodel_fwrap.c:2048`).
pub unsafe fn getimodflags(flags: *mut i32, limflags: i32) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if imod.obj.len() as i32 > limflags {
        libc::fprintf(
            stderr,
            c"getimodflags: Too many objects for Fortran program\n".as_ptr(),
        );
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
}

/// Original: `getmodelname` (`imodel_fwrap.c:2077`).
pub unsafe fn getmodelname(fname: *mut c_char, fsize: FortStrLenT) -> i32 {
    let imod = match (*(&raw const S_IMOD)).as_ref() {
        Some(imod) => imod,
        None => return FWRAP_ERROR_NO_MODEL,
    };
    if c2f_string(imod.name.as_ptr(), fname, fsize) != 0 {
        return FWRAP_ERROR_STRING_LEN;
    }
    FWRAP_NOERROR
}

/// Original: `getimodobjname` (`imodel_fwrap.c:2090`).
pub unsafe fn getimodobjname(ob: i32, fname: *mut c_char, fsize: FortStrLenT) -> i32 {
    let err = check_assign_object(ob);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    let mut i = 0i32;
    while i < fsize && (i as usize) < IOBJ_STRSIZE && obj.name[i as usize] != 0 {
        *fname.add(i as usize) = obj.name[i as usize];
        i += 1;
    }
    while i < fsize {
        *fname.add(i as usize) = 0x20;
        i += 1;
    }
    FWRAP_NOERROR
}

/// Original: `getimodclip` (`imodel_fwrap.c:2108`).
pub unsafe fn getimodclip(objnum: i32, clip: *mut f32) -> i32 {
    let mut iout = 0usize;
    let err = check_assign_object(objnum);
    if err != 0 {
        return err;
    }

    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    imod.cindex.object = objnum - 1;

    /* DNM 9/19/04: modified for multiple clip planes */
    let obj = &imod.obj[S_OBJ];
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
}

/* DNM 6/8/01: change this to save the values until a model is written */
/// Original: `putimodmaxes` (`imodel_fwrap.c:2137`).
pub unsafe fn putimodmaxes(xmax: i32, ymax: i32, zmax: i32) -> i32 {
    S_XMAX_PUT = xmax;
    S_YMAX_PUT = ymax;
    S_ZMAX_PUT = zmax;
    S_MAXES_PUT = 1;
    FWRAP_NOERROR
}

/// Original: `putimodflag` (`imodel_fwrap.c:2153`).
pub unsafe fn putimodflag(objnum: i32, flag: i32) {
    let flags = &mut *(&raw mut S_FLAGS_PUT);
    flags.push(objnum - 1);
    flags.push(flag);
}

/// Original: `putimodzscale` (`imodel_fwrap.c:2174`).
pub unsafe fn putimodzscale(zscale: f32) {
    S_ZSCALE_PUT = zscale;
}

/// Original: `putimodrotation` (`imodel_fwrap.c:2182`).
pub unsafe fn putimodrotation(xrot: f32, yrot: f32, zrot: f32) {
    S_ROTATION_PUT.x = xrot;
    S_ROTATION_PUT.y = yrot;
    S_ROTATION_PUT.z = zrot;
}

/* DNM 12/3/01: added so that old wimp models can be converted */
/// Original: `fromvmsfloats` (`imodel_fwrap.c:2190`).
pub fn fromvmsfloats(data: &mut [u8], amt: i32) {
    imod_from_vms_floats(data, amt);
}

/// Original: `getscatsize` (`imodel_fwrap.c:2198`).
pub unsafe fn getscatsize(objnum: i32, size: &mut i32) -> i32 {
    let err = check_assign_object(objnum);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    *size = imod.obj[S_OBJ].pdrawsize;
    FWRAP_NOERROR
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
    let err = check_assign_object(objnum);
    if err != 0 {
        return err;
    }
    let imod = (*(&raw const S_IMOD)).as_ref().unwrap();
    let obj = &imod.obj[S_OBJ];
    /* B3DNINT (`b3dutil.h:33`) */
    *red = b3d_nint_local(255. * obj.red as f64);
    *green = b3d_nint_local(255. * obj.green as f64);
    *blue = b3d_nint_local(255. * obj.blue as f64);
    FWRAP_NOERROR
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
    if (*(&raw const S_IMOD)).is_none() {
        return FWRAP_ERROR_NO_MODEL;
    }
    let tmpstr = f2c_string(fname, fsize);
    if tmpstr.is_null() {
        return FWRAP_ERROR_MEMORY;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    libc::strncpy(imod.name.as_mut_ptr(), tmpstr, IMOD_STRSIZE - 1);
    imod.name[IMOD_STRSIZE - 1] = 0;
    libc::free(tmpstr.cast::<c_void>());
    FWRAP_NOERROR
}

/// Original: `putimodobjname` (`imodel_fwrap.c:2316`).
pub unsafe fn putimodobjname(objnum: i32, fname: *const c_char, fsize: FortStrLenT) {
    (*(&raw mut S_NAMES_PUT)).push(NameStruct {
        ob: objnum - 1,
        name: f2c_string(fname, fsize),
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

    let co = check_assign_object(ob);
    if co != 0 {
        return co;
    }
    let imod = (*(&raw mut S_IMOD)).as_mut().unwrap();
    let obj_index = S_OBJ;
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
            if intot as i32 + nest.ninside >= array_size
                || outtot as i32 + nest.noutside >= array_size
            {
                return FWRAP_ERROR_FILE_TO_BIG;
            }
            for lis in 0..nest.ninside as usize {
                *in_cont.add(intot) = nest.inside[lis] + 1;
                intot += 1;
            }
            if in_only == 0 {
                for lis in 0..nest.noutside as usize {
                    *out_cont.add(outtot) = nest.outside[lis] + 1;
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
}
