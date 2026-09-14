//! Translation of `IMOD/libimod/iobj.c` and its paired header
//! `IMOD/include/iobj.h`.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::set_or_clear_flags;

use super::icont::{
    ICONT_TEMPUSE, imod_contour_area, imod_contour_get_bbox, imodel_contour_centroid,
};
use super::imodel::{
    IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_ENDPOLY, IMOD_OBJFLAG_OFF,
    IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, IOBJ_STRSIZE, Iclip_planes, Icont, Imesh, Iobj, Ipoint,
};
use super::istore::{istore_delete_cont_surf, istore_shift_index, istore_sort};

/* object bit flags (`iobj.h:21`).  `IMOD_OBJFLAG_OPEN`, `_OUT`, `_SCAT` and
`_OFF` are declared with the rest of the model data in `imodel.rs`. */

/// Original: `IMOD_OBJFLAG_WILD` (`iobj.h:22`).
pub const IMOD_OBJFLAG_WILD: u32 = 1 << 4;
/// Original: `IMOD_OBJFLAG_FILL` (`iobj.h:27`).
pub const IMOD_OBJFLAG_FILL: u32 = 1 << 8;
/// Original: `IMOD_OBJFLAG_MESH` (`iobj.h:29`).
pub const IMOD_OBJFLAG_MESH: u32 = 1 << 10;
/// Original: `IMOD_OBJFLAG_NOLINE` (`iobj.h:30`).
pub const IMOD_OBJFLAG_NOLINE: u32 = 1 << 11;
/// Original: `IMOD_OBJFLAG_DCUE` (`iobj.h:31`).
pub const IMOD_OBJFLAG_DCUE: u32 = 1 << 2;
/// Original: `IMOD_OBJFLAG_FCOLOR` (`iobj.h:32`).
pub const IMOD_OBJFLAG_FCOLOR: u32 = 1 << 14;
/// Original: `IMOD_OBJFLAG_FCOLOR_PNT` (`iobj.h:33`).
pub const IMOD_OBJFLAG_FCOLOR_PNT: u32 = 1 << 6;
/// Original: `IMOD_OBJFLAG_PNT_ON_SEC` (`iobj.h:34`).
pub const IMOD_OBJFLAG_PNT_ON_SEC: u32 = 1 << 7;
/// Original: `IMOD_OBJFLAG_PLANAR` (`iobj.h:35`).
pub const IMOD_OBJFLAG_PLANAR: u32 = 1 << 13;
/// Original: `IMOD_OBJFLAG_ANTI_ALIAS` (`iobj.h:36`).
pub const IMOD_OBJFLAG_ANTI_ALIAS: u32 = 1 << 15;
/// Original: `IMOD_OBJFLAG_USE_VALUE` (`iobj.h:37`).
pub const IMOD_OBJFLAG_USE_VALUE: u32 = 1 << 12;
/// Original: `IMOD_OBJFLAG_SCALAR` (`iobj.h:38`).
pub const IMOD_OBJFLAG_SCALAR: u32 = 1 << 16;
/// Original: `IMOD_OBJFLAG_MCOLOR` (`iobj.h:39`).
pub const IMOD_OBJFLAG_MCOLOR: u32 = 1 << 17;
/// Original: `IMOD_OBJFLAG_TIME` (`iobj.h:40`).
pub const IMOD_OBJFLAG_TIME: u32 = 1 << 18;
/// Original: `IMOD_OBJFLAG_TWO_SIDE` (`iobj.h:41`).
pub const IMOD_OBJFLAG_TWO_SIDE: u32 = 1 << 19;
/// Original: `IMOD_OBJFLAG_THICK_CONT` (`iobj.h:42`).
pub const IMOD_OBJFLAG_THICK_CONT: u32 = 1 << 20;
/// Original: `IMOD_OBJFLAG_EXTRA_MODV` (`iobj.h:43`).
pub const IMOD_OBJFLAG_EXTRA_MODV: u32 = 1 << 21;
/// Original: `IMOD_OBJFLAG_EXTRA_EDIT` (`iobj.h:44`).
pub const IMOD_OBJFLAG_EXTRA_EDIT: u32 = 1 << 22;
/// Original: `IMOD_OBJFLAG_PNT_NOMODV` (`iobj.h:45`).
pub const IMOD_OBJFLAG_PNT_NOMODV: u32 = 1 << 23;
/// Original: `IMOD_OBJFLAG_MODV_ONLY` (`iobj.h:46`).
pub const IMOD_OBJFLAG_MODV_ONLY: u32 = 1 << 24;
/// Original: `IMOD_OBJFLAG_POLY_CONT` (`iobj.h:47`).
pub const IMOD_OBJFLAG_POLY_CONT: u32 = 1 << 26;
/// Original: `IMOD_OBJFLAG_DRAW_LABEL` (`iobj.h:48`).
pub const IMOD_OBJFLAG_DRAW_LABEL: u32 = 1 << 27;
/// Original: `IMOD_OBJFLAG_SCALE_WDTH` (`iobj.h:49`).
pub const IMOD_OBJFLAG_SCALE_WDTH: u32 = 1 << 28;
/// Original: `IMOD_OBJFLAG_TEMPUSE` (`iobj.h:50`).
pub const IMOD_OBJFLAG_TEMPUSE: u32 = 1 << 31;
/// Original: `IMOD_OBJFLAG_LINE` (`iobj.h:52`).
pub const IMOD_OBJFLAG_LINE: u32 = IMOD_OBJFLAG_NOLINE;

/// Original: `IOBJ_EXSIZE` (`imodel.h:33`), the length of `Iobj.extra`.
pub const IOBJ_EXSIZE: usize = 16;

/// Original: `iobjConnect` (`iobj.h:55`).
pub fn iobj_connect(flag: u32) -> u32 {
    (!flag) & IMOD_OBJFLAG_SCAT
}

/// Original: `iobjClose` (`iobj.h:57`).
pub fn iobj_close(flag: u32) -> i32 {
    (((!flag) & IMOD_OBJFLAG_OPEN != 0) && iobj_connect(flag) != 0) as i32
}

/// Original: `iobjOpen` (`iobj.h:58`).
pub fn iobj_open(flag: u32) -> i32 {
    ((iobj_connect(flag) != 0) && (flag & IMOD_OBJFLAG_OPEN != 0)) as i32
}

/// Original: `iobjFill` (`iobj.h:59`).
pub fn iobj_fill(flag: u32) -> u32 {
    flag & IMOD_OBJFLAG_FILL
}

/// Original: `iobjOff` (`iobj.h:60`).
pub fn iobj_off(flag: u32) -> u32 {
    flag & IMOD_OBJFLAG_OFF
}

/// Original: `iobjMesh` (`iobj.h:61`).
pub fn iobj_mesh(flag: u32) -> u32 {
    flag & IMOD_OBJFLAG_MESH
}

/// Original: `iobjScat` (`iobj.h:62`).
pub fn iobj_scat(flag: u32) -> u32 {
    flag & IMOD_OBJFLAG_SCAT
}

/// Original: `iobjLine` (`iobj.h:63`).
pub fn iobj_line(flag: u32) -> u32 {
    (!flag) & IMOD_OBJFLAG_NOLINE
}

/// Original: `iobjTime` (`iobj.h:64`).
pub fn iobj_time(flag: u32) -> u32 {
    flag & IMOD_OBJFLAG_TIME
}

/// Original: `iobjDraw` (`iobj.h:65`).
pub fn iobj_draw(flag: u32) -> u32 {
    (!flag) & IMOD_OBJFLAG_OFF
}

/// Original: `iobjFlagTime` (`iobj.h:66`).
pub fn iobj_flag_time(o: &Iobj) -> u32 {
    iobj_time(o.flags)
}

/// Original: `iobjPlanar` (`iobj.h:67`).
pub fn iobj_planar(flag: u32) -> i32 {
    (iobj_close(flag) != 0 || (iobj_open(flag) != 0 && (flag & IMOD_OBJFLAG_PLANAR) != 0)) as i32
}

/// Original: `MATFLAGS2_SKIP_LOW` (`iobj.h:70`).
pub const MATFLAGS2_SKIP_LOW: u32 = 1;
/// Original: `MATFLAGS2_SKIP_HIGH` (`iobj.h:71`).
pub const MATFLAGS2_SKIP_HIGH: u32 = 1 << 1;
/// Original: `MATFLAGS2_CONSTANT` (`iobj.h:72`).
pub const MATFLAGS2_CONSTANT: u32 = 1 << 2;

/// Original: `IOBJ_SYM_CIRCLE` (`iobj.h:76`).
pub const IOBJ_SYM_CIRCLE: i32 = 0;
/// Original: `IOBJ_SYM_NONE` (`iobj.h:77`).
pub const IOBJ_SYM_NONE: i32 = 1;
/// Original: `IOBJ_SYM_SQUARE` (`iobj.h:78`).
pub const IOBJ_SYM_SQUARE: i32 = 2;
/// Original: `IOBJ_SYM_TRIANGLE` (`iobj.h:79`).
pub const IOBJ_SYM_TRIANGLE: i32 = 3;
/// Original: `IOBJ_SYM_STAR` (`iobj.h:80`).
pub const IOBJ_SYM_STAR: i32 = 4;
/// Original: `IOBJ_SYM_LAST` (`iobj.h:81`).
pub const IOBJ_SYM_LAST: i32 = 5;

/// Original: `IOBJ_SYMF_FILL` (`iobj.h:83`).
pub const IOBJ_SYMF_FILL: u32 = 1;
/// Original: `IOBJ_SYMF_ENDS` (`iobj.h:84`).
pub const IOBJ_SYMF_ENDS: u32 = 1 << 1;
/// Original: `IOBJ_SYMF_ARROW` (`iobj.h:85`).
pub const IOBJ_SYMF_ARROW: u32 = 1 << 2;

/// Original: `IMOD_OBJM_OFF` (`iobj.h:88`).
pub const IMOD_OBJM_OFF: i32 = 0;
/// Original: `IMOD_OBJM_CIRCLE` (`iobj.h:89`).
pub const IMOD_OBJM_CIRCLE: i32 = 1;
/// Original: `IMOD_OBJM_SQUARE` (`iobj.h:90`).
pub const IMOD_OBJM_SQUARE: i32 = 2;
/// Original: `IMOD_OBJM_TRIANGLE` (`iobj.h:91`).
pub const IMOD_OBJM_TRIANGLE: i32 = 3;

/// Original: `IobjMaxContour` (`iobj.h:96`).
pub const IOBJ_MAX_CONTOUR: i32 = 33;
/// Original: `IobjLineWidth` (`iobj.h:97`).
pub const IOBJ_LINE_WIDTH: i32 = 34;
/// Original: `IobjPointSize` (`iobj.h:98`).
pub const IOBJ_POINT_SIZE: i32 = 35;
/// Original: `IobjMaxMesh` (`iobj.h:99`).
pub const IOBJ_MAX_MESH: i32 = 36;
/// Original: `IobjMaxSurface` (`iobj.h:100`).
pub const IOBJ_MAX_SURFACE: i32 = 37;
/// Original: `IobjLineWidth2` (`iobj.h:101`).
pub const IOBJ_LINE_WIDTH2: i32 = 38;
/// Original: `IobjSymType` (`iobj.h:102`).
pub const IOBJ_SYM_TYPE: i32 = 39;
/// Original: `IobjSymSize` (`iobj.h:103`).
pub const IOBJ_SYM_SIZE: i32 = 40;
/// Original: `IobjSymFlags` (`iobj.h:104`).
pub const IOBJ_SYM_FLAGS: i32 = 41;

/// Original: `IobjFlagClosed` (`iobj.h:106`).
pub const IOBJ_FLAG_CLOSED: i32 = 3;
/// Original: `IobjFlagConnected` (`iobj.h:107`).
pub const IOBJ_FLAG_CONNECTED: i32 = 9;
/// Original: `IobjFlagFilled` (`iobj.h:108`).
pub const IOBJ_FLAG_FILLED: i32 = 8;
/// Original: `IobjFlagDraw` (`iobj.h:109`).
pub const IOBJ_FLAG_DRAW: i32 = 1;
/// Original: `IobjFlagPntOnSec` (`iobj.h:110`).
pub const IOBJ_FLAG_PNT_ON_SEC: i32 = 7;
/// Original: `IobjFlagMesh` (`iobj.h:111`).
pub const IOBJ_FLAG_MESH: i32 = 10;
/// Original: `IobjFlagLine` (`iobj.h:112`).
pub const IOBJ_FLAG_LINE: i32 = 11;
/// Original: `IobjFlagTime` (`iobj.h:113`).
pub const IOBJ_FLAG_TIME: i32 = 12;
/// Original: `IobjFlagExtraInModv` (`iobj.h:114`).
pub const IOBJ_FLAG_EXTRA_IN_MODV: i32 = 21;
/// Original: `IobjFlagExtraInSlicer` (`iobj.h:115`).
pub const IOBJ_FLAG_EXTRA_IN_SLICER: i32 = 22;
/// Original: `IobjFlagPlanar` (`iobj.h:116`).
pub const IOBJ_FLAG_PLANAR: i32 = 23;

/// Original: `IOBJ_EX_PNT_LIMIT` (`iobj.h:120`).
pub const IOBJ_EX_PNT_LIMIT: usize = 0;
/// Original: `IOBJ_EX_2D_TRANS` (`iobj.h:121`).
pub const IOBJ_EX_2D_TRANS: usize = 1;
/// Original: `IOBJ_EX_LABEL_SIZE` (`iobj.h:122`).
pub const IOBJ_EX_LABEL_SIZE: usize = 2;
/// Original: `IOBJ_EX_FLAGS` (`iobj.h:123`).
pub const IOBJ_EX_FLAGS: usize = 3;
/// Original: `IOBJ_EX_LASSO_ID` (`iobj.h:124`).
pub const IOBJ_EX_LASSO_ID: usize = IOBJ_EXSIZE - 1;

/// Original: `IOBJ_EXFLAG_ISO_PAINT` (`iobj.h:127`).
pub const IOBJ_EXFLAG_ISO_PAINT: u32 = 1;
/// Original: `IOBJ_EXFLAG_SLICER_ONLY` (`iobj.h:128`).
pub const IOBJ_EXFLAG_SLICER_ONLY: u32 = 1 << 1;
/// Original: `IOBJ_EXFLAG_MESH_ON_IMG` (`iobj.h:129`).
pub const IOBJ_EXFLAG_MESH_ON_IMG: u32 = 1 << 2;

/// Original: `imodObjectNew` (`iobj.c:24`).
pub fn imod_object_new() -> Option<Iobj> {
    imod_objects_new(1).and_then(|mut objects| objects.pop())
}

/// Original: `imodObjectsNew` (`iobj.c:37`).
pub fn imod_objects_new(size: i32) -> Option<Vec<Iobj>> {
    if size < 0 {
        return None;
    }
    let mut objects = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let mut object = Iobj::default();
        imod_object_default(&mut object);
        objects.push(object);
    }
    Some(objects)
}

/// Original: `imodObjectDefault` (`iobj.c:55`).
pub fn imod_object_default(object: &mut Iobj) {
    object.cont.clear();
    object.mesh.clear();
    object.store.clear();
    object.label = None;
    object.name = [0; IOBJ_STRSIZE];
    object.extra = [0; 16];
    object.flags = (1 << 27) | (1 << 28);
    object.axis = 0;
    object.drawmode = 1;
    object.red = 0.5;
    object.green = 0.5;
    object.blue = 0.5;
    object.pdrawsize = 0;
    object.symbol = 1;
    object.symsize = 3;
    object.linewidth2 = 1;
    object.linewidth = 1;
    object.linesty = 0;
    object.symflags = 0;
    object.sympad = 0;
    object.trans = 0;
    object.surfsize = 0;
    crate::imod::libimod::iplane::imod_clips_initialize(&mut object.clips);
    object.ambient = 102;
    object.diffuse = 255;
    object.specular = 127;
    object.shininess = 4;
    object.fillred = 0;
    object.fillgreen = 0;
    object.fillblue = 0;
    object.quality = 0;
    object.mat2 = 0;
    object.valblack = 0;
    object.valwhite = 255;
    object.matflags2 = 0;
    object.mesh_thickness = 0;
}

/// Original: `imodObjectDelete` (`iobj.c:109`).
pub fn imod_object_delete(object: &mut Iobj) -> i32 {
    object.cont.clear();
    object.mesh.clear();
    object.store.clear();
    crate::imod::libimod::ilabel::imod_label_delete(object.label.take());
    0
}

/// Original: `imodObjectsDelete` (`iobj.c:119`).
pub fn imod_objects_delete(objects: &mut Vec<Iobj>) -> i32 {
    if objects.is_empty() {
        return -1;
    }
    for object in objects.iter_mut() {
        imod_object_delete(object);
    }
    objects.clear();
    0
}

/// Original: `imodObjectChecksum` (`iobj.c:145`).
pub fn imod_object_checksum(obj: &Iobj, ob_num: i32) -> f64 {
    let mut osum = ob_num as f64;
    let mut psum = 0.;
    osum += (obj.red + obj.green + obj.blue) as f64;
    osum += obj.flags as f64;
    osum += obj.pdrawsize as f64;
    osum += obj.symbol as f64;
    osum += obj.symsize as f64;
    osum += obj.linewidth2 as f64;
    osum += obj.linewidth as f64;
    osum += obj.symflags as f64;
    osum += obj.trans as f64;
    osum += obj.cont.len() as f64;
    // The source's operands are all `unsigned char`, promoted to `int` before
    // the additions; the sums do not fit in a byte.
    osum += (obj.ambient as i32 + obj.diffuse as i32 + obj.specular as i32 + obj.shininess as i32)
        as f64;
    let clips = &obj.clips;
    osum += (clips.count as i32 + clips.flags as i32 + clips.trans as i32) as f64;
    // `clips->plane + obj->mat2` is `int + b3dUInt32`, so the addition is done
    // in unsigned arithmetic and wraps.
    osum += (clips.plane as u32).wrapping_add(obj.mat2) as f64;
    // `IMOD_CLIPSIZE` bounds the source arrays; the translated `Iclip_planes`
    // carries seven slots, and a corrupt count is clamped to them.
    for i in 0..(clips.count as usize).min(clips.normal.len()) {
        osum += (clips.normal[i].x + clips.normal[i].y + clips.normal[i].z) as f64;
        osum += (clips.point[i].x + clips.point[i].y + clips.point[i].z) as f64;
    }
    osum += obj.extra[IOBJ_EX_PNT_LIMIT].wrapping_add(obj.extra[IOBJ_EX_2D_TRANS]) as f64;
    osum += (obj.fillred as i32 + obj.fillgreen as i32 + obj.fillblue as i32 + obj.quality as i32)
        as f64;
    osum += (obj.valblack as i32
        + obj.valwhite as i32
        + obj.matflags2 as i32
        + obj.mesh_thickness as i32) as f64;
    osum += super::istore::istore_checksum(&obj.store);
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        psum += cont.surf as f64;
        psum += cont.pts.len() as f64;
        if co != 0 {
            psum += ((cont.pts.len() as i32 - obj.cont[co - 1].pts.len() as i32) % 13) as f64;
        }
        psum += (cont.flags & !(1 << 4) & !(1 << 31)) as f64;
        psum += cont.time as f64;
        for pt in 0..cont.pts.len() {
            psum += (cont.pts[pt].x * ((pt % 7) + 1) as f32) as f64;
            psum += (cont.pts[pt].y * ((pt % 5) + 1) as f32) as f64;
            psum += cont.pts[pt].z as f64;
        }
        psum += super::istore::istore_checksum(&cont.store);
        if !cont.sizes.is_empty() {
            for pt in 0..cont.pts.len() {
                psum += cont.sizes[pt] as f64;
            }
        }
    }
    osum + psum
}

/// Original: `imodObjectCopy` (`iobj.c:202`).
pub fn imod_object_copy(from: &Iobj, to: &mut Iobj) -> i32 {
    *to = from.clone();
    0
}

/// Original: `imodObjectCopyClear` (`iobj.c:216`).
pub fn imod_object_copy_clear(from: &Iobj, to: &mut Iobj) -> i32 {
    *to = from.clone();
    to.cont.clear();
    to.mesh.clear();
    to.store.clear();
    to.label = None;
    0
}

/// Original: `imodObjectDup` (`iobj.c:234`).
pub fn imod_object_dup(object: &Iobj) -> Option<Iobj> {
    Some(object.clone())
}

/// Original: `imodObjectGetContour` (`iobj.c:303`).
pub fn imod_object_get_contour(in_object: Option<&Iobj>, in_index: i32) -> Option<&Icont> {
    let in_object = in_object?;
    if in_index < 0 {
        return None;
    }
    if in_object.cont.is_empty() {
        return None;
    }
    if in_index >= in_object.cont.len() as i32 {
        return None;
    }
    Some(&in_object.cont[in_index as usize])
}

/// Original: `imodObjectGetMesh` (`iobj.c:316`).
pub fn imod_object_get_mesh(in_object: Option<&Iobj>, in_index: i32) -> Option<&Imesh> {
    let in_object = in_object?;
    if in_index < 0 {
        return None;
    }
    if in_object.mesh.is_empty() {
        return None;
    }
    if in_index >= in_object.mesh.len() as i32 {
        return None;
    }
    Some(&in_object.mesh[in_index as usize])
}

/// Original: `imodObjectAddMesh` (`iobj.c:332`).
///
/// `imodel_mesh_add` (`imesh.c:340`) appends a copy of the mesh to the object's
/// mesh array and increments `meshsize`; `imesh.c` has no translated module yet,
/// so the array append is done in place here.
pub fn imod_object_add_mesh(in_object: &mut Iobj, in_mesh: Imesh) -> i32 {
    let surf = in_mesh.surf;
    in_object.mesh.push(in_mesh);
    in_object.surfsize = in_object.surfsize.max(surf as i32);
    in_object.mesh.len() as i32 - 1
}

/// Original: `imodObjectSort` (`iobj.c:349`).
pub fn imod_object_sort(obj: &mut Iobj) -> i32 {
    imod_object_sort_by_surf(obj, 0)
}

/// Original: `imodObjectSortBySurf` (`iobj.c:360`).
pub fn imod_object_sort_by_surf(obj: &mut Iobj, if_by_surf: i32) -> i32 {
    let has_time = iobj_flag_time(obj);

    /*   Sept 1996. added time value as key to sort.
     *              Empty contours no longer need to be deleted.
     */

    if iobj_scat(obj.flags) != 0 {
        return -1;
    }
    if obj.cont.is_empty() {
        return -1;
    }
    if obj.cont.len() < 2 {
        return 0;
    }

    let mut keys = vec![0f64; obj.cont.len()];

    /* Find the min and max Z values and surface numbers */
    let mut minz = i32::MAX;
    let mut min_surf = i32::MAX;
    let mut maxz = -i32::MAX;
    let mut max_surf = -i32::MAX;
    for i in 0..obj.cont.len() {
        if !obj.cont[i].pts.is_empty() {
            let sz = (obj.cont[i].pts[0].z as f64 + 0.5).floor() as i32;
            minz = if minz < sz { minz } else { sz };
            maxz = if maxz > sz { maxz } else { sz };
            min_surf = if min_surf < obj.cont[i].surf {
                min_surf
            } else {
                obj.cont[i].surf
            };
            max_surf = if max_surf > obj.cont[i].surf {
                max_surf
            } else {
                obj.cont[i].surf
            };
        }
    }

    /* Set up scaling to put surface higherthan Z and time higher than surface */
    let surf_scale = (2i32.wrapping_mul(maxz.wrapping_add(2).wrapping_sub(minz))) as f64;
    let time_scale =
        (2i32.wrapping_mul(max_surf.wrapping_add(2).wrapping_sub(min_surf))) as f64 * surf_scale;

    /* Assign a key to each contour */
    for i in 0..obj.cont.len() {
        let sz = if !obj.cont[i].pts.is_empty() {
            ((obj.cont[i].pts[0].z as f64 + 0.5).floor() as i32).wrapping_sub(minz)
        } else {
            maxz.wrapping_add(1)
        };
        keys[i] = sz as f64;
        if if_by_surf != 0 {
            keys[i] += surf_scale * (obj.cont[i].surf.wrapping_sub(min_surf)) as f64;
        }
        if has_time != 0 {
            keys[i] += time_scale * obj.cont[i].time as f64;
        }
    }

    /* For each contour find the minimum key contour above it in array */
    for i in 0..obj.cont.len() - 1 {
        let mut sindex = i;
        let mut min_key = keys[i];

        for j in i + 1..obj.cont.len() {
            if keys[j] < min_key {
                sindex = j;
                min_key = keys[j];
            }
        }

        /* Swap in the lowest key contour */
        if sindex != i {
            obj.cont.swap(i, sindex);
            min_key = keys[i];
            keys[i] = keys[sindex];
            keys[sindex] = min_key;

            /* Swap any contour general store information in place */
            for j in 0..obj.store.len() {
                let stp = &mut obj.store[j];
                /* GEN_STORE_NOINDEX | 3 */
                if stp.flags & ((1 << 4) | 3) != 0 {
                    break;
                }
                /* GEN_STORE_SURFACE */
                if stp.flags & (1 << 6) != 0 {
                    continue;
                }
                if stp.index.i() == i as i32 {
                    stp.index.set_i(sindex as i32);
                } else if stp.index.i() == sindex as i32 {
                    stp.index.set_i(i as i32);
                }
            }
        }
    }

    /* Sort the storage list at the end */
    istore_sort(&mut obj.store);
    0
}

/// Original: `imodObjectVolume` (`iobj.c:468`).
pub fn imod_object_volume(obj: &Iobj) -> f32 {
    let mut ca = 0.0f32;

    if iobj_close(obj.flags) == 0 {
        return 0.;
    }

    for co in 0..obj.cont.len() {
        ca += imod_contour_area(Some(&obj.cont[co]));
    }
    ca
}

/// Original: `imodel_object_centroid` (`iobj.c:492`).
pub fn imodel_object_centroid(obj: &mut Iobj, rcp: &mut Ipoint) -> i32 {
    let mut cpt = Ipoint::default();
    let mut weight = 0f64;
    let mut tweight = 0f64;

    rcp.x = 0.0;
    rcp.y = 0.0;
    rcp.z = 0.0;

    let obj_flags = obj.flags;
    for co in 0..obj.cont.len() {
        if obj.cont[co].pts.is_empty() {
            continue;
        }
        set_or_clear_flags(
            &mut obj.cont[co].flags,
            ICONT_TEMPUSE,
            (obj_flags & IMOD_OBJFLAG_OPEN) as i32,
        );
        if imodel_contour_centroid(Some(&obj.cont[co]), &mut cpt, &mut weight) != 0 {
            return 1;
        }
        set_or_clear_flags(&mut obj.cont[co].flags, ICONT_TEMPUSE, 0);
        tweight += weight;
        rcp.x += cpt.x;
        rcp.y += cpt.y;
        rcp.z += cpt.z; /* z-scale is done outside of function */
    }
    if tweight == 0. {
        return 1;
    }
    rcp.x = (rcp.x as f64 / tweight) as f32;
    rcp.y = (rcp.y as f64 / tweight) as f32;
    rcp.z = (rcp.z as f64 / tweight) as f32;
    0
}

/// Original: `imodObjectAddContour` (`iobj.c:531`).
pub fn imod_object_add_contour(obj: &mut Iobj, ncont: Icont) -> i32 {
    let index = obj.cont.len() as i32;
    imod_object_insert_contour(obj, ncont, index)
}

/// Original: `imodObjectInsertContour` (`iobj.c:544`).
pub fn imod_object_insert_contour(obj: &mut Iobj, ncont: Icont, index: i32) -> i32 {
    if index < 0 || index > obj.cont.len() as i32 {
        return -1;
    }

    obj.cont.insert(index as usize, ncont);
    let surf = obj.cont[index as usize].surf;
    if index < obj.cont.len() as i32 - 1 {
        istore_shift_index(&mut obj.store, index, -1, 1);
    }
    if obj.surfsize < surf {
        obj.surfsize = surf;
    }

    index
}

/// Original: `imodObjectRemoveContour` (`iobj.c:582`).
pub fn imod_object_remove_contour(obj: &mut Iobj, index: i32) -> i32 {
    if index < 0 {
        return 1;
    }
    if index >= obj.cont.len() as i32 {
        return 1;
    }

    istore_delete_cont_surf(&mut obj.store, index, 0);
    obj.cont.remove(index as usize);

    /* DMN 9/20/04: clean out labels for non-existing surfaces */
    imod_object_clean_surf(obj);
    0
}

/// Original: `imodObjectCleanSurf` (`iobj.c:610`).
pub fn imod_object_clean_surf(obj: &mut Iobj) {
    /* Update the maximum surface number while we are at it */
    obj.surfsize = 0;
    for co in 0..obj.cont.len() {
        obj.surfsize = if obj.surfsize > obj.cont[co].surf {
            obj.surfsize
        } else {
            obj.cont[co].surf
        };
    }
    for co in 0..obj.mesh.len() {
        let surf = obj.mesh[co].surf as i32;
        obj.surfsize = if obj.surfsize > surf {
            obj.surfsize
        } else {
            surf
        };
    }

    if obj.label.is_none() {
        return;
    }

    let mut i = obj.label.as_ref().unwrap().label.len() as i32 - 1;
    while i >= 0 {
        let mut found = 0;
        let index = obj.label.as_ref().unwrap().label[i as usize].index;
        for co in 0..obj.cont.len() {
            if obj.cont[co].surf == index {
                found = 1;
                break;
            }
        }
        if found == 0 {
            crate::imod::libimod::ilabel::imod_label_item_delete(obj.label.as_mut(), index);
        }
        i -= 1;
    }
}

/// Original: `imodObjectGetBBox` (`iobj.c:886`).
///
/// The mesh branch's `imodMeshGetBBox` (`imesh.c:953`) has no translated module
/// yet, so its body is evaluated in place; `imodMeshPolyNormFactors`
/// (`imesh.c:312`) likewise.
pub fn imod_object_get_bbox(obj: &Iobj, ll: &mut Ipoint, ur: &mut Ipoint) -> i32 {
    let mut min = Ipoint::default();
    let mut max = Ipoint::default();
    let mut cont_ret = -1;

    min.x = f32::MAX;
    min.y = f32::MAX;
    min.z = f32::MAX;
    max.x = -f32::MAX;
    max.y = -f32::MAX;
    max.z = -f32::MAX;

    if obj.cont.is_empty() && obj.mesh.is_empty() {
        return -1;
    }

    *ll = min;
    *ur = max;
    for co in 0..obj.cont.len() {
        if imod_contour_get_bbox(Some(&obj.cont[co]), &mut min, &mut max) != 0 {
            continue;
        }

        if min.x < ll.x {
            ll.x = min.x;
        }
        if min.y < ll.y {
            ll.y = min.y;
        }
        if min.z < ll.z {
            ll.z = min.z;
        }

        if max.x > ur.x {
            ur.x = max.x;
        }
        if max.y > ur.y {
            ur.y = max.y;
        }
        if max.z > ur.z {
            ur.z = max.z;
        }
        cont_ret = 0;
    }

    if !obj.cont.is_empty() {
        return cont_ret;
    }

    for co in 0..obj.mesh.len() {
        /* imodMeshGetBBox (imesh.c:953) */
        let mesh = &obj.mesh[co];
        if mesh.list.is_empty() || mesh.vert.is_empty() {
            continue;
        }
        min.x = 1.0e30;
        min.y = 1.0e30;
        min.z = 1.0e30;
        max.x = -1.0e30;
        max.y = -1.0e30;
        max.z = -1.0e30;
        let mut i = 0usize;
        while i < mesh.list.len() {
            match mesh.list[i] {
                /* IMOD_MESH_BGNBIGPOLY, IMOD_MESH_BGNPOLY.  `imesh.c:967` never
                advances `i` inside this loop, so a mesh that actually starts a
                BGNPOLY here spins forever in the C as well. */
                -24 | -21 => {
                    i += 1;
                    while mesh.list[i] != IMOD_MESH_ENDPOLY {
                        let pt = mesh.vert[mesh.list[i] as usize];
                        min.x = min.x.min(pt.x);
                        min.y = min.y.min(pt.y);
                        min.z = min.z.min(pt.z);
                        max.x = max.x.max(pt.x);
                        max.y = max.y.max(pt.y);
                        max.z = max.z.max(pt.z);
                    }
                }
                IMOD_MESH_BGNPOLYNORM2 | IMOD_MESH_BGNPOLYNORM => {
                    /* imodMeshPolyNormFactors (imesh.c:312) */
                    let (list_inc, vert_base) = if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                        (2usize, 1usize)
                    } else {
                        (1usize, 0usize)
                    };
                    i += 1;
                    while mesh.list[i] != IMOD_MESH_ENDPOLY {
                        for _ in 0..3 {
                            let pt = mesh.vert[mesh.list[i + vert_base] as usize];
                            i += list_inc;
                            min.x = min.x.min(pt.x);
                            min.y = min.y.min(pt.y);
                            min.z = min.z.min(pt.z);
                            max.x = max.x.max(pt.x);
                            max.y = max.y.max(pt.y);
                            max.z = max.z.max(pt.z);
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if min.x > 1.0e29 {
            continue;
        }

        if min.x < ll.x {
            ll.x = min.x;
        }
        if min.y < ll.y {
            ll.y = min.y;
        }
        if min.z < ll.z {
            ll.z = min.z;
        }

        if max.x > ur.x {
            ur.x = max.x;
        }
        if max.y > ur.y {
            ur.y = max.y;
        }
        if max.z > ur.z {
            ur.z = max.z;
        }
    }
    1
}

/// Original: `imodObjectGetColor` (`iobj.c:938`).
pub fn imod_object_get_color(
    in_object: &Iobj,
    out_red: &mut f32,
    out_green: &mut f32,
    out_blue: &mut f32,
) {
    *out_red = in_object.red;
    *out_green = in_object.green;
    *out_blue = in_object.blue;
}

/// Original: `imodObjectSetColor` (`iobj.c:950`).
pub fn imod_object_set_color(in_object: &mut Iobj, in_red: f32, in_green: f32, in_blue: f32) {
    in_object.red = in_red;
    in_object.green = in_green;
    in_object.blue = in_blue;
}

/// Original: `imodObjectGetMaxContour` (`iobj.c:962`).
pub fn imod_object_get_max_contour(in_object: Option<&Iobj>) -> i32 {
    let Some(in_object) = in_object else {
        return 0;
    };
    in_object.cont.len() as i32
}

/// Original: `imodObjectGetMaxPoints` (`iobj.c:971`).
pub fn imod_object_get_max_points(in_object: Option<&Iobj>) -> i32 {
    let mut max_pts = 0;
    let Some(in_object) = in_object else {
        return 0;
    };
    for co in 0..in_object.cont.len() {
        let psize = in_object.cont[co].pts.len() as i32;
        max_pts = if max_pts > psize { max_pts } else { psize };
    }
    max_pts
}

/// Original: `imodObjectGetName` (`iobj.c:984`).
///
/// The C returns `inObject->name`, a `char *` into the fixed 64-byte array, or
/// NULL for a NULL object.  The slice stops at the first NUL, which is what
/// every caller's `strlen`/`%s` does with it, and `None` is the NULL return.
pub fn imod_object_get_name(in_object: Option<&Iobj>) -> Option<&[u8]> {
    let in_object = in_object?;
    let end = in_object
        .name
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(IOBJ_STRSIZE);
    Some(&in_object.name[..end])
}

/// Original: `imodObjectSetName` (`iobj.c:994`).
pub fn imod_object_set_name(obj: &mut Iobj, in_name: &[u8]) -> i32 {
    let mut retval = 0;
    for i in 0..IOBJ_STRSIZE {
        obj.name[i] = 0x00;
    }
    let mut len = in_name.len();
    if len > IOBJ_STRSIZE - 1 {
        len = IOBJ_STRSIZE - 1;
        retval += 1;
    }
    for i in 0..len {
        obj.name[i] = in_name[i];
    }
    retval
}

/// Original: `imodObjectGetValue` (`iobj.c:1044`).
pub fn imod_object_get_value(in_object: &Iobj, in_value_type: i32) -> i32 {
    match in_value_type {
        IOBJ_MAX_CONTOUR => in_object.cont.len() as i32,
        IOBJ_LINE_WIDTH => in_object.linewidth as i32,
        IOBJ_LINE_WIDTH2 => in_object.linewidth2 as i32,
        IOBJ_POINT_SIZE => in_object.pdrawsize,
        IOBJ_MAX_MESH => in_object.mesh.len() as i32,
        IOBJ_MAX_SURFACE => in_object.surfsize,
        IOBJ_SYM_TYPE => in_object.symbol as i32,
        IOBJ_SYM_SIZE => in_object.symsize as i32,
        IOBJ_SYM_FLAGS => in_object.symflags as i32,

        IOBJ_FLAG_CLOSED => iobj_close(in_object.flags),

        IOBJ_FLAG_CONNECTED => (iobj_scat(in_object.flags) == 0) as i32,

        IOBJ_FLAG_FILLED => iobj_fill(in_object.flags) as i32,

        IOBJ_FLAG_DRAW => (iobj_draw(in_object.flags) == 0) as i32,

        IOBJ_FLAG_MESH => iobj_mesh(in_object.flags) as i32,

        IOBJ_FLAG_LINE => (iobj_line(in_object.flags) == 0) as i32,

        IOBJ_FLAG_PLANAR => {
            if iobj_planar(in_object.flags) != 0 {
                1
            } else {
                0
            }
        }

        IOBJ_FLAG_TIME => iobj_flag_time(in_object) as i32,

        IOBJ_FLAG_EXTRA_IN_MODV => (in_object.flags & IMOD_OBJFLAG_EXTRA_MODV) as i32,

        IOBJ_FLAG_PNT_ON_SEC => (in_object.flags & IMOD_OBJFLAG_PNT_ON_SEC) as i32,

        IOBJ_FLAG_EXTRA_IN_SLICER => {
            (in_object.extra[IOBJ_EX_FLAGS] & IOBJ_EXFLAG_SLICER_ONLY) as i32
        }
        _ => 0,
    }
}

/// Original: `setObjFlag` (`iobj.c:1104`).
pub fn set_obj_flag(in_object: &mut Iobj, flag: u32, state: i32) {
    if state != 0 {
        in_object.flags |= flag;
    } else {
        in_object.flags &= !flag;
    }
}

/// Original: `imodObjectSetValue` (`iobj.c:1122`).
pub fn imod_object_set_value(in_object: &mut Iobj, in_value_type: i32, in_value: i32) {
    match in_value_type {
        IOBJ_LINE_WIDTH => {
            in_object.linewidth = in_value as u8;
        }

        IOBJ_LINE_WIDTH2 => {
            in_object.linewidth2 = in_value as u8;
        }

        IOBJ_POINT_SIZE => {
            in_object.pdrawsize = in_value;
        }

        IOBJ_SYM_TYPE => {
            in_object.symbol = in_value as u8;
        }

        IOBJ_SYM_SIZE => {
            in_object.symsize = in_value as u8;
        }

        IOBJ_SYM_FLAGS => {
            in_object.symflags = in_value as u8;
        }

        IOBJ_FLAG_CLOSED => {
            set_obj_flag(in_object, IMOD_OBJFLAG_OPEN, (in_value == 0) as i32);
        }

        IOBJ_FLAG_CONNECTED => {
            set_obj_flag(in_object, IMOD_OBJFLAG_SCAT, (in_value == 0) as i32);
        }

        IOBJ_FLAG_FILLED => {
            set_obj_flag(in_object, IMOD_OBJFLAG_FILL, in_value);
        }

        IOBJ_FLAG_DRAW => {
            set_obj_flag(in_object, IMOD_OBJFLAG_OFF, (in_value == 0) as i32);
        }

        IOBJ_FLAG_MESH => {
            set_obj_flag(in_object, IMOD_OBJFLAG_MESH, in_value);
        }

        IOBJ_FLAG_LINE => {
            set_obj_flag(in_object, IMOD_OBJFLAG_NOLINE, (in_value == 0) as i32);
        }

        IOBJ_FLAG_EXTRA_IN_MODV => {
            set_obj_flag(in_object, IMOD_OBJFLAG_EXTRA_MODV, in_value);
        }

        IOBJ_FLAG_PNT_ON_SEC => {
            set_obj_flag(in_object, IMOD_OBJFLAG_PNT_ON_SEC, in_value);
        }

        IOBJ_FLAG_EXTRA_IN_SLICER => {
            set_or_clear_flags(
                &mut in_object.extra[IOBJ_EX_FLAGS],
                IOBJ_EXFLAG_SLICER_ONLY,
                in_value,
            );
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::icont::imod_contour_new;
    use crate::imod::libimod::ipoint::imod_point_append;
    use crate::imod::libimod::istore::{Istore, StoreUnion, istore_insert};
    use std::fmt::Write as _;

    /// The reference output below was produced by a C driver compiled against
    /// the pinned `IMOD/libimod/iobj.c` and linked to the reference `libimod`
    /// (`/tmp/imod-reference-build/buildlib`).  Floating point values are
    /// printed as raw IEEE bit patterns so the comparison is exact rather than
    /// rounded through `%g`.
    const NATIVE: &str = r#"def flags 402653184 axis 0 draw 1 rgb 3f000000 3f000000 3f000000 pdraw 0 sym 1 3 lw 1 1 ls 0 sf 0 sp 0 tr 0 ss 0 amb 102 dif 255 spec 127 shin 4 fill 0 0 0 q 0 mat2 0 vb 0 vw 255 mf2 0 mt 0
F0 vol 4599c55c
F0 bbox 0 416fc0df 3fc8c991 bf800000 42a6e7dd 42c5908b 40400000
F0 cent 0 42505f7a 425ae1f6 3fa24ee3
F0 maxc 6 maxp 5
F0 color 3f000000 3f000000 3f000000
F0 csum 40c3aaaedd6a0000
F0 val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F0 surfsize 3
F0 getcont 0 1 0 0
F0 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F0 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F0 sort 0
F0 cont [5,0,0,bf800000] [5,3,0,bf800000] [5,1,0,00000000] [5,3,0,3f800000] [5,0,0,40000000] [5,2,0,40400000]
F0 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F0 sortsurf 0
F0 cont [5,0,0,bf800000] [5,0,0,40000000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F0 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F0 cleansurf 3
F0 insert 2
F0 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F0 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F0 surfsize 11
F0 remove 0
F0 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F0 store [3,0,0,1] [3,0,0,21] [3,0,0,41]
F0 surfsize 11
F0 badremove 1 1
F0 vol 4585309e
F0 bbox 0 3fc00000 3fc8c991 bf800000 42a6e7dd 42c5908b 41100000
F0 cent 0 424a2ebe 4262c210 3fb59ee8
F0 maxc 6 maxp 5
F0 color 3f000000 3f000000 3f000000
F0 csum 40c0b24ba12e0000
F0 val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F0 surfsize 11
F0 getcont 0 1 0 0
F0 setval 0:0:1:1:0:1:3:0 0:0:1:1:0:1:3:0 0:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 136:0:1:1:0:1:3:0 392:0:1:1:0:1:3:0 904:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:0:3:0 1928:2:1:1:1:0:1:0 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1
F0 setcolor 3e800000 3f000000 3f400000
F1 vol 00000000
F1 bbox 0 3edac1b5 40573dae bf800000 42ba2ff4 42c6c90d 40400000
F1 cent 0 42416b67 424c894a 3f12eac1
F1 maxc 6 maxp 5
F1 color 3f000000 3f000000 3f000000
F1 csum 40c3fae40d054000
F1 val 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F1 surfsize 3
F1 getcont 0 1 0 0
F1 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F1 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F1 sort 0
F1 cont [5,0,0,bf800000] [5,3,0,bf800000] [5,1,0,00000000] [5,3,0,3f800000] [5,0,0,40000000] [5,2,0,40400000]
F1 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F1 sortsurf 0
F1 cont [5,0,0,bf800000] [5,0,0,40000000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F1 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F1 cleansurf 3
F1 insert 2
F1 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F1 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F1 surfsize 11
F1 remove 0
F1 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F1 store [3,0,0,1] [3,0,0,21] [3,0,0,41]
F1 surfsize 11
F1 badremove 1 1
F1 vol 00000000
F1 bbox 0 3edac1b5 40200000 bf800000 42ba2ff4 42c6c90d 41100000
F1 cent 0 42494091 4253642c 3f2f6a93
F1 maxc 6 maxp 5
F1 color 3f000000 3f000000 3f000000
F1 csum 40c1414be9ed4000
F1 val 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F1 surfsize 11
F1 getcont 0 1 0 0
F1 setval 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 8:0:1:1:0:1:3:0 136:0:1:1:0:1:3:0 392:0:1:1:0:1:3:0 904:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:0:3:0 1928:2:1:1:1:0:1:0 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1
F1 setcolor 3e800000 3f000000 3f400000
F2 vol 00000000
F2 bbox 0 40309161 3fffddff bf800000 42b54dea 42c2c805 40400000
F2 cent 0 4241b341 423125f5 3f0cd320
F2 maxc 6 maxp 5
F2 color 3f000000 3f000000 3f000000
F2 csum 40c0c510a6e80000
F2 val 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F2 surfsize 3
F2 getcont 0 1 0 0
F2 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F2 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F2 sort -1
F2 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F2 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F2 sortsurf -1
F2 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F2 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F2 cleansurf 3
F2 insert 2
F2 cont [5,0,0,bf800000] [5,3,0,3f800000] [1,11,0,41100000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F2 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,4,11] [3,0,4,31]
F2 surfsize 11
F2 remove 0
F2 cont [5,0,0,bf800000] [5,3,0,3f800000] [1,11,0,41100000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F2 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F2 surfsize 11
F2 badremove 1 1
F2 vol 00000000
F2 bbox 0 3fc00000 3fffddff bf800000 42a9fb64 42c2c805 41100000
F2 cent 0 423b4efc 423728a7 3e66438a
F2 maxc 6 maxp 5
F2 color 3f000000 3f000000 3f000000
F2 csum 40bbf253ad500000
F2 val 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F2 surfsize 11
F2 getcont 0 1 0 0
F2 setval 512:0:1:1:0:1:3:0 512:0:1:1:0:1:3:0 512:0:1:1:0:1:3:0 520:0:1:1:0:1:3:0 520:0:1:1:0:1:3:0 520:0:1:1:0:1:3:0 520:0:1:1:0:1:3:0 648:0:1:1:0:1:3:0 904:0:1:1:0:1:3:0 904:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:0:3:0 1928:2:1:1:1:0:1:0 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1
F2 setcolor 3e800000 3f000000 3f400000
F3 vol 00000000
F3 bbox 0 3ffa01f3 3e48018f bf800000 42c758ce 42c3ac27 40400000
F3 cent 0 4233039a 42473540 3f3710f3
F3 maxc 6 maxp 5
F3 color 3f000000 3f000000 3f000000
F3 csum 4110984a8f9c5c00
F3 val 0 0 0 0 0 0 0 0 0 1 0 0 262144 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F3 surfsize 3
F3 getcont 0 1 0 0
F3 cont [5,0,1,bf800000] [5,3,3,3f800000] [5,2,2,40400000] [5,1,1,00000000] [5,0,3,40000000] [5,3,2,bf800000]
F3 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F3 sort 0
F3 cont [5,0,1,bf800000] [5,1,1,00000000] [5,3,2,bf800000] [5,2,2,40400000] [5,3,3,3f800000] [5,0,3,40000000]
F3 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,1,11] [3,0,1,31]
F3 sortsurf 0
F3 cont [5,0,1,bf800000] [5,1,1,00000000] [5,2,2,40400000] [5,3,2,bf800000] [5,0,3,40000000] [5,3,3,3f800000]
F3 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,1,11] [3,0,1,31]
F3 cleansurf 3
F3 insert 2
F3 cont [5,0,1,bf800000] [5,1,1,00000000] [1,11,0,41100000] [5,2,2,40400000] [5,3,2,bf800000] [5,0,3,40000000] [5,3,3,3f800000]
F3 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,1,11] [3,0,1,31]
F3 surfsize 11
F3 remove 0
F3 cont [5,0,1,bf800000] [5,1,1,00000000] [1,11,0,41100000] [5,3,2,bf800000] [5,0,3,40000000] [5,3,3,3f800000]
F3 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,1,11] [3,0,1,31]
F3 surfsize 11
F3 badremove 1 1
F3 vol 00000000
F3 bbox 0 3fc00000 3e48018f bf800000 42c758ce 42c3ac27 41100000
F3 cent 0 4225661a 4242e0f0 3e96db4a
F3 maxc 6 maxp 5
F3 color 3f000000 3f000000 3f000000
F3 csum 411077b689495c00
F3 val 0 0 0 0 0 0 0 0 0 1 0 0 262144 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F3 surfsize 11
F3 getcont 0 1 0 0
F3 setval 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262152:0:1:1:0:1:3:0 262280:0:1:1:0:1:3:0 262536:0:1:1:0:1:3:0 263048:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:0:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:0:1:3:0 264072:2:1:1:1:1:3:0 264072:2:1:1:1:1:3:0 264072:2:1:1:1:1:3:0 264072:2:1:1:1:1:3:0 264072:2:1:1:1:0:3:0 264072:2:1:1:1:0:1:0 264072:2:1:1:1:0:1:1 264072:2:1:1:1:0:1:1 264072:2:1:1:1:0:1:1 264072:2:1:1:1:0:1:1
F3 setcolor 3e800000 3f000000 3f400000
F4 vol 45881c10
F4 bbox 0 3f0ef91e 3fc47d88 bf800000 42c702dd 42c228a4 40400000
F4 cent 0 42738e74 42632b90 3f6da447
F4 maxc 6 maxp 5
F4 color 3f000000 3f000000 3f000000
F4 csum 40c8ec8bd19e0000
F4 val 0 1 0 1 0 0 0 0 256 1 1024 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F4 surfsize 3
F4 getcont 0 1 0 0
F4 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F4 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F4 sort 0
F4 cont [5,0,0,bf800000] [5,3,0,bf800000] [5,1,0,00000000] [5,3,0,3f800000] [5,0,0,40000000] [5,2,0,40400000]
F4 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F4 sortsurf 0
F4 cont [5,0,0,bf800000] [5,0,0,40000000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F4 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F4 cleansurf 3
F4 insert 2
F4 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F4 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F4 surfsize 11
F4 remove 0
F4 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F4 store [3,0,0,1] [3,0,0,21] [3,0,0,41]
F4 surfsize 11
F4 badremove 1 1
F4 vol 45822ad9
F4 bbox 0 3f0ef91e 40200000 bf800000 42c702dd 42c228a4 41100000
F4 cent 0 4270e86e 42653edf 3f7cfdc2
F4 maxc 6 maxp 5
F4 color 3f000000 3f000000 3f000000
F4 csum 40c59150afbe0000
F4 val 0 1 0 1 0 0 0 0 256 1 1024 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F4 surfsize 11
F4 getcont 0 1 0 0
F4 setval 1282:0:1:1:0:1:3:0 1280:0:1:1:0:1:3:0 1280:0:1:1:0:1:3:0 1288:0:1:1:0:1:3:0 1288:0:1:1:0:1:3:0 1288:0:1:1:0:1:3:0 1288:0:1:1:0:1:3:0 1416:0:1:1:0:1:3:0 1416:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:0:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:0:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:1:3:0 1928:2:1:1:1:0:3:0 1928:2:1:1:1:0:1:0 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1 1928:2:1:1:1:0:1:1
F4 setcolor 3e800000 3f000000 3f400000
F5 vol 00000000
F5 bbox 0 40a1b943 40fc27f8 bf800000 42b2ec85 42c6f7ed 40400000
F5 cent 0 42430b3b 42759b13 3f0f5545
F5 maxc 6 maxp 5
F5 color 3f000000 3f000000 3f000000
F5 csum 40d4aeddc4cc0000
F5 val 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 3 1 1 3 0 0 0 0
F5 surfsize 3
F5 getcont 0 1 0 0
F5 cont [5,0,0,bf800000] [5,3,0,3f800000] [5,2,0,40400000] [5,1,0,00000000] [5,0,0,40000000] [5,3,0,bf800000]
F5 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F5 sort 0
F5 cont [5,0,0,bf800000] [5,3,0,bf800000] [5,1,0,00000000] [5,3,0,3f800000] [5,0,0,40000000] [5,2,0,40400000]
F5 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F5 sortsurf 0
F5 cont [5,0,0,bf800000] [5,0,0,40000000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F5 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,2,11] [3,0,2,31]
F5 cleansurf 3
F5 insert 2
F5 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,1,0,00000000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F5 store [3,0,0,1] [3,0,0,21] [3,0,0,41] [3,0,3,11] [3,0,3,31]
F5 surfsize 11
F5 remove 0
F5 cont [5,0,0,bf800000] [5,0,0,40000000] [1,11,0,41100000] [5,2,0,40400000] [5,3,0,bf800000] [5,3,0,3f800000]
F5 store [3,0,0,1] [3,0,0,21] [3,0,0,41]
F5 surfsize 11
F5 badremove 1 1
F5 vol 00000000
F5 bbox 0 3fc00000 40200000 bf800000 42b2ec85 42c6f7ed 41100000
F5 cent 0 4240e61f 426c942a 3f32ff3d
F5 maxc 6 maxp 5
F5 color 3f000000 3f000000 3f000000
F5 csum 40d2bca6572c0000
F5 val 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 6 1 0 0 11 1 1 3 0 0 0 0
F5 surfsize 11
F5 getcont 0 1 0 0
F5 setval 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10248:0:1:1:0:1:3:0 10376:0:1:1:0:1:3:0 10632:0:1:1:0:1:3:0 11144:0:1:1:0:1:3:0 12168:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:0:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:0:1:3:0 10120:2:1:1:1:1:3:0 10120:2:1:1:1:1:3:0 10120:2:1:1:1:1:3:0 10120:2:1:1:1:1:3:0 10120:2:1:1:1:0:3:0 10120:2:1:1:1:0:1:0 10120:2:1:1:1:0:1:1 10120:2:1:1:1:0:1:1 10120:2:1:1:1:0:1:1 10120:2:1:1:1:0:1:1
F5 setcolor 3e800000 3f000000 3f400000
E vol 00000000
E bbox -1 c0f00000 c1080000 c1180000 40f00000 41080000 41180000
E cent 1 00000000 00000000 00000000
E maxc 0 maxp 0
E color 3f000000 3f000000 3f000000
E csum 41b80002f1800000
E val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 3 0 0 0 0
E surfsize 0
E getcont 0 0 0 0
E sort -1 sortsurf -1
E cleansurf 0
S sort 0
S vol 44e61d37
S bbox 0 41488b11 4014d529 bf800000 42a86390 42bb66e6 bf800000
S cent 0 421b4237 4251700c bf800000
S maxc 1 maxp 4
S color 3f000000 3f000000 3f000000
S csum 409aecfbd5c00000
S val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 3 0 0 0 0
S surfsize 0
S getcont 0 1 0 0
Z sort 0
Z vol 00000000
Z bbox -1 7f7fffff 7f7fffff 7f7fffff ff7fffff ff7fffff ff7fffff
Z cent 1 00000000 00000000 00000000
Z maxc 3 maxp 0
Z color 3f000000 3f000000 3f000000
Z csum 4087cc0000000000
Z val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 3 1 0 0 3 1 1 3 0 0 0 0
Z surfsize 3
Z getcont 0 1 0 0
Z cont [0,0,0,c479c000] [0,3,0,c479c000] [0,2,0,c479c000]
M add 0
M vol 00000000
M bbox 1 3f800000 3f800000 00000000 40a00000 40e00000 40400000
M cent 1 00000000 00000000 00000000
M maxc 0 maxp 0
M color 3f000000 3f000000 3f000000
M csum 41b80002f1800000
M val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 5 1 1 3 0 0 0 0
M surfsize 5
M getcont 0 0 0 1
M add 1
M vol 00000000
M bbox 1 3f800000 c0c00000 00000000 41400000 40e00000 40800000
M cent 1 00000000 00000000 00000000
M maxc 0 maxp 0
M color 3f000000 3f000000 3f000000
M csum 41b80002f1800000
M val 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 2 5 1 1 3 0 0 0 0
M surfsize 5
M getcont 0 0 0 1
M cleansurf 5
N set 0 [short name]
N set 1 [abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk]
N set 0 []
"#;

    fn frand(sd: &mut u32) -> f32 {
        *sd = sd.wrapping_mul(1103515245).wrapping_add(12345);
        ((*sd >> 16) & 0x7fff) as f32 / 327.67f32
    }

    fn fb(f: f32) -> String {
        format!("{:08x}", f.to_bits())
    }

    fn db(d: f64) -> String {
        format!("{:016x}", d.to_bits())
    }

    fn build(sd: &mut u32, flags: u32, ncont: i32, npt: i32, withtime: i32, nstore: i32) -> Iobj {
        let mut o = imod_object_new().unwrap();
        o.flags = flags;
        for c in 0..ncont {
            let mut ct = imod_contour_new().unwrap();
            let z = ((c * 7) % 5) as f32 - 1.0f32;
            for _ in 0..npt {
                let pt = Ipoint {
                    x: frand(sd),
                    y: frand(sd),
                    z,
                };
                imod_point_append(&mut ct, pt);
            }
            ct.surf = (c * 3) % 4;
            ct.time = if withtime != 0 { ((c * 5) % 3) + 1 } else { 0 };
            imod_object_add_contour(&mut o, ct);
        }
        for k in 0..nstore {
            let st = Istore {
                type_: 3,
                flags: 0,
                index: StoreUnion::from_i((k * 3) % if ncont != 0 { ncont } else { 1 }),
                value: StoreUnion::from_i(10 * k + 1),
            };
            istore_insert(&mut o.store, st);
        }
        o
    }

    fn dumpstore(out: &mut String, tag: &str, o: &Iobj) {
        write!(out, "{tag} store").unwrap();
        for st in &o.store {
            write!(
                out,
                " [{},{},{},{}]",
                st.type_,
                st.flags,
                st.index.i(),
                st.value.i()
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    fn dumpconts(out: &mut String, tag: &str, o: &Iobj) {
        write!(out, "{tag} cont").unwrap();
        for c in &o.cont {
            write!(
                out,
                " [{},{},{},{}]",
                c.pts.len(),
                c.surf,
                c.time,
                fb(if !c.pts.is_empty() {
                    c.pts[0].z
                } else {
                    -999.0f32
                })
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    fn report(out: &mut String, tag: &str, o: &mut Iobj) {
        writeln!(out, "{tag} vol {}", fb(imod_object_volume(o))).unwrap();
        let mut ll = Ipoint {
            x: -7.5,
            y: -8.5,
            z: -9.5,
        };
        let mut ur = Ipoint {
            x: 7.5,
            y: 8.5,
            z: 9.5,
        };
        let i = imod_object_get_bbox(o, &mut ll, &mut ur);
        write!(out, "{tag} bbox {i} {} {} {}", fb(ll.x), fb(ll.y), fb(ll.z)).unwrap();
        writeln!(out, " {} {} {}", fb(ur.x), fb(ur.y), fb(ur.z)).unwrap();
        let mut cen = Ipoint::default();
        let i = imodel_object_centroid(o, &mut cen);
        writeln!(
            out,
            "{tag} cent {i} {} {} {}",
            fb(cen.x),
            fb(cen.y),
            fb(cen.z)
        )
        .unwrap();
        writeln!(
            out,
            "{tag} maxc {} maxp {}",
            imod_object_get_max_contour(Some(o)),
            imod_object_get_max_points(Some(o))
        )
        .unwrap();
        let (mut r, mut g, mut b) = (0f32, 0f32, 0f32);
        imod_object_get_color(o, &mut r, &mut g, &mut b);
        writeln!(out, "{tag} color {} {} {}", fb(r), fb(g), fb(b)).unwrap();
        writeln!(out, "{tag} csum {}", db(imod_object_checksum(o, 3))).unwrap();
        write!(out, "{tag} val").unwrap();
        for i in 0..45 {
            write!(out, " {}", imod_object_get_value(o, i)).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "{tag} surfsize {}", o.surfsize).unwrap();
        let len = o.cont.len() as i32;
        writeln!(
            out,
            "{tag} getcont {} {} {} {}",
            imod_object_get_contour(Some(o), -1).is_some() as i32,
            imod_object_get_contour(Some(o), 0).is_some() as i32,
            imod_object_get_contour(Some(o), len).is_some() as i32,
            imod_object_get_mesh(Some(o), 0).is_some() as i32
        )
        .unwrap();
    }

    fn driver() -> String {
        let mut out = String::new();
        let flagset = [
            0u32,
            IMOD_OBJFLAG_OPEN,
            IMOD_OBJFLAG_SCAT,
            IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_TIME,
            IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_FILL | IMOD_OBJFLAG_OFF,
            IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_PLANAR | IMOD_OBJFLAG_OPEN,
        ];

        let o = imod_object_new().unwrap();
        write!(
            out,
            "def flags {} axis {} draw {} rgb {} {} {} pdraw {} sym {} {} lw {} {} ",
            o.flags,
            o.axis,
            o.drawmode,
            fb(o.red),
            fb(o.green),
            fb(o.blue),
            o.pdrawsize,
            o.symbol,
            o.symsize,
            o.linewidth2,
            o.linewidth
        )
        .unwrap();
        write!(
            out,
            "ls {} sf {} sp {} tr {} ss {} amb {} dif {} spec {} shin {} fill {} {} {} q {} ",
            o.linesty,
            o.symflags,
            o.sympad,
            o.trans,
            o.surfsize,
            o.ambient,
            o.diffuse,
            o.specular,
            o.shininess,
            o.fillred,
            o.fillgreen,
            o.fillblue,
            o.quality
        )
        .unwrap();
        writeln!(
            out,
            "mat2 {} vb {} vw {} mf2 {} mt {}",
            o.mat2, o.valblack, o.valwhite, o.matflags2, o.mesh_thickness
        )
        .unwrap();

        for i in 0..6 {
            let tag = format!("F{i}");
            let tag = tag.as_str();
            let mut sd = 12345u32 + i as u32;
            let mut o = build(
                &mut sd,
                flagset[i],
                6,
                5,
                (flagset[i] & IMOD_OBJFLAG_TIME != 0) as i32,
                5,
            );
            report(&mut out, tag, &mut o);
            dumpconts(&mut out, tag, &o);
            dumpstore(&mut out, tag, &o);

            writeln!(out, "{tag} sort {}", imod_object_sort(&mut o)).unwrap();
            dumpconts(&mut out, tag, &o);
            dumpstore(&mut out, tag, &o);

            writeln!(
                out,
                "{tag} sortsurf {}",
                imod_object_sort_by_surf(&mut o, 1)
            )
            .unwrap();
            dumpconts(&mut out, tag, &o);
            dumpstore(&mut out, tag, &o);

            imod_object_clean_surf(&mut o);
            writeln!(out, "{tag} cleansurf {}", o.surfsize).unwrap();

            let mut ct = imod_contour_new().unwrap();
            imod_point_append(
                &mut ct,
                Ipoint {
                    x: 1.5,
                    y: 2.5,
                    z: 9.0,
                },
            );
            ct.surf = 11;
            writeln!(
                out,
                "{tag} insert {}",
                imod_object_insert_contour(&mut o, ct, 2)
            )
            .unwrap();
            dumpconts(&mut out, tag, &o);
            dumpstore(&mut out, tag, &o);
            writeln!(out, "{tag} surfsize {}", o.surfsize).unwrap();

            writeln!(
                out,
                "{tag} remove {}",
                imod_object_remove_contour(&mut o, 3)
            )
            .unwrap();
            dumpconts(&mut out, tag, &o);
            dumpstore(&mut out, tag, &o);
            writeln!(out, "{tag} surfsize {}", o.surfsize).unwrap();
            writeln!(
                out,
                "{tag} badremove {} {}",
                imod_object_remove_contour(&mut o, -1),
                imod_object_remove_contour(&mut o, 1000)
            )
            .unwrap();

            report(&mut out, tag, &mut o);

            write!(out, "{tag} setval").unwrap();
            for j in 0..45 {
                imod_object_set_value(&mut o, j, if j % 3 != 0 { 1 } else { 0 });
                write!(
                    out,
                    " {}:{}:{}:{}:{}:{}:{}:{}",
                    o.flags,
                    o.extra[IOBJ_EX_FLAGS],
                    o.linewidth,
                    o.linewidth2,
                    o.pdrawsize,
                    o.symbol,
                    o.symsize,
                    o.symflags
                )
                .unwrap();
            }
            writeln!(out).unwrap();
            imod_object_set_color(&mut o, 0.25, 0.5, 0.75);
            writeln!(
                out,
                "{tag} setcolor {} {} {}",
                fb(o.red),
                fb(o.green),
                fb(o.blue)
            )
            .unwrap();
        }

        let mut o = imod_object_new().unwrap();
        report(&mut out, "E", &mut o);
        writeln!(
            out,
            "E sort {} sortsurf {}",
            imod_object_sort(&mut o),
            imod_object_sort_by_surf(&mut o, 1)
        )
        .unwrap();
        imod_object_clean_surf(&mut o);
        writeln!(out, "E cleansurf {}", o.surfsize).unwrap();

        let mut sd = 999u32;
        let mut o = build(&mut sd, 0, 1, 4, 0, 0);
        writeln!(out, "S sort {}", imod_object_sort(&mut o)).unwrap();
        report(&mut out, "S", &mut o);

        let mut sd = 555u32;
        let mut o = build(&mut sd, 0, 3, 0, 0, 0);
        writeln!(out, "Z sort {}", imod_object_sort(&mut o)).unwrap();
        report(&mut out, "Z", &mut o);
        dumpconts(&mut out, "Z", &o);

        /* Mesh-only object.  `imodMeshNew`/`imodMeshAddVert`/`imodMeshAddIndex`
        live in the untranslated `imesh.c`, so the mesh is built directly. */
        let mut o = imod_object_new().unwrap();
        let mut ms = Imesh::default();
        for i in 0..6 {
            ms.vert.push(Ipoint {
                x: (i * 3 % 7) as f32,
                y: (i * 5 % 11) as f32 - 2.0f32,
                z: (i % 3) as f32 * 1.5f32,
            });
        }
        ms.list.push(IMOD_MESH_BGNPOLYNORM);
        for i in 0..3 {
            ms.list.push(0);
            ms.list.push(i + 3);
        }
        ms.list.push(IMOD_MESH_ENDPOLY);
        ms.list.push(-1);
        ms.surf = 5;
        writeln!(out, "M add {}", imod_object_add_mesh(&mut o, ms)).unwrap();
        report(&mut out, "M", &mut o);
        let mut ms = Imesh::default();
        ms.list.push(IMOD_MESH_BGNPOLYNORM2);
        for i in 0..3 {
            ms.list.push(i);
        }
        ms.list.push(IMOD_MESH_ENDPOLY);
        ms.list.push(-1);
        for i in 0..4 {
            ms.vert.push(Ipoint {
                x: 10.0f32 + i as f32,
                y: -3.0f32 * i as f32,
                z: 2.0f32 * i as f32,
            });
        }
        ms.surf = 2;
        writeln!(out, "M add {}", imod_object_add_mesh(&mut o, ms)).unwrap();
        report(&mut out, "M", &mut o);
        imod_object_clean_surf(&mut o);
        writeln!(out, "M cleansurf {}", o.surfsize).unwrap();

        let mut o = imod_object_new().unwrap();
        let name = |o: &Iobj| {
            let bytes: Vec<u8> = o
                .name
                .iter()
                .take_while(|c| **c != 0)
                .map(|c| *c as u8)
                .collect();
            String::from_utf8(bytes).unwrap()
        };
        writeln!(
            out,
            "N set {} [{}]",
            imod_object_set_name(&mut o, b"short name"),
            name(&o)
        )
        .unwrap();
        let nm: Vec<u8> = (0..150).map(|i| b'a' + (i % 26) as u8).collect();
        writeln!(
            out,
            "N set {} [{}]",
            imod_object_set_name(&mut o, &nm),
            name(&o)
        )
        .unwrap();
        writeln!(
            out,
            "N set {} [{}]",
            imod_object_set_name(&mut o, b""),
            name(&o)
        )
        .unwrap();
        out
    }

    /// Differential against the pinned C `iobj.c`: every line of the reference
    /// driver's output must be reproduced exactly.
    #[test]
    fn iobj_matches_native_driver() {
        let mine = driver();
        if std::env::var("IOBJ_DUMP").is_ok() {
            std::fs::write("/tmp/iobj_rust.txt", &mine).unwrap();
        }
        for (line, (a, b)) in mine.lines().zip(NATIVE.lines()).enumerate() {
            assert_eq!(a, b, "line {}", line + 1);
        }
        assert_eq!(mine.lines().count(), NATIVE.lines().count());
    }
}

/// Original: `imodObjectSortSurf` (`iobj.c:645`).
///
/// `starts[poly]` is a `int *` into the mesh index list in the C; here it is
/// the equivalent offset into `obj.mesh[me].list`, since this translation
/// carries the list as a `Vec<i32>` throughout.
///
/// Deviation: in the contourless branch the C leaves `surfMeshes`/`newSurfs`
/// holding the previous iteration's values when a mesh at the wanted
/// resolution has no vertices or list, then reads and re-frees the pointer it
/// already freed.  That is undefined behaviour with no reproducible result, so
/// `surf_meshes` is taken (freed) at the end of each iteration and such an
/// iteration contributes nothing.
pub fn imod_object_sort_surf(obj: &mut Iobj) -> i32 {
    let npoly: i32;
    let mut ninpoly: i32;
    let mut iwork: i32;
    let mut nwork: i32;
    let mut found: i32;
    let mut last_surf: i32 = 0;
    let mut new_surfs: i32 = 0;
    let mut error: i32 = 0;
    let mut refvert: i32;
    let mut work: i32;
    let mut nsurfs: i32;
    let mut ind: i32;
    let mut resol: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;
    let mut new_meshes: Vec<Imesh> = Vec::new();
    let mut zmin: f32;
    let mut zmax: f32;
    let mut ptx: f32;
    let mut pty: f32;
    let mut ptz: f32;

    crate::imod::libimod::imesh::imod_mesh_nearest_res(
        &obj.mesh,
        obj.mesh.len() as i32,
        0,
        &mut resol,
    );

    /* If there are no contours, sort the meshes themselves */
    if obj.cont.is_empty() {
        for me in 0..obj.mesh.len() {
            last_surf = 0;
            let mut surf_meshes: Option<Vec<Imesh>> = None;
            if crate::imod::libimod::imesh::imesh_resol(obj.mesh[me].flag) == resol {
                if !obj.mesh[me].vert.is_empty() && !obj.mesh[me].list.is_empty() {
                    /* For non-empty mesh, sort and adjust surface numbers up */
                    let sorted = crate::imod::libimod::imesh::imesh_sort_surfaces(
                        Some(&obj.mesh[me]),
                        &mut new_surfs,
                        &mut error,
                    );

                    let mut sorted = match sorted {
                        Some(sorted) => sorted,
                        None => return error,
                    };
                    for i in 0..new_surfs as usize {
                        sorted[i].surf += last_surf as i16;
                    }
                    last_surf += new_surfs;
                    surf_meshes = Some(sorted);
                }
            } else {
                /* Just copy a different resolution mesh */
                let dup = match crate::imod::libimod::imesh::imod_mesh_dup(Some(&obj.mesh[me])) {
                    Some(dup) => dup,
                    None => return 2,
                };
                surf_meshes = Some(vec![dup]);
                new_surfs = 1;
            }

            /* add the new meshes to the collection and free their structures */
            if let Some(surf_meshes) = surf_meshes {
                for i in 0..new_surfs as usize {
                    if i >= surf_meshes.len() {
                        break;
                    }
                    if crate::imod::libimod::imesh::imodel_mesh_add(
                        Some(&surf_meshes[i]),
                        &mut new_meshes,
                    ) != 0
                    {
                        return 2;
                    }
                }
            }
        }

        /* Success.  Replace the object's mesh with the new collection */
        let size = obj.mesh.len() as i32;
        crate::imod::libimod::imesh::imod_meshes_delete(Some(std::mem::take(&mut obj.mesh)), size);
        obj.mesh = new_meshes;
        obj.surfsize = last_surf;
        return 0;
    }

    /* Count polygons in all meshes based on start flags */

    let mut count = 0;
    for me in 0..obj.mesh.len() {
        if crate::imod::libimod::imesh::imesh_resol(obj.mesh[me].flag) == resol {
            for i in 0..obj.mesh[me].list.len() {
                let listp = obj.mesh[me].list[i];
                if listp == crate::imod::libimod::imesh::IMOD_MESH_BGNPOLYNORM
                    || listp == crate::imod::libimod::imesh::IMOD_MESH_BGNPOLYNORM2
                {
                    count += 1;
                }
            }
        }
    }
    npoly = count;

    if npoly == 0 {
        return 1;
    }

    /* Get arrays for Z values and surface assignment for each polygon, which
    mesh it is in, address of start of polygons, for number of
    vertices in the polygon, and for list of polys to work on */

    let mut zmins: Vec<f32> = vec![0.; npoly as usize];
    let mut zmaxs: Vec<f32> = vec![0.; npoly as usize];
    let mut surfs: Vec<i32> = vec![0; npoly as usize];
    let mut meshes: Vec<usize> = vec![0; npoly as usize];
    let mut starts: Vec<usize> = vec![0; npoly as usize];
    let mut nverts: Vec<i32> = vec![0; npoly as usize];
    let mut towork: Vec<i32> = vec![0; npoly as usize];
    let mut lincs: Vec<i32> = vec![0; npoly as usize + 2];

    /* Find min and max Z values of each polygon, and collect other info */

    let mut poly: i32 = 0;
    for me in 0..obj.mesh.len() {
        if crate::imod::libimod::imesh::imesh_resol(obj.mesh[me].flag) != resol {
            continue;
        }
        let mut i: i32 = 0;
        while i < obj.mesh[me].list.len() as i32 {
            let start_code = obj.mesh[me].list[i as usize];
            i += 1;
            if crate::imod::libimod::imesh::imod_mesh_poly_norm_factors(
                start_code,
                &mut lincs[poly as usize],
                &mut vert_base,
                &mut norm_add,
            ) != 0
            {
                zmin = obj.mesh[me].vert[obj.mesh[me].list[(i + vert_base) as usize] as usize].z;
                zmax = zmin;
                surfs[poly as usize] = 0;
                meshes[poly as usize] = me;
                starts[poly as usize] = (i + vert_base) as usize; /* index of first vert ind */
                ninpoly = 0;
                while obj.mesh[me].list[i as usize]
                    != crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY
                {
                    ind = obj.mesh[me].list[(i + vert_base) as usize];
                    i += lincs[poly as usize];
                    if obj.mesh[me].vert[ind as usize].z < zmin {
                        zmin = obj.mesh[me].vert[ind as usize].z;
                    }
                    if obj.mesh[me].vert[ind as usize].z > zmax {
                        zmax = obj.mesh[me].vert[ind as usize].z;
                    }
                    ninpoly += 1;
                }
                nverts[poly as usize] = ninpoly;
                zmins[poly as usize] = zmin;
                zmaxs[poly as usize] = zmax;
                poly += 1;
            }
        }
    }

    /* loop through all polygons, looking for next one that's not assigned
    yet */

    nsurfs = 0;
    for poly in 0..npoly {
        if surfs[poly as usize] == 0 {
            nsurfs += 1;
            surfs[poly as usize] = nsurfs;
            towork[0] = poly;
            nwork = 1;
            iwork = 0;
            while iwork < nwork {
                /* To work on a polygon, scan through all the rest that are not
                assigned yet, find ones that overlap in Z */

                work = towork[iwork as usize];
                for scan in 0..npoly {
                    if surfs[scan as usize] == 0
                        && zmins[work as usize] <= zmaxs[scan as usize]
                        && zmins[scan as usize] <= zmaxs[work as usize]
                    {
                        /* Look for common vertices between the polygons */

                        let mut refp = starts[work as usize];
                        found = 0;
                        let mut i = 0;
                        while i < nverts[work as usize] && found == 0 {
                            refvert = obj.mesh[meshes[work as usize]].list[refp];
                            refp += lincs[work as usize] as usize;
                            let mut scanp = starts[scan as usize];
                            for _j in 0..nverts[scan as usize] {
                                if obj.mesh[meshes[scan as usize]].list[scanp] == refvert {
                                    found = 1;
                                    break;
                                }
                                scanp += lincs[scan as usize] as usize;
                            }
                            i += 1;
                        }

                        /* If found a match, then add the scan polygon to work list as
                        well as assigning it to this surface */

                        if found != 0 {
                            towork[nwork as usize] = scan;
                            nwork += 1;
                            surfs[scan as usize] = nsurfs;
                        }
                    }
                }
                iwork += 1;
            }
        }
    }

    /* Now go through contours, looking for polygons with a matching vertex */

    obj.surfsize = 0;
    for co in 0..obj.cont.len() {
        found = 0;
        obj.cont[co].surf = 0;
        let mut pt = 0;
        while pt < obj.cont[co].pts.len() && found == 0 {
            ptx = obj.cont[co].pts[pt].x;
            pty = obj.cont[co].pts[pt].y;
            ptz = obj.cont[co].pts[pt].z;
            let mut poly = 0;
            while poly < npoly && found == 0 {
                if ptz >= zmins[poly as usize] && ptz <= zmaxs[poly as usize] {
                    let mut scanp = starts[poly as usize];
                    for _j in 0..nverts[poly as usize] {
                        ind = obj.mesh[meshes[poly as usize]].list[scanp];
                        let vert = obj.mesh[meshes[poly as usize]].vert[ind as usize];
                        if vert.x == ptx && vert.y == pty && vert.z == ptz {
                            found = 1;
                            obj.cont[co].surf = surfs[poly as usize];
                            if surfs[poly as usize] > obj.surfsize {
                                obj.surfsize = surfs[poly as usize];
                            }
                            break;
                        }
                        scanp += lincs[poly as usize] as usize;
                    }
                }
                poly += 1;
            }
            pt += 1;
        }
    }
    imod_object_clean_surf(obj);
    0
}

/// `imodObjectGetLabel` (`iobj.c:1013`).
pub fn imod_object_get_label(
    obj: Option<&crate::imod::libimod::imodel::Iobj>,
) -> Option<&crate::imod::libimod::ilabel::Ilabel> {
    obj?.label.as_ref()
}

/// `imodObjectNewLabel` (`iobj.c:1025`).  Frees any existing label first.
pub fn imod_object_new_label(
    obj: Option<&mut crate::imod::libimod::imodel::Iobj>,
) -> Option<&mut crate::imod::libimod::ilabel::Ilabel> {
    let obj = obj?;
    crate::imod::libimod::ilabel::imod_label_delete(obj.label.take());
    obj.label = Some(crate::imod::libimod::ilabel::imod_label_new());
    obj.label.as_mut()
}
