//! View or camera handling functions, from `IMOD/libimod/iview.c` and
//! `IMOD/include/iview.h`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::imodel::{
    IMOD_CLIPSIZE, IMOD_ERROR_CORRUPT, IMOD_ERROR_READ, IMOD_ERROR_WRITE, Iclip_planes, Imod, Iobj,
    Iobjview, Ipoint, Iref_image, Iview,
};
use super::imodel_files::{
    IMODF_HAS_MESH_THICK, IMODF_MAT1_IS_BYTES, imod_get_float, imod_get_int, imod_put_bytes,
    imod_put_float, imod_put_floats, imod_put_int, imod_put_ints, imod_put_scaled_points,
};
use super::iplane::imod_clips_initialize;

/// Original: `VIEW_STRSIZE` (`imodel.h:36`).
pub const VIEW_STRSIZE: usize = 32;

/// Original: `VIEW_WORLD_LIGHT` (`imodel.h:203`).
pub const VIEW_WORLD_LIGHT: u32 = 1 << 1;
/// Original: `VIEW_WORLD_CLIP_IMAGE` (`imodel.h:220`).
pub const VIEW_WORLD_CLIP_IMAGE: u32 = 1 << 14;

/// Original: `ID_VIEW` (`imodel.h:68`).
pub const ID_VIEW: u32 = u32::from_be_bytes(*b"VIEW");
/// Original: `ID_MCLP` (`imodel.h:77`).
pub const ID_MCLP: u32 = u32::from_be_bytes(*b"MCLP");
/// Original: `ID_IMNX` (`imodel.h:83`).
pub const ID_IMNX: u32 = u32::from_be_bytes(*b"MINX");
/// Original: `SIZE_IMNX` (`imodel.h:84`).
pub const SIZE_IMNX: i32 = 72;

/// Original: `BYTES_PER_OBJVIEW` (`iview.c:143`), `43 + 24 * IMOD_CLIPSIZE`.
pub const BYTES_PER_OBJVIEW: i32 = 43 + 24 * IMOD_CLIPSIZE as i32;

/// Original: `imodViewNew` (`iview.c:22`).
///
/// The C `malloc`s `size` uninitialized `Iview` structures; every caller either
/// runs `imodViewDefault` on the new element or overwrites the whole array, so
/// this returns default-constructed elements instead of uninitialized memory.
pub fn imod_view_new(size: i32) -> Option<Vec<Iview>> {
    if size < 0 {
        return None;
    }
    let mut view: Vec<Iview> = Vec::with_capacity(size as usize);
    for _ in 0..size {
        view.push(Iview::default());
    }
    Some(view)
}

/// Original: `imodViewDelete` (`iview.c:32`).
pub fn imod_view_delete(vw: Option<Vec<Iview>>) {
    drop(vw);
}

/// Original: `imodViewDefault` (`iview.c:42`).
///
/// Note that this is not the same as the derived `Iview::default()` in
/// `imodel.rs`: `imodClipsInitialize` leaves every clip normal at
/// (0, 0, -1), while the derived value leaves them at (0, 0, 0).
pub fn imod_view_default(vw: &mut Iview) {
    vw.fovy = 0.0;
    vw.rad = 1.0;
    vw.aspect = 1.0;
    vw.cnear = 0.0;
    vw.cfar = 1.0;
    vw.rot.x = 0.0;
    vw.rot.y = 0.0;
    vw.rot.z = 0.0;
    vw.trans.x = 0.0;
    vw.trans.y = 0.0;
    vw.trans.z = 0.0;
    vw.scale.x = 1.0;
    vw.scale.y = 1.0;
    vw.scale.z = 1.0;

    vw.world = VIEW_WORLD_LIGHT;
    for i in 0..16 {
        vw.mat[i] = 0.0;
    }
    vw.mat[0] = 1.0;
    vw.mat[5] = 1.0;
    vw.mat[10] = 1.0;
    vw.mat[15] = 1.0;
    for i in 0..VIEW_STRSIZE {
        vw.label[i] = 0x00;
    }
    vw.lightx = 0.0;
    vw.lighty = 0.0;
    vw.plax = 5.0;
    vw.dcstart = 0.0;
    vw.dcend = 1.0;
    vw.objview = Vec::new();

    imod_clips_initialize(&mut vw.clips);
}

/// Original: `imodViewDefaultScale` (`iview.c:113`).
pub fn imod_view_default_scale(imod: &Imod, vw: &mut Iview, image_max: &Ipoint, bin_scale: f32) {
    let mut maxp = Ipoint::default();
    let mut minp = Ipoint::default();

    super::imodel::imod_get_bounding_box(imod, &mut minp, &mut maxp);

    /* If there is no extent and an image maximum has been supplied,
    then use image box as limiting coordinates */
    if maxp.x == minp.x && maxp.y == minp.y && maxp.z == minp.z && image_max.x != 0. {
        minp.x = 0.;
        minp.y = 0.;
        minp.z = 0.;
        maxp.x = image_max.x;
        maxp.y = image_max.y;
        maxp.z = image_max.z;
    }

    maxp.z *= bin_scale * imod.zscale;
    minp.z *= bin_scale * imod.zscale;

    vw.trans.x = (minp.x + maxp.x) * -0.5;
    vw.trans.y = (minp.y + maxp.y) * -0.5;
    vw.trans.z = (minp.z + maxp.z) * -0.5;
    if imod.zscale != 0. {
        vw.trans.z /= bin_scale * imod.zscale;
    }

    maxp.x -= minp.x;
    maxp.y -= minp.y;
    maxp.z -= minp.z;
    vw.rad =
        ((((maxp.x * maxp.x) + (maxp.y * maxp.y) + (maxp.z * maxp.z)) as f64).sqrt() * 0.5) as f32;

    /* DNM: Make this happen once here and let all callers rely on it */
    vw.rad = (vw.rad as f64 * 0.85) as f32;

    /* If the rad is zero, make it one to avoid division by 0 */
    if vw.rad == 0. {
        vw.rad = 1.;
    }
}

/// Original: `imodViewModelDefault` (`iview.c:129`).
pub fn imod_view_model_default(imod: &Imod, vw: &mut Iview, image_max: &Ipoint) {
    imod_view_default(vw);
    imod_view_default_scale(imod, vw, image_max, 1.);
}

/* For backward compatibility, need to add all new elements at end of write
so the array elements after the first have to be written at end */
/// Original: `imodViewWrite` (`iview.c:149`).
///
/// The object-view record is written in the source's order: flags, colour,
/// pdrawsize, three line bytes, the backward-compatible clip count byte, three
/// clip bytes, clip plane 0's normal and point, the material bytes, and only
/// then normals 1..IMOD_CLIPSIZE-1 followed by points 1..IMOD_CLIPSIZE-1.
/// Clip points are scaled by [scale] and normals by its inverse.  The model
/// clip chunk likewise writes all `count` normals and then all `count` points,
/// not normal/point pairs.
pub fn imod_view_write(vw: &Iview, fout: &mut ImodFile, scale: &Ipoint) -> i32 {
    let mut id: u32;
    let nbwrite: i32;
    let mut clip_out: u8;
    let mut norm_scale = Ipoint::default();

    id = ID_VIEW;
    if imod_put_int(fout, id as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }
    id = 176;
    nbwrite = vw.objview.len() as i32 * BYTES_PER_OBJVIEW;
    if !vw.objview.is_empty() {
        id = id.wrapping_add(8).wrapping_add(nbwrite as u32);
    }
    if imod_put_int(fout, id as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }
    /* imodPutFloats(fout, &vw->fovy, 30): the 30 contiguous floats from
    {fovy} through the 16 elements of {mat}. */
    let mut head = [0f32; 30];
    head[0] = vw.fovy;
    head[1] = vw.rad;
    head[2] = vw.aspect;
    head[3] = vw.cnear;
    head[4] = vw.cfar;
    head[5] = vw.rot.x;
    head[6] = vw.rot.y;
    head[7] = vw.rot.z;
    head[8] = vw.trans.x;
    head[9] = vw.trans.y;
    head[10] = vw.trans.z;
    head[11] = vw.scale.x;
    head[12] = vw.scale.y;
    head[13] = vw.scale.z;
    head[14..30].copy_from_slice(&vw.mat);
    if imod_put_floats(fout, &head, 30).is_err() {
        return IMOD_ERROR_WRITE;
    }
    if imod_put_int(fout, vw.world as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }
    if imod_put_bytes(fout, &vw.label, VIEW_STRSIZE as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }
    /* imodPutFloats(fout, &vw->dcstart, 5) */
    let tail = [vw.dcstart, vw.dcend, vw.lightx, vw.lighty, vw.plax];
    if imod_put_floats(fout, &tail, 5).is_err() {
        return IMOD_ERROR_WRITE;
    }

    norm_scale.x = 1. / scale.x;
    norm_scale.y = 1. / scale.y;
    norm_scale.z = 1. / scale.z;

    if !vw.objview.is_empty() {
        if imod_put_int(fout, vw.objview.len() as i32).is_err() {
            return IMOD_ERROR_WRITE;
        }
        if imod_put_int(fout, nbwrite).is_err() {
            return IMOD_ERROR_WRITE;
        }
        for i in 0..vw.objview.len() {
            let ov = &vw.objview[i];
            let clips = &ov.clips;
            if imod_put_ints(fout, &[ov.flags as i32], 1).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_floats(fout, &[ov.red, ov.green, ov.blue], 3).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_ints(fout, &[ov.pdrawsize], 1).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_bytes(fout, &[ov.linewidth, ov.linesty, ov.trans], 3).is_err() {
                return IMOD_ERROR_WRITE;
            }

            /* For backward compatibility, if there is one clip plane and it is
            off, set clip to 0 */
            clip_out = clips.count;
            if clip_out == 1 && (clips.flags & 1) == 0 {
                clip_out = 0;
            }
            if imod_put_bytes(fout, &[clip_out], 1).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_bytes(fout, &[clips.flags, clips.trans, clips.plane], 3).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_scaled_points(fout, &clips.normal[0..1], 1, &norm_scale).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_scaled_points(fout, &clips.point[0..1], 1, scale).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_bytes(
                fout,
                &[ov.ambient, ov.diffuse, ov.specular, ov.shininess],
                4,
            )
            .is_err()
            {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_bytes(
                fout,
                &[ov.fillred, ov.fillgreen, ov.fillblue, ov.quality],
                4,
            )
            .is_err()
            {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_ints(fout, &[ov.mat2 as i32], 1).is_err() {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_bytes(
                fout,
                &[ov.valblack, ov.valwhite, ov.matflags2, ov.mesh_thickness],
                4,
            )
            .is_err()
            {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_scaled_points(
                fout,
                &clips.normal[1..IMOD_CLIPSIZE],
                IMOD_CLIPSIZE as i32 - 1,
                &norm_scale,
            )
            .is_err()
            {
                return IMOD_ERROR_WRITE;
            }
            if imod_put_scaled_points(
                fout,
                &clips.point[1..IMOD_CLIPSIZE],
                IMOD_CLIPSIZE as i32 - 1,
                scale,
            )
            .is_err()
            {
                return IMOD_ERROR_WRITE;
            }
        }
    }

    /* Write the clip plane chunk */
    if vw.clips.count != 0 {
        id = ID_MCLP;
        if imod_put_int(fout, id as i32).is_err() {
            return IMOD_ERROR_WRITE;
        }
        id = 4 + 24 * vw.clips.count as u32;
        if imod_put_int(fout, id as i32).is_err() {
            return IMOD_ERROR_WRITE;
        }
        if imod_put_bytes(
            fout,
            &[
                vw.clips.count,
                vw.clips.flags,
                vw.clips.trans,
                vw.clips.plane,
            ],
            4,
        )
        .is_err()
        {
            return IMOD_ERROR_WRITE;
        }
        if imod_put_scaled_points(
            fout,
            &vw.clips.normal[..vw.clips.count as usize],
            vw.clips.count as i32,
            &norm_scale,
        )
        .is_err()
        {
            return IMOD_ERROR_WRITE;
        }
        if imod_put_scaled_points(
            fout,
            &vw.clips.point[..vw.clips.count as usize],
            vw.clips.count as i32,
            scale,
        )
        .is_err()
        {
            return IMOD_ERROR_WRITE;
        }
    }

    0
}

/// Original: `imodViewModelNew` (`iview.c:396`).
pub fn imod_view_model_new(imod: &mut Imod) -> i32 {
    let mut nvw = Iview::default();
    imod_view_default(&mut nvw);
    imod.view.push(nvw);
    0
}

/// Original: `imodObjviewToObject` (`iview.c:418`).
pub fn imod_objview_to_object(objview: &Iobjview, obj: &mut Iobj) {
    obj.flags = objview.flags;
    obj.red = objview.red;
    obj.green = objview.green;
    obj.blue = objview.blue;
    obj.pdrawsize = objview.pdrawsize;
    obj.linewidth = objview.linewidth;
    obj.linesty = objview.linesty;
    obj.trans = objview.trans;
    obj.clips = objview.clips.clone();
    /* memcpy (&obj->ambient, &objview->ambient, 4) */
    obj.ambient = objview.ambient;
    obj.diffuse = objview.diffuse;
    obj.specular = objview.specular;
    obj.shininess = objview.shininess;
    /* memcpy (&obj->fillred, &objview->fillred, 12) */
    obj.fillred = objview.fillred;
    obj.fillgreen = objview.fillgreen;
    obj.fillblue = objview.fillblue;
    obj.quality = objview.quality;
    obj.mat2 = objview.mat2;
    obj.valblack = objview.valblack;
    obj.valwhite = objview.valwhite;
    obj.matflags2 = objview.matflags2;
    obj.mesh_thickness = objview.mesh_thickness;
}

/// Original: `imodObjviewFromObject` (`iview.c:438`).
pub fn imod_objview_from_object(obj: &Iobj, objview: &mut Iobjview) {
    objview.flags = obj.flags;
    objview.red = obj.red;
    objview.green = obj.green;
    objview.blue = obj.blue;
    objview.pdrawsize = obj.pdrawsize;
    objview.linewidth = obj.linewidth;
    objview.linesty = obj.linesty;
    objview.trans = obj.trans;
    objview.clips = obj.clips.clone();
    /* memcpy (&objview->ambient, &obj->ambient, 4) */
    objview.ambient = obj.ambient;
    objview.diffuse = obj.diffuse;
    objview.specular = obj.specular;
    objview.shininess = obj.shininess;
    /* memcpy (&objview->fillred, &obj->fillred, 12) */
    objview.fillred = obj.fillred;
    objview.fillgreen = obj.fillgreen;
    objview.fillblue = obj.fillblue;
    objview.quality = obj.quality;
    objview.mat2 = obj.mat2;
    objview.valblack = obj.valblack;
    objview.valwhite = obj.valwhite;
    objview.matflags2 = obj.matflags2;
    objview.mesh_thickness = obj.mesh_thickness;
}

/// Original: `imodViewUse` (`iview.c:458`).
pub fn imod_view_use(imod: &mut Imod) {
    let mut nobj: usize;

    /* First copy the view structure, saving and restoring the object view
    count and pointer from the default view */
    let vw = imod.view[imod.cview as usize].clone();
    let defviewsave = imod.view[0].objview.clone();
    let mut labsav = [0u8; VIEW_STRSIZE];
    labsav.copy_from_slice(&imod.view[0].label);
    imod.view[0] = vw.clone();
    imod.view[0].label.copy_from_slice(&labsav);
    imod.view[0].objview = defviewsave;

    /* Now set object characteristics based on the saved values in the
    current view.  Use no more data than the current # of objects */
    nobj = imod.obj.len();
    if nobj > vw.objview.len() {
        nobj = vw.objview.len();
    }

    for i in 0..nobj {
        imod_objview_to_object(&vw.objview[i], &mut imod.obj[i]);
    }
}

/// Original: `imodViewStore` (`iview.c:490`).
pub fn imod_view_store(imod: &mut Imod, cview: i32) -> i32 {
    /* First delete any existing object view data */
    let mut labsav = [0u8; VIEW_STRSIZE];
    labsav.copy_from_slice(&imod.view[cview as usize].label);
    let defview = imod.view[0].clone();
    imod.view[cview as usize] = defview;
    imod.view[cview as usize].label.copy_from_slice(&labsav);
    imod.view[cview as usize].objview = vec![Iobjview::default(); imod.obj.len()];

    for i in 0..imod.obj.len() {
        let obj = imod.obj[i].clone();
        imod_objview_from_object(&obj, &mut imod.view[cview as usize].objview[i]);
    }
    0
}

/// Original: `imodObjviewComplete` (`iview.c:519`).
pub fn imod_objview_complete(imod: &mut Imod) -> i32 {
    /* Loop through all real views */
    for i in 1..imod.view.len() {
        if imod.view[i].objview.len() < imod.obj.len() {
            /* If there are missing object views, first allocate enough objviews */
            let objsize = imod.obj.len();
            imod.view[i].objview.resize(objsize, Iobjview::default());

            /* Then copy the missing objects into the view */
            for j in 0..objsize {
                let obj = imod.obj[j].clone();
                imod_objview_from_object(&obj, &mut imod.view[i].objview[j]);
            }
        }
    }
    0
}

/// Original: `imodObjviewDelete` (`iview.c:553`).
pub fn imod_objview_delete(imod: &mut Imod, index: i32) {
    for i in 1..imod.view.len() {
        let vw = &mut imod.view[i];

        /* If this set of object views includes this object, shift all of
        the ones above down */
        if index < vw.objview.len() as i32 {
            for j in (index as usize + 1)..vw.objview.len() {
                vw.objview[j - 1] = vw.objview[j].clone();
            }

            /* reduce count, free array if they are all gone */
            let new_len = vw.objview.len() - 1;
            vw.objview.truncate(new_len);
        }
    }
}

/// Original: `imodObjviewsFree` (`iview.c:579`).
pub fn imod_objviews_free(imod: &mut Imod) {
    for i in 1..imod.view.len() {
        let vw = &mut imod.view[i];
        if !vw.objview.is_empty() {
            vw.objview = Vec::new();
        }
    }
}

/* Image File view functions. */

/// Original: `imodIMNXNew` (`iview.c:648`).
pub fn imod_imnx_new() -> Option<Iref_image> {
    let mut r = Iref_image {
        oscale: Ipoint::default(),
        otrans: Ipoint::default(),
        orot: Ipoint::default(),
        cscale: Ipoint::default(),
        ctrans: Ipoint::default(),
        crot: Ipoint::default(),
    };
    r.cscale.x = 1.;
    r.cscale.y = 1.;
    r.cscale.z = 1.;
    r.ctrans.x = 0.;
    r.ctrans.y = 0.;
    r.ctrans.z = 0.;
    r.crot = r.ctrans;
    r.oscale = r.cscale;
    r.otrans = r.ctrans;
    r.orot = r.crot;
    Some(r)
}

/// The `Iclip_planes` copy the C does with plain struct assignment.
#[allow(dead_code)]
fn clips_assign(to: &mut Iclip_planes, from: &Iclip_planes) {
    *to = from.clone();
}

/// Original: `imodViewModelRead` (`iview.c:254`).
///
/// `bytesRead` is not tracked separately here: the source uses it only to
/// compute the final `fseek(fin, lbuf - bytesRead, SEEK_CUR)`, which lands on
/// the byte after the chunk, and this seeks there directly.
pub fn imod_view_model_read(imod: &mut Imod, file: &mut ImodFile) -> Result<(), i32> {
    let lbuf = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;

    /* only current value selected. */
    if lbuf == 4 {
        imod.cview = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        return Ok(());
    }

    let start = file.stream_position().map_err(|_| IMOD_ERROR_READ)?;
    let mut vw = Iview::default();
    vw.objview.clear();

    /* Need to initialize clip planes for view and for all object views because
    they may not be read in (for view), or ones past the first may not be
    read in (for object views) */
    crate::imod::libimod::iplane::imod_clips_initialize(&mut vw.clips);

    if lbuf >= 56 {
        /* imodGetFloats(fin, &vw->fovy, 14) */
        vw.fovy = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.rad = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.aspect = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.cnear = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.cfar = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        for point in [&mut vw.rot, &mut vw.trans, &mut vw.scale] {
            point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        }
    }
    if lbuf >= 156 {
        for value in &mut vw.mat {
            *value = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        }
        vw.world = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        file.read_exact(&mut vw.label)
            .map_err(|_| IMOD_ERROR_READ)?;
    }
    if lbuf >= 176 {
        vw.dcstart = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.dcend = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.lightx = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.lighty = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        vw.plax = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    }

    if lbuf >= 180 {
        let objvsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        let bytes_objv = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;

        if objvsize > 0 {
            let bytes_missing =
                crate::imod::libimod::iview::BYTES_PER_OBJVIEW - bytes_objv / objvsize;
            for _ in 0..objvsize {
                let mut ov = Iobjview::default();
                crate::imod::libimod::iplane::imod_clips_initialize(&mut ov.clips);
                ov.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
                ov.red = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.green = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.blue = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.pdrawsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
                let mut head = [0; 3];
                file.read_exact(&mut head).map_err(|_| IMOD_ERROR_READ)?;
                ov.linewidth = head[0];
                ov.linesty = head[1];
                ov.trans = head[2];

                /* Get clip parameters and first clip plane, then fix the count */
                let mut clip = [0; 4];
                file.read_exact(&mut clip).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.count = clip[0];
                ov.clips.flags = clip[1];
                ov.clips.trans = clip[2];
                ov.clips.plane = clip[3];
                ov.clips.normal[0].x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.normal[0].y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.normal[0].z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.point[0].x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.point[0].y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                ov.clips.point[0].z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;

                crate::imod::libimod::iplane::imod_clips_fix_count(&mut ov.clips, imod.flags);

                let mut material = [0; 4];
                file.read_exact(&mut material)
                    .map_err(|_| IMOD_ERROR_READ)?;
                ov.ambient = material[0];
                ov.diffuse = material[1];
                ov.specular = material[2];
                ov.shininess = material[3];

                /* DNM 9/4/03: read mat1 and mat3 as bytes, or as ints
                for old model */
                if imod.flags & IMODF_MAT1_IS_BYTES != 0 {
                    file.read_exact(&mut material)
                        .map_err(|_| IMOD_ERROR_READ)?;
                    ov.fillred = material[0];
                    ov.fillgreen = material[1];
                    ov.fillblue = material[2];
                    ov.quality = material[3];
                    ov.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
                    file.read_exact(&mut material)
                        .map_err(|_| IMOD_ERROR_READ)?;
                    ov.valblack = material[0];
                    ov.valwhite = material[1];
                    ov.matflags2 = material[2];
                    ov.mesh_thickness = material[3];
                } else {
                    /* imodGetInts(fin, (int *)&ov->fillred, 3): mat1, mat2 and
                    mat3 as three ints over the byte members. */
                    let mat1 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
                    ov.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
                    let mat3 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
                    ov.fillred = mat1 as u8;
                    ov.fillgreen = (mat1 >> 8) as u8;
                    ov.fillblue = (mat1 >> 16) as u8;
                    ov.quality = (mat1 >> 24) as u8;
                    ov.valblack = mat3 as u8;
                    ov.valwhite = (mat3 >> 8) as u8;
                    ov.matflags2 = (mat3 >> 16) as u8;
                    ov.mesh_thickness = (mat3 >> 24) as u8;
                }
                if imod.flags & IMODF_HAS_MESH_THICK == 0 {
                    ov.mesh_thickness = 0;
                }

                /* If more elements are added in future, will need to test
                bytesMissing before reading them in.  For each one, test against
                total bytes in elements beyond this one */

                /* Read additional clip planes */
                if bytes_missing <= 0 {
                    for i in 1..IMOD_CLIPSIZE {
                        ov.clips.normal[i].x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        ov.clips.normal[i].y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        ov.clips.normal[i].z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                    }
                    for i in 1..IMOD_CLIPSIZE {
                        ov.clips.point[i].x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        ov.clips.point[i].y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        ov.clips.point[i].z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                    }
                }

                /* But if this code tries to read a future file, it needs
                to skip over the rest of the data per object */
                if bytes_missing < 0 {
                    file.seek(SeekFrom::Current(-bytes_missing as i64))
                        .map_err(|_| IMOD_ERROR_READ)?;
                }

                vw.objview.push(ov);
            }
        }
    }

    let here = file.stream_position().map_err(|_| IMOD_ERROR_READ)?;
    if start + lbuf as u64 > here {
        file.seek(SeekFrom::Start(start + lbuf as u64))
            .map_err(|_| IMOD_ERROR_READ)?;
    }

    imod.view.push(vw);
    Ok(())
}
/// Original: `imodViewClipRead` (`iview.c:386`).
///
/// Delegates to `imodClipsRead` (`iplane.c:166`), which sizes the read from
/// the chunk length and reads all the normals before all the points.
pub fn imod_view_clip_read(imod: &mut Imod, file: &mut ImodFile) -> Result<(), i32> {
    let last = match imod.view.len().checked_sub(1) {
        Some(last) => last,
        None => return Err(IMOD_ERROR_CORRUPT),
    };
    crate::imod::libimod::iplane::imod_clips_read(&mut imod.view[last].clips, file);
    Ok(())
}
/// Original: `imodIMNXRead` (`iview.c:598`).
pub fn imod_imnx_read(imod: &mut Imod, file: &mut ImodFile) -> Result<(), i32> {
    if imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? != 72 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let mut reference = Iref_image::default();
    for point in [
        &mut reference.oscale,
        &mut reference.otrans,
        &mut reference.orot,
        &mut reference.cscale,
        &mut reference.ctrans,
        &mut reference.crot,
    ] {
        point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    }
    imod.ref_image = Some(reference);
    Ok(())
}
/// Original: `imodViewModelWrite` (`iview.c:230`).
///
/// Each view is written by `imodViewWrite` (`iview.c:149`), translated in
/// `iview.rs`. The source passes `(mod->xybin, mod->xybin, mod->zbin)`.
pub fn imod_view_model_write(imod: &Imod, file: &mut ImodFile) -> Result<(), i32> {
    if imod.view.len() < 2 {
        return Ok(());
    }
    imod_put_int(file, ID_VIEW as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 4).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, imod.cview).map_err(|_| IMOD_ERROR_WRITE)?;

    let scale = Ipoint {
        x: imod.xybin as f32,
        y: imod.xybin as f32,
        z: imod.zbin as f32,
    };
    for i in 1..imod.view.len() {
        crate::imod::libimod::iview::imod_view_write(&imod.view[i], file, &scale);
    }
    Ok(())
}
/// Original: `imodIMNXWrite` (`iview.c:625`).
pub fn imod_imnx_write(imod: &Imod, file: &mut ImodFile) -> Result<(), i32> {
    let Some(reference) = imod.ref_image else {
        return Ok(());
    };
    imod_put_int(file, ID_IMNX as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 72).map_err(|_| IMOD_ERROR_WRITE)?;
    for point in [
        reference.oscale,
        reference.otrans,
        reference.orot,
        reference.cscale,
        reference.ctrans,
        reference.crot,
    ] {
        for value in [point.x, point.y, point.z] {
            imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    Ok(())
}
