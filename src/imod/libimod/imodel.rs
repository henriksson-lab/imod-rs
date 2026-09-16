//! Data declarations from `IMOD/include/imodel.h`, `iobj.h`, `icont.h`, and
//! `imesh.h` used by the translated command-line model utilities.
//!
//! This is deliberately a direct data representation: model-file decoding is
//! implemented by the source-mapped `imodel_files` unit, not by a substitute
//! serialization format.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::robuststat::rs_sort_ints;
use crate::imod::libiimod::mrcfiles::{LoadInfo, MrcHeader};
use crate::imod::libimod::icont::{imod_contour_copy, imod_contours_delete, imod_contours_new};
use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, B3D_Z, Imat, imod_mat_copy, imod_mat_delete, imod_mat_inverse, imod_mat_mult,
    imod_mat_new, imod_mat_rot, imod_mat_scale, imod_mat_trans, imod_mat_transform,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_WILD, imod_object_checksum, imod_object_clean_surf, imod_objects_delete,
};
use crate::imod::libimod::iplane::{imod_clips_initialize, imod_clips_trans};
use crate::imod::libimod::imesh::MeshParams;
use crate::imod::libimod::ipoint::imod_point_normalize;
use crate::imod::libimod::istore::{istore_checksum, istore_delete_cont_surf};
use crate::imod::libimod::objgroup::{obj_group_list_checksum, obj_group_list_delete};

/// Original: `IMOD_STRSIZE` (`imodel.h:30`).
pub const IMOD_STRSIZE: usize = 128;

/// Original: `IOBJ_STRSIZE` (`imodel.h:32`).
pub const IOBJ_STRSIZE: usize = 64;

/// Original: `IMOD_CLIPSIZE` (`imodel.h:34`).
pub const IMOD_CLIPSIZE: usize = 6;

/// Original: `SIZE_CLIP` (`imodel.h:78`).
pub const SIZE_CLIP: i32 = 28;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Iplane {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

/// Original: `Iindex` (`include/imodel.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Iindex {
    pub object: i32,
    pub contour: i32,
    pub point: i32,
}

/// Original: `IobjGroup` (`include/objgroup.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Iobj_group {
    pub obj_list: Vec<i32>,
    pub name: [u8; 32],
}

/// Original: `IclipPlanes` (`include/imodel.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct Iclip_planes {
    pub count: u8,
    pub flags: u8,
    pub trans: u8,
    pub plane: u8,
    pub normal: [Ipoint; IMOD_CLIPSIZE],
    pub point: [Ipoint; IMOD_CLIPSIZE],
}

impl Default for Iclip_planes {
    fn default() -> Self {
        Self {
            count: 0,
            flags: 0,
            trans: 0,
            plane: 0,
            // `imodClipsInitialize` gives every unused plane its native
            // default -Z normal.  In particular, the 3dmod clipping editor
            // uses this sentinel to recognize a plane being enabled for the
            // first time and place it at the model/object midpoint.
            normal: [Ipoint {
                x: 0.,
                y: 0.,
                z: -1.,
            }; IMOD_CLIPSIZE],
            point: [Ipoint::default(); IMOD_CLIPSIZE],
        }
    }
}

/// Original: `Iobjview` (`include/imodel.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Iobjview {
    pub flags: u32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub pdrawsize: i32,
    pub linewidth: u8,
    pub linesty: u8,
    pub trans: u8,
    pub clips: Iclip_planes,
    pub ambient: u8,
    pub diffuse: u8,
    pub specular: u8,
    pub shininess: u8,
    pub fillred: u8,
    pub fillgreen: u8,
    pub fillblue: u8,
    pub quality: u8,
    pub mat2: u32,
    pub valblack: u8,
    pub valwhite: u8,
    pub matflags2: u8,
    pub mesh_thickness: u8,
}

/// Original: `Iview` (`include/imodel.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct Iview {
    pub fovy: f32,
    pub rad: f32,
    pub aspect: f32,
    pub cnear: f32,
    pub cfar: f32,
    pub rot: Ipoint,
    pub trans: Ipoint,
    pub scale: Ipoint,
    pub mat: [f32; 16],
    pub world: u32,
    pub label: [u8; 32],
    pub dcstart: f32,
    pub dcend: f32,
    pub lightx: f32,
    pub lighty: f32,
    pub plax: f32,
    pub clips: Iclip_planes,
    pub objview: Vec<Iobjview>,
}

impl Default for Iview {
    fn default() -> Self {
        let mut mat = [0.; 16];
        mat[0] = 1.;
        mat[5] = 1.;
        mat[10] = 1.;
        mat[15] = 1.;
        Self {
            rad: 1.,
            aspect: 1.,
            cfar: 1.,
            scale: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            mat,
            world: 2,
            plax: 5.,
            dcend: 1.,
            fovy: 0.,
            cnear: 0.,
            rot: Ipoint::default(),
            trans: Ipoint::default(),
            label: [0; 32],
            dcstart: 0.,
            lightx: 0.,
            lighty: 0.,
            clips: Iclip_planes::default(),
            objview: Vec::new(),
        }
    }
}

/// Original: `IrefImage` (`include/imodel.h`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iref_image {
    pub oscale: Ipoint,
    pub otrans: Ipoint,
    pub orot: Ipoint,
    pub cscale: Ipoint,
    pub ctrans: Ipoint,
    pub crot: Ipoint,
}

/// Original: `SlicerAngles` (`include/imodel.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct Slicer_angles {
    pub time: i32,
    pub angles: [f32; 3],
    pub center: Ipoint,
    pub label: [u8; 32],
}
impl Default for Slicer_angles {
    fn default() -> Self {
        Self {
            time: 0,
            angles: [0.; 3],
            center: Ipoint::default(),
            label: [0; 32],
        }
    }
}
impl Default for Iref_image {
    fn default() -> Self {
        Self {
            oscale: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            cscale: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            otrans: Ipoint::default(),
            orot: Ipoint::default(),
            ctrans: Ipoint::default(),
            crot: Ipoint::default(),
        }
    }
}

/// Original: `Mod_Contour` (`include/imodel.h`).
#[derive(Clone, Debug, Default)]
pub struct Icont {
    pub temp_val: f64,
    pub pts: Vec<Ipoint>,
    pub sizes: Vec<f32>,
    pub flags: u32,
    pub time: i32,
    pub surf: i32,
    /// `imodel.h:377` places `Ilabel *label` after `surf`.
    pub label: Option<super::ilabel::Ilabel>,
    pub store: Vec<super::istore::Istore>,
}

/// Original: `Mod_Mesh` (`include/imodel.h`).
#[derive(Clone, Debug, Default)]
pub struct Imesh {
    pub vert: Vec<Ipoint>,
    pub list: Vec<i32>,
    pub flag: u32,
    pub time: i16,
    pub surf: i16,
    pub store: Vec<super::istore::Istore>,
}

/// Original: `Mod_Object` (`include/imodel.h`), with fields consumed by
/// `imodinfo.cpp` retained in source order.
#[derive(Clone, Debug)]
pub struct Iobj {
    pub cont: Vec<Icont>,
    pub mesh: Vec<Imesh>,
    /// `Iobj::meshParam` (`imodel.h:443`), optional per-object meshing
    /// parameters persisted in MEPA/SKLI chunks.
    pub mesh_param: Option<MeshParams>,
    pub store: Vec<super::istore::Istore>,
    /// `imodel.h:442` declares `Ilabel *label` on the object.
    pub label: Option<super::ilabel::Ilabel>,
    /// `Mod_Object::name` (`imodel.h:392`), a fixed 64-byte field that
    /// `imodel_write` emits whole.  NATIVE.md §3/§5: the padding past the
    /// string is part of the on-disk layout, so this is `[u8; N]` with
    /// explicit NUL padding and never a `String`.
    pub name: [u8; IOBJ_STRSIZE],
    pub extra: [u32; 16],
    pub flags: u32,
    pub axis: i32,
    pub drawmode: i32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    /// `Iobj::fgcolor`, the mapped colour-index value used by `display.cpp`.
    pub fgcolor: i32,
    pub pdrawsize: i32,
    pub symbol: u8,
    pub symsize: u8,
    pub linewidth2: u8,
    pub linewidth: u8,
    pub linesty: u8,
    pub symflags: u8,
    pub sympad: u8,
    pub trans: u8,
    pub surfsize: i32,
    pub clips: Iclip_planes,
    pub ambient: u8,
    pub diffuse: u8,
    pub specular: u8,
    pub shininess: u8,
    pub fillred: u8,
    pub fillgreen: u8,
    pub fillblue: u8,
    pub quality: u8,
    pub mat2: u32,
    pub valblack: u8,
    pub valwhite: u8,
    pub matflags2: u8,
    pub mesh_thickness: u8,
}

/// `Iobj` cannot derive `Default` because `name` is a 64-byte array; this
/// reproduces the derived all-zero value field for field.
impl Default for Iobj {
    fn default() -> Self {
        Self {
            cont: Vec::new(),
            mesh: Vec::new(),
            mesh_param: None,
            store: Vec::new(),
            label: None,
            name: [0; IOBJ_STRSIZE],
            extra: [0; 16],
            flags: 0,
            axis: 0,
            drawmode: 0,
            red: 0.,
            green: 0.,
            blue: 0.,
            fgcolor: 0,
            pdrawsize: 0,
            symbol: 0,
            symsize: 0,
            linewidth2: 0,
            linewidth: 0,
            linesty: 0,
            symflags: 0,
            sympad: 0,
            trans: 0,
            surfsize: 0,
            clips: Iclip_planes::default(),
            ambient: 0,
            diffuse: 0,
            specular: 0,
            shininess: 0,
            fillred: 0,
            fillgreen: 0,
            fillblue: 0,
            quality: 0,
            mat2: 0,
            valblack: 0,
            valwhite: 0,
            matflags2: 0,
            mesh_thickness: 0,
        }
    }
}

/// Original: `Mod_Model` (`include/imodel.h`), fields consumed by
/// `imodinfo.cpp`.
#[derive(Clone, Debug)]
pub struct Imod {
    pub obj: Vec<Iobj>,
    /// `Mod_Model::ctime` (`imodel.h:457`), the current time index.
    pub ctime: i32,
    pub store: Vec<super::istore::Istore>,
    /// `Mod_Model::name` (`imodel.h:460`), a fixed 128-byte field that
    /// `imodel_write` emits whole.  Native leaves everything past the 13
    /// bytes `imodDefault` writes as heap residue, which is a documented
    /// non-achievable; this is `[u8; N]` with explicit padding, never a
    /// `String`, which would truncate at the first NUL and lose the field.
    pub name: [u8; IMOD_STRSIZE],
    pub xmax: i32,
    pub ymax: i32,
    pub zmax: i32,
    pub flags: u32,
    pub drawmode: i32,
    pub mousemode: i32,
    pub blacklevel: i32,
    pub whitelevel: i32,
    pub xoffset: f32,
    pub yoffset: f32,
    pub zoffset: f32,
    pub xscale: f32,
    pub yscale: f32,
    pub zscale: f32,
    pub cindex: Iindex,
    pub res: i32,
    pub thresh: i32,
    pub pixsize: f32,
    pub units: i32,
    pub csum: i32,
    /// `Mod_Model::tmax` (`imodel.h:480`), the largest model time index.
    /// This is runtime state and is not serialized by the model-file paths.
    pub tmax: i32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub cview: i32,
    pub view: Vec<Iview>,
    /// `Mod_Model::editGlobalClip` (`imodel.h:488`), "Flag that global clip is
    /// selected".  Runtime state, not part of the file format -- `imodel_write`
    /// never emits it -- and `imodDefault` leaves it 0.  Read by
    /// `mv_objed.cpp:1555-1760`, `mv_ogl.cpp:3069` and `mv_input.cpp:167-1032`.
    ///
    pub edit_global_clip: i32,
    /// `Mod_Model::curObjGroup` (`imodel.h:489`), current object-group index.
    pub cur_obj_group: i32,
    pub ref_image: Option<Iref_image>,
    /// `Mod_Model::fileName` (`imodel.h:492`), owned filename or native NULL.
    pub file_name: Option<String>,
    /// `Mod_Model::xybin` / `zbin` (`imodel.h:493-494`), runtime image binning.
    pub xybin: i32,
    pub zbin: i32,
    pub slicer_ang: Vec<Slicer_angles>,
    pub group_list: Vec<Iobj_group>,
    pub cur_mesh_surf: i32,
}

impl Default for Imod {
    fn default() -> Self {
        Self {
            obj: Vec::new(),
            ctime: 0,
            store: Vec::new(),
            name: [0; IMOD_STRSIZE],
            xmax: 0,
            ymax: 0,
            zmax: 0,
            flags: 0,
            drawmode: 0,
            mousemode: 0,
            blacklevel: 0,
            whitelevel: 0,
            xoffset: 0.,
            yoffset: 0.,
            zoffset: 0.,
            xscale: 0.,
            yscale: 0.,
            zscale: 0.,
            cindex: Iindex {
                object: -1,
                contour: -1,
                point: -1,
            },
            res: 0,
            thresh: 0,
            pixsize: 0.,
            units: 0,
            csum: 0,
            tmax: 0,
            alpha: 0.,
            beta: 0.,
            gamma: 0.,
            cview: 0,
            view: vec![Iview::default()],
            edit_global_clip: 0,
            cur_obj_group: -1,
            ref_image: None,
            file_name: None,
            xybin: 1,
            zbin: 1,
            slicer_ang: Vec::new(),
            group_list: Vec::new(),
            cur_mesh_surf: -1,
        }
    }
}

pub const IMOD_ERROR_READ: i32 = 10;
pub const IMOD_ERROR_WRITE: i32 = 11;
pub const IMOD_ERROR_VERSION: i32 = 20;
pub const IMOD_ERROR_FORMAT: i32 = 21;
pub const IMOD_ERROR_CORRUPT: i32 = 30;
pub const IMOD_ERROR_MEMORY: i32 = 50;

pub const IMOD_OBJFLAG_OFF: u32 = 1 << 1;
pub const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;
pub const IMOD_OBJFLAG_OUT: u32 = 1 << 5;
pub const IMOD_OBJFLAG_SCAT: u32 = 1 << 9;
pub const ICONT_OPEN: u32 = 1 << 3;
pub const ICONT_WILD: u32 = 1 << 4;
pub const IMOD_MESH_END: i32 = -1;
pub const IMOD_MESH_ENDPOLY: i32 = -22;
pub const IMOD_MESH_BGNPOLYNORM: i32 = -23;
pub const IMOD_MESH_BGNPOLYNORM2: i32 = -25;

pub const IMODF_FLIPYZ: u32 = 1 << 16;
pub const IMODF_TILTOK: u32 = 1 << 15;
pub const IMODF_OTRANS_ORIGIN: u32 = 1 << 14;
pub const IMODF_MULTIPLE_CLIP: u32 = 1 << 12;
pub const IMODF_NEW_TO_3DMOD: u32 = 1 << 11;
pub const IMODF_Z_FROM_MINUSPT5: u32 = 1 << 10;
pub const IMODF_ROT90X: u32 = 1 << 17;

/// Original: `IMOD_MMOVIE` (`imodel.h:27`).
pub const IMOD_MMOVIE: i32 = 2;

/// Original: `IMOD_UNIT_*` (`imodel.h:39`).
pub const IMOD_UNIT_PIXEL: i32 = 0;
pub const IMOD_UNIT_KILO: i32 = 3;
pub const IMOD_UNIT_METER: i32 = 1;
pub const IMOD_UNIT_CM: i32 = -2;
pub const IMOD_UNIT_MM: i32 = -3;
pub const IMOD_UNIT_UM: i32 = -6;
pub const IMOD_UNIT_NM: i32 = -9;
pub const IMOD_UNIT_ANGSTROM: i32 = -10;
pub const IMOD_UNIT_PM: i32 = -12;

/// Original: `imodGetCurMeshSurf` (`imodel.c:191`).
pub fn imod_get_cur_mesh_surf(imod: &Imod) -> i32 {
    let Some(object) = imod.obj.get(imod.cindex.object as usize) else {
        return -1;
    };
    if object.cont.is_empty() && !object.mesh.is_empty() && imod.cur_mesh_surf <= object.surfsize {
        return imod.cur_mesh_surf;
    }
    -1
}

/// Original: `imodSetCurMeshSurf` (`imodel.c:207`).
pub fn imod_set_cur_mesh_surf(imod: &mut Imod, surf: i32) {
    let Some(object) = imod.obj.get(imod.cindex.object as usize) else {
        return;
    };
    if object.cont.is_empty() && !object.mesh.is_empty() && surf <= object.surfsize {
        imod.cur_mesh_surf = surf;
    }
}

/// Original: `imodGetMaxObject` (`imodel.c:2190`).
pub fn imod_get_max_object(imod: &Imod) -> i32 {
    imod.obj.len() as i32
}

/// Original: `imodGetZScale` (`imodel.c:2194`).
pub fn imod_get_z_scale(imod: &Imod) -> f32 {
    imod.zscale
}

/// Original: `imodGetPixelSize` (`imodel.c:2198`).
pub fn imod_get_pixel_size(imod: &Imod) -> f32 {
    imod.pixsize
}

/// Original: `imodGetMaxTime` (`imodel.c:1444`).
pub fn imod_get_max_time(imod: Option<&Imod>) -> i32 {
    let mut max_time = 0;
    let Some(imod) = imod else {
        return max_time;
    };
    for obj in &imod.obj {
        for cont in &obj.cont {
            if cont.time > max_time {
                max_time = cont.time;
            }
        }
    }
    max_time
}

/// Original: `imodCleanSurf` (`imodel.c:1563`).
pub fn imod_clean_surf(imod: &mut Imod) {
    for obj in &mut imod.obj {
        let mut max_surf = 0;
        for cont in &obj.cont {
            if cont.surf > max_surf {
                max_surf = cont.surf;
            }
        }
        for mesh in &obj.mesh {
            if i32::from(mesh.surf) > max_surf {
                max_surf = i32::from(mesh.surf);
            }
        }
        obj.surfsize = max_surf;
    }
}

/// Original: `imodNewContour` (`imodel.c:715`).
pub fn imod_new_contour(mod_: &mut Imod) -> i32 {
    if mod_.cindex.object < 0 || mod_.cindex.object as usize >= mod_.obj.len() {
        return -1;
    }
    let object = mod_.cindex.object as usize;
    let previous = mod_.cindex.contour;
    let mut cont = Icont::default();
    if previous >= 0 && (previous as usize) < mod_.obj[object].cont.len() {
        let old = &mod_.obj[object].cont[previous as usize];
        cont.surf = old.surf;
        if old.flags & ICONT_OPEN != 0 {
            cont.flags = ICONT_OPEN;
        }
    }
    mod_.obj[object].cont.push(cont);
    mod_.cindex.contour = mod_.obj[object].cont.len() as i32 - 1;
    mod_.cindex.point = -1;
    0
}

/// Original: `imodNewPoint` (`imodel.c:1062`).
pub fn imod_new_point(imod: &mut Imod, point: Option<Ipoint>) -> i32 {
    let Some(point) = point else {
        return 0;
    };
    if imod.cindex.object < 0
        || imod.cindex.object as usize >= imod.obj.len()
        || imod.cindex.contour < 0
        || imod.cindex.contour as usize >= imod.obj[imod.cindex.object as usize].cont.len()
    {
        return 0;
    }
    imod.cindex.point += 1;
    let cont = &mut imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize];
    cont.pts.push(point);
    cont.pts.len() as i32
}

/// Original: `imodInsertPoint` (`imodel.c:1075`).
pub fn imod_insert_point(imod: Option<&mut Imod>, point: Option<Ipoint>, mut index: i32) -> i32 {
    let Some(imod) = imod else { return 0 };
    let Some(point) = point else { return 0 };
    if imod.cindex.object < 0
        || imod.cindex.object as usize >= imod.obj.len()
        || imod.cindex.contour < 0
        || imod.cindex.contour as usize >= imod.obj[imod.cindex.object as usize].cont.len()
    {
        return 0;
    }
    let cont = &mut imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize];
    if index < 0 {
        index = 0;
    }
    if index > cont.pts.len() as i32 {
        index = cont.pts.len() as i32;
    }
    cont.pts.insert(index as usize, point);
    imod.cindex.point = index;
    cont.pts.len() as i32
}

/// Original: `imodDeletePoint` (`imodel.c:1098`).
pub fn imod_delete_point(imod: &mut Imod) -> i32 {
    if imod.cindex.object < 0
        || imod.cindex.object as usize >= imod.obj.len()
        || imod.cindex.contour < 0
        || imod.cindex.contour as usize >= imod.obj[imod.cindex.object as usize].cont.len()
    {
        return -1;
    }
    let object = imod.cindex.object as usize;
    let contour = imod.cindex.contour as usize;
    if !imod.obj[object].cont[contour].pts.is_empty() && imod.cindex.point < 0 {
        return -1;
    }
    if imod.obj[object].cont[contour].pts.is_empty() {
        imod.obj[object].cont.remove(contour);
        return 0;
    }
    let index = imod.cindex.point;
    if index < 0 || index as usize >= imod.obj[object].cont[contour].pts.len() {
        return -1;
    }
    if index != 0 || imod.obj[object].cont[contour].pts.len() == 1 {
        imod.cindex.point = index - 1;
    }
    imod.obj[object].cont[contour].pts.remove(index as usize);
    imod.obj[object].cont[contour].pts.len() as i32
}

/// Original: `imodPrevPoint` (`imodel.c:1129`).
pub fn imod_prev_point(mod_: Option<&mut Imod>) -> i32 {
    let Some(mod_) = mod_ else {
        return -1;
    };
    if mod_.cindex.object < 0 || mod_.cindex.contour < 0 {
        return -1;
    }
    if mod_.cindex.point <= 0 {
        return mod_.cindex.point;
    }
    mod_.cindex.point -= 1;
    mod_.cindex.point
}

/// Original: `imodNextPoint` (`imodel.c:1155`).
pub fn imod_next_point(mod_: &mut Imod) -> i32 {
    if mod_.cindex.object < 0
        || mod_.cindex.object as usize >= mod_.obj.len()
        || mod_.cindex.contour < 0
        || mod_.cindex.contour as usize >= mod_.obj[mod_.cindex.object as usize].cont.len()
    {
        return -1;
    }
    let size = mod_.obj[mod_.cindex.object as usize].cont[mod_.cindex.contour as usize]
        .pts
        .len() as i32;
    mod_.cindex.point += 1;
    if mod_.cindex.point >= size {
        mod_.cindex.point = size - 1;
    }
    mod_.cindex.point
}

/// Original: `imodGetFilename` (`imodel.c:2202`).
///
/// The source returns `imod->fileName`, a separately allocated path that this
/// data representation does not carry; the model `name` array is returned in
/// its place.  The C returns a `char *` into a NUL-terminated buffer, so the
/// slice stops at the first NUL exactly as `strlen` would.
pub fn imod_get_filename(imod: &Imod) -> &[u8] {
    let end = imod
        .name
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(IMOD_STRSIZE);
    &imod.name[..end]
}

/// Original: `imodGetFlipped` (`imodel.c:2206`).
pub fn imod_get_flipped(imod: &Imod) -> u32 {
    imod.flags & IMODF_FLIPYZ
}

/// Original: `imodGetIndex` (`imodel.c:131`).
pub fn imod_get_index(imod: &Imod, object: &mut i32, contour: &mut i32, point: &mut i32) {
    *object = imod.cindex.object;
    *contour = imod.cindex.contour;
    *point = imod.cindex.point;
}

/// Original: `imodSetIndex` (`imodel.c:145`).
pub fn imod_set_index(imod: &mut Imod, mut object: i32, mut contour: i32, mut point: i32) {
    if object < 0 {
        object = -1;
        contour = -1;
        point = -1;
    }
    if object >= 0 && object as usize >= imod.obj.len() {
        object = 0;
    }
    imod.cindex.object = object;
    let Some(obj) = imod.obj.get(object as usize) else {
        imod.cindex.contour = -1;
        imod.cindex.point = -1;
        return;
    };
    if contour < 0 {
        contour = -1;
        point = -1;
    }
    if contour >= 0 && contour as usize >= obj.cont.len() {
        contour = obj.cont.len() as i32 - 1;
    }
    imod.cindex.contour = contour;
    let Some(cont) = obj.cont.get(contour as usize) else {
        imod.cindex.point = -1;
        return;
    };
    if point >= 0 && point as usize >= cont.pts.len() {
        point = cont.pts.len() as i32 - 1;
    }
    if point < 0 || cont.pts.is_empty() {
        point = -1;
    }
    imod.cindex.point = point;
}

/// Original: `imodel_maxpt` (`imodel.c:224`).
pub fn imodel_maxpt(imod: &Imod, pnt: &mut Ipoint) {
    *pnt = Ipoint {
        x: -1.,
        y: -1.,
        z: -1.,
    };
    for object in &imod.obj {
        for contour in &object.cont {
            if let Some(point) = contour.pts.first() {
                *pnt = *point;
                for object in &imod.obj {
                    for contour in &object.cont {
                        for point in &contour.pts {
                            pnt.x = pnt.x.max(point.x);
                            pnt.y = pnt.y.max(point.y);
                            pnt.z = pnt.z.max(point.z);
                        }
                    }
                }
                return;
            }
        }
    }
}

/// Original: `imodel_dist` (`imodel.c:352`).
pub fn imodel_dist(imod: &Imod) -> f64 {
    if imod.cindex.point < 1 {
        return 0.0;
    }
    let point = &imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize].pts
        [imod.cindex.point as usize];
    let previous = &imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize].pts
        [imod.cindex.point as usize - 1];
    let mut x = point.x as f64 - previous.x as f64;
    let mut y = point.y as f64 - previous.y as f64;
    x *= x;
    y *= y;
    (x + y).sqrt()
}

/// Original: `imodNewObject` (`imodel.c:384`).
pub fn imod_new_object(mod_: &mut Imod) -> i32 {
    const MAX_STOCK_COLORS: usize = 35;
    let colors: [[f32; 3]; MAX_STOCK_COLORS] = [
        [0., 1., 0.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.5],
        [0.2, 0.2, 0.8],
        [0.8, 0.2, 0.2],
        [0.9, 0.6, 0.4],
        [0.6, 0.4, 0.9],
        [0.1, 0.6, 0.4],
        [0.6, 0.1, 0.4],
        [0.2, 0.6, 0.8],
        [1., 0.5, 0.],
        [0.4, 0.6, 0.1],
        [0.1, 0.1, 0.6],
        [0.9, 0.9, 0.4],
        [0.9, 0.4, 0.6],
        [0.4, 0.9, 0.9],
        [0.6, 0.2, 0.2],
        [0.2, 0.8, 0.6],
        [0.4, 0.6, 0.9],
        [0.1, 0.6, 0.1],
        [0.8, 0.5, 0.2],
        [1., 0., 0.5],
        [0., 0.5, 1.],
        [0.6, 0.2, 0.8],
        [0.5, 1., 0.],
        [0.1, 0.4, 0.6],
        [0.6, 0.4, 0.1],
        [0.8, 0.2, 0.6],
        [0.4, 0.1, 0.6],
        [0.2, 0.8, 0.2],
        [0.9, 0.4, 0.9],
    ];

    if mod_.obj.try_reserve_exact(1).is_err() {
        mod_.cindex.object = -1;
        mod_.cindex.contour = -1;
        mod_.cindex.point = -1;
        return 1;
    }

    let color_ind = mod_.obj.len() % MAX_STOCK_COLORS;
    mod_.obj.push(Iobj {
        extra: [0; 16],
        cont: Vec::new(),
        mesh: Vec::new(),
        mesh_param: None,
        store: Vec::new(),
        label: None,
        name: [0; IOBJ_STRSIZE],
        flags: (1 << 27) | (1 << 28),
        axis: 0,
        drawmode: 1,
        red: colors[color_ind][0],
        green: colors[color_ind][1],
        blue: colors[color_ind][2],
        fgcolor: 0,
        pdrawsize: 0,
        symbol: 1,
        symsize: 3,
        linewidth2: 1,
        linewidth: 1,
        linesty: 0,
        symflags: 0,
        sympad: 0,
        trans: 0,
        surfsize: 0,
        clips: Iclip_planes {
            count: 0,
            flags: 0,
            trans: 0,
            plane: 0,
            normal: [Ipoint {
                x: 0.,
                y: 0.,
                z: -1.,
            }; IMOD_CLIPSIZE],
            point: [Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            }; IMOD_CLIPSIZE],
        },
        ambient: 102,
        diffuse: 255,
        specular: 127,
        shininess: 4,
        fillred: 0,
        fillgreen: 0,
        fillblue: 0,
        quality: 0,
        mat2: 0,
        valblack: 0,
        valwhite: 255,
        matflags2: 0,
        mesh_thickness: 0,
    });
    mod_.cindex.object = mod_.obj.len() as i32 - 1;
    mod_.cindex.contour = -1;
    mod_.cindex.point = -1;
    0
}

/// Original: `imodNextObject` (`imodel.c:585`).
pub fn imod_next_object(imod: Option<&mut Imod>) -> i32 {
    let Some(imod) = imod else {
        return -1;
    };
    if imod.obj.is_empty() {
        return -1;
    }
    if imod.cindex.object >= imod.obj.len() as i32 - 1 {
        return imod.cindex.object;
    }

    imod.cindex.object += 1;
    if imod.cindex.object < 0 {
        imod.cindex.object = 0;
    }
    if imod.obj.get(imod.cindex.object as usize).is_none() {
        return -1;
    }

    if imod.cindex.contour >= imod.obj[imod.cindex.object as usize].cont.len() as i32 {
        imod.cindex.contour = imod.obj[imod.cindex.object as usize].cont.len() as i32 - 1;
    }
    if imod.cindex.contour >= 0
        && imod.obj[imod.cindex.object as usize]
            .cont
            .get(imod.cindex.contour as usize)
            .is_some()
    {
        if imod.cindex.point
            >= imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize]
                .pts
                .len() as i32
        {
            imod.cindex.point = imod.obj[imod.cindex.object as usize].cont
                [imod.cindex.contour as usize]
                .pts
                .len() as i32
                - 1;
        }
        if imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize]
            .pts
            .is_empty()
        {
            imod.cindex.point = -1;
        }
    } else {
        imod.cindex.point = -1;
    }

    imod.cindex.object
}

/// Original: `imodPrevObject` (`imodel.c:625`).
pub fn imod_prev_object(mod_: Option<&mut Imod>) -> i32 {
    let Some(mod_) = mod_ else {
        return -1;
    };
    if mod_.obj.is_empty() {
        return -1;
    }
    if mod_.cindex.object <= 0 {
        return mod_.cindex.object;
    }

    if mod_.cindex.object > mod_.obj.len() as i32 {
        mod_.cindex.object = mod_.obj.len() as i32;
    }
    mod_.cindex.object -= 1;
    if mod_.obj.get(mod_.cindex.object as usize).is_none() {
        return -1;
    }

    if mod_.cindex.contour >= mod_.obj[mod_.cindex.object as usize].cont.len() as i32 {
        mod_.cindex.contour = mod_.obj[mod_.cindex.object as usize].cont.len() as i32 - 1;
    }
    if mod_.cindex.contour >= 0
        && mod_.obj[mod_.cindex.object as usize]
            .cont
            .get(mod_.cindex.contour as usize)
            .is_some()
    {
        if mod_.cindex.point
            >= mod_.obj[mod_.cindex.object as usize].cont[mod_.cindex.contour as usize]
                .pts
                .len() as i32
        {
            mod_.cindex.point = mod_.obj[mod_.cindex.object as usize].cont
                [mod_.cindex.contour as usize]
                .pts
                .len() as i32
                - 1;
        }
        if mod_.obj[mod_.cindex.object as usize].cont[mod_.cindex.contour as usize]
            .pts
            .is_empty()
        {
            mod_.cindex.point = -1;
        }
    } else {
        mod_.cindex.point = -1;
    }

    mod_.cindex.object
}

/// Original: `imodPrevContour` (`imodel.c:922`).
pub fn imod_prev_contour(mod_: Option<&mut Imod>) -> i32 {
    let Some(mod_) = mod_ else {
        return -1;
    };
    if mod_.obj.get(mod_.cindex.object as usize).is_none() {
        mod_.cindex.contour = -1;
        return -1;
    }
    if mod_.obj[mod_.cindex.object as usize].cont.is_empty() {
        return -1;
    }
    if mod_.cindex.contour == 0 {
        return mod_.cindex.contour;
    }
    if mod_.cindex.contour < 0 {
        mod_.cindex.contour = mod_.obj[mod_.cindex.object as usize].cont.len() as i32 - 1;
    } else {
        mod_.cindex.contour -= 1;
    }
    if mod_.cindex.point
        > mod_.obj[mod_.cindex.object as usize].cont[mod_.cindex.contour as usize]
            .pts
            .len() as i32
            - 1
    {
        mod_.cindex.point = mod_.obj[mod_.cindex.object as usize].cont[mod_.cindex.contour as usize]
            .pts
            .len() as i32
            - 1;
    }
    mod_.cindex.contour
}

/// Original: `imodNextContour` (`imodel.c:962`).
pub fn imod_next_contour(imod: Option<&mut Imod>) -> i32 {
    let Some(imod) = imod else {
        return -1;
    };
    if imod.obj.get(imod.cindex.object as usize).is_none() {
        return -1;
    }
    if imod.obj[imod.cindex.object as usize].cont.is_empty() {
        return -1;
    }
    if imod.cindex.contour == imod.obj[imod.cindex.object as usize].cont.len() as i32 - 1 {
        return imod.cindex.contour;
    }
    if imod.cindex.contour < 0 {
        imod.cindex.contour = 0;
    } else {
        imod.cindex.contour += 1;
    }
    if imod.cindex.point
        >= imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize]
            .pts
            .len() as i32
    {
        imod.cindex.point = imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize]
            .pts
            .len() as i32
            - 1;
    }
    imod.cindex.contour
}

/// Original: `imodContourGet` (`imodel.c:1001`).
pub fn imod_contour_get(imod: Option<&Imod>) -> Option<&Icont> {
    let imod = imod?;
    let object = imod.obj.get(imod.cindex.object as usize)?;
    if imod.cindex.contour < 0 || imod.cindex.contour >= object.cont.len() as i32 {
        return None;
    }
    object.cont.get(imod.cindex.contour as usize)
}

/// Original: `imodContourGetFirst` (`imodel.c:1021`).
pub fn imod_contour_get_first(imod: Option<&mut Imod>) -> Option<&Icont> {
    let imod = imod?;
    let (mut object, mut contour, mut point) = (0, 0, 0);
    imod_get_index(imod, &mut object, &mut contour, &mut point);
    imod_set_index(imod, object, 0, point);
    imod_contour_get(Some(imod))
}

/// Original: `imodContourGetNext` (`imodel.c:1036`).
pub fn imod_contour_get_next(imod: Option<&mut Imod>) -> Option<&Icont> {
    let imod = imod?;
    let (mut object, mut contour, mut point) = (0, 0, 0);
    imod_get_index(imod, &mut object, &mut contour, &mut point);
    let current_object = imod.obj.get(imod.cindex.object as usize)?;
    if current_object.cont.is_empty() {
        return None;
    }
    contour += 1;
    if contour >= current_object.cont.len() as i32 {
        return None;
    }
    imod_set_index(imod, object, contour, point);
    imod_contour_get(Some(imod))
}

/// Original: `imodel_minpt` (`imodel.c:278`).
pub fn imodel_minpt(imod: &Imod, pnt: &mut Ipoint) {
    *pnt = Ipoint {
        x: -1.,
        y: -1.,
        z: -1.,
    };
    for object in &imod.obj {
        for contour in &object.cont {
            if let Some(point) = contour.pts.first() {
                *pnt = *point;
                for object in &imod.obj {
                    for contour in &object.cont {
                        for point in &contour.pts {
                            pnt.x = pnt.x.min(point.x);
                            pnt.y = pnt.y.min(point.y);
                            pnt.z = pnt.z.min(point.z);
                        }
                    }
                }
                return;
            }
        }
    }
}

/// Original: `imodGetBoundingBox` (`imodel.c:325`).
pub fn imod_get_bounding_box(imod: &Imod, min: &mut Ipoint, max: &mut Ipoint) {
    *min = Ipoint {
        x: -1.,
        y: -1.,
        z: -1.,
    };
    *max = *min;
    let mut got_one = false;
    for object in &imod.obj {
        let mut object_min = Ipoint {
            x: f32::MAX,
            y: f32::MAX,
            z: f32::MAX,
        };
        let mut object_max = Ipoint {
            x: -f32::MAX,
            y: -f32::MAX,
            z: -f32::MAX,
        };
        let mut object_got_one = false;
        if !object.cont.is_empty() {
            for contour in &object.cont {
                for point in &contour.pts {
                    object_min.x = object_min.x.min(point.x);
                    object_min.y = object_min.y.min(point.y);
                    object_min.z = object_min.z.min(point.z);
                    object_max.x = object_max.x.max(point.x);
                    object_max.y = object_max.y.max(point.y);
                    object_max.z = object_max.z.max(point.z);
                    object_got_one = true;
                }
            }
        } else {
            for mesh in &object.mesh {
                for point in &mesh.vert {
                    object_min.x = object_min.x.min(point.x);
                    object_min.y = object_min.y.min(point.y);
                    object_min.z = object_min.z.min(point.z);
                    object_max.x = object_max.x.max(point.x);
                    object_max.y = object_max.y.max(point.y);
                    object_max.z = object_max.z.max(point.z);
                    object_got_one = true;
                }
            }
        }
        if object_got_one {
            if got_one {
                min.x = min.x.min(object_min.x);
                min.y = min.y.min(object_min.y);
                min.z = min.z.min(object_min.z);
                max.x = max.x.max(object_max.x);
                max.y = max.y.max(object_max.y);
                max.z = max.z.max(object_max.z);
            } else {
                *min = object_min;
                *max = object_max;
                got_one = true;
            }
        }
    }
}

/// Original: `imodFlipYZ` (`imodel.c:1584`).
pub fn imod_flip_yz(imod: &mut Imod) {
    for object in &mut imod.obj {
        for contour in &mut object.cont {
            for point in &mut contour.pts {
                std::mem::swap(&mut point.y, &mut point.z);
            }
        }
        for mesh in &mut object.mesh {
            for point in &mut mesh.vert {
                std::mem::swap(&mut point.y, &mut point.z);
            }
        }
        flip_clips(&mut object.clips);
    }
    for view in &mut imod.view {
        flip_clips(&mut view.clips);
        for object in &mut view.objview {
            flip_clips(&mut object.clips);
        }
    }
    std::mem::swap(&mut imod.ymax, &mut imod.zmax);
}

/// Original: `imodTransFromRefImage` (`imodel.c:1851`).
///
/// Transforms the model in `imod` according to the old and current shift,
/// scale, and rotation in the `IrefImage` structure `iref`, with the
/// additional scaling required for binning in each dimension specified in
/// `bin_scale`.  Returns 1 for memory errors.
pub fn imod_trans_from_ref_image(imod: &mut Imod, iref: &Iref_image, bin_scale: Ipoint) -> i32 {
    let mut pnt: Ipoint;
    let sphere_scale: f32;

    /* Before any transforming, unflip a flipped model */
    if imod.flags & IMODF_FLIPYZ != 0 {
        imod_flip_yz(imod);
        imod.flags &= !IMODF_FLIPYZ;
    }

    /* First transform to "absolute" image coords using old reference
    image data */
    /* Compute separate matrices for point and clip plane normal transforms */
    let mat = imod_mat_new(3);
    let mat_clip = imod_mat_new(3);
    let mat_norm = imod_mat_new(3);
    let (Some(mut mat), Some(mut mat_clip), Some(mut mat_norm)) = (mat, mat_clip, mat_norm) else {
        return 1;
    };

    /* Model coordinates range from -0.5 to nz - 0.5 when pixels are considered to
    have thickness in Z, so adjust up by 0.5, apply operations, adjust back down */
    pnt = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.5,
    };
    imod_mat_trans(&mut mat, &pnt);

    imod_mat_scale(&mut mat, &iref.oscale);
    pnt = Ipoint {
        x: 1. / iref.oscale.x,
        y: 1. / iref.oscale.y,
        z: 1. / iref.oscale.z,
    };
    imod_mat_scale(&mut mat_clip, &pnt);

    pnt = Ipoint {
        x: -iref.otrans.x,
        y: -iref.otrans.y,
        z: -iref.otrans.z,
    };
    imod_mat_trans(&mut mat, &pnt);

    /* DNM 11/5/98: because tilt angles were not properly set into the
    model data when IrefImage was created, do rotations
    only if the flag is set that tilts have been stored properly */
    /* DNM 1/3/04: and skip it if there is no rotation change */
    if imod.flags & IMODF_TILTOK != 0
        && (iref.crot.x != iref.orot.x || iref.crot.y != iref.orot.y || iref.crot.z != iref.orot.z)
    {
        imod_mat_rot(&mut mat, -iref.orot.x as f64, B3D_X);
        imod_mat_rot(&mut mat, -iref.orot.y as f64, B3D_Y);
        imod_mat_rot(&mut mat, -iref.orot.z as f64, B3D_Z);

        imod_mat_rot(&mut mat_clip, -iref.orot.x as f64, B3D_X);
        imod_mat_rot(&mut mat_clip, -iref.orot.y as f64, B3D_Y);
        imod_mat_rot(&mut mat_clip, -iref.orot.z as f64, B3D_Z);

        /* Next transform from these "absolute" coords to new reference
        image coords */

        imod_mat_rot(&mut mat, iref.crot.z as f64, B3D_Z);
        imod_mat_rot(&mut mat, iref.crot.y as f64, B3D_Y);
        imod_mat_rot(&mut mat, iref.crot.x as f64, B3D_X);

        imod_mat_rot(&mut mat_clip, iref.crot.z as f64, B3D_Z);
        imod_mat_rot(&mut mat_clip, iref.crot.y as f64, B3D_Y);
        imod_mat_rot(&mut mat_clip, iref.crot.x as f64, B3D_X);
    }

    imod_mat_trans(&mut mat, &iref.ctrans);

    pnt = Ipoint {
        x: 1. / (iref.cscale.x * bin_scale.x),
        y: 1. / (iref.cscale.y * bin_scale.y),
        z: 1. / (iref.cscale.z * bin_scale.z),
    };
    imod_mat_scale(&mut mat, &pnt);
    pnt = Ipoint {
        x: 0.,
        y: 0.,
        z: -0.5,
    };
    imod_mat_trans(&mut mat, &pnt);

    /* Mesh normals scaling does not need to include the binning scale because
    they already include Z-scaling and the model display will be adjusted to
    display with proper Z-scaling including the binning difference
    But clip normals do need the bin scaling included */
    imod_mat_copy(&mat_clip, &mut mat_norm);
    imod_mat_scale(&mut mat_norm, &iref.cscale);
    pnt = Ipoint {
        x: iref.cscale.x * bin_scale.x,
        y: iref.cscale.y * bin_scale.y,
        z: iref.cscale.z * bin_scale.z,
    };
    imod_mat_scale(&mut mat_clip, &pnt);

    imod_trans_from_mats(imod, &mat, &mat_norm, &mat_clip);
    sphere_scale = (((iref.oscale.x / iref.cscale.x
        + iref.oscale.y / iref.cscale.y
        + iref.oscale.z / iref.cscale.z) as f64)
        / 3.) as f32;
    if (sphere_scale as f64 - 1.).abs() > 0.01 {
        imod_scale_spheres(imod, sphere_scale);
    }

    imod_mat_delete(&mut mat);
    imod_mat_delete(&mut mat_clip);
    imod_mat_delete(&mut mat_norm);
    0
}

/// Original: `imodTransFromMats` (`imodel.c:1995`).
///
/// Transforms the model in `imod` given three matrices: `mat` for transforming
/// points, `mat_norm` for transforming the mesh normals, and `mat_clip` for
/// transforming clip plane normals.
pub fn imod_trans_from_mats(imod: &mut Imod, mat: &Imat, mat_norm: &Imat, mat_clip: &Imat) {
    let mut pnt = Ipoint::default();

    for ob in 0..imod.obj.len() {
        /* Transform points in contours */
        for co in 0..imod.obj[ob].cont.len() {
            for pt in 0..imod.obj[ob].cont[co].pts.len() {
                imod_mat_transform(mat, &imod.obj[ob].cont[co].pts[pt], &mut pnt);
                imod.obj[ob].cont[co].pts[pt] = pnt;
            }
        }

        imod_clips_trans(&mut imod.obj[ob].clips, mat, mat_clip);

        /* Transform the mesh points and the normals */
        for me in 0..imod.obj[ob].mesh.len() {
            if imod.obj[ob].mesh[me].vert.is_empty() {
                continue;
            }
            let mut i = 0;
            while i < imod.obj[ob].mesh[me].vert.len() {
                imod_mat_transform(mat, &imod.obj[ob].mesh[me].vert[i], &mut pnt);
                imod.obj[ob].mesh[me].vert[i] = pnt;
                imod_mat_transform(mat_norm, &imod.obj[ob].mesh[me].vert[i + 1], &mut pnt);
                imod_point_normalize(&mut pnt);
                imod.obj[ob].mesh[me].vert[i + 1] = pnt;
                i += 2;
            }
        }
    }

    /* Now transform clip points in views and object views */
    for i in 0..imod.view.len() {
        imod_clips_trans(&mut imod.view[i].clips, mat, mat_clip);
        for ob in 0..imod.view[i].objview.len() {
            imod_clips_trans(&mut imod.view[i].objview[ob].clips, mat, mat_clip);
        }
    }
}

/// Original: `imodScaleSpheres` (`imodel.c:2049`, static).
///
/// Scale object sphere size and point sizes by the scaling change.
pub fn imod_scale_spheres(imod: &mut Imod, scale: f32) {
    for ob in 0..imod.obj.len() {
        let obj = &mut imod.obj[ob];
        obj.pdrawsize = ((obj.pdrawsize as f32) * scale) as i32;
        for co in 0..obj.cont.len() {
            let cont = &mut obj.cont[co];
            if !cont.sizes.is_empty() {
                for pt in 0..cont.pts.len() {
                    cont.sizes[pt] *= scale;
                }
            }
        }
    }
}

/// Original: `imodNew` (`imodel.c:29`).
///
/// Allocates a new model structure and returns it, or `None` if there is an
/// error.  Initializes the model with `imodDefault`.
pub fn imod_new() -> Option<Imod> {
    /* The source's `model->file = NULL` has no counterpart: this data
    representation carries no `FILE *`. */
    let mut model = Imod::default();
    imod_default(&mut model);
    Some(model)
}

/// Original: `imodDefault` (`imodel.c:46`).
///
/// Initializes model structure `model` to default values and allocates an
/// initial view.  Returns 0 (there are no errors).
///
/// Deviation note: `Imod` has no `FILE *file` member: Rust callers own the
/// `ImodFile` borrow explicitly instead of storing an alias in the model.
pub fn imod_default(model: &mut Imod) -> i32 {
    let newmodname = b"IMOD-NewModel";

    model.obj.clear();
    model.flags = IMODF_NEW_TO_3DMOD | IMODF_Z_FROM_MINUSPT5;
    let mut i = 0;
    while i < 13 {
        model.name[i] = newmodname[i];
        i += 1;
    }
    model.name[i] = 0x00;
    model.drawmode = 1;
    model.mousemode = IMOD_MMOVIE;
    model.blacklevel = 0;
    model.whitelevel = 255;

    model.xoffset = 0.;
    model.yoffset = 0.;
    model.zoffset = 0.;

    model.xscale = 1.;
    model.yscale = 1.;
    model.zscale = 1.;

    model.cindex.object = -1;
    model.cindex.contour = -1;
    model.cindex.point = -1;

    model.res = 3;
    model.thresh = 128;
    model.pixsize = 1.0;
    model.units = 0; /* if unit is 0, pixsize is undefined */

    model.csum = 0;
    model.tmax = 0;
    model.xmax = 1;
    model.ymax = 1;
    model.zmax = 1;

    model.alpha = 0.0f32;
    model.beta = 0.0f32;
    model.gamma = 0.0f32;

    model.view.clear(); /* available views. */
    model.cview = 0; /* current view.    */
    /* imodViewModelNew(model) -- `iview.c:396` */
    crate::imod::libimod::iview::imod_view_model_new(model);

    model.ref_image = None;
    model.cur_obj_group = -1;
    model.file_name = None;
    model.xybin = 1;
    model.zbin = 1;
    model.cur_mesh_surf = -1;
    model.store.clear();
    model.slicer_ang.clear();
    model.group_list.clear();
    0
}

/// Original: `imodDelete` (`imodel.c:111`).
///
/// Deletes a model; frees all memory used in objects, views, and object views.
pub fn imod_delete(imod: &mut Imod) {
    /* imodObjviewsFree(imod) -- `iview.c:579` */
    for i in 1..imod.view.len() {
        imod.view[i].objview.clear();
    }
    /* imodViewDelete(imod->view) -- `iview.c:32` */
    imod.view.clear();
    imod_objects_delete(&mut imod.obj);
    imod.ref_image = None;
    imod.file_name = None;
    imod.store.clear();
    obj_group_list_delete(&mut imod.group_list);
}

/// Original: `imodDeleteObject` (`imodel.c:484`).
///
/// Deletes the object at `index` in model `mod_`.  Returns -1 for error.
pub fn imod_delete_object(mod_: &mut Imod, index: i32) -> i32 {
    if index < 0 || index >= mod_.obj.len() as i32 {
        return -1;
    }

    mod_.cindex.object = index;

    /* Delete all contours in object before we delete object. */
    let contsize = mod_.obj[index as usize].cont.len() as i32;
    imod_contours_delete(&mut mod_.obj[index as usize].cont, contsize);

    let objsize = mod_.obj.len() as i32;

    /* Copy objects above deleted object down one.  The source's loop reads one
    element past the end of the array when the deleted object is not the last;
    `Vec::remove` has the same visible effect without the overrun. */
    mod_.obj.remove(index as usize);

    /* imodObjviewDelete(mod, index) -- `iview.c:553` */
    for i in 1..mod_.view.len() {
        if (index as usize) < mod_.view[i].objview.len() {
            mod_.view[i].objview.remove(index as usize);
        }
    }

    if objsize > 1 {
        if index != 0 {
            mod_.cindex.object = index - 1;
        }

        let cur = mod_.cindex.object as usize;
        if mod_.cindex.contour >= mod_.obj[cur].cont.len() as i32 {
            mod_.cindex.contour = mod_.obj[cur].cont.len() as i32 - 1;
        }

        if mod_.obj[cur].cont.is_empty() || mod_.cindex.contour < 0 {
            mod_.cindex.point = -1;
            mod_.cindex.contour = -1;
        } else if mod_.cindex.point
            >= mod_.obj[cur].cont[mod_.cindex.contour as usize].pts.len() as i32
        {
            mod_.cindex.point =
                mod_.obj[cur].cont[mod_.cindex.contour as usize].pts.len() as i32 - 1;
        }
    } else {
        /* Delete last object in model */
        mod_.cindex.point = -1;
        mod_.cindex.contour = -1;
        mod_.cindex.object = -1;
    }
    0
}

/// Original: `imodMoveObject` (`imodel.c:546`).
///
/// Moves an object from index `ob_old` to index `ob_new` in the model `imod`.
/// Returns 1 if memory error.
pub fn imod_move_object(imod: &mut Imod, ob_old: i32, ob_new: i32) -> i32 {
    if ob_old == ob_new {
        return 0;
    }

    /* imodObjviewComplete(imod) -- `iview.c:519`, with the
    `imodObjviewFromObject` (`iview.c:438`) copy inlined */
    for iv in 1..imod.view.len() {
        if imod.view[iv].objview.len() < imod.obj.len() {
            for j in imod.view[iv].objview.len()..imod.obj.len() {
                let obj = &imod.obj[j];
                let objview = Iobjview {
                    flags: obj.flags,
                    red: obj.red,
                    green: obj.green,
                    blue: obj.blue,
                    pdrawsize: obj.pdrawsize,
                    linewidth: obj.linewidth,
                    linesty: obj.linesty,
                    trans: obj.trans,
                    clips: obj.clips.clone(),
                    ambient: obj.ambient,
                    diffuse: obj.diffuse,
                    specular: obj.specular,
                    shininess: obj.shininess,
                    fillred: obj.fillred,
                    fillgreen: obj.fillgreen,
                    fillblue: obj.fillblue,
                    quality: obj.quality,
                    mat2: obj.mat2,
                    valblack: obj.valblack,
                    valwhite: obj.valwhite,
                    matflags2: obj.matflags2,
                    mesh_thickness: obj.mesh_thickness,
                };
                imod.view[iv].objview.push(objview);
            }
        }
    }

    /* Copy object structures; the source's save-shift-restore loop is the same
    permutation as removing the object and reinserting it at the new index. */
    let obj_save = imod.obj.remove(ob_old as usize);
    imod.obj.insert(ob_new as usize, obj_save);

    /* Copy object views */
    for iv in 1..imod.view.len() {
        let vw = &mut imod.view[iv];
        if (ob_old as usize) < vw.objview.len() && (ob_new as usize) < vw.objview.len() {
            let obvw_save = vw.objview.remove(ob_old as usize);
            vw.objview.insert(ob_new as usize, obvw_save);
        }
    }
    0
}

/// Original: `imodObjectGet` (`imodel.c:663`).
///
/// Returns the current object in model `imod`, or `None` if no legal object is
/// selected.
pub fn imod_object_get(imod: Option<&Imod>) -> Option<&Iobj> {
    let imod = imod?;
    if imod.cindex.object < 0 || imod.cindex.object >= imod.obj.len() as i32 {
        return None;
    }
    Some(&imod.obj[imod.cindex.object as usize])
}

/// Original: `imodObjectGetFirst` (`imodel.c:676`).
///
/// Sets first object in model `imod` as current object and returns it.
pub fn imod_object_get_first(imod: Option<&mut Imod>) -> Option<&Iobj> {
    let imod = imod?;
    let (mut ob, mut co, mut pt) = (0, 0, 0);
    imod_get_index(imod, &mut ob, &mut co, &mut pt);
    imod_set_index(imod, 0, co, pt);
    imod_object_get(Some(imod))
}

/// Original: `imodObjectGetNext` (`imodel.c:691`).
///
/// Advances the current object index by one in model `imod` and returns the
/// new current object, or `None` if error or if the existing current object is
/// the last one in the model.
pub fn imod_object_get_next(imod: Option<&mut Imod>) -> Option<&Iobj> {
    let imod = imod?;
    let (mut ob, mut co, mut pt) = (0, 0, 0);
    imod_get_index(imod, &mut ob, &mut co, &mut pt);
    ob += 1;
    if ob >= imod.obj.len() as i32 {
        return None;
    }
    imod_set_index(imod, ob, co, pt);
    imod_object_get(Some(imod))
}

/// Original: `imodDelCurrentContour` (`imodel.c:771`).
///
/// Deletes the current contour of the model `imod`.
pub fn imod_del_current_contour(imod: &mut Imod) {
    let index = imod.cindex.contour;
    imod_delete_contour(imod, index);
}

/// Original: `imodDeleteContour` (`imodel.c:783`).
///
/// Deletes the contour at `index` in the current object of the model `mod_`.
/// Returns the size of the current object or -1 for error.
pub fn imod_delete_contour(mod_: &mut Imod, index: i32) -> i32 {
    let ob = mod_.cindex.object;
    if ob < 0 || ob >= mod_.obj.len() as i32 {
        return -1;
    }
    if index < 0 || index >= mod_.obj[ob as usize].cont.len() as i32 {
        return -1;
    }

    /* If contour has any points, free them. */
    {
        let cont = &mut mod_.obj[ob as usize].cont[index as usize];
        cont.pts.clear();
        cont.sizes.clear();
        cont.store.clear();
        /* DNM: need to delete labels if any (`imodel.c:1155`) */
        crate::imod::libimod::ilabel::imod_label_delete(cont.label.take());
    }
    istore_delete_cont_surf(&mut mod_.obj[ob as usize].store, index, 0);

    /* Push extra contours into hole and change contour array to new size */
    mod_.obj[ob as usize].cont.remove(index as usize);

    /* DMN 9/20/04: clean out labels for non-existing surfaces */
    imod_object_clean_surf(&mut mod_.obj[ob as usize]);

    mod_.cindex.contour = -1;
    mod_.cindex.point = -1;
    mod_.obj[ob as usize].cont.len() as i32
}

/// Original: `imodDeleteListOfConts` (`imodel.c:840`).
///
/// Deletes the list of `num_conts` contours in `contours` in the current object
/// of the model `mod_`.  Returns the size of the current object or -1 for
/// error.
pub fn imod_delete_list_of_conts(mod_: &mut Imod, contours: &[i32], mut num_conts: i32) -> i32 {
    let ob = mod_.cindex.object;
    if ob < 0 || ob >= mod_.obj.len() as i32 || num_conts <= 0 {
        return -1;
    }
    for i in 0..num_conts as usize {
        if contours[i] < 0 || contours[i] >= mod_.obj[ob as usize].cont.len() as i32 {
            return -1;
        }
    }

    /* Copy and sort */
    let mut sorted: Vec<i32> = contours[..num_conts as usize].to_vec();
    rs_sort_ints(&mut sorted, num_conts);

    /* Eliminate duplicates */
    let mut ind = 1usize;
    for i in 1..num_conts as usize {
        if sorted[i] != sorted[i - 1] {
            sorted[ind] = sorted[i];
            ind += 1;
        }
    }
    num_conts = ind as i32;

    let mut new_conts: Vec<Icont> = Vec::new();
    if mod_.obj[ob as usize].cont.len() as i32 > num_conts {
        let Some(conts) = imod_contours_new(mod_.obj[ob as usize].cont.len() as i32 - num_conts)
        else {
            return -1;
        };
        new_conts = conts;
    }

    let mut last_del = -1i32;
    let mut out_ind = 0usize;
    for ind in 0..num_conts as usize {
        let cur_del = sorted[ind];

        /* Copy any non-deleted conts over now */
        for i in (last_del + 1)..cur_del {
            let src = mod_.obj[ob as usize].cont[i as usize].clone();
            imod_contour_copy(&src, &mut new_conts[out_ind]);
            out_ind += 1;
        }

        /* If contour has any points, free them. */
        {
            let cont = &mut mod_.obj[ob as usize].cont[cur_del as usize];
            cont.pts.clear();
            cont.sizes.clear();
            cont.store.clear();
        }
        istore_delete_cont_surf(&mut mod_.obj[ob as usize].store, cur_del, 0);
        last_del = cur_del;
    }

    /* Copy final contours if any */
    for i in (last_del + 1)..mod_.obj[ob as usize].cont.len() as i32 {
        let src = mod_.obj[ob as usize].cont[i as usize].clone();
        imod_contour_copy(&src, &mut new_conts[out_ind]);
        out_ind += 1;
    }

    new_conts.truncate(out_ind);
    mod_.obj[ob as usize].cont = new_conts;

    imod_object_clean_surf(&mut mod_.obj[ob as usize]);
    mod_.cindex.contour = -1;
    mod_.cindex.point = -1;
    mod_.obj[ob as usize].cont.len() as i32
}

/// Original: `imodPointGet` (`imodel.c:1177`).
///
/// Returns the current point in model `imod`, or `None` if there is no current
/// contour or current point.
pub fn imod_point_get(imod: &mut Imod) -> Option<&Ipoint> {
    if imod.cindex.point < 0 {
        return None;
    }

    if imod_contour_get(Some(&*imod)).is_none() {
        return None;
    }
    let ob = imod.cindex.object as usize;
    let co = imod.cindex.contour as usize;

    if imod.cindex.point >= imod.obj[ob].cont[co].pts.len() as i32 {
        imod.cindex.point = imod.obj[ob].cont[co].pts.len() as i32 - 1;
    }
    if imod.cindex.point < 0 {
        return None;
    }

    Some(&imod.obj[ob].cont[co].pts[imod.cindex.point as usize])
}

/// Original: `imodPointGetFirst` (`imodel.c:1201`).
///
/// Sets the current point index to 0 in the current contour in model `imod`
/// and returns the first point.
pub fn imod_point_get_first(imod: Option<&mut Imod>) -> Option<&Ipoint> {
    let imod = imod?;
    let (mut ob, mut co, mut pt) = (0, 0, 0);
    imod_get_index(imod, &mut ob, &mut co, &mut pt);
    imod_set_index(imod, ob, co, 0);
    imod_point_get(imod)
}

/// Original: `imodPointGetNext` (`imodel.c:1217`).
///
/// Advances to the next point in the current contour of model `imod` and
/// returns the point, or `None` if there is none.
pub fn imod_point_get_next(imod: Option<&mut Imod>) -> Option<&Ipoint> {
    let imod = imod?;
    let (mut ob, mut co, mut pt) = (0, 0, 0);
    imod_get_index(imod, &mut ob, &mut co, &mut pt);

    let psize = match imod_contour_get(Some(&*imod)) {
        None => return None,
        Some(cont) => cont.pts.len() as i32,
    };
    if psize == 0 {
        return None;
    }

    pt += 1;
    if pt >= psize {
        return None;
    }
    imod_set_index(imod, ob, co, pt);
    imod_point_get(imod)
}

/// Original: `imodTransform` (`imodel.c:1246`).
///
/// Transforms all contour points in model `imod` with the 3D transform in
/// `mat`.  Returns -1 if error.
pub fn imod_transform(imod: Option<&mut Imod>, mat: Option<&Imat>) -> i32 {
    let (Some(imod), Some(mat)) = (imod, mat) else {
        return -1;
    };
    let mut pnt = Ipoint::default();
    for ob in 0..imod.obj.len() {
        for co in 0..imod.obj[ob].cont.len() {
            for pt in 0..imod.obj[ob].cont[co].pts.len() {
                imod_mat_transform(mat, &imod.obj[ob].cont[co].pts[pt], &mut pnt);
                imod.obj[ob].cont[co].pts[pt] = pnt;
            }
        }
    }
    0
}

/// Original: `imodel_transform_slice` (`imodel.c:1274`).
///
/// Transforms all contour points in `model` with Z values that round to
/// `slice`.  The 2D transformation in `mat` is applied to X and Y.  Returns 0.
pub fn imodel_transform_slice(model: &mut Imod, mat: &[f32], slice: i32) -> i32 {
    for ob in 0..model.obj.len() {
        for co in 0..model.obj[ob].cont.len() {
            for pt in 0..model.obj[ob].cont[co].pts.len() {
                let zval = (model.obj[ob].cont[co].pts[pt].z as f64 + 0.5).floor() as i32;
                if zval == slice {
                    let x = model.obj[ob].cont[co].pts[pt].x;
                    let y = model.obj[ob].cont[co].pts[pt].y;
                    model.obj[ob].cont[co].pts[pt].x = (x * mat[0]) + (y * mat[3]) + mat[6];
                    model.obj[ob].cont[co].pts[pt].y = (x * mat[1]) + (y * mat[4]) + mat[7];
                }
            }
        }
    }
    0
}

/// Original: `imodel_model_clean` (`imodel.c:1310`).
///
/// Cleans model `mod_` by removing empty contours, removing empty objects if
/// `keep_empty_objs` is 0, constraining points within the model xmax, ymax and
/// zmax values, and making all points have the same Z value as the first point
/// for non-wild closed contours.  Returns 0.
pub fn imodel_model_clean(mod_: &mut Imod, keep_empty_objs: i32) -> i32 {
    /* push all points inside of boundries. */
    let mut ob = 0i32;
    while ob < mod_.obj.len() as i32 {
        if mod_.obj[ob as usize].cont.is_empty()
            && mod_.obj[ob as usize].mesh.is_empty()
            && keep_empty_objs == 0
        {
            mod_.cindex.object = ob;
            imod_delete_object(mod_, ob);
            ob -= 1;
            ob += 1;
            continue;
        }

        let mut co = 0i32;
        while co < mod_.obj[ob as usize].cont.len() as i32 {
            if mod_.obj[ob as usize].cont[co as usize].pts.is_empty() {
                mod_.cindex.object = ob;
                mod_.cindex.contour = co;
                imod_delete_contour(mod_, co);
                co -= 1;
                co += 1;
                continue;
            }
            for pt in 0..mod_.obj[ob as usize].cont[co as usize].pts.len() {
                if mod_.obj[ob as usize].cont[co as usize].pts[pt].x > mod_.xmax as f32 {
                    mod_.obj[ob as usize].cont[co as usize].pts[pt].x = mod_.xmax as f32;
                }
                if mod_.obj[ob as usize].cont[co as usize].pts[pt].y > mod_.ymax as f32 {
                    mod_.obj[ob as usize].cont[co as usize].pts[pt].y = mod_.ymax as f32;
                }
                if mod_.zmax != 0
                    && mod_.obj[ob as usize].cont[co as usize].pts[pt].z > mod_.zmax as f32
                {
                    mod_.obj[ob as usize].cont[co as usize].pts[pt].z = mod_.zmax as f32;
                }

                if !((mod_.obj[ob as usize].flags & IMOD_OBJFLAG_OPEN) != 0
                    || (mod_.obj[ob as usize].flags & IMOD_OBJFLAG_WILD) != 0
                    || (mod_.obj[ob as usize].flags & IMOD_OBJFLAG_SCAT) != 0)
                {
                    mod_.obj[ob as usize].cont[co as usize].pts[pt].z =
                        mod_.obj[ob as usize].cont[co as usize].pts[0].z;
                }
            }
            co += 1;
        }
        ob += 1;
    }
    0
}

/// Original: `imodUnits` (`imodel.c:1360`).
///
/// Returns a string (e.g., "nm") for the pixel size units of model `mod_`.
pub fn imod_units(mod_: &Imod) -> &'static str {
    let units = mod_.units;
    let retval: &'static str;

    match units {
        IMOD_UNIT_PIXEL => retval = "pixels",
        IMOD_UNIT_KILO => retval = "km",
        IMOD_UNIT_METER => retval = "m",
        IMOD_UNIT_CM => retval = "cm",
        IMOD_UNIT_MM => retval = "mm",
        IMOD_UNIT_UM => retval = "um",
        IMOD_UNIT_NM => retval = "nm",
        IMOD_UNIT_ANGSTROM => retval = "A",
        IMOD_UNIT_PM => retval = "pm",
        _ => retval = "unknown units",
    }
    retval
}

/// Original: `imodChecksum` (`imodel.c:1469`).
///
/// Computes a checksum from the coordinates and most other features of model
/// `imod` and returns the value.
pub fn imod_checksum(imod: &Imod) -> i32 {
    let mut any_pts = 0;
    let debug = if std::env::var_os("IMOD_DEBUG_CHECKSUM").is_none() {
        0
    } else {
        1
    };
    let mut sum = 0.0f64;
    let mut last_sum = 0.0f64;

    sum += imod.zscale as f64;
    sum += imod.pixsize as f64;
    sum += imod.obj.len() as f64;
    sum += imod.blacklevel as f64;
    sum += imod.whitelevel as f64;
    sum += imod.res as f64;
    sum += imod.thresh as f64;
    sum += imod.units as f64;
    sum += imod.pixsize as f64;
    sum += imod.view.len() as f64;
    sum += (imod.flags & !IMODF_FLIPYZ & !IMODF_NEW_TO_3DMOD & !IMODF_ROT90X) as f64;
    if debug != 0 {
        let _ =
            ImodFile::Stderr.write_all(c_format("\ninitial %f\n", &[CArg::Dbl(sum)]).as_bytes());
        last_sum = sum;
    }

    for i in 0..imod.slicer_ang.len() {
        let slanp = &imod.slicer_ang[i];
        let len = slanp
            .label
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(slanp.label.len());
        sum += (slanp.center.x + slanp.center.y + slanp.center.z) as f64
            + (slanp.angles[0] + slanp.angles[0] + slanp.angles[0]) as f64
            + len as f64;
    }
    sum += obj_group_list_checksum(&imod.group_list);
    if debug != 0 {
        let _ =
            ImodFile::Stderr.write_all(c_format("obj group %f\n", &[CArg::Dbl(sum)]).as_bytes());
        last_sum = sum;
    }

    for ob in 0..imod.obj.len() {
        sum += imod_object_checksum(&imod.obj[ob], ob as i32);
        let mut co = 0;
        while co < imod.obj[ob].cont.len() && any_pts == 0 {
            if !imod.obj[ob].cont[co].pts.is_empty() {
                any_pts = 1;
            }
            co += 1;
        }
    }
    if any_pts != 0 {
        sum += istore_checksum(&imod.store);
    }
    if debug != 0 {
        let _ = ImodFile::Stderr.write_all(c_format("object %f\n", &[CArg::Dbl(sum)]).as_bytes());
        last_sum = sum;
    }

    /* Add properties of views.  Do not add rad and trans because they are
    changed just by opening model view window.
    34/27/04: Add the rotation angle only for added views, not for the basic
    view */
    for co in 0..imod.view.len() {
        let view = &imod.view[co];
        sum += (view.fovy + view.cnear + view.cfar) as f64;
        if co != 0 {
            sum += (view.rot.x + view.rot.y + view.rot.z) as f64;
        }
        sum += (view.scale.x + view.scale.y + view.scale.z) as f64;
        sum += (view.world as f32 + view.dcstart + view.dcend + view.plax) as f64;
        sum += (view.lightx + view.lighty) as f64;
        if co != 0 {
            for ob in 0..view.objview.len() {
                let obv = &view.objview[ob];
                let mut osum = ob as f64;
                osum += (obv.red + obv.green + obv.blue) as f64;
                osum += obv.flags as f64;
                osum += obv.pdrawsize as f64;
                osum += obv.linewidth as f64;
                osum += obv.trans as f64;
                osum += (obv.ambient as i32
                    + obv.diffuse as i32
                    + obv.specular as i32
                    + obv.shininess as i32) as f64;
                let clips = &obv.clips;
                osum += (clips.count as i32 + clips.flags as i32 + clips.trans as i32) as f64;
                osum += (clips.plane as u32).wrapping_add(obv.mat2) as f64;
                for i in 0..clips.count as usize {
                    osum += (clips.normal[i].x + clips.normal[i].y + clips.normal[i].z) as f64;
                    osum += (clips.point[i].x + clips.point[i].y + clips.point[i].z) as f64;
                }
                osum += (obv.fillred as i32
                    + obv.fillgreen as i32
                    + obv.fillblue as i32
                    + obv.quality as i32) as f64;
                osum += (obv.valblack as i32
                    + obv.valwhite as i32
                    + obv.matflags2 as i32
                    + obv.mesh_thickness as i32) as f64;
                sum += osum;
            }
        }
    }
    if debug != 0 {
        let _ = ImodFile::Stderr.write_all(c_format("views %f\n", &[CArg::Dbl(sum)]).as_bytes());
        last_sum = sum;
    }
    let _ = last_sum;

    /* This will catch fractional values - not perfect but probably good */
    let mut isum = (sum / 1000000.) as i32;
    isum = (1000. * (sum - 1000000. * isum as f64)) as i32;
    if debug != 0 {
        let _ = ImodFile::Stderr.write_all(
            c_format(
                "checksum = %f %d\n",
                &[CArg::Dbl(sum), CArg::Int(isum as i64)],
            )
            .as_bytes(),
        );
    }
    isum
}

/// Original: `imodRot90X` (`imodel.c:1630`).
///
/// Rotates the model in `imod` by -90 degrees about X if `to_native` is 0, or
/// by +90 degrees if `to_native` is non-zero.
pub fn imod_rot90x(imod: &mut Imod, to_native: i32) {
    let yconst: f32;
    let yfac: f32;
    let zconst: f32;
    let zfac: f32;

    /* Set up the constants and multipliers from getting between Y and Z */
    if to_native != 0 {
        yconst = -0.5;
        yfac = 1.;
        zconst = imod.zmax as f32 - 0.5;
        zfac = -1.;
    } else {
        zconst = 0.5;
        zfac = 1.;
        yconst = imod.ymax as f32 - 0.5;
        yfac = -1.;
    }

    /* Rotate the contours */
    for ob in 0..imod.obj.len() {
        for co in 0..imod.obj[ob].cont.len() {
            for pt in 0..imod.obj[ob].cont[co].pts.len() {
                let tmp = yconst + yfac * imod.obj[ob].cont[co].pts[pt].y;
                imod.obj[ob].cont[co].pts[pt].y = zconst + zfac * imod.obj[ob].cont[co].pts[pt].z;
                imod.obj[ob].cont[co].pts[pt].z = tmp;
            }
        }

        /* Rotate object clip planes and meshes */
        rot_clips(yconst, yfac, zconst, zfac, &mut imod.obj[ob].clips);
        for co in 0..imod.obj[ob].mesh.len() {
            let mut pt = 0usize;
            while pt < imod.obj[ob].mesh[co].vert.len() {
                let tmp = yconst + yfac * imod.obj[ob].mesh[co].vert[pt].y;
                imod.obj[ob].mesh[co].vert[pt].y = zconst + zfac * imod.obj[ob].mesh[co].vert[pt].z;
                imod.obj[ob].mesh[co].vert[pt].z = tmp;
                let tmp = yfac * imod.obj[ob].mesh[co].vert[pt + 1].y;
                imod.obj[ob].mesh[co].vert[pt + 1].y = zfac * imod.obj[ob].mesh[co].vert[pt + 1].z;
                imod.obj[ob].mesh[co].vert[pt + 1].z = tmp;
                pt += 2;
            }
        }
    }

    /* Rotate the view clip planes */
    for i in 0..imod.view.len() {
        rot_clips(yconst, yfac, zconst, zfac, &mut imod.view[i].clips);
        for ob in 0..imod.view[i].objview.len() {
            rot_clips(
                yconst,
                yfac,
                zconst,
                zfac,
                &mut imod.view[i].objview[ob].clips,
            );
        }
    }

    std::mem::swap(&mut imod.ymax, &mut imod.zmax);
}

/// Original: `imodInvertZ` (`imodel.c:1695`).
///
/// Inverts the model in `imod` in the Z direction and shifts it in Z to retain
/// positive values.
pub fn imod_invert_z(imod: &mut Imod) {
    let zconst = imod.zmax as f32 - 1.;

    /* Invert contours and meshes */
    for ob in 0..imod.obj.len() {
        for co in 0..imod.obj[ob].cont.len() {
            for pt in 0..imod.obj[ob].cont[co].pts.len() {
                imod.obj[ob].cont[co].pts[pt].z = zconst - imod.obj[ob].cont[co].pts[pt].z;
            }
        }

        invert_clips(zconst, &mut imod.obj[ob].clips);
        for co in 0..imod.obj[ob].mesh.len() {
            let mut pt = 0usize;
            while pt < imod.obj[ob].mesh[co].vert.len() {
                imod.obj[ob].mesh[co].vert[pt].z = zconst - imod.obj[ob].mesh[co].vert[pt].z;
                imod.obj[ob].mesh[co].vert[pt + 1].z = -imod.obj[ob].mesh[co].vert[pt + 1].z;
                pt += 2;
            }

            /* Inverting a triangle changes the order of vertices and makes the
            normal point the wrong way, so have to rearrange triangle indices
            too */
            let mut i = 0usize;
            while i < imod.obj[ob].mesh[co].list.len() {
                match imod.obj[ob].mesh[co].list[i] {
                    /* Swap first two indexes for vertex and normal for old meshes */
                    IMOD_MESH_BGNPOLYNORM => {
                        i += 1;
                        while imod.obj[ob].mesh[co].list[i] != IMOD_MESH_ENDPOLY {
                            let mlist = &mut imod.obj[ob].mesh[co].list;
                            let pt = mlist[i];
                            mlist[i] = mlist[i + 1];
                            mlist[i + 1] = pt;
                            let pt = mlist[i + 3];
                            mlist[i + 3] = mlist[i + 4];
                            mlist[i + 4] = pt;
                            i += 6;
                        }
                    }

                    /* Swap first two indexes of the three for new meshes */
                    IMOD_MESH_BGNPOLYNORM2 => {
                        i += 1;
                        while imod.obj[ob].mesh[co].list[i] != IMOD_MESH_ENDPOLY {
                            let mlist = &mut imod.obj[ob].mesh[co].list;
                            let pt = mlist[i];
                            mlist[i] = mlist[i + 1];
                            mlist[i + 1] = pt;
                            i += 3;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
    }

    /* Invert view clip planes */
    for i in 0..imod.view.len() {
        invert_clips(zconst, &mut imod.view[i].clips);
        for ob in 0..imod.view[i].objview.len() {
            invert_clips(zconst, &mut imod.view[i].objview[ob].clips);
        }
    }
}

/// Original: `imodSetRefImage` (`imodel.c:1770`).
///
/// Sets the `IrefImage` structure `ref_image` in `imod` with the image
/// coordinate information in the MRC header `hdata`.  Returns 1 for memory
/// error allocating the structure.
pub fn imod_set_ref_image(imod: &mut Imod, hdata: &MrcHeader) -> i32 {
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
    let r = imod.ref_image.as_mut().unwrap();
    r.ctrans.x = hdata.xorg;
    r.ctrans.y = hdata.yorg;
    r.ctrans.z = hdata.zorg;
    r.crot.x = hdata.tiltangles[3];
    r.crot.y = hdata.tiltangles[4];
    r.crot.z = hdata.tiltangles[5];
    r.cscale.x = 1.;
    r.cscale.y = 1.;
    r.cscale.z = 1.;
    if hdata.xlen != 0. && hdata.mx != 0 {
        r.cscale.x = hdata.xlen / hdata.mx as f32;
    }
    if hdata.ylen != 0. && hdata.my != 0 {
        r.cscale.y = hdata.ylen / hdata.my as f32;
    }
    /* The source tests `hdata->xlen` here, not `zlen` */
    if hdata.xlen != 0. && hdata.mz != 0 {
        r.cscale.z = hdata.zlen / hdata.mz as f32;
    }
    r.oscale.x = 1.;
    r.oscale.y = 1.;
    r.oscale.z = 1.;
    r.orot.x = 0.;
    r.orot.y = 0.;
    r.orot.z = 0.;

    /* 3/16/22: otrans needs to be set too for transformations to work */
    /* 5/19/22: And the flag needs to be set for rotations to be handled right */
    r.otrans = r.ctrans;
    imod.flags |= IMODF_TILTOK;
    0
}

/// Original: `imodTransForSubsetLoad` (`imodel.c:1808`).
///
/// Shifts the model `imod` back to full volume coordinates if it was saved
/// after being loaded on a subset of the image whose header is in `hdata`, and
/// also shifts it to the coordinates of a subset currently being loaded.
pub fn imod_trans_for_subset_load(imod: &mut Imod, hdata: &MrcHeader, li: Option<&LoadInfo>) {
    let mut xload = 0.0f32;
    let mut yload = 0.0f32;
    let mut zload = 0.0f32;

    if let Some(r) = imod.ref_image {
        if r.cscale.x != 0. && r.cscale.y != 0. && r.cscale.z != 0. {
            xload = (hdata.xorg - r.ctrans.x) / r.cscale.x;
            yload = (hdata.yorg - r.ctrans.y) / r.cscale.y;
            zload = (hdata.zorg - r.ctrans.z) / r.cscale.z;
        }
    }
    if let Some(li) = li {
        xload -= li.xmin as f32;
        yload -= li.ymin as f32;
        zload -= li.zmin as f32;
    }
    if xload != 0. || yload != 0. || zload != 0. {
        for ob in 0..imod.obj.len() {
            for co in 0..imod.obj[ob].cont.len() {
                for pt in 0..imod.obj[ob].cont[co].pts.len() {
                    imod.obj[ob].cont[co].pts[pt].x += xload;
                    imod.obj[ob].cont[co].pts[pt].y += yload;
                    imod.obj[ob].cont[co].pts[pt].z += zload;
                }
            }
        }
    }
}

/// Original: `imodTransToMatchImage` (`imodel.c:1957`).
///
/// Transforms the model in `imod` to match the image coordinates specified in
/// the MRC header `hdata`, ignoring a change in tilt angles if
/// `ignore_angles` is nonzero.  Returns -1 if the model has no `IrefImage`
/// structure, or 1 for a memory error in `imodTransFromRefImage`.
pub fn imod_trans_to_match_image(imod: &mut Imod, hdata: &MrcHeader, ignore_angles: i32) -> i32 {
    let unit_pt = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let Some(existing) = imod.ref_image else {
        return -1;
    };
    let mut use_ref = Iref_image {
        oscale: Ipoint::default(),
        otrans: Ipoint::default(),
        orot: Ipoint::default(),
        cscale: Ipoint::default(),
        ctrans: Ipoint::default(),
        crot: Ipoint::default(),
    };

    /* get the target transformation */
    use_ref.ctrans.x = hdata.xorg;
    use_ref.ctrans.y = hdata.yorg;
    use_ref.ctrans.z = hdata.zorg;
    if ignore_angles != 0 {
        use_ref.crot = existing.crot;
    } else {
        use_ref.crot.x = hdata.tiltangles[3];
        use_ref.crot.y = hdata.tiltangles[4];
        use_ref.crot.z = hdata.tiltangles[5];
        imod.flags |= IMODF_TILTOK;
    }
    use_ref.cscale = unit_pt;
    if hdata.xlen != 0. && hdata.mx != 0 {
        use_ref.cscale.x = hdata.xlen / hdata.mx as f32;
    }
    if hdata.ylen != 0. && hdata.my != 0 {
        use_ref.cscale.y = hdata.ylen / hdata.my as f32;
    }
    if hdata.zlen != 0. && hdata.mz != 0 {
        use_ref.cscale.z = hdata.zlen / hdata.mz as f32;
    }

    use_ref.otrans = existing.ctrans;
    use_ref.orot = existing.crot;
    use_ref.oscale = existing.cscale;
    imod_trans_from_ref_image(imod, &use_ref, unit_pt)
}

/// Original: `exchangef` (`imodel.c:2043`, static).
///
/// Exchange floats.
fn exchangef(a: &mut f32, b: &mut f32) {
    let tmp = *a;
    *a = *b;
    *b = tmp;
}

/// Original: `imodTransModel3D` (`imodel.c:2079`).
///
/// Transforms `model` with the 3D matrix in `mat`, which includes
/// translations.  `norm_mat` is a 3D matrix for normal transformations, or
/// `None` if none are needed.  `new_cen` are the center coordinates of the
/// volume being transformed to.  `zscale` is the Z scale factor.  `doflip`
/// should be nonzero if the model is in Y-Z flipped coordinates.
pub fn imod_trans_model3d(
    model: &mut Imod,
    mat: &mut Imat,
    norm_mat: Option<&mut Imat>,
    mut new_cen: Ipoint,
    zscale: f32,
    doflip: i32,
) {
    let mut mat_work = imod_mat_new(3).unwrap();
    let mut mat_work2 = imod_mat_new(3).unwrap();
    let mut mat_use = imod_mat_new(3).unwrap();
    let mut clip_mat = imod_mat_new(3).unwrap();
    let mut tmp_pt = Ipoint::default();
    let old_cen = Ipoint {
        x: -model.xmax as f32 * 0.5f32,
        y: -model.ymax as f32 * 0.5f32,
        z: -(model.zmax as f32 * 0.5f32 - 0.5f32),
    };
    let mut norm_mat = norm_mat;

    /* If data are flipped, exchange Y and Z columns then Y and Z rows */
    if doflip != 0 {
        let (a, b) = mat.data.split_at_mut(8);
        exchangef(&mut a[4], &mut b[0]);
        let (a, b) = mat.data.split_at_mut(2);
        exchangef(&mut a[1], &mut b[0]);
        let (a, b) = mat.data.split_at_mut(9);
        exchangef(&mut a[6], &mut b[0]);
        let (a, b) = mat.data.split_at_mut(10);
        exchangef(&mut a[5], &mut b[0]);
        let (a, b) = mat.data.split_at_mut(14);
        exchangef(&mut a[13], &mut b[0]);
        if let Some(nm) = norm_mat.as_deref_mut() {
            let (a, b) = nm.data.split_at_mut(8);
            exchangef(&mut a[4], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(2);
            exchangef(&mut a[1], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(9);
            exchangef(&mut a[6], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(10);
            exchangef(&mut a[5], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(14);
            exchangef(&mut a[13], &mut b[0]);
        }
    }

    /* Compute a transformation by translating to origin, applying the given
    transform, then translating back to new center.
    Apply the zscale after translating, then de-scale after applying the
    transform. */
    imod_mat_trans(&mut mat_work, &old_cen);
    tmp_pt.x = 1.;
    tmp_pt.y = 1.;
    tmp_pt.z = zscale;
    imod_mat_scale(&mut mat_work, &tmp_pt);
    imod_mat_mult(&mat_work, mat, &mut mat_use);
    tmp_pt.z = 1. / zscale;
    imod_mat_scale(&mut mat_use, &tmp_pt);
    new_cen.z -= 0.5f32;
    imod_mat_trans(&mut mat_use, &new_cen);

    /* If no normal transform supplied, set up transform for normals as copy of
    original transform, no shifts, then take the inverse and transpose it. */
    let mut owned_norm: Imat;
    let norm_use: &Imat = match norm_mat {
        Some(nm) => nm,
        None => {
            imod_mat_copy(mat, &mut mat_work);
            mat_work.data[12] = 0.;
            mat_work.data[13] = 0.;
            mat_work.data[14] = 0.;
            mat_work.data[15] = 1.;
            let mut nm = imod_mat_inverse(&mat_work).unwrap();
            let (a, b) = nm.data.split_at_mut(4);
            exchangef(&mut a[1], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(8);
            exchangef(&mut a[2], &mut b[0]);
            let (a, b) = nm.data.split_at_mut(9);
            exchangef(&mut a[6], &mut b[0]);
            owned_norm = nm;
            &owned_norm
        }
    };

    /* The mesh normals already contain the Z scaling, but the clip normals
    require a matrix that is prescaled by 1/zscale and post-scaled by
    z-scale */
    imod_mat_scale(&mut mat_work2, &tmp_pt);
    imod_mat_mult(&mat_work2, norm_use, &mut clip_mat);
    tmp_pt.z = zscale;
    imod_mat_scale(&mut clip_mat, &tmp_pt);

    imod_trans_from_mats(model, &mat_use, norm_use, &clip_mat);

    imod_mat_delete(&mut mat_work);
    imod_mat_delete(&mut mat_work2);
    imod_mat_delete(&mut mat_use);
    imod_mat_delete(&mut clip_mat);
}

/// Original: `flipClips` (`imodel.c:2148`, static).
///
/// Flip clipping planes.
fn flip_clips(clips: &mut Iclip_planes) {
    for i in 0..clips.count as usize {
        let tmp = clips.point[i].y;
        clips.point[i].y = clips.point[i].z;
        clips.point[i].z = tmp;
        let tmp = clips.normal[i].y;
        clips.normal[i].y = clips.normal[i].z;
        clips.normal[i].z = tmp;
    }
}

/// Original: `rotClips` (`imodel.c:2159`, static).
///
/// Rotate clipping planes by 90 with given factors.
fn rot_clips(yconst: f32, yfac: f32, zconst: f32, zfac: f32, clips: &mut Iclip_planes) {
    /* Clipping plane point is the negative of an actual point so need to take
    negative before and after */
    for i in 0..clips.count as usize {
        let tmp = -yconst + yfac * clips.point[i].y;
        clips.point[i].y = -zconst + zfac * clips.point[i].z;
        clips.point[i].z = tmp;
        let tmp = yfac * clips.normal[i].y;
        clips.normal[i].y = zfac * clips.normal[i].z;
        clips.normal[i].z = tmp;
    }
}

/// Original: `invertClips` (`imodel.c:2178`, static).
///
/// Invert clipping planes in Z.
fn invert_clips(zconst: f32, clips: &mut Iclip_planes) {
    for i in 0..clips.count as usize {
        clips.point[i].z = -zconst - clips.point[i].z;
        clips.normal[i].z = -clips.normal[i].z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_accessors_preserve_source_hierarchical_clamping() {
        let mut model = Imod {
            obj: vec![Iobj {
                cont: vec![
                    Icont {
                        pts: vec![Ipoint::default(), Ipoint::default()],
                        ..Icont::default()
                    },
                    Icont::default(),
                ],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_set_index(&mut model, 8, 8, 8);
        let (mut object, mut contour, mut point) = (0, 0, 0);
        imod_get_index(&model, &mut object, &mut contour, &mut point);
        assert_eq!((object, contour, point), (0, 1, -1));
        imod_set_index(&mut model, -1, 1, 1);
        imod_get_index(&model, &mut object, &mut contour, &mut point);
        assert_eq!((object, contour, point), (-1, -1, -1));
    }

    #[test]
    fn reference_transform_and_yz_flip_cover_contours_and_meshes() {
        let mut model = Imod {
            ymax: 20,
            zmax: 30,
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint {
                        x: 2.,
                        y: 3.,
                        z: 4.,
                    }],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    // `imodTransFromMats` (`imodel.c:2021-2029`) walks the
                    // vertex list in vertex/normal pairs.
                    vert: vec![
                        Ipoint {
                            x: 5.,
                            y: 6.,
                            z: 7.,
                        },
                        Ipoint {
                            x: 0.,
                            y: 0.,
                            z: 2.,
                        },
                    ],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_flip_yz(&mut model);
        assert_eq!(
            model.obj[0].cont[0].pts[0],
            Ipoint {
                x: 2.,
                y: 4.,
                z: 3.
            }
        );
        assert_eq!((model.ymax, model.zmax), (30, 20));
        let reference = Iref_image {
            oscale: Ipoint {
                x: 2.,
                y: 2.,
                z: 2.,
            },
            otrans: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            cscale: Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            ctrans: Ipoint {
                x: 3.,
                y: 3.,
                z: 3.,
            },
            ..Iref_image::default()
        };
        assert_eq!(
            imod_trans_from_ref_image(
                &mut model,
                &reference,
                Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                }
            ),
            0
        );
        assert_eq!(
            model.obj[0].cont[0].pts[0],
            Ipoint {
                x: 6.,
                y: 10.,
                z: 8.5
            }
        );
        // Composed matrix is x' = 2x + 2, y' = 2y + 2, z' = 2z + 2.5, so the
        // flipped mesh vertex (5, 7, 6) maps to (12, 16, 14.5).
        assert_eq!(
            model.obj[0].mesh[0].vert[0],
            Ipoint {
                x: 12.,
                y: 16.,
                z: 14.5
            }
        );
        // The normal goes through matNorm (0.5 on the diagonal) and is then
        // renormalized, so the flipped normal (0, 2, 0) becomes (0, 1, 0).
        assert_eq!(
            model.obj[0].mesh[0].vert[1],
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.
            }
        );
    }

    #[test]
    fn imodel_dist_uses_current_and_previous_contour_point_xy_only() {
        let mut model = Imod::default();
        let mut object = Iobj::default();
        let mut contour = Icont::default();
        contour.pts = vec![
            Ipoint {
                x: 1.,
                y: 2.,
                z: 100.,
            },
            Ipoint {
                x: 4.,
                y: 6.,
                z: -100.,
            },
        ];
        object.cont.push(contour);
        model.obj.push(object);
        model.cindex = Iindex {
            object: 0,
            contour: 0,
            point: 1,
        };
        assert_eq!(imodel_dist(&model), 5.);
        model.cindex.point = 0;
        assert_eq!(imodel_dist(&model), 0.);
    }
}

/// Differential harness for the model-level `imodel.c` group -- defaults,
/// units, checksum, index accessors, whole-model transforms, the 90-degree X
/// rotation and Z inversion, cleaning, object/contour deletion and the
/// reference-image transforms -- against a driver compiled directly against
/// the pinned source and linked to the reference build's `libimod`.
#[cfg(test)]
mod source_driver_model {
    use super::*;
    use crate::imod::libimod::icont::imod_contour_new;
    use crate::imod::libimod::iobj::imod_object_add_contour;
    use crate::imod::libimod::ipoint::imod_point_append;

    fn g9(v: f64) -> String {
        crate::imod::libcfshr::b3dutil::c_format(
            "%.9g",
            &[crate::imod::libcfshr::b3dutil::CArg::Dbl(v)],
        )
    }

    fn dumpmodel(out: &mut String, tag: &str, m: &Imod) {
        out.push_str(&format!(
            "{tag} objs={} views={} cindex {} {} {}\n",
            m.obj.len(),
            m.view.len(),
            m.cindex.object,
            m.cindex.contour,
            m.cindex.point
        ));
        out.push_str(&format!(
            "{tag} max {} {} {} flags={}\n",
            m.xmax, m.ymax, m.zmax, m.flags
        ));
        for ob in 0..m.obj.len() {
            let o = &m.obj[ob];
            out.push_str(&format!(
                "{tag} ob {ob} conts={} meshes={} surfsize={} flags={} pdraw={} rgb {} {} {}\n",
                o.cont.len(),
                o.mesh.len(),
                o.surfsize,
                o.flags,
                o.pdrawsize,
                g9(o.red as f64),
                g9(o.green as f64),
                g9(o.blue as f64)
            ));
            out.push_str(&format!(
                "{tag} ob {ob} clips {} {} {} {}\n",
                o.clips.count, o.clips.flags, o.clips.trans, o.clips.plane
            ));
            for i in 0..o.clips.count as usize {
                out.push_str(&format!(
                    "{tag} ob {ob} cl {i} {} {} {} {} {} {}\n",
                    g9(o.clips.normal[i].x as f64),
                    g9(o.clips.normal[i].y as f64),
                    g9(o.clips.normal[i].z as f64),
                    g9(o.clips.point[i].x as f64),
                    g9(o.clips.point[i].y as f64),
                    g9(o.clips.point[i].z as f64)
                ));
            }
            for co in 0..o.cont.len() {
                let c = &o.cont[co];
                out.push_str(&format!(
                    "{tag} ob {ob} co {co} n={} flags={} surf={} time={}\n",
                    c.pts.len(),
                    c.flags,
                    c.surf,
                    c.time
                ));
                for pt in 0..c.pts.len() {
                    out.push_str(&format!(
                        "{tag} ob {ob} co {co} pt {pt} {} {} {}\n",
                        g9(c.pts[pt].x as f64),
                        g9(c.pts[pt].y as f64),
                        g9(c.pts[pt].z as f64)
                    ));
                }
            }
            for co in 0..o.mesh.len() {
                let me = &o.mesh[co];
                out.push_str(&format!(
                    "{tag} ob {ob} me {co} v={} l={}\n",
                    me.vert.len(),
                    me.list.len()
                ));
                for pt in 0..me.vert.len() {
                    out.push_str(&format!(
                        "{tag} ob {ob} me {co} v {pt} {} {} {}\n",
                        g9(me.vert[pt].x as f64),
                        g9(me.vert[pt].y as f64),
                        g9(me.vert[pt].z as f64)
                    ));
                }
                for pt in 0..me.list.len() {
                    out.push_str(&format!("{tag} ob {ob} me {co} l {pt} {}\n", me.list[pt]));
                }
            }
        }
        for i in 0..m.view.len() {
            let v = &m.view[i];
            out.push_str(&format!(
                "{tag} vw {i} objv={} clips {} {} {} {}\n",
                v.objview.len(),
                v.clips.count,
                v.clips.flags,
                v.clips.trans,
                v.clips.plane
            ));
            for co in 0..v.clips.count as usize {
                out.push_str(&format!(
                    "{tag} vw {i} cl {co} {} {} {} {} {} {}\n",
                    g9(v.clips.normal[co].x as f64),
                    g9(v.clips.normal[co].y as f64),
                    g9(v.clips.normal[co].z as f64),
                    g9(v.clips.point[co].x as f64),
                    g9(v.clips.point[co].y as f64),
                    g9(v.clips.point[co].z as f64)
                ));
            }
            for ob in 0..v.objview.len() {
                let ov = &v.objview[ob];
                out.push_str(&format!(
                    "{tag} vw {i} ov {ob} flags={} rgb {} {} {} pdraw={} clips {} {} {} {}\n",
                    ov.flags,
                    g9(ov.red as f64),
                    g9(ov.green as f64),
                    g9(ov.blue as f64),
                    ov.pdrawsize,
                    ov.clips.count,
                    ov.clips.flags,
                    ov.clips.trans,
                    ov.clips.plane
                ));
                for co in 0..ov.clips.count as usize {
                    out.push_str(&format!(
                        "{tag} vw {i} ov {ob} cl {co} {} {} {} {} {} {}\n",
                        g9(ov.clips.normal[co].x as f64),
                        g9(ov.clips.normal[co].y as f64),
                        g9(ov.clips.normal[co].z as f64),
                        g9(ov.clips.point[co].x as f64),
                        g9(ov.clips.point[co].y as f64),
                        g9(ov.clips.point[co].z as f64)
                    ));
                }
            }
        }
    }

    fn addcont(m: &mut Imod, ob: usize, xy: &[f32], n: usize, z: f32) {
        let mut c = imod_contour_new().unwrap();
        for i in 0..n {
            imod_point_append(
                &mut c,
                Ipoint {
                    x: xy[2 * i],
                    y: xy[2 * i + 1],
                    z,
                },
            );
        }
        imod_object_add_contour(&mut m.obj[ob], c);
    }

    fn setclips(cl: &mut Iclip_planes, count: usize) {
        imod_clips_initialize(cl);
        cl.count = count as u8;
        cl.flags = 3;
        cl.trans = 7;
        cl.plane = 1;
        for i in 0..count {
            cl.normal[i].x = 0.25 * (i + 1) as f32;
            cl.normal[i].y = -0.5 - i as f32;
            cl.normal[i].z = 1.5 - 0.5 * i as f32;
            cl.point[i].x = -2. - i as f32;
            cl.point[i].y = 3. + 2. * i as f32;
            cl.point[i].z = -4. + i as f32;
        }
    }

    #[test]
    fn source_c_driver_differential_model() {
        let sq = [0.0f32, 0., 10., 0., 10., 10., 0., 10.];
        let tri = [2.0f32, 3., 20., 5., 11., 25.];
        let un = [0.0f32, 3., 1., -2., -3., -6., -9.];
        let mut out = String::new();
        let mut m = imod_new().unwrap();

        out.push_str("--- default ---\n");
        {
            let name: String = m.name[..13]
                .iter()
                .map(|b| *b as u8 as char)
                .collect::<String>();
            out.push_str(&format!("name {name}\n"));
        }
        out.push_str(&format!(
            "flags {} draw {} mouse {} bl {} wl {}\n",
            m.flags, m.drawmode, m.mousemode, m.blacklevel, m.whitelevel
        ));
        out.push_str(&format!(
            "off {} {} {} scale {} {} {}\n",
            g9(m.xoffset as f64),
            g9(m.yoffset as f64),
            g9(m.zoffset as f64),
            g9(m.xscale as f64),
            g9(m.yscale as f64),
            g9(m.zscale as f64)
        ));
        out.push_str(&format!(
            "cindex {} {} {} res {} thresh {} pix {} units {} csum {}\n",
            m.cindex.object,
            m.cindex.contour,
            m.cindex.point,
            m.res,
            m.thresh,
            g9(m.pixsize as f64),
            m.units,
            m.csum
        ));
        out.push_str(&format!(
            "max {} {} {} abg {} {} {} views {} cview {} cms {}\n",
            m.xmax,
            m.ymax,
            m.zmax,
            g9(m.alpha as f64),
            g9(m.beta as f64),
            g9(m.gamma as f64),
            m.view.len(),
            m.cview,
            m.cur_mesh_surf
        ));

        out.push_str("--- units ---\n");
        for i in 0..7 {
            m.units = un[i] as i32;
            out.push_str(&format!("u {} {}\n", m.units, imod_units(&m)));
        }
        m.units = 42;
        out.push_str(&format!("u {} {}\n", m.units, imod_units(&m)));
        m.units = -10;
        out.push_str(&format!("u {} {}\n", m.units, imod_units(&m)));
        m.units = -12;
        out.push_str(&format!("u {} {}\n", m.units, imod_units(&m)));
        m.units = 0;

        out.push_str("--- build ---\n");
        m.xmax = 100;
        m.ymax = 80;
        m.zmax = 20;
        m.pixsize = 1.75;
        m.zscale = 1.25;
        m.res = 3;
        m.thresh = 128;
        imod_new_object(&mut m);
        imod_new_object(&mut m);
        imod_new_object(&mut m);
        addcont(&mut m, 0, &sq, 4, 3.);
        addcont(&mut m, 0, &tri, 3, 5.);
        addcont(&mut m, 1, &sq, 4, 7.);
        addcont(&mut m, 2, &sq, 0, 0.);
        m.obj[0].surfsize = 2;
        m.obj[1].flags |= IMOD_OBJFLAG_OPEN;
        let mut cl = m.obj[0].clips.clone();
        setclips(&mut cl, 2);
        m.obj[0].clips = cl;
        let mut cl = m.view[0].clips.clone();
        setclips(&mut cl, 1);
        m.view[0].clips = cl;
        {
            let o = &mut m.obj[1];
            let mut mesh = Imesh::default();
            for i in 0..6 {
                mesh.vert.push(Ipoint {
                    x: i as f32 * 1.5,
                    y: 10. - i as f32,
                    z: 2. + 0.5 * i as f32,
                });
            }
            mesh.list = vec![
                IMOD_MESH_BGNPOLYNORM2,
                0,
                1,
                2,
                3,
                4,
                5,
                IMOD_MESH_ENDPOLY,
                IMOD_MESH_END,
            ];
            o.mesh.push(mesh);
        }
        /* second view with object views: `imodViewDefault` (`iview.c:42`) then
        `imodObjviewComplete` (`iview.c:519`) with the `imodObjviewFromObject`
        (`iview.c:438`) copy inlined, as in `imod_move_object`. */
        {
            let mut vw = Iview::default();
            imod_clips_initialize(&mut vw.clips);
            m.view.push(vw);
            for iv in 1..m.view.len() {
                for j in m.view[iv].objview.len()..m.obj.len() {
                    let obj = &m.obj[j];
                    let objview = Iobjview {
                        flags: obj.flags,
                        red: obj.red,
                        green: obj.green,
                        blue: obj.blue,
                        pdrawsize: obj.pdrawsize,
                        linewidth: obj.linewidth,
                        linesty: obj.linesty,
                        trans: obj.trans,
                        clips: obj.clips.clone(),
                        ambient: obj.ambient,
                        diffuse: obj.diffuse,
                        specular: obj.specular,
                        shininess: obj.shininess,
                        fillred: obj.fillred,
                        fillgreen: obj.fillgreen,
                        fillblue: obj.fillblue,
                        quality: obj.quality,
                        mat2: obj.mat2,
                        valblack: obj.valblack,
                        valwhite: obj.valwhite,
                        matflags2: obj.matflags2,
                        mesh_thickness: obj.mesh_thickness,
                    };
                    m.view[iv].objview.push(objview);
                }
            }
            let mut cl = m.view[1].objview[0].clips.clone();
            setclips(&mut cl, 2);
            m.view[1].objview[0].clips = cl;
            m.view[1].objview[1].pdrawsize = 9;
        }
        dumpmodel(&mut out, "b", &m);

        out.push_str("--- checksum ---\n");
        out.push_str(&format!("csum {}\n", imod_checksum(&m)));
        out.push_str(&format!(
            "maxtime {} maxobj {} zscale {} pixsize {} flipped {}\n",
            imod_get_max_time(Some(&m)),
            imod_get_max_object(&m),
            g9(imod_get_z_scale(&m) as f64),
            g9(imod_get_pixel_size(&m) as f64),
            imod_get_flipped(&m)
        ));

        out.push_str("--- getters ---\n");
        {
            imod_set_index(&mut m, 0, 0, 0);
            let found = imod_object_get_first(Some(&mut m)).is_some();
            out.push_str(&format!(
                "first {}\n",
                if found { m.cindex.object } else { -99 }
            ));
            for _ in 0..3 {
                let found = imod_object_get_next(Some(&mut m)).is_some();
                out.push_str(&format!(
                    "next {}\n",
                    if found { m.cindex.object } else { -99 }
                ));
            }
            imod_set_index(&mut m, 0, 1, 0);
            let p = *imod_point_get_first(Some(&mut m)).unwrap();
            out.push_str(&format!(
                "pf {} {} {} {}\n",
                m.cindex.point,
                g9(p.x as f64),
                g9(p.y as f64),
                g9(p.z as f64)
            ));
            loop {
                let Some(p) = imod_point_get_next(Some(&mut m)) else {
                    break;
                };
                let p = *p;
                out.push_str(&format!(
                    "pn {} {} {} {}\n",
                    m.cindex.point,
                    g9(p.x as f64),
                    g9(p.y as f64),
                    g9(p.z as f64)
                ));
            }
            let n = match imod_contour_get(Some(&m)) {
                Some(c) => c.pts.len() as i32,
                None => -1,
            };
            out.push_str(&format!("cg {n}\n"));
            out.push_str(&format!("pg {}\n", imod_point_get(&mut m).is_some() as i32));
            imod_set_index(&mut m, 0, 0, 0);
        }

        out.push_str("--- transform ---\n");
        {
            let mut mat = imod_mat_new(3).unwrap();
            imod_mat_rot(&mut mat, 20., B3D_Z);
            let t = Ipoint {
                x: 3.,
                y: -2.,
                z: 1.,
            };
            imod_mat_trans(&mut mat, &t);
            out.push_str(&format!(
                "tr {}\n",
                imod_transform(Some(&mut m), Some(&mat))
            ));
            dumpmodel(&mut out, "tr", &m);
            imod_mat_delete(&mut mat);
        }
        {
            let m2 = [2.0f32, 0., 0., 0., 3., 0., 5., 6., 1.];
            out.push_str(&format!("ts {}\n", imodel_transform_slice(&mut m, &m2, 3)));
            dumpmodel(&mut out, "ts", &m);
        }

        out.push_str("--- rot90x/invertz ---\n");
        imod_rot90x(&mut m, 0);
        dumpmodel(&mut out, "r0", &m);
        imod_rot90x(&mut m, 1);
        dumpmodel(&mut out, "r1", &m);
        imod_invert_z(&mut m);
        dumpmodel(&mut out, "iz", &m);

        out.push_str("--- clean/delete ---\n");
        out.push_str(&format!("clean {}\n", imodel_model_clean(&mut m, 0)));
        dumpmodel(&mut out, "cl", &m);
        imod_set_index(&mut m, 0, -1, -1);
        out.push_str(&format!("dc {}\n", imod_delete_contour(&mut m, 0)));
        dumpmodel(&mut out, "dc", &m);
        {
            let lst = [0i32, 0i32];
            imod_set_index(&mut m, 0, -1, -1);
            out.push_str(&format!(
                "dlc {}\n",
                imod_delete_list_of_conts(&mut m, &lst, 2)
            ));
        }
        dumpmodel(&mut out, "dlc", &m);
        out.push_str(&format!("mv {}\n", imod_move_object(&mut m, 0, 1)));
        dumpmodel(&mut out, "mv", &m);
        out.push_str(&format!("do {}\n", imod_delete_object(&mut m, 0)));
        dumpmodel(&mut out, "do", &m);
        out.push_str(&format!("do {}\n", imod_delete_object(&mut m, 0)));
        dumpmodel(&mut out, "do2", &m);

        out.push_str("--- refimage ---\n");
        {
            let mut m2 = imod_new().unwrap();
            imod_new_object(&mut m2);
            addcont(&mut m2, 0, &sq, 4, 3.);
            m2.xmax = 100;
            m2.ymax = 80;
            m2.zmax = 20;
            let mut h: MrcHeader = MrcHeader::default();
            h.xorg = 5.;
            h.yorg = -3.;
            h.zorg = 2.;
            h.xlen = 200.;
            h.ylen = 160.;
            h.zlen = 40.;
            h.mx = 100;
            h.my = 80;
            h.mz = 20;
            h.tiltangles[3] = 1.;
            h.tiltangles[4] = 2.;
            h.tiltangles[5] = 3.;
            out.push_str(&format!("sri {}\n", imod_set_ref_image(&mut m2, &h)));
            let r = m2.ref_image.unwrap();
            out.push_str(&format!(
                "ref {} {} {} | {} {} {} | {} {} {} | {} {} {} flags={}\n",
                g9(r.ctrans.x as f64),
                g9(r.ctrans.y as f64),
                g9(r.ctrans.z as f64),
                g9(r.crot.x as f64),
                g9(r.crot.y as f64),
                g9(r.crot.z as f64),
                g9(r.cscale.x as f64),
                g9(r.cscale.y as f64),
                g9(r.cscale.z as f64),
                g9(r.otrans.x as f64),
                g9(r.otrans.y as f64),
                g9(r.otrans.z as f64),
                m2.flags
            ));
            {
                let mut li: LoadInfo = LoadInfo::default();
                li.xmin = 2;
                li.ymin = 4;
                li.zmin = 1;
                h.xorg = 9.;
                h.yorg = 1.;
                h.zorg = 5.;
                imod_trans_for_subset_load(&mut m2, &h, Some(&li));
                dumpmodel(&mut out, "sl", &m2);
                imod_trans_for_subset_load(&mut m2, &h, None);
                dumpmodel(&mut out, "sl2", &m2);
            }
            h.tiltangles[3] = 4.;
            h.tiltangles[4] = 5.;
            h.tiltangles[5] = 6.;
            h.xlen = 400.;
            out.push_str(&format!(
                "ttmi {}\n",
                imod_trans_to_match_image(&mut m2, &h, 0)
            ));
            dumpmodel(&mut out, "tm", &m2);
            out.push_str(&format!(
                "ttmi {}\n",
                imod_trans_to_match_image(&mut m2, &h, 1)
            ));
            dumpmodel(&mut out, "tm2", &m2);
        }

        out.push_str("--- transmodel3d ---\n");
        {
            let mut m3 = imod_new().unwrap();
            imod_new_object(&mut m3);
            addcont(&mut m3, 0, &sq, 4, 3.);
            m3.xmax = 100;
            m3.ymax = 80;
            m3.zmax = 20;
            let mut cl = m3.obj[0].clips.clone();
            setclips(&mut cl, 1);
            m3.obj[0].clips = cl;
            let mut tm = imod_mat_new(3).unwrap();
            imod_mat_rot(&mut tm, 15., B3D_Z);
            let cen = Ipoint {
                x: 50.,
                y: 40.,
                z: 10.,
            };
            imod_trans_model3d(&mut m3, &mut tm, None, cen, 1.5, 0);
            dumpmodel(&mut out, "t3", &m3);
            imod_mat_delete(&mut tm);
            let mut tm = imod_mat_new(3).unwrap();
            imod_mat_rot(&mut tm, 15., B3D_Y);
            imod_trans_model3d(&mut m3, &mut tm, None, cen, 1., 1);
            dumpmodel(&mut out, "t3f", &m3);
        }

        let want = super::EXPECTED_MODEL_DRIVER_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from imodel.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
    }
}

#[cfg(test)]
/// Verbatim stdout of the driver linked against `IMOD/libimod/imodel.c`.
const EXPECTED_MODEL_DRIVER_OUTPUT: &str = r#"--- default ---
name IMOD-NewModel
flags 3072 draw 1 mouse 2 bl 0 wl 255
off 0 0 0 scale 1 1 1
cindex -1 -1 -1 res 3 thresh 128 pix 1 units 0 csum 0
max 1 1 1 abg 0 0 0 views 1 cview 0 cms -1
--- units ---
u 0 pixels
u 3 km
u 1 m
u -2 cm
u -3 mm
u -6 um
u -9 nm
u 42 unknown units
u -10 A
u -12 pm
--- build ---
b objs=3 views=2 cindex 2 -1 -1
b max 100 80 20 flags=3072
b ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
b ob 0 clips 2 3 7 1
b ob 0 cl 0 0.25 -0.5 1.5 -2 3 -4
b ob 0 cl 1 0.5 -1.5 1 -3 5 -3
b ob 0 co 0 n=4 flags=0 surf=0 time=0
b ob 0 co 0 pt 0 0 0 3
b ob 0 co 0 pt 1 10 0 3
b ob 0 co 0 pt 2 10 10 3
b ob 0 co 0 pt 3 0 10 3
b ob 0 co 1 n=3 flags=0 surf=0 time=0
b ob 0 co 1 pt 0 2 3 5
b ob 0 co 1 pt 1 20 5 5
b ob 0 co 1 pt 2 11 25 5
b ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
b ob 1 clips 0 0 0 0
b ob 1 co 0 n=4 flags=0 surf=0 time=0
b ob 1 co 0 pt 0 0 0 7
b ob 1 co 0 pt 1 10 0 7
b ob 1 co 0 pt 2 10 10 7
b ob 1 co 0 pt 3 0 10 7
b ob 1 me 0 v=6 l=9
b ob 1 me 0 v 0 0 10 2
b ob 1 me 0 v 1 1.5 9 2.5
b ob 1 me 0 v 2 3 8 3
b ob 1 me 0 v 3 4.5 7 3.5
b ob 1 me 0 v 4 6 6 4
b ob 1 me 0 v 5 7.5 5 4.5
b ob 1 me 0 l 0 -25
b ob 1 me 0 l 1 0
b ob 1 me 0 l 2 1
b ob 1 me 0 l 3 2
b ob 1 me 0 l 4 3
b ob 1 me 0 l 5 4
b ob 1 me 0 l 6 5
b ob 1 me 0 l 7 -22
b ob 1 me 0 l 8 -1
b ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
b ob 2 clips 0 0 0 0
b ob 2 co 0 n=0 flags=0 surf=0 time=0
b vw 0 objv=0 clips 1 3 7 1
b vw 0 cl 0 0.25 -0.5 1.5 -2 3 -4
b vw 1 objv=3 clips 0 0 0 0
b vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
b vw 1 ov 0 cl 0 0.25 -0.5 1.5 -2 3 -4
b vw 1 ov 0 cl 1 0.5 -1.5 1 -3 5 -3
b vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
b vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
--- checksum ---
csum 925560250
maxtime 0 maxobj 3 zscale 1.25 pixsize 1.75 flipped 0
--- getters ---
first 0
next 1
next 2
next -99
pf 0 2 3 5
pn 1 20 5 5
pn 2 11 25 5
cg 3
pg 1
--- transform ---
tr 0
tr objs=3 views=2 cindex 0 0 0
tr max 100 80 20 flags=3072
tr ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
tr ob 0 clips 2 3 7 1
tr ob 0 cl 0 0.25 -0.5 1.5 -2 3 -4
tr ob 0 cl 1 0.5 -1.5 1 -3 5 -3
tr ob 0 co 0 n=4 flags=0 surf=0 time=0
tr ob 0 co 0 pt 0 3 -2 4
tr ob 0 co 0 pt 1 12.3969259 1.42020154 4
tr ob 0 co 0 pt 2 8.97672462 10.8171272 4
tr ob 0 co 0 pt 3 -0.42020154 7.39692593 4
tr ob 0 co 1 n=3 flags=0 surf=0 time=0
tr ob 0 co 1 pt 0 3.85332489 1.50311828 6
tr ob 0 co 1 pt 1 20.0837517 9.53886604 6
tr ob 0 co 1 pt 2 4.78611469 25.2545376 6
tr ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
tr ob 1 clips 0 0 0 0
tr ob 1 co 0 n=4 flags=0 surf=0 time=0
tr ob 1 co 0 pt 0 3 -2 8
tr ob 1 co 0 pt 1 12.3969259 1.42020154 8
tr ob 1 co 0 pt 2 8.97672462 10.8171272 8
tr ob 1 co 0 pt 3 -0.42020154 7.39692593 8
tr ob 1 me 0 v=6 l=9
tr ob 1 me 0 v 0 0 10 2
tr ob 1 me 0 v 1 1.5 9 2.5
tr ob 1 me 0 v 2 3 8 3
tr ob 1 me 0 v 3 4.5 7 3.5
tr ob 1 me 0 v 4 6 6 4
tr ob 1 me 0 v 5 7.5 5 4.5
tr ob 1 me 0 l 0 -25
tr ob 1 me 0 l 1 0
tr ob 1 me 0 l 2 1
tr ob 1 me 0 l 3 2
tr ob 1 me 0 l 4 3
tr ob 1 me 0 l 5 4
tr ob 1 me 0 l 6 5
tr ob 1 me 0 l 7 -22
tr ob 1 me 0 l 8 -1
tr ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
tr ob 2 clips 0 0 0 0
tr ob 2 co 0 n=0 flags=0 surf=0 time=0
tr vw 0 objv=0 clips 1 3 7 1
tr vw 0 cl 0 0.25 -0.5 1.5 -2 3 -4
tr vw 1 objv=3 clips 0 0 0 0
tr vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
tr vw 1 ov 0 cl 0 0.25 -0.5 1.5 -2 3 -4
tr vw 1 ov 0 cl 1 0.5 -1.5 1 -3 5 -3
tr vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
tr vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
ts 0
ts objs=3 views=2 cindex 0 0 0
ts max 100 80 20 flags=3072
ts ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
ts ob 0 clips 2 3 7 1
ts ob 0 cl 0 0.25 -0.5 1.5 -2 3 -4
ts ob 0 cl 1 0.5 -1.5 1 -3 5 -3
ts ob 0 co 0 n=4 flags=0 surf=0 time=0
ts ob 0 co 0 pt 0 3 -2 4
ts ob 0 co 0 pt 1 12.3969259 1.42020154 4
ts ob 0 co 0 pt 2 8.97672462 10.8171272 4
ts ob 0 co 0 pt 3 -0.42020154 7.39692593 4
ts ob 0 co 1 n=3 flags=0 surf=0 time=0
ts ob 0 co 1 pt 0 3.85332489 1.50311828 6
ts ob 0 co 1 pt 1 20.0837517 9.53886604 6
ts ob 0 co 1 pt 2 4.78611469 25.2545376 6
ts ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
ts ob 1 clips 0 0 0 0
ts ob 1 co 0 n=4 flags=0 surf=0 time=0
ts ob 1 co 0 pt 0 3 -2 8
ts ob 1 co 0 pt 1 12.3969259 1.42020154 8
ts ob 1 co 0 pt 2 8.97672462 10.8171272 8
ts ob 1 co 0 pt 3 -0.42020154 7.39692593 8
ts ob 1 me 0 v=6 l=9
ts ob 1 me 0 v 0 0 10 2
ts ob 1 me 0 v 1 1.5 9 2.5
ts ob 1 me 0 v 2 3 8 3
ts ob 1 me 0 v 3 4.5 7 3.5
ts ob 1 me 0 v 4 6 6 4
ts ob 1 me 0 v 5 7.5 5 4.5
ts ob 1 me 0 l 0 -25
ts ob 1 me 0 l 1 0
ts ob 1 me 0 l 2 1
ts ob 1 me 0 l 3 2
ts ob 1 me 0 l 4 3
ts ob 1 me 0 l 5 4
ts ob 1 me 0 l 6 5
ts ob 1 me 0 l 7 -22
ts ob 1 me 0 l 8 -1
ts ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
ts ob 2 clips 0 0 0 0
ts ob 2 co 0 n=0 flags=0 surf=0 time=0
ts vw 0 objv=0 clips 1 3 7 1
ts vw 0 cl 0 0.25 -0.5 1.5 -2 3 -4
ts vw 1 objv=3 clips 0 0 0 0
ts vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
ts vw 1 ov 0 cl 0 0.25 -0.5 1.5 -2 3 -4
ts vw 1 ov 0 cl 1 0.5 -1.5 1 -3 5 -3
ts vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
ts vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
--- rot90x/invertz ---
r0 objs=3 views=2 cindex 0 0 0
r0 max 100 20 80 flags=3072
r0 ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
r0 ob 0 clips 2 3 7 1
r0 ob 0 cl 0 0.25 1.5 0.5 -2 -4.5 -82.5
r0 ob 0 cl 1 0.5 1 1.5 -3 -3.5 -84.5
r0 ob 0 co 0 n=4 flags=0 surf=0 time=0
r0 ob 0 co 0 pt 0 3 4.5 81.5
r0 ob 0 co 0 pt 1 12.3969259 4.5 78.0797958
r0 ob 0 co 0 pt 2 8.97672462 4.5 68.6828766
r0 ob 0 co 0 pt 3 -0.42020154 4.5 72.1030731
r0 ob 0 co 1 n=3 flags=0 surf=0 time=0
r0 ob 0 co 1 pt 0 3.85332489 6.5 77.9968796
r0 ob 0 co 1 pt 1 20.0837517 6.5 69.9611359
r0 ob 0 co 1 pt 2 4.78611469 6.5 54.2454605
r0 ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
r0 ob 1 clips 0 0 0 0
r0 ob 1 co 0 n=4 flags=0 surf=0 time=0
r0 ob 1 co 0 pt 0 3 8.5 81.5
r0 ob 1 co 0 pt 1 12.3969259 8.5 78.0797958
r0 ob 1 co 0 pt 2 8.97672462 8.5 68.6828766
r0 ob 1 co 0 pt 3 -0.42020154 8.5 72.1030731
r0 ob 1 me 0 v=6 l=9
r0 ob 1 me 0 v 0 0 2.5 69.5
r0 ob 1 me 0 v 1 1.5 2.5 -9
r0 ob 1 me 0 v 2 3 3.5 71.5
r0 ob 1 me 0 v 3 4.5 3.5 -7
r0 ob 1 me 0 v 4 6 4.5 73.5
r0 ob 1 me 0 v 5 7.5 4.5 -5
r0 ob 1 me 0 l 0 -25
r0 ob 1 me 0 l 1 0
r0 ob 1 me 0 l 2 1
r0 ob 1 me 0 l 3 2
r0 ob 1 me 0 l 4 3
r0 ob 1 me 0 l 5 4
r0 ob 1 me 0 l 6 5
r0 ob 1 me 0 l 7 -22
r0 ob 1 me 0 l 8 -1
r0 ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
r0 ob 2 clips 0 0 0 0
r0 ob 2 co 0 n=0 flags=0 surf=0 time=0
r0 vw 0 objv=0 clips 1 3 7 1
r0 vw 0 cl 0 0.25 1.5 0.5 -2 -4.5 -82.5
r0 vw 1 objv=3 clips 0 0 0 0
r0 vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
r0 vw 1 ov 0 cl 0 0.25 1.5 0.5 -2 -4.5 -82.5
r0 vw 1 ov 0 cl 1 0.5 1 1.5 -3 -3.5 -84.5
r0 vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
r0 vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
r1 objs=3 views=2 cindex 0 0 0
r1 max 100 80 20 flags=3072
r1 ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
r1 ob 0 clips 2 3 7 1
r1 ob 0 cl 0 0.25 -0.5 1.5 -2 3 -4
r1 ob 0 cl 1 0.5 -1.5 1 -3 5 -3
r1 ob 0 co 0 n=4 flags=0 surf=0 time=0
r1 ob 0 co 0 pt 0 3 -2 4
r1 ob 0 co 0 pt 1 12.3969259 1.42020416 4
r1 ob 0 co 0 pt 2 8.97672462 10.8171234 4
r1 ob 0 co 0 pt 3 -0.42020154 7.39692688 4
r1 ob 0 co 1 n=3 flags=0 surf=0 time=0
r1 ob 0 co 1 pt 0 3.85332489 1.50312042 6
r1 ob 0 co 1 pt 1 20.0837517 9.53886414 6
r1 ob 0 co 1 pt 2 4.78611469 25.2545395 6
r1 ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
r1 ob 1 clips 0 0 0 0
r1 ob 1 co 0 n=4 flags=0 surf=0 time=0
r1 ob 1 co 0 pt 0 3 -2 8
r1 ob 1 co 0 pt 1 12.3969259 1.42020416 8
r1 ob 1 co 0 pt 2 8.97672462 10.8171234 8
r1 ob 1 co 0 pt 3 -0.42020154 7.39692688 8
r1 ob 1 me 0 v=6 l=9
r1 ob 1 me 0 v 0 0 10 2
r1 ob 1 me 0 v 1 1.5 9 2.5
r1 ob 1 me 0 v 2 3 8 3
r1 ob 1 me 0 v 3 4.5 7 3.5
r1 ob 1 me 0 v 4 6 6 4
r1 ob 1 me 0 v 5 7.5 5 4.5
r1 ob 1 me 0 l 0 -25
r1 ob 1 me 0 l 1 0
r1 ob 1 me 0 l 2 1
r1 ob 1 me 0 l 3 2
r1 ob 1 me 0 l 4 3
r1 ob 1 me 0 l 5 4
r1 ob 1 me 0 l 6 5
r1 ob 1 me 0 l 7 -22
r1 ob 1 me 0 l 8 -1
r1 ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
r1 ob 2 clips 0 0 0 0
r1 ob 2 co 0 n=0 flags=0 surf=0 time=0
r1 vw 0 objv=0 clips 1 3 7 1
r1 vw 0 cl 0 0.25 -0.5 1.5 -2 3 -4
r1 vw 1 objv=3 clips 0 0 0 0
r1 vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
r1 vw 1 ov 0 cl 0 0.25 -0.5 1.5 -2 3 -4
r1 vw 1 ov 0 cl 1 0.5 -1.5 1 -3 5 -3
r1 vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
r1 vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
iz objs=3 views=2 cindex 0 0 0
iz max 100 80 20 flags=3072
iz ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
iz ob 0 clips 2 3 7 1
iz ob 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
iz ob 0 cl 1 0.5 -1.5 -1 -3 5 -16
iz ob 0 co 0 n=4 flags=0 surf=0 time=0
iz ob 0 co 0 pt 0 3 -2 15
iz ob 0 co 0 pt 1 12.3969259 1.42020416 15
iz ob 0 co 0 pt 2 8.97672462 10.8171234 15
iz ob 0 co 0 pt 3 -0.42020154 7.39692688 15
iz ob 0 co 1 n=3 flags=0 surf=0 time=0
iz ob 0 co 1 pt 0 3.85332489 1.50312042 13
iz ob 0 co 1 pt 1 20.0837517 9.53886414 13
iz ob 0 co 1 pt 2 4.78611469 25.2545395 13
iz ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
iz ob 1 clips 0 0 0 0
iz ob 1 co 0 n=4 flags=0 surf=0 time=0
iz ob 1 co 0 pt 0 3 -2 11
iz ob 1 co 0 pt 1 12.3969259 1.42020416 11
iz ob 1 co 0 pt 2 8.97672462 10.8171234 11
iz ob 1 co 0 pt 3 -0.42020154 7.39692688 11
iz ob 1 me 0 v=6 l=9
iz ob 1 me 0 v 0 0 10 17
iz ob 1 me 0 v 1 1.5 9 -2.5
iz ob 1 me 0 v 2 3 8 16
iz ob 1 me 0 v 3 4.5 7 -3.5
iz ob 1 me 0 v 4 6 6 15
iz ob 1 me 0 v 5 7.5 5 -4.5
iz ob 1 me 0 l 0 -25
iz ob 1 me 0 l 1 1
iz ob 1 me 0 l 2 0
iz ob 1 me 0 l 3 2
iz ob 1 me 0 l 4 4
iz ob 1 me 0 l 5 3
iz ob 1 me 0 l 6 5
iz ob 1 me 0 l 7 -22
iz ob 1 me 0 l 8 -1
iz ob 2 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
iz ob 2 clips 0 0 0 0
iz ob 2 co 0 n=0 flags=0 surf=0 time=0
iz vw 0 objv=0 clips 1 3 7 1
iz vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
iz vw 1 objv=3 clips 0 0 0 0
iz vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
iz vw 1 ov 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
iz vw 1 ov 0 cl 1 0.5 -1.5 -1 -3 5 -16
iz vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
iz vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
--- clean/delete ---
clean 0
cl objs=3 views=2 cindex 2 -1 -1
cl max 100 80 20 flags=3072
cl ob 0 conts=2 meshes=0 surfsize=2 flags=402653184 pdraw=0 rgb 0 1 0
cl ob 0 clips 2 3 7 1
cl ob 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
cl ob 0 cl 1 0.5 -1.5 -1 -3 5 -16
cl ob 0 co 0 n=4 flags=0 surf=0 time=0
cl ob 0 co 0 pt 0 3 -2 15
cl ob 0 co 0 pt 1 12.3969259 1.42020416 15
cl ob 0 co 0 pt 2 8.97672462 10.8171234 15
cl ob 0 co 0 pt 3 -0.42020154 7.39692688 15
cl ob 0 co 1 n=3 flags=0 surf=0 time=0
cl ob 0 co 1 pt 0 3.85332489 1.50312042 13
cl ob 0 co 1 pt 1 20.0837517 9.53886414 13
cl ob 0 co 1 pt 2 4.78611469 25.2545395 13
cl ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
cl ob 1 clips 0 0 0 0
cl ob 1 co 0 n=4 flags=0 surf=0 time=0
cl ob 1 co 0 pt 0 3 -2 11
cl ob 1 co 0 pt 1 12.3969259 1.42020416 11
cl ob 1 co 0 pt 2 8.97672462 10.8171234 11
cl ob 1 co 0 pt 3 -0.42020154 7.39692688 11
cl ob 1 me 0 v=6 l=9
cl ob 1 me 0 v 0 0 10 17
cl ob 1 me 0 v 1 1.5 9 -2.5
cl ob 1 me 0 v 2 3 8 16
cl ob 1 me 0 v 3 4.5 7 -3.5
cl ob 1 me 0 v 4 6 6 15
cl ob 1 me 0 v 5 7.5 5 -4.5
cl ob 1 me 0 l 0 -25
cl ob 1 me 0 l 1 1
cl ob 1 me 0 l 2 0
cl ob 1 me 0 l 3 2
cl ob 1 me 0 l 4 4
cl ob 1 me 0 l 5 3
cl ob 1 me 0 l 6 5
cl ob 1 me 0 l 7 -22
cl ob 1 me 0 l 8 -1
cl ob 2 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
cl ob 2 clips 0 0 0 0
cl vw 0 objv=0 clips 1 3 7 1
cl vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
cl vw 1 objv=3 clips 0 0 0 0
cl vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
cl vw 1 ov 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
cl vw 1 ov 0 cl 1 0.5 -1.5 -1 -3 5 -16
cl vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
cl vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
dc 1
dc objs=3 views=2 cindex 0 -1 -1
dc max 100 80 20 flags=3072
dc ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
dc ob 0 clips 2 3 7 1
dc ob 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dc ob 0 cl 1 0.5 -1.5 -1 -3 5 -16
dc ob 0 co 0 n=3 flags=0 surf=0 time=0
dc ob 0 co 0 pt 0 3.85332489 1.50312042 13
dc ob 0 co 0 pt 1 20.0837517 9.53886414 13
dc ob 0 co 0 pt 2 4.78611469 25.2545395 13
dc ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
dc ob 1 clips 0 0 0 0
dc ob 1 co 0 n=4 flags=0 surf=0 time=0
dc ob 1 co 0 pt 0 3 -2 11
dc ob 1 co 0 pt 1 12.3969259 1.42020416 11
dc ob 1 co 0 pt 2 8.97672462 10.8171234 11
dc ob 1 co 0 pt 3 -0.42020154 7.39692688 11
dc ob 1 me 0 v=6 l=9
dc ob 1 me 0 v 0 0 10 17
dc ob 1 me 0 v 1 1.5 9 -2.5
dc ob 1 me 0 v 2 3 8 16
dc ob 1 me 0 v 3 4.5 7 -3.5
dc ob 1 me 0 v 4 6 6 15
dc ob 1 me 0 v 5 7.5 5 -4.5
dc ob 1 me 0 l 0 -25
dc ob 1 me 0 l 1 1
dc ob 1 me 0 l 2 0
dc ob 1 me 0 l 3 2
dc ob 1 me 0 l 4 4
dc ob 1 me 0 l 5 3
dc ob 1 me 0 l 6 5
dc ob 1 me 0 l 7 -22
dc ob 1 me 0 l 8 -1
dc ob 2 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
dc ob 2 clips 0 0 0 0
dc vw 0 objv=0 clips 1 3 7 1
dc vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dc vw 1 objv=3 clips 0 0 0 0
dc vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
dc vw 1 ov 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dc vw 1 ov 0 cl 1 0.5 -1.5 -1 -3 5 -16
dc vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
dc vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
dlc 0
dlc objs=3 views=2 cindex 0 -1 -1
dlc max 100 80 20 flags=3072
dlc ob 0 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
dlc ob 0 clips 2 3 7 1
dlc ob 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dlc ob 0 cl 1 0.5 -1.5 -1 -3 5 -16
dlc ob 1 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
dlc ob 1 clips 0 0 0 0
dlc ob 1 co 0 n=4 flags=0 surf=0 time=0
dlc ob 1 co 0 pt 0 3 -2 11
dlc ob 1 co 0 pt 1 12.3969259 1.42020416 11
dlc ob 1 co 0 pt 2 8.97672462 10.8171234 11
dlc ob 1 co 0 pt 3 -0.42020154 7.39692688 11
dlc ob 1 me 0 v=6 l=9
dlc ob 1 me 0 v 0 0 10 17
dlc ob 1 me 0 v 1 1.5 9 -2.5
dlc ob 1 me 0 v 2 3 8 16
dlc ob 1 me 0 v 3 4.5 7 -3.5
dlc ob 1 me 0 v 4 6 6 15
dlc ob 1 me 0 v 5 7.5 5 -4.5
dlc ob 1 me 0 l 0 -25
dlc ob 1 me 0 l 1 1
dlc ob 1 me 0 l 2 0
dlc ob 1 me 0 l 3 2
dlc ob 1 me 0 l 4 4
dlc ob 1 me 0 l 5 3
dlc ob 1 me 0 l 6 5
dlc ob 1 me 0 l 7 -22
dlc ob 1 me 0 l 8 -1
dlc ob 2 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
dlc ob 2 clips 0 0 0 0
dlc vw 0 objv=0 clips 1 3 7 1
dlc vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dlc vw 1 objv=3 clips 0 0 0 0
dlc vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
dlc vw 1 ov 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
dlc vw 1 ov 0 cl 1 0.5 -1.5 -1 -3 5 -16
dlc vw 1 ov 1 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
dlc vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
mv 0
mv objs=3 views=2 cindex 0 -1 -1
mv max 100 80 20 flags=3072
mv ob 0 conts=1 meshes=1 surfsize=0 flags=402653192 pdraw=0 rgb 0 1 1
mv ob 0 clips 0 0 0 0
mv ob 0 co 0 n=4 flags=0 surf=0 time=0
mv ob 0 co 0 pt 0 3 -2 11
mv ob 0 co 0 pt 1 12.3969259 1.42020416 11
mv ob 0 co 0 pt 2 8.97672462 10.8171234 11
mv ob 0 co 0 pt 3 -0.42020154 7.39692688 11
mv ob 0 me 0 v=6 l=9
mv ob 0 me 0 v 0 0 10 17
mv ob 0 me 0 v 1 1.5 9 -2.5
mv ob 0 me 0 v 2 3 8 16
mv ob 0 me 0 v 3 4.5 7 -3.5
mv ob 0 me 0 v 4 6 6 15
mv ob 0 me 0 v 5 7.5 5 -4.5
mv ob 0 me 0 l 0 -25
mv ob 0 me 0 l 1 1
mv ob 0 me 0 l 2 0
mv ob 0 me 0 l 3 2
mv ob 0 me 0 l 4 4
mv ob 0 me 0 l 5 3
mv ob 0 me 0 l 6 5
mv ob 0 me 0 l 7 -22
mv ob 0 me 0 l 8 -1
mv ob 1 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
mv ob 1 clips 2 3 7 1
mv ob 1 cl 0 0.25 -0.5 -1.5 -2 3 -15
mv ob 1 cl 1 0.5 -1.5 -1 -3 5 -16
mv ob 2 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
mv ob 2 clips 0 0 0 0
mv vw 0 objv=0 clips 1 3 7 1
mv vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
mv vw 1 objv=3 clips 0 0 0 0
mv vw 1 ov 0 flags=402653192 rgb 0 1 1 pdraw=9 clips 0 0 0 0
mv vw 1 ov 1 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
mv vw 1 ov 1 cl 0 0.25 -0.5 -1.5 -2 3 -15
mv vw 1 ov 1 cl 1 0.5 -1.5 -1 -3 5 -16
mv vw 1 ov 2 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
do 0
do objs=2 views=2 cindex 0 -1 -1
do max 100 80 20 flags=3072
do ob 0 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
do ob 0 clips 2 3 7 1
do ob 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
do ob 0 cl 1 0.5 -1.5 -1 -3 5 -16
do ob 1 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
do ob 1 clips 0 0 0 0
do vw 0 objv=0 clips 1 3 7 1
do vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
do vw 1 objv=2 clips 0 0 0 0
do vw 1 ov 0 flags=402653184 rgb 0 1 0 pdraw=0 clips 2 3 7 1
do vw 1 ov 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
do vw 1 ov 0 cl 1 0.5 -1.5 -1 -3 5 -16
do vw 1 ov 1 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
do 0
do2 objs=1 views=2 cindex 0 -1 -1
do2 max 100 80 20 flags=3072
do2 ob 0 conts=0 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 1 0 1
do2 ob 0 clips 0 0 0 0
do2 vw 0 objv=0 clips 1 3 7 1
do2 vw 0 cl 0 0.25 -0.5 -1.5 -2 3 -15
do2 vw 1 objv=1 clips 0 0 0 0
do2 vw 1 ov 0 flags=402653184 rgb 1 0 1 pdraw=0 clips 0 0 0 0
--- refimage ---
sri 0
ref 5 -3 2 | 1 2 3 | 2 2 2 | 5 -3 2 flags=35840
sl objs=1 views=1 cindex 0 -1 -1
sl max 100 80 20 flags=35840
sl ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
sl ob 0 clips 0 0 0 0
sl ob 0 co 0 n=4 flags=0 surf=0 time=0
sl ob 0 co 0 pt 0 0 -2 3.5
sl ob 0 co 0 pt 1 10 -2 3.5
sl ob 0 co 0 pt 2 10 8 3.5
sl ob 0 co 0 pt 3 0 8 3.5
sl vw 0 objv=0 clips 0 0 0 0
sl2 objs=1 views=1 cindex 0 -1 -1
sl2 max 100 80 20 flags=35840
sl2 ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
sl2 ob 0 clips 0 0 0 0
sl2 ob 0 co 0 n=4 flags=0 surf=0 time=0
sl2 ob 0 co 0 pt 0 2 0 5
sl2 ob 0 co 0 pt 1 12 0 5
sl2 ob 0 co 0 pt 2 12 10 5
sl2 ob 0 co 0 pt 3 2 10 5
sl2 vw 0 objv=0 clips 0 0 0 0
ttmi 0
tm objs=1 views=1 cindex 0 -1 -1
tm max 100 80 20 flags=35840
tm ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
tm ob 0 clips 0 0 0 0
tm ob 0 co 0 n=4 flags=0 surf=0 time=0
tm ob 0 co 0 pt 0 2.07669926 1.7241416 6.59688473
tm ob 0 co 0 pt 1 7.063025 2.28233266 6.11247635
tm ob 0 co 0 pt 2 6.7978096 12.252079 6.68063784
tm ob 0 co 0 pt 3 1.81148374 11.6938877 7.16504574
tm vw 0 objv=0 clips 0 0 0 0
ttmi 0
tm2 objs=1 views=1 cindex 0 -1 -1
tm2 max 100 80 20 flags=35840
tm2 ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
tm2 ob 0 clips 0 0 0 0
tm2 ob 0 co 0 n=4 flags=0 surf=0 time=0
tm2 ob 0 co 0 pt 0 2.03834963 3.7241416 8.09688473
tm2 ob 0 co 0 pt 1 4.53151226 4.28233242 7.61247635
tm2 ob 0 co 0 pt 2 4.3989048 14.252079 8.18063736
tm2 ob 0 co 0 pt 3 1.90574193 13.6938877 8.66504574
tm2 vw 0 objv=0 clips 0 0 0 0
--- transmodel3d ---
t3 objs=1 views=1 cindex 0 -1 -1
t3 max 100 80 20 flags=3072
t3 ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
t3 ob 0 clips 1 3 7 1
t3 ob 0 cl 0 0.231693774 -0.261283815 0.937042594 -14.764782 13.9581232 -4
t3 ob 0 co 0 n=4 flags=0 surf=0 time=0
t3 ob 0 co 0 pt 0 12.0564728 -11.5779839 3
t3 ob 0 co 0 pt 1 21.7157307 -8.98979378 3
t3 ob 0 co 0 pt 2 19.1275406 0.669464111 3
t3 ob 0 co 0 pt 3 9.4682827 -1.91872597 3
t3 vw 0 objv=0 clips 0 0 0 0
t3f objs=1 views=1 cindex 0 -1 -1
t3f max 100 80 20 flags=3072
t3f ob 0 conts=1 meshes=0 surfsize=0 flags=402653184 pdraw=0 rgb 0 1 0
t3f ob 0 clips 1 3 7 1
t3f ob 0 cl 0 0.156173766 -0.312347561 0.937042594 -2.00000095 2.99999809 -4
t3f ob 0 co 0 n=4 flags=0 surf=0 time=0
t3f ob 0 co 0 pt 0 0 1.90734863e-06 3
t3f ob 0 co 0 pt 1 10 1.90734863e-06 3
t3f ob 0 co 0 pt 2 9.99999809 10.0000019 3
t3f ob 0 co 0 pt 3 0 10.0000019 3
t3f vw 0 objv=0 clips 0 0 0 0
"#;
