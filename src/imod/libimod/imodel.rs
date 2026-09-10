//! Data declarations from `IMOD/include/imodel.h`, `iobj.h`, `icont.h`, and
//! `imesh.h` used by the translated command-line model utilities.
//!
//! This is deliberately a direct data representation: model-file decoding is
//! implemented by the source-mapped `imodel_files` unit, not by a substitute
//! serialization format.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Iplane {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

/// Original: `Iindex` (`include/imodel.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Iindex {
    pub object: i32,
    pub contour: i32,
    pub point: i32,
}

/// Original: `IobjGroup` (`include/objgroup.h`).
#[derive(Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Iobj_group {
    pub obj_list: Vec<i32>,
    pub name: [u8; 32],
}

/// Original: `IclipPlanes` (`include/imodel.h`).
#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Iclip_planes {
    pub count: u8,
    pub flags: u8,
    pub trans: u8,
    pub plane: u8,
    pub normal: [Ipoint; 7],
    pub point: [Ipoint; 7],
}

impl Default for Iclip_planes {
    fn default() -> Self {
        Self {
            count: 0,
            flags: 0,
            trans: 0,
            plane: 0,
            normal: [Ipoint::default(); 7],
            point: [Ipoint::default(); 7],
        }
    }
}

/// Original: `Iobjview` (`include/imodel.h`).
#[derive(Clone, Debug, Default, PartialEq)]
#[repr(C)]
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
#[repr(C)]
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
#[repr(C)]
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
#[repr(C)]
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
#[repr(C)]
pub struct Icont {
    pub temp_val: f64,
    pub pts: Vec<Ipoint>,
    pub sizes: Vec<f32>,
    pub flags: u32,
    pub time: i32,
    pub surf: i32,
    pub store: Vec<super::istore::Istore>,
}

/// Original: `Mod_Mesh` (`include/imodel.h`).
#[derive(Clone, Debug, Default)]
#[repr(C)]
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
#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct Iobj {
    pub cont: Vec<Icont>,
    pub mesh: Vec<Imesh>,
    pub store: Vec<super::istore::Istore>,
    pub name: String,
    pub extra: [u32; 16],
    pub flags: u32,
    pub axis: i32,
    pub drawmode: i32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
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

/// Original: `Mod_Model` (`include/imodel.h`), fields consumed by
/// `imodinfo.cpp`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Imod {
    pub obj: Vec<Iobj>,
    pub store: Vec<super::istore::Istore>,
    pub name: String,
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
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub cview: i32,
    pub view: Vec<Iview>,
    pub ref_image: Option<Iref_image>,
    pub slicer_ang: Vec<Slicer_angles>,
    pub group_list: Vec<Iobj_group>,
    pub cur_mesh_surf: i32,
}

impl Default for Imod {
    fn default() -> Self {
        Self {
            obj: Vec::new(),
            store: Vec::new(),
            name: String::new(),
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
            alpha: 0.,
            beta: 0.,
            gamma: 0.,
            cview: 0,
            view: vec![Iview::default()],
            ref_image: None,
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
pub fn imod_get_filename(imod: &Imod) -> &str {
    &imod.name
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
        store: Vec::new(),
        name: String::new(),
        flags: (1 << 27) | (1 << 28),
        axis: 0,
        drawmode: 1,
        red: colors[color_ind][0],
        green: colors[color_ind][1],
        blue: colors[color_ind][2],
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
            }; 7],
            point: [Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            }; 7],
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
        for plane in 0..object.clips.count.min(7) as usize {
            std::mem::swap(
                &mut object.clips.normal[plane].y,
                &mut object.clips.normal[plane].z,
            );
            std::mem::swap(
                &mut object.clips.point[plane].y,
                &mut object.clips.point[plane].z,
            );
        }
    }
    for view in &mut imod.view {
        for plane in 0..view.clips.count.min(7) as usize {
            std::mem::swap(
                &mut view.clips.normal[plane].y,
                &mut view.clips.normal[plane].z,
            );
            std::mem::swap(
                &mut view.clips.point[plane].y,
                &mut view.clips.point[plane].z,
            );
        }
        for object in &mut view.objview {
            for plane in 0..object.clips.count.min(7) as usize {
                std::mem::swap(
                    &mut object.clips.normal[plane].y,
                    &mut object.clips.normal[plane].z,
                );
                std::mem::swap(
                    &mut object.clips.point[plane].y,
                    &mut object.clips.point[plane].z,
                );
            }
        }
    }
    std::mem::swap(&mut imod.ymax, &mut imod.zmax);
}

/// Original: `imodTransFromRefImage` (`imodel.c:1850`).
///
/// This applies the old-reference-to-current-reference sequence directly to
/// model point data.  The source uses `Imat`; the arithmetic and rotation
/// order here are kept identical while retaining typed Rust model storage.
pub fn imod_trans_from_ref_image(
    imod: &mut Imod,
    reference: &Iref_image,
    bin_scale: Ipoint,
) -> i32 {
    if imod.flags & IMODF_FLIPYZ != 0 {
        imod_flip_yz(imod);
        imod.flags &= !IMODF_FLIPYZ;
    }
    if reference.oscale.x == 0.
        || reference.oscale.y == 0.
        || reference.oscale.z == 0.
        || reference.cscale.x == 0.
        || reference.cscale.y == 0.
        || reference.cscale.z == 0.
        || bin_scale.x == 0.
        || bin_scale.y == 0.
        || bin_scale.z == 0.
    {
        return 1;
    }
    let rotate = imod.flags & IMODF_TILTOK != 0 && (reference.crot != reference.orot);
    for object in &mut imod.obj {
        for contour in &mut object.cont {
            for point in &mut contour.pts {
                point.z += 0.5;
                point.x = point.x * reference.oscale.x - reference.otrans.x;
                point.y = point.y * reference.oscale.y - reference.otrans.y;
                point.z = point.z * reference.oscale.z - reference.otrans.z;
                if rotate {
                    let (s, c) = (-reference.orot.x).to_radians().sin_cos();
                    let y = point.y * c - point.z * s;
                    point.z = point.y * s + point.z * c;
                    point.y = y;
                    let (s, c) = (-reference.orot.y).to_radians().sin_cos();
                    let x = point.x * c + point.z * s;
                    point.z = -point.x * s + point.z * c;
                    point.x = x;
                    let (s, c) = (-reference.orot.z).to_radians().sin_cos();
                    let x = point.x * c - point.y * s;
                    point.y = point.x * s + point.y * c;
                    point.x = x;
                    let (s, c) = reference.crot.z.to_radians().sin_cos();
                    let x = point.x * c - point.y * s;
                    point.y = point.x * s + point.y * c;
                    point.x = x;
                    let (s, c) = reference.crot.y.to_radians().sin_cos();
                    let x = point.x * c + point.z * s;
                    point.z = -point.x * s + point.z * c;
                    point.x = x;
                    let (s, c) = reference.crot.x.to_radians().sin_cos();
                    let y = point.y * c - point.z * s;
                    point.z = point.y * s + point.z * c;
                    point.y = y;
                }
                point.x = (point.x + reference.ctrans.x) / (reference.cscale.x * bin_scale.x);
                point.y = (point.y + reference.ctrans.y) / (reference.cscale.y * bin_scale.y);
                point.z = (point.z + reference.ctrans.z) / (reference.cscale.z * bin_scale.z) - 0.5;
            }
        }
        for mesh in &mut object.mesh {
            for point in &mut mesh.vert {
                point.z += 0.5;
                point.x = point.x * reference.oscale.x - reference.otrans.x;
                point.y = point.y * reference.oscale.y - reference.otrans.y;
                point.z = point.z * reference.oscale.z - reference.otrans.z;
                if rotate {
                    let (s, c) = (-reference.orot.x).to_radians().sin_cos();
                    let y = point.y * c - point.z * s;
                    point.z = point.y * s + point.z * c;
                    point.y = y;
                    let (s, c) = (-reference.orot.y).to_radians().sin_cos();
                    let x = point.x * c + point.z * s;
                    point.z = -point.x * s + point.z * c;
                    point.x = x;
                    let (s, c) = (-reference.orot.z).to_radians().sin_cos();
                    let x = point.x * c - point.y * s;
                    point.y = point.x * s + point.y * c;
                    point.x = x;
                    let (s, c) = reference.crot.z.to_radians().sin_cos();
                    let x = point.x * c - point.y * s;
                    point.y = point.x * s + point.y * c;
                    point.x = x;
                    let (s, c) = reference.crot.y.to_radians().sin_cos();
                    let x = point.x * c + point.z * s;
                    point.z = -point.x * s + point.z * c;
                    point.x = x;
                    let (s, c) = reference.crot.x.to_radians().sin_cos();
                    let y = point.y * c - point.z * s;
                    point.z = point.y * s + point.z * c;
                    point.y = y;
                }
                point.x = (point.x + reference.ctrans.x) / (reference.cscale.x * bin_scale.x);
                point.y = (point.y + reference.ctrans.y) / (reference.cscale.y * bin_scale.y);
                point.z = (point.z + reference.ctrans.z) / (reference.cscale.z * bin_scale.z) - 0.5;
            }
        }
    }
    0
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
                    vert: vec![Ipoint {
                        x: 5.,
                        y: 6.,
                        z: 7.,
                    }],
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
