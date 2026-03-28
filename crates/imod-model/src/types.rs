use imod_core::Point3f;

/// Chunk IDs used in the IMOD binary model format.
/// Each is a 4-byte big-endian identifier.
pub mod chunk_id {
    pub const IMOD: u32 = u32::from_be_bytes(*b"IMOD");
    pub const V1_2: u32 = u32::from_be_bytes(*b"V1.2");
    pub const OBJT: u32 = u32::from_be_bytes(*b"OBJT");
    pub const CONT: u32 = u32::from_be_bytes(*b"CONT");
    pub const MESH: u32 = u32::from_be_bytes(*b"MESH");
    pub const VIEW: u32 = u32::from_be_bytes(*b"VIEW");
    pub const IEOF: u32 = u32::from_be_bytes(*b"IEOF");
    pub const IMAT: u32 = u32::from_be_bytes(*b"IMAT");
    pub const SIZE: u32 = u32::from_be_bytes(*b"SIZE");
    pub const LABL: u32 = u32::from_be_bytes(*b"LABL");
    pub const OLBL: u32 = u32::from_be_bytes(*b"OLBL");
    pub const CLIP: u32 = u32::from_be_bytes(*b"CLIP");
    pub const MCLP: u32 = u32::from_be_bytes(*b"MCLP");
    pub const IMNX: u32 = u32::from_be_bytes(*b"MINX");
    pub const MOST: u32 = u32::from_be_bytes(*b"MOST");
    pub const OBST: u32 = u32::from_be_bytes(*b"OBST");
    pub const COST: u32 = u32::from_be_bytes(*b"COST");
    pub const MEST: u32 = u32::from_be_bytes(*b"MEST");
    pub const MEPA: u32 = u32::from_be_bytes(*b"MEPA");
    pub const SLAN: u32 = u32::from_be_bytes(*b"SLAN");
    pub const OGRP: u32 = u32::from_be_bytes(*b"OGRP");
}

/// An IMOD model, consisting of objects which contain contours and meshes.
#[derive(Debug, Clone)]
pub struct ImodModel {
    pub name: String,
    pub xmax: i32,
    pub ymax: i32,
    pub zmax: i32,
    pub flags: u32,
    pub drawmode: i32,
    pub mousemode: i32,
    pub black_level: i32,
    pub white_level: i32,
    pub offset: Point3f,
    pub scale: Point3f,
    pub pixel_size: f32,
    pub units: i32,
    pub objects: Vec<ImodObject>,
    pub views: Vec<ImodView>,
    pub ref_image: Option<IrefImage>,
    pub slicer_angles: Vec<SlicerAngle>,
    pub store: Vec<u8>,
    /// Model-level clipping planes (MCLP chunk).
    pub clips: Option<IclipPlanes>,
}

impl Default for ImodModel {
    fn default() -> Self {
        Self {
            name: String::new(),
            xmax: 0,
            ymax: 0,
            zmax: 0,
            flags: 0,
            drawmode: 1,
            mousemode: 1,
            black_level: 0,
            white_level: 255,
            offset: Point3f::default(),
            scale: Point3f { x: 1.0, y: 1.0, z: 1.0 },
            pixel_size: 1.0,
            units: 0,
            objects: Vec::new(),
            views: Vec::new(),
            ref_image: None,
            slicer_angles: Vec::new(),
            store: Vec::new(),
            clips: None,
        }
    }
}

/// An object in the model, containing contours and optionally meshes.
#[derive(Debug, Clone)]
pub struct ImodObject {
    pub name: String,
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
    pub linestyle: u8,
    pub trans: u8,
    pub contours: Vec<ImodContour>,
    pub meshes: Vec<ImodMesh>,
    pub imat: Option<ImatData>,
    pub clips: Option<IclipPlanes>,
    pub store: Vec<u8>,
}

impl Default for ImodObject {
    fn default() -> Self {
        Self {
            name: String::new(),
            flags: 0,
            axis: 0,
            drawmode: 1,
            red: 0.0,
            green: 1.0,
            blue: 0.0,
            pdrawsize: 0,
            symbol: 0,
            symsize: 3,
            linewidth2: 1,
            linewidth: 1,
            linestyle: 0,
            trans: 0,
            contours: Vec::new(),
            meshes: Vec::new(),
            imat: None,
            clips: None,
            store: Vec::new(),
        }
    }
}

/// A contour: an ordered list of 3D points.
#[derive(Debug, Clone)]
pub struct ImodContour {
    pub points: Vec<Point3f>,
    pub flags: u32,
    pub time: i32,
    pub surf: i32,
    pub sizes: Option<Vec<f32>>,
}

impl Default for ImodContour {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            flags: 0,
            time: 0,
            surf: 0,
            sizes: None,
        }
    }
}

/// Clipping planes for an object.
#[derive(Debug, Clone)]
pub struct IclipPlanes {
    pub count: u8,
    pub flags: u8,
    pub trans: u8,
    pub plane: u8,
    pub normals: Vec<Point3f>,
    pub points: Vec<Point3f>,
}

impl Default for IclipPlanes {
    fn default() -> Self {
        Self {
            count: 0,
            flags: 0,
            trans: 0,
            plane: 0,
            normals: Vec::new(),
            points: Vec::new(),
        }
    }
}

/// Material data for an object (IMAT chunk, 16 bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct ImatData {
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

impl Default for ImatData {
    fn default() -> Self {
        Self {
            ambient: 102,
            diffuse: 255,
            specular: 127,
            shininess: 64,
            fillred: 0,
            fillgreen: 0,
            fillblue: 0,
            quality: 0,
            mat2: 0,
            valblack: 0,
            valwhite: 255,
            matflags2: 0,
            mesh_thickness: 0,
        }
    }
}

/// A single view (VIEW chunk).
#[derive(Debug, Clone)]
pub struct ImodView {
    pub fovy: f32,
    pub rad: f32,
    pub aspect: f32,
    pub cnear: f32,
    pub cfar: f32,
    pub rot: Point3f,
    pub trans: Point3f,
    pub scale: Point3f,
    pub world: u32,
    pub label: String,
}

impl Default for ImodView {
    fn default() -> Self {
        Self {
            fovy: 0.0,
            rad: 1.0,
            aspect: 1.0,
            cnear: 0.0,
            cfar: 1.0,
            rot: Point3f::default(),
            trans: Point3f::default(),
            scale: Point3f { x: 1.0, y: 1.0, z: 1.0 },
            world: 0,
            label: String::new(),
        }
    }
}

/// Slicer angle (SLAN chunk, 60 bytes).
#[derive(Debug, Clone)]
pub struct SlicerAngle {
    pub time: i32,
    pub angles: [f32; 3],
    pub center: Point3f,
    pub label: String,
}

impl Default for SlicerAngle {
    fn default() -> Self {
        Self {
            time: 0,
            angles: [0.0; 3],
            center: Point3f::default(),
            label: String::new(),
        }
    }
}

/// Reference image transform (MINX chunk, 72 bytes = 6 Point3f).
#[derive(Debug, Clone, PartialEq)]
pub struct IrefImage {
    pub oscale: Point3f,
    pub otrans: Point3f,
    pub orot: Point3f,
    pub cscale: Point3f,
    pub ctrans: Point3f,
    pub crot: Point3f,
}

impl Default for IrefImage {
    fn default() -> Self {
        Self {
            oscale: Point3f { x: 1.0, y: 1.0, z: 1.0 },
            otrans: Point3f::default(),
            orot: Point3f::default(),
            cscale: Point3f { x: 1.0, y: 1.0, z: 1.0 },
            ctrans: Point3f::default(),
            crot: Point3f::default(),
        }
    }
}

/// A mesh: vertices and an index list with drawing commands.
#[derive(Debug, Clone)]
pub struct ImodMesh {
    pub vertices: Vec<Point3f>,
    pub indices: Vec<i32>,
    pub flag: u32,
    pub time: i16,
    pub surf: i16,
}

impl Default for ImodMesh {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            flag: 0,
            time: 0,
            surf: 0,
        }
    }
}
