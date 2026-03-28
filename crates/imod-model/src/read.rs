use std::io::{self, Read, Seek};
use std::path::Path;

use imod_core::{ImodError, Point3f};

use crate::types::*;

/// Read an IMOD model file.
///
/// The IMOD model format is a chunk-based binary format. All multi-byte values
/// are big-endian. The file starts with "IMOD" + "V1.2", followed by the model
/// header, then OBJT/CONT/MESH chunks for each object, and ends with IEOF.
pub fn read_model(path: impl AsRef<Path>) -> Result<ImodModel, ImodError> {
    let data = std::fs::read(path.as_ref())?;
    let mut cursor = io::Cursor::new(&data);
    read_model_from(&mut cursor)
}

pub fn read_model_from<R: Read + Seek>(r: &mut R) -> Result<ImodModel, ImodError> {
    // Read file ID: "IMOD" + "V1.2"
    let file_id = read_u32(r)?;
    if file_id != chunk_id::IMOD {
        return Err(ImodError::InvalidHeader("not an IMOD file".into()));
    }
    let version = read_u32(r)?;
    if version != chunk_id::V1_2 {
        return Err(ImodError::InvalidHeader(format!(
            "unsupported IMOD version: 0x{version:08x}"
        )));
    }

    // Read model header
    let mut name_bytes = [0u8; 128];
    r.read_exact(&mut name_bytes)?;
    let name = String::from_utf8_lossy(&name_bytes)
        .trim_end_matches('\0')
        .to_string();

    let xmax = read_i32(r)?;
    let ymax = read_i32(r)?;
    let zmax = read_i32(r)?;
    let objsize = read_i32(r)?;
    let flags = read_u32(r)?;
    let drawmode = read_i32(r)?;
    let mousemode = read_i32(r)?;
    let black_level = read_i32(r)?;
    let white_level = read_i32(r)?;
    let xoffset = read_f32(r)?;
    let yoffset = read_f32(r)?;
    let zoffset = read_f32(r)?;
    let xscale = read_f32(r)?;
    let yscale = read_f32(r)?;
    let zscale = read_f32(r)?;
    let _cindex_obj = read_i32(r)?;
    let _cindex_cont = read_i32(r)?;
    let _cindex_pt = read_i32(r)?;
    let _res = read_i32(r)?;
    let _thresh = read_i32(r)?;
    let pixel_size = read_f32(r)?;
    let units = read_i32(r)?;
    let _csum = read_i32(r)?;
    let _alpha = read_f32(r)?;
    let _beta = read_f32(r)?;
    let _gamma = read_f32(r)?;

    let mut model = ImodModel {
        name,
        xmax,
        ymax,
        zmax,
        flags,
        drawmode,
        mousemode,
        black_level,
        white_level,
        offset: Point3f { x: xoffset, y: yoffset, z: zoffset },
        scale: Point3f { x: xscale, y: yscale, z: zscale },
        pixel_size,
        units,
        objects: Vec::with_capacity(objsize as usize),
    };

    // Read chunks until IEOF
    let mut current_obj: Option<ImodObject> = None;

    loop {
        let chunk = match read_u32(r) {
            Ok(c) => c,
            Err(_) => break, // EOF
        };

        match chunk {
            chunk_id::OBJT => {
                // Push previous object if any
                if let Some(obj) = current_obj.take() {
                    model.objects.push(obj);
                }
                current_obj = Some(read_object_header(r)?);
            }
            chunk_id::CONT => {
                let cont = read_contour(r)?;
                if let Some(ref mut obj) = current_obj {
                    obj.contours.push(cont);
                }
            }
            chunk_id::MESH => {
                let mesh = read_mesh(r)?;
                if let Some(ref mut obj) = current_obj {
                    obj.meshes.push(mesh);
                }
            }
            chunk_id::IEOF => {
                if let Some(obj) = current_obj.take() {
                    model.objects.push(obj);
                }
                break;
            }
            _ => {
                // Unknown chunk: read size and skip
                let size = read_i32(r)?;
                if size > 0 {
                    let mut skip = vec![0u8; size as usize];
                    r.read_exact(&mut skip)?;
                }
            }
        }
    }

    Ok(model)
}

fn read_object_header<R: Read>(r: &mut R) -> Result<ImodObject, ImodError> {
    let mut name_bytes = [0u8; 64];
    r.read_exact(&mut name_bytes)?;
    let name = String::from_utf8_lossy(&name_bytes)
        .trim_end_matches('\0')
        .to_string();

    let mut extra = [0u8; 64]; // IOBJ_EXSIZE = 16 ints = 64 bytes
    r.read_exact(&mut extra)?;

    let contsize = read_i32(r)?;
    let flags = read_u32(r)?;
    let axis = read_i32(r)?;
    let drawmode = read_i32(r)?;
    let red = read_f32(r)?;
    let green = read_f32(r)?;
    let blue = read_f32(r)?;
    let pdrawsize = read_i32(r)?;

    let symbol = read_u8(r)?;
    let symsize = read_u8(r)?;
    let linewidth2 = read_u8(r)?;
    let linewidth = read_u8(r)?;
    let linestyle = read_u8(r)?;
    let _symflags = read_u8(r)?;
    let _sympad = read_u8(r)?;
    let trans = read_u8(r)?;

    let _meshsize = read_i32(r)?;
    let _surfsize = read_i32(r)?;

    Ok(ImodObject {
        name,
        flags,
        axis,
        drawmode,
        red,
        green,
        blue,
        pdrawsize,
        symbol,
        symsize,
        linewidth2,
        linewidth,
        linestyle,
        trans,
        contours: Vec::with_capacity(contsize as usize),
        meshes: Vec::new(),
    })
}

fn read_contour<R: Read>(r: &mut R) -> Result<ImodContour, ImodError> {
    let psize = read_i32(r)?;
    let flags = read_u32(r)?;
    let time = read_i32(r)?;
    let surf = read_i32(r)?;

    let mut points = Vec::with_capacity(psize as usize);
    for _ in 0..psize {
        let x = read_f32(r)?;
        let y = read_f32(r)?;
        let z = read_f32(r)?;
        points.push(Point3f { x, y, z });
    }

    Ok(ImodContour {
        points,
        flags,
        time,
        surf,
        sizes: None,
    })
}

fn read_mesh<R: Read>(r: &mut R) -> Result<ImodMesh, ImodError> {
    let vsize = read_i32(r)?;
    let lsize = read_i32(r)?;
    let flag = read_u32(r)?;
    let time = read_i16(r)?;
    let surf = read_i16(r)?;

    let mut vertices = Vec::with_capacity(vsize as usize);
    for _ in 0..vsize {
        let x = read_f32(r)?;
        let y = read_f32(r)?;
        let z = read_f32(r)?;
        vertices.push(Point3f { x, y, z });
    }

    let mut indices = Vec::with_capacity(lsize as usize);
    for _ in 0..lsize {
        indices.push(read_i32(r)?);
    }

    Ok(ImodMesh {
        vertices,
        indices,
        flag,
        time,
        surf,
    })
}

// Big-endian reading helpers
fn read_u8<R: Read>(r: &mut R) -> Result<u8, ImodError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, ImodError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_be_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, ImodError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_be_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, ImodError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, ImodError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}
