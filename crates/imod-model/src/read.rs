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

/// Read an IMOD model from any reader that implements `Read + Seek`.
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
        views: Vec::new(),
        ref_image: None,
        slicer_angles: Vec::new(),
        store: Vec::new(),
        clips: None,
        unknown_chunks: Vec::new(),
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
            chunk_id::IMAT => {
                let size = read_i32(r)?;
                if size >= 16 {
                    let imat = read_imat(r)?;
                    if let Some(ref mut obj) = current_obj {
                        obj.imat = Some(imat);
                    }
                    // Skip any extra bytes beyond 16
                    if size > 16 {
                        let mut skip = vec![0u8; (size - 16) as usize];
                        r.read_exact(&mut skip)?;
                    }
                }
            }
            chunk_id::CLIP => {
                let size = read_i32(r)?;
                let clip = read_clip(r, size)?;
                if let Some(ref mut obj) = current_obj {
                    obj.clips = Some(clip);
                }
            }
            chunk_id::MCLP => {
                let size = read_i32(r)?;
                let clip = read_clip(r, size)?;
                model.clips = Some(clip);
            }
            chunk_id::VIEW => {
                let size = read_i32(r)?;
                let view = read_view(r, size)?;
                model.views.push(view);
            }
            chunk_id::IMNX => {
                let size = read_i32(r)?;
                if size >= 72 {
                    let ref_image = read_iref_image(r)?;
                    model.ref_image = Some(ref_image);
                    if size > 72 {
                        let mut skip = vec![0u8; (size - 72) as usize];
                        r.read_exact(&mut skip)?;
                    }
                } else if size > 0 {
                    let mut skip = vec![0u8; size as usize];
                    r.read_exact(&mut skip)?;
                }
            }
            chunk_id::SLAN => {
                let size = read_i32(r)?;
                let slan = read_slicer_angle(r, size)?;
                model.slicer_angles.push(slan);
            }
            chunk_id::SIZE => {
                let size = read_i32(r)?;
                // SIZE chunk contains per-point sizes for the current contour
                if size > 0 {
                    let mut skip = vec![0u8; size as usize];
                    r.read_exact(&mut skip)?;
                }
            }
            chunk_id::MOST => {
                let size = read_i32(r)?;
                if size > 0 {
                    let mut data = vec![0u8; size as usize];
                    r.read_exact(&mut data)?;
                    model.store = data;
                }
            }
            chunk_id::OBST => {
                let size = read_i32(r)?;
                if size > 0 {
                    let mut data = vec![0u8; size as usize];
                    r.read_exact(&mut data)?;
                    if let Some(ref mut obj) = current_obj {
                        obj.store = data;
                    }
                }
            }
            chunk_id::COST | chunk_id::MEST => {
                // Contour store / mesh store: read and skip for now
                let size = read_i32(r)?;
                if size > 0 {
                    let mut skip = vec![0u8; size as usize];
                    r.read_exact(&mut skip)?;
                }
            }
            chunk_id::IEOF => {
                if let Some(obj) = current_obj.take() {
                    model.objects.push(obj);
                }
                break;
            }
            other => {
                // Unknown/unrecognised chunk (includes niche types like SYNA,
                // GRAF, etc.): read size + raw bytes and preserve for round-trip.
                let size = read_i32(r)?;
                let mut data = vec![0u8; size.max(0) as usize];
                if size > 0 {
                    r.read_exact(&mut data)?;
                }
                model.unknown_chunks.push((other, data));
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
        imat: None,
        clips: None,
        store: Vec::new(),
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

fn read_point3f<R: Read>(r: &mut R) -> Result<Point3f, ImodError> {
    let x = read_f32(r)?;
    let y = read_f32(r)?;
    let z = read_f32(r)?;
    Ok(Point3f { x, y, z })
}

fn read_imat<R: Read>(r: &mut R) -> Result<ImatData, ImodError> {
    let ambient = read_u8(r)?;
    let diffuse = read_u8(r)?;
    let specular = read_u8(r)?;
    let shininess = read_u8(r)?;
    let fillred = read_u8(r)?;
    let fillgreen = read_u8(r)?;
    let fillblue = read_u8(r)?;
    let quality = read_u8(r)?;
    let mat2 = read_u32(r)?;
    let valblack = read_u8(r)?;
    let valwhite = read_u8(r)?;
    let matflags2 = read_u8(r)?;
    let mesh_thickness = read_u8(r)?;
    Ok(ImatData {
        ambient, diffuse, specular, shininess,
        fillred, fillgreen, fillblue, quality,
        mat2, valblack, valwhite, matflags2, mesh_thickness,
    })
}

fn read_clip<R: Read>(r: &mut R, size: i32) -> Result<IclipPlanes, ImodError> {
    let count = read_u8(r)?;
    let flags = read_u8(r)?;
    let trans = read_u8(r)?;
    let plane = read_u8(r)?;
    // Each plane has a normal (Point3f) and a point (Point3f) = 24 bytes
    // Remaining bytes after 4-byte header: size - 4
    let num_planes = count.max(1) as usize;
    let mut normals = Vec::with_capacity(num_planes);
    let mut points = Vec::with_capacity(num_planes);
    for _ in 0..num_planes {
        normals.push(read_point3f(r)?);
    }
    for _ in 0..num_planes {
        points.push(read_point3f(r)?);
    }
    // Skip any remaining data
    let read_bytes = 4 + (num_planes as i32) * 24;
    if size > read_bytes {
        let mut skip = vec![0u8; (size - read_bytes) as usize];
        r.read_exact(&mut skip)?;
    }
    Ok(IclipPlanes { count, flags, trans, plane, normals, points })
}

fn read_view<R: Read>(r: &mut R, size: i32) -> Result<ImodView, ImodError> {
    // VIEW chunk: fovy(4) + rad(4) + aspect(4) + cnear(4) + cfar(4) +
    // rot(12) + trans(12) + scale(12) + world(4) + label(32) = 92 bytes minimum
    let fovy = read_f32(r)?;
    let rad = read_f32(r)?;
    let aspect = read_f32(r)?;
    let cnear = read_f32(r)?;
    let cfar = read_f32(r)?;
    let rot = read_point3f(r)?;
    let trans = read_point3f(r)?;
    let scale = read_point3f(r)?;
    let world = read_u32(r)?;
    let mut label_bytes = [0u8; 32];
    r.read_exact(&mut label_bytes)?;
    let label = String::from_utf8_lossy(&label_bytes)
        .trim_end_matches('\0')
        .to_string();
    // Skip remaining bytes
    let read_bytes = 92;
    if size > read_bytes {
        let mut skip = vec![0u8; (size - read_bytes) as usize];
        r.read_exact(&mut skip)?;
    }
    Ok(ImodView { fovy, rad, aspect, cnear, cfar, rot, trans, scale, world, label })
}

fn read_iref_image<R: Read>(r: &mut R) -> Result<IrefImage, ImodError> {
    let oscale = read_point3f(r)?;
    let otrans = read_point3f(r)?;
    let orot = read_point3f(r)?;
    let cscale = read_point3f(r)?;
    let ctrans = read_point3f(r)?;
    let crot = read_point3f(r)?;
    Ok(IrefImage { oscale, otrans, orot, cscale, ctrans, crot })
}

fn read_slicer_angle<R: Read>(r: &mut R, size: i32) -> Result<SlicerAngle, ImodError> {
    // SLAN: time(4) + angles(12) + center(12) + label(32) = 60 bytes
    let time = read_i32(r)?;
    let a0 = read_f32(r)?;
    let a1 = read_f32(r)?;
    let a2 = read_f32(r)?;
    let center = read_point3f(r)?;
    let mut label_bytes = [0u8; 32];
    r.read_exact(&mut label_bytes)?;
    let label = String::from_utf8_lossy(&label_bytes)
        .trim_end_matches('\0')
        .to_string();
    let read_bytes = 60;
    if size > read_bytes {
        let mut skip = vec![0u8; (size - read_bytes) as usize];
        r.read_exact(&mut skip)?;
    }
    Ok(SlicerAngle { time, angles: [a0, a1, a2], center, label })
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
