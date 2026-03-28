use std::io::{self, Write};
use std::path::Path;

use imod_core::ImodError;

use crate::types::*;

/// Write an IMOD model file.
pub fn write_model(path: impl AsRef<Path>, model: &ImodModel) -> Result<(), ImodError> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = io::BufWriter::new(file);
    write_model_to(&mut w, model)
}

pub fn write_model_to<W: Write>(w: &mut W, model: &ImodModel) -> Result<(), ImodError> {
    // File ID
    write_u32(w, chunk_id::IMOD)?;
    write_u32(w, chunk_id::V1_2)?;

    // Model header
    let mut name_bytes = [0u8; 128];
    let name = model.name.as_bytes();
    let len = name.len().min(127);
    name_bytes[..len].copy_from_slice(&name[..len]);
    w.write_all(&name_bytes)?;

    write_i32(w, model.xmax)?;
    write_i32(w, model.ymax)?;
    write_i32(w, model.zmax)?;
    write_i32(w, model.objects.len() as i32)?;
    write_u32(w, model.flags)?;
    write_i32(w, model.drawmode)?;
    write_i32(w, model.mousemode)?;
    write_i32(w, model.black_level)?;
    write_i32(w, model.white_level)?;
    write_f32(w, model.offset.x)?;
    write_f32(w, model.offset.y)?;
    write_f32(w, model.offset.z)?;
    write_f32(w, model.scale.x)?;
    write_f32(w, model.scale.y)?;
    write_f32(w, model.scale.z)?;
    // cindex
    write_i32(w, 0)?;
    write_i32(w, 0)?;
    write_i32(w, 0)?;
    // res, thresh
    write_i32(w, 3)?;
    write_i32(w, 128)?;
    write_f32(w, model.pixel_size)?;
    write_i32(w, model.units)?;
    write_i32(w, 0)?; // csum
    write_f32(w, 0.0)?; // alpha
    write_f32(w, 0.0)?; // beta
    write_f32(w, 0.0)?; // gamma

    // Write objects
    for obj in &model.objects {
        write_object(w, obj)?;
    }

    // IEOF
    write_u32(w, chunk_id::IEOF)?;
    w.flush()?;

    Ok(())
}

fn write_object<W: Write>(w: &mut W, obj: &ImodObject) -> Result<(), ImodError> {
    write_u32(w, chunk_id::OBJT)?;

    // Name
    let mut name_bytes = [0u8; 64];
    let name = obj.name.as_bytes();
    let len = name.len().min(63);
    name_bytes[..len].copy_from_slice(&name[..len]);
    w.write_all(&name_bytes)?;

    // Extra (16 ints = 64 bytes, all zeros)
    w.write_all(&[0u8; 64])?;

    write_i32(w, obj.contours.len() as i32)?;
    write_u32(w, obj.flags)?;
    write_i32(w, obj.axis)?;
    write_i32(w, obj.drawmode)?;
    write_f32(w, obj.red)?;
    write_f32(w, obj.green)?;
    write_f32(w, obj.blue)?;
    write_i32(w, obj.pdrawsize)?;

    w.write_all(&[obj.symbol])?;
    w.write_all(&[obj.symsize])?;
    w.write_all(&[obj.linewidth2])?;
    w.write_all(&[obj.linewidth])?;
    w.write_all(&[obj.linestyle])?;
    w.write_all(&[0u8])?; // symflags
    w.write_all(&[0u8])?; // sympad
    w.write_all(&[obj.trans])?;

    write_i32(w, obj.meshes.len() as i32)?;
    write_i32(w, 0)?; // surfsize

    // Write contours
    for cont in &obj.contours {
        write_contour(w, cont)?;
    }

    // Write meshes
    for mesh in &obj.meshes {
        write_mesh(w, mesh)?;
    }

    Ok(())
}

fn write_contour<W: Write>(w: &mut W, cont: &ImodContour) -> Result<(), ImodError> {
    write_u32(w, chunk_id::CONT)?;
    write_i32(w, cont.points.len() as i32)?;
    write_u32(w, cont.flags)?;
    write_i32(w, cont.time)?;
    write_i32(w, cont.surf)?;

    for pt in &cont.points {
        write_f32(w, pt.x)?;
        write_f32(w, pt.y)?;
        write_f32(w, pt.z)?;
    }

    Ok(())
}

fn write_mesh<W: Write>(w: &mut W, mesh: &ImodMesh) -> Result<(), ImodError> {
    write_u32(w, chunk_id::MESH)?;
    write_i32(w, mesh.vertices.len() as i32)?;
    write_i32(w, mesh.indices.len() as i32)?;
    write_u32(w, mesh.flag)?;
    write_i16(w, mesh.time)?;
    write_i16(w, mesh.surf)?;

    for v in &mesh.vertices {
        write_f32(w, v.x)?;
        write_f32(w, v.y)?;
        write_f32(w, v.z)?;
    }

    for &idx in &mesh.indices {
        write_i32(w, idx)?;
    }

    Ok(())
}

// Big-endian writing helpers
fn write_i16<W: Write>(w: &mut W, v: i16) -> Result<(), ImodError> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}

fn write_i32<W: Write>(w: &mut W, v: i32) -> Result<(), ImodError> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), ImodError> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<(), ImodError> {
    w.write_all(&v.to_be_bytes())?;
    Ok(())
}
