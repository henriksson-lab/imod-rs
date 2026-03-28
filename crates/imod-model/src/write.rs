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

    // Write MINX (ref image)
    if let Some(ref ref_image) = model.ref_image {
        write_u32(w, chunk_id::IMNX)?;
        write_i32(w, 72)?;
        write_point3f(w, &ref_image.oscale)?;
        write_point3f(w, &ref_image.otrans)?;
        write_point3f(w, &ref_image.orot)?;
        write_point3f(w, &ref_image.cscale)?;
        write_point3f(w, &ref_image.ctrans)?;
        write_point3f(w, &ref_image.crot)?;
    }

    // Write views
    for view in &model.views {
        write_view(w, view)?;
    }

    // Write slicer angles
    for slan in &model.slicer_angles {
        write_slicer_angle(w, slan)?;
    }

    // Write MCLP (model-level clipping planes)
    if let Some(ref clip) = model.clips {
        let num_planes = clip.count.max(1) as usize;
        let size = 4 + (num_planes as i32) * 24;
        write_u32(w, chunk_id::MCLP)?;
        write_i32(w, size)?;
        w.write_all(&[clip.count, clip.flags, clip.trans, clip.plane])?;
        for n in &clip.normals {
            write_point3f(w, n)?;
        }
        for p in &clip.points {
            write_point3f(w, p)?;
        }
    }

    // Write model store
    if !model.store.is_empty() {
        write_u32(w, chunk_id::MOST)?;
        write_i32(w, model.store.len() as i32)?;
        w.write_all(&model.store)?;
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

    // Write IMAT
    if let Some(ref imat) = obj.imat {
        write_u32(w, chunk_id::IMAT)?;
        write_i32(w, 16)?;
        w.write_all(&[imat.ambient, imat.diffuse, imat.specular, imat.shininess])?;
        w.write_all(&[imat.fillred, imat.fillgreen, imat.fillblue, imat.quality])?;
        write_u32(w, imat.mat2)?;
        w.write_all(&[imat.valblack, imat.valwhite, imat.matflags2, imat.mesh_thickness])?;
    }

    // Write CLIP
    if let Some(ref clip) = obj.clips {
        let num_planes = clip.count.max(1) as usize;
        let size = 4 + (num_planes as i32) * 24;
        write_u32(w, chunk_id::CLIP)?;
        write_i32(w, size)?;
        w.write_all(&[clip.count, clip.flags, clip.trans, clip.plane])?;
        for n in &clip.normals {
            write_point3f(w, n)?;
        }
        for p in &clip.points {
            write_point3f(w, p)?;
        }
    }

    // Write object store
    if !obj.store.is_empty() {
        write_u32(w, chunk_id::OBST)?;
        write_i32(w, obj.store.len() as i32)?;
        w.write_all(&obj.store)?;
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

fn write_point3f<W: Write>(w: &mut W, p: &imod_core::Point3f) -> Result<(), ImodError> {
    write_f32(w, p.x)?;
    write_f32(w, p.y)?;
    write_f32(w, p.z)?;
    Ok(())
}

fn write_view<W: Write>(w: &mut W, view: &ImodView) -> Result<(), ImodError> {
    write_u32(w, chunk_id::VIEW)?;
    write_i32(w, 92)?;
    write_f32(w, view.fovy)?;
    write_f32(w, view.rad)?;
    write_f32(w, view.aspect)?;
    write_f32(w, view.cnear)?;
    write_f32(w, view.cfar)?;
    write_point3f(w, &view.rot)?;
    write_point3f(w, &view.trans)?;
    write_point3f(w, &view.scale)?;
    write_u32(w, view.world)?;
    let mut label_bytes = [0u8; 32];
    let lbl = view.label.as_bytes();
    let len = lbl.len().min(31);
    label_bytes[..len].copy_from_slice(&lbl[..len]);
    w.write_all(&label_bytes)?;
    Ok(())
}

fn write_slicer_angle<W: Write>(w: &mut W, slan: &SlicerAngle) -> Result<(), ImodError> {
    write_u32(w, chunk_id::SLAN)?;
    write_i32(w, 60)?;
    write_i32(w, slan.time)?;
    write_f32(w, slan.angles[0])?;
    write_f32(w, slan.angles[1])?;
    write_f32(w, slan.angles[2])?;
    write_point3f(w, &slan.center)?;
    let mut label_bytes = [0u8; 32];
    let lbl = slan.label.as_bytes();
    let len = lbl.len().min(31);
    label_bytes[..len].copy_from_slice(&lbl[..len]);
    w.write_all(&label_bytes)?;
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
