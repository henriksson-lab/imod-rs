//! Binary model-file routines from `IMOD/libimod/imodel_files.c`.
//!
//! IMOD binary chunks are big-endian irrespective of the host byte order.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::imodel::{
    IMOD_ERROR_CORRUPT, IMOD_ERROR_FORMAT, IMOD_ERROR_READ, IMOD_ERROR_WRITE, Icont, Imesh, Imod,
    Iobj, Iobjview, Ipoint, Iref_image, Iview,
};
use super::istore::{imod_read_store, imod_write_store};
use super::objgroup::{obj_group_list_write, obj_group_read};

const ID_IMOD: u32 = u32::from_be_bytes(*b"IMOD");
const IMOD_V12: u32 = u32::from_be_bytes(*b"V1.2");
const ID_OBJT: u32 = u32::from_be_bytes(*b"OBJT");
const ID_CONT: u32 = u32::from_be_bytes(*b"CONT");
const ID_MESH: u32 = u32::from_be_bytes(*b"MESH");
const ID_SIZE: u32 = u32::from_be_bytes(*b"SIZE");
const ID_CLIP: u32 = u32::from_be_bytes(*b"CLIP");
const ID_IMAT: u32 = u32::from_be_bytes(*b"IMAT");
const ID_VIEW: u32 = u32::from_be_bytes(*b"VIEW");
const ID_MCLP: u32 = u32::from_be_bytes(*b"MCLP");
const ID_OGRP: u32 = u32::from_be_bytes(*b"OGRP");
const ID_MOST: u32 = u32::from_be_bytes(*b"MOST");
const ID_OBST: u32 = u32::from_be_bytes(*b"OBST");
const ID_COST: u32 = u32::from_be_bytes(*b"COST");
const ID_MEST: u32 = u32::from_be_bytes(*b"MEST");

/// Original: `writeAsciiClips` (`imodel_files.c:1834`).
pub fn write_ascii_clips(
    file: &mut dyn Write,
    clips: &super::imodel::Iclip_planes,
    prefix: &str,
) -> Result<(), i32> {
    if clips.count != 0 {
        writeln!(
            file,
            "{prefix} {} {} {} {}",
            clips.count, clips.flags, clips.trans, clips.plane
        )
        .map_err(|_| IMOD_ERROR_WRITE)?;
        for index in 0..clips.count.min(7) as usize {
            let normal = clips.normal[index];
            let point = clips.point[index];
            writeln!(
                file,
                "{} {} {} {} {} {}",
                normal.x, normal.y, normal.z, point.x, point.y, point.z
            )
            .map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    Ok(())
}

/// Original: `imodWriteAscii` (`imodel_files.c:1631`).
pub fn imod_write_ascii(imod: &Imod, file: &mut dyn Write) -> Result<(), i32> {
    writeln!(file, "# imod ascii file version 2.0\n").map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(file, "imod {}", imod.obj.len()).map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(file, "max {} {} {}", imod.xmax, imod.ymax, imod.zmax)
        .map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(
        file,
        "offsets {} {} {}",
        imod.xoffset, imod.yoffset, imod.zoffset
    )
    .map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(file, "angles {} {} {}", imod.alpha, imod.beta, imod.gamma)
        .map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(
        file,
        "scale {} {} {}",
        imod.xscale, imod.yscale, imod.zscale
    )
    .map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(file, "mousemode  {}\ndrawmode   {}\nb&w_level  {},{}\nresolution {}\nthreshold  {}\npixsize    {}", imod.mousemode, imod.drawmode, imod.blacklevel, imod.whitelevel, imod.res, imod.thresh, imod.pixsize).map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(
        file,
        "units      {}",
        match imod.units {
            -10 => "um",
            -9 => "nm",
            -6 => "mm",
            -3 => "m",
            0 => "pixels",
            1 => "km",
            2 => "unknown units",
            _ => "unknown units",
        }
    )
    .map_err(|_| IMOD_ERROR_WRITE)?;
    writeln!(
        file,
        "flipped    {}",
        if imod.flags & super::imodel::IMODF_FLIPYZ != 0 {
            1
        } else {
            0
        }
    )
    .map_err(|_| IMOD_ERROR_WRITE)?;
    if let Some(reference) = imod.ref_image {
        writeln!(
            file,
            "refcurscale {} {} {}",
            reference.cscale.x, reference.cscale.y, reference.cscale.z
        )
        .map_err(|_| IMOD_ERROR_WRITE)?;
        writeln!(
            file,
            "refcurtrans {} {} {}",
            reference.ctrans.x, reference.ctrans.y, reference.ctrans.z
        )
        .map_err(|_| IMOD_ERROR_WRITE)?;
        if imod.flags & super::imodel::IMODF_TILTOK != 0 {
            writeln!(
                file,
                "refcurrot {} {} {}",
                reference.crot.x, reference.crot.y, reference.crot.z
            )
            .map_err(|_| IMOD_ERROR_WRITE)?;
        }
        if imod.flags & super::imodel::IMODF_OTRANS_ORIGIN != 0 {
            writeln!(
                file,
                "refoldtrans {} {} {}",
                reference.otrans.x, reference.otrans.y, reference.otrans.z
            )
            .map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    for angle in &imod.slicer_ang {
        let label_text = String::from_utf8_lossy(&angle.label);
        let label = label_text.trim_end_matches('\0');
        writeln!(
            file,
            "slicerAngle {} {} {} {} {} {} {} {}",
            angle.time,
            angle.angles[0],
            angle.angles[1],
            angle.angles[2],
            angle.center.x,
            angle.center.y,
            angle.center.z,
            label
        )
        .map_err(|_| IMOD_ERROR_WRITE)?;
    }
    writeln!(file, "currentview  {}", imod.cview).map_err(|_| IMOD_ERROR_WRITE)?;
    for (iv, view) in imod.view.iter().enumerate().skip(1) {
        writeln!(file, "view {}\nviewfovy  {}\nviewcnear {}\nviewcfar  {}\nviewflags {}\nviewtrans {} {} {}\nviewrot {} {} {}\nviewlight {} {}\ndepthcue {} {}\nviewlabel {}", iv, view.fovy, view.cnear, view.cfar, view.world, view.trans.x, view.trans.y, view.trans.z, view.rot.x, view.rot.y, view.rot.z, view.lightx, view.lighty, view.dcstart, view.dcend, String::from_utf8_lossy(&view.label).trim_end_matches('\0')).map_err(|_| IMOD_ERROR_WRITE)?;
        write_ascii_clips(file, &view.clips, "globalclips")?;
    }
    for (ob, obj) in imod.obj.iter().enumerate() {
        let real_meshes = obj
            .mesh
            .iter()
            .filter(|mesh| (mesh.flag >> 24) & 0x3f == 0)
            .count();
        writeln!(
            file,
            "\nobject {} {} {}\nname {}\ncolor {} {} {} {}",
            ob,
            obj.cont.len(),
            real_meshes,
            obj.name,
            obj.red,
            obj.green,
            obj.blue,
            obj.trans
        )
        .map_err(|_| IMOD_ERROR_WRITE)?;
        if obj.fillred != 0 || obj.fillgreen != 0 || obj.fillblue != 0 {
            writeln!(
                file,
                "Fillcolor {} {} {}",
                obj.fillred, obj.fillgreen, obj.fillblue
            )
            .map_err(|_| IMOD_ERROR_WRITE)?;
        }
        for (flag, line) in [
            (super::imodel::IMOD_OBJFLAG_OPEN, "open"),
            (super::imodel::IMOD_OBJFLAG_SCAT, "scattered"),
            (super::imodel::IMOD_OBJFLAG_OFF, "nodraw"),
            (super::imodel::IMOD_OBJFLAG_OUT, "insideout"),
            (1 << 8, "fill"),
            (1 << 10, "drawmesh"),
            (1 << 11, "nolines"),
            (1 << 19, "bothsides"),
            (1 << 14, "usefill"),
            (1 << 6, "pntusefill"),
            (1 << 7, "pntonsec"),
            (1 << 15, "antialias"),
            (1 << 18, "hastimes"),
            (1 << 12, "usevalue"),
            (1 << 17, "valcolor"),
        ] {
            if obj.flags & flag != 0 {
                writeln!(file, "{line}").map_err(|_| IMOD_ERROR_WRITE)?;
            }
        }
        writeln!(file, "linewidth {}\nsurfsize  {}\npointsize {}\naxis      {}\ndrawmode  {}\nwidth2D   {}\nsymbol    {}\nsymsize   {}\nsymflags  {}", obj.linewidth, obj.surfsize, obj.pdrawsize, obj.axis, obj.drawmode, obj.linewidth2, obj.symbol, obj.symsize, obj.symflags).map_err(|_| IMOD_ERROR_WRITE)?;
        writeln!(file, "ambient   {}\ndiffuse   {}\nspecular  {}\nshininess {}\nobquality {}\nvalblack  {}\nvalwhite  {}\nmeshthick {}\nmatflags2 {}", obj.ambient, obj.diffuse, obj.specular, obj.shininess, obj.quality, obj.valblack, obj.valwhite, obj.mesh_thickness, obj.matflags2).map_err(|_| IMOD_ERROR_WRITE)?;
        write_ascii_clips(file, &obj.clips, "objclips")?;
        let mut valmin = 0.;
        let mut valmax = 0.;
        if super::istore::istore_get_min_max(&obj.store, 11, &mut valmin, &mut valmax) != 0 {
            writeln!(file, "valminmax {} {}", valmin, valmax).map_err(|_| IMOD_ERROR_WRITE)?;
        }
        for (co, cont) in obj.cont.iter().enumerate() {
            let mut def_props = super::istore::DrawProps::default();
            let mut cont_props = super::istore::DrawProps::default();
            let mut cont_state = 0;
            let mut surf_state = 0;
            super::istore::istore_default_draw_props(obj, &mut def_props);
            super::istore::istore_cont_surf_draw_props(
                &obj.store,
                &def_props,
                &mut cont_props,
                co as i32,
                cont.surf,
                &mut cont_state,
                &mut surf_state,
            );
            if cont_state & (1 << 9) != 0 {
                writeln!(
                    file,
                    "contour {} {} {} {}",
                    co,
                    cont.surf,
                    cont.pts.len(),
                    cont_props.value1
                )
                .map_err(|_| IMOD_ERROR_WRITE)?;
            } else {
                writeln!(file, "contour {} {} {}", co, cont.surf, cont.pts.len())
                    .map_err(|_| IMOD_ERROR_WRITE)?;
            }
            let mut cursor = 0;
            let mut state = 0;
            let mut changes = 0;
            let mut props = super::istore::DrawProps::default();
            let mut next = super::istore::istore_first_change_index(&cont.store);
            for (pt, point) in cont.pts.iter().enumerate() {
                if pt as i32 == next {
                    next = super::istore::istore_next_change(
                        &cont.store,
                        &mut cursor,
                        &cont_props,
                        &mut props,
                        &mut state,
                        &mut changes,
                    );
                }
                let size = cont.sizes.get(pt).copied().unwrap_or(-1.);
                if state & (1 << 9) != 0 {
                    writeln!(
                        file,
                        "{} {} {} {} {}",
                        point.x, point.y, point.z, size, props.value1
                    )
                    .map_err(|_| IMOD_ERROR_WRITE)?;
                } else if size >= 0. {
                    writeln!(file, "{} {} {} {}", point.x, point.y, point.z, size)
                        .map_err(|_| IMOD_ERROR_WRITE)?;
                } else {
                    writeln!(file, "{} {} {}", point.x, point.y, point.z)
                        .map_err(|_| IMOD_ERROR_WRITE)?;
                }
            }
            if cont.flags != 0 {
                writeln!(file, "contflags {}", cont.flags).map_err(|_| IMOD_ERROR_WRITE)?;
            }
            if cont.time != 0 {
                writeln!(file, "conttime {}", cont.time).map_err(|_| IMOD_ERROR_WRITE)?;
            }
        }
        let mut real_mesh = 0;
        for mesh in &obj.mesh {
            if (mesh.flag >> 24) & 0x3f != 0 {
                continue;
            }
            writeln!(
                file,
                "mesh {} {} {}",
                real_mesh,
                mesh.vert.len(),
                mesh.list.len()
            )
            .map_err(|_| IMOD_ERROR_WRITE)?;
            real_mesh += 1;
            for point in &mesh.vert {
                writeln!(file, "{} {} {}", point.x, point.y, point.z)
                    .map_err(|_| IMOD_ERROR_WRITE)?;
            }
            for index in &mesh.list {
                writeln!(file, "{index}").map_err(|_| IMOD_ERROR_WRITE)?;
            }
            if mesh.flag != 0 {
                writeln!(file, "Meshflags {}", mesh.flag).map_err(|_| IMOD_ERROR_WRITE)?;
            }
            if mesh.time != 0 {
                writeln!(file, "Meshtime {}", mesh.time).map_err(|_| IMOD_ERROR_WRITE)?;
            }
            if mesh.surf != 0 {
                writeln!(file, "Meshsurf {}", mesh.surf).map_err(|_| IMOD_ERROR_WRITE)?;
            }
        }
    }
    writeln!(file, "# end of IMOD model").map_err(|_| IMOD_ERROR_WRITE)
}
const ID_IMNX: u32 = u32::from_be_bytes(*b"MINX");
const ID_IEOF: u32 = u32::from_be_bytes(*b"IEOF");

/// Original: `imodGetInt` (`imodel_files.c:2074`).
pub fn imod_get_int(file: &mut File) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_be_bytes(bytes))
}

/// Original: `imodGetFloat` (`imodel_files.c:2018`).
pub fn imod_get_float(file: &mut File) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_bits(u32::from_be_bytes(bytes)))
}

/// Original: `imodGetShort` (`imodel_files.c:2106`).
pub fn imod_get_short(file: &mut File) -> io::Result<i16> {
    let mut bytes = [0_u8; 2];
    file.read_exact(&mut bytes)?;
    Ok(i16::from_be_bytes(bytes))
}

/// Original: `imodGetByte` (`imodel_files.c:2129`).
pub fn imod_get_byte(file: &mut File) -> io::Result<u8> {
    let mut bytes = [0_u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

/// Original: `imodPutInt` (`imodel_files.c:2090`).
pub fn imod_put_int(file: &mut File, value: i32) -> io::Result<()> {
    file.write_all(&value.to_be_bytes())
}

/// Original: `imodPutFloat` (`imodel_files.c:2039`).
pub fn imod_put_float(file: &mut File, value: f32) -> io::Result<()> {
    file.write_all(&value.to_bits().to_be_bytes())
}

/// Original: `imodPutShort` (`imodel_files.c:2115`).
pub fn imod_put_short(file: &mut File, value: i16) -> io::Result<()> {
    file.write_all(&value.to_be_bytes())
}

/// Original: `imodPutByte` (`imodel_files.c:2146`).
pub fn imod_put_byte(file: &mut File, value: u8) -> io::Result<()> {
    file.write_all(&[value])
}

/// Original: `imodel_read_header` (`imodel_files.c:835`).
pub fn imodel_read_header(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
    let mut name = [0_u8; 128];
    file.read_exact(&mut name).map_err(|_| IMOD_ERROR_READ)?;
    imod.name = String::from_utf8_lossy(&name)
        .trim_end_matches('\0')
        .to_owned();
    imod.xmax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.ymax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zmax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.obj
        .reserve(imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?.max(0) as usize);
    imod.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    imod.drawmode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.mousemode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.blacklevel = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.whitelevel = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.xoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.yoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.xscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.yscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.object = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.contour = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.point = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.res = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.thresh = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.pixsize = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.units = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let _ = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.alpha = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.beta = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.gamma = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    Ok(())
}

/// Original: `imodel_read_object` (`imodel_files.c:860`).
pub fn imodel_read_object(file: &mut File) -> Result<Iobj, i32> {
    let mut name = [0_u8; 64];
    file.read_exact(&mut name).map_err(|_| IMOD_ERROR_READ)?;
    let mut extra = [0_u32; 16];
    for value in &mut extra {
        *value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    }
    let contsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    let axis = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let drawmode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let red = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    let green = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    let blue = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    let pdrawsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let symbol = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let mut symsize = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    if symsize == 0 {
        symsize = 3;
    }
    let mut linewidth2 = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    if linewidth2 == 0 {
        linewidth2 = 1;
    }
    let linewidth = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let linesty = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let symflags = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let sympad = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let trans = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let meshsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let surfsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let mut object = Iobj {
        name: String::from_utf8_lossy(&name)
            .trim_end_matches('\0')
            .to_owned(),
        extra,
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
        linesty,
        symflags,
        sympad,
        trans,
        surfsize,
        ..Iobj::default()
    };
    object.cont.reserve(contsize.max(0) as usize);
    object.mesh.reserve(meshsize.max(0) as usize);
    Ok(object)
}

/// Original: `imodel_read_contour` (`imodel_files.c:984`).
pub fn imodel_read_contour(file: &mut File) -> Result<Icont, i32> {
    let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    let time = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let surf = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if size < 0 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let mut cont = Icont {
        flags,
        time,
        surf,
        ..Icont::default()
    };
    for _ in 0..size {
        cont.pts.push(Ipoint {
            x: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            y: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            z: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
        });
    }
    Ok(cont)
}

/// Original: `imodel_read_ptsizes` (`imodel_files.c:1022`).
pub fn imodel_read_ptsizes(cont: &mut Icont, file: &mut File) -> Result<(), i32> {
    let byte_count = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if byte_count != (cont.pts.len() * 4) as i32 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for _ in 0..cont.pts.len() {
        cont.sizes
            .push(imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?);
    }
    Ok(())
}

/// Original: `imodel_read_mesh` (`imodel_files.c:1043`).
pub fn imodel_read_mesh(file: &mut File) -> Result<Imesh, i32> {
    let vertices = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let lists = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let flag = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    let time = imod_get_short(file).map_err(|_| IMOD_ERROR_READ)?;
    let surf = imod_get_short(file).map_err(|_| IMOD_ERROR_READ)?;
    if vertices < 0 || lists < 0 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let mut mesh = Imesh {
        flag,
        time,
        surf,
        ..Imesh::default()
    };
    for _ in 0..vertices {
        mesh.vert.push(Ipoint {
            x: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            y: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            z: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
        });
    }
    for _ in 0..lists {
        mesh.list
            .push(imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?);
    }
    Ok(mesh)
}

/// Original: `imodel_read_imat` (`imodel_files.c:1106`).
pub fn imodel_read_imat(object: &mut Iobj, file: &mut File, flags: u32) -> Result<(), i32> {
    let _ = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let mut value = [0; 4];
    file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
    object.ambient = value[0];
    object.diffuse = value[1];
    object.specular = value[2];
    object.shininess = value[3];
    if flags & (1 << 13) != 0 {
        file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
        object.fillred = value[0];
        object.fillgreen = value[1];
        object.fillblue = value[2];
        object.quality = value[3];
        object.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
        object.valblack = value[0];
        object.valwhite = value[1];
        object.matflags2 = value[2];
        object.mesh_thickness = value[3];
    } else {
        let value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        object.fillred = (value >> 24) as u8;
        object.fillgreen = (value >> 16) as u8;
        object.fillblue = (value >> 8) as u8;
        object.quality = value as u8;
        object.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        let value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        object.valblack = (value >> 24) as u8;
        object.valwhite = (value >> 16) as u8;
        object.matflags2 = (value >> 8) as u8;
    }
    if flags & (1 << 9) == 0 {
        object.mesh_thickness = 0;
    }
    Ok(())
}

/// Original: `imodel_read_clip` (`imodel_files.c:1099`).
pub fn imodel_read_clip(object: &mut Iobj, file: &mut File, _flags: u32) -> Result<(), i32> {
    let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if size < 4 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let mut head = [0; 4];
    file.read_exact(&mut head).map_err(|_| IMOD_ERROR_READ)?;
    object.clips.count = head[0];
    object.clips.flags = head[1];
    object.clips.trans = head[2];
    object.clips.plane = head[3];
    if object.clips.count > 7 || size != 4 + 24 * object.clips.count as i32 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for index in 0..object.clips.count as usize {
        for point in [
            &mut object.clips.normal[index],
            &mut object.clips.point[index],
        ] {
            point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        }
    }
    Ok(())
}

/// Original: `imodIMNXRead` (`iview.c:598`).
pub fn imod_imnx_read(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
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

/// Original: `imodViewModelRead` (`iview.c:254`), retaining native view values and object views.
pub fn imod_view_model_read(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
    let bytes = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if bytes == 4 {
        imod.cview = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        return Ok(());
    }
    if bytes < 176 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let start = file.stream_position().map_err(|_| IMOD_ERROR_READ)?;
    let mut view = Iview::default();
    view.fovy = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.rad = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.aspect = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.cnear = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.cfar = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    for point in [&mut view.rot, &mut view.trans, &mut view.scale] {
        point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    }
    for value in &mut view.mat {
        *value = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    }
    view.world = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    file.read_exact(&mut view.label)
        .map_err(|_| IMOD_ERROR_READ)?;
    view.dcstart = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.dcend = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.lightx = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.lighty = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    view.plax = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    if bytes >= 180 {
        let count = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        let total = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        if count < 0 || total < 0 || count > 0 && total / count < 67 {
            return Err(IMOD_ERROR_CORRUPT);
        }
        for _ in 0..count {
            let mut object = Iobjview::default();
            object.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
            object.red = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            object.green = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            object.blue = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            object.pdrawsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
            let mut head = [0; 3];
            file.read_exact(&mut head).map_err(|_| IMOD_ERROR_READ)?;
            object.linewidth = head[0];
            object.linesty = head[1];
            object.trans = head[2];
            let mut clip = [0; 4];
            file.read_exact(&mut clip).map_err(|_| IMOD_ERROR_READ)?;
            object.clips.count = clip[0];
            object.clips.flags = clip[1];
            object.clips.trans = clip[2];
            object.clips.plane = clip[3];
            for point in [&mut object.clips.normal[0], &mut object.clips.point[0]] {
                point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            }
            let mut material = [0; 4];
            file.read_exact(&mut material)
                .map_err(|_| IMOD_ERROR_READ)?;
            object.ambient = material[0];
            object.diffuse = material[1];
            object.specular = material[2];
            object.shininess = material[3];
            file.read_exact(&mut material)
                .map_err(|_| IMOD_ERROR_READ)?;
            object.fillred = material[0];
            object.fillgreen = material[1];
            object.fillblue = material[2];
            object.quality = material[3];
            object.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
            file.read_exact(&mut material)
                .map_err(|_| IMOD_ERROR_READ)?;
            object.valblack = material[0];
            object.valwhite = material[1];
            object.matflags2 = material[2];
            object.mesh_thickness = material[3];
            if count > 0 && total / count >= 43 + 24 * 7 {
                for index in 1..7 {
                    for point in [
                        &mut object.clips.normal[index],
                        &mut object.clips.point[index],
                    ] {
                        point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                        point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
                    }
                }
            }
            view.objview.push(object);
        }
    }
    file.seek(SeekFrom::Start(start + bytes as u64))
        .map_err(|_| IMOD_ERROR_READ)?;
    imod.view.push(view);
    Ok(())
}

/// Original: `imodViewClipRead` (`iview.c:386`).
pub fn imod_view_clip_read(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
    let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if size < 4 || imod.view.is_empty() {
        return Err(IMOD_ERROR_CORRUPT);
    }
    let mut head = [0; 4];
    file.read_exact(&mut head).map_err(|_| IMOD_ERROR_READ)?;
    let clips = &mut imod.view.last_mut().unwrap().clips;
    clips.count = head[0];
    clips.flags = head[1];
    clips.trans = head[2];
    clips.plane = head[3];
    if clips.count > 7 || size != 4 + clips.count as i32 * 24 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for index in 0..clips.count as usize {
        for point in [&mut clips.normal[index], &mut clips.point[index]] {
            point.x = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.y = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
            point.z = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
        }
    }
    Ok(())
}

/// Original: `imodel_read` (`imodel_files.c:555`).
pub fn imodel_read(imod: &mut Imod, file: &mut File, version: u32) -> Result<(), i32> {
    if version != IMOD_V12 {
        return Err(IMOD_ERROR_FORMAT);
    }
    imodel_read_header(imod, file)?;
    let mut object = None::<usize>;
    let mut contour = None::<usize>;
    loop {
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        match id {
            ID_OBJT => {
                imod.obj.push(imodel_read_object(file)?);
                object = Some(imod.obj.len() - 1);
                contour = None;
            }
            ID_CONT => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                imod.obj[index].cont.push(imodel_read_contour(file)?);
                contour = Some(imod.obj[index].cont.len() - 1);
            }
            ID_SIZE => {
                let (Some(object), Some(contour)) = (object, contour) else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                imodel_read_ptsizes(&mut imod.obj[object].cont[contour], file)?;
            }
            ID_MESH => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                imod.obj[index].mesh.push(imodel_read_mesh(file)?);
            }
            ID_MOST => {
                let mut error = 0;
                imod.store = imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }
            ID_OBST => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                let mut error = 0;
                imod.obj[index].store = imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }
            ID_COST => {
                let (Some(object), Some(contour)) = (object, contour) else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                let mut error = 0;
                imod.obj[object].cont[contour].store =
                    imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }
            ID_MEST => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                let Some(mesh) = imod.obj[index].mesh.last_mut() else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                let mut error = 0;
                mesh.store = imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }
            ID_IMAT => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                imodel_read_imat(&mut imod.obj[index], file, imod.flags)?;
            }
            ID_CLIP => {
                let Some(index) = object else {
                    return Err(IMOD_ERROR_CORRUPT);
                };
                imodel_read_clip(&mut imod.obj[index], file, imod.flags)?;
            }
            ID_VIEW => imod_view_model_read(imod, file)?,
            ID_MCLP => imod_view_clip_read(imod, file)?,
            ID_OGRP => obj_group_read(&mut imod.group_list, file)?,
            ID_IMNX => imod_imnx_read(imod, file)?,
            ID_IEOF => return Ok(()),
            _ => {
                let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
                if size < 0 || file.seek(SeekFrom::Current(size as i64)).is_err() {
                    return Err(IMOD_ERROR_READ);
                }
            }
        }
    }
}

/// Original: `imodReadFile` (`imodel_files.c:131`).
pub fn imod_read_file(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
    file.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;
    if imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32 != ID_IMOD {
        return imod_read_ascii(imod, file);
    }
    let version = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    imodel_read(imod, file, version)
}

/// Original: `imodReadAscii` (`imodel_files.c:1183`).
pub fn imod_read_ascii(imod: &mut Imod, file: &mut File) -> Result<(), i32> {
    file.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;
    let mut lines = BufReader::new(file).lines();
    let first = lines
        .next()
        .ok_or(IMOD_ERROR_FORMAT)
        .and_then(|line| line.map_err(|_| IMOD_ERROR_READ))?;
    let mut first_words = first.split_whitespace();
    if first_words.next() != Some("imod") {
        return Err(IMOD_ERROR_FORMAT);
    }
    let count = first_words
        .next()
        .ok_or(IMOD_ERROR_FORMAT)?
        .parse::<usize>()
        .map_err(|_| IMOD_ERROR_FORMAT)?;
    imod.obj = vec![Iobj::default(); count];
    let mut current = None::<usize>;
    while let Some(line) = lines.next() {
        let line = line.map_err(|_| IMOD_ERROR_READ)?;
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        if words[0] == "object" {
            let index = words
                .get(1)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let contours = words
                .get(2)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let meshes = words
                .get(3)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            if index >= imod.obj.len() {
                return Err(IMOD_ERROR_CORRUPT);
            }
            imod.obj[index].cont = vec![Icont::default(); contours];
            imod.obj[index].mesh = vec![Imesh::default(); meshes];
            current = Some(index);
            continue;
        }
        if words[0] == "contour" {
            let object = current.ok_or(IMOD_ERROR_CORRUPT)?;
            let index = words
                .get(1)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let surf = words
                .get(2)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<i32>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let points = words
                .get(3)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let contour = imod.obj[object]
                .cont
                .get_mut(index)
                .ok_or(IMOD_ERROR_CORRUPT)?;
            contour.surf = surf;
            for _ in 0..points {
                let line = lines
                    .next()
                    .ok_or(IMOD_ERROR_READ)
                    .and_then(|line| line.map_err(|_| IMOD_ERROR_READ))?;
                let values: Vec<f32> = line
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                if values.len() < 3 {
                    return Err(IMOD_ERROR_FORMAT);
                }
                contour.pts.push(Ipoint {
                    x: values[0],
                    y: values[1],
                    z: values[2],
                });
                if values.len() > 3 && values[3] >= 0. {
                    contour.sizes.push(values[3]);
                }
            }
            continue;
        }
        if words[0] == "mesh" {
            let object = current.ok_or(IMOD_ERROR_CORRUPT)?;
            let index = words
                .get(1)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let vertices = words
                .get(2)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let lists = words
                .get(3)
                .ok_or(IMOD_ERROR_FORMAT)?
                .parse::<usize>()
                .map_err(|_| IMOD_ERROR_FORMAT)?;
            let mesh = imod.obj[object]
                .mesh
                .get_mut(index)
                .ok_or(IMOD_ERROR_CORRUPT)?;
            for _ in 0..vertices {
                let line = lines
                    .next()
                    .ok_or(IMOD_ERROR_READ)
                    .and_then(|line| line.map_err(|_| IMOD_ERROR_READ))?;
                let values: Vec<f32> = line
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                if values.len() < 3 {
                    return Err(IMOD_ERROR_FORMAT);
                }
                mesh.vert.push(Ipoint {
                    x: values[0],
                    y: values[1],
                    z: values[2],
                });
            }
            for _ in 0..lists {
                mesh.list.push(
                    lines
                        .next()
                        .ok_or(IMOD_ERROR_READ)
                        .and_then(|line| line.map_err(|_| IMOD_ERROR_READ))?
                        .trim()
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?,
                );
            }
            continue;
        }
        match words[0] {
            "name" => {
                if let Some(object) = current {
                    imod.obj[object].name = words[1..].join(" ");
                }
            }
            "color" => {
                if let Some(object) = current {
                    let o = &mut imod.obj[object];
                    o.red = words
                        .get(1)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?;
                    o.green = words
                        .get(2)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?;
                    o.blue = words
                        .get(3)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?;
                    o.trans = words.get(4).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
            }
            "max" => {
                imod.xmax = words
                    .get(1)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                imod.ymax = words
                    .get(2)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                imod.zmax = words
                    .get(3)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
            }
            "scale" => {
                imod.xscale = words
                    .get(1)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                imod.yscale = words
                    .get(2)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
                imod.zscale = words
                    .get(3)
                    .ok_or(IMOD_ERROR_FORMAT)?
                    .parse()
                    .map_err(|_| IMOD_ERROR_FORMAT)?;
            }
            "refcurscale" | "refcurtrans" | "refcurrot" | "refoldtrans" => {
                let mut reference = imod.ref_image.unwrap_or_default();
                let point = Ipoint {
                    x: words
                        .get(1)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?,
                    y: words
                        .get(2)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?,
                    z: words
                        .get(3)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?,
                };
                if words[0] == "refcurscale" {
                    reference.cscale = point;
                } else if words[0] == "refcurtrans" {
                    reference.ctrans = point;
                } else if words[0] == "refcurrot" {
                    reference.crot = point;
                    imod.flags |= 1 << 15;
                } else {
                    reference.otrans = point;
                    imod.flags |= 1 << 14;
                }
                imod.ref_image = Some(reference);
            }
            "view" => imod.view.push(Iview::default()),
            "viewfovy" => {
                if let Some(view) = imod.view.last_mut() {
                    view.fovy = words
                        .get(1)
                        .ok_or(IMOD_ERROR_FORMAT)?
                        .parse()
                        .map_err(|_| IMOD_ERROR_FORMAT)?;
                }
            }
            "viewtrans" => {
                if let Some(view) = imod.view.last_mut() {
                    view.trans = Ipoint {
                        x: words
                            .get(1)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                        y: words
                            .get(2)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                        z: words
                            .get(3)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                    };
                }
            }
            "viewrot" => {
                if let Some(view) = imod.view.last_mut() {
                    view.rot = Ipoint {
                        x: words
                            .get(1)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                        y: words
                            .get(2)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                        z: words
                            .get(3)
                            .ok_or(IMOD_ERROR_FORMAT)?
                            .parse()
                            .map_err(|_| IMOD_ERROR_FORMAT)?,
                    };
                }
            }
            "currentview" => {
                imod.cview = words
                    .get(1)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0)
            }
            _ => {}
        }
    }
    Ok(())
}

/// Original: `imodRead` (`imodel_files.c:171`).
pub fn imod_read(path: impl AsRef<Path>) -> Result<Imod, i32> {
    let mut file = File::open(path).map_err(|_| IMOD_ERROR_READ)?;
    let mut imod = Imod::default();
    imod_read_file(&mut imod, &mut file)?;
    Ok(imod)
}

/// Original: `imodel_write_contour` (`imodel_files.c:449`).
pub fn imodel_write_contour(cont: &Icont, file: &mut File) -> Result<(), i32> {
    imod_put_int(file, ID_CONT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.pts.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.flags as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.time).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.surf).map_err(|_| IMOD_ERROR_WRITE)?;
    for point in &cont.pts {
        imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    if !cont.sizes.is_empty() {
        imod_put_int(file, ID_SIZE as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_int(file, (cont.sizes.len() * 4) as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        for size in &cont.sizes {
            imod_put_float(file, *size).map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    if imod_write_store(&cont.store, ID_COST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodel_write_mesh` (`imodel_files.c:480`).
pub fn imodel_write_mesh(mesh: &Imesh, file: &mut File) -> Result<(), i32> {
    imod_put_int(file, ID_MESH as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.vert.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.list.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.flag as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_short(file, mesh.time).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_short(file, mesh.surf).map_err(|_| IMOD_ERROR_WRITE)?;
    for point in &mesh.vert {
        imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for list in &mesh.list {
        imod_put_int(file, *list).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    if imod_write_store(&mesh.store, ID_MEST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodel_write_object` (`imodel_files.c:345`).
pub fn imodel_write_object(object: &Iobj, file: &mut File) -> Result<(), i32> {
    imod_put_int(file, ID_OBJT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    let mut name = [0_u8; 64];
    let source = object.name.as_bytes();
    name[..source.len().min(63)].copy_from_slice(&source[..source.len().min(63)]);
    file.write_all(&name).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in object.extra {
        imod_put_int(file, value as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_int(file, object.cont.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.flags as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.axis).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.drawmode).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.red).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.green).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.blue).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.pdrawsize).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in [
        object.symbol,
        object.symsize,
        object.linewidth2,
        object.linewidth,
        object.linesty,
        object.symflags,
        object.sympad,
    ] {
        imod_put_byte(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_byte(file, object.trans).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.mesh.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.surfsize).map_err(|_| IMOD_ERROR_WRITE)?;
    for contour in &object.cont {
        imodel_write_contour(contour, file)?;
    }
    for mesh in &object.mesh {
        imodel_write_mesh(mesh, file)?;
    }
    if object.clips.count > 0 {
        imod_put_int(file, ID_CLIP as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_int(file, 4 + object.clips.count as i32 * 24).map_err(|_| IMOD_ERROR_WRITE)?;
        file.write_all(&[
            object.clips.count,
            object.clips.flags,
            object.clips.trans,
            object.clips.plane,
        ])
        .map_err(|_| IMOD_ERROR_WRITE)?;
        for index in 0..object.clips.count.min(7) as usize {
            for point in [object.clips.normal[index], object.clips.point[index]] {
                imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
                imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
                imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
            }
        }
    }
    imod_put_int(file, ID_IMAT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 16).map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.ambient,
        object.diffuse,
        object.specular,
        object.shininess,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.fillred,
        object.fillgreen,
        object.fillblue,
        object.quality,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.mat2 as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.valblack,
        object.valwhite,
        object.matflags2,
        object.mesh_thickness,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    if imod_write_store(&object.store, ID_OBST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodViewModelWrite` / `imodViewWrite` (`iview.c:230`, `iview.c:136`).
pub fn imod_view_model_write(imod: &Imod, file: &mut File) -> Result<(), i32> {
    if imod.view.len() < 2 {
        return Ok(());
    }
    imod_put_int(file, ID_VIEW as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 4).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, imod.cview).map_err(|_| IMOD_ERROR_WRITE)?;
    for view in imod.view.iter().skip(1) {
        imod_put_int(file, ID_VIEW as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        let object_bytes = view.objview.len() * (43 + 24 * 7);
        let size = 176
            + if view.objview.is_empty() {
                0
            } else {
                8 + object_bytes
            };
        imod_put_int(file, size as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        for value in [
            view.fovy,
            view.rad,
            view.aspect,
            view.cnear,
            view.cfar,
            view.rot.x,
            view.rot.y,
            view.rot.z,
            view.trans.x,
            view.trans.y,
            view.trans.z,
            view.scale.x,
            view.scale.y,
            view.scale.z,
        ] {
            imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
        }
        for value in view.mat {
            imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
        }
        imod_put_int(file, view.world as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        file.write_all(&view.label).map_err(|_| IMOD_ERROR_WRITE)?;
        for value in [
            view.dcstart,
            view.dcend,
            view.lightx,
            view.lighty,
            view.plax,
        ] {
            imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
        }
        if !view.objview.is_empty() {
            imod_put_int(file, view.objview.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
            imod_put_int(file, object_bytes as i32).map_err(|_| IMOD_ERROR_WRITE)?;
            for object in &view.objview {
                imod_put_int(file, object.flags as i32).map_err(|_| IMOD_ERROR_WRITE)?;
                for value in [object.red, object.green, object.blue] {
                    imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
                }
                imod_put_int(file, object.pdrawsize).map_err(|_| IMOD_ERROR_WRITE)?;
                file.write_all(&[
                    object.linewidth,
                    object.linesty,
                    object.trans,
                    object.clips.count,
                    object.clips.flags,
                    object.clips.trans,
                    object.clips.plane,
                ])
                .map_err(|_| IMOD_ERROR_WRITE)?;
                for index in 0..7 {
                    for point in [object.clips.normal[index], object.clips.point[index]] {
                        imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
                        imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
                        imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
                    }
                }
                file.write_all(&[
                    object.ambient,
                    object.diffuse,
                    object.specular,
                    object.shininess,
                    object.fillred,
                    object.fillgreen,
                    object.fillblue,
                    object.quality,
                ])
                .map_err(|_| IMOD_ERROR_WRITE)?;
                imod_put_int(file, object.mat2 as i32).map_err(|_| IMOD_ERROR_WRITE)?;
                file.write_all(&[
                    object.valblack,
                    object.valwhite,
                    object.matflags2,
                    object.mesh_thickness,
                ])
                .map_err(|_| IMOD_ERROR_WRITE)?;
            }
        }
        if view.clips.count > 0 {
            imod_put_int(file, ID_MCLP as i32).map_err(|_| IMOD_ERROR_WRITE)?;
            imod_put_int(file, 4 + 24 * view.clips.count as i32).map_err(|_| IMOD_ERROR_WRITE)?;
            file.write_all(&[
                view.clips.count,
                view.clips.flags,
                view.clips.trans,
                view.clips.plane,
            ])
            .map_err(|_| IMOD_ERROR_WRITE)?;
            for index in 0..view.clips.count as usize {
                for point in [view.clips.normal[index], view.clips.point[index]] {
                    imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
                    imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
                    imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
                }
            }
        }
    }
    Ok(())
}

/// Original: `imodIMNXWrite` (`iview.c:625`).
pub fn imod_imnx_write(imod: &Imod, file: &mut File) -> Result<(), i32> {
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

/// Original: `imodel_write` (`imodel_files.c:276`).
pub fn imodel_write(imod: &Imod, file: &mut File) -> Result<(), i32> {
    file.seek(SeekFrom::Start(0))
        .map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, ID_IMOD as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, IMOD_V12 as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    let mut name = [0_u8; 128];
    let source = imod.name.as_bytes();
    name[..source.len().min(127)].copy_from_slice(&source[..source.len().min(127)]);
    file.write_all(&name).map_err(|_| IMOD_ERROR_WRITE)?;
    // `imodel_files.c:286` sets these bits before serializing the header.
    // They declare the byte material fields, multiple clip-plane support, and
    // mesh-thickness field emitted by this writer.
    let flags = imod.flags | (1 << 13) | (1 << 12) | (1 << 9);
    for value in [
        imod.xmax,
        imod.ymax,
        imod.zmax,
        imod.obj.len() as i32,
        flags as i32,
        imod.drawmode,
        imod.mousemode,
        imod.blacklevel,
        imod.whitelevel,
    ] {
        imod_put_int(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for value in [
        imod.xoffset,
        imod.yoffset,
        imod.zoffset,
        imod.xscale,
        imod.yscale,
        imod.zscale,
    ] {
        imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for value in [
        imod.cindex.object,
        imod.cindex.contour,
        imod.cindex.point,
        imod.res,
        imod.thresh,
    ] {
        imod_put_int(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_float(file, imod.pixsize).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, imod.units).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 0).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in [imod.alpha, imod.beta, imod.gamma] {
        imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for object in &imod.obj {
        imodel_write_object(object, file)?;
    }
    imod_view_model_write(imod, file)?;
    imod_imnx_write(imod, file)?;
    obj_group_list_write(&imod.group_list, file)?;
    if imod_write_store(&imod.store, ID_MOST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    imod_put_int(file, ID_IEOF as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    file.flush().map_err(|_| IMOD_ERROR_WRITE)
}

/// Original: `imodWrite` (`imodel_files.c:249`).
pub fn imod_write(imod: &Imod, file: &mut File) -> Result<(), i32> {
    imodel_write(imod, file)
}

/// Original: `imodFileWrite` (`imodel_files.c:77`).
pub fn imod_file_write(imod: &Imod, path: impl AsRef<Path>) -> Result<(), i32> {
    let mut file = File::create(path).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_write(imod, &mut file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_group_ogrp_chunk_round_trips_in_a_real_binary_model() {
        let path = std::env::temp_dir().join(format!("imod-rs-ogrp-{}.mod", std::process::id()));
        let model = Imod {
            group_list: vec![super::super::imodel::Iobj_group {
                name: *b"selected objects\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
                obj_list: vec![0, 2, 5],
            }],
            ..Imod::default()
        };
        imod_file_write(&model, &path).unwrap();
        let decoded = imod_read(&path).unwrap();
        assert_eq!(decoded.group_list, model.group_list);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_sets_source_format_flags_and_skips_default_view_chunk() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-imodel-write-default-view-{}",
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();
        imodel_write(&Imod::default(), &mut file).unwrap();
        drop(file);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            i32::from_be_bytes(bytes[152..156].try_into().unwrap()),
            (1 << 13) | (1 << 12) | (1 << 9)
        );
        assert_eq!(bytes.len(), 244);
        assert_eq!(&bytes[240..], b"IEOF");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn binary_imod_roundtrip_keeps_header_contours_sizes_and_meshes() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-imodel-files-{}", std::process::id()));
        let imod = Imod {
            name: "binary fixture".into(),
            xmax: 10,
            ymax: 20,
            zmax: 30,
            xscale: 1.0,
            yscale: 1.0,
            zscale: 2.0,
            pixsize: 1.5,
            units: -9,
            obj: vec![Iobj {
                name: "object".into(),
                red: 1.0,
                surfsize: 2,
                cont: vec![Icont {
                    surf: 2,
                    sizes: vec![3.0, 4.0],
                    pts: vec![
                        Ipoint {
                            x: 1.0,
                            y: 2.0,
                            z: 3.0,
                        },
                        Ipoint {
                            x: 4.0,
                            y: 5.0,
                            z: 6.0,
                        },
                    ],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    vert: vec![Ipoint {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }],
                    list: vec![-1],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&imod, &path).unwrap();
        let actual = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(actual.name, imod.name);
        assert_eq!(actual.obj[0].cont[0].pts, imod.obj[0].cont[0].pts);
        assert_eq!(actual.obj[0].cont[0].sizes, imod.obj[0].cont[0].sizes);
        assert_eq!(actual.obj[0].mesh[0].list, imod.obj[0].mesh[0].list);
    }

    #[test]
    fn binary_imod_roundtrip_keeps_model_object_contour_and_mesh_stores() {
        let path = std::env::temp_dir().join(format!("imod-rs-store-{}.mod", std::process::id()));
        let store = super::super::istore::Istore {
            type_: 10,
            flags: 1 << 2,
            index: super::super::istore::StoreUnion { i: 1 },
            value: super::super::istore::StoreUnion { f: 2.5 },
        };
        let model = Imod {
            store: vec![store],
            obj: vec![Iobj {
                store: vec![store],
                cont: vec![Icont {
                    pts: vec![Ipoint::default()],
                    store: vec![store],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    store: vec![store],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&model, &path).unwrap();
        let actual = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for list in [
            &actual.store,
            &actual.obj[0].store,
            &actual.obj[0].cont[0].store,
            &actual.obj[0].mesh[0].store,
        ] {
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].type_, 10);
            assert_eq!(unsafe { list[0].value.f }, 2.5);
        }
    }

    #[test]
    fn reads_bundled_etomo_fiducial_model() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
        let model = imod_read(fixture).unwrap();
        assert!(!model.obj.is_empty());
        assert!(model.obj.iter().any(|object| !object.cont.is_empty()));
    }

    #[test]
    fn extended_binary_chunks_roundtrip() {
        let path = std::env::temp_dir().join(format!("imod-rs-ext-{}", std::process::id()));
        let mut view = Iview::default();
        view.fovy = 33.;
        view.clips.count = 1;
        view.clips.point[0] = Ipoint {
            x: 5.,
            y: 6.,
            z: 7.,
        };
        let imod = Imod {
            flags: (1 << 13) | (1 << 9),
            cview: 1,
            view: vec![Iview::default(), view],
            ref_image: Some(Iref_image {
                ctrans: Ipoint {
                    x: 4.,
                    y: 5.,
                    z: 6.,
                },
                ..Iref_image::default()
            }),
            obj: vec![Iobj {
                mat2: 11,
                mesh_thickness: 15,
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&imod, &path).unwrap();
        let output = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(output.obj[0].mat2, 11);
        assert_eq!(output.obj[0].mesh_thickness, 15);
        assert_eq!(output.cview, 1);
        assert_eq!(output.view.len(), 2);
        assert_eq!(
            output.view[1].clips.point[0],
            Ipoint {
                x: 5.,
                y: 6.,
                z: 7.
            }
        );
        assert_eq!(
            output.ref_image.unwrap().ctrans,
            Ipoint {
                x: 4.,
                y: 5.,
                z: 6.
            }
        );
    }

    #[test]
    fn ascii_model_preserves_geometry_view_and_reference_data() {
        let path = std::env::temp_dir().join(format!("imod-rs-ascii-{}", std::process::id()));
        std::fs::write(&path, "imod 1\nmax 10 20 30\nscale 1 2 3\nrefcurscale 2 3 4\nrefcurtrans 5 6 7\nobject 0 1 0\nname old model object\ncolor 0.1 0.2 0.3 4\ncontour 0 2 2\n1 2 3 4\n5 6 7 8\nview 1\nviewfovy 33\nviewtrans 2 3 4\ncurrentview 1\n").unwrap();
        let output = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(output.obj[0].name, "old model object");
        assert_eq!(output.obj[0].cont[0].pts.len(), 2);
        assert_eq!(output.obj[0].cont[0].sizes, vec![4., 8.]);
        assert_eq!(output.view.len(), 2);
        assert_eq!(output.view[1].fovy, 33.);
        assert_eq!(
            output.ref_image.unwrap().ctrans,
            Ipoint {
                x: 5.,
                y: 6.,
                z: 7.
            }
        );
    }
}
