use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{read_model, ImodModel, ImodObject};

/// Convert an IMOD model to VRML V1.0 format.
///
/// Exports meshes, contours, and scattered-point spheres from an IMOD model
/// to the Virtual Reality Modeling Language (VRML 1.0) format.
#[derive(Parser)]
#[command(name = "imod2vrml", version, about)]
struct Args {
    /// Input IMOD model file
    input: String,

    /// Output VRML file (.wrl)
    output: String,

    /// Use low-resolution meshes (if any exist)
    #[arg(short = 'l')]
    low_res: bool,

    /// Output separate point/normal list for each polygon
    #[arg(short = 's')]
    separate: bool,
}

// IMOD object flag bits
const IMOD_OBJFLAG_OFF: u32 = 1 << 1;
const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;
const IMOD_OBJFLAG_SCAT: u32 = 1 << 9;
const IMOD_OBJFLAG_FILL: u32 = 1 << 8;
const IMOD_OBJFLAG_LINE: u32 = 1 << 11;

// Mesh index commands
const IMOD_MESH_END: i32 = -22;
const IMOD_MESH_ENDPOLY: i32 = -23;
const IMOD_MESH_BGNPOLYNORM: i32 = -21;
const IMOD_MESH_BGNPOLYNORM2: i32 = -24;

fn is_scat(flags: u32) -> bool { (flags & IMOD_OBJFLAG_SCAT) != 0 }
fn is_off(flags: u32) -> bool { (flags & IMOD_OBJFLAG_OFF) != 0 }
fn is_fill(flags: u32) -> bool { (flags & IMOD_OBJFLAG_FILL) != 0 }
fn is_line(flags: u32) -> bool { (flags & IMOD_OBJFLAG_LINE) != 0 }
fn is_close(flags: u32) -> bool { (flags & IMOD_OBJFLAG_OPEN) == 0 }

fn point_get_size(obj: &ImodObject, cont_idx: usize, pt_idx: usize) -> f32 {
    let cont = &obj.contours[cont_idx];
    if let Some(ref sizes) = cont.sizes {
        if pt_idx < sizes.len() && sizes[pt_idx] >= 0.0 {
            return sizes[pt_idx];
        }
    }
    if obj.pdrawsize > 0 {
        return obj.pdrawsize as f32;
    }
    0.0
}

fn has_spheres(obj: &ImodObject) -> bool {
    if is_scat(obj.flags) || obj.pdrawsize > 0 {
        return true;
    }
    for cont in &obj.contours {
        if cont.sizes.is_some() {
            return true;
        }
    }
    false
}

fn mesh_poly_norm_factors(code: i32) -> Option<(usize, usize, i32)> {
    match code {
        IMOD_MESH_BGNPOLYNORM => Some((3, 1, 0)),
        IMOD_MESH_BGNPOLYNORM2 => Some((1, 0, 1)),
        _ => None,
    }
}

fn write_material(w: &mut impl Write, obj: &ImodObject) {
    let diffuse = 255.0 / 255.0; // default IMOD diffuse
    let specular = 127.0 / 255.0;
    let mut shininess = 64.0 / 255.0;
    if shininess < 0.01 {
        shininess = 0.01;
    }
    let (r, g, b) = (obj.red, obj.green, obj.blue);

    writeln!(w, "\tMaterial{{").ok();
    writeln!(w, "\t\tambientColor {} {} {}", r, g, b).ok();
    writeln!(w, "\t\tdiffuseColor {} {} {}", r * diffuse, g * diffuse, b * diffuse).ok();
    writeln!(w, "\t\tspecularColor {} {} {}", r * specular, g * specular, b * specular).ok();
    writeln!(w, "\t\temissiveColor 0 0 0").ok();
    writeln!(w, "\t\tshininess {}", shininess).ok();
    writeln!(w, "\t\ttransparency {}", obj.trans as f32 / 100.0).ok();
    writeln!(w, "\t}} #Material").ok();
}

fn write_camera(w: &mut impl Write, model: &ImodModel) {
    let (mut minp, mut maxp) = model_bounds(model);
    let zscale = model.scale.z;
    maxp.z *= zscale;
    minp.z *= zscale;
    let r = ((maxp.x - minp.x).powi(2) + (maxp.y - minp.y).powi(2) + (maxp.z - minp.z).powi(2)).sqrt();
    let xpos = (maxp.x - minp.x) * 0.5 + minp.x;
    let ypos = (maxp.y - minp.y) * 0.5 + minp.y;
    let zpos = (maxp.z - minp.z) * 0.5 + minp.z;

    writeln!(w, "\tDEF Viewer Info {{ string \"examiner\" }}").ok();
    writeln!(w, "\tTranslation {{ \n\t\ttranslation {} {} {}\n\t}}", -xpos, -ypos, -zpos).ok();
    writeln!(w, "\tPerspectiveCamera {{").ok();
    writeln!(w, "\t\tposition 0 0 {}", r).ok();
    writeln!(w, "\t\torientation 0 0 0 0").ok();
    writeln!(w, "\t\tfocalDistance {}", r).ok();
    writeln!(w, "\t}}").ok();
}

fn model_bounds(model: &ImodModel) -> (Point3f, Point3f) {
    let mut minp = Point3f { x: f32::MAX, y: f32::MAX, z: f32::MAX };
    let mut maxp = Point3f { x: f32::MIN, y: f32::MIN, z: f32::MIN };
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                minp.x = minp.x.min(pt.x);
                minp.y = minp.y.min(pt.y);
                minp.z = minp.z.min(pt.z);
                maxp.x = maxp.x.max(pt.x);
                maxp.y = maxp.y.max(pt.y);
                maxp.z = maxp.z.max(pt.z);
            }
        }
        for mesh in &obj.meshes {
            for (i, v) in mesh.vertices.iter().enumerate() {
                if i % 2 != 0 { continue; } // skip normals
                minp.x = minp.x.min(v.x);
                minp.y = minp.y.min(v.y);
                minp.z = minp.z.min(v.z);
                maxp.x = maxp.x.max(v.x);
                maxp.y = maxp.y.max(v.y);
                maxp.z = maxp.z.max(v.z);
            }
        }
    }
    (minp, maxp)
}

fn write_scat_contours(w: &mut impl Write, obj: &ImodObject, zscale: f32) {
    let obj_has_spheres = is_scat(obj.flags) || obj.pdrawsize > 0;
    for (c, cont) in obj.contours.iter().enumerate() {
        if cont.points.is_empty() {
            continue;
        }
        if !obj_has_spheres && cont.sizes.is_none() {
            continue;
        }
        for (p, pt) in cont.points.iter().enumerate() {
            let size = point_get_size(obj, c, p);
            if size > 0.0 {
                writeln!(
                    w,
                    "\tDEF PntDat Separator {{ Translation {{ translation {} {} {}}}Sphere {{ radius {} }} }}",
                    pt.x, pt.y, pt.z * zscale, size,
                ).ok();
            }
        }
    }
}

fn write_contours(w: &mut impl Write, obj: &ImodObject, zscale: f32) {
    for cont in &obj.contours {
        if cont.points.is_empty() {
            continue;
        }
        writeln!(w, "\tDEF ContourData Coordinate3 {{").ok();
        writeln!(w, "\t\tpoint [").ok();
        for (pt_i, pt) in cont.points.iter().enumerate() {
            let sep = if pt_i == cont.points.len() - 1 { ']' } else { ',' };
            writeln!(w, "\t\t{} {} {}{}", pt.x, pt.y, pt.z * zscale, sep).ok();
        }
        writeln!(w, "\t}}").ok();

        if is_line(obj.flags) {
            writeln!(w, "\tIndexedLineSet {{").ok();
            write!(w, "\t\tcoordIndex [").ok();
            for pt_i in 0..cont.points.len() {
                write!(w, "{},", pt_i).ok();
                if pt_i % 10 == 9 {
                    writeln!(w).ok();
                }
            }
            if is_close(obj.flags) {
                write!(w, "0").ok();
            }
            writeln!(w, "]\n\t}}").ok();
        } else {
            writeln!(w, "\tPointSet {{\n\t\tstartIndex 0\n\t\tnumPoints -1\n\t}}").ok();
        }
    }
}

fn write_filled_contours(w: &mut impl Write, obj: &ImodObject, zscale: f32) {
    for cont in &obj.contours {
        if cont.points.len() < 3 {
            continue;
        }
        write_material(w, obj);

        writeln!(w, "\tDEF ContourData Coordinate3 {{").ok();
        writeln!(w, "\t\tpoint [").ok();
        for (pt_i, pt) in cont.points.iter().enumerate() {
            let sep = if pt_i == cont.points.len() - 1 { ']' } else { ',' };
            writeln!(w, "\t\t{} {} {}{}", pt.x, pt.y, pt.z * zscale, sep).ok();
        }
        writeln!(w, "\t}}").ok();

        writeln!(w, "\tIndexedFaceSet {{").ok();
        write!(w, "\t\tcoordIndex [").ok();
        for pt_i in 0..cont.points.len() {
            write!(w, "{},", pt_i).ok();
            if pt_i % 10 == 9 {
                writeln!(w).ok();
            }
        }
        if is_close(obj.flags) {
            write!(w, "0").ok();
        }
        writeln!(w, "]\n\t}}").ok();

        if is_line(obj.flags) {
            write_material(w, obj);
            writeln!(w, "\tIndexedLineSet {{").ok();
            write!(w, "\t\tcoordIndex [").ok();
            for pt_i in 0..cont.points.len() {
                write!(w, "{},", pt_i).ok();
                if pt_i % 10 == 9 {
                    writeln!(w).ok();
                }
            }
            write!(w, "0").ok();
            writeln!(w, "]\n\t}}").ok();
        }
    }
}

fn write_mesh(w: &mut impl Write, obj: &ImodObject, ob: usize, separate: bool, zscale: f32) {
    if is_fill(obj.flags) {
        writeln!(w, "\tNormalBinding {{ value PER_VERTEX }}").ok();
    } else {
        writeln!(w, "\tNormalBinding {{ value OVERALL }}").ok();
    }

    for (me, mesh) in obj.meshes.iter().enumerate() {
        let verts = &mesh.vertices;
        let indices = &mesh.indices;

        // Output coordinates (unless separate mode for fill/line)
        if !separate || !(is_fill(obj.flags) || is_line(obj.flags)) {
            writeln!(w, "\tDEF Obj{}Mesh{}Data  Coordinate3 {{", ob, me).ok();
            writeln!(w, "\t\tpoint [").ok();
            for i in (0..verts.len()).step_by(2) {
                let sep = if i >= verts.len() - 2 { ']' } else { ',' };
                writeln!(w, "{:.5} {:.5} {:.5}{}", verts[i].x, verts[i].y, verts[i].z * zscale, sep).ok();
            }
            writeln!(w, "\t}}").ok();
        }

        // Output normals if fill and not separate
        if !separate && is_fill(obj.flags) {
            writeln!(w, "\tDEF Obj{}Mesh{}NData Normal {{", ob, me).ok();
            writeln!(w, "\t\tvector [").ok();
            for i in (1..verts.len()).step_by(2) {
                let n = &verts[i];
                let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
                let (nx, ny, nz) = if len > 1e-12 {
                    (n.x / len, n.y / len, n.z / len)
                } else {
                    (0.0, 0.0, 1.0)
                };
                let sep = if i >= verts.len() - 1 { ']' } else { ',' };
                writeln!(w, "{:.3} {:.3} {:.3}{}", nx, ny, nz, sep).ok();
            }
            writeln!(w, "\t}}").ok();
        }

        if is_fill(obj.flags) || is_line(obj.flags) {
            // Walk index list for polygon groups
            let mut i = 0;
            while i < indices.len() {
                if indices[i] == IMOD_MESH_END {
                    break;
                }
                if let Some((list_inc, vert_base, _norm_add)) = mesh_poly_norm_factors(indices[i]) {
                    i += 1;
                    let mut ilist = Vec::new();
                    while i < indices.len() && indices[i] != IMOD_MESH_ENDPOLY {
                        ilist.push(indices[i + vert_base] / 2);
                        i += list_inc;
                        if i >= indices.len() { break; }
                        ilist.push(indices[i + vert_base] / 2);
                        i += list_inc;
                        if i >= indices.len() { break; }
                        ilist.push(indices[i + vert_base] / 2);
                        i += list_inc;
                    }
                    if i < indices.len() && indices[i] == IMOD_MESH_ENDPOLY {
                        i += 1;
                    }

                    if ilist.is_empty() {
                        continue;
                    }
                    // Face set
                    if is_fill(obj.flags) {
                        write_material(w, obj);
                        writeln!(w, "\tIndexedFaceSet {{").ok();
                        write!(w, "\t\tcoordIndex [").ok();
                        let ntri = ilist.len() / 3;
                        for idx in 0..ntri {
                            let ind = 3 * idx;
                            let sep = if idx >= ntri - 1 { ']' } else { ',' };
                            writeln!(w, "{},{},{},-1{}", ilist[ind], ilist[ind + 1], ilist[ind + 2], sep).ok();
                        }
                        writeln!(w, "\t}}").ok();
                    }

                    // Line set
                    if is_line(obj.flags) {
                        write_material(w, obj);
                        writeln!(w, "\tIndexedLineSet {{").ok();
                        write!(w, "\t\tcoordIndex [").ok();
                        let ntri = ilist.len() / 3;
                        for idx in 0..ntri {
                            let ind = 3 * idx;
                            let sep = if idx >= ntri - 1 { ']' } else { ',' };
                            writeln!(w, "{},{},{},{},-1{}", ilist[ind], ilist[ind + 1], ilist[ind + 2], ilist[ind], sep).ok();
                        }
                        writeln!(w, "\t}}").ok();
                    }
                } else {
                    i += 1;
                }
            }
        } else {
            // Point set
            write_material(w, obj);
            writeln!(w, "\tPointSet {{\n\t\tstartIndex 0\n\t\tnumPoints -1\n\t}}").ok();
        }
    }
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2vrml - Reading model {}: {}", args.input, e);
        process::exit(3);
    });

    let out_file = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2vrml - Could not open {}: {}", args.output, e);
        process::exit(10);
    });
    let mut w = BufWriter::new(out_file);

    let zscale = model.scale.z;

    writeln!(w, "#VRML V1.0 ascii").ok();
    writeln!(w, "\nSeparator {{").ok();

    // Info
    writeln!(w, "\tSeparator {{").ok();
    writeln!(w, "\t\tInfo {{").ok();
    writeln!(w, "\t\t\tstring \"Created by Imod\"").ok();
    writeln!(w, "\t\t}}").ok();
    writeln!(w, "\t}}").ok();

    // Light
    writeln!(w, "\tDirectionalLight {{\n\t\tdirection 1 -1 -1  \n\t}}").ok();

    // Camera
    write_camera(&mut w, &model);

    // Objects
    for (ob, obj) in model.objects.iter().enumerate() {
        if is_off(obj.flags) {
            continue;
        }

        writeln!(w, "DEF Object{}Data Separator {{", ob).ok();

        let obj_has_spheres = has_spheres(obj);

        // Scattered points first
        if obj_has_spheres {
            write_material(&mut w, obj);
            write_scat_contours(&mut w, obj, zscale);
        }

        // Mesh or contours
        if !obj.meshes.is_empty() {
            write_mesh(&mut w, obj, ob, args.separate, zscale);
        } else if is_fill(obj.flags) && is_close(obj.flags) {
            write_filled_contours(&mut w, obj, zscale);
        } else if !is_scat(obj.flags) {
            write_material(&mut w, obj);
            write_contours(&mut w, obj, zscale);
        }

        writeln!(w, "}} #Object{}Data", ob).ok();
    }

    writeln!(w, "}}").ok();
}
