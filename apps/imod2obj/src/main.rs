use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{read_model, ImodMesh, ImodModel, ImodObject};

/// Convert an IMOD model to Wavefront OBJ format.
///
/// Exports meshes and scattered-point spheres from an IMOD model file
/// to OBJ format, with optional MTL material library generation.
#[derive(Parser)]
#[command(name = "imod2obj", version, about)]
struct Args {
    /// Input IMOD model file
    input: String,

    /// Output OBJ file
    output: String,

    /// Output MTL material file (optional)
    mtl: Option<String>,

    /// Output low-resolution meshes (if any exist)
    #[arg(short = 'l')]
    low_res: bool,

    /// Output all objects (including those switched off)
    #[arg(short = 'a')]
    all_objects: bool,

    /// Rotate model 90 degrees (flip Y and Z axes)
    #[arg(short = 'r')]
    rotate: bool,

    /// Output normals
    #[arg(short = 'n')]
    normals: bool,

    /// Flip normals
    #[arg(short = 'f')]
    flip_normals: bool,

    /// Only print spheres for scattered objects
    #[arg(short = 'o')]
    only_scat_spheres: bool,

    /// Number of segments per sphere (default 8)
    #[arg(short = 's', default_value_t = 8)]
    sphere_segments: usize,

    /// Use icosahedrons instead of standard sphere meshes
    #[arg(short = 'i')]
    icosahedrons: bool,
}

// IMOD object flag bits
const IMOD_OBJFLAG_OFF: u32 = 1 << 1;
const IMOD_OBJFLAG_SCAT: u32 = 1 << 9;

// IMOD mesh index-list commands
const IMOD_MESH_END: i32 = -22;
const IMOD_MESH_ENDPOLY: i32 = -23;
const IMOD_MESH_BGNPOLYNORM: i32 = -21;
const IMOD_MESH_BGNPOLYNORM2: i32 = -24;

fn is_scat(flags: u32) -> bool {
    (flags & IMOD_OBJFLAG_SCAT) != 0
}
fn is_off(flags: u32) -> bool {
    (flags & IMOD_OBJFLAG_OFF) != 0
}

/// Counters for summary output.
struct Stats {
    num_objs: usize,
    num_vertices: usize,
    num_faces: usize,
    num_normals: usize,
    num_spheres: usize,
}

fn safe_obj_name(obj: &ImodObject, ob: usize) -> String {
    let mut name = format!("obj{}_", ob + 1);
    for c in obj.name.chars() {
        match c {
            ' ' | '\t' | '\n' => name.push('_'),
            '(' | '[' | '{' => name.push('<'),
            ')' | ']' | '}' => name.push('>'),
            '\'' | '"' => name.push('`'),
            '#' | '.' | ',' | '\\' | ':' | '+' | '&' | ';' | '|' => name.push('_'),
            '\0' => break,
            other => name.push(other),
        }
    }
    name
}

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

/// Decode a mesh polygon start code into (list_inc, vert_base, norm_add).
fn mesh_poly_norm_factors(code: i32) -> Option<(usize, usize, i32)> {
    match code {
        IMOD_MESH_BGNPOLYNORM => Some((3, 1, 0)),
        IMOD_MESH_BGNPOLYNORM2 => Some((1, 0, 1)),
        _ => None,
    }
}

fn write_mesh(
    w: &mut impl Write,
    _obj: &ImodObject,
    mesh: &ImodMesh,
    args: &Args,
    stats: &mut Stats,
    zscale: f32,
) {
    let verts = &mesh.vertices;

    // Vertices come in pairs: vertex, normal (interleaved)
    let f_vert = stats.num_vertices;
    for i in (0..verts.len()).step_by(2) {
        let v = &verts[i];
        if args.rotate {
            writeln!(w, "v {:.5} {:.5} {:.5}", v.x, v.z * zscale, v.y).ok();
        } else {
            writeln!(w, "v {:.5} {:.5} {:.5}", v.x, v.y, v.z * zscale).ok();
        }
        stats.num_vertices += 1;
    }

    let f_norm = stats.num_normals;
    if args.normals {
        writeln!(w).ok();
        for i in (1..verts.len()).step_by(2) {
            let n = &verts[i];
            let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
            let (nx, ny, nz) = if len > 1e-12 {
                (n.x / len, n.y / len, n.z / len)
            } else {
                (0.0, 0.0, 1.0)
            };
            if args.flip_normals {
                writeln!(w, "vn {:.3} {:.3} {:.3}", nx, ny, nz).ok();
            } else {
                writeln!(w, "vn {:.3} {:.3} {:.3}", -nx, -ny, -nz).ok();
            }
            stats.num_normals += 1;
        }
    }

    // Walk the index list for polygon strips
    writeln!(w).ok();
    let indices = &mesh.indices;
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

            // Output faces
            let ntri = ilist.len() / 3;
            for t in 0..ntri {
                let idx = t * 3;
                let (a, b, c) = (
                    ilist[idx + 2] as usize + f_vert + 1,
                    ilist[idx + 1] as usize + f_vert + 1,
                    ilist[idx] as usize + f_vert + 1,
                );
                if args.normals {
                    let (na, nb, nc) = (
                        ilist[idx + 2] as usize + f_norm + 1,
                        ilist[idx + 1] as usize + f_norm + 1,
                        ilist[idx] as usize + f_norm + 1,
                    );
                    writeln!(w, "f {}//{} {}//{} {}//{}", a, na, b, nb, c, nc).ok();
                } else {
                    writeln!(w, "f {} {} {}", a, b, c).ok();
                }
                stats.num_faces += 1;
            }
        } else {
            i += 1;
        }
    }
}

fn write_sphere(
    w: &mut impl Write,
    pt: Point3f,
    radius: f32,
    segments: usize,
    args: &Args,
    stats: &mut Stats,
) {
    if segments < 4 {
        return;
    }
    let n_pitch = segments / 2 + 1;
    let pitch_inc = PI / n_pitch as f32;
    let seg_inc = 2.0 * PI / segments as f32;

    // Top and bottom vertices
    writeln!(w, "v {:.5} {:.5} {:.5}", pt.x, pt.y + radius, pt.z).ok();
    writeln!(w, "v {:.5} {:.5} {:.5}", pt.x, pt.y - radius, pt.z).ok();
    stats.num_vertices += 2;

    let f_vert = stats.num_vertices;
    for p in 1..n_pitch {
        let out = (radius * (p as f32 * pitch_inc).sin()).abs();
        let y = radius * (p as f32 * pitch_inc).cos();
        for s in 0..segments {
            let x = out * (s as f32 * seg_inc).cos();
            let z = out * (s as f32 * seg_inc).sin();
            writeln!(w, "v {:.5} {:.5} {:.5}", x + pt.x, y + pt.y, z + pt.z).ok();
            stats.num_vertices += 1;
        }
    }
    writeln!(w).ok();

    // Normals (if requested)
    let f_norm = stats.num_normals;
    if args.normals {
        writeln!(w, "vn 0.0 1.0 0.0").ok();
        writeln!(w, "vn 0.0 -1.0 0.0").ok();
        stats.num_normals += 2;
        for p in 1..n_pitch {
            let out_n = (1.0f32 * (p as f32 * pitch_inc).sin()).abs();
            let y_n = (p as f32 * pitch_inc).cos();
            for s in 0..segments {
                let x_n = out_n * (s as f32 * seg_inc).cos();
                let z_n = out_n * (s as f32 * seg_inc).sin();
                let x_n = if x_n.abs() < 0.001 { 0.0 } else { x_n };
                let z_n = if z_n.abs() < 0.001 { 0.0 } else { z_n };
                writeln!(w, "vn {:.5} {:.5} {:.5}", x_n, y_n, z_n).ok();
                stats.num_normals += 1;
            }
        }
        writeln!(w).ok();
    }

    // Square faces between intermediate points
    for p in 1..n_pitch - 1 {
        for s in 0..segments {
            let i = p * segments + s;
            let j = if s == segments - 1 { i - segments } else { i };
            if args.normals {
                writeln!(
                    w,
                    "f {}//{} {}//{} {}//{} {}//{}",
                    i + 1 - segments + f_vert, i + 1 - segments + f_norm,
                    j + 2 - segments + f_vert, j + 2 - segments + f_norm,
                    j + 2 + f_vert, j + 2 + f_norm,
                    i + 1 + f_vert, i + 1 + f_norm,
                ).ok();
            } else {
                writeln!(
                    w, "f {} {} {} {}",
                    i + 1 - segments + f_vert,
                    j + 2 - segments + f_vert,
                    j + 2 + f_vert,
                    i + 1 + f_vert,
                ).ok();
            }
            stats.num_faces += 1;
        }
    }

    // Triangle faces to top and bottom
    let off_last_verts = f_vert + segments * (n_pitch - 2);
    let off_last_norms = f_norm + segments * (n_pitch - 2);
    for s in 0..segments {
        let j = if s == segments - 1 {
            -(1i64)
        } else {
            s as i64
        };
        let j_idx = (j + 1) as usize;
        if args.normals {
            writeln!(
                w, "f {}//{} {}//{} {}//{}",
                f_vert - 1, f_norm - 1,
                j_idx + 1 + f_vert, j_idx + 1 + f_norm,
                s + 1 + f_vert, s + 1 + f_norm,
            ).ok();
            writeln!(
                w, "f {}//{} {}//{} {}//{}",
                f_vert, f_norm,
                s + 1 + off_last_verts, s + 1 + off_last_norms,
                j_idx + 1 + off_last_verts, j_idx + 1 + off_last_norms,
            ).ok();
        } else {
            writeln!(
                w, "f {} {} {}",
                f_vert - 1, j_idx + 1 + f_vert, s + 1 + f_vert,
            ).ok();
            writeln!(
                w, "f {} {} {}",
                f_vert, s + 1 + off_last_verts, j_idx + 1 + off_last_verts,
            ).ok();
        }
        stats.num_faces += 2;
    }
    writeln!(w).ok();
}

fn write_icosahedron(
    w: &mut impl Write,
    pt: Point3f,
    radius: f32,
    stats: &mut Stats,
) {
    let ico_verts: [(f32, f32, f32); 12] = [
        (0.000, 1.000, 0.000),
        (0.894, 0.447, 0.000),
        (0.276, 0.447, 0.851),
        (-0.724, 0.447, 0.526),
        (-0.724, 0.447, -0.526),
        (0.276, 0.447, -0.851),
        (0.724, -0.447, 0.526),
        (-0.276, -0.447, 0.851),
        (-0.894, -0.447, 0.000),
        (-0.276, -0.447, -0.851),
        (0.724, -0.447, -0.526),
        (0.000, -1.000, 0.000),
    ];

    for (vx, vy, vz) in &ico_verts {
        writeln!(
            w, "v {:.5} {:.5} {:.5}",
            vx * radius + pt.x, vy * radius + pt.y, vz * radius + pt.z,
        ).ok();
        stats.num_vertices += 1;
    }
    writeln!(w).ok();

    // 20 triangular faces using relative (negative) indices
    let faces = [
        (-12, -10, -11), (-12, -9, -10), (-12, -8, -9), (-12, -7, -8), (-12, -11, -7),
        (-1, -6, -5), (-1, -5, -4), (-1, -4, -3), (-1, -3, -2), (-1, -2, -6),
        (-11, -10, -6), (-10, -9, -5), (-9, -8, -4), (-8, -7, -3), (-7, -11, -2),
        (-6, -10, -5), (-5, -9, -4), (-4, -8, -3), (-3, -7, -2), (-2, -11, -6),
    ];
    for (a, b, c) in &faces {
        writeln!(w, "f {} {} {}", a, b, c).ok();
    }
    stats.num_faces += 20;
    writeln!(w).ok();
}

fn write_scat_contours(
    w: &mut impl Write,
    obj: &ImodObject,
    ob: usize,
    args: &Args,
    stats: &mut Stats,
    zscale: f32,
) {
    writeln!(w, "# obj{} SPHERES:\n", ob + 1).ok();

    let obj_has_spheres = is_scat(obj.flags) || obj.pdrawsize > 0;

    for (c, cont) in obj.contours.iter().enumerate() {
        if cont.points.is_empty() {
            continue;
        }
        if !obj_has_spheres && cont.sizes.is_none() {
            continue;
        }

        for (p, pt) in cont.points.iter().enumerate() {
            writeln!(w, "#   cont {} pt {}", c + 1, p + 1).ok();
            writeln!(w, "g obj_{}_cont_{}_pt_{}", ob + 1, c + 1, p + 1).ok();

            let mut out_pt = Point3f {
                x: pt.x,
                y: pt.y,
                z: pt.z * zscale,
            };
            if args.rotate {
                out_pt.y = pt.z * zscale;
                out_pt.z = pt.y;
            }

            let radius = point_get_size(obj, c, p);

            if args.icosahedrons {
                write_icosahedron(w, out_pt, radius, stats);
            } else {
                write_sphere(w, out_pt, radius, args.sphere_segments, args, stats);
            }
            stats.num_spheres += 1;
        }
    }
    writeln!(w).ok();
}

fn write_mtl(w: &mut impl Write, model: &ImodModel) {
    writeln!(w, "# WaveFront *.mtl file (generated from an IMOD model by imod2obj)").ok();
    writeln!(w).ok();

    for (ob, obj) in model.objects.iter().enumerate() {
        let name = safe_obj_name(obj, ob);
        // Without IMAT data we use defaults matching IMOD defaults
        let ambient = 102.0 / 255.0;
        let diffuse = 255.0 / 255.0;
        let specular = 127.0 / 255.0;
        let mut shininess = 64.0 / 255.0;
        if shininess < 0.01 {
            shininess = 0.01;
        }
        let (r, g, b) = (obj.red, obj.green, obj.blue);

        writeln!(w, "\n#MATERIAL FOR OBJECT {}:", ob + 1).ok();
        writeln!(w, "newmtl {}", name).ok();
        writeln!(w, "Ka {} {} {}", r * ambient, g * ambient, b * ambient).ok();
        writeln!(w, "Kd {} {} {}", r * diffuse, g * diffuse, b * diffuse).ok();
        writeln!(w, "Ks {} {} {}", r * specular, g * specular, b * specular).ok();
        writeln!(w, "Ns {}", shininess * 1000.0).ok();
        writeln!(w, "d {}", obj.trans as f32 / 100.0).ok();
        writeln!(w, "Tr {}", obj.trans as f32 / 100.0).ok();
    }
    writeln!(w).ok();
    writeln!(w, "# For more info on MTL file format see:").ok();
    writeln!(w, "#  http://en.wikipedia.org/wiki/Material_Template_Library").ok();
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2obj - Reading model {}: {}", args.input, e);
        process::exit(3);
    });

    let out_file = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2obj - Could not open {}: {}", args.output, e);
        process::exit(10);
    });
    let mut w = BufWriter::new(out_file);

    let zscale = model.scale.z;
    let has_mtl = args.mtl.is_some();

    let mut stats = Stats {
        num_objs: 0,
        num_vertices: 0,
        num_faces: 0,
        num_normals: 0,
        num_spheres: 0,
    };

    // OBJ header
    writeln!(w, "# WaveFront *.obj file (generated from an IMOD model by imod2obj)").ok();
    if let Some(ref mtl_name) = args.mtl {
        writeln!(w, "# Material values are stored in this .mtl file:").ok();
        writeln!(w, "mtllib {}", mtl_name).ok();
    }
    writeln!(w, "\n").ok();

    // Write each object
    for (ob, obj) in model.objects.iter().enumerate() {
        if !args.all_objects && is_off(obj.flags) {
            continue;
        }

        let obj_scattered = is_scat(obj.flags);
        let obj_has_spheres = has_spheres(obj);

        let name = safe_obj_name(obj, ob);

        writeln!(w).ok();
        writeln!(w, "g {}", name).ok();
        if has_mtl {
            writeln!(w, "usemtl {}", name).ok();
        }
        writeln!(w).ok();

        // Print mesh if not scattered and has meshes
        if !obj_scattered && !obj.meshes.is_empty() {
            for mesh in &obj.meshes {
                write_mesh(&mut w, obj, mesh, &args, &mut stats, zscale);
            }
        }

        // Print spheres if applicable
        if obj_has_spheres {
            if obj_scattered || !args.only_scat_spheres {
                write_scat_contours(&mut w, obj, ob, &args, &mut stats, zscale);
            }
        }

        stats.num_objs += 1;
        writeln!(w, "\n").ok();
    }

    writeln!(w, "\n").ok();
    writeln!(w, "# For more info on OBJ file format see:").ok();
    writeln!(w, "#  http://www.andrewnoske.com/wiki/index.php?title=OBJ_file_format").ok();

    drop(w);

    // Write MTL file if requested
    if let Some(ref mtl_path) = args.mtl {
        let mtl_file = File::create(mtl_path).unwrap_or_else(|e| {
            eprintln!("ERROR: imod2obj - Could not open MTL file {}: {}", mtl_path, e);
            process::exit(10);
        });
        let mut mw = BufWriter::new(mtl_file);
        write_mtl(&mut mw, &model);
    }

    eprintln!("Finished writing '{}'", args.input);
    eprintln!("  # objects on: {}", stats.num_objs);
    eprintln!("  # spheres:    {}", stats.num_spheres);
    eprintln!("  # vertices:   {}", stats.num_vertices);
    eprintln!("  # faces:      {}", stats.num_faces);
    eprintln!("  mtl file generated: {}", if has_mtl { "yes" } else { "no" });
}
