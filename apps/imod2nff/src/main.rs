use clap::Parser;
use imod_model::{read_model, ImodModel};
use std::fs::File;
use std::io::{self, BufWriter, Write};

// IMOD object flag bits
const IMOD_OBJFLAG_SCAT: u32 = 1 << 9;
const IMOD_OBJFLAG_MESH: u32 = 1 << 10;
const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;

// Mesh index list sentinel values
const IMOD_MESH_ENDPOLY: i32 = -23;
const IMOD_MESH_BGNPOLYNORM: i32 = -21;
const IMOD_MESH_BGNPOLYNORM2: i32 = -24;
const IMOD_MESH_BGNPOLY: i32 = -20;
const IMOD_MESH_BGNBIGPOLY: i32 = -25;

/// Convert IMOD model files to the NFF (Neutral File Format) file format.
#[derive(Parser)]
#[command(name = "imod2nff", about = "Convert IMOD model to NFF format")]
struct Args {
    /// Force output of mesh data
    #[arg(short = 'm')]
    force_mesh: bool,

    /// Input IMOD model file
    input: String,

    /// Output NFF file
    output: String,
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("Imod2NFF: Error reading model {}: {}", args.input, e);
        std::process::exit(3);
    });

    let fout = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("Couldn't open output file {}: {}", args.output, e);
        std::process::exit(10);
    });
    let mut w = BufWriter::new(fout);

    write_nff(&model, &mut w, &args).unwrap_or_else(|e| {
        eprintln!("Error writing NFF: {}", e);
        std::process::exit(1);
    });
}

fn write_nff(model: &ImodModel, w: &mut impl Write, args: &Args) -> io::Result<()> {
    let xs = model.scale.x;
    let ys = model.scale.y;
    let zs = model.scale.z;

    for (objnum, obj) in model.objects.iter().enumerate() {
        writeln!(w, "# object {} with {} contours.", objnum, obj.contours.len())?;
        writeln!(w, "f {} {} {} 0 0 0 0 0", obj.red, obj.green, obj.blue)?;

        let use_mesh = args.force_mesh && !obj.meshes.is_empty()
            || (obj.flags & IMOD_OBJFLAG_MESH) != 0;

        if use_mesh {
            for mesh in &obj.meshes {
                let indices = &mesh.indices;
                let mut i = 0;
                while i < indices.len() {
                    let cmd = indices[i];
                    let (list_inc, vert_base, norm_add) = mesh_poly_norm_factors(cmd);
                    if list_inc > 0 {
                        i += 1;
                        while i < indices.len() && indices[i] != IMOD_MESH_ENDPOLY {
                            writeln!(w, "pp 3")?;
                            for _ in 0..3 {
                                if i + vert_base >= indices.len() {
                                    break;
                                }
                                let vi = indices[i + vert_base] as usize;
                                let ni = (indices[i] as usize) + norm_add;
                                if vi < mesh.vertices.len() && ni < mesh.vertices.len() {
                                    let v = &mesh.vertices[vi];
                                    let n = &mesh.vertices[ni];
                                    writeln!(
                                        w,
                                        "{} {} {} {} {} {}",
                                        v.x * xs,
                                        v.y * ys,
                                        v.z * zs,
                                        n.x * xs,
                                        n.y * ys,
                                        n.z * zs
                                    )?;
                                }
                                i += list_inc;
                            }
                        }
                    } else if cmd == IMOD_MESH_BGNPOLY || cmd == IMOD_MESH_BGNBIGPOLY {
                        i += 1;
                        while i < indices.len() && indices[i] != IMOD_MESH_ENDPOLY {
                            i += 1;
                        }
                    }
                    i += 1;
                }
            }
        } else {
            // Contour-based output
            for cont in &obj.contours {
                if cont.points.is_empty() {
                    continue;
                }

                if (obj.flags & IMOD_OBJFLAG_SCAT) != 0 {
                    let r = obj.pdrawsize as f32 * 0.5;
                    for pt in &cont.points {
                        writeln!(
                            w,
                            "s {} {} {} {}",
                            (pt.x + model.offset.x) * xs,
                            (pt.y + model.offset.y) * ys,
                            (pt.z + model.offset.z) * zs,
                            r
                        )?;
                    }
                } else if (obj.flags & IMOD_OBJFLAG_OPEN) == 0 {
                    // Closed contour
                    writeln!(w, "p {}", cont.points.len())?;
                    for pt in &cont.points {
                        writeln!(
                            w,
                            "{} {} {}",
                            (pt.x + model.offset.x) * xs,
                            (pt.y + model.offset.y) * ys,
                            (pt.z + model.offset.z) * zs
                        )?;
                    }
                } else {
                    // Open contour - needs special handling for degenerate cases
                    let n = cont.points.len();
                    match n {
                        1 => {
                            writeln!(w, "p 3")?;
                            let pt = &cont.points[0];
                            for _ in 0..3 {
                                writeln!(
                                    w,
                                    "{} {} {}",
                                    (pt.x + model.offset.x) * xs,
                                    (pt.y + model.offset.y) * ys,
                                    (pt.z + model.offset.z) * zs
                                )?;
                            }
                        }
                        2 => {
                            writeln!(w, "p 3")?;
                            for idx in [0, 1, 0] {
                                let pt = &cont.points[idx];
                                writeln!(
                                    w,
                                    "{} {} {}",
                                    (pt.x + model.offset.x) * xs,
                                    (pt.y + model.offset.y) * ys,
                                    (pt.z + model.offset.z) * zs
                                )?;
                            }
                        }
                        _ => {
                            writeln!(w, "p {}", (n * 2) - 2)?;
                            for pt in &cont.points {
                                writeln!(
                                    w,
                                    "{} {} {}",
                                    (pt.x + model.offset.x) * xs,
                                    (pt.y + model.offset.y) * ys,
                                    (pt.z + model.offset.z) * zs
                                )?;
                            }
                            for i in (1..n - 1).rev() {
                                let pt = &cont.points[i];
                                writeln!(
                                    w,
                                    "{} {} {}",
                                    (pt.x + model.offset.x) * xs,
                                    (pt.y + model.offset.y) * ys,
                                    (pt.z + model.offset.z) * zs
                                )?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Returns (list_inc, vert_base, norm_add) for mesh polygon norm commands.
fn mesh_poly_norm_factors(cmd: i32) -> (usize, usize, usize) {
    match cmd {
        IMOD_MESH_BGNPOLYNORM => (2, 0, 1),
        IMOD_MESH_BGNPOLYNORM2 => (1, 0, 0),
        _ => (0, 0, 0),
    }
}
