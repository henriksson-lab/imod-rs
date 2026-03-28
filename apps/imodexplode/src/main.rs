use std::process;

use clap::Parser;
use imod_math::parse_list;
use imod_model::{read_model, write_model};

/// Shift objects in an IMOD model by specified offsets ("explode" the model).
///
/// Each set of objects specified by -o can be shifted by independent dx, dy, dz
/// offsets. This moves both contour point data and mesh vertex data for the
/// specified objects. Multiple object sets with different offsets can be entered.
#[derive(Parser)]
#[command(name = "imodexplode", version, about)]
struct Args {
    /// Object set and offset specifications.
    ///
    /// Provide one or more groups of: -o LIST -x DX -y DY -z DZ
    /// where LIST is a comma-separated list of object numbers (1-based),
    /// and at least one offset must be non-zero. Only non-zero offsets need
    /// to be entered.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    raw_args: Vec<String>,
}

struct ShiftSet {
    objects: Vec<i32>,
    dx: f32,
    dy: f32,
    dz: f32,
}

fn parse_shift_sets(raw: &[String]) -> (Vec<ShiftSet>, String, String) {
    let mut sets = Vec::new();
    let mut positional = Vec::new();
    let mut i = 0;

    while i < raw.len() {
        let arg = &raw[i];
        if arg == "-o" {
            i += 1;
            if i >= raw.len() {
                eprintln!("ERROR: imodexplode - -o requires a list argument");
                process::exit(1);
            }
            let list = parse_list(&raw[i]).unwrap_or_else(|e| {
                eprintln!("ERROR: imodexplode - Error parsing object list: {}", e);
                process::exit(1);
            });
            sets.push(ShiftSet {
                objects: list,
                dx: 0.0,
                dy: 0.0,
                dz: 0.0,
            });
        } else if arg == "-x" {
            i += 1;
            if sets.is_empty() {
                eprintln!("ERROR: imodexplode - Must define object list before offsets");
                process::exit(1);
            }
            sets.last_mut().unwrap().dx = raw[i].parse().unwrap_or(0.0);
        } else if arg == "-y" {
            i += 1;
            if sets.is_empty() {
                eprintln!("ERROR: imodexplode - Must define object list before offsets");
                process::exit(1);
            }
            sets.last_mut().unwrap().dy = raw[i].parse().unwrap_or(0.0);
        } else if arg == "-z" {
            i += 1;
            if sets.is_empty() {
                eprintln!("ERROR: imodexplode - Must define object list before offsets");
                process::exit(1);
            }
            sets.last_mut().unwrap().dz = raw[i].parse().unwrap_or(0.0);
        } else {
            positional.push(raw[i].clone());
        }
        i += 1;
    }

    if positional.len() != 2 {
        eprintln!(
            "Usage: imodexplode -o list -x dx -y dy -z dz ... infile outfile\n\
             Shifts the objects in each list by the specified offsets."
        );
        process::exit(1);
    }
    (sets, positional[0].clone(), positional[1].clone())
}

fn main() {
    let args = Args::parse();
    let (sets, input, output) = parse_shift_sets(&args.raw_args);

    let mut model = read_model(&input).unwrap_or_else(|e| {
        eprintln!("ERROR: imodexplode - Error reading model {}: {}", input, e);
        process::exit(1);
    });

    for set in &sets {
        for &ob_num in &set.objects {
            let ob = (ob_num - 1) as usize;
            if ob >= model.objects.len() {
                eprintln!("WARNING: imodexplode - No object # {}", ob_num);
                continue;
            }

            // Shift contour data
            for cont in &mut model.objects[ob].contours {
                for pt in &mut cont.points {
                    pt.x += set.dx;
                    pt.y += set.dy;
                    pt.z += set.dz;
                }
            }

            // Shift mesh data (every other vertex = position, alternating with normal)
            for mesh in &mut model.objects[ob].meshes {
                for (i, vert) in mesh.vertices.iter_mut().enumerate() {
                    if i % 2 == 0 {
                        vert.x += set.dx;
                        vert.y += set.dy;
                        vert.z += set.dz;
                    }
                }
            }
        }
    }

    write_model(&output, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: imodexplode - Writing model: {}", e);
        process::exit(1);
    });
}
