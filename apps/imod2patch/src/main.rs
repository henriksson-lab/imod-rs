use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_model::read_model;

/// Convert an IMOD model (patch vectors) to a patch text file.
///
/// Reads an IMOD model where each contour has exactly 2 points representing
/// a position and displacement vector, and writes a text patch file with
/// integer position and floating-point displacement columns.
#[derive(Parser)]
#[command(name = "imod2patch", version, about)]
struct Args {
    /// Input IMOD model file (containing patch vectors)
    input: String,

    /// Output patch text file
    output: String,
}

/// IMOD model flag: Y and Z are flipped
const IMODF_FLIPYZ: u32 = 1 << 14;

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2patch - Reading model {}: {}", args.input, e);
        process::exit(1);
    });

    let out_file = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2patch - Could not open {}: {}", args.output, e);
        process::exit(1);
    });
    let mut w = BufWriter::new(out_file);

    // Count patches (contours with at least 2 points)
    let mut npatch = 0usize;
    for obj in &model.objects {
        for cont in &obj.contours {
            if cont.points.len() >= 2 {
                npatch += 1;
            }
        }
    }

    // Write header
    writeln!(w, "{}   edited positions", npatch).unwrap();

    let flip = (model.flags & IMODF_FLIPYZ) != 0;

    for obj in &model.objects {
        for cont in &obj.contours {
            if cont.points.len() < 2 {
                continue;
            }
            let pts = &cont.points;
            let ix = (pts[0].x + 0.5) as i32;
            let iy = (pts[0].y + 0.5) as i32;
            let iz = (pts[0].z + 0.5) as i32;
            let pix = model.pixel_size;
            let dx = (pts[1].x - pts[0].x) / pix;
            let dy = (pts[1].y - pts[0].y) / pix;
            let dz = (pts[1].z - pts[0].z) / pix;

            if flip {
                write!(w, "{:6} {:5} {:5} {:8.2} {:8.2} {:8.2}",
                       ix, iz, iy, dx, dz, dy).unwrap();
            } else {
                write!(w, "{:6} {:5} {:5} {:8.2} {:8.2} {:8.2}",
                       ix, iy, iz, dx, dy, dz).unwrap();
            }
            writeln!(w).unwrap();
        }
    }
}
