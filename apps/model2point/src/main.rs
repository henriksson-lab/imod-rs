use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_model::read_model;

/// Convert an IMOD model file to a text file of point coordinates.
///
/// Each point is written on a separate line. Coordinates can be output as
/// floating-point or integer values. Optional object/contour numbers can
/// be prepended to each line.
#[derive(Parser)]
#[command(name = "model2point", version, about)]
struct Args {
    /// Input IMOD model file.
    input: String,

    /// Output text file for point list.
    output: String,

    /// Output floating-point coordinates (default).
    #[arg(short = 'f', long = "float")]
    floating: bool,

    /// Output integer (nearest-integer) coordinates.
    #[arg(short = 'i', long = "integer")]
    integer: bool,

    /// Print object and contour numbers before each coordinate.
    #[arg(short = 'o', long = "object")]
    print_object: bool,

    /// Print contour number before each coordinate.
    #[arg(short = 'c', long = "contour")]
    print_contour: bool,

    /// Number objects/contours starting from zero instead of one.
    #[arg(short = 'z', long = "zero")]
    numbered_from_zero: bool,

    /// Output Z coordinates starting from zero (subtract 0.5).
    #[arg(long = "zfromzero")]
    z_from_zero: bool,
}

fn main() {
    let args = Args::parse();

    if args.floating && args.integer {
        eprintln!("ERROR: model2point - cannot specify both --float and --integer");
        process::exit(1);
    }
    if args.z_from_zero && args.integer {
        eprintln!("ERROR: model2point - Z-from-zero requires floating-point output");
        process::exit(1);
    }

    let use_float = !args.integer; // default is float
    let z_offset: f32 = if args.z_from_zero { -0.5 } else { 0.0 };
    let num_offset: usize = if args.numbered_from_zero { 0 } else { 1 };

    let model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: model2point - reading model file: {}", e);
            process::exit(1);
        }
    };

    let outfile = match File::create(&args.output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: model2point - opening output file: {}", e);
            process::exit(1);
        }
    };
    let mut out = BufWriter::new(outfile);

    let mut npnts: usize = 0;

    for (obj_idx, obj) in model.objects.iter().enumerate() {
        for (cont_idx, cont) in obj.contours.iter().enumerate() {
            for pt in &cont.points {
                let x = pt.x;
                let y = pt.y;
                let z = pt.z + z_offset;

                if args.print_object {
                    write!(out, "{:6}", obj_idx + num_offset).unwrap();
                }
                if args.print_contour || args.print_object {
                    write!(out, "{:6}", cont_idx + num_offset).unwrap();
                }

                if use_float {
                    writeln!(out, "{:12.2}{:12.2}{:12.2}", x, y, z).unwrap();
                } else {
                    writeln!(
                        out,
                        "{:7}{:7}{:7}",
                        x.round() as i32,
                        y.round() as i32,
                        z.round() as i32,
                    )
                    .unwrap();
                }

                npnts += 1;
            }
        }
    }

    println!("{} points output to file", npnts);
}
