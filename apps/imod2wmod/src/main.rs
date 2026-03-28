use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_model::read_model;

/// Convert an IMOD model to WIMP (VMS) format.
///
/// Reads an IMOD binary model file and writes it in the WIMP text format
/// used on VMS systems. Each contour is output with its object number,
/// point count, display switch, and coordinates.
#[derive(Parser)]
#[command(name = "imod2wmod", version, about)]
struct Args {
    /// Input IMOD model file
    input: String,

    /// Output WIMP model file
    output: String,
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2wmod - Reading model {}: {}", args.input, e);
        process::exit(3);
    });

    let out_file = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2wmod - Could not open {}: {}", args.output, e);
        process::exit(10);
    });
    let mut w = BufWriter::new(out_file);

    // Map each IMOD object to a WIMP display color index.
    // Use indices 247-255 for the first 9 objects, then wrap.
    for (ob, obj) in model.objects.iter().enumerate() {
        let display_switch = if ob < 9 { 247 + ob } else { 247 + (ob % 9) };

        for (co, cont) in obj.contours.iter().enumerate() {
            if cont.points.is_empty() {
                continue;
            }

            // WIMP header for each contour
            writeln!(w, "  Object #: {}", ob + 1).ok();
            writeln!(w, "  Number of points = {}", cont.points.len()).ok();
            writeln!(w, "  Display = {}", display_switch).ok();
            writeln!(w, "  Contour {} of object {}", co + 1, ob + 1).ok();

            for (pt_i, pt) in cont.points.iter().enumerate() {
                writeln!(w, "  {} {} {} {}", pt_i + 1, pt.x, pt.y, pt.z).ok();
            }
        }
    }
}
