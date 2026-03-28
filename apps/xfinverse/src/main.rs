//! xfinverse - Invert transforms in an .xf file.
//!
//! Reads a file of 2D linear transforms and writes the inverse of each one.
//! Optionally sets the translation (shift) components to zero.

use clap::Parser;
use imod_transforms::{read_xf_file, write_xf_file};
use std::process;

#[derive(Parser)]
#[command(name = "xfinverse", about = "Invert transforms in an .xf file")]
struct Args {
    /// Input transform file
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output file for inverse transforms
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Set shifts (translations) to zero in the output
    #[arg(short = 'z', long = "zero-shifts")]
    zero_shifts: bool,
}

fn main() {
    let args = Args::parse();

    let transforms = read_xf_file(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: XFINVERSE - reading transform file {}: {}", args.input, e);
        process::exit(1);
    });

    if transforms.is_empty() {
        eprintln!("ERROR: XFINVERSE - no transforms in input file");
        process::exit(1);
    }

    let inverted: Vec<_> = transforms
        .iter()
        .map(|xf| {
            let mut inv = xf.inverse();
            if args.zero_shifts {
                inv.dx = 0.0;
                inv.dy = 0.0;
            }
            inv
        })
        .collect();

    write_xf_file(&args.output, &inverted).unwrap_or_else(|e| {
        eprintln!("ERROR: XFINVERSE - writing output file {}: {}", args.output, e);
        process::exit(1);
    });

    eprintln!("{} inverse transforms written", inverted.len());
}
