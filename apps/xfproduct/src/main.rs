//! xfproduct - Multiply/compose transforms from two .xf files.
//!
//! Concatenates two lists of transforms by multiplying corresponding pairs.
//! If one file has a single transform, it is applied to every transform in the
//! other file.  An optional --one flag restricts the single transform to be
//! applied only to one specific index (0-based), copying the rest unchanged.
//! Translation components can be independently scaled with --scale.

use clap::Parser;
use imod_transforms::{read_xf_file, write_xf_file};
use std::process;

#[derive(Parser)]
#[command(name = "xfproduct", about = "Multiply transforms from two .xf files")]
struct Args {
    /// First input transform file (applied first)
    #[arg(short = '1', long = "input1")]
    input1: String,

    /// Second input transform file (applied second)
    #[arg(short = '2', long = "input2")]
    input2: String,

    /// Output file for product transforms
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Scale translations: two comma-separated factors for file1,file2
    #[arg(short = 's', long = "scale", value_delimiter = ',', num_args = 2)]
    scale: Option<Vec<f32>>,

    /// When one file has a single transform, apply it only to this
    /// 0-based index and copy the rest unchanged.
    #[arg(long = "one")]
    one: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let xf1 = read_xf_file(&args.input1).unwrap_or_else(|e| {
        eprintln!("ERROR: XFPRODUCT - reading first file {}: {}", args.input1, e);
        process::exit(1);
    });
    let xf2 = read_xf_file(&args.input2).unwrap_or_else(|e| {
        eprintln!("ERROR: XFPRODUCT - reading second file {}: {}", args.input2, e);
        process::exit(1);
    });

    let n1 = xf1.len();
    let n2 = xf2.len();

    if n1 == 0 {
        eprintln!("ERROR: XFPRODUCT - no transforms in first input file");
        process::exit(1);
    }
    if n2 == 0 {
        eprintln!("ERROR: XFPRODUCT - no transforms in second input file");
        process::exit(1);
    }

    eprintln!("{} transforms in first file", n1);
    eprintln!("{} transforms in second file", n2);

    let (scale1, scale2) = match &args.scale {
        Some(v) => (v[0], v[1]),
        None => (1.0, 1.0),
    };

    // Determine output count and how to index
    let nout = if n1 == n2 {
        n1
    } else if n2 == 1 {
        eprintln!("Single second transform applied to all first transforms");
        n1
    } else if n1 == 1 {
        eprintln!("Single first transform applied to all second transforms");
        n2
    } else {
        eprintln!("WARNING: XFPRODUCT - number of transforms does not match");
        n1.min(n2)
    };

    let nsingle = args.one;

    let mut results = Vec::with_capacity(nout);

    for iz in 0..nout {
        let idx1 = if n1 == 1 { 0 } else { iz };
        let idx2 = if n2 == 1 { 0 } else { iz };

        // Scale the translations
        let mut f1 = xf1[idx1];
        f1.dx *= scale1;
        f1.dy *= scale1;
        let mut f2 = xf2[idx2];
        f2.dx *= scale2;
        f2.dy *= scale2;

        // Determine whether to copy or multiply
        let should_copy_1 = n2 == 1 && n1 != 1 && nsingle.is_some_and(|s| s < n1 && s != iz);
        let should_copy_2 = n1 == 1 && n2 != 1 && nsingle.is_some_and(|s| s < n2 && s != iz);

        let product = if should_copy_1 {
            f1
        } else if should_copy_2 {
            f2
        } else {
            // xfmult: first applied first, second applied second
            // product = second * first  (in matrix terms)
            f1.then(&f2)
        };

        results.push(product);
    }

    write_xf_file(&args.output, &results).unwrap_or_else(|e| {
        eprintln!("ERROR: XFPRODUCT - writing output file {}: {}", args.output, e);
        process::exit(1);
    });

    eprintln!("{} new transforms written", nout);
}
