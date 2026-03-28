use std::process;

use clap::Parser;
use imod_mrc::{MrcReader, MrcWriter};

/// Add (append) sections from one or more files to an existing image stack.
///
/// The X and Y dimensions of all files must match. All sections from each
/// input file are appended. The output header is updated to reflect the
/// new number of sections.
#[derive(Parser)]
#[command(name = "addtostack", version, about)]
struct Args {
    /// Existing MRC stack to append to (will be modified in-place unless --copy).
    stack: String,

    /// Files to append (all sections from each).
    #[arg(required = true)]
    inputs: Vec<String>,

    /// Create a copy of the stack instead of appending in-place.
    #[arg(short = 'c', long = "copy")]
    copy: bool,

    /// Output file name (required if --copy is used).
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    let out_path = if args.copy {
        args.output.as_deref().unwrap_or_else(|| {
            eprintln!("ERROR: addtostack - --output required when using --copy");
            process::exit(1);
        })
    } else {
        &args.stack
    };

    // Read the base stack
    let mut base_reader = MrcReader::open(&args.stack).unwrap_or_else(|e| {
        eprintln!("ERROR: addtostack - opening {}: {}", args.stack, e);
        process::exit(1);
    });

    let base_h = base_reader.header().clone();
    let nx = base_h.nx as usize;
    let ny = base_h.ny as usize;
    let base_nz = base_h.nz as usize;

    // Read all base slices
    let mut all_slices: Vec<Vec<f32>> = Vec::with_capacity(base_nz + 100);
    for z in 0..base_nz {
        all_slices.push(base_reader.read_slice_f32(z).unwrap_or_else(|e| {
            eprintln!("ERROR: addtostack - reading base slice {}: {}", z, e);
            process::exit(1);
        }));
    }

    let mut dmin = base_h.amin;
    let mut dmax = base_h.amax;
    let mut dsum = base_h.amean as f64 * base_nz as f64;

    // Append sections from each input file
    let mut n_appended = 0usize;
    for infile in &args.inputs {
        let mut reader = MrcReader::open(infile).unwrap_or_else(|e| {
            eprintln!("ERROR: addtostack - opening {}: {}", infile, e);
            process::exit(1);
        });

        let ih = reader.header().clone();
        if ih.nx as usize != nx || ih.ny as usize != ny {
            eprintln!(
                "ERROR: addtostack - size mismatch: {} is {}x{} but base is {}x{}",
                infile, ih.nx, ih.ny, nx, ny
            );
            process::exit(1);
        }

        let in_nz = ih.nz as usize;
        for z in 0..in_nz {
            let slice = reader.read_slice_f32(z).unwrap_or_else(|e| {
                eprintln!("ERROR: addtostack - reading {} slice {}: {}", infile, z, e);
                process::exit(1);
            });

            // Compute statistics for this slice
            let mut smin = f32::MAX;
            let mut smax = f32::MIN;
            let mut smean = 0.0_f64;
            for &v in &slice {
                smin = smin.min(v);
                smax = smax.max(v);
                smean += v as f64;
            }
            smean /= (nx * ny) as f64;
            dmin = dmin.min(smin);
            dmax = dmax.max(smax);
            dsum += smean;

            all_slices.push(slice);
            n_appended += 1;
        }
    }

    let total_nz = all_slices.len();
    let dmean = (dsum / total_nz as f64) as f32;

    // Write output
    let mut out_header = base_h.clone();
    out_header.nz = total_nz as i32;
    out_header.mz = total_nz as i32;
    out_header.zlen = out_header.zlen * total_nz as f32 / base_nz as f32;
    out_header.amin = dmin;
    out_header.amax = dmax;
    out_header.amean = dmean;
    out_header.add_label(&format!(
        "ADDTOSTACK: {} sections from {} files appended",
        n_appended,
        args.inputs.len()
    ));

    let mut writer = MrcWriter::create(out_path, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: addtostack - creating output: {}", e);
        process::exit(1);
    });

    for slice in &all_slices {
        writer.write_slice_f32(slice).unwrap_or_else(|e| {
            eprintln!("ERROR: addtostack - writing slice: {}", e);
            process::exit(1);
        });
    }

    writer.finish(dmin, dmax, dmean).unwrap();

    println!(
        "ADDTOSTACK: {} sections from {} files appended, total {} sections",
        n_appended,
        args.inputs.len(),
        total_nz
    );
}
