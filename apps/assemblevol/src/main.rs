use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Assemble a single MRC volume from an array of sub-volume files arranged
/// in X, Y, and Z.  Can reassemble a tomogram chopped into pieces by Tomopieces.
#[derive(Parser)]
#[command(name = "assemblevol", about = "Assemble volume from pieces")]
struct Args {
    /// Input MRC files (in order: X varies fastest, then Y, then Z)
    #[arg(short = 'i', long, num_args = 1..)]
    input: Vec<String>,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Number of files in X
    #[arg(long)]
    nxfiles: usize,

    /// Number of files in Y
    #[arg(long)]
    nyfiles: usize,

    /// Number of files in Z
    #[arg(long)]
    nzfiles: usize,

    /// Start,end pixel coordinates to extract in X for each X position.
    /// Pairs of values: start1,end1,start2,end2,...  (0,0 = full extent)
    #[arg(long, num_args = 0..)]
    xextract: Option<Vec<i32>>,

    /// Start,end pixel coordinates to extract in Y for each Y position.
    #[arg(long, num_args = 0..)]
    yextract: Option<Vec<i32>>,

    /// Start,end pixel coordinates to extract in Z for each Z position.
    #[arg(long, num_args = 0..)]
    zextract: Option<Vec<i32>>,
}

/// Parse extraction ranges from flat pairs into (low, high) vectors.
/// Returns None (meaning "use full extent") for each position where both are 0.
fn parse_ranges(vals: &Option<Vec<i32>>, count: usize) -> Vec<Option<(usize, usize)>> {
    match vals {
        None => vec![None; count],
        Some(v) => {
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let lo = if 2 * i < v.len() { v[2 * i] } else { 0 };
                let hi = if 2 * i + 1 < v.len() { v[2 * i + 1] } else { 0 };
                if lo == 0 && hi == 0 {
                    result.push(None); // full extent - determined from file
                } else if lo == -1 && hi == -1 {
                    // Special: extract only first pixel
                    result.push(Some((0, 0)));
                } else {
                    result.push(Some((lo as usize, hi as usize)));
                }
            }
            result
        }
    }
}

fn main() {
    let args = Args::parse();
    let expected = args.nxfiles * args.nyfiles * args.nzfiles;

    if args.input.len() != expected {
        eprintln!(
            "Error: expected {} input files ({}x{}x{}), got {}",
            expected, args.nxfiles, args.nyfiles, args.nzfiles, args.input.len()
        );
        std::process::exit(1);
    }

    let mut x_ranges = parse_ranges(&args.xextract, args.nxfiles);
    let mut y_ranges = parse_ranges(&args.yextract, args.nyfiles);
    let mut z_ranges = parse_ranges(&args.zextract, args.nzfiles);

    // First pass: read headers to determine dimensions and fill in unspecified ranges
    let mut mode_first: Option<MrcMode> = None;
    let mut first_header: Option<MrcHeader> = None;

    let mut file_idx = 0;
    for iz in 0..args.nzfiles {
        for iy in 0..args.nyfiles {
            for ix in 0..args.nxfiles {
                let reader = MrcReader::open(&args.input[file_idx]).unwrap_or_else(|e| {
                    eprintln!("Error opening {}: {}", args.input[file_idx], e);
                    std::process::exit(1);
                });
                let h = reader.header();
                let fnx = h.nx as usize;
                let fny = h.ny as usize;
                let fnz = h.nz as usize;
                let fmode = h.data_mode().unwrap_or(MrcMode::Float);

                if first_header.is_none() {
                    first_header = Some(h.clone());
                    mode_first = Some(fmode);
                } else if Some(fmode) != mode_first {
                    eprintln!("Error: mode mismatch for file {}", args.input[file_idx]);
                    std::process::exit(1);
                }

                // Fill in unspecified ranges from file dimensions
                if x_ranges[ix].is_none() {
                    x_ranges[ix] = Some((0, fnx - 1));
                }
                if y_ranges[iy].is_none() {
                    y_ranges[iy] = Some((0, fny - 1));
                }
                if z_ranges[iz].is_none() {
                    z_ranges[iz] = Some((0, fnz - 1));
                }

                // Validate
                let (xlo, xhi) = x_ranges[ix].unwrap();
                let (ylo, yhi) = y_ranges[iy].unwrap();
                let (zlo, zhi) = z_ranges[iz].unwrap();
                if xhi >= fnx || yhi >= fny || zhi >= fnz {
                    eprintln!(
                        "Error: extraction range exceeds dimensions for file {}",
                        args.input[file_idx]
                    );
                    std::process::exit(1);
                }
                let _ = (xlo, ylo, zlo); // suppress warnings

                file_idx += 1;
            }
        }
    }

    // Compute output dimensions
    let x_sizes: Vec<usize> = x_ranges.iter().map(|r| {
        let (lo, hi) = r.unwrap();
        hi + 1 - lo
    }).collect();
    let y_sizes: Vec<usize> = y_ranges.iter().map(|r| {
        let (lo, hi) = r.unwrap();
        hi + 1 - lo
    }).collect();
    let z_sizes: Vec<usize> = z_ranges.iter().map(|r| {
        let (lo, hi) = r.unwrap();
        hi + 1 - lo
    }).collect();

    let nx_out: usize = x_sizes.iter().sum();
    let ny_out: usize = y_sizes.iter().sum();
    let nz_out: usize = z_sizes.iter().sum();

    let fh = first_header.unwrap();
    let mode = mode_first.unwrap();

    let px = fh.pixel_size_x();
    let py = fh.pixel_size_y();
    let pz = fh.pixel_size_z();

    let mut out_header = MrcHeader::new(nx_out as i32, ny_out as i32, nz_out as i32, mode);
    out_header.xlen = nx_out as f32 * px;
    out_header.ylen = ny_out as f32 * py;
    out_header.zlen = nz_out as f32 * pz;
    out_header.mx = nx_out as i32;
    out_header.my = ny_out as i32;
    out_header.mz = nz_out as i32;

    // Adjust origin: shift by lower-left extraction start
    let (xlo0, _) = x_ranges[0].unwrap();
    let (ylo0, _) = y_ranges[0].unwrap();
    let (zlo0, _) = z_ranges[0].unwrap();
    out_header.xorg = fh.xorg - xlo0 as f32 * px;
    out_header.yorg = fh.yorg - ylo0 as f32 * py;
    out_header.zorg = fh.zorg - zlo0 as f32 * pz;

    // Copy tilt angles from first file
    out_header.tilt_angles = fh.tilt_angles;

    out_header.add_label("assemblevol: Reassemble a volume from pieces");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating output: {}", e);
        std::process::exit(1);
    });

    // Output buffer for a composed slice
    let mut out_slice = vec![0.0f32; nx_out * ny_out];
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum: f64 = 0.0;

    // Second pass: read data and assemble
    for indz in 0..args.nzfiles {
        let (zlo, zhi) = z_ranges[indz].unwrap();

        for iz in zlo..=zhi {
            // Clear output slice
            out_slice.fill(0.0);

            let mut y_offset = 0usize;
            for indy in 0..args.nyfiles {
                let (ylo, yhi) = y_ranges[indy].unwrap();
                let ny_box = yhi + 1 - ylo;

                let mut x_offset = 0usize;
                for indx in 0..args.nxfiles {
                    let (xlo, xhi) = x_ranges[indx].unwrap();
                    let nx_box = xhi + 1 - xlo;

                    // File index in the flat array
                    let fidx = indx + args.nxfiles * (indy + args.nyfiles * indz);
                    let mut reader = MrcReader::open(&args.input[fidx]).unwrap_or_else(|e| {
                        eprintln!("Error opening {}: {}", args.input[fidx], e);
                        std::process::exit(1);
                    });
                    let fnx = reader.header().nx as usize;

                    let slice = reader.read_slice_f32(iz).unwrap_or_else(|e| {
                        eprintln!("Error reading slice {} from {}: {}", iz, args.input[fidx], e);
                        std::process::exit(1);
                    });

                    // Insert sub-region into output slice
                    for iy_box in 0..ny_box {
                        let src_row = (ylo + iy_box) * fnx;
                        let dst_row = (y_offset + iy_box) * nx_out;
                        for ix_box in 0..nx_box {
                            out_slice[dst_row + x_offset + ix_box] =
                                slice[src_row + xlo + ix_box];
                        }
                    }

                    x_offset += nx_box;
                }
                y_offset += ny_box;
            }

            let (smin, smax, smean) = min_max_mean(&out_slice);
            global_min = global_min.min(smin);
            global_max = global_max.max(smax);
            global_sum += smean as f64;

            writer.write_slice_f32(&out_slice).unwrap_or_else(|e| {
                eprintln!("Error writing output slice: {}", e);
                std::process::exit(1);
            });
        }
    }

    let global_mean = (global_sum / nz_out as f64) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap_or_else(|e| {
        eprintln!("Error finishing output: {}", e);
        std::process::exit(1);
    });

    eprintln!("{} files reassembled", args.input.len());
}
