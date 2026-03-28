//! extractmagrad - Extract magnification gradients from MRC header.
//!
//! Reads intensity values and tilt angles from an MRC image file header,
//! reads a table of magnification gradients as a function of intensity,
//! and outputs a file with tilt angles and interpolated mag gradients.
//!
//! Translated from IMOD's extractmagrad.f

use clap::Parser;
use imod_mrc::MrcReader;
use std::io::{BufRead, BufReader, Write};
use std::process;

#[derive(Parser)]
#[command(
    name = "extractmagrad",
    about = "Extract magnification gradient from header and gradient table"
)]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Output file for tilt/gradient list
    #[arg(short = 'o', long)]
    output: String,

    /// Gradient table file
    #[arg(short = 'g', long)]
    gradient: String,

    /// Rotation angle of tilt axis
    #[arg(short = 'r', long)]
    rotation: f32,

    /// Pixel size in nm (default: from header, in Angstroms / 10)
    #[arg(short = 'p', long)]
    pixel: Option<f32>,

    /// Delta gradient to add to all gradients
    #[arg(long, default_value_t = 0.0)]
    dgrad: f32,

    /// Delta rotation to add to all rotations
    #[arg(long, default_value_t = 0.0)]
    drot: f32,
}

struct GradTable {
    c2: Vec<f32>,
    grad: Vec<f32>,
    rot: Vec<f32>,
    lin: Vec<f32>,
    version: i32,
    crossover: f32,
}

fn read_gradient_table(path: &str) -> GradTable {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: extractmagrad - opening gradient table: {}", e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines
        .next()
        .unwrap_or_else(|| {
            eprintln!("ERROR: extractmagrad - empty gradient table");
            process::exit(1);
        })
        .unwrap();

    let first_vals: Vec<f32> = first_line
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    let (version, crossover, start_lines) = if first_vals.len() == 1 {
        let ver = first_vals[0] as i32;
        let cross_line = lines
            .next()
            .unwrap_or_else(|| {
                eprintln!("ERROR: extractmagrad - missing crossover line");
                process::exit(1);
            })
            .unwrap();
        let crossover: f32 = cross_line.trim().parse().unwrap_or_else(|_| {
            eprintln!("ERROR: extractmagrad - bad crossover value");
            process::exit(1);
        });
        (ver, crossover, Vec::new())
    } else {
        (1, 0.0, vec![first_line])
    };

    let mut c2 = Vec::new();
    let mut grad = Vec::new();
    let mut rot = Vec::new();
    let mut lin = Vec::new();

    for line in &start_lines {
        let vals: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            c2.push(vals[0]);
            grad.push(vals[1]);
            rot.push(vals[2]);
            if version > 1 {
                lin.push(1.0 / (vals[0] - crossover));
            }
        }
    }

    for line in lines {
        let line = line.unwrap();
        let vals: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            c2.push(vals[0]);
            grad.push(vals[1]);
            rot.push(vals[2]);
            if version > 1 {
                lin.push(1.0 / (vals[0] - crossover));
            }
        }
    }

    GradTable {
        c2,
        grad,
        rot,
        lin,
        version,
        crossover,
    }
}

/// Extract tilt angles and intensities from SERI-format extended header.
fn extract_seri_tilts_and_intensities(
    header: &imod_mrc::MrcHeader,
    ext_data: &[u8],
    nz: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nint = header.nint as u16;
    let nreal = header.nreal as u16;

    // nreal bits: bit0=tilt(2), bit1=piececoords(6), bit2=stage(4),
    //             bit3=mag(2), bit4=intensity(2), bit5=dose(4)
    let bit_sizes: [(u16, usize); 6] = [(1, 2), (2, 6), (4, 4), (8, 2), (16, 2), (32, 4)];

    let mut tilt_offset = None;
    let mut intensity_offset = None;
    let mut current_offset = nint as usize * 2;

    for &(bit, size) in &bit_sizes {
        if nreal & bit != 0 {
            if bit == 1 {
                tilt_offset = Some(current_offset);
            }
            if bit == 16 {
                intensity_offset = Some(current_offset);
            }
            current_offset += size;
        }
    }

    let bytes_per_section = if current_offset > 0 {
        current_offset
    } else if header.next > 0 && nz > 0 {
        header.next as usize / nz
    } else {
        return (vec![], vec![]);
    };

    let mut tilts = Vec::with_capacity(nz);
    let mut intensities = Vec::with_capacity(nz);

    for iz in 0..nz {
        let base = iz * bytes_per_section;

        if let Some(t_off) = tilt_offset {
            let off = base + t_off;
            if off + 2 <= ext_data.len() {
                let raw = i16::from_le_bytes([ext_data[off], ext_data[off + 1]]);
                tilts.push(raw as f32 / 100.0);
            } else {
                tilts.push(-999.0);
            }
        } else {
            tilts.push(-999.0);
        }

        if let Some(i_off) = intensity_offset {
            let off = base + i_off;
            if off + 2 <= ext_data.len() {
                let raw = i16::from_le_bytes([ext_data[off], ext_data[off + 1]]);
                intensities.push(raw as f32 / 25000.0);
            } else {
                intensities.push(0.0);
            }
        } else {
            intensities.push(0.0);
        }
    }

    (tilts, intensities)
}

fn main() {
    let args = Args::parse();

    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: extractmagrad - opening input: {}", e);
        process::exit(1);
    });

    let header = reader.header();
    let nz = header.nz as usize;
    let ext_data = reader.ext_header();

    // Get pixel size
    let pixel_size = if let Some(p) = args.pixel {
        p * 10.0 // Convert nm to Angstroms
    } else {
        let delta_x = header.xlen / header.mx as f32;
        if delta_x == 1.0 {
            eprintln!(
                "ERROR: extractmagrad - You must enter a pixel size since the pixel spacing in the header is 1"
            );
            process::exit(1);
        }
        delta_x
    };

    // Extract tilt angles and intensities from extended header
    let (tilts, intensities) = extract_seri_tilts_and_intensities(header, ext_data, nz);

    if tilts.is_empty() || tilts.iter().all(|&t| t == -999.0) {
        eprintln!("ERROR: extractmagrad - No tilt angles in the image file header");
        process::exit(1);
    }
    if intensities.is_empty() {
        eprintln!("ERROR: extractmagrad - No intensities in the image file header");
        process::exit(1);
    }

    // Read gradient table
    let table = read_gradient_table(&args.gradient);

    // Pack down non-empty values
    let mut tilt_out = Vec::new();
    let mut c2_out = Vec::new();
    let n = tilts.len().min(intensities.len()).min(nz);
    for i in 0..n {
        if tilts[i] != -999.0 {
            tilt_out.push(tilts[i]);
            c2_out.push(intensities[i]);
        }
    }

    // Write output
    let mut out = std::fs::File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: extractmagrad - creating output: {}", e);
        process::exit(1);
    });

    writeln!(out, "{:8}", 1).unwrap(); // mag version
    writeln!(
        out,
        "{:6}{:8.3}{:9.2}",
        tilt_out.len(),
        pixel_size,
        args.rotation
    )
    .unwrap();

    let num_in_table = table.c2.len();
    for i in 0..tilt_out.len() {
        let (grad, rot) = if c2_out[i] < table.c2[0] {
            (table.grad[0], table.rot[0])
        } else if c2_out[i] > table.c2[num_in_table - 1] {
            (
                table.grad[num_in_table - 1],
                table.rot[num_in_table - 1],
            )
        } else {
            let mut found_grad = table.grad[0];
            let mut found_rot = table.rot[0];
            for j in 0..num_in_table - 1 {
                let mut frac =
                    (c2_out[i] - table.c2[j]) / (table.c2[j + 1] - table.c2[j]);
                if frac >= 0.0 && frac <= 1.0 {
                    if table.version > 1 {
                        frac = (1.0 / (c2_out[i] - table.crossover) - table.lin[j])
                            / (table.lin[j + 1] - table.lin[j]);
                    }
                    if table.version > 1
                        && table.c2[j] < table.crossover
                        && table.c2[j + 1] > table.crossover
                    {
                        if c2_out[i] < table.crossover {
                            found_grad = table.grad[j];
                            found_rot = table.rot[j];
                        } else {
                            found_grad = table.grad[j + 1];
                            found_rot = table.rot[j + 1];
                        }
                    } else {
                        found_grad =
                            (1.0 - frac) * table.grad[j] + frac * table.grad[j + 1];
                        found_rot =
                            (1.0 - frac) * table.rot[j] + frac * table.rot[j + 1];
                    }
                    break;
                }
            }
            (found_grad, found_rot)
        };
        writeln!(
            out,
            "{:8.2}{:8.3}{:8.3}",
            tilt_out[i],
            grad + args.dgrad,
            rot + args.drot
        )
        .unwrap();
    }

    println!("{} gradients output to file", tilt_out.len());
}
