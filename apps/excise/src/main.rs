use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_mrc::{MrcReader, MrcWriter};

/// Excise (extract) pieces from an image at positions specified in a coordinate file.
///
/// Reads a list of (x, y, z) integer coordinates from a text file, and extracts
/// sub-images of the given size centered at each coordinate. If the sub-image extends
/// beyond the edge, it is padded with the mean intensity of the input.
#[derive(Parser)]
#[command(name = "excise", version, about)]
struct Args {
    /// Input image file (MRC).
    input: String,

    /// File with coordinates to excise (x y z per line).
    #[arg(short = 'p', long = "points")]
    points: String,

    /// Output file for excised pieces (MRC).
    output: String,

    /// X dimension of output pieces.
    #[arg(long = "nx")]
    out_nx: usize,

    /// Y dimension of output pieces.
    #[arg(long = "ny")]
    out_ny: usize,

    /// X offset from coordinate to center of output piece.
    #[arg(long = "xoffset", default_value_t = 0)]
    x_offset: i32,

    /// Y offset from coordinate to center of output piece.
    #[arg(long = "yoffset", default_value_t = 0)]
    y_offset: i32,
}

struct Point3i {
    x: i32,
    y: i32,
    z: i32,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: excise - opening input: {}", e);
        process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let dmean_fill = h.amean;

    // Read coordinates
    let pf = File::open(&args.points).unwrap_or_else(|e| {
        eprintln!("ERROR: excise - opening point file: {}", e);
        process::exit(1);
    });

    let mut coords: Vec<Point3i> = Vec::new();
    for line in BufReader::new(pf).lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let x: i32 = parts[0].parse().unwrap_or_else(|_| {
            eprintln!("ERROR: excise - bad coordinate: {}", line);
            process::exit(1);
        });
        let y: i32 = parts[1].parse().unwrap_or_else(|_| {
            eprintln!("ERROR: excise - bad coordinate: {}", line);
            process::exit(1);
        });
        let z: i32 = parts[2].parse().unwrap_or_else(|_| {
            eprintln!("ERROR: excise - bad coordinate: {}", line);
            process::exit(1);
        });
        coords.push(Point3i { x, y, z });
    }

    let npoint = coords.len();
    if npoint == 0 {
        eprintln!("ERROR: excise - no coordinates read");
        process::exit(1);
    }

    let out_nx = args.out_nx;
    let out_ny = args.out_ny;

    // Create output
    let mut out_header = h.clone();
    out_header.nx = out_nx as i32;
    out_header.ny = out_ny as i32;
    out_header.nz = npoint as i32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = npoint as i32;
    out_header.xlen = out_nx as f32;
    out_header.ylen = out_ny as f32;
    out_header.zlen = npoint as f32;
    out_header.add_label("EXCISE: pieces excised from image file");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: excise - creating output: {}", e);
        process::exit(1);
    });

    // Build list of unique Z values needed
    let mut z_vals: Vec<i32> = coords.iter().map(|c| c.z).collect();
    z_vals.sort();
    z_vals.dedup();

    // Cache: read sections as needed
    let mut dmin = f32::MAX;
    let mut dmax = f32::MIN;
    let mut dsum = 0.0_f64;

    // Process by Z section
    for &iz in &z_vals {
        if iz < 0 || iz as usize >= nz {
            eprintln!("ERROR: excise - section {} does not exist", iz);
            process::exit(1);
        }

        let section = reader.read_slice_f32(iz as usize).unwrap_or_else(|e| {
            eprintln!("ERROR: excise - reading section {}: {}", iz, e);
            process::exit(1);
        });

        // Extract pieces for all points in this section
        for pt in &coords {
            if pt.z != iz {
                continue;
            }

            let cx = pt.x + args.x_offset;
            let cy = pt.y + args.y_offset;
            let x1 = cx - out_nx as i32 / 2;
            let y1 = cy - out_ny as i32 / 2;

            let mut piece = vec![0.0f32; out_nx * out_ny];

            for oy in 0..out_ny {
                for ox in 0..out_nx {
                    let ix = x1 + ox as i32;
                    let iy = y1 + oy as i32;
                    let val = if ix >= 0 && ix < nx as i32 && iy >= 0 && iy < ny as i32 {
                        section[iy as usize * nx + ix as usize]
                    } else {
                        dmean_fill
                    };
                    piece[oy * out_nx + ox] = val;
                }
            }

            // Statistics for this piece
            let mut pmin = f32::MAX;
            let mut pmax = f32::MIN;
            let mut pmean = 0.0_f64;
            for &v in &piece {
                pmin = pmin.min(v);
                pmax = pmax.max(v);
                pmean += v as f64;
            }
            pmean /= (out_nx * out_ny) as f64;
            dmin = dmin.min(pmin);
            dmax = dmax.max(pmax);
            dsum += pmean;

            writer.write_slice_f32(&piece).unwrap_or_else(|e| {
                eprintln!("ERROR: excise - writing piece: {}", e);
                process::exit(1);
            });
        }
    }

    let dmean = (dsum / npoint as f64) as f32;
    writer.finish(dmin, dmax, dmean).unwrap();

    println!("{} pieces excised", npoint);
}
