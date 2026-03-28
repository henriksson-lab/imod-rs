//! rotmont - Rotate montage sections by +/-90 degrees.
//!
//! Reads a montage MRC file and its piece list, rotates selected sections
//! by +90 (counterclockwise) or -90 (clockwise) degrees, writes the
//! rotated images and an updated piece coordinate list.
//!
//! Translated from IMOD's rotmont.f

use clap::Parser;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::io::{BufRead, BufReader, Write};
use std::process;

#[derive(Parser)]
#[command(name = "rotmont", about = "Rotate montage sections by +/- 90 degrees")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Input piece list file
    #[arg(short = 'p', long)]
    pieces: String,

    /// Output MRC image file
    #[arg(short = 'o', long)]
    output: String,

    /// Output piece list file
    #[arg(long)]
    outpieces: String,

    /// Rotate clockwise (-90) instead of counterclockwise (+90)
    #[arg(short = 'c', long, default_value_t = false)]
    clockwise: bool,

    /// Sections to rotate (comma-separated; default: all)
    #[arg(short = 's', long)]
    sections: Option<String>,

    /// Amounts to add to X, Y, Z piece coordinates
    #[arg(long, num_args = 3, value_names = ["DX", "DY", "DZ"], default_values_t = [0, 0, 0])]
    add: Vec<i32>,
}

fn read_piece_list(path: &str) -> Vec<(i32, i32, i32)> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: rotmont - opening piece list {}: {}", path, e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut pieces = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let vals: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            pieces.push((vals[0], vals[1], vals[2]));
        }
    }
    pieces
}

fn check_list(coords: &[i32], frame_size: i32) -> (i32, i32, i32) {
    let mut sorted: Vec<i32> = coords.to_vec();
    sorted.sort();
    sorted.dedup();
    let min_coord = sorted[0];
    let num_pieces = sorted.len() as i32;
    if num_pieces <= 1 {
        return (min_coord, num_pieces, 0);
    }
    let spacing = sorted[1] - sorted[0];
    let overlap = frame_size - spacing;
    (min_coord, num_pieces, overlap)
}

fn rotate_array(array: &[f32], nx: usize, ny: usize, clockwise: bool) -> Vec<f32> {
    // Output is ny x nx
    let mut out = vec![0.0f32; ny * nx];
    if !clockwise {
        // +90 (counterclockwise)
        for iy in 0..ny {
            let ixo = ny - 1 - iy;
            for ix in 0..nx {
                out[ix * ny + ixo] = array[iy * nx + ix];
            }
        }
    } else {
        // -90 (clockwise)
        for ix in 0..nx {
            let iyo = nx - 1 - ix;
            for iy in 0..ny {
                out[iyo * ny + iy] = array[iy * nx + ix];
            }
        }
    }
    out
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: rotmont - opening input: {}", e);
        process::exit(1);
    });
    let header = reader.header();
    let nxin = header.nx as usize;
    let nyin = header.ny as usize;
    let nzin = header.nz as usize;

    let mut pieces = read_piece_list(&args.pieces);
    if pieces.is_empty() {
        eprintln!("ERROR: rotmont - empty piece list");
        process::exit(1);
    }

    let ix_coords: Vec<i32> = pieces.iter().map(|p| p.0).collect();
    let iy_coords: Vec<i32> = pieces.iter().map(|p| p.1).collect();
    let (minx, nxpieces, nxoverlap) = check_list(&ix_coords, nxin as i32);
    let (miny, nypieces, nyoverlap) = check_list(&iy_coords, nyin as i32);

    if nxpieces <= 0 || nypieces <= 0 {
        eprintln!("ERROR: rotmont - piece list not valid");
        process::exit(1);
    }

    // Rotate piece coordinates
    if !args.clockwise {
        let max_rot_xy = (nypieces - 1) * (nyin as i32 - nyoverlap) + miny;
        for p in pieces.iter_mut() {
            let tmp = p.0;
            p.0 = max_rot_xy - p.1;
            p.1 = tmp;
        }
    } else {
        let max_rot_xy = (nxpieces - 1) * (nxin as i32 - nxoverlap) + minx;
        for p in pieces.iter_mut() {
            let tmp = p.1;
            p.1 = max_rot_xy - p.0;
            p.0 = tmp;
        }
    }

    // Parse section list
    let section_list: Vec<i32> = if let Some(ref s) = args.sections {
        s.split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect()
    } else {
        Vec::new() // empty means all
    };

    let nxout = nyin;
    let nyout = nxin;

    let out_mode = imod_core::MrcMode::from_i32(header.mode).unwrap_or(imod_core::MrcMode::Float);
    let out_header = MrcHeader::new(nxout as i32, nyout as i32, nzin as i32, out_mode);
    let mut writer = MrcWriter::create(&args.output, out_header)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: rotmont - creating output: {}", e);
            process::exit(1);
        });

    let mut out_pieces: Vec<(i32, i32, i32)> = Vec::new();
    let mut dmin_out = f32::MAX;
    let mut dmax_out = f32::MIN;
    let mut grand_sum = 0.0f64;
    let mut nzout = 0usize;

    for iz in 0..nzin {
        let do_rotate = if section_list.is_empty() {
            true
        } else {
            section_list.contains(&(pieces[iz].2))
        };

        if do_rotate {
            let section = reader.read_slice_f32(iz).unwrap_or_else(|e| {
                eprintln!("ERROR: rotmont - reading section {}: {}", iz, e);
                process::exit(1);
            });

            let rotated = rotate_array(&section, nxin, nyin, args.clockwise);

            // Compute stats
            let mut tmin = f32::MAX;
            let mut tmax = f32::MIN;
            let mut tsum = 0.0f64;
            for &v in &rotated {
                tmin = tmin.min(v);
                tmax = tmax.max(v);
                tsum += v as f64;
            }
            let tmean = tsum / (nxout * nyout) as f64;

            dmin_out = dmin_out.min(tmin);
            dmax_out = dmax_out.max(tmax);
            grand_sum += tmean;

            writer.write_slice_f32(&rotated).unwrap_or_else(|e| {
                eprintln!("ERROR: rotmont - writing section: {}", e);
                process::exit(1);
            });

            out_pieces.push((
                pieces[iz].0 + args.add[0],
                pieces[iz].1 + args.add[1],
                pieces[iz].2 + args.add[2],
            ));
            nzout += 1;
        }
    }

    let dmean_out = if nzout > 0 { (grand_sum / nzout as f64) as f32 } else { 0.0 };
    writer.finish(dmin_out, dmax_out, dmean_out).unwrap_or_else(|e| {
        eprintln!("ERROR: rotmont - finalizing output: {}", e);
        process::exit(1);
    });

    // Write piece list
    let mut pf = std::fs::File::create(&args.outpieces).unwrap_or_else(|e| {
        eprintln!("ERROR: rotmont - creating piece list: {}", e);
        process::exit(1);
    });
    for (x, y, z) in &out_pieces {
        writeln!(pf, "{:6}{:6}{:4}", x, y, z).unwrap();
    }

    println!(
        "Rotated {} sections, output {} x {} x {}",
        nzout, nxout, nyout, nzout
    );
}
