//! extposition - Extract position numbers from point files.
//!
//! Produces a list of position numbers for portions of images extracted
//! by EXTSTACK. Requires reference point file, extraction point file,
//! and position-1 point file. Assigns position numbers based on angular
//! ordering around centroid of reference points.
//!
//! Translated from IMOD's extposition.f

use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::process;

#[derive(Parser)]
#[command(
    name = "extposition",
    about = "Extract position numbers from point files"
)]
struct Args {
    /// Reference (complete) point file
    #[arg(short = 'r', long)]
    reference: String,

    /// Expected number of reference points per section
    #[arg(short = 'e', long)]
    expected: usize,

    /// Extraction point file
    #[arg(short = 'x', long)]
    extraction: String,

    /// Position 1 point file
    #[arg(short = 'p', long)]
    position: String,

    /// Output file for list of positions
    #[arg(short = 'o', long)]
    output: String,

    /// Count positions clockwise (default: counter-clockwise)
    #[arg(long, default_value_t = false)]
    clockwise: bool,
}

fn read_points(path: &str) -> Vec<(i32, i32, i32)> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: extposition - opening {}: {}", path, e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut pts = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let vals: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            pts.push((vals[0], vals[1], vals[2]));
        }
    }
    pts
}

fn main() {
    let args = Args::parse();

    let ref_pts = read_points(&args.reference);
    let ext_pts = read_points(&args.extraction);
    let pos_pts = read_points(&args.position);

    // Compute centroids per section from reference points
    let max_sec = ref_pts.iter().map(|p| p.2).max().unwrap_or(0) as usize + 1;
    let mut xcen = vec![0.0f64; max_sec];
    let mut ycen = vec![0.0f64; max_sec];
    let mut ccnt = vec![0.0f64; max_sec];

    for p in &ref_pts {
        let z = p.2 as usize;
        if z < max_sec {
            xcen[z] += p.0 as f64;
            ycen[z] += p.1 as f64;
            ccnt[z] += 1.0;
        }
    }

    for z in 0..max_sec {
        if ccnt[z] > 0.0 {
            if ccnt[z] as usize != args.expected {
                eprintln!(
                    "ERROR: Section {} -- expected {} reference points, found {}",
                    z, args.expected, ccnt[z] as usize
                );
                process::exit(1);
            }
            xcen[z] /= ccnt[z];
            ycen[z] /= ccnt[z];
        }
    }

    // Open output file
    let mut out = std::fs::File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: extposition - creating {}: {}", args.output, e);
        process::exit(1);
    });

    let max_ext_sec = ext_pts.iter().map(|p| p.2).max().unwrap_or(0);

    for iz in 0..=max_ext_sec {
        // Get extraction points for this section
        let ext_here: Vec<(f64, f64)> = ext_pts
            .iter()
            .filter(|p| p.2 == iz)
            .map(|p| (p.0 as f64, p.1 as f64))
            .collect();
        if ext_here.is_empty() {
            continue;
        }

        let z = iz as usize;
        if z >= max_sec {
            for _ in 0..ext_here.len() {
                writeln!(out, " 0").unwrap();
            }
            continue;
        }

        // Find position-1 point for this section
        let pos1 = pos_pts.iter().find(|p| p.2 == iz);

        // Get reference points for this section with angles
        let mut ref_angles: Vec<(f64, usize)> = Vec::new();
        for (i, rp) in ref_pts.iter().enumerate() {
            if rp.2 == iz {
                let ang =
                    (rp.1 as f64 - ycen[z]).atan2(rp.0 as f64 - xcen[z]) * 180.0 / std::f64::consts::PI;
                ref_angles.push((ang, i));
            }
        }
        ref_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let ind = ref_angles.len();

        // Find closest reference point to position-1 point
        let ind_min = if let Some(p1) = pos1 {
            let mut min_dist = f64::MAX;
            let mut min_idx = 0usize;
            for (i, &(_, ri)) in ref_angles.iter().enumerate() {
                let rp = &ref_pts[ri];
                let dist = (((rp.0 - p1.0) as f64).powi(2) + ((rp.1 - p1.1) as f64).powi(2))
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = i;
                }
            }
            Some(min_idx)
        } else {
            None
        };

        // Assign position numbers to extraction points
        for (ex, ey) in &ext_here {
            if let Some(ind_min_val) = ind_min {
                // Find closest reference point to this extraction point
                let mut min_dist = f64::MAX;
                let mut ip_min = 0usize;
                for (i, &(_, ri)) in ref_angles.iter().enumerate() {
                    let rp = &ref_pts[ri];
                    let dist = (ex - rp.0 as f64).powi(2) + (ey - rp.1 as f64).powi(2);
                    if dist < min_dist {
                        min_dist = dist;
                        ip_min = i;
                    }
                }
                let mut itype = ip_min as i32 + 1 - ind_min_val as i32;
                if itype <= 0 {
                    itype += ind as i32;
                }
                if args.clockwise {
                    itype = 2 + ind as i32 - itype;
                    if itype > ind as i32 {
                        itype = 1;
                    }
                }
                writeln!(out, "{:2}", itype).unwrap();
            } else {
                writeln!(out, " 0").unwrap();
            }
        }
    }
}
