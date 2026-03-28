//! maxjoinsize - Compute maximum size and offsets for joining serial sections.
//!
//! Reads transforms from a .tomoxg file and image sizes from files listed
//! in a .info file, and computes the maximum output size and centering offset
//! needed to contain all transformed data.
//!
//! Translated from IMOD's maxjoinsize.f

use clap::Parser;
use imod_mrc::MrcReader;
use std::io::{BufRead, BufReader};
use std::process;

#[derive(Parser)]
#[command(
    name = "maxjoinsize",
    about = "Compute max join size for serial section joining"
)]
struct Args {
    /// Number of sections (files) to join
    num_sections: usize,

    /// Number of lines to skip in .info file before filenames
    lines_to_skip: usize,

    /// Root name for .tomoxg and .info files
    root_name: String,
}

/// A 2D affine transform stored as [[a11, a12, dx], [a21, a22, dy]]
#[derive(Clone, Copy)]
struct Xform {
    a: [[f64; 3]; 2],
}

impl Xform {
    #[allow(dead_code)]
    fn identity() -> Self {
        Xform {
            a: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    fn apply(&self, xcen: f64, ycen: f64, x: f64, y: f64) -> (f64, f64) {
        let dx = x - xcen;
        let dy = y - ycen;
        let xout = self.a[0][0] * dx + self.a[0][1] * dy + self.a[0][2] + xcen;
        let yout = self.a[1][0] * dx + self.a[1][1] * dy + self.a[1][2] + ycen;
        (xout, yout)
    }
}

fn read_xforms(path: &str) -> Vec<Xform> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: maxjoinsize - opening {}: {}", path, e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut xforms = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 6 {
            xforms.push(Xform {
                a: [
                    [vals[0], vals[1], vals[4]],
                    [vals[2], vals[3], vals[5]],
                ],
            });
        }
    }
    xforms
}

fn main() {
    let args = Args::parse();

    let xf_path = format!("{}.tomoxg", args.root_name);
    let info_path = format!("{}.info", args.root_name);

    // Read transforms
    let xforms = read_xforms(&xf_path);
    if xforms.len() != args.num_sections {
        eprintln!(
            "ERROR: maxjoinsize - wrong number of transforms ({}, expected {})",
            xforms.len(),
            args.num_sections
        );
        process::exit(1);
    }

    // Read info file and skip header lines
    let info_file = std::fs::File::open(&info_path).unwrap_or_else(|e| {
        eprintln!("ERROR: maxjoinsize - opening {}: {}", info_path, e);
        process::exit(1);
    });
    let info_reader = BufReader::new(info_file);
    let all_lines: Vec<String> = info_reader.lines().filter_map(|l| l.ok()).collect();

    // Get image filenames after skipping lines
    let mut filenames = Vec::new();
    for i in args.lines_to_skip..all_lines.len() {
        if filenames.len() >= args.num_sections {
            break;
        }
        filenames.push(all_lines[i].trim().to_string());
    }

    if filenames.len() != args.num_sections {
        eprintln!("ERROR: maxjoinsize - not enough filenames in info file");
        process::exit(1);
    }

    // Get image sizes
    let mut nx_list = Vec::new();
    let mut ny_list = Vec::new();
    let mut maxx = 0i32;
    let mut maxy = 0i32;

    for fname in &filenames {
        let reader = MrcReader::open(fname).unwrap_or_else(|e| {
            eprintln!("ERROR: maxjoinsize - opening {}: {}", fname, e);
            process::exit(1);
        });
        let h = reader.header();
        nx_list.push(h.nx);
        ny_list.push(h.ny);
        maxx = maxx.max(h.nx);
        maxy = maxy.max(h.ny);
    }

    // Transform the 4 corners and find bounding box
    let xcen = maxx as f64 / 2.0;
    let ycen = maxy as f64 / 2.0;
    let mut xmin = xcen;
    let mut ymin = ycen;
    let mut xmax = xcen;
    let mut ymax = ycen;

    for i in 0..args.num_sections {
        let xhalf = nx_list[i] as f64 / 2.0;
        let yhalf = ny_list[i] as f64 / 2.0;
        for &dirx in &[-1.0f64, 1.0] {
            for &diry in &[-1.0f64, 1.0] {
                let (xcorn, ycorn) = xforms[i].apply(
                    xcen,
                    ycen,
                    xcen + dirx * xhalf,
                    ycen + diry * yhalf,
                );
                xmin = xmin.min(xcorn);
                ymin = ymin.min(ycorn);
                xmax = xmax.max(xcorn);
                ymax = ymax.max(ycorn);
            }
        }
    }

    let ixofs = ((xmax + xmin) * 0.5 - xcen).round() as i32;
    let iyofs = ((ymax + ymin) * 0.5 - ycen).round() as i32;
    let newx = (2.0 * ((xmax - xmin) * 0.5).round()) as i32;
    let newy = (2.0 * ((ymax - ymin) * 0.5).round()) as i32;

    println!("Maximum size required:{:9}{:9}", newx, newy);
    println!("Offset needed to center:{:8}{:8}", ixofs, iyofs);
}
