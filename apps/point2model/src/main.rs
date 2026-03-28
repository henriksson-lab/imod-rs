use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};

/// Convert a text file of point coordinates into an IMOD model.
///
/// Reads lines of "x y z" (whitespace or comma separated) and creates a model
/// with one object containing those points. By default all points go into a
/// single contour; use --planar to split by Z value, or --per-contour N to
/// put N points per contour.
#[derive(Parser)]
#[command(name = "point2model", version, about)]
struct Args {
    /// Input text file with x y z coordinates (one point per line).
    input: String,

    /// Output IMOD model file.
    output: String,

    /// Treat contours as open (default: closed).
    #[arg(short = 'o', long = "open")]
    open: bool,

    /// Mark object as scattered points.
    #[arg(short = 's', long = "scat")]
    scattered: bool,

    /// Sort points into separate contours by Z value (planar contours).
    #[arg(short = 'p', long = "planar")]
    planar: bool,

    /// Number of points per contour (0 = all in one contour).
    #[arg(short = 'n', long = "per-contour", default_value_t = 0)]
    per_contour: usize,

    /// Sphere radius for display.
    #[arg(long = "sphere", default_value_t = 0)]
    sphere: i32,

    /// Number of lines to skip at the start of the file.
    #[arg(long = "skip", default_value_t = 0)]
    skip: usize,

    /// Circle size for display.
    #[arg(long = "circle", default_value_t = 0)]
    circle: i32,
}

fn main() {
    let args = Args::parse();

    let file = match File::open(&args.input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: point2model - error opening {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let reader = BufReader::new(file);
    let mut points: Vec<Point3f> = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!(
                    "ERROR: point2model - error reading line {}: {}",
                    line_num + 1,
                    e
                );
                process::exit(1);
            }
        };

        if line_num < args.skip {
            continue;
        }

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse x y z from whitespace or comma-separated values
        let parts: Vec<&str> = line.split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() < 3 {
            eprintln!(
                "WARNING: point2model - skipping line {} (need at least 3 values): {}",
                line_num + 1,
                line
            );
            continue;
        }

        let x: f32 = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "WARNING: point2model - skipping line {} (bad x value): {}",
                    line_num + 1,
                    line
                );
                continue;
            }
        };
        let y: f32 = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "WARNING: point2model - skipping line {} (bad y value): {}",
                    line_num + 1,
                    line
                );
                continue;
            }
        };
        let z: f32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "WARNING: point2model - skipping line {} (bad z value): {}",
                    line_num + 1,
                    line
                );
                continue;
            }
        };

        points.push(Point3f { x, y, z });
    }

    if points.is_empty() {
        eprintln!("ERROR: point2model - no points read from {}", args.input);
        process::exit(1);
    }

    println!("Read {} points from {}", points.len(), args.input);

    // Build contours
    let contours = if args.planar {
        // Group by Z value
        let mut by_z: std::collections::BTreeMap<i64, Vec<Point3f>> =
            std::collections::BTreeMap::new();
        for p in &points {
            // Use integer Z as key (bit pattern for exact grouping)
            let key = (p.z * 1000.0).round() as i64;
            by_z.entry(key).or_default().push(*p);
        }
        by_z.into_values()
            .map(|pts| ImodContour {
                points: pts,
                ..Default::default()
            })
            .collect::<Vec<_>>()
    } else if args.per_contour > 0 {
        points
            .chunks(args.per_contour)
            .map(|chunk| ImodContour {
                points: chunk.to_vec(),
                ..Default::default()
            })
            .collect::<Vec<_>>()
    } else {
        vec![ImodContour {
            points,
            ..Default::default()
        }]
    };

    let n_contours = contours.len();
    let n_points: usize = contours.iter().map(|c| c.points.len()).sum();

    // Compute bounding box for model header
    let mut xmax: f32 = 0.0;
    let mut ymax: f32 = 0.0;
    let mut zmax: f32 = 0.0;
    for cont in &contours {
        for p in &cont.points {
            xmax = xmax.max(p.x);
            ymax = ymax.max(p.y);
            zmax = zmax.max(p.z);
        }
    }

    // Build object flags
    let mut obj_flags: u32 = 0;
    if args.open {
        obj_flags |= 1 << 3; // IMOD_OBJFLAG_OPEN
    }
    if args.scattered {
        obj_flags |= 1 << 1; // IMOD_OBJFLAG_SCAT
    }

    let obj = ImodObject {
        flags: obj_flags,
        pdrawsize: args.sphere,
        symsize: args.circle as u8,
        contours,
        ..Default::default()
    };

    let model = ImodModel {
        xmax: (xmax.ceil() as i32).max(1),
        ymax: (ymax.ceil() as i32).max(1),
        zmax: (zmax.ceil() as i32).max(1),
        objects: vec![obj],
        ..Default::default()
    };

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: point2model - error writing {}: {}", args.output, e);
        process::exit(1);
    }

    println!(
        "Created model with 1 object, {} contour(s), {} point(s)",
        n_contours, n_points
    );
    println!("Wrote {}", args.output);
}
