use std::process;

use clap::Parser;
use imod_model::{read_model, write_model};

/// Remap model coordinates: shift, remap Z values, or reorder points.
///
/// Allows remapping of Z values in a model one-to-one to new Z values, and/or
/// shifting all X, Y, Z coordinates by constants. Points whose Z maps to a
/// value of -999..-990 are removed.
#[derive(Parser)]
#[command(name = "remapmodel", version, about)]
struct Args {
    /// Input IMOD model file.
    input: String,

    /// Output IMOD model file.
    output: String,

    /// Amount to add to all X coordinates.
    #[arg(long = "addx", default_value_t = 0.0)]
    add_x: f32,

    /// Amount to add to all Y coordinates.
    #[arg(long = "addy", default_value_t = 0.0)]
    add_y: f32,

    /// Amount to add to all Z coordinates.
    #[arg(long = "addz", default_value_t = 0.0)]
    add_z: f32,

    /// Old Z value list (comma-separated or range, e.g. "0-10,15").
    #[arg(long = "old")]
    old_z: Option<String>,

    /// New Z value list (same count as old list).
    #[arg(long = "new")]
    new_z: Option<String>,

    /// Use full range of Z values in model as old list.
    #[arg(long = "full")]
    full_range: bool,

    /// Reorder points by Z: 1 = ascending, -1 = descending, 0 = no.
    #[arg(long = "reorder", default_value_t = 0)]
    reorder: i32,
}

/// Parse a list string like "1,3-5,8" into a Vec of integers.
fn parse_list(s: &str) -> Vec<i32> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: i32 = a.trim().parse().unwrap_or_else(|_| {
                eprintln!("ERROR: remapmodel - bad range start: {}", a);
                process::exit(1);
            });
            let end: i32 = b.trim().parse().unwrap_or_else(|_| {
                eprintln!("ERROR: remapmodel - bad range end: {}", b);
                process::exit(1);
            });
            if start <= end {
                for v in start..=end {
                    result.push(v);
                }
            } else {
                for v in (end..=start).rev() {
                    result.push(v);
                }
            }
        } else {
            let v: i32 = part.parse().unwrap_or_else(|_| {
                eprintln!("ERROR: remapmodel - bad value: {}", part);
                process::exit(1);
            });
            result.push(v);
        }
    }
    result
}

fn main() {
    let args = Args::parse();

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: remapmodel - reading model: {}", e);
            process::exit(1);
        }
    };

    // Collect all unique Z values from the model
    let mut all_z: Vec<i32> = Vec::new();
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                let iz = pt.z.round() as i32;
                if !all_z.contains(&iz) {
                    all_z.push(iz);
                }
            }
        }
    }
    all_z.sort();

    // Build old Z list
    let old_list = if let Some(ref old_str) = args.old_z {
        if args.full_range {
            eprintln!("ERROR: remapmodel - cannot use both --old and --full");
            process::exit(1);
        }
        parse_list(old_str)
    } else if args.full_range {
        let min_z = all_z.first().copied().unwrap_or(0);
        let max_z = all_z.last().copied().unwrap_or(0);
        (min_z..=max_z).collect()
    } else {
        all_z.clone()
    };

    // Build new Z list
    let new_list = if let Some(ref new_str) = args.new_z {
        parse_list(new_str)
    } else if args.add_x != 0.0 || args.add_y != 0.0 || args.add_z != 0.0 {
        // No new list needed if just shifting
        old_list.clone()
    } else {
        eprintln!("ERROR: remapmodel - no new Z list specified and no shift");
        process::exit(1);
    };

    if old_list.len() != new_list.len() {
        eprintln!(
            "ERROR: remapmodel - old list has {} values but new list has {}",
            old_list.len(),
            new_list.len()
        );
        process::exit(1);
    }

    // Build Z mapping: old -> new
    let mut z_map = std::collections::HashMap::new();
    for (old, new) in old_list.iter().zip(new_list.iter()) {
        z_map.insert(*old, *new);
    }

    // Apply mapping and shifts to all points
    for obj in &mut model.objects {
        for cont in &mut obj.contours {
            // Apply Z remapping and shifts, remove points with Z mapped to -999..-990
            cont.points.retain_mut(|pt| {
                let iz = pt.z.round() as i32;
                let new_z = z_map.get(&iz).copied().unwrap_or(iz);

                if (-999..=-990).contains(&new_z) {
                    return false; // remove this point
                }

                pt.x += args.add_x;
                pt.y += args.add_y;
                pt.z += args.add_z + (new_z - iz) as f32;
                true
            });

            // Reorder points by Z if requested
            if args.reorder != 0 {
                if args.reorder > 0 {
                    cont.points.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
                } else {
                    cont.points.sort_by(|a, b| b.z.partial_cmp(&a.z).unwrap());
                }
            }
        }
    }

    // Update model max Z
    let mut max_z = model.zmax;
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                max_z = max_z.max(pt.z.round() as i32);
            }
        }
    }
    model.zmax = max_z;

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: remapmodel - writing model: {}", e);
        process::exit(1);
    }

    println!("Model remapped and written to {}", args.output);
}
