use std::process;

use clap::Parser;
use imod_model::{ImodModel, read_model, write_model};

/// Extract a subset of objects from an IMOD model file.
///
/// The list of objects can include ranges, e.g. 1-3,6,9,13-15
#[derive(Parser)]
#[command(name = "imodextract", version, about)]
struct Args {
    /// Treat the list as object group numbers instead of object numbers
    #[arg(short = 'g', default_value_t = false)]
    groups: bool,

    /// Delete the listed objects (keep everything else)
    #[arg(short = 'd', default_value_t = false)]
    delete: bool,

    /// Comma-separated list of object numbers (1-based, ranges allowed e.g. 1-3,6)
    list: String,

    /// Input model file
    input: String,

    /// Output model file
    output: String,
}

/// Parse a list string like "1-3,6,9,13-15" into a sorted Vec of integers.
fn parse_list(s: &str) -> Result<Vec<i32>, String> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start: i32 = start_s
                .trim()
                .parse()
                .map_err(|_| format!("Invalid number in range: {}", start_s))?;
            let end: i32 = end_s
                .trim()
                .parse()
                .map_err(|_| format!("Invalid number in range: {}", end_s))?;
            if start > end {
                return Err(format!("Invalid range: {}-{}", start, end));
            }
            for i in start..=end {
                result.push(i);
            }
        } else {
            let n: i32 = part
                .parse()
                .map_err(|_| format!("Invalid number: {}", part))?;
            result.push(n);
        }
    }
    result.sort();
    result.dedup();
    Ok(result)
}

fn main() {
    let args = Args::parse();

    if args.groups {
        eprintln!("ERROR: imodextract - Object group extraction is not yet supported in this version");
        process::exit(1);
    }

    let list = match parse_list(&args.list) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR: imodextract - Parsing object list: {}", e);
            process::exit(1);
        }
    };

    let model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: imodextract - Reading model {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let num_objects = model.objects.len() as i32;

    // Validate all numbers
    for &n in &list {
        if n < 1 || n > num_objects {
            eprintln!(
                "ERROR: imodextract - Invalid object number {} (model has {} objects)",
                n, num_objects
            );
            process::exit(1);
        }
    }

    // Convert to 0-based indices
    let indices_0based: Vec<usize> = list.iter().map(|&n| (n - 1) as usize).collect();

    // Determine which objects to keep
    let keep_indices: Vec<usize> = if args.delete {
        // Keep everything NOT in the list
        (0..num_objects as usize)
            .filter(|i| !indices_0based.contains(i))
            .collect()
    } else {
        indices_0based
    };

    if keep_indices.is_empty() {
        eprintln!("ERROR: imodextract - No objects are left after the operation");
        process::exit(1);
    }

    // Build output model
    let mut out_model = ImodModel {
        name: model.name.clone(),
        xmax: model.xmax,
        ymax: model.ymax,
        zmax: model.zmax,
        flags: model.flags,
        drawmode: model.drawmode,
        mousemode: model.mousemode,
        black_level: model.black_level,
        white_level: model.white_level,
        offset: model.offset,
        scale: model.scale,
        pixel_size: model.pixel_size,
        units: model.units,
        objects: Vec::with_capacity(keep_indices.len()),
        views: model.views.clone(),
        ref_image: model.ref_image.clone(),
        slicer_angles: model.slicer_angles.clone(),
        store: model.store.clone(),
    };

    for &idx in &keep_indices {
        out_model.objects.push(model.objects[idx].clone());
    }

    if let Err(e) = write_model(&args.output, &out_model) {
        eprintln!("ERROR: imodextract - Writing model {}: {}", args.output, e);
        process::exit(1);
    }

    println!(
        "Extracted {} object(s) to {}",
        out_model.objects.len(),
        args.output
    );
}
