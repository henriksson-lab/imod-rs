use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};

const MAX_VALUE_COLS: usize = 6;
const DEFAULT_SCALE: f32 = 10.0;

/// Convert a patch vector file to an IMOD model.
///
/// Reads a patch file (produced by corrsearch3d or similar) containing position
/// and displacement vectors, and creates an IMOD model with two-point contours
/// representing each vector.
#[derive(Parser)]
#[command(name = "patch2imod", version, about)]
struct Args {
    /// Scale vectors by given value
    #[arg(short = 's', default_value_t = DEFAULT_SCALE)]
    scale: f32,

    /// Do NOT flip Y and Z coordinates
    #[arg(short = 'f')]
    no_flip: bool,

    /// Add given name to model object
    #[arg(short = 'n')]
    name: Option<String>,

    /// Set up clipping planes enclosing area of given size
    #[arg(short = 'c')]
    clip_size: Option<i32>,

    /// Set flag to display values in false color in 3dmod
    #[arg(short = 'd')]
    display_values: bool,

    /// Value column number (1-6) to store as primary value
    #[arg(short = 'v')]
    val_col: Option<i32>,

    /// Ignore zero values when computing SD-limited maximum
    #[arg(short = 'z')]
    ignore_zero: bool,

    /// Use all lines in file (do not get count from first line)
    #[arg(short = 'l')]
    count_lines: bool,

    /// Give each contour a time equal to Z value plus 1
    #[arg(short = 't')]
    times_for_z: bool,

    /// Input patch file
    patch_file: String,

    /// Output IMOD model file
    output_model: String,
}

/// IMOD model flags
const IMODF_FLIPYZ: u32 = 1 << 14;
/// IMOD object flags
const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;
const IMOD_OBJFLAG_THICK_CONT: u32 = 1 << 10;
const IMOD_OBJFLAG_USE_VALUE: u32 = 1 << 7;

fn main() {
    let args = Args::parse();

    let file = File::open(&args.patch_file).unwrap_or_else(|e| {
        eprintln!("ERROR: patch2imod - Couldn't open {}: {}", args.patch_file, e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = reader.lines().map(|l| l.unwrap_or_default()).collect();

    let mut npatch: usize;
    let mut residuals = false;
    let mut data_start = 0;
    let mut val_col_out: i32 = args.val_col.unwrap_or(-1);
    let mut value_ids = [0i32; MAX_VALUE_COLS];
    let mut num_val_ids = 0usize;
    let mut val_type_map = [0usize; MAX_VALUE_COLS];
    let mut ordered_ids = [0i32; MAX_VALUE_COLS];

    if args.count_lines {
        // Count lines with enough content
        npatch = lines.iter().filter(|l| l.trim().len() > 2).count();
        if npatch == 0 {
            eprintln!("ERROR: patch2imod - No usable lines in the file");
            process::exit(1);
        }
        if val_col_out > 0 {
            eprintln!("ERROR: patch2imod - Cannot enter a column ID with -l");
            process::exit(1);
        }
    } else {
        // Parse first line for count and value IDs
        let first_line = &lines[0];
        residuals = first_line.contains("residuals");
        let tokens: Vec<&str> = first_line.split_whitespace().collect();
        if tokens.is_empty() {
            eprintln!("ERROR: patch2imod - Empty first line");
            process::exit(1);
        }
        npatch = tokens[0].parse().unwrap_or_else(|_| {
            eprintln!("ERROR: patch2imod - Cannot parse patch count from first line");
            process::exit(1);
        });
        // Parse additional value IDs from first line
        for token in tokens.iter().skip(1) {
            if let Ok(v) = token.parse::<i32>() {
                if num_val_ids < MAX_VALUE_COLS {
                    value_ids[num_val_ids] = v;
                    num_val_ids += 1;
                }
            }
        }

        // Convert val_col_out from ID to column index
        if val_col_out < 0 {
            val_col_out = (-val_col_out - 1).max(0);
        } else {
            let mut found = false;
            for (idx, &vid) in value_ids[..num_val_ids].iter().enumerate() {
                if vid == val_col_out {
                    println!("The value with ID {} is in extra column {}", val_col_out, idx + 1);
                    val_col_out = idx as i32;
                    found = true;
                    break;
                }
            }
            if !found {
                eprintln!("ERROR: patch2imod - No value column ID matching entered ID");
                process::exit(1);
            }
        }

        data_start = 1;
    }

    // Build ordered IDs and type map
    let vco = val_col_out.max(0) as usize;
    if num_val_ids > 0 {
        ordered_ids[0] = value_ids[vco.min(num_val_ids - 1)];
        val_type_map[vco] = 0;
        let mut oi = 1;
        let mut ci = 0;
        for i in 0..MAX_VALUE_COLS {
            if i == vco {
                continue;
            }
            if ci < num_val_ids {
                ordered_ids[oi] = value_ids[ci];
            }
            val_type_map[i] = oi;
            oi += 1;
            ci += 1;
        }
    }

    if npatch == 0 {
        eprintln!("ERROR: patch2imod - Implausible number of patches = 0");
        process::exit(1);
    }

    // Parse patch data
    let scale = args.scale;
    let no_flip = args.no_flip;
    let mut contours = Vec::with_capacity(npatch);
    let mut xmin = 1_000_000i32;
    let mut ymin = 1_000_000i32;
    let mut zmin = 1_000_000i32;
    let mut xmax = -1_000_000i32;
    let mut ymax = -1_000_000i32;
    let mut zmax = -1_000_000i32;
    let mut dz_vary = false;
    let mut max_val_cols = 0usize;
    let mut val_min = [1.0e30f32; MAX_VALUE_COLS];
    let mut val_max = [-1.0e30f32; MAX_VALUE_COLS];
    let mut val_sum = [0.0f64; MAX_VALUE_COLS];
    let mut val_sq_sum = [0.0f64; MAX_VALUE_COLS];
    let mut num_vals = [0usize; MAX_VALUE_COLS];
    // Store general values per contour: (contour_index, type_index, value)
    let mut stored_values: Vec<(usize, usize, f32)> = Vec::new();

    let data_lines: Vec<&str> = if args.count_lines {
        lines.iter().filter(|l| l.trim().len() > 2).map(|s| s.as_str()).collect()
    } else {
        lines[data_start..].iter().map(|s| s.as_str()).collect()
    };

    for (pat, line) in data_lines.iter().enumerate().take(npatch) {
        let xx: f32;
        let yy: f32;
        let iz: i32;
        let dx: f32;
        let dy: f32;
        let mut dz: f32 = 0.0;
        let mut values = [0.0f32; MAX_VALUE_COLS];
        let mut nread;

        if residuals {
            let parts: Vec<f32> = line.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            if parts.len() < 5 {
                eprintln!("ERROR: patch2imod - Error reading line {}", pat + 1);
                process::exit(1);
            }
            xx = parts[0];
            yy = parts[1];
            iz = parts[2] as i32;
            dx = parts[3];
            dy = parts[4];
            nread = 5;
        } else {
            let cleaned = line.replace(',', " ");
            let parts: Vec<&str> = cleaned.split_whitespace().collect();
            if parts.len() < 6 {
                eprintln!("ERROR: patch2imod - Error reading line {}", pat + 1);
                process::exit(1);
            }
            let ix: i32 = parts[0].parse().unwrap_or(0);
            let p1: i32 = parts[1].parse().unwrap_or(0);
            let p2: i32 = parts[2].parse().unwrap_or(0);
            let d0: f32 = parts[3].parse().unwrap_or(0.0);
            let d1: f32 = parts[4].parse().unwrap_or(0.0);
            let d2: f32 = parts[5].parse().unwrap_or(0.0);

            if no_flip {
                // ix, iy, iz format
                xx = ix as f32;
                yy = p1 as f32;
                iz = p2;
                dx = d0;
                dy = d1;
                dz = d2;
            } else {
                // ix, iz, iy format (IMOD convention for non-flipped)
                xx = ix as f32;
                yy = p2 as f32;
                iz = p1;
                dx = d0;
                dz = d1;
                dy = d2;
            }

            nread = parts.len();
            // Read extra value columns
            for vi in 0..(nread - 6).min(MAX_VALUE_COLS) {
                if let Ok(v) = parts[6 + vi].parse::<f32>() {
                    values[vi] = v;
                }
            }
        }

        let p0 = Point3f { x: xx, y: yy, z: iz as f32 };
        let p1 = Point3f {
            x: xx + scale * dx,
            y: yy + scale * dy,
            z: iz as f32 + scale * dz,
        };

        let mut cont = ImodContour {
            points: vec![p0, p1],
            ..Default::default()
        };
        if args.times_for_z {
            cont.time = (iz + 1).max(1);
        }
        contours.push(cont);

        xmin = xmin.min(xx as i32);
        ymin = ymin.min(yy as i32);
        zmin = zmin.min(iz);
        xmax = xmax.max(xx as i32 + 1);
        ymax = ymax.max(yy as i32 + 1);
        zmax = zmax.max(iz + 1);
        if dz != 0.0 {
            dz_vary = true;
        }

        // Store extra values
        let num_extra = if residuals { 0 } else { (nread - 6).min(MAX_VALUE_COLS) };
        max_val_cols = max_val_cols.max(num_extra);
        for i in 0..num_extra {
            let ind = val_type_map[i];
            let value = values[i];
            val_min[ind] = val_min[ind].min(value);
            val_max[ind] = val_max[ind].max(value);
            if value != 0.0 || !args.ignore_zero {
                num_vals[ind] += 1;
                val_sum[ind] += value as f64;
                val_sq_sum[ind] += (value as f64) * (value as f64);
            }
            stored_values.push((pat, ind, value));
        }
    }

    // Build object
    let mut obj_flags = IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_THICK_CONT;
    if args.display_values {
        obj_flags |= IMOD_OBJFLAG_USE_VALUE;
    }
    if args.times_for_z {
        obj_flags |= 1 << 5; // IMOD_OBJFLAG_TIME
    }

    let mut obj = ImodObject {
        contours,
        flags: obj_flags,
        ..Default::default()
    };

    if let Some(ref name) = args.name {
        obj.name = name.clone();
    }

    if residuals {
        obj.symbol = 0; // IOBJ_SYM_NONE
        obj.symsize = 7;
    } else if dz_vary {
        obj.symbol = 1; // IOBJ_SYM_CIRCLE
    }

    // Limit val_max by mean + 10*SD
    for i in 0..max_val_cols {
        if num_vals[i] > 10 {
            let n = num_vals[i] as f64;
            let variance = (val_sq_sum[i] - val_sum[i] * val_sum[i] / n) / (n - 1.0);
            if variance > 0.0 {
                let sd_max = val_sum[i] / n + 10.0 * variance.sqrt();
                val_max[i] = val_max[i].min(sd_max as f32);
            }
        }
    }

    let mut model_flags = 0u32;
    if !residuals && !no_flip {
        model_flags |= IMODF_FLIPYZ;
    }

    let model = ImodModel {
        xmax: xmax + xmin,
        ymax: ymax + ymin,
        zmax: zmax + zmin,
        pixel_size: scale,
        flags: model_flags,
        objects: vec![obj],
        ..Default::default()
    };

    write_model(&args.output_model, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: patch2imod - writing model: {}", e);
        process::exit(1);
    });
}
