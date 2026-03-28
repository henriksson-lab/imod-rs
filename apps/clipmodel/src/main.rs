use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodObject, read_model, write_model};

/// Clip a model to a bounding box in 3D coordinate space.
///
/// Can include or exclude points within defined X/Y/Z ranges, delete
/// specific objects, or clip points from the start/end of contours.
#[derive(Parser)]
#[command(name = "clipmodel", version, about)]
struct Args {
    /// X range to include/exclude (min,max)
    #[arg(long = "xminmax")]
    x_range: Option<Vec<String>>,

    /// Y range to include/exclude (min,max)
    #[arg(long = "yminmax")]
    y_range: Option<Vec<String>>,

    /// Z range to include/exclude (min,max)
    #[arg(long = "zminmax")]
    z_range: Option<Vec<String>>,

    /// Exclude (1) or include (0) points in the coordinate block.
    /// Can be specified multiple times for multiple operations.
    #[arg(short = 'e', long = "exclude")]
    exclude: Option<Vec<i32>>,

    /// List of objects to operate on (1-based, ranges allowed)
    #[arg(short = 'O', long = "objects")]
    object_list: Option<String>,

    /// Output as point file: 0=model, 1=points, -1=corner points
    #[arg(short = 'p', long = "point", default_value_t = 0)]
    point_output: i32,

    /// Keep only the longest included contour segment (0) or all (1)
    #[arg(short = 'l', long = "longest", default_value_t = false)]
    longest_only: bool,

    /// Keep empty contours in output
    #[arg(short = 'k', long = "keep-empty", default_value_t = false)]
    keep_empty: bool,

    /// Number of points to clip from start of contours
    #[arg(long = "clip-start")]
    clip_start: Option<usize>,

    /// Number of points to clip from end of contours
    #[arg(long = "clip-end")]
    clip_end: Option<usize>,

    /// Input model file
    input: String,

    /// Output model file
    output: String,
}

fn parse_range(s: &str) -> Result<(f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("Expected min,max but got: {}", s));
    }
    let a: f32 = parts[0].trim().parse().map_err(|e| format!("{}", e))?;
    let b: f32 = parts[1].trim().parse().map_err(|e| format!("{}", e))?;
    Ok((a, b))
}

fn parse_list(s: &str) -> Result<Vec<i32>, String> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: i32 = a.trim().parse().map_err(|_| format!("Invalid: {}", a))?;
            let end: i32 = b.trim().parse().map_err(|_| format!("Invalid: {}", b))?;
            for i in start..=end {
                result.push(i);
            }
        } else {
            result.push(part.parse().map_err(|_| format!("Invalid: {}", part))?);
        }
    }
    Ok(result)
}

/// Check whether a point is inside the given bounding box.
fn point_in_box(p: &Point3f, x_range: (f32, f32), y_range: (f32, f32), z_range: (f32, f32)) -> bool {
    p.x >= x_range.0 && p.x <= x_range.1
        && p.y >= y_range.0 && p.y <= y_range.1
        && p.z >= z_range.0 && p.z <= z_range.1
}

fn main() {
    let args = Args::parse();

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: clipmodel - Reading model {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let obj_list: Option<Vec<i32>> = args.object_list.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clipmodel - Parsing object list: {}", e);
            process::exit(1);
        })
    });

    // Handle object deletion
    if let Some(ref list) = obj_list {
        if args.exclude.is_none() && args.x_range.is_none() && args.y_range.is_none()
            && args.z_range.is_none() && args.clip_start.is_none()
        {
            // Delete objects mode
            let keep: Vec<ImodObject> = model
                .objects
                .iter()
                .enumerate()
                .filter(|(i, _)| !list.contains(&((*i + 1) as i32)))
                .map(|(_, o)| o.clone())
                .collect();
            model.objects = keep;
            if let Err(e) = write_model(&args.output, &model) {
                eprintln!("ERROR: clipmodel - Writing model: {}", e);
                process::exit(1);
            }
            println!("Wrote model with {} objects", model.objects.len());
            return;
        }
    }

    // Handle clip-from-start-and-end
    if args.clip_start.is_some() || args.clip_end.is_some() {
        let clip_start = args.clip_start.unwrap_or(0);
        let clip_end = args.clip_end.unwrap_or(0);

        for (i, obj) in model.objects.iter_mut().enumerate() {
            let should_process = match &obj_list {
                Some(list) => list.contains(&((i + 1) as i32)),
                None => true,
            };
            if !should_process {
                continue;
            }
            for cont in &mut obj.contours {
                let n = cont.points.len();
                if clip_start + clip_end >= n {
                    cont.points.clear();
                } else {
                    cont.points = cont.points[clip_start..(n - clip_end)].to_vec();
                }
            }
            if !args.keep_empty {
                obj.contours.retain(|c| !c.points.is_empty());
            }
        }

        if let Err(e) = write_model(&args.output, &model) {
            eprintln!("ERROR: clipmodel - Writing model: {}", e);
            process::exit(1);
        }
        println!("Clipped contours, wrote {}", args.output);
        return;
    }

    // Coordinate-based clipping
    let default_range = (-1.0e9_f32, 1.0e9_f32);

    // Determine number of clipping operations
    let num_ops = args.exclude.as_ref().map(|v| v.len()).unwrap_or(1);

    for op in 0..num_ops {
        let is_exclude = args
            .exclude
            .as_ref()
            .map(|v| v.get(op).copied().unwrap_or(0) != 0)
            .unwrap_or(false);

        let x_range = args
            .x_range
            .as_ref()
            .and_then(|v| v.get(op))
            .map(|s| parse_range(s).unwrap_or_else(|e| {
                eprintln!("ERROR: clipmodel - Bad X range: {}", e);
                process::exit(1);
            }))
            .unwrap_or(default_range);

        let y_range = args
            .y_range
            .as_ref()
            .and_then(|v| v.get(op))
            .map(|s| parse_range(s).unwrap_or_else(|e| {
                eprintln!("ERROR: clipmodel - Bad Y range: {}", e);
                process::exit(1);
            }))
            .unwrap_or(default_range);

        let z_range = args
            .z_range
            .as_ref()
            .and_then(|v| v.get(op))
            .map(|s| parse_range(s).unwrap_or_else(|e| {
                eprintln!("ERROR: clipmodel - Bad Z range: {}", e);
                process::exit(1);
            }))
            .unwrap_or(default_range);

        for obj in &mut model.objects {
            for cont in &mut obj.contours {
                if args.longest_only && !is_exclude {
                    // Keep only the longest contiguous segment inside the box
                    let mut best_start = 0;
                    let mut best_len = 0;
                    let mut cur_start = 0;
                    let mut cur_len = 0;

                    for (i, p) in cont.points.iter().enumerate() {
                        let inside = point_in_box(p, x_range, y_range, z_range);
                        if inside {
                            if cur_len == 0 {
                                cur_start = i;
                            }
                            cur_len += 1;
                            if cur_len > best_len {
                                best_len = cur_len;
                                best_start = cur_start;
                            }
                        } else {
                            cur_len = 0;
                        }
                    }

                    if best_len > 0 {
                        cont.points = cont.points[best_start..best_start + best_len].to_vec();
                    } else {
                        cont.points.clear();
                    }
                } else {
                    // Filter points
                    cont.points.retain(|p| {
                        let inside = point_in_box(p, x_range, y_range, z_range);
                        if is_exclude {
                            !inside
                        } else {
                            inside
                        }
                    });
                }
            }

            if !args.keep_empty {
                obj.contours.retain(|c| !c.points.is_empty());
            }
        }
    }

    // Point file output
    if args.point_output != 0 {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(&args.output).unwrap_or_else(|e| {
            eprintln!("ERROR: clipmodel - Creating output file: {}", e);
            process::exit(1);
        });
        let mut count = 0;
        for obj in &model.objects {
            for cont in &obj.contours {
                for p in &cont.points {
                    writeln!(f, "{:.2} {:.2} {:.2}", p.x, p.y, p.z).ok();
                    count += 1;
                }
            }
        }
        println!("Wrote {} points to {}", count, args.output);
        return;
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: clipmodel - Writing model: {}", e);
        process::exit(1);
    }

    let total_points: usize = model
        .objects
        .iter()
        .flat_map(|o| &o.contours)
        .map(|c| c.points.len())
        .sum();
    println!(
        "Wrote model with {} objects, {} total points",
        model.objects.len(),
        total_points
    );
}
