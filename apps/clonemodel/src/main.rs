use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodModel, ImodObject, read_model, write_model};

/// Create a new model containing multiple copies of an input model at specified
/// positions and orientations.
///
/// The location file is CSV with a header line, containing columns:
///   contour, x, y, z, xAngle, yAngle, zAngle [, value]
#[derive(Parser)]
#[command(name = "clonemodel", version, about)]
struct Args {
    /// CSV file with locations/orientations (contour,x,y,z,xAngle,yAngle,zAngle[,value])
    #[arg(short = 'a', long = "at", required = true)]
    at_points: String,

    /// X range to include (min,max)
    #[arg(short = 'x', long = "xrange")]
    x_range: Option<String>,

    /// Y range to include (min,max)
    #[arg(short = 'y', long = "yrange")]
    y_range: Option<String>,

    /// Z range to include (min,max)
    #[arg(short = 'z', long = "zrange")]
    z_range: Option<String>,

    /// Value range to filter by (min,max)
    #[arg(long = "vrange")]
    v_range: Option<String>,

    /// List of contour numbers to include
    #[arg(long = "contours")]
    contours: Option<String>,

    /// Input model file (template to clone)
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

/// Apply a 3D rotation (Z-Y-X Euler angles in degrees) and translation to a point.
fn transform_point(p: Point3f, center: Point3f, x_angle: f32, y_angle: f32, z_angle: f32, translation: Point3f) -> Point3f {
    let deg2rad = PI / 180.0;

    // Center the point
    let px = p.x - center.x;
    let py = p.y - center.y;
    let pz = p.z - center.z;

    // Rotate Z
    let cz = (z_angle * deg2rad).cos();
    let sz = (z_angle * deg2rad).sin();
    let x1 = cz * px - sz * py;
    let y1 = sz * px + cz * py;
    let z1 = pz;

    // Rotate Y
    let cy = (y_angle * deg2rad).cos();
    let sy = (y_angle * deg2rad).sin();
    let x2 = cy * x1 + sy * z1;
    let y2 = y1;
    let z2 = -sy * x1 + cy * z1;

    // Rotate X
    let cx = (x_angle * deg2rad).cos();
    let sx = (x_angle * deg2rad).sin();
    let x3 = x2;
    let y3 = cx * y2 - sx * z2;
    let z3 = sx * y2 + cx * z2;

    Point3f {
        x: x3 + translation.x,
        y: y3 + translation.y,
        z: z3 + translation.z,
    }
}

/// Standard pseudo-color ramp (simplified version of IMOD's cmapStandardRamp).
fn pseudo_color(fraction: f32) -> (f32, f32, f32) {
    let f = fraction.clamp(0.0, 1.0);
    // HSV-like mapping: blue -> cyan -> green -> yellow -> red
    let r;
    let g;
    let b;
    if f < 0.25 {
        r = 0.0;
        g = 4.0 * f;
        b = 1.0;
    } else if f < 0.5 {
        r = 0.0;
        g = 1.0;
        b = 2.0 - 4.0 * f;
    } else if f < 0.75 {
        r = 4.0 * f - 2.0;
        g = 1.0;
        b = 0.0;
    } else {
        r = 1.0;
        g = 4.0 - 4.0 * f;
        b = 0.0;
    }
    (r, g, b)
}

fn main() {
    let args = Args::parse();

    let (x_min, x_max) = args
        .x_range
        .as_ref()
        .map(|s| parse_range(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clonemodel - Bad X range: {}", e);
            process::exit(1);
        }))
        .unwrap_or((0.0, f32::MAX));

    let (y_min, y_max) = args
        .y_range
        .as_ref()
        .map(|s| parse_range(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clonemodel - Bad Y range: {}", e);
            process::exit(1);
        }))
        .unwrap_or((0.0, f32::MAX));

    let (z_min, z_max) = args
        .z_range
        .as_ref()
        .map(|s| parse_range(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clonemodel - Bad Z range: {}", e);
            process::exit(1);
        }))
        .unwrap_or((0.0, f32::MAX));

    let v_range = args.v_range.as_ref().map(|s| {
        parse_range(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clonemodel - Bad V range: {}", e);
            process::exit(1);
        })
    });

    let contour_list = args.contours.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: clonemodel - Bad contour list: {}", e);
            process::exit(1);
        })
    });

    // Read input model
    let in_model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: clonemodel - Reading model {}: {}", args.input, e);
            process::exit(1);
        }
    };

    // Read CSV location file
    let coord_file = match File::open(&args.at_points) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "ERROR: clonemodel - Opening location file {}: {}",
                args.at_points, e
            );
            process::exit(1);
        }
    };
    let reader = BufReader::new(coord_file);
    let mut csv_lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    if csv_lines.is_empty() {
        eprintln!("ERROR: clonemodel - Location file is empty");
        process::exit(1);
    }

    // Skip header line
    csv_lines.remove(0);

    // First pass: find value range if values present
    let mut v_min = f32::MAX;
    let mut v_max = f32::MIN;
    let mut values_present = false;

    struct LocationEntry {
        contour: i32,
        x: f32,
        y: f32,
        z: f32,
        x_angle: f32,
        y_angle: f32,
        z_angle: f32,
        value: Option<f32>,
    }

    let mut entries = Vec::new();
    for line in &csv_lines {
        let parts: Vec<f32> = line
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();

        if parts.len() < 7 {
            eprintln!("ERROR: clonemodel - Cannot parse line: {}", line);
            process::exit(1);
        }

        let entry = LocationEntry {
            contour: parts[0] as i32,
            x: parts[1],
            y: parts[2],
            z: parts[3],
            x_angle: parts[4],
            y_angle: parts[5],
            z_angle: parts[6],
            value: if parts.len() >= 8 {
                values_present = true;
                let v = parts[7];
                if v < v_min {
                    v_min = v;
                }
                if v > v_max {
                    v_max = v;
                }
                Some(v)
            } else {
                None
            },
        };
        entries.push(entry);
    }

    if let Some((vr_min, vr_max)) = v_range {
        v_min = vr_min;
        v_max = vr_max;
    }

    // Model center for rotation
    let center = Point3f {
        x: in_model.xmax as f32 / 2.0,
        y: in_model.ymax as f32 / 2.0,
        z: in_model.zmax as f32 / 2.0,
    };

    // Build output model
    let mut out_model = ImodModel {
        pixel_size: in_model.pixel_size,
        units: in_model.units,
        ..ImodModel::default()
    };

    let mut max_x = 0.0_f32;
    let mut max_y = 0.0_f32;
    let mut max_z = 0.0_f32;

    for entry in &entries {
        // Check contour filter
        if let Some(ref clist) = contour_list {
            if !clist.contains(&entry.contour) {
                continue;
            }
        }

        // Check spatial range
        if entry.x < x_min || entry.x > x_max
            || entry.y < y_min || entry.y > y_max
            || entry.z < z_min || entry.z > z_max
        {
            continue;
        }

        let translation = Point3f {
            x: entry.x,
            y: entry.y,
            z: entry.z,
        };

        max_x = max_x.max(entry.x);
        max_y = max_y.max(entry.y);
        max_z = max_z.max(entry.z);

        // Clone each object from the input model
        for in_obj in &in_model.objects {
            let mut new_obj = in_obj.clone();

            // Assign pseudo-color if values present
            if values_present {
                if let Some(value) = entry.value {
                    let frac = if (v_max - v_min).abs() > 1e-10 {
                        (value - v_min) / (v_max - v_min)
                    } else {
                        0.5
                    };
                    let (r, g, b) = pseudo_color(frac);
                    new_obj.red = r;
                    new_obj.green = g;
                    new_obj.blue = b;
                }
            }

            // Transform all contour points
            for cont in &mut new_obj.contours {
                for pt in &mut cont.points {
                    *pt = transform_point(
                        *pt,
                        center,
                        entry.x_angle,
                        entry.y_angle,
                        entry.z_angle,
                        translation,
                    );
                }
            }

            // Transform mesh vertices
            for mesh in &mut new_obj.meshes {
                for v in &mut mesh.vertices {
                    *v = transform_point(
                        *v,
                        center,
                        entry.x_angle,
                        entry.y_angle,
                        entry.z_angle,
                        translation,
                    );
                }
            }

            out_model.objects.push(new_obj);
        }
    }

    out_model.xmax = (max_x + center.x) as i32;
    out_model.ymax = (max_y + center.y) as i32;
    out_model.zmax = (max_z + center.z) as i32;

    if let Err(e) = write_model(&args.output, &out_model) {
        eprintln!("ERROR: clonemodel - Writing model {}: {}", args.output, e);
        process::exit(1);
    }

    println!(
        "Wrote {} objects to {}",
        out_model.objects.len(),
        args.output
    );
}
