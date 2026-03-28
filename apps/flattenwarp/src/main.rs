use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, read_model, write_model};
use imod_transforms::LinearTransform;
use imod_warp::WarpFile;

/// Compute warping transforms to flatten a tomographic volume.
///
/// Reads a model with contours tracing the top and/or bottom surfaces of
/// a section, computes the middle surface, and generates warp transforms
/// that flatten the volume so the section is level. Can work with either
/// contour-based or scattered-point surface models.
#[derive(Parser)]
#[command(name = "flattenwarp", version, about)]
struct Args {
    /// Input model file with surface contours.
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output warp transform file.
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// Output patch file for warp transforms.
    #[arg(long = "patch")]
    patch_file: Option<String>,

    /// Output file for middle contours.
    #[arg(long = "middle")]
    middle_file: Option<String>,

    /// Binning of tomogram in XY and Z.
    #[arg(long = "binning", num_args = 1..=2, default_values_t = vec![1, 1])]
    binning: Vec<i32>,

    /// Treat as single surface (not top + bottom).
    #[arg(long = "one")]
    one_surface: bool,

    /// Flip option: -1 = auto, 0 = no flip, 1 = flip Y/Z, 2 = rotate.
    #[arg(long = "flip", default_value_t = -1)]
    flip_option: i32,

    /// Warp spacing in X and Y.
    #[arg(long = "spacing", num_args = 2)]
    warp_spacing: Vec<f32>,

    /// Lambda values for smoothing (one or more).
    #[arg(long = "lambda")]
    lambda: Vec<f32>,

    /// Show intermediate contours in output.
    #[arg(long = "show")]
    show_contours: bool,

    /// Restore original orientation in output.
    #[arg(long = "restore")]
    restore_orientation: bool,
}

/// Contour data with metadata for sorting and interpolation.
struct ContData {
    points: Vec<Point3f>,
    y_val: i32,
    x_min: f32,
    x_max: f32,
}

/// Interpolate Z value at a given X position along a contour.
fn interpolate_cont(points: &[Point3f], x_val: f32) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    // Find bracketing points
    for i in 0..points.len() - 1 {
        let x0 = points[i].x;
        let x1 = points[i + 1].x;
        if (x0 <= x_val && x_val <= x1) || (x1 <= x_val && x_val <= x0) {
            let frac = if (x1 - x0).abs() < 1.0e-6 {
                0.5
            } else {
                (x_val - x0) / (x1 - x0)
            };
            return Some(points[i].z + frac * (points[i + 1].z - points[i].z));
        }
    }
    None
}

/// Flip Y and Z coordinates in a model (for handling flipped tomograms).
fn flip_model_yz(model: &mut ImodModel) {
    for obj in &mut model.objects {
        for cont in &mut obj.contours {
            for pt in &mut cont.points {
                std::mem::swap(&mut pt.y, &mut pt.z);
            }
        }
    }
    std::mem::swap(&mut model.ymax, &mut model.zmax);
}

fn main() {
    let args = Args::parse();

    // Read input model
    let mut model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: flattenwarp - error reading model {}: {}", args.input, e);
        process::exit(1);
    });

    let output_file = match &args.output {
        Some(f) => f.clone(),
        None => {
            if args.lambda.len() < 2 {
                eprintln!("ERROR: flattenwarp - no output file specified");
                process::exit(1);
            }
            String::new()
        }
    };

    let xy_binning = args.binning.get(0).copied().unwrap_or(1);
    let _z_binning = args.binning.get(1).copied().unwrap_or(1);

    // Handle flip/rotation state
    let _flipped = if args.flip_option > 0 || (args.flip_option < 0 && model.zmax > model.ymax) {
        flip_model_yz(&mut model);
        true
    } else {
        false
    };

    let num_objects = model.objects.len();
    if num_objects == 0 {
        eprintln!("ERROR: flattenwarp - model has no objects");
        process::exit(1);
    }

    // Check if the model has scattered points (flags bit 1 = scattered)
    let _is_scattered = model.objects[0].flags & (1 << 1) != 0;

    // Collect contour data from the model
    let mut all_contours: Vec<Vec<ContData>> = Vec::new();
    let num_surfaces = if args.one_surface || num_objects < 2 { 1 } else { 2 };

    for ob in 0..num_surfaces.min(num_objects) {
        let obj = &model.objects[ob];
        let mut contours = Vec::new();

        for cont in &obj.contours {
            if cont.points.is_empty() {
                continue;
            }

            let y_val = cont.points[0].y as i32;
            let x_min = cont.points.iter().map(|p| p.x).fold(f32::MAX, f32::min);
            let x_max = cont.points.iter().map(|p| p.x).fold(f32::MIN, f32::max);

            contours.push(ContData {
                points: cont.points.clone(),
                y_val,
                x_min,
                x_max,
            });
        }

        // Sort by Y value
        contours.sort_by_key(|c| c.y_val);
        all_contours.push(contours);
    }

    if all_contours.is_empty() || all_contours[0].is_empty() {
        eprintln!("ERROR: flattenwarp - no usable contours found in model");
        process::exit(1);
    }

    // Determine bounding box
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;

    for surface in &all_contours {
        for cd in surface {
            x_min = x_min.min(cd.x_min);
            x_max = x_max.max(cd.x_max);
            y_min = y_min.min(cd.y_val as f32);
            y_max = y_max.max(cd.y_val as f32);
        }
    }

    // Determine warp grid spacing
    let x_spacing = if args.warp_spacing.len() >= 2 && args.warp_spacing[0] > 0.0 {
        args.warp_spacing[0]
    } else {
        (x_max - x_min) / 4.5
    };

    let y_spacing = if args.warp_spacing.len() >= 2 && args.warp_spacing[1] > 0.0 {
        args.warp_spacing[1]
    } else {
        (y_max - y_min) / 3.0
    };

    let num_x_locs = ((x_max - x_min) / x_spacing) as usize + 1;
    let num_y_locs = ((y_max - y_min) / y_spacing) as usize + 1;
    let num_locs = num_x_locs * num_y_locs;

    println!("Surface bounds: X [{:.0}, {:.0}], Y [{:.0}, {:.0}]", x_min, x_max, y_min, y_max);
    println!(
        "Warp grid: {} x {} locations (spacing {:.1} x {:.1})",
        num_x_locs, num_y_locs, x_spacing, y_spacing
    );

    let _xcen = (x_min + x_max) / 2.0;
    let _ycen = (y_min + y_max) / 2.0;

    // Compute the middle surface (average of top and bottom, or single surface)
    // and the warp displacements needed to flatten it.
    let mut warp_dz: Vec<f32> = vec![0.0; num_locs];
    let mut valid_count = 0;
    let mut z_center = 0.0_f64;

    // First pass: compute middle Z at each grid location
    let mut mid_z_values: Vec<Option<f32>> = vec![None; num_locs];

    for iy in 0..num_y_locs {
        let y_pos = y_min + iy as f32 * y_spacing;

        for ix in 0..num_x_locs {
            let x_pos = x_min + ix as f32 * x_spacing;
            let idx = iy * num_x_locs + ix;

            if num_surfaces == 2 && all_contours.len() >= 2 {
                // Find nearest contours in both surfaces for this Y
                let z_top = find_z_at_xy(&all_contours[0], x_pos, y_pos);
                let z_bot = find_z_at_xy(&all_contours[1], x_pos, y_pos);

                if let (Some(zt), Some(zb)) = (z_top, z_bot) {
                    let mid = (zt + zb) / 2.0;
                    mid_z_values[idx] = Some(mid);
                    z_center += mid as f64;
                    valid_count += 1;
                }
            } else {
                let z_val = find_z_at_xy(&all_contours[0], x_pos, y_pos);
                if let Some(z) = z_val {
                    mid_z_values[idx] = Some(z);
                    z_center += z as f64;
                    valid_count += 1;
                }
            }
        }
    }

    if valid_count == 0 {
        eprintln!("ERROR: flattenwarp - no valid grid positions found");
        process::exit(1);
    }

    z_center /= valid_count as f64;
    let z_center_f = z_center as f32;

    // Compute warp displacements
    for idx in 0..num_locs {
        if let Some(mid) = mid_z_values[idx] {
            warp_dz[idx] = z_center_f - mid;
        }
    }

    println!(
        "Center Z = {:.1}, {} valid warp locations out of {}",
        z_center_f, valid_count, num_locs
    );

    // Write middle contour model if requested
    if let Some(ref mid_file) = args.middle_file {
        let mut mid_model = ImodModel::default();
        mid_model.xmax = model.xmax;
        mid_model.ymax = model.ymax;
        mid_model.zmax = model.zmax;

        let mut obj = ImodObject::default();
        obj.name = "Middle contours".into();
        obj.flags |= 1 << 3; // open
        obj.red = 0.0;
        obj.green = 1.0;
        obj.blue = 1.0;

        // Generate middle contours at each Y position
        for iy in 0..num_y_locs {
            let y_pos = y_min + iy as f32 * y_spacing;
            let mut points = Vec::new();

            for ix in 0..num_x_locs {
                let x_pos = x_min + ix as f32 * x_spacing;
                let idx = iy * num_x_locs + ix;
                if let Some(mid) = mid_z_values[idx] {
                    points.push(Point3f { x: x_pos, y: y_pos, z: mid });
                }
            }

            if points.len() >= 2 {
                obj.contours.push(ImodContour {
                    points,
                    ..Default::default()
                });
            }
        }

        mid_model.objects.push(obj);
        if let Err(e) = write_model(mid_file, &mid_model) {
            eprintln!("ERROR: flattenwarp - error writing middle contour model: {}", e);
        } else {
            println!("Wrote middle contour model to {}", mid_file);
        }
    }

    // Write the warp output file
    if !output_file.is_empty() {
        let mut warp = WarpFile {
            nx: model.xmax,
            ny: model.ymax,
            binning: xy_binning,
            pixel_size: 1.0,
            version: 1,
            flags: 0,
            sections: Vec::new(),
        };

        // Create warp transforms for each Z slice
        // The warp is a Z-shift at each XY grid point
        let nz = model.zmax;
        for iz in 0..nz {
            let mut sec = imod_warp::WarpTransform {
                z: iz,
                nx: num_x_locs as i32,
                ny: num_y_locs as i32,
                control_x: Vec::new(),
                control_y: Vec::new(),
                transforms: Vec::new(),
            };

            for iy in 0..num_y_locs {
                for ix in 0..num_x_locs {
                    let x_pos = x_min + ix as f32 * x_spacing;
                    let y_pos = y_min + iy as f32 * y_spacing;
                    let idx = iy * num_x_locs + ix;

                    sec.control_x.push(x_pos);
                    sec.control_y.push(y_pos);

                    // Identity transform with Z displacement encoded as DY
                    sec.transforms.push(LinearTransform {
                        a11: 1.0,
                        a12: 0.0,
                        a21: 0.0,
                        a22: 1.0,
                        dx: 0.0,
                        dy: warp_dz[idx],
                    });
                }
            }

            warp.sections.push(sec);
        }

        if let Err(e) = warp.write_to_file(&output_file) {
            eprintln!("ERROR: flattenwarp - error writing warp file: {}", e);
            process::exit(1);
        }
        println!("Wrote warp file to {}", output_file);
    }

    // Write patch file if requested
    if let Some(ref patch_file) = args.patch_file {
        let mut f = std::fs::File::create(patch_file).unwrap_or_else(|e| {
            eprintln!("ERROR: flattenwarp - could not create patch file: {}", e);
            process::exit(1);
        });

        use std::io::Write;
        writeln!(f, "{} positions", valid_count).ok();

        for iy in 0..num_y_locs {
            let y_pos = y_min + iy as f32 * y_spacing;
            for ix in 0..num_x_locs {
                let x_pos = x_min + ix as f32 * x_spacing;
                let idx = iy * num_x_locs + ix;
                if mid_z_values[idx].is_some() {
                    writeln!(
                        f,
                        "{:.2} {:.2} {:.2}  0.00 0.00 {:.2}",
                        x_pos, y_pos, z_center_f, warp_dz[idx]
                    ).ok();
                }
            }
        }
        println!("Wrote patch file to {}", patch_file);
    }
}

/// Find the Z value at a given XY position by finding the nearest Y contour
/// and interpolating along X.
fn find_z_at_xy(contours: &[ContData], x: f32, y: f32) -> Option<f32> {
    if contours.is_empty() {
        return None;
    }

    // Find the two nearest contours by Y value
    let mut best_dist = f32::MAX;
    let mut best_z: Option<f32> = None;

    for cd in contours {
        let dy = (cd.y_val as f32 - y).abs();
        if dy < best_dist && x >= cd.x_min && x <= cd.x_max {
            if let Some(z) = interpolate_cont(&cd.points, x) {
                best_dist = dy;
                best_z = Some(z);
            }
        }
    }

    best_z
}
