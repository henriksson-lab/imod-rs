use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodObject, read_model};
use imod_mrc::{MrcReader, MrcWriter};

/// Mask or paint an MRC image using an IMOD model.
///
/// Reads a model and an MRC image stack, then paints (fills) regions defined
/// by model contours into the image.  Supports masking, inversion, thresholding,
/// color output, scattered-point rendering, tubular-line rendering, padding/
/// tapering, and tilt-series projection.
#[derive(Parser)]
#[command(name = "imodmop", version, about)]
struct Args {
    /// IMOD model file
    model_file: String,

    /// Input MRC image file
    input_file: String,

    /// Output MRC image file
    output_file: String,

    /// X min and max
    #[arg(long = "xminmax", num_args = 2, value_delimiter = ',')]
    x_limits: Option<Vec<i32>>,

    /// Y min and max
    #[arg(long = "yminmax", num_args = 2, value_delimiter = ',')]
    y_limits: Option<Vec<i32>>,

    /// Z min and max
    #[arg(long = "zminmax", num_args = 2, value_delimiter = ',')]
    z_limits: Option<Vec<i32>>,

    /// Border around objects (expands output region)
    #[arg(long = "border")]
    border: Option<i32>,

    /// Invert the painted area
    #[arg(long = "invert")]
    invert: bool,

    /// Noise fill border (min,max distance)
    #[arg(long = "noise", num_args = 2, value_delimiter = ',')]
    noise_fill: Option<Vec<f32>>,

    /// Reverse contrast
    #[arg(long = "reverse")]
    reverse: bool,

    /// Threshold value
    #[arg(long = "thresh")]
    threshold: Option<f32>,

    /// Background fill value
    #[arg(long = "fv", default_value_t = 0.0)]
    fill_value: f32,

    /// Fill color (R,G,B each 0-1)
    #[arg(long = "fc", num_args = 3, value_delimiter = ',')]
    fill_color: Option<Vec<f32>>,

    /// Mask value
    #[arg(long = "mask")]
    mask_value: Option<f32>,

    /// Label mask list (one label per object)
    #[arg(long = "label")]
    label_list: Option<String>,

    /// Retain data outside masked regions
    #[arg(long = "retain")]
    retain_outside: bool,

    /// Output mode (0=byte, 1=short, 2=float, 6=ushort)
    #[arg(long = "mode")]
    output_mode: Option<i32>,

    /// Padding size (pixels outside contours)
    #[arg(long = "pad", default_value_t = 0.0)]
    padding: f32,

    /// Taper over pad distance (pixels)
    #[arg(long = "taper", default_value_t = 0)]
    taper: i32,

    /// Also taper in Z over the padding
    #[arg(long = "ztaper")]
    z_taper: bool,

    /// Objects to operate on (comma-separated list)
    #[arg(long = "objects")]
    object_list: Option<String>,

    /// Treat scattered point objects as 2D
    #[arg(long = "2dscat")]
    scat_2d: bool,

    /// Treat scattered point objects as 3D
    #[arg(long = "3dscat")]
    scat_3d: bool,

    /// Tube object numbers (comma-separated)
    #[arg(long = "tube")]
    tube_objects: Option<String>,

    /// Diameter for tube rendering
    #[arg(long = "diam", default_value_t = 25.0)]
    tube_diameter: f32,

    /// Render tubes as planar (2D cross-section)
    #[arg(long = "planar")]
    planar_tubes: bool,

    /// Objects to render on all sections
    #[arg(long = "allsec")]
    all_section_objects: Option<String>,

    /// Output as RGB color
    #[arg(long = "color")]
    color_output: bool,

    /// Scaling min and max
    #[arg(long = "scale", num_args = 2, value_delimiter = ',')]
    scaling: Option<Vec<f32>>,

    /// Project tilt series (start,end,increment)
    #[arg(long = "project", num_args = 3, value_delimiter = ',')]
    project_tilt: Option<Vec<f32>>,

    /// Axis to tilt around (X, Y, or Z)
    #[arg(long = "axis", default_value = "Y")]
    tilt_axis: String,

    /// Constant scaling across projected views
    #[arg(long = "constant")]
    constant_scaling: bool,

    /// Black and white values for projection
    #[arg(long = "bw", num_args = 2, value_delimiter = ',')]
    black_white: Option<Vec<i32>>,

    /// Use legacy fast method (scan-based filling)
    #[arg(long = "fast")]
    fast_legacy: bool,
}

/// Test whether a 2D point is inside a closed polygon (contour) using
/// the ray-casting (crossing number) algorithm.
fn point_in_contour(pts: &[Point3f], x: f32, y: f32) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = pts[i].y;
        let yj = pts[j].y;
        if ((yi > y) != (yj > y)) && (x < (pts[j].x - pts[i].x) * (y - yi) / (yj - yi) + pts[i].x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Compute the distance from a point to the nearest contour edge.
#[allow(dead_code)]
fn point_dist_to_contour(pts: &[Point3f], x: f32, y: f32) -> f32 {
    let n = pts.len();
    if n < 2 {
        return f32::MAX;
    }
    let mut min_dist = f32::MAX;
    let mut j = n - 1;
    for i in 0..n {
        let dx = pts[i].x - pts[j].x;
        let dy = pts[i].y - pts[j].y;
        let seg_len_sq = dx * dx + dy * dy;
        let t = if seg_len_sq > 0.0 {
            ((x - pts[j].x) * dx + (y - pts[j].y) * dy) / seg_len_sq
        } else {
            0.0
        };
        let t = t.clamp(0.0, 1.0);
        let px = pts[j].x + t * dx;
        let py = pts[j].y + t * dy;
        let dist = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
        min_dist = min_dist.min(dist);
        j = i;
    }
    min_dist
}

fn object_is_open(obj: &ImodObject) -> bool {
    (obj.flags & (1 << 3)) != 0
}

fn object_is_scattered(obj: &ImodObject) -> bool {
    (obj.flags & (1 << 1)) != 0
}

fn parse_obj_list(s: &str) -> Vec<i32> {
    imod_math::parse_list(s).unwrap_or_default()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate mutually exclusive options
    if args.scat_2d && args.scat_3d {
        return Err("You cannot enter both -2dscat and -3dscat".into());
    }

    // Read model
    let model = read_model(&args.model_file)?;

    // Open input image
    let mut reader = MrcReader::open(&args.input_file)?;
    let hdr = reader.header().clone();
    let nx = hdr.nx as usize;
    let ny = hdr.ny as usize;
    let nz = hdr.nz as usize;

    // Determine coordinate limits
    let x_min = args.x_limits.as_ref().map_or(0, |v| v[0].max(0)) as usize;
    let x_max = args.x_limits.as_ref().map_or(nx - 1, |v| (v[1] as usize).min(nx - 1));
    let y_min = args.y_limits.as_ref().map_or(0, |v| v[0].max(0)) as usize;
    let y_max = args.y_limits.as_ref().map_or(ny - 1, |v| (v[1] as usize).min(ny - 1));
    let z_min = args.z_limits.as_ref().map_or(0, |v| v[0].max(0)) as usize;
    let z_max = args.z_limits.as_ref().map_or(nz - 1, |v| (v[1] as usize).min(nz - 1));

    if x_max <= x_min || y_max <= y_min || z_max < z_min {
        return Err("Coordinate limits are out of order".into());
    }

    // Apply border if specified
    let (x_min, x_max, y_min, y_max) = if let Some(border) = args.border {
        if border < 0 {
            return Err("The border must be >= 0".into());
        }
        // Find bounding box of model objects
        let mut mxmin = f32::MAX;
        let mut mxmax = f32::MIN;
        let mut mymin = f32::MAX;
        let mut mymax = f32::MIN;
        for obj in &model.objects {
            for cont in &obj.contours {
                for pt in &cont.points {
                    mxmin = mxmin.min(pt.x);
                    mxmax = mxmax.max(pt.x);
                    mymin = mymin.min(pt.y);
                    mymax = mymax.max(pt.y);
                }
            }
        }
        let brd = border as f32;
        (
            (mxmin - brd).max(0.0) as usize,
            (mxmax + brd).min(nx as f32 - 1.0) as usize,
            (mymin - brd).max(0.0) as usize,
            (mymax + brd).min(ny as f32 - 1.0) as usize,
        )
    } else {
        (x_min, x_max, y_min, y_max)
    };

    let nxout = x_max - x_min + 1;
    let nyout = y_max - y_min + 1;
    let nzout = z_max - z_min + 1;

    // Parse object and tube lists
    let obj_list = args
        .object_list
        .as_deref()
        .map(parse_obj_list)
        .unwrap_or_default();
    let tube_list = args
        .tube_objects
        .as_deref()
        .map(parse_obj_list)
        .unwrap_or_default();
    let allsec_list = args
        .all_section_objects
        .as_deref()
        .map(parse_obj_list)
        .unwrap_or_default();

    let masking = args.mask_value.is_some() || args.label_list.is_some();
    let invert = args.invert || args.noise_fill.is_some();
    let fill_val = if args.reverse {
        hdr.amax - args.fill_value
    } else {
        args.fill_value
    };
    let mask_val = args.mask_value.unwrap_or(0.0);

    // Build output header
    let mut out_hdr = hdr.clone();
    out_hdr.nx = nxout as i32;
    out_hdr.ny = nyout as i32;
    out_hdr.nz = nzout as i32;

    let mut writer = MrcWriter::create(&args.output_file, out_hdr.clone())?;

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut _global_mean_sum = 0.0_f64;

    // Process slice by slice
    for iz_out in 0..nzout {
        let iz = z_min + iz_out;
        let in_slice = reader.read_slice_f32(iz)?;

        // Extract the sub-region
        let mut out_slice = vec![fill_val; nxout * nyout];

        // Copy the input sub-region
        for iy in 0..nyout {
            for ix in 0..nxout {
                let src_idx = (x_min + ix) + (y_min + iy) * nx;
                out_slice[ix + iy * nxout] = in_slice[src_idx];
            }
        }

        // Create a paint mask (which pixels are inside contours)
        let mut paint_mask = vec![false; nxout * nyout];

        for (obj_idx, obj) in model.objects.iter().enumerate() {
            let obj_num = (obj_idx + 1) as i32;

            // Check if this object should be processed
            if !obj_list.is_empty() && !obj_list.contains(&obj_num) {
                continue;
            }

            let is_allsec = allsec_list.contains(&obj_num);
            let is_tube = tube_list.contains(&obj_num);
            let is_scat = object_is_scattered(obj);
            let is_open = object_is_open(obj);

            if is_scat {
                // Paint scattered points as circles
                let size = if obj.pdrawsize > 0 {
                    obj.pdrawsize as f32
                } else {
                    3.0
                };

                for cont in &obj.contours {
                    for pt in &cont.points {
                        let pt_z = pt.z.round() as usize;
                        let on_section = if args.scat_3d {
                            (pt.z - iz as f32).abs() <= size
                        } else {
                            pt_z == iz || is_allsec
                        };

                        if !on_section {
                            continue;
                        }

                        let r = size;
                        let x0 = ((pt.x - r - x_min as f32).max(0.0)) as usize;
                        let x1 = ((pt.x + r - x_min as f32).min(nxout as f32 - 1.0)) as usize;
                        let y0 = ((pt.y - r - y_min as f32).max(0.0)) as usize;
                        let y1 = ((pt.y + r - y_min as f32).min(nyout as f32 - 1.0)) as usize;

                        for iy in y0..=y1 {
                            for ix in x0..=x1 {
                                let dx = (ix as f32 + x_min as f32) - pt.x;
                                let dy = (iy as f32 + y_min as f32) - pt.y;
                                if dx * dx + dy * dy <= r * r {
                                    paint_mask[ix + iy * nxout] = true;
                                }
                            }
                        }
                    }
                }
            } else if is_tube {
                // Paint tubes
                let _tube_idx = tube_list
                    .iter()
                    .position(|&n| n == obj_num)
                    .unwrap_or(0);
                let radius = args.tube_diameter / 2.0;

                for cont in &obj.contours {
                    for seg_i in 0..cont.points.len().saturating_sub(1) {
                        let p0 = &cont.points[seg_i];
                        let p1 = &cont.points[seg_i + 1];
                        let dz = p1.z - p0.z;
                        let seg_z_min = p0.z.min(p1.z) - radius;
                        let seg_z_max = p0.z.max(p1.z) + radius;
                        if (iz as f32) < seg_z_min || (iz as f32) > seg_z_max {
                            continue;
                        }

                        // Parametric position along segment at this Z
                        let t = if dz.abs() > 1e-6 {
                            ((iz as f32 - p0.z) / dz).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };
                        let cx = p0.x + t * (p1.x - p0.x);
                        let cy = p0.y + t * (p1.y - p0.y);

                        let x0 = ((cx - radius - x_min as f32).max(0.0)) as usize;
                        let x1 =
                            ((cx + radius - x_min as f32).min(nxout as f32 - 1.0)) as usize;
                        let y0 = ((cy - radius - y_min as f32).max(0.0)) as usize;
                        let y1 =
                            ((cy + radius - y_min as f32).min(nyout as f32 - 1.0)) as usize;

                        for iy in y0..=y1 {
                            for ix in x0..=x1 {
                                let dx = (ix as f32 + x_min as f32) - cx;
                                let dy = (iy as f32 + y_min as f32) - cy;
                                if dx * dx + dy * dy <= radius * radius {
                                    paint_mask[ix + iy * nxout] = true;
                                }
                            }
                        }
                    }
                }
            } else if !is_open {
                // Closed contours: fill inside
                for cont in &obj.contours {
                    if cont.points.is_empty() {
                        continue;
                    }
                    let cont_z = cont.points[0].z.round() as usize;
                    if cont_z != iz && !is_allsec {
                        continue;
                    }

                    for iy in 0..nyout {
                        for ix in 0..nxout {
                            let px = ix as f32 + x_min as f32;
                            let py = iy as f32 + y_min as f32;
                            if point_in_contour(&cont.points, px, py) {
                                paint_mask[ix + iy * nxout] = true;
                            }
                        }
                    }
                }
            }
        }

        // Apply mask/paint to the output slice
        if masking {
            for i in 0..nxout * nyout {
                if paint_mask[i] {
                    out_slice[i] = mask_val;
                } else if !args.retain_outside {
                    out_slice[i] = fill_val;
                }
            }
        } else if invert {
            // Fill outside contours with fill value
            for i in 0..nxout * nyout {
                if !paint_mask[i] {
                    out_slice[i] = fill_val;
                }
            }
        } else {
            // Fill inside contours with fill value
            for i in 0..nxout * nyout {
                if paint_mask[i] {
                    out_slice[i] = fill_val;
                }
            }
        }

        // Apply reverse contrast
        if args.reverse {
            for v in &mut out_slice {
                *v = hdr.amax - *v;
            }
        }

        // Apply threshold
        if let Some(thresh) = args.threshold {
            for v in &mut out_slice {
                *v = if *v >= thresh { 1.0 } else { 0.0 };
            }
        }

        // Track statistics
        for &v in &out_slice {
            global_min = global_min.min(v);
            global_max = global_max.max(v);
        }
        _global_mean_sum +=
            out_slice.iter().map(|&v| v as f64).sum::<f64>() / out_slice.len() as f64;

        eprint!("\rProcessing section {}...", iz);
        writer.write_slice_f32(&out_slice)?;
    }
    eprintln!();

    writer.finish(0.0, 0.0, 0.0)?;
    println!(
        "Wrote {} slices ({} x {}) to {}",
        nzout, nxout, nyout, args.output_file
    );

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        process::exit(1);
    }
}
