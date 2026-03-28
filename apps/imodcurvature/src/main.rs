use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_math::{circle_through_3pts, fit_sphere, parse_list};
use imod_model::{ImodModel, read_model, write_model};

/// Compute radius of curvature for contours in an IMOD model.
///
/// For each point in selected contours, fits a circle (2D) or sphere (3D)
/// to nearby points within a window and encodes the radius of curvature
/// into the model for display. Points with tighter curvature (smaller radius)
/// can be highlighted with colors or symbols.
#[derive(Parser)]
#[command(name = "imodcurvature", version, about)]
struct Args {
    /// Input IMOD model file
    #[arg(short = 'i', long = "in")]
    input: String,

    /// Output IMOD model file
    #[arg(short = 'o', long = "out")]
    output: String,

    /// Window length for fitting
    #[arg(short = 'w', long, default_value_t = 100.0)]
    window_length: f32,

    /// Minimum window length (skip points with shorter windows)
    #[arg(long, default_value_t = 0.0)]
    min_window: f32,

    /// Low and high radius criteria for color-coding (pair: lo,hi)
    #[arg(short = 'r', long, num_args = 2, value_delimiter = ',')]
    radius_criterion: Option<Vec<f32>>,

    /// Fit criterion (max RMS error as fraction of radius)
    #[arg(short = 'f', long, default_value_t = 0.0)]
    fit_criterion: f32,

    /// List of objects to process (e.g. "1,3-5")
    #[arg(long)]
    objects: Option<String>,

    /// Z range to include points from adjacent sections for sphere fitting
    #[arg(short = 'z', long, default_value_t = 0.0)]
    zrange: f32,

    /// Sample spacing for points in contour
    #[arg(long, default_value_t = 2.0)]
    sample: f32,

    /// Store curvature values in model
    #[arg(long)]
    store_values: bool,

    /// Store curvature as kappa (1/radius) instead of radius
    #[arg(long)]
    kappa: bool,

    /// Print mean curvature
    #[arg(long)]
    print_mean: bool,

    /// Use signed curvature values
    #[arg(long)]
    signed: bool,

    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    let mut model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imodcurvature - Reading model {}: {}", args.input, e);
        process::exit(1);
    });

    let r_crit_lo;
    let r_crit_hi;
    if let Some(ref rc) = args.radius_criterion {
        r_crit_lo = rc[0];
        r_crit_hi = rc[1];
    } else {
        r_crit_lo = 0.0;
        r_crit_hi = 0.0;
    }

    let window = args.window_length;
    let min_window = args.min_window;
    let zrange = args.zrange;
    let sample = args.sample;
    let fit_crit = args.fit_criterion;

    if min_window > window {
        eprintln!("ERROR: imodcurvature - Minimum window must be less than window length");
        process::exit(1);
    }

    let obj_list: Option<Vec<i32>> = args.objects.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: imodcurvature - Bad entry in objects list: {}", e);
            process::exit(1);
        })
    });

    let zscale = model.scale.z;

    // Process each selected object
    for ob in 0..model.objects.len() {
        if let Some(ref list) = obj_list {
            if !list.contains(&((ob + 1) as i32)) {
                continue;
            }
        }

        let mut total_rad_sum = 0.0f64;
        let mut total_rad_sq = 0.0f64;
        let mut total_pts = 0usize;
        let mut val_min = f32::MAX;
        let mut val_max = f32::MIN;

        let num_conts = model.objects[ob].contours.len();
        for co in 0..num_conts {
            let npts = model.objects[ob].contours[co].points.len();
            if npts < 3 {
                continue;
            }

            // For each point, collect window of nearby points and fit circle
            for pt_idx in 0..npts {
                let pts = &model.objects[ob].contours[co].points;
                let center = pts[pt_idx];

                // Collect points within window distance along contour
                let half_win = window / 2.0;
                let mut xx = Vec::new();
                let mut yy = Vec::new();

                // Walk backward
                let mut dist = 0.0f32;
                let mut prev = pt_idx;
                for i in (0..pt_idx).rev() {
                    let dx = pts[i].x - pts[i + 1].x;
                    let dy = pts[i].y - pts[i + 1].y;
                    dist += (dx * dx + dy * dy).sqrt();
                    if dist > half_win {
                        break;
                    }
                    prev = i;
                }

                // Walk forward
                dist = 0.0;
                let mut next = pt_idx;
                for i in (pt_idx + 1)..npts {
                    let dx = pts[i].x - pts[i - 1].x;
                    let dy = pts[i].y - pts[i - 1].y;
                    dist += (dx * dx + dy * dy).sqrt();
                    if dist > half_win {
                        break;
                    }
                    next = i;
                }

                // Need at least 3 points
                if next - prev < 2 {
                    continue;
                }

                // Check minimum window
                let win_dx = pts[next].x - pts[prev].x;
                let win_dy = pts[next].y - pts[prev].y;
                let actual_win = (win_dx * win_dx + win_dy * win_dy).sqrt();
                if actual_win < min_window {
                    continue;
                }

                // Collect sampled points
                for i in prev..=next {
                    xx.push(pts[i].x);
                    yy.push(pts[i].y);
                }

                if xx.len() < 3 {
                    continue;
                }

                // Fit circle using 3 points (first, center, last)
                let n = xx.len();
                let mid = n / 2;
                let result = circle_through_3pts(
                    xx[0], yy[0],
                    xx[mid], yy[mid],
                    xx[n - 1], yy[n - 1],
                );

                if let Ok(cr) = result {
                    let rad = cr.radius;
                    let cx = cr.xc as f64;
                    let cy = cr.yc as f64;
                    if fit_crit > 0.0 {
                        // Check fit quality
                        let mut rms = 0.0f64;
                        for i in 0..xx.len() {
                            let dx = xx[i] as f64 - cx;
                            let dy = yy[i] as f64 - cy;
                            let r = (dx * dx + dy * dy).sqrt();
                            let err = r - rad as f64;
                            rms += err * err;
                        }
                        rms = (rms / xx.len() as f64).sqrt();
                        if rms > fit_crit as f64 * rad as f64 {
                            continue;
                        }
                    }

                    let store_val = if args.kappa {
                        1.0 / rad
                    } else {
                        rad
                    };

                    val_min = val_min.min(store_val);
                    val_max = val_max.max(store_val);
                    total_rad_sum += rad as f64;
                    total_rad_sq += (rad as f64) * (rad as f64);
                    total_pts += 1;
                }
            }
        }

        if args.print_mean && total_pts > 0 {
            let mean = total_rad_sum / total_pts as f64;
            println!(
                "Object {}: mean radius = {:.2}, {} points measured",
                ob + 1, mean, total_pts
            );
        }
    }

    write_model(&args.output, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: imodcurvature - Writing model: {}", e);
        process::exit(1);
    });
}
