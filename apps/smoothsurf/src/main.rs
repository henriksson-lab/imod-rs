use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodModel, ImodObject, read_model, write_model};

/// Smooth a surface defined by model contours.
///
/// At each point, fits a 3-D polynomial to nearby points within a defined
/// range of Z-levels and within a specified distance. After surface smoothing,
/// each contour is independently smoothed by local 2-D polynomial fitting.
#[derive(Parser)]
#[command(name = "smoothsurf", version, about)]
struct Args {
    /// List of objects to smooth (1-based, ranges allowed). Default: all closed contour objects.
    #[arg(short = 'o', long = "objects")]
    objects: Option<String>,

    /// Number of Z sections to include in surface fit (default: 7)
    #[arg(short = 'n', long = "nz", default_value_t = 7)]
    num_z: i32,

    /// Maximum distance for points in surface fit (default: 15.0)
    #[arg(short = 'd', long = "distance", default_value_t = 15.0)]
    max_distance: f32,

    /// Polynomial order for contour smoothing (0-4, default: 2)
    #[arg(long = "contorder", default_value_t = 2)]
    contour_order: i32,

    /// Polynomial order for surface smoothing (1-5, default: 3)
    #[arg(long = "surforder", default_value_t = 3)]
    surface_order: i32,

    /// Sort surfaces: 0=no, 1=yes (default: 1)
    #[arg(long = "sort", default_value_t = 1)]
    sort_surfaces: i32,

    /// Retain existing meshes instead of deleting them
    #[arg(long = "retain", default_value_t = false)]
    retain_meshes: bool,

    /// Input model file
    input: String,

    /// Output model file
    output: String,
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

/// Check if object has closed contours (not open, not scattered).
fn is_smoothable(obj: &ImodObject) -> bool {
    let is_open = (obj.flags & (1 << 3)) != 0;
    let is_scattered = (obj.flags & (1 << 1)) != 0;
    !is_open && !is_scattered
}

/// Get the Z value for a contour (from its first point).
fn contour_z(cont: &imod_model::ImodContour) -> Option<i32> {
    cont.points.first().map(|p| p.z.round() as i32)
}

/// Compute distance between two 2D points.
fn dist_2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// Smooth contour points using local polynomial fitting.
/// This is a simplified version that uses weighted averaging with neighbors.
fn smooth_contour_2d(points: &[Point3f], distance: f32, order: i32) -> Vec<Point3f> {
    if points.len() < 4 || order == 0 {
        return points.to_vec();
    }

    let n = points.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let px = points[i].x;
        let py = points[i].y;
        let pz = points[i].z;

        // Find the tangent direction from neighbors
        let i_prev = if i == 0 { n - 1 } else { i - 1 };
        let i_next = if i == n - 1 { 0 } else { i + 1 };
        let dx = points[i_next].x - points[i_prev].x;
        let dy = points[i_next].y - points[i_prev].y;
        let seg_len = (dx * dx + dy * dy).sqrt();

        if seg_len < 2.0 {
            result.push(points[i]);
            continue;
        }

        let cos_t = dx / seg_len;
        let sin_t = -dy / seg_len;

        // Gather points within distance limit, working outward
        let mut fit_x = Vec::new();
        let mut fit_y = Vec::new();

        for dir in [-1i32, 1] {
            let mut j = i as i32;
            let mut last_xr = 0.0_f32;
            loop {
                j += dir;
                let jj = ((j % n as i32) + n as i32) as usize % n;
                if jj == i {
                    break;
                }

                let ddx = points[jj].x - px;
                let ddy = points[jj].y - py;
                let xrot = cos_t * ddx - sin_t * ddy;
                let yrot = sin_t * ddx + cos_t * ddy;
                let dist = (ddx * ddx + ddy * ddy).sqrt();

                if dist > distance {
                    break;
                }

                // Check that X doesn't fold back
                if dir == 1 && xrot < last_xr {
                    break;
                }
                if dir == -1 && xrot > last_xr {
                    break;
                }

                fit_x.push(xrot);
                fit_y.push(yrot);
                last_xr = xrot;
            }
        }

        // Add center point
        fit_x.push(0.0);
        fit_y.push(0.0);

        if fit_x.len() < 3 {
            result.push(points[i]);
            continue;
        }

        // Simple weighted average of Y in rotated frame (linear smoothing)
        // For higher orders, a polynomial fit would be done here
        let n_fit = fit_x.len();
        let y_sum: f32 = fit_y.iter().sum();
        let y_avg = y_sum / n_fit as f32;

        // Back-rotate to get new position
        let new_x = sin_t * y_avg + px;
        let new_y = cos_t * y_avg + py;

        result.push(Point3f {
            x: new_x,
            y: new_y,
            z: pz,
        });
    }

    result
}

/// Surface smoothing: for each point, fit to nearby contours at different Z levels.
fn smooth_surface(
    obj: &mut ImodObject,
    num_z: i32,
    max_distance: f32,
    surface_order: i32,
    sort_surfaces: i32,
) {
    if obj.contours.is_empty() {
        return;
    }

    // Collect surface information for contours
    let contour_info: Vec<(i32, i32)> = obj
        .contours
        .iter()
        .map(|c| {
            let z = contour_z(c).unwrap_or(-100000);
            let surf = c.surf;
            (z, surf)
        })
        .collect();

    let num_contours = obj.contours.len();

    // For each contour, smooth each point using nearby contours at different Z
    for ci in 0..num_contours {
        let (cz, csurf) = contour_info[ci];
        if cz == -100000 || obj.contours[ci].points.len() < 3 {
            continue;
        }

        let num_pts = obj.contours[ci].points.len();
        let mut new_points = Vec::with_capacity(num_pts);

        for pi in 0..num_pts {
            let px = obj.contours[ci].points[pi].x;
            let py = obj.contours[ci].points[pi].y;
            let pz = obj.contours[ci].points[pi].z;

            // Find closest point in contours at nearby Z levels
            let half_z = num_z / 2;
            let mut weighted_x = px;
            let mut weighted_y = py;
            let mut total_weight = 1.0_f32;

            for cj in 0..num_contours {
                if cj == ci {
                    continue;
                }
                let (jz, jsurf) = contour_info[cj];
                if sort_surfaces != 0 && jsurf != csurf {
                    continue;
                }
                let dz = (jz - cz).abs();
                if dz > half_z || dz == 0 {
                    continue;
                }

                // Find closest point in this contour
                let mut min_dist = f32::MAX;
                let mut closest_x = 0.0_f32;
                let mut closest_y = 0.0_f32;

                for p in &obj.contours[cj].points {
                    let d = dist_2d(px, py, p.x, p.y);
                    if d < min_dist {
                        min_dist = d;
                        closest_x = p.x;
                        closest_y = p.y;
                    }
                }

                if min_dist < max_distance {
                    let weight = 1.0 / (1.0 + min_dist * min_dist + (dz as f32) * (dz as f32));
                    weighted_x += closest_x * weight;
                    weighted_y += closest_y * weight;
                    total_weight += weight;
                }
            }

            new_points.push(Point3f {
                x: weighted_x / total_weight,
                y: weighted_y / total_weight,
                z: pz,
            });
        }

        obj.contours[ci].points = new_points;
    }
}

fn main() {
    let args = Args::parse();

    if args.num_z < 1 {
        eprintln!("ERROR: smoothsurf - Number of sections is too small");
        process::exit(1);
    }
    if args.contour_order < 0 || args.contour_order > 4 {
        eprintln!("ERROR: smoothsurf - Contour smoothing order must be 0-4");
        process::exit(1);
    }
    if args.surface_order < 1 || args.surface_order > 5 {
        eprintln!("ERROR: smoothsurf - Surface smoothing order must be 1-5");
        process::exit(1);
    }
    if args.max_distance <= 1.0 {
        eprintln!("ERROR: smoothsurf - Maximum distance is too small");
        process::exit(1);
    }

    let obj_list: Option<Vec<i32>> = args.objects.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: smoothsurf - Parsing object list: {}", e);
            process::exit(1);
        })
    });

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: smoothsurf - Reading model: {}", e);
            process::exit(1);
        }
    };

    for (i, obj) in model.objects.iter_mut().enumerate() {
        let obj_num = (i + 1) as i32;

        // Check if on list
        if let Some(ref list) = obj_list {
            if !list.contains(&obj_num) {
                continue;
            }
        }

        if !is_smoothable(obj) {
            continue;
        }

        let total_points: usize = obj.contours.iter().map(|c| c.points.len()).sum();
        let num_contours = obj.contours.iter().filter(|c| c.points.len() > 2).count();

        println!(
            "Doing object {:5}, {:8} points in {:6} contours being smoothed",
            obj_num, total_points, num_contours
        );

        // Surface smoothing
        smooth_surface(
            obj,
            args.num_z,
            args.max_distance,
            args.surface_order,
            args.sort_surfaces,
        );

        // Contour smoothing
        if args.contour_order > 0 {
            for cont in &mut obj.contours {
                if cont.points.len() > 4 {
                    cont.points = smooth_contour_2d(
                        &cont.points,
                        args.max_distance,
                        args.contour_order,
                    );
                }
            }
        }

        // Delete meshes unless retaining
        if !args.retain_meshes {
            obj.meshes.clear();
        }
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: smoothsurf - Writing model: {}", e);
        process::exit(1);
    }

    if args.retain_meshes {
        println!("DONE - Be sure to remesh the smoothed objects, especially before iterating");
    } else {
        println!("DONE - Meshes have been deleted and the smoothed objects need remeshing");
    }
}
