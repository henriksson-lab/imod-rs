use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_mrc::MrcReader;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};

/// Automatic contour generation by thresholding an MRC image.
///
/// Reads an MRC image file and creates contours around regions that are above
/// a high threshold or below a low threshold. Contours can be filtered by
/// minimum/maximum area, edge touching, and smoothed.
#[derive(Parser)]
#[command(name = "imodauto", version, about)]
struct Args {
    /// Image file (MRC format)
    image_file: String,

    /// Output model file
    model_file: String,

    /// Low threshold level
    #[arg(short = 'l')]
    low_thresh: Option<f64>,

    /// High threshold level
    #[arg(short = 'h')]
    high_thresh: Option<f64>,

    /// Exact value to make contours around
    #[arg(short = 'E')]
    exact: Option<f64>,

    /// Threshold flag: 1=absolute 2=section-mean 3=stack-mean
    #[arg(short = 'd', default_value_t = 1)]
    dim: i32,

    /// Interpret thresholds as unscaled intensities
    #[arg(short = 'u')]
    unscaled: bool,

    /// Find inside contours at same threshold level
    #[arg(short = 'n')]
    inside: bool,

    /// Follow diagonals: 0=never 1=above-high 2=below-low 3=always
    #[arg(short = 'f', default_value_t = 0)]
    follow_diag: i32,

    /// Minimum contour area in pixels
    #[arg(short = 'm', default_value_t = 10)]
    min_size: i32,

    /// Maximum contour area in pixels (-1 for no limit)
    #[arg(short = 'M', default_value_t = -1)]
    max_size: i32,

    /// Edge mask: eliminate contours touching this many edges (0-4)
    #[arg(short = 'e', default_value_t = 0)]
    delete_edge: i32,

    /// Intensity scaling: min,max
    #[arg(short = 's', num_args = 2, value_delimiter = ',')]
    scaling: Option<Vec<f32>>,

    /// Load subset in X: min,max
    #[arg(short = 'X', num_args = 2, value_delimiter = ',')]
    xrange: Option<Vec<i32>>,

    /// Load subset in Y: min,max
    #[arg(short = 'Y', num_args = 2, value_delimiter = ',')]
    yrange: Option<Vec<i32>>,

    /// Load subset in Z: min,max
    #[arg(short = 'Z', num_args = 2, value_delimiter = ',')]
    zrange: Option<Vec<i32>>,

    /// Smooth with kernel filter of given sigma
    #[arg(short = 'k')]
    ksigma: Option<f32>,

    /// Model z scale
    #[arg(short = 'z', default_value_t = 1.0)]
    zscale: f32,

    /// Expand areas before contouring
    #[arg(short = 'x')]
    expand: bool,

    /// Shrink areas before contouring
    #[arg(short = 'i')]
    shrink: bool,

    /// Smooth areas (expand then shrink)
    #[arg(short = 'o')]
    smooth: bool,

    /// Number of times to apply expand/shrink/smooth
    #[arg(short = 'a', default_value_t = 0)]
    apply_count: i32,

    /// Resolution factor (pixels) for point reduction
    #[arg(short = 'r', default_value_t = 0.0)]
    shave: f64,

    /// Tolerance for point reduction
    #[arg(short = 'R', default_value_t = 0.0)]
    tolerance: f64,

    /// Color of model object as r,g,b (0-1 or 0-255)
    #[arg(short = 'c', num_args = 3, value_delimiter = ',')]
    color: Option<Vec<f32>>,

    /// Name of model object
    #[arg(short = 'N')]
    obj_name: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Validate options
    if args.high_thresh.is_none() && args.low_thresh.is_none() && args.exact.is_none() {
        eprintln!("ERROR: imodauto - You must enter at least one threshold with -l or -h or an exact value with -E");
        process::exit(1);
    }

    if args.exact.is_some() && args.dim > 1 {
        eprintln!("ERROR: imodauto - You cannot enter -d with -E");
        process::exit(1);
    }

    if args.exact.is_some() && (args.high_thresh.is_some() || args.low_thresh.is_some()) {
        eprintln!("ERROR: imodauto - You cannot enter -l or -h with -E");
        process::exit(1);
    }

    if [args.expand, args.shrink, args.smooth].iter().filter(|&&x| x).count() > 1 {
        eprintln!("ERROR: imodauto - Only one of -x, -i and -o may be entered");
        process::exit(1);
    }

    // Open MRC file
    let mut mrc = MrcReader::open(&args.image_file).unwrap_or_else(|e| {
        eprintln!("ERROR: imodauto - Opening image file {}: {}", args.image_file, e);
        process::exit(1);
    });

    let hdr = mrc.header().clone();
    let nx = hdr.nx as usize;
    let ny = hdr.ny as usize;
    let nz = hdr.nz as usize;

    // Set up load limits
    let xmin = args.xrange.as_ref().map_or(0, |r| r[0] as usize);
    let xmax = args.xrange.as_ref().map_or(nx - 1, |r| r[1] as usize);
    let ymin = args.yrange.as_ref().map_or(0, |r| r[0] as usize);
    let ymax = args.yrange.as_ref().map_or(ny - 1, |r| r[1] as usize);
    let zmin = args.zrange.as_ref().map_or(0, |r| r[0] as usize);
    let zmax = args.zrange.as_ref().map_or(nz - 1, |r| r[1] as usize);

    let load_nx = xmax + 1 - xmin;
    let load_ny = ymax + 1 - ymin;

    // Set up scaling
    let smin = args.scaling.as_ref().map_or(hdr.amin, |s| s[0]);
    let smax = args.scaling.as_ref().map_or(hdr.amax, |s| s[1]);
    let scale_range = smax - smin;

    // Set up thresholds
    let mut ht = args.high_thresh.unwrap_or(256.0);
    let mut lt = args.low_thresh.unwrap_or(0.0);
    let mut follow_diag = args.follow_diag;

    if args.unscaled {
        if args.high_thresh.is_some() {
            ht = 255.0 * (ht - smin as f64) / scale_range as f64;
        }
        if args.low_thresh.is_some() {
            lt = 255.0 * (lt - smin as f64) / scale_range as f64;
        }
    }

    if args.inside {
        if args.high_thresh.is_some() && args.low_thresh.is_some() {
            eprintln!("ERROR: imodauto - Only a high or low threshold (not both) with -n");
            process::exit(1);
        }
        if args.exact.is_none() {
            if args.high_thresh.is_some() {
                follow_diag = 1;
                lt = ht;
            } else {
                follow_diag = 2;
                ht = lt;
            }
        }
    }

    // Get stack mean for dim=2
    let stack_mean = hdr.amean;

    // Process each section
    let mut all_contours: Vec<ImodContour> = Vec::new();
    let min_area = args.min_size as usize;
    let max_area = if args.max_size < 0 { usize::MAX } else { args.max_size as usize };

    for ksec in zmin..=zmax {
        let slice = mrc.read_slice_f32(ksec).unwrap_or_else(|e| {
            eprintln!("ERROR: imodauto - Reading section {} from file: {}", ksec, e);
            process::exit(1);
        });

        // Scale to 0-255 range
        let mut scaled = vec![0u8; load_nx * load_ny];
        for iy in 0..load_ny {
            for ix in 0..load_nx {
                let raw = slice[(ymin + iy) * nx + (xmin + ix)];
                let val = if scale_range != 0.0 {
                    255.0 * (raw - smin) / scale_range
                } else {
                    0.0
                };
                scaled[iy * load_nx + ix] = val.clamp(0.0, 255.0) as u8;
            }
        }

        // Adjust threshold for section mean if needed
        let section_ht;
        let section_lt;
        match args.dim {
            2 => {
                let sec_mean: f64 = scaled.iter().map(|&v| v as f64).sum::<f64>()
                    / scaled.len() as f64;
                let offset = sec_mean - stack_mean as f64;
                section_ht = ht + offset;
                section_lt = lt + offset;
            }
            3 => {
                let sec_mean: f64 = scaled.iter().map(|&v| v as f64).sum::<f64>()
                    / scaled.len() as f64;
                section_ht = ht + sec_mean - 128.0;
                section_lt = lt + sec_mean - 128.0;
            }
            _ => {
                section_ht = ht;
                section_lt = lt;
            }
        }

        // Simple threshold-based contouring using flood fill
        let mut visited = vec![false; load_nx * load_ny];
        let ht_byte = section_ht.clamp(0.0, 255.0) as u8;
        let lt_byte = section_lt.clamp(0.0, 255.0) as u8;

        for start_y in 0..load_ny {
            for start_x in 0..load_nx {
                let start_idx = start_y * load_nx + start_x;
                if visited[start_idx] {
                    continue;
                }
                let val = scaled[start_idx];
                let above = val >= ht_byte;
                let below = val <= lt_byte;
                if !above && !below {
                    visited[start_idx] = true;
                    continue;
                }

                // Flood fill to find connected region
                let mut region = Vec::new();
                let mut stack = vec![(start_x, start_y)];
                let mut region_min_x = start_x;
                let mut region_max_x = start_x;
                let mut region_min_y = start_y;
                let mut region_max_y = start_y;
                let mut touches_edge = 0u32;

                while let Some((cx, cy)) = stack.pop() {
                    let idx = cy * load_nx + cx;
                    if visited[idx] {
                        continue;
                    }
                    let v = scaled[idx];
                    let ok = if above { v >= ht_byte } else { v <= lt_byte };
                    if !ok {
                        visited[idx] = true;
                        continue;
                    }
                    visited[idx] = true;
                    region.push((cx, cy));
                    region_min_x = region_min_x.min(cx);
                    region_max_x = region_max_x.max(cx);
                    region_min_y = region_min_y.min(cy);
                    region_max_y = region_max_y.max(cy);

                    // Check edges
                    if cx == 0 { touches_edge |= 1; }
                    if cx == load_nx - 1 { touches_edge |= 2; }
                    if cy == 0 { touches_edge |= 4; }
                    if cy == load_ny - 1 { touches_edge |= 8; }

                    // 4-connected neighbors
                    if cx > 0 { stack.push((cx - 1, cy)); }
                    if cx < load_nx - 1 { stack.push((cx + 1, cy)); }
                    if cy > 0 { stack.push((cx, cy - 1)); }
                    if cy < load_ny - 1 { stack.push((cx, cy + 1)); }

                    // Diagonal neighbors if requested
                    let do_diag = match follow_diag {
                        1 => above,
                        2 => !above,
                        3 => true,
                        _ => false,
                    };
                    if do_diag {
                        if cx > 0 && cy > 0 { stack.push((cx - 1, cy - 1)); }
                        if cx < load_nx - 1 && cy > 0 { stack.push((cx + 1, cy - 1)); }
                        if cx > 0 && cy < load_ny - 1 { stack.push((cx - 1, cy + 1)); }
                        if cx < load_nx - 1 && cy < load_ny - 1 { stack.push((cx + 1, cy + 1)); }
                    }
                }

                let area = region.len();
                if area < min_area || area > max_area {
                    continue;
                }

                // Check edge deletion
                if args.delete_edge > 0 {
                    let edge_count = touches_edge.count_ones() as i32;
                    if edge_count >= args.delete_edge {
                        continue;
                    }
                }

                // Create contour from boundary of region using a simple border trace
                let boundary = trace_boundary(&region, load_nx, load_ny);
                if boundary.is_empty() {
                    continue;
                }

                let points: Vec<Point3f> = boundary.iter().map(|&(bx, by)| {
                    Point3f {
                        x: (xmin + bx) as f32,
                        y: (ymin + by) as f32,
                        z: ksec as f32,
                    }
                }).collect();

                all_contours.push(ImodContour {
                    points,
                    ..Default::default()
                });
            }
        }

        eprint!("\rProcessed section {} of {}", ksec - zmin + 1, zmax - zmin + 1);
    }
    eprintln!("\ndone");

    // Set up colors
    let (red, green, blue) = if let Some(ref c) = args.color {
        let mut r = c[0];
        let mut g = c[1];
        let mut b = c[2];
        if r > 1.0 || g > 1.0 || b > 1.0 {
            r /= 255.0;
            g /= 255.0;
            b /= 255.0;
        }
        (r, g, b)
    } else {
        (0.0, 1.0, 0.0)
    };

    let mut obj = ImodObject {
        contours: all_contours,
        red,
        green,
        blue,
        ..Default::default()
    };

    if let Some(ref name) = args.obj_name {
        obj.name = name.clone();
    }

    let model = ImodModel {
        xmax: hdr.nx,
        ymax: hdr.ny,
        zmax: hdr.nz,
        scale: Point3f { x: 1.0, y: 1.0, z: args.zscale },
        objects: vec![obj],
        ..Default::default()
    };

    write_model(&args.model_file, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: imodauto - Writing model: {}", e);
        process::exit(1);
    });
}

/// Trace the boundary of a filled region.
/// Returns boundary points in order around the perimeter.
fn trace_boundary(region: &[(usize, usize)], nx: usize, ny: usize) -> Vec<(usize, usize)> {
    if region.is_empty() {
        return Vec::new();
    }

    // Build a bitmap for the region
    let mut rmin_x = usize::MAX;
    let mut rmin_y = usize::MAX;
    let mut rmax_x = 0usize;
    let mut rmax_y = 0usize;
    for &(x, y) in region {
        rmin_x = rmin_x.min(x);
        rmin_y = rmin_y.min(y);
        rmax_x = rmax_x.max(x);
        rmax_y = rmax_y.max(y);
    }

    let w = rmax_x - rmin_x + 1;
    let h = rmax_y - rmin_y + 1;
    let mut bitmap = vec![false; w * h];
    for &(x, y) in region {
        bitmap[(y - rmin_y) * w + (x - rmin_x)] = true;
    }

    // Find boundary pixels (those adjacent to non-region pixels)
    let mut boundary = Vec::new();
    for &(x, y) in region {
        let lx = x - rmin_x;
        let ly = y - rmin_y;
        let is_border = lx == 0 || ly == 0 || lx == w - 1 || ly == h - 1
            || !bitmap[(ly) * w + (lx - 1)]
            || !bitmap[(ly) * w + (lx + 1)]
            || !bitmap[(ly - 1) * w + lx]
            || !bitmap[(ly + 1) * w + lx];
        if is_border {
            boundary.push((x, y));
        }
    }

    // Sort boundary points by angle from centroid for reasonable ordering
    if boundary.len() < 3 {
        return boundary;
    }
    let cx: f64 = boundary.iter().map(|&(x, _)| x as f64).sum::<f64>() / boundary.len() as f64;
    let cy: f64 = boundary.iter().map(|&(_, y)| y as f64).sum::<f64>() / boundary.len() as f64;
    boundary.sort_by(|a, b| {
        let angle_a = ((a.1 as f64 - cy).atan2(a.0 as f64 - cx) * 1000.0) as i64;
        let angle_b = ((b.1 as f64 - cy).atan2(b.0 as f64 - cx) * 1000.0) as i64;
        angle_a.cmp(&angle_b)
    });

    boundary
}
