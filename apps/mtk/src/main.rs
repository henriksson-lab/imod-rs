use std::f64::consts::PI;
use std::io::Write;
use std::process;

use clap::Parser;
use imod_math::parse_list;
use imod_model::{ImodModel, ImodObject, read_model};
use rand::Rng;

/// MTK - 3D spatial analysis of closest approach distances between objects.
///
/// Analyzes the distribution of distances of closest approach between objects
/// (lines, scattered points) in 3D IMOD models.  Computes density graphs of
/// neighbors as a function of radial distance, accounting for the geometry of
/// the objects (point-to-point, line-to-line, point-to-line, etc).
/// Supports type shuffling and random point generation for significance testing.
#[derive(Parser)]
#[command(name = "mtk", version, about)]
struct Args {
    /// IMOD model file
    #[arg(required = true)]
    model_file: String,

    /// Radial bin width in model units
    #[arg(short = 'd', long, default_value_t = 0.005)]
    bin_width: f64,

    /// Number of radial bins
    #[arg(short = 'n', long, default_value_t = 50)]
    num_bins: usize,

    /// Reference object types (comma-separated, or "all")
    #[arg(short = 'r', long, default_value = "all")]
    ref_types: String,

    /// Neighbor object types (comma-separated, or "all")
    #[arg(short = 't', long, default_value = "all")]
    neigh_types: String,

    /// Power for radial weighting in volume shell computation
    #[arg(long, default_value_t = 1.0)]
    power: f64,

    /// Number of points to fit lines over (for line objects)
    #[arg(long, default_value_t = 2)]
    fit_points: usize,

    /// Z scale factor (0 to read from model header)
    #[arg(long, default_value_t = 0.0)]
    z_scale: f64,

    /// X/Y scale in microns per pixel (0 to read from model header)
    #[arg(long, default_value_t = 0.0)]
    xy_scale: f64,

    /// Starting Z to include (0 for all)
    #[arg(long, default_value_t = 0.0)]
    z_start: f64,

    /// Ending Z to include (0 for all)
    #[arg(long, default_value_t = 0.0)]
    z_end: f64,

    /// Number of random shuffles for significance testing
    #[arg(short = 's', long, default_value_t = 0)]
    num_shuffles: usize,

    /// Sampling length for lines (0 = use closest approach to whole line)
    #[arg(long, default_value_t = 0.0)]
    sample_len: f64,

    /// Measure to closest point on segment instead of segment start
    #[arg(long, default_value_t = false)]
    close_seg: bool,

    /// Count only nearest neighbor for each reference object
    #[arg(long, default_value_t = false)]
    nearest_only: bool,

    /// Output file (default: stdout)
    #[arg(short = 'o', long)]
    output: Option<String>,
}

/// Determines whether an IMOD object contains scattered points, open contours (lines),
/// or closed contours.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ObjGeom {
    Scattered,  // scattered point object
    OpenLine,   // open contour = line segments
    Closed,     // closed contour
}

fn object_geom(obj: &ImodObject) -> ObjGeom {
    if (obj.flags & (1 << 1)) != 0 {
        ObjGeom::Scattered
    } else if (obj.flags & (1 << 3)) != 0 {
        ObjGeom::OpenLine
    } else {
        ObjGeom::Closed
    }
}

/// A 3D working object: either a set of points from a scattered object,
/// or the points of a single contour from a line object.
#[derive(Clone, Debug)]
struct WorkObj {
    points: Vec<[f64; 3]>,
    obj_type: i32,  // 1-based IMOD object number
    geom: ObjGeom,
}

fn parse_types(s: &str) -> Option<Vec<i32>> {
    if s.eq_ignore_ascii_case("all") {
        None
    } else {
        Some(parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: mtk - bad type list '{}': {}", s, e);
            process::exit(1);
        }))
    }
}

fn type_matches(obj_type: i32, filter: &Option<Vec<i32>>) -> bool {
    match filter {
        None => true,
        Some(types) => types.contains(&obj_type),
    }
}

/// Extract working objects from the model. Each contour in a line object becomes
/// one working object; all points in a scattered object form a single working object.
fn extract_work_objects(
    model: &ImodModel,
    xy_scale: f64,
    z_scale: f64,
    z_start: f64,
    z_end: f64,
) -> Vec<WorkObj> {
    let mut wobjs = Vec::new();
    let use_z_limits = z_start != 0.0 || z_end != 0.0;

    for (oi, obj) in model.objects.iter().enumerate() {
        let obj_type = (oi + 1) as i32;
        let geom = object_geom(obj);

        match geom {
            ObjGeom::Scattered => {
                // All points from all contours form one working object per point
                for cont in &obj.contours {
                    for pt in &cont.points {
                        let z = pt.z as f64 * z_scale;
                        if use_z_limits && (z < z_start || z > z_end) {
                            continue;
                        }
                        wobjs.push(WorkObj {
                            points: vec![[pt.x as f64 * xy_scale, pt.y as f64 * xy_scale, z]],
                            obj_type,
                            geom,
                        });
                    }
                }
            }
            ObjGeom::OpenLine | ObjGeom::Closed => {
                // Each contour is a working object
                for cont in &obj.contours {
                    if cont.points.len() < 2 {
                        continue;
                    }
                    let pts: Vec<[f64; 3]> = cont
                        .points
                        .iter()
                        .map(|p| [p.x as f64 * xy_scale, p.y as f64 * xy_scale, p.z as f64 * z_scale])
                        .collect();

                    // Check Z limits
                    if use_z_limits {
                        let zmin = pts.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
                        let zmax = pts.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
                        if zmax < z_start || zmin > z_end {
                            continue;
                        }
                    }
                    wobjs.push(WorkObj {
                        points: pts,
                        obj_type,
                        geom,
                    });
                }
            }
        }
    }
    wobjs
}

/// Distance of closest approach between two segments (p1->p2) and (p3->p4).
fn segment_segment_dist(p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3], p4: &[f64; 3]) -> f64 {
    let d1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let d2 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];
    let r = [p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]];

    let a = dot3(&d1, &d1);
    let e = dot3(&d2, &d2);
    let f = dot3(&d2, &r);

    let eps = 1.0e-10;

    if a <= eps && e <= eps {
        // Both degenerate to points
        return dist3(p1, p3);
    }
    if a <= eps {
        // First segment degenerates to a point
        let t = (f / e).clamp(0.0, 1.0);
        let closest = [p3[0] + t * d2[0], p3[1] + t * d2[1], p3[2] + t * d2[2]];
        return dist3(p1, &closest);
    }

    let c = dot3(&d1, &r);
    if e <= eps {
        // Second segment degenerates to a point
        let s = (-c / a).clamp(0.0, 1.0);
        let closest = [p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]];
        return dist3(p3, &closest);
    }

    let b = dot3(&d1, &d2);
    let denom = a * e - b * b;

    let mut s = if denom.abs() > eps {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut t = (b * s + f) / e;

    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }

    let closest1 = [
        p1[0] + s * d1[0],
        p1[1] + s * d1[1],
        p1[2] + s * d1[2],
    ];
    let closest2 = [
        p3[0] + t * d2[0],
        p3[1] + t * d2[1],
        p3[2] + t * d2[2],
    ];
    dist3(&closest1, &closest2)
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Point to segment closest distance.
fn point_segment_dist(pt: &[f64; 3], seg_a: &[f64; 3], seg_b: &[f64; 3]) -> f64 {
    let d = [seg_b[0] - seg_a[0], seg_b[1] - seg_a[1], seg_b[2] - seg_a[2]];
    let v = [pt[0] - seg_a[0], pt[1] - seg_a[1], pt[2] - seg_a[2]];
    let d_dot_d = dot3(&d, &d);
    if d_dot_d < 1e-20 {
        return dist3(pt, seg_a);
    }
    let t = (dot3(&v, &d) / d_dot_d).clamp(0.0, 1.0);
    let closest = [
        seg_a[0] + t * d[0],
        seg_a[1] + t * d[1],
        seg_a[2] + t * d[2],
    ];
    dist3(pt, &closest)
}

/// Compute closest approach distance between two working objects.
fn closest_approach(a: &WorkObj, b: &WorkObj, _close_seg: bool) -> f64 {
    let a_is_point = a.geom == ObjGeom::Scattered || a.points.len() == 1;
    let b_is_point = b.geom == ObjGeom::Scattered || b.points.len() == 1;

    if a_is_point && b_is_point {
        // Point to point
        return dist3(&a.points[0], &b.points[0]);
    }

    if a_is_point {
        // Point to line
        let pt = &a.points[0];
        let mut min_d = f64::INFINITY;
        for i in 0..b.points.len() - 1 {
            let d = point_segment_dist(pt, &b.points[i], &b.points[i + 1]);
            min_d = min_d.min(d);
        }
        return min_d;
    }

    if b_is_point {
        let pt = &b.points[0];
        let mut min_d = f64::INFINITY;
        for i in 0..a.points.len() - 1 {
            let d = point_segment_dist(pt, &a.points[i], &a.points[i + 1]);
            min_d = min_d.min(d);
        }
        return min_d;
    }

    // Line to line: closest approach between all segment pairs
    let mut min_d = f64::INFINITY;
    for i in 0..a.points.len() - 1 {
        for j in 0..b.points.len() - 1 {
            let d = segment_segment_dist(
                &a.points[i],
                &a.points[i + 1],
                &b.points[j],
                &b.points[j + 1],
            );
            min_d = min_d.min(d);
        }
    }
    min_d
}

/// Compute 3D density graphs using closest approach distances.
///
/// For each reference object, finds distances to all neighbor objects and bins them.
/// The density is normalized by spherical shell volume:  V = 4*pi*r^2*dr * (r^power).
fn close_dist_graph(
    wobjs: &[WorkObj],
    bin_width: f64,
    num_bins: usize,
    ref_types: &Option<Vec<i32>>,
    neigh_types: &Option<Vec<i32>>,
    power: f64,
    close_seg: bool,
    nearest_only: bool,
) -> (Vec<f64>, Vec<f64>) {
    let rad_max = num_bins as f64 * bin_width;

    let mut graph = vec![0.0_f64; num_bins];
    let mut frac_sum = vec![0.0_f64; num_bins];
    for (iref, ref_obj) in wobjs.iter().enumerate() {
        if !type_matches(ref_obj.obj_type, ref_types) {
            continue;
        }

        let mut nearest_dist = f64::INFINITY;
        let mut nearest_bin: Option<usize> = None;

        for (ineigh, neigh_obj) in wobjs.iter().enumerate() {
            if ineigh == iref || !type_matches(neigh_obj.obj_type, neigh_types) {
                continue;
            }
            let d = closest_approach(ref_obj, neigh_obj, close_seg);
            if d < rad_max {
                let ibin = (d / bin_width) as usize;
                let ibin = ibin.min(num_bins - 1);
                if nearest_only {
                    if d < nearest_dist {
                        nearest_dist = d;
                        nearest_bin = Some(ibin);
                    }
                } else {
                    graph[ibin] += 1.0;
                }
            }
        }

        if nearest_only {
            if let Some(ibin) = nearest_bin {
                graph[ibin] += 1.0;
            }
        }

        // Accumulate shell volumes for normalization
        for ibin in 0..num_bins {
            let r = (ibin as f64 + 0.5) * bin_width;
            let shell_vol = 4.0 * PI * r * r * bin_width * r.powf(power - 1.0);
            frac_sum[ibin] += shell_vol;
        }
    }

    // Normalize: density = count / (num_refs * shell_volume)
    for ibin in 0..num_bins {
        if frac_sum[ibin] > 0.0 {
            graph[ibin] /= frac_sum[ibin];
        }
    }

    (graph, frac_sum)
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.model_file).unwrap_or_else(|e| {
        eprintln!("ERROR: mtk - failed to read model '{}': {}", args.model_file, e);
        process::exit(1);
    });

    let ref_types = parse_types(&args.ref_types);
    let neigh_types = parse_types(&args.neigh_types);

    // Determine scaling from model header or args
    let xy_scale = if args.xy_scale > 0.0 {
        args.xy_scale
    } else {
        model.pixel_size as f64
    };
    let z_scale = if args.z_scale > 0.0 {
        args.z_scale * xy_scale
    } else {
        xy_scale * model.scale.z as f64
    };

    eprintln!("XY scale: {:.6} um/pixel, Z scale: {:.6}", xy_scale, z_scale);

    // Extract working objects
    let wobjs = extract_work_objects(&model, xy_scale, z_scale, args.z_start, args.z_end);
    if wobjs.is_empty() {
        eprintln!("ERROR: mtk - no objects found in model");
        process::exit(1);
    }

    // Report object counts by type
    let mut type_counts: std::collections::BTreeMap<i32, (usize, ObjGeom)> =
        std::collections::BTreeMap::new();
    for w in &wobjs {
        let entry = type_counts.entry(w.obj_type).or_insert((0, w.geom));
        entry.0 += 1;
    }
    eprintln!("Object  count  geometry");
    for (&t, &(count, geom)) in &type_counts {
        let geom_str = match geom {
            ObjGeom::Scattered => "scattered",
            ObjGeom::OpenLine => "open line",
            ObjGeom::Closed => "closed",
        };
        eprintln!("{:5}  {:6}  {}", t, count, geom_str);
    }
    eprintln!("Total working objects: {}", wobjs.len());

    // Compute closest-approach density graph
    let (graph, _frac_sum) = close_dist_graph(
        &wobjs,
        args.bin_width,
        args.num_bins,
        &ref_types,
        &neigh_types,
        args.power,
        args.close_seg,
        args.nearest_only,
    );

    // Output results
    let mut out: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("ERROR: mtk - cannot create '{}': {}", path, e);
            process::exit(1);
        }))
    } else {
        Box::new(std::io::stdout())
    };

    writeln!(out, "# MTK closest-approach density vs. distance").unwrap();
    writeln!(
        out,
        "# Model: {}  {} working objects  power={:.1}",
        args.model_file,
        wobjs.len(),
        args.power
    )
    .unwrap();
    writeln!(out, "# bin_center  density").unwrap();
    for ibin in 0..args.num_bins {
        let r = (ibin as f64 + 0.5) * args.bin_width;
        writeln!(out, "{:.6}  {:.8}", r, graph[ibin]).unwrap();
    }

    // Shuffle test: shuffle object types and recompute
    if args.num_shuffles > 0 {
        let mut rng = rand::rng();
        let real_integ: f64 = graph.iter().sum();
        let mut n_above = 0usize;

        writeln!(out, "\n# Shuffle test: {} shuffles", args.num_shuffles).unwrap();
        for _ in 0..args.num_shuffles {
            let mut shuffled = wobjs.clone();
            let n = shuffled.len();
            // Shuffle types only (keeping positions)
            let mut types: Vec<i32> = shuffled.iter().map(|w| w.obj_type).collect();
            for k in (1..n).rev() {
                let j = rng.random_range(0..=k);
                types.swap(k, j);
            }
            for (i, w) in shuffled.iter_mut().enumerate() {
                w.obj_type = types[i];
            }

            let (shuf_graph, _) = close_dist_graph(
                &shuffled,
                args.bin_width,
                args.num_bins,
                &ref_types,
                &neigh_types,
                args.power,
                args.close_seg,
                args.nearest_only,
            );
            let shuf_integ: f64 = shuf_graph.iter().sum();
            if shuf_integ >= real_integ {
                n_above += 1;
            }
        }

        writeln!(
            out,
            "# Real integral: {:.6}  Shuffles above: {}/{}  ({:.1}%)",
            real_integ,
            n_above,
            args.num_shuffles,
            100.0 * n_above as f64 / args.num_shuffles as f64
        )
        .unwrap();
    }
}
