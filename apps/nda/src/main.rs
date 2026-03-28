use std::f64::consts::PI;
use std::io::Write;
use std::process;

use clap::Parser;
use imod_math::parse_list;
use imod_model::{ImodModel, read_model};
use rand::Rng;

/// NDA - Neighbor Density Analysis for 2D point patterns.
///
/// Analyzes spatial point patterns where points may be of different types.
/// Produces graphs of point density versus radial distance from reference points,
/// with edge correction for arbitrary boundary polygons.  Can evaluate statistical
/// significance via type shuffling or generation of random point patterns.
#[derive(Parser)]
#[command(name = "nda", version, about)]
struct Args {
    /// IMOD model file containing points to analyze
    #[arg(required = true)]
    model_file: String,

    /// Object number for boundary contour (1-based), or 0 to use bounding rectangle
    #[arg(short = 'b', long, default_value_t = 0)]
    boundary_object: i32,

    /// Contour index within boundary object (0-based)
    #[arg(long, default_value_t = 0)]
    boundary_contour: usize,

    /// Section Z value to analyze (if boundary_object=0)
    #[arg(short = 'z', long, default_value_t = 0.0)]
    section: f32,

    /// Radial bin width in model units
    #[arg(short = 'd', long, default_value_t = 1.0)]
    bin_width: f64,

    /// Number of radial bins
    #[arg(short = 'n', long, default_value_t = 50)]
    num_bins: usize,

    /// Reference point types (comma-separated, or "all")
    #[arg(short = 'r', long, default_value = "all")]
    ref_types: String,

    /// Neighbor point types (comma-separated, or "all")
    #[arg(short = 't', long, default_value = "all")]
    neigh_types: String,

    /// Number of random shuffles for significance testing
    #[arg(short = 's', long, default_value_t = 0)]
    num_shuffles: usize,

    /// Number of random point sets for significance testing
    #[arg(short = 'R', long, default_value_t = 0)]
    num_random: usize,

    /// Padding distance for boundary when using auto-boundary
    #[arg(short = 'p', long, default_value_t = 0.0)]
    pad: f32,

    /// Output file for density values (default: stdout)
    #[arg(short = 'o', long)]
    output: Option<String>,
}

/// A 2D point with a type label.
#[derive(Clone, Debug)]
struct TypedPoint {
    x: f64,
    y: f64,
    point_type: i32,
}

/// Polygon boundary defined by vertices (closed: last connects to first).
#[derive(Clone, Debug)]
struct Boundary {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Boundary {
    fn num_verts(&self) -> usize {
        self.x.len()
    }

    /// Signed area via shoelace formula (positive if CCW).
    fn area(&self) -> f64 {
        let n = self.num_verts();
        if n < 3 {
            return 0.0;
        }
        let mut a = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            a += (self.y[j] + self.y[i]) * (self.x[j] - self.x[i]);
        }
        (a * 0.5).abs()
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let xmin = self.x.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = self.x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = self.y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = self.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (xmin, ymin, xmax, ymax)
    }
}

/// Test whether point (px, py) is inside the polygon using ray casting.
fn point_inside(bnd: &Boundary, px: f64, py: f64) -> bool {
    let n = bnd.num_verts();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let xi = bnd.x[i];
        let yi = bnd.y[i];
        let xj = bnd.x[j];
        let yj = bnd.y[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Parse type specification: "all" => None (match all), otherwise parse as list.
fn parse_types(s: &str) -> Option<Vec<i32>> {
    if s.eq_ignore_ascii_case("all") {
        None
    } else {
        Some(parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: nda - bad type list '{}': {}", s, e);
            process::exit(1);
        }))
    }
}

fn type_matches(point_type: i32, filter: &Option<Vec<i32>>) -> bool {
    match filter {
        None => true,
        Some(types) => types.contains(&point_type),
    }
}

/// Extract 2D typed points from an IMOD model.
/// Each object becomes a type (1-based object index).
/// Points on the specified Z section are extracted.
fn extract_points(model: &ImodModel, z_section: f32, z_tolerance: f32) -> Vec<TypedPoint> {
    let mut pts = Vec::new();
    for (obj_idx, obj) in model.objects.iter().enumerate() {
        let ptype = (obj_idx + 1) as i32;
        for cont in &obj.contours {
            for pt in &cont.points {
                if (pt.z - z_section).abs() <= z_tolerance {
                    pts.push(TypedPoint {
                        x: pt.x as f64,
                        y: pt.y as f64,
                        point_type: ptype,
                    });
                }
            }
        }
    }
    pts
}

/// Build boundary from a specific object/contour, or from bounding rectangle of points.
fn build_boundary(
    model: &ImodModel,
    obj_idx: i32,
    cont_idx: usize,
    points: &[TypedPoint],
    pad: f32,
) -> Boundary {
    if obj_idx > 0 {
        let oi = (obj_idx - 1) as usize;
        if oi >= model.objects.len() {
            eprintln!("ERROR: nda - boundary object {} does not exist", obj_idx);
            process::exit(1);
        }
        let obj = &model.objects[oi];
        if cont_idx >= obj.contours.len() {
            eprintln!(
                "ERROR: nda - contour {} does not exist in object {}",
                cont_idx, obj_idx
            );
            process::exit(1);
        }
        let cont = &obj.contours[cont_idx];
        Boundary {
            x: cont.points.iter().map(|p| p.x as f64).collect(),
            y: cont.points.iter().map(|p| p.y as f64).collect(),
        }
    } else {
        // Bounding rectangle around all points with padding
        if points.is_empty() {
            eprintln!("ERROR: nda - no points found for bounding rectangle");
            process::exit(1);
        }
        let pad = pad as f64;
        let xmin = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - pad;
        let xmax = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + pad;
        let ymin = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - pad;
        let ymax = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + pad;
        Boundary {
            x: vec![xmin, xmax, xmax, xmin],
            y: vec![ymin, ymin, ymax, ymax],
        }
    }
}

/// Compute density vs. radial distance graphs with edge correction.
///
/// For each reference point, we determine what fraction of each annular bin
/// lies within the boundary polygon (by counting boundary crossings), then
/// count neighbor points falling in each bin and divide by the corrected area.
///
/// This is the core algorithm from the Fortran `dengraph` subroutine.
fn density_graph(
    boundary: &Boundary,
    points: &[TypedPoint],
    bin_width: f64,
    num_bins: usize,
    ref_types: &Option<Vec<i32>>,
    neigh_types: &Option<Vec<i32>>,
) -> (Vec<f64>, Vec<f64>) {
    let nv = boundary.num_verts();
    let rad_max = num_bins as f64 * bin_width;
    let rad_max_sq = rad_max * rad_max;

    // Precompute edge squared lengths (boundary is treated as closed polygon)
    let mut edge_sq = Vec::with_capacity(nv);
    for i in 0..nv {
        let j = (i + 1) % nv;
        let dx = boundary.x[j] - boundary.x[i];
        let dy = boundary.y[j] - boundary.y[i];
        edge_sq.push(dx * dx + dy * dy);
    }

    let mut graph = vec![0.0_f64; num_bins];
    let mut frac_sum = vec![0.0_f64; num_bins];

    // Precompute neighbor membership
    let is_neigh: Vec<bool> = points.iter().map(|p| type_matches(p.point_type, neigh_types)).collect();

    for (iref, ref_pt) in points.iter().enumerate() {
        if !type_matches(ref_pt.point_type, ref_types) {
            continue;
        }
        let x0 = ref_pt.x;
        let y0 = ref_pt.y;

        // Find boundary crossings with each annular bin
        let mut num_cross: Vec<Vec<f64>> = vec![Vec::new(); num_bins];

        let mut x1 = boundary.x[0];
        let mut y1 = boundary.y[0];
        let mut d1_sq = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);

        for ivert in 0..nv {
            let j = (ivert + 1) % nv;
            let db_sq = edge_sq[ivert];
            let x2 = boundary.x[j];
            let y2 = boundary.y[j];
            let d2_sq = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2);

            // Distance to nearest point on edge segment
            let dist_sq = if d2_sq > d1_sq + db_sq {
                d1_sq
            } else if d1_sq > d2_sq + db_sq {
                d2_sq
            } else {
                d2_sq - (d2_sq + db_sq - d1_sq).powi(2) / (4.0 * db_sq)
            };

            let rad_mid_max_sq = ((num_bins as f64 - 0.5) * bin_width).powi(2);
            if dist_sq < rad_mid_max_sq {
                let dist_min = dist_sq.max(0.0).sqrt();
                let ibin_min = (dist_min / bin_width + 0.5) as usize;
                let b_over_2a = -(d2_sq - d1_sq - db_sq) / (2.0 * db_sq);
                let rad_part = b_over_2a * b_over_2a - d1_sq / db_sq;
                let dist_max_sq = d1_sq.max(d2_sq);

                for ibin in ibin_min..num_bins {
                    let rr = (ibin as f64 + 0.5) * bin_width;
                    let rsq = rr * rr;
                    if rsq > dist_max_sq {
                        break;
                    }
                    let rad = rad_part + rsq / db_sq;
                    if rad >= 0.0 {
                        let root = rad.max(0.0).sqrt();
                        for &dir in &[-1.0_f64, 1.0_f64] {
                            let fsol = b_over_2a + dir * root;
                            // Open on one end to prevent double-counting at endpoints
                            if fsol > 0.0 && fsol <= 1.0 {
                                let xsolve = fsol * (x2 - x1) + x1 - x0;
                                let ysolve = fsol * (y2 - y1) + y1 - y0;
                                let angle = ysolve.atan2(xsolve).to_degrees();
                                num_cross[ibin].push(angle);
                            }
                        }
                    }
                }
            }

            x1 = x2;
            y1 = y2;
            d1_sq = d2_sq;
        }

        // Determine fraction of each annulus inside boundary
        let ibin_start = num_cross.iter().position(|c| !c.is_empty()).unwrap_or(num_bins);
        let ibin_end = num_cross.iter().rposition(|c| !c.is_empty()).map_or(0, |i| i + 1);

        let mut frac_add = 1.0;
        for ibin in 0..ibin_end.min(num_bins) {
            if ibin >= ibin_start && !num_cross[ibin].is_empty() {
                let crossings = &mut num_cross[ibin];
                crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut diff_sum = 0.0;
                let mut k = 1;
                while k < crossings.len() {
                    diff_sum += crossings[k] - crossings[k - 1];
                    k += 2;
                }
                // Check if point at -180 degrees is inside boundary
                let test_x = x0 - (ibin as f64 + 0.5) * bin_width;
                if point_inside(boundary, test_x, y0) {
                    diff_sum = 360.0 - diff_sum;
                }
                frac_add = diff_sum / 360.0;
            }
            frac_sum[ibin] += frac_add;
        }

        // Add neighbor counts to bins
        for (inay, nay_pt) in points.iter().enumerate() {
            if inay == iref || !is_neigh[inay] {
                continue;
            }
            let dx = (x0 - nay_pt.x).abs();
            if dx >= rad_max {
                continue;
            }
            let dy = (y0 - nay_pt.y).abs();
            if dy >= rad_max {
                continue;
            }
            let rsq = dx * dx + dy * dy;
            if rsq < rad_max_sq {
                let ibin = (rsq.sqrt() / bin_width) as usize;
                if ibin < num_bins {
                    graph[ibin] += 1.0;
                }
            }
        }
    }

    // Convert counts to densities
    for ibin in 0..num_bins {
        let annulus_area = PI * bin_width * bin_width * (2 * ibin + 1) as f64;
        let total_area = annulus_area * frac_sum[ibin];
        if total_area > 0.0 {
            graph[ibin] /= total_area;
        } else {
            graph[ibin] = 0.0;
        }
    }

    (graph, frac_sum)
}

/// Generate random points inside boundary polygon.
fn random_points_in_boundary(boundary: &Boundary, n: usize, rng: &mut impl Rng) -> Vec<(f64, f64)> {
    let (xmin, ymin, xmax, ymax) = boundary.bounding_box();
    let mut pts = Vec::with_capacity(n);
    while pts.len() < n {
        let x = rng.random_range(xmin..xmax);
        let y = rng.random_range(ymin..ymax);
        if point_inside(boundary, x, y) {
            pts.push((x, y));
        }
    }
    pts
}

/// Compute the integral of density above baseline for selected bins.
fn integrate_graph(graph: &[f64], bin_width: f64, start: usize, end: usize, baseline: f64) -> f64 {
    let mut sum = 0.0;
    for ibin in start..=end.min(graph.len() - 1) {
        let annulus_area = PI * bin_width * bin_width * (2 * ibin + 1) as f64;
        sum += annulus_area * (graph[ibin] - baseline);
    }
    sum
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.model_file).unwrap_or_else(|e| {
        eprintln!("ERROR: nda - failed to read model '{}': {}", args.model_file, e);
        process::exit(1);
    });

    let ref_types = parse_types(&args.ref_types);
    let neigh_types = parse_types(&args.neigh_types);

    // Extract 2D points on the specified section
    let points = extract_points(&model, args.section, 0.5);
    if points.is_empty() {
        eprintln!("ERROR: nda - no points found on section z={}", args.section);
        process::exit(1);
    }

    // Build boundary
    let boundary = build_boundary(
        &model,
        args.boundary_object,
        args.boundary_contour,
        &points,
        args.pad,
    );

    // Filter to points inside boundary
    let points: Vec<TypedPoint> = points
        .into_iter()
        .filter(|p| point_inside(&boundary, p.x, p.y))
        .collect();

    let area = boundary.area();
    let density = points.len() as f64 / area;
    eprintln!(
        "{} points, area = {:.6}, density = {:.6}",
        points.len(),
        area,
        density
    );

    // Report per-type counts
    let mut type_counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for p in &points {
        *type_counts.entry(p.point_type).or_insert(0) += 1;
    }
    eprintln!("Type   number    density");
    for (&t, &count) in &type_counts {
        eprintln!("{:5}{:8}  {:.6}", t, count, count as f64 / area);
    }

    // Compute density graph
    let (graph, _frac_sum) =
        density_graph(&boundary, &points, args.bin_width, args.num_bins, &ref_types, &neigh_types);

    // Open output
    let mut out: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(
            std::fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("ERROR: nda - cannot create '{}': {}", path, e);
                process::exit(1);
            }),
        )
    } else {
        Box::new(std::io::stdout())
    };

    // Write header
    writeln!(out, "# NDA density vs. radial distance").unwrap();
    writeln!(
        out,
        "# Model: {}  Section z={}  {} points  area={:.6}",
        args.model_file,
        args.section,
        points.len(),
        area
    )
    .unwrap();
    writeln!(out, "# bin_center  density").unwrap();

    for ibin in 0..args.num_bins {
        let r = (ibin as f64 + 0.5) * args.bin_width;
        writeln!(out, "{:.6}  {:.8}", r, graph[ibin]).unwrap();
    }

    // Shuffling test: shuffle types among points, recompute, accumulate statistics
    if args.num_shuffles > 0 {
        let mut rng = rand::rng();
        let mut sum_integ = vec![0.0_f64; args.num_shuffles];
        let real_integ = integrate_graph(&graph, args.bin_width, 0, args.num_bins - 1, density);

        writeln!(out, "\n# Shuffle test: {} shuffles", args.num_shuffles).unwrap();
        for i_shuf in 0..args.num_shuffles {
            let mut shuffled = points.clone();
            // Fisher-Yates shuffle of types
            let n = shuffled.len();
            for k in (1..n).rev() {
                let j = rng.random_range(0..=k);
                let tmp = shuffled[k].point_type;
                shuffled[k].point_type = shuffled[j].point_type;
                shuffled[j].point_type = tmp;
            }
            let (shuf_graph, _) = density_graph(
                &boundary,
                &shuffled,
                args.bin_width,
                args.num_bins,
                &ref_types,
                &neigh_types,
            );
            sum_integ[i_shuf] =
                integrate_graph(&shuf_graph, args.bin_width, 0, args.num_bins - 1, density);
        }

        let n_above = sum_integ.iter().filter(|&&v| v >= real_integ).count();
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

    // Random point test
    if args.num_random > 0 {
        let mut rng = rand::rng();
        let real_integ = integrate_graph(&graph, args.bin_width, 0, args.num_bins - 1, density);

        writeln!(out, "\n# Random test: {} random sets", args.num_random).unwrap();
        let mut n_above = 0usize;
        for _ in 0..args.num_random {
            let rand_xy = random_points_in_boundary(&boundary, points.len(), &mut rng);
            let rand_pts: Vec<TypedPoint> = rand_xy
                .iter()
                .zip(points.iter())
                .map(|(&(x, y), orig)| TypedPoint {
                    x,
                    y,
                    point_type: orig.point_type,
                })
                .collect();
            let (rand_graph, _) = density_graph(
                &boundary,
                &rand_pts,
                args.bin_width,
                args.num_bins,
                &ref_types,
                &neigh_types,
            );
            let rand_integ =
                integrate_graph(&rand_graph, args.bin_width, 0, args.num_bins - 1, density);
            if rand_integ >= real_integ {
                n_above += 1;
            }
        }
        writeln!(
            out,
            "# Real integral: {:.6}  Random above: {}/{}  ({:.1}%)",
            real_integ,
            n_above,
            args.num_random,
            100.0 * n_above as f64 / args.num_random as f64
        )
        .unwrap();
    }
}
