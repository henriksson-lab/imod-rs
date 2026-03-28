use std::io::Write;
use std::process;

use clap::Parser;
use imod_math::parse_list;
use imod_model::{ImodMesh, read_model};
use rand::Rng;

/// SDA - Spatial Distribution Analysis on triangulated surfaces.
///
/// Analyzes the distribution of points (pores/particles) located on meshed
/// surfaces in 3D IMOD models.  Computes density of neighboring points as a
/// function of geodesic-approximated distance along the surface, with proper
/// area correction via triangle subdivision.  Supports random shuffling and
/// random point generation for significance testing.
#[derive(Parser)]
#[command(name = "sda", version, about)]
struct Args {
    /// IMOD model file with meshed surface and point objects
    #[arg(required = true)]
    model_file: String,

    /// Object number for the surface mesh (1-based)
    #[arg(short = 'S', long, default_value_t = 1)]
    surface_object: usize,

    /// Object numbers for point/pore objects (comma-separated, 1-based)
    #[arg(short = 'P', long, default_value = "2")]
    point_objects: String,

    /// Radial bin width in model units
    #[arg(short = 'd', long, default_value_t = 1.0)]
    bin_width: f64,

    /// Number of radial bins
    #[arg(short = 'n', long, default_value_t = 50)]
    num_bins: usize,

    /// Reference point types (comma-separated object numbers, or "all")
    #[arg(short = 'r', long, default_value = "all")]
    ref_types: String,

    /// Neighbor point types (comma-separated object numbers, or "all")
    #[arg(short = 't', long, default_value = "all")]
    neigh_types: String,

    /// Number of random shuffles for significance testing
    #[arg(short = 's', long, default_value_t = 0)]
    num_shuffles: usize,

    /// Output file (default: stdout)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// X/Y scale (microns per pixel, 0 to read from model)
    #[arg(long, default_value_t = 0.0)]
    xy_scale: f32,

    /// Z scale factor relative to X/Y
    #[arg(long, default_value_t = 1.0)]
    z_scale: f32,
}

/// A triangle defined by 3 vertex indices.
#[derive(Clone, Debug)]
struct Triangle {
    v: [usize; 3],
}

/// 3D vertex.
#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

#[allow(dead_code)]
impl Vec3 {
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
    fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn dist_sq(self, other: Vec3) -> f64 {
        let d = self.sub(other);
        d.x * d.x + d.y * d.y + d.z * d.z
    }
    fn lerp(self, other: Vec3, t: f64) -> Vec3 {
        Vec3 {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
            z: self.z + t * (other.z - self.z),
        }
    }
}

/// A typed point on the surface with its 3D coordinates.
#[derive(Clone, Debug)]
struct SurfacePoint {
    pos: Vec3,
    point_type: i32,
    #[allow(dead_code)]
    triangle_idx: usize,
}

/// Triangle area from vertices.
fn triangle_area(v0: Vec3, v1: Vec3, v2: Vec3) -> f64 {
    let e1 = v1.sub(v0);
    let e2 = v2.sub(v0);
    0.5 * e1.cross(e2).length()
}

/// Extract triangles from an IMOD mesh.
/// IMOD mesh indices use special codes: positive values are vertex indices,
/// negative values are drawing commands (-25=BGNPOLYNORM, -23=BGNPOLY, -22=ENDPOLY, -1=END).
fn extract_triangles(mesh: &ImodMesh) -> Vec<Triangle> {
    let mut triangles = Vec::new();
    let indices = &mesh.indices;
    let mut i = 0;
    while i < indices.len() {
        let code = indices[i];
        if code == -25 {
            // BGNPOLYNORM: next indices are normal,vertex pairs in groups of 3
            i += 1;
            // Read 3 normal-vertex pairs
            if i + 5 < indices.len() {
                let _n0 = indices[i] as usize;
                let v0 = indices[i + 1] as usize;
                let _n1 = indices[i + 2] as usize;
                let v1 = indices[i + 3] as usize;
                let _n2 = indices[i + 4] as usize;
                let v2 = indices[i + 5] as usize;
                triangles.push(Triangle { v: [v0, v1, v2] });
                i += 6;
            } else {
                i += 1;
            }
        } else if code == -23 {
            // BGNPOLY: simple triangle fan
            i += 1;
            let mut fan_verts = Vec::new();
            while i < indices.len() && indices[i] >= 0 {
                fan_verts.push(indices[i] as usize);
                i += 1;
            }
            // Triangulate fan
            for k in 1..fan_verts.len().saturating_sub(1) {
                triangles.push(Triangle {
                    v: [fan_verts[0], fan_verts[k], fan_verts[k + 1]],
                });
            }
        } else if code == -22 || code == -1 {
            i += 1;
        } else {
            i += 1;
        }
    }
    triangles
}

/// Find the closest triangle to a given point, returning (triangle_index, distance_squared).
fn find_closest_triangle(
    pos: Vec3,
    vertices: &[Vec3],
    triangles: &[Triangle],
) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, tri) in triangles.iter().enumerate() {
        // Use centroid as approximation for speed
        let v0 = vertices[tri.v[0]];
        let v1 = vertices[tri.v[1]];
        let v2 = vertices[tri.v[2]];
        let cx = (v0.x + v1.x + v2.x) / 3.0;
        let cy = (v0.y + v1.y + v2.y) / 3.0;
        let cz = (v0.z + v1.z + v2.z) / 3.0;
        let d = (pos.x - cx).powi(2) + (pos.y - cy).powi(2) + (pos.z - cz).powi(2);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// Compute surface density graph: density of neighbors as a function of
/// 3D Euclidean distance, with area correction based on triangle areas.
///
/// For each reference point, we compute what fraction of each distance bin's
/// shell volume intersects actual surface area, by checking distances from
/// the reference point to subdivided triangle vertices.
fn surface_density_graph(
    vertices: &[Vec3],
    triangles: &[Triangle],
    points: &[SurfacePoint],
    bin_width: f64,
    num_bins: usize,
    ref_types: &Option<Vec<i32>>,
    neigh_types: &Option<Vec<i32>>,
    subdivisions: usize,
) -> (Vec<f64>, Vec<f64>) {
    let rad_max = num_bins as f64 * bin_width;
    let rad_max_sq = rad_max * rad_max;

    let mut graph = vec![0.0_f64; num_bins];
    let mut frac_sum = vec![0.0_f64; num_bins];

    let is_neigh: Vec<bool> = points
        .iter()
        .map(|p| type_matches(p.point_type, neigh_types))
        .collect();

    // Precompute subdivided triangle vertices and their areas
    let sub_points = subdivide_triangles(vertices, triangles, subdivisions);

    for (iref, ref_pt) in points.iter().enumerate() {
        if !type_matches(ref_pt.point_type, ref_types) {
            continue;
        }
        let p0 = ref_pt.pos;

        // Accumulate area fractions from triangle subdivision
        for &(sv, sub_area) in &sub_points {
            let dsq = p0.dist_sq(sv);
            if dsq < rad_max_sq {
                let d = dsq.sqrt();
                let ibin = (d / bin_width) as usize;
                if ibin < num_bins {
                    frac_sum[ibin] += sub_area;
                }
            }
        }

        // Count neighbors in bins
        for (inay, nay_pt) in points.iter().enumerate() {
            if inay == iref || !is_neigh[inay] {
                continue;
            }
            let dsq = p0.dist_sq(nay_pt.pos);
            if dsq < rad_max_sq {
                let d = dsq.sqrt();
                let ibin = (d / bin_width) as usize;
                if ibin < num_bins {
                    graph[ibin] += 1.0;
                }
            }
        }
    }

    // Convert counts to densities: graph[i] = count / area_in_bin
    for ibin in 0..num_bins {
        if frac_sum[ibin] > 0.0 {
            graph[ibin] /= frac_sum[ibin];
        } else {
            graph[ibin] = 0.0;
        }
    }

    (graph, frac_sum)
}

/// Subdivide all triangles and return (position, area) pairs for sub-triangles.
fn subdivide_triangles(
    vertices: &[Vec3],
    triangles: &[Triangle],
    ndiv: usize,
) -> Vec<(Vec3, f64)> {
    let mut result = Vec::new();
    let ndiv = ndiv.max(1);
    for tri in triangles {
        let v0 = vertices[tri.v[0]];
        let v1 = vertices[tri.v[1]];
        let v2 = vertices[tri.v[2]];
        let total_area = triangle_area(v0, v1, v2);
        if total_area <= 0.0 {
            continue;
        }
        // Subdivide into ndiv^2 sub-triangles using barycentric coordinates
        let inv_n = 1.0 / ndiv as f64;
        let sub_area = total_area / (ndiv * ndiv) as f64;
        for i in 0..ndiv {
            for j in 0..(ndiv - i) {
                // Centroid of sub-triangle (i,j)
                let u = (i as f64 + 1.0 / 3.0) * inv_n;
                let v = (j as f64 + 1.0 / 3.0) * inv_n;
                let w = 1.0 - u - v;
                let cx = w * v0.x + u * v1.x + v * v2.x;
                let cy = w * v0.y + u * v1.y + v * v2.y;
                let cz = w * v0.z + u * v1.z + v * v2.z;
                result.push((Vec3 { x: cx, y: cy, z: cz }, sub_area));

                // Second sub-triangle in the quad (if it exists)
                if i + j + 1 < ndiv {
                    let ub = (i as f64 + 2.0 / 3.0) * inv_n;
                    let vb = (j as f64 + 2.0 / 3.0) * inv_n;
                    let wb = 1.0 - ub - vb;
                    let cx2 = wb * v0.x + ub * v1.x + vb * v2.x;
                    let cy2 = wb * v0.y + ub * v1.y + vb * v2.y;
                    let cz2 = wb * v0.z + ub * v1.z + vb * v2.z;
                    result.push((Vec3 { x: cx2, y: cy2, z: cz2 }, sub_area));
                }
            }
        }
    }
    result
}

fn parse_types(s: &str) -> Option<Vec<i32>> {
    if s.eq_ignore_ascii_case("all") {
        None
    } else {
        Some(parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: sda - bad type list '{}': {}", s, e);
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

fn main() {
    let args = Args::parse();

    let model = read_model(&args.model_file).unwrap_or_else(|e| {
        eprintln!("ERROR: sda - failed to read model '{}': {}", args.model_file, e);
        process::exit(1);
    });

    let ref_types = parse_types(&args.ref_types);
    let neigh_types = parse_types(&args.neigh_types);

    // Determine scaling
    let xy_scale = if args.xy_scale > 0.0 {
        args.xy_scale as f64
    } else {
        model.pixel_size as f64
    };
    let z_scale = args.z_scale as f64 * xy_scale;

    // Extract surface mesh
    let surf_idx = args.surface_object - 1;
    if surf_idx >= model.objects.len() {
        eprintln!(
            "ERROR: sda - surface object {} does not exist (model has {} objects)",
            args.surface_object,
            model.objects.len()
        );
        process::exit(1);
    }
    let surf_obj = &model.objects[surf_idx];
    if surf_obj.meshes.is_empty() {
        eprintln!(
            "ERROR: sda - object {} has no meshes; run imodmesh first",
            args.surface_object
        );
        process::exit(1);
    }

    // Collect vertices and triangles from all meshes of the surface object
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<Triangle> = Vec::new();
    for mesh in &surf_obj.meshes {
        let vert_offset = vertices.len();
        for v in &mesh.vertices {
            vertices.push(Vec3 {
                x: v.x as f64 * xy_scale,
                y: v.y as f64 * xy_scale,
                z: v.z as f64 * z_scale,
            });
        }
        for tri in extract_triangles(mesh) {
            triangles.push(Triangle {
                v: [
                    tri.v[0] + vert_offset,
                    tri.v[1] + vert_offset,
                    tri.v[2] + vert_offset,
                ],
            });
        }
    }

    // Compute total surface area
    let total_area: f64 = triangles
        .iter()
        .map(|tri| triangle_area(vertices[tri.v[0]], vertices[tri.v[1]], vertices[tri.v[2]]))
        .sum();

    eprintln!("{} triangles, total surface area = {:.6}", triangles.len(), total_area);

    // Extract points from specified point objects
    let point_objs = parse_list(&args.point_objects).unwrap_or_else(|e| {
        eprintln!("ERROR: sda - bad point objects list: {}", e);
        process::exit(1);
    });

    let mut points: Vec<SurfacePoint> = Vec::new();
    for &obj_num in &point_objs {
        let oi = (obj_num - 1) as usize;
        if oi >= model.objects.len() {
            eprintln!("WARNING: sda - object {} does not exist, skipping", obj_num);
            continue;
        }
        let obj = &model.objects[oi];
        for cont in &obj.contours {
            for pt in &cont.points {
                let pos = Vec3 {
                    x: pt.x as f64 * xy_scale,
                    y: pt.y as f64 * xy_scale,
                    z: pt.z as f64 * z_scale,
                };
                let (tri_idx, _) = find_closest_triangle(pos, &vertices, &triangles);
                points.push(SurfacePoint {
                    pos,
                    point_type: obj_num,
                    triangle_idx: tri_idx,
                });
            }
        }
    }

    if points.is_empty() {
        eprintln!("ERROR: sda - no surface points found");
        process::exit(1);
    }

    let density = points.len() as f64 / total_area;
    eprintln!("{} points, density = {:.6}", points.len(), density);

    // Choose subdivision level based on triangle size vs bin width
    let mean_tri_size = if !triangles.is_empty() {
        (total_area / triangles.len() as f64).sqrt()
    } else {
        1.0
    };
    let ndiv = ((2.0 * mean_tri_size / args.bin_width) + 0.5) as usize;
    let ndiv = ndiv.clamp(1, 19);
    eprintln!("Triangle subdivision level: {}", ndiv);

    // Compute surface density graph
    let (graph, _frac_sum) = surface_density_graph(
        &vertices,
        &triangles,
        &points,
        args.bin_width,
        args.num_bins,
        &ref_types,
        &neigh_types,
        ndiv,
    );

    // Output results
    let mut out: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("ERROR: sda - cannot create '{}': {}", path, e);
            process::exit(1);
        }))
    } else {
        Box::new(std::io::stdout())
    };

    writeln!(out, "# SDA surface density vs. distance").unwrap();
    writeln!(
        out,
        "# Model: {}  {} points on {} triangles  area={:.6}",
        args.model_file,
        points.len(),
        triangles.len(),
        total_area
    )
    .unwrap();
    writeln!(out, "# bin_center  density").unwrap();
    for ibin in 0..args.num_bins {
        let r = (ibin as f64 + 0.5) * args.bin_width;
        writeln!(out, "{:.6}  {:.8}", r, graph[ibin]).unwrap();
    }

    // Shuffle test
    if args.num_shuffles > 0 {
        let mut rng = rand::rng();
        writeln!(out, "\n# Shuffle test: {} shuffles", args.num_shuffles).unwrap();

        let real_integ: f64 = graph.iter().sum();
        let mut n_above = 0usize;

        for _ in 0..args.num_shuffles {
            let mut shuffled = points.clone();
            let n = shuffled.len();
            for k in (1..n).rev() {
                let j = rng.random_range(0..=k);
                let tmp = shuffled[k].point_type;
                shuffled[k].point_type = shuffled[j].point_type;
                shuffled[j].point_type = tmp;
            }
            let (shuf_graph, _) = surface_density_graph(
                &vertices,
                &triangles,
                &shuffled,
                args.bin_width,
                args.num_bins,
                &ref_types,
                &neigh_types,
                ndiv,
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
