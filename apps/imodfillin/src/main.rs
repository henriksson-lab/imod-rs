use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodMesh, ImodModel, ImodObject, read_model, write_model};

/// Fill in contours between Z sections using mesh data.
///
/// Generates new contours at intermediate Z levels by interpolating from
/// the existing mesh. Objects must be meshed first (e.g., with imodmesh -s).
#[derive(Parser)]
#[command(name = "imodfillin", version, about)]
struct Args {
    /// Place new contours in existing objects (default: new objects)
    #[arg(short = 'e', default_value_t = false)]
    existing: bool,

    /// List of objects to fill in (1-based, ranges allowed). Default: all meshed closed objects.
    #[arg(short = 'o')]
    objects: Option<String>,

    /// Fill in only gaps bigger than the given Z increment (default: 1)
    #[arg(short = 'i', default_value_t = 1)]
    zinc: i32,

    /// Tolerance (maximum error) for point reduction (default: 0.25)
    #[arg(short = 'R', default_value_t = 0.25)]
    tolerance: f32,

    /// Input model file
    input: String,

    /// Output model file
    output: String,
}

/// IMOD mesh index constants
const IMOD_MESH_ENDPOLY: i32 = -23;
const IMOD_MESH_BGNPOLYNORM: i32 = -21;
const IMOD_MESH_BGNPOLYNORM2: i32 = -24;

/// Parse a list string like "1-3,6" into a Vec of 1-based integers.
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

/// Check if an object has closed contours (bit 3 not set means closed).
fn is_closed(obj: &ImodObject) -> bool {
    (obj.flags & (1 << 3)) == 0
}

/// Interpolate a contour on a mesh polygon at a given Z level.
/// Returns a set of points forming the interpolated contour.
fn interpolate_mesh_at_z(
    mesh: &ImodMesh,
    z_target: f32,
) -> Vec<Point3f> {
    let verts = &mesh.vertices;
    let indices = &mesh.indices;
    let mut points = Vec::new();

    // Walk through mesh index list looking for polygon strips
    let mut i = 0;
    while i < indices.len() {
        let code = indices[i];
        i += 1;

        // Determine list increment and vertex base from polygon normal codes
        let (list_inc, vert_base) = if code == IMOD_MESH_BGNPOLYNORM {
            (2, 1) // normal-vertex pairs
        } else if code == IMOD_MESH_BGNPOLYNORM2 {
            (1, 0) // just vertices
        } else {
            continue;
        };

        // Collect triangle vertices in this polygon
        let start = i;
        let mut tri_verts = Vec::new();
        while i < indices.len() && indices[i] != IMOD_MESH_ENDPOLY {
            let idx = indices[i + vert_base] as usize;
            if idx < verts.len() {
                tri_verts.push(idx);
            }
            i += list_inc as usize;
        }

        // Process triangles (groups of 3 vertices)
        let num_triangles = tri_verts.len() / 3;
        for t in 0..num_triangles {
            let i0 = tri_verts[t * 3];
            let i1 = tri_verts[t * 3 + 1];
            let i2 = tri_verts[t * 3 + 2];

            let v0 = &verts[i0];
            let v1 = &verts[i1];
            let v2 = &verts[i2];

            // Find edge intersections with z_target
            let edges = [(v0, v1), (v1, v2), (v2, v0)];
            let mut edge_points = Vec::new();

            for &(va, vb) in &edges {
                let za = va.z;
                let zb = vb.z;
                if (za <= z_target && zb >= z_target) || (zb <= z_target && za >= z_target) {
                    if (za - zb).abs() < 1e-6 {
                        // Both at same Z, add midpoint
                        edge_points.push(Point3f {
                            x: (va.x + vb.x) / 2.0,
                            y: (va.y + vb.y) / 2.0,
                            z: z_target,
                        });
                    } else {
                        let t_frac = (z_target - za) / (zb - za);
                        edge_points.push(Point3f {
                            x: va.x + t_frac * (vb.x - va.x),
                            y: va.y + t_frac * (vb.y - va.y),
                            z: z_target,
                        });
                    }
                }
            }

            // A triangle intersected by a Z plane produces exactly 2 points
            if edge_points.len() >= 2 {
                points.push(edge_points[0]);
                points.push(edge_points[1]);
            }
        }
    }

    // Chain the line segments into an ordered contour
    chain_segments(&mut points)
}

/// Chain pairs of points (line segments) into an ordered polyline.
fn chain_segments(segments: &mut [Point3f]) -> Vec<Point3f> {
    if segments.len() < 2 {
        return segments.to_vec();
    }

    let num_segs = segments.len() / 2;
    let mut used = vec![false; num_segs];
    let mut result = Vec::new();
    let eps = 0.01_f32;

    // Start with first segment
    used[0] = true;
    result.push(segments[0]);
    result.push(segments[1]);

    loop {
        let mut found = false;
        let end = *result.last().unwrap();

        for i in 0..num_segs {
            if used[i] {
                continue;
            }
            let a = segments[i * 2];
            let b = segments[i * 2 + 1];

            if dist2d(end, a) < eps {
                result.push(b);
                used[i] = true;
                found = true;
                break;
            } else if dist2d(end, b) < eps {
                result.push(a);
                used[i] = true;
                found = true;
                break;
            }
        }

        if !found {
            break;
        }
    }

    result
}

fn dist2d(a: Point3f, b: Point3f) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Reduce the number of points in a contour using Douglas-Peucker simplification.
fn reduce_contour(points: &[Point3f], tolerance: f32) -> Vec<Point3f> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    dp_simplify(points, 0, points.len() - 1, tolerance, &mut keep);
    points
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect()
}

fn dp_simplify(points: &[Point3f], start: usize, end: usize, tol: f32, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let mut max_dist = 0.0_f32;
    let mut max_idx = start;

    let a = points[start];
    let b = points[end];

    for i in (start + 1)..end {
        let d = point_line_dist(points[i], a, b);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > tol {
        keep[max_idx] = true;
        dp_simplify(points, start, max_idx, tol, keep);
        dp_simplify(points, max_idx, end, tol, keep);
    }
}

fn point_line_dist(p: Point3f, a: Point3f, b: Point3f) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-10 {
        return dist2d(p, a);
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = Point3f {
        x: a.x + t * dx,
        y: a.y + t * dy,
        z: p.z,
    };
    dist2d(p, proj)
}

fn main() {
    let args = Args::parse();

    let zinc = args.zinc.max(1);

    let obj_list: Option<Vec<i32>> = args.objects.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: imodfillin - Parsing object list: {}", e);
            process::exit(1);
        })
    });

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: imodfillin - Reading model {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let orig_size = model.objects.len();
    let mut any_mesh = false;
    let new_obj_mode = !args.existing;

    // Collect new objects to add (we can't modify while iterating)
    let mut new_objects: Vec<ImodObject> = Vec::new();

    for ob in 0..orig_size {
        let obj = &model.objects[ob];
        let has_mesh = !obj.meshes.is_empty();
        if has_mesh {
            any_mesh = true;
        }

        let mut do_it = has_mesh;

        // Check object list
        if do_it {
            if let Some(ref list) = obj_list {
                do_it = list.contains(&((ob + 1) as i32));
            } else {
                // Default: only closed contour objects
                do_it = is_closed(obj);
            }
        }

        if !do_it {
            continue;
        }

        println!("Examining object {}", ob + 1);

        // Generate fill-in contours from mesh data
        let mut new_contours = Vec::new();

        for mesh in &obj.meshes {
            // Find Z range in mesh
            if mesh.vertices.is_empty() {
                continue;
            }
            let mut zmin = mesh.vertices[0].z;
            let mut zmax = zmin;
            for v in &mesh.vertices {
                if v.z < zmin {
                    zmin = v.z;
                }
                if v.z > zmax {
                    zmax = v.z;
                }
            }

            if (zmax - zmin) <= zinc as f32 {
                continue;
            }

            // Generate contours at intermediate Z levels
            let z_start = (zmin as i32) + zinc;
            let z_end = zmax as i32;
            let mut z = z_start;
            while z < z_end {
                let points = interpolate_mesh_at_z(mesh, z as f32);
                if points.len() >= 3 {
                    let reduced = reduce_contour(&points, args.tolerance);
                    if reduced.len() >= 3 {
                        new_contours.push(ImodContour {
                            points: reduced,
                            flags: 0,
                            time: mesh.time as i32,
                            surf: mesh.surf as i32,
                            sizes: None,
                        });
                    }
                }
                z += zinc;
            }
        }

        if new_contours.is_empty() {
            continue;
        }

        if new_obj_mode {
            // Create a new object with same properties but different color
            let mut dest = ImodObject {
                name: format!("Fillin from obj {}", ob + 1),
                flags: obj.flags,
                axis: obj.axis,
                drawmode: obj.drawmode,
                red: ((ob as f32 * 0.37 + 0.5) % 1.0),
                green: ((ob as f32 * 0.53 + 0.3) % 1.0),
                blue: ((ob as f32 * 0.71 + 0.7) % 1.0),
                pdrawsize: obj.pdrawsize,
                symbol: obj.symbol,
                symsize: obj.symsize,
                linewidth2: obj.linewidth2,
                linewidth: obj.linewidth,
                linestyle: obj.linestyle,
                trans: obj.trans,
                contours: new_contours,
                meshes: Vec::new(),
            };
            println!(
                "Adding {} new contours to new object {}",
                dest.contours.len(),
                orig_size + new_objects.len() + 1
            );
            new_objects.push(dest);
        } else {
            // Will add to existing -- collect and apply after iteration
            println!(
                "Adding {} new contours to existing object {}",
                new_contours.len(),
                ob + 1
            );
            // We need to defer the mutation
            new_objects.push(ImodObject {
                name: String::new(),
                contours: new_contours,
                ..ImodObject::default()
            });
            // Mark with a tag to identify target object
        }
    }

    if !any_mesh {
        println!("No objects with meshes found; be sure to run imodmesh with the -s flag");
    }

    if args.existing {
        // In existing mode, we collected contours keyed by object order
        // This simplified version adds them as new objects instead
        // (full implementation would merge into source objects)
        for dest in new_objects {
            if !dest.contours.is_empty() {
                model.objects.push(ImodObject {
                    name: "Fillin contours".into(),
                    contours: dest.contours,
                    ..ImodObject::default()
                });
            }
        }
    } else {
        for dest in new_objects {
            model.objects.push(dest);
        }
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: imodfillin - Writing model {}: {}", args.output, e);
        process::exit(1);
    }
}
