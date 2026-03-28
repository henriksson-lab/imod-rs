use std::collections::BTreeMap;
use std::process;

use clap::Parser;
use imod_mesh::{skin_contours, Contour2d};
use imod_model::{ImodMesh, read_model, write_model};

/// Add triangle mesh data to IMOD model objects by skinning contours between
/// adjacent Z sections.
#[derive(Parser)]
#[command(name = "imodmesh", version, about)]
struct Args {
    /// Only mesh the given objects (1-based, comma-separated).
    #[arg(short = 'o', value_delimiter = ',')]
    objects: Option<Vec<usize>>,

    /// Erase existing meshes instead of creating new ones.
    #[arg(short = 'e')]
    erase: bool,

    /// Append mesh data (keep existing meshes on other Z ranges).
    #[arg(short = 'a')]
    append: bool,

    /// Tolerance for point reduction (default 0.25).
    #[arg(short = 'R', default_value_t = 0.25)]
    tolerance: f32,

    /// Cap the ends of meshed objects.
    #[arg(short = 'c')]
    cap: bool,

    /// Input model file.
    input: String,

    /// Output model file (defaults to overwriting input).
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: imodmesh - error reading {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let num_objects = model.objects.len();

    for ob in 0..num_objects {
        // Check if this object is in the requested list
        if let Some(ref list) = args.objects {
            if !list.contains(&(ob + 1)) {
                continue;
            }
        }

        let obj = &model.objects[ob];

        if args.erase {
            // Just clear meshes
            model.objects[ob].meshes.clear();
            println!("Object {}: meshes erased", ob + 1);
            continue;
        }

        if obj.contours.is_empty() {
            continue;
        }

        // Group contours by Z value (rounded to integer section)
        let mut by_z: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for (ci, cont) in obj.contours.iter().enumerate() {
            if cont.points.is_empty() {
                continue;
            }
            // Use the Z of the first point as the section number
            let z = cont.points[0].z.round() as i32;
            by_z.entry(z).or_default().push(ci);
        }

        let z_sections: Vec<i32> = by_z.keys().copied().collect();
        if z_sections.len() < 2 {
            println!(
                "Object {}: only {} Z section(s), skipping",
                ob + 1,
                z_sections.len()
            );
            continue;
        }

        let is_open = (obj.flags & (1 << 3)) != 0;

        let mut new_meshes: Vec<ImodMesh> = Vec::new();

        // Skin contours between adjacent Z sections
        for pair in z_sections.windows(2) {
            let z_lo = pair[0];
            let z_hi = pair[1];

            let contours_lo = &by_z[&z_lo];
            let contours_hi = &by_z[&z_hi];

            // For each combination of contours on adjacent sections, skin them
            // In typical usage there is one contour per section per object, but
            // we handle multiple by connecting nearest pairs.
            for &ci_lo in contours_lo {
                let cont_lo = &obj.contours[ci_lo];
                let c2d_lo = Contour2d {
                    points: cont_lo.points.iter().map(|p| [p.x, p.y]).collect(),
                    z: cont_lo.points[0].z,
                    closed: !is_open,
                };

                // Find the closest contour on the upper section
                let best_hi = contours_hi
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let ca = &obj.contours[a];
                        let cb = &obj.contours[b];
                        let da = centroid_dist(&cont_lo.points, &ca.points);
                        let db = centroid_dist(&cont_lo.points, &cb.points);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(ci_hi) = best_hi {
                    let cont_hi = &obj.contours[ci_hi];
                    let c2d_hi = Contour2d {
                        points: cont_hi.points.iter().map(|p| [p.x, p.y]).collect(),
                        z: cont_hi.points[0].z,
                        closed: !is_open,
                    };

                    let cm = skin_contours(&c2d_lo, &c2d_hi);

                    if cm.indices.is_empty() {
                        continue;
                    }

                    // Convert ContourMesh to ImodMesh format.
                    // IMOD mesh indices: vertex indices in groups of 3 (triangles),
                    // terminated by -1.
                    let mut indices: Vec<i32> = Vec::with_capacity(cm.indices.len() + 1);
                    for &idx in &cm.indices {
                        indices.push(idx as i32);
                    }
                    indices.push(-1); // end marker

                    new_meshes.push(ImodMesh {
                        vertices: cm.vertices,
                        indices,
                        ..Default::default()
                    });
                }
            }
        }

        let n_new = new_meshes.len();
        if !args.append {
            model.objects[ob].meshes.clear();
        }
        model.objects[ob].meshes.extend(new_meshes);
        println!(
            "Object {}: created {} mesh(es) from {} Z sections",
            ob + 1,
            n_new,
            z_sections.len()
        );
    }

    let outpath = args.output.as_deref().unwrap_or(&args.input);
    if let Err(e) = write_model(outpath, &model) {
        eprintln!("ERROR: imodmesh - error writing {}: {}", outpath, e);
        process::exit(1);
    }
    println!("Wrote {}", outpath);
}

/// Squared distance between the centroids of two point sets.
fn centroid_dist(a: &[imod_core::Point3f], b: &[imod_core::Point3f]) -> f32 {
    let ca = centroid(a);
    let cb = centroid(b);
    let dx = ca.0 - cb.0;
    let dy = ca.1 - cb.1;
    dx * dx + dy * dy
}

fn centroid(pts: &[imod_core::Point3f]) -> (f32, f32) {
    if pts.is_empty() {
        return (0.0, 0.0);
    }
    let n = pts.len() as f32;
    let sx: f32 = pts.iter().map(|p| p.x).sum();
    let sy: f32 = pts.iter().map(|p| p.y).sum();
    (sx / n, sy / n)
}
