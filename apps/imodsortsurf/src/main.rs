use std::collections::BTreeMap;
use std::process;

use clap::Parser;
use imod_model::{ImodContour, ImodMesh, ImodObject, read_model, write_model};

/// Sort contours into surfaces based on mesh connections, or split surfaces
/// into separate objects.
#[derive(Parser)]
#[command(name = "imodsortsurf", version, about)]
struct Args {
    /// List of objects to sort (1-based, ranges allowed)
    #[arg(short = 'o')]
    objects: Option<String>,

    /// Split surfaces into new objects
    #[arg(short = 's', default_value_t = false)]
    split: bool,

    /// Use existing surface numbers instead of sorting from the mesh
    #[arg(short = 'e', default_value_t = false)]
    existing: bool,

    /// Make new objects the same color as source object
    #[arg(short = 'c', default_value_t = false)]
    keep_color: bool,

    /// Keep surface numbers after moving to new objects
    #[arg(short = 'k', default_value_t = false)]
    keep_surf: bool,

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

/// Sort contours into surfaces based on mesh connectivity.
/// Assigns surface numbers to contours based on which mesh polygon group they belong to.
fn sort_surfaces_from_mesh(obj: &mut ImodObject) -> i32 {
    if obj.meshes.is_empty() {
        return 0;
    }

    // For each contour, determine surface from the mesh that contains points
    // at the same Z. Group contours that are connected by mesh triangles.
    // Simplified: assign surface based on Z-connectivity in the mesh.

    // Collect all unique Z values from contours
    let mut contour_z: Vec<Option<i32>> = Vec::new();
    for cont in &obj.contours {
        if cont.points.is_empty() {
            contour_z.push(None);
        } else {
            contour_z.push(Some(cont.points[0].z.round() as i32));
        }
    }

    // Group contours by connectivity through mesh: simplified heuristic
    // assigns each contour a surface based on mesh surface numbers
    let mut num_surfaces = 0;
    for mesh in &obj.meshes {
        let surf = mesh.surf as i32;
        if surf >= num_surfaces {
            num_surfaces = surf + 1;
        }
    }

    // If mesh has surface info, use it to tag contours
    if num_surfaces > 0 {
        // Match contours to meshes by Z overlap
        for cont in &mut obj.contours {
            if cont.points.is_empty() {
                continue;
            }
            let cz = cont.points[0].z.round() as i32;
            // Find closest mesh surface
            let mut best_surf = 0;
            let mut best_dist = i32::MAX;
            for mesh in &obj.meshes {
                for v in &mesh.vertices {
                    let d = (v.z.round() as i32 - cz).abs();
                    if d < best_dist {
                        best_dist = d;
                        best_surf = mesh.surf as i32;
                    }
                }
            }
            cont.surf = best_surf;
        }
    }

    num_surfaces.max(1)
}

/// Split contours with different surface numbers into separate objects.
fn split_surfaces_to_objects(
    obj: &ImodObject,
    keep_color: bool,
    keep_surf: bool,
) -> Vec<ImodObject> {
    // Group contours and meshes by surface number
    let mut surf_contours: BTreeMap<i32, Vec<ImodContour>> = BTreeMap::new();
    let mut surf_meshes: BTreeMap<i32, Vec<ImodMesh>> = BTreeMap::new();

    for cont in &obj.contours {
        surf_contours
            .entry(cont.surf)
            .or_default()
            .push(cont.clone());
    }

    for mesh in &obj.meshes {
        surf_meshes
            .entry(mesh.surf as i32)
            .or_default()
            .push(mesh.clone());
    }

    // All unique surfaces
    let mut surfaces: Vec<i32> = surf_contours.keys().copied().collect();
    for k in surf_meshes.keys() {
        if !surfaces.contains(k) {
            surfaces.push(*k);
        }
    }
    surfaces.sort();

    if surfaces.len() <= 1 {
        return vec![obj.clone()];
    }

    let mut result = Vec::new();
    let mut first = true;

    for &surf in &surfaces {
        if first {
            // First surface stays in the original object (modified)
            let mut new_obj = obj.clone();
            new_obj.contours = surf_contours.remove(&surf).unwrap_or_default();
            new_obj.meshes = surf_meshes.remove(&surf).unwrap_or_default();
            if !keep_surf {
                for c in &mut new_obj.contours {
                    c.surf = 0;
                }
                for m in &mut new_obj.meshes {
                    m.surf = 0;
                }
            }
            result.push(new_obj);
            first = false;
        } else {
            let mut new_obj = ImodObject {
                name: obj.name.clone(),
                flags: obj.flags,
                axis: obj.axis,
                drawmode: obj.drawmode,
                pdrawsize: obj.pdrawsize,
                symbol: obj.symbol,
                symsize: obj.symsize,
                linewidth2: obj.linewidth2,
                linewidth: obj.linewidth,
                linestyle: obj.linestyle,
                trans: obj.trans,
                ..ImodObject::default()
            };
            if keep_color {
                new_obj.red = obj.red;
                new_obj.green = obj.green;
                new_obj.blue = obj.blue;
            }
            new_obj.contours = surf_contours.remove(&surf).unwrap_or_default();
            new_obj.meshes = surf_meshes.remove(&surf).unwrap_or_default();
            if !keep_surf {
                for c in &mut new_obj.contours {
                    c.surf = 0;
                }
                for m in &mut new_obj.meshes {
                    m.surf = 0;
                }
            }
            result.push(new_obj);
        }
    }

    result
}

fn main() {
    let args = Args::parse();

    let obj_list: Option<Vec<i32>> = args.objects.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: imodsortsurf - Parsing object list: {}", e);
            process::exit(1);
        })
    });

    if !args.split && (args.existing || args.keep_color || args.keep_surf) {
        eprintln!(
            "WARNING: imodsortsurf - -e, -c, and -k have no effect when not splitting into objects"
        );
    }

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: imodsortsurf - Reading model: {}", e);
            process::exit(1);
        }
    };

    let num_obj = model.objects.len();
    let mut new_objects: Vec<(usize, Vec<ImodObject>)> = Vec::new();

    for ob in 0..num_obj {
        let do_it = if let Some(ref list) = obj_list {
            list.contains(&((ob + 1) as i32))
        } else {
            true
        };

        if !do_it {
            continue;
        }

        let obj = &model.objects[ob];
        if obj.meshes.is_empty() && !args.existing {
            println!(
                "Object {} has no mesh data and cannot be sorted",
                ob + 1
            );
            continue;
        }

        if args.existing && args.split {
            let split = split_surfaces_to_objects(obj, args.keep_color, args.keep_surf);
            println!(
                "Object {} split into {} objects using existing surfaces",
                ob + 1,
                split.len()
            );
            new_objects.push((ob, split));
        } else {
            // Sort surfaces from mesh
            let obj_mut = &mut model.objects[ob];
            let num_surf = sort_surfaces_from_mesh(obj_mut);

            if args.split {
                let split =
                    split_surfaces_to_objects(&model.objects[ob], args.keep_color, args.keep_surf);
                println!(
                    "Object {} sorted into {} objects",
                    ob + 1,
                    split.len()
                );
                new_objects.push((ob, split));
            } else {
                println!("Object {} sorted into {} surfaces", ob + 1, num_surf);
            }
        }
    }

    // Apply splits: replace original objects and append new ones
    // Process in reverse order to maintain correct indices
    for (ob, split) in new_objects.into_iter().rev() {
        if split.is_empty() {
            continue;
        }
        // Replace original object with first split
        model.objects[ob] = split[0].clone();
        // Append remaining splits
        for extra in split.into_iter().skip(1) {
            model.objects.push(extra);
        }
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: imodsortsurf - Writing model: {}", e);
        process::exit(1);
    }
}
