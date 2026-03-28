use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodModel, ImodObject, read_model};

/// Print information about IMOD model files.
///
/// Reads one or more .mod files and prints statistics including number of
/// objects, contours per object, total points, bounding box, contour lengths,
/// and areas for closed contours.
#[derive(Parser)]
#[command(name = "imodinfo", version, about)]
struct Args {
    /// Be verbose (repeat for more detail: -v prints contour stats, -vv prints points)
    #[arg(short, action = clap::ArgAction::Count)]
    verbose: u8,

    /// IMOD model file(s) to inspect
    #[arg(required = true)]
    files: Vec<String>,
}

fn units_string(units: i32) -> &'static str {
    match units {
        0 => "pixels",
        1 => "km",
        3 => "m",
        -2 => "cm",
        -3 => "mm",
        -6 => "um",
        -9 => "nm",
        -10 => "Angstroms",
        -12 => "pm",
        _ => "unknown",
    }
}

fn contour_length(points: &[Point3f], pixel_size: f32, z_scale: f32) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0_f64;
    for i in 1..points.len() {
        let dx = (points[i].x - points[i - 1].x) as f64 * pixel_size as f64;
        let dy = (points[i].y - points[i - 1].y) as f64 * pixel_size as f64;
        let dz = (points[i].z - points[i - 1].z) as f64 * pixel_size as f64 * z_scale as f64;
        length += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    length
}

/// Compute the signed area of a contour projected onto the XY plane (in pixel coords).
/// Uses the shoelace formula. Result is in pixel^2; caller multiplies by pixel_size^2.
fn contour_area_pixels(points: &[Point3f]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += (points[i].x as f64) * (points[j].y as f64);
        area -= (points[j].x as f64) * (points[i].y as f64);
    }
    (area / 2.0).abs()
}

fn object_is_open(obj: &ImodObject) -> bool {
    (obj.flags & (1 << 3)) != 0 // IMOD_OBJFLAG_OPEN
}

fn object_is_scattered(obj: &ImodObject) -> bool {
    (obj.flags & (1 << 1)) != 0 // IMOD_OBJFLAG_SCAT
}

fn print_model_info(model: &ImodModel, path: &str, verbose: u8) {
    println!("# MODEL {}", path);
    println!("# NAME  {}", model.name);
    println!("# PIX SCALE:  x = {}", model.scale.x);
    println!("#             y = {}", model.scale.y);
    println!("#             z = {}", model.scale.z);
    println!("# PIX SIZE      = {}", model.pixel_size);
    println!("# UNITS: {}", units_string(model.units));
    println!("# IMAGE SIZE: {} x {} x {}", model.xmax, model.ymax, model.zmax);
    println!();

    if model.objects.is_empty() {
        println!("Model has no objects!!!");
        println!();
        return;
    }

    let total_contours: usize = model.objects.iter().map(|o| o.contours.len()).sum();
    let total_points: usize = model
        .objects
        .iter()
        .flat_map(|o| &o.contours)
        .map(|c| c.points.len())
        .sum();
    let total_meshes: usize = model.objects.iter().map(|o| o.meshes.len()).sum();

    println!(
        "# {} objects, {} contours, {} total points, {} meshes",
        model.objects.len(),
        total_contours,
        total_points,
        total_meshes
    );
    println!();

    // Compute bounding box across all contour points
    let mut bb_min = Point3f {
        x: f32::MAX,
        y: f32::MAX,
        z: f32::MAX,
    };
    let mut bb_max = Point3f {
        x: f32::MIN,
        y: f32::MIN,
        z: f32::MIN,
    };
    let mut has_points = false;
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                has_points = true;
                bb_min.x = bb_min.x.min(pt.x);
                bb_min.y = bb_min.y.min(pt.y);
                bb_min.z = bb_min.z.min(pt.z);
                bb_max.x = bb_max.x.max(pt.x);
                bb_max.y = bb_max.y.max(pt.y);
                bb_max.z = bb_max.z.max(pt.z);
            }
        }
    }
    if has_points {
        println!(
            "# Bounding box: ({}, {}, {}) to ({}, {}, {})",
            bb_min.x, bb_min.y, bb_min.z, bb_max.x, bb_max.y, bb_max.z
        );
        println!();
    }

    for (ob, obj) in model.objects.iter().enumerate() {
        print_object_info(model, ob, obj, verbose);
    }
}

fn print_object_info(model: &ImodModel, ob: usize, obj: &ImodObject, verbose: u8) {
    println!("OBJECT {}", ob + 1);
    println!("NAME:  {}", obj.name);
    println!("       {} contours", obj.contours.len());

    if object_is_scattered(obj) {
        println!("       object uses scattered points.");
    } else if object_is_open(obj) {
        println!("       object uses open contours.");
    } else {
        println!("       object uses closed contours.");
    }

    println!(
        "       color (red, green, blue) = ({}, {}, {})",
        obj.red, obj.green, obj.blue
    );
    if obj.meshes.len() > 0 {
        println!("       {} meshes", obj.meshes.len());
    }
    println!();

    let pixel_size = model.pixel_size;
    let z_scale = model.scale.z;
    let is_open = object_is_open(obj);
    let is_scattered = object_is_scattered(obj);

    for (co, cont) in obj.contours.iter().enumerate() {
        let npt = cont.points.len();

        if verbose >= 1 {
            print!(
                "\tCONTOUR #{},{},{}  {} points",
                co + 1,
                ob + 1,
                cont.surf,
                npt
            );
        }

        if cont.points.is_empty() {
            if verbose >= 1 {
                println!();
            }
            continue;
        }

        if verbose >= 1 && !is_scattered {
            let dist = contour_length(&cont.points, pixel_size, z_scale);
            if !is_open {
                let area = contour_area_pixels(&cont.points) * (pixel_size as f64).powi(2);
                println!(", length = {}, area = {}", dist, area);
            } else {
                println!(", length = {} {}", dist, units_string(model.units));
            }
        } else if verbose >= 1 {
            println!();
        }

        if verbose >= 2 {
            println!("\t\tx\ty\tz");
            for pt in &cont.points {
                println!("\t\t{}\t{}\t{}", pt.x, pt.y, pt.z);
            }
        }
    }

    // Per-object summary
    if !is_scattered && !is_open {
        let total_length: f64 = obj
            .contours
            .iter()
            .map(|c| contour_length(&c.points, pixel_size, z_scale))
            .sum();
        let total_area: f64 = obj
            .contours
            .iter()
            .map(|c| contour_area_pixels(&c.points) * (pixel_size as f64).powi(2))
            .sum();
        let total_pts: usize = obj.contours.iter().map(|c| c.points.len()).sum();
        println!(
            "  Total length = {:.6}, total area = {:.6}, total points = {}",
            total_length, total_area, total_pts
        );
    } else {
        let total_length: f64 = obj
            .contours
            .iter()
            .map(|c| contour_length(&c.points, pixel_size, z_scale))
            .sum();
        let total_pts: usize = obj.contours.iter().map(|c| c.points.len()).sum();
        println!(
            "  Total length = {:.6} {}, total points = {}",
            total_length,
            units_string(model.units),
            total_pts
        );
    }
    println!();
}

fn main() {
    let args = Args::parse();

    for path in &args.files {
        match read_model(path) {
            Ok(model) => {
                print_model_info(&model, path, args.verbose);
                println!();
            }
            Err(e) => {
                eprintln!("ERROR: imodinfo - error reading {}: {}", path, e);
                process::exit(1);
            }
        }
    }
}
