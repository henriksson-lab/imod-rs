use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{read_model, ImodModel, ImodObject};

/// Convert an IMOD model to QuickDraw 3D Meta file format.
///
/// Exports contours (as polylines or filled polygons), scattered points
/// (as ellipsoids), and mesh data from an IMOD model into the 3D Metafile
/// text format.
#[derive(Parser)]
#[command(name = "imod2meta", version, about)]
struct Args {
    /// Input IMOD model file
    input: String,

    /// Output 3D Meta file
    output: String,

    /// Output only this object number (0-based)
    #[arg(short = 'o', long)]
    object_only: Option<usize>,

    /// Output lines only (no filled polygons)
    #[arg(short = 'l')]
    lines_only: bool,
}

// IMOD object flag bits
const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;
const IMOD_OBJFLAG_SCAT: u32 = 1 << 9;
const IMOD_OBJFLAG_FILL: u32 = 1 << 8;

fn is_scat(flags: u32) -> bool { (flags & IMOD_OBJFLAG_SCAT) != 0 }
fn is_fill(flags: u32) -> bool { (flags & IMOD_OBJFLAG_FILL) != 0 }
fn is_close(flags: u32) -> bool { (flags & IMOD_OBJFLAG_OPEN) == 0 }

fn model_bounds(model: &ImodModel) -> (Point3f, Point3f) {
    let mut minp = Point3f { x: f32::MAX, y: f32::MAX, z: f32::MAX };
    let mut maxp = Point3f { x: f32::MIN, y: f32::MIN, z: f32::MIN };
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                minp.x = minp.x.min(pt.x);
                minp.y = minp.y.min(pt.y);
                minp.z = minp.z.min(pt.z);
                maxp.x = maxp.x.max(pt.x);
                maxp.y = maxp.y.max(pt.y);
                maxp.z = maxp.z.max(pt.z);
            }
        }
    }
    (minp, maxp)
}

fn write_attributes(w: &mut impl Write, obj: &ImodObject) {
    writeln!(w, "\tContainer (").ok();
    writeln!(w, "\t\tAttributeSet ( )").ok();
    writeln!(w, "\t\tDiffuseColor ( {} {} {} )", obj.red, obj.green, obj.blue).ok();
    writeln!(w, "\t)").ok();
}

fn write_contour_lines(w: &mut impl Write, cont: &imod_model::ImodContour, closed: bool, zscale: f32) {
    let psize = if closed { cont.points.len() + 1 } else { cont.points.len() };
    writeln!(w, "\tPolyline ( {}", psize).ok();
    for pt in &cont.points {
        writeln!(w, "\t\t{} {} {}", pt.x, pt.y, zscale * pt.z).ok();
    }
    if closed {
        let pt = &cont.points[0];
        writeln!(w, "\t\t{} {} {}", pt.x, pt.y, zscale * pt.z).ok();
    }
    writeln!(w, "\t)").ok();
}

fn write_contour_poly(w: &mut impl Write, cont: &imod_model::ImodContour, zscale: f32) {
    writeln!(w, "\tGeneralPolygon (\n\t\t1\n\t\t{}", cont.points.len()).ok();
    for pt in &cont.points {
        writeln!(w, "\t\t{} {} {}", pt.x, pt.y, zscale * pt.z).ok();
    }
    writeln!(w, "\t)").ok();
}

fn write_contour_scat(w: &mut impl Write, obj: &ImodObject, cont: &imod_model::ImodContour, zscale: f32) {
    let psize = (obj.pdrawsize as f32) * 0.1;
    for pt in &cont.points {
        writeln!(w, "Container (").ok();
        writeln!(
            w, "\tEllipsoid ( 0 0 {} {} 0 0 0 {} 0 {} {} {} )",
            psize, psize, psize, pt.x, pt.y, zscale * pt.z,
        ).ok();
        write_attributes(w, obj);
        writeln!(w, ")").ok();
    }
}

fn write_object(w: &mut impl Write, model: &ImodModel, ob: usize, lines_only: bool) {
    let obj = &model.objects[ob];
    let zscale = model.scale.z;

    for cont in &obj.contours {
        if cont.points.is_empty() {
            continue;
        }

        if is_close(obj.flags) {
            writeln!(w, "Container (").ok();
            if is_fill(obj.flags) && !lines_only {
                write_contour_poly(w, cont, zscale);
            } else {
                write_contour_lines(w, cont, true, zscale);
            }
            write_attributes(w, obj);
            writeln!(w, ")").ok();
            continue;
        }

        if is_scat(obj.flags) {
            write_contour_scat(w, obj, cont, zscale);
            continue;
        }

        writeln!(w, "Container (").ok();
        write_contour_lines(w, cont, false, zscale);
        write_attributes(w, obj);
        writeln!(w, ")").ok();
    }
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2meta - Reading model {}: {}", args.input, e);
        process::exit(3);
    });

    let out_file = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: imod2meta - Could not open {}: {}", args.output, e);
        process::exit(10);
    });
    let mut w = BufWriter::new(out_file);

    let zscale = model.scale.z;
    let (min, max) = model_bounds(&model);
    let mean = Point3f {
        x: (max.x + min.x) * 0.5,
        y: (max.y + min.y) * 0.5,
        z: (max.z + min.z) * 0.5,
    };

    writeln!(w, "3DMetafile (1 0 Stream Label0> )").ok();
    writeln!(w, "\n#Created by imod2meta\n").ok();
    writeln!(w, "BeginGroup ( DisplayGroup( ) )").ok();
    writeln!(w, "Translate ( {} {} {} )", -mean.x, -mean.y, -zscale * mean.z).ok();

    if let Some(ob) = args.object_only {
        if ob < model.objects.len() {
            eprintln!("object {}", ob);
            write_object(&mut w, &model, ob, args.lines_only);
        } else {
            eprintln!("ERROR: imod2meta - Object {} out of range", ob);
            process::exit(1);
        }
    } else {
        for ob in 0..model.objects.len() {
            write_object(&mut w, &model, ob, args.lines_only);
        }
    }

    writeln!(w, "EndGroup ( )").ok();
}
