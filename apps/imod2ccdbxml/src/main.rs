use clap::Parser;
use imod_model::{read_model, ImodModel, ImodObject};
use std::fs::File;
use std::io::{self, BufWriter, Write};

// IMOD object flag bits
const IMOD_OBJFLAG_OFF: u32 = 1 << 3;
// const IMOD_OBJFLAG_SCAT: u32 = 1 << 9; // reserved for future use

// Contour flag
const ICONT_OPEN: u32 = 1 << 3;

/// Convert an IMOD model to CCDB Annotation XML format.
///
/// CCDB (Cell Centered Database) Annotation XML stores contour-based
/// annotations with geometry data.
#[derive(Parser)]
#[command(name = "imod2ccdbxml", about = "Convert IMOD model to CCDB Annotation XML")]
struct Args {
    /// Output all objects (by default those switched off are omitted)
    #[arg(short = 'a')]
    all_objects: bool,

    /// Flip Y values to match ccdbXML's default coordinate system
    #[arg(short = 'f')]
    flip_y: bool,

    /// Output points using individual POINT tags
    #[arg(short = 'p')]
    points_separate: bool,

    /// Use object label values for ONTO_URI tags
    #[arg(short = 'l')]
    use_labels: bool,

    /// Include empty contours
    #[arg(short = 'e')]
    include_empty: bool,

    /// Input IMOD model file
    input: String,

    /// Output XML file
    output: String,
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("Error reading imod model {}: {}", args.input, e);
        std::process::exit(3);
    });

    let fout = File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("Couldn't open output file {}: {}", args.output, e);
        std::process::exit(10);
    });
    let mut w = BufWriter::new(fout);

    imod_to_ccdbxml(&model, &mut w, &args).unwrap_or_else(|e| {
        eprintln!("Error writing CCDB XML: {}", e);
        std::process::exit(1);
    });
}

fn xml_safe_name(obj: &ImodObject, ob: usize) -> String {
    let mut name = format!("obj{}_", ob + 1);
    for c in obj.name.chars() {
        match c {
            ' ' | '\t' | '\n' => name.push('_'),
            '<' => name.push(']'),
            '>' => name.push('>'),
            '\'' | '"' => name.push('`'),
            '#' | '.' | ',' | '\\' | ':' => name.push('_'),
            _ => name.push(c),
        }
    }
    name
}

fn get_xyz(model: &ImodModel, pt: &imod_core::Point3f, flip_y: bool) -> (f32, f32, f32) {
    let x = pt.x;
    let mut y = pt.y;
    let z = pt.z;
    if flip_y {
        y = model.ymax as f32 - y;
    }
    (x, y, z)
}

fn imod_to_ccdbxml(model: &ImodModel, w: &mut impl Write, args: &Args) -> io::Result<()> {
    writeln!(w, "<?xml version=\"1.0\" encoding=\"utf-8\"?>")?;
    writeln!(
        w,
        "<file type=\"CCDBAnnotationSchema\" file_version=\"1.0\">"
    )?;
    writeln!(w, "<ANNOTATION>")?;
    writeln!(w)?;
    writeln!(
        w,
        "  <RESOURCES dataset_id=\"unknown\" filepath=\"unknown\">"
    )?;
    if !args.flip_y {
        writeln!(
            w,
            "  <COORDINATE_ORIGIN x=\"left\" y=\"bottom\" z=\"bottom\"/>"
        )?;
        writeln!(
            w,
            "  <COORDINATE_DIRECTION x=\"right\" y=\"up\" z=\"up\"/>"
        )?;
    }
    writeln!(w)?;

    for (ob, obj) in model.objects.iter().enumerate() {
        print_object(model, ob, obj, w, args)?;
    }

    writeln!(w)?;
    writeln!(w, "</ANNOTATION>")?;
    writeln!(w, "</file>")?;
    Ok(())
}

fn print_object(
    model: &ImodModel,
    ob: usize,
    obj: &ImodObject,
    w: &mut impl Write,
    args: &Args,
) -> io::Result<()> {
    if !args.all_objects && (obj.flags & IMOD_OBJFLAG_OFF) != 0 {
        return Ok(());
    }

    let safe_name = xml_safe_name(obj, ob);
    writeln!(w)?;
    writeln!(w)?;
    writeln!(w, "<!-- #DATA FOR OBJECT {}-->", safe_name)?;
    writeln!(w)?;

    writeln!(
        w,
        "<GEOMETRY user_name=\"guest\" modified_time=\"1307058082\" program=\"IMOD\">"
    )?;

    // Determine if the object is closed
    let is_obj_closed = (obj.flags & (1 << 2)) != 0; // IobjFlagClosed bit

    for cont in &obj.contours {
        if !args.include_empty && cont.points.is_empty() {
            continue;
        }

        writeln!(w, "  <POLYGON>")?;

        if args.points_separate {
            for pt in &cont.points {
                let (x, y, z) = get_xyz(model, pt, args.flip_y);
                writeln!(w, "    <POINT>{},{},{}</POINT>", x, y, z)?;
            }
        } else {
            let is_cont_closed =
                is_obj_closed && (cont.flags & ICONT_OPEN) == 0;

            if is_cont_closed {
                if !cont.points.is_empty() {
                    writeln!(w, "    <Z_VALUE>{}</Z_VALUE>", cont.points[0].z)?;
                }
                write!(w, "    <LINESTRING_2D>")?;
                for (p, pt) in cont.points.iter().enumerate() {
                    if p > 0 {
                        write!(w, ",")?;
                    }
                    let (x, y, _z) = get_xyz(model, pt, args.flip_y);
                    write!(w, "{} {}", x, y)?;
                }
                writeln!(w, "</LINESTRING_2D>")?;
            } else {
                write!(w, "    <LINESTRING_3D>")?;
                for (p, pt) in cont.points.iter().enumerate() {
                    if p > 0 {
                        write!(w, ",")?;
                    }
                    let (x, y, z) = get_xyz(model, pt, args.flip_y);
                    write!(w, "{} {} {}", x, y, z)?;
                }
                writeln!(w, "</LINESTRING_3D>")?;
            }
        }

        writeln!(w, "  </POLYGON>")?;
    }

    writeln!(w, "</GEOMETRY>")?;
    writeln!(w)?;
    Ok(())
}
