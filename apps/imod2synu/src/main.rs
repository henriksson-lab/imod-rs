use clap::Parser;
use imod_model::{read_model, ImodModel};
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Convert an IMOD model file to the SYNU format used by UCSD.
///
/// Writes one SYNU file per object, with contour data for each section.
#[derive(Parser)]
#[command(name = "imod2synu", about = "Convert IMOD model to SYNU format")]
struct Args {
    /// Override Z scale
    #[arg(short = 'z')]
    zscale: Option<f64>,

    /// Input IMOD model file
    input: String,
}

fn main() {
    let args = Args::parse();

    let mut model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("Imod2synu: Couldn't read model file {}: {}", args.input, e);
        std::process::exit(10);
    });

    if let Some(zs) = args.zscale {
        model.scale.z = zs as f32;
    }

    print!("Creating synu files...");
    io::stdout().flush().ok();
    imod_to_synu(&model).unwrap_or_else(|e| {
        eprintln!("Error writing SYNU files: {}", e);
        std::process::exit(1);
    });
    println!(" Done!");
}

/// Write the model in SYNU format: one file per object.
///
/// SYNU format writes contour data with section numbers.
fn imod_to_synu(model: &ImodModel) -> io::Result<()> {
    let zscale = model.scale.z;

    for (ob, obj) in model.objects.iter().enumerate() {
        let filename = format!("object.{:03}", ob + 1);
        let f = File::create(&filename)?;
        let mut w = BufWriter::new(f);

        writeln!(w, "# SYNU format object {} from IMOD model", ob + 1)?;
        writeln!(w, "# Object name: {}", obj.name)?;
        writeln!(w, "# Color: {} {} {}", obj.red, obj.green, obj.blue)?;

        for cont in &obj.contours {
            if cont.points.is_empty() {
                continue;
            }
            // Section number from Z coordinate of first point
            let section = cont.points[0].z as i32;
            writeln!(w, "s {}", section)?;
            writeln!(w, "c {}", cont.points.len())?;

            for pt in &cont.points {
                writeln!(w, "{} {} {}", pt.x, pt.y, pt.z * zscale)?;
            }
        }
    }

    Ok(())
}
