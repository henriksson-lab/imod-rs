use std::process;

use clap::Parser;
use imod_model::{read_model, write_model};

/// Join (merge) multiple IMOD model files into one output model.
///
/// All objects from each input model are appended into a single output model.
/// The header values (image size, pixel size, scale, etc.) are taken from the
/// first input model.
#[derive(Parser)]
#[command(name = "imodjoin", version, about)]
struct Args {
    /// Input model files (at least one required).
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<String>,

    /// Output model file.
    #[arg(short = 'o', long = "output", required = true)]
    output: String,
}

fn main() {
    let args = Args::parse();

    if args.inputs.is_empty() {
        eprintln!("ERROR: imodjoin - no input models specified");
        process::exit(1);
    }

    // Read the first model as the base
    let mut joined = match read_model(&args.inputs[0]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "ERROR: imodjoin - error reading {}: {}",
                args.inputs[0], e
            );
            process::exit(1);
        }
    };

    let first_objs = joined.objects.len();
    println!(
        "Model 1 ({}): {} object(s)",
        args.inputs[0], first_objs
    );

    // Append objects from all subsequent models
    for (i, path) in args.inputs[1..].iter().enumerate() {
        let model = match read_model(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("ERROR: imodjoin - error reading {}: {}", path, e);
                process::exit(1);
            }
        };

        let n = model.objects.len();
        println!("Model {} ({}): {} object(s)", i + 2, path, n);

        // Update image extents to encompass all models
        joined.xmax = joined.xmax.max(model.xmax);
        joined.ymax = joined.ymax.max(model.ymax);
        joined.zmax = joined.zmax.max(model.zmax);

        for obj in model.objects {
            joined.objects.push(obj);
        }
    }

    println!(
        "Joined model: {} total object(s)",
        joined.objects.len()
    );

    if let Err(e) = write_model(&args.output, &joined) {
        eprintln!("ERROR: imodjoin - error writing {}: {}", args.output, e);
        process::exit(1);
    }
    println!("Wrote {}", args.output);
}
