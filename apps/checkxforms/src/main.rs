//! checkxforms - Check transform files for outliers by reporting maximum
//! rotation and maximum displacement across all transforms.
//!
//! For each input file, reports the number of transforms, maximum rotation
//! angle with its section number, and maximum X/Y displacement with its
//! section number.

use clap::Parser;
use imod_transforms::read_xf_file;

#[derive(Parser)]
#[command(name = "checkxforms", about = "Check transform file(s) for outliers")]
struct Args {
    /// Input transform file(s)
    #[arg(required = true)]
    files: Vec<String>,
}

fn main() {
    let args = Args::parse();

    println!(
        "{:>22}{:>8}{:>13}{:>10}{:>14}{:>10}",
        "", "# of", "maximum", "at", "maximum", "at"
    );
    println!(
        "{:>20}{:>10}{:>13}{:>10}{:>14}{:>10}",
        "", "sections", "rotation", "section", "X/Y offset", "section"
    );

    for file in &args.files {
        let transforms = match read_xf_file(file) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error reading {}: {}", file, e);
                continue;
            }
        };

        if transforms.is_empty() {
            eprintln!("No transforms in {}", file);
            continue;
        }

        let nf = transforms.len();
        let mut max_rot = 0.0f32;
        let mut sec_rot = 0usize;
        let mut max_dxy = 0.0f32;
        let mut sec_dxy = 0usize;

        for (i, xf) in transforms.iter().enumerate() {
            let rot = xf.rotation_angle().abs();
            if rot > max_rot {
                max_rot = rot;
                sec_rot = i;
            }
            let dx_abs = xf.dx.abs();
            let dy_abs = xf.dy.abs();
            if dx_abs > max_dxy {
                max_dxy = dx_abs;
                sec_dxy = i;
            }
            if dy_abs > max_dxy {
                max_dxy = dy_abs;
                sec_dxy = i;
            }
        }

        println!(
            "{:>25}{:>13.1}{:>8}{:>14.1}{:>9}",
            nf, max_rot, sec_rot, max_dxy, sec_dxy
        );
    }
}
