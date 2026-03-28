use clap::Parser;
use imod_mrc::MrcReader;

/// Print header information from an MRC image file.
#[derive(Parser)]
#[command(name = "mrcinfo", about = "Print MRC file header information")]
struct Args {
    /// Input MRC file(s)
    #[arg(required = true)]
    files: Vec<String>,

    /// Print all labels
    #[arg(short = 'l', long)]
    labels: bool,

    /// Print pixel size
    #[arg(short = 'p', long)]
    pixel: bool,

    /// Print only image size as: nx ny nz
    #[arg(short = 's', long)]
    size: bool,

    /// Print only the data mode
    #[arg(short = 'm', long)]
    mode: bool,
}

fn main() {
    let args = Args::parse();

    for path in &args.files {
        let reader = match MrcReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error opening {}: {}", path, e);
                std::process::exit(1);
            }
        };

        let h = reader.header();

        if args.size {
            println!("{} {} {}", h.nx, h.ny, h.nz);
            continue;
        }

        if args.mode {
            println!("{}", h.mode);
            continue;
        }

        if args.files.len() > 1 {
            println!("\n{}:", path);
        }

        println!("  Dimensions:    {} x {} x {}", h.nx, h.ny, h.nz);
        println!(
            "  Mode:          {} ({})",
            h.mode,
            match h.data_mode() {
                Some(m) => format!("{:?}", m),
                None => "unknown".into(),
            }
        );
        println!("  Pixel size:    {:.4} x {:.4} x {:.4} Angstroms",
            h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z());
        println!("  Cell size:     {:.2} x {:.2} x {:.2}", h.xlen, h.ylen, h.zlen);
        println!("  Grid:          {} x {} x {}", h.mx, h.my, h.mz);
        println!("  Min/Max/Mean:  {:.4} / {:.4} / {:.4}", h.amin, h.amax, h.amean);
        if h.rms >= 0.0 {
            println!("  RMS:           {:.4}", h.rms);
        }
        println!("  Origin:        {:.2} {:.2} {:.2}", h.xorg, h.yorg, h.zorg);
        println!("  Map:           {} {} {}", h.mapc, h.mapr, h.maps);

        if h.next > 0 {
            println!("  Extended hdr:  {} bytes (type: {:?})", h.next, h.ext_header_type());
        }

        if h.is_imod() {
            println!("  IMOD stamp:    yes (flags: 0x{:x})", h.imod_flags);
        }

        if reader.is_swapped() {
            println!("  Byte-swapped:  yes");
        }

        if args.labels || args.pixel {
            // always show labels in verbose mode
        }

        let nlabels = h.nlabl as usize;
        if nlabels > 0 && (args.labels || !args.pixel) {
            println!("  Labels ({}):", nlabels);
            for i in 0..nlabels {
                if let Some(label) = h.label(i) {
                    println!("    {}: {}", i, label);
                }
            }
        }

        if args.pixel {
            println!("{:.6} {:.6} {:.6}", h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z());
        }
    }
}
