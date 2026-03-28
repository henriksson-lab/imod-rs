use clap::Parser;
use imod_mrc::MrcReader;

/// Print and query MRC file header information.
///
/// Similar to mrcinfo but with more options for querying specific fields
/// in a machine-readable format, matching IMOD's `header` program.
#[derive(Parser)]
#[command(name = "header", about = "Print/query MRC header values")]
struct Args {
    /// Input MRC file(s)
    #[arg(required = true)]
    files: Vec<String>,

    /// Print only image size (nx ny nz)
    #[arg(short = 's', long)]
    size: bool,

    /// Print only the data mode
    #[arg(short = 'm', long)]
    mode: bool,

    /// Print only pixel size (dx dy dz)
    #[arg(short = 'p', long)]
    pixel: bool,

    /// Print only the origin (xorg yorg zorg)
    #[arg(short = 'o', long)]
    origin: bool,

    /// Print only the minimum density value
    #[arg(long)]
    min: bool,

    /// Print only the maximum density value
    #[arg(long)]
    max: bool,

    /// Print only the mean density value
    #[arg(long)]
    mean: bool,

    /// Print only the RMS density value
    #[arg(long)]
    rms: bool,

    /// Brief output (one-line summary)
    #[arg(short = 'b', long)]
    brief: bool,
}

fn main() {
    let args = Args::parse();
    let silent = args.size || args.mode || args.pixel || args.origin
        || args.min || args.max || args.mean || args.rms;

    for path in &args.files {
        let reader = match MrcReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error opening {}: {}", path, e);
                std::process::exit(1);
            }
        };

        let h = reader.header();

        // Silent/query mode: print only requested fields
        if silent {
            if args.size {
                println!("{:>8}{:>8}{:>8}", h.nx, h.ny, h.nz);
            }
            if args.mode {
                println!("{:>4}", h.mode);
            }
            if args.pixel {
                println!(
                    "{:>15.5}{:>15.5}{:>15.5}",
                    h.pixel_size_x(),
                    h.pixel_size_y(),
                    h.pixel_size_z()
                );
            }
            if args.origin {
                println!(
                    "{:>15.5}{:>15.5}{:>15.5}",
                    h.xorg, h.yorg, h.zorg
                );
            }
            if args.min {
                println!("{:>13.5}", h.amin);
            }
            if args.max {
                println!("{:>13.5}", h.amax);
            }
            if args.mean {
                println!("{:>13.5}", h.amean);
            }
            if args.rms {
                if h.rms >= 0.0 {
                    println!("{:>13.5}", h.rms);
                } else {
                    println!("{:>13.5} (not computed)", h.rms);
                }
            }
            continue;
        }

        // Brief mode
        if args.brief {
            println!(
                "{}: {} x {} x {}  mode {}  pixel {:.4} {:.4} {:.4}  min/max/mean {:.4}/{:.4}/{:.4}",
                path,
                h.nx, h.ny, h.nz,
                h.mode,
                h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z(),
                h.amin, h.amax, h.amean
            );
            continue;
        }

        // Full output
        if args.files.len() > 1 {
            println!("\n{}:", path);
        }

        println!();
        println!(" Number of columns, rows, sections .....  {:>8}  {:>8}  {:>8}", h.nx, h.ny, h.nz);
        println!(" Map mode ..............................  {:>8}", h.mode);
        match h.data_mode() {
            Some(m) => println!("   ({:?})", m),
            None => println!("   (unknown mode)"),
        }
        println!(" Start cols, rows, sects ...............  {:>8}  {:>8}  {:>8}", h.nxstart, h.nystart, h.nzstart);
        println!(" Grid size .............................  {:>8}  {:>8}  {:>8}", h.mx, h.my, h.mz);
        println!(" Cell size (Angstroms) .................  {:>12.4}  {:>12.4}  {:>12.4}", h.xlen, h.ylen, h.zlen);
        println!(" Cell angles ...........................  {:>12.4}  {:>12.4}  {:>12.4}", h.alpha, h.beta, h.gamma);
        println!(" Axis mapping ..........................  {:>8}  {:>8}  {:>8}", h.mapc, h.mapr, h.maps);
        println!(" Minimum density .......................  {:>16.4}", h.amin);
        println!(" Maximum density .......................  {:>16.4}", h.amax);
        println!(" Mean density ..........................  {:>16.4}", h.amean);
        if h.rms >= 0.0 {
            println!(" RMS deviation .........................  {:>16.4}", h.rms);
        } else {
            println!(" RMS deviation .........................  {:>16.4} (not computed)", h.rms);
        }
        println!(" Space group ...........................  {:>8}", h.ispg);
        if h.next > 0 {
            println!(" Extended header size (bytes) ...........  {:>8}", h.next);
            println!(" Extended header type ...................  {:?}", h.ext_header_type());
        }

        println!(" Pixel size (Angstroms) ................  {:>12.6}  {:>12.6}  {:>12.6}",
            h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z());
        println!(" Origin ................................  {:>12.3}  {:>12.3}  {:>12.3}", h.xorg, h.yorg, h.zorg);

        if h.tilt_angles.iter().any(|&t| t != 0.0) {
            println!(" Tilt angles (orig) ....................  {:>8.2}  {:>8.2}  {:>8.2}",
                h.tilt_angles[0], h.tilt_angles[1], h.tilt_angles[2]);
            println!(" Tilt angles (curr) ....................  {:>8.2}  {:>8.2}  {:>8.2}",
                h.tilt_angles[3], h.tilt_angles[4], h.tilt_angles[5]);
        }

        if h.is_imod() {
            println!(" IMOD stamp ............................  present (flags: 0x{:x})", h.imod_flags);
        }

        if reader.is_swapped() {
            println!(" Byte order ............................  swapped");
        }

        let nlabels = h.nlabl as usize;
        if nlabels > 0 {
            println!();
            println!(" Number of labels: {}", nlabels);
            for i in 0..nlabels {
                if let Some(label) = h.label(i) {
                    println!("  {}: {}", i, label);
                }
            }
        }

        println!();
    }
}
