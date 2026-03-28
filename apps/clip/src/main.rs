use clap::{Parser, Subcommand};
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;

/// Command Line Image Processing for MRC files.
#[derive(Parser)]
#[command(name = "clip", about = "Command line image processing")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print statistics (min, max, mean, SD) for each section
    Stats {
        /// Input MRC file
        input: String,
    },
    /// Flip image (mirror along an axis)
    Flip {
        /// Axis to flip: x, y, or z
        axis: String,
        /// Input MRC file
        input: String,
        /// Output MRC file
        output: String,
    },
    /// Multiply two images
    Multiply {
        /// First input MRC file
        input1: String,
        /// Second input MRC file
        input2: String,
        /// Output MRC file
        output: String,
    },
    /// Add two images
    Add {
        /// First input MRC file
        input1: String,
        /// Second input MRC file
        input2: String,
        /// Output MRC file
        output: String,
    },
    /// Resize (extract subregion or pad)
    Resize {
        /// Input MRC file
        input: String,
        /// Output MRC file
        output: String,
        /// Output X size
        #[arg(short = 'x')]
        nx: usize,
        /// Output Y size
        #[arg(short = 'y')]
        ny: usize,
    },
    /// Apply a median filter
    Median {
        /// Input MRC file
        input: String,
        /// Output MRC file
        output: String,
    },
    /// Compute the gradient (edge detection)
    Gradient {
        /// Input MRC file
        input: String,
        /// Output MRC file
        output: String,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Stats { input } => cmd_stats(&input),
        Command::Flip { axis, input, output } => cmd_flip(&axis, &input, &output),
        Command::Multiply { input1, input2, output } => cmd_binop(&input1, &input2, &output, "multiply"),
        Command::Add { input1, input2, output } => cmd_binop(&input1, &input2, &output, "add"),
        Command::Resize { input, output, nx, ny } => cmd_resize(&input, &output, nx, ny),
        Command::Median { input, output } => cmd_filter(&input, &output, "median"),
        Command::Gradient { input, output } => cmd_filter(&input, &output, "gradient"),
    }
}

fn cmd_stats(input: &str) {
    let mut reader = open_input(input);
    let h = reader.header().clone();
    let nz = h.nz as usize;

    println!("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}", "Sec", "Min", "Max", "Mean", "SD");
    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let (min, max, mean, sd) = imod_math::min_max_mean_sd(&data);
        println!("  {:>4}  {:>12.4}  {:>12.4}  {:>12.4}  {:>12.4}", z, min, max, mean, sd);
    }
}

fn cmd_flip(axis: &str, input: &str, output: &str) {
    let mut reader = open_input(input);
    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let mut slice = Slice::from_data(nx, ny, data);

        match axis {
            "x" => {
                for y in 0..ny {
                    for x in 0..nx / 2 {
                        let a = slice.get(x, y);
                        let b = slice.get(nx - 1 - x, y);
                        slice.set(x, y, b);
                        slice.set(nx - 1 - x, y, a);
                    }
                }
            }
            "y" => {
                for y in 0..ny / 2 {
                    for x in 0..nx {
                        let a = slice.get(x, y);
                        let b = slice.get(x, ny - 1 - y);
                        slice.set(x, y, b);
                        slice.set(x, ny - 1 - y, a);
                    }
                }
            }
            _ => {
                eprintln!("Unknown flip axis: {} (use x or y)", axis);
                std::process::exit(1);
            }
        }

        let (smin, smax, smean) = min_max_mean(&slice.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&slice.data).unwrap();
    }

    let gmean = (gsum / (nx * ny * nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}

fn cmd_binop(input1: &str, input2: &str, output: &str, op: &str) {
    let mut r1 = open_input(input1);
    let mut r2 = open_input(input2);
    let h1 = r1.header().clone();
    let h2 = r2.header().clone();

    if h1.nx != h2.nx || h1.ny != h2.ny || h1.nz != h2.nz {
        eprintln!("Error: input files have different dimensions");
        std::process::exit(1);
    }

    let nx = h1.nx as usize;
    let ny = h1.ny as usize;
    let nz = h1.nz as usize;

    let out_header = MrcHeader::new(h1.nx, h1.ny, h1.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let d1 = r1.read_slice_f32(z).unwrap();
        let d2 = r2.read_slice_f32(z).unwrap();
        let s1 = Slice::from_data(nx, ny, d1);
        let s2 = Slice::from_data(nx, ny, d2);

        let result = match op {
            "multiply" => imod_slice::multiply(&s1, &s2),
            "add" => imod_slice::add(&s1, &s2),
            _ => unreachable!(),
        };

        let (smin, smax, smean) = min_max_mean(&result.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&result.data).unwrap();
    }

    let gmean = (gsum / (nx * ny * nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}

fn cmd_resize(input: &str, output: &str, out_nx: usize, out_ny: usize) {
    let mut reader = open_input(input);
    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let nz = h.nz as usize;

    let out_header = MrcHeader::new(out_nx as i32, out_ny as i32, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    // Center the input in the output
    let ox = (out_nx as isize - in_nx as isize) / 2;
    let oy = (out_ny as isize - in_ny as isize) / 2;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let src = Slice::from_data(in_nx, in_ny, data);
        let mean = src.statistics().2;
        let mut dst = Slice::new(out_nx, out_ny, mean);

        for dy in 0..out_ny {
            for dx in 0..out_nx {
                let sx = dx as isize - ox;
                let sy = dy as isize - oy;
                if sx >= 0 && sx < in_nx as isize && sy >= 0 && sy < in_ny as isize {
                    dst.set(dx, dy, src.get(sx as usize, sy as usize));
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&dst.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&dst.data).unwrap();
    }

    let gmean = (gsum / (out_nx * out_ny * nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}

fn cmd_filter(input: &str, output: &str, filter: &str) {
    let mut reader = open_input(input);
    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let src = Slice::from_data(nx, ny, data);

        let result = match filter {
            "median" => imod_slice::median_3x3(&src),
            "gradient" => imod_slice::sobel(&src),
            _ => unreachable!(),
        };

        let (smin, smax, smean) = min_max_mean(&result.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&result.data).unwrap();
    }

    let gmean = (gsum / (nx * ny * nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}

fn open_input(path: &str) -> MrcReader {
    MrcReader::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", path, e);
        std::process::exit(1);
    })
}
