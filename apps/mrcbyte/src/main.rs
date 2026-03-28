use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Convert an MRC file to byte (mode 0) with optional contrast scaling.
#[derive(Parser)]
#[command(name = "mrcbyte", about = "Convert MRC file to byte mode")]
struct Args {
    /// Input MRC file
    input: String,

    /// Output MRC file
    output: String,

    /// Black level (input value that maps to 0)
    #[arg(short = 'b', long, default_value_t = f32::NAN)]
    black: f32,

    /// White level (input value that maps to 255)
    #[arg(short = 'w', long, default_value_t = f32::NAN)]
    white: f32,

    /// Contrast ramp: scale by mean +/- N standard deviations
    #[arg(short = 'c', long)]
    contrast: Option<f32>,
}

fn main() {
    let args = Args::parse();

    let mut reader = match MrcReader::open(&args.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error opening {}: {}", args.input, e);
            std::process::exit(1);
        }
    };

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    // Determine scaling range
    let (black, white) = if let Some(nsigma) = args.contrast {
        // Compute global statistics from first slice sample
        let slice0 = reader.read_slice_f32(0).unwrap();
        let (mean, sd) = imod_math::mean_sd(&slice0);
        let black = mean - nsigma * sd;
        let white = mean + nsigma * sd;
        eprintln!("Contrast: mean={:.2}, sd={:.2}, black={:.2}, white={:.2}", mean, sd, black, white);
        (black, white)
    } else if args.black.is_nan() || args.white.is_nan() {
        // Auto-scale from header min/max
        let black = if args.black.is_nan() { h.amin } else { args.black };
        let white = if args.white.is_nan() { h.amax } else { args.white };
        (black, white)
    } else {
        (args.black, args.white)
    };

    let range = white - black;
    let scale = if range.abs() > 1e-10 { 255.0 / range } else { 1.0 };

    // Create output
    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Byte);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.xorg = h.xorg;
    out_header.yorg = h.yorg;
    out_header.zorg = h.zorg;

    let label = format!("mrcbyte: Scaled {:.2} to {:.2} -> 0 to 255", black, white);
    out_header.add_label(&label);

    let mut writer = match MrcWriter::create(&args.output, out_header) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error creating {}: {}", args.output, e);
            std::process::exit(1);
        }
    };

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0_f64;
    let total_pix = (nx * ny * nz) as f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let scaled: Vec<f32> = data
            .iter()
            .map(|&v| ((v - black) * scale).clamp(0.0, 255.0))
            .collect();

        let (smin, smax, smean) = min_max_mean(&scaled);
        if smin < global_min { global_min = smin; }
        if smax > global_max { global_max = smax; }
        global_sum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&scaled).unwrap();
    }

    let global_mean = (global_sum / total_pix) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap();

    eprintln!(
        "Converted {} ({} x {} x {}, mode {}) -> {} (byte)",
        args.input, nx, ny, nz, h.mode, args.output
    );
}
