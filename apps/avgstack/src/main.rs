use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Average sections from an MRC stack into a single output image.
#[derive(Parser)]
#[command(name = "avgstack", about = "Average sections in an MRC stack")]
struct Args {
    /// Input MRC stack file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file (single averaged section)
    #[arg(short = 'o', long)]
    output: String,

    /// First section to average (0-based, default: 0)
    #[arg(short = 'f', long, default_value_t = 0)]
    first: usize,

    /// Last section to average (0-based, default: last section)
    #[arg(short = 'l', long)]
    last: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: AVGSTACK - opening input: {e}");
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let mode = h.data_mode().unwrap_or_else(|| {
        eprintln!("ERROR: AVGSTACK - unsupported input mode {}", h.mode);
        std::process::exit(1);
    });

    if matches!(mode, MrcMode::ComplexFloat | MrcMode::ComplexShort) {
        eprintln!("ERROR: AVGSTACK - use the clip program to average FFTs");
        std::process::exit(1);
    }

    let npix = nx * ny;
    let ifst = args.first.min(nz - 1);
    let ilst = args.last.unwrap_or(nz - 1).min(nz - 1).max(ifst);
    let numsec = ilst - ifst + 1;

    eprintln!("avgstack: averaging sections {ifst} to {ilst} ({numsec} sections)");

    // Accumulate sum in f64 for precision
    let mut sum = vec![0.0_f64; npix];

    for z in ifst..=ilst {
        eprintln!(" Adding section {z}");
        let data = reader.read_slice_f32(z).unwrap_or_else(|e| {
            eprintln!("ERROR: AVGSTACK - reading section {z}: {e}");
            std::process::exit(1);
        });
        for (s, &v) in sum.iter_mut().zip(data.iter()) {
            *s += v as f64;
        }
    }

    // Compute average
    let inv = 1.0 / numsec as f64;
    let avg: Vec<f32> = sum.iter().map(|&s| (s * inv) as f32).collect();

    let (dmin, dmax, dmean) = min_max_mean(&avg);

    // Write output: single section, mode 2 (float)
    let mut out_header = MrcHeader::new(nx as i32, ny as i32, 1, MrcMode::Float);
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = 1;
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    // Set Z cell size to one pixel thickness
    if h.mx > 0 {
        out_header.zlen = h.xlen / h.mx as f32;
    }
    out_header.add_label(&format!("avgstack: {numsec} sections averaged"));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: AVGSTACK - creating output: {e}");
        std::process::exit(1);
    });

    writer.write_slice_f32(&avg).unwrap_or_else(|e| {
        eprintln!("ERROR: AVGSTACK - writing output: {e}");
        std::process::exit(1);
    });

    writer.finish(dmin, dmax, dmean).unwrap();
    eprintln!(
        "avgstack: done. Min={dmin:.5e} Max={dmax:.5e} Mean={dmean:.5e}"
    );
}
