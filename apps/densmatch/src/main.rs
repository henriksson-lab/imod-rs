use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcReader, MrcWriter};

/// Scale density values in one volume so that its mean and standard deviation
/// match that of a reference volume (or a specified target mean/SD).
#[derive(Parser)]
#[command(name = "densmatch", about = "Match density (mean/SD) between volumes")]
struct Args {
    /// Reference MRC file (used to compute target mean/SD)
    #[arg(short = 'r', long)]
    reference: Option<String>,

    /// Volume to be scaled
    #[arg(short = 's', long)]
    scaled: String,

    /// Output file (if omitted, scaled file is rewritten)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Target mean and SD instead of a reference file
    #[arg(short = 't', long, num_args = 2, value_names = &["MEAN", "SD"])]
    target: Option<Vec<f32>>,

    /// Report scale factors only, without modifying data
    #[arg(long)]
    report: bool,

    /// Use all pixels for sampling instead of central eighth
    #[arg(long)]
    all: bool,

    /// Maximum number of sample pixels (default 1000000)
    #[arg(long, default_value_t = 1_000_000)]
    max_samples: usize,
}

/// Sample a volume to compute mean and SD.
/// If `use_all` is false, samples the central eighth (central half in each dimension).
/// Returns (mean, sd).
fn sample_volume(reader: &mut MrcReader, use_all: bool, max_samples: usize) -> (f32, f32) {
    let h = reader.header();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    // Determine region to sample
    let (x_start, x_use) = if use_all {
        (0, nx)
    } else {
        (nx / 4, (nx / 2).max(1))
    };
    let (y_start, y_use) = if use_all {
        (0, ny)
    } else {
        (ny / 4, (ny / 2).max(1))
    };
    let (z_start, z_use) = if use_all {
        (0, nz)
    } else {
        (nz / 4, (nz / 2).max(1))
    };

    let total_pixels = (x_use as f64) * (y_use as f64) * (z_use as f64);
    let del_sample = if use_all {
        1usize
    } else {
        ((total_pixels / max_samples as f64).cbrt() as usize).max(1)
    };

    // Ensure at least 10 samples per dimension
    let num_samp_x = ((x_use - 1) / del_sample + 1).max(x_use.min(10));
    let num_samp_y = ((y_use - 1) / del_sample + 1).max(y_use.min(10));
    let num_samp_z = ((z_use - 1) / del_sample + 1).max(z_use.min(10));

    let dx = if num_samp_x <= 1 { 1.0 } else { (x_use - 1) as f64 / (num_samp_x - 1) as f64 };
    let dy = if num_samp_y <= 1 { 1.0 } else { (y_use - 1) as f64 / (num_samp_y - 1) as f64 };
    let dz = if num_samp_z <= 1 { 1.0 } else { (z_use - 1) as f64 / (num_samp_z - 1) as f64 };

    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut count: u64 = 0;

    for jz in 0..num_samp_z {
        let iz = z_start + (jz as f64 * dz) as usize;
        let slice = reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("Error reading slice {}: {}", iz, e);
            std::process::exit(1);
        });

        for jy in 0..num_samp_y {
            let iy = y_start + (jy as f64 * dy) as usize;
            let row_start = iy * nx;

            for jx in 0..num_samp_x {
                let ix = x_start + (jx as f64 * dx) as usize;
                let val = slice[row_start + ix] as f64;
                sum += val;
                sum_sq += val * val;
                count += 1;
            }
        }
    }

    if count < 2 {
        return (sum as f32, 0.0);
    }
    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq - sum * sum / n) / (n - 1.0);
    (mean as f32, variance.max(0.0).sqrt() as f32)
}

fn main() {
    let args = Args::parse();

    // Determine target mean and SD
    let (ref_mean, ref_sd) = if let Some(ref target) = args.target {
        if args.reference.is_some() {
            eprintln!("Error: cannot specify both --reference and --target");
            std::process::exit(1);
        }
        (target[0], target[1])
    } else if let Some(ref ref_file) = args.reference {
        let mut ref_reader = MrcReader::open(ref_file).unwrap_or_else(|e| {
            eprintln!("Error opening reference file: {}", e);
            std::process::exit(1);
        });
        let (m, s) = sample_volume(&mut ref_reader, args.all, args.max_samples);
        eprintln!("Volume 1 (reference): mean = {:.4}, SD = {:.4}", m, s);
        (m, s)
    } else {
        eprintln!("Error: either --reference or --target must be specified");
        std::process::exit(1);
    };

    // Open volume to be scaled
    let mut scaled_reader = MrcReader::open(&args.scaled).unwrap_or_else(|e| {
        eprintln!("Error opening scaled file: {}", e);
        std::process::exit(1);
    });

    let (scaled_mean, scaled_sd) = sample_volume(&mut scaled_reader, args.all, args.max_samples);
    eprintln!(
        "Volume 2 (to scale): mean = {:.4}, SD = {:.4}",
        scaled_mean, scaled_sd
    );

    if scaled_sd == 0.0 {
        eprintln!("Error: scaled volume has zero standard deviation");
        std::process::exit(1);
    }

    let scale_fac = ref_sd / scaled_sd;
    let add_fac = ref_mean - scaled_mean * scale_fac;

    if args.report {
        println!(
            "Scale factors to multiply by then add: {:14.6e} {:14.6e}",
            scale_fac, add_fac
        );
        return;
    }

    // Read header for output
    let sh = scaled_reader.header().clone();
    let nx = sh.nx as usize;
    let ny = sh.ny as usize;
    let nz = sh.nz as usize;
    let mode = sh.data_mode().unwrap_or(MrcMode::Float);

    let out_path = args.output.as_deref().unwrap_or(&args.scaled);

    let mut out_header = sh.clone();
    out_header.add_label("densmatch: Scaled volume to match another");

    let mut writer = MrcWriter::create(out_path, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating output file: {}", e);
        std::process::exit(1);
    });

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum: f64 = 0.0;
    let is_byte = mode == MrcMode::Byte;

    for iz in 0..nz {
        let mut slice = scaled_reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("Error reading slice {}: {}", iz, e);
            std::process::exit(1);
        });

        for val in slice.iter_mut() {
            let scaled = scale_fac * *val + add_fac;
            *val = if is_byte {
                scaled.clamp(0.0, 255.0)
            } else {
                scaled
            };
        }

        let (smin, smax, smean) = min_max_mean(&slice);
        global_min = global_min.min(smin);
        global_max = global_max.max(smax);
        global_sum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&slice).unwrap_or_else(|e| {
            eprintln!("Error writing slice {}: {}", iz, e);
            std::process::exit(1);
        });
    }

    let global_mean = (global_sum / (nx * ny * nz) as f64) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap_or_else(|e| {
        eprintln!("Error finishing output: {}", e);
        std::process::exit(1);
    });

    eprintln!("Density matching complete.");
}
