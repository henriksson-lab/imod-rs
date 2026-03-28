use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::io::{BufRead, BufReader};

/// Normalize density between sections using exposure dose data,
/// optionally taking the log.  Can output weighting factors for Tilt.
#[derive(Parser)]
#[command(name = "densnorm", about = "Normalize density between sections")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: Option<String>,

    /// Output MRC image file
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Weight output file (text file of normalization factors)
    #[arg(short = 'w', long)]
    weight: Option<String>,

    /// File containing exposure values (one per line)
    #[arg(long)]
    expfile: Option<String>,

    /// Tilt file (angles; weights computed as 1/cos(tilt))
    #[arg(long)]
    tiltfile: Option<String>,

    /// Reference image file for absolute normalization
    #[arg(long)]
    rifile: Option<String>,

    /// Mean of reference image (alternative to rifile)
    #[arg(long)]
    rimean: Option<f32>,

    /// Exposure of reference image
    #[arg(long)]
    riexp: Option<f32>,

    /// Take log of output (value = base added before log)
    #[arg(long)]
    log: Option<f32>,

    /// Output mode (0=byte, 1=short, 2=float, 6=ushort)
    #[arg(long)]
    mode: Option<i32>,

    /// Scaling factor for output
    #[arg(long)]
    scale: Option<f32>,

    /// Reverse contrast
    #[arg(long, default_value = "false")]
    reverse: bool,

    /// Ignore exposure data in header
    #[arg(long, default_value = "false")]
    ignore: bool,

    /// Inverse cosine power for tilt-based weighting
    #[arg(long, default_value_t = 1)]
    power: i32,

    /// Minimum log factor (fraction of range)
    #[arg(long, default_value_t = 0.001)]
    minlog: f32,
}

fn read_values_file(path: &str) -> Vec<f32> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: DENSNORM - opening {path}: {e}");
        std::process::exit(1);
    });
    let reader = BufReader::new(f);
    reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            line.trim().parse::<f32>().ok()
        })
        .collect()
}

fn main() {
    let args = Args::parse();

    if args.output.is_none() && args.weight.is_none() {
        eprintln!("ERROR: DENSNORM - You must specify either an image or weighting output file");
        std::process::exit(1);
    }
    if args.input.is_none() && args.output.is_some() {
        eprintln!("ERROR: DENSNORM - You must specify an input image file for image output");
        std::process::exit(1);
    }

    let if_log = args.log.is_some();
    let base_log = args.log.unwrap_or(0.0);

    // Get exposure doses from one of the sources
    let mut doses: Vec<f32>;
    let mut relative = true;

    if let Some(ref exp_path) = args.expfile {
        doses = read_values_file(exp_path);
    } else if let Some(ref tilt_path) = args.tiltfile {
        let tilts = read_values_file(tilt_path);
        let cos_power = args.power;
        doses = tilts
            .iter()
            .map(|&t| 1.0 / t.to_radians().cos().powi(cos_power))
            .collect();
    } else if args.ignore {
        doses = Vec::new();
    } else {
        // No dose source provided
        doses = Vec::new();
    }

    // Check for zero doses
    for (i, &d) in doses.iter().enumerate() {
        if d == 0.0 {
            eprintln!(
                "ERROR: DENSNORM - Dose {} is zero; use --ignore to ignore doses",
                i
            );
            std::process::exit(1);
        }
    }

    // Handle reference data for absolute normalization
    let mut ref_mean: Option<f32> = args.rimean;
    let ref_dose: Option<f32> = args.riexp;

    if let Some(ref ri_path) = args.rifile {
        let mut ri_reader = MrcReader::open(ri_path).unwrap_or_else(|e| {
            eprintln!("ERROR: DENSNORM - opening reference image: {e}");
            std::process::exit(1);
        });
        let ri_h = ri_reader.header().clone();
        let ri_nx = ri_h.nx as usize;
        let ri_ny = ri_h.ny as usize;

        // Compute mean of reference image
        let mut dsum: f64 = 0.0;
        for z in 0..ri_h.nz as usize {
            let slice = ri_reader.read_slice_f32(z).unwrap();
            dsum += slice.iter().map(|&v| v as f64).sum::<f64>();
        }
        ref_mean = Some((dsum / (ri_nx as f64 * ri_ny as f64 * ri_h.nz as f64)) as f32);
    }

    if ref_mean.is_some() && ref_dose.is_some() {
        relative = false;
    }

    // Build normalization weights
    if !doses.is_empty() {
        if relative {
            // Relative: normalize so mean weight = 1
            let inv: Vec<f32> = doses.iter().map(|&d| 1.0 / d).collect();
            let sum: f32 = inv.iter().sum::<f32>() / inv.len() as f32;
            doses = inv.iter().map(|&v| v / sum).collect();
        } else {
            // Absolute weights
            let rd = ref_dose.unwrap();
            let rm = ref_mean.unwrap();
            doses = doses.iter().map(|&d| rd / (d * rm)).collect();
        }
    } else if args.output.is_some() {
        if !if_log {
            eprintln!(
                "ERROR: DENSNORM - You must enter some kind of exposure data or produce log output"
            );
            std::process::exit(1);
        }
    }

    // Write weight file if requested
    if let Some(ref wgt_path) = args.weight {
        let content: String = doses.iter().map(|d| format!("{:.10}\n", d)).collect();
        std::fs::write(wgt_path, content).unwrap_or_else(|e| {
            eprintln!("ERROR: DENSNORM - writing weight file: {e}");
            std::process::exit(1);
        });
    }

    // Done if not producing image output
    let output_path = match args.output {
        Some(ref p) => p.clone(),
        None => return,
    };
    let input_path = args.input.as_ref().unwrap();

    let mut reader = MrcReader::open(input_path).unwrap_or_else(|e| {
        eprintln!("ERROR: DENSNORM - opening input: {e}");
        std::process::exit(1);
    });
    let h = reader.header().clone();
    let _nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let dmin_in = h.amin;
    let dmax_in = h.amax;

    let mode_out = args
        .mode
        .and_then(MrcMode::from_i32)
        .unwrap_or(h.data_mode().unwrap_or(MrcMode::Float));

    // If no doses were loaded, use unit factors
    if doses.is_empty() {
        doses = vec![1.0; nz];
    }
    if doses.len() < nz {
        eprintln!("ERROR: DENSNORM - number of exposures does not match number of images");
        std::process::exit(1);
    }

    // Determine output scale
    let mut scale = if let Some(s) = args.scale {
        s
    } else if mode_out != MrcMode::Float && (!relative || if_log) {
        let mut s = 25000.0_f32;
        if if_log {
            s = 5000.0;
        }
        match mode_out {
            MrcMode::UShort => s *= 2.0,
            MrcMode::Byte => s /= 100.0,
            _ => {}
        }
        eprintln!("Output will be scaled by default factor of {s}");
        s
    } else {
        1.0
    };
    if args.reverse {
        scale = -scale;
    }

    let range_frac = args.minlog;

    let out_header = MrcHeader::new(h.nx, h.ny, h.nz, mode_out);
    let mut writer = MrcWriter::create(&output_path, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: DENSNORM - creating output: {e}");
        std::process::exit(1);
    });

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0_f64;

    for iz in 0..nz {
        let mut section = reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: DENSNORM - reading section {iz}: {e}");
            std::process::exit(1);
        });

        let dose_fac = doses[iz];
        let val_min = range_frac * (dmax_in - dmin_in) * dose_fac;

        if if_log {
            for v in section.iter_mut() {
                *v = scale * ((*v + base_log) * dose_fac).max(val_min).log10();
            }
        } else if relative {
            for v in section.iter_mut() {
                *v = *v * scale * dose_fac;
            }
        } else {
            for v in section.iter_mut() {
                *v = scale * (*v * dose_fac - 1.0);
            }
        }

        let (smin, smax, smean) = min_max_mean(&section);
        global_min = global_min.min(smin);
        global_max = global_max.max(smax);
        global_sum += smean as f64 * ny as f64;

        writer.write_slice_f32(&section).unwrap_or_else(|e| {
            eprintln!("ERROR: DENSNORM - writing section {iz}: {e}");
            std::process::exit(1);
        });
    }

    let global_mean = (global_sum / (ny as f64 * nz as f64)) as f32;

    let rel_text = if relative { "Relative" } else { "Absolute" };
    let log_text = if if_log { "logarithmic" } else { "linear" };
    eprintln!("DENSNORM: {rel_text} density normalization, {log_text}");

    writer
        .finish(global_min, global_max, global_mean)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: DENSNORM - finalizing output: {e}");
            std::process::exit(1);
        });
}
