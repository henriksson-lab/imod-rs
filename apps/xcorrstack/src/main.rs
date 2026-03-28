//! xcorrstack - Cross-correlate each section in a stack with a single reference
//! image, writing the correlation maps to an output stack.
//!
//! Uses FFT-based cross-correlation with optional bandpass filtering.

use clap::Parser;
use imod_core::MrcMode;
use imod_fft::cross_correlate_2d;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::process;

#[derive(Parser)]
#[command(name = "xcorrstack", about = "Cross-correlate pairs of images in a stack")]
struct Args {
    /// Input image stack file
    #[arg(short = 's', long = "stack")]
    stack: String,

    /// Single reference image file
    #[arg(short = 'r', long = "single")]
    single: String,

    /// Output file for correlation maps
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Starting and ending section numbers (0-based)
    #[arg(long = "sections", num_args = 2, value_names = ["START", "END"])]
    sections: Option<Vec<i32>>,

    /// High-pass filter cutoff (sigma1, spatial freq units)
    #[arg(long = "sigma1", default_value = "0.0")]
    sigma1: f32,

    /// Low-pass filter cutoff (sigma2, spatial freq units)
    #[arg(long = "sigma2", default_value = "0.0")]
    sigma2: f32,

    /// Filter radius1 (high-pass start)
    #[arg(long = "radius1", default_value = "0.0")]
    radius1: f32,

    /// Filter radius2 (low-pass start)
    #[arg(long = "radius2", default_value = "0.0")]
    radius2: f32,

    /// Fill value for padding
    #[arg(long = "fill")]
    fill: Option<f32>,
}

fn main() {
    let args = Args::parse();

    // Open stack and reference image
    let mut stack_reader = MrcReader::open(&args.stack).unwrap_or_else(|e| {
        eprintln!("ERROR: XCORRSTACK - opening stack {}: {}", args.stack, e);
        process::exit(1);
    });

    let mut ref_reader = MrcReader::open(&args.single).unwrap_or_else(|e| {
        eprintln!("ERROR: XCORRSTACK - opening reference {}: {}", args.single, e);
        process::exit(1);
    });

    let stack_hdr = stack_reader.header().clone();
    let ref_hdr = ref_reader.header().clone();

    let nx = stack_hdr.nx as usize;
    let ny = stack_hdr.ny as usize;
    let nz = stack_hdr.nz as usize;

    if ref_hdr.nx as usize > nx || ref_hdr.ny as usize > ny {
        eprintln!("ERROR: XCORRSTACK - reference image must not be larger than stack images");
        process::exit(1);
    }

    // Determine section range
    let (iz_start, iz_end) = if let Some(ref sec) = args.sections {
        (sec[0] as usize, sec[1] as usize)
    } else {
        (0, nz - 1)
    };

    if iz_start > iz_end || iz_end >= nz {
        eprintln!("ERROR: XCORRSTACK - section range out of bounds");
        process::exit(1);
    }

    let nz_out = iz_end + 1 - iz_start;

    // Read reference image
    let ref_data = ref_reader.read_slice_f32(0).unwrap_or_else(|e| {
        eprintln!("ERROR: XCORRSTACK - reading reference image: {}", e);
        process::exit(1);
    });

    // Pad reference to stack size if needed, or use directly
    let ref_padded = if ref_hdr.nx as usize == nx && ref_hdr.ny as usize == ny {
        ref_data
    } else {
        // Pad with fill value or mean
        let fill = args.fill.unwrap_or_else(|| {
            ref_data.iter().sum::<f32>() / ref_data.len() as f32
        });
        let mut padded = vec![fill; nx * ny];
        let rnx = ref_hdr.nx as usize;
        let rny = ref_hdr.ny as usize;
        let ox = (nx - rnx) / 2;
        let oy = (ny - rny) / 2;
        for y in 0..rny {
            for x in 0..rnx {
                padded[(y + oy) * nx + x + ox] = ref_data[y * rnx + x];
            }
        }
        padded
    };

    // Create output file
    let mut out_header = MrcHeader::new(nx as i32, ny as i32, nz_out as i32, MrcMode::Float);
    out_header.xlen = stack_hdr.xlen;
    out_header.ylen = stack_hdr.ylen;
    out_header.zlen = nz_out as f32 * stack_hdr.pixel_size_z();

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: XCORRSTACK - creating output file: {}", e);
        process::exit(1);
    });

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0f64;
    let total_pix = nx * ny * nz_out;

    // Process each section
    for iz in iz_start..=iz_end {
        eprintln!("Working on section {}", iz);

        let stack_data = stack_reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: XCORRSTACK - reading section {}: {}", iz, e);
            process::exit(1);
        });

        // Cross-correlate using FFT
        let corr = cross_correlate_2d(&stack_data, &ref_padded, nx, ny);

        // Update statistics
        for &v in &corr {
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
            global_sum += v as f64;
        }

        writer.write_slice_f32(&corr).unwrap_or_else(|e| {
            eprintln!("ERROR: XCORRSTACK - writing section: {}", e);
            process::exit(1);
        });
    }

    let global_mean = (global_sum / total_pix as f64) as f32;
    writer
        .finish(global_min, global_max, global_mean)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: XCORRSTACK - finalizing output: {}", e);
            process::exit(1);
        });

    eprintln!("Done - {} sections correlated", nz_out);
}
