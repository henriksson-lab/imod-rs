//! taperprep - Prepare subvolume parameters and compute padded output size.
//!
//! This is a utility used by taperoutvol and combinefft. It reads an MRC file,
//! takes subvolume extent and taper/pad parameters, and computes the output
//! dimensions (optionally sized for FFT) and origin adjustments.
//!
//! Translated from IMOD's taperprep.f90

use clap::Parser;
use imod_mrc::MrcReader;
use std::process;

#[derive(Parser)]
#[command(
    name = "taperprep",
    about = "Compute padded output dimensions for subvolume extraction with taper"
)]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// X range (min max), default: full extent
    #[arg(long, num_args = 2, value_names = ["XMIN", "XMAX"])]
    xminmax: Option<Vec<i32>>,

    /// Y range (min max), default: full extent
    #[arg(long, num_args = 2, value_names = ["YMIN", "YMAX"])]
    yminmax: Option<Vec<i32>>,

    /// Z range (min max), default: full extent
    #[arg(long, num_args = 2, value_names = ["ZMIN", "ZMAX"])]
    zminmax: Option<Vec<i32>>,

    /// Taper/pad widths in X, Y, Z
    #[arg(long, num_args = 3, value_names = ["PX", "PY", "PZ"], default_values_t = [0, 0, 0])]
    taper: Vec<i32>,

    /// Do not increase sizes for FFT (just add padding)
    #[arg(long, default_value_t = false)]
    no_fft: bool,
}

fn nice_fft_size(n: i32) -> i32 {
    let mut size = n.max(2);
    if size % 2 != 0 {
        size += 1;
    }
    loop {
        let mut m = size;
        while m % 2 == 0 {
            m /= 2;
        }
        while m % 3 == 0 {
            m /= 3;
        }
        while m % 5 == 0 {
            m /= 5;
        }
        if m == 1 {
            return size;
        }
        size += 2;
    }
}

fn main() {
    let args = Args::parse();

    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: taperprep - opening input: {}", e);
        process::exit(1);
    });
    let header = reader.header();
    let nx = header.nx;
    let ny = header.ny;
    let nz = header.nz;

    let ix_low = args.xminmax.as_ref().map_or(0, |v| v[0]);
    let ix_high = args.xminmax.as_ref().map_or(nx - 1, |v| v[1]);
    let iy_low = args.yminmax.as_ref().map_or(0, |v| v[0]);
    let iy_high = args.yminmax.as_ref().map_or(ny - 1, |v| v[1]);
    let iz_low = args.zminmax.as_ref().map_or(0, |v| v[0]);
    let iz_high = args.zminmax.as_ref().map_or(nz - 1, |v| v[1]);

    if ix_low < 0 || ix_high >= nx || iy_low < 0 || iy_high >= ny || iz_low < 0 || iz_high >= nz {
        eprintln!("ERROR: taperprep - block not all inside volume");
        process::exit(1);
    }

    let nx_box = ix_high + 1 - ix_low;
    let ny_box = iy_high + 1 - iy_low;
    let nz_box = iz_high + 1 - iz_low;

    let num_pad_x = args.taper[0];
    let num_pad_y = args.taper[1];
    let num_pad_z = args.taper[2];

    let (nx_out, ny_out, nz_out) = if args.no_fft {
        (
            nx_box + 2 * num_pad_x,
            ny_box + 2 * num_pad_y,
            nz_box + 2 * num_pad_z,
        )
    } else {
        let nxo = nice_fft_size(2 * ((nx_box + 1) / 2 + num_pad_x));
        let nyo = nice_fft_size(2 * ((ny_box + 1) / 2 + num_pad_y));
        let nzo = if nz_box > 1 || num_pad_z > 0 {
            nice_fft_size(2 * ((nz_box + 1) / 2 + num_pad_z))
        } else {
            nz_box
        };
        (nxo, nyo, nzo)
    };

    let delta_x = header.xlen / header.mx as f32;
    let delta_y = header.ylen / header.my as f32;
    let delta_z = header.zlen / header.mz as f32;

    let origin_x = header.xorg - delta_x * (ix_low as f32 - (nx_out - nx_box) as f32 / 2.0);
    let origin_y = header.yorg - delta_y * (iy_low as f32 - (ny_out - ny_box) as f32 / 2.0);
    let origin_z = header.zorg - delta_z * (iz_low as f32 - (nz_out - nz_box) as f32 / 2.0);

    println!("Input size:  {} x {} x {}", nx, ny, nz);
    println!("Box size:    {} x {} x {}", nx_box, ny_box, nz_box);
    println!("Output size: {} x {} x {}", nx_out, ny_out, nz_out);
    println!(
        "Cell:        {:.3} x {:.3} x {:.3}",
        nx_out as f32 * delta_x,
        ny_out as f32 * delta_y,
        nz_out as f32 * delta_z
    );
    println!(
        "Origin:      {:.3} {:.3} {:.3}",
        origin_x, origin_y, origin_z
    );
}
