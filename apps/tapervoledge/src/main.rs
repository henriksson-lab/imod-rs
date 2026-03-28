//! tapervoledge - Taper volume edges.
//!
//! Cuts a subset out of an image volume, tapers the intensity down to the
//! mean value at the edges over a specified range of pixels, and embeds the
//! result into a larger volume with borders suitable for FFT.
//!
//! Translated from IMOD's tapervoledge.f

use clap::Parser;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::process;

#[derive(Parser)]
#[command(name = "tapervoledge", about = "Taper volume edges for FFT")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Size of image block in X, Y, Z (default: whole volume)
    #[arg(long, num_args = 3, value_names = ["NX", "NY", "NZ"])]
    size: Option<Vec<i32>>,

    /// Center index coordinates of block (default: center of volume)
    #[arg(long, num_args = 3, value_names = ["CX", "CY", "CZ"])]
    center: Option<Vec<i32>>,

    /// Width of pad borders in X, Y, Z
    #[arg(long, num_args = 3, value_names = ["PX", "PY", "PZ"], default_values_t = [0, 0, 0])]
    pad: Vec<i32>,

    /// Width of taper region in X, Y, Z
    #[arg(long, num_args = 3, value_names = ["TX", "TY", "TZ"], default_values_t = [0, 0, 0])]
    taper: Vec<i32>,
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

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: tapervoledge - opening input: {}", e);
        process::exit(1);
    });
    let nx = reader.header().nx;
    let ny = reader.header().ny;
    let nz = reader.header().nz;
    let mode = reader.header().mode;

    let nxbox = args.size.as_ref().map_or(nx, |v| v[0]);
    let nybox = args.size.as_ref().map_or(ny, |v| v[1]);
    let nzbox = args.size.as_ref().map_or(nz, |v| v[2]);

    let ixcen = args.center.as_ref().map_or(nx / 2, |v| v[0]);
    let iycen = args.center.as_ref().map_or(ny / 2, |v| v[1]);
    let izcen = args.center.as_ref().map_or(nz / 2, |v| v[2]);

    let ixlo = ixcen - nxbox / 2;
    let ixhi = ixlo + nxbox - 1;
    let iylo = iycen - nybox / 2;
    let iyhi = iylo + nybox - 1;
    let izlo = izcen - nzbox / 2;
    let izhi = izlo + nzbox - 1;

    if ixlo < 0 || ixhi >= nx || iylo < 0 || iyhi >= ny || izlo < 0 || izhi >= nz {
        eprintln!("ERROR: tapervoledge - block not all inside volume");
        process::exit(1);
    }

    let npadx = args.pad[0];
    let npady = args.pad[1];
    let npadz = args.pad[2];

    let nx3 = nice_fft_size(2 * ((nxbox + 1) / 2 + npadx));
    let ny3 = nice_fft_size(2 * ((nybox + 1) / 2 + npady));
    let nz3 = if nzbox == 1 {
        if npadz != 0 {
            eprintln!("ERROR: tapervoledge - no padding allowed in Z for single section");
            process::exit(1);
        }
        1
    } else {
        nice_fft_size(2 * ((nzbox + 1) / 2 + npadz))
    };

    let nxtap = args.taper[0].max(0).min(nxbox / 2);
    let nytap = args.taper[1].max(0).min(nybox / 2);
    let nztap = args.taper[2].max(0).min(nzbox / 2);

    // Read the subvolume
    let mut volume = vec![0.0f32; (nxbox * nybox * nzbox) as usize];
    for iz in izlo..=izhi {
        let section = reader.read_slice_f32(iz as usize).unwrap_or_else(|e| {
            eprintln!("ERROR: tapervoledge - reading section {}: {}", iz, e);
            process::exit(1);
        });
        let zidx = (iz - izlo) as usize;
        for iy in iylo..=iyhi {
            let yidx = (iy - iylo) as usize;
            for ix in ixlo..=ixhi {
                let xidx = (ix - ixlo) as usize;
                volume[zidx * (nxbox * nybox) as usize + yidx * nxbox as usize + xidx] =
                    section[(iy * nx + ix) as usize];
            }
        }
    }

    // Compute mean value of the volume (for edge fill)
    let vol_sum: f64 = volume.iter().map(|&v| v as f64).sum();
    let vol_mean = (vol_sum / volume.len() as f64) as f32;

    // Create padded output volume filled with mean
    let out_size = (nx3 * ny3 * nz3) as usize;
    let mut output = vec![vol_mean; out_size];

    // Compute offsets to center the box in the padded volume
    let xoff = (nx3 - nxbox) / 2;
    let yoff = (ny3 - nybox) / 2;
    let zoff = (nz3 - nzbox) / 2;

    // Copy data into padded volume
    for iz in 0..nzbox {
        for iy in 0..nybox {
            for ix in 0..nxbox {
                let src = (iz * nybox * nxbox + iy * nxbox + ix) as usize;
                let dst = ((iz + zoff) * ny3 * nx3 + (iy + yoff) * nx3 + (ix + xoff)) as usize;
                output[dst] = volume[src];
            }
        }
    }

    // Apply taper at edges
    // Taper in X
    if nxtap > 0 {
        for iz in 0..nzbox {
            for iy in 0..nybox {
                for it in 0..nxtap {
                    let frac = (it + 1) as f32 / (nxtap + 1) as f32;
                    let _att = 1.0 - frac; // attenuation toward mean
                    // Left edge
                    let idx_l = ((iz + zoff) * ny3 * nx3 + (iy + yoff) * nx3 + (xoff + it))
                        as usize;
                    output[idx_l] = vol_mean + frac * (output[idx_l] - vol_mean);
                    // Right edge
                    let idx_r = ((iz + zoff) * ny3 * nx3
                        + (iy + yoff) * nx3
                        + (xoff + nxbox - 1 - it)) as usize;
                    output[idx_r] = vol_mean + frac * (output[idx_r] - vol_mean);
                }
            }
        }
    }

    // Taper in Y
    if nytap > 0 {
        for iz in 0..nzbox {
            for ix in 0..nxbox {
                for it in 0..nytap {
                    let frac = (it + 1) as f32 / (nytap + 1) as f32;
                    let idx_l = ((iz + zoff) * ny3 * nx3 + (yoff + it) * nx3 + (ix + xoff))
                        as usize;
                    output[idx_l] = vol_mean + frac * (output[idx_l] - vol_mean);
                    let idx_r = ((iz + zoff) * ny3 * nx3
                        + (yoff + nybox - 1 - it) * nx3
                        + (ix + xoff)) as usize;
                    output[idx_r] = vol_mean + frac * (output[idx_r] - vol_mean);
                }
            }
        }
    }

    // Taper in Z (only for 3D volumes)
    if nztap > 0 && nz3 > 1 {
        for iy in 0..nybox {
            for ix in 0..nxbox {
                for it in 0..nztap {
                    let frac = (it + 1) as f32 / (nztap + 1) as f32;
                    let idx_l = ((zoff + it) * ny3 * nx3 + (iy + yoff) * nx3 + (ix + xoff))
                        as usize;
                    output[idx_l] = vol_mean + frac * (output[idx_l] - vol_mean);
                    let idx_r = ((zoff + nzbox - 1 - it) * ny3 * nx3
                        + (iy + yoff) * nx3
                        + (ix + xoff)) as usize;
                    output[idx_r] = vol_mean + frac * (output[idx_r] - vol_mean);
                }
            }
        }
    }

    // Compute output stats
    let mut dmin = f32::MAX;
    let mut dmax = f32::MIN;
    let mut dmean_sum = 0.0f64;
    for &v in &output {
        dmin = dmin.min(v);
        dmax = dmax.max(v);
        dmean_sum += v as f64;
    }
    let dmean = (dmean_sum / out_size as f64) as f32;

    // Write output
    let out_mode = imod_core::MrcMode::from_i32(mode).unwrap_or(imod_core::MrcMode::Float);
    let out_header = MrcHeader::new(nx3, ny3, nz3, out_mode);
    let mut writer = MrcWriter::create(&args.output, out_header)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: tapervoledge - creating output: {}", e);
            process::exit(1);
        });

    for iz in 0..nz3 {
        let start = (iz * ny3 * nx3) as usize;
        let end = start + (ny3 * nx3) as usize;
        writer.write_slice_f32(&output[start..end]).unwrap_or_else(|e| {
            eprintln!("ERROR: tapervoledge - writing section: {}", e);
            process::exit(1);
        });
    }

    writer.finish(dmin, dmax, dmean).unwrap_or_else(|e| {
        eprintln!("ERROR: tapervoledge - finalizing: {}", e);
        process::exit(1);
    });

    println!(
        "Output volume {} x {} x {}, min={:.4} max={:.4} mean={:.4}",
        nx3, ny3, nz3, dmin, dmax, dmean
    );
}
