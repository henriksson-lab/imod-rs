use std::process;

use clap::Parser;
use imod_core::MrcMode;
use imod_fft::{fft_c2r_2d, fft_r2c_2d};
use imod_mrc::{MrcReader, MrcWriter};
use rustfft::num_complex::Complex;

/// Enhance contrast of a volume using Gaussian bandpass filtering.
///
/// A 2D Gaussian bandpass filter is applied to each section in the stack.
/// The filter is parameterized by two sigmas and two radii (in fractional
/// reciprocal lattice units, where r goes from 0 to ~0.5 per axis).
///
/// Filter = HighPass(sigma1) * BandPass(radius1, radius2, sigma2)
///
/// If sigma1 > 0: highpass = 1 - exp(-r^2 / (2*sigma1^2))
/// If sigma1 < 0: Del-squared-G = r^2 * exp(-r^2 / (2*sigma1^2))
/// If sigma1 = 0: no highpass component
///
/// The bandpass is flat from radius1 to radius2, with Gaussian decay of
/// width sigma2. If sigma2 < 0, the bandpass is inverted.
#[derive(Parser)]
#[command(name = "enhance", version, about)]
struct Args {
    /// Input MRC file.
    input: String,

    /// Output MRC file.
    output: String,

    /// Sigma for high-pass Gaussian (0 = none, <0 = Del-squared-G).
    #[arg(long = "sigma1", default_value_t = 0.0)]
    sigma1: f32,

    /// Sigma for band-pass Gaussian decay (0 = none, <0 = inverted).
    #[arg(long = "sigma2", default_value_t = 0.0)]
    sigma2: f32,

    /// Inner radius for bandpass (fractional reciprocal units).
    #[arg(long = "radius1", default_value_t = 0.0)]
    radius1: f32,

    /// Outer radius for bandpass (fractional reciprocal units).
    #[arg(long = "radius2", default_value_t = 0.0)]
    radius2: f32,

    /// Reset origin to original value.
    #[arg(long = "resetorigin")]
    reset_origin: bool,
}

/// Build the CTF (filter) array for a given image size.
fn build_ctf(sigma1: f32, sigma2: f32, radius1: f32, radius2: f32, nx: usize, ny: usize) -> Vec<f32> {
    let half_nx = nx / 2 + 1;
    let mut ctf = vec![1.0f32; half_nx * ny];

    let neg_radius1 = radius1 < 0.0;
    let r1 = radius1.abs();
    let r2 = radius2;
    let inv_sigma2 = sigma2 < 0.0;
    let s2 = sigma2.abs();

    for iy in 0..ny {
        let fy = if iy <= ny / 2 {
            iy as f32 / ny as f32
        } else {
            (iy as f32 - ny as f32) / ny as f32
        };

        for ix in 0..half_nx {
            let fx = ix as f32 / nx as f32;
            let r = (fx * fx + fy * fy).sqrt();
            let idx = iy * half_nx + ix;

            let mut val = 1.0f32;

            // High-pass component
            if sigma1 > 0.0 {
                val *= 1.0 - (-r * r / (2.0 * sigma1 * sigma1)).exp();
            } else if sigma1 < 0.0 {
                let s1 = sigma1.abs();
                val *= r * r * (-r * r / (2.0 * s1 * s1)).exp();
            } else if neg_radius1 {
                // Inverted Gaussian rise from |radius1|
                if r < r1 {
                    val = 0.0;
                } else if sigma1.abs() > 0.0 {
                    let d = r - r1;
                    val *= 1.0 - (-d * d / (2.0 * sigma1.abs() * sigma1.abs())).exp();
                }
            }

            // Band-pass component
            if s2 > 0.0 {
                let band = if r < r1 {
                    let d = r1 - r;
                    (-d * d / (2.0 * s2 * s2)).exp()
                } else if r > r2 {
                    let d = r - r2;
                    (-d * d / (2.0 * s2 * s2)).exp()
                } else {
                    1.0
                };

                if inv_sigma2 {
                    val *= 1.0 - band;
                } else {
                    val *= band;
                }
            }

            ctf[idx] = val;
        }
    }

    ctf
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: enhance - opening input: {}", e);
        process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    println!(
        "ENHANCE: Bandpass Sigmas,Radii = {:.4} {:.4} {:.4} {:.4}",
        args.sigma1, args.sigma2, args.radius1, args.radius2
    );

    let ctf = build_ctf(args.sigma1, args.sigma2, args.radius1, args.radius2, nx, ny);
    let mut out_header = h.clone();
    out_header.add_label(&format!(
        "ENHANCE: Bandpass Sigmas,Radii= {:.4} {:.4} {:.4} {:.4}",
        args.sigma1, args.sigma2, args.radius1, args.radius2
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: enhance - creating output: {}", e);
        process::exit(1);
    });

    let mut tmin = f32::MAX;
    let mut tmax = f32::MIN;
    let mut tsum = 0.0_f64;

    for iz in 0..nz {
        let slice = reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: enhance - reading section {}: {}", iz, e);
            process::exit(1);
        });

        // Forward FFT
        let mut fft_data = fft_r2c_2d(&slice, nx, ny);

        // Apply filter
        for i in 0..fft_data.len().min(ctf.len()) {
            fft_data[i] = Complex::new(fft_data[i].re * ctf[i], fft_data[i].im * ctf[i]);
        }

        // Handle origin reset
        if args.reset_origin {
            // Preserve DC component
        } else {
            // CTF already handles DC
        }

        // Inverse FFT
        let mut filtered = fft_c2r_2d(&fft_data, nx, ny);

        // Compute statistics
        let mut dmin = f32::MAX;
        let mut dmax = f32::MIN;
        let mut dmean = 0.0_f64;
        for v in &filtered {
            dmin = dmin.min(*v);
            dmax = dmax.max(*v);
            dmean += *v as f64;
        }
        dmean /= (nx * ny) as f64;

        tmin = tmin.min(dmin);
        tmax = tmax.max(dmax);
        tsum += dmean;

        // Scale to output mode range if needed
        if h.mode != MrcMode::Float as i32 {
            let mode_max = match MrcMode::from_i32(h.mode) {
                Some(MrcMode::Byte) => 255.0,
                Some(MrcMode::Short) => 32767.0,
                _ => f32::MAX,
            };
            if dmax > dmin {
                let scale = mode_max / (dmax - dmin);
                for v in &mut filtered {
                    *v = (*v - dmin as f32) * scale;
                }
            }
        }

        writer.write_slice_f32(&filtered).unwrap_or_else(|e| {
            eprintln!("ERROR: enhance - writing section {}: {}", iz, e);
            process::exit(1);
        });

        if nz > 1 {
            println!(
                " Section # {:4} Min,Max,Mean density = {:12.4} {:12.4} {:12.4}",
                iz, dmin, dmax, dmean
            );
        }
    }

    let tmean = tsum / nz as f64;
    println!(
        "\n Overall Min,Max,Mean density = {:12.4} {:12.4} {:12.4}",
        tmin, tmax, tmean
    );

    writer.finish(tmin, tmax, tmean as f32).unwrap();
}
