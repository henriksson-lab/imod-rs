use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::MrcMode;
use imod_mrc::{MrcReader, MrcWriter};

/// Prepare images for nonlinear anisotropic diffusion (pre-NAD).
///
/// Applies a 2D nonlinear anisotropic diffusion filter to each tilt projection
/// in a stack, parameterized by the tilt angle. Sigma and iteration counts
/// control the amount of filtering. A tilt-angle file provides per-slice angles.
#[derive(Parser)]
#[command(name = "prenad", version, about)]
struct Args {
    /// Input aligned tilt stack (MRC).
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output filtered stack (MRC).
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Tilt angles file (one angle per line in degrees).
    #[arg(short = 'a', long = "angles")]
    angles: Option<String>,

    /// Sigma for Gaussian smoothing of structure tensor.
    #[arg(short = 's', long = "sigma", default_value_t = 1.0)]
    sigma: f32,

    /// Minimum number of iterations per tilt.
    #[arg(long = "minite", default_value_t = 3)]
    min_iterations: usize,

    /// Maximum number of iterations per tilt.
    #[arg(long = "maxite", default_value_t = 6)]
    max_iterations: usize,

    /// Masked Variance Difference threshold (optional, auto-stop).
    #[arg(long = "mvd")]
    mvd: Option<f32>,
}

/// 2D Gaussian smoothing of a slice (separable, simple box approximation).
fn gaussian_smooth_2d(data: &[f32], nx: usize, ny: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.5 {
        return data.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f32> = (0..=radius)
        .map(|i| (-(i as f32 * i as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = kernel[0] + 2.0 * kernel[1..].iter().sum::<f32>();
    let kernel: Vec<f32> = kernel.iter().map(|v| v / sum).collect();

    // Smooth in X
    let mut tmp = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut val = kernel[0] * data[y * nx + x];
            for k in 1..=radius {
                let xl = if x >= k { x - k } else { k - x };
                let xr = if x + k < nx { x + k } else { 2 * nx - 2 - x - k };
                val += kernel[k] * (data[y * nx + xl] + data[y * nx + xr]);
            }
            tmp[y * nx + x] = val;
        }
    }

    // Smooth in Y
    let mut out = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut val = kernel[0] * tmp[y * nx + x];
            for k in 1..=radius {
                let yl = if y >= k { y - k } else { k - y };
                let yr = if y + k < ny { y + k } else { 2 * ny - 2 - y - k };
                val += kernel[k] * (tmp[yl * nx + x] + tmp[yr * nx + x]);
            }
            out[y * nx + x] = val;
        }
    }

    out
}

/// Compute gradient magnitude squared from smoothed image.
fn gradient_magnitude_sq(data: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut gms = vec![0.0f32; nx * ny];
    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let dx = (data[y * nx + x + 1] - data[y * nx + x - 1]) * 0.5;
            let dy = (data[(y + 1) * nx + x] - data[(y - 1) * nx + x]) * 0.5;
            gms[y * nx + x] = dx * dx + dy * dy;
        }
    }
    gms
}

/// One iteration of edge-enhancing 2D diffusion on a single slice.
fn diffuse_2d(input: &[f32], nx: usize, ny: usize, sigma: f32, lambda: f32) -> Vec<f32> {
    let smoothed = gaussian_smooth_2d(input, nx, ny, sigma);
    let gms = gradient_magnitude_sq(&smoothed, nx, ny);

    // Diffusivity: g(s) = 1 - exp(-3.315 / (s/lambda^2)^4)  (Weickert EED)
    let lambda2 = lambda * lambda;
    let diffusivity: Vec<f32> = gms
        .iter()
        .map(|&s| {
            if s < 1e-10 {
                1.0
            } else {
                let ratio = s / lambda2;
                1.0 - (-3.315 / (ratio * ratio * ratio * ratio)).exp()
            }
        })
        .collect();

    let dt = 0.20; // stable time step for 2D
    let mut out = input.to_vec();

    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let idx = y * nx + x;
            let g = diffusivity[idx];
            let laplacian = input[idx - 1] + input[idx + 1]
                + input[idx - nx] + input[idx + nx]
                - 4.0 * input[idx];
            out[idx] = input[idx] + dt * g * laplacian;
        }
    }

    out
}

/// Compute masked variance difference between two slices.
fn masked_variance_diff(original: &[f32], filtered: &[f32]) -> f64 {
    let n = original.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        let diff = (filtered[i] - original[i]).abs() as f64;
        sum += diff;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: prenad - opening input: {}", e);
        process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    // Read tilt angles
    let angles: Vec<f64> = if let Some(ref angle_file) = args.angles {
        let f = File::open(angle_file).unwrap_or_else(|e| {
            eprintln!("ERROR: prenad - opening angles file: {}", e);
            process::exit(1);
        });
        BufReader::new(f)
            .lines()
            .filter_map(|l| l.ok())
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect()
    } else {
        vec![0.0; nz]
    };

    if angles.len() < nz {
        eprintln!(
            "WARNING: prenad - only {} angles for {} slices, padding with 0",
            angles.len(),
            nz
        );
    }

    let mut out_header = h.clone();
    out_header.mode = MrcMode::Float as i32;
    out_header.add_label("prenad: nonlinear anisotropic diffusion filtering");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: prenad - creating output: {}", e);
        process::exit(1);
    });

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for iz in 0..nz {
        let angle_deg = if iz < angles.len() { angles[iz] } else { 0.0 };
        let angle_rad = angle_deg.abs() * std::f64::consts::PI / 180.0;

        // Scale sigma by cos(angle) -- higher angles need less smoothing
        let cos_a = angle_rad.cos().max(0.1) as f32;
        let effective_sigma = args.sigma / cos_a;

        // Determine iteration count: use angle to bias between min and max
        let angle_frac = (angle_rad.abs() / (std::f64::consts::PI / 2.0)).min(1.0);
        let n_iter = args.min_iterations
            + ((args.max_iterations - args.min_iterations) as f64 * angle_frac) as usize;

        let original = reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: prenad - reading slice {}: {}", iz, e);
            process::exit(1);
        });

        let mut current = original.clone();
        let lambda = 30.0_f32; // contrast parameter

        for it in 0..n_iter {
            current = diffuse_2d(&current, nx, ny, effective_sigma, lambda);

            if let Some(mvd_thresh) = args.mvd {
                let mvd = masked_variance_diff(&original, &current);
                if it >= args.min_iterations && mvd < mvd_thresh as f64 {
                    break;
                }
            }
        }

        // Statistics
        for &v in &current {
            gmin = gmin.min(v);
            gmax = gmax.max(v);
            gsum += v as f64;
        }

        writer.write_slice_f32(&current).unwrap_or_else(|e| {
            eprintln!("ERROR: prenad - writing slice {}: {}", iz, e);
            process::exit(1);
        });

        println!(
            "Slice {:4} angle={:7.2} iters={} sigma_eff={:.2}",
            iz, angle_deg, n_iter, effective_sigma
        );
    }

    let gmean = gsum / (nx * ny * nz) as f64;
    writer.finish(gmin, gmax, gmean as f32).unwrap();

    println!(
        "Done. Min={:.4} Max={:.4} Mean={:.4}",
        gmin, gmax, gmean
    );
}
