use std::process;

use clap::Parser;
use imod_mrc::{MrcReader, MrcWriter};

/// 3D nonlinear anisotropic diffusion (edge-enhancing) filter.
///
/// Applies 3D edge-enhancing diffusion (EED) to a volume. The filter
/// preserves edges while smoothing homogeneous regions, using a diffusion
/// tensor derived from the structure tensor of the image.
#[derive(Parser)]
#[command(name = "nad_eed_3d", version, about)]
struct Args {
    /// Input MRC file.
    input: String,

    /// Output MRC file.
    output: String,

    /// K (lambda) value, threshold for gradients.
    #[arg(short = 'k', long = "lambda", default_value_t = 1.0)]
    lambda: f32,

    /// Number of iterations.
    #[arg(short = 'n', long = "iterations", default_value_t = 20)]
    iterations: usize,

    /// Sigma for smoothing of structure tensor.
    #[arg(short = 's', long = "sigma", default_value_t = 0.0)]
    sigma: f32,

    /// Time step.
    #[arg(short = 't', long = "timestep", default_value_t = 0.1)]
    timestep: f32,

    /// Output only this Z slice (1-based).
    #[arg(short = 'o', long = "oneslice")]
    one_slice: Option<usize>,

    /// Mode for output file (0=byte, 1=short, 2=float).
    #[arg(short = 'm', long = "mode")]
    out_mode: Option<i32>,
}

/// 3D Gaussian convolution (separable) with periodic boundary conditions.
fn gauss_conv_3d(vol: &mut Vec<f32>, nx: usize, ny: usize, nz: usize, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }

    let precision = (3.0 * sigma).ceil() as usize;
    let length = precision + 1;

    // Build kernel
    let mut kernel = vec![0.0f32; length + 1];
    for i in 0..=length {
        kernel[i] = (1.0 / (sigma * (2.0 * std::f32::consts::PI).sqrt()))
            * (-(i as f32 * i as f32) / (2.0 * sigma * sigma)).exp();
    }
    let sum: f32 = kernel[0] + 2.0 * kernel[1..=length].iter().sum::<f32>();
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Convolve along X
    let mut help = vec![0.0f32; nx + 2 * length];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                help[i + length] = vol[k * ny * nx + j * nx + i];
            }
            // Periodic boundary
            for p in 0..length {
                help[length - 1 - p] = help[nx + length - 1 - p];
                help[nx + length + p] = help[length + p];
            }
            for i in 0..nx {
                let mut s = kernel[0] * help[i + length];
                for p in 1..=length.min(nx) {
                    s += kernel[p] * (help[i + length + p] + help[i + length - p]);
                }
                vol[k * ny * nx + j * nx + i] = s;
            }
        }
    }

    // Convolve along Y
    let mut help = vec![0.0f32; ny + 2 * length];
    for k in 0..nz {
        for i in 0..nx {
            for j in 0..ny {
                help[j + length] = vol[k * ny * nx + j * nx + i];
            }
            for p in 0..length {
                help[length - 1 - p] = help[ny + length - 1 - p];
                help[ny + length + p] = help[length + p];
            }
            for j in 0..ny {
                let mut s = kernel[0] * help[j + length];
                for p in 1..=length.min(ny) {
                    s += kernel[p] * (help[j + length + p] + help[j + length - p]);
                }
                vol[k * ny * nx + j * nx + i] = s;
            }
        }
    }

    // Convolve along Z
    let mut help = vec![0.0f32; nz + 2 * length];
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                help[k + length] = vol[k * ny * nx + j * nx + i];
            }
            for p in 0..length {
                help[length - 1 - p] = help[nz + length - 1 - p];
                help[nz + length + p] = help[length + p];
            }
            for k in 0..nz {
                let mut s = kernel[0] * help[k + length];
                for p in 1..=length.min(nz) {
                    s += kernel[p] * (help[k + length + p] + help[k + length - p]);
                }
                vol[k * ny * nx + j * nx + i] = s;
            }
        }
    }
}

/// Inline index helper for 3D volume with 1-based boundary padding.
#[inline]
fn idx(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    k * ny * nx + j * nx + i
}

/// One EED diffusion step.
fn eed_step(
    u: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    sigma: f32,
    lambda: f32,
    ht: f32,
) {
    let f = u.to_vec();

    // Smooth a copy to compute structure tensor
    let mut smoothed = f.clone();
    gauss_conv_3d(&mut smoothed, nx, ny, nz, sigma);

    // Compute gradient magnitudes and EED diffusivity
    let lambda4 = lambda * lambda * lambda * lambda;
    let rxx = ht / (2.0);
    let ryy = ht / (2.0);
    let rzz = ht / (2.0);

    for k in 1..nz.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                let id = idx(i, j, k, nx, ny);

                // Central differences for gradient
                let dx = (smoothed[idx(i + 1, j, k, nx, ny)] - smoothed[idx(i.wrapping_sub(1).max(0), j, k, nx, ny)]) * 0.5;
                let dy = (smoothed[idx(i, j + 1, k, nx, ny)] - smoothed[idx(i, j.wrapping_sub(1).max(0), k, nx, ny)]) * 0.5;
                let dz = (smoothed[idx(i, j, k + 1, nx, ny)] - smoothed[idx(i, j, k.wrapping_sub(1).max(0), nx, ny)]) * 0.5;
                let grad_sq = dx * dx + dy * dy + dz * dz;

                // EED diffusivity (Weickert)
                let g = if grad_sq < 1e-10 {
                    1.0
                } else {
                    let ratio = grad_sq / lambda4;
                    1.0 - (-3.315 / ratio).exp()
                };

                // 3D isotropic diffusion with EED diffusivity
                let lap = rxx * g * (f[idx(i + 1, j, k, nx, ny)] + f[idx(i - 1, j, k, nx, ny)] - 2.0 * f[id])
                    + ryy * g * (f[idx(i, j + 1, k, nx, ny)] + f[idx(i, j - 1, k, nx, ny)] - 2.0 * f[id])
                    + rzz * g * (f[idx(i, j, k + 1, nx, ny)] + f[idx(i, j, k - 1, nx, ny)] - 2.0 * f[id]);

                u[id] = f[id] + lap;
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    println!("Program nad_eed_3d");

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: nad_eed_3d - opening input: {}", e);
        process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    println!("dimensions:          {} x {} x {}", nx, ny, nz);
    println!("K (lambda):          {}", args.lambda);
    println!("sigma:               {}", args.sigma);
    println!("time step:           {}", args.timestep);
    println!("iterations:          {}", args.iterations);

    // Read entire volume
    let mut volume = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        let slice = reader.read_slice_f32(z).unwrap_or_else(|e| {
            eprintln!("ERROR: nad_eed_3d - reading slice {}: {}", z, e);
            process::exit(1);
        });
        volume.extend_from_slice(&slice);
    }

    // Analyze initial statistics
    let (mut vmin, mut vmax, mut vmean) = stats(&volume);
    println!(
        "initial: min={:.6} max={:.6} mean={:.6}",
        vmin, vmax, vmean
    );

    // Run diffusion iterations
    for p in 1..=args.iterations {
        eed_step(
            &mut volume,
            nx,
            ny,
            nz,
            args.sigma,
            args.lambda,
            args.timestep,
        );

        let (mn, mx, me) = stats(&volume);
        vmin = mn;
        vmax = mx;
        vmean = me;
        println!(
            "iteration {:4}: min={:.6} max={:.6} mean={:.6}",
            p, vmin, vmax, vmean
        );
    }

    // Write output
    let (kst, knd) = if let Some(s) = args.one_slice {
        let s = s.clamp(1, nz);
        (s - 1, s)
    } else {
        (0, nz)
    };
    let nzout = knd - kst;

    let out_mode = args.out_mode.unwrap_or(h.mode);
    let mut out_header = h.clone();
    out_header.nx = nx as i32;
    out_header.ny = ny as i32;
    out_header.nz = nzout as i32;
    out_header.mode = out_mode;
    out_header.mx = nx as i32;
    out_header.my = ny as i32;
    out_header.mz = nzout as i32;
    out_header.add_label(&format!(
        "nad_eed_3d: K={} n={} s={} t={}",
        args.lambda, args.iterations, args.sigma, args.timestep
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: nad_eed_3d - creating output: {}", e);
        process::exit(1);
    });

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;
    for k in kst..knd {
        let slice = &volume[k * ny * nx..(k + 1) * ny * nx];
        for &v in slice {
            gmin = gmin.min(v);
            gmax = gmax.max(v);
            gsum += v as f64;
        }
        writer.write_slice_f32(slice).unwrap_or_else(|e| {
            eprintln!("ERROR: nad_eed_3d - writing slice: {}", e);
            process::exit(1);
        });
    }

    let gmean = gsum / (nx * ny * nzout) as f64;
    writer.finish(gmin, gmax, gmean as f32).unwrap();

    println!(
        "Output: min={:.6} max={:.6} mean={:.6}",
        gmin, gmax, gmean
    );
}

fn stats(data: &[f32]) -> (f32, f32, f32) {
    let mut mn = f32::MAX;
    let mut mx = f32::MIN;
    let mut sum = 0.0_f64;
    for &v in data {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    (mn, mx, (sum / data.len() as f64) as f32)
}
