use clap::Parser;
use imod_core::MrcMode;
use imod_fft::{cross_correlate_2d, fft_r2c_2d, fft_c2r_2d};
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{write_xf_file, LinearTransform};
use rustfft::num_complex::Complex;

/// Align movie frames by cross-correlation and produce a summed image.
///
/// Reads a stack of movie frames, aligns each frame to a running reference
/// using FFT cross-correlation, applies the shifts, and sums the aligned frames.
/// Supports gain reference correction, dose weighting, and output of aligned stacks.
#[derive(Parser)]
#[command(name = "alignframes", about = "Align and sum movie frames")]
struct Args {
    /// Input movie stack (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output aligned sum (MRC)
    #[arg(short = 'o', long)]
    output: String,

    /// Output transform file with per-frame shifts (.xf)
    #[arg(short = 'x', long)]
    xform_output: Option<String>,

    /// Number of alignment iterations
    #[arg(short = 'n', long, default_value_t = 3)]
    iterations: usize,

    /// Truncate frames: first frame to use (0-based)
    #[arg(long, default_value_t = 0)]
    first: usize,

    /// Last frame to use (0-based, default: last)
    #[arg(long)]
    last: Option<usize>,

    /// Binning for alignment (full-res sum is still produced)
    #[arg(short = 'b', long, default_value_t = 1)]
    bin: usize,

    /// Group N frames for better SNR during alignment
    #[arg(short = 'g', long, default_value_t = 1)]
    group: usize,

    /// Gain reference MRC file. Each frame is divided by this before alignment.
    #[arg(long)]
    gain: Option<String>,

    /// Dose per frame in electrons/A^2. Applies exposure-dependent frequency
    /// weighting: weight = exp(-dose * frequency^2 / 2).
    #[arg(long)]
    dose_per_frame: Option<f32>,

    /// Output the aligned (but not summed) frames as an MRC stack.
    #[arg(long)]
    output_aligned_stack: Option<String>,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let first = args.first;
    let last = args.last.unwrap_or(nz - 1).min(nz - 1);
    let n_frames = last - first + 1;

    eprintln!("alignframes: {} x {} x {} frames, using {}-{} ({} frames)",
        nx, ny, nz, first, last, n_frames);

    // --- Load gain reference ---
    let gain_ref: Option<Vec<f32>> = args.gain.as_ref().map(|gain_path| {
        eprintln!("alignframes: loading gain reference from {}", gain_path);
        let mut gain_reader = MrcReader::open(gain_path).unwrap_or_else(|e| {
            eprintln!("Error opening gain reference {}: {}", gain_path, e);
            std::process::exit(1);
        });
        let gh = gain_reader.header().clone();
        if gh.nx as usize != nx || gh.ny as usize != ny {
            eprintln!(
                "Error: gain reference size {}x{} does not match frame size {}x{}",
                gh.nx, gh.ny, nx, ny
            );
            std::process::exit(1);
        }
        gain_reader.read_slice_f32(0).unwrap()
    });

    // Read all frames
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    for z in first..=last {
        let mut frame = reader.read_slice_f32(z).unwrap();

        // Apply gain correction: divide each pixel by the gain reference
        if let Some(ref gain) = gain_ref {
            for i in 0..frame.len() {
                if gain[i] != 0.0 {
                    frame[i] /= gain[i];
                } else {
                    frame[i] = 0.0;
                }
            }
        }

        frames.push(frame);
    }

    if gain_ref.is_some() {
        eprintln!("alignframes: gain correction applied to {} frames", n_frames);
    }

    // FFT size (power of 2)
    let fft_nx = next_pow2(nx / args.bin);
    let fft_ny = next_pow2(ny / args.bin);

    // Bin frames for alignment if requested
    let align_frames: Vec<Vec<f32>> = if args.bin > 1 {
        frames.iter().map(|f| {
            let s = Slice::from_data(nx, ny, f.clone());
            imod_slice::bin(&s, args.bin).data
        }).collect()
    } else {
        frames.clone()
    };

    let align_nx = nx / args.bin;
    let align_ny = ny / args.bin;

    // Iterative alignment
    let mut shifts: Vec<(f32, f32)> = vec![(0.0, 0.0); n_frames];

    for iter in 0..args.iterations {
        // Build reference from current aligned sum
        let mut ref_data = vec![0.0f32; align_nx * align_ny];
        for (fi, frame) in align_frames.iter().enumerate() {
            let (dx, dy) = shifts[fi];
            let s = Slice::from_data(align_nx, align_ny, frame.clone());
            for y in 0..align_ny {
                for x in 0..align_nx {
                    let sx = x as f32 - dx / args.bin as f32;
                    let sy = y as f32 - dy / args.bin as f32;
                    ref_data[y * align_nx + x] += s.interpolate_bilinear(sx, sy, 0.0);
                }
            }
        }
        let inv_n = 1.0 / n_frames as f32;
        for v in &mut ref_data { *v *= inv_n; }

        // Pad reference
        let ref_padded = pad_image(&ref_data, align_nx, align_ny, fft_nx, fft_ny);

        // Align each frame to reference
        for (fi, frame) in align_frames.iter().enumerate() {
            let frame_padded = pad_image(frame, align_nx, align_ny, fft_nx, fft_ny);
            let cc = cross_correlate_2d(&ref_padded, &frame_padded, fft_nx, fft_ny);
            let (px, py) = find_peak(&cc, fft_nx, fft_ny);
            let dx = if px > fft_nx / 2 { px as f32 - fft_nx as f32 } else { px as f32 };
            let dy = if py > fft_ny / 2 { py as f32 - fft_ny as f32 } else { py as f32 };
            shifts[fi] = (dx * args.bin as f32, dy * args.bin as f32);
        }

        let max_shift = shifts.iter().map(|(dx, dy)| dx.abs().max(dy.abs())).fold(0.0f32, f32::max);
        eprintln!("  iter {}: max shift = {:.2} px", iter + 1, max_shift);
    }

    // --- Output aligned (but not summed) stack ---
    if let Some(ref aligned_path) = args.output_aligned_stack {
        eprintln!("alignframes: writing aligned stack to {}", aligned_path);
        let mut aligned_header = MrcHeader::new(nx as i32, ny as i32, n_frames as i32, MrcMode::Float);
        aligned_header.xlen = h.xlen;
        aligned_header.ylen = h.ylen;
        aligned_header.zlen = h.zlen;
        aligned_header.mx = nx as i32;
        aligned_header.my = ny as i32;
        aligned_header.mz = n_frames as i32;
        aligned_header.add_label("alignframes: aligned stack");

        let mut aligned_writer = MrcWriter::create(aligned_path, aligned_header).unwrap();
        let mut amin = f32::MAX;
        let mut amax = f32::MIN;
        let mut asum = 0.0f64;

        for (fi, frame) in frames.iter().enumerate() {
            let (dx, dy) = shifts[fi];
            let s = Slice::from_data(nx, ny, frame.clone());
            let mut aligned_frame = vec![0.0f32; nx * ny];
            for y in 0..ny {
                for x in 0..nx {
                    let sx = x as f32 - dx;
                    let sy = y as f32 - dy;
                    aligned_frame[y * nx + x] = s.interpolate_bilinear(sx, sy, 0.0);
                }
            }

            let (fmin, fmax, fmean) = min_max_mean(&aligned_frame);
            if fmin < amin { amin = fmin; }
            if fmax > amax { amax = fmax; }
            asum += fmean as f64 * (nx * ny) as f64;

            aligned_writer.write_slice_f32(&aligned_frame).unwrap();
        }

        let amean = (asum / (nx * ny * n_frames) as f64) as f32;
        aligned_writer.finish(amin, amax, amean).unwrap();
        eprintln!("alignframes: wrote {} aligned frames to {}", n_frames, aligned_path);
    }

    // --- Apply shifts at full resolution and sum, with optional dose weighting ---
    let sum = if let Some(dose_per_frame) = args.dose_per_frame {
        eprintln!(
            "alignframes: applying dose weighting ({} e/A^2 per frame)",
            dose_per_frame
        );
        dose_weighted_sum(&frames, &shifts, nx, ny, dose_per_frame)
    } else {
        // Simple unweighted sum
        let mut sum = vec![0.0f32; nx * ny];
        for (fi, frame) in frames.iter().enumerate() {
            let (dx, dy) = shifts[fi];
            let s = Slice::from_data(nx, ny, frame.clone());
            for y in 0..ny {
                for x in 0..nx {
                    let sx = x as f32 - dx;
                    let sy = y as f32 - dy;
                    sum[y * nx + x] += s.interpolate_bilinear(sx, sy, 0.0);
                }
            }
        }
        let inv_n = 1.0 / n_frames as f32;
        for v in &mut sum { *v *= inv_n; }
        sum
    };

    // Write output
    let (gmin, gmax, gmean) = min_max_mean(&sum);
    let mut out_header = MrcHeader::new(nx as i32, ny as i32, 1, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.pixel_size_z();
    out_header.mx = nx as i32;
    out_header.my = ny as i32;
    out_header.mz = 1;
    out_header.add_label(&format!("alignframes: {} frames aligned, {} iterations", n_frames, args.iterations));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    writer.write_slice_f32(&sum).unwrap();
    writer.finish(gmin, gmax, gmean).unwrap();

    // Write transforms if requested
    if let Some(ref xf_path) = args.xform_output {
        let xforms: Vec<LinearTransform> = shifts.iter()
            .map(|&(dx, dy)| LinearTransform::translation(dx, dy))
            .collect();
        write_xf_file(xf_path, &xforms).unwrap();
        eprintln!("alignframes: wrote {} transforms to {}", n_frames, xf_path);
    }

    eprintln!("alignframes: wrote aligned sum to {}", args.output);
}

/// Produce a dose-weighted sum of aligned frames.
///
/// Each frame is shifted, FFT'd, multiplied by an exposure-dependent weight,
/// then all are summed in Fourier space and transformed back.
/// Weight for frame k = exp(-cumulative_dose_k * frequency^2 / 2).
fn dose_weighted_sum(
    frames: &[Vec<f32>],
    shifts: &[(f32, f32)],
    nx: usize,
    ny: usize,
    dose_per_frame: f32,
) -> Vec<f32> {
    let fft_nx = next_pow2(nx);
    let fft_ny = next_pow2(ny);
    let nxc = fft_nx / 2 + 1;

    let mut sum_freq: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); nxc * fft_ny];
    let mut weight_sum: Vec<f32> = vec![0.0; nxc * fft_ny];

    for (fi, frame) in frames.iter().enumerate() {
        let (dx, dy) = shifts[fi];
        let cumulative_dose = dose_per_frame * (fi + 1) as f32;

        // Shift the frame
        let s = Slice::from_data(nx, ny, frame.clone());
        let mut shifted = vec![0.0f32; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let sx = x as f32 - dx;
                let sy = y as f32 - dy;
                shifted[y * nx + x] = s.interpolate_bilinear(sx, sy, 0.0);
            }
        }

        // Pad and FFT
        let padded = pad_image(&shifted, nx, ny, fft_nx, fft_ny);
        let freq = fft_r2c_2d(&padded, fft_nx, fft_ny);

        // Apply dose weighting and accumulate
        for fy in 0..fft_ny {
            let fy_norm = if fy <= fft_ny / 2 {
                fy as f32 / fft_ny as f32
            } else {
                (fft_ny - fy) as f32 / fft_ny as f32
            };
            for fx in 0..nxc {
                let fx_norm = fx as f32 / fft_nx as f32;
                let freq_sq = fx_norm * fx_norm + fy_norm * fy_norm;

                // Dose weight: exp(-cumulative_dose * freq^2 / 2)
                let w = (-cumulative_dose * freq_sq / 2.0).exp();

                let idx = fy * nxc + fx;
                sum_freq[idx] += freq[idx] * w;
                weight_sum[idx] += w;
            }
        }
    }

    // Normalize by total weight at each frequency
    for i in 0..sum_freq.len() {
        if weight_sum[i] > 0.0 {
            sum_freq[i] /= weight_sum[i];
        }
    }

    // Inverse FFT
    let result_padded = fft_c2r_2d(&sum_freq, fft_nx, fft_ny);

    // Extract original size
    let ox = (fft_nx - nx) / 2;
    let oy = (fft_ny - ny) / 2;
    let mut result = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            result[y * nx + x] = result_padded[(y + oy) * fft_nx + (x + ox)];
        }
    }

    result
}

fn pad_image(data: &[f32], nx: usize, ny: usize, fft_nx: usize, fft_ny: usize) -> Vec<f32> {
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = (sum / data.len() as f64) as f32;
    let mut padded = vec![mean; fft_nx * fft_ny];
    let ox = (fft_nx - nx) / 2;
    let oy = (fft_ny - ny) / 2;
    for y in 0..ny {
        for x in 0..nx {
            padded[(y + oy) * fft_nx + (x + ox)] = data[y * nx + x];
        }
    }
    padded
}

fn find_peak(cc: &[f32], nx: usize, ny: usize) -> (usize, usize) {
    let mut max_val = f32::NEG_INFINITY;
    let mut mx = 0;
    let mut my = 0;
    for y in 0..ny {
        for x in 0..nx {
            if cc[y * nx + x] > max_val {
                max_val = cc[y * nx + x];
                mx = x;
                my = y;
            }
        }
    }
    (mx, my)
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}
