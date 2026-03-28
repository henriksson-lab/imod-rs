use clap::Parser;
use imod_core::MrcMode;
use imod_fft::cross_correlate_2d;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{write_xf_file, LinearTransform};

/// Align movie frames by cross-correlation and produce a summed image.
///
/// Reads a stack of movie frames, aligns each frame to a running reference
/// using FFT cross-correlation, applies the shifts, and sums the aligned frames.
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

    // Read all frames
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    for z in first..=last {
        frames.push(reader.read_slice_f32(z).unwrap());
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

    // Apply shifts at full resolution and sum
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
