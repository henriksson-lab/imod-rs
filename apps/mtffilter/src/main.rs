use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Apply MTF (modulation transfer function) correction or low-pass filtering
/// to images using Fourier filtering.
#[derive(Parser)]
#[command(name = "mtffilter", about = "Fourier-space frequency filtering")]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Low-pass filter cutoff (fraction of Nyquist, 0-1). 0 = no filter.
    #[arg(short = 'l', long, default_value_t = 0.0)]
    lowpass: f32,

    /// High-pass filter cutoff (fraction of Nyquist, 0-1). 0 = no filter.
    #[arg(short = 'H', long, default_value_t = 0.0)]
    highpass: f32,

    /// Gaussian falloff sigma (in Fourier pixels)
    #[arg(short = 's', long, default_value_t = 0.05)]
    sigma: f32,
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

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.add_label(&format!(
        "mtffilter: lowpass={:.3} highpass={:.3} sigma={:.3}",
        args.lowpass, args.highpass, args.sigma
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    let mut planner = FftPlanner::<f32>::new();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let filtered = filter_2d(&data, nx, ny, &mut planner, args.lowpass, args.highpass, args.sigma);

        let (smin, smax, smean) = min_max_mean(&filtered);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&filtered).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
    eprintln!("mtffilter: filtered {} sections", nz);
}

fn filter_2d(
    data: &[f32],
    nx: usize,
    ny: usize,
    planner: &mut FftPlanner<f32>,
    lowpass: f32,
    highpass: f32,
    sigma: f32,
) -> Vec<f32> {
    let fft_x = planner.plan_fft_forward(nx);
    let ifft_x = planner.plan_fft_inverse(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let ifft_y = planner.plan_fft_inverse(ny);

    // Forward FFT: rows then columns
    let mut freq = vec![Complex::new(0.0f32, 0.0); nx * ny];
    for j in 0..ny {
        let mut row: Vec<Complex<f32>> = data[j * nx..(j + 1) * nx]
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();
        fft_x.process(&mut row);
        freq[j * nx..(j + 1) * nx].copy_from_slice(&row);
    }
    let mut col = vec![Complex::new(0.0f32, 0.0); ny];
    for i in 0..nx {
        for j in 0..ny { col[j] = freq[j * nx + i]; }
        fft_y.process(&mut col);
        for j in 0..ny { freq[j * nx + i] = col[j]; }
    }

    // Apply filter in frequency space
    let nyq_x = nx as f32 / 2.0;
    let nyq_y = ny as f32 / 2.0;
    let sigma_sq = if sigma > 0.0 { 2.0 * sigma * sigma } else { 1e-10 };

    for j in 0..ny {
        let fy = if j <= ny / 2 { j as f32 } else { (ny - j) as f32 };
        let fy_norm = fy / nyq_y;

        for i in 0..nx {
            let fx = if i <= nx / 2 { i as f32 } else { (nx - i) as f32 };
            let fx_norm = fx / nyq_x;
            let r = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();

            let mut filter = 1.0f32;

            if lowpass > 0.0 {
                if r > lowpass {
                    let d = r - lowpass;
                    filter *= (-d * d / sigma_sq).exp();
                }
            }

            if highpass > 0.0 {
                if r < highpass {
                    let d = highpass - r;
                    filter *= 1.0 - (-d * d / sigma_sq).exp();
                }
            }

            freq[j * nx + i] *= filter;
        }
    }

    // Inverse FFT: columns then rows
    for i in 0..nx {
        for j in 0..ny { col[j] = freq[j * nx + i]; }
        ifft_y.process(&mut col);
        for j in 0..ny { freq[j * nx + i] = col[j]; }
    }

    let scale = 1.0 / (nx * ny) as f32;
    let mut output = vec![0.0f32; nx * ny];
    for j in 0..ny {
        let mut row: Vec<Complex<f32>> = freq[j * nx..(j + 1) * nx].to_vec();
        ifft_x.process(&mut row);
        for i in 0..nx {
            output[j * nx + i] = row[i].re * scale;
        }
    }

    output
}
