use clap::Parser;
use imod_fft::cross_correlate_2d;
use imod_mrc::MrcReader;
use imod_transforms::{write_xf_file, LinearTransform};

/// Find translational alignment between adjacent sections in a tilt series
/// using cross-correlation.
#[derive(Parser)]
#[command(name = "tiltxcorr", about = "Cross-correlation alignment of tilt series")]
struct Args {
    /// Input tilt series MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output transform file (.xf)
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt or .rawtlt)
    #[arg(short = 't', long)]
    tilt_file: Option<String>,

    /// Reference section (0-based, default: middle)
    #[arg(short = 'r', long)]
    reference: Option<usize>,

    /// Exclude views from alignment (comma-separated 0-based indices)
    #[arg(short = 'e', long)]
    exclude: Option<String>,

    /// Filter radius 1 (low-freq cutoff, fraction of Nyquist, default 0.0)
    #[arg(long, default_value_t = 0.0)]
    filter_radius1: f32,

    /// Filter radius 2 (high-freq cutoff, fraction of Nyquist, default 0.25)
    #[arg(long, default_value_t = 0.25)]
    filter_radius2: f32,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", args.input, e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    // Ensure nx/ny are suitable for FFT (should be even)
    // For now we pad to nearest power of 2 if needed
    let fft_nx = next_power_of_2(nx);
    let fft_ny = next_power_of_2(ny);

    let ref_section = args.reference.unwrap_or(nz / 2);

    let excluded: Vec<usize> = args
        .exclude
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default();

    eprintln!("tiltxcorr: {} x {} x {}, reference section {}", nx, ny, nz, ref_section);

    // Read all sections
    let mut sections: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        sections.push(reader.read_slice_f32(z).unwrap());
    }

    // Compute transforms by correlating each section with its neighbor
    let mut transforms = vec![LinearTransform::identity(); nz];

    // Work outward from reference in both directions
    // Forward: ref+1, ref+2, ...
    for z in (ref_section + 1)..nz {
        if excluded.contains(&z) || excluded.contains(&(z - 1)) {
            transforms[z] = transforms[z - 1];
            continue;
        }
        let (dx, dy) = find_shift(&sections[z - 1], &sections[z], nx, ny, fft_nx, fft_ny);
        transforms[z] = LinearTransform::translation(
            transforms[z - 1].dx + dx,
            transforms[z - 1].dy + dy,
        );
    }

    // Backward: ref-1, ref-2, ...
    for z in (0..ref_section).rev() {
        if excluded.contains(&z) || excluded.contains(&(z + 1)) {
            transforms[z] = transforms[z + 1];
            continue;
        }
        let (dx, dy) = find_shift(&sections[z + 1], &sections[z], nx, ny, fft_nx, fft_ny);
        transforms[z] = LinearTransform::translation(
            transforms[z + 1].dx + dx,
            transforms[z + 1].dy + dy,
        );
    }

    // Write output
    write_xf_file(&args.output, &transforms).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {}", args.output, e);
        std::process::exit(1);
    });

    eprintln!("tiltxcorr: wrote {} transforms to {}", nz, args.output);
    for (z, xf) in transforms.iter().enumerate() {
        eprintln!("  section {:>3}: dx={:>8.2}, dy={:>8.2}", z, xf.dx, xf.dy);
    }
}

/// Find the translational shift between two images using cross-correlation.
/// Returns (dx, dy) that should be applied to `target` to align it to `reference`.
fn find_shift(
    reference: &[f32],
    target: &[f32],
    nx: usize,
    ny: usize,
    fft_nx: usize,
    fft_ny: usize,
) -> (f32, f32) {
    // Pad images to FFT size
    let ref_padded = pad_image(reference, nx, ny, fft_nx, fft_ny);
    let tgt_padded = pad_image(target, nx, ny, fft_nx, fft_ny);

    // Cross-correlate
    let cc = cross_correlate_2d(&ref_padded, &tgt_padded, fft_nx, fft_ny);

    // Find peak
    let (px, py) = find_peak(&cc, fft_nx, fft_ny);

    // Convert to shift (handle wrap-around)
    let dx = if px > fft_nx / 2 { px as f32 - fft_nx as f32 } else { px as f32 };
    let dy = if py > fft_ny / 2 { py as f32 - fft_ny as f32 } else { py as f32 };

    (dx, dy)
}

fn pad_image(data: &[f32], nx: usize, ny: usize, fft_nx: usize, fft_ny: usize) -> Vec<f32> {
    // Compute mean for padding
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
    let mut max_x = 0;
    let mut max_y = 0;
    for y in 0..ny {
        for x in 0..nx {
            let v = cc[y * nx + x];
            if v > max_val {
                max_val = v;
                max_x = x;
                max_y = y;
            }
        }
    }
    (max_x, max_y)
}

fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}
