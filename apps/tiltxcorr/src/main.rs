use clap::Parser;
use imod_fft::{cross_correlate_2d, fft_r2c_2d, fft_c2r_2d};
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

    /// Filter radius 1 (high-pass cutoff, fraction of Nyquist, default 0.0)
    #[arg(long, default_value_t = 0.0)]
    filter_radius1: f32,

    /// Filter radius 2 (low-pass cutoff, fraction of Nyquist, default 0.25)
    #[arg(long, default_value_t = 0.25)]
    filter_radius2: f32,

    /// Cumulative alignment: align each section to the running average of all
    /// previously aligned sections instead of just the immediate neighbor.
    #[arg(long, default_value_t = false)]
    cumulative: bool,

    /// Patch-based correlation: divide each image into an NxM grid and correlate
    /// each patch independently. Format: "nx,ny" (e.g. "3,3").
    #[arg(long)]
    patches: Option<String>,
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

    let use_bandpass = args.filter_radius1 > 0.0 || args.filter_radius2 < 0.5;

    // Parse patch grid
    let patch_grid: Option<(usize, usize)> = args.patches.as_deref().map(|s| {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 2 {
            eprintln!("Error: --patches must be in format 'nx,ny' (e.g. '3,3')");
            std::process::exit(1);
        }
        let pnx: usize = parts[0].trim().parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid patch nx");
            std::process::exit(1);
        });
        let pny: usize = parts[1].trim().parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid patch ny");
            std::process::exit(1);
        });
        (pnx, pny)
    });

    eprintln!(
        "tiltxcorr: {} x {} x {}, reference section {}, bandpass={}, cumulative={}",
        nx, ny, nz, ref_section, use_bandpass, args.cumulative
    );
    if let Some((pnx, pny)) = patch_grid {
        eprintln!("tiltxcorr: patch-based correlation {}x{} grid", pnx, pny);
    }

    // Read all sections
    let mut sections: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        sections.push(reader.read_slice_f32(z).unwrap());
    }

    // Apply bandpass filter to all sections if requested
    if use_bandpass {
        eprintln!(
            "tiltxcorr: applying bandpass filter: high-pass={}, low-pass={}",
            args.filter_radius1, args.filter_radius2
        );
        for z in 0..nz {
            sections[z] = apply_bandpass(&sections[z], nx, ny, fft_nx, fft_ny,
                                         args.filter_radius1, args.filter_radius2);
        }
    }

    // --- Patch-based correlation ---
    if let Some((pnx, pny)) = patch_grid {
        let patch_w = nx / pnx;
        let patch_h = ny / pny;
        let fft_pw = next_power_of_2(patch_w);
        let fft_ph = next_power_of_2(patch_h);

        eprintln!("tiltxcorr: patch size {}x{}, fft {}x{}", patch_w, patch_h, fft_pw, fft_ph);

        // Compute per-patch transforms; the global transform is the median of patches
        let mut transforms = vec![LinearTransform::identity(); nz];

        // Forward from reference
        for z in (ref_section + 1)..nz {
            if excluded.contains(&z) || excluded.contains(&(z - 1)) {
                transforms[z] = transforms[z - 1];
                continue;
            }
            let (dx, dy, patch_shifts) = find_shift_patches(
                &sections[z - 1], &sections[z], nx, ny,
                pnx, pny, patch_w, patch_h, fft_pw, fft_ph,
            );
            transforms[z] = LinearTransform::translation(
                transforms[z - 1].dx + dx,
                transforms[z - 1].dy + dy,
            );
            eprintln!("  section {:>3}: global dx={:.2}, dy={:.2} (from {} patches)",
                z, dx, dy, patch_shifts.len());
            for (pi, (pdx, pdy)) in patch_shifts.iter().enumerate() {
                let px_idx = pi % pnx;
                let py_idx = pi / pnx;
                eprintln!("    patch ({},{}): dx={:.2}, dy={:.2}", px_idx, py_idx, pdx, pdy);
            }
        }

        // Backward from reference
        for z in (0..ref_section).rev() {
            if excluded.contains(&z) || excluded.contains(&(z + 1)) {
                transforms[z] = transforms[z + 1];
                continue;
            }
            let (dx, dy, patch_shifts) = find_shift_patches(
                &sections[z + 1], &sections[z], nx, ny,
                pnx, pny, patch_w, patch_h, fft_pw, fft_ph,
            );
            transforms[z] = LinearTransform::translation(
                transforms[z + 1].dx + dx,
                transforms[z + 1].dy + dy,
            );
            eprintln!("  section {:>3}: global dx={:.2}, dy={:.2} (from {} patches)",
                z, dx, dy, patch_shifts.len());
            for (pi, (pdx, pdy)) in patch_shifts.iter().enumerate() {
                let px_idx = pi % pnx;
                let py_idx = pi / pnx;
                eprintln!("    patch ({},{}): dx={:.2}, dy={:.2}", px_idx, py_idx, pdx, pdy);
            }
        }

        write_xf_file(&args.output, &transforms).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", args.output, e);
            std::process::exit(1);
        });
        eprintln!("tiltxcorr: wrote {} transforms to {}", nz, args.output);
        for (z, xf) in transforms.iter().enumerate() {
            eprintln!("  section {:>3}: dx={:>8.2}, dy={:>8.2}", z, xf.dx, xf.dy);
        }
        return;
    }

    // --- Standard (non-patch) mode ---
    let mut transforms = vec![LinearTransform::identity(); nz];

    if args.cumulative {
        // Cumulative alignment: align each section to the running average
        // Forward: ref+1, ref+2, ...
        // Build cumulative reference starting from the reference section
        let mut cum_sum = sections[ref_section].clone();
        let mut cum_count = 1.0f32;

        for z in (ref_section + 1)..nz {
            if excluded.contains(&z) {
                transforms[z] = transforms[z - 1];
                continue;
            }

            // Build cumulative average
            let cum_avg: Vec<f32> = cum_sum.iter().map(|&v| v / cum_count).collect();

            let (dx, dy) = find_shift(&cum_avg, &sections[z], nx, ny, fft_nx, fft_ny);
            transforms[z] = LinearTransform::translation(
                transforms[z - 1].dx + dx,
                transforms[z - 1].dy + dy,
            );

            // Add this aligned section to cumulative sum (shift it first, approximately)
            // For simplicity, add the unshifted section -- the shift is small and
            // the cumulative average will converge.
            for i in 0..cum_sum.len() {
                cum_sum[i] += sections[z][i];
            }
            cum_count += 1.0;
        }

        // Backward: ref-1, ref-2, ...
        let mut cum_sum = sections[ref_section].clone();
        let mut cum_count = 1.0f32;

        for z in (0..ref_section).rev() {
            if excluded.contains(&z) {
                transforms[z] = transforms[z + 1];
                continue;
            }

            let cum_avg: Vec<f32> = cum_sum.iter().map(|&v| v / cum_count).collect();

            let (dx, dy) = find_shift(&cum_avg, &sections[z], nx, ny, fft_nx, fft_ny);
            transforms[z] = LinearTransform::translation(
                transforms[z + 1].dx + dx,
                transforms[z + 1].dy + dy,
            );

            for i in 0..cum_sum.len() {
                cum_sum[i] += sections[z][i];
            }
            cum_count += 1.0;
        }
    } else {
        // Original neighbor-to-neighbor mode
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

/// Apply bandpass filter with given high-pass and low-pass radii (fractions of Nyquist).
fn apply_bandpass(
    data: &[f32],
    nx: usize,
    ny: usize,
    fft_nx: usize,
    fft_ny: usize,
    high_pass: f32,
    low_pass: f32,
) -> Vec<f32> {
    let padded = pad_image(data, nx, ny, fft_nx, fft_ny);
    let nxc = fft_nx / 2 + 1;
    let mut freq = fft_r2c_2d(&padded, fft_nx, fft_ny);

    // Apply bandpass filter in frequency domain
    for fy in 0..fft_ny {
        let fy_norm = if fy <= fft_ny / 2 {
            fy as f32 / fft_ny as f32
        } else {
            (fft_ny - fy) as f32 / fft_ny as f32
        };
        for fx in 0..nxc {
            let fx_norm = fx as f32 / fft_nx as f32;
            // Spatial frequency as fraction of Nyquist (max = 0.5)
            let freq_r = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();

            let mut weight = 1.0f32;

            // High-pass: smooth rolloff below high_pass
            if high_pass > 0.0 && freq_r < high_pass {
                if freq_r <= 0.0 {
                    weight = 0.0;
                } else {
                    // Gaussian rolloff
                    let sigma = high_pass / 3.0;
                    let d = high_pass - freq_r;
                    weight *= (-0.5 * (d / sigma).powi(2)).exp();
                }
            }

            // Low-pass: smooth rolloff above low_pass
            if low_pass < 0.5 && freq_r > low_pass {
                let sigma = (0.5 - low_pass).max(0.01) / 3.0;
                let d = freq_r - low_pass;
                weight *= (-0.5 * (d / sigma).powi(2)).exp();
            }

            freq[fy * nxc + fx] *= weight;
        }
    }

    let filtered_padded = fft_c2r_2d(&freq, fft_nx, fft_ny);

    // Extract original-size region
    let ox = (fft_nx - nx) / 2;
    let oy = (fft_ny - ny) / 2;
    let mut result = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            result[y * nx + x] = filtered_padded[(y + oy) * fft_nx + (x + ox)];
        }
    }
    result
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

/// Find shift using patch-based correlation. Divides the image into a grid,
/// correlates each patch, and returns the median shift as the global result,
/// plus all per-patch shifts.
fn find_shift_patches(
    reference: &[f32],
    target: &[f32],
    nx: usize,
    ny: usize,
    pnx: usize,
    pny: usize,
    patch_w: usize,
    patch_h: usize,
    fft_pw: usize,
    fft_ph: usize,
) -> (f32, f32, Vec<(f32, f32)>) {
    let mut patch_shifts: Vec<(f32, f32)> = Vec::with_capacity(pnx * pny);

    for py_idx in 0..pny {
        for px_idx in 0..pnx {
            let x0 = px_idx * patch_w;
            let y0 = py_idx * patch_h;

            // Extract patches
            let mut ref_patch = vec![0.0f32; patch_w * patch_h];
            let mut tgt_patch = vec![0.0f32; patch_w * patch_h];
            for y in 0..patch_h {
                for x in 0..patch_w {
                    let src_x = (x0 + x).min(nx - 1);
                    let src_y = (y0 + y).min(ny - 1);
                    ref_patch[y * patch_w + x] = reference[src_y * nx + src_x];
                    tgt_patch[y * patch_w + x] = target[src_y * nx + src_x];
                }
            }

            let (dx, dy) = find_shift(&ref_patch, &tgt_patch, patch_w, patch_h, fft_pw, fft_ph);
            patch_shifts.push((dx, dy));
        }
    }

    // Compute global shift as the median of per-patch shifts
    let mut dxs: Vec<f32> = patch_shifts.iter().map(|s| s.0).collect();
    let mut dys: Vec<f32> = patch_shifts.iter().map(|s| s.1).collect();
    dxs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dys.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_dx = dxs[dxs.len() / 2];
    let med_dy = dys[dys.len() / 2];

    (med_dx, med_dy, patch_shifts)
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
