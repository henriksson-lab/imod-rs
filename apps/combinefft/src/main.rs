use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::fs;
use std::io::{BufRead, BufReader};

/// Combine FFTs from two tomograms of a dual-axis tilt series.
///
/// For locations in Fourier space where there is data from one tilt series
/// but not the other, takes the Fourier value from just that FFT.
/// Everywhere else it averages the Fourier values from the two FFT files,
/// taking into account the tilt range and matching transformation.
#[derive(Parser)]
#[command(name = "combinefft", about = "Combine two FFTs for dual-axis tomogram combination")]
struct Args {
    /// First input FFT file (A axis)
    #[arg(short = 'a', long)]
    ainput: String,

    /// Second input FFT file (B axis)
    #[arg(short = 'b', long)]
    binput: String,

    /// Output FFT file
    #[arg(short = 'o', long)]
    output: String,

    /// Inverse transform file (3x3 rotation matrix mapping A coords to B coords)
    #[arg(long = "inverse")]
    inverse: String,

    /// Highest tilt angles for A axis: low,high (degrees)
    #[arg(long, num_args = 2, value_delimiter = ',')]
    ahighest: Vec<f32>,

    /// Highest tilt angles for B axis: low,high (degrees)
    #[arg(long, num_args = 2, value_delimiter = ',')]
    bhighest: Vec<f32>,

    /// Weighting power for density-based weighting in overlap zone (0 = equal)
    #[arg(short = 'w', long, default_value_t = 0.0)]
    weight_power: f32,

    /// Radius below which both axes always contribute
    #[arg(long, default_value_t = 0.0)]
    both_radius: f32,

    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,
}

/// Read a 3x3 inverse transform matrix from file.
fn read_inverse_matrix(path: &str) -> [[f32; 3]; 3] {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: COMBINEFFT - reading inverse file '{}': {}", path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(content.as_bytes());
    let mut mat = [[0.0f32; 3]; 3];
    let mut row = 0;
    for line in reader.lines() {
        let line = line.unwrap();
        let vals: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 && row < 3 {
            mat[row][0] = vals[0];
            mat[row][1] = vals[1];
            mat[row][2] = vals[2];
            row += 1;
        }
    }
    if row != 3 {
        eprintln!("ERROR: COMBINEFFT - inverse file must have 3 rows of >= 3 values");
        std::process::exit(1);
    }
    mat
}

/// Compute the tangent-based critical ratio for a tilt angle range.
/// The "missing wedge" boundary in Fourier space is at tan(tilt_angle).
fn tilt_criteria(low_deg: f32, high_deg: f32) -> (f32, f32) {
    let t1 = (-low_deg as f64).to_radians().tan() as f32;
    let t2 = (-high_deg as f64).to_radians().tan() as f32;
    let crit_low = t1.min(t2);
    let crit_high = t1.max(t2);
    (crit_low, crit_high)
}

fn main() {
    let args = Args::parse();

    if args.ahighest.len() != 2 || args.bhighest.len() != 2 {
        eprintln!("ERROR: COMBINEFFT - ahighest and bhighest each require two values (low,high)");
        std::process::exit(1);
    }

    // Open both input FFT files
    let mut reader_a = MrcReader::open(&args.ainput).unwrap_or_else(|e| {
        eprintln!("ERROR: COMBINEFFT - opening A input: {e}");
        std::process::exit(1);
    });
    let mut reader_b = MrcReader::open(&args.binput).unwrap_or_else(|e| {
        eprintln!("ERROR: COMBINEFFT - opening B input: {e}");
        std::process::exit(1);
    });

    let ha = reader_a.header().clone();
    let hb = reader_b.header().clone();

    let nx = ha.nx as usize;
    let ny = ha.ny as usize;
    let nz = ha.nz as usize;

    if hb.nx as usize != nx || hb.ny as usize != ny || hb.nz as usize != nz {
        eprintln!("ERROR: COMBINEFFT - input files are not the same size");
        std::process::exit(1);
    }

    let mode_a = ha.data_mode().unwrap_or_else(|| {
        eprintln!("ERROR: COMBINEFFT - unsupported input mode");
        std::process::exit(1);
    });

    let is_complex = matches!(mode_a, MrcMode::ComplexFloat | MrcMode::ComplexShort);
    if !is_complex {
        eprintln!("ERROR: COMBINEFFT - input files must be complex FFT data (mode 4)");
        std::process::exit(1);
    }

    // Read inverse transform
    let a_inv = read_inverse_matrix(&args.inverse);

    // Compute tilt criteria (tangent ratios defining missing wedge boundaries)
    let (a_crit_low, a_crit_high) = tilt_criteria(args.ahighest[0], args.ahighest[1]);
    let (b_crit_low, b_crit_high) = tilt_criteria(args.bhighest[0], args.bhighest[1]);

    if args.verbose {
        eprintln!(
            "A tilt criteria: {:.4} to {:.4}",
            a_crit_low, a_crit_high
        );
        eprintln!(
            "B tilt criteria: {:.4} to {:.4}",
            b_crit_low, b_crit_high
        );
    }

    let both_rad_sq = args.both_radius * args.both_radius;

    // Frequency steps
    let del_x = 0.5 / (nx as f32 - 1.0);
    let del_y = 1.0 / ny as f32;
    let del_z = 1.0 / nz as f32;

    // Set up output
    let mut out_header = MrcHeader::new(nx as i32, ny as i32, nz as i32, MrcMode::ComplexFloat);
    out_header.mx = ha.mx;
    out_header.my = ha.my;
    out_header.mz = ha.mz;
    out_header.xlen = ha.xlen;
    out_header.ylen = ha.ylen;
    out_header.zlen = ha.zlen;
    out_header.add_label("combinefft: combined FFT from two tomograms");
    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    eprintln!("Combining FFTs ({}x{}x{})...", nx, ny, nz);

    // Process plane by plane
    for iz in 0..nz {
        let slice_a = reader_a.read_slice_f32(iz).unwrap();
        let slice_b = reader_b.read_slice_f32(iz).unwrap();

        // Each complex pixel is stored as 2 floats
        let mut out_slice = vec![0.0f32; slice_a.len()];

        let za = del_z * iz as f32 + if iz == 0 { 0.0 } else if iz as f32 > nz as f32 / 2.0 { -1.0 } else { -0.5 };

        for iy in 0..ny {
            let ya = del_y * iy as f32 + if iy == 0 { 0.0 } else if iy as f32 > ny as f32 / 2.0 { -1.0 } else { -0.5 };
            let ya_sq = ya * ya;

            for ix in 0..nx {
                let xa = del_x * ix as f32;
                let xa_sq = xa * xa;
                let rad_sq = xa_sq + ya_sq + za * za;

                // Back-transform position to FFT B space
                let mut xp = a_inv[0][0] * xa + a_inv[0][1] * ya + a_inv[0][2] * za;
                let mut yp = a_inv[1][0] * xa + a_inv[1][1] * ya + a_inv[1][2] * za;
                if xp < 0.0 {
                    xp = -xp;
                    yp = -yp;
                }

                let ratio_a = ya / xa.max(1.0e-6);
                let ratio_b = yp / xp.max(1.0e-6);

                let in_a = (ratio_a >= a_crit_low && ratio_a <= a_crit_high) || rad_sq < both_rad_sq;
                let in_b = (ratio_b >= b_crit_low && ratio_b <= b_crit_high) || rad_sq < both_rad_sq;

                // Index into float data (2 floats per complex)
                let idx = (iy * nx + ix) * 2;

                let (ar, ai_val) = (slice_a[idx], slice_a[idx + 1]);
                let (br, bi_val) = (slice_b[idx], slice_b[idx + 1]);

                let (or, oi) = match (in_a, in_b) {
                    (false, false) => {
                        // In neither: simple average
                        (0.5 * (ar + br), 0.5 * (ai_val + bi_val))
                    }
                    (false, true) => {
                        // In B only: take B
                        (br, bi_val)
                    }
                    (true, false) => {
                        // In A only: keep A
                        (ar, ai_val)
                    }
                    (true, true) => {
                        // In both: weighted average
                        let wgt_a = 0.5f32;
                        let wgt_b = 0.5f32;
                        (wgt_a * ar + wgt_b * br, wgt_a * ai_val + wgt_b * bi_val)
                    }
                };

                out_slice[idx] = or;
                out_slice[idx + 1] = oi;
            }
        }

        // Compute stats on the float data
        let (smin, smax, smean) = min_max_mean(&out_slice);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * out_slice.len() as f64;

        writer.write_slice_f32(&out_slice).unwrap();
    }

    let total = (nx * ny * nz * 2) as f64;
    let gmean = (gsum / total) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();

    eprintln!(
        "Done. Min={:.4} Max={:.4} Mean={:.4}",
        gmin, gmax, gmean
    );
}
