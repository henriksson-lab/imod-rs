use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_transforms::read_tilt_file;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Correct CTF (contrast transfer function) by phase-flipping strips of a
/// tilt series. Each image is divided into strips perpendicular to the tilt
/// axis; the defocus for each strip is computed from the tilt geometry, and
/// the CTF phase is flipped in Fourier space.
#[derive(Parser)]
#[command(name = "ctfphaseflip", about = "CTF phase-flip correction of a tilt series")]
struct Args {
    /// Input tilt series MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output corrected MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Defocus value in nm (negative = underfocus, as is convention)
    #[arg(short = 'd', long, default_value_t = -3000.0)]
    defocus: f32,

    /// Voltage in kV
    #[arg(short = 'v', long, default_value_t = 300.0)]
    voltage: f32,

    /// Spherical aberration in mm
    #[arg(short = 'c', long, default_value_t = 2.7)]
    cs: f32,

    /// Pixel size in Angstroms (overrides header if set)
    #[arg(short = 'p', long)]
    pixel_size: Option<f32>,

    /// Width of strips in pixels
    #[arg(short = 'w', long, default_value_t = 256)]
    strip_width: usize,
}

fn main() {
    let args = Args::parse();

    let tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let pixel_a = args.pixel_size.unwrap_or(h.pixel_size_x());

    if tilt_angles.len() < nz {
        eprintln!("Warning: tilt file has {} angles but stack has {} sections", tilt_angles.len(), nz);
    }

    // Electron wavelength in Angstroms
    let wavelength = electron_wavelength(args.voltage);
    let cs_a = args.cs as f64 * 1e7; // mm -> Angstroms
    let defocus_a = args.defocus as f64 * 10.0; // nm -> Angstroms

    eprintln!(
        "ctfphaseflip: voltage={:.0}kV, Cs={:.1}mm, defocus={:.0}nm, pixel={:.2}A, strips={}px",
        args.voltage, args.cs, args.defocus, pixel_a, args.strip_width
    );

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.add_label("ctfphaseflip: CTF phase correction");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    let mut planner = FftPlanner::<f32>::new();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let tilt_deg = if z < tilt_angles.len() { tilt_angles[z] } else { 0.0 };
        let tilt_rad = (tilt_deg as f64) * PI as f64 / 180.0;

        let mut output = data.clone();

        // Process in vertical strips
        let n_strips = (nx + args.strip_width - 1) / args.strip_width;
        let center_x = nx as f64 / 2.0;

        for strip in 0..n_strips {
            let x_start = strip * args.strip_width;
            let x_end = (x_start + args.strip_width).min(nx);
            let _strip_w = x_end - x_start;
            let strip_center_x = (x_start + x_end) as f64 / 2.0;

            // Defocus varies across the image due to tilt
            let dx_from_center = (strip_center_x - center_x) * pixel_a as f64;
            let strip_defocus = defocus_a + dx_from_center * tilt_rad.sin();

            // Extract strip, FFT each column, phase-flip, IFFT
            let fft_col = planner.plan_fft_forward(ny);
            let ifft_col = planner.plan_fft_inverse(ny);

            for x in x_start..x_end {
                // Extract column
                let mut col: Vec<Complex<f32>> = (0..ny)
                    .map(|y| Complex::new(data[y * nx + x], 0.0))
                    .collect();

                fft_col.process(&mut col);

                // Apply CTF phase flip
                for fy in 0..ny {
                    let freq_y = if fy <= ny / 2 { fy as f64 } else { fy as f64 - ny as f64 };
                    let s = freq_y / (ny as f64 * pixel_a as f64); // spatial frequency
                    let s2 = s * s;

                    let ctf_phase = PI as f64 * wavelength * s2
                        * (strip_defocus - 0.5 * cs_a * wavelength * wavelength * s2);

                    // Flip phase where CTF is negative
                    if ctf_phase.sin() < 0.0 {
                        col[fy] = -col[fy];
                    }
                }

                ifft_col.process(&mut col);

                let scale = 1.0 / ny as f32;
                for y in 0..ny {
                    output[y * nx + x] = col[y].re * scale;
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&output);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&output).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
    eprintln!("ctfphaseflip: corrected {} sections", nz);
}

/// Relativistic electron wavelength in Angstroms.
fn electron_wavelength(voltage_kv: f32) -> f64 {
    let v = voltage_kv as f64 * 1000.0; // Volts
    let m0 = 9.10938e-31; // electron rest mass (kg)
    let e = 1.60218e-19; // electron charge (C)
    let c = 2.99792e8; // speed of light (m/s)
    let h = 6.62607e-34; // Planck constant (J·s)

    let lambda_m = h / (2.0 * m0 * e * v * (1.0 + e * v / (2.0 * m0 * c * c))).sqrt();
    lambda_m * 1e10 // meters -> Angstroms
}
