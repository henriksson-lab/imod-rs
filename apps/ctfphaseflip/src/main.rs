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
///
/// Supports per-view defocus files with astigmatism, 2D strip processing,
/// and configurable tilt axis angle.
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

    /// Defocus value in nm (negative = underfocus, as is convention).
    /// Used when no defocus file is provided.
    #[arg(short = 'd', long, default_value_t = -3000.0)]
    defocus: f32,

    /// Defocus file: each line has "view defocus1 [defocus2 astig_angle]".
    /// defocus values in nm. If only one defocus column, it is used for both.
    #[arg(long = "defocus-file")]
    defocus_file: Option<String>,

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

    /// Rotation angle of the tilt axis in degrees (from Y axis, counterclockwise positive)
    #[arg(long = "tilt-axis-angle", default_value_t = 0.0)]
    tilt_axis_angle: f32,
}

/// Per-view defocus information, potentially with astigmatism.
#[derive(Clone, Debug)]
struct ViewDefocus {
    defocus1: f32, // nm
    defocus2: f32, // nm (same as defocus1 if no astigmatism)
    astig_angle: f32, // degrees
}

/// Parse a defocus file. Each line: "view defocus1 [defocus2 astig_angle]"
/// defocus values in nm.
fn read_defocus_file(path: &str) -> Result<Vec<ViewDefocus>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read defocus file {}: {}", path, e))?;

    let mut entries = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(format!(
                "Defocus file line {}: expected at least 'view defocus1', got '{}'",
                line_num + 1,
                line
            ));
        }
        let _view: usize = fields[0].parse().map_err(|_| {
            format!("Defocus file line {}: invalid view number '{}'", line_num + 1, fields[0])
        })?;
        let def1: f32 = fields[1].parse().map_err(|_| {
            format!("Defocus file line {}: invalid defocus1 '{}'", line_num + 1, fields[1])
        })?;

        let (def2, astig_ang) = if fields.len() >= 4 {
            let d2: f32 = fields[2].parse().map_err(|_| {
                format!("Defocus file line {}: invalid defocus2 '{}'", line_num + 1, fields[2])
            })?;
            let aa: f32 = fields[3].parse().map_err(|_| {
                format!("Defocus file line {}: invalid astig_angle '{}'", line_num + 1, fields[3])
            })?;
            (d2, aa)
        } else if fields.len() == 3 {
            // Could be defocus2 without astig angle, or something else;
            // treat as defocus2 with astig_angle = 0
            let d2: f32 = fields[2].parse().map_err(|_| {
                format!("Defocus file line {}: invalid defocus2 '{}'", line_num + 1, fields[2])
            })?;
            (d2, 0.0)
        } else {
            // Only one defocus value: no astigmatism
            (def1, 0.0)
        };

        entries.push(ViewDefocus {
            defocus1: def1,
            defocus2: def2,
            astig_angle: astig_ang,
        });
    }
    Ok(entries)
}

/// Compute effective defocus at a given angle from the astigmatism axis.
/// angle_rad: the direction angle in radians (relative to X axis in frequency space)
/// astig_angle_rad: the astigmatism angle in radians
/// Returns effective defocus in Angstroms.
#[inline]
fn effective_defocus(def1_a: f64, def2_a: f64, angle_rad: f64, astig_angle_rad: f64) -> f64 {
    let da = angle_rad - astig_angle_rad;
    let cos2 = da.cos() * da.cos();
    let sin2 = da.sin() * da.sin();
    def1_a * cos2 + def2_a * sin2
}

fn main() {
    let args = Args::parse();

    let tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    // Load per-view defocus if provided
    let view_defocus: Option<Vec<ViewDefocus>> = args.defocus_file.as_ref().map(|path| {
        read_defocus_file(path).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        })
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
        eprintln!(
            "Warning: tilt file has {} angles but stack has {} sections",
            tilt_angles.len(),
            nz
        );
    }

    // Electron wavelength in Angstroms
    let wavelength = electron_wavelength(args.voltage);
    let cs_a = args.cs as f64 * 1e7; // mm -> Angstroms

    // Default defocus in Angstroms (when no defocus file)
    let _default_defocus_a = args.defocus as f64 * 10.0; // nm -> Angstroms

    let tilt_axis_rad = (args.tilt_axis_angle as f64) * PI as f64 / 180.0;

    eprintln!(
        "ctfphaseflip: voltage={:.0}kV, Cs={:.1}mm, pixel={:.2}A, strips={}px, tilt_axis={:.1}deg",
        args.voltage, args.cs, pixel_a, args.strip_width, args.tilt_axis_angle
    );
    if view_defocus.is_some() {
        eprintln!("ctfphaseflip: using per-view defocus file with astigmatism support");
    } else {
        eprintln!("ctfphaseflip: defocus={:.0}nm", args.defocus);
    }

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.add_label("ctfphaseflip: CTF phase correction (2D strips)");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    let mut planner = FftPlanner::<f32>::new();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    // Precompute the strip geometry.
    // Strips are oriented along the tilt axis direction. The "strip coordinate"
    // is the distance perpendicular to the tilt axis from the image center.
    // For tilt_axis_angle = 0, tilt axis is vertical (along Y), so strips are
    // horizontal bands and the perpendicular direction is X.
    //
    // In general, the perpendicular direction to the tilt axis is:
    //   perp = (cos(tilt_axis_angle), sin(tilt_axis_angle))
    // (the tilt axis itself is at angle tilt_axis_angle from Y, which means
    //  the axis direction is (-sin(tilt_axis_angle), cos(tilt_axis_angle))).

    let perp_x = tilt_axis_rad.cos();
    let perp_y = tilt_axis_rad.sin();

    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let tilt_deg = if z < tilt_angles.len() {
            tilt_angles[z]
        } else {
            0.0
        };
        let tilt_rad = (tilt_deg as f64) * PI as f64 / 180.0;

        // Get defocus for this view
        let (def1_nm, def2_nm, astig_angle_deg) = if let Some(ref vd) = view_defocus {
            if z < vd.len() {
                (vd[z].defocus1, vd[z].defocus2, vd[z].astig_angle)
            } else {
                (args.defocus, args.defocus, 0.0)
            }
        } else {
            (args.defocus, args.defocus, 0.0)
        };
        let def1_a = def1_nm as f64 * 10.0; // nm -> Angstroms
        let def2_a = def2_nm as f64 * 10.0;
        let astig_angle_rad = (astig_angle_deg as f64) * PI as f64 / 180.0;
        let has_astigmatism = (def1_a - def2_a).abs() > 0.01;

        let mut output = data.clone();

        // Compute the range of the perpendicular coordinate across the image
        // to determine strip boundaries.
        // The perpendicular distance for pixel (x, y) from center:
        //   d = (x - cx) * perp_x + (y - cy) * perp_y
        let mut min_perp = f64::MAX;
        let mut max_perp = f64::MIN;
        // Check corners
        for &(cx, cy) in &[
            (0.0, 0.0),
            (nx as f64, 0.0),
            (0.0, ny as f64),
            (nx as f64, ny as f64),
        ] {
            let d = (cx - center_x) * perp_x + (cy - center_y) * perp_y;
            if d < min_perp {
                min_perp = d;
            }
            if d > max_perp {
                max_perp = d;
            }
        }

        let strip_w = args.strip_width as f64;
        let n_strips = ((max_perp - min_perp) / strip_w).ceil() as usize;

        // For 2D strip processing, we need to extract rectangular strips
        // oriented along the tilt axis. For simplicity and correctness, we
        // process each strip as follows:
        //
        // 1. Identify which pixels belong to this strip based on their
        //    perpendicular distance from center.
        // 2. For each strip, compute the center defocus from tilt geometry.
        // 3. Extract a rectangular sub-image (in the original pixel grid)
        //    that covers the strip, perform 2D FFT, apply CTF correction,
        //    and IFFT back.
        //
        // For efficiency with non-zero tilt axis angles, we use overlapping
        // rectangular patches and blend at boundaries. However, for the
        // primary case (tilt_axis_angle near 0), strips are nearly vertical
        // bands, so we extract column bands.
        //
        // Implementation: We divide the image into strips along the
        // perpendicular direction. For each strip, we extract all columns
        // that overlap with it and do a 2D FFT on the strip region.

        // For general tilt axis angle: process strips as rectangular regions
        // along the perpendicular coordinate.
        // We'll use a weighting approach: each pixel accumulates its corrected
        // value weighted by strip membership.
        let mut weighted_sum = vec![0.0f32; nx * ny];
        let mut weight_total = vec![0.0f32; nx * ny];

        for strip_idx in 0..n_strips {
            let strip_center_perp =
                min_perp + (strip_idx as f64 + 0.5) * strip_w;

            // Defocus at this strip position (distance from tilt axis center)
            let dx_a = strip_center_perp * pixel_a as f64;
            let strip_base_defocus = def1_a + dx_a * tilt_rad.sin();
            let strip_base_defocus2 = def2_a + dx_a * tilt_rad.sin();

            // Determine bounding box of pixels in this strip
            // A pixel (x,y) is in this strip if its perp distance is within
            // [strip_center_perp - strip_w/2, strip_center_perp + strip_w/2]
            let strip_lo = strip_center_perp - strip_w / 2.0;
            let strip_hi = strip_center_perp + strip_w / 2.0;

            // Find bounding box in pixel coordinates
            // We need to find the min/max x and y that can satisfy:
            //   strip_lo <= (x - cx)*perp_x + (y - cy)*perp_y <= strip_hi
            // To get a bounding box, scan the edges.
            let mut bb_x0 = nx;
            let mut bb_x1 = 0usize;
            let mut bb_y0 = ny;
            let mut bb_y1 = 0usize;

            for y in 0..ny {
                for x in 0..nx {
                    let d = (x as f64 - center_x) * perp_x + (y as f64 - center_y) * perp_y;
                    if d >= strip_lo && d < strip_hi {
                        if x < bb_x0 {
                            bb_x0 = x;
                        }
                        if x >= bb_x1 {
                            bb_x1 = x + 1;
                        }
                        if y < bb_y0 {
                            bb_y0 = y;
                        }
                        if y >= bb_y1 {
                            bb_y1 = y + 1;
                        }
                    }
                }
            }

            if bb_x1 <= bb_x0 || bb_y1 <= bb_y0 {
                continue;
            }

            let sw = bb_x1 - bb_x0;
            let sh = bb_y1 - bb_y0;

            // Extract strip sub-image
            let mut strip_data: Vec<Complex<f32>> = Vec::with_capacity(sw * sh);
            for y in bb_y0..bb_y1 {
                for x in bb_x0..bb_x1 {
                    strip_data.push(Complex::new(data[y * nx + x], 0.0));
                }
            }

            // 2D FFT: apply row FFTs, then column FFTs
            let fft_row = planner.plan_fft_forward(sw);
            let fft_col = planner.plan_fft_forward(sh);

            // Row FFTs
            for row in 0..sh {
                let start = row * sw;
                let end = start + sw;
                fft_row.process(&mut strip_data[start..end]);
            }

            // Column FFTs (transpose-process-transpose approach via manual iteration)
            {
                let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); sh];
                for col in 0..sw {
                    for row in 0..sh {
                        col_buf[row] = strip_data[row * sw + col];
                    }
                    fft_col.process(&mut col_buf);
                    for row in 0..sh {
                        strip_data[row * sw + col] = col_buf[row];
                    }
                }
            }

            // Apply CTF phase flip in 2D
            for fy_idx in 0..sh {
                let freq_y = if fy_idx <= sh / 2 {
                    fy_idx as f64
                } else {
                    fy_idx as f64 - sh as f64
                };
                let sy = freq_y / (sh as f64 * pixel_a as f64);

                for fx_idx in 0..sw {
                    let freq_x = if fx_idx <= sw / 2 {
                        fx_idx as f64
                    } else {
                        fx_idx as f64 - sw as f64
                    };
                    let sx = freq_x / (sw as f64 * pixel_a as f64);

                    let s2 = sx * sx + sy * sy;

                    // Compute effective defocus for this frequency direction
                    let def_for_ctf = if has_astigmatism {
                        let angle = sy.atan2(sx);
                        effective_defocus(
                            strip_base_defocus,
                            strip_base_defocus2,
                            angle,
                            astig_angle_rad,
                        )
                    } else {
                        strip_base_defocus
                    };

                    let ctf_phase = PI as f64 * wavelength * s2
                        * (def_for_ctf - 0.5 * cs_a * wavelength * wavelength * s2);

                    // Flip phase where CTF is negative
                    if ctf_phase.sin() < 0.0 {
                        strip_data[fy_idx * sw + fx_idx] =
                            -strip_data[fy_idx * sw + fx_idx];
                    }
                }
            }

            // 2D IFFT
            let ifft_row = planner.plan_fft_inverse(sw);
            let ifft_col = planner.plan_fft_inverse(sh);

            // Column IFFTs
            {
                let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); sh];
                for col in 0..sw {
                    for row in 0..sh {
                        col_buf[row] = strip_data[row * sw + col];
                    }
                    ifft_col.process(&mut col_buf);
                    for row in 0..sh {
                        strip_data[row * sw + col] = col_buf[row];
                    }
                }
            }

            // Row IFFTs
            for row in 0..sh {
                let start = row * sw;
                let end = start + sw;
                ifft_row.process(&mut strip_data[start..end]);
            }

            let scale = 1.0 / (sw * sh) as f32;

            // Write corrected values back, weighted by strip membership
            for y in bb_y0..bb_y1 {
                for x in bb_x0..bb_x1 {
                    let d = (x as f64 - center_x) * perp_x + (y as f64 - center_y) * perp_y;
                    if d >= strip_lo && d < strip_hi {
                        // Triangular weighting: full weight at center, tapering at edges
                        let dist_from_center =
                            ((d - strip_center_perp) / (strip_w / 2.0)).abs() as f32;
                        let w = (1.0 - dist_from_center).max(0.0);

                        let local_y = y - bb_y0;
                        let local_x = x - bb_x0;
                        let val = strip_data[local_y * sw + local_x].re * scale;
                        weighted_sum[y * nx + x] += val * w;
                        weight_total[y * nx + x] += w;
                    }
                }
            }
        }

        // Combine weighted results
        for idx in 0..nx * ny {
            if weight_total[idx] > 1e-10 {
                output[idx] = weighted_sum[idx] / weight_total[idx];
            }
            // else keep original value
        }

        let (smin, smax, smean) = min_max_mean(&output);
        if smin < gmin {
            gmin = smin;
        }
        if smax > gmax {
            gmax = smax;
        }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&output).unwrap();
    }

    writer
        .finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32)
        .unwrap();
    eprintln!("ctfphaseflip: corrected {} sections (2D strips)", nz);
}

/// Relativistic electron wavelength in Angstroms.
fn electron_wavelength(voltage_kv: f32) -> f64 {
    let v = voltage_kv as f64 * 1000.0; // Volts
    let m0 = 9.10938e-31; // electron rest mass (kg)
    let e = 1.60218e-19; // electron charge (C)
    let c = 2.99792e8; // speed of light (m/s)
    let h = 6.62607e-34; // Planck constant (J*s)

    let lambda_m = h / (2.0 * m0 * e * v * (1.0 + e * v / (2.0 * m0 * c * c))).sqrt();
    lambda_m * 1e10 // meters -> Angstroms
}
