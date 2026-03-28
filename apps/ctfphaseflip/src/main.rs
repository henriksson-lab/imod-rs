use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_transforms::read_tilt_file;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

mod gpu;

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

    /// Maximum strip width in pixels. When not specified, auto-computed from
    /// pixel size and defocus to prevent aliasing: max_width = pixel_size / (wavelength * defocus_range_per_pixel).
    /// The actual strip_width is clamped to this maximum.
    #[arg(long = "max-strip-width")]
    max_strip_width: Option<usize>,

    /// Amplitude contrast fraction (w). The full CTF is:
    /// CTF = -sin(chi)*sqrt(1-w^2) + cos(chi)*w
    /// Typical values: 0.07 for cryo, 0.1-0.15 for negative stain.
    #[arg(long = "amplitude-contrast", default_value_t = 0.07)]
    amplitude_contrast: f32,

    /// Minimum spatial frequency (in 1/Angstrom) below which CTF correction
    /// is not applied. Frequencies below this cutoff are left unchanged (CTF=1).
    #[arg(long = "cuton-freq", default_value_t = 0.0)]
    cuton_freq: f32,

    /// Use GPU (wgpu compute shaders) for CTF correction. Falls back to CPU
    /// if no suitable GPU adapter is found.
    #[arg(long = "gpu", default_value_t = false)]
    gpu: bool,

    /// Phase plate constant phase shift in degrees. When non-zero, this
    /// constant phase is added to the CTF aberration phase before computing
    /// the correction. Typical values: 90 degrees for a Volta phase plate.
    #[arg(long = "plate-phase", default_value_t = 0.0)]
    plate_phase: f32,
}

/// Per-view defocus information, potentially with astigmatism.
#[derive(Clone, Debug)]
struct ViewDefocus {
    defocus1: f32, // nm
    defocus2: f32, // nm (same as defocus1 if no astigmatism)
    astig_angle: f32, // degrees
}

/// Parse a defocus file. Each line: "view defocus1 [defocus2 astig_angle]"
/// defocus values in nm. Returns (entries, view_indices).
fn read_defocus_file(path: &str) -> Result<(Vec<ViewDefocus>, Vec<usize>), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read defocus file {}: {}", path, e))?;

    let mut entries = Vec::new();
    let mut view_indices = Vec::new();
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
        let view: usize = fields[0].parse().map_err(|_| {
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

        view_indices.push(view);
        entries.push(ViewDefocus {
            defocus1: def1,
            defocus2: def2,
            astig_angle: astig_ang,
        });
    }
    Ok((entries, view_indices))
}

/// Interpolate defocus values for views that are missing from the defocus file.
/// Given a sparse set of per-view defocus entries, linearly interpolate
/// defocus1, defocus2, and astig_angle for intermediate views.
/// `entries` are the parsed defocus values (in file order, one per listed view).
/// `view_indices` are the view numbers from the file. `n_views` is the total.
fn interpolate_defocus(entries: &[ViewDefocus], view_indices: &[usize], n_views: usize) -> Vec<ViewDefocus> {
    if entries.is_empty() || n_views == 0 {
        return Vec::new();
    }
    let mut result = vec![ViewDefocus { defocus1: 0.0, defocus2: 0.0, astig_angle: 0.0 }; n_views];
    // Mark which views have data
    let mut has_data = vec![false; n_views];
    for (&vi, entry) in view_indices.iter().zip(entries.iter()) {
        if vi < n_views {
            result[vi] = entry.clone();
            has_data[vi] = true;
        }
    }
    // Interpolate gaps
    for v in 0..n_views {
        if has_data[v] { continue; }
        // Find nearest defined view before and after
        let before = (0..v).rev().find(|&i| has_data[i]);
        let after = ((v + 1)..n_views).find(|&i| has_data[i]);
        match (before, after) {
            (Some(b), Some(a)) => {
                let t = (v - b) as f32 / (a - b) as f32;
                result[v] = ViewDefocus {
                    defocus1: result[b].defocus1 + t * (result[a].defocus1 - result[b].defocus1),
                    defocus2: result[b].defocus2 + t * (result[a].defocus2 - result[b].defocus2),
                    astig_angle: result[b].astig_angle + t * (result[a].astig_angle - result[b].astig_angle),
                };
            }
            (Some(b), None) => { result[v] = result[b].clone(); }
            (None, Some(a)) => { result[v] = result[a].clone(); }
            (None, None) => {} // should not happen
        }
    }
    result
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

    // Load per-view defocus if provided, with interpolation for missing views
    let view_defocus_raw: Option<(Vec<ViewDefocus>, Vec<usize>)> = args.defocus_file.as_ref().map(|path| {
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

    // Interpolate defocus for all views (fills gaps between specified views)
    let view_defocus: Option<Vec<ViewDefocus>> = view_defocus_raw.map(|(entries, indices)| {
        if entries.len() < nz {
            eprintln!(
                "ctfphaseflip: defocus file has {} entries for {} views, interpolating missing values",
                entries.len(), nz
            );
        }
        interpolate_defocus(&entries, &indices, nz)
    });

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
    let amp_contrast = args.amplitude_contrast as f64;
    let cuton_freq = args.cuton_freq as f64;

    // Default defocus in Angstroms (when no defocus file)
    let _default_defocus_a = args.defocus as f64 * 10.0; // nm -> Angstroms

    let tilt_axis_rad = (args.tilt_axis_angle as f64) * PI as f64 / 180.0;
    let plate_phase_rad = (args.plate_phase as f64) * PI as f64 / 180.0;

    // Compute effective strip width, clamping to max_strip_width if given,
    // or auto-computing a safe maximum from defocus and pixel size.
    let strip_width = {
        let max_w = if let Some(mw) = args.max_strip_width {
            mw
        } else {
            // Auto-compute: max safe strip width based on defocus gradient.
            // The defocus changes across the strip due to tilt. The maximum
            // frequency that can be corrected without aliasing is limited by
            // defocus variation within the strip.
            // max_width = pixel_size / (wavelength * |defocus_range_per_pixel|)
            // defocus_range_per_pixel ~ max(|defocus|) * sin(max_tilt) / image_width
            let max_defocus_a = if let Some(ref vd) = view_defocus {
                vd.iter()
                    .map(|v| v.defocus1.abs().max(v.defocus2.abs()) as f64 * 10.0)
                    .fold(0.0f64, f64::max)
            } else {
                (args.defocus.abs() as f64) * 10.0
            };
            if max_defocus_a > 1.0 && wavelength > 0.0 {
                let defocus_range_per_pixel = max_defocus_a / (nx as f64);
                let max_safe = (pixel_a as f64) / (wavelength * defocus_range_per_pixel);
                let max_safe = max_safe.max(16.0) as usize;
                max_safe
            } else {
                args.strip_width // no meaningful limit
            }
        };
        args.strip_width.min(max_w)
    };

    // Try to initialise GPU session if requested
    let gpu_session = if args.gpu {
        match gpu::GpuCtfSession::new() {
            Some(session) => {
                eprintln!("ctfphaseflip: GPU acceleration enabled");
                Some(session)
            }
            None => {
                eprintln!("ctfphaseflip: GPU init failed, falling back to CPU");
                None
            }
        }
    } else {
        None
    };

    eprintln!(
        "ctfphaseflip: voltage={:.0}kV, Cs={:.1}mm, pixel={:.2}A, strips={}px, tilt_axis={:.1}deg",
        args.voltage, args.cs, pixel_a, strip_width, args.tilt_axis_angle
    );
    if args.plate_phase.abs() > 1e-6 {
        eprintln!("ctfphaseflip: phase plate correction={:.1}deg", args.plate_phase);
    }
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

        let strip_w = strip_width as f64;
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

            // Apply CTF phase flip in 2D with amplitude contrast support.
            // The full CTF is: CTF(s) = -sin(chi)*sqrt(1-w^2) + cos(chi)*w
            // where chi is the phase aberration and w is the amplitude contrast.
            // We flip the sign of Fourier coefficients wherever CTF < 0.
            // Frequencies below cuton_freq are left unchanged.
            if let Some(ref gpu) = gpu_session {
                // GPU path: split complex data into separate re/im arrays
                let mut re_data: Vec<f32> = strip_data.iter().map(|c| c.re).collect();
                let mut im_data: Vec<f32> = strip_data.iter().map(|c| c.im).collect();

                let gpu_params = gpu::CtfParams {
                    sw: sw as u32,
                    sh: sh as u32,
                    pixel_a: pixel_a,
                    wavelength: wavelength as f32,
                    cs_a: cs_a as f32,
                    amp_contrast: amp_contrast as f32,
                    cuton_freq: cuton_freq as f32,
                    defocus1: strip_base_defocus as f32,
                    defocus2: strip_base_defocus2 as f32,
                    astig_angle_rad: astig_angle_rad as f32,
                    has_astigmatism: if has_astigmatism { 1 } else { 0 },
                    plate_phase_rad: plate_phase_rad as f32,
                    tilt_axis_rad: tilt_axis_rad as f32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };

                gpu.correct_strip(&mut re_data, &mut im_data, gpu_params);

                // Write results back into complex array
                for i in 0..strip_data.len() {
                    strip_data[i] = Complex::new(re_data[i], im_data[i]);
                }
            } else {
                // CPU path
                let w = amp_contrast;
                let w_phase = (1.0 - w * w).sqrt();
                for fy_idx in 0..sh {
                    let freq_y = if fy_idx <= sh / 2 {
                        fy_idx as f64
                    } else {
                        fy_idx as f64 - sh as f64
                    };

                    for fx_idx in 0..sw {
                        let freq_x = if fx_idx <= sw / 2 {
                            fx_idx as f64
                        } else {
                            fx_idx as f64 - sw as f64
                        };

                        // Rotate frequency coordinates by the tilt axis angle
                        // to account for directional defocus in the frequency domain
                        let (rot_freq_x, rot_freq_y) = if tilt_axis_rad.abs() > 1e-6 {
                            let cos_ta = tilt_axis_rad.cos();
                            let sin_ta = tilt_axis_rad.sin();
                            (
                                freq_x * cos_ta + freq_y * sin_ta,
                                -freq_x * sin_ta + freq_y * cos_ta,
                            )
                        } else {
                            (freq_x, freq_y)
                        };

                        let sx = rot_freq_x / (sw as f64 * pixel_a as f64);
                        let sy = rot_freq_y / (sh as f64 * pixel_a as f64);

                        let s2 = sx * sx + sy * sy;
                        let s = s2.sqrt();

                        // Skip correction below cuton frequency
                        if s < cuton_freq {
                            continue;
                        }

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

                        // CTF phase: chi(s) = pi * lambda * s^2 * (defocus - 0.5 * Cs * lambda^2 * s^2)
                        let chi = std::f64::consts::PI * wavelength * s2
                            * (def_for_ctf - 0.5 * cs_a * wavelength * wavelength * s2);

                        // Add phase plate constant phase shift
                        let chi_total = chi + plate_phase_rad;

                        // Full CTF with amplitude contrast:
                        // CTF = -sin(chi)*sqrt(1-w^2) + cos(chi)*w
                        let ctf = -chi_total.sin() * w_phase + chi_total.cos() * w;

                        // Phase-flip: negate Fourier coefficient where CTF < 0
                        if ctf < 0.0 {
                            strip_data[fy_idx * sw + fx_idx] =
                                -strip_data[fy_idx * sw + fx_idx];
                        }
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
