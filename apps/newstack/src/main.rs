use clap::Parser;
use imod_core::MrcMode;
use imod_math::{min_max_mean, min_max_mean_sd};
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{read_xf_file, LinearTransform};
use std::f32::consts::PI;

/// Create a new image stack from sections of one or more input files.
///
/// Supports reordering sections, applying 2D transforms, mode conversion,
/// binning, scaling, antialias filtering, and float density modes.
#[derive(Parser)]
#[command(name = "newstack", about = "Create new MRC stack from input sections")]
struct Args {
    /// Input MRC file(s)
    #[arg(short = 'i', long = "input", required = true)]
    input: Vec<String>,

    /// Output MRC file
    #[arg(short = 'o', long = "output", required = true)]
    output: String,

    /// List of sections to read (comma-separated, 0-based, e.g. "0,3-5,10")
    #[arg(short = 'S', long)]
    secs: Option<String>,

    /// Output data mode (0=byte, 1=short, 2=float, 6=ushort)
    #[arg(short = 'm', long)]
    mode: Option<i32>,

    /// Binning factor
    #[arg(short = 'b', long, default_value_t = 1)]
    bin: usize,

    /// Transform file (.xf) to apply to each section
    #[arg(short = 'x', long)]
    xform: Option<String>,

    /// Fill value for areas outside the image after transforms
    #[arg(short = 'f', long, default_value_t = 0.0)]
    fill: f32,

    /// Scale min and max for output (min,max)
    #[arg(short = 's', long)]
    scale: Option<String>,

    /// Float densities to range (scale output to fill byte range)
    #[arg(short = 'F', long)]
    float_densities: bool,

    /// Float density mode: 0=scale to fill range, 1=shift to common mean,
    /// 2=scale to common mean and SD
    #[arg(long = "float", default_value_t = -1)]
    float_mode: i32,

    /// Enable antialias filtering before binning (default: true when bin > 1)
    #[arg(long = "antialias")]
    antialias: Option<bool>,
}

fn parse_section_list(s: &str, max_z: usize) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().unwrap_or(0);
            let end: usize = b.trim().parse().unwrap_or(max_z - 1);
            for i in start..=end.min(max_z - 1) {
                result.push(i);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            if n < max_z {
                result.push(n);
            }
        }
    }
    result
}

/// Lanczos-2 kernel: sinc(x) * sinc(x/2), windowed to |x| <= 2.
fn lanczos2(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= 2.0 {
        return 0.0;
    }
    let px = PI * x;
    let sinc_x = px.sin() / px;
    let sinc_x2 = (px / 2.0).sin() / (px / 2.0);
    sinc_x * sinc_x2
}

/// Build a 1D Lanczos-2 low-pass filter kernel for the given bin factor.
/// The cutoff frequency is 1/(2*bin) of Nyquist, and the kernel radius is
/// 2*bin pixels (4 lobes of Lanczos-2 scaled to the bin factor).
fn build_lanczos2_kernel(bin: usize) -> Vec<f32> {
    let radius = 2 * bin; // 4 pixels in Lanczos-2 units, scaled by bin
    let len = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = (i as f32 - radius as f32) / bin as f32;
        let w = lanczos2(x);
        kernel.push(w);
        sum += w;
    }
    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply separable 1D convolution (horizontal then vertical) with the given kernel.
fn apply_separable_filter(data: &[f32], nx: usize, ny: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;

    // Horizontal pass
    let mut temp = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = x as isize + ki as isize - radius as isize;
                let sx = sx.clamp(0, nx as isize - 1) as usize;
                sum += data[y * nx + sx] * kv;
            }
            temp[y * nx + x] = sum;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = y as isize + ki as isize - radius as isize;
                let sy = sy.clamp(0, ny as isize - 1) as usize;
                sum += temp[sy * nx + x] * kv;
            }
            result[y * nx + x] = sum;
        }
    }

    result
}

/// Output range limits for a given MRC mode.
fn mode_range(mode: MrcMode) -> (f32, f32) {
    match mode {
        MrcMode::Byte => (0.0, 255.0),
        MrcMode::Short => (-32768.0, 32767.0),
        MrcMode::UShort => (0.0, 65535.0),
        _ => (f32::MIN, f32::MAX), // Float: no clamping needed
    }
}

/// Represents a section source: which file and which z-index within that file.
struct SectionSource {
    file_idx: usize,
    z: usize,
}

/// Build the list of all available sections across multiple input files.
/// Returns (sources, per-file nz, nx, ny from first file).
fn build_multi_file_section_list(
    inputs: &[String],
) -> (Vec<SectionSource>, usize, usize) {
    let mut sources = Vec::new();
    let mut first_nx = 0usize;
    let mut first_ny = 0usize;

    for (file_idx, path) in inputs.iter().enumerate() {
        let reader = MrcReader::open(path).unwrap_or_else(|e| {
            eprintln!("Error opening {}: {}", path, e);
            std::process::exit(1);
        });
        let h = reader.header();
        let nx = h.nx as usize;
        let ny = h.ny as usize;
        let nz = h.nz as usize;

        if file_idx == 0 {
            first_nx = nx;
            first_ny = ny;
        } else if nx != first_nx || ny != first_ny {
            eprintln!(
                "Warning: {} has dimensions {}x{}, expected {}x{}",
                path, nx, ny, first_nx, first_ny
            );
        }

        for z in 0..nz {
            sources.push(SectionSource { file_idx, z });
        }
    }
    (sources, first_nx, first_ny)
}

fn main() {
    let args = Args::parse();

    // Build section sources from all input files
    let (all_sources, in_nx, in_ny) = build_multi_file_section_list(&args.input);
    let total_sections = all_sources.len();

    // Open first input to get header info for output
    let first_reader = MrcReader::open(&args.input[0]).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", args.input[0], e);
        std::process::exit(1);
    });
    let in_header = first_reader.header().clone();
    drop(first_reader);

    // Determine sections to process
    let section_indices: Vec<usize> = match &args.secs {
        Some(s) => parse_section_list(s, total_sections),
        None => (0..total_sections).collect(),
    };
    let out_nz = section_indices.len();

    // Load transforms if provided
    let transforms: Option<Vec<LinearTransform>> = args.xform.as_ref().map(|path| {
        read_xf_file(path).unwrap_or_else(|e| {
            eprintln!("Error reading transform file {}: {}", path, e);
            std::process::exit(1);
        })
    });

    // Determine output dimensions
    let out_nx = in_nx / args.bin;
    let out_ny = in_ny / args.bin;

    // Determine effective float mode
    // --float_densities (legacy -F) is equivalent to --float 0
    let float_mode: Option<i32> = if args.float_mode >= 0 {
        Some(args.float_mode)
    } else if args.float_densities {
        Some(0)
    } else {
        None
    };

    // Determine output mode
    let out_mode = match args.mode {
        Some(m) => MrcMode::from_i32(m).unwrap_or(MrcMode::Float),
        None => {
            if float_mode.is_some() {
                MrcMode::Byte
            } else {
                in_header.data_mode().unwrap_or(MrcMode::Float)
            }
        }
    };

    // Determine antialias setting
    let use_antialias = args.antialias.unwrap_or(args.bin > 1);

    // Build antialias kernel if needed
    let aa_kernel = if use_antialias && args.bin > 1 {
        Some(build_lanczos2_kernel(args.bin))
    } else {
        None
    };

    // Parse scaling
    let scale_range: Option<(f32, f32)> = args.scale.as_ref().and_then(|s| {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 2 {
            Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
        } else {
            None
        }
    });

    // =========================================================================
    // For float modes 1 and 2, we need a first pass to compute global mean/SD
    // =========================================================================
    let (global_target_mean, global_target_sd) = if matches!(float_mode, Some(1) | Some(2)) {
        eprintln!("newstack: first pass - computing global statistics...");
        let mut all_sum = 0.0_f64;
        let mut all_sum_sq = 0.0_f64;
        let mut all_count = 0u64;

        // Open readers for each file as needed
        let mut open_readers: Vec<Option<MrcReader>> =
            (0..args.input.len()).map(|_| None).collect();

        for &sec_idx in &section_indices {
            let src = &all_sources[sec_idx];

            // Open reader for this file if not already open
            if open_readers[src.file_idx].is_none() {
                open_readers[src.file_idx] =
                    Some(MrcReader::open(&args.input[src.file_idx]).unwrap());
            }
            let reader = open_readers[src.file_idx].as_mut().unwrap();
            let data = reader.read_slice_f32(src.z).unwrap();

            for &v in &data {
                let vd = v as f64;
                all_sum += vd;
                all_sum_sq += vd * vd;
            }
            all_count += data.len() as u64;
        }

        let mean = (all_sum / all_count as f64) as f32;
        let variance = (all_sum_sq / all_count as f64) - (mean as f64 * mean as f64);
        let sd = (variance.max(0.0) as f32).sqrt();
        eprintln!("newstack: global mean={:.2}, SD={:.2}", mean, sd);
        (mean, sd)
    } else {
        (0.0f32, 1.0f32)
    };

    // Create output header
    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, out_mode);
    let bin_f = args.bin as f32;
    out_header.xlen = in_header.xlen / bin_f * (out_nx as f32 / (in_nx as f32 / bin_f));
    out_header.ylen = in_header.ylen / bin_f * (out_ny as f32 / (in_ny as f32 / bin_f));
    out_header.zlen = in_header.zlen * out_nz as f32 / in_header.nz as f32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;

    let label = if args.input.len() > 1 {
        format!(
            "newstack: {} sections from {} files",
            out_nz,
            args.input.len()
        )
    } else {
        format!("newstack: {} sections from {}", out_nz, args.input[0])
    };
    out_header.add_label(&label);

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating {}: {}", args.output, e);
        std::process::exit(1);
    });

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0_f64;
    let total_pix = (out_nx * out_ny * out_nz) as f64;

    let xcen = in_nx as f32 / 2.0;
    let ycen = in_ny as f32 / 2.0;

    // Open readers for second pass
    let mut open_readers: Vec<Option<MrcReader>> =
        (0..args.input.len()).map(|_| None).collect();

    for (out_z, &sec_idx) in section_indices.iter().enumerate() {
        let src = &all_sources[sec_idx];

        // Open reader for this file if not already open
        if open_readers[src.file_idx].is_none() {
            open_readers[src.file_idx] =
                Some(MrcReader::open(&args.input[src.file_idx]).unwrap_or_else(|e| {
                    eprintln!("Error opening {}: {}", args.input[src.file_idx], e);
                    std::process::exit(1);
                }));
        }
        let reader = open_readers[src.file_idx].as_mut().unwrap();
        let mut data = reader.read_slice_f32(src.z).unwrap();

        // Apply transform if provided
        if let Some(ref xforms) = transforms {
            let xf_idx = out_z.min(xforms.len() - 1);
            let xf = &xforms[xf_idx];
            let inv = xf.inverse();
            let src_slice = Slice::from_data(in_nx, in_ny, data);
            let mut dst = Slice::new(in_nx, in_ny, args.fill);
            for y in 0..in_ny {
                for x in 0..in_nx {
                    let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                    dst.set(x, y, src_slice.interpolate_bilinear(sx, sy, args.fill));
                }
            }
            data = dst.data;
        }

        // Antialias filtering before binning
        if let Some(ref kernel) = aa_kernel {
            data = apply_separable_filter(&data, in_nx, in_ny, kernel);
        }

        // Bin
        if args.bin > 1 {
            let src_slice = Slice::from_data(in_nx, in_ny, data);
            let binned = imod_slice::bin(&src_slice, args.bin);
            data = binned.data;
        }

        // Apply scaling / float modes
        if let Some((smin, smax)) = scale_range {
            let (dmin, dmax, _) = min_max_mean(&data);
            let range = dmax - dmin;
            if range > 1e-10 {
                let scale = (smax - smin) / range;
                for v in &mut data {
                    *v = (*v - dmin) * scale + smin;
                }
            }
        } else if let Some(fm) = float_mode {
            match fm {
                0 => {
                    // Scale each section to fill output range
                    let (dmin, dmax, _) = min_max_mean(&data);
                    let range = dmax - dmin;
                    let (out_lo, out_hi) = mode_range(out_mode);
                    if range > 1e-10 && out_lo.is_finite() && out_hi.is_finite() {
                        let scale = (out_hi - out_lo) / range;
                        for v in &mut data {
                            *v = (*v - dmin) * scale + out_lo;
                        }
                    } else if range > 1e-10 {
                        // Float output: just shift min to 0
                        for v in &mut data {
                            *v -= dmin;
                        }
                    }
                }
                1 => {
                    // Shift each section to the common mean
                    let (_, _, sec_mean) = min_max_mean(&data);
                    let shift = global_target_mean - sec_mean;
                    for v in &mut data {
                        *v += shift;
                    }
                }
                2 => {
                    // Scale each section to common mean and SD
                    let (_, _, sec_mean, sec_sd) = min_max_mean_sd(&data);
                    if sec_sd > 1e-10 {
                        let scale = global_target_sd / sec_sd;
                        for v in &mut data {
                            *v = (*v - sec_mean) * scale + global_target_mean;
                        }
                    } else {
                        // Zero variance section: just shift to mean
                        for v in &mut data {
                            *v = global_target_mean;
                        }
                    }
                }
                _ => {
                    eprintln!("Warning: unknown float mode {}, ignoring", fm);
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&data);
        if smin < global_min {
            global_min = smin;
        }
        if smax > global_max {
            global_max = smax;
        }
        global_sum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&data).unwrap();
    }

    let global_mean = (global_sum / total_pix) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap();

    eprintln!(
        "newstack: {} x {} x {} -> {} x {} x {} (mode {:?}){}",
        in_nx,
        in_ny,
        total_sections,
        out_nx,
        out_ny,
        out_nz,
        out_mode,
        if aa_kernel.is_some() {
            " [antialias]"
        } else {
            ""
        }
    );
    if let Some(fm) = float_mode {
        eprintln!("newstack: float mode {}", fm);
    }
}
