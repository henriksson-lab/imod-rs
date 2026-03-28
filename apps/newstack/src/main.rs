use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{read_xf_file, LinearTransform};

/// Create a new image stack from sections of one or more input files.
///
/// Supports reordering sections, applying 2D transforms, mode conversion,
/// binning, and scaling.
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

fn main() {
    let args = Args::parse();

    // Open first input to get dimensions
    let mut reader = MrcReader::open(&args.input[0]).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", args.input[0], e);
        std::process::exit(1);
    });
    let in_header = reader.header().clone();
    let in_nx = in_header.nx as usize;
    let in_ny = in_header.ny as usize;
    let in_nz = in_header.nz as usize;

    // Determine sections to process
    let sections = match &args.secs {
        Some(s) => parse_section_list(s, in_nz),
        None => (0..in_nz).collect(),
    };
    let out_nz = sections.len();

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

    // Determine output mode
    let out_mode = match args.mode {
        Some(m) => MrcMode::from_i32(m).unwrap_or(MrcMode::Float),
        None => {
            if args.float_densities {
                MrcMode::Byte
            } else {
                in_header.data_mode().unwrap_or(MrcMode::Float)
            }
        }
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

    // Create output header
    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, out_mode);
    let bin_f = args.bin as f32;
    out_header.xlen = in_header.xlen / bin_f * (out_nx as f32 / (in_nx as f32 / bin_f));
    out_header.ylen = in_header.ylen / bin_f * (out_ny as f32 / (in_ny as f32 / bin_f));
    out_header.zlen = in_header.zlen * out_nz as f32 / in_nz as f32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;

    let label = format!("newstack: {} sections from {}", out_nz, args.input[0]);
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

    for (out_z, &in_z) in sections.iter().enumerate() {
        let mut data = reader.read_slice_f32(in_z).unwrap();

        // Apply transform if provided
        if let Some(ref xforms) = transforms {
            let xf_idx = out_z.min(xforms.len() - 1);
            let xf = &xforms[xf_idx];
            let inv = xf.inverse();
            let src = Slice::from_data(in_nx, in_ny, data);
            let mut dst = Slice::new(in_nx, in_ny, args.fill);
            for y in 0..in_ny {
                for x in 0..in_nx {
                    let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                    dst.set(x, y, src.interpolate_bilinear(sx, sy, args.fill));
                }
            }
            data = dst.data;
        }

        // Bin
        if args.bin > 1 {
            let src = Slice::from_data(in_nx, in_ny, data);
            let binned = imod_slice::bin(&src, args.bin);
            data = binned.data;
        }

        // Scale
        if let Some((smin, smax)) = scale_range {
            let (dmin, dmax, _) = min_max_mean(&data);
            let range = dmax - dmin;
            if range > 1e-10 {
                let scale = (smax - smin) / range;
                for v in &mut data {
                    *v = (*v - dmin) * scale + smin;
                }
            }
        } else if args.float_densities {
            let (dmin, dmax, _) = min_max_mean(&data);
            let range = dmax - dmin;
            if range > 1e-10 {
                let scale = 255.0 / range;
                for v in &mut data {
                    *v = (*v - dmin) * scale;
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&data);
        if smin < global_min { global_min = smin; }
        if smax > global_max { global_max = smax; }
        global_sum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&data).unwrap();
    }

    let global_mean = (global_sum / total_pix) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap();

    eprintln!(
        "newstack: {} x {} x {} -> {} x {} x {} (mode {:?})",
        in_nx, in_ny, in_nz, out_nx, out_ny, out_nz, out_mode
    );
}
