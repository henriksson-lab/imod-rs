use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Squeeze (reduce) or expand a volume by interpolation.
///
/// Uses trilinear interpolation to resize a volume by the given factor(s).
/// A factor > 1 squeezes (reduces), < 1 expands.
#[derive(Parser)]
#[command(name = "squeezevol", about = "Squeeze or expand a volume by a factor")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,

    /// Reduction factor (default 1.6). Values > 1 reduce, < 1 expand.
    #[arg(short = 'f', long, default_value_t = 1.6)]
    factor: f64,

    /// Expansion factor (reciprocal of squeeze factor, mutually exclusive with -f)
    #[arg(short = 'e', long)]
    expand: Option<f64>,

    /// X squeeze factor (overrides -f for X)
    #[arg(short = 'x', long = "xfactor")]
    x_factor: Option<f64>,
    /// Y squeeze factor (overrides -f for Y)
    #[arg(short = 'y', long = "yfactor")]
    y_factor: Option<f64>,
    /// Z squeeze factor (overrides -f for Z)
    #[arg(short = 'z', long = "zfactor")]
    z_factor: Option<f64>,

    /// Use linear interpolation (default: trilinear)
    #[arg(short = 'l', long)]
    linear: bool,
}

fn main() {
    let args = Args::parse();

    if args.expand.is_some() && args.factor != 1.6 {
        eprintln!("Error: cannot specify both --factor and --expand");
        std::process::exit(1);
    }

    let base_factor = if let Some(e) = args.expand {
        if e == 0.0 {
            eprintln!("Error: expand factor cannot be zero");
            std::process::exit(1);
        }
        1.0 / e
    } else {
        args.factor
    };

    let fx = args.x_factor.unwrap_or(base_factor);
    let fy = args.y_factor.unwrap_or(base_factor);
    let fz = args.z_factor.unwrap_or(base_factor);

    if fx <= 0.0 || fy <= 0.0 || fz <= 0.0 {
        eprintln!("Error: all factors must be positive");
        std::process::exit(1);
    }

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error opening input: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let out_nx = (in_nx as f64 / fx) as usize;
    let out_ny = (in_ny as f64 / fy) as usize;
    let out_nz = (in_nz as f64 / fz) as usize;

    if out_nx == 0 || out_ny == 0 || out_nz == 0 {
        eprintln!("Error: output dimensions would be zero ({} x {} x {})", out_nx, out_ny, out_nz);
        std::process::exit(1);
    }

    eprintln!(
        "Squeezing {} x {} x {} -> {} x {} x {} (factors {:.3} x {:.3} x {:.3})",
        in_nx, in_ny, in_nz, out_nx, out_ny, out_nz, fx, fy, fz
    );

    // Read all input slices
    let mut in_vol: Vec<Vec<f32>> = Vec::with_capacity(in_nz);
    for z in 0..in_nz {
        in_vol.push(reader.read_slice_f32(z).unwrap());
    }

    // Set up output header
    let px_x = if h.mx > 0 { h.xlen / h.mx as f32 } else { 1.0 };
    let px_y = if h.my > 0 { h.ylen / h.my as f32 } else { 1.0 };
    let px_z = if h.mz > 0 { h.zlen / h.mz as f32 } else { 1.0 };

    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    out_header.xlen = out_nx as f32 * px_x * fx as f32;
    out_header.ylen = out_ny as f32 * px_y * fy as f32;
    out_header.zlen = out_nz as f32 * px_z * fz as f32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;
    // Adjust origin so that the center of the output volume matches the input center
    out_header.xorg = h.xorg + 0.5 * (out_nx as f64 * fx - in_nx as f64) as f32 * px_x;
    out_header.yorg = h.yorg + 0.5 * (out_ny as f64 * fy - in_ny as f64) as f32 * px_y;
    out_header.zorg = h.zorg + 0.5 * (out_nz as f64 * fz - in_nz as f64) as f32 * px_z;
    out_header.add_label(&format!(
        "squeezevol: factor {:.3} x {:.3} x {:.3}",
        fx, fy, fz
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating output: {}", e);
        std::process::exit(1);
    });

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for oz in 0..out_nz {
        let mut out_slice = vec![0.0f32; out_nx * out_ny];
        // Map output Z to input Z
        let src_z = oz as f64 * fz + (fz - 1.0) * 0.5;

        let iz0 = (src_z.floor() as isize).clamp(0, in_nz as isize - 1) as usize;
        let iz1 = (iz0 + 1).min(in_nz - 1);
        let dz = (src_z - iz0 as f64) as f32;
        let dz = dz.clamp(0.0, 1.0);

        for oy in 0..out_ny {
            let src_y = oy as f64 * fy + (fy - 1.0) * 0.5;
            let iy0 = (src_y.floor() as isize).clamp(0, in_ny as isize - 1) as usize;
            let iy1 = (iy0 + 1).min(in_ny - 1);
            let dy = (src_y - iy0 as f64) as f32;
            let dy = dy.clamp(0.0, 1.0);

            for ox in 0..out_nx {
                let src_x = ox as f64 * fx + (fx - 1.0) * 0.5;
                let ix0 = (src_x.floor() as isize).clamp(0, in_nx as isize - 1) as usize;
                let ix1 = (ix0 + 1).min(in_nx - 1);
                let dx = (src_x - ix0 as f64) as f32;
                let dx = dx.clamp(0.0, 1.0);

                // Trilinear interpolation
                let v000 = in_vol[iz0][iy0 * in_nx + ix0];
                let v100 = in_vol[iz0][iy0 * in_nx + ix1];
                let v010 = in_vol[iz0][iy1 * in_nx + ix0];
                let v110 = in_vol[iz0][iy1 * in_nx + ix1];
                let v001 = in_vol[iz1][iy0 * in_nx + ix0];
                let v101 = in_vol[iz1][iy0 * in_nx + ix1];
                let v011 = in_vol[iz1][iy1 * in_nx + ix0];
                let v111 = in_vol[iz1][iy1 * in_nx + ix1];

                let c00 = v000 * (1.0 - dx) + v100 * dx;
                let c10 = v010 * (1.0 - dx) + v110 * dx;
                let c01 = v001 * (1.0 - dx) + v101 * dx;
                let c11 = v011 * (1.0 - dx) + v111 * dx;

                let c0 = c00 * (1.0 - dy) + c10 * dy;
                let c1 = c01 * (1.0 - dy) + c11 * dy;

                out_slice[oy * out_nx + ox] = c0 * (1.0 - dz) + c1 * dz;
            }
        }

        let (smin, smax, smean) = min_max_mean(&out_slice);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&out_slice).unwrap();
    }

    let gmean = (gsum / (out_nx * out_ny * out_nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}
