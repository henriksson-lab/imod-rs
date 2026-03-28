use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Trim or pad a volume to specified dimensions, or extract a subvolume.
#[derive(Parser)]
#[command(name = "trimvol", about = "Trim or extract a subvolume from an MRC file")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,

    /// Starting X coordinate (0-based)
    #[arg(short = 'x')]
    x0: Option<usize>,
    /// Starting Y coordinate
    #[arg(short = 'y')]
    y0: Option<usize>,
    /// Starting Z coordinate
    #[arg(short = 'z')]
    z0: Option<usize>,

    /// Output X size
    #[arg(long = "nx")]
    nx: Option<usize>,
    /// Output Y size
    #[arg(long = "ny")]
    ny: Option<usize>,
    /// Output Z size
    #[arg(long = "nz")]
    nz: Option<usize>,

    /// Swap Y and Z (rotate volume for XZ viewing)
    #[arg(short = 'r', long)]
    rotate: bool,

    /// Convert to bytes with automatic scaling
    #[arg(short = 'f', long)]
    float_to_byte: bool,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let x0 = args.x0.unwrap_or(0);
    let y0 = args.y0.unwrap_or(0);
    let z0 = args.z0.unwrap_or(0);
    let out_nx = args.nx.unwrap_or(in_nx - x0);
    let out_ny = args.ny.unwrap_or(in_ny - y0);
    let out_nz = args.nz.unwrap_or(in_nz - z0);

    let out_mode = if args.float_to_byte { MrcMode::Byte } else { MrcMode::Float };

    if args.rotate {
        // Swap Y and Z: output is (nx, nz, ny)
        let rot_nx = out_nx;
        let rot_ny = out_nz;
        let rot_nz = out_ny;

        let mut out_header = MrcHeader::new(rot_nx as i32, rot_ny as i32, rot_nz as i32, out_mode);
        out_header.add_label("trimvol: rotated Y/Z");
        let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

        // Read all needed slices
        let mut slices = Vec::with_capacity(out_nz);
        for iz in z0..z0 + out_nz {
            slices.push(reader.read_slice_f32(iz).unwrap());
        }

        // Find global stats for byte scaling
        let (global_min, global_max) = if args.float_to_byte {
            let mut mn = f32::MAX;
            let mut mx = f32::MIN;
            for s in &slices {
                for &v in s { if v < mn { mn = v; } if v > mx { mx = v; } }
            }
            (mn, mx)
        } else {
            (0.0, 1.0)
        };
        let byte_scale = if args.float_to_byte && (global_max - global_min).abs() > 1e-10 {
            255.0 / (global_max - global_min)
        } else {
            1.0
        };

        let mut gmin = f32::MAX;
        let mut gmax = f32::MIN;
        let mut gsum = 0.0_f64;

        // Output Z corresponds to input Y
        for oy_out_z in 0..rot_nz {
            let iy = y0 + oy_out_z;
            let mut out_data = vec![0.0f32; rot_nx * rot_ny];
            // Output Y corresponds to input Z
            for ox_out_y in 0..rot_ny {
                let iz_local = ox_out_y;
                for ox in 0..rot_nx {
                    let ix = x0 + ox;
                    let mut v = slices[iz_local][iy * in_nx + ix];
                    if args.float_to_byte {
                        v = ((v - global_min) * byte_scale).clamp(0.0, 255.0);
                    }
                    out_data[ox_out_y * rot_nx + ox] = v;
                }
            }

            let (smin, smax, smean) = min_max_mean(&out_data);
            if smin < gmin { gmin = smin; }
            if smax > gmax { gmax = smax; }
            gsum += smean as f64 * (rot_nx * rot_ny) as f64;

            writer.write_slice_f32(&out_data).unwrap();
        }

        let gmean = (gsum / (rot_nx * rot_ny * rot_nz) as f64) as f32;
        writer.finish(gmin, gmax, gmean).unwrap();
        eprintln!("trimvol: {} x {} x {} -> {} x {} x {} (rotated)", in_nx, in_ny, in_nz, rot_nx, rot_ny, rot_nz);
    } else {
        // Simple trim
        let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, out_mode);
        out_header.add_label("trimvol: subvolume extraction");
        let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

        let mut gmin = f32::MAX;
        let mut gmax = f32::MIN;
        let mut gsum = 0.0_f64;

        for iz in z0..z0 + out_nz {
            let data = reader.read_slice_f32(iz).unwrap();
            let mut out_data = Vec::with_capacity(out_nx * out_ny);
            for iy in y0..y0 + out_ny {
                let start = iy * in_nx + x0;
                out_data.extend_from_slice(&data[start..start + out_nx]);
            }

            if args.float_to_byte {
                let (dmin, dmax, _) = min_max_mean(&out_data);
                let range = dmax - dmin;
                if range > 1e-10 {
                    let s = 255.0 / range;
                    for v in &mut out_data { *v = (*v - dmin) * s; }
                }
            }

            let (smin, smax, smean) = min_max_mean(&out_data);
            if smin < gmin { gmin = smin; }
            if smax > gmax { gmax = smax; }
            gsum += smean as f64 * (out_nx * out_ny) as f64;

            writer.write_slice_f32(&out_data).unwrap();
        }

        let gmean = (gsum / (out_nx * out_ny * out_nz) as f64) as f32;
        writer.finish(gmin, gmax, gmean).unwrap();
        eprintln!("trimvol: {} x {} x {} -> {} x {} x {}", in_nx, in_ny, in_nz, out_nx, out_ny, out_nz);
    }
}
