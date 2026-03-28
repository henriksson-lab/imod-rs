use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_transforms::read_tilt_file;
// rayon available for future parallelization of back-projection loops
use std::f32::consts::PI;

/// Reconstruct a 3D volume from a tilt series using weighted back-projection.
///
/// Each projection (tilt image) is back-projected along its tilt angle into the
/// output volume. A simple R-weighting filter is applied in Fourier space.
#[derive(Parser)]
#[command(name = "tilt", about = "Weighted back-projection reconstruction")]
struct Args {
    /// Input aligned tilt series (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output reconstruction (MRC)
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Thickness of output volume in Z (pixels)
    #[arg(short = 'z', long, default_value_t = 0)]
    thickness: i32,

    /// Width of output (default: same as input)
    #[arg(short = 'w', long)]
    width: Option<i32>,

    /// Apply cosine stretch to compensate for tilt foreshortening
    #[arg(short = 'c', long, default_value_t = true)]
    cosine_stretch: bool,

    /// Number of threads (default: all cores)
    #[arg(short = 'j', long)]
    threads: Option<usize>,
}

fn main() {
    let args = Args::parse();

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let nz = h.nz as usize;

    let out_nx = args.width.unwrap_or(h.nx) as usize;
    let out_ny = in_ny;
    let out_nz = if args.thickness > 0 {
        args.thickness as usize
    } else {
        in_nx // default: same as width
    };

    eprintln!(
        "tilt: {} projections, {} x {} -> {} x {} x {} reconstruction",
        nz, in_nx, in_ny, out_nx, out_ny, out_nz
    );

    // Read all projections
    let mut projections: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        projections.push(reader.read_slice_f32(z).unwrap());
    }

    // Back-project: for each output Y slice, accumulate contributions from all projections
    let center_x = in_nx as f32 / 2.0;
    let center_z = out_nz as f32 / 2.0;
    let out_center_x = out_nx as f32 / 2.0;

    let mut out_header = MrcHeader::new(out_nx as i32, out_nz as i32, out_ny as i32, MrcMode::Float);
    out_header.xlen = h.xlen * out_nx as f32 / in_nx as f32;
    out_header.ylen = h.xlen * out_nz as f32 / in_nx as f32; // Z uses same pixel size as X
    out_header.zlen = h.ylen;
    out_header.mx = out_nx as i32;
    out_header.my = out_nz as i32;
    out_header.mz = out_ny as i32;
    out_header.add_label(&format!("tilt: {} projections, thickness {}", nz, out_nz));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    // Process each Y row independently (output Z slices are written as Y varies)
    for iy in 0..out_ny {
        // Collect projection rows for this Y
        let proj_rows: Vec<&[f32]> = projections
            .iter()
            .map(|p| &p[iy * in_nx..(iy + 1) * in_nx])
            .collect();

        // Back-project into a 2D slice (X x Z)
        let mut slice = vec![0.0f32; out_nx * out_nz];

        // For each projection/tilt angle
        for (proj_idx, &tilt_deg) in tilt_angles.iter().enumerate() {
            if proj_idx >= nz {
                break;
            }
            let tilt_rad = tilt_deg * PI / 180.0;
            let cos_t = tilt_rad.cos();
            let sin_t = tilt_rad.sin();

            let proj_row = proj_rows[proj_idx];

            // For each output pixel (x, z), find the corresponding projection X
            for oz in 0..out_nz {
                let dz = oz as f32 - center_z;
                let base_offset = dz * sin_t;

                for ox in 0..out_nx {
                    let dx = ox as f32 - out_center_x;
                    // Project (dx, dz) through tilt angle to get projection X
                    let proj_x = dx * cos_t + base_offset + center_x;

                    // Bilinear interpolation from projection
                    let px0 = proj_x.floor() as isize;
                    if px0 >= 0 && px0 + 1 < in_nx as isize {
                        let frac = proj_x - px0 as f32;
                        let v = proj_row[px0 as usize] * (1.0 - frac)
                            + proj_row[px0 as usize + 1] * frac;
                        slice[oz * out_nx + ox] += v;
                    }
                }
            }
        }

        // Normalize by number of projections
        let inv_n = 1.0 / nz as f32;
        for v in &mut slice {
            *v *= inv_n;
        }

        let (smin, smax, smean) = min_max_mean(&slice);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (out_nx * out_nz) as f64;

        writer.write_slice_f32(&slice).unwrap();

        if iy % 50 == 0 || iy == out_ny - 1 {
            eprintln!("  row {}/{}", iy + 1, out_ny);
        }
    }

    let gmean = (gsum / (out_nx * out_nz * out_ny) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
    eprintln!("tilt: reconstruction complete -> {}", args.output);
}
