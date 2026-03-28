mod gpu;

use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_transforms::read_tilt_file;
use rayon::prelude::*;
use std::f32::consts::PI;

/// Reconstruct a 3D volume from a tilt series using weighted back-projection.
///
/// Each projection (tilt image) is back-projected along its tilt angle into the
/// output volume. Uses rayon for parallel reconstruction across Y rows.
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

    /// Number of threads (default: all cores)
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Use GPU-accelerated back-projection (via wgpu compute shaders)
    #[arg(long)]
    gpu: bool,
}

/// Precomputed per-projection constants to avoid recomputing in the inner loop.
struct ProjParams {
    cos_t: f32,
    sin_t: f32,
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
        in_nx
    };

    let n_threads = rayon::current_num_threads();
    eprintln!(
        "tilt: {} projections, {} x {} -> {} x {} x {} ({} threads)",
        nz, in_nx, in_ny, out_nx, out_ny, out_nz, n_threads
    );

    // Read all projections into a flat buffer for cache-friendly access
    let mut proj_data: Vec<f32> = Vec::with_capacity(nz * in_nx * in_ny);
    for z in 0..nz {
        proj_data.extend_from_slice(&reader.read_slice_f32(z).unwrap());
    }

    // Precompute trig for each projection
    let proj_params: Vec<ProjParams> = tilt_angles
        .iter()
        .take(nz)
        .map(|&deg| {
            let rad = deg * PI / 180.0;
            ProjParams {
                cos_t: rad.cos(),
                sin_t: rad.sin(),
            }
        })
        .collect();

    let center_x = in_nx as f32 / 2.0;
    let center_z = out_nz as f32 / 2.0;
    let out_center_x = out_nx as f32 / 2.0;
    let inv_n = 1.0 / nz as f32;

    // Parallel reconstruction: each Y row is independent
    // Process in chunks for progress reporting
    let chunk_size = 32.max(out_ny / 20);
    let mut out_header = MrcHeader::new(out_nx as i32, out_nz as i32, out_ny as i32, MrcMode::Float);
    out_header.xlen = h.xlen * out_nx as f32 / in_nx as f32;
    out_header.ylen = h.xlen * out_nz as f32 / in_nx as f32;
    out_header.zlen = h.ylen;
    out_header.mx = out_nx as i32;
    out_header.my = out_nz as i32;
    out_header.mz = out_ny as i32;
    out_header.add_label(&format!("tilt: {} projections, thickness {}", nz, out_nz));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;
    let mut rows_done = 0usize;

    // Try GPU path if requested
    let tilt_cos_sin: Vec<(f32, f32)> = proj_params
        .iter()
        .map(|pp| (pp.cos_t, pp.sin_t))
        .collect();

    let gpu_session = if args.gpu {
        match gpu::GpuSession::new(
            &proj_data,
            &tilt_cos_sin,
            in_nx,
            in_ny,
            nz,
            out_nx,
            out_nz,
        ) {
            Some(session) => {
                eprintln!("tilt: using GPU back-projection");
                Some(session)
            }
            None => {
                eprintln!("tilt: WARNING: GPU init failed, falling back to CPU");
                None
            }
        }
    } else {
        None
    };

    if let Some(ref session) = gpu_session {
        // GPU path: dispatch one row at a time
        for iy in 0..out_ny {
            let slice = session.reconstruct_row(iy);

            let (smin, smax, smean) = min_max_mean(&slice);
            if smin < gmin { gmin = smin; }
            if smax > gmax { gmax = smax; }
            gsum += smean as f64 * (out_nx * out_nz) as f64;
            writer.write_slice_f32(&slice).unwrap();

            rows_done += 1;
            if rows_done % chunk_size == 0 || rows_done == out_ny {
                eprintln!("  {}/{} rows (GPU)", rows_done, out_ny);
            }
        }
    } else {
        // CPU path: parallel reconstruction with rayon
        for chunk_start in (0..out_ny).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(out_ny);
            let chunk_rows: Vec<usize> = (chunk_start..chunk_end).collect();

            let slices: Vec<Vec<f32>> = chunk_rows
                .par_iter()
                .map(|&iy| {
                    let mut slice = vec![0.0f32; out_nx * out_nz];

                    for (pi, pp) in proj_params.iter().enumerate() {
                        let row_offset = pi * in_nx * in_ny + iy * in_nx;
                        let proj_row = &proj_data[row_offset..row_offset + in_nx];

                        for oz in 0..out_nz {
                            let dz = oz as f32 - center_z;
                            let base_offset = dz * pp.sin_t;
                            let slice_row = oz * out_nx;

                            for ox in 0..out_nx {
                                let dx = ox as f32 - out_center_x;
                                let proj_x = dx * pp.cos_t + base_offset + center_x;

                                let px0 = proj_x.floor() as isize;
                                if px0 >= 0 && px0 + 1 < in_nx as isize {
                                    let frac = proj_x - px0 as f32;
                                    let px0u = px0 as usize;
                                    let v = proj_row[px0u] * (1.0 - frac)
                                        + proj_row[px0u + 1] * frac;
                                    slice[slice_row + ox] += v;
                                }
                            }
                        }
                    }

                    for v in &mut slice {
                        *v *= inv_n;
                    }
                    slice
                })
                .collect();

            for slice in &slices {
                let (smin, smax, smean) = min_max_mean(slice);
                if smin < gmin { gmin = smin; }
                if smax > gmax { gmax = smax; }
                gsum += smean as f64 * (out_nx * out_nz) as f64;
                writer.write_slice_f32(slice).unwrap();
            }

            rows_done += chunk_end - chunk_start;
            eprintln!("  {}/{} rows", rows_done, out_ny);
        }
    }

    let gmean = (gsum / (out_nx * out_nz * out_ny) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
    eprintln!("tilt: reconstruction complete -> {}", args.output);
}
