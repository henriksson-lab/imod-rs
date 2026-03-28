use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Reduce an MRC image stack by binning (averaging) pixels.
///
/// Reduces each section by averaging NxN blocks of pixels in X and Y,
/// and optionally averaging groups of sections in Z. This produces a
/// smaller stack suitable for quick viewing or initial alignment.
#[derive(Parser)]
#[command(name = "reducestack", about = "Reduce a stack by binning/averaging")]
struct Args {
    /// Input MRC file
    input: String,

    /// Output MRC file
    output: String,

    /// Binning factor in X and Y
    #[arg(short = 'b', long = "bin", default_value_t = 2)]
    binxy: usize,

    /// Binning factor in Z (default: 1, i.e. keep all sections)
    #[arg(short = 'z', long = "binz", default_value_t = 1)]
    binz: usize,

    /// Starting section (0-based, default: 0)
    #[arg(short = 's', long)]
    start: Option<usize>,

    /// Ending section (0-based, inclusive, default: last)
    #[arg(short = 'e', long)]
    end: Option<usize>,

    /// Antialias: apply a simple low-pass filter before binning
    #[arg(long)]
    antialias: bool,
}

/// Apply a simple 3x3 box-blur as a crude anti-alias pre-filter.
fn box_blur(data: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for dy in 0..3i32 {
                let jy = iy as i32 + dy - 1;
                if jy < 0 || jy >= ny as i32 {
                    continue;
                }
                for dx in 0..3i32 {
                    let jx = ix as i32 + dx - 1;
                    if jx < 0 || jx >= nx as i32 {
                        continue;
                    }
                    sum += data[jy as usize * nx + jx as usize];
                    count += 1;
                }
            }
            out[iy * nx + ix] = sum / count as f32;
        }
    }
    out
}

/// Bin a single 2D slice by averaging binxy x binxy blocks.
fn bin_slice(data: &[f32], in_nx: usize, in_ny: usize, binxy: usize) -> Vec<f32> {
    let out_nx = in_nx / binxy;
    let out_ny = in_ny / binxy;
    let inv = 1.0 / (binxy * binxy) as f32;
    let mut out = vec![0.0f32; out_nx * out_ny];

    for oy in 0..out_ny {
        for ox in 0..out_nx {
            let mut sum = 0.0f32;
            for dy in 0..binxy {
                let iy = oy * binxy + dy;
                let row_off = iy * in_nx;
                for dx in 0..binxy {
                    let ix = ox * binxy + dx;
                    sum += data[row_off + ix];
                }
            }
            out[oy * out_nx + ox] = sum * inv;
        }
    }
    out
}

fn main() {
    let args = Args::parse();

    if args.binxy < 1 {
        eprintln!("ERROR: REDUCESTACK - binning factor must be >= 1");
        std::process::exit(1);
    }
    if args.binz < 1 {
        eprintln!("ERROR: REDUCESTACK - Z binning factor must be >= 1");
        std::process::exit(1);
    }

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: REDUCESTACK - opening input: {e}");
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let z_start = args.start.unwrap_or(0);
    let z_end = args.end.unwrap_or(in_nz - 1).min(in_nz - 1);
    if z_start > z_end {
        eprintln!("ERROR: REDUCESTACK - start section > end section");
        std::process::exit(1);
    }

    let num_sections = z_end - z_start + 1;
    let out_nx = in_nx / args.binxy;
    let out_ny = in_ny / args.binxy;
    let out_nz = num_sections / args.binz;

    if out_nx == 0 || out_ny == 0 || out_nz == 0 {
        eprintln!(
            "ERROR: REDUCESTACK - output dimensions {}x{}x{} are zero; binning factor too large",
            out_nx, out_ny, out_nz
        );
        std::process::exit(1);
    }

    eprintln!(
        "Reducing {}x{}x{} -> {}x{}x{} (bin {}x{}x{})",
        in_nx, in_ny, num_sections, out_nx, out_ny, out_nz, args.binxy, args.binxy, args.binz
    );

    // Set up output
    let mut out_header =
        MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    // Preserve pixel spacing scaled by binning
    if h.mx > 0 {
        let pixel_x = h.xlen / h.mx as f32;
        let pixel_y = h.ylen / h.my as f32;
        let pixel_z = if h.mz > 0 {
            h.zlen / h.mz as f32
        } else {
            1.0
        };
        out_header.xlen = out_nx as f32 * pixel_x * args.binxy as f32;
        out_header.ylen = out_ny as f32 * pixel_y * args.binxy as f32;
        out_header.zlen = out_nz as f32 * pixel_z * args.binz as f32;
    }
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;
    out_header.add_label(&format!(
        "reducestack: bin {}x{}x{}",
        args.binxy, args.binxy, args.binz
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    let inv_binz = 1.0 / args.binz as f32;

    for oz in 0..out_nz {
        // Accumulate binz sections
        let mut accum = vec![0.0f32; out_nx * out_ny];

        for dz in 0..args.binz {
            let iz = z_start + oz * args.binz + dz;
            let slice = reader.read_slice_f32(iz).unwrap();

            let processed = if args.antialias {
                let blurred = box_blur(&slice, in_nx, in_ny);
                bin_slice(&blurred, in_nx, in_ny, args.binxy)
            } else {
                bin_slice(&slice, in_nx, in_ny, args.binxy)
            };

            for (a, &p) in accum.iter_mut().zip(processed.iter()) {
                *a += p;
            }
        }

        // Average over Z
        if args.binz > 1 {
            for v in accum.iter_mut() {
                *v *= inv_binz;
            }
        }

        let (smin, smax, smean) = min_max_mean(&accum);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * accum.len() as f64;

        writer.write_slice_f32(&accum).unwrap();
    }

    let total = (out_nx * out_ny * out_nz) as f64;
    let gmean = (gsum / total) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();

    eprintln!(
        "Done. Min={:.4} Max={:.4} Mean={:.4}",
        gmin, gmax, gmean
    );
}
