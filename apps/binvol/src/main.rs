use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Bin (reduce) a volume by averaging NxNxN blocks of voxels.
#[derive(Parser)]
#[command(name = "binvol", about = "Bin a volume by averaging voxels")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,
    /// Binning factor in X and Y
    #[arg(short = 'b', long, default_value_t = 2)]
    binxy: usize,
    /// Binning factor in Z (default: same as XY)
    #[arg(short = 'z', long)]
    binz: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let binz = args.binz.unwrap_or(args.binxy);

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let out_nx = in_nx / args.binxy;
    let out_ny = in_ny / args.binxy;
    let out_nz = in_nz / binz;

    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    out_header.xlen = h.xlen * out_nx as f32 / in_nx as f32;
    out_header.ylen = h.ylen * out_ny as f32 / in_ny as f32;
    out_header.zlen = h.zlen * out_nz as f32 / in_nz as f32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;
    out_header.add_label(&format!("binvol: bin {}x{}x{}", args.binxy, args.binxy, binz));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    // Read all input slices for Z binning
    let mut all_slices: Vec<Vec<f32>> = Vec::with_capacity(in_nz);
    for z in 0..in_nz {
        all_slices.push(reader.read_slice_f32(z).unwrap());
    }

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;
    let total = (out_nx * out_ny * out_nz) as f64;
    let inv_vol = 1.0 / (args.binxy * args.binxy * binz) as f32;

    for oz in 0..out_nz {
        let mut out_data = vec![0.0f32; out_nx * out_ny];
        let z_start = oz * binz;

        for dz in 0..binz {
            let iz = z_start + dz;
            if iz >= in_nz { break; }
            let slice = &all_slices[iz];
            for oy in 0..out_ny {
                for ox in 0..out_nx {
                    let mut sum = 0.0f32;
                    for dy in 0..args.binxy {
                        let iy = oy * args.binxy + dy;
                        for dx in 0..args.binxy {
                            let ix = ox * args.binxy + dx;
                            sum += slice[iy * in_nx + ix];
                        }
                    }
                    out_data[oy * out_nx + ox] += sum;
                }
            }
        }

        for v in &mut out_data {
            *v *= inv_vol;
        }

        let (smin, smax, smean) = min_max_mean(&out_data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&out_data).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / total) as f32).unwrap();
    eprintln!("binvol: {} x {} x {} -> {} x {} x {}", in_nx, in_ny, in_nz, out_nx, out_ny, out_nz);
}
