use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Cut a subvolume and taper/pad its edges to the mean density,
/// suitable for FFT processing without edge artifacts.
#[derive(Parser)]
#[command(name = "taperoutvol", about = "Taper edges of a volume to mean value")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,

    /// X range: min,max (0-based, default: full)
    #[arg(long)]
    xminmax: Option<String>,
    /// Y range: min,max (0-based, default: full)
    #[arg(long)]
    yminmax: Option<String>,
    /// Z range: min,max (0-based, default: full)
    #[arg(long)]
    zminmax: Option<String>,

    /// Taper pad sizes in X, Y, Z (comma-separated, e.g. "16,16,8")
    #[arg(short, long)]
    taper: Option<String>,
}

fn parse_range(s: &str) -> (usize, usize) {
    let parts: Vec<&str> = s.split(',').collect();
    let lo: usize = parts[0].trim().parse().expect("invalid range");
    let hi: usize = parts[1].trim().parse().expect("invalid range");
    (lo, hi)
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error opening input: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    // Parse subvolume ranges (default: full volume)
    let (ix_lo, ix_hi) = args.xminmax.as_deref().map_or((0, in_nx - 1), parse_range);
    let (iy_lo, iy_hi) = args.yminmax.as_deref().map_or((0, in_ny - 1), parse_range);
    let (iz_lo, iz_hi) = args.zminmax.as_deref().map_or((0, in_nz - 1), parse_range);

    let nx_box = ix_hi + 1 - ix_lo;
    let ny_box = iy_hi + 1 - iy_lo;
    let nz_box = iz_hi + 1 - iz_lo;

    // Parse taper pad sizes (default: pad to next suitable FFT size)
    let (pad_x, pad_y, pad_z) = if let Some(ref t) = args.taper {
        let parts: Vec<usize> = t.split(',').map(|s| s.trim().parse().expect("invalid taper")).collect();
        match parts.len() {
            1 => (parts[0], parts[0], parts[0]),
            3 => (parts[0], parts[1], parts[2]),
            _ => {
                eprintln!("Error: taper must be 1 or 3 comma-separated values");
                std::process::exit(1);
            }
        }
    } else {
        // Default: 10% pad on each side, at least 8 pixels
        let px = (nx_box / 10).max(8);
        let py = (ny_box / 10).max(8);
        let pz = (nz_box / 10).max(8);
        (px, py, pz)
    };

    let nx_out = nx_box + 2 * pad_x;
    let ny_out = ny_box + 2 * pad_y;
    let nz_out = nz_box + 2 * pad_z;

    // Compute mean of the input volume for tapering
    let volume_mean = h.amean;

    // Set up output header
    let mut out_header = MrcHeader::new(nx_out as i32, ny_out as i32, nz_out as i32, MrcMode::Float);
    let px_x = if h.mx > 0 { h.xlen / h.mx as f32 } else { 1.0 };
    let px_y = if h.my > 0 { h.ylen / h.my as f32 } else { 1.0 };
    let px_z = if h.mz > 0 { h.zlen / h.mz as f32 } else { 1.0 };
    out_header.xlen = nx_out as f32 * px_x;
    out_header.ylen = ny_out as f32 * px_y;
    out_header.zlen = nz_out as f32 * px_z;
    out_header.mx = nx_out as i32;
    out_header.my = ny_out as i32;
    out_header.mz = nz_out as i32;
    // Adjust origin: shift by the extraction offset minus pad
    out_header.xorg = h.xorg + (ix_lo as f32 - pad_x as f32) * px_x;
    out_header.yorg = h.yorg + (iy_lo as f32 - pad_y as f32) * px_y;
    out_header.zorg = h.zorg + (iz_lo as f32 - pad_z as f32) * px_z;
    out_header.add_label("taperoutvol: Taper outside of excised volume");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating output: {}", e);
        std::process::exit(1);
    });

    let iz_start = iz_lo as isize - pad_z as isize;
    let iz_end = iz_start + nz_out as isize;
    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for iz in iz_start..iz_end {
        // Clamp Z to valid input range
        let iz_read = iz.clamp(iz_lo as isize, iz_hi as isize) as usize;
        let in_slice = reader.read_slice_f32(iz_read).unwrap();

        // Extract the subregion from the input slice
        let mut box_data = vec![0.0f32; nx_box * ny_box];
        for iy in 0..ny_box {
            let src_y = iy_lo + iy;
            for ix in 0..nx_box {
                let src_x = ix_lo + ix;
                box_data[iy * nx_box + ix] = in_slice[src_y * in_nx + src_x];
            }
        }

        // Create output slice padded with mean, then taper
        let mut out_slice = vec![volume_mean; nx_out * ny_out];

        // Place the box data in the center
        for iy in 0..ny_box {
            for ix in 0..nx_box {
                out_slice[(iy + pad_y) * nx_out + (ix + pad_x)] = box_data[iy * nx_box + ix];
            }
        }

        // Taper X edges
        for iy in 0..ny_box {
            let oy = iy + pad_y;
            for ix in 0..pad_x {
                let frac = (ix as f32 + 0.5) / pad_x as f32;
                // Left taper
                let val = box_data[iy * nx_box];
                out_slice[oy * nx_out + ix] = volume_mean + frac * (val - volume_mean);
                // Right taper
                let val = box_data[iy * nx_box + nx_box - 1];
                let ox = pad_x + nx_box + pad_x - 1 - ix;
                out_slice[oy * nx_out + ox] = volume_mean + frac * (val - volume_mean);
            }
        }

        // Taper Y edges
        for iy in 0..pad_y {
            let frac = (iy as f32 + 0.5) / pad_y as f32;
            for ix in 0..nx_out {
                // Top taper
                let ref_val = out_slice[pad_y * nx_out + ix];
                out_slice[iy * nx_out + ix] = volume_mean + frac * (ref_val - volume_mean);
                // Bottom taper
                let bot_ref = out_slice[(pad_y + ny_box - 1) * nx_out + ix];
                let oy = pad_y + ny_box + pad_y - 1 - iy;
                out_slice[oy * nx_out + ix] = volume_mean + frac * (bot_ref - volume_mean);
            }
        }

        // Taper Z edges: attenuate toward mean if outside the box range
        if iz < iz_lo as isize || iz > iz_hi as isize {
            let atten = if iz < iz_lo as isize {
                (iz - iz_start) as f32 / (iz_lo as isize - iz_start) as f32
            } else {
                (iz_end - 1 - iz) as f32 / (iz_end - 1 - iz_hi as isize) as f32
            };
            let base = (1.0 - atten) * volume_mean;
            for v in out_slice.iter_mut() {
                *v = base + atten * *v;
            }
        }

        let (smin, smax, smean) = min_max_mean(&out_slice);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * (nx_out * ny_out) as f64;

        writer.write_slice_f32(&out_slice).unwrap();
    }

    let gmean = (gsum / (nx_out * ny_out * nz_out) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}
