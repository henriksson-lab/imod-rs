use clap::Parser;
use imod_core::MrcMode;
use imod_math::{mean_sd, min_max_mean};
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;

/// Erase X-ray and hot pixel artifacts from CCD camera images.
///
/// Detects pixels that deviate significantly from their neighbors and
/// replaces them with the local mean.
#[derive(Parser)]
#[command(name = "ccderaser", about = "Erase CCD artifacts (X-rays, hot pixels)")]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Peak criterion: number of SDs above local mean to flag a pixel
    #[arg(short = 'p', long, default_value_t = 8.0)]
    peak_criterion: f32,

    /// Difference criterion: pixels whose difference from neighbor mean
    /// exceeds this many SDs of the image are replaced
    #[arg(short = 'd', long, default_value_t = 6.0)]
    diff_criterion: f32,

    /// Maximum radius of connected artifact to erase (pixels)
    #[arg(short = 'r', long, default_value_t = 4)]
    max_radius: usize,

    /// Size of border to scan for extra peaks
    #[arg(short = 'b', long, default_value_t = 2)]
    border: usize,

    /// Find and erase all artifacts automatically
    #[arg(short = 'f', long, default_value_t = true)]
    find: bool,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.mx = h.mx;
    out_header.my = h.my;
    out_header.mz = h.mz;
    out_header.add_label(&format!(
        "ccderaser: peak={:.1} diff={:.1} maxrad={}",
        args.peak_criterion, args.diff_criterion, args.max_radius
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();
    let mut total_replaced = 0usize;

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let mut slice = Slice::from_data(nx, ny, data);

        // Compute image statistics
        let (_img_mean, img_sd) = mean_sd(&slice.data);
        let diff_thresh = args.diff_criterion * img_sd;

        let mut replaced = 0;

        // Scan for hot pixels / X-ray artifacts
        let border = args.border;
        for y in border..ny - border {
            for x in border..nx - border {
                let val = slice.get(x, y);

                // Compare with local 3x3 neighborhood (excluding center)
                let mut local_sum = 0.0f32;
                let mut local_count = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        local_sum += slice.get(
                            (x as i32 + dx) as usize,
                            (y as i32 + dy) as usize,
                        );
                        local_count += 1;
                    }
                }
                let local_mean = local_sum / local_count as f32;
                let diff = (val - local_mean).abs();

                if diff > diff_thresh {
                    // Replace with local mean, then grow to find connected artifact
                    replace_artifact(&mut slice, x, y, args.max_radius, diff_thresh);
                    replaced += 1;
                }
            }
        }

        total_replaced += replaced;
        if replaced > 0 {
            eprintln!("  section {}: replaced {} pixels", z, replaced);
        }

        let (smin, smax, smean) = min_max_mean(&slice.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&slice.data).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
    eprintln!("ccderaser: replaced {} total pixels in {} sections", total_replaced, nz);
}

/// Replace an artifact pixel and grow outward to find connected hot pixels.
fn replace_artifact(slice: &mut Slice, cx: usize, cy: usize, max_radius: usize, thresh: f32) {
    let nx = slice.nx;
    let ny = slice.ny;

    // Collect neighbor mean from a ring just outside the artifact
    let r = max_radius + 1;
    let mut ring_sum = 0.0f32;
    let mut ring_count = 0;

    for dy in -(r as i32)..=(r as i32) {
        for dx in -(r as i32)..=(r as i32) {
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist > max_radius as f32 && dist <= r as f32 + 0.5 {
                let px = cx as i32 + dx;
                let py = cy as i32 + dy;
                if px >= 0 && px < nx as i32 && py >= 0 && py < ny as i32 {
                    ring_sum += slice.get(px as usize, py as usize);
                    ring_count += 1;
                }
            }
        }
    }

    let fill_val = if ring_count > 0 { ring_sum / ring_count as f32 } else { slice.get(cx, cy) };

    // Replace pixels within max_radius that deviate from the ring mean
    for dy in -(max_radius as i32)..=(max_radius as i32) {
        for dx in -(max_radius as i32)..=(max_radius as i32) {
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= (max_radius * max_radius) as i32 {
                let px = cx as i32 + dx;
                let py = cy as i32 + dy;
                if px >= 0 && px < nx as i32 && py >= 0 && py < ny as i32 {
                    let val = slice.get(px as usize, py as usize);
                    if (val - fill_val).abs() > thresh {
                        slice.set(px as usize, py as usize, fill_val);
                    }
                }
            }
        }
    }
}
