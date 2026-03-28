use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};
use imod_mrc::MrcReader;

/// Find section boundaries and surfaces in tomographic volumes.
///
/// Analyzes density statistics at multiple scales to find where a
/// section (sample) is located within a reconstructed tomogram. Can output
/// a surface model marking the top and bottom boundaries, a pitch model
/// for tomogram positioning, and/or boundary coordinates.
#[derive(Parser)]
#[command(name = "findsection", version, about)]
struct Args {
    /// Input tomogram file(s).
    #[arg(short = 't', long = "tomo", required = true)]
    tomo_files: Vec<String>,

    /// Output surface model file.
    #[arg(short = 's', long = "surface")]
    surface_model: Option<String>,

    /// Output tomo pitch model file.
    #[arg(long = "pitch")]
    pitch_model: Option<String>,

    /// Fit pitch lines separately for top and bottom.
    #[arg(long = "separate")]
    separate_pitch: bool,

    /// Number of samples along Y for boundary analysis.
    #[arg(long = "samples")]
    num_samples: Option<usize>,

    /// Sample extent in Y.
    #[arg(long = "extent")]
    sample_extent: Option<i32>,

    /// Criterion SD value for identifying high-SD boxes.
    #[arg(long = "high")]
    high_sd_crit: Option<f32>,

    /// Bead model file for bead-based analysis.
    #[arg(long = "bead")]
    bead_file: Option<String>,

    /// Bead diameter in pixels.
    #[arg(long = "diameter", default_value_t = 5.0)]
    bead_diameter: f32,

    /// Number of default scales (binnings) to use.
    #[arg(long = "scales", default_value_t = 1)]
    num_scales: usize,

    /// Size of boxes in X, Y, Z (comma-separated).
    #[arg(long = "size", value_delimiter = ',', num_args = 3)]
    box_size: Vec<i32>,

    /// Spacing of boxes in X, Y, Z (comma-separated).
    #[arg(long = "spacing", value_delimiter = ',', num_args = 3)]
    box_spacing: Vec<i32>,

    /// Block size for column analysis.
    #[arg(long = "block")]
    block_size: Option<i32>,

    /// X min and max range.
    #[arg(long = "xminmax", num_args = 2)]
    x_range: Vec<i32>,

    /// Y min and max range.
    #[arg(long = "yminmax", num_args = 2)]
    y_range: Vec<i32>,

    /// Z min and max range.
    #[arg(long = "zminmax", num_args = 2)]
    z_range: Vec<i32>,

    /// Thick dimension is Y (1) or not (0), or auto-detect (-1).
    #[arg(long = "flipped", default_value_t = -1)]
    flipped: i32,

    /// Volume rootname for output.
    #[arg(long = "volume")]
    volume_root: Option<String>,

    /// Point rootname for output.
    #[arg(long = "point")]
    point_root: Option<String>,

    /// Debug output level.
    #[arg(long = "debug", default_value_t = 0)]
    debug: i32,
}

/// Statistics for a box of pixels.
#[allow(dead_code)]
struct BoxStats {
    mean: f64,
    sd: f64,
}

/// Compute mean and standard deviation over a subvolume box.
fn compute_box_stats(data: &[f32], nx: usize, ny: usize,
                     x0: usize, y0: usize, bx: usize, by: usize) -> BoxStats {
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut n = 0usize;

    for iy in y0..(y0 + by).min(ny) {
        for ix in x0..(x0 + bx).min(nx) {
            let v = data[iy * nx + ix] as f64;
            sum += v;
            sum2 += v * v;
            n += 1;
        }
    }

    if n < 2 {
        return BoxStats { mean: sum, sd: 0.0 };
    }
    let mean = sum / n as f64;
    let var = (sum2 - sum * sum / n as f64) / (n - 1) as f64;
    BoxStats {
        mean,
        sd: var.max(0.0).sqrt(),
    }
}

/// Simple median for a sorted slice of f32 values.
fn simple_median(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    }
}

/// Compute MADN (median absolute deviation, normalized) for a slice.
#[allow(dead_code)]
fn madn(vals: &[f32], med: f32) -> f32 {
    let mut devs: Vec<f32> = vals.iter().map(|v| (v - med).abs()).collect();
    simple_median(&mut devs) * 1.4826
}

fn main() {
    let args = Args::parse();

    if args.tomo_files.is_empty() {
        eprintln!("ERROR: findsection - no input tomogram file(s) specified");
        process::exit(1);
    }

    if args.box_size.is_empty() {
        eprintln!("ERROR: findsection - size of boxes must be entered");
        process::exit(1);
    }

    // Open first tomogram to get dimensions
    let mut reader = MrcReader::open(&args.tomo_files[0]).unwrap_or_else(|e| {
        eprintln!("ERROR: findsection - could not open {}: {}", args.tomo_files[0], e);
        process::exit(1);
    });

    let hdr = reader.header();
    let nx = hdr.nx as usize;
    let ny = hdr.ny as usize;
    let nz = hdr.nz as usize;

    // Determine thick dimension
    let thick_ind = if args.flipped > 0 || (args.flipped < 0 && ny < nz) {
        1 // Y is thick dimension
    } else {
        2 // Z is thick dimension (normal)
    };

    println!("Volume dimensions: {} x {} x {}", nx, ny, nz);
    println!(
        "Thick dimension: {}",
        if thick_ind == 1 { "Y" } else { "Z" }
    );

    let bx = args.box_size.get(0).copied().unwrap_or(32) as usize;
    let by = args.box_size.get(1).copied().unwrap_or(32) as usize;
    let bz = args.box_size.get(2).copied().unwrap_or(32) as usize;

    let sx = if args.box_spacing.len() >= 3 { args.box_spacing[0] as usize } else { bx / 2 };
    let _sy = if args.box_spacing.len() >= 3 { args.box_spacing[1] as usize } else { by / 2 };
    let sz = if args.box_spacing.len() >= 3 { args.box_spacing[2] as usize } else { bz / 2 };

    // Determine analysis ranges
    let x_min = args.x_range.get(0).copied().unwrap_or(0) as usize;
    let x_max = args.x_range.get(1).copied().unwrap_or(nx as i32 - 1) as usize;
    let y_min = args.y_range.get(0).copied().unwrap_or(0) as usize;
    let y_max = args.y_range.get(1).copied().unwrap_or(ny as i32 - 1) as usize;
    let z_min = args.z_range.get(0).copied().unwrap_or(0) as usize;
    let z_max = args.z_range.get(1).copied().unwrap_or(nz as i32 - 1) as usize;

    // For the standard case (thick_ind == Z), scan through Z slices and
    // compute SD statistics in boxes to find where the section boundaries are.
    let num_z_blocks = if sz > 0 { (z_max - z_min) / sz } else { 1 };
    let num_x_blocks = if sx > 0 { (x_max - x_min) / sx } else { 1 };

    // Compute SD profiles along the thick dimension
    let mut z_sds: Vec<f32> = Vec::with_capacity(num_z_blocks);

    for iz_block in 0..num_z_blocks {
        let iz_center = z_min + iz_block * sz + bz / 2;
        if iz_center >= nz {
            break;
        }

        let mut block_sds = Vec::new();

        // Sample a few Z slices within the block
        let z_start = iz_center.saturating_sub(bz / 2);
        let z_end = (iz_center + bz / 2).min(nz);

        for iz in z_start..z_end {
            let slice_data = match reader.read_slice_f32(iz) {
                Ok(d) => d,
                Err(_) => continue,
            };

            for ix_block in 0..num_x_blocks {
                let x0 = x_min + ix_block * sx;
                if x0 + bx > nx {
                    break;
                }
                // Sample in the middle Y range
                let y0 = (y_min + y_max) / 2 - by / 2;
                let stats = compute_box_stats(&slice_data, nx, ny, x0, y0, bx, by);
                block_sds.push(stats.sd as f32);
            }
        }

        if !block_sds.is_empty() {
            let med = simple_median(&mut block_sds);
            z_sds.push(med);
        }
    }

    if z_sds.is_empty() {
        eprintln!("ERROR: findsection - no valid data blocks found");
        process::exit(1);
    }

    // Find section boundaries: the section is where SD is highest.
    // Find the peak in the SD profile and then find where it drops to
    // half the peak on either side.
    let max_sd = z_sds.iter().cloned().fold(0.0_f32, f32::max);
    let edge_sds = z_sds.clone();
    let min_sd = simple_median(&mut edge_sds.iter().copied().take(3).collect::<Vec<_>>());
    let threshold = min_sd + (max_sd - min_sd) * 0.5;

    let mut bot_boundary = 0usize;
    let mut top_boundary = z_sds.len() - 1;

    for (i, &sd) in z_sds.iter().enumerate() {
        if sd > threshold {
            bot_boundary = i;
            break;
        }
    }
    for i in (0..z_sds.len()).rev() {
        if z_sds[i] > threshold {
            top_boundary = i;
            break;
        }
    }

    let bot_z = z_min + bot_boundary * sz;
    let top_z = z_min + top_boundary * sz;
    let thickness = top_z as f32 - bot_z as f32;

    println!("Section boundaries:");
    println!("  Bottom: Z = {}", bot_z);
    println!("  Top:    Z = {}", top_z);
    println!("  Thickness: {:.0} pixels", thickness);

    // Write surface model if requested
    if let Some(ref surface_name) = args.surface_model {
        let mut model = ImodModel::default();
        model.xmax = nx as i32;
        model.ymax = ny as i32;
        model.zmax = nz as i32;

        // Bottom surface object
        let mut bot_obj = ImodObject::default();
        bot_obj.name = "Bottom surface".into();
        bot_obj.red = 0.0;
        bot_obj.green = 1.0;
        bot_obj.blue = 0.0;

        let bot_cont = ImodContour {
            points: vec![
                Point3f { x: 0.0, y: ny as f32 / 2.0, z: bot_z as f32 },
                Point3f { x: nx as f32, y: ny as f32 / 2.0, z: bot_z as f32 },
            ],
            ..Default::default()
        };
        bot_obj.contours.push(bot_cont);
        model.objects.push(bot_obj);

        // Top surface object
        let mut top_obj = ImodObject::default();
        top_obj.name = "Top surface".into();
        top_obj.red = 1.0;
        top_obj.green = 0.0;
        top_obj.blue = 0.0;

        let top_cont = ImodContour {
            points: vec![
                Point3f { x: 0.0, y: ny as f32 / 2.0, z: top_z as f32 },
                Point3f { x: nx as f32, y: ny as f32 / 2.0, z: top_z as f32 },
            ],
            ..Default::default()
        };
        top_obj.contours.push(top_cont);
        model.objects.push(top_obj);

        if let Err(e) = write_model(surface_name, &model) {
            eprintln!("ERROR: findsection - error writing surface model: {}", e);
            process::exit(1);
        }
        println!("Wrote surface model to {}", surface_name);
    }

    // Write pitch model if requested
    if let Some(ref pitch_name) = args.pitch_model {
        let mut model = ImodModel::default();
        model.xmax = nx as i32;
        model.ymax = ny as i32;
        model.zmax = nz as i32;

        let mut obj = ImodObject::default();
        obj.name = "Pitch lines".into();
        obj.flags |= 1 << 3; // IMOD_OBJFLAG_OPEN
        obj.red = 1.0;
        obj.green = 1.0;
        obj.blue = 0.0;

        // Bottom pitch line
        let bot_cont = ImodContour {
            points: vec![
                Point3f { x: 0.0, y: ny as f32 / 2.0, z: bot_z as f32 },
                Point3f { x: nx as f32, y: ny as f32 / 2.0, z: bot_z as f32 },
            ],
            ..Default::default()
        };
        obj.contours.push(bot_cont);

        // Top pitch line
        let top_cont = ImodContour {
            points: vec![
                Point3f { x: 0.0, y: ny as f32 / 2.0, z: top_z as f32 },
                Point3f { x: nx as f32, y: ny as f32 / 2.0, z: top_z as f32 },
            ],
            ..Default::default()
        };
        obj.contours.push(top_cont);

        model.objects.push(obj);

        if let Err(e) = write_model(pitch_name, &model) {
            eprintln!("ERROR: findsection - error writing pitch model: {}", e);
            process::exit(1);
        }
        println!("Wrote pitch model to {}", pitch_name);
    }
}
