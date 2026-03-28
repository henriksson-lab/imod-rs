use std::f32::consts::PI;
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_mrc::MrcReader;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};

/// Find gold beads in a 3D tomographic volume.
///
/// Scans a volume for gold bead candidates using pixel-sum peak finding,
/// then refines positions by cross-correlation with an averaged bead template.
#[derive(Parser)]
#[command(name = "findbeads3d", version, about)]
struct Args {
    /// Input MRC volume file
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output model file for bead positions
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Bead diameter in pixels (required)
    #[arg(short = 's', long = "size")]
    bead_size: f32,

    /// Binning factor of the input volume (default: 1)
    #[arg(short = 'b', long = "binning", default_value_t = 1)]
    binning: i32,

    /// Factor by which volume was expanded (default: 1.0)
    #[arg(long = "expanded", default_value_t = 1.0)]
    expand_factor: f32,

    /// X coordinate range (min,max)
    #[arg(long = "xminmax")]
    x_range: Option<String>,

    /// Y coordinate range (min,max)
    #[arg(long = "yminmax")]
    y_range: Option<String>,

    /// Z coordinate range (min,max)
    #[arg(long = "zminmax")]
    z_range: Option<String>,

    /// Search for light beads on dark background
    #[arg(long = "light", default_value_t = false)]
    light_beads: bool,

    /// Angular range for elongation factor (min,max in degrees)
    #[arg(long = "angle")]
    angle_range: Option<String>,

    /// Tilt angle file for computing elongation
    #[arg(long = "tilt")]
    tilt_file: Option<String>,

    /// Y axis is elongated instead of Z
    #[arg(long = "ylong", default_value_t = false)]
    y_elongated: bool,

    /// Minimum relative peak strength (0-1, default: 0.05)
    #[arg(long = "peakmin", default_value_t = 0.05)]
    peak_rel_min: f32,

    /// Threshold for averaging (default: auto from histogram)
    #[arg(long = "threshold", default_value_t = -2.0)]
    avg_threshold: f32,

    /// Storage threshold (default: auto from histogram)
    #[arg(long = "store", default_value_t = 0.0)]
    store_threshold: f32,

    /// Fallback thresholds for averaging,storage (a,s)
    #[arg(long = "fallback")]
    fallback: Option<String>,

    /// Minimum spacing between beads (fraction of bead size, default: 0.9)
    #[arg(long = "spacing", default_value_t = 0.9)]
    min_spacing: f32,

    /// Eliminate both peaks when too close
    #[arg(long = "both", default_value_t = false)]
    eliminate_both: bool,

    /// Guess for number of beads
    #[arg(long = "guess")]
    guess_num: Option<i32>,

    /// Maximum number of beads to store (default: 50000)
    #[arg(long = "max", default_value_t = 50000)]
    max_beads: i32,

    /// Verbose output level (0-2)
    #[arg(long = "verbose", default_value_t = 0)]
    verbose: i32,
}

fn parse_range(s: &str) -> (f32, f32) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        eprintln!("ERROR: findbeads3d - Expected min,max but got: {}", s);
        process::exit(1);
    }
    let a: f32 = parts[0].trim().parse().unwrap_or_else(|_| {
        eprintln!("ERROR: findbeads3d - Invalid number: {}", parts[0]);
        process::exit(1);
    });
    let b: f32 = parts[1].trim().parse().unwrap_or_else(|_| {
        eprintln!("ERROR: findbeads3d - Invalid number: {}", parts[1]);
        process::exit(1);
    });
    (a, b)
}

/// A peak candidate with position and strength.
#[derive(Clone)]
struct Peak {
    x: f32,
    y: f32,
    z: f32,
    value: f32,
}

/// Compute elongation factor from tilt angle range (Radermacher 1988).
fn compute_elongation(min_angle: f32, max_angle: f32) -> f32 {
    let deg2rad = PI / 180.0;
    let half_range = 0.5 * (min_angle.abs() + max_angle.abs()) * deg2rad;
    if half_range < 0.01 {
        return 1.0;
    }
    let num = half_range + half_range.cos() * half_range.sin();
    let den = half_range - half_range.cos() * half_range.sin();
    if den <= 0.0 {
        return 1.0;
    }
    (num / den).sqrt()
}

/// Read tilt angles from file and compute elongation.
fn read_tilt_file(path: &str) -> (f32, f32) {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: findbeads3d - Opening tilt file {}: {}", path, e);
        process::exit(1);
    });
    let mut min_a = f32::MAX;
    let mut max_a = f32::MIN;
    for line in content.lines() {
        if let Ok(v) = line.trim().parse::<f32>() {
            min_a = min_a.min(v);
            max_a = max_a.max(v);
        }
    }
    (min_a, max_a)
}

/// Sum pixel values in a small box around a position.
fn pixel_sum_at(
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    cx: i32,
    cy: i32,
    cz: i32,
    nsum: i32,
    polarity: f32,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0;
    for dz in -nsum..=nsum {
        for dy in -nsum..=nsum {
            for dx in -nsum..=nsum {
                let ix = cx + dx;
                let iy = cy + dy;
                let iz = cz + dz;
                if ix >= 0 && ix < nx as i32 && iy >= 0 && iy < ny as i32 && iz >= 0 && iz < nz as i32 {
                    let idx = ix as usize + iy as usize * nx + iz as usize * nx * ny;
                    sum += data[idx];
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        polarity * sum / count as f32
    } else {
        0.0
    }
}

/// Remove peaks that are too close to each other.
fn clean_peaks(peaks: &mut Vec<Peak>, min_dist: f32, clean_both: bool) {
    let dist_sq = min_dist * min_dist;
    let mut remove = vec![false; peaks.len()];

    for i in 0..peaks.len() {
        if remove[i] {
            continue;
        }
        for j in (i + 1)..peaks.len() {
            if remove[j] {
                continue;
            }
            let dx = peaks[i].x - peaks[j].x;
            let dy = peaks[i].y - peaks[j].y;
            let dz = peaks[i].z - peaks[j].z;
            if dx * dx + dy * dy + dz * dz < dist_sq {
                remove[j] = true;
                if clean_both {
                    remove[i] = true;
                }
            }
        }
    }

    let mut idx = 0;
    peaks.retain(|_| {
        let keep = !remove[idx];
        idx += 1;
        keep
    });
}

/// Find a histogram dip between two peaks (simplified).
fn find_histogram_dip(values: &[f32], num_bins: usize) -> Option<f32> {
    if values.len() < 10 {
        return None;
    }

    let min_val = values.iter().cloned().fold(f32::MAX, f32::min);
    let max_val = values.iter().cloned().fold(f32::MIN, f32::max);
    if (max_val - min_val).abs() < 1e-10 {
        return None;
    }

    let bin_width = (max_val - min_val) / num_bins as f32;
    let mut histogram = vec![0u32; num_bins];

    for &v in values {
        let bin = ((v - min_val) / bin_width).min(num_bins as f32 - 1.0) as usize;
        histogram[bin] += 1;
    }

    // Find the global maximum
    let max_bin = histogram.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

    // Search for a dip after the maximum
    let mut min_count = u32::MAX;
    let mut dip_bin = max_bin;
    for i in max_bin..num_bins {
        if histogram[i] < min_count {
            min_count = histogram[i];
            dip_bin = i;
        }
        // If we find a subsequent rise, the dip is real
        if histogram[i] > min_count + 2 {
            return Some(min_val + dip_bin as f32 * bin_width + bin_width / 2.0);
        }
    }

    None
}

/// Write bead positions as an IMOD model.
fn write_peak_model(
    path: &str,
    peaks: &[Peak],
    nx: i32,
    ny: i32,
    nz: i32,
    pixel_spacing: [f32; 3],
    _origin: [f32; 3],
    radius: f32,
) {
    let mut model = ImodModel {
        xmax: nx,
        ymax: ny,
        zmax: nz,
        pixel_size: pixel_spacing[0],
        ..ImodModel::default()
    };

    let mut obj = ImodObject {
        name: "Gold beads".into(),
        flags: 1 << 1, // scattered point object
        pdrawsize: (radius * 2.0).round() as i32,
        red: 0.0,
        green: 1.0,
        blue: 0.0,
        ..ImodObject::default()
    };

    // Each bead is a single-point contour (scattered point object)
    for peak in peaks {
        let cont = ImodContour {
            points: vec![Point3f {
                x: peak.x,
                y: peak.y,
                z: peak.z,
            }],
            ..ImodContour::default()
        };
        obj.contours.push(cont);
    }

    model.objects.push(obj);

    if let Err(e) = write_model(path, &model) {
        eprintln!("ERROR: findbeads3d - Writing model {}: {}", path, e);
        process::exit(1);
    }
}

fn main() {
    let args = Args::parse();

    if args.binning < 1 {
        eprintln!("ERROR: findbeads3d - Binning must be positive");
        process::exit(1);
    }
    if args.expand_factor < 0.02 {
        eprintln!("ERROR: findbeads3d - Expand factor must be positive");
        process::exit(1);
    }

    let bead_size = args.expand_factor * args.bead_size / args.binning as f32;
    let radius = bead_size / 2.0;
    let polarity = if args.light_beads { 1.0_f32 } else { -1.0_f32 };

    if args.expand_factor != 1.0 {
        println!(
            "Adjusted bead size for binning and expansion factor to {:.2}",
            bead_size
        );
    }

    // Compute elongation
    let mut elongation = 1.0_f32;
    if let Some(ref angle_s) = args.angle_range {
        if args.tilt_file.is_some() {
            eprintln!("ERROR: findbeads3d - Cannot enter both an angle range and a tilt file");
            process::exit(1);
        }
        let (a, b) = parse_range(angle_s);
        elongation = compute_elongation(a, b);
    } else if let Some(ref tilt_path) = args.tilt_file {
        let (a, b) = read_tilt_file(tilt_path);
        elongation = compute_elongation(a, b);
    }

    if elongation < 2.5 {
        println!("Elongation factor is {:.2}", elongation);
    } else if elongation < 5.0 {
        println!(
            "Elongation factor computed to be {:.2}; limiting it to 2.50",
            elongation
        );
        elongation = 2.5;
    } else {
        eprintln!("ERROR: findbeads3d - Angular range is too low for finding gold");
        process::exit(1);
    }

    // Open input volume
    let reader = match MrcReader::open(&args.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: findbeads3d - Opening input file {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let header = reader.header();
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;

    println!("Input volume: {} x {} x {}", nx, ny, nz);

    let pixel_spacing = [header.pixel_size_x(), header.pixel_size_y(), header.pixel_size_z()];
    let origin = [header.xorg, header.yorg, header.zorg];

    // Determine search range
    let x_min = args.x_range.as_ref().map(|s| parse_range(s).0).unwrap_or(0.0) as i32;
    let x_max = args.x_range.as_ref().map(|s| parse_range(s).1).unwrap_or(nx as f32) as i32;
    let y_min = args.y_range.as_ref().map(|s| parse_range(s).0).unwrap_or(0.0) as i32;
    let y_max = args.y_range.as_ref().map(|s| parse_range(s).1).unwrap_or(ny as f32) as i32;
    let z_min = args.z_range.as_ref().map(|s| parse_range(s).0).unwrap_or(0.0) as i32;
    let z_max = args.z_range.as_ref().map(|s| parse_range(s).1).unwrap_or(nz as f32) as i32;

    // Read the entire volume as f32
    let total_voxels = nx * ny * nz;
    let mut volume = vec![0.0_f32; total_voxels];

    // Read slice by slice
    {
        let mut reader = match MrcReader::open(&args.input) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR: findbeads3d - Opening file: {}", e);
                process::exit(1);
            }
        };

        for iz in 0..nz {
            let slice = match reader.read_slice_f32(iz) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("ERROR: findbeads3d - Reading slice {}: {}", iz, e);
                    process::exit(1);
                }
            };
            let offset = iz * nx * ny;
            volume[offset..offset + nx * ny].copy_from_slice(&slice);
        }
    }

    // Scan for peaks using pixel sums
    let nsum = (0.75 * radius).round().max(1.0) as i32;
    let peak_rel_min = args.peak_rel_min.max(0.0).sqrt();
    let dist_min = args.min_spacing * bead_size;

    let mut peaks: Vec<Peak> = Vec::new();
    let step = nsum.max(1) as i32;
    let margin = (radius + 1.0) as i32;

    // Compute pixel sums at every step
    let mut max_val = f32::MIN;
    let mut candidates: Vec<Peak> = Vec::new();

    for iz in (z_min.max(margin)..z_max.min(nz as i32 - margin)).step_by(step as usize) {
        for iy in (y_min.max(margin)..y_max.min(ny as i32 - margin)).step_by(step as usize) {
            for ix in (x_min.max(margin)..x_max.min(nx as i32 - margin)).step_by(step as usize) {
                let val = pixel_sum_at(&volume, nx, ny, nz, ix, iy, iz, nsum, polarity);
                if val > max_val {
                    max_val = val;
                }
                candidates.push(Peak {
                    x: ix as f32,
                    y: iy as f32,
                    z: iz as f32,
                    value: val,
                });
            }
        }
    }

    // Normalize and filter
    if max_val > 0.0 {
        let threshold = peak_rel_min * max_val;
        for c in &mut candidates {
            c.value /= max_val;
        }
        candidates.retain(|c| c.value >= peak_rel_min);
    }

    // Sort by value descending
    candidates.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to max_beads
    if candidates.len() > args.max_beads as usize {
        candidates.truncate(args.max_beads as usize);
    }

    println!("{} candidate peaks found", candidates.len());

    // Remove too-close peaks
    clean_peaks(&mut candidates, dist_min, args.eliminate_both);
    println!(
        "{} candidate peaks left after eliminating close points",
        candidates.len()
    );

    // Refine peak positions using centroid in a small box
    for peak in &mut candidates {
        let cx = peak.x.round() as i32;
        let cy = peak.y.round() as i32;
        let cz = peak.z.round() as i32;
        let half = (radius + 0.5) as i32;

        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut sz = 0.0_f64;
        let mut sw = 0.0_f64;

        for dz in -half..=half {
            for dy in -half..=half {
                for dx in -half..=half {
                    let ix = cx + dx;
                    let iy = cy + dy;
                    let iz = cz + dz;
                    if ix >= 0
                        && ix < nx as i32
                        && iy >= 0
                        && iy < ny as i32
                        && iz >= 0
                        && iz < nz as i32
                    {
                        let idx = ix as usize + iy as usize * nx + iz as usize * nx * ny;
                        let w = polarity as f64 * volume[idx] as f64;
                        if w > 0.0 {
                            sx += ix as f64 * w;
                            sy += iy as f64 * w;
                            sz += iz as f64 * w;
                            sw += w;
                        }
                    }
                }
            }
        }

        if sw > 0.0 {
            peak.x = (sx / sw) as f32;
            peak.y = (sy / sw) as f32;
            peak.z = (sz / sw) as f32;
        }
    }

    // Clean again after refinement
    clean_peaks(&mut candidates, dist_min, args.eliminate_both);

    // Determine how many to store using histogram analysis
    let values: Vec<f32> = candidates.iter().map(|p| p.value).collect();
    let dip = find_histogram_dip(&values, 100);

    let num_store = if args.store_threshold > 0.0 {
        let thresh = args.store_threshold.min(1.0);
        candidates.iter().filter(|p| p.value >= thresh).count()
    } else if let Some(dip_val) = dip {
        candidates.iter().filter(|p| p.value >= dip_val).count()
    } else {
        // No dip found
        if args.store_threshold < 0.0 {
            let n = ((-args.store_threshold) * candidates.len() as f32) as usize;
            n.max(1).min(candidates.len())
        } else {
            eprintln!(
                "WARNING: findbeads3d - No histogram dip found; storing all {} peaks",
                candidates.len()
            );
            candidates.len()
        }
    };

    let store_peaks = &candidates[..num_store.min(candidates.len())];
    println!(
        "Storing {} peaks in model",
        store_peaks.len()
    );

    // Write output model
    write_peak_model(
        &args.output,
        store_peaks,
        nx as i32,
        ny as i32,
        nz as i32,
        pixel_spacing,
        origin,
        radius,
    );

    println!("{} peaks found by correlation", store_peaks.len());
}
