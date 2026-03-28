use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, read_model, write_model};

/// Pick the best seed beads for fiducial tracking.
///
/// Reads one or more tracked bead models along with elongation/residual data
/// and selects the best subset of beads for use as a seed model. Selection is
/// based on tracking completeness, residual quality, elongation (overlap
/// detection), clustering, and spatial distribution.
#[derive(Parser)]
#[command(name = "pickbestseed", version, about)]
struct Args {
    /// Tracked model file(s) from beadtrack
    #[arg(short = 'T', long = "tracked", required = true)]
    tracked_models: Vec<String>,

    /// Elongation/residual data file(s), one per tracked model
    #[arg(short = 'r', long = "resid", required = true)]
    elong_files: Vec<String>,

    /// Output seed model file
    #[arg(short = 'O', long = "output")]
    output: String,

    /// Bead diameter in pixels
    #[arg(short = 'b', long = "size")]
    bead_size: f32,

    /// Image size in X and Y
    #[arg(short = 'I', long = "image", num_args = 2)]
    image_size: Vec<i32>,

    /// Middle Z value of the tilt series
    #[arg(short = 'm', long = "middle")]
    middle_z: i32,

    /// Target number of beads to select
    #[arg(short = 'n', long = "number")]
    target_number: Option<i32>,

    /// Target density of beads (per megapixel)
    #[arg(short = 'd', long = "density")]
    target_density: Option<f32>,

    /// Use two surfaces (top and bottom)
    #[arg(long = "two")]
    two_surfaces: bool,

    /// Surface data file(s) for two-surface mode
    #[arg(long = "surface")]
    surface_files: Vec<String>,

    /// Boundary model file for restricting area
    #[arg(short = 'B', long = "boundary")]
    boundary_model: Option<String>,

    /// Exclude inside areas of boundary contours
    #[arg(long = "exclude")]
    exclude_areas: bool,

    /// Append to existing seed model
    #[arg(long = "append")]
    append: bool,

    /// Borders in X and Y to exclude
    #[arg(long = "border", num_args = 2)]
    borders: Option<Vec<i32>>,

    /// Rotation angle
    #[arg(long = "rotation", default_value_t = 0.0)]
    rotation: f32,

    /// Highest tilt angle
    #[arg(long = "highest", default_value_t = 0.0)]
    highest_tilt: f32,

    /// Weights for scoring: completeness, num_models, residual, deviation
    #[arg(short = 'w', long = "weights", num_args = 4, value_delimiter = ',')]
    weights: Option<Vec<f32>>,

    /// Maximum number of clustered points allowed (0-4)
    #[arg(long = "cluster", default_value_t = 0)]
    clustered: i32,

    /// Maximum number of elongated points allowed (0-3)
    #[arg(long = "elongated")]
    elongated: Option<i32>,

    /// Lower target for clustered/elongated beads
    #[arg(long = "lower")]
    lower_target: Option<f32>,

    /// Limit majority surface to target
    #[arg(long = "nobeef")]
    no_beef_up: bool,

    /// Seed Z values for each tracked model
    #[arg(long = "zseed")]
    seed_z: Vec<i32>,

    /// Verbose output level
    #[arg(short = 'v', long = "verbose", default_value_t = 0)]
    verbose: i32,
}

/// Track data for a single bead contour
struct TrackData {
    model: i32,
    residual: f32,
    edge_sd_mean: f32,
    edge_sd_median: f32,
    edge_sd_sd: f32,
    elong_mean: f32,
    elong_median: f32,
    elong_sd: f32,
    wsum_mean: f32,
    top_bot: i32,
    score: f64,
    selected: bool,
}

fn main() {
    let args = Args::parse();

    if args.image_size.len() != 2 {
        eprintln!("ERROR: pickbestseed - Image size must have 2 values");
        process::exit(1);
    }

    let nx_image = args.image_size[0];
    let ny_image = args.image_size[1];
    let num_models = args.tracked_models.len();

    if args.elong_files.len() != num_models {
        eprintln!("ERROR: pickbestseed - Number of elongation files must match tracked models");
        process::exit(1);
    }

    let has_density = args.target_density.is_some();
    let has_number = args.target_number.is_some();
    if (has_density as i32 + has_number as i32) != 1 {
        eprintln!("ERROR: pickbestseed - Target number or density must be entered, not both");
        process::exit(1);
    }

    let weights = args.weights.unwrap_or(vec![1.0, 1.0, 1.0, 1.0]);
    let x_border = args.borders.as_ref().map_or(0, |b| b[0]);
    let y_border = args.borders.as_ref().map_or(0, |b| b[1]);

    // Read tracked models
    let mut track_mods = Vec::new();
    let mut max_conts = 0usize;
    for filename in &args.tracked_models {
        let model = read_model(filename).unwrap_or_else(|e| {
            eprintln!("ERROR: pickbestseed - Reading tracked model {}: {}", filename, e);
            process::exit(1);
        });
        if model.objects.is_empty() || model.objects[0].contours.is_empty() {
            eprintln!("ERROR: pickbestseed - No contours in tracked model {}", filename);
            process::exit(1);
        }
        max_conts = max_conts.max(model.objects[0].contours.len());
        track_mods.push(model);
    }

    // Initialize track data
    let mut tracks: Vec<TrackData> = (0..max_conts).map(|_| TrackData {
        model: -1,
        residual: 0.0,
        edge_sd_mean: 0.0,
        edge_sd_median: 0.0,
        edge_sd_sd: 0.0,
        elong_mean: 0.0,
        elong_median: 0.0,
        elong_sd: 0.0,
        wsum_mean: 0.0,
        top_bot: 0,
        score: 0.0,
        selected: false,
    }).collect();

    // Read elongation/residual files
    for (i, filename) in args.elong_files.iter().enumerate() {
        let file = File::open(filename).unwrap_or_else(|e| {
            eprintln!("ERROR: pickbestseed - Opening file {}: {}", filename, e);
            process::exit(1);
        });
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap_or_default();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 9 {
                continue;
            }
            let co: usize = parts[1].parse().unwrap_or(0);
            let resid: f32 = parts[2].parse().unwrap_or(-1.0);
            let sd_mean: f32 = parts[3].parse().unwrap_or(-1.0);
            if resid < 0.0 || sd_mean < 0.0 {
                continue;
            }
            if co < 1 || co > max_conts {
                eprintln!("ERROR: pickbestseed - Contour number {} out of range in {}", co, filename);
                process::exit(1);
            }
            let idx = co - 1;
            tracks[idx].model = i as i32;
            tracks[idx].residual = resid;
            tracks[idx].edge_sd_mean = sd_mean;
            tracks[idx].edge_sd_median = parts[4].parse().unwrap_or(0.0);
            tracks[idx].edge_sd_sd = parts[5].parse().unwrap_or(0.0);
            tracks[idx].elong_mean = parts[6].parse().unwrap_or(0.0);
            tracks[idx].elong_median = parts[7].parse().unwrap_or(0.0);
            tracks[idx].elong_sd = parts[8].parse().unwrap_or(0.0);
            if parts.len() > 9 {
                tracks[idx].wsum_mean = parts[9].parse().unwrap_or(0.0);
            }
        }
    }

    // Read surface files if any
    for filename in &args.surface_files {
        let file = File::open(filename).unwrap_or_else(|e| {
            eprintln!("ERROR: pickbestseed - Opening file {}: {}", filename, e);
            process::exit(1);
        });
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap_or_default();
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            let co: usize = parts[1].parse().unwrap_or(0);
            let top_bot: i32 = parts[3].parse().unwrap_or(0);
            if co >= 1 && co <= max_conts {
                tracks[co - 1].top_bot = top_bot;
            }
        }
    }

    // Compute total area
    let total_area = ((nx_image - 2 * x_border) * (ny_image - 2 * y_border)) as f64;
    println!("Total area = {:.2} megapixels", total_area / 1.0e6);

    // Determine target number
    let target_number = if let Some(n) = args.target_number {
        n as usize
    } else {
        let density = args.target_density.unwrap();
        (density as f64 * total_area / 1.0e6).round() as usize
    };

    let area_xmin = x_border as f32;
    let area_xmax = (nx_image - x_border) as f32;
    let area_ymin = y_border as f32;
    let area_ymax = (ny_image - y_border) as f32;

    // Score each track: combine completeness, residual, edge SD, elongation
    // Use the first tracked model as reference
    let ref_obj = &track_mods[0].objects[0];
    let num_sections = track_mods[0].zmax;
    let wgt_complete = weights[0];
    let wgt_nmod = weights[1] / (num_models as f32 - 1.0).max(1.0);
    let wgt_resid = weights[2];
    let wgt_dev = weights[3];

    // Get mean residual and edge SD for normalization
    let mut sum_resid = 0.0f64;
    let mut sum_edge = 0.0f64;
    let mut n_valid = 0usize;
    for t in &tracks {
        if t.model >= 0 {
            sum_resid += t.residual as f64;
            sum_edge += t.edge_sd_mean as f64;
            n_valid += 1;
        }
    }
    let mean_resid = if n_valid > 0 { sum_resid / n_valid as f64 } else { 1.0 };
    let mean_edge = if n_valid > 0 { sum_edge / n_valid as f64 } else { 1.0 };

    for (co, track) in tracks.iter_mut().enumerate() {
        if track.model < 0 {
            track.score = f64::MAX;
            continue;
        }

        // Check if point is within borders
        if co < ref_obj.contours.len() && !ref_obj.contours[co].points.is_empty() {
            let mid_pt = &ref_obj.contours[co].points[0];
            if mid_pt.x < area_xmin || mid_pt.x > area_xmax
                || mid_pt.y < area_ymin || mid_pt.y > area_ymax
            {
                track.score = f64::MAX;
                continue;
            }
        }

        // Completeness: fraction of sections tracked
        let npts = if co < ref_obj.contours.len() {
            ref_obj.contours[co].points.len()
        } else {
            0
        };
        let completeness = npts as f64 / num_sections.max(1) as f64;

        // Normalize residual and edge SD
        let norm_resid = track.residual as f64 / mean_resid.max(0.001);
        let norm_edge = track.edge_sd_mean as f64 / mean_edge.max(0.001);

        // Score: lower is better
        track.score = wgt_complete as f64 * (1.0 - completeness)
            + wgt_resid as f64 * norm_resid
            + wgt_dev as f64 * norm_edge;
    }

    // Select best beads by score, maintaining spatial distribution
    let target_spacing = (total_area / target_number as f64).sqrt() as f32;
    let exclude_dist = target_spacing * 0.75;

    // Sort indices by score
    let mut indices: Vec<usize> = (0..tracks.len()).collect();
    indices.sort_by(|&a, &b| {
        tracks[a].score.partial_cmp(&tracks[b].score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected_count = 0usize;
    let mut selected_positions: Vec<(f32, f32)> = Vec::new();

    for &idx in &indices {
        if selected_count >= target_number {
            break;
        }
        if tracks[idx].score >= f64::MAX * 0.5 {
            continue;
        }
        if idx >= ref_obj.contours.len() || ref_obj.contours[idx].points.is_empty() {
            continue;
        }

        let pt = &ref_obj.contours[idx].points[0];

        // Check distance from already selected points
        let too_close = selected_positions.iter().any(|&(sx, sy)| {
            let dx = pt.x - sx;
            let dy = pt.y - sy;
            (dx * dx + dy * dy).sqrt() < exclude_dist
        });

        if too_close {
            continue;
        }

        tracks[idx].selected = true;
        selected_positions.push((pt.x, pt.y));
        selected_count += 1;
    }

    // If we haven't reached target, do a second pass ignoring spacing
    if selected_count < target_number {
        for &idx in &indices {
            if selected_count >= target_number {
                break;
            }
            if tracks[idx].selected || tracks[idx].score >= f64::MAX * 0.5 {
                continue;
            }
            if idx >= ref_obj.contours.len() || ref_obj.contours[idx].points.is_empty() {
                continue;
            }
            tracks[idx].selected = true;
            selected_count += 1;
        }
    }

    println!("Selected {} beads out of {} candidates", selected_count, n_valid);

    // Build output seed model
    let iz_middle = args.middle_z;
    let mut seed_contours = Vec::new();

    for (co, track) in tracks.iter().enumerate() {
        if !track.selected {
            continue;
        }
        if co >= ref_obj.contours.len() {
            continue;
        }
        let ref_cont = &ref_obj.contours[co];

        // Find the point closest to middle Z
        let mut best_pt = None;
        let mut best_dz = f32::MAX;
        for pt in &ref_cont.points {
            let dz = (pt.z - iz_middle as f32).abs();
            if dz < best_dz {
                best_dz = dz;
                best_pt = Some(*pt);
            }
        }

        if let Some(pt) = best_pt {
            let cont = ImodContour {
                points: vec![Point3f { x: pt.x, y: pt.y, z: iz_middle as f32 }],
                surf: if args.two_surfaces { track.top_bot.max(0) } else { 0 },
                ..Default::default()
            };
            seed_contours.push(cont);
        }
    }

    // Read and append if requested
    let mut base_contours = Vec::new();
    if args.append {
        if let Ok(base_mod) = read_model(&args.output) {
            if !base_mod.objects.is_empty() {
                base_contours = base_mod.objects[0].contours.clone();
            }
        }
    }
    base_contours.extend(seed_contours);

    let obj = ImodObject {
        contours: base_contours,
        pdrawsize: (args.bead_size / 2.0) as i32,
        ..Default::default()
    };

    let model = ImodModel {
        xmax: nx_image,
        ymax: ny_image,
        zmax: track_mods[0].zmax,
        objects: vec![obj],
        ..Default::default()
    };

    write_model(&args.output, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: pickbestseed - Writing model: {}", e);
        process::exit(1);
    });

    println!("Wrote seed model with {} points to {}", model.objects[0].contours.len(), args.output);
}
