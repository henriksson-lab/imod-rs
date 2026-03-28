//! RAPTOR -- Rapid Automatic Particle Tracking On Regions.
//!
//! Automatic fiducial bead detection and tracking for cryo-ET tilt series.
//! Detects gold beads in each tilt image via template matching, then links
//! detections across frames using RANSAC-based correspondence and union-find
//! trajectory building.  3D positions are estimated and refined via bundle
//! adjustment.  Outputs an IMOD model file with one contour per tracked bead.

mod correspondence;
mod detect;
mod optimize;
mod refine;
mod trajectory;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{write_model, ImodContour, ImodModel, ImodObject};
use imod_mrc::MrcReader;
use rayon::prelude::*;

use correspondence::match_frames;
use detect::{create_template, detect_beads};
use optimize::{bundle_adjust, estimate_3d_positions, remove_outliers};
use refine::{merge_similar, refine_positions};
use trajectory::build_trajectories;

/// Automatic fiducial bead detection and tracking for tilt series.
#[derive(Parser)]
#[command(name = "raptor", about = "Automatic fiducial detection and tracking")]
struct Args {
    /// Input tilt series (MRC stack)
    #[arg(short = 'i', long)]
    input: String,

    /// Output fiducial model file (.fid)
    #[arg(short = 'o', long)]
    output: String,

    /// Bead diameter in pixels
    #[arg(short = 'd', long)]
    diameter: usize,

    /// Tilt angle file (one angle per line, in degrees)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Detection threshold (0..1, relative to best CC score)
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Minimum views per trajectory (0 = 60% of total frames)
    #[arg(long, default_value_t = 0)]
    min_views: usize,

    /// Maximum shift between consecutive frames (pixels)
    #[arg(long, default_value_t = 50.0)]
    max_shift: f32,

    /// RANSAC iterations for frame-to-frame matching
    #[arg(long, default_value_t = 1000)]
    ransac_iters: usize,

    /// RANSAC inlier distance threshold (pixels)
    #[arg(long, default_value_t = 8.0)]
    inlier_thresh: f32,

    /// Bundle adjustment iterations
    #[arg(long, default_value_t = 5)]
    bundle_iters: usize,

    /// Outlier removal percentile (e.g. 0.9 = remove top 10%)
    #[arg(long, default_value_t = 0.9)]
    outlier_percentile: f32,
}

/// Read tilt angles from a file (one angle per line, in degrees).
fn read_tilt_angles(path: &str) -> Vec<f32> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read tilt file {path}: {e}"));
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim()
                .parse::<f32>()
                .unwrap_or_else(|e| panic!("failed to parse tilt angle '{line}': {e}"))
        })
        .collect()
}

/// Find the index of the frame closest to zero tilt.
fn zero_tilt_index(tilt_angles: &[f32]) -> usize {
    tilt_angles
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() {
    let args = Args::parse();

    // 1. Read MRC stack and tilt angles
    let mut reader = MrcReader::open(&args.input).expect("failed to open input MRC file");
    let header = reader.header().clone();
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;

    let tilt_angles = read_tilt_angles(&args.tilt_file);
    if tilt_angles.len() != nz {
        panic!(
            "tilt file has {} angles but MRC has {} frames",
            tilt_angles.len(),
            nz
        );
    }

    eprintln!(
        "RAPTOR: input {}x{}x{} ({} frames), tilt range {:.1} to {:.1}",
        nx,
        ny,
        nz,
        nz,
        tilt_angles.iter().cloned().reduce(f32::min).unwrap_or(0.0),
        tilt_angles.iter().cloned().reduce(f32::max).unwrap_or(0.0),
    );

    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        let frame = reader
            .read_slice_f32(z)
            .unwrap_or_else(|e| panic!("failed to read frame {z}: {e}"));
        frames.push(frame);
    }

    // 2. Create template from diameter
    let template = create_template(args.diameter);
    eprintln!(
        "Template: {}x{} pixels (diameter={})",
        args.diameter, args.diameter, args.diameter
    );

    // 3. Detect beads in all frames (parallel)
    eprintln!("Detecting beads...");
    let all_detections: Vec<Vec<detect::Detection>> = frames
        .par_iter()
        .enumerate()
        .map(|(f, frame)| {
            let mut dets = detect_beads(frame, nx, ny, &template, args.diameter, args.threshold);
            for d in &mut dets {
                d.frame = f;
            }
            dets
        })
        .collect();

    for (f, dets) in all_detections.iter().enumerate() {
        eprintln!("  frame {f}: {} detections", dets.len());
    }

    // 4. Match beads between adjacent frames (bidirectional from zero-tilt)
    eprintln!("Matching frames...");
    let zero_idx = zero_tilt_index(&tilt_angles);
    eprintln!("  zero-tilt frame: {zero_idx} ({:.1} deg)", tilt_angles[zero_idx]);

    // Build pairs: from zero outward in both directions
    let mut frame_pairs: Vec<(usize, usize)> = Vec::new();

    // Forward from zero-tilt: zero->zero+1, zero+1->zero+2, ...
    for f in zero_idx..nz.saturating_sub(1) {
        frame_pairs.push((f, f + 1));
    }
    // Backward from zero-tilt: zero->zero-1, zero-1->zero-2, ...
    for f in (1..=zero_idx).rev() {
        frame_pairs.push((f, f - 1));
    }

    let all_matches: Vec<Vec<correspondence::Match>> = frame_pairs
        .par_iter()
        .map(|&(fa, fb)| {
            match_frames(
                &all_detections[fa],
                &all_detections[fb],
                args.max_shift,
                args.ransac_iters,
                args.inlier_thresh,
            )
        })
        .collect();

    // Build the match list indexed for build_trajectories.
    // build_trajectories expects all_matches[i] to be matches from frame i to frame i+1.
    // We need to reorganize our bidirectional matches into consecutive-frame format.
    let mut consecutive_matches: Vec<Vec<correspondence::Match>> =
        (0..nz.saturating_sub(1)).map(|_| Vec::new()).collect();

    for (pair_idx, &(fa, fb)) in frame_pairs.iter().enumerate() {
        if fb == fa + 1 {
            // Forward match: frame fa -> fa+1
            consecutive_matches[fa] = all_matches[pair_idx].clone();
        } else if fa == fb + 1 {
            // Backward match: frame fa -> fb (i.e., fa-1 <- fa)
            // Reverse the match direction: idx_a becomes idx_b and vice versa
            let reversed: Vec<correspondence::Match> = all_matches[pair_idx]
                .iter()
                .map(|m| correspondence::Match {
                    idx_a: m.idx_b,
                    idx_b: m.idx_a,
                })
                .collect();
            // This maps frame fb -> fb+1 = fa
            if fb < consecutive_matches.len() {
                if consecutive_matches[fb].is_empty() {
                    consecutive_matches[fb] = reversed;
                } else {
                    // Merge: add matches that don't conflict
                    let existing = &consecutive_matches[fb];
                    let used_a: std::collections::HashSet<usize> =
                        existing.iter().map(|m| m.idx_a).collect();
                    let used_b: std::collections::HashSet<usize> =
                        existing.iter().map(|m| m.idx_b).collect();
                    for m in &reversed {
                        if !used_a.contains(&m.idx_a) && !used_b.contains(&m.idx_b) {
                            consecutive_matches[fb].push(*m);
                        }
                    }
                }
            }
        }
    }

    for (f, m) in consecutive_matches.iter().enumerate() {
        eprintln!("  frames {f}->{}: {} matches", f + 1, m.len());
    }

    // 5. Build trajectories
    eprintln!("Building trajectories...");
    let min_views = if args.min_views == 0 {
        (nz as f32 * 0.6).ceil() as usize
    } else {
        args.min_views
    };

    let mut trajectories = build_trajectories(&consecutive_matches, &all_detections, nz, min_views);
    eprintln!(
        "Found {} trajectories (min_views={})",
        trajectories.len(),
        min_views
    );

    if trajectories.is_empty() {
        eprintln!("No trajectories found. Try lowering --threshold or --min-views.");
        std::process::exit(1);
    }

    // 6. Estimate 3D positions
    eprintln!("Estimating 3D positions...");
    let mut beads = estimate_3d_positions(&trajectories, &tilt_angles);
    let rms = optimize::bundle_adjust(&mut beads, &trajectories, &tilt_angles, 0);
    eprintln!("  initial RMS residual: {rms:.2} pixels");

    // 7. Bundle adjustment (2 rounds with outlier removal)
    for round in 0..2 {
        eprintln!("Bundle adjustment round {}...", round + 1);
        let rms = bundle_adjust(&mut beads, &trajectories, &tilt_angles, args.bundle_iters);
        eprintln!("  RMS after adjustment: {rms:.2} pixels");

        let removed = remove_outliers(
            &mut beads,
            &mut trajectories,
            &tilt_angles,
            args.outlier_percentile,
        );
        eprintln!(
            "  removed {removed} outliers, {} beads remaining",
            beads.len()
        );
    }

    // Final bundle adjustment
    let final_rms = bundle_adjust(&mut beads, &trajectories, &tilt_angles, args.bundle_iters);
    eprintln!("Final RMS residual: {final_rms:.2} pixels");

    // 8. Refine positions
    eprintln!("Refining positions...");
    refine_positions(
        &beads,
        &mut trajectories,
        &frames,
        nx,
        ny,
        &template,
        args.diameter,
        &tilt_angles,
        args.diameter, // search_radius = diameter
    );

    // Merge near-duplicate trajectories
    let merged = merge_similar(&mut trajectories, args.diameter as f32 * 0.5);
    if merged > 0 {
        eprintln!("  merged {merged} duplicate trajectories");
    }
    eprintln!("Final trajectory count: {}", trajectories.len());

    // 9. Write output .fid model
    let mut model = ImodModel::default();
    model.xmax = nx as i32;
    model.ymax = ny as i32;
    model.zmax = nz as i32;

    let mut obj = ImodObject::default();
    obj.name = "RAPTOR fiducials".to_string();

    for traj in &trajectories {
        let mut contour = ImodContour::default();
        for (frame, pos) in traj.positions.iter().enumerate() {
            if let Some((x, y)) = pos {
                contour.points.push(Point3f {
                    x: *x,
                    y: *y,
                    z: frame as f32,
                });
            }
        }
        if !contour.points.is_empty() {
            obj.contours.push(contour);
        }
    }

    model.objects.push(obj);

    write_model(&args.output, &model).expect("failed to write output model");
    eprintln!(
        "Wrote {} contours ({} points total) to {}",
        model.objects[0].contours.len(),
        model.objects[0]
            .contours
            .iter()
            .map(|c| c.points.len())
            .sum::<usize>(),
        args.output
    );
}
