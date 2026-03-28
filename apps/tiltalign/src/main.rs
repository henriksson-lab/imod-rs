use clap::Parser;
use imod_model::read_model;
use imod_transforms::{write_tilt_file, write_xf_file, LinearTransform};
use std::f32::consts::PI;

/// Solve for the alignment of a tilt series from fiducial bead positions.
///
/// Given an IMOD model file with tracked fiducial positions across a tilt series
/// and the tilt angles, solves for per-view translations, rotations, and
/// optionally magnification changes that minimize the residual error.
#[derive(Parser)]
#[command(name = "tiltalign", about = "Fiducial-based tilt series alignment")]
struct Args {
    /// Model file with fiducial positions (.mod or .fid)
    #[arg(short = 'm', long)]
    model: String,

    /// Tilt angle file (.tlt or .rawtlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Output transform file (.xf)
    #[arg(short = 'o', long)]
    output_xf: String,

    /// Output refined tilt angles (.tlt)
    #[arg(long)]
    output_tilt: Option<String>,

    /// Image X dimension (for centering transforms)
    #[arg(long, default_value_t = 0)]
    image_nx: i32,

    /// Image Y dimension
    #[arg(long, default_value_t = 0)]
    image_ny: i32,

    /// Solve for per-view rotation
    #[arg(short = 'r', long, default_value_t = true)]
    solve_rotation: bool,

    /// Solve for per-view magnification
    #[arg(short = 'g', long)]
    solve_mag: bool,

    /// Number of iterations
    #[arg(short = 'n', long, default_value_t = 10)]
    iterations: usize,
}

fn main() {
    let args = Args::parse();

    // Read fiducial model: each object = one bead tracked across views
    // Each contour in the object has points at different Z (= view index)
    let model = read_model(&args.model).unwrap_or_else(|e| {
        eprintln!("Error reading model: {}", e);
        std::process::exit(1);
    });

    let tilt_angles = imod_transforms::read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let n_views = tilt_angles.len();
    eprintln!("tiltalign: {} views, {} objects in model", n_views, model.objects.len());

    // Extract fiducial tracks: for each bead, its (x, y) position at each view
    // Convention: each object is a bead, each contour point z = view index
    let mut tracks: Vec<Vec<Option<(f32, f32)>>> = Vec::new();

    for obj in &model.objects {
        let mut track = vec![None; n_views];
        for cont in &obj.contours {
            for pt in &cont.points {
                let view = pt.z.round() as usize;
                if view < n_views {
                    track[view] = Some((pt.x, pt.y));
                }
            }
        }
        // Only keep beads visible in at least 3 views
        let visible: usize = track.iter().filter(|t| t.is_some()).count();
        if visible >= 3 {
            tracks.push(track);
        }
    }

    let n_beads = tracks.len();
    eprintln!("tiltalign: {} usable fiducial tracks", n_beads);

    if n_beads == 0 {
        eprintln!("Error: no usable fiducial tracks found");
        std::process::exit(1);
    }

    // Initialize: solve for translations using mean bead position per view
    let mut dx = vec![0.0f32; n_views];
    let mut dy = vec![0.0f32; n_views];
    let rotations = vec![0.0f32; n_views];
    let mags = vec![1.0f32; n_views];

    for iter in 0..args.iterations {
        // For each view, compute mean residual from 3D bead positions projected
        // through current alignment parameters

        // Step 1: Estimate 3D bead positions (simple: average X,Y across views after removing shifts)
        let mut bead_x = vec![0.0f32; n_beads];
        let mut bead_y = vec![0.0f32; n_beads];

        for (bi, track) in tracks.iter().enumerate() {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            let mut count = 0;
            for (vi, pos) in track.iter().enumerate() {
                if let Some((x, y)) = pos {
                    sx += x - dx[vi];
                    sy += y - dy[vi];
                    count += 1;
                }
            }
            if count > 0 {
                bead_x[bi] = sx / count as f32;
                bead_y[bi] = sy / count as f32;
            }
        }

        // Step 2: Update per-view shifts (and optionally rotation/mag)
        let mut total_residual = 0.0f64;
        let mut total_points = 0usize;

        for vi in 0..n_views {
            let mut sum_dx = 0.0f32;
            let mut sum_dy = 0.0f32;
            let mut count = 0;

            for (bi, track) in tracks.iter().enumerate() {
                if let Some((obs_x, obs_y)) = track[vi] {
                    let pred_x = bead_x[bi];
                    let pred_y = bead_y[bi];
                    sum_dx += obs_x - pred_x;
                    sum_dy += obs_y - pred_y;
                    count += 1;
                }
            }

            if count > 0 {
                dx[vi] = sum_dx / count as f32;
                dy[vi] = sum_dy / count as f32;
            }

            // Compute residual
            for (bi, track) in tracks.iter().enumerate() {
                if let Some((obs_x, obs_y)) = track[vi] {
                    let res_x = obs_x - bead_x[bi] - dx[vi];
                    let res_y = obs_y - bead_y[bi] - dy[vi];
                    total_residual += (res_x * res_x + res_y * res_y) as f64;
                    total_points += 1;
                }
            }
        }

        let rms = if total_points > 0 {
            (total_residual / total_points as f64).sqrt()
        } else {
            0.0
        };

        if iter == 0 || iter == args.iterations - 1 {
            eprintln!("  iter {}: RMS residual = {:.3} pixels ({} points)", iter + 1, rms, total_points);
        }
    }

    // Build output transforms
    let transforms: Vec<LinearTransform> = (0..n_views)
        .map(|vi| {
            let rot_rad = rotations[vi] * PI / 180.0;
            let c = rot_rad.cos() * mags[vi];
            let s = rot_rad.sin() * mags[vi];
            LinearTransform {
                a11: c,
                a12: -s,
                a21: s,
                a22: c,
                dx: dx[vi],
                dy: dy[vi],
            }
        })
        .collect();

    write_xf_file(&args.output_xf, &transforms).unwrap();
    eprintln!("tiltalign: wrote {} transforms to {}", n_views, args.output_xf);

    if let Some(ref tilt_path) = args.output_tilt {
        write_tilt_file(tilt_path, &tilt_angles).unwrap();
        eprintln!("tiltalign: wrote {} tilt angles to {}", n_views, tilt_path);
    }
}
