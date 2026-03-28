//! 3D bead position estimation and bundle adjustment.
//!
//! Uses the parallel projection model for cryo-ET:
//!   proj_x = X * cos(tilt) + Z * sin(tilt)
//!   proj_y = Y

use crate::trajectory::Trajectory;
use imod_math::gaussj;

/// 3D position of a bead.
#[derive(Debug, Clone)]
pub struct Bead3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Solve for 3D bead positions from 2D detections and tilt angles.
///
/// For each trajectory, collects all (proj_x, proj_y, tilt_angle) observations
/// and solves the overdetermined system via least-squares:
///   proj_x_i = X * cos(theta_i) + Z * sin(theta_i)
///   proj_y_i = Y
///
/// The X,Z system is solved via normal equations (A^T A)^-1 A^T b.
/// Y is simply the mean of all observed y coordinates.
pub fn estimate_3d_positions(
    trajectories: &[Trajectory],
    tilt_angles: &[f32], // degrees
) -> Vec<Bead3D> {
    let n_frames = tilt_angles.len();
    let angles_rad: Vec<f32> = tilt_angles.iter().map(|a| a.to_radians()).collect();

    let mut beads = Vec::with_capacity(trajectories.len());

    for traj in trajectories {
        // Collect observations
        let mut obs_x: Vec<(f32, f32)> = Vec::new(); // (cos_theta, sin_theta) and proj_x
        let mut obs_px: Vec<f32> = Vec::new();
        let mut sum_y = 0.0f32;
        let mut n_obs = 0usize;

        for (frame, pos) in traj.positions.iter().enumerate() {
            if frame >= n_frames {
                break;
            }
            if let Some((px, py)) = pos {
                let ct = angles_rad[frame].cos();
                let st = angles_rad[frame].sin();
                obs_x.push((ct, st));
                obs_px.push(*px);
                sum_y += py;
                n_obs += 1;
            }
        }

        if n_obs < 2 {
            // Not enough observations; use first observation as fallback
            if let Some((_frame, Some((px, py)))) = traj
                .positions
                .iter()
                .enumerate()
                .find(|(_, p)| p.is_some())
            {
                beads.push(Bead3D {
                    x: *px,
                    y: *py,
                    z: 0.0,
                });
            } else {
                beads.push(Bead3D {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });
            }
            continue;
        }

        let y_3d = sum_y / n_obs as f32;

        // Solve for X and Z via normal equations:
        // A = [[cos0, sin0], [cos1, sin1], ...]
        // b = [px0, px1, ...]
        // A^T A is 2x2, A^T b is 2x1
        let mut ata = [0.0f32; 4]; // 2x2 row-major
        let mut atb = [0.0f32; 2]; // 2x1

        for (i, &(ct, st)) in obs_x.iter().enumerate() {
            ata[0] += ct * ct; // [0,0]
            ata[1] += ct * st; // [0,1]
            ata[2] += st * ct; // [1,0]
            ata[3] += st * st; // [1,1]
            atb[0] += ct * obs_px[i];
            atb[1] += st * obs_px[i];
        }

        // Solve (A^T A) * [X, Z]^T = A^T b using gaussj
        let result = gaussj(&mut ata, 2, 2, &mut atb, 1, 1);

        if result.is_ok() {
            beads.push(Bead3D {
                x: atb[0],
                y: y_3d,
                z: atb[1],
            });
        } else {
            // Singular system: fallback to mean x
            let mean_x = obs_px.iter().sum::<f32>() / n_obs as f32;
            beads.push(Bead3D {
                x: mean_x,
                y: y_3d,
                z: 0.0,
            });
        }
    }

    beads
}

/// Project a 3D bead to 2D at a given tilt angle (radians).
#[inline]
fn project(bead: &Bead3D, cos_tilt: f32, sin_tilt: f32) -> (f32, f32) {
    let px = bead.x * cos_tilt + bead.z * sin_tilt;
    let py = bead.y;
    (px, py)
}

/// Compute the RMS reprojection error across all beads and observations.
fn compute_rms(
    beads: &[Bead3D],
    trajectories: &[Trajectory],
    angles_rad: &[f32],
) -> f32 {
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for (bi, traj) in trajectories.iter().enumerate() {
        if bi >= beads.len() {
            break;
        }
        let bead = &beads[bi];
        for (frame, pos) in traj.positions.iter().enumerate() {
            if frame >= angles_rad.len() {
                break;
            }
            if let Some((obs_x, obs_y)) = pos {
                let ct = angles_rad[frame].cos();
                let st = angles_rad[frame].sin();
                let (px, py) = project(bead, ct, st);
                let dx = px - obs_x;
                let dy = py - obs_y;
                sum_sq += (dx * dx + dy * dy) as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt() as f32
}

/// Bundle adjustment: jointly optimize bead positions.
///
/// Minimizes reprojection error using iterative Gauss-Newton.
/// Each bead has 3 parameters (X, Y, Z) optimized independently since
/// the parallel projection model decouples beads.
///
/// Returns the final RMS residual.
pub fn bundle_adjust(
    beads: &mut Vec<Bead3D>,
    trajectories: &[Trajectory],
    tilt_angles: &[f32],
    iterations: usize,
) -> f32 {
    let angles_rad: Vec<f32> = tilt_angles.iter().map(|a| a.to_radians()).collect();

    for _iter in 0..iterations {
        for (bi, traj) in trajectories.iter().enumerate() {
            if bi >= beads.len() {
                continue;
            }

            // Collect observations for this bead
            let mut obs: Vec<(usize, f32, f32)> = Vec::new(); // (frame, obs_x, obs_y)
            for (frame, pos) in traj.positions.iter().enumerate() {
                if frame >= angles_rad.len() {
                    break;
                }
                if let Some((ox, oy)) = pos {
                    obs.push((frame, *ox, *oy));
                }
            }

            if obs.len() < 2 {
                continue;
            }

            // Build Jacobian (2*n_obs x 3) and residual vector (2*n_obs x 1)
            // For observation i:
            //   r_x = obs_x - (X*cos + Z*sin)  =>  dr/dX = -cos, dr/dY = 0, dr/dZ = -sin
            //   r_y = obs_y - Y                 =>  dr/dX = 0,    dr/dY = -1, dr/dZ = 0
            // J^T J is 3x3, J^T r is 3x1
            let mut jtj = [0.0f32; 9]; // 3x3
            let mut jtr = [0.0f32; 3]; // 3x1

            let bead = &beads[bi];
            for &(frame, ox, oy) in &obs {
                let ct = angles_rad[frame].cos();
                let st = angles_rad[frame].sin();
                let (px, py) = project(bead, ct, st);

                let rx = ox - px;
                let ry = oy - py;

                // Jacobian row for x-residual: [-cos, 0, -sin]
                let jx = [-ct, 0.0, -st];
                // Jacobian row for y-residual: [0, -1, 0]
                let jy = [0.0, -1.0, 0.0];

                // Accumulate J^T J and J^T r
                for a in 0..3 {
                    jtr[a] += jx[a] * rx + jy[a] * ry;
                    for b in 0..3 {
                        jtj[a * 3 + b] += jx[a] * jx[b] + jy[a] * jy[b];
                    }
                }
            }

            // Solve (J^T J) * delta = J^T r
            // Note: gaussj solves A*x = b, replacing b with x
            let mut jtj_copy = jtj;
            let mut jtr_copy = jtr;
            if gaussj(&mut jtj_copy, 3, 3, &mut jtr_copy, 1, 1).is_ok() {
                // delta = jtr_copy (solution replaces b)
                // Update: bead -= delta (since residual = obs - proj, and J = -d(proj)/d(params))
                // Actually J^T r with J = d(residual)/d(params) = -d(proj)/d(params)
                // So delta = (J^T J)^-1 J^T r, and we want params += delta
                // But our J rows are -d(proj)/d(params), so:
                //   J^T r = sum(-d(proj)/d(params) * (obs - proj))
                // Normal equations give: delta such that params_new = params - delta
                // Wait: let's be careful.
                // residual r = obs - proj(params)
                // J = dr/dparams = -d(proj)/d(params)
                // Gauss-Newton: delta = (J^T J)^-1 J^T r
                // params_new = params + delta (since linearization: r ≈ r0 + J*delta)
                // minimizing ||r0 + J*delta||^2 gives delta = -(J^T J)^-1 J^T r0... no.
                // Standard: min ||r + J*dp||^2 => dp = -(J^T J)^-1 J^T r
                // So update is params += dp = -(J^T J)^-1 J^T r
                // We solved (J^T J)*x = J^T r, so x = (J^T J)^-1 J^T r
                // Therefore dp = -x
                beads[bi].x -= jtr_copy[0];
                beads[bi].y -= jtr_copy[1];
                beads[bi].z -= jtr_copy[2];
            }
        }
    }

    compute_rms(beads, trajectories, &angles_rad)
}

/// Remove outlier beads based on per-bead residual.
///
/// Computes the RMS residual for each bead, then removes beads (and
/// corresponding trajectories) whose residual exceeds the given percentile
/// of the residual distribution.
///
/// Returns the number of beads removed.
pub fn remove_outliers(
    beads: &mut Vec<Bead3D>,
    trajectories: &mut Vec<Trajectory>,
    tilt_angles: &[f32],
    percentile: f32, // e.g. 0.9 = remove top 10%
) -> usize {
    if beads.is_empty() {
        return 0;
    }

    let angles_rad: Vec<f32> = tilt_angles.iter().map(|a| a.to_radians()).collect();

    // Compute per-bead RMS residual
    let mut residuals: Vec<f32> = Vec::with_capacity(beads.len());
    for (bi, traj) in trajectories.iter().enumerate() {
        if bi >= beads.len() {
            residuals.push(0.0);
            continue;
        }
        let bead = &beads[bi];
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;

        for (frame, pos) in traj.positions.iter().enumerate() {
            if frame >= angles_rad.len() {
                break;
            }
            if let Some((ox, oy)) = pos {
                let ct = angles_rad[frame].cos();
                let st = angles_rad[frame].sin();
                let (px, py) = project(bead, ct, st);
                let dx = px - ox;
                let dy = py - oy;
                sum_sq += (dx * dx + dy * dy) as f64;
                count += 1;
            }
        }

        let rms = if count > 0 {
            (sum_sq / count as f64).sqrt() as f32
        } else {
            0.0
        };
        residuals.push(rms);
    }

    // Find the percentile threshold
    let mut sorted = residuals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((percentile * (sorted.len() - 1) as f32) as usize).min(sorted.len() - 1);
    let threshold = sorted[idx];

    // Remove beads above threshold
    let mut removed = 0usize;
    let mut i = 0;
    while i < beads.len() {
        if residuals[i] > threshold && threshold > 0.0 {
            beads.remove(i);
            trajectories.remove(i);
            residuals.remove(i);
            removed += 1;
        } else {
            i += 1;
        }
    }

    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Trajectory;

    /// Helper: create a trajectory from known 3D position and tilt angles.
    fn make_trajectory(bead: &Bead3D, tilt_angles: &[f32]) -> Trajectory {
        let angles_rad: Vec<f32> = tilt_angles.iter().map(|a| a.to_radians()).collect();
        let positions: Vec<Option<(f32, f32)>> = angles_rad
            .iter()
            .map(|theta| {
                let px = bead.x * theta.cos() + bead.z * theta.sin();
                let py = bead.y;
                Some((px, py))
            })
            .collect();
        Trajectory { positions }
    }

    #[test]
    fn estimate_single_bead() {
        let true_bead = Bead3D {
            x: 100.0,
            y: 200.0,
            z: 50.0,
        };
        let tilts: Vec<f32> = (-60..=60).step_by(3).map(|a| a as f32).collect();
        let traj = make_trajectory(&true_bead, &tilts);

        let beads = estimate_3d_positions(&[traj], &tilts);
        assert_eq!(beads.len(), 1);

        let b = &beads[0];
        assert!(
            (b.x - true_bead.x).abs() < 0.1,
            "X: expected {}, got {}",
            true_bead.x,
            b.x
        );
        assert!(
            (b.y - true_bead.y).abs() < 0.1,
            "Y: expected {}, got {}",
            true_bead.y,
            b.y
        );
        assert!(
            (b.z - true_bead.z).abs() < 0.1,
            "Z: expected {}, got {}",
            true_bead.z,
            b.z
        );
    }

    #[test]
    fn bundle_adjust_converges() {
        let true_bead = Bead3D {
            x: 100.0,
            y: 200.0,
            z: 50.0,
        };
        let tilts: Vec<f32> = (-60..=60).step_by(3).map(|a| a as f32).collect();
        let traj = make_trajectory(&true_bead, &tilts);

        // Start from a perturbed position
        let mut beads = vec![Bead3D {
            x: 110.0,
            y: 205.0,
            z: 40.0,
        }];

        let rms = bundle_adjust(&mut beads, &[traj], &tilts, 10);
        assert!(rms < 1.0, "RMS should converge, got {rms}");

        let b = &beads[0];
        assert!(
            (b.x - true_bead.x).abs() < 1.0,
            "X: expected {}, got {}",
            true_bead.x,
            b.x
        );
    }
}
