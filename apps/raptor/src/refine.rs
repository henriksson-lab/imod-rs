//! Position refinement and trajectory merging.
//!
//! After 3D reconstruction, bead positions can be refined by re-matching the
//! template at the projected locations, and duplicate trajectories can be merged.

use crate::detect::detect_beads;
use crate::optimize::Bead3D;
use crate::trajectory::Trajectory;

/// Refine bead positions by re-matching the template near projected locations.
///
/// For each bead, projects its 3D position into each frame, then searches
/// a local window around the projected position for a better template match.
/// Updates the trajectory positions with the refined detections.
pub fn refine_positions(
    beads: &[Bead3D],
    trajectories: &mut Vec<Trajectory>,
    frames: &[Vec<f32>],
    nx: usize,
    ny: usize,
    template: &[f32],
    tsize: usize,
    tilt_angles: &[f32],
    search_radius: usize,
) {
    let angles_rad: Vec<f32> = tilt_angles.iter().map(|a| a.to_radians()).collect();
    let n_frames = frames.len().min(angles_rad.len());

    for (bi, traj) in trajectories.iter_mut().enumerate() {
        if bi >= beads.len() {
            continue;
        }
        let bead = &beads[bi];

        // Ensure positions vector is long enough
        while traj.positions.len() < n_frames {
            traj.positions.push(None);
        }

        for frame in 0..n_frames {
            let ct = angles_rad[frame].cos();
            let st = angles_rad[frame].sin();
            let proj_x = bead.x * ct + bead.z * st;
            let proj_y = bead.y;

            // Extract a local subimage around the projected position
            let cx = proj_x.round() as isize;
            let cy = proj_y.round() as isize;
            let half = search_radius as isize + tsize as isize / 2;

            let x0 = (cx - half).max(0) as usize;
            let y0 = (cy - half).max(0) as usize;
            let x1 = ((cx + half + 1) as usize).min(nx);
            let y1 = ((cy + half + 1) as usize).min(ny);

            if x1 <= x0 + tsize || y1 <= y0 + tsize {
                // Window too small for template matching
                continue;
            }

            let sub_nx = x1 - x0;
            let sub_ny = y1 - y0;

            // Extract sub-image
            let img = &frames[frame];
            let mut sub = vec![0.0f32; sub_nx * sub_ny];
            for sy in 0..sub_ny {
                for sx in 0..sub_nx {
                    let ix = x0 + sx;
                    let iy = y0 + sy;
                    if ix < nx && iy < ny {
                        sub[sy * sub_nx + sx] = img[iy * nx + ix];
                    }
                }
            }

            // Detect in the sub-image with a low threshold
            let dets = detect_beads(&sub, sub_nx, sub_ny, template, tsize, 0.1);

            if let Some(best) = dets.first() {
                // Convert back to global coordinates
                let gx = best.x + x0 as f32;
                let gy = best.y + y0 as f32;

                // Only update if the refined position is within search_radius
                let dx = gx - proj_x;
                let dy = gy - proj_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= search_radius as f32 {
                    traj.positions[frame] = Some((gx, gy));
                }
            }
        }
    }
}

/// Merge trajectories that track the same bead (very close in most frames).
///
/// Two trajectories are merged if the mean distance between their shared
/// observations is less than `max_dist`.  The trajectory with more observations
/// absorbs the other's positions for frames it was missing.
///
/// Returns the number of trajectories removed.
pub fn merge_similar(trajectories: &mut Vec<Trajectory>, max_dist: f32) -> usize {
    let max_dist_sq = max_dist * max_dist;
    let n = trajectories.len();
    let mut merged_into = vec![false; n];
    let mut merge_count = 0usize;

    for i in 0..n {
        if merged_into[i] {
            continue;
        }
        for j in (i + 1)..n {
            if merged_into[j] {
                continue;
            }

            // Compute mean distance across shared frames
            let mut sum_dist_sq = 0.0f32;
            let mut shared = 0usize;

            let len = trajectories[i].positions.len().min(trajectories[j].positions.len());
            for f in 0..len {
                if let (Some((x1, y1)), Some((x2, y2))) =
                    (&trajectories[i].positions[f], &trajectories[j].positions[f])
                {
                    let dx = x1 - x2;
                    let dy = y1 - y2;
                    sum_dist_sq += dx * dx + dy * dy;
                    shared += 1;
                }
            }

            if shared < 2 {
                continue;
            }

            let mean_dist_sq = sum_dist_sq / shared as f32;
            if mean_dist_sq <= max_dist_sq {
                // Merge j into i: fill in missing frames from j
                let positions_j: Vec<Option<(f32, f32)>> =
                    trajectories[j].positions.clone();
                let traj_i = &mut trajectories[i];

                while traj_i.positions.len() < positions_j.len() {
                    traj_i.positions.push(None);
                }

                for (f, pos_j) in positions_j.into_iter().enumerate() {
                    if traj_i.positions[f].is_none() {
                        traj_i.positions[f] = pos_j;
                    }
                }

                merged_into[j] = true;
                merge_count += 1;
            }
        }
    }

    // Remove merged trajectories
    let mut idx = 0;
    trajectories.retain(|_| {
        let keep = !merged_into[idx];
        idx += 1;
        keep
    });

    merge_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Trajectory;

    #[test]
    fn merge_duplicate_trajectories() {
        let t1 = Trajectory {
            positions: vec![Some((10.0, 20.0)), Some((11.0, 21.0)), None],
        };
        let t2 = Trajectory {
            positions: vec![Some((10.1, 20.1)), Some((11.1, 21.1)), Some((12.0, 22.0))],
        };
        let t3 = Trajectory {
            positions: vec![Some((100.0, 200.0)), Some((101.0, 201.0)), None],
        };

        let mut trajs = vec![t1, t2, t3];
        let removed = merge_similar(&mut trajs, 1.0);

        assert_eq!(removed, 1, "should merge one duplicate");
        assert_eq!(trajs.len(), 2, "should have 2 trajectories left");

        // The merged trajectory should have frame 2 filled in
        let merged = &trajs[0];
        assert!(merged.positions[2].is_some(), "frame 2 should be filled from merged traj");
    }

    #[test]
    fn no_merge_distant_trajectories() {
        let t1 = Trajectory {
            positions: vec![Some((10.0, 20.0)), Some((11.0, 21.0))],
        };
        let t2 = Trajectory {
            positions: vec![Some((100.0, 200.0)), Some((101.0, 201.0))],
        };

        let mut trajs = vec![t1, t2];
        let removed = merge_similar(&mut trajs, 1.0);

        assert_eq!(removed, 0);
        assert_eq!(trajs.len(), 2);
    }
}
