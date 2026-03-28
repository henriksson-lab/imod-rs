//! Frame-to-frame bead matching using RANSAC-based translation estimation.

use crate::detect::Detection;

/// A match between beads in two frames.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub idx_a: usize,
    pub idx_b: usize,
}

/// Simple pseudo-random number generator (xorshift32) to avoid pulling in
/// a full RNG crate.  Not cryptographic, but fine for RANSAC sampling.
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn usize_below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

/// Find corresponding beads between two frames using RANSAC.
///
/// The motion model is a simple 2D translation.  The algorithm:
/// 1. Build candidate pairs: for each detection in A, find detections in B
///    within `max_shift` pixels.
/// 2. RANSAC: randomly pick one candidate pair, compute translation, count
///    inliers, keep the best model.
/// 3. Return all inlier matches for the best model.
pub fn match_frames(
    dets_a: &[Detection],
    dets_b: &[Detection],
    max_shift: f32,
    ransac_iters: usize,
    inlier_thresh: f32,
) -> Vec<Match> {
    if dets_a.is_empty() || dets_b.is_empty() {
        return Vec::new();
    }

    let max_shift_sq = max_shift * max_shift;
    let inlier_thresh_sq = inlier_thresh * inlier_thresh;

    // Build candidate pairs within max_shift
    let mut candidates: Vec<(usize, usize, f32, f32)> = Vec::new(); // (idx_a, idx_b, dx, dy)
    for (ia, a) in dets_a.iter().enumerate() {
        for (ib, b) in dets_b.iter().enumerate() {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            if dx * dx + dy * dy <= max_shift_sq {
                candidates.push((ia, ib, dx, dy));
            }
        }
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    let mut rng = Rng::new(42);
    let mut best_inliers: Vec<Match> = Vec::new();

    for _ in 0..ransac_iters {
        // Pick a random candidate pair to define the translation
        let ci = rng.usize_below(candidates.len());
        let (_, _, tx, ty) = candidates[ci];

        // Count inliers: for each detection in A, find the closest detection
        // in B after applying the translation, and check if within threshold.
        let mut inliers = Vec::new();
        let mut used_b = vec![false; dets_b.len()];

        // Greedy assignment: process A detections in order, assign best
        // available B detection.
        let mut assignments: Vec<(usize, usize, f32)> = Vec::new();
        for (ia, a) in dets_a.iter().enumerate() {
            let pred_x = a.x + tx;
            let pred_y = a.y + ty;

            let mut best_dist_sq = f32::MAX;
            let mut best_ib = 0;
            for (ib, b) in dets_b.iter().enumerate() {
                let dx = b.x - pred_x;
                let dy = b.y - pred_y;
                let d2 = dx * dx + dy * dy;
                if d2 < best_dist_sq {
                    best_dist_sq = d2;
                    best_ib = ib;
                }
            }

            if best_dist_sq <= inlier_thresh_sq {
                assignments.push((ia, best_ib, best_dist_sq));
            }
        }

        // Sort by distance so better matches get priority
        assignments.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (ia, ib, _) in &assignments {
            if !used_b[*ib] {
                used_b[*ib] = true;
                inliers.push(Match {
                    idx_a: *ia,
                    idx_b: *ib,
                });
            }
        }

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    best_inliers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Detection;

    #[test]
    fn match_translated_beads() {
        // Create two sets of detections with a known translation
        let dets_a: Vec<Detection> = vec![
            Detection { x: 10.0, y: 20.0, score: 1.0, frame: 0 },
            Detection { x: 50.0, y: 30.0, score: 1.0, frame: 0 },
            Detection { x: 80.0, y: 70.0, score: 1.0, frame: 0 },
        ];

        let tx = 3.0f32;
        let ty = -2.0f32;
        let dets_b: Vec<Detection> = dets_a
            .iter()
            .map(|d| Detection {
                x: d.x + tx,
                y: d.y + ty,
                score: 1.0,
                frame: 1,
            })
            .collect();

        let matches = match_frames(&dets_a, &dets_b, 20.0, 200, 5.0);
        assert_eq!(matches.len(), 3, "all three beads should match");
    }

    #[test]
    fn no_match_beyond_max_shift() {
        let dets_a = vec![Detection { x: 10.0, y: 10.0, score: 1.0, frame: 0 }];
        let dets_b = vec![Detection { x: 100.0, y: 100.0, score: 1.0, frame: 1 }];

        let matches = match_frames(&dets_a, &dets_b, 20.0, 100, 5.0);
        assert!(matches.is_empty());
    }
}
