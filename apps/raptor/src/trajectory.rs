//! Build bead trajectories from pairwise frame-to-frame matches.

use crate::correspondence::Match;
use crate::detect::Detection;

/// A trajectory: one bead tracked across multiple frames.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Indexed by frame number.  `None` if the bead was not tracked in that frame.
    pub positions: Vec<Option<(f32, f32)>>,
}

/// Union-Find (disjoint set) data structure for merging trajectory chains.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path compression
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Build trajectories from pairwise matches.
///
/// Each detection across all frames is assigned a unique global index.
/// Matches between consecutive frames create edges that union-find merges
/// into connected components.  Each component becomes one trajectory.
///
/// `all_matches[i]` contains the matches between frame `i` and frame `i+1`.
/// `all_detections[i]` contains the detections in frame `i`.
///
/// Trajectories shorter than `min_length` are discarded.
pub fn build_trajectories(
    all_matches: &[Vec<Match>],
    all_detections: &[Vec<Detection>],
    n_frames: usize,
    min_length: usize,
) -> Vec<Trajectory> {
    if n_frames == 0 || all_detections.is_empty() {
        return Vec::new();
    }

    // Compute global index offsets for each frame's detections
    let mut offsets = Vec::with_capacity(n_frames);
    let mut total = 0usize;
    for f in 0..n_frames {
        offsets.push(total);
        if f < all_detections.len() {
            total += all_detections[f].len();
        }
    }

    if total == 0 {
        return Vec::new();
    }

    // Build union-find over all detections
    let mut uf = UnionFind::new(total);

    // Union matched detections across consecutive frames
    for (frame_idx, matches) in all_matches.iter().enumerate() {
        if frame_idx + 1 >= n_frames {
            break;
        }
        let off_a = offsets[frame_idx];
        let off_b = offsets[frame_idx + 1];

        for m in matches {
            let ga = off_a + m.idx_a;
            let gb = off_b + m.idx_b;
            if ga < total && gb < total {
                uf.union(ga, gb);
            }
        }
    }

    // Group detections by their union-find root
    let mut components: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();

    for f in 0..n_frames {
        if f >= all_detections.len() {
            continue;
        }
        for (d_idx, _det) in all_detections[f].iter().enumerate() {
            let global = offsets[f] + d_idx;
            let root = uf.find(global);
            components.entry(root).or_default().push((f, d_idx));
        }
    }

    // Convert each component into a trajectory
    let mut trajectories = Vec::new();

    for (_root, members) in &components {
        // Count how many distinct frames this trajectory spans
        let n_present = members.len();
        if n_present < min_length {
            continue;
        }

        let mut positions = vec![None; n_frames];
        for &(frame, d_idx) in members {
            let det = &all_detections[frame][d_idx];
            positions[frame] = Some((det.x, det.y));
        }

        trajectories.push(Trajectory { positions });
    }

    // Sort trajectories by the frame of their first appearance, then by x position
    trajectories.sort_by(|a, b| {
        let first_a = a.positions.iter().position(|p| p.is_some()).unwrap_or(n_frames);
        let first_b = b.positions.iter().position(|p| p.is_some()).unwrap_or(n_frames);
        first_a.cmp(&first_b).then_with(|| {
            let xa = a.positions.iter().find_map(|p| p.map(|(x, _)| x)).unwrap_or(0.0);
            let xb = b.positions.iter().find_map(|p| p.map(|(x, _)| x)).unwrap_or(0.0);
            xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    trajectories
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence::Match;
    use crate::detect::Detection;

    #[test]
    fn build_simple_trajectory() {
        // 3 frames, 2 beads each, perfectly matched
        let dets: Vec<Vec<Detection>> = (0..3)
            .map(|f| {
                vec![
                    Detection { x: 10.0 + f as f32, y: 20.0, score: 1.0, frame: f },
                    Detection { x: 50.0 + f as f32, y: 60.0, score: 1.0, frame: f },
                ]
            })
            .collect();

        let matches: Vec<Vec<Match>> = vec![
            vec![Match { idx_a: 0, idx_b: 0 }, Match { idx_a: 1, idx_b: 1 }],
            vec![Match { idx_a: 0, idx_b: 0 }, Match { idx_a: 1, idx_b: 1 }],
        ];

        let trajs = build_trajectories(&matches, &dets, 3, 2);
        assert_eq!(trajs.len(), 2, "should produce 2 trajectories");

        for t in &trajs {
            let count = t.positions.iter().filter(|p| p.is_some()).count();
            assert_eq!(count, 3, "each trajectory should span all 3 frames");
        }
    }

    #[test]
    fn short_trajectories_filtered() {
        let dets: Vec<Vec<Detection>> = vec![
            vec![Detection { x: 10.0, y: 20.0, score: 1.0, frame: 0 }],
            vec![Detection { x: 11.0, y: 20.0, score: 1.0, frame: 1 }],
        ];
        let matches = vec![vec![Match { idx_a: 0, idx_b: 0 }]];

        // min_length=3 should filter out the 2-frame trajectory
        let trajs = build_trajectories(&matches, &dets, 2, 3);
        assert!(trajs.is_empty());

        // min_length=2 should keep it
        let trajs = build_trajectories(&matches, &dets, 2, 2);
        assert_eq!(trajs.len(), 1);
    }
}
