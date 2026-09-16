//! Owned translation of `svlMarkerCorrespondenceLBModel.{h,cpp}`.
//!
//! The original model used TNT arrays, raw allocation, and a manually owned
//! inference pointer.  This version stores all tables in `Vec`s and gives the
//! SVL graph/inference ordinary Rust ownership.

use std::collections::BTreeSet;

use crate::imod::raptor::lasik::svl::lib::pgm::svl_cluster_graph::SvlClusterGraph;
use crate::imod::raptor::lasik::svl::lib::pgm::svl_factor::SvlFactor;
use crate::imod::raptor::lasik::svl::lib::pgm::svl_message_passing::{
    SvlMessagePassingAlgorithm, SvlMessagePassingInference,
};

const MIN_VALUE: f64 = 1.0e-10;

/// Safe replacement for the source `TNT::Array2D<double>`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GglMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
impl GglMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self, String> {
        let cols = rows.first().map_or(0, Vec::len);
        if cols == 0 || rows.iter().any(|row| row.len() != cols) {
            return Err("matrix rows have different widths".into());
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data: rows.into_iter().flatten().collect(),
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
}

/// Parsed source configuration stream (`ReadConfig`) with no unchecked input
/// extraction or uninitialized source fields.
#[derive(Clone, Debug)]
pub struct MarkerCorrespondenceConfig {
    pub marker_count: usize,
    pub singleton_choice: i32,
    pub pairwise_choice: i32,
    pub distance_intercept: f64,
    pub max_marker_pair_distance: f64,
    pub garbage_can_potential: f64,
    pub pairwise_scale: f64,
    pub minimum_pairwise_cliques: usize,
    pub proximity_threshold: f64,
    pub minimum_candidates: usize,
    pub distance_min: Vec<f64>,
    pub distance_max: Vec<f64>,
}
impl MarkerCorrespondenceConfig {
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut words = input.split_whitespace();
        let marker_count = words
            .next()
            .ok_or("missing marker count")?
            .parse::<usize>()
            .map_err(|_| "invalid marker count")?;
        let singleton_choice = words
            .next()
            .ok_or("missing singleton choice")?
            .parse()
            .map_err(|_| "invalid singleton choice")?;
        let pairwise_choice = words
            .next()
            .ok_or("missing pairwise choice")?
            .parse()
            .map_err(|_| "invalid pairwise choice")?;
        let distance_intercept = words
            .next()
            .ok_or("missing distance intercept")?
            .parse()
            .map_err(|_| "invalid distance intercept")?;
        let max_marker_pair_distance = words
            .next()
            .ok_or("missing max pair distance")?
            .parse()
            .map_err(|_| "invalid max pair distance")?;
        let garbage_can_potential = words
            .next()
            .ok_or("missing garbage potential")?
            .parse()
            .map_err(|_| "invalid garbage potential")?;
        let pairwise_scale = words
            .next()
            .ok_or("missing pairwise scale")?
            .parse()
            .map_err(|_| "invalid pairwise scale")?;
        let minimum_pairwise_cliques = words
            .next()
            .ok_or("missing min cliques")?
            .parse()
            .map_err(|_| "invalid min cliques")?;
        let proximity_threshold = words
            .next()
            .ok_or("missing proximity threshold")?
            .parse()
            .map_err(|_| "invalid proximity threshold")?;
        let minimum_candidates = words
            .next()
            .ok_or("missing min candidates")?
            .parse()
            .map_err(|_| "invalid min candidates")?;
        let mut distance_min = Vec::with_capacity(marker_count);
        let mut distance_max = Vec::with_capacity(marker_count);
        for _ in 0..marker_count {
            distance_min.push(
                words
                    .next()
                    .ok_or("missing minimum distance")?
                    .parse()
                    .map_err(|_| "invalid minimum distance")?,
            );
            distance_max.push(
                words
                    .next()
                    .ok_or("missing maximum distance")?
                    .parse()
                    .map_err(|_| "invalid maximum distance")?,
            );
        }
        Ok(Self {
            marker_count,
            singleton_choice,
            pairwise_choice,
            distance_intercept,
            max_marker_pair_distance,
            garbage_can_potential,
            pairwise_scale,
            minimum_pairwise_cliques,
            proximity_threshold,
            minimum_candidates,
            distance_min,
            distance_max,
        })
    }
}

/// Source `svlMarkerCorrespondenceLBModel` with the active cpp pathways.
#[derive(Clone, Debug)]
pub struct SvlMarkerCorrespondenceLbModel {
    marker_locations: GglMatrix,
    marker_locations2: GglMatrix,
    marker_candidates: GglMatrix,
    singleton_scores: GglMatrix,
    pub config: MarkerCorrespondenceConfig,
    /// Native `_allowVals` begins with its historical `-1` sentinel; model
    /// states start at index one and include the final garbage-can candidate.
    pub allowed_values: Vec<Vec<isize>>,
    pub pair_to_clique: Vec<Vec<isize>>,
    pub pair_distances: Vec<Vec<f64>>,
    pub graph: SvlClusterGraph,
    cards: Vec<usize>,
    locked_values: Vec<Option<usize>>,
    model_built: bool,
    max_product: bool,
    max_messages: usize,
}
impl SvlMarkerCorrespondenceLbModel {
    pub const USAGE: &'static str = "";
    pub fn new(
        marker_locations: GglMatrix,
        marker_locations2: GglMatrix,
        marker_candidates: GglMatrix,
        singleton_scores: GglMatrix,
        config: MarkerCorrespondenceConfig,
    ) -> Result<Self, String> {
        if marker_locations.rows != 2
            || marker_locations2.rows != 2
            || marker_candidates.rows != 2
            || marker_locations.cols != marker_locations2.cols
            || marker_locations.cols != config.marker_count
            || singleton_scores.rows != marker_locations.cols
            || singleton_scores.cols != marker_candidates.cols
        {
            return Err("marker/candidate matrix dimensions do not match configuration".into());
        }
        Ok(Self {
            marker_locations,
            marker_locations2,
            marker_candidates,
            singleton_scores,
            locked_values: vec![None; config.marker_count],
            config,
            allowed_values: Vec::new(),
            pair_to_clique: Vec::new(),
            pair_distances: Vec::new(),
            graph: SvlClusterGraph::new(),
            cards: Vec::new(),
            model_built: false,
            max_product: false,
            max_messages: 10_000,
        })
    }
    /// Source `ReadLockPots`; each valid `(marker, candidate)` pair replaces
    /// the prior lock, matching the stream loop.
    pub fn read_lock_pots(&mut self, input: &str) -> Result<(), String> {
        let values = input
            .split_whitespace()
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "invalid lock entry")?;
        if values.len() % 2 != 0 {
            return Err("lock entries must be marker/candidate pairs".into());
        }
        for pair in values.chunks_exact(2) {
            *self
                .locked_values
                .get_mut(pair[0])
                .ok_or("locked marker out of range")? = Some(pair[1]);
        }
        Ok(())
    }
    pub fn allowed_vals(&self, marker: usize) -> Option<&[isize]> {
        self.allowed_values.get(marker).map(Vec::as_slice)
    }
    pub fn clear(&mut self) {
        self.cards.clear();
        self.graph = SvlClusterGraph::new();
        self.model_built = false;
    }
    pub fn dist_l2(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        (x2 - x1).hypot(y2 - y1)
    }
    pub fn dist_l1(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        (x2 - x1).abs() + (y2 - y1).abs()
    }
    pub fn norm_dot(
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
        x4: f64,
        y4: f64,
        max_a: f64,
        min_a: f64,
        max_b: f64,
        min_b: f64,
    ) -> f64 {
        let ax = x2 - x1;
        let ay = y2 - y1;
        let bx = x4 - x3;
        let by = y4 - y3;
        let na = ax.hypot(ay);
        let nb = bx.hypot(by);
        if na == 0.0 || nb == 0.0 || na >= max_a || na <= min_a || nb >= max_b || nb <= min_b {
            0.0
        } else {
            ((ax / na * bx / nb + ay / na * by / nb).abs()).powi(10)
        }
    }
    /// Source `ComputeAllowedValues` / `AllowValuesByInitialProximity`.
    pub fn compute_allowed_values(&mut self) {
        let candidates = self.marker_candidates.cols;
        let wanted = self.config.minimum_candidates.min(candidates);
        self.allowed_values = (0..self.config.marker_count)
            .map(|marker| {
                let mut assigned = vec![false; candidates];
                let mut values = vec![-1];
                let mut threshold = self.config.proximity_threshold;
                // `AllowValuesByInitialProximity` repeatedly scans candidates
                // in input order, expanding the threshold by 1.5.
                while values.len() < wanted {
                    for candidate in 0..candidates {
                        let distance = Self::dist_l2(
                            self.marker_locations.get(0, marker),
                            self.marker_locations.get(1, marker),
                            self.marker_candidates.get(0, candidate),
                            self.marker_candidates.get(1, candidate),
                        );
                        if !assigned[candidate] && distance < threshold {
                            assigned[candidate] = true;
                            values.push(candidate as isize);
                        }
                    }
                    threshold *= 1.5;
                }
                values.push(candidates as isize);
                values
            })
            .collect();
    }
    pub fn build_cards(&mut self) {
        self.cards = self
            .allowed_values
            .iter()
            .map(|values| values.len() - 1)
            .collect();
    }
    pub fn build_model(&mut self) -> Result<(), String> {
        if self.model_built {
            return Ok(());
        }
        self.compute_allowed_values();
        self.build_cards();
        self.graph = SvlClusterGraph::with_cards(self.cards.clone())?;
        self.build_singleton_potentials()?;
        self.build_pairwise_cliques();
        self.build_pairwise_potentials()?;
        self.graph.bethe_approx();
        self.model_built = true;
        Ok(())
    }
    pub fn build_singleton_potentials(&mut self) -> Result<(), String> {
        for marker in 0..self.config.marker_count {
            let mut factor = SvlFactor::with_variable(marker as i32, self.cards[marker])?;
            for (state, &candidate) in self.allowed_values[marker].iter().skip(1).enumerate() {
                let candidate = candidate as usize;
                let value = if let Some(locked) = self.locked_values[marker] {
                    if locked == candidate { 1.0 } else { MIN_VALUE }
                } else {
                    self.singleton_value(marker, candidate)
                };
                factor.data[state] = value.max(MIN_VALUE);
            }
            self.graph.add_factor_clique(factor);
        }
        Ok(())
    }
    fn singleton_value(&self, marker: usize, candidate: usize) -> f64 {
        let count = self.marker_candidates.cols;
        if candidate == count {
            return self.config.garbage_can_potential;
        }
        match self.config.singleton_choice {
            1 => 1.0,
            2 => (0..self.config.marker_count)
                .filter(|&other| other != marker)
                .map(|other| {
                    (0..count)
                        .filter(|&c| c != candidate)
                        .map(|c| {
                            Self::norm_dot(
                                self.marker_locations.get(0, marker),
                                self.marker_locations.get(1, marker),
                                self.marker_candidates.get(0, candidate),
                                self.marker_candidates.get(1, candidate),
                                self.marker_locations.get(0, other),
                                self.marker_locations.get(1, other),
                                self.marker_candidates.get(0, c),
                                self.marker_candidates.get(1, c),
                                self.config.distance_max[marker],
                                self.config.distance_min[marker],
                                self.config.distance_max[other],
                                self.config.distance_min[other],
                            )
                        })
                        .fold(0.0, f64::max)
                })
                .sum(),
            3 => self.singleton_scores.get(marker, candidate),
            4 => {
                if Self::dist_l2(
                    self.marker_locations.get(0, marker),
                    self.marker_locations.get(1, marker),
                    self.marker_candidates.get(0, candidate),
                    self.marker_candidates.get(1, candidate),
                ) < self.config.distance_max[marker]
                {
                    1.0
                } else {
                    MIN_VALUE
                }
            }
            5 => {
                self.singleton_scores.get(marker, candidate)
                    * (-Self::dist_l2(
                        self.marker_locations2.get(0, marker),
                        self.marker_locations2.get(1, marker),
                        self.marker_candidates.get(0, candidate),
                        self.marker_candidates.get(1, candidate),
                    )
                    .powi(2)
                        / self.config.distance_max[marker].powi(2))
                    .exp()
            }
            _ => MIN_VALUE,
        }
    }
    /// Source `BuildPairwiseCliques`, including its distance/minimum-neighbour selection.
    pub fn build_pairwise_cliques(&mut self) {
        let n = self.config.marker_count;
        self.pair_distances = vec![vec![0.0; n]; n];
        self.pair_to_clique = vec![vec![-1; n]; n];
        let mut minimum_range = vec![f64::INFINITY; n];
        for i in 0..n {
            let mut distances = Vec::new();
            for j in 0..n {
                let d = Self::dist_l2(
                    self.marker_locations.get(0, i),
                    self.marker_locations.get(1, i),
                    self.marker_locations.get(0, j),
                    self.marker_locations.get(1, j),
                );
                self.pair_distances[i][j] = d;
                if i != j {
                    distances.push(d);
                }
            }
            distances.sort_by(f64::total_cmp);
            minimum_range[i] = distances
                .get(self.config.minimum_pairwise_cliques)
                .copied()
                .unwrap_or(f64::INFINITY);
        }
        let mut clique = 1isize;
        for i in 0..n {
            for j in i + 1..n {
                if self.pair_distances[i][j] < self.config.max_marker_pair_distance
                    || self.pair_distances[i][j] <= minimum_range[i]
                {
                    self.pair_to_clique[i][j] = clique;
                    clique += 1;
                }
            }
        }
    }
    pub fn build_pairwise_potentials(&mut self) -> Result<(), String> {
        for first in 0..self.config.marker_count {
            for second in first + 1..self.config.marker_count {
                if self.pair_to_clique[first][second] < 0 {
                    continue;
                }
                let mut factor = SvlFactor::with_variables(
                    vec![first as i32, second as i32],
                    vec![self.cards[first], self.cards[second]],
                )?;
                for first_state in 0..self.cards[first] {
                    for second_state in 0..self.cards[second] {
                        let first_candidate = self.allowed_values[first][first_state + 1] as usize;
                        let second_candidate = self.allowed_values[second][second_state + 1] as usize;
                        let value =
                            self.pairwise_value(first, first_candidate, second, second_candidate);
                        let index = factor
                            .index_of(&[first_state, second_state])
                            .ok_or("invalid pairwise factor index")?;
                        factor.data[index] = value.max(MIN_VALUE) * self.config.pairwise_scale;
                    }
                }
                self.graph.add_factor_clique(factor);
            }
        }
        Ok(())
    }
    fn pairwise_value(
        &self,
        first: usize,
        first_candidate: usize,
        second: usize,
        second_candidate: usize,
    ) -> f64 {
        if let Some(value) = self.locked_values[first] {
            if value != first_candidate {
                return MIN_VALUE;
            }
        }
        if let Some(value) = self.locked_values[second] {
            if value != second_candidate {
                return MIN_VALUE;
            }
        }
        let candidates = self.marker_candidates.cols;
        if first_candidate == candidates || second_candidate == candidates {
            return self.config.garbage_can_potential;
        }
        if first_candidate == second_candidate {
            return MIN_VALUE;
        }
        let dot = Self::norm_dot(
            self.marker_locations.get(0, first),
            self.marker_locations.get(1, first),
            self.marker_candidates.get(0, first_candidate),
            self.marker_candidates.get(1, first_candidate),
            self.marker_locations.get(0, second),
            self.marker_locations.get(1, second),
            self.marker_candidates.get(0, second_candidate),
            self.marker_candidates.get(1, second_candidate),
            self.config.distance_max[first],
            self.config.distance_min[first],
            self.config.distance_max[second],
            self.config.distance_min[second],
        );
        match self.config.pairwise_choice {
            1 => (0..self.config.marker_count)
                .filter(|&other| other != first && other != second)
                .map(|other| {
                    (0..candidates)
                        .filter(|&candidate| {
                            candidate != first_candidate && candidate != second_candidate
                        })
                        .map(|candidate| {
                            dot * Self::norm_dot(
                                self.marker_locations.get(0, first),
                                self.marker_locations.get(1, first),
                                self.marker_candidates.get(0, first_candidate),
                                self.marker_candidates.get(1, first_candidate),
                                self.marker_locations.get(0, other),
                                self.marker_locations.get(1, other),
                                self.marker_candidates.get(0, candidate),
                                self.marker_candidates.get(1, candidate),
                                self.config.distance_max[first],
                                self.config.distance_min[first],
                                self.config.distance_max[other],
                                self.config.distance_min[other],
                            )
                        })
                        .fold(0.0, f64::max)
                })
                .sum(),
            2 => dot,
            3 | 4 => {
                let difference = (self.pair_distances[first][second]
                    - Self::dist_l2(
                        self.marker_candidates.get(0, first_candidate),
                        self.marker_candidates.get(1, first_candidate),
                        self.marker_candidates.get(0, second_candidate),
                        self.marker_candidates.get(1, second_candidate),
                    ))
                .abs()
                    + 1e-9;
                let base = if difference <= self.config.distance_intercept {
                    1.0 - difference / self.config.distance_intercept
                } else {
                    MIN_VALUE
                };
                if self.config.pairwise_choice == 3 {
                    base * dot
                } else {
                    base
                }
            }
            5 => 1.0,
            6 | 7 | 8 => {
                let displacement = Self::dist_l2(
                    self.marker_locations2.get(0, first)
                        - self.marker_candidates.get(0, first_candidate),
                    self.marker_locations2.get(1, first)
                        - self.marker_candidates.get(1, first_candidate),
                    self.marker_locations2.get(0, second)
                        - self.marker_candidates.get(0, second_candidate),
                    self.marker_locations2.get(1, second)
                        - self.marker_candidates.get(1, second_candidate),
                );
                let mut base = (-(displacement / self.config.distance_intercept).powi(2)).exp();
                if self.config.pairwise_choice == 6 {
                    base *= dot;
                }
                if self.config.pairwise_choice == 8 {
                    for (marker, candidate) in
                        [(first, first_candidate), (second, second_candidate)]
                    {
                        base *= (-Self::dist_l2(
                            self.marker_locations2.get(0, marker),
                            self.marker_locations2.get(1, marker),
                            self.marker_candidates.get(0, candidate),
                            self.marker_candidates.get(1, candidate),
                        )
                        .powi(2)
                            / self.config.distance_max[marker].powi(2))
                        .exp();
                    }
                }
                base
            }
            _ => MIN_VALUE,
        }
    }
    /// Source `GetFinalMarginalBeliefs`; columns retain candidate indices and
    /// use `-1e12` for unallowed candidates exactly as the TNT output does.
    pub fn final_marginal_beliefs(&mut self) -> Result<GglMatrix, String> {
        self.build_model()?;
        let algorithm = if self.max_product {
            SvlMessagePassingAlgorithm::RbpMaxProd
        } else {
            SvlMessagePassingAlgorithm::RbpSumProd
        };
        let mut inference = SvlMessagePassingInference::new(self.graph.clone());
        inference.inference(algorithm, self.max_messages)?;
        let mut result = GglMatrix::new(self.config.marker_count, self.marker_candidates.cols + 1);
        result.data.fill(-1e12);
        for marker in 0..self.config.marker_count {
            let belief = inference
                .clique_potentials
                .iter()
                .find(|factor| factor.variables == vec![marker as i32])
                .ok_or("missing singleton belief")?;
            for (state, &candidate) in self.allowed_values[marker].iter().skip(1).enumerate() {
                result.set(marker, candidate as usize, belief.data[state].ln());
            }
        }
        Ok(result)
    }
    /// Source debug `GetInitialBeliefs` (including its historical [1][1] write
    /// when the input is large enough).
    pub fn initial_beliefs(&self) -> GglMatrix {
        let mut result = GglMatrix::new(self.config.marker_count, self.marker_candidates.cols);
        if result.rows > 1 && result.cols > 1 {
            result.set(1, 1, 3.0);
        }
        result
    }
    pub fn is_valid_param(value: (i32, i32)) -> bool {
        value.0 != -1 && value.1 != -1
    }
    pub fn allowed_value_set(&self, marker: usize) -> Option<BTreeSet<isize>> {
        self.allowed_values
            .get(marker)
            .map(|values| values.iter().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> MarkerCorrespondenceConfig {
        MarkerCorrespondenceConfig::parse("2 3 5 10 100 0.01 1 1 10 2 0 100 0 100").unwrap()
    }
    fn model() -> SvlMarkerCorrespondenceLbModel {
        SvlMarkerCorrespondenceLbModel::new(
            GglMatrix::from_rows(vec![vec![0., 10.], vec![0., 0.]]).unwrap(),
            GglMatrix::from_rows(vec![vec![0., 10.], vec![0., 0.]]).unwrap(),
            GglMatrix::from_rows(vec![vec![0., 10.], vec![0., 0.]]).unwrap(),
            GglMatrix::from_rows(vec![vec![2., 1.], vec![1., 2.]]).unwrap(),
            config(),
        )
        .unwrap()
    }
    #[test]
    fn config_locks_and_allowed_values_follow_source_layout() {
        let mut model = model();
        model.read_lock_pots("0 1").unwrap();
        model.compute_allowed_values();
        assert_eq!(model.allowed_vals(0).unwrap(), &[-1, 0, 2]);
        model.build_model().unwrap();
        assert_eq!(model.graph.num_cliques(), 3);
    }
    #[test]
    fn distance_and_pairwise_uniform_potentials_are_owned() {
        let mut model = model();
        model.build_model().unwrap();
        let pair = model.graph.initial_potentials.last().unwrap();
        assert!(pair.data.iter().all(|value| *value >= MIN_VALUE));
        assert_eq!(
            SvlMarkerCorrespondenceLbModel::dist_l1(1., 2., 4., -2.),
            7.0
        );
    }
}
