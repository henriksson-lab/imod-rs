//! Translation of `IMOD/librgctf/brute_force_search.{h,cpp}`.

use crate::imod::librgctf::va04::va04a;

/// Exhaustive grid search with optional VA04 local minimization at every grid
/// point.  The C++ function pointer plus untyped parameter cursor is one owned
/// Rust closure here.
pub struct BruteForceSearch {
    number_of_dimensions: usize,
    target_function: Option<Box<dyn Fn(&[f32]) -> f32 + Send + Sync>>,
    starting_value: Vec<f32>,
    best_value: Vec<f32>,
    half_range: Vec<f32>,
    step_size: Vec<f32>,
    dimension_at_max: Vec<bool>,
    best_score: f32,
    num_iterations: usize,
    minimise_at_every_step: bool,
    print_progress_bar: bool,
}

impl Default for BruteForceSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl BruteForceSearch {
    /// C++ `BruteForceSearch::BruteForceSearch`.
    pub fn new() -> Self {
        Self {
            number_of_dimensions: 0,
            target_function: None,
            starting_value: Vec::new(),
            best_value: Vec::new(),
            half_range: Vec::new(),
            step_size: Vec::new(),
            dimension_at_max: Vec::new(),
            best_score: f32::MAX,
            num_iterations: 0,
            minimise_at_every_step: false,
            print_progress_bar: false,
        }
    }

    /// C++ `BruteForceSearch::Init` without its non-owning `void *` state.
    pub fn init<F>(
        &mut self,
        function_to_minimize: F,
        starting_value: &[f32],
        half_range: &[f32],
        step_size: &[f32],
        minimise_at_every_step: bool,
        print_progress_bar: bool,
    ) where
        F: Fn(&[f32]) -> f32 + Send + Sync + 'static,
    {
        assert!(
            self.target_function.is_none(),
            "Brute force search object is already setup"
        );
        assert!(!starting_value.is_empty(), "Bad number of dimensions");
        assert_eq!(starting_value.len(), half_range.len());
        assert_eq!(starting_value.len(), step_size.len());
        assert!(
            step_size.iter().all(|value| *value > 0.0),
            "Step sizes must be positive"
        );
        self.number_of_dimensions = starting_value.len();
        self.starting_value = starting_value.to_vec();
        self.best_value = vec![0.0; self.number_of_dimensions];
        self.half_range = half_range.to_vec();
        self.step_size = step_size.to_vec();
        self.dimension_at_max = vec![false; self.number_of_dimensions];
        self.minimise_at_every_step = minimise_at_every_step;
        self.print_progress_bar = print_progress_bar;
        let mut current_values = vec![-f32::MAX; self.number_of_dimensions];
        self.num_iterations = 0;
        loop {
            let completed = self.increment_current_values(&mut current_values);
            if completed {
                break;
            }
            self.num_iterations += 1;
        }
        self.dimension_at_max.fill(false);
        self.target_function = Some(Box::new(function_to_minimize));
    }

    /// C++ `BruteForceSearch::GetBestValue`.
    pub fn get_best_value(&self, index: usize) -> f32 {
        assert!(index < self.number_of_dimensions, "Index does not exist");
        self.best_value[index]
    }

    /// C++ inline `BruteForceSearch::GetBestScore`.
    pub fn get_best_score(&self) -> f32 {
        self.best_score
    }

    /// C++ `BruteForceSearch::IncrementCurrentValues`.
    ///
    /// Returns whether this increment has completed the exhaustive grid.
    pub fn increment_current_values(&mut self, current_values: &mut [f32]) -> bool {
        assert!(
            self.target_function.is_some() || self.number_of_dimensions > 0,
            "Brute force search object not allocated"
        );
        assert_eq!(current_values.len(), self.number_of_dimensions);
        let mut completed = true;
        for index in 0..self.number_of_dimensions {
            if !self.dimension_at_max[index] {
                completed = false;
                if current_values[index] < self.starting_value[index] - self.half_range[index] {
                    current_values[index] = self.starting_value[index] - self.half_range[index];
                } else {
                    current_values[index] += self.step_size[index];
                }
                for previous in 0..index {
                    current_values[previous] =
                        self.starting_value[previous] - self.half_range[previous];
                    assert!(self.dimension_at_max[previous], "failed sanity check");
                    self.dimension_at_max[previous] = false;
                }
                if current_values[index] >= self.starting_value[index] + self.half_range[index] {
                    current_values[index] = self.starting_value[index] + self.half_range[index];
                    self.dimension_at_max[index] = true;
                    if index == self.number_of_dimensions - 1 {
                        completed = true;
                    }
                }
                break;
            }
        }
        for index in 0..self.number_of_dimensions {
            if current_values[index] < self.starting_value[index] - self.half_range[index] {
                current_values[index] = self.starting_value[index] - self.half_range[index];
            }
        }
        completed
    }

    /// C++ `BruteForceSearch::Run`.
    pub fn run(&mut self) {
        self.best_score =
            self.target_function
                .as_ref()
                .expect("BruteForceSearch object not allocated")(&self.starting_value);
        self.best_value.clone_from(&self.starting_value);
        let mut current_values = vec![-f32::MAX; self.number_of_dimensions];
        self.dimension_at_max.fill(false);
        let mut all_values = Vec::with_capacity(self.number_of_dimensions * self.num_iterations);
        for _ in 0..self.num_iterations {
            assert!(
                !self.increment_current_values(&mut current_values),
                "Failed sanity check"
            );
            all_values.extend_from_slice(&current_values);
        }
        let accuracy = self
            .step_size
            .iter()
            .map(|value| value * 0.5)
            .collect::<Vec<_>>();
        let (all_scores, all_local_best_values) = {
            let target = self
                .target_function
                .as_ref()
                .expect("BruteForceSearch object not allocated");
            let mut all_scores = Vec::with_capacity(self.num_iterations);
            let mut all_local_best_values =
                Vec::with_capacity(self.number_of_dimensions * self.num_iterations);
            for values in all_values.chunks_exact(self.number_of_dimensions) {
                if self.minimise_at_every_step {
                    let mut locally_best = values.to_vec();
                    let mut score = target(&locally_best);
                    let mut calls = 0;
                    va04a(
                        self.number_of_dimensions,
                        &accuracy,
                        100.0,
                        &mut calls,
                        |candidate| target(candidate),
                        &mut score,
                        0,
                        1,
                        50,
                        &mut locally_best,
                    );
                    all_scores.push(score);
                    all_local_best_values.extend_from_slice(&locally_best);
                } else {
                    all_scores.push(target(values));
                }
            }
            (all_scores, all_local_best_values)
        };
        for (iteration, (&score, values)) in all_scores
            .iter()
            .zip(all_values.chunks_exact(self.number_of_dimensions))
            .enumerate()
        {
            if score < self.best_score {
                self.best_score = score;
                if self.minimise_at_every_step {
                    self.best_value.copy_from_slice(
                        &all_local_best_values[iteration * self.number_of_dimensions
                            ..(iteration + 1) * self.number_of_dimensions],
                    );
                } else {
                    self.best_value.copy_from_slice(values);
                }
            }
        }
        let _ = self.print_progress_bar;
    }
}

#[cfg(test)]
mod tests {
    use super::BruteForceSearch;

    #[test]
    fn exhaustive_search_visits_the_source_grid_order() {
        let mut search = BruteForceSearch::new();
        search.init(
            |value| (value[0] + 1.0).powi(2) + (value[1] + 1.0).powi(2),
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            false,
            false,
        );
        search.run();
        assert_eq!(search.get_best_score(), 0.0);
        assert_eq!(search.get_best_value(0), -1.0);
        assert_eq!(search.get_best_value(1), -1.0);
    }

    #[test]
    fn local_va04_pass_improves_a_grid_point() {
        let mut search = BruteForceSearch::new();
        search.init(
            |value| (value[0] - 0.3).powi(2),
            &[0.0],
            &[1.0],
            &[1.0],
            true,
            false,
        );
        search.run();
        assert!(
            search.get_best_score() < 1.0e-3,
            "{}",
            search.get_best_score()
        );
        assert!((search.get_best_value(0) - 0.3).abs() < 0.1);
    }
}
