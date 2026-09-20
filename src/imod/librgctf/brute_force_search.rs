//! Translation of `IMOD/librgctf/brute_force_search.{h,cpp}`.

use super::conjugate_gradient::ConjugateGradient;
use super::functions::ctf_num_omp_threads;

/// C++ `BruteForceSearch` (`brute_force_search.h:1`).
///
/// The C++ pair of a `float (*)(void *, float [])` and a `void *parameters`
/// block is one Rust closure here.
pub struct BruteForceSearch<'a> {
    number_of_dimensions: i32,
    is_in_memory: bool,
    target_function: Option<Box<dyn FnMut(&[f32]) -> f32 + 'a>>,
    starting_value: Vec<f32>,
    best_value: Vec<f32>,
    half_range: Vec<f32>,
    step_size: Vec<f32>,
    dimension_at_max: Vec<bool>,
    best_score: f32,
    num_iterations: i32,
    minimise_at_every_step: bool,
    print_progress_bar: bool,
}

impl<'a> Default for BruteForceSearch<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> BruteForceSearch<'a> {
    /// C++ `BruteForceSearch::BruteForceSearch` (`brute_force_search.cpp:3`).
    pub fn new() -> Self {
        // Nothing to do until Init is called
        Self {
            number_of_dimensions: 0,
            is_in_memory: false,
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

    /// C++ `BruteForceSearch::Init` (`brute_force_search.cpp:35`).
    #[allow(clippy::too_many_arguments)]
    pub fn init<F>(
        &mut self,
        function_to_minimize: F,
        num_dim: i32,
        wanted_starting_value: &[f32],
        wanted_half_range: &[f32],
        wanted_step_size: &[f32],
        should_minimise_at_every_step: bool,
        should_print_progress_bar: bool,
    ) where
        F: FnMut(&[f32]) -> f32 + 'a,
    {
        // Local variables
        let mut current_values = vec![0.0f32; num_dim as usize];
        let mut search_done: bool;

        // Allocate memory
        self.number_of_dimensions = num_dim;
        self.starting_value = vec![0.0; self.number_of_dimensions as usize];
        self.best_value = vec![0.0; self.number_of_dimensions as usize];
        self.half_range = vec![0.0; self.number_of_dimensions as usize];
        self.step_size = vec![0.0; self.number_of_dimensions as usize];
        self.dimension_at_max = vec![false; self.number_of_dimensions as usize];
        self.is_in_memory = true;

        // Copy starting values, half range and step size over; initialise
        for dim_counter in 0..self.number_of_dimensions as usize {
            self.starting_value[dim_counter] = wanted_starting_value[dim_counter];
            self.half_range[dim_counter] = wanted_half_range[dim_counter];
            self.step_size[dim_counter] = wanted_step_size[dim_counter];
            self.dimension_at_max[dim_counter] = false;
            current_values[dim_counter] = -f32::MAX;
        }

        self.minimise_at_every_step = should_minimise_at_every_step;

        // Work out how many iterations the exhaustive search will take
        self.num_iterations = 0;
        loop {
            search_done = false;
            self.increment_current_values(&mut current_values, &mut search_done);
            if search_done {
                break;
            }
            self.num_iterations += 1;
        }

        self.target_function = Some(Box::new(function_to_minimize));
        self.print_progress_bar = should_print_progress_bar;
    }

    /// C++ `BruteForceSearch::GetBestValue` (`brute_force_search.cpp:88`).
    pub fn get_best_value(&self, index: i32) -> f32 {
        self.best_value[index as usize]
    }

    /// C++ inline `BruteForceSearch::GetBestScore` (`brute_force_search.h:29`).
    pub fn get_best_score(&self) -> f32 {
        self.best_score
    }

    /// C++ `BruteForceSearch::IncrementCurrentValues` (`brute_force_search.cpp:95`).
    ///
    /// In the BF loop this is called before the scoring function; before the
    /// loop, every entry of `current_values` is set to -huge.
    pub fn increment_current_values(
        &mut self,
        current_values: &mut [f32],
        search_is_now_completed: &mut bool,
    ) {
        // do the increment
        *search_is_now_completed = true;
        for i in 0..self.number_of_dimensions as usize {
            // if we haven't reached the max for this dimension, increment it,
            // and reset all previous dimensions to their starting point
            if !self.dimension_at_max[i] {
                // if we got here, it means our search is not over yet
                *search_is_now_completed = false;
                // increment the ith dimension
                if current_values[i] < self.starting_value[i] - self.half_range[i] {
                    current_values[i] = self.starting_value[i] - self.half_range[i];
                } else {
                    current_values[i] += self.step_size[i];
                }
                // Reset all previous dimensions to their starting values and
                // reset their dimension_at_max flags to false
                if i > 0 {
                    for j in 0..i {
                        current_values[j] = self.starting_value[j] - self.half_range[j];
                        self.dimension_at_max[j] = false;
                    }
                }
                // if the ith dimension has reached or gone over its max, set it
                // at the max, and set its logical flag dimension_at_max to true
                if current_values[i] >= self.starting_value[i] + self.half_range[i] {
                    current_values[i] = self.starting_value[i] + self.half_range[i];
                    self.dimension_at_max[i] = true;
                    if i == (self.number_of_dimensions - 1) as usize {
                        *search_is_now_completed = true;
                    }
                }
                break;
            }
        } // end of loop over dimensions

        for i in 0..self.number_of_dimensions as usize {
            if current_values[i] < self.starting_value[i] - self.half_range[i] {
                current_values[i] = self.starting_value[i] - self.half_range[i];
            }
        }
    }

    /// C++ `BruteForceSearch::Run` (`brute_force_search.cpp:158`).
    ///
    /// DNM parallelized the non-minimizing arm with OpenMP by storing the step
    /// values first; the iterations are independent, so running them in order
    /// here gives the same `all_scores`.
    pub fn run(&mut self) {
        // Private variables
        let mut current_values = [0.0f32; 16];
        let mut accuracy_for_local_minimization = vec![0.0f32; self.number_of_dimensions as usize];
        let mut current_values_for_local_minimization =
            vec![0.0f32; self.number_of_dimensions as usize];
        // DNM: New arrays for holding the values at each step, the scores from
        // each step, and the best values from the CG search at each step
        let mut all_values =
            vec![0.0f32; (self.number_of_dimensions * self.num_iterations) as usize];
        let mut all_scores = vec![0.0f32; self.num_iterations as usize];
        let mut all_local_best_values =
            vec![0.0f32; (self.number_of_dimensions * self.num_iterations) as usize];
        let mut num_iterations_completed: i32;
        let mut current_score: f32;
        let num_threads: i32;
        let max_threads = 12;

        // The starting values and the corresponding score
        {
            let starting_value = self.starting_value.clone();
            let function = self
                .target_function
                .as_deref_mut()
                .expect("BruteForceSearch object not allocated");
            self.best_score = function(&starting_value);
        }
        for i in 0..self.number_of_dimensions as usize {
            self.best_value[i] = self.starting_value[i];
        }

        // The starting point for the brute-force search
        for i in 0..self.number_of_dimensions as usize {
            current_values[i] = -f32::MAX;
            self.dimension_at_max[i] = false;
        }

        // DNM: Go through the search steps and save the values to be set at each step
        for current_iteration in 0..self.num_iterations {
            let mut search_completed = false;
            self.increment_current_values(
                &mut current_values[..self.number_of_dimensions as usize],
                &mut search_completed,
            );
            for i in 0..self.number_of_dimensions as usize {
                all_values[(self.number_of_dimensions * current_iteration) as usize + i] =
                    current_values[i];
            }
        }

        // The accuracy for the local minimization
        if self.minimise_at_every_step {
            for i in 0..self.number_of_dimensions as usize {
                accuracy_for_local_minimization[i] = self.step_size[i] * 0.5;
            }
        }

        // How many iterations have we completed?
        num_iterations_completed = 0;

        num_threads = ctf_num_omp_threads(max_threads);
        let _ = num_threads;

        // start the brute-force search iterations
        // DNM the minimizer did not work right with threads, so just run it normally
        if self.minimise_at_every_step {
            for current_iteration in 0..self.num_iterations {
                for i in 0..self.number_of_dimensions as usize {
                    current_values[i] =
                        all_values[(self.number_of_dimensions * current_iteration) as usize + i];
                }

                let number_of_dimensions = self.number_of_dimensions;
                let function = self
                    .target_function
                    .as_deref_mut()
                    .expect("BruteForceSearch object not allocated");
                let mut local_minimizer = ConjugateGradient::new();
                for i in 0..number_of_dimensions as usize {
                    current_values_for_local_minimization[i] = current_values[i];
                }
                local_minimizer.init(
                    |candidate| function(candidate),
                    number_of_dimensions,
                    &current_values_for_local_minimization,
                    &accuracy_for_local_minimization,
                );
                local_minimizer.run();
                // DNM: store the results
                all_scores[current_iteration as usize] = local_minimizer.get_best_score();
                for i in 0..number_of_dimensions as usize {
                    all_local_best_values
                        [(number_of_dimensions * current_iteration) as usize + i] =
                        local_minimizer.get_best_value(i as i32);
                }
            }
        } else {
            for current_iteration in 0..self.num_iterations {
                for i in 0..self.number_of_dimensions as usize {
                    current_values[i] =
                        all_values[(self.number_of_dimensions * current_iteration) as usize + i];
                }

                // Try the current parameters by calling the scoring function
                let number_of_dimensions = self.number_of_dimensions;
                let function = self
                    .target_function
                    .as_deref_mut()
                    .expect("BruteForceSearch object not allocated");
                all_scores[current_iteration as usize] =
                    function(&current_values[..number_of_dimensions as usize]);
            }
        }

        // DNM: Loop through the iterations again and look for the best result
        for current_iteration in 0..self.num_iterations {
            for i in 0..self.number_of_dimensions as usize {
                current_values[i] =
                    all_values[(self.number_of_dimensions * current_iteration) as usize + i];
            }
            current_score = all_scores[current_iteration as usize];

            // if the score is the best we've seen so far, remember the values
            // and the score
            if current_score < self.best_score {
                self.best_score = current_score;
                if self.minimise_at_every_step {
                    for i in 0..self.number_of_dimensions as usize {
                        self.best_value[i] = all_local_best_values
                            [(self.number_of_dimensions * current_iteration) as usize + i];
                    }
                } else {
                    for i in 0..self.number_of_dimensions as usize {
                        self.best_value[i] = current_values[i];
                    }
                }
            }

            // Progress
            num_iterations_completed += 1;
        } // End of loop over exhaustive search operation
        let _ = num_iterations_completed;
        let _ = self.print_progress_bar;
        let _ = self.is_in_memory;
    }
}
