//! Translation of `IMOD/librgctf/conjugate_gradient.{h,cpp}`.

use super::va04::va04a;

/// C++ `ConjugateGradient` (`conjugate_gradient.h:1`).
///
/// The C++ pair of a `float (*)(void *, float [])` and an untyped `void *`
/// parameter block is one Rust closure here; nothing else about the class
/// changes.
pub struct ConjugateGradient<'a> {
    is_in_memory: bool,
    /// Number of parameters to refine
    n: i32,
    /// Number of calls to the scoring function
    num_function_calls: i32,
    /// Final best parameters determined by conjugate gradient minimization
    best_values: Vec<f32>,
    /// Recalculated accuracy for subroutine va04
    e: Vec<f32>,
    /// Anticipated change in the parameters during the minimization
    escale: f32,
    /// Final best score
    best_score: f32,
    target_function: Option<Box<dyn FnMut(&[f32]) -> f32 + 'a>>,
}

impl<'a> Default for ConjugateGradient<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ConjugateGradient<'a> {
    /// C++ `ConjugateGradient::ConjugateGradient` (`conjugate_gradient.cpp:3`).
    pub fn new() -> Self {
        Self {
            is_in_memory: false,
            n: 0,
            num_function_calls: 0,
            best_values: Vec::new(),
            e: Vec::new(),
            escale: 0.0,
            best_score: f32::MAX,
            target_function: None,
        }
    }

    /// C++ `ConjugateGradient::Init` (`conjugate_gradient.cpp:26`).
    pub fn init<F>(
        &mut self,
        function_to_minimize: F,
        num_dim: i32,
        starting_value: &[f32],
        accuracy: &[f32],
    ) -> f32
    where
        F: FnMut(&[f32]) -> f32 + 'a,
    {
        // Copy pointers to the target function and the needed parameters
        self.target_function = Some(Box::new(function_to_minimize));

        if self.is_in_memory {
            self.best_values.clear();
            self.e.clear();
            self.is_in_memory = false;
        }

        // Allocate memory
        self.n = num_dim;
        self.best_values = vec![0.0; self.n as usize];
        self.e = vec![0.0; self.n as usize];
        self.is_in_memory = true;

        // Initialise values
        self.escale = 100.0;
        self.num_function_calls = 0;

        for dim_counter in 0..self.n as usize {
            self.best_values[dim_counter] = starting_value[dim_counter];
            self.e[dim_counter] = accuracy[dim_counter];
        }

        // Call the target function to find out our starting score
        let function = self
            .target_function
            .as_deref_mut()
            .expect("objective installed above");
        self.best_score = function(&starting_value[..self.n as usize]);

        self.best_score
    }

    /// C++ `ConjugateGradient::Run` (`conjugate_gradient.cpp:67`).
    pub fn run(&mut self) -> f32 {
        let iprint = 0;
        let icon = 1;
        let maxit = 50;

        let function = self
            .target_function
            .as_deref_mut()
            .expect("ConjugateGradient::Run before Init");
        va04a(
            self.n,
            &self.e,
            self.escale,
            &mut self.num_function_calls,
            function,
            &mut self.best_score,
            iprint,
            icon,
            maxit,
            &mut self.best_values,
        );

        self.best_score
    }

    /// C++ inline `ConjugateGradient::GetBestValue` (`conjugate_gradient.h:25`).
    pub fn get_best_value(&self, index: i32) -> f32 {
        self.best_values[index as usize]
    }

    /// C++ inline `ConjugateGradient::GetBestScore` (`conjugate_gradient.h:26`).
    pub fn get_best_score(&self) -> f32 {
        self.best_score
    }

    /// C++ inline `ConjugateGradient::GetPointerToBestValues` (`conjugate_gradient.h:27`).
    pub fn get_pointer_to_best_values(&self) -> &[f32] {
        &self.best_values
    }
}
