//! Safe translation of `IMOD/librgctf/conjugate_gradient.cpp` and `.h`.

use crate::imod::librgctf::va04::va04a;

/// Errors from constructing or running a [`ConjugateGradient`] minimizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConjugateGradientError {
    /// The source rejects zero-dimensional minimization with `MyDebugAssertTrue`.
    ZeroDimensions,
    /// The source copies one starting value and accuracy per dimension.
    MismatchedDimensions,
    /// [`ConjugateGradient::run`] requires a preceding successful initialization.
    NotInitialized,
}

/// C++ `ConjugateGradient`, with owned parameter vectors and a Rust objective.
pub struct ConjugateGradient<'a> {
    n: usize,
    num_function_calls: i32,
    best_values: Vec<f32>,
    e: Vec<f32>,
    escale: f32,
    best_score: f32,
    target_function: Option<Box<dyn FnMut(&[f32]) -> f32 + 'a>>,
}

/// `ConjugateGradient::ConjugateGradient` (`conjugate_gradient.cpp:3`),
/// exposed as an owned factory for translated construction call sites.
pub fn conjugate_gradient<'a>() -> ConjugateGradient<'a> {
    ConjugateGradient::new()
}

impl<'a> Default for ConjugateGradient<'a> {
    fn default() -> Self {
        Self {
            n: 0,
            num_function_calls: 0,
            best_values: Vec::new(),
            e: Vec::new(),
            escale: 0.0,
            best_score: f32::MAX,
            target_function: None,
        }
    }
}

impl<'a> ConjugateGradient<'a> {
    /// Original default constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Original `ConjugateGradient::Init`.
    ///
    /// Reinitializing replaces all owned state and objective, as the C++ class
    /// deletes its old arrays before allocating replacements.
    pub fn init<F>(
        &mut self,
        function_to_minimize: F,
        starting_value: &[f32],
        accuracy: &[f32],
    ) -> Result<f32, ConjugateGradientError>
    where
        F: FnMut(&[f32]) -> f32 + 'a,
    {
        if starting_value.is_empty() {
            return Err(ConjugateGradientError::ZeroDimensions);
        }
        if accuracy.len() != starting_value.len() {
            return Err(ConjugateGradientError::MismatchedDimensions);
        }
        self.n = starting_value.len();
        self.best_values = starting_value.to_vec();
        self.e = accuracy.to_vec();
        self.escale = 100.0;
        self.num_function_calls = 0;
        self.target_function = Some(Box::new(function_to_minimize));
        let function = self
            .target_function
            .as_deref_mut()
            .expect("objective installed above");
        self.best_score = function(&self.best_values);
        Ok(self.best_score)
    }

    /// Original `ConjugateGradient::Run`.
    pub fn run(&mut self) -> Result<f32, ConjugateGradientError> {
        if self.n == 0 || self.best_values.len() != self.n || self.e.len() != self.n {
            return Err(ConjugateGradientError::NotInitialized);
        }
        let Some(function) = self.target_function.as_deref_mut() else {
            return Err(ConjugateGradientError::NotInitialized);
        };
        va04a(
            self.n,
            &self.e,
            self.escale,
            &mut self.num_function_calls,
            |candidate| function(candidate),
            &mut self.best_score,
            0,
            1,
            50,
            &mut self.best_values,
        );
        Ok(self.best_score)
    }

    /// Original inline `GetBestValue`.
    pub fn best_value(&self, index: usize) -> Option<f32> {
        self.best_values.get(index).copied()
    }

    /// Original inline `GetBestScore`.
    pub fn best_score(&self) -> f32 {
        self.best_score
    }

    /// Native replacement for inline `GetPointerToBestValues`.
    pub fn best_values(&self) -> &[f32] {
        &self.best_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_owns_input_vectors_and_evaluates_the_initial_score() {
        let mut minimizer = conjugate_gradient();
        let start = [3.0, -2.0];
        assert_eq!(
            minimizer.init(|values| values[0] + values[1], &start, &[0.5, 0.25]),
            Ok(1.0)
        );
        assert_eq!(minimizer.best_values(), &start);
        assert_eq!(minimizer.best_value(2), None);
    }

    #[test]
    fn run_preserves_a_constant_objective_and_requires_initialization() {
        let mut minimizer = ConjugateGradient::new();
        assert_eq!(minimizer.run(), Err(ConjugateGradientError::NotInitialized));
        minimizer.init(|_| -3.5, &[4.0], &[1.0]).unwrap();
        assert_eq!(minimizer.run(), Ok(-3.5));
        assert_eq!(minimizer.best_values().len(), 1);
    }

    #[test]
    fn init_rejects_invalid_dimensions() {
        let mut minimizer = ConjugateGradient::new();
        assert_eq!(
            minimizer.init(|_| 0.0, &[], &[]),
            Err(ConjugateGradientError::ZeroDimensions)
        );
        assert_eq!(
            minimizer.init(|_| 0.0, &[1.0], &[]),
            Err(ConjugateGradientError::MismatchedDimensions)
        );
    }
}
