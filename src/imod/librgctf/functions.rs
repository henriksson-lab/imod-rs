//! Translation of `IMOD/librgctf/functions.{h,cpp}`.

use std::cmp::Ordering;
use std::sync::{OnceLock, RwLock};

const PI: f32 = 3.141_592_653_59;

/// C++ inline `IsEven`.
pub fn is_even(number_to_check: i32) -> bool {
    number_to_check % 2 == 0
}

/// C++ inline `deg_2_rad`.
pub fn deg_2_rad(degrees: f32) -> f32 {
    degrees * PI / 180.0
}

/// Rust trait replacing the two C++ `myroundint` overloads.
pub trait RoundInput {
    fn my_round_int(self) -> i32;
}

impl RoundInput for f32 {
    /// C++ `myroundint(float)`.
    fn my_round_int(self) -> i32 {
        if self > 0.0 {
            (self + 0.5) as i32
        } else {
            (self - 0.5) as i32
        }
    }
}

impl RoundInput for f64 {
    /// C++ `myroundint(double)`.
    fn my_round_int(self) -> i32 {
        if self > 0.0 {
            (self + 0.5) as i32
        } else {
            (self - 0.5) as i32
        }
    }
}

/// C++ `rankSort`.
pub fn rank_sort(values: &[f32]) -> Vec<usize> {
    let mut sorted: Vec<(f32, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(i, value)| (value, i))
        .collect();
    sorted.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let mut rank = 0;
    let mut previous = None;
    let mut result = vec![0; values.len()];
    for (position, (value, original_index)) in sorted.into_iter().enumerate() {
        if previous != Some(value) {
            rank = position;
            previous = Some(value);
        }
        result[original_index] = rank;
    }
    result
}

/// The Rust string callback corresponding to source `CharArgType`.
pub type PrintFunction = fn(&str);

static PRINT_FUNCTION: OnceLock<RwLock<Option<PrintFunction>>> = OnceLock::new();

/// C++ `internalSetPrintFunc`.
pub fn internal_set_print_func(function: Option<PrintFunction>) {
    *PRINT_FUNCTION
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("print callback lock poisoned") = function;
}

/// Safe Rust replacement for C++ variadic `wxPrintf`.
pub fn print_message(message: &str) {
    if let Some(function) = *PRINT_FUNCTION
        .get_or_init(|| RwLock::new(None))
        .read()
        .expect("print callback lock poisoned")
    {
        function(message);
    } else {
        print!("{message}");
    }
}

/// C++ `ctfNumOMPthreads` in this crate's non-OpenMP configuration.
pub fn ctf_num_omp_threads(_optimal_threads: i32) -> i32 {
    1
}

/// C++ `ctfOMPthreadNum` in this crate's non-OpenMP configuration.
pub fn ctf_omp_thread_num() -> i32 {
    0
}

/// C++ `ctfWallTime`.
pub fn ctf_wall_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before Unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::{RoundInput, ctf_num_omp_threads, deg_2_rad, is_even, rank_sort};

    #[test]
    fn scalar_and_rank_helpers_match_source_rules() {
        assert!(is_even(-4));
        assert!(!is_even(3));
        assert!((deg_2_rad(180.0) - std::f32::consts::PI).abs() < 0.000_001);
        assert_eq!(1.5_f32.my_round_int(), 2);
        assert_eq!((-1.5_f64).my_round_int(), -2);
        assert_eq!(rank_sort(&[3.0, 1.0, 1.0, 2.0]), vec![3, 0, 0, 2]);
        assert_eq!(ctf_num_omp_threads(64), 1);
    }
}
