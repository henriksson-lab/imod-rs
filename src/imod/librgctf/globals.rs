//! Translation of `IMOD/librgctf/globals.cpp`.

use std::sync::{LazyLock, Mutex};

use super::random_number_generator::RandomNumberGenerator;

/// C++ `global_random_number_generator`.
///
/// Rust constructs it on first access rather than during C++ static
/// initialization; its seed and process-global C-stream behavior are retained.
pub static GLOBAL_RANDOM_NUMBER_GENERATOR: LazyLock<Mutex<RandomNumberGenerator>> =
    LazyLock::new(|| Mutex::new(RandomNumberGenerator::with_seed(-1, false)));

#[cfg(test)]
mod tests {
    use super::GLOBAL_RANDOM_NUMBER_GENERATOR;

    #[test]
    fn source_global_random_stream_is_available() {
        let mut generator = GLOBAL_RANDOM_NUMBER_GENERATOR
            .lock()
            .expect("random generator lock poisoned");
        assert!((-1.0..=1.0).contains(&generator.uniform_random()));
    }
}
