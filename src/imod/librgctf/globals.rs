//! Translation of `IMOD/librgctf/globals.cpp`.

use std::sync::{LazyLock, Mutex, MutexGuard};

use super::random_number_generator::RandomNumberGenerator;

/// C++ `RandomNumberGenerator global_random_number_generator(-1)`
/// (`globals.cpp:3`).
///
/// C++ runs the constructor during static initialization; Rust runs it on
/// first access.  Either way the seed is `time(NULL)` and the stream is the
/// process-global C `rand`, since `use_internal` defaults to false.
static GLOBAL_RANDOM_NUMBER_GENERATOR: LazyLock<Mutex<RandomNumberGenerator>> =
    LazyLock::new(|| Mutex::new(RandomNumberGenerator::with_seed(-1, false)));

/// Borrows `global_random_number_generator`.
pub fn global_random_number_generator() -> MutexGuard<'static, RandomNumberGenerator> {
    GLOBAL_RANDOM_NUMBER_GENERATOR
        .lock()
        .expect("global random generator lock poisoned")
}
