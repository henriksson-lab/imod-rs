//! Translation of `IMOD/raptor/lasik/svl/lib/base/svlCodeProfiler.h` and
//! `svlCodeProfiler.cpp`, the parts `MarkersCorrespond` reaches.
//!
//! `svlCodeProfiler::enabled` is `false` (`svlCodeProfiler.cpp:49`) and only
//! the configuration manager, which the program never runs, sets it; so
//! `getHandle` returns -1 and `tic`/`toc` do nothing.  The profiling table
//! behind them is recorded in `DEAD_CODE.md`.

/// `svlCodeProfiler::enabled`.
const ENABLED: bool = false;

/// `svlCodeProfiler::getHandle(name)` (`svlCodeProfiler.cpp:54`).
pub fn get_handle(_name: &str) -> i32 {
    if !ENABLED {
        return -1;
    }
    unreachable!("svlCodeProfiler::enabled is never set")
}

/// `svlCodeProfiler::tic(handle)` (`svlCodeProfiler.h:121`).
pub fn tic(_handle: i32) {
    if ENABLED {
        unreachable!("svlCodeProfiler::enabled is never set")
    }
}

/// `svlCodeProfiler::toc(handle)` (`svlCodeProfiler.h:124`).
pub fn toc(_handle: i32) {
    if ENABLED {
        unreachable!("svlCodeProfiler::enabled is never set")
    }
}
