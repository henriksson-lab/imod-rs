//! Owned translation of `IMOD/raptor/lasik/svl/lib/base/svlCodeProfiler.{h,cpp}`.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct SvlCodeProfilerEntry {
    start: Instant,
    total_cpu: Duration,
    total_wall: Duration,
    total_calls: usize,
}

impl SvlCodeProfilerEntry {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            total_cpu: Duration::ZERO,
            total_wall: Duration::ZERO,
            total_calls: 0,
        }
    }
    fn clear(&mut self) {
        self.start = Instant::now();
        self.total_cpu = Duration::ZERO;
        self.total_wall = Duration::ZERO;
        self.total_calls = 0;
    }
    fn tic(&mut self) {
        self.start = Instant::now();
    }
    fn toc(&mut self) {
        let elapsed = self.start.elapsed();
        // Stable Rust has no process-CPU clock; its monotonic clock safely
        // replaces both source timing totals without FFI.
        self.total_cpu += elapsed;
        self.total_wall += elapsed;
        self.start = Instant::now();
        self.total_calls += 1;
    }
}

/// Owned, per-run version of C++ `svlCodeProfiler`'s static registry.
#[derive(Debug, Default)]
pub struct SvlCodeProfiler {
    pub enabled: bool,
    entries: Vec<SvlCodeProfilerEntry>,
    names: BTreeMap<String, usize>,
}

impl SvlCodeProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Source `getHandle`; `-1` is its disabled sentinel.
    pub fn get_handle(&mut self, name: &str) -> isize {
        if !self.enabled {
            return -1;
        }
        if let Some(&handle) = self.names.get(name) {
            return handle as isize;
        }
        let handle = self.entries.len();
        self.names.insert(name.to_owned(), handle);
        self.entries.push(SvlCodeProfilerEntry::new());
        handle as isize
    }

    /// Header-inline `clear`.
    pub fn clear(&mut self, handle: isize) {
        if self.enabled {
            if let Some(entry) = usize::try_from(handle)
                .ok()
                .and_then(|i| self.entries.get_mut(i))
            {
                entry.clear();
            }
        }
    }
    /// Header-inline `tic`.
    pub fn tic(&mut self, handle: isize) {
        if self.enabled {
            if let Some(entry) = usize::try_from(handle)
                .ok()
                .and_then(|i| self.entries.get_mut(i))
            {
                entry.tic();
            }
        }
    }
    /// Header-inline `toc`.
    pub fn toc(&mut self, handle: isize) {
        if self.enabled {
            if let Some(entry) = usize::try_from(handle)
                .ok()
                .and_then(|i| self.entries.get_mut(i))
            {
                entry.toc();
            }
        }
    }
    /// Header-inline `calendarSeconds`.
    pub fn calendar_seconds(&self, handle: isize) -> f64 {
        if !self.enabled {
            return -1.0;
        }
        usize::try_from(handle)
            .ok()
            .and_then(|i| self.entries.get(i))
            .map_or(-1.0, |entry| entry.total_wall.as_secs_f64())
    }
    /// Header-inline `time`.
    pub fn time(&self, handle: isize) -> f64 {
        if !self.enabled {
            return -1.0;
        }
        usize::try_from(handle)
            .ok()
            .and_then(|i| self.entries.get(i))
            .map_or(-1.0, |entry| entry.total_cpu.as_secs_f64())
    }
    /// Header-inline `calls`.
    pub fn calls(&self, handle: isize) -> isize {
        if !self.enabled {
            return -1;
        }
        usize::try_from(handle)
            .ok()
            .and_then(|i| self.entries.get(i))
            .map_or(-1, |entry| entry.total_calls as isize)
    }

    /// Source `print`, returned as owned text instead of written to a C++ stream.
    pub fn print(&self) -> String {
        if !self.enabled || self.entries.is_empty() {
            return String::new();
        }
        let mut output =
            String::from("  CALLS        CPU TIME   WALL TIME      TIME PER   FUNCTION\n");
        for (name, &handle) in &self.names {
            let entry = &self.entries[handle];
            let cpu_milliseconds = entry.total_cpu.as_millis();
            let wall_seconds = entry.total_wall.as_secs();
            output.push_str(&format!(
                "{:>7}   {:>3}:{:02}:{:02}.{:03}   {:>3}:{:02}:{:02}    ",
                entry.total_calls,
                cpu_milliseconds / 3_600_000,
                cpu_milliseconds / 60_000 % 60,
                cpu_milliseconds / 1_000 % 60,
                cpu_milliseconds % 1_000,
                wall_seconds / 3_600,
                wall_seconds / 60 % 60,
                wall_seconds % 60
            ));
            if entry.total_calls == 0 {
                output.push_str("      0  s   ");
            } else {
                let per = entry.total_cpu.as_secs_f64() / entry.total_calls as f64;
                if per < 1e-6 {
                    output.push_str(&format!(
                        "{:>3}.{:03} ns   ",
                        (1e9 * per) as u64,
                        (1e12 * per) as u64 % 1_000
                    ));
                } else if per < 1e-3 {
                    output.push_str(&format!(
                        "{:>3}.{:03} us   ",
                        (1e6 * per) as u64,
                        (1e9 * per) as u64 % 1_000
                    ));
                } else if per < 1.0 {
                    output.push_str(&format!(
                        "{:>3}.{:03} ms   ",
                        (1e3 * per) as u64,
                        (1e6 * per) as u64 % 1_000
                    ));
                } else if per < 1e3 {
                    output.push_str(&format!(
                        "{:>3}.{:03}  s   ",
                        per as u64,
                        (1e3 * per) as u64 % 1_000
                    ));
                } else {
                    output.push_str("   >999  s   ");
                }
            }
            output.push_str(name);
            output.push('\n');
        }
        output
    }
}

/// Source `svlCodeProfilerConfig`, detached from C++ static registration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SvlCodeProfilerConfig;

impl SvlCodeProfilerConfig {
    pub const NAME: &'static str = "svlBase.svlCodeProfiler";
    pub fn usage(self) -> &'static str {
        "      enabled      :: enable profiling (default: false)\n"
    }
    /// Source `setConfiguration`.
    pub fn set_configuration(
        self,
        profiler: &mut SvlCodeProfiler,
        name: &str,
        value: &str,
    ) -> Result<(), &'static str> {
        if name != "enabled" {
            return Err("unrecognized configuration option for svlBase.svlCodeProfiler");
        }
        profiler.enabled = value.eq_ignore_ascii_case("true") || value == "1";
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SvlCodeProfiler, SvlCodeProfilerConfig};
    #[test]
    fn disabled_profiler_retains_source_sentinels() {
        let mut profiler = SvlCodeProfiler::new();
        assert_eq!(profiler.get_handle("work"), -1);
        assert_eq!(profiler.time(-1), -1.0);
        assert_eq!(profiler.calendar_seconds(-1), -1.0);
        assert_eq!(profiler.calls(-1), -1);
        assert!(profiler.print().is_empty());
    }
    #[test]
    fn handles_accumulate_calls_and_print_in_name_order() {
        let mut profiler = SvlCodeProfiler::new();
        profiler.enabled = true;
        let beta = profiler.get_handle("beta");
        let alpha = profiler.get_handle("alpha");
        assert_eq!(profiler.get_handle("beta"), beta);
        profiler.tic(beta);
        profiler.toc(beta);
        assert_eq!(profiler.calls(beta), 1);
        profiler.clear(alpha);
        let report = profiler.print();
        assert!(report.starts_with("  CALLS        CPU TIME"));
        assert!(report.find("alpha").unwrap() < report.find("beta").unwrap());
    }
    #[test]
    fn configuration_accepts_only_source_enabled_values() {
        let config = SvlCodeProfilerConfig;
        let mut profiler = SvlCodeProfiler::new();
        config
            .set_configuration(&mut profiler, "enabled", "TRUE")
            .unwrap();
        assert!(profiler.enabled);
        config
            .set_configuration(&mut profiler, "enabled", "0")
            .unwrap();
        assert!(!profiler.enabled);
        assert!(
            config
                .set_configuration(&mut profiler, "other", "1")
                .is_err()
        );
        assert!(config.usage().contains("enable profiling"));
    }
}
