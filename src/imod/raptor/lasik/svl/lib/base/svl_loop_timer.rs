//! Safe translation of `lasik/svl/lib/base/svlLoopTimer.{h,cpp}`.
//!
//! The C++ version measures `clock()` ticks.  Rust's [`Instant`] is monotonic
//! wall-clock time, so this implementation represents the same elapsed and
//! estimated intervals as owned [`Duration`] values without C timing FFI.

use std::time::{Duration, Instant};

/// C++ `svlLoopTimerRatio`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SvLoopTimerRatio {
    #[default]
    Geometric,
    Additive,
}

/// C++ `svlLoopTimerUnit`, with a monotonic start instant in place of `clock()`.
#[derive(Clone, Debug)]
pub struct SvLoopTimerUnit {
    tic: Instant,
    pub last_inc_toc: Duration,
    pub ratio_type: SvLoopTimerRatio,
    pub i: i32,
    pub n: i32,
    pub width: usize,
    pub ratio: f32,
}

impl SvLoopTimerUnit {
    pub fn new(n: i32, ratio: f32, ratio_type: SvLoopTimerRatio) -> Self {
        Self {
            tic: Instant::now(),
            last_inc_toc: Duration::ZERO,
            ratio_type,
            i: 0,
            n,
            width: (n.unsigned_abs() as u64 + 1).to_string().len(),
            ratio,
        }
    }
    /// C++ `svlLoopTimerUnit::inc`.
    pub fn inc(&mut self) {
        self.i += 1;
        self.last_inc_toc = self.tic.elapsed();
    }
    /// C++ `svlLoopTimerUnit::toc`.
    pub fn toc(&self) -> Duration {
        self.tic.elapsed()
    }
}

/// C++ `svlLoopTimer`.
#[derive(Debug, Default)]
pub struct SvLoopTimer {
    pub units: Vec<SvLoopTimerUnit>,
}

impl SvLoopTimer {
    pub fn new() -> Self {
        Self::default()
    }
    /// C++ `push`.
    pub fn push(&mut self, n: i32, ratio: f32, ratio_type: SvLoopTimerRatio) {
        self.units.push(SvLoopTimerUnit::new(n, ratio, ratio_type));
    }
    /// C++ `pop`; returns the popped owned timer instead of silently discarding it.
    pub fn pop(&mut self) -> Option<SvLoopTimerUnit> {
        self.units.pop()
    }
    /// C++ `clear`.
    pub fn clear(&mut self) {
        self.units.clear();
    }
    /// C++ `inc`.
    pub fn inc(&mut self) -> Option<()> {
        self.units.last_mut().map(|unit| unit.inc())
    }
    /// C++ `toc`.
    pub fn toc(&self) -> Option<Duration> {
        self.units.last().map(SvLoopTimerUnit::toc)
    }
    /// C++ `print` rendered to an owned string rather than a global ostream.
    pub fn print(&self) -> Option<String> {
        self.toc()
            .map(|duration| print_time(duration.as_secs() as i64))
    }
    /// C++ `printETC` rendered to an owned string.
    pub fn print_etc(&self) -> Option<String> {
        let mut index = self.units.len().checked_sub(1)?;
        let current = &self.units[index];
        let toc = current.toc();
        let etc = if toc.is_zero() {
            None
        } else {
            get_etc_unit(current)
        };
        let mut output = format!(
            "{:>width$}/{}: {} > {}",
            current.i,
            current.n,
            print_time(toc.as_secs() as i64),
            etc.map_or_else(
                || "-:--:--".to_owned(),
                |value| print_time(value.as_secs() as i64)
            ),
            width = current.width
        );
        let mut last_total = toc + etc.unwrap_or(Duration::ZERO);
        while index > 0 {
            index -= 1;
            let unit = &self.units[index];
            let zero = unit.last_inc_toc.is_zero() && last_total.is_zero();
            let estimate = if zero {
                None
            } else if last_total.is_zero() {
                get_etc_unit(unit)
            } else {
                get_etc(
                    unit.i + 1,
                    unit.n,
                    unit.last_inc_toc + last_total,
                    unit.ratio,
                    unit.ratio_type,
                )
            };
            let display = estimate
                .and_then(|value| value.checked_sub(self.units[index + 1].last_inc_toc))
                .unwrap_or(Duration::ZERO)
                + last_total;
            output.push_str(&format!(
                "  -- {:>width$}/{}: {} > {}",
                unit.i,
                unit.n,
                print_time(unit.toc().as_secs() as i64),
                if zero {
                    "-:--:--".to_owned()
                } else {
                    print_time(display.as_secs() as i64)
                },
                width = unit.width
            ));
            last_total = unit.last_inc_toc + estimate.unwrap_or(Duration::ZERO);
        }
        Some(output)
    }
}

/// C++ `printTime`.
pub fn print_time(seconds: i64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds - hours * 3600) / 60;
    let seconds = seconds - hours * 3600 - minutes * 60;
    format!("{hours}:{minutes:02}:{seconds:02}")
}

/// C++ `printMsTime`.
pub fn print_ms_time(milliseconds: i64) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds - hours * 3_600_000) / 60_000;
    let seconds = (milliseconds - hours * 3_600_000 - minutes * 60_000) / 1000;
    let millis = milliseconds - hours * 3_600_000 - minutes * 60_000 - seconds * 1000;
    format!("{hours}:{minutes:02}:{seconds:02}.{millis:02}")
}

/// C++ `getETC(int, int, long, float, svlLoopTimerRatio)`.
pub fn get_etc(
    i: i32,
    n: i32,
    toc: Duration,
    ratio: f32,
    ratio_type: SvLoopTimerRatio,
) -> Option<Duration> {
    if i == 0 {
        return None;
    }
    let ticks = toc.as_nanos() as f64;
    let estimate = match ratio_type {
        SvLoopTimerRatio::Geometric if ratio == 1.0 => ticks * (n - i) as f64 / i as f64,
        SvLoopTimerRatio::Geometric => {
            let mut done = 0.0_f32;
            let mut remaining = 0.0_f32;
            let mut factor = 1.0_f32;
            for _ in 0..i {
                done += factor;
                factor *= ratio;
            }
            for _ in i..n {
                remaining += factor;
                factor *= ratio;
            }
            ticks * remaining as f64 / done as f64
        }
        SvLoopTimerRatio::Additive => {
            let mut done = 0.0_f32;
            let mut remaining = 0.0_f32;
            let mut factor = 1.0_f32;
            for _ in 0..i {
                done += factor;
                factor += ratio;
            }
            for _ in i..n {
                remaining += factor;
                factor += ratio;
            }
            ticks * remaining as f64 / done as f64
        }
    };
    (estimate.is_finite() && estimate >= 0.0)
        .then(|| Duration::from_nanos(estimate.min(u64::MAX as f64) as u64))
}

/// C++ `getETC(svlLoopTimerUnit&)`.
pub fn get_etc_unit(unit: &SvLoopTimerUnit) -> Option<Duration> {
    get_etc(
        unit.i,
        unit.n,
        unit.last_inc_toc,
        unit.ratio,
        unit.ratio_type,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_formatters_preserve_hour_minute_second_layout() {
        assert_eq!(print_time(3_661), "1:01:01");
        assert_eq!(print_ms_time(3_661_009), "1:01:01.09");
    }
    #[test]
    fn equal_geometric_work_scales_by_remaining_iterations() {
        assert_eq!(
            get_etc(
                2,
                5,
                Duration::from_secs(10),
                1.0,
                SvLoopTimerRatio::Geometric
            ),
            Some(Duration::from_secs(15))
        );
        assert_eq!(
            get_etc(
                0,
                5,
                Duration::from_secs(10),
                1.0,
                SvLoopTimerRatio::Geometric
            ),
            None
        );
    }
    #[test]
    fn timer_stack_is_owned_and_formats_nested_estimates() {
        let mut timer = SvLoopTimer::new();
        timer.push(2, 1.0, SvLoopTimerRatio::Geometric);
        timer.inc();
        assert!(timer.print().is_some());
        assert!(timer.print_etc().is_some());
        assert!(timer.pop().is_some());
        assert!(timer.toc().is_none());
    }
}
