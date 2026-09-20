//! Translation of `IMOD/libcfshr/coresprocsthreads.c`.

use core::sync::atomic::{AtomicI32, Ordering};

const MAX_CPU_SOCKETS: usize = 64;
static S_CPU_IS_AMD: AtomicI32 = AtomicI32::new(-1);

/// Physical-core and logical-processor counts reported by the operating system.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessorCounts {
    pub physical: i32,
    pub logical: i32,
}

/// C `numCoresAndLogicalProcs` (`coresprocsthreads.c:40`).
///
/// The crate is built without `_OPENMP`; this is the Linux `/proc/cpuinfo`
/// branch selected by the vendored source on the current target.
///
/// A malformed or unavailable CPU description yields the same zero, partial,
/// or negative-physical sentinel count that the C function would have written
/// through its output pointers.
pub fn num_cores_and_logical_procs() -> ProcessorCounts {
    let mut processor_core_count = 0;
    let mut logical_processor_count = 0;
    let mut parse_error = false;
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo");
    if let Ok(cpuinfo) = cpuinfo {
        let mut socket_flags = [false; MAX_CPU_SOCKETS];
        let mut current_id = None;
        let mut current_cores = None;

        for line in cpuinfo.lines() {
            if S_CPU_IS_AMD.load(Ordering::SeqCst) < 0 && line.contains("vendor_id") {
                S_CPU_IS_AMD.store(line.contains("AuthenticAMD") as i32, Ordering::SeqCst);
            }
            if line.contains("physical id") {
                if current_id.is_some() {
                    parse_error = true;
                    break;
                }
                let Some((_, value)) = line.split_once(':') else {
                    parse_error = true;
                    break;
                };
                let id = value.trim().parse::<i32>().unwrap_or(0);
                if !(0..MAX_CPU_SOCKETS as i32).contains(&id) {
                    parse_error = true;
                    break;
                }
                current_id = Some(id);
            }
            if line.contains("cpu cores") {
                if current_cores.is_some() {
                    parse_error = true;
                    break;
                }
                let Some((_, value)) = line.split_once(':') else {
                    parse_error = true;
                    break;
                };
                let cores = value.trim().parse::<i32>().unwrap_or(0);
                if cores <= 0 {
                    parse_error = true;
                    break;
                }
                current_cores = Some(cores);
            }
            if let (Some(id), Some(cores)) = (current_id, current_cores) {
                logical_processor_count += 1;
                if !socket_flags[id as usize] {
                    processor_core_count += cores;
                    socket_flags[id as usize] = true;
                }
                current_id = None;
                current_cores = None;
            }
        }
        if parse_error {
            processor_core_count *= -1;
        }
    }
    if S_CPU_IS_AMD.load(Ordering::SeqCst) < 0 {
        S_CPU_IS_AMD.store(0, Ordering::SeqCst);
    }
    let counts = ProcessorCounts {
        physical: processor_core_count,
        logical: logical_processor_count,
    };
    counts
}

/// C `numOMPthreads` (`coresprocsthreads.c:193`).
/// This build has no `_OPENMP`, so the source's complete selected branch returns one.
pub fn num_omp_threads(_optimal_threads: i32) -> i32 {
    1
}

/// C `b3dCpuIsAMD` (`coresprocsthreads.c:287`).
pub fn b3d_cpu_is_amd() -> i32 {
    if S_CPU_IS_AMD.load(Ordering::SeqCst) < 0 {
        num_omp_threads(4);
    }
    S_CPU_IS_AMD.load(Ordering::SeqCst)
}

/// C `wallTime` (`coresprocsthreads.c:294`).
pub fn wall_time() -> f64 {
    // `gettimeofday(&tv, NULL)`: the wall clock since the epoch, in seconds
    // and microseconds, which is what `SystemTime::UNIX_EPOCH` yields.
    let value = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO);
    value.as_secs() as f64 + value.subsec_micros() as f64 / 1_000_000.
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_no_openmp_source_branch_uses_one_thread() {
        assert_eq!(num_omp_threads(0), 1);
        assert_eq!(num_omp_threads(128), 1);
    }
    #[test]
    fn wall_time_and_cpu_probe_return_source_domain_values() {
        let first = wall_time();
        let second = wall_time();
        assert!(second >= first);
        let value = b3d_cpu_is_amd();
        assert!(value == -1 || value == 0 || value == 1);
    }
    #[test]
    fn linux_cpuinfo_parser_preserves_result_contract() {
        let counts = num_cores_and_logical_procs();
        assert!(counts.logical >= 0);
        if counts.physical > 0 {
            assert!(counts.logical > 0);
        }
    }
}
