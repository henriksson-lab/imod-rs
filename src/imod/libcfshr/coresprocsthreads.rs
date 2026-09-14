//! Translation of `IMOD/libcfshr/coresprocsthreads.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use core::sync::atomic::{AtomicI32, Ordering};

use super::b3dutil::{ImodFile, fgetline};

const CPUINFO_LINE: i32 = 80;
const MAX_CPU_SOCKETS: usize = 64;
static S_CPU_IS_AMD: AtomicI32 = AtomicI32::new(-1);

/// C `numCoresAndLogicalProcs` (`coresprocsthreads.c:40`).
///
/// The crate is built without `_OPENMP`; this is the Linux `/proc/cpuinfo`
/// branch selected by the vendored source on the current target.
pub unsafe fn num_cores_and_logical_procs(physical: *mut i32, logical: *mut i32) -> i32 {
    let mut processor_core_count = 0;
    let mut logical_processor_count = 0;
    let filename = "/proc/cpuinfo";
    let mode = "r";
    let file = ImodFile::open(filename, mode);
    if let Some(mut file) = file {
        let mut socket_flags = [0_u8; MAX_CPU_SOCKETS];
        let mut line = [0_u8; CPUINFO_LINE as usize];
        let mut current_id = -1;
        let mut current_cores = -1;
        let mut error = 0;
        loop {
            error = 0;
            let length = fgetline(&mut file, &mut line, CPUINFO_LINE);
            if length == 0 {
                continue;
            }
            if length == -2 {
                break;
            }
            error = 1;
            if length == -1 {
                break;
            }
            if S_CPU_IS_AMD.load(Ordering::SeqCst) < 0
                && !libc::strstr(line.as_ptr().cast(), c"vendor_id".as_ptr()).is_null()
            {
                S_CPU_IS_AMD.store(
                    (!libc::strstr(line.as_ptr().cast(), c"AuthenticAMD".as_ptr()).is_null())
                        as i32,
                    Ordering::SeqCst,
                );
            }
            if !libc::strstr(line.as_ptr().cast(), c"physical id".as_ptr()).is_null() {
                if current_id >= 0 {
                    break;
                }
                let colon = libc::strchr(line.as_ptr().cast(), b':' as i32);
                if !colon.is_null() {
                    current_id = libc::atoi(colon.add(1));
                }
                if colon.is_null() || current_id < 0 || current_id as usize >= MAX_CPU_SOCKETS {
                    break;
                }
            }
            if !libc::strstr(line.as_ptr().cast(), c"cpu cores".as_ptr()).is_null() {
                if current_cores >= 0 {
                    break;
                }
                let colon = libc::strchr(line.as_ptr().cast(), b':' as i32);
                if !colon.is_null() {
                    current_cores = libc::atoi(colon.add(1));
                }
                if colon.is_null() || current_cores <= 0 {
                    break;
                }
            }
            if current_id >= 0 && current_cores > 0 {
                logical_processor_count += 1;
                if socket_flags[current_id as usize] == 0 {
                    processor_core_count += current_cores;
                }
                socket_flags[current_id as usize] = 1;
                current_id = -1;
                current_cores = -1;
            }
            error = 0;
            if length < 0 {
                break;
            }
        }
        if error != 0 {
            processor_core_count *= -1;
        }
        // `fclose(file)`: the handle closes when it leaves scope.
    }
    if S_CPU_IS_AMD.load(Ordering::SeqCst) < 0 {
        S_CPU_IS_AMD.store(0, Ordering::SeqCst);
    }
    *physical = processor_core_count;
    *logical = logical_processor_count;
    if processor_core_count <= 0 || logical_processor_count < 0 {
        1
    } else {
        0
    }
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
    let mut value = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    unsafe {
        libc::gettimeofday(&mut value, core::ptr::null_mut());
    }
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.
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
        let mut physical = 0;
        let mut logical = 0;
        let error = unsafe { num_cores_and_logical_procs(&mut physical, &mut logical) };
        assert!(error == 0 || error == 1);
        if error == 0 {
            assert!(physical > 0 && logical >= 0);
        }
    }
}
