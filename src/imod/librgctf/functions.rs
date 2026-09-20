//! Translation of `IMOD/librgctf/functions.{h,cpp}`.

use std::cmp::Ordering;
use std::io::Read;
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

/// C++ `ctfNumOMPthreads` in this crate's non-OpenMP configuration.
pub fn ctf_num_omp_threads(_optimal_threads: i32) -> i32 {
    1
}

/// C++ `ctfWallTime`.
pub fn ctf_wall_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before Unix epoch")
        .as_secs_f64()
}

/// C `fgetline` local to `functions.cpp`.
///
/// This copy deliberately preserves its unusual EOF convention: EOF after
/// `length` bytes is reported as `-(length + 2)`.
pub fn fgetline(input: &mut dyn Read, output: &mut [u8], limit: i32) -> i32 {
    if limit < 3 || output.len() < limit as usize {
        return -1;
    }
    let mut length = 0usize;
    let mut eof = false;
    loop {
        let mut byte = [0_u8; 1];
        match input.read(&mut byte) {
            Ok(0) => {
                eof = true;
                break;
            }
            Ok(_) => {}
            Err(_) => {
                eof = true;
                break;
            }
        }
        if length >= limit as usize - 1 || byte[0] == b'\n' {
            break;
        }
        output[length] = byte[0];
        length += 1;
    }
    if length > 0 && output[length - 1] == b'\r' {
        length -= 1;
    }
    output[length] = 0;
    if eof {
        -(length as i32 + 2)
    } else {
        length as i32
    }
}

/// C `numCoresAndLogicalProcs` Linux `/proc/cpuinfo` implementation.
///
/// The physical count is the sum of `cpu cores` once per physical socket;
/// logical is the number of successfully paired socket/core records.
pub fn num_cores_and_logical_procs(physical: &mut i32, logical: &mut i32) -> i32 {
    const MAX_CPU_SOCKETS: usize = 64;
    let mut processor_cores = 0_i32;
    let mut logical_processors = 0_i32;
    if let Ok(mut input) = std::fs::File::open("/proc/cpuinfo") {
        let mut socket_seen = [false; MAX_CPU_SOCKETS];
        let mut line = [0_u8; 80];
        let mut current_id = -1_i32;
        let mut current_cores = -1_i32;
        let mut error = false;
        loop {
            let length = fgetline(&mut input, &mut line, 80);
            if length == 0 {
                continue;
            }
            if length == -2 {
                break;
            }
            if length == -1 {
                error = true;
                break;
            }
            let text = String::from_utf8_lossy(&line[..length.unsigned_abs() as usize]);
            if text.contains("physical id") {
                if current_id >= 0 {
                    error = true;
                    break;
                }
                current_id = text
                    .split_once(':')
                    .and_then(|(_, value)| value.trim().parse().ok())
                    .unwrap_or(-1);
                if current_id < 0 || current_id as usize >= MAX_CPU_SOCKETS {
                    error = true;
                    break;
                }
            }
            if text.contains("cpu cores") {
                if current_cores >= 0 {
                    error = true;
                    break;
                }
                current_cores = text
                    .split_once(':')
                    .and_then(|(_, value)| value.trim().parse().ok())
                    .unwrap_or(-1);
                if current_cores <= 0 {
                    error = true;
                    break;
                }
            }
            if current_id >= 0 && current_cores > 0 {
                logical_processors += 1;
                let socket = current_id as usize;
                if !socket_seen[socket] {
                    processor_cores += current_cores;
                    socket_seen[socket] = true;
                }
                current_id = -1;
                current_cores = -1;
            }
            if length < 0 {
                break;
            }
        }
        if error {
            processor_cores = -processor_cores;
        }
    }
    *physical = processor_cores;
    *logical = logical_processors;
    i32::from(processor_cores <= 0 || logical_processors < 0)
}

#[cfg(test)]
mod tests {
    use super::{RoundInput, ctf_num_omp_threads, deg_2_rad, fgetline, is_even, rank_sort};
    use std::io::Cursor;

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

    #[test]
    fn local_fgetline_preserves_crlf_and_eof_status() {
        let mut input = Cursor::new(b"one\r\ntwo".to_vec());
        let mut line = [0_u8; 8];
        assert_eq!(fgetline(&mut input, &mut line, 8), 3);
        assert_eq!(&line[..4], b"one\0");
        assert_eq!(fgetline(&mut input, &mut line, 8), -5);
        assert_eq!(&line[..4], b"two\0");
    }
}
