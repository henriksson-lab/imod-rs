//! Translation of `IMOD/librgctf/functions.{h,cpp}` — the IMOD version of
//! cisTEM's `functions.cpp`, which carries renamed copies of three `b3dutil.c`
//! routines so that `libctffind` does not depend on `libcfshr`.

use std::cmp::Ordering;
use std::io::Read;
use std::sync::Mutex;

use crate::imod::libcfshr::b3dutil::{CArg, c_format};

use super::defines::PI;

/// C++ inline `IsEven` (`functions.h:7`).
pub fn is_even(number_to_check: i32) -> bool {
    number_to_check % 2 == 0
}

/// C++ inline `deg_2_rad` (`functions.h:13`).
///
/// `PI` is a `double` macro, so the multiply and the divide happen in double
/// and only the return narrows to `float`.
pub fn deg_2_rad(degrees: f32) -> f32 {
    (f64::from(degrees) * PI / 180.) as f32
}

/// C++ inline `myroundint(double)` (`functions.h:18`).
pub fn myroundint(a: f64) -> i32 {
    if a > 0.0 {
        (a + 0.5) as i32
    } else {
        (a - 0.5) as i32
    }
}

/// C++ inline `myroundint(float)` (`functions.h:23`).
pub fn myroundint_float(a: f32) -> i32 {
    if a > 0.0 {
        (a + 0.5) as i32
    } else {
        (a - 0.5) as i32
    }
}

/// C++ `rankSort` (`functions.cpp:24`).
///
/// `std::sort` over `std::pair<float,size_t>` orders by the value and then by
/// the original index, so the comparison is total and the result does not
/// depend on the sort's stability.
pub fn rank_sort(v_temp: &[f32]) -> Vec<usize> {
    let mut v_sort: Vec<(f32, usize)> = vec![(0.0, 0); v_temp.len()];

    for i in 0..v_sort.len() {
        v_sort[i] = (v_temp[i], i);
    }

    v_sort.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
        Some(Ordering::Equal) | None => a.1.cmp(&b.1),
        Some(order) => order,
    });

    // The source declares `std::pair<double, size_t> rank;`, whose default
    // constructor value-initialises both members to zero — so the first
    // element only starts a new rank when its value differs from 0.0.
    let mut rank: (f64, usize) = (0.0, 0);
    let mut result: Vec<usize> = vec![0; v_temp.len()];

    for i in 0..v_sort.len() {
        if f64::from(v_sort[i].0) != rank.0 {
            rank = (f64::from(v_sort[i].0), i);
        }
        result[v_sort[i].1] = rank.1;
    }
    result
}

/// The Rust form of C++ `CharArgType` (`functions.h:29`).
pub type CharArgType = fn(&str);

/// C++ `static CharArgType sPrintFunc` (`functions.cpp:46`).
static S_PRINT_FUNC: Mutex<Option<CharArgType>> = Mutex::new(None);

/// C++ `internalSetPrintFunc` (`functions.cpp:48`).
pub fn internal_set_print_func(func: Option<CharArgType>) {
    *S_PRINT_FUNC.lock().expect("print callback lock poisoned") = func;
}

/// C++ `wxPrintf` (`functions.cpp:54`).
///
/// The source formats with `vsprintf` into a 512-byte buffer and then either
/// hands the text to the registered callback or writes it with
/// `printf("%s", …)`.  The variadic half of that signature is the call site's;
/// callers here format with [`wx_printf_fmt`] or pass an already-built string.
pub fn wx_printf(error_mess: &str) {
    let func = *S_PRINT_FUNC.lock().expect("print callback lock poisoned");
    if let Some(func) = func {
        func(error_mess);
    } else {
        print!("{error_mess}");
    }
}

/// C++ `wxPrintf`'s variadic call form, with the C library's own conversions.
///
/// Rust's `{}` is not C's `%f`/`%g`, so the format string and its arguments go
/// through the tree's C-format writer before reaching [`wx_printf`].
pub fn wx_printf_fmt(format: &str, args: &[CArg]) {
    wx_printf(&c_format(format, args));
}

/// C++ `#define CPUINFO_LINE 80` (`functions.cpp:72`).
const CPUINFO_LINE: i32 = 80;
/// C++ `#define MAX_CPU_SOCKETS 64` (`functions.cpp:73`).
const MAX_CPU_SOCKETS: usize = 64;

/// C++ `static int fgetline(FILE *, char [], int)` (`functions.cpp:74`), the
/// renamed copy of `b3dutil.c`'s `fgetline`.
///
/// The unusual EOF convention is the source's: EOF after `length` bytes is
/// reported as `-(length + 2)`.
pub fn fgetline(fp: Option<&mut dyn Read>, s: &mut [u8], limit: i32) -> i32 {
    let Some(fp) = fp else {
        return -1;
    };

    if limit < 3 {
        return -1;
    }

    let mut c: Option<u8> = None;
    let mut i = 0usize;
    loop {
        let mut byte = [0u8; 1];
        match fp.read(&mut byte) {
            Ok(1) => c = Some(byte[0]),
            _ => {
                c = None;
                break;
            }
        }
        if !(i < (limit as usize - 1)) || c == Some(b'\n') {
            break;
        }
        s[i] = byte[0];
        i += 1;
    }

    /* 1/25/12: Take off a return too! */
    if i > 0 && s[i - 1] == b'\r' {
        i -= 1;
    }

    s[i] = 0;
    let length = i as i32;

    if c.is_none() {
        -1 * (length + 2)
    } else {
        length
    }
}

/// C++ `static int numCoresAndLogicalProcs(int *, int *)` (`functions.cpp:112`),
/// the Linux `/proc/cpuinfo` branch of the renamed `b3dutil.c` copy.
pub fn num_cores_and_logical_procs(physical: &mut i32, logical: &mut i32) -> i32 {
    let mut processor_core_count = 0i32;
    let mut logical_processor_count = 0i32;

    let mut socket_flags = [0u8; MAX_CPU_SOCKETS];
    let mut linebuf = [0u8; CPUINFO_LINE as usize];
    let mut err;
    let (mut len, mut cur_id, mut cur_cores): (i32, i32, i32);

    /* Linux: look at /proc/cpuinfo */
    if let Ok(file) = std::fs::File::open("/proc/cpuinfo") {
        let mut fp = std::io::BufReader::new(file);
        cur_id = -1;
        cur_cores = -1;
        socket_flags.fill(0);
        loop {
            err = 0;
            len = fgetline(Some(&mut fp), &mut linebuf, CPUINFO_LINE);
            if len == 0 {
                continue;
            }
            if len == -2 {
                break;
            }
            err = 1;
            if len == -1 {
                break;
            }

            let text = String::from_utf8_lossy(
                &linebuf[..linebuf.iter().position(|&b| b == 0).unwrap_or(0)],
            )
            .into_owned();

            /* Look for a "physical id :" and a "cpu cores :" in either order */
            if text.contains("physical id") {
                /* Error if already got a physical id without cpu cores */
                if cur_id >= 0 {
                    break;
                }
                let colon = text.find(':');
                if let Some(colon) = colon {
                    cur_id = atoi(&text[colon + 1..]);
                }

                /* Error if no colon or ID out of range */
                if colon.is_none() || cur_id < 0 || cur_id as usize >= MAX_CPU_SOCKETS {
                    break;
                }
            }
            if text.contains("cpu cores") {
                /* Error if already got a cpu cores without physical id  */
                if cur_cores >= 0 {
                    break;
                }
                let colon = text.find(':');
                if let Some(colon) = colon {
                    cur_cores = atoi(&text[colon + 1..]);
                }

                /* Error if no colon or core count illegal */
                if colon.is_none() || cur_cores <= 0 {
                    break;
                }
            }

            if cur_id >= 0 && cur_cores > 0 {
                logical_processor_count += 1;
                if socket_flags[cur_id as usize] == 0 {
                    processor_core_count += cur_cores;
                }
                socket_flags[cur_id as usize] = 1;
                cur_id = -1;
                cur_cores = -1;
            }
            err = 0;
            if len < 0 {
                break;
            }
        }
        if err != 0 {
            processor_core_count *= -1;
        }
    }
    *physical = processor_core_count;
    *logical = logical_processor_count;
    i32::from(processor_core_count <= 0 || logical_processor_count < 0)
}

/// C `atoi` over the tail of a `/proc/cpuinfo` field: leading blanks, an
/// optional sign, then as many digits as there are.
fn atoi(text: &str) -> i32 {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    let mut sign = 1i64;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let mut value: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = value * 10 + i64::from(bytes[i] - b'0');
        if value > i64::from(i32::MAX) + 1 {
            value = i64::from(i32::MAX) + 1;
        }
        i += 1;
    }
    (sign * value) as i32
}

/// The three function-level `static` variables of `ctfNumOMPthreads`
/// (`functions.cpp:260-263`), which are initialised once and then reused.
struct OmpThreadState {
    lim_threads: i32,
    num_procs: i32,
    force_threads: i32,
    omp_num_procs: i32,
}

static OMP_THREAD_STATE: Mutex<OmpThreadState> = Mutex::new(OmpThreadState {
    lim_threads: -1,
    num_procs: -1,
    force_threads: -1,
    omp_num_procs: -1,
});

/// C++ `ctfNumOMPthreads` (`functions.cpp:253`).
///
/// `functions.cpp` is compiled with `$(OPENMP)`, so this is the `_OPENMP`
/// branch.  `omp_get_num_procs()` is the count of processors available to the
/// process, which is what `std::thread::available_parallelism` reports here.
pub fn ctf_num_omp_threads(optimal_threads: i32) -> i32 {
    let mut num_threads = optimal_threads;
    let mut physical_procs = 0i32;
    let mut logical_processor_count = 0i32;
    let mut processor_core_count = 0i32;
    let mut state = OMP_THREAD_STATE.lock().expect("thread state lock poisoned");

    /* One-time determination of number of physical and logical cores */
    if state.num_procs < 0 {
        state.num_procs = std::thread::available_parallelism().map_or(1, |n| n.get() as i32);
        state.omp_num_procs = state.num_procs;

        /* if there are legal numbers and the logical count is the OMP
        number, set the physical processor count */
        if num_cores_and_logical_procs(&mut processor_core_count, &mut logical_processor_count) == 0
            && processor_core_count > 0
            && logical_processor_count == state.num_procs
        {
            physical_procs = processor_core_count;
        }
        if std::env::var_os("IMOD_REPORT_CORES").is_some() {
            print!(
                "{}",
                c_format(
                    "core count = %d  logical processors = %d  OMP num = %d => physical \
processors = %d\n",
                    &[
                        CArg::Int(i64::from(processor_core_count)),
                        CArg::Int(i64::from(logical_processor_count)),
                        CArg::Int(i64::from(state.num_procs)),
                        CArg::Int(i64::from(physical_procs)),
                    ],
                )
            );
        }
        use std::io::Write;
        let _ = std::io::stdout().flush();

        if physical_procs > 0 {
            state.num_procs = state.num_procs.min(physical_procs);
        }
    }

    /* Limit by number of real cores */
    num_threads = 1.max(state.num_procs.min(num_threads));

    /* One-time determination of the limit set by OMP_NUM_THREADS */
    if state.lim_threads < 0 {
        if let Some(omp_num) = std::env::var_os("OMP_NUM_THREADS") {
            state.lim_threads = atoi(&omp_num.to_string_lossy());
        }
        state.lim_threads = 0.max(state.lim_threads);
    }

    /* Limit to number set by OMP_NUM_THREADS and to number of real cores */
    if state.lim_threads > 0 {
        num_threads = state.lim_threads.min(num_threads);
    }

    /* One-time determination of whether user wants to force a number of threads */
    if state.force_threads < 0 {
        state.force_threads = 0;
        if let Some(omp_num) = std::env::var_os("IMOD_FORCE_OMP_THREADS") {
            let omp_num = omp_num.to_string_lossy().into_owned();
            if omp_num == "ALL_CORES" {
                if state.num_procs > 0 {
                    state.force_threads = state.num_procs;
                }
            } else if omp_num == "ALL_HYPER" {
                if state.omp_num_procs > 0 {
                    state.force_threads = state.omp_num_procs;
                }
            } else {
                state.force_threads = atoi(&omp_num);
                state.force_threads = 0.max(state.force_threads);
            }
        }
    }

    /* Force the number if set */
    if state.force_threads > 0 {
        num_threads = state.force_threads;
    }

    if std::env::var_os("IMOD_REPORT_CORES").is_some() {
        print!(
            "{}",
            c_format(
                "numProcs %d  limThreads %d  numThreads %d\n",
                &[
                    CArg::Int(i64::from(state.num_procs)),
                    CArg::Int(i64::from(state.lim_threads)),
                    CArg::Int(i64::from(num_threads)),
                ],
            )
        );
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    num_threads
}

/// C++ `ctfOMPthreadNum` (`functions.cpp:333`).
///
/// This translation runs the library's parallel loops sequentially, so the
/// caller is always the source's thread 0.
pub fn ctf_omp_thread_num() -> i32 {
    0
}

/// C++ `ctfWallTime` (`functions.cpp:346`), the `gettimeofday` branch.
pub fn ctf_wall_time() -> f64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before Unix epoch");
    now.as_secs() as f64 + f64::from(now.subsec_micros()) / 1000000.
}

#[cfg(test)]
mod tests {
    use super::{deg_2_rad, fgetline, is_even, myroundint, myroundint_float, rank_sort};
    use std::io::Cursor;

    #[test]
    fn scalar_and_rank_helpers_match_source_rules() {
        assert!(is_even(-4));
        assert!(!is_even(3));
        assert_eq!(deg_2_rad(180.0), (super::PI) as f32);
        assert_eq!(myroundint_float(1.5), 2);
        assert_eq!(myroundint(-1.5), -2);
        assert_eq!(rank_sort(&[3.0, 1.0, 1.0, 2.0]), vec![3, 0, 0, 2]);
    }

    #[test]
    fn local_fgetline_preserves_crlf_and_eof_status() {
        let mut input = Cursor::new(b"one\r\ntwo".to_vec());
        let mut line = [0u8; 8];
        assert_eq!(fgetline(Some(&mut input), &mut line, 8), 3);
        assert_eq!(&line[..4], b"one\0");
        assert_eq!(fgetline(Some(&mut input), &mut line, 8), -5);
        assert_eq!(&line[..4], b"two\0");
    }
}
