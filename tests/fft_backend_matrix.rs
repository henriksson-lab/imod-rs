//! Cross-backend numerical differentials at IMOD's `odfft`/`todfft` execution
//! boundaries (`RUST_NATIVE_BACKENDS_PLAN.md`, Phase 3 items 4-6 and the
//! Phase 4 "RustFFT numerical test" row).
//!
//! Intended backends: every case runs the *same* transform twice, once with
//! `IMOD_RS_FFT_BACKEND=parity` and once with `IMOD_RS_FFT_BACKEND=rustfft`.
//! The selector is a process-local `OnceLock`, so the two runs must be two
//! processes; this suite re-executes its own test binary for the worker.
//!
//! The command-level fixtures live with their commands (`binvol_cli`,
//! `newstack_stream`, `clip_cli`); this suite covers the transform buffers
//! themselves at dimensions no command fixture reaches.
#![cfg(feature = "rustfft-backend")]

// Per-operation tolerances.
//
// Derived by measurement, not assumed: every case below transforms values
// drawn uniformly from [-1, 1], so the transform magnitudes are order 1 and
// these absolute bounds are effectively relative ones.  Over the whole matrix
// in this file the measured parity-versus-RustFFT difference in a forward
// spectrum peaked at 4.2e-7 maximum absolute and 9.6e-8 RMS, and a full
// forward/inverse round trip departed from its own input by at most 7.8e-7
// maximum absolute and 2.1e-7 RMS on either backend -- two to seven ulps of
// `f32` at unit magnitude, which is what a different summation order in a
// single-precision transform costs.  The constants are those measurements
// rounded up by an order of magnitude, which leaves room for another planner's
// radix choices while still failing any structural error (a sign convention, a
// normalization, a transposed axis, a dropped Nyquist term) by orders of
// magnitude.

/// Maximum absolute parity-versus-RustFFT difference in a forward spectrum.
const SPECTRUM_MAXIMUM: f64 = 4.0e-6;
/// RMS parity-versus-RustFFT difference in a forward spectrum.
const SPECTRUM_RMS: f64 = 1.0e-6;
/// Maximum absolute forward/inverse round-trip error, per backend.
const ROUND_TRIP_MAXIMUM: f64 = 8.0e-6;
/// RMS forward/inverse round-trip error, per backend.
const ROUND_TRIP_RMS: f64 = 2.0e-6;

/// Environment variable naming the case the worker must run.
const CASE: &str = "IMOD_RS_FFT_MATRIX_CASE";
/// Environment variable naming the file the worker writes its buffers to.
const OUTPUT: &str = "IMOD_RS_FFT_MATRIX_OUTPUT";

/// Deterministic input for one case, identical in both processes.
///
/// A 32-bit linear congruential sequence keyed by the case dimensions; the
/// values span roughly [-1, 1] so the reported errors are directly comparable
/// between cases.
fn case_input(routine: &str, nx: usize, ny: usize) -> Vec<f32> {
    let mut state = (nx as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(ny as u32)
        .wrapping_mul(40_503)
        .wrapping_add(routine.len() as u32)
        | 1;
    let mut values = vec![
        0.0_f32;
        if routine == "odfft_complex" {
            2 * nx * ny
        } else {
            (nx + 2) * ny
        }
    ];
    for row in 0..ny {
        let (base, count) = if routine == "odfft_complex" {
            (row * 2 * nx, 2 * nx)
        } else {
            (row * (nx + 2), nx)
        };
        for index in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            values[base + index] = (state >> 8) as f32 / 8_388_608.0 - 1.0;
        }
    }
    values
}

/// Runs one case in this process with whatever backend the environment
/// selected, writing the forward buffer followed by the round-trip buffer.
///
/// It is a `#[test]` because the parent re-executes this test binary with a
/// name filter; with no case in the environment it does nothing.
#[test]
fn fft_matrix_worker() {
    let Ok(case) = std::env::var(CASE) else {
        return;
    };
    let fields = case.split(':').collect::<Vec<_>>();
    let routine = fields[0];
    let nx = fields[1].parse::<usize>().unwrap();
    let ny = fields[2].parse::<usize>().unwrap();
    let inverse = fields[3].parse::<i32>().unwrap();
    let mut values = case_input(routine, nx, ny);
    let mut dumped = Vec::<u8>::new();
    unsafe {
        match routine {
            "odfft_real" => {
                imod_rs::imod::libfft::odfft_c(values.as_mut_ptr(), nx as i32, ny as i32, 0);
                dumped.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                imod_rs::imod::libfft::odfft_c(values.as_mut_ptr(), nx as i32, ny as i32, inverse);
            }
            "odfft_complex" => {
                imod_rs::imod::libfft::odfft_c(values.as_mut_ptr(), nx as i32, ny as i32, -1);
                dumped.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                imod_rs::imod::libfft::odfft_c(values.as_mut_ptr(), nx as i32, ny as i32, inverse);
            }
            "todfft" => {
                imod_rs::imod::libfft::todfft_c(values.as_mut_ptr(), nx as i32, ny as i32, 0);
                dumped.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                imod_rs::imod::libfft::todfft_c(values.as_mut_ptr(), nx as i32, ny as i32, inverse);
            }
            other => panic!("unknown case routine {other}"),
        }
    }
    dumped.extend(values.iter().flat_map(|value| value.to_le_bytes()));
    std::fs::write(std::env::var(OUTPUT).unwrap(), dumped).unwrap();
}

/// Runs `fft_matrix_worker` in a child process on the named backend and
/// returns its forward buffer and its round-trip buffer.
fn run_case(
    routine: &str,
    nx: usize,
    ny: usize,
    inverse: i32,
    backend: &str,
) -> (Vec<f32>, Vec<f32>) {
    let output = std::env::temp_dir().join(format!(
        "imod-rs-fft-matrix-{}-{routine}-{nx}x{ny}-{inverse}-{backend}.bin",
        std::process::id()
    ));
    // The child is a libtest process of its own; capture its output rather
    // than letting its "running 1 test" lines interleave with this suite's.
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fft_matrix_worker", "--test-threads=1"])
        .env(CASE, format!("{routine}:{nx}:{ny}:{inverse}"))
        .env(OUTPUT, &output)
        .env("IMOD_RS_FFT_BACKEND", backend)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{backend} worker for {routine} {nx}x{ny}: {}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&output);
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let half = values.len() / 2;
    (values[..half].to_vec(), values[half..].to_vec())
}

/// The real samples of a padded row-major buffer.
///
/// A real `odfft`/`todfft` inverse writes only `nx` samples per row and leaves
/// the two extra floats of the `nx + 2` row as scratch: `hermft` transforms
/// through them, and `newstack.f90:2470` replicates the last real column over
/// them afterwards rather than reading them.  A round-trip comparison
/// therefore covers the `nx` real samples of each row only.
fn data_region(values: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut region = Vec::with_capacity(nx * ny);
    for row in 0..ny {
        region.extend_from_slice(&values[row * (nx + 2)..row * (nx + 2) + nx]);
    }
    region
}

/// Maximum absolute difference and RMS difference between two buffers,
/// together with the maximum magnitude of the first one.
fn differences(left: &[f32], right: &[f32]) -> (f64, f64, f64) {
    assert_eq!(left.len(), right.len());
    let mut maximum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    let mut scale = 0.0_f64;
    for (one, other) in left.iter().zip(right.iter()) {
        let difference = (*one as f64 - *other as f64).abs();
        maximum = maximum.max(difference);
        sum_squares += difference * difference;
        scale = scale.max((*one as f64).abs());
    }
    (maximum, (sum_squares / left.len() as f64).sqrt(), scale)
}

/// `odfft` real-to-complex forward and complex-to-real inverse over sizes with
/// radix-2/3/5 and the general prime-factor kernel (`38 = 2 * 19`).
///
/// Both backends transform the same deterministic input; the forward spectra
/// are compared to each other and each round trip to the original samples.
#[test]
fn odfft_real_transforms_match_parity_across_mixed_radix_sizes() {
    for (nx, ny) in [(30, 7), (64, 48), (100, 7), (38, 3), (18, 1), (240, 5)] {
        let input = case_input("odfft_real", nx, ny);
        let (parity_forward, parity_back) = run_case("odfft_real", nx, ny, 1, "parity");
        let (rustfft_forward, rustfft_back) = run_case("odfft_real", nx, ny, 1, "rustfft");
        let (maximum, rms, scale) = differences(&parity_forward, &rustfft_forward);
        let (parity_maximum, parity_rms, _) = differences(
            &data_region(&input, nx, ny),
            &data_region(&parity_back, nx, ny),
        );
        let (rustfft_maximum, rustfft_rms, _) = differences(
            &data_region(&input, nx, ny),
            &data_region(&rustfft_back, nx, ny),
        );
        eprintln!(
            "odfft_real {nx}x{ny}: spectrum max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}; \
             round trip parity max {parity_maximum:.3e} rms {parity_rms:.3e}, \
             rustfft max {rustfft_maximum:.3e} rms {rustfft_rms:.3e}"
        );
        assert!(
            maximum < SPECTRUM_MAXIMUM,
            "odfft_real {nx}x{ny} spectrum max {maximum}"
        );
        assert!(
            rms < SPECTRUM_RMS,
            "odfft_real {nx}x{ny} spectrum rms {rms}"
        );
        assert!(
            rustfft_maximum < ROUND_TRIP_MAXIMUM && rustfft_rms < ROUND_TRIP_RMS,
            "odfft_real {nx}x{ny} rustfft round trip max {rustfft_maximum} rms {rustfft_rms}"
        );
        assert!(
            parity_maximum < ROUND_TRIP_MAXIMUM && parity_rms < ROUND_TRIP_RMS,
            "odfft_real {nx}x{ny} parity round trip max {parity_maximum} rms {parity_rms}"
        );
    }
}

/// `odfft` complex-to-complex, the direction pair `thrdfft` and `clip fft -3d`
/// use, at odd, prime and composite lengths the real path cannot take.
#[test]
fn odfft_complex_transforms_match_parity_across_odd_and_prime_sizes() {
    for (nx, ny) in [(19, 4), (45, 3), (7, 11), (30, 2), (17, 17), (135, 2)] {
        let input = case_input("odfft_complex", nx, ny);
        let (parity_forward, parity_back) = run_case("odfft_complex", nx, ny, -2, "parity");
        let (rustfft_forward, rustfft_back) = run_case("odfft_complex", nx, ny, -2, "rustfft");
        let (maximum, rms, scale) = differences(&parity_forward, &rustfft_forward);
        let (parity_maximum, parity_rms, _) = differences(&input, &parity_back);
        let (rustfft_maximum, rustfft_rms, _) = differences(&input, &rustfft_back);
        eprintln!(
            "odfft_complex {nx}x{ny}: spectrum max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}; \
             round trip parity max {parity_maximum:.3e} rms {parity_rms:.3e}, \
             rustfft max {rustfft_maximum:.3e} rms {rustfft_rms:.3e}"
        );
        assert!(
            maximum < SPECTRUM_MAXIMUM,
            "odfft_complex {nx}x{ny} spectrum max {maximum}"
        );
        assert!(
            rms < SPECTRUM_RMS,
            "odfft_complex {nx}x{ny} spectrum rms {rms}"
        );
        assert!(
            rustfft_maximum < ROUND_TRIP_MAXIMUM && rustfft_rms < ROUND_TRIP_RMS,
            "odfft_complex {nx}x{ny} rustfft round trip max {rustfft_maximum} rms {rustfft_rms}"
        );
        assert!(
            parity_maximum < ROUND_TRIP_MAXIMUM && parity_rms < ROUND_TRIP_RMS,
            "odfft_complex {nx}x{ny} parity round trip max {parity_maximum} rms {parity_rms}"
        );
    }
}

/// `todfft` packed two-dimensional forward and inverse, including strongly
/// rectangular shapes and a Y length that is prime.
#[test]
fn todfft_transforms_match_parity_across_mixed_radix_sizes() {
    for (nx, ny) in [(30, 45), (64, 48), (100, 7), (38, 19), (18, 17), (216, 5)] {
        let input = case_input("todfft", nx, ny);
        let (parity_forward, parity_back) = run_case("todfft", nx, ny, 1, "parity");
        let (rustfft_forward, rustfft_back) = run_case("todfft", nx, ny, 1, "rustfft");
        let (maximum, rms, scale) = differences(&parity_forward, &rustfft_forward);
        let (parity_maximum, parity_rms, _) = differences(
            &data_region(&input, nx, ny),
            &data_region(&parity_back, nx, ny),
        );
        let (rustfft_maximum, rustfft_rms, _) = differences(
            &data_region(&input, nx, ny),
            &data_region(&rustfft_back, nx, ny),
        );
        eprintln!(
            "todfft {nx}x{ny}: spectrum max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}; \
             round trip parity max {parity_maximum:.3e} rms {parity_rms:.3e}, \
             rustfft max {rustfft_maximum:.3e} rms {rustfft_rms:.3e}"
        );
        assert!(
            maximum < SPECTRUM_MAXIMUM,
            "todfft {nx}x{ny} spectrum max {maximum}"
        );
        assert!(rms < SPECTRUM_RMS, "todfft {nx}x{ny} spectrum rms {rms}");
        assert!(
            rustfft_maximum < ROUND_TRIP_MAXIMUM && rustfft_rms < ROUND_TRIP_RMS,
            "todfft {nx}x{ny} rustfft round trip max {rustfft_maximum} rms {rustfft_rms}"
        );
        assert!(
            parity_maximum < ROUND_TRIP_MAXIMUM && parity_rms < ROUND_TRIP_RMS,
            "todfft {nx}x{ny} parity round trip max {parity_maximum} rms {parity_rms}"
        );
    }
}

/// `todfft.c:55` documents `1 or -1` as the same inverse transform, and
/// `clip_fftvol` (`clip/fft.cpp:264`) is the caller that passes -1.  Both
/// backends must treat the two directions identically.
#[test]
fn todfft_negative_direction_inverts_like_the_positive_direction() {
    for backend in ["parity", "rustfft"] {
        for (nx, ny) in [(30, 45), (18, 17)] {
            let (_, positive) = run_case("todfft", nx, ny, 1, backend);
            let (_, negative) = run_case("todfft", nx, ny, -1, backend);
            assert_eq!(positive, negative, "{backend} todfft {nx}x{ny} idir -1");
        }
    }
}
