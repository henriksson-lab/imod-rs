//! The tolerance mode of `tests/common/golden.rs` (`PORTABILITY.md`): golden
//! replays off Linux x86_64 accept floating-point differences within
//! `golden::REL_TOL`.  These tests run everywhere, so the comparison itself is
//! checked on Linux x86_64 too, on synthetic MRC files and text.

mod common;

use common::golden::{REL_TOL, approx_equal, approx_fingerprint, fingerprint};

/// A float (mode 2) MRC file of `values`, `nx` by 1 by 1, with the header
/// statistics set from the values.
fn mrc(values: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; 1024];
    let put = |bytes: &mut Vec<u8>, at: usize, v: [u8; 4]| bytes[at..at + 4].copy_from_slice(&v);
    put(&mut bytes, 0, (values.len() as i32).to_le_bytes());
    put(&mut bytes, 4, 1i32.to_le_bytes());
    put(&mut bytes, 8, 1i32.to_le_bytes());
    put(&mut bytes, 12, 2i32.to_le_bytes());
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    put(&mut bytes, 76, min.to_le_bytes());
    put(&mut bytes, 80, max.to_le_bytes());
    put(&mut bytes, 84, mean.to_le_bytes());
    bytes[208..212].copy_from_slice(b"MAP ");
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|k| (k as f32 * 0.37).sin() * scale + 10.0)
        .collect()
}

#[test]
fn mrc_values_within_the_tolerance_pass_and_beyond_it_fail() {
    let base = ramp(4096, 50.0);
    let expected = mrc(&base);
    // One ulp of difference everywhere: what another libm gives
    let ulp: Vec<f32> = base
        .iter()
        .map(|v| f32::from_bits(v.to_bits() + 1))
        .collect();
    assert_eq!(approx_equal(&expected, &mrc(&ulp)), Ok(()));
    let print = fingerprint(&expected).unwrap();
    assert_eq!(approx_fingerprint(&print, &mrc(&ulp)), Ok(()));
    // A real defect: one region shifted by far more than the tolerance
    let mut wrong = base.clone();
    for v in &mut wrong[1000..1100] {
        *v += 1.0;
    }
    assert!(approx_equal(&expected, &mrc(&wrong)).is_err());
    assert!(approx_fingerprint(&print, &mrc(&wrong)).is_err());
    // A single value off by 100 times the tolerance
    let mut one = base.clone();
    one[7] *= 1.0 + 100.0 * REL_TOL as f32;
    assert!(approx_equal(&expected, &mrc(&one)).is_err());
    // A different size is a structural difference
    assert!(approx_equal(&expected, &mrc(&base[..4000])).is_err());
}

#[test]
fn text_numbers_compare_within_the_tolerance_or_a_last_digit() {
    let expected = b"Mean = 1.23456 at 12 points\n  0.5000  -3.25E+02 done\n";
    // Last printed digit flipped, and a float differing in the 6th digit
    let ours = b"Mean = 1.23457 at 12 points\n  0.4999  -3.25E+02 done\n";
    assert_eq!(approx_equal(expected, ours), Ok(()));
    // Integers must match exactly; so must the words
    assert!(
        approx_equal(
            expected,
            b"Mean = 1.23456 at 13 points\n  0.5000  -3.25E+02 done\n"
        )
        .is_err()
    );
    assert!(
        approx_equal(
            expected,
            b"Mean = 1.23456 at 12 pixels\n  0.5000  -3.25E+02 done\n"
        )
        .is_err()
    );
    // Beyond both the relative tolerance and a last-digit flip
    assert!(
        approx_equal(
            expected,
            b"Mean = 1.23656 at 12 points\n  0.5000  -3.25E+02 done\n"
        )
        .is_err()
    );
    // The same through a fingerprint
    let print = fingerprint(expected).unwrap();
    assert_eq!(approx_fingerprint(&print, ours), Ok(()));
    assert!(
        approx_fingerprint(
            &print,
            b"Mean = 1.33456 at 12 points\n  0.5000  -3.25E+02 done\n"
        )
        .is_err()
    );
}
