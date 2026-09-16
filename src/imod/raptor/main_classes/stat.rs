//! Translation of `IMOD/raptor/mainClasses/stat.{h,cpp}`.

/// C++ `average(float *, int)`.
pub fn average(values: &[f32]) -> f64 {
    values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64
}

/// C++ `meanAndVariance(float *, int, float *, float *)`.
///
/// The source intentionally accumulates in `float` and uses the sample
/// variance denominator (`n - 1`); this preserves both details.
pub fn mean_and_variance(values: &[f32]) -> (f32, f32) {
    let mut q = 0.0_f32;
    let mut q2 = 0.0_f32;
    for &value in values {
        q += value;
        q2 += value * value;
    }
    let mean = q / values.len() as f32;
    // The C denominator is signed `size - 1`; retaining float arithmetic
    // preserves its NaN result for a zero-length caller instead of Rust's
    // `usize` underflow panic.
    let variance = (q2 - values.len() as f32 * mean * mean) / (values.len() as f32 - 1.0);
    (mean, variance)
}

/// C++ `max(float *, int)`.
pub fn max(values: &[f32]) -> f32 {
    values.iter().copied().fold(-1.0e37_f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::{average, max, mean_and_variance};

    #[test]
    fn source_statistics_use_sample_variance() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(average(&values), 2.5);
        assert_eq!(mean_and_variance(&values), (2.5, 5.0 / 3.0));
    }

    #[test]
    fn maximum_keeps_the_source_empty_sentinel() {
        assert_eq!(max(&[-4.0, 8.0, 2.0]), 8.0);
        assert_eq!(max(&[]), -1.0e37);
    }
    #[test]
    fn empty_input_follows_c_floating_point_semantics() {
        let (mean, variance) = mean_and_variance(&[]);
        assert!(mean.is_nan() && variance.is_nan());
        assert!(average(&[]).is_nan());
    }
}
