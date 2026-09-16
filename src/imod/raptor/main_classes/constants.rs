//! Translation of `IMOD/raptor/mainClasses/constants.{h,cpp}`.

/// C++ `PI`.
pub const PI: f64 = 3.141_592_653_589_79;
/// C++ `maxJumps`.
pub const MAX_JUMPS: u32 = 2;
/// C++ `initialPairwiseScore`.
pub const INITIAL_PAIRWISE_SCORE: f64 = 999.0;
/// C++ `percentile`.
pub const PERCENTILE: f32 = 0.7;
/// C++ `minWeight`.
pub const MIN_WEIGHT: f64 = 1.0e-4;
/// C++ `delta`.
pub const DELTA: f64 = 5.0;
/// C++ `tol`.
pub const TOL: f64 = 0.1;
/// C++ `sparseMatrixZeroVal`, specifically `numeric_limits<double>::min()`.
pub const SPARSE_MATRIX_ZERO_VAL: f64 = f64::MIN_POSITIVE;
/// C++ `peakThresholdFillContours`.
pub const PEAK_THRESHOLD_FILL_CONTOURS: f64 = 0.45;
/// C++ `maxTargetsNextFrame`.
pub const MAX_TARGETS_NEXT_FRAME: i32 = 180;
/// C++ `maxTargetsPrevFrame`.
pub const MAX_TARGETS_PREV_FRAME: i32 = 80;

/// C++ `getDate`, using the process-local time representation equivalent to
/// C `strftime("%c")` rather than retaining C time buffers.
pub fn get_date() -> String {
    chrono::Local::now().format("%c").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_constants_have_exact_values() {
        assert_eq!(PI, 3.141_592_653_589_79);
        assert_eq!(
            (MAX_JUMPS, INITIAL_PAIRWISE_SCORE, PERCENTILE),
            (2, 999.0, 0.7)
        );
        assert_eq!((MIN_WEIGHT, DELTA, TOL), (1.0e-4, 5.0, 0.1));
        assert_eq!(SPARSE_MATRIX_ZERO_VAL, f64::MIN_POSITIVE);
        assert_eq!(
            (
                PEAK_THRESHOLD_FILL_CONTOURS,
                MAX_TARGETS_NEXT_FRAME,
                MAX_TARGETS_PREV_FRAME
            ),
            (0.45, 180, 80)
        );
    }
}
