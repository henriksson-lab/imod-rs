//! Translation of `IMOD/raptor/mainClasses/constants.h` and `constants.cpp`.

/// `PI`.
pub const PI: f64 = 3.14159265358979;

/// `maxJumps`.
pub const MAX_JUMPS: u32 = 2;

/// `initialPairwiseScore`.
pub const INITIAL_PAIRWISE_SCORE: f64 = 999.0;

/// `percentile`: percentile to display Mean Squared Residual error without
/// outliers during optimization procedure.
pub const PERCENTILE: f32 = 0.7;

/// `minWeight`: to avoid numerical error when trajectories have missing
/// markers.  We set it to very low score.
pub const MIN_WEIGHT: f64 = 1e-4;

/// `delta`: Huber penalty constant (step between quadratic to linear error).
pub const DELTA: f64 = 5.0;

/// `tol`.
pub const TOL: f64 = 0.1;

/// `sparseMatrixZeroVal = numeric_limits<double>::min()`: below this value
/// we drop elements in sparse matrices.
pub const SPARSE_MATRIX_ZERO_VAL: f64 = f64::MIN_POSITIVE;

/// `peakThresholdFillContours`.
pub const PEAK_THRESHOLD_FILL_CONTOURS: f64 = 0.45;

/// `maxTargetsNextFrame`: maximum number of candidates.
pub const MAX_TARGETS_NEXT_FRAME: i32 = 180;

/// `maxTargetsPrevFrame`: maximum number of targets if we use automatic
/// number of marker estimation.
pub const MAX_TARGETS_PREV_FRAME: i32 = 80;

/// `getDate()` (`constants.cpp:15`): `strftime(buffer, 80, "%c",
/// localtime(time))`.  The program never calls `setlocale`, so `%c` is the
/// C locale's `"%a %b %e %H:%M:%S %Y"`, formatted here; `localtime_r` is the
/// operating-system service that knows the time zone.
pub fn get_date() -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let rawtime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0);
    let timeinfo = crate::imod::libcfshr::b3dutil::local_time(rawtime)
        .unwrap_or_else(|| unsafe { std::mem::zeroed() });
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {}",
        DAYS[timeinfo.tm_wday as usize],
        MONTHS[timeinfo.tm_mon as usize],
        timeinfo.tm_mday,
        timeinfo.tm_hour,
        timeinfo.tm_min,
        timeinfo.tm_sec,
        timeinfo.tm_year + 1900
    )
}
