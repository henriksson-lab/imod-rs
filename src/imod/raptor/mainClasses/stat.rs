//! Translation of `IMOD/raptor/mainClasses/stat.h` and `stat.cpp`.

/// `average(float* f, int size)` (`stat.cpp:23`).
pub fn average(f: &[f32], size: i32) -> f64 {
    let mut ans = 0.0f64;
    for i in 0..size as usize {
        ans += f[i] as f64;
    }
    ans / size as f64
}

/// `meanAndVariance(float* f, int size, float* average, float* variance)`
/// (`stat.cpp:30`).  The sums are `float`; the divisions are `double`
/// and narrow on assignment.
pub fn mean_and_variance(f: &[f32], size: i32, average: &mut f32, variance: &mut f32) {
    let mut q2 = 0.0f32;
    let mut q = 0.0f32;
    for i in 0..size as usize {
        q += f[i];
        q2 += f[i] * f[i];
    }
    *average = (q as f64 / size as f64) as f32;
    // `q2-size*(*average)*(*average)` is all `float`: `size` converts to
    // `float` for the product; only the division is `double`.
    *variance = ((q2 - size as f32 * *average * *average) as f64 / (size - 1) as f64) as f32;
}

/// `max(float* f, int size)` (`stat.cpp:42`).
pub fn max(f: &[f32], size: i32) -> f32 {
    let mut ans = -1E+37f32;
    for i in 0..size as usize {
        if f[i] > ans {
            ans = f[i];
        }
    }
    ans
}
