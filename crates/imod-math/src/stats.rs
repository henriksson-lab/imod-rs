/// Compute mean of a slice.
pub fn mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    (sum / data.len() as f64) as f32
}

/// Compute mean and standard deviation.
pub fn mean_sd(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n < 2 {
        return (mean(data), 0.0);
    }
    let n_f64 = n as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &x in data {
        let v = x as f64;
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n_f64;
    let variance = (sum_sq - sum * sum / n_f64) / (n_f64 - 1.0);
    (mean as f32, variance.max(0.0).sqrt() as f32)
}

/// Compute min, max, mean of a slice.
pub fn min_max_mean(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0_f64;
    for &x in data {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
    }
    (min, max, (sum / data.len() as f64) as f32)
}

/// Compute min, max, mean, and standard deviation.
pub fn min_max_mean_sd(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = data.len() as f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &x in data {
        let v = x as f64;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let sd = if data.len() > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    (min, max, mean as f32, sd as f32)
}

/// Robust statistics: median and normalized median absolute deviation (MADN).
/// MADN = MAD / 0.6745, which estimates the standard deviation for normal data.
pub fn robust_stat(data: &mut [f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = percentile_sorted(data, 0.5);

    // Compute absolute deviations from median
    let mut devs: Vec<f32> = data.iter().map(|&x| (x - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = percentile_sorted(&devs, 0.5);
    let madn = mad / 0.6745;

    (median, madn)
}

/// Sample mean and SD by reading every `sample_step`-th element.
/// Used for quick statistics on large images.
pub fn sample_mean_sd(data: &[f32], sample_step: usize) -> (f32, f32) {
    let step = sample_step.max(1);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0u64;
    let mut i = 0;
    while i < data.len() {
        let v = data[i] as f64;
        sum += v;
        sum_sq += v * v;
        count += 1;
        i += step;
    }
    if count < 2 {
        return ((sum as f32), 0.0);
    }
    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq - sum * sum / n) / (n - 1.0);
    (mean as f32, variance.max(0.0).sqrt() as f32)
}

/// Percentile of a pre-sorted slice (frac in 0..1).
fn percentile_sorted(sorted: &[f32], frac: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = frac * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    if hi >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    let t = pos - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// Linear regression: fit y = a + b*x.
/// Returns (intercept, slope, correlation_coefficient).
pub fn linear_regression(x: &[f32], y: &[f32]) -> Option<(f32, f32, f32)> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return None;
    }
    let n_f64 = n as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let xi = x[i] as f64;
        let yi = y[i] as f64;
        sx += xi;
        sy += yi;
        sxx += xi * xi;
        syy += yi * yi;
        sxy += xi * yi;
    }
    let denom = n_f64 * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return None;
    }
    let slope = (n_f64 * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n_f64;

    let var_x = sxx / n_f64 - (sx / n_f64).powi(2);
    let var_y = syy / n_f64 - (sy / n_f64).powi(2);
    let r = if var_x > 0.0 && var_y > 0.0 {
        let cov = sxy / n_f64 - (sx / n_f64) * (sy / n_f64);
        (cov / (var_x * var_y).sqrt()) as f32
    } else {
        0.0
    };

    Some((intercept as f32, slope as f32, r))
}
