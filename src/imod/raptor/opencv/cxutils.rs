//! Owned translation of `IMOD/raptor/opencv/cxutils.cpp`.

pub const CV_TERMCRIT_ITER: i32 = 1;
pub const CV_TERMCRIT_EPS: i32 = 2;
pub const CV_C: i32 = 1;
pub const CV_L1: i32 = 2;
pub const CV_L2: i32 = 4;
pub const CV_MINMAX: i32 = 32;

/// C `CvTermCriteria`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CvTermCriteria {
    pub kind: i32,
    pub max_iter: i32,
    pub epsilon: f64,
}
pub fn cv_term_criteria(kind: i32, max_iter: i32, epsilon: f64) -> CvTermCriteria {
    CvTermCriteria {
        kind,
        max_iter,
        epsilon: epsilon as f32 as f64,
    }
}

/// Owned, dense, row-major `CvMat` accepted by this unit.
#[derive(Clone, Debug, PartialEq)]
pub struct CvMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}
impl<T> CvMatrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, CvUtilsError> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(CvUtilsError::UnmatchedSizes);
        }
        Ok(Self { rows, cols, data })
    }
}

/// C `CvRNG`, retaining the source multiply-with-carry sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvRng(pub u64);
impl CvRng {
    pub fn new(seed: i64) -> Self {
        Self(if seed == 0 { u64::MAX } else { seed as u64 })
    }
    pub fn rand_int(&mut self) -> u32 {
        self.0 = (self.0 as u32 as u64)
            .wrapping_mul(1_554_115_554)
            .wrapping_add(self.0 >> 32);
        self.0 as u32
    }
    pub fn rand_real(&mut self) -> f64 {
        self.rand_int() as f64 * 2.328_306_436_538_696_3e-10
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvUtilsError {
    BadArgument,
    BadSize,
    UnmatchedSizes,
    UnsupportedFormat,
    OutOfRange,
}

/// C `cvCheckTermCriteria` as reached by `cvKMeans2`.
pub fn cv_check_term_criteria(
    mut criteria: CvTermCriteria,
    default_epsilon: f64,
    default_max_iter: i32,
) -> Result<CvTermCriteria, CvUtilsError> {
    if criteria.kind & !(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS) != 0 {
        return Err(CvUtilsError::BadArgument);
    }
    if criteria.kind & CV_TERMCRIT_ITER == 0 {
        criteria.max_iter = default_max_iter;
    }
    if criteria.kind & CV_TERMCRIT_EPS == 0 {
        criteria.epsilon = default_epsilon;
    }
    if criteria.max_iter <= 0 || !criteria.epsilon.is_finite() || criteria.epsilon < 0.0 {
        return Err(CvUtilsError::OutOfRange);
    }
    Ok(criteria)
}

/// C `cvKMeans2` for its supported `CV_32FC1` samples and `CV_32SC1` labels.
pub fn cv_k_means_2(
    samples: &CvMatrix<f32>,
    cluster_count: usize,
    labels: &mut CvMatrix<i32>,
    termcrit: CvTermCriteria,
) -> Result<(), CvUtilsError> {
    if cluster_count == 0 {
        return Err(CvUtilsError::OutOfRange);
    }
    if samples.rows == 0
        || labels.data.len() != samples.rows
        || !(labels.rows == 1 || labels.cols == 1)
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let criteria = cv_check_term_criteria(termcrit, 1e-6, 100)?;
    let clusters = cluster_count.min(samples.rows);
    let mut rng = CvRng::new(-1);
    for label in &mut labels.data {
        *label = (rng.rand_int() as usize % clusters) as i32;
    }
    let mut centers = vec![0.0_f64; clusters * samples.cols];
    let mut old_centers = centers.clone();
    let mut counters = vec![0usize; clusters];
    let epsilon = criteria.epsilon * criteria.epsilon;
    for iteration in 0..criteria.max_iter {
        centers.fill(0.0);
        counters.fill(0);
        for row in 0..samples.rows {
            let cluster = labels.data[row] as usize;
            for col in 0..samples.cols {
                centers[cluster * samples.cols + col] +=
                    samples.data[row * samples.cols + col] as f64;
            }
            counters[cluster] += 1;
        }
        let mut max_dist = if iteration == 0 {
            epsilon * 2.0
        } else {
            0.0_f64
        };
        for cluster in 0..clusters {
            if counters[cluster] != 0 {
                for col in 0..samples.cols {
                    centers[cluster * samples.cols + col] /= counters[cluster] as f64;
                }
            } else {
                let row = rng.rand_int() as usize % samples.rows;
                for col in 0..samples.cols {
                    centers[cluster * samples.cols + col] =
                        samples.data[row * samples.cols + col] as f64;
                }
            }
            if iteration > 0 {
                let mut distance = 0.0;
                for col in 0..samples.cols {
                    let value = centers[cluster * samples.cols + col]
                        - old_centers[cluster * samples.cols + col];
                    distance += value * value;
                }
                max_dist = max_dist.max(distance);
            }
        }
        for row in 0..samples.rows {
            let mut best = 0;
            let mut minimum = f64::MAX;
            for cluster in 0..clusters {
                let mut distance = 0.0;
                for col in 0..samples.cols {
                    let value = centers[cluster * samples.cols + col]
                        - samples.data[row * samples.cols + col] as f64;
                    distance += value * value;
                }
                if minimum > distance {
                    minimum = distance;
                    best = cluster;
                }
            }
            labels.data[row] = best as i32;
        }
        if max_dist < epsilon {
            break;
        }
        std::mem::swap(&mut centers, &mut old_centers);
    }
    counters.fill(0);
    for &label in &labels.data {
        counters[label as usize] += 1;
    }
    for cluster in 0..clusters {
        while counters[cluster] == 0 {
            let row = rng.rand_int() as usize % samples.rows;
            let old = labels.data[row] as usize;
            if counters[old] > 1 {
                labels.data[row] = cluster as i32;
                counters[old] -= 1;
                counters[cluster] += 1;
            }
        }
    }
    Ok(())
}

/// C `cvSolveCubic`, retaining the original root order.
pub fn cv_solve_cubic(coefficients: &[f64], roots: &mut [f64; 3]) -> Result<i32, CvUtilsError> {
    if coefficients.len() != 3 && coefficients.len() != 4 {
        return Err(CvUtilsError::BadSize);
    }
    let (mut a0, a1, a2, a3) = if coefficients.len() == 4 {
        (
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
        )
    } else {
        (1.0, coefficients[0], coefficients[1], coefficients[2])
    };
    let (mut x0, mut x1, mut x2) = (0.0, 0.0, 0.0);
    let mut count = 0;
    if a0 == 0.0 {
        if a1 == 0.0 {
            if a2 == 0.0 {
                count = if a3 == 0.0 { -1 } else { 0 };
            } else {
                x0 = a3 / a2;
                count = 1;
            }
        } else {
            let mut d = a2 * a2 - 4.0 * a1 * a3;
            if d >= 0.0 {
                d = d.sqrt();
                let q = (-a2 + if a2 < 0.0 { -d } else { d }) * 0.5;
                x0 = q / a1;
                x1 = a3 / q;
                count = if d > 0.0 { 2 } else { 1 };
            }
        }
    } else {
        a0 = 1.0 / a0;
        let a1 = a1 * a0;
        let a2 = a2 * a0;
        let a3 = a3 * a0;
        let q = (a1 * a1 - 3.0 * a2) / 9.0;
        let r = (2.0 * a1 * a1 * a1 - 9.0 * a1 * a2 + 27.0 * a3) / 54.0;
        let q_cubed = q * q * q;
        let mut d = q_cubed - r * r;
        if d >= 0.0 {
            let theta = (r / q_cubed.sqrt()).acos();
            let t0 = -2.0 * q.sqrt();
            let t1 = theta / 3.0;
            let t2 = a1 / 3.0;
            x0 = t0 * t1.cos() - t2;
            x1 = t0 * (t1 + 2.0 * std::f64::consts::PI / 3.0).cos() - t2;
            x2 = t0 * (t1 + 4.0 * std::f64::consts::PI / 3.0).cos() - t2;
            count = 3;
        } else {
            d = (-d).sqrt();
            let mut e = (d + r.abs()).powf(0.333_333_333_333);
            if r > 0.0 {
                e = -e;
            }
            x0 = e + q / e - a1 / 3.0;
            count = 1;
        }
    }
    *roots = [x0, x1, x2];
    Ok(count)
}

/// C `cvNormalize` for its scalar matrix paths. Masked-off destination values are preserved.
pub fn cv_normalize(
    values: &[f64],
    destination: &mut [f64],
    a: f64,
    b: f64,
    norm_type: i32,
    mask: Option<&[u8]>,
) -> Result<(), CvUtilsError> {
    if values.len() != destination.len() || mask.is_some_and(|mask| mask.len() != values.len()) {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let selected = |index: usize| mask.is_none_or(|mask| mask[index] != 0);
    let (scale, shift) = if norm_type == CV_MINMAX {
        let mut minimum = 0.0;
        let mut maximum = 0.0;
        let mut first = true;
        for (index, &value) in values.iter().enumerate() {
            if selected(index) {
                if first {
                    minimum = value;
                    maximum = value;
                    first = false;
                } else {
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                }
            }
        }
        let dmin = a.min(b);
        let dmax = a.max(b);
        let scale = (dmax - dmin)
            * if maximum - minimum > f64::EPSILON {
                1.0 / (maximum - minimum)
            } else {
                0.0
            };
        (scale, dmin - minimum * scale)
    } else if norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C {
        let mut norm = 0.0_f64;
        for (index, &value) in values.iter().enumerate() {
            if selected(index) {
                if norm_type == CV_L2 {
                    norm += value * value;
                } else if norm_type == CV_L1 {
                    norm += value.abs();
                } else {
                    norm = norm.max(value.abs());
                }
            }
        }
        if norm_type == CV_L2 {
            norm = norm.sqrt();
        }
        (if norm > f64::EPSILON { 1.0 / norm } else { 0.0 }, 0.0)
    } else {
        return Err(CvUtilsError::BadArgument);
    };
    for (index, &value) in values.iter().enumerate() {
        if selected(index) {
            destination[index] = value * scale + shift;
        }
    }
    Ok(())
}

/// C `cvRandShuffle` on an owned element sequence.
pub fn cv_rand_shuffle<T>(values: &mut [T], rng: Option<&mut CvRng>, iter_factor: f64) {
    let mut local = CvRng::new(-1);
    let rng = rng.unwrap_or(&mut local);
    let iterations = (iter_factor * values.len() as f64).round().max(0.0) as usize * 2;
    for _ in (0..iterations).step_by(2) {
        if !values.is_empty() {
            let first = (rng.rand_real() * values.len() as f64) as usize;
            let second = (rng.rand_real() * values.len() as f64) as usize;
            values.swap(first.min(values.len() - 1), second.min(values.len() - 1));
        }
    }
}

/// C `cvRange` for `CV_32SC1`.
pub fn cv_range_i32(values: &mut [i32], start: f64, end: f64) {
    let delta = (end - start) / values.len() as f64;
    let mut value = start;
    let integer = value.round();
    let increment = delta.round();
    if (value - integer).abs() < f64::EPSILON && (delta - increment).abs() < f64::EPSILON {
        let mut current = integer as i32;
        for output in values {
            *output = current;
            current += increment as i32;
        }
    } else {
        for output in values {
            *output = value.round() as i32;
            value += delta;
        }
    }
}
/// C `cvRange` for `CV_32FC1`.
pub fn cv_range_f32(values: &mut [f32], start: f64, end: f64) {
    let delta = (end - start) / values.len() as f64;
    let mut value = start;
    for output in values {
        *output = value as f32;
        value += delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roots_follow_source_order() {
        let mut roots = [0.0; 3];
        assert_eq!(cv_solve_cubic(&[1.0, -6.0, 11.0, -6.0], &mut roots), Ok(3));
        assert_eq!(roots, [1.0, 3.0, 2.0]);
    }
    #[test]
    fn kmeans_normalization_and_range_work() {
        let samples = CvMatrix::new(4, 1, vec![0.0, 0.1, 9.9, 10.0]).unwrap();
        let mut labels = CvMatrix::new(4, 1, vec![0; 4]).unwrap();
        cv_k_means_2(
            &samples,
            2,
            &mut labels,
            cv_term_criteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 1e-6),
        )
        .unwrap();
        assert_ne!(labels.data[0], labels.data[2]);
        let mut normalized = [0.0; 2];
        cv_normalize(&[2.0, 4.0], &mut normalized, 0.0, 1.0, CV_MINMAX, None).unwrap();
        assert_eq!(normalized, [0.0, 1.0]);
        let mut range = [0; 4];
        cv_range_i32(&mut range, 0.0, 4.0);
        assert_eq!(range, [0, 1, 2, 3]);
    }
}
