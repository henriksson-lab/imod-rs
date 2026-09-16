//! Safe, generic translation of `svlStatsUtils.{h,cpp}`.

use std::collections::BTreeSet;

pub trait StatsNumber: Copy + PartialOrd {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}
macro_rules! stats_number { ($($type:ty),+ $(,)?) => { $(impl StatsNumber for $type { fn to_f64(self)->f64 { self as f64 } fn from_f64(value:f64)->Self { value as $type } })+ }; }
stats_number!(f32, f64, i32, i64, u32, u64, usize);

pub fn contains_invalid_entries(values: &[f64]) -> bool {
    values.iter().any(|value| !value.is_finite())
}
pub fn logistic(theta: &[f64], data: &[f64]) -> Option<f64> {
    (theta.len() == data.len()).then(|| {
        1.0 / (1.0
            + (-theta
                .iter()
                .zip(data)
                .map(|(left, right)| left * right)
                .sum::<f64>())
            .exp())
    })
}
pub fn entropy(values: &[f64]) -> f64 {
    let z: f64 = values.iter().sum();
    let h: f64 = values
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| value * value.ln())
        .sum();
    (z.ln() - h / z) / std::f64::consts::LN_2
}
pub fn exp_and_normalize(values: &mut [f64]) {
    if let Some(&maximum) = values.iter().max_by(|left, right| left.total_cmp(right)) {
        let z: f64 = values
            .iter_mut()
            .map(|value| {
                *value = (*value - maximum).exp();
                *value
            })
            .sum();
        for value in values {
            *value /= z;
        }
    }
}
pub fn random_permutation(n: usize) -> Vec<usize> {
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    random_permutation_with_seed(n, &mut seed)
}
pub fn random_permutation_with_seed(n: usize, seed: &mut u64) -> Vec<usize> {
    let mut result: Vec<usize> = (0..n).collect();
    for index in 0..n.saturating_sub(1) {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let other = index + (*seed as usize % (n - index));
        result.swap(index, other);
    }
    result
}
pub fn predecessor(array: &mut [i32], limit: i32) -> bool {
    predecessor_with_limits(array, &vec![limit; array.len()])
}
pub fn successor(array: &mut [i32], limit: i32) -> bool {
    successor_with_limits(array, &vec![limit; array.len()])
}
pub fn predecessor_with_limits(array: &mut [i32], limits: &[i32]) -> bool {
    if array.len() != limits.len() || limits.iter().any(|limit| *limit <= 0) {
        return false;
    }
    for (value, &limit) in array.iter_mut().zip(limits) {
        *value -= 1;
        if *value < 0 {
            *value = limit - 1;
        } else {
            break;
        }
    }
    true
}
pub fn successor_with_limits(array: &mut [i32], limits: &[i32]) -> bool {
    if array.len() != limits.len() || limits.iter().any(|limit| *limit <= 0) {
        return false;
    }
    for (value, &limit) in array.iter_mut().zip(limits) {
        *value += 1;
        if *value >= limit {
            *value = 0;
        } else {
            break;
        }
    }
    true
}
pub fn min_elem<T: PartialOrd + Copy>(values: &[T]) -> Option<T> {
    values
        .iter()
        .copied()
        .reduce(|left, right| if right < left { right } else { left })
}
pub fn max_elem<T: PartialOrd + Copy>(values: &[T]) -> Option<T> {
    values
        .iter()
        .copied()
        .reduce(|left, right| if right > left { right } else { left })
}
pub fn mean<T: StatsNumber>(values: &[T]) -> Option<T> {
    (!values.is_empty()).then(|| {
        T::from_f64(values.iter().map(|value| value.to_f64()).sum::<f64>() / values.len() as f64)
    })
}
pub fn median<T: StatsNumber>(values: &[T]) -> Option<T> {
    let mut copy = values.to_vec();
    destructive_median(&mut copy)
}
pub fn destructive_median<T: StatsNumber>(values: &mut [T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.to_f64().total_cmp(&right.to_f64()));
    let upper = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[upper]
    } else {
        T::from_f64(0.5 * (values[upper].to_f64() + values[upper - 1].to_f64()))
    })
}
pub fn mode<T: Copy>(values: &[T]) -> Option<T> {
    // The C++ loop never updates `maxCount`, making every element a new mode.
    values.last().copied()
}
pub fn variance<T: StatsNumber>(values: &[T]) -> Option<T> {
    let average = mean(values)?.to_f64();
    Some(T::from_f64(
        values
            .iter()
            .map(|value| (value.to_f64() - average).powi(2))
            .sum::<f64>()
            / values.len() as f64,
    ))
}
pub fn stdev<T: StatsNumber>(values: &[T]) -> Option<T> {
    Some(T::from_f64(variance(values)?.to_f64().max(0.0).sqrt()))
}
pub fn argmin<T: PartialOrd>(values: &[T]) -> i32 {
    values.iter().enumerate().skip(1).fold(
        if values.is_empty() { -1 } else { 0 },
        |best, (index, value)| {
            if *value < values[best as usize] {
                index as i32
            } else {
                best
            }
        },
    )
}
pub fn argmax<T: PartialOrd>(values: &[T]) -> i32 {
    values.iter().enumerate().skip(1).fold(
        if values.is_empty() { -1 } else { 0 },
        |best, (index, value)| {
            if *value > values[best as usize] {
                index as i32
            } else {
                best
            }
        },
    )
}
pub fn argmins<T: PartialOrd>(rows: &[Vec<T>]) -> Vec<i32> {
    rows.iter().map(|row| argmin(row)).collect()
}
pub fn argmaxs<T: PartialOrd>(rows: &[Vec<T>]) -> Vec<i32> {
    rows.iter().map(|row| argmax(row)).collect()
}
pub fn excess_kurtosis<T: StatsNumber>(values: &[T]) -> Option<T> {
    let average = mean(values)?.to_f64();
    let variance = variance(values)?.to_f64();
    Some(T::from_f64(
        values
            .iter()
            .map(|value| (value.to_f64() - average).powi(4))
            .sum::<f64>()
            / (values.len() as f64 * variance * variance)
            - 3.0,
    ))
}
pub fn percentiles<T: PartialOrd>(values: &[T]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            values.iter().filter(|other| *other < value).count() as f32 / values.len() as f32
        })
        .collect()
}
pub fn range<T: PartialOrd + Copy>(values: &[T]) -> Option<(T, T)> {
    Some((min_elem(values)?, max_elem(values)?))
}
pub fn range_nested<T: PartialOrd + Copy>(values: &[Vec<T>]) -> Option<(T, T)> {
    let mut all = values.iter().flatten();
    let first = *all.next()?;
    Some(all.fold((first, first), |(minimum, maximum), value| {
        (
            if *value < minimum { *value } else { minimum },
            if *value > maximum { *value } else { maximum },
        )
    }))
}
pub fn extract_sub_vector<T: Clone>(values: &[T], indices: &[usize]) -> Option<Vec<T>> {
    indices
        .iter()
        .map(|&index| values.get(index).cloned())
        .collect()
}
pub fn remove_outliers<T: Clone>(values: &[T], scores: &[f64], keep_size: usize) -> Option<Vec<T>> {
    if values.len() != scores.len() {
        return None;
    }
    if keep_size >= values.len() {
        return Some(values.to_vec());
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| scores[left].total_cmp(&scores[right]));
    let start = (values.len() - keep_size) / 2;
    Some(
        indices[start..start + keep_size]
            .iter()
            .map(|&index| values[index].clone())
            .collect(),
    )
}
pub fn powerset<T: Ord + Clone>(values: &BTreeSet<T>) -> BTreeSet<BTreeSet<T>> {
    let mut result = BTreeSet::from([BTreeSet::new()]);
    for value in values {
        let additions: Vec<_> = result
            .iter()
            .map(|set| {
                let mut next = set.clone();
                next.insert(value.clone());
                next
            })
            .collect();
        result.extend(additions);
    }
    result
}
pub fn huber_function(x: f64, m: f64) -> f64 {
    if x < -m {
        m * (-2.0 * x - m)
    } else if x > m {
        m * (2.0 * x - m)
    } else {
        x * x
    }
}
pub fn huber_derivative(x: f64, m: f64) -> f64 {
    if x < -m {
        -2.0 * m
    } else if x > m {
        2.0 * m
    } else {
        2.0 * x
    }
}
pub fn huber_function_and_derivative(x: f64, m: f64) -> (f64, f64) {
    (huber_function(x, m), huber_derivative(x, m))
}
pub fn bhattacharyya_distance(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let coefficient: f64 = p
        .iter()
        .zip(q)
        .map(|(left, right)| (left * right).sqrt())
        .sum();
    let zp: f64 = p.iter().sum();
    let zq: f64 = q.iter().sum();
    (zp > 0.0 && zq > 0.0).then(|| -(coefficient / (zp * zq).sqrt()).ln())
}
pub fn euclidean_distance_sq(p: &[f64], q: &[f64]) -> Option<f64> {
    (p.len() == q.len()).then(|| {
        p.iter()
            .zip(q)
            .map(|(left, right)| (left - right).powi(2))
            .sum()
    })
}
pub fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}
pub fn argrand(values: &[f64]) -> Option<usize> {
    argrand_at(values, 0.5)
}
pub fn argrand_at(values: &[f64], random_unit: f64) -> Option<usize> {
    let cutoff = sum(values) * random_unit;
    let mut cumulative = 0.0;
    for (index, value) in values.iter().enumerate() {
        cumulative += value;
        if cumulative >= cutoff {
            return Some(index);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numerical_functions_match_source_contracts() {
        assert_eq!(logistic(&[2.], &[1.]), Some(1.0 / (1.0 + (-2.0_f64).exp())));
        assert!((entropy(&[1., 1.]) - 1.0).abs() < 1e-12);
        let mut values = vec![0., 1.];
        exp_and_normalize(&mut values);
        assert!((sum(&values) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn generic_and_discrete_helpers_follow_little_endian_counters() {
        assert_eq!(median(&[1_i32, 4, 2, 3]), Some(2));
        assert_eq!(mode(&[2, 3, 3, 2]), Some(2));
        assert_eq!(mode(&[2, 3]), Some(3));
        let mut digits = vec![1, 0];
        successor(&mut digits, 2);
        assert_eq!(digits, vec![0, 1]);
        predecessor(&mut digits, 2);
        assert_eq!(digits, vec![1, 0]);
        assert_eq!(argmax(&[1., 4., 4.]), 1);
    }
    #[test]
    fn distances_and_sets_are_owned() {
        assert_eq!(euclidean_distance_sq(&[1., 2.], &[4., 6.]), Some(25.));
        assert_eq!(
            remove_outliers(&[10, 20, 30], &[3., 1., 2.], 1),
            Some(vec![30])
        );
        let set = BTreeSet::from([1, 2]);
        assert_eq!(powerset(&set).len(), 4);
    }
}
