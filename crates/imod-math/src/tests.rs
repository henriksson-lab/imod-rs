use crate::*;

#[test]
fn test_mean() {
    assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
    assert_eq!(mean(&[]), 0.0);
    assert_eq!(mean(&[42.0]), 42.0);
}

#[test]
fn test_mean_sd() {
    let (m, sd) = mean_sd(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((m - 5.0).abs() < 1e-5);
    assert!((sd - 2.138).abs() < 0.01);
}

#[test]
fn test_min_max_mean() {
    let (min, max, mean) = min_max_mean(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    assert_eq!(min, 1.0);
    assert_eq!(max, 9.0);
    assert!((mean - 23.0 / 6.0).abs() < 1e-5);
}

#[test]
fn test_robust_stat() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // outlier at 100
    let (median, madn) = robust_stat(&mut data);
    assert!((median - 3.5).abs() < 1e-5);
    // MAD should be small since most data is clustered
    assert!(madn < 5.0);
}

#[test]
fn test_sample_mean_sd() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let (m, _sd) = sample_mean_sd(&data, 1);
    assert!((m - 49.5).abs() < 1e-3);

    // Sampling every 10th element
    let (m2, _sd2) = sample_mean_sd(&data, 10);
    assert!((m2 - 45.0).abs() < 1e-3); // 0, 10, 20, ..., 90 -> mean = 45
}

#[test]
fn test_linear_regression() {
    // Perfect linear: y = 2 + 3x
    let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    let (intercept, slope, r) = linear_regression(&x, &y).unwrap();
    assert!((intercept - 2.0).abs() < 1e-4);
    assert!((slope - 3.0).abs() < 1e-4);
    assert!((r - 1.0).abs() < 1e-4);
}

#[test]
fn test_linear_regression_negative_slope() {
    let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 10.0 - 2.0 * xi).collect();
    let (intercept, slope, r) = linear_regression(&x, &y).unwrap();
    assert!((intercept - 10.0).abs() < 1e-4);
    assert!((slope - (-2.0)).abs() < 1e-4);
    assert!((r - (-1.0)).abs() < 1e-4);
}
