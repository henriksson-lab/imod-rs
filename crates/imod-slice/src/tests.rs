use crate::*;

#[test]
fn slice_create_and_access() {
    let mut s = Slice::new(10, 10, 5.0);
    assert_eq!(s.get(0, 0), 5.0);
    s.set(3, 4, 42.0);
    assert_eq!(s.get(3, 4), 42.0);
    assert_eq!(s.len(), 100);
}

#[test]
fn slice_statistics() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let s = Slice::from_data(10, 10, data);
    let (min, max, mean) = s.statistics();
    assert_eq!(min, 0.0);
    assert_eq!(max, 99.0);
    assert!((mean - 49.5).abs() < 1e-4);
}

#[test]
fn bilinear_interpolation() {
    let mut s = Slice::new(4, 4, 0.0);
    s.set(1, 1, 10.0);
    // At exact pixel
    assert!((s.interpolate_bilinear(1.0, 1.0, 0.0) - 10.0).abs() < 1e-5);
    // Halfway between (1,1) and (2,1)
    assert!((s.interpolate_bilinear(1.5, 1.0, 0.0) - 5.0).abs() < 1e-5);
}

#[test]
fn scale_op() {
    let mut s = Slice::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    scale(&mut s, 2.0, 1.0);
    assert_eq!(s.data, vec![3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn clamp_op() {
    let mut s = Slice::from_data(2, 2, vec![-1.0, 0.5, 1.5, 3.0]);
    clamp(&mut s, 0.0, 1.0);
    assert_eq!(s.data, vec![0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn threshold_op() {
    let mut s = Slice::from_data(2, 2, vec![0.3, 0.5, 0.7, 0.9]);
    threshold(&mut s, 0.5, 0.0, 1.0);
    assert_eq!(s.data, vec![0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn add_slices() {
    let a = Slice::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Slice::from_data(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
    let c = add(&a, &b);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn subtract_slices() {
    let a = Slice::from_data(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
    let b = Slice::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let c = subtract(&a, &b);
    assert_eq!(c.data, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn bin_downsample() {
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let s = Slice::from_data(4, 4, data);
    let binned = bin(&s, 2);
    assert_eq!(binned.nx, 2);
    assert_eq!(binned.ny, 2);
    // Top-left 2x2: (1+2+5+6)/4 = 3.5
    assert!((binned.get(0, 0) - 3.5).abs() < 1e-5);
    // Bottom-right 2x2: (11+12+15+16)/4 = 13.5
    assert!((binned.get(1, 1) - 13.5).abs() < 1e-5);
}

#[test]
fn sobel_edge_detection() {
    // Uniform image should have zero edges
    let s = Slice::new(8, 8, 5.0);
    let edges = sobel(&s);
    for &v in &edges.data {
        assert!(v.abs() < 1e-5);
    }
}

#[test]
fn blur_preserves_mean() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32).sin() * 10.0 + 50.0).collect();
    let s = Slice::from_data(8, 8, data);
    let blurred = blur_3x3(&s);
    let (_, _, mean_orig) = s.statistics();
    let (_, _, mean_blur) = blurred.statistics();
    // Mean should be approximately preserved (not exact due to boundary)
    assert!((mean_orig - mean_blur).abs() < 2.0);
}

#[test]
fn subregion_extraction() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let s = Slice::from_data(10, 10, data);
    let sub = s.subregion(2, 3, 4, 3);
    assert_eq!(sub.nx, 4);
    assert_eq!(sub.ny, 3);
    assert_eq!(sub.get(0, 0), 32.0); // row 3, col 2
    assert_eq!(sub.get(3, 2), 55.0); // row 5, col 5
}

#[test]
fn invert_op() {
    let mut s = Slice::from_data(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    invert(&mut s);
    assert_eq!(s.data, vec![3.0, 2.0, 1.0, 0.0]);
}
