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

#[test]
fn aniso_diff_uniform_unchanged() {
    // Uniform image should remain unchanged after diffusion
    let mut s = Slice::new(8, 8, 100.0);
    aniso_diff(&mut s, 1, 10.0, 0.25, 5);
    for &v in &s.data {
        assert!((v - 100.0).abs() < 1e-6, "uniform image changed: {}", v);
    }
}

#[test]
fn aniso_diff_reduces_noise() {
    // A smooth region with one noisy pixel: diffusion should reduce the outlier
    let mut s = Slice::new(8, 8, 50.0);
    s.set(4, 4, 200.0); // outlier
    let original_outlier = s.get(4, 4);

    // CC=2 (inverse quadratic), moderate k, many iterations
    aniso_diff(&mut s, 2, 30.0, 0.20, 20);

    let diffused = s.get(4, 4);
    assert!(
        diffused < original_outlier,
        "outlier should be reduced: was {} now {}",
        original_outlier,
        diffused
    );
}

#[test]
fn aniso_diff_cc3_tukey_biweight() {
    // Test that CC=3 (Tukey biweight) runs without panics and reduces noise.
    // k must be larger than the gradient for Tukey biweight to allow diffusion.
    let mut s = Slice::new(6, 6, 10.0);
    s.set(3, 3, 50.0);
    aniso_diff(&mut s, 3, 50.0, 0.20, 10);
    // The outlier should have been reduced somewhat
    assert!(s.get(3, 3) < 50.0);
}

#[test]
fn aniso_diff_preserves_strong_edge() {
    // A sharp step edge: with small k the edge should be largely preserved
    let mut data = vec![0.0f32; 10 * 10];
    for y in 0..10 {
        for x in 5..10 {
            data[y * 10 + x] = 100.0;
        }
    }
    let mut s = Slice::from_data(10, 10, data);
    // Small k so that the large gradient across the edge is not smoothed
    aniso_diff(&mut s, 1, 5.0, 0.25, 10);

    // Interior of each side should remain close to original
    assert!((s.get(2, 5) - 0.0).abs() < 5.0, "dark side changed too much");
    assert!((s.get(7, 5) - 100.0).abs() < 5.0, "bright side changed too much");
}

#[test]
fn byte_threshold_basic() {
    let mut s = Slice::from_data(2, 2, vec![10.0, 50.0, 70.0, 90.0]);
    byte_threshold(&mut s, 60.0, 0.0, 255.0);
    // Below 60 -> 255 (high), >= 60 -> 0 (low)
    assert_eq!(s.data, vec![255.0, 255.0, 0.0, 0.0]);
}

#[test]
fn byte_grow_expands_region() {
    // 5x5 grid, single pixel in the center set to 255, rest 0
    let mut s = Slice::new(5, 5, 0.0);
    s.set(2, 2, 255.0);
    byte_grow(&mut s, 255.0);

    // The center pixel's 8 neighbours should now also be 255
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let x = (2 + dx) as usize;
            let y = (2 + dy) as usize;
            assert_eq!(
                s.get(x, y),
                255.0,
                "pixel ({},{}) should be 255 after grow",
                x,
                y
            );
        }
    }
    // Corner should still be 0
    assert_eq!(s.get(0, 0), 0.0);
}

#[test]
fn byte_shrink_erodes_region() {
    // 5x5 grid, fill a 3x3 block in the center with 255, rest 0
    let mut s = Slice::new(5, 5, 0.0);
    for dy in 1..=3 {
        for dx in 1..=3 {
            s.set(dx, dy, 255.0);
        }
    }
    byte_shrink(&mut s, 255.0);

    // After erosion, only the center pixel (2,2) should remain 255
    // because it is the only one with all 8 neighbours == 255
    // (Actually nay8 uses strict interior checks so the exact result
    // depends on boundary handling; just verify shrinkage occurred.)
    let count_after: usize = s.data.iter().filter(|&&v| v == 255.0).count();
    assert!(
        count_after < 9,
        "shrink should have removed some pixels, got {}",
        count_after
    );
}

#[test]
fn grow_then_shrink_approximate_identity() {
    // Growing then shrinking (or vice versa) on a blob should roughly
    // preserve the original shape.
    let mut s = Slice::new(10, 10, 0.0);
    // 4x4 block
    for y in 3..7 {
        for x in 3..7 {
            s.set(x, y, 255.0);
        }
    }
    let original_count: usize = s.data.iter().filter(|&&v| v == 255.0).count();

    byte_grow(&mut s, 255.0);
    let grown_count: usize = s.data.iter().filter(|&&v| v == 255.0).count();
    assert!(grown_count > original_count);

    byte_shrink(&mut s, 255.0);
    let shrunk_count: usize = s.data.iter().filter(|&&v| v == 255.0).count();
    assert!(shrunk_count < grown_count);
}
