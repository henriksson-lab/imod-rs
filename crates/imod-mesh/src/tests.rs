use crate::*;

#[test]
fn skin_two_circles() {
    let n = 8;
    let c1 = Contour2d {
        points: (0..n)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                [angle.cos() * 10.0, angle.sin() * 10.0]
            })
            .collect(),
        z: 0.0,
        closed: true,
    };
    let c2 = Contour2d {
        points: (0..n)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                [angle.cos() * 12.0, angle.sin() * 12.0]
            })
            .collect(),
        z: 1.0,
        closed: true,
    };

    let mesh = skin_contours(&c1, &c2);
    assert_eq!(mesh.vertices.len(), 2 * n);
    assert!(!mesh.indices.is_empty());
    // Should have groups of 3 (triangles)
    assert_eq!(mesh.indices.len() % 3, 0);
}

#[test]
fn skin_empty_contour() {
    let c1 = Contour2d { points: Vec::new(), z: 0.0, closed: true };
    let c2 = Contour2d {
        points: vec![[0.0, 0.0], [1.0, 0.0]],
        z: 1.0,
        closed: true,
    };
    let mesh = skin_contours(&c1, &c2);
    assert!(mesh.vertices.is_empty());
}

#[test]
fn simplify_straight_line() {
    let points: Vec<[f32; 2]> = (0..10).map(|i| [i as f32, 0.0]).collect();
    let simplified = simplify_contour(&points, 0.1);
    // Straight line should reduce to just endpoints
    assert_eq!(simplified.len(), 2);
    assert_eq!(simplified[0], [0.0, 0.0]);
    assert_eq!(simplified[1], [9.0, 0.0]);
}

#[test]
fn simplify_preserves_corners() {
    let points = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [3.0, 2.0],
        [3.0, 3.0],
    ];
    let simplified = simplify_contour(&points, 0.1);
    // Should keep: start, corner at (3,0), end
    assert!(simplified.len() <= 4);
    assert_eq!(simplified[0], [0.0, 0.0]);
    assert_eq!(simplified.last().unwrap(), &[3.0, 3.0]);
}

#[test]
fn simplify_few_points() {
    let points = vec![[0.0, 0.0], [1.0, 1.0]];
    let simplified = simplify_contour(&points, 1.0);
    assert_eq!(simplified.len(), 2);
}
