use crate::*;

#[test]
fn identity_transform() {
    let xf = LinearTransform::identity();
    let (x, y) = xf.apply_raw(3.0, 4.0);
    assert!((x - 3.0).abs() < 1e-6);
    assert!((y - 4.0).abs() < 1e-6);
}

#[test]
fn rotation_90_degrees() {
    let xf = LinearTransform::rotation(90.0);
    let (x, y) = xf.apply_raw(1.0, 0.0);
    assert!(x.abs() < 1e-5);
    assert!((y - 1.0).abs() < 1e-5);
}

#[test]
fn translation() {
    let xf = LinearTransform::translation(10.0, -5.0);
    let (x, y) = xf.apply_raw(1.0, 2.0);
    assert!((x - 11.0).abs() < 1e-6);
    assert!((y - (-3.0)).abs() < 1e-6);
}

#[test]
fn multiply_then() {
    // Rotate 90, then translate
    let rot = LinearTransform::rotation(90.0);
    let trans = LinearTransform::translation(10.0, 0.0);
    let combined = rot.then(&trans);
    let (x, y) = combined.apply_raw(1.0, 0.0);
    // Rotate (1,0) -> (0,1), then translate -> (10, 1)
    assert!((x - 10.0).abs() < 1e-4);
    assert!((y - 1.0).abs() < 1e-4);
}

#[test]
fn inverse_roundtrip() {
    let xf = LinearTransform {
        a11: 0.95,
        a12: -0.31,
        a21: 0.31,
        a22: 0.95,
        dx: 5.0,
        dy: -3.0,
    };
    let inv = xf.inverse();
    let roundtrip = xf.then(&inv);
    assert!((roundtrip.a11 - 1.0).abs() < 1e-5);
    assert!(roundtrip.a12.abs() < 1e-5);
    assert!(roundtrip.a21.abs() < 1e-5);
    assert!((roundtrip.a22 - 1.0).abs() < 1e-5);
    assert!(roundtrip.dx.abs() < 1e-4);
    assert!(roundtrip.dy.abs() < 1e-4);
}

#[test]
fn apply_with_center() {
    // Identity with center should not change anything
    let xf = LinearTransform::identity();
    let (x, y) = xf.apply(100.0, 100.0, 50.0, 50.0);
    assert!((x - 50.0).abs() < 1e-6);
    assert!((y - 50.0).abs() < 1e-6);
}

#[test]
fn rotation_angle_extraction() {
    let xf = LinearTransform::rotation(45.0);
    assert!((xf.rotation_angle() - 45.0).abs() < 1e-4);
}

#[test]
fn scale_factor_extraction() {
    let xf = LinearTransform::scale(2.0);
    assert!((xf.scale_factor() - 2.0).abs() < 1e-5);
}

#[test]
fn xf_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.xf");

    let xforms = vec![
        LinearTransform {
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
            dx: 0.0,
            dy: 0.0,
        },
        LinearTransform {
            a11: 0.9848077,
            a12: -0.1736482,
            a21: 0.1736482,
            a22: 0.9848077,
            dx: 5.123,
            dy: -3.456,
        },
    ];

    write_xf_file(&path, &xforms).unwrap();
    let read_back = read_xf_file(&path).unwrap();

    assert_eq!(read_back.len(), 2);
    for (a, b) in read_back.iter().zip(xforms.iter()) {
        assert!((a.a11 - b.a11).abs() < 1e-5);
        assert!((a.a12 - b.a12).abs() < 1e-5);
        assert!((a.a21 - b.a21).abs() < 1e-5);
        assert!((a.a22 - b.a22).abs() < 1e-5);
        assert!((a.dx - b.dx).abs() < 0.01);
        assert!((a.dy - b.dy).abs() < 0.01);
    }
}

#[test]
fn tilt_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.tlt");

    let angles: Vec<f32> = (-60..=60).step_by(3).map(|i| i as f32).collect();
    write_tilt_file(&path, &angles).unwrap();
    let read_back = read_tilt_file(&path).unwrap();

    assert_eq!(read_back.len(), angles.len());
    for (a, b) in read_back.iter().zip(angles.iter()) {
        assert!((a - b).abs() < 0.1);
    }
}

#[test]
fn generate_tilt_angles_test() {
    let angles = generate_tilt_angles(-60.0, 3.0, 41);
    assert_eq!(angles.len(), 41);
    assert!((angles[0] - (-60.0)).abs() < 1e-6);
    assert!((angles[20] - 0.0).abs() < 1e-4);
    assert!((angles[40] - 60.0).abs() < 1e-4);
}

#[test]
fn determinant() {
    let xf = LinearTransform::rotation(30.0);
    assert!((xf.determinant() - 1.0).abs() < 1e-5);

    let xf2 = LinearTransform::scale(3.0);
    assert!((xf2.determinant() - 9.0).abs() < 1e-5);
}
