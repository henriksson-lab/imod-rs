use crate::*;

#[test]
fn triangulate_square() {
    let points = vec![
        Point2d { x: 0.0, y: 0.0 },
        Point2d { x: 1.0, y: 0.0 },
        Point2d { x: 1.0, y: 1.0 },
        Point2d { x: 0.0, y: 1.0 },
    ];
    let tri = triangulate(&points);
    assert_eq!(tri.points.len(), 4);
    assert_eq!(tri.triangles.len(), 2); // square -> 2 triangles
}

#[test]
fn triangulate_random_points() {
    let points = vec![
        Point2d { x: 0.0, y: 0.0 },
        Point2d { x: 5.0, y: 0.0 },
        Point2d { x: 2.5, y: 4.0 },
        Point2d { x: 1.0, y: 2.0 },
        Point2d { x: 4.0, y: 2.0 },
        Point2d { x: 2.5, y: 1.0 },
    ];
    let tri = triangulate(&points);
    assert!(!tri.triangles.is_empty());
    // All triangle vertices should be valid indices
    for t in &tri.triangles {
        assert!(t.a < points.len());
        assert!(t.b < points.len());
        assert!(t.c < points.len());
    }
}

#[test]
fn find_containing_triangle() {
    let points = vec![
        Point2d { x: 0.0, y: 0.0 },
        Point2d { x: 10.0, y: 0.0 },
        Point2d { x: 5.0, y: 10.0 },
    ];
    let tri = triangulate(&points);
    assert_eq!(tri.triangles.len(), 1);

    // Center of triangle should be inside
    let found = tri.find_containing_triangle(5.0, 3.0);
    assert!(found.is_some());

    // Far away point should be outside
    let not_found = tri.find_containing_triangle(100.0, 100.0);
    assert!(not_found.is_none());
}

#[test]
fn triangulate_too_few_points() {
    let points = vec![Point2d { x: 0.0, y: 0.0 }, Point2d { x: 1.0, y: 1.0 }];
    let tri = triangulate(&points);
    assert!(tri.triangles.is_empty());
}

#[test]
fn warp_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.warp");

    let wf = WarpFile {
        nx: 512,
        ny: 512,
        binning: 1,
        pixel_size: 1.5,
        version: 1,
        flags: 0,
        sections: vec![WarpTransform {
            z: 0,
            nx: 512,
            ny: 512,
            control_x: vec![100.0, 200.0, 300.0],
            control_y: vec![100.0, 200.0, 300.0],
            transforms: vec![
                imod_transforms::LinearTransform::identity(),
                imod_transforms::LinearTransform::identity(),
                imod_transforms::LinearTransform::identity(),
            ],
        }],
    };

    wf.write_to_file(&path).unwrap();
    let read_back = WarpFile::from_file(&path).unwrap();

    assert_eq!(read_back.nx, 512);
    assert_eq!(read_back.sections.len(), 1);
    assert_eq!(read_back.sections[0].transforms.len(), 3);
    assert!((read_back.pixel_size - 1.5).abs() < 1e-4);
}
