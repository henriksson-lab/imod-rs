use crate::*;
use imod_core::Point3f;

#[test]
fn roundtrip_simple_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mod");

    let model = ImodModel {
        name: "test model".into(),
        xmax: 512,
        ymax: 512,
        zmax: 100,
        pixel_size: 1.5,
        objects: vec![ImodObject {
            name: "fiducials".into(),
            red: 1.0,
            green: 0.0,
            blue: 0.0,
            contours: vec![
                ImodContour {
                    points: vec![
                        Point3f { x: 10.0, y: 20.0, z: 5.0 },
                        Point3f { x: 15.0, y: 25.0, z: 5.0 },
                        Point3f { x: 20.0, y: 30.0, z: 5.0 },
                    ],
                    ..Default::default()
                },
                ImodContour {
                    points: vec![
                        Point3f { x: 100.0, y: 200.0, z: 50.0 },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    };

    write_model(&path, &model).unwrap();
    let read_back = read_model(&path).unwrap();

    assert_eq!(read_back.name, "test model");
    assert_eq!(read_back.xmax, 512);
    assert_eq!(read_back.ymax, 512);
    assert_eq!(read_back.zmax, 100);
    assert!((read_back.pixel_size - 1.5).abs() < 1e-6);
    assert_eq!(read_back.objects.len(), 1);

    let obj = &read_back.objects[0];
    assert_eq!(obj.name, "fiducials");
    assert!((obj.red - 1.0).abs() < 1e-6);
    assert_eq!(obj.contours.len(), 2);
    assert_eq!(obj.contours[0].points.len(), 3);
    assert!((obj.contours[0].points[0].x - 10.0).abs() < 1e-5);
    assert!((obj.contours[0].points[2].z - 5.0).abs() < 1e-5);
    assert_eq!(obj.contours[1].points.len(), 1);
    assert!((obj.contours[1].points[0].x - 100.0).abs() < 1e-5);
}

#[test]
fn roundtrip_multiple_objects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.mod");

    let model = ImodModel {
        name: "multi object model".into(),
        xmax: 1024,
        ymax: 1024,
        zmax: 200,
        objects: vec![
            ImodObject {
                name: "obj1".into(),
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                contours: vec![ImodContour {
                    points: vec![Point3f { x: 1.0, y: 2.0, z: 3.0 }],
                    ..Default::default()
                }],
                ..Default::default()
            },
            ImodObject {
                name: "obj2".into(),
                red: 0.0,
                green: 0.0,
                blue: 1.0,
                contours: vec![
                    ImodContour {
                        points: vec![
                            Point3f { x: 10.0, y: 20.0, z: 30.0 },
                            Point3f { x: 40.0, y: 50.0, z: 60.0 },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    write_model(&path, &model).unwrap();
    let read_back = read_model(&path).unwrap();

    assert_eq!(read_back.objects.len(), 2);
    assert_eq!(read_back.objects[0].name, "obj1");
    assert_eq!(read_back.objects[1].name, "obj2");
    assert!((read_back.objects[1].blue - 1.0).abs() < 1e-6);
    assert_eq!(read_back.objects[1].contours[0].points.len(), 2);
}

#[test]
fn roundtrip_with_mesh() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mesh.mod");

    let model = ImodModel {
        name: "mesh model".into(),
        xmax: 256,
        ymax: 256,
        zmax: 50,
        objects: vec![ImodObject {
            name: "meshed".into(),
            meshes: vec![ImodMesh {
                vertices: vec![
                    Point3f { x: 0.0, y: 0.0, z: 0.0 },
                    Point3f { x: 1.0, y: 0.0, z: 0.0 },
                    Point3f { x: 0.0, y: 1.0, z: 0.0 },
                ],
                indices: vec![0, 1, 2, -1], // triangle + end marker
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    write_model(&path, &model).unwrap();
    let read_back = read_model(&path).unwrap();

    assert_eq!(read_back.objects[0].meshes.len(), 1);
    let mesh = &read_back.objects[0].meshes[0];
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.indices.len(), 4);
    assert_eq!(mesh.indices[3], -1);
}

#[test]
fn empty_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.mod");

    let model = ImodModel::default();
    write_model(&path, &model).unwrap();
    let read_back = read_model(&path).unwrap();

    assert_eq!(read_back.objects.len(), 0);
}

#[test]
fn chunk_ids_are_correct() {
    assert_eq!(&chunk_id::IMOD.to_be_bytes(), b"IMOD");
    assert_eq!(&chunk_id::OBJT.to_be_bytes(), b"OBJT");
    assert_eq!(&chunk_id::CONT.to_be_bytes(), b"CONT");
    assert_eq!(&chunk_id::MESH.to_be_bytes(), b"MESH");
    assert_eq!(&chunk_id::IEOF.to_be_bytes(), b"IEOF");
}
