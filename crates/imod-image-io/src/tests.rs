use crate::*;
use imod_core::MrcMode;
use imod_mrc::{MrcHeader, MrcWriter};

#[test]
fn open_mrc_via_image_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mrc");

    // Write a test MRC file
    let nx = 8;
    let ny = 8;
    let nz = 3;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Float);
    let mut writer = MrcWriter::create(&path, header).unwrap();
    for z in 0..nz {
        let data: Vec<f32> = (0..nx * ny).map(|i| (i + z * 10) as f32).collect();
        writer.write_slice_f32(&data).unwrap();
    }
    writer.finish(0.0, 100.0, 50.0).unwrap();

    // Open via generic interface
    let mut img = open_image(&path).unwrap();
    let info = img.info();
    assert_eq!(info.nx, 8);
    assert_eq!(info.ny, 8);
    assert_eq!(info.nz, 3);

    let slice = img.read_slice(0).unwrap();
    assert_eq!(slice.nx, 8);
    assert_eq!(slice.ny, 8);
    assert!((slice.get(0, 0) - 0.0).abs() < 1e-5);
}

#[test]
fn format_detection() {
    assert_eq!(ImageFormat::from_path("foo.mrc"), ImageFormat::Mrc);
    assert_eq!(ImageFormat::from_path("bar.st"), ImageFormat::Mrc);
    assert_eq!(ImageFormat::from_path("baz.rec"), ImageFormat::Mrc);
    assert_eq!(ImageFormat::from_path("qux.ali"), ImageFormat::Mrc);
    assert_eq!(ImageFormat::from_path("img.tif"), ImageFormat::Tiff);
    assert_eq!(ImageFormat::from_path("img.tiff"), ImageFormat::Tiff);
    assert_eq!(ImageFormat::from_path("pic.jpg"), ImageFormat::Jpeg);
    assert_eq!(ImageFormat::from_path("data.hdf"), ImageFormat::Hdf5);
    assert_eq!(ImageFormat::from_path("unknown.xyz"), ImageFormat::Unknown);
}

#[test]
fn read_all_slices() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stack.mrc");

    let nx = 4;
    let ny = 4;
    let nz = 5;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Byte);
    let mut writer = MrcWriter::create(&path, header).unwrap();
    for z in 0..nz {
        let data: Vec<f32> = (0..nx * ny).map(|_| z as f32 * 10.0).collect();
        writer.write_slice_f32(&data).unwrap();
    }
    writer.finish(0.0, 40.0, 20.0).unwrap();

    let mut img = open_image(&path).unwrap();
    let slices = img.read_all().unwrap();
    assert_eq!(slices.len(), 5);
    assert!((slices[0].get(0, 0) - 0.0).abs() < 1e-5);
    assert!((slices[4].get(0, 0) - 40.0).abs() < 1e-5);
}
