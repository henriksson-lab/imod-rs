use crate::{MrcHeader, MrcReader, MrcWriter};
use imod_core::MrcMode;
use std::io::{Cursor, Write};

#[test]
fn header_size_is_1024_bytes() {
    let header = MrcHeader::new(64, 64, 1, MrcMode::Byte);
    let mut buf = Cursor::new(Vec::new());
    binrw::BinWrite::write_le(&header, &mut buf).unwrap();
    assert_eq!(buf.get_ref().len(), MrcHeader::SIZE);
}

#[test]
fn roundtrip_header() {
    let mut header = MrcHeader::new(128, 256, 10, MrcMode::Float);
    header.add_label("Test label written by imod-mrc");
    header.amin = -1.5;
    header.amax = 42.0;
    header.amean = 3.14;

    let mut buf = Cursor::new(Vec::new());
    binrw::BinWrite::write_le(&header, &mut buf).unwrap();

    buf.set_position(0);
    let read_back: MrcHeader = binrw::BinRead::read_le(&mut buf).unwrap();

    assert_eq!(read_back.nx, 128);
    assert_eq!(read_back.ny, 256);
    assert_eq!(read_back.nz, 10);
    assert_eq!(read_back.mode, MrcMode::Float as i32);
    assert_eq!(read_back.amin, -1.5);
    assert_eq!(read_back.amax, 42.0);
    assert!((read_back.amean - 3.14).abs() < 1e-6);
    assert_eq!(read_back.label(0).unwrap(), "Test label written by imod-mrc");
    assert!(read_back.is_imod());
}

#[test]
fn roundtrip_byte_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mrc");

    let nx = 16;
    let ny = 8;
    let nz = 2;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Byte);

    // Write
    let mut writer = MrcWriter::create(&path, header).unwrap();
    let slice0: Vec<f32> = (0..nx * ny).map(|i| (i % 256) as f32).collect();
    let slice1: Vec<f32> = (0..nx * ny).map(|i| ((i + 128) % 256) as f32).collect();
    writer.write_slice_f32(&slice0).unwrap();
    writer.write_slice_f32(&slice1).unwrap();
    writer.finish(0.0, 255.0, 127.0).unwrap();

    // Read back
    let mut reader = MrcReader::open(&path).unwrap();
    assert_eq!(reader.header().nx, nx);
    assert_eq!(reader.header().ny, ny);
    assert_eq!(reader.header().nz, nz);

    let read0 = reader.read_slice_f32(0).unwrap();
    let read1 = reader.read_slice_f32(1).unwrap();
    assert_eq!(read0, slice0);
    assert_eq!(read1, slice1);
}

#[test]
fn roundtrip_float_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_float.mrc");

    let nx = 10;
    let ny = 10;
    let nz = 1;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Float);

    let data: Vec<f32> = (0..nx * ny).map(|i| i as f32 * 0.1 - 5.0).collect();

    let mut writer = MrcWriter::create(&path, header).unwrap();
    writer.write_slice_f32(&data).unwrap();
    writer.finish(-5.0, 4.9, 0.0).unwrap();

    let mut reader = MrcReader::open(&path).unwrap();
    let read = reader.read_slice_f32(0).unwrap();
    for (a, b) in read.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn mode_bytes_per_pixel() {
    assert_eq!(MrcMode::Byte.bytes_per_pixel(), 1);
    assert_eq!(MrcMode::Short.bytes_per_pixel(), 2);
    assert_eq!(MrcMode::Float.bytes_per_pixel(), 4);
    assert_eq!(MrcMode::ComplexFloat.bytes_per_pixel(), 8);
    assert_eq!(MrcMode::UShort.bytes_per_pixel(), 2);
    assert_eq!(MrcMode::Rgb.bytes_per_pixel(), 3);
}

#[test]
fn pixel_size_calculation() {
    let mut h = MrcHeader::new(100, 200, 50, MrcMode::Byte);
    h.xlen = 1000.0; // 10 Å/pixel
    h.ylen = 2000.0; // 10 Å/pixel
    h.zlen = 500.0; // 10 Å/pixel

    assert!((h.pixel_size_x() - 10.0).abs() < 1e-6);
    assert!((h.pixel_size_y() - 10.0).abs() < 1e-6);
    assert!((h.pixel_size_z() - 10.0).abs() < 1e-6);
}

#[test]
fn read_big_endian_float_file() {
    // Write a big-endian MRC file manually and verify MrcReader detects the swap
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_be.mrc");

    let nx: i32 = 4;
    let ny: i32 = 4;
    let nz: i32 = 1;
    let npix = (nx * ny) as usize;

    // Build a big-endian header manually
    let mut header = MrcHeader::new(nx, ny, nz, MrcMode::Float);
    header.stamp = [0x11, 0x11, 0x00, 0x00]; // big-endian stamp

    let mut buf = Cursor::new(Vec::new());
    binrw::BinWrite::write_be(&header, &mut buf).unwrap();

    // Append float data in big-endian
    let data: Vec<f32> = (0..npix).map(|i| i as f32 * 1.5).collect();
    for &v in &data {
        buf.write_all(&v.to_be_bytes()).unwrap();
    }

    std::fs::write(&path, buf.into_inner()).unwrap();

    // Read it back — should auto-detect big-endian
    let mut reader = MrcReader::open(&path).unwrap();
    assert!(reader.is_swapped());
    assert_eq!(reader.header().nx, nx);
    assert_eq!(reader.header().ny, ny);

    let read = reader.read_slice_f32(0).unwrap();
    for (a, b) in read.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

#[test]
fn read_big_endian_short_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_be_short.mrc");

    let nx: i32 = 8;
    let ny: i32 = 4;
    let nz: i32 = 1;
    let npix = (nx * ny) as usize;

    let mut header = MrcHeader::new(nx, ny, nz, MrcMode::Short);
    header.stamp = [0x11, 0x11, 0x00, 0x00];

    let mut buf = Cursor::new(Vec::new());
    binrw::BinWrite::write_be(&header, &mut buf).unwrap();

    let data: Vec<i16> = (0..npix as i16).map(|i| i * 100 - 1000).collect();
    for &v in &data {
        buf.write_all(&v.to_be_bytes()).unwrap();
    }

    std::fs::write(&path, buf.into_inner()).unwrap();

    let mut reader = MrcReader::open(&path).unwrap();
    assert!(reader.is_swapped());

    let read = reader.read_slice_f32(0).unwrap();
    for (a, b) in read.iter().zip(data.iter()) {
        assert!((a - *b as f32).abs() < 1e-6);
    }
}

#[test]
fn roundtrip_complex_float_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_complex.mrc");

    let nx = 8;
    let ny = 4;
    let nz = 1;
    let npix = (nx * ny) as usize;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::ComplexFloat);

    let complex_data: Vec<(f32, f32)> = (0..npix)
        .map(|i| (i as f32 * 0.5, -(i as f32) * 0.3))
        .collect();

    let mut writer = MrcWriter::create(&path, header).unwrap();
    writer.write_slice_complex(&complex_data).unwrap();
    writer.finish(-10.0, 10.0, 0.0).unwrap();

    // Read back as complex pairs
    let mut reader = MrcReader::open(&path).unwrap();
    let read_complex = reader.read_slice_complex(0).unwrap();
    assert_eq!(read_complex.len(), npix);
    for (a, b) in read_complex.iter().zip(complex_data.iter()) {
        assert!((a.0 - b.0).abs() < 1e-6, "re mismatch: {} vs {}", a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6, "im mismatch: {} vs {}", a.1, b.1);
    }

    // Read back as magnitude via read_slice_f32
    let mut reader = MrcReader::open(&path).unwrap();
    let magnitudes = reader.read_slice_f32(0).unwrap();
    assert_eq!(magnitudes.len(), npix);
    for (i, &mag) in magnitudes.iter().enumerate() {
        let (re, im) = complex_data[i];
        let expected = (re * re + im * im).sqrt();
        assert!(
            (mag - expected).abs() < 1e-5,
            "magnitude mismatch at {i}: {mag} vs {expected}"
        );
    }
}

#[test]
fn subarea_reading() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_subarea.mrc");

    let nx = 16;
    let ny = 12;
    let nz = 1;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Float);

    let data: Vec<f32> = (0..(nx * ny) as usize)
        .map(|i| i as f32)
        .collect();

    let mut writer = MrcWriter::create(&path, header).unwrap();
    writer.write_slice_f32(&data).unwrap();
    writer.finish(0.0, (nx * ny - 1) as f32, 0.0).unwrap();

    let mut reader = MrcReader::open(&path).unwrap();

    // Read a 4x3 subarea starting at (2, 3)
    let sub = reader.read_subarea_f32(0, 2, 3, 4, 3).unwrap();
    assert_eq!(sub.len(), 4 * 3);

    // Verify values: pixel at (x, y) in full image = y * nx + x
    let nx = nx as usize;
    for row in 0..3usize {
        for col in 0..4usize {
            let expected = ((3 + row) * nx + (2 + col)) as f32;
            let actual = sub[row * 4 + col];
            assert!(
                (actual - expected).abs() < 1e-6,
                "subarea mismatch at ({col},{row}): {actual} vs {expected}"
            );
        }
    }

    // Out-of-bounds should error
    assert!(reader.read_subarea_f32(0, 14, 0, 4, 1).is_err());
    assert!(reader.read_subarea_f32(0, 0, 10, 1, 4).is_err());
}

#[test]
fn y_slice_reading() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_yslice.mrc");

    let nx = 8;
    let ny = 6;
    let nz = 3;
    let header = MrcHeader::new(nx, ny, nz, MrcMode::Float);

    let mut writer = MrcWriter::create(&path, header).unwrap();
    for z in 0..nz as usize {
        let data: Vec<f32> = (0..(nx * ny) as usize)
            .map(|i| (z * 1000 + i) as f32)
            .collect();
        writer.write_slice_f32(&data).unwrap();
    }
    writer.finish(0.0, 2047.0, 0.0).unwrap();

    let mut reader = MrcReader::open(&path).unwrap();
    let y = 2;
    let yslice = reader.read_y_slice_f32(y).unwrap();
    assert_eq!(yslice.len(), nx as usize * nz as usize);

    let nx = nx as usize;
    for z in 0..nz as usize {
        for x in 0..nx {
            let expected = (z * 1000 + y * nx + x) as f32;
            let actual = yslice[z * nx + x];
            assert!(
                (actual - expected).abs() < 1e-6,
                "y-slice mismatch at z={z},x={x}: {actual} vs {expected}"
            );
        }
    }
}
