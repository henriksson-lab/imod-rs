mod common;

use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_RGB, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};

#[test]
fn tif2mrc_roundtrips_the_native_legacy_tiff_path() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
        header.amin = 1.;
        header.amax = 4.;
        header.amean = 2.5;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [1_u8, 2, 3, 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), 1 * (pixels.len()))
                },
                1,
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
        assert!(
            common::imod_cmd("mrc2tif")
                .arg("-o")
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        let result = common::imod_cmd("tif2mrc")
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains(&format!("Opening {} for input\n", tiff.display()))
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut written = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut written), 0);
        assert_eq!(
            (written.nx, written.ny, written.nz, written.mode),
            (2, 2, 1, MRC_MODE_BYTE)
        );
        let mut bytes = [0_u8; 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                written.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr().cast::<u8>(),
                        1 * (bytes.len()),
                    )
                },
                1,
                bytes.len(),
                &mut file,
            ),
            bytes.len()
        );
        drop(file);
        assert_eq!(bytes, pixels);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn tif2mrc_converts_real_rgb_tiff_to_source_average_grayscale() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-rgb-{}", std::process::id());
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let tiff_c = tiff.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(tiff_c, "wb").unwrap();
        let mut ifd = 0;
        let mut data_offset = 0;
        let rgb = [30_u8, 60, 90, 9, 12, 15];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                &mut file,
                2,
                1,
                MRC_MODE_RGB,
                &rgb,
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        let rgb_second = [0_u8, 3, 6, 30, 33, 36];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                &mut file,
                2,
                1,
                MRC_MODE_RGB,
                &rgb_second,
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        drop(file);
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-g")
                .arg(&tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut written = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut written), 0);
        assert_eq!(
            (written.nx, written.ny, written.nz, written.mode),
            (2, 1, 2, MRC_MODE_BYTE)
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                written.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut gray = [0_u8; 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        gray.as_mut_ptr().cast::<u8>(),
                        1 * (gray.len()),
                    )
                },
                1,
                gray.len(),
                &mut file,
            ),
            gray.len()
        );
        drop(file);
        assert_eq!(gray, [188, 140, 131, 161]);
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn tif2mrc_converts_signed_32_bit_tiff_through_the_source_float_path() {
    let stamp = format!("imod-rs-tif2mrc-signed32-{}", std::process::id());
    let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
    let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
    let mut image = vec![0_u8; 162];
    image[..8].copy_from_slice(b"II*\0\x08\0\0\0");
    image[8..10].copy_from_slice(&11_u16.to_le_bytes());
    for (entry, (tag, kind, count, value)) in [
        (256_u16, 4_u16, 1_u32, 2_u32),
        (257, 4, 1, 2),
        (258, 3, 1, 32),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, 146),
        (277, 3, 1, 1),
        (278, 4, 1, 2),
        (279, 4, 1, 16),
        (284, 3, 1, 1),
        (339, 3, 1, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let start = 10 + entry * 12;
        image[start..start + 2].copy_from_slice(&tag.to_le_bytes());
        image[start + 2..start + 4].copy_from_slice(&kind.to_le_bytes());
        image[start + 4..start + 8].copy_from_slice(&count.to_le_bytes());
        image[start + 8..start + 12].copy_from_slice(&value.to_le_bytes());
    }
    image[146..150].copy_from_slice(&(-1_i32).to_le_bytes());
    image[150..154].copy_from_slice(&2_i32.to_le_bytes());
    image[154..158].copy_from_slice(&(-300_000_000_i32).to_le_bytes());
    image[158..162].copy_from_slice(&(400_000_000_i32).to_le_bytes());
    std::fs::write(&tiff, image).unwrap();
    let conversion = common::imod_cmd("tif2mrc")
        .arg(&tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        conversion.status.success(),
        "{}",
        String::from_utf8_lossy(&conversion.stderr)
    );
    unsafe {
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (2, 2, 1, 2));
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut values = [0_f32; 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        values.as_mut_ptr().cast::<u8>(),
                        4 * (values.len()),
                    )
                },
                4,
                values.len(),
                &mut file,
            ),
            values.len()
        );
        drop(file);
        assert_eq!(values, [-300_000_000., 400_000_000., -1., 2.]);
    }
    std::fs::remove_file(tiff).unwrap();
    std::fs::remove_file(output).unwrap();
}

#[test]
fn tif2mrc_reads_every_directory_in_a_native_legacy_tiff_stack() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-stack-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_BYTE), 0);
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), 1 * (pixels.len()))
                },
                1,
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
        assert!(
            common::imod_cmd("mrc2tif")
                .arg("-o")
                .arg("-s")
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(&output, b"pre-existing multi-page output").unwrap();
        let conversion = common::imod_cmd("tif2mrc")
            .args(["-o", "1,1"])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            conversion.status.success(),
            "{}",
            String::from_utf8_lossy(&conversion.stderr)
        );
        let stdout = String::from_utf8_lossy(&conversion.stdout);
        assert!(
            stdout.contains("Reading multi-paged TIFF file."),
            "{stdout}"
        );
        assert!(
            stdout.contains("Warning: output file size option ignored for multi-paged file"),
            "{stdout}"
        );
        assert!(
            stdout.contains("Converting 2 images size 2 x 2"),
            "{stdout}"
        );
        assert!(
            stdout.contains("Min = 129, Max = 136, Mean = 132.5"),
            "{stdout}"
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut written = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut written), 0);
        assert_eq!((written.nx, written.ny, written.nz), (2, 2, 2));
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                written.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut bytes = [0_u8; 8];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr().cast::<u8>(),
                        1 * (bytes.len()),
                    )
                },
                1,
                bytes.len(),
                &mut file,
            ),
            bytes.len()
        );
        drop(file);
        assert_eq!(bytes, pixels);
        let backup = std::path::PathBuf::from(format!("{}~", output.display()));
        assert_eq!(
            std::fs::read(&backup).unwrap(),
            b"pre-existing multi-page output"
        );
        for path in [input, tiff, output, backup] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn tif2mrc_applies_source_background_inversion_and_subtraction() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-bg-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let background = std::env::temp_dir().join(format!("{stamp}-bg.mrc"));
        let input_tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let background_tiff = std::env::temp_dir().join(format!("{stamp}-bg.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        for (path, pixels) in [
            (&input, [10_u8, 11, 12, 13]),
            (&background, [1_u8, 2, 3, 4]),
        ] {
            let name = path.to_str().unwrap();
            let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(name, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!(
                imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            pixels.as_ptr().cast::<u8>(),
                            1 * (pixels.len()),
                        )
                    },
                    1,
                    pixels.len(),
                    &mut file,
                ),
                pixels.len()
            );
            drop(file);
        }
        assert!(
            common::imod_cmd("mrc2tif")
                .arg("-o")
                .arg(&input)
                .arg(&input_tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("mrc2tif")
                .arg("-o")
                .arg(&background)
                .arg(&background_tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-b")
                .arg(&background_tiff)
                .arg(&input_tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let output_name = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_name, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut pixels = [0_u8; 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        pixels.as_mut_ptr().cast::<u8>(),
                        1 * (pixels.len()),
                    )
                },
                1,
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
        assert_eq!(pixels, [13, 13, 13, 13]);
        for path in [input, background, input_tiff, background_tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn tif2mrc_reports_source_specific_pixel_spacing_conflict() {
    let stamp = format!("imod-rs-tif2mrc-pixel-conflict-{}", std::process::id());
    let missing_input = std::env::temp_dir().join(format!("{stamp}.tif"));
    let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
    let result = common::imod_cmd("tif2mrc")
        .args([
            "-p",
            "1.5",
            "-P",
            missing_input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stdout).contains("You cannot enter both -p and -P"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!output.exists());
}

#[test]
fn tif2mrc_rejects_background_for_real_multi_directory_tiff() {
    unsafe {
        let stamp = format!(
            "imod-rs-tif2mrc-multipage-background-{}",
            std::process::id()
        );
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let name = tiff.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(name, "wb").unwrap();
        let mut ifd = 0;
        let mut data_offset = 0;
        for pixels in [[1_u8, 2], [3, 4]] {
            assert_eq!(
                imod_rs::imod::mrc::tiff::tiff_write_image(
                    &mut file,
                    2,
                    1,
                    MRC_MODE_BYTE,
                    &pixels,
                    &mut ifd,
                    &mut data_offset,
                    0.,
                    255.,
                ),
                0
            );
        }
        drop(file);
        let result = common::imod_cmd("tif2mrc")
            .args(["-b", tiff.to_str().unwrap()])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success());
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.contains("Reading multi-paged TIFF file."),
            "{stdout}"
        );
        assert!(
            stdout.contains("Background subtraction not supported for multi-paged images."),
            "{stdout}"
        );
        assert!(!output.exists());
        std::fs::remove_file(tiff).unwrap();
    }
}

#[test]
fn tif2mrc_accepts_negative_chunk_criterion_for_real_legacy_tiff() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-negative-chunks-{}", std::process::id());
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff_c = tiff.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(tiff_c, "wb").unwrap();
        let mut ifd = 0;
        let mut data_offset = 0;
        let pixels = [5_u8, 8];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                &mut file,
                2,
                1,
                MRC_MODE_BYTE,
                &pixels,
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        drop(file);
        let result = common::imod_cmd("tif2mrc")
            .args(["-t", "-1"])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 1, 1, MRC_MODE_BYTE)
        );
        assert_ne!(header.bytes_signed, 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut written = [0_u8; 2];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        written.as_mut_ptr().cast::<u8>(),
                        1 * (written.len()),
                    )
                },
                1,
                written.len(),
                &mut file,
            ),
            written.len()
        );
        drop(file);
        // Source byte-mode output is signed and therefore shifts the raw
        // storage by 128 while preserving logical image values 5 and 8.
        assert_eq!(written, [133, 136]);
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Real-TIFF-variant parity fixtures.
//
// Every constant below is the hex dump of a real TIFF file produced by a real
// encoder and then verified byte-for-byte against the native IMOD `tif2mrc`
// (`/tmp/imod-reference-build/mrc/tif2mrc`).  Provenance:
//
//   GRAY8_MINISBLACK  Pillow 12.3 `Image.fromarray(g8, "L").save(...)`
//   GRAY8_MINISWHITE  the same file with libtiff `tiffset -s 262 0`
//   GRAY8_LZW         Pillow, `compression="tiff_lzw"`
//   GRAY8_TILED       libtiff `tiffcp -t -w 16 -l 16` of GRAY8_MINISBLACK
//   GRAY8_BIGTIFF     Pillow, `big_tiff=True` (the file starts `II+\0`)
//   USHORT16_DEFLATE  Pillow, `compression="tiff_adobe_deflate"`
//   USHORT16_PLAIN    hand-built uncompressed IFD, SampleFormat = 1 (UINT)
//   SHORT16_SIGNED    hand-built uncompressed IFD, SampleFormat = 2 (INT)
//   RGBA8             Pillow `Image.fromarray(rgba, "RGBA")` (4 samples,
//                     ExtraSamples = unassociated alpha)
//   PALETTE8          Pillow palette-mode save (PhotometricInterpretation = 3)
//
// The pixel patterns are deterministic: for the 16x12 images, index
// `i = row * 16 + col` gives byte `(3 * i + 5) % 256`, unsigned short
// `(337 * i + 1000) % 65536`, signed short `271 * i - 20000`, and RGBA
// `(3i, 5i, 7i, 11i) % 256`.  The palette image stores index `i % 64`.
// ---------------------------------------------------------------------------

/// Decode one of the hex fixture constants above into TIFF file bytes.
fn tiff_fixture_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// Write one hex fixture to a uniquely named temporary file and return its path.
///
/// The name carries the process id as well as the caller's stamp: every caller
/// passes a fixed literal (`"bilevel"`, `"mismatch-byte"`, ...), so without it
/// two concurrent test processes write the same `/tmp` path and race.  That is
/// not hypothetical -- it made three of this file's tests fail whenever a
/// second gate ran alongside one, while passing 31/31 in isolation.
fn write_tiff_fixture(stamp: &str, hex: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-tif2mrc-{stamp}-{}.tif",
        std::process::id()
    ));
    std::fs::write(&path, tiff_fixture_bytes(hex)).unwrap();
    path
}

/// Run `tif2mrc` on one fixture and return (stdout, MRC header, MRC data bytes).
fn convert_one_fixture(stamp: &str, hex: &str, args: &[&str]) -> (String, MrcHeader, Vec<u8>) {
    convert_one_fixture_with_environment(stamp, hex, args, &[])
}

/// As `convert_one_fixture`, with an isolated backend selection for its child
/// process.  Do not mutate the test process environment: integration tests
/// may run concurrently and the selected TIFF backend is process-global.
fn convert_one_fixture_with_environment(
    stamp: &str,
    hex: &str,
    args: &[&str],
    environment: &[(&str, &str)],
) -> (String, MrcHeader, Vec<u8>) {
    let tiff = write_tiff_fixture(stamp, hex);
    let output = std::env::temp_dir().join(format!("imod-rs-tif2mrc-{stamp}-out.mrc"));
    let _ = std::fs::remove_file(&output);
    let mut command = common::imod_cmd("tif2mrc");
    command.args(args).arg(&tiff).arg(&output);
    for &(key, value) in environment {
        command.env(key, value);
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "stdout {} stderr {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    let bytes = std::fs::read(&output).unwrap();
    let header = unsafe {
        let name = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(name, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        drop(file);
        header
    };
    std::fs::remove_file(tiff).unwrap();
    std::fs::remove_file(output).unwrap();
    (stdout, header, bytes[1024..].to_vec())
}

/// The 16x12 byte image the grayscale fixtures hold, in MRC row order
/// (`tif2mrc` inverts the TIFF rows).
fn expected_byte_image() -> Vec<u8> {
    let mut data = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            data.push(((3 * (row * 16 + col) + 5) % 256) as u8);
        }
    }
    data
}

/// The same image after the signed-byte storage shift `mrc_head_write` applies.
fn expected_signed_byte_image() -> Vec<u8> {
    expected_byte_image()
        .into_iter()
        .map(|value| value.wrapping_sub(128))
        .collect()
}

#[test]
fn tif2mrc_prints_min_max_mean_for_a_single_page_file() {
    // `tif2mrc.c:612` prints the statistics line on the ordinary (non
    // multi-paged) path too, right before the final `mrc_head_write`.
    let (stdout, header, data) = convert_one_fixture("minmaxmean-line", GRAY8_MINISBLACK, &[]);
    assert!(
        stdout.ends_with("Min = 0, Max = 254, Mean = 116.833\n"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    assert_eq!(data, expected_signed_byte_image());
}

#[test]
fn tif2mrc_reads_miniswhite_photometric_as_grayscale() {
    let (stdout, header, data) = convert_one_fixture("miniswhite", GRAY8_MINISWHITE, &[]);
    assert!(
        stdout.contains("Min = 0, Max = 254, Mean = 116.833"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    // PhotometricInterpretation 0 divides to 0 in the `tif2mrc.c:481` test, so
    // it takes the same byte path as min-is-black and the samples are not
    // inverted.
    assert_eq!(data, expected_signed_byte_image());
}

#[test]
fn tif2mrc_reads_lzw_compressed_byte_tiff() {
    let (stdout, header, data) = convert_one_fixture("lzw", GRAY8_LZW, &[]);
    assert!(
        stdout.contains("Min = 0, Max = 254, Mean = 116.833"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    assert_eq!(data, expected_signed_byte_image());
}

#[test]
fn tif2mrc_reads_tiled_byte_tiff() {
    let (stdout, header, data) = convert_one_fixture("tiled", GRAY8_TILED, &[]);
    assert!(
        stdout.contains("Min = 0, Max = 254, Mean = 116.833"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    assert_eq!(data, expected_signed_byte_image());
}

#[test]
fn tif2mrc_reads_bigtiff_byte_file() {
    assert_eq!(&tiff_fixture_bytes(GRAY8_BIGTIFF)[..4], b"II+\0");
    let (stdout, header, data) = convert_one_fixture("bigtiff", GRAY8_BIGTIFF, &[]);
    assert!(
        stdout.contains("Min = 0, Max = 254, Mean = 116.833"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    assert_eq!(data, expected_signed_byte_image());
}

#[test]
fn tif2mrc_reads_deflate_compressed_unsigned_short_tiff() {
    let (stdout, header, data) = convert_one_fixture("deflate16", USHORT16_DEFLATE, &[]);
    assert!(
        stdout.contains("Min = 1000, Max = 65367, Mean = 33183.5"),
        "{stdout}"
    );
    // `manageMode` (`tif2mrc.c:649`) promotes 16-bit data the file declares
    // unsigned to MRC_MODE_USHORT when neither -s nor -u/-d was given.
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 6)
    );
    assert_eq!(
        (header.amin, header.amax, header.amean),
        (1000., 65367., 33183.5)
    );
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            expected.extend_from_slice(
                &(((337 * (row * 16 + col) + 1000) % 65536) as u16).to_ne_bytes(),
            );
        }
    }
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_stores_sampleformat_int_16_bit_data_as_signed_mode() {
    let (stdout, header, data) = convert_one_fixture("sint16", SHORT16_SIGNED, &[]);
    assert!(
        stdout.contains("Min = -20000, Max = 31761, Mean = 5880.5"),
        "{stdout}"
    );
    // SAMPLEFORMAT_INT leaves iifile->type at IITYPE_SHORT, so the
    // `IITYPE_USHORT` promotion in `manageMode` does not fire.
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 1)
    );
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            expected
                .extend_from_slice(&((271 * (row * 16 + col) as i32 - 20000) as i16).to_ne_bytes());
        }
    }
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_stores_sampleformat_uint_16_bit_data_as_unsigned_mode() {
    let (_, signed_header, signed_data) =
        convert_one_fixture("uint16-cmp-signed", SHORT16_SIGNED, &[]);
    let (stdout, header, data) = convert_one_fixture("uint16", USHORT16_PLAIN, &[]);
    assert!(
        stdout.contains("Min = 1000, Max = 65367, Mean = 33183.5"),
        "{stdout}"
    );
    assert_eq!(header.mode, 6);
    assert_eq!(signed_header.mode, 1);
    assert_eq!(data.len(), signed_data.len());
}

#[test]
fn tif2mrc_forces_signed_short_mode_for_unsigned_file_with_dash_s() {
    let (stdout, header, _) = convert_one_fixture("uint16-forced", USHORT16_PLAIN, &["-s"]);
    // `tif2mrc.c:650`: -s suppresses the unsigned promotion, so the same file
    // lands in mode 1 and the statistics wrap through signed shorts.
    assert_eq!(header.mode, 1);
    assert!(
        stdout.contains("Min = -32521, Max = 32678, Mean = 74.1667"),
        "{stdout}"
    );
}

#[test]
fn tif2mrc_drops_the_alpha_sample_of_an_rgba_tiff() {
    let (stdout, header, data) = convert_one_fixture("rgba", RGBA8, &[]);
    // RGB output takes the `tif2mrc.c:604` branch, which fixes the statistics
    // and prints no Min/Max/Mean line.
    assert!(!stdout.contains("Min ="), "{stdout}");
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 16)
    );
    assert_eq!((header.amin, header.amax, header.amean), (0., 255., 128.));
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            let i = row * 16 + col;
            expected.push(((3 * i) % 256) as u8);
            expected.push(((5 * i) % 256) as u8);
            expected.push(((7 * i) % 256) as u8);
        }
    }
    assert_eq!(data.len(), 16 * 12 * 3);
    assert_eq!(data, expected);
}

/// `t_graya16.tif`, written by tifffile 2026.3.3 from a uint16 `(2, 4, 2)`
/// array with `photometric="minisblack"` and `extrasamples="unassalpha"`.
/// This is deliberately grayscale plus alpha rather than RGBA: IMOD accepts
/// it as two 16-bit grayscale sections (`iitif.c:493-498`), while 16-bit RGB
/// and RGBA are rejected by its source type check.
const GRAYA16: &str = concat!(
    "49492a00080000000f00000104000100000004000000010104000100000002000000",
    "02010300020000001000100003010300010000000100000006010300010000000100",
    "00001101040001000000e00000001501030001000000020000001601040001000000",
    "020000001701040001000000200000001a01050001000000c20000001b0105000100",
    "0000ca0000001c010300010000000100000028010300010000000100000031010200",
    "0c000000d20000005201030001000000020000000000000001000000010000000100",
    "0000010000007469666666696c652e7079000000e80360ead00750c3b80b409ca00f",
    "30758813204e70171027581b8813401f0000"
);

#[test]
fn tif2mrc_libtiff_exposes_16_bit_grayscale_alpha_as_two_sections() {
    let (stdout, header, data) = convert_one_fixture("graya16-parity", GRAYA16, &[]);
    assert!(
        stdout.contains("Converting 2 images size 4 x 2"),
        "{stdout}"
    );
    assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 2, 2, 6));
    let expected = [
        5000_u16, 6000, 7000, 8000, 1000, 2000, 3000, 4000, 20000, 10000, 5000, 0, 60000, 50000,
        40000, 30000,
    ]
    .into_iter()
    .flat_map(u16::to_ne_bytes)
    .collect::<Vec<_>>();
    assert_eq!(data, expected);
}

/// `tiff` 0.11.3 exposes `ColorType::GrayA(16)` but rejects it in
/// `Image::expand_chunk` before yielding pixels.  The opt-in reader must not
/// silently drop the alpha channel, because that would differ from the two
/// sections that IMOD/libtiff produces above.  Keep this explicit rejection
/// until the dependency supports GrayA samples or a complete source-shaped
/// replacement handles contiguous and planar strips/tiles/compression.
#[cfg(feature = "rust-tiff")]
#[test]
fn tif2mrc_rust_reader_explicitly_rejects_16_bit_grayscale_alpha() {
    let tiff = write_tiff_fixture("graya16-rust-unsupported", GRAYA16);
    let output = std::env::temp_dir().join("imod-rs-tif2mrc-graya16-rust-unsupported.mrc");
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("tif2mrc")
        .env("IMOD_RS_TIFF_BACKEND", "rust")
        .arg(&tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("Rust TIFF reader could not open or decode requested TIFF"),
        "stdout {} stderr {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output.exists());
    std::fs::remove_file(tiff).unwrap();
}

#[cfg(feature = "rust-tiff")]
#[test]
fn tif2mrc_rust_backend_reads_planar_separate_rgb_tiff() {
    let (stdout, header, data) = convert_one_fixture_with_environment(
        "planar-rgb-rust",
        RGB8_PLANAR_SEPARATE,
        &[],
        &[("IMOD_RS_TIFF_BACKEND", "rust")],
    );
    assert!(!stdout.contains("Min ="), "{stdout}");
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 16)
    );
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            let i = row * 16 + col;
            expected.extend([
                ((3 * i) % 256) as u8,
                ((5 * i) % 256) as u8,
                ((7 * i) % 256) as u8,
            ]);
        }
    }
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_converts_rgba_tiff_to_grayscale_with_dash_g() {
    let (stdout, header, data) = convert_one_fixture("rgba-gray", RGBA8, &["-g"]);
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    assert!(
        stdout.contains("Min = 0, Max = 218, Mean = 120.365"),
        "{stdout}"
    );
    // `convertrgb` (`tif2mrc.c:678`) averages the three stored samples with
    // integer division after the reader has already dropped alpha.
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            let i = row * 16 + col;
            let gray = ((((3 * i) % 256) + ((5 * i) % 256) + ((7 * i) % 256)) / 3) as u8;
            expected.push(gray.wrapping_sub(128));
        }
    }
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_reads_palette_tiff_through_the_library_index_path() {
    let (stdout, header, data) = convert_one_fixture("palette", PALETTE8, &[]);
    // The libtiff reader reports mode MRC_MODE_BYTE for this file, so
    // `tiff_open_file` records PhotometricInterpretation 1 and the colormap
    // expansion at `tif2mrc.c:485` never runs: the stored index values are
    // written straight through.
    assert!(
        stdout.contains("Min = 0, Max = 63, Mean = 31.5"),
        "{stdout}"
    );
    assert_eq!(
        (header.nx, header.ny, header.nz, header.mode),
        (16, 12, 1, 0)
    );
    let mut expected = Vec::new();
    for row in (0..12_usize).rev() {
        for col in 0..16_usize {
            expected.push((((row * 16 + col) % 64) as u8).wrapping_sub(128));
        }
    }
    assert_eq!(data, expected);
}

/// The Rust TIFF crate does not itself expand indexed palettes. IMOD's
/// `tif2mrc` library path keeps the original indices instead, so compare that
/// observable MRC result directly with the opt-in Rust reader.
#[cfg(feature = "rust-tiff")]
#[test]
fn tif2mrc_rust_reader_preserves_palette_indices_like_libtiff() {
    let tiff = write_tiff_fixture("palette-rust-cross-read", PALETTE8);
    let parity_output = std::env::temp_dir().join("imod-rs-tif2mrc-palette-rust-parity.mrc");
    let rust_output = std::env::temp_dir().join("imod-rs-tif2mrc-palette-rust.mrc");
    let _ = std::fs::remove_file(&parity_output);
    let _ = std::fs::remove_file(&rust_output);
    for (backend, output) in [("parity", &parity_output), ("rust", &rust_output)] {
        let result = common::imod_cmd("tif2mrc")
            .env("IMOD_RS_TIFF_BACKEND", backend)
            .arg(&tiff)
            .arg(output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{backend}: stdout {} stderr {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(
        std::fs::read(&rust_output).unwrap(),
        std::fs::read(&parity_output).unwrap(),
        "the Rust reader must preserve the library reader's palette-index output"
    );
    std::fs::remove_file(tiff).unwrap();
    std::fs::remove_file(parity_output).unwrap();
    std::fs::remove_file(rust_output).unwrap();
}

/// IMOD's `iiTIFFCheck` specifically accepts true unsigned 4-bit grayscale
/// (`iitif.c:420-426`) and its section reader expands each nibble to a byte.
/// Cross-check the full executable result because the `tiff` crate intentionally
/// leaves sub-byte samples packed in its U8 decoding buffer.
#[cfg(feature = "rust-tiff")]
#[test]
fn tif2mrc_rust_reader_expands_four_bit_grayscale_like_libtiff() {
    for (name, fixture) in [
        ("four-bit-msb", GRAY4_MINISBLACK),
        ("four-bit-lsb", GRAY4_MINISBLACK_LSB),
    ] {
        let tiff = write_tiff_fixture(&format!("{name}-rust-cross-read"), fixture);
        let parity_output = std::env::temp_dir().join(format!("imod-rs-tif2mrc-{name}-parity.mrc"));
        let rust_output = std::env::temp_dir().join(format!("imod-rs-tif2mrc-{name}-rust.mrc"));
        let _ = std::fs::remove_file(&parity_output);
        let _ = std::fs::remove_file(&rust_output);
        for (backend, output) in [("parity", &parity_output), ("rust", &rust_output)] {
            let result = common::imod_cmd("tif2mrc")
                .env("IMOD_RS_TIFF_BACKEND", backend)
                .arg(&tiff)
                .arg(output)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{name}/{backend}: stdout {} stderr {}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
        assert_eq!(
            std::fs::read(&rust_output).unwrap(),
            std::fs::read(&parity_output).unwrap(),
            "the Rust reader must expand {name} samples in the same fill order as libtiff"
        );
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(parity_output).unwrap();
        std::fs::remove_file(rust_output).unwrap();
    }
}

#[test]
fn tif2mrc_writes_the_pre_read_header_before_a_type_mismatch_exit() {
    // `tif2mrc.c:403` writes the first header with the still-zero `xsize`,
    // `ysize` and `mode`, so an early `exitError` leaves those zeros on disk.
    let byte_tiff = write_tiff_fixture("mismatch-byte", GRAY8_MINISBLACK);
    let short_tiff = write_tiff_fixture("mismatch-short", USHORT16_PLAIN);
    let output = std::env::temp_dir().join(format!(
        "imod-rs-tif2mrc-mismatch-{}.mrc",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("tif2mrc")
        .arg(&byte_tiff)
        .arg(&short_tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("All files must have the same data type."),
        "{stdout}"
    );
    // `exitError` supplies its own newline, so nothing follows the message.
    assert!(stdout.ends_with("data type.\n"), "{stdout:?}");
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(bytes.len(), 1024 + 192);
    let mut header = [0_i32; 4];
    for (index, slot) in header.iter_mut().enumerate() {
        *slot = i32::from_ne_bytes(bytes[4 * index..4 * index + 4].try_into().unwrap());
    }
    assert_eq!(header, [0, 0, 2, 0]);
    for path in [byte_tiff, short_tiff, output] {
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn tif2mrc_names_the_unopenable_file_in_its_error() {
    let missing =
        std::env::temp_dir().join(format!("imod-rs-tif2mrc-absent-{}.tif", std::process::id()));
    let output =
        std::env::temp_dir().join(format!("imod-rs-tif2mrc-absent-{}.mrc", std::process::id()));
    let result = common::imod_cmd("tif2mrc")
        .arg(&missing)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    // `tif2mrc.c:241` formats the offending path into the message.
    assert!(
        stdout.ends_with(&format!("Couldn't open {}.\n", missing.display())),
        "{stdout:?}"
    );
    assert!(!output.exists());
}

#[test]
fn tif2mrc_ntsc_grayscale_rounds_through_a_float_accumulator() {
    // `tif2mrc.c:671-675` accumulates into a `float fpixel` one weighted term
    // at a time, and the weights are double constants, so the running sum is
    // rounded back to float after each term.  These four RGB triples are ones
    // where that differs from evaluating the whole sum in f32: the source
    // gives 4, 11, 16, 13 and a single f32 expression gives 3, 10, 15, 12.
    let (stdout, header, data) = convert_one_fixture("ntsc-rounding", NTSC_RGB4, &["-G"]);
    assert!(
        stdout.ends_with("Min = 4, Max = 16, Mean = 11\n"),
        "{stdout:?}"
    );
    assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 1, 1, 0));
    let expected: Vec<u8> = [4_u8, 11, 16, 13]
        .iter()
        .map(|value| value.wrapping_sub(128))
        .collect();
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_equal_weight_grayscale_uses_integer_division() {
    // The non-NTSC branch (`tif2mrc.c:678-683`) sums into an `int` and divides
    // by 3, so the same four pixels truncate to 3, 10, 19, 7.
    let (stdout, _, data) = convert_one_fixture("equal-weight", NTSC_RGB4, &["-g"]);
    assert!(
        stdout.ends_with("Min = 3, Max = 19, Mean = 9.75\n"),
        "{stdout:?}"
    );
    let expected: Vec<u8> = [3_u8, 10, 19, 7]
        .iter()
        .map(|value| value.wrapping_sub(128))
        .collect();
    assert_eq!(data, expected);
}

#[test]
fn tif2mrc_reads_a_bilevel_tiff_through_the_zero_pixel_size_legacy_path() {
    // A 1-bit TIFF is rejected by iiTIFFCheck, so `tif2mrc` falls back to the
    // b3dtiff reader in `IMOD/mrc/tiff.c`.  There `pixSize = BitsPerSample / 8`
    // is 0, so `nleft` is 0 and no strip bytes are ever transferred; the image
    // that reaches the caller is the untouched allocation.  This fixture omits
    // the BitsPerSample tag (as Pillow's bilevel writer does), which leaves
    // `Tf_info.BitsPerSample` at 0 and skips the 1-bit expansion entirely, so
    // native emits an all-zero image and this locks that in.
    let tiff = write_tiff_fixture("bilevel", BILEVEL1);
    let output = std::env::temp_dir().join(format!(
        "imod-rs-tif2mrc-bilevel-{}.mrc",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("tif2mrc")
        .arg(&tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.ends_with("Min = 0, Max = 0, Mean = 0\n"),
        "{stdout:?}"
    );
    // iiTIFFCheck reports the rejection once for the multi-page probe and once
    // for the conversion loop.
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: iiTIFFCheck - Unsupported type of TIFF file\n\
         ERROR: iiTIFFCheck - Unsupported type of TIFF file\n"
    );
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(bytes.len(), 1024 + 16 * 12);
    assert!(bytes[1024..].iter().all(|byte| *byte == 128));
    for path in [tiff, output] {
        std::fs::remove_file(path).unwrap();
    }
}

/// `iiTIFFCheck` does not admit 1-bit input (`iitif.c:417-426`), so this is
/// deliberately outside the opt-in reader rather than a place to invent a
/// byte-expansion result.  The default route retains the historic legacy
/// fallback tested above; the explicitly selected Rust route must reject it,
/// never silently re-enter that parity path.
#[cfg(feature = "rust-tiff")]
#[test]
fn tif2mrc_rust_reader_explicitly_rejects_bilevel_tiff() {
    let tiff = write_tiff_fixture("bilevel-rust-reject", BILEVEL1);
    let output = std::env::temp_dir().join(format!(
        "imod-rs-tif2mrc-bilevel-rust-reject-{}.mrc",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("tif2mrc")
        .env("IMOD_RS_TIFF_BACKEND", "rust")
        .arg(&tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: Rust-native backend - Rust TIFF reader could not open or decode requested TIFF\n"
    );
    assert!(!output.exists());
    std::fs::remove_file(tiff).unwrap();
}

/// `t_gray8.tif`, produced as described in the module comment above.
const GRAY8_MINISBLACK: &str = concat!(
    "49492a0008000000090000010400010000001000000001010400010000000c0000000201030001000000080000000301",
    "0300010000000100000006010300010000000100000011010400010000007a00000016010400010000000c0000001701",
    "040001000000c00000001c01030001000000010000000000000005080b0e1114171a1d202326292c2f3235383b3e4144",
    "474a4d505356595c5f6265686b6e7174777a7d808386898c8f9295989b9ea1a4a7aaadb0b3b6b9bcbfc2c5c8cbced1d4",
    "d7dadde0e3e6e9eceff2f5f8fbfe0104070a0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e6164",
    "676a6d707376797c7f8285888b8e9194979a9da0a3a6a9acafb2b5b8bbbec1c4c7cacdd0d3d6d9dcdfe2e5e8ebeef1f4",
    "f7fafd000306090c0f1215181b1e2124272a2d303336393c3f42"
);

/// `t_white8.tif`, produced as described in the module comment above.
const GRAY8_MINISWHITE: &str = concat!(
    "49492a003a010000090000010400010000001000000001010400010000000c0000000201030001000000080000000301",
    "0300010000000100000006010300010000000100000011010400010000007a00000016010400010000000c0000001701",
    "040001000000c00000001c01030001000000010000000000000005080b0e1114171a1d202326292c2f3235383b3e4144",
    "474a4d505356595c5f6265686b6e7174777a7d808386898c8f9295989b9ea1a4a7aaadb0b3b6b9bcbfc2c5c8cbced1d4",
    "d7dadde0e3e6e9eceff2f5f8fbfe0104070a0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e6164",
    "676a6d707376797c7f8285888b8e9194979a9da0a3a6a9acafb2b5b8bbbec1c4c7cacdd0d3d6d9dcdfe2e5e8ebeef1f4",
    "f7fafd000306090c0f1215181b1e2124272a2d303336393c3f4209000001030001000000100000000101030001000000",
    "0c0000000201030001000000080000000301030001000000010000000601030001000000000000001101040001000000",
    "7a00000016010300010000000c0000001701040001000000c00000001c010300010000000100000000000000"
);

/// `t_lzw8.tif`, produced as described in the module comment above.
const GRAY8_LZW: &str = concat!(
    "49492a00e400000080014100b0704428170d0744023130a4582f190d4703b1f104884725134a0532b164b85f31194d06",
    "b371c4e8773d1f5008343225188f49255309b4f28548a7552b560b35b2e578bf6131590cb67345a8d76d375c0e3733a5",
    "d8ef793d5f0fb7f00408070503420130b064381f11094502b170c468371d0f4804323124984f29154b05b2f184c86735",
    "1b4e0733b1e4f87f41215108b4724528974d27540a3532a558af592d570bb5f30588c765335a0d36b365b8df71395d0e",
    "b773c5e8f77d3f4000303024180f09054301b0f0844827150b460331b0e4783f21404000090000010300010000001000",
    "000001010300010000000c00000002010300010000000800000003010300010000000500000006010300010000000100",
    "000011010400010000000800000016010300010000000c0000001701040001000000db0000001c010300010000000100",
    "000000000000"
);

/// `t_tiled8.tif`, produced as described in the module comment above.
const GRAY8_TILED: &str = concat!(
    "49492a000801000005080b0e1114171a1d202326292c2f3235383b3e4144474a4d505356595c5f6265686b6e7174777a",
    "7d808386898c8f9295989b9ea1a4a7aaadb0b3b6b9bcbfc2c5c8cbced1d4d7dadde0e3e6e9eceff2f5f8fbfe0104070a",
    "0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e6164676a6d707376797c7f8285888b8e9194979a",
    "9da0a3a6a9acafb2b5b8bbbec1c4c7cacdd0d3d6d9dcdfe2e5e8ebeef1f4f7fafd000306090c0f1215181b1e2124272a",
    "2d303336393c3f4200000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000b0000010300010000001000000001010300010000000c00",
    "000002010300010000000800000003010300010000000100000006010300010000000100000012010300010000000100",
    "00001c010300010000000100000042010300010000001000000043010300010000001000000044010400010000000800",
    "000045010400010000000001000000000000"
);

/// `t_big8.tif`, produced as described in the module comment above.
const GRAY8_BIGTIFF: &str = concat!(
    "49492b000800000010000000000000000900000000000000000104000100000000000000100000000000000001010400",
    "01000000000000000c000000000000000201030001000000000000000800000000000000030103000100000000000000",
    "01000000000000000601030001000000000000000100000000000000110104000100000000000000d400000000000000",
    "1601040001000000000000000c00000000000000170104000100000000000000c0000000000000001c01030001000000",
    "000000000100000000000000000000000000000005080b0e1114171a1d202326292c2f3235383b3e4144474a4d505356",
    "595c5f6265686b6e7174777a7d808386898c8f9295989b9ea1a4a7aaadb0b3b6b9bcbfc2c5c8cbced1d4d7dadde0e3e6",
    "e9eceff2f5f8fbfe0104070a0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e6164676a6d707376",
    "797c7f8285888b8e9194979a9da0a3a6a9acafb2b5b8bbbec1c4c7cacdd0d3d6d9dcdfe2e5e8ebeef1f4f7fafd000306",
    "090c0f1215181b1e2124272a2d303336393c3f42"
);

/// `t_zip16.tif`, produced as described in the module comment above.
const USHORT16_DEFLATE: &str = concat!(
    "49492a0094010000789c0180017ffee80339058a06db072c097d0ace0b1f0d700ec10f12116312b41305155616a717f8",
    "18491a9a1beb1c3c1e8d1fde202f228023d12422267327c428152a662bb72c082e592faa30fb314c339d34ee353f3790",
    "38e139323b833cd43d253f7640c74118436944ba450b475c48ad49fe4a4f4ca04df14e42509351e45235548655d75628",
    "587959ca5a1b5c6c5dbd5e0e605f61b06201645265a366f4674569966ae76b386d896eda6f2b717c72cd731e756f76c0",
    "771179627ab37b047d557ea67ff78048829983ea843b868c87dd882e8a7f8bd08c218e728fc39014926593b694079658",
    "97a998fa994b9b9c9ced9d3e9f8fa0e0a131a382a4d3a524a775a8c6a917ab68acb9ad0aaf5bb0acb1fdb24eb49fb5f0",
    "b641b892b9e3ba34bc85bdd6be27c078c1c9c21ac46bc5bcc60dc85ec9afca00cc51cda2cef3cf44d195d2e6d337d588",
    "d6d9d72ad97bdaccdb1ddd6edebfdf10e161e2b2e303e554e6a5e7f6e847ea98ebe9ec3aee8befdcf02df27ef3cff420",
    "f671f7c2f813fa64fbb5fc06fe57ff2a66bf7a00090000010300010000001000000001010300010000000c0000000201",
    "030001000000100000000301030001000000080000000601030001000000010000001101040001000000080000001601",
    "0300010000000c00000017010400010000008b0100001c010300010000000100000000000000"
);

/// `t_sint16.tif`, produced as described in the module comment above.
const SHORT16_SIGNED: &str = concat!(
    "49492a00080000000b0000010400010000001000000001010400010000000c0000000201030001000000100000000301",
    "030001000000010000000601030001000000010000001101040001000000920000001501030001000000010000001601",
    "0400010000000c0000001701040001000000800100001c01030001000000010000005301030001000000020000000000",
    "0000e0b1efb2feb30db51cb62bb73ab849b958ba67bb76bc85bd94bea3bfb2c0c1c1d0c2dfc3eec4fdc50cc71bc82ac9",
    "39ca48cb57cc66cd75ce84cf93d0a2d1b1d2c0d3cfd4ded5edd6fcd70bd91ada29db38dc47dd56de65df74e083e192e2",
    "a1e3b0e4bfe5cee6dde7ece8fbe90aeb19ec28ed37ee46ef55f064f173f282f391f4a0f5aff6bef7cdf8dcf9ebfafafb",
    "09fd18fe27ff36004501540263037204810590069f07ae08bd09cc0adb0bea0cf90d080f171026113512441353146215",
    "711680178f189e19ad1abc1bcb1cda1de91ef81f072116222523342443255226612770287f298e2a9d2bac2cbb2dca2e",
    "d92fe830f73106331534243533364237513860396f3a7e3b8d3c9c3dab3eba3fc940d841e742f6430545144623473248",
    "4149504a5f4b6e4c7d4d8c4e9b4faa50b951c852d753e654f555045713582259315a405b4f5c5e5d6d5e7c5f8b609a61",
    "a962b863c764d665e566f4670369126a216b306c3f6d4e6e5d6f6c707b718a729973a874b775c676d577e478f379027b",
    "117c"
);

/// `t_uint16.tif`, produced as described in the module comment above.
const USHORT16_PLAIN: &str = concat!(
    "49492a00080000000b0000010400010000001000000001010400010000000c0000000201030001000000100000000301",
    "030001000000010000000601030001000000010000001101040001000000920000001501030001000000010000001601",
    "0400010000000c0000001701040001000000800100001c01030001000000010000005301030001000000010000000000",
    "0000e80339058a06db072c097d0ace0b1f0d700ec10f12116312b41305155616a717f818491a9a1beb1c3c1e8d1fde20",
    "2f228023d12422267327c428152a662bb72c082e592faa30fb314c339d34ee353f379038e139323b833cd43d253f7640",
    "c74118436944ba450b475c48ad49fe4a4f4ca04df14e42509351e45235548655d75628587959ca5a1b5c6c5dbd5e0e60",
    "5f61b06201645265a366f4674569966ae76b386d896eda6f2b717c72cd731e756f76c0771179627ab37b047d557ea67f",
    "f78048829983ea843b868c87dd882e8a7f8bd08c218e728fc39014926593b69407965897a998fa994b9b9c9ced9d3e9f",
    "8fa0e0a131a382a4d3a524a775a8c6a917ab68acb9ad0aaf5bb0acb1fdb24eb49fb5f0b641b892b9e3ba34bc85bdd6be",
    "27c078c1c9c21ac46bc5bcc60dc85ec9afca00cc51cda2cef3cf44d195d2e6d337d588d6d9d72ad97bdaccdb1ddd6ede",
    "bfdf10e161e2b2e303e554e6a5e7f6e847ea98ebe9ec3aee8befdcf02df27ef3cff420f671f7c2f813fa64fbb5fc06fe",
    "57ff"
);

/// Two uncompressed 8-pixel rows, packed MSB-first as 0..7 and 8..15.
/// This is a classic TIFF IFD made with the same baseline tag layout libtiff
/// accepts for its true 4-bit grayscale path.
const GRAY4_MINISBLACK: &str = concat!(
    "49492a00080000000900000104000100000008000000010104000100000002000000",
    "02010300010000000400000003010300010000000100000006010300010000000100",
    "000011010400010000007a0000001501030001000000010000001601040001000000",
    "02000000170104000100000008000000000000000123456789abcdef"
);

/// The same pixels as `GRAY4_MINISBLACK`, with TIFF FillOrder 2 and each
/// pair reversed in storage so the low nibble is the first sample.
const GRAY4_MINISBLACK_LSB: &str = concat!(
    "49492a00080000000a00000104000100000008000000010104000100000002000000",
    "02010300010000000400000003010300010000000100000006010300010000000100",
    "00000a01030001000000020000001101040001000000860000001501030001000000",
    "01000000160104000100000002000000170104000100000008000000000000001032",
    "547698badcfe"
);

/// `t_planar_rgb8.tif`, generated as an 8-bit RGB TIFF with Pillow 12.3 and
/// converted by libtiff 4.5.1 `tiffcp -p separate -s`.  It contains the same
/// 16x12 RGB sample formula documented above, but its three strip planes are
/// deliberately separate rather than TIFF's ordinary RGBRGB... layout.
const RGB8_PLANAR_SEPARATE: &str = concat!(
    "49492a0048020000000306090c0f1215181b1e2124272a2d303336393c3f4245484b4e5154575a5d606366696c6f7275",
    "787b7e8184878a8d909396999c9fa2a5a8abaeb1b4b7babdc0c3c6c9cccfd2d5d8dbdee1e4e7eaedf0f3f6f9fcff0205",
    "080b0e1114171a1d202326292c2f3235383b3e4144474a4d505356595c5f6265686b6e7174777a7d808386898c8f9295",
    "989b9ea1a4a7aaadb0b3b6b9bcbfc2c5c8cbced1d4d7dadde0e3e6e9eceff2f5f8fbfe0104070a0d101316191c1f2225",
    "282b2e3134373a3d00050a0f14191e23282d32373c41464b50555a5f64696e73787d82878c91969ba0a5aaafb4b9bec3",
    "c8cdd2d7dce1e6ebf0f5faff04090e13181d22272c31363b40454a4f54595e63686d72777c81868b90959a9fa4a9aeb3",
    "b8bdc2c7ccd1d6dbe0e5eaeff4f9fe03080d12171c21262b30353a3f44494e53585d62676c71767b80858a8f94999ea3",
    "a8adb2b7bcc1c6cbd0d5dadfe4e9eef3f8fd02070c11161b20252a2f34393e43484d52575c61666b70757a7f84898e93",
    "989da2a7acb1b6bb00070e151c232a31383f464d545b626970777e858c939aa1a8afb6bdc4cbd2d9e0e7eef5fc030a11",
    "181f262d343b424950575e656c737a81888f969da4abb2b9c0c7ced5dce3eaf1f8ff060d141b222930373e454c535a61",
    "686f767d848b9299a0a7aeb5bcc3cad1d8dfe6edf4fb020910171e252c333a41484f565d646b727980878e959ca3aab1",
    "b8bfc6cdd4dbe2e9f0f7fe050c131a21282f363d444b525960676e757c838a91989fa6adb4bbc2c9d0d7dee5ecf3fa01",
    "080f161d242b32390b0000010300010000001000000001010300010000000c0000000201030003000000d20200000301",
    "030001000000010000000601030001000000020000001101040003000000de0200001201030001000000010000001501",
    "0300010000000300000016010300010000000c0000001701030003000000d80200001c01030001000000020000000000",
    "0000080008000800c000c000c00008000000c800000088010000"
);

/// `t_rgba8.tif`, produced as described in the module comment above.
const RGBA8: &str = concat!(
    "49492a00080000000b0000010400010000001000000001010400010000000c0000000201030004000000920000000301",
    "0300010000000100000006010300010000000200000011010400010000009a0000001501030001000000040000001601",
    "0400010000000c0000001701040001000000000300001c01030001000000010000005201030001000000020000000000",
    "00000800080008000800000000000305070b060a0e16090f15210c141c2c0f192337121e2a421523314d182838581b2d",
    "3f631e32466e21374d79243c548427415b8f2a46629a2d4b69a5305070b0335577bb365a7ec6395f85d13c648cdc3f69",
    "93e7426e9af24573a1fd4878a8084b7daf134e82b61e5187bd29548cc4345791cb3f5a96d24a5d9bd95560a0e06063a5",
    "e76b66aaee7669aff5816cb4fc8c6fb9039772be0aa275c311ad78c818b87bcd1fc37ed226ce81d72dd984dc34e487e1",
    "3bef8ae642fa8deb490590f0501093f5571b96fa5e2699ff65319c046c3c9f097347a20e7a52a513815da8188868ab1d",
    "8f73ae22967eb1279d89b42ca494b731ab9fba36b2aabd3bb9b5c040c0c0c345c7cbc64aced6c94fd5e1cc54dceccf59",
    "e3f7d25eea02d563f10dd868f818db6dff23de72062ee1770d39e47c1444e7811b4fea86225aed8b2965f0903070f395",
    "377bf69a3e86f99f4591fca44c9cffa953a702ae5ab205b361bd08b868c80bbd6fd30ec276de11c77de914cc84f417d1",
    "8bff1ad6920a1ddb991520e0a02023e5a72b26eaae3629efb5412cf4bc4c2ff9c35732feca623503d16d3808d8783b0d",
    "df833e12e68e4117ed99441cf4a44721fbaf4a2602ba4d2b09c5503010d0533517db563a1ee6593f25f15c442cfc5f49",
    "3307624e3a126553411d685848286b5d4f336e62563e71675d49746c645477716b5f7a76726a7d7b7975808080808385",
    "878b868a8e96898f95a18c949cac8f99a3b7929eaac295a3b1cd98a8b8d89badbfe39eb2c6eea1b7cdf9a4bcd404a7c1",
    "db0faac6e21aadcbe925b0d0f030b3d5f73bb6dafe46b9df0551bce40c5cbfe91367c2ee1a72c5f3217dc8f82888cbfd",
    "2f93ce02369ed1073da9d40c44b4d7114bbfda1652cadd1b59d5e02060e0e32567ebe62a6ef6e92f7501ec347c0cef39",
    "8317f23e8a22f543912df8489838fb4d9f43fe52a64e0157ad59045cb4640761bb6f0a66c27a0d6bc9851070d0901375",
    "d79b167adea6197fe5b11c84ecbc1f89f3c7228efad2259301dd289808e82b9d0ff32ea216fe31a71d0934ac241437b1",
    "2b1f3ab6322a3dbb3935"
);

/// `t_pal8.tif`, produced as described in the module comment above.
const PALETTE8: &str = concat!(
    "49492a00080000000a0000010400010000001000000001010400010000000c0000000201030001000000080000000301",
    "0300010000000100000006010300010000000300000011010400010000008606000016010400010000000c0000001701",
    "040001000000c00000001c01030001000000010000004001030000030000860000000000000000000007000e0015001c",
    "0023002a00310038003f0046004d0054005b0062006900700077007e0085008c0093009a00a100a800af00b600bd00c4",
    "00cb00d200d900e000e700ee00f500fc0003000a00110018001f0026002d0034003b0042004900500057005e0065006c",
    "0073007a00810088008f0096009d00a400ab00b200b900c000c700ce00d500dc00e300ea00f100f800ff0006000d0014",
    "001b0022002900300037003e0045004c0053005a00610068006f0076007d0084008b0092009900a000a700ae00b500bc",
    "00c300ca00d100d800df00e600ed00f400fb0002000900100017001e0025002c0033003a00410048004f0056005d0064",
    "006b0072007900800087008e0095009c00a300aa00b100b800bf00c600cd00d400db00e200e900f000f700fe0005000c",
    "0013001a00210028002f0036003d0044004b0052005900600067006e0075007c0083008a00910098009f00a600ad00b4",
    "00bb00c200c900d000d700de00e500ec00f300fa00010008000f0016001d0024002b0032003900400047004e0055005c",
    "0063006a00710078007f0086008d0094009b00a200a900b000b700be00c500cc00d300da00e100e800ef00f600fd0004",
    "000b0012001900200027002e0035003c0043004a00510058005f0066006d0074007b0082008900900097009e00a500ac",
    "00b300ba00c100c800cf00d600dd00e400eb00f200f90000000b00160021002c00370042004d00580063006e00790084",
    "008f009a00a500b000bb00c600d100dc00e700f200fd00080013001e00290034003f004a00550060006b00760081008c",
    "009700a200ad00b800c300ce00d900e400ef00fa00050010001b00260031003c00470052005d00680073007e00890094",
    "009f00aa00b500c000cb00d600e100ec00f70002000d00180023002e00390044004f005a00650070007b00860091009c",
    "00a700b200bd00c800d300de00e900f400ff000a00150020002b00360041004c00570062006d00780083008e009900a4",
    "00af00ba00c500d000db00e600f100fc00070012001d00280033003e00490054005f006a00750080008b009600a100ac",
    "00b700c200cd00d800e300ee00f90004000f001a00250030003b00460051005c00670072007d00880093009e00a900b4",
    "00bf00ca00d500e000eb00f60001000c00170022002d00380043004e00590064006f007a00850090009b00a600b100bc",
    "00c700d200dd00e800f300fe00090014001f002a00350040004b00560061006c00770082008d009800a300ae00b900c4",
    "00cf00da00e500f000fb00060011001c00270032003d00480053005e00690074007f008a009500a000ab00b600c100cc",
    "00d700e200ed00f80003000e00190024002f003a00450050005b00660071007c00870092009d00a800b300be00c900d4",
    "00df00ea00f50000000d001a002700340041004e005b006800750082008f009c00a900b600c300d000dd00ea00f70004",
    "0011001e002b003800450052005f006c00790086009300a000ad00ba00c700d400e100ee00fb000800150022002f003c",
    "0049005600630070007d008a009700a400b100be00cb00d800e500f200ff000c0019002600330040004d005a00670074",
    "0081008e009b00a800b500c200cf00dc00e900f600030010001d002a003700440051005e006b007800850092009f00ac",
    "00b900c600d300e000ed00fa000700140021002e003b004800550062006f007c0089009600a300b000bd00ca00d700e4",
    "00f100fe000b001800250032003f004c0059006600730080008d009a00a700b400c100ce00db00e800f50002000f001c",
    "0029003600430050005d006a007700840091009e00ab00b800c500d200df00ec00f9000600130020002d003a00470054",
    "0061006e007b0088009500a200af00bc00c900d600e300f000fd000a001700240031003e004b005800650072007f008c",
    "009900a600b300c000cd00da00e700f40001000e001b002800350042004f005c0069007600830090009d00aa00b700c4",
    "00d100de00eb00f800050012001f002c0039004600530060006d007a0087009400a100ae00bb00c800d500e200ef00fc",
    "0009001600230030003d004a005700640071007e008b009800a500b200bf00cc00d900e600f300010203040506070809",
    "0a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233343536373839",
    "3a3b3c3d3e3f000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223242526272829",
    "2a2b2c2d2e2f303132333435363738393a3b3c3d3e3f000102030405060708090a0b0c0d0e0f10111213141516171819",
    "1a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
);

/// `ntsc4.tif`: a hand-built 4x1 uncompressed contiguous RGB TIFF whose pixels
/// are (0,5,5), (0,15,15), (0,19,39) and (0,21,1) - four of the 30863 RGB
/// triples out of 2^24 where the source's sequential float accumulation and a
/// single f32 expression give different grayscale values.
const NTSC_RGB4: &str = concat!(
    "49492a00080000000a000001040001000000040000000101040001000000010000000201030003000000860000000301",
    "0300010000000100000006010300010000000200000011010400010000008c0000001501030001000000030000001601",
    "0400010000000100000017010400010000000c0000001c01030001000000010000000000000008000800080000050500",
    "0f0f001327001501"
);

/// `bilevel1.tif`: Pillow 12.3 `Image.convert("1").save(...)`, a 16x12 1-bit
/// image.  Pillow omits the BitsPerSample tag for bilevel data.
const BILEVEL1: &str = concat!(
    "49492a0008000000080000010400010000001000000001010400010000000c0000000301030001000000010000000601",
    "0300010000000100000011010400010000006e00000016010400010000000c0000001701040001000000180000001c01",
    "0300010000000100000000000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
);
