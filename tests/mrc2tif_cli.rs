mod common;

use imod_rs::imod::libiimod::iimage::{ii_delete, ii_new};
use imod_rs::imod::libiimod::iitif::{IICOMPRESSION_ZIP, ii_tiff_check};
use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_SHORT, MrcHeader, mrc_get_scale, mrc_head_new, mrc_head_read,
    mrc_head_write,
};
use imod_rs::imod::mrc::tiff::tiff_ifd_number;
use std::process::Command;

/// Backend selection is evaluated before PIP option parsing.  In particular,
/// a typo must not turn into an ordinary usage error or silently run the Qt
/// encoder, because that would make an experimental invocation impossible to
/// audit.
#[test]
fn mrc2tif_rejects_an_unknown_native_encoder_before_argument_parsing() {
    let result = common::imod_cmd("mrc2tif")
        .env("IMOD_RS_MRC2TIF_ENCODER", "not-a-backend")
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains(
            "ERROR: Rust-native backend - IMOD_RS_MRC2TIF_ENCODER must be parity or rust"
        ),
        "stdout was: {stdout}"
    );
}

#[cfg(feature = "rust-image-encoder")]
#[test]
fn mrc2tif_rust_png_encoder_writes_a_decodable_image() {
    unsafe {
        let stamp = format!("imod-rs-rust-png-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.png"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
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
            4
        );
        drop(file);

        let result = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_MRC2TIF_ENCODER", "rust")
            .arg("-p")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let decoded = image::open(&output).unwrap().to_luma8();
        assert_eq!(decoded.dimensions(), (2, 2));
        // mrcReadZ applies IMOD's signed-byte map before mrc2tif reverses rows.
        assert_eq!(decoded.into_raw(), vec![131, 132, 129, 130]);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

/// The source QImage block in `mrc2tif.cpp:567-596` selects `Format_RGB888`,
/// flips MRC rows, and gives its padded rows to the PNG writer.  Exercise the
/// same command boundary through both the retained Qt implementation and the
/// explicitly selected Rust implementation.  PNG is lossless, so decoded
/// RGB samples must agree exactly; ImageMagick's independent PNG decoder also
/// checks the emitted container's colour model and dimensions.
#[cfg(feature = "rust-image-encoder")]
#[test]
fn mrc2tif_rust_png_encoder_matches_qimage_for_rgb_and_row_orientation() {
    unsafe {
        let stamp = format!("imod-rs-rust-png-rgb-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let parity_output = std::env::temp_dir().join(format!("{stamp}-parity.png"));
        let rust_output = std::env::temp_dir().join(format!("{stamp}-rust.png"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(
            mrc_head_new(
                &mut header,
                2,
                2,
                1,
                imod_rs::imod::libiimod::mrcfiles::MRC_MODE_RGB
            ),
            0
        );
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        // MRC row 0 is the bottom row.  mrc2tif.cpp:577-580 flips it for
        // the top-down PNG coordinate system.
        let pixels = [
            10_u8, 20, 30, 40, 50, 60, // MRC bottom row
            70, 80, 90, 100, 110, 120, // MRC top row
        ];
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

        let parity = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_MRC2TIF_ENCODER", "parity")
            .arg("-p")
            .arg(&input)
            .arg(&parity_output)
            .output()
            .unwrap();
        assert!(
            parity.status.success(),
            "{}",
            String::from_utf8_lossy(&parity.stderr)
        );
        let rust = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_MRC2TIF_ENCODER", "rust")
            .arg("-p")
            .arg(&input)
            .arg(&rust_output)
            .output()
            .unwrap();
        assert!(
            rust.status.success(),
            "{}",
            String::from_utf8_lossy(&rust.stderr)
        );

        let expected = vec![70, 80, 90, 100, 110, 120, 10, 20, 30, 40, 50, 60];
        let parity_decoded = image::open(&parity_output).unwrap().to_rgb8();
        let rust_decoded = image::open(&rust_output).unwrap().to_rgb8();
        assert_eq!(parity_decoded.dimensions(), (2, 2));
        assert_eq!(rust_decoded.dimensions(), (2, 2));
        assert_eq!(parity_decoded.into_raw(), expected);
        assert_eq!(rust_decoded.into_raw(), expected);

        for output in [&parity_output, &rust_output] {
            let identify = Command::new("identify")
                .args(["-format", "%m %w %h %[channels]"])
                .arg(output)
                .output()
                .unwrap();
            assert!(
                identify.status.success(),
                "ImageMagick could not decode {}: {}",
                output.display(),
                String::from_utf8_lossy(&identify.stderr)
            );
            assert_eq!(String::from_utf8(identify.stdout).unwrap(), "PNG 2 2 srgb");
        }
        for path in [input, parity_output, rust_output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

/// The Qt boundary receives a signed `resolution`: `-r` is DPI (negative
/// internally), while `-P`/`-m` pass positive pixels/cm.  Check both actual
/// output formats through ImageMagick rather than by inspecting encoder
/// internals.  This also keeps the experimental backend tied to the retained
/// QImage result while it remains the default.
#[cfg(feature = "rust-image-encoder")]
#[test]
fn mrc2tif_rust_jpeg_and_png_preserve_qimage_resolution_units() {
    unsafe {
        let stamp = format!("imod-rs-rust-image-resolution-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_BYTE), 0);
        // Two five-million-Angstrom pixels: -P derives 20 pixels/cm, a
        // comfortably representable JFIF density that exercises the positive
        // source resolution convention.
        header.xlen = 10_000_000.0;
        header.ylen = 5_000_000.0;
        header.zlen = 5_000_000.0;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(&[10_u8, 20], 1, 2, &mut file),
            2
        );
        drop(file);

        for (format_option, extension, expected_dpi_unit, expected_dpi_density) in [
            ("-j", "jpg", "PixelsPerInch", 300.0),
            ("-p", "png", "PixelsPerCentimeter", 118.11),
        ] {
            for (resolution_option, suffix, expected_unit, expected_density) in [
                (
                    vec!["-r", "300"],
                    "dpi",
                    expected_dpi_unit,
                    expected_dpi_density,
                ),
                (vec!["-P"], "pixels-cm", "PixelsPerCentimeter", 20.0),
            ] {
                let parity = std::env::temp_dir()
                    .join(format!("{stamp}-{extension}-{suffix}-parity.{extension}"));
                let rust = std::env::temp_dir()
                    .join(format!("{stamp}-{extension}-{suffix}-rust.{extension}"));
                for (backend, output) in [("parity", &parity), ("rust", &rust)] {
                    let result = common::imod_cmd("mrc2tif")
                        .env("IMOD_RS_MRC2TIF_ENCODER", backend)
                        .arg(format_option)
                        .args(&resolution_option)
                        .arg(&input)
                        .arg(output)
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{format_option}/{suffix}/{backend}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
                for output in [&parity, &rust] {
                    let identify = Command::new("identify")
                        .args(["-format", "%x|%y|%U"])
                        .arg(output)
                        .output()
                        .unwrap();
                    assert!(
                        identify.status.success(),
                        "ImageMagick could not read resolution metadata from {}: {}",
                        output.display(),
                        String::from_utf8_lossy(&identify.stderr)
                    );
                    let metadata = String::from_utf8(identify.stdout).unwrap();
                    let mut fields = metadata.split('|');
                    let x = fields.next().unwrap().parse::<f64>().unwrap();
                    let y = fields.next().unwrap().parse::<f64>().unwrap();
                    let unit = fields.next().unwrap();
                    assert_eq!(unit, expected_unit, "{}", output.display());
                    assert!((x - expected_density).abs() < 0.02, "{metadata}");
                    assert!((y - expected_density).abs() < 0.02, "{metadata}");
                }
                std::fs::remove_file(parity).unwrap();
                std::fs::remove_file(rust).unwrap();
            }
        }
        std::fs::remove_file(input).unwrap();
    }
}

/// JPEG is intentionally compared after independent decoding, not by file
/// bytes: Qt and the Rust `image` encoder are allowed to choose different
/// legal JPEG marker/order encodings.  This is a command-boundary test of two
/// valid MRC inputs (gray and RGB), including the source-owned conversion and
/// MRC bottom-up to image top-down row reversal before the encoder boundary.
///
/// The bounds below are a measured cross-decoder tolerance at quality 95,
/// rather than a claim that either JPEG is lossless.  Keeping separate max and
/// mean-error limits makes a future gross scaling/channel-order regression
/// visible even if an individual quantized sample happens to be near a bin
/// boundary.
#[cfg(feature = "rust-image-encoder")]
#[test]
fn mrc2tif_rust_jpeg_encoder_matches_qimage_after_decoding_gray_and_rgb() {
    unsafe {
        let stamp = format!("imod-rs-rust-jpeg-differential-{}", std::process::id());
        for (kind, mode, channels) in [
            ("gray", MRC_MODE_BYTE, 1_usize),
            (
                "rgb",
                imod_rs::imod::libiimod::mrcfiles::MRC_MODE_RGB,
                3_usize,
            ),
        ] {
            // An odd width exercises QImage RGB888 row padding.  Several
            // changing directions avoid a test that could pass with swapped
            // rows or channels merely because its fixture is symmetric.
            let width = 17_i32;
            let height = 13_i32;
            let input = std::env::temp_dir().join(format!("{stamp}-{kind}.mrc"));
            let parity_output = std::env::temp_dir().join(format!("{stamp}-{kind}-parity.jpg"));
            let rust_output = std::env::temp_dir().join(format!("{stamp}-{kind}-rust.jpg"));
            let input_c = input.to_str().unwrap();
            let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, width, height, 1, mode), 0);
            header.amin = 0.0;
            header.amax = 255.0;
            header.amean = 127.5;
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut pixels = Vec::with_capacity(width as usize * height as usize * channels);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    // Deliberately smooth, microscopy-like ramps.  JPEG's
                    // lossy differential should measure encoder variation,
                    // not dominate the assertion with a checkerboard's
                    // intentionally unrecoverable high-frequency energy.
                    let base = (x * 8 + y * 5) as u8;
                    if channels == 1 {
                        pixels.push(base);
                    } else {
                        pixels.extend_from_slice(&[
                            base,
                            (x * 4 + y * 11) as u8,
                            (x * 9 + y * 3) as u8,
                        ]);
                    }
                }
            }
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

            for (backend, output) in [("parity", &parity_output), ("rust", &rust_output)] {
                let result = common::imod_cmd("mrc2tif")
                    .env("IMOD_RS_MRC2TIF_ENCODER", backend)
                    .args(["-j", "-q", "95"])
                    .arg(&input)
                    .arg(output)
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{kind}/{backend}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }

            let parity_decoded = if channels == 1 {
                image::open(&parity_output).unwrap().to_luma8().into_raw()
            } else {
                image::open(&parity_output).unwrap().to_rgb8().into_raw()
            };
            let rust_decoded = if channels == 1 {
                image::open(&rust_output).unwrap().to_luma8().into_raw()
            } else {
                image::open(&rust_output).unwrap().to_rgb8().into_raw()
            };
            assert_eq!(parity_decoded.len(), rust_decoded.len());
            let differences: Vec<u8> = parity_decoded
                .iter()
                .zip(&rust_decoded)
                .map(|(&left, &right)| left.abs_diff(right))
                .collect();
            let maximum = *differences.iter().max().unwrap();
            let mean = differences.iter().map(|&value| value as f64).sum::<f64>()
                / differences.len() as f64;
            assert!(
                maximum <= 6 && mean <= 1.25,
                "{kind} quality-95 JPEG decoder differential: max={maximum}, mean={mean}"
            );

            for output in [&parity_output, &rust_output] {
                let identify = Command::new("identify")
                    .args(["-format", "%m %w %h %[channels] %Q"])
                    .arg(output)
                    .output()
                    .unwrap();
                assert!(
                    identify.status.success(),
                    "ImageMagick could not decode {}: {}",
                    output.display(),
                    String::from_utf8_lossy(&identify.stderr)
                );
                let expected_channels = if channels == 1 { "gray" } else { "srgb" };
                assert_eq!(
                    String::from_utf8(identify.stdout).unwrap(),
                    format!("JPEG 17 13 {expected_channels} 95")
                );
            }
            for path in [input, parity_output, rust_output] {
                std::fs::remove_file(path).unwrap();
            }
        }
    }
}

#[cfg(feature = "rust-tiff")]
#[test]
fn mrc2tif_rust_tiff_writer_roundtrips_a_basic_mrc_image() {
    unsafe {
        let stamp = format!("imod-rs-rust-tiff-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
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
            4
        );
        drop(file);

        let result = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_TIFF_BACKEND", "rust")
            .arg(&input)
            .arg(&tiff)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let result = common::imod_cmd("tif2mrc")
            // The new Rust TIFF writer must produce a standard TIFF file
            // that the default IMOD/libtiff reader accepts; using the Rust
            // reader here would only test one experimental backend against
            // itself.
            .env_remove("IMOD_RS_TIFF_BACKEND")
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                1024 as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut written = [0_u8; 4];
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
            4
        );
        drop(file);
        assert_eq!(written, pixels);
        for path in [input, tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(feature = "rust-tiff")]
#[test]
fn mrc2tif_rust_tiff_writer_preserves_imod_resolution_units() {
    use tiff::decoder::Decoder;
    use tiff::tags::Tag;

    unsafe {
        let stamp = format!("imod-rs-rust-tiff-resolution-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let inches_tiff = std::env::temp_dir().join(format!("{stamp}-inches.tif"));
        let centimeters_tiff = std::env::temp_dir().join(format!("{stamp}-centimeters.tif"));
        let inches_mrc = std::env::temp_dir().join(format!("{stamp}-inches.mrc"));
        let centimeters_mrc = std::env::temp_dir().join(format!("{stamp}-centimeters.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_BYTE), 0);
        // Two samples across ten Angstroms: -P must write 20,000,000 pixels/cm.
        header.xlen = 10.0;
        header.ylen = 5.0;
        header.zlen = 5.0;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(&[11_u8, 22], 1, 2, &mut file),
            2
        );
        drop(file);

        let result = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_TIFF_BACKEND", "rust")
            .args(["-r", "300"])
            .arg(&input)
            .arg(&inches_tiff)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut decoder = Decoder::new(std::fs::File::open(&inches_tiff).unwrap()).unwrap();
        assert_eq!(
            decoder
                .get_tag(Tag::ResolutionUnit)
                .unwrap()
                .into_u16()
                .unwrap(),
            2
        );
        assert_eq!(decoder.get_tag_u32_vec(Tag::XResolution).unwrap(), [300, 1]);
        assert_eq!(decoder.get_tag_u32_vec(Tag::YResolution).unwrap(), [300, 1]);
        let result = common::imod_cmd("tif2mrc")
            .env_remove("IMOD_RS_TIFF_BACKEND")
            .arg("-P")
            .arg(&inches_tiff)
            .arg(&inches_mrc)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let inches_c = inches_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(inches_c, "rb").unwrap();
        let mut read_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut read_header), 0);
        drop(file);
        assert!((mrc_get_scale(&read_header).0 - 2.54e8 / 300.0).abs() < 1.0);

        let result = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_TIFF_BACKEND", "rust")
            .arg("-P")
            .arg(&input)
            .arg(&centimeters_tiff)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut decoder = Decoder::new(std::fs::File::open(&centimeters_tiff).unwrap()).unwrap();
        assert_eq!(
            decoder
                .get_tag(Tag::ResolutionUnit)
                .unwrap()
                .into_u16()
                .unwrap(),
            3
        );
        assert_eq!(
            decoder.get_tag_u32_vec(Tag::XResolution).unwrap(),
            [20_000_000, 1]
        );
        assert_eq!(
            decoder.get_tag_u32_vec(Tag::YResolution).unwrap(),
            [20_000_000, 1]
        );
        let result = common::imod_cmd("tif2mrc")
            .env_remove("IMOD_RS_TIFF_BACKEND")
            .arg("-P")
            .arg(&centimeters_tiff)
            .arg(&centimeters_mrc)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let centimeters_c = centimeters_mrc.to_str().unwrap();
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(centimeters_c, "rb").unwrap();
        assert_eq!(mrc_head_read(&mut file, &mut read_header), 0);
        drop(file);
        assert!((mrc_get_scale(&read_header).0 - 5.0).abs() < 0.001);

        for path in [
            input,
            inches_tiff,
            centimeters_tiff,
            inches_mrc,
            centimeters_mrc,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(feature = "rust-tiff")]
#[test]
fn mrc2tif_rust_tiff_writer_roundtrips_lzw_and_zip_images() {
    unsafe {
        let stamp = format!("imod-rs-rust-tiff-compression-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
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
            4
        );
        drop(file);

        for compression in ["lzw", "zip"] {
            let tiff = std::env::temp_dir().join(format!("{stamp}-{compression}.tif"));
            let output = std::env::temp_dir().join(format!("{stamp}-{compression}.mrc"));
            let output_c = output.to_str().unwrap();
            let result = common::imod_cmd("mrc2tif")
                .env("IMOD_RS_TIFF_BACKEND", "rust")
                .args(["-c", compression])
                .arg(&input)
                .arg(&tiff)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{compression}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let result = common::imod_cmd("tif2mrc")
                .env_remove("IMOD_RS_TIFF_BACKEND")
                .arg(&tiff)
                .arg(&output)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{compression}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let mut file =
                imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
            assert_eq!(
                imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    1024 as i32,
                    imod_rs::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut written = [0_u8; 4];
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
                4
            );
            drop(file);
            assert_eq!(written, pixels, "{compression}");
            std::fs::remove_file(tiff).unwrap();
            std::fs::remove_file(output).unwrap();
        }
        std::fs::remove_file(input).unwrap();
    }
}

#[cfg(feature = "rust-tiff")]
#[test]
fn mrc2tif_rust_tiff_writer_roundtrips_a_two_page_stack() {
    unsafe {
        let stamp = format!("imod-rs-rust-tiff-stack-{}", std::process::id());
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
            8
        );
        drop(file);
        let result = common::imod_cmd("mrc2tif")
            .env("IMOD_RS_TIFF_BACKEND", "rust")
            .arg("-s")
            .arg(&input)
            .arg(&tiff)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let result = common::imod_cmd("tif2mrc")
            .env_remove("IMOD_RS_TIFF_BACKEND")
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut out_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut out_header), 0);
        assert_eq!((out_header.nx, out_header.ny, out_header.nz), (2, 2, 2));
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                out_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut written = [0_u8; 8];
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
            8
        );
        drop(file);
        assert_eq!(written, pixels);
        for path in [input, tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_rejects_source_illegal_compression_before_opening_input() {
    let result = common::imod_cmd("mrc2tif")
        .arg("-c")
        .arg("42")
        .arg("not-opened.mrc")
        .arg("not-written.tif")
        .output()
        .unwrap();
    assert!(!result.status.success());
    // `mrc2tif.cpp:116` installs `"\nERROR: %s - "` as the exit prefix, and
    // `PipSetError` (`parse_params.c:2049`) writes it plus one more space to
    // **stdout**, not stderr.  These assertions had encoded the wrong stream.
    assert!(String::from_utf8_lossy(&result.stdout).contains("Compression value 42 not allowed"));
}

/// `mrc2tif.cpp` accepts TIFF JPEG compression (`-c jpeg`), but the selected
/// Rust `tiff` encoder has no JPEG write implementation.  The opt-in backend
/// must fail before attempting input I/O, rather than silently use libtiff or
/// write a mismarked TIFF.
#[cfg(feature = "rust-tiff")]
#[test]
fn mrc2tif_rust_tiff_writer_rejects_jpeg_compression_before_opening_input() {
    let result = common::imod_cmd("mrc2tif")
        .env("IMOD_RS_TIFF_BACKEND", "rust")
        .args(["-c", "jpeg", "not-opened.mrc", "not-written.tif"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: mrc2tif -  Rust TIFF writer does not support JPEG compression; use the parity backend\n"
    );
}

#[test]
fn mrc2tif_uses_source_validation_diagnostic_before_opening_input() {
    let result = common::imod_cmd("mrc2tif")
        .args(["-o", "-P", "not-opened.mrc", "not-written.tif"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: mrc2tif -  Resolution setting is not available with old writing code\n"
    );
}

#[test]
fn mrc2tif_malformed_option_and_qimage_request_reaches_source_input_opening() {
    let malformed = common::imod_cmd("mrc2tif")
        .args(["-r", "not-a-number", "not-opened.mrc", "not-written.tif"])
        .output()
        .unwrap();
    assert!(!malformed.status.success());
    let qimage = common::imod_cmd("mrc2tif")
        .args(["-j", "not-opened.mrc", "not-written.jpg"])
        .output()
        .unwrap();
    assert!(!qimage.status.success());
    #[cfg(feature = "qt")]
    assert!(String::from_utf8_lossy(&qimage.stdout).contains("Couldn't open not-opened.mrc"));
    #[cfg(not(feature = "qt"))]
    assert!(String::from_utf8_lossy(&qimage.stdout).contains("QImage Qt boundary"));
}

#[test]
fn mrc2tif_rejects_source_out_of_range_z_after_reading_header() {
    unsafe {
        let input =
            std::env::temp_dir().join(format!("imod-rs-mrc2tif-z-{}.mrc", std::process::id()));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 1, 1, 1, MRC_MODE_BYTE), 0);
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        drop(file);
        let result = common::imod_cmd("mrc2tif")
            .args(["-z", "1:1"])
            .arg(&input)
            .arg(input.with_extension("tif"))
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains("zmin,zmax values are reversed or out of the range 0 to 0")
        );
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn mrc2tif_chunked_tiff_roundtrips_each_source_y_range() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-chunks-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let roundtrip = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let roundtrip_c = roundtrip.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 4, 1, MRC_MODE_BYTE), 0);
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
        let result = common::imod_cmd("mrc2tif")
            .arg("-T")
            .arg("2:1")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).starts_with("Writing TIFF images. .\r\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("Actual tile size = 16 x 16\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let result = common::imod_cmd("tif2mrc")
            .arg(&output)
            .arg(&roundtrip)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(roundtrip_c, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 4, 1, MRC_MODE_BYTE)
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut written = [0_u8; 8];
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
        assert_eq!(written, pixels);
        for path in [input, output, roundtrip] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_contrast_scales_a_real_short_mrc_before_tiff_writing() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-contrast-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_SHORT), 0);
        header.amin = 0.0;
        header.amax = 300.0;
        header.amean = 150.0;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [0_i16, 100, 200, 300];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(
                        pixels.as_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-C", "0:255"])
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut written_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut written_header), 0);
        assert_eq!(written_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                written_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut written = [0_u8; 4];
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
        // `tif2mrc` stores byte-mode MRC data with the source signed-byte
        // offset, so these are the on-disk forms of TIFF values 0,85,170,255.
        assert_eq!(written, [128, 213, 42, 127]);
        for path in [input, tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_old_writer_converts_real_mrc_pixels_to_classic_tiff() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
        header.amin = 1.;
        header.amax = 4.;
        header.amean = 2.5;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [1u8, 2, 3, 4];
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
        let result = common::imod_cmd("mrc2tif")
            .arg("-o")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(&output).unwrap();
        assert_eq!(&bytes[..4], b"II*\0");
        // `mrcHeadNew` marks byte data as signed; mrcReadZ applies its source
        // byte map before the legacy writer flips the two scan lines.
        assert_eq!(&bytes[8..12], &[131, 132, 129, 130]);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn mrc2tif_new_libtiff_writer_uses_local_datetime_tag() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-datetime-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 1, 1, 1, MRC_MODE_BYTE), 0);
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(&[7_u8], 1, 1, &mut file),
            1
        );
        drop(file);
        let result = common::imod_cmd("mrc2tif")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(result.status.success(), "{:?}", result);
        let expected = chrono::Local::now().format("%Y:%m:").to_string();
        let bytes = std::fs::read(&output).unwrap();
        let datetime = bytes
            .windows(19)
            .find(|value| value.starts_with(expected.as_bytes()))
            .expect("TIFF DateTime tag must use the local calendar date");
        assert_eq!(datetime[4], b':');
        assert_eq!(datetime[7], b':');
        assert_eq!(datetime[10], b' ');
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn mrc2tif_writes_explicit_and_header_pixel_spacing_as_real_tiff_resolution() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-resolution-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let from_header = std::env::temp_dir().join(format!("{stamp}-header.tif"));
        let from_inches = std::env::temp_dir().join(format!("{stamp}-inches.tif"));
        let from_zero = std::env::temp_dir().join(format!("{stamp}-zero.tif"));
        let header_mrc = std::env::temp_dir().join(format!("{stamp}-header.mrc"));
        let inches_mrc = std::env::temp_dir().join(format!("{stamp}-inches.mrc"));
        let zero_mrc = std::env::temp_dir().join(format!("{stamp}-zero.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_BYTE), 0);
        header.xlen = 10.0;
        header.ylen = 5.0;
        header.zlen = 5.0;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(&[11_u8, 22], 1, 2, &mut file),
            2
        );
        drop(file);

        assert!(
            common::imod_cmd("mrc2tif")
                .arg("-P")
                .arg(&input)
                .arg(&from_header)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-P")
                .arg(&from_header)
                .arg(&header_mrc)
                .status()
                .unwrap()
                .success()
        );
        let header_file = header_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(header_file, "rb").unwrap();
        let mut read_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut read_header), 0);
        drop(file);
        assert!((mrc_get_scale(&read_header).0 - 5.0).abs() < 0.001);

        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-r", "300"])
                .arg(&input)
                .arg(&from_inches)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-P")
                .arg(&from_inches)
                .arg(&inches_mrc)
                .status()
                .unwrap()
                .success()
        );
        let inches_file = inches_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(inches_file, "rb").unwrap();
        assert_eq!(mrc_head_read(&mut file, &mut read_header), 0);
        drop(file);
        assert!((mrc_get_scale(&read_header).0 - 2.54e8 / 300.0).abs() < 1.0);
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-r", "not-a-number"])
                .arg(&input)
                .arg(&from_zero)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-P")
                .arg(&from_zero)
                .arg(&zero_mrc)
                .status()
                .unwrap()
                .success()
        );
        let zero_file = zero_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(zero_file, "rb").unwrap();
        assert_eq!(mrc_head_read(&mut file, &mut read_header), 0);
        drop(file);
        assert!((mrc_get_scale(&read_header).0 - 1.0).abs() < 0.001);
        for path in [
            input,
            from_header,
            from_inches,
            from_zero,
            header_mrc,
            inches_mrc,
            zero_mrc,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_uses_each_mdoc_pixel_spacing_for_numbered_tiff_output() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-mdoc-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let mdoc = std::path::PathBuf::from(format!("{}.mdoc", input.display()));
        let root = std::env::temp_dir().join(format!("{stamp}-out"));
        let first_mrc = std::env::temp_dir().join(format!("{stamp}-first.mrc"));
        let second_mrc = std::env::temp_dir().join(format!("{stamp}-second.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 1, 1, 2, MRC_MODE_BYTE), 0);
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(&[17_u8, 34], 1, 2, &mut file),
            2
        );
        drop(file);
        std::fs::write(
            &mdoc,
            "[ZValue = 0]\nPixelSpacing = 3.0\n\n[ZValue = 1]\nPixelSpacing = 7.0\n",
        )
        .unwrap();
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-m", "-i", "42"])
                .arg(&input)
                .arg(&root)
                .status()
                .unwrap()
                .success()
        );
        let first_tif = std::path::PathBuf::from(format!("{}.042.tif", root.display()));
        let second_tif = std::path::PathBuf::from(format!("{}.043.tif", root.display()));
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-P")
                .arg(&first_tif)
                .arg(&first_mrc)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg("-P")
                .arg(&second_tif)
                .arg(&second_mrc)
                .status()
                .unwrap()
                .success()
        );
        let first_file = first_mrc.to_str().unwrap();
        let second_file = second_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(first_file, "rb").unwrap();
        let mut first_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut first_header), 0);
        drop(file);
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(second_file, "rb").unwrap();
        let mut second_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut second_header), 0);
        drop(file);
        assert!((mrc_get_scale(&first_header).0 - 3.0).abs() < 0.001);
        assert!((mrc_get_scale(&second_header).0 - 7.0).abs() < 0.001);
        for path in [input, mdoc, first_tif, second_tif, first_mrc, second_mrc] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_zip_stack_quality_and_slice_scaling_use_source_libtiff_paths() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-stack-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let stack = std::env::temp_dir().join(format!("{stamp}.tif"));
        let stack_mrc = std::env::temp_dir().join(format!("{stamp}-stack.mrc"));
        let scaled = std::env::temp_dir().join(format!("{stamp}-scaled"));
        let scaled_tif = std::path::PathBuf::from(format!("{}.000.tif", scaled.display()));
        let scaled_mrc = std::env::temp_dir().join(format!("{stamp}-scaled.mrc"));
        let auto = std::env::temp_dir().join(format!("{stamp}-auto"));
        let auto_tif = auto.clone();
        let auto_mrc = std::env::temp_dir().join(format!("{stamp}-auto.mrc"));
        let auto_input = std::env::temp_dir().join(format!("{stamp}-auto-input.mrc"));
        let numeric = std::env::temp_dir().join(format!("{stamp}-numeric.tif"));
        let numeric_mrc = std::env::temp_dir().join(format!("{stamp}-numeric.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_SHORT), 0);
        header.amin = 0.0;
        header.amax = 300.0;
        header.amean = 150.0;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [0_i16, 100, 200, 300, 300, 200, 100, 0];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(
                        pixels.as_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);

        // `zip` must select source IICOMPRESSION_ZIP (8), which in turn
        // accepts the 1..9 quality setting in `tiffWriteSetup`.
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-s", "-c", "zip", "-q", "9"])
                .arg(&input)
                .arg(&stack)
                .status()
                .unwrap()
                .success()
        );
        // The direct libtiff writer records both source sections as IFDs.
        let stack_c = stack.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(stack_c, "rb").unwrap();
        assert_eq!(tiff_ifd_number(&mut file), 2);
        drop(file);
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = Some(stack_c.to_owned());
        (*reader).fmode = "rb".to_owned();
        (*reader).fp = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(stack_c, "rb");
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, IICOMPRESSION_ZIP);
        assert_eq!((*reader).nz, 2);
        ii_delete(reader);
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&stack)
                .arg(&stack_mrc)
                .status()
                .unwrap()
                .success()
        );
        let stack_mrc_c = stack_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(stack_mrc_c, "rb").unwrap();
        let mut stack_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut stack_header), 0);
        assert_eq!((stack_header.nz, stack_header.mode), (2, MRC_MODE_SHORT));
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                stack_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut stack_pixels = [0_i16; 8];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        stack_pixels.as_mut_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * stack_pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                stack_pixels.len(),
                &mut file,
            ),
            stack_pixels.len()
        );
        drop(file);
        assert_eq!(stack_pixels, pixels);

        // `-S`, unlike `-C`, gives explicit data min/max limits before the
        // source linear byte conversion.  Values below/above 100..200 clamp.
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-z", "0:0", "-S", "100:200"])
                .arg(&input)
                .arg(&scaled)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&scaled_tif)
                .arg(&scaled_mrc)
                .status()
                .unwrap()
                .success()
        );
        let scaled_c = scaled_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(scaled_c, "rb").unwrap();
        let mut scaled_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut scaled_header), 0);
        assert_eq!(scaled_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                scaled_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut scaled_pixels = [0_u8; 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(&mut scaled_pixels, 1, 4, &mut file),
            scaled_pixels.len()
        );
        drop(file);
        // MRC signed-byte storage is the TIFF [0, 0, 255, 255] result shifted by 128.
        assert_eq!(scaled_pixels, [128, 128, 127, 127]);

        // Auto-contrast requires the source sampling minimum of five pixels.
        let auto_input_c = auto_input.to_str().unwrap();
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(auto_input_c, "wb").unwrap();
        let mut auto_input_header = MrcHeader::default();
        assert_eq!(
            mrc_head_new(&mut auto_input_header, 3, 2, 1, MRC_MODE_SHORT),
            0
        );
        auto_input_header.amin = 0.0;
        auto_input_header.amax = 500.0;
        auto_input_header.amean = 250.0;
        auto_input_header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut auto_input_header), 0);
        let auto_input_pixels = [0_i16, 100, 200, 300, 400, 500];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(
                        auto_input_pixels.as_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * auto_input_pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                auto_input_pixels.len(),
                &mut file,
            ),
            auto_input_pixels.len()
        );
        drop(file);
        // Auto-contrast computes the source sample mean/SD per section and
        // converts the real slice to byte values before libtiff writing.
        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-z", "0:0", "-a", "200:10"])
                .arg(&auto_input)
                .arg(&auto)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&auto_tif)
                .arg(&auto_mrc)
                .status()
                .unwrap()
                .success()
        );
        let auto_c = auto_mrc.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(auto_c, "rb").unwrap();
        let mut auto_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut auto_header), 0);
        assert_eq!(auto_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                auto_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut auto_pixels = [0_u8; 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        auto_pixels.as_mut_ptr().cast::<u8>(),
                        1 * (auto_pixels.len()),
                    )
                },
                1,
                auto_pixels.len(),
                &mut file,
            ),
            auto_pixels.len()
        );
        drop(file);
        assert_eq!(auto_pixels, [58, 63, 69, 74, 80, 85]);

        assert!(
            common::imod_cmd("mrc2tif")
                .args(["-O", "0", "-c", "32946"])
                .arg(&auto_input)
                .arg(&numeric)
                .status()
                .unwrap()
                .success()
        );
        let numeric_c = numeric.to_str().unwrap();
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = Some(numeric_c.to_owned());
        (*reader).fmode = "rb".to_owned();
        (*reader).fp = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(numeric_c, "rb");
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, 32946);
        ii_delete(reader);
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&numeric)
                .arg(&numeric_mrc)
                .status()
                .unwrap()
                .success()
        );
        let numeric_mrc_c = numeric_mrc.to_str().unwrap();
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(numeric_mrc_c, "rb").unwrap();
        let mut numeric_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut numeric_header), 0);
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                numeric_header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut numeric_pixels = [0_i16; 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        numeric_pixels.as_mut_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * numeric_pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                numeric_pixels.len(),
                &mut file,
            ),
            numeric_pixels.len()
        );
        drop(file);
        assert_eq!(numeric_pixels, auto_input_pixels);
        for path in [
            input,
            stack,
            stack_mrc,
            scaled_tif,
            scaled_mrc,
            auto_input,
            auto_tif,
            auto_mrc,
            numeric,
            numeric_mrc,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_jpeg_compression_uses_the_installed_libtiff_codec() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-jpeg-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let jpeg = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
        header.amin = 0.0;
        header.amax = 255.0;
        header.amean = 127.5;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [
            128_u8, 144, 160, 176, 192, 208, 224, 240, 129, 145, 161, 177, 193, 209, 225, 241,
        ];
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
                .args(["-O", "0", "-c", "jpeg", "-q", "100"])
                .arg(&input)
                .arg(&jpeg)
                .status()
                .unwrap()
                .success()
        );
        let jpeg_c = jpeg.to_str().unwrap();
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = Some(jpeg_c.to_owned());
        (*reader).fmode = "rb".to_owned();
        (*reader).fp = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(jpeg_c, "rb");
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, 7);
        ii_delete(reader);
        assert!(
            common::imod_cmd("tif2mrc")
                .arg(&jpeg)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let output_c = output.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(output_c, "rb").unwrap();
        let mut output_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut output_header), 0);
        drop(file);
        assert_eq!(
            (output_header.nx, output_header.ny, output_header.mode),
            (4, 4, MRC_MODE_BYTE)
        );
        for path in [input, jpeg, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

/// Reads every directory of the TIFF at [path] and returns whether it carries
/// an `ImageDescription` (tag 270) together with its `SMinSampleValue` (tag
/// 340) and `SMaxSampleValue` (tag 341).  libtiff gives the two sample-value
/// tags the field type of the image data, so a float image keeps them inline.
fn tiff_page_description_and_min_max(
    path: &std::path::Path,
) -> Vec<(bool, Option<f64>, Option<f64>)> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        &bytes[0..2],
        b"II",
        "the libtiff writer makes little-endian files"
    );
    let short_at =
        |offset: usize| u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
    let long_at =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let mut pages = Vec::new();
    let mut directory = long_at(4);
    while directory != 0 {
        let entries = short_at(directory) as usize;
        let mut described = false;
        let mut minimum = None;
        let mut maximum = None;
        for entry in 0..entries {
            let field = directory + 2 + entry * 12;
            match short_at(field) {
                270 => described = true,
                tag @ (340 | 341) => {
                    let field_type = short_at(field + 2);
                    let start = match field_type {
                        11 => field + 8,
                        12 => long_at(field + 8),
                        other => panic!("unexpected sample value field type {other}"),
                    };
                    let value = if field_type == 11 {
                        f64::from(f32::from_le_bytes(
                            bytes[start..start + 4].try_into().unwrap(),
                        ))
                    } else {
                        f64::from_le_bytes(bytes[start..start + 8].try_into().unwrap())
                    };
                    if tag == 340 {
                        minimum = Some(value);
                    } else {
                        maximum = Some(value);
                    }
                }
                _ => {}
            }
        }
        pages.push((described, minimum, maximum));
        directory = long_at(directory + 2 + entries * 12);
    }
    pages
}

/// `mrc2tif -s` sets `iifile->amin`/`amax` itself from this section's own
/// min and max before each `tiffWriteSection` -- `sliceMin`/`sliceMax` are
/// re-seeded at the top of the Z loop (`mrc2tif.cpp:475-476`) and fed to the
/// file at `mrc2tif.cpp:557-564` -- so every directory of the stack gets a
/// min/max pair of its own through `constrainAndStoreMinMax`
/// (`iitif.c:2591-2593`), and the last one is then overwritten by `tiffClose`
/// with the whole-stack range (`mrc2tif.cpp:637-638`, `iitif.c:696-700`).
/// Nothing in this program calls `tiffAddDescription`, so no directory carries
/// an `ImageDescription`.
#[test]
fn mrc2tif_stack_writes_running_min_max_on_every_directory_and_no_description() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-pages-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let input_c = input.to_str().unwrap();
        let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(input_c, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 3, 2, 3, 2), 0);
        header.nlabl = 1;
        header.labels[0][..80].copy_from_slice(
            b"imod-rs mrc2tif page fixture                                                    ",
        );
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels: Vec<f32> = (0..6)
            .map(|value| value as f32)
            .chain((0..6).map(|value| 10.0 + value as f32))
            .chain((0..6).map(|value| -3.0 + value as f32))
            .collect();
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), 4 * (pixels.len()))
                },
                4,
                pixels.len(),
                &mut file,
            ),
            18
        );
        drop(file);

        let result = common::imod_cmd("mrc2tif")
            .arg("-s")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let pages = tiff_page_description_and_min_max(&output);
        assert_eq!(pages.len(), 3, "one directory per section of the stack");
        for (index, page) in pages.iter().enumerate() {
            assert!(!page.0, "directory {index} must have no ImageDescription");
        }
        assert_eq!((pages[0].1, pages[0].2), (Some(0.0), Some(5.0)));
        assert_eq!((pages[1].1, pages[1].2), (Some(10.0), Some(15.0)));
        assert_eq!((pages[2].1, pages[2].2), (Some(-3.0), Some(15.0)));
        for path in [input, output] {
            let _ = std::fs::remove_file(path);
        }
    }
}
