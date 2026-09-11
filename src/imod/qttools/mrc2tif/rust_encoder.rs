//! Experimental Rust replacement for mrc2tif's Qt `QImage::save` boundary.
//!
//! This module owns only the encoder boundary.  The translated command still
//! owns section selection, scaling, row orientation, naming, and lifecycle.

/// Writes the already-oriented QImage-equivalent byte buffer with a Rust
/// JPEG/PNG encoder.
///
/// `data` has `height` rows padded to `bytes_per_line`; the padding is removed
/// before invoking the Rust encoders, which consume tightly packed rows.
#[cfg(feature = "rust-image-encoder")]
pub fn save(
    data: &[u8],
    width: i32,
    height: i32,
    bytes_per_line: i32,
    rgb: bool,
    resolution: i32,
    filename: &str,
    format: &str,
    quality: i32,
) -> Result<(), String> {
    use image::{ColorType, ImageEncoder};
    use std::fs::File;

    if width <= 0 || height <= 0 {
        return Err("Rust encoder requires positive image dimensions".into());
    }
    let channels = if rgb { 3usize } else { 1usize };
    let row_bytes = width as usize * channels;
    if bytes_per_line < row_bytes as i32 || data.len() < bytes_per_line as usize * height as usize {
        return Err("Rust encoder received invalid QImage row layout".into());
    }
    let mut pixels = vec![0_u8; row_bytes * height as usize];
    for row in 0..height as usize {
        let source =
            &data[row * bytes_per_line as usize..row * bytes_per_line as usize + row_bytes];
        pixels[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(source);
    }
    let color = if rgb { ColorType::Rgb8 } else { ColorType::L8 };
    let file = File::create(filename)
        .map_err(|error| format!("Rust encoder could not create {filename}: {error}"))?;
    match format {
        "JPEG" => {
            let jpeg_quality = if quality < 0 {
                75
            } else {
                quality.clamp(1, 100) as u8
            };
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(file, jpeg_quality);
            if resolution != 0 {
                let density = resolution.unsigned_abs();
                let Ok(density) = u16::try_from(density) else {
                    return Err(format!(
                        "Rust JPEG encoder cannot represent mrc2tif resolution {resolution}: JFIF density is limited to 65535"
                    ));
                };
                // The QImage shim converts a negative `-r` value from DPI
                // and leaves positive `-P`/`-m` values as pixels/cm.  JFIF
                // has those same two explicit units, so preserve them rather
                // than converting the value through a rounded dots/meter
                // intermediate.
                encoder.set_pixel_density(image::codecs::jpeg::PixelDensity {
                    density: (density, density),
                    unit: if resolution < 0 {
                        image::codecs::jpeg::PixelDensityUnit::Inches
                    } else {
                        image::codecs::jpeg::PixelDensityUnit::Centimeters
                    },
                });
            }
            encoder
                .write_image(&pixels, width as u32, height as u32, color.into())
                .map_err(|error| format!("Rust JPEG encoder failed: {error}"))
        }
        "PNG" => {
            let mut encoder = png::Encoder::new(file, width as u32, height as u32);
            encoder.set_color(if rgb {
                png::ColorType::Rgb
            } else {
                png::ColorType::Grayscale
            });
            encoder.set_depth(png::BitDepth::Eight);
            if resolution != 0 {
                // QImage stores dots per metre.  Its source calculation is
                // `resolution / (resolution > 0 ? .01 : -.0254)`, then the
                // floating result is narrowed to its integer DPM field.
                let dots_per_meter =
                    (resolution as f64 / if resolution > 0 { 0.01 } else { -0.0254 }) as u64;
                let Ok(dots_per_meter) = u32::try_from(dots_per_meter) else {
                    return Err(format!(
                        "Rust PNG encoder cannot represent mrc2tif resolution {resolution} as dots per meter"
                    ));
                };
                if dots_per_meter == 0 {
                    return Err(format!(
                        "Rust PNG encoder cannot represent mrc2tif resolution {resolution} as dots per meter"
                    ));
                }
                encoder.set_pixel_dims(Some(png::PixelDimensions {
                    xppu: dots_per_meter,
                    yppu: dots_per_meter,
                    unit: png::Unit::Meter,
                }));
            }
            encoder
                .write_header()
                .and_then(|mut writer| writer.write_image_data(&pixels))
                .map_err(|error| format!("Rust PNG encoder failed: {error}"))
        }
        _ => Err(format!("Rust encoder does not support {format}")),
    }
}

/// Stable unavailable-backend result for builds without the optional encoder.
#[cfg(not(feature = "rust-image-encoder"))]
pub fn save(
    _data: &[u8],
    _width: i32,
    _height: i32,
    _bytes_per_line: i32,
    _rgb: bool,
    _resolution: i32,
    _filename: &str,
    _format: &str,
    _quality: i32,
) -> Result<(), String> {
    Err("Rust encoder backend requires Cargo feature rust-image-encoder".into())
}

#[cfg(all(test, feature = "rust-image-encoder"))]
mod tests {
    use super::save;

    #[test]
    fn png_preserves_the_unpadded_qimage_rows() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-rust-encoder-{}-{}.png",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_file(&path);
        // Two grayscale pixels per row, followed by QImage's two padding bytes.
        save(
            &[4, 9, 99, 99, 16, 25, 88, 88],
            2,
            2,
            4,
            false,
            0,
            path.to_str().unwrap(),
            "PNG",
            -1,
        )
        .unwrap();
        let decoded = image::open(&path).unwrap().to_luma8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.into_raw(), vec![4, 9, 16, 25]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn resolution_values_outside_jfif_and_png_limits_fail_explicitly() {
        let jpeg = std::env::temp_dir().join(format!(
            "imod-rs-rust-encoder-density-{}-{}.jpg",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let png = std::env::temp_dir().join(format!(
            "imod-rs-rust-encoder-density-{}-{}.png",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_file(&jpeg);
        let _ = std::fs::remove_file(&png);
        assert!(
            save(
                &[1],
                1,
                1,
                1,
                false,
                -65_536,
                jpeg.to_str().unwrap(),
                "JPEG",
                -1,
            )
            .unwrap_err()
            .contains("JFIF density is limited to 65535")
        );
        assert!(
            save(
                &[1],
                1,
                1,
                1,
                false,
                i32::MAX,
                png.to_str().unwrap(),
                "PNG",
                -1,
            )
            .unwrap_err()
            .contains("cannot represent mrc2tif resolution")
        );
        std::fs::remove_file(jpeg).unwrap();
        std::fs::remove_file(png).unwrap();
    }
}
