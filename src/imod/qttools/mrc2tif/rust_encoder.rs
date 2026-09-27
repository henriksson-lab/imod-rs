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
    use std::io::Write as _;

    if width <= 0 || height <= 0 {
        return Err("Rust encoder requires positive image dimensions".into());
    }
    let channels = if rgb { 3usize } else { 1usize };
    let row_bytes = width as usize * channels;
    if bytes_per_line < row_bytes as i32 || data.len() < bytes_per_line as usize * height as usize {
        return Err("Rust encoder received invalid QImage row layout".into());
    }
    // The caller's rows carry QImage's 32-bit stride padding, which the Rust
    // encoders do not accept.  Strip it -- but only when there is any: an
    // image whose rows are already a multiple of four bytes arrives tightly
    // packed, and copying the whole image again to produce identical bytes is
    // pure overhead once per section.
    let unpadded;
    let pixels: &[u8] = if bytes_per_line as usize == row_bytes {
        &data[..row_bytes * height as usize]
    } else {
        let mut rows = vec![0_u8; row_bytes * height as usize];
        for row in 0..height as usize {
            let source =
                &data[row * bytes_per_line as usize..row * bytes_per_line as usize + row_bytes];
            rows[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(source);
        }
        unpadded = rows;
        &unpadded
    };
    let color = if rgb { ColorType::Rgb8 } else { ColorType::L8 };
    let file = File::create(filename)
        .map_err(|error| format!("Rust encoder could not create {filename}: {error}"))?;
    match format {
        "JPEG" => {
            // Qt's `QImage::save(..., "JPEG", quality)` (`qjpeghandler.cpp`
            // `write_jpeg_image`) drives libjpeg with `jpeg_set_defaults`, a
            // JFIF density taken from the QImage, `jpeg_set_quality(q, TRUE)`
            // and no other change.  `libjpeg_baseline` reproduces libjpeg's
            // encoder byte for byte for exactly that configuration, so the
            // file is the one the Qt boundary wrote.
            let quality = if quality >= 0 { quality.min(100) } else { 75 };
            // QImage keeps dots per metre as an `int`; mrc2tif sets it from
            // `-r`/`-P`/`-m` as `useResol / (useResol > 0. ? 0.01 : -0.0254)`
            // (`mrc2tif.cpp:586-589`, narrowed by the `int` parameter, and a
            // zero leaves it alone).  An unset QImage has Qt's default of
            // `qt_defaultDpiX() * 100 / 2.54` = 3937, from the 100 dpi Qt
            // assumes with no screen.
            let mut dots_per_meter = 3937_i32;
            if resolution != 0 {
                let dpm_f = resolution as f64 / if resolution > 0 { 0.01 } else { -0.0254 };
                // The C++ narrows a double to `int` here; out of range (e.g.
                // `-P` on a 1 A pixel: 1e8 pixels/cm -> 1e10 dots/m) that is
                // x86's `cvttsd2si`, which yields INT_MIN, not Rust's
                // saturating `as`.  The resulting density is what native writes.
                let dpm = if dpm_f.is_nan() || !(-2147483648.0..2147483648.0).contains(&dpm_f) {
                    i32::MIN
                } else {
                    dpm_f as i32
                };
                if dpm != 0 {
                    dots_per_meter = dpm;
                }
            }
            // `write_jpeg_image` writes whichever of dots/inch and dots/cm
            // is closer to integral; `qRound` of a positive value is
            // `int(d + 0.5)`, and libjpeg's densities are `UINT16`.
            let dpm = dots_per_meter as f64;
            let q_round = |d: f64| -> i32 {
                if d >= 0.0 {
                    (d + 0.5) as i32
                } else {
                    (d - ((d - 1.0) as i32) as f64 + 0.5) as i32 + (d - 1.0) as i32
                }
            };
            let diff_inch = 2.0 * (dpm * 2.54 / 100.0 - q_round(dpm * 2.54 / 100.0) as f64).abs();
            let diff_cm = 2.0 * (dpm / 100.0 - q_round(dpm / 100.0) as f64).abs() * 2.54;
            let (unit, density) = if diff_inch < diff_cm {
                (1_u8, q_round(dpm * 2.54 / 100.0) as u16)
            } else {
                (2_u8, ((dots_per_meter + 50) / 100) as u16)
            };
            let encoded = libjpeg_baseline(
                pixels,
                width as usize,
                height as usize,
                rgb,
                quality,
                unit,
                density,
            );
            let mut file = file;
            file.write_all(&encoded)
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
                .and_then(|mut writer| writer.write_image_data(pixels))
                .map_err(|error| format!("Rust PNG encoder failed: {error}"))
        }
        _ => Err(format!("Rust encoder does not support {format}")),
    }
}

/// libjpeg's `jpeg_natural_order`: position in the 8x8 block of the k-th
/// coefficient in zigzag order.
#[cfg(feature = "rust-image-encoder")]
#[rustfmt::skip]
const NATURAL_ORDER: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// One quantization table as libjpeg-turbo's `jcdctmgr.c` prepares it for
/// `JDCT_ISLOW` (`compute_reciprocal(quantval << 3)`), natural order.
#[cfg(feature = "rust-image-encoder")]
struct JpegDivisors {
    quantval: [u16; 64],
    recip: [u32; 64],
    corr: [u32; 64],
    /// `1 << (32 - r)`, libjpeg-turbo's `dtbl[DCTSIZE2 * 2]`.
    scale: [u32; 64],
}

/// One Huffman table: its DHT bits/values and `jpeg_make_c_derived_tbl`'s
/// code and size for every symbol.
#[cfg(feature = "rust-image-encoder")]
struct JpegHuffman {
    bits: &'static [u8; 16],
    values: &'static [u8],
    code: [u32; 256],
    size: [u32; 256],
}

/// The JPEG bit writer of libjpeg's `jchuff.c`: bits are packed MSB first,
/// every emitted 0xFF byte is followed by a stuffed 0x00, and the final
/// partial byte is filled with one bits (`flush_bits`).
#[cfg(feature = "rust-image-encoder")]
struct JpegBits<'a> {
    out: &'a mut Vec<u8>,
    buffer: u64,
    count: u32,
}

#[cfg(feature = "rust-image-encoder")]
impl JpegBits<'_> {
    #[inline(always)]
    fn put(&mut self, code: u32, size: u32) {
        // `size` is at most 16 (a Huffman code) or 11 (a value); fewer than
        // 32 bits are pending before a put, so the live bits stay below 48.
        // Bits above `count` are stale and never read.
        self.buffer = (self.buffer << size) | u64::from(code & ((1_u32 << size) - 1));
        self.count += size;
        if self.count >= 32 {
            self.count -= 32;
            let word = (self.buffer >> self.count) as u32;
            // Same bytes as emitting one at a time: only a 0xFF byte needs
            // the stuffed zero, and a word without one goes out whole.
            if (!word).wrapping_sub(0x0101_0101) & word & 0x8080_8080 == 0 {
                self.out.extend_from_slice(&word.to_be_bytes());
            } else {
                for byte in word.to_be_bytes() {
                    self.out.push(byte);
                    if byte == 0xFF {
                        self.out.push(0);
                    }
                }
            }
        }
    }

    fn flush(&mut self) {
        while self.count >= 8 {
            self.count -= 8;
            let byte = (self.buffer >> self.count) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
        }
        if self.count > 0 {
            let byte = ((self.buffer << (8 - self.count)) as u8) | (0xFF_u8 >> self.count);
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
            self.count = 0;
        }
        self.buffer = 0;
    }
}

/// One pass of libjpeg's `jpeg_fdct_islow` (`jfdctint.c`) over eight 1-D
/// transforms at once: `input[k * 8 + lane]` is element `k` of transform
/// `lane`, and so is the output.  `FIRST` selects pass 1 (rows: the even part
/// is scaled up by PASS1_BITS, the rest descaled by CONST_BITS - PASS1_BITS)
/// or pass 2 (columns: descaled by PASS1_BITS and CONST_BITS + PASS1_BITS).
/// The arithmetic per lane is the C's, in the C's order; the lanes are only
/// laid out so the compiler can process them side by side.
#[cfg(feature = "rust-image-encoder")]
#[inline(always)]
fn islow_pass<const FIRST: bool>(input: &[i32; 64], output: &mut [i32; 64]) {
    const CONST_BITS: i32 = 13;
    const PASS1_BITS: i32 = 2;
    const FIX_0_298631336: i32 = 2446;
    const FIX_0_390180644: i32 = 3196;
    const FIX_0_541196100: i32 = 4433;
    const FIX_0_765366865: i32 = 6270;
    const FIX_0_899976223: i32 = 7373;
    const FIX_1_175875602: i32 = 9633;
    const FIX_1_501321110: i32 = 12299;
    const FIX_1_847759065: i32 = 15137;
    const FIX_1_961570560: i32 = 16069;
    const FIX_2_053119869: i32 = 16819;
    const FIX_2_562915447: i32 = 20995;
    const FIX_3_072711026: i32 = 25172;
    let odd_shift = if FIRST {
        CONST_BITS - PASS1_BITS
    } else {
        CONST_BITS + PASS1_BITS
    };
    let descale = |x: i32, n: i32| (x + (1 << (n - 1))) >> n;
    for lane in 0..8 {
        let x = |k: usize| input[k * 8 + lane];
        let tmp0 = x(0) + x(7);
        let tmp7 = x(0) - x(7);
        let tmp1 = x(1) + x(6);
        let tmp6 = x(1) - x(6);
        let tmp2 = x(2) + x(5);
        let tmp5 = x(2) - x(5);
        let tmp3 = x(3) + x(4);
        let tmp4 = x(3) - x(4);
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        if FIRST {
            output[lane] = (tmp10 + tmp11) << PASS1_BITS;
            output[4 * 8 + lane] = (tmp10 - tmp11) << PASS1_BITS;
        } else {
            output[lane] = descale(tmp10 + tmp11, PASS1_BITS);
            output[4 * 8 + lane] = descale(tmp10 - tmp11, PASS1_BITS);
        }
        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        output[2 * 8 + lane] = descale(z1 + tmp13 * FIX_0_765366865, odd_shift);
        output[6 * 8 + lane] = descale(z1 + tmp12 * -FIX_1_847759065, odd_shift);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        output[7 * 8 + lane] = descale(tmp4 + z1 + z3, odd_shift);
        output[5 * 8 + lane] = descale(tmp5 + z2 + z4, odd_shift);
        output[3 * 8 + lane] = descale(tmp6 + z2 + z3, odd_shift);
        output[8 + lane] = descale(tmp7 + z1 + z4, odd_shift);
    }
}

/// libjpeg baseline JPEG encoder for the configuration Qt's JPEG writer uses:
/// `jpeg_set_defaults`, `jpeg_set_quality(quality, TRUE)`, JFIF APP0 with the
/// given density, `JDCT_ISLOW`, standard Huffman tables, no restart markers.
/// Grayscale input is one component; RGB is converted to YCbCr and written
/// with 2x2-subsampled chroma, as `jpeg_set_colorspace(JCS_YCbCr)` sets up.
/// Every stage follows libjpeg-turbo's C code (`jccolor.c`, `jcsample.c`,
/// `jcprepct.c` edge expansion, `jccoefct.c` dummy blocks, `jfdctint.c`,
/// `jcdctmgr.c` quantization, `jchuff.c`, `jcmarker.c`), whose SIMD versions
/// produce the same output, so the bytes equal the Qt/libjpeg file.
#[cfg(feature = "rust-image-encoder")]
fn libjpeg_baseline(
    pixels: &[u8],
    width: usize,
    height: usize,
    rgb: bool,
    quality: i32,
    density_unit: u8,
    density: u16,
) -> Vec<u8> {
    #[rustfmt::skip]
    const STD_LUMINANCE_QUANT: [u32; 64] = [
        16, 11, 10, 16,  24,  40,  51,  61,
        12, 12, 14, 19,  26,  58,  60,  55,
        14, 13, 16, 24,  40,  57,  69,  56,
        14, 17, 22, 29,  51,  87,  80,  62,
        18, 22, 37, 56,  68, 109, 103,  77,
        24, 35, 55, 64,  81, 104, 113,  92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103,  99,
    ];
    #[rustfmt::skip]
    const STD_CHROMINANCE_QUANT: [u32; 64] = [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ];
    const BITS_DC_LUMINANCE: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    const VAL_DC: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const BITS_DC_CHROMINANCE: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    const BITS_AC_LUMINANCE: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
    #[rustfmt::skip]
    const VAL_AC_LUMINANCE: [u8; 162] = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ];
    const BITS_AC_CHROMINANCE: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
    #[rustfmt::skip]
    const VAL_AC_CHROMINANCE: [u8; 162] = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ];

    // `jpeg_quality_scaling` and `jpeg_add_quant_table(..., force_baseline)`.
    let quality = quality.clamp(1, 100);
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    } as i64;
    let divisors = |basic: &[u32; 64]| -> JpegDivisors {
        let mut table = JpegDivisors {
            quantval: [0; 64],
            recip: [0; 64],
            corr: [0; 64],
            scale: [0; 64],
        };
        for i in 0..64 {
            let temp = ((basic[i] as i64 * scale + 50) / 100).clamp(1, 255);
            table.quantval[i] = temp as u16;
            // `compute_reciprocal(divisor)` with a 16-bit DCTELEM.
            let divisor = (temp as u32) << 3;
            let b = 31 - divisor.leading_zeros();
            let mut r = 16 + b;
            let mut fq = (1_u32 << r) / divisor;
            let fr = (1_u32 << r) % divisor;
            let mut c = divisor / 2;
            if fr == 0 {
                fq >>= 1;
                r -= 1;
            } else if fr <= divisor / 2 {
                c += 1;
            } else {
                fq += 1;
            }
            table.recip[i] = fq & 0xFFFF;
            table.corr[i] = c & 0xFFFF;
            table.scale[i] = 1 << (32 - r);
        }
        table
    };
    let quant = [
        divisors(&STD_LUMINANCE_QUANT),
        divisors(&STD_CHROMINANCE_QUANT),
    ];
    // `jpeg_make_c_derived_tbl`.
    let huffman = |bits: &'static [u8; 16], values: &'static [u8]| -> JpegHuffman {
        let mut table = JpegHuffman {
            bits,
            values,
            code: [0; 256],
            size: [0; 256],
        };
        let mut code = 0_u32;
        let mut p = 0;
        for length in 1..=16_u32 {
            for _ in 0..bits[length as usize - 1] {
                table.code[values[p] as usize] = code;
                table.size[values[p] as usize] = length;
                code += 1;
                p += 1;
            }
            code <<= 1;
        }
        table
    };
    let dc = [
        huffman(&BITS_DC_LUMINANCE, &VAL_DC),
        huffman(&BITS_DC_CHROMINANCE, &VAL_DC),
    ];
    let ac = [
        huffman(&BITS_AC_LUMINANCE, &VAL_AC_LUMINANCE),
        huffman(&BITS_AC_CHROMINANCE, &VAL_AC_CHROMINANCE),
    ];

    // `write_file_header`, `write_frame_header`, `write_scan_header`.
    let mut out = Vec::with_capacity(width * height / 4 + 1024);
    out.extend_from_slice(&[
        0xFF, 0xD8, 0xFF, 0xE0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1,
    ]);
    out.push(density_unit);
    out.extend_from_slice(&density.to_be_bytes());
    out.extend_from_slice(&density.to_be_bytes());
    out.extend_from_slice(&[0, 0]);
    let tables = if rgb { 2 } else { 1 };
    for (index, table) in quant.iter().enumerate().take(tables) {
        out.extend_from_slice(&[0xFF, 0xDB, 0, 67, index as u8]);
        for k in 0..64 {
            out.push(table.quantval[NATURAL_ORDER[k]] as u8);
        }
    }
    let components = if rgb { 3_u8 } else { 1 };
    out.extend_from_slice(&[0xFF, 0xC0, 0, 8 + 3 * components, 8]);
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(components);
    if rgb {
        out.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    } else {
        out.extend_from_slice(&[1, 0x11, 0]);
    }
    for index in 0..tables {
        for (class, table) in [(0x00_u8, &dc[index]), (0x10, &ac[index])] {
            let count: usize = table.bits.iter().map(|&b| b as usize).sum();
            out.extend_from_slice(&[0xFF, 0xC4]);
            out.extend_from_slice(&((2 + 1 + 16 + count) as u16).to_be_bytes());
            out.push(class | index as u8);
            out.extend_from_slice(table.bits);
            out.extend_from_slice(&table.values[..count]);
        }
    }
    out.extend_from_slice(&[0xFF, 0xDA, 0, 6 + 2 * components, components]);
    if rgb {
        out.extend_from_slice(&[1, 0x00, 2, 0x11, 3, 0x11]);
    } else {
        out.extend_from_slice(&[1, 0x00]);
    }
    out.extend_from_slice(&[0, 63, 0]);

    // `convsamp` + `jpeg_fdct_islow` + `quantize` for one block of samples
    // already level-shifted by CENTERJSAMPLE, stored column by column
    // (`data[c * 8 + r]` is row r, column c), so pass 1 runs the eight rows
    // side by side.  `coef` receives the quantized coefficients in natural
    // order.
    let forward = |data: &mut [i32; 64], table: &JpegDivisors, coef: &mut [i32; 64]| {
        let mut rows = [0_i32; 64];
        islow_pass::<true>(data, &mut rows);
        // rows[k * 8 + r] is pass-1 output k of row r; pass 2 wants element
        // k (row k) of column-transform c at [k * 8 + c].
        for k in 0..8 {
            for c in 0..8 {
                data[k * 8 + c] = rows[c * 8 + k];
            }
        }
        islow_pass::<false>(data, &mut rows);
        // libjpeg-turbo's `quantize`: `((|x| + corr) * recip) >> r`, taken
        // in two steps of 16 as its SIMD version does
        // (`pmulhuw` by recip, then by scale = 1 << (32 - r)); the two are
        // the same floor division by 2^r, and every product fits in 32 bits
        // (|x| + corr < 2^16, recip < 2^16, and the first quotient times
        // scale < 2^30).
        for i in 0..64 {
            let temp = rows[i];
            let magnitude = temp.unsigned_abs();
            let product =
                (((magnitude + table.corr[i]) * table.recip[i]) >> 16) * table.scale[i] >> 16;
            coef[i] = if temp < 0 {
                -(product as i32)
            } else {
                product as i32
            };
        }
    };
    // `encode_one_block`.
    let encode = |bits: &mut JpegBits,
                  coef: &[i32; 64],
                  last_dc: &mut i32,
                  dct: &JpegHuffman,
                  act: &JpegHuffman| {
        let temp = coef[0] - *last_dc;
        *last_dc = coef[0];
        let magnitude = temp.unsigned_abs();
        let nbits = 32 - magnitude.leading_zeros();
        bits.put(dct.code[nbits as usize], dct.size[nbits as usize]);
        if nbits != 0 {
            let value = if temp < 0 {
                (temp - 1) as u32
            } else {
                temp as u32
            };
            bits.put(value, nbits);
        }
        // The AC run-length loop over the zigzag order, driven by a mask of
        // the nonzero coefficients rather than a test per coefficient (the
        // symbols emitted are the same).
        let mut zigzag = [0_i32; 64];
        for k in 1..64 {
            zigzag[k] = coef[NATURAL_ORDER[k]];
        }
        let mut nonzero = 0_u64;
        for k in 1..64 {
            nonzero |= u64::from(zigzag[k] != 0) << k;
        }
        let mut previous = 0_u32;
        while nonzero != 0 {
            let k = nonzero.trailing_zeros();
            nonzero &= nonzero - 1;
            let mut run = k - previous - 1;
            previous = k;
            while run > 15 {
                bits.put(act.code[0xF0], act.size[0xF0]);
                run -= 16;
            }
            let temp = zigzag[k as usize];
            let magnitude = temp.unsigned_abs();
            let nbits = 32 - magnitude.leading_zeros();
            let symbol = ((run << 4) + nbits) as usize;
            bits.put(act.code[symbol], act.size[symbol]);
            let value = if temp < 0 {
                (temp - 1) as u32
            } else {
                temp as u32
            };
            bits.put(value, nbits);
        }
        let run = 63 - previous;
        if run > 0 {
            bits.put(act.code[0], act.size[0]);
        }
    };

    let mut data = [0_i32; 64];
    let mut coef = [0_i32; 64];
    let mut scan = Vec::with_capacity(width * height / 4);
    let mut bits = JpegBits {
        out: &mut scan,
        buffer: 0,
        count: 0,
    };
    if !rgb {
        // One component: blocks of the image padded by edge replication
        // (`expand_right_edge`, `expand_bottom_edge`), raster order.
        let mut last_dc = 0;
        for by in 0..height.div_ceil(8) {
            let rows: [usize; 8] = std::array::from_fn(|r| (by * 8 + r).min(height - 1) * width);
            for bx in 0..width.div_ceil(8) {
                let x0 = bx * 8;
                if x0 + 8 <= width {
                    for r in 0..8 {
                        let row = &pixels[rows[r] + x0..rows[r] + x0 + 8];
                        for c in 0..8 {
                            data[c * 8 + r] = row[c] as i32 - 128;
                        }
                    }
                } else {
                    for r in 0..8 {
                        for c in 0..8 {
                            data[c * 8 + r] =
                                pixels[rows[r] + (x0 + c).min(width - 1)] as i32 - 128;
                        }
                    }
                }
                forward(&mut data, &quant[0], &mut coef);
                encode(&mut bits, &coef, &mut last_dc, &dc[0], &ac[0]);
            }
        }
    } else {
        // `rgb_ycc_convert` into full-resolution planes padded to whole
        // MCUs by edge replication, then `h2v2_downsample` for Cb and Cr.
        const SCALEBITS: i32 = 16;
        const ONE_HALF: i64 = 1 << (SCALEBITS - 1);
        const CBCR_OFFSET: i64 = 128 << SCALEBITS;
        let fix = |x: f64| (x * (1_i64 << SCALEBITS) as f64 + 0.5) as i64;
        let mcus_x = width.div_ceil(16);
        let mcus_y = height.div_ceil(16);
        let plane_w = mcus_x * 16;
        let plane_h = mcus_y * 16;
        let mut y_plane = vec![0_u8; plane_w * plane_h];
        let mut cb_full = vec![0_u8; plane_w * plane_h];
        let mut cr_full = vec![0_u8; plane_w * plane_h];
        let (r_y, g_y, b_y) = (fix(0.29900), fix(0.58700), fix(0.11400));
        let (r_cb, g_cb, b_cb) = (-fix(0.16874), -fix(0.33126), fix(0.50000));
        let (g_cr, b_cr) = (-fix(0.41869), -fix(0.08131));
        for y in 0..plane_h {
            let source = &pixels[y.min(height - 1) * width * 3..];
            for x in 0..plane_w {
                let p = x.min(width - 1) * 3;
                let (r, g, b) = (source[p] as i64, source[p + 1] as i64, source[p + 2] as i64);
                let at = y * plane_w + x;
                y_plane[at] = ((r_y * r + g_y * g + b_y * b + ONE_HALF) >> SCALEBITS) as u8;
                cb_full[at] = ((r_cb * r + g_cb * g + b_cb * b + CBCR_OFFSET + ONE_HALF - 1)
                    >> SCALEBITS) as u8;
                cr_full[at] = ((b_cb * r + g_cr * g + b_cr * b + CBCR_OFFSET + ONE_HALF - 1)
                    >> SCALEBITS) as u8;
            }
        }
        let chroma_w = mcus_x * 8;
        let chroma_h = mcus_y * 8;
        let real_chroma_rows = height.div_ceil(2);
        let downsample = |full: &[u8]| -> Vec<u8> {
            let mut plane = vec![0_u8; chroma_w * chroma_h];
            for oy in 0..chroma_h {
                let iy = oy.min(real_chroma_rows - 1) * 2;
                let row0 = &full[iy * plane_w..];
                let row1 = &full[(iy + 1) * plane_w..];
                for ox in 0..chroma_w {
                    let bias = if ox & 1 == 0 { 1 } else { 2 };
                    let sum = row0[2 * ox] as u32
                        + row0[2 * ox + 1] as u32
                        + row1[2 * ox] as u32
                        + row1[2 * ox + 1] as u32
                        + bias;
                    plane[oy * chroma_w + ox] = (sum >> 2) as u8;
                }
            }
            plane
        };
        let cb_plane = downsample(&cb_full);
        let cr_plane = downsample(&cr_full);
        let y_blocks_w = width.div_ceil(8);
        let y_blocks_h = height.div_ceil(8);
        let last_col_width = if y_blocks_w % 2 == 0 { 2 } else { 1 };
        let last_row_height = if y_blocks_h % 2 == 0 { 2 } else { 1 };
        let mut last_dc = [0_i32; 3];
        let mut blocks = [[0_i32; 64]; 4];
        let load = |data: &mut [i32; 64], plane: &[u8], stride: usize, x0: usize, y0: usize| {
            for r in 0..8 {
                let row = &plane[(y0 + r) * stride + x0..(y0 + r) * stride + x0 + 8];
                for c in 0..8 {
                    data[c * 8 + r] = row[c] as i32 - 128;
                }
            }
        };
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                // `compress_data`: the Y component's 2x2 blocks, with dummy
                // blocks past the image carrying the previous block's DC.
                let block_count = if mx < mcus_x - 1 { 2 } else { last_col_width };
                for yindex in 0..2 {
                    let first = yindex * 2;
                    if my < mcus_y - 1 || yindex < last_row_height {
                        for xindex in 0..block_count {
                            load(
                                &mut data,
                                &y_plane,
                                plane_w,
                                mx * 16 + xindex * 8,
                                my * 16 + yindex * 8,
                            );
                            forward(&mut data, &quant[0], &mut blocks[first + xindex]);
                        }
                        for xindex in block_count..2 {
                            let dc_value = blocks[first + xindex - 1][0];
                            blocks[first + xindex] = [0; 64];
                            blocks[first + xindex][0] = dc_value;
                        }
                    } else {
                        let dc_value = blocks[first - 1][0];
                        for xindex in 0..2 {
                            blocks[first + xindex] = [0; 64];
                            blocks[first + xindex][0] = dc_value;
                        }
                    }
                }
                for block in &blocks {
                    encode(&mut bits, block, &mut last_dc[0], &dc[0], &ac[0]);
                }
                load(&mut data, &cb_plane, chroma_w, mx * 8, my * 8);
                forward(&mut data, &quant[1], &mut coef);
                encode(&mut bits, &coef, &mut last_dc[1], &dc[1], &ac[1]);
                load(&mut data, &cr_plane, chroma_w, mx * 8, my * 8);
                forward(&mut data, &quant[1], &mut coef);
                encode(&mut bits, &coef, &mut last_dc[2], &dc[1], &ac[1]);
            }
        }
    }
    bits.flush();
    out.extend_from_slice(&scan);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
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
    fn resolution_values_outside_png_limits_fail_explicitly() {
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
        // JPEG takes Qt's path instead: libjpeg's `UINT16` density simply
        // wraps, as the Qt boundary's `cinfo.X_density = qRound(...)` does.
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
            .is_ok()
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
