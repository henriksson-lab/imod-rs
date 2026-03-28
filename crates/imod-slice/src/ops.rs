use crate::Slice;

/// Scale pixel values: out = val * scale + offset.
pub fn scale(slice: &mut Slice, scale: f32, offset: f32) {
    for v in &mut slice.data {
        *v = *v * scale + offset;
    }
}

/// Clamp pixel values to [min, max].
pub fn clamp(slice: &mut Slice, min: f32, max: f32) {
    for v in &mut slice.data {
        *v = v.clamp(min, max);
    }
}

/// Threshold: values >= threshold become high, others become low.
pub fn threshold(slice: &mut Slice, thresh: f32, low: f32, high: f32) {
    for v in &mut slice.data {
        *v = if *v >= thresh { high } else { low };
    }
}

/// Invert: out = max + min - val.
pub fn invert(slice: &mut Slice) {
    let (min, max, _) = slice.statistics();
    for v in &mut slice.data {
        *v = max + min - *v;
    }
}

/// Add two slices pixel-by-pixel. They must be the same size.
pub fn add(a: &Slice, b: &Slice) -> Slice {
    assert_eq!(a.nx, b.nx);
    assert_eq!(a.ny, b.ny);
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x + y).collect();
    Slice::from_data(a.nx, a.ny, data)
}

/// Subtract b from a pixel-by-pixel.
pub fn subtract(a: &Slice, b: &Slice) -> Slice {
    assert_eq!(a.nx, b.nx);
    assert_eq!(a.ny, b.ny);
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x - y).collect();
    Slice::from_data(a.nx, a.ny, data)
}

/// Multiply two slices pixel-by-pixel.
pub fn multiply(a: &Slice, b: &Slice) -> Slice {
    assert_eq!(a.nx, b.nx);
    assert_eq!(a.ny, b.ny);
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x * y).collect();
    Slice::from_data(a.nx, a.ny, data)
}

/// Apply a 3x3 convolution kernel to the slice.
pub fn convolve_3x3(slice: &Slice, kernel: &[f32; 9]) -> Slice {
    let nx = slice.nx;
    let ny = slice.ny;
    let mut out = Slice::new(nx, ny, 0.0);
    let mean = slice.statistics().2;

    for y in 0..ny {
        for x in 0..nx {
            let mut sum = 0.0;
            let mut ki = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    sum += slice.get_clamped(x as isize + dx as isize, y as isize + dy as isize, mean) * kernel[ki];
                    ki += 1;
                }
            }
            out.set(x, y, sum);
        }
    }
    out
}

/// Sobel edge detection (magnitude of gradient).
pub fn sobel(slice: &Slice) -> Slice {
    let gx_kernel = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    let gy_kernel = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];
    let gx = convolve_3x3(slice, &gx_kernel);
    let gy = convolve_3x3(slice, &gy_kernel);

    let data: Vec<f32> = gx.data.iter().zip(gy.data.iter())
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect();
    Slice::from_data(slice.nx, slice.ny, data)
}

/// Gaussian blur with a 3x3 approximation kernel.
pub fn blur_3x3(slice: &Slice) -> Slice {
    let kernel = [
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    ];
    convolve_3x3(slice, &kernel)
}

/// Median filter with a 3x3 window.
pub fn median_3x3(slice: &Slice) -> Slice {
    let nx = slice.nx;
    let ny = slice.ny;
    let mean = slice.statistics().2;
    let mut out = Slice::new(nx, ny, 0.0);
    let mut window = [0.0f32; 9];

    for y in 0..ny {
        for x in 0..nx {
            let mut i = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    window[i] = slice.get_clamped(x as isize + dx as isize, y as isize + dy as isize, mean);
                    i += 1;
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.set(x, y, window[4]);
        }
    }
    out
}

/// Bin (downsample) by an integer factor, averaging pixels.
pub fn bin(slice: &Slice, factor: usize) -> Slice {
    let nx_out = slice.nx / factor;
    let ny_out = slice.ny / factor;
    let inv_area = 1.0 / (factor * factor) as f32;
    let mut out = Slice::new(nx_out, ny_out, 0.0);

    for yo in 0..ny_out {
        for xo in 0..nx_out {
            let mut sum = 0.0;
            for dy in 0..factor {
                for dx in 0..factor {
                    sum += slice.get(xo * factor + dx, yo * factor + dy);
                }
            }
            out.set(xo, yo, sum * inv_area);
        }
    }
    out
}
