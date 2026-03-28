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

// ---------------------------------------------------------------------------
// Edge stopping functions for anisotropic diffusion
// ---------------------------------------------------------------------------

/// Compute the edge-stopping (conductance) coefficient for one gradient value.
///
/// * `cc == 1`: Perona-Malik  g = exp(-grad^2 / k^2)
/// * `cc == 2`: Perona-Malik  g = 1 / (1 + grad^2 / k^2)
/// * `cc == 3` (or anything else): Tukey biweight  g = 0.5*(1 - grad^2/k^2)^2  if |diff| <= k, else 0
#[inline]
fn edge_stopping(cc: i32, grad: f64, diff: f64, ksq: f64, k: f64) -> f64 {
    match cc {
        1 => (-grad * grad / ksq).exp(),
        2 => 1.0 / (1.0 + grad * grad / ksq),
        _ => {
            // Tukey biweight
            if diff.abs() > k {
                0.0
            } else {
                let t = 1.0 - grad * grad / ksq;
                0.5 * t * t
            }
        }
    }
}

/// One iteration of the Perona-Malik anisotropic diffusion update.
///
/// `image_old` and `image_new` are padded arrays of size `(ny+2) x (nx+2)`.
/// Row-major with 1-based interior indexing (row 0 and row ny+1 are padding).
fn update_matrix(
    image_new: &mut [f64],
    image_old: &[f64],
    ny: usize,
    nx: usize,
    stride: usize, // = nx + 2
    cc: i32,
    k: f64,
    lambda: f64,
) {
    let ksq = k * k;

    for i in 1..=ny {
        let ip1 = if i == ny { ny } else { i + 1 };
        let im1 = if i == 1 { 1 } else { i - 1 };
        for j in 1..=nx {
            let jp1 = if j == nx { nx } else { j + 1 };
            let jm1 = if j == 1 { 1 } else { j - 1 };

            let center = image_old[i * stride + j];
            let diff_n = image_old[im1 * stride + j] - center;
            let diff_s = image_old[ip1 * stride + j] - center;
            let diff_e = image_old[i * stride + jp1] - center;
            let diff_w = image_old[i * stride + jm1] - center;

            // In the original code grad == diff for the standard approach
            let cn = edge_stopping(cc, diff_n, diff_n, ksq, k);
            let cs = edge_stopping(cc, diff_s, diff_s, ksq, k);
            let ce = edge_stopping(cc, diff_e, diff_e, ksq, k);
            let cw = edge_stopping(cc, diff_w, diff_w, ksq, k);

            image_new[i * stride + j] =
                center + lambda * (cn * diff_n + cs * diff_s + ce * diff_e + cw * diff_w);
        }
    }
}

/// Perona-Malik anisotropic diffusion (edge-preserving denoising).
///
/// This is a direct translation of IMOD's `sliceAnisoDiff` / `updateMatrix`.
///
/// * `cc` - edge stopping function type (1, 2, or 3).
/// * `k`  - gradient threshold parameter.
/// * `lambda` - step size (should be <= 0.25 for stability).
/// * `iterations` - number of diffusion iterations.
pub fn aniso_diff(slice: &mut Slice, cc: i32, k: f64, lambda: f64, iterations: usize) {
    let nx = slice.nx;
    let ny = slice.ny;
    let stride = nx + 2; // padded row width
    let padded_len = (ny + 2) * stride;

    // Allocate two padded buffers (1-indexed interior)
    let mut buf_a = vec![0.0f64; padded_len];
    let mut buf_b = vec![0.0f64; padded_len];

    // Copy slice data into buf_a with 1-based interior indexing
    for j in 0..ny {
        for i in 0..nx {
            buf_a[(j + 1) * stride + (i + 1)] = slice.get(i, j) as f64;
        }
    }

    // Run iterations, alternating buffers
    let mut imout = &buf_a as &[f64];
    for iter in 0..iterations {
        if iter % 2 == 0 {
            update_matrix(&mut buf_b, &buf_a, ny, nx, stride, cc, k, lambda);
            imout = &buf_b;
        } else {
            update_matrix(&mut buf_a, &buf_b, ny, nx, stride, cc, k, lambda);
            imout = &buf_a;
        }
    }

    // Copy result back
    for j in 0..ny {
        for i in 0..nx {
            slice.set(i, j, imout[(j + 1) * stride + (i + 1)] as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// Morphological operations
// ---------------------------------------------------------------------------

/// Count how many of the 8-connected neighbours (plus the pixel itself) equal `val`.
/// Returns that count minus 1 (i.e. pure neighbour count when self == val).
/// Pixels outside the *strict interior* (0 < x < nx, 0 < y < ny) are not counted.
///
/// This matches IMOD's `nay8`.
fn nay8(slice: &Slice, i: usize, j: usize, val: f32) -> i32 {
    if slice.get(i, j) != val {
        return 0;
    }
    let mut k: i32 = 0;
    for dn in -1i32..=1 {
        let y = j as i32 + dn;
        for dm in -1i32..=1 {
            let x = i as i32 + dm;
            if x > 0
                && y > 0
                && (x as usize) < slice.nx
                && (y as usize) < slice.ny
                && slice.get(x as usize, y as usize) == val
            {
                k += 1;
            }
        }
    }
    k - 1
}

/// Morphological dilation (grow) of a binary-ish f32 image.
///
/// Every pixel whose value equals `val` will expand its region by one pixel
/// into neighbouring pixels that currently hold the slice minimum value.
///
/// Translation of IMOD's `sliceByteGrow`.
pub fn byte_grow(slice: &mut Slice, val: f32) {
    let nx = slice.nx;
    let ny = slice.ny;
    let (min_val, max_val, _) = slice.statistics();
    let marker = max_val - 1.0;

    // Pass 1 - mark neighbours of `val` pixels that are currently min
    for j in 0..ny {
        for i in 0..nx {
            if slice.get(i, j) != val {
                continue;
            }
            for m in -1i32..=1 {
                let y = j as i32 + m;
                if y < 0 || y >= ny as i32 {
                    continue;
                }
                for n in -1i32..=1 {
                    let x = i as i32 + n;
                    if x == i as i32 && y == j as i32 {
                        continue;
                    }
                    if x < 0 || x >= nx as i32 {
                        continue;
                    }
                    if slice.get(x as usize, y as usize) == min_val {
                        slice.set(x as usize, y as usize, marker);
                    }
                }
            }
        }
    }

    // Pass 2 - convert markers to `val`
    for j in 0..ny {
        for i in 0..nx {
            if slice.get(i, j) == marker {
                slice.set(i, j, val);
            }
        }
    }
}

/// Morphological erosion (shrink) of a binary-ish f32 image.
///
/// Pixels that equal `val` but do not have all 8 neighbours also equal to `val`
/// are set to the slice minimum value.
///
/// Translation of IMOD's `sliceByteShrink`.  The original C code used a
/// threshold of `nay8 < 7` to decide which pixels to remove.
pub fn byte_shrink(slice: &mut Slice, val: f32) {
    let nx = slice.nx;
    let ny = slice.ny;
    let (min_val, _, _) = slice.statistics();

    // Build a mask of pixels to remove
    let mut remove = vec![false; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            if nay8(slice, i, j, val) < 7 {
                remove[j * nx + i] = true;
            }
        }
    }

    // Apply removals
    for j in 0..ny {
        for i in 0..nx {
            if remove[j * nx + i] {
                slice.set(i, j, min_val);
            }
        }
    }
}

/// Byte-level threshold: pixels below `thresh` become `high`, others become `low`.
///
/// Note: this matches IMOD's `sliceByteThreshold` convention where pixels
/// *below* the threshold are set to the *maximum* (high) value and pixels
/// *at or above* the threshold are set to the *minimum* (low) value.
pub fn byte_threshold(slice: &mut Slice, thresh: f32, low: f32, high: f32) {
    for v in &mut slice.data {
        *v = if *v < thresh { high } else { low };
    }
}
