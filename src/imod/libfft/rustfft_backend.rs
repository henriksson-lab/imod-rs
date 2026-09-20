//! Experimental RustFFT adapters for IMOD's packed float FFT layouts.
//!
//! The callers retain IMOD's `nx + 2` real-array padding and interleaved
//! complex representation.  This module only converts at the execution
//! boundary.  `array` is the caller's whole float buffer; the layout the
//! direction implies (`2 * nx * ny` interleaved floats, or `ny` rows of
//! `nx + 2`) is indexed out of it here.

#[cfg(feature = "rustfft-backend")]
use num_complex::Complex32;

/// Original static `normalize` (`libfft/fftw_wrap.c`).
pub fn normalize(array: &mut [f32], scale: f32, data_size: usize) -> Result<(), String> {
    let Some(values) = array.get_mut(..data_size) else {
        return Err(format!(
            "FFT packed buffer has {} floats; {} are required",
            array.len(),
            data_size
        ));
    };
    for value in values {
        *value *= scale;
    }
    Ok(())
}

/// Native `cleanupFFTplans`.  RustFFT plans are scoped to their execution,
/// so no process-global FFTW plan ring remains to destroy.
pub fn cleanup_fft_plans() {}

/// Native `usingFFTW`.  This backend provides the FFTW-compatible packed
/// layout and direction contract (the source value is 1 outside the MKL build).
pub const fn using_fftw() -> i32 {
    1
}

/// Native `niceFFTlimit`.
pub const fn nice_fft_limit() -> i32 {
    13
}

/// Executes IMOD `odfft` layouts with RustFFT.
#[cfg(feature = "rustfft-backend")]
pub fn odfft(array: &mut [f32], nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    use rustfft::FftPlanner;

    if nx <= 0 || ny <= 0 || (idir >= 0 && nx & 1 != 0) {
        return Ok(());
    }
    let nx = nx as usize;
    let ny = ny as usize;
    let scale = 1.0 / (nx as f32).sqrt();
    let required = if idir == -1 || idir == -2 {
        2 * nx * ny
    } else {
        (nx + 2) * ny
    };
    if array.len() < required {
        return Err(format!(
            "FFT packed buffer has {} floats; {required} are required",
            array.len()
        ));
    }
    let mut planner = FftPlanner::<f32>::new();
    match idir {
        -1 | -2 => {
            let data = &mut array[..2 * nx * ny];
            let fft = if idir == -1 {
                planner.plan_fft_forward(nx)
            } else {
                planner.plan_fft_inverse(nx)
            };
            for row in 0..ny {
                let mut values = (0..nx)
                    .map(|index| {
                        Complex32::new(
                            data[2 * (row * nx + index)],
                            data[2 * (row * nx + index) + 1],
                        )
                    })
                    .collect::<Vec<_>>();
                fft.process(&mut values);
                for (index, value) in values.into_iter().enumerate() {
                    data[2 * (row * nx + index)] = value.re;
                    data[2 * (row * nx + index) + 1] = value.im;
                }
            }
        }
        0 | 1 => {
            let stride = nx + 2;
            let data = &mut array[..stride * ny];
            let half = nx / 2 + 1;
            if idir == 0 {
                let fft = planner.plan_fft_forward(nx);
                for row in 0..ny {
                    let mut values = (0..nx)
                        .map(|index| Complex32::new(data[row * stride + index], 0.0))
                        .collect::<Vec<_>>();
                    fft.process(&mut values);
                    for (index, value) in values.into_iter().take(half).enumerate() {
                        data[row * stride + 2 * index] = value.re;
                        data[row * stride + 2 * index + 1] = value.im;
                    }
                }
            } else {
                let fft = planner.plan_fft_inverse(nx);
                for row in 0..ny {
                    let mut values = vec![Complex32::new(0.0, 0.0); nx];
                    for index in 0..half {
                        values[index] = Complex32::new(
                            data[row * stride + 2 * index],
                            data[row * stride + 2 * index + 1],
                        );
                    }
                    for index in 1..nx / 2 {
                        values[nx - index] = values[index].conj();
                    }
                    fft.process(&mut values);
                    for (index, value) in values.into_iter().enumerate() {
                        data[row * stride + index] = value.re;
                    }
                    data[row * stride + nx] = 0.0;
                    data[row * stride + nx + 1] = 0.0;
                }
            }
        }
        _ => {}
    }
    normalize(array, scale, required)
}

/// Executes IMOD `todfft` layouts with RustFFT.
#[cfg(feature = "rustfft-backend")]
pub fn todfft(array: &mut [f32], nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    use rustfft::FftPlanner;

    if nx <= 0 || ny <= 0 || nx & 1 != 0 {
        return Ok(());
    }
    let nx = nx as usize;
    let ny = ny as usize;
    let stride = nx + 2;
    let half = nx / 2 + 1;
    if array.len() < stride * ny {
        return Err(format!(
            "FFT packed buffer has {} floats; {} are required",
            array.len(),
            stride * ny
        ));
    }
    let data = &mut array[..stride * ny];
    let scale = 1.0 / ((nx * ny) as f32).sqrt();
    let mut planner = FftPlanner::<f32>::new();
    if idir == 0 {
        let row_fft = planner.plan_fft_forward(nx);
        let col_fft = planner.plan_fft_forward(ny);
        let mut spectrum = vec![Complex32::new(0.0, 0.0); half * ny];
        for row in 0..ny {
            let mut values = (0..nx)
                .map(|index| Complex32::new(data[row * stride + index], 0.0))
                .collect::<Vec<_>>();
            row_fft.process(&mut values);
            for index in 0..half {
                spectrum[row * half + index] = values[index];
            }
        }
        for index in 0..half {
            let mut column = (0..ny)
                .map(|row| spectrum[row * half + index])
                .collect::<Vec<_>>();
            col_fft.process(&mut column);
            for (row, value) in column.into_iter().enumerate() {
                spectrum[row * half + index] = value;
            }
        }
        for row in 0..ny {
            for index in 0..half {
                let value = spectrum[row * half + index];
                data[row * stride + 2 * index] = value.re;
                data[row * stride + 2 * index + 1] = value.im;
            }
        }
    } else {
        // `todfft.c:55` documents both `1` and `-1` as the inverse direction,
        // and the C source's forward branch is the only one it tests for, so
        // every other direction takes the inverse path.  `clip_fftvol`
        // (`clip/fft.cpp:264`) is the caller that passes -1.
        let row_fft = planner.plan_fft_inverse(nx);
        let col_fft = planner.plan_fft_inverse(ny);
        let mut full = vec![Complex32::new(0.0, 0.0); nx * ny];
        for row in 0..ny {
            for index in 0..half {
                full[row * nx + index] = Complex32::new(
                    data[row * stride + 2 * index],
                    data[row * stride + 2 * index + 1],
                );
            }
        }
        for row in 0..ny {
            for index in 1..nx / 2 {
                full[((ny - row) % ny) * nx + nx - index] = full[row * nx + index].conj();
            }
        }
        for index in 0..nx {
            let mut column = (0..ny)
                .map(|row| full[row * nx + index])
                .collect::<Vec<_>>();
            col_fft.process(&mut column);
            for (row, value) in column.into_iter().enumerate() {
                full[row * nx + index] = value;
            }
        }
        for row in 0..ny {
            let row_data = &mut full[row * nx..(row + 1) * nx];
            row_fft.process(row_data);
            for (index, value) in row_data.iter().enumerate() {
                data[row * stride + index] = value.re;
            }
            data[row * stride + nx] = 0.0;
            data[row * stride + nx + 1] = 0.0;
        }
    }
    normalize(array, scale, stride * ny)
}

/// Executes IMOD `thrdfft` layouts with RustFFT.
#[cfg(feature = "rustfft-backend")]
pub fn thrdfft(
    array: &mut [f32],
    _brray: Option<&mut [f32]>,
    nx: i32,
    ny: i32,
    nz: i32,
    idir: i32,
) -> Result<(), String> {
    use rustfft::FftPlanner;

    if nx <= 0 || ny <= 0 || nz <= 0 || nx & 1 != 0 {
        return Ok(());
    }
    let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
    let stride = nx + 2;
    let half = nx / 2 + 1;
    let required = stride * ny * nz;
    if array.len() < required {
        return Err(format!(
            "FFT packed buffer has {} floats; {required} are required",
            array.len()
        ));
    }
    let scale = 1.0 / ((nx * ny * nz) as f32).sqrt();
    let mut planner = FftPlanner::<f32>::new();
    let data = &mut array[..required];
    let spectrum_index = |z: usize, y: usize, x: usize| (z * ny + y) * half + x;
    if idir == 0 {
        let x_fft = planner.plan_fft_forward(nx);
        let y_fft = planner.plan_fft_forward(ny);
        let z_fft = planner.plan_fft_forward(nz);
        let mut spectrum = vec![Complex32::new(0., 0.); half * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                let base = (z * ny + y) * stride;
                let mut row = (0..nx)
                    .map(|x| Complex32::new(data[base + x], 0.))
                    .collect::<Vec<_>>();
                x_fft.process(&mut row);
                for x in 0..half {
                    spectrum[spectrum_index(z, y, x)] = row[x];
                }
            }
        }
        for z in 0..nz {
            for x in 0..half {
                let mut column = (0..ny)
                    .map(|y| spectrum[spectrum_index(z, y, x)])
                    .collect::<Vec<_>>();
                y_fft.process(&mut column);
                for (y, value) in column.into_iter().enumerate() {
                    spectrum[spectrum_index(z, y, x)] = value;
                }
            }
        }
        for y in 0..ny {
            for x in 0..half {
                let mut column = (0..nz)
                    .map(|z| spectrum[spectrum_index(z, y, x)])
                    .collect::<Vec<_>>();
                z_fft.process(&mut column);
                for (z, value) in column.into_iter().enumerate() {
                    spectrum[spectrum_index(z, y, x)] = value;
                }
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                let base = (z * ny + y) * stride;
                for x in 0..half {
                    let value = spectrum[spectrum_index(z, y, x)];
                    data[base + 2 * x] = value.re;
                    data[base + 2 * x + 1] = value.im;
                }
            }
        }
    } else {
        let x_fft = planner.plan_fft_inverse(nx);
        let y_fft = planner.plan_fft_inverse(ny);
        let z_fft = planner.plan_fft_inverse(nz);
        let full_index = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        let mut full = vec![Complex32::new(0., 0.); nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                let base = (z * ny + y) * stride;
                for x in 0..half {
                    full[full_index(z, y, x)] =
                        Complex32::new(data[base + 2 * x], data[base + 2 * x + 1]);
                }
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                for x in 1..nx / 2 {
                    full[full_index((nz - z) % nz, (ny - y) % ny, nx - x)] =
                        full[full_index(z, y, x)].conj();
                }
            }
        }
        for y in 0..ny {
            for x in 0..nx {
                let mut column = (0..nz)
                    .map(|z| full[full_index(z, y, x)])
                    .collect::<Vec<_>>();
                z_fft.process(&mut column);
                for (z, value) in column.into_iter().enumerate() {
                    full[full_index(z, y, x)] = value;
                }
            }
        }
        for z in 0..nz {
            for x in 0..nx {
                let mut column = (0..ny)
                    .map(|y| full[full_index(z, y, x)])
                    .collect::<Vec<_>>();
                y_fft.process(&mut column);
                for (y, value) in column.into_iter().enumerate() {
                    full[full_index(z, y, x)] = value;
                }
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                let row = &mut full[(z * ny + y) * nx..(z * ny + y + 1) * nx];
                x_fft.process(row);
                let base = (z * ny + y) * stride;
                for (x, value) in row.iter().enumerate() {
                    data[base + x] = value.re;
                }
                data[base + nx] = 0.;
                data[base + nx + 1] = 0.;
            }
        }
    }
    normalize(array, scale, required)
}

/// Native `odfftc`, the by-value C wrapper for [`odfft`].
pub fn odfftc(array: &mut [f32], nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    odfft(array, nx, ny, idir)
}

/// Native `todfftc`, the by-value C wrapper for [`todfft`].
pub fn todfftc(array: &mut [f32], nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    todfft(array, nx, ny, idir)
}

/// Native `thrdfftc`, the by-value C wrapper for [`thrdfft`].
pub fn thrdfftc(
    array: &mut [f32],
    brray: Option<&mut [f32]>,
    nx: i32,
    ny: i32,
    nz: i32,
    idir: i32,
) -> Result<(), String> {
    thrdfft(array, brray, nx, ny, nz, idir)
}

/// Native `parallelTodfft`.  The vendored FFTW build explicitly rejects this
/// Intel-MKL-only entry point; retain that observable unavailable outcome
/// rather than silently running a different parallel algorithm.
pub fn parallel_todfft(
    _array: &mut [f32],
    _nx: i32,
    _ny: i32,
    _idir: i32,
    _num_images: i32,
    _num_threads: i32,
) -> Result<(), String> {
    Err("ERROR: parallelTodfft can be used only with Intel FFT routines, not FFTW".to_owned())
}

#[cfg(not(feature = "rustfft-backend"))]
pub fn odfft(_array: &mut [f32], _nx: i32, _ny: i32, _idir: i32) -> Result<(), String> {
    Err("RustFFT backend requires Cargo feature rustfft-backend".into())
}

#[cfg(not(feature = "rustfft-backend"))]
pub fn todfft(_array: &mut [f32], _nx: i32, _ny: i32, _idir: i32) -> Result<(), String> {
    Err("RustFFT backend requires Cargo feature rustfft-backend".into())
}

#[cfg(not(feature = "rustfft-backend"))]
pub fn thrdfft(
    _array: &mut [f32],
    _brray: Option<&mut [f32]>,
    _nx: i32,
    _ny: i32,
    _nz: i32,
    _idir: i32,
) -> Result<(), String> {
    Err("RustFFT backend requires Cargo feature rustfft-backend".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_fft_capability_and_factor_rules_are_retained() {
        assert_eq!(using_fftw(), 1);
        assert_eq!(nice_fft_limit(), 13);
        let mut values = [2., 4., 9.];
        normalize(&mut values, 0.5, 2).unwrap();
        assert_eq!(values, [1., 2., 9.]);
        assert!(normalize(&mut values, 1., 4).is_err());
        assert_eq!(
            parallel_todfft(&mut [], 0, 0, 0, 0, 0),
            Err(
                "ERROR: parallelTodfft can be used only with Intel FFT routines, not FFTW"
                    .to_owned()
            )
        );
    }

    #[cfg(feature = "rustfft-backend")]
    #[test]
    fn packed_one_dimensional_wrapper_round_trips_with_source_normalization() {
        let original = [1., 2., -3., 4.];
        let mut packed = [original[0], original[1], original[2], original[3], 0., 0.];
        odfftc(&mut packed, 4, 1, 0).unwrap();
        odfftc(&mut packed, 4, 1, 1).unwrap();
        for (actual, expected) in packed[..4].iter().zip(original) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(&packed[4..], &[0., 0.]);
    }

    #[cfg(feature = "rustfft-backend")]
    #[test]
    fn packed_one_dimensional_forward_matches_native_fftw_wrapper_fixture() {
        // Captured by compiling IMOD/libfft/fftw_wrap.c against system FFTW3f
        // and invoking odfftc(values, 4, 1, 0).  This covers the native
        // nx+2 packing order and the wrapper's 1/sqrt(nx) normalization.
        let mut packed = [1., 2., -3., 4., 0., 0.];
        odfftc(&mut packed, 4, 1, 0).unwrap();
        let native = [2., 0., 2., 1., -4., 0.];
        for (actual, expected) in packed.iter().zip(native) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(feature = "rustfft-backend")]
    #[test]
    fn packed_two_dimensional_forward_matches_native_fftw_wrapper_fixture() {
        // Native `todfftc([1,2;3,4], 2, 2, 0)` packed into nx+2 rows.
        let mut packed = [1., 2., 0., 0., 3., 4., 0., 0.];
        todfftc(&mut packed, 2, 2, 0).unwrap();
        let native = [5., 0., -1., 0., -2., 0., 0., 0.];
        for (actual, expected) in packed.iter().zip(native) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(feature = "rustfft-backend")]
    #[test]
    fn packed_three_dimensional_wrapper_round_trips_with_source_normalization() {
        let original = [1., -2., 3., 4., 5., -6., 7., 8.];
        let mut packed = vec![0.; 4 * 2 * 2];
        for z in 0..2 {
            for y in 0..2 {
                let source = (z * 2 + y) * 2;
                let target = (z * 2 + y) * 4;
                packed[target..target + 2].copy_from_slice(&original[source..source + 2]);
            }
        }
        thrdfftc(&mut packed, None, 2, 2, 2, 0).unwrap();
        thrdfftc(&mut packed, None, 2, 2, 2, 1).unwrap();
        for z in 0..2 {
            for y in 0..2 {
                let source = (z * 2 + y) * 2;
                let target = (z * 2 + y) * 4;
                for (actual, expected) in packed[target..target + 2]
                    .iter()
                    .zip(&original[source..source + 2])
                {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
                assert_eq!(&packed[target + 2..target + 4], &[0., 0.]);
            }
        }
    }

    #[cfg(feature = "rustfft-backend")]
    #[test]
    fn packed_three_dimensional_forward_matches_native_fftw_wrapper_fixture() {
        // Native `thrdfftc([1..8], NULL, 2, 2, 2, 0)`, with nx+2 packed rows.
        let mut packed = [
            1., 2., 0., 0., 3., 4., 0., 0., 5., 6., 0., 0., 7., 8., 0., 0.,
        ];
        thrdfftc(&mut packed, None, 2, 2, 2, 0).unwrap();
        let native = [
            12.727_921_5,
            0.,
            -1.414_213_54,
            0.,
            -2.828_427_08,
            0.,
            0.,
            0.,
            -5.656_854_15,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        ];
        for (actual, expected) in packed.iter().zip(native) {
            assert!((actual - expected).abs() < 2.0e-5);
        }
    }
}
