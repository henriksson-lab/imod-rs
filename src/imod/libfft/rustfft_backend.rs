//! Experimental RustFFT adapters for IMOD's packed float FFT layouts.
//!
//! The callers retain IMOD's `nx + 2` real-array padding and interleaved
//! complex representation.  This module only converts at the execution
//! boundary.

#[cfg(feature = "rustfft-backend")]
use num_complex::Complex32;

/// Executes IMOD `odfft` layouts with RustFFT.
#[cfg(feature = "rustfft-backend")]
pub unsafe fn odfft(array: *mut f32, nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    use rustfft::FftPlanner;

    if array.is_null() || nx <= 0 || ny <= 0 || (idir >= 0 && nx & 1 != 0) {
        return Ok(());
    }
    let nx = nx as usize;
    let ny = ny as usize;
    let scale = 1.0 / (nx as f32).sqrt();
    let mut planner = FftPlanner::<f32>::new();
    match idir {
        -1 | -2 => {
            let data = unsafe { std::slice::from_raw_parts_mut(array, 2 * nx * ny) };
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
                if idir == -2 {
                    for value in &mut values {
                        *value *= scale;
                    }
                }
                fft.process(&mut values);
                if idir == -1 {
                    for value in &mut values {
                        *value *= scale;
                    }
                }
                for (index, value) in values.into_iter().enumerate() {
                    data[2 * (row * nx + index)] = value.re;
                    data[2 * (row * nx + index) + 1] = value.im;
                }
            }
        }
        0 | 1 => {
            let stride = nx + 2;
            let data = unsafe { std::slice::from_raw_parts_mut(array, stride * ny) };
            let half = nx / 2 + 1;
            if idir == 0 {
                let fft = planner.plan_fft_forward(nx);
                for row in 0..ny {
                    let mut values = (0..nx)
                        .map(|index| Complex32::new(data[row * stride + index], 0.0))
                        .collect::<Vec<_>>();
                    fft.process(&mut values);
                    for (index, value) in values.into_iter().take(half).enumerate() {
                        data[row * stride + 2 * index] = value.re * scale;
                        data[row * stride + 2 * index + 1] = value.im * scale;
                    }
                }
            } else {
                let fft = planner.plan_fft_inverse(nx);
                for row in 0..ny {
                    let mut values = vec![Complex32::new(0.0, 0.0); nx];
                    for index in 0..half {
                        values[index] = Complex32::new(
                            data[row * stride + 2 * index] * scale,
                            data[row * stride + 2 * index + 1] * scale,
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
    Ok(())
}

/// Executes IMOD `todfft` layouts with RustFFT.
#[cfg(feature = "rustfft-backend")]
pub unsafe fn todfft(array: *mut f32, nx: i32, ny: i32, idir: i32) -> Result<(), String> {
    use rustfft::FftPlanner;

    if array.is_null() || nx <= 0 || ny <= 0 || nx & 1 != 0 {
        return Ok(());
    }
    let nx = nx as usize;
    let ny = ny as usize;
    let stride = nx + 2;
    let half = nx / 2 + 1;
    let data = unsafe { std::slice::from_raw_parts_mut(array, stride * ny) };
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
                spectrum[row * half + index] = value * scale;
            }
        }
        for row in 0..ny {
            for index in 0..half {
                let value = spectrum[row * half + index];
                data[row * stride + 2 * index] = value.re;
                data[row * stride + 2 * index + 1] = value.im;
            }
        }
    } else if idir == 1 {
        let row_fft = planner.plan_fft_inverse(nx);
        let col_fft = planner.plan_fft_inverse(ny);
        let mut full = vec![Complex32::new(0.0, 0.0); nx * ny];
        for row in 0..ny {
            for index in 0..half {
                full[row * nx + index] = Complex32::new(
                    data[row * stride + 2 * index] * scale,
                    data[row * stride + 2 * index + 1] * scale,
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
    Ok(())
}

#[cfg(not(feature = "rustfft-backend"))]
pub unsafe fn odfft(_array: *mut f32, _nx: i32, _ny: i32, _idir: i32) -> Result<(), String> {
    Err("RustFFT backend requires Cargo feature rustfft-backend".into())
}

#[cfg(not(feature = "rustfft-backend"))]
pub unsafe fn todfft(_array: *mut f32, _nx: i32, _ny: i32, _idir: i32) -> Result<(), String> {
    Err("RustFFT backend requires Cargo feature rustfft-backend".into())
}
