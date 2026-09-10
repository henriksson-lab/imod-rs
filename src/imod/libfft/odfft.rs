//! Translation of `IMOD/libfft/odfft.c`.
use super::{cmplft, hermft, realft};

/// C `odfft`.
pub unsafe fn odfft(array: *mut f32, nxp: *mut i32, nyp: *mut i32, idirp: *mut i32) {
    unsafe {
        let nx = *nxp;
        let ny = *nyp;
        let idir = *idirp;
        if nx <= 0 || ny <= 0 || (idir >= 0 && nx & 1 != 0) {
            return;
        }
        let scale = (1.0 / nx as f64).sqrt() as f32;
        let mut dim = [0_i32; 6];
        match idir {
            -2 | -1 => {
                let stride = 2 * nx;
                let total = stride * ny;
                dim[1] = total;
                dim[2] = 2;
                dim[3] = total;
                dim[4] = total;
                dim[5] = stride;
                if idir == -2 {
                    for i in (0..total - 1).step_by(2) {
                        *array.add(i as usize) *= scale;
                        *array.add((i + 1) as usize) = -*array.add((i + 1) as usize) * scale;
                    }
                }
                cmplft(array, array.add(1), nx, dim.as_mut_ptr());
                if idir == -2 {
                    for i in (0..total - 1).step_by(2) {
                        *array.add((i + 1) as usize) = -*array.add((i + 1) as usize);
                    }
                } else {
                    for i in 0..total {
                        *array.add(i as usize) *= scale;
                    }
                }
            }
            0 => {
                let stride = nx + 2;
                let total = stride * ny;
                dim[1] = total;
                dim[2] = 2;
                dim[3] = total;
                dim[4] = total;
                dim[5] = stride;
                realft(array, array.add(1), nx / 2, dim.as_mut_ptr());
                for i in 0..total {
                    *array.add(i as usize) *= scale;
                }
            }
            1 => {
                let stride = nx + 2;
                let total = stride * ny;
                dim[1] = total;
                dim[2] = 2;
                dim[3] = total;
                dim[4] = total;
                dim[5] = stride;
                for i in (0..total - 1).step_by(2) {
                    *array.add(i as usize) *= scale;
                    *array.add((i + 1) as usize) = -*array.add((i + 1) as usize) * scale;
                }
                let mut index = 1;
                for _ in 0..ny {
                    *array.add(index as usize) = *array.add((nx - 1 + index) as usize);
                    index += stride;
                }
                hermft(array, array.add(1), nx / 2, dim.as_mut_ptr());
            }
            _ => {}
        }
    }
}

/// C `odfftc`.
pub unsafe fn odfft_c(array: *mut f32, nx: i32, ny: i32, idir: i32) {
    unsafe {
        let mut nx = nx;
        let mut ny = ny;
        let mut idir = idir;
        odfft(array, &mut nx, &mut ny, &mut idir);
    }
}

/// C `usingFFTW` for the translated IMOD FFT backend.
pub fn using_fftw() -> i32 {
    0
}

/// C `niceFFTlimit`.
pub fn nice_fft_limit() -> i32 {
    5
}

/// C `cleanupFFTplans`.
pub fn cleanup_fft_plans() {}

#[cfg(test)]
mod tests {
    use super::odfft_c;

    #[test]
    fn one_dimensional_real_round_trip_uses_padded_layout() {
        let mut values = vec![0.0_f32; 10];
        values[..8].copy_from_slice(&[1., 2., 4., 8., 16., 3., 7., 9.]);
        let expected = values.clone();
        unsafe {
            odfft_c(values.as_mut_ptr(), 8, 1, 0);
            odfft_c(values.as_mut_ptr(), 8, 1, 1);
        }
        for index in 0..8 {
            assert!((values[index] - expected[index]).abs() < 1.0e-4);
        }
    }

    #[test]
    fn one_dimensional_complex_round_trip_uses_source_directions() {
        let mut values = [
            1.0_f32, -2.0, 3.0, 4.0, -5.0, 2.0, 7.0, -1.0, 2.0, 6.0, -3.0, 8.0,
        ];
        let expected = values;
        unsafe {
            odfft_c(values.as_mut_ptr(), 6, 1, -1);
            odfft_c(values.as_mut_ptr(), 6, 1, -2);
        }
        for index in 0..values.len() {
            assert!((values[index] - expected[index]).abs() < 1.0e-4);
        }
    }
}
