//! Translation of `IMOD/libfft/cmplft.c`.

use super::{diprp, mdftkd, srfp};

/// C `cmplft`.
pub unsafe fn cmplft(x: *mut f32, y: *mut f32, n: i32, dim: *mut i32) {
    unsafe {
        let mut error = 0_i32;
        let mut psym = 0_i32;
        let mut factor = [0_i32; 16];
        let mut sym = [0_i32; 16];
        let mut unsym = [0_i32; 16];
        if n <= 1 {
            return;
        }
        srfp(
            n,
            19,
            8,
            factor.as_mut_ptr(),
            sym.as_mut_ptr(),
            &mut psym,
            unsym.as_mut_ptr(),
            &mut error,
        );
        if error != 0 {
            return;
        }
        mdftkd(n, factor.as_mut_ptr(), dim, x, y);
        diprp(n, sym.as_mut_ptr(), psym, unsym.as_mut_ptr(), dim, x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::cmplft;

    #[test]
    fn mixed_radix_kernel_transforms_a_four_point_complex_sequence() {
        let mut values = [1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let mut dimensions = [0_i32, 8, 2, 8, 2, 2];
        unsafe {
            cmplft(
                values.as_mut_ptr(),
                values.as_mut_ptr().add(1),
                4,
                dimensions.as_mut_ptr(),
            );
        }
        let expected = [10.0_f32, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
        for index in 0..8 {
            assert!((values[index] - expected[index]).abs() < 1.0e-4);
        }
    }

    #[test]
    fn mixed_radix_sizes_keep_an_impulse_flat_in_fourier_space() {
        for size in [2_i32, 3, 5, 8, 12, 19] {
            let mut values = vec![0.0_f32; (2 * size) as usize];
            values[0] = 1.0;
            let mut dimensions = [0_i32, 2 * size, 2, 2 * size, 2, 2];
            unsafe {
                cmplft(
                    values.as_mut_ptr(),
                    values.as_mut_ptr().add(1),
                    size,
                    dimensions.as_mut_ptr(),
                );
            }
            for frequency in 0..size as usize {
                assert!((values[2 * frequency] - 1.0).abs() < 1.0e-4, "size {size}");
                assert!(values[2 * frequency + 1].abs() < 1.0e-4, "size {size}");
            }
        }
    }
}
