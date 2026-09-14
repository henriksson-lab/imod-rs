//! Translation of `IMOD/libfft/cmplft.c`.

use super::{diprp, mdftkd, srfp};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use std::io::Write as _;

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
            &mut factor,
            &mut sym,
            &mut psym,
            &mut unsym,
            &mut error,
        );
        if error != 0 {
            // `cmplft.c:61-64`.  `srfp` sets `error` when `n` has a prime
            // factor above `pmax` (19); the source prints and ends the
            // program rather than leaving the caller with untransformed
            // data.  No command reaches it -- `clip_nicesize` rejects such a
            // size and `newstack` and `binvol` pad to a `niceFrame` one.
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "invalid number of points for cmplft.  n = %d\n",
                &[CArg::Int(n as i64)],
            ));
            std::process::exit(1);
        }
        // `mdftkd` walks `x` and `y` as biased pointers into one caller
        // buffer: every IMOD caller passes `y` inside the same array as `x`,
        // whose total float count is `d[1]` (`odfft.c:93`, `todfft.c:107,159`,
        // `realft.c:42`, `hermft.c:85`).  The translated `mdftkd` takes that
        // buffer as one slice plus the two biases, so rebuild it here.
        let dim_slice = std::slice::from_raw_parts(dim, 6);
        let y_bias = y.offset_from(x) as usize;
        mdftkd(
            n,
            &factor,
            dim_slice,
            std::slice::from_raw_parts_mut(x, dim_slice[1] as usize),
            0,
            y_bias,
        );
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
