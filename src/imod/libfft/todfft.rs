//! Translation of `IMOD/libfft/todfft.c`.
use super::{cmplft, hermft, realft};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use std::io::Write as _;

/// C `todfft`.
pub unsafe fn todfft(array: *mut f32, nxp: *mut i32, nyp: *mut i32, idirp: *mut i32) {
    unsafe {
        let nx = *nxp;
        let ny = *nyp;
        let idir = *idirp;
        match crate::imod::backends::fft_backend() {
            Ok(crate::imod::backends::FftBackend::Rustfft) => {
                // The backend indexes `ny` rows of `nx + 2` floats out of
                // the caller's buffer.
                let len = if nx > 0 && ny > 0 && nx & 1 == 0 {
                    (nx as usize + 2) * ny as usize
                } else {
                    0
                };
                let buffer = std::slice::from_raw_parts_mut(array, len);
                if let Err(error) = super::rustfft_backend::todfft(buffer, nx, ny, idir) {
                    eprintln!("ERROR: Rust-native backend - {error}");
                }
                return;
            }
            Ok(crate::imod::backends::FftBackend::Parity) => {}
            Err(error) => {
                eprintln!("{error}");
                return;
            }
        }
        // `todfft.c:74-78`: `if (2*nxo2 != nx)`, printing and ending the
        // program.  The source has no other guard here -- no check on a
        // non-positive `nx` or `ny` -- so neither does this.
        let nxo2 = nx / 2;
        if 2 * nxo2 != nx {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "ERROR: todfft - nx= %d must be even with IMOD FFT routines\n",
                &[CArg::Int(nx as i64)],
            ));
            std::process::exit(1);
        }
        let stride = nx + 2;
        let total = stride * ny;
        let scale = (1.0 / (nx * ny) as f64).sqrt() as f32;
        let mut dim = [0_i32; 6];
        if idir == 0 {
            dim[1] = total;
            dim[2] = 2;
            dim[3] = total;
            dim[4] = total;
            dim[5] = stride;
            realft(array, array.add(1), nxo2, dim.as_mut_ptr());
            dim[2] = stride;
            dim[4] = stride;
            dim[5] = 2;
            cmplft(array, array.add(1), ny, dim.as_mut_ptr());
            for index in (0..total - 1).step_by(2) {
                *array.add(index as usize) *= scale;
                *array.add((index + 1) as usize) *= scale;
            }
            return;
        }
        dim[1] = total;
        dim[2] = stride;
        dim[3] = total;
        dim[4] = stride;
        dim[5] = 2;
        for index in (0..total - 1).step_by(2) {
            *array.add(index as usize) *= scale;
            *array.add((index + 1) as usize) = -*array.add((index + 1) as usize) * scale;
        }
        cmplft(array, array.add(1), ny, dim.as_mut_ptr());
        let mut index = 1;
        for _ in 0..ny {
            *array.add(index as usize) = *array.add((nx - 1 + index) as usize);
            index += stride;
        }
        dim[2] = 2;
        dim[4] = total;
        dim[5] = stride;
        hermft(array, array.add(1), nxo2, dim.as_mut_ptr());
    }
}

/// C `todfftc`.
pub unsafe fn todfft_c(array: *mut f32, nx: i32, ny: i32, idir: i32) {
    unsafe {
        let mut nx = nx;
        let mut ny = ny;
        let mut idir = idir;
        todfft(array, &mut nx, &mut ny, &mut idir);
    }
}

/// C `parallelTodfft`.  The IMOD C backend prints an error when not using MKL;
/// sequential native execution has the same transform result.
pub unsafe fn parallel_todfft(
    array: *mut f32,
    nx: i32,
    ny: i32,
    idir: i32,
    num_images: i32,
    _num_threads: i32,
) {
    unsafe {
        for image in 0..num_images {
            todfft_c(array.add((image * (nx + 2) * ny) as usize), nx, ny, idir);
        }
    }
}

/// C `fftStartTimer` when `OLD_FFT_TIMES` is not defined.
pub fn fft_start_timer() -> f64 {
    0.0
}

/// C `fftAddTime` when `OLD_FFT_TIMES` is not defined.
pub unsafe fn fft_add_time(_start: f64, _cumul: *mut f64) {}

#[cfg(test)]
mod tests {
    use super::todfft_c;

    #[test]
    fn two_dimensional_round_trip_uses_imod_padded_layout() {
        let mut values = vec![0.0_f32; 6 * 4];
        for y in 0..4 {
            for x in 0..4 {
                values[y * 6 + x] = (3 * y + x) as f32;
            }
        }
        let expected = values.clone();
        unsafe {
            todfft_c(values.as_mut_ptr(), 4, 4, 0);
            todfft_c(values.as_mut_ptr(), 4, 4, 1);
        }
        for y in 0..4 {
            for x in 0..4 {
                assert!((values[y * 6 + x] - expected[y * 6 + x]).abs() < 1.0e-4);
            }
        }
    }
}
