//! Translation of `IMOD/libfft/todfft.c`.

use super::{cmplft, hermft, realft};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use std::io::Write as _;

/// C `todfft` over IMOD's `(nx + 2) * ny` packed-real storage.
pub fn todfft(array: &mut [f32], nx: i32, ny: i32, idir: i32) {
    let total = usize::try_from(nx)
        .ok()
        .and_then(|width| width.checked_add(2))
        .and_then(|width| {
            usize::try_from(ny)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .expect("TODFFT dimensions must be nonnegative and fit in memory");
    assert!(
        array.len() >= total,
        "packed TODFFT buffer is shorter than its dimensions require"
    );
    let array = &mut array[..total];

    match crate::imod::backends::fft_backend() {
        Ok(crate::imod::backends::FftBackend::Rustfft) => {
            if let Err(error) = super::rustfft_backend::todfft(array, nx, ny, idir) {
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
    // program.  The source has no other guard here.
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
        realft(array, nxo2, &mut dim);
        dim[2] = stride;
        dim[4] = stride;
        dim[5] = 2;
        cmplft(array, ny, &mut dim);
        for index in (0..total - 1).step_by(2) {
            array[index as usize] *= scale;
            array[(index + 1) as usize] *= scale;
        }
        return;
    }
    dim[1] = total;
    dim[2] = stride;
    dim[3] = total;
    dim[4] = stride;
    dim[5] = 2;
    for index in (0..total - 1).step_by(2) {
        array[index as usize] *= scale;
        array[(index + 1) as usize] = -array[(index + 1) as usize] * scale;
    }
    cmplft(array, ny, &mut dim);
    let mut index = 1;
    for _ in 0..ny {
        array[index as usize] = array[(nx - 1 + index) as usize];
        index += stride;
    }
    dim[2] = 2;
    dim[4] = total;
    dim[5] = stride;
    hermft(array, nxo2, &mut dim);
}

/// C `todfftc`.
pub fn todfft_c(array: &mut [f32], nx: i32, ny: i32, idir: i32) {
    todfft(array, nx, ny, idir);
}

/// C `parallelTodfft`.  The IMOD C backend prints an error when not using MKL;
/// sequential native execution has the same transform result.
pub fn parallel_todfft(
    array: &mut [f32],
    nx: i32,
    ny: i32,
    idir: i32,
    num_images: i32,
    _num_threads: i32,
) {
    let image_len = usize::try_from(nx)
        .ok()
        .and_then(|width| width.checked_add(2))
        .and_then(|width| {
            usize::try_from(ny)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .expect("TODFFT dimensions must be nonnegative and fit in memory");
    let image_count = usize::try_from(num_images).expect("TODFFT image count must be nonnegative");
    assert!(
        array.len() >= image_len * image_count,
        "TODFFT image buffer is shorter than its dimensions require"
    );
    for image in array[..image_len * image_count].chunks_exact_mut(image_len) {
        todfft(image, nx, ny, idir);
    }
}

/// C `fftStartTimer` when `OLD_FFT_TIMES` is not defined.
pub fn fft_start_timer() -> f64 {
    0.0
}

/// C `fftAddTime` when `OLD_FFT_TIMES` is not defined.
pub fn fft_add_time(_start: f64, _cumul: &mut f64) {}

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
        todfft_c(&mut values, 4, 4, 0);
        todfft_c(&mut values, 4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                assert!((values[y * 6 + x] - expected[y * 6 + x]).abs() < 1.0e-4);
            }
        }
    }

    #[test]
    fn batched_transforms_use_each_packed_image() {
        let mut values = vec![0.0_f32; 6 * 2 * 2];
        values[0] = 1.;
        values[12] = 2.;
        let expected = values.clone();

        super::parallel_todfft(&mut values, 4, 2, 0, 2, 1);
        super::parallel_todfft(&mut values, 4, 2, 1, 2, 1);

        for image in 0..2 {
            for y in 0..2 {
                for x in 0..4 {
                    let index = image * 12 + y * 6 + x;
                    assert!((values[index] - expected[index]).abs() < 1.0e-4);
                }
            }
        }
    }
}
