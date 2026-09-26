//! Translation of `IMOD/libfft/thrdfft.c`.

use super::{odfft, todfft};

/// C `thrdfft`.
pub fn thrdfft(array: &mut [f32], brray: &mut [f32], nx: i32, ny: i32, nz: i32, idir: i32) {
    let stride = nx + 2;
    let nxo2 = (nx + 2) / 2;
    let plane_len = (stride * ny) as usize;
    let array_len = plane_len * nz as usize;
    let work_len = (2 * nz * nxo2) as usize;
    assert!(array.len() >= array_len);
    assert!(brray.len() >= work_len);
    let oddir = if idir != 0 { -2 } else { -1 };
    let back = 1;
    if idir == 0 {
        for plane in array[..array_len].chunks_mut(plane_len) {
            todfft::todfft_c(plane, nx, ny, idir);
        }
    }
    // Unchecked gather/scatter (TO_OPT.md, "combinefft single-thread").
    // Soundness: with `stride = nx + 2` and `nxo2 = (nx + 2) / 2`, the largest
    // array index formed is `(ny - 1) * stride + (nz - 1) * ny * stride +
    // 2 * (nxo2 - 1) + 1 <= array_len - stride + stride - 1 < array_len`
    // (`2 * nxo2 <= stride`), and the largest work index is
    // `2 * ((nz - 1) + (nxo2 - 1) * nz) + 1 = work_len - 1`; both lengths are
    // asserted above.  Loop bounds are the source's, each counter is
    // non-negative, and the element copies are the same in the same order.
    let (nyu, nzu, nxo2u, strideu) = (
        ny.max(0) as usize,
        nz.max(0) as usize,
        nxo2.max(0) as usize,
        stride as usize,
    );
    for y in 0..nyu {
        for z in 0..nzu {
            let base = y * strideu + z * nyu * strideu;
            for x in 0..nxo2u {
                unsafe {
                    *brray.get_unchecked_mut(2 * (z + x * nzu)) =
                        *array.get_unchecked(base + 2 * x);
                    *brray.get_unchecked_mut(2 * (z + x * nzu) + 1) =
                        *array.get_unchecked(base + 2 * x + 1);
                }
            }
        }
        odfft::odfft_c(brray, nz, nxo2, oddir);
        for z in 0..nzu {
            let base = y * strideu + z * nyu * strideu;
            for x in 0..nxo2u {
                unsafe {
                    *array.get_unchecked_mut(base + 2 * x) =
                        *brray.get_unchecked(2 * (z + x * nzu));
                    *array.get_unchecked_mut(base + 2 * x + 1) =
                        *brray.get_unchecked(2 * (z + x * nzu) + 1);
                }
            }
        }
    }
    if idir != 0 {
        for plane in array[..array_len].chunks_mut(plane_len) {
            todfft::todfft_c(plane, nx, ny, back);
        }
    }
}

/// C `thrdfftc`.
///
/// The C entry point receives dimensions by value, unlike the Fortran-facing
/// `thrdfft` form.  Rust values already have that calling convention.
pub fn thrdfftc(array: &mut [f32], brray: &mut [f32], nx: i32, ny: i32, nz: i32, idir: i32) {
    thrdfft(array, brray, nx, ny, nz, idir);
}

#[cfg(test)]
mod tests {
    use super::thrdfft;

    #[test]
    fn three_dimensional_round_trip_preserves_padded_real_volume() {
        let (nx, ny, nz) = (4_i32, 2_i32, 2_i32);
        let stride = (nx + 2) as usize;
        let mut values = vec![0.0_f32; stride * ny as usize * nz as usize];
        for z in 0..nz as usize {
            for y in 0..ny as usize {
                for x in 0..nx as usize {
                    values[x + stride * (y + ny as usize * z)] = (x + 3 * y + 7 * z) as f32;
                }
            }
        }
        let expected = values.clone();
        let mut work = vec![0.0_f32; 2 * nz as usize * ((nx + 2) / 2) as usize];
        thrdfft(&mut values, &mut work, nx, ny, nz, 0);
        thrdfft(&mut values, &mut work, nx, ny, nz, -1);
        for z in 0..nz as usize {
            for y in 0..ny as usize {
                for x in 0..nx as usize {
                    let index = x + stride * (y + ny as usize * z);
                    assert!((values[index] - expected[index]).abs() < 1.0e-4);
                }
            }
        }
    }
}
