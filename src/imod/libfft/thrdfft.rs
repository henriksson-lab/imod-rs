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
    for y in 0..ny {
        for z in 0..nz {
            let base = y * stride + z * ny * stride;
            for x in 0..nxo2 {
                brray[(2 * (z + x * nz)) as usize] = array[(base + 2 * x) as usize];
                brray[(2 * (z + x * nz) + 1) as usize] = array[(base + 2 * x + 1) as usize];
            }
        }
        odfft::odfft_c(brray, nz, nxo2, oddir);
        for z in 0..nz {
            let base = y * stride + z * ny * stride;
            for x in 0..nxo2 {
                array[(base + 2 * x) as usize] = brray[(2 * (z + x * nz)) as usize];
                array[(base + 2 * x + 1) as usize] = brray[(2 * (z + x * nz) + 1) as usize];
            }
        }
    }
    if idir != 0 {
        for plane in array[..array_len].chunks_mut(plane_len) {
            todfft::todfft_c(plane, nx, ny, back);
        }
    }
}

/// C `thrdfftc` ABI boundary.
///
/// # Safety
///
/// `array` must point to `(nx + 2) * ny * nz` floats and `brray` to
/// `2 * nz * ((nx + 2) / 2)` floats.  Neither region may overlap.
pub unsafe fn thrdfft_c(array: *mut f32, brray: *mut f32, nx: i32, ny: i32, nz: i32, idir: i32) {
    let array_len = ((nx + 2) * ny * nz) as usize;
    let work_len = (2 * nz * ((nx + 2) / 2)) as usize;
    let array = unsafe { core::slice::from_raw_parts_mut(array, array_len) };
    let brray = unsafe { core::slice::from_raw_parts_mut(brray, work_len) };
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
