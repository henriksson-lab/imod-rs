//! Translation of `IMOD/libfft/thrdfft.c`.

use super::{odfft, todfft};

/// C `thrdfft`.
pub unsafe fn thrdfft(
    array: *mut f32,
    brray: *mut f32,
    nxp: *mut i32,
    nyp: *mut i32,
    nzp: *mut i32,
    idirp: *mut i32,
) {
    unsafe {
        let nx = *nxp;
        let ny = *nyp;
        let nz = *nzp;
        let idir = *idirp;
        let stride = nx + 2;
        let nxo2 = (nx + 2) / 2;
        let oddir = if idir != 0 { -2 } else { -1 };
        let back = 1;
        if idir == 0 {
            for z in 0..nz {
                todfft(array.add((stride * ny * z) as usize), nxp, nyp, idirp);
            }
        }
        for y in 0..ny {
            for z in 0..nz {
                let base = y * stride + z * ny * stride;
                for x in 0..nxo2 {
                    *brray.add((2 * (z + x * nz)) as usize) = *array.add((base + 2 * x) as usize);
                    *brray.add((2 * (z + x * nz) + 1) as usize) =
                        *array.add((base + 2 * x + 1) as usize);
                }
            }
            let mut local_nz = nz;
            let mut local_count = nxo2;
            let mut local_dir = oddir;
            odfft(brray, &mut local_nz, &mut local_count, &mut local_dir);
            for z in 0..nz {
                let base = y * stride + z * ny * stride;
                for x in 0..nxo2 {
                    *array.add((base + 2 * x) as usize) = *brray.add((2 * (z + x * nz)) as usize);
                    *array.add((base + 2 * x + 1) as usize) =
                        *brray.add((2 * (z + x * nz) + 1) as usize);
                }
            }
        }
        if idir != 0 {
            for z in 0..nz {
                let mut local_back = back;
                todfft(
                    array.add((stride * ny * z) as usize),
                    nxp,
                    nyp,
                    &mut local_back,
                );
            }
        }
    }
}

/// C `thrdfftc`.
pub unsafe fn thrdfft_c(array: *mut f32, brray: *mut f32, nx: i32, ny: i32, nz: i32, idir: i32) {
    unsafe {
        let mut nx = nx;
        let mut ny = ny;
        let mut nz = nz;
        let mut idir = idir;
        thrdfft(array, brray, &mut nx, &mut ny, &mut nz, &mut idir);
    }
}
