//! Translation of `IMOD/flib/subrs/hvem/taperinvol.f90`.
//!
//! Callers pass `array` and `brray` as the *same* Fortran array more than
//! once (`corrsearch3d.f90`'s `loadExtractProcess` tapers a patch into the
//! place it already occupies when no kernel smoothing follows), which is
//! legal Fortran sequence association and is why the box copy runs
//! backwards.  So the unit takes one buffer and the two starting element
//! offsets; distinct arrays are two disjoint ranges of it.

/// Original `taperInVol` (`taperinvol.f90:8`).
///
/// TAPERINVOL pads a volume, dimensions NXBOX by NYBOX by NZBOX in ARRAY
/// (`buf[ind_array..]`), into the center of a larger array.  The padded image
/// in BRRAY (`buf[ind_brray..]`) will have size NX by NY by NZ while BRRAY is
/// assumed to be dimensioned NXDIM by NY by NZ.  The values of the image at
/// its edge will be tapered down to the mean value at the edge, over a width
/// of the original image area equal to NXTAP, NYTAP, and NZTAP.
#[allow(clippy::too_many_arguments)]
pub fn taper_in_vol(
    buf: &mut [f32],
    ind_array: usize,
    nx_box: i32,
    ny_box: i32,
    nz_box: i32,
    ind_brray: usize,
    nx_dim: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    nx_taper: i32,
    ny_taper: i32,
    nz_taper: i32,
) {
    // `array(ix, iy, iz)` and `brray(ix, iy, iz)`, 1-based column major.
    let ia = |ix: i32, iy: i32, iz: i32| -> usize {
        ind_array + ((ix - 1) + nx_box * ((iy - 1) + ny_box * (iz - 1))) as usize
    };
    let ib = |ix: i32, iy: i32, iz: i32| -> usize {
        ind_brray + ((ix - 1) + nx_dim * ((iy - 1) + ny * (iz - 1))) as usize
    };
    let mut ix1: i32;
    let mut ix2: i32;
    let mut iy1: i32;
    let mut iy2: i32;
    let mut iz1: i32;
    let mut iz2: i32;
    //
    // Get the edge mean.  Each `sum` of a `real*4` section accumulates in
    // `real*4` in array-element order, and the six are added left to right.
    let mut s1 = 0.0_f32;
    for iy in 1..=ny_box {
        for ix in 1..=nx_box {
            s1 += buf[ia(ix, iy, 1)];
        }
    }
    let mut s2 = 0.0_f32;
    for iy in 1..=ny_box {
        for ix in 1..=nx_box {
            s2 += buf[ia(ix, iy, nz_box)];
        }
    }
    let mut s3 = 0.0_f32;
    for iz in 2..=nz_box - 1 {
        for ix in 1..=nx_box {
            s3 += buf[ia(ix, 1, iz)];
        }
    }
    let mut s4 = 0.0_f32;
    for iz in 2..=nz_box - 1 {
        for ix in 1..=nx_box {
            s4 += buf[ia(ix, ny_box, iz)];
        }
    }
    let mut s5 = 0.0_f32;
    for iz in 2..=nz_box - 1 {
        for iy in 2..=ny_box - 1 {
            s5 += buf[ia(1, iy, iz)];
        }
    }
    let mut s6 = 0.0_f32;
    for iz in 2..=nz_box - 1 {
        for iy in 2..=ny_box - 1 {
            s6 += buf[ia(nx_box, iy, iz)];
        }
    }
    let arsum: f32 = s1 + s2 + s3 + s4 + s5 + s6;
    let nsum = 2 * nx_box * ny_box + (nz_box - 2) * (2 * (nx_box + ny_box - 2));
    let dmean: f32 = arsum / nsum as f32;
    //
    let ix_low = (nx - nx_box) / 2;
    let ix_high = ix_low + nx_box;
    let iy_low = (ny - ny_box) / 2;
    let iy_high = iy_low + ny_box;
    let iz_low = (nz - nz_box) / 2;
    let iz_high = iz_low + nz_box;
    // The source copies element by element from the last backwards.  Every
    // destination index is at or above its source and the mapping is
    // monotonic, so copying whole rows (`copy_within` is a `memmove`) in the
    // same descending row order reads every source value before anything
    // overwrites it, exactly as the element loop does.
    for iz in (1..=nz_box).rev() {
        for iy in (1..=ny_box).rev() {
            let src = ia(1, iy, iz);
            buf.copy_within(
                src..src + nx_box as usize,
                ib(1 + ix_low, iy + iy_low, iz + iz_low),
            );
        }
    }

    let mut nz_top = nz + 1;
    if nx_box != nx || ny_box != ny || nz_box != nz {
        for iz in iz_low + 1..=iz_high {
            let mut nx_top = nx + 1;
            let mut ny_top = ny + 1;
            nz_top = nz + 1;

            //
            // if there is a mismatch between left and right, add a column on
            // right; similarly for bottom versus top, add a row on top
            //
            if nx - ix_high > ix_low {
                nx_top = nx;
                for iy in 1..=ny {
                    buf[ib(nx, iy, iz)] = dmean;
                }
            }
            if ny - iy_high > iy_low {
                ny_top = ny;
                let k = ib(1, ny, iz);
                buf[k..k + nx as usize].fill(dmean);
            }
            //
            for iy in iy_low + 1..=iy_high {
                for ix in 1..=ix_low {
                    buf[ib(ix, iy, iz)] = dmean;
                    buf[ib(nx_top - ix, iy, iz)] = dmean;
                }
            }
            for iy in 1..=iy_low {
                let iy_top_line = ny_top - iy;
                let k = ib(1, iy, iz);
                buf[k..k + nx as usize].fill(dmean);
                let k = ib(1, iy_top_line, iz);
                buf[k..k + nx as usize].fill(dmean);
            }
        }
        //
        if nz - iz_high > iz_low {
            nz_top = nz;
            for iy in 1..=ny {
                let k = ib(1, iy, nz);
                buf[k..k + nx as usize].fill(dmean);
            }
        }
        for iz in 1..=iz_low {
            let iz_top_line = nz_top - iz;
            for iy in 1..=ny {
                let k = ib(1, iy, iz);
                buf[k..k + nx as usize].fill(dmean);
            }
            for iy in 1..=ny {
                let k = ib(1, iy, iz_top_line);
                buf[k..k + nx as usize].fill(dmean);
            }
        }
    }
    //
    // do 8 pixels at once, using symmetry.  if the box was odd in a
    // dimension, deflect the middle pixel out to the edge to keep it from
    // being attenuated twice
    //
    for iz in 1..=(nz_box + 1) / 2 {
        let mut frac_z = 1.0_f32;
        if iz <= nz_taper {
            frac_z = iz as f32 / (nz_taper as f32 + 1.);
        }
        iz1 = iz + iz_low;
        iz2 = iz_high + 1 - iz;
        if iz2 == iz1 {
            iz2 = nz;
        }
        for iy in 1..=(ny_box + 1) / 2 {
            let mut frac_y = 1.0_f32;
            if iy <= ny_taper {
                frac_y = iy as f32 / (ny_taper as f32 + 1.);
            }
            iy1 = iy + iy_low;
            iy2 = iy_high + 1 - iy;
            if iy2 == iy1 {
                iy2 = ny;
            }
            let mut nx_limit = (nx_box + 1) / 2;
            if frac_y == 1. && frac_z == 1. {
                nx_limit = nx_taper;
            }
            for ix in 1..=nx_limit {
                let mut frac_x = 1.0_f32;
                if ix <= nx_taper {
                    frac_x = ix as f32 / (nx_taper as f32 + 1.);
                }
                // `min(fracX, fracY, fracZ)`: all three are finite here.
                let fmin = frac_x.min(frac_y).min(frac_z);
                ix1 = ix + ix_low;
                ix2 = ix_high + 1 - ix;
                if ix2 == ix1 {
                    ix2 = nx;
                }
                if fmin < 1. {
                    for &(jx, jy, jz) in &[
                        (ix1, iy1, iz1),
                        (ix1, iy2, iz1),
                        (ix2, iy1, iz1),
                        (ix2, iy2, iz1),
                        (ix1, iy1, iz2),
                        (ix1, iy2, iz2),
                        (ix2, iy1, iz2),
                        (ix2, iy2, iz2),
                    ] {
                        let k = ib(jx, jy, jz);
                        buf[k] = fmin * (buf[k] - dmean) + dmean;
                    }
                }
            }
        }
    }
}
