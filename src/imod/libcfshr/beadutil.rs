//! Translation of `IMOD/libcfshr/beadutil.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::percentile::percentile_float;

/// C `makeModelBead`.
pub unsafe fn make_model_bead(box_size: i32, bead_size: f32, array: *mut f32) {
    unsafe {
        let icen = box_size / 2;
        let ndiv = 10_i32;
        let fcen = (box_size as f32 - 1. / ndiv as f32) / 2.;
        let radsq = bead_size * bead_size / 4.;
        for iy in 0..=icen {
            for ix in 0..=icen {
                let mut height = 0.;
                for iyd in 0..ndiv {
                    for ixd in 0..ndiv {
                        let dely = iy as f32 + iyd as f32 / ndiv as f32 - fcen;
                        let delx = ix as f32 + ixd as f32 / ndiv as f32 - fcen;
                        let distsq = delx * delx + dely * dely;
                        if distsq < radsq {
                            height += (radsq - distsq).sqrt();
                        }
                    }
                }
                height /= -(ndiv * ndiv) as f32;
                *array.add((ix + iy * box_size) as usize) = height;
                *array.add((box_size - 1 - ix + (box_size - 1 - iy) * box_size) as usize) = height;
                *array.add((ix + (box_size - 1 - iy) * box_size) as usize) = height;
                *array.add((box_size - 1 - ix + iy * box_size) as usize) = height;
            }
        }
    }
}

/// Fortran wrapper C `makemodelbead`.
pub unsafe fn makemodelbead(box_size: *mut i32, bead_size: *mut f32, array: *mut f32) {
    unsafe { make_model_bead(*box_size, *bead_size, array) }
}

/// C `beadIntegral`.
pub unsafe fn bead_integral(
    array: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    r_center: f32,
    r_inner: f32,
    r_outer: f32,
    xcen: f32,
    ycen: f32,
    cenmean: *mut f32,
    annmean: *mut f32,
    temp: *mut f32,
    ann_pct: f32,
    median: *mut f32,
) -> f64 {
    unsafe {
        let xpcen = xcen - 0.5;
        let ypcen = ycen - 0.5;
        let rcensq = r_center * r_center;
        let rinsq = r_inner * r_inner;
        let routsq = r_outer * r_outer;
        let mut ncen = 0_i32;
        let mut nann = 0_i32;
        let mut censum = 0_f64;
        let mut annsum = 0_f64;
        let ixcen = (xpcen + 0.5).floor() as i32;
        let iycen = (ypcen + 0.5).floor() as i32;
        let iradout = (r_outer + 1.5) as i32;
        for iy in iycen - iradout..=iycen + iradout {
            if iy < 0 || iy >= ny {
                continue;
            }
            let dy = iy as f32 - ypcen;
            let idx = ((0_f64.max((routsq - dy * dy) as f64)).sqrt() + 1.5) as i32;
            for ix in ixcen - idx..=ixcen + idx {
                if ix < 0 || ix >= nx {
                    continue;
                }
                let dx = ix as f32 - xpcen;
                let radsq = dy * dy + dx * dx;
                let value = *array.add((ix + iy * nxdim) as usize);
                if radsq <= rcensq {
                    ncen += 1;
                    censum += value as f64;
                } else if radsq <= routsq && radsq >= rinsq {
                    if ann_pct > 0. {
                        *temp.add(nann as usize) = value;
                    }
                    nann += 1;
                    annsum += value as f64;
                }
            }
        }
        if nann == 0 || ncen == 0 {
            return 0.;
        }
        *annmean = (annsum / nann as f64) as f32;
        *cenmean = (censum / ncen as f64) as f32;
        if ann_pct > 0. {
            *median = percentile_float(
                (ann_pct * nann as f32 + 1.) as i32,
                core::slice::from_raw_parts_mut(temp, nann as usize),
                nann,
            );
        }
        if ann_pct < 0. && !median.is_null() {
            *median = ncen as f32;
        }
        censum / ncen as f64 - *annmean as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_bead_has_source_sign_and_symmetry() {
        let mut array = vec![0.; 25];
        unsafe {
            make_model_bead(5, 3., array.as_mut_ptr());
        }
        assert!(array[12] < 0.);
        assert_eq!(array[0], array[24]);
    }
    #[test]
    fn integral_is_center_minus_annulus() {
        let mut array = vec![1.; 49];
        array[3 + 3 * 7] = 9.;
        let mut cen = 0.;
        let mut ann = 0.;
        let value = unsafe {
            bead_integral(
                array.as_mut_ptr(),
                7,
                7,
                7,
                0.6,
                1.0,
                2.0,
                3.5,
                3.5,
                &mut cen,
                &mut ann,
                core::ptr::null_mut(),
                0.,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(cen, 9.);
        assert_eq!(ann, 1.);
        assert_eq!(value, 8.);
    }

    #[test]
    fn integral_uses_source_percentile_selection() {
        let mut array = vec![1.; 49];
        array[3 + 3 * 7] = 9.;
        array[2 + 2 * 7] = 3.;
        array[4 + 2 * 7] = 7.;
        let mut temp = [0.; 49];
        let mut center = 0.;
        let mut annulus = 0.;
        let mut percentile = 0.;
        unsafe {
            bead_integral(
                array.as_mut_ptr(),
                7,
                7,
                7,
                0.6,
                1.0,
                2.0,
                3.5,
                3.5,
                &mut center,
                &mut annulus,
                temp.as_mut_ptr(),
                0.5,
                &mut percentile,
            );
        }
        let annular_count = temp
            .iter()
            .position(|value| *value == 0.)
            .unwrap_or(temp.len());
        let mut source_selected = temp[..annular_count].to_vec();
        let expected = crate::imod::libcfshr::percentile::percentile_float(
            (0.5 * annular_count as f32 + 1.) as i32,
            &mut source_selected,
            annular_count as i32,
        );
        assert_eq!(percentile, expected);
    }

    #[test]
    fn integral_reports_center_count_for_negative_percentile_and_honors_stride() {
        // The useful 5-pixel rows are separated by three padding values.
        let mut padded = vec![-99.; 8 * 5];
        for y in 0..5 {
            for x in 0..5 {
                padded[x + y * 8] = 2.;
            }
        }
        padded[2 + 2 * 8] = 10.;
        let (mut center, mut annulus, mut count) = (0., 0., -1.);
        let result = unsafe {
            bead_integral(
                padded.as_mut_ptr(),
                8,
                5,
                5,
                0.6,
                1.,
                2.,
                2.5,
                2.5,
                &mut center,
                &mut annulus,
                core::ptr::null_mut(),
                -1.,
                &mut count,
            )
        };
        assert_eq!((center, annulus, result), (10., 2., 8.));
        assert_eq!(count, 1.);
    }
}
