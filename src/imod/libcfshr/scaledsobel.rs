//! Translation of `IMOD/libcfshr/scaledsobel.c`.
#![allow(dead_code)]

/// `scaledSobel` (`scaledsobel.c:49`).
pub unsafe fn scaled_sobel(
    in_image: *mut f32,
    nxin: i32,
    nyin: i32,
    scale_fac: f32,
    min_interp: f32,
    mut linear: i32,
    center: f32,
    out_image: *mut f32,
    nxout: *mut i32,
    nyout: *mut i32,
    x_offset: *mut f32,
    y_offset: *mut f32,
) -> i32 {
    unsafe {
        let mut interp_scale = scale_fac;
        let mut binning = 1;
        if linear < 0 && scale_fac < 1.1 {
            linear = 1;
        }
        if linear >= 0 {
            for factor in 2..100 {
                let scale = scale_fac / factor as f32;
                if scale < 1. || scale < min_interp {
                    break;
                }
                interp_scale = scale;
                binning = factor;
            }
        }
        let mut nxbin = nxin / binning;
        let mut nybin = nyin / binning;
        *x_offset = ((nxin % binning) / 2) as f32;
        *y_offset = ((nyin % binning) / 2) as f32;
        let nxo = (nxbin as f32 / interp_scale) as i32;
        let nyo = (nybin as f32 / interp_scale) as i32;
        *x_offset += (nxbin as f32 - interp_scale * nxo as f32) / 2.;
        *y_offset += (nybin as f32 - interp_scale * nyo as f32) / 2.;
        *nxout = nxo;
        *nyout = nyo;
        if in_image.is_null() || center < 0. {
            return 0;
        }
        if linear < 0 {
            let mut width = 0;
            let error = crate::imod::libcfshr::zoomdown::select_zoom_filter(
                4,
                1. / interp_scale as f64,
                &mut width,
            );
            if error != 0 {
                return error;
            }
        }
        let size = if binning > 1 {
            nxbin * nybin
        } else {
            nxo * nyo
        };
        let temporary = libc::malloc(size as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if temporary.is_null() {
            return 1;
        }
        let (source, destination) = if binning > 1 {
            let error = crate::imod::libcfshr::reduce_by_binning::reduce_by_binning(
                core::slice::from_raw_parts(
                    in_image.cast::<u8>(),
                    (nxin as usize * nyin as usize) * 4,
                ),
                crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT,
                nxin,
                nyin,
                binning,
                core::slice::from_raw_parts_mut(temporary.cast::<u8>(), size as usize * 4),
                0,
                &mut nxbin,
                &mut nybin,
            );
            if error != 0 {
                libc::free(temporary.cast());
                return error;
            }
            (temporary, out_image)
        } else {
            (in_image, temporary)
        };
        let edge = crate::imod::libcfshr::taperpad::slice_edge_mean(
            core::slice::from_raw_parts(source, (nxbin * nybin) as usize),
            nxbin,
            0,
            nxbin - 1,
            0,
            nybin - 1,
        ) as f32;
        if linear >= 0 || scale_fac <= 1. {
            let matrix = [[1. / interp_scale, 0.], [0., 1. / interp_scale]];
            crate::imod::libcfshr::cubinterp::cubinterp(
                source,
                destination,
                nxbin,
                nybin,
                nxo,
                nyo,
                &matrix,
                nxbin as f32 / 2.,
                nybin as f32 / 2.,
                0.,
                0.,
                1.,
                edge,
                linear,
            );
        } else {
            let error = crate::imod::libcfshr::zoomdown::zoom_filt_interp(
                core::slice::from_raw_parts(source, (nxbin * nybin) as usize),
                core::slice::from_raw_parts_mut(destination, (nxo * nyo) as usize),
                nxbin,
                nybin,
                nxo,
                nyo,
                nxbin as f32 / 2.,
                nybin as f32 / 2.,
                0.,
                0.,
                edge,
            );
            if error != 0 {
                libc::free(temporary.cast());
                return error;
            }
        }
        if binning > 1 {
            for index in 0..nxo * nyo {
                *temporary.add(index as usize) = *out_image.add(index as usize);
            }
        }
        if center == 0. {
            for index in 0..nxo * nyo {
                *out_image.add(index as usize) = *temporary.add(index as usize);
            }
            libc::free(temporary.cast());
            return 0;
        }
        for y in 1..nyo - 1 {
            for x in 1..nxo - 1 {
                let index = x + y * nxo;
                let row = (*temporary.add((index - nxo - 1) as usize)
                    + center * *temporary.add((index - nxo) as usize)
                    + *temporary.add((index - nxo + 1) as usize))
                    - (*temporary.add((index + nxo - 1) as usize)
                        + center * *temporary.add((index + nxo) as usize)
                        + *temporary.add((index + nxo + 1) as usize));
                let col = (*temporary.add((index + nxo + 1) as usize)
                    + center * *temporary.add((index + 1) as usize)
                    + *temporary.add((index - nxo + 1) as usize))
                    - (*temporary.add((index + nxo - 1) as usize)
                        + center * *temporary.add((index - 1) as usize)
                        + *temporary.add((index - nxo - 1) as usize));
                *out_image.add(index as usize) = (row * row + col * col).sqrt();
            }
            *out_image.add((y * nxo) as usize) = *out_image.add((y * nxo + 1) as usize);
            *out_image.add((y * nxo + nxo - 1) as usize) =
                *out_image.add((y * nxo + nxo - 2) as usize);
        }
        for x in 0..nxo {
            *out_image.add(x as usize) = *out_image.add((x + nxo) as usize);
            *out_image.add((x + nxo * (nyo - 1)) as usize) =
                *out_image.add((x + nxo * (nyo - 2)) as usize);
        }
        libc::free(temporary.cast());
        0
    }
}

/// `scaledsobel` (`scaledsobel.c:182`).
pub unsafe fn scaledsobel(
    image: *mut f32,
    nx: *const i32,
    ny: *const i32,
    scale: *const f32,
    min_interp: *const f32,
    linear: *const i32,
    center: *const f32,
    output: *mut f32,
    nxout: *mut i32,
    nyout: *mut i32,
    x_offset: *mut f32,
    y_offset: *mut f32,
) -> i32 {
    unsafe {
        scaled_sobel(
            image,
            *nx,
            *ny,
            *scale,
            *min_interp,
            *linear,
            *center,
            output,
            nxout,
            nyout,
            x_offset,
            y_offset,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn size_only_path_matches_source_offsets() {
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        unsafe {
            assert_eq!(
                scaled_sobel(
                    core::ptr::null_mut(),
                    11,
                    9,
                    2.5,
                    1.,
                    1,
                    -1.,
                    core::ptr::null_mut(),
                    &mut nxout,
                    &mut nyout,
                    &mut xo,
                    &mut yo
                ),
                0
            )
        };
        assert_eq!((nxout, nyout), (4, 3));
        assert!(xo.abs() < 1.0e-6 && (yo - 0.125).abs() < 1.0e-6);
    }
    #[test]
    fn constant_image_has_zero_sobel_response() {
        let image = vec![3.; 64];
        let mut output = vec![99.; 64];
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        unsafe {
            assert_eq!(
                scaled_sobel(
                    image.as_ptr().cast_mut(),
                    8,
                    8,
                    1.,
                    1.,
                    1,
                    2.,
                    output.as_mut_ptr(),
                    &mut nxout,
                    &mut nyout,
                    &mut xo,
                    &mut yo
                ),
                0
            )
        };
        assert_eq!((nxout, nyout), (8, 8));
        assert!(output.iter().all(|value| *value == 0.));
    }
    #[test]
    fn center_zero_returns_scaled_image() {
        let image = vec![7.; 36];
        let mut output = vec![0.; 36];
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        unsafe {
            scaled_sobel(
                image.as_ptr().cast_mut(),
                6,
                6,
                1.,
                1.,
                1,
                0.,
                output.as_mut_ptr(),
                &mut nxout,
                &mut nyout,
                &mut xo,
                &mut yo,
            )
        };
        assert!(output.iter().all(|value| *value == 7.));
    }
}
