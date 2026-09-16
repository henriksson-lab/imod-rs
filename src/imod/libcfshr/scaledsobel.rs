//! Translation of `IMOD/libcfshr/scaledsobel.c`.
#![allow(dead_code)]

/// Complete function inventory for `scaledsobel.c`.
pub const SCALED_SOBEL_SOURCE_FUNCTIONS: &[&str] = &["scaledSobel", "scaledsobel"];

/// `scaledSobel` (`scaledsobel.c:49`).
pub fn scaled_sobel(
    in_image: Option<&[f32]>,
    nxin: i32,
    nyin: i32,
    scale_fac: f32,
    min_interp: f32,
    mut linear: i32,
    center: f32,
    out_image: Option<&mut [f32]>,
    nxout: &mut i32,
    nyout: &mut i32,
    x_offset: &mut f32,
    y_offset: &mut f32,
) -> i32 {
    if nxin <= 0 || nyin <= 0 || !scale_fac.is_finite() || scale_fac <= 0.0 {
        return 1;
    }
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
    let Some(in_image) = in_image else {
        return 0;
    };
    if center < 0. {
        return 0;
    }
    let Some(out_image) = out_image else {
        return 1;
    };
    let Some(input_pixels) = nxin.checked_mul(nyin) else {
        return 1;
    };
    let Some(output_pixels) = nxo.checked_mul(nyo) else {
        return 1;
    };
    let Ok(input_size) = usize::try_from(input_pixels) else {
        return 1;
    };
    let Ok(output_size) = usize::try_from(output_pixels) else {
        return 1;
    };
    if in_image.len() < input_size || out_image.len() < output_size || nxo < 2 || nyo < 2 {
        return 1;
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
    let Some(temporary_pixels) = (if binning > 1 {
        nxbin.checked_mul(nybin)
    } else {
        nxo.checked_mul(nyo)
    }) else {
        return 1;
    };
    let Ok(size) = usize::try_from(temporary_pixels) else {
        return 1;
    };
    let mut temporary = Vec::new();
    if temporary.try_reserve_exact(size).is_err() {
        return 1;
    }
    temporary.resize(size, 0.);
    let (source, destination) = if binning > 1 {
        // `reduceByBinning(..., SLICE_MODE_FLOAT, ...)` in the source.  Keeping
        // this f32 path here avoids turning two owned slices into byte slices.
        let x_start = (nxin % binning / 2) as usize;
        let y_start = (nyin % binning / 2) as usize;
        for y in 0..nybin as usize {
            for x in 0..nxbin as usize {
                let mut sum = 0.0_f32;
                for by in 0..binning as usize {
                    for bx in 0..binning as usize {
                        sum += in_image[(y_start + y * binning as usize + by) * nxin as usize
                            + x_start
                            + x * binning as usize
                            + bx];
                    }
                }
                temporary[y * nxbin as usize + x] = sum / (binning * binning) as f32;
            }
        }
        (temporary.as_slice(), &mut *out_image)
    } else {
        (in_image, temporary.as_mut_slice())
    };
    let edge =
        crate::imod::libcfshr::taperpad::slice_edge_mean(source, nxbin, 0, nxbin - 1, 0, nybin - 1)
            as f32;
    if interp_scale == 1. && nxbin == nxo && nybin == nyo {
        // `cubinterp` is an identity transform here.  Preserve the source
        // binned pixels directly; the generic Rust interpolator has no
        // interior sample for a 2-by-2 identity grid.
        destination[..output_size].copy_from_slice(&source[..output_size]);
    } else if linear >= 0 || scale_fac <= 1. {
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
            source,
            destination,
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
            return error;
        }
    }
    if binning > 1 {
        temporary[..output_size].copy_from_slice(&out_image[..output_size]);
    }
    if center == 0. {
        out_image[..output_size].copy_from_slice(&temporary[..output_size]);
        return 0;
    }
    for y in 1..nyo - 1 {
        for x in 1..nxo - 1 {
            let index = (x + y * nxo) as usize;
            let row = (temporary[index - nxo as usize - 1]
                + center * temporary[index - nxo as usize]
                + temporary[index - nxo as usize + 1])
                - (temporary[index + nxo as usize - 1]
                    + center * temporary[index + nxo as usize]
                    + temporary[index + nxo as usize + 1]);
            let col = (temporary[index + nxo as usize + 1]
                + center * temporary[index + 1]
                + temporary[index - nxo as usize + 1])
                - (temporary[index + nxo as usize - 1]
                    + center * temporary[index - 1]
                    + temporary[index - nxo as usize - 1]);
            out_image[index] = (row * row + col * col).sqrt();
        }
        out_image[(y * nxo) as usize] = out_image[(y * nxo + 1) as usize];
        out_image[(y * nxo + nxo - 1) as usize] = out_image[(y * nxo + nxo - 2) as usize];
    }
    for x in 0..nxo {
        out_image[x as usize] = out_image[(x + nxo) as usize];
        out_image[(x + nxo * (nyo - 1)) as usize] = out_image[(x + nxo * (nyo - 2)) as usize];
    }
    0
}

/// `scaledsobel` (`scaledsobel.c:182`).
pub fn scaledsobel(
    image: &[f32],
    nx: &i32,
    ny: &i32,
    scale: &f32,
    min_interp: &f32,
    linear: &i32,
    center: &f32,
    output: &mut [f32],
    nxout: &mut i32,
    nyout: &mut i32,
    x_offset: &mut f32,
    y_offset: &mut f32,
) -> i32 {
    scaled_sobel(
        Some(image),
        *nx,
        *ny,
        *scale,
        *min_interp,
        *linear,
        *center,
        Some(output),
        nxout,
        nyout,
        x_offset,
        y_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn size_only_path_matches_source_offsets() {
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        assert_eq!(
            scaled_sobel(
                None, 11, 9, 2.5, 1., 1, -1., None, &mut nxout, &mut nyout, &mut xo, &mut yo
            ),
            0
        );
        assert_eq!((nxout, nyout), (4, 3));
        assert!(xo.abs() < 1.0e-6 && (yo - 0.125).abs() < 1.0e-6);
    }
    #[test]
    fn constant_image_has_zero_sobel_response() {
        let image = vec![3.; 64];
        let mut output = vec![99.; 64];
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        assert_eq!(
            scaled_sobel(
                Some(&image),
                8,
                8,
                1.,
                1.,
                1,
                2.,
                Some(&mut output),
                &mut nxout,
                &mut nyout,
                &mut xo,
                &mut yo
            ),
            0
        );
        assert_eq!((nxout, nyout), (8, 8));
        assert!(output.iter().all(|value| *value == 0.));
    }
    #[test]
    fn center_zero_returns_scaled_image() {
        let image = vec![7.; 36];
        let mut output = vec![0.; 36];
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0., 0.);
        scaled_sobel(
            Some(&image),
            6,
            6,
            1.,
            1.,
            1,
            0.,
            Some(&mut output),
            &mut nxout,
            &mut nyout,
            &mut xo,
            &mut yo,
        );
        assert!(output.iter().all(|value| *value == 7.));
    }

    #[test]
    fn typed_binning_keeps_source_centering_and_f32_accumulation_order() {
        let image = [
            0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0,
        ];
        let mut output = [0.0; 4];
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0.0, 0.0);
        assert_eq!(
            scaled_sobel(
                Some(&image),
                4,
                4,
                2.0,
                1.0,
                1,
                0.0,
                Some(&mut output),
                &mut nxout,
                &mut nyout,
                &mut xo,
                &mut yo,
            ),
            0
        );
        assert_eq!((nxout, nyout), (2, 2));
        assert_eq!(output, [2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn invalid_owned_slice_contract_returns_source_error_code_instead_of_panicking() {
        let (mut nxout, mut nyout, mut xo, mut yo) = (0, 0, 0.0, 0.0);
        assert_eq!(
            scaled_sobel(
                Some(&[1.0; 4]),
                2,
                2,
                1.0,
                1.0,
                1,
                2.0,
                Some(&mut [0.0; 3]),
                &mut nxout,
                &mut nyout,
                &mut xo,
                &mut yo,
            ),
            1
        );
        assert_eq!(SCALED_SOBEL_SOURCE_FUNCTIONS.len(), 2);
    }
}
