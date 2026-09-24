//! Translation of `IMOD/libcfshr/scaledsobel.c`.

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
    // Not in the source: a non-positive or non-finite `scaleFac` makes
    // `(int)(nxbin / interpScale)` below a float-to-int overflow, which is UB in
    // C; refuse it rather than reproduce one compiler's conversion.
    if !scale_fac.is_finite() || scale_fac <= 0.0 {
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
    // `*xOffset += (nxbin - interpScale * nxo) / 2.;` -- a float difference
    // divided by a double `2.`, added to the float in double, then narrowed.
    *x_offset = (*x_offset as f64 + (nxbin as f32 - interp_scale * nxo as f32) as f64 / 2.) as f32;
    *y_offset = (*y_offset as f64 + (nybin as f32 - interp_scale * nyo as f32) as f64 / 2.) as f32;
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
    // Not in the source: the C indexes past its buffers when they are short,
    // and its edge copies (`scaledsobel.c:167-177`) read `outImage[j*nxo + 1]`
    // and `outImage[i + nxo * (nyo - 2)]` out of bounds when a filtered output
    // is under 1 by 2 pixels, and `sliceEdgeMean` reads an empty image.
    // Refuse those rather than overrun.
    if nxin <= 0
        || nyin <= 0
        || in_image.len() < input_size
        || out_image.len() < output_size
        || (center != 0. && (nxo < 1 || nyo < 2))
    {
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
        // `scaledsobel.c:106`: `reduceByBinning(inImage, SLICE_MODE_FLOAT, nxin,
        // nyin, binning, tmpImage, 0, &nxbin, &nybin)`, return value unused.
        // The routine takes the C's `void *`, so it gets byte views of the two
        // `float` buffers.
        // SAFETY: `f32` has no padding and no invalid bit patterns, `u8` has
        // alignment 1, and each view covers exactly its slice's storage, so
        // reading and writing it as bytes is sound; the two borrows are
        // distinct allocations, as `tmpImage` and `inImage` are in the C.
        let in_bytes = unsafe {
            core::slice::from_raw_parts(
                in_image.as_ptr().cast::<u8>(),
                core::mem::size_of_val(in_image),
            )
        };
        let tmp_bytes = unsafe {
            core::slice::from_raw_parts_mut(
                temporary.as_mut_ptr().cast::<u8>(),
                core::mem::size_of_val(temporary.as_slice()),
            )
        };
        crate::imod::libcfshr::reduce_by_binning::reduce_by_binning(
            in_bytes,
            crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT,
            nxin,
            nyin,
            binning,
            tmp_bytes,
            0,
            &mut nxbin,
            &mut nybin,
        );
        (temporary.as_slice(), &mut *out_image)
    } else {
        (in_image, temporary.as_mut_slice())
    };
    let edge =
        crate::imod::libcfshr::taperpad::slice_edge_mean(source, nxbin, 0, nxbin - 1, 0, nybin - 1)
            as f32;
    // `amat[0][0] = amat[1][1] = 1. / interpScale;` -- a double quotient stored
    // into `float amat[2][2]`; `nxbin / 2.` is likewise a double narrowed to
    // cubinterp's `float` parameter.  There is no identity special case: the
    // source always interpolates, and for `linear > 0` cubinterp fills the
    // last row and column of an identity mapping with `edge`.
    let recip = (1. / interp_scale as f64) as f32;
    if linear >= 0 || scale_fac <= 1. {
        let matrix = [[recip, 0.], [0., recip]];
        crate::imod::libcfshr::cubinterp::cubinterp(
            source,
            destination,
            nxbin,
            nybin,
            nxo,
            nyo,
            &matrix,
            (nxbin as f64 / 2.) as f32,
            (nybin as f64 / 2.) as f32,
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
            (nxbin as f64 / 2.) as f32,
            (nybin as f64 / 2.) as f32,
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
            // `double Sr, Sc` (`scaledsobel.c:57`): each side is a float
            // expression, widened on assignment; `(float)sqrt(Sr*Sr + Sc*Sc)`
            // is the double square root, narrowed on store.
            let row = ((temporary[index - nxo as usize - 1]
                + center * temporary[index - nxo as usize]
                + temporary[index - nxo as usize + 1])
                - (temporary[index + nxo as usize - 1]
                    + center * temporary[index + nxo as usize]
                    + temporary[index + nxo as usize + 1])) as f64;
            let col = ((temporary[index + nxo as usize + 1]
                + center * temporary[index + 1]
                + temporary[index - nxo as usize + 1])
                - (temporary[index + nxo as usize - 1]
                    + center * temporary[index - 1]
                    + temporary[index - nxo as usize - 1])) as f64;
            out_image[index] = (row * row + col * col).sqrt() as f32;
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
    fn binning_then_linear_identity_interp_matches_native() {
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
        // Native `scaledSobel` (reference libcfshr.so) gives `2.5 7.5 7.5 7.5`:
        // linear `cubinterp` on a 2x2 identity grid fills the last row and
        // column with the edge mean, (2.5 + 4.5 + 10.5 + 12.5) / 4.
        assert_eq!(output, [2.5, 7.5, 7.5, 7.5]);
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
    }
}
