//! Translation of `IMOD/libcfshr/cubinterp.c`.
#![allow(dead_code)]

/// `cubinterp` (`cubinterp.c:50`).
///
/// The C routine computes safe cubic spans per line to remove tests from its
/// inner loop.  This direct scalar translation retains the same cubic-versus-
/// quadratic edge decision for every output pixel; OpenMP is intentionally not
/// represented because it does not alter the output image.
pub unsafe fn cubinterp(
    array: *mut f32,
    bray: *mut f32,
    nxa: i32,
    nya: i32,
    nxb: i32,
    nyb: i32,
    amat: *const [[f32; 2]; 2],
    xc: f32,
    yc: f32,
    xt: f32,
    yt: f32,
    scale: f32,
    dmean: f32,
    linear: i32,
) {
    unsafe {
        let matrix = *amat;
        let xcen = nxb as f32 / 2. + xt + 0.5;
        let ycen = nyb as f32 / 2. + yt + 0.5;
        let xco = xc + 0.5;
        let yco = yc + 0.5;
        let denominator = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
        let a11 = matrix[1][1] / denominator;
        let a12 = -matrix[1][0] / denominator;
        let a21 = -matrix[0][1] / denominator;
        let a22 = matrix[0][0] / denominator;

        for iy in 1..=nyb {
            let dyo = iy as f32 - ycen;
            let xbase = a12 * dyo + xco - a11 * xcen;
            let ybase = a22 * dyo + yco - a21 * xcen;
            for ix in 1..=nxb {
                let xp = a11 * ix as f32 + xbase;
                let yp = a21 * ix as f32 + ybase;
                let out = bray.add(((iy - 1) * nxb + ix - 1) as usize);

                // The central C loop is valid precisely for this four-by-four
                // neighbourhood.  It does not multiply this cubic result by scale.
                if linear == 0
                    && xp >= 2.01
                    && xp <= nxa as f32 - 1.01
                    && yp >= 2.01
                    && yp <= nya as f32 - 1.01
                {
                    let ixp = xp as i32;
                    let iyp = yp as i32;
                    let dx = xp - ixp as f32;
                    let dy = yp - iyp as f32;
                    let dxm1 = dx - 1.;
                    let dxdxm1 = dx * dxm1;
                    let fx1 = -dxm1 * dxdxm1;
                    let fx4 = dx * dxdxm1;
                    let fx2 = 1. + dx * dx * (dx - 2.);
                    let fx3 = dx * (1. - dxdxm1);
                    let dym1 = dy - 1.;
                    let dydym1 = dy * dym1;
                    let mut rows = [0.; 4];
                    for row in 0..4 {
                        let base = (iyp - 2 + row) * nxa + ixp - 2;
                        rows[row as usize] = fx1 * *array.add(base as usize)
                            + fx2 * *array.add((base + 1) as usize)
                            + fx3 * *array.add((base + 2) as usize)
                            + fx4 * *array.add((base + 3) as usize);
                    }
                    *out = -dym1 * dydym1 * rows[0]
                        + (1. + dy * dy * (dy - 2.)) * rows[1]
                        + dy * (1. - dydym1) * rows[2]
                        + dy * dydym1 * rows[3];
                    continue;
                }

                if linear == 0 {
                    let ixp = (xp + 0.5).floor() as i32;
                    let iyp = (yp + 0.5).floor() as i32;
                    let mut value = dmean;
                    if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                        let dx = xp - ixp as f32;
                        let dy = yp - iyp as f32;
                        let left = (ixp - 1).max(1);
                        let right = (ixp + 1).min(nxa);
                        let above = (iyp - 1).max(1);
                        let below = (iyp + 1).min(nya);
                        let v2 = *array.add((ixp - 1 + (above - 1) * nxa) as usize);
                        let v4 = *array.add((left - 1 + (iyp - 1) * nxa) as usize);
                        let v5 = *array.add((ixp - 1 + (iyp - 1) * nxa) as usize);
                        let v6 = *array.add((right - 1 + (iyp - 1) * nxa) as usize);
                        let v8 = *array.add((ixp - 1 + (below - 1) * nxa) as usize);
                        let vmax = v2.max(v4).max(v5).max(v6).max(v8);
                        let vmin = v2.min(v4).min(v5).min(v6).min(v8);
                        let a = (v6 + v4) * 0.5 - v5;
                        let b = (v8 + v2) * 0.5 - v5;
                        let c = (v6 - v4) * 0.5;
                        let d = (v8 - v2) * 0.5;
                        value = (scale * (a * dx * dx + b * dy * dy + c * dx + d * dy + v5))
                            .clamp(vmin, vmax);
                    }
                    *out = value;
                } else if linear > 0 {
                    let ixp = xp as i32;
                    let iyp = yp as i32;
                    if ixp >= 1 && ixp < nxa && iyp >= 1 && iyp < nya {
                        let dx = xp - ixp as f32;
                        let dy = yp - iyp as f32;
                        let base = (ixp - 1) + (iyp - 1) * nxa;
                        *out = (1. - dy)
                            * ((1. - dx) * *array.add(base as usize)
                                + dx * *array.add((base + 1) as usize))
                            + dy * ((1. - dx) * *array.add((base + nxa) as usize)
                                + dx * *array.add((base + nxa + 1) as usize));
                    } else {
                        *out = dmean;
                    }
                } else {
                    let ixp = (xp + 0.5).floor() as i32;
                    let iyp = (yp + 0.5).floor() as i32;
                    if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                        *out = *array.add(((ixp - 1) + (iyp - 1) * nxa) as usize);
                    } else {
                        *out = dmean;
                    }
                }
            }
        }
    }
}

/// `cubinterpfwrap` (`cubinterp.c:312`).
pub unsafe fn cubinterpfwrap(
    array: *mut f32,
    bray: *mut f32,
    nxa: *const i32,
    nya: *const i32,
    nxb: *const i32,
    nyb: *const i32,
    amat: *const f32,
    xc: *const f32,
    yc: *const f32,
    xt: *const f32,
    yt: *const f32,
    scale: *const f32,
    dmean: *const f32,
    linear: *const i32,
) {
    unsafe {
        let matrix = [[*amat, *amat.add(1)], [*amat.add(2), *amat.add(3)]];
        cubinterp(
            array, bray, *nxa, *nya, *nxb, *nyb, &matrix, *xc, *yc, *xt, *yt, *scale, *dmean,
            *linear,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_linear_and_nearest_copy_the_image() {
        let input = (0..25).map(|value| value as f32).collect::<Vec<_>>();
        let matrix = [[1., 0.], [0., 1.]];
        for mode in [-1, 1] {
            let mut output = vec![-1.; 25];
            unsafe {
                cubinterp(
                    input.as_ptr().cast_mut(),
                    output.as_mut_ptr(),
                    5,
                    5,
                    5,
                    5,
                    &matrix,
                    2.5,
                    2.5,
                    0.,
                    0.,
                    1.,
                    -99.,
                    mode,
                )
            };
            assert_eq!(output[2 + 2 * 5], input[2 + 2 * 5]);
            if mode > 0 {
                for y in 0..4 {
                    assert_eq!(&output[y * 5..y * 5 + 4], &input[y * 5..y * 5 + 4]);
                    assert_eq!(output[y * 5 + 4], -99.);
                }
                assert_eq!(&output[20..], &[-99.; 5]);
            } else {
                assert_eq!(output, input);
            }
        }
    }

    #[test]
    fn cubic_identity_preserves_safe_interior_and_wrapper_matrix_order() {
        let input = (0..49).map(|value| value as f32).collect::<Vec<_>>();
        let mut output = vec![0.; 49];
        let values = [1., 0., 0., 1.];
        let (nxa, nya, nxb, nyb) = (7, 7, 7, 7);
        unsafe {
            cubinterpfwrap(
                input.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                &nxa,
                &nya,
                &nxb,
                &nyb,
                values.as_ptr(),
                &3.5,
                &3.5,
                &0.,
                &0.,
                &1.,
                &-1.,
                &0,
            )
        };
        assert_eq!(output[3 + 3 * 7], input[3 + 3 * 7]);
        assert_eq!(output, input);
    }

    #[test]
    fn translated_image_uses_source_fill_value() {
        let input = vec![3.; 25];
        let mut output = vec![0.; 25];
        let matrix = [[1., 0.], [0., 1.]];
        unsafe {
            cubinterp(
                input.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                5,
                5,
                5,
                5,
                &matrix,
                2.5,
                2.5,
                10.,
                0.,
                1.,
                -7.,
                1,
            )
        };
        assert!(output.iter().all(|value| *value == -7.));
    }
}
