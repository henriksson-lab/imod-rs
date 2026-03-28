/// Interpolation mode for image transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Cubic interpolation (highest quality, falls back to quadratic at edges).
    Cubic,
    /// Bilinear interpolation.
    Linear,
    /// Nearest-neighbor interpolation (fastest, lowest quality).
    NearestNeighbor,
}

/// Applies a 2D linear (affine) transformation to an image using cubic, linear,
/// or nearest-neighbor interpolation.
///
/// For cubic mode, the inner region uses full cubic interpolation while the edges
/// fall back to quadratic interpolation with range clamping. This eliminates all
/// range tests from the inner loop for maximum performance.
///
/// # Coordinate transformation
///
/// The mapping from input coordinates `(xi, yi)` to output coordinates `(xo, yo)` is:
///
/// ```text
/// xo = a11*(xi - xc) + a12*(yi - yc) + nxb/2 + xt
/// yo = a21*(xi - xc) + a22*(yi - yc) + nyb/2 + yt
/// ```
///
/// where `xi` runs from 0 at the left edge of the first pixel to `nxa` at the
/// right edge of the last pixel, and similarly for `yi`.
///
/// The matrix layout is `amat = [[a11, a21], [a12, a22]]` (column-major / Fortran order).
///
/// # Arguments
///
/// * `array` - Input image, row-major, dimensions `nxa * nya`.
/// * `bray`  - Output image, row-major, dimensions `nxb * nyb`.
/// * `nxa`, `nya` - Dimensions of the input image.
/// * `nxb`, `nyb` - Dimensions of the output image.
/// * `amat` - 2x2 transformation matrix `[[a11, a21], [a12, a22]]`.
/// * `xc`, `yc` - Center of the input image for the transform.
/// * `xt`, `yt` - Translation added in the output image.
/// * `scale` - Multiplicative intensity scale factor (applied only in quadratic fallback).
/// * `dmean` - Fill value for pixels that fall outside the input image.
/// * `mode` - Interpolation mode to use.
pub fn cubinterp(
    array: &[f32],
    bray: &mut [f32],
    nxa: i32,
    nya: i32,
    nxb: i32,
    nyb: i32,
    amat: &[[f32; 2]; 2],
    xc: f32,
    yc: f32,
    xt: f32,
    yt: f32,
    scale: f32,
    dmean: f32,
    mode: InterpolationMode,
) {
    let use_cubic = mode == InterpolationMode::Cubic;
    let use_linear = mode == InterpolationMode::Linear;

    // Compute the inverse transformation
    let xcen = nxb as f32 / 2.0 + xt + 0.5;
    let ycen = nyb as f32 / 2.0 + yt + 0.5;
    let xco = xc + 0.5;
    let yco = yc + 0.5;
    let denom = amat[0][0] * amat[1][1] - amat[1][0] * amat[0][1];
    let a11 = amat[1][1] / denom;
    let a12 = -amat[1][0] / denom;
    let a21 = -amat[0][1] / denom;
    let a22 = amat[0][0] / denom;
    let llnxa = nxa as usize;

    // Loop over output image rows
    for iy in 1..=nyb {
        let ixbase: usize = (iy as usize - 1) * nxb as usize;
        let dyo = iy as f32 - ycen;
        let xbase = a12 * dyo + xco - a11 * xcen;
        let ybase = a22 * dyo + yco - a21 * xcen;
        let mut xst: f32 = 1.0;
        let mut xnd: f32 = nxb as f32;
        let mut linefb = false;

        // Solve for limits in X of region that comes from safe range in X
        if a11.abs() > 1.0e-10 {
            let xlft = (2.01 - xbase) / a11;
            let xrt = (nxa as f32 - 1.01 - xbase) / a11;
            xst = xst.max(xlft.min(xrt));
            xnd = xnd.min(xlft.max(xrt));
        } else if xbase < 2.0 || xbase >= nxa as f32 - 1.0 {
            xst = nxb as f32;
            xnd = 1.0;
            if xbase >= 0.5 || xbase <= nxa as f32 + 0.5 {
                linefb = true;
            }
        }

        // Solve for limits in X of region from safe range in Y
        if a21.abs() > 1.0e-10 {
            let xlft = (2.01 - ybase) / a21;
            let xrt = (nya as f32 - 1.01 - ybase) / a21;
            xst = xst.max(xlft.min(xrt));
            xnd = xnd.min(xlft.max(xrt));
        } else if ybase < 2.0 || ybase >= nya as f32 - 1.0 {
            xst = nxb as f32;
            xnd = 1.0;
            if ybase >= 0.5 || ybase <= nya as f32 + 0.5 {
                linefb = true;
            }
        }

        // Truncate ending value down and starting value up
        let mut ixnd = xnd.max(-1.0e5) as i32;
        let mut ixst =
            nxb + 1 - (nxb as f32 + 1.0 - xst.min(nxb as f32 + 1.0)) as i32;

        let ixfbst: i32;
        let ixfbnd: i32;

        if ixst > ixnd {
            // Crossed: fill the whole line
            ixst = nxb / 2;
            ixnd = ixst - 1;
            ixfbst = ixst;
            ixfbnd = ixnd;
        } else if linefb {
            ixfbst = 1;
            ixfbnd = nxb;
        } else {
            ixfbst = 1.max(ixst - 2);
            ixfbnd = nxb.min(ixnd + 2);
        }

        // Fill outside the fallback region with dmean
        for ix in 1..ixfbst {
            bray[ix as usize + ixbase - 1] = dmean;
        }
        for ix in (ixfbnd + 1)..=nxb {
            bray[ix as usize + ixbase - 1] = dmean;
        }

        // Two-pass fallback: before and after the central safe region
        let mut iqst = ixfbst;
        let mut iqnd = ixst - 1;
        for _ifall in 0..2 {
            if use_cubic {
                // Quadratic interpolation fallback
                for ix in iqst..=iqnd {
                    let xp = a11 * ix as f32 + xbase;
                    let yp = a21 * ix as f32 + ybase;
                    let ixp = (xp + 0.5).floor() as i32; // B3DNINT
                    let iyp = (yp + 0.5).floor() as i32;
                    let mut dennew = dmean;

                    if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                        let dx = xp - ixp as f32;
                        let dy = yp - iyp as f32;
                        let ixpp1 = (ixp + 1).min(nxa);
                        let ixpm1 = (ixp - 1).max(1);
                        let iypp1 = (iyp + 1).min(nya);
                        let iypm1 = (iyp - 1).max(1);

                        let v2 = array[idx1(ixp, iypm1, llnxa)];
                        let v4 = array[idx1(ixpm1, iyp, llnxa)];
                        let v5 = array[idx1(ixp, iyp, llnxa)];
                        let v6 = array[idx1(ixpp1, iyp, llnxa)];
                        let v8 = array[idx1(ixp, iypp1, llnxa)];

                        let vmax = v2.max(v4).max(v5).max(v6).max(v8);
                        let vmin = v2.min(v4).min(v5).min(v6).min(v8);

                        let a = (v6 + v4) * 0.5 - v5;
                        let b = (v8 + v2) * 0.5 - v5;
                        let c = (v6 - v4) * 0.5;
                        let d = (v8 - v2) * 0.5;

                        dennew = (scale * (a * dx * dx + b * dy * dy + c * dx + d * dy + v5))
                            .clamp(vmin, vmax);
                    }
                    bray[ix as usize + ixbase - 1] = dennew;
                }
            } else {
                // Linear or nearest-neighbor fallback
                for ix in iqst..=iqnd {
                    let xp = a11 * ix as f32 + xbase;
                    let yp = a21 * ix as f32 + ybase;
                    let mut dennew = dmean;

                    if use_linear {
                        let ixp = xp as i32;
                        let iyp = yp as i32;
                        if ixp >= 1 && ixp < nxa && iyp >= 1 && iyp < nya {
                            let dx = xp - ixp as f32;
                            let dy = yp - iyp as f32;
                            let ind = idx1(ixp, iyp, llnxa);
                            dennew = (1.0 - dy)
                                * ((1.0 - dx) * array[ind] + dx * array[ind + 1])
                                + dy * ((1.0 - dx) * array[ind + llnxa]
                                    + dx * array[ind + llnxa + 1]);
                        }
                    } else {
                        // Nearest neighbor
                        let ixp = (xp + 0.5) as i32;
                        let iyp = (yp + 0.5) as i32;
                        if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                            dennew = array[idx1(ixp, iyp, llnxa)];
                        }
                    }
                    bray[ix as usize + ixbase - 1] = dennew;
                }
            }
            iqst = ixnd + 1;
            iqnd = ixfbnd;
        }

        // Central region: full-quality interpolation without range tests
        if use_cubic {
            for ix in ixst..=ixnd {
                let xp = a11 * ix as f32 + xbase;
                let yp = a21 * ix as f32 + ybase;
                let ixp = xp as i32;
                let iyp = yp as i32;
                let dx = xp - ixp as f32;
                let dy = yp - iyp as f32;

                let dxm1 = dx - 1.0;
                let dxdxm1 = dx * dxm1;
                let fx1 = -dxm1 * dxdxm1;
                let fx4 = dx * dxdxm1;
                let fx2 = 1.0 + dx * dx * (dx - 2.0);
                let fx3 = dx * (1.0 - dxdxm1);

                let dym1 = dy - 1.0;
                let dydym1 = dy * dym1;

                let ind = idx1(ixp, iyp, llnxa);
                let ind_m = ind - llnxa; // row above
                let ind_p = ind + llnxa; // row below
                let ind_p2 = ind + 2 * llnxa; // two rows below

                let v1 = fx1 * array[ind_m - 1]
                    + fx2 * array[ind_m]
                    + fx3 * array[ind_m + 1]
                    + fx4 * array[ind_m + 2];
                let v2 = fx1 * array[ind - 1]
                    + fx2 * array[ind]
                    + fx3 * array[ind + 1]
                    + fx4 * array[ind + 2];
                let v3 = fx1 * array[ind_p - 1]
                    + fx2 * array[ind_p]
                    + fx3 * array[ind_p + 1]
                    + fx4 * array[ind_p + 2];
                let v4 = fx1 * array[ind_p2 - 1]
                    + fx2 * array[ind_p2]
                    + fx3 * array[ind_p2 + 1]
                    + fx4 * array[ind_p2 + 2];

                bray[ix as usize + ixbase - 1] = -dym1 * dydym1 * v1
                    + (1.0 + dy * dy * (dy - 2.0)) * v2
                    + dy * (1.0 - dydym1) * v3
                    + dy * dydym1 * v4;
            }
        } else if use_linear {
            for ix in ixst..=ixnd {
                let xp = a11 * ix as f32 + xbase;
                let yp = a21 * ix as f32 + ybase;
                let ixp = xp as i32;
                let iyp = yp as i32;
                let dx = xp - ixp as f32;
                let dy = yp - iyp as f32;
                let ind = idx1(ixp, iyp, llnxa);
                bray[ix as usize + ixbase - 1] = (1.0 - dy)
                    * ((1.0 - dx) * array[ind] + dx * array[ind + 1])
                    + dy * ((1.0 - dx) * array[ind + llnxa] + dx * array[ind + llnxa + 1]);
            }
        } else {
            // Nearest neighbor
            for ix in ixst..=ixnd {
                let xp = a11 * ix as f32 + xbase;
                let yp = a21 * ix as f32 + ybase;
                let ixp = (xp + 0.5) as i32;
                let iyp = (yp + 0.5) as i32;
                let ind = idx1(ixp, iyp, llnxa);
                bray[ix as usize + ixbase - 1] = array[ind];
            }
        }
    }
}

/// Convert 1-based (column, row) indices to a 0-based flat index.
/// Equivalent to C expression: `array[ixp + (iyp - 1) * nxa - 1]`
#[inline(always)]
fn idx1(col: i32, row: i32, nxa: usize) -> usize {
    (col as usize - 1) + (row as usize - 1) * nxa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_nearest_neighbor() {
        // Use a larger image so the safe region covers the center
        let n = 16i32;
        let input: Vec<f32> = (1..=(n * n)).map(|v| v as f32).collect();
        let mut output = vec![0.0f32; (n * n) as usize];

        let amat = [[1.0f32, 0.0], [0.0, 1.0]];
        let xc = n as f32 / 2.0;
        let yc = n as f32 / 2.0;

        cubinterp(
            &input, &mut output, n, n, n, n,
            &amat, xc, yc, 0.0, 0.0, 1.0, 0.0,
            InterpolationMode::NearestNeighbor,
        );

        // Interior pixels (away from edges) should match input
        for y in 2..(n - 2) as usize {
            for x in 2..(n - 2) as usize {
                let idx = y * n as usize + x;
                assert!(
                    (output[idx] - input[idx]).abs() < 1.0,
                    "mismatch at ({}, {}): {} vs {}",
                    x, y, output[idx], input[idx]
                );
            }
        }
    }

    #[test]
    fn fill_value_when_outside() {
        let nxa = 4i32;
        let nya = 4i32;
        let input = vec![1.0f32; (nxa * nya) as usize];

        let nxb = 4i32;
        let nyb = 4i32;
        let mut output = vec![0.0f32; (nxb * nyb) as usize];

        // Large translation pushes everything out of bounds
        let amat = [[1.0f32, 0.0], [0.0, 1.0]];

        cubinterp(
            &input,
            &mut output,
            nxa,
            nya,
            nxb,
            nyb,
            &amat,
            nxa as f32 / 2.0,
            nya as f32 / 2.0,
            100.0,
            100.0,
            1.0,
            -999.0,
            InterpolationMode::NearestNeighbor,
        );

        // Everything should be the fill value
        for v in &output {
            assert_eq!(*v, -999.0);
        }
    }
}
