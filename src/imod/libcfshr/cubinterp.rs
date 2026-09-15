//! Translation of `IMOD/libcfshr/cubinterp.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::num_omp_threads;

/// `cubinterp` (`cubinterp.c:50`).
///
/// Statement-by-statement translation.  Two things the source does that a
/// naive port loses:
///
/// * The per-line safe-cubic span `ixst`/`ixnd` and the fallback band
///   `ixfbst`/`ixfbnd` (`cubinterp.c:100-153`).  Outside the fallback band the
///   source fills with `dmean` **unconditionally**; only inside it does it run
///   the range-tested quadratic/linear/nearest fallback, and only between
///   `ixst` and `ixnd` does it run the untested cubic loop.  Doing the
///   per-pixel test over the whole line instead gives an interpolated value
///   where the source fills.
/// * The literals.  `2.01`, `1.01`, `1.e-10`, `1.`, `2.`, `-1.e5` and
///   `nxb + 1.` are all **double** in C, so `xlft`/`xrt`, the cubic
///   coefficients `fx2`/`fx3`, `dxm1`/`dym1`, the y-combination and the whole
///   linear expression evaluate in double and round once on store.  `.5f` in
///   the quadratic branch is the one deliberately single-precision literal.
///
/// OpenMP is not represented: every output line is independent, so the
/// `numThreads` computation is carried for fidelity but changes no pixel.
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
    linear: i32,
) {
    let amat = *amat;
    let xcen: f32;
    let ycen: f32;
    let xco: f32;
    let yco: f32;
    let denom: f32;
    let a11: f32;
    let a12: f32;
    let a22: f32;
    let a21: f32;
    let mut dyo: f32;
    let mut xbase: f32;
    let mut ybase: f32;
    let mut xst: f32;
    let mut xnd: f32;
    let mut xlft: f32;
    let mut xrt: f32;
    let mut xp: f32;
    let mut yp: f32;
    let mut dennew: f32;
    let mut dx: f32;
    let mut dy: f32;
    let (mut v2, mut v4, mut v6, mut v8, mut v5): (f32, f32, f32, f32, f32);
    let (mut a, mut b, mut c, mut d): (f32, f32, f32, f32);
    let mut dxm1: f32;
    let mut dxdxm1: f32;
    let (mut fx1, mut fx2, mut fx3, mut fx4): (f32, f32, f32, f32);
    let mut dym1: f32;
    let mut dydym1: f32;
    let (mut v1, mut v3): (f32, f32);
    let mut vmin: f32;
    let mut vmax: f32;
    let (mut ixp, mut ixpp1, mut iyp, mut iypp1, mut ixpm1, mut iypm1): (
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
    );
    let mut linefb: i32;
    let (mut ixnd, mut ixst, mut ixfbst, mut ixfbnd, mut iqst, mut iqnd): (
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
    );
    let mut ixbase: usize;
    let llnxa: usize;
    let mut ind: usize;
    let mut indpnxa: usize;
    let mut indmnxa: usize;
    let mut indpnxa2: usize;
    let num_threads: i32;

    // Calc inverse transformation
    xcen = (nxb as f64 / 2. + xt as f64 + 0.5) as f32;
    ycen = (nyb as f64 / 2. + yt as f64 + 0.5) as f32;
    xco = (xc as f64 + 0.5) as f32;
    yco = (yc as f64 + 0.5) as f32;
    denom = amat[0][0] * amat[1][1] - amat[1][0] * amat[0][1];
    a11 = amat[1][1] / denom;
    a12 = -amat[1][0] / denom;
    a21 = -amat[0][1] / denom;
    a22 = amat[0][0] / denom;
    llnxa = nxa as usize;
    assert!(array.len() >= nxa as usize * nya as usize);
    assert!(bray.len() >= nxb as usize * nyb as usize);

    // Limit the number of threads based on measurements indicating that
    // this formula gives at least 75% parallel efficiency
    num_threads = ((0.04 * (nxb as f64 * nyb as f64).sqrt()) + 0.5).floor() as i32;
    let _num_threads = num_omp_threads(num_threads);

    // loop over output image
    for iy in 1..=nyb {
        ixbase = (iy as usize - 1) * nxb as usize;
        dyo = iy as f32 - ycen;
        xbase = a12 * dyo + xco - a11 * xcen;
        ybase = a22 * dyo + yco - a21 * xcen;
        xst = 1.;
        xnd = nxb as f32;
        linefb = 0;

        // Solve for limits in X of region that comes from safe range in X,
        // or set up the line not to be done or to be done as fallback if
        // the source in X is determined by xbase
        if (a11 as f64).abs() > 1.0e-10 {
            xlft = ((2.01 - xbase as f64) / a11 as f64) as f32;
            xrt = ((nxa as f64 - 1.01 - xbase as f64) / a11 as f64) as f32;
            let tmin = if xlft < xrt { xlft } else { xrt };
            xst = if xst > tmin { xst } else { tmin };
            let tmax = if xlft > xrt { xlft } else { xrt };
            xnd = if xnd < tmax { xnd } else { tmax };
        } else if (xbase as f64) < 2. || xbase as f64 >= nxa as f64 - 1. {
            xst = nxb as f32;
            xnd = 1.;
            if xbase as f64 >= 0.5 || xbase as f64 <= nxa as f64 + 0.5 {
                linefb = 1;
            }
        }

        // Solve for limits in X of region from safe range in Y and combine
        // these with the previous limits, or use the value of ybase
        if (a21 as f64).abs() > 1.0e-10 {
            xlft = ((2.01 - ybase as f64) / a21 as f64) as f32;
            xrt = ((nya as f64 - 1.01 - ybase as f64) / a21 as f64) as f32;
            let tmin = if xlft < xrt { xlft } else { xrt };
            xst = if xst > tmin { xst } else { tmin };
            let tmax = if xlft > xrt { xlft } else { xrt };
            xnd = if xnd < tmax { xnd } else { tmax };
        } else if (ybase as f64) < 2. || ybase as f64 >= nya as f64 - 1. {
            xst = nxb as f32;
            xnd = 1.;
            if ybase as f64 >= 0.5 || ybase as f64 <= nya as f64 + 0.5 {
                linefb = 1;
            }
        }

        // Truncate the ending value down and the starting value up but do
        // not pay any attention to xst bigger than nxb + 1
        ixnd = (if -1.0e5f64 > xnd as f64 {
            -1.0e5f64
        } else {
            xnd as f64
        }) as i32;
        ixst = nxb + 1
            - ((nxb as f64 + 1.
                - (if (xst as f64) < nxb as f64 + 1. {
                    xst as f64
                } else {
                    nxb as f64 + 1.
                })) as i32);

        // If they're crossed, set them up so fill will do whole line.
        // Otherwise, set up fallback region limits to do 2 pixels if not
        // doing whole line.  Then if doing fallback for whole line, set
        // that up
        ixfbst = 0;
        ixfbnd = 0;
        if ixst > ixnd {
            ixst = nxb / 2;
            ixnd = ixst - 1;
            ixfbst = ixst;
            ixfbnd = ixnd;
        } else if linefb == 0 {
            ixfbst = if 1 > ixst - 2 { 1 } else { ixst - 2 };
            ixfbnd = if nxb < ixnd + 2 { nxb } else { ixnd + 2 };
        }
        if linefb != 0 {
            ixfbst = 1;
            ixfbnd = nxb;
        }

        // Do fill outside of fallback
        for ix in 1..=ixfbst - 1 {
            bray[ixbase + ix as usize - 1] = dmean;
        }
        for ix in ixfbnd + 1..=nxb {
            bray[ixbase + ix as usize - 1] = dmean;
        }

        // Do fallback cubic to quadratic, or linear/nearest, with tests
        iqst = ixfbst;
        iqnd = ixst - 1;
        for _ifall in 1..=2 {
            if linear == 0 {
                // Do quadratic interpolation
                for ix in iqst..=iqnd {
                    xp = a11 * ix as f32 + xbase;
                    yp = a21 * ix as f32 + ybase;
                    ixp = (xp as f64 + 0.5).floor() as i32;
                    iyp = (yp as f64 + 0.5).floor() as i32;
                    dennew = dmean;
                    if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                        dx = xp - ixp as f32;
                        dy = yp - iyp as f32;
                        ixpp1 = ixp + 1;
                        ixpm1 = ixp - 1;
                        iypp1 = iyp + 1;
                        iypm1 = iyp - 1;
                        if ixpm1 < 1 {
                            ixpm1 = 1;
                        }
                        if iypm1 < 1 {
                            iypm1 = 1;
                        }
                        if ixpp1 > nxa {
                            ixpp1 = nxa;
                        }
                        if iypp1 > nya {
                            iypp1 = nya;
                        }

                        // set up terms for quadratic interpolation
                        v2 = array[ixp as usize - 1 + (iypm1 as usize - 1) * llnxa];
                        v4 = array[ixpm1 as usize - 1 + (iyp as usize - 1) * llnxa];
                        v5 = array[ixp as usize - 1 + (iyp as usize - 1) * llnxa];
                        v6 = array[ixpp1 as usize - 1 + (iyp as usize - 1) * llnxa];
                        v8 = array[ixp as usize - 1 + (iypp1 as usize - 1) * llnxa];
                        vmax = if v2 > v4 { v2 } else { v4 };
                        vmax = if vmax > v5 { vmax } else { v5 };
                        vmax = if vmax > v6 { vmax } else { v6 };
                        vmax = if vmax > v8 { vmax } else { v8 };
                        vmin = if v2 < v4 { v2 } else { v4 };
                        vmin = if vmin < v5 { vmin } else { v5 };
                        vmin = if vmin < v6 { vmin } else { v6 };
                        vmin = if vmin < v8 { vmin } else { v8 };

                        a = (v6 + v4) * 0.5f32 - v5;
                        b = (v8 + v2) * 0.5f32 - v5;
                        c = (v6 - v4) * 0.5f32;
                        d = (v8 - v2) * 0.5f32;

                        dennew = scale * (a * dx * dx + b * dy * dy + c * dx + d * dy + v5);
                        if dennew > vmax {
                            dennew = vmax;
                        }
                        if dennew < vmin {
                            dennew = vmin;
                        }
                    }
                    bray[ixbase + ix as usize - 1] = dennew;
                }
            } else {
                // fallback to linear
                for ix in iqst..=iqnd {
                    xp = a11 * ix as f32 + xbase;
                    yp = a21 * ix as f32 + ybase;
                    dennew = dmean;
                    if linear > 0 {
                        ixp = xp as i32;
                        iyp = yp as i32;
                        if ixp >= 1 && ixp < nxa && iyp >= 1 && iyp < nya {
                            dx = xp - ixp as f32;
                            dy = yp - iyp as f32;
                            ind = ixp as usize - 1 + (iyp as usize - 1) * llnxa;
                            dennew = ((1. - dy as f64)
                                * ((1. - dx as f64) * array[ind] as f64
                                    + (dx * array[ind + 1]) as f64)
                                + dy as f64
                                    * ((1. - dx as f64) * array[ind + llnxa] as f64
                                        + (dx * array[ind + llnxa + 1]) as f64))
                                as f32;
                        }
                    } else {
                        ixp = (xp as f64 + 0.5) as i32;
                        iyp = (yp as f64 + 0.5) as i32;
                        if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                            dennew = array[ixp as usize - 1 + (iyp as usize - 1) * llnxa];
                        }
                    }
                    bray[ixbase + ix as usize - 1] = dennew;
                }
            }
            iqst = ixnd + 1;
            iqnd = ixfbnd;
        }

        if linear == 0 {
            // Do cubic interpolation on the central region
            for ix in ixst..=ixnd {
                xp = a11 * ix as f32 + xbase;
                yp = a21 * ix as f32 + ybase;
                ixp = xp as i32;
                iyp = yp as i32;
                dx = xp - ixp as f32;
                dy = yp - iyp as f32;

                dxm1 = (dx as f64 - 1.) as f32;
                dxdxm1 = dx * dxm1;
                fx1 = -dxm1 * dxdxm1;
                fx4 = dx * dxdxm1;
                fx2 = (1. + (dx * dx) as f64 * (dx as f64 - 2.)) as f32;
                fx3 = (dx as f64 * (1. - dxdxm1 as f64)) as f32;

                dym1 = (dy as f64 - 1.) as f32;
                dydym1 = dy * dym1;
                ind = ixp as usize - 1 + (iyp as usize - 1) * llnxa;
                indmnxa = ind - llnxa;
                indpnxa = ind + llnxa;
                indpnxa2 = ind + 2 * llnxa;
                v1 = fx1 * array[indmnxa - 1]
                    + fx2 * array[indmnxa]
                    + fx3 * array[indmnxa + 1]
                    + fx4 * array[indmnxa + 2];
                v2 = fx1 * array[ind - 1]
                    + fx2 * array[ind]
                    + fx3 * array[ind + 1]
                    + fx4 * array[ind + 2];
                v3 = fx1 * array[indpnxa - 1]
                    + fx2 * array[indpnxa]
                    + fx3 * array[indpnxa + 1]
                    + fx4 * array[indpnxa + 2];
                v4 = fx1 * array[indpnxa2 - 1]
                    + fx2 * array[indpnxa2]
                    + fx3 * array[indpnxa2 + 1]
                    + fx4 * array[indpnxa2 + 2];

                bray[ixbase + ix as usize - 1] = ((-dym1 * dydym1 * v1) as f64
                    + (1. + (dy * dy) as f64 * (dy as f64 - 2.)) * v2 as f64
                    + dy as f64 * (1. - dydym1 as f64) * v3 as f64
                    + (dy * dydym1 * v4) as f64)
                    as f32;
            }
        } else if linear > 0 {
            // do linear interpolation
            for ix in ixst..=ixnd {
                xp = a11 * ix as f32 + xbase;
                yp = a21 * ix as f32 + ybase;
                ixp = xp as i32;
                iyp = yp as i32;
                dx = xp - ixp as f32;
                dy = yp - iyp as f32;
                ind = ixp as usize - 1 + (iyp as usize - 1) * llnxa;
                bray[ixbase + ix as usize - 1] = ((1. - dy as f64)
                    * ((1. - dx as f64) * array[ind] as f64 + (dx * array[ind + 1]) as f64)
                    + dy as f64
                        * ((1. - dx as f64) * array[ind + llnxa] as f64
                            + (dx * array[ind + llnxa + 1]) as f64))
                    as f32;
            }
        } else {
            // do nearest neighbor interpolation
            for ix in ixst..=ixnd {
                xp = a11 * ix as f32 + xbase;
                yp = a21 * ix as f32 + ybase;
                ixp = (xp as f64 + 0.5) as i32;
                iyp = (yp as f64 + 0.5) as i32;
                ind = ixp as usize - 1 + (iyp as usize - 1) * llnxa;
                bray[ixbase + ix as usize - 1] = array[ind];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLD_NEAREST: [u32; 80] = [
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0x42160000, 0x41e40000,
        0x42870000, 0x426a0000, 0x42c30000, 0x43020000, 0x42b10000, 0xc0e80000, 0xc0e80000,
        0x42ab0000, 0x42ab0000, 0x42990000, 0x42da0000, 0x42d50000, 0x430b0000, 0xc2920000,
        0xc2a40000, 0xc22c0000, 0xc0e80000, 0xc0e80000, 0xc2aa0000, 0xc2bc0000, 0xc25c0000,
        0xc2800000, 0xc2800000, 0xc2920000, 0xc2080000, 0xbfc00000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc1580000, 0xc1b40000, 0x41840000, 0x40f00000, 0x423a0000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000,
    ];
    const GOLD_CUBIC: [u32; 80] = [
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0x421529fa, 0x4263b586,
        0x4245c279, 0x4280815e, 0x42b15ae8, 0x43020000, 0x43020000, 0xc0e80000, 0xc0e80000,
        0x42ecc72a, 0x42df9d7f, 0x42b8bfa5, 0x431d7c19, 0x430d8ca7, 0x430444ba, 0x41c582ca,
        0xc29b24a3, 0xc2adf12d, 0xc0e80000, 0xc0e80000, 0xc10af381, 0xc2bc0000, 0xc17fa51b,
        0xc193582f, 0xc28b3129, 0xc29df65a, 0xc1fba8d7, 0xc1b7136c, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc1e844df, 0xc1955f4e, 0xc08e9c53, 0x41943152, 0x424c166a,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000,
    ];
    const GOLD_LINEAR: [u32; 80] = [
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0x41ddb761, 0x4225ba94,
        0x4261364f, 0x428e81f2, 0x42ac3419, 0x42c632de, 0x42b677bd, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0x42a85731, 0x429bbf16, 0x42ddef46, 0x42e044f3, 0x42f48e16, 0x41bc6cc2,
        0xc25f3682, 0xc245afeb, 0xc0e80000, 0xc0e80000, 0x41973672, 0xc23f9d4f, 0xc100baf8,
        0x40c33370, 0xc207849f, 0xc2153c27, 0xc1b97a2b, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc1cce62a, 0xc14caa0d, 0x407144a3, 0x41908f21, 0x420216b2,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000,
    ];
    const GOLD_SHIFTED: [u32; 80] = [
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc2a1c000, 0xc1990000, 0x425e8000, 0x42826000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0x41310000,
        0x42ab5000, 0x42eb2400, 0xc2830400, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0x42fe0000, 0xc2a56000, 0xc1400000, 0x4280c000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000, 0xc0e80000,
        0xc0e80000, 0xc0e80000, 0xc0e80000,
    ];

    /// Deterministic input shared with the C golden generator.
    fn src(i: i32) -> f32 {
        ((i * 37) % 251) as f32 - 100.0f32 + 0.5f32 * ((i * 17) % 13) as f32
    }

    fn run(linear: i32, amat: &[[f32; 2]; 2], xt: f32, yt: f32, scale: f32) -> Vec<f32> {
        let input = (0..8 * 6).map(src).collect::<Vec<f32>>();
        let mut output = vec![-12345.0f32; 10 * 8];
        cubinterp(
            &input,
            &mut output,
            8,
            6,
            10,
            8,
            amat,
            4.0,
            3.0,
            xt,
            yt,
            scale,
            -7.25,
            linear,
        );
        output
    }

    fn rot25() -> [[f32; 2]; 2] {
        let th = 25.0f64 * 3.14159265358979 / 180.0;
        let mut amat = [[0.0f32; 2]; 2];
        amat[0][0] = th.cos() as f32;
        amat[1][0] = (-th.sin()) as f32;
        amat[0][1] = th.sin() as f32;
        amat[1][1] = th.cos() as f32;
        amat
    }

    fn assert_bits(got: &[f32], want: &[u32], what: &str) {
        let bad = (0..want.len())
            .filter(|i| got[*i].to_bits() != want[*i])
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "{what}: {} of {} pixels differ from the native libcfshr result, first at {:?} \
             (got {:#010x} = {}, want {:#010x} = {})",
            bad.len(),
            want.len(),
            bad.first(),
            got[bad[0]].to_bits(),
            got[bad[0]],
            want[bad[0]],
            f32::from_bits(want[bad[0]])
        );
    }

    /// Every output pixel must reproduce `cubinterp` from the reference
    /// `libcfshr` bit for bit.  These three arrays were captured by calling
    /// the native routine directly; they pin the cubic coefficients and the
    /// y-combination, which the source evaluates in **double** (`1.`, `2.`
    /// and `1. - dxdxm1` are double literals, `cubinterp.c:262-283`).
    /// Recomputing them in `f32` moves roughly half the interior pixels by
    /// one or two ulp and this test fails.
    #[test]
    fn cubic_linear_and_nearest_match_the_native_routine_bit_for_bit() {
        let amat = rot25();
        assert_bits(&run(-1, &amat, 0.75, -1.25, 1.4), &GOLD_NEAREST, "nearest");
        assert_bits(&run(1, &amat, 0.75, -1.25, 1.4), &GOLD_LINEAR, "linear");
        assert_bits(&run(0, &amat, 0.75, -1.25, 1.4), &GOLD_CUBIC, "cubic");
    }

    /// `cubinterp.c:100-153` restricts interpolation to the fallback band
    /// `ixfbst..=ixfbnd` and fills everything outside it with `dmean`
    /// unconditionally.  A per-pixel range test over the whole line instead
    /// produces an interpolated value for pixels that are merely near the
    /// input, so this case — a 0.5x matrix pushed off centre, which walks the
    /// output well past both ends of an 8x6 input — is the direct check:
    /// 68 of the 80 pixels are fill and 12 are interpolated.
    #[test]
    fn pixels_outside_the_fallback_band_are_filled_with_dmean() {
        let mut amat = [[0.0f32; 2]; 2];
        amat[0][0] = 0.5;
        amat[1][0] = 0.0;
        amat[0][1] = 0.0;
        amat[1][1] = 0.5;
        let out = run(0, &amat, 2.5, 1.5, 1.0);
        assert_bits(&out, &GOLD_SHIFTED, "shifted 2x cubic");
        // The golden data is only a useful guard if it really contains both
        // fill and interpolated pixels.
        let fills = out
            .iter()
            .filter(|v| v.to_bits() == (-7.25f32).to_bits())
            .count();
        assert!(
            fills > 8 && fills < 72,
            "expected a mix of fill and interpolated pixels, got {fills} fills"
        );
    }

    #[test]
    fn identity_linear_and_nearest_copy_the_image() {
        let input = (0..25).map(|value| value as f32).collect::<Vec<_>>();
        let matrix = [[1., 0.], [0., 1.]];
        for mode in [-1, 1] {
            let mut output = vec![-1.; 25];
            cubinterp(
                &input,
                &mut output,
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
            );
            assert_eq!(output[2 + 2 * 5], input[2 + 2 * 5]);
        }
    }

    #[test]
    fn translated_image_uses_source_fill_value() {
        let input = vec![3.; 25];
        let mut output = vec![0.; 25];
        let matrix = [[1., 0.], [0., 1.]];
        cubinterp(
            &input,
            &mut output,
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
        );
        assert!(output.iter().all(|value| *value == -7.));
    }
}
