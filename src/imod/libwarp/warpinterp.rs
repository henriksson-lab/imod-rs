//! Translation of `IMOD/libwarp/warpinterp.c`.
//!
//! Converted to idiomatic Rust per `NATIVE.md`: `float *` becomes a slice and
//! the walking `float *buf` becomes an index.  None of the arithmetic moved —
//! see the note on the literals below.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::num_omp_threads;
use crate::imod::libwarp::warputils::interpolate_grid;

/// Original `warpInterp` (`warpinterp.c:50`).
///
/// Two things a naive port loses:
///
/// * The literals.  `1.`, `2.`, `0.5`, `0.999` and `nxb - 1.` are **double**
///   in C, so `dxm1`/`dym1`, the cubic coefficients `fx2`/`fx3`, the whole
///   linear expression, the y-combination, `xstep`/`ystep` and the block
///   coordinate limits all evaluate in double and round once on store.  A
///   `float * float` product inside one of those expressions — `dx * b`,
///   `fx * array[ind + 1]`, `gridfy * xmap[0][1]` — is still single precision
///   and only widens for the addition, which is why the casts here are placed
///   per operand rather than around the expression.
/// * `scale` is a declared parameter that the source never reads -- the
///   intensity scaling the doc comment describes is not applied.  It is kept
///   here with the same position so callers translate unchanged.
///
/// OpenMP is not represented: every output line (undistort branch) or grid
/// block (warp branch) writes a disjoint part of `bray`, so the `numThreads`
/// computation is carried for fidelity but changes no pixel.
#[allow(clippy::too_many_arguments)]
pub fn warp_interp(
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
    _scale: f32,
    dmean: f32,
    linear: i32,
    lin_first: i32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_strt: f32,
    y_grid_strt: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
) {
    /* Calc inverse transformation */
    let mut xcen = (nxb as f64 / 2. + xt as f64 + 0.5) as f32;
    let mut ycen = (nyb as f64 / 2. + yt as f64 + 0.5) as f32;
    let mut xco = (xc as f64 + 0.5) as f32;
    let mut yco = (yc as f64 + 0.5) as f32;
    let denom = amat[0][0] * amat[1][1] - amat[1][0] * amat[0][1];
    let a11 = amat[1][1] / denom;
    let a12 = -amat[1][0] / denom;
    let a21 = -amat[0][1] / denom;
    let a22 = amat[0][0] / denom;
    let llnxa = nxa as i64;

    /* Limit the number of threads  TODO: SEE IF NEEDED */
    let num_threads = ((0.04 * (nxb as f64 * nyb as f64).sqrt()) + 0.5).floor() as i32;
    let _num_threads = num_omp_threads(num_threads);

    if lin_first == 0 {
        /* UNDISTORT FOLLOWED BY LINEAR TRANSFORM */
        /* Use 1-based pixel coordinates to match fortran version of undistinterp */
        /* loop over output image */
        for iy in 1..=nyb {
            let ixbase = (iy as i64 - 1) * nxb as i64 - 1;
            let dyo = iy as f32 - ycen;
            let xbase = a12 * dyo + xco - a11 * xcen;
            let ybase = a22 * dyo + yco - a21 * xcen;

            if linear > 0 {
                /* linear interpolation */
                for ix in 1..=nxb {
                    let mut xp = a11 * ix as f32 + xbase;
                    let mut yp = a21 * ix as f32 + ybase;
                    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
                    interpolate_grid(
                        xp,
                        yp,
                        dx_grid,
                        dy_grid,
                        ixg_dim,
                        nx_grid,
                        ny_grid,
                        x_grid_strt,
                        y_grid_strt,
                        x_grid_intrv,
                        y_grid_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    xp += dx;
                    yp += dy;
                    let ixp = xp as i32;
                    let iyp = yp as i32;
                    bray[(ix as i64 + ixbase) as usize] = dmean;
                    if ixp >= 1 && ixp < nxa && iyp >= 1 && iyp < nya {
                        dx = xp - ixp as f32;
                        dy = yp - iyp as f32;
                        let ixpm1 = (ixp - 1) as i64;
                        let iypm1 = (iyp - 1) as i64;
                        bray[(ix as i64 + ixbase) as usize] = ((1. - dy as f64)
                            * ((1. - dx as f64) * array[(ixpm1 + iypm1 * llnxa) as usize] as f64
                                + (dx * array[(ixp as i64 + iypm1 * llnxa) as usize]) as f64)
                            + dy as f64
                                * ((1. - dx as f64)
                                    * array[(ixpm1 + iyp as i64 * llnxa) as usize] as f64
                                    + (dx * array[(ixp as i64 + iyp as i64 * llnxa) as usize])
                                        as f64))
                            as f32;
                    }
                }
            } else if linear < 0 {
                /* Nearest neighbor */
                for ix in 1..=nxb {
                    let xp = a11 * ix as f32 + xbase;
                    let yp = a21 * ix as f32 + ybase;
                    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
                    interpolate_grid(
                        xp,
                        yp,
                        dx_grid,
                        dy_grid,
                        ixg_dim,
                        nx_grid,
                        ny_grid,
                        x_grid_strt,
                        y_grid_strt,
                        x_grid_intrv,
                        y_grid_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    // `xp + dx` is a float sum; only `+ 0.5` widens it.
                    let ixp = ((xp + dx) as f64 + 0.5) as i32;
                    let iyp = ((yp + dy) as f64 + 0.5) as i32;
                    bray[(ix as i64 + ixbase) as usize] =
                        if ixp >= 1 && ixp <= nxa && iyp >= 1 && iyp <= nya {
                            array[((ixp - 1) as i64 + (iyp as i64 - 1) * llnxa) as usize]
                        } else {
                            dmean
                        };
                }
            } else {
                /* cubic interpolation */
                for ix in 1..=nxb {
                    let mut xp = a11 * ix as f32 + xbase;
                    let mut yp = a21 * ix as f32 + ybase;
                    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
                    interpolate_grid(
                        xp,
                        yp,
                        dx_grid,
                        dy_grid,
                        ixg_dim,
                        nx_grid,
                        ny_grid,
                        x_grid_strt,
                        y_grid_strt,
                        x_grid_intrv,
                        y_grid_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    xp += dx;
                    yp += dy;
                    let ixp = xp as i32;
                    let iyp = yp as i32;
                    bray[(ix as i64 + ixbase) as usize] = dmean;
                    if ixp >= 2 && ixp < nxa - 1 && iyp >= 2 && iyp < nya - 1 {
                        dx = xp - ixp as f32;
                        dy = yp - iyp as f32;
                        let ixpp1 = (ixp + 1) as i64;
                        let ixpm1 = (ixp - 1) as i64;
                        let iypp1 = (iyp + 1) as i64;
                        let iypm1 = (iyp - 1) as i64;
                        let ixpm2 = (ixp - 2) as i64;
                        let iypm2 = (iyp - 2) as i64;
                        let dxm1 = (dx as f64 - 1.) as f32;
                        let dxdxm1 = dx * dxm1;
                        let fx1 = -dxm1 * dxdxm1;
                        let fx4 = dx * dxdxm1;
                        let fx2 = (1. + (dx * dx) as f64 * (dx as f64 - 2.)) as f32;
                        let fx3 = (dx as f64 * (1. - dxdxm1 as f64)) as f32;

                        let dym1 = (dy as f64 - 1.) as f32;
                        let dydym1 = dy * dym1;
                        let ixp = ixp as i64;
                        let iyp = iyp as i64;

                        let v1 = fx1 * array[(ixpm2 + iypm2 * llnxa) as usize]
                            + fx2 * array[(ixpm1 + iypm2 * llnxa) as usize]
                            + fx3 * array[(ixp + iypm2 * llnxa) as usize]
                            + fx4 * array[(ixpp1 + iypm2 * llnxa) as usize];
                        let v2 = fx1 * array[(ixpm2 + iypm1 * llnxa) as usize]
                            + fx2 * array[(ixpm1 + iypm1 * llnxa) as usize]
                            + fx3 * array[(ixp + iypm1 * llnxa) as usize]
                            + fx4 * array[(ixpp1 + iypm1 * llnxa) as usize];
                        let v3 = fx1 * array[(ixpm2 + iyp * llnxa) as usize]
                            + fx2 * array[(ixpm1 + iyp * llnxa) as usize]
                            + fx3 * array[(ixp + iyp * llnxa) as usize]
                            + fx4 * array[(ixpp1 + iyp * llnxa) as usize];
                        let v4 = fx1 * array[(ixpm2 + iypp1 * llnxa) as usize]
                            + fx2 * array[(ixpm1 + iypp1 * llnxa) as usize]
                            + fx3 * array[(ixp + iypp1 * llnxa) as usize]
                            + fx4 * array[(ixpp1 + iypp1 * llnxa) as usize];
                        bray[(ix as i64 + ixbase) as usize] = ((-dym1 * dydym1 * v1) as f64
                            + (1. + (dy * dy) as f64 * (dy as f64 - 2.)) * v2 as f64
                            + dy as f64 * (1. - dydym1 as f64) * v3 as f64
                            + (dy * dydym1 * v4) as f64)
                            as f32;
                    }
                }
            }
        }
    } else {
        /* WARPING AFTER LINEAR TRANSFORMATION */
        /* Subtract 1 to work with 0-based coordinates now to match what midas does */
        xco = (xco as f64 - 1.) as f32;
        yco = (yco as f64 - 1.) as f32;
        xcen = (xcen as f64 - 1.) as f32;
        ycen = (ycen as f64 - 1.) as f32;
        let ox = xco - a11 * xcen - a12 * ycen;
        let oy = yco - a21 * xcen - a22 * ycen;

        /* Determine indexes of grid points ending starting and ending grid intervals */
        let mut ixg_start = ((-x_grid_strt / x_grid_intrv) as f64).floor() as i32 + 1;
        ixg_start = if 0 > ixg_start { 0 } else { ixg_start };
        let mut ixg_end =
            ((nxb as f64 - 1. - x_grid_strt as f64) / x_grid_intrv as f64).ceil() as i32;
        ixg_end = if nx_grid < ixg_end { nx_grid } else { ixg_end };
        let mut iyg_start = ((-y_grid_strt / y_grid_intrv) as f64).floor() as i32 + 1;
        iyg_start = if 0 > iyg_start { 0 } else { iyg_start };
        let mut iyg_end =
            ((nyb as f64 - 1. - y_grid_strt as f64) / y_grid_intrv as f64).ceil() as i32;
        iyg_end = if ny_grid < iyg_end { ny_grid } else { iyg_end };

        for iygrid in iyg_start..=iyg_end {
            /* Get indexes of grid points in Y on low and high side of block, and y
            coordinates */
            let indy = [
                if 0 > iygrid - 1 { 0 } else { iygrid - 1 },
                if ny_grid - 1 < iygrid {
                    ny_grid - 1
                } else {
                    iygrid
                },
            ];
            let mut ylim = [
                y_grid_strt + y_grid_intrv * (iygrid - 1) as f32,
                y_grid_strt + y_grid_intrv * iygrid as f32,
            ];
            let mut iylim = [(ylim[0] as f64).ceil() as i32, 0];
            if iygrid == iyg_start {
                iylim[0] = 0;
                ylim[0] = (if 0. < ylim[0] as f64 {
                    0.
                } else {
                    ylim[0] as f64
                }) as f32;
            }
            iylim[1] = (ylim[1] as f64).ceil() as i32 - 1;
            if iygrid == iyg_end {
                iylim[1] = nyb - 1;
                let high = nyb as f64 - 0.999;
                ylim[1] = (if high > ylim[1] as f64 {
                    high
                } else {
                    ylim[1] as f64
                }) as f32;
            }

            /* Loop on X blocks, get indexes and limiting coordinates in X */
            for ixgrid in ixg_start..=ixg_end {
                let indx = [
                    if 0 > ixgrid - 1 { 0 } else { ixgrid - 1 },
                    if nx_grid - 1 < ixgrid {
                        nx_grid - 1
                    } else {
                        ixgrid
                    },
                ];
                let mut xlim = [
                    x_grid_strt + x_grid_intrv * (ixgrid - 1) as f32,
                    x_grid_strt + x_grid_intrv * ixgrid as f32,
                ];
                let mut ixlim = [(xlim[0] as f64).ceil() as i32, 0];
                if ixgrid == ixg_start {
                    ixlim[0] = 0;
                    xlim[0] = (if 0. < xlim[0] as f64 {
                        0.
                    } else {
                        xlim[0] as f64
                    }) as f32;
                }
                ixlim[1] = (xlim[1] as f64).ceil() as i32 - 1;
                if ixgrid == ixg_end {
                    ixlim[1] = nxb - 1;
                    let high = nxb as f64 - 0.999;
                    xlim[1] = (if high > xlim[1] as f64 {
                        high
                    } else {
                        xlim[1] as f64
                    }) as f32;
                }

                /* Evaluate mapping of each corner point and see if inside */
                let mut all_in = 1;
                let mut xmap = [[0.0_f32; 2]; 2];
                let mut ymap = [[0.0_f32; 2]; 2];
                for iy in 0..2usize {
                    for ix in 0..2usize {
                        let index = (indx[ix] + ixg_dim * indy[iy]) as usize;
                        let x = xlim[ix] + dx_grid[index];
                        let y = ylim[iy] + dy_grid[index];
                        xmap[ix][iy] = x * a11 + y * a12 + ox;
                        ymap[ix][iy] = x * a21 + y * a22 + oy;
                        if (xmap[ix][iy] as f64) < 1.
                            || xmap[ix][iy] >= (nxa - 2) as f32
                            || (ymap[ix][iy] as f64) < 1.
                            || ymap[ix][iy] >= (nya - 2) as f32
                        {
                            all_in = 0;
                        }
                    }
                }
                let xbox = xlim[1] - xlim[0];
                let ybox = ylim[1] - ylim[0];

                /* Loop on lines, set up start coordinate and steps */
                for j in iylim[0]..=iylim[1] {
                    let gridfy = (j as f32 - ylim[0]) / ybox;
                    let xstep = (((1. - gridfy as f64) * (xmap[1][0] - xmap[0][0]) as f64
                        + (gridfy * (xmap[1][1] - xmap[0][1])) as f64)
                        / xbox as f64) as f32;
                    let ystep = (((1. - gridfy as f64) * (ymap[1][0] - ymap[0][0]) as f64
                        + (gridfy * (ymap[1][1] - ymap[0][1])) as f64)
                        / xbox as f64) as f32;
                    let x = ((1. - gridfy as f64) * xmap[0][0] as f64
                        + (gridfy * xmap[0][1]) as f64
                        + (xstep * (ixlim[0] as f32 - xlim[0])) as f64)
                        as f32;
                    let y = ((1. - gridfy as f64) * ymap[0][0] as f64
                        + (gridfy * ymap[0][1]) as f64
                        + (ystep * (ixlim[0] as f32 - xlim[0])) as f64)
                        as f32;

                    /* Loop across lines in different cases */
                    // `buf = &bray[ixlim[0] + (size_t)j * nxb]` then `*buf++`.
                    let buf = (ixlim[0] as i64 + j as i64 * nxb as i64) as usize;
                    if linear != 0 {
                        /* Linear or nearest neighbor interpolation */
                        if all_in != 0 {
                            if linear > 0 {
                                for i in 0..=(ixlim[1] - ixlim[0]) {
                                    let xp = x + i as f32 * xstep;
                                    let yp = y + i as f32 * ystep;
                                    let ixp = xp as i32;
                                    let iyp = yp as i32;
                                    let fx = xp - ixp as f32;
                                    let fy = yp - iyp as f32;
                                    let ind = (ixp as i64 + iyp as i64 * llnxa) as usize;
                                    bray[buf + i as usize] = ((1. - fy as f64)
                                        * ((1. - fx as f64) * array[ind] as f64
                                            + (fx * array[ind + 1]) as f64)
                                        + fy as f64
                                            * ((1. - fx as f64)
                                                * array[ind + llnxa as usize] as f64
                                                + (fx * array[ind + llnxa as usize + 1]) as f64))
                                        as f32;
                                }
                            } else {
                                for i in 0..=(ixlim[1] - ixlim[0]) {
                                    let xp = x + i as f32 * xstep;
                                    let yp = y + i as f32 * ystep;
                                    let ixp = (xp as f64 + 0.5) as i32;
                                    let iyp = (yp as f64 + 0.5) as i32;
                                    let ind = (ixp as i64 + iyp as i64 * llnxa) as usize;
                                    bray[buf + i as usize] = array[ind];
                                }
                            }
                        } else {
                            for i in 0..=(ixlim[1] - ixlim[0]) {
                                let xp = x + i as f32 * xstep;
                                let yp = y + i as f32 * ystep;
                                if linear > 0 {
                                    let ixp = xp as i32;
                                    let iyp = yp as i32;
                                    if ixp >= 0 && ixp < nxa - 1 && iyp >= 0 && iyp < nya - 1 {
                                        let fx = xp - ixp as f32;
                                        let fy = yp - iyp as f32;
                                        let ind = (ixp as i64 + iyp as i64 * llnxa) as usize;
                                        bray[buf + i as usize] = ((1. - fy as f64)
                                            * ((1. - fx as f64) * array[ind] as f64
                                                + (fx * array[ind + 1]) as f64)
                                            + fy as f64
                                                * ((1. - fx as f64)
                                                    * array[ind + llnxa as usize] as f64
                                                    + (fx * array[ind + llnxa as usize + 1])
                                                        as f64))
                                            as f32;
                                    } else {
                                        bray[buf + i as usize] = dmean;
                                    }
                                } else {
                                    let ixp = (xp as f64 + 0.5) as i32;
                                    let iyp = (yp as f64 + 0.5) as i32;
                                    bray[buf + i as usize] =
                                        if ixp >= 0 && ixp < nxa && iyp >= 0 && iyp < nya {
                                            array[(ixp as i64 + iyp as i64 * llnxa) as usize]
                                        } else {
                                            dmean
                                        };
                                }
                            }
                        }
                    } else {
                        /* Cubic */
                        for i in 0..=(ixlim[1] - ixlim[0]) {
                            let xp = x + i as f32 * xstep;
                            let yp = y + i as f32 * ystep;
                            let ixp = xp as i32;
                            let iyp = yp as i32;
                            if all_in != 0
                                || (ixp >= 1 && ixp < nxa - 2 && iyp >= 1 && iyp < nya - 2)
                            {
                                let dx = xp - ixp as f32;
                                let dy = yp - iyp as f32;
                                let dxm1 = (dx as f64 - 1.) as f32;
                                let dxdxm1 = dx * dxm1;
                                let fx1 = -dxm1 * dxdxm1;
                                let fx4 = dx * dxdxm1;
                                let fx2 = (1. + (dx * dx) as f64 * (dx as f64 - 2.)) as f32;
                                let fx3 = (dx as f64 * (1. - dxdxm1 as f64)) as f32;

                                let dym1 = (dy as f64 - 1.) as f32;
                                let dydym1 = dy * dym1;
                                let ind = (ixp as i64 + iyp as i64 * llnxa) as usize;
                                let indmnxa = ind - llnxa as usize;
                                let indpnxa = ind + llnxa as usize;
                                let indpnxa2 = ind + 2 * llnxa as usize;
                                let v1 = fx1 * array[indmnxa - 1]
                                    + fx2 * array[indmnxa]
                                    + fx3 * array[indmnxa + 1]
                                    + fx4 * array[indmnxa + 2];
                                let v2 = fx1 * array[ind - 1]
                                    + fx2 * array[ind]
                                    + fx3 * array[ind + 1]
                                    + fx4 * array[ind + 2];
                                let v3 = fx1 * array[indpnxa - 1]
                                    + fx2 * array[indpnxa]
                                    + fx3 * array[indpnxa + 1]
                                    + fx4 * array[indpnxa + 2];
                                let v4 = fx1 * array[indpnxa2 - 1]
                                    + fx2 * array[indpnxa2]
                                    + fx3 * array[indpnxa2 + 1]
                                    + fx4 * array[indpnxa2 + 2];

                                bray[buf + i as usize] = ((-dym1 * dydym1 * v1) as f64
                                    + (1. + (dy * dy) as f64 * (dy as f64 - 2.)) * v2 as f64
                                    + dy as f64 * (1. - dydym1 as f64) * v3 as f64
                                    + (dy * dydym1 * v4) as f64)
                                    as f32;
                            } else {
                                bray[buf + i as usize] = dmean;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// C Fortran wrapper `warpinterp` (`warpwrapfort.c:266`).
#[allow(clippy::too_many_arguments)]
pub fn warpinterp_fortran(
    array: &[f32],
    bray: &mut [f32],
    nxa: &i32,
    nya: &i32,
    nxb: &i32,
    nyb: &i32,
    amat: &[f32],
    xc: &f32,
    yc: &f32,
    xt: &f32,
    yt: &f32,
    scale: &f32,
    dmean: &f32,
    linear: &i32,
    lin_first: &i32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: &i32,
    nx_grid: &i32,
    ny_grid: &i32,
    x_grid_strt: &f32,
    y_grid_strt: &f32,
    x_grid_intrv: &f32,
    y_grid_intrv: &f32,
) {
    // `warpwrapfort.c:266-274` transposes the Fortran `amat(2,2)` into the
    // C row-major `amat[2][2]` before calling.
    let cmat = [[amat[0], amat[1]], [amat[2], amat[3]]];
    warp_interp(
        array,
        bray,
        *nxa,
        *nya,
        *nxb,
        *nyb,
        &cmat,
        *xc,
        *yc,
        *xt,
        *yt,
        *scale,
        *dmean,
        *linear,
        *lin_first,
        dx_grid,
        dy_grid,
        *ixg_dim,
        *nx_grid,
        *ny_grid,
        *x_grid_strt,
        *y_grid_strt,
        *x_grid_intrv,
        *y_grid_intrv,
    );
}
