//! Translation of `IMOD/libcfshr/sdsearch.c`: `montBigSearch`, `montbigsearch`,
//! `montSdCalc`, `montsdcalc` — measuring the SD of the difference between two
//! images and searching for the displacement that minimises it.
//!
//! The C passes `sd`, `dden`, `dxMin`, `dyMin`, `sdMin` and `ddenMin` by
//! pointer and they are `&mut` here.  In particular `montSdCalc` writes
//! `*dden` only when it compared at least one pixel (`sdsearch.c:308-309`), so
//! `dden` is in/out: a call that compares nothing leaves the caller's value.

/// Original: `montBigSearch` (`sdsearch.c:36-203`).
///
/// `checked` is `unsigned char checked[2 * CHKLIM + 1][2 * CHKLIM + 1]` on
/// the C stack, cleared only inside `±ndLim`; the translation zero-fills all
/// of it, which reads the same because every access is guarded by
/// `abs <= ndLim`.  `sd` and `dden` are uninitialised locals in the C.
/// `montSdCalc` always writes `sd` but writes `dden` only when `nsum > 0`.
/// A call comparing no pixel returns `sd = 9999.`, which is taken as a new
/// minimum only if `*sdMin` exceeds 9999; if that happens before any call has
/// written `dden`, native stores stack residue where this stores 0.
#[allow(clippy::too_many_arguments)]
pub fn mont_big_search(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ix_box0: i32,
    iy_box0: i32,
    ix_box1: i32,
    iy_box1: i32,
    dx_min: &mut f32,
    dy_min: &mut f32,
    sd_min: &mut f32,
    dden_min: &mut f32,
    num_iter: i32,
    lim_step: i32,
) {
    const CHKLIM: i32 = 100;
    let mut checked = [[0_u8; (2 * CHKLIM + 1) as usize]; (2 * CHKLIM + 1) as usize];
    let mut x_changed: i32;
    let mut y_changed: i32;
    let mut keepon: i32;
    let mut idx_min: i32;
    let mut idy_min: i32;
    let mut new_idx: i32;
    let mut new_idy: i32;
    let mut ndxy: i32;
    let mut nd_lim: i32;
    let mut idir: i32;
    let mut abs_new_idx: i32;
    let mut abs_new_idy: i32;
    let dxcen: f32;
    let dycen: f32;
    let dxy: f32;
    let mut sd = 0.0_f32;
    let mut dden = 0.0_f32;

    dxcen = *dx_min; /* all moves relative to center */
    dycen = *dy_min; /* at the initial position */
    mont_sd_calc(
        array, brray, nx, ny, ix_box0, iy_box0, ix_box1, iy_box1, *dx_min, *dy_min, sd_min,
        dden_min,
    );
    ndxy = 1; /* initial # of steps to move by */
    for _iter in 0..num_iter - 1 {
        ndxy *= 2;
    }
    dxy = (1. / ndxy as f64) as f32; /* final true step size */
    nd_lim = ndxy * lim_step; /* limiting # of steps */
    if nd_lim > CHKLIM {
        nd_lim = CHKLIM;
    }

    /*  set grid to unchecked state except at center */
    // The source clears the square column by column (`checked[j][i]` with
    // `j` inner); the order of the stores is not observable, so each row of
    // the square is cleared as one slice.
    for j in -nd_lim..=nd_lim {
        checked[(j + CHKLIM) as usize][(CHKLIM - nd_lim) as usize..=(CHKLIM + nd_lim) as usize]
            .fill(0);
    }
    checked[CHKLIM as usize][CHKLIM as usize] = 1;

    idx_min = 0;
    idy_min = 0;
    for _iter in 1..=num_iter {
        x_changed = 1;
        y_changed = 1;

        /* keep doing the x-y-diagonal series as long as something changes */
        while x_changed != 0 || y_changed != 0 {
            /*     move in x until reach minimum */
            /* first try a positive step */
            x_changed = 0;
            idir = 1;
            new_idx = idx_min + idir * ndxy;
            abs_new_idx = if new_idx < 0 { -new_idx } else { new_idx };
            if abs_new_idx <= nd_lim
                && checked[(new_idx + CHKLIM) as usize][(idy_min + CHKLIM) as usize] == 0
            {
                mont_sd_calc(
                    array,
                    brray,
                    nx,
                    ny,
                    ix_box0,
                    iy_box0,
                    ix_box1,
                    iy_box1,
                    dxcen + new_idx as f32 * dxy,
                    dycen + idy_min as f32 * dxy,
                    &mut sd,
                    &mut dden,
                );

                /* mark as checked*/
                checked[(new_idx + CHKLIM) as usize][(idy_min + CHKLIM) as usize] = 1;
                if sd < *sd_min {
                    *sd_min = sd;
                    *dden_min = dden;
                    x_changed = 1;
                    idx_min = new_idx;
                } else {
                    /* if positive step does no good */
                    idir = -1; /* switch direction */
                }
            } else {
                /* or if positive has been */
                idir = -1; /* checked or is too far, switch */
            }

            /*   now, in whichever direction is selected, keep on moving
            until get to a higher value or edge or area */
            keepon = 1;
            while keepon != 0 {
                new_idx = idx_min + idir * ndxy;
                abs_new_idx = if new_idx < 0 { -new_idx } else { new_idx };
                if abs_new_idx <= nd_lim
                    && checked[(new_idx + CHKLIM) as usize][(idy_min + CHKLIM) as usize] == 0
                {
                    mont_sd_calc(
                        array,
                        brray,
                        nx,
                        ny,
                        ix_box0,
                        iy_box0,
                        ix_box1,
                        iy_box1,
                        dxcen + new_idx as f32 * dxy,
                        dycen + idy_min as f32 * dxy,
                        &mut sd,
                        &mut dden,
                    );
                    checked[(new_idx + CHKLIM) as usize][(idy_min + CHKLIM) as usize] = 1;

                    if sd < *sd_min {
                        *sd_min = sd;
                        *dden_min = dden;
                        x_changed = 1;
                        idx_min = new_idx;
                    } else {
                        /* if value is higher, stop going */
                        keepon = 0;
                    }
                } else {
                    /* or if already checked, or too */
                    keepon = 0; /* far, stop going */
                }
            }

            /*  follow exact same procedure in y direction */
            y_changed = 0;
            idir = 1;
            new_idy = idy_min + idir * ndxy;
            abs_new_idy = if new_idy < 0 { -new_idy } else { new_idy };
            if abs_new_idy <= nd_lim
                && checked[(idx_min + CHKLIM) as usize][(new_idy + CHKLIM) as usize] == 0
            {
                mont_sd_calc(
                    array,
                    brray,
                    nx,
                    ny,
                    ix_box0,
                    iy_box0,
                    ix_box1,
                    iy_box1,
                    dxcen + idx_min as f32 * dxy,
                    dycen + new_idy as f32 * dxy,
                    &mut sd,
                    &mut dden,
                );
                checked[(idx_min + CHKLIM) as usize][(new_idy + CHKLIM) as usize] = 1;
                if sd < *sd_min {
                    *sd_min = sd;
                    *dden_min = dden;
                    y_changed = 1;
                    idy_min = new_idy;
                } else {
                    idir = -1;
                }
            } else {
                idir = -1;
            }

            keepon = 1;
            while keepon != 0 {
                new_idy = idy_min + idir * ndxy;
                abs_new_idy = if new_idy < 0 { -new_idy } else { new_idy };
                if abs_new_idy <= nd_lim
                    && checked[(idx_min + CHKLIM) as usize][(new_idy + CHKLIM) as usize] == 0
                {
                    mont_sd_calc(
                        array,
                        brray,
                        nx,
                        ny,
                        ix_box0,
                        iy_box0,
                        ix_box1,
                        iy_box1,
                        dxcen + idx_min as f32 * dxy,
                        dycen + new_idy as f32 * dxy,
                        &mut sd,
                        &mut dden,
                    );
                    checked[(idx_min + CHKLIM) as usize][(new_idy + CHKLIM) as usize] = 1;
                    if sd < *sd_min {
                        *sd_min = sd;
                        *dden_min = dden;
                        y_changed = 1;
                        idy_min = new_idy;
                    } else {
                        keepon = 0;
                    }
                } else {
                    keepon = 0;
                }
            }

            /*  if no change above, check the 4 corners */
            let mut ix_dir = -1;
            while ix_dir <= 1 {
                let mut iy_dir = -1;
                while iy_dir <= 1 {
                    /* keep checking corners only if nothing changed yet*/
                    if !(x_changed != 0 || y_changed != 0) {
                        new_idx = idx_min + ix_dir * ndxy;
                        new_idy = idy_min + iy_dir * ndxy;
                        abs_new_idy = if new_idy < 0 { -new_idy } else { new_idy };
                        abs_new_idx = if new_idx < 0 { -new_idx } else { new_idx };
                        if abs_new_idy <= nd_lim
                            && abs_new_idx <= nd_lim
                            && checked[(new_idx + CHKLIM) as usize][(new_idy + CHKLIM) as usize]
                                == 0
                        {
                            mont_sd_calc(
                                array,
                                brray,
                                nx,
                                ny,
                                ix_box0,
                                iy_box0,
                                ix_box1,
                                iy_box1,
                                dxcen + new_idx as f32 * dxy,
                                dycen + new_idy as f32 * dxy,
                                &mut sd,
                                &mut dden,
                            );
                            checked[(new_idx + CHKLIM) as usize][(new_idy + CHKLIM) as usize] = 1;
                            if sd < *sd_min {
                                *sd_min = sd;
                                *dden_min = dden;
                                x_changed = 1;
                                y_changed = 1;
                                idx_min = new_idx;
                                idy_min = new_idy;
                            }
                        }
                    }
                    iy_dir += 2;
                }
                ix_dir += 2;
            }

            /*  if nothing ever changed in last loop, drop out of loop */
        }
        ndxy /= 2; /*  and cut the step size */
    }
    *dx_min = dxcen + idx_min as f32 * dxy;
    *dy_min = dycen + idy_min as f32 * dxy;
}

/// Original: `montbigsearch` (`sdsearch.c:208-214`), the Fortran wrapper to
/// `montBigSearch`, where the array index limits are numbered from 1.
#[allow(clippy::too_many_arguments)]
pub fn montbigsearch(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ix_box0: i32,
    iy_box0: i32,
    ix_box1: i32,
    iy_box1: i32,
    dx_min: &mut f32,
    dy_min: &mut f32,
    sd_min: &mut f32,
    dden_min: &mut f32,
    num_iter: i32,
    lim_step: i32,
) {
    mont_big_search(
        array,
        brray,
        nx,
        ny,
        ix_box0 - 1,
        iy_box0 - 1,
        ix_box1 - 1,
        iy_box1 - 1,
        dx_min,
        dy_min,
        sd_min,
        dden_min,
        num_iter,
        lim_step,
    );
}

/// Original: `montSdCalc` (`sdsearch.c:228-312`).
///
/// `diff` is a C `float`, so `diff * diff` is a single-precision product
/// widened for the double `sumsq`; `c5 = 1. - dxsq - dysq` is evaluated in
/// double; `B3DNINT(x)` is `(int)floor(x + 0.5)` with a double `0.5`.
/// `*dden` is written only when `nsum > 0`.
#[allow(clippy::too_many_arguments)]
pub fn mont_sd_calc(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ix_box0: i32,
    iy_box0: i32,
    ix_box1: i32,
    iy_box1: i32,
    dx: f32,
    dy: f32,
    sd: &mut f32,
    dden: &mut f32,
) {
    let mut nsum: i32;
    let mut ixp: i32;
    let mut iyp: i32;
    let idx: i32;
    let idy: i32;
    let ixb0: i32;
    let iyb0: i32;
    let ixb1: i32;
    let iyb1: i32;
    let mut iya: i32;
    let mut iypp1: i32;
    let mut iypm1: i32;
    let xp: f32;
    let yp: f32;
    let dxint: f32;
    let dyint: f32;
    let mut diff: f32;
    let c2: f32;
    let c8: f32;
    let c6: f32;
    let c4: f32;
    let c5: f32;
    let dxsq: f32;
    let dysq: f32;
    let mut binterp: f32;
    let mut sum: f64;
    let mut sumsq: f64;

    nsum = 0;
    sum = 0.;
    sumsq = 0.;
    *sd = 9999.;

    /* convert dx and dy into integer idx,idy and fractions dxint, dyint
    use high end of box to be sure to get right dxint and dyint values */
    yp = iy_box1 as f32 + dy;
    iyp = (yp as f64 + 0.5).floor() as i32;
    dyint = yp - iyp as f32;
    idy = (dy - dyint) as i32;
    xp = ix_box1 as f32 + dx;
    ixp = (xp as f64 + 0.5).floor() as i32;
    dxint = xp - ixp as f32;
    idx = (dx - dxint) as i32;
    if dxint == 0. && dyint == 0. {
        /* If dx, dy are integers, don't need interpolation */
        /* compute limits for box in brray */
        ixb0 = if 0 < ix_box0 + idx { ix_box0 + idx } else { 0 };
        ixb1 = if nx - 1 < ix_box1 + idx {
            nx - 1
        } else {
            ix_box1 + idx
        };
        iyb0 = if 0 < iy_box0 + idy { iy_box0 + idy } else { 0 };
        iyb1 = if ny - 1 < iy_box1 + idy {
            ny - 1
        } else {
            iy_box1 + idy
        };

        if ixb0 <= ixb1 && ixb0 < nx && ixb1 >= 0 && iyb0 <= iyb1 && iyb0 < ny && iyb1 >= 0 {
            // Each row of the box is taken as one slice of `brray` and one of
            // `array` (one bounds check per row instead of two per pixel); the
            // elements are visited in the source's order and the sums formed
            // in the same order, so no value changes.
            let w = (ixb1 + 1 - ixb0) as usize;
            for iyb in iyb0..=iyb1 {
                iya = iyb - idy;
                let brow = &brray[(ixb0 + iyb * nx) as usize..][..w];
                let arow = &array[(ixb0 - idx + iya * nx) as usize..][..w];
                for (&b, &a) in brow.iter().zip(arow) {
                    diff = b - a;
                    sum += diff as f64;
                    sumsq += (diff * diff) as f64;
                }
            }
            nsum = (ixb1 + 1 - ixb0) * (iyb1 + 1 - iyb0);
        }
    } else {
        /* compute the coefficients for quadratic interpolation */
        dysq = dyint * dyint;
        c8 = (0.5 * (dysq + dyint) as f64) as f32;
        c2 = (0.5 * (dysq - dyint) as f64) as f32;
        dxsq = dxint * dxint;
        c6 = (0.5 * (dxsq + dxint) as f64) as f32;
        c4 = (0.5 * (dxsq - dxint) as f64) as f32;
        c5 = (1. - dxsq as f64 - dysq as f64) as f32;

        ixb0 = if 1 < ix_box0 + idx { ix_box0 + idx } else { 1 };
        ixb1 = if nx - 2 < ix_box1 + idx {
            nx - 2
        } else {
            ix_box1 + idx
        };
        iyb0 = if 1 < iy_box0 + idy { iy_box0 + idy } else { 1 };
        iyb1 = if ny - 2 < iy_box1 + idy {
            ny - 2
        } else {
            iy_box1 + idy
        };
        if ixb0 <= ixb1 && ixb0 < nx - 1 && ixb1 > 0 && iyb0 <= iyb1 && iyb0 < ny - 1 && iyb1 > 0 {
            // Row slices as in the integer branch: the centre row with one
            // element either side (`ixp - 1`, `ixp + 1`), the rows below and
            // above, and the `array` row.  Same elements, same order, same
            // operation order in `binterp`, so no value changes.
            let w = (ixb1 + 1 - ixb0) as usize;
            iyp = iyb0;
            while iyp <= iyb1 {
                iya = iyp - idy;
                iypp1 = iyp + 1;
                iypm1 = iyp - 1;
                let bcen = &brray[(ixb0 - 1 + iyp * nx) as usize..][..w + 2];
                let bdn = &brray[(ixb0 + iypm1 * nx) as usize..][..w];
                let bup = &brray[(ixb0 + iypp1 * nx) as usize..][..w];
                let arow = &array[(ixb0 - idx + iya * nx) as usize..][..w];
                for k in 0..w {
                    binterp = c5 * bcen[k + 1]
                        + c4 * bcen[k]
                        + c6 * bcen[k + 2]
                        + c2 * bdn[k]
                        + c8 * bup[k];
                    diff = binterp - arow[k];
                    sum += diff as f64;
                    sumsq += (diff * diff) as f64;
                }
                ixp = ixb1 + 1;
                iyp += 1;
            }
            nsum = (ixb1 + 1 - ixb0) * (iyb1 + 1 - iyb0);
        }
    }
    if nsum > 0 {
        *dden = (sum / nsum as f64) as f32;
    }
    if nsum > 1 {
        *sd = ((sumsq - sum * sum / nsum as f64) / (nsum as f64 - 1.)).sqrt() as f32;
    }
}

/// Original: `montsdcalc` (`sdsearch.c:317-322`), the Fortran wrapper to
/// `montSdCalc`, where the array index limits are numbered from 1.
#[allow(clippy::too_many_arguments)]
pub fn montsdcalc(
    array: &[f32],
    brray: &[f32],
    nx: i32,
    ny: i32,
    ix_box0: i32,
    iy_box0: i32,
    ix_box1: i32,
    iy_box1: i32,
    dx: f32,
    dy: f32,
    sd: &mut f32,
    dden: &mut f32,
) {
    mont_sd_calc(
        array,
        brray,
        nx,
        ny,
        ix_box0 - 1,
        iy_box0 - 1,
        ix_box1 - 1,
        iy_box1 - 1,
        dx,
        dy,
        sd,
        dden,
    );
}
