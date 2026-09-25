//! Translation of `IMOD/flib/tilt/projsumlocal.f90`.
//!
//! Array and index conventions are described in `super` (`tilt/mod.rs`).

/// Original `projSumLocal` (`projsumlocal.f90:5`).
///
/// Assesses making a regular step (a jump) in the local reprojection, and then
/// does the steps of the jump.  This operation took twice as in C++ with the
/// Intel compiler as with the Intel Fortran compiler
///
/// All reals are `real*4` except `sum` (`real*8`): each interpolated term is
/// formed in single precision and widened as it is added, `(sum + a) + b`.
/// Integer-to-real mixes (`ithickReproj - ycenAdj - 1`, `numJump * delZ`,
/// `indJump * delX`) convert the integer to single.
///
/// `ind = max(1., min(float(numWarpDelz), xx / dxWarpDelz))` is gfortran's
/// `MIN`/`MAX`, whose result with a NaN operand is unspecified; it is written
/// here as `f32::min`/`max`, which agree with it for every non-NaN value.
///
/// If `numJump <= 0` while `tryJump` stays true the source's `do while` never
/// terminates (nothing in the loop changes); that is reproduced, not guarded.
#[allow(clippy::too_many_arguments)]
pub fn proj_sum_local(
    xx: &mut f32,
    yy: &mut f32,
    zz: &mut f32,
    sum: &mut f64,
    xproj: f32,
    yproj: f32,
    array: &[f32],
    x_proj_fs: &[f32],
    x_proj_zs: &[f32],
    y_proj_fs: &[f32],
    y_proj_zs: &[f32],
    num_warp_delz: i32,
    dx_warp_delz: f32,
    warp_delz: &[f32],
    ithick_reproj: i32,
    sin_beta: f32,
    nx_load: i32,
    in_load_start: i32,
    in_load_end: i32,
    in_plane_size: i32,
    z_jump: f32,
    ycen_adj: f32,
) {
    let mut del_x: f32 = 0.;
    let mut del_y: f32 = 0.;
    let mut try_jump = true;
    while try_jump {
        //
        // If jumping is OK, save the current position and compute
        // how many steps can be jumped, stopping below the top
        let xx_good = *xx;
        let yy_good = *yy;
        let zz_good = *zz;
        let mut ind = fortran_int!(f32: 1f32.max((num_warp_delz as f32).min(*xx / dx_warp_delz)));
        let del_z = warp_delz[(ind - 1) as usize];
        let mut num_jump = fortran_int!(f32: z_jump / del_z);
        if *zz + z_jump > ithick_reproj as f32 - ycen_adj - 1f32 {
            num_jump = fortran_int!(f32: (ithick_reproj as f32 - ycen_adj - 1f32 - *zz) / del_z);
            try_jump = false;
        }
        if num_jump > 0 {
            //
            // Make the jump, find the projecting point;
            // if it's out of bounds restore last point
            *zz += num_jump as f32 * del_z;
            *xx += num_jump as f32 * sin_beta;
            loaded_projecting_point(
                xproj,
                yproj,
                *zz,
                nx_load,
                in_load_start,
                in_load_end,
                x_proj_fs,
                x_proj_zs,
                y_proj_fs,
                y_proj_zs,
                xx,
                yy,
            );
            if *yy < in_load_start as f32
                || *yy > in_load_end as f32
                || *xx < 1.
                || *xx >= nx_load as f32
            {
                num_jump = 0;
                *xx = xx_good;
                *yy = yy_good;
                *zz = zz_good;
                try_jump = false;
            } else {
                del_x = (*xx - xx_good) / num_jump as f32;
                del_y = (*yy - yy_good) / num_jump as f32;
            }
        }
        //
        // Loop on points from last one to final one
        //
        for ind_jump in 1..=num_jump {
            *xx = xx_good + ind_jump as f32 * del_x;
            *yy = yy_good + ind_jump as f32 * del_y;
            *zz = zz_good + ind_jump as f32 * del_z;
            let ix = fortran_int!(f32: *xx);
            let fx = *xx - ix as f32;
            let one_mfx = 1. - fx;
            let iy = fortran_int!(f32: *yy).min(in_load_end - 1);
            let fy = *yy - iy as f32;
            let one_mfy = 1. - fy;
            // BUG again, need ycenAdj
            let iz = fortran_int!(f32: *zz + ycen_adj);
            let fz = *zz + ycen_adj - iz as f32;
            let one_mfz = 1. - fz;
            let d11 = one_mfx * one_mfy;
            let d12 = one_mfx * fy;
            let d21 = fx * one_mfy;
            let d22 = fx * fy;
            ind = in_plane_size * (iy - in_load_start) + (iz - 1) * nx_load + ix;
            // The eight reads below are array(ind + {0, 1} + {0, inPlaneSize}
            // + {0, nxLoad}); they are all in bounds exactly when the smallest
            // and largest are, so one test stands for the eight.
            let lo = ind as i64 - 1 + (in_plane_size as i64).min(0) + (nx_load as i64).min(0);
            let hi = ind as i64 + (in_plane_size as i64).max(0) + (nx_load as i64).max(0);
            if lo < 0 || hi >= array.len() as i64 {
                panic!(
                    "projSumLocal: array index {lo}..={hi} outside the array (Fortran would read out of bounds)"
                );
            }
            // SAFETY: every argument below is ind + {0, 1} + {0, inPlaneSize} +
            // {0, nxLoad}, so i - 1 lies in [lo, hi], checked just above.
            let a = |i: i32| unsafe { *array.get_unchecked((i - 1) as usize) };
            *sum = *sum
                + (one_mfz
                    * (d11 * a(ind)
                        + d12 * a(ind + in_plane_size)
                        + d21 * a(ind + 1)
                        + d22 * a(ind + in_plane_size + 1))) as f64
                + (fz
                    * (d11 * a(ind + nx_load)
                        + d12 * a(ind + in_plane_size + nx_load)
                        + d21 * a(ind + 1 + nx_load)
                        + d22 * a(ind + in_plane_size + 1 + nx_load))) as f64;
            // fx = xx
            // fy = yy
            // call loadedProjectingPoint(xproj, yproj, zz, &
            // nxload, inloadstr, inloadend, fx, fy)
            // diffxmax = max(diffxmax , abs(fx - xx))
            // diffymax = max(diffymax , abs(fy - yy))
        }
    }
}

/// Original `loadedProjectingPoint` (`projsumlocal.f90:99`).
///
/// Finds loaded point that projects to xproj, yproj at centered Z value
/// zz, using stored values for [xy]zfac[fv].
/// Takes starting value in xx, yy and returns found value.
/// X coordinate needs to be a loaded X index
/// Y coordinate yy is in slices of reconstruction, yproj in original proj
/// And this also runs slower in C, aside from being needed to call from projSumLocal
///
/// All `real*4`; `nxLoad + 1` in the bound test is an integer sum converted to
/// single, `inLoadStart - 1.` a single-precision difference.
#[allow(clippy::too_many_arguments)]
pub fn loaded_projecting_point(
    xproj: f32,
    yproj: f32,
    zz: f32,
    nx_load: i32,
    in_load_start: i32,
    in_load_end: i32,
    x_proj_fs: &[f32],
    x_proj_zs: &[f32],
    y_proj_fs: &[f32],
    y_proj_zs: &[f32],
    xx: &mut f32,
    yy: &mut f32,
) {
    // print *,'Finding proj pt to', xproj, yproj, zz
    let mut iter = 0;
    let mut if_done = 0;
    while if_done == 0 && iter < 5 {
        let mut ix = xx.floor() as i32;
        let mut iy = yy.floor() as i32;
        let mut if_out = 0;
        if ix < 1 || ix >= nx_load || iy < in_load_start || iy >= in_load_end {
            if_out = 1;
            ix = (nx_load - 1).min(1.max(ix));
            iy = (in_load_end - 1).min(in_load_start.max(iy));
        }
        let ind = nx_load * (iy - in_load_start) + ix;
        let i11 = (ind - 1) as usize;
        let i21 = ind as usize;
        let i12 = (ind + nx_load - 1) as usize;
        let xp11 = x_proj_fs[i11] + x_proj_zs[i11] * zz;
        let yp11 = y_proj_fs[i11] + y_proj_zs[i11] * zz;
        let xp21 = x_proj_fs[i21] + x_proj_zs[i21] * zz;
        let yp21 = y_proj_fs[i21] + y_proj_zs[i21] * zz;
        let xp12 = x_proj_fs[i12] + x_proj_zs[i12] * zz;
        let yp12 = y_proj_fs[i12] + y_proj_zs[i12] * zz;
        let x_err = xproj - xp11;
        let y_err = yproj - yp11;
        let dxpx = xp21 - xp11;
        let dxpy = xp12 - xp11;
        let dypx = yp21 - yp11;
        let dypy = yp12 - yp11;
        let den = dxpx * dypy - dxpy * dypx;
        let fx = (x_err * dypy - y_err * dxpy) / den;
        let fy = (dxpx * y_err - dypx * x_err) / den;
        *xx = ix as f32 + fx;
        *yy = iy as f32 + fy;
        if fx > -0.1 && fx < 1.1 && fy > -0.1 && fy < 1.1 {
            if_done = 1;
        }
        if if_out != 0
            && (iter > 0
                || *xx < 0.
                || *xx > (nx_load + 1) as f32
                || *yy < in_load_start as f32 - 1.
                || *yy > in_load_end as f32 + 1.)
        {
            if_done = 1;
        }
        iter += 1;
    }
}
