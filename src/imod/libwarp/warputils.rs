//! Translation of `IMOD/libwarp/warputils.c` and its direct declarations.
#![allow(dead_code)]
#![allow(static_mut_refs)]

use crate::imod::libcfshr::linearxforms::{xf_apply, xf_copy, xf_invert};
use crate::imod::libcfshr::simplestat::ls_fit2;
use crate::imod::libwarp::warpfiles::{
    WARP_CONTROL_PTS, WARP_INVERSE, get_grid_parameters, get_num_warp_points, get_warp_file_size,
    get_warp_grid, grid_size_from_spacing, read_warp_file, set_grid_size_to_make,
};

/// Original `interpolateGrid` (`warputils.c:33`).
pub unsafe fn interpolate_grid(
    x: f32,
    y: f32,
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_interval: f32,
    y_grid_interval: f32,
    dx: *mut f32,
    dy: *mut f32,
) {
    unsafe {
        let xgrid = (x - x_grid_start) / x_grid_interval;
        let mut ixg = xgrid as i32;
        ixg = 0.max((nx_grid - 2).min(ixg));
        let fx_one = 0_f32.max(1_f32.min(xgrid - ixg as f32));
        let fx = 1. - fx_one;
        let ixg_one = ixg + 1;
        let ygrid = (y - y_grid_start) / y_grid_interval;
        let mut iyg = ygrid as i32;
        iyg = 0.max((ny_grid - 2).min(iyg));
        let fy_one = 0_f32.max(1_f32.min(ygrid - iyg as f32));
        let fy = 1. - fy_one;
        let iyg_one = iyg + 1;
        let c00 = fx * fy;
        let c10 = fx_one * fy;
        let c01 = fx * fy_one;
        let c11 = fx_one * fy_one;
        *dx = c00 * *dx_grid.add((ixg + iyg * ixg_dim) as usize)
            + c10 * *dx_grid.add((ixg_one + iyg * ixg_dim) as usize)
            + c01 * *dx_grid.add((ixg + iyg_one * ixg_dim) as usize)
            + c11 * *dx_grid.add((ixg_one + iyg_one * ixg_dim) as usize);
        *dy = c00 * *dy_grid.add((ixg + iyg * ixg_dim) as usize)
            + c10 * *dy_grid.add((ixg_one + iyg * ixg_dim) as usize)
            + c01 * *dy_grid.add((ixg + iyg_one * ixg_dim) as usize)
            + c11 * *dy_grid.add((ixg_one + iyg_one * ixg_dim) as usize);
    }
}

/// Original `findInversePoint` (`warputils.c:70`).
pub unsafe fn find_inverse_point(
    x: f32,
    y: f32,
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_interval: f32,
    y_grid_interval: f32,
    xnew: *mut f32,
    ynew: *mut f32,
    dx: *mut f32,
    dy: *mut f32,
) {
    unsafe {
        let max_iter = 10;
        let change_crit = 0.01_f64;
        let mut xlast = x;
        let mut ylast = y;
        for _ in 0..max_iter {
            interpolate_grid(
                xlast,
                ylast,
                dx_grid,
                dy_grid,
                ixg_dim,
                nx_grid,
                ny_grid,
                x_grid_start,
                y_grid_start,
                x_grid_interval,
                y_grid_interval,
                dx,
                dy,
            );
            *xnew = x - *dx;
            *ynew = y - *dy;
            if ((*xnew - xlast) as f64).abs() < change_crit
                && ((*ynew - ylast) as f64).abs() < change_crit
            {
                break;
            }
            xlast = *xnew;
            ylast = *ynew;
        }
    }
}

/// Original `invertWarpGrid` (`warputils.c:101`).
pub unsafe fn invert_warp_grid(
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_interval: f32,
    y_grid_interval: f32,
    xform: *mut f32,
    xcen: f32,
    ycen: f32,
    dx_inv: *mut f32,
    dy_inv: *mut f32,
    xf_inv: *mut f32,
    rows: i32,
) {
    unsafe {
        xf_invert(
            core::slice::from_raw_parts(xform, (3 * rows) as usize),
            core::slice::from_raw_parts_mut(xf_inv, (3 * rows) as usize),
            rows as usize,
        );
        for iy in 0..ny_grid {
            for ix in 0..nx_grid {
                let mut ygrid = y_grid_start + iy as f32 * y_grid_interval;
                let mut xgrid = x_grid_start + ix as f32 * x_grid_interval;
                (xgrid, ygrid) = xf_apply(
                    core::slice::from_raw_parts(xform, (3 * rows) as usize),
                    xcen,
                    ycen,
                    xgrid,
                    ygrid,
                    rows as usize,
                );
                let (mut xnew, mut ynew, mut dx, mut dy) = (0., 0., 0., 0.);
                find_inverse_point(
                    xgrid,
                    ygrid,
                    dx_grid,
                    dy_grid,
                    ixg_dim,
                    nx_grid,
                    ny_grid,
                    x_grid_start,
                    y_grid_start,
                    x_grid_interval,
                    y_grid_interval,
                    &mut xnew,
                    &mut ynew,
                    &mut dx,
                    &mut dy,
                );
                *dx_inv.add((ix + iy * ixg_dim) as usize) = -dx;
                *dy_inv.add((ix + iy * ixg_dim) as usize) = -dy;
            }
        }
    }
}

/// Original `extractLinearXform` (`warputils.c:238`).
pub unsafe fn extract_linear_xform(
    x_pos: *mut f32,
    y_pos: *mut f32,
    x_vector: *mut f32,
    y_vector: *mut f32,
    n_points: i32,
    xcen: f32,
    ycen: f32,
    new_xvec: *mut f32,
    new_yvec: *mut f32,
    xf_inv: *mut f32,
    rows: i32,
) -> i32 {
    unsafe {
        if n_points < 3 || rows < 2 || rows > 3 {
            return 1;
        }
        let mut ptmp = vec![0.; n_points as usize];
        let mut mat = [0.; 6];
        for i in 0..n_points as usize {
            *new_xvec.add(i) = *x_pos.add(i) - xcen + *x_vector.add(i);
            *new_yvec.add(i) = *y_pos.add(i) - ycen + *y_vector.add(i);
            ptmp[i] = *x_pos.add(i) - xcen;
        }
        ls_fit2(
            new_xvec,
            new_yvec,
            ptmp.as_ptr(),
            n_points,
            &mut mat[0],
            &mut mat[2],
            &mut mat[4],
        );
        for i in 0..n_points as usize {
            ptmp[i] = *y_pos.add(i) - ycen;
        }
        ls_fit2(
            new_xvec,
            new_yvec,
            ptmp.as_ptr(),
            n_points,
            &mut mat[1],
            &mut mat[3],
            &mut mat[5],
        );
        for i in 0..n_points as usize {
            let (xtmp, ytmp) = xf_apply(&mat, 0., 0., *new_xvec.add(i), *new_yvec.add(i), 2);
            *new_xvec.add(i) = xtmp + xcen - *x_pos.add(i);
            *new_yvec.add(i) = ytmp + ycen - *y_pos.add(i);
        }
        xf_copy(
            &mat,
            2,
            core::slice::from_raw_parts_mut(xf_inv, (3 * rows) as usize),
            rows as usize,
        );
        0
    }
}

/// Original `multiplyWarpings` (`warputils.c:141`).
pub unsafe fn multiply_warpings(
    dx_grid1: *mut f32,
    dy_grid1: *mut f32,
    ixg_dim1: i32,
    nx_grid1: i32,
    ny_grid1: i32,
    x_start1: f32,
    y_start1: f32,
    x_intrv1: f32,
    y_intrv1: f32,
    xform1: *mut f32,
    xcen: f32,
    ycen: f32,
    dx_grid2: *mut f32,
    dy_grid2: *mut f32,
    ixg_dim2: i32,
    nx_grid2: i32,
    ny_grid2: i32,
    x_start2: f32,
    y_start2: f32,
    x_intrv2: f32,
    y_intrv2: f32,
    xform2: *mut f32,
    dx_prod: *mut f32,
    dy_prod: *mut f32,
    xf_prod: *mut f32,
    use_second: i32,
    rows: i32,
) -> i32 {
    unsafe {
        let (ixg_dim, nx_grid, x_grid_start, x_grid_intrv, ny_grid, y_grid_start, y_grid_intrv) =
            if use_second != 0 {
                (
                    ixg_dim2, nx_grid2, x_start2, x_intrv2, ny_grid2, y_start2, y_intrv2,
                )
            } else {
                (
                    ixg_dim1, nx_grid1, x_start1, x_intrv1, ny_grid1, y_start1, y_intrv1,
                )
            };
        if rows / 2 != 1 || nx_grid == 0 {
            return 1;
        }
        let mut xpos = vec![0.; (ixg_dim * ny_grid) as usize];
        let mut ypos = vec![0.; (ixg_dim * ny_grid) as usize];
        let mut xfinv1 = [0.; 9];
        let mut xfinv2 = [0.; 9];
        xf_invert(
            core::slice::from_raw_parts(xform2, (3 * rows) as usize),
            &mut xfinv2,
            rows as usize,
        );
        xf_invert(
            core::slice::from_raw_parts(xform1, (3 * rows) as usize),
            &mut xfinv1,
            rows as usize,
        );
        for iy in 0..ny_grid {
            for ix in 0..nx_grid {
                let ind = (ix + nx_grid * iy) as usize;
                let xgrid = x_grid_start + ix as f32 * x_grid_intrv;
                let ygrid = y_grid_start + iy as f32 * y_grid_intrv;
                let (mut dx, mut dy) = (0., 0.);
                if nx_grid2 != 0 {
                    interpolate_grid(
                        xgrid, ygrid, dx_grid2, dy_grid2, ixg_dim2, nx_grid2, ny_grid2, x_start2,
                        y_start2, x_intrv2, y_intrv2, &mut dx, &mut dy,
                    );
                }
                let (mut xnew, mut ynew) =
                    xf_apply(&xfinv2, xcen, ycen, xgrid + dx, ygrid + dy, rows as usize);
                interpolate_grid(
                    xnew, ynew, dx_grid1, dy_grid1, ixg_dim1, nx_grid1, ny_grid1, x_start1,
                    y_start1, x_intrv1, y_intrv1, &mut dx, &mut dy,
                );
                (xnew, ynew) = xf_apply(&xfinv1, xcen, ycen, xnew + dx, ynew + dy, rows as usize);
                *dx_prod.add(ind) = xnew - xgrid;
                *dy_prod.add(ind) = ynew - ygrid;
                xpos[ind] = xgrid;
                ypos[ind] = ygrid;
            }
        }
        let err = extract_linear_xform(
            xpos.as_mut_ptr(),
            ypos.as_mut_ptr(),
            dx_prod,
            dy_prod,
            nx_grid * ny_grid,
            xcen,
            ycen,
            dx_prod,
            dy_prod,
            xf_prod,
            rows,
        );
        if err == 0 && ixg_dim != nx_grid {
            for iy in (0..ny_grid).rev() {
                for ix in (0..nx_grid).rev() {
                    let ind = (ix + nx_grid * iy) as usize;
                    *dx_prod.add((ix + ixg_dim * iy) as usize) = *dx_prod.add(ind);
                    *dy_prod.add((ix + ixg_dim * iy) as usize) = *dy_prod.add(ind);
                }
            }
        }
        err
    }
}

static mut S_NUM_NEIGH: Option<Vec<i32>> = None;
static mut S_IND_NEIGH_START: Option<Vec<i32>> = None;
static mut S_NEIGHBORS: Option<Vec<i32>> = None;
static mut N_IN_LIST: i32 = 0;

/// Original `extrapolateGrid` (`warputils.c:303`).  The source's OpenMP work shares no
/// point writes, so this direct sequential traversal preserves its per-point arithmetic.
pub unsafe fn extrapolate_grid(
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    solved: *mut i8,
    xdim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_interval: f32,
    y_interval: f32,
    reuse: i32,
) -> i32 {
    unsafe {
        let dx_corn = [-1, 1, 1, -1];
        let dy_corn = [-1, -1, 1, 1];
        let dx_along = [1, 0, -1, 0];
        let dy_along = [0, 1, 0, -1];
        let ixstep = [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0];
        let iystep = [-1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1];
        let ngrid = (xdim * ny_grid) as usize;
        let block_type = solved.add(ngrid);
        let ind_list = block_type.add(ngrid) as *mut i32;
        if reuse != 0 && N_IN_LIST == 0 {
            return 0;
        }
        if reuse != 0
            && (S_NUM_NEIGH.is_none() || S_IND_NEIGH_START.is_none() || S_NEIGHBORS.is_none())
        {
            return 1;
        }
        if reuse == 0 {
            S_NUM_NEIGH = None;
            S_IND_NEIGH_START = None;
            S_NEIGHBORS = None;
            N_IN_LIST = 0;
            for iy in 0..ny_grid {
                for ix in 0..nx_grid {
                    let ind = ix + iy * xdim;
                    if *solved.add(ind as usize) == 0 {
                        ind_list.add(N_IN_LIST as usize).write_unaligned(ind);
                        N_IN_LIST += 1;
                    }
                }
            }
            if N_IN_LIST == 0 {
                return 0;
            }
            for iy in 0..ny_grid - 1 {
                for ix in 0..nx_grid - 1 {
                    let ind = ix + iy * xdim;
                    *block_type.add(ind as usize) = 0;
                    if *solved.add(ind as usize) != 0 {
                        if *solved.add((ind + 1) as usize) != 0
                            && *solved.add((ind + xdim) as usize) != 0
                            && *solved.add((ind + xdim + 1) as usize) != 0
                        {
                            *block_type.add(ind as usize) = 5;
                        } else if *solved.add((ind + 1) as usize) != 0
                            && *solved.add((ind + xdim) as usize) != 0
                        {
                            *block_type.add(ind as usize) = 3;
                        } else if *solved.add((ind + 1) as usize) != 0
                            && *solved.add((ind + xdim + 1) as usize) != 0
                        {
                            *block_type.add(ind as usize) = 4;
                        } else if *solved.add((ind + xdim) as usize) != 0
                            && *solved.add((ind + xdim + 1) as usize) != 0
                        {
                            *block_type.add(ind as usize) = 2;
                        }
                    } else if *solved.add((ind + 1) as usize) != 0
                        && *solved.add((ind + xdim) as usize) != 0
                        && *solved.add((ind + xdim + 1) as usize) != 0
                    {
                        *block_type.add(ind as usize) = 1;
                    }
                }
            }
            S_NUM_NEIGH = Some(vec![0; N_IN_LIST as usize]);
            S_IND_NEIGH_START = Some(vec![0; N_IN_LIST as usize]);
            S_NEIGHBORS = Some(Vec::new());
        }
        // `warputils.c:401-406`.  Every one of these is a `float` variable
        // assigned a **double** expression -- `3.` and `2.` are double
        // literals, so the division happens in double and only the store
        // narrows.  Dividing an `f32` by 3 in single precision rounds once
        // instead of twice and lands an ulp away, which moves the block
        // centres the neighbour search measures from.  Index 0 is never read:
        // `blockType` 0 means no block.
        let xf_ofs_x = [
            0.0_f32,
            (2. * f64::from(x_interval) / 3.) as f32,
            (f64::from(x_interval) / 3.) as f32,
            (f64::from(x_interval) / 3.) as f32,
            (2. * f64::from(x_interval) / 3.) as f32,
            (f64::from(x_interval) / 2.) as f32,
        ];
        let xf_ofs_y = [
            0.0_f32,
            (2. * f64::from(y_interval) / 3.) as f32,
            (2. * f64::from(y_interval) / 3.) as f32,
            (f64::from(y_interval) / 3.) as f32,
            (f64::from(y_interval) / 3.) as f32,
            (f64::from(y_interval) / 2.) as f32,
        ];
        // `float range = 2.;` (`warputils.c:322`).
        const RANGE: f32 = 2.;
        let xyint = x_interval.min(y_interval);
        for lind in 0..N_IN_LIST as usize {
            let ixyind = ind_list.add(lind).read_unaligned();
            let ix = ixyind % xdim;
            let iy = ixyind / xdim;
            if reuse == 0 {
                let nums = S_NUM_NEIGH.as_mut().unwrap();
                let starts = S_IND_NEIGH_START.as_mut().unwrap();
                let neigh = S_NEIGHBORS.as_mut().unwrap();
                nums[lind] = 0;
                starts[lind] = neigh.len() as i32;
                let mut dmin = 1.0e30_f32;
                let (mut minx, mut miny) = (0, 0);
                for delta in 1..nx_grid.max(ny_grid) {
                    // `warputils.c:436`: `delta` is an `int` and `0.7` a
                    // double literal, so the whole comparison is evaluated in
                    // double with `xyint` and `dmin` promoted -- doing it in
                    // `f32` breaks the search a delta early or late and
                    // changes which block ends up nearest.
                    if (delta as f64 - 0.7)
                        * (delta as f64 - 0.7)
                        * f64::from(xyint)
                        * f64::from(xyint)
                        > f64::from(dmin)
                    {
                        break;
                    }
                    for dir in 0..4_usize {
                        let (mut ixcorn, mut iycorn) =
                            (ix + dx_corn[dir] * delta, iy + dy_corn[dir] * delta);
                        if (dx_along[dir] != 0 && (iycorn < 0 || iycorn >= ny_grid - 1))
                            || (dy_along[dir] != 0 && (ixcorn < 0 || ixcorn >= nx_grid - 1))
                        {
                            continue;
                        }
                        let mut num = 2 * delta;
                        if dx_along[dir] != 0 {
                            let start = ixcorn.clamp(0, nx_grid - 2);
                            let end = (ixcorn + (num - 1) * dx_along[dir]).clamp(0, nx_grid - 2);
                            ixcorn = start;
                            num = (start - end).abs() + 1;
                        } else {
                            let start = iycorn.clamp(0, ny_grid - 2);
                            let end = (iycorn + (num - 1) * dy_along[dir]).clamp(0, ny_grid - 2);
                            iycorn = start;
                            num = (start - end).abs() + 1;
                        }
                        for _ in 0..num {
                            let btype = *block_type.add((ixcorn + iycorn * xdim) as usize) as i32;
                            if btype != 0 {
                                let dx =
                                    x_interval * (ixcorn - ix) as f32 + xf_ofs_x[btype as usize];
                                let dy =
                                    y_interval * (iycorn - iy) as f32 + xf_ofs_y[btype as usize];
                                let dist = dx * dx + dy * dy;
                                if dist < dmin {
                                    dmin = dist;
                                    minx = ixcorn;
                                    miny = iycorn;
                                }
                            }
                            ixcorn += dx_along[dir];
                            iycorn += dy_along[dir];
                        }
                    }
                }
                // `warputils.c:489-495`.  `sqrt` takes and returns a double
                // and the result is cast back to `float` before the
                // multiplication; `B3DMAX(xInterval, 1.)` compares a `float`
                // with a double literal, so its value -- and the division and
                // the `+ 1.` after it -- are double, narrowed only by the
                // store into `dlook`.
                let dist = RANGE * (f64::from(dmin).sqrt() as f32);
                let mut dlook = (f64::from(dist) / f64::from(x_interval).max(1.) + 1.) as f32;
                let jxmin = ((ix as f32 - dlook - 1.).round() as i32).clamp(0, nx_grid - 2);
                let jxmax = ((ix as f32 + dlook).round() as i32).clamp(0, nx_grid - 2);
                dlook = (f64::from(dist) / f64::from(y_interval).max(1.) + 1.) as f32;
                let jymin = ((iy as f32 - dlook - 1.).round() as i32).clamp(0, ny_grid - 2);
                let jymax = ((iy as f32 + dlook).round() as i32).clamp(0, ny_grid - 2);
                let distcrit = dist * dist;
                for jy in jymin..=jymax {
                    for jx in jxmin..=jxmax {
                        let btype = *block_type.add((jx + jy * xdim) as usize) as i32;
                        if btype != 0 {
                            let dxcen = (jx - ix) as f32 * x_interval + xf_ofs_x[btype as usize];
                            let dycen = (jy - iy) as f32 * y_interval + xf_ofs_y[btype as usize];
                            let dd = dxcen * dxcen + dycen * dycen;
                            if dd <= distcrit {
                                // `warputils.c:511-515`: `atan2` is the
                                // double routine, its arguments promoted, and
                                // the division and offset stay in double until
                                // the store into the `float` `angle`.
                                let mut angle = (f64::from(dycen).atan2(f64::from(dxcen))
                                    / 0.017453293
                                    + 157.5) as f32;
                                if angle < 0. {
                                    angle += 360.;
                                }
                                // `warputils.c:516`.
                                let ind_dom = ((angle / 45.) as i32).clamp(0, 7);
                                let mut boundary = false;
                                for is in ind_dom..=ind_dom + 4 {
                                    let nayx = jx + ixstep[is as usize];
                                    let nayy = jy + iystep[is as usize];
                                    if nayx >= 0
                                        && nayx < nx_grid - 1
                                        && nayy >= 0
                                        && nayy < ny_grid - 1
                                        && *block_type.add((nayx + nayy * xdim) as usize) == 0
                                    {
                                        boundary = true;
                                        break;
                                    }
                                }
                                if boundary || (jx == minx && jy == miny) {
                                    nums[lind] += 1;
                                    neigh.push(jx + jy * xdim);
                                }
                            }
                        }
                    }
                }
            }
            let nums = S_NUM_NEIGH.as_ref().unwrap();
            let starts = S_IND_NEIGH_START.as_ref().unwrap();
            let neigh = S_NEIGHBORS.as_ref().unwrap();
            *dx_grid.add(ixyind as usize) = 0.;
            *dy_grid.add(ixyind as usize) = 0.;
            let mut wsum = 0.;
            for nayx in 0..nums[lind] {
                let jxyind = neigh[(starts[lind] + nayx) as usize];
                let jx = jxyind % xdim;
                let jy = jxyind / xdim;
                let btype = *block_type.add(jxyind as usize) as usize;
                let dxcen = (jx - ix) as f32 * x_interval + xf_ofs_x[btype];
                let dycen = (jy - iy) as f32 * y_interval + xf_ofs_y[btype];
                let dist = dxcen * dxcen + dycen * dycen;
                let dx00 = *dx_grid.add(jxyind as usize);
                let dx10 = *dx_grid.add((jxyind + 1) as usize);
                let dx01 = *dx_grid.add((jxyind + xdim) as usize);
                let dx11 = *dx_grid.add((jxyind + xdim + 1) as usize);
                let dy00 = *dy_grid.add(jxyind as usize);
                let dy10 = *dy_grid.add((jxyind + 1) as usize);
                let dy01 = *dy_grid.add((jxyind + xdim) as usize);
                let dy11 = *dy_grid.add((jxyind + xdim + 1) as usize);
                let (mut a11, mut a12, mut dx, mut a21, mut a22, mut dy) = match btype {
                    1 => (
                        dx11 - dx01,
                        dx11 - dx10,
                        dx10 + dx01 - dx11,
                        dy11 - dy01,
                        dy11 - dy10,
                        dy10 + dy01 - dy11,
                    ),
                    2 => (
                        dx11 - dx01,
                        dx01 - dx00,
                        dx00,
                        dy11 - dy01,
                        dy01 - dy00,
                        dy00,
                    ),
                    3 => (
                        dx10 - dx00,
                        dx01 - dx00,
                        dx00,
                        dy10 - dy00,
                        dy01 - dy00,
                        dy00,
                    ),
                    4 => (
                        dx10 - dx00,
                        dx11 - dx10,
                        dx00,
                        dy10 - dy00,
                        dy11 - dy10,
                        dy00,
                    ),
                    _ => (
                        (dx10 - dx00 + dx11 - dx01) / 2.,
                        (dx01 - dx00 + dx11 - dx10) / 2.,
                        (3. * dx00 + dx01 + dx10 - dx11) / 4.,
                        (dy10 - dy00 + dy11 - dy01) / 2.,
                        (dy01 - dy00 + dy11 - dy10) / 2.,
                        (3. * dy00 + dy01 + dy10 - dy11) / 4.,
                    ),
                };
                a11 = 1. + a11 / x_interval;
                a12 /= y_interval;
                a21 /= x_interval;
                a22 = 1. + a22 / y_interval;
                dx = dx + dxcen - a11 * dxcen - a12 * dycen;
                dy = dy + dycen - a21 * dxcen - a22 * dycen;
                *dx_grid.add(ixyind as usize) += dx / dist;
                *dy_grid.add(ixyind as usize) += dy / dist;
                // `warputils.c:583`: `1.` is a double literal, so the
                // reciprocal is formed in double and the sum narrows only on
                // the store into the `float` `wsum`.
                wsum = (f64::from(wsum) + 1. / f64::from(dist)) as f32;
            }
            *dx_grid.add(ixyind as usize) /= wsum;
            *dy_grid.add(ixyind as usize) /= wsum;
        }
        if S_NEIGHBORS.is_none() { 1 } else { 0 }
    }
}

/// Original `extrapolateDone` (`warputils.c:683`).
pub unsafe fn extrapolate_done() {
    unsafe {
        S_NUM_NEIGH = None;
        S_IND_NEIGH_START = None;
        S_NEIGHBORS = None;
    }
}

/// Original static `newGridLimits` (`warputils.c:771`).
pub unsafe fn new_grid_limits(
    nx_grid: *mut i32,
    x_start: *mut f32,
    x_interval: f32,
    xdim: i32,
    ixmin: i32,
    ixmax: i32,
    x_big_str: f32,
    x_big_end: f32,
) -> i32 {
    unsafe {
        let nxgin = *nx_grid;
        let mut addlo = ((*x_start - x_big_str) / x_interval).ceil() as i32;
        while *x_start - addlo as f32 * x_interval <= ixmin as f32 {
            addlo -= 1;
        }
        addlo = addlo.max(0);
        let mut addhi =
            ((x_big_end - (*x_start + (nxgin - 1) as f32 * x_interval)) / x_interval).ceil() as i32;
        addhi = addhi.max(0);
        while *x_start + (addhi + nxgin - 1) as f32 * x_interval >= ixmax as f32 {
            addhi -= 1;
        }
        addhi = addhi.max(0);
        let extra = nxgin + addlo + addhi - xdim;
        if extra > 0 {
            let sublo = (extra / 2).min(addlo);
            let subhi = (extra - sublo).min(addhi);
            if sublo + subhi < extra {
                return 0;
            }
            addlo -= sublo;
            addhi -= extra - sublo;
        }
        *x_start -= addlo as f32 * x_interval;
        *nx_grid += addlo + addhi;
        addlo
    }
}

/// Original `expandAndExtrapGrid` (`warputils.c:707`).
pub unsafe fn expand_and_extrap_grid(
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    xdim: i32,
    _ydim: i32,
    nx_grid: *mut i32,
    ny_grid: *mut i32,
    x_start: *mut f32,
    y_start: *mut f32,
    x_interval: f32,
    y_interval: f32,
    x_big_str: f32,
    y_big_str: f32,
    x_big_end: f32,
    y_big_end: f32,
    ixmin: i32,
    ixmax: i32,
    iymin: i32,
    iymax: i32,
) -> i32 {
    unsafe {
        let nx_orig = *nx_grid;
        let ny_orig = *ny_grid;
        let addx = new_grid_limits(
            nx_grid, x_start, x_interval, xdim, ixmin, ixmax, x_big_str, x_big_end,
        );
        let addy = new_grid_limits(
            ny_grid, y_start, y_interval, _ydim, iymin, iymax, y_big_str, y_big_end,
        );
        if *nx_grid == nx_orig && *ny_grid == ny_orig {
            return 0;
        }
        let mut solved = vec![0_i8; (6 * xdim * *ny_grid) as usize];
        for iy in (0..*ny_grid).rev() {
            for ix in (0..*nx_grid).rev() {
                let jout = ix + iy * xdim;
                if ix < addx || ix >= nx_orig + addx || iy < addy || iy >= ny_orig + addy {
                    *solved.get_unchecked_mut(jout as usize) = 0;
                    *dx_grid.add(jout as usize) = 0.;
                    *dy_grid.add(jout as usize) = 0.;
                } else {
                    let jin = ix - addx + (iy - addy) * xdim;
                    *solved.get_unchecked_mut(jout as usize) = 1;
                    *dx_grid.add(jout as usize) = *dx_grid.add(jin as usize);
                    *dy_grid.add(jout as usize) = *dy_grid.add(jin as usize);
                }
            }
        }
        let jout = extrapolate_grid(
            dx_grid,
            dy_grid,
            solved.as_mut_ptr(),
            xdim,
            *nx_grid,
            *ny_grid,
            x_interval,
            y_interval,
            0,
        );
        extrapolate_done();
        jout
    }
}

/// Original `readCheckWarpFile` (`warputils.c:803`).
pub unsafe fn read_check_warp_file(
    filename: *mut i8,
    need_dist: i32,
    need_inv: i32,
    nx: *mut i32,
    ny: *mut i32,
    nz: *mut i32,
    ibinning: *mut i32,
    pixel_size: *mut f32,
    iflags: *mut i32,
    err_string: *mut i8,
    len_string: i32,
) -> i32 {
    unsafe {
        let mut version = 0;
        let mut ierr = read_warp_file(
            filename,
            nx,
            ny,
            nz,
            ibinning,
            pixel_size,
            &mut version,
            iflags,
        );
        *err_string.add((len_string - 1) as usize) = 0;
        let mut error = |text: &[u8]| {
            core::ptr::copy_nonoverlapping(
                text.as_ptr().cast::<i8>(),
                err_string,
                text.len().min((len_string - 1) as usize),
            );
            -2
        };
        if need_dist != 0 {
            if ierr < 0 {
                return error(b"OPENING OR READING DISTORTION FILE");
            }
            if *iflags & WARP_INVERSE == 0 {
                return error(b"DISTORTION CORRECTION CAN BE DONE ONLY WITH INVERSE TRANSFORMS");
            }
            if *iflags & WARP_CONTROL_PTS != 0 {
                return error(b"DISTORTION CORRECTION CAN BE DONE ONLY WITH A WARPING GRID, NOT CONTROL POINTS");
            }
        } else if ierr < 0 && (version != 0 || ierr != -3) {
            if ierr > -3 {
                return error(b"OPENING OR READING TRANSFORM FILE");
            }
            return error(b"INAPPROPRIATE VALUE OR MEMORY ERROR PROCESSING TRANSFORM FILE AS A WARPING FILE (IT DOES NOT APPEAR TO BE A LINEAR TRANSFORM FILE)");
        }
        if ierr < 0 {
            ierr = -1;
        } else if need_inv != 0 && *iflags & WARP_INVERSE == 0 {
            return error(b"THIS PROGRAM WILL WORK ONLY WITH INVERSE WARP DISPLACEMENTS");
        }
        ierr
    }
}

/// Original `findMaxGridSize` (`warputils.c:840`).
pub unsafe fn find_max_grid_size(
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    n_control: *mut i32,
    max_nxg: *mut i32,
    max_nyg: *mut i32,
    err_string: *mut i8,
    len_string: i32,
) -> i32 {
    unsafe {
        // `ERR_RETURN` copies the message and returns -2 on every failure
        // exit, so the caller always has something to print.
        let err_return = |message: &[u8]| {
            if len_string > 0 {
                core::ptr::copy_nonoverlapping(
                    message.as_ptr().cast(),
                    err_string,
                    message.len().min((len_string - 1) as usize),
                );
                *err_string.add(message.len().min((len_string - 1) as usize)) = 0;
            }
            -2
        };
        let mut nxwarp = 0;
        let mut nywarp = 0;
        let mut nzwarp = 0;
        let mut control = 0;
        if get_warp_file_size(&mut nxwarp, &mut nywarp, &mut nzwarp, &mut control) != 0 {
            return err_return(b"GETTING MAX GRID SIZE - THERE IS NO CURRENT WARP FILE");
        }
        let (mut x_min, mut y_min) = (1.0e20_f32, 1.0e20_f32);
        *max_nxg = 0;
        *max_nyg = 0;
        for iz in 0..nzwarp {
            *n_control.add(iz as usize) = 4;
            if control != 0 && get_num_warp_points(iz, n_control.add(iz as usize)) != 0 {
                return err_return(b"GETTING NUMBER OF CONTROL POINTS");
            }
            if *n_control.add(iz as usize) >= 3 {
                if control != 0 && grid_size_from_spacing(iz, -1., -1., 1) != 0 {
                    return err_return(b"SETTING GRID SIZE FROM SPACING OF CONTROL POINTS");
                }
                let (mut nxg, mut nyg, mut xs, mut ys, mut xi, mut yi) = (0, 0, 0., 0., 0., 0.);
                if get_grid_parameters(iz, &mut nxg, &mut nyg, &mut xs, &mut ys, &mut xi, &mut yi)
                    != 0
                {
                    return err_return(b"GETTING GRID PARAMETERS");
                }
                x_min = x_min.min(xi);
                y_min = y_min.min(yi);
            }
        }
        //
        // Allow the grid to be expanded to fit the actual image, at the
        // minimum interval.  Allow an extra position by adding 2.
        //
        // `B3DMIN(0., xmin)` compares a double against a float, so the
        // subtraction is a double one, and `B3DNINT` is
        // `(int)floor(x + 0.5)`, not a round-half-away-from-zero.  The
        // quotient is then `int / float`.
        //
        if x_min < 1.0e19 {
            let ix =
                ((nxwarp as f32).max(xmax) as f64 - 0_f64.min(xmin as f64) + 0.5).floor() as i32;
            let iy =
                ((nywarp as f32).max(ymax) as f64 - 0_f64.min(ymin as f64) + 0.5).floor() as i32;
            *max_nxg = (ix as f32 / x_min).ceil() as i32 + 2;
            *max_nyg = (iy as f32 / y_min).ceil() as i32 + 2;
        }
        0
    }
}

/// Original `getSizeAdjustedGrid` (`warputils.c:900`).
pub unsafe fn get_size_adjusted_grid(
    iz: i32,
    xnbig: f32,
    ynbig: f32,
    x_offset: f32,
    y_offset: f32,
    adjust_start: i32,
    warp_scale: f32,
    i_binning: i32,
    nx_grid: *mut i32,
    ny_grid: *mut i32,
    x_grid_start: *mut f32,
    y_grid_start: *mut f32,
    x_grid_interval: *mut f32,
    y_grid_interval: *mut f32,
    field_dx: *mut f32,
    field_dy: *mut f32,
    ixgdim: i32,
    iygdim: i32,
    err_string: *mut i8,
    len_string: i32,
) -> i32 {
    unsafe {
        // `ERR_RETURN` copies the message into `errString` and returns -2;
        // every failure exit of the source does that, so the caller always has
        // something to print.
        let err_return = |message: &[u8]| {
            if len_string > 0 {
                core::ptr::copy_nonoverlapping(
                    message.as_ptr().cast(),
                    err_string,
                    message.len().min((len_string - 1) as usize),
                );
                *err_string.add(message.len().min((len_string - 1) as usize)) = 0;
            }
            -2
        };
        let (mut nxwarp, mut nywarp, mut nzwarp, mut control_points) = (0, 0, 0, 0);
        if get_warp_file_size(&mut nxwarp, &mut nywarp, &mut nzwarp, &mut control_points) != 0 {
            return err_return(b"GETTING SIZE-ADJUSTED GRID - THERE IS NO CURRENT WARP FILE");
        }
        if control_points != 0 {
            // `warputils.c:915-917` assigns both results to `ierr` and never
            // tests it, so neither call can fail the routine.
            let _ = grid_size_from_spacing(iz, -1., -1., 1);
            let _ = get_grid_parameters(
                iz,
                nx_grid,
                ny_grid,
                x_grid_start,
                y_grid_start,
                x_grid_interval,
                y_grid_interval,
            );
            // The area actually needed.  `nxwarp / 2.` and `xnbig / 2.` are
            // double quotients in the source.
            let xmin = (nxwarp as f64 / 2. - xnbig as f64 / 2. + x_offset as f64) as f32;
            let xmax = xmin + xnbig;
            let ymin = (nywarp as f64 / 2. - ynbig as f64 / 2. + y_offset as f64) as f32;
            let ymax = ymin + ynbig;
            if xmax > nxwarp as f32 || xmin < 0. || ymax > nywarp as f32 || ymin < 0. {
                *nx_grid = adjust_size_and_start(
                    nxwarp,
                    ixgdim,
                    xmin,
                    xmax,
                    *x_grid_interval,
                    x_grid_start,
                );
                *ny_grid = adjust_size_and_start(
                    nywarp,
                    iygdim,
                    ymin,
                    ymax,
                    *y_grid_interval,
                    y_grid_start,
                );
                // `warputils.c:931` also assigns to `ierr` without testing.
                let _ = set_grid_size_to_make(
                    iz,
                    *nx_grid,
                    *ny_grid,
                    *x_grid_start,
                    *y_grid_start,
                    *x_grid_interval,
                    *y_grid_interval,
                );
            }
        }
        if get_warp_grid(
            iz,
            nx_grid,
            ny_grid,
            x_grid_start,
            y_grid_start,
            x_grid_interval,
            y_grid_interval,
            field_dx,
            field_dy,
            ixgdim,
        ) != 0
        {
            return err_return(b"GETTING WARP GRID OR DISTORTION FIELD");
        }
        let (mut x_add, mut y_add) = (0., 0.);
        if adjust_start != 0 {
            // `(xnbig - nxwarp) / 2.` is a double quotient.
            *x_grid_start = (*x_grid_start as f64 + (xnbig - nxwarp as f32) as f64 / 2.
                - x_offset as f64) as f32;
            *y_grid_start = (*y_grid_start as f64 + (ynbig - nywarp as f32) as f64 / 2.
                - y_offset as f64) as f32;
            x_add = x_offset;
            y_add = y_offset;
        }
        if control_points == 0
            && expand_and_extrap_grid(
                field_dx,
                field_dy,
                ixgdim,
                iygdim,
                nx_grid,
                ny_grid,
                x_grid_start,
                y_grid_start,
                *x_grid_interval,
                *y_grid_interval,
                0.,
                0.,
                xnbig,
                ynbig,
                x_offset.round() as i32,
                (xnbig + x_offset).round() as i32,
                y_offset.round() as i32,
                (ynbig + y_offset).round() as i32,
            ) != 0
        {
            return err_return(b"EXTRAPOLATING WARPING/DISTORTION GRID TO FULL AREA");
        }
        let bin_ratio = warp_scale / i_binning as f32;
        *x_grid_start *= bin_ratio;
        *y_grid_start *= bin_ratio;
        *x_grid_interval *= bin_ratio;
        *y_grid_interval *= bin_ratio;
        for iy in 0..*ny_grid {
            for ix in 0..*nx_grid {
                let index = (ix + iy * ixgdim) as usize;
                *field_dx.add(index) = (*field_dx.add(index) + x_add) * bin_ratio;
                *field_dy.add(index) = (*field_dy.add(index) + y_add) * bin_ratio;
            }
        }
        0
    }
}

/// Original static `adjustSizeAndStart` (`warputils.c:1000`).
unsafe fn adjust_size_and_start(
    nxwarp: i32,
    ixgdim: i32,
    xmin: f32,
    xmax: f32,
    x_grid_interval: f32,
    x_grid_start: *mut f32,
) -> i32 {
    unsafe {
        let mut grid_start = xmin.min(0.) + x_grid_interval / 10.;
        let grid_end = xmax.max(nxwarp as f32) - x_grid_interval / 10.;
        let grid_count = ixgdim.min(((grid_end - grid_start) / x_grid_interval).ceil() as i32 + 1);
        grid_start = (grid_end + grid_start) / 2. - x_grid_interval * (grid_count - 1) as f32 / 2.;
        *x_grid_start -= ((*x_grid_start - grid_start) / x_grid_interval).round() * x_grid_interval;
        grid_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libwarp::warpfiles::{
        new_warp_file, set_warp_grid, warp_files_done, write_warp_file,
    };
    use std::ffi::CString;

    #[test]
    fn interpolate_grid_clamps_and_bilinearly_weights_source_corners() {
        let mut dx_grid = [0., 2., 4., 6.];
        let mut dy_grid = [10., 12., 14., 16.];
        let mut dx = 0.;
        let mut dy = 0.;
        unsafe {
            interpolate_grid(
                0.5,
                0.5,
                dx_grid.as_mut_ptr(),
                dy_grid.as_mut_ptr(),
                2,
                2,
                2,
                0.,
                0.,
                1.,
                1.,
                &mut dx,
                &mut dy,
            );
        }
        assert_eq!(dx, 3.);
        assert_eq!(dy, 13.);
    }

    #[test]
    fn find_inverse_point_iterates_source_displacement_field() {
        let mut dx_grid = [1.; 4];
        let mut dy_grid = [-2.; 4];
        let (mut xnew, mut ynew, mut dx, mut dy) = (0., 0., 0., 0.);
        unsafe {
            find_inverse_point(
                4.,
                5.,
                dx_grid.as_mut_ptr(),
                dy_grid.as_mut_ptr(),
                2,
                2,
                2,
                0.,
                0.,
                1.,
                1.,
                &mut xnew,
                &mut ynew,
                &mut dx,
                &mut dy,
            );
        }
        assert_eq!((xnew, ynew, dx, dy), (3., 7., 1., -2.));
    }

    #[test]
    fn extract_linear_xform_removes_embedded_translation() {
        let mut x = [0., 1., 0., 1.];
        let mut y = [0., 0., 1., 1.];
        let mut vx = [3.; 4];
        let mut vy = [-2.; 4];
        let mut outx = [0.; 4];
        let mut outy = [0.; 4];
        let mut xf = [0.; 6];
        unsafe {
            assert_eq!(
                extract_linear_xform(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    vx.as_mut_ptr(),
                    vy.as_mut_ptr(),
                    4,
                    0.,
                    0.,
                    outx.as_mut_ptr(),
                    outy.as_mut_ptr(),
                    xf.as_mut_ptr(),
                    2
                ),
                0
            );
        }
        assert!(outx.iter().all(|value| value.abs() < 0.00001));
        assert!(outy.iter().all(|value| value.abs() < 0.00001));
        assert!((xf[4] + 3.).abs() < 0.00001 && (xf[5] - 2.).abs() < 0.00001);
    }

    #[test]
    fn extrapolate_grid_fills_missing_point_from_three_corners() {
        let mut dx = [2., 2., 2., 0.];
        let mut dy = [-1., -1., -1., 0.];
        let mut solved = vec![0_i8; 24];
        solved[0] = 1;
        solved[1] = 1;
        solved[2] = 1;
        unsafe {
            assert_eq!(
                extrapolate_grid(
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr(),
                    solved.as_mut_ptr(),
                    2,
                    2,
                    2,
                    1.,
                    1.,
                    0
                ),
                0
            );
            extrapolate_done();
        }
        assert!((dx[3] - 2.).abs() < 0.00001);
        assert!((dy[3] + 1.).abs() < 0.00001);
    }

    #[test]
    fn size_adjusted_grid_reports_the_source_no_current_file_error() {
        unsafe {
            warp_files_done();
            let (mut nx, mut ny) = (0, 0);
            let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
            let mut dx = [0.; 4];
            let mut dy = [0.; 4];
            let mut error = [0_i8; 80];
            assert_eq!(
                get_size_adjusted_grid(
                    0,
                    2.,
                    2.,
                    0.,
                    0.,
                    0,
                    1.,
                    1,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr(),
                    2,
                    2,
                    error.as_mut_ptr(),
                    error.len() as i32
                ),
                -2
            );
            assert!(
                std::ffi::CStr::from_ptr(error.as_ptr())
                    .to_bytes()
                    .starts_with(b"GETTING SIZE-ADJUSTED GRID")
            );
        }
    }

    #[test]
    fn expand_and_extrap_grid_moves_start_and_retains_source_grid() {
        let mut dx = [0.; 16];
        let mut dy = [0.; 16];
        for index in [0_usize, 1, 4, 5] {
            dx[index] = 4.;
            dy[index] = -3.;
        }
        let (mut nx, mut ny, mut xs, mut ys) = (2, 2, 1., 1.);
        unsafe {
            assert_eq!(
                expand_and_extrap_grid(
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr(),
                    4,
                    4,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    1.,
                    1.,
                    0.,
                    0.,
                    2.,
                    2.,
                    -10,
                    10,
                    -10,
                    10
                ),
                0
            );
        }
        assert_eq!((nx, ny, xs, ys), (3, 3, 0., 0.));
        assert_eq!((dx[5], dx[6], dx[9], dx[10]), (4., 4., 4., 4.));
        assert_eq!((dy[5], dy[6], dy[9], dy[10]), (-3., -3., -3., -3.));
    }

    #[test]
    fn read_check_warp_file_reads_an_actual_inverse_grid_file() {
        unsafe {
            warp_files_done();
            new_warp_file(8, 6, 1, 1., WARP_INVERSE);
            let mut dx = [1., 1., 1., 1.];
            let mut dy = [-2., -2., -2., -2.];
            assert_eq!(
                set_warp_grid(0, 2, 2, 0., 0., 4., 3., dx.as_mut_ptr(), dy.as_mut_ptr(), 2),
                0
            );
            let path = std::env::temp_dir().join("imod_rs_read_check_warp_file.xf");
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            assert_eq!(write_warp_file(name.as_ptr(), 1), 0);
            warp_files_done();
            let (mut nx, mut ny, mut nz, mut bin, mut pix, mut flags) = (0, 0, 0, 0, 0., 0);
            let mut error = [0_i8; 128];
            assert_eq!(
                read_check_warp_file(
                    name.as_ptr() as *mut i8,
                    1,
                    1,
                    &mut nx,
                    &mut ny,
                    &mut nz,
                    &mut bin,
                    &mut pix,
                    &mut flags,
                    error.as_mut_ptr(),
                    error.len() as i32
                ),
                0
            );
            assert_eq!((nx, ny, nz, flags), (8, 6, 1, WARP_INVERSE));
            let _ = std::fs::remove_file(path);
            warp_files_done();
        }
    }
}
