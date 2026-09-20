//! Translation of `IMOD/libwarp/warputils.c` and its direct declarations.
//!
//! Converted to idiomatic Rust per `NATIVE.md`: the source's `float *` grid
//! arguments are slices, its `char *errString` / `int lenString` pair is a
//! `&mut String`, and its three `static int *` neighbour tables are
//! thread-local `Vec`s.  Nothing here prints, so no `c_format` boundary is
//! needed.

use std::cell::{Cell, RefCell};

use crate::imod::libcfshr::linearxforms::{xf_apply, xf_copy, xf_invert};
use crate::imod::libcfshr::simplestat::ls_fit2;
use crate::imod::libwarp::warpfiles::{
    get_grid_parameters, get_num_warp_points, get_warp_file_size, get_warp_grid,
    grid_size_from_spacing, read_warp_file, set_grid_size_to_make,
};

/// C `MAX_THREADS` (`warputils.c:281`).
const MAX_THREADS: i32 = 16;

/// Original `interpolateGrid` (`warputils.c:33`).
///
/// `B3DMIN`/`B3DMAX` are `(a) < (b) ? (a) : (b)`, not `f32::min`/`max`, and the
/// `0.`/`1.` clamp literals are **double**, so `fx1` and `fy1` are formed in
/// double and narrowed only on the store.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_grid(
    x: f32,
    y: f32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    let xgrid = (x - x_grid_start) / x_grid_intrv;
    let mut ixg = xgrid as i32;
    ixg = {
        let inner = if nx_grid - 2 < ixg { nx_grid - 2 } else { ixg };
        if 0 > inner { 0 } else { inner }
    };
    /* NO EXTRAPOLATIONS ALLOWED */
    let fx1 = {
        let d = (xgrid - ixg as f32) as f64;
        let inner = if 1. < d { 1. } else { d };
        (if 0. > inner { 0. } else { inner }) as f32
    };
    let fx = 1. - fx1;
    let ixg1 = ixg + 1;
    let ygrid = (y - y_grid_start) / y_grid_intrv;
    let mut iyg = ygrid as i32;
    iyg = {
        let inner = if ny_grid - 2 < iyg { ny_grid - 2 } else { iyg };
        if 0 > inner { 0 } else { inner }
    };
    let fy1 = {
        let d = (ygrid - iyg as f32) as f64;
        let inner = if 1. < d { 1. } else { d };
        (if 0. > inner { 0. } else { inner }) as f32
    };
    let fy = 1. - fy1;
    let iyg1 = iyg + 1;
    let c00 = fx * fy;
    let c10 = fx1 * fy;
    let c01 = fx * fy1;
    let c11 = fx1 * fy1;

    /* interpolate */
    *dx = c00 * dx_grid[(ixg + iyg * ixg_dim) as usize]
        + c10 * dx_grid[(ixg1 + iyg * ixg_dim) as usize]
        + c01 * dx_grid[(ixg + iyg1 * ixg_dim) as usize]
        + c11 * dx_grid[(ixg1 + iyg1 * ixg_dim) as usize];
    *dy = c00 * dy_grid[(ixg + iyg * ixg_dim) as usize]
        + c10 * dy_grid[(ixg1 + iyg * ixg_dim) as usize]
        + c01 * dy_grid[(ixg + iyg1 * ixg_dim) as usize]
        + c11 * dy_grid[(ixg1 + iyg1 * ixg_dim) as usize];
}

/// Original `findInversePoint` (`warputils.c:70`).
#[allow(clippy::too_many_arguments)]
pub fn find_inverse_point(
    x: f32,
    y: f32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
    xnew: &mut f32,
    ynew: &mut f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    let max_iter = 10;
    let change_crit = 0.01_f64;
    let mut xlast = x;
    let mut ylast = y;

    for _iter in 0..max_iter {
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
            x_grid_intrv,
            y_grid_intrv,
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

/// Original `invertWarpGrid` (`warputils.c:101`).
#[allow(clippy::too_many_arguments)]
pub fn invert_warp_grid(
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
    xform: &[f32],
    xcen: f32,
    ycen: f32,
    dx_inv: &mut [f32],
    dy_inv: &mut [f32],
    xf_inv: &mut [f32],
    rows: i32,
) {
    xf_invert(xform, xf_inv, rows as usize);
    for iy in 0..ny_grid {
        for ix in 0..nx_grid {
            let mut ygrid = y_grid_start + iy as f32 * y_grid_intrv;
            let mut xgrid = x_grid_start + ix as f32 * x_grid_intrv;
            (xgrid, ygrid) = xf_apply(xform, xcen, ycen, xgrid, ygrid, rows as usize);
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
                x_grid_intrv,
                y_grid_intrv,
                &mut xnew,
                &mut ynew,
                &mut dx,
                &mut dy,
            );
            dx_inv[(ix + iy * ixg_dim) as usize] = -dx;
            dy_inv[(ix + iy * ixg_dim) as usize] = -dy;
        }
    }
}

/// Original `multiplyWarpings` (`warputils.c:141`).
///
/// The source passes `dxProd`/`dyProd` to `extractLinearXform` as **both** the
/// input vectors and the output vectors, which Rust cannot express with one
/// `&[f32]` and one `&mut [f32]`; the input copy is taken here instead.
/// `extractLinearXform`'s first loop writes `newXvec[i]` from `xVector[i]` in
/// the same iteration, so a copy is exactly what the aliasing did.
#[allow(clippy::too_many_arguments)]
pub fn multiply_warpings(
    dx_grid1: &[f32],
    dy_grid1: &[f32],
    ixg_dim1: i32,
    nx_grid1: i32,
    ny_grid1: i32,
    x_start1: f32,
    y_start1: f32,
    x_intrv1: f32,
    y_intrv1: f32,
    xform1: &[f32],
    xcen: f32,
    ycen: f32,
    dx_grid2: &[f32],
    dy_grid2: &[f32],
    ixg_dim2: i32,
    nx_grid2: i32,
    ny_grid2: i32,
    x_start2: f32,
    y_start2: f32,
    x_intrv2: f32,
    y_intrv2: f32,
    xform2: &[f32],
    dx_prod: &mut [f32],
    dy_prod: &mut [f32],
    xf_prod: &mut [f32],
    use_second: i32,
    rows: i32,
) -> i32 {
    /* Select the parameters of the output grid */
    let ixg_dim = if use_second != 0 { ixg_dim2 } else { ixg_dim1 };
    let nx_grid = if use_second != 0 { nx_grid2 } else { nx_grid1 };
    let x_grid_start = if use_second != 0 { x_start2 } else { x_start1 };
    let x_grid_intrv = if use_second != 0 { x_intrv2 } else { x_intrv1 };
    let ny_grid = if use_second != 0 { ny_grid2 } else { ny_grid1 };
    let y_grid_start = if use_second != 0 { y_start2 } else { y_start1 };
    let y_grid_intrv = if use_second != 0 { y_intrv2 } else { y_intrv1 };
    let mut xfinv1 = [0.0_f32; 9];
    let mut xfinv2 = [0.0_f32; 9];

    if rows / 2 != 1 || nx_grid == 0 {
        return 1;
    }

    /* Need arrays for positions for extracting linear transform.  Make them only big
    enough for packed data */
    let mut xpos = vec![0.0_f32; (ixg_dim * ny_grid) as usize];
    let mut ypos = vec![0.0_f32; (ixg_dim * ny_grid) as usize];

    xf_invert(xform2, &mut xfinv2, rows as usize);
    xf_invert(xform1, &mut xfinv1, rows as usize);
    for iy in 0..ny_grid {
        let ygrid = y_grid_start + iy as f32 * y_grid_intrv;
        for ix in 0..nx_grid {
            let ind = (ix + nx_grid * iy) as usize;
            let xgrid = x_grid_start + ix as f32 * x_grid_intrv;
            let mut dx = 0.0_f32;
            let mut dy = 0.0_f32;

            /* Back-transform through the second grid if it exists */
            if nx_grid2 != 0 {
                interpolate_grid(
                    xgrid, ygrid, dx_grid2, dy_grid2, ixg_dim2, nx_grid2, ny_grid2, x_start2,
                    y_start2, x_intrv2, y_intrv2, &mut dx, &mut dy,
                );
            }

            /* Then by inverse of f2, then through the first grid, then by inverse of f1 */
            let (mut xnew, mut ynew) =
                xf_apply(&xfinv2, xcen, ycen, xgrid + dx, ygrid + dy, rows as usize);
            interpolate_grid(
                xnew, ynew, dx_grid1, dy_grid1, ixg_dim1, nx_grid1, ny_grid1, x_start1, y_start1,
                x_intrv1, y_intrv1, &mut dx, &mut dy,
            );
            (xnew, ynew) = xf_apply(&xfinv1, xcen, ycen, xnew + dx, ynew + dy, rows as usize);

            /* Store data packed contiguously */
            dx_prod[ind] = xnew - xgrid;
            dy_prod[ind] = ynew - ygrid;
            xpos[ind] = xgrid;
            ypos[ind] = ygrid;
        }
    }
    let x_in = dx_prod[..(nx_grid * ny_grid) as usize].to_vec();
    let y_in = dy_prod[..(nx_grid * ny_grid) as usize].to_vec();
    let err = extract_linear_xform(
        &xpos,
        &ypos,
        &x_in,
        &y_in,
        nx_grid * ny_grid,
        xcen,
        ycen,
        dx_prod,
        dy_prod,
        xf_prod,
        rows,
    );
    if err == 0 {
        /* Get the linear transform and unpack into larger array if needed */
        if ixg_dim != nx_grid {
            for iy in (0..ny_grid).rev() {
                for ix in (0..nx_grid).rev() {
                    let ind = (ix + nx_grid * iy) as usize;
                    dx_prod[(ix + ixg_dim * iy) as usize] = dx_prod[ind];
                    dy_prod[(ix + ixg_dim * iy) as usize] = dy_prod[ind];
                }
            }
        }
    }

    xpos.clear();
    ypos.clear();
    err
}

/// Original `extractLinearXform` (`warputils.c:238`).
///
/// The source allows `newXvec == xVector`; here the input vectors are a
/// separate `&[f32]` and callers that aliased copy first (see
/// [`multiply_warpings`]).
#[allow(clippy::too_many_arguments)]
pub fn extract_linear_xform(
    x_pos: &[f32],
    y_pos: &[f32],
    x_vector: &[f32],
    y_vector: &[f32],
    n_points: i32,
    xcen: f32,
    ycen: f32,
    new_xvec: &mut [f32],
    new_yvec: &mut [f32],
    xfinv: &mut [f32],
    rows: i32,
) -> i32 {
    let mut mat = [0.0_f32; 6];

    if n_points < 3 || rows < 2 || rows > 3 {
        return 1;
    }

    let mut ptmp = vec![0.0_f32; n_points as usize];

    /* Need to fit points as function of points plus vector; this gives the
    inverse of the embedded transform.  This is the only use of [xy]Vector so new[XY]vec
    can be the same */
    for i in 0..n_points as usize {
        new_xvec[i] = x_pos[i] - xcen + x_vector[i];
        new_yvec[i] = y_pos[i] - ycen + y_vector[i];
        ptmp[i] = x_pos[i] - xcen;
    }
    let [m0, m1, m2, m3, m4, m5] = &mut mat;
    ls_fit2(
        &new_xvec[..n_points as usize],
        &new_yvec[..n_points as usize],
        &ptmp,
        n_points,
        m0,
        m2,
        Some(m4),
    );
    for i in 0..n_points as usize {
        ptmp[i] = y_pos[i] - ycen;
    }
    ls_fit2(
        &new_xvec[..n_points as usize],
        &new_yvec[..n_points as usize],
        &ptmp,
        n_points,
        m1,
        m3,
        Some(m5),
    );

    /* Apply this transform and subtract the point positions to get the new vectors */
    for i in 0..n_points as usize {
        let (xtmp, ytmp) = xf_apply(&mat, 0., 0., new_xvec[i], new_yvec[i], 2);
        new_xvec[i] = xtmp + xcen - x_pos[i];
        new_yvec[i] = ytmp + ycen - y_pos[i];
    }

    /* Return the inverse of the embedded transform, let caller use it or take inverse */
    xf_copy(&mat, 2, xfinv, rows as usize);
    ptmp.clear();
    0
}

thread_local! {
    /// C static `sNumNeigh` (`warputils.c:282`).
    static S_NUM_NEIGH: RefCell<Option<Vec<i32>>> = const { RefCell::new(None) };
    /// C static `sIndNeighStart` (`warputils.c:283`).
    static S_IND_NEIGH_START: RefCell<Option<Vec<i32>>> = const { RefCell::new(None) };
    /// C static `sNeighbors` (`warputils.c:284`).
    static S_NEIGHBORS: RefCell<Option<Vec<i32>>> = const { RefCell::new(None) };
    /// C function-static `ninList` (`warputils.c:318`).
    static N_IN_LIST: Cell<i32> = const { Cell::new(0) };
}

/// Original `extrapolateGrid` (`warputils.c:303`).
///
/// `solved` keeps the source's layout exactly: `6 * xdim * nyGrid` bytes, of
/// which `[0, n)` are the solved flags, `[n, 2n)` are `blockType`, and the rest
/// holds `indList` as native-endian `int`s.  That layout is load-bearing — on a
/// `reuse` call the block types and the index list are read back from the
/// caller's buffer, which is why `getWarpGrid` keeps a `6 * ngrid` static.
/// Reading the `int`s out of a byte slice also removes the source's unaligned
/// `(int *)(blockType + xdim * nyGrid)` cast.
///
/// The `#pragma omp parallel for` over `thread` is not reproduced: each thread
/// appends to its own neighbour list and the lists are concatenated in thread
/// order afterwards, over a partition of `lind` that is increasing, so a single
/// sequential list is the same array with the same `sIndNeighStart` offsets.
#[allow(clippy::too_many_arguments)]
pub fn extrapolate_grid(
    dx_grid: &mut [f32],
    dy_grid: &mut [f32],
    solved: &mut [u8],
    xdim: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_interval: f32,
    y_interval: f32,
    reuse: i32,
) -> i32 {
    let dx_corn = [-1, 1, 1, -1];
    let dy_corn = [-1, -1, 1, 1];
    let dx_along = [1, 0, -1, 0];
    let dy_along = [0, 1, 0, -1];
    let ixstep = [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0];
    let iystep = [-1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1];
    let ngrid = (xdim * ny_grid) as usize;
    let quantum = 1000;
    let mut xf_ofs_x = [0.0_f32; 6];
    let mut xf_ofs_y = [0.0_f32; 6];
    let range = 2.0_f32;

    if reuse != 0 && N_IN_LIST.get() == 0 {
        return 0;
    }
    if reuse != 0
        && (S_NUM_NEIGH.with_borrow(|v| v.is_none())
            || S_IND_NEIGH_START.with_borrow(|v| v.is_none())
            || S_NEIGHBORS.with_borrow(|v| v.is_none()))
    {
        return 1;
    }

    // `numOMPthreads(B3DMIN(ninList, MAX_THREADS))` is evaluated before the
    // `!reuse` branch zeroes `ninList`, so it uses the previous call's count.
    // Only the partitioning of `lind` between threads depends on it, and the
    // sequential build produces the same concatenation, so the value is
    // computed for fidelity and not used.
    let n_for_threads = if N_IN_LIST.get() < MAX_THREADS {
        N_IN_LIST.get()
    } else {
        MAX_THREADS
    };
    let num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(n_for_threads);
    let _num_threads = if num_threads < MAX_THREADS {
        num_threads
    } else {
        MAX_THREADS
    };
    let _quantum = quantum;

    let mut num_neigh: Vec<i32>;
    let mut ind_neigh_start: Vec<i32>;
    let mut neighbors: Vec<i32>;

    if reuse == 0 {
        S_NUM_NEIGH.with_borrow_mut(|v| *v = None);
        S_IND_NEIGH_START.with_borrow_mut(|v| *v = None);
        S_NEIGHBORS.with_borrow_mut(|v| *v = None);
        let mut nin_list = 0;

        /* Make list of indexes to do */
        for iy in 0..ny_grid {
            for ix in 0..nx_grid {
                let ixyind = ix + iy * xdim;
                if solved[ixyind as usize] == 0 {
                    let at = 2 * ngrid + 4 * nin_list as usize;
                    solved[at..at + 4].copy_from_slice(&ixyind.to_ne_bytes());
                    nin_list += 1;
                }
            }
        }
        N_IN_LIST.set(nin_list);
        if nin_list == 0 {
            return 0;
        }

        /* First analyze all blocks (indexed by lower left corner) for whether they have all
        4 or 3 corners solved.  Number them by missing point (CCW from lower left) */
        for iy in 0..ny_grid - 1 {
            for ix in 0..nx_grid - 1 {
                let ixyind = (ix + iy * xdim) as usize;
                solved[ngrid + ixyind] = 0;
                if solved[ixyind] != 0 {
                    if solved[ixyind + 1] != 0
                        && solved[ixyind + xdim as usize] != 0
                        && solved[ixyind + xdim as usize + 1] != 0
                    {
                        solved[ngrid + ixyind] = 5;
                    } else if solved[ixyind + 1] != 0 && solved[ixyind + xdim as usize] != 0 {
                        solved[ngrid + ixyind] = 3;
                    } else if solved[ixyind + 1] != 0 && solved[ixyind + xdim as usize + 1] != 0 {
                        solved[ngrid + ixyind] = 4;
                    } else if solved[ixyind + xdim as usize] != 0
                        && solved[ixyind + xdim as usize + 1] != 0
                    {
                        solved[ngrid + ixyind] = 2;
                    }
                } else if solved[ixyind + 1] != 0
                    && solved[ixyind + xdim as usize] != 0
                    && solved[ixyind + xdim as usize + 1] != 0
                {
                    solved[ngrid + ixyind] = 1;
                }
            }
        }

        num_neigh = vec![0; nin_list as usize];
        ind_neigh_start = vec![0; nin_list as usize];
        neighbors = Vec::new();
    } else {
        num_neigh = S_NUM_NEIGH.take().unwrap();
        ind_neigh_start = S_IND_NEIGH_START.take().unwrap();
        neighbors = S_NEIGHBORS.take().unwrap();
    }

    /* The coordinate offsets from lower left corner to center of points */
    // `xInterval / 3.` and `2. * xInterval / 3.` are double expressions with a
    // `float` result, so each rounds once on store.
    xf_ofs_x[2] = (x_interval as f64 / 3.) as f32;
    xf_ofs_x[3] = xf_ofs_x[2];
    xf_ofs_x[1] = (2. * x_interval as f64 / 3.) as f32;
    xf_ofs_x[4] = xf_ofs_x[1];
    xf_ofs_x[5] = (x_interval as f64 / 2.) as f32;
    xf_ofs_y[3] = (y_interval as f64 / 3.) as f32;
    xf_ofs_y[4] = xf_ofs_y[3];
    xf_ofs_y[1] = (2. * y_interval as f64 / 3.) as f32;
    xf_ofs_y[2] = xf_ofs_y[1];
    xf_ofs_y[5] = (y_interval as f64 / 2.) as f32;
    let xyint = if x_interval < y_interval {
        x_interval
    } else {
        y_interval
    };

    for lind in 0..N_IN_LIST.get() as usize {
        let at = 2 * ngrid + 4 * lind;
        let ixyind =
            i32::from_ne_bytes([solved[at], solved[at + 1], solved[at + 2], solved[at + 3]]);
        let ix = ixyind % xdim;
        let iy = ixyind / xdim;

        // `minx`/`miny` are uninitialised in the source when no block is found;
        // nothing reads them in that case because `sNumNeigh[lind]` stays 0.
        let mut minx = 0;
        let mut miny = 0;

        if reuse == 0 {
            num_neigh[lind] = 0;
            ind_neigh_start[lind] = neighbors.len() as i32;

            let mut dmin = 1.0e30_f32;

            /* find closest block with transform */
            /* Search progressively larger squares until half-side > minimum distance */
            let delta_lim = if nx_grid > ny_grid { nx_grid } else { ny_grid };
            for delta in 1..delta_lim {
                // `delta` is an int and `0.7` a double literal, so the whole
                // comparison is done in double with `xyint` and `dmin` promoted.
                if (delta as f64 - 0.7) * (delta as f64 - 0.7) * xyint as f64 * xyint as f64
                    > dmin as f64
                {
                    break;
                }

                /* Look at the 4 edges of the square, 4 directions;  Start in corner */
                for dir in 0..4_usize {
                    let mut ixcorn = ix + dx_corn[dir] * delta;
                    let mut iycorn = iy + dy_corn[dir] * delta;

                    /* If whole edge is out, skip */
                    if (dx_along[dir] != 0 && (iycorn < 0 || iycorn >= ny_grid - 1))
                        || (dy_along[dir] != 0 && (ixcorn < 0 || ixcorn >= nx_grid - 1))
                    {
                        continue;
                    }
                    let mut num = 2 * delta;

                    /* Adjust the other dimension's limits */
                    let start;
                    let end;
                    if dx_along[dir] != 0 {
                        start = {
                            let inner = if nx_grid - 2 < ixcorn {
                                nx_grid - 2
                            } else {
                                ixcorn
                            };
                            if 0 > inner { 0 } else { inner }
                        };
                        let e = ixcorn + (num - 1) * dx_along[dir];
                        end = {
                            let inner = if nx_grid - 2 < e { nx_grid - 2 } else { e };
                            if 0 > inner { 0 } else { inner }
                        };
                        ixcorn = start;
                    } else {
                        start = {
                            let inner = if ny_grid - 2 < iycorn {
                                ny_grid - 2
                            } else {
                                iycorn
                            };
                            if 0 > inner { 0 } else { inner }
                        };
                        let e = iycorn + (num - 1) * dy_along[dir];
                        end = {
                            let inner = if ny_grid - 2 < e { ny_grid - 2 } else { e };
                            if 0 > inner { 0 } else { inner }
                        };
                        iycorn = start;
                    }

                    /* Test each position along the edge */
                    num = (if start - end > end - start {
                        start - end
                    } else {
                        end - start
                    }) + 1;
                    for _i in 0..num {
                        let btype = solved[ngrid + (ixcorn + iycorn * xdim) as usize] as i32;
                        if btype != 0 {
                            let dx = x_interval * (ixcorn - ix) as f32 + xf_ofs_x[btype as usize];
                            let dy = y_interval * (iycorn - iy) as f32 + xf_ofs_y[btype as usize];
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

            /* Get actual distance to look, range of indexes to search, and
            the criterion which is square of maximum distance */
            // `sqrt` is the double routine and its result is cast back to float
            // before the multiply; `B3DMAX(xInterval, 1.)` compares a float with
            // a double literal, so `dlook` is a double expression narrowed on
            // store.  `B3DNINT` is `(int)floor(x + 0.5)`, not `round()`.
            let dist = range * ((dmin as f64).sqrt() as f32);
            let mut dlook = ({
                let d = if x_interval as f64 > 1. {
                    x_interval as f64
                } else {
                    1.
                };
                dist as f64 / d + 1.
            }) as f32;
            let jxmin = {
                let n = ((ix as f32 - dlook - 1.) as f64 + 0.5).floor() as i32;
                if 0 > n { 0 } else { n }
            };
            let jxmax = {
                let n = ((ix as f32 + dlook) as f64 + 0.5).floor() as i32;
                if nx_grid - 2 < n { nx_grid - 2 } else { n }
            };
            dlook = ({
                let d = if y_interval as f64 > 1. {
                    y_interval as f64
                } else {
                    1.
                };
                dist as f64 / d + 1.
            }) as f32;
            let jymin = {
                let n = ((iy as f32 - dlook - 1.) as f64 + 0.5).floor() as i32;
                if 0 > n { 0 } else { n }
            };
            let jymax = {
                let n = ((iy as f32 + dlook) as f64 + 0.5).floor() as i32;
                if ny_grid - 2 < n { ny_grid - 2 } else { n }
            };
            let distcrit = dist * dist;

            /* Loop in the neighborhood, find boundary points within range */
            for jy in jymin..=jymax {
                for jx in jxmin..=jxmax {
                    let btype = solved[ngrid + (jx + jy * xdim) as usize] as i32;
                    if btype != 0 {
                        let dxcen = (jx - ix) as f32 * x_interval + xf_ofs_x[btype as usize];
                        let dycen = (jy - iy) as f32 * y_interval + xf_ofs_y[btype as usize];
                        let dist = dxcen * dxcen + dycen * dycen;
                        if dist <= distcrit {
                            /* Find dominant direction to the point */
                            // `atan2` is the double routine and the divide and
                            // offset stay in double until the store into `angle`.
                            let mut angle =
                                ((dycen as f64).atan2(dxcen as f64) / 0.017453293 + 157.5) as f32;
                            if angle < 0. {
                                angle += 360.;
                            }
                            let ind_dom = {
                                let v = (angle / 45.) as i32;
                                let inner = if 7 < v { 7 } else { v };
                                if 0 > inner { 0 } else { inner }
                            };

                            /* Check that this point is a boundary, i.e. does not
                            have a neighbor in any one of the 5 directions toward
                            or at right angles to the dominant direction */
                            let mut boundary = 0;
                            let mut is = ind_dom;
                            while is <= ind_dom + 4 && boundary == 0 {
                                let nayx = jx + ixstep[is as usize];
                                let nayy = jy + iystep[is as usize];
                                if nayx >= 0
                                    && nayx < nx_grid - 1
                                    && nayy >= 0
                                    && nayy < ny_grid - 1
                                    && solved[ngrid + (nayx + nayy * xdim) as usize] == 0
                                {
                                    boundary = 1;
                                }
                                is += 1;
                            }

                            /* For boundary or min point, add to weighted sum */
                            if boundary != 0 || (jx == minx && jy == miny) {
                                num_neigh[lind] += 1;
                                neighbors.push(jx + jy * xdim);
                            }
                        }
                    }
                }
            }
        }

        /* Now go through the neighbor list */
        dx_grid[ixyind as usize] = 0.;
        dy_grid[ixyind as usize] = 0.;
        let mut wsum = 0.0_f32;
        for nayx in 0..num_neigh[lind] {
            let jxyind = neighbors[(ind_neigh_start[lind] + nayx) as usize];
            let jx = jxyind % xdim;
            let jy = jxyind / xdim;
            let btype = solved[ngrid + jxyind as usize] as i32;
            let dxcen = (jx - ix) as f32 * x_interval + xf_ofs_x[btype as usize];
            let dycen = (jy - iy) as f32 * y_interval + xf_ofs_y[btype as usize];
            let dist = dxcen * dxcen + dycen * dycen;

            /* Compute the vector transform centered on lower left of block */
            let dx00 = dx_grid[jxyind as usize];
            let dx10 = dx_grid[(jxyind + 1) as usize];
            let dx01 = dx_grid[(jxyind + xdim) as usize];
            let dx11 = dx_grid[(jxyind + xdim + 1) as usize];
            let dy00 = dy_grid[jxyind as usize];
            let dy10 = dy_grid[(jxyind + 1) as usize];
            let dy01 = dy_grid[(jxyind + xdim) as usize];
            let dy11 = dy_grid[(jxyind + xdim + 1) as usize];
            let (mut a11, mut a12, mut dx, mut a21, mut a22, mut dy) = match btype {
                /* These are either an exact transform (1-4) or best linear (5) */
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
                // `case 5` -- and the source leaves the six values indeterminate
                // for any other `btype`, which cannot occur.
                _ => (
                    ((dx10 - dx00 + dx11 - dx01) as f64 / 2.) as f32,
                    ((dx01 - dx00 + dx11 - dx10) as f64 / 2.) as f32,
                    ((3. * dx00 as f64 + dx01 as f64 + dx10 as f64 - dx11 as f64) / 4.) as f32,
                    ((dy10 - dy00 + dy11 - dy01) as f64 / 2.) as f32,
                    ((dy01 - dy00 + dy11 - dy10) as f64 / 2.) as f32,
                    ((3. * dy00 as f64 + dy01 as f64 + dy10 as f64 - dy11 as f64) / 4.) as f32,
                ),
            };

            /* Convert this to a coordinate transform centered on the point being
            filled in, in which case dx, dy are the vector there. */
            a11 = (1. + (a11 / x_interval) as f64) as f32;
            a12 /= y_interval;
            a21 /= x_interval;
            a22 = (1. + (a22 / y_interval) as f64) as f32;
            dx = dx + dxcen - a11 * dxcen - a12 * dycen;
            dy = dy + dycen - a21 * dxcen - a22 * dycen;

            dx_grid[ixyind as usize] += dx / dist;
            dy_grid[ixyind as usize] += dy / dist;
            // `1.` is a double literal, so the reciprocal is formed in double
            // and the sum narrows only on the store into the float `wsum`.
            wsum = (wsum as f64 + 1. / dist as f64) as f32;
        }
        dx_grid[ixyind as usize] /= wsum;
        dy_grid[ixyind as usize] /= wsum;
    }

    /* Now copy thread neighbor lists to one array and adjust start indices */
    S_NUM_NEIGH.with_borrow_mut(|v| *v = Some(num_neigh));
    S_IND_NEIGH_START.with_borrow_mut(|v| *v = Some(ind_neigh_start));
    S_NEIGHBORS.with_borrow_mut(|v| *v = Some(neighbors));

    // `sNeighbors == NULL ? 1 : 0`; a `Vec` cannot fail to allocate here.
    0
}

/// Original `extrapolateDone` (`warputils.c:683`).
pub fn extrapolate_done() {
    S_NUM_NEIGH.with_borrow_mut(|v| *v = None);
    S_IND_NEIGH_START.with_borrow_mut(|v| *v = None);
    S_NEIGHBORS.with_borrow_mut(|v| *v = None);
}

/// Original `expandAndExtrapGrid` (`warputils.c:707`).
#[allow(clippy::too_many_arguments)]
pub fn expand_and_extrap_grid(
    dx_grid: &mut [f32],
    dy_grid: &mut [f32],
    xdim: i32,
    ydim: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_start: &mut f32,
    y_start: &mut f32,
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
    let nx_orig = *nx_grid;
    let ny_orig = *ny_grid;

    /* Determine new limits in X and Y */
    let addx = new_grid_limits(
        nx_grid, x_start, x_interval, xdim, ixmin, ixmax, x_big_str, x_big_end,
    );
    let addy = new_grid_limits(
        ny_grid, y_start, y_interval, ydim, iymin, iymax, y_big_str, y_big_end,
    );
    if *nx_grid == nx_orig && *ny_grid == ny_orig {
        return 0;
    }

    /* Shift the grid and mark the solved array */
    let mut solved = vec![0_u8; (6 * xdim * *ny_grid) as usize];
    for iy in (0..*ny_grid).rev() {
        for ix in (0..*nx_grid).rev() {
            let jout = ix + iy * xdim;
            if ix < addx || ix >= nx_orig + addx || iy < addy || iy >= ny_orig + addy {
                solved[jout as usize] = 0;
                dx_grid[jout as usize] = 0.;
                dy_grid[jout as usize] = 0.;
            } else {
                solved[jout as usize] = 1;
                let jin = ix - addx + (iy - addy) * xdim;
                dx_grid[jout as usize] = dx_grid[jin as usize];
                dy_grid[jout as usize] = dy_grid[jin as usize];
            }
        }
    }
    let jout = extrapolate_grid(
        dx_grid,
        dy_grid,
        &mut solved,
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

/// Original static `newGridLimits` (`warputils.c:771`).
#[allow(clippy::too_many_arguments)]
pub fn new_grid_limits(
    nx_grid: &mut i32,
    x_start: &mut f32,
    x_interval: f32,
    xdim: i32,
    ixmin: i32,
    ixmax: i32,
    x_big_str: f32,
    x_big_end: f32,
) -> i32 {
    let nxgin = *nx_grid;

    /* Add grid points to get outside the big start and end, but limit it to stay between
    ixmin and ixmax */
    let mut addlo = (((*x_start - x_big_str) / x_interval) as f64).ceil() as i32;
    while *x_start - addlo as f32 * x_interval <= ixmin as f32 {
        addlo -= 1;
    }
    addlo = if 0 > addlo { 0 } else { addlo };
    let mut addhi = (((x_big_end - (*x_start + (nxgin - 1) as f32 * x_interval)) / x_interval)
        as f64)
        .ceil() as i32;
    addhi = if 0 > addhi { 0 } else { addhi };
    while *x_start + (addhi + nxgin - 1) as f32 * x_interval >= ixmax as f32 {
        addhi -= 1;
    }
    addhi = if 0 > addhi { 0 } else { addhi };

    /* Then if this is too many points for the array, trim equally from both sides */
    let extra = nxgin + addlo + addhi - xdim;
    if extra > 0 {
        let sublo = if extra / 2 < addlo { extra / 2 } else { addlo };
        let subhi = if extra - sublo < addhi {
            extra - sublo
        } else {
            addhi
        };

        /* If that can't work without throwing away existing data, forget it */
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

/// Original `readCheckWarpFile` (`warputils.c:803`).
///
/// `ERR_RETURN` is `strncpy(errString, a, lenString - 1); return -2;`.  With a
/// `&mut String` the `lenString` truncation is gone; every caller in this tree
/// passes a 1024-byte buffer and the longest message is 130 characters, so no
/// call was ever truncated.
#[allow(clippy::too_many_arguments)]
pub fn read_check_warp_file(
    filename: &str,
    need_dist: i32,
    need_inv: i32,
    nx: &mut i32,
    ny: &mut i32,
    nz: &mut i32,
    ibinning: &mut i32,
    pixel_size: &mut f32,
    iflags: &mut i32,
    err_string: &mut String,
) -> i32 {
    let mut iversion = 0;
    let mut ierr = read_warp_file(
        filename,
        nx,
        ny,
        nz,
        ibinning,
        pixel_size,
        &mut iversion,
        iflags,
    );
    err_string.clear();
    if need_dist != 0 {
        if ierr < 0 {
            err_string.push_str("OPENING OR READING DISTORTION FILE");
            return -2;
        }
        if *iflags % 2 == 0 {
            err_string.push_str("DISTORTION CORRECTION CAN BE DONE ONLY WITH INVERSE TRANSFORMS");
            return -2;
        }
        if (*iflags / 2) % 2 != 0 {
            err_string.push_str(
                "DISTORTION CORRECTION CAN BE DONE ONLY WITH A WARPING GRID, NOT CONTROL POINTS",
            );
            return -2;
        }
    } else if ierr < 0 && (iversion != 0 || ierr != -3) {
        if ierr > -3 {
            err_string.push_str("OPENING OR READING TRANSFORM FILE");
            return -2;
        }
        err_string.push_str(
            "INAPPROPRIATE VALUE OR MEMORY ERROR PROCESSING TRANSFORM FILE AS A WARPING FILE (IT DOES NOT APPEAR TO BE A LINEAR TRANSFORM FILE)",
        );
        return -2;
    }
    if ierr < 0 {
        ierr = -1;
    } else if need_inv != 0 && *iflags % 2 == 0 {
        err_string.push_str("THIS PROGRAM WILL WORK ONLY WITH INVERSE WARP DISPLACEMENTS");
        return -2;
    }
    ierr
}

/// Original `findMaxGridSize` (`warputils.c:840`).
#[allow(clippy::too_many_arguments)]
pub fn find_max_grid_size(
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    n_control: &mut [i32],
    max_nxg: &mut i32,
    max_nyg: &mut i32,
    err_string: &mut String,
) -> i32 {
    let (mut nxwarp, mut nywarp, mut nzwarp, mut control_pts) = (0, 0, 0, 0);
    let (mut nx_grid, mut ny_grid) = (0, 0);
    let (mut x_grid_strt, mut y_grid_strt, mut x_grid_intrv, mut y_grid_intrv) = (0., 0., 0., 0.);
    err_string.clear();
    if get_warp_file_size(&mut nxwarp, &mut nywarp, &mut nzwarp, &mut control_pts) != 0 {
        err_string.push_str("GETTING MAX GRID SIZE - THERE IS NO CURRENT WARP FILE");
        return -2;
    }

    let mut x_int_min = 1.0e20_f32;
    let mut y_int_min = 1.0e20_f32;
    *max_nxg = 0;
    *max_nyg = 0;
    for iz in 0..nzwarp {
        n_control[iz as usize] = 4;
        if control_pts != 0 && get_num_warp_points(iz, &mut n_control[iz as usize]) != 0 {
            err_string.push_str("GETTING NUMBER OF CONTROL POINTS");
            return -2;
        }
        if n_control[iz as usize] >= 3 {
            if control_pts != 0 && grid_size_from_spacing(iz, -1., -1., 1) != 0 {
                err_string.push_str("SETTING GRID SIZE FROM SPACING OF CONTROL POINTS");
                return -2;
            }
            if get_grid_parameters(
                iz,
                &mut nx_grid,
                &mut ny_grid,
                &mut x_grid_strt,
                &mut y_grid_strt,
                &mut x_grid_intrv,
                &mut y_grid_intrv,
            ) != 0
            {
                err_string.push_str("GETTING GRID PARAMETERS");
                return -2;
            }
            x_int_min = if x_int_min < x_grid_intrv {
                x_int_min
            } else {
                x_grid_intrv
            };
            y_int_min = if y_int_min < y_grid_intrv {
                y_int_min
            } else {
                y_grid_intrv
            };
        }
    }

    /* Allow the grid to be expanded to fit the actual image, at the minimum interval.
    Allow an extra position by adding 2 */
    // `B3DMAX(nxwarp, xmax)` yields a float; `B3DMIN(0., xmin)` yields a double
    // because `0.` is a double literal, so the subtraction is done in double
    // and `B3DNINT` is `(int)floor(x + 0.5)`.
    if x_int_min < 1.0e19 {
        let hi_x = if (nxwarp as f32) > xmax {
            nxwarp as f32
        } else {
            xmax
        };
        let lo_x = if 0. < xmin as f64 { 0. } else { xmin as f64 };
        let iz = ((hi_x as f64 - lo_x) + 0.5).floor() as i32;
        let hi_y = if (nywarp as f32) > ymax {
            nywarp as f32
        } else {
            ymax
        };
        let lo_y = if 0. < ymin as f64 { 0. } else { ymin as f64 };
        let iy = ((hi_y as f64 - lo_y) + 0.5).floor() as i32;
        *max_nxg = ((iz as f32 / x_int_min) as f64).ceil() as i32 + 2;
        *max_nyg = ((iy as f32 / y_int_min) as f64).ceil() as i32 + 2;
    }
    0
}

/// Original `getSizeAdjustedGrid` (`warputils.c:900`).
///
/// `gridSizeFromSpacing`, `getGridParameters` and `setGridSizeToMake` are all
/// assigned to `ierr` and never tested (`warputils.c:915-931`), so none of them
/// can fail the routine.
#[allow(clippy::too_many_arguments)]
pub fn get_size_adjusted_grid(
    iz: i32,
    xnbig: f32,
    ynbig: f32,
    x_offset: f32,
    y_offset: f32,
    adjust_start: i32,
    warp_scale: f32,
    i_binning: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_grid_strt: &mut f32,
    y_grid_strt: &mut f32,
    x_grid_intrv: &mut f32,
    y_grid_intrv: &mut f32,
    field_dx: &mut [f32],
    field_dy: &mut [f32],
    ixgdim: i32,
    iygdim: i32,
    err_string: &mut String,
) -> i32 {
    let (mut nxwarp, mut nywarp, mut nzwarp, mut control_pts) = (0, 0, 0, 0);
    err_string.clear();
    if get_warp_file_size(&mut nxwarp, &mut nywarp, &mut nzwarp, &mut control_pts) != 0 {
        err_string.push_str("GETTING SIZE-ADJUSTED GRID - THERE IS NO CURRENT WARP FILE");
        return -2;
    }

    /* For control points, see if we need to expand the sampled grid.  For stability, we
    always provide the full grid over the aligned area and expand by extrapolation */
    if control_pts != 0 {
        let _ierr = grid_size_from_spacing(iz, -1., -1., 1);
        let _ierr = get_grid_parameters(
            iz,
            nx_grid,
            ny_grid,
            x_grid_strt,
            y_grid_strt,
            x_grid_intrv,
            y_grid_intrv,
        );

        /* The area actually needed */
        // `nxwarp / 2.` and `xnbig / 2.` are double quotients.
        let xmin = (nxwarp as f64 / 2. - xnbig as f64 / 2. + x_offset as f64) as f32;
        let xmax = xmin + xnbig;
        let ymin = (nywarp as f64 / 2. - ynbig as f64 / 2. + y_offset as f64) as f32;
        let ymax = ymin + ynbig;
        if xmax > nxwarp as f32 || xmin < 0. || ymax > nywarp as f32 || ymin < 0. {
            /* If the area is larger, get new size and starting points */
            *nx_grid =
                adjust_size_and_start(nxwarp, ixgdim, xmin, xmax, *x_grid_intrv, x_grid_strt);
            *ny_grid =
                adjust_size_and_start(nywarp, iygdim, ymin, ymax, *y_grid_intrv, y_grid_strt);
            let _ierr = set_grid_size_to_make(
                iz,
                *nx_grid,
                *ny_grid,
                *x_grid_strt,
                *y_grid_strt,
                *x_grid_intrv,
                *y_grid_intrv,
            );
        }
    }
    if get_warp_grid(
        iz,
        nx_grid,
        ny_grid,
        x_grid_strt,
        y_grid_strt,
        x_grid_intrv,
        y_grid_intrv,
        field_dx,
        field_dy,
        ixgdim,
    ) != 0
    {
        err_string.push_str("GETTING WARP GRID OR DISTORTION FIELD");
        return -2;
    }

    /* If images are not full field, adjust grid start by half the difference between
    image and field size, still in warp file pixels.
    Also subtract the offset and set up to add it to the vectors */
    let mut x_add = 0.0_f32;
    let mut y_add = 0.0_f32;
    if adjust_start != 0 {
        // `(xnbig - nxwarp) / 2.` is a double quotient.
        *x_grid_strt =
            (*x_grid_strt as f64 + (xnbig - nxwarp as f32) as f64 / 2. - x_offset as f64) as f32;
        *y_grid_strt =
            (*y_grid_strt as f64 + (ynbig - nywarp as f32) as f64 / 2. - y_offset as f64) as f32;
        x_add = x_offset;
        y_add = y_offset;
    }

    /* Then expand a grid to fill the space */
    if control_pts == 0
        && expand_and_extrap_grid(
            field_dx,
            field_dy,
            ixgdim,
            iygdim,
            nx_grid,
            ny_grid,
            x_grid_strt,
            y_grid_strt,
            *x_grid_intrv,
            *y_grid_intrv,
            0.,
            0.,
            xnbig,
            ynbig,
            (x_offset as f64 + 0.5).floor() as i32,
            ((xnbig + x_offset) as f64 + 0.5).floor() as i32,
            (y_offset as f64 + 0.5).floor() as i32,
            ((ynbig + y_offset) as f64 + 0.5).floor() as i32,
        ) != 0
    {
        err_string.push_str("EXTRAPOLATING WARPING/DISTORTION GRID TO FULL AREA");
        return -2;
    }

    /* Next adjust grid start and interval and field itself for the
    overall binning or change of scale */
    let bin_ratio = warp_scale / i_binning as f32;
    *x_grid_strt *= bin_ratio;
    *y_grid_strt *= bin_ratio;
    *x_grid_intrv *= bin_ratio;
    *y_grid_intrv *= bin_ratio;

    /* scale field */
    for iy in 0..*ny_grid {
        for i in 0..*nx_grid {
            let ind = (i + iy * ixgdim) as usize;
            field_dx[ind] = (field_dx[ind] + x_add) * bin_ratio;
            field_dy[ind] = (field_dy[ind] + y_add) * bin_ratio;
        }
    }
    0
}

/// Original static `adjustSizeAndStart` (`warputils.c:1000`).
///
/// Every `/ 2.` and `/ 10.` here is a double literal, and `B3DNINT` is
/// `(int)floor(a + 0.5)`, not `round()`.
pub fn adjust_size_and_start(
    nxwarp: i32,
    ixgdim: i32,
    xmin: f32,
    xmax: f32,
    x_grid_intrv: f32,
    x_grid_strt: &mut f32,
) -> i32 {
    /* compute starts and ends that encompass the needed area and get new grid size */
    let mut gstr = ({
        let lo = if 0. < xmin { 0.0_f32 } else { xmin };
        lo as f64 + x_grid_intrv as f64 / 10.
    }) as f32;
    let gend = ({
        let hi = if nxwarp as f32 > xmax {
            nxwarp as f32
        } else {
            xmax
        };
        hi as f64 - x_grid_intrv as f64 / 10.
    }) as f32;
    let ceiled = (((gend - gstr) / x_grid_intrv) as f64).ceil() as i32 + 1;
    let nxgr = if ixgdim < ceiled { ixgdim } else { ceiled };

    /* Get ideal new starting point and then shift to nearest point that keeps points
    on the same grid - this stabilizes the grid against changes in amount of expansion */
    gstr = ((gend + gstr) as f64 / 2. - (x_grid_intrv * (nxgr - 1) as f32) as f64 / 2.) as f32;
    *x_grid_strt -= ((((*x_grid_strt - gstr) / x_grid_intrv) as f64 + 0.5).floor() as i32) as f32
        * x_grid_intrv;
    nxgr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libwarp::warpfiles::{WARP_CONTROL_PTS, WARP_INVERSE};
    use crate::imod::libwarp::warpfiles::{
        new_warp_file, set_warp_grid, warp_files_done, write_warp_file,
    };

    #[test]
    fn interpolate_grid_clamps_and_bilinearly_weights_source_corners() {
        let dx_grid = [0., 2., 4., 6.];
        let dy_grid = [10., 12., 14., 16.];
        let mut dx = 0.;
        let mut dy = 0.;
        interpolate_grid(
            0.5, 0.5, &dx_grid, &dy_grid, 2, 2, 2, 0., 0., 1., 1., &mut dx, &mut dy,
        );
        assert_eq!(dx, 3.);
        assert_eq!(dy, 13.);
    }

    #[test]
    fn find_inverse_point_iterates_source_displacement_field() {
        let dx_grid = [1.; 4];
        let dy_grid = [-2.; 4];
        let (mut xnew, mut ynew, mut dx, mut dy) = (0., 0., 0., 0.);
        find_inverse_point(
            4., 5., &dx_grid, &dy_grid, 2, 2, 2, 0., 0., 1., 1., &mut xnew, &mut ynew, &mut dx,
            &mut dy,
        );
        assert_eq!((xnew, ynew, dx, dy), (3., 7., 1., -2.));
    }

    #[test]
    fn extract_linear_xform_removes_embedded_translation() {
        let x = [0., 1., 0., 1.];
        let y = [0., 0., 1., 1.];
        let vx = [3.; 4];
        let vy = [-2.; 4];
        let mut outx = [0.; 4];
        let mut outy = [0.; 4];
        let mut xf = [0.; 6];
        assert_eq!(
            extract_linear_xform(
                &x, &y, &vx, &vy, 4, 0., 0., &mut outx, &mut outy, &mut xf, 2
            ),
            0
        );
        assert!(outx.iter().all(|value| value.abs() < 0.00001));
        assert!(outy.iter().all(|value| value.abs() < 0.00001));
        assert!((xf[4] + 3.).abs() < 0.00001 && (xf[5] - 2.).abs() < 0.00001);
    }

    #[test]
    fn extrapolate_grid_fills_missing_point_from_three_corners() {
        let mut dx = [2., 2., 2., 0.];
        let mut dy = [-1., -1., -1., 0.];
        let mut solved = vec![0_u8; 24];
        solved[0] = 1;
        solved[1] = 1;
        solved[2] = 1;
        assert_eq!(
            extrapolate_grid(&mut dx, &mut dy, &mut solved, 2, 2, 2, 1., 1., 0),
            0
        );
        extrapolate_done();
        assert!((dx[3] - 2.).abs() < 0.00001);
        assert!((dy[3] + 1.).abs() < 0.00001);
    }

    #[test]
    fn size_adjusted_grid_reports_the_source_no_current_file_error() {
        warp_files_done();
        let (mut nx, mut ny) = (0, 0);
        let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
        let mut dx = [0.; 4];
        let mut dy = [0.; 4];
        let mut error = String::new();
        assert_eq!(
            get_size_adjusted_grid(
                0, 2., 2., 0., 0., 0, 1., 1, &mut nx, &mut ny, &mut xs, &mut ys, &mut xi, &mut yi,
                &mut dx, &mut dy, 2, 2, &mut error
            ),
            -2
        );
        assert!(error.starts_with("GETTING SIZE-ADJUSTED GRID"));
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
        assert_eq!(
            expand_and_extrap_grid(
                &mut dx, &mut dy, 4, 4, &mut nx, &mut ny, &mut xs, &mut ys, 1., 1., 0., 0., 2., 2.,
                -10, 10, -10, 10
            ),
            0
        );
        assert_eq!((nx, ny, xs, ys), (3, 3, 0., 0.));
        assert_eq!((dx[5], dx[6], dx[9], dx[10]), (4., 4., 4., 4.));
        assert_eq!((dy[5], dy[6], dy[9], dy[10]), (-3., -3., -3., -3.));
    }

    #[test]
    fn read_check_warp_file_reads_an_actual_inverse_grid_file() {
        warp_files_done();
        new_warp_file(8, 6, 1, 1., WARP_INVERSE);
        let dx = [1., 1., 1., 1.];
        let dy = [-2., -2., -2., -2.];
        assert_eq!(set_warp_grid(0, 2, 2, 0., 0., 4., 3., &dx, &dy, 2), 0);
        let path =
            std::env::temp_dir().join(format!("imod-rs-read-check-warp-{}.xf", std::process::id()));
        let name = path.to_string_lossy().into_owned();
        assert_eq!(write_warp_file(&name, 1), 0);
        warp_files_done();
        let (mut nx, mut ny, mut nz, mut bin, mut pix, mut flags) = (0, 0, 0, 0, 0., 0);
        let mut error = String::new();
        assert_eq!(
            read_check_warp_file(
                &name, 1, 1, &mut nx, &mut ny, &mut nz, &mut bin, &mut pix, &mut flags, &mut error
            ),
            0
        );
        assert_eq!((nx, ny, nz, flags), (8, 6, 1, WARP_INVERSE));
        let _ = std::fs::remove_file(path);
        warp_files_done();
    }

    #[test]
    fn control_pts_flag_constant_matches_the_source() {
        assert_eq!(WARP_CONTROL_PTS, 2);
    }
}
