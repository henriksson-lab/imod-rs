//! Translation of `IMOD/flib/tiltalign/solve_xyzd.cpp` — functions for
//! initializing dxy and xyz values.
//!
//! # One source, two builds
//!
//! The unit is compiled into both `tiltalign` and `beadtrack` (with
//! `-DBEADTRACK`), and has no conditional code of its own; the build only
//! reaches it through `errorExit` (`utilfuncs.rs`).  So every function that
//! calls `errorExit` takes `const BEADTRACK: bool` and forwards it to
//! [`error_exit`], as `utilfuncs.rs` settles.  All three calls here pass
//! `ifLocal = 0`, which exits in both builds, so the parameter selects nothing
//! observable in this unit; it is carried so the call reads as the source's.
//!
//! # LAPACK
//!
//! `solvePackedSums` calls `dspsv_` (`solve_xyzd.cpp:455`), provided by the
//! `faer`-backed boundary `flib/subrs/lapack/dspsv.rs` (not a translation; see
//! its docs and the CLAUDE.md LAPACK/BLAS paragraph for the committed
//! tolerance).  The source passes its `double *b` as `IPIV`, so the integer
//! pivots land in the first `4 * m` bytes of `bl`; that is reproduced (see
//! [`solve_packed_sums`]), because `solveXyzd` reads `bl` back as coordinates
//! without testing `ierr` (`solve_xyzd.cpp:225-240`).
//!
//! # Allocation
//!
//! `B3DMALLOC` becomes zero-filled `Vec`s (`NATIVE.md` §4); a `Vec` allocation
//! failure aborts instead of returning `NULL`, so the allocation-failure
//! `errorExit` tests are kept with an always-true "allocated" flag.  The
//! source's single `float` block in `solve_xyzd_iter` (twenty `nview`-long
//! arrays carved out of one `malloc`) and the `double` block in `solveXyzd`
//! (`sx`/`xml`/`sdl`/`bl`) are separate `Vec`s; no array is indexed outside its
//! own part.  The one place where uninitialised memory is observable is `bl`
//! after a failed solve (above): its elements past the pivot bytes are `malloc`
//! residue in the source and zero here.
//!
//! # Arithmetic
//!
//! `cos`/`sin`/`sqrt` of a `float` are the C++ `float` overloads (the
//! reference object imports `sincosf`/`sqrtf`), i.e. `f32` methods here.
//! `pow(float, 2.f)` is the `float` overload, which g++ folds to a
//! single-precision `x * x` (the object imports no `pow`); it is written as
//! that product.  Note the two error sums group differently:
//! `error += pow(..) + pow(..)` adds the two `float` squares in single
//! precision first (`:262`), while `error = error + pow(..) + pow(..)` adds
//! each to the `double` in turn (`:702`).

use std::io::Write;

use super::arraymaxes::MAX_REAL_FOR_DIRECT_INIT;
use super::utilfuncs::error_exit;
use crate::imod::flib::subrs::lapack::dspsv::dspsv;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes, wall_time};

/// Original: `solveXyzd` (`solve_xyzd.cpp:70`).
///
/// Obtains estimates for the underlying (real) x, y, z coordinates of each
/// point and for the delta X and Y of each view, for the given values of tilt,
/// rotation, mag and compression.  `sprod` should hold at least
/// `(3 * npt - 1) * (3 * npt - 2) / 2` elements, where
/// `npt = min(numRealPt, maxRealForDirectInit)`.
pub fn solve_xyzd<const BEADTRACK: bool>(
    xx: &[f32],
    yy: &[f32],
    isec_view: &[i32],
    ireal_start: &[i32],
    nview: i32,
    num_real_pt: i32,
    tilt: &[f32],
    rot: &[f32],
    gmag: &[f32],
    comp: &[f32],
    xyz: &mut [f32],
    dxy: &mut [f32],
    rot_inc: f32,
    sprod: &mut [f64],
    error: &mut f64,
    ierr: &mut i32,
) {
    let mut indqk = [0i32; 4];
    let mut base_base: i32;
    let mut big_base = [false; 2];
    let mut big_row: bool;
    let mut ad = [0f32; 2];
    let mut be = [0f32; 2];
    let mut cf = [0f32; 2];
    let mut fac: f32;
    let mut xybar = [0f32; 2];
    let mut xyzcen = [0f32; 3];
    let num_cols: i32;
    let mut num_rows: i32;
    let mut icol: i32 = 0;
    let mut i: i32;
    let mut ipt: i32;
    let mut num_on_view: i32 = 0;
    let real_pt_dim = 3 * num_real_pt;
    let mut wall_start: f64;
    let mut add_time: f64;

    // If the overall data size is large or the number of points exceeds the limit, use
    // the old iterative method
    if num_real_pt * nview > 10000 || num_real_pt > MAX_REAL_FOR_DIRECT_INIT {
        wall_start = wall_time();
        let _ = wall_start;
        *ierr = 0;
        ipt = 95.max(90 + (num_real_pt as f32).sqrt() as i32);
        solve_iteratively::<BEADTRACK>(
            ipt,
            nview,
            ireal_start,
            isec_view,
            num_real_pt,
            xyz,
            xx,
            yy,
            dxy,
            tilt,
            rot,
            gmag,
            comp,
            rot_inc,
            error,
        );
        //printf("Iterative time %8.3f  error %19.5f\n",
        //     1000. * (wallTime() - wallStart), error);
        return;
    }
    //
    // Load data matrix, loop on views then real points projecting on each
    num_cols = 3 * num_real_pt - 2;
    num_rows = 0;
    *ierr = 1;
    add_time = 0.;
    if num_real_pt < 2 {
        *ierr = 0;
        xyz[0] = 0.;
        xyz[1] = 0.;
        xyz[2] = 0.;
        return;
    }

    // valRow and baseRow share one float block in the source (valRow is the
    // first realPtDim elements, baseRow the next 2 * realPtDim); sx, xml, sdl and
    // bl share one double block.
    let mut val_row = vec![0f32; real_pt_dim as usize];
    let mut base_row = vec![0f32; 2 * real_pt_dim as usize];
    let mut sx = vec![0f64; real_pt_dim as usize];
    let mut xml = vec![0f64; real_pt_dim as usize];
    let mut sdl = vec![0f64; real_pt_dim as usize];
    let mut bl = vec![0f64; real_pt_dim as usize];
    let mut ind_on_view = vec![0i32; real_pt_dim as usize];
    let allocated = true;
    if !allocated {
        error_exit::<BEADTRACK>("Allocating working arrays in solveXyzd", 0);
    }

    for jpt in 0..num_cols {
        sx[jpt as usize] = 0.;
    }
    for jpt in 0..(num_cols + 1) * num_cols / 2 {
        sprod[jpt as usize] = 0.;
    }

    for iv in 1..=nview {
        //
        // Set up ad, be, cf, and get indexes to points on view and get xybar
        geometric_coeffs_and_indices(
            iv,
            ireal_start,
            isec_view,
            num_real_pt,
            &mut ind_on_view,
            &mut num_on_view,
            xx,
            yy,
            tilt,
            rot,
            gmag,
            comp,
            rot_inc,
            &mut ad,
            &mut be,
            &mut cf,
            &mut xybar,
        );
        //
        // Set up the terms for the dxy: if last point is not on view, subtract
        // a / n etc from the columns for all points on view; if last point is on
        // view, add a / n to the columns for all points not in the view
        for ixy in 1..=2 {
            big_base[ixy - 1] = false;
            base_base = (ixy as i32 - 1) * real_pt_dim;
            for ipt in 0..num_cols {
                base_row[(base_base + ipt) as usize] = 0.;
            }
            for ipt in 1..=num_real_pt - 1 {
                if (ind_on_view[(ipt - 1) as usize] > 0
                    && ind_on_view[(num_real_pt - 1) as usize] == 0)
                    || (ind_on_view[(ipt - 1) as usize] == 0
                        && ind_on_view[(num_real_pt - 1) as usize] > 0)
                {
                    fac = 1.;
                    if ind_on_view[(num_real_pt - 1) as usize] == 0 {
                        fac = -1.;
                    }
                    icol = 3 * ipt - 2;
                    base_row[(base_base + icol - 1) as usize] =
                        fac * ad[ixy - 1] / num_on_view as f32;
                    base_row[(base_base + icol) as usize] = fac * be[ixy - 1] / num_on_view as f32;
                    base_row[(base_base + icol + 1) as usize] =
                        fac * cf[ixy - 1] / num_on_view as f32;
                    //
                    // Keep track of whether this vector is non - zero
                    big_base[ixy - 1] = true;
                }
            }
        }
        //
        // Loop on points again and set up the equations for their projections
        for jpt in 1..=num_real_pt {
            if ind_on_view[(jpt - 1) as usize] > 0 {
                for ixy in 1..=2 {
                    base_base = (ixy as i32 - 1) * real_pt_dim;
                    for ipt in 0..num_cols {
                        val_row[ipt as usize] = base_row[(base_base + ipt) as usize];
                    }
                    big_row = false;
                    if jpt < num_real_pt {
                        icol = 3 * jpt - 2;
                        val_row[(icol - 1) as usize] += ad[ixy - 1];
                        val_row[(icol + 1 - 1) as usize] += be[ixy - 1];
                        val_row[(icol + 2 - 1) as usize] += cf[ixy - 1];
                    } else {
                        //
                        // Last point is negative sum of all the rest, and makes a big row of data
                        for ipt in 1..=num_real_pt - 1 {
                            icol = 3 * ipt - 2;
                            val_row[(icol - 1) as usize] -= ad[ixy - 1];
                            val_row[(icol + 1 - 1) as usize] -= be[ixy - 1];
                            val_row[(icol + 2 - 1) as usize] -= cf[ixy - 1];
                        }
                        big_row = true;
                    }
                    //
                    // Get the projection position and put it in the last column
                    // Adjust that position by the mean of points on view
                    if ixy == 1 {
                        val_row[(num_cols - 1) as usize] =
                            xx[(ind_on_view[(jpt - 1) as usize] - 1) as usize] - xybar[0];
                    } else {
                        val_row[(num_cols - 1) as usize] =
                            yy[(ind_on_view[(jpt - 1) as usize] - 1) as usize] - xybar[1];
                    }
                    //
                    // Row is done.  Accumulate it
                    wall_start = wall_time();
                    if big_row || big_base[ixy - 1] {
                        add_row_to_sums(&val_row, num_cols, &mut sx, sprod);
                    } else {
                        //
                        // Just handle non - zero columns if there are a few, it is a lot quicker
                        indqk[0] = icol;
                        indqk[1] = icol + 1;
                        indqk[2] = icol + 2;
                        indqk[3] = num_cols;
                        for j in 1..=4usize {
                            sx[(indqk[j - 1] - 1) as usize] +=
                                val_row[(indqk[j - 1] - 1) as usize] as f64;
                            for i in 1..=j {
                                icol = packed_index(indqk[i - 1], indqk[j - 1]);
                                sprod[(icol - 1) as usize] += (val_row[(indqk[i - 1] - 1) as usize]
                                    * val_row[(indqk[j - 1] - 1) as usize])
                                    as f64;
                            }
                        }
                    }
                    add_time = add_time + wall_time() - wall_start;
                    num_rows = num_rows + 1;
                }
            }
        }
    }
    let _ = ImodFile::Stdout.flush();
    wall_start = wall_time();
    let _ = (wall_start, add_time);

    solve_packed_sums(
        &sx, sprod, num_cols, num_rows, &mut xml, &mut sdl, &mut bl, ierr,
    );
    //
    // retrieve solution
    for v in xyz[..(3 * num_real_pt) as usize].iter_mut() {
        *v = 0.;
    }
    // Fixed in translation (2026-09-26, `BUGS.md`): the source does not test
    // `ierr` here and reads `bl` — the pivot indices plus `malloc` residue
    // after a failed `dspsv` — back as coordinates.  After a failed solve the
    // coordinates stay 0 (the callers already test `ierr` where they use it).
    i = if *ierr != 0 { num_real_pt } else { 1 };
    while i <= num_real_pt - 1 {
        for ixy in 1..=3 {
            icol = 3 * (i - 1) + ixy;
            xyz[(i * 3 + ixy - 4) as usize] = bl[(icol - 1) as usize] as f32;
            xyz[(num_real_pt * 3 + ixy - 4) as usize] -= xyz[(i * 3 + ixy - 4) as usize];
        }
        i += 1;
    }
    //
    // compute total error, equals F reported by funct
    *error = 0.;
    for iv in 1..=nview {
        geometric_coeffs_and_indices(
            iv,
            ireal_start,
            isec_view,
            num_real_pt,
            &mut ind_on_view,
            &mut num_on_view,
            xx,
            yy,
            tilt,
            rot,
            gmag,
            comp,
            rot_inc,
            &mut ad,
            &mut be,
            &mut cf,
            &mut xybar,
        );
        xyzcen[0] = 0.;
        xyzcen[1] = 0.;
        xyzcen[2] = 0.;
        for jpt in 1..=num_real_pt {
            if ind_on_view[(jpt - 1) as usize] > 0 {
                for ipt in 0..3 {
                    xyzcen[ipt as usize] += xyz[(3 * (jpt - 1) + ipt) as usize];
                }
            }
        }
        for ixy in 1..=2 {
            dxy[(iv * 2 + ixy - 3) as usize] = xybar[(ixy - 1) as usize]
                - (ad[(ixy - 1) as usize] * xyzcen[0]
                    + be[(ixy - 1) as usize] * xyzcen[1]
                    + cf[(ixy - 1) as usize] * xyzcen[2])
                    / num_on_view as f32;
        }
        for jpt in 1..=num_real_pt {
            i = ind_on_view[(jpt - 1) as usize];
            if i > 0 {
                let px = ad[0] * xyz[(jpt * 3 - 3) as usize]
                    + be[0] * xyz[(jpt * 3 - 2) as usize]
                    + cf[0] * xyz[(jpt * 3 - 1) as usize]
                    + dxy[(iv * 2 - 2) as usize]
                    - xx[(i - 1) as usize];
                let py = ad[1] * xyz[(jpt * 3 - 3) as usize]
                    + be[1] * xyz[(jpt * 3 - 2) as usize]
                    + cf[1] * xyz[(jpt * 3 - 1) as usize]
                    + dxy[(iv * 2 - 1) as usize]
                    - yy[(i - 1) as usize];
                *error += (px * px + py * py) as f64;
            }
        }
    }
}

/// Original: `geometricCoeffsAndIndices` (`solve_xyzd.cpp:280`, file static).
///
/// Sets up the ad, be, cf for a view, counts points on view, gets indices to
/// them and computes the mean projection position.
fn geometric_coeffs_and_indices(
    iv: i32,
    ireal_start: &[i32],
    isec_view: &[i32],
    num_real_pt: i32,
    ind_on_view: &mut [i32],
    num_on_view: &mut i32,
    xx: &[f32],
    yy: &[f32],
    tilt: &[f32],
    rot: &[f32],
    gmag: &[f32],
    comp: &[f32],
    rot_inc: f32,
    ad: &mut [f32; 2],
    be: &mut [f32; 2],
    cf: &mut [f32; 2],
    xybar: &mut [f32; 2],
) {
    let iv0 = (iv - 1) as usize;
    let cos_theta = tilt[iv0].cos();
    let c_sin_theta = comp[iv0] * tilt[iv0].sin();
    let g_cos_phi = gmag[iv0] * (rot[iv0] + rot_inc).cos();
    let g_sin_phi = gmag[iv0] * (rot[iv0] + rot_inc).sin();
    //
    ad[0] = cos_theta * g_cos_phi;
    be[0] = -g_sin_phi;
    cf[0] = c_sin_theta * g_cos_phi;
    ad[1] = cos_theta * g_sin_phi;
    be[1] = g_cos_phi;
    cf[1] = c_sin_theta * g_sin_phi;
    //
    // Count  and get indices and get the mean projection coordinate
    *num_on_view = 0;
    xybar[0] = 0.;
    xybar[1] = 0.;
    for jpt in 1..=num_real_pt {
        ind_on_view[(jpt - 1) as usize] = 0;
        for i in ireal_start[(jpt - 1) as usize]..=ireal_start[(jpt + 1 - 1) as usize] - 1 {
            if iv == isec_view[(i - 1) as usize] {
                ind_on_view[(jpt - 1) as usize] = i;
                *num_on_view = *num_on_view + 1;
                xybar[0] = xybar[0] + xx[(i - 1) as usize];
                xybar[1] = xybar[1] + yy[(i - 1) as usize];
            }
        }
    }
    xybar[0] = xybar[0] / *num_on_view as f32;
    xybar[1] = xybar[1] / *num_on_view as f32;
}

/// Original: `solveIteratively` (`solve_xyzd.cpp:324`, file static).
///
/// Implements the original iterative solution method: try either with initial
/// dxy solved to equalize centroids section-to-section, or with dxy 0; find
/// which way gives lowest error somewhere along the line, and redo it that way
/// to do just the best number of iterations.
///
/// `isolMin` is uninitialised in the source and stays so if no error on the
/// list is below `1.e30` (a NaN error); it starts at 0 here.
fn solve_iteratively<const BEADTRACK: bool>(
    max_solve: i32,
    nview: i32,
    ireal_start: &[i32],
    isec_view: &[i32],
    num_real_pt: i32,
    xyz: &mut [f32],
    xx: &[f32],
    yy: &[f32],
    dxy: &mut [f32],
    tilt: &[f32],
    rot: &[f32],
    gmag: &[f32],
    comp: &[f32],
    rot_inc: f32,
    error: &mut f64,
) {
    let mut imin_tilt: i32;
    let mut isol_min_init: i32 = 0;
    let mut isol_min: i32 = 0;
    let mut isolve: i32 = 0;
    let mut nsum: i32;
    let mut err_min_init: f32 = 0.;
    let mut err_min: f32 = 0.;
    let mut cgx: f32;
    let mut cgy: f32;

    let mut err_list = vec![0f32; (max_solve + 5) as usize];
    let allocated = true;
    if !allocated {
        error_exit::<BEADTRACK>("Allocating array in solveIteratively", 0);
    }
    //
    // Get the view at minimum tilt and get the CG of the points on that view
    //
    imin_tilt = 1;
    for iv in 1..=nview {
        let t = tilt[(iv - 1) as usize];
        let tm = tilt[(imin_tilt - 1) as usize];
        if (if t >= 0. { t } else { -t }) < (if tm >= 0. { tm } else { -tm }) {
            imin_tilt = iv;
        }
    }
    cgx = 0.;
    cgy = 0.;
    nsum = 0;
    for ipt in 1..=num_real_pt {
        for i in ireal_start[(ipt - 1) as usize]..=ireal_start[(ipt + 1 - 1) as usize] - 1 {
            if isec_view[(i - 1) as usize] == imin_tilt {
                cgx = cgx + xx[(i - 1) as usize];
                cgy = cgy + yy[(i - 1) as usize];
                nsum = nsum + 1;
            }
        }
    }
    cgx = cgx / nsum as f32;
    cgy = cgy / nsum as f32;

    // initial trial with call to INIT_DXY
    init_dxy(
        xx,
        yy,
        isec_view,
        ireal_start,
        nview,
        num_real_pt,
        imin_tilt,
        dxy,
        cgx,
        cgy,
    );
    //
    for itry in 1..=2 {
        //
        // second time through, save minimum error and iteration # from
        // first trial that used call to init_dxy
        //
        if itry == 2 {
            isol_min_init = isol_min;
            err_min_init = err_min;
        }
        //
        solve_xyzd_iter::<BEADTRACK>(
            xx,
            yy,
            isec_view,
            ireal_start,
            nview,
            num_real_pt,
            tilt,
            rot,
            gmag,
            comp,
            rot_inc,
            xyz,
            dxy,
            max_solve,
            error,
            &mut err_list,
            &mut isolve,
        );
        //
        // find iteration with minimum error
        //
        err_min = 1.0e30;
        for i in 1..=isolve - 1 {
            if err_list[(i - 1) as usize] < err_min {
                isol_min = i;
                err_min = err_list[(i - 1) as usize];
            }
        }
        //
        // set dxy to 0 for second try, or leave at zero for final setup
        //
        for iv in 1..=nview {
            dxy[(iv * 2 - 2) as usize] = cgx;
            dxy[(iv * 2 - 1) as usize] = cgy;
        }
    }
    //
    if err_min_init < err_min {
        isol_min = isol_min_init;
        init_dxy(
            xx,
            yy,
            isec_view,
            ireal_start,
            nview,
            num_real_pt,
            imin_tilt,
            dxy,
            cgx,
            cgy,
        );
        //print *, 'DXY set to equalize centroids gave best initialization'
    } else {
        //print *, 'DXY set to zero gave best initialization'
    }
    //
    solve_xyzd_iter::<BEADTRACK>(
        xx,
        yy,
        isec_view,
        ireal_start,
        nview,
        num_real_pt,
        tilt,
        rot,
        gmag,
        comp,
        rot_inc,
        xyz,
        dxy,
        isol_min,
        error,
        &mut err_list,
        &mut isolve,
    );
}

/// Original: `addRowToSums` (`solve_xyzd.cpp:410`, file static).
///
/// Accumulate sums and sums of cross-products from a row of data.  The product
/// is `float * float`, widened for the `double` sum.
fn add_row_to_sums(row: &[f32], m: i32, sx: &mut [f64], sprod: &mut [f64]) {
    let mut ind: i32;
    for j in 0..m {
        sx[j as usize] += row[j as usize] as f64;
    }
    for j in 1..=m {
        ind = packed_index(1, j);
        for ii in 0..j {
            sprod[(ind + ii - 1) as usize] += (row[ii as usize] * row[(j - 1) as usize]) as f64;
        }
    }
}

/// Original: `solvePackedSums` (`solve_xyzd.cpp:425`, file static).
///
/// A subroutine for calling the linear solver.  `sx`, `xm`, `sd` and `b` are
/// four consecutive parts of one allocation in the caller; `ss` is the caller's
/// `sprod`.
///
/// The source passes `b` as `dspsv`'s integer `IPIV` argument (`:455`), so the
/// `m` pivots are written, as `int`s, over the first `4 * m` bytes of `b`
/// (little-endian: an even-indexed pivot is the low half of a `double`).  On
/// success all `m` elements of `b` are then overwritten; when `dspsv` fails
/// (`ierr != 0`) the function returns early and `b` keeps those bytes, which
/// `solveXyzd` then reads back as coordinates.  The pivots come from
/// `dspsv.rs`'s `faer` factorisation, in `DSPTRF`'s encoding; they equal the
/// reference's only where the two pivot rules agree (see `dspsv.rs`).
///
/// Measured (`/big/henriksson/realbench/wave3-solve/`, 81 direct solves,
/// m = 3..597, against the reference `dspsv_`): the unscaled solution differs
/// by at most 6.9e-15 relative, `ierr` is identical, and the returned `xyz`,
/// `dxy` and `error` are bit-identical.  **`ierr` is not guaranteed on a
/// system that is singular in exact arithmetic**: with every view at one tilt
/// and `comp * sin(tilt) == cos(tilt)` each bead's x and z columns are equal
/// before scaling and differ in the last bit after it, and whether an
/// elimination step then yields an exactly zero pivot depends on rounding
/// order — the reference returned INFO 0 / 10 where `faer` returned 1 / 7.
/// After a failed solve the coordinates are pivot bits and `malloc` residue in
/// the reference (zero here), so nothing about that path is matchable.
fn solve_packed_sums(
    sx: &[f64],
    ss: &mut [f64],
    mp: i32,
    nrows: i32,
    xm: &mut [f64],
    sd: &mut [f64],
    b: &mut [f64],
    ierr: &mut i32,
) {
    let m: i32;
    let mut ind: i32;
    //
    // First get the means and SDs
    m = mp - 1;
    for i in 1..=mp {
        let i0 = (i - 1) as usize;
        xm[i0] = sx[i0] / nrows as f64;
        sd[i0] = ((ss[(packed_index(i, i) - 1) as usize] - sx[i0] * sx[i0] / nrows as f64)
            / (nrows as f64 - 1.))
            .sqrt();
    }

    //
    // What we have now is a raw sum of squares and cross - products
    // Scale the matrix by the sd's; this scales the RHS (b) variable in the last column
    // This is what multrd does, and is not the same as multr does, which is to convert
    // the matrix to true correlation coefficients, because when there is no constant
    // term the solution involves raw sums of squares instead of deviations from mean
    for j in 1..=mp {
        ind = packed_index(1, j);
        for i in 1..=j {
            let k = (ind + i - 1 - 1) as usize;
            ss[k] = ss[k] / (sd[(i - 1) as usize] * sd[(j - 1) as usize]);
        }
    }
    //
    // Now we can call lapack with this matrix.  Since it is not a covariance matrix,
    // can't use the positive - definite routine, have to use the symmetric matrix one
    ind = packed_index(1, mp);
    let one = 1;
    {
        // dspsv("U", &m, &one, ss, b, &ss[ind - 1], &m, &ierr, 1): AP is ss up to
        // ind - 1, the right-hand side is ss from ind - 1, and IPIV is b.  The
        // pivot indices are scratch here; `b` is written only on success
        // (the source leaves the pivot bits in it, which `solveXyzd` then read
        // as coordinates: fixed there, `BUGS.md`).
        let (ap, rhs) = ss.split_at_mut((ind - 1) as usize);
        let mut ipiv = vec![0i32; m as usize];
        dspsv("U", m, one, ap, &mut ipiv, &mut rhs[..m as usize], m, ierr);
    }
    if *ierr != 0 {
        return;
    }
    //
    // scale and return b
    for i in 1..=m {
        b[(i - 1) as usize] =
            ss[(ind + i - 1 - 1) as usize] * sd[(mp - 1) as usize] / sd[(i - 1) as usize];
    }
}

/// Original: `packedIndex` (`solve_xyzd.cpp:466`, file static).
///
/// Silly function because statement functions are now obsolete.
fn packed_index(i: i32, j: i32) -> i32 {
    i + (j - 1) * j / 2
}

/// Original: `solve_xyzd_iter` (`solve_xyzd.cpp:497`, file static).
///
/// Obtains estimates for the underlying (real) x, y, z coordinates of each
/// point and for the delta X and Y of each view by alternating regressions for
/// each point's x, y, z and averages for each view's dx, dy; repeated up to
/// `maxSolve` times, or until the biggest change in dx or dy becomes tiny.
///
/// `iwatch` is 0 in the source, so its trace `printf` never runs; `dxLast` and
/// `dyLast`, which only it reads, are uninitialised there and 0 here.
fn solve_xyzd_iter<const BEADTRACK: bool>(
    xx: &[f32],
    yy: &[f32],
    isec_view: &[i32],
    ireal_start: &[i32],
    nview: i32,
    num_real_pt: i32,
    tilt: &[f32],
    rot: &[f32],
    gmag: &[f32],
    comp: &[f32],
    rot_inc: f32,
    xyz: &mut [f32],
    dxy: &mut [f32],
    max_solve: i32,
    error: &mut f64,
    err_list: &mut [f32],
    isolve: &mut i32,
) {
    //
    let mut bvec = [0f32; 3];
    let mut bsum: f32;
    let mut cos_theta: f32;
    let mut csin_theta: f32;
    let mut del1: f32;
    let mut del2: f32;
    let mut del_dx: f32;
    let mut del_dy: f32;
    let mut del_new: f32;
    let mut diff_max: f32;
    let mut dx_last: f32 = 0.;
    let mut dx_new: f32;
    let mut dx_tmp: f32;
    let mut dy_last: f32 = 0.;
    let mut dy_new: f32;
    let mut dy_tmp: f32;
    let mut err_diff: f32;
    let mut gcos_phi: f32;
    let mut gsin_phi: f32;
    let mut iv: i32;
    let mut iwatch: i32;
    let mut jump_base: i32;
    let mut jump_fast: i32;
    let mut ncyc_skip: i32;
    let ncyc_skip_lim: i32;
    //
    // precompute the a-f and relevant cross-products for the regression
    //
    let nv = nview as usize;
    let mut nsum = vec![0i32; nv];
    let mut a = vec![0f32; nv];
    let allocated = true;
    if !allocated {
        error_exit::<BEADTRACK>("Allocating arrays in solve_xyzd_iter", 0);
    }
    let mut b = vec![0f32; nv];
    let mut c = vec![0f32; nv];
    let mut asq = vec![0f32; nv];
    let mut bsq = vec![0f32; nv];
    let mut csq = vec![0f32; nv];
    let mut axb = vec![0f32; nv];
    let mut axc = vec![0f32; nv];
    let mut bxc = vec![0f32; nv];
    let mut d = vec![0f32; nv];
    let mut e = vec![0f32; nv];
    let mut f = vec![0f32; nv];
    let mut dx_sum = vec![0f32; nv];
    let mut dy_sum = vec![0f32; nv];
    let mut dxsq_sum = vec![0f32; nv];
    let mut dysq_sum = vec![0f32; nv];
    let mut del_dxy_last = vec![0f32; 2 * nv];
    let mut dxy_last = vec![0f32; 2 * nv];

    iwatch = 0;
    // if (iwatch==0) iwatch=1
    for i in 1..=nview {
        let i0 = (i - 1) as usize;
        cos_theta = tilt[i0].cos();
        csin_theta = comp[i0] * tilt[i0].sin();
        gcos_phi = gmag[i0] * (rot[i0] + rot_inc).cos();
        gsin_phi = gmag[i0] * (rot[i0] + rot_inc).sin();
        //
        a[i0] = cos_theta * gcos_phi;
        b[i0] = -gsin_phi;
        c[i0] = csin_theta * gcos_phi;
        d[i0] = cos_theta * gsin_phi;
        e[i0] = gcos_phi;
        f[i0] = csin_theta * gsin_phi;
    }
    //
    // the square and cross product terms for the x and y paired
    // observations are always added together - so just combine them here
    regression_cross_products(
        nview, &a, &b, &c, &d, &e, &f, &mut asq, &mut bsq, &mut csq, &mut axb, &mut axc, &mut bxc,
    );
    //
    // start looping until maxSolve times, or diffMax gets small, or error
    // really blows up a lot from first iteration
    //
    *isolve = 1;
    diff_max = 1.;
    jump_fast = 0;
    jump_base = 3;
    ncyc_skip_lim = 5;
    err_list[0] = 1.;
    while *isolve <= max_solve
        && diff_max as f64 > 1.0e-7
        && (*isolve <= 3
            || err_list[0.max(*isolve - 2) as usize] as f64 <= 1000. * err_list[0] as f64)
    {
        //
        // zero arrays for sums of xd, dy and squares for each view
        //
        for iv in 1..=nview {
            let iv0 = (iv - 1) as usize;
            dx_sum[iv0] = 0.;
            dy_sum[iv0] = 0.;
            dxsq_sum[iv0] = 0.;
            dysq_sum[iv0] = 0.;
            nsum[iv0] = 0;
        }
        //
        // loop on each real point
        //
        for jpt in 1..=num_real_pt {
            if jpt < num_real_pt {
                one_xyz_by_regression(
                    &a,
                    &b,
                    &c,
                    &d,
                    &e,
                    &f,
                    &asq,
                    &bsq,
                    &csq,
                    &axb,
                    &axc,
                    &bxc,
                    dxy,
                    xx,
                    yy,
                    ireal_start[(jpt - 1) as usize],
                    ireal_start[jpt as usize] - 1,
                    isec_view,
                    &mut bvec,
                );
            } else {
                //
                // get coordinates of last point as minus sum of all other points
                //
                for i in 1..=3 {
                    bsum = 0.;
                    for kpt in 1..=num_real_pt - 1 {
                        bsum = bsum - xyz[(kpt * 3 + i - 4) as usize];
                    }
                    bvec[(i - 1) as usize] = bsum;
                }
            }
            //
            // store new x, y, z
            //
            for i in 1..=3 {
                xyz[(jpt * 3 + i - 4) as usize] = bvec[(i - 1) as usize];
            }
            //
            // for each view that point is projected into, add its contribution
            // to the dx and dy differences in that view
            //
            for i in ireal_start[(jpt - 1) as usize]..=ireal_start[(jpt + 1 - 1) as usize] - 1 {
                iv = isec_view[(i - 1) as usize];
                let v = (iv - 1) as usize;
                dx_tmp = xx[(i - 1) as usize] - (a[v] * bvec[0] + b[v] * bvec[1] + c[v] * bvec[2]);
                dy_tmp = yy[(i - 1) as usize] - (d[v] * bvec[0] + e[v] * bvec[1] + f[v] * bvec[2]);
                dx_sum[v] = dx_sum[v] + dx_tmp;
                dxsq_sum[v] = dxsq_sum[v] + dx_tmp * dx_tmp;
                dy_sum[v] = dy_sum[v] + dy_tmp;
                dysq_sum[v] = dysq_sum[v] + dy_tmp * dy_tmp;
                nsum[v] = nsum[v] + 1;
            }
            //
        }
        //
        // set new dx and dy equal to the average dx and dy for each view,
        // and compute total error using the sums of squares gotten above
        //
        *error = 0.;
        diff_max = 0.;
        for iv in 1..=nview {
            let v = (iv - 1) as usize;
            dx_new = dx_sum[v] / nsum[v] as f32;
            dy_new = dy_sum[v] / nsum[v] as f32;
            del_dx = dx_new - dxy[(iv * 2 - 2) as usize];
            del_dy = dy_new - dxy[(iv * 2 - 1) as usize];

            // ACCUM_MAX(diffMax, B3DABS(delDx)): if (B3DABS(delDx) > diffMax) ...
            let abs_dx = if del_dx >= 0. { del_dx } else { -del_dx };
            if abs_dx > diff_max {
                diff_max = abs_dx;
            }
            let abs_dy = if del_dy >= 0. { del_dy } else { -del_dy };
            if abs_dy > diff_max {
                diff_max = abs_dy;
            }
            dxy[(iv * 2 - 2) as usize] = dx_new;
            dxy[(iv * 2 - 1) as usize] = dy_new;
            //
            // save info to make possible big jumps in dx and dy next time round
            //
            if jump_fast == 1 {
                del_dxy_last[(2 * iv - 2) as usize] = del_dx;
                del_dxy_last[(2 * iv - 1) as usize] = del_dy;
                dxy_last[(2 * iv - 2) as usize] = dxy[(iv * 2 - 2) as usize];
                dxy_last[(2 * iv - 1) as usize] = dxy[(iv * 2 - 1) as usize];
            }
            *error = *error + dxsq_sum[v] as f64 + dysq_sum[v] as f64
                - (nsum[v] as f32
                    * (dxy[(iv * 2 - 2) as usize] * dxy[(iv * 2 - 2) as usize]
                        + dxy[(iv * 2 - 1) as usize] * dxy[(iv * 2 - 1) as usize]))
                    as f64;
        }
        //
        // stick error on list, calculate difference from last error
        //
        err_list[(*isolve - 1) as usize] = *error as f32;
        if *isolve > 1 {
            let cur = err_list[(*isolve - 1) as usize];
            let prev = err_list[(*isolve - 1 - 1) as usize];
            let diff = cur - prev;
            err_diff =
                (if diff >= 0. { diff } else { -diff }) / (if cur > prev { cur } else { prev });
            let _ = err_diff;
        }
        *isolve = *isolve + 1;
        //
        // try to make big jumps in dx or dy; even this stuff doesn't make big
        // data sets converge adequately
        //
        if jump_fast == 2 {
            for iv in 1..=nview {
                for i in 1..=2 {
                    ncyc_skip = 1;
                    del2 = dxy[(iv * 2 + i - 3) as usize] - dxy_last[(i + 2 * iv - 3) as usize];
                    del1 = del_dxy_last[(i + 2 * iv - 3) as usize];
                    if del2 * del1 > 0.
                        && (if del2 >= 0. { del2 } else { -del2 })
                            < (if del1 >= 0. { del1 } else { -del1 })
                    {
                        ncyc_skip = ncyc_skip_lim;
                        let d12 = del1 - del2;
                        if (if d12 >= 0. { d12 } else { -d12 })
                            > (if del1 >= 0. { del1 } else { -del1 }) / ncyc_skip_lim as f32
                        {
                            ncyc_skip = (del1 / (del1 - del2)) as i32;
                        }
                    }
                    del_new = ncyc_skip as f32 * del1
                        + (del2 - del1) * ncyc_skip as f32 * (ncyc_skip + 1) as f32 / 2.;
                    dxy[(iv * 2 + i - 3) as usize] = dxy_last[(i + 2 * iv - 3) as usize] + del_new;
                }
            }
        }
        //
        if *isolve > jump_base {
            jump_fast = (jump_fast + 1) % 3;
        }
        //
        if iwatch > 0 {
            del_dx = dxy[(iwatch * 2 - 2) as usize] - dx_last;
            del_dy = dxy[(iwatch * 2 - 1) as usize] - dy_last;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "%4d%4d%10.4f%10.4f%10.4f%10.4f%20.5f\n",
                &[
                    CArg::Int(iwatch as i64),
                    CArg::Int(*isolve as i64),
                    CArg::Dbl(dxy[(iwatch * 2 - 2) as usize] as f64),
                    CArg::Dbl(del_dx as f64),
                    CArg::Dbl(dxy[(iwatch * 2 - 1) as usize] as f64),
                    CArg::Dbl(del_dy as f64),
                    CArg::Dbl(*error),
                ],
            ));
            dx_last = dxy[(iwatch * 2 - 2) as usize];
            dy_last = dxy[(iwatch * 2 - 1) as usize];
        }
    }
    //
    // after convergence, recompute error de novo: this may give more
    // accurate double precision result, or may be superfluous

    *error = 0.;
    for jpt in 1..=num_real_pt {
        for i in ireal_start[(jpt - 1) as usize]..=ireal_start[(jpt + 1 - 1) as usize] - 1 {
            iv = isec_view[(i - 1) as usize];
            let v = (iv - 1) as usize;
            let px = a[v] * xyz[(jpt * 3 - 3) as usize]
                + b[v] * xyz[(jpt * 3 - 2) as usize]
                + c[v] * xyz[(jpt * 3 - 1) as usize]
                + dxy[(iv * 2 - 2) as usize]
                - xx[(i - 1) as usize];
            let py = d[v] * xyz[(jpt * 3 - 3) as usize]
                + e[v] * xyz[(jpt * 3 - 2) as usize]
                + f[v] * xyz[(jpt * 3 - 1) as usize]
                + dxy[(iv * 2 - 1) as usize]
                - yy[(i - 1) as usize];
            *error = *error + (px * px) as f64 + (py * py) as f64;
        }
    }
    //
    err_list[(*isolve - 1) as usize] = *error as f32;
    // write(*,'(4f20.5)') (errList(i), i=1, isolve)
    if iwatch > 0 {
        iwatch = iwatch % nview + 1;
        let _ = iwatch;
    }
}

/// Original: `init_dxy` (`solve_xyzd.cpp:725`, file static).
///
/// Sets the initial values of dx and dy for each view: it sets dx and dy to
/// the given centroid for the view with minimum tilt (`iminTilt`), then sets
/// the dx and dy of each view so that the points shared between adjacent views
/// have the same center of gravity.
fn init_dxy(
    xx: &[f32],
    yy: &[f32],
    isec_view: &[i32],
    ireal_start: &[i32],
    nview: i32,
    num_real_pt: i32,
    imin_tilt: i32,
    dxy: &mut [f32],
    cgx: f32,
    cgy: f32,
) {
    let mut dx_sum: f32;
    let mut dy_sum: f32;
    let iv_start: i32;
    let mut iv_end: i32;
    let mut iv: i32;
    let mut nsum: i32;
    let mut in_one: i32;
    let mut in_two: i32;
    //
    // consider each pair of views from the one with min tilt out to ends
    //
    dxy[(imin_tilt * 2 - 2) as usize] = cgx;
    dxy[(imin_tilt * 2 - 1) as usize] = cgy;
    iv_start = imin_tilt;
    iv_end = 2;
    let mut idir = -1;
    while idir <= 1 {
        iv = iv_start;
        while idir * (iv - iv_end) <= 0 {
            //
            // find each real point in both views, add up disparities
            //
            dx_sum = 0.;
            dy_sum = 0.;
            nsum = 0;
            for ipt in 1..=num_real_pt {
                in_one = 0;
                in_two = 0;
                for i in ireal_start[(ipt - 1) as usize]..=ireal_start[(ipt + 1 - 1) as usize] - 1 {
                    if isec_view[(i - 1) as usize] == iv + idir {
                        in_one = i;
                    }
                    if isec_view[(i - 1) as usize] == iv {
                        in_two = i;
                    }
                }
                if in_one * in_two > 0 {
                    dx_sum = dx_sum + xx[(in_two - 1) as usize] - xx[(in_one - 1) as usize];
                    dy_sum = dy_sum + yy[(in_two - 1) as usize] - yy[(in_one - 1) as usize];
                    nsum = nsum + 1;
                }
            }
            //
            // adjust this section's dx, dy by the disparity
            //
            dxy[((iv + idir - 1) * 2) as usize] = dxy[(iv * 2 - 2) as usize] + dx_sum / nsum as f32;
            dxy[((iv + idir - 1) * 2 + 1) as usize] =
                dxy[(iv * 2 - 1) as usize] + dy_sum / nsum as f32;
            iv += idir;
        }
        iv_end = nview - 1;
        idir += 2;
    }
}

/// Original: `regressionCrossProducts` (`solve_xyzd.cpp:773`).
///
/// Takes the 6 factors of x, y, z and computes the cross-products and squares
/// needed for quick regression.
pub fn regression_cross_products(
    nview: i32,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    e: &[f32],
    f: &[f32],
    asq: &mut [f32],
    bsq: &mut [f32],
    csq: &mut [f32],
    axb: &mut [f32],
    axc: &mut [f32],
    bxc: &mut [f32],
) {
    for i in 0..nview as usize {
        asq[i] = a[i] * a[i] + d[i] * d[i];
        bsq[i] = b[i] * b[i] + e[i] * e[i];
        csq[i] = c[i] * c[i] + f[i] * f[i];
        axb[i] = a[i] * b[i] + d[i] * e[i];
        axc[i] = a[i] * c[i] + d[i] * f[i];
        bxc[i] = b[i] * c[i] + e[i] * f[i];
    }
}

/// Original: `oneXYZbyRegression` (`solve_xyzd.cpp:793`).
///
/// Solves for the x, y, z coordinates of a single fiducial given the limits of
/// its projection points, the set of all projection points, and the 6 factors
/// and cross-products.
///
/// Fixed in translation (2026-09-26, `BUGS.md`): the source's first Cramer
/// numerator ends `am13 * (bv1 * am32 - am22 * bv3)` (`:838`), where the
/// determinant with the right-hand side in column 1 has `bv2 * am32`, so its
/// x is wrong unless `bv1 == bv2`.  Here the term is `bv2 * am32`.
pub fn one_xyz_by_regression(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    e: &[f32],
    f: &[f32],
    asq: &[f32],
    bsq: &[f32],
    csq: &[f32],
    axb: &[f32],
    axc: &[f32],
    bxc: &[f32],
    dxy: &[f32],
    xx: &[f32],
    yy: &[f32],
    ind_start: i32,
    ind_end: i32,
    isec_view: &[i32],
    bvec: &mut [f32],
) {
    let mut iv: i32;
    let mut ypr: f32;
    let mut am11: f32;
    let mut am12: f32;
    let mut am13: f32;
    let am21: f32;
    let mut am22: f32;
    let mut am23: f32;
    let am31: f32;
    let am32: f32;
    let mut am33: f32;
    let det: f32;
    let mut bv1: f32;
    let mut bv2: f32;
    let mut bv3: f32;
    let mut xpr: f32;
    //
    // zero the matrix elements for sums of squares and cross-products
    am11 = 0.;
    am12 = 0.;
    am13 = 0.;
    am22 = 0.;
    am23 = 0.;
    am33 = 0.;
    bv1 = 0.;
    bv2 = 0.;
    bv3 = 0.;
    //
    // for each projection of the real point, add terms to matrix
    for i in ind_start..=ind_end {
        iv = isec_view[(i - 1) as usize];
        let v = (iv - 1) as usize;
        xpr = xx[(i - 1) as usize] - dxy[(iv * 2 - 2) as usize];
        ypr = yy[(i - 1) as usize] - dxy[(iv * 2 - 1) as usize];
        am11 = am11 + asq[v];
        am22 = am22 + bsq[v];
        am33 = am33 + csq[v];
        am12 = am12 + axb[v];
        am13 = am13 + axc[v];
        am23 = am23 + bxc[v];
        bv1 = bv1 + a[v] * xpr + d[v] * ypr;
        bv2 = bv2 + b[v] * xpr + e[v] * ypr;
        bv3 = bv3 + c[v] * xpr + f[v] * ypr;
    }
    //
    // fill out lower triangle of matrix
    am31 = am13;
    am21 = am12;
    am32 = am23;
    //
    // solve it
    det = am11 * (am22 * am33 - am23 * am32) - am12 * (am21 * am33 - am23 * am31)
        + am13 * (am21 * am32 - am22 * am31);
    bvec[0] = (bv1 * (am22 * am33 - am23 * am32) - am12 * (bv2 * am33 - am23 * bv3)
        + am13 * (bv2 * am32 - am22 * bv3))
        / det;
    bvec[1] = (am11 * (bv2 * am33 - am23 * bv3) - bv1 * (am21 * am33 - am23 * am31)
        + am13 * (am21 * bv3 - bv2 * am31))
        / det;
    bvec[2] = (am11 * (am22 * bv3 - bv2 * am32) - am12 * (am21 * bv3 - bv2 * am31)
        + bv1 * (am21 * am32 - am22 * am31))
        / det;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BUGS.md` "`oneXYZbyRegression`": with the source's `bv1 * am32` the
    /// x of an exactly determined point is wrong whenever `bv1 != bv2`.  The
    /// fixed Cramer numerator recovers a point from noise-free projections.
    #[test]
    fn one_xyz_by_regression_recovers_the_point() {
        // Three views: tilt about Y by -30, 0, 40 degrees, a small rotation.
        let angles = [-30.0_f32, 0.0, 40.0];
        let rot = [0.1_f32, -0.05, 0.2];
        let (mut a, mut b, mut c, mut d, mut e, mut f) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for (t, r) in angles.iter().zip(rot.iter()) {
            let (ct, st) = (t.to_radians().cos(), t.to_radians().sin());
            let (cr, sr) = (r.cos(), r.sin());
            // Projection of (x, y, z): tilt about Y, then rotate in the plane.
            a.push(cr * ct);
            b.push(-sr);
            c.push(cr * st);
            d.push(sr * ct);
            e.push(cr);
            f.push(sr * st);
        }
        let n = angles.len();
        let sq = |p: &[f32], q: &[f32], r: &[f32], s: &[f32]| -> Vec<f32> {
            (0..n).map(|i| p[i] * q[i] + r[i] * s[i]).collect()
        };
        let (asq, bsq, csq) = (sq(&a, &a, &d, &d), sq(&b, &b, &e, &e), sq(&c, &c, &f, &f));
        let (axb, axc, bxc) = (sq(&a, &b, &d, &e), sq(&a, &c, &d, &f), sq(&b, &c, &e, &f));
        let dxy = vec![1.5_f32, -2.0, 0.5, 0.25, -1.0, 3.0];
        let (px, py, pz) = (37.0_f32, -12.0_f32, 8.0_f32);
        let mut xx = vec![];
        let mut yy = vec![];
        for i in 0..n {
            xx.push(a[i] * px + b[i] * py + c[i] * pz + dxy[2 * i]);
            yy.push(d[i] * px + e[i] * py + f[i] * pz + dxy[2 * i + 1]);
        }
        let isec_view = [1, 2, 3];
        let mut bvec = [0f32; 3];
        one_xyz_by_regression(
            &a, &b, &c, &d, &e, &f, &asq, &bsq, &csq, &axb, &axc, &bxc, &dxy, &xx, &yy, 1, 3,
            &isec_view, &mut bvec,
        );
        for (got, want) in bvec.iter().zip([px, py, pz]) {
            assert!((got - want).abs() < 1.0e-2, "{bvec:?} vs {px} {py} {pz}");
        }
    }
}
