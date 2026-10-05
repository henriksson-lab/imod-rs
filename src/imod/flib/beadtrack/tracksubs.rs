//! Translation of `IMOD/flib/beadtrack/tracksubs.cpp` — the free-standing
//! functions (formerly Fortran subroutines) of `beadtrack`: projecting a
//! bead's next position from its neighbours, locating the image piece that
//! holds a box, centroid and edge statistics of a bead box, the rescue
//! search, elongation, outlier-eliminating transform fits, and adding points
//! to the model.
//!
//! # File-scope statics
//!
//! `static CGPixels *cp; static ArrayMaxes *mx;` (`tracksubs.cpp:18-19`), set
//! by `tracksubsSetPointers`, are parameters here, per `tiltalign/alivar.rs`:
//! `&CGPixels`/`&mut CGPixels` for the functions that dereference `cp`
//! (`&mut` wherever `edgeForCG` fills `cp->edgePixels`), `&ArrayMaxes` for
//! `nextPos`'s `mx->maxView`.  The `fmod*` globals of `fortmodel.h` are the
//! fields of the `FortModel` passed in, as in `proc_model.rs`:
//! `fmodP_coord[k * 3 - 3 + j]` is `fm.p_coord[k - 1][j]`.
//!
//! `calcElongation` passes `cp->elongSmooth` as the `boxTmp` of
//! `bestCenterForCG` and `edgeForCG` while those also dereference `cp`; the
//! array is lent out of the struct with `mem::take` for the duration and put
//! back before every return, which is the same storage.
//!
//! # Not translated
//!
//! The three-argument `errorExit` (`tracksubs.cpp:22`, a leftover Fortran
//! entry point whose only caller is commented out, `beadtrack.cpp:1904-1907`)
//! and `rescueFromSobel` (`tracksubs.cpp:601`, "Not used, did not help"; its
//! only call is commented out at `beadtrack.cpp:2895`) are dead in the source
//! and recorded in `DEAD_CODE.md`.  `beadtrack` reaches the two-argument
//! `errorExit` of `utilfuncs.cpp` (`utilfuncs.rs`), a distinct C++ overload.
//!
//! # Arithmetic widths
//!
//! `B3DNINT(a)` is `(int)floor((a) + 0.5)` with a `double` `0.5`, so a
//! `float` operand widens first.  Every `pow(float, 2.f)` resolves to
//! libstdc++'s `float` overload (`__builtin_powf`), which GCC folds to a
//! single-precision square: the reference `tracksubs.o` imports no `powf`,
//! and only `pow` for the two `double` probabilities of `findxf_wo_outliers`.

use std::io::Write;

use super::cgpixels::CGPixels;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::tiltalign::arraymaxes::ArrayMaxes;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_i_max, b3d_i_min, c_format_bytes, number_in_list,
};
use crate::imod::libcfshr::filtxcorr::apply_kernel_filter;
use crate::imod::libcfshr::findtransform::find_transform;
use crate::imod::libcfshr::linearxforms::xf_apply;
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libcfshr::robuststat::rs_fast_median_in_place;
use crate::imod::libcfshr::simplestat::{avg_sd, ls_fit, ls_fit2};
use crate::imod::libcfshr::statfuncs::err_func;
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_sample, iiu_alt_size, iiu_ret_delta,
};
use crate::imod::libimod::fortmodel::fort_object_mover;

/// `RADIANS_PER_DEGREE` (`b3dutil.h:68`), a `double` literal.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `B3DNINT(a)` (`b3dutil.h:33`, `(int)floor((a) + 0.5)`) with the conversion
/// the reference build emits: `cvttsd2si`, `INT_MIN` for NaN or out of range.
macro_rules! b3dnint_cvtt {
    ($x:expr) => {{
        let v: f64 = (($x) as f64 + 0.5).floor();
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is part of the x86-64 baseline.
            unsafe { core::arch::x86_64::_mm_cvttsd_si32(core::arch::x86_64::_mm_set_sd(v)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            v as i32
        }
    }};
}

/// Original: `cosd` (`tracksubs.cpp:27`).
///
/// `angle * RADIANS_PER_DEGREE` is a `double` product, narrowed to `float`
/// for `cosf`.
pub fn cosd(angle: f32) -> f32 {
    ((angle as f64 * RADIANS_PER_DEGREE) as f32).cos()
}

/// Original: `sind` (`tracksubs.cpp:31`).
pub fn sind(angle: f32) -> f32 {
    ((angle as f64 * RADIANS_PER_DEGREE) as f32).sin()
}

/// Original: `tracksubsSetPointers` (`tracksubs.cpp:36`).
///
/// The source stores the two pointers in file-scope statics; the functions
/// of this module take them as parameters instead, so there is nothing to
/// store.
pub fn tracksubs_set_pointers(cgpx: &mut CGPixels, mx_in: &mut ArrayMaxes) {
    let _ = (cgpx, mx_in);
}

/// Original: `nextPos` (`tracksubs.cpp:58`).
///
/// Computes the projected position of a point from positions on nearby views
/// in the absence of a tilt alignment.  `izExclude` is `tc->izExclude`, empty
/// where the source has `NULL` with `numExclude == 0`; `numberInList` returns
/// its no-list value for either.
#[allow(clippy::too_many_arguments)]
pub fn next_pos(
    fm: &FortModel,
    mx: &ArrayMaxes,
    iobj: i32,
    ip_near: i32,
    idir: i32,
    iz_next: i32,
    tilt: &[f32],
    max_fit: i32,
    min_fit: i32,
    axis_rot: f32,
    tilt_min: f32,
    iz_exclude: &[i32],
    num_exclude: i32,
    xnext: &mut f32,
    ynext: &mut f32,
    min_tilt_ind: i32,
    minz_delz_near_zero: i32,
    max_delz_near_zero: i32,
    iz_del_to_near: &mut i32,
) {
    let mut b1 = [0f32; 2];
    let mut yrot = vec![0f32; mx.max_view as usize];
    let mut xx = vec![0f32; mx.max_view as usize];
    let mut yy = vec![0f32; mx.max_view as usize];
    let mut zz = vec![0f32; mx.max_view as usize];
    let mut xrot = vec![0f32; mx.max_view as usize];
    let cos_rot: f32;
    let sin_rot: f32;
    let ibase: i32;
    let num_in_obj: i32;
    let mut mfit: i32;
    let mut ip_end: i32;
    let ipt_near: i32;
    let mut ip: i32;
    let mut ipt: i32;
    let mut ipb: i32 = 0;
    let mut ipp: i32 = 0;
    let mut i_past: i32;
    let mut i_before: i32;
    let mut iz: i32;
    let mut xsum: f32;
    let mut ysum: f32;
    let mut slope: f32 = 0.;
    let mut bint: f32 = 0.;
    let mut ro: f32 = 0.;
    let mut cons: f32 = 0.;
    let theta_next: f32;
    let xtmp: f32;
    let ytmp: f32;
    let mut theta: f32;
    let excl = Some(iz_exclude);
    // `fmodP_coord[k * 3 - 1]`, the Z of 1-based point `k`.
    let zc = |k: i32| fm.p_coord[(k - 1) as usize][2];

    ibase = fm.ibase_obj[(iobj - 1) as usize];
    num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
    cos_rot = cosd(axis_rot);
    sin_rot = sind(axis_rot);
    *iz_del_to_near = 0;
    mfit = 0;
    ip_end = 1;
    if idir == -1 {
        ip_end = num_in_obj;
    }
    //
    // set pointer past the view if possible
    //
    ipt_near = fm.object[(ibase + ip_near - 1) as usize];
    ip = ip_near;
    ipt = ip_near;
    while idir * (ipt - ip_end) >= 0 {
        iz = (zc(fm.object[(ibase + ipt - 1) as usize]) as f64 + 0.5).floor() as i32;
        if idir * (iz - iz_next) < 0 {
            break;
        }
        if number_in_list(iz + 1, excl, num_exclude, 0) == 0 {
            ip = ipt;
        }
        ipt -= idir;
    }
    if idir * ((zc(fm.object[(ibase + ip - 1) as usize]) as f64 + 0.5).floor() as i32 - iz_next)
        == -1
    {
        //
        // if point is adjacent in that direction, then starting from there,
        // load maxFit points
        //
        while idir * (ip - ip_end) >= 0 && mfit < max_fit {
            ipt = fm.object[(ibase + ip - 1) as usize];
            if number_in_list(
                (zc(ipt) as f64 + 0.5).floor() as i32 + 1,
                excl,
                num_exclude,
                0,
            ) == 0
            {
                let p = fm.p_coord[(ipt - 1) as usize];
                xx[mfit as usize] = p[0];
                yy[mfit as usize] = p[1];
                zz[mfit as usize] = p[2];
                mfit += 1;
            }
            ip -= idir;
        }
    } else if (zc(ipt_near) as f64 + 0.5).floor() as i32 == iz_next {
        //
        // otherwise, if there is already a point on the section, just take it
        //
        mfit = 1;
        xx[0] = fm.p_coord[(ipt_near - 1) as usize][0];
        yy[0] = fm.p_coord[(ipt_near - 1) as usize][1];
        zz[0] = iz_next as f32;
    } else {
        //
        // otherwise, set pointers to points past and before view and get
        // points from both directions
        //
        i_past = ip_near;
        if zc(fm.object[(ibase + ip_near - 1) as usize]) < iz_next as f32 {
            i_past = ip_near + 1;
        }
        i_before = i_past - 1;
        //
        // starting from there, load maxFit nearest points
        //
        while (i_before > 0 || i_past <= num_in_obj) && mfit < max_fit {
            if i_before > 0 {
                ipb = fm.object[(ibase + i_before - 1) as usize];
            }
            if i_past <= num_in_obj {
                ipp = fm.object[(ibase + i_past - 1) as usize];
            }
            //
            // take the only one that's legal
            //
            if i_before <= 0 {
                ipt = ipp;
                i_past += 1;
            } else if i_past > num_in_obj {
                ipt = ipb;
                i_before -= 1;
            } else {
                //
                // or if both are legal, take the one closest to target view
                //
                let db = zc(ipb) - iz_next as f32;
                let dp = zc(ipp) - iz_next as f32;
                if (if db >= 0. { db } else { -db }) < (if dp >= 0. { dp } else { -dp }) {
                    ipt = ipb;
                    i_before -= 1;
                } else {
                    ipt = ipp;
                    i_past += 1;
                }
            }
            if number_in_list(
                (zc(ipt) as f64 + 0.5).floor() as i32 + 1,
                excl,
                num_exclude,
                0,
            ) == 0
            {
                let p = fm.p_coord[(ipt - 1) as usize];
                xx[mfit as usize] = p[0];
                yy[mfit as usize] = p[1];
                zz[mfit as usize] = p[2];
                mfit += 1;
            }
        }
    }
    //
    // Just average the points if there are not enough
    if mfit < min_fit {
        //
        // If there are no points at all, take the nearest even though it is on excluded view
        let d1 = (iz_next + 1) - min_tilt_ind;
        let d2 = iz_next - (zc(ipt_near) as f64 + 0.5).floor() as i32;
        if mfit == 0
            || (mfit == 1
                && minz_delz_near_zero > 0
                && (if d1 >= 0 { d1 } else { -d1 }) <= minz_delz_near_zero
                && (if d2 >= 0 { d2 } else { -d2 }) <= max_delz_near_zero)
        {
            mfit = 1;
            let _ = mfit;
            *xnext = fm.p_coord[(ipt_near - 1) as usize][0];
            *ynext = fm.p_coord[(ipt_near - 1) as usize][1];
            *iz_del_to_near = iz_next - (zc(ipt_near) as f64 + 0.5).floor() as i32;
        } else {
            xsum = 0.;
            ysum = 0.;
            for i in 0..mfit as usize {
                xsum += xx[i];
                ysum += yy[i];
            }
            *xnext = xsum / mfit as f32;
            *ynext = ysum / mfit as f32;
        }
    } else if mfit >= min_fit + 2 && {
        let t = tilt[iz_next as usize];
        (if t >= 0. { t } else { -t }) >= tilt_min
    } {
        //
        // Or, if there are enough for sine-cosine fit and angle is big enough, rotate the
        // points so the tilt axis is along Y, fit Y to Z and X to cos/sin, and rotate
        // the result back.  This may never happen unless there are too few points for
        // tiltalign, but it was tested July 2012 and gave possibly even better results
        for i in 0..mfit as usize {
            xrot[i] = cos_rot * xx[i] + sin_rot * yy[i];
            yrot[i] = -sin_rot * xx[i] + cos_rot * yy[i];
            theta = tilt[(zz[i] as f64 + 0.5).floor() as i32 as usize];
            xx[i] = cosd(theta);
            yy[i] = sind(theta);
        }
        ls_fit(&zz, &yrot, mfit, &mut slope, &mut bint, &mut ro);
        ytmp = iz_next as f32 * slope + bint;
        let (b1a, b1b) = b1.split_at_mut(1);
        ls_fit2(
            &xx,
            &yy,
            &xrot,
            mfit,
            &mut b1a[0],
            &mut b1b[0],
            Some(&mut cons),
        );
        theta_next = tilt[iz_next as usize];
        xtmp = b1[0] * cosd(theta_next) + b1[1] * sind(theta_next) + cons;
        *xnext = cos_rot * xtmp - sin_rot * ytmp;
        *ynext = sin_rot * xtmp + cos_rot * ytmp;
        //printf("sin/cos fit: %4d %2d %9.2f %9.2f %3d\n", iobj, mfit, xnext, ynext, izNext);
    } else {
        //
        // Or just do straight line fits to both components
        ls_fit(&zz, &xx, mfit, &mut slope, &mut bint, &mut ro);
        *xnext = iz_next as f32 * slope + bint;
        ls_fit(&zz, &yy, mfit, &mut slope, &mut bint, &mut ro);
        *ynext = iz_next as f32 * slope + bint;
        //printf("linear fit: %4d %2d %.2f %.2f\n", iobj, mfit, xnext, ynext);
    }
    let _ = ImodFile::Stdout.flush();
}

/// Original: `findPiece` (`tracksubs.cpp:223`).
///
/// Finds the piece (`ipcz`, 0-based, -1 for none) holding a box of size
/// `nxBox` x `nyBox` centred at `xnext, ynext` on view `izNext`, and the box's
/// index range in that piece.  `ix0..iy1` are written only on the paths the
/// source assigns them.  `prexf` is read only when `ifXfs` is non-zero.
///
/// `b3dIMax(3, a, b, c)` / `b3dIMin(3, ...)` take a count first
/// (`b3dutil.c:1307`), which the slice form drops.
#[allow(clippy::too_many_arguments)]
pub fn find_piece(
    ix_pclist: &[i32],
    iy_pclist: &[i32],
    iz_pclist: &[i32],
    n_pclist: i32,
    nx: i32,
    ny: i32,
    nx_box: i32,
    ny_box: i32,
    xnext: f32,
    ynext: f32,
    iz_next: i32,
    ix0: &mut i32,
    ix1: &mut i32,
    iy0: &mut i32,
    iy1: &mut i32,
    ipcz: &mut i32,
    if_xfs: i32,
    prexf: &[f32],
    need_taper: &mut bool,
    need_fill: &mut bool,
) {
    let crit_non_blank: f32;
    let mut x_ok: bool;
    let mut y_ok: bool;
    let indx0: i32;
    let indx1: i32;
    let indy0: i32;
    let indy1: i32;
    let mut ind_good0: i32;
    let mut ind_good1: i32;
    let mut nx_good: i32;
    let mut ny_good: i32;
    let mut ipc_at_z: i32;
    //
    crit_non_blank = 0.75;
    // Fixed in translation (2026-09-26, `BUGS.md`): `B3DNINT` converts with
    // `cvttsd2si`, which gives `INT_MIN` for a NaN or out-of-range position (a
    // NaN `xnext` comes from a degenerate `findxf_wo_outliers` fit); the index
    // arithmetic then wraps, the box is accepted, and native dies reading the
    // image.  Here a position that is not a representable pixel index is in no
    // piece: the bead is simply not found on this view.
    if !(xnext.is_finite() && ynext.is_finite() && xnext.abs() < 1.0e9 && ynext.abs() < 1.0e9) {
        *ipcz = -1;
        *need_taper = false;
        *need_fill = false;
        return;
    }
    indx0 = b3dnint_cvtt!(xnext).wrapping_sub(nx_box / 2);
    indx1 = indx0.wrapping_add(nx_box).wrapping_sub(1);
    indy0 = b3dnint_cvtt!(ynext).wrapping_sub(ny_box / 2);
    indy1 = indy0.wrapping_add(ny_box).wrapping_sub(1);
    *ipcz = -1;
    *need_taper = false;
    *need_fill = false;
    nx_good = nx_box;
    ny_good = ny_box;
    if if_xfs != 0 {
        // Fixed in translation (2026-10-03, `BUGS.md`): a point whose Z is not a
        // view of the transform list (`transferfid` hands beadtrack a model with
        // points at Z -1 for its two-view stack) makes the source read
        // `prexf[-2]`/`prexf[-1]`, before the array.  Such a shift is taken as
        // 0; no piece lies at that Z, so the bead is not found there either way.
        let shift = |k: i32| -> i32 {
            let ind = iz_next.wrapping_mul(6).wrapping_add(k);
            if iz_next < 0 || ind < 0 || ind as usize >= prexf.len() {
                0
            } else {
                (prexf[ind as usize] as f64 + 0.5).floor() as i32
            }
        };
        let dx = shift(4);
        let dy = shift(5);
        ind_good0 = b3d_i_max(&[indx0, 0, dx]);
        ind_good1 = b3d_i_min(&[indx1, nx - 1, nx + dx - 1]);
        nx_good = if 0 > ind_good1.wrapping_add(1).wrapping_sub(ind_good0) {
            0
        } else {
            ind_good1.wrapping_add(1).wrapping_sub(ind_good0)
        };
        ind_good0 = b3d_i_max(&[indy0, 0, dy]);
        ind_good1 = b3d_i_min(&[indy1, ny - 1, ny + dy - 1]);
        ny_good = if 0 > ind_good1.wrapping_add(1).wrapping_sub(ind_good0) {
            0
        } else {
            ind_good1.wrapping_add(1).wrapping_sub(ind_good0)
        };
        if ((nx_good * ny_good) as f32) < crit_non_blank * nx_box as f32 * ny_box as f32 {
            return;
        }
        *need_taper = nx_good < nx_box || ny_good < ny_box;
    }
    //
    for ipc in 1..=n_pclist {
        let k = (ipc - 1) as usize;
        if iz_next == iz_pclist[k]
            && indx0 >= ix_pclist[k]
            && indx1 < ix_pclist[k] + nx
            && indy0 >= iy_pclist[k]
            && indy1 < iy_pclist[k] + ny
        {
            *ipcz = ipc - 1;
            *ix0 = indx0.wrapping_sub(ix_pclist[k]);
            *ix1 = indx1.wrapping_sub(ix_pclist[k]);
            *iy0 = indy0.wrapping_sub(iy_pclist[k]);
            *iy1 = indy1.wrapping_sub(iy_pclist[k]);
            return;
        }
    }
    //
    // See if it can load a partial box and taper it
    ipc_at_z = 0;
    for ipc in 1..=n_pclist {
        let k = (ipc - 1) as usize;
        if iz_next == iz_pclist[k] {
            if ipc_at_z > 0 {
                return;
            }
            x_ok = indx0 >= ix_pclist[k] && indx1 < ix_pclist[k] + nx;
            y_ok = indy0 >= iy_pclist[k] && indy1 < iy_pclist[k] + ny;
            *need_fill = true;

            // One direction must be OK, so there is missing data just on one side
            if y_ok && indx0 < ix_pclist[k] && indx1 > ix_pclist[k] {
                nx_good -= ix_pclist[k] - indx0;
            } else if y_ok && indx1 >= ix_pclist[k] + nx && indx0 < ix_pclist[k] + nx {
                nx_good -= indx1 - (ix_pclist[k] + nx - 1);
            } else if x_ok && indy0 < iy_pclist[k] && indy1 > iy_pclist[k] {
                ny_good -= iy_pclist[k] - indy0;
            } else if x_ok && indy1 >= iy_pclist[k] + ny && indy0 < iy_pclist[k] + ny {
                ny_good -= indy1 - (iy_pclist[k] + ny - 1);
            } else {
                *need_fill = false;
            }
            if !*need_fill
                || ((nx_good * ny_good) as f32) < crit_non_blank * nx_box as f32 * ny_box as f32
            {
                return;
            }
            ipc_at_z = ipc;
            *ix0 = if 0 > indx0 - ix_pclist[k] {
                0
            } else {
                indx0 - ix_pclist[k]
            };
            *ix1 = if nx - 1 < indx1 - ix_pclist[k] {
                nx - 1
            } else {
                indx1 - ix_pclist[k]
            };
            *iy0 = if 0 > indy0 - iy_pclist[k] {
                0
            } else {
                indy0 - iy_pclist[k]
            };
            *iy1 = if ny - 1 < indy1 - iy_pclist[k] {
                ny - 1
            } else {
                indy1 - iy_pclist[k]
            };
        }
    }
    *ipcz = ipc_at_z - 1;
}

/// Original: `QDshift` (`tracksubs.cpp:307`).
///
/// A simple image shift with quadratic interpolation.  `a..d` are `float`
/// locals assigned from `double` expressions (`* .5`).
pub fn qd_shift(array: &[f32], bray: &mut [f32], nx: i32, ny: i32, xt: f32, yt: f32) {
    let dx: f32;
    let dy: f32;
    let dxsq: f32;
    let dysq: f32;
    let (mut a, mut b, mut c, mut d): (f32, f32, f32, f32);
    let (mut v2, mut v4, mut v6, mut v8, mut v5): (f32, f32, f32, f32, f32);
    let (mut ixpp1, mut ixpm1, mut iypp1, mut iypm1): (i32, i32, i32, i32);
    //
    // Loop over output image
    //
    dx = -xt;
    dy = -yt;
    dxsq = dx * dx;
    dysq = dy * dy;
    for iyp in 0..ny {
        for ixp in 0..nx {
            //
            // do quadratic interpolation
            //
            ixpp1 = if nx - 1 < ixp + 1 { nx - 1 } else { ixp + 1 };
            ixpm1 = if 0 > ixp - 1 { 0 } else { ixp - 1 };
            iypp1 = if ny - 1 < iyp + 1 { ny - 1 } else { iyp + 1 };
            iypm1 = if 0 > iyp - 1 { 0 } else { iyp - 1 };
            //
            // set up terms for quadratic interpolation
            //
            v2 = array[(iypm1 * nx + ixp) as usize];
            v4 = array[(iyp * nx + ixpm1) as usize];
            v5 = array[(iyp * nx + ixp) as usize];
            v6 = array[(iyp * nx + ixpp1) as usize];
            v8 = array[(iypp1 * nx + ixp) as usize];
            //
            a = ((v6 + v4) as f64 * 0.5 - v5 as f64) as f32;
            b = ((v8 + v2) as f64 * 0.5 - v5 as f64) as f32;
            c = ((v6 - v4) as f64 * 0.5) as f32;
            d = ((v8 - v2) as f64 * 0.5) as f32;
            //

            bray[(iyp * nx + ixp) as usize] = a * dxsq + b * dysq + c * dx + d * dy + v5;
            //
        }
    }
}

/// Original: `peakFind` (`tracksubs.cpp:353`).
///
/// Finds the absolute peak of `array`, dimensioned `nxPlus` by `nyRot`, with
/// no interpolation.  `ixPeak`/`iyPeak` are uninitialised in the source when
/// no element exceeds `-1.e30` (all NaN, say); they start at 0 here.
pub fn peak_find(
    array: &[f32],
    nx_plus: i32,
    ny_rot: i32,
    xpeak: &mut f32,
    ypeak: &mut f32,
    peak: &mut f32,
) {
    let nx_rot: i32;
    // Fixed in translation (2026-09-26, `BUGS.md`): `ixPeak`/`iyPeak` are
    // uninitialised in the source and stay so when nothing exceeds -1.e30 (an
    // all-NaN array); here they start at the origin pixel, i.e. a zero shift.
    let mut ix_peak: i32 = 1;
    let mut iy_peak: i32 = 1;
    nx_rot = nx_plus - 2;
    //
    // find peak
    //
    *peak = -1.0e30f64 as f32;
    for iy in 1..=ny_rot {
        for ix in 1..=nx_rot {
            if array[((iy - 1) * nx_plus + ix - 1) as usize] > *peak {
                *peak = array[((iy - 1) * nx_plus + ix - 1) as usize];
                ix_peak = ix;
                iy_peak = iy;
            }
        }
    }
    // print *,ixPeak, iyPeak
    //
    // return adjusted pixel coordinate minus 1
    //
    *xpeak = (ix_peak as f64 - 1.) as f32;
    *ypeak = (iy_peak as f64 - 1.) as f32;
    if *xpeak > (nx_rot / 2) as f32 {
        *xpeak -= nx_rot as f32;
    }
    if *ypeak > (ny_rot / 2) as f32 {
        *ypeak -= ny_rot as f32;
    }
    // print *,xpeak, ypeak
}

/// Original: `edgeForCG` (`tracksubs.cpp:389`).
///
/// Mean or median of the edge pixels around `ixcen, iycen`, and their SD when
/// `cp->getEdgeSD` is set; `ierr` is 1 when fewer than half the edge points
/// fall in the box.
#[allow(clippy::too_many_arguments)]
pub fn edge_for_cg(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    ixcen: i32,
    iycen: i32,
    edge: &mut f32,
    edge_sd: &mut f32,
    ierr: &mut i32,
) {
    let mut iy: i32;
    let mut ix: i32;
    let mut nsum: i32;
    let mut sum: f32;
    sum = 0.;
    nsum = 0;
    *edge = 0.;
    *ierr = 1;
    //
    // find edge mean - require half the points to be present
    //
    if cp.edge_median != 0 || cp.get_edge_sd != 0 {
        for i in 0..cp.num_edge as usize {
            ix = ixcen + cp.idx_edge[i] - 1;
            iy = iycen + cp.idy_edge[i] - 1;
            if ix >= 0 && ix < nx_box && iy >= 0 && iy < ny_box {
                cp.edge_pixels[nsum as usize] = box_tmp[(iy * nx_box + ix) as usize];
                nsum += 1;
            }
        }
    } else {
        for i in 0..cp.num_edge as usize {
            ix = ixcen + cp.idx_edge[i] - 1;
            iy = iycen + cp.idy_edge[i] - 1;
            if ix >= 0 && ix < nx_box && iy >= 0 && iy < ny_box {
                sum += box_tmp[(iy * nx_box + ix) as usize];
                nsum += 1;
            }
        }
    }
    if nsum < cp.num_edge / 2 {
        return;
    }
    *ierr = 0;
    if cp.get_edge_sd != 0 {
        avg_sd(&cp.edge_pixels, nsum, edge, edge_sd, &mut sum);
    }
    if cp.edge_median != 0 {
        rs_fast_median_in_place(&mut cp.edge_pixels, nsum, edge);
    } else if cp.get_edge_sd == 0 {
        *edge = sum / nsum as f32;
    }
}

/// Original: `bestCenterForCG` (`tracksubs.cpp:437`).
///
/// `best` is not assigned when the centre is within 2 pixels of the box edge
/// (as in the source).  `ixBest`/`iyBest` are always set on the first
/// iteration, where `best == 0.`.
#[allow(clippy::too_many_arguments)]
pub fn best_center_for_cg(
    cp: &CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: f32,
    ypeak: f32,
    ixcen: &mut i32,
    iycen: &mut i32,
    best: &mut f32,
) {
    let mut ix_best: i32 = 0;
    let mut iy_best: i32 = 0;
    let mut sum4: f32;
    //
    *ixcen = nx_box / 2 + (xpeak as f64 + 0.5).floor() as i32;
    *iycen = ny_box / 2 + (ypeak as f64 + 0.5).floor() as i32;
    //
    // look around, find most extreme 4 points as center
    //
    if *ixcen >= 2 && *ixcen <= nx_box - 2 && *iycen >= 2 && *iycen <= ny_box - 2 {
        *best = 0.;
        for iy in *iycen - 1..=*iycen + 1 {
            for ix in *ixcen - 1..=*ixcen + 1 {
                sum4 = box_tmp[((iy - 1) * nx_box + ix - 1) as usize]
                    + box_tmp[((iy - 1) * nx_box + ix) as usize]
                    + box_tmp[(iy * nx_box + ix - 1) as usize]
                    + box_tmp[(iy * nx_box + ix) as usize];
                if cp.i_polarity as f32 * sum4 > cp.i_polarity as f32 * *best || *best == 0. {
                    ix_best = ix;
                    iy_best = iy;
                    *best = sum4;
                }
            }
        }
        *ixcen = ix_best;
        *iycen = iy_best;
    }
}

/// Original: `calcCG` (`tracksubs.cpp:474`).
///
/// Centroid of the bead around `xpeak, ypeak`, revising them; returns the
/// polarity-signed pixel sum in `wsum` and the edge SD in `edgeSD`.
#[allow(clippy::too_many_arguments)]
pub fn calc_cg(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: &mut f32,
    ypeak: &mut f32,
    wsum: &mut f32,
    edge_sd: &mut f32,
) {
    let mut ixcen: i32 = 0;
    let mut iycen: i32 = 0;
    let mut iy: i32;
    let mut ix: i32;
    let mut i: i32 = 0;
    let mut xsum: f32;
    let mut ysum: f32;
    let mut weight: f32;
    let mut edge: f32 = 0.;
    let mut best_sum: f32 = 0.;
    let mut pos_sum: f32;
    //
    best_center_for_cg(
        cp,
        box_tmp,
        nx_box,
        ny_box,
        *xpeak,
        *ypeak,
        &mut ixcen,
        &mut iycen,
        &mut best_sum,
    );
    xsum = 0.;
    ysum = 0.;
    *wsum = 0.;
    *edge_sd = 0.;
    //
    // find edge mean - require half the points to be present
    edge_for_cg(
        cp, box_tmp, nx_box, ny_box, ixcen, iycen, &mut edge, edge_sd, &mut i,
    );
    if i != 0 {
        return;
    }
    //
    // subtract edge and get weighted sum of pixel coordinates for POSITIVE
    // pixels
    //
    pos_sum = 0.;
    for i in 1..=cp.num_inside as usize {
        ix = ixcen + cp.idx_in[i - 1];
        iy = iycen + cp.idyin[i - 1];
        if ix >= 1 && ix <= nx_box && iy >= 1 && iy <= ny_box {
            weight = box_tmp[((iy - 1) * nx_box + ix - 1) as usize] - edge;
            if cp.i_polarity as f32 * weight > 0. {
                xsum += ix as f32 * weight;
                ysum += iy as f32 * weight;
                pos_sum += weight;
            }
            *wsum += weight;
        }
    }
    if pos_sum == 0. {
        return;
    }
    *xpeak = ((xsum / pos_sum) as f64 - 0.5 - (nx_box / 2) as f64) as f32;
    *ypeak = ((ysum / pos_sum) as f64 - 0.5 - (ny_box / 2) as f64) as f32;
    let w = *wsum * cp.i_polarity as f32;
    *wsum = if 0. > w as f64 { 0. } else { w };
}

/// Original: `wsumForSobelPeak` (`tracksubs.cpp:521`).
///
/// Sum of central pixels above the edge around `xpeak, ypeak`, without
/// adjusting the centre or computing a centroid.
#[allow(clippy::too_many_arguments)]
pub fn wsum_for_sobel_peak(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: f32,
    ypeak: f32,
    wsum: &mut f32,
    edge_sd: &mut f32,
) {
    let ixcen: i32;
    let iycen: i32;
    let mut iy: i32;
    let mut ix: i32;
    let mut i: i32 = 0;
    let mut edge: f32 = 0.;
    //
    ixcen = nx_box / 2 + (xpeak as f64 + 0.5).floor() as i32;
    iycen = ny_box / 2 + (ypeak as f64 + 0.5).floor() as i32;
    *wsum = 0.;
    *edge_sd = 0.;
    //
    // find edge mean - require half the points to be present
    edge_for_cg(
        cp, box_tmp, nx_box, ny_box, ixcen, iycen, &mut edge, edge_sd, &mut i,
    );
    if i != 0 {
        return;
    }
    //
    // subtract edge and get weight sum for ALL pixels
    //
    for i in 1..=cp.num_inside as usize {
        ix = ixcen + cp.idx_in[i - 1];
        iy = iycen + cp.idyin[i - 1];
        if ix >= 1 && ix <= nx_box && iy >= 1 && iy <= ny_box {
            *wsum += box_tmp[((iy - 1) * nx_box + ix - 1) as usize] - edge;
        }
    }
    let w = *wsum * cp.i_polarity as f32;
    *wsum = if 0. > w as f64 { 0. } else { w };
}

/// Original: `rescue` (`tracksubs.cpp:556`).
///
/// Searches progressively wider rings for a centroid peak above `relaxCrit`.
/// `sdBest` is uninitialised in the source when `getEdgeSD` is off, and read
/// only when it is on; it starts at 0 here.  `idx`, `idy` are `float` loop
/// variables starting at the integer quotient `-nxBox / 2`.
#[allow(clippy::too_many_arguments)]
pub fn rescue(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: &mut f32,
    ypeak: &mut f32,
    rad_max: f32,
    relax_crit: f32,
    step_size: f32,
    wsum: &mut f32,
    edge_sd: &mut f32,
) {
    let mut idx: f32;
    let mut idy: f32;
    let mut rad: f32;
    let rad_inc: f32;
    let mut rad_sq: f32;
    let mut rad_out_sq: f32;
    let mut w_best: f32;
    let mut dist_sq: f32;
    let mut xtmp: f32;
    let mut ytmp: f32;
    let mut wtmp: f32 = 0.;
    let mut sdtmp: f32 = 0.;
    let mut sd_best: f32 = 0.;
    *wsum = 0.;
    *edge_sd = 0.;
    if relax_crit <= 0. {
        return;
    }
    rad = 0.;
    rad_inc = 1.5;
    while rad <= rad_max && *wsum == 0. {
        rad_sq = rad * rad;
        let r = if rad + rad_inc < rad_max {
            rad + rad_inc
        } else {
            rad_max
        };
        rad_out_sq = r * r;
        w_best = 0.;
        idy = (-ny_box / 2) as f32;
        while idy <= (ny_box / 2) as f32 {
            idx = (-nx_box / 2) as f32;
            while idx <= (nx_box / 2) as f32 {
                dist_sq = idx * idx + idy * idy;
                if dist_sq > rad_sq && dist_sq <= rad_out_sq {
                    xtmp = idx;
                    ytmp = idy;
                    calc_cg(
                        cp, box_tmp, nx_box, ny_box, &mut xtmp, &mut ytmp, &mut wtmp, &mut sdtmp,
                    );
                    if wtmp > w_best {
                        w_best = wtmp;
                        *xpeak = xtmp;
                        *ypeak = ytmp;
                        if cp.get_edge_sd != 0 {
                            sd_best = sdtmp;
                        }
                    }
                }
                idx += step_size;
            }
            idy += step_size;
        }
        rad += rad_inc;
        if w_best >= relax_crit {
            *wsum = w_best;
            if cp.get_edge_sd != 0 {
                *edge_sd = sd_best;
            }
            //
            // Tried refining the position and the wsum and even testing on that new wsum, but
            // this was bad in many variations for a set where everything was in a cluster
        }
    }
}

/// Original: `checkSobelPeak` (`tracksubs.cpp:646`).
///
/// Makes sure `wsums[indPeak - 1]` has been computed.  `maxPeaks` is unused
/// in the source too.
#[allow(clippy::too_many_arguments)]
pub fn check_sobel_peak(
    cp: &mut CGPixels,
    ind_peak: i32,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    x_peaks: &[f32],
    y_peaks: &[f32],
    peaks: &[f32],
    wsums: &mut [f32],
    edge_sds: &mut [f32],
    max_peaks: i32,
) {
    let _ = max_peaks;
    let k = (ind_peak - 1) as usize;
    if wsums[k] >= 0. {
        return;
    }
    if (peaks[k] as f64) < -1.0e29 {
        wsums[k] = 0.;
        return;
    }
    wsum_for_sobel_peak(
        cp,
        box_tmp,
        nx_box,
        ny_box,
        x_peaks[k],
        y_peaks[k],
        &mut wsums[k],
        &mut edge_sds[k],
    );
}

/// Original: `calcElongation` (`tracksubs.cpp:669`).
///
/// A measure of elongation of the density at least 20% above the edge
/// intensity; `nxBox`, `nyBox` must be the basic box size, which the
/// `cp->elong*` arrays are dimensioned to.
///
/// Fixed in translation (2026-09-26, `BUGS.md`): `bestSum` is read
/// uninitialised in the source when `bestCenterForCG` leaves it unassigned (a
/// centre 1 pixel from, or on, the box edge) and the range test below still
/// passes.  Here such a centre gives no measurement (`elongation = -1`), as
/// the range test does for a centre outside the box.
#[allow(clippy::too_many_arguments)]
pub fn calc_elongation(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: f32,
    ypeak: f32,
    elongation: &mut f32,
) {
    let mut i: i32 = 0;
    let mut ixcen: i32 = 0;
    let mut iycen: i32 = 0;
    let mut num_pos: i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut ind_check: i32;
    let idelx: [i32; 4] = [-1, 1, 0, 0];
    let idely: [i32; 4] = [0, 0, -1, 1];
    let xmean: f32;
    let ymean: f32;
    let mut edge: f32 = 0.;
    let mut edge_sd: f32 = 0.;
    let thresh: f32;
    let thresh_frac: f32;
    let root: f32;
    let mut best_sum: f32 = 0.;
    let mut dxsum: f64;
    let mut dysum: f64;
    let mut dxsqsum: f64;
    let mut dysqsum: f64;
    let mut dxysum: f64;
    let edge_sd_save: i32;
    let edge_median_save: i32;
    //
    thresh_frac = 0.20;
    //
    // Smooth the data then find an adjusted center from the smoothed data
    let mut elong_smooth = std::mem::take(&mut cp.elong_smooth);
    'body: {
        apply_kernel_filter(
            box_tmp,
            &mut elong_smooth,
            nx_box,
            nx_box,
            ny_box,
            &cp.elong_kernel,
            cp.kern_dim_elong,
        );
        // NaN marks "not assigned by bestCenterForCG" (see the doc comment).
        best_sum = f32::NAN;
        best_center_for_cg(
            cp,
            &elong_smooth,
            nx_box,
            ny_box,
            xpeak,
            ypeak,
            &mut ixcen,
            &mut iycen,
            &mut best_sum,
        );
        *elongation = -1.;
        //
        // Get an edge median regardless of normal setting
        edge_sd_save = cp.get_edge_sd;
        edge_median_save = cp.edge_median;
        cp.get_edge_sd = 0;
        cp.edge_median = 1;
        edge_for_cg(
            cp,
            &elong_smooth,
            nx_box,
            ny_box,
            ixcen,
            iycen,
            &mut edge,
            &mut edge_sd,
            &mut i,
        );
        cp.get_edge_sd = edge_sd_save;
        cp.edge_median = edge_median_save;
        if i != 0
            || ixcen <= 0
            || ixcen > nx_box
            || iycen <= 0
            || iycen > ny_box
            || best_sum.is_nan()
        {
            break 'body;
        }
        //
        // Get threshold value and start a list of points to check with the center point
        thresh = (edge as f64 + thresh_frac as f64 * (best_sum as f64 / 4. - edge as f64)) as f32;
        num_pos = 1;
        cp.ix_elong[0] = ixcen;
        cp.iy_elong[0] = iycen;
        ind_check = 1;
        for i in 0..(nx_box * ny_box) as usize {
            cp.elong_mask[i] = 0;
        }
        cp.elong_mask[((iycen - 1) * nx_box + ixcen - 1) as usize] = 1;
        dxsum = 0.;
        dysum = 0.;
        //
        // Make a list of pixels above the threshold by checking the four neighbors of each
        // point on list, adding to list and setting a mask as each is found
        while ind_check <= num_pos {
            for i in 0..4 {
                let xn = cp.ix_elong[(ind_check - 1) as usize] + idelx[i];
                let yn = cp.iy_elong[(ind_check - 1) as usize] + idely[i];
                let mx_ = if nx_box < xn { nx_box } else { xn };
                ix = (if 1 > mx_ { 1 } else { mx_ }) - 1;
                let my_ = if ny_box < yn { ny_box } else { yn };
                iy = (if 1 > my_ { 1 } else { my_ }) - 1;
                if cp.elong_mask[(iy * nx_box + ix) as usize] == 0
                    && cp.i_polarity as f32 * (elong_smooth[(iy * nx_box + ix) as usize] - thresh)
                        > 0.
                {
                    cp.elong_mask[(iy * nx_box + ix) as usize] = 1;
                    cp.ix_elong[num_pos as usize] = ix + 1;
                    cp.iy_elong[num_pos as usize] = iy + 1;
                    num_pos += 1;
                    dxsum += (ix + 1) as f64;
                    dysum += (iy + 1) as f64;
                }
            }
            ind_check += 1;
        }
        if num_pos < 4 {
            break 'body;
        }

        // Get the means and moments and apply the equation for elongation
        xmean = (dxsum / num_pos as f64) as f32;
        ymean = (dysum / num_pos as f64) as f32;
        dxsqsum = 0.;
        dysqsum = 0.;
        dxysum = 0.;
        for i in 0..num_pos as usize {
            let ddx = cp.ix_elong[i] as f32 - xmean;
            let ddy = cp.iy_elong[i] as f32 - ymean;
            dxsqsum += (ddx * ddx) as f64;
            dysqsum += (ddy * ddy) as f64;
            dxysum += (ddx * ddy) as f64;
        }

        root = (4. * dxysum * dxysum + (dxsqsum - dysqsum) * (dxsqsum - dysqsum)).sqrt() as f32;
        *elongation =
            ((dxsqsum + dysqsum + root as f64) / (dxsqsum + dysqsum - root as f64)) as f32;
        // This is supposedly the axis angle but it seemed flaky.  Trying to find a long axis
        // by looking at points above threshold was problematic
        // angle = atan2(2. * dxysum, dxsqsum - dysqsum)
    }
    cp.elong_smooth = elong_smooth;
}

/// Original: `calcOuterMAD` (`tracksubs.cpp:752`).
///
/// Mean and median absolute deviation in an outer region of the box ("did
/// not turn out to be useful", but `beadtrack.cpp:2079,3041` still call it).
/// `background` is written only when at least 11 points are present.  The
/// asymmetric range test (`nxBox - 2` vs `nyBox - 3`) is the source's.
#[allow(clippy::too_many_arguments)]
pub fn calc_outer_mad(
    cp: &mut CGPixels,
    box_tmp: &[f32],
    nx_box: i32,
    ny_box: i32,
    xpeak: f32,
    ypeak: f32,
    outer_mad: &mut f32,
    background: &mut f32,
) {
    let mut rmedian: f32 = 0.;
    let ixcen: i32;
    let iycen: i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut nsum: i32;

    apply_kernel_filter(
        box_tmp,
        &mut cp.elong_smooth,
        nx_box,
        nx_box,
        ny_box,
        &cp.outer_kernel,
        cp.kern_dim_outer,
    );
    nsum = 0;
    ixcen = (xpeak as f64 + 0.5).floor() as i32;
    iycen = (ypeak as f64 + 0.5).floor() as i32;
    for i in 1..=cp.num_outer as usize {
        ix = ixcen + cp.idx_outer[i - 1];
        iy = iycen + cp.idy_outer[i - 1];
        if ix >= 3 && ix <= nx_box - 2 && iy >= 3 && iy <= ny_box - 3 {
            nsum += 1;
            cp.outer_pixels[(nsum - 1) as usize] = box_tmp[((iy - 1) * nx_box + ix - 1) as usize];
        }
    }
    *outer_mad = 0.;
    if nsum < 11 {
        return;
    }
    //
    rs_fast_median_in_place(&mut cp.outer_pixels, nsum, background);

    nsum = 0;
    for i in 1..=cp.num_outer as usize {
        ix = ixcen + cp.idx_outer[i - 1];
        iy = iycen + cp.idy_outer[i - 1];
        if ix >= 3 && ix <= nx_box - 2 && iy >= 3 && iy <= ny_box - 3 {
            nsum += 1;
            cp.outer_pixels[(nsum - 1) as usize] =
                cp.elong_smooth[((iy - 1) * nx_box + ix - 1) as usize];
        }
    }
    rs_fast_median_in_place(&mut cp.outer_pixels, nsum, &mut rmedian);
    for i in 0..nsum as usize {
        let d = cp.outer_pixels[i] - rmedian;
        cp.outer_pixels[i] = if d >= 0. { d } else { -d };
    }
    rs_fast_median_in_place(&mut cp.outer_pixels, nsum, outer_mad);
}

/// Original: `add_point` (`tracksubs.cpp:796`).
///
/// Adds a point to the output model after `ipNear` (or at the end when it is
/// 0), updating `ipNear` to the new point's position when non-zero.
pub fn add_point(
    fm: &mut FortModel,
    iobj: i32,
    ip_near: &mut i32,
    xpos: f32,
    ypos: f32,
    iz_next: i32,
) {
    let failed: bool;
    let mut ip_add: i32;
    let mut ibase: i32;
    //
    // OOPS the lib is not compiled this way
    failed = fort_object_mover(fm, iobj);
    if failed {
        exit_error(b"Insufficient object space");
    }
    ibase = fm.ibase_obj[(iobj - 1) as usize];
    fm.n_point += 1;
    let np = (fm.n_point - 1) as usize;
    fm.p_coord[np][0] = xpos;
    fm.p_coord[np][1] = ypos;
    fm.p_coord[np][2] = iz_next as f32;
    fm.pt_label[np] = 0;
    fm.npt_in_obj[(iobj - 1) as usize] += 1;
    ip_add = *ip_near;
    if ip_add == 0 {
        ip_add = fm.npt_in_obj[(iobj - 1) as usize];
        if ip_add == 1 {
            fm.ibase_obj[(iobj - 1) as usize] = fm.ibase_free;
            ibase = fm.ibase_free;
        }
    } else if ((fm.p_coord[(fm.object[(ibase + ip_add - 1) as usize] - 1) as usize][2] as f64 + 0.5)
        .floor() as i32)
        < iz_next
    {
        ip_add += 1;
    }
    //TODO: do j = ibase + npt_in_obj(iobj), ibase + ipAdd + 1, -1
    let mut j = ibase + fm.npt_in_obj[(iobj - 1) as usize];
    while j >= ibase + ip_add + 1 {
        fm.object[(j - 1) as usize] = fm.object[(j - 2) as usize];
        j -= 1;
    }
    fm.object[(ibase + ip_add - 1) as usize] = fm.n_point;
    fm.ibase_free += 1;
    fm.ntot_in_obj += 1;
    if *ip_near != 0 {
        *ip_near = ip_add;
    }
}

/// Original: `setsiz_sam_cel` (`tracksubs.cpp:839`).
///
/// Sets the header size, sampling and cell size of unit `iunit` for
/// dimensions `nx, ny, nz`, preserving the pixel size of unit 1.
pub fn setsiz_sam_cel(iunit: i32, nx: i32, ny: i32, nz: i32) {
    let nxyz: [i32; 3];
    let nxyzst: [i32; 3] = [0, 0, 0];
    let mut cell: [f32; 6] = [0., 0., 0., 90., 90., 90.];
    let delta: [f32; 3];
    //
    delta = iiu_ret_delta(1);
    nxyz = [nx, ny, nz];
    cell[0] = nx as f32 * delta[0];
    cell[1] = ny as f32 * delta[1];
    cell[2] = nz as f32 * delta[2];
    iiu_alt_size(iunit, &nxyz, &nxyzst);
    iiu_alt_sample(iunit, &nxyz);
    iiu_alt_cell(iunit, &cell);
}

/// Original: `splitPack` (`tracksubs.cpp:859`).
///
/// Splits `array` (dimensioned `nxDim` by `ny`, data `nx` by `ny`) into the
/// four corners of `brray`.
pub fn split_pack(array: &[f32], nx_dim: i32, nx: i32, ny: i32, brray: &mut [f32]) {
    let mut ixnew: i32;
    let mut iy_new: i32;
    for iy in 1..=ny {
        for ix in 1..=nx {
            ixnew = ((ix + nx / 2 - 1) % nx) + 1;
            iy_new = ((iy + ny / 2 - 1) % ny) + 1;
            brray[((iy_new - 1) * nx + ixnew - 1) as usize] =
                array[((iy - 1) * nx_dim + ix - 1) as usize];
        }
    }
}

/// Original: `countMissing` (`tracksubs.cpp:878`).
///
/// Counts the views missing from object `iobj`; `missing` is indexed from 1
/// to `nviewAll`, as in the source.
#[allow(clippy::too_many_arguments)]
pub fn count_missing(
    fm: &FortModel,
    iobj: i32,
    nview_all: i32,
    iz_exclude: &[i32],
    num_exlude: i32,
    missing: &mut [bool],
    listz: &mut [i32],
    num_list_z: &mut i32,
) {
    let num_in_obj: i32;
    let ibase: i32;
    let mut iz: i32;
    //
    num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
    *num_list_z = 0;
    if num_in_obj == 0 {
        return;
    }
    ibase = fm.ibase_obj[(iobj - 1) as usize];
    for i in 1..=nview_all as usize {
        missing[i] = true;
    }
    for i in 1..=num_exlude as usize {
        // Source-level UB (`tracksubs.cpp:892`): `SkipViews` is never checked
        // against the stack size, so a skipped view past `maxView` writes past
        // the `B3DMALLOC(bool, mx->maxView)` array (`beadtrack.cpp:335`) --
        // native survives by landing in malloc's slack.  Entries past
        // `nviewAll` are never read (the loops below stop there), so dropping
        // the out-of-bounds write changes nothing observable; indexing it
        // panicked (exit 101 where native exits 0).
        if let Some(slot) = missing.get_mut(iz_exclude[i - 1] as usize) {
            *slot = false;
        }
    }
    for ip in 1..=num_in_obj {
        iz = (fm.p_coord[(fm.object[(ibase + ip - 1) as usize] - 1) as usize][2] as f64 + 0.5)
            .floor() as i32
            + 1;
        missing[iz as usize] = false;
    }
    *num_list_z = 0;
    for i in 1..=nview_all {
        if missing[i as usize] {
            *num_list_z += 1;
            listz[(*num_list_z - 1) as usize] = i;
        }
    }
}

/// Original: `findxf_wo_outliers` (`tracksubs.cpp:928`).
///
/// Calls `findTransform` for a 2D transformation and eliminates outlying
/// position pairs from the solution.  `xr` has `msizeXR` columns per point;
/// columns 14-18 (1-based) save the data, 13 holds deviations and 19 the
/// original index.
#[allow(clippy::too_many_arguments)]
pub fn findxf_wo_outliers(
    xr: &mut [f32],
    msize_xr: i32,
    num_data: i32,
    xcen: f32,
    ycen: f32,
    if_trans: i32,
    if_ro_trans: i32,
    max_drop: i32,
    crit_prob: f32,
    crit_abs: f32,
    elim_min: f32,
    idrop: &mut [i32],
    num_drop: &mut i32,
    f: &mut [f32],
    dev_avg: &mut f32,
    dev_sd: &mut f32,
    dev_max: &mut f32,
    ipnt_max: &mut i32,
) {
    let mut last_drop: i32;
    let mut itmp: i32;
    let mut nkeep: i32;
    let isave_base: i32;
    let icol_dev: i32;
    let index_col: i32;
    let mut num_tmp: i32;
    let prob_per_pt: f32;
    let abs_per_pt: f32;
    let mut sigma_from_avg: f32;
    let mut sigma_from_sd: f32;
    let mut sigma: f32;
    let mut z: f32;
    let mut prob: f32;
    let ms = msize_xr as usize;
    //gprob(z) = 1. - 0.5 * erfcc(z / 1.414214);
    isave_base = 13;
    icol_dev = 13;
    index_col = 19;
    //
    // get probability per single point from the overall criterion prob
    //
    prob_per_pt = (1. - crit_prob as f64).powf(1. / num_data as f64) as f32;
    abs_per_pt = (1. - crit_abs as f64).powf(1. / num_data as f64) as f32;
    //
    // copy the data into far columns
    //
    for i in 0..num_data as usize {
        for j in 0..5usize {
            xr[i * ms + j + isave_base as usize] = xr[i * ms + j];
        }
    }
    find_transform(
        xr,
        msize_xr,
        4,
        num_data,
        xcen,
        ycen,
        if_trans,
        if_ro_trans,
        1,
        f,
        dev_avg,
        dev_sd,
        dev_max,
        ipnt_max,
    );
    *num_drop = 0;
    if max_drop == 0 || *dev_max < elim_min {
        return;
    }
    //
    // Sort the residuals and keep index back to initial values
    //
    last_drop = 0;
    for i in 0..num_data as usize {
        idrop[i] = i as i32 + 1;
    }
    for i in 0..(num_data - 1).max(0) as usize {
        for j in i + 1..num_data as usize {
            if xr[(idrop[i] - 1) as usize * ms + (icol_dev - 1) as usize]
                > xr[(idrop[j] - 1) as usize * ms + (icol_dev - 1) as usize]
            {
                itmp = idrop[i];
                idrop[i] = idrop[j];
                idrop[j] = itmp;
            }
        }
    }
    //
    // load the data in this order, save index in xr
    //
    for i in 0..num_data as usize {
        for j in 0..5usize {
            xr[i * ms + j] = xr[(idrop[i] - 1) as usize * ms + j + isave_base as usize];
        }
        xr[i * ms + (index_col - 1) as usize] = idrop[i] as f32;
    }
    //
    // Drop successively more points: get mean and S.d. of the remaining
    // points and check how many of the points pass the criterion
    // for outliers.
    //
    for jdrop in 1..=max_drop + 1 {
        num_tmp = num_data - jdrop;
        find_transform(
            xr,
            msize_xr,
            4,
            num_tmp,
            xcen,
            ycen,
            if_trans,
            if_ro_trans,
            1,
            f,
            dev_avg,
            dev_sd,
            dev_max,
            ipnt_max,
        );
        //
        // get deviations for points out of fit
        //
        for i in (num_data - jdrop) as usize..num_data as usize {
            let (xx, yy) = xf_apply(f, 0., 0., xr[i * ms], xr[i * ms + 1], 2);
            let ex = xx - xr[i * ms];
            let ey = yy - xr[i * ms + 1];
            xr[i * ms + (icol_dev - 1) as usize] = (ex * ex + ey * ey).sqrt();
        }
        //
        // estimate the sigma for the error distribution as the maximum of
        // the values implied by the mean and the SD of the deviations
        //
        sigma_from_avg = (*dev_avg as f64 / (8. / 3.14159f64).sqrt()) as f32;
        sigma_from_sd = (*dev_sd as f64 / (3. - 8. / 3.14159f64).sqrt()) as f32;
        sigma = if sigma_from_avg > sigma_from_sd {
            sigma_from_avg
        } else {
            sigma_from_sd
        };
        sigma = if sigma as f64 > 1.0e-10 {
            sigma
        } else {
            1.0e-10f64 as f32
        };

        nkeep = 0;
        for j in num_data - jdrop..num_data {
            z = xr[j as usize * ms + (icol_dev - 1) as usize] / sigma;
            // Sure looks like 2 * (gprob - 0.5) is errFunc!
            //gprob = 1. - 0.5 * (1. - errFunc(z / 1.414214));
            prob = (err_func(z as f64 / 1.414214)
                - (2. / 3.14159f64).sqrt() * z as f64 * ((-z * z) as f64 / 2.).exp())
                as f32;
            if prob < prob_per_pt {
                nkeep += 1;
            }
            if prob >= abs_per_pt {
                let m = if *num_drop > num_data - j {
                    *num_drop
                } else {
                    num_data - j
                };
                *num_drop = if max_drop < m { max_drop } else { m };
            }
        }
        //
        // If all points are outliers, this is a candidate for a set to drop
        // When only the first point is kept, and all the rest of the points
        // were outliers on the previous round, then this is a safe place to
        // draw the line between good data and outliers.  In this case, set
        // numDrop; and at end take the biggest numDrop that fits these criteria
        //
        if nkeep == 0 {
            last_drop = jdrop;
        }
        if nkeep == 1 && last_drop == jdrop - 1 && last_drop > 0 {
            *num_drop = last_drop;
        }
        // print *,'drop', jdrop, ', keep', nkeep, ', lastdrop', lastDrop, &
        // ',  ndrop =', numDrop
    }
    //
    // when finish loop, need to redo with right amount of data and restore
    // data
    //
    for i in 0..*num_drop {
        idrop[i as usize] =
            (xr[(num_data + i - *num_drop) as usize * ms + (index_col - 1) as usize] as f64 + 0.5)
                .floor() as i32;
    }
    num_tmp = num_data - *num_drop;
    find_transform(
        xr,
        msize_xr,
        4,
        num_tmp,
        xcen,
        ycen,
        if_trans,
        if_ro_trans,
        1,
        f,
        dev_avg,
        dev_sd,
        dev_max,
        ipnt_max,
    );
    *ipnt_max =
        (xr[(*ipnt_max - 1) as usize * ms + (index_col - 1) as usize] as f64 + 0.5).floor() as i32;
    for i in 0..num_data as usize {
        for j in 0..5usize {
            xr[i * ms + j] = xr[i * ms + j + isave_base as usize];
        }
    }
}

/// Original: `adjustXYZinAreas` (`tracksubs.cpp:1060`).
///
/// Shifts the XYZ values of the local areas so that points shared between
/// two areas have the same mean position, then averages each bead into
/// `xyzObj`.  `ninObjList` and `indObjList` must hold `nobjLists + 1`
/// entries: the source writes the entry past the last area.
///
/// # Out-of-bounds indexing in the source, fixed in translation
///
/// The three `for (i = 1; i <= 3; i++)` loops over `dxyzAvg`, `sumDxyz` and
/// `dxyzLast` (`tracksubs.cpp:1228-1242`) index 1..3 of 3-element stack
/// arrays: element 0 is never accumulated, tested or reset, and element 3
/// is past the end.  Fixed in translation (2026-09-26, `BUGS.md`): the loops
/// run over 0..3, so X is averaged and tested like Y and Z and the printed
/// averages show X's value instead of 0.000.
/// The three `B3DMALLOC`ed arrays are never freed in the source.
#[allow(clippy::too_many_arguments)]
pub fn adjust_xyz_in_areas(
    iobj_lists: &[i32],
    list_size: i32,
    ind_obj_list: &mut [i32],
    nin_obj_list: &mut [i32],
    xyz_all: &mut [f32],
    nobj_lists: i32,
    xyz_obj: &mut [f32],
    num_obj: i32,
) {
    let mut num_prev_areas: Vec<i32>;
    let mut ind_start_in_prev_list: Vec<i32>;
    let mut list_prev_inds: Vec<i32> = Vec::new();
    let mut dxyz = [0f32; 3];
    let mut sum_dxyz = [0f32; 3];
    let mut dxyz_last = [0f32; 3];
    let mut dxyz_avg = [0f32; 3];
    let mut dxyz_max = [0f32; 3];
    let mut avg_xyz = [0f32; 3];
    let mut ind_free: i32;
    let mut iobj: i32;
    let mut ind_obj: i32;
    let mut jobj: i32;
    let max_iter: i32;
    let mut num_in_err: i32;
    let mut num_in_sum: i32;
    let mut ind: i32;
    let num_avg_for_test: i32;
    let interval_for_test: i32;
    let ibase_orig_obj: i32;
    let mut jnd_obj: i32;
    let crit_max_move: f32;
    let crit_move_diff: f32;
    let mut error: f64;
    // `xyzAll[k * 3 - 3 + j]`, component `j` of 1-based entry `k`.
    macro_rules! xa {
        ($k:expr, $j:expr) => {
            xyz_all[(($k) * 3 - 3 + ($j)) as usize]
        };
    }

    num_prev_areas = vec![0; (list_size + num_obj) as usize];
    ind_start_in_prev_list = vec![0; (list_size + num_obj) as usize];
    max_iter = 1000;
    crit_max_move = 0.01;
    crit_move_diff = 0.001;
    interval_for_test = 20;
    num_avg_for_test = 10;

    // Go through points twice, first time just counting how big the list array needs to be
    // then allocate it, then second time, fill the list of indices
    ibase_orig_obj =
        ind_obj_list[(nobj_lists - 1) as usize] + nin_obj_list[(nobj_lists - 1) as usize];
    nin_obj_list[nobj_lists as usize] = num_obj;
    ind_obj_list[nobj_lists as usize] = ibase_orig_obj;
    for v in num_prev_areas.iter_mut() {
        *v = 0;
    }
    for lp in 1..=2 {
        ind_free = 1;
        for iseq in 1..=nobj_lists + 1 {
            for ind_list in 1..=nin_obj_list[(iseq - 1) as usize] {
                ind_obj = ind_obj_list[(iseq - 1) as usize] + ind_list - 1;
                ind_start_in_prev_list[(ind_obj - 1) as usize] = ind_free;
                if iseq <= nobj_lists {
                    iobj = iobj_lists[(ind_obj - 1) as usize];
                    //
                    // Skip all points with no solved xyz
                    if xa!(ind_obj, 0) == 0. && xa!(ind_obj, 1) == 0. && xa!(ind_obj, 2) == 0. {
                        continue;
                    }
                } else {
                    iobj = ind_list;
                }
                for jseq in 1..=iseq - 1 {
                    for jnd_list in 1..=nin_obj_list[(jseq - 1) as usize] {
                        jnd_obj = ind_obj_list[(jseq - 1) as usize] + jnd_list - 1;
                        jobj = iobj_lists[(jnd_obj - 1) as usize];
                        if jobj == iobj
                            && (xa!(jnd_obj, 0) != 0.
                                || xa!(jnd_obj, 1) != 0.
                                || xa!(jnd_obj, 2) != 0.)
                        {
                            if lp > 1 {
                                list_prev_inds[(ind_free - 1) as usize] = jnd_obj;
                                num_prev_areas[(ind_obj - 1) as usize] += 1;
                            }
                            ind_free += 1;
                        }
                    }
                }
                if lp == 2 {
                    /*printf("area %d  obj %d %d %d", iseq, iobj,indStartInPrevList[indObj - 1],
                    numPrevAreas[indObj - 1]);
                    for (int it = 0; it < numPrevAreas[indObj - 1]; it++)
                      printf("  %d", listPrevInds[it + indStartInPrevList[indObj - 1] - 1]);
                      printf("\n");*/
                }
            }
        }
        if lp == 1 {
            list_prev_inds = vec![0; ind_free as usize];
        }
    }
    //
    // Iterate until changes become small.  The termination logic is adopted from
    // libcfshr/find_piece_shifts.c
    for i in 0..3 {
        dxyz_last[i] = 0.;
        dxyz_avg[i] = 0.;
    }
    for iter in 1..=max_iter {
        num_in_err = 0;
        for i in 0..3 {
            sum_dxyz[i] = 0.;
            dxyz_max[i] = 0.;
        }
        for iseq in 2..=nobj_lists {
            // For each area, add up all shifts relative to same points in previous areas
            for i in 0..3 {
                dxyz[i] = 0.;
            }
            num_in_sum = 0;
            for ind_list in 1..=nin_obj_list[(iseq - 1) as usize] {
                ind_obj = ind_obj_list[(iseq - 1) as usize] + ind_list - 1;
                for i in 1..=num_prev_areas[(ind_obj - 1) as usize] {
                    ind = list_prev_inds
                        [(ind_start_in_prev_list[(ind_obj - 1) as usize] + i - 1 - 1) as usize];
                    dxyz[0] += xa!(ind_obj, 0) - xa!(ind, 0);
                    dxyz[1] += xa!(ind_obj, 1) - xa!(ind, 1);
                    dxyz[2] += xa!(ind_obj, 2) - xa!(ind, 2);
                    //dxyz(1:3) += (xyzAll(1:3, indObj) - xyzAll(1:3, ind));
                }
                num_in_sum += num_prev_areas[(ind_obj - 1) as usize];
            }
            let _ = ImodFile::Stdout.flush();
            // Get the mean amount of shift needed, and accumulate the mean and maximum
            // absolute value of the shift for each dimension
            //dxyz /= B3DMAX(1, numInSum);
            //sumDxyz += B3DABS(dxyz);
            for i in 1..=3usize {
                dxyz[i - 1] /= (if 1 > num_in_sum { 1 } else { num_in_sum }) as f32;
                let a = if dxyz[i - 1] >= 0. {
                    dxyz[i - 1]
                } else {
                    -dxyz[i - 1]
                };
                sum_dxyz[i - 1] += a;
                dxyz_max[i - 1] = if dxyz_max[i - 1] > a {
                    dxyz_max[i - 1]
                } else {
                    a
                };
            }

            // Subtract the shift from each xyz in the current area
            for ind_list in 1..=nin_obj_list[(iseq - 1) as usize] {
                ind_obj = ind_obj_list[(iseq - 1) as usize] + ind_list - 1;
                if xa!(ind_obj, 0) != 0. || xa!(ind_obj, 1) != 0. || xa!(ind_obj, 2) != 0. {
                    for i in 0..3 {
                        xyz_all[((ind_obj - 1) * 3 + i) as usize] -= dxyz[i as usize];
                    }
                }
            }
        }

        // Compute error as mean difference from the average position
        error = 0.;
        for iobj in 1..=num_obj {
            for i in 0..3 {
                avg_xyz[i] = 0.;
            }
            let k = (ibase_orig_obj + iobj - 1 - 1) as usize;
            num_in_sum = num_prev_areas[k];
            for i in 1..=num_in_sum {
                ind = list_prev_inds[(ind_start_in_prev_list[k] + i - 1 - 1) as usize];
                avg_xyz[0] += xa!(ind, 0) / num_in_sum as f32;
                avg_xyz[1] += xa!(ind, 1) / num_in_sum as f32;
                avg_xyz[2] += xa!(ind, 2) / num_in_sum as f32;
                //avgXyz(1:3) += xyzAll(1:3, ind) / numInSum;
            }

            for i in 1..=num_in_sum {
                ind = list_prev_inds[(ind_start_in_prev_list[k] + i - 1 - 1) as usize];
                for j in 1..=3 {
                    let d = avg_xyz[(j - 1) as usize] - xyz_all[(ind * 3 + j - 4) as usize];
                    error += (d * d) as f64;
                }
            }
            num_in_err += num_in_sum;
        }

        error = (error / (if 1 > num_in_err { 1 } else { num_in_err }) as f64).sqrt();
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "%4d%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n",
            &[
                CArg::Int(iter as i64),
                CArg::Dbl(error),
                CArg::Dbl(dxyz_max[0] as f64),
                CArg::Dbl(dxyz_max[1] as f64),
                CArg::Dbl(dxyz_max[2] as f64),
                CArg::Dbl(dxyz_avg[0] as f64),
                CArg::Dbl(dxyz_avg[1] as f64),
                CArg::Dbl(dxyz_avg[2] as f64),
            ],
        ));
        let _ = ImodFile::Stdout.flush();

        // If the maximum move is ever less than this criterion in all dimensions, finished
        if dxyz_max[0] < crit_max_move && dxyz_max[1] < crit_max_move && dxyz_max[2] < crit_max_move
        {
            break;
        }

        // Periodically accumulate the average move over several iterations
        if (iter % interval_for_test) >= interval_for_test - num_avg_for_test {
            for i in 0..3usize {
                dxyz_avg[i] += sum_dxyz[i] / (nobj_lists - 1) as f32;
            }
        }

        // Then test whether the average move has fallen by less than this criterion
        if (iter % interval_for_test) == interval_for_test - 1 {
            for i in 0..3usize {
                dxyz_avg[i] /= num_avg_for_test as f32;
            }
            if dxyz_last[0] - dxyz_avg[0] < crit_move_diff
                && dxyz_last[1] - dxyz_avg[1] < crit_move_diff
                && dxyz_last[2] - dxyz_avg[2] < crit_move_diff
            {
                break;
            }
            for i in 0..3usize {
                dxyz_last[i] = dxyz_avg[i];
                dxyz_avg[i] = 0.;
            }
        }
    }

    // Average each bead into the array of one per object
    for i in 0..(3 * num_obj) as usize {
        xyz_obj[i] = 0.;
    }
    for iobj in 1..=num_obj {
        let k = (ibase_orig_obj + iobj - 1 - 1) as usize;
        num_in_sum = num_prev_areas[k];
        for i in 1..=num_in_sum {
            ind = list_prev_inds[(ind_start_in_prev_list[k] + i - 1 - 1) as usize];
            xyz_obj[(iobj * 3 - 3) as usize] += xa!(ind, 0) / num_in_sum as f32;
            xyz_obj[(iobj * 3 - 2) as usize] += xa!(ind, 1) / num_in_sum as f32;
            xyz_obj[(iobj * 3 - 1) as usize] += xa!(ind, 2) / num_in_sum as f32;
            //xyzObj(1:3, iobj) += xyzAll(1:3, ind) / numInSum;
        }
    }
}

#[cfg(test)]
mod fixed_tests {
    use super::*;

    /// `BUGS.md` "`peakFind`": on an all-NaN array nothing exceeds -1.e30 and
    /// the source returns uninitialised indices; the defined peak is the
    /// origin, a zero shift.
    #[test]
    fn peak_find_on_nan_is_a_zero_shift() {
        let array = vec![f32::NAN; 10 * 8];
        let (mut x, mut y, mut peak) = (9.0_f32, 9.0_f32, 0.0_f32);
        peak_find(&array, 10, 8, &mut x, &mut y, &mut peak);
        assert_eq!((x, y), (0.0, 0.0));
    }

    /// `BUGS.md` "A NaN position": native converts a NaN position to
    /// `INT_MIN`, wraps the box indices and fails reading the image.  The
    /// defined behaviour finds no piece, so the bead is missing on that view.
    #[test]
    fn find_piece_rejects_a_nan_position() {
        let (ixl, iyl, izl) = ([0], [0], [0]);
        let (mut ix0, mut ix1, mut iy0, mut iy1, mut ipcz) = (0, 0, 0, 0, 7);
        let (mut taper, mut fill) = (true, true);
        find_piece(
            &ixl,
            &iyl,
            &izl,
            1,
            64,
            64,
            16,
            16,
            f32::NAN,
            20.0,
            0,
            &mut ix0,
            &mut ix1,
            &mut iy0,
            &mut iy1,
            &mut ipcz,
            0,
            &[],
            &mut taper,
            &mut fill,
        );
        assert_eq!(ipcz, -1);
        assert!(!taper && !fill);
        // An ordinary position still finds the piece.
        find_piece(
            &ixl,
            &iyl,
            &izl,
            1,
            64,
            64,
            16,
            16,
            30.0,
            20.0,
            0,
            &mut ix0,
            &mut ix1,
            &mut iy0,
            &mut iy1,
            &mut ipcz,
            0,
            &[],
            &mut taper,
            &mut fill,
        );
        assert_eq!((ipcz, ix0, ix1, iy0, iy1), (0, 22, 37, 12, 27));
    }

    /// `BUGS.md` "A point at a Z outside the transform list": native reads
    /// `prexf[-2]`; the defined shift is 0 and no piece is found at Z -1.
    #[test]
    fn find_piece_at_negative_z_with_transforms_finds_nothing() {
        let (ixl, iyl, izl) = ([0, 0], [0, 0], [0, 1]);
        let (mut ix0, mut ix1, mut iy0, mut iy1, mut ipcz) = (0, 0, 0, 0, 7);
        let (mut taper, mut fill) = (true, true);
        let prexf = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        find_piece(
            &ixl, &iyl, &izl, 2, 64, 64, 16, 16, 30.0, 20.0, -1, &mut ix0, &mut ix1, &mut iy0,
            &mut iy1, &mut ipcz, 1, &prexf, &mut taper, &mut fill,
        );
        assert_eq!(ipcz, -1);
        assert!(!fill);
    }
}
