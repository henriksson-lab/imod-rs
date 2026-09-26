//! Translation of `IMOD/flib/blend/solvescaling.f90`: `solveScaling` and its
//! two contained procedures, `getAverageDiffs` and `findBestScalings`.
//!
//! All three `use blendvars` (the contained ones by host association), so each
//! takes `bv: &mut BlendVars` first (see the design note in
//! [`super::blendvars`]).  The contained procedures reach the host's locals by
//! host association; here the host variables each one reads or writes are
//! passed explicitly, as in `flib/model/xfmodel.rs`.  The host's loop indices
//! (`ixFrame`, `iyFrame`, `ixy`, `ipc`, `jedge`, `ind`, `ix`, `iy`, `indVal`)
//! and scratch reals (`xx`, `yy`, `sclA`, `sclB`) that the contained procedures
//! also assign are always reassigned by the host before it reads them again,
//! so they are locals of each routine here.
//!
//! `dxGridMean(limEdge, 2)` and `dyGridMean(limEdge, 2)` are the caller's
//! arrays, flat column-major with leading dimension `bv.lim_edge`.
//! `critForLogs /0.2/` has SAVE but is never assigned, so it is a constant.
//! `MIN`/`MAX` of reals are written in source order (`bsubs.rs` module note).

use super::blendvars::BlendVars;
use super::bsubs::find_best_shifts;
use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use std::io::Write;

/// `real*4 critForLogs /0.2/` (`solvescaling.f90:16`).
const CRIT_FOR_LOGS: f32 = 0.2;

/// Original: `subroutine solveScaling(dxgridmean, dygridmean, izSect)`
/// (`solvescaling.f90:8`).
///
/// Analyzes edge intensity differences and solves for individual piece
/// scalings and possibly a gradient for section `izSect`.  `dxgridmean` is
/// loaded with differences for calling `find_best_shifts`, and `dygridmean`
/// is supplied just because it is needed in that call.
pub fn solve_scaling(
    bv: &mut BlendVars,
    dxgridmean: &mut [f32],
    dygridmean: &mut [f32],
    iz_sect: i32,
) {
    let mut grad_new = [0.0f32; 2];
    let mut hxf = [0.0f32; 10];
    let (mut tmin, mut tmax, mut tmean) = (0.0f32, 0.0f32, 0.0f32);
    let (mut bavg, mut bmax, mut aavg, mut amax) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut diff_min, mut diff_max, mut ratio_min) = (0.0f32, 0.0f32, 0.0f32);
    let mut num_sum = 0i32;
    let mut use_logs = false;
    let mp = |ix: i32, iy: i32, ext: &[usize; 2]| (ix - 1) as usize + ext[0] * (iy - 1) as usize;

    // Get base for the relative ratios: find min of all edges and see if it is less than
    // entered or default base
    let mut zero_base = bv.den_zero_base;
    let mut all_min = 1.0e30f32;
    let mut all_max = -all_min;
    for iy_frame in 1..=bv.ny_pieces {
        for ix_frame in 1..=bv.nx_pieces {
            let ipc = bv.map_piece[mp(ix_frame, iy_frame, &bv.map_piece_ext)];
            let k = mp(ix_frame, iy_frame, &bv.piece_scaling_ext);
            bv.piece_scaling[k] = 1.;
            if ipc > 0 {
                for ixy in 1..=2 {
                    let jedge = bv.iedge_upper[mp(ipc, ixy, &bv.iedge_upper_ext)];
                    if jedge > 0 && bv.if_skip_edge[mp(jedge, ixy, &bv.if_skip_edge_ext)] == 0 {
                        let ind = (ixy - 1) * bv.ix_dim_den_buf + jedge
                            - bv.iedge_cur_base[(ixy - 1) as usize];
                        let iu = (ind - 1) as usize;
                        let (nx, ny) = (bv.nx_den_buf[iu], bv.ny_den_buf[iu]);
                        let off = bv.den_abuf_ext[0] * iu;
                        array_min_max_mean_fortran(
                            &bv.den_abuf[off..],
                            &nx,
                            &ny,
                            &1,
                            &nx,
                            &1,
                            &ny,
                            &mut tmin,
                            &mut tmax,
                            &mut tmean,
                        );
                        // `solvescaling.f90:37-38, 41-42`: the reference object
                        // merges the two updates into
                        // `allMin = minss(allMin, minss(tminA, tminB))` and
                        // `allMax = maxss(maxss(tmaxA, tmaxB), allMax)`.
                        let (tmin_a, tmax_a) = (tmin, tmax);
                        let off = bv.den_bbuf_ext[0] * iu;
                        array_min_max_mean_fortran(
                            &bv.den_bbuf[off..],
                            &nx,
                            &ny,
                            &1,
                            &nx,
                            &1,
                            &ny,
                            &mut tmin,
                            &mut tmax,
                            &mut tmean,
                        );
                        all_min = minss(all_min, minss(tmin_a, tmin));
                        all_max = maxss(maxss(tmax_a, tmax), all_max);
                    }
                }
            }
        }
    }

    let base_lim = all_min - 0.02 * (all_max - all_min);
    // `solvescaling.f90:50`: `minss zeroBase, allMin - ...`.
    zero_base = minss(zero_base, base_lim);

    let xcen = bv.nxyz_in[0] as f32 / 2.;
    let ycen = bv.nxyz_in[1] as f32 / 2.;
    let mut x_grad = 0.0f32;
    let mut y_grad = 0.0f32;
    let mut jwhich = 0i32;
    if bv.i_dens_from_edges > 1 {
        // Get first estimate of gradient
        get_average_diffs(
            bv,
            0,
            &mut grad_new,
            dxgridmean,
            dygridmean,
            xcen,
            ycen,
            x_grad,
            y_grad,
            zero_base,
            &mut diff_min,
            &mut diff_max,
            &mut ratio_min,
        );
        x_grad = grad_new[0];
        y_grad = grad_new[1];
        jwhich = 1;
    }

    // Get (new) differences for solving scalings
    get_average_diffs(
        bv,
        jwhich,
        &mut grad_new,
        dxgridmean,
        dygridmean,
        xcen,
        ycen,
        x_grad,
        y_grad,
        zero_base,
        &mut diff_min,
        &mut diff_max,
        &mut ratio_min,
    );

    // Solve linear equations and get result into scalings
    find_best_scalings(
        bv,
        dxgridmean,
        dygridmean,
        iz_sect,
        &mut hxf,
        &mut num_sum,
        &mut bavg,
        &mut bmax,
        &mut aavg,
        &mut amax,
        &mut use_logs,
        diff_min,
        diff_max,
        ratio_min,
    );

    // gfortran `Fw.d` editing: a value too wide for the field is `w` asterisks.
    let f_edit = |value: f32, w: usize, d: usize| -> String {
        let text = format!("{value:>w$.d$}");
        if text.len() > w { "*".repeat(w) } else { text }
    };

    // If getting gradients, iterate
    if bv.i_dens_from_edges > 1 {
        // Get incremental estimate of gradient with piece diffs taken into account
        get_average_diffs(
            bv,
            3,
            &mut grad_new,
            dxgridmean,
            dygridmean,
            xcen,
            ycen,
            x_grad,
            y_grad,
            zero_base,
            &mut diff_min,
            &mut diff_max,
            &mut ratio_min,
        );
        x_grad += grad_new[0];
        y_grad += grad_new[1];
        let _ = writeln!(
            ImodFile::Stdout,
            "Intensity gradient over full piece extent in X and Y (%):{}{}",
            f_edit(x_grad * bv.nxyz_in[0] as f32 * 100., 8, 4),
            f_edit(y_grad * bv.nxyz_in[1] as f32 * 100., 8, 4)
        );

        // Get edge diffs with new gradient and solve again
        get_average_diffs(
            bv,
            1,
            &mut grad_new,
            dxgridmean,
            dygridmean,
            xcen,
            ycen,
            x_grad,
            y_grad,
            zero_base,
            &mut diff_min,
            &mut diff_max,
            &mut ratio_min,
        );
        find_best_scalings(
            bv,
            dxgridmean,
            dygridmean,
            iz_sect,
            &mut hxf,
            &mut num_sum,
            &mut bavg,
            &mut bmax,
            &mut aavg,
            &mut amax,
            &mut use_logs,
            diff_min,
            diff_max,
            ratio_min,
        );
    }
    let _ = writeln!(
        ImodFile::Stdout,
        "Intensity min&max diffs before:{}{}  after scaling:{}{}",
        f_edit(bavg, 8, 4),
        f_edit(bmax, 8, 4),
        f_edit(aavg, 8, 4),
        f_edit(amax, 8, 4)
    );
    bv.x_grad_scaling = x_grad;
    bv.y_grad_scaling = y_grad;

    // Compute adjustments to the density differences for adjusting edge functions
    for iy_frame in 1..=bv.ny_pieces {
        for ix_frame in 1..=bv.nx_pieces {
            let ipc = bv.map_piece[mp(ix_frame, iy_frame, &bv.map_piece_ext)];
            if ipc > 0 {
                for ixy in 1..=2 {
                    let jedge = bv.iedge_upper[mp(ipc, ixy, &bv.iedge_upper_ext)];
                    if jedge > 0 && bv.if_skip_edge[mp(jedge, ixy, &bv.if_skip_edge_ext)] == 0 {
                        let ind = (ixy - 1) * bv.ix_dim_den_buf + jedge
                            - bv.iedge_cur_base[(ixy - 1) as usize];
                        let iu = (ind - 1) as usize;
                        for ix in 1..=bv.nx_den_buf[iu] {
                            for iy in 1..=bv.ny_den_buf[iu] {
                                let xx = (bv.ix_den_start[iu]
                                    + (ix - 1) * bv.interval_den[(ixy - 1) as usize])
                                    as f32
                                    - xcen;
                                let yy = (bv.iy_den_start[iu]
                                    + (iy - 1) * bv.interval_den[(2 - ixy) as usize])
                                    as f32
                                    - ycen;
                                let ind_val = (iy - 1) * bv.nx_den_buf[iu] + ix;
                                let ka = (ind_val - 1) as usize + bv.den_abuf_ext[0] * iu;
                                let kb = (ind_val - 1) as usize + bv.den_bbuf_ext[0] * iu;
                                let den_a = bv.den_abuf[ka];
                                let den_b = bv.den_bbuf[kb];
                                let scl_a = den_a
                                    * bv.piece_scaling
                                        [mp(ix_frame, iy_frame, &bv.piece_scaling_ext)]
                                    / (1. + bv.x_grad_scaling * xx + bv.y_grad_scaling * yy);
                                let scl_b = den_b
                                    * bv.piece_scaling[mp(
                                        ix_frame + 2 - ixy,
                                        iy_frame + ixy - 1,
                                        &bv.piece_scaling_ext,
                                    )]
                                    / (1.
                                        + bv.x_grad_scaling * (xx + bv.ix_den_offset[iu] as f32)
                                        + bv.y_grad_scaling * (yy + bv.iy_den_offset[iu] as f32));
                                let kd = (ind_val - 1) as usize + bv.delta_den_buf_ext[0] * iu;
                                bv.delta_den_buf[kd] = (scl_b - scl_a) - (den_b - den_a);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Original: `subroutine getAverageDiffs(iwhich, gradFromEdge)`, contained in
/// `solveScaling` (`solvescaling.f90:131`).
///
/// `iwhich` is the sum of 1 to correct for gradient, 2 to correct for piece
/// differences.  Host variables: reads `xcen`, `ycen`, `xGrad`, `yGrad`,
/// `zeroBase`; assigns `ratioMin`, `diffMin`, `diffMax` and the host dummies
/// `dxgridmean`, `dyGridMean`.
#[allow(clippy::too_many_arguments)]
pub fn get_average_diffs(
    bv: &mut BlendVars,
    iwhich: i32,
    grad_from_edge: &mut [f32; 2],
    dxgridmean: &mut [f32],
    dygridmean: &mut [f32],
    xcen: f32,
    ycen: f32,
    x_grad: f32,
    y_grad: f32,
    zero_base: f32,
    diff_min: &mut f32,
    diff_max: &mut f32,
    ratio_min: &mut f32,
) {
    let mut edge_sum = [0.0f32; 2];
    let mut nsum = [0i32; 2];
    let mut off_sum = [0.0f32; 2];
    let mp = |ix: i32, iy: i32, ext: &[usize; 2]| (ix - 1) as usize + ext[0] * (iy - 1) as usize;
    let lim_edge = bv.lim_edge;
    *ratio_min = 1.0e10;
    *diff_min = 1.0e30;
    *diff_max = -*diff_min;
    let do_grad = iwhich % 2 != 0;
    // `doPiece = iwhich / 2`: an integer assigned to a logical is nonzero-true.
    let do_piece = iwhich / 2 != 0;
    grad_from_edge[0] = 0.;
    grad_from_edge[1] = 0.;
    for iy_frame in 1..=bv.ny_pieces {
        for ix_frame in 1..=bv.nx_pieces {
            let ipc = bv.map_piece[mp(ix_frame, iy_frame, &bv.map_piece_ext)];
            if ipc > 0 {
                for ixy in 1..=2 {
                    let jedge = bv.iedge_upper[mp(ipc, ixy, &bv.iedge_upper_ext)];
                    if jedge > 0 && bv.if_skip_edge[mp(jedge, ixy, &bv.if_skip_edge_ext)] == 0 {
                        let ind = (ixy - 1) * bv.ix_dim_den_buf + jedge
                            - bv.iedge_cur_base[(ixy - 1) as usize];
                        let iu = (ind - 1) as usize;
                        let mut diff_sum = 0.0f32;
                        let mut ratio_sum = 0.0f32;
                        //
                        // Get mean relative difference in this edge
                        for ix in 1..=bv.nx_den_buf[iu] {
                            for iy in 1..=bv.ny_den_buf[iu] {
                                let ind_val = (iy - 1) * bv.nx_den_buf[iu] + ix;
                                let mut den_a = bv.den_abuf
                                    [(ind_val - 1) as usize + bv.den_abuf_ext[0] * iu]
                                    - zero_base;
                                let mut den_b = bv.den_bbuf
                                    [(ind_val - 1) as usize + bv.den_bbuf_ext[0] * iu]
                                    - zero_base;
                                //
                                // Adjust for piece and/or for existing gradient
                                if do_piece {
                                    den_a *= bv.piece_scaling
                                        [mp(ix_frame, iy_frame, &bv.piece_scaling_ext)];
                                    if ixy == 1 {
                                        den_b *= bv.piece_scaling
                                            [mp(ix_frame + 1, iy_frame, &bv.piece_scaling_ext)];
                                    } else {
                                        den_b *= bv.piece_scaling
                                            [mp(ix_frame, iy_frame + 1, &bv.piece_scaling_ext)];
                                    }
                                }
                                if do_grad {
                                    let xx = (bv.ix_den_start[iu]
                                        + (ix - 1) * bv.interval_den[(ixy - 1) as usize])
                                        as f32
                                        - xcen;
                                    let yy = (bv.iy_den_start[iu]
                                        + (iy - 1) * bv.interval_den[(2 - ixy) as usize])
                                        as f32
                                        - ycen;
                                    den_a /= 1. + x_grad * xx + y_grad * yy;
                                    den_b /= 1.
                                        + x_grad * (xx + bv.ix_den_offset[iu] as f32)
                                        + y_grad * (yy + bv.iy_den_offset[iu] as f32);
                                }
                                let diff = 2. * (den_b - den_a) / (den_a + den_b);
                                diff_sum += diff;
                                let ratio = den_b / den_a;
                                ratio_sum += ratio;
                            }
                        }
                        let npts = (bv.nx_den_buf[iu] * bv.ny_den_buf[iu]) as f32;
                        let diff = diff_sum / npts;
                        let ratio = ratio_sum / npts;
                        let ke = (jedge - 1) as usize + lim_edge as usize * (ixy - 1) as usize;
                        dxgridmean[ke] = diff;
                        if ratio > 1.0e-6 {
                            dygridmean[ke] = ratio.ln();
                        }
                        // `solvescaling.f90:194-196`: `minss diffMin, diff`,
                        // `maxss diffMax, diff`, `minss ratio, ratioMin`.
                        *diff_min = minss(*diff_min, diff);
                        *diff_max = maxss(*diff_max, diff);
                        *ratio_min = if ratio < *ratio_min {
                            ratio
                        } else {
                            *ratio_min
                        };
                        let xyu = (ixy - 1) as usize;
                        edge_sum[xyu] += diff;
                        nsum[xyu] += 1;
                        if ixy == 1 {
                            off_sum[xyu] += bv.ix_den_offset[iu] as f32;
                        } else {
                            off_sum[xyu] += bv.iy_den_offset[iu] as f32;
                        }
                    }
                }
            }
        }
    }
    if nsum[0] > 0 {
        grad_from_edge[0] = edge_sum[0] / off_sum[0];
    }
    if nsum[1] > 0 {
        grad_from_edge[1] = edge_sum[1] / off_sum[1];
    }
}

/// Original: `subroutine findBestScalings()`, contained in `solveScaling`
/// (`solvescaling.f90:222`).
///
/// Decides whether to use the logs, and copies them from `dyGridMean` if so,
/// finds the density solution, and computes piece scalings from it.  Host
/// variables: reads `ratioMin`, `diffMin`, `diffMax`, `izsect`; assigns
/// `useLogs`, `dxGridMean`, `hxf`, `numSum`, `bavg`, `bmax`, `aavg`, `amax`.
#[allow(clippy::too_many_arguments)]
pub fn find_best_scalings(
    bv: &mut BlendVars,
    dxgridmean: &mut [f32],
    dygridmean: &mut [f32],
    izsect: i32,
    hxf: &mut [f32; 10],
    num_sum: &mut i32,
    bavg: &mut f32,
    bmax: &mut f32,
    aavg: &mut f32,
    amax: &mut f32,
    use_logs: &mut bool,
    diff_min: f32,
    diff_max: f32,
    ratio_min: f32,
) {
    let (a, b) = (diff_min.abs(), diff_max.abs());
    *use_logs = ratio_min > 0.001 && (if a > b { a } else { b }) > CRIT_FOR_LOGS;
    if *use_logs {
        let n = 2 * bv.lim_edge as usize;
        dxgridmean[..n].copy_from_slice(&dygridmean[..n]);
    }
    let lim_edge = bv.lim_edge;
    find_best_shifts(
        bv,
        dxgridmean,
        dygridmean,
        lim_edge,
        2,
        izsect,
        &mut hxf[..],
        num_sum,
        bavg,
        bmax,
        aavg,
        amax,
        false,
    );

    // unpack the result
    for iy_frame in (1..=bv.ny_pieces).rev() {
        for ix_frame in 1..=bv.nx_pieces {
            let ipc = bv.map_piece
                [(ix_frame - 1) as usize + bv.map_piece_ext[0] * (iy_frame - 1) as usize];
            if ipc > 0 {
                let k = (ix_frame - 1) as usize + bv.piece_scaling_ext[0] * (iy_frame - 1) as usize;
                if *use_logs {
                    bv.piece_scaling[k] = bv.den_solution[(ipc - 1) as usize].exp();
                } else {
                    bv.piece_scaling[k] = 1. / (1. - bv.den_solution[(ipc - 1) as usize]);
                }
            }
        }
    }
}
