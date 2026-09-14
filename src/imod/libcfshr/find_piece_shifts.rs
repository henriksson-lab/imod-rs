//! Translation of `IMOD/libcfshr/find_piece_shifts.c`: finds piece shifts by
//! iteration.
//!
//! The source carves its scratch arrays out of the caller's `work` array with
//! reinterpreting casts — `int *ivarToList = (int *)work;` and an
//! `unsigned char *` view further along it (`find_piece_shifts.c:110-118`,
//! `:508-514`).  Rust cannot retype a slice's elements without `unsafe`, so
//! each of those windows is a local `Vec` of its own element type here, laid
//! out in the same order and with the same lengths.  Nothing but the first
//! `2 * nvar` floats of `work` is read back by a caller — those are the
//! per-edge mean weights the documentation describes, and they are written to
//! `work` exactly as the source writes them.
#![allow(dead_code)]

use super::robuststat::{rs_madn, rs_median, rs_trimmed_mean};

/// Original `findPieceShifts` (`find_piece_shifts.c:88`).
#[allow(clippy::too_many_arguments)]
pub fn find_piece_shifts(
    ivarpc: &[i32],
    nvar: i32,
    indvar: &[i32],
    ixpclist: &[i32],
    iypclist: &[i32],
    dxedge: &[f32],
    dyedge: &[f32],
    idir: i32,
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: i32,
    dxyvar: &mut [f32],
    var_step: i32,
    edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: i32,
    work: &mut [f32],
    mut fort: i32,
    leave_ind: i32,
    skip_crit: i32,
    robust_crit: f32,
    crit_max_move: f32,
    crit_move_diff: f32,
    max_iter: i32,
    num_avg_for_test: i32,
    interval_for_test: i32,
    num_iter: &mut i32,
    w_err_mean: &mut f32,
    w_err_max: &mut f32,
) -> i32 {
    let edge_med_crit = 6;
    let mut num_edge = [0i32; 2];
    let mut edge_tr_mean_x = [0.0f32; 2];
    let mut edge_tr_mean_y = [0.0f32; 2];
    let mut edge_dev_med = [0.0f32; 2];
    let mut edge_dev_madn = [0.0f32; 2];
    let mut erx = [0.0f32; 4];
    let mut ery = [0.0f32; 4];
    let mut errd = [0.0f32; 4];
    // The source leaves these six uninitialised; `findLowestThree` sets them
    // all before they are read on any path that has at least three neighbours.
    let (mut xlow, mut xsec, mut xthr) = (0.0f32, 0.0f32, 0.0f32);
    let (mut ylow, mut ysec, mut ythr) = (0.0f32, 0.0f32, 0.0f32);
    let (mut elow, mut esec, mut ethr) = (0.0f32, 0.0f32, 0.0f32);
    let mut mad: f32 = 0.;
    let ktune: f32 = 2.0f32 * 4.685f32 * 0.6745f32 * robust_crit;
    let med_thresh: f32 = 2.0f32 * robust_crit;

    /* Set pointers to temporary arrays from the work array */
    let mut ivar_to_list = vec![0i32; nvar as usize];
    let mut list_to_var = vec![0i32; nvar as usize];
    let mut neigh_ind = vec![0i32; (nvar + 1) as usize];
    let mut neigh_list = vec![0i32; (4 * nvar) as usize];
    let mut dxy_edge = vec![0.0f32; (8 * nvar) as usize];
    let mut neigh_wgt = vec![0.0f32; (4 * nvar) as usize];
    let mut edge_dir = vec![0u8; (4 * nvar) as usize];
    let mut placed = vec![0u8; (2 * nvar) as usize]; // placed needs 2 * nvar bytes

    /* Set the xy stride parameter */
    let xy_step: i32;
    if edge_step == 1 && pc_step == 1 && var_step == 1 {
        xy_step = 2;
    } else if edge_step > 1 && pc_step > 1 && var_step > 1 {
        xy_step = 1;
    } else {
        return 1;
    }
    if fort != 0 {
        fort = 1;
    }

    /* If doing robust, get median edge displacement and median deviation */
    if robust_crit > 0. {
        for ixy in 0..2usize {
            let mut nsum = 0i32;

            /* Find the X or Y edges and save them in array */
            for ivar in 0..nvar as usize {
                let ipc = ivarpc[ivar] - fort;
                let iedge = edge_upper[(xy_step * ipc + pc_step * ixy as i32) as usize] - fort;
                let ind = xy_step * iedge + edge_step * ixy as i32;
                if iedge >= 0 && ifskip_edge[ind as usize] < skip_crit && ind != leave_ind - fort {
                    dxy_edge[nsum as usize] = -idir as f32 * dxedge[ind as usize];
                    dxy_edge[(nsum + nvar) as usize] = -idir as f32 * dyedge[ind as usize];
                    nsum += 1;
                }
            }
            num_edge[ixy] = nsum;

            /* If there are enough, get a trimmed mean vector, the deviations from
            the mean, and the median and MADN of those deviations */
            if nsum >= edge_med_crit {
                {
                    let (head, tail) = dxy_edge.split_at_mut((2 * nvar) as usize);
                    rs_trimmed_mean(head, nsum, 0.2f32, tail, &mut edge_tr_mean_x[ixy]);
                    rs_trimmed_mean(
                        &head[nvar as usize..],
                        nsum,
                        0.2f32,
                        tail,
                        &mut edge_tr_mean_y[ixy],
                    );
                }
                for i in 0..nsum as usize {
                    let ex = dxy_edge[i] - edge_tr_mean_x[ixy];
                    let ey = dxy_edge[i + nvar as usize] - edge_tr_mean_y[ixy];
                    // `(float)sqrt((double)ex * ex + ey * ey)`: the first term
                    // is a double product, the second a single-precision one.
                    dxy_edge[i + (2 * nvar) as usize] =
                        (ex as f64 * ex as f64 + (ey * ey) as f64).sqrt() as f32;
                    /* printf("%d %.2f,%.2f  %.2f,%.2f  %.3f\n", i, dxyEdge[i],
                    dxyEdge[i+nvar], ex, ey, dxyEdge[i + 2 * nvar]); */
                }
                {
                    let (head, tail) = dxy_edge.split_at_mut((2 * nvar) as usize);
                    rs_median(tail, nsum, head, &mut edge_dev_med[ixy]);
                    rs_madn(tail, nsum, edge_dev_med[ixy], head, &mut edge_dev_madn[ixy]);
                }
                /*printf("%d %s edges: trMean %.2f, %.2f, median dev %.3f MADN %.3f\n",
                nsum, ixy ?"Y":"X",  edgeTrMeanX[ixy],  edgeTrMeanY[ixy],
                edgeDevMed[ixy], edgeDevMADN[ixy]); */
            }
        }
    }

    initialize(
        ixpclist,
        iypclist,
        ivarpc,
        edge_lower,
        edge_upper,
        piece_lower,
        piece_upper,
        ifskip_edge,
        dxedge,
        Some(dyedge),
        &mut dxy_edge,
        dxyvar,
        &mut neigh_ind,
        Some(&mut neigh_wgt),
        &mut edge_dir,
        &mut neigh_list,
        &mut placed,
        &mut ivar_to_list,
        &mut list_to_var,
        indvar,
        nvar,
        fort,
        idir,
        leave_ind,
        skip_crit,
        xy_step,
        edge_step,
        pc_step,
    );

    let _num_neigh = neigh_ind[nvar as usize];

    /* Iterate */
    let mut xmove_last: f32 = 1.0e10;
    let mut xmove_avg: f32 = 0.;
    let mut ymove_last: f32 = 1.0e10;
    let mut ymove_avg: f32 = 0.;
    mad = 0.;
    let weight_interval = {
        let inner = if nvar / 2 < max_iter / 10 {
            nvar / 2
        } else {
            max_iter / 10
        };
        if 1 > inner { 1 } else { inner }
    };
    let mut compute_weights = 0;
    let mut did_weights = 0;
    let mut iter = 1;
    while iter <= max_iter {
        let mut sumxmove = 0.0f64;
        let mut sumymove = 0.0f64;
        let mut xmovemax = 0.0f32;
        let mut ymovemax = 0.0f32;
        let mut dxsum = 0.0f64;
        let mut dysum = 0.0f64;
        if robust_crit > 0. && iter % weight_interval == 0 {
            compute_weights = 1;
        }

        if compute_weights != 0 {
            let mut num_median = 0i32;
            let mut med_sum = 0.0f64;
            for list in 0..nvar as usize {
                for i in neigh_ind[list]..neigh_ind[list + 1] {
                    let j = (i - neigh_ind[list]) as usize;
                    let nay = neigh_list[i as usize];
                    erx[j] = dxyvar[2 * nay as usize] - dxyvar[2 * list] - dxy_edge[2 * i as usize];
                    ery[j] = dxyvar[2 * nay as usize + 1]
                        - dxyvar[2 * list + 1]
                        - dxy_edge[2 * i as usize + 1];
                    find_lowest_three(erx[j], j as i32, &mut xlow, &mut xsec, &mut xthr);
                    find_lowest_three(ery[j], j as i32, &mut ylow, &mut ysec, &mut ythr);
                }
                let nsum = {
                    let n = neigh_ind[list + 1] - neigh_ind[list];
                    if 1 > n { 1 } else { n }
                };
                if nsum >= 3 {
                    /* Get the median into xsec, ysec if 4 points */
                    if nsum > 3 {
                        xsec = ((xsec + xthr) as f64 / 2.) as f32;
                        ysec = ((ysec + ythr) as f64 / 2.) as f32;
                    }

                    /* Find deviation from median and get a median deviation */
                    for j in 0..nsum as usize {
                        let ex = erx[j] - xsec;
                        let ey = ery[j] - ysec;
                        errd[j] = ((ex * ex + ey * ey) as f64).sqrt() as f32;
                        find_lowest_three(errd[j], j as i32, &mut elow, &mut esec, &mut ethr);
                    }
                    if nsum > 3 {
                        esec = ((esec + ethr) as f64 / 2.) as f32;
                    }
                    med_sum += esec as f64;
                    num_median += 1;

                    /* Get the bisquare weighting factor and compute a weight */
                    if mad > 0. {
                        esec = if mad > esec { mad } else { esec };
                        for j in 0..nsum as usize {
                            let uu = (errd[j] - med_thresh * mad) / (ktune * esec);
                            let mut wgt = 0.0f32;
                            if uu < 1. {
                                // `uu <= 0 ? 1.f : (1. - uu * uu) * (1. - uu * uu)`
                                // -- the second arm is a double expression.
                                wgt = if uu <= 0. {
                                    1.0f32
                                } else {
                                    ((1. - (uu * uu) as f64) * (1. - (uu * uu) as f64)) as f32
                                };
                            }
                            neigh_wgt[(neigh_ind[list] + j as i32) as usize] = wgt;
                        }
                    }
                } else if nsum == 2
                    && (if num_edge[0] < num_edge[1] {
                        num_edge[0]
                    } else {
                        num_edge[1]
                    }) > edge_med_crit
                    && (if edge_dev_madn[0] < edge_dev_madn[1] {
                        edge_dev_madn[0]
                    } else {
                        edge_dev_madn[1]
                    }) > 1.0e-6
                {
                    /* For two edges, get their deviation from the median edge vector */
                    let mut i = 0i32;
                    for j in 0..2 {
                        i = neigh_ind[list] + j;
                        let ixy = (edge_dir[i as usize] / 2) as usize;
                        let isign = if edge_dir[i as usize] % 2 != 0 { 1 } else { -1 };
                        let ex = dxy_edge[2 * i as usize] - isign as f32 * edge_tr_mean_x[ixy];
                        let ey = dxy_edge[2 * i as usize + 1] - isign as f32 * edge_tr_mean_y[ixy];
                        let esec2 = ((ex * ex + ey * ey) as f64).sqrt() as f32;
                        let uu = (esec2 - edge_dev_med[ixy]) / (edge_dev_madn[ixy] * 4.685f32);
                        neigh_wgt[i as usize] = 0.;
                        if uu < 1. {
                            neigh_wgt[i as usize] = if uu <= 0. {
                                1.0f32
                            } else {
                                ((1. - (uu * uu) as f64) * (1. - (uu * uu) as f64)) as f32
                            };
                        }
                    }
                    let a = neigh_wgt[(i - 1) as usize];
                    let b = neigh_wgt[i as usize];
                    if (if a > b { a } else { b }) < 1.0e-2 {
                        neigh_wgt[i as usize] = 1.;
                        neigh_wgt[(i - 1) as usize] = 1.;
                    }
                }
            }
            if mad > 0. {
                did_weights = 1;
                compute_weights = 0;
            }

            mad = 0.;
            if num_median != 0 {
                let q = med_sum / num_median as f64;
                mad = (if 1.0e-5f64 > q { 1.0e-5f64 } else { q }) as f32;
            }
        }

        /* Loop on pieces, adjusting each one by weighted average error in edges */
        for list in 0..nvar as usize {
            let mut xsum = 0.0f32;
            let mut ysum = 0.0f32;
            let mut wsum = 0.0f32;
            for i in neigh_ind[list]..neigh_ind[list + 1] {
                let nay = neigh_list[i as usize];
                let wgt = neigh_wgt[i as usize];
                let ex = dxyvar[2 * nay as usize] - dxyvar[2 * list] - dxy_edge[2 * i as usize];
                let ey = dxyvar[2 * nay as usize + 1]
                    - dxyvar[2 * list + 1]
                    - dxy_edge[2 * i as usize + 1];
                xsum += ex * wgt;
                ysum += ey * wgt;
                wsum += wgt;
            }
            if wsum > 1.0e-6 {
                xsum /= wsum;
                ysum /= wsum;
            }
            dxyvar[2 * list] += xsum;
            dxyvar[2 * list + 1] += ysum;
            dxsum += dxyvar[2 * list] as f64;
            dysum += dxyvar[2 * list + 1] as f64;
            sumxmove += (xsum as f64).abs();
            sumymove += (ysum as f64).abs();
            // `B3DMAX(xmovemax, fabs((double)xsum))` is a double comparison
            // whose result is rounded back to the float `xmovemax`.
            let ax = (xsum as f64).abs();
            xmovemax = (if xmovemax as f64 > ax {
                xmovemax as f64
            } else {
                ax
            }) as f32;
            let ay = (ysum as f64).abs();
            ymovemax = (if ymovemax as f64 > ay {
                ymovemax as f64
            } else {
                ay
            }) as f32;
        }

        /* Shift to zero mean */
        let ex = (dxsum / nvar as f64) as f32;
        let ey = (dysum / nvar as f64) as f32;
        for list in 0..nvar as usize {
            dxyvar[2 * list] -= ex;
            dxyvar[2 * list + 1] -= ey;
        }

        /* stop if change was low */
        if xmovemax < crit_max_move && ymovemax < crit_max_move {
            if robust_crit <= 0. || did_weights != 0 {
                break;
            }
            compute_weights = 1;
        }
        /* Average the mean moves over some iterations, and test for a change
        in it periodically */
        if iter % interval_for_test >= interval_for_test - num_avg_for_test {
            xmove_avg += (sumxmove / nvar as f64) as f32;
            ymove_avg += (sumymove / nvar as f64) as f32;
        }
        if iter % interval_for_test == interval_for_test - 1 {
            xmove_avg /= num_avg_for_test as f32;
            ymove_avg /= num_avg_for_test as f32;
            if xmove_last - xmove_avg < crit_move_diff && ymove_last - ymove_avg < crit_move_diff {
                if robust_crit <= 0. || did_weights != 0 {
                    break;
                }
                compute_weights = 1;
            }
            xmove_last = xmove_avg;
            xmove_avg = 0.;
            ymove_last = ymove_avg;
            ymove_avg = 0.;
        }
        iter += 1;
    }

    // Compute weighted error mean and max
    let mut errsum = 0.0f64;
    let mut errmax = 0.0f64;
    let mut nsum = 0i32;
    for list in 0..nvar as usize {
        for i in neigh_ind[list]..neigh_ind[list + 1] {
            let nay = neigh_list[i as usize];
            let exf = dxyvar[2 * nay as usize] - dxyvar[2 * list] - dxy_edge[2 * i as usize];
            let ey =
                dxyvar[2 * nay as usize + 1] - dxyvar[2 * list + 1] - dxy_edge[2 * i as usize + 1];
            // `(float)sqrt((double)ex*ex+ey*ey) * neighWgt[i]`.
            let ex = ((exf as f64 * exf as f64 + (ey * ey) as f64).sqrt()
                * neigh_wgt[i as usize] as f64) as f32;
            errsum += ex as f64;
            errmax = if errmax > ex as f64 {
                errmax
            } else {
                ex as f64
            };
            nsum += 1;
        }
    }
    *w_err_max = errmax as f32;
    *w_err_mean = (errsum / (if 1 > nsum { 1 } else { nsum }) as f64) as f32;

    /* Rearrange the data */
    for ivar in 0..(2 * nvar) as usize {
        dxy_edge[ivar] = dxyvar[ivar];
    }
    for ivar in 0..nvar as usize {
        dxyvar[(xy_step * ivar as i32) as usize] = dxy_edge[2 * ivar_to_list[ivar] as usize];
        dxyvar[(xy_step * ivar as i32 + var_step) as usize] =
            dxy_edge[2 * ivar_to_list[ivar] as usize + 1];
    }

    for ivar in 0..(2 * nvar) as usize {
        dxy_edge[ivar] = 0.;
        placed[ivar] = 0;
    }

    *num_iter = iter;

    // Get the mean weight for each edge
    for list in 0..nvar as usize {
        for i in neigh_ind[list]..neigh_ind[list + 1] {
            let mut nay = neigh_list[i as usize];
            let mut _ipc = list as i32;
            let ixy = (edge_dir[i as usize] / 2) as i32;
            if edge_dir[i as usize] % 2 != 0 {
                _ipc = nay;
                nay = list as i32;
            }
            let ivar = list_to_var[nay as usize];
            dxy_edge[(2 * ivar + ixy) as usize] += neigh_wgt[i as usize];
            placed[(2 * ivar + ixy) as usize] = placed[(2 * ivar + ixy) as usize].wrapping_add(1);
        }
    }

    // Repack them into the work array, use -1 if no edge
    for ivar in 0..(2 * nvar) as usize {
        if placed[ivar] != 0 {
            work[ivar] = dxy_edge[ivar] / placed[ivar] as f32;
        } else {
            work[ivar] = -1.;
        }
    }
    0
}

/// Original `findpieceshifts` (`find_piece_shifts.c:447`).
#[allow(clippy::too_many_arguments)]
pub fn findpieceshifts(
    ivarpc: &[i32],
    nvar: &i32,
    indvar: &[i32],
    ixpclist: &[i32],
    iypclist: &[i32],
    dxedge: &[f32],
    dyedge: &[f32],
    idir: &i32,
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: &i32,
    dxyvar: &mut [f32],
    var_step: &i32,
    edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: &i32,
    work: &mut [f32],
    fort: &i32,
    leave_ind: &i32,
    skip_crit: &i32,
    robust_crit: &f32,
    crit_max_move: &f32,
    crit_move_diff: &f32,
    max_iter: &i32,
    num_avg_for_test: &i32,
    interval_for_test: &i32,
    num_iter: &mut i32,
    w_err_mean: &mut f32,
    w_err_max: &mut f32,
) -> i32 {
    find_piece_shifts(
        ivarpc,
        *nvar,
        indvar,
        ixpclist,
        iypclist,
        dxedge,
        dyedge,
        *idir,
        piece_lower,
        piece_upper,
        ifskip_edge,
        *edge_step,
        dxyvar,
        *var_step,
        edge_lower,
        edge_upper,
        *pc_step,
        work,
        *fort,
        *leave_ind,
        *skip_crit,
        *robust_crit,
        *crit_max_move,
        *crit_move_diff,
        *max_iter,
        *num_avg_for_test,
        *interval_for_test,
        num_iter,
        w_err_mean,
        w_err_max,
    )
}

/// Original `findPieceScalings` (`find_piece_shifts.c:487`).
#[allow(clippy::too_many_arguments)]
pub fn find_piece_scalings(
    ivarpc: &[i32],
    nvar: i32,
    indvar: &[i32],
    ixpclist: &[i32],
    iypclist: &[i32],
    dden_edge: &[f32],
    idir: i32,
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: i32,
    dden_var: &mut [f32],
    edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: i32,
    _work: &mut [f32],
    mut fort: i32,
    leave_ind: i32,
    skip_crit: i32,
    crit_max_move: f32,
    crit_move_diff: f32,
    max_iter: i32,
    num_avg_for_test: i32,
    interval_for_test: i32,
    num_iter: &mut i32,
    w_err_mean: &mut f32,
    w_err_max: &mut f32,
) -> i32 {
    /* Set pointers to temporary arrays from the work array */
    let mut ivar_to_list = vec![0i32; nvar as usize];
    let mut list_to_var = vec![0i32; nvar as usize];
    let mut neigh_ind = vec![0i32; (nvar + 1) as usize];
    let mut neigh_list = vec![0i32; (4 * nvar) as usize];
    let mut dtmp_edge = vec![0.0f32; (4 * nvar) as usize];
    let mut edge_dir = vec![0u8; (4 * nvar) as usize];
    let mut placed = vec![0u8; (2 * nvar) as usize];

    /* Set the xy stride parameter */
    let xy_step: i32;
    if edge_step == 1 && pc_step == 1 {
        xy_step = 2;
    } else if edge_step > 1 && pc_step > 1 {
        xy_step = 1;
    } else {
        return 1;
    }
    if fort != 0 {
        fort = 1;
    }

    initialize(
        ixpclist,
        iypclist,
        ivarpc,
        edge_lower,
        edge_upper,
        piece_lower,
        piece_upper,
        ifskip_edge,
        dden_edge,
        None,
        &mut dtmp_edge,
        dden_var,
        &mut neigh_ind,
        None,
        &mut edge_dir,
        &mut neigh_list,
        &mut placed,
        &mut ivar_to_list,
        &mut list_to_var,
        indvar,
        nvar,
        fort,
        idir,
        leave_ind,
        skip_crit,
        xy_step,
        edge_step,
        pc_step,
    );

    let _num_neigh = neigh_ind[nvar as usize];

    /* Iterate */
    let mut xmove_last: f32 = 1.0e10;
    let mut xmove_avg: f32 = 0.;
    let mut iter = 1;
    while iter <= max_iter {
        let mut sumxmove = 0.0f64;
        let mut xmovemax = 0.0f32;
        let mut dxsum = 0.0f64;

        /* Loop on pieces, adjusting each one by weighted average error in edges */
        for list in 0..nvar as usize {
            let mut xsum = 0.0f32;
            let mut wsum = 0.0f32;
            for i in neigh_ind[list]..neigh_ind[list + 1] {
                let nay = neigh_list[i as usize];
                let ex = dden_var[nay as usize] - dden_var[list] - dtmp_edge[i as usize];
                xsum += ex;
                wsum += 1.;
            }
            if wsum > 1.0e-6 {
                xsum /= wsum;
            }
            dden_var[list] += xsum;
            dxsum += dden_var[list] as f64;
            sumxmove += (xsum as f64).abs();
            let ax = (xsum as f64).abs();
            xmovemax = (if xmovemax as f64 > ax {
                xmovemax as f64
            } else {
                ax
            }) as f32;
        }

        /* Shift to zero mean */
        let ex = (dxsum / nvar as f64) as f32;
        for list in 0..nvar as usize {
            dden_var[list] -= ex;
        }

        /* stop if change was low */
        if xmovemax < crit_max_move {
            break;
        }
        /* Average the mean moves over some iterations, and test for a change
        in it periodically */
        if iter % interval_for_test >= interval_for_test - num_avg_for_test {
            xmove_avg += (sumxmove / nvar as f64) as f32;
        }
        if iter % interval_for_test == interval_for_test - 1 {
            xmove_avg /= num_avg_for_test as f32;
            if xmove_last - xmove_avg < crit_move_diff {
                break;
            }
            xmove_last = xmove_avg;
            xmove_avg = 0.;
        }
        iter += 1;
    }

    // Compute weighted error mean and max
    let mut errsum = 0.0f64;
    let mut errmax = 0.0f64;
    let mut nsum = 0i32;
    for list in 0..nvar as usize {
        for i in neigh_ind[list]..neigh_ind[list + 1] {
            let nay = neigh_list[i as usize];
            let ex = dden_var[nay as usize] - dden_var[list] - dtmp_edge[i as usize];
            errsum += (ex as f64).abs();
            errmax = if errmax > ex as f64 {
                errmax
            } else {
                ex as f64
            };
            nsum += 1;
        }
    }
    *w_err_max = errmax as f32;
    *w_err_mean = (errsum / (if 1 > nsum { 1 } else { nsum }) as f64) as f32;

    /* Rearrange the data */
    for ivar in 0..nvar as usize {
        dtmp_edge[ivar] = dden_var[ivar];
    }
    for ivar in 0..nvar as usize {
        dden_var[ivar] = dtmp_edge[ivar_to_list[ivar] as usize];
    }

    *num_iter = iter;
    0
}

/// Original `findpiecescalings` (`find_piece_shifts.c:635`).
#[allow(clippy::too_many_arguments)]
pub fn findpiecescalings(
    ivarpc: &[i32],
    nvar: &i32,
    indvar: &[i32],
    ixpclist: &[i32],
    iypclist: &[i32],
    ddenedge: &[f32],
    idir: &i32,
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: &i32,
    ddenvar: &mut [f32],
    edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: &i32,
    work: &mut [f32],
    fort: &i32,
    leave_ind: &i32,
    skip_crit: &i32,
    crit_max_move: &f32,
    crit_move_diff: &f32,
    max_iter: &i32,
    num_avg_for_test: &i32,
    interval_for_test: &i32,
    num_iter: &mut i32,
    w_err_mean: &mut f32,
    w_err_max: &mut f32,
) -> i32 {
    find_piece_scalings(
        ivarpc,
        *nvar,
        indvar,
        ixpclist,
        iypclist,
        ddenedge,
        *idir,
        piece_lower,
        piece_upper,
        ifskip_edge,
        *edge_step,
        ddenvar,
        edge_lower,
        edge_upper,
        *pc_step,
        work,
        *fort,
        *leave_ind,
        *skip_crit,
        *crit_max_move,
        *crit_move_diff,
        *max_iter,
        *num_avg_for_test,
        *interval_for_test,
        num_iter,
        w_err_mean,
        w_err_max,
    )
}

/// Original `pickAlternativeShifts` (`find_piece_shifts.c:710`).
#[allow(clippy::too_many_arguments)]
pub fn pick_alternative_shifts(
    ivarpc: &[i32],
    nvar: i32,
    indvar: &[i32],
    dxedge: &mut [f32],
    dyedge: &mut [f32],
    _piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: i32,
    _edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: i32,
    mut fort: i32,
    alt_dxys: &mut [f32],
    num_alts: i32,
    alt_ixy: i32,
    err_thresh: f32,
    reduce_fac: f32,
    new_thresh: f32,
    mut fixed_edges: Option<&mut [i32]>,
    num_fixed: &mut i32,
) -> i32 {
    let mut prm1st = [0i32; 2];
    let mut prm2nd = [0i32; 2];
    let mut alt1st = [0i32; 2];
    let mut alt2nd = [0i32; 2];
    let alt_step = 2 * num_alts;

    /* Set the xy stride parameter */
    let xy_step: i32;
    if edge_step == 1 && pc_step == 1 {
        xy_step = 2;
    } else if edge_step > 1 && pc_step > 1 {
        xy_step = 1;
    } else {
        return 1;
    }
    if fort != 0 {
        fort = 1;
    }

    // Look at each piece as a lower left corner; start in middle FWIW
    let mut var_start = nvar / 2;
    let mut var_end = 0;
    let mut idir = -1;
    while idir <= 1 {
        let mut ivar = var_start;
        while idir * (ivar - var_end) <= 0 {
            let ipc = ivarpc[ivar as usize] - fort;
            let mut num_full = 0;

            // Look for two edges starting in each direction
            for ixy in 0..2usize {
                let iyx = 1 - ixy;
                let mut iedge = edge_upper[(xy_step * ipc + pc_step * ixy as i32) as usize] - fort;
                let mut ind = xy_step * iedge + edge_step * ixy as i32;
                if iedge >= 0 && ifskip_edge[ind as usize] == 0 {
                    let mut neigh = piece_upper[ind as usize] - fort;
                    if indvar[neigh as usize] >= 0 {
                        // Got first edge, save it, look for the second edge in other direction
                        prm1st[ixy] = ind;
                        alt1st[ixy] = alt_step * iedge + alt_ixy * ixy as i32;
                        iedge =
                            edge_upper[(xy_step * neigh + pc_step * iyx as i32) as usize] - fort;
                        ind = xy_step * iedge + edge_step * iyx as i32;
                        if iedge >= 0 && ifskip_edge[ind as usize] == 0 {
                            neigh = piece_upper[ind as usize] - fort;
                            if indvar[neigh as usize] >= 0 {
                                prm2nd[ixy] = ind;
                                alt2nd[ixy] = alt_step * iedge + alt_ixy * iyx as i32;
                                num_full += 1;
                            }
                        }
                    }
                }
            }

            if num_full < 2 {
                ivar += idir;
                continue;
            }

            // Find minimum error, allowing up to 2 to be substituted; keep track of original
            // error
            let mut min_err: f32 = 1.0e10;
            let mut first_err: f32 = -1.;
            let (mut min_ind1, mut min_ind2, mut min_ind3, mut min_ind4) = (0i32, 0i32, 0i32, 0i32);
            for ind1 in -1..num_alts {
                for ind2 in -1..num_alts {
                    for ind3 in -1..num_alts {
                        for ind4 in -1..num_alts {
                            if i32::from(ind1 >= 0)
                                + i32::from(ind2 >= 0)
                                + i32::from(ind3 >= 0)
                                + i32::from(ind4 >= 0)
                                > 1
                            {
                                continue;
                            }
                            if (ind1 >= 0 && alt_dxys[(alt1st[0] + 2 * ind1) as usize] < -1.0e20)
                                || (ind2 >= 0
                                    && alt_dxys[(alt2nd[0] + 2 * ind2) as usize] < -1.0e20)
                                || (ind3 >= 0
                                    && alt_dxys[(alt1st[1] + 2 * ind3) as usize] < -1.0e20)
                                || (ind4 >= 0
                                    && alt_dxys[(alt2nd[1] + 2 * ind4) as usize] < -1.0e20)
                            {
                                continue;
                            }

                            // The error is edge 1 plus edge 2 - (edge 3 + edge 4)
                            let delx = (if ind1 < 0 {
                                dxedge[prm1st[0] as usize]
                            } else {
                                alt_dxys[(alt1st[0] + 2 * ind1) as usize]
                            }) + (if ind2 < 0 {
                                dxedge[prm2nd[0] as usize]
                            } else {
                                alt_dxys[(alt2nd[0] + 2 * ind2) as usize]
                            }) - ((if ind3 < 0 {
                                dxedge[prm1st[1] as usize]
                            } else {
                                alt_dxys[(alt1st[1] + 2 * ind3) as usize]
                            }) + (if ind4 < 0 {
                                dxedge[prm2nd[1] as usize]
                            } else {
                                alt_dxys[(alt2nd[1] + 2 * ind4) as usize]
                            }));
                            let dely = (if ind1 < 0 {
                                dyedge[prm1st[0] as usize]
                            } else {
                                alt_dxys[(alt1st[0] + 2 * ind1 + 1) as usize]
                            }) + (if ind2 < 0 {
                                dyedge[prm2nd[0] as usize]
                            } else {
                                alt_dxys[(alt2nd[0] + 2 * ind2 + 1) as usize]
                            }) - ((if ind3 < 0 {
                                dyedge[prm1st[1] as usize]
                            } else {
                                alt_dxys[(alt1st[1] + 2 * ind3 + 1) as usize]
                            }) + (if ind4 < 0 {
                                dyedge[prm2nd[1] as usize]
                            } else {
                                alt_dxys[(alt2nd[1] + 2 * ind4 + 1) as usize]
                            }));
                            let err = ((delx * delx + dely * dely) as f64).sqrt() as f32;
                            if first_err < 0. {
                                first_err = err;
                            }
                            if err < min_err {
                                min_err = err;
                                min_ind1 = ind1;
                                min_ind2 = ind2;
                                min_ind3 = ind3;
                                min_ind4 = ind4;
                            }
                        }
                    }
                }
            }
            // Replace whatever came out better
            if first_err > err_thresh && min_err < first_err * reduce_fac && min_err <= new_thresh {
                // `REPLACE_DXY(mnd, prm, alt)` (`find_piece_shifts.c:865`).
                for (mnd, prm, alt) in [
                    (min_ind1, prm1st[0], alt1st[0]),
                    (min_ind2, prm2nd[0], alt2nd[0]),
                    (min_ind3, prm1st[1], alt1st[1]),
                    (min_ind4, prm2nd[1], alt2nd[1]),
                ] {
                    if mnd >= 0 {
                        let ftmp = dxedge[prm as usize];
                        dxedge[prm as usize] = alt_dxys[(alt + 2 * mnd) as usize];
                        alt_dxys[(alt + 2 * mnd) as usize] = ftmp;
                        let ftmp = dyedge[prm as usize];
                        dyedge[prm as usize] = alt_dxys[(alt + 2 * mnd + 1) as usize];
                        alt_dxys[(alt + 2 * mnd + 1) as usize] = ftmp;
                        if let Some(fixed) = fixed_edges.as_deref_mut() {
                            fixed[*num_fixed as usize] = prm;
                            *num_fixed += 1;
                        }
                    }
                }
            }
            ivar += idir;
        }
        var_start = nvar / 2 + 1;
        var_end = nvar - 1;
        idir += 2;
    }
    0
}

/// Original `pickalternativeshifts` (`find_piece_shifts.c:846`).
#[allow(clippy::too_many_arguments)]
pub fn pickalternativeshifts(
    ivarpc: &[i32],
    nvar: &i32,
    indvar: &[i32],
    dxedge: &mut [f32],
    dyedge: &mut [f32],
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    edge_step: &i32,
    edge_lower: &[i32],
    edge_upper: &[i32],
    pc_step: &i32,
    fort: &i32,
    alt_dxys: &mut [f32],
    num_alts: &i32,
    alt_ixy: &i32,
    err_thresh: &f32,
    reduce_fac: &f32,
    new_thresh: &f32,
    fixed_edges: Option<&mut [i32]>,
    num_fixed: &mut i32,
) -> i32 {
    pick_alternative_shifts(
        ivarpc,
        *nvar,
        indvar,
        dxedge,
        dyedge,
        piece_lower,
        piece_upper,
        ifskip_edge,
        *edge_step,
        edge_lower,
        edge_upper,
        *pc_step,
        *fort,
        alt_dxys,
        *num_alts,
        *alt_ixy,
        *err_thresh,
        *reduce_fac,
        *new_thresh,
        fixed_edges,
        num_fixed,
    )
}

/// Original `findLowestThree` (`find_piece_shifts.c:864`).
fn find_lowest_three(val: f32, ind: i32, lowest: &mut f32, second: &mut f32, third: &mut f32) {
    if ind == 0 {
        *lowest = val;
    } else if val < *lowest {
        if ind > 1 {
            *third = *second;
        }
        *second = *lowest;
        *lowest = val;
    } else if ind <= 1 || val < *second {
        if ind > 1 {
            *third = *second;
        }
        *second = val;
    } else if ind <= 2 || val < *third {
        *third = val;
    }
}

/// Original `initialize` (`find_piece_shifts.c:885`).
///
/// Initialize either of those routines: figure out the division of pieces into
/// groups and initialize the shifts or scaling differences by summing all the
/// edge differences that involve one piece.
#[allow(clippy::too_many_arguments)]
fn initialize(
    ixpclist: &[i32],
    iypclist: &[i32],
    ivarpc: &[i32],
    edge_lower: &[i32],
    edge_upper: &[i32],
    piece_lower: &[i32],
    piece_upper: &[i32],
    ifskip_edge: &[i32],
    dxedge: &[f32],
    dyedge: Option<&[f32]>,
    dxy_edge: &mut [f32],
    dxyvar: &mut [f32],
    neigh_ind: &mut [i32],
    mut neigh_wgt: Option<&mut [f32]>,
    edge_dir: &mut [u8],
    neigh_list: &mut [i32],
    placed: &mut [u8],
    ivar_to_list: &mut [i32],
    list_to_var: &mut [i32],
    indvar: &[i32],
    nvar: i32,
    fort: i32,
    idir: i32,
    leave_ind: i32,
    skip_crit: i32,
    xy_step: i32,
    edge_step: i32,
    pc_step: i32,
) {
    let bigint = 100000000;
    let dxy_dim = if dyedge.is_some() { 2 } else { 1 };

    /* Initialize stuff */
    for ivar in 0..nvar as usize {
        placed[ivar] = 0;
        ivar_to_list[ivar] = -1;
    }
    let mut num_neigh = 0i32;
    let mut num_on_list = 0i32;

    /* Set up initial placement of pieces, loop on possibly unconnected sets */
    loop {
        /* find min/max coordinates of the unplaced pieces and the piece nearest
        the middle of this */
        let mut minxpc = bigint;
        let mut minypc = bigint;
        let mut maxxpc = -bigint;
        let mut maxypc = -bigint;
        for ivar in 0..nvar as usize {
            if placed[ivar] == 0 {
                let x = ixpclist[(ivarpc[ivar] - fort) as usize];
                let y = iypclist[(ivarpc[ivar] - fort) as usize];
                minxpc = if minxpc < x { minxpc } else { x };
                maxxpc = if maxxpc > x { maxxpc } else { x };
                minypc = if minypc < y { minypc } else { y };
                maxypc = if maxypc > y { maxypc } else { y };
            }
        }
        if minxpc == bigint {
            break;
        }

        let mut distmin: f32 = 1.0e30;
        let mut imin = 0i32;
        for ivar in 0..nvar as usize {
            if placed[ivar] == 0 {
                // `ixpclist[...] - 0.5 * (maxxpc + minxpc)` is an int minus a
                // double, so `dx` and `dy` are floats holding a double result.
                let dx = (ixpclist[(ivarpc[ivar] - fort) as usize] as f64
                    - 0.5 * (maxxpc + minxpc) as f64) as f32;
                let dy = (iypclist[(ivarpc[ivar] - fort) as usize] as f64
                    - 0.5 * (maxypc + minypc) as f64) as f32;
                let dist = dx * dx + dy * dy;
                if dist < distmin {
                    distmin = dist;
                    imin = ivar as i32;
                }
            }
        }

        /* Add this one to the list of variables then start searching */
        ivar_to_list[imin as usize] = num_on_list;
        list_to_var[num_on_list as usize] = imin;
        let mut list_ind = num_on_list;
        num_on_list += 1;
        while list_ind < num_on_list {
            let mut xsum = 0.0f32;
            let mut ysum = 0.0f32;
            let mut nsum = 0i32;

            /* Look at a piece and its edges for neighbors */
            let ipc = ivarpc[list_to_var[list_ind as usize] as usize] - fort;
            neigh_ind[list_ind as usize] = num_neigh;
            for ixy in 0..2i32 {
                for lowup in 0..2i32 {
                    let isign = 2 * lowup - 1;
                    let iedge = if lowup != 0 {
                        edge_upper[(xy_step * ipc + pc_step * ixy) as usize] - fort
                    } else {
                        edge_lower[(xy_step * ipc + pc_step * ixy) as usize] - fort
                    };
                    let ind = xy_step * iedge + edge_step * ixy;
                    if iedge >= 0
                        && ifskip_edge[ind as usize] < skip_crit
                        && ind != leave_ind - fort
                    {
                        let neighpc = if lowup != 0 {
                            piece_upper[ind as usize] - fort
                        } else {
                            piece_lower[ind as usize] - fort
                        };
                        let neighvar = indvar[neighpc as usize] - fort;
                        if neighvar < 0 {
                            continue;
                        }
                        /* If neighbor is not on var list yet, add it */
                        if ivar_to_list[neighvar as usize] < 0 {
                            list_to_var[num_on_list as usize] = neighvar;
                            ivar_to_list[neighvar as usize] = num_on_list;
                            num_on_list += 1;
                        }

                        /* Add neighbor to neighbor list and also the edge shift with the
                        right polarity */
                        let nay = ivar_to_list[neighvar as usize];
                        let ex = -idir as f32 * isign as f32 * dxedge[ind as usize];
                        dxy_edge[(dxy_dim * num_neigh) as usize] = ex;
                        let mut ey = 0.0f32;
                        if let Some(dyedge) = dyedge {
                            ey = -idir as f32 * isign as f32 * dyedge[ind as usize];
                            dxy_edge[(2 * num_neigh + 1) as usize] = ey;
                        }
                        if let Some(neigh_wgt) = neigh_wgt.as_deref_mut() {
                            neigh_wgt[num_neigh as usize] = 1.;
                        }
                        edge_dir[num_neigh as usize] = (lowup + 2 * ixy) as u8;
                        neigh_list[num_neigh as usize] = nay;
                        num_neigh += 1;

                        /* if the neighbor is placed, add estimated piece shift to sum */
                        /* dxyvar is indexed as in C for convenience and then rearranged
                        to correct indexing for the return */
                        if placed[neighvar as usize] != 0 {
                            xsum += dxyvar[(dxy_dim * nay) as usize] - ex;
                            if dyedge.is_some() {
                                ysum += dxyvar[(2 * nay + 1) as usize] - ey;
                            }
                            nsum += 1;
                        }
                    }
                }
            }

            /* Get average piece shift and place this piece */
            if nsum != 0 {
                xsum /= nsum as f32;
                ysum /= nsum as f32;
            }
            dxyvar[(dxy_dim * list_ind) as usize] = xsum;
            if dyedge.is_some() {
                dxyvar[(2 * list_ind + 1) as usize] = ysum;
            }
            placed[list_to_var[list_ind as usize] as usize] = 1;
            list_ind += 1;
        }
    }
    neigh_ind[nvar as usize] = num_neigh;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_three_preserves_source_ordered_insertions() {
        let mut low = 100.;
        let mut second = 100.;
        let mut third = 100.;
        find_lowest_three(4., 0, &mut low, &mut second, &mut third);
        find_lowest_three(2., 1, &mut low, &mut second, &mut third);
        find_lowest_three(3., 2, &mut low, &mut second, &mut third);
        assert_eq!((low, second, third), (2., 3., 4.));
    }

    #[test]
    fn find_piece_shifts_solves_and_centers_a_two_piece_overlap() {
        // Compact C layout: X edge 0 joins piece 0 to piece 1; all
        // other lower/upper edge slots are absent.
        let ivarpc = [0_i32, 1];
        let indvar = [0_i32, 1];
        let ixpclist = [0_i32, 1];
        let iypclist = [0_i32, 0];
        let piece_lower = [0_i32, 1];
        let edge_lower = [-1_i32, -1, 0, -1];
        let edge_upper = [0_i32, -1, -1, -1];
        let if_skip_edge = [0_i32, 0];
        let dxedge = [4.0_f32, 0.0];
        let dyedge = [0.0_f32, 0.0];
        let piece_upper = [1_i32, 0];
        let mut dxyvar = [0.0_f32; 4];
        let mut work = [0.0_f32; 64];
        let mut num_iter = 0_i32;
        let mut error_mean = 0.0_f32;
        let mut error_max = 0.0_f32;

        assert_eq!(
            find_piece_shifts(
                &ivarpc,
                2,
                &indvar,
                &ixpclist,
                &iypclist,
                &dxedge,
                &dyedge,
                1,
                &piece_lower,
                &piece_upper,
                &if_skip_edge,
                1,
                &mut dxyvar,
                1,
                &edge_lower,
                &edge_upper,
                1,
                &mut work,
                0,
                -1,
                1,
                0.0,
                1.0e-6,
                1.0e-6,
                50,
                1,
                1,
                &mut num_iter,
                &mut error_mean,
                &mut error_max,
            ),
            0
        );
        assert!(num_iter > 0);
        assert!((dxyvar[0] - 2.0).abs() < 1.0e-5);
        assert!((dxyvar[2] + 2.0).abs() < 1.0e-5);
        assert!(dxyvar[1].abs() < 1.0e-5 && dxyvar[3].abs() < 1.0e-5);
        assert!(error_mean.abs() < 1.0e-5 && error_max.abs() < 1.0e-5);

        assert_eq!(
            find_piece_shifts(
                &ivarpc,
                2,
                &indvar,
                &ixpclist,
                &iypclist,
                &dxedge,
                &dyedge,
                1,
                &piece_lower,
                &piece_upper,
                &if_skip_edge,
                1,
                &mut dxyvar,
                2,
                &edge_lower,
                &edge_upper,
                1,
                &mut work,
                0,
                -1,
                1,
                0.0,
                1.0e-6,
                1.0e-6,
                1,
                1,
                1,
                &mut num_iter,
                &mut error_mean,
                &mut error_max,
            ),
            1
        );
    }

    #[test]
    fn find_piece_scalings_solves_and_centers_a_two_piece_overlap() {
        let ivarpc = [0_i32, 1];
        let indvar = [0_i32, 1];
        let ixpclist = [0_i32, 1];
        let iypclist = [0_i32, 0];
        let piece_lower = [0_i32, 1];
        let edge_lower = [-1_i32, -1, 0, -1];
        let edge_upper = [0_i32, -1, -1, -1];
        let if_skip_edge = [0_i32, 0];
        let dden_edge = [0.2_f32, 0.0];
        let piece_upper = [1_i32, 0];
        let mut dden_var = [0.0_f32; 2];
        let mut work = [0.0_f32; 64];
        let mut num_iter = 0_i32;
        let mut error_mean = 0.0_f32;
        let mut error_max = 0.0_f32;

        assert_eq!(
            find_piece_scalings(
                &ivarpc,
                2,
                &indvar,
                &ixpclist,
                &iypclist,
                &dden_edge,
                1,
                &piece_lower,
                &piece_upper,
                &if_skip_edge,
                1,
                &mut dden_var,
                &edge_lower,
                &edge_upper,
                1,
                &mut work,
                0,
                -1,
                1,
                1.0e-6,
                1.0e-6,
                50,
                1,
                1,
                &mut num_iter,
                &mut error_mean,
                &mut error_max,
            ),
            0
        );
        assert!(num_iter > 0);
        assert!((dden_var[0] - 0.1).abs() < 1.0e-5);
        assert!((dden_var[1] + 0.1).abs() < 1.0e-5);
        assert!(error_mean.abs() < 1.0e-5 && error_max.abs() < 1.0e-5);

        assert_eq!(
            find_piece_scalings(
                &ivarpc,
                2,
                &indvar,
                &ixpclist,
                &iypclist,
                &dden_edge,
                1,
                &piece_lower,
                &piece_upper,
                &if_skip_edge,
                1,
                &mut dden_var,
                &edge_lower,
                &edge_upper,
                2,
                &mut work,
                0,
                -1,
                1,
                1.0e-6,
                1.0e-6,
                1,
                1,
                1,
                &mut num_iter,
                &mut error_mean,
                &mut error_max,
            ),
            1
        );
    }

    #[test]
    fn pick_alternative_shifts_replaces_the_single_bad_cycle_edge() {
        // 2 by 2 piece square.  Compact edge slots are X0, Y0, X1, Y1;
        // only X0 has an inconsistent displacement, and its one supplied
        // alternative closes the cycle exactly.
        let ivarpc = [0_i32, 1, 2, 3];
        let indvar = [0_i32, 1, 2, 3];
        let mut dxedge = [5.0_f32, 0.0, 1.0, 0.0];
        let mut dyedge = [0.0_f32; 4];
        let piece_lower = [0_i32, 0, 2, 1];
        let piece_upper = [1_i32, 2, 3, 3];
        let if_skip_edge = [0_i32; 4];
        let edge_lower = [-1_i32, -1, 0, -1, -1, 0, 1, 1];
        let edge_upper = [0_i32, 0, -1, 1, 1, -1, -1, -1];
        // X alternatives occupy 0..4 and Y alternatives begin at 4.
        let mut alternatives = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut fixed_edges = [0_i32; 4];
        let mut num_fixed = 0_i32;

        assert_eq!(
            pick_alternative_shifts(
                &ivarpc,
                4,
                &indvar,
                &mut dxedge,
                &mut dyedge,
                &piece_lower,
                &piece_upper,
                &if_skip_edge,
                1,
                &edge_lower,
                &edge_upper,
                1,
                0,
                &mut alternatives,
                1,
                4,
                1.0,
                0.5,
                0.1,
                Some(&mut fixed_edges),
                &mut num_fixed,
            ),
            0
        );
        assert_eq!(num_fixed, 1);
        assert_eq!(fixed_edges[0], 0);
        assert_eq!(dxedge[0], 1.0);
        assert_eq!(alternatives[0], 5.0);
    }
}
