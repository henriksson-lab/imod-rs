//! Translation of `IMOD/flib/tiltalign/leaveout.cpp` — routines for leaving
//! out a random subset and getting errors.
//!
//! The file-scope `static AlignVariables *av;` (`leaveout.cpp:16`), set by
//! `leaveoutSetPointers`, is a parameter here (`alivar.rs`).
//!
//! # `rand()`
//!
//! The source draws from the C library's `rand()`, seeded by `srand` in
//! `tiltalign.cpp:445,767`.  The process-wide glibc TYPE_3 state already lives
//! in `libcfshr/b3dutil.rs` behind `b3dsrand`/`b3drand`, which is what the
//! translated `srand(randSeed)` must call so that both units share one
//! generator.  `b3drand()` returns `(float)rand() / (float)RAND_MAX`, and
//! `(float)RAND_MAX` is exactly 2^31, so `b3drand() * RAND_MAX as f32`
//! recovers `(float)rand()` exactly — which is the only form in which this
//! unit uses the value (`(numAtMin * (float)rand()) / (float)RAND_MAX`).
//!
//! # Arithmetic
//!
//! `pow(float, 2.f)` is the C++ `float` overload, which g++ folds to a
//! single-precision `x * x`; it is written as that product.

use std::io::Write;

use super::alivar::AlignVariables;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3drand, balanced_group_limits, c_format_bytes,
};
use crate::imod::libcfshr::robuststat::rs_sort_indexed_floats;

/// glibc's `RAND_MAX`.
const RAND_MAX: i32 = 2147483647;

/// Original: `leaveoutSetPointers` (`leaveout.cpp:18`).
///
/// The source stores the pointer in a file-scope static; the functions of this
/// module take it as a parameter instead, so there is nothing to store.
pub fn leaveout_set_pointers(av_in: &mut AlignVariables) {
    let _ = av_in;
}

/// Original: `leaveOutPoints` (`leaveout.cpp:27`) — leave out given fraction
/// of points, with contiguous number to use predictions from or 0 for
/// contours, and additional points of padding to each side.
pub fn leave_out_points(
    av: &mut AlignVariables,
    contig_to_predict: i32,
    pad_to_leave_out: i32,
    frac_leave_out: f32,
) {
    let mut excluded: Vec<i32> = Vec::new();
    let mut neigh_ind: Vec<i32> = Vec::new();
    let mut cross_ind: Vec<i32> = Vec::new();
    let mut real_in_range: Vec<i32> = Vec::new();
    let mut infeasible: Vec<i32> = Vec::new();
    let mut num_excluded_at_z: Vec<i32> = Vec::new();
    let mut neigh_dist: Vec<f32> = Vec::new();
    let mut avail_ind: i32;
    let mut avail: i32;
    let mut ind: i32;
    let mut num_leave: i32;
    let mut num_avail: i32;
    let mut num_neigh: i32;
    let total_leave: i32;
    let mut num_per_range: i32;
    let mut num_zranges: i32;
    let mut iv_start: i32 = 0;
    let mut iv_end: i32 = 0;
    let mut range_size: i32;
    let pred_above: i32;
    let mut num_in_range: i32;
    let mut min_used: i32;
    let mut num_at_min: i32;
    let mut num: i32;
    let mut iv_cen: i32;
    let mut leave_start: i32;
    let mut leave_end: i32;
    let pred_below: i32;
    let num_proj = av.ireal_str[av.nreal_pt as usize] - 1;
    let contig_pad = contig_to_predict + 2 * pad_to_leave_out;
    let mut num_ind: i32;
    let mut min_zused: i32;
    let mut num_at_zmin: i32;
    let mut num_train_proj: i32;
    let mut num_train_real: i32;
    let num_real_try = 10;
    let num_proj_try = 10;
    let num_leave_try = 2;
    let mut no_good: bool = false;
    let mut found_point: bool;

    pred_below = contig_to_predict / 2;
    pred_above = contig_to_predict - (1 + pred_below);
    av.real_left_out[..av.nreal_pt as usize].fill(0);
    av.proj_left_out[..num_proj as usize].fill(0);
    av.proj_to_predict[..num_proj as usize].fill(0);

    for ind in 0..num_proj as usize {
        av.weight[ind] = 1.;
    }

    // Set left out for the test set and compute # available for training set
    num_train_real = av.nreal_pt;
    num_train_proj = num_proj;
    for ind in 0..av.nreal_pt as usize {
        if av.real_in_test_set[ind] != 0 {
            av.real_left_out[ind] = 1;
            num_train_real -= 1;
            for jnd in av.ireal_str[ind] - 1..av.ireal_str[ind + 1] - 1 {
                av.proj_left_out[jnd as usize] = 1;
                av.weight[jnd as usize] = 0.;
            }
            num_train_proj -= av.ireal_str[ind + 1] - av.ireal_str[ind];
        }
    }

    // Contours
    if contig_to_predict == 0 {
        excluded.resize(av.nreal_pt as usize, 0);
        num_leave = 1.max((num_train_real as f32 * frac_leave_out) as i32);
        num_avail = num_train_real;

        for ind in 0..av.nreal_pt as usize {
            excluded[ind] = av.real_in_test_set[ind];
        }

        // If only leaving one out, just exclude ones left out before
        if num_leave == 1 {
            for ind in 0..av.nreal_pt as usize {
                if av.real_in_test_set[ind] == 0 {
                    excluded[ind] = av.times_left_out[ind];
                }
            }
        }

        // Loop on number to leave out, get an index to ones available
        for leave in 0..num_leave {
            min_used = 1000000;
            num_at_min = 0;
            for ind in 0..av.nreal_pt as usize {
                if excluded[ind] == 0 {
                    num = av.times_left_out[ind];
                    if num < min_used {
                        min_used = av.times_left_out[ind];
                        num_at_min = 1;
                    } else if num == min_used {
                        num_at_min += 1;
                    }
                }
            }

            avail_ind =
                ((num_at_min as f32 * (b3drand() * RAND_MAX as f32)) / RAND_MAX as f32) as i32;
            // B3DCLAMP(availInd, 0, numAtMin - 1)
            avail_ind = if num_at_min - 1 < avail_ind {
                num_at_min - 1
            } else {
                avail_ind
            };
            avail_ind = if 0 > avail_ind { 0 } else { avail_ind };

            // Find the available one with that index
            avail = 0;
            for ind in 0..av.nreal_pt as usize {
                if excluded[ind] == 0 && av.times_left_out[ind] == min_used {
                    if avail == avail_ind {
                        // Mark it as excluded here, and real point and proj points left out
                        excluded[ind] = 1;
                        av.real_left_out[ind] = 1;
                        av.times_left_out[ind] += 1;
                        for jnd in av.ireal_str[ind] - 1..av.ireal_str[ind + 1] - 1 {
                            av.proj_left_out[jnd as usize] = 1;
                            av.weight[jnd as usize] = 0.;
                            av.proj_to_predict[jnd as usize] = 1;
                        }

                        // Find the nearest neighbors: get all distances to available ones
                        num_neigh = num_train_real / num_leave - 1;
                        if leave < num_train_real % num_leave {
                            num_neigh += 1;
                        }
                        neigh_dist.clear();
                        neigh_ind.clear();
                        cross_ind.clear();
                        for jnd in 0..av.nreal_pt as usize {
                            if excluded[jnd] == 0 {
                                let dx = av.xyz[jnd * 3] - av.xyz[ind * 3];
                                let dy = av.xyz[jnd * 3 + 1] - av.xyz[ind * 3 + 1];
                                neigh_dist.push(dx * dx + dy * dy);
                                neigh_ind.push(neigh_ind.len() as i32);
                                cross_ind.push(jnd as i32);
                            }
                        }

                        // Sort them, exclude the closest ones, and adjust number available
                        let n_ind = neigh_ind.len() as i32;
                        rs_sort_indexed_floats(&neigh_dist, &mut neigh_ind, n_ind);
                        // B3DMIN(numNeigh, neighInd.size()) compares as size_t.
                        let lim = (num_neigh as usize).min(neigh_ind.len());
                        for jnd in 0..lim {
                            excluded[cross_ind[neigh_ind[jnd] as usize] as usize] = 1;
                        }

                        num_avail -= num_neigh + 1;
                        break;
                    }
                    avail += 1;
                }
            }
            if num_avail < 0 || (num_avail == 0 && leave < num_leave - 1) {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "leaveOutPoints problem: numLeave %d leave %d numAvail %d\n",
                    &[
                        CArg::Int(num_leave as i64),
                        CArg::Int(leave as i64),
                        CArg::Int(num_avail as i64),
                    ],
                ));
            }
        }
    } else {
        // Contiguous points
        total_leave = 1.max((num_train_proj as f32 * frac_leave_out / contig_pad as f32) as i32);
        num_per_range = ((total_leave as f64).powf(0.667) + 0.5).floor() as i32;
        num_zranges = total_leave / num_per_range;
        if num_zranges > av.nview {
            num_zranges = av.nview;
            num_per_range = 1.max(total_leave / num_zranges);
        }
        av.times_left_out[..av.nreal_pt as usize].fill(0);

        // Loop on the ranges, get limits and number to leave out
        for range in 0..num_zranges {
            num_leave = num_per_range;
            if range < total_leave % num_per_range {
                num_leave += 1;
            }
            balanced_group_limits(av.nview, num_zranges, range, &mut iv_start, &mut iv_end);
            iv_start += 1;
            iv_end += 1;
            range_size = iv_end + 1 - iv_start;

            // Make list of contours with points in range
            real_in_range.clear();
            for ind in 0..av.nreal_pt as usize {
                if av.real_in_test_set[ind] != 0 {
                    continue;
                }
                num_in_range = 0;
                for jnd in av.ireal_str[ind] - 1..av.ireal_str[ind + 1] - 1 {
                    let jnd = jnd as usize;
                    if av.proj_left_out[jnd] == 0
                        && av.isec_view[jnd] >= iv_start
                        && av.isec_view[jnd] <= iv_end
                    {
                        num_in_range += 1;
                    }
                }

                let lim = if contig_to_predict / 2 < range_size - 1 {
                    contig_to_predict / 2
                } else {
                    range_size - 1
                };
                if num_in_range > lim {
                    real_in_range.push(ind as i32);
                }
            }

            num_in_range = real_in_range.len() as i32;
            if num_in_range == 0 {
                break;
            }
            num_avail = num_in_range;

            // Set up arrays to keep track of exclusions etc
            infeasible.clear();
            infeasible.resize(num_in_range as usize, 0);
            excluded.clear();
            excluded.resize(num_in_range as usize, 0);
            num_excluded_at_z.clear();
            num_excluded_at_z.resize(range_size as usize, 0);

            // Try to leave out the target number; do nested multiple trials if ncessary
            for leave in 0..num_leave {
                found_point = false;
                let mut leave_try = 0;
                while leave_try < num_leave_try && num_avail != 0 && !found_point {
                    let mut real_try = 0;
                    while real_try < num_real_try && num_avail != 0 && !found_point {
                        // Find minimum times used and number there
                        min_used = 1000000;
                        num_at_min = 0;
                        for in_ran in 0..num_in_range as usize {
                            if infeasible[in_ran] == 0 && excluded[in_ran] == 0 {
                                num = av.times_left_out[real_in_range[in_ran] as usize];
                                if num < min_used {
                                    min_used = num;
                                    num_at_min = 1;
                                } else if num == min_used {
                                    num_at_min += 1;
                                }
                            }
                        }

                        // Select one and find it
                        avail_ind = ((num_at_min as f32 * (b3drand() * RAND_MAX as f32))
                            / RAND_MAX as f32) as i32;
                        avail_ind = if num_at_min - 1 < avail_ind {
                            num_at_min - 1
                        } else {
                            avail_ind
                        };
                        avail_ind = if 0 > avail_ind { 0 } else { avail_ind };
                        avail = 0;
                        for in_ran in 0..num_in_range as usize {
                            ind = real_in_range[in_ran];
                            if infeasible[in_ran] == 0
                                && excluded[in_ran] == 0
                                && av.times_left_out[ind as usize] == min_used
                            {
                                if avail == avail_ind {
                                    let indu = ind as usize;
                                    // Now try to find a random point to center exclusion at
                                    let proj_lim = if num_proj_try < range_size {
                                        num_proj_try
                                    } else {
                                        range_size
                                    };
                                    for _proj_try in 0..proj_lim {
                                        // Find Z values with fewest points left out
                                        num_at_zmin = 0;
                                        min_zused = 100000;
                                        for iv in 0..range_size as usize {
                                            num = num_excluded_at_z[iv];
                                            if num < min_zused {
                                                min_zused = num;
                                                num_at_zmin = 1;
                                            } else if num == min_zused {
                                                num_at_zmin += 1;
                                            }
                                        }

                                        // Select among the number at minimum and find what
                                        // view that is
                                        num_ind = ((num_at_zmin as f32
                                            * (b3drand() * RAND_MAX as f32))
                                            / RAND_MAX as f32)
                                            as i32;
                                        num_ind = if num_at_zmin - 1 < num_ind {
                                            num_at_zmin - 1
                                        } else {
                                            num_ind
                                        };
                                        num_ind = if 0 > num_ind { 0 } else { num_ind };
                                        num = 0;
                                        let mut iv = 0;
                                        while iv < range_size {
                                            if num_excluded_at_z[iv as usize] == min_zused {
                                                if num == num_ind {
                                                    break;
                                                }
                                                num += 1;
                                            }
                                            iv += 1;
                                        }
                                        iv_cen = iv_start + iv;

                                        // This is the range to leave out,
                                        // There must not have any left out points one beyond it
                                        leave_start = iv_cen - (pred_below + pad_to_leave_out);
                                        leave_end = iv_cen + (pred_above + pad_to_leave_out);
                                        no_good = false;
                                        found_point = false;
                                        for jnd in
                                            av.ireal_str[indu] - 1..av.ireal_str[indu + 1] - 1
                                        {
                                            let jnd = jnd as usize;
                                            if av.proj_left_out[jnd] != 0
                                                && av.isec_view[jnd] >= leave_start - 1
                                                && av.isec_view[jnd] <= leave_end + 1
                                            {
                                                no_good = true;
                                                break;
                                            }
                                            if av.isec_view[jnd] == iv_cen {
                                                found_point = true;
                                            }
                                        }
                                        if !no_good & found_point {
                                            // We have a winner: mark the points in range
                                            num_excluded_at_z[(iv_cen - iv_start) as usize] += 1;
                                            for jnd in
                                                av.ireal_str[indu] - 1..av.ireal_str[indu + 1] - 1
                                            {
                                                let jnd = jnd as usize;
                                                if av.isec_view[jnd] >= leave_start
                                                    && av.isec_view[jnd] <= leave_end
                                                {
                                                    av.proj_left_out[jnd] = 1;
                                                    av.weight[jnd] = 0.;
                                                }
                                                if av.isec_view[jnd] >= iv_cen - pred_below
                                                    && av.isec_view[jnd] <= iv_cen + pred_above
                                                {
                                                    av.proj_to_predict[jnd] = 1;
                                                }
                                            }
                                            av.times_left_out[indu] += 1;

                                            // Mark this as excluded (possibly temporarily) and
                                            // find neighbors
                                            excluded[in_ran] = 1;
                                            num_neigh = num_in_range / num_leave - 1;
                                            if leave < num_in_range % num_leave {
                                                num_neigh += 1;
                                            }
                                            neigh_dist.clear();
                                            neigh_ind.clear();
                                            cross_ind.clear();
                                            for jn_ran in 0..num_in_range as usize {
                                                let jnd = real_in_range[jn_ran] as usize;
                                                if excluded[jn_ran] == 0 {
                                                    let dx = av.xyz[jnd * 3] - av.xyz[indu * 3];
                                                    let dy =
                                                        av.xyz[jnd * 3 + 1] - av.xyz[indu * 3 + 1];
                                                    neigh_dist.push(dx * dx + dy * dy);
                                                    neigh_ind.push(neigh_ind.len() as i32);
                                                    cross_ind.push(jn_ran as i32);
                                                }
                                            }

                                            // Sort them, exclude the closest ones
                                            let n_ind = neigh_ind.len() as i32;
                                            rs_sort_indexed_floats(
                                                &neigh_dist,
                                                &mut neigh_ind,
                                                n_ind,
                                            );
                                            let lim = (num_neigh as usize).min(neigh_ind.len());
                                            for jnd in 0..lim {
                                                excluded
                                                    [cross_ind[neigh_ind[jnd] as usize] as usize] =
                                                    1;
                                            }

                                            break;
                                        }
                                    }

                                    // After trying different points, if still no good, mark
                                    // as infeasible
                                    if no_good || !found_point {
                                        infeasible[in_ran] = 1;
                                        num_avail -= 1;
                                        found_point = false;
                                    }
                                    break;
                                }
                                avail += 1;
                            }
                        }
                        real_try += 1;
                    }
                    leave_try += 1;
                }
                // If that didn't work, get rid of excluded and try again
                if !found_point && leave_try < num_leave_try - 1 {
                    excluded.clear();
                    excluded.resize(num_in_range as usize, 0);
                }
            }
        }
    }
}

/// Original: `getLeaveOutErrors` (`leaveout.cpp:351`) — get the errors from
/// points to predict.
pub fn get_leave_out_errors(av: &mut AlignVariables, weights: &[f32], rob: i32) {
    let rob_ind = (if rob > 0 { rob } else { 0 }) as usize;
    let mut err_sq: f64;
    for ind in 0..(av.ireal_str[av.nreal_pt as usize] - 1) as usize {
        if av.proj_to_predict[ind] != 0 {
            err_sq = (av.xresid[ind] * av.xresid[ind] + av.yresid[ind] * av.yresid[ind]) as f64;
            av.lv_out_err_sum[rob_ind] += err_sq.sqrt();
            av.lv_out_err_sq_sum[rob_ind] += err_sq;
            av.num_lv_out_err[rob_ind] += 1;
            if av.robust_weights != 0 || rob < 0 {
                av.lv_out_wgt_sum[rob_ind] += err_sq.sqrt() * weights[ind] as f64;
                av.lv_out_wgt_sq_sum[rob_ind] += err_sq * weights[ind] as f64;
                av.num_lv_out_wgt_err[rob_ind] += 1;
            }
        }
    }
}

/// Original: `getTestSetErrors` (`leaveout.cpp:371`) — get the errors from a
/// test set.
pub fn get_test_set_errors(av: &mut AlignVariables, rob_ind: i32) {
    let rob_ind = rob_ind as usize;
    let mut err_sq: f64;
    for jnd in 0..av.nreal_pt as usize {
        if av.real_in_test_set[jnd] != 0 {
            for ind in av.ireal_str[jnd] - 1..av.ireal_str[jnd + 1] - 1 {
                let ind = ind as usize;
                err_sq = (av.xresid[ind] * av.xresid[ind] + av.yresid[ind] * av.yresid[ind]) as f64;
                av.lv_out_err_sum[rob_ind] += err_sq.sqrt();
                av.lv_out_err_sq_sum[rob_ind] += err_sq;
                av.num_lv_out_err[rob_ind] += 1;
            }
        }
    }
}

/// Original: `clearLeaveOutErrors` (`leaveout.cpp:389`) — clear all the
/// errors, which are non-robust and robust leave-out, and non-robust and
/// robust test set, with unweighted for all and weighted for leave-out.
pub fn clear_leave_out_errors(av: &mut AlignVariables) {
    for ind in 0..6 {
        av.lv_out_err_sum[ind] = 0.;
        av.lv_out_err_sq_sum[ind] = 0.;
        av.num_lv_out_err[ind] = 0;
        av.lv_out_wgt_sum[ind] = 0.;
        av.lv_out_wgt_sq_sum[ind] = 0.;
        av.num_lv_out_wgt_err[ind] = 0;
    }
}
