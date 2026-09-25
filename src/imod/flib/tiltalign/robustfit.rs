//! Translation of `IMOD/flib/tiltalign/robustfit.cpp` — routines related to
//! robust fitting.
//!
//! The file-scope `static AlignVariables *av;` and `static ArrayMaxes *mx;`
//! (`robustfit.cpp:16-17`), set by `robustSetPointers`, are parameters here
//! (`alivar.rs`).  A `NULL` `av->realInTestSet` is an empty `Vec`, so the
//! source's `av->realInTestSet && av->realInTestSet[i]` is
//! `!av.real_in_test_set.is_empty() && av.real_in_test_set[i] != 0`.
//!
//! # Arithmetic
//!
//! `pow(x, 2)` on a `float` is the C++ `pow(float, int)` overload, which
//! promotes to `double`; g++ folds `pow(d, 2.0)` to `d * d`, so it is written
//! as that product of the widened value.  `trackResid += sqrt(...)` is a
//! `float` plus a `double`, rounded back to `float` per addition; `B3DMIN`/
//! `B3DMAX` with a `double` literal compare and choose in `double`.

use std::io::Write;

use super::alivar::AlignVariables;
use super::arraymaxes::{ArrayMaxes, MAX_WGT_RINGS};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::robuststat::{rs_fast_madn, rs_fast_median, rs_sort_indexed_floats};

/// Original: `robustSetPointers` (`robustfit.cpp:23`).
///
/// The source stores the two pointers in file-scope statics; the functions of
/// this module take them as parameters instead, so there is nothing to store.
pub fn robust_set_pointers(av_in: &mut AlignVariables, mx_in: &mut ArrayMaxes) {
    let _ = (av_in, mx_in);
}

/// Original: `setupWeightGroups` (`robustfit.cpp:43`) — sets up weighting
/// groups, consisting of sets of views divided into rings if possible.
///  - `maxRings` = maximum number of rings allowed
///  - `minRes` = minimum number of residuals in the group
///  - `minTiltView` = view number at minimum tilt
///  - `ierr` = return error value, 0 for success or 1 for failure
///
/// For ordinary weighting per point, `ivStartWgtGroup` has the starting view
/// index of the weight group; `ipStartWgtView` has the starting projection
/// point index for each view subset in each group; `indProjWgtList` has the
/// true 2D point index for each projection point index.
pub fn setup_weight_groups(
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    max_rings: i32,
    min_res: i32,
    min_tilt_view: i32,
    ierr: &mut i32,
) {
    let max_views_for_rings: [i32; MAX_WGT_RINGS as usize] = [1, 10, 8, 6, 5, 5, 4, 4, 3, 3];
    let mut ireal_ring_list: Vec<i32> = vec![0; mx.max_real as usize];
    let mut dist_real: Vec<f32> = vec![0.; mx.max_real as usize];
    let mut nreal_per_ring: i32;
    let mut needed_views: i32;
    let mut num_view_groups: i32;
    let mut num_extra: i32;
    let mut iex_start: i32;
    let mut ngrp_before_ex: i32;
    let mut ind_proj: i32;
    let mut ind_group: i32;
    let mut ind_view: i32;
    let mut ivbase: i32;
    let mut irbase: i32;
    let mut nin_group: i32;
    let mut nin_ring: i32;
    let mut ireal: i32;
    let mut num_pts: i32;
    let mut num_real_used: i32;
    let mut exit_group: bool;
    let mut exit_num_view: bool;

    let mut nreal_for_views = av.nreal_pt;
    if av.patch_track_model != 0 {
        nreal_for_views = av.num_full_tracks_used;
    }

    // Get distances from center and get sorted indexes to them
    // irealRingList are indexes from 0 not 1
    num_real_used = 0;
    for i in 0..av.nreal_pt as usize {
        if !(av.leaving_out != 0 && av.real_left_out[i] != 0)
            && !(!av.real_in_test_set.is_empty() && av.real_in_test_set[i] != 0)
        {
            let x = av.xyz[i * 3] as f64;
            let y = av.xyz[i * 3 + 1] as f64;
            dist_real[num_real_used as usize] = (x * x + y * y).sqrt() as f32;
            ireal_ring_list[num_real_used as usize] = num_real_used;
            num_real_used += 1;
        }
    }
    rs_sort_indexed_floats(&dist_real, &mut ireal_ring_list, num_real_used);

    // Loop from largest number of rings down, first evaluate plausibility if
    // all points are present.  Here use the number of full tracks for patch tracking
    // NUM_RING_LOOP:
    let mut nring = max_rings;
    while nring >= 1 {
        *ierr = 1;
        nreal_per_ring = nreal_for_views / nring;
        if nreal_per_ring == 0 {
            nring -= 1;
            continue;
        }
        needed_views = 1.max(min_res / nreal_per_ring);
        if needed_views > max_views_for_rings[(nring - 1) as usize] && nring > 1 {
            nring -= 1;
            continue;
        }
        //
        // Try to set up groups of views of increasing sizes until one works
        // NUM_VIEW_LOOP:
        exit_num_view = false;
        let mut num_views = needed_views;
        while num_views <= av.nview && !exit_num_view {
            num_view_groups = av.nview / num_views;
            num_extra = av.nview % num_views;
            iex_start = if 1 > min_tilt_view - num_extra / 2 {
                1
            } else {
                min_tilt_view - num_extra / 2
            };
            ngrp_before_ex = (iex_start - 1) / num_views;
            ind_proj = 1;
            ivbase = 0;
            ind_group = 1;
            ind_view = 1;
            //
            // Loop on the groups of views
            // GROUP_LOOP:
            exit_group = false;
            let mut igroup = 1;
            while igroup <= num_view_groups && !exit_group {
                irbase = 0;
                nin_group = num_views;
                if igroup > ngrp_before_ex && igroup <= ngrp_before_ex + num_extra {
                    nin_group += 1;
                }
                //
                // loop on the rings in views; here base the number on the total number
                // of real points not full tracks, because each will be considered for
                // whether they have points in the view
                // RING_LOOP:
                let mut iring = 1;
                while iring <= nring && !exit_group {
                    nin_ring = num_real_used / nring;
                    if iring > nring - (num_real_used % nring) {
                        nin_ring += 1;
                    }
                    //
                    // This is one weight group, set the starting view index of it
                    av.iv_start_wgt_group[(ind_group - 1) as usize] = ind_view;
                    ind_group += 1;
                    //
                    // loop on the views in the group; for each one, set the starting index
                    // in the projection list
                    for iv in ivbase + 1..=ivbase + nin_group {
                        av.ip_start_wgt_view[(ind_view - 1) as usize] = ind_proj;
                        ind_view += 1;
                        //
                        // Loop on the real points in the ring, and for each one on the
                        // given view, add its projection point index to the list
                        for ind in irbase + 1..=irbase + nin_ring {
                            ireal = ireal_ring_list[(ind - 1) as usize] + 1;
                            let mut iproj = av.ireal_str[(ireal - 1) as usize];
                            while iproj <= av.ireal_str[ireal as usize] - 1 {
                                if av.isec_view[(iproj - 1) as usize] == iv
                                    && !(av.leaving_out != 0
                                        && av.proj_left_out[(iproj - 1) as usize] != 0)
                                {
                                    av.ind_proj_wgt_list[(ind_proj - 1) as usize] = iproj;
                                    ind_proj += 1;
                                }
                                iproj += 1;
                            }
                        }
                    }
                    //
                    // After each ring, increase the ring base index
                    irbase += nin_ring;
                    //
                    // But if there are too few in this group, make view groups bigger if
                    // possible for this number of rings; otherwise go on to try fewer
                    // rings; but if there is only one ring and it is up to all views,
                    // push on
                    num_pts = ind_proj
                        - av.ip_start_wgt_view
                            [(av.iv_start_wgt_group[(ind_group - 2) as usize] - 1) as usize];
                    if num_pts < min_res {
                        *ierr = 1;
                        if num_views >= max_views_for_rings[(nring - 1) as usize] && nring > 1 {
                            exit_group = true;
                            exit_num_view = true;
                            break;
                        }
                        if nring > 1 || num_views < av.nview {
                            exit_group = true;
                            break;
                        }
                    }
                    iring += 1;
                } // RING_LOOP
                if exit_group {
                    break;
                }
                //
                // After each view group, increase the view base number
                ivbase += nin_group;
                *ierr = 0;
                igroup += 1;
            } // GROUP_LOOP
            if exit_num_view {
                break;
            }
            //
            // If we got here with err 0, this setup fits constraints, finalize index lists
            if *ierr == 0 {
                av.ip_start_wgt_view[(ind_view - 1) as usize] = ind_proj;
                av.iv_start_wgt_group[(ind_group - 1) as usize] = ind_view;
                av.num_wgt_groups = nring * num_view_groups;
                if av.leaving_out == 0 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "\nStarting robust fitting with%5d weight groups:%4d view groups in%3d \
                         rings\n",
                        &[
                            CArg::Int(av.num_wgt_groups as i64),
                            CArg::Int(num_view_groups as i64),
                            CArg::Int(nring as i64),
                        ],
                    ));
                }
                *ierr = 0;
                let _ = ImodFile::Stdout.flush();
                return;
            }
            num_views += 1;
        } // NUM_VIEW_LOOP
        nring -= 1;
    } // NUM_RING_LOOP
    *ierr = 1;
    let _ = ImodFile::Stdout.flush();
}

/// Original: `computeWeights` (`robustfit.cpp:201`) — computes the weights
/// given the current set of residuals and the weighting groups.  `distRes` and
/// `work` are temp arrays that need to be at least as big as the biggest view
/// group, and `iwork` needs to be as big as number of points on view.
///
/// `tiltalign.cpp:978` passes `int` arrays cast to `float *` as the two float
/// scratch arrays; they are written before they are read, so the caller may
/// hand over any `f32` scratch of the same length.
pub fn compute_weights(
    av: &mut AlignVariables,
    ind_all_real: &[i32],
    dist_res: &mut [f32],
    work: &mut [f32],
    iwork: &mut [i32],
) {
    let mut dev: f32;
    let mut rmedian: f32 = 0.;
    let mut r_madn: f32 = 0.;
    let mut adj_median: f32;
    let too_many_zero_delta: f32;
    let mut nin_group: i32;
    let mut ind: i32;
    let mut nin_view: i32;
    let mut num_low: i32 = 0;
    let mut max_small: i32;
    let min_non_zero: i32;
    let mut num_above: i32;
    let whole_track: bool;

    whole_track = av.patch_track_model != 0 && av.robust_by_track != 0;
    min_non_zero = 4;
    too_many_zero_delta = 0.02;
    //
    // Loop on the groups of views
    for igroup in 1..=av.num_wgt_groups {
        nin_group = 0;
        //
        // loop on the views in the group and get the residuals, divide by the smoothed
        // median residual
        let mut indv = av.iv_start_wgt_group[(igroup - 1) as usize];
        while indv <= av.iv_start_wgt_group[igroup as usize] - 1 {
            nin_view =
                av.ip_start_wgt_view[indv as usize] - av.ip_start_wgt_view[(indv - 1) as usize];
            for i in 1..=nin_view {
                ind = av.ind_proj_wgt_list
                    [(i + av.ip_start_wgt_view[(indv - 1) as usize] - 1 - 1) as usize];
                if whole_track {
                    //
                    // recompute mean residual for track
                    av.track_resid[(ind - 1) as usize] = 0.;
                    let mut j = av.ireal_str[(ind - 1) as usize];
                    while j <= av.ireal_str[ind as usize] - 1 {
                        if !(av.leaving_out != 0 && av.proj_left_out[(j - 1) as usize] != 0) {
                            let xr = av.xresid[(j - 1) as usize] as f64;
                            let yr = av.yresid[(j - 1) as usize] as f64;
                            av.track_resid[(ind - 1) as usize] = (av.track_resid[(ind - 1) as usize]
                                as f64
                                + (xr * xr + yr * yr).sqrt())
                                as f32;
                        }
                        j += 1;
                    }
                    av.track_resid[(ind - 1) as usize] /=
                        (av.ireal_str[ind as usize] - av.ireal_str[(ind - 1) as usize]) as f32;
                    dist_res[(nin_group + i - 1) as usize] = av.track_resid[(ind - 1) as usize]
                        / av.view_median_res[(av.itrack_group
                            [(ind_all_real[(ind - 1) as usize] - 1) as usize]
                            - 1) as usize];
                } else {
                    let xr = av.xresid[(ind - 1) as usize] as f64;
                    let yr = av.yresid[(ind - 1) as usize] as f64;
                    dist_res[(nin_group + i - 1) as usize] = ((xr * xr + yr * yr).sqrt()
                        / av.view_median_res[(av.isec_view[(ind - 1) as usize] - 1) as usize]
                            as f64)
                        as f32;
                }
            }
            nin_group += nin_view;
            indv += 1;
        }
        //
        // Get overall median and MADN and compute weights
        rs_fast_median(dist_res, nin_group, work, &mut rmedian);
        rs_fast_madn(dist_res, nin_group, rmedian, work, &mut r_madn);
        nin_group = 0;
        let mut indv = av.iv_start_wgt_group[(igroup - 1) as usize];
        while indv <= av.iv_start_wgt_group[igroup as usize] - 1 {
            nin_view =
                av.ip_start_wgt_view[indv as usize] - av.ip_start_wgt_view[(indv - 1) as usize];
            weights_for_view(
                av,
                rmedian,
                whole_track,
                nin_view,
                indv,
                dist_res,
                nin_group,
                r_madn,
                too_many_zero_delta,
                &mut num_low,
            );

            max_small = (nin_view as f32 * av.small_wgt_max_frac) as i32;
            if num_low > max_small {
                //
                // If there are too many small weights, then find the first one that must
                // be given a weight above threshold and adjust the median to accomplish
                // that
                for i in 1..=nin_view {
                    iwork[(i - 1) as usize] = i - 1;
                }
                rs_sort_indexed_floats(&dist_res[nin_group as usize..], iwork, nin_view);
                dev = (1. - (1.1 * av.small_wgt_threshold as f64).sqrt()).sqrt() as f32;
                adj_median = dist_res
                    [(nin_group + iwork[(nin_view - max_small - 1) as usize]) as usize]
                    - dev * av.kfac_robust * r_madn;
                weights_for_view(
                    av,
                    adj_median,
                    whole_track,
                    nin_view,
                    indv,
                    dist_res,
                    nin_group,
                    r_madn,
                    too_many_zero_delta,
                    &mut num_low,
                );
                if num_low > max_small {
                    let _ = ImodFile::Stdout
                        .write_all(b"WARNING: Median adjustment for small weights bad\n");
                }
            }
            nin_group += nin_view;
            indv += 1;
        }
    }
    if whole_track {
        return;
    }
    //
    // Now look at each real point and make sure it has enough non-zero weights, or
    // specifically points above the delta value; and if not, add delta to all weights
    for ind in 1..=av.nreal_pt {
        if (av.leaving_out != 0 && av.real_left_out[(ind - 1) as usize] != 0)
            || (!av.real_in_test_set.is_empty() && av.real_in_test_set[(ind - 1) as usize] != 0)
        {
            continue;
        }
        num_above = 0;
        for i in av.ireal_str[(ind - 1) as usize]..=av.ireal_str[ind as usize] - 1 {
            if !(av.leaving_out != 0 && av.proj_left_out[(i - 1) as usize] != 0)
                && av.weight[(i - 1) as usize] > too_many_zero_delta
            {
                num_above += 1;
            }
        }
        if num_above < min_non_zero {
            for i in av.ireal_str[(ind - 1) as usize]..=av.ireal_str[ind as usize] - 1 {
                if !(av.leaving_out != 0 && av.proj_left_out[(i - 1) as usize] != 0) {
                    // B3DMIN(1., weight + delta): a double comparison and choice.
                    let sum = (av.weight[(i - 1) as usize] + too_many_zero_delta) as f64;
                    av.weight[(i - 1) as usize] = (if 1. < sum { 1. } else { sum }) as f32;
                }
            }
        }
    }
}

/// Original: `weightsForView` (`robustfit.cpp:312`, file `static`) — compute
/// the weights for one view in a weight group, or for tracks in a track group.
#[allow(clippy::too_many_arguments)]
fn weights_for_view(
    av: &mut AlignVariables,
    view_median: f32,
    whole_track: bool,
    nin_view: i32,
    indv: i32,
    dist_res: &[f32],
    nin_group: i32,
    r_madn: f32,
    too_many_zero_delta: f32,
    num_low: &mut i32,
) {
    let mut track_wgt: f32;
    let mut dev: f32;
    let mut ind: i32;
    *num_low = 0;
    if whole_track {
        //
        // For whole track, loop on track, get one weight for each, make sure it is not
        // 0, and assign it to all projection points
        for i in 1..=nin_view {
            ind = av.ind_proj_wgt_list
                [(i + av.ip_start_wgt_view[(indv - 1) as usize] - 1 - 1) as usize];
            dev =
                (dist_res[(nin_group + i - 1) as usize] - view_median) / (av.kfac_robust * r_madn);
            if dev <= 0. {
                track_wgt = 1.;
            } else if dev >= 1. {
                track_wgt = too_many_zero_delta;
            } else {
                // B3DMAX(tooManyZeroDelta, pow(1. - dev * dev, 2)) in double.
                let base = 1. - (dev * dev) as f64;
                let p = base * base;
                track_wgt = (if too_many_zero_delta as f64 > p {
                    too_many_zero_delta as f64
                } else {
                    p
                }) as f32;
            }
            if track_wgt < av.small_wgt_threshold {
                *num_low += 1;
            }
            let mut jj = av.ireal_str[(ind - 1) as usize];
            while jj <= av.ireal_str[ind as usize] - 1 {
                if !(av.leaving_out != 0 && av.proj_left_out[(jj - 1) as usize] != 0) {
                    av.weight[(jj - 1) as usize] = track_wgt;
                }
                jj += 1;
            }
        }
    } else {
        //
        // Otherwise, loop on projection points in the view-weight group
        for i in 1..=nin_view {
            ind = av.ind_proj_wgt_list
                [(i + av.ip_start_wgt_view[(indv - 1) as usize] - 1 - 1) as usize];
            dev =
                (dist_res[(nin_group + i - 1) as usize] - view_median) / (av.kfac_robust * r_madn);
            if dev <= 0. {
                av.weight[(ind - 1) as usize] = 1.;
            } else if dev >= 1. {
                av.weight[(ind - 1) as usize] = 0.;
            } else {
                let base = 1. - (dev * dev) as f64;
                av.weight[(ind - 1) as usize] = (base * base) as f32;
            }
            if av.weight[(ind - 1) as usize] < av.small_wgt_threshold {
                *num_low += 1;
            }
        }
    }
}

/// Original: `setupTrackWeightGroups` (`robustfit.cpp:374`) — sets up
/// weighting that is the same for all points in a contour with patch track
/// data.
///  - `maxRings` = maximum number of rings allowed
///  - `minRes` = minimum number of tracks in the group
///  - `ierr` = return error value, 0 for success or 1 for failure
///
/// For weighting per contour, `ivStartWgtGroup` has the starting track group
/// index of the weight group; `ipStartWgtView` has the starting real point
/// index for each track group subset in each weight group; `indProjWgtList`
/// has the true real point number for each real point index.
pub fn setup_track_weight_groups(
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    max_rings: i32,
    min_res: i32,
    ind_all_real: &[i32],
    ierr: &mut i32,
) {
    let mut ireal_ring_list: Vec<i32> = vec![0; mx.max_real as usize];
    let mut list_track: Vec<i32> = vec![0; av.num_track_groups as usize];
    let mut dist_real: Vec<f32> = vec![0.; mx.max_real as usize];
    let mut num_view_groups: i32;
    let mut ind_proj: i32;
    let mut ind_group: i32;
    let mut ind_view: i32;
    let mut num_pairs: i32;
    let mut irbase: i32;
    let mut nin_ring: i32;
    let mut ireal: i32;
    let mut num_pts: i32;
    let mut max_combine: i32;
    let mut nin_list: i32;
    let mut num_real_used = 0;
    let mut exit_group: bool;

    // Get distances from center and get sorted indexes to them
    for i in 0..av.nreal_pt as usize {
        if !(av.leaving_out != 0 && av.real_left_out[i] != 0)
            && !(!av.real_in_test_set.is_empty() && av.real_in_test_set[i] != 0)
        {
            let x = av.xyz[i * 3] as f64;
            let y = av.xyz[i * 3 + 1] as f64;
            dist_real[num_real_used as usize] = (x * x + y * y).sqrt() as f32;
            ireal_ring_list[num_real_used as usize] = num_real_used;
            num_real_used += 1;
        }
    }
    rs_sort_indexed_floats(&dist_real, &mut ireal_ring_list, num_real_used);

    // Evaluate possibility of rings before any track group combine
    // NUM_RING_LOOP:
    let mut nring = max_rings;
    while nring >= 1 {
        *ierr = 1;
        max_combine = 1.max(2 * (av.num_track_groups / 2));
        if nring > 1 {
            if av.num_full_tracks_used / nring < min_res {
                nring -= 1;
                continue;
            }
            max_combine = 1;
        }

        // Combine track groups until we find a set that works with given # of rings
        num_view_groups = av.num_track_groups;
        // NUM_TGROUP_LOOP:
        for num_combine in 1..=max_combine {
            if num_combine > 1 {
                if (num_combine % 2) > 0 {
                    continue;
                }
                num_view_groups = max_combine / num_combine;
            }
            ind_group = 1;
            ind_proj = 1;
            ind_view = 1;

            // GROUP_LOOP:
            exit_group = false;
            let mut igroup = 1;
            while igroup <= num_view_groups && !exit_group {
                //
                // Make list of track groups in the view group
                if num_combine == 1 {
                    nin_list = 1;
                    list_track[0] = igroup;
                } else {
                    // Combine by pairs. If there are leftover pairs of groups, add them to
                    // last one
                    num_pairs = num_combine / 2;
                    if igroup == num_view_groups {
                        num_pairs += (max_combine / 2) % num_pairs;
                    }
                    for i in 1..=num_pairs {
                        list_track[(2 * i - 1 - 1) as usize] = (num_combine / 2) * (igroup - 1) + i;
                        list_track[(2 * i - 1) as usize] =
                            av.num_track_groups + 1 - list_track[(2 * i - 1 - 1) as usize];
                    }
                    nin_list = 2 * num_pairs;

                    // Add an odd one in the middle to the last group
                    if igroup == num_view_groups && av.num_track_groups % 2 > 0 {
                        nin_list += 1;
                        list_track[(nin_list - 1) as usize] = (av.num_track_groups + 1) / 2;
                    }
                }
                //
                // Loop on the rings, count up tracks actually in there
                irbase = 0;
                // RING_LOOP:
                for iring in 1..=nring {
                    nin_ring = num_real_used / nring;
                    if iring > nring - (num_real_used % nring) {
                        nin_ring += 1;
                    }

                    // This starts a weight group, save starting track group index
                    av.iv_start_wgt_group[(ind_group - 1) as usize] = ind_view;
                    ind_group += 1;
                    //
                    // Loop on the track groups in the weight group, for each one, set the
                    // starting index in the "projection" list
                    for iv in 1..=nin_list {
                        av.ip_start_wgt_view[(ind_view - 1) as usize] = ind_proj;
                        ind_view += 1;
                        // Loop on the tracks in the ring, find ones in this track group
                        for ind in irbase + 1..=irbase + nin_ring {
                            ireal = ireal_ring_list[(ind - 1) as usize] + 1;
                            if av.itrack_group[(ind_all_real[(ireal - 1) as usize] - 1) as usize]
                                == list_track[(iv - 1) as usize]
                            {
                                av.ind_proj_wgt_list[(ind_proj - 1) as usize] = ireal;
                                ind_proj += 1;
                            }
                        }
                    }

                    // After each ring, increase the ring base index
                    irbase += nin_ring;
                    num_pts = ind_proj
                        - av.ip_start_wgt_view
                            [(av.iv_start_wgt_group[(ind_group - 2) as usize] - 1) as usize];
                    if num_pts < min_res {
                        *ierr = 1;
                        exit_group = true;
                        break;
                    }
                } // RING_LOOP
                if exit_group {
                    break;
                }
                *ierr = 0;
                igroup += 1;
            } // GROUP_LOOP
            //
            // If we got here with err 0, this setup fits constraints, finalize index lists
            if *ierr == 0 {
                av.ip_start_wgt_view[(ind_view - 1) as usize] = ind_proj;
                av.iv_start_wgt_group[(ind_group - 1) as usize] = ind_view;
                av.num_wgt_groups = nring * num_view_groups;
                if av.leaving_out == 0 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "\nStarting robust fitting with%5d weight groups:%4d track groups in%3d \
                         rings\n",
                        &[
                            CArg::Int(av.num_wgt_groups as i64),
                            CArg::Int(num_view_groups as i64),
                            CArg::Int(nring as i64),
                        ],
                    ));
                }
                *ierr = 0;
                return;
            }
        } // NUM_TGROUP_LOOP
        nring -= 1;
    } // NUM_RING_LOOP
    *ierr = 1;
}
