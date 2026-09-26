//! Translation of `IMOD/flib/tiltalign/patchtrack.cpp` — routines for dealing
//! with patch track data.
//!
//! The file-scope `static AlignVariables *av;` (`patchtrack.cpp:16`), set by
//! `patchTrackSetPointers`, is a parameter here (`alivar.rs`).
//! `loadPatchSubset`'s `static bool firstTime` persists across calls and is
//! the process-global [`FIRST_TIME`].
//!
//! # Representation
//!
//! - `B3DMALLOC`ed scratch (`midpoints`, `checkFlags`, `processList`,
//!   `ninTemp`, `usedFullTrack`, `usedReal`) is a zeroed `Vec`
//!   (`NATIVE.md` §4); the source's allocation-failure tests see every array
//!   allocated.
//! - `makeFullTracksForInit` returns either `av`'s own `xx`/`yy`/`isecView`/
//!   `irealStr` (by storing the member pointers in its out-parameters) or four
//!   new arrays.  The translation fills the out-`Vec`s only in the second case
//!   and leaves them untouched in the first, where it returns `false`; a caller
//!   that gets `false` reads `av`'s arrays itself, which is what the source's
//!   aliased pointers are.  The new arrays' `malloc`-failure fallback to `av`'s
//!   arrays cannot occur (a `Vec` allocation failure aborts).
//! - `copyXYZfromFullTracksToReal` frees its two arrays; it takes them by
//!   value.  Its `copyArray` within `av->xyz` copies between two disjoint
//!   triples, so the slice is split at the later one.
//! - `restoreFromPatchSample`'s `afac` is `NULL` when the six factor arrays are
//!   not passed; it is an `Option`, and the other five are only read when it is
//!   `Some`.
//!
//! # Upstream defect fixed in translation (2026-09-26, `BUGS.md`)
//!
//! `loadPatchSubset` prints `"Need to add %d for view %d\n"` with the view
//! number and the count **swapped** (`patchtrack.cpp:326`); here the count
//! comes first, as the message reads.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use super::alivar::AlignVariables;
use super::solve_xyzd::{one_xyz_by_regression, regression_cross_products};
use super::utilfuncs::{allocate_patch_arrays, copy_array, count_num_in_view, memory_error};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::histogram::kernel_histogram;
use crate::imod::libcfshr::robuststat::rs_sort_indexed_floats;

/// Original: `static bool firstTime = true;` in `loadPatchSubset`
/// (`patchtrack.cpp:243`).
static FIRST_TIME: AtomicBool = AtomicBool::new(true);

/// Original: `patchTrackSetPointers` (`patchtrack.cpp:22`).
///
/// The source stores the pointer in a file-scope static; the functions of this
/// module take it as a parameter instead, so there is nothing to store.
pub fn patch_track_set_pointers(av_in: &mut AlignVariables) {
    let _ = av_in;
}

/// Original: `analyzePatchTracks` (`patchtrack.cpp:31`).
///
/// Sort the tracks into groups corresponding to an original full track and set
/// up various mappings.  `*numProjPtIn` is read into a local that nothing
/// uses.
pub fn analyze_patch_tracks(av: &mut AlignVariables, num_proj_pt_in: &i32) {
    let mut nin_temp: Vec<i32>;
    let mut midpoints: Vec<f32>;
    let mut check_flags: Vec<i32>;
    let mut process_list: Vec<i32>;
    let mut avg_len: f32;
    let mut diff_max: f32;
    let _num_proj_pt = *num_proj_pt_in;
    let mut len: i32;
    let mut max_len: i32;
    let mut num_peaks: i32;
    let mut last_real_done: i32;
    let mut num_on_list: i32;
    let mut list_cur: i32;
    let mut ind_min: i32 = 0;
    let mut ind_max: i32 = 0;
    let mut idir: i32 = 0;
    let mut ireal: i32;
    let mut min_rview: i32;
    let mut max_rview: i32;
    let mut i: i32;
    let mut iview: i32;
    let mut found: bool;

    len = av.nreal_pt + 4;
    allocate_patch_arrays(av, len, 0);
    midpoints = vec![0.; len as usize];
    check_flags = vec![0; av.nfile_views as usize];
    process_list = vec![0; av.nfile_views as usize];
    memory_error(true, "temporary arrays for patch track maps");
    //
    // Build up mapRealToTrack.  Use processList for list of ones to process, and
    // checkFlags for flag of which ends need checking
    av.num_full_patch_tracks = 0;
    for i in 0..len as usize {
        av.map_real_to_track[i] = 0;
    }
    last_real_done = -1; // Indexes from 0 for all real pt #s in this loop
    while last_real_done < av.nreal_pt - 1 {
        //
        // Find the next point that is not mapped yet
        while last_real_done < av.nreal_pt - 1 {
            last_real_done += 1;
            if av.map_real_to_track[last_real_done as usize] == 0 {
                break;
            }
        }
        if av.map_real_to_track[last_real_done as usize] > 0 {
            break;
        }
        //
        // Start a new track and the list with the next real point, both ends need checking
        av.num_full_patch_tracks += 1;
        av.map_real_to_track[last_real_done as usize] = av.num_full_patch_tracks;
        process_list[0] = last_real_done;
        check_flags[0] = 3;
        num_on_list = 1;
        list_cur = 0;
        //
        // process next point on list: find first and last view
        while list_cur < num_on_list {
            min_rview = 100000;
            max_rview = -100000;
            ireal = process_list[list_cur as usize];
            i = av.ireal_str[ireal as usize] - 1;
            while i < av.ireal_str[(ireal + 1) as usize] - 1 {
                if av.isec_view[i as usize] < min_rview {
                    ind_min = i;
                    min_rview = av.isec_view[i as usize];
                }
                if av.isec_view[i as usize] > max_rview {
                    ind_max = i;
                    max_rview = av.isec_view[i as usize];
                }
                i += 1;
            }
            //
            // Check higher end first, then substitute lower end and check it
            for idir in 1..=2 {
                if (idir == 1 && check_flags[list_cur as usize] > 1)
                    || (idir == 2 && (check_flags[list_cur as usize] % 2) > 0)
                {
                    found = false;

                    // LOOP_REAL:
                    let mut i = last_real_done + 1;
                    while i < av.nreal_pt && !found {
                        if av.map_real_to_track[i as usize] == 0 {
                            let mut j = av.ireal_str[i as usize] - 1;
                            while j < av.ireal_str[(i + 1) as usize] - 1 {
                                let dx = av.xx[ind_max as usize] - av.xx[j as usize];
                                let dy = av.yy[ind_max as usize] - av.yy[j as usize];
                                if av.isec_view[j as usize] == max_rview
                                    && (if dx >= 0. { dx } else { -dx }) < 0.1
                                    && (if dy >= 0. { dy } else { -dy }) < 0.1
                                {
                                    //
                                    // Found the adjacent one in the track, add it to the list
                                    av.map_real_to_track[i as usize] = av.num_full_patch_tracks;
                                    check_flags[num_on_list as usize] = 3 - idir;
                                    process_list[num_on_list as usize] = i;
                                    num_on_list += 1;
                                    found = true;
                                    break;
                                }
                                j += 1;
                            }
                        }
                        i += 1;
                    } // LOOP_REAL
                }

                // Set up for other direction
                max_rview = min_rview;
                ind_max = ind_min;
            }
            //
            // Done with this one, advance on list
            list_cur += 1;
        }
    }
    //
    // Now find out how many in each track and set up indexes
    len = av.num_full_patch_tracks + 4;
    allocate_patch_arrays(av, len, 1);
    nin_temp = vec![0; av.num_full_patch_tracks as usize];
    memory_error(true, "temp array for patch track indices");
    for v in nin_temp.iter_mut() {
        *v = 0;
    }
    avg_len = 0.;
    max_len = 0;
    for j in 0..av.nreal_pt as usize {
        i = av.map_real_to_track[j] - 1;
        nin_temp[i as usize] += 1;
        len = av.isec_view[(av.ireal_str[j + 1] - 2) as usize] + 1
            - av.isec_view[(av.ireal_str[j] - 1) as usize];
        midpoints[j] = ((av.isec_view[(av.ireal_str[j] - 1) as usize]
            + av.isec_view[(av.ireal_str[j + 1] - 2) as usize]) as f64
            / 2.) as f32;
        avg_len = avg_len + len as f32;
        max_len = if max_len > len { max_len } else { len };
    }
    avg_len = avg_len / av.nreal_pt as f32;
    av.ind_full_track[0] = 1;
    for j in 0..av.num_full_patch_tracks as usize {
        av.ind_full_track[j + 1] = av.ind_full_track[j] + nin_temp[j];
    }
    //
    // Then populate the map using the indices
    for v in nin_temp.iter_mut() {
        *v = 0;
    }
    for j in 0..av.nreal_pt {
        i = av.map_real_to_track[j as usize] - 1;
        av.map_track_to_real[(av.ind_full_track[i as usize] + nin_temp[i as usize] - 1) as usize] =
            j + 1;
        nin_temp[i as usize] += 1;
    }
    //
    // Get a kernel histogram of the midpoints and find peaks in it
    kernel_histogram(
        &midpoints[..av.nreal_pt as usize],
        &mut av.dmag[..av.nview as usize],
        0.5,
        (av.nview as f64 + 0.5) as f32,
        (avg_len as f64 / 8.) as f32,
        0,
    );
    num_peaks = 0;
    av.lin_alf[0] = 0;
    let nv = av.nview;
    for i in 1..nv - 1 {
        let iu = i as usize;
        iview = i + 1;
        //
        // If point is a peak, check whether it is within 1/4 of maximum length of last one,
        // and if so take the stronger of the two
        // Take first of a pair of equal points (this can happen)
        let dm = &av.dmag;
        if (dm[iu] > dm[iu - 1] && dm[iu] > dm[iu + 1])
            || (dm[iu] > dm[iu - 1]
                && dm[iu] == dm[iu + 1]
                && dm[iu] > dm[((if nv < i + 3 { nv } else { i + 3 }) - 1) as usize])
        {
            if num_peaks > 0
                && ((iview - av.lin_alf[((if 1 > num_peaks { 1 } else { num_peaks }) - 1) as usize])
                    as f64)
                    < max_len as f64 / 4.
            {
                if av.dmag[iu] > av.dmag[(av.lin_alf[(num_peaks - 1) as usize] - 1) as usize] {
                    av.lin_alf[(num_peaks - 1) as usize] = iview;
                }
            } else {
                av.lin_alf[num_peaks as usize] = iview;
                num_peaks += 1;
            }
        }
    }
    //
    // And if somehow that failed to find any peaks, simply say there is one in the middle
    if num_peaks == 0 {
        num_peaks = 1;
        av.lin_alf[0] = (av.nview as f64 / 2.) as i32;
    }
    //
    // Put each track in a group based on which peak its midpoint is nearest
    for j in 0..av.nreal_pt as usize {
        diff_max = 1.0e10;
        for i in 0..num_peaks {
            let d = midpoints[j] - av.lin_alf[i as usize] as f32;
            let ad = if d >= 0. { d } else { -d };
            if diff_max > ad {
                diff_max = ad;
                idir = i + 1;
            }
        }
        av.itrack_group[j] = idir;
    }
    av.num_track_groups = num_peaks;

    /*printf("%d full tracks\n", av->numFullPatchTracks);
    for (j = 0; j < av->numFullPatchTracks; j++) {
      printf("%5d", j);
      int numOut = 1;
      for (i = av->indFullTrack[j]; i < av->indFullTrack[j + 1]; i++) {
        printf("%5d", av->mapTrackToReal[i - 1]);
        numOut++;
        if (numOut % 15 == 0)
          printf("\n");
      }
      printf("\n");
      }*/
    av.num_full_tracks_used = av.num_full_patch_tracks;
}

/// Original: `loadPatchSubset` (`patchtrack.cpp:231`).
///
/// Selects a subset of tracks to avoid trying to solve for a global solution
/// with a huge number of tracks.  All tracks in a full track are used.
/// `numTarget` is the target number of full tracks to select, and
/// `minInView` is the minimum number of points in each view.
///
/// The `bool *` scratch arrays are `Vec<bool>`.  An `ifull` past the last full
/// track (reachable in the source only through `B3DNINT` rounding at the end of
/// a gap) indexes past `usedFullTrack` there and panics here.
#[allow(clippy::too_many_arguments)]
pub fn load_patch_subset(
    av: &mut AlignVariables,
    allxx: &mut [f32],
    allyy: &mut [f32],
    n_all_proj_pt: &mut i32,
    nproj_pt: &mut i32,
    ind_all_real: &mut [i32],
    n_all_real_pt: &mut i32,
    iall_real_str: &mut [i32],
    iall_secv: &mut [i32],
    all_imod_obj: &[i32],
    num_in_view: &mut [i32],
    num_target: i32,
    min_in_view: i32,
) {
    let mut used_full_track: Vec<bool>;
    let mut used_real: Vec<bool>;
    let mut gap_size: Vec<f32>;
    let mut ind_gap_start: Vec<i32>;
    let mut ind_sort: Vec<i32>;
    let mut num_used_full: i32;
    let len_twice_avg: i32;
    let mut num: i32;
    let len_very_long: i32;
    let mut last_used: i32;
    let mut num_gaps: i32;
    let mut ifull: i32;
    let mut isort: i32;
    let mut num_fill: i32;
    let mut midpoint: i32;
    let mut ireal: i32;
    let density: f32;
    let mut found: bool;
    let mut stdout = ImodFile::Stdout;
    //
    used_full_track = vec![false; av.num_full_patch_tracks as usize];
    used_real = vec![false; av.nreal_pt as usize];
    memory_error(true, "arrays in loadPatchSubset");
    for i in 0..av.num_full_patch_tracks as usize {
        used_full_track[i] = false;
    }
    for i in 0..av.nreal_pt as usize {
        used_real[i] = false;
    }
    ind_sort = vec![0; av.nreal_pt as usize];
    ind_gap_start = vec![0; av.num_full_patch_tracks as usize];
    gap_size = vec![0.; av.num_full_patch_tracks as usize];
    //
    // First save all the data
    *n_all_real_pt = av.nreal_pt;
    *n_all_proj_pt = *nproj_pt;
    copy_array(allxx, 1, *nproj_pt, &av.xx, 1);
    copy_array(allyy, 1, *nproj_pt, &av.yy, 1);
    copy_array(iall_real_str, 1, av.nreal_pt + 1, &av.ireal_str, 1);
    copy_array(iall_secv, 1, *nproj_pt, &av.isec_view, 1);
    //
    av.nreal_pt = 0;
    *nproj_pt = 0;
    num_used_full = 0;
    density = num_target as f32 / av.num_full_patch_tracks as f32;
    //
    // Include any track that is 2x longer than average of at least 0.8 * full set of views
    len_twice_avg = ((2. * *n_all_proj_pt as f64) / *n_all_real_pt as f64 + 0.5).floor() as i32;
    len_very_long = (0.8 * av.nview as f64 + 0.5).floor() as i32;
    //printf("Adding long tracks if any\n");
    for ireal in 0..*n_all_real_pt {
        num = iall_real_str[(ireal + 1) as usize] - iall_real_str[ireal as usize];
        if !used_real[ireal as usize] && (num >= len_twice_avg || num >= len_very_long) {
            add_full_track(
                av,
                av.map_real_to_track[ireal as usize],
                &mut used_full_track,
                &mut num_used_full,
                ind_all_real,
                &mut used_real,
                iall_real_str,
                nproj_pt,
                allxx,
                allyy,
                iall_secv,
                all_imod_obj,
            );
        }
    }
    //
    // Make a list of all gaps between used tracks and sort it
    // Need 0-based indexes for C call to rsSortIndexedFloats
    last_used = 1;
    num_gaps = 0;
    for ifull in 2..=av.num_full_patch_tracks {
        if used_full_track[(ifull - 1) as usize] || ifull == av.num_full_patch_tracks {
            ind_gap_start[num_gaps as usize] = last_used;
            ind_sort[num_gaps as usize] = num_gaps;
            gap_size[num_gaps as usize] = (ifull - last_used) as f32;
            num_gaps += 1;
            last_used = ifull;
        }
    }
    if num_gaps > 1 {
        rs_sort_indexed_floats(&gap_size, &mut ind_sort, num_gaps);
    }
    //printf("Number of gaps %d\n", numGaps);
    //
    // Fill in gaps from the biggest down, using evenly spaced selections to achieve
    // desired density
    isort = num_gaps;
    while isort >= 1 && num_used_full < num_target {
        num_fill = ((density * gap_size[ind_sort[(isort - 1) as usize] as usize]) as f64 + 0.5)
            .floor() as i32
            - 1;
        if num_fill == 0 {
            break;
        }
        for i in 1..=num_fill {
            ifull = ((i as f32 / density) as f64 + 0.5).floor() as i32
                + ind_gap_start[ind_sort[(isort - 1) as usize] as usize];
            if !used_full_track[(ifull - 1) as usize] {
                add_full_track(
                    av,
                    ifull,
                    &mut used_full_track,
                    &mut num_used_full,
                    ind_all_real,
                    &mut used_real,
                    iall_real_str,
                    nproj_pt,
                    allxx,
                    allyy,
                    iall_secv,
                    all_imod_obj,
                );
            }
        }
        isort -= 1;
    }
    // print *,'Checking enough in each view'
    //
    // Now the worry is that there may not be a minimum number per section
    // Need 1-based indexes to give countNumInView as long as it is Fortran
    for i in 0..av.nreal_pt {
        ind_sort[i as usize] = i + 1;
    }
    count_num_in_view(
        av,
        &ind_sort,
        av.nreal_pt,
        &av.ireal_str,
        &av.isec_view,
        av.nview,
        num_in_view,
        (!av.real_in_test_set.is_empty()).then_some(&av.real_in_test_set[..]),
        None,
    );
    for iv in 1..=av.nview {
        if num_in_view[(iv - 1) as usize] < min_in_view {
            num_fill = min_in_view - num_in_view[(iv - 1) as usize];
            let _ = stdout.write_all(&c_format_bytes(
                "Need to add %d for view %d\n",
                &[CArg::Int(num_fill as i64), CArg::Int(iv as i64)],
            ));
            //
            // Go to the midpoint of equally spaced segments and find nearest not used with
            // point on view
            for iseg in 1..=num_fill {
                midpoint = (((iseg as f64 - 0.5) * av.num_full_patch_tracks as f64)
                    / num_fill as f64
                    + 0.5)
                    .floor() as i32;
                found = false;
                // print *,'Finding nearest to', midpoint
                // FIND_NEAREST:
                let mut idiff = 0;
                while idiff <= av.num_full_patch_tracks && !found {
                    let mut idir = -1;
                    while idir <= 1 && !found {
                        ifull = midpoint + idir * idiff;
                        if ifull >= 1
                            && ifull <= av.num_full_patch_tracks
                            && !used_full_track[(ifull - 1) as usize]
                        {
                            let mut j = av.ind_full_track[(ifull - 1) as usize];
                            while j <= av.ind_full_track[ifull as usize] - 1 && !found {
                                ireal = av.map_track_to_real[(j - 1) as usize];
                                let mut i = iall_real_str[(ireal - 1) as usize];
                                while i <= iall_real_str[ireal as usize] - 1 {
                                    if iall_secv[(i - 1) as usize] == iv {
                                        add_full_track(
                                            av,
                                            ifull,
                                            &mut used_full_track,
                                            &mut num_used_full,
                                            ind_all_real,
                                            &mut used_real,
                                            iall_real_str,
                                            nproj_pt,
                                            allxx,
                                            allyy,
                                            iall_secv,
                                            all_imod_obj,
                                        );
                                        found = true;
                                        break; // exit FIND_NEAREST;
                                    }
                                    i += 1;
                                }
                                j += 1;
                            }
                        }
                        idir += 2;
                    }
                    idiff += 1;
                } // FIND_NEAREST
            }
            for i in 0..av.nreal_pt {
                ind_sort[i as usize] = i + 1;
            }
            count_num_in_view(
                av,
                &ind_sort,
                av.nreal_pt,
                &av.ireal_str,
                &av.isec_view,
                av.nview,
                num_in_view,
                (!av.real_in_test_set.is_empty()).then_some(&av.real_in_test_set[..]),
                None,
            );
        }
    }
    if FIRST_TIME.load(Ordering::Relaxed) {
        let _ = stdout.write_all(&c_format_bytes(
            "Selected %d of %d full tracks, %d of %d track segments\n",
            &[
                CArg::Int(num_used_full as i64),
                CArg::Int(av.num_full_patch_tracks as i64),
                CArg::Int(av.nreal_pt as i64),
                CArg::Int(*n_all_real_pt as i64),
            ],
        ));
        let _ = stdout.flush();
    }
    FIRST_TIME.store(false, Ordering::Relaxed);
    av.ireal_str[av.nreal_pt as usize] = *nproj_pt + 1;
    av.num_full_tracks_used = num_used_full;
}

/// Original: `addFullTrack` (`patchtrack.cpp:379`, file static).
///
/// Adds all the individual tracks in a full track.
#[allow(clippy::too_many_arguments)]
fn add_full_track(
    av: &mut AlignVariables,
    ind_full: i32,
    used_full_track: &mut [bool],
    num_used_full: &mut i32,
    ind_all_real: &mut [i32],
    used_real: &mut [bool],
    iall_real_str: &[i32],
    nproj_pt: &mut i32,
    allxx: &[f32],
    allyy: &[f32],
    iall_secv: &[i32],
    all_imod_obj: &[i32],
) {
    let mut ind_real: i32;
    used_full_track[(ind_full - 1) as usize] = true;
    *num_used_full = *num_used_full + 1;
    //printf("Adding full track %d\n",indFull);
    let mut j = av.ind_full_track[(ind_full - 1) as usize];
    while j <= av.ind_full_track[ind_full as usize] - 1 {
        ind_real = av.map_track_to_real[(j - 1) as usize];
        //printf("indAll %d  %d\n", av->nrealPt, indReal);
        ind_all_real[av.nreal_pt as usize] = ind_real;
        av.ireal_str[av.nreal_pt as usize] = *nproj_pt + 1;
        used_real[av.nreal_pt as usize] = true;
        if av.apply_extra_weights != 0 {
            av.imod_obj_num[av.nreal_pt as usize] = all_imod_obj[(ind_real - 1) as usize];
        }
        av.nreal_pt += 1;
        for i in
            (iall_real_str[(ind_real - 1) as usize] - 1)..(iall_real_str[ind_real as usize] - 1)
        {
            let iu = i as usize;
            av.xx[*nproj_pt as usize] = allxx[iu];
            av.yy[*nproj_pt as usize] = allyy[iu];
            av.isec_view[*nproj_pt as usize] = iall_secv[iu];
            *nproj_pt += 1;
            //printf("%.1f %.1f %d\n", allxx[i - 1], allyy[i - 1], iallSecv[i - 1]);
        }
        j += 1;
    }
}

/// Original: `restoreFromPatchSample` (`patchtrack.cpp:411`).
///
/// Restores the data arrays from doing a subset of tracks and solves for XYZ
/// for the remainder of the real points.  If `afac` is `NULL` (`None`), it
/// assumes that `allXYZ` has been restored to a desired state already and does
/// not recompute it; `bfac`..`ffac` are then not read.
#[allow(clippy::too_many_arguments)]
pub fn restore_from_patch_sample(
    av: &mut AlignVariables,
    allxx: &[f32],
    allyy: &[f32],
    n_all_proj_pt: i32,
    nproj_pt: &mut i32,
    all_xyz: &mut [f32],
    ind_all_real: &mut [i32],
    n_all_real_pt: i32,
    iall_real_str: &[i32],
    iall_secv: &[i32],
    all_imod_obj: &[i32],
    afac: Option<&[f32]>,
    bfac: &[f32],
    cfac: &[f32],
    dfac: &[f32],
    efac: &[f32],
    ffac: &[f32],
) {
    let mut asq: Vec<f32> = Vec::new();
    let mut bsq: Vec<f32> = Vec::new();
    let mut csq: Vec<f32> = Vec::new();
    let mut axb: Vec<f32> = Vec::new();
    let mut axc: Vec<f32> = Vec::new();
    let mut bxc: Vec<f32> = Vec::new();
    let mut in_sample: Vec<bool> = Vec::new();
    let mut ind: i32;
    if let Some(afac) = afac {
        asq.resize(av.nview as usize, 0.);
        bsq.resize(av.nview as usize, 0.);
        csq.resize(av.nview as usize, 0.);
        axb.resize(av.nview as usize, 0.);
        axc.resize(av.nview as usize, 0.);
        bxc.resize(av.nview as usize, 0.);
        //
        // Copy the xyz's into the larger array and mark them in the sample array
        in_sample.resize(n_all_real_pt as usize, false);
        for i in 1..=av.nreal_pt {
            ind = ind_all_real[(i - 1) as usize];
            in_sample[(ind - 1) as usize] = true;
            copy_array(
                &mut all_xyz[((ind - 1) * 3) as usize..],
                1,
                3,
                &av.xyz[((i - 1) * 3) as usize..],
                1,
            );
        }
        //
        // Get the cross-products then solve for xyz for the rest of the points
        regression_cross_products(
            av.nview, afac, bfac, cfac, dfac, efac, ffac, &mut asq, &mut bsq, &mut csq, &mut axb,
            &mut axc, &mut bxc,
        );
        for i in 1..=n_all_real_pt {
            if !in_sample[(i - 1) as usize] {
                one_xyz_by_regression(
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                    &asq,
                    &bsq,
                    &csq,
                    &axb,
                    &axc,
                    &bxc,
                    &av.dxy,
                    allxx,
                    allyy,
                    iall_real_str[(i - 1) as usize],
                    iall_real_str[i as usize] - 1,
                    iall_secv,
                    &mut all_xyz[(3 * (i - 1)) as usize..],
                );
            }
        }
        let _ = ImodFile::Stdout.flush();
    } else {
        for i in 1..=n_all_real_pt {
            ind_all_real[(i - 1) as usize] = i;
        }
    }
    //
    // Restore all the data
    av.nreal_pt = n_all_real_pt;
    *nproj_pt = n_all_proj_pt;
    if av.apply_extra_weights != 0 {
        copy_array(&mut av.imod_obj_num, 1, av.nreal_pt + 1, all_imod_obj, 1);
    }
    for i in 1..=av.nreal_pt {
        copy_array(
            &mut av.xyz[((i - 1) * 3) as usize..],
            1,
            3,
            &all_xyz[((i - 1) * 3) as usize..],
            1,
        );
    }
    copy_array(&mut av.xx, 1, *nproj_pt, allxx, 1);
    copy_array(&mut av.yy, 1, *nproj_pt, allyy, 1);
    copy_array(&mut av.ireal_str, 1, av.nreal_pt + 1, iall_real_str, 1);
    copy_array(&mut av.isec_view, 1, *nproj_pt, iall_secv, 1);
}

/// Original: `makeFullTracksForInit` (`patchtrack.cpp:472`).
///
/// Composes full tracks out of chopped up ones and returns substitute arrays
/// of x, y, sec, and real start and new # of real points.  Also returns arrays
/// that make it easy to propagate the X/Y/Z values for actal tracks.  This was
/// needed during initialization after adding some tracked points to chopped up
/// tracks.
///
/// Returns `false` with `nrealPt = av->nrealPt` and the six out-`Vec`s left
/// untouched when the source would hand back `av`'s own arrays (module doc); the
/// caller then uses `av.xx`, `av.yy`, `av.isec_view` and `av.ireal_str`.
#[allow(clippy::too_many_arguments)]
pub fn make_full_tracks_for_init(
    av: &AlignVariables,
    xx_p: &mut Vec<f32>,
    yy_p: &mut Vec<f32>,
    isec_view_p: &mut Vec<i32>,
    ireal_str_p: &mut Vec<i32>,
    nreal_pt: &mut i32,
    ind_all_real: &[i32],
    sub_sampled: bool,
    used_full_to_real_p: &mut Vec<i32>,
    ind_used_to_real_p: &mut Vec<i32>,
) -> bool {
    let tot_pts = av.ireal_str[av.nreal_pt as usize] - 1;
    let mut ireal: i32;
    let mut k: i32;
    let mut ind_next_pt: i32;
    let mut num_used_full: i32;
    let mut num_used_real: i32;
    let mut already: bool;
    let mut in_subset: bool;
    let mut used_full: bool;
    let use_av_arrs = av.patch_track_model == 0 || av.num_full_patch_tracks == av.nreal_pt;

    // Use existing variables and return false
    if use_av_arrs {
        *nreal_pt = av.nreal_pt;
        return false;
    }

    // If not using existing variables, try to create array, and just forget it if it fails
    let mut xx: Vec<f32> = vec![0.; (tot_pts + 10) as usize];
    let mut yy: Vec<f32> = vec![0.; (tot_pts + 10) as usize];
    let mut isec_view: Vec<i32> = vec![0; (tot_pts + 10) as usize];
    let mut ireal_str: Vec<i32> = vec![0; (av.num_full_patch_tracks + 10) as usize];
    let mut used_full_to_real: Vec<i32> = vec![0; (av.nreal_pt + 10) as usize];
    let mut ind_used_to_real: Vec<i32> = vec![0; (av.nreal_pt + 10) as usize];

    // Set up indexes
    ireal_str[0] = 1;
    ind_next_pt = 0;
    num_used_full = 0;
    num_used_real = 0;
    ind_used_to_real[0] = 0;

    // Loop on the full tracks
    for track in 0..av.num_full_patch_tracks as usize {
        used_full = false;

        // Loop on the real tracks comprising a full track (ireal #'d from 1)
        for ind in av.ind_full_track[track]..av.ind_full_track[track + 1] {
            ireal = av.map_track_to_real[(ind - 1) as usize];
            in_subset = !sub_sampled;

            // If a subset was sampled, make sure the real track is in the subset
            // indAllReal is also #'d from 1, so make a 0-based index to save for usedFullToReal
            if in_subset {
                k = ireal - 1;
            } else {
                k = 0;
                while k < av.nreal_pt {
                    if ind_all_real[k as usize] == ireal {
                        in_subset = true;
                        break;
                    }
                    k += 1;
                }
            }

            // Loop on the points in the track, check each one with points already added to the
            // full track
            if in_subset {
                //PRINT3(ireal, numUsedReal, k);
                used_full_to_real[num_used_real as usize] = k;
                num_used_real += 1;
                for j in (av.ireal_str[k as usize] - 1)..(av.ireal_str[(k + 1) as usize] - 1) {
                    let ju = j as usize;
                    already = false;
                    for i in (ireal_str[num_used_full as usize] - 1)..ind_next_pt {
                        if av.isec_view[ju] == isec_view[i as usize] {
                            already = true;
                            break;
                        }
                    }

                    // Add unique point
                    if !already {
                        xx[ind_next_pt as usize] = av.xx[ju];
                        yy[ind_next_pt as usize] = av.yy[ju];
                        //printf("%d %.4f %.4f %d\n", numUsedFull, xx[indNextPt] , yy[indNextPt],
                        //     av->isecView[j]);
                        isec_view[ind_next_pt as usize] = av.isec_view[ju];
                        ind_next_pt += 1;
                    }
                }
                used_full = true;
            }
        }
        if used_full {
            num_used_full += 1;
            ireal_str[num_used_full as usize] = ind_next_pt + 1;
            ind_used_to_real[num_used_full as usize] = num_used_real;
            //PRINT3(track, numUsedFull, numUsedReal);
        }
    }

    // Return arrays
    *nreal_pt = num_used_full;
    *xx_p = xx;
    *yy_p = yy;
    *isec_view_p = isec_view;
    *ireal_str_p = ireal_str;
    *used_full_to_real_p = used_full_to_real;
    *ind_used_to_real_p = ind_used_to_real;
    true
}

/// Original: `copyXYZfromFullTracksToReal` (`patchtrack.cpp:588`).
///
/// After running initialization routine on full tracks, this is called to copy
/// the X/Y/Z for each full track into the spots for the real tracks comprising
/// it.  Takes ownership of (the source frees) both arrays.
pub fn copy_xyz_from_full_tracks_to_real(
    av: &mut AlignVariables,
    nreal_pt: i32,
    used_full_to_real: Vec<i32>,
    ind_used_to_real: Vec<i32>,
) {
    let mut ireal: i32;
    //for (j = 0; j < nrealPt; j++)
    //printf("%d %.4f %.4f %.4f\n", j, av->xyz[j*3], av->xyz[j*3 + 1], av->xyz[j*3 + 2]);
    let mut ifull = nreal_pt - 1;
    while ifull >= 0 {
        let mut j = ind_used_to_real[(ifull + 1) as usize] - 1;
        while j >= ind_used_to_real[ifull as usize] {
            ireal = used_full_to_real[j as usize];
            //printf("%d %d %d\n", ifull, j, ireal);
            if ireal != ifull {
                // copyArray(&av->xyz[ireal * 3], 1, 3, &av->xyz[ifull * 3], 1): two
                // disjoint triples of one array.
                let (to, from) = (ireal * 3, ifull * 3);
                if to > from {
                    let (lo, hi) = av.xyz.split_at_mut(to as usize);
                    copy_array(hi, 1, 3, &lo[from as usize..], 1);
                } else {
                    let (lo, hi) = av.xyz.split_at_mut(from as usize);
                    copy_array(&mut lo[to as usize..], 1, 3, hi, 1);
                }
            }
            j -= 1;
        }
        ifull -= 1;
    }
    //for (j = 0; j < av->nrealPt; j++)
    //printf("%d %.4f %.4f %.4f\n", j, av->xyz[j*3], av->xyz[j*3 + 1], av->xyz[j*3 + 2]);
    drop(ind_used_to_real);
    drop(used_full_to_real);
}
