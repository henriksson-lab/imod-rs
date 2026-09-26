//! Translation of `IMOD/flib/tiltalign/utilfuncs.cpp` — utility and allocation
//! functions.
//!
//! # One source, two builds
//!
//! `tiltalign` compiles this unit plain and `beadtrack` compiles it with
//! `-DBEADTRACK` (`flib/beadtrack/Makefile:40-41`).  The only conditional code
//! is the `#ifndef BEADTRACK` arm of `errorExit` (`utilfuncs.cpp:23-28`), so
//! the build choice is the `const BEADTRACK: bool` generic parameter of
//! [`error_exit`]: a `tiltalign` unit calls `error_exit::<false>`, a unit
//! compiled into `beadtrack` calls `error_exit::<true>`.  A caller that is
//! itself compiled into both programs (`solve_xyzd.cpp`, `map_vars.cpp`) takes
//! the same `const BEADTRACK: bool` and forwards it.  Everything else in the
//! unit is identical in both builds and has one function.
//!
//! # File-scope statics
//!
//! `static AlignVariables *av;` and `static ArrayMaxes *mx;`
//! (`utilfuncs.cpp:17-18`) are set by `allocateAlivar` and read by
//! `allocatePatchArrays` and `countNumInView`.  Per `alivar.rs` they are
//! parameters: `&mut AlignVariables` for the allocators, `&AlignVariables` for
//! `count_num_in_view` (which reads only `testSetFracStep`).  The file-scope
//! `mx` is written and never read in this unit.
//!
//! # Allocation
//!
//! `B3DMALLOC` becomes a zero-filled `Vec` of the same length, and the
//! allocation-failure tests (`memoryError`, the `ierr` expressions) see every
//! array as allocated: a `Vec` allocation failure aborts instead of returning
//! `NULL`.  The source's arrays are uninitialised; a `Vec` is zeroed
//! (`NATIVE.md` §4).

use std::io::Write;

use super::alivar::AlignVariables;
use super::arraymaxes::{ArrayMaxes, MAX_WGT_RINGS, MAXGRP};
use super::mapsepgroups::MapSepGroups;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::parse_params::exit_error;

/// Original: `errorExit` (`utilfuncs.cpp:21`).
///
/// `BEADTRACK` selects the build: `false` is `tiltalign`'s, where a non-zero
/// `ifLocal` prints a warning and returns; `true` is `beadtrack`'s
/// `-DBEADTRACK` build, where the `#ifndef BEADTRACK` block is compiled out
/// and every call exits.  The source passes `cstr` to `exitError` as its
/// *format*; no caller's message contains a `%`, and `exit_error` takes the
/// formatted message.
pub fn error_exit<const BEADTRACK: bool>(cstr: &str, if_local: i32) {
    if !BEADTRACK && if_local != 0 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes("\nWARNING: %s\n", &[CArg::Str(cstr)]));
        return;
    }
    exit_error(cstr.as_bytes());
}

/// Original: `allocatePatchArrays` (`utilfuncs.cpp:32`).
pub fn allocate_patch_arrays(av: &mut AlignVariables, len: i32, iwhich: i32) {
    if iwhich == 0 {
        av.map_real_to_track = vec![0; len as usize];
        av.map_track_to_real = vec![0; len as usize];
        av.itrack_group = vec![0; len as usize];
        av.track_resid = vec![0.; len as usize];

        memory_error(true, "arrays for patch track maps");
    } else {
        av.ind_full_track = vec![0; len as usize];
        memory_error(true, "array for patch track indices");
    }
}

/// Original: `allocateAlivar` (`utilfuncs.cpp:50`).
///
/// The source also stores `avIn`/`mxIn` in the file-scope pointers; here the
/// later calls take them as parameters (module doc).
pub fn allocate_alivar(
    av: &mut AlignVariables,
    mx: &mut ArrayMaxes,
    num_proj_pt: i32,
    num_view: i32,
    num_real: i32,
    ierr: &mut i32,
) {
    mx.max_proj_pt = num_proj_pt + 10;
    mx.max_view = num_view + 4;
    mx.max_real = num_real + 4;
    let max_proj_pt = mx.max_proj_pt as usize;
    let max_view = mx.max_view as usize;
    let max_real = mx.max_real as usize;
    av.xx = vec![0.; max_proj_pt];
    av.yy = vec![0.; max_proj_pt];
    av.xyz = vec![0.; 3 * max_real];
    av.dxy = vec![0.; 2 * max_view];
    av.xresid = vec![0.; max_proj_pt];
    av.yresid = vec![0.; max_proj_pt];
    av.weight = vec![0.; max_proj_pt];
    av.view_median_res = vec![0.; max_view];
    av.isec_view = vec![0; max_proj_pt];
    av.ireal_str = vec![0; max_real];
    av.ind_proj_wgt_list = vec![0; max_proj_pt];
    av.iv_start_wgt_group = vec![0; (mx.max_view * MAX_WGT_RINGS) as usize];
    av.ip_start_wgt_view = vec![0; (mx.max_view * MAX_WGT_RINGS) as usize];
    av.map_tilt = vec![0; max_view];
    av.map_gmag = vec![0; max_view];
    av.map_comp = vec![0; max_view];
    av.map_dmag = vec![0; max_view];
    av.map_skew = vec![0; max_view];
    av.map_rot = vec![0; max_view];
    av.lin_tilt = vec![0; max_view];
    av.lin_gmag = vec![0; max_view];
    av.lin_comp = vec![0; max_view];
    av.lin_dmag = vec![0; max_view];
    av.lin_skew = vec![0; max_view];
    av.lin_rot = vec![0; max_view];
    av.map_alf = vec![0; max_view];
    av.lin_alf = vec![0; max_view];
    av.frc_tilt = vec![0.; max_view];
    av.frc_gmag = vec![0.; max_view];
    av.frc_comp = vec![0.; max_view];
    av.frc_dmag = vec![0.; max_view];
    av.frc_skew = vec![0.; max_view];
    av.frc_rot = vec![0.; max_view];
    av.rot = vec![0.; max_view];
    av.tilt = vec![0.; max_view];
    av.gmag = vec![0.; max_view];
    av.comp = vec![0.; max_view];
    av.tilt_inc = vec![0.; max_view];
    av.dmag = vec![0.; max_view];
    av.skew = vec![0.; max_view];
    av.frc_alf = vec![0.; max_view];
    av.alf = vec![0.; max_view];
    av.map_view_to_file = vec![0; max_view];
    av.map_file_to_view = vec![0; av.nfile_views as usize];
    av.glb_rot = vec![0.; max_view];
    av.glb_tilt = vec![0.; max_view];
    av.glb_alf = vec![0.; max_view];
    av.glb_gmag = vec![0.; max_view];
    av.glb_dmag = vec![0.; max_view];
    av.glb_skew = vec![0.; max_view];

    // The source's `ierr = (all non-NULL) ? 0 : 1` (`utilfuncs.cpp:109-118`):
    // every `Vec` above is allocated.
    *ierr = 0;
}

/// Original: `allocateMapsep` (`utilfuncs.cpp:121`).
pub fn allocate_mapsep(sg: &mut MapSepGroups, mx: &ArrayMaxes, ierr: &mut i32) {
    sg.iviews_in_group = vec![0; (mx.max_view * MAXGRP) as usize];
    sg.num_sep_in_group = vec![0; MAXGRP as usize];
    // `ierr = (sg->iviewsInGroup && sg->numSepInGroup) ? 0 : 1;`
    *ierr = 0;
}

/// Original: `copyArray` (`utilfuncs.cpp:131`) — copies an array with
/// `memcpy` (so it better not overlap), with the destination and its Fortran
/// indexes first so that this can easily replace a vector assignment.
///
/// The source's trailing `int size = 4` is the element size in bytes; every
/// caller copies `int` to `int` or `float` to `float` at the default 4, so the
/// element type `T` carries it.  The one caller that copies within a single
/// array (`patchtrack.cpp:598`, two disjoint triples of `av->xyz`) splits the
/// array to get the two slices, which `memcpy`'s no-overlap rule already
/// requires.
pub fn copy_array<T: Copy>(to_arr: &mut [T], to1: i32, to2: i32, from_arr: &[T], from1: i32) {
    let out = (to1 - 1) as usize;
    let inp = (from1 - 1) as usize;
    let n = (to2 + 1 - to1) as usize;
    to_arr[out..out + n].copy_from_slice(&from_arr[inp..inp + n]);
}

/// Original: `memoryError` (`utilfuncs.cpp:139`) — the opposite of the
/// Fortran one: it is an error if the argument is false.
pub fn memory_error(all_non_null: bool, descrip: &str) {
    if !all_non_null {
        exit_error(&c_format_bytes("Allocating %s", &[CArg::Str(descrip)]));
    }
}

/// Original: `countNumInView` (`utilfuncs.cpp:150`) — counts the number of
/// points in each view, given the list of `nrealPt` point numbers in
/// `listReal`, the starting index of each real point in `irealStr`, and the
/// array of view numbers for all the points in `isecView`.  Returns the count
/// in `numInView`.
///
/// `av` is the file-scope pointer (`testSetFracStep` only).  The possibly-NULL
/// `realInTestSet` and `totalPtNum` are `Option`s.
///
/// Fixed in translation (2026-09-26, `BUGS.md`): the source indexes
/// `realInTestSet[listReal[j]]` with the 1-based point number
/// (`utilfuncs.cpp:161`), i.e. the flag of the *next* point, and for the last
/// point reads one past an array `tiltalign.cpp` allocates `nrealPt` long.
/// Here the point's own flag, `realInTestSet[listReal[j] - 1]`, is read.
#[allow(clippy::too_many_arguments)]
pub fn count_num_in_view(
    av: &AlignVariables,
    list_real: &[i32],
    nreal_pt: i32,
    ireal_str: &[i32],
    isec_view: &[i32],
    nview: i32,
    num_in_view: &mut [i32],
    real_in_test_set: Option<&[i32]>,
    mut total_pt_num: Option<&mut i32>,
) {
    //
    for i in 0..nview {
        num_in_view[i as usize] = 0;
    }
    if let Some(total) = total_pt_num.as_deref_mut() {
        *total = 0;
    }
    for j in 0..nreal_pt as usize {
        if let Some(test_set) = real_in_test_set
            && av.test_set_frac_step > 0.
            && test_set[(list_real[j] - 1) as usize] != 0
        {
            continue;
        }
        let ist = ireal_str[(list_real[j] - 1) as usize];
        let ind = ireal_str[list_real[j] as usize] - 1;
        let mut i = ist;
        while i <= ind {
            num_in_view[(isec_view[(i - 1) as usize] - 1) as usize] += 1;
            i += 1;
        }
        if let Some(total) = total_pt_num.as_deref_mut() {
            *total += ind + 1 - ist;
        }
    }
}

/// Original: `formattedError` (`utilfuncs.cpp:177`) — switch an error from 3
/// to 4 decimal places below a threshold, with an optional width
/// specification.
///
/// The source returns a pointer into a `static char buffer[64]` written with
/// `snprintf(buffer, 63, ...)`; the result is returned by value, cut at the
/// same 62 characters.
pub fn formatted_error(err: f32, thresh: f32, width: i32) -> String {
    let format = if width != 0 {
        c_format_bytes(
            "%%%d.%df",
            &[
                CArg::Int(width as i64),
                CArg::Int(if err < thresh { 4 } else { 3 }),
            ],
        )
    } else {
        c_format_bytes("%%.%df", &[CArg::Int(if err < thresh { 4 } else { 3 })])
    };
    let format = String::from_utf8_lossy(&format[..format.len().min(30)]).into_owned();
    let mut buffer = c_format_bytes(&format, &[CArg::Dbl(err as f64)]);
    buffer.truncate(62);
    String::from_utf8_lossy(&buffer).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BUGS.md` "`countNumInView`": the source read `realInTestSet` with the
    /// 1-based point number (the next point's flag, and one past the array for
    /// the last point).  Each point's own flag decides here.
    #[test]
    fn count_num_in_view_skips_the_flagged_point_itself() {
        let mut av = AlignVariables::default();
        av.test_set_frac_step = 0.5;
        // Three points: point 1 on views 1,2; point 2 on views 2,3; point 3 on
        // view 3 only.
        let ireal_str = [1, 3, 5, 6];
        let isec_view = [1, 2, 2, 3, 3];
        let list_real = [1, 2, 3];
        // Only the last point is in the test set.
        let test_set = [0, 0, 1];
        let mut num_in_view = [0; 3];
        let mut total = 0;
        count_num_in_view(
            &av,
            &list_real,
            3,
            &ireal_str,
            &isec_view,
            3,
            &mut num_in_view,
            Some(&test_set),
            Some(&mut total),
        );
        assert_eq!(num_in_view, [1, 2, 1]);
        assert_eq!(total, 4);
    }
}
