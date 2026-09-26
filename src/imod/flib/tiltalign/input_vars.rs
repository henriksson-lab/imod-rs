//! Translation of `IMOD/flib/tiltalign/input_vars.cpp` — functions for
//! inputting and setting up variables.
//!
//! Compiled into `tiltalign` only (`flib/beadtrack/Makefile` does not list
//! it), so every `errorExit` is the `tiltalign` build, `error_exit::<false>`.
//!
//! # File-scope and function-local statics
//!
//! - `av`, `mx`, `sg` (`input_vars.cpp:18-20`), set by
//!   `inputVarsSetPointers`, are parameters (`alivar.rs`).
//! - `input_vars` keeps 44 function-local statics (`:45-58`, `:70-71`): the
//!   saved map lists, the grouping specifications that `automap` fills on the
//!   global pass (`ifLocal == 0`) or the first local pass (`ifLocal == 1`) and
//!   reuses on later local passes (`ifLocal == 2`), and the option values
//!   (`ioptRot`, `ioptTilt`, `irefTiltIn`, `ioptMag`, `ioptDel`, `ioptDist`,
//!   `ioptAlf`, `nviewFixIn`) that later passes reuse without re-reading.  They
//!   persist across calls, so they live in one process-global
//!   [`InputVarsStatics`] behind a `Mutex`, locked once per call.  The five
//!   `static IntVec`s are constructed with `mx->maxView` elements on the first
//!   call only, as a C++ function-local static is; `mapListDist` is resized on
//!   every call (`:79-80`).  `ioptDel` is never initialised except by the
//!   zero-initialisation of the static and is only ever set to 2 (`:466`), so
//!   once any distortion option was entered it stays 2 for the rest of the run.
//! - `prependLocal`'s `static char buf[64]` is written before every read and
//!   its contents are copied by every caller before the next call, so it is
//!   returned by value.
//!
//! # Arithmetic
//!
//! `float dtor = 0.0174532;` is the double literal rounded to `float`.  The
//! `tiltInc` expressions (`:314-326`) multiply two `float`s (single-precision
//! product) and add `(1. - frc) * value`, which is double; the whole difference
//! is then stored to `float`.
//!
//! # Upstream defects fixed in translation (2026-09-26, `BUGS.md`)
//!
//! - The compression automap (`:425-427`) passes the **X-tilt** grouping
//!   statics (`nmapDefXtilt`, `nRanSpecXtilt`, …) in the source; it has
//!   compression statics of its own here.
//! - `automap`'s `nRanSpecIn` is by value in the source, so the statics
//!   `nRanSpec*` stay 0; it is by reference here (`map_vars.rs`).
//!
//! # Upstream, kept as written
//!
//! - The dummy-dmag block (`:544`) is guarded by `ifLocal <= -1`, which no
//!   caller passes; it is translated but unreachable.

use std::io::Write;
use std::sync::Mutex;

use super::alivar::AlignVariables;
use super::arraymaxes::{ArrayMaxes, MAXGRP};
use super::map_vars::{
    analyze_maps, automap, input_separate_groups, map_separate_group, set_grp_size,
};
use super::mapsepgroups::MapSepGroups;
use super::utilfuncs::{allocate_mapsep, error_exit, memory_error};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use crate::imod::libcfshr::gettiltangles::get_tilt_angles;
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_float, pip_get_integer, pip_get_integer_array, pip_number_of_entries,
};

const NGRP: usize = MAXGRP as usize;

/// The function-local statics of `input_vars` (`input_vars.cpp:45-58`,
/// `:70-71`).
pub struct InputVarsStatics {
    /// Whether the `static IntVec`s have been constructed (first call).
    constructed: bool,
    map_list: Vec<i32>,
    map_list_rot: Vec<i32>,
    map_list_tilt: Vec<i32>,
    map_list_mag: Vec<i32>,
    map_list_xtilt: Vec<i32>,
    map_list_dist: [Vec<i32>; 2],
    nmap_def_rot: i32,
    n_ran_spec_rot: i32,
    nmap_spec_rot: [i32; NGRP],
    iv_spec_str_rot: [i32; NGRP],
    iv_spec_end_rot: [i32; NGRP],
    nmap_def_tilt: i32,
    n_ran_spec_tilt: i32,
    nmap_spec_tilt: [i32; NGRP],
    iv_spec_str_tilt: [i32; NGRP],
    iv_spec_end_tilt: [i32; NGRP],
    nmap_def_mag: i32,
    n_ran_spec_mag: i32,
    nmap_spec_mag: [i32; NGRP],
    iv_spec_str_mag: [i32; NGRP],
    iv_spec_end_mag: [i32; NGRP],
    nmap_def_xtilt: i32,
    n_ran_spec_xtilt: i32,
    nmap_spec_xtilt: [i32; NGRP],
    iv_spec_str_xtilt: [i32; NGRP],
    iv_spec_end_xtilt: [i32; NGRP],
    nmap_def_comp: i32,
    n_ran_spec_comp: i32,
    nmap_spec_comp: [i32; NGRP],
    iv_spec_str_comp: [i32; NGRP],
    iv_spec_end_comp: [i32; NGRP],
    nmap_def_dist: [i32; 2],
    n_ran_spec_dist: [i32; 2],
    nmap_spec_dist: [i32; NGRP * 2],
    iv_spec_str_dist: [i32; NGRP * 2],
    iv_spec_end_dist: [i32; NGRP * 2],
    iopt_rot: i32,
    iopt_tilt: i32,
    iref_tilt_in: i32,
    iopt_mag: i32,
    iopt_del: i32,
    iopt_dist: [i32; 2],
    iopt_alf: i32,
    nview_fix_in: i32,
}

static INPUT_VARS_STATICS: Mutex<InputVarsStatics> = Mutex::new(InputVarsStatics {
    constructed: false,
    map_list: Vec::new(),
    map_list_rot: Vec::new(),
    map_list_tilt: Vec::new(),
    map_list_mag: Vec::new(),
    map_list_xtilt: Vec::new(),
    map_list_dist: [Vec::new(), Vec::new()],
    nmap_def_rot: 0,
    n_ran_spec_rot: 0,
    nmap_spec_rot: [0; NGRP],
    iv_spec_str_rot: [0; NGRP],
    iv_spec_end_rot: [0; NGRP],
    nmap_def_tilt: 0,
    n_ran_spec_tilt: 0,
    nmap_spec_tilt: [0; NGRP],
    iv_spec_str_tilt: [0; NGRP],
    iv_spec_end_tilt: [0; NGRP],
    nmap_def_mag: 0,
    n_ran_spec_mag: 0,
    nmap_spec_mag: [0; NGRP],
    iv_spec_str_mag: [0; NGRP],
    iv_spec_end_mag: [0; NGRP],
    nmap_def_xtilt: 0,
    n_ran_spec_xtilt: 0,
    nmap_spec_xtilt: [0; NGRP],
    iv_spec_str_xtilt: [0; NGRP],
    iv_spec_end_xtilt: [0; NGRP],
    nmap_def_comp: 0,
    n_ran_spec_comp: 0,
    nmap_spec_comp: [0; NGRP],
    iv_spec_str_comp: [0; NGRP],
    iv_spec_end_comp: [0; NGRP],
    nmap_def_dist: [0; 2],
    n_ran_spec_dist: [0; 2],
    nmap_spec_dist: [0; NGRP * 2],
    iv_spec_str_dist: [0; NGRP * 2],
    iv_spec_end_dist: [0; NGRP * 2],
    iopt_rot: 0,
    iopt_tilt: 0,
    iref_tilt_in: 0,
    iopt_mag: 0,
    iopt_del: 0,
    iopt_dist: [0; 2],
    iopt_alf: 0,
    nview_fix_in: 0,
});

/// Original: `inputVarsSetPointers` (`input_vars.cpp:27`).
///
/// The source stores the three pointers in file-scope statics; the functions
/// of this module take them as parameters instead, so there is nothing to
/// store.
pub fn input_vars_set_pointers(
    av_in: &mut AlignVariables,
    mx_in: &mut ArrayMaxes,
    sg_in: &mut MapSepGroups,
) {
    let _ = (av_in, mx_in, sg_in);
}

/// Original: `input_vars` (`input_vars.cpp:39`).
///
/// INPUT_VARS gets the specifications for the geometric variables rotation,
/// tilt, mag, compression, X stretch, skew, and X axis tilt.  It fills the VAR
/// array and the different variable and mapping arrays.
///
/// `varName` is the caller's array of 8-character names; `numInView` is the
/// caller's array (never `NULL` at the two call sites, `tiltalign.cpp:315,
/// 2254`).  The source's uninitialised `fixdum` and `ioptComp` start at 0:
/// `fixdum` is only written, and `ioptComp` is only read on the global pass,
/// where it is set.
#[allow(clippy::too_many_arguments)]
pub fn input_vars(
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    sg: &mut MapSepGroups,
    var: &mut [f32],
    var_name: &mut [u8],
    num_var_search: &mut i32,
    num_var_angles: &mut i32,
    num_var_scaled: &mut i32,
    min_tilt_ind: &mut i32,
    num_comp_search: &mut i32,
    if_local: i32,
    map_tilt_start: &mut i32,
    map_alf_start: &mut i32,
    map_alf_end: &mut i32,
    if_bt_search: &mut i32,
    tilt_orig: &mut [f32],
    tilt_add: &mut f32,
    num_in_view: &[i32],
    nin_thresh: i32,
    rot_entered: &mut f32,
) {
    let mut guard = INPUT_VARS_STATICS.lock().unwrap_or_else(|e| e.into_inner());
    let st = &mut *guard;
    if !st.constructed {
        st.constructed = true;
        st.map_list = vec![0; mx.max_view as usize];
        st.map_list_rot = vec![0; mx.max_view as usize];
        st.map_list_tilt = vec![0; mx.max_view as usize];
        st.map_list_mag = vec![0; mx.max_view as usize];
        st.map_list_xtilt = vec![0; mx.max_view as usize];
    }
    let mut group_size: Vec<f32> = vec![0.; mx.max_view as usize];
    //
    let mut dump = [[0u8; 9]; 8];
    let dtor: f32 = 0.0174532_f64 as f32;
    let dist_def_group: [&str; 2] = ["XStretchDefaultGrouping", "SkewDefaultGrouping"];
    let dist_non_def_group: [&str; 2] = ["XStretchNondefaultGroup", "SkewNondefaultGroup"];
    //
    let mut def_map_opt: String;
    let mut non_def_map_opt: String;
    let power_tilt: f32;
    let power_comp: f32;
    let power_mag: f32;
    let power_skew: f32;
    let power_dmag: f32;
    let mut power: f32;
    let powe_rrot: f32;
    let power_alf: f32;
    let mut rot_start: f32 = 0.;
    let def_rot: f32;
    let mut tilt_min: f32;
    let mut orig_dev: f32;
    let mut fixdum: f32 = 0.;
    let mut iref1: i32;
    let mut iflin: i32;
    let mut nview_fix: i32;
    let iref2: i32;
    let mut iref_comp: i32;
    let mut iopt_comp: i32 = 0;
    let mut iffix: i32;
    let iref_tilt: i32;
    let mut iref_dmag: i32;
    let mut ivdum: i32;
    let mut nvar_tmp: i32;
    let mut num_dmag_var: i32;
    let mut ivl: i32;
    let mut ivh: i32;
    let mut no_sep_tilt_groups: i32;
    let mut num_sep_save: i32;
    let mut _map_fix: i32;
    let mut ierr: i32 = 0;
    let mut stdout = ImodFile::Stdout;
    //
    st.map_list_dist[0].resize(mx.max_view as usize, 0);
    st.map_list_dist[1].resize(mx.max_view as usize, 0);
    power_tilt = 1.;
    power_comp = 1.;
    power_mag = 0.;
    power_skew = 0.;
    power_dmag = 1.5;
    powe_rrot = 0.;
    power_alf = 0.;
    iffix = 0;
    //
    *tilt_add = 0.;
    *num_var_search = 0;
    no_sep_tilt_groups = 0;
    if if_local == 0 {
        allocate_mapsep(sg, mx, &mut ierr);
        memory_error(ierr == 0, "arrays for mapsep");
    }

    // print *,nview, ' views'
    // print *,(numInView(i), i = 1, nview)
    if if_local == 0 {
        input_separate_groups::<false>(
            av,
            mx,
            &mut sg.num_separate_groups,
            &mut sg.num_sep_in_group,
            &mut sg.iviews_in_group,
        );
        for ig in 1..=sg.num_separate_groups {
            map_separate_group::<false>(
                &mut sg.iviews_in_group[((ig - 1) * mx.max_view) as usize..],
                &mut sg.num_sep_in_group[(ig - 1) as usize],
                &av.map_file_to_view,
                av.nfile_views,
            );
        }
        pip_get_integer(b"NoSeparateTiltGroups", &mut no_sep_tilt_groups);
        rot_start = 0.;
        pip_get_float(b"RotationAngle", &mut rot_start);
    }
    //
    if if_local < 2 {
        av.if_rot_fix = 0;
        pip_get_integer(b"RotationFixedView", &mut av.if_rot_fix);
        st.iopt_rot = 0;
        pip_get_integer(
            prepend_local("RotOption", if_local).as_bytes(),
            &mut st.iopt_rot,
        );
    }
    //
    // 4 / 10 / 04: Eliminated global rotation variable, simplified treatment
    // of rotation
    //
    if if_local == 0 {
        *rot_entered = rot_start;
        rot_start = dtor * rot_start;
    } else {
        rot_start = 0.;
    }
    //
    if av.if_rot_fix > 0 {
        av.if_rot_fix = nearest_view(av, av.if_rot_fix);
    }
    //
    // set up appropriate mapping list or read it in
    // set the meaning of ifRotFix =  0 for a one global rotation variable
    // and the others incremental to it, + for one variable fixed,
    // - 1 for all fixed, -2 for single variable, -3 otherwise
    //
    iref1 = av.if_rot_fix;
    if av.xyz_fixed != 0 && st.iopt_rot > 0 {
        iref1 = 0;
    }
    iflin = 0;
    def_rot = rot_start;
    //
    // All one variable - set default to true angle
    //
    if st.iopt_rot < 0 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = 1;
        }
        av.if_rot_fix = -2;
        //
        // All fixed - also set default to angle, set reference to 1
        //
    } else if st.iopt_rot == 0 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = 0;
        }
        av.if_rot_fix = -1;
        iref1 = 1;
        //
        // All separate variables
        //
    } else if st.iopt_rot == 1 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = i;
        }
        //
        // Specified mapping
        // 5 / 2 / 02: changed irotfix (?) to iref1 and made output conditional
        //
    } else if st.iopt_rot == 2 {
        _map_fix = 0;
        if iref1 > 0 {
            _map_fix = av.map_view_to_file[(iref1 - 1) as usize];
        }
        get_map_list(
            "Rot",
            if_local,
            av.nview,
            &mut st.map_list,
            &mut st.map_list_rot,
        );
        //
        // automap
        //
    } else {
        power = 0.;
        if st.iopt_rot == 3 {
            iflin = 1;
            power = powe_rrot;
        }
        set_grp_size(&av.tilt, av.nview, power, &mut group_size);
        def_map_opt = prepend_local("RotDefaultGrouping", if_local);
        non_def_map_opt = prepend_local("RotNondefaultGroup", if_local);
        automap::<false>(
            mx,
            sg,
            av.nview,
            &mut st.map_list,
            &group_size,
            &av.map_file_to_view,
            av.nfile_views,
            1,
            &def_map_opt,
            &non_def_map_opt,
            num_in_view,
            nin_thresh,
            if_local,
            &mut st.nmap_def_rot,
            &mut st.n_ran_spec_rot,
            &mut st.iv_spec_str_rot,
            &mut st.iv_spec_end_rot,
            &mut st.nmap_spec_rot,
        );
    }
    //
    // Make sure ifRotFix = 0 means what it is supposed to
    //
    if if_local > 0 && av.if_rot_fix == 0 {
        av.if_rot_fix = -3;
    }
    //
    // analyze map list
    //
    analyze_maps::<false>(
        mx,
        &mut av.rot,
        &mut av.map_rot,
        &mut av.lin_rot,
        &mut av.frc_rot,
        &mut av.fixed_rot,
        &mut fixdum,
        iflin,
        &st.map_list,
        av.nview,
        iref1,
        0,
        def_rot,
        b"rot ",
        var,
        var_name,
        num_var_search,
        &av.map_view_to_file,
    );
    //for (i = 0; i < av->nview; i++)
    //printf(" %d%s", av->mapRot[i], ((i + 1) % 25 == 0 || i == av->nview - 1) ? "\n" : "");
    //
    if if_local == 0 {
        //
        // get initial tilt angles for all views, convert to radians
        // save adjusted angles in tiltOrig, then map as radians into tilt
        //
        let lim_tilt = av.nfile_views as usize;
        get_tilt_angles(&mut av.nfile_views, &mut tilt_orig[..lim_tilt]);
        *tilt_add = 0.;
        pip_get_float(b"AngleOffset", tilt_add);
        for i in 1..=av.nfile_views {
            tilt_orig[(i - 1) as usize] += *tilt_add;
            if av.map_file_to_view[(i - 1) as usize] != 0 {
                av.tilt[(av.map_file_to_view[(i - 1) as usize] - 1) as usize] =
                    tilt_orig[(i - 1) as usize] * dtor;
            }
        }
    }
    //
    // For tilt, set up default on increments and maps
    // also find section closest to zero: it will have mag set to 1.0
    // This needs to be done fresh on each local area since views can change
    *map_tilt_start = *num_var_search + 1;
    tilt_min = 100.;
    for i in 1..=av.nview {
        let v = (i - 1) as usize;
        av.tilt_inc[v] = 0.;
        av.map_tilt[v] = 0;
        st.map_list[v] = 0;
        let d = av.tilt[v] - *tilt_add * dtor;
        orig_dev = if d >= 0. { d } else { -d };
        if tilt_min > orig_dev {
            tilt_min = orig_dev;
            *min_tilt_ind = i;
        }
    }
    //
    // get type of mapping
    //
    if if_local <= 1 {
        st.iopt_tilt = 0;
        pip_get_integer(
            prepend_local("TiltOption", if_local).as_bytes(),
            &mut st.iopt_tilt,
        );
    }
    //
    // set up mapList appropriately or get whole map
    //
    iflin = 0;
    nview_fix = 0;
    if st.iopt_tilt >= 1 && st.iopt_tilt <= 3 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = i;
        }
        if st.iopt_tilt > 1 {
            st.map_list[(*min_tilt_ind - 1) as usize] = 0;
        }
        if st.iopt_tilt != 2 {
            if if_local <= 1
                && pip_get_integer(
                    prepend_local("TiltFixedView", if_local).as_bytes(),
                    &mut st.nview_fix_in,
                ) != 0
            {
                error_exit::<false>("You must enter a fixed view with this Tilt option", 0);
            }
            nview_fix = nearest_view(av, st.nview_fix_in);
            st.map_list[(nview_fix - 1) as usize] = 0;
        }
    } else if st.iopt_tilt == 4 {
        get_map_list(
            "Tilt",
            if_local,
            av.nview,
            &mut st.map_list,
            &mut st.map_list_tilt,
        );
    } else if st.iopt_tilt >= 5 {
        if st.iopt_tilt > 6 {
            if if_local <= 1
                && pip_get_integer(
                    prepend_local("TiltSecondFixedView", if_local).as_bytes(),
                    &mut st.nview_fix_in,
                ) != 0
            {
                error_exit::<false>(
                    // Fixed in translation (`BUGS.md`): the source reads
                    // "this av->tilt option", a search-and-replace artifact.
                    "You must enter a second fixed view with this tilt option",
                    0,
                );
            }
            nview_fix = nearest_view(av, st.nview_fix_in);
        }
        //
        // automap: get map, then set variable # 0 for ones mapped to
        // minimum tilt
        //
        power = 0.;
        if st.iopt_tilt == 5 || st.iopt_tilt == 7 {
            iflin = 1;
            power = power_tilt;
        }
        set_grp_size(&av.tilt, av.nview, power, &mut group_size);
        //
        // Cancel the separate groups if appropriate by saving, zeroing, and restoring them
        num_sep_save = sg.num_separate_groups;
        if (no_sep_tilt_groups == 1 && av.patch_track_model != 0) || no_sep_tilt_groups > 1 {
            sg.num_separate_groups = 0;
        }
        def_map_opt = prepend_local("TiltDefaultGrouping", if_local);
        non_def_map_opt = prepend_local("TiltNondefaultGroup", if_local);
        automap::<false>(
            mx,
            sg,
            av.nview,
            &mut st.map_list,
            &group_size,
            &av.map_file_to_view,
            av.nfile_views,
            1,
            &def_map_opt,
            &non_def_map_opt,
            num_in_view,
            nin_thresh,
            if_local,
            &mut st.nmap_def_tilt,
            &mut st.n_ran_spec_tilt,
            &mut st.iv_spec_str_tilt,
            &mut st.iv_spec_end_tilt,
            &mut st.nmap_spec_tilt,
        );
        sg.num_separate_groups = num_sep_save;
    }
    //
    // analyze map list
    //
    iref1 = *min_tilt_ind;
    let mut iref2_v = nview_fix;
    if av.xyz_fixed != 0 && st.iopt_tilt != 0 {
        iref1 = 0;
        iref2_v = 0;
    }
    iref2 = iref2_v;
    analyze_maps::<false>(
        mx,
        &mut av.tilt,
        &mut av.map_tilt,
        &mut av.lin_tilt,
        &mut av.frc_tilt,
        &mut av.fixed_tilt,
        &mut av.fixed_tilt2,
        iflin,
        &st.map_list,
        av.nview,
        iref1,
        iref2,
        -999.,
        b"tilt",
        var,
        var_name,
        num_var_search,
        &av.map_view_to_file,
    );
    //
    // set tiltInc so it will give back the right tilt for whatever
    // mapping or interpolation scheme is used
    //
    for iv in 1..=av.nview {
        let v = (iv - 1) as usize;
        av.tilt_inc[v] = 0.;
        if av.map_tilt[v] != 0 {
            let frc = av.frc_tilt[v];
            let vm = var[(av.map_tilt[v] - 1) as usize];
            if av.lin_tilt[v] > 0 {
                av.tilt_inc[v] = (av.tilt[v] as f64
                    - ((frc * vm) as f64
                        + (1. - frc as f64) * var[(av.lin_tilt[v] - 1) as usize] as f64))
                    as f32;
            } else if av.lin_tilt[v] == -1 {
                av.tilt_inc[v] = (av.tilt[v] as f64
                    - ((frc * vm) as f64 + (1. - frc as f64) * av.fixed_tilt as f64))
                    as f32;
            } else if av.lin_tilt[v] == -2 {
                av.tilt_inc[v] = (av.tilt[v] as f64
                    - ((frc * vm) as f64 + (1. - frc as f64) * av.fixed_tilt2 as f64))
                    as f32;
            } else {
                av.tilt_inc[v] = av.tilt[v] - vm;
            }
        }
    }
    *num_var_angles = *num_var_search;
    //
    // get reference view to fix magnification at 1.0
    //
    if if_local <= 1 {
        st.iref_tilt_in = av.map_view_to_file[(*min_tilt_ind - 1) as usize];
        pip_get_integer(
            prepend_local("MagReferenceView", if_local).as_bytes(),
            &mut st.iref_tilt_in,
        );
    }
    iref_tilt = nearest_view(av, st.iref_tilt_in);
    //
    // get type of mag variable set up
    //
    if if_local <= 1 {
        st.iopt_mag = 0;
        pip_get_integer(
            prepend_local("MagOption", if_local).as_bytes(),
            &mut st.iopt_mag,
        );
    }
    //
    // set up appropriate mapping list or read it in
    //
    iflin = 0;
    if st.iopt_mag <= 0 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = 0;
        }
    } else if st.iopt_mag == 1 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = i;
        }
    } else if st.iopt_mag == 2 {
        get_map_list(
            "Mag",
            if_local,
            av.nview,
            &mut st.map_list,
            &mut st.map_list_mag,
        );
    } else {
        power = 0.;
        if st.iopt_mag == 3 {
            iflin = 1;
            power = power_mag;
        }
        set_grp_size(&av.tilt, av.nview, power, &mut group_size);
        def_map_opt = prepend_local("MagDefaultGrouping", if_local);
        non_def_map_opt = prepend_local("MagNondefaultGroup", if_local);
        automap::<false>(
            mx,
            sg,
            av.nview,
            &mut st.map_list,
            &group_size,
            &av.map_file_to_view,
            av.nfile_views,
            1,
            &def_map_opt,
            &non_def_map_opt,
            num_in_view,
            nin_thresh,
            if_local,
            &mut st.nmap_def_mag,
            &mut st.n_ran_spec_mag,
            &mut st.iv_spec_str_mag,
            &mut st.iv_spec_end_mag,
            &mut st.nmap_spec_mag,
        );
    }
    //
    // analyze map list
    //
    iref1 = iref_tilt;
    if av.xyz_fixed != 0 && st.iopt_mag != 0 {
        iref1 = 0;
    }
    analyze_maps::<false>(
        mx,
        &mut av.gmag,
        &mut av.map_gmag,
        &mut av.lin_gmag,
        &mut av.frc_gmag,
        &mut av.fixed_gmag,
        &mut fixdum,
        iflin,
        &st.map_list,
        av.nview,
        iref1,
        0,
        1.,
        b"mag ",
        var,
        var_name,
        num_var_search,
        &av.map_view_to_file,
    );
    //
    // set up compression variables if desired: the ones at fixed zero tilt
    // and at least one other one must have compression of 1.0
    // get reference view to fix compression at 1.0
    //
    iref_comp = 0;
    if if_local == 0 {
        iopt_comp = 0;
        iref_comp = 1;
        pip_get_integer(b"CompReferenceView", &mut iref_comp);
        pip_get_integer(b"CompOption", &mut iopt_comp);
        if iopt_comp == 0 {
            iref_comp = 0;
        }
    }
    iflin = 0;
    if iref_comp <= 0 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = 0;
        }
        iref_comp = 1;
    } else {
        iref_comp = nearest_view(av, iref_comp);
        //
        // get type of comp variable set up
        //
        //
        // set up appropriate mapping list or read it in
        //
        if iopt_comp == 1 {
            for i in 1..=av.nview {
                st.map_list[(i - 1) as usize] = i;
            }
        } else if iopt_comp == 2 {
            // `getMapList("Comp", 0, av->nview, mapList, mapList)`: the same
            // vector as list and save.  With ifLocal 0 the save is the copy
            // `mapSave[i] = mapList[i]` onto itself, so a scratch copy stands
            // in for the aliased save argument and is discarded.
            let mut map_save = st.map_list.clone();
            get_map_list("Comp", 0, av.nview, &mut st.map_list, &mut map_save);
        } else {
            power = 0.;
            if iopt_comp == 3 {
                iflin = 1;
                power = power_comp;
            }
            set_grp_size(&av.tilt, av.nview, power, &mut group_size);
            def_map_opt = prepend_local("CompDefaultGrouping", if_local);
            non_def_map_opt = prepend_local("CompNondefaultGroup", if_local);
            automap::<false>(
                mx,
                sg,
                av.nview,
                &mut st.map_list,
                &group_size,
                &av.map_file_to_view,
                av.nfile_views,
                1,
                &def_map_opt,
                &non_def_map_opt,
                num_in_view,
                nin_thresh,
                if_local,
                // Fixed in translation (2026-09-26, `BUGS.md`): the source passes
                // the X-tilt grouping statics here (`:425-427`); compression has
                // its own.
                &mut st.nmap_def_comp,
                &mut st.n_ran_spec_comp,
                &mut st.iv_spec_str_comp,
                &mut st.iv_spec_end_comp,
                &mut st.nmap_spec_comp,
            );
            // if (.not.pipinput)
            // TODO?
            //write(6, 111) (mapList[i - 1], i = 1, av->nview);
        }
    }
    //
    // if view has tilt fixed at zero, see if mapped to any other
    // view not fixed at zero tilt; if not, need to fix at 1.0
    //
    for iv in 1..=av.nview {
        let v = (iv - 1) as usize;
        if av.tilt[v] == 0. && av.map_tilt[v] == 0 {
            iffix = 1;
            for jv in 1..=av.nview {
                let w = (jv - 1) as usize;
                if st.map_list[v] == st.map_list[w] && (av.tilt[w] != 0. || av.map_tilt[w] != 0) {
                    iffix = 0;
                }
            }
            if iffix == 1 {
                st.map_list[v] = st.map_list[(iref_comp - 1) as usize];
            }
        }
    }
    //
    // analyze map list
    //
    nvar_tmp = *num_var_search;
    analyze_maps::<false>(
        mx,
        &mut av.comp,
        &mut av.map_comp,
        &mut av.lin_comp,
        &mut av.frc_comp,
        &mut av.fixed_comp,
        &mut fixdum,
        iflin,
        &st.map_list,
        av.nview,
        iref_comp,
        0,
        1.,
        b"comp",
        var,
        var_name,
        num_var_search,
        &av.map_view_to_file,
    );
    //
    *num_comp_search = *num_var_search - nvar_tmp;
    // get type of distortion variable set up
    //
    if if_local <= 1 {
        st.iopt_dist[0] = 0;
        st.iopt_dist[1] = 0;
        pip_get_integer(
            prepend_local("XStretchOption", if_local).as_bytes(),
            &mut st.iopt_dist[0],
        );
        pip_get_integer(
            prepend_local("SkewOption", if_local).as_bytes(),
            &mut st.iopt_dist[1],
        );
        if st.iopt_dist[0] > 0 || st.iopt_dist[1] > 0 {
            st.iopt_del = 2;
        }
    }
    //
    // get reference and dummy views for distortion: take reference on the
    // same side of minimum as the mag reference, but avoid the minimum tilt
    //
    iref_dmag = if 1 > av.nview / 4 { 1 } else { av.nview / 4 };
    ivdum = if 1 > 3 * av.nview / 4 {
        1
    } else {
        3 * av.nview / 4
    };
    if iref_dmag == *min_tilt_ind || (iref_tilt > *min_tilt_ind && ivdum != *min_tilt_ind) {
        iref_dmag = ivdum;
        ivdum = if 1 > av.nview / 4 { 1 } else { av.nview / 4 };
    }
    //
    power = power_dmag;
    for idist in 1..=2 {
        let d = (idist - 1) as usize;
        iflin = 0;
        if st.iopt_del <= 0 || st.iopt_dist[d] <= 0 {
            for i in 1..=av.nview {
                st.map_list[(i - 1) as usize] = 0;
            }
        } else if idist == 1 || st.iopt_del > 1 {
            //
            // set up appropriate mapping list or read it in
            //
            if st.iopt_dist[d] <= 1 {
                for i in 1..=av.nview {
                    st.map_list[(i - 1) as usize] = i;
                }
            } else if st.iopt_dist[d] == 2 {
                let dist_opt_root: [&str; 2] = ["XStretch", "Skew"];
                get_map_list(
                    dist_opt_root[d],
                    if_local,
                    av.nview,
                    &mut st.map_list,
                    &mut st.map_list_dist[d],
                );
            } else {
                if idist == 1 {
                    if st.iopt_dist[d] > 3 {
                        iflin = 1;
                    }
                    if st.iopt_dist[d] < 3 {
                        power = 1.;
                    }
                } else if st.iopt_dist[d] == 3 {
                    iflin = 1;
                } else {
                    power = 1.;
                }

                set_grp_size(&av.tilt, av.nview, power, &mut group_size);
                def_map_opt = prepend_local(dist_def_group[d], if_local);
                non_def_map_opt = prepend_local(dist_non_def_group[d], if_local);
                automap::<false>(
                    mx,
                    sg,
                    av.nview,
                    &mut st.map_list,
                    &group_size,
                    &av.map_file_to_view,
                    av.nfile_views,
                    1,
                    &def_map_opt,
                    &non_def_map_opt,
                    num_in_view,
                    nin_thresh,
                    if_local,
                    &mut st.nmap_def_dist[d],
                    &mut st.n_ran_spec_dist[d],
                    &mut st.iv_spec_str_dist[d * NGRP..],
                    &mut st.iv_spec_end_dist[d * NGRP..],
                    &mut st.nmap_spec_dist[d * NGRP..],
                );
            }
        }
        iref1 = iref_dmag;
        if av.xyz_fixed != 0 && st.iopt_dist[d] != 0 {
            iref1 = 0;
        }
        if idist == 1 {
            //
            // analyze map list
            //
            av.map_dmag_start = *num_var_search + 1;
            analyze_maps::<false>(
                mx,
                &mut av.dmag,
                &mut av.map_dmag,
                &mut av.lin_dmag,
                &mut av.frc_dmag,
                &mut av.fixed_dmag,
                &mut fixdum,
                iflin,
                &st.map_list,
                av.nview,
                iref1,
                0,
                0.,
                b"dmag",
                var,
                var_name,
                num_var_search,
                &av.map_view_to_file,
            );
            av.map_dum_dmag = av.map_dmag_start - 2;
            //
            // provided there are at least two dmag variables, turn
            // one into a dummy - try for one 1 / 4 of the way through the views
            //
            num_dmag_var = *num_var_search + 1 - av.map_dmag_start;
            // if (numDmagVar > 1 .and. ifLocal <= 1) then
            if num_dmag_var > 1 && if_local <= -1 {
                ivl = ivdum;
                ivh = ivl + 1;
                while ivl > 0 && ivh <= av.nview && av.map_dum_dmag < av.map_dmag_start {
                    if ivl > 0 {
                        if av.map_dmag[(ivl - 1) as usize] != 0 {
                            av.map_dum_dmag = av.map_dmag[(ivl - 1) as usize];
                        }
                        ivl -= 1;
                    }
                    if ivh <= av.nview && av.map_dum_dmag < av.map_dmag_start {
                        if av.map_dmag[(ivh - 1) as usize] != 0 {
                            av.map_dum_dmag = av.map_dmag[(ivh - 1) as usize];
                        }
                        ivh += 1;
                    }
                }
                //
                // pack down the variable list in case they mean anything
                //
                for i in av.map_dum_dmag..=*num_var_search - 1 {
                    var[(i - 1) as usize] = var[i as usize];
                    var_name.copy_within(
                        (8 * i) as usize..(8 * i + 8) as usize,
                        (8 * (i - 1)) as usize,
                    );
                }
                //
                // if a variable is mapped to the dummy, change its mapping
                // to the endpoint; if it mapped past there, drop it by one to
                // pack the real variables down
                //
                for i in 1..=av.nview {
                    let v = (i - 1) as usize;
                    if av.map_dmag[v] == av.map_dum_dmag {
                        av.map_dmag[v] = *num_var_search;
                    } else if av.map_dmag[v] > av.map_dum_dmag {
                        av.map_dmag[v] -= 1;
                    }
                    if av.lin_dmag[v] == av.map_dum_dmag {
                        av.lin_dmag[v] = *num_var_search;
                    } else if av.lin_dmag[v] > av.map_dum_dmag {
                        av.lin_dmag[v] -= 1;
                    }
                }
                //
                // now set the dummy variable to the end of list and eliminate
                // that from the list
                //
                av.map_dum_dmag = *num_var_search;
                // dumDmagFac = 1. / (numDmagVar - 1.)
                av.dum_dmag_fac = -1.;
                *num_var_search -= 1;
            }
            *num_var_scaled = *num_var_search;
        } else {
            analyze_maps::<false>(
                mx,
                &mut av.skew,
                &mut av.map_skew,
                &mut av.lin_skew,
                &mut av.frc_skew,
                &mut av.fixed_skew,
                &mut fixdum,
                iflin,
                &st.map_list,
                av.nview,
                iref1,
                0,
                0.,
                b"skew",
                var,
                var_name,
                num_var_search,
                &av.map_view_to_file,
            );
        }
        power = power_skew;
    }
    //
    // get type of alpha variable set up
    //
    if if_local <= 1 {
        st.iopt_alf = 0;
        pip_get_integer(
            prepend_local("XTiltOption", if_local).as_bytes(),
            &mut st.iopt_alf,
        );
    }
    //
    // set up appropriate mapping list or read it in
    //
    av.if_any_alf = st.iopt_alf;
    iflin = 0;
    if st.iopt_alf <= 0 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = 0;
        }
    } else if st.iopt_alf == 1 {
        for i in 1..=av.nview {
            st.map_list[(i - 1) as usize] = i;
        }
    } else if st.iopt_alf == 2 {
        get_map_list(
            "XTilt",
            if_local,
            av.nview,
            &mut st.map_list,
            &mut st.map_list_xtilt,
        );
    } else {
        power = 0.;
        if st.iopt_alf == 3 {
            iflin = 1;
            power = power_alf;
        }
        set_grp_size(&av.tilt, av.nview, power, &mut group_size);
        def_map_opt = prepend_local("XTiltDefaultGrouping", if_local);
        non_def_map_opt = prepend_local("XTiltNondefaultGroup", if_local);
        automap::<false>(
            mx,
            sg,
            av.nview,
            &mut st.map_list,
            &group_size,
            &av.map_file_to_view,
            av.nfile_views,
            1,
            &def_map_opt,
            &non_def_map_opt,
            num_in_view,
            nin_thresh,
            if_local,
            &mut st.nmap_def_xtilt,
            &mut st.n_ran_spec_xtilt,
            &mut st.iv_spec_str_xtilt,
            &mut st.iv_spec_end_xtilt,
            &mut st.nmap_spec_xtilt,
        );
    }
    //
    // analyze map list - fix reference at minimum tilt
    //
    *map_alf_start = *num_var_search + 1;
    analyze_maps::<false>(
        mx,
        &mut av.alf,
        &mut av.map_alf,
        &mut av.lin_alf,
        &mut av.frc_alf,
        &mut av.fixed_alf,
        &mut fixdum,
        iflin,
        &st.map_list,
        av.nview,
        *min_tilt_ind,
        0,
        0.,
        b"Xtlt",
        var,
        var_name,
        num_var_search,
        &av.map_view_to_file,
    );
    *map_alf_end = *num_var_search;
    //
    // Add projection stretch variable if desired.
    //
    av.map_proj_stretch = 0;
    av.map_beam_tilt = 0;
    *if_bt_search = 0;
    if if_local == 0 {
        av.proj_str_rot = def_rot;
        av.proj_skew = 0.;
        av.beam_tilt = 0.;
        pip_get_boolean(b"ProjectionStretch", &mut av.map_proj_stretch);
        pip_get_integer(b"BeamTiltOption", if_bt_search);
        pip_get_float(b"FixedOrInitialBeamTilt", &mut av.beam_tilt);
        av.beam_tilt *= dtor;
        if av.map_proj_stretch > 0 {
            av.map_proj_stretch = *num_var_search + 1;
            *num_var_search += 1;
            // varName(mapProjStretch) = 'projstr '
            let o = (8 * (av.map_proj_stretch - 1)) as usize;
            var_name[o..o + 8].copy_from_slice(b"projskew");
            var[(av.map_proj_stretch - 1) as usize] = 0.;
            // var(mapProjStretch + 1) = 0.
        }
        //
        // If beam tilt option is 1, set up variable and set search back to 0
        //
        if *if_bt_search == 1 {
            *num_var_search += 1;
            av.map_beam_tilt = *num_var_search;
            let o = (8 * (av.map_beam_tilt - 1)) as usize;
            var_name[o..o + 8].copy_from_slice(b"beamtilt");
            var[(av.map_beam_tilt - 1) as usize] = av.beam_tilt;
            *if_bt_search = 0;
        }
    }
    //
    if if_local > 1 {
        return;
    }
    if av.xyz_fixed == 0
        && *num_comp_search != 0
        && (st.iopt_tilt == 1 || (st.iopt_tilt > 1 && nview_fix == 0))
    {
        let _ = stdout.write_all(
            b"\nWARNING: Only one tilt angle is fixed -\n\
WARNING: there should be two fixed tilt angles when doing compression",
        );
    }
    if st.iopt_alf != 0
        && av.if_rot_fix != -1
        && av.if_rot_fix != -2
        && *map_alf_end > *map_alf_start
    {
        let _ = stdout.write_all(
            b"\nWARNING: You are attempting to solve for x-axis tilts and rotation angle -\n\
WARNING: Results will be very unreliable; try solving for just one rotation",
        );
    }
    if st.iopt_dist[1] != 0
        && av.if_rot_fix != -1
        && av.if_rot_fix != -2
        && (*if_bt_search != 0 || av.map_beam_tilt != 0)
    {
        let _ = stdout.write_all(
            b"\nWARNING: You are trying to solve for beam tilt, skew, and rotation angle -\n\
WARNING: This is almost impossible; try solving for just one rotation",
        );
    }
    if av.if_any_alf == 0 {
        let _ = stdout.write_all(&c_format_bytes(
            "\n                        Variable mappings %s\n",
            &[CArg::Str(if if_local == 1 {
                "for lower left local area"
            } else {
                ""
            })],
        ));
        let _ = stdout.write_all(
            b" View  Rotation     Tilt  (+ incr.)      Mag       Comp      Dmag      Skew\n",
        );
    } else {
        let _ = stdout.write_all(&c_format_bytes(
            "\n                        Variable mappings %s\n",
            &[CArg::Str(if if_local == 1 {
                " for lower left local area"
            } else {
                ""
            })],
        ));
        let _ = stdout.write_all(
            b" View Rotation    Tilt  (+ incr.)   Mag      Comp     Dmag      Skew    X-Tilt\n",
        );
    }
    for iv in 1..=av.nview {
        let v = (iv - 1) as usize;
        for i in 0..=7 {
            dump[i] = *b"  fixed \0";
        }
        dump[2] = *b"        \0";
        set_dump_name(av.map_rot[v], av.lin_rot[v], -1, var_name, &mut dump[0]);
        if av.map_tilt[v] > 0 {
            set_dump_name(av.map_tilt[v], av.lin_tilt[v], -1, var_name, &mut dump[1]);
            let ti = av.tilt_inc[v];
            let ati = if ti >= 0. { ti } else { -ti };
            if ati as f64 > 5.0e-3 && (if_local == 0 || av.incr_tilt == 0) {
                // snprintf(dump[2], 9, ...): at most 8 characters plus the NUL
                let s = c_format_bytes("+%7.2f", &[CArg::Dbl((ti / dtor) as f64)]);
                let n = s.len().min(8);
                dump[2][..n].copy_from_slice(&s[..n]);
                dump[2][n] = 0;
            }
        }
        set_dump_name(av.map_gmag[v], av.lin_gmag[v], -1, var_name, &mut dump[3]);
        set_dump_name(av.map_comp[v], av.lin_comp[v], -1, var_name, &mut dump[4]);
        set_dump_name(
            av.map_dmag[v],
            av.lin_dmag[v],
            av.map_dum_dmag,
            var_name,
            &mut dump[5],
        );
        set_dump_name(av.map_skew[v], av.lin_skew[v], -1, var_name, &mut dump[6]);
        set_dump_name(av.map_alf[v], av.lin_alf[v], -1, var_name, &mut dump[7]);
        let cs = |d: &[u8; 9]| -> Vec<u8> {
            let n = d.iter().position(|&c| c == 0).unwrap_or(9);
            d[..n].to_vec()
        };
        if av.if_any_alf == 0 {
            let _ = stdout.write_all(&c_format_bytes(
                "%4d   %s   %s %s   %s   %s   %s   %s\n",
                &[
                    CArg::Int(av.map_view_to_file[v] as i64),
                    CArg::Bytes(&cs(&dump[0])),
                    CArg::Bytes(&cs(&dump[1])),
                    CArg::Bytes(&cs(&dump[2])),
                    CArg::Bytes(&cs(&dump[3])),
                    CArg::Bytes(&cs(&dump[4])),
                    CArg::Bytes(&cs(&dump[5])),
                    CArg::Bytes(&cs(&dump[6])),
                ],
            ));
        } else {
            dump[2][7] = 0x00;
            let _ = stdout.write_all(&c_format_bytes(
                "%4d  %s  %s%s  %s %s %s  %s  %s\n",
                &[
                    CArg::Int(av.map_view_to_file[v] as i64),
                    CArg::Bytes(&cs(&dump[0])),
                    CArg::Bytes(&cs(&dump[1])),
                    CArg::Bytes(&cs(&dump[2])),
                    CArg::Bytes(&cs(&dump[3])),
                    CArg::Bytes(&cs(&dump[4])),
                    CArg::Bytes(&cs(&dump[5])),
                    CArg::Bytes(&cs(&dump[6])),
                    CArg::Bytes(&cs(&dump[7])),
                ],
            ));
        }
    }
    let _ = stdout.write_all(b"\n");
}

/// Original: `setDumpName` (`input_vars.cpp:733`, file static).
///
/// `dump` is one `char[9]` row of the caller's table.  The source's `strncpy`s
/// copy from the 8-character, unterminated `varName` slots; `strncpy` stops at
/// a NUL and zero-pads, which is reproduced.
fn set_dump_name(map: i32, lin: i32, mapdum: i32, var_name: &[u8], dump: &mut [u8; 9]) {
    // strncpy(dst, src, n): copy up to the first NUL, then zero-fill to n.
    fn strncpy(dst: &mut [u8], src: &[u8], n: usize) {
        let mut i = 0;
        while i < n && src[i] != 0 {
            dst[i] = src[i];
            i += 1;
        }
        while i < n {
            dst[i] = 0;
            i += 1;
        }
    }
    if map == 0 {
        return;
    }
    if map == mapdum {
        if lin == 0 {
            *dump = *b"dummy   \0";
        } else if lin < 0 {
            *dump = *b"dum+fix \0";
        } else {
            *dump = *b"dum+    \0";
            let name = &var_name[(8 * (lin - 1)) as usize..];
            dump[4] = name[0];
            dump[5] = name[5];
            dump[6] = name[6];
            dump[7] = name[7];
        }
    } else {
        let name = &var_name[(8 * (map - 1)) as usize..];
        strncpy(&mut dump[..], name, 8);
        dump[8] = 0x00;
        if lin < 0 {
            strncpy(&mut dump[1..], &name[5..], 3);
            dump[4..9].copy_from_slice(b"+fix\0");
        } else if lin == mapdum {
            strncpy(&mut dump[1..], &name[5..], 3);
            dump[4..9].copy_from_slice(b"+dum\0");
        } else if lin > 0 {
            strncpy(&mut dump[1..], &name[5..], 3);
            dump[4] = b'+';
            let name = &var_name[(8 * (lin - 1)) as usize..];
            strncpy(&mut dump[5..], &name[5..], 3);
        }
    }
}

/// Original: `reload_vars` (`input_vars.cpp:781`).
///
/// Reloads the values of a variable `vlocal` for local fits from the global
/// values in `glb`.  `map` is the mapping from variable index to view for this
/// variable, `frc` is the fraction for linear fits, `nview` is number of
/// views, `mapStart` and `mapEnd` are the start and end of variable range for
/// this parameter, `var` is the master variable list which is also
/// initialized, `fixed` is to filled with the fixed value where the map is 0,
/// `incr` indicates that the variable is being done incrementally, and
/// `mapLocalToAll` is the mapping from the current set of views to the global
/// views.
#[allow(clippy::too_many_arguments)]
pub fn reload_vars(
    glb: &[f32],
    vlocal: &mut [f32],
    map: &[i32],
    frc: &[f32],
    nview: i32,
    map_start: i32,
    map_end: i32,
    var: &mut [f32],
    fixed: &mut f32,
    incr: i32,
    map_local_to_all: &[i32],
) {
    //
    let mut m: i32;
    let mut nsum: i32;
    let mut sum: f32;
    //
    // first just reload all of the local parameters from the global
    // fixed will have the right value for any case where there is ONE
    // fixed parameter
    //
    for i in 1..=nview {
        m = map_local_to_all[(i - 1) as usize];
        vlocal[(i - 1) as usize] = glb[(m - 1) as usize];
        if map[(i - 1) as usize] == 0 {
            *fixed = glb[(m - 1) as usize];
        }
    }
    //
    // if doing incremental, set all the var's to 0 and return
    //
    if incr != 0 {
        for m in map_start..=map_end {
            var[(m - 1) as usize] = 0.;
        }
        *fixed = 0.;
        return;
    }
    //
    // next set the value of each variable as the average value of all
    // the parameters that map solely to it
    //
    for m in map_start..=map_end {
        sum = 0.;
        nsum = 0;
        for i in 1..=nview {
            if map[(i - 1) as usize] == m && frc[(i - 1) as usize] == 1. {
                sum += glb[(map_local_to_all[(i - 1) as usize] - 1) as usize];
                nsum += 1;
            }
        }
        if nsum == 0 {
            let buf = c_format("Nothing mapped to variable %d", &[CArg::Int(m as i64)]);
            error_exit::<false>(&buf, 0);
        }
        var[(m - 1) as usize] = sum / nsum as f32;
    }
}

/// Original: `expandLocalToAll` (`input_vars.cpp:839`).
///
/// Takes a variable from a local solution with some omitted views and expands
/// the variable to the original global views, filling in the empty spots by
/// interpolation, or copying at the ends.  `var` is the variable array,
/// dimensioned `VAR(IDIM,*)`, and `index` is the index for the first
/// dimension; `nvLocal` is the local # of views, `nview` is the global # of
/// views, `mapa2l` is the mapping from all to local views.
///
/// `varLast` is uninitialised in the source and read only when no view at or
/// above the current one is mapped and none below it is either; it starts at
/// 0 here.
pub fn expand_local_to_all(
    var: &mut [f32],
    idim: i32,
    index: i32,
    nv_local: i32,
    nview: i32,
    mapa2l: &[i32],
) {
    let mut iout_last: i32;
    let mut iout_next: i32;
    let mut var_last: f32 = 0.;
    let mut f: f32;
    //
    if nv_local == nview {
        return;
    }
    iout_last = -1;
    let mut iout = nview;
    while iout >= 1 {
        if mapa2l[(iout - 1) as usize] > 0 {
            var_last = var[((mapa2l[(iout - 1) as usize] - 1) * idim + index - 1) as usize];
            var[((iout - 1) * idim + index - 1) as usize] = var_last;
            iout_last = iout;
        } else {
            //
            // Find the next one down
            iout_next = iout - 1;
            while iout_next > 1
                && mapa2l[((if 1 > iout_next { 1 } else { iout_next }) - 1) as usize] == 0
            {
                iout_next -= 1;
            }
            //
            // If there is no next one (at the bottom), use the last one
            // If there is no last one, use the next one
            // if there are both, interpolate
            if iout_next == 0
                || mapa2l[((if 1 > iout_next { 1 } else { iout_next }) - 1) as usize] == 0
            {
                var[((iout - 1) * idim + index - 1) as usize] = var_last;
            } else if iout_last < 1 {
                var[((iout - 1) * idim + index - 1) as usize] =
                    var[((mapa2l[(iout_next - 1) as usize] - 1) * idim + index - 1) as usize];
            } else {
                f = (iout - iout_next) as f32 / (iout_last - iout_next) as f32;
                var[((iout - 1) * idim + index - 1) as usize] = f * var_last
                    + (1. - f)
                        * var[((mapa2l[(iout_next - 1) as usize] - 1) * idim + index - 1) as usize];
                // print *,iout, ioutLast, ioutNext, mapa2l(ioutNext), mapa2l(ioutNext), &
                // varLast, var(index, mapa2l(ioutNext)), var(index, iout)
            }
        }
        iout -= 1;
    }
}

/// Original: `nearest_view` (`input_vars.cpp:882`).
///
/// Finds the nearest existing view to the given file view number.
pub fn nearest_view(av: &AlignVariables, iview: i32) -> i32 {
    let mut ivdel: i32;
    let mut nearest: i32;
    if iview > av.nfile_views || iview <= 0 {
        let buf = c_format(
            "View %d is beyond the known range of the file",
            &[CArg::Int(iview as i64)],
        );
        error_exit::<false>(&buf, 0);
    }
    nearest = av.map_file_to_view[(iview - 1) as usize];
    ivdel = 1;
    while nearest == 0 {
        if iview - ivdel > 0 && av.map_file_to_view[(iview - ivdel - 1) as usize] > 0 {
            nearest = av.map_file_to_view[(iview - ivdel - 1) as usize];
        }
        if iview + ivdel <= av.nfile_views && av.map_file_to_view[(iview + ivdel - 1) as usize] > 0
        {
            nearest = av.map_file_to_view[(iview + ivdel - 1) as usize];
        }
        ivdel += 1;
    }
    nearest
}

/// Original: `prependLocal` (`input_vars.cpp:908`, file static).
///
/// The source returns its `static char buf[64]`; every caller consumes or
/// copies it before the next call, so the string is returned by value.
fn prepend_local(option: &str, if_local: i32) -> String {
    if if_local != 0 {
        c_format("Local%s", &[CArg::Str(option)])
    } else {
        option.to_string()
    }
}

/// Original: `getMapList` (`input_vars.cpp:925`, file static).
///
/// Gets a specific mapping list for the variable named varName.  `option` is a
/// base name for the option, `nview` has the usual meaning, `mapList` is
/// returned with the mappings.  If `ifLocal` is 0 or 1 the map list is copied
/// into `mapSave`; if it is 2 then `mapList` is simply copied from `mapSave`.
fn get_map_list(
    option: &str,
    if_local: i32,
    nview: i32,
    map_list: &mut [i32],
    map_save: &mut [i32],
) {
    let mut num_entry: i32;
    let mut num_tot: i32;
    let mut num_got: i32;

    if if_local > 1 {
        for i in 1..=nview {
            map_list[(i - 1) as usize] = map_save[(i - 1) as usize];
        }
        return;
    }
    let buf = c_format("%sMapping", &[CArg::Str(option)]);
    //
    let map_option = prepend_local(&buf, if_local);
    num_entry = 0;
    pip_number_of_entries(map_option.as_bytes(), &mut num_entry);
    num_tot = 0;
    for _i in 1..=num_entry {
        num_got = 0;
        pip_get_integer_array(
            map_option.as_bytes(),
            &mut map_list[num_tot as usize..],
            &mut num_got,
            nview - num_tot,
        );
        num_tot += num_got;
    }
    if num_tot < nview {
        let buf = c_format(
            "Not enough mapping values entered with %s",
            &[CArg::Str(&map_option)],
        );
        error_exit::<false>(&buf, 0);
    }
    for i in 1..=nview {
        map_save[(i - 1) as usize] = map_list[(i - 1) as usize];
    }
}
