//! Translation of `IMOD/flib/beadtrack/proc_vars.cpp` — gets the variable
//! specifications and sets up mappings.
//!
//! Compiled into `beadtrack` only, whose `utilfuncs`/`map_vars` are the
//! `-DBEADTRACK` build, so the `map_vars` functions are called with
//! `BEADTRACK = true`.
//!
//! The file-scope `static TiltControl *tc; static MapSepGroups *sg; static
//! AlignVariables *av; static ArrayMaxes *mx;` (`proc_vars.cpp:19-22`), set by
//! `procVarsSetPointers`, are parameters here (`alivar.rs`, `tltcntrl.rs`).
//!
//! # Representation
//!
//! `mapList`, `grpSize` and `varName` are `B3DMALLOC`ed and uninitialised in
//! the source; they are zeroed `Vec`s here (`NATIVE.md` §4).  `makeMapList`
//! writes every element of `mapList` that is read, `setGrpSize` every element
//! of `grpSize`, and `analyze_maps` every `varName` slot it later reads, so
//! nothing observable depends on the difference.  The allocation-failure
//! `exitError` cannot occur.
//!
//! `makeMapList` is called with `mapList` both as its output list and as its
//! `numInView` argument, with `ninThresh = 0`, where `numInView` is never read
//! (`map_vars.rs`); an empty slice stands in for the aliased argument.

use super::tltcntrl::TiltControl;
use crate::imod::flib::tiltalign::alivar::AlignVariables;
use crate::imod::flib::tiltalign::arraymaxes::ArrayMaxes;
use crate::imod::flib::tiltalign::map_vars::{
    analyze_maps, make_map_list, map_separate_group, set_grp_size,
};
use crate::imod::flib::tiltalign::mapsepgroups::MapSepGroups;

/// Original: `procVarsSetPointers` (`proc_vars.cpp:24`).
///
/// The source stores the four pointers in file-scope statics; [`proc_vars`]
/// takes them as parameters instead, so there is nothing to store.
pub fn proc_vars_set_pointers(
    tlcnt: &mut TiltControl,
    sg_in: &mut MapSepGroups,
    av_in: &mut AlignVariables,
    mx_in: &mut ArrayMaxes,
) {
    let _ = (tlcnt, sg_in, av_in, mx_in);
}

/// Original: `proc_vars` (`proc_vars.cpp:37`).
///
/// proc_vars gets the specifications for the geometric variables rotation,
/// tilt, mag, etc.  It fills the VAR array and the different variable and
/// mapping arrays.
#[allow(clippy::too_many_arguments)]
pub fn proc_vars(
    tc: &TiltControl,
    sg: &mut MapSepGroups,
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    if_map_tilt: i32,
    iref_tilt: i32,
    var: &mut [f32],
    nvar_search: &mut i32,
) {
    //

    let mut map_list: Vec<i32>;
    let mut rot_start: f32;
    let def_rot: f32;
    let mut power: f32;
    let mut fix_dum: f32 = 0.;
    let mut grp_size: Vec<f32>;
    let _nvar_angles: i32;
    let mut iflin: i32;
    let iref1: i32;
    let dtor: f32 = 0.01745329252_f64 as f32;
    let mut var_name: Vec<u8>;
    map_list = vec![0; mx.max_view as usize];
    grp_size = vec![0.; mx.max_view as usize];
    var_name = vec![0; (8 * 4 * mx.max_view) as usize];
    //
    // Remap the separate group information for the current views
    //
    let mv = mx.max_view as usize;
    for iv in 0..sg.num_separate_groups as usize {
        sg.num_sep_in_group[iv] = tc.nsep_in_grp_in[iv];
        for i in 0..sg.num_sep_in_group[iv] as usize {
            sg.iviews_in_group[iv * mv + i] = tc.ivsep_in[iv * mv + i];
        }
        map_separate_group::<true>(
            &mut sg.iviews_in_group[iv * mv..],
            &mut sg.num_sep_in_group[iv],
            &av.map_file_to_view,
            av.nfile_views,
        );
    }
    //
    // Use the grouping to get the mapping
    //
    iflin = 1;
    set_grp_size(&av.tilt, av.nview, 0., &mut grp_size);
    make_map_list(
        mx,
        sg,
        av.nview,
        &mut map_list,
        &grp_size,
        &av.map_file_to_view,
        av.nfile_views,
        tc.nmap_rot,
        &tc.iv_spec_str_rot,
        &tc.iv_spec_end_rot,
        &tc.nmap_spec_rot,
        tc.n_ran_spec_rot,
        &[],
        0,
    );
    //
    // Set up default rotation value so fixed rotation will be correct
    //
    *nvar_search = 0;
    iref1 = av.if_rot_fix;
    rot_start = 0.;
    if iref1 > 0 {
        rot_start = tc.rot_orig[(av.map_view_to_file[(iref1 - 1) as usize] - 1) as usize];
    }
    def_rot = rot_start * dtor;
    //
    // get the variables allocated
    //
    analyze_maps::<true>(
        mx,
        &mut av.rot,
        &mut av.map_rot,
        &mut av.lin_rot,
        &mut av.frc_rot,
        &mut av.fixed_rot,
        &mut fix_dum,
        iflin,
        &map_list,
        av.nview,
        iref1,
        0,
        def_rot,
        b"rot ",
        var,
        &mut var_name,
        nvar_search,
        &av.map_view_to_file,
    );
    //
    // reload the old rotation values by looking at all mappings that are
    // not linear combos
    //
    for i in 0..av.nview as usize {
        av.rot[i] = tc.rot_orig[(av.map_view_to_file[i] - 1) as usize] * dtor;
        if av.map_rot[i] > 0 && av.lin_rot[i] == 0 {
            var[(av.map_rot[i] - 1) as usize] = av.rot[i];
        }
    }
    //
    // set up for fixed tilt angles
    //
    for i in 0..av.nview as usize {
        map_list[i] = 0;
    }
    iflin = 0;
    power = 0.;
    if if_map_tilt != 0 {
        if tc.nmap_tilt > 1 {
            iflin = 1;
            power = 1.;
        }
        set_grp_size(&av.tilt, av.nview, power, &mut grp_size);
        make_map_list(
            mx,
            sg,
            av.nview,
            &mut map_list,
            &grp_size,
            &av.map_file_to_view,
            av.nfile_views,
            tc.nmap_tilt,
            &tc.iv_spec_str_tilt,
            &tc.iv_spec_end_tilt,
            &tc.nmap_spec_tilt,
            tc.n_ran_spec_tilt,
            &[],
            0,
        );
    }
    //
    analyze_maps::<true>(
        mx,
        &mut av.tilt,
        &mut av.map_tilt,
        &mut av.lin_tilt,
        &mut av.frc_tilt,
        &mut av.fixed_tilt,
        &mut av.fixed_tilt2,
        iflin,
        &map_list,
        av.nview,
        iref_tilt,
        0,
        -999.,
        b"tilt",
        var,
        &mut var_name,
        nvar_search,
        &av.map_view_to_file,
    );
    //
    // set tiltInc so it will give back the right tilt for whatever
    // mapping or interpolation scheme is used
    //
    for iv in 0..av.nview as usize {
        av.tilt_inc[iv] = 0.;
        if av.map_tilt[iv] != 0 {
            let frc = av.frc_tilt[iv];
            let vm = var[(av.map_tilt[iv] - 1) as usize];
            if av.lin_tilt[iv] > 0 {
                av.tilt_inc[iv] = (av.tilt[iv] as f64
                    - ((frc * vm) as f64
                        + (1. - frc as f64) * var[(av.lin_tilt[iv] - 1) as usize] as f64))
                    as f32;
            } else if av.lin_tilt[iv] == -1 {
                av.tilt_inc[iv] = (av.tilt[iv] as f64
                    - ((frc * vm) as f64 + (1. - frc as f64) * av.fixed_tilt as f64))
                    as f32;
            } else if av.lin_tilt[iv] == -2 {
                av.tilt_inc[iv] = (av.tilt[iv] as f64
                    - ((frc * vm) as f64 + (1. - frc as f64) * av.fixed_tilt2 as f64))
                    as f32;
            } else {
                av.tilt_inc[iv] = av.tilt[iv] - vm;
            }
        }
    }
    //
    // Now reload the previous values and put them into vars
    //
    for i in 0..av.nview as usize {
        if av.map_tilt[i] > 0 {
            av.tilt[i] = dtor * tc.tilt_orig[(av.map_view_to_file[i] - 1) as usize];
        }
        let ti = av.tilt_inc[i];
        if av.map_tilt[i] > 0
            && av.lin_tilt[i] == 0
            && ((if ti >= 0. { ti } else { -ti }) as f64) < 1.0e-6
        {
            var[(av.map_tilt[i] - 1) as usize] = av.tilt[i];
        }
    }
    _nvar_angles = *nvar_search;
    //
    // Set up the mag mapList
    //
    iflin = 1;
    set_grp_size(&av.tilt, av.nview, 0., &mut grp_size);
    make_map_list(
        mx,
        sg,
        av.nview,
        &mut map_list,
        &grp_size,
        &av.map_file_to_view,
        av.nfile_views,
        tc.nmap_mag,
        &tc.iv_spec_str_mag,
        &tc.iv_spec_end_mag,
        &tc.nmap_spec_mag,
        tc.n_ran_spec_mag,
        &[],
        0,
    );
    //
    // Set up mag variables
    //
    analyze_maps::<true>(
        mx,
        &mut av.gmag,
        &mut av.map_gmag,
        &mut av.lin_gmag,
        &mut av.frc_gmag,
        &mut av.fixed_gmag,
        &mut fix_dum,
        iflin,
        &map_list,
        av.nview,
        iref_tilt,
        0,
        1.,
        b"mag ",
        var,
        &mut var_name,
        nvar_search,
        &av.map_view_to_file,
    );
    //
    // Reload values and stuff them into vars by looking at all mappings
    // that are not linear combos
    //
    for i in 0..av.nview as usize {
        av.gmag[i] = tc.gmag_orig[(av.map_view_to_file[i] - 1) as usize];
        if av.map_gmag[i] > 0 && av.lin_gmag[i] == 0 {
            var[(av.map_gmag[i] - 1) as usize] = av.gmag[i];
        }
    }
    //
    // This should be done once before calling
    //
    av.map_proj_stretch = 0;
    av.proj_str_rot = def_rot;
    av.proj_skew = 0.;
    av.map_beam_tilt = 0;
    av.beam_tilt = 0.;
    // Args assigned to: nvarSearch
    drop(map_list);
    drop(var_name);
    drop(grp_size);
}
