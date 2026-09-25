//! Translation of `IMOD/flib/tiltalign/map_vars.cpp` — routines for variable
//! mapping.
//!
//! # File-scope statics
//!
//! `static AlignVariables *av; static ArrayMaxes *mx; static MapSepGroups *sg;`
//! (`map_vars.cpp:28-30`), set by `mapVarsSetPointers`, are parameters here
//! (`alivar.rs`): `&ArrayMaxes` where a function reads `mx->maxView`,
//! `&MapSepGroups` where it reads the separate groups, and `&AlignVariables`
//! for `inputSeparateGroups`' `av->nview`.  None of them is written through the
//! pointer in this unit.
//!
//! # One source, two builds
//!
//! Compiled into both `tiltalign` and `beadtrack` (`-DBEADTRACK`), with no
//! conditional code of its own.  Functions that call `errorExit` take
//! `const BEADTRACK: bool` and forward it to [`error_exit`] (`utilfuncs.rs`);
//! every call here passes `ifLocal = 0`, which exits in both builds, so the
//! parameter selects nothing observable in this unit.
//!
//! # Representation
//!
//! `char *varName` is the caller's array of 8-character names, `&mut [u8]`;
//! option names are `&str`.  A `NULL` `numInView` (callers with
//! `ninThresh == 0`, where `groupList` never reads it) is an empty slice.
//! `B3DMALLOC`'d `mapVarNum` is a zeroed `Vec`; `IntVec inran` is a
//! `std::vector<int>`, zeroed in the source too.

use super::alivar::AlignVariables;
use super::arraymaxes::{ArrayMaxes, MAXGRP};
use super::mapsepgroups::MapSepGroups;
use super::utilfuncs::error_exit;
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use crate::imod::libcfshr::parse_params::{
    pip_get_integer, pip_get_string, pip_get_three_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::parselist::{ParseListError, parselist};

/// Original: `mapVarsSetPointers` (`map_vars.cpp:38`).
///
/// The source stores the three pointers in file-scope statics; the functions
/// of this module take them as parameters instead, so there is nothing to
/// store.
pub fn map_vars_set_pointers(
    av_in: &mut AlignVariables,
    mx_in: &mut ArrayMaxes,
    sg_in: &mut MapSepGroups,
) {
    let _ = (av_in, mx_in, sg_in);
}

/// Original: `analyze_maps` (`map_vars.cpp:67`).
///
/// Takes a map list and other information about how to set up a mapping and
/// allocates variables, and mappings for getting values from variables.
/// `gmag` is the variable array, one per view; `mapGmag` is filled with
/// mappings to solution variables in `var`; `linGmag` with the mapping to the
/// second variable for linear mapping; `fracGmag` with the fractions for linear
/// mapping; `fixedGmag`/`fixedGmag2` are the fixed values ramped to for the
/// first/second reference; `iflin` is 1 for linear mapping, 0 for block;
/// `irefTilt`/`iref2` are reference views set to zero (0 if none); `defval` is a
/// default value to fill the array with, if not -999; `name` is a 4-character
/// variable name; `varName` is an array of 8-character strings filled with map
/// names; `numVarSearch` is the number of variables in `var`, updated.
///
/// `lastAdded` is uninitialised in the source and only read once
/// `numMagSearch > 0`; `ivvar` is only read when `ivfix != 0`, where it is set.
pub fn analyze_maps<const BEADTRACK: bool>(
    mx: &ArrayMaxes,
    gmag: &mut [f32],
    map_gmag: &mut [i32],
    lin_gmag: &mut [i32],
    frac_gmag: &mut [f32],
    fixed_gmag: &mut f32,
    fixed_gmag2: &mut f32,
    iflin: i32,
    map_list: &[i32],
    nview: i32,
    iref_tilt: i32,
    iref2: i32,
    defval: f32,
    name: &[u8],
    var: &mut [f32],
    var_name: &mut [u8],
    num_var_search: &mut i32,
    map_view_to_file: &[i32],
) {
    //
    let mut num_mag_search: i32;
    let mut mapref1: i32;
    let mut mapref2: i32;
    let mut imap: i32;
    let mut list_max: i32;
    let mut ivmin: i32 = 0;
    let mut ivmax: i32 = 0;
    let mut num_in_group: i32 = 0;
    let mut next_min_view: i32 = 0;
    let mut next_max_view: i32 = 0;
    let mut num_in_next: i32 = 0;
    let mut nadd: i32;
    let mut ivadd: i32;
    let mut ivfix: i32;
    let mut linval: i32;
    let mut ivvar: i32 = 0;
    let mut last_added: i32 = 0;
    let mut map_var_num = vec![0i32; mx.max_view as usize];
    let allocated = true;
    if !allocated {
        error_exit::<BEADTRACK>("Allocating array for mapping", 0);
    }
    //
    num_mag_search = 0;
    for iv in 1..=nview {
        if defval as f64 != -999. {
            gmag[(iv - 1) as usize] = defval;
        }
    }
    mapref1 = -999;
    if iref_tilt > 0 {
        mapref1 = map_list[(iref_tilt - 1) as usize];
    }
    mapref2 = -999;
    if iref2 > 0 {
        mapref2 = map_list[(iref2 - 1) as usize];
    }
    if iflin <= 0 {
        for iv in 1..=nview {
            let v = (iv - 1) as usize;
            map_gmag[v] = 0;
            lin_gmag[v] = 0;
            frac_gmag[v] = 1.;
            if map_list[v] != mapref1 && map_list[v] != mapref2 {
                //
                // if value is same as for minimum tilt, all set for fixed mag
                // otherwise see if this var # is already encountered
                //
                imap = 0;
                for it in 1..=num_mag_search {
                    if map_list[v] == map_var_num[(it - 1) as usize] {
                        imap = it;
                    }
                }
                if imap == 0 {
                    //
                    // if not, need to add another mag variable
                    //
                    num_mag_search = num_mag_search + 1;
                    map_var_num[(num_mag_search - 1) as usize] = map_list[v];
                    var[(*num_var_search + num_mag_search - 1) as usize] = gmag[v];
                    fill_var_name(
                        &mut var_name[(8 * (*num_var_search + num_mag_search - 1)) as usize..],
                        name,
                        map_view_to_file[v],
                    );
                    imap = num_mag_search;
                }
                //
                // now map the mag to the variable
                //
                map_gmag[v] = *num_var_search + imap;
            }
        }
    } else {
        //
        // LINEAR MAPPING OPTION
        //
        list_max = 0;
        for iv in 1..=nview {
            list_max = if list_max > map_list[(iv - 1) as usize] {
                list_max
            } else {
                map_list[(iv - 1) as usize]
            };
        }
        //
        // go through the variable groupings
        //
        imap = 1;
        while imap <= list_max {
            group_min_max(
                map_list,
                nview,
                imap,
                &mut ivmin,
                &mut ivmax,
                &mut num_in_group,
            );
            //
            // if any in this group:
            //
            if num_in_group > 0 {
                group_min_max(
                    map_list,
                    nview,
                    imap + 1,
                    &mut next_min_view,
                    &mut next_max_view,
                    &mut num_in_next,
                );
                //
                // if any in next group, the min view in that group is the
                // endpoint for interpolation, otherwise the max view in this
                // group is the endpoint
                //
                nadd = 2;
                ivadd = ivmin;
                ivfix = 0;
                linval = -1;
                if num_in_next == 0 {
                    next_min_view = ivmax;
                    //
                    // ending and only two in group, add only one
                    //
                    if num_in_group <= 2 {
                        nadd = 1;
                    }
                    //
                    // but if this one is fixed group, add end to stack unless
                    // only 2, in which case add nothing
                    //
                    if imap == mapref1 || imap == mapref2 {
                        map_gmag[(ivmin - 1) as usize] = 0;
                        ivadd = next_min_view;
                        ivfix = ivmin;
                        ivvar = next_min_view;
                        if num_in_group <= 2 {
                            nadd = 0;
                        }
                        if imap == mapref2 {
                            linval = -2;
                        }
                    }
                } else {
                    //
                    // if continuing and this one is fixed, add next start to list
                    //
                    if imap == mapref1 || imap == mapref2 {
                        map_gmag[(ivmin - 1) as usize] = 0;
                        ivadd = next_min_view;
                        //
                        // but if next one is fixed also, use this one's end instead
                        //
                        if imap + 1 == mapref1 || imap + 1 == mapref2 {
                            ivadd = ivmax;
                        }
                        nadd = 1;
                        ivfix = ivmin;
                        ivvar = ivadd;
                        if imap == mapref2 {
                            linval = -2;
                        }
                        //
                        // if continuing and next one is fixed, add this start to list
                        //
                    } else if imap + 1 == mapref1 || imap + 1 == mapref2 {
                        nadd = 1;
                        ivfix = next_min_view;
                        ivvar = ivmin;
                        if imap + 1 == mapref2 {
                            linval = -2;
                        }
                    }
                }
                //
                // add the starting and ending views for interpolation as
                // variables, if they are not already on the list
                //
                for _iadd in 1..=nadd {
                    if num_mag_search == 0 || last_added != ivadd {
                        num_mag_search = num_mag_search + 1;
                        var[(*num_var_search + num_mag_search - 1) as usize] =
                            gmag[(ivadd - 1) as usize];
                        fill_var_name(
                            &mut var_name[(8 * (*num_var_search + num_mag_search - 1)) as usize..],
                            name,
                            map_view_to_file[(ivadd - 1) as usize],
                        );
                        last_added = ivadd;
                        map_gmag[(ivadd - 1) as usize] = *num_var_search + num_mag_search;
                        lin_gmag[(ivadd - 1) as usize] = 0;
                        frac_gmag[(ivadd - 1) as usize] = 1.;
                    }
                    ivadd = next_min_view;
                }
                //
                //
                // now go through rest of views (excluding endpoints), and for
                // each one in the group, map to next to last variable and set
                // fraction to the fraction of the distance in view number from
                // the starting to the ending point
                //
                if ivfix == 0 && (num_in_next > 0 || num_in_group > 2) {
                    for iv in ivmin + 1..=ivmax {
                        if iv != next_min_view && map_list[(iv - 1) as usize] == imap {
                            map_gmag[(iv - 1) as usize] = map_gmag[(ivmin - 1) as usize];
                            lin_gmag[(iv - 1) as usize] = map_gmag[(next_min_view - 1) as usize];
                            frac_gmag[(iv - 1) as usize] =
                                (next_min_view - iv) as f32 / (next_min_view - ivmin) as f32;
                        }
                    }
                } else if ivfix == 0 {
                    //
                    // ending, not fixed, and only two in group: map 2nd to first
                    //
                    map_gmag[(ivmax - 1) as usize] = map_gmag[(ivmin - 1) as usize];
                    lin_gmag[(ivmax - 1) as usize] = 0;
                    frac_gmag[(ivmax - 1) as usize] = 1.;
                } else if num_in_next == 0 && num_in_group <= 2 {
                    //
                    // ending in a fixed group of 2
                    //
                    map_gmag[(ivmax - 1) as usize] = 0;
                } else {
                    //
                    // linear ramp between fixed and variable
                    //
                    for iv in ivmin..=ivmax {
                        if iv != ivfix && iv != ivvar && map_list[(iv - 1) as usize] == imap {
                            map_gmag[(iv - 1) as usize] = map_gmag[(ivvar - 1) as usize];
                            lin_gmag[(iv - 1) as usize] = linval;
                            frac_gmag[(iv - 1) as usize] =
                                (ivfix - iv) as f32 / (ivfix - ivvar) as f32;
                        }
                    }
                }
                //
                // set the fixed value that is being ramped to
                //
                if imap == mapref1 {
                    *fixed_gmag = gmag[(ivfix - 1) as usize];
                }
                if imap == mapref2 {
                    *fixed_gmag2 = gmag[(ivfix - 1) as usize];
                }
            }
            imap += 1;
        }
    }
    *num_var_search += num_mag_search;
}

/// Original: `fillVarName` (`map_vars.cpp:274`, file static).
///
/// `sprintf(buf, "%4d", view)` and the first four characters of it, after the
/// four of `name`.
fn fill_var_name(var_name: &mut [u8], name: &[u8], view: i32) {
    let buf = c_format("%4d", &[CArg::Int(view as i64)]).into_bytes();
    for ind in 0..4 {
        var_name[ind] = name[ind];
        var_name[ind + 4] = buf.get(ind).copied().unwrap_or(0);
    }
}

/// Original: `automap` (`map_vars.cpp:285`).
///
/// `nRanSpecIn` is by value in the source, and `inputGroupings` sets this
/// function's copy of it (its parameter is `int &`), which `makeMapList` then
/// reads; the caller's variable is not changed.
pub fn automap<const BEADTRACK: bool>(
    mx: &ArrayMaxes,
    sg: &MapSepGroups,
    nview: i32,
    map_list: &mut [i32],
    group_size: &[f32],
    map_file_to_view: &[i32],
    nfile_views: i32,
    if_required: i32,
    default_option: &str,
    non_default_option: &str,
    num_in_view: &[i32],
    nin_thresh: i32,
    if_local: i32,
    nmap_def: &mut i32,
    mut n_ran_spec_in: i32,
    iv_spec_str_in: &mut [i32],
    iv_spec_end_in: &mut [i32],
    nmap_spec: &mut [i32],
) {
    let max_groups = MAXGRP;
    if if_local <= 1 {
        input_groupings::<BEADTRACK>(
            nfile_views,
            if_required,
            default_option,
            non_default_option,
            nmap_def,
            iv_spec_str_in,
            iv_spec_end_in,
            nmap_spec,
            &mut n_ran_spec_in,
            max_groups,
        );
    }
    make_map_list(
        mx,
        sg,
        nview,
        map_list,
        group_size,
        map_file_to_view,
        nfile_views,
        *nmap_def,
        iv_spec_str_in,
        iv_spec_end_in,
        nmap_spec,
        n_ran_spec_in,
        num_in_view,
        nin_thresh,
    );
}

/// Original: `inputSeparateGroups` (`map_vars.cpp:304`).
///
/// Input the separate groups and check that not every view is in each one.
///
/// `parselist` returns `NULL` with a count of 0 for a string holding no
/// numbers, which the source reports as an allocation failure; the translated
/// `parselist` returns an empty list there, mapped back to that branch.
pub fn input_separate_groups<const BEADTRACK: bool>(
    av: &AlignVariables,
    mx: &ArrayMaxes,
    num_separate_groups: &mut i32,
    num_sep_in_group: &mut [i32],
    iviews_in_group: &mut [i32],
) {
    let mut sep_string: Vec<u8> = Vec::new();
    let mut ierr: i32;

    ierr = pip_number_of_entries(b"SeparateGroup", num_separate_groups);
    let _ = ierr;
    if *num_separate_groups > MAXGRP {
        error_exit::<BEADTRACK>("Too many separate groups for arrays", 0);
    }
    for ig in 1..=*num_separate_groups {
        let g = (ig - 1) as usize;
        ierr = pip_get_string(b"SeparateGroup", &mut sep_string);
        let view_list: Option<Vec<i32>> = match parselist(&String::from_utf8_lossy(&sep_string)) {
            Ok(list) => {
                num_sep_in_group[g] = list.len() as i32;
                if list.is_empty() { None } else { Some(list) }
            }
            Err(ParseListError::LeadingSlash) => {
                num_sep_in_group[g] = -1;
                None
            }
            Err(ParseListError::InvalidCharacter) => {
                num_sep_in_group[g] = -3;
                None
            }
        };
        let Some(view_list) = view_list else {
            if num_sep_in_group[g] == -1 {
                error_exit::<BEADTRACK>("A SeparateGroup entry of \"/\" is not allowed", 0);
            } else {
                error_exit::<BEADTRACK>("Allocating list for SeparateGroup entry", 0);
            }
            return;
        };
        for i in 0..num_sep_in_group[g] {
            iviews_in_group[(ig - 1) as usize * mx.max_view as usize + i as usize] =
                view_list[i as usize];
        }

        // Check each view to see if it is in the group
        for iv in 1..=av.nview {
            ierr = 0;
            for i in 1..=num_sep_in_group[g] {
                if iviews_in_group[((ig - 1) * mx.max_view + i - 1) as usize] == iv {
                    ierr = 1;
                    break;
                }
            }

            // We get here with a 1 if it is in, so if there is ever a view not in the group,
            // the group is ok
            if ierr == 0 {
                break;
            }
        }
        if ierr > 0 {
            error_exit::<BEADTRACK>("An entry for a separate group contains all views", 0);
        }
    }
}

/// Original: `inputGroupings` (`map_vars.cpp:351`).
///
/// Inputs the groupings for one variable.  The messages are formatted into a
/// `char buf[100]` in the source; none of the translated callers' option names
/// makes them overflow it.
pub fn input_groupings<const BEADTRACK: bool>(
    n_file_views: i32,
    if_required: i32,
    default_option: &str,
    non_default_option: &str,
    nmap_def: &mut i32,
    iv_spec_str: &mut [i32],
    iv_spec_end: &mut [i32],
    nmap_spec: &mut [i32],
    n_ran_spec: &mut i32,
    max_groups: i32,
) {
    //
    let mut ivstr: i32 = 0;
    let mut ivend: i32 = 0;
    //
    *n_ran_spec = 0;

    if pip_get_integer(default_option.as_bytes(), nmap_def) > 0 {
        if if_required == 0 {
            return;
        }
        let buf = c_format("Option %s must be entered", &[CArg::Str(default_option)]);
        error_exit::<BEADTRACK>(&buf, 0);
    }
    pip_number_of_entries(non_default_option.as_bytes(), n_ran_spec);
    //
    if *n_ran_spec > max_groups {
        error_exit::<BEADTRACK>("Too many nondefault groupings for arrays", 0);
    }
    //
    for iran in 1..=*n_ran_spec {
        pip_get_three_integers(
            non_default_option.as_bytes(),
            &mut ivstr,
            &mut ivend,
            &mut nmap_spec[(iran - 1) as usize],
        );
        if ivstr > ivend {
            let buf = c_format(
                "Start of range (%d) is past end of range (%d) for entry to %s",
                &[
                    CArg::Int(ivstr as i64),
                    CArg::Int(ivend as i64),
                    CArg::Str(non_default_option),
                ],
            );
            error_exit::<BEADTRACK>(&buf, 0);
        }
        if ivstr < 0 || ivstr > n_file_views || ivend < 0 || ivend > n_file_views {
            let buf = c_format(
                "Starting or ending view of range (%d or %d) in not in file for entry to %s",
                &[
                    CArg::Int(ivstr as i64),
                    CArg::Int(ivend as i64),
                    CArg::Str(non_default_option),
                ],
            );
            error_exit::<BEADTRACK>(&buf, 0);
        }
        iv_spec_str[(iran - 1) as usize] = ivstr;
        iv_spec_end[(iran - 1) as usize] = ivend;
    }
}

/// Original: `makeMapList` (`map_vars.cpp:402`).
///
/// Makes a mapping list, consisting of group number for each view returned in
/// `mapList`.  `groupSize` is an array of relative group sizes for the views,
/// `mapFileToView` maps file views to current views, `nmapDef` is the default
/// group size, `nRanSpecIn` is the number of special ranges, `ivSpecStrIn` and
/// `ivSpecEndIn` hold the starting and ending file views in each range and
/// `nmapSpecIn` the group size in each range.  `numInView` has the number of
/// points in each view and `ninThresh` the threshold number required for
/// counting a view toward the number needed to form a group; if `ninThresh`
/// is zero, `numInView` is ignored.
///
/// The three local range arrays are `MAXGRP` long in the source; a list that
/// splits into more ranges than that overruns them there and panics here.
pub fn make_map_list(
    mx: &ArrayMaxes,
    sg: &MapSepGroups,
    nview: i32,
    map_list: &mut [i32],
    group_size: &[f32],
    map_file_to_view: &[i32],
    n_file_views: i32,
    nmap_def: i32,
    iv_spec_str_in: &[i32],
    iv_spec_end_in: &[i32],
    nmap_spec_in: &[i32],
    n_ran_spec_in: i32,
    num_in_view: &[i32],
    nin_thresh: i32,
) {
    let _ = n_file_views;
    let mut iv_spec_str = [0i32; MAXGRP as usize];
    let mut iv_spec_end = [0i32; MAXGRP as usize];
    let mut nmap_spec = [0i32; MAXGRP as usize];
    let mut inran = vec![0i32; mx.max_view as usize];
    //
    let mut nran_spec: i32;
    let mut ivstr: i32;
    let mut ivend: i32;
    let mut nran: i32;
    let mut ir: i32;
    let mut ivar: i32;
    let mut nin_range: i32;
    let mut ifsep: i32;
    let mut sep_view: i32;
    //
    // First process special ranges
    //
    nran_spec = 0;
    for iran in 1..=n_ran_spec_in {
        ivstr = iv_spec_str_in[(iran - 1) as usize];
        ivend = iv_spec_end_in[(iran - 1) as usize];
        //
        // convert and trim nonexistent views from range
        //
        while ivstr <= ivend && map_file_to_view[(ivstr - 1) as usize] == 0 {
            ivstr = ivstr + 1;
        }
        while ivstr <= ivend && map_file_to_view[(ivend - 1) as usize] == 0 {
            ivend = ivend - 1;
        }
        //
        // if there is still a range, add it to list
        //
        if ivstr <= ivend {
            nran_spec = nran_spec + 1;
            iv_spec_str[(nran_spec - 1) as usize] = map_file_to_view[(ivstr - 1) as usize];
            iv_spec_end[(nran_spec - 1) as usize] = map_file_to_view[(ivend - 1) as usize];
            nmap_spec[(nran_spec - 1) as usize] = nmap_spec_in[(iran - 1) as usize];
        }
    }
    //
    // build list of all uninterrupted ranges
    //
    nran = nran_spec + 1;
    iv_spec_str[(nran - 1) as usize] = 1;
    iv_spec_end[(nran - 1) as usize] = nview;
    nmap_spec[(nran - 1) as usize] = nmap_def;
    for iran in 1..=nran_spec {
        let ia = (iran - 1) as usize;
        ir = nran_spec + 1;
        //
        // for each special range, scan rest of ranges and see if they overlap
        //
        while ir <= nran {
            let r = (ir - 1) as usize;
            if !(iv_spec_str[ia] > iv_spec_end[r] || iv_spec_end[ia] < iv_spec_str[r]) {
                //
                // overlap: then look at three cases
                //
                if iv_spec_str[ia] <= iv_spec_str[r] && iv_spec_end[ia] >= iv_spec_end[r] {
                    //
                    // CASE #1: complete overlap: wipe out the range from the list,
                    // move rest of list down
                    //
                    nran = nran - 1;
                    for ii in ir..=nran {
                        iv_spec_str[(ii - 1) as usize] = iv_spec_str[(ii + 1 - 1) as usize];
                        iv_spec_end[(ii - 1) as usize] = iv_spec_end[(ii + 1 - 1) as usize];
                        nmap_spec[(ii - 1) as usize] = nmap_spec[(ii + 1 - 1) as usize];
                    }
                } else if iv_spec_str[r] < iv_spec_str[ia] && iv_spec_end[r] > iv_spec_end[ia] {
                    //
                    // CASE #2: interior subset; need to split range in 2
                    //
                    nran = nran + 1;
                    iv_spec_end[(nran - 1) as usize] = iv_spec_end[r];
                    iv_spec_str[(nran - 1) as usize] = iv_spec_end[ia] + 1;
                    nmap_spec[(nran - 1) as usize] = nmap_spec[r];
                    iv_spec_end[r] = iv_spec_str[ia] - 1;
                } else {
                    //
                    // CASE #3, subset at one end, need to truncate range
                    //
                    if iv_spec_str[ia] > iv_spec_str[r] {
                        iv_spec_end[r] = iv_spec_str[ia] - 1;
                    } else {
                        iv_spec_str[r] = iv_spec_end[ia] + 1;
                    }
                }
            }
            ir = ir + 1;
        }
    }
    ivar = 1;
    //
    // for each range, make list of views in range; exclude the special
    // ones unless nmap is negative
    //
    for iran in 1..=nran {
        let ia = (iran - 1) as usize;
        nin_range = 0;
        for iv in iv_spec_str[ia]..=iv_spec_end[ia] {
            ifsep = 0;
            if nmap_spec[ia] > 0 {
                for ig in 1..=sg.num_separate_groups {
                    for jj in 1..=sg.num_sep_in_group[(ig - 1) as usize] {
                        if iv == sg.iviews_in_group[((ig - 1) * mx.max_view + jj - 1) as usize] {
                            ifsep = 1;
                        }
                    }
                }
            }
            if ifsep == 0 {
                nin_range = nin_range + 1;
                inran[(nin_range - 1) as usize] = iv;
            }
        }
        //
        // get the ones in range grouped; then group each special set
        //
        group_list(
            &inran,
            nin_range,
            if nmap_spec[ia] >= 0 {
                nmap_spec[ia]
            } else {
                -nmap_spec[ia]
            },
            group_size,
            num_in_view,
            nin_thresh,
            &mut ivar,
            map_list,
        );
        if nmap_spec[ia] > 0 {
            for ig in 1..=sg.num_separate_groups {
                nin_range = 0;
                for jj in 1..=sg.num_sep_in_group[(ig - 1) as usize] {
                    sep_view = sg.iviews_in_group[((ig - 1) * mx.max_view + jj - 1) as usize];
                    if sep_view >= iv_spec_str[ia] && sep_view <= iv_spec_end[ia] {
                        nin_range = nin_range + 1;
                        inran[(nin_range - 1) as usize] = sep_view;
                    }
                }
                group_list(
                    &inran,
                    nin_range,
                    nmap_spec[ia],
                    group_size,
                    num_in_view,
                    nin_thresh,
                    &mut ivar,
                    map_list,
                );
            }
        }
    }
}

/// Original: `groupList` (`map_vars.cpp:551`, file static).
///
/// Determines groupings for views in a range.  `inran` has the list of views
/// in the range, `ninRange` is the number of views in the list, `nmap` is the
/// average group size, and `groupSize` holds relative group sizes for all
/// views.  `numInView` has the number of points in each view, and `ninThresh`
/// is the threshold number required to count a view toward the total of views
/// in a group.  `ivar` comes in with the first free group number and is
/// returned at two past the last group number to avoid connections.  `mapList`
/// is indexed by view number and is filled with the group numbers.
///
/// `setSum`, `setcum`, `setTarg` and `cumNext` are `float`; each
/// `1. / (groupSize * nmap)` is a `double` quotient of a `float` product, so
/// the accumulations and comparisons are carried out in `double` and stored
/// back to `float`, as written.
fn group_list(
    inran: &[i32],
    nin_range: i32,
    nmap: i32,
    group_size: &[f32],
    num_in_view: &[i32],
    nin_thresh: i32,
    ivar: &mut i32,
    map_list: &mut [i32],
) {
    let nsets: i32;
    let mut iset_str: i32;
    let mut iset_end: i32;
    let mut last_map: i32;
    let mut nin_var: i32;
    let mut i: i32;
    let mut iran: i32;
    let mut if_few: i32;
    let mut next_ran: i32;
    let mut set_sum: f32;
    let mut setcum: f32;
    let mut set_targ: f32;
    let mut cum_next: f32;
    //
    if nin_range == 0 {
        return;
    }
    //
    // First determine if there are any views below theshold - this triggers
    // a priority toward having groups include a minimum # of views above
    // threshold, rather than a priority of maintaining a maximum group size
    if_few = 0;
    if nin_thresh > 0 {
        for iran in 1..=nin_range {
            if num_in_view[(inran[(iran - 1) as usize] - 1) as usize] < nin_thresh {
                if_few = 1;
            }
        }
    }
    //
    // compute the expected number of sets, taking into account the relative
    // groupings
    set_sum = 0.;
    if if_few == 0 {
        //
        // Count all views from start to end of range
        i = inran[0];
        while i <= inran[(nin_range - 1) as usize] {
            set_sum =
                (set_sum as f64 + 1. / (group_size[(i - 1) as usize] * nmap as f32) as f64) as f32;
            i += 1;
        }
    } else {
        //
        // Or count only views in range and above threshold for # of points
        for iran in 1..=nin_range {
            i = inran[(iran - 1) as usize];
            if num_in_view[(i - 1) as usize] >= nin_thresh {
                set_sum = (set_sum as f64
                    + 1. / (group_size[(i - 1) as usize] * nmap as f32) as f64)
                    as f32;
            }
        }
    }
    // B3DMAX(1, B3DNINT(setSum))
    let nint = (set_sum as f64 + 0.5).floor() as i32;
    nsets = if 1 > nint { 1 } else { nint };

    iset_str = inran[0];
    iset_end = iset_str - 1;
    last_map = iset_str;
    setcum = 0.;
    iran = 0;
    cum_next = 0.;
    for iset in 1..=nsets {
        //
        // find the ending point that is at or before the next target sum
        // for a set
        //
        set_targ = (iset as f32 * set_sum) / nsets as f32;
        if if_few == 0 {
            //
            // Simple case: advance as long as the next one will be below or
            // just at the target
            while iset_end < inran[(nin_range - 1) as usize]
                && setcum as f64
                    + 1. / (group_size[(iset_end + 1 - 1) as usize] * nmap as f32) as f64
                    <= set_targ as f64 + 0.01
            {
                iset_end = iset_end + 1;
                setcum = (setcum as f64
                    + 1. / (group_size[(iset_end - 1) as usize] * nmap as f32) as f64)
                    as f32;
            }
        } else {
            //
            // Complex case: index on view in list and advance as long as cumul
            // for next above threshold on list is not past target
            while iran < nin_range && cum_next as f64 <= set_targ as f64 + 0.01 {
                //
                // Advance index to next, skipping ones below threshold
                iran = iran + 1;
                while iran < nin_range {
                    if num_in_view[(inran[(iran - 1) as usize] - 1) as usize] >= nin_thresh {
                        break;
                    }
                    iran = iran + 1;
                }
                //
                // Set view number of end, and get cumulative number
                iset_end = inran[(iran - 1) as usize];
                setcum = (setcum as f64
                    + 1. / (group_size[(iset_end - 1) as usize] * nmap as f32) as f64)
                    as f32;
                //
                // if not at end, get next eligible one
                if iran < nin_range {
                    next_ran = iran + 1;
                    while next_ran < nin_range {
                        if num_in_view[(inran[(next_ran - 1) as usize] - 1) as usize] >= nin_thresh
                        {
                            break;
                        }
                        next_ran = next_ran + 1;
                    }
                    //
                    // If it is at end, set group to the end; otherwise get cum
                    if num_in_view[(inran[(next_ran - 1) as usize] - 1) as usize] >= nin_thresh {
                        cum_next = (setcum as f64
                            + 1. / (group_size[(inran[(next_ran - 1) as usize] - 1) as usize]
                                * nmap as f32) as f64) as f32;
                    } else {
                        iran = nin_range;
                        iset_end = inran[(iran - 1) as usize];
                    }
                }
            }
        }
        //
        // If it's a valid set, assign the group numbers
        // Make sure the last set goes to the end (shouldn't be needed)
        if iset_end >= iset_str {
            if iset == nsets {
                iset_end = inran[(nin_range - 1) as usize];
            }
            nin_var = 0;
            for i in 1..=nin_range {
                let v = inran[(i - 1) as usize];
                if v >= iset_str && v <= iset_end {
                    //
                    // if it's a long way since last entry, skip a group number
                    //
                    if v - last_map > nmap + 1 {
                        *ivar = *ivar + 1;
                    }
                    nin_var = nin_var + 1;
                    map_list[(v - 1) as usize] = *ivar;
                    last_map = v;
                }
            }
            if nin_var > 0 {
                *ivar = *ivar + 1;
            }
            iset_str = iset_end + 1;
        }
    }
    //
    // skip a group number at the end to prevent connections between sets
    //
    *ivar = *ivar + 1;
}

/// Original: `groupMinMax` (`map_vars.cpp:686`, file static).
fn group_min_max(
    map_list: &[i32],
    nview: i32,
    imap: i32,
    ivmin: &mut i32,
    ivmax: &mut i32,
    num_in_group: &mut i32,
) {
    *ivmin = 0;
    *ivmax = 0;
    *num_in_group = 0;
    for iv in 1..=nview {
        if map_list[(iv - 1) as usize] == imap {
            if *ivmin == 0 {
                *ivmin = iv;
            }
            *ivmax = iv;
            *num_in_group = *num_in_group + 1;
        }
    }
}

/// Original: `setGrpSize` (`map_vars.cpp:707`).
///
/// Computes relative group sizes for each view; group size is proportional to
/// cosine of the tilt angle to the given power.  `cos` and `pow` are the C++
/// `float` overloads (`cosf`/`powf` in the reference object).
pub fn set_grp_size(tilt: &[f32], nview: i32, power: f32, group_size: &mut [f32]) {
    let mut sum: f32;
    let ang_max: f32;
    //
    // 12 / 13 / 08: Originally meant the maximum angle to be 65 but it was not
    // in radians so it never had any effect.  So make it high so grouping
    // will be same for anything but 180 degree data
    ang_max = (80. * 3.14159 / 180.) as f32;
    if power == 0. {
        for i in 1..=nview {
            group_size[(i - 1) as usize] = 1.;
        }
    } else {
        sum = 0.;
        for i in 1..=nview {
            let t = tilt[(i - 1) as usize];
            let abs_t = if t >= 0. { t } else { -t };
            let ang = if abs_t < ang_max { abs_t } else { ang_max };
            group_size[(i - 1) as usize] = ang.cos().powf(power);
            sum = sum + group_size[(i - 1) as usize];
        }
        for i in 1..=nview {
            group_size[(i - 1) as usize] = nview as f32 * group_size[(i - 1) as usize] / sum;
        }
    }
}

/// Original: `mapSeparateGroup` (`map_vars.cpp:736`).
///
/// Remap the view numbers in a separate group from file views to internal
/// views.
///
/// Upstream, kept as written: `numSepInGroup` is passed **by value**
/// (`tafuncs.h:133`), so when a view is dropped (its file view is not in the
/// alignment) the list is shifted down here but the caller's count is not
/// reduced; the caller keeps the old count and the last entry is duplicated
/// (`input_vars.cpp:103`, `proc_vars.cpp:59`).
pub fn map_separate_group<const BEADTRACK: bool>(
    iviews_in_group: &mut [i32],
    mut num_sep_in_group: i32,
    map_file_to_view: &[i32],
    n_file_views: i32,
) {
    let mut i: i32;
    //
    i = 1;
    while i <= num_sep_in_group {
        if iviews_in_group[(i - 1) as usize] <= 0
            || iviews_in_group[(i - 1) as usize] > n_file_views
        {
            error_exit::<BEADTRACK>(
                "View in separate group is outside known range of image file",
                0,
            );
        }
        iviews_in_group[(i - 1) as usize] =
            map_file_to_view[(iviews_in_group[(i - 1) as usize] - 1) as usize];
        if iviews_in_group[(i - 1) as usize] == 0 {
            num_sep_in_group = num_sep_in_group - 1;
            for j in i..=num_sep_in_group {
                iviews_in_group[(j - 1) as usize] = iviews_in_group[(j + 1 - 1) as usize];
            }
        } else {
            i = i + 1;
        }
    }
}
