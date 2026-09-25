//! Translation of `IMOD/flib/beadtrack/proc_model.cpp` — processes the model
//! data, sorts out the model points that are to be included in the analysis,
//! and converts the coordinates to "index" coordinates with the origin at the
//! center of the section.
//!
//! The source reads the `fortmodel` module arrays through the `fmod*` globals
//! of `fortmodel.h` (`fmodIbase_obj`, `fmodNpt_in_obj`, `fmodObject`,
//! `fmodP_coord`); here they are the fields of the `FortModel` passed in, as
//! for every other `use fortmodel` unit.  `fmodP_coord[k * 3 - 3 .. k * 3 - 1]`
//! is `p_coord(1..3, k)`, i.e. `fm.p_coord[k - 1][0..3]`, and `fmodObject` is
//! indexed from 0 with the 0-based `ibase_obj` offsets, as in the source.
//! `B3DNINT(z) + 1` is `(z as f64 + 0.5).floor() as i32 + 1`: the `0.5` is a
//! `double`, so the `float` coordinate widens before the add.

use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::number_in_list;
use crate::imod::libcfshr::parse_params::exit_error;

/// Original: `proc_model` (`proc_model.cpp:18`).
///
/// `izExclude` may be `NULL` in the source when nothing is excluded; the
/// caller passes `(!v.is_empty()).then_some(&v[..])` (`alivar.rs`), which
/// `numberInList` tests.
#[allow(clippy::too_many_arguments)]
pub fn proc_model(
    fm: &FortModel,
    xcen: f32,
    ycen: f32,
    xdelt: f32,
    ydelt: f32,
    xorig: f32,
    yorig: f32,
    scale_xy: f32,
    nview_tot: i32,
    min_in_view: i32,
    izcur: i32,
    nz_local: i32,
    list_obj: &[i32],
    nin_list: i32,
    num_in_view: &mut [i32],
    iv_orig: &mut [i32],
    xx: &mut [f32],
    yy: &mut [f32],
    isec_view: &mut [i32],
    max_proj_pt: i32,
    max_real: i32,
    ireal_str: &mut [i32],
    iobj_ali: &mut [i32],
    nview: &mut i32,
    nprojpt: &mut i32,
    n_real_pt: &mut i32,
    iz_exclude: Option<&[i32]>,
    num_exclude: i32,
) {
    //
    let mut izst: i32 = 0;
    let mut iznd: i32 = 0;
    let mut iz_strt_min: i32;
    let iz_end_max: i32;
    let mut iobject: i32;
    let mut ibase: i32;
    let mut iz: i32;
    let num_try: i32;
    let mut iz_strt_try: i32;
    let mut iz_end_try: i32;
    let mut need_do: i32;
    let mut num_in_obj: i32;
    let mut num_legal: i32;
    let mut max_view: i32;
    let mut min_view: i32;
    let min_views_present: i32;
    let mut adj_npnts: f32;
    let mut adj_maxpt: f32;
    min_views_present = 3;

    //
    // find z limits based on nzLocal
    //
    if nz_local == 0 || nz_local >= nview_tot {
        izst = 1;
        iznd = nview_tot;
    } else {
        //
        // count number of points on each view
        //
        iz_strt_min = if 1 > izcur - nz_local {
            1
        } else {
            izcur - nz_local
        };
        iz_end_max = if nview_tot < izcur + nz_local {
            nview_tot
        } else {
            izcur + nz_local
        };
        for i in iz_strt_min..=iz_end_max {
            num_in_view[(i - 1) as usize] = 0;
        }
        for l in 0..nin_list as usize {
            iobject = list_obj[l];
            ibase = fm.ibase_obj[(iobject - 1) as usize];
            for ipt in 0..fm.npt_in_obj[(iobject - 1) as usize] {
                iz = (fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize][2] as f64 + 0.5)
                    .floor() as i32
                    + 1;
                if iz >= iz_strt_min
                    && iz <= iz_end_max
                    && number_in_list(iz, iz_exclude, num_exclude, 0) == 0
                {
                    num_in_view[(iz - 1) as usize] += 1;
                }
            }
        }
        //
        // Force the current view to be included if it has points
        if num_in_view[(izcur - 1) as usize] > 0 {
            iz_strt_min = if 1 > izcur + 1 - nz_local {
                1
            } else {
                izcur + 1 - nz_local
            };
        }
        //
        // find placement of the window of views that maximizes an adjusted
        // total number of points, where views within the range that would
        // be equally spaced around the current one are given a 10% premium
        //
        num_try = iz_end_max + 2 - iz_strt_min - nz_local;
        adj_maxpt = -1.;
        for itry in 1..=num_try {
            adj_npnts = 0.;
            iz_strt_try = iz_strt_min + itry - 1;
            iz_end_try = if nview_tot < iz_strt_try + nz_local - 1 {
                nview_tot
            } else {
                iz_strt_try + nz_local - 1
            };
            for i in iz_strt_try..=iz_end_try {
                adj_npnts += num_in_view[(i - 1) as usize] as f32;
                if (if i - izcur >= 0 {
                    i - izcur
                } else {
                    -(i - izcur)
                }) < nz_local / 2
                {
                    adj_npnts =
                        (adj_npnts as f64 + 0.1 * num_in_view[(i - 1) as usize] as f64) as f32;
                }
            }
            if adj_npnts > adj_maxpt {
                adj_maxpt = adj_npnts;
                izst = iz_strt_try;
                iznd = iz_end_try;
            }
        }
    }
    //
    // scan to get number of points on each view; eliminate consideration
    // of views with not enough points, and objects with less than 2 points
    // on the legal views; reiterate until all counted points come up above
    // the minimum
    //
    for i in 0..nview_tot as usize {
        iv_orig[i] = 0;
        num_in_view[i] = 0;
    }
    for i in izst..=iznd {
        iv_orig[(i - 1) as usize] = 2 * min_in_view;
    }
    need_do = 1;
    //
    while need_do == 1 {
        for i in izst..=iznd {
            num_in_view[(i - 1) as usize] = 0;
        }
        for l in 1..=nin_list as usize {
            iobject = list_obj[l - 1];
            num_in_obj = fm.npt_in_obj[(iobject - 1) as usize];
            if num_in_obj >= min_views_present {
                ibase = fm.ibase_obj[(iobject - 1) as usize];
                num_legal = 0;
                for ipt in 0..num_in_obj {
                    iz = (fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize][2] as f64
                        + 0.5)
                        .floor() as i32
                        + 1;
                    if iv_orig[(iz - 1) as usize] >= min_in_view
                        && number_in_list(iz, iz_exclude, num_exclude, 0) == 0
                    {
                        num_legal += 1;
                    }
                }
                if num_legal >= min_views_present {
                    for ipt in 0..num_in_obj {
                        iz = (fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize][2]
                            as f64
                            + 0.5)
                            .floor() as i32
                            + 1;
                        if iv_orig[(iz - 1) as usize] >= min_in_view
                            && number_in_list(iz, iz_exclude, num_exclude, 0) == 0
                        {
                            num_in_view[(iz - 1) as usize] += 1;
                        }
                    }
                }
            }
        }
        need_do = 0;
        for i in izst..=iznd {
            if num_in_view[(i - 1) as usize] < min_in_view && num_in_view[(i - 1) as usize] > 0 {
                need_do = 1;
            }
            iv_orig[(i - 1) as usize] = num_in_view[(i - 1) as usize];
        }
    }
    //
    // find number of views with enough points, build cross - indexes
    //
    *nview = 0;
    max_view = 0;
    min_view = nview_tot;
    for i in izst..=iznd {
        if num_in_view[(i - 1) as usize] != 0 {
            *nview += 1;
            num_in_view[(i - 1) as usize] = *nview;
            iv_orig[(*nview - 1) as usize] = i;
            min_view = if min_view < i { min_view } else { i };
            max_view = if max_view > i { max_view } else { i };
        }
    }
    let _ = (min_view, max_view);
    //
    // go through model finding objects with more than one point in the
    // proper z range, and convert to index coordinates, origin at center
    //
    *nprojpt = 0;
    *n_real_pt = 0;
    for l in 1..=nin_list as usize {
        iobject = list_obj[l - 1];
        num_in_obj = fm.npt_in_obj[(iobject - 1) as usize];
        ibase = fm.ibase_obj[(iobject - 1) as usize];
        if num_in_obj >= min_views_present {
            num_legal = 0;
            for ipt in 0..num_in_obj {
                iz = (fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize][2] as f64 + 0.5)
                    .floor() as i32
                    + 1;
                if num_in_view[(iz - 1) as usize] > 0 {
                    num_legal += 1;
                }
            }
            if num_legal >= min_views_present {
                *n_real_pt += 1;
                if *n_real_pt > max_real {
                    exit_error(b"Too many fiducials for arrays");
                }
                ireal_str[(*n_real_pt - 1) as usize] = *nprojpt + 1;
                iobj_ali[(*n_real_pt - 1) as usize] = iobject;
                for ipt in 0..num_in_obj {
                    //loop on points
                    //
                    // find out if the z coordinate of this point is on the list
                    //
                    iz = (fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize][2] as f64
                        + 0.5)
                        .floor() as i32
                        + 1;
                    if num_in_view[(iz - 1) as usize] > 0 {
                        //
                        // if so, add index coordinates to list
                        //
                        *nprojpt += 1;
                        if *nprojpt > max_proj_pt {
                            exit_error(b"Too many projection points for arrays");
                        }
                        let pt = fm.p_coord[(fm.object[(ipt + ibase) as usize] - 1) as usize];
                        xx[(*nprojpt - 1) as usize] = ((pt[0] + xorig) / xdelt - xcen) / scale_xy;
                        yy[(*nprojpt - 1) as usize] = ((pt[1] + yorig) / ydelt - ycen) / scale_xy;
                        isec_view[(*nprojpt - 1) as usize] = num_in_view[(iz - 1) as usize];
                    }
                }
            }
        }
    }
    ireal_str[*n_real_pt as usize] = *nprojpt + 1; //for convenient looping
}
