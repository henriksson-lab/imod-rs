//! Translation of `IMOD/flib/tiltalign/input_model.cpp` — functions for
//! reading and writing model data.
//!
//! Compiled into `tiltalign` only, so every `errorExit` is the `tiltalign`
//! build, `error_exit::<false>`.
//!
//! # File-scope statics and module state
//!
//! - `av`, `mx`, `sEvalFunct` (`input_model.cpp:20-22`), set by
//!   `inputModelSetPointers`, are parameters (`alivar.rs`).
//! - The `fmod*` globals of `fortmodel.h` are the fields of the [`FortModel`]
//!   passed in, as for every other `use fortmodel` unit
//!   (`beadtrack/proc_model.rs`): `fmodP_coord[(k - 1) * 3 + j]` is
//!   `fm.p_coord[k - 1][j]`, `fmodObject` is indexed from 0 with the 0-based
//!   `ibase_obj` offsets, and `fmodObj_color[i * 2 - 1]` is
//!   `fm.obj_color[i - 1][1]`.  The imodel_fwrap calls (`getmodelname`,
//!   `getimodhead`, `putobjcolor`, …) reach the model that `readFortModel`
//!   opened through that unit's own state, as in the source.
//! - The convenience forwarders `putObjColor`, `putImodFlag`, `putScatSize`
//!   (`fortmodel.c:493-516`) exist only to take constants by value in C; the
//!   translated `putobjcolor`/`putimodflag`/`putscatsize` already do, so they
//!   are called directly.
//!
//! # Representation
//!
//! - `FILE **` outputs are `&mut Option<ImodFile>` (`NULL` = `None`); `char
//!   **pointFile` is `&mut Option<String>`.
//! - `parselist` returns `NULL` exactly when it produced no number (an empty
//!   line, a leading `/`, or a parse error); the translated `parselist`
//!   returns `Ok` with an empty list for the first, so both `!list` tests are
//!   "`Err` or empty" here.
//! - `B3DMALLOC`ed arrays are zeroed `Vec`s; every element read is written
//!   first.  `B3DMALLOC(int, ninOutList)` with a negative count (an
//!   `IncludeStartEndInc` whose end is before its start) is `malloc` of a huge
//!   size and returns `NULL` in the source, which the loop that follows never
//!   touches; the `Vec` is sized `max(0, n)`.  An increment of 0 divides by
//!   zero there (SIGFPE in the source, a panic here).
//! - `std::map::insert` does not replace an existing key, so a repeated object
//!   in `ObjectsWithExtraWeight` keeps its first weight: `entry().or_insert`.
//!
//! # Arithmetic
//!
//! `B3DNINT((z + zorig) / delta[2])`: the sum and quotient are `float`; the
//! `0.5` of `B3DNINT` is a `double`, so the quotient widens before the add.
//! `xcen = nxyz[0] / 2.` divides in `double` and stores to `float`.
//!
//! # Upstream, kept as written
//!
//! - `objWeights[B3DMIN(numWgts, iobject)]` (`:115`): with a single weight,
//!   every object after the first reads element 1, which `resize` zeroed and
//!   `PipGetFloatArray` never wrote — so only the first listed object gets the
//!   weight and the rest get 0 (`BUGS.md`).

use super::alivar::AlignVariables;
use super::arraymaxes::ArrayMaxes;
use super::evalfunct::EvalFunct;
use super::patchtrack::analyze_patch_tracks;
use super::utilfuncs::{allocate_alivar, error_exit, memory_error};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_open_file, c_format, fortran_string, number_in_list,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_float_array, pip_get_string, pip_get_three_integers, pip_get_two_floats,
    pip_get_two_integers,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::robuststat::rs_sort_ints;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_open};
use crate::imod::libiimod::unit_header::{iiu_ret_delta, iiu_ret_origin, iiu_ret_size};
use crate::imod::libimod::fortmodel::{fort_mod_obj_to_cont, read_fort_model, write_fort_model};
use crate::imod::libimod::imodel_fwrap::{
    clearimodobjstore, getimodhead, getimodmaxes, getimodscales, getmodelname, imodopenerror,
    putimodflag, putimodrotation, putimodzscale, putobjcolor, putscatsize,
};

/// Original: `inputModelSetPointers` (`input_model.cpp:24`).
///
/// The source stores the three pointers in file-scope statics; [`input_model`]
/// takes them as parameters instead, so there is nothing to store.
pub fn input_model_set_pointers(
    av_in: &mut AlignVariables,
    mx_in: &mut ArrayMaxes,
    evfn: &mut EvalFunct,
) {
    let _ = (av_in, mx_in, evfn);
}

/// Original: `input_model` (`input_model.cpp:37`).
///
/// INPUT_MODEL reads in the model data, reads the header info from an image
/// file, sorts out the model points that are to be included in the analysis,
/// and converts the coordinates to "index" coordinates with the origin at the
/// center of the section.
#[allow(clippy::too_many_arguments)]
pub fn input_model(
    av: &mut AlignVariables,
    mx: &mut ArrayMaxes,
    s_eval_funct: &mut EvalFunct,
    fm: &mut FortModel,
    imod_obj: &mut [i32],
    imod_cont: &mut [i32],
    num_proj_pt: &mut i32,
    xcen: &mut f32,
    ycen: &mut f32,
    delta: &mut [f32],
    listz: &mut [i32],
    max_zlist: i32,
    model_file: &mut String,
    residual_file: &mut String,
    point_file: &mut Option<String>,
    iu_angle: &mut Option<ImodFile>,
    iu_xtilt: &mut Option<ImodFile>,
    sol_file_fp: &mut Option<ImodFile>,
    xorig: &mut f32,
    yorig: &mut f32,
    zorig: &mut f32,
    image_binned: i32,
) {
    let mut solu_file: Vec<u8> = Vec::new();
    let mut angle_file: Vec<u8> = Vec::new();
    let mut temp_str: Vec<u8> = Vec::new();
    let mut inc_exc_list: Vec<i32> = Vec::new();
    let obj_list: Vec<i32>;
    let mut obj_weights: Vec<f32>;
    //
    let mut nxyz = [0i32; 3];
    //
    let mut num_in_zlist: i32;
    let mut iobject: i32;
    let mut num_in_obj: i32 = 0;
    let mut ipt: i32;
    let mut iz: i32;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let min_zval: i32;
    let mut if_specify: i32;
    let mut nin_out_list: i32;
    let mut ivstr: i32 = 0;
    let mut ivend: i32 = 0;
    let mut ivinc: i32 = 0;
    let mut if_on_list: i32;
    let mut ibase: i32 = 0;
    let mut nproj_tmp: i32;
    let mut num_wgts: i32;
    let mut num_wgt_obj: i32;
    let mut inlist: i32;
    let mut ierr: i32;
    let mut if_flip: i32 = 0;
    let ierr2: i32;
    let ierr3: i32;
    let xdelt: f32;
    let ydelt: f32;
    let mut xy_scale: f32 = 0.;
    let mut zscale: f32 = 0.;
    let mut xofs: f32 = 0.;
    let mut yofs: f32 = 0.;
    let mut zofs: f32 = 0.;
    let mut list_string = [0u8; 1024];
    *point_file = None;
    //
    // read model in
    //
    fm.fm_mod_size_type = 2;
    if pip_get_string(b"ModelFile", &mut temp_str) != 0 {
        error_exit::<false>("No input fiducial model file entered", 0);
    }
    if read_fort_model(&String::from_utf8_lossy(&temp_str), fm).is_err() {
        let message = imodopenerror();
        exit_error(
            c_format(
                "Reading model file: %s",
                &[CArg::Str(&fortran_string(message.as_bytes()))],
            )
            .as_bytes(),
        );
    }
    let _ = getmodelname(&mut list_string);
    let model_name = fortran_string(&list_string);

    av.patch_track_model = (model_name == "Patch Tracking Model") as i32;
    if pip_get_two_integers(b"LeaveOutPredictAndPad", &mut i, &mut j) == 0 && j < 0 {
        av.patch_track_model = 1;
    }
    //
    // get dimensions of file and header info on origin and delta
    //
    ierr = pip_get_string(b"ImageFile", &mut temp_str);
    //
    if ierr != 0 {
        //
        // DNM 7 / 28 / 02: get defaults from  model header, including z size
        //
        let _ = getimodhead(
            &mut xy_scale,
            &mut zscale,
            &mut xofs,
            &mut yofs,
            &mut zofs,
            &mut if_flip,
        );
        {
            let (d0, rest) = delta.split_at_mut(1);
            let (d1, d2) = rest.split_at_mut(1);
            let _ = getimodscales(&mut d0[0], &mut d1[0], &mut d2[0]);
        }
        {
            let (n0, rest) = nxyz.split_at_mut(1);
            let (n1, n2) = rest.split_at_mut(1);
            let _ = getimodmaxes(&mut n0[0], &mut n1[0], &mut n2[0]);
        }
        *xorig = -xofs;
        *yorig = -yofs;
        *zorig = -zofs;
        {
            let (n0, rest) = nxyz.split_at_mut(1);
            let _ = pip_get_two_integers(b"ImageSizeXandY", &mut n0[0], &mut rest[0]);
        }
        let _ = pip_get_two_floats(b"ImageOriginXandY", xorig, yorig);
        {
            let (d0, rest) = delta.split_at_mut(1);
            let _ = pip_get_two_floats(b"ImagePixelSizeXandY", &mut d0[0], &mut rest[0]);
        }
        nxyz[0] /= image_binned;
        nxyz[1] /= image_binned;
    } else {
        unsafe { iiu_open(1, &String::from_utf8_lossy(&temp_str), "ro") };
        let (n, _mxyz, _nxyzst) = iiu_ret_size(1);
        nxyz = n;
        let origin = iiu_ret_origin(1);
        *xorig = origin[0];
        *yorig = origin[1];
        *zorig = origin[2];
        let d = iiu_ret_delta(1);
        delta[..3].copy_from_slice(&d);
        unsafe { iiu_close(1) };
    }
    xdelt = delta[0];
    ydelt = delta[1];
    *xcen = (nxyz[0] as f64 / 2.) as f32;
    *ycen = (nxyz[1] as f64 / 2.) as f32;
    //
    // Find out about extra weights
    if pip_get_string(b"ObjectsWithExtraWeight", &mut temp_str) == 0 {
        av.apply_extra_weights = 1;
        obj_list = match parselist(&String::from_utf8_lossy(&temp_str)) {
            Ok(list) if !list.is_empty() => list,
            _ => {
                error_exit::<false>("Parsing list of objects with weights", 0);
                unreachable!()
            }
        };
        num_wgt_obj = obj_list.len() as i32;
        obj_weights = vec![0.; num_wgt_obj as usize];
        num_wgts = 0;
        if pip_get_float_array(
            b"ExtraWeights",
            &mut obj_weights,
            &mut num_wgts,
            num_wgt_obj,
        ) != 0
        {
            error_exit::<false>(
                "ExtraWeights must be entered if ObjectsWithExtraWeight is",
                0,
            );
        }
        if num_wgts != 1 && num_wgts != num_wgt_obj {
            error_exit::<false>(
                "There must be either one weight, or one weight per object",
                0,
            );
        }
        iobject = 0;
        while iobject < num_wgt_obj {
            av.object_weight_map
                .entry(obj_list[iobject as usize])
                .or_insert(
                    obj_weights[(if num_wgts < iobject {
                        num_wgts
                    } else {
                        iobject
                    }) as usize],
                );
            iobject += 1;
        }
    }
    //
    // scan to get list of z values
    //
    num_in_zlist = 0;
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        num_in_obj = fm.npt_in_obj[(iobject - 1) as usize];
        if num_in_obj > 1 {
            ipt = 1;
            while ipt <= num_in_obj {
                iz = (((fm.p_coord[(fm.object
                    [(ipt + fm.ibase_obj[(iobject - 1) as usize] - 1) as usize]
                    - 1) as usize][2]
                    + *zorig)
                    / delta[2]) as f64
                    + 0.5)
                    .floor() as i32;
                if number_in_list(iz, Some(listz), num_in_zlist, 0) == 0 {
                    num_in_zlist += 1;
                    if num_in_zlist > max_zlist {
                        error_exit::<false>("Too many Z values in model for temporary arrays", 0);
                    }
                    listz[(num_in_zlist - 1) as usize] = iz;
                }
                ipt += 1;
            }
        }
        iobject += 1;
    }
    if fm.n_point == 0 {
        error_exit::<false>("The fiducial model is empty", 0);
    }
    if num_in_zlist == 0 {
        error_exit::<false>(
            "The fiducial model has no contours with more than one point",
            0,
        );
    }
    if num_in_zlist == 1 {
        error_exit::<false>("The fiducial model has points on only one view", 0);
    }
    //
    // order list of z values
    //
    rs_sort_ints(listz, num_in_zlist);
    //
    // set minimum z value if any are negative, and set number of views
    // in file to be maximum of number acquired from model or file or
    // range of z values
    //
    min_zval = if 0 < listz[0] { 0 } else { listz[0] };
    //
    // DNM 7 / 12 / 02: now that we convert to index coordinates, we
    // use the minimum just computed.
    // # of file views should be the size of file, but make it big
    // enough to hold all z values actually found
    //
    av.nfile_views = if listz[(num_in_zlist - 1) as usize] - min_zval + 1 > nxyz[2] {
        listz[(num_in_zlist - 1) as usize] - min_zval + 1
    } else {
        nxyz[2]
    };
    //
    // get name of file for output model
    //
    if pip_get_string(b"OutputModelAndResidual", &mut temp_str) == 0 {
        *model_file = String::from_utf8_lossy(&temp_str).into_owned();
        model_file.push_str(".3dmod");
        *residual_file = String::from_utf8_lossy(&temp_str).into_owned();
        residual_file.push_str(".resid");
    } else {
        if pip_get_string(b"OutputModelFile", &mut temp_str) == 0 {
            *model_file = String::from_utf8_lossy(&temp_str).into_owned();
        }
        if pip_get_string(b"OutputResidualFile", &mut temp_str) == 0 {
            *residual_file = String::from_utf8_lossy(&temp_str).into_owned();
        }
    }
    //
    //
    // get name of files for angle list output and 3 - D point output
    //
    *iu_angle = None;
    {
        let mut point: Vec<u8> = Vec::new();
        if pip_get_string(b"OutputFidXYZFile", &mut point) == 0 {
            *point_file = Some(String::from_utf8_lossy(&point).into_owned());
        }
    }
    if pip_get_string(b"OutputTiltFile", &mut angle_file) == 0 {
        *iu_angle = Some(b3d_open_file(&String::from_utf8_lossy(&angle_file), "w"));
    }
    //
    // if iuXtilt comes in non - zero, ask about file for x axis tilts
    //
    *iu_xtilt = None;
    if pip_get_string(b"OutputXAxisTiltFile", &mut angle_file) == 0 {
        *iu_xtilt = Some(b3d_open_file(&String::from_utf8_lossy(&angle_file), "w"));
    }
    //
    // open output file to put transforms into
    //
    if pip_get_string(b"OutputTransformFile", &mut solu_file) != 0 {
        error_exit::<false>("An output transform file must be entered", 0);
    }
    *sol_file_fp = Some(b3d_open_file(&String::from_utf8_lossy(&solu_file), "w"));
    //
    // find out which z values to include; end up with a list of z values
    //
    nin_out_list = 0;
    if_specify = 0;
    ierr = pip_get_three_integers(b"IncludeStartEndInc", &mut ivstr, &mut ivend, &mut ivinc);
    ierr2 = pip_get_string(b"IncludeList", &mut temp_str);
    ierr3 = pip_get_string(b"ExcludeList", &mut temp_str);
    if ierr + ierr2 + ierr3 < 2 {
        error_exit::<false>(
            "You may enter only one of IncludeStartEndInc, IncludeList, or ExcludeList",
            0,
        );
    }
    if ierr == 0 {
        if_specify = 1;
    }
    if ierr2 + ierr3 == 1 {
        match parselist(&String::from_utf8_lossy(&temp_str)) {
            Ok(list) if !list.is_empty() => {
                nin_out_list = list.len() as i32;
                inc_exc_list = list;
            }
            _ => {
                error_exit::<false>("Error parsing IncludeList or ExcludeList entry", 0);
            }
        }
        if ierr3 == 0 {
            if_specify = 3;
        }
    }
    if if_specify == 1 {
        nin_out_list = 1 + (ivend - ivstr) / ivinc;
        inc_exc_list = vec![0; nin_out_list.max(0) as usize];
        i = 1;
        while i <= nin_out_list {
            inc_exc_list[(i - 1) as usize] = ivstr + (i - 1) * ivinc;
            i += 1;
        }
    }
    //
    // go through list of z values in model and make sure they are on
    // an include list or not on an exclude list
    //
    if nin_out_list > 0 {
        i = 1;
        while i <= num_in_zlist {
            if_on_list = 0;
            j = 1;
            while j <= nin_out_list {
                if listz[(i - 1) as usize] + 1 == inc_exc_list[(j - 1) as usize] {
                    if_on_list = 1;
                }
                j += 1;
            }
            //
            // remove points on exclude list or not on include list
            //
            if (if if_specify <= 2 { 1 } else { 0 }) != (if if_on_list == 1 { 1 } else { 0 }) {
                num_in_zlist -= 1;
                j = i;
                while j <= num_in_zlist {
                    listz[(j - 1) as usize] = listz[j as usize];
                    j += 1;
                }
            } else {
                i += 1;
            }
        }
    }
    //
    // Count the number of points in valid objects on selected views
    *num_proj_pt = 0;
    av.nreal_pt = 0;
    imod_obj[..num_in_zlist.max(0) as usize].fill(0);
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        num_in_obj = fm.npt_in_obj[(iobject - 1) as usize];
        ibase = fm.ibase_obj[(iobject - 1) as usize];
        if num_in_obj > 1 {
            //
            // First determine if the object is usable at all: if it has at least 2 points
            inlist = 0;
            ipt = 1;
            while ipt <= num_in_obj {
                if number_in_list(
                    (((fm.p_coord[(fm.object[(ipt + ibase - 1) as usize] - 1) as usize][2]
                        + *zorig)
                        / delta[2]) as f64
                        + 0.5)
                        .floor() as i32,
                    Some(listz),
                    num_in_zlist,
                    0,
                ) != 0
                {
                    inlist += 1;
                    if inlist > 1 {
                        break;
                    }
                }
                ipt += 1;
            }
            //
            // If so, then count the points on valid views, and keep track of those views
            if inlist > 1 {
                av.nreal_pt += 1;
                ipt = 1;
                while ipt <= num_in_obj {
                    iz = (((fm.p_coord[(fm.object[(ipt + ibase - 1) as usize] - 1) as usize][2]
                        + *zorig)
                        / delta[2]) as f64
                        + 0.5)
                        .floor() as i32;
                    i = 1;
                    while i <= num_in_zlist {
                        if iz == listz[(i - 1) as usize] {
                            *num_proj_pt += 1;
                            imod_obj[(i - 1) as usize] += 1;
                            break;
                        }
                        i += 1;
                    }
                    ipt += 1;
                }
            }
        }
        iobject += 1;
    }
    //
    // Trim the Z list if that excluded any views
    inlist = 0;
    i = 1;
    while i <= num_in_zlist {
        if imod_obj[(i - 1) as usize] > 0 {
            inlist += 1;
            listz[(inlist - 1) as usize] = listz[(i - 1) as usize];
        }
        i += 1;
    }
    num_in_zlist = inlist;
    if num_in_zlist < 2 {
        error_exit::<false>("The fiducial model has usable points on only one view", 0);
    }
    av.nview = num_in_zlist;
    //
    // Do some big allocations

    {
        let (nview, nreal_pt) = (av.nview, av.nreal_pt);
        allocate_alivar(av, mx, *num_proj_pt, nview, nreal_pt, &mut ierr);
    }
    memory_error(ierr == 0, "arrays in av->ar");
    s_eval_funct.allocate_funct_vars(av, mx, &mut ierr);
    memory_error(ierr == 0, "arrays for funct");
    if av.apply_extra_weights != 0 {
        av.imod_obj_num = vec![0; av.nreal_pt.max(0) as usize];
        memory_error(true, "array for model object numbers");
    }
    //
    // go through model finding objects with more than one point in the
    // proper z range, and convert to index coordinates, origin at center
    //
    *num_proj_pt = 0;
    av.nreal_pt = 0;
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        //loop on objects
        num_in_obj = fm.npt_in_obj[(iobject - 1) as usize];
        ibase = fm.ibase_obj[(iobject - 1) as usize];
        if num_in_obj > 1 {
            //non - empty fmodObject
            nproj_tmp = *num_proj_pt; //save number at start of fmodObject
            ipt = 1;
            while ipt <= num_in_obj {
                //loop on points
                //
                // find out if the z coordinate of this point is on the list
                //
                iz = (((fm.p_coord[(fm.object[(ipt + ibase - 1) as usize] - 1) as usize][2]
                    + *zorig)
                    / delta[2]) as f64
                    + 0.5)
                    .floor() as i32;
                inlist = 0;
                i = 1;
                while i <= num_in_zlist {
                    if iz == listz[(i - 1) as usize] {
                        inlist = i;
                    }
                    i += 1;
                }
                if inlist > 0 {
                    //
                    // if so, tentatively add index coordinates to list but first
                    // check for two points on same view
                    //
                    i = *num_proj_pt + 1;
                    while i <= nproj_tmp {
                        if inlist == av.isec_view[(i - 1) as usize] {
                            //
                            // Find the other point
                            j = 1;
                            while j <= ipt - 1 {
                                if (((fm.p_coord
                                    [(fm.object[(j + ibase - 1) as usize] - 1) as usize][2]
                                    + *zorig)
                                    / delta[2]) as f64
                                    + 0.5)
                                    .floor() as i32
                                    == iz
                                {
                                    inlist = j;
                                }
                                j += 1;
                            }
                            fort_mod_obj_to_cont(
                                iobject,
                                &fm.obj_color,
                                &mut ibase,
                                &mut num_in_obj,
                            );
                            exit_error(
                                c_format(
                                    "Two points (# %d and %d) on view %d in contour %d of object %d",
                                    &[
                                        CArg::Int(inlist as i64),
                                        CArg::Int(ipt as i64),
                                        CArg::Int((iz + 1) as i64),
                                        CArg::Int(num_in_obj as i64),
                                        CArg::Int(ibase as i64),
                                    ],
                                )
                                .as_bytes(),
                            );
                        }
                        i += 1;
                    }
                    nproj_tmp += 1;
                    if nproj_tmp > mx.max_proj_pt {
                        error_exit::<false>("Too many projection points for arrays", 0);
                    }
                    let pc = fm.p_coord[(fm.object[(ipt + ibase - 1) as usize] - 1) as usize];
                    av.xx[(nproj_tmp - 1) as usize] = (pc[0] + *xorig) / xdelt - *xcen;
                    av.yy[(nproj_tmp - 1) as usize] = (pc[1] + *yorig) / ydelt - *ycen;
                    av.isec_view[(nproj_tmp - 1) as usize] = inlist;
                }
                ipt += 1;
            }
            //
            // if there are at least 2 points, take it as a real point
            //
            if nproj_tmp - *num_proj_pt >= 2 {
                av.nreal_pt += 1;
                if av.nreal_pt > mx.max_real {
                    error_exit::<false>("Too many fiducial points for arrays", 0);
                }
                av.ireal_str[(av.nreal_pt - 1) as usize] = *num_proj_pt + 1;
                fort_mod_obj_to_cont(
                    iobject,
                    &fm.obj_color,
                    &mut imod_obj[(av.nreal_pt - 1) as usize],
                    &mut imod_cont[(av.nreal_pt - 1) as usize],
                );
                *num_proj_pt = nproj_tmp;
                fm.npt_in_obj[(iobject - 1) as usize] = 1;
                if av.apply_extra_weights != 0 {
                    av.imod_obj_num[(av.nreal_pt - 1) as usize] =
                        imod_obj[(av.nreal_pt - 1) as usize];
                }
            } else {
                fm.npt_in_obj[(iobject - 1) as usize] = 0;
            }
        } else {
            fm.npt_in_obj[(iobject - 1) as usize] = 0;
        }
        iobject += 1;
    }
    av.ireal_str[av.nreal_pt as usize] = *num_proj_pt + 1; //Needed to get number in real sometimes
    //
    // For a patch tracking model, find which contours belong to an original full track and
    // make lists and maps back and forth
    if av.patch_track_model != 0 {
        analyze_patch_tracks(av, num_proj_pt);
    }
    //
    // shift listz to be a view list, and make the map of file to view
    // listz will be copied into mapviewtofile
    av.map_file_to_view[..av.nfile_views.max(0) as usize].fill(0);
    i = 1;
    while i <= num_in_zlist {
        listz[(i - 1) as usize] = (listz[(i - 1) as usize] - min_zval) + 1;
        av.map_file_to_view[(listz[(i - 1) as usize] - 1) as usize] = i;
        i += 1;
    }
}

/// Original: `write_xyz_model` (`input_model.cpp:367`).
///
/// WRITE_XYZ_MODEL will write out a model containing only a single point per
/// object with the solved XYZ coordinates, to file modelFile, if that string
/// is non-blank.  IGROUP is an array with the group number if the points have
/// been assigned to surfaces.  A `NULL` `modelFile` is `None`.
pub fn write_xyz_model(
    fm: &mut FortModel,
    model_file: Option<&str>,
    xyz: &[f32],
    igroup: &[i32],
    nrealpt: i32,
) {
    //
    let mut xyzmax: f32;
    let mut x_im_scale: f32 = 0.;
    let mut y_im_scale: f32 = 0.;
    let mut z_im_scale: f32 = 0.;
    let mut imod_obj_size: Vec<i32>;
    let mut map_imod_obj: Vec<i32>;
    let mut num_group1: Vec<i32>;
    let mut imod_obj: i32;
    let isize: i32;
    let mut new_obj: i32;
    let mut max_imod_obj: i32;
    let mut map_ind: i32;
    let fone: f32 = 1.;
    let fzero: f32 = 0.;
    //
    let mut ireal: i32;
    let mut iobject: i32;
    let mut ipt: i32;
    let mut i: i32;
    //
    let Some(model_file) = model_file else {
        return;
    };
    //
    // get a scattered point size (simplified 1/30/03)
    //
    xyzmax = 0.;
    ireal = 1;
    while ireal <= nrealpt {
        i = 1;
        while i <= 3 {
            let v = xyz[(ireal * 3 + i - 4) as usize];
            let a = if v >= 0. { v } else { -v };
            xyzmax = if xyzmax > a { xyzmax } else { a };
            i += 1;
        }
        ireal += 1;
    }
    {
        let q = xyzmax as f64 / 100.;
        isize = (if 3. > q { 3. } else { q }) as i32;
    }
    //
    // Count the contours per object so that objects can be reused if all
    // their points are in group 2 and get moved elsewhere
    max_imod_obj = 0;
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        let o = 256 - fm.obj_color[(iobject - 1) as usize][1];
        max_imod_obj = if max_imod_obj > o { max_imod_obj } else { o };
        iobject += 1;
    }

    imod_obj_size = vec![0; (max_imod_obj * 2).max(0) as usize];
    map_imod_obj = vec![0; (max_imod_obj * 2).max(0) as usize];
    num_group1 = vec![0; max_imod_obj.max(0) as usize];
    ireal = 0;
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        if fm.npt_in_obj[(iobject - 1) as usize] > 0 {
            imod_obj = 256 - fm.obj_color[(iobject - 1) as usize][1];
            imod_obj_size[(imod_obj - 1) as usize] += 1;
            ireal += 1;
            if igroup[(ireal - 1) as usize] == 1 {
                num_group1[(imod_obj - 1) as usize] += 1;
            }
        }
        iobject += 1;
    }
    //
    // loop on model objects that are non - zero, stuff point coords into
    // first point of object (now only point), and set color by group
    //
    // get scaling factors, might as well apply each one to each coordinate
    // 12 / 11 / 08: No longer invert Z.  Model now fits exactly on tomogram
    // opened either with - Y or without, except for X - axis tilt
    // (invert Z so that model can be visualized on tomogram by shifting)
    //
    let _ = getimodscales(&mut x_im_scale, &mut y_im_scale, &mut z_im_scale);
    ireal = 0;
    iobject = 1;
    while iobject <= fm.max_mod_obj {
        if fm.npt_in_obj[(iobject - 1) as usize] > 0 && ireal < nrealpt {
            ipt = fm.object[fm.ibase_obj[(iobject - 1) as usize] as usize];
            ireal += 1;
            fm.p_coord[(ipt - 1) as usize][0] = xyz[(ireal * 3 - 3) as usize] * x_im_scale;
            fm.p_coord[(ipt - 1) as usize][1] = xyz[(ireal * 3 - 2) as usize] * y_im_scale;
            fm.p_coord[(ipt - 1) as usize][2] = xyz[(ireal * 3 - 1) as usize] * z_im_scale;
            //
            // DNM 5 / 15 / 02: only change color if there is a group
            // assignment, but then put it in the object this one maps to
            // If no map yet, find first free object
            // Also, define as scattered and put sizes out
            //
            imod_obj = 256 - fm.obj_color[(iobject - 1) as usize][1];
            if igroup[(ireal - 1) as usize] != 0 {
                imod_obj_size[(imod_obj - 1) as usize] -= 1;
                //
                // Multiply by 2 to get index to mapping for group 1 or 2 in object
                // If already mapped, fine
                // If in group 1, map to self and set to green
                // If in group 2, map to self if there are no group 1's in object,
                // otherwise map to first empty object, and set to magenta
                map_ind = 2 * (imod_obj - 1) + igroup[(ireal - 1) as usize];
                if map_imod_obj[(map_ind - 1) as usize] != 0 {
                    imod_obj = map_imod_obj[(map_ind - 1) as usize];
                } else if igroup[(ireal - 1) as usize] == 1 {
                    map_imod_obj[(map_ind - 1) as usize] = imod_obj;
                    putobjcolor(imod_obj, 0, 255, 0);
                } else {
                    if num_group1[(imod_obj - 1) as usize] > 0 {
                        new_obj = 1;
                        while new_obj <= max_imod_obj * 2 - 1 {
                            if imod_obj_size[(new_obj - 1) as usize] == 0 {
                                break;
                            }
                            new_obj += 1;
                        }
                        imod_obj = new_obj;
                    }
                    map_imod_obj[(map_ind - 1) as usize] = imod_obj;
                    putobjcolor(imod_obj, 255, 0, 255);
                }
                fm.obj_color[(iobject - 1) as usize][1] = 256 - imod_obj;
                imod_obj_size[(imod_obj - 1) as usize] += 1;
            }
            putimodflag(imod_obj, 2);
            putscatsize(imod_obj, isize);
            let _ = clearimodobjstore(imod_obj);
        }
        iobject += 1;
    }
    putimodzscale(fone);
    putimodrotation(fzero, fzero, fzero);
    //
    fm.n_object = ireal;
    let _ = write_fort_model(model_file, fm);
}
