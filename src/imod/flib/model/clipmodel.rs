//! Translation of `IMOD/flib/model/clipmodel.f90`.
//!
//! CLIPMODEL clips out a portion of a model and outputs either a new model
//! file or a simple list of point coordinates.  The main program maps to
//! [`clipmodel`]; its `CONTAINS` subroutines (`addToDeletionList`,
//! `loadModelObject`, `testInsideBoundary`) read and write the host's
//! variables, so they are translated as closures-free inline functions taking
//! those variables explicitly; the external units `removeEmptyContours` and
//! `contourArea` map to [`remove_empty_contours`] and [`contour_area`].  The
//! `fortmodel` module arrays are the [`FortModel`] that `readw_or_imod` fills.
//!
//! Formatted output uses gfortran editing: `G15.6` ([`densmatch_g_edit`]),
//! `Fw.d` ([`format_f`]), `Iw` (right-justified).  `pixelMicrons` is a
//! `character*7`, so `'pixels'` is written with its blank padding.

use std::io::{BufRead, Write};

use crate::imod::flib::image::densmatch::densmatch_g_edit;
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist2, rdlist2};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::{get_model_object_range, readw_or_imod};
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::model::write_wmod::{put_model_objects, write_wmod};
use crate::imod::libcfshr::b3dutil::{exit, number_in_list};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_integer, pip_get_integer_array, pip_get_string, pip_get_two_floats,
    pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libimod::imodel_fwrap::{
    deleteimodpoint, deletelistofconts, findaddminmax1value, getcontvalue, getimodflags,
    getimodhead, getimodheado, getimodobjsize, getobjvaluethresh, getpointvalue, imodpartialmode,
};

/// `parameter (LIMZ = 100000, LIMOBJ = 100000)` (`clipmodel.f90:20`).
const LIMZ: usize = 100000;
const LIMOBJ: i32 = 100000;
/// `parameter (numOptions = 19)` (`clipmodel.f90:55`).
const NUM_OPTIONS: i32 = 19;
/// Fallback PIP table `options(1)` (`clipmodel.f90:57-65`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@point:PointOutput:I:@\
xminmax:XMinAndMax:FPM:@yminmax:YMinAndMax:FPM:@zminmax:ZMinAndMax:FPM:@\
bound:BoundaryObjects:LI:@coplanar:CoplanarBoundary:B:@\
exclude:ExcludeOrInclude:IA:@longest:LongestContourSegment:B:@\
keep:KeepEmptyContours:B:@objects:ObjectList:LI:@ends:ClipFromStartAndEnd:IP:@\
values:ValuesInOrOutOfRange:I:@range:RangeForValues:FP:@\
update:UpdateObjectMinMax:B:@areas:AreasInOrOutOfRange:I:@\
param:ParameterFile:PF:@help:usage:B:";

/// Rust-only: the values clipmodel reports, for a direct caller
/// (`CLAUDE.md`, "Wherever we control both sides, use a direct function call
/// now").  One entry per clipping pass, recorded where FORMAT 102 prints
/// them: `(nonEmptyOld, nonEmptyNew)` and `(numPtsTotOld, numPtsTot)`.
/// autofidseed reads the number after "Number of points ... to".
#[derive(Clone, Debug, Default)]
pub struct ClipmodelResult {
    pub contours_reduced: Vec<(i32, i32)>,
    pub points_reduced: Vec<(i32, i32)>,
}

thread_local! {
    /// Where [`clipmodel`] records its [`ClipmodelResult`] on this thread,
    /// when a direct caller set it through [`clipmodel_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<ClipmodelResult>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program [`clipmodel`] on this thread with its reported
/// values recorded into `sink`.  The program ends through `exit`, so run it
/// under `commands::call_in_process`.
pub fn clipmodel_recording(sink: std::sync::Arc<std::sync::Mutex<ClipmodelResult>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    clipmodel();
}

/// gfortran `Iw` editing of an integer: right-justified, asterisks when it
/// does not fit.
fn fmt_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// `read(5,*)` with no `END=`/`ERR=`: the gfortran runtime reports a failed
/// read and stops with status 2.
fn read_abort(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}

/// The host-association state the `CONTAINS` subroutines of `clipmodel`
/// share with the main program.
struct Host {
    num_to_delete: i32,
    ipnt_to_delete: Vec<i32>,
    imod_obj: i32,
    xx: f32,
    yy: f32,
    zz: f32,
    min_zbound: i32,
    max_zbound: i32,
    map_zbound: Vec<i32>,
    num_bound_cont: i32,
    iz_bound: Vec<i32>,
    ind_bound_cont: Vec<i32>,
    num_pt_in_bound: Vec<i32>,
    x_bound: Vec<f32>,
    y_bound: Vec<f32>,
}

/// Original: `addToDeletionList` (`clipmodel.f90:582`, contained).
///
/// Adds a range of point numbers to the list of points to delete.
fn add_to_deletion_list(h: &mut Host, istart: i32, iend: i32) {
    for iptmp in istart..=iend {
        h.num_to_delete += 1;
        h.ipnt_to_delete[h.num_to_delete as usize - 1] = iptmp;
    }
}

/// Original: `loadModelObject` (`clipmodel.f90:592`, contained).
///
/// Loads one model object, gives an error message, scales it.
fn load_model_object(h: &Host, fm: &mut FortModel) {
    if !get_model_object_range(h.imod_obj, h.imod_obj, fm) {
        // `write(*,'(/,a,i6)')`
        println!(
            "\nERROR: CLIPMODEL - Loading data for object #{}",
            fmt_i(h.imod_obj, 6)
        );
        exit(1);
    }
    scale_model(0, fm);
}

/// Original: `testInsideBoundary` (`clipmodel.f90:601`, contained).
///
/// Tests whether the current `xx, yy, zz` is inside any of the boundary
/// contours on the Z value that it maps to (nearest one with contours, or
/// same Z value).
fn test_inside_boundary(h: &Host, inside_bounds: &mut bool) {
    *inside_bounds = false;
    let mut izpt = h.zz.round() as i32;
    if izpt < h.min_zbound || izpt > h.max_zbound {
        return;
    }
    *inside_bounds = true;
    izpt = h.map_zbound[(izpt + 1 - h.min_zbound) as usize - 1];
    for ibound in 1..=h.num_bound_cont as usize {
        if izpt == h.iz_bound[ibound - 1] {
            let start = h.ind_bound_cont[ibound - 1] as usize - 1;
            if inside(
                &h.x_bound[start..],
                &h.y_bound[start..],
                h.num_pt_in_bound[ibound - 1],
                h.xx,
                h.yy,
            ) {
                return;
            }
        }
    }
    *inside_bounds = false;
}

/// Original program: `clipmodel` (`clipmodel.f90:17`).
pub fn clipmodel() {
    let mut listz = vec![0i32; LIMZ];
    let mut iobj_elim = vec![0i32; LIMOBJ as usize];
    let mut iflags = vec![0i32; LIMOBJ as usize];
    let mut incl_excl_list = vec![0i32; LIMOBJ as usize];
    let mut iobj_bound = vec![0i32; LIMOBJ as usize];
    let mut num_obj_bound: i32;
    let mut model_file = String::new();
    let mut input_model = String::new();
    let mut in_or_ex: &str;
    let mut pixel_microns: &str;
    let mut ierr: i32;
    let mut if_cut: i32;
    let mut if_exclude: i32;
    let mut if_flip: i32 = 0;
    let mut if_keep_all: i32;
    let mut if_loop: i32 = 0;
    let mut if_point: i32;
    let mut ind_y: i32;
    let mut ind_z: i32;
    let mut iobj: i32;
    let mut num_obj_elim: i32 = 0;
    let num_obj_tot: i32;
    let mut num_pts_tot: i32;
    let mut num_pts_tot_old: i32;
    let mut num_rem_end: i32 = 0;
    let mut num_rem_start: i32 = 0;
    let mut num_loops: i32;
    let mut no_actions: i32 = 0;
    let mut ixyz_sum: i32 = 0;
    let mut if_val_range: i32;
    let mut iopt_values: i32;
    let mut ipt_del_size: i32;
    let mut max_num_pts: i32;
    let mut cut_sum: f32;
    let mut volume: f32;
    let mut xmax: f32 = 0.;
    let mut xmin: f32 = 0.;
    let mut val_range_min: f32 = 0.;
    let mut val_range_max: f32 = 0.;
    let mut xy_scale: f32 = 0.;
    let mut ymax: f32 = 0.;
    let mut ymin: f32 = 0.;
    let mut xofs: f32 = 0.;
    let mut yofs: f32 = 0.;
    let mut zofs: f32 = 0.;
    let z_full_scale: f32;
    let mut z_scale: f32 = 0.;
    let mut zmax: f32 = 0.;
    let mut zmin: f32 = 0.;
    let mut gsval: f32 = 0.;
    let mut keep_empty: bool;
    let mut update_mm: bool;
    let mut coplanar_boundary: bool;
    let mut iopt_areas: i32;
    let pip_input: bool;
    let mut num_opt_arg = 0;
    let mut num_non_opt_arg = 0;
    let mut ix_min: i32 = 0;
    let mut iy_min: i32 = 0;
    let mut iz_min: i32 = 0;
    // `use fortmodel`
    let mut fm = FortModel::default();
    let mut h = Host {
        num_to_delete: 0,
        ipnt_to_delete: Vec::new(),
        imod_obj: 0,
        xx: 0.,
        yy: 0.,
        zz: 0.,
        min_zbound: 0,
        max_zbound: 0,
        map_zbound: Vec::new(),
        num_bound_cont: 0,
        iz_bound: Vec::new(),
        ind_bound_cont: Vec::new(),
        num_pt_in_bound: Vec::new(),
        x_bound: Vec::new(),
        y_bound: Vec::new(),
    };
    // Locals the source leaves undefined until first set.
    let mut ipt_inside: i32 = 0;
    let mut ipt_start: i32 = 0;
    let mut lpnt: i32 = 0;

    if_point = 0;
    if_keep_all = 1;
    keep_empty = false;
    if_val_range = 0;
    iopt_values = 0;
    iopt_areas = 0;
    ipt_del_size = 0;
    update_mm = false;
    num_obj_bound = 0;
    coplanar_boundary = false;

    // Pip startup: set error, parse options, check help, set flag if used
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "clipmodel",
        "ERROR: CLIPMODEL - ",
        true,
        1,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;

    if pip_get_in_out_file(
        "InputFile",
        1,
        "Name of input model file",
        &mut input_model,
        320,
    ) != 0
    {
        exit_error("No input file specified");
    }
    //
    if pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output model or point file",
        &mut model_file,
        320,
    ) != 0
    {
        exit_error("No output file specified");
    }
    //
    if pip_input {
        pip_get_integer(b"PointOutput", &mut if_point);
        let mut i = 1 - if_keep_all;
        pip_get_boolean(b"LongestContourSegment", &mut i);
        if_keep_all = 1 - i;
        pip_get_logical("KeepEmptyContours", &mut keep_empty);
        pip_get_integer(b"ValuesInOrOutOfRange", &mut iopt_values);
        if_val_range =
            1 - pip_get_two_floats(b"RangeForValues", &mut val_range_min, &mut val_range_max);
        pip_get_logical("UpdateObjectMinMax", &mut update_mm);
        if update_mm && keep_empty {
            exit_error("You cannot use -update with -keep");
        }
        pip_get_integer(b"AreasInOrOutOfRange", &mut iopt_areas);
        if iopt_areas != 0 && iopt_values != 0 {
            exit_error("You cannot use -areas with -values");
        }
        if iopt_areas != 0 && if_val_range == 0 {
            exit_error("You must enter an area range with -range when deleting by area");
        }
    } else {
        print!(" 0 for model output, 1 for point file output (-1 for corner points also): ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut if_point)],
        ) {
            read_abort(err);
        }
        //
        print!(
            " 0 to retain only longest included segment,\n  or 1 to retain all points within limits: "
        );
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut if_keep_all)],
        ) {
            read_abort(err);
        }
    }
    imodpartialmode(1);
    fm.fm_boost_read_in_by = 1.05;
    if if_point == 0 {
        fm.fm_max_obj_loaded = 1;
    }
    let exist = readw_or_imod(&input_model, &mut fm);
    if !exist {
        exit_error("Opening model file");
    }
    //
    for i in 1..=LIMOBJ as usize {
        iflags[i - 1] = 0;
    }
    ierr = getimodflags(&mut iflags);
    if ierr != 0 {
        println!(" Error getting object types, assuming all are closed contours");
    }
    num_obj_tot = getimodobjsize();
    pixel_microns = "microns";
    //
    // DNM 7/7/02: get image scales, and switch to exiting on error
    //
    ierr = getimodhead(
        &mut xy_scale,
        &mut z_scale,
        &mut xofs,
        &mut yofs,
        &mut zofs,
        &mut if_flip,
    );
    if ierr != 0 {
        exit_error("Getting model header");
    }
    ierr = getimodheado(&mut xy_scale, &mut z_scale);
    if ierr != 0 {
        xy_scale = 1.;
        // `pixelMicrons = 'pixels'` into a `character*7`
        pixel_microns = "pixels ";
        if iopt_areas.abs() > 1 {
            exit_error("The model has no pixel size so the area range must be specified in pixels");
        }
    }
    if iopt_areas.abs() == 2 {
        val_range_min = val_range_min * 1.0e-6 / (xy_scale * xy_scale);
        val_range_max = val_range_max * 1.0e-6 / (xy_scale * xy_scale);
    } else if iopt_areas.abs() > 2 {
        val_range_min /= xy_scale * xy_scale;
        val_range_max /= xy_scale * xy_scale;
    }
    z_full_scale = xy_scale * z_scale;
    if if_flip == 0 {
        ind_y = 2;
        ind_z = 3;
    } else {
        ind_z = 2;
        ind_y = 3;
    }
    num_loops = 1000000;
    let mut list_string: Vec<u8> = Vec::new();
    if pip_input {
        num_loops = 0;
        pip_number_of_entries(b"XMinAndMax", &mut ix_min);
        pip_number_of_entries(b"YMinAndMax", &mut iy_min);
        pip_number_of_entries(b"ZMinAndMax", &mut iz_min);
        ixyz_sum = ix_min + iy_min + iz_min;
        no_actions = pip_get_integer_array(
            b"ExcludeOrInclude",
            &mut incl_excl_list,
            &mut num_loops,
            LIMOBJ,
        );
        iobj = 1 - pip_get_string(b"ObjectList", &mut list_string);
        if iobj > 0 {
            let mut lim = LIMOBJ;
            let _ = parselist2(
                &String::from_utf8_lossy(&list_string),
                &mut iobj_elim,
                &mut num_obj_elim,
                &mut lim,
            );
        }
        let if_cut_entered =
            1 - pip_get_two_integers(b"ClipFromStartAndEnd", &mut num_rem_start, &mut num_rem_end);
        if iobj > 0 && (no_actions == 0 || ixyz_sum > 0) {
            exit_error("You cannot enter an object list with coordinate inclusion/exclusions");
        }
        if if_cut_entered > 0 && iobj == 0 {
            exit_error("You must enter an object list when clipping from starts and ends");
        }
        if iopt_values.abs() > 1 && (no_actions == 0 || ix_min + iy_min + iz_min > 0) {
            exit_error(
                "You cannot clip points by value together with coordinate inclusion/exclusions",
            );
        }
        if if_cut_entered > 0 && iopt_values.abs() > 1 {
            exit_error("You cannot clip points by value when clipping from starts and ends");
        }
        let mut bound_list: Vec<u8> = Vec::new();
        let loop_ = 1 - pip_get_string(b"BoundaryObjects", &mut bound_list);
        if loop_ > 0 {
            let mut lim = LIMOBJ;
            let _ = parselist2(
                &String::from_utf8_lossy(&bound_list),
                &mut iobj_bound,
                &mut num_obj_bound,
                &mut lim,
            );
        }
        if iobj > 0 && loop_ > 0 {
            exit_error("You cannot use a boundary object when clipping from starts and ends");
        }
        if iopt_values.abs() > 1 && loop_ > 0 {
            exit_error("You cannot use a boundary object when clipping points by value");
        }
        if no_actions != 0
            && iobj == 0
            && iopt_values == 0
            && ixyz_sum == 0
            && loop_ == 0
            && iopt_areas == 0
        {
            exit_error("You must specify something to do");
        }
        if no_actions != 0 && iobj == 0 {
            num_loops = 1;
            incl_excl_list[0] = 0;
            no_actions = 0;
        }
        if no_actions != 0 {
            num_loops = 1;
            incl_excl_list[0] = -1;
            if if_cut_entered > 0 {
                incl_excl_list[0] = -2;
            }
        } else {
            if ix_min > 0 && ix_min != num_loops {
                exit_error(
                    "The number of -xminmax entries must match the number of operations entered with -exclude",
                );
            }
            if iy_min > 0 && iy_min != num_loops {
                exit_error(
                    "The number of -yminmax entries must match the number of operations entered with -exclude",
                );
            }
            if iz_min > 0 && iz_min != num_loops {
                exit_error(
                    "The number of -zminmax entries must match the number of operations entered with -exclude",
                );
            }
            if if_point < 0 && (ix_min != num_loops || iy_min != num_loops) {
                exit_error(
                    "If you want corner points, then you MUST enter actual min and max X & Y",
                );
            }
        }
    }
    //
    // Process and store the boundary contours first
    if num_obj_bound > 0 {
        let mut coplanar_int = coplanar_boundary as i32;
        pip_get_boolean(b"CoplanarBoundary", &mut coplanar_int);
        coplanar_boundary = coplanar_int != 0;
        h.num_bound_cont = 0;
        let mut num_bound_point = 0;
        h.min_zbound = 100000000;
        h.max_zbound = -h.min_zbound;
        for loop_ in 1..=num_obj_bound as usize {
            h.imod_obj = iobj_bound[loop_ - 1];
            if iflags[h.imod_obj as usize - 1] % 4 != 0 {
                exit_error("Only closed contour objects can be used for boundary objects");
            }
            load_model_object(&h, &mut fm);
            for iobj in 1..=fm.max_mod_obj as usize {
                num_bound_point += fm.npt_in_obj[iobj - 1];
                // The source reads the first point even of an empty contour,
                // which past the loaded points is `p_coord(indZ, 0)`, out of
                // bounds; such a contour is stored nowhere below, so its Z is
                // not taken here.
                let jp = fm.object[(1 + fm.ibase_obj[iobj - 1]) as usize - 1].abs();
                if fm.npt_in_obj[iobj - 1] == 0 && jp < 1 {
                    continue;
                }
                let iz = fm.p_coord[jp as usize - 1][ind_z as usize - 1].round() as i32;
                h.min_zbound = h.min_zbound.min(iz);
                h.max_zbound = h.max_zbound.max(iz);
            }
            h.num_bound_cont += fm.max_mod_obj;
        }
        if h.num_bound_cont == 0 || num_bound_point == 0 {
            exit_error("The boundary objects have no contours with points");
        }
        h.x_bound = vec![0.; num_bound_point as usize];
        h.y_bound = vec![0.; num_bound_point as usize];
        h.num_pt_in_bound = vec![0; h.num_bound_cont as usize];
        h.ind_bound_cont = vec![0; h.num_bound_cont as usize];
        h.iz_bound = vec![0; h.num_bound_cont as usize];
        h.map_zbound = vec![0; (h.max_zbound + 4 - h.min_zbound).max(0) as usize];
        memory_error(0, "arrays for boundary contours");

        // Now store them
        h.num_bound_cont = 0;
        num_bound_point = 0;
        for loop_ in 1..=num_obj_bound as usize {
            h.imod_obj = iobj_bound[loop_ - 1];
            load_model_object(&h, &mut fm);
            for iobj in 1..=fm.max_mod_obj as usize {
                let num_in_obj = fm.npt_in_obj[iobj - 1];
                if num_in_obj > 0 {
                    h.num_bound_cont += 1;
                    let nb = h.num_bound_cont as usize - 1;
                    h.num_pt_in_bound[nb] = num_in_obj;
                    h.ind_bound_cont[nb] = num_bound_point + 1;
                    let ibase = fm.ibase_obj[iobj - 1];
                    let jp = fm.object[(1 + ibase) as usize - 1].abs();
                    h.iz_bound[nb] = fm.p_coord[jp as usize - 1][ind_z as usize - 1].round() as i32;
                    for ipt in 1..=num_in_obj {
                        let jpnt = fm.object[(ipt + ibase) as usize - 1].abs();
                        num_bound_point += 1;
                        h.x_bound[num_bound_point as usize - 1] = fm.p_coord[jpnt as usize - 1][0];
                        h.y_bound[num_bound_point as usize - 1] =
                            fm.p_coord[jpnt as usize - 1][ind_y as usize - 1];
                    }
                }
            }
        }
        //
        // Set up mapping of Z values in range either to the same Z, or to the nearest
        // Z with boundary contours
        for iz in h.min_zbound..=h.max_zbound {
            h.map_zbound[(iz + 1 - h.min_zbound) as usize - 1] = iz;
        }
        if !coplanar_boundary {
            for iz in h.min_zbound..=h.max_zbound {
                let mut min_diff = 10000000;
                for iobj in 1..=h.num_bound_cont as usize {
                    let ipt = (h.iz_bound[iobj - 1] - iz).abs();
                    if ipt < min_diff || (ipt == min_diff && iz > h.iz_bound[iobj - 1]) {
                        min_diff = ipt;
                        h.map_zbound[(iz + 1 - h.min_zbound) as usize - 1] = h.iz_bound[iobj - 1];
                    }
                }
            }
        }
    }
    //
    let mut loop_ = 1;
    while loop_ <= num_loops {
        xmin = -1.0e9;
        xmax = 1.0e9;
        ymin = -1.0e9;
        ymax = 1.0e9;
        zmin = -1.0e9;
        zmax = 1.0e9;
        if pip_input {
            if_exclude = incl_excl_list[loop_ as usize - 1];
        } else {
            print!(
                " Enter 0 to include or 1 to exclude points in a specified coordinate block,\n   or -1 or -2 to exclude or shorten particular objects: "
            );
            let _ = std::io::stdout().flush();
            if_exclude = 0;
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut if_exclude)],
            ) {
                read_abort(err);
            }
        }
        if if_exclude >= 0 {
            in_or_ex = "in";
            if if_exclude != 0 {
                in_or_ex = "ex";
            }
            //
            if pip_input {
                pip_get_two_floats(b"XMinAndMax", &mut xmin, &mut xmax);
                pip_get_two_floats(b"YMinAndMax", &mut ymin, &mut ymax);
                pip_get_two_floats(b"ZMinAndMax", &mut zmin, &mut zmax);
            } else {
                loop {
                    print!(
                        " Minimum and maximum X and Y to {}clude (/ for no limits): ",
                        in_or_ex
                    );
                    let _ = std::io::stdout().flush();
                    if let Err(err) = list_read(
                        &mut std::io::stdin().lock(),
                        &mut [
                            ListItem::Real(&mut xmin),
                            ListItem::Real(&mut xmax),
                            ListItem::Real(&mut ymin),
                            ListItem::Real(&mut ymax),
                        ],
                    ) {
                        read_abort(err);
                    }
                    if if_point < 0 && ymax > 9.0e8 {
                        println!(
                            " If you want corner points, then you MUST enter actual min and max X & Y"
                        );
                        continue;
                    }
                    break;
                }
                print!(
                    " Minimum and maximum Z to {}clude (/ for no limits): ",
                    in_or_ex
                );
                let _ = std::io::stdout().flush();
                if let Err(err) = list_read(
                    &mut std::io::stdin().lock(),
                    &mut [ListItem::Real(&mut zmin), ListItem::Real(&mut zmax)],
                ) {
                    read_abort(err);
                }
            }
            if xmax - xmin < 9.0e8 && ymax - ymin < 9.0e8 && zmax - zmin < 9.0e8 {
                volume = (zmax + 1. - zmin)
                    * (xmax + 1. - xmin)
                    * (ymax + 1. - ymin)
                    * z_full_scale
                    * (xy_scale * xy_scale);
                // FORMAT 105: (/,' Selected volume is',g15.6,' cubic ',a)
                println!(
                    "\n Selected volume is{} cubic {}",
                    densmatch_g_edit(volume, 15, 6),
                    pixel_microns
                );
            }

            // Increase Z limits a bit in case there are slightly off Z values
            zmin -= 0.005;
            zmax += 0.005;
        } else if !pip_input {
            if if_exclude < -1 {
                print!(" Number of points to remove from start, # to remove from end: ");
                let _ = std::io::stdout().flush();
                if let Err(err) = list_read(
                    &mut std::io::stdin().lock(),
                    &mut [
                        ListItem::Integer(&mut num_rem_start),
                        ListItem::Integer(&mut num_rem_end),
                    ],
                ) {
                    read_abort(err);
                }
                println!(" Enter list of object numbers for contours to shorten (ranges OK)");
            } else {
                println!(" Enter list of numbers of objects to eliminate (ranges OK)");
            }
            let mut lim = LIMOBJ;
            let _ = rdlist2(
                &mut std::io::stdin().lock(),
                &mut iobj_elim,
                &mut num_obj_elim,
                &mut lim,
            );
        }
        //
        // look at each object, find longest contiguous segment within area
        num_pts_tot = 0;
        num_pts_tot_old = 0;
        cut_sum = 0.;
        let mut non_empty_old = 0;
        let mut non_empty_new = 0;
        h.imod_obj = 1;
        while h.imod_obj <= num_obj_tot {
            if number_in_list(
                h.imod_obj,
                Some(&iobj_bound[..num_obj_bound.max(0) as usize]),
                num_obj_bound,
                0,
            ) > 0
            {
                h.imod_obj += 1;
                continue;
            }
            load_model_object(&h, &mut fm);
            let imod_obj = h.imod_obj;
            //
            max_num_pts = 0;
            for iobj in 1..=fm.max_mod_obj as usize {
                num_pts_tot_old += fm.npt_in_obj[iobj - 1];
                max_num_pts = max_num_pts.max(fm.npt_in_obj[iobj - 1]);
            }

            // Maintain array for keeping track of deleted points
            if max_num_pts > ipt_del_size {
                ipt_del_size = max_num_pts + 1000;
                h.ipnt_to_delete = vec![0; ipt_del_size as usize];
                memory_error(0, "array for deleted point list");
            }

            // If doing values and no range was entered, get the object threshold
            // If there is not one, skip the object
            let mut do_values = iopt_values != 0;
            if do_values && if_val_range == 0 {
                do_values = getobjvaluethresh(imod_obj, &mut val_range_max) == 0;
                val_range_min = -1.0e37;
            }
            //
            // Loop on the contours
            for iobj in 1..=fm.max_mod_obj {
                let io = iobj as usize - 1;
                let mut num_in_obj = fm.npt_in_obj[io];
                h.num_to_delete = 0;
                if num_in_obj > 0 {
                    non_empty_old += 1;
                }
                if do_values && iopt_values.abs() % 2 > 0 {
                    //
                    // If doing values, get the contour value and eliminate it by just zeroing it
                    // But also set up to delete the contour data
                    if getcontvalue(imod_obj, iobj, &mut gsval) == 0 {
                        let mut inside_ = gsval >= val_range_min && gsval <= val_range_max;
                        if iopt_values < 0 {
                            inside_ = !inside_;
                        }
                        if inside_ {
                            add_to_deletion_list(&mut h, 1, num_in_obj);
                            fm.npt_in_obj[io] = 0;
                            num_in_obj = 0;
                        }
                    }
                }
                //
                // If doing areas, get the area in 3D and set up to delete the same way
                if iopt_areas != 0 && iflags[imod_obj as usize - 1] % 4 == 0 {
                    gsval = contour_area(iobj, ind_y, ind_z, z_scale, &fm);
                    let mut inside_ = gsval >= val_range_min && gsval <= val_range_max;
                    if iopt_areas < 0 {
                        inside_ = !inside_;
                    }
                    if inside_ {
                        add_to_deletion_list(&mut h, 1, num_in_obj);
                        fm.npt_in_obj[io] = 0;
                        num_in_obj = 0;
                    }
                }
                //
                // Then do other operations if that did not delete contour
                if num_in_obj > 0 {
                    let mut ipt_out = 0;
                    let ibase = fm.ibase_obj[io];
                    if do_values && iopt_values.abs() > 1 {
                        //
                        // point clipping by value, repack object array with retained points
                        for ipt in 1..=num_in_obj {
                            if getpointvalue(imod_obj, iobj, ipt, &mut gsval) == 0 {
                                let mut inside_ = gsval >= val_range_min && gsval <= val_range_max;
                                if iopt_values < 0 {
                                    inside_ = !inside_;
                                }
                                if inside_ {
                                    add_to_deletion_list(&mut h, ipt, ipt);
                                } else {
                                    ipt_out += 1;
                                    fm.object[(ibase + ipt_out) as usize - 1] =
                                        fm.object[(ibase + ipt) as usize - 1];
                                }
                            }
                        }
                        fm.npt_in_obj[io] = ipt_out;
                    } else if if_exclude >= 0 {
                        if if_keep_all == 0 && iflags[imod_obj as usize - 1] % 4 != 2 {
                            //
                            // keeping only longest segment inside limits: can be used for
                            // open or closed contours but not scattered points
                            // Loop and find largest segment
                            let mut last_inside = false;
                            let mut ipt_start_best = 1;
                            let mut ipt_end_best = 0;
                            for ipt in 1..=num_in_obj {
                                let jpnt = fm.object[(ipt + ibase) as usize - 1].abs() as usize;
                                h.xx = fm.p_coord[jpnt - 1][0];
                                h.yy = fm.p_coord[jpnt - 1][ind_y as usize - 1];
                                h.zz = fm.p_coord[jpnt - 1][ind_z as usize - 1];
                                let mut inside_ = (h.xx >= xmin && h.xx <= xmax)
                                    && (h.yy >= ymin && h.yy <= ymax)
                                    && (h.zz >= zmin && h.zz <= zmax);
                                if inside_ && num_obj_bound > 0 && loop_ == 1 {
                                    test_inside_boundary(&h, &mut inside_);
                                }
                                if if_exclude != 0 {
                                    inside_ = !inside_;
                                }
                                if inside_ {
                                    ipt_inside = ipt;
                                }
                                if inside_ && !last_inside {
                                    ipt_start = ipt;
                                }
                                if (last_inside && !inside_) || (ipt == num_in_obj && inside_) {
                                    if ipt_end_best - ipt_start_best < ipt_inside - ipt_start {
                                        ipt_end_best = ipt_inside;
                                        ipt_start_best = ipt_start;
                                    }
                                }
                                last_inside = inside_;
                            }
                            //
                            // Add to deletion list
                            add_to_deletion_list(&mut h, 1, ipt_start_best - 1);
                            add_to_deletion_list(&mut h, ipt_end_best + 1, num_in_obj);
                            //
                            // truncate object and point base to new start
                            fm.npt_in_obj[io] = ipt_end_best + 1 - ipt_start_best;
                            fm.ibase_obj[io] += ipt_start_best - 1;
                        } else {
                            //
                            // keeping all points inside limits
                            // repack the object array with just the points being retained
                            if_cut = 0;
                            let mut if_start_cut = 0;
                            let mod_flag = iflags[imod_obj as usize - 1] % 4;
                            for ipt in 1..=num_in_obj {
                                let jpnt = fm.object[(ipt + ibase) as usize - 1].abs();
                                let jp = jpnt as usize - 1;
                                h.xx = fm.p_coord[jp][0];
                                h.yy = fm.p_coord[jp][ind_y as usize - 1];
                                h.zz = fm.p_coord[jp][ind_z as usize - 1];
                                let mut inside_ = (h.xx >= xmin && h.xx <= xmax)
                                    && (h.yy >= ymin && h.yy <= ymax)
                                    && (h.zz >= zmin && h.zz <= zmax);
                                if inside_ && num_obj_bound > 0 && loop_ == 1 {
                                    test_inside_boundary(&h, &mut inside_);
                                }
                                if if_exclude != 0 {
                                    inside_ = !inside_;
                                }
                                if inside_ {
                                    if if_cut != 0 && mod_flag != 2 {
                                        if ipt_out == 0 {
                                            //
                                            // if last point was cut out and we are still at the
                                            // start, then mark the start as cut
                                            //
                                            if_start_cut = 1;
                                        } else {
                                            //
                                            // otherwise, compute distance between this & last point
                                            //
                                            let lp = lpnt as usize - 1;
                                            let dx =
                                                xy_scale * (fm.p_coord[jp][0] - fm.p_coord[lp][0]);
                                            let dy = xy_scale
                                                * (fm.p_coord[jp][ind_y as usize - 1]
                                                    - fm.p_coord[lp][ind_y as usize - 1]);
                                            let dz = z_full_scale
                                                * (fm.p_coord[jp][ind_z as usize - 1]
                                                    - fm.p_coord[lp][ind_z as usize - 1]);
                                            cut_sum += (dx * dx + dy * dy + dz * dz).sqrt();
                                        }
                                    }
                                    if_cut = 0;
                                    ipt_out += 1;
                                    fm.object[(ibase + ipt_out) as usize - 1] =
                                        fm.object[(ibase + ipt) as usize - 1];
                                    lpnt = jpnt;
                                } else {
                                    if_cut = 1;
                                    add_to_deletion_list(&mut h, ipt, ipt);
                                }
                            }
                            fm.npt_in_obj[io] = ipt_out;
                            //
                            // If there are any points left, and either the start was cut
                            // or the last point was cut, and this is a closed contour,
                            // then add distance between endpoints as a cut edge
                            if ipt_out > 0 && (if_start_cut > 0 || if_cut != 0) && mod_flag == 0 {
                                let jp = fm.object[(1 + ibase) as usize - 1].abs() as usize - 1;
                                let lp = lpnt as usize - 1;
                                let dx = xy_scale * (fm.p_coord[jp][0] - fm.p_coord[lp][0]);
                                let dy = xy_scale
                                    * (fm.p_coord[jp][ind_y as usize - 1]
                                        - fm.p_coord[lp][ind_y as usize - 1]);
                                let dz = z_full_scale
                                    * (fm.p_coord[jp][ind_z as usize - 1]
                                        - fm.p_coord[lp][ind_z as usize - 1]);
                                cut_sum += (dx * dx + dy * dy + dz * dz).sqrt();
                            }
                        }
                        //
                        // if getting rid of certain color, check for color
                    } else {
                        let mut iobj_test = 256 - fm.obj_color[io][1];
                        if fm.obj_color[io][0] == 0 {
                            iobj_test = -iobj_test;
                        }
                        let mut if_treat = 0;
                        for icl in 1..=num_obj_elim as usize {
                            if iobj_test == iobj_elim[icl - 1] {
                                if_treat = 1;
                            }
                        }
                        if if_treat == 1 {
                            if if_exclude == -1 {
                                //
                                // Getting rid of whole object, point deletion should be gratuitous
                                fm.npt_in_obj[io] = 0;
                                add_to_deletion_list(&mut h, 1, num_in_obj);
                            } else {
                                //
                                // Clipping ends, do point deletion and move object base
                                add_to_deletion_list(&mut h, 1, num_rem_start);
                                add_to_deletion_list(
                                    &mut h,
                                    num_in_obj + 1 - num_rem_end,
                                    num_in_obj,
                                );
                                fm.npt_in_obj[io] = 0.max(num_in_obj - num_rem_start - num_rem_end);
                                fm.ibase_obj[io] += num_rem_start;
                            }
                        }
                    }
                    num_pts_tot += fm.npt_in_obj[io];
                    if fm.npt_in_obj[io] == 0 {
                        fm.n_object -= 1;
                    } else {
                        non_empty_new += 1;
                    }
                }
                //
                // Delete points from model in inverse order
                let mut ipt = h.num_to_delete;
                while ipt >= 1 {
                    ierr = deleteimodpoint(imod_obj, iobj, h.ipnt_to_delete[ipt as usize - 1]);
                    if ierr != 0 {
                        exit_error("Deleting point from IMOD model");
                    }
                    ipt -= 1;
                }
            }
            if !keep_empty {
                remove_empty_contours(imod_obj, &mut fm);
            }
            if update_mm {
                if findaddminmax1value(imod_obj) != 0 {
                    exit_error("Adding min/max to object in IMOD model");
                }
            }
            scale_model(1, &mut fm);
            put_model_objects(&mut fm);
            h.imod_obj += 1;
        }
        //
        // write out result, offer to go back for more
        //
        // FORMAT 102
        println!(
            "\n Number of non-empty contours reduced from{} to{}\n Number of points reduced from{} to{}\n",
            fmt_i(non_empty_old, 8),
            fmt_i(non_empty_new, 9),
            fmt_i(num_pts_tot_old, 10),
            fmt_i(num_pts_tot, 10)
        );
        RESULT_SINK.with_borrow(|slot| {
            if let Some(sink) = slot {
                let mut result = sink.lock().expect("clipmodel result sink");
                result.contours_reduced.push((non_empty_old, non_empty_new));
                result.points_reduced.push((num_pts_tot_old, num_pts_tot));
            }
        });
        //
        if cut_sum > 0. {
            // FORMAT 103
            println!(
                " Total length of cut edges is{} {}\n Approximate surface area of cut edges is{} square {}\n",
                format_f(cut_sum as f64, 15, 5),
                pixel_microns,
                format_f((cut_sum * z_full_scale) as f64, 18, 8),
                pixel_microns
            );
        }
        //
        if !pip_input {
            print!(" 0 to write out results, 1 to enter new block to include or exclude: ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut if_loop)],
            ) {
                read_abort(err);
            }
            if if_loop == 0 {
                break;
            }
        }
        loop_ += 1;
    }
    //
    // now either just write model, or write point file
    //
    if if_point == 0 {
        fm.n_point = -1;
        write_wmod(&model_file, &mut fm);
        println!(" CLIPPED MODEL WRITTEN");
    } else {
        let mut unit1 = std::io::BufWriter::new(dopen(1, &model_file, "new", "f"));
        if !get_model_object_range(1, num_obj_tot, &mut fm) {
            exit_error("Reloading data to make point output");
        }
        //
        // This should be done before point output
        scale_model(0, &mut fm);
        let ix_min = xmin.round() as i32;
        let iy_min = ymin.round() as i32;
        let iz_min = zmin.round() as i32;
        let ix_max = xmax.round() as i32;
        let iy_max = ymax.round() as i32;
        let iz_max = zmax.round() as i32;
        let mut nlistz = 0usize;
        //
        // loop through objects again
        //
        for iobj in 1..=fm.max_mod_obj as usize {
            let ibase = fm.ibase_obj[iobj - 1];
            for ipt in 1..=fm.npt_in_obj[iobj - 1] {
                let jpnt = fm.object[(ipt + ibase) as usize - 1].abs() as usize - 1;
                let ixx = ix_min.max(ix_max.min(fm.p_coord[jpnt][0].round() as i32));
                let iyy = iy_min.max(iy_max.min(fm.p_coord[jpnt][1].round() as i32));
                let izz = iz_min.max(iz_max.min(fm.p_coord[jpnt][2].round() as i32));
                //
                // if need corners, see if z coord is on list of z's
                //
                if if_point < 0 {
                    let mut in_list = 0;
                    for il in 1..=nlistz {
                        if izz == listz[il - 1] {
                            in_list = il;
                        }
                    }
                    //
                    // if not, add to list of z's and write corner
                    //
                    if in_list == 0 {
                        nlistz += 1;
                        listz[nlistz - 1] = izz;
                        let _ = writeln!(
                            unit1,
                            "{}{}{}",
                            fmt_i(ix_min, 8),
                            fmt_i(iy_min, 8),
                            fmt_i(izz, 8)
                        );
                        let _ = writeln!(
                            unit1,
                            "{}{}{}",
                            fmt_i(ix_max, 8),
                            fmt_i(iy_max, 8),
                            fmt_i(izz, 8)
                        );
                    }
                }
                let _ = writeln!(unit1, "{}{}{}", fmt_i(ixx, 8), fmt_i(iyy, 8), fmt_i(izz, 8));
            }
        }
        let _ = unit1.flush();
        drop(unit1);
        println!(" POINT LIST WRITTEN");
    }
    exit(0);
}

/// Original: `removeEmptyContours` (`clipmodel.f90:630`).
///
/// Removes all empty contours for the given object.  4/28/20: modified to
/// assume what is true, that one object is loaded, and to call efficient
/// function.
pub fn remove_empty_contours(loaded_obj: i32, fm: &mut FortModel) {
    let mut iobj_to_delete = vec![0i32; fm.max_mod_obj.max(0) as usize];
    let mut num_keep = 0usize;
    let mut num_delete = 0usize;
    for iobj in 1..=fm.max_mod_obj as usize {
        if fm.npt_in_obj[iobj - 1] > 0 {
            num_keep += 1;
            fm.npt_in_obj[num_keep - 1] = fm.npt_in_obj[iobj - 1];
            fm.ibase_obj[num_keep - 1] = fm.ibase_obj[iobj - 1];
            fm.obj_color[num_keep - 1][0] = fm.obj_color[iobj - 1][0];
            fm.obj_color[num_keep - 1][1] = fm.obj_color[iobj - 1][1];
        } else {
            num_delete += 1;
            iobj_to_delete[num_delete - 1] = iobj as i32;
        }
    }
    fm.max_mod_obj = num_keep as i32;
    if num_delete == 0 {
        return;
    }
    if deletelistofconts(loaded_obj, &mut iobj_to_delete[..num_delete]) != 0 {
        exit_error("Deleting contours from IMOD model");
    }
}

/// Original: `contourArea` (`clipmodel.f90:659`).
pub fn contour_area(iobj: i32, ind_y: i32, ind_z: i32, z_scale: f32, fm: &FortModel) -> f32 {
    let mut contour_area: f32 = 0.;
    let num_in_obj = fm.npt_in_obj[iobj as usize - 1];
    let ibase = fm.ibase_obj[iobj as usize - 1];
    if num_in_obj < 3 {
        return contour_area;
    }
    let mut xnorm: f32 = 0.;
    let mut ynorm: f32 = 0.;
    let mut znorm: f32 = 0.;
    let iy = ind_y as usize - 1;
    let iz = ind_z as usize - 1;
    for ipt in 1..=num_in_obj {
        let mut jpnt = fm.object[(ipt + ibase) as usize - 1].abs() as usize - 1;
        let xx = fm.p_coord[jpnt][0];
        let yy = fm.p_coord[jpnt][iy];
        let zz = fm.p_coord[jpnt][iz] * z_scale;
        jpnt = fm.object[(ipt % num_in_obj + 1 + ibase) as usize - 1].abs() as usize - 1;
        let xnext = fm.p_coord[jpnt][0];
        let ynext = fm.p_coord[jpnt][iy];
        let znext = fm.p_coord[jpnt][iz] * z_scale;
        xnorm += yy * znext - zz * ynext;
        ynorm += zz * xnext - xx * znext;
        znorm += xx * ynext - yy * xnext;
    }
    contour_area = (xnorm * xnorm + ynorm * ynorm + znorm * znorm).sqrt() * 0.5;
    contour_area
}
