//! Translation of `IMOD/flib/model/ccderaser.f90`.
//!
//! The `ccdvars` module is [`CcdVars`], passed by reference to every unit
//! that `use`s it; the `fortmodel` module is the [`FortModel`] that
//! `readw_or_imod` fills.  The main program is [`ccderaser`], its contained
//! function `contIsBelowThresh` is [`cont_is_below_thresh`] with the host
//! variables it reads and writes passed explicitly.  `searchPeaks` is
//! [`search_peaks`]; its locals live in [`SearchPeaksLocals`] so that its
//! five contained procedures ([`check_and_erase_peak`], [`add_point_to_patch`],
//! [`clean_area_recompute_diffs`], [`clean_edge_for_diffs`],
//! [`neighbor_mean_with_tests`]) reach them by host association as the source
//! does.  The external subroutines and functions follow one to one.
//!
//! Arrays keep the source's column-major layout and 1-based indices:
//! `array(ix, iy)` is `array[(ix - 1) + (iy - 1) * nx]`, computed as a linear
//! address exactly as gfortran (no bounds checking) computes it, so a read
//! one column outside the image (`cleanLine` next to an edge) lands on the
//! same element native reads.  `inList`/`adjacent(-maxDev:maxDev, ...)` are
//! flat with the offset added.
//!
//! The `SAVE`d locals of `computeDiffs` and `cleanArea` (`warned`) are fields
//! of [`CcdVars`], so they persist for the program's life as `SAVE` does and
//! start fresh for each in-process run.  The automatic arrays of `cleanArea`
//! (`adjValue`, `rmat`, sized by `LIMPATCH`) are held there too and reused:
//! the source reads only elements it has written in the same call.
//!
//! Output goes to the C stdout ([`ImodFile::Stdout`]) that `exitError` and
//! `checklist` also write, with the gfortran editing of each descriptor
//! reproduced by the functions at the end of this file.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdsec};
use crate::imod::flib::subrs::model::fortmodel::{FortModel, allocate_fort_model};
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model_to_image;
use crate::imod::flib::subrs::model::write_wmod::write_wmod;
use crate::imod::flib::subrs::piecesubs::lookup_piece::lookup_piece;
use crate::imod::flib::subrs::piecesubs::read_piece_list::read_piece_list2;
use crate::imod::flib::subrs::statsubs::polyterm::poly_term;
use crate::imod::libcfshr::b3dutil::{ImodFile, exit, pid_to_stderr};
use crate::imod::libcfshr::filtxcorr::{apply_kernel_filter, scaled_gaussian_kernel};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_integer,
    pip_get_two_integers,
};
use crate::imod::libcfshr::piecefuncs::{adjustpieceoverlap, checklist};
use crate::imod::libcfshr::regression::mult_regress;
use crate::imod::libcfshr::robuststat::{
    rs_fast_median_in_place, rs_median_of_sorted, rs_sort_floats,
};
use crate::imod::libcfshr::simplestat::{
    array_min_max_mean_fortran, array_min_max_mean_sd_fortran, sums_to_avg_sd, sums_to_avg_sd_dbl,
};
use crate::imod::libcfshr::statfuncs::gaussian_deviate;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::{
    iiu_alt_mode, iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt, iiu_trans_header, iiu_write_header,
};
use crate::imod::libimod::imodel_fwrap::{
    getcontpointsizes, getcontvalue, getimodobjsize, getimodsizes, getobjskiplowvalues,
    getobjvaluethresh, newimod, putimageref, putimodflag, putimodmaxes, putsymsize, putsymtype,
};
use std::io::{BufRead, Write};

/// `parameter (LIMDIFF = 512, ...)` (`ccderaser.f90:27`).
const LIMDIFF: i32 = 512;
/// `parameter (..., LIMPATCHOUT = 40000)` (`ccderaser.f90:27`).
const LIMPATCHOUT: i32 = 40000;
/// `parameter (LIMPATCH = 40000, ...)` (`ccderaser.f90:28`).
const LIMPATCH: i32 = 40000;
/// `parameter (..., LIMPTOUT = 25 * LIMPATCHOUT)` (`ccderaser.f90:28`).
const LIMPTOUT: i32 = 25 * LIMPATCHOUT;
/// `parameter (numOptions = 40)` (`ccderaser.f90:97`).
const NUM_OPTIONS: i32 = 40;
/// Fallback PIP table `options(1)` (`ccderaser.f90:99-114`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@\
halffloat:HalfFloatModeOutput:I:@piece:PieceListFile:FN:@\
overlaps:OverlapsForModel:IP:@find:FindPeaks:B:@peak:PeakCriterion:F:@\
diff:DiffCriterion:F:@grow:GrowCriterion:F:@scan:ScanCriterion:F:@\
radius:MaximumRadius:F:@giant:GiantCriterion:F:@large:ExtraLargeRadius:F:@\
big:BigDiffCriterion:F:@maxdiff:MaxPixelsInDiffPatch:I:@outer:OuterRadius:F:@\
width:AnnulusWidth:F:@xyscan:XYScanSize:I:@edge:EdgeExclusionWidth:I:@\
iterations:SearchIterations:I:@points:PointModel:FN:@model:ModelFile:FN:@\
lines:LineObjects:LI:@boundary:BoundaryObjects:LI:@\
allsec:AllSectionObjects:LI:@circle:CircleObjects:LI:@better:BetterRadius:FA:@\
expand:ExpandCircleIterations:I:@halo:IncludeHaloInExpand:I:@\
target:TargetRadiusForExpand:F:@merge:MergePatches:B:@\
skip:SkipTurnedOffPoints:B:@border:BorderSize:I:@order:PolynomialOrder:I:@\
exclude:ExcludeAdjacent:B:@trial:TrialMode:B:@verbose:Verbose:B:@\
PID:ProcessID:B:@param:ParameterFile:PF:@help:usage:B:";

/// Linear index of the column-major element `(ix, iy)` of an array whose
/// leading dimension is `nx`, with the source's 1-based indices.
macro_rules! at {
    ($nx:expr, $ix:expr, $iy:expr) => {
        ((($ix) - 1) as isize + ((($iy) - 1) as isize) * (($nx) as isize)) as usize
    };
}

/// Original module `ccdvars` (`ccderaser.f90:24-43`), plus the `SAVE`d
/// locals of `computeDiffs` and `cleanArea` and `cleanArea`'s automatic
/// arrays (see the module comment).
pub struct CcdVars {
    pub num_pix_border: i32,
    pub iorder: i32,
    pub if_include_adj: i32,
    pub i_scan_size: i32,
    pub num_edge_pixels: i32,
    pub if_verbose: i32,
    pub mat_size: i32,
    pub max_in_diff_patch: i32,
    pub nx_alloc_scan: i32,
    pub ny_alloc_scan: i32,
    pub noise_extra_bord: i32,
    /// `nxyz(3)`, equivalenced with `nx`, `ny`, `nz`.
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub max_dev: i32,
    /// `logical*1 inList(-maxDev:maxDev, -maxDev:maxDev)`.
    pub in_list: Vec<bool>,
    /// `integer*2 adjacent(-maxDev:maxDev, -maxDev:maxDev)`.
    pub adjacent: Vec<i16>,
    pub scan_overlap: f32,
    pub crit_scan: f32,
    pub crit_grow: f32,
    pub crit_big_diff: f32,
    pub frac_big_diff: f32,
    pub crit_main: f32,
    pub crit_diff: f32,
    pub radius_max: f32,
    pub outer_radius: f32,
    pub crit_giant: f32,
    pub giant_radius: f32,
    /// `ixFix(LIMPATCH)`, `iyFix(LIMPATCH)`.
    pub ix_fix: Vec<i32>,
    pub iy_fix: Vec<i32>,
    pub ind_patch: Vec<i32>,
    pub ix_out: Vec<i32>,
    pub iy_out: Vec<i32>,
    pub iz_out: Vec<i32>,
    pub exceed_crit: Vec<f32>,
    /// `scanAreaChanged(nxAllocScan, nyAllocScan)`.
    pub scan_area_changed: Vec<bool>,
    /// `vicinity(vicDim, vicDim)`, `vicinitySqr`, `sdArray(vicDim**2)`.
    pub vicinity: Vec<f32>,
    pub vicinity_sqr: Vec<f32>,
    pub vic_dim: i32,
    pub sd_array: Vec<f32>,
    pub fill_with_noise: bool,
    pub include_halo: bool,
    /// `computeDiffs`: `save numSum, sum8, sumsq8, abssum8, abssq8`.
    cd_num_sum: i32,
    cd_sum8: f64,
    cd_sumsq8: f64,
    cd_abssum8: f64,
    cd_abssq8: f64,
    /// `cleanArea`: `logical warned /.false./`, `save warned`.
    ca_warned: bool,
    /// `cleanArea`'s automatic `adjValue(LIMPATCH)`.
    ca_adj_value: Vec<f32>,
    /// `cleanArea`'s automatic `rmat(matSize, LIMPATCH)`.
    ca_rmat: Vec<f32>,
}

/// Original program `ccderaser` (`ccderaser.f90:45`).
///
/// This program replaces deviant pixels with interpolated values from
/// surrounding pixels.  It is designed to correct defects in electron
/// microscope images from CCD cameras.
pub fn ccderaser() {
    const LIMKERNEL: i32 = 15;
    let mut delta = [0.0_f32; 3];
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut in_file = String::new();
    let mut out_file: String;
    let mut point_file = String::new();
    let mut model_out: String;
    let mut ix_pc_list: Vec<i32> = Vec::new();
    let mut iy_pc_list: Vec<i32> = Vec::new();
    let mut iz_pc_list: Vec<i32> = Vec::new();
    let mut dat = [b' '; 9];
    let mut tim = [b' '; 8];
    let mut line = String::new();
    let mut mode = 0_i32;
    let mut im_file_out: i32;
    let mut num_obj_do_all: i32;
    let mut num_obj_line: i32;
    let mut num_obj_bound: i32;
    let mut num_obj_circle: i32;
    let mut ierr: i32 = 0;
    let mut ierr2: i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut tmin, mut tmax, mut tsum): (f32, f32, f32);
    let (mut dmin_tmp, mut dmax_tmp, mut dmeant) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut zmin, mut zmax, mut xmin, mut xmax, mut y_min, mut ymax): (
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
    );
    let mut diam_merge: f32;
    let mut cur_obj_thresh = 0.0_f32;
    let mut cont_value = 0.0_f32;
    let mut num_pt_out: i32;
    let mut num_patch_out: i32;
    let num_obj_orig: i32;
    let mut if_merge: i32;
    let mut if_half_float: i32;
    let mut annulus_width = 0.0_f32;
    let mut rad_sq: f32;
    let mut size_max: f32;
    let mut size: f32;
    let (mut xcen, mut ycen, mut dist): (f32, f32, f32);
    let mut rough_mean: f32;
    let mut expand_target_rad: f32;
    let mut smooth_sigma: f32;
    let mut smooth_kernel = vec![0.0_f32; (LIMKERNEL * LIMKERNEL) as usize];
    let mut if_peak_search: i32;
    let (mut num_patch, mut num_pixels): (i32, i32);
    let mut if_touch: i32;
    let mut num_search_iterations: i32;
    let mut if_trial_mode: i32;
    let max_objects_out: i32;
    let mut if_grew: i32;
    let mut num_patch_tmp: i32;
    let mut num_pixels_tmp = 0_i32;
    let mut i_border = 0_i32;
    let mut j_bord_low: i32;
    let mut j_bord_high: i32;
    let mut num_better_in = 0_i32;
    let mut num_circle_obj: i32;
    let mut num_expand_iter: i32;
    let mut ind_free: i32;
    let mut ind_cont: i32;
    let mut num_sizes = 0_i32;
    let mut num_con_size = 0_i32;
    let (mut ix_fix_min, mut ix_fix_max, mut iy_fix_min, mut iy_fix_max): (i32, i32, i32, i32);
    let mut num_grow_iter: i32;
    let taperedpatch_crit: i32;
    let mut max_sizes: i32;
    let mut lim_sizes: i32;
    let mut max_bound: i32;
    let mut num_tapering: i32;
    let mut lim_pc_list: i32;
    let mut num_pc_list: i32;
    let (mut min_xpiece, mut num_xpieces, mut nx_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut min_ypiece, mut num_ypieces, mut ny_overlap) = (0_i32, 0_i32, 0_i32);
    let mut new_xoverlap: i32;
    let mut new_yoverlap: i32;
    let (mut ipcx, mut ipcy, mut ipcz) = (0_i32, 0_i32, 0_i32);
    let mut last_obj_with_thresh: i32;
    let mut kernel_dim = 0_i32;
    let mut circle_cont: bool;
    let mut all_sec_cont: bool;
    let mut cont_on_sec_or_all_sec: bool;
    let mut skip_low_values: bool;
    let mut low_turned_off = false;
    let mut nearby: bool;
    let mut lim_obj: i32;
    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    let (mut imod_obj, mut imod_cont) = (0_i32, 0_i32);
    // `use fortmodel`
    let mut fm = FortModel::default();
    let mut out = ImodFile::Stdout;

    // `read(*,'(a)') name` with no `END=`.
    let read_a = || -> String {
        let mut text = String::new();
        if matches!(std::io::stdin().lock().read_line(&mut text), Ok(0) | Err(_)) {
            let _ = ImodFile::Stdout.flush();
            eprintln!("Fortran runtime error: End of file");
            exit(2);
        }
        text.trim_end_matches(['\r', '\n']).to_owned()
    };
    // The Fortran wrapper `pipgetstring` (`pip_fwrap.c:206`): the variable is
    // left untouched unless the option is found.
    // `c2fString` there copies at most the variable's declared length
    // (ccderaser.f90:53,62: file names `character*320`, `line` `character*1024`); a longer entry fails with `In PipGetString, string is too
    // long for character variable`, which exits under the exit prefix.
    let get_string = |option: &[u8], length: usize, string: &mut String| -> i32 {
        let mut record = vec![b' '; length];
        let err = crate::imod::libcfshr::pip_fwrap::pipgetstring_(option, &mut record);
        if err == 0 {
            *string = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        }
        err
    };
    let blank = |string: &str| string.bytes().all(|b| b == b' ');

    let mut v = CcdVars {
        num_pix_border: 0,
        iorder: 0,
        if_include_adj: 0,
        i_scan_size: 0,
        num_edge_pixels: 0,
        if_verbose: 0,
        mat_size: 0,
        max_in_diff_patch: 0,
        nx_alloc_scan: 0,
        ny_alloc_scan: 0,
        noise_extra_bord: 0,
        nx: 0,
        ny: 0,
        nz: 0,
        max_dev: 0,
        in_list: Vec::new(),
        adjacent: Vec::new(),
        scan_overlap: 0.,
        crit_scan: 0.,
        crit_grow: 0.,
        crit_big_diff: 0.,
        frac_big_diff: 0.,
        crit_main: 0.,
        crit_diff: 0.,
        radius_max: 0.,
        outer_radius: 0.,
        crit_giant: 0.,
        giant_radius: 0.,
        ix_fix: vec![0; LIMPATCH as usize],
        iy_fix: vec![0; LIMPATCH as usize],
        ind_patch: Vec::new(),
        ix_out: Vec::new(),
        iy_out: Vec::new(),
        iz_out: Vec::new(),
        exceed_crit: Vec::new(),
        scan_area_changed: Vec::new(),
        vicinity: Vec::new(),
        vicinity_sqr: Vec::new(),
        vic_dim: 0,
        sd_array: Vec::new(),
        fill_with_noise: false,
        include_halo: false,
        cd_num_sum: 0,
        cd_sum8: 0.,
        cd_sumsq8: 0.,
        cd_abssum8: 0.,
        cd_abssq8: 0.,
        ca_warned: false,
        ca_adj_value: Vec::new(),
        ca_rmat: Vec::new(),
    };
    //
    // Set all defaults here
    //
    num_obj_do_all = 1;
    num_obj_line = 1;
    num_obj_bound = 1;
    num_obj_circle = 1;
    v.num_pix_border = 2; // Default in adoc
    v.iorder = 2; // Default in adoc
    v.if_include_adj = 1;
    v.crit_main = 10.; // Default in adoc
    v.crit_diff = 10.; // Default in adoc
    v.crit_grow = 4.0; // Default in adoc
    v.crit_scan = 3.0; // Default in adoc
    v.radius_max = 2.1; // Default in adoc
    v.outer_radius = 4.1;
    v.crit_giant = 12.; // Default in adoc
    v.giant_radius = 8.; // Default in adoc
    v.crit_big_diff = 19.; // Default in adoc
    v.frac_big_diff = 0.25;
    v.scan_overlap = 0.1;
    if_peak_search = 0;
    v.i_scan_size = 100; // Default in adoc
    v.if_verbose = 0;
    if_trial_mode = 0;
    v.num_edge_pixels = 0; // Default in adoc
    v.max_in_diff_patch = 2; // Default in adoc
    v.noise_extra_bord = 5;
    if_merge = 0;
    max_objects_out = 4;
    num_expand_iter = 0;
    taperedpatch_crit = 1000;
    lim_pc_list = 10000;
    new_xoverlap = 0;
    new_yoverlap = 0;
    model_out = String::new();
    if_half_float = 0;
    smooth_sigma = 0.;
    expand_target_rad = 0.;
    v.include_halo = false;
    num_search_iterations = 3; // Default in adoc
    skip_low_values = false;
    last_obj_with_thresh = -1;
    v.nx_alloc_scan = 0;
    v.ny_alloc_scan = 0;
    v.max_dev = 300; // Latest fixed value
    b3d_date(&mut dat);
    time(&mut tim);
    allocate_fort_model(&mut fm);
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "ccderaser",
        "ERROR: CCDERASER - ",
        true,
        2,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    if pip_get_in_out_file("InputFile", 1, "Name of input file", &mut in_file, 320) != 0 {
        exit_error("No input file specified");
    }
    //
    out_file = String::new();
    if pip_input {
        //
        // Get an output file string; or if none, look on command line
        //
        pip_get_boolean(b"TrialMode", &mut if_trial_mode);
        ierr = get_string(b"OutputFile", 320, &mut out_file);
        if ierr > 0 && num_non_opt_arg > 1 {
            // The Fortran wrapper takes the 1-based argument number.
            // `PipGetNonOptionArg(2, outFile)` into the `character*320`
            // (`pip_fwrap.c:190`).
            let mut record = [b' '; 320];
            ierr = crate::imod::libcfshr::pip_fwrap::pipgetnonoptionarg_(2, &mut record);
            if ierr == 0 {
                out_file = crate::imod::libcfshr::b3dutil::fortran_string(&record);
            }
        }
        pip_get_integer(b"HalfFloatModeOutput", &mut if_half_float);
    } else {
        let _ = writeln!(
            out,
            " Enter output file name (Return to put modified sections back in input file)"
        );
        let _ = out.flush();
        out_file = read_a();
    }
    //
    if if_trial_mode != 0 || !blank(&out_file) {
        imopen(1, in_file.trim_end_matches(' '), "ro");
    } else {
        imopen(1, in_file.trim_end_matches(' '), "old");
    }

    let _ = out.flush();
    // SAFETY: `irdhdr` writes three integers into each of `nxyz` and `mxyz`
    // and one value through each scalar pointer, all live locals.
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
    }
    let _ = std::io::stdout().flush();
    v.nx = nxyz[0];
    v.ny = nxyz[1];
    v.nz = nxyz[2];
    let (nx, ny, nz) = (v.nx, v.ny, v.nz);
    delta = iiu_ret_delta(1);

    im_file_out = 1;
    if if_trial_mode == 0 && !blank(&out_file) {
        im_file_out = 2;
        let _ = out.flush();
        imopen(2, out_file.trim_end_matches(' '), "new");
        iiu_trans_header(2, 1);
        if if_half_float > 1 || (if_half_float == 1 && mode == 2) {
            iiu_alt_mode(2, 12);
        }
    }
    //
    fm.max_mod_obj = 0;
    fm.n_point = 0;
    fm.ibase_free = 0;
    num_pc_list = 0;
    if pip_input {
        point_file = String::new();
        get_string(b"PieceListFile", 320, &mut point_file);
        pip_get_two_integers(b"OverlapsForModel", &mut new_xoverlap, &mut new_yoverlap);
        lim_pc_list = lim_pc_list.max(nz + 10);
        ix_pc_list = vec![0; lim_pc_list as usize];
        iy_pc_list = vec![0; lim_pc_list as usize];
        iz_pc_list = vec![0; lim_pc_list as usize];
        memory_error(0, "arrays for piece coordinates");
        read_piece_list2(
            point_file.trim_end_matches(' '),
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
            lim_pc_list,
        );
        if num_pc_list > 0 {
            if num_pc_list < nz {
                exit_error("Not enough piece coordinates in file");
            }
            checklist(
                &ix_pc_list[..num_pc_list as usize],
                1,
                nx,
                &mut min_xpiece,
                &mut num_xpieces,
                &mut nx_overlap,
            );
            checklist(
                &iy_pc_list[..num_pc_list as usize],
                1,
                ny,
                &mut min_ypiece,
                &mut num_ypieces,
                &mut ny_overlap,
            );
            if num_xpieces < 1 || num_ypieces < 1 {
                exit_error("Piece coordinates are not regularly spaced apart");
            }
            adjustpieceoverlap(
                &mut ix_pc_list[..num_pc_list as usize],
                nx,
                min_xpiece,
                nx_overlap,
                new_xoverlap,
            );
            adjustpieceoverlap(
                &mut iy_pc_list[..num_pc_list as usize],
                ny,
                min_ypiece,
                ny_overlap,
                new_yoverlap,
            );
        }

        ierr = get_string(b"ModelFile", 320, &mut point_file);
    } else {
        let _ = write!(out, " Model file: ");
        let _ = out.flush();
        point_file = read_a();
    }
    lim_obj = 10;
    if !pip_input || ierr == 0 {
        let _ = out.flush();
        if !readw_or_imod(point_file.trim_end_matches(' '), &mut fm) {
            exit_error("Reading model file");
        }
        scale_model_to_image(1, 0, &mut fm);
        pip_get_logical("SkipTurnedOffPoints", &mut skip_low_values);
        //
        // Convert model coordinates to coordinates within each piece
        if num_pc_list > 0 {
            for ipt in 1..=fm.n_point {
                let p = fm.p_coord[(ipt - 1) as usize];
                let indx = p[0] as i32;
                let indy = p[1] as i32;
                let indz = p[2].round() as i32;
                lookup_piece(
                    &ix_pc_list,
                    &iy_pc_list,
                    &iz_pc_list,
                    num_pc_list,
                    nx,
                    ny,
                    indx,
                    indy,
                    indz,
                    &mut ipcx,
                    &mut ipcy,
                    &mut ipcz,
                );
                if ipcx < 0 {
                    // write(pointFile, '(a, 3i6, a)')
                    point_file = format!(
                        "The model point at{}{}{} is not located within a piece",
                        i_edit(indx + 1, 6),
                        i_edit(indy + 1, 6),
                        i_edit(indz + 1, 6)
                    );
                    exit_error(&point_file);
                }
                let pc = &mut fm.p_coord[(ipt - 1) as usize];
                pc[0] = ipcx as f32 + pc[0] - indx as f32;
                pc[1] = ipcy as f32 + pc[1] - indy as f32;
                pc[2] = ipcz as f32;
            }
        }
        lim_obj = getimodobjsize() + 10;
    }
    num_obj_orig = fm.max_mod_obj;

    let mut diff_arr = vec![0.0_f32; (LIMDIFF * LIMDIFF) as usize];
    v.exceed_crit = vec![0.0; LIMPATCHOUT as usize];
    v.ix_out = vec![0; LIMPTOUT as usize];
    v.iy_out = vec![0; LIMPTOUT as usize];
    v.iz_out = vec![0; LIMPTOUT as usize];
    v.ind_patch = vec![0; LIMPATCHOUT as usize];
    memory_error(0, "arrays for patches");
    // The four object lists are `allocate`d with `limObj` entries and filled
    // by `parseList` without a limit, so a list longer than that writes past
    // the allocation natively; they are allocated with room to spare here.
    // Nothing reads an entry past the parsed count, so only that overflow
    // differs.
    let lim_list = (lim_obj as usize).max(100000);
    let mut iobj_circle = vec![0_i32; lim_list];
    let mut iobjline = vec![0_i32; lim_list];
    let mut iobj_do_all = vec![0_i32; lim_list];
    let mut iobj_bound = vec![0_i32; lim_list];
    let mut better_radius = vec![0.0_f32; lim_obj as usize];
    let mut better_in = vec![0.0_f32; lim_obj as usize];
    memory_error(0, "arrays for model object data");
    for i in 1..=lim_obj {
        better_radius[(i - 1) as usize] = -1.;
    }
    iobj_do_all[0] = -999;
    iobjline[0] = -999;
    iobj_bound[0] = -999;
    iobj_circle[0] = -999;
    //
    if pip_input {
        //
        // get old parameters for model based erasing
        //
        if get_string(b"AllSectionObjects", 1024, &mut line) == 0 {
            let _ = parselist(&line, &mut iobj_do_all, &mut num_obj_do_all);
        } else {
            num_obj_do_all = 0;
        }

        if get_string(b"LineObjects", 1024, &mut line) == 0 {
            let _ = parselist(&line, &mut iobjline, &mut num_obj_line);
        } else {
            num_obj_line = 0;
        }

        if get_string(b"CircleObjects", 1024, &mut line) == 0 {
            let _ = parselist(&line, &mut iobj_circle, &mut num_obj_circle);
        } else {
            num_obj_circle = 0;
        }
        if get_string(b"BoundaryObjects", 1024, &mut line) == 0 {
            let _ = parselist(&line, &mut iobj_bound, &mut num_obj_bound);
        } else {
            num_obj_bound = 0;
        }
        ierr2 = 0;
        pip_get_boolean(b"ProcessID", &mut ierr2);
        if ierr2 != 0 {
            let _ = out.flush();
            pid_to_stderr();
        }

        pip_get_integer(b"ExpandCircleIterations", &mut num_expand_iter);
        pip_get_logical("IncludeHaloInExpand", &mut v.include_halo);
        pip_get_integer(b"BorderSize", &mut v.num_pix_border);
        pip_get_integer(b"PolynomialOrder", &mut v.iorder);
        let mut i = 0_i32;
        // Fixed in translation (BUGS.md): the source (`ccderaser.f90:315`) tests
        // only that the boolean was entered, so `ExcludeAdjacent 0` also
        // excludes; the value entered is honoured here.
        if pip_get_boolean(b"ExcludeAdjacent", &mut i) == 0 && i != 0 {
            v.if_include_adj = 0;
        }
        //
        // get the new parameters for auto peak search
        //
        pip_get_boolean(b"FindPeaks", &mut if_peak_search);
        pip_get_integer(b"XYScanSize", &mut v.i_scan_size);
        pip_get_integer(b"EdgeExclusionWidth", &mut v.num_edge_pixels);
        pip_get_float(b"MaximumRadius", &mut v.radius_max);
        pip_get_float(b"PeakCriterion", &mut v.crit_main);
        pip_get_float(b"GrowCriterion", &mut v.crit_grow);
        pip_get_float(b"ScanCriterion", &mut v.crit_scan);
        pip_get_float(b"DiffCriterion", &mut v.crit_diff);
        pip_get_integer(b"verbose", &mut v.if_verbose);
        pip_get_integer(b"MergePatches", &mut if_merge);
        pip_get_integer(b"MaxPixelsInDiffPatch", &mut v.max_in_diff_patch);
        ierr = pip_get_float(b"OuterRadius", &mut v.outer_radius);
        ierr2 = pip_get_float(b"AnnulusWidth", &mut annulus_width); // Default in adoc
        if ierr == 0 && ierr2 == 0 {
            exit_error("You cannot enter both -outer and -width");
        }
        if ierr2 == 0 {
            v.outer_radius = v.radius_max + annulus_width;
        }
        pip_get_float(b"ExtraLargeRadius", &mut v.giant_radius);
        pip_get_float(b"GiantCriterion", &mut v.crit_giant);
        pip_get_float(b"BigDiffCriterion", &mut v.crit_big_diff);
        pip_get_integer(b"SearchIterations", &mut num_search_iterations);
        num_search_iterations = 1.max(num_search_iterations);
        pip_get_float(b"TargetRadiusForExpand", &mut expand_target_rad);

        get_string(b"PointModel", 320, &mut model_out);
        num_better_in = 0;
        pip_get_float_array(b"BetterRadius", &mut better_in, &mut num_better_in, lim_obj);
        if num_obj_circle > 0 && iobj_circle[0] != -999 {
            if num_better_in > 1 && num_better_in != num_obj_circle {
                exit_error(
                    "The number of better radius values must be either 1 or the same as the number of circle objects",
                );
            }
            for i in 1..=num_obj_circle {
                let io = iobj_circle[(i - 1) as usize];
                if io > 0 && io <= lim_obj {
                    // Fixed in translation (BUGS.md): with no `-better` entry
                    // the source reads `betterIn(0)`, one element before the
                    // allocation (glibc's chunk-size word, 0 in practice).
                    // No better radius is defined as 0 here, which the only
                    // reader (`betterRadius(itype) > 0`, `:545`) treats as
                    // "none", like the initial -1.
                    let k = i.min(num_better_in);
                    better_radius[(io - 1) as usize] = if k == 0 {
                        0.
                    } else {
                        better_in[(k - 1) as usize]
                    };
                }
            }
        }
        // Fixed in translation (BUGS.md): the source (`ccderaser.f90:354-355`)
        // prints a leftover debug line, `[CCE1]` with the value and the X
        // pixel size, when `-better` has exactly one value; it is not printed.
    } else {
        //
        // interactive input for old parameters only
        //
        let _ = writeln!(
            out,
            " Enter list of objects specifying replacement on ALL sections (/ for all, Return for none)"
        );
        let _ = out.flush();
        //
        let _ = rdlist(
            &mut std::io::stdin().lock(),
            &mut iobj_do_all,
            &mut num_obj_do_all,
        );
        //
        let _ = writeln!(
            out,
            " Enter list of objects specifying horizontal or vertical line replacements (/ for all, Return for none)"
        );
        let _ = out.flush();
        //
        let _ = rdlist(
            &mut std::io::stdin().lock(),
            &mut iobjline,
            &mut num_obj_line,
        );
        //
        let _ = write!(out, " border size [/ for{}]: ", i_edit(v.num_pix_border, 2));
        let _ = out.flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut v.num_pix_border)],
        ) {
            read_runtime_error(err);
        }
        //
        let _ = write!(
            out,
            " order of 2-D polynomial fit [/ for{}]: ",
            i_edit(v.iorder, 2)
        );
        let _ = out.flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut v.iorder)],
        ) {
            read_runtime_error(err);
        }
        //
        let _ = write!(
            out,
            " 0 to exclude or 1 to include adjacent points in fit: "
        );
        let _ = out.flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut v.if_include_adj)],
        ) {
            read_runtime_error(err);
        }
        // Fixed in translation (BUGS.md): the source asks only for the
        // all-section and line lists here, so the circle and boundary lists
        // keep their initial "every object" entries (`numObjCircle =
        // numObjBound = 1`, `-999`, `ccderaser.f90:121-122`) and every run
        // stops with "Object 1 is included in more than one list".  Interactive
        // input is for the old point-replacement parameters only, so the two
        // lists are empty, as when `-circle`/`-boundary` are not entered.
        num_obj_circle = 0;
        num_obj_bound = 0;
    }
    pip_done();

    if fm.max_mod_obj == 0 && if_peak_search == 0 {
        exit_error("No model points and no peak search specified");
    }
    //
    // check input values, set reasonable limits
    //
    v.num_pix_border = 5.min(1.max(v.num_pix_border));
    v.fill_with_noise = v.iorder < 0;
    v.iorder = 3.min(0.max(v.iorder));
    v.mat_size = 2 + (v.iorder * (v.iorder + 3)) / 2;
    v.i_scan_size = (LIMDIFF - 10).min(20.max(v.i_scan_size));
    // `ccderaser.f90:393-394`: the outer `min` is `minss inner, limit` in the
    // reference object.
    v.radius_max = f_min(f_max(0.5, v.radius_max), 10.);
    v.outer_radius = f_min(
        f_max(v.radius_max + 0.75, v.outer_radius),
        v.radius_max + 10.,
    );
    if v.crit_main == 0. {
        v.crit_main = 1000.;
    }
    v.crit_main = f_max(2., v.crit_main);
    if v.crit_diff == 0. {
        v.crit_diff = 1000.;
    }
    v.crit_diff = f_max(2., v.crit_diff);
    v.crit_scan = f_max(2., v.crit_scan);
    v.crit_grow = f_max(2., v.crit_grow);
    v.num_edge_pixels = 0.max((nx / 3).min(v.num_edge_pixels));

    if if_peak_search > 0 {
        v.ind_patch[0] = 1;
    }

    // print *
    let _ = writeln!(out);
    tmin = 1.0e10;
    tmax = -1.0e10;
    tsum = 0.;
    //
    // Check that lists are mutually exclusive
    num_circle_obj = 0;
    for itype in 1..=getimodobjsize() {
        let mut ibase = 0_i32;
        if type_on_list(itype, &iobj_circle, num_obj_circle) {
            ibase = 1;
            if type_on_list(itype, &iobj_do_all, num_obj_do_all) {
                // write(*,106) itype, '...'
                let _ = writeln!(
                    out,
                    "ERROR: CCDERASER - Object{} is on both the circle and the all-section list",
                    i_edit(itype, 4)
                );
                exit(1);
            }
            num_circle_obj += 1;
            if num_obj_circle == 1 && iobj_circle[0] == -999 {
                // `betterIn(0)` with no `-better` entry: defined as 0, as above.
                let k = num_circle_obj.min(num_better_in);
                better_radius[(itype - 1) as usize] = if k == 0 {
                    0.
                } else {
                    better_in[(k - 1) as usize]
                };
            }
        }
        if type_on_list(itype, &iobjline, num_obj_line) {
            ibase += 1;
        }
        if type_on_list(itype, &iobj_bound, num_obj_bound) {
            ibase += 1;
        }
        if ibase > 1 {
            let _ = writeln!(
                out,
                "ERROR: CCDERASER - Object{} is included in more than one list (circle, line, boundary)",
                i_edit(itype, 4)
            );
            exit(1);
        }
    }
    if num_circle_obj > 0 && num_better_in > 1 && num_circle_obj != num_better_in {
        exit_error(
            "THE NUMBER OF BETTER RADIUS VALUES MUST BE EITHER 1 OR THE SAME AS THE NUMBER OF CIRCLE OBJECTS",
        );
    }
    //
    // Determine maximum size needed for boundary array
    max_bound = 0;
    max_sizes = 0;
    lim_sizes = 10;
    for iobj in 1..=fm.max_mod_obj {
        let num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
        let itype = 256 - fm.obj_color[(iobj - 1) as usize][1];
        if type_on_list(itype, &iobj_bound, num_obj_bound) {
            max_bound = max_bound.max(num_in_obj);
        }
        if type_on_list(itype, &iobj_circle, num_obj_circle) {
            max_sizes = max_sizes.max(num_in_obj);
            lim_sizes += num_in_obj;
        }
    }
    let im_size = (nx * ny).max(max_sizes) + 10;
    let mut xbound = vec![0.0_f32; (max_bound + 10) as usize];
    let mut ybound = vec![0.0_f32; (max_bound + 10) as usize];
    let mut obj_taper = vec![false; (fm.max_mod_obj + 10) as usize];
    let mut sizes = vec![0.0_f32; lim_sizes as usize];
    let mut ind_size = vec![0_i32; (fm.max_mod_obj + 10) as usize];
    let mut array = vec![0.0_f32; im_size as usize];
    memory_error(0, "arrays for boundary contours, sizes, and image");

    //
    // Convert boundary objects to interior point collections
    xmin = 0.;
    xmax = 0.;
    y_min = 0.;
    ymax = 0.;
    for iobj in 1..=fm.max_mod_obj {
        let itype = 256 - fm.obj_color[(iobj - 1) as usize][1];
        obj_taper[(iobj - 1) as usize] = false;
        if fm.npt_in_obj[(iobj - 1) as usize] > 2 && type_on_list(itype, &iobj_bound, num_obj_bound)
        {
            if contour_area(iobj, &fm) < taperedpatch_crit as f32 {
                fill_boundary_arrays(
                    iobj,
                    &mut xbound,
                    &mut ybound,
                    &mut xmin,
                    &mut xmax,
                    &mut y_min,
                    &mut ymax,
                    &fm,
                );
                convert_boundary(
                    iobj, nx, ny, &xbound, &ybound, xmin, xmax, y_min, ymax, &mut fm,
                );
            } else {
                obj_taper[(iobj - 1) as usize] = true;
            }
        }
    }
    //
    // Check all but circle objects and tapered patch opjects
    for iobj in 1..=fm.max_mod_obj {
        if fm.npt_in_obj[(iobj - 1) as usize] > 0 {
            let ibase = fm.ibase_obj[(iobj - 1) as usize];
            let itype = 256 - fm.obj_color[(iobj - 1) as usize][1];
            objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);

            if type_on_list(itype, &iobjline, num_obj_line) {
                if fm.npt_in_obj[(iobj - 1) as usize] != 2 {
                    let _ = writeln!(
                        out,
                        "ERROR: CCDERASER - object{}, contour{} does not have exactly two points",
                        i_edit(imod_obj, 4),
                        i_edit(imod_cont, 6)
                    );
                    exit(1);
                }
                let ip1 = fm.object[ibase as usize];
                let ip2 = fm.object[(ibase + 1) as usize];
                let p1 = fm.p_coord[(ip1 - 1) as usize];
                let p2 = fm.p_coord[(ip2 - 1) as usize];
                if p1[2].round() as i32 != p2[2].round() as i32 {
                    let _ = writeln!(
                        out,
                        "ERROR: CCDERASER - object{}, contour{} is supposed to be a line and is not in one Z-plane",
                        i_edit(imod_obj, 4),
                        i_edit(imod_cont, 6)
                    );
                    exit(1);
                } else if (p1[0] + 0.5).round() as i32 != (p2[0] + 0.5).round() as i32
                    && (p1[1] + 0.5).round() as i32 != (p2[1] + 0.5).round() as i32
                {
                    let _ = writeln!(
                        out,
                        "ERROR: CCDERASER - object{}, contour{} is supposed to be a line and is not horizontal or vertical",
                        i_edit(imod_obj, 4),
                        i_edit(imod_cont, 6)
                    );
                    exit(1);
                }
            } else if !type_on_list(itype, &iobj_circle, num_obj_circle)
                && !obj_taper[(iobj - 1) as usize]
            {
                zmin = 1.0e10;
                zmax = -1.0e10;
                xmin = zmin;
                xmax = zmax;
                y_min = zmin;
                ymax = zmax;
                let num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
                if num_in_obj > LIMPATCH {
                    let _ = writeln!(
                        out,
                        "ERROR: CCDERASER - object{}, contour{} has too many points for arrays",
                        i_edit(imod_obj, 4),
                        i_edit(imod_cont, 6)
                    );
                    exit(1);
                }
                for ip in 1..=num_in_obj {
                    let ipt = fm.object[(ibase + ip - 1) as usize];
                    let p = fm.p_coord[(ipt - 1) as usize];
                    xmin = f_min(xmin, p[0]);
                    xmax = f_max(xmax, p[0]);
                    y_min = f_min(y_min, p[1]);
                    ymax = f_max(ymax, p[1]);
                    zmin = f_min(zmin, p[2]);
                    zmax = f_max(zmax, p[2]);
                }
                v.max_dev = v
                    .max_dev
                    .max(2 * (2. + xmax - xmin) as i32)
                    .max(2 * (2. + ymax - y_min) as i32);
                if !type_on_list(itype, &iobj_do_all, num_obj_do_all) && zmax != zmin {
                    // Fixed in translation (BUGS.md): the source
                    // (`ccderaser.f90:520`) writes `write(*,107) 'is not in one
                    // Z-plane'` without the object and contour format 107
                    // edits, so gfortran stops with a runtime error (status
                    // 2).  The intended message, with the numbers, and the
                    // `exit(1)` that follows it are what happen here.
                    let _ = writeln!(
                        out,
                        "ERROR: CCDERASER - object{}, contour{} is not in one Z-plane",
                        i_edit(imod_obj, 4),
                        i_edit(imod_cont, 6)
                    );
                    exit(1);
                }
            }
        }
    }
    //
    // Load point sizes
    ind_free = 1;
    ind_cont = 1;
    size_max = -1.;
    for itype in 1..=getimodobjsize() {
        if type_on_list(itype, &iobj_circle, num_obj_circle) {
            ierr = getimodsizes(
                itype,
                &mut sizes[(ind_free - 1) as usize..],
                lim_sizes + 1 - ind_free,
                &mut num_sizes,
            );
            if ierr != 0 {
                exit_error("Loading point sizes from model");
            }
            //
            // Make indexes to all contours in this object
            for iobj in 1..=fm.max_mod_obj {
                objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);
                if fm.npt_in_obj[(iobj - 1) as usize] > 0 && itype == imod_obj {
                    ind_size[(iobj - 1) as usize] = ind_cont;
                    ind_cont += fm.npt_in_obj[(iobj - 1) as usize];
                    //
                    // If there is a better radius for this object, get the sizes
                    // for this contour and replacethe defaults
                    if better_radius[(itype - 1) as usize] > 0. {
                        getcontpointsizes(
                            imod_obj,
                            imod_cont,
                            &mut array,
                            im_size,
                            &mut num_con_size,
                        );
                        if num_con_size > 0 && num_con_size != fm.npt_in_obj[(iobj - 1) as usize] {
                            exit_error("Mismatch between contour and point array size");
                        }
                        for i in 1..=fm.npt_in_obj[(iobj - 1) as usize] {
                            if num_con_size == 0 || array[(i - 1) as usize] < 0. {
                                sizes[(ind_size[(iobj - 1) as usize] + i - 2) as usize] =
                                    better_radius[(itype - 1) as usize];
                            }
                        }
                    }
                }
            }
            for i in ind_free..=ind_free + num_sizes - 1 {
                size_max = f_max(size_max, sizes[(i - 1) as usize]);
            }
            ind_free += num_sizes;
            if ind_cont > ind_free {
                exit_error("Mismatch between loaded point sizes and points in contours");
            }
        }
    }
    if v.if_verbose != 0 && size_max > 0. {
        let _ = writeln!(out, " Maximum point size{}", ld_real(size_max));
    }
    if 3.1416_f32 * size_max.powi(2) + 5. > LIMPATCH as f32 {
        exit_error("The largest circle radius is too big for the arrays");
    }

    let mut grow_array: Vec<f32> = Vec::new();
    if num_expand_iter > 0 && expand_target_rad > 0. && size_max > expand_target_rad {
        //
        // FT of a Gaussian with sigma is a Gaussian with sigma' = 1 / (pi * sigma)
        // Reduction by r puts bNyquist at r / 2
        // Make this equivalent to 1.6 * sigma' => sigma' = r / 3.2
        // sigma = 1 / (pi * sigma') = 3.2 / (pi * r) ~ 1 / r
        // r = expandTargetPix / curPixel
        smooth_sigma = size_max / expand_target_rad;
        grow_array = vec![0.0_f32; im_size as usize];
        memory_error(0, "array for smoothed image");
        num_expand_iter = (num_expand_iter as f32 * smooth_sigma) as i32;
        v.noise_extra_bord = (v.noise_extra_bord as f32 * smooth_sigma).ceil() as i32;
        scaled_gaussian_kernel(&mut smooth_kernel, &mut kernel_dim, LIMKERNEL, smooth_sigma);
        if v.if_verbose != 0 {
            let _ = writeln!(
                out,
                " smoothSigma {}   kernel dim {}",
                ld_real(smooth_sigma),
                ld_int(kernel_dim)
            );
        }
    }
    //
    let mut itype = 2;
    if v.fill_with_noise {
        itype = 2 * (v.max_dev + v.noise_extra_bord);
    }
    v.vic_dim = itype;
    v.vicinity = vec![0.0; (itype * itype) as usize];
    v.vicinity_sqr = vec![0.0; (itype * itype) as usize];
    v.sd_array = vec![0.0; (itype * itype) as usize];
    memory_error(0, "arrays for local noise measurement");
    let dev_dim = (2 * v.max_dev + 1) as usize;
    v.in_list = vec![false; dev_dim * dev_dim];
    v.adjacent = vec![0; dev_dim * dev_dim];
    memory_error(0, "arrays for adjacent points");
    v.ca_adj_value = vec![0.0; LIMPATCH as usize];
    v.ca_rmat = vec![0.0; (v.mat_size * LIMPATCH) as usize];
    //
    // start looping on sections; need to read regardless
    //
    for iz_sect in 0..=nz - 1 {
        let _ = write!(out, "Section{} -", i_edit(iz_sect, 5));
        // SAFETY: unit 1 is open for reading; `array` holds at least
        // `nx * ny` floats.
        unsafe {
            iiu_set_position(1, iz_sect, 0);
            if irdsec(1, &mut array).is_err() {
                exit_error("Reading file");
            }
        }

        // Smooth: Kernel dim 15 is 75 msec on 4K images
        if smooth_sigma > 0. {
            apply_kernel_filter(
                &array,
                &mut grow_array,
                nx,
                nx,
                ny,
                &smooth_kernel,
                kernel_dim,
            );
        }
        //
        // Get a quick mean for improving computation of SD
        rough_mean = dmean;
        if nx > 10 && ny > 10 {
            rough_mean = 0.;
            for iy1 in 1..=8 {
                let iy2 = ny * iy1 / 10 - 1;
                for ix1 in 1..=8 {
                    rough_mean += array[((nx * ix1) / 10 + nx * iy2 - 1) as usize] / 64.;
                }
            }
        }

        let mut num_fix = 0_i32;
        let mut line_fix = 0_i32;
        num_tapering = 0;
        //
        // scan through points to see if any need fixing
        // Go backwards in case this model is a peak model
        //
        let mut iobj = num_obj_orig;
        while iobj >= 1 {
            let this_obj = iobj;
            iobj -= 1;
            let iobj = this_obj;
            if fm.npt_in_obj[(iobj - 1) as usize] > 0 {
                let ibase = fm.ibase_obj[(iobj - 1) as usize];
                let itype = 256 - fm.obj_color[(iobj - 1) as usize][1];
                circle_cont = type_on_list(itype, &iobj_circle, num_obj_circle);
                all_sec_cont = type_on_list(itype, &iobj_do_all, num_obj_do_all);
                cont_on_sec_or_all_sec = fm.p_coord[(fm.object[ibase as usize] - 1) as usize][2]
                    .round() as i32
                    == iz_sect
                    || all_sec_cont;

                // If skipping low values, test if point below threshold
                if skip_low_values
                    && cont_is_below_thresh(
                        iobj,
                        &fm.obj_color,
                        &mut last_obj_with_thresh,
                        &mut cur_obj_thresh,
                        &mut low_turned_off,
                        &mut cont_value,
                    )
                {
                    continue;
                }
                //
                // first see if this has a line to do
                //
                if type_on_list(itype, &iobjline, num_obj_line) {
                    if cont_on_sec_or_all_sec {
                        if line_fix == 0 {
                            let _ = write!(out, " fixing lines -");
                        }
                        line_fix += 1;
                        let ip1 = fm.object[ibase as usize];
                        let ip2 = fm.object[(ibase + 1) as usize];
                        let p1 = fm.p_coord[(ip1 - 1) as usize];
                        let p2 = fm.p_coord[(ip2 - 1) as usize];
                        let ix1 = (p1[0] + 0.5).round() as i32;
                        let ix2 = (p2[0] + 0.5).round() as i32;
                        let iy1 = (p1[1] + 0.5).round() as i32;
                        let iy2 = (p2[1] + 0.5).round() as i32;
                        //
                        // Check against other lines to determine borders.  First look
                        // for lines below, then for lines above
                        //
                        j_bord_low = 0;
                        let mut idir = -1;
                        while idir <= 1 {
                            j_bord_low = i_border;
                            i_border = 0;
                            nearby = true;
                            //
                            // Look for adjacent lines then ones next farther out, until
                            // find no lines with overlap in the other coordinate
                            //
                            while i_border <= 5 && nearby {
                                nearby = false;
                                i_border += 1;
                                let mut jobj = num_obj_orig;
                                while jobj >= 1 {
                                    if fm.npt_in_obj[(jobj - 1) as usize] > 0 {
                                        let jbase = fm.ibase_obj[(jobj - 1) as usize];
                                        let jtype = 256 - fm.obj_color[(jobj - 1) as usize][1];
                                        if type_on_list(jtype, &iobjline, num_obj_line)
                                            && (fm.p_coord[(fm.object[jbase as usize] - 1) as usize]
                                                [2]
                                            .round()
                                                as i32
                                                == iz_sect
                                                || type_on_list(
                                                    jtype,
                                                    &iobj_do_all,
                                                    num_obj_do_all,
                                                ))
                                        {
                                            let jp1 = fm.object[jbase as usize];
                                            let jp2 = fm.object[(jbase + 1) as usize];
                                            let q1 = fm.p_coord[(jp1 - 1) as usize];
                                            let q2 = fm.p_coord[(jp2 - 1) as usize];
                                            let jx1 = (q1[0] + 0.5).round() as i32;
                                            let jx2 = (q2[0] + 0.5).round() as i32;
                                            let jy1 = (q1[1] + 0.5).round() as i32;
                                            let jy2 = (q2[1] + 0.5).round() as i32;
                                            //
                                            // Check if line is on the line in question and
                                            // check endpoints for overlap
                                            //
                                            if (iy1 == iy2
                                                && jy1 == jy2
                                                && jy1 == iy1 + idir * i_border
                                                && !(jx1.max(jx2) < ix1.min(ix2)
                                                    || jx1.min(jx2) > ix1.max(ix2)))
                                                || (ix1 == ix2
                                                    && jx1 == jx2
                                                    && jx1 == ix1 + idir * i_border
                                                    && !(jy1.max(jy2) < iy1.min(iy2)
                                                        || jy1.min(jy2) > iy1.max(iy2)))
                                            {
                                                nearby = true;
                                            }
                                        }
                                    }
                                    jobj -= 1;
                                }
                            }
                            idir += 2;
                        }
                        j_bord_high = i_border;
                        // print *,ix1, iy1, ix2, iy2, iBordLo, iBordHi
                        clean_line(
                            &mut array,
                            nx,
                            ny,
                            ix1,
                            iy1,
                            ix2,
                            iy2,
                            j_bord_low,
                            j_bord_high,
                        );
                    }
                    //
                    // Next taper-inside contour
                } else if cont_on_sec_or_all_sec && obj_taper[(iobj - 1) as usize] {
                    if num_tapering == 0 {
                        let _ = write!(out, " tapering large patches -");
                    }
                    num_tapering += 1;
                    fill_boundary_arrays(
                        iobj,
                        &mut xbound,
                        &mut ybound,
                        &mut xmin,
                        &mut xmax,
                        &mut y_min,
                        &mut ymax,
                        &fm,
                    );
                    taper_inside_cont(
                        &mut array,
                        nx,
                        ny,
                        &xbound,
                        &ybound,
                        fm.npt_in_obj[(iobj - 1) as usize],
                        xmin,
                        xmax,
                        y_min,
                        ymax,
                        5 * (xmax + ymax + 2. - xmin - y_min).round() as i32,
                        &mut ierr,
                    );
                    if ierr != 0 {
                        objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);
                        let _ = writeln!(
                            out,
                            "ERROR: CCDERASER - object{}, contour{} does not have enough adjacent points",
                            i_edit(imod_obj, 4),
                            i_edit(imod_cont, 6)
                        );
                        exit(1);
                    }
                    //
                    // Then check circle or other contour at Z
                } else if cont_on_sec_or_all_sec || circle_cont {
                    //
                    // Set up to loop once or on circle points
                    let loop_start = 1;
                    let mut loop_end = 1;
                    num_grow_iter = 0;
                    if circle_cont {
                        loop_end = fm.npt_in_obj[(iobj - 1) as usize];
                    }
                    for lp in loop_start..=loop_end {
                        ix_fix_min = 100000000;
                        ix_fix_max = -100000000;
                        iy_fix_min = 100000000;
                        iy_fix_max = -100000000;
                        let mut num_in_obj: i32;
                        if circle_cont {
                            num_grow_iter = num_expand_iter;
                            //
                            // add circle point to patch if on this Z
                            let ipt = fm.object[(ibase + lp - 1) as usize];
                            num_in_obj = 0;
                            size = sizes[(ind_size[(iobj - 1) as usize] + lp - 2) as usize];
                            if fm.p_coord[(ipt - 1) as usize][2].round() as i32 == iz_sect
                                && size > 0.
                            {
                                add_circle_to_patch(
                                    iobj,
                                    lp,
                                    size,
                                    &mut num_in_obj,
                                    &mut ix_fix_min,
                                    &mut ix_fix_max,
                                    &mut iy_fix_min,
                                    &mut iy_fix_max,
                                    &mut v,
                                    &fm,
                                );
                            }
                            diam_merge = v.max_dev as f32;
                        } else {
                            //
                            // Or add contour points to patch
                            num_in_obj = fm.npt_in_obj[(iobj - 1) as usize].min(LIMPATCH);
                            for ip in 1..=num_in_obj {
                                let ipt = fm.object[(ibase + ip - 1) as usize];
                                let p = fm.p_coord[(ipt - 1) as usize];
                                let k = (ip - 1) as usize;
                                v.ix_fix[k] = (p[0] + 1.01) as i32;
                                v.iy_fix[k] = (p[1] + 1.01) as i32;
                                ix_fix_min = ix_fix_min.min(v.ix_fix[k]);
                                ix_fix_max = ix_fix_max.max(v.ix_fix[k]);
                                iy_fix_min = iy_fix_min.min(v.iy_fix[k]);
                                iy_fix_max = iy_fix_max.max(v.iy_fix[k]);
                            }
                            diam_merge = 2. * v.radius_max;
                        }
                        //
                        if num_fix == 0 {
                            let _ = write!(out, " fixing points -");
                        }
                        num_fix += 1;
                        //
                        // see if there are patches to merge - loop on objects until
                        // patch stops growing
                        //
                        if !all_sec_cont && num_in_obj > 0 && if_merge != 0 {
                            //
                            // The radius criterion for being too large is based on the
                            // maximum radius entry unless there is a circle in the merge,
                            // then it is based on the maxDev parameter
                            rad_sq = diam_merge.powi(2);
                            if_grew = 1;
                            while if_grew > 0 {
                                if_grew = 0;
                                for i in 1..=num_obj_orig {
                                    if skip_low_values
                                        && cont_is_below_thresh(
                                            i,
                                            &fm.obj_color,
                                            &mut last_obj_with_thresh,
                                            &mut cur_obj_thresh,
                                            &mut low_turned_off,
                                            &mut cont_value,
                                        )
                                    {
                                        continue;
                                    }
                                    let jbase = fm.ibase_obj[(i - 1) as usize];
                                    let jtype = 256 - fm.obj_color[(i - 1) as usize][1];
                                    if type_on_list(jtype, &iobj_circle, num_obj_circle) {
                                        //
                                        // For a circle contour, loop on points, first check Z,
                                        // space in patch, and against the min/max of patch
                                        for ip1 in 1..=fm.npt_in_obj[(i - 1) as usize] {
                                            let ipt = fm.object[(jbase + ip1 - 1) as usize];
                                            size = sizes
                                                [(ind_size[(i - 1) as usize] + ip1 - 2) as usize];
                                            if fm.p_coord[(ipt - 1) as usize][2].round() as i32
                                                == iz_sect
                                                && size > 0.
                                                && (i != iobj || lp != ip1)
                                                && num_in_obj as f32
                                                    + 3.1416_f32 * size.powi(2)
                                                    + 5.
                                                    <= LIMPATCH as f32
                                            {
                                                xcen = fm.p_coord[(ipt - 1) as usize][0];
                                                ycen = fm.p_coord[(ipt - 1) as usize][1];
                                                if xcen + size + 1. >= ix_fix_min as f32
                                                    && xcen - size - 1. <= ix_fix_max as f32
                                                    && ycen + size + 1. >= iy_fix_min as f32
                                                    && ycen - size - 1. <= iy_fix_max as f32
                                                {
                                                    //
                                                    // Then check each pixel for ones close enough to
                                                    // touch and ones far enough to make patch too big
                                                    if_touch = 0;
                                                    let mut ip2 = 1;
                                                    while ip2 <= num_in_obj && if_touch >= 0 {
                                                        let k = (ip2 - 1) as usize;
                                                        dist = ((xcen - v.ix_fix[k] as f32)
                                                            .powi(2)
                                                            + (ycen - v.iy_fix[k] as f32).powi(2))
                                                        .sqrt();
                                                        if dist - size < 1.5 {
                                                            if_touch = 1;
                                                        }
                                                        if dist + size > v.max_dev as f32 {
                                                            if_touch = -1;
                                                        }
                                                        ip2 += 1;
                                                    }
                                                    //
                                                    // Merge if touch and size not too big
                                                    if if_touch > 0 {
                                                        if v.if_verbose != 0 {
                                                            let (mut ix1, mut iy1) = (0, 0);
                                                            let (mut ix2, mut iy2) = (0, 0);
                                                            objtocont(
                                                                iobj,
                                                                &fm.obj_color,
                                                                &mut ix1,
                                                                &mut iy1,
                                                            );
                                                            objtocont(
                                                                i,
                                                                &fm.obj_color,
                                                                &mut ix2,
                                                                &mut iy2,
                                                            );
                                                            let _ = writeln!(
                                                                out,
                                                                "Merging{}{}{} to{}{}{}",
                                                                i_edit(ix2, 5),
                                                                i_edit(iy2, 5),
                                                                i_edit(ip1, 5),
                                                                i_edit(ix1, 5),
                                                                i_edit(iy1, 5),
                                                                i_edit(lp, 5)
                                                            );
                                                        }
                                                        diam_merge = v.max_dev as f32;
                                                        add_circle_to_patch(
                                                            i,
                                                            ip1,
                                                            size,
                                                            &mut num_in_obj,
                                                            &mut ix_fix_min,
                                                            &mut ix_fix_max,
                                                            &mut iy_fix_min,
                                                            &mut iy_fix_max,
                                                            &mut v,
                                                            &fm,
                                                        );
                                                        sizes[(ind_size[(i - 1) as usize] + ip1 - 2)
                                                            as usize] = 0.;
                                                        if_grew = 1;
                                                    }
                                                }
                                            }
                                        }
                                        //
                                        // Check regular contour
                                    } else if i != iobj
                                        && fm.npt_in_obj[(i - 1) as usize] > 0
                                        && fm.p_coord[(fm.object[jbase as usize] - 1) as usize][2]
                                            .round()
                                            as i32
                                            == iz_sect
                                        && !type_on_list(jtype, &iobjline, num_obj_line)
                                        && !type_on_list(jtype, &iobj_do_all, num_obj_do_all)
                                        && num_in_obj + fm.npt_in_obj[(i - 1) as usize] <= LIMPATCH
                                    {
                                        //
                                        // Loop on all pairs of points and make sure none are
                                        // too far and see if one touches
                                        //
                                        if_touch = 0;
                                        let mut ip1 = 1;
                                        while ip1 <= fm.npt_in_obj[(i - 1) as usize]
                                            && if_touch >= 0
                                        {
                                            let ipt = fm.object[(jbase + ip1 - 1) as usize];
                                            let ix1 =
                                                (fm.p_coord[(ipt - 1) as usize][0] + 1.01) as i32;
                                            let iy1 =
                                                (fm.p_coord[(ipt - 1) as usize][1] + 1.01) as i32;
                                            let mut ip2 = 1;
                                            while ip2 <= num_in_obj && if_touch >= 0 {
                                                let k = (ip2 - 1) as usize;
                                                let ix2 = (ix1 - v.ix_fix[k]).pow(2)
                                                    + (iy1 - v.iy_fix[k]).pow(2);
                                                if ix2 <= 2 {
                                                    if_touch = 1;
                                                }
                                                if ix2 as f32 > rad_sq {
                                                    if_touch = -1;
                                                }
                                                ip2 += 1;
                                            }
                                            ip1 += 1;
                                        }
                                        //
                                        // Merge the patch, set # of points to 0, and set # of
                                        // points 0 for starting patch to avoid duplicate hits
                                        //
                                        if if_touch > 0 {
                                            if v.if_verbose != 0 {
                                                let (mut ix1, mut iy1) = (0, 0);
                                                let (mut ix2, mut iy2) = (0, 0);
                                                objtocont(iobj, &fm.obj_color, &mut ix1, &mut iy1);
                                                objtocont(i, &fm.obj_color, &mut ix2, &mut iy2);
                                                let _ = writeln!(
                                                    out,
                                                    "Merging{}{} to{}{}{}",
                                                    i_edit(ix2, 5),
                                                    i_edit(iy2, 5),
                                                    i_edit(ix1, 5),
                                                    i_edit(iy1, 5),
                                                    i_edit(lp, 5)
                                                );
                                            }
                                            for ip in 1..=fm.npt_in_obj[(i - 1) as usize] {
                                                let ipt = fm.object[(jbase + ip - 1) as usize];
                                                num_in_obj += 1;
                                                let n = (num_in_obj - 1) as usize;
                                                v.ix_fix[n] = (fm.p_coord[(ipt - 1) as usize][0]
                                                    + 1.01)
                                                    as i32;
                                                v.iy_fix[n] = (fm.p_coord[(ipt - 1) as usize][1]
                                                    + 1.01)
                                                    as i32;
                                                // Fixed in translation (BUGS.md): the source
                                                // (`ccderaser.f90:867-870`) takes the min/max
                                                // over `ixFix(ip)`, the first points of the
                                                // patch, so the box never grows to include the
                                                // merged contour; the points just appended,
                                                // `ixFix(numInObj)`, are what was meant.
                                                let k = n;
                                                ix_fix_min = ix_fix_min.min(v.ix_fix[k]);
                                                ix_fix_max = ix_fix_max.max(v.ix_fix[k]);
                                                iy_fix_min = iy_fix_min.min(v.iy_fix[k]);
                                                iy_fix_max = iy_fix_max.max(v.iy_fix[k]);
                                            }
                                            fm.npt_in_obj[(i - 1) as usize] = 0;
                                            if_grew = 1;
                                        }
                                    }
                                }
                            }
                            //
                            // Zero out this contour or point
                            if circle_cont {
                                sizes[(ind_size[(iobj - 1) as usize] + lp - 2) as usize] = 0.;
                            } else {
                                fm.npt_in_obj[(iobj - 1) as usize] = 0;
                            }
                        }
                        //
                        if num_in_obj > 0 {
                            if smooth_sigma > 0. {
                                clean_area(
                                    &mut array,
                                    Some(&grow_array),
                                    &mut num_in_obj,
                                    num_grow_iter,
                                    &mut v,
                                );
                            } else {
                                clean_area(
                                    &mut array,
                                    None,
                                    &mut num_in_obj,
                                    num_grow_iter,
                                    &mut v,
                                );
                            }
                        }
                    }
                }
            }
        }
        //
        // do peak search for automatic removal
        //
        num_pixels = 0;
        num_patch = 0;
        num_patch_out = 0;
        num_pt_out = 0;
        if if_peak_search > 0 {
            for iobj in 1..=num_search_iterations {
                num_patch_tmp = 0;
                search_peaks(
                    &mut array,
                    &mut diff_arr,
                    iz_sect,
                    &mut num_patch_tmp,
                    &mut num_pixels_tmp,
                    &mut num_pt_out,
                    &mut num_patch_out,
                    iobj,
                    rough_mean,
                    &mut v,
                );
                num_patch += num_patch_tmp;
                num_pixels += num_pixels_tmp;
                if num_patch_tmp == 0 {
                    if v.if_verbose > 0 {
                        let _ = writeln!(out, " Done after:{}", ld_int(iobj - 1));
                    }
                    break;
                }
            }
        }
        if num_patch > 0 {
            let _ = write!(
                out,
                "{} pixels replaced in{} peaks -",
                i_edit(num_pixels, 7),
                i_edit(num_patch, 6)
            );
        }
        //
        // Save points from this section in model if requested
        //
        if !blank(&model_out) {
            for iobj in 1..=num_patch_out {
                num_pixels = v.ind_patch[iobj as usize] - v.ind_patch[(iobj - 1) as usize];
                if fm.n_point + num_pixels <= fm.max_pt && fm.max_mod_obj < fm.max_obj_num {
                    //
                    // If there is room for both another object and all the points,
                    // then set up the object and copy the points
                    //
                    fm.max_mod_obj += 1;
                    let m = (fm.max_mod_obj - 1) as usize;
                    fm.obj_color[m][1] = 256
                        - 1.max(
                            max_objects_out.min((1. + v.exceed_crit[(iobj - 1) as usize]) as i32),
                        );
                    fm.obj_color[m][0] = 1;
                    fm.npt_in_obj[m] = num_pixels;
                    fm.ibase_obj[m] = fm.ibase_free;
                    fm.ibase_free += num_pixels;
                    for i in v.ind_patch[(iobj - 1) as usize]..=v.ind_patch[iobj as usize] - 1 {
                        fm.n_point += 1;
                        let n = (fm.n_point - 1) as usize;
                        let k = (i - 1) as usize;
                        fm.p_coord[n][0] = v.ix_out[k] as f32 - 0.5;
                        fm.p_coord[n][1] = v.iy_out[k] as f32 - 0.5;
                        fm.p_coord[n][2] = v.iz_out[k] as f32;
                        fm.object[n] = fm.n_point;
                    }
                }
            }
        }
        //
        array_min_max_mean_fortran(
            &array,
            &nx,
            &ny,
            &1,
            &nx,
            &1,
            &ny,
            &mut dmin_tmp,
            &mut dmax_tmp,
            &mut dmeant,
        );
        tmin = f_min(tmin, dmin_tmp);
        tmax = f_max(tmax, dmax_tmp);
        tsum += dmeant;
        //
        // write out if any changes or if new output file
        //
        if (num_fix > 0 || line_fix > 0 || num_patch > 0 || im_file_out == 2) && if_trial_mode == 0
        {
            // SAFETY: the output unit is open and `array` holds a section.
            unsafe {
                iiu_set_position(im_file_out, iz_sect, 0);
                iiu_write_section(im_file_out, array.as_mut_ptr().cast());
            }
        }
        let _ = writeln!(out, " Done");
        let _ = out.flush();
    }
    //
    let origin = iiu_ret_origin(1);
    let cur_tilt = iiu_ret_tilt(1);
    let tmean = tsum / nz as f32;
    if mode == 1 {
        // `ccderaser.f90:970-977`, operand order from the reference object:
        // `maxss tmin, -32768.` but `maxss 0., tmin`; `minss tmax, limit`.
        tmin = f_max(tmin, -32768.);
        tmax = f_min(tmax, 32767.);
    } else if mode == 6 {
        tmin = f_max(0., tmin);
        tmax = f_min(tmax, 65535.);
    } else if mode == 0 {
        tmin = f_max(0., tmin);
        tmax = f_min(tmax, 255.);
    }
    // write(titlech, 109) dat, tim
    // 109 format('CCDERASER: Bad points replaced with interpolated values', t57, a9, 2x, a8)
    let mut title = [b' '; 80];
    let text = b"CCDERASER: Bad points replaced with interpolated values";
    title[..text.len()].copy_from_slice(text);
    title[56..65].copy_from_slice(&dat);
    title[67..75].copy_from_slice(&tim);
    if if_trial_mode == 0 {
        iiu_write_header(im_file_out, &title, 1, tmin, tmax, tmean);
        // SAFETY: the unit was opened above.
        unsafe { iiu_close(im_file_out) };
    } else {
        let _ = writeln!(
            out,
            " New minimum and maximum density would be:{}{}",
            ld_real(tmin),
            ld_real(tmax)
        );
    }

    //
    // put out a model, even if there are no points.  Slide objects down
    // over original input model if any
    //
    if !blank(&model_out) {
        if num_obj_orig > 0 {
            for iobj in num_obj_orig + 1..=fm.max_mod_obj {
                let i = (iobj - num_obj_orig - 1) as usize;
                let j = (iobj - 1) as usize;
                fm.obj_color[i][0] = fm.obj_color[j][0];
                fm.obj_color[i][1] = fm.obj_color[j][1];
                fm.npt_in_obj[i] = fm.npt_in_obj[j];
                fm.ibase_obj[i] = fm.ibase_obj[j];
            }
            fm.max_mod_obj -= num_obj_orig;
        }
        //
        // Convert to montage coordinates
        // Have to shift montage coordinates to 0 because it is a new model
        if num_pc_list > 0 {
            for iobj in 1..=fm.max_mod_obj {
                let ibase = fm.ibase_obj[(iobj - 1) as usize];
                for ip in 1..=fm.npt_in_obj[(iobj - 1) as usize] {
                    let ipt = fm.object[(ibase + ip - 1) as usize];
                    let pc = &mut fm.p_coord[(ipt - 1) as usize];
                    let i = pc[2].round() as i32 + 1;
                    let k = (i - 1) as usize;
                    pc[0] = pc[0] + ix_pc_list[k] as f32 - min_xpiece as f32;
                    pc[1] = pc[1] + iy_pc_list[k] as f32 - min_ypiece as f32;
                    pc[2] = iz_pc_list[k] as f32;
                }
            }
        }
        //
        newimod();
        for i in 1..=max_objects_out {
            putimodflag(i, 2);
            putsymtype(i, 0);
            putsymsize(i, 5);
        }
        putimageref(&delta, &origin, &cur_tilt);
        putimodmaxes(nx, ny, nz);
        scale_model_to_image(1, 1, &mut fm);
        let _ = out.flush();
        write_wmod(model_out.trim_end_matches(' '), &mut fm);
        let _ = std::io::stdout().flush();
        let _ = write!(
            out,
            "In the output model, contours have been sorted into{} objects based on how\n much a peak exceeds the criterion.\n Object 1 has peaks that exceed the criterion by < 1 SD,\n object 2 has peaks that exceed the criterion by 1 - 2 SDs,\n object {} has peaks that exceed the criterion by >{} SDs\n",
            i_edit(max_objects_out, 3),
            i_edit(max_objects_out, 2),
            i_edit(max_objects_out - 1, 2)
        );
    }

    let _ = out.flush();
    exit(0);
}

/// Original contained function `contIsBelowThresh` (`ccderaser.f90:1063`).
///
/// contIsBelowThresh tests whether the contour jjobj has a value below
/// threshold and low values are turned off.  First get the threshold and
/// whether low is turned off for the IMOD object if it is not the same as
/// last one, then test the contour.  The host variables it uses are passed
/// explicitly.
fn cont_is_below_thresh(
    jjobj: i32,
    obj_color: &[[i32; 2]],
    last_obj_with_thresh: &mut i32,
    cur_obj_thresh: &mut f32,
    low_turned_off: &mut bool,
    cont_value: &mut f32,
) -> bool {
    let (mut mod_obj, mut mod_cont) = (0_i32, 0_i32);
    let mut result = false;
    objtocont(jjobj, obj_color, &mut mod_obj, &mut mod_cont);
    if mod_obj != *last_obj_with_thresh {
        if getobjvaluethresh(mod_obj, cur_obj_thresh) == 0 {
            *low_turned_off = getobjskiplowvalues(mod_obj) > 0;
        } else {
            *low_turned_off = false;
        }
        *last_obj_with_thresh = mod_obj;
    }
    if *low_turned_off && getcontvalue(mod_obj, mod_cont, cont_value) == 0 {
        result = *cont_value < *cur_obj_thresh;
    }
    result
}

/// `LIMLIST` of `searchPeaks` (`ccderaser.f90:1091`).
const LIMLIST: i32 = 400;

/// The local variables of `searchPeaks` that its contained procedures reach
/// by host association, with its dummy arguments.
pub struct SearchPeaksLocals<'a> {
    v: &'a mut CcdVars,
    array: &'a mut [f32],
    diff_arr: &'a mut [f32],
    iz_sect: i32,
    num_patch: i32,
    num_pixels: i32,
    num_pt_out: i32,
    num_patch_out: i32,
    big_diffs: Vec<f32>,
    num_pt_save: i32,
    i_scan_x: i32,
    iy_start: i32,
    iy_end: i32,
    ix_start: i32,
    ix_end: i32,
    jx: i32,
    jxs: i32,
    jxn: i32,
    jy: i32,
    jys: i32,
    jyn: i32,
    ix: i32,
    iy: i32,
    psum: f32,
    polarity: f32,
    ix_peak: i32,
    iy_peak: i32,
    num_sum: i32,
    num_in_patch: i32,
    grow_crit: f32,
    sd_diff: f32,
    abs_diff_avg: f32,
    abs_diff_sd: f32,
    pix_diff: f32,
    diff_avg: f32,
    diff_sd: f32,
    diff_crit: f32,
    diff_sd_max: f32,
    store_patch: bool,
    ix_offset: i32,
    iy_offset: i32,
    ix_min: i32,
    ix_max: i32,
    iy_min: i32,
    iy_max: i32,
    sum8: f64,
    sumsq8: f64,
}

/// Original `searchPeaks` (`ccderaser.f90:1086`).
///
/// SEARCHPEAKS finds X rays given all the parameters being passed in and set
/// in module
pub fn search_peaks(
    array: &mut [f32],
    diff_arr: &mut [f32],
    iz_sect: i32,
    num_patch: &mut i32,
    num_pixels: &mut i32,
    num_pt_out: &mut i32,
    num_patch_out: &mut i32,
    iter_num: i32,
    rough_mean: f32,
    v: &mut CcdVars,
) {
    let mut s = SearchPeaksLocals {
        v,
        array,
        diff_arr,
        iz_sect,
        num_patch: 0,
        num_pixels: 0,
        num_pt_out: *num_pt_out,
        num_patch_out: *num_patch_out,
        big_diffs: vec![0.0; LIMPATCH as usize],
        num_pt_save: 0,
        i_scan_x: 0,
        iy_start: 0,
        iy_end: 0,
        ix_start: 0,
        ix_end: 0,
        jx: 0,
        jxs: 0,
        jxn: 0,
        jy: 0,
        jys: 0,
        jyn: 0,
        ix: 0,
        iy: 0,
        psum: 0.,
        polarity: 0.,
        ix_peak: 0,
        iy_peak: 0,
        num_sum: 0,
        num_in_patch: 0,
        grow_crit: 0.,
        sd_diff: 0.,
        abs_diff_avg: 0.,
        abs_diff_sd: 0.,
        pix_diff: 0.,
        diff_avg: 0.,
        diff_sd: 0.,
        diff_crit: 0.,
        diff_sd_max: 0.,
        store_patch: false,
        ix_offset: 0,
        iy_offset: 0,
        ix_min: 0,
        ix_max: 0,
        iy_min: 0,
        iy_max: 0,
        sum8: 0.,
        sumsq8: 0.,
    };
    let mut ix_list = [0_i32; LIMLIST as usize];
    let mut iy_list = [0_i32; LIMLIST as usize];
    let (mut dmin, mut dmax) = (0.0_f32, 0.0_f32);
    let (mut scan_avg, mut scan_sd) = (0.0_f32, 0.0_f32);
    let mut scan_crit: f32;
    let mut sumsq: f32;
    let mut num_at_peak: i32;
    let mut looking_at: i32;
    let mut num_in_list: i32;
    let mut on_list: bool;
    let mut above_crit = false;
    let mut diff_avg_sum: f32;
    let mut num_diff_avg: i32;
    let mut rad_max_sq: f32;
    let mut num_patch_save: i32;
    let (nx, ny) = (s.v.nx, s.v.ny);
    let mut out = ImodFile::Stdout;

    s.num_patch = 0;
    s.num_pixels = 0;
    let nx_use = nx - 2 * s.v.num_edge_pixels;
    let ny_use = ny - 2 * s.v.num_edge_pixels;
    s.diff_sd_max = 0.;
    num_diff_avg = 0;
    diff_avg_sum = 0.;
    //
    // set up extent of scan regions
    //
    let scan_overlap = s.v.scan_overlap;
    let i_scan_size = s.v.i_scan_size;
    // `ccderaser.f90:1108-1109, 1116-1117`: `maxss expr, 1.`.
    let mut num_scan_x = f_max(
        (nx_use as f32 - scan_overlap * i_scan_size as f32)
            / (i_scan_size as f32 * (1. - scan_overlap)),
        1.,
    ) as i32;
    let mut nx_scan =
        (nx_use as f32 / (num_scan_x as f32 - (num_scan_x - 1) as f32 * scan_overlap)) as i32;
    while nx_scan > LIMDIFF - num_scan_x {
        num_scan_x += 1;
        nx_scan =
            (nx_use as f32 / (num_scan_x as f32 - (num_scan_x - 1) as f32 * scan_overlap)) as i32;
    }

    let mut num_scan_y = f_max(
        (ny_use as f32 - scan_overlap * i_scan_size as f32)
            / (i_scan_size as f32 * (1. - scan_overlap)),
        1.,
    ) as i32;
    let mut ny_scan =
        (ny_use as f32 / (num_scan_y as f32 - (num_scan_y - 1) as f32 * scan_overlap)) as i32;
    while ny_scan > LIMDIFF - num_scan_y {
        num_scan_y += 1;
        ny_scan =
            (ny_use as f32 / (num_scan_y as f32 - (num_scan_y - 1) as f32 * scan_overlap)) as i32;
    }

    // Allocate array to keep track of whether an area had changes, and initialize it
    // to true on the first iteration
    if num_scan_x > s.v.nx_alloc_scan || num_scan_y > s.v.ny_alloc_scan {
        s.v.scan_area_changed = vec![false; (num_scan_x * num_scan_y) as usize];
        memory_error(0, "array for track scan area changes");
        s.v.nx_alloc_scan = num_scan_x;
        s.v.ny_alloc_scan = num_scan_y;
    }
    if iter_num == 1 {
        s.v.scan_area_changed.fill(true);
    }
    let nx_alloc = s.v.nx_alloc_scan;
    //
    // loop on scan regions, getting start and end in each dimension
    //
    for i_scan_y in 1..=num_scan_y {
        s.iy_start =
            1 + s.v.num_edge_pixels + (ny_use - ny_scan) * (i_scan_y - 1) / 1.max(num_scan_y - 1);
        s.iy_end = s.iy_start + ny_scan - 1;
        if i_scan_y == num_scan_y {
            s.iy_end = ny - s.v.num_edge_pixels;
        }
        s.i_scan_x = 1;
        while s.i_scan_x <= num_scan_x {
            let i_scan_x = s.i_scan_x;
            let sac = at!(nx_alloc, i_scan_x, i_scan_y);
            if !s.v.scan_area_changed[sac] {
                s.i_scan_x += 1;
                continue;
            }
            s.v.scan_area_changed[sac] = false;
            s.ix_start = 1
                + s.v.num_edge_pixels
                + (nx_use - nx_scan) * (i_scan_x - 1) / 1.max(num_scan_x - 1);
            s.ix_end = s.ix_start + nx_scan - 1;
            if i_scan_x == num_scan_x {
                s.ix_end = nx - s.v.num_edge_pixels;
            }
            //
            // get statistics and scan for points outside the reduced criterion
            //
            array_min_max_mean_sd_fortran(
                s.array,
                &nx,
                &ny,
                &s.ix_start,
                &s.ix_end,
                &s.iy_start,
                &s.iy_end,
                &mut dmin,
                &mut dmax,
                &mut s.sum8,
                &mut s.sumsq8,
                &mut scan_avg,
                &mut scan_sd,
            );
            scan_crit = s.v.crit_scan * scan_sd;
            //
            // get mean of difference from neighbors now, for looking at big differences
            //
            s.ix_offset = 1 - s.ix_start;
            s.iy_offset = 1 - s.iy_start;
            compute_diffs(
                s.array,
                nx,
                ny,
                s.diff_arr,
                LIMDIFF,
                LIMDIFF,
                s.ix_start + 1,
                s.ix_end - 1,
                s.iy_start + 1,
                s.iy_end - 1,
                s.ix_offset,
                s.iy_offset,
                &mut s.diff_avg,
                &mut s.diff_sd,
                &mut s.abs_diff_avg,
                &mut s.abs_diff_sd,
                0,
                s.v,
            );
            //
            // Now loop on points in patch looking for peaks above criterion
            s.iy = s.iy_start;
            while s.iy <= s.iy_end {
                s.ix = s.ix_start;
                while s.ix <= s.ix_end {
                    let (ix, iy) = (s.ix, s.iy);
                    if (s.array[at!(nx, ix, iy)] - scan_avg).abs() > scan_crit {
                        //
                        // found a point above criterion, need to walk to peak
                        // Start a list of peak points to check
                        s.polarity = 1.0_f32.copysign(s.array[at!(nx, ix, iy)] - scan_avg);
                        s.ix_peak = ix;
                        s.iy_peak = iy;
                        num_at_peak = 1;
                        ix_list[0] = ix;
                        iy_list[0] = iy;
                        looking_at = 1;
                        //
                        // Check points on list; loop until all checked
                        while looking_at <= num_at_peak {
                            let la = (looking_at - 1) as usize;
                            s.jxs = 1.max(ix_list[la] - 1);
                            s.jxn = nx.min(ix_list[la] + 1);
                            s.jys = 1.max(iy_list[la] - 1);
                            s.jyn = ny.min(iy_list[la] + 1);
                            s.jy = s.jys;
                            while s.jy <= s.jyn {
                                s.jx = s.jxs;
                                while s.jx <= s.jxn {
                                    let (jx, jy) = (s.jx, s.jy);
                                    let peak = s.array[at!(nx, s.ix_peak, s.iy_peak)];
                                    if s.polarity * (s.array[at!(nx, jx, jy)] - peak) >= 0. {
                                        //
                                        // If the point is equal, add it to the list
                                        if s.array[at!(nx, jx, jy)] == peak {
                                            on_list = false;
                                            for ip in 1..=num_at_peak {
                                                let k = (ip - 1) as usize;
                                                if jx == ix_list[k] && jy == iy_list[k] {
                                                    on_list = true;
                                                }
                                            }
                                            if !on_list && num_at_peak < LIMLIST {
                                                num_at_peak += 1;
                                                ix_list[(num_at_peak - 1) as usize] = jx;
                                                iy_list[(num_at_peak - 1) as usize] = jy;
                                            }
                                        } else {
                                            //
                                            // If the point is higher, move to it, reset the list, keep looking
                                            // around current center in case an even higher one occurs
                                            s.ix_peak = jx;
                                            s.iy_peak = jy;
                                            num_at_peak = 1;
                                            ix_list[0] = jx;
                                            iy_list[0] = jy;
                                            looking_at = 0;
                                        }
                                    }
                                    s.jx += 1;
                                }
                                s.jy += 1;
                            }
                            looking_at += 1;
                        }
                        if num_at_peak > 1 {
                            let n = num_at_peak as usize;
                            s.ix_peak = (ix_list[..n].iter().sum::<i32>() as f32
                                / num_at_peak as f32)
                                .round() as i32;
                            s.iy_peak = (iy_list[..n].iter().sum::<i32>() as f32
                                / num_at_peak as f32)
                                .round() as i32;
                        }
                        let (radius_max, outer_radius, crit_main) =
                            (s.v.radius_max, s.v.outer_radius, s.v.crit_main);
                        check_and_erase_peak(
                            &mut s,
                            radius_max,
                            outer_radius,
                            crit_main,
                            false,
                            &mut above_crit,
                            "P",
                            rough_mean,
                        );
                        if above_crit {
                            s.v.scan_area_changed[sac] = true;
                        }
                        if s.v.crit_giant > 0.
                            && !above_crit
                            && (s.array[at!(nx, ix, iy)] - scan_avg).abs()
                                > (s.v.crit_scan + 1.) * scan_sd
                        {
                            let (giant_radius, crit_giant) = (s.v.giant_radius, s.v.crit_giant);
                            let outer = giant_radius + s.v.outer_radius - s.v.radius_max;
                            check_and_erase_peak(
                                &mut s,
                                giant_radius,
                                outer,
                                crit_giant,
                                true,
                                &mut above_crit,
                                "Giant p",
                                rough_mean,
                            );
                            if above_crit {
                                s.v.scan_area_changed[sac] = true;
                            }
                        }
                    }
                    s.ix += 1;
                }
                s.iy += 1;
            }
            //
            // Next search for single-pixel difference anomalies
            s.diff_crit = s.v.crit_diff * s.diff_sd;
            s.grow_crit = s.v.crit_grow * s.diff_sd;
            rad_max_sq = s.v.radius_max.powi(2);
            s.iy = s.iy_start + 1;
            while s.iy <= s.iy_end - 1 {
                s.ix = s.ix_start + 1;
                while s.ix <= s.ix_end - 1 {
                    let (ix, iy) = (s.ix, s.iy);
                    s.pix_diff = s.diff_arr[at!(LIMDIFF, ix + s.ix_offset, iy + s.iy_offset)];
                    if (s.pix_diff - s.diff_avg).abs() > s.diff_crit {
                        //
                        // Make a list of adjacent points that exceed criterion in
                        // either direction.  Accumulate sum of points on list
                        // and neighboring  points not on list
                        //
                        num_in_list = 1;
                        ix_list[0] = ix;
                        iy_list[0] = iy;
                        looking_at = 1;
                        s.ix_min = ix;
                        s.ix_max = s.ix_min;
                        s.iy_min = iy;
                        s.iy_max = s.iy_min;
                        s.psum = s.array[at!(nx, ix, iy)];
                        while looking_at <= num_in_list {
                            let la = (looking_at - 1) as usize;
                            s.jx = ix_list[la] - 1;
                            while s.jx <= ix_list[la] + 1 {
                                s.jy = iy_list[la] - 1;
                                while s.jy <= iy_list[la] + 1 {
                                    let (jx, jy) = (s.jx, s.jy);
                                    //
                                    // compute min and max distance from existing points
                                    //
                                    let mut min_dist_sq =
                                        (jx - ix_list[0]).pow(2) + (jy - iy_list[0]).pow(2);
                                    let mut max_dist_sq = min_dist_sq;
                                    for i in 1..=num_in_list {
                                        let k = (i - 1) as usize;
                                        let i_dist_sq =
                                            (jx - ix_list[k]).pow(2) + (jy - iy_list[k]).pow(2);
                                        min_dist_sq = min_dist_sq.min(i_dist_sq);
                                        max_dist_sq = max_dist_sq.max(i_dist_sq);
                                    }
                                    //
                                    // if point not on list and is within diameter and is
                                    // interior and exceeds grow criterion, add to list
                                    //
                                    if min_dist_sq > 0
                                        && max_dist_sq as f32 <= 4. * rad_max_sq
                                        && jx > s.ix_start
                                        && jx < s.ix_end
                                        && jy > s.iy_start
                                        && jy < s.iy_end
                                        && (s.diff_arr
                                            [at!(LIMDIFF, jx + s.ix_offset, jy + s.iy_offset)]
                                            - s.diff_avg)
                                            .abs()
                                            > s.grow_crit
                                        && num_in_list < LIMLIST
                                    {
                                        num_in_list += 1;
                                        ix_list[(num_in_list - 1) as usize] = jx;
                                        iy_list[(num_in_list - 1) as usize] = jy;
                                        s.ix_min = s.ix_min.min(jx);
                                        s.ix_max = s.ix_max.max(jx);
                                        s.iy_min = s.iy_min.min(jy);
                                        s.iy_max = s.iy_max.max(jy);
                                        s.psum += s.array[at!(nx, jx, jy)];
                                    }
                                    s.jy += 1;
                                }
                                s.jx += 1;
                            }
                            looking_at += 1;
                        }
                        //
                        // get polarity from mean of points on list versus mean
                        // of neighboring points
                        //
                        sumsq = 0.;
                        s.jx = s.ix_min - 1;
                        while s.jx <= s.ix_max + 1 {
                            s.jy = s.iy_min - 1;
                            while s.jy <= s.iy_max + 1 {
                                sumsq += s.array[at!(nx, s.jx, s.jy)];
                                s.jy += 1;
                            }
                            s.jx += 1;
                        }
                        sumsq = (sumsq - s.psum)
                            / ((s.ix_max + 3 - s.ix_min) * (s.iy_max + 3 - s.iy_min) - num_in_list)
                                as f32;
                        s.polarity = 1.0_f32.copysign(s.psum / num_in_list as f32 - sumsq);
                        //
                        // and order the list by differences
                        //

                        for i in 1..=num_in_list - 1 {
                            for j in i + 1..=num_in_list {
                                let (ki, kj) = ((i - 1) as usize, (j - 1) as usize);
                                if (s.diff_arr[at!(
                                    LIMDIFF,
                                    ix_list[ki] + s.ix_offset,
                                    iy_list[ki] + s.iy_offset
                                )] - s.diff_arr[at!(
                                    LIMDIFF,
                                    ix_list[kj] + s.ix_offset,
                                    iy_list[kj] + s.iy_offset
                                )]) * s.polarity
                                    < 0.
                                {
                                    ix_list.swap(ki, kj);
                                    iy_list.swap(ki, kj);
                                }
                            }
                        }
                        //
                        // Move points to fix list, starting with strongest, and
                        // for each other point, recompute difference measure
                        // excluding the ones already on fix list
                        //
                        s.num_in_patch = 0;
                        s.store_patch =
                            s.num_patch_out < LIMPATCHOUT - 1 && s.num_pt_out < LIMPTOUT;
                        s.num_pt_save = s.num_pt_out;
                        num_patch_save = s.num_patch_out;
                        add_point_to_patch(&mut s, ix_list[0], iy_list[0]);
                        for i in 2..=num_in_list {
                            let k = (i - 1) as usize;
                            s.num_sum = 0;
                            s.psum = 0.;
                            s.jx = ix_list[k] - 1;
                            while s.jx <= ix_list[k] + 1 {
                                s.jy = iy_list[k] - 1;
                                while s.jy <= iy_list[k] + 1 {
                                    let (jx, jy) = (s.jx, s.jy);
                                    on_list = jx == ix_list[k] && jy == iy_list[k];
                                    for j in 1..=s.num_in_patch {
                                        let m = (j - 1) as usize;
                                        if jx == s.v.ix_fix[m] && jy == s.v.iy_fix[m] {
                                            on_list = true;
                                        }
                                    }
                                    if !on_list {
                                        s.psum += s.array[at!(nx, jx, jy)];
                                        s.num_sum += 1;
                                    }
                                    s.jy += 1;
                                }
                                s.jx += 1;
                            }
                            //
                            // add point to list if it passes the grow criterion
                            //
                            if s.num_sum > 0 {
                                s.pix_diff = s.array[at!(nx, ix_list[k], iy_list[k])]
                                    - s.psum / s.num_sum as f32;
                                if s.polarity * (s.pix_diff - s.diff_avg) > s.grow_crit {
                                    add_point_to_patch(&mut s, ix_list[k], iy_list[k]);
                                }
                            }
                        }

                        if s.num_in_patch <= s.v.max_in_diff_patch {
                            s.pix_diff = s.diff_arr[at!(
                                LIMDIFF,
                                s.v.ix_fix[0] + s.ix_offset,
                                s.v.iy_fix[0] + s.iy_offset
                            )];
                            s.sd_diff = (s.pix_diff - s.diff_avg).abs() / s.diff_sd;
                            if s.store_patch {
                                s.v.exceed_crit[s.num_patch_out as usize] =
                                    s.sd_diff - s.v.crit_diff;
                                s.num_patch_out += 1;
                                s.v.ind_patch[s.num_patch_out as usize] = s.num_pt_out + 1;
                            }
                            //
                            // report if verbose, fix the patch
                            //
                            if s.v.if_verbose > 0 {
                                // 104 format(/,'Diff peak at',2i6,' = ',f8.0,', diff =',f8.0,
                                //     ', ',f7.2, ' SDs above mean',$)
                                let _ = write!(
                                    out,
                                    "\nDiff peak at{}{} = {}, diff ={}, {} SDs above mean",
                                    i_edit(s.v.ix_fix[0], 6),
                                    i_edit(s.v.iy_fix[0], 6),
                                    f_edit(s.array[at!(nx, ix, iy)], 8, 0),
                                    f_edit(s.pix_diff, 8, 0),
                                    f_edit(s.sd_diff, 7, 2)
                                );
                            }
                            clean_area_recompute_diffs(&mut s);
                            s.v.scan_area_changed[sac] = true;
                        } else {
                            s.num_pt_out = s.num_pt_save;
                            s.num_patch_out = num_patch_save;
                        }
                    }
                    s.ix += 1;
                }
                s.iy += 1;
            }
            //
            // End of patch.  Accumulate current difference averages/Sd's
            diff_avg_sum += s.diff_avg;
            num_diff_avg += 1;
            // `ccderaser.f90:1389`: `maxss diffSd, diffSdMax`.
            s.diff_sd_max = f_max(s.diff_sd, s.diff_sd_max);
            s.i_scan_x += 1;
        }
    }
    //
    // Now clean out the borders, iterating for an edge if anything was replaced on the
    // previous round.  Skip this if no blocks were even scanned on this round because
    // cleanEdgeForDiffs needs a valid diffAvg and diffSdMax
    if num_diff_avg != 0 {
        s.diff_avg = diff_avg_sum / num_diff_avg as f32;
        s.diff_crit = s.v.crit_diff * s.diff_sd_max;
        let mut num_edge_fix = [1_i32; 4];
        let npb = s.v.num_pix_border;
        s.i_scan_x = 1;
        while s.i_scan_x <= 4 {
            if num_edge_fix[0] > 0 {
                clean_edge_for_diffs(&mut s, 1, npb, 1, ny, &mut num_edge_fix[0]);
            }
            if num_edge_fix[1] > 0 {
                clean_edge_for_diffs(&mut s, nx + 1 - npb, nx, 1, ny, &mut num_edge_fix[1]);
            }
            if num_edge_fix[2] > 0 {
                clean_edge_for_diffs(&mut s, 1, nx, 1, npb, &mut num_edge_fix[2]);
            }
            if num_edge_fix[3] > 0 {
                clean_edge_for_diffs(&mut s, 1, nx, ny + 1 - npb, ny, &mut num_edge_fix[3]);
            }
            s.i_scan_x += 1;
        }
    }
    *num_patch = s.num_patch;
    *num_pixels = s.num_pixels;
    *num_pt_out = s.num_pt_out;
    *num_patch_out = s.num_patch_out;
}

/// Original contained subroutine `checkAndErasePeak` of `searchPeaks`
/// (`ccderaser.f90:1415`).
///
/// checkAndErasePeak finds the mean and SD in the annulus defined by its
/// arguments and tests the currently found peak with the given criterion.
/// It sets aboveCrit true if it erases the peak
fn check_and_erase_peak(
    s: &mut SearchPeaksLocals,
    rad_max: f32,
    outer_rad: f32,
    crit_peak: f32,
    doing_big: bool,
    above_crit: &mut bool,
    big_text: &str,
    rough_mean: f32,
) {
    let nx = s.v.nx;
    let ny = s.v.ny;
    let mut rad_sq: f32;
    let (mut ring_avg, mut ring_sd) = (0.0_f32, 0.0_f32);
    let mut peak_xcen: f32;
    let mut peak_ycen: f32;
    let mut num_plast = 0_i32;
    let mut ixpk_sum: i32;
    let mut iypk_sum: i32;
    let mut ixp_last = 0_i32;
    let mut iyp_last = 0_i32;
    let mut iouter = outer_rad.round() as i32;
    let rad_inner_sq = rad_max.powi(2);
    let outer_sq = outer_rad.powi(2);
    s.num_pt_save = s.num_pt_out;
    //
    // find mean and SD in the annulus
    //
    s.jxs = 1.max(s.ix_peak - iouter);
    s.jxn = nx.min(s.ix_peak + iouter);
    s.jys = 1.max(s.iy_peak - iouter);
    s.jyn = ny.min(s.iy_peak + iouter);
    s.num_sum = 0;
    s.sum8 = 0.;
    s.sumsq8 = 0.;
    s.jy = s.jys;
    while s.jy <= s.jyn {
        s.jx = s.jxs;
        while s.jx <= s.jxn {
            let (jx, jy) = (s.jx, s.jy);
            rad_sq = ((jx - s.ix_peak).pow(2) + (jy - s.iy_peak).pow(2)) as f32;
            if rad_sq > rad_inner_sq && rad_sq <= outer_sq {
                s.num_sum += 1;
                let d = s.array[at!(nx, jx, jy)] - rough_mean;
                s.sum8 += d as f64;
                s.sumsq8 += (d * d) as f64;
            }
            s.jx += 1;
        }
        s.jy += 1;
    }
    sums_to_avg_sd_dbl(s.sum8, s.sumsq8, s.num_sum, 1, &mut ring_avg, &mut ring_sd);
    ring_avg += rough_mean;
    //
    // Make sure the peak passes the main criterion
    //
    s.sd_diff = 0.;
    if ring_sd > 0. {
        s.sd_diff = (s.array[at!(nx, s.ix_peak, s.iy_peak)] - ring_avg).abs() / ring_sd;
    }
    *above_crit = s.sd_diff > crit_peak;
    if *above_crit {
        //
        // look inside radius and make list of points above the grow criterion
        // Repeat the loop for a big patch or one that would qualify for that, so that
        // the center can be moved
        s.grow_crit = ring_sd * s.v.crit_grow;
        iouter = rad_max.round() as i32;
        peak_xcen = s.ix_peak as f32;
        peak_ycen = s.iy_peak as f32;
        let mut num_grow_loop = 1;
        if doing_big || (s.v.crit_giant > 0. && s.sd_diff > s.v.crit_giant) {
            num_grow_loop = 3;
        }
        for loop_grow in 1..=num_grow_loop {
            //
            // initialize for this time through the loop, get limits to look at
            s.num_in_patch = 0;
            ixpk_sum = 0;
            iypk_sum = 0;
            s.num_pt_out = s.num_pt_save;
            s.jxs = 1.max(peak_xcen.round() as i32 - iouter);
            s.jxn = nx.min(peak_xcen.round() as i32 + iouter);
            s.jys = 1.max(peak_ycen.round() as i32 - iouter);
            s.jyn = ny.min(peak_ycen.round() as i32 + iouter);
            s.store_patch = s.num_patch_out < LIMPATCHOUT - 1 && s.num_pt_out < LIMPTOUT;
            s.jy = s.jys;
            while s.jy <= s.jyn {
                s.jx = s.jxs;
                while s.jx <= s.jxn {
                    let (jx, jy) = (s.jx, s.jy);
                    //
                    // Find points within radius limit and above the grow criterion
                    rad_sq = (jx as f32 - peak_xcen).powi(2) + (jy as f32 - peak_ycen).powi(2);
                    if rad_sq < rad_inner_sq
                        && s.polarity * (s.array[at!(nx, jx, jy)] - ring_avg) > s.grow_crit
                        && s.num_in_patch < LIMPATCH
                    {
                        add_point_to_patch(s, jx, jy);
                        ixpk_sum += jx;
                        iypk_sum += jy;
                        //
                        // If doing a big patch, store the point's absolute difference
                        if doing_big {
                            let a = s.array[at!(nx, jx, jy)];
                            let d1 = (a - s.array[at!(nx, 1.max(jx - 1), jy)]).abs();
                            let d2 = (a - s.array[at!(nx, nx.min(jx + 1), jy)]).abs();
                            let d3 = (a - s.array[at!(nx, jx, 1.max(jy - 1))]).abs();
                            let d4 = (a - s.array[at!(nx, jx, ny.min(jy + 1))]).abs();
                            s.big_diffs[(s.num_in_patch - 1) as usize] =
                                // `ccderaser.f90:1495-1499`: reassociated by the
                                // reference object as `maxss(maxss(d4, d3),
                                // maxss(d2, d1))`.
                                (f_max(f_max(d4, d3), f_max(d2, d1)) - s.abs_diff_avg).abs()
                                    / s.abs_diff_sd;
                        }
                    }
                    s.jx += 1;
                }
                s.jy += 1;
            }
            //
            // If looping more than once, check if it has stabilized and revise the center
            if num_grow_loop > 1 {
                if loop_grow > 1
                    && num_plast == s.num_in_patch
                    && ixp_last == ixpk_sum
                    && iyp_last == iypk_sum
                {
                    break;
                }
                peak_xcen = ixpk_sum as f32 / s.num_in_patch as f32;
                peak_ycen = iypk_sum as f32 / s.num_in_patch as f32;
                num_plast = s.num_in_patch;
                ixp_last = ixpk_sum;
                iyp_last = iypk_sum;
            }
        }
        //
        // If doing big peak, make sure there were enough big differences and abort if not
        if doing_big {
            rs_sort_floats(&mut s.big_diffs, s.num_in_patch);
            s.jx = s
                .num_in_patch
                .min(3.max((s.v.frac_big_diff * s.num_in_patch as f32).round() as i32));
            let mut total = 0.0_f32;
            for k in s.num_in_patch + 1 - s.jx..=s.num_in_patch {
                total += s.big_diffs[(k - 1) as usize];
            }
            if total / (s.jx as f32) < s.v.crit_big_diff {
                s.num_pt_out = s.num_pt_save;
                *above_crit = false;
                return;
            }
        }
        if s.store_patch {
            s.v.exceed_crit[s.num_patch_out as usize] = s.sd_diff - s.v.crit_main;
            s.num_patch_out += 1;
            s.v.ind_patch[s.num_patch_out as usize] = s.num_pt_out + 1;
        }
        //
        // fix the patch!
        //
        if s.v.if_verbose > 0 {
            // 103 format(/,a,'eak at',2i6,' = ',f8.0,',',f7.2, ' SDs above mean',f8.0,$)
            let _ = write!(
                ImodFile::Stdout,
                "\n{}eak at{}{} = {},{} SDs above mean{}",
                big_text,
                i_edit(s.ix_peak, 6),
                i_edit(s.iy_peak, 6),
                f_edit(s.array[at!(nx, s.ix_peak, s.iy_peak)], 8, 0),
                f_edit(s.sd_diff, 7, 2),
                f_edit(ring_avg, 8, 0)
            );
        }
        clean_area_recompute_diffs(s);
    }
}

/// Original contained subroutine `addPointToPatch` of `searchPeaks`
/// (`ccderaser.f90:1541`).
///
/// addPointToPatch adds one point to the patch, maintain min and max of
/// patch, and save in output array if there is space
fn add_point_to_patch(s: &mut SearchPeaksLocals, ix_add: i32, iy_add: i32) {
    s.num_in_patch += 1;
    s.v.ix_fix[(s.num_in_patch - 1) as usize] = ix_add;
    s.v.iy_fix[(s.num_in_patch - 1) as usize] = iy_add;
    if s.num_in_patch == 1 {
        s.ix_min = ix_add;
        s.ix_max = ix_add;
        s.iy_min = iy_add;
        s.iy_max = iy_add;
    } else {
        s.ix_min = s.ix_min.min(ix_add);
        s.ix_max = s.ix_max.max(ix_add);
        s.iy_min = s.iy_min.min(iy_add);
        s.iy_max = s.iy_max.max(iy_add);
    }
    if s.store_patch && s.num_pt_out < LIMPTOUT {
        s.num_pt_out += 1;
        let k = (s.num_pt_out - 1) as usize;
        s.v.ix_out[k] = ix_add;
        s.v.iy_out[k] = iy_add;
        s.v.iz_out[k] = s.iz_sect;
    }
}

/// Original contained subroutine `cleanAreaRecomputeDiffs` of `searchPeaks`
/// (`ccderaser.f90:1569`).
///
/// cleanAreaRecomputeDiffs calls cleanArea then adjusts the limits of patch
/// and recomputes the differences in this area in a way that maintains the
/// means/sds
fn clean_area_recompute_diffs(s: &mut SearchPeaksLocals) {
    //
    // fix the difference array 1 pixel beyonds limits of patch by first calling
    // to subtract this area, then calling after the replacement to add it back
    let (nx, ny) = (s.v.nx, s.v.ny);
    s.ix_min = (s.ix_min - 1).max(s.ix_start + 1);
    s.ix_max = (s.ix_max + 1).min(s.ix_end - 1);
    s.iy_min = (s.iy_min - 1).max(s.iy_start + 1);
    s.iy_max = (s.iy_max + 1).min(s.iy_end - 1);
    compute_diffs(
        s.array,
        nx,
        ny,
        s.diff_arr,
        LIMDIFF,
        LIMDIFF,
        s.ix_min,
        s.ix_max,
        s.iy_min,
        s.iy_max,
        s.ix_offset,
        s.iy_offset,
        &mut s.diff_avg,
        &mut s.diff_sd,
        &mut s.abs_diff_avg,
        &mut s.abs_diff_sd,
        -1,
        s.v,
    );
    clean_area(s.array, None, &mut s.num_in_patch, 0, s.v);
    compute_diffs(
        s.array,
        nx,
        ny,
        s.diff_arr,
        LIMDIFF,
        LIMDIFF,
        s.ix_min,
        s.ix_max,
        s.iy_min,
        s.iy_max,
        s.ix_offset,
        s.iy_offset,
        &mut s.diff_avg,
        &mut s.diff_sd,
        &mut s.abs_diff_avg,
        &mut s.abs_diff_sd,
        1,
        s.v,
    );
    s.num_patch += 1;
    s.num_pixels += s.num_in_patch;
}

/// Original contained subroutine `cleanEdgeForDiffs` of `searchPeaks`
/// (`ccderaser.f90:1593`).
///
/// cleanEdgeForDiffs does a simple scan for pixel differences above maximum
/// possible criterion in one border region and returns the count of replaced
/// pixels
fn clean_edge_for_diffs(
    s: &mut SearchPeaksLocals,
    ix_left: i32,
    ix_right: i32,
    iy_bot: i32,
    iy_top: i32,
    num_replace: &mut i32,
) {
    let nx = s.v.nx;
    let nx_edge = ix_right + 1 - ix_left;
    let ny_edge = iy_top + 1 - iy_bot;
    // real*4 edgeDiff(ixRight + 1 - ixLeft, iyTop + 1 - iyBot)
    let mut edge_diff = vec![0.0_f32; (nx_edge.max(0) * ny_edge.max(0)) as usize];
    let mut num_high: i32;
    *num_replace = 0;
    //
    // Compute and save differences and check if any are high
    num_high = 0;
    s.iy = iy_bot;
    while s.iy <= iy_top {
        s.ix = ix_left;
        while s.ix <= ix_right {
            neighbor_mean_with_tests(s);
            s.pix_diff = s.array[at!(nx, s.ix, s.iy)] - s.psum;
            edge_diff[at!(nx_edge, s.ix + 1 - ix_left, s.iy + 1 - iy_bot)] = s.pix_diff;
            if (s.pix_diff - s.diff_avg).abs() > s.diff_crit {
                num_high += 1;
            }
            s.ix += 1;
        }
        s.iy += 1;
    }
    //
    // Done if none are high; otherwise scan again and replace a pixel if it is the
    // highest difference of its neighbors
    if num_high == 0 {
        return;
    }
    s.iy = iy_bot;
    while s.iy <= iy_top {
        s.ix = ix_left;
        while s.ix <= ix_right {
            let (ix, iy) = (s.ix, s.iy);
            s.pix_diff = edge_diff[at!(nx_edge, ix + 1 - ix_left, iy + 1 - iy_bot)];
            if (s.pix_diff - s.diff_avg).abs() > s.diff_crit {
                s.jxn = ix + 1 - ix_left;
                s.jyn = iy + 1 - iy_bot;
                s.pix_diff = s.pix_diff.abs();
                let (jxn, jyn) = (s.jxn, s.jyn);
                let ed = |x: i32, y: i32| edge_diff[at!(nx_edge, x, y)].abs();
                if s.pix_diff >= ed(1.max(jxn - 1), jyn)
                    && s.pix_diff >= ed(1.max(jxn - 1), 1.max(jyn - 1))
                    && s.pix_diff >= ed(1.max(jxn - 1), ny_edge.min(jyn + 1))
                    && s.pix_diff >= ed(nx_edge.min(jxn + 1), jyn)
                    && s.pix_diff >= ed(nx_edge.min(jxn + 1), 1.max(jyn - 1))
                    && s.pix_diff >= ed(nx_edge.min(jxn + 1), ny_edge.min(jyn + 1))
                    && s.pix_diff >= ed(jxn, 1.max(jyn - 1))
                    && s.pix_diff >= ed(jxn, ny_edge.min(jyn + 1))
                {
                    neighbor_mean_with_tests(s);
                    //
                    // report if verbose and replace
                    if s.v.if_verbose > 0 {
                        // 134 format(/,'Border peak on scan',i2, ' at',2i6,' = ',f8.0,
                        //     ', diff =',f8.0, ', mean =', f8.0)
                        let _ = writeln!(
                            ImodFile::Stdout,
                            "\nBorder peak on scan{} at{}{} = {}, diff ={}, mean ={}",
                            i_edit(s.i_scan_x, 2),
                            i_edit(ix, 6),
                            i_edit(iy, 6),
                            f_edit(s.array[at!(nx, ix, iy)], 8, 0),
                            f_edit(s.pix_diff, 8, 0),
                            f_edit(s.psum, 8, 0)
                        );
                    }
                    s.array[at!(nx, ix, iy)] = s.psum;
                    *num_replace += 1;
                    s.num_patch += 1;
                    s.num_pixels += 1;
                    if s.num_patch_out < LIMPATCHOUT - 1 && s.num_pt_out < LIMPTOUT - 1 {
                        s.num_patch_out += 1;
                        s.num_pt_out += 1;
                        s.v.ind_patch[s.num_patch_out as usize] = s.num_pt_out + 1;
                        let k = (s.num_pt_out - 1) as usize;
                        s.v.ix_out[k] = ix;
                        s.v.iy_out[k] = iy;
                        s.v.iz_out[k] = s.iz_sect;
                        s.sd_diff = (s.pix_diff - s.diff_avg).abs() / s.diff_sd_max;
                        s.v.exceed_crit[(s.num_patch_out - 1) as usize] = s.sd_diff - s.v.crit_diff;
                    }
                }
            }
            s.ix += 1;
        }
        s.iy += 1;
    }
}

/// Original contained subroutine `neighborMeanWithTests` of `searchPeaks`
/// (`ccderaser.f90:1660`).
///
/// neighborMeanWithTests computes the mean of neighboring pixels for pixels
/// in the border region
fn neighbor_mean_with_tests(s: &mut SearchPeaksLocals) {
    let (nx, ny) = (s.v.nx, s.v.ny);
    s.num_sum = -1;
    s.psum = -s.array[at!(nx, s.ix, s.iy)];
    s.jy = s.iy - 1;
    while s.jy <= s.iy + 1 {
        if s.jy >= 1 && s.jy <= ny {
            s.jx = s.ix - 1;
            while s.jx <= s.ix + 1 {
                if s.jx >= 1 && s.jx <= nx {
                    s.num_sum += 1;
                    s.psum += s.array[at!(nx, s.jx, s.jy)];
                }
                s.jx += 1;
            }
        }
        s.jy += 1;
    }
    s.psum /= s.num_sum as f32;
}

/// Original `computeDiffs` (`ccderaser.f90:1688`).
///
/// COMPUTEDIFFS finds a mean difference between each pixel and its 8
/// neighbors over the range ixStart-ixEnd, iyStart-iyEnd in array, places the
/// result in diffArr using the offsets in ixofs, iyofs and returns the mean
/// and standard deviation of these differences in diffAvg and diffSd.  It
/// also computes the absolute value of differences between all vertically
/// and horizontally adjacent pairs of pixels and returns the mean and SD of
/// that difference in absDiffAvg and absDiffSd.  The `SAVE`d sums are fields
/// of `v`.
pub fn compute_diffs(
    array: &[f32],
    nx: i32,
    _ny: i32,
    diff_arr: &mut [f32],
    ixdim: i32,
    _iy_dim: i32,
    ix_start: i32,
    ix_end: i32,
    iy_start: i32,
    iy_end: i32,
    ix_offset: i32,
    iy_offset: i32,
    diff_avg: &mut f32,
    diff_sd: &mut f32,
    abs_diff_avg: &mut f32,
    abs_diff_sd: &mut f32,
    incremental: i32,
    v: &mut CcdVars,
) {
    let add_fac: f32;
    // Fortran integer `sign(1, incremental)`: +1 for zero.
    let isign = if incremental >= 0 { 1 } else { -1 };
    if incremental == 0 {
        v.cd_sum8 = 0.;
        v.cd_sumsq8 = 0.;
        v.cd_abssum8 = 0.;
        v.cd_abssq8 = 0.;
        v.cd_num_sum = 0;
        add_fac = 1.;
    } else {
        add_fac = isign as f32;
    }
    v.cd_num_sum += isign * (iy_end + 1 - iy_start) * (ix_end + 1 - ix_start);
    for iy in iy_start..=iy_end {
        for ix in ix_start..=ix_end {
            let mut pix_diff = array[at!(nx, ix, iy)]
                - (array[at!(nx, ix - 1, iy)]
                    + array[at!(nx, ix + 1, iy)]
                    + array[at!(nx, ix, iy + 1)]
                    + array[at!(nx, ix, iy - 1)]
                    + array[at!(nx, ix - 1, iy - 1)]
                    + array[at!(nx, ix - 1, iy + 1)]
                    + array[at!(nx, ix + 1, iy - 1)]
                    + array[at!(nx, ix + 1, iy + 1)])
                    / 8.;
            v.cd_sum8 += (add_fac * pix_diff) as f64;
            v.cd_sumsq8 += (add_fac * (pix_diff * pix_diff)) as f64;
            diff_arr[at!(ixdim, ix + ix_offset, iy + iy_offset)] = pix_diff;
            pix_diff = (array[at!(nx, ix, iy)] - array[at!(nx, ix - 1, iy)]).abs();
            v.cd_abssum8 += (add_fac * pix_diff) as f64;
            v.cd_abssq8 += (add_fac * (pix_diff * pix_diff)) as f64;
            pix_diff = (array[at!(nx, ix, iy)] - array[at!(nx, ix, iy - 1)]).abs();
            v.cd_abssum8 += (add_fac * pix_diff) as f64;
            v.cd_abssq8 += (add_fac * (pix_diff * pix_diff)) as f64;
        }
    }
    sums_to_avg_sd_dbl(v.cd_sum8, v.cd_sumsq8, v.cd_num_sum, 1, diff_avg, diff_sd);
    sums_to_avg_sd_dbl(
        v.cd_abssum8,
        v.cd_abssq8,
        v.cd_num_sum * 2,
        1,
        abs_diff_avg,
        abs_diff_sd,
    );
}

/// Original `cleanLine` (`ccderaser.f90:1738`).
///
/// CLEANLINE replaces points along a line with points from adjacent
/// lines
pub fn clean_line(
    array: &mut [f32],
    ixdim: i32,
    _iy_dim: i32,
    ix1: i32,
    iy1: i32,
    ix2: i32,
    iy2: i32,
    j_bord_low: i32,
    j_bord_high: i32,
) {
    if ix1 == ix2 {
        for iy in iy1.min(iy2)..=iy1.max(iy2) {
            array[at!(ixdim, ix1, iy)] = (array[at!(ixdim, ix1 - j_bord_low, iy)]
                + array[at!(ixdim, ix1 + j_bord_high, iy)])
                / 2.;
        }
    } else {
        for ix in ix1.min(ix2)..=ix1.max(ix2) {
            array[at!(ixdim, ix, iy1)] = (array[at!(ixdim, ix, iy1 - j_bord_low)]
                + array[at!(ixdim, ix, iy1 + j_bord_high)])
                / 2.;
        }
    }
}

/// Original `cleanArea` (`ccderaser.f90:1759`).
///
/// CLEANAREA replaces a patch given the list of points in ixfix, iyfix, by
/// finding surrounding points, fitting a polynomial to them, and replacing
/// points in the patch with fitted values.  Can grow the patch if
/// numGrowIter is > 0.  `grow_arr` is `None` where the source passes `array`
/// itself as `growArr`; the grow phase only reads it, before any write.
pub fn clean_area(
    array: &mut [f32],
    grow_arr: Option<&[f32]>,
    num_in_obj: &mut i32,
    num_grow_iter: i32,
    v: &mut CcdVars,
) {
    let (nx, ny) = (v.nx, v.ny);
    let max_dev = v.max_dev;
    let dev_dim = 2 * max_dev + 1;
    // `inList(i, j)` / `adjacent(i, j)` with bounds `-maxDev:maxDev`.
    macro_rules! dv {
        ($i:expr, $j:expr) => {
            ((($i) + max_dev) as isize + ((($j) + max_dev) as isize) * dev_dim as isize) as usize
        };
    }
    let mat_size = v.mat_size;
    let mut xm = vec![0.0_f32; mat_size as usize];
    let mut sd = vec![0.0_f32; mat_size as usize];
    let mut ssd = vec![0.0_f32; (mat_size * mat_size) as usize];
    let mut b1 = vec![0.0_f32; mat_size as usize];
    let mut vector = vec![0.0_f32; mat_size as usize];
    let mut c1 = 0.0_f32;
    let (mut ix_cen, mut iy_cen) = (0_i32, 0_i32);
    let (mut min_xlist, mut min_ylist, mut max_xlist, mut max_ylist): (i32, i32, i32, i32);
    let (mut ixl, mut iyl) = (0_i32, 0_i32);
    let mut num_bord_m1 = 0_i32;
    let (mut ix_bord_low, mut ix_bord_high, mut iy_bord_low, mut iy_bord_high) =
        (0_i32, 0_i32, 0_i32, 0_i32);
    let mut num_vals: i32;
    let mut num_pat_pix: i32;
    let mut num_added: i32;
    let mut num_sd_points = 0_i32;
    let mut icut: i32;
    let mut patch_sum: f32;
    let mut polarity: f32;
    let mut cutoff: f32;
    let mut adj_median = 0.0_f32;
    let mut pctile: f32;
    let mut opp_cutoff = 0.0_f32;
    let mut big_sd = 0.0_f32;
    let mut big_mean = 0.0_f32;
    let mut xsum: f32;
    let mut xsq_sum: f32;
    let mut xmean: f32;
    let mut out = ImodFile::Stdout;

    let percentile = 0.05_f32;
    let halo_pctile = 0.02_f32;
    let min_sds_for_median = 5;
    let mut igrow = 0;
    while igrow <= num_grow_iter {
        //
        // get range of patch and initialize initialize inlist and adjacent arrays
        //
        min_xlist = v.ix_fix[0];
        min_ylist = v.iy_fix[0];
        max_xlist = v.ix_fix[0];
        max_ylist = v.iy_fix[0];
        for k in 2..=*num_in_obj {
            let k = (k - 1) as usize;
            min_xlist = min_xlist.min(v.ix_fix[k]);
            min_ylist = min_ylist.min(v.iy_fix[k]);
            max_xlist = max_xlist.max(v.ix_fix[k]);
            max_ylist = max_ylist.max(v.iy_fix[k]);
        }
        ix_cen = (max_xlist + min_xlist) / 2;
        iy_cen = (max_ylist + min_ylist) / 2;
        //
        v.in_list.fill(false);
        v.adjacent.fill(0);
        //
        // Mark all points as inlist and all adjacent points as 1's
        for k in 1..=*num_in_obj {
            let k = (k - 1) as usize;
            ixl = v.ix_fix[k] - ix_cen;
            iyl = v.iy_fix[k] - iy_cen;
            v.in_list[dv!(ixl, iyl)] = true;
            for i in -1..=1 {
                for j in -1..=1 {
                    v.adjacent[dv!(ixl + i, iyl + j)] = 1;
                }
            }
        }
        //
        // get limits of region including border
        num_bord_m1 = v.num_pix_border - 1;
        ix_bord_low = 1.max(min_xlist - v.num_pix_border);
        ix_bord_high = nx.min(max_xlist + v.num_pix_border);
        iy_bord_low = 1.max(min_ylist - v.num_pix_border);
        iy_bord_high = ny.min(max_ylist + v.num_pix_border);
        //
        // if growing, get the sum of points in patch and an array of adjacent points
        if igrow == num_grow_iter {
            break;
        }
        let grow: &[f32] = match grow_arr {
            Some(g) => g,
            None => &*array,
        };
        num_vals = 0;
        num_pat_pix = 0;
        patch_sum = 0.;
        for iy in iy_bord_low..=iy_bord_high {
            for ix in ix_bord_low..=ix_bord_high {
                let ix_offset = ix - ix_cen;
                let iy_offset = iy - iy_cen;
                if v.in_list[dv!(ix_offset, iy_offset)] {
                    num_pat_pix += 1;
                    patch_sum += grow[at!(nx, ix, iy)];
                } else if v.adjacent[dv!(ix_offset, iy_offset)] > 0 && num_vals < LIMPATCH {
                    num_vals += 1;
                    v.ca_adj_value[(num_vals - 1) as usize] = grow[at!(nx, ix, iy)];
                }
            }
        }
        //
        // Get the median to determine polarity and the percentile cutoff
        rs_sort_floats(&mut v.ca_adj_value, num_vals);
        rs_median_of_sorted(&v.ca_adj_value, num_vals, &mut adj_median);

        // 1/15/25: experimented with trying to switch polarity to capture halos by looking
        // whether points more than 1.3 MADNs from the median were 1.5 x as numerous on the
        // opposite side, and > 10% while ones on proper side were < 10%  This gave no visible
        // reduction in halos and impaired removal of black bead material slightly
        // The current method of cutting off both keeps the iterations alive longer and lets
        // the main polarity expanding include more

        polarity = 1.0_f32.copysign(patch_sum / 1.max(num_pat_pix) as f32 - adj_median);
        pctile = percentile;
        if polarity < 0. {
            pctile = 1. - percentile;
        }
        icut = num_vals.min(1.max((pctile * num_vals as f32 - 0.5) as i32 + 1));
        cutoff = 2. * adj_median - v.ca_adj_value[(icut - 1) as usize];
        // print *,adjMedian, polarity, icut, adjValue(icut), cutoff
        if v.include_halo {
            pctile = halo_pctile;
            if polarity > 0. {
                pctile = 1. - pctile;
            }
            icut = num_vals.min(1.max((pctile * num_vals as f32 - 0.5) as i32 + 1));
            opp_cutoff = 2. * adj_median - v.ca_adj_value[(icut - 1) as usize];
        }
        num_added = 0;
        //
        // Added adjacent points above the cutoff to the patch
        for iy in iy_bord_low..=iy_bord_high {
            for ix in ix_bord_low..=ix_bord_high {
                let ix_offset = ix - ix_cen;
                let iy_offset = iy - iy_cen;
                if v.adjacent[dv!(ix_offset, iy_offset)] > 0
                    && !v.in_list[dv!(ix_offset, iy_offset)]
                    && *num_in_obj < LIMPATCH
                {
                    if polarity * (grow[at!(nx, ix, iy)] - cutoff) > 0. {
                        v.in_list[dv!(ix_offset, iy_offset)] = true;
                        *num_in_obj += 1;
                        v.ix_fix[(*num_in_obj - 1) as usize] = ix;
                        v.iy_fix[(*num_in_obj - 1) as usize] = iy;
                        num_added += 1;
                    } else if v.include_halo {
                        //
                        // And test in other direction if including halo
                        if -polarity * (grow[at!(nx, ix, iy)] - opp_cutoff) > 0. {
                            v.in_list[dv!(ix_offset, iy_offset)] = true;
                            *num_in_obj += 1;
                            v.ix_fix[(*num_in_obj - 1) as usize] = ix;
                            v.iy_fix[(*num_in_obj - 1) as usize] = iy;
                            num_added += 1;
                        }
                    }
                }
            }
        }
        if v.if_verbose > 0 {
            let _ = writeln!(
                out,
                "Iteration{}{} adjacent points added to patch",
                i_edit(igrow, 3),
                i_edit(num_added, 6)
            );
        }
        if num_added == 0 {
            break;
        }
        igrow += 1;
    }
    //
    // Mark higher level adjacent points to get a region of width "nborder"
    // Fixed in translation (BUGS.md): the source (`ccderaser.f90:1915`) marks
    // around the stale `ixl`, `iyl` left by the patch-point loop, so levels 2
    // and up are only ever set around that one point; the ring is meant to
    // grow around the point just tested, `ixOffset`, `iyOffset`.
    for igrow in 1..=v.num_pix_border - v.if_include_adj {
        for iy in iy_bord_low..=iy_bord_high {
            for ix in ix_bord_low..=ix_bord_high {
                let ix_offset = ix - ix_cen;
                let iy_offset = iy - iy_cen;
                if v.adjacent[dv!(ix_offset, iy_offset)] as i32 == igrow
                    && !v.in_list[dv!(ix_offset, iy_offset)]
                {
                    for i in -1..=1 {
                        for j in -1..=1 {
                            if v.adjacent[dv!(ix_offset + i, iy_offset + j)] == 0 {
                                v.adjacent[dv!(ix_offset + i, iy_offset + j)] = (igrow + 1) as i16;
                            }
                        }
                    }
                }
            }
        }
    }

    // total pixels in current area
    let _num_pix = (iy_bord_high + 1 - iy_bord_low) * (ix_bord_high + 1 - ix_bord_low);
    //
    // list of points is complete: fill regression matrix with data for all
    // points outside the list.  the ixbordlo etc. should be correct
    //
    let mut num_points = 0_i32;
    let nindep = v.iorder * (v.iorder + 3) / 2;
    xsum = 0.;
    xsq_sum = 0.;
    let min_adjacent = 2 - v.if_include_adj;
    for iy in iy_bord_low..=iy_bord_high {
        for ix in ix_bord_low..=ix_bord_high {
            let ix_offset = ix - ix_cen;
            let iy_offset = iy - iy_cen;
            let near_edge = ix - ix_bord_low < num_bord_m1
                || ix_bord_high - ix < num_bord_m1
                || iy - iy_bord_low < num_bord_m1
                || iy_bord_high - iy < num_bord_m1;
            if !v.in_list[dv!(ix_offset, iy_offset)]
                && (near_edge || v.adjacent[dv!(ix_offset, iy_offset)] as i32 >= min_adjacent)
            {
                num_points += 1;
                if nindep > 0 {
                    if num_points > LIMPATCH {
                        if !v.ca_warned {
                            let _ = write!(
                                out,
                                "\nWARNING: CCDERASER - SOME PATCHES ARE TOO LARGE FOR POLYNOMIAL FITS, USING MEAN OF SURROUNDING PIXELS\n"
                            );
                        }
                        v.ca_warned = true;
                    } else {
                        let col = ((num_points - 1) * mat_size) as usize;
                        poly_term(ix_offset, iy_offset, v.iorder, &mut v.ca_rmat[col..]);
                        v.ca_rmat[col + nindep as usize] = array[at!(nx, ix, iy)];
                    }
                }
                xsum += array[at!(nx, ix, iy)];
                xsq_sum += array[at!(nx, ix, iy)] * array[at!(nx, ix, iy)];
            }
        }
    }
    //
    // do regression or just get mean for order 0
    //
    if v.if_verbose > 0 {
        // 104 format(/,i5,' points to fix at',2i6,',',i4,' points being fit')
        let _ = writeln!(
            out,
            "\n{} points to fix at{}{},{} points being fit",
            i_edit(*num_in_obj, 5),
            i_edit(ix_cen, 6),
            i_edit(iy_cen, 6),
            i_edit(num_points, 4)
        );
    }
    if nindep > 0 && num_points <= LIMPATCH {
        // call multr(xr, nindep+1, npnts, sx, ss, ssd, d, r, xm, sd, b, b1, c1, rsq , fra)
        // The Fortran wrapper passes `wgtCol - 1` = -1.
        mult_regress(
            &v.ca_rmat,
            mat_size,
            1,
            nindep,
            num_points,
            1,
            -1,
            &mut b1,
            mat_size,
            Some(std::slice::from_mut(&mut c1)),
            &mut xm,
            &mut sd,
            &mut ssd,
        );
    }
    xmean = xsum / num_points as f32;
    //
    // If filling with noise, get enlarged area and first scan the area marking points
    // to include
    if v.fill_with_noise {
        num_sd_points = 0;
        let vic_dim = v.vic_dim;
        let ix_extra_low = 1.max(ix_bord_low - v.noise_extra_bord);
        let ix_extra_high = nx.min(ix_bord_high + v.noise_extra_bord);
        let iy_extra_low = 1.max(iy_bord_low - v.noise_extra_bord);
        let iy_extra_high = ny.min(iy_bord_high + v.noise_extra_bord);
        for iy in iy_extra_low..=iy_extra_high {
            for ix in ix_extra_low..=ix_extra_high {
                let jx = ix + 1 - ix_extra_low;
                let jy = iy + 1 - iy_extra_low;
                let ix_offset = ix - ix_cen;
                let iy_offset = iy - iy_cen;
                let near_edge = ix - ix_bord_low < num_bord_m1
                    || ix_bord_high - ix < num_bord_m1
                    || iy - iy_bord_low < num_bord_m1
                    || iy_bord_high - iy < num_bord_m1;
                if !v.in_list[dv!(ix_offset, iy_offset)]
                    && (near_edge || v.adjacent[dv!(ix_offset, iy_offset)] as i32 >= min_adjacent)
                {
                    v.vicinity[at!(vic_dim, jx, jy)] = array[at!(nx, ix, iy)];
                    v.vicinity_sqr[at!(vic_dim, jx, jy)] =
                        array[at!(nx, ix, iy)] * array[at!(nx, ix, iy)];
                } else {
                    v.vicinity_sqr[at!(vic_dim, jx, jy)] = -1.;
                }
            }
        }
        //
        // Now measure very local SD at each point if there are at least 6 included
        for iy in 2..=(iy_extra_high - iy_extra_low) - 1 {
            for ix in 2..=(ix_extra_high - ix_extra_low) - 1 {
                ixl = 0;
                xsum = 0.;
                xsq_sum = 0.;
                for jy in iy - 1..=iy + 1 {
                    for jx in ix - 1..=ix + 1 {
                        if v.vicinity_sqr[at!(vic_dim, jx, jy)] >= 0. {
                            xsum += v.vicinity[at!(vic_dim, jx, jy)];
                            xsq_sum += v.vicinity_sqr[at!(vic_dim, jx, jy)];
                            ixl += 1;
                        }
                    }
                }
                if ixl >= 6 {
                    num_sd_points += 1;
                    sums_to_avg_sd(
                        xsum,
                        xsq_sum,
                        ixl,
                        &mut big_mean,
                        &mut v.sd_array[(num_sd_points - 1) as usize],
                    );
                }
            }
        }
        //
        // If there are enough points, take the median as the SD to use
        if num_sd_points >= min_sds_for_median {
            rs_fast_median_in_place(&mut v.sd_array, num_sd_points, &mut big_sd);
        }
    }
    //
    // replace points on list with values calculated from fit
    // cannot truncate range by nbordm1 because could be on edge of image
    //
    let mut iy = iy_bord_high;
    while iy >= iy_bord_low {
        for ix in ix_bord_low..=ix_bord_high {
            let ix_offset = ix - ix_cen;
            let iy_offset = iy - iy_cen;
            if v.in_list[dv!(ix_offset, iy_offset)] {
                if nindep > 0 && num_points <= LIMPATCH {
                    poly_term(ix_offset, iy_offset, v.iorder, &mut vector);
                    xmean = c1;
                    for i in 1..=nindep {
                        xmean += b1[(i - 1) as usize] * vector[(i - 1) as usize];
                    }
                }
                if v.fill_with_noise && num_sd_points >= min_sds_for_median {
                    let gauss = gaussian_deviate(0);
                    array[at!(nx, ix, iy)] = xmean + gauss * big_sd;
                } else {
                    array[at!(nx, ix, iy)] = xmean;
                }
            }
        }
        iy -= 1;
    }
}

/// Original `fillBoundaryArrays` (`ccderaser.f90:2071`).
///
/// FILLBOUNDARYARRAYS puts a contour into X and Y boundary arrays and returns
/// the min and max values.
pub fn fill_boundary_arrays(
    iobj: i32,
    xbound: &mut [f32],
    ybound: &mut [f32],
    xmin: &mut f32,
    xmax: &mut f32,
    y_min: &mut f32,
    ymax: &mut f32,
    fm: &FortModel,
) {
    let ibase = fm.ibase_obj[(iobj - 1) as usize];
    *xmin = 1.0e20;
    *xmax = -1.0e20;
    *y_min = *xmin;
    *ymax = *xmax;
    //
    // Put points in boundary array and get min/max
    for ip in 1..=fm.npt_in_obj[(iobj - 1) as usize] {
        let ipt = fm.object[(ibase + ip - 1) as usize];
        let k = (ip - 1) as usize;
        xbound[k] = fm.p_coord[(ipt - 1) as usize][0];
        ybound[k] = fm.p_coord[(ipt - 1) as usize][1];
        *xmin = f_min(*xmin, xbound[k]);
        *xmax = f_max(*xmax, xbound[k]);
        *y_min = f_min(*y_min, ybound[k]);
        *ymax = f_max(*ymax, ybound[k]);
    }
}

/// Original `convertBoundary` (`ccderaser.f90:2098`).
///
/// CONVERTBOUNDARY Converts a boundary contour to a list of interior
/// points, in the same contour
pub fn convert_boundary(
    iobj: i32,
    nx: i32,
    ny: i32,
    xbound: &[f32],
    ybound: &[f32],
    xmin: f32,
    xmax: f32,
    y_min: f32,
    ymax: f32,
    fm: &mut FortModel,
) {
    let ibase = fm.ibase_obj[(iobj - 1) as usize];
    let zz = fm.p_coord[(fm.object[ibase as usize] - 1) as usize][2];
    fm.ibase_obj[(iobj - 1) as usize] = fm.ibase_free;
    let mut num_in_obj = 0;
    let ix_start = 1.max((xmin - 2.).round() as i32);
    let ix_end = nx.min((xmax + 2.).round() as i32);
    let iy_start = 1.max((y_min - 2.).round() as i32);
    let iy_end = ny.min((ymax + 2.).round() as i32);
    //
    // Look at all pixels in range, add to object at new base
    for iy in iy_start..=iy_end {
        for ix in ix_start..=ix_end {
            let xx = ix as f32 - 0.5;
            let yy = iy as f32 - 0.5;
            if inside(xbound, ybound, fm.npt_in_obj[(iobj - 1) as usize], xx, yy) {
                num_in_obj += 1;
                fm.n_point += 1;
                if fm.n_point > fm.max_pt {
                    exit_error("Not enough model array space to convert boundary contours");
                }
                fm.ibase_free += 1;
                fm.object[(fm.ibase_free - 1) as usize] = fm.n_point;
                let pc = &mut fm.p_coord[(fm.n_point - 1) as usize];
                pc[0] = xx;
                pc[1] = yy;
                pc[2] = zz;
            }
        }
    }
    fm.npt_in_obj[(iobj - 1) as usize] = num_in_obj;
}

/// Original `addCircleToPatch` (`ccderaser.f90:2137`).
///
/// ADDCIRCLETOPATCH adds a circle to the current patch, keeping track of
/// the overall mins and maxes
pub fn add_circle_to_patch(
    iobj: i32,
    ipt: i32,
    size: f32,
    num_in_obj: &mut i32,
    ix_fix_min: &mut i32,
    ix_fix_max: &mut i32,
    iy_fix_min: &mut i32,
    iy_fix_max: &mut i32,
    v: &mut CcdVars,
    fm: &FortModel,
) {
    let ip = fm.object[(fm.ibase_obj[(iobj - 1) as usize] + ipt - 1) as usize];
    let xcen = fm.p_coord[(ip - 1) as usize][0];
    let ycen = fm.p_coord[(ip - 1) as usize][1];
    let ix_start = 1.max((xcen - size - 2.).round() as i32);
    let ix_end = v.nx.min((xcen + size + 2.).round() as i32);
    let iy_start = 1.max((ycen - size - 2.).round() as i32);
    let iy_end = v.ny.min((ycen + size + 2.).round() as i32);
    let num_in_start = *num_in_obj;
    for iy in iy_start..=iy_end {
        for ix in ix_start..=ix_end {
            let xx = ix as f32 - 0.5;
            let yy = iy as f32 - 0.5;
            if (xx - xcen).powi(2) + (yy - ycen).powi(2) <= size.powi(2) {
                let mut if_in_patch = 0;
                if ix >= *ix_fix_min && ix <= *ix_fix_max && iy >= *iy_fix_min && iy <= *iy_fix_max
                {
                    for ip in 1..=num_in_start {
                        let k = (ip - 1) as usize;
                        if ix == v.ix_fix[k] && iy == v.iy_fix[k] {
                            if_in_patch = 1;
                            break;
                        }
                    }
                }
                if if_in_patch == 0 {
                    *num_in_obj += 1;
                    //
                    // This is not supposed to happen, but better to check...
                    if *num_in_obj > LIMPATCH {
                        exit_error("Too many points in patch to merge another circle in");
                    }
                    v.ix_fix[(*num_in_obj - 1) as usize] = ix;
                    v.iy_fix[(*num_in_obj - 1) as usize] = iy;
                    *ix_fix_min = (*ix_fix_min).min(ix);
                    *ix_fix_max = (*ix_fix_max).max(ix);
                    *iy_fix_min = (*iy_fix_min).min(iy);
                    *iy_fix_max = (*iy_fix_max).max(iy);
                }
            }
        }
    }
}

/// Original `contourArea` (`ccderaser.f90:2192`).
///
/// CONTOURAREA Measures the area of a contour
pub fn contour_area(iobj: i32, fm: &FortModel) -> f32 {
    let mut area = 0.0_f32;
    let num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
    if num_in_obj < 3 {
        return area;
    }
    let ibase = fm.ibase_obj[(iobj - 1) as usize];
    for ipt in 1..=num_in_obj {
        let next_pt = ipt % num_in_obj + 1;
        let ip1 = fm.object[(ibase + ipt - 1) as usize];
        let ip2 = fm.object[(ibase + next_pt - 1) as usize];
        let p1 = fm.p_coord[(ip1 - 1) as usize];
        let p2 = fm.p_coord[(ip2 - 1) as usize];
        area += (p2[1] + p1[1]) * (p2[0] - p1[0]);
    }
    (area * 0.5).abs()
}

/// Original `taperInsideCont` (`ccderaser.f90:2215`).
///
/// TAPERINSIDECONT erases points inside of a large contour and tapers down on
/// the inside from a border value to the mean value.
pub fn taper_inside_cont(
    array: &mut [f32],
    nx: i32,
    ny: i32,
    xbound: &[f32],
    ybound: &[f32],
    num_in_obj: i32,
    xmin: f32,
    xmax: f32,
    y_min: f32,
    ymax: f32,
    max_adj_val: i32,
    iferr: &mut i32,
) {
    // real*4 adjValues(maxAdjVal)
    let mut adj_values = vec![0.0_f32; max_adj_val.max(0) as usize];
    let mut sum: f32;
    let (mut segment_x, mut segment_y, mut vector_x, mut vector_y): (f32, f32, f32, f32);
    let (mut x_line, mut yline): (f32, f32);
    let mut dist: f32;
    let mut dist_min: f32;
    let mut t: f32;
    let mut tmin = 0.0_f32;
    let (mut dx, mut dy): (f32, f32);
    let mut fill: f32;
    let (mut xx, mut yy): (f32, f32);
    let mut frac: f32;
    let mut vec_len: f32;
    let mut num_sum: i32;
    let mut ip_next: i32;
    let (mut ix, mut iy): (i32, i32);
    let (mut ixf, mut iyf): (i32, i32);
    let mut num_vec_pts: i32;
    let mut ip_min = 1_i32;
    let taper = 8.0_f32;
    let taper_sq = taper.powi(2);
    *iferr = 1;
    let xb = |k: i32| xbound[(k - 1) as usize];
    let yb = |k: i32| ybound[(k - 1) as usize];
    //
    // First we need the mean outside the periphery
    sum = 0.;
    num_sum = 0;
    for ip in 1..=num_in_obj {
        ip_next = ip % num_in_obj + 1;
        segment_x = xb(ip_next) - xb(ip);
        segment_y = yb(ip_next) - yb(ip);
        vec_len = (segment_x.powi(2) + segment_y.powi(2)).sqrt();
        if vec_len > 0. {
            vector_x = 1.5 * segment_x / vec_len;
            vector_y = 1.5 * segment_y / vec_len;
            num_vec_pts = vec_len as i32;
            //
            // Go to middle of line and test a point to one side
            x_line = xb(ip) + segment_x * 0.5;
            yline = yb(ip) + segment_y * 0.5;
            ix = (x_line + vector_y + 0.5).round() as i32;
            iy = (yline - vector_x + 0.5).round() as i32;
            if inside(xbound, ybound, num_in_obj, ix as f32 - 0.5, iy as f32 - 0.5) {
                //
                // If that wasn't outside, reverse direction and test that point and give up
                // on segment if neither one is outside
                vector_x = -vector_x;
                vector_y = -vector_y;
                ix = (x_line + vector_y + 0.5).round() as i32;
                iy = (yline - vector_x + 0.5).round() as i32;
                if inside(xbound, ybound, num_in_obj, ix as f32 - 0.5, iy as f32 - 0.5) {
                    continue;
                }
            }
            //
            // Then loop along line and collect some points
            for i in 1..=num_vec_pts {
                x_line = xb(ip) + (segment_x * i as f32) / num_vec_pts as f32;
                yline = yb(ip) + (segment_y * i as f32) / num_vec_pts as f32;
                ix = (x_line + vector_y + 0.5).round() as i32;
                iy = (yline - vector_x + 0.5).round() as i32;
                if ix > 0 && ix <= nx && iy > 0 && iy <= ny {
                    sum += array[at!(nx, ix, iy)];
                    num_sum += 1;
                    if num_sum <= max_adj_val {
                        adj_values[(num_sum - 1) as usize] = array[at!(nx, ix, iy)];
                    }
                }
            }
        }
    }

    if num_sum < 3 {
        return;
    }
    *iferr = 0;
    fill = sum / num_sum as f32;
    //
    // Use the median instead of the mean if the array did not fill up and there are enough
    // points
    if num_sum <= max_adj_val && num_sum > 20 {
        rs_fast_median_in_place(&mut adj_values, num_sum, &mut fill);
    }
    let ix_start = 1.max(nx.min((xmin - 0.5).floor() as i32));
    let iy_start = 1.max(ny.min((y_min - 0.5).floor() as i32));
    let ix_end = 1.max(nx.min((xmax - 0.5).ceil() as i32));
    let iy_end = 1.max(ny.min((ymax - 0.5).ceil() as i32));
    for iy in iy_start..=iy_end {
        for ix in ix_start..=ix_end {
            xx = ix as f32 - 0.5;
            yy = iy as f32 - 0.5;
            if inside(xbound, ybound, num_in_obj, xx, yy) {
                //
                // Find nearest distance to contour
                dist_min = 1.0e30;
                t = 0.;
                for ip in 1..=num_in_obj {
                    ip_next = ip % num_in_obj + 1;
                    dx = xb(ip_next) - xb(ip);
                    dy = yb(ip_next) - yb(ip);
                    if dx != 0. || dy != 0. {
                        t = ((xx - xb(ip)) * dx + (yy - yb(ip)) * dy) / (dx.powi(2) + dy.powi(2));
                    }
                    // `ccderaser.f90:2294`: `maxss t, 0.` then `minss ., 1.`.
                    t = f_min(f_max(t, 0.), 1.);
                    dist = (xx - (xb(ip) + t * dx)).powi(2) + (yy - (yb(ip) + t * dy)).powi(2);
                    if dist < dist_min {
                        dist_min = dist;
                        ip_min = ip;
                        tmin = t;
                    }
                }
                //
                // If it is close to contour, get pixel on other side
                array[at!(nx, ix, iy)] = fill;
                if dist_min < taper_sq {
                    ip_next = ip_min % num_in_obj + 1;
                    dx = xb(ip_next) - xb(ip_min);
                    dy = yb(ip_next) - yb(ip_min);
                    x_line = xb(ip_min) + dx * tmin;
                    yline = yb(ip_min) + dy * tmin;
                    segment_x = x_line - xx;
                    segment_y = yline - yy;
                    vec_len = (segment_x.powi(2) + segment_y.powi(2)).sqrt();
                    if vec_len > 1.0e-6 {
                        xx = x_line + 1.2 * segment_x / vec_len;
                        yy = yline + 1.2 * segment_y / vec_len;
                        ixf = (xx + 0.5).round() as i32;
                        iyf = (yy + 0.5).round() as i32;
                        if ixf > 0 && ixf <= nx && iyf > 0 && iyf <= ny {
                            frac = (dist_min / taper_sq).sqrt();
                            array[at!(nx, ix, iy)] =
                                frac * fill + (1. - frac) * array[at!(nx, ixf, iyf)];
                        }
                    }
                }
            }
        }
    }
}

/// Original `typeOnList` (`ccderaser.f90:2333`).
pub fn type_on_list(itype: i32, i_type_list: &[i32], num_type_list: i32) -> bool {
    if num_type_list == 1 && i_type_list[0] == -999 {
        return true;
    }
    for i in 1..=num_type_list {
        if itype == i_type_list[(i - 1) as usize] {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// gfortran runtime boundary: the `MIN`/`MAX` intrinsics on reals, formatted
// and list-directed output editing and the runtime error of a failed read.
// Not translations of source units: what gfortran and libgfortran do for the
// intrinsics, edit descriptors and `print *` items this program uses.
// ---------------------------------------------------------------------------

/// gfortran `MIN(a, b)` on `real*4`, as `a < b ? a : b` (`minss a, b`).
/// Each call site passes the operands in the order the reference object
/// emits (checked against `-fverbose-asm` at every site).
fn f_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// gfortran `MAX(a, b)` on `real*4`, as `a > b ? a : b` (`maxss`).
fn f_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `Iw` editing: right-justified in `w`; `w` asterisks if it does not fit.
fn i_edit(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// `Fw.d` editing: the value rounded to `d` decimals (round-half-even on the
/// exact binary value, as libgfortran does), a trailing point for `d = 0`,
/// the leading zero dropped when that is what makes it fit, `w` asterisks
/// when it still does not.  `NaN`/`Infinity` are right-justified.
fn f_edit(value: f32, w: usize, d: usize) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let mut text = if value < 0. { "-Infinity" } else { "Infinity" };
        if text.len() > w {
            text = if value < 0. { "-Inf" } else { "Inf" };
        }
        if text.len() > w {
            return "*".repeat(w);
        }
        return format!("{text:>w$}");
    }
    let mut text = format!("{value:.d$}");
    if d == 0 {
        text.push('.');
    }
    if text.len() > w {
        if let Some(rest) = text.strip_prefix("0.") {
            text = format!(".{rest}");
        } else if let Some(rest) = text.strip_prefix("-0.") {
            text = format!("-.{rest}");
        }
    }
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// A list-directed (`print *`) `integer*4` item: a blank separator (the
/// record's leading blank when it is the first item) and `I11`.
fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}

/// A list-directed `real*4` item (its blank separator included): `F` form
/// with nine significant digits and four trailing blanks for magnitudes in
/// [0.1, 1e9), `E` form with a two-digit exponent otherwise; `NaN` and
/// `Infinity` right-justified in the 17 columns.
fn ld_real(value: f32) -> String {
    if value.is_nan() {
        return format!("{:>17}", "NaN");
    }
    if value.is_infinite() {
        return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
    }
    if value == 0. {
        return format!("{:>13}    ", format!("{value:.8}"));
    }
    let scientific = format!("{:.8e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=9).contains(&k) {
        let mut text = format!("{:.*}", (9 - k) as usize, value);
        if k == 9 {
            text.push('.');
        }
        format!("{text:>13}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>17}",
            format!(
                "{}{}E{}{:02}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}

/// A formatted or list-directed `read` with no `END=`/`ERR=` that fails:
/// libgfortran reports it and stops with status 2.
fn read_runtime_error(err: ListReadError) -> ! {
    let _ = ImodFile::Stdout.flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}
