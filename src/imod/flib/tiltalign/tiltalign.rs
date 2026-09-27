//! Translation of `IMOD/flib/tiltalign/tiltalign.cpp` with its class header
//! `IMOD/flib/tiltalign/tiltalign.h` merged in.
//!
//! TILTALIGN solves for the displacements, rotations, tilts, and
//! magnification differences relating a set of tilted views of an object,
//! from a set of fiducial points identified in a series of views and read
//! from a model in which each fiducial point is a separate object or contour.
//!
//! # Structure
//!
//! The C++ `TiltAlign` class is the [`TiltAlign`] struct, one method per member
//! function with the original name in each doc comment: `main`,
//! `alignAndOutputResults`, `processFillInPoints`,
//! `setupAndDoLocalAlignments`, `addToLocalIfInRange`,
//! `findLocalSizeAndPoints`, `findMedianResidual`, `doLeaveOutRuns`,
//! `outputLeaveOutErrors`.  The file-scope `main` is [`tiltalign`].
//!
//! The file-scope statics `alignVars`/`av`, `arrayMaxes`/`mx` and
//! `sepGroups`/`sg` (`tiltalign.cpp:29-34`) are the fields `av`, `mx`, `sg` of
//! the struct: every member function reaches them, and the callees of the
//! other tiltalign units take them as parameters (`alivar.rs`), which is what
//! the source's `*SetPointers` calls in `main` arranged.  The `fmod*` globals
//! of `fortmodel.c` that `input_model` fills and `write_xyz_model` reads are
//! the field `fm`, as for every other `use fortmodel` unit.  `static int zero
//! = 0, one = 1` (`:36`) exist only to pass constants by address and are the
//! literals here.
//!
//! # Representation
//!
//! - Every `float *`/`int *` member is an owned `Vec`; `NULL` (never
//!   allocated) is an empty `Vec`.  `B3DMALLOC` leaves memory uninitialised
//!   where a `Vec` is zeroed; where the source reads an element it never wrote
//!   this is noted in place.  The class instance is a stack object in the
//!   source's `main`, so members the constructor does not set are
//!   uninitialised there and 0 here; every one of them is written before it is
//!   read on the paths the program takes, except where noted.
//! - `FILE *` members are `Option<ImodFile>`; `fclose` is a `take()` (the
//!   `ImodFile` flushes when dropped).  `mSolFileFP` is never closed in the
//!   source; it is flushed at exit like every open stream.
//! - `char *` names are `Option<String>`; `mRobFailMess[ROB_MESS_SIZE + 1]` is
//!   a 121-byte buffer read up to its first NUL; `mPixUnits` a `String`.
//! - `computeWeights(mIndAllReal, (float *)mIndSave, (float *)mJptSave, (int
//!   *)mXyzErr)` (`:978`) reinterprets two `int` members as `float` scratch
//!   and a `float` member as `int` scratch.  The same reinterpretation is made
//!   here over the same storage (`i32` and `f32` have one size and alignment
//!   and every bit pattern is valid in both), so the members keep the source's
//!   contents across the call.
//!
//! # Arithmetic
//!
//! This is C++: `sqrt`, `sin`, `cos`, `atan` and `pow(x, 2.f)` of a `float`
//! resolve to the `float` overloads (`sqrtf`, `sinf`, `cosf`, `atanf`, and
//! `powf(x, 2)` which GCC folds to `x * x`; those are the reference object's
//! imports), so they are `f32` methods here.  `sqrt` of a `double` expression
//! (`:2184`) is the `double` one.  Literals such as `0.1`, `2.`, `1.05` are
//! `double` and widen the `float` they meet; `B3DNINT` adds a `double` `0.5`.
//! `mDtor = RADIANS_PER_DEGREE` stores the double constant into a `float`.
//!
//! # `rand`
//!
//! `srand`/`rand` are the C library's TYPE_3 generator shared with
//! `leaveout.cpp`; the translation's is `b3dutil::b3dsrand`/`b3drand`, and
//! `(float)rand() / (float)RAND_MAX` is exactly `b3drand()` (`leaveout.rs`).
//!
//! # Deviations (uninitialised or out-of-bounds reads in the source)
//!
//! - `crossValOption` (`:109`) is never initialised and `PipGetInteger`
//!   leaves it unchanged when `CrossValidate` is not entered.  Fixed in
//!   translation (2026-09-26, `BUGS.md`): it defaults to 0, no
//!   cross-validation.
//! - `countNumInView` and `addToLocalIfInRange` index the test-set flag arrays
//!   with 1-based point numbers, reading the next point's flag and one element
//!   past arrays allocated `nrealPt` long (`BUGS.md`, wave 2 and wave 6
//!   sections).  Fixed in translation (2026-09-26): each reads the point's own
//!   flag.
//! - With one fiducial point the initial `solveXyzd` (`:630`) reads the
//!   uninitialised `xxUse`/`nrealUse` pointers; here it uses `av`'s arrays.
//! - `ivst`/`ivnd` (`:1228`) and `err`/`robErr` (`:2709-2712`) start at 0.

use std::io::Write as _;
use std::sync::atomic::{AtomicI32, Ordering};

use super::alivar::AlignVariables;
use super::arraymaxes::{ArrayMaxes, MAX_REAL_FOR_DIRECT_INIT, MAX_WGT_RINGS};
use super::beamtilt::{beamtilt_set_pointers, run_metro, search_beam_tilt};
use super::evalfunct::EvalFunct;
use super::fill_matrices::{
    convert_for_beamtilt, fill_beam_matrices, fill_dist_matrix, fill_proj_matrix, fill_rot_matrix,
    fill_xtilt_matrix, fill_ytilt_matrix, mat_product,
};
use super::find_surfaces::find_surfaces;
use super::input_model::{input_model, input_model_set_pointers, write_xyz_model};
use super::input_vars::{
    expand_local_to_all, input_vars, input_vars_set_pointers, nearest_view, reload_vars,
};
use super::leaveout::{
    clear_leave_out_errors, get_leave_out_errors, get_test_set_errors, leave_out_points,
    leaveout_set_pointers,
};
use super::map_vars::map_vars_set_pointers;
use super::mapsepgroups::MapSepGroups;
use super::patchtrack::{
    copy_xyz_from_full_tracks_to_real, load_patch_subset, make_full_tracks_for_init,
    patch_track_set_pointers, restore_from_patch_sample,
};
use super::robustfit::{
    compute_weights, robust_set_pointers, setup_track_weight_groups, setup_weight_groups,
};
use super::solve_xyzd::solve_xyzd;
use super::utilfuncs::{copy_array, count_num_in_view, error_exit, formatted_error, memory_error};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::amat_to_rotmagstr::amat_to_rotmagstr;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_open_file, b3drand, b3dsrand, c_format_bytes, exit as c_exit,
    imod_backup_file, imod_prog_name, imod_usage_header, number_in_list,
};
use crate::imod::libcfshr::linearxforms::{xf_copy, xf_invert, xf_mult, xf_unit};
use crate::imod::libcfshr::parse_params::{
    pip_allow_comma_defaults, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_three_integers, pip_get_two_floats, pip_get_two_integers, pip_read_or_parse_options,
};
use crate::imod::libcfshr::readlinevalues::{
    ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};
use crate::imod::libcfshr::regression::polynomial_fit;
use crate::imod::libcfshr::robuststat::rs_fast_median_in_place;
use crate::imod::libcfshr::simplestat::{ls_fit, ls_fit2};
use crate::imod::libimod::imodel_fwrap::{addimodpoint, writeimod};

/// `tiltalign.h:1`: `#define ROB_MESS_SIZE 120`.
const ROB_MESS_SIZE: usize = 120;
/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `b3dutil.h:34`: `#define B3DABS(a) ((a) >= 0 ? (a) : -(a))`.
macro_rules! b3dabs {
    ($a:expr) => {{
        let a = $a;
        if a >= 0 as _ { a } else { -a }
    }};
}

/// The source's `printf`, through the C-format writer on libc-order stdout.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// The source's `fprintf` to an open `ImodFile`.
macro_rules! fprintf {
    ($fp:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = $fp.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Original: `main` (`tiltalign.cpp:38`).
///
/// Constructs the `TiltAlign` object, makes the `*SetPointers` calls (which
/// have nothing to store in the translation: the callees take `av`, `mx`,
/// `sg` and the `EvalFunct` as parameters), runs the program, and exits 0.
pub fn tiltalign(argv: &[String]) -> i32 {
    let mut ta = TiltAlign::new();
    leaveout_set_pointers(&mut ta.av);
    map_vars_set_pointers(&mut ta.av, &mut ta.mx, &mut ta.sg);
    robust_set_pointers(&mut ta.av, &mut ta.mx);
    patch_track_set_pointers(&mut ta.av);
    input_vars_set_pointers(&mut ta.av, &mut ta.mx, &mut ta.sg);
    input_model_set_pointers(&mut ta.av, &mut ta.mx, &mut ta.m_eval_funct);
    beamtilt_set_pointers(&mut ta.av);
    ta.main(argv);
    c_exit(0);
}

/// Original: `class TiltAlign` (`tiltalign.h:4-190`), plus the file-scope
/// statics of `tiltalign.cpp` that every member reaches (module doc).
/// Members follow the header's order.
pub struct TiltAlign {
    /// `static AlignVariables alignVars` / `*av` (`tiltalign.cpp:29-30`).
    av: AlignVariables,
    /// `static ArrayMaxes arrayMaxes` / `*mx` (`tiltalign.cpp:31-32`).
    mx: ArrayMaxes,
    /// `static MapSepGroups sepGroups` / `*sg` (`tiltalign.cpp:33-34`).
    sg: MapSepGroups,
    /// The `fmod*` globals of `fortmodel.c`.
    fm: FortModel,
    /// `EvalFunct mEvalFunct` (`tiltalign.h:18`, public).
    pub m_eval_funct: EvalFunct,
    m_max_h: i32,
    m_nin_real: Vec<i32>,
    m_igroup: Vec<i32>,
    /// Array of object numbers for each real point, an "all" array.
    m_imod_obj: Vec<i32>,
    m_imod_cont: Vec<i32>,
    m_var: Vec<f32>,
    m_var_err: Vec<f32>,
    m_grad: Vec<f32>,
    m_h: Vec<f32>,
    m_var_name: Vec<u8>,
    m_tilt_orig: Vec<f32>,
    m_view_res: Vec<f32>,
    m_xyz_err: Vec<f32>,
    m_num_in_view: Vec<i32>,
    m_ind_save: Vec<i32>,
    m_jpt_save: Vec<i32>,
    m_err_save: Vec<f32>,
    m_wgt_prev: Vec<f32>,
    m_var_save: Vec<f32>,
    m_view_errsum: Vec<f32>,
    m_view_errsq: Vec<f32>,
    m_view_mean_res: Vec<f32>,
    m_view_sd_res: Vec<f32>,
    m_order_err: bool,
    m_nearby_err: bool,
    m_residual_file: String,
    m_fixed_xyz_file: Option<String>,
    /// `char mRobFailMess[ROB_MESS_SIZE + 1]`.
    m_rob_fail_mess: Vec<u8>,
    /// `char mPixUnits[8]`.
    m_pix_units: String,
    m_fl: Vec<f32>,
    m_xz_fac: Vec<f32>,
    m_yz_fac: Vec<f32>,
    m_all_xyz: Vec<f32>,
    m_fixed_xyz: Vec<f32>,
    m_all_xx: Vec<f32>,
    m_all_yy: Vec<f32>,
    m_glb_fl: Vec<f32>,
    m_glb_xz_fac: Vec<f32>,
    m_glb_yz_fac: Vec<f32>,
    m_iall_sec_vw: Vec<i32>,
    m_iall_real_str: Vec<i32>,
    m_list_real: Vec<i32>,
    m_ind_all_real: Vec<i32>,
    m_map_all_to_local: Vec<i32>,
    m_map_local_to_all: Vec<i32>,
    m_map_fill_local_to_all: Vec<i32>,
    m_map_all_file_to_view: Vec<i32>,
    m_map_all_view_to_file: Vec<i32>,
    m_ss_afac: Vec<f32>,
    m_ss_bfac: Vec<f32>,
    m_ss_cfac: Vec<f32>,
    m_ss_dfac: Vec<f32>,
    m_ss_efac: Vec<f32>,
    m_ss_ffac: Vec<f32>,
    m_xfill_final: Vec<f32>,
    m_yfill_final: Vec<f32>,
    m_num_in_fill_sum: Vec<i32>,
    m_ifill_all_real_str: Vec<i32>,
    m_iall_fill_view: Vec<i32>,
    m_too_few_fid: bool,
    m_sub_sample_tracks: bool,
    m_num_sample_target: i32,
    m_min_sampled_in_view: i32,
    m_make_filled_fid: i32,
    m_max_cycles: i32,
    m_dtor: f32,
    m_num_local_res: i32,
    m_num_surface: i32,
    m_metro_error: i32,
    m_map_alf_end: i32,
    m_if_var_out: i32,
    m_if_res_out: i32,
    m_if_xyz_out: i32,
    m_if_local: i32,
    m_nvar_search: i32,
    m_nvar_angle: i32,
    m_nvar_scaled: i32,
    m_err_crit: f32,
    m_fac_metro: f32,
    m_znew: f32,
    m_xtilt_new: f32,
    m_scale_xy: f32,
    m_znew_input: f32,
    m_x_shift: f32,
    m_z_shift: f32,
    m_dy_avg: f32,
    m_y_shift: f32,
    m_iu_angle: Option<ImodFile>,
    m_iu_xtilt: Option<ImodFile>,
    m_iun_local: Option<ImodFile>,
    m_sol_file_fp: Option<ImodFile>,
    m_n_all_real_pt: i32,
    m_map_alf_start: i32,
    m_num_patch_x: i32,
    m_num_patch_y: i32,
    m_n_all_view: i32,
    m_n_all_proj_pt: i32,
    m_ipatch_x: i32,
    m_ipatch_y: i32,
    m_num_bot: i32,
    m_num_top: i32,
    m_ix_min: i32,
    m_ix_max: i32,
    m_iy_min: i32,
    m_iy_max: i32,
    m_num_proj_pt: i32,
    m_min_tilt_view: i32,
    m_ncomp_search: i32,
    m_map_tilt_start: i32,
    m_if_bt_search: i32,
    m_if_original_z: i32,
    m_xcen: f32,
    m_ycen: f32,
    m_dx_min: f32,
    m_tilt_add: f32,
    m_if_zfac: i32,
    m_if_do_robust: i32,
    m_did_robust: bool,
    m_did_full_robust: bool,
    m_min_res_robust: i32,
    m_min_local_res_robust: i32,
    m_pixel_delta: [f32; 3],
    m_pixel_size: f32,
    m_errsum_local: f32,
    m_err_local_min: f32,
    m_err_local_max: f32,
    m_rot_entered: f32,
    m_bin_step_ini: f32,
    m_bin_step_final: f32,
    m_scan_step: f32,
    m_if_do_local: i32,
    m_nin_thresh: i32,
    m_pipinput: i32,
    m_num_wgt_total: i32,
    m_num_wgt_zero: i32,
    m_num_wgt1: i32,
    m_num_wgt2: i32,
    m_num_wgt5: i32,
    m_max_del_wgt_below_crit: i32,
    m_max_robust_one_cycle: i32,
    m_robust_tot_cycle_fac: f32,
    m_del_wgt_mean_crit: f32,
    m_del_wgt_max_crit: f32,
    m_robust_max_del_err: f32,
    m_wgt_err_sum_local: f32,
    m_wgt_err_local_min: f32,
    m_wgt_err_local_max: f32,
    m_undo_wgt_relax_crit: f32,
    m_num_rob_failed: i32,
    m_num_fillin_pts: i32,
    m_local_mu_ratio_sum: f32,
    m_min_local_mu_ratio: f32,
    m_max_local_mu_ratio: f32,
    m_lv_out_num_predict: i32,
    m_lv_out_num_pad: i32,
    m_frac_leave_out: f32,
    m_frac_predict: f32,
    m_num_train_real: i32,
    m_lv_out_num_runs: i32,
    m_lv_out_coverage: f32,
    m_lv_out_target_cover: f32,
    m_lv_out_min_coverage: f32,
    m_lv_out_max_coverage: f32,
    m_weight_none_left_out: Vec<f32>,
    /// `float mTestSetFracOrStep` (`tiltalign.h:185`): declared and never used
    /// in the source (the value lives in `av->testSetFracStep`).
    m_test_set_frac_or_step: f32,
    m_all_real_in_test_set: Vec<i32>,
    m_var_all_points: Vec<f32>,
    m_lv_out_save_all_xyz: Vec<f32>,
    m_fit_num_patches_in_fid_area: bool,
}

/// `static int iterCount` in `findMedianResidual` (`tiltalign.cpp:2487`):
/// counted and never read (its only reader is a commented-out `write`).
static FIND_MEDIAN_ITER_COUNT: AtomicI32 = AtomicI32::new(0);

impl TiltAlign {
    /// Original: constructor `TiltAlign::TiltAlign()` (`tiltalign.cpp:52`).
    pub fn new() -> TiltAlign {
        let mut ta = TiltAlign {
            av: AlignVariables::default(),
            mx: ArrayMaxes::default(),
            sg: MapSepGroups::default(),
            fm: FortModel::default(),
            m_eval_funct: EvalFunct::new(),
            m_max_h: 0,
            m_nin_real: Vec::new(),
            m_igroup: Vec::new(),
            m_imod_obj: Vec::new(),
            m_imod_cont: Vec::new(),
            m_var: Vec::new(),
            m_var_err: Vec::new(),
            m_grad: Vec::new(),
            m_h: Vec::new(),
            m_var_name: Vec::new(),
            m_tilt_orig: Vec::new(),
            m_view_res: Vec::new(),
            m_xyz_err: Vec::new(),
            m_num_in_view: Vec::new(),
            m_ind_save: Vec::new(),
            m_jpt_save: Vec::new(),
            m_err_save: Vec::new(),
            m_wgt_prev: Vec::new(),
            m_var_save: Vec::new(),
            m_view_errsum: Vec::new(),
            m_view_errsq: Vec::new(),
            m_view_mean_res: Vec::new(),
            m_view_sd_res: Vec::new(),
            m_order_err: false,
            m_nearby_err: false,
            m_residual_file: String::new(),
            m_fixed_xyz_file: None,
            m_rob_fail_mess: vec![0u8; ROB_MESS_SIZE + 1],
            m_pix_units: String::new(),
            m_fl: Vec::new(),
            m_xz_fac: Vec::new(),
            m_yz_fac: Vec::new(),
            m_all_xyz: Vec::new(),
            m_fixed_xyz: Vec::new(),
            m_all_xx: Vec::new(),
            m_all_yy: Vec::new(),
            m_glb_fl: Vec::new(),
            m_glb_xz_fac: Vec::new(),
            m_glb_yz_fac: Vec::new(),
            m_iall_sec_vw: Vec::new(),
            m_iall_real_str: Vec::new(),
            m_list_real: Vec::new(),
            m_ind_all_real: Vec::new(),
            m_map_all_to_local: Vec::new(),
            m_map_local_to_all: Vec::new(),
            m_map_fill_local_to_all: Vec::new(),
            m_map_all_file_to_view: Vec::new(),
            m_map_all_view_to_file: Vec::new(),
            m_ss_afac: Vec::new(),
            m_ss_bfac: Vec::new(),
            m_ss_cfac: Vec::new(),
            m_ss_dfac: Vec::new(),
            m_ss_efac: Vec::new(),
            m_ss_ffac: Vec::new(),
            m_xfill_final: Vec::new(),
            m_yfill_final: Vec::new(),
            m_num_in_fill_sum: Vec::new(),
            m_ifill_all_real_str: Vec::new(),
            m_iall_fill_view: Vec::new(),
            m_too_few_fid: false,
            m_sub_sample_tracks: false,
            m_num_sample_target: 0,
            m_min_sampled_in_view: 0,
            m_make_filled_fid: 0,
            m_max_cycles: 0,
            m_dtor: 0.,
            m_num_local_res: 0,
            m_num_surface: 0,
            m_metro_error: 0,
            m_map_alf_end: 0,
            m_if_var_out: 0,
            m_if_res_out: 0,
            m_if_xyz_out: 0,
            m_if_local: 0,
            m_nvar_search: 0,
            m_nvar_angle: 0,
            m_nvar_scaled: 0,
            m_err_crit: 0.,
            m_fac_metro: 0.,
            m_znew: 0.,
            m_xtilt_new: 0.,
            m_scale_xy: 0.,
            m_znew_input: 0.,
            m_x_shift: 0.,
            m_z_shift: 0.,
            m_dy_avg: 0.,
            m_y_shift: 0.,
            m_iu_angle: None,
            m_iu_xtilt: None,
            m_iun_local: None,
            m_sol_file_fp: None,
            m_n_all_real_pt: 0,
            m_map_alf_start: 0,
            m_num_patch_x: 0,
            m_num_patch_y: 0,
            m_n_all_view: 0,
            m_n_all_proj_pt: 0,
            m_ipatch_x: 0,
            m_ipatch_y: 0,
            m_num_bot: 0,
            m_num_top: 0,
            m_ix_min: 0,
            m_ix_max: 0,
            m_iy_min: 0,
            m_iy_max: 0,
            m_num_proj_pt: 0,
            m_min_tilt_view: 0,
            m_ncomp_search: 0,
            m_map_tilt_start: 0,
            m_if_bt_search: 0,
            m_if_original_z: 0,
            m_xcen: 0.,
            m_ycen: 0.,
            m_dx_min: 0.,
            m_tilt_add: 0.,
            m_if_zfac: 0,
            m_if_do_robust: 0,
            m_did_robust: false,
            m_did_full_robust: false,
            m_min_res_robust: 0,
            m_min_local_res_robust: 0,
            m_pixel_delta: [0.; 3],
            m_pixel_size: 0.,
            m_errsum_local: 0.,
            m_err_local_min: 0.,
            m_err_local_max: 0.,
            m_rot_entered: 0.,
            m_bin_step_ini: 0.,
            m_bin_step_final: 0.,
            m_scan_step: 0.,
            m_if_do_local: 0,
            m_nin_thresh: 0,
            m_pipinput: 0,
            m_num_wgt_total: 0,
            m_num_wgt_zero: 0,
            m_num_wgt1: 0,
            m_num_wgt2: 0,
            m_num_wgt5: 0,
            m_max_del_wgt_below_crit: 0,
            m_max_robust_one_cycle: 0,
            m_robust_tot_cycle_fac: 0.,
            m_del_wgt_mean_crit: 0.,
            m_del_wgt_max_crit: 0.,
            m_robust_max_del_err: 0.,
            m_wgt_err_sum_local: 0.,
            m_wgt_err_local_min: 0.,
            m_wgt_err_local_max: 0.,
            m_undo_wgt_relax_crit: 0.,
            m_num_rob_failed: 0,
            m_num_fillin_pts: 0,
            m_local_mu_ratio_sum: 0.,
            m_min_local_mu_ratio: 0.,
            m_max_local_mu_ratio: 0.,
            m_lv_out_num_predict: 0,
            m_lv_out_num_pad: 0,
            m_frac_leave_out: 0.,
            m_frac_predict: 0.,
            m_num_train_real: 0,
            m_lv_out_num_runs: 0,
            m_lv_out_coverage: 0.,
            m_lv_out_target_cover: 0.,
            m_lv_out_min_coverage: 0.,
            m_lv_out_max_coverage: 0.,
            m_weight_none_left_out: Vec::new(),
            m_test_set_frac_or_step: 0.,
            m_all_real_in_test_set: Vec::new(),
            m_var_all_points: Vec::new(),
            m_lv_out_save_all_xyz: Vec::new(),
            m_fit_num_patches_in_fid_area: false,
        };
        ta.m_dtor = RADIANS_PER_DEGREE as f32;
        ta.m_max_cycles = 1000;
        ta.m_fac_metro = 0.25;
        ta.m_num_local_res = 50;
        ta.m_too_few_fid = false;
        ta.m_min_res_robust = 100;
        ta.m_min_local_res_robust = 65;
        ta.m_del_wgt_mean_crit = 0.001;
        ta.m_del_wgt_max_crit = 0.01;
        ta.m_max_del_wgt_below_crit = 4;
        ta.m_max_robust_one_cycle = 10;
        ta.m_robust_tot_cycle_fac = 3.;
        ta.m_if_do_robust = 0;
        ta.m_dx_min = 0.;
        ta.m_dy_avg = 0.;
        ta.m_if_zfac = 0;
        ta.m_if_original_z = 0;
        ta.m_if_do_local = 0;
        ta.m_bin_step_ini = 1.;
        ta.m_bin_step_final = 0.25;
        ta.m_scan_step = 0.02;
        ta.m_nin_thresh = 3;
        ta.m_undo_wgt_relax_crit = 2.;
        ta.m_robust_max_del_err = 4.;
        ta.m_pixel_size = 1.;
        ta.m_fixed_xyz_file = None;
        ta.m_iu_xtilt = None;
        ta.m_lv_out_num_predict = 3;
        ta.m_lv_out_num_pad = 1;
        ta.m_frac_leave_out = -1.;
        ta.m_lv_out_target_cover = 12000.;
        ta.m_lv_out_min_coverage = 0.3;
        ta.m_lv_out_max_coverage = 5.;
        ta.m_all_real_in_test_set = Vec::new();
        ta.m_lv_out_save_all_xyz = Vec::new();
        ta.m_min_sampled_in_view = 20;
        ta.m_num_sample_target = 100;
        ta.m_pix_units = "pixels".to_string();
        ta.m_rob_fail_mess[ROB_MESS_SIZE] = 0x00;
        ta
    }

    /// Original: `TiltAlign::main` (`tiltalign.cpp:95`).
    pub fn main(&mut self, argv: &[String]) {
        let mut max_var: i32;
        let maxtemp: i32;
        let mut sprod: Vec<f64>;
        let mut error: f64 = 0.;
        let mut error_scan = [0f64; 40];
        let mut model_file = String::new();
        let mut point_file: Option<String> = None;
        let mut filled_in_file: Option<String>;
        let mut temp_name: Vec<u8> = Vec::new();
        let mut dir_done = [false; 3];
        let mut warn_on_rob_fail: i32;
        let mut created_day: i32 = 0;
        let mut iv: i32;
        let mut index: i32;
        let mut i: i32;
        let mut xtmp: f32 = 0.;
        let mut ytmp: f32;
        let mut ztmp: f32;
        let mut x_origin: f32 = 0.;
        let mut y_origin: f32 = 0.;
        let mut z_origin: f32 = 0.;
        let mut j: i32;
        let mut num_train_proj: i32;
        let lv_out_max_real_pt: i32;
        let lv_out_max_proj_pt: i32;
        // `int crossValOption` (`:109`) is never initialised (module doc).
        let mut cross_val_option: i32 = 0;
        let mut tilt_new: f32 = 0.;
        let mut tilt_max: f32;
        let mut fixed_max: f32;
        let itmp: i32;
        let mut ierr: i32 = 0;
        let mut min_init_error: i32 = 0;
        let rot_scan_err_crit: f32;
        let transpose_xy_adj: f32;
        let rot_inc_for_init: f32;
        let mut rand_thresh: f32;
        let mut xx_use: Vec<f32> = Vec::new();
        let mut yy_use: Vec<f32> = Vec::new();
        let mut real_str_use: Vec<i32> = Vec::new();
        let mut sec_view_use: Vec<i32> = Vec::new();
        let mut used_full_to_real: Vec<i32> = Vec::new();
        let mut ind_used_to_real: Vec<i32> = Vec::new();
        let mut did_full_tracks = false;
        let mut image_binned: i32;
        let num_init_steps: i32;
        let min_local_track_res_rob: i32;
        let mut rand_seed: i32;
        let mut num_wanted: i32;
        let mut num_needed: i32;
        // `nrealUse` is only set by `makeFullTracksForInit` (module doc).
        let mut nreal_use: i32;
        let min_track_res_rob: i32;
        let mut line_for_read = vec![0u8; 120];
        let progname = imod_prog_name(argv.first().map_or("", String::as_str));
        //
        let mut num_opt_arg: i32 = 0;
        let mut num_non_opt_arg: i32 = 0;
        //
        // fallbacks from ../../manpages/autodoc2man 2 2  tiltalign
        //
        let num_options = 125;
        let options: [&[u8]; 125] = [
            b":ModelFile:FN:",
            b":ImageFile:FN:",
            b":ImageSizeXandY:IP:",
            b":ImageOriginXandY:FP:",
            b":ImagePixelSizeXandY:FP:",
            b":UnbinnedPixelSize:F:",
            b":ImagesAreBinned:I:",
            b":OutputModelFile:FN:",
            b":OutputResidualFile:FN:",
            b":OutputModelAndResidual:FN:",
            b":OutputFilledInModel:FN:",
            b":OutputTopBotResiduals:FN:",
            b":OutputFidXYZFile:FN:",
            b":FixedXYZInputFile:FN:",
            b":OutputTiltFile:FN:",
            b":OutputUnadjustedTiltFile:FN:",
            b":OutputXAxisTiltFile:FN:",
            b":OutputTransformFile:FN:",
            b":OutputZFactorFile:FN:",
            b":IncludeStartEndInc:IT:",
            b":IncludeList:LI:",
            b":ExcludeList:LI:",
            b":RotationAngle:F:",
            b":SeparateGroup:LIM:",
            b":NoSeparateTiltGroups:I:",
            b"first:FirstTiltAngle:F:",
            b"increment:TiltIncrement:F:",
            b"tiltfile:TiltFile:FN:",
            b"angles:TiltAngles:FAM:",
            b":AngleOffset:F:",
            b":ProjectionStretch:B:",
            b":BeamTiltOption:I:",
            b":FixedOrInitialBeamTilt:F:",
            b":RotOption:I:",
            b":RotDefaultGrouping:I:",
            b":RotNondefaultGroup:ITM:",
            b":RotationFixedView:I:",
            b":TiltOption:I:",
            b":TiltFixedView:I:",
            b":TiltSecondFixedView:I:",
            b":TiltDefaultGrouping:I:",
            b":TiltNondefaultGroup:ITM:",
            b":MagReferenceView:I:",
            b":MagOption:I:",
            b":MagDefaultGrouping:I:",
            b":MagNondefaultGroup:ITM:",
            b":CompReferenceView:I:",
            b":CompOption:I:",
            b":CompDefaultGrouping:I:",
            b":CompNondefaultGroup:ITM:",
            b":XStretchOption:I:",
            b":XStretchDefaultGrouping:I:",
            b":XStretchNondefaultGroup:ITM:",
            b":SkewOption:I:",
            b":SkewDefaultGrouping:I:",
            b":SkewNondefaultGroup:ITM:",
            b":XTiltOption:I:",
            b":XTiltDefaultGrouping:I:",
            b":XTiltNondefaultGroup:ITM:",
            b":ResidualReportCriterion:F:",
            b":SurfacesToAnalyze:I:",
            b":MetroFactor:F:",
            b":MaximumCycles:I:",
            b":AxisZShift:F:",
            b":ShiftZFromOriginal:B:",
            b":AxisXShift:F:",
            b":RobustFitting:B:",
            b":WeightWholeTracks:B:",
            b":KFactorScaling:F:",
            b":WarnOnRobustFailure:B:",
            b":MinWeightGroupSizes:IP:",
            b":CrossValidate:I:",
            b":FractionToLeaveOut:FP:",
            b":LeaveOutPredictAndPad:IP:",
            // Fixed in translation (2026-09-26, `BUGS.md`): the source's fallback
            // table types these two `B` (`tiltalign.cpp:156-157`) although they
            // are read as a float and a float pair, and lacks the two
            // extra-weight options `input_model` reads, so a run without an
            // autodoc exits "Illegal option: ObjectsWithExtraWeight".
            b":CVCoverageTargetOrFactor:F:",
            b":CVMinAndMaxCoverageFactor:FP:",
            b":ObjectsWithExtraWeight:LI:",
            b":ExtraWeights:FA:",
            b":RandomSeed:I:",
            b":TestSetIntervalOrFrac:F:",
            b":LocalAlignments:B:",
            b":OutputLocalFile:FN:",
            b":NumberOfLocalPatchesXandY:IP:",
            b":TargetPatchSizeXandY:IP:",
            b":MinSizeOrOverlapXandY:FP:",
            b":MinFidsTotalAndEachSurface:IP:",
            b":FixXYZCoordinates:B:",
            b":LocalOutputOptions:IT:",
            b":LocalRotOption:I:",
            b":LocalRotDefaultGrouping:I:",
            b":LocalRotNondefaultGroup:ITM:",
            b":LocalTiltOption:I:",
            b":LocalTiltFixedView:I:",
            b":LocalTiltSecondFixedView:I:",
            b":LocalTiltDefaultGrouping:I:",
            b":LocalTiltNondefaultGroup:ITM:",
            b":LocalMagReferenceView:I:",
            b":LocalMagOption:I:",
            b":LocalMagDefaultGrouping:I:",
            b":LocalMagNondefaultGroup:ITM:",
            b":LocalXStretchOption:I:",
            b":LocalXStretchDefaultGrouping:I:",
            b":LocalXStretchNondefaultGroup:ITM:",
            b":LocalSkewOption:I:",
            b":LocalSkewDefaultGrouping:I:",
            b":LocalSkewNondefaultGroup:ITM:",
            b":LocalXTiltOption:I:",
            b":LocalXTiltDefaultGrouping:I:",
            b":LocalXTiltNondefaultGroup:ITM:",
            b":RotMapping:IAM:",
            b":LocalRotMapping:IAM:",
            b":TiltMapping:IAM:",
            b":LocalTiltMapping:IAM:",
            b":MagMapping:IAM:",
            b":LocalMagMapping:IAM:",
            b":CompMapping:IAM:",
            b":XStretchMapping:IAM:",
            b":LocalXStretchMapping:IAM:",
            b":SkewMapping:IAM:",
            b":LocalSkewMapping:IAM:",
            b":XTiltMapping:IAM:",
            b":LocalXTiltMapping:IAM:",
            b":CreatedDayStamp:I:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
        ];
        //
        maxtemp = 400000;
        self.av.xyz_fixed = 0;
        self.av.kfac_robust = 4.685;
        min_track_res_rob = 30;
        min_local_track_res_rob = 20;
        self.av.small_wgt_threshold = 0.5;
        self.av.incr_gmag = 0;
        self.av.incr_dmag = 0;
        self.av.incr_skew = 0;
        self.av.incr_rot = 0;
        self.av.incr_tilt = 0;
        self.av.incr_alf = 0;
        self.av.test_set_frac_step = 0.;
        image_binned = 1;
        rot_inc_for_init = 10.;
        rot_scan_err_crit = 5.;
        self.av.small_wgt_max_frac = 0.25;
        warn_on_rob_fail = 0;
        rand_seed = 1234567;
        self.av.real_in_test_set = Vec::new();
        self.av.apply_extra_weights = 0;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        pip_allow_comma_defaults(1);
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            progname.as_bytes(),
            3,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
            Some(imod_usage_header_for_pip),
        );
        self.m_pipinput = (num_opt_arg + num_non_opt_arg > 0) as i32;
        if num_opt_arg + num_non_opt_arg == 0 {
            error_exit::<false>(
                "Sequential interactive input is no longer supported; use Etomo to convert a com file to PIP input",
                0,
            );
        }
        //
        // Do temporary allocations for imodObj, imodCont, listz needed here
        self.m_iall_real_str = vec![0; maxtemp as usize];
        self.m_ind_all_real = vec![0; maxtemp as usize];
        self.m_list_real = vec![0; maxtemp as usize];
        memory_error(true, "temporary oversized arrays");
        //
        pip_get_integer(b"CreatedDayStamp", &mut created_day);
        self.m_fit_num_patches_in_fid_area = created_day >= 1658;
        pip_get_integer(b"ImagesAreBinned", &mut image_binned);
        image_binned = if 1 > image_binned { 1 } else { image_binned };
        input_model(
            &mut self.av,
            &mut self.mx,
            &mut self.m_eval_funct,
            &mut self.fm,
            &mut self.m_iall_real_str,
            &mut self.m_ind_all_real,
            &mut self.m_num_proj_pt,
            &mut self.m_xcen,
            &mut self.m_ycen,
            &mut self.m_pixel_delta,
            &mut self.m_list_real,
            maxtemp,
            &mut model_file,
            &mut self.m_residual_file,
            &mut point_file,
            &mut self.m_iu_angle,
            &mut self.m_iu_xtilt,
            &mut self.m_sol_file_fp,
            &mut x_origin,
            &mut y_origin,
            &mut z_origin,
            image_binned,
        );
        //
        // Get things into the actual arrays now
        self.m_imod_obj = vec![0; self.mx.max_real as usize];
        self.m_imod_cont = vec![0; self.mx.max_real as usize];
        memory_error(true, "arrays for object/contour #s");
        copy_array(
            &mut self.m_imod_obj,
            1,
            self.av.nreal_pt,
            &self.m_iall_real_str,
            1,
        );
        copy_array(
            &mut self.m_imod_cont,
            1,
            self.av.nreal_pt,
            &self.m_ind_all_real,
            1,
        );
        copy_array(
            &mut self.av.map_view_to_file,
            1,
            self.av.nview,
            &self.m_list_real,
            1,
        );
        self.m_iall_real_str = Vec::new();
        self.m_ind_all_real = Vec::new();
        self.m_list_real = Vec::new();
        //
        // Do big allocations
        let max_real = self.mx.max_real as usize;
        let max_view = self.mx.max_view as usize;
        let max_proj_pt = self.mx.max_proj_pt as usize;
        let nfile_views = self.av.nfile_views as usize;
        self.m_nin_real = vec![0; max_real];
        self.m_igroup = vec![0; max_real];
        self.m_tilt_orig = vec![0.; nfile_views];
        self.m_view_res = vec![0.; max_view];
        self.m_xyz_err = vec![0.; max_real * 3];
        self.m_num_in_view = vec![0; max_view];
        self.m_ind_save = vec![0; max_proj_pt];
        self.m_jpt_save = vec![0; max_proj_pt];
        self.m_err_save = vec![0.; max_proj_pt];
        self.m_view_errsum = vec![0.; max_view];
        self.m_view_errsq = vec![0.; max_view];
        self.m_view_mean_res = vec![0.; max_view];
        self.m_view_sd_res = vec![0.; max_view];
        self.m_fl = vec![0.; 6 * max_view];
        self.m_xz_fac = vec![0.; max_view];
        self.m_yz_fac = vec![0.; max_view];
        self.m_all_xyz = vec![0.; 3 * max_real];
        self.m_all_xx = vec![0.; max_proj_pt];
        self.m_all_yy = vec![0.; max_proj_pt];
        self.m_fixed_xyz = vec![0.; 3 * max_real];
        self.m_glb_fl = vec![0.; 6 * max_view];
        self.m_glb_xz_fac = vec![0.; nfile_views];
        self.m_glb_yz_fac = vec![0.; nfile_views];
        self.m_iall_sec_vw = vec![0; max_proj_pt];
        self.m_iall_real_str = vec![0; max_real];
        self.m_list_real = vec![0; max_real];
        self.m_ind_all_real = vec![0; max_real];
        self.m_map_all_to_local = vec![0; max_view];
        self.m_map_local_to_all = vec![0; max_view];
        self.m_map_all_file_to_view = vec![0; nfile_views];
        self.m_map_all_view_to_file = vec![0; max_view];
        memory_error(true, "main program arrays");

        copy_array(
            &mut self.m_map_all_view_to_file,
            1,
            self.av.nview,
            &self.av.map_view_to_file,
            1,
        );
        //
        // Allocate the variable arrays to maximum plausible size
        max_var = 7 * self.mx.max_view + 3 * self.mx.max_real;
        self.m_var = vec![0.; max_var as usize];
        self.m_var_err = vec![0.; max_var as usize];
        self.m_grad = vec![0.; max_var as usize];
        self.m_var_name = vec![0u8; 8 * 7 * max_view];
        memory_error(true, "variable arrays");
        //
        // Copy variables for allocating leave-out stuff before modifying
        lv_out_max_proj_pt = self.m_num_proj_pt;
        lv_out_max_real_pt = self.av.nreal_pt;
        //
        // If it is a patch tracking model, convert to a subsample if too many tracks
        i = 1;
        while i <= self.av.nreal_pt {
            self.m_ind_all_real[(i - 1) as usize] = i;
            i += 1;
        }
        self.m_sub_sample_tracks = self.av.patch_track_model != 0
            && self.av.num_full_patch_tracks as f64 > 1.1 * self.m_num_sample_target as f64;
        if self.m_sub_sample_tracks {
            load_patch_subset(
                &mut self.av,
                &mut self.m_all_xx,
                &mut self.m_all_yy,
                &mut self.m_n_all_proj_pt,
                &mut self.m_num_proj_pt,
                &mut self.m_ind_all_real,
                &mut self.m_n_all_real_pt,
                &mut self.m_iall_real_str,
                &mut self.m_iall_sec_vw,
                &self.m_imod_obj,
                &mut self.m_num_in_view,
                self.m_num_sample_target,
                self.m_min_sampled_in_view,
            );
            self.m_ss_afac = vec![0.; max_view];
            self.m_ss_bfac = vec![0.; max_view];
            self.m_ss_cfac = vec![0.; max_view];
            self.m_ss_dfac = vec![0.; max_view];
            self.m_ss_efac = vec![0.; max_view];
            self.m_ss_ffac = vec![0.; max_view];
            memory_error(true, "subsample factor arrays");
        }

        i = 1;
        while i <= self.av.nreal_pt {
            self.m_list_real[(i - 1) as usize] = i;
            self.m_nin_real[(i - 1) as usize] =
                self.av.ireal_str[i as usize] - self.av.ireal_str[(i - 1) as usize];
            i += 1;
        }
        count_num_in_view(
            &self.av,
            &self.m_list_real,
            self.av.nreal_pt,
            &self.av.ireal_str,
            &self.av.isec_view,
            self.av.nview,
            &mut self.m_num_in_view,
            None,
            None,
        );
        input_vars(
            &mut self.av,
            &self.mx,
            &mut self.sg,
            &mut self.m_var,
            &mut self.m_var_name,
            &mut self.m_nvar_search,
            &mut self.m_nvar_angle,
            &mut self.m_nvar_scaled,
            &mut self.m_min_tilt_view,
            &mut self.m_ncomp_search,
            0,
            &mut self.m_map_tilt_start,
            &mut self.m_map_alf_start,
            &mut self.m_map_alf_end,
            &mut self.m_if_bt_search,
            &mut self.m_tilt_orig,
            &mut self.m_tilt_add,
            &self.m_num_in_view,
            self.m_nin_thresh,
            &mut self.m_rot_entered,
        );
        //
        // Adjust the entered rotation angle to be within +/-range and then set an adjustment
        // variable if X and Y will probably be transposed in aligned stack
        while b3dabs!(self.m_rot_entered) as f64 > 200. {
            self.m_rot_entered = (self.m_rot_entered as f64
                - if self.m_rot_entered < 0. { -360. } else { 360. })
                as f32;
        }
        let mut transpose_adj: f32 = 0.;
        if b3dabs!(self.m_rot_entered) as f64 > 45. && (b3dabs!(self.m_rot_entered) as f64) < 135. {
            transpose_adj = self.m_ycen - self.m_xcen;
        }
        transpose_xy_adj = transpose_adj;
        //
        filled_in_file = None;
        self.m_err_crit = 3.0;
        self.m_num_surface = 0;
        self.m_znew_input = 0.;
        self.m_xtilt_new = 0.;
        if pip_get_string(b"OutputFilledInModel", &mut temp_name) == 0 {
            filled_in_file = Some(String::from_utf8_lossy(&temp_name).into_owned());
        }
        pip_get_float(b"ResidualReportCriterion", &mut self.m_err_crit);
        pip_get_integer(b"SurfacesToAnalyze", &mut self.m_num_surface);
        pip_get_float(b"MetroFactor", &mut self.m_fac_metro);
        pip_get_integer(b"MaximumCycles", &mut self.m_max_cycles);
        pip_get_boolean(b"RobustFitting", &mut self.m_if_do_robust);
        pip_get_boolean(b"WeightWholeTracks", &mut self.av.robust_by_track);
        if self.m_if_do_robust > 0 && self.av.robust_by_track != 0 && self.av.patch_track_model == 0
        {
            // Fixed in translation (`BUGS.md`): the source's message has no newline.
            printf!("Option to weight whole contours is being ignored for non-patch track model\n");
            self.av.robust_by_track = 0;
        }
        if pip_get_two_integers(
            b"MinWeightGroupSizes",
            &mut self.m_min_res_robust,
            &mut self.m_min_local_res_robust,
        ) != 0
        {
            // They were loaded with the point residual values, so change if track residuals
            if self.av.patch_track_model != 0 && self.av.robust_by_track != 0 {
                self.m_min_res_robust = min_track_res_rob;
                self.m_min_local_res_robust = min_local_track_res_rob;
            }
        }

        if pip_get_float(b"KFactorScaling", &mut xtmp) == 0 {
            self.av.kfac_robust *= xtmp;
        }
        pip_get_boolean(b"WarnOnRobustFailure", &mut warn_on_rob_fail);

        // Cross-validation: get the relevant options unless skipping
        pip_get_integer(b"CrossValidate", &mut cross_val_option);
        if cross_val_option != 0 && std::env::var_os("TILTALIGN_SKIP_CROSS_VAL").is_some() {
            printf!("Skipping cross-validation: TILTALIGN_SKIP_CROSS_VAL is set in environment\n");
            cross_val_option = 0;
        }

        if cross_val_option != 0 {
            pip_get_two_integers(
                b"LeaveOutPredictAndPad",
                &mut self.m_lv_out_num_predict,
                &mut self.m_lv_out_num_pad,
            );
            self.m_lv_out_num_pad = b3dabs!(self.m_lv_out_num_pad);
            if pip_get_float(b"FractionToLeaveOut", &mut self.m_frac_leave_out) == 0 {
                // B3DCLAMP(mFracLeaveOut, .01, 0.2): MAX(.01, MIN(0.2, val)) in double
                let inner: f64 = if 0.2 < self.m_frac_leave_out as f64 {
                    0.2
                } else {
                    self.m_frac_leave_out as f64
                };
                self.m_frac_leave_out = (if 0.01 > inner { 0.01 } else { inner }) as f32;
            }
            pip_get_float(b"CVCoverageTargetOrFactor", &mut self.m_lv_out_target_cover);
            pip_get_two_floats(
                b"CVMinAndMaxCoverageFactor",
                &mut self.m_lv_out_min_coverage,
                &mut self.m_lv_out_max_coverage,
            );
            pip_get_integer(b"RandomSeed", &mut rand_seed);
            if pip_get_float(b"TestSetIntervalOrFrac", &mut self.av.test_set_frac_step) == 0
                && self.m_sub_sample_tracks
            {
                error_exit::<false>(
                    "You cannot evaluate a test set when subsampling a patch tracking model",
                    0,
                );
            }
            if self.av.test_set_frac_step as f64 > 0.75 && (self.av.test_set_frac_step as f64) < 1.9
            {
                error_exit::<false>(
                    "A test set fraction should be 0.75 or less, an interval should be 2 or more",
                    0,
                );
            }
        } else {
            self.m_lv_out_num_predict = -1;
        }

        pip_get_float(b"AxisZShift", &mut self.m_znew_input);
        pip_get_boolean(b"ShiftZFromOriginal", &mut self.m_if_original_z);
        pip_get_float(b"AxisXShift", &mut self.m_xtilt_new);
        pip_get_boolean(b"LocalAlignments", &mut self.m_if_do_local);
        if self.m_if_do_robust > 0 {
            self.m_wgt_prev = vec![0.; max_proj_pt];
            self.m_var_save = vec![0.; max_var as usize];
            memory_error(true, "arrays for robust fitting");
        }
        if pip_get_float(b"UnbinnedPixelSize", &mut self.m_pixel_size) == 0 {
            self.m_pixel_size *= image_binned as f32;
            self.m_pix_units = "nm".to_string();
        }
        //
        // See if there an input file for fixed XYZ coordinate in local alignments
        // read it in, insist on exact number of points, and adjust if 90 deg rotation
        temp_name.clear();
        ierr = pip_get_string(b"FixedXYZInputFile", &mut temp_name);
        if ierr == 0 {
            let fixed_name = String::from_utf8_lossy(&temp_name).into_owned();
            self.m_fixed_xyz_file = Some(fixed_name.clone());
            let mut fixed_fp = b3d_open_file(&fixed_name, "r");
            index = 0;
            ierr = read_lines_for_values(
                &mut fixed_fp,
                &mut index,
                self.av.nreal_pt * 3,
                &mut line_for_read,
                120,
                0,
                "f",
                &mut [ReadValueArray::Floats(&mut self.m_fixed_xyz)],
            );
            if ierr == -3 {
                error_exit::<false>(
                    "More coordinates in fixed XYZ input file than fiducial points",
                    0,
                );
            }
            if ierr != 0 {
                exit_from_value_read_error(ierr, "fixed XYZ input file");
            }
            // Fixed in translation (2026-09-26, `BUGS.md`): `index` counts values,
            // three per point, and the source tests it against `nrealPt`
            // (`tiltalign.cpp:409`), accepting a file with 1 to 3 times fewer
            // values than needed and reading unset coordinates.
            if index < 3 * self.av.nreal_pt {
                error_exit::<false>(
                    "Fewer coordinates in fixed XYZ input file than fiducial points",
                    0,
                );
            }
            drop(fixed_fp);
            index = 0;
            while index < self.av.nreal_pt {
                self.m_fixed_xyz[(index * 3) as usize] -= transpose_xy_adj;
                self.m_fixed_xyz[(index * 3 + 1) as usize] += transpose_xy_adj;
                index += 1;
            }
        }
        //
        if b3dnint!(self.m_znew_input) != 1000 {
            self.m_znew_input /= image_binned as f32;
        }
        self.m_xtilt_new /= image_binned as f32;
        self.m_order_err = true;
        self.m_nearby_err = self.m_err_crit < 0.;
        self.m_err_crit = b3dabs!(self.m_err_crit);
        num_train_proj = self.m_num_proj_pt;
        self.m_num_train_real = self.av.nreal_pt;
        //
        // Allocate for leave-out
        if self.m_lv_out_num_predict >= 0 {
            self.m_weight_none_left_out = vec![0.; lv_out_max_proj_pt as usize];
            self.av.proj_left_out = vec![0; lv_out_max_proj_pt as usize];
            self.av.proj_to_predict = vec![0; lv_out_max_proj_pt as usize];
            self.av.real_left_out = vec![0; lv_out_max_real_pt as usize];
            self.av.times_left_out = vec![0; lv_out_max_real_pt as usize];
            self.m_var_all_points = vec![0.; max_var as usize];
            self.av.real_in_test_set = vec![0; lv_out_max_real_pt as usize];
            self.m_all_real_in_test_set = vec![0; lv_out_max_real_pt as usize];
            self.m_lv_out_save_all_xyz = vec![0.; 3 * lv_out_max_real_pt as usize];
            memory_error(true, "arrays for leave-out tests");

            // Set up random seed
            if rand_seed == 0 {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs() as i64);
                rand_seed = (now & 0xFFFFFF) as i32;
            }
            b3dsrand(&rand_seed);

            // Clear these out
            for k in 0..lv_out_max_real_pt as usize {
                self.av.real_in_test_set[k] = 0;
                self.m_all_real_in_test_set[k] = 0;
            }

            // Set up test set to hold in reserve as fixed steps or random selection
            if self.av.test_set_frac_step > 0. {
                if self.av.test_set_frac_step > 1. {
                    ierr = b3dnint!(self.av.test_set_frac_step);
                    i = ierr / 2;
                    while i < self.av.nreal_pt {
                        self.av.real_in_test_set[i as usize] = 1;
                        self.m_all_real_in_test_set[i as usize] = 1;
                        self.m_num_train_real -= 1;
                        num_train_proj -=
                            self.av.ireal_str[(i + 1) as usize] - self.av.ireal_str[i as usize];
                        i += ierr;
                    }
                } else {
                    // Adjust the threshold for selection depending on how many are still needed
                    num_wanted = b3dnint!(self.av.test_set_frac_step * self.av.nreal_pt as f32);
                    rand_thresh = self.av.test_set_frac_step;
                    i = 0;
                    while i < self.av.nreal_pt {
                        num_needed = num_wanted - (self.av.nreal_pt - self.m_num_train_real);
                        if num_needed > 0 {
                            rand_thresh = num_needed as f32 / (self.av.nreal_pt - i) as f32;
                        }
                        if b3drand() < rand_thresh {
                            self.av.real_in_test_set[i as usize] = 1;
                            self.m_all_real_in_test_set[i as usize] = 1;
                            self.m_num_train_real -= 1;
                            num_train_proj -=
                                self.av.ireal_str[(i + 1) as usize] - self.av.ireal_str[i as usize];
                        }
                        i += 1;
                    }
                }
                printf!(
                    "Using %d points for training and %d for a test set\n",
                    CArg::Int(self.m_num_train_real as i64),
                    CArg::Int((self.av.nreal_pt - self.m_num_train_real) as i64)
                );
            }

            // Initialize the weights here!
            i = 0;
            while i < self.av.nreal_pt {
                self.av.real_left_out[i as usize] = self.av.real_in_test_set[i as usize];
                j = self.av.ireal_str[i as usize] - 1;
                while j < self.av.ireal_str[(i + 1) as usize] - 1 {
                    self.av.weight[j as usize] = if self.av.real_in_test_set[i as usize] != 0 {
                        0.
                    } else {
                        1.
                    };
                    j += 1;
                }
                i += 1;
            }

            if self.m_frac_leave_out < 0. {
                self.m_frac_leave_out = (0.1 * self.m_num_train_real as f64) as f32;
                // B3DCLAMP(mFracLeaveOut, 0.04, 0.1)
                let inner: f64 = if 0.1 < self.m_frac_leave_out as f64 {
                    0.1
                } else {
                    self.m_frac_leave_out as f64
                };
                self.m_frac_leave_out = (if 0.04 > inner { 0.04 } else { inner }) as f32;
            }
            self.m_frac_predict = self.m_frac_leave_out;
            if self.m_lv_out_num_predict > 0 {
                self.m_frac_predict = (self.m_frac_leave_out * self.m_lv_out_num_predict as f32)
                    / (self.m_lv_out_num_predict + 2 * self.m_lv_out_num_pad) as f32;
            }
        }
        //
        // Count missing points if filling in; allocate arrays, initialize
        // Just skip it if this is patch tracking model: the allocation is wrong if a subset
        // is being used
        self.m_make_filled_fid = if filled_in_file.is_some() && self.av.patch_track_model == 0 {
            1
        } else {
            0
        };
        if self.m_make_filled_fid != 0 {
            self.m_num_fillin_pts = 0;
            i = 1;
            while i <= self.av.nreal_pt {
                iv = 1;
                while iv <= self.av.nview {
                    let start = (self.av.ireal_str[(i - 1) as usize] - 1) as usize;
                    if number_in_list(
                        iv,
                        Some(&self.av.isec_view[start..]),
                        self.av.ireal_str[i as usize] - self.av.ireal_str[(i - 1) as usize],
                        0,
                    ) == 0
                    {
                        self.m_num_fillin_pts += 1;
                    }
                    iv += 1;
                }
                i += 1;
            }
            iv = self.m_num_fillin_pts + 4;
            self.av.ifill_view = vec![0; iv as usize];
            self.av.xfill_proj = vec![0.; iv as usize];
            self.av.yfill_proj = vec![0.; iv as usize];
            self.av.ifill_real_start = vec![0; (self.av.nreal_pt + 4) as usize];
            self.m_xfill_final = vec![0.; iv as usize];
            self.m_yfill_final = vec![0.; iv as usize];
            self.m_num_in_fill_sum = vec![0; iv as usize];
            self.m_ifill_all_real_str = vec![0; (self.av.nreal_pt + 4) as usize];
            self.m_iall_fill_view = vec![0; iv as usize];
            self.m_map_fill_local_to_all = vec![0; iv as usize];
            memory_error(true, "arrays for filling in points");
            self.m_num_fillin_pts = 0;
            i = 1;
            while i <= self.av.nreal_pt {
                self.m_ifill_all_real_str[(i - 1) as usize] = self.m_num_fillin_pts + 1;
                iv = 1;
                while iv <= self.av.nview {
                    let start = (self.av.ireal_str[(i - 1) as usize] - 1) as usize;
                    if number_in_list(
                        iv,
                        Some(&self.av.isec_view[start..]),
                        self.av.ireal_str[i as usize] - self.av.ireal_str[(i - 1) as usize],
                        0,
                    ) == 0
                    {
                        self.m_num_fillin_pts += 1;
                        self.m_iall_fill_view[(self.m_num_fillin_pts - 1) as usize] = iv;
                        self.m_map_fill_local_to_all[(self.m_num_fillin_pts - 1) as usize] =
                            self.m_num_fillin_pts;
                    }
                    iv += 1;
                }
                i += 1;
            }
            self.m_ifill_all_real_str[self.av.nreal_pt as usize] = self.m_num_fillin_pts + 1;
            for k in 0..self.m_num_fillin_pts as usize {
                self.m_xfill_final[k] = 0.;
                self.m_yfill_final[k] = 0.;
                self.m_num_in_fill_sum[k] = 0;
            }
            copy_array(
                &mut self.av.ifill_real_start,
                1,
                self.av.nreal_pt + 1,
                &self.m_ifill_all_real_str,
                1,
            );
            copy_array(
                &mut self.av.ifill_view,
                1,
                self.m_num_fillin_pts,
                &self.m_iall_fill_view,
                1,
            );
        }
        //
        // scale the points down to range of 1.0: helps convergence
        //
        self.m_if_var_out = 1;
        self.m_if_res_out = 1;
        self.m_if_xyz_out = 1;
        self.m_if_local = 0;
        self.m_metro_error = 0;
        self.m_num_rob_failed = 0;
        //
        self.m_scale_xy = 0.;
        i = 1;
        while i <= self.m_num_proj_pt {
            let ax = b3dabs!(self.av.xx[(i - 1) as usize]);
            self.m_scale_xy = if self.m_scale_xy > ax {
                self.m_scale_xy
            } else {
                ax
            };
            let ay = b3dabs!(self.av.yy[(i - 1) as usize]);
            self.m_scale_xy = if self.m_scale_xy > ay {
                self.m_scale_xy
            } else {
                ay
            };
            i += 1;
        }
        i = 1;
        while i <= self.m_num_proj_pt {
            self.av.xx[(i - 1) as usize] /= self.m_scale_xy;
            self.av.yy[(i - 1) as usize] /= self.m_scale_xy;
            i += 1;
        }
        //
        // call new solveXyzd to get initial values of x, y, z
        // Allocate a double array because of potential alignment issues using real array as
        // double in a subroutine, deallocate when done
        // TODO pow
        itmp = 3 * if self.mx.max_real < MAX_REAL_FOR_DIRECT_INIT {
            self.mx.max_real
        } else {
            MAX_REAL_FOR_DIRECT_INIT
        };
        sprod = vec![0f64; ((itmp * itmp) / 2) as usize];
        memory_error(true, "array for solvexyzd");

        self.m_eval_funct
            .remap_params(&mut self.av, &mut self.m_var);

        nreal_use = self.av.nreal_pt;
        //wallStart = wallTime();
        if self.av.nreal_pt > 1 {
            num_init_steps = (90. / rot_inc_for_init as f64).ceil() as i32;
            dir_done[0] = false;
            dir_done[1] = false;
            dir_done[2] = false;
            did_full_tracks = make_full_tracks_for_init(
                &self.av,
                &mut xx_use,
                &mut yy_use,
                &mut sec_view_use,
                &mut real_str_use,
                &mut nreal_use,
                &self.m_ind_all_real,
                self.m_sub_sample_tracks,
                &mut used_full_to_real,
                &mut ind_used_to_real,
            );
            //
            // Scan around the given initial rotation, going in each direction until an
            // error is reached that is a criterion ratio bigger than the current minimum
            let mut itry = 0;
            'try_loop: while itry <= num_init_steps {
                iv = -1;
                while iv <= 1 {
                    index = iv * itry;
                    if itry > 0 && !dir_done[(iv + 1) as usize] {
                        if error_scan[(20 + iv * (itry - 1)) as usize]
                            > rot_scan_err_crit as f64 * error_scan[(20 + min_init_error) as usize]
                        {
                            dir_done[(iv + 1) as usize] = true;
                        }
                    }
                    if dir_done[(iv + 1) as usize] || index == num_init_steps {
                        iv += 2;
                        continue;
                    }
                    {
                        let (xx_u, yy_u, sv_u, rs_u): (&[f32], &[f32], &[i32], &[i32]) =
                            if did_full_tracks {
                                (&xx_use, &yy_use, &sec_view_use, &real_str_use)
                            } else {
                                (
                                    &self.av.xx,
                                    &self.av.yy,
                                    &self.av.isec_view,
                                    &self.av.ireal_str,
                                )
                            };
                        solve_xyzd::<false>(
                            xx_u,
                            yy_u,
                            sv_u,
                            rs_u,
                            self.av.nview,
                            nreal_use,
                            &self.av.tilt,
                            &self.av.rot,
                            &self.av.gmag,
                            &self.av.comp,
                            &mut self.av.xyz,
                            &mut self.av.dxy,
                            self.m_dtor * rot_inc_for_init * index as f32,
                            &mut sprod,
                            &mut error,
                            &mut ierr,
                        );
                    }
                    /* printf("%.1f %4d%12.6f\n", rotIncForInit * index, ierr, error);
                    ... */
                    if itry == 0 {
                        if ierr != 0 {
                            error_exit::<false>(
                                "Solving for initial values of X/Y/Z coordinates",
                                0,
                            );
                        }
                        min_init_error = 0;
                        error_scan[20] = error;
                        itry += 1;
                        continue 'try_loop;
                    }
                    if ierr != 0 {
                        error_scan[(20 + index) as usize] = 1.1
                            * rot_scan_err_crit as f64
                            * error_scan[(20 + min_init_error) as usize];
                    } else {
                        error_scan[(20 + index) as usize] = error;
                    }
                    if error_scan[(20 + index) as usize]
                        < error_scan[(20 + min_init_error) as usize]
                    {
                        min_init_error = index;
                    }
                    iv += 2;
                }
                itry += 1;
            }
            //
            // Give warning if minimum is far from expected
            if b3dabs!(min_init_error as f32 * rot_inc_for_init) as f64 > 15. {
                printf!(
                    "\nWARNING: Based on initial fitting errors, the rotation angle seems to be closer to %6.1f than to the specified angle, %6.1f\nWARNING: An incorrect rotation angle will throw off prealignment, beadtracking, and this alignment\n\n",
                    CArg::Dbl(
                        (self.m_rot_entered + min_init_error as f32 * rot_inc_for_init) as f64
                    ),
                    CArg::Dbl(self.m_rot_entered as f64)
                );
            }
        }
        //
        // Redo initialization at 0 increment
        xtmp = 0.;
        let _ = xtmp;
        {
            let (xx_u, yy_u, sv_u, rs_u): (&[f32], &[f32], &[i32], &[i32]) = if did_full_tracks {
                (&xx_use, &yy_use, &sec_view_use, &real_str_use)
            } else {
                (
                    &self.av.xx,
                    &self.av.yy,
                    &self.av.isec_view,
                    &self.av.ireal_str,
                )
            };
            solve_xyzd::<false>(
                xx_u,
                yy_u,
                sv_u,
                rs_u,
                self.av.nview,
                nreal_use,
                &self.av.tilt,
                &self.av.rot,
                &self.av.gmag,
                &self.av.comp,
                &mut self.av.xyz,
                &mut self.av.dxy,
                0.,
                &mut sprod,
                &mut error,
                &mut ierr,
            );
        }
        if did_full_tracks {
            copy_xyz_from_full_tracks_to_real(
                &mut self.av,
                nreal_use,
                std::mem::take(&mut used_full_to_real),
                std::mem::take(&mut ind_used_to_real),
            );
            drop(xx_use);
            drop(yy_use);
            drop(real_str_use);
            drop(sec_view_use);
        }
        //write(*,'(a,f8.2)') 'Initialization time', 1000 * (wallTime() - wallStart)
        //call flush(6)
        let _ = ImodFile::Stdout.flush();
        // write(*, '(3(2f9.4,f8.4))') ((xyz(i, iv), i = 1, 3), iv = 1, nrealPt)
        drop(sprod);
        //
        // Get the h array big enough for the solution (and temp use later)
        max_var = self.m_nvar_search + 3 * self.av.nreal_pt;
        self.m_max_h = if 2 * self.av.nfile_views > (max_var + 3) * max_var {
            2 * self.av.nfile_views
        } else {
            (max_var + 3) * max_var
        };
        self.m_h = vec![0.; self.m_max_h as usize];
        memory_error(true, "array for h matrix");
        //
        clear_leave_out_errors(&mut self.av);
        self.align_and_output_results();
        //
        // restore the data if a subsample of patches was tracked
        if self.m_sub_sample_tracks {
            restore_from_patch_sample(
                &mut self.av,
                &self.m_all_xx,
                &self.m_all_yy,
                self.m_n_all_proj_pt,
                &mut self.m_num_proj_pt,
                &mut self.m_all_xyz,
                &mut self.m_ind_all_real,
                self.m_n_all_real_pt,
                &self.m_iall_real_str,
                &self.m_iall_sec_vw,
                &self.m_imod_obj,
                Some(&self.m_ss_afac),
                &self.m_ss_bfac,
                &self.m_ss_cfac,
                &self.m_ss_dfac,
                &self.m_ss_efac,
                &self.m_ss_ffac,
            );
        }
        //
        // shift the fiducials to real positions in X and Y for xyz output file
        // and for possible use with local alignments
        // Continue to use zero - centroid xyz for find_surfaces but output
        // a 3D model with real positions also
        //
        self.m_n_all_real_pt = self.av.nreal_pt;
        for k in 0..self.av.nreal_pt as usize {
            self.m_iall_real_str[k] = self.av.ireal_str[k];
            self.m_all_xyz[k * 3] = self.av.xyz[k * 3] - self.m_dx_min + self.m_xcen;
            self.m_all_xyz[k * 3 + 1] = self.av.xyz[k * 3 + 1] - self.m_dy_avg + self.m_ycen;
            self.m_all_xyz[k * 3 + 2] = self.av.xyz[k * 3 + 2] - self.m_znew;
        }
        self.m_iall_real_str[self.av.nreal_pt as usize] =
            self.av.ireal_str[self.av.nreal_pt as usize];

        // Put out point file
        // Adjust the X and Y coordinates by difference in centers if there is a 90-degree
        // rotation for aligned stack predicted, and adjust the Dim: that is put out too;
        // take the adjustment away afterwards.
        // Note that solvematch works without all this because it gets centered coordinates from
        // the given size; that is why size had to be changed too
        if let Some(point_name) = point_file.as_deref() {
            let mut tmp_fp = b3d_open_file(point_name, "w");
            j = 0;
            while j < self.av.nreal_pt {
                let ju = j as usize;
                fprintf!(
                    tmp_fp,
                    "%4d %9.2f %9.2f %9.2f %6d %4d",
                    CArg::Int((j + 1) as i64),
                    CArg::Dbl((self.m_all_xyz[ju * 3] + transpose_xy_adj) as f64),
                    CArg::Dbl((self.m_all_xyz[ju * 3 + 1] - transpose_xy_adj) as f64),
                    CArg::Dbl(self.m_all_xyz[ju * 3 + 2] as f64),
                    CArg::Int(self.m_imod_obj[ju] as i64),
                    CArg::Int(self.m_imod_cont[ju] as i64)
                );
                if j != 0 {
                    fprintf!(tmp_fp, "\n");
                } else {
                    fprintf!(
                        tmp_fp,
                        " Pix: %11.5f Dim: %5d %5d\n",
                        CArg::Dbl(self.m_pixel_delta[0] as f64),
                        CArg::Int(b3dnint!(2. * (self.m_xcen + transpose_xy_adj) as f64) as i64),
                        CArg::Int(b3dnint!(2. * (self.m_ycen - transpose_xy_adj) as f64) as i64)
                    );
                }
                j += 1;
            }
            drop(tmp_fp);
        }
        //
        // analyze for surfaces if desired.  Find the biggest tilt and the
        // biggest fixed tilt, get recommended new value for the biggest fixed
        // tilt if it is not too small
        //
        tilt_max = 0.;
        fixed_max = 0.;
        iv = 1;
        while iv <= self.av.nview {
            let t = self.av.tilt[(iv - 1) as usize];
            if b3dabs!(t) > b3dabs!(tilt_max) {
                tilt_max = t;
            }
            if self.av.map_tilt[(iv - 1) as usize] == 0 && b3dabs!(t) > b3dabs!(fixed_max) {
                fixed_max = t;
            }
            iv += 1;
        }
        if fixed_max >= 5. {
            tilt_max = fixed_max;
        }
        if self.m_num_surface > 0 {
            find_surfaces(
                &self.av.xyz,
                self.av.nreal_pt,
                self.m_num_surface,
                tilt_max,
                &mut tilt_new,
                &mut self.m_igroup,
                self.m_ncomp_search,
                self.m_tilt_add,
                self.m_znew,
                self.m_znew_input,
                image_binned,
            );
        }
        //
        // Write separate residual outputs now that surfaces are known
        //
        if self.m_pipinput != 0 && self.m_num_surface > 1 {
            temp_name.clear();
            if pip_get_string(b"OutputTopBotResiduals", &mut temp_name) == 0 {
                let temp_root = String::from_utf8_lossy(&temp_name).into_owned();
                self.m_residual_file = temp_root.clone();
                self.m_residual_file += ".botres";
                j = 1;
                while j <= 2 {
                    self.m_num_bot = 0;
                    let mut jpt = 1;
                    while jpt <= self.av.nreal_pt {
                        if self.m_igroup[(jpt - 1) as usize] == j {
                            self.m_num_bot += self.av.ireal_str[jpt as usize]
                                - self.av.ireal_str[(jpt - 1) as usize];
                        }
                        jpt += 1;
                    }
                    if self.m_num_bot > 0 {
                        let mut tmp_fp = b3d_open_file(&self.m_residual_file, "w");
                        fprintf!(tmp_fp, "%6d residuals\n", CArg::Int(self.m_num_bot as i64));
                        jpt = 1;
                        while jpt <= self.av.nreal_pt {
                            if self.m_igroup[(jpt - 1) as usize] == j {
                                i = self.av.ireal_str[(jpt - 1) as usize];
                                while i <= self.av.ireal_str[jpt as usize] - 1 {
                                    let iu = (i - 1) as usize;
                                    fprintf!(
                                        tmp_fp,
                                        "%10.2f %9.2f %4d %7.2f %7.2f\n",
                                        CArg::Dbl((self.av.xx[iu] + self.m_xcen) as f64),
                                        CArg::Dbl((self.av.yy[iu] + self.m_ycen) as f64),
                                        CArg::Int(
                                            (self.av.map_view_to_file
                                                [(self.av.isec_view[iu] - 1) as usize]
                                                - 1)
                                                as i64
                                        ),
                                        CArg::Dbl(self.av.xresid[iu] as f64),
                                        CArg::Dbl(self.av.yresid[iu] as f64)
                                    );
                                    i += 1;
                                }
                            }
                            jpt += 1;
                        }
                        drop(tmp_fp);
                    }
                    self.m_residual_file = temp_root.clone();
                    self.m_residual_file += ".topres";
                    j += 1;
                }
            }
        }

        // Do leave out tests for global solution if option is 1
        self.m_lv_out_coverage = self.m_lv_out_target_cover;
        if self.m_lv_out_target_cover >= 100. {
            self.m_lv_out_coverage = self.m_lv_out_target_cover / num_train_proj as f32;
            // B3DCLAMP(mLvOutCoverage, mLvOutMinCoverage, mLvOutMaxCoverage)
            let inner = if self.m_lv_out_max_coverage < self.m_lv_out_coverage {
                self.m_lv_out_max_coverage
            } else {
                self.m_lv_out_coverage
            };
            self.m_lv_out_coverage = if self.m_lv_out_min_coverage > inner {
                self.m_lv_out_min_coverage
            } else {
                inner
            };
        }
        {
            let runs: f64 = (self.m_lv_out_coverage / self.m_frac_predict) as f64 + 0.5;
            self.m_lv_out_num_runs = (if 1. > runs { 1. } else { runs }) as i32;
        }
        self.m_did_full_robust = self.m_did_robust;

        // Reset the random seed and restore the array after the global runs so the same
        // results are obtained for locals if this is skipped
        if cross_val_option < 2 {
            self.do_leave_out_runs();
            self.output_leave_out_errors("Global", "GLOBAL");
            if self.m_lv_out_num_predict > 0 {
                copy_array(
                    &mut self.m_var,
                    1,
                    self.m_nvar_search,
                    &self.m_var_all_points,
                    1,
                );
                self.m_eval_funct
                    .remap_params(&mut self.av, &mut self.m_var);
                b3dsrand(&rand_seed);
            }
        }
        //
        // Ask about local alignments
        //
        self.m_if_local = self.m_if_do_local;
        if self.m_if_local != 0 {
            self.setup_and_do_local_alignments();
            drop(self.m_iun_local.take());
            if !self.m_too_few_fid {
                if self.m_if_res_out > 0 {
                    printf!("\n");
                }
                if self.m_if_do_robust > 0 && self.m_num_wgt_total > 0 {
                    printf!("\nSummary of robust weighting in local areas:\n");
                    printf!(
                        "%6d weights: %4d are 0, %4d are < .1, %4d are < .2, %5d (%4.1f%%) are < .5\n",
                        CArg::Int(self.m_num_wgt_total as i64),
                        CArg::Int(self.m_num_wgt_zero as i64),
                        CArg::Int(self.m_num_wgt1 as i64),
                        CArg::Int(self.m_num_wgt2 as i64),
                        CArg::Int(self.m_num_wgt5 as i64),
                        CArg::Dbl((100. * self.m_num_wgt5 as f64) / self.m_num_wgt_total as f64)
                    );
                }
                printf!(
                    "\n Ratio of local measurements to unknowns, mean: %5.2f  range %5.2f to %5.2f\n",
                    CArg::Dbl(
                        (self.m_local_mu_ratio_sum
                            / (self.m_num_patch_x * self.m_num_patch_y) as f32)
                            as f64
                    ),
                    CArg::Dbl(self.m_min_local_mu_ratio as f64),
                    CArg::Dbl(self.m_max_local_mu_ratio as f64)
                );
                let local_mean = formatted_error(
                    self.m_errsum_local / (self.m_num_patch_x * self.m_num_patch_y) as f32,
                    0.3,
                    9,
                );
                printf!(
                    " Residual error local mean:  %s    range %7.3f to %7.3f %s\n",
                    CArg::Str(&local_mean),
                    CArg::Dbl(self.m_err_local_min as f64),
                    CArg::Dbl(self.m_err_local_max as f64),
                    CArg::Str(&self.m_pix_units)
                );
                if self.m_if_do_robust > 0 && self.m_num_wgt_total > 0 {
                    let denom = self.m_num_patch_x * self.m_num_patch_y - self.m_num_rob_failed;
                    let wgt_mean = formatted_error(
                        self.m_wgt_err_sum_local / (if 1 > denom { 1 } else { denom }) as f32,
                        0.3,
                        9,
                    );
                    printf!(
                        "\n Weighted error local mean:  %s    range %7.3f to %7.3f %s\n",
                        CArg::Str(&wgt_mean),
                        CArg::Dbl(self.m_wgt_err_local_min as f64),
                        CArg::Dbl(self.m_wgt_err_local_max as f64),
                        CArg::Str(&self.m_pix_units)
                    );
                }
            }
            self.output_leave_out_errors("Local", "LOCAL");
        }
        //
        // Now handle filled in points by adding to model before writing XYZ model
        if self.m_make_filled_fid != 0 {
            let mut jpt = 1;
            while jpt <= self.m_n_all_real_pt {
                i = self.m_ifill_all_real_str[(jpt - 1) as usize];
                while i <= self.m_ifill_all_real_str[jpt as usize] - 1 {
                    let iu = (i - 1) as usize;
                    if self.m_num_in_fill_sum[iu] != 0 {
                        let nsum = b3dabs!(self.m_num_in_fill_sum[iu]) as f32;
                        xtmp = (self.m_xfill_final[iu] / nsum) * self.m_pixel_delta[0] - x_origin;
                        ytmp = (self.m_yfill_final[iu] / nsum) * self.m_pixel_delta[1] - y_origin;
                        ztmp = (self.m_map_all_view_to_file
                            [(self.m_iall_fill_view[iu] - 1) as usize]
                            - 1) as f32
                            * self.m_pixel_delta[2]
                            - z_origin;
                        if addimodpoint(
                            self.m_imod_obj[(jpt - 1) as usize],
                            self.m_imod_cont[(jpt - 1) as usize],
                            ztmp,
                            1,
                            xtmp,
                            ytmp,
                            ztmp,
                        ) != 0
                        {
                            error_exit::<false>("Adding a point for filling in fiducial model", 0);
                        }
                    }
                    i += 1;
                }
                jpt += 1;
            }
            let filled_name = filled_in_file.as_deref().unwrap_or("");
            ierr = imod_backup_file(filled_name);
            if writeimod(filled_name) != 0 {
                error_exit::<false>("Writing filled in model file", 0);
            }
        }
        for k in 0..self.m_n_all_real_pt as usize {
            self.m_all_xyz[3 * k] += transpose_xy_adj;
            self.m_all_xyz[3 * k + 1] -= transpose_xy_adj;
        }
        write_xyz_model(
            &mut self.fm,
            Some(&model_file),
            &self.m_all_xyz,
            &self.m_igroup,
            self.m_n_all_real_pt,
        );
        //
        if self.m_metro_error != 0 {
            printf!(
                "WARNING: %d  Minimization errors occurred\n",
                CArg::Int(self.m_metro_error as i64)
            );
        }
        if self.m_if_local == 0 && self.m_num_rob_failed > 0 && warn_on_rob_fail == 0 {
            let end = self
                .m_rob_fail_mess
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(ROB_MESS_SIZE);
            error_exit::<false>(&String::from_utf8_lossy(&self.m_rob_fail_mess[..end]), 0);
        }
        if self.m_num_rob_failed > 0 {
            printf!(
                "WARNING: Robust fitting failed in %d searches; non-robust result was restored\n",
                CArg::Int(self.m_num_rob_failed as i64)
            );
        }

        // Batchruntomo is looking for 'Minimum numbers of fiducials are too high'
        if self.m_too_few_fid {
            error_exit::<false>(
                "Minimum numbers of fiducials are too high - check if there are enough fiducials on the minority surface",
                0,
            );
        }

        c_exit(0);
    }
}

impl TiltAlign {
    /// Original: `TiltAlign::alignAndOutputResults` (`tiltalign.cpp:847`).
    ///
    /// Run the alignment for global or local area, compute transforms and other
    /// output, and output results.
    pub fn align_and_output_results(&mut self) {
        let mut prev_mean: f32;
        let mut prev_max: f32;
        let mut f_previous: f32;
        let mut all_mean: f32;
        let mut f_last: f32;
        let f_original: f32;
        let mut zmin: f32;
        let mut zmax: f32;
        let zmiddle: f32;
        let cos_theta: f32;
        let sin_theta: f32;
        let mut xtmp: f32;
        let mut a11: f32;
        let mut a12: f32;
        let mut a21: f32;
        let mut a22: f32;
        let mut dmat = [0f32; 9];
        let mut xtmat = [0f32; 9];
        let mut ytmat = [0f32; 9];
        let mut prmat = [0f32; 4];
        let mut rmat = [0f32; 9];
        let mut beam_inv = [0f32; 9];
        let mut beam_mat = [0f32; 9];
        let mut sin_bet: f32 = 0.;
        let mut cos_del: f32 = 0.;
        let mut sin_del: f32 = 0.;
        let mut cos_alf: f32 = 0.;
        let mut sin_alf: f32 = 0.;
        let mut cos_bet: f32 = 0.;
        let mut cos_beam: f32 = 0.;
        let mut sin_beam: f32 = 0.;
        let mut if_averaged: i32;
        let mut metro_robust: i32;
        let mut iord: i32;
        let mut ipt: i32;
        let mut num_unknown_tot: i32;
        let num_unknown_tot2: i32;
        let mut nord: i32;
        let mut ivt: i32;
        let mut ivnd: i32 = 0;
        let mut ivst: i32 = 1;
        let mut i: i32 = 0;
        let mut ierr: i32 = 0;
        let mut jpt: i32;
        let mut itmp: i32;
        let max_var: i32;
        let mut j: i32;
        let mut nin_view_sum: i32;
        let mut ib: i32;
        let err_mean_nm: f32;
        let mut comp_inc: f32;
        let mut comp_abs: f32;
        let unknown_ratio2: f32;
        let mut wgt_err_mean: f32 = 0.;
        let mut proj_str_factor: f32;
        let mut proj_str_axis: f32;
        let mut cos2rot: f32 = 0.;
        let mut sin2rot: f32 = 0.;
        let mut tilt_out: f32;
        let mut fa = [0f32; 6];
        let mut fb = [0f32; 6];
        let mut fc = [0f32; 6];
        let mut fpstr = [0f32; 6];
        let mut sxoz: f32;
        let mut szox: f32;
        let mut sxox: f32;
        let mut szoz: f32;
        let mut xo: f32;
        let mut zo: f32;
        let mut num_one_cycle: i32;
        let mut num_tot_cycles: i32 = 0;
        let mut max_tot_cycles: i32 = 0;
        let ndxtry: i32;
        let mut num_below_crit: i32;
        let mut vw_err_sum: f32;
        let mut vw_err_sq: f32;
        let mut dxtry: f32;
        let mut offmin: f32 = 0.;
        let mut dxmid: f32 = 0.;
        let mut offsum: f32;
        let mut xt_const: f32;
        let mut xt_fac: f32;
        let mut off: f32;
        let mut cos_tmp: f32 = 0.;
        let mut sin_tmp: f32 = 0.;
        let mut afac: f32 = 0.;
        let mut bfac: f32 = 0.;
        let mut cfac: f32 = 0.;
        let mut dfac: f32 = 0.;
        let mut efac: f32 = 0.;
        let mut ffac: f32 = 0.;
        let mut err_sd: f32 = 0.;
        let mut dysum: f32;
        let mut roll_points: f32;
        let mut nvar_geometric: i32 = 0;
        let mut nview_add: i32;
        let mut ixtry: i32;
        let mut nw_tot: i32 = 0;
        let mut nw0: i32 = 0;
        let mut nw1: i32 = 0;
        let mut nw2: i32 = 0;
        let mut nw5: i32 = 0;
        let mut iv: i32;
        let mut index: i32;
        let mut err_mean: f32 = 0.;
        let mut err_sqsm: f32;
        let mut err_sum: f32;
        let mut err_no_sd: f32;
        let mut resid_err: f32;
        let rms_scale: f32;
        let mut denom: f32;
        let mut tmp: f32;
        let mut f_final: f32 = 0.;
        //float xzOther, yzOther;
        let mut pmat = [0f64; 9];
        let mut wgt_err_sum: f64;
        let mut wgt_sum: f64;
        let mut rob_failed: bool;
        let rob_too_few: bool;
        let mut message = String::new();
        let mut unadj_tilt_file: Option<String> = None;
        let mut temp_name: Vec<u8> = Vec::new();

        if self.av.leaving_out == 0 {
            //
            // pack the xyz into the var list
            nvar_geometric = self.m_nvar_search;
            jpt = 1;
            while jpt <= self.av.nreal_pt - 1 {
                i = 1;
                while i <= 3 {
                    self.m_nvar_search += 1;
                    self.m_var[(self.m_nvar_search - 1) as usize] =
                        self.av.xyz[(jpt * 3 + i - 4) as usize];
                    i += 1;
                }
                jpt += 1;
            }
        } else {
            copy_array(
                &mut self.m_var,
                1,
                self.m_nvar_search,
                &self.m_var_all_points,
                1,
            );
            self.m_eval_funct
                .remap_params(&mut self.av, &mut self.m_var);
        }
        //
        // Make sure the h array is big enough
        max_var = self.m_nvar_search + 3;
        index = if (max_var + 3) * max_var > 3 * self.av.nview + 3 {
            (max_var + 3) * max_var
        } else {
            3 * self.av.nview + 3
        };
        if index > self.m_max_h {
            self.m_h = Vec::new();
            self.m_max_h = index;
            self.m_h = vec![0.; self.m_max_h as usize];
            memory_error(true, "array for h matrix");
        }
        //
        // Do beam tilt search only for global alignment
        self.av.robust_weights = 0;
        rms_scale = self.m_scale_xy * self.m_scale_xy / self.m_num_proj_pt as f32;
        metro_robust = self.m_metro_error;
        if self.m_if_bt_search == 0 || self.m_if_local > 0 || self.av.leaving_out != 0 {
            let ncycle = if self.av.leaving_out != 0 {
                -b3dabs!(self.m_max_cycles)
            } else {
                self.m_max_cycles
            };
            let if_hush = self.av.leaving_out;
            run_metro(
                &mut self.av,
                &mut self.m_eval_funct,
                &self.mx,
                self.m_nvar_search,
                &mut self.m_var,
                &mut self.m_var_err,
                &mut self.m_grad,
                &mut self.m_h,
                self.m_if_local,
                self.m_fac_metro,
                ncycle,
                if_hush,
                rms_scale,
                &mut f_final,
                &mut i,
                &mut self.m_metro_error,
                0,
                self.m_make_filled_fid,
            );
        } else {
            search_beam_tilt(
                &mut self.av,
                &mut self.m_eval_funct,
                &self.mx,
                self.m_bin_step_ini,
                self.m_bin_step_final,
                self.m_scan_step,
                self.m_nvar_search,
                &mut self.m_var,
                &mut self.m_var_err,
                &mut self.m_grad,
                &mut self.m_h,
                self.m_if_local,
                self.m_fac_metro,
                self.m_max_cycles,
                rms_scale,
                &mut f_final,
                &mut i,
                &mut self.m_metro_error,
                self.m_make_filled_fid,
            );
        }

        // Copy the var array for the full solution with all points, unweighted
        if self.m_lv_out_num_predict >= 0 && self.av.leaving_out == 0 {
            copy_array(
                &mut self.m_var_all_points,
                1,
                self.m_nvar_search,
                &self.m_var,
                1,
            );
            if self.av.test_set_frac_step > 0. {
                self.m_eval_funct.solve_left_out_xyzs(&mut self.av, true);
                get_test_set_errors(&mut self.av, 2);
            }
        }

        // If doing leave-out on robust, get the leave-out errors of the global solution
        // WITH the weights
        if self.av.leaving_out != 0 && self.m_if_do_robust != 0 {
            if self.m_lv_out_num_predict == 0 && (self.m_if_local == 0 || self.av.xyz_fixed == 0) {
                self.m_eval_funct.solve_left_out_xyzs(&mut self.av, false);
            }
            get_leave_out_errors(&mut self.av, &self.m_weight_none_left_out, -1);
        }

        //
        // If doing robust fitting, just restart the search with all current values
        if self.m_if_do_robust != 0 && self.m_metro_error > metro_robust {
            printf!("\nSkipping robust fitting because of minimization error\n\n");
        }
        rob_failed = false;
        self.m_did_robust = false;
        if self.m_if_do_robust != 0 && self.m_metro_error == metro_robust {
            max_tot_cycles =
                (b3dabs!(self.m_max_cycles) as f32 * self.m_robust_tot_cycle_fac) as i32;
            num_tot_cycles = 0;
            num_one_cycle = 0;
            num_below_crit = 0;
            jpt = self.m_min_res_robust;
            index = MAX_WGT_RINGS;
            if self.m_if_local != 0 {
                jpt = self.m_min_local_res_robust;
                index = 1;
            } else {
                self.m_num_wgt_total = 0;
                self.m_num_wgt_zero = 0;
                self.m_num_wgt1 = 0;
                self.m_num_wgt2 = 0;
                self.m_num_wgt5 = 0;
            }
            self.find_median_residual();
            if self.av.patch_track_model != 0 && self.av.robust_by_track != 0 {
                setup_track_weight_groups(
                    &mut self.av,
                    &self.mx,
                    if index < 5 { index } else { 5 },
                    jpt,
                    &self.m_ind_all_real,
                    &mut ierr,
                );
            } else {
                setup_weight_groups(
                    &mut self.av,
                    &self.mx,
                    index,
                    jpt,
                    self.m_min_tilt_view,
                    &mut ierr,
                );
            }
            rob_too_few = ierr != 0;
            if rob_too_few && self.av.leaving_out == 0 {
                printf!("WARNING: Too few data points to do robust fitting\n");
            }
            self.av.robust_weights = 1;
            jpt = 0;
            while jpt < self.m_num_proj_pt {
                if self.av.leaving_out == 0 && self.av.test_set_frac_step <= 0. {
                    self.av.weight[jpt as usize] = 1.;
                }
                self.m_err_save[jpt as usize] = 1.;
                jpt += 1;
            }
            f_original = f_final;
            f_final *= 10.;
            f_last = f_final;
            if_averaged = 1;
            metro_robust = 0;
            copy_array(&mut self.m_var_save, 1, self.m_nvar_search, &self.m_var, 1);
            while num_tot_cycles < max_tot_cycles && !rob_too_few {
                f_previous = f_last;
                f_last = f_final;
                copy_array(
                    &mut self.m_wgt_prev,
                    1,
                    self.m_num_proj_pt,
                    &self.m_err_save,
                    1,
                );
                copy_array(
                    &mut self.m_err_save,
                    1,
                    self.m_num_proj_pt,
                    &self.av.weight,
                    1,
                );
                {
                    // `(float *)mIndSave, (float *)mJptSave, (int *)mXyzErr`: the same
                    // storage reinterpreted (module doc).
                    // SAFETY: `i32` and `f32` have the same size and alignment and every
                    // bit pattern is a valid value of both; each view covers exactly the
                    // `Vec`'s initialised elements and is the only live reference to
                    // them for the duration of the call.
                    let dist_res: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.m_ind_save.as_mut_ptr().cast::<f32>(),
                            self.m_ind_save.len(),
                        )
                    };
                    let work: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.m_jpt_save.as_mut_ptr().cast::<f32>(),
                            self.m_jpt_save.len(),
                        )
                    };
                    let iwork: &mut [i32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.m_xyz_err.as_mut_ptr().cast::<i32>(),
                            self.m_xyz_err.len(),
                        )
                    };
                    compute_weights(&mut self.av, &self.m_ind_all_real, dist_res, work, iwork);
                }
                run_metro(
                    &mut self.av,
                    &mut self.m_eval_funct,
                    &self.mx,
                    self.m_nvar_search,
                    &mut self.m_var,
                    &mut self.m_var_err,
                    &mut self.m_grad,
                    &mut self.m_h,
                    1,
                    self.m_fac_metro,
                    -b3dabs!(self.m_max_cycles),
                    1,
                    rms_scale,
                    &mut f_final,
                    &mut i,
                    &mut metro_robust,
                    if num_tot_cycles > 0 { 1 } else { 0 },
                    self.m_make_filled_fid,
                );
                num_tot_cycles += i;
                if i <= 1 {
                    num_one_cycle += 1;
                } else {
                    num_one_cycle = 0;
                }
                //
                // Count up the weights below various levels and analyze change in weights
                nw5 = 0;
                nw0 = 0;
                nw1 = 0;
                nw2 = 0;
                nw_tot = 0;
                err_mean = 0.; // Mean change of weights < 0.5
                err_sd = 0.; // Max change in all weights
                prev_mean = 0.; // Mean difference between new weight and previous (2 ago) weights
                prev_max = 0.; // Max difference between new weight and previous (2 ago) weights
                all_mean = 0.; // Mean change of all weights
                jpt = 1;
                while jpt <= self.av.nreal_pt {
                    index = self.av.ireal_str[jpt as usize] - 1;
                    if self.av.patch_track_model != 0 && self.av.robust_by_track != 0 {
                        index = self.av.ireal_str[(jpt - 1) as usize];
                    }
                    i = self.av.ireal_str[(jpt - 1) as usize];
                    while i <= index {
                        let iu = (i - 1) as usize;
                        let w = self.av.weight[iu];
                        nw_tot += 1;
                        if w < 0.5 {
                            nw5 += 1;
                            err_mean += b3dabs!(w - self.m_err_save[iu]);
                        }
                        prev_mean += b3dabs!(w - self.m_wgt_prev[iu]);
                        all_mean += b3dabs!(w - self.m_err_save[iu]);
                        if w == 0. {
                            nw0 += 1;
                        }
                        if (w as f64) < 0.1 {
                            nw1 += 1;
                        }
                        if (w as f64) < 0.2 {
                            nw2 += 1;
                        }
                        let d1 = b3dabs!(w - self.m_err_save[iu]);
                        err_sd = if err_sd > d1 { err_sd } else { d1 };
                        let d2 = b3dabs!(w - self.m_wgt_prev[iu]);
                        prev_max = if prev_max > d2 { prev_max } else { d2 };
                        i += 1;
                    }
                    jpt += 1;
                }
                if nw5 > 0 {
                    err_mean /= nw5 as f32;
                }
                prev_mean /= self.m_num_proj_pt as f32;
                all_mean /= self.m_num_proj_pt as f32;
                // write(*, '(i5,f12.6,f9.4,f11.6,f9.4,f11.6,f9.4)') numTotCycles, &
                //   sqrt(fFinal*rmsScale), errMean, allMean, errSd, prevMean, prevMax
                if err_sd < self.m_del_wgt_max_crit || err_mean < self.m_del_wgt_mean_crit {
                    num_below_crit += 1;
                } else {
                    num_below_crit = 0;
                }
                rob_failed = metro_robust > 0 || f_final > self.m_robust_max_del_err * f_original;
                if rob_failed
                    || num_below_crit >= self.m_max_del_wgt_below_crit
                    || num_one_cycle >= self.m_max_robust_one_cycle
                    || (if_averaged == 1
                        && err_sd < self.m_del_wgt_max_crit
                        && err_mean < self.m_del_wgt_mean_crit)
                {
                    break;
                }
                if if_averaged == 0
                    && ((prev_mean < all_mean && prev_max < err_sd)
                        || (f_final / f_last) as f64 > 1.05
                        || ((f_final / f_last) as f64 > 1.02
                            && (f_last / f_previous) as f64 > 1.01))
                {
                    // print *,'Averaging previous weights'

                    jpt = 0;
                    while jpt < self.m_num_proj_pt {
                        let ju = jpt as usize;
                        self.av.weight[ju] =
                            (0.5 * (self.m_err_save[ju] + self.m_wgt_prev[ju]) as f64) as f32;
                        self.m_err_save[ju] = 1.;
                        jpt += 1;
                    }
                    if_averaged = 1;
                } else {
                    if_averaged = 0;
                }
            }
            //
            if rob_failed
                || (num_tot_cycles > max_tot_cycles
                    && (err_sd > self.m_undo_wgt_relax_crit * self.m_del_wgt_max_crit
                        || err_mean > self.m_undo_wgt_relax_crit * self.m_del_wgt_mean_crit))
            {
                //
                // Issue message and undo the failed search
                let text: Vec<u8> = if metro_robust > 0 {
                    b"Robust fitting ended with minimization error\n".to_vec()
                } else if rob_failed {
                    c_format_bytes(
                        "Robust fitting ended because F error increased to %13.6f\n",
                        &[CArg::Dbl((f_final * rms_scale).sqrt() as f64)],
                    )
                } else {
                    c_format_bytes(
                        "Robust fitting ended after %d cycles without converging\n",
                        &[CArg::Int(num_tot_cycles as i64)],
                    )
                };
                // strncpy pads with NULs; snprintf(…, ROB_MESS_SIZE, …) keeps at most
                // ROB_MESS_SIZE - 1 characters.  Either way the message ends at the
                // first NUL.
                let keep = text.len().min(ROB_MESS_SIZE - 1);
                self.m_rob_fail_mess[..keep].copy_from_slice(&text[..keep]);
                self.m_rob_fail_mess[keep..ROB_MESS_SIZE].fill(0);
                if self.av.leaving_out == 0 {
                    printf!("%s", CArg::Bytes(&self.m_rob_fail_mess[..keep]));
                    printf!("Restarting non-robust search to restore original result\n");
                }
                copy_array(&mut self.m_var, 1, self.m_nvar_search, &self.m_var_save, 1);
                self.av.robust_weights = 0;
                let ncycle = if self.av.leaving_out != 0 {
                    -b3dabs!(self.m_max_cycles)
                } else {
                    self.m_max_cycles
                };
                let if_hush = self.av.leaving_out;
                run_metro(
                    &mut self.av,
                    &mut self.m_eval_funct,
                    &self.mx,
                    self.m_nvar_search,
                    &mut self.m_var,
                    &mut self.m_var_err,
                    &mut self.m_grad,
                    &mut self.m_h,
                    self.m_if_local,
                    self.m_fac_metro,
                    ncycle,
                    if_hush,
                    rms_scale,
                    &mut f_final,
                    &mut i,
                    &mut self.m_metro_error,
                    0,
                    self.m_make_filled_fid,
                );
                self.m_num_rob_failed += 1;
            } else if !rob_too_few {
                self.m_did_robust = true;

                if self.av.leaving_out == 0 {
                    //
                    // Output results on success
                    printf!(
                        " Total cycles for robust fitting:%5d         Final   F :  %14.6f\n",
                        CArg::Int(num_tot_cycles as i64),
                        CArg::Dbl((f_final * rms_scale).sqrt() as f64)
                    );
                    if num_tot_cycles > max_tot_cycles {
                        printf!(
                            "\nRobust fitting ended after %4d cycles but meet relaxed convergence criteria\n\n",
                            CArg::Int(num_tot_cycles as i64)
                        );
                    }
                    printf!(
                        " Final mean and max weight change %7.4f %7.4f\n",
                        CArg::Dbl(err_mean as f64),
                        CArg::Dbl(err_sd as f64)
                    );
                    printf!(
                        "%6d weights: %4d are 0, %4d are < .1, %4d are < .2, %5d (%4.1f%%) are < .5\n\n",
                        CArg::Int(nw_tot as i64),
                        CArg::Int(nw0 as i64),
                        CArg::Int(nw1 as i64),
                        CArg::Int(nw2 as i64),
                        CArg::Int(nw5 as i64),
                        CArg::Dbl((100. * nw5 as f64) / nw_tot as f64)
                    );
                    if self.m_if_local != 0 {
                        self.m_num_wgt_total += nw_tot;
                        self.m_num_wgt_zero += nw0;
                        self.m_num_wgt1 += nw1;
                        self.m_num_wgt2 += nw2;
                        self.m_num_wgt5 += nw5;
                    }
                }
            } else {
                let text = b"TOO FEW DATA POINTS TO DO ROBUST FITTING\n";
                self.m_rob_fail_mess[..text.len()].copy_from_slice(text);
                self.m_rob_fail_mess[text.len()..ROB_MESS_SIZE].fill(0);
                self.m_num_rob_failed += 1;
            }

            // Save the weights from full solution and get test set errors
            if self.m_lv_out_num_predict >= 0 && self.av.leaving_out == 0 {
                copy_array(
                    &mut self.m_weight_none_left_out,
                    1,
                    self.m_num_proj_pt,
                    &self.av.weight,
                    1,
                );
                if self.av.test_set_frac_step > 0. {
                    self.m_eval_funct.solve_left_out_xyzs(&mut self.av, true);
                    get_test_set_errors(&mut self.av, 3);
                }
            }
        }

        // Return now if leaving out points, everything is still scaled
        if self.av.leaving_out != 0 {
            return;
        }

        //
        // unscale all the points, dx, dy, and restore angles to degrees
        //
        let nvs = self.m_nvar_search;
        index = 0;
        i = 1;
        while i <= self.m_nvar_angle {
            self.m_var[(i - 1) as usize] /= self.m_dtor;
            index += 1;
            self.m_var_err[(i - 1) as usize] =
                (self.m_h[(index * nvs - nvs + index - 1) as usize].sqrt() / nvs as f32)
                    / self.m_dtor;
            i += 1;
        }
        //
        i = self.m_nvar_angle + 1;
        while i <= self.m_nvar_scaled {
            index += 1;
            self.m_var_err[(i - 1) as usize] =
                self.m_h[(index * nvs - nvs + index - 1) as usize].sqrt() / nvs as f32;
            i += 1;
        }
        //
        i = self.m_map_alf_start;
        while i <= self.m_map_alf_end {
            self.m_var[(i - 1) as usize] /= self.m_dtor;
            index += 1;
            self.m_var_err[(i - 1) as usize] =
                (self.m_h[(index * nvs - nvs + index - 1) as usize].sqrt() / nvs as f32)
                    / self.m_dtor;
            i += 1;
        }
        // leave projection skew and beam tilt as radians
        i = self.m_map_alf_end + 1;
        while i <= nvar_geometric {
            index += 1;
            self.m_var_err[(i - 1) as usize] =
                self.m_h[(index * nvs - nvs + index - 1) as usize].sqrt() / nvs as f32;
            i += 1;
        }
        //
        i = 1;
        while i <= self.av.nreal_pt {
            j = 1;
            while j <= 3 {
                self.av.xyz[(i * 3 + j - 4) as usize] *= self.m_scale_xy;
                index += 1;
                if i < self.av.nreal_pt {
                    self.m_xyz_err[(i * 3 + j - 4) as usize] = self.m_scale_xy
                        * self.m_h[(index * nvs - nvs + index - 1) as usize].sqrt()
                        / nvs as f32;
                }
                j += 1;
            }
            i += 1;
        }
        //
        err_sum = 0.;
        err_sqsm = 0.;
        wgt_err_sum = 0.;
        wgt_sum = 0.;
        for k in 0..self.av.nview as usize {
            self.m_view_res[k] = 0.;
            self.m_num_in_view[k] = 0;
            self.m_view_errsum[k] = 0.;
            self.m_view_errsq[k] = 0.;
        }

        // Have to exclude test set here (this is not done in leave-out runs)
        j = 0;
        while j < self.av.nreal_pt {
            i = self.av.ireal_str[j as usize] - 1;
            while i < self.av.ireal_str[(j + 1) as usize] - 1 {
                let iu = i as usize;
                self.av.xx[iu] *= self.m_scale_xy;
                self.av.yy[iu] *= self.m_scale_xy;
                self.av.xresid[iu] *= self.m_scale_xy;
                self.av.yresid[iu] *= self.m_scale_xy;
                if self.m_lv_out_num_predict >= 0 && self.av.real_in_test_set[j as usize] != 0 {
                    i += 1;
                    continue;
                }
                resid_err = (self.av.xresid[iu] * self.av.xresid[iu]
                    + self.av.yresid[iu] * self.av.yresid[iu])
                    .sqrt();
                wgt_err_sum += (resid_err * self.av.weight[iu].sqrt()) as f64;
                wgt_sum += self.av.weight[iu].sqrt() as f64;
                iv = self.av.isec_view[iu];
                self.m_num_in_view[(iv - 1) as usize] += 1;
                self.m_view_errsum[(iv - 1) as usize] += resid_err;
                self.m_view_errsq[(iv - 1) as usize] += resid_err * resid_err;
                i += 1;
            }
            j += 1;
        }
        //
        iv = 1;
        while iv <= self.av.nview {
            let ivu = (iv - 1) as usize;
            self.av.dxy[(iv * 2 - 2) as usize] *= self.m_scale_xy;
            self.av.dxy[(iv * 2 - 1) as usize] *= self.m_scale_xy;
            //
            // save global solution now
            //
            if self.m_if_local == 0 {
                self.av.glb_alf[ivu] = self.av.alf[ivu];
                self.av.glb_tilt[ivu] = self.av.tilt[ivu];
                self.av.glb_rot[ivu] = self.av.rot[ivu];
                self.av.glb_skew[ivu] = self.av.skew[ivu];
                self.av.glb_gmag[ivu] = self.av.gmag[ivu];
                self.av.glb_dmag[ivu] = self.av.dmag[ivu];
            }
            self.av.rot[ivu] /= self.m_dtor;
            self.av.tilt[ivu] /= self.m_dtor;
            self.av.skew[ivu] /= self.m_dtor;
            self.av.alf[ivu] /= self.m_dtor;
            self.m_view_res[ivu] = self.m_view_errsum[ivu] / self.m_num_in_view[ivu] as f32;
            err_sum += self.m_view_errsum[ivu];
            err_sqsm += self.m_view_errsq[ivu];
            //
            // find mean and sd residual of minimum number of points in a local
            // group of views
            //
            nview_add = 1;
            nin_view_sum = 0;
            while nin_view_sum < self.m_num_local_res && nview_add < self.av.nview {
                ivst = if 1 > iv - nview_add / 2 {
                    1
                } else {
                    iv - nview_add / 2
                };
                ivnd = if self.av.nview < ivst + nview_add - 1 {
                    self.av.nview
                } else {
                    ivst + nview_add - 1
                };
                nin_view_sum = 0;
                ivt = ivst;
                while ivt <= ivnd {
                    nin_view_sum += self.m_num_in_view[(ivt - 1) as usize];
                    ivt += 1;
                }
                nview_add += 1;
            }
            vw_err_sum = 0.;
            vw_err_sq = 0.;
            ivt = ivst;
            while ivt <= ivnd {
                vw_err_sum += self.m_view_errsum[(ivt - 1) as usize];
                vw_err_sq += self.m_view_errsq[(ivt - 1) as usize];
                ivt += 1;
            }
            self.m_view_mean_res[ivu] = vw_err_sum / nin_view_sum as f32;
            self.m_view_sd_res[ivu] = ((vw_err_sq - vw_err_sum * vw_err_sum / nin_view_sum as f32)
                / (nin_view_sum - 1) as f32)
                .sqrt();
            iv += 1;
        }
        //
        // convert the projection stretch to a matrix
        // (This only works directly into fpstr if it is symmetric)
        //
        fill_proj_matrix(
            self.av.proj_str_rot,
            self.av.proj_skew,
            &mut fpstr,
            &mut cos_tmp,
            &mut sin_tmp,
            &mut cos2rot,
            &mut sin2rot,
        );
        (xo, zo, proj_str_factor, proj_str_axis) =
            amat_to_rotmagstr(fpstr[0], fpstr[2], fpstr[1], fpstr[3]);
        let _ = (proj_str_factor, proj_str_axis);
        //
        // if doing local solution, need to find rotation to match
        // the original set of points
        //
        if self.m_if_local != 0 {
            sxoz = 0.;
            szox = 0.;
            sxox = 0.;
            szoz = 0.;
            i = 1;
            while i <= self.av.nreal_pt {
                // Fixed in translation (2026-09-26, `BUGS.md`): the source tests
                // `realInTestSet[j]` with `j` left at `nrealPt` by the residual loop
                // above (`tiltalign.cpp:1254`); the point `i`'s own flag is tested.
                if self.m_lv_out_num_predict >= 0 && self.av.real_in_test_set[(i - 1) as usize] != 0
                {
                    i += 1;
                    continue;
                }
                let ia = ((self.m_ind_all_real[(i - 1) as usize] - 1) * 3) as usize;
                xo = self.m_all_xyz[ia] - self.m_xcen - self.m_x_shift;
                zo = self.m_all_xyz[ia + 2] - self.m_z_shift;
                sxox += xo * self.av.xyz[(i * 3 - 3) as usize];
                sxoz += xo * self.av.xyz[(i * 3 - 1) as usize];
                szox += zo * self.av.xyz[(i * 3 - 3) as usize];
                szoz += zo * self.av.xyz[(i * 3 - 1) as usize];
                i += 1;
            }
            roll_points = 0.;
            if ((sxox + szoz) as f64) > 1.0e-5 * b3dabs!(sxoz - szox) as f64 {
                roll_points = ((sxoz - szox) / (sxox + szoz)).atan() / self.m_dtor;
            }
            //
            // rolls the points, reduce this amount from the tilts
            //
            cos_theta = (self.m_dtor * roll_points).cos();
            sin_theta = (self.m_dtor * roll_points).sin();
            i = 1;
            while i <= self.av.nreal_pt {
                let b = (i * 3 - 3) as usize;
                xtmp = self.av.xyz[b] * cos_theta + self.av.xyz[b + 2] * sin_theta;
                self.av.xyz[b + 2] = -self.av.xyz[b] * sin_theta + self.av.xyz[b + 2] * cos_theta;
                self.av.xyz[b] = xtmp;
                i += 1;
            }
            i = 1;
            while i <= self.av.nview {
                self.av.tilt[(i - 1) as usize] -= roll_points;
                i += 1;
            }
        }
        //
        comp_inc = 1.;
        comp_abs = 1.;
        num_unknown_tot = nvar_geometric + 3 * (self.av.nreal_pt - 1);
        if self.av.xyz_fixed != 0 {
            num_unknown_tot = nvar_geometric;
        }
        num_unknown_tot2 = num_unknown_tot + 2 * (self.av.nview - 1);
        //unknownRatio = (2. * mNumProjPt) / B3DMAX(numUnknownTot, 1);
        unknown_ratio2 = ((2. * self.m_num_proj_pt as f64)
            / (if num_unknown_tot2 > 1 {
                num_unknown_tot2
            } else {
                1
            }) as f64) as f32;
        printf!(
            "%4d views,%5d geometric variables,%5d 3-D points,%6d projection points\n  Ratio of total measured values to all unknowns =%6d/%4d =%7.2f\n",
            CArg::Int(self.av.nview as i64),
            CArg::Int(nvar_geometric as i64),
            CArg::Int(self.av.nreal_pt as i64),
            CArg::Int(self.m_num_proj_pt as i64),
            CArg::Int((2 * self.m_num_proj_pt) as i64),
            CArg::Int(num_unknown_tot2 as i64),
            CArg::Dbl(unknown_ratio2 as f64)
        );
        if self.m_if_local != 0 {
            self.m_min_local_mu_ratio = if self.m_min_local_mu_ratio < unknown_ratio2 {
                self.m_min_local_mu_ratio
            } else {
                unknown_ratio2
            };
            self.m_max_local_mu_ratio = if self.m_max_local_mu_ratio > unknown_ratio2 {
                self.m_max_local_mu_ratio
            } else {
                unknown_ratio2
            };
            self.m_local_mu_ratio_sum += unknown_ratio2;
        }

        if self.m_if_var_out != 0 {
            if self.m_ncomp_search == 0 {
                if self.av.map_proj_stretch > 0 {
                    printf!(
                        "\nProjection skew is %7.2f degrees\n",
                        CArg::Dbl((self.av.proj_skew / self.m_dtor) as f64)
                    );
                }
                if self.av.map_beam_tilt > 0 || self.m_if_bt_search != 0 {
                    printf!(
                        "\nBeam tilt angle is %7.2f degrees\n",
                        CArg::Dbl((self.av.beam_tilt / self.m_dtor) as f64)
                    );
                }
                if self.m_map_alf_start > self.m_map_alf_end {
                    if self.m_if_local == 0 {
                        printf!(
                            "\n At minimum tilt, rotation angle is %7.2f\n",
                            CArg::Dbl(self.av.rot[(self.m_min_tilt_view - 1) as usize] as f64)
                        );
                    }
                    printf!(
                        "\n view   rotation    tilt    deltilt     mag      dmag      skew    resid-%s\n",
                        CArg::Str(&self.m_pix_units)
                    );
                    i = 1;
                    while i <= self.av.nview {
                        let iu = (i - 1) as usize;
                        j = self.av.map_view_to_file[iu];
                        printf!(
                            "%4d %9.1f %9.1f %9.2f %9.4f %9.4f %9.2f %9.2f\n",
                            CArg::Int(j as i64),
                            CArg::Dbl(self.av.rot[iu] as f64),
                            CArg::Dbl(self.av.tilt[iu] as f64),
                            CArg::Dbl(
                                (self.av.tilt[iu] - self.m_tilt_orig[(j - 1) as usize]) as f64
                            ),
                            CArg::Dbl(self.av.gmag[iu] as f64),
                            CArg::Dbl(self.av.dmag[iu] as f64),
                            CArg::Dbl(self.av.skew[iu] as f64),
                            CArg::Dbl((self.m_view_res[iu] * self.m_pixel_size) as f64)
                        );
                        i += 1;
                    }
                } else if self.av.if_rot_fix == -1 || self.av.if_rot_fix == -2 {
                    if self.av.if_rot_fix == -1 {
                        printf!(
                            "\n Fixed rotation angle is %7.2f\n",
                            CArg::Dbl(self.av.rot[0] as f64)
                        );
                    }
                    if self.av.if_rot_fix == -2 {
                        printf!(
                            "\n Overall rotation angle is %7.2f",
                            CArg::Dbl(self.av.rot[0] as f64)
                        );
                    }
                    printf!(
                        "\n view     tilt    deltilt     mag      dmag      skew     X-tilt   resid-%s\n",
                        CArg::Str(&self.m_pix_units)
                    );
                    i = 1;
                    while i <= self.av.nview {
                        let iu = (i - 1) as usize;
                        j = self.av.map_view_to_file[iu];
                        printf!(
                            "%4d %9.1f %9.2f %9.4f %9.4f %9.2f %9.2f %9.2f\n",
                            CArg::Int(j as i64),
                            CArg::Dbl(self.av.tilt[iu] as f64),
                            CArg::Dbl(
                                (self.av.tilt[iu] - self.m_tilt_orig[(j - 1) as usize]) as f64
                            ),
                            CArg::Dbl(self.av.gmag[iu] as f64),
                            CArg::Dbl(self.av.dmag[iu] as f64),
                            CArg::Dbl(self.av.skew[iu] as f64),
                            CArg::Dbl(self.av.alf[iu] as f64),
                            CArg::Dbl((self.m_view_res[iu] * self.m_pixel_size) as f64)
                        );
                        i += 1;
                    }
                } else {
                    if self.m_map_alf_end > self.m_map_alf_start {
                        printf!(
                            "\nWARNING: Solutions for both rotation and X-axis tilt are very unreliable\n"
                        );
                    }
                    printf!(
                        "\n view rotation  tilt  deltilt    mag     dmag    skew   X-tilt  resid-%s\n",
                        CArg::Str(&self.m_pix_units)
                    );
                    i = 1;
                    while i <= self.av.nview {
                        let iu = (i - 1) as usize;
                        j = self.av.map_view_to_file[iu];
                        printf!(
                            "%4d %7.1f %7.1f %7.2f %8.4f %8.4f %7.2f %7.2f %7.2f\n",
                            CArg::Int(j as i64),
                            CArg::Dbl(self.av.rot[iu] as f64),
                            CArg::Dbl(self.av.tilt[iu] as f64),
                            CArg::Dbl(
                                (self.av.tilt[iu] - self.m_tilt_orig[(j - 1) as usize]) as f64
                            ),
                            CArg::Dbl(self.av.gmag[iu] as f64),
                            CArg::Dbl(self.av.dmag[iu] as f64),
                            CArg::Dbl(self.av.skew[iu] as f64),
                            CArg::Dbl(self.av.alf[iu] as f64),
                            CArg::Dbl((self.m_view_res[iu] * self.m_pixel_size) as f64)
                        );
                        i += 1;
                    }
                }
            } else {
                printf!(
                    "\n view   rotation    tilt      mag    comp-inc  comp-abs    dmag      skew\n"
                );
                i = 1;
                while i <= self.av.nview {
                    let iu = (i - 1) as usize;
                    //
                    // for 0 tilts, output same compression values as last view
                    //
                    if self.av.tilt[iu] != 0. {
                        comp_inc = self.av.comp[iu];
                        comp_abs = comp_inc * self.av.gmag[iu];
                    }
                    printf!(
                        "%4d %9.1f %9.1f %9.4f %9.4f %9.4f %9.4f %9.2f\n",
                        CArg::Int(self.av.map_view_to_file[iu] as i64),
                        CArg::Dbl(self.av.rot[iu] as f64),
                        CArg::Dbl(self.av.tilt[iu] as f64),
                        CArg::Dbl(self.av.gmag[iu] as f64),
                        CArg::Dbl(comp_inc as f64),
                        CArg::Dbl(comp_abs as f64),
                        CArg::Dbl(self.av.dmag[iu] as f64),
                        CArg::Dbl(self.av.skew[iu] as f64)
                    );
                    i += 1;
                }
            }
            printf!("\n");
            if self.m_iu_angle.is_none() && self.m_if_local == 0 {
                for k in 0..self.av.nview {
                    if k % 10 == 0 {
                        printf!(" ANGLES");
                    }
                    printf!(" %6.2f", CArg::Dbl(self.av.tilt[k as usize] as f64));
                    if k % 10 == 9 || k == self.av.nview - 1 {
                        printf!("\n");
                    }
                }
            }
            if self.m_ncomp_search > 0 {
                for k in 0..self.av.nview {
                    if k % 10 == 0 {
                        printf!(" COMPRESS");
                    }
                    printf!(" %6.4f", CArg::Dbl(self.av.comp[k as usize] as f64));
                    if k % 10 == 9 || k == self.av.nview - 1 {
                        printf!("\n");
                    }
                }
            }
        }
        if self.m_if_xyz_out != 0 {
            if self.m_if_do_robust != 0 && self.av.robust_by_track != 0 {
                message = "   Weights".to_string();
            }
            printf!(
                "\n                     3-D point coordinates (with centroid zero)\n   #       X         Y         Z      obj  cont  resid-%s%s\n",
                CArg::Str(&self.m_pix_units),
                CArg::Str(&message)
            );
            j = 1;
            while j <= self.av.nreal_pt {
                vw_err_sum = 0.;
                i = self.av.ireal_str[(j - 1) as usize];
                while i <= self.av.ireal_str[j as usize] - 1 {
                    let iu = (i - 1) as usize;
                    vw_err_sum += (self.av.xresid[iu] * self.av.xresid[iu]
                        + self.av.yresid[iu] * self.av.yresid[iu])
                        .sqrt();
                    i += 1;
                }
                let ia = (self.m_ind_all_real[(j - 1) as usize] - 1) as usize;
                printf!(
                    "%4d %9.2f %9.2f %9.2f %6d %4d %11.4f",
                    CArg::Int(self.m_ind_all_real[(j - 1) as usize] as i64),
                    CArg::Dbl(self.av.xyz[(j * 3 - 3) as usize] as f64),
                    CArg::Dbl(self.av.xyz[(j * 3 - 2) as usize] as f64),
                    CArg::Dbl(self.av.xyz[(j * 3 - 1) as usize] as f64),
                    CArg::Int(self.m_imod_obj[ia] as i64),
                    CArg::Int(self.m_imod_cont[ia] as i64),
                    CArg::Dbl(
                        (vw_err_sum * self.m_pixel_size / self.m_nin_real[(j - 1) as usize] as f32)
                            as f64
                    )
                );
                if !message.is_empty() {
                    printf!(
                        " %11.4f\n",
                        CArg::Dbl(
                            self.av.weight[(self.av.ireal_str[(j - 1) as usize] - 1) as usize]
                                as f64
                        )
                    );
                } else {
                    printf!("\n");
                }
                j += 1;
            }
        }

        //
        // get min, max and midpoint of z values
        //
        zmin = 1.0e10;
        zmax = -1.0e10;
        ipt = 1;
        while ipt <= self.av.nreal_pt {
            let z = self.av.xyz[(ipt * 3 - 1) as usize];
            zmin = if zmin < z { zmin } else { z };
            zmax = if zmax > z { zmax } else { z };
            ipt += 1;
        }
        zmiddle = ((zmax + zmin) as f64 / 2.) as f32;
        if self.m_if_local == 0 {
            printf!(
                "\n Midpoint of Z range relative to centroid in Z: %7.2f\n",
                CArg::Dbl(zmiddle as f64)
            );
        }

        self.av.if_any_alf = 0;
        if self.m_map_alf_end > self.m_map_alf_start {
            self.av.if_any_alf = 1;
        }
        //
        // Output unmodified tilt angles
        if self.m_pipinput != 0 && self.m_if_local == 0 {
            if pip_get_string(b"OutputUnadjustedTiltFile", &mut temp_name) == 0 {
                let name = String::from_utf8_lossy(&temp_name).into_owned();
                let mut tmp_fp = b3d_open_file(&name, "w");
                unadj_tilt_file = Some(name);
                i = 1;
                while i <= self.av.nfile_views {
                    tilt_out = self.m_tilt_orig[(i - 1) as usize];
                    if self.av.map_file_to_view[(i - 1) as usize] != 0 {
                        tilt_out =
                            self.av.tilt[(self.av.map_file_to_view[(i - 1) as usize] - 1) as usize];
                    }
                    fprintf!(tmp_fp, "%7.2f\n", CArg::Dbl(tilt_out as f64));
                    i += 1;
                }
                drop(tmp_fp);
            }
        }
        //
        // Modify angles to account for beam tilt
        //
        if self.av.beam_tilt != 0. {
            iv = 1;
            while iv <= self.av.nview {
                let ivu = (iv - 1) as usize;
                convert_for_beamtilt(
                    &mut self.av.alf[ivu],
                    &mut self.av.tilt[ivu],
                    &mut self.av.rot[ivu],
                    self.av.beam_tilt,
                    self.av.if_any_alf,
                );
                iv += 1;
            }
        }
        //
        // output lists of angles that are complete for all file views
        //
        if self.m_if_local == 0 && self.m_iu_angle.is_some() {
            if let Some(iu_angle) = self.m_iu_angle.as_mut() {
                i = 1;
                while i <= self.av.nfile_views {
                    tilt_out = self.m_tilt_orig[(i - 1) as usize];
                    if self.av.map_file_to_view[(i - 1) as usize] != 0 {
                        tilt_out =
                            self.av.tilt[(self.av.map_file_to_view[(i - 1) as usize] - 1) as usize];
                    }
                    fprintf!(iu_angle, "%7.2f\n", CArg::Dbl(tilt_out as f64));
                    i += 1;
                }
            }
            // fclose(mIuAngle): the source's pointer stays non-NULL after the close;
            // nothing tests it again on a path that follows (mIfLocal == 0 and not
            // leaving out happens once).
            if let Some(iu_angle) = self.m_iu_angle.as_mut() {
                let _ = iu_angle.flush();
            }
            if self.m_iu_xtilt.is_none()
                && (self.av.if_any_alf != 0 || self.av.beam_tilt != 0.)
                && unadj_tilt_file.is_none()
            {
                printf!(
                    "\nWARNING: The solution includes X-axis tilts that change through the series\nWARNING: X-axis tilts should be output to a file and fed to the Tilt program\n"
                );
            }
        }
        drop(unadj_tilt_file);
        //
        if self.m_if_local == 0 && self.m_iu_xtilt.is_some() {
            if let Some(iu_xtilt) = self.m_iu_xtilt.as_mut() {
                i = 1;
                while i <= self.av.nfile_views {
                    tilt_out = 0.;
                    if self.av.map_file_to_view[(i - 1) as usize] != 0 {
                        tilt_out =
                            self.av.alf[(self.av.map_file_to_view[(i - 1) as usize] - 1) as usize];
                    }
                    fprintf!(iu_xtilt, "%7.2f\n", CArg::Dbl(tilt_out as f64));
                    i += 1;
                }
                let _ = iu_xtilt.flush();
            }
        }
        //
        // compute xforms, shift the dy's to minimize total shift, allow
        // user to shift dx's (and tilt axis) similarly or specify new location
        // of tilt axis
        // shift axis in z by making proper shifts in x
        //
        iv = 1;
        while iv <= self.av.nview {
            let ivu = (iv - 1) as usize;
            //
            // To compute transform, first get the coefficients of the full
            // projection.  Assume 0 beam tilt, it is already corrected
            //
            fill_dist_matrix(
                self.av.gmag[ivu],
                self.av.dmag[ivu],
                self.av.skew[ivu] * self.m_dtor,
                self.av.comp[ivu],
                1,
                &mut dmat,
                &mut cos_del,
                &mut sin_del,
            );
            fill_beam_matrices(
                0.,
                &mut beam_inv,
                &mut beam_mat,
                &mut cos_beam,
                &mut sin_beam,
            );
            fill_xtilt_matrix(
                self.av.alf[ivu] * self.m_dtor,
                self.av.if_any_alf,
                &mut xtmat,
                &mut cos_alf,
                &mut sin_alf,
            );
            fill_ytilt_matrix(
                self.av.tilt[ivu] * self.m_dtor,
                &mut ytmat,
                &mut cos_bet,
                &mut sin_bet,
            );
            fill_proj_matrix(
                self.av.proj_str_rot,
                self.av.proj_skew,
                &mut prmat,
                &mut cos_tmp,
                &mut sin_tmp,
                &mut cos2rot,
                &mut sin2rot,
            );
            fill_rot_matrix(
                self.av.rot[ivu] * self.m_dtor,
                &mut rmat,
                &mut cos_tmp,
                &mut sin_tmp,
            );
            EvalFunct::matrix_to_coef(
                &dmat, &xtmat, &beam_inv, &ytmat, &beam_mat, &prmat, &rmat, &mut afac, &mut bfac,
                &mut cfac, &mut dfac, &mut efac, &mut ffac,
            );
            if self.m_sub_sample_tracks && self.m_if_local == 0 {
                self.m_ss_afac[ivu] = afac;
                self.m_ss_bfac[ivu] = bfac;
                self.m_ss_cfac[ivu] = cfac;
                self.m_ss_dfac[ivu] = dfac;
                self.m_ss_efac[ivu] = efac;
                self.m_ss_ffac[ivu] = ffac;
            }
            //
            // Solve for transformation that maps 1, 0, 0 to cos beta, 0
            // and 0, 1, 0 to sin alf * sin beta, cos alf
            //
            denom = bfac * dfac - afac * efac;
            ib = 6 * (iv - 1);
            let b = ib as usize;
            self.m_fl[b] = (dfac * sin_alf * sin_bet - efac * cos_bet) / denom;
            self.m_fl[b + 2] = (bfac * cos_bet - afac * sin_alf * sin_bet) / denom;
            self.m_fl[b + 1] = dfac * cos_alf / denom;
            self.m_fl[b + 3] = -afac * cos_alf / denom;
            self.m_fl[b + 4] = -(self.m_fl[b] * self.av.dxy[(iv * 2 - 2) as usize]
                + self.m_fl[b + 2] * self.av.dxy[(iv * 2 - 1) as usize]);
            self.m_fl[b + 5] = -(self.m_fl[b + 1] * self.av.dxy[(iv * 2 - 2) as usize]
                + self.m_fl[b + 3] * self.av.dxy[(iv * 2 - 1) as usize]);
            //
            // Compute Z - dependent factors to add to X and Y in backprojection
            // This method does not depend on distortion model:
            // Compute coefficients of distortion plus tilts, solve for
            // transformation that aligns images to that, determine Z component
            // of projection equation to aligned images, and subtract component
            // expected to be applied in backprojection
            //
            for k in 0..9 {
                pmat[k] = dmat[k] as f64;
            }
            mat_product(&mut pmat, 3, 3, &xtmat, 3, 3);
            mat_product(&mut pmat, 3, 3, &ytmat, 2, 3);
            denom = (pmat[1] * pmat[3] - pmat[0] * pmat[4]) as f32;
            a11 = ((pmat[3] * sin_alf as f64 * sin_bet as f64 - pmat[4] * cos_bet as f64)
                / denom as f64) as f32;
            a12 = ((pmat[1] * cos_bet as f64 - pmat[0] * sin_alf as f64 * sin_bet as f64)
                / denom as f64) as f32;
            a21 = (pmat[3] * cos_alf as f64 / denom as f64) as f32;
            a22 = (-pmat[0] * cos_alf as f64 / denom as f64) as f32;
            self.m_xz_fac[ivu] = ((a11 as f64 * pmat[2] + a12 as f64 * pmat[5])
                / self.av.comp[ivu] as f64
                - (cos_alf * sin_bet) as f64) as f32;
            self.m_yz_fac[ivu] = ((a21 as f64 * pmat[2] + a22 as f64 * pmat[5])
                / self.av.comp[ivu] as f64
                + sin_alf as f64) as f32;
            //
            // Alternate based on solving equations from type 1 distortion model
            // (commented out in the source, as is the old way and its validation by
            // inverse multiplication, `tiltalign.cpp:1535-1578`)
            //
            // Set up to fit shifts against sines and cosines to find z shift from original
            let nview = self.av.nview;
            self.m_h[(iv - 1) as usize] = self.m_fl[b + 4];
            self.m_h[(iv + nview - 1) as usize] = (self.m_dtor * self.av.tilt[ivu]).sin();
            self.m_h[(iv + 2 * nview - 1) as usize] = (self.m_dtor * self.av.tilt[ivu]).cos();
            iv += 1;
        }

        // If Z change is relative to original position, determine component needed to
        // get back to original.  Fit to both sine and cosine so that it doesn't try to
        // fit just a sine to something with a shift (cosine) component, which doesn't give
        // consistent results for series with asymmetric extents
        self.m_znew = self.m_znew_input;
        if self.m_if_local == 0 && self.m_if_original_z != 0 && b3dnint!(self.m_znew) != 1000 {
            let nview = self.av.nview as usize;
            ls_fit2(
                &self.m_h[nview..],
                &self.m_h[2 * nview..],
                &self.m_h,
                self.av.nview,
                &mut self.m_z_shift,
                &mut dxmid,
                Some(&mut offmin),
            );
            self.m_znew += self.m_z_shift;
        }
        if b3dnint!(self.m_znew) == 1000 {
            self.m_znew = zmiddle;
        }
        if self.m_if_local != 0 {
            self.m_znew = -self.m_z_shift;
        }

        // adjust dx by the factor needed to shift axis in Z and set up for working on Y
        dysum = 0.;
        for k in 0..self.av.nview as usize {
            self.m_fl[6 * k + 4] -= self.m_znew * (self.m_dtor * self.av.tilt[k]).sin();
            self.m_h[k] = (1. - (self.m_dtor * self.av.tilt[k]).cos() as f64) as f32;
            dysum += self.m_fl[6 * k + 5];
        }
        self.m_dy_avg = dysum / self.av.nview as f32;
        if self.m_if_local == 0 {
            //
            // find value of X shift that minimizes overall loss of image - do
            // exhaustive scan centered on dx of the minimum tilt image
            //
            offmin = 1.0e10;
            dxmid = self.m_fl[(6 * (self.m_min_tilt_view - 1) + 4) as usize];
            //
            // DNM 11 / 10 / 01: eliminate real variable do loop in deference to f95
            // do dxtry = dxmid - 0.1 * xcen, dxmid + 0.1 * xcen, 0.1
            //
            ndxtry = (2. * self.m_xcen as f64) as i32;
            ixtry = 0;
            while ixtry <= ndxtry {
                dxtry = (dxmid as f64 + 0.1 * (ixtry as f32 - self.m_xcen) as f64) as f32;
                offsum = 0.;
                xt_fac = self.m_xtilt_new + dxtry;
                xt_const = self.m_xtilt_new - xt_fac;
                for k in 0..self.av.nview as usize {
                    off = b3dabs!(self.m_fl[6 * k + 4] + xt_const + xt_fac * self.m_h[k])
                        - self.m_xcen * self.m_h[k];
                    if off > 0. {
                        offsum += off;
                    }
                }
                if offsum < offmin {
                    offmin = offsum;
                    self.m_dx_min = dxtry;
                }
                ixtry += 1;
            }
            xt_fac = self.m_xtilt_new + self.m_dx_min;
            xt_const = self.m_xtilt_new - xt_fac;
            //
            // Put tilt axis at the new position, and get the final dy to
            // add up to 0.
            //
            for k in 0..self.av.nview as usize {
                self.m_fl[6 * k + 5] -= self.m_dy_avg;
                self.m_fl[6 * k + 4] += xt_const + xt_fac * self.m_h[k];
            }
            //
            // output a transform for each file view, find the nearest one
            // for non-included view
            //
            for k in 0..self.av.nfile_views {
                i = nearest_view(&self.av, k + 1);
                ib = 6 * (i - 1);
                let b = ib as usize;
                if let Some(sol_fp) = self.m_sol_file_fp.as_mut() {
                    fprintf!(
                        sol_fp,
                        " %11.7f %11.7f %11.7f %11.7f %11.3f %11.3f\n",
                        CArg::Dbl(self.m_fl[b] as f64),
                        CArg::Dbl(self.m_fl[b + 2] as f64),
                        CArg::Dbl(self.m_fl[b + 1] as f64),
                        CArg::Dbl(self.m_fl[b + 3] as f64),
                        CArg::Dbl(self.m_fl[b + 4] as f64),
                        CArg::Dbl(self.m_fl[b + 5] as f64)
                    );
                }
            }
            if !self.m_residual_file.is_empty() {
                let mut tmp_fp = b3d_open_file(&self.m_residual_file, "w");
                fprintf!(
                    tmp_fp,
                    "%6d residuals\n",
                    CArg::Int(self.m_num_proj_pt as i64)
                );
                for k in 0..self.m_num_proj_pt as usize {
                    fprintf!(
                        tmp_fp,
                        "%10.2f %9.2f %4d %7.2f %7.2f\n",
                        CArg::Dbl((self.av.xx[k] + self.m_xcen) as f64),
                        CArg::Dbl((self.av.yy[k] + self.m_ycen) as f64),
                        CArg::Int(
                            (self.av.map_view_to_file[(self.av.isec_view[k] - 1) as usize] - 1)
                                as i64
                        ),
                        CArg::Dbl(self.av.xresid[k] as f64),
                        CArg::Dbl(self.av.yresid[k] as f64)
                    );
                }
                drop(tmp_fp);
            }
            //
            // output the z factors if option requested
            //
            temp_name.clear();
            if pip_get_string(b"OutputZFactorFile", &mut temp_name) == 0 {
                self.m_if_zfac = 1;
                let mut tmp_fp = b3d_open_file(&String::from_utf8_lossy(&temp_name), "w");
                for k in 0..self.av.nfile_views as usize {
                    i = nearest_view(&self.av, k as i32 + 1) - 1;
                    self.m_glb_xz_fac[k] = self.m_xz_fac[i as usize];
                    self.m_glb_yz_fac[k] = self.m_yz_fac[i as usize];
                    fprintf!(
                        tmp_fp,
                        "%12.6f %11.6f\n",
                        CArg::Dbl(self.m_xz_fac[i as usize] as f64),
                        CArg::Dbl(self.m_yz_fac[i as usize] as f64)
                    );
                }
                drop(tmp_fp);
            }
        } else {
            //
            // If local, the procedure now is first to process each relevant item
            // for the local views, in place or into a different array (grad)
            // This is mixing local views and global values, so mapping from local
            // to global view number is needed to access the global values
            // Then expand this array to the global views.
            // Then create output for all file views, taking the global or null
            // value as appropriate for an excluded view
            //
            // Do this for tilt angles, alpha if they exist, z factors, and xforms
            //
            let iun_local = self.m_iun_local.as_mut().expect("local file is open");
            for k in 0..self.av.nview as usize {
                self.m_grad[k] = self.av.tilt[k]
                    - self.av.glb_tilt[(self.m_map_local_to_all[k] - 1) as usize] / self.m_dtor;
            }
            expand_local_to_all(
                &mut self.m_grad,
                1,
                1,
                self.av.nview,
                self.m_n_all_view,
                &self.m_map_all_to_local,
            );
            for k in 0..self.av.nfile_views as usize {
                iv = self.m_map_all_file_to_view[k];
                self.m_h[k] = 0.;
                if iv > 0 {
                    // Fixed in translation (2026-09-26, `BUGS.md`): the source reads
                    // `mGrad[i]` with the file-view index (`tiltalign.cpp:1701`);
                    // the values are in all-view order, so view `iv` is read.
                    self.m_h[k] = self.m_grad[(iv - 1) as usize];
                }
            }
            for k in 0..self.av.nfile_views {
                fprintf!(iun_local, " %6.2f", CArg::Dbl(self.m_h[k as usize] as f64));
                if k % 10 == 9 || k == self.av.nfile_views - 1 {
                    fprintf!(iun_local, "\n");
                }
            }
            if self.m_map_alf_start <= self.m_map_alf_end || self.av.beam_tilt != 0. {
                for k in 0..self.av.nview as usize {
                    self.m_grad[k] = self.av.alf[k]
                        - self.av.glb_alf[(self.m_map_local_to_all[k] - 1) as usize] / self.m_dtor;
                }
                expand_local_to_all(
                    &mut self.m_grad,
                    1,
                    1,
                    self.av.nview,
                    self.m_n_all_view,
                    &self.m_map_all_to_local,
                );
                for k in 0..self.av.nfile_views as usize {
                    iv = self.m_map_all_file_to_view[k] - 1;
                    self.m_h[k] = 0.;
                    // Fixed in translation (2026-09-26, `BUGS.md`): the source tests
                    // `iv > 0` on the 0-based index (`tiltalign.cpp:1716`), so the
                    // first view's X-axis tilt is always written as 0.
                    if iv >= 0 {
                        self.m_h[k] = self.m_grad[iv as usize];
                    }
                }
                for k in 0..self.av.nfile_views {
                    fprintf!(iun_local, " %6.2f", CArg::Dbl(self.m_h[k as usize] as f64));
                    if k % 10 == 9 || k == self.av.nfile_views - 1 {
                        fprintf!(iun_local, "\n");
                    }
                }
            }
            //
            // Z factors if they were output globally
            //
            if self.m_if_zfac != 0 {
                expand_local_to_all(
                    &mut self.m_xz_fac,
                    1,
                    1,
                    self.av.nview,
                    self.m_n_all_view,
                    &self.m_map_all_to_local,
                );
                expand_local_to_all(
                    &mut self.m_yz_fac,
                    1,
                    1,
                    self.av.nview,
                    self.m_n_all_view,
                    &self.m_map_all_to_local,
                );
                for k in 0..self.av.nfile_views as usize {
                    iv = self.m_map_all_file_to_view[k] - 1;
                    self.m_h[k] = self.m_glb_xz_fac[k];
                    self.m_grad[k] = self.m_glb_yz_fac[k];
                    if iv >= 0 {
                        self.m_h[k] = self.m_xz_fac[iv as usize];
                        self.m_grad[k] = self.m_yz_fac[iv as usize];
                    }
                }
                for k in 0..self.av.nfile_views {
                    fprintf!(
                        iun_local,
                        " %11.6f %11.6f",
                        CArg::Dbl(self.m_h[k as usize] as f64),
                        CArg::Dbl(self.m_grad[k as usize] as f64)
                    );
                    if k % 3 == 2 || k == self.av.nfile_views - 1 {
                        fprintf!(iun_local, "\n");
                    }
                }
            }
            //
            // add the shifts to the dx and dy to get transforms that
            // work to get back to the original point positions.
            // Compose the inverse of an adjusting transform
            //
            for k in 0..self.av.nview as usize {
                let b = 6 * k;
                self.m_fl[b + 4] += self.m_x_shift * (self.m_dtor * self.av.tilt[k]).cos();
                self.m_fl[b + 5] += self.m_y_shift;
                // call xfwrite(6, fl(1, 1, iv),*199)
                let g = (6 * (self.m_map_local_to_all[k] - 1)) as usize;
                xf_invert(&self.m_glb_fl[g..g + 6], &mut fa, 2);
                xf_mult(&fa, &self.m_fl[b..b + 6], &mut fb, 2);
                xf_invert(&fb, &mut self.m_fl[b..b + 6], 2);
            }
            for k in 1..=6 {
                expand_local_to_all(
                    &mut self.m_fl,
                    6,
                    k,
                    self.av.nview,
                    self.m_n_all_view,
                    &self.m_map_all_to_local,
                );
            }
            for k in 0..self.av.nfile_views as usize {
                iv = self.m_map_all_file_to_view[k] - 1;
                if iv >= 0 {
                    xf_copy(&self.m_fl[(6 * iv) as usize..], 2, &mut fc, 2);
                } else {
                    xf_unit(&mut fc, 1., 2);
                }
                fprintf!(
                    iun_local,
                    " %11.7f %11.7f %11.7f %11.7f %11.3f %11.3f\n",
                    CArg::Dbl(fc[0] as f64),
                    CArg::Dbl(fc[2] as f64),
                    CArg::Dbl(fc[1] as f64),
                    CArg::Dbl(fc[3] as f64),
                    CArg::Dbl(fc[4] as f64),
                    CArg::Dbl(fc[5] as f64)
                );
            }
            // If we need to restore the maps and isecView (commented out in the
            // source, `tiltalign.cpp:1773-1790`)
        }
        //
        // print out points with high residuals
        // Leave the mean and SD in pixels, they are needed below; do weighted mean scaled
        //
        err_mean = err_sum / self.m_num_proj_pt as f32;
        err_mean_nm = self.m_pixel_size * err_mean;
        err_sd = ((err_sqsm - err_sum * err_sum / self.m_num_proj_pt as f32)
            / (self.m_num_proj_pt - 1) as f32)
            .sqrt();
        if self.m_if_do_robust != 0 {
            wgt_err_mean = (self.m_pixel_size as f64 * wgt_err_sum / wgt_sum) as f32;
        }
        if self.m_if_do_local == 0 {
            printf!(
                "\n Residual error mean and sd:  %s %7.3f %s\n",
                CArg::Str(&formatted_error(err_mean_nm, 0.3, 8)),
                CArg::Dbl((self.m_pixel_size * err_sd) as f64),
                CArg::Str(&self.m_pix_units)
            );
            if self.m_if_do_robust != 0 && self.m_did_robust {
                printf!(
                    " Residual error weighted mean: %s         %s\n",
                    CArg::Str(&formatted_error(wgt_err_mean, 0.3, 7)),
                    CArg::Str(&self.m_pix_units)
                );
            }
        } else if self.m_if_local == 0 {
            printf!(
                "\n Residual error mean and sd:  %s %7.3f %s    (Global)\n",
                CArg::Str(&formatted_error(err_mean_nm, 0.3, 8)),
                CArg::Dbl((self.m_pixel_size * err_sd) as f64),
                CArg::Str(&self.m_pix_units)
            );
            if self.m_if_do_robust != 0 && self.m_did_robust {
                printf!(
                    " Residual error weighted mean: %s         %s    (Global)\n",
                    CArg::Str(&formatted_error(wgt_err_mean, 0.3, 7)),
                    CArg::Str(&self.m_pix_units)
                );
            }
        } else {
            printf!(
                "\n Residual error mean and sd:  %s %7.3f %s    (Local area %3d %3d)\n",
                CArg::Str(&formatted_error(err_mean_nm, 0.3, 8)),
                CArg::Dbl((self.m_pixel_size * err_sd) as f64),
                CArg::Str(&self.m_pix_units),
                CArg::Int(self.m_ipatch_x as i64),
                CArg::Int(self.m_ipatch_y as i64)
            );
            self.m_errsum_local += err_mean_nm;
            self.m_err_local_min = if self.m_err_local_min < err_mean_nm {
                self.m_err_local_min
            } else {
                err_mean_nm
            };
            self.m_err_local_max = if self.m_err_local_max > err_mean_nm {
                self.m_err_local_max
            } else {
                err_mean_nm
            };
            if self.m_if_do_robust != 0 && self.m_did_robust {
                printf!(
                    " Residual error weighted mean: %s         %s    (Local area %3d %3d)\n",
                    CArg::Str(&formatted_error(wgt_err_mean, 0.3, 7)),
                    CArg::Str(&self.m_pix_units),
                    CArg::Int(self.m_ipatch_x as i64),
                    CArg::Int(self.m_ipatch_y as i64)
                );

                self.m_wgt_err_sum_local += wgt_err_mean;
                self.m_wgt_err_local_min = if self.m_wgt_err_local_min < wgt_err_mean {
                    self.m_wgt_err_local_min
                } else {
                    wgt_err_mean
                };
                self.m_wgt_err_local_max = if self.m_wgt_err_local_max > wgt_err_mean {
                    self.m_wgt_err_local_max
                } else {
                    wgt_err_mean
                };
            }
        }
        if self.m_if_res_out > 0 {
            //
            // DEPENDENCY WARNING: Beadfixer relies on the # # ... line up to the
            // second X
            //
            printf!(
                "\n         Projection points with large residuals (in pixels)\n obj  cont  view   index coordinates      residuals        # of\n   #     #     #      X         Y        X        Y        S.D.%s\n",
                CArg::Str(if self.m_if_do_robust != 0 {
                    "   Weights"
                } else {
                    ""
                })
            );

            nord = 0;
            jpt = 1;
            while jpt <= self.av.nreal_pt {
                if self.m_lv_out_num_predict >= 0
                    && self.av.real_in_test_set[(jpt - 1) as usize] != 0
                {
                    jpt += 1;
                    continue;
                }
                i = self.av.ireal_str[(jpt - 1) as usize];
                while i <= self.av.ireal_str[jpt as usize] - 1 {
                    let iu = (i - 1) as usize;
                    let resid = (self.av.xresid[iu] * self.av.xresid[iu]
                        + self.av.yresid[iu] * self.av.yresid[iu])
                        .sqrt();
                    if self.m_nearby_err {
                        iv = self.av.isec_view[iu];
                        err_no_sd = (resid - self.m_view_mean_res[(iv - 1) as usize])
                            / self.m_view_sd_res[(iv - 1) as usize];
                    } else {
                        err_no_sd = (resid - err_mean) / err_sd;
                    }
                    if err_no_sd > self.m_err_crit {
                        if self.m_order_err {
                            nord += 1;
                            self.m_err_save[(nord - 1) as usize] = err_no_sd;
                            self.m_ind_save[(nord - 1) as usize] = i;
                            self.m_jpt_save[(nord - 1) as usize] = jpt;
                        } else {
                            let ia = (self.m_ind_all_real[(jpt - 1) as usize] - 1) as usize;
                            printf!(
                                "%4d %5d %5d %9.2f %9.2f %8.2f %8.2f %8.2f",
                                CArg::Int(self.m_imod_obj[ia] as i64),
                                CArg::Int(self.m_imod_cont[ia] as i64),
                                CArg::Int(
                                    self.av.map_view_to_file[(self.av.isec_view[iu] - 1) as usize]
                                        as i64
                                ),
                                CArg::Dbl((self.av.xx[iu] + self.m_xcen) as f64),
                                CArg::Dbl((self.av.yy[iu] + self.m_ycen) as f64),
                                CArg::Dbl(self.av.xresid[iu] as f64),
                                CArg::Dbl(self.av.yresid[iu] as f64),
                                CArg::Dbl(err_no_sd as f64)
                            );
                            if self.m_if_do_robust != 0 {
                                printf!(" %8.4f", CArg::Dbl(self.av.weight[iu] as f64));
                            }
                            printf!("\n");
                        }
                    }
                    i += 1;
                }
                jpt += 1;
            }
            if self.m_order_err {
                i = 1;
                while i <= nord - 1 {
                    j = i + 1;
                    while j <= nord {
                        let (iu, ju) = ((i - 1) as usize, (j - 1) as usize);
                        if self.m_err_save[iu] < self.m_err_save[ju] {
                            tmp = self.m_err_save[iu];
                            self.m_err_save[iu] = self.m_err_save[ju];
                            self.m_err_save[ju] = tmp;
                            itmp = self.m_ind_save[iu];
                            self.m_ind_save[iu] = self.m_ind_save[ju];
                            self.m_ind_save[ju] = itmp;
                            itmp = self.m_jpt_save[iu];
                            self.m_jpt_save[iu] = self.m_jpt_save[ju];
                            self.m_jpt_save[ju] = itmp;
                        }
                        j += 1;
                    }
                    i += 1;
                }
                iord = 1;
                while iord <= nord {
                    let ou = (iord - 1) as usize;
                    i = self.m_ind_save[ou];
                    let iu = (i - 1) as usize;
                    let ia = (self.m_ind_all_real[(self.m_jpt_save[ou] - 1) as usize] - 1) as usize;
                    printf!(
                        "%4d %5d %5d %9.2f %9.2f %8.2f %8.2f %8.2f",
                        CArg::Int(self.m_imod_obj[ia] as i64),
                        CArg::Int(self.m_imod_cont[ia] as i64),
                        CArg::Int(
                            self.av.map_view_to_file[(self.av.isec_view[iu] - 1) as usize] as i64
                        ),
                        CArg::Dbl((self.av.xx[iu] + self.m_xcen) as f64),
                        CArg::Dbl((self.av.yy[iu] + self.m_ycen) as f64),
                        CArg::Dbl(self.av.xresid[iu] as f64),
                        CArg::Dbl(self.av.yresid[iu] as f64),
                        CArg::Dbl(self.m_err_save[ou] as f64)
                    );
                    if self.m_if_do_robust != 0 {
                        printf!(" %8.4f", CArg::Dbl(self.av.weight[iu] as f64));
                    }
                    printf!("\n");
                    iord += 1;
                }
            }
        }
        //
        // Process fill-in points
        if self.m_make_filled_fid != 0 {
            self.process_fill_in_points();
        }
    }
}

impl TiltAlign {
    /// Original: `TiltAlign::processFillInPoints` (`tiltalign.cpp:1910`).
    ///
    /// Scale filled-in points, adjust them by local residuals, and accumulate in
    /// final spot.
    pub fn process_fill_in_points(&mut self) {
        const NUM_RESID_FIT_FOR_FILL: usize = 6;
        const MIN_RESID_FIT_FOR_FILL: i32 = 6;
        let mut num_fit: i32;
        let mut min_view: i32;
        let mut max_view: i32;
        let mut iv: i32;
        let mut jpt: i32;
        let mut i: i32;
        let mut j: i32;
        let mut xview_fit = [0f32; NUM_RESID_FIT_FOR_FILL];
        let mut xres_fit = [0f32; NUM_RESID_FIT_FOR_FILL];
        let mut yres_fit = [0f32; NUM_RESID_FIT_FOR_FILL];
        let mut xres_adj: f32;
        let mut yres_adj: f32;
        let mut slope: f32 = 0.;
        let mut bint: f32 = 0.;
        let mut rval: f32 = 0.;
        let mut xf_proj: f32;
        let mut yf_proj: f32;
        let mut found: bool;
        jpt = 1;
        while jpt <= self.av.nreal_pt {
            i = self.av.ifill_real_start[(jpt - 1) as usize];
            while i <= self.av.ifill_real_start[jpt as usize] - 1 {
                let iu = (i - 1) as usize;
                //
                // Find nearest points for fitting residuals
                num_fit = 0;
                min_view = self.av.nview + 1;
                max_view = 0;
                found = false;
                // DIFF_LOOP:
                iv = 1;
                while iv <= self.av.nview && !found {
                    j = self.av.ireal_str[(jpt - 1) as usize];
                    while j <= self.av.ireal_str[jpt as usize] - 1 {
                        let ju = (j - 1) as usize;
                        if b3dabs!(self.av.isec_view[ju] - self.av.ifill_view[iu]) == iv {
                            num_fit += 1;
                            xview_fit[(num_fit - 1) as usize] = self.av.isec_view[ju] as f32;
                            xres_fit[(num_fit - 1) as usize] = self.av.xresid[ju];
                            yres_fit[(num_fit - 1) as usize] = self.av.yresid[ju];
                            min_view = if min_view < self.av.isec_view[ju] {
                                min_view
                            } else {
                                self.av.isec_view[ju]
                            };
                            max_view = if max_view > self.av.isec_view[ju] {
                                max_view
                            } else {
                                self.av.isec_view[ju]
                            };
                            if num_fit == NUM_RESID_FIT_FOR_FILL as i32 {
                                found = true;
                                break; // DIFF_LOOP;
                            }
                        }
                        j += 1;
                    }
                    iv += 1;
                } // DIFF_LOOP
                //
                // Do fit if there are enough points
                xres_adj = 0.;
                yres_adj = 0.;
                if num_fit >= MIN_RESID_FIT_FOR_FILL {
                    if self.av.ifill_view[iu] < min_view {
                        iv = min_view;
                    } else if self.av.ifill_view[iu] > max_view {
                        iv = max_view;
                    } else {
                        iv = self.av.ifill_view[iu];
                    }
                    ls_fit(
                        &xview_fit, &xres_fit, num_fit, &mut slope, &mut bint, &mut rval,
                    );
                    xres_adj = bint + slope * iv as f32;
                    ls_fit(
                        &xview_fit, &yres_fit, num_fit, &mut slope, &mut bint, &mut rval,
                    );
                    yres_adj = bint + slope * iv as f32;
                }
                //
                // Get the projection point and either add it or replace a global one
                xf_proj = self.av.xfill_proj[iu] * self.m_scale_xy + self.m_xcen - xres_adj;
                yf_proj = self.av.yfill_proj[iu] * self.m_scale_xy + self.m_ycen - yres_adj;
                j = self.m_map_fill_local_to_all[iu];
                let ju = (j - 1) as usize;
                if self.m_num_in_fill_sum[ju] < 0 {
                    self.m_xfill_final[ju] = xf_proj;
                    self.m_yfill_final[ju] = yf_proj;
                    self.m_num_in_fill_sum[ju] = 1;
                } else {
                    self.m_xfill_final[ju] += xf_proj;
                    self.m_yfill_final[ju] += yf_proj;
                    self.m_num_in_fill_sum[ju] += 1;
                }
                i += 1;
            }
            jpt += 1;
        }
    }

    /// Original: `TiltAlign::setupAndDoLocalAlignments` (`tiltalign.cpp:1982`).
    ///
    /// Get parameters for doing local alignments, set them up, and loop on the
    /// local areas.
    pub fn setup_and_do_local_alignments(&mut self) {
        let mut ll: i32;
        let mut nxp_min: i32;
        let mut nyp_min: i32;
        let mut ny_target: i32;
        let mut nx_target: i32;
        let mut tot_num_in_areas: i32;
        let mut num_pt_in_area: i32 = 0;
        let ixs_patch: i32;
        let iys_patch: i32;
        let mut ix_patch: i32;
        let mut iy_patch: i32;
        let mut kk: i32;
        let mut nxp: i32;
        let mut nyp: i32;
        let mut iv: i32;
        let idx_patch: i32;
        let idy_patch: i32;
        let mut min_fid_tot: i32;
        let mut min_fid_surf: i32;
        let mut if_xyz_fix: i32;
        let mut min_surf: i32 = 0;
        let mut localv: i32;
        let itmp: i32;
        let mut j: i32;
        let mut i: i32;
        let mut all_ymax: f32;
        let mut all_xmin: f32;
        let mut all_xmax: f32;
        let mut all_ymin: f32;
        let mut ypmin: f32;
        let mut xp_min: f32;
        let mut xsum: f32;
        let mut ysum: f32;
        let mut zsum: f32;
        let mut fixed_dum: f32 = 0.;
        let all_border: f32;
        let use_target: bool;
        let mut temp_name: Vec<u8> = Vec::new();

        self.m_if_do_local = self.m_if_local;
        self.m_errsum_local = 0.;
        self.m_err_local_min = 1.0e10;
        self.m_err_local_max = -10.;
        self.m_wgt_err_sum_local = 0.;
        self.m_wgt_err_local_min = 1.0e10;
        self.m_wgt_err_local_max = -10.;
        self.m_local_mu_ratio_sum = 0.;
        self.m_min_local_mu_ratio = 1.0e10;
        self.m_max_local_mu_ratio = -10.;
        self.m_if_var_out = 0;
        self.m_if_res_out = 0;
        self.m_if_xyz_out = 0;
        self.m_num_patch_x = 5;
        self.m_num_patch_y = 5;
        xp_min = 0.5;
        ypmin = 0.5;
        min_fid_tot = 8;
        min_fid_surf = 3;
        if_xyz_fix = 0;
        clear_leave_out_errors(&mut self.av);
        nx_target = 700;
        ny_target = 700;
        if pip_get_string(b"OutputLocalFile", &mut temp_name) != 0 {
            error_exit::<false>("No output file for local transforms specified", 0);
        }
        itmp = pip_get_two_integers(
            b"NumberOfLocalPatchesXandY",
            &mut self.m_num_patch_x,
            &mut self.m_num_patch_y,
        );
        kk = pip_get_two_integers(b"TargetPatchSizeXandY", &mut nx_target, &mut ny_target);
        pip_get_two_floats(b"MinSizeOrOverlapXandY", &mut xp_min, &mut ypmin);
        pip_get_two_integers(
            b"MinFidsTotalAndEachSurface",
            &mut min_fid_tot,
            &mut min_fid_surf,
        );
        pip_get_boolean(b"FixXYZCoordinates", &mut if_xyz_fix);
        pip_get_three_integers(
            b"LocalOutputOptions",
            &mut self.m_if_var_out,
            &mut self.m_if_xyz_out,
            &mut self.m_if_res_out,
        );
        if itmp == 0 && kk == 0 {
            error_exit::<false>(
                "You cannot enter both a number of local patches and a target size",
                0,
            );
        }
        use_target = itmp > 0;
        if (use_target || self.m_fit_num_patches_in_fid_area) && (xp_min > 1. || ypmin > 1.) {
            if use_target {
                error_exit::<false>(
                    "You cannot enter a minimum patch size with a target size",
                    0,
                );
            } else {
                error_exit::<false>(
                    "You can no longer enter a minimum patch size with numbers of patches",
                    0,
                );
            }
        }
        //
        self.m_iun_local = Some(b3d_open_file(&String::from_utf8_lossy(&temp_name), "w"));
        self.m_if_local = 1;
        self.av.xyz_fixed = (if_xyz_fix != 0) as i32;
        //
        // set for incremental solution - could be input as option at this point
        //
        self.av.incr_dmag = 1;
        self.av.incr_gmag = 1;
        self.av.incr_skew = 1;
        self.av.incr_tilt = 1;
        self.av.incr_rot = 1;
        self.av.incr_alf = 1;
        //
        // save transforms, scale angles back to radians (global solution
        // already saved)
        //
        for k in 0..self.av.nview as usize {
            xf_copy(&self.m_fl[6 * k..], 2, &mut self.m_glb_fl[6 * k..], 2);
            self.av.tilt[k] = self.av.glb_tilt[k];
        }
        for k in 0..self.av.nfile_views as usize {
            self.m_map_all_file_to_view[k] = self.av.map_file_to_view[k];
        }
        self.m_n_all_view = self.av.nview;
        //
        self.m_n_all_proj_pt = self.m_num_proj_pt;
        for k in 0..self.m_num_proj_pt as usize {
            self.m_all_xx[k] = self.av.xx[k];
            self.m_all_yy[k] = self.av.yy[k];
            self.m_iall_sec_vw[k] = self.av.isec_view[k];
        }
        // write(*,121)
        // 121    format(/,11x,'Absolute 3-D point coordinates' &
        // ,/,'   #',7x,'X',9x,'Y',9x,'Z')
        // write(*,'(i4,3f10.2)', err = 86) &
        // (j, (allxyz(i, j), i = 1, 3), j = 1, nrealPt)

        if use_target || self.m_fit_num_patches_in_fid_area {
            //
            // If using a target or for newer behavior, use the real extent of the data,
            all_xmin = 1.0e10;
            all_xmax = -1.0e10;
            all_ymin = 1.0e10;
            all_ymax = -1.0e10;
            all_border = 5.;
            for k in 0..self.m_n_all_real_pt as usize {
                let v = self.m_all_xyz[k * 3] - all_border;
                all_xmin = if all_xmin < v { all_xmin } else { v };
                let v = self.m_all_xyz[k * 3] + all_border;
                all_xmax = if all_xmax > v { all_xmax } else { v };
                let v = self.m_all_xyz[k * 3 + 1] - all_border;
                all_ymin = if all_ymin < v { all_ymin } else { v };
                let v = self.m_all_xyz[k * 3 + 1] + all_border;
                all_ymax = if all_ymax > v { all_ymax } else { v };
            }
            //
            // get the number of patches that fill the extent at the target size, then get the
            // real size with the defined overlap
            if use_target {
                self.m_num_patch_x = (((all_xmax - all_xmin - nx_target as f32) as f64)
                    / (nx_target as f64 * (1. - xp_min as f64))
                    + 1.) as i32;
                self.m_num_patch_y = (((all_ymax - all_ymin - ny_target as f32) as f64)
                    / (ny_target as f64 * (1. - ypmin as f64))
                    + 1.) as i32;
            } else {
                self.m_num_patch_x = ((self.m_num_patch_x as f32 * (all_xmax - all_xmin)) as f64
                    / (2. * self.m_xcen as f64)
                    + 0.75)
                    .floor() as i32;
                self.m_num_patch_y = ((self.m_num_patch_y as f32 * (all_ymax - all_ymin)) as f64
                    / (2. * self.m_ycen as f64)
                    + 0.75)
                    .floor() as i32;
            }
            self.m_num_patch_x = if 2 > self.m_num_patch_x {
                2
            } else {
                self.m_num_patch_x
            };
            self.m_num_patch_y = if 2 > self.m_num_patch_y {
                2
            } else {
                self.m_num_patch_y
            };
            nxp_min = ((all_xmax - all_xmin)
                / (self.m_num_patch_x as f32 - xp_min * (self.m_num_patch_x - 1) as f32))
                as i32;
            if use_target && nxp_min as f64 > 1.05 * nx_target as f64 {
                self.m_num_patch_x += 1;
                nxp_min = ((all_xmax - all_xmin)
                    / (self.m_num_patch_x as f32 - xp_min * (self.m_num_patch_x - 1) as f32))
                    as i32;
            }

            //
            nyp_min = ((all_ymax - all_ymin)
                / (self.m_num_patch_y as f32 - ypmin * (self.m_num_patch_y - 1) as f32))
                as i32;
            if use_target && nyp_min as f64 > 1.05 * ny_target as f64 {
                self.m_num_patch_y += 1;
                nyp_min = ((all_ymax - all_ymin)
                    / (self.m_num_patch_y as f32 - ypmin * (self.m_num_patch_y - 1) as f32))
                    as i32;
            }

            let nx1 = self.m_num_patch_x - 1;
            idx_patch = ((all_xmax - all_xmin - nxp_min as f32)
                / (if 1 > nx1 { 1 } else { nx1 }) as f32
                + 1.) as i32;
            ixs_patch = (all_xmin + (nxp_min / 2) as f32) as i32;
            let ny1 = self.m_num_patch_y - 1;
            idy_patch = ((all_ymax - all_ymin - nyp_min as f32)
                / (if 1 > ny1 { 1 } else { ny1 }) as f32
                + 1.) as i32;
            iys_patch = (all_ymin + (nyp_min / 2) as f32) as i32;
            printf!(
                "Extent of fiducials is %5d and %5d pixels in X and Y\nDoing %2d by %2d local areas, minimum size %4d x%5d\n",
                CArg::Int(b3dnint!((all_xmax - all_xmin) as f64 - 2. * all_border as f64) as i64),
                CArg::Int(b3dnint!((all_ymax - all_ymin) as f64 - 2. * all_border as f64) as i64),
                CArg::Int(self.m_num_patch_x as i64),
                CArg::Int(self.m_num_patch_y as i64),
                CArg::Int(nxp_min as i64),
                CArg::Int(nyp_min as i64)
            );
            let _ = ImodFile::Stdout.flush();
        } else {
            //
            // legacy behavior with # of patches entered: get the minimum patch
            // size from full size of image area
            //
            self.m_num_patch_x = if 2 > self.m_num_patch_x {
                2
            } else {
                self.m_num_patch_x
            };
            self.m_num_patch_y = if 2 > self.m_num_patch_y {
                2
            } else {
                self.m_num_patch_y
            };
            if xp_min > 1. {
                nxp_min = xp_min as i32;
            } else {
                nxp_min = (2. * self.m_xcen
                    / (self.m_num_patch_x as f32 - xp_min * (self.m_num_patch_x - 1) as f32))
                    as i32;
            }
            if ypmin > 1. {
                nyp_min = ypmin as i32;
            } else {
                nyp_min = (2. * self.m_ycen
                    / (self.m_num_patch_y as f32 - ypmin * (self.m_num_patch_y - 1) as f32))
                    as i32;
            }
            //
            // set up starting patch locations and intervals
            //
            let nx1 = self.m_num_patch_x - 1;
            let ny1 = self.m_num_patch_y - 1;
            idx_patch = (b3dnint!(2. * self.m_xcen) - nxp_min) / if 1 > nx1 { 1 } else { nx1 };
            idy_patch = (b3dnint!(2. * self.m_ycen) - nyp_min) / if 1 > ny1 { 1 } else { ny1 };
            ixs_patch = nxp_min / 2;
            iys_patch = nyp_min / 2;
        }
        //
        // If there are filled in points, change sums from 1 to -1 so globals can be replaced
        if self.m_make_filled_fid != 0 {
            for k in 0..self.m_num_fillin_pts as usize {
                self.m_num_in_fill_sum[k] = -self.m_num_in_fill_sum[k];
            }
        }

        // Handle setting the number of leave-out runs for the local areas based
        if self.m_lv_out_num_predict >= 0 {
            tot_num_in_areas = 0;

            // Get the total number of points included in all the areas
            self.m_ipatch_y = 1;
            while self.m_ipatch_y <= self.m_num_patch_y {
                self.m_ipatch_x = 1;
                while self.m_ipatch_x <= self.m_num_patch_x {
                    ix_patch = ixs_patch + (self.m_ipatch_x - 1) * idx_patch;
                    iy_patch = iys_patch + (self.m_ipatch_y - 1) * idy_patch;
                    nxp = nxp_min - 40;
                    nyp = nyp_min - 40;
                    if self.find_local_size_and_points(
                        ix_patch,
                        iy_patch,
                        min_fid_tot,
                        min_fid_surf,
                        &mut nxp,
                        &mut nyp,
                        &mut min_surf,
                    ) != 0
                    {
                        return;
                    }
                    count_num_in_view(
                        &self.av,
                        &self.m_ind_all_real,
                        self.av.nreal_pt,
                        &self.m_iall_real_str,
                        &self.m_iall_sec_vw,
                        self.m_n_all_view,
                        &mut self.m_num_in_view,
                        (!self.m_all_real_in_test_set.is_empty())
                            .then_some(&self.m_all_real_in_test_set[..]),
                        Some(&mut num_pt_in_area),
                    );
                    tot_num_in_areas += num_pt_in_area;
                    self.m_ipatch_x += 1;
                }
                self.m_ipatch_y += 1;
            }

            // reduce a fixed coverage by the excess of points in runs over actual points
            // or base a coverage on a target and the total in the runs
            // Fixed in translation (2026-09-26, `BUGS.md`): the source indexes
            // `mIallRealStr[av->nrealPt]` (`tiltalign.cpp:2175`), where `nrealPt` is
            // the count in the *last* local area; the total, `mNAllRealPt`, is used.
            self.m_lv_out_coverage = (self.m_lv_out_target_cover
                * self.m_iall_real_str[self.m_n_all_real_pt as usize] as f32)
                / tot_num_in_areas as f32;
            if self.m_lv_out_target_cover >= 100. {
                self.m_lv_out_coverage = self.m_lv_out_target_cover / tot_num_in_areas as f32;
                self.m_lv_out_coverage = if self.m_lv_out_coverage < self.m_lv_out_max_coverage {
                    self.m_lv_out_coverage
                } else {
                    self.m_lv_out_max_coverage
                };
                if self.m_lv_out_coverage < self.m_lv_out_min_coverage {
                    // But if coverage is too low, don't apply the full minimum coverage past a
                    // two-fold excess, compromise between minimum coverage and target #
                    if 2. * (self.m_lv_out_coverage as f64) < self.m_lv_out_min_coverage as f64 {
                        self.m_lv_out_coverage = (2.
                            * self.m_lv_out_coverage as f64
                            * self.m_lv_out_min_coverage as f64)
                            .sqrt() as f32;
                    } else {
                        self.m_lv_out_coverage =
                            if self.m_lv_out_coverage > self.m_lv_out_min_coverage {
                                self.m_lv_out_coverage
                            } else {
                                self.m_lv_out_min_coverage
                            };
                    }
                }
            }
            let runs: f64 = (self.m_lv_out_coverage / self.m_frac_predict) as f64 + 0.5;
            self.m_lv_out_num_runs = (if 1. > runs { 1. } else { runs }) as i32;
        }

        //
        // LOOP ON LOCAL REGIONS
        //
        self.m_ipatch_y = 1;
        while self.m_ipatch_y <= self.m_num_patch_y {
            self.m_ipatch_x = 1;
            while self.m_ipatch_x <= self.m_num_patch_x {
                ix_patch = ixs_patch + (self.m_ipatch_x - 1) * idx_patch;
                iy_patch = iys_patch + (self.m_ipatch_y - 1) * idy_patch;
                nxp = nxp_min - 40;
                nyp = nyp_min - 40;
                if self.find_local_size_and_points(
                    ix_patch,
                    iy_patch,
                    min_fid_tot,
                    min_fid_surf,
                    &mut nxp,
                    &mut nyp,
                    &mut min_surf,
                ) != 0
                {
                    return;
                }
                //
                // Get count of points in each view so empty views can be eliminated, then
                // get mapping from all views to remaining views in this local area
                count_num_in_view(
                    &self.av,
                    &self.m_ind_all_real,
                    self.av.nreal_pt,
                    &self.m_iall_real_str,
                    &self.m_iall_sec_vw,
                    self.m_n_all_view,
                    &mut self.m_num_in_view,
                    (!self.m_all_real_in_test_set.is_empty())
                        .then_some(&self.m_all_real_in_test_set[..]),
                    None,
                );
                localv = 0;
                iv = 1;
                while iv <= self.av.nfile_views {
                    self.av.map_file_to_view[(iv - 1) as usize] = 0;
                    iv += 1;
                }
                iv = 1;
                while iv <= self.m_n_all_view {
                    let ivu = (iv - 1) as usize;
                    if self.m_num_in_view[ivu] > 0 {
                        localv += 1;
                        let lu = (localv - 1) as usize;
                        self.m_map_all_to_local[ivu] = localv;
                        self.m_map_local_to_all[lu] = iv;
                        self.av.map_view_to_file[lu] = self.m_map_all_view_to_file[ivu];
                        self.av.map_file_to_view[(self.m_map_all_view_to_file[ivu] - 1) as usize] =
                            localv;
                        self.av.tilt[lu] = self.av.glb_tilt[ivu];
                    } else {
                        self.m_map_all_to_local[ivu] = 0;
                    }
                    iv += 1;
                }
                self.av.nview = localv;
                //
                // Now load the coordinate data with these local view numbers
                self.m_num_proj_pt = 0;
                ll = 0;
                while ll < self.av.nreal_pt {
                    let lu = ll as usize;
                    self.m_list_real[lu] = ll + 1;
                    i = self.m_ind_all_real[lu] - 1;
                    self.av.ireal_str[lu] = self.m_num_proj_pt + 1;
                    if self.av.apply_extra_weights != 0 {
                        self.av.imod_obj_num[lu] = self.m_imod_obj[i as usize];
                    }
                    kk = self.m_iall_real_str[i as usize] - 1;
                    while kk < self.m_iall_real_str[(i + 1) as usize] - 1 {
                        let ku = kk as usize;
                        let pu = self.m_num_proj_pt as usize;
                        self.av.xx[pu] = self.m_all_xx[ku];
                        self.av.yy[pu] = self.m_all_yy[ku];
                        self.av.isec_view[pu] =
                            self.m_map_all_to_local[(self.m_iall_sec_vw[ku] - 1) as usize];
                        self.m_num_proj_pt += 1;
                        kk += 1;
                    }
                    ll += 1;
                }
                self.av.ireal_str[self.av.nreal_pt as usize] = self.m_num_proj_pt + 1;
                count_num_in_view(
                    &self.av,
                    &self.m_list_real,
                    self.av.nreal_pt,
                    &self.av.ireal_str,
                    &self.av.isec_view,
                    self.av.nview,
                    &mut self.m_num_in_view,
                    (!self.av.real_in_test_set.is_empty()).then_some(&self.av.real_in_test_set[..]),
                    None,
                );

                if self.av.test_set_frac_step > 0. {
                    i = 0;
                    while i < self.av.nreal_pt {
                        let iu = i as usize;
                        self.av.real_left_out[iu] = self.av.real_in_test_set[iu];
                        j = self.av.ireal_str[iu] - 1;
                        while j < self.av.ireal_str[iu + 1] - 1 {
                            self.av.weight[j as usize] = if self.av.real_in_test_set[iu] != 0 {
                                0.
                            } else {
                                1.
                            };
                            j += 1;
                        }
                        i += 1;
                    }
                }

                input_vars(
                    &mut self.av,
                    &self.mx,
                    &mut self.sg,
                    &mut self.m_var,
                    &mut self.m_var_name,
                    &mut self.m_nvar_search,
                    &mut self.m_nvar_angle,
                    &mut self.m_nvar_scaled,
                    &mut self.m_min_tilt_view,
                    &mut self.m_ncomp_search,
                    self.m_if_local,
                    &mut self.m_map_tilt_start,
                    &mut self.m_map_alf_start,
                    &mut self.m_map_alf_end,
                    &mut self.m_if_bt_search,
                    &mut self.m_tilt_orig,
                    &mut self.m_tilt_add,
                    &self.m_num_in_view,
                    self.m_nin_thresh,
                    &mut self.m_rot_entered,
                );
                //
                // DNM 7 / 16 / 04: Add pixel size to local file
                // 2 / 15 / 07: Output after first read of variables
                self.av.if_any_alf = 0;
                if self.m_map_alf_end > self.m_map_alf_start || self.av.beam_tilt != 0. {
                    self.av.if_any_alf = 1;
                }
                if self.m_if_local == 1 {
                    if let Some(iun_local) = self.m_iun_local.as_mut() {
                        fprintf!(
                            iun_local,
                            " %5d %5d %5d %5d %5d %5d %5d %11.5f %3d\n",
                            CArg::Int(self.m_num_patch_x as i64),
                            CArg::Int(self.m_num_patch_y as i64),
                            CArg::Int(ixs_patch as i64),
                            CArg::Int(iys_patch as i64),
                            CArg::Int(idx_patch as i64),
                            CArg::Int(idy_patch as i64),
                            CArg::Int(self.av.if_any_alf as i64),
                            CArg::Dbl(self.m_pixel_delta[0] as f64),
                            CArg::Int(self.m_if_zfac as i64)
                        );
                    }
                }
                self.m_if_local = 2;
                //
                // take care of initializing the mapped variables properly
                // (a commented-out block in the source, `tiltalign.cpp:2272-2287`)
                //
                // reload the geometric variables
                //
                i = self.m_map_tilt_start - 1;
                let _ = i;
                reload_vars(
                    &self.av.glb_rot,
                    &mut self.av.rot,
                    &self.av.map_rot,
                    &self.av.frc_rot,
                    self.av.nview,
                    1,
                    self.m_map_tilt_start - 1,
                    &mut self.m_var,
                    &mut fixed_dum,
                    1,
                    &self.m_map_local_to_all,
                );
                reload_vars(
                    &self.av.glb_tilt,
                    &mut self.av.tilt,
                    &self.av.map_tilt,
                    &self.av.frc_tilt,
                    self.av.nview,
                    self.m_map_tilt_start,
                    self.m_nvar_angle,
                    &mut self.m_var,
                    &mut fixed_dum,
                    self.av.incr_tilt,
                    &self.m_map_local_to_all,
                );
                //
                // if doing tilt incremental, just set tiltInc to the global tilt and
                // all the equations work in map_vars
                //
                if self.av.incr_tilt != 0 {
                    self.av.fixed_tilt2 = 0.;
                    self.av.fixed_tilt = 0.;
                    i = 1;
                    while i <= self.av.nview {
                        self.av.tilt_inc[(i - 1) as usize] = self.av.glb_tilt
                            [(self.m_map_local_to_all[(i - 1) as usize] - 1) as usize];
                        i += 1;
                    }
                }
                i = self.m_nvar_angle + 1;
                j = self.av.map_dmag_start - self.m_ncomp_search - 1;
                let _ = (i, j);
                reload_vars(
                    &self.av.glb_gmag,
                    &mut self.av.gmag,
                    &self.av.map_gmag,
                    &self.av.frc_gmag,
                    self.av.nview,
                    self.m_nvar_angle + 1,
                    self.av.map_dmag_start - self.m_ncomp_search - 1,
                    &mut self.m_var,
                    &mut self.av.fixed_gmag,
                    self.av.incr_gmag,
                    &self.m_map_local_to_all,
                );
                reload_vars(
                    &self.av.glb_dmag,
                    &mut self.av.dmag,
                    &self.av.map_dmag,
                    &self.av.frc_dmag,
                    self.av.nview,
                    self.av.map_dmag_start,
                    self.m_nvar_scaled,
                    &mut self.m_var,
                    &mut self.av.fixed_dmag,
                    self.av.incr_dmag,
                    &self.m_map_local_to_all,
                );
                i = self.m_nvar_scaled + 1;
                j = self.m_map_alf_start - 1;
                let _ = (i, j);
                reload_vars(
                    &self.av.glb_skew,
                    &mut self.av.skew,
                    &self.av.map_skew,
                    &self.av.frc_skew,
                    self.av.nview,
                    self.m_nvar_scaled + 1,
                    self.m_map_alf_start - 1,
                    &mut self.m_var,
                    &mut self.av.fixed_skew,
                    self.av.incr_skew,
                    &self.m_map_local_to_all,
                );
                reload_vars(
                    &self.av.glb_alf,
                    &mut self.av.alf,
                    &self.av.map_alf,
                    &self.av.frc_alf,
                    self.av.nview,
                    self.m_map_alf_start,
                    self.m_nvar_search,
                    &mut self.m_var,
                    &mut self.av.fixed_alf,
                    self.av.incr_alf,
                    &self.m_map_local_to_all,
                );
                //
                // get new scaling and scale projection points
                //
                self.m_scale_xy = 0.;
                i = 1;
                while i <= self.m_num_proj_pt {
                    let ax = b3dabs!(self.av.xx[(i - 1) as usize]);
                    let ay = b3dabs!(self.av.yy[(i - 1) as usize]);
                    let m = if ax > ay { ax } else { ay };
                    self.m_scale_xy = if self.m_scale_xy > m {
                        self.m_scale_xy
                    } else {
                        m
                    };
                    i += 1;
                }
                for k in 0..self.m_num_proj_pt as usize {
                    self.av.xx[k] /= self.m_scale_xy;
                    self.av.yy[k] /= self.m_scale_xy;
                }
                //
                // load the xyz's and shift them to zero mean and scale them down
                // Use the fixed input values if they were provided
                xsum = 0.;
                ysum = 0.;
                zsum = 0.;
                i = 1;
                while i <= self.av.nreal_pt {
                    j = self.m_ind_all_real[(i - 1) as usize];
                    let (d, s) = ((i * 3 - 3) as usize, (j * 3 - 3) as usize);
                    if self.m_fixed_xyz_file.is_some() {
                        self.av.xyz[d] = self.m_fixed_xyz[s] - self.m_xcen;
                        self.av.xyz[d + 1] = self.m_fixed_xyz[s + 1] - self.m_ycen;
                        self.av.xyz[d + 2] = self.m_fixed_xyz[s + 2];
                    } else {
                        self.av.xyz[d] = self.m_all_xyz[s] - self.m_xcen;
                        self.av.xyz[d + 1] = self.m_all_xyz[s + 1] - self.m_ycen;
                        self.av.xyz[d + 2] = self.m_all_xyz[s + 2];
                    }
                    xsum += self.av.xyz[d];
                    ysum += self.av.xyz[d + 1];
                    zsum += self.av.xyz[d + 2];
                    i += 1;
                }
                self.m_x_shift = xsum / self.av.nreal_pt as f32;
                self.m_y_shift = ysum / self.av.nreal_pt as f32;
                self.m_z_shift = zsum / self.av.nreal_pt as f32;
                for k in 0..self.av.nreal_pt as usize {
                    self.av.xyz[3 * k] = (self.av.xyz[3 * k] - self.m_x_shift) / self.m_scale_xy;
                    self.av.xyz[3 * k + 1] =
                        (self.av.xyz[3 * k + 1] - self.m_y_shift) / self.m_scale_xy;
                    self.av.xyz[3 * k + 2] =
                        (self.av.xyz[3 * k + 2] - self.m_z_shift) / self.m_scale_xy;
                }
                //
                // Get subset of points to fill in
                if self.m_make_filled_fid != 0 {
                    self.m_num_fillin_pts = 0;
                    ll = 1;
                    while ll <= self.av.nreal_pt {
                        i = self.m_ind_all_real[(ll - 1) as usize];
                        self.av.ifill_real_start[(ll - 1) as usize] = self.m_num_fillin_pts + 1;
                        kk = self.m_ifill_all_real_str[(i - 1) as usize];
                        while kk <= self.m_ifill_all_real_str[i as usize] - 1 {
                            let local = self.m_map_all_to_local
                                [(self.m_iall_fill_view[(kk - 1) as usize] - 1) as usize];
                            if local > 0 {
                                self.av.ifill_view[self.m_num_fillin_pts as usize] = local;
                                self.m_map_fill_local_to_all[self.m_num_fillin_pts as usize] = kk;
                                self.m_num_fillin_pts += 1;
                            }
                            kk += 1;
                        }
                        ll += 1;
                    }
                    self.av.ifill_real_start[self.av.nreal_pt as usize] = self.m_num_fillin_pts + 1;
                }

                printf!(
                    "\n Doing local area %2d %2d, centered on %4d %4d, size %4d %4d,  %2d %s\n",
                    CArg::Int(self.m_ipatch_x as i64),
                    CArg::Int(self.m_ipatch_y as i64),
                    CArg::Int(ix_patch as i64),
                    CArg::Int(iy_patch as i64),
                    CArg::Int(nxp as i64),
                    CArg::Int(nyp as i64),
                    CArg::Int((self.m_num_bot + self.m_num_top) as i64),
                    CArg::Str(if self.av.patch_track_model != 0 {
                        "full tracks"
                    } else {
                        "fiducials"
                    })
                );
                if min_surf > 0 {
                    printf!(
                        "    (%3d on bottom and %2d on top)\n",
                        CArg::Int(self.m_num_bot as i64),
                        CArg::Int(self.m_num_top as i64)
                    );
                }
                self.m_max_cycles = -b3dabs!(self.m_max_cycles);
                self.align_and_output_results();

                self.do_leave_out_runs();
                self.m_ipatch_x += 1;
            }
            self.m_ipatch_y += 1;
        }
    }

    /// Original: `TiltAlign::findLocalSizeAndPoints` (`tiltalign.cpp:2398`).
    ///
    /// Find the points whose real X and Y coordinates are within the bounds of
    /// the patch; expand the patch if necessary to achieve the minimum number of
    /// fiducials.  Load points from the "all" arrays into the current arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn find_local_size_and_points(
        &mut self,
        ix_patch: i32,
        iy_patch: i32,
        min_fid_tot: i32,
        min_fid_surf: i32,
        nxp: &mut i32,
        nyp: &mut i32,
        min_surf: &mut i32,
    ) -> i32 {
        let mut num_full_used: i32;
        let mut i: i32;
        let mut j: i32;
        let mut kk: i32;
        let mut nreal_before: i32;
        let mut num_bot_before: i32;
        let mut num_top_before: i32;
        num_full_used = 0;
        *min_surf = 0;
        while (*nxp as f32) < 4. * self.m_xcen
            && (*nyp as f32) < 4. * self.m_ycen
            && (num_full_used < min_fid_tot
                || (self.m_num_surface >= 2 && *min_surf < min_fid_surf))
        {
            *nxp += 40;
            *nyp += 40;
            self.av.nreal_pt = 0;
            self.m_num_bot = 0;
            self.m_num_top = 0;
            self.m_ix_min = ix_patch - *nxp / 2;
            self.m_ix_max = self.m_ix_min + *nxp;
            self.m_iy_min = iy_patch - *nyp / 2;
            self.m_iy_max = self.m_iy_min + *nyp;
            if self.av.patch_track_model != 0 {
                num_full_used = 0;
                j = 1;
                while j <= self.av.num_full_patch_tracks {
                    nreal_before = self.av.nreal_pt;
                    num_top_before = self.m_num_top;
                    num_bot_before = self.m_num_bot;
                    kk = self.av.ind_full_track[(j - 1) as usize];
                    while kk <= self.av.ind_full_track[j as usize] - 1 {
                        i = self.av.map_track_to_real[(kk - 1) as usize];
                        self.add_to_local_if_in_range(i);
                        kk += 1;
                    }
                    if self.av.nreal_pt > nreal_before {
                        num_full_used += 1;
                        if self.m_num_top - num_top_before > self.m_num_bot - num_bot_before {
                            self.m_num_top = num_top_before + 1;
                            self.m_num_bot = num_bot_before;
                        } else {
                            self.m_num_top = num_top_before;
                            self.m_num_bot = num_bot_before + 1;
                        }
                    }
                    j += 1;
                }
            } else {
                i = 1;
                while i <= self.m_n_all_real_pt {
                    self.add_to_local_if_in_range(i);
                    i += 1;
                }
                num_full_used = self.m_num_top + self.m_num_bot;
            }
            *min_surf = if self.m_num_bot < self.m_num_top {
                self.m_num_bot
            } else {
                self.m_num_top
            };
        }
        if (*nxp as f32) >= 4. * self.m_xcen && (*nyp as f32) >= 4. * self.m_ycen {
            self.m_too_few_fid = true;
            return 1;
        }
        self.m_num_train_real = num_full_used;
        0
    }

    /// Original: `TiltAlign::addToLocalIfInRange` (`tiltalign.cpp:2455`).
    ///
    /// Adds one real point to a local area if its xyz position is in range.
    /// Fixed in translation (2026-09-26, `BUGS.md`): the source indexes
    /// `mAllRealInTestSet[ireal]` with the 1-based point number
    /// (`tiltalign.cpp:2461,2463`), i.e. the next point's flag, and one past the
    /// array for the last point; the point's own flag, `[ireal - 1]`, is read.
    pub fn add_to_local_if_in_range(&mut self, ireal: i32) {
        let x = self.m_all_xyz[(ireal * 3 - 3) as usize];
        let y = self.m_all_xyz[(ireal * 3 - 2) as usize];
        if x >= self.m_ix_min as f32
            && x <= self.m_ix_max as f32
            && y >= self.m_iy_min as f32
            && y <= self.m_iy_max as f32
        {
            self.m_ind_all_real[self.av.nreal_pt as usize] = ireal;
            if self.m_lv_out_num_predict >= 0 && self.av.test_set_frac_step > 0. {
                self.av.real_in_test_set[self.av.nreal_pt as usize] =
                    self.m_all_real_in_test_set[(ireal - 1) as usize];
            }
            self.av.nreal_pt += 1;
            if self.m_lv_out_num_predict < 0
                || self.av.test_set_frac_step <= 0.
                || self.m_all_real_in_test_set[(ireal - 1) as usize] == 0
            {
                if self.m_num_surface >= 2 {
                    if self.m_igroup[(ireal - 1) as usize] == 1 {
                        self.m_num_bot += 1;
                    }
                    if self.m_igroup[(ireal - 1) as usize] == 2 {
                        self.m_num_top += 1;
                    }
                } else {
                    self.m_num_top += 1;
                }
            }
        }
    }

    /// Original: `TiltAlign::findMedianResidual` (`tiltalign.cpp:2480`).
    ///
    /// Find the median residual of a view, or of a patch track group when doing
    /// robust fitting by whole track.
    pub fn find_median_residual(&mut self) {
        let mut xvfit = [0f32; 25];
        let mut slope = [0f32; 5];
        let mut tmp_res: Vec<f32> = Vec::new();
        let mut tmp_median: Vec<f32>;
        let mut iorder: i32 = 2;
        let n_full_fit: i32 = 15;
        let mut num_iter: i32 = 3;
        let mut iter: i32;
        let mut nfit: i32;
        let num_median: i32;
        let mut j: i32;
        let mut iv: i32;
        let mut ivst: i32;
        let mut ivnd: i32;
        let mut i: i32;
        let mut nin_view_sum: i32;
        let mut nin_track: i32;
        let mut found: bool;
        let mut all_zero = true;

        // Set up number of medians to find and smoothing iterations
        if self.av.patch_track_model != 0 && self.av.robust_by_track != 0 {
            self.av.view_median_res[0] = 1.;
            if self.av.num_track_groups == 1 {
                return;
            }
            num_median = self.av.num_track_groups;
            num_iter = 0;
        } else {
            num_median = self.av.nview;
        }
        tmp_median = vec![0.; num_median as usize];

        // Find medians of either track groups or views
        iv = 1;
        while iv <= num_median {
            nin_view_sum = 0;
            tmp_res.clear();
            i = 0;
            while i < self.av.nreal_pt {
                let iu = i as usize;
                if (!self.av.real_in_test_set.is_empty() && self.av.real_in_test_set[iu] != 0)
                    || (self.av.leaving_out != 0 && self.av.real_left_out[iu] != 0)
                {
                    i += 1;
                    continue;
                }
                //
                // Track group: compute mean residual of each track and save it
                if self.av.patch_track_model != 0 && self.av.robust_by_track != 0 {
                    if self.av.itrack_group[(self.m_ind_all_real[iu] - 1) as usize] == iv {
                        // Fixed in translation (BUGS.md, tiltalign): the source
                        // counts the track here (`ninViewSum += 1`) even when all
                        // of its projections are left out and nothing is added to
                        // `tmpRes`, so the median then reads past the values it
                        // collected.  Defined: a track counts when its mean
                        // residual is added, below.
                        self.av.track_resid[iu] = 0.;
                        nin_track = 0;
                        j = self.av.ireal_str[iu] - 1;
                        while j < self.av.ireal_str[iu + 1] - 1 {
                            let ju = j as usize;
                            if !(self.av.leaving_out != 0 && self.av.proj_left_out[ju] != 0) {
                                self.av.track_resid[iu] += (self.av.xresid[ju]
                                    * self.av.xresid[ju]
                                    + self.av.yresid[ju] * self.av.yresid[ju])
                                    .sqrt();
                                nin_track += 1;
                            }
                            j += 1;
                        }
                        if nin_track != 0 {
                            self.av.track_resid[iu] /= nin_track as f32;
                            tmp_res.push(self.av.track_resid[iu]);
                            nin_view_sum += 1;
                        }
                    }
                } else {
                    //
                    // View: get residual of each point
                    j = self.av.ireal_str[iu] - 1;
                    while j < self.av.ireal_str[iu + 1] - 1 {
                        let ju = j as usize;
                        if self.av.isec_view[ju] == iv
                            && !(self.av.leaving_out != 0 && self.av.proj_left_out[ju] != 0)
                        {
                            nin_view_sum += 1;
                            tmp_res.push(
                                (self.av.xresid[ju] * self.av.xresid[ju]
                                    + self.av.yresid[ju] * self.av.yresid[ju])
                                    .sqrt(),
                            );
                        }
                        j += 1;
                    }
                }
                i += 1;
            }
            rs_fast_median_in_place(
                &mut tmp_res,
                nin_view_sum,
                &mut self.av.view_median_res[(iv - 1) as usize],
            );
            // write(*,'(2i4,f9.3,a)') iterCount, iv, viewMedianRes(iv) * scaleXY, '  FMR'
            if self.av.view_median_res[(iv - 1) as usize] != 0. {
                all_zero = false;
            }
            iv += 1;
        }
        FIND_MEDIAN_ITER_COUNT.fetch_add(1, Ordering::Relaxed);
        if all_zero {
            for k in 0..num_median as usize {
                self.av.view_median_res[k] = 1.;
            }
            return;
        }
        //
        // Make sure there are no zeros, just copy nearest value
        iv = 1;
        while iv <= num_median {
            if self.av.view_median_res[(iv - 1) as usize] == 0. {
                found = false;
                // NEAR_VIEW_LOOP:
                i = 1;
                while i <= num_median - 1 && !found {
                    iter = -1;
                    while iter <= 1 {
                        j = iv + iter * i;
                        if j > 0 && j <= num_median {
                            if self.av.view_median_res[(j - 1) as usize] != 0. {
                                self.av.view_median_res[(iv - 1) as usize] =
                                    self.av.view_median_res[(j - 1) as usize];
                                found = true;
                                break; // NEAR_VIEW_LOOP;
                            }
                        }
                        iter += 2;
                    }
                    i += 1;
                } // NEAR_VIEW_LOOP
            }
            iv += 1;
        }
        //
        // smooth with iterations
        tmp_res.resize(((iorder + 1) * (iorder + 6 + num_median)) as usize, 0.);
        iter = 1;
        while iter <= num_iter {
            copy_array(&mut tmp_median, 1, num_median, &self.av.view_median_res, 1);
            iv = 1;
            while iv <= num_median {
                ivst = if 1 > iv - n_full_fit / 2 {
                    1
                } else {
                    iv - n_full_fit / 2
                };
                ivnd = if num_median < iv + n_full_fit / 2 {
                    num_median
                } else {
                    iv + n_full_fit / 2
                };
                nfit = ivnd + 1 - ivst;
                if nfit < 5 {
                    iorder = 1;
                }
                i = ivst;
                while i <= ivnd {
                    xvfit[(i + 1 - ivst - 1) as usize] = (i - iv) as f32;
                    i += 1;
                }
                polynomial_fit(
                    &xvfit,
                    &tmp_median[(ivst - 1) as usize..],
                    nfit,
                    iorder,
                    &mut slope,
                    &mut self.av.view_median_res[(iv - 1) as usize..],
                    &mut tmp_res,
                );
                // write(*,'(2i4,f9.3,a)') iterCount, iv, viewMedianRes(iv) * scaleXY, '  FMR'
                iv += 1;
            }
            FIND_MEDIAN_ITER_COUNT.fetch_add(1, Ordering::Relaxed);
            iter += 1;
        }
    }

    /// Original: `TiltAlign::doLeaveOutRuns` (`tiltalign.cpp:2597`).
    ///
    /// Does global leave-out runs, or runs on one local area, accumulating
    /// errors after each.
    pub fn do_leave_out_runs(&mut self) {
        let mut num_runs: i32;
        let save_rob_fail: i32;
        let mut save_fail_mess = vec![0u8; ROB_MESS_SIZE + 1];
        if self.m_lv_out_num_predict < 0 {
            return;
        }

        save_rob_fail = self.m_num_rob_failed;
        if self.m_num_rob_failed != 0 {
            save_fail_mess[..ROB_MESS_SIZE].copy_from_slice(&self.m_rob_fail_mess[..ROB_MESS_SIZE]);
        }

        // Scale down point, take subset again
        if self.m_if_local == 0 {
            copy_array(
                &mut self.m_lv_out_save_all_xyz,
                1,
                3 * self.av.nreal_pt,
                &self.m_all_xyz,
                1,
            );
            if self.m_sub_sample_tracks {
                load_patch_subset(
                    &mut self.av,
                    &mut self.m_all_xx,
                    &mut self.m_all_yy,
                    &mut self.m_n_all_proj_pt,
                    &mut self.m_num_proj_pt,
                    &mut self.m_ind_all_real,
                    &mut self.m_n_all_real_pt,
                    &mut self.m_iall_real_str,
                    &mut self.m_iall_sec_vw,
                    &self.m_imod_obj,
                    &mut self.m_num_in_view,
                    self.m_num_sample_target,
                    self.m_min_sampled_in_view,
                );
            }
        }
        for k in 0..self.m_num_proj_pt as usize {
            self.av.xx[k] /= self.m_scale_xy;
            self.av.yy[k] /= self.m_scale_xy;
        }
        for k in 0..self.av.nreal_pt as usize {
            self.av.xyz[3 * k] /= self.m_scale_xy;
            self.av.xyz[3 * k + 1] /= self.m_scale_xy;
        }
        for k in 0..self.av.nview as usize {
            self.av.rot[k] *= self.m_dtor;
            self.av.tilt[k] *= self.m_dtor;
            self.av.skew[k] *= self.m_dtor;
            self.av.alf[k] *= self.m_dtor;
        }

        self.av.leaving_out = 1;
        for k in 0..self.av.nreal_pt as usize {
            self.av.times_left_out[k] = 0;
        }

        // When doing contours, and fraction gives only one per run and coverage is high enough,
        // limit runs to number of points
        num_runs = self.m_lv_out_num_runs;
        if self.m_lv_out_num_predict == 0
            && ((self.m_num_train_real as f32 * self.m_frac_leave_out) as i32) < 2
            && self.m_lv_out_coverage as f64 > 0.95
        {
            num_runs = if self.m_lv_out_num_runs < self.m_num_train_real {
                self.m_lv_out_num_runs
            } else {
                self.m_num_train_real
            };
        }
        printf!(
            "\nRunning %d searches with points left out\n",
            CArg::Int(num_runs as i64)
        );
        for _jpt in 0..num_runs {
            leave_out_points(
                &mut self.av,
                self.m_lv_out_num_predict,
                self.m_lv_out_num_pad,
                self.m_frac_leave_out,
            );
            self.align_and_output_results();
            if self.m_lv_out_num_predict == 0 && (self.m_if_local == 0 || self.av.xyz_fixed == 0) {
                self.m_eval_funct.solve_left_out_xyzs(&mut self.av, false);
            }
            get_leave_out_errors(
                &mut self.av,
                &self.m_weight_none_left_out,
                if self.m_if_do_robust != 0 { 1 } else { 0 },
            );
            if self.av.test_set_frac_step > 0. {
                self.m_eval_funct.solve_left_out_xyzs(&mut self.av, true);
                get_test_set_errors(&mut self.av, if self.m_if_do_robust != 0 { 5 } else { 4 });
            }
        }
        self.av.leaving_out = 0;

        // Restore points
        for k in 0..self.m_num_proj_pt as usize {
            self.av.xx[k] *= self.m_scale_xy;
            self.av.yy[k] *= self.m_scale_xy;
        }
        if self.m_if_local == 0 {
            copy_array(
                &mut self.m_all_xyz,
                1,
                3 * self.av.nreal_pt,
                &self.m_lv_out_save_all_xyz,
                1,
            );
            if self.m_sub_sample_tracks {
                restore_from_patch_sample(
                    &mut self.av,
                    &self.m_all_xx,
                    &self.m_all_yy,
                    self.m_n_all_proj_pt,
                    &mut self.m_num_proj_pt,
                    &mut self.m_all_xyz,
                    &mut self.m_ind_all_real,
                    self.m_n_all_real_pt,
                    &self.m_iall_real_str,
                    &self.m_iall_sec_vw,
                    &self.m_imod_obj,
                    None,
                    &self.m_ss_bfac,
                    &self.m_ss_cfac,
                    &self.m_ss_dfac,
                    &self.m_ss_efac,
                    &self.m_ss_ffac,
                );
            }
        }
        self.m_num_rob_failed = save_rob_fail;
        if self.m_num_rob_failed != 0 {
            self.m_rob_fail_mess[..ROB_MESS_SIZE].copy_from_slice(&save_fail_mess[..ROB_MESS_SIZE]);
        }
    }

    /// Original: `TiltAlign::outputLeaveOutErrors` (`tiltalign.cpp:2676`).
    ///
    /// Print out either the leave-out or the test set errors, for non-robust and
    /// robust if both are present.
    pub fn output_leave_out_errors(&mut self, lc_type: &str, uc_type: &str) {
        let mut err = [0f32; 2];
        let mut rob_err = [0f32; 2];
        let mut ind: i32;
        let mut jnd: i32;
        let num_out: i32 = if self.m_if_do_robust != 0 && self.m_did_robust {
            2
        } else {
            1
        };
        let rob_type: [&str; 3] = ["", "non-robust ", "    robust "];
        let rob_uc: [&str; 3] = ["", "NON-ROBUST ", "ROBUST "];
        let lv_type: [&str; 3] = ["leave-out", " test set", "lv-o test"];
        let lv_uc: [&str; 3] = ["LEAVE-OUT", "TEST SET", "LV-O TEST"];
        let mut do_benefit: bool;

        let num_lty = if self.av.test_set_frac_step > 0. {
            3
        } else {
            1
        };
        for lty in 0..num_lty {
            printf!("\n");
            for knd in 0..num_out {
                jnd = knd + num_out - 1;
                ind = knd + 2 * lty;
                let (ku, ju, iu) = (knd as usize, jnd as usize, ind as usize);
                if self.av.num_lv_out_err[iu] != 0 {
                    err[ku] = ((self.m_pixel_size * self.m_scale_xy) as f64
                        * self.av.lv_out_err_sum[iu]
                        / self.av.num_lv_out_err[iu] as f64) as f32;
                    printf!(
                        " %s %s%s error (%d pts): %s",
                        CArg::Str(lc_type),
                        CArg::Str(rob_type[ju]),
                        CArg::Str(lv_type[lty as usize]),
                        CArg::Int(self.av.num_lv_out_err[iu] as i64),
                        CArg::Str(&formatted_error(err[ku], 0.5, 0))
                    );
                    if self.av.num_lv_out_wgt_err[iu] != 0 && self.m_did_full_robust {
                        rob_err[ku] = ((self.m_pixel_size * self.m_scale_xy) as f64
                            * self.av.lv_out_wgt_sum[iu]
                            / self.av.num_lv_out_wgt_err[iu] as f64)
                            as f32;
                        printf!(
                            " %s  weighted %s",
                            CArg::Str(&self.m_pix_units),
                            CArg::Str(&formatted_error(rob_err[ku], 0.5, 0))
                        );
                    }
                    printf!(" %s\n", CArg::Str(&self.m_pix_units));
                } else if self.m_lv_out_num_predict >= 0 {
                    printf!(
                        "NO %s %s%s ERRORS COMPUTED\n",
                        CArg::Str(uc_type),
                        CArg::Str(rob_uc[ju]),
                        CArg::Str(lv_uc[lty as usize])
                    );
                }
            }

            // Why make user do arithmetic: summarize the benefit
            ind = 2 * lty;
            do_benefit = self.av.num_lv_out_err[ind as usize] != 0
                && self.av.num_lv_out_err[(ind + 1) as usize] != 0;
            if num_out == 2 && do_benefit {
                printf!(
                    "        Benefit from robust fitting: %s %.4f",
                    CArg::Str(if lty != 0 { "" } else { "unweighted" }),
                    CArg::Dbl((err[0] - err[1]) as f64)
                );
                if lty == 0 && do_benefit && self.m_did_full_robust {
                    printf!(
                        "   *weighted %.4f",
                        CArg::Dbl((rob_err[0] - rob_err[1]) as f64)
                    );
                }
                printf!(
                    " %s%s\n",
                    CArg::Str(&self.m_pix_units),
                    CArg::Str(if lty == 0 && do_benefit { "*" } else { "" })
                );
            }
        }
    }
}
