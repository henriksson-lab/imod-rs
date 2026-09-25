//! Translation of `IMOD/imodutil/findsection.cpp` with its class header
//! `IMOD/imodutil/findsection.h` merged in.
//!
//! The C++ `FindSect` class becomes the `FindSect` struct, one method per
//! member function.  The source keeps the instance in a file-level static
//! (`sFindSect`, `findsection.cpp:23`) only so the plain-function callback
//! `amoebaFunc` can reach it; here `dualAmoeba` takes a closure over the
//! instance instead, and `amoebaFunc` is that closure.  Because the instance is
//! a static, every member the constructor does not set starts at zero, which is
//! what `FindSect::new` reproduces.
//!
//! The source's `float *` members that alias into bigger allocations
//! (`mSDs`/`mMeans` per tomogram inside `allSDs`/`allMeans`, `mBoundaries`
//! inside `allBoundaries`, `mColMedians` behind `mColSlice`) become an owned
//! `Vec` plus an element offset.
//!
//! No OpenMP region exists in this unit; the only library region it reaches
//! is `multiBinStats`' per-box sums, which is already thread-independent.

use crate::imod::libcfshr::amoeba::dual_amoeba;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_i_max, balanced_group_limits, c_format_bytes, imod_backup_file,
    imod_prog_name, imod_usage_header,
};
use crate::imod::libcfshr::convexbound::convex_bound;
use crate::imod::libcfshr::histogram::kernel_histogram;
use crate::imod::libcfshr::insidecontour::inside_contour;
use crate::imod::libcfshr::linearxforms::xf_apply;
use crate::imod::libcfshr::multibinstat::{MAX_MBS_SCALES, multi_bin_setup, multi_bin_stats};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_non_option_arg,
    pip_get_string, pip_get_three_integers, pip_get_two_floats, pip_get_two_integers,
    pip_number_of_entries, pip_read_or_parse_options,
};
use crate::imod::libcfshr::percentile::percentile_float;
use crate::imod::libcfshr::regression::{mult_regress, robust_regress};
use crate::imod::libcfshr::robuststat::{
    rs_fast_madn, rs_fast_median, rs_fast_median_in_place, rs_median_of_sorted,
    rs_percentile_of_sorted, rs_sort_floats,
};
use crate::imod::libcfshr::simplestat::{array_min_max_mean, avg_sd, ls_fit, ls_fit2};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_get_scale};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_mrc_header, iiu_open, iiu_read_sec_part, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_tilt, iiu_create_header, iiu_ret_basic_head,
    iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt, iiu_write_header_str,
};
use crate::imod::libimod::icont::{imod_contour_area, imod_contour_new};
use crate::imod::libimod::imat::{
    Axis3, Imat, imod_mat_id, imod_mat_new, imod_mat_rot, imod_mat_transform,
};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, IMOD_UNIT_NM, IMODF_OTRANS_ORIGIN, Icont, Imod, Ipoint,
    Iref_image, imod_delete, imod_new, imod_new_contour, imod_new_object, imod_set_ref_image,
    imod_trans_from_ref_image,
};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_PLANAR, IMOD_OBJFLAG_PNT_ON_SEC, IMOD_OBJFLAG_TIME, IOBJ_EX_PNT_LIMIT,
};
use crate::imod::libimod::ipoint::imod_point_append_xyz;
use std::io::Write as _;

/// `findsection.h:7`.
const MAX_FIT_COL: usize = 8;
/// `findsection.h:8`.
const MAX_FIT_DATA: usize = 50;
/// `findsection.cpp:20`.
const MAX_DEFAULT_SCALES: i32 = 12;
/// `findsection.cpp:2148`.
const MAX_AMVAR: usize = 2;
/// `findsection.cpp:2149`.
const MAX_MULT: usize = 32;
/// `iobj.h:76`.
const IOBJ_SYM_CIRCLE: u8 = 0;
/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;
const B3D_X: usize = 0;
const B3D_Y: usize = 1;
const B3D_Z: usize = 2;

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// The source's `printf`, through the C-format writer on libc-order stdout.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// `class FindSect` (`findsection.h:10-112`).
pub struct FindSect {
    m_num_boxes: [[i32; 3]; MAX_MBS_SCALES],
    m_binning: [[i32; 3]; MAX_MBS_SCALES],
    m_box_spacing: [[i32; 3]; MAX_MBS_SCALES],
    /// `mSDs`: `allSDs` in the source, addressed from `m_sds_off`.
    m_sds: Vec<f32>,
    m_sds_off: usize,
    m_buffer: Vec<f32>,
    /// `mMeans`: `allMeans` in the source, addressed from `m_means_off`.
    m_means: Vec<f32>,
    m_means_off: usize,
    /// `mColSlice`, with `mColMedians = mColSlice + maxColSlice` as an offset;
    /// `None` is the source's `mColMedians == NULL`.
    m_col_slice: Vec<f32>,
    m_col_medians: Option<usize>,
    m_stat_start_inds: [i32; MAX_MBS_SCALES + 1],
    m_edge_medians: [f32; MAX_MBS_SCALES],
    m_edge_madns: [f32; MAX_MBS_SCALES],
    m_cen_medians: [f32; MAX_MBS_SCALES],
    m_cen_madns: [f32; MAX_MBS_SCALES],
    m_num_in_block: [[i32; 3]; MAX_MBS_SCALES],
    m_num_blocks: [[i32; 3]; MAX_MBS_SCALES],
    m_best_low_edge: [i32; MAX_MBS_SCALES],
    m_best_high_edge: [i32; MAX_MBS_SCALES],
    /// `float mFitMat[MAX_FIT_COL][MAX_FIT_DATA]`, row-major as in C.
    m_fit_mat: [f32; MAX_FIT_COL * MAX_FIT_DATA],
    m_fit_work: [f32; 2 * MAX_FIT_DATA + MAX_FIT_COL * MAX_FIT_COL],
    m_good_row_col: [i32; 2],
    m_var_list: [[i32; 2]; 5],
    m_nxyz: [i32; 3],
    m_num_vars: i32,
    m_num_thicknesses: i32,
    m_thick_median: f32,
    m_thick_madn: f32,
    /// `mBoundaries`: `allBoundaries` in the source, addressed from
    /// `m_bound_off`.
    m_boundaries: Vec<f32>,
    m_bound_off: usize,
    m_block_centers: Vec<f32>,
    m_thicknesses: Vec<f32>,
    m_bound_rot: Vec<f32>,
    m_proj_means: Vec<f32>,
    m_proj_slice: Vec<f32>,
    /// `int *mBestBoxSize`, which points at `boxSize[mBestScale]` in `main`.
    m_best_box_size: [i32; 3],
    /// `int *mStartCoord`, which points at `main`'s `startCoord`.
    m_start_coord: [i32; 3],
    m_num_high_sd: i32,
    m_num_beads: i32,
    m_min_num_thick_for_check: i32,
    m_crit_thick_madn: f32,
    m_bead_diameter: f32,
    m_simplex_iter: i32,
    m_after_simplex: bool,
    m_boost_high_sd_thickness: f32,
    m_bound_low: f32,
    m_bound_high: f32,
    m_bead_low: f32,
    m_bead_high: f32,
    m_min_spread: f32,
    m_spread_norms: [f32; 3],
    m_bead_weight_fac: f32,
    m_bead_area: f32,
    m_struct_area: f32,
    m_convex_xtmp: Vec<f32>,
    m_convex_ytmp: Vec<f32>,
    m_convex_cont: Option<Icont>,
    m_col_max_edge_diff_crit: f32,
    m_frac_col_max_edge_diff: f32,
    m_crit_edge_madn: f32,
    m_num_high_inside_crit: i32,
    m_thick_ind: usize,
    m_fit_pitch_separately: i32,
    m_debug_output: i32,
    m_scan_block_size: i32,
    m_best_scale: usize,
    m_kfactor: f32,
    m_max_change: f32,
    m_max_oscill: f32,
    m_max_iter: i32,
    m_max_falloff_frac: f32,
    m_low_fit_frac: f32,
    m_high_fit_frac: f32,
    m_boundary_frac: f32,
    m_pitch_boundary_frac: f32,
    m_min_frac_bounds_in_col: f32,
    m_column_to_cen_madn_crit: f32,
    m_too_thin_crit: f32,
    m_farther_from_mean_crit: f32,
    m_mean_diff_thick_frac: f32,
    m_num_extra_sum: i32,
    m_extra_for_pitch: f32,
    m_extra_pitch_sum: f32,
    m_min_for_robust_pitch: i32,
    m_bead_excl_pctl_spread: f32,
    m_bead_excl_pctl_thick: f32,
    m_proj_layer_stat_type: i32,
    m_proj_edge_crit: f32,
    m_proj_use_extrap: i32,
    m_use_proj_for_spread: i32,
}

/// C `main` (`findsection.cpp:28`).
pub fn findsection(arguments: &[String]) -> i32 {
    let mut find_sect = FindSect::new();
    find_sect.main(arguments);
    let _ = ImodFile::Stdout.flush();
    crate::imod::libcfshr::b3dutil::exit(0);
}

/// C `getSlice` (`findsection.cpp:2934`), the `multiBinStats` callback that
/// loads a needed slice.
fn get_slice(iz: &mut i32, fdata: &mut [i32], buffer: &mut [f32]) -> i32 {
    unsafe {
        iiu_set_position(fdata[0], *iz, 0);
        iiu_read_sec_part(
            fdata[0],
            buffer.as_mut_ptr().cast(),
            fdata[2] + 1 - fdata[1],
            fdata[1],
            fdata[2],
            fdata[3],
            fdata[4],
        )
    }
}

impl FindSect {
    /// `FindSect::FindSect` (`findsection.cpp:37`).  The fields the
    /// constructor does not set start at zero, as they do for the source's
    /// static instance.
    pub fn new() -> Self {
        let mut s = FindSect {
            m_num_boxes: [[0; 3]; MAX_MBS_SCALES],
            m_binning: [[0; 3]; MAX_MBS_SCALES],
            m_box_spacing: [[0; 3]; MAX_MBS_SCALES],
            m_sds: Vec::new(),
            m_sds_off: 0,
            m_buffer: Vec::new(),
            m_means: Vec::new(),
            m_means_off: 0,
            m_col_slice: Vec::new(),
            m_col_medians: None,
            m_stat_start_inds: [0; MAX_MBS_SCALES + 1],
            m_edge_medians: [0.; MAX_MBS_SCALES],
            m_edge_madns: [0.; MAX_MBS_SCALES],
            m_cen_medians: [0.; MAX_MBS_SCALES],
            m_cen_madns: [0.; MAX_MBS_SCALES],
            m_num_in_block: [[0; 3]; MAX_MBS_SCALES],
            m_num_blocks: [[0; 3]; MAX_MBS_SCALES],
            m_best_low_edge: [0; MAX_MBS_SCALES],
            m_best_high_edge: [0; MAX_MBS_SCALES],
            m_fit_mat: [0.; MAX_FIT_COL * MAX_FIT_DATA],
            m_fit_work: [0.; 2 * MAX_FIT_DATA + MAX_FIT_COL * MAX_FIT_COL],
            m_good_row_col: [0; 2],
            m_var_list: [[0; 2]; 5],
            m_nxyz: [0; 3],
            m_num_vars: 0,
            m_num_thicknesses: 0,
            m_thick_median: 0.,
            m_thick_madn: 0.,
            m_boundaries: Vec::new(),
            m_bound_off: 0,
            m_block_centers: Vec::new(),
            m_thicknesses: Vec::new(),
            m_bound_rot: Vec::new(),
            m_proj_means: Vec::new(),
            m_proj_slice: Vec::new(),
            m_best_box_size: [0; 3],
            m_start_coord: [0; 3],
            m_num_high_sd: 0,
            m_num_beads: 0,
            m_min_num_thick_for_check: 0,
            m_crit_thick_madn: 0.,
            m_bead_diameter: 0.,
            m_simplex_iter: 0,
            m_after_simplex: false,
            m_boost_high_sd_thickness: 0.,
            m_bound_low: 0.,
            m_bound_high: 0.,
            m_bead_low: 0.,
            m_bead_high: 0.,
            m_min_spread: 0.,
            m_spread_norms: [0.; 3],
            m_bead_weight_fac: 0.,
            m_bead_area: 0.,
            m_struct_area: 0.,
            m_convex_xtmp: Vec::new(),
            m_convex_ytmp: Vec::new(),
            m_convex_cont: None,
            m_col_max_edge_diff_crit: 0.,
            m_frac_col_max_edge_diff: 0.,
            m_crit_edge_madn: 0.,
            m_num_high_inside_crit: 0,
            m_thick_ind: 0,
            m_fit_pitch_separately: 0,
            m_debug_output: 0,
            m_scan_block_size: 0,
            m_best_scale: 0,
            m_kfactor: 0.,
            m_max_change: 0.,
            m_max_oscill: 0.,
            m_max_iter: 0,
            m_max_falloff_frac: 0.,
            m_low_fit_frac: 0.,
            m_high_fit_frac: 0.,
            m_boundary_frac: 0.,
            m_pitch_boundary_frac: 0.,
            m_min_frac_bounds_in_col: 0.,
            m_column_to_cen_madn_crit: 0.,
            m_too_thin_crit: 0.,
            m_farther_from_mean_crit: 0.,
            m_mean_diff_thick_frac: 0.,
            m_num_extra_sum: 0,
            m_extra_for_pitch: 0.,
            m_extra_pitch_sum: 0.,
            m_min_for_robust_pitch: 0,
            m_bead_excl_pctl_spread: 0.,
            m_bead_excl_pctl_thick: 0.,
            m_proj_layer_stat_type: 0,
            m_proj_edge_crit: 0.,
            m_proj_use_extrap: 0,
            m_use_proj_for_spread: 0,
        };

        // Initializations
        s.m_debug_output = 0;
        s.m_num_thicknesses = 0;
        s.m_fit_pitch_separately = 0;
        s.m_scan_block_size = -1;
        s.m_col_medians = None;
        s.m_col_slice = Vec::new();
        s.m_bead_diameter = 5.;
        s.m_boost_high_sd_thickness = 0.;
        s.m_min_num_thick_for_check = 7; // Minimum number of thickness to do checking with

        // Criterion MADNs below median thickness for eliminating both points if other
        // analysis does not do either one
        s.m_crit_thick_madn = 8.;

        // Parameters
        // 1: Minimum # of points for using robust fit to get pitch line on one surface
        s.m_min_for_robust_pitch = 6;

        // findColumnMidpoint parameters
        // 3: Number of edge MADN's above edge median that maximum value must be to proceed
        s.m_col_max_edge_diff_crit = 2.;
        // 4: Fraction of maximum - edge difference to achieve
        s.m_frac_col_max_edge_diff = 0.5;
        // 5: Number of edge MADNs above edge to achieve as well
        s.m_crit_edge_madn = 3.;
        // 6: Number of box medians that need to be above those criteria
        s.m_num_high_inside_crit = 3;

        // fitColumnBoundaries parameters
        // 7: Number of center MADN's below the center median for inside median to be too low
        s.m_column_to_cen_madn_crit = 5.;
        // 8: Fraction of inside - edge median difference that it must fall toward edge
        s.m_max_falloff_frac = 0.3;
        // 9, 10: Low and high limits of range of fractions of inside - edge median
        // difference to fit
        s.m_low_fit_frac = 0.2;
        s.m_high_fit_frac = 0.8;
        // 11: Fraction of inside - edge median difference at which to save boundary
        s.m_boundary_frac = 0.5;
        // 12: Fraction of difference at which to estimate extra boundary distance for
        // pitch output
        s.m_pitch_boundary_frac = 0.25;
        // 13: Minimum fraction of boxes in column that must yield boundaries
        s.m_min_frac_bounds_in_col = 0.5;

        // checkBlockThicknesses parameters
        // 14: Criterion fraction of median thickness for considering block too thin
        s.m_too_thin_crit = 0.5;
        // 15: Drop a boundary if it is this much farther from local mean than other
        // boundary is
        s.m_farther_from_mean_crit = 2.;
        // 16: Drop a boundary if its difference from the mean is this fraction of median
        // thickness
        s.m_mean_diff_thick_frac = 0.35;

        // Robust fitting parameters
        // 17: K-factor for the weighting function
        s.m_kfactor = 4.68_f64 as f32;
        // 18: Maximum change in weights for terminatiom
        s.m_max_change = 0.02_f64 as f32;
        // 19: Maximum change in weights for terminating on an oscillation
        s.m_max_oscill = 0.05_f64 as f32;
        // 20: Maximum iterations
        s.m_max_iter = 30;

        // 28: Fraction of beads to exclude from spread measurement
        s.m_bead_excl_pctl_spread = 0.04;
        // 29: Basic amount to weight bead separation in the spread measure
        s.m_bead_weight_fac = 0.33;
        // 31: Type of data to use for layer projections: 1=mean, 2=median, 3=75th %ile,
        //     4 = fraction of boxes above criterion
        s.m_proj_layer_stat_type = 0;
        // 32: Fraction of way from baseline to peak for finding rise point of layer
        // projections
        s.m_proj_edge_crit = 0.1;
        // 33: Use extrapolation to baseline rather than point where projection crosses crit
        s.m_proj_use_extrap = 0;
        // 34: 1 to use 2nd moment of projection values, 2 to use 4th moment as spread
        s.m_use_proj_for_spread = 0;
        // 35: Fraction of beads to exclude from determining low and high boundaries
        s.m_bead_excl_pctl_thick = 0.01;
        s
    }

    /// `FindSect::main` (`findsection.cpp:124`).
    pub fn main(&mut self, argv: &[String]) {
        let progname_owned = imod_prog_name(argv.first().map_or("", String::as_str));
        let progname = progname_owned.as_bytes();
        let mut box_size = [[0i32; 3]; MAX_MBS_SCALES];
        let mut box_start = [[0i32; 3]; MAX_MBS_SCALES];
        let mut start_coord = [0i32; 3];
        let mut end_coord = [0i32; 3];
        let mut mxyz = [0i32; 3];
        let mut num_opt_args: i32 = 0;
        let mut num_non_opt_args: i32 = 0;
        let num_tomos: i32;
        let mut ind: i32;
        let mut if_flip: i32;
        let mut ixyz: i32 = 0;
        let mut ierr: i32;
        let mut mode: i32 = 0;
        let mut num_binnings: i32;
        let mut num_spacings: i32 = 0;
        let mut num_sizes: i32 = 0;
        let mut tomo: i32 = 0;
        let mut func_data = [0i32; 5];
        let in_unit_base: i32 = 3;
        let mut nxyz_tmp = [0i32; 3];
        let mut edge_extent = [0i32; 3];
        let mut center_extent = [0i32; 3];
        let mut starts = [0i32; 5];
        let mut ends = [0i32; 5];
        let mut ixy_cen = [0i32; 2];
        let mut num_stat: i32;
        let mut size: i32;
        let mut edge_boxes: i32;
        let mut cen_boxes: i32;
        let mut num_bound: i32 = 0;
        let mut num_good: i32;
        let y_ind: usize;
        let num_xblocks: i32;
        let num_yblocks: i32;
        let mut major: i32;
        let mut minor: i32;
        let mut max_zero_wgt: i32;
        let mut num_fit: i32 = 0;
        let mut cen_starts = [[0i32; 3]; MAX_MBS_SCALES];
        let mut cen_ends = [[0i32; 3]; MAX_MBS_SCALES];
        let mut patch_lim = [0i32; 2];
        let mut iz_mid: i32 = 0;
        let mut max_edge_boxes: i32;
        let mut max_cen_boxes: i32;
        let mut max_column_buf: i32 = 0;
        let mut z_range: i32;
        let mut rem: i32;
        let mut box_num: i32;
        let mut iz_inside = [0i32; 2];
        let mut num_iter: i32 = 0;
        let mut wgt_col_in: i32;
        let mut wgt_col_out: i32;
        let mut num_samples: i32 = 0;
        let mut max_pitch_fit: i32;
        let mut sam_block_space: i32;
        let mut max_blocks: i32;
        let mut num_tomo_opt: i32 = 0;
        let mut num_bin_entries: i32 = 0;
        let min_boxes: i32;
        let max_pixels: i32;
        let tomo_arr_size: i32;
        let mut max_col_boxes: i32;
        let mut max_col_slice: i32;
        let bound_arr_size: i32;
        let mut low_sd_error = [0i32; MAX_MBS_SCALES];
        let mut dmin: f32 = 0.;
        let mut dmax: f32 = 0.;
        let mut dmean: f32 = 0.;
        let mut tmin: f32 = 0.;
        let mut tmax: f32 = 0.;
        let mut tmean: f32 = 0.;
        let mut inside_med: f32 = 0.;
        let mut last_ratio: f32;
        let mut ratio_diff: f32;
        // `lastDiff` is read before it is set when the first scaling has too few
        // center boxes (`findsection.cpp:930`); the C value is stack residue.
        let mut last_diff: f32 = 0.;
        let mut pixel_delta: [f32; 3];
        let mut new_cell = [0f32; 6];
        let mut origin: [f32; 3];
        let mut new_origin = [0f32; 3];
        let mut cur_tilt: [f32; 3];
        let mut ratio: f32;
        let mut fit_const = [0f32; 2];
        let mut xx: f32 = 0.;
        let mut yy: f32 = 0.;
        let mut xy_spacing: f32 = 0.;
        let mut madn_fac: f32;
        let num_box_per_block: f32;
        let mut fit_sd = [0f32; MAX_FIT_COL];
        let mut fit_mean = [0f32; MAX_FIT_COL];
        let mut fit_solution = [0f32; MAX_FIT_COL];
        let default_scales: [i32; MAX_DEFAULT_SCALES as usize] =
            [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
        let mut axis_rotation: f32 = 0.;
        let mut tilt_xvert = [0f32; 4];
        let mut tilt_yvert = [0f32; 4];
        let mut rot_mat = [[0f32; 2]; 3];
        let if_recon_area: i32;
        let mut nx_series: i32;
        let mut ny_series: i32;
        let mut sign: i32;
        let mut surface_lim = [0i32; 2];
        let mut combine_lim = [0i32; 2];
        let mut col_xyz = [0i32; 3];
        let mut mean_sum: f64;
        let mut edge_den_means = [0f32; MAX_MBS_SCALES];
        let mut cen_den_means = [0f32; MAX_MBS_SCALES];
        let mut frac_above_edge = [0f32; MAX_MBS_SCALES];
        let mut min_range = [0i32; 3];
        let linear_format = " %2d %8s  %7.0f  %6.1f %5.1f   %7.0f  %6.1f %5.1f  %8.2f\n";
        let sqrt_format = " %2d %8s  %7.1f  %6.2f %5.2f   %7.1f  %6.2f %5.2f  %8.3f\n";
        let mut table_format = linear_format;
        let mut filename: Vec<u8>;
        let mut any_samples = false;
        let mut vol_root: Option<Vec<u8>> = None;
        let mut point_root: Option<Vec<u8>> = None;
        let mut pitch_name: Option<Vec<u8>> = None;
        let mut surface_name: Option<Vec<u8>> = None;
        let mut bead_file: Option<Vec<u8>> = None;
        let mut pitch_model: Option<Imod> = None;
        let mut bead_model: Option<Imod> = None;
        let mut tomo_inds: Vec<i32> = Vec::new();
        let unit_pt = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let mut sample_extent: i32 = 0;
        let mut num_pitch_pairs: i32 = 0;
        let mut high_sd_crit: f32 = 0.;
        let mut lowest_sd_for_edges: i32 = 0;
        let mut buffer_start_inds = [0i32; MAX_MBS_SCALES + 1];
        let mut in_header: *mut MrcHeader = std::ptr::null_mut();
        let mut smooth_bound: Vec<f32>;
        let low_sd_err_strings: [&str; 4] = [
            "No peak in median SD value found except at top or bottom",
            "No minimum in median SD value found on one side of peak",
            "Minimum values of median SD do not occur on both sides of middle",
            "Median SD does not rise far enough above its minimum value",
        ];

        // values used to set default center and edge areas
        let border_frac: f32;
        let mut edge_thick_frac: f32;
        let edge_area_frac: f32 = 0.5;
        let cen_area_frac: f32 = 0.33;

        // Parameters
        // 21: Fraction that the difference between distinguishability of center from edge
        // points must improve to adopt a higher scaling for analysis
        let mut cen_edge_ratio_diff_crit: f32 = 0.33;
        // 22: Threshold weight from robust fit for including a point in the final smoothing
        let mut wgt_thresh: f32 = 0.2;
        // 23: Threshold weight from robust fit for counting a point as "good"
        let mut good_thresh: f32 = 0.6;
        // 24: Fraction for percentile of positions included in auto-combine Z limits
        let mut combine_high_pctl: f32 = 0.1;
        // 25: Percentile of positions less than combineOutsidePix outside the Z limits
        let mut combine_low_pctl: f32 = 0.01;
        // 26: Number of pixels to back off from the low-percentile of positions
        let mut combine_outside_pix: i32 = 20;
        // 27: Fraction of depth extent to use for center samples (.1, or .4 if highSD)
        let mut cen_thick_frac: f32 = 0.;
        // 30: Take square root of SD values
        let mut take_sqrt: i32 = -1;

        // Fallbacks from    ../manpages/autodoc2man 2 1 findsection
        let num_options: i32 = 28;
        let options: [&[u8]; 28] = [
            b"tomo:TomogramFile:FNM:",
            b"surface:SurfaceModel:FN:",
            b"pitch:TomoPitchModel:FN:",
            b"separate:SeparatePitchLineFits:B:",
            b"samples:NumberOfSamples:I:",
            b"extent:SampleExtentInY:I:",
            b"high:HighSDboxCriterion:F:",
            b"bead:BeadModelFile:FN:",
            b"diameter:BeadDiameter:F:",
            b"boost:BoostHighSDThickness:F:",
            b"lowest:LowestSDforEdges:B:",
            b"scales:NumberOfDefaultScales:I:",
            b"binning:BinningInXYZ:ITM:",
            b"size:SizeOfBoxesInXYZ:ITM:",
            b"spacing:SpacingInXYZ:ITM:",
            b"block:BlockSize:I:",
            b"xminmax:XMinAndMax:IP:",
            b"yminmax:YMinAndMax:IP:",
            b"zminmax:ZMinAndMax:IP:",
            b"flipped:ThickDimensionIsY:I:",
            b"axis:AxisRotationAngle:F:",
            b"tilt:TiltSeriesSizeXY:IP:",
            b"edge:EdgeExtentInXYZ:IT:",
            b"center:CenterExtentInXYZ:IT:",
            b"control:ControlValue:FPM:",
            b"volume:VolumeRootname:CH:",
            b"point:PointRootname:CH:",
            b"debug:DebugOutput:I:",
        ];

        // Startup with fallback
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            progname,
            2,
            1,
            1,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(imod_usage_header_for_pip),
        );

        // Get input files and make sure multiple files are same size
        ierr = pip_number_of_entries(b"TomogramFile", &mut num_tomo_opt);
        num_tomos = num_tomo_opt + num_non_opt_args;
        if num_tomos == 0 {
            exit_error(b"No input tomogram file(s) specified");
        }

        ind = 0;
        while ind < num_tomos {
            filename = Vec::new();
            if ind < num_tomo_opt {
                pip_get_string(b"TomogramFile", &mut filename);
            } else if let Ok(arg) = pip_get_non_option_arg(ind - num_tomo_opt) {
                filename = arg;
            }
            if unsafe {
                iiu_open(
                    in_unit_base + ind,
                    &String::from_utf8_lossy(&filename),
                    "RO",
                )
            } != 0
            {
                let _ = ImodFile::Stdout.flush();
                crate::imod::libcfshr::b3dutil::exit(1);
            }
            unsafe {
                iiu_ret_basic_head(
                    in_unit_base + ind,
                    nxyz_tmp.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin,
                    &mut dmax,
                    &mut dmean,
                );
            }
            if ind != 0 {
                if nxyz_tmp[0] != self.m_nxyz[0]
                    || nxyz_tmp[1] != self.m_nxyz[1]
                    || nxyz_tmp[2] != self.m_nxyz[2]
                {
                    exit_error(b"All sample volumes must be the same size in X, Y, and Z");
                }
            } else {
                self.m_nxyz[0] = nxyz_tmp[0];
                self.m_nxyz[1] = nxyz_tmp[1];
                self.m_nxyz[2] = nxyz_tmp[2];
            }
            ind += 1;
        }

        // Set the index for the thickness axis
        self.m_thick_ind = B3D_Z;
        if_flip = -1;
        pip_get_integer(b"ThickDimensionIsY", &mut if_flip);
        if num_tomos > 1 && if_flip == 0 {
            exit_error(b"Multiple tomograms must have thickness in Y dimension");
        }
        if if_flip > 0 || num_tomos > 1 || (if_flip < 0 && self.m_nxyz[B3D_Y] < self.m_nxyz[B3D_Z])
        {
            self.m_thick_ind = B3D_Y;
        }
        y_ind = 3 - self.m_thick_ind;

        // Get whether high SD is being done, adjust default thicknesses
        pip_get_float(b"HighSDboxCriterion", &mut high_sd_crit);
        if high_sd_crit > 0. && num_tomos > 1 {
            exit_error(b"You cannot analyze for high SD with multiple tomograms");
        }
        if cen_thick_frac <= 0. {
            cen_thick_frac = if high_sd_crit > 0. { 0.4 } else { 0.1 };
        }
        edge_thick_frac = if high_sd_crit > 0. { 0.075 } else { 0.025 };
        pip_get_boolean(b"LowestSDforEdges", &mut lowest_sd_for_edges);
        pip_get_float(b"BoostHighSDThickness", &mut self.m_boost_high_sd_thickness);

        // Allow any parameter to be set
        pip_number_of_entries(b"ControlValue", &mut ixyz);
        // `cppdefs.h:17-18`: `SET_CONTROL_FLOAT` / `SET_CONTROL_INT`, which report
        // through `cout` -- a float streamed with the default precision is `%g`.
        macro_rules! set_control_float {
            ($target:expr, $name:literal) => {{
                $target = yy;
                printf!(concat!($name, " set to %g\n"), CArg::Dbl(yy as f64));
            }};
        }
        macro_rules! set_control_int {
            ($target:expr, $name:literal) => {{
                $target = b3dnint!(yy);
                printf!(
                    concat!($name, " set to %d\n"),
                    CArg::Int(b3dnint!(yy) as i64)
                );
            }};
        }
        ind = 0;
        while ind < ixyz {
            pip_get_two_floats(b"ControlValue", &mut xx, &mut yy);
            match b3dnint!(xx) {
                1 => set_control_int!(self.m_min_for_robust_pitch, "mMinForRobustPitch"),
                3 => set_control_float!(self.m_col_max_edge_diff_crit, "mColMaxEdgeDiffCrit"),
                4 => set_control_float!(self.m_frac_col_max_edge_diff, "mFracColMaxEdgeDiff"),
                5 => set_control_float!(self.m_crit_edge_madn, "mCritEdgeMADN"),
                6 => set_control_int!(self.m_num_high_inside_crit, "mNumHighInsideCrit"),
                7 => set_control_float!(self.m_column_to_cen_madn_crit, "mColumnToCenMADNCrit"),
                8 => set_control_float!(self.m_max_falloff_frac, "mMaxFalloffFrac"),
                9 => set_control_float!(self.m_low_fit_frac, "mLowFitFrac"),
                10 => set_control_float!(self.m_high_fit_frac, "mHighFitFrac"),
                11 => set_control_float!(self.m_boundary_frac, "mBoundaryFrac"),
                12 => set_control_float!(self.m_pitch_boundary_frac, "mPitchBoundaryFrac"),
                13 => set_control_float!(self.m_min_frac_bounds_in_col, "mMinFracBoundsInCol"),
                14 => set_control_float!(self.m_too_thin_crit, "mTooThinCrit"),
                15 => set_control_float!(self.m_farther_from_mean_crit, "mFartherFromMeanCrit"),
                16 => set_control_float!(self.m_mean_diff_thick_frac, "mMeanDiffThickFrac"),
                17 => set_control_float!(self.m_kfactor, "mKfactor"),
                18 => set_control_float!(self.m_max_change, "mMaxChange"),
                19 => set_control_float!(self.m_max_oscill, "mMaxOscill"),
                20 => set_control_int!(self.m_max_iter, "mMaxIter"),
                21 => set_control_float!(cen_edge_ratio_diff_crit, "cenEdgeRatioDiffCrit"),
                22 => set_control_float!(wgt_thresh, "wgtThresh"),
                23 => set_control_float!(good_thresh, "goodThresh"),
                24 => set_control_float!(combine_high_pctl, "combineHighPctl"),
                25 => set_control_float!(combine_low_pctl, "combineLowPctl"),
                26 => set_control_int!(combine_outside_pix, "combineOutsidePix"),
                27 => set_control_float!(cen_thick_frac, "cenThickFrac"),
                28 => set_control_float!(self.m_bead_excl_pctl_spread, "mBeadExclPctlSpread"),
                29 => set_control_float!(self.m_bead_weight_fac, "mBeadWeightFac"),
                30 => set_control_int!(take_sqrt, "takeSqrt"),
                31 => set_control_int!(self.m_proj_layer_stat_type, "mProjLayerStatType"),
                32 => set_control_float!(self.m_proj_edge_crit, "mProjEdgeCrit"),
                33 => set_control_int!(self.m_proj_use_extrap, "mProjUseExtrap"),
                34 => set_control_int!(self.m_use_proj_for_spread, "mUseProjForSpread"),
                35 => set_control_float!(self.m_bead_excl_pctl_thick, "mBeadExclPctlThick"),
                _ => {}
            }
            ind += 1;
        }
        pip_get_integer(b"DebugOutput", &mut self.m_debug_output);

        // Get the numbers of binnings, sizes, and spacings
        num_binnings = 1;
        ierr = pip_get_integer(b"NumberOfDefaultScales", &mut num_binnings);
        pip_number_of_entries(b"BinningInXYZ", &mut num_bin_entries);
        if ierr == 0 && num_bin_entries != 0 {
            exit_error(b"You cannot enter both -scalings and -binning");
        }
        if num_binnings > MAX_DEFAULT_SCALES {
            exit_error_fmt!(
                "There are only %d default scalings available",
                CArg::Int(MAX_DEFAULT_SCALES as i64)
            );
        }
        num_binnings = if num_binnings > num_bin_entries {
            num_binnings
        } else {
            num_bin_entries
        };
        if num_binnings > MAX_MBS_SCALES as i32 {
            exit_error_fmt!(
                "Too many binnings for array sizes; only %d allowed",
                CArg::Int(MAX_MBS_SCALES as i64)
            );
        }

        pip_number_of_entries(b"SpacingInXYZ", &mut num_spacings);
        if pip_number_of_entries(b"SizeOfBoxesInXYZ", &mut num_sizes) != 0 || num_sizes == 0 {
            exit_error(b"Size of boxes must be entered for at least the first binning");
        }
        if num_sizes > num_binnings || num_spacings > num_binnings {
            exit_error(b"It makes no sense to enter more sizes or spacings than binnings");
        }

        // Get the entries for each binning.  When there is no size entry, scale it down
        // from the last size entered.  When there is no spacing, scale it down from the
        // last spacing entered.  When there are no spacings at all, assign it the box size
        // divided by 2
        for scl in 0..num_binnings as usize {
            let scli = scl as i32;

            // Binning, read in the 3 values or take the default isotropic binning
            if scli < num_bin_entries {
                let [b0, b1, b2] = &mut self.m_binning[scl];
                pip_get_three_integers(b"BinningInXYZ", b0, b1, b2);
            } else {
                for ixyz in 0..3 {
                    self.m_binning[scl][ixyz] = default_scales[scl];
                }
            }

            // Size : read in the size or scale down the last size
            if scli < num_sizes {
                let [b0, b1, b2] = &mut box_size[scl];
                pip_get_three_integers(b"SizeOfBoxesInXYZ", b0, b1, b2);
            } else {
                let last = (num_sizes - 1) as usize;
                for ixyz in 0..3 {
                    box_size[scl][ixyz] = b3dnint!(
                        (box_size[last][ixyz] as f32 * self.m_binning[last][ixyz] as f32)
                            / self.m_binning[scl][ixyz] as f32
                    );
                    let maxv = self.m_nxyz[ixyz] / self.m_binning[scl][ixyz];
                    let minned = if maxv < box_size[scl][ixyz] {
                        maxv
                    } else {
                        box_size[scl][ixyz]
                    };
                    box_size[scl][ixyz] = if 1 > minned { 1 } else { minned };
                }
                if self.m_debug_output != 0 {
                    printf!(
                        "Scale %d  box size %d %d %d\n",
                        CArg::Int(scli as i64 + 1),
                        CArg::Int(box_size[scl][0] as i64),
                        CArg::Int(box_size[scl][1] as i64),
                        CArg::Int(box_size[scl][2] as i64)
                    );
                }
            }
            for ixyz in 0..3 {
                let v = box_size[scl][ixyz] * self.m_binning[scl][ixyz];
                min_range[ixyz] = if min_range[ixyz] > v {
                    min_range[ixyz]
                } else {
                    v
                };
            }

            // Spacing: read it in, or set up initial one, or scale down the last one
            if scli < num_spacings {
                let [b0, b1, b2] = &mut self.m_box_spacing[scl];
                pip_get_three_integers(b"SpacingInXYZ", b0, b1, b2);
            } else if scl == 0 {
                for ixyz in 0..3 {
                    let half = box_size[scl][ixyz] / 2;
                    self.m_box_spacing[scl][ixyz] = if 1 > half { 1 } else { half };
                }
                num_spacings = 1;
            } else {
                let last = (num_spacings - 1) as usize;
                for ixyz in 0..3 {
                    let v = b3dnint!(
                        (self.m_box_spacing[last][ixyz] as f32 * self.m_binning[last][ixyz] as f32)
                            / self.m_binning[scl][ixyz] as f32
                    );
                    self.m_box_spacing[scl][ixyz] = if 1 > v { 1 } else { v };
                }
                if self.m_debug_output != 0 {
                    printf!(
                        "Scale %d  box spacing %d %d %d\n",
                        CArg::Int(scli as i64 + 1),
                        CArg::Int(self.m_box_spacing[scl][0] as i64),
                        CArg::Int(self.m_box_spacing[scl][1] as i64),
                        CArg::Int(self.m_box_spacing[scl][2] as i64)
                    );
                }
            }
        }

        // Get output file names and check for validity
        let mut string = Vec::new();
        if pip_get_string(b"VolumeRootname", &mut string) == 0 {
            vol_root = Some(std::mem::take(&mut string));
        }
        if pip_get_string(b"PointRootname", &mut string) == 0 {
            point_root = Some(std::mem::take(&mut string));
        }
        if pip_get_string(b"SurfaceModel", &mut string) == 0 {
            surface_name = Some(std::mem::take(&mut string));
        }
        pip_get_boolean(b"SeparatePitchLineFits", &mut self.m_fit_pitch_separately);
        if pip_get_string(b"TomoPitchModel", &mut string) == 0 {
            pitch_name = Some(std::mem::take(&mut string));
        }
        if take_sqrt < 0 {
            take_sqrt = if high_sd_crit > 0. { 1 } else { 0 };
        }
        if pip_get_string(b"BeadModelFile", &mut string) == 0 {
            bead_file = Some(std::mem::take(&mut string));
        }
        if surface_name.is_some() && num_tomos > 1 {
            exit_error(b"You cannot output a surface model with multiple input tomograms");
        }
        ierr = pip_get_integer(b"NumberOfSamples", &mut num_samples);
        if ierr == 0 && (num_tomos > 1 || high_sd_crit > 0.) {
            exit_error(
                b"You cannot specify sampling with multiple input tomograms or analysis \
for high SD",
            );
        }
        if ierr != 0 && pitch_name.is_some() && high_sd_crit <= 0. && num_tomos == 1 {
            exit_error(
                b"You must specify the number of samples for a boundary model from a \
single tomogram",
            );
        }
        ierr = pip_get_integer(b"SampleExtentInY", &mut sample_extent);
        if ierr == 0 && (num_tomos > 1 || high_sd_crit > 0.) {
            exit_error(
                b"You cannot specify sample extent with multiple tomograms or analysis \
for high SD",
            );
        }
        if ierr == 0 && pitch_name.is_none() {
            exit_error(
                b"You cannot specify sample extent unless outputting a model for \
tomopitch",
            );
        }

        // Get bead model and transform it to this volume
        if let Some(bead_name) = &bead_file {
            let Ok(model) = imod_read(String::from_utf8_lossy(bead_name).to_string()) else {
                // `findsection.cpp:411` passes the NULL model pointer, not the
                // name, to `%s`; glibc prints "(null)".
                exit_error(b"Reading in bead model file (null)");
            };
            let mut model = model;
            let Some(mod_ref) = model.ref_image else {
                exit_error(
                    b"Bead model does not contain coordinate information needed to transform \
to volume being analyzed",
                );
            };
            if model.flags & IMODF_OTRANS_ORIGIN == 0 {
                exit_error(
                    b"Bead model does not contain coordinate information needed to transform \
to volume being analyzed",
                );
            }
            pip_get_float(b"BeadDiameter", &mut self.m_bead_diameter);
            let ctrans = iiu_ret_origin(in_unit_base);
            pixel_delta = iiu_ret_delta(in_unit_base);
            cur_tilt = iiu_ret_tilt(in_unit_base);
            let use_ref = Iref_image {
                ctrans: Ipoint {
                    x: ctrans[0],
                    y: ctrans[1],
                    z: ctrans[2],
                },
                crot: Ipoint {
                    x: cur_tilt[0],
                    y: cur_tilt[1],
                    z: cur_tilt[2],
                },
                cscale: Ipoint {
                    x: pixel_delta[0],
                    y: pixel_delta[1],
                    z: pixel_delta[2],
                },
                otrans: mod_ref.ctrans,
                orot: mod_ref.crot,
                oscale: mod_ref.cscale,
            };
            imod_trans_from_ref_image(&mut model, &use_ref, unit_pt);
            bead_model = Some(model);
        }

        border_frac = if surface_name.is_some() { 0.025 } else { 0.05 };

        // Initialize coordinate limits with a border on the other two axes; get limits
        for ind in 0..3 {
            start_coord[ind] = if ind == self.m_thick_ind {
                0
            } else {
                b3dnint!(border_frac * self.m_nxyz[ind] as f32)
            };
            if self.m_nxyz[ind] - 2 * start_coord[ind] < min_range[ind] {
                start_coord[ind] = (self.m_nxyz[ind] - min_range[ind]) / 2;
            }
            end_coord[ind] = self.m_nxyz[ind] - 1 - start_coord[ind];
        }
        {
            let (s, e) = (&mut start_coord, &mut end_coord);
            pip_get_two_integers(b"XMinAndMax", &mut s[0], &mut e[0]);
            pip_get_two_integers(b"YMinAndMax", &mut s[1], &mut e[1]);
            pip_get_two_integers(b"ZMinAndMax", &mut s[2], &mut e[2]);
        }
        for ind in 0..3 {
            if start_coord[ind] < 0 || end_coord[ind] >= self.m_nxyz[ind] {
                exit_error_fmt!(
                    "Starting or ending %c coordinate outside of range for volume%s",
                    CArg::Chr(b'X' + ind as u8),
                    CArg::Str(if self.m_binning[0][ind] > 1 {
                        "; enter box size in binned coordinates"
                    } else {
                        ""
                    })
                );
            }
        }

        if self.m_debug_output != 0 {
            printf!(
                "Analyzing X %d %d  Y %d %d  Z %d %d\n",
                CArg::Int(start_coord[0] as i64),
                CArg::Int(end_coord[0] as i64),
                CArg::Int(start_coord[1] as i64),
                CArg::Int(end_coord[1] as i64),
                CArg::Int(start_coord[2] as i64),
                CArg::Int(end_coord[2] as i64)
            );
        }

        // Get rotation angle and stack size and make a boundary of reconstructable region
        nx_series = self.m_nxyz[B3D_X];
        ny_series = self.m_nxyz[y_ind];
        ierr = pip_get_two_integers(b"TiltSeriesSizeXY", &mut nx_series, &mut ny_series);
        if_recon_area = 1 - pip_get_float(b"AxisRotationAngle", &mut axis_rotation);
        if ierr == 0 && if_recon_area == 0 {
            exit_error(b"Axis rotation angle must also be entered with tilt series size");
        }
        if if_recon_area != 0 {
            let nx = self.m_nxyz[B3D_X] as f64;
            let ny = self.m_nxyz[y_ind] as f64;
            tilt_xvert[0] = (nx / 2. - nx_series as f64 / 2.) as f32;
            tilt_yvert[0] = (ny / 2. - ny_series as f64 / 2.) as f32;
            tilt_xvert[1] = (nx / 2. + nx_series as f64 / 2.) as f32;
            tilt_yvert[1] = tilt_yvert[0];
            tilt_xvert[2] = tilt_xvert[1];
            tilt_yvert[2] = (ny / 2. + ny_series as f64 / 2.) as f32;
            tilt_xvert[3] = tilt_xvert[0];
            tilt_yvert[3] = tilt_yvert[2];
            rot_mat[0][0] = (axis_rotation as f64 * RADIANS_PER_DEGREE).cos() as f32;
            rot_mat[1][0] = (axis_rotation as f64 * RADIANS_PER_DEGREE).sin() as f32;
            rot_mat[0][1] = -rot_mat[1][0];
            rot_mat[1][1] = rot_mat[0][0];
            rot_mat[2][0] = 0.;
            rot_mat[2][1] = 0.;
            let flat = [
                rot_mat[0][0],
                rot_mat[0][1],
                rot_mat[1][0],
                rot_mat[1][1],
                rot_mat[2][0],
                rot_mat[2][1],
            ];
            for ind in 0..4 {
                let (xp, yp) = xf_apply(
                    &flat,
                    (nx / 2.) as f32,
                    (ny / 2.) as f32,
                    tilt_xvert[ind],
                    tilt_yvert[ind],
                    2,
                );
                tilt_xvert[ind] = xp;
                tilt_yvert[ind] = yp;
            }
        }

        // set up and get the extent of edge and center analysis
        for ixyz in 0..3 {
            size = end_coord[ixyz] + 1 - start_coord[ixyz];
            let (e, c) = if ixyz == self.m_thick_ind {
                (
                    b3dnint!(edge_thick_frac * size as f32),
                    b3dnint!(cen_thick_frac * size as f32),
                )
            } else {
                (
                    b3dnint!(edge_area_frac * size as f32),
                    b3dnint!(cen_area_frac * size as f32),
                )
            };
            edge_extent[ixyz] = if 1 > e { 1 } else { e };
            center_extent[ixyz] = if 1 > c { 1 } else { c };
        }
        {
            let [e0, e1, e2] = &mut edge_extent;
            pip_get_three_integers(b"EdgeExtentInXYZ", e0, e1, e2);
            let [c0, c1, c2] = &mut center_extent;
            pip_get_three_integers(b"CenterExtentInXYZ", c0, c1, c2);
        }
        if self.m_debug_output != 0 {
            printf!(
                "Extents edge %d %d %d  center %d %d %d\n",
                CArg::Int(edge_extent[0] as i64),
                CArg::Int(edge_extent[1] as i64),
                CArg::Int(edge_extent[2] as i64),
                CArg::Int(center_extent[0] as i64),
                CArg::Int(center_extent[1] as i64),
                CArg::Int(center_extent[2] as i64)
            );
        }

        // Set up the computation
        ierr = multi_bin_setup(
            &self.m_binning,
            &box_size,
            &self.m_box_spacing,
            num_binnings,
            &start_coord,
            &end_coord,
            &mut box_start,
            &mut self.m_num_boxes,
            &mut buffer_start_inds,
            &mut self.m_stat_start_inds,
        );
        if ierr != 0 {
            exit_error_fmt!(
                "%s",
                CArg::Str(if ierr == 1 {
                    "Coordinate limits are not usable"
                } else {
                    "Box size is too large for binned volume size"
                })
            );
        }

        // Set default scanning size if not entered, then make bigger if # of boxes is
        // limited in one direction, to give equivalent number of boxes in block
        pip_get_integer(b"BlockSize", &mut self.m_scan_block_size);
        if self.m_scan_block_size < 0 {
            self.m_scan_block_size = if surface_name.is_some() { 200 } else { 100 };
        }
        min_boxes = if self.m_num_boxes[0][B3D_X] < self.m_num_boxes[0][y_ind] {
            self.m_num_boxes[0][B3D_X]
        } else {
            self.m_num_boxes[0][y_ind]
        };
        {
            let a = box_size[0][B3D_X] * self.m_binning[0][B3D_X];
            let b = box_size[0][y_ind] * self.m_binning[0][y_ind];
            max_pixels = if a > b { a } else { b };
        }
        num_box_per_block = self.m_scan_block_size as f32 / max_pixels as f32;
        if (min_boxes as f32) < num_box_per_block {
            self.m_scan_block_size =
                max_pixels * b3dnint!(num_box_per_block * num_box_per_block / min_boxes as f32);
        }

        // Get sizes for center and edge samples and make sure array will be big enough
        max_edge_boxes = 0;
        max_cen_boxes = 0;
        max_pitch_fit = 0;
        max_blocks = 0;
        max_col_boxes = 0;
        max_col_slice = 0;
        for scl in 0..num_binnings as usize {
            edge_boxes = 2;
            cen_boxes = 1;
            for ixyz in 0..3 {
                let v = b3dnint!(
                    (edge_extent[ixyz] as f32 / self.m_binning[scl][ixyz] as f32
                        - box_size[scl][ixyz] as f32)
                        / self.m_box_spacing[scl][ixyz] as f32
                        + 1.
                );
                edge_boxes *= if 1 > v { 1 } else { v };
                num_stat = b3dnint!(
                    (center_extent[ixyz] as f32 / self.m_binning[scl][ixyz] as f32
                        - box_size[scl][ixyz] as f32)
                        / self.m_box_spacing[scl][ixyz] as f32
                        + 1.
                );
                let nb = self.m_num_boxes[scl][ixyz];
                let minned = if nb < num_stat { nb } else { num_stat };
                num_stat = if 1 > minned { 1 } else { minned };
                cen_starts[scl][ixyz] = (self.m_num_boxes[scl][ixyz] - num_stat) / 2;
                cen_ends[scl][ixyz] = cen_starts[scl][ixyz] + num_stat - 1;
                cen_boxes *= num_stat;
            }
            {
                let dims = self.m_num_boxes[scl];
                let (mut s, mut e) = (cen_starts[scl][1], cen_ends[scl][1]);
                self.invert_y_if_flipped(&mut s, &mut e, &dims);
                cen_starts[scl][1] = s;
                cen_ends[scl][1] = e;
            }
            max_edge_boxes = if max_edge_boxes > edge_boxes {
                max_edge_boxes
            } else {
                edge_boxes
            };
            max_cen_boxes = if max_cen_boxes > cen_boxes {
                max_cen_boxes
            } else {
                cen_boxes
            };
            if self.m_debug_output != 0 {
                printf!(
                    "%d cen s-e x %d %d  y %d %d  z %d %d\n",
                    CArg::Int(scl as i64),
                    CArg::Int(cen_starts[scl][0] as i64),
                    CArg::Int(cen_ends[scl][0] as i64),
                    CArg::Int(cen_starts[scl][1] as i64),
                    CArg::Int(cen_ends[scl][1] as i64),
                    CArg::Int(cen_starts[scl][2] as i64),
                    CArg::Int(cen_ends[scl][2] as i64)
                );
            }

            self.setup_blocks(self.m_num_boxes[scl][B3D_X], scl, B3D_X);
            self.setup_blocks(self.m_num_boxes[scl][y_ind], scl, y_ind);
            let v = (self.m_num_blocks[scl][B3D_X] + 2) * (self.m_num_blocks[scl][y_ind] + 2);
            max_blocks = if max_blocks > v { max_blocks } else { v };

            // fitColumnBoundaries needs space for a block of values on each surface and for
            // two fitting arrays that could be an unknown fraction of the Z extent
            let v = 2 * self.m_num_boxes[scl][self.m_thick_ind]
                + 4 * (self.m_num_in_block[scl][B3D_X] + 3) * (self.m_num_in_block[scl][y_ind] + 3);
            max_column_buf = if max_column_buf > v {
                max_column_buf
            } else {
                v
            };

            // Make sure there is enough space for fitting lines for pitch model
            if pitch_name.is_some() {
                if num_tomos > 1 {
                    size = self.m_num_blocks[scl][y_ind];
                } else if sample_extent == 0 {
                    size = 1;
                } else {
                    size = self.m_binning[scl][y_ind]
                        * (self.m_num_in_block[scl][y_ind] * self.m_box_spacing[scl][y_ind]
                            + box_size[scl][y_ind]);
                    let v = b3dnint!(sample_extent as f32 / size as f32);
                    size = if 1 > v { 1 } else { v };
                }
                let v = size * self.m_num_blocks[scl][B3D_X];
                max_pitch_fit = if max_pitch_fit > v { max_pitch_fit } else { v };
            }

            // Keep track of biggest thickness and biggest box slice in X-Z for column outputs
            let v = self.m_num_boxes[scl][self.m_thick_ind];
            max_col_boxes = if max_col_boxes > v { max_col_boxes } else { v };
            let v = self.m_num_boxes[scl][B3D_X] * self.m_num_boxes[scl][self.m_thick_ind];
            max_col_slice = if max_col_slice > v { max_col_slice } else { v };
        }

        // `b3dIMax(6, ...)`: the 6 is the count of values that follow.
        size = b3d_i_max(&[
            buffer_start_inds[num_binnings as usize],
            2 * max_edge_boxes,
            2 * max_cen_boxes,
            (if max_cen_boxes > max_column_buf {
                max_cen_boxes
            } else {
                max_column_buf
            }) + max_column_buf,
            3 * max_pitch_fit,
            max_blocks,
        ]);
        self.m_buffer = vec![0.; (size * num_tomos) as usize];
        tomo_arr_size = self.m_stat_start_inds[num_binnings as usize];
        self.m_means = vec![0.; (tomo_arr_size * num_tomos) as usize];
        self.m_sds = vec![0.; (tomo_arr_size * num_tomos) as usize];
        self.m_thicknesses = vec![0.; (max_blocks * num_tomos) as usize];
        if vol_root.is_some() {
            self.m_col_slice = vec![0.; (max_col_slice + max_col_boxes) as usize];
            self.m_col_medians = Some(max_col_slice as usize);
        }

        // Set up a tomopitch model
        if pitch_name.is_some() {
            let Some(mut model) = imod_new() else {
                exit_error(b"Creating model for tomopitch output");
            };
            if imod_new_object(&mut model) != 0 {
                exit_error(b"Creating model for tomopitch output");
            }
            model.obj[0].extra[IOBJ_EX_PNT_LIMIT] = 2;
            model.obj[0].flags = IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_PLANAR;
            if num_tomos > 1 {
                model.obj[0].flags |= IMOD_OBJFLAG_TIME;
            }
            model.xmax = self.m_nxyz[0];
            model.ymax = self.m_nxyz[1];
            model.zmax = self.m_nxyz[2];
            pitch_model = Some(model);
        }

        // Set up sequences indexes to do tomograms from center outward
        for tomo_seq in 0..num_tomos {
            tomo = num_tomos / 2;
            if tomo_seq % 2 != 0 {
                tomo -= (tomo_seq + 1) / 2;
            } else {
                tomo += tomo_seq / 2;
            }
            tomo_inds.push(tomo);
        }

        // START LOOP ON TOMOGRAMS
        self.m_num_extra_sum = 0;
        self.m_extra_pitch_sum = 0.;
        for tomo_seq in 0..num_tomos as usize {
            tomo = tomo_inds[tomo_seq];
            self.m_sds_off = (tomo * tomo_arr_size) as usize;
            self.m_means_off = (tomo * tomo_arr_size) as usize;

            func_data[0] = in_unit_base + tomo;
            func_data[1] = start_coord[0];
            func_data[2] = end_coord[0];
            func_data[3] = start_coord[1];
            func_data[4] = end_coord[1];

            ierr = multi_bin_stats(
                &self.m_binning,
                &box_size,
                &self.m_box_spacing,
                num_binnings,
                &start_coord,
                &end_coord,
                &box_start,
                &self.m_num_boxes,
                &buffer_start_inds,
                &self.m_stat_start_inds,
                &mut self.m_buffer,
                &mut self.m_means[self.m_means_off..],
                &mut self.m_sds[self.m_sds_off..],
                &mut func_data,
                get_slice,
            );
            if ierr != 0 {
                exit_error(b"Reading data from file");
            }

            if take_sqrt != 0 {
                table_format = sqrt_format;
                for scl in 0..num_binnings as usize {
                    for iz in 0..self.m_num_boxes[scl][B3D_Z] {
                        let write_buf = self.m_sds_off
                            + (self.m_stat_start_inds[scl]
                                + iz * self.m_num_boxes[scl][B3D_Y] * self.m_num_boxes[scl][B3D_X])
                                as usize;
                        for iy_box in 0..self.m_num_boxes[scl][B3D_Y] {
                            for ix_box in 0..self.m_num_boxes[scl][B3D_X] {
                                let ind = write_buf
                                    + (ix_box + iy_box * self.m_num_boxes[scl][B3D_X]) as usize;
                                self.m_sds[ind] = (self.m_sds[ind] as f64).sqrt() as f32;
                            }
                        }
                    }
                }
            }

            // Write data
            if let Some(vol_root) = &vol_root {
                let vol_root = String::from_utf8_lossy(vol_root).into_owned();
                let mut write_src_sds = false;
                pixel_delta = iiu_ret_delta(in_unit_base + tomo);
                origin = iiu_ret_origin(in_unit_base + tomo);
                cur_tilt = iiu_ret_tilt(in_unit_base + tomo);
                for loop_ in 0..2 {
                    for scl in 0..num_binnings as usize {
                        // Adjust origin and pixel size
                        for ixyz in 0..3 {
                            new_cell[ixyz] = self.m_num_boxes[scl][ixyz] as f32
                                * pixel_delta[ixyz]
                                * self.m_binning[scl][ixyz] as f32
                                * self.m_box_spacing[scl][ixyz] as f32;
                            new_origin[ixyz] = origin[ixyz]
                                - pixel_delta[ixyz]
                                    * (start_coord[ixyz]
                                        + box_start[scl][ixyz] * self.m_binning[scl][ixyz])
                                        as f32;
                            new_cell[ixyz + 3] = 90.;
                        }
                        let name = c_format_bytes(
                            "%s%d-scale%d.%s",
                            &[
                                CArg::Str(&vol_root),
                                CArg::Int(tomo as i64),
                                CArg::Int(scl as i64),
                                CArg::Str(if loop_ != 0 { "SDs" } else { "means" }),
                            ],
                        );
                        unsafe { iiu_open(1, &String::from_utf8_lossy(&name), "NEW") };
                        let nb = self.m_num_boxes[scl];
                        iiu_create_header(1, &nb, &nb, 2, &[[0u8; 80]; 10], 0);
                        iiu_alt_cell(1, &new_cell);
                        iiu_alt_origin(1, &new_origin);
                        iiu_alt_tilt(1, &cur_tilt);
                        dmin = 1.0e37;
                        dmax = -dmin;
                        dmean = 0.;
                        for iz in 0..nb[B3D_Z] {
                            let (src, off) = if write_src_sds {
                                (&mut self.m_sds, self.m_sds_off)
                            } else {
                                (&mut self.m_means, self.m_means_off)
                            };
                            let write_buf = off
                                + (self.m_stat_start_inds[scl] + iz * nb[B3D_Y] * nb[B3D_X])
                                    as usize;
                            unsafe { iiu_write_section(1, src[write_buf..].as_mut_ptr().cast()) };
                            array_min_max_mean(
                                &src[write_buf..],
                                nb[B3D_X],
                                nb[B3D_Y],
                                0,
                                nb[B3D_X] - 1,
                                0,
                                nb[B3D_Y] - 1,
                                &mut tmin,
                                &mut tmax,
                                &mut tmean,
                            );
                            dmin = if dmin < tmin { dmin } else { tmin };
                            dmax = if dmax > tmax { dmax } else { tmax };
                            dmean += tmean / nb[B3D_Z] as f32;
                        }
                        iiu_write_header_str(1, "FINDSECTION", 0, dmin, dmax, dmean);
                        unsafe { iiu_close(1) };
                    }
                    write_src_sds = true;
                }

                // Set up output file for column medians
                for scl in 0..num_binnings as usize {
                    let name = c_format_bytes(
                        "%s%d-scale%d.colmed",
                        &[
                            CArg::Str(&vol_root),
                            CArg::Int(tomo as i64),
                            CArg::Int(scl as i64),
                        ],
                    );
                    unsafe { iiu_open(1, &String::from_utf8_lossy(&name), "NEW") };
                    self.setup_blocks(self.m_num_boxes[scl][B3D_X], scl, B3D_X);
                    self.setup_blocks(self.m_num_boxes[scl][y_ind], scl, y_ind);
                    printf!(
                        "%d: %d x blocks of %d+   %d y blocks of %d+\n",
                        CArg::Int(scl as i64),
                        CArg::Int(self.m_num_blocks[scl][B3D_X] as i64),
                        CArg::Int(self.m_num_in_block[scl][0] as i64),
                        CArg::Int(self.m_num_blocks[scl][y_ind] as i64),
                        CArg::Int(self.m_num_in_block[scl][y_ind] as i64)
                    );
                    col_xyz[0] = self.m_num_boxes[scl][0];
                    col_xyz[1] = self.m_num_boxes[scl][self.m_thick_ind];
                    col_xyz[2] = self.m_num_blocks[scl][y_ind];
                    iiu_create_header(1, &col_xyz, &col_xyz, 2, &[[0u8; 80]; 10], 0);
                    dmin = 1.0e37;
                    dmax = -dmin;
                    dmean = 0.;

                    for iy_box in 0..self.m_num_blocks[scl][y_ind] {
                        for ix_box in 0..self.m_num_blocks[scl][B3D_X] {
                            // Get the range of boxes in the block
                            let mut ixyz = 0;
                            while ixyz <= y_ind {
                                let (mut s, mut e) = (0, 0);
                                balanced_group_limits(
                                    self.m_num_boxes[scl][ixyz],
                                    self.m_num_blocks[scl][ixyz],
                                    if ixyz != 0 { iy_box } else { ix_box },
                                    &mut s,
                                    &mut e,
                                );
                                starts[ixyz] = s;
                                ends[ixyz] = e;
                                ixyz += y_ind;
                            }

                            // Get the medians
                            self.find_column_midpoint(
                                starts[0],
                                ends[0],
                                starts[y_ind],
                                ends[y_ind],
                                scl,
                                max_cen_boxes as usize,
                                &mut iz_inside,
                                &mut iz_mid,
                                &mut inside_med,
                            );

                            // Replicate medians into the boxes within the slice
                            let medians = self.m_col_medians.unwrap_or(0);
                            for iz in 0..self.m_num_boxes[scl][self.m_thick_ind] {
                                for ind in starts[0]..=ends[0] {
                                    self.m_col_slice
                                        [(iz * self.m_num_boxes[scl][0] + ind) as usize] =
                                        self.m_col_slice[medians + iz as usize];
                                }
                            }
                        }

                        // Write slice for this Y
                        unsafe { iiu_write_section(1, self.m_col_slice.as_mut_ptr().cast()) };
                        array_min_max_mean(
                            &self.m_col_slice,
                            col_xyz[0],
                            col_xyz[1],
                            0,
                            col_xyz[0] - 1,
                            0,
                            col_xyz[1] - 1,
                            &mut tmin,
                            &mut tmax,
                            &mut tmean,
                        );
                        dmin = if dmin < tmin { dmin } else { tmin };
                        dmax = if dmax > tmax { dmax } else { tmax };
                        dmean += tmean / self.m_num_blocks[scl][y_ind] as f32;
                    }
                    iiu_write_header_str(1, "FINDSECTION", 0, dmin, dmax, dmean);
                    unsafe { iiu_close(1) };
                }
            }
        }

        // Evaluate statistics of edge first
        for scl in 0..num_binnings as usize {
            for ixyz in 0..3 {
                num_stat = b3dnint!(
                    (edge_extent[ixyz] as f32 / self.m_binning[scl][ixyz] as f32
                        - box_size[scl][ixyz] as f32)
                        / self.m_box_spacing[scl][ixyz] as f32
                        + 1.
                );
                let nb = self.m_num_boxes[scl][ixyz];
                let minned = if nb < num_stat { nb } else { num_stat };
                num_stat = if 1 > minned { 1 } else { minned };

                // For thickness, get inclusive limits of excluded region
                if ixyz == self.m_thick_ind {
                    num_stat = self.m_num_boxes[scl][ixyz] - num_stat;
                }
                starts[ixyz] = (self.m_num_boxes[scl][ixyz] - num_stat) / 2;
                ends[ixyz] = starts[ixyz] + num_stat - 1;
            }
            {
                let dims = self.m_num_boxes[scl];
                let (mut s, mut e) = (starts[1], ends[1]);
                self.invert_y_if_flipped(&mut s, &mut e, &dims);
                starts[1] = s;
                ends[1] = e;
            }

            // Save excluded region in the thickness dimension in s/e[3] and the absolute
            // limits in s/e[4], set fallbacks for edge limits
            starts[3] = starts[self.m_thick_ind];
            ends[3] = ends[self.m_thick_ind];
            starts[4] = 0;
            ends[4] = self.m_num_boxes[scl][self.m_thick_ind] - 1;
            self.m_best_low_edge[scl] = 0;
            self.m_best_high_edge[scl] = ends[4];

            if high_sd_crit > 0. && lowest_sd_for_edges != 0 {
                low_sd_error[scl] = self.find_lowest_sd_edges(&mut starts, &mut ends, scl);
                if low_sd_error[scl] != 0 && self.m_debug_output != 0 {
                    printf!(
                        "Error %d finding best limits for edge at scale %d\n",
                        CArg::Int(ierr as i64),
                        CArg::Int(scl as i64 + 1)
                    );
                }
            }

            num_stat = 0;
            mean_sum = 0.;
            if self.m_debug_output != 0 {
                printf!(
                    "edge boxes %d %d %d %d %d %d %d %d\n",
                    CArg::Int(starts[0] as i64),
                    CArg::Int(ends[0] as i64),
                    CArg::Int(starts[1] as i64),
                    CArg::Int(ends[1] as i64),
                    CArg::Int(starts[2] as i64),
                    CArg::Int(ends[2] as i64),
                    CArg::Int(starts[3] as i64),
                    CArg::Int(ends[3] as i64)
                );
            }

            // Now add each tomogram into edge sample
            for tomo_seq in 0..num_tomos as usize {
                tomo = tomo_inds[tomo_seq];
                self.m_sds_off = (tomo * tomo_arr_size) as usize;
                self.m_means_off = (tomo * tomo_arr_size) as usize;

                // Set start and end of lower sample from absolute start and one below
                // excluded
                starts[self.m_thick_ind] = starts[4];
                ends[self.m_thick_ind] = starts[3] - 1;
                self.add_boxes_to_sample(
                    starts[0],
                    ends[0],
                    starts[1],
                    ends[1],
                    starts[2],
                    ends[2],
                    scl,
                    &mut num_stat,
                    &mut mean_sum,
                );

                starts[self.m_thick_ind] = ends[3] + 1;
                ends[self.m_thick_ind] = ends[4];
                if self.m_debug_output != 0 {
                    printf!(
                        "AND  boxes %d %d %d %d %d %d\n",
                        CArg::Int(starts[0] as i64),
                        CArg::Int(ends[0] as i64),
                        CArg::Int(starts[1] as i64),
                        CArg::Int(ends[1] as i64),
                        CArg::Int(starts[2] as i64),
                        CArg::Int(ends[2] as i64)
                    );
                }
                self.add_boxes_to_sample(
                    starts[0],
                    ends[0],
                    starts[1],
                    ends[1],
                    starts[2],
                    ends[2],
                    scl,
                    &mut num_stat,
                    &mut mean_sum,
                );
            }

            edge_den_means[scl] = (mean_sum / num_stat as f64) as f32;
            {
                let (head, tail) = self.m_buffer.split_at_mut(num_stat as usize);
                rs_fast_median(head, num_stat, tail, &mut self.m_edge_medians[scl]);
                rs_fast_madn(
                    head,
                    num_stat,
                    self.m_edge_medians[scl],
                    tail,
                    &mut self.m_edge_madns[scl],
                );
            }
            if self.m_debug_output != 0 {
                printf!(
                    "edge: %d  %8.3f %8.3f\n",
                    CArg::Int(num_stat as i64),
                    CArg::Dbl(self.m_edge_medians[scl] as f64),
                    CArg::Dbl(self.m_edge_madns[scl] as f64)
                );
            }
        }
        if self.m_debug_output != 0 {
            printf!("\n");
        }

        printf!(
            "               center   center SDs      edge     edge SDs      distinct-\n\
scale  binning  mean   median  MADN     mean   median  MADN      ness\n"
        );
        for scl in 0..num_binnings as usize {
            // Get the block size and number of blocks for center sampling
            self.setup_blocks(
                cen_ends[scl][B3D_X] + 1 - cen_starts[scl][B3D_X],
                scl,
                B3D_X,
            );
            self.setup_blocks(
                cen_ends[scl][y_ind] + 1 - cen_starts[scl][y_ind],
                scl,
                y_ind,
            );
            z_range = cen_ends[scl][self.m_thick_ind] + 1 - cen_starts[scl][self.m_thick_ind];

            self.m_cen_medians[scl] = 0.;
            num_stat = 0;
            mean_sum = 0.;

            // Do each tomogram
            for tomo_seq in 0..num_tomos as usize {
                tomo = tomo_inds[tomo_seq];
                self.m_sds_off = (tomo * tomo_arr_size) as usize;
                self.m_means_off = (tomo * tomo_arr_size) as usize;

                // Loop on the blocks
                for iy_box in 0..self.m_num_blocks[scl][y_ind] {
                    for ix_box in 0..self.m_num_blocks[scl][B3D_X] {
                        // Get the range of boxes in the block
                        let mut ixyz = 0;
                        while ixyz <= y_ind {
                            box_num = if ixyz != 0 { iy_box } else { ix_box };
                            rem = (cen_ends[scl][ixyz] + 1 - cen_starts[scl][ixyz])
                                % self.m_num_blocks[scl][ixyz];
                            starts[ixyz] = cen_starts[scl][ixyz]
                                + box_num * self.m_num_in_block[scl][ixyz]
                                + if box_num <= rem { box_num } else { rem };
                            ends[ixyz] = starts[ixyz]
                                + self.m_num_in_block[scl][ixyz]
                                + if box_num < rem { 0 } else { -1 };
                            ixyz += y_ind;
                        }

                        if high_sd_crit > 0. {
                            // Shift the range to be centered on the best edge limits
                            if lowest_sd_for_edges != 0 {
                                let v = (self.m_best_low_edge[scl] + self.m_best_high_edge[scl]
                                    - z_range)
                                    / 2;
                                starts[self.m_thick_ind] = if v > 0 { v } else { 0 };
                                let a = starts[self.m_thick_ind] + z_range - 1;
                                let b = self.m_num_boxes[scl][self.m_thick_ind];
                                ends[self.m_thick_ind] = if a < b { a } else { b };
                            }
                            self.add_boxes_to_sample(
                                starts[0],
                                ends[0],
                                starts[1],
                                ends[1],
                                starts[2],
                                ends[2],
                                scl,
                                &mut num_stat,
                                &mut mean_sum,
                            );

                            // Get the midpoint and ranges inside and add boxes
                        } else if self.find_column_midpoint(
                            starts[0],
                            ends[0],
                            starts[y_ind],
                            ends[y_ind],
                            scl,
                            max_cen_boxes as usize,
                            &mut iz_inside,
                            &mut iz_mid,
                            &mut inside_med,
                        ) == 0
                        {
                            let b = iz_mid - z_range / 2;
                            starts[self.m_thick_ind] =
                                if iz_inside[0] > b { iz_inside[0] } else { b };
                            let b = starts[self.m_thick_ind] + z_range - 1;
                            ends[self.m_thick_ind] =
                                if iz_inside[1] < b { iz_inside[1] } else { b };
                            self.add_boxes_to_sample(
                                starts[0],
                                ends[0],
                                starts[1],
                                ends[1],
                                starts[2],
                                ends[2],
                                scl,
                                &mut num_stat,
                                &mut mean_sum,
                            );
                        }
                    }
                }
            }
            if num_stat < 10 {
                continue;
            }

            cen_den_means[scl] = (mean_sum / num_stat as f64) as f32;
            any_samples = true;

            {
                let (head, tail) = self.m_buffer.split_at_mut(num_stat as usize);
                rs_fast_median(head, num_stat, tail, &mut self.m_cen_medians[scl]);
                rs_fast_madn(
                    head,
                    num_stat,
                    self.m_cen_medians[scl],
                    tail,
                    &mut self.m_cen_madns[scl],
                );
            }
            madn_fac = (self.m_cen_medians[0] - self.m_edge_medians[0]) / self.m_cen_madns[0];
            {
                // B3DCLAMP(madnFac, 1., 2.): double comparisons with NaN falling through
                let m = madn_fac as f64;
                let minned = if 2. < m { 2. } else { m };
                madn_fac = (if 1. > minned { 1. } else { minned }) as f32;
            }
            frac_above_edge[scl] = (self.m_cen_medians[scl]
                - madn_fac * self.m_cen_madns[scl]
                - self.m_edge_medians[scl])
                / self.m_edge_madns[scl];

            // If doing high SD, find fraction of center boxes distinct from the edge
            if high_sd_crit > 0. {
                num_good = 0;
                for ind in 0..num_stat as usize {
                    if (self.m_buffer[ind] - self.m_edge_medians[scl]) / self.m_edge_madns[scl]
                        > madn_fac
                    {
                        num_good += 1;
                    }
                }
                frac_above_edge[scl] = num_good as f32 / num_stat as f32;
            }
            let bin_str = c_format_bytes(
                "%d,%d,%d",
                &[
                    CArg::Int(self.m_binning[scl][0] as i64),
                    CArg::Int(self.m_binning[scl][1] as i64),
                    CArg::Int(self.m_binning[scl][1] as i64),
                ],
            );
            printf!(
                table_format,
                CArg::Int(scl as i64 + 1),
                CArg::Bytes(&bin_str),
                CArg::Dbl(cen_den_means[scl] as f64),
                CArg::Dbl(self.m_cen_medians[scl] as f64),
                CArg::Dbl(self.m_cen_madns[scl] as f64),
                CArg::Dbl(edge_den_means[scl] as f64),
                CArg::Dbl(self.m_edge_medians[scl] as f64),
                CArg::Dbl(self.m_edge_madns[scl] as f64),
                CArg::Dbl(frac_above_edge[scl] as f64)
            );
        }

        if !any_samples {
            exit_error_fmt!(
                "%soo few boxes in center sample where boundaries of section could \
be detected",
                CArg::Str(if num_binnings > 1 {
                    "For all scalings, t"
                } else {
                    "T"
                })
            );
        }

        // Next, pick the best binning
        self.m_best_scale = 0;
        last_ratio = 0.;
        ratio_diff = 0.;
        for scl in 0..num_binnings as usize {
            ratio = frac_above_edge[scl];
            if self.m_cen_medians[scl] == 0. {
                printf!(
                    "WARNING: Too few boxes in center sample where boundaries of section could \
be detected for scaling # %d\n",
                    CArg::Int(scl as i64 + 1)
                );
                continue;
            }

            // Require SOME distinction between edge and center
            if scl != 0
                && (high_sd_crit > 0.
                    || (((self.m_cen_medians[scl] - self.m_edge_medians[scl])
                        / self.m_edge_madns[scl]) as f64)
                        > 0.05)
            {
                // Compute ratio change per step since the last best scale
                ratio_diff = (ratio - last_ratio) / (scl as i32 - self.m_best_scale as i32) as f32;
                if ratio_diff < cen_edge_ratio_diff_crit * last_diff {
                    continue;
                }
                self.m_best_scale = scl;
            }
            last_ratio = ratio;
            last_diff = ratio_diff;
        }
        if num_binnings > 1 {
            printf!(
                "Selected scaling # %d as the best one for analysis\n",
                CArg::Int(self.m_best_scale as i64 + 1)
            );
        }

        // report fate of minimum SD edge picking for that scaling
        if high_sd_crit > 0. && lowest_sd_for_edges != 0 {
            let err = low_sd_error[self.m_best_scale];
            if err != 0 {
                // `findsection.cpp:945` indexes the four-entry table with the 1-based
                // error code, so codes 1-3 print the *next* message and code 4 reads
                // past the array; that read has no defined value to reproduce.
                printf!(
                    "For this scaling, an error occurred finding slices with a\n   \
value of median SD to use for edge samples with this scaling:\n   %s\n",
                    CArg::Str(
                        low_sd_err_strings
                            .get(err as usize)
                            .copied()
                            .unwrap_or("(null)")
                    )
                );
            } else {
                printf!(
                    "The edge statistics for this scaling were computed from boxes around\n   \
%d and %d (out of %d) instead of from top and bottom boxes\n",
                    CArg::Int(self.m_best_low_edge[self.m_best_scale] as i64),
                    CArg::Int(self.m_best_high_edge[self.m_best_scale] as i64),
                    CArg::Int(self.m_num_boxes[self.m_best_scale][self.m_thick_ind] as i64)
                );
            }
        }

        // Do everything for cryo (highSD) analysis and exit
        if high_sd_crit > 0. {
            let best_box = box_size[self.m_best_scale];
            self.analyze_high_sd(
                high_sd_crit,
                bead_model.as_ref(),
                &best_box,
                &start_coord,
                &end_coord,
                pitch_model.as_mut(),
            );
            if point_root.is_some() || pitch_name.is_some() {
                in_header = unsafe { iiu_mrc_header(in_unit_base + tomo, "findsection", 1, 0) };
            }
            if let Some(point_root) = &point_root {
                let name = c_format_bytes(
                    "%s%d-highSDbound.mod",
                    &[CArg::Bytes(point_root), CArg::Int(tomo as i64)],
                );
                let boundaries = std::mem::take(&mut self.m_boundaries);
                self.dump_point_model(
                    &boundaries,
                    self.m_num_high_sd,
                    &String::from_utf8_lossy(&name),
                    unsafe { &*in_header },
                );
                self.m_boundaries = boundaries;
            }
            if let Some(pitch_name) = &pitch_name {
                let model = pitch_model.as_mut().unwrap();
                imod_set_ref_image(model, unsafe { &*in_header });
                self.write_model(&String::from_utf8_lossy(pitch_name), model);
            }
            let _ = ImodFile::Stdout.flush();
            crate::imod::libcfshr::b3dutil::exit(0);
        }

        // Set up the blocking from scratch and allocate more arrays
        let scl = self.m_best_scale;
        self.setup_blocks(self.m_num_boxes[scl][B3D_X], scl, B3D_X);
        self.setup_blocks(self.m_num_boxes[scl][y_ind], scl, y_ind);
        num_xblocks = self.m_num_blocks[scl][B3D_X];
        num_yblocks = self.m_num_blocks[scl][y_ind];
        bound_arr_size = 2 * num_xblocks * num_yblocks;
        self.m_boundaries = vec![0.; (bound_arr_size * num_tomos) as usize];
        self.m_block_centers = vec![0.; bound_arr_size as usize];
        smooth_bound = vec![0.; bound_arr_size as usize];

        // Loop on tomos to get boundaries and all thickness
        for tomo_seq in 0..num_tomos as usize {
            tomo = tomo_inds[tomo_seq];
            self.m_sds_off = (tomo * tomo_arr_size) as usize;
            self.m_means_off = (tomo * tomo_arr_size) as usize;
            self.m_bound_off = (tomo * bound_arr_size) as usize;

            // Now find a boundary position in each block; loop on the blocks
            for iy_block in 0..num_yblocks {
                for ix_block in 0..num_xblocks {
                    let ind = (2 * (iy_block * num_xblocks + ix_block)) as usize;

                    // Get the range of boxes in the block
                    let mut ixyz = 0;
                    while ixyz <= y_ind {
                        box_num = if ixyz != 0 { iy_block } else { ix_block };
                        rem = self.m_num_boxes[scl][ixyz] % self.m_num_blocks[scl][ixyz];
                        starts[ixyz] = box_num * self.m_num_in_block[scl][ixyz]
                            + if box_num <= rem { box_num } else { rem };
                        ends[ixyz] = starts[ixyz]
                            + self.m_num_in_block[scl][ixyz]
                            + if box_num < rem { 0 } else { -1 };

                        // And get the unbinned center coordinate of the block
                        self.m_block_centers[ind + if 1 < ixyz { 1 } else { ixyz }] =
                            (start_coord[ixyz] as f64
                                + self.m_binning[scl][ixyz] as f64
                                    * (((starts[ixyz] + ends[ixyz]) * self.m_box_spacing[scl][ixyz]
                                        + box_size[scl][ixyz])
                                        as f64
                                        / 2.
                                        + box_start[scl][ixyz] as f64))
                                as f32;
                        ixyz += y_ind;
                    }

                    // Get the boundaries and scale to unbinned coordinates there too
                    let bo = self.m_bound_off + ind;
                    if if_recon_area != 0
                        && inside_contour(
                            &tilt_xvert,
                            &tilt_yvert,
                            4,
                            self.m_block_centers[ind],
                            self.m_block_centers[ind + 1],
                        ) == 0
                    {
                        self.m_boundaries[bo] = -1.;
                        self.m_boundaries[bo + 1] = -1.;
                    } else {
                        let mut boundary = [0f32; 2];
                        self.fit_column_boundaries(
                            starts[0],
                            ends[0],
                            starts[y_ind],
                            ends[y_ind],
                            &mut boundary,
                        );
                        self.m_boundaries[bo] = boundary[0];
                        self.m_boundaries[bo + 1] = boundary[1];
                    }
                    let ti = self.m_thick_ind;
                    for loop_ in 0..2 {
                        if self.m_boundaries[bo + loop_] > 0. {
                            self.m_boundaries[bo + loop_] = (start_coord[ti] as f64
                                + self.m_binning[scl][ti] as f64
                                    * ((self.m_boundaries[bo + loop_]
                                        * self.m_box_spacing[scl][ti] as f32)
                                        as f64
                                        + 0.5 * box_size[scl][ti] as f64
                                        + box_start[scl][ti] as f64))
                                as f32;
                        }
                    }

                    // Add thicknesses to a collection
                    if self.m_boundaries[bo] >= 0. && self.m_boundaries[bo + 1] >= 0. {
                        self.m_thicknesses[self.m_num_thicknesses as usize] =
                            self.m_boundaries[bo + 1] - self.m_boundaries[bo];
                        self.m_num_thicknesses += 1;
                    }
                }
            }
        }
        if self.m_num_extra_sum != 0 {
            self.m_extra_for_pitch = self.m_extra_pitch_sum / self.m_num_extra_sum as f32;
        }
        if self.m_debug_output != 0 {
            printf!(
                "Mean extra distance for tomopitch lines = %.1f\n",
                CArg::Dbl(self.m_extra_for_pitch as f64)
            );
        }

        if self.m_num_thicknesses >= self.m_min_num_thick_for_check {
            rs_fast_median(
                &self.m_thicknesses,
                self.m_num_thicknesses,
                &mut self.m_buffer,
                &mut self.m_thick_median,
            );
            rs_fast_madn(
                &self.m_thicknesses,
                self.m_num_thicknesses,
                self.m_thick_median,
                &mut self.m_buffer,
                &mut self.m_thick_madn,
            );
            if self.m_debug_output != 0 {
                printf!(
                    "mThickMedian = %g,  mThickMADN = %g,  mNumThicknesses = %d\n",
                    CArg::Dbl(self.m_thick_median as f64),
                    CArg::Dbl(self.m_thick_madn as f64),
                    CArg::Int(self.m_num_thicknesses as i64)
                );
            }
        }

        // START OF BIG REMAINING LOOP ON TOMOGRAMS
        for tomo_seq in 0..num_tomos as usize {
            tomo = tomo_inds[tomo_seq];
            self.m_sds_off = (tomo * tomo_arr_size) as usize;
            self.m_means_off = (tomo * tomo_arr_size) as usize;
            self.m_bound_off = (tomo * bound_arr_size) as usize;

            // Eliminate points that give a bad thickness if possible
            self.check_block_thicknesses();

            in_header = unsafe { iiu_mrc_header(in_unit_base + tomo, "findsection", 1, 0) };
            if let Some(point_root) = &point_root {
                let name = c_format_bytes(
                    "%s%d-colbound.mod",
                    &[CArg::Bytes(point_root), CArg::Int(tomo as i64)],
                );
                let boundaries = self.m_boundaries[self.m_bound_off..].to_vec();
                self.dump_point_model(
                    &boundaries,
                    num_xblocks * num_yblocks,
                    &String::from_utf8_lossy(&name),
                    unsafe { &*in_header },
                );
            }

            // Now for fitting/smoothing, loop on every block, both surfaces
            for bot_top in 0..2 {
                for iy_block in 0..num_yblocks {
                    for ix_block in 0..num_xblocks {
                        ixy_cen[0] = ix_block;
                        ixy_cen[1] = iy_block;
                        let ind = (2 * (iy_block * num_xblocks + ix_block) + bot_top) as usize;
                        smooth_bound[ind] = -1.;
                        if self.m_boundaries[self.m_bound_off + ind] < 0. {
                            continue;
                        }

                        // Try for a square 5 x 5 region first, but if there are not 2 good
                        // rows of data at least, drop back to 7 x 2 region
                        self.get_fitting_region(
                            &ixy_cen,
                            5,
                            5,
                            bot_top,
                            &mut starts,
                            &mut ends,
                            &mut num_bound,
                            &mut xy_spacing,
                        );
                        if self.m_good_row_col[0] < 2 || self.m_good_row_col[1] < 2 {
                            self.get_fitting_region(
                                &ixy_cen,
                                7,
                                2,
                                bot_top,
                                &mut starts,
                                &mut ends,
                                &mut num_bound,
                                &mut xy_spacing,
                            );
                        }

                        // Set up major and minor axis
                        wgt_col_in = -1;
                        major = 0;
                        if ends[0] - starts[0] < ends[1] - starts[1] {
                            major = 1;
                        }
                        minor = 1 - major;

                        // If there are enough points for robust fit, set the variable list
                        if num_bound >= 6 {
                            self.build_variable_list(major, minor, num_bound, num_bound);
                            max_zero_wgt = num_bound / 6;

                            // Load matrix and do the fit; if it works, record weight column
                            self.load_fitting_matrix(
                                &ixy_cen,
                                &starts,
                                &ends,
                                bot_top,
                                xy_spacing,
                                0.,
                                -1,
                                -1,
                                &mut num_fit,
                            );
                            ierr = robust_regress(
                                &mut self.m_fit_mat,
                                MAX_FIT_DATA as i32,
                                0,
                                self.m_num_vars,
                                num_fit,
                                1,
                                &mut fit_solution,
                                MAX_FIT_COL as i32,
                                Some(&mut fit_const[..]),
                                &mut fit_mean,
                                &mut fit_sd,
                                &mut self.m_fit_work,
                                self.m_kfactor,
                                &mut num_iter,
                                self.m_max_iter,
                                max_zero_wgt,
                                self.m_max_change,
                                self.m_max_oscill,
                            );
                            if self.m_debug_output > 1 {
                                printf!(
                                    "block %d %d bt %d fr %d %d %d %d err %d iter %d  c %.1f\n",
                                    CArg::Int(ix_block as i64),
                                    CArg::Int(iy_block as i64),
                                    CArg::Int(bot_top as i64),
                                    CArg::Int(starts[0] as i64),
                                    CArg::Int(ends[0] as i64),
                                    CArg::Int(starts[1] as i64),
                                    CArg::Int(ends[1] as i64),
                                    CArg::Int(ierr as i64),
                                    CArg::Int(num_iter as i64),
                                    CArg::Dbl(fit_const[0] as f64)
                                );
                            }
                            if ierr == 0 {
                                wgt_col_in = self.m_num_vars + 1;
                            }
                        }

                        num_fit = num_bound;
                        num_good = num_bound;

                        if wgt_col_in > 0 {
                            // If there are weights, count up the number of points still
                            // included in numBound, and # with weights above a higher
                            // threshold in numGood
                            num_bound = 0;
                            num_good = 0;
                            for ixyz in 0..num_fit as usize {
                                let w = self.m_fit_mat[wgt_col_in as usize * MAX_FIT_DATA + ixyz];
                                if w >= wgt_thresh {
                                    num_bound += 1;
                                }
                                if w >= good_thresh {
                                    num_good += 1;
                                }
                            }
                            if self.m_debug_output > 1 {
                                printf!(
                                    "total %d  retain %d  good %d\n",
                                    CArg::Int(num_fit as i64),
                                    CArg::Int(num_bound as i64),
                                    CArg::Int(num_good as i64)
                                );
                            }
                        }

                        // Set up variables for final fit with possible weighting, with
                        // requirements on number to fit as well as number with higher
                        // weighting; load and fit
                        smooth_bound[ind] = self.m_boundaries[self.m_bound_off + ind];
                        if num_bound >= 5 {
                            self.build_variable_list(major, minor, num_bound, num_good);
                            wgt_col_out = self.m_num_vars + 1;
                            self.load_fitting_matrix(
                                &ixy_cen,
                                &starts,
                                &ends,
                                bot_top,
                                xy_spacing,
                                wgt_thresh,
                                wgt_col_in,
                                wgt_col_out,
                                &mut num_fit,
                            );
                            ierr = mult_regress(
                                &self.m_fit_mat,
                                MAX_FIT_DATA as i32,
                                0,
                                self.m_num_vars,
                                num_fit,
                                1,
                                wgt_col_in,
                                &mut fit_solution,
                                MAX_FIT_COL as i32,
                                Some(&mut fit_const[..]),
                                &mut fit_mean,
                                &mut fit_sd,
                                &mut self.m_fit_work,
                            );
                            if self.m_debug_output > 1 {
                                printf!(
                                    "block %d %d bt %d fr %d %d %d %d err %d xm %.1f %.1f %.1f \
b %f %f c %.1f\n",
                                    CArg::Int(ix_block as i64),
                                    CArg::Int(iy_block as i64),
                                    CArg::Int(bot_top as i64),
                                    CArg::Int(starts[0] as i64),
                                    CArg::Int(ends[0] as i64),
                                    CArg::Int(starts[1] as i64),
                                    CArg::Int(ends[1] as i64),
                                    CArg::Int(ierr as i64),
                                    CArg::Dbl(fit_mean[0] as f64),
                                    CArg::Dbl(fit_mean[1] as f64),
                                    CArg::Dbl(fit_mean[2] as f64),
                                    CArg::Dbl(fit_solution[0] as f64),
                                    CArg::Dbl(fit_solution[1] as f64),
                                    CArg::Dbl(fit_const[0] as f64)
                                );
                            }
                            if ierr == 0 {
                                smooth_bound[ind] = fit_const[0];
                            }
                        }
                    }
                }
            }

            if let Some(point_root) = &point_root {
                let name = c_format_bytes(
                    "%s%d-smooth.mod",
                    &[CArg::Bytes(point_root), CArg::Int(tomo as i64)],
                );
                self.dump_point_model(
                    &smooth_bound,
                    num_xblocks * num_yblocks,
                    &String::from_utf8_lossy(&name),
                    unsafe { &*in_header },
                );
            }

            if let Some(surface_name) = &surface_name {
                let nxyz = self.m_nxyz;
                self.make_surface_model(
                    &smooth_bound,
                    &String::from_utf8_lossy(surface_name),
                    unsafe { &*in_header },
                    &nxyz,
                );
            }

            // For outputting tomopitch model, first assign the header for first tomo
            if pitch_name.is_some() {
                let model = pitch_model.as_mut().unwrap();
                if tomo == 0 {
                    imod_set_ref_image(model, unsafe { &*in_header });
                }

                // Add to model for sample tomograms
                if num_tomos > 1 {
                    if self.add_to_pitch_model(model, &smooth_bound, 0, num_yblocks - 1, tomo + 1)
                        == 0
                    {
                        num_pitch_pairs += 1;
                    }
                } else {
                    // Otherwise figure out number of blocks in sample extent
                    if sample_extent != 0 {
                        size = ((self.m_block_centers
                            [(2 * num_xblocks * (num_yblocks - 1) + 1) as usize]
                            - self.m_block_centers[1])
                            / (if num_yblocks - 1 > 1 {
                                num_yblocks - 1
                            } else {
                                1
                            }) as f32) as i32;
                        let v = b3dnint!(sample_extent as f32 / size as f32);
                        size = if 1 > v { 1 } else { v };
                    } else {
                        size = 1;
                    }

                    // Limit number of samples if needed
                    if num_yblocks / size < num_samples {
                        num_samples = num_yblocks / size;
                        printf!(
                            "WARNING: With a block size of %d, there can be only %d samples\n",
                            CArg::Int(self.m_scan_block_size as i64),
                            CArg::Int(num_samples as i64)
                        );
                    }

                    // Indent by one block if there is enough extra stuff
                    let mut ind = 0;
                    if size * num_samples < (num_yblocks - 2 * size) / 2 {
                        ind = size;
                    }

                    // Get spacing between samples and # requiring spacing + 1
                    let div = if 1 > num_samples - 1 {
                        1
                    } else {
                        num_samples - 1
                    };
                    sam_block_space = (num_yblocks - 2 * ind - size) / div;
                    rem = (num_yblocks - 2 * ind - size) % div;

                    // Loop on the samples
                    for loop_ in 0..num_samples {
                        if self.add_to_pitch_model(model, &smooth_bound, ind, ind + size - 1, 0)
                            == 0
                        {
                            num_pitch_pairs += 1;
                        }
                        ind += sam_block_space + if loop_ < rem { 1 } else { 0 };
                    }
                }
            }

            unsafe { iiu_close(in_unit_base + tomo) };
        }

        // Finish tomopitch model
        if let Some(pitch_name) = &pitch_name {
            let model = pitch_model.as_mut().unwrap();
            self.write_model(&String::from_utf8_lossy(pitch_name), model);
            if num_pitch_pairs < 2 {
                exit_error(b"Only one pair of lines was placed in the model for tomopitch");
            }
            if num_pitch_pairs < num_tomos || (num_tomos == 1 && num_pitch_pairs < num_samples) {
                printf!(
                    "WARNING: %s - A pair of lines was found for only %d of the %d samples %s\n",
                    CArg::Bytes(progname),
                    CArg::Int(num_pitch_pairs as i64),
                    CArg::Int(if num_tomos > 1 {
                        num_tomos
                    } else {
                        num_samples
                    } as i64),
                    CArg::Str(if num_tomos > 1 {
                        "tomograms"
                    } else {
                        "positions"
                    })
                );
            }
        }

        // For single tomogram, get the median on each surface and output Z limits
        if num_tomos == 1 {
            for loop_ in 0..2usize {
                patch_lim[loop_] = -2;
                num_stat = 0;
                sign = 2 * loop_ as i32 - 1;
                for ind in 0..(num_xblocks * num_yblocks) as usize {
                    if smooth_bound[2 * ind + loop_] >= 0. {
                        self.m_buffer[num_stat as usize] = smooth_bound[2 * ind + loop_];
                        num_stat += 1;
                    }
                }
                if num_stat > 3 {
                    rs_sort_floats(&mut self.m_buffer, num_stat);
                    rs_median_of_sorted(&self.m_buffer, num_stat, &mut inside_med);

                    // Determine the integer slice that the median plus extra amount occurs in
                    let ti = self.m_thick_ind;
                    let hi = self.m_nxyz[ti] - 1;
                    let v =
                        b3dnint!((inside_med + sign as f32 * self.m_extra_for_pitch) as f64 - 0.5);
                    let minned = if hi < v { hi } else { v };
                    patch_lim[loop_] = if 0 > minned { 0 } else { minned };

                    // Determine the same for absolute limits of surfaces
                    let v = if loop_ != 0 {
                        b3dnint!(
                            (self.m_buffer[(num_stat - 1) as usize] + self.m_extra_for_pitch)
                                as f64
                                - 0.5
                        )
                    } else {
                        b3dnint!((self.m_buffer[0] - self.m_extra_for_pitch) as f64 - 0.5)
                    };
                    let minned = if hi < v { hi } else { v };
                    surface_lim[loop_] = if 0 > minned { 0 } else { minned };

                    // Do combination of the more extreme of a "high" percentile limit, and
                    // a very low percentile limit with some number pixels allowed to be
                    // outside
                    let lp = loop_ as i32;
                    ratio = ((combine_low_pctl * (1 - lp) as f32) as f64
                        + lp as f64 * (1. - combine_low_pctl as f64))
                        as f32;
                    rs_percentile_of_sorted(&self.m_buffer, num_stat, ratio, &mut inside_med);
                    ratio = ((combine_high_pctl * (1 - lp) as f32) as f64
                        + lp as f64 * (1. - combine_high_pctl as f64))
                        as f32;
                    rs_percentile_of_sorted(&self.m_buffer, num_stat, ratio, &mut yy);
                    if loop_ != 0 {
                        let b = inside_med - combine_outside_pix as f32;
                        let m = if yy > b { yy } else { b };
                        combine_lim[loop_] = b3dnint!((m + self.m_extra_for_pitch) as f64 - 0.5);
                    } else {
                        let b = inside_med + combine_outside_pix as f32;
                        let m = if yy < b { yy } else { b };
                        combine_lim[loop_] = b3dnint!((m - self.m_extra_for_pitch) as f64 - 0.5);
                    }
                }
            }

            // Output results if any
            if patch_lim[0] >= 0 && patch_lim[1] >= 0 {
                let nxyz = self.m_nxyz;
                let (mut a, mut b) = (patch_lim[0], patch_lim[1]);
                self.invert_y_if_flipped(&mut a, &mut b, &nxyz);
                patch_lim = [a, b];
                printf!(
                    "Median Z values of surfaces, numbered from 1, are: %d  %d\n",
                    CArg::Int(patch_lim[0] as i64 + 1),
                    CArg::Int(patch_lim[1] as i64 + 1)
                );
                let (mut a, mut b) = (combine_lim[0], combine_lim[1]);
                self.invert_y_if_flipped(&mut a, &mut b, &nxyz);
                combine_lim = [a, b];
                printf!(
                    "Z limits for autopatchfit combine, numbered from 1, are: %d  %d\n",
                    CArg::Int(combine_lim[0] as i64 + 1),
                    CArg::Int(combine_lim[1] as i64 + 1)
                );
                let (mut a, mut b) = (surface_lim[0], surface_lim[1]);
                self.invert_y_if_flipped(&mut a, &mut b, &nxyz);
                surface_lim = [a, b];
                printf!(
                    "Absolute limits of surfaces, numbered from 1, are: %d  %d\n",
                    CArg::Int(surface_lim[0] as i64 + 1),
                    CArg::Int(surface_lim[1] as i64 + 1)
                );
            } else {
                printf!("Too few surface points to determine summary Z values for surface\n");
            }
        }

        let _ = ImodFile::Stdout.flush();
        crate::imod::libcfshr::b3dutil::exit(0);
    }

    /// `FindSect::dumpPointModel` (`findsection.cpp:1272`): write out a model of
    /// points, either the raw column boundaries or the smoothed boundaries.
    fn dump_point_model(
        &self,
        boundaries: &[f32],
        num_pts: i32,
        filename: &str,
        in_header: &MrcHeader,
    ) {
        let Some(mut imod) = imod_new() else {
            exit_error(b"Setting up model");
        };
        if imod_new_object(&mut imod) != 0
            || imod_new_contour(&mut imod).is_err()
            || imod_new_object(&mut imod) != 0
            || imod_new_contour(&mut imod).is_err()
        {
            exit_error(b"Setting up model");
        }
        let flipped = self.m_thick_ind == 1;
        imod_set_ref_image(&mut imod, in_header);
        for pt in 0..num_pts as usize {
            for ind in 0..2 {
                if boundaries[2 * pt + ind] >= 0.
                    && imod_point_append_xyz(
                        &mut imod.obj[ind].cont[0],
                        self.m_block_centers[2 * pt],
                        if flipped {
                            boundaries[2 * pt + ind]
                        } else {
                            self.m_block_centers[2 * pt + 1]
                        },
                        if flipped {
                            self.m_block_centers[2 * pt + 1]
                        } else {
                            boundaries[2 * pt + ind]
                        },
                    ) == 0
                {
                    exit_error(b"Adding point to model");
                }
            }
        }
        imod.obj[0].flags = IMOD_OBJFLAG_SCAT | IMOD_OBJFLAG_PNT_ON_SEC;
        imod.obj[0].pdrawsize = 4;
        imod.obj[1].flags = IMOD_OBJFLAG_SCAT | IMOD_OBJFLAG_PNT_ON_SEC;
        imod.obj[1].pdrawsize = 4;
        self.write_model(filename, &imod);
        imod_delete(&mut imod);
    }

    /// `FindSect::makeSurfaceModel` (`findsection.cpp:1302`): put out a model of
    /// the smoothed boundaries as open contours along the surface, usable for
    /// flattenwarp.
    fn make_surface_model(
        &self,
        boundaries: &[f32],
        filename: &str,
        in_header: &MrcHeader,
        _nxyz: &[i32; 3],
    ) {
        let Some(mut imod) = imod_new() else {
            exit_error(b"Setting up model");
        };
        if imod_new_object(&mut imod) != 0 {
            exit_error(b"Setting up model");
        }
        let flipped = self.m_thick_ind == 1;
        let num_xblocks = self.m_num_blocks[self.m_best_scale][B3D_X];
        let num_yblocks = self.m_num_blocks[self.m_best_scale][3 - self.m_thick_ind];
        let mut z_round: f32;
        imod_set_ref_image(&mut imod, in_header);
        let (xscale, _yscale, _zscale) = mrc_get_scale(in_header);
        imod.xmax = self.m_nxyz[0];
        imod.ymax = self.m_nxyz[1];
        imod.zmax = self.m_nxyz[2];

        // Loop on levels in Y, and on bottom and top surfaces
        for iy in 0..num_yblocks {
            for ind in 0..2usize {
                let mut have_cont = false;

                // Loop across
                for ix in 0..num_xblocks {
                    let pt = (iy * num_xblocks + ix) as usize;
                    if boundaries[2 * pt + ind] >= 0. {
                        // Add contour only when the first point is found
                        if !have_cont {
                            if imod_new_contour(&mut imod).is_err() {
                                exit_error(b"Adding contour to model");
                            }
                            have_cont = true;
                        }
                        z_round = b3dnint!(self.m_block_centers[2 * pt + 1]) as f32;
                        let ci = imod.cindex;
                        let cont = &mut imod.obj[ci.object as usize].cont[ci.contour as usize];
                        if imod_point_append_xyz(
                            cont,
                            self.m_block_centers[2 * pt],
                            if flipped {
                                boundaries[2 * pt + ind]
                            } else {
                                z_round
                            },
                            if flipped {
                                z_round
                            } else {
                                boundaries[2 * pt + ind]
                            },
                        ) == 0
                        {
                            exit_error(b"Adding point to model");
                        }
                    }
                }
            }
        }
        imod.obj[0].flags = IMOD_OBJFLAG_OPEN;
        imod.obj[0].symsize = 7;
        imod.obj[0].symbol = IOBJ_SYM_CIRCLE;
        if xscale != 0. && xscale != 1.0 {
            imod.pixsize = xscale / 10.;
            imod.units = IMOD_UNIT_NM;
        }
        self.write_model(filename, &imod);
        imod_delete(&mut imod);
    }

    /// `FindSect::writeModel` (`findsection.cpp:1356`).
    fn write_model(&self, filename: &str, imod: &Imod) {
        imod_backup_file(filename);
        let fp = ImodFile::open(filename, "wb");
        let Some(mut fp) = fp else {
            exit_error_fmt!("Opening or writing model %s", CArg::Str(filename));
        };
        if imod_write(imod, &mut fp).is_err() {
            exit_error_fmt!("Opening or writing model %s", CArg::Str(filename));
        }
        let _ = fp.flush();
    }

    /// `FindSect::addToPitchModel` (`findsection.cpp:1370`): fit two lines
    /// separately or together with the same slope to the points on bottom and
    /// top surfaces in the given range of blocks in Y.  Use robust fitting if
    /// there are enough points.  Returns 1 if there are inadequate points or a
    /// fitting error.
    fn add_to_pitch_model(
        &mut self,
        imod: &mut Imod,
        boundaries: &[f32],
        y_start: i32,
        y_end: i32,
        time: i32,
    ) -> i32 {
        let mut fit_sd = [0f32; MAX_FIT_COL];
        let mut fit_mean = [0f32; MAX_FIT_COL];
        let mut fit_solution = [0f32; MAX_FIT_COL];
        let flipped = self.m_thick_ind == 1;
        let num_xblocks = self.m_num_blocks[self.m_best_scale][B3D_X];
        let mut err: i32;
        let mut num_pts: i32 = 0;
        let mut num_bot: i32 = 0;
        let mut num_fit: i32;
        let mut max_zero_wgt: i32;
        let mut ind_fit: i32;
        let mut num_iter: i32 = 0;
        let max_pts = (2 * num_xblocks * (y_end + 1 - y_start)) as usize;
        let mut z_round: f32 = 0.;
        let mut y_val: f32;
        let mut y_shift = [0f32; 2];
        let xfit = 0usize;
        let yfit = max_pts;
        let zfit = 2 * max_pts;
        // `slopes[1]`/`intercepts[1]` are left unset when a separate robust fit of
        // the bottom succeeds and the top has too few points for one
        // (`findsection.cpp:1417-1443`); the C reads stack residue there.
        let mut slopes = [0f32; 2];
        let mut intercepts = [0f32; 2];
        let mut xmin = [0f32; 2];
        let mut xmax = [0f32; 2];
        let min_on_side = if self.m_fit_pitch_separately != 0 {
            3
        } else {
            2
        };

        // Loop on bottom and top surfaces
        for ind in 0..2usize {
            xmin[ind] = 1.0e37;
            xmax[ind] = -1.0e37;
            num_bot = num_pts;

            // Load the points into arrays
            for iy in y_start..=y_end {
                for ix in 0..num_xblocks {
                    let pt = (iy * num_xblocks + ix) as usize;
                    if boundaries[2 * pt + ind] >= 0. {
                        let n = num_pts as usize;
                        self.m_buffer[xfit + n] = self.m_block_centers[2 * pt];
                        xmin[ind] = if xmin[ind] < self.m_buffer[xfit + n] {
                            xmin[ind]
                        } else {
                            self.m_buffer[xfit + n]
                        };
                        xmax[ind] = if xmax[ind] > self.m_buffer[xfit + n] {
                            xmax[ind]
                        } else {
                            self.m_buffer[xfit + n]
                        };
                        self.m_buffer[yfit + n] = boundaries[2 * pt + ind];
                        self.m_buffer[zfit + n] = ind as f32;
                        num_pts += 1;
                    }
                }
            }
        }
        if (self.m_fit_pitch_separately != 0 && num_pts < 5)
            || num_bot < min_on_side
            || num_pts - num_bot < min_on_side
        {
            return 1;
        }

        // Apply the criterion in tomopitch for a line too short (there it is 0.1 * 0.9)
        if ((xmax[0] - xmin[0]) as f64) < 0.1 * self.m_nxyz[0] as f64
            || ((xmax[1] - xmin[1]) as f64) < 0.1 * self.m_nxyz[0] as f64
        {
            return 1;
        }

        // Do fit to each surface separately or to both at once, set up slope/intercept
        // for top in the latter case
        err = 1;
        if self.m_fit_pitch_separately != 0 {
            num_fit = num_bot;
            ind_fit = 0;
            for ind in 0..2usize {
                // Use robust fit if there are enough points
                if num_fit >= self.m_min_for_robust_pitch && num_fit <= MAX_FIT_DATA as i32 {
                    for ix in 0..num_fit as usize {
                        self.m_fit_mat[ix] = self.m_buffer[xfit + ix + ind_fit as usize];
                        self.m_fit_mat[MAX_FIT_DATA + ix] =
                            self.m_buffer[yfit + ix + ind_fit as usize];
                    }
                    max_zero_wgt = num_fit / 6;
                    err = robust_regress(
                        &mut self.m_fit_mat,
                        MAX_FIT_DATA as i32,
                        0,
                        1,
                        num_fit,
                        1,
                        &mut fit_solution,
                        MAX_FIT_COL as i32,
                        Some(&mut intercepts[ind..]),
                        &mut fit_mean,
                        &mut fit_sd,
                        &mut self.m_fit_work,
                        self.m_kfactor,
                        &mut num_iter,
                        self.m_max_iter,
                        max_zero_wgt,
                        self.m_max_change,
                        self.m_max_oscill,
                    );
                    slopes[ind] = fit_solution[0];
                }

                // Otherwise, or if there was an error in the robust fit, do standard fit
                if err != 0 {
                    let f = ind_fit as usize;
                    ls_fit(
                        &self.m_buffer[xfit + f..],
                        &self.m_buffer[yfit + f..],
                        num_fit,
                        &mut slopes[ind],
                        &mut intercepts[ind],
                        &mut z_round,
                    );
                }
                num_fit = num_pts - num_bot;
                ind_fit = num_bot;
            }
        } else {
            // Require 2 more points for robust fit
            if num_pts >= self.m_min_for_robust_pitch + 2 && num_pts <= MAX_FIT_DATA as i32 {
                for ix in 0..num_pts as usize {
                    self.m_fit_mat[ix] = self.m_buffer[xfit + ix];
                    self.m_fit_mat[MAX_FIT_DATA + ix] = self.m_buffer[zfit + ix];
                    self.m_fit_mat[2 * MAX_FIT_DATA + ix] = self.m_buffer[yfit + ix];
                }
                max_zero_wgt = num_pts / 6;
                err = robust_regress(
                    &mut self.m_fit_mat,
                    MAX_FIT_DATA as i32,
                    0,
                    2,
                    num_pts,
                    1,
                    &mut fit_solution,
                    MAX_FIT_COL as i32,
                    Some(&mut intercepts[0..]),
                    &mut fit_mean,
                    &mut fit_sd,
                    &mut self.m_fit_work,
                    self.m_kfactor,
                    &mut num_iter,
                    self.m_max_iter,
                    max_zero_wgt,
                    self.m_max_change,
                    self.m_max_oscill,
                );
                slopes[0] = fit_solution[0];
                z_round = fit_solution[1];
            }

            // Do regular fit if robust not tried or failed
            if err != 0 {
                ls_fit2(
                    &self.m_buffer[xfit..],
                    &self.m_buffer[zfit..],
                    &self.m_buffer[yfit..],
                    num_pts,
                    &mut slopes[0],
                    &mut z_round,
                    Some(&mut intercepts[0]),
                );
            }
            slopes[1] = slopes[0];
            intercepts[1] = intercepts[0] + z_round;
        }

        // Find a shift equal to maximum residual on each side
        y_shift[0] = 0.;
        y_shift[1] = 0.;
        for pt in 0..num_pts {
            let ind = if pt < num_bot { 0 } else { 1 };
            let p = pt as usize;
            y_val =
                self.m_buffer[yfit + p] - (self.m_buffer[xfit + p] * slopes[ind] + intercepts[ind]);
            if (ind != 0 && y_val > y_shift[ind]) || (ind == 0 && y_val < y_shift[ind]) {
                y_shift[ind] = y_val;
            }
        }

        // Make the lines
        for ind in 0..2usize {
            y_shift[ind] += (2 * ind as i32 - 1) as f32 * self.m_extra_for_pitch;
            if imod_new_contour(imod).is_err() {
                exit_error(b"Adding contour to model");
            }
            let ci = imod.cindex;
            let cont = &mut imod.obj[ci.object as usize].cont[ci.contour as usize];
            z_round = b3dnint!(
                (self.m_block_centers[(2 * y_start * num_xblocks + 1) as usize]
                    + self.m_block_centers[(2 * y_end * num_xblocks + 1) as usize])
                    as f64
                    / 2.
            ) as f32;
            y_val = xmin[ind] * slopes[ind] + intercepts[ind] + y_shift[ind];
            if imod_point_append_xyz(
                cont,
                xmin[ind],
                if flipped { y_val } else { z_round },
                if flipped { z_round } else { y_val },
            ) == 0
            {
                exit_error(b"Adding point to model");
            }
            y_val = xmax[ind] * slopes[ind] + intercepts[ind] + y_shift[ind];
            if imod_point_append_xyz(
                cont,
                xmax[ind],
                if flipped { y_val } else { z_round },
                if flipped { z_round } else { y_val },
            ) == 0
            {
                exit_error(b"Adding point to model");
            }
            cont.time = time;
        }
        0
    }

    /// `FindSect::setupBlocks` (`findsection.cpp:1504`): determine number of
    /// non-overlapping blocks and number of analyzed points in block for a given
    /// scale index and axis.
    fn setup_blocks(&mut self, num_boxes: i32, scl_ind: usize, ixyz: usize) {
        self.m_num_in_block[scl_ind][ixyz] = self.m_scan_block_size
            / (self.m_box_spacing[scl_ind][ixyz] * self.m_binning[scl_ind][ixyz]);
        let v = self.m_num_in_block[scl_ind][ixyz];
        let minned = if num_boxes < v { num_boxes } else { v };
        self.m_num_in_block[scl_ind][ixyz] = if 1 > minned { 1 } else { minned };
        self.m_num_blocks[scl_ind][ixyz] = num_boxes / self.m_num_in_block[scl_ind][ixyz];
        self.m_num_in_block[scl_ind][ixyz] = num_boxes / self.m_num_blocks[scl_ind][ixyz];
    }

    /// `FindSect::addBoxesToSample` (`findsection.cpp:1517`): for the range of
    /// boxes indicated, and the scaling binInd, add the SD values to the list in
    /// mBuffer and accumulate the sum of means in meanSum.
    ///
    /// The source can be handed an end one past the last box
    /// (`findsection.cpp:860-861`); such an index lands in the next scaling's
    /// statistics, which the whole-allocation indexing here reproduces.  Past
    /// the end of the allocation the C reads heap residue, and 0 is used.
    #[allow(clippy::too_many_arguments)]
    fn add_boxes_to_sample(
        &mut self,
        start_x: i32,
        end_x: i32,
        start_y: i32,
        end_y: i32,
        start_z: i32,
        end_z: i32,
        bin_ind: usize,
        num_stat: &mut i32,
        mean_sum: &mut f64,
    ) {
        let mut box_ind: usize;
        for iz_box in start_z..=end_z {
            for iy_box in start_y..=end_y {
                for ix_box in start_x..=end_x {
                    box_ind = (self.m_stat_start_inds[bin_ind]
                        + (iz_box * self.m_num_boxes[bin_ind][B3D_Y] + iy_box)
                            * self.m_num_boxes[bin_ind][B3D_X]
                        + ix_box) as usize;
                    self.m_buffer[*num_stat as usize] = self
                        .m_sds
                        .get(self.m_sds_off + box_ind)
                        .copied()
                        .unwrap_or(0.);
                    *num_stat += 1;
                    *mean_sum += self
                        .m_means
                        .get(self.m_means_off + box_ind)
                        .copied()
                        .unwrap_or(0.) as f64;
                }
            }
        }
    }

    /// `FindSect::invertYifFlipped` (`findsection.cpp:1538`): adjust a range in
    /// Y to apply to a rotated volume if the data are flipped.
    fn invert_y_if_flipped(&self, start: &mut i32, end: &mut i32, dims: &[i32; 3]) {
        if self.m_thick_ind == B3D_Y {
            let tmp = *start;
            *start = dims[1] - 1 - *end;
            *end = dims[1] - 1 - tmp;
        }
    }

    /// `FindSect::findColumnMidpoint` (`findsection.cpp:1556`).  `buffer` is the
    /// offset into `mBuffer` that the source passes as a pointer.
    ///
    /// For a column of boxes through the thickness dimension, find the midpoint
    /// of the region with structure.  Return value is 1 if the maximum median in
    /// the column is not different enough from edge; 2 if there are too few
    /// median SD's above a criterion; or 3 if the inside positions end up being
    /// crossed.
    #[allow(clippy::too_many_arguments)]
    fn find_column_midpoint(
        &mut self,
        start_x: i32,
        end_x: i32,
        start_y: i32,
        end_y: i32,
        bin_ind: usize,
        buffer: usize,
        iz_inside: &mut [i32; 2],
        iz_mid: &mut i32,
        inside_median: &mut f32,
    ) -> i32 {
        let mut med_sd: f32 = 0.;
        let edge_diff: f32;
        let edge_crit: f32;
        let mut sd_max: f32 = -1.;
        let num_xy_box = ((end_y + 1 - start_y) * (end_x + 1 - start_x)) as usize;
        let mut box_ind: usize;
        let mut dir: i32;
        let mut y_stride: i32 = 1;
        let mut z_stride: i32 = 1;
        let mut z_start: i32;
        let mut z_end: i32;
        let mut num_above: i32;
        let ti = self.m_thick_ind;

        if ti == B3D_Z {
            z_stride = self.m_num_boxes[bin_ind][B3D_Y];
        } else {
            y_stride = self.m_num_boxes[bin_ind][B3D_Y];
        }

        // First find a maximum in the column, taking the median of values across each
        // plane
        for iz_box in 0..self.m_num_boxes[bin_ind][ti] {
            let mut d = 0usize;
            for iy_box in start_y..=end_y {
                for ix_box in start_x..=end_x {
                    box_ind = (self.m_stat_start_inds[bin_ind]
                        + (iz_box * z_stride + iy_box * y_stride)
                            * self.m_num_boxes[bin_ind][B3D_X]
                        + ix_box) as usize;
                    self.m_buffer[buffer + d] = self.m_sds[self.m_sds_off + box_ind];
                    d += 1;
                }
            }
            rs_fast_median_in_place(&mut self.m_buffer[buffer..], num_xy_box as i32, &mut med_sd);
            sd_max = if sd_max > med_sd { sd_max } else { med_sd };
            self.m_buffer[buffer + num_xy_box + iz_box as usize] = med_sd;
            if let Some(medians) = self.m_col_medians {
                self.m_col_slice[medians + iz_box as usize] = med_sd;
            }
        }

        // If difference from edge is below criterion, return error
        edge_diff = sd_max - self.m_edge_medians[bin_ind];
        if edge_diff < self.m_col_max_edge_diff_crit * self.m_edge_madns[bin_ind] {
            return 1;
        }

        // Criterion is maximum of a number of MADNs above edge and a fraction of max-edge
        // diff
        {
            let a = edge_diff * self.m_frac_col_max_edge_diff;
            let b = self.m_crit_edge_madn * self.m_edge_madns[bin_ind];
            edge_crit = self.m_edge_medians[bin_ind] + if a > b { a } else { b };
        }

        // From each direction, find an edge above the criterion
        z_start = 0;
        z_end = self.m_num_boxes[bin_ind][ti];
        for loop_ in 0..2usize {
            dir = 1 - 2 * loop_ as i32;
            num_above = 0;
            let mut iz_box = z_start;
            while iz_box != z_end {
                if self.m_buffer[buffer + num_xy_box + iz_box as usize] >= edge_crit {
                    num_above += 1;
                    if num_above >= self.m_num_high_inside_crit {
                        iz_inside[loop_] = iz_box;
                        break;
                    }
                }
                iz_box += dir;
            }
            if num_above < self.m_num_high_inside_crit {
                return 2;
            }
            z_start = self.m_num_boxes[bin_ind][ti] - 1;
            z_end = -1;
        }

        if iz_inside[0] > iz_inside[1] {
            return 3;
        }
        *iz_mid = (iz_inside[0] + iz_inside[1]) / 2;
        rs_fast_median_in_place(
            &mut self.m_buffer[buffer + num_xy_box + iz_inside[0] as usize..],
            iz_inside[1] + 1 - iz_inside[0],
            inside_median,
        );
        0
    }

    /// `FindSect::fitColumnBoundaries` (`findsection.cpp:1630`): finds
    /// boundaries of a column defined by the range of boxes in X and Y, looking
    /// outward from middle, and fits a line to the falling phase to find the
    /// boundary at a set level.
    fn fit_column_boundaries(
        &mut self,
        start_x: i32,
        end_x: i32,
        start_y: i32,
        end_y: i32,
        boundary: &mut [f32; 2],
    ) {
        let mut box_ind: usize;
        let mut ind: usize;
        let mut dir: i32;
        let mut y_stride: i32 = 1;
        let mut z_stride: i32 = 1;
        let mut num_extra: i32 = 0;
        let mut iz_mid: i32 = 0;
        let mut iz_inside = [0i32; 2];
        let mut num_in_col = [0i32; 2];
        let mut inside_med: f32 = 0.;
        let fall_crit: f32;
        let low_fit_crit: f32;
        let high_fit_crit: f32;
        let boundary_level: f32;
        let pitch_level: f32;
        let mut last_sd: f32 = 0.;
        let mut slope: f32 = 0.;
        let mut intercept: f32 = 0.;
        let mut ro: f32 = 0.;
        let mut extra_med: f32 = 0.;
        let mut iz: i32;
        let mut iz_add: i32;
        let mut fit_start: i32;
        let mut num_fit: i32;
        let scl = self.m_best_scale;
        let ti = self.m_thick_ind;
        let z_range = self.m_num_boxes[scl][ti];
        let max_in_col = ((end_x + 1 - start_x) * (end_y + 1 - start_y)) as usize;
        let xfit = 4 * max_in_col;
        let yfit = xfit + z_range as usize;
        let sds_base = self.m_sds_off + self.m_stat_start_inds[scl] as usize;

        if ti == B3D_Z {
            z_stride = self.m_num_boxes[scl][B3D_Y];
        } else {
            y_stride = self.m_num_boxes[scl][B3D_Y];
        }
        boundary[0] = -1.;
        boundary[1] = -1.;

        // Find midpoint and inside median, reject if it is too low
        if self.find_column_midpoint(
            start_x,
            end_x,
            start_y,
            end_y,
            scl,
            0,
            &mut iz_inside,
            &mut iz_mid,
            &mut inside_med,
        ) != 0
        {
            return;
        }
        if inside_med
            < self.m_cen_medians[scl] - self.m_column_to_cen_madn_crit * self.m_cen_madns[scl]
        {
            if self.m_debug_output > 1 {
                printf!(
                    "skipping %d %d low median %f\n",
                    CArg::Int(start_x as i64),
                    CArg::Int(start_y as i64),
                    CArg::Dbl(inside_med as f64)
                );
            }
            return;
        }

        // Set various criteria
        let em = self.m_edge_medians[scl];
        fall_crit = em + self.m_max_falloff_frac * (inside_med - em);
        low_fit_crit = em + self.m_low_fit_frac * (inside_med - em);
        high_fit_crit = em + self.m_high_fit_frac * (inside_med - em);
        boundary_level = em + self.m_boundary_frac * (inside_med - em);
        pitch_level = em + self.m_pitch_boundary_frac * (inside_med - em);

        let nbx = self.m_num_boxes[scl][B3D_X];
        num_in_col[0] = 0;
        num_in_col[1] = 0;
        for iy_box in start_y..=end_y {
            for ix_box in start_x..=end_x {
                for loop_ in 0..2usize {
                    dir = 2 * loop_ as i32 - 1;
                    num_fit = 0;
                    fit_start = -1;
                    let mut iz_box = iz_mid;
                    while iz_box >= 0 && iz_box < z_range {
                        box_ind = sds_base
                            + ((iz_box * z_stride + iy_box * y_stride) * nbx + ix_box) as usize;
                        let sd = self.m_sds[box_ind];
                        if sd < fall_crit {
                            // Going below the fall criterion triggers various checks
                            // First, if the starting point is below it, skip this box column
                            if iz_box == iz_mid {
                                break;
                            }

                            // If this point is below the low fit criterion or is higher than
                            // the last, start fit on previous point, unless it is above the
                            // high criterion; in which case start on this one
                            if sd < low_fit_crit || sd > last_sd {
                                fit_start = iz_box - dir;
                                if last_sd > high_fit_crit {
                                    fit_start = iz_box;
                                }

                                // But if we are at end of range, start fit on this point
                            } else if iz_box == 0 || iz_box == z_range - 1 {
                                fit_start = iz_box;
                            }

                            if fit_start >= 0 {
                                // Add points to fit arrays until it goes above the high crit
                                iz = fit_start;
                                while iz != iz_mid {
                                    ind = sds_base
                                        + ((iz * z_stride + iy_box * y_stride) * nbx + ix_box)
                                            as usize;
                                    if self.m_sds[ind] > high_fit_crit {
                                        break;
                                    }
                                    self.m_buffer[xfit + num_fit as usize] = iz as f32;
                                    self.m_buffer[yfit + num_fit as usize] = self.m_sds[ind];
                                    num_fit += 1;
                                    iz -= dir;
                                }

                                // If there is only one point, add one that is out of the fit
                                // range in the other direction if possible
                                if num_fit == 1 {
                                    iz = b3dnint!(self.m_buffer[xfit]);
                                    iz_add = -1;
                                    if (self.m_buffer[yfit] as f64)
                                        < 0.5 * (low_fit_crit + high_fit_crit) as f64
                                    {
                                        if iz - dir != iz_mid {
                                            iz_add = iz - dir;
                                        }
                                    } else if iz + dir >= 0 && iz + dir < z_range {
                                        iz_add = iz + dir;
                                    }
                                    if iz_add >= 0 {
                                        ind = sds_base
                                            + ((iz_add * z_stride + iy_box * y_stride) * nbx
                                                + ix_box)
                                                as usize;
                                        self.m_buffer[xfit + num_fit as usize] = iz_add as f32;
                                        self.m_buffer[yfit + num_fit as usize] = self.m_sds[ind];
                                        num_fit += 1;
                                    }
                                }

                                // Do the fit if at least 2 points and get Z value at boundary
                                // level
                                if num_fit >= 2 {
                                    ls_fit(
                                        &self.m_buffer[xfit..],
                                        &self.m_buffer[yfit..],
                                        num_fit,
                                        &mut slope,
                                        &mut intercept,
                                        &mut ro,
                                    );
                                    self.m_buffer
                                        [loop_ * max_in_col + num_in_col[loop_] as usize] =
                                        (boundary_level - intercept) / slope;
                                    num_in_col[loop_] += 1;
                                    self.m_buffer[2 * max_in_col + num_extra as usize] =
                                        ((boundary_level - pitch_level) as f64 / slope as f64).abs()
                                            as f32;
                                    num_extra += 1;
                                }
                                break;
                            }
                        }
                        last_sd = sd;
                        iz_box += dir;
                    }
                }
            }
        }

        // Get median of boundary values if there are enough of them
        for loop_ in 0..2usize {
            // `mMinFracBoundsInCol * maxInCol` is a float product, widened for the max
            let m = (self.m_min_frac_bounds_in_col * max_in_col as f32) as f64;
            if num_in_col[loop_] >= b3dnint!(if 1. > m { 1. } else { m }) {
                rs_fast_median_in_place(
                    &mut self.m_buffer[loop_ * max_in_col..],
                    num_in_col[loop_],
                    &mut boundary[loop_],
                );
            }
        }

        // Get the median of extra Z values if there are enough, and add to sum
        let m = self.m_min_frac_bounds_in_col as f64 * 2. * max_in_col as f64;
        if num_extra >= b3dnint!(if 1. > m { 1. } else { m }) {
            self.m_num_extra_sum += 1;
            rs_fast_median_in_place(
                &mut self.m_buffer[2 * max_in_col..],
                num_extra,
                &mut extra_med,
            );
            self.m_extra_pitch_sum += extra_med * self.m_binning[scl][ti] as f32;
        }
    }

    /// `FindSect::checkBlockThicknesses` (`findsection.cpp:1766`): tries to
    /// identify blocks that have insufficient thickness and eliminate one or
    /// both boundaries.
    fn check_block_thicknesses(&mut self) {
        let scl = self.m_best_scale;
        let y_ind = 3 - self.m_thick_ind;
        let num_xblocks = self.m_num_blocks[scl][B3D_X];
        let num_yblocks = self.m_num_blocks[scl][y_ind];
        let mut ixy_cen = [0i32; 2];
        let mut starts = [0i32; 2];
        let mut ends = [0i32; 2];
        let mut num_bound: i32 = 0;
        let mut xy_spacing: f32 = 0.;
        let mut mean_bound: f32 = 0.;
        let mut mean_diff = [0f32; 2];
        let mut tmp1: f32 = 0.;
        let mut tmp2: f32 = 0.;
        let mut num_madns: f32;
        let mut min_bound: i32 = 10000;
        let bo = self.m_bound_off;

        if self.m_num_thicknesses < self.m_min_num_thick_for_check {
            return;
        }

        // Look for outlier thicknesses
        for iy_block in 0..num_yblocks {
            for ix_block in 0..num_xblocks {
                let ind = bo + (2 * (iy_block * num_xblocks + ix_block)) as usize;
                if self.m_boundaries[ind] < 0.
                    || self.m_boundaries[ind + 1] < 0.
                    || (self.m_boundaries[ind + 1] - self.m_boundaries[ind]) / self.m_thick_median
                        > self.m_too_thin_crit
                {
                    continue;
                }

                if self.m_debug_output != 0 {
                    printf!(
                        "block too thin %d %d  %.1f\n",
                        CArg::Int(ix_block as i64),
                        CArg::Int(iy_block as i64),
                        CArg::Dbl((self.m_boundaries[ind + 1] - self.m_boundaries[ind]) as f64)
                    );
                }
                ixy_cen[0] = ix_block;
                ixy_cen[1] = iy_block;

                // Find a sampling region and extract boundaries for ones not involved in an
                // outlier thickness
                for bot_top in 0..2 {
                    self.get_fitting_region(
                        &ixy_cen,
                        5,
                        5,
                        bot_top,
                        &mut starts,
                        &mut ends,
                        &mut num_bound,
                        &mut xy_spacing,
                    );
                    if self.m_good_row_col[0] < 2 || self.m_good_row_col[1] < 2 {
                        self.get_fitting_region(
                            &ixy_cen,
                            7,
                            2,
                            bot_top,
                            &mut starts,
                            &mut ends,
                            &mut num_bound,
                            &mut xy_spacing,
                        );
                    }
                    num_bound = 0;
                    let bt = bot_top as usize;
                    for ix in starts[0]..=ends[0] {
                        for iy in starts[1]..=ends[1] {
                            let jj = bo + (2 * (iy * num_xblocks + ix)) as usize;
                            if self.m_boundaries[jj + bt] >= 0.
                                && (self.m_boundaries[jj + 1 - bt] < 0.
                                    || (self.m_boundaries[jj + 1] - self.m_boundaries[jj])
                                        / self.m_thick_median
                                        > self.m_too_thin_crit)
                            {
                                self.m_buffer[num_bound as usize] = self.m_boundaries[jj + bt];
                                num_bound += 1;
                            }
                        }
                    }
                    min_bound = if min_bound < num_bound {
                        min_bound
                    } else {
                        num_bound
                    };
                    mean_diff[bt] = -1.;
                    if num_bound > 2 {
                        if num_bound > 5 {
                            rs_fast_median_in_place(&mut self.m_buffer, num_bound, &mut mean_bound);
                        } else {
                            avg_sd(
                                &self.m_buffer,
                                num_bound,
                                &mut mean_bound,
                                &mut tmp1,
                                &mut tmp2,
                            );
                        }
                        mean_diff[bt] =
                            ((self.m_boundaries[ind + bt] - mean_bound) as f64).abs() as f32;
                    }
                }

                // Evaluate the difference from the local mean for each boundary
                // Drop it if it is sufficiently larger than the difference for the other
                // boundary or if it is bigger than a fraction of the median thickness
                if mean_diff[0] >= 0. && mean_diff[1] >= 0. {
                    if mean_diff[0] > self.m_farther_from_mean_crit * mean_diff[1]
                        || mean_diff[0] > self.m_mean_diff_thick_frac * self.m_thick_median
                    {
                        if self.m_debug_output != 0 {
                            printf!(
                                "Dropping bottom %.1f diffs %.1f %.1f\n",
                                CArg::Dbl(self.m_boundaries[ind] as f64),
                                CArg::Dbl(mean_diff[0] as f64),
                                CArg::Dbl(mean_diff[1] as f64)
                            );
                        }
                        self.m_boundaries[ind] = -1.;
                    }
                    if mean_diff[1] > self.m_farther_from_mean_crit * mean_diff[0]
                        || mean_diff[1] > self.m_mean_diff_thick_frac * self.m_thick_median
                    {
                        if self.m_debug_output != 0 {
                            printf!(
                                "Dropping top %.1f diffs %.1f %.1f\n",
                                CArg::Dbl(self.m_boundaries[ind + 1] as f64),
                                CArg::Dbl(mean_diff[0] as f64),
                                CArg::Dbl(mean_diff[1] as f64)
                            );
                        }
                        self.m_boundaries[ind + 1] = -1.;
                    }
                }

                // If that didn't eliminate either, apply an outlier criterion to the
                // deviation from median thickness as long as there are at least two
                // boundary points left
                num_madns = (self.m_thick_median
                    - (self.m_boundaries[ind + 1] - self.m_boundaries[ind]))
                    / self.m_thick_madn;
                if self.m_boundaries[ind] >= 0.
                    && self.m_boundaries[ind + 1] >= 0.
                    && min_bound >= 2
                    && num_madns > self.m_crit_thick_madn
                {
                    if self.m_debug_output != 0 {
                        printf!(
                            "Dropping both: %.1f MADNs from median\n",
                            CArg::Dbl(num_madns as f64)
                        );
                    }
                    self.m_boundaries[ind] = -1.;
                    self.m_boundaries[ind + 1] = -1.;
                }
            }
        }
    }

    /// `FindSect::getFittingRegion` (`findsection.cpp:1861`): for a block whose
    /// X, Y center is in blockCen, finds a region to fit with desired size
    /// extentNum in the major direction, and maximum extent maxDepth in the
    /// other, for the surface in botTop.
    #[allow(clippy::too_many_arguments)]
    fn get_fitting_region(
        &mut self,
        block_cen: &[i32; 2],
        extent_num: i32,
        max_depth: i32,
        bot_top: i32,
        block_start: &mut [i32],
        block_end: &mut [i32],
        num_pts: &mut i32,
        xy_spacing: &mut f32,
    ) {
        let scl = self.m_best_scale;
        let y_ind = 3 - self.m_thick_ind;
        let num_xblocks = self.m_num_blocks[scl][B3D_X];
        let num_yblocks = self.m_num_blocks[scl][y_ind];
        let mut stride = [0i32; 2];
        let mut num_blk = [0i32; 2];
        let mut extent_axis: i32 = -1;
        let distance: f32;
        let mut spacing = [0f32; 2];
        let mut shifted: bool;
        let mut ind: i32;
        let mut other: usize;
        let mut end: i32;
        let mut start: i32;
        let dep_axis: usize;
        let mut num_good: i32;
        stride[0] = 2;
        stride[1] = 2 * num_xblocks;
        let nblocks = |ixy: usize| self.m_num_blocks[scl][ixy * y_ind];

        // Find direction that defines the extent
        for ixy in 0..2usize {
            start = if 0 > block_cen[ixy] - 1 {
                0
            } else {
                block_cen[ixy] - 1
            };
            let a = nblocks(ixy) - 1;
            end = if a < start + 2 { a } else { start + 2 };
            start = if 0 > end - 2 { 0 } else { end - 2 };
            if start == end {
                extent_axis = 1 - ixy as i32;
                continue;
            }
            ind = block_cen[1 - ixy] * stride[1 - ixy] + ixy as i32;
            spacing[ixy] = (self.m_block_centers[(end * stride[ixy] + ind) as usize]
                - self.m_block_centers[(start * stride[ixy] + ind) as usize])
                / (end - start) as f32;
        }

        // If either axis is feasible, take the one with the bigger number of blocks if
        // either is limited, or the one with smaller spacing
        if extent_axis < 0 {
            if num_xblocks < extent_num || num_yblocks < extent_num {
                extent_axis = if num_xblocks < num_yblocks { 1 } else { 0 };
            } else {
                extent_axis = if (spacing[1] as f64) < 0.9 * spacing[0] as f64 {
                    1
                } else {
                    0
                };
            }
        }
        let extent_axis = extent_axis as usize;
        dep_axis = 1 - extent_axis;
        num_blk[extent_axis] = extent_num;
        *xy_spacing = if 1. > spacing[extent_axis] as f64 {
            1.
        } else {
            spacing[extent_axis]
        };

        // Get the number in the other direction: shoot for equal extent, limit by maximum
        if spacing[dep_axis] == 0. {
            num_blk[dep_axis] = 1;
        } else {
            distance = spacing[extent_axis] * (extent_num - 1) as f32;
            num_blk[dep_axis] = b3dnint!(1. + (distance / spacing[dep_axis]) as f64);
            num_blk[dep_axis] = if num_blk[dep_axis] < max_depth {
                num_blk[dep_axis]
            } else {
                max_depth
            };
        }

        // Get the start and end in each direction
        for ixy in 0..2usize {
            let a = block_cen[ixy] - num_blk[ixy] / 2;
            block_start[ixy] = if 0 > a { 0 } else { a };
            let a = nblocks(ixy) - 1;
            let b = block_start[ixy] + num_blk[ixy] - 1;
            block_end[ixy] = if a < b { a } else { b };
            let a = block_end[ixy] + 1 - num_blk[ixy];
            block_start[ixy] = if 0 > a { 0 } else { a };
            num_blk[ixy] = block_end[ixy] + 1 - block_start[ixy];
        }

        // Try to slide region if there are empty rows on one side
        for ixy in 0..2usize {
            shifted = false;
            other = 1 - ixy;
            while !self.has_boundaries(
                ixy,
                block_start[ixy],
                block_start[other],
                block_end[other],
                bot_top,
            ) && self.has_boundaries(
                ixy,
                block_end[ixy] + 1,
                block_start[other],
                block_end[other],
                bot_top,
            ) && block_start[ixy] < block_cen[ixy]
            {
                shifted = true;
                block_start[ixy] += 1;
                block_end[ixy] += 1;
                if self.m_debug_output > 1 {
                    printf!("Shifted + on axis %d\n", CArg::Int(ixy as i64));
                }
            }
            while !shifted
                && !self.has_boundaries(
                    ixy,
                    block_end[ixy],
                    block_start[other],
                    block_end[other],
                    bot_top,
                )
                && self.has_boundaries(
                    ixy,
                    block_start[ixy] - 1,
                    block_start[other],
                    block_end[other],
                    bot_top,
                )
                && block_end[ixy] > block_cen[ixy]
            {
                block_start[ixy] -= 1;
                block_end[ixy] -= 1;
                if self.m_debug_output > 1 {
                    printf!("Shifted - on axis %d\n", CArg::Int(ixy as i64));
                }
            }
        }

        // Count the good rows and columns and total points
        for ixy in 0..2usize {
            other = 1 - ixy;

            // For counting in a direction, this determines number good in other direction
            // Loop on other direction
            self.m_good_row_col[other] = 0;
            *num_pts = 0;
            for outer in block_start[other]..=block_end[other] {
                num_good = 0;

                // Loop on main direction and count ones with data
                for inner in block_start[ixy]..=block_end[ixy] {
                    if self.m_boundaries[self.m_bound_off
                        + (inner * stride[ixy] + outer * stride[other] + bot_top) as usize]
                        >= 0.
                    {
                        num_good += 1;
                    }
                }

                // The line is good if it has at least half of the blocks with data
                let a = num_blk[ixy] as f64 / 2. - 0.1;
                if num_good as f64 >= if 0.99 > a { 0.99 } else { a } {
                    self.m_good_row_col[other] += 1;
                }
                *num_pts += num_good;
            }
        }
    }

    /// `FindSect::hasBoundaries` (`findsection.cpp:1974`): tests whether there
    /// are any boundaries along one axis, at the given row or column, between
    /// start and end on the other axis, on surface given by botTop.
    fn has_boundaries(
        &self,
        axis: usize,
        row_col: i32,
        start: i32,
        end: i32,
        bot_top: i32,
    ) -> bool {
        let mut stride = [2i32, 2];
        stride[1] = 2 * self.m_num_blocks[self.m_best_scale][B3D_X];
        if row_col < 0
            || row_col >= self.m_num_blocks[self.m_best_scale][axis * (3 - self.m_thick_ind)]
        {
            return false;
        }
        for ind in start..=end {
            if self.m_boundaries[self.m_bound_off
                + (row_col * stride[axis] + ind * stride[1 - axis] + bot_top) as usize]
                >= 0.
            {
                return true;
            }
        }
        false
    }

    /// `FindSect::loadFittingMatrix` (`findsection.cpp:1989`).
    #[allow(clippy::too_many_arguments)]
    fn load_fitting_matrix(
        &mut self,
        ixy_cen: &[i32; 2],
        starts: &[i32],
        ends: &[i32],
        bot_top: i32,
        xy_spacing: f32,
        wgt_thresh: f32,
        wgt_col_in: i32,
        wgt_col_out: i32,
        num_fit: &mut i32,
    ) {
        let mut ixy: i32;
        let mut ind_wgt: i32 = -1;
        let nbx = self.m_num_blocks[self.m_best_scale][B3D_X];
        *num_fit = 0;
        let cen_ind = (2 * (ixy_cen[0] + ixy_cen[1] * nbx)) as usize;
        for iy_block in starts[1]..=ends[1] {
            for ix_block in starts[0]..=ends[0] {
                let blk_ind = (2 * (ix_block + iy_block * nbx)) as usize;
                if self.m_boundaries[self.m_bound_off + blk_ind + bot_top as usize] >= 0. {
                    ind_wgt += 1;
                    if wgt_col_in > 0
                        && self.m_fit_mat[wgt_col_in as usize * MAX_FIT_DATA + ind_wgt as usize]
                            < wgt_thresh
                    {
                        continue;
                    }
                    let nf = *num_fit as usize;
                    if wgt_col_in > 0 && wgt_col_out > 0 {
                        self.m_fit_mat[wgt_col_out as usize * MAX_FIT_DATA + nf] =
                            self.m_fit_mat[wgt_col_in as usize * MAX_FIT_DATA + ind_wgt as usize];
                    }
                    for var in 0..self.m_num_vars as usize {
                        ixy = self.m_var_list[var][0];
                        self.m_fit_mat[var * MAX_FIT_DATA + nf] = (self.m_block_centers
                            [blk_ind + ixy as usize]
                            - self.m_block_centers[cen_ind + ixy as usize])
                            / xy_spacing;
                        ixy = self.m_var_list[var][1];
                        if ixy >= 0 {
                            self.m_fit_mat[var * MAX_FIT_DATA + nf] *= (self.m_block_centers
                                [blk_ind + ixy as usize]
                                - self.m_block_centers[cen_ind + ixy as usize])
                                / xy_spacing;
                        }
                    }
                    self.m_fit_mat[self.m_num_vars as usize * MAX_FIT_DATA + nf] =
                        self.m_boundaries[self.m_bound_off + blk_ind + bot_top as usize];
                    *num_fit += 1;
                }
            }
        }
    }

    /// `FindSect::buildVariableList` (`findsection.cpp:2026`): sets up indices
    /// for a polynomial fit appropriate to the number of points and number of
    /// "good" points with higher weights.
    fn build_variable_list(&mut self, major: i32, minor: i32, num_bound: i32, num_good: i32) {
        self.m_num_vars = 1;
        self.m_var_list[0][0] = major;
        self.m_var_list[0][1] = -1;
        let good_minor = self.m_good_row_col[minor as usize];
        if num_bound >= 10 && num_good > 8 && good_minor >= 2 {
            self.m_var_list[1][0] = minor;
            self.m_var_list[1][1] = -1;
            self.m_num_vars = 2;
        }
        if num_bound >= 14 && num_good > 11 && good_minor >= 2 {
            self.m_var_list[2][0] = major;
            self.m_var_list[2][1] = major;
            self.m_num_vars = 3;
        }
        if num_bound >= 20 && num_good > 17 && good_minor >= 4 {
            self.m_var_list[3][0] = major;
            self.m_var_list[3][1] = minor;
            self.m_var_list[4][0] = minor;
            self.m_var_list[4][1] = minor;
            self.m_num_vars = 5;
        }
    }

    /// `FindSect::findLowestSDEdges` (`findsection.cpp:2054`): does samples
    /// through the entire thickness extent with the size of the edge samples and
    /// searches for a peak in the middle and dips on either side.
    fn find_lowest_sd_edges(
        &mut self,
        starts: &mut [i32; 5],
        ends: &mut [i32; 5],
        scl: usize,
    ) -> i32 {
        let mut num_stat: i32;
        let num_samples: i32;
        let sample_thick: i32;
        let divided_extent: i32;
        let samp_spacing: i32;
        let mut samp: i32;
        let mut ind: usize;
        let mut mean_sum: f64;
        let mut peak: f32;
        let mut dip: f32;
        let mut dip_ind = [0i32; 2];
        let mut peak_ind: i32;
        let mut num_above: i32;
        let above_dip_madn_crit: f32 = 2.;
        let above_dip_num_crit: i32 = 3;
        let ti = self.m_thick_ind;

        // Get thickness, number of samples and spacing
        sample_thick = starts[3];
        starts[ti] = 0;
        ends[ti] = starts[3] - 1;
        samp_spacing = if 1 > sample_thick / 2 {
            1
        } else {
            sample_thick / 2
        };
        divided_extent = self.m_num_boxes[scl][ti] - (sample_thick - samp_spacing);
        num_samples = (divided_extent + samp_spacing - 1) / samp_spacing;
        let mut edge_medians = vec![0f32; num_samples.max(0) as usize];
        let mut edge_madns = vec![0f32; num_samples.max(0) as usize];
        for samp in 0..num_samples as usize {
            num_stat = 0;
            mean_sum = 0.;
            let a = samp as i32 * samp_spacing + sample_thick - 1;
            ends[ti] = if a < ends[4] { a } else { ends[4] };
            starts[ti] = ends[ti] + 1 - sample_thick;
            self.add_boxes_to_sample(
                starts[0],
                ends[0],
                starts[1],
                ends[1],
                starts[2],
                ends[2],
                scl,
                &mut num_stat,
                &mut mean_sum,
            );
            let (head, tail) = self.m_buffer.split_at_mut(num_stat as usize);
            rs_fast_median(head, num_stat, tail, &mut edge_medians[samp]);
            rs_fast_madn(
                head,
                num_stat,
                edge_medians[samp],
                tail,
                &mut edge_madns[samp],
            );
            if self.m_debug_output > 1 {
                printf!(
                    "%d %d %8.3f %8.3f\n",
                    CArg::Int(starts[ti] as i64),
                    CArg::Int(ends[ti] as i64),
                    CArg::Dbl(edge_medians[samp] as f64),
                    CArg::Dbl(edge_madns[samp] as f64)
                );
            }
        }

        // Find peak, disqualifying a peak at the end of the range
        peak = -1.;
        peak_ind = -1;
        samp = 1;
        while samp < num_samples - 1 {
            let s = samp as usize;
            if edge_medians[s] > edge_medians[s - 1]
                && edge_medians[s] > edge_medians[s + 1]
                && edge_medians[s] > peak
            {
                peak = edge_medians[s];
                peak_ind = samp;
            }
            samp += 1;
        }
        if peak_ind < 0 {
            return 1;
        }

        // Find minima on each side of peak
        let mut direc = -1;
        while direc <= 1 {
            ind = ((direc + 1) / 2) as usize;
            samp = peak_ind + direc;
            dip = (peak as f64 + 10.) as f32;
            while samp >= 0 && samp < num_samples {
                if edge_medians[samp as usize] < dip {
                    dip = edge_medians[samp as usize];
                    dip_ind[ind] = samp;
                }
                samp += direc;
            }
            if dip >= peak {
                return 2;
            }
            direc += 2;
        }

        // Sanity check: are the two dips on their respective sides of the middle?
        if dip_ind[0] >= num_samples / 2 || dip_ind[1] <= num_samples / 2 {
            return 3;
        }

        // And structure test: how many samples are above n MADNs from the dip?
        num_above = 0;
        ind = dip_ind[0] as usize;
        if edge_medians[ind] > edge_medians[dip_ind[1] as usize] {
            ind = dip_ind[1] as usize;
        }
        for samp in dip_ind[0] + 1..dip_ind[1] {
            if (edge_medians[samp as usize] - edge_medians[ind]) / edge_madns[ind]
                > above_dip_madn_crit
            {
                num_above += 1;
            }
        }
        if num_above < above_dip_num_crit {
            return 4;
        }

        // Set the best edges to middle of dip samples
        if dip_ind[0] > 0 {
            self.m_best_low_edge[scl] = (dip_ind[0] + 1) * samp_spacing;
        }
        if dip_ind[1] < ends[4] {
            self.m_best_high_edge[scl] = (dip_ind[1] + 1) * samp_spacing;
        }

        // Set the limits for the edge samples; i.e. set absolute limits in s/e[4] and
        // make up excluded region in s/e[3]
        starts[4] = self.m_best_low_edge[scl];
        ends[4] = self.m_best_high_edge[scl];
        starts[3] = starts[4] + sample_thick;
        ends[3] = ends[4] - sample_thick;
        if self.m_debug_output != 0 {
            printf!(
                "Set new limits for edge at %d %d\n",
                CArg::Int(self.m_best_low_edge[scl] as i64),
                CArg::Int(self.m_best_high_edge[scl] as i64)
            );
        }
        0
    }

    /// `FindSect::analyzeHighSD` (`findsection.cpp:2154`): separate analysis of
    /// thickness based on distribution of high SD values and on optional bead
    /// model.
    #[allow(clippy::too_many_arguments)]
    fn analyze_high_sd(
        &mut self,
        high_sd_crit: f32,
        bead_model: Option<&Imod>,
        box_size: &[i32; 3],
        start_coord: &[i32; 3],
        end_coord: &[i32; 3],
        pitch_model: Option<&mut Imod>,
    ) -> i32 {
        let mut box_ind: usize;
        let mut ix_box: i32;
        let mut iy_box: i32;
        let mut iz_box: i32;
        let mut thick: i32;
        let mut ix: i32;
        let mut iy: i32;
        let mut ind: i32;
        let mut mid_ind: i32;
        let mut ind_max: i32 = 0;
        let mut bin_width: i32 = 0;
        let mut num_bins: i32 = 0;
        let mut best_mult: usize;
        let mut dir: i32;
        let mut mid_at_rise: i32;
        let mut num_low: i32;
        let mut redo: i32;
        let mut num_fit: i32;
        let mut num_above: i32;
        let mut cross_crit = [0f32; 2];
        let mut cross_base = [0f32; 2];
        let mut peak_slope = [0f32; 2];
        let mut proj_slope = [0f32; 2];
        let mut proj_intcp = [0f32; 2];
        let mut base_at_crit = [0f32; 2];
        let mut max_bin: f32 = 0.;
        let mut sem: f32 = 0.;
        let mut crit: f32;
        let mut bead_y: f32;
        let mut proj_min: f32;
        let mut proj_peak: f32 = 0.;
        let mut extrap: f32;
        let mut base: f32;
        let mut above_crit: f32;
        let mut ro: f32 = 0.;
        let mut cumul: f32;
        let mut base_avg: f32 = 0.;
        let mut base_sd: f32 = 0.;
        let mut z_col: f32;
        let mut zfrac: f32;
        let x_coeff: f32;
        let y_coeff: f32;
        let z_coeff: f32;
        let mut xy_const: f32;
        let mut z_slice: f32;
        let mut x_col: f32;
        let mut y_col: f32;
        let mut slope: f32 = 0.;
        let mut intcp: f32 = 0.;
        let slope_diff: f32;
        let peak_mean: f32;
        let edge_mean: f32;
        let mut num_dips = [0i32; MAX_MULT];
        let mut level_pt = Ipoint::default();
        let mut pitch_pt = Ipoint::default();
        let scl = self.m_best_scale;
        let mut min_base: i32;
        let ti = self.m_thick_ind;
        let y_ind = 3 - ti;
        let flipped = ti != 2;
        let da: [f32; MAX_AMVAR] = [2., 2.];
        let mut a: [f32; MAX_AMVAR] = [0., 0.];
        let mut yy = [0f32; MAX_AMVAR + 1];
        let ptol_facs: [f32; MAX_AMVAR] = [0.05, 0.003];
        let ftol_facs: [f32; MAX_AMVAR] = [5.0e-4, 1.0e-5];
        let delfac: f32 = 2.;
        let mut iter: i32 = 0;
        let nvar: usize;
        let alpha: f32;
        let beta: f32;
        let thickness: f32;
        let mut col_min: f32 = 0.;
        let mut col_max: f32 = 0.;
        let mut col_val: f32;
        let mut spread: f32 = 0.;
        let cos_alpha: f32;
        let sin_alpha: f32;
        let mut bins: Vec<f32> = Vec::new();
        let mut combo_bins: Vec<f32> = Vec::new();
        let mut proj_medians: Vec<f32> = Vec::new();
        let mut proj_pct75: Vec<f32> = Vec::new();
        let mut proj_pct90: Vec<f32> = Vec::new();
        let mut num_in_slice: i32;
        let mut iz: i32;
        let mono_crit: f32 = 0.1;
        let min_base_frac: f32 = 0.002;
        let all_rising_crit: f32 = 3.;
        let rise_backoff_crit: f32 = 2.;
        let max_rise_backoff: i32 = 3;
        let max_low_frac_past_rise: f32 = 0.15;
        let max_bump_low_frac: f32 = 0.1;
        let mut bump_sum: f32;
        let mut bump_crit: f32;
        let bump_sum_crit_fac: f32 = 20.;
        let mut good_bump: bool;
        let xcen = (self.m_nxyz[0] as f64 / 2.) as f32;
        let ycen = (self.m_nxyz[y_ind] as f64 / 2.) as f32;
        let zcen = (self.m_nxyz[ti] as f64 / 2.) as f32;
        let binnings = self.m_binning[scl];
        let spacings = self.m_box_spacing[scl];
        let num_boxes = self.m_num_boxes[scl];
        let mut mat: Imat = imod_mat_new(3).unwrap();
        nvar = 2;

        self.m_best_box_size = *box_size;
        self.m_start_coord = *start_coord;
        self.m_num_beads = 0;
        if let Some(bead_model) = bead_model {
            for obj in &bead_model.obj {
                for cont in &obj.cont {
                    self.m_num_beads += cont.pts.len() as i32;
                }
            }
        }

        let sds_base = self.m_sds_off + self.m_stat_start_inds[scl] as usize;

        // Loop on columns twice looking for points above threshold
        for loop_ in 0..2 {
            self.m_num_high_sd = 0;
            for ilong in 0..num_boxes[y_ind] {
                for ix_box in 0..num_boxes[B3D_X] {
                    col_min = 1.0e10;
                    col_max = -1.0e10;
                    for thick in self.m_best_low_edge[scl]..=self.m_best_high_edge[scl] {
                        iz_box = if flipped { ilong } else { thick };
                        iy_box = if flipped { thick } else { ilong };
                        box_ind = sds_base
                            + ((iz_box * num_boxes[B3D_Y] + iy_box) * num_boxes[B3D_X] + ix_box)
                                as usize;
                        if (self.m_sds[box_ind] - self.m_edge_medians[scl]) / self.m_edge_madns[scl]
                            > high_sd_crit
                        {
                            if loop_ == 0 {
                                // First time, just counting columns with such points
                                self.m_num_high_sd += 1;
                                break;
                            } else {
                                // Second time, get Z value and keep track of min/max
                                col_val = (start_coord[ti] as f64
                                    + binnings[ti] as f64
                                        * ((thick * spacings[ti]) as f64
                                            + box_size[ti] as f64 / 2.))
                                    as f32;
                                col_min = if col_min < col_val { col_min } else { col_val };
                                col_max = if col_max > col_val { col_max } else { col_val };
                            }
                        }
                    }

                    // End of column, add the min and max points,
                    if loop_ != 0 && col_max >= col_min {
                        let n = self.m_num_high_sd as usize;
                        self.m_block_centers[2 * n] = (start_coord[B3D_X] as f64
                            + binnings[B3D_X] as f64
                                * ((ix_box * spacings[B3D_X]) as f64 + box_size[B3D_X] as f64 / 2.))
                            as f32;
                        self.m_block_centers[2 * n + 1] = (start_coord[y_ind] as f64
                            + binnings[y_ind] as f64
                                * ((ilong * spacings[y_ind]) as f64 + box_size[y_ind] as f64 / 2.))
                            as f32;
                        self.m_boundaries[2 * n] = col_min;
                        self.m_boundaries[2 * n + 1] = col_max;
                        self.m_num_high_sd += 1;
                    }
                }
            }

            // First time allocate
            if loop_ == 0 {
                if self.m_num_high_sd == 0 {
                    exit_error(b"There are no boxes with SD above the criterion value");
                }
                if self.m_debug_output != 0 {
                    printf!("numhigh %d\n", CArg::Int(self.m_num_high_sd as i64));
                }
                let nh = self.m_num_high_sd as usize;
                let nb = self.m_num_beads as usize;
                self.m_block_centers = vec![0.; 2 * nh + 2 * nb];
                self.m_boundaries = vec![0.; 2 * nh + nb];
                self.m_bound_off = 0;
                self.m_bound_rot = vec![0.; 3 * nh + nb];
                let nbins = (self.m_nxyz[ti] / spacings[ti] + 5) as usize;
                bins = vec![0.; nbins];
                combo_bins = vec![0.; nbins];
                self.m_proj_slice = vec![0.; (num_boxes[y_ind] * num_boxes[B3D_X]) as usize];
                let n = num_boxes[ti] as usize;
                self.m_proj_means = vec![0.; n];
                proj_medians = vec![0.; n];
                proj_pct75 = vec![0.; n];
                proj_pct90 = vec![0.; n];
            }
        }
        let bead_centers = 2 * self.m_num_high_sd as usize;
        let bead_bounds = 2 * self.m_num_high_sd as usize;

        // Put bead positions on the ends of these arrays
        if let Some(bead_model) = bead_model {
            self.m_num_beads = 0;
            for obj in &bead_model.obj {
                for cont in &obj.cont {
                    for pt in &cont.pts {
                        bead_y = if flipped { pt.z } else { pt.y };
                        if pt.x >= start_coord[B3D_X] as f32
                            && pt.x <= end_coord[B3D_X] as f32
                            && bead_y >= start_coord[y_ind] as f32
                            && bead_y <= end_coord[y_ind] as f32
                        {
                            let n = self.m_num_beads as usize;
                            self.m_block_centers[bead_centers + n * 2] = pt.x;
                            self.m_block_centers[bead_centers + n * 2 + 1] = bead_y;
                            self.m_boundaries[bead_bounds + n] = if flipped { pt.y } else { pt.z };
                            self.m_num_beads += 1;
                        }
                    }
                }
            }
            if self.m_num_beads != 0 {
                let a = 2 * self.m_num_high_sd;
                let ind = 2 * if a > self.m_num_beads {
                    a
                } else {
                    self.m_num_beads
                };
                self.m_convex_xtmp = vec![0.; ind as usize];
                self.m_convex_ytmp = vec![0.; ind as usize];
                self.m_convex_cont = imod_contour_new();
                if let Some(cont) = self.m_convex_cont.as_mut() {
                    cont.pts = vec![Ipoint::default(); ind as usize];
                }
                if self.m_convex_cont.is_none() {
                    exit_error(b"Allocating arrays for convex hulls");
                }
            }
        }

        // Run amoeba fit twice and get solution
        self.m_simplex_iter = 0;
        self.m_after_simplex = false;
        {
            // `amoebaFunc` (`findsection.cpp:2926`): the callback into the instance.
            let mut amoeba_func = |yvec: &[f32]| -> f32 {
                let mut spread = 0.;
                self.spread_func(yvec, &mut spread);
                spread
            };
            dual_amoeba(
                &mut yy,
                nvar,
                delfac,
                &ptol_facs,
                &ftol_facs,
                &mut a,
                &da,
                &mut amoeba_func,
                &mut iter,
            );
        }
        self.m_after_simplex = true;
        self.spread_func(&a, &mut spread);

        // Get final rotation matrix first for forward then for back rotation
        alpha = a[0];
        beta = a[1];
        let _ = imod_mat_rot(&mut mat, -beta as f64, Axis3::Y);
        let _ = imod_mat_rot(&mut mat, -alpha as f64, Axis3::X);
        cos_alpha = (alpha as f64 * RADIANS_PER_DEGREE).cos() as f32;
        sin_alpha = (alpha as f64 * RADIANS_PER_DEGREE).sin() as f32;
        x_coeff = mat.data[2];
        y_coeff = mat.data[6];
        z_coeff = mat.data[10];
        imod_mat_id(&mut mat);
        let _ = imod_mat_rot(&mut mat, alpha as f64, Axis3::X);
        let _ = imod_mat_rot(&mut mat, beta as f64, Axis3::Y);

        if self.m_proj_layer_stat_type > 0 {
            // If using projection layers, fetch data from slices at this rotation and
            // average them
            let all_sds_len = self.m_sds.len();
            for thick in 0..num_boxes[ti] {
                num_in_slice = 0;
                num_above = 0;
                z_slice = (start_coord[ti] as f64
                    + binnings[ti] as f64
                        * ((thick * spacings[ti]) as f64 + box_size[ti] as f64 / 2.))
                    as f32;
                for ilong in 0..num_boxes[y_ind] {
                    for ix_box in 0..num_boxes[B3D_X] {
                        x_col = (start_coord[B3D_X] as f64
                            + binnings[B3D_X] as f64
                                * ((ix_box * spacings[B3D_X]) as f64 + box_size[B3D_X] as f64 / 2.))
                            as f32;
                        y_col = (start_coord[y_ind] as f64
                            + binnings[y_ind] as f64
                                * ((ilong * spacings[y_ind]) as f64 + box_size[y_ind] as f64 / 2.))
                            as f32;
                        xy_const = x_coeff * (x_col - xcen) + y_coeff * (y_col - ycen);
                        z_col =
                            ((z_slice - zcen - xy_const) / z_coeff + zcen) / binnings[ti] as f32;
                        if z_col >= 0. && z_col <= num_boxes[ti] as f32 {
                            iz = z_col as i32;
                            zfrac = z_col - iz as f32;
                            iz_box = if flipped { ilong } else { iz };
                            iy_box = if flipped { iz } else { ilong };
                            box_ind = sds_base
                                + ((iz_box * num_boxes[B3D_Y] + iy_box) * num_boxes[B3D_X] + ix_box)
                                    as usize;
                            let n = num_in_slice as usize;
                            let this = if box_ind < all_sds_len {
                                self.m_sds[box_ind]
                            } else {
                                0.
                            };
                            self.m_proj_slice[n] = ((1. - zfrac as f64) * this as f64) as f32;

                            // `iz + 1` can be one past the last box (`findsection.cpp:2359`);
                            // inside the allocation that is the next data, past it
                            // the C reads residue and 0 is used here.
                            iz_box = if flipped { ilong } else { iz + 1 };
                            iy_box = if flipped { iz + 1 } else { ilong };
                            box_ind = sds_base
                                + ((iz_box * num_boxes[B3D_Y] + iy_box) * num_boxes[B3D_X] + ix_box)
                                    as usize;
                            let next = if box_ind < all_sds_len {
                                self.m_sds[box_ind]
                            } else {
                                0.
                            };
                            self.m_proj_slice[n] = (self.m_proj_slice[n] + zfrac * next
                                - self.m_edge_medians[scl])
                                / self.m_edge_madns[scl];
                            if self.m_proj_slice[n] > high_sd_crit {
                                num_above += 1;
                            }
                            num_in_slice += 1;
                        }
                    }
                }
                let t = thick as usize;
                avg_sd(
                    &self.m_proj_slice,
                    num_in_slice,
                    &mut self.m_proj_means[t],
                    &mut col_min,
                    &mut col_max,
                );
                rs_fast_median_in_place(&mut self.m_proj_slice, num_in_slice, &mut proj_medians[t]);
                proj_pct75[t] = percentile_float(
                    b3dnint!(0.75 * num_in_slice as f64),
                    &mut self.m_proj_slice,
                    num_in_slice,
                );
                proj_pct90[t] = num_above as f32 / num_in_slice as f32;
                if self.m_debug_output > 1 {
                    printf!(
                        "%d  %.3f  %.3f %.3f %.3f\n",
                        CArg::Int(b3dnint!(z_slice) as i64),
                        CArg::Dbl(self.m_proj_means[t] as f64),
                        CArg::Dbl(proj_medians[t] as f64),
                        CArg::Dbl(proj_pct75[t] as f64),
                        CArg::Dbl(proj_pct90[t] as f64)
                    );
                }
            }

            // Assign the chosen array.  `projTemp` aliases one of the percentile
            // arrays and is overwritten with the indices below.
            let mut proj_means = std::mem::take(&mut self.m_proj_means);
            let (proj_stat, proj_temp): (&mut Vec<f32>, &mut Vec<f32>) =
                if self.m_proj_layer_stat_type == 1 {
                    (&mut proj_means, &mut proj_pct90)
                } else if self.m_proj_layer_stat_type == 2 {
                    (&mut proj_medians, &mut proj_pct90)
                } else if self.m_proj_layer_stat_type == 3 {
                    (&mut proj_pct75, &mut proj_pct90)
                } else {
                    (&mut proj_pct90, &mut proj_pct75)
                };

            // Find peak of projected values
            ind_max = -1;
            for ind in 0..num_boxes[ti] {
                if ind_max < 0 || proj_stat[ind as usize] > proj_peak {
                    proj_peak = proj_stat[ind as usize];
                    ind_max = ind;
                }
            }
            // `projStat[B3DMAX(0, indMax + 1)]` (`findsection.cpp:2403`) reads one past
            // the array when the peak is last; that residue is taken as 0 here.
            peak_mean = ((proj_stat[ind_max as usize]
                + proj_stat[if 0 > ind_max - 1 { 0 } else { ind_max - 1 } as usize]
                + proj_stat
                    .get(if 0 > ind_max + 1 { 0 } else { ind_max + 1 } as usize)
                    .copied()
                    .unwrap_or(0.)) as f64
                / 3.) as f32;
            let i3 = (num_boxes[ti] - 3) as usize;
            edge_mean = ((proj_stat[0]
                + proj_stat[1]
                + proj_stat[2]
                + proj_stat[i3]
                + proj_stat[i3 + 1]
                + proj_stat[i3 + 2]) as f64
                / 6.) as f32;
            if self.m_debug_output != 0 {
                printf!(
                    "Peak %.3f  edge %.3f  difference %.3f\n",
                    CArg::Dbl(peak_mean as f64),
                    CArg::Dbl(edge_mean as f64),
                    CArg::Dbl((peak_mean - edge_mean) as f64)
                );
            }

            // Fit to the bottom and the top at the edges
            num_fit = if 10 > self.m_best_low_edge[scl] + 1 {
                10
            } else {
                self.m_best_low_edge[scl] + 1
            };
            for ind in 0..num_boxes[ti] as usize {
                proj_temp[ind] = ind as f32;
            }
            ls_fit(
                proj_temp,
                proj_stat,
                num_fit,
                &mut proj_slope[0],
                &mut proj_intcp[0],
                &mut ro,
            );
            ind = num_fit;
            let a = num_boxes[ti] - self.m_best_high_edge[scl];
            num_fit = if 10 > a { 10 } else { a };
            thick = num_boxes[ti] - num_fit;
            ls_fit(
                &proj_temp[thick as usize..],
                &proj_stat[thick as usize..],
                num_fit,
                &mut proj_slope[1],
                &mut proj_intcp[1],
                &mut ro,
            );

            // For each side, walk in from fit looking for point that is up by the criterion
            // above the baseline, where the base is the maximum of the minimum value seen
            // and the current extrapolation from the baseline
            for loop_ in 0..2usize {
                proj_min = proj_stat[ind as usize];
                dir = 1 - 2 * loop_ as i32;
                cross_crit[loop_] = -1.;
                while dir * (ind_max - ind) > 0 {
                    let i = ind as usize;
                    extrap = proj_slope[loop_] * ind as f32 + proj_intcp[loop_];
                    proj_min = if proj_min < proj_stat[i] {
                        proj_min
                    } else {
                        proj_stat[i]
                    };
                    base = if proj_min > extrap { proj_min } else { extrap };
                    if self.m_debug_output > 1 {
                        printf!(
                            "extrap = %g,  projMin = %g,  base = %g,  projStat[ind] = %g\n",
                            CArg::Dbl(extrap as f64),
                            CArg::Dbl(proj_min as f64),
                            CArg::Dbl(base as f64),
                            CArg::Dbl(proj_stat[i] as f64)
                        );
                    }
                    above_crit = proj_stat[i] - (base + self.m_proj_edge_crit * (proj_peak - base));

                    // Get position where it crosses criterion, and extrapolate back to
                    // baseline from immediate slope.  Also get slope from here to peak
                    if above_crit >= 0. {
                        peak_slope[loop_] = ((proj_peak - proj_stat[i]) as f64
                            / (ind as f64 - ind_max as f64).abs())
                            as f32;
                        cross_crit[loop_] = ind as f32
                            - dir as f32 * above_crit
                                / (proj_stat[i] - proj_stat[(ind + dir) as usize]);
                        let s = (ind - 2 * loop_ as i32) as usize;
                        ls_fit(
                            &proj_temp[s..],
                            &proj_stat[s..],
                            3,
                            &mut slope,
                            &mut intcp,
                            &mut ro,
                        );
                        cross_base[loop_] = (base - intcp) / slope;
                        base_at_crit[loop_] = base;
                        break;
                    }
                    ind += dir;
                }

                // Starting point for other direction
                ind = thick - 1;
            }

            // If only one side exists or if other side has steep slope, redo that side
            // with the base from the good side
            if cross_crit[0] < 0. && cross_crit[1] < 0. {
                self.m_proj_means = proj_means;
                return 1;
            }
            slope_diff = ((proj_slope[0].abs() - proj_slope[1].abs()) as f64).abs() as f32;
            redo = -1;
            if cross_crit[0] < 0. || peak_slope[0] - proj_slope[0] < slope_diff {
                redo = 0;
            }
            if cross_crit[1] < 0. || peak_slope[1] - proj_slope[1] < slope_diff {
                redo = 1;
            }
            if redo >= 0 {
                let r = redo as usize;
                cross_crit[r] = -1.;
                base = base_at_crit[1 - r];
                ind = if redo > 0 { num_boxes[ti] - 1 } else { 0 };
                dir = 1 - 2 * redo;
                while dir * (ind_max - ind) > 0 {
                    let i = ind as usize;
                    above_crit = proj_stat[i] - (base + self.m_proj_edge_crit * (proj_peak - base));
                    if above_crit >= 0. {
                        cross_crit[r] = ind as f32
                            - dir as f32 * above_crit
                                / (proj_stat[i] - proj_stat[(ind + dir) as usize]);
                        let s = (ind - 2 * redo) as usize;
                        ls_fit(
                            &proj_temp[s..],
                            &proj_stat[s..],
                            3,
                            &mut slope,
                            &mut intcp,
                            &mut ro,
                        );
                        cross_base[r] = (base - intcp) / slope;
                        break;
                    }
                    ind += dir;
                }
                if cross_crit[r] < 0. {
                    self.m_proj_means = proj_means;
                    return 1;
                }
            }
            self.m_proj_means = proj_means;

            // scale both sets of numbers
            for ind in 0..2usize {
                cross_crit[ind] = (start_coord[ti] as f64
                    + binnings[ti] as f64
                        * ((cross_crit[ind] * spacings[ti] as f32) as f64
                            + box_size[ti] as f64 / 2.)) as f32;
                cross_base[ind] = (start_coord[ti] as f64
                    + binnings[ti] as f64
                        * ((cross_base[ind] * spacings[ti] as f32) as f64
                            + box_size[ti] as f64 / 2.)) as f32;
            }
            printf!(
                "Surfaces at crossing point %.1f  %.1f  thick %1.f\n",
                CArg::Dbl(cross_crit[0] as f64),
                CArg::Dbl(cross_crit[1] as f64),
                CArg::Dbl((cross_crit[1] - cross_crit[0]) as f64)
            );
            printf!(
                "Surfaces extrapolated to base %.1f  %.1f  thick %1.f\n",
                CArg::Dbl(cross_base[0] as f64),
                CArg::Dbl(cross_base[1] as f64),
                CArg::Dbl((cross_base[1] - cross_base[0]) as f64)
            );
            if self.m_proj_use_extrap != 0 {
                self.m_bound_low = cross_base[0];
                self.m_bound_high = cross_base[1];
            } else {
                self.m_bound_low = cross_crit[0];
                self.m_bound_high = cross_crit[1];
            }
        } else {
            // Otherwise (default) work with distributions of column boundaries

            // Analyze slightly smoothed histograms for minimal bumpiness above 0.1 of peak
            let nh = self.m_num_high_sd as usize;
            for loop_ in 0..2usize {
                for mult in 1..MAX_MULT {
                    let values = self.m_bound_rot[loop_ * nh..loop_ * nh + nh].to_vec();
                    self.make_combined_bins(
                        &values,
                        self.m_num_high_sd,
                        &mut bins,
                        &mut combo_bins,
                        mult as i32,
                        &mut num_bins,
                        &mut bin_width,
                        &mut max_bin,
                        &mut ind_max,
                    );
                    if self.m_debug_output != 0 {
                        printf!(
                            "numBins = %d,  binWidth = %d,  maxBin = %g,  indMax = %d\n",
                            CArg::Int(num_bins as i64),
                            CArg::Int(bin_width as i64),
                            CArg::Dbl(max_bin as f64),
                            CArg::Int(ind_max as i64)
                        );
                    }

                    // Count non-monotonic points above criterion level
                    num_dips[mult] = 0;
                    for ind in 1..(num_bins - 1) as usize {
                        if combo_bins[ind] >= mono_crit * max_bin
                            && combo_bins[ind] < combo_bins[ind - 1]
                            && combo_bins[ind] < combo_bins[ind + 1]
                        {
                            num_dips[mult] += 1;
                        }
                    }
                    if self.m_debug_output != 0 {
                        printf!(
                            "mult = %d,  numDips[mult] = %d\n",
                            CArg::Int(mult as i64),
                            CArg::Int(num_dips[mult] as i64)
                        );
                    }
                    if num_dips[mult] == 0 {
                        break;
                    }
                }
                best_mult = 1;
                for mult in 1..MAX_MULT {
                    // Done if 0 dips
                    if num_dips[mult] == 0 {
                        best_mult = mult;
                        break;
                    }

                    // New minimum, take it.  Otherwise stop when no new minimum and 1 or 2
                    // dips, or 2nd time after no new minimum with 3 dips
                    if num_dips[mult] < num_dips[best_mult] {
                        best_mult = mult;
                    } else if (num_dips[best_mult] < 3 && mult - best_mult > 0)
                        || (num_dips[best_mult] == 3 && mult - best_mult >= 2)
                    {
                        break;
                    }
                }
                if best_mult == MAX_MULT {
                    exit_error_fmt!(
                        "Could not get smooth enough histogram even by combining bins by %d",
                        CArg::Int(MAX_MULT as i64)
                    );
                }
                if self.m_debug_output != 0 {
                    printf!("bestMult = %d\n", CArg::Int(best_mult as i64));
                }
                let values = self.m_bound_rot[loop_ * nh..loop_ * nh + nh].to_vec();
                self.make_combined_bins(
                    &values,
                    self.m_num_high_sd,
                    &mut bins,
                    &mut combo_bins,
                    best_mult as i32,
                    &mut num_bins,
                    &mut bin_width,
                    &mut max_bin,
                    &mut ind_max,
                );
                if ind_max <= 3 || ind_max >= num_bins - 4 {
                    exit_error_fmt!(
                        "The peak in a histogram with bins combined by %d is too close to \
the edge of the volume",
                        CArg::Int(best_mult as i64)
                    );
                }
                if self.m_debug_output > 1 {
                    for ind in 0..num_bins as usize {
                        printf!(
                            "%d  %.1f\n",
                            CArg::Int(ind as i64),
                            CArg::Dbl(combo_bins[ind] as f64)
                        );
                    }
                }

                // find minimum baseline
                {
                    let v = min_base_frac * self.m_num_high_sd as f32;
                    min_base = (if 2. > v { 2. } else { v }) as i32;
                }
                if loop_ != 0 {
                    dir = -1;
                    ind = num_bins - 1;
                } else {
                    dir = 1;
                    ind = 0;
                }

                // Advance past minimum but make sure not to get too close to peak
                cumul = 0.;
                while cumul < min_base as f32 {
                    cumul += bins[ind as usize];
                    ind += dir;
                }
                if loop_ != 0 {
                    let a = num_bins - 3;
                    let minned = if a < ind { a } else { ind };
                    ind = if ind_max + 4 > minned {
                        ind_max + 4
                    } else {
                        minned
                    };
                } else {
                    let a = ind_max - 4;
                    let minned = if a < ind { a } else { ind };
                    ind = if 3 > minned { 3 } else { minned };
                }
                mid_ind = ind;

                // Look for point where it rises; get baseline mean and SD
                mid_at_rise = -1;
                while dir * (ind_max - 3 - mid_ind) > 0 {
                    let (start, count) = if loop_ != 0 {
                        (mid_ind as usize, num_bins - 1 - mid_ind)
                    } else {
                        (0usize, mid_ind)
                    };
                    avg_sd(
                        &combo_bins[start..],
                        count,
                        &mut base_avg,
                        &mut base_sd,
                        &mut sem,
                    );
                    crit = base_avg + all_rising_crit * base_sd;

                    // If next three bins are all above triggering criterion, make sure bins
                    // stay high to peak, or that a contiguous "bump" contains a lot of
                    // counts above the criterion
                    if base_sd > 0.
                        && combo_bins[(mid_ind + dir) as usize] > crit
                        && combo_bins[(mid_ind + 2 * dir) as usize] > crit
                        && combo_bins[(mid_ind + 3 * dir) as usize] > crit
                    {
                        bump_crit = bump_sum_crit_fac * crit;
                        crit = base_avg + rise_backoff_crit * base_sd;
                        num_low = 0;
                        bump_sum = 0.;
                        good_bump = false;
                        ind = 1;
                        while ind < dir * (ind_max - mid_ind) {
                            let v = combo_bins[(mid_ind + ind * dir) as usize];
                            if v <= crit {
                                num_low += 1;
                            }
                            bump_sum += v;
                            if bump_sum > bump_crit
                                && num_low <= b3dnint!(max_bump_low_frac * ind as f32)
                            {
                                good_bump = true;
                            }
                            ind += 1;
                        }
                        if !good_bump
                            && num_low
                                > b3dnint!(
                                    max_low_frac_past_rise * (dir * (ind_max - mid_ind) - 4) as f32
                                )
                        {
                            mid_ind += dir;
                            continue;
                        }

                        // If it passes that, then try to back off from higher criterion
                        ind = 0;
                        while ind < max_rise_backoff {
                            if combo_bins[mid_ind as usize] <= crit {
                                break;
                            }
                            mid_ind -= dir;
                            ind += 1;
                        }
                        mid_at_rise = mid_ind;
                        break;
                    }
                    mid_ind += dir;
                }

                // Error if no good spot found
                if ind < 0 {
                    exit_error(b"Could not find distinct rise above baseline in histogram");
                }
                if self.m_debug_output != 0 {
                    printf!(
                        "%s rise after bin %d\n",
                        CArg::Str(if loop_ != 0 { "High:" } else { "Low: " }),
                        CArg::Int(mid_at_rise as i64)
                    );
                }

                if loop_ != 0 {
                    self.m_bound_high = (mid_at_rise * bin_width) as f32;
                } else {
                    self.m_bound_low = ((mid_at_rise + 1) * bin_width) as f32;
                }
            }
        }

        if self.m_boost_high_sd_thickness > 0. {
            zfrac = (0.5
                * (self.m_boost_high_sd_thickness * (self.m_bound_high - self.m_bound_low)) as f64)
                as f32;
            {
                let v = self.m_bound_low - zfrac;
                self.m_bound_low = if 0. > v { 0. } else { v };
            }
            {
                let a = self.m_nxyz[ti] as f64 - 1.;
                let b = (self.m_bound_high + zfrac) as f64;
                self.m_bound_high = (if a < b { a } else { b }) as f32;
            }
        }

        // Now combine results from beads
        if self.m_num_beads != 0 {
            self.compute_bead_limits(self.m_bead_excl_pctl_thick);
            let v = self.m_bead_low + zcen;
            self.m_bound_low = if self.m_bound_low < v {
                self.m_bound_low
            } else {
                v
            };
            let v = self.m_bead_high + zcen;
            self.m_bound_high = if self.m_bound_high > v {
                self.m_bound_high
            } else {
                v
            };
        }

        thickness = self.m_bound_high - self.m_bound_low;

        if self.m_debug_output > 0 {
            printf!(
                "Iterations %d  spread measure %.2f\n",
                CArg::Int(self.m_simplex_iter as i64),
                CArg::Dbl(spread as f64)
            );
        }
        ix = b3dnint!(self.m_bound_low);
        iy = b3dnint!(self.m_bound_high);
        let nxyz = self.m_nxyz;
        self.invert_y_if_flipped(&mut ix, &mut iy, &nxyz);
        printf!(
            "Thickness = %.1f  boundaries (from 1) = %d %d  alpha = %.2f  beta = %.2f\n",
            CArg::Dbl(thickness as f64),
            CArg::Int(ix as i64 + 1),
            CArg::Int(iy as i64 + 1),
            CArg::Dbl(alpha as f64),
            CArg::Dbl(beta as f64)
        );
        let Some(pitch_model) = pitch_model else {
            return 0;
        };

        // Make the tomopitch model, end up with lines in the same slice
        for iy in -1..=1 {
            for loop_ in 0..2 {
                if imod_new_contour(pitch_model).is_err() {
                    exit_error(b"Adding contour to model");
                }
                let ci = pitch_model.cindex;
                let cont = &mut pitch_model.obj[ci.object as usize].cont[ci.contour as usize];
                pitch_pt.y = ((iy * self.m_nxyz[y_ind]) as f64 / 3.) as f32;
                level_pt.z = (if loop_ != 0 {
                    self.m_bound_high
                } else {
                    self.m_bound_low
                }) - zcen;
                level_pt.y = (pitch_pt.y + level_pt.z * sin_alpha) / cos_alpha;
                let mut ix = -1;
                while ix <= 1 {
                    level_pt.x = ((ix * self.m_nxyz[0]) as f64 * 0.84 / 2.) as f32;
                    imod_mat_transform(&mat, &level_pt, &mut pitch_pt);
                    if imod_point_append_xyz(
                        cont,
                        pitch_pt.x + xcen,
                        if flipped {
                            pitch_pt.z + zcen
                        } else {
                            pitch_pt.y + ycen
                        },
                        if flipped {
                            pitch_pt.y + ycen
                        } else {
                            pitch_pt.z + zcen
                        },
                    ) == 0
                    {
                        exit_error(b"Adding point to model");
                    }
                    ix += 2;
                }
            }
        }
        0
    }

    /// `FindSect::makeCombinedBins` (`findsection.cpp:2687`): make a simple
    /// histogram of the given depth values with the given multiplier of the
    /// basic Z spacing and then make simple weighted combination of 3 adjacent
    /// bins in comboBins.
    #[allow(clippy::too_many_arguments)]
    fn make_combined_bins(
        &self,
        values: &[f32],
        num_vals: i32,
        bins: &mut [f32],
        combo_bins: &mut [f32],
        mult: i32,
        num_bins: &mut i32,
        bin_width: &mut i32,
        max_bin: &mut f32,
        ind_max: &mut i32,
    ) {
        let ti = self.m_thick_ind;
        *bin_width = mult * self.m_box_spacing[self.m_best_scale][ti];
        *num_bins = self.m_nxyz[ti] / *bin_width;
        let first_val = (-0.5 * self.m_nxyz[ti] as f64) as f32;
        let last_val = first_val + (*bin_width * *num_bins) as f32;
        let nb = *num_bins as usize;
        kernel_histogram(
            &values[..num_vals as usize],
            &mut bins[..nb],
            first_val,
            last_val,
            0.,
            0,
        );

        // Make combined bins and find peak
        *ind_max = -1;
        for ind in 0..*num_bins {
            let lo = (if 0 > ind - 1 { 0 } else { ind - 1 }) as usize;
            let hi = (if *num_bins - 1 < ind + 1 {
                *num_bins - 1
            } else {
                ind + 1
            }) as usize;
            combo_bins[ind as usize] = (0.25 * bins[lo] as f64
                + 0.5 * bins[ind as usize] as f64
                + 0.25 * bins[hi] as f64) as f32;
            if *ind_max < 0 || *max_bin < combo_bins[ind as usize] {
                *ind_max = ind;
                *max_bin = combo_bins[ind as usize];
            }
        }
    }

    /// `FindSect::spreadFunc` (`findsection.cpp:2714`): the actual function for
    /// computing a normalized measure of spreading.
    pub fn spread_func(&mut self, yvec: &[f32], spread: &mut f32) {
        let ti = self.m_thick_ind;
        let xcen = (self.m_nxyz[0] as f64 / 2.) as f32;
        let ycen = (self.m_nxyz[3 - ti] as f64 / 2.) as f32;
        let zcen = (self.m_nxyz[ti] as f64 / 2.) as f32;
        let mut mat: Imat = imod_mat_new(3).unwrap();
        let x_coeff: f32;
        let y_coeff: f32;
        let z_coeff: f32;
        let mut xy_const: f32;
        let mut weight: f32;
        let zero_dist: f32;
        let zero_boost_lim: f32 = 0.05;
        let nh = self.m_num_high_sd as usize;
        let nbeads = self.m_num_beads as usize;
        let bead_bounds = 2 * nh;
        let high_bounds = nh;
        let temp = 2 * nh + nbeads;
        let mut medians = [0f32; 2];
        let mut madns = [0f32; 2];
        let scl = self.m_best_scale;
        let y_ind = 3 - ti;
        let flipped = ti != 2;
        let binnings = self.m_binning[scl];
        let spacings = self.m_box_spacing[scl];
        let num_boxes = self.m_num_boxes[scl];
        let mut z_slice: f32;
        let mut x_col: f32;
        let mut y_col: f32;
        let mut z_col: f32;
        let mut zfrac: f32;
        let mut dum1: f32 = 0.;
        let mut dum2: f32 = 0.;
        let mut proj_min: f32;

        let _ = imod_mat_rot(&mut mat, -yvec[1] as f64, Axis3::Y);
        let _ = imod_mat_rot(&mut mat, -yvec[0] as f64, Axis3::X);
        x_coeff = mat.data[2];
        y_coeff = mat.data[6];
        z_coeff = mat.data[10];

        // Compute the rotated boundaries, rotate around center to retain precision
        for ind in 0..nh {
            xy_const = x_coeff * (self.m_block_centers[2 * ind] - xcen)
                + y_coeff * (self.m_block_centers[2 * ind + 1] - ycen);
            self.m_bound_rot[ind] = xy_const + z_coeff * (self.m_boundaries[2 * ind] - zcen);
            self.m_bound_rot[high_bounds + ind] =
                xy_const + z_coeff * (self.m_boundaries[2 * ind + 1] - zcen);
        }

        // Rotate the beads if any
        for ind in 0..nbeads {
            self.m_bound_rot[bead_bounds + ind] = x_coeff
                * (self.m_block_centers[2 * (ind + nh)] - xcen)
                + y_coeff * (self.m_block_centers[2 * (ind + nh) + 1] - ycen)
                + z_coeff * (self.m_boundaries[ind + 2 * nh] - zcen);
        }

        // Compute median and MADN of low and high boundaries
        for lohi in 0..2usize {
            rs_fast_median_in_place(
                &mut self.m_bound_rot[lohi * nh..],
                self.m_num_high_sd,
                &mut medians[lohi],
            );
            let (head, tail) = self.m_bound_rot.split_at_mut(temp);
            rs_fast_madn(
                &head[lohi * nh..],
                self.m_num_high_sd,
                medians[lohi],
                tail,
                &mut madns[lohi],
            );
        }

        // Fetch data from slices at this rotation and average them
        if self.m_use_proj_for_spread != 0 {
            let sds_base = self.m_sds_off + self.m_stat_start_inds[scl] as usize;
            let all_sds_len = self.m_sds.len();
            proj_min = 1.0e20;
            for thick in 0..num_boxes[ti] {
                let mut num_in_slice = 0usize;
                z_slice = (self.m_start_coord[ti] as f64
                    + binnings[ti] as f64
                        * ((thick * spacings[ti]) as f64 + self.m_best_box_size[ti] as f64 / 2.))
                    as f32;
                for ilong in 0..num_boxes[y_ind] {
                    for ix_box in 0..num_boxes[B3D_X] {
                        x_col = (self.m_start_coord[B3D_X] as f64
                            + binnings[B3D_X] as f64
                                * ((ix_box * spacings[B3D_X]) as f64
                                    + self.m_best_box_size[B3D_X] as f64 / 2.))
                            as f32;
                        y_col = (self.m_start_coord[y_ind] as f64
                            + binnings[y_ind] as f64
                                * ((ilong * spacings[y_ind]) as f64
                                    + self.m_best_box_size[y_ind] as f64 / 2.))
                            as f32;
                        xy_const = x_coeff * (x_col - xcen) + y_coeff * (y_col - ycen);
                        z_col =
                            ((z_slice - zcen - xy_const) / z_coeff + zcen) / binnings[ti] as f32;
                        if z_col >= 0. && z_col <= num_boxes[ti] as f32 {
                            let iz = z_col as i32;
                            zfrac = z_col - iz as f32;
                            let mut iz_box = if flipped { ilong } else { iz };
                            let mut iy_box = if flipped { iz } else { ilong };
                            let mut box_ind = sds_base
                                + ((iz_box * num_boxes[B3D_Y] + iy_box) * num_boxes[B3D_X] + ix_box)
                                    as usize;
                            let this = if box_ind < all_sds_len {
                                self.m_sds[box_ind]
                            } else {
                                0.
                            };
                            self.m_proj_slice[num_in_slice] =
                                ((1. - zfrac as f64) * this as f64) as f32;

                            // See `analyze_high_sd` for the `iz + 1` read past the end.
                            iz_box = if flipped { ilong } else { iz + 1 };
                            iy_box = if flipped { iz + 1 } else { ilong };
                            box_ind = sds_base
                                + ((iz_box * num_boxes[B3D_Y] + iy_box) * num_boxes[B3D_X] + ix_box)
                                    as usize;
                            let next = if box_ind < all_sds_len {
                                self.m_sds[box_ind]
                            } else {
                                0.
                            };
                            self.m_proj_slice[num_in_slice] = (self.m_proj_slice[num_in_slice]
                                + zfrac * next
                                - self.m_edge_medians[scl])
                                / self.m_edge_madns[scl];
                            num_in_slice += 1;
                        }
                    }
                }
                self.m_proj_means[thick as usize] = 0.;
                if num_in_slice != 0 {
                    avg_sd(
                        &self.m_proj_slice,
                        num_in_slice as i32,
                        &mut self.m_proj_means[thick as usize],
                        &mut dum1,
                        &mut dum2,
                    );
                }
                let v = self.m_proj_means[thick as usize];
                proj_min = if proj_min < v { proj_min } else { v };
            }

            // Need centroid and moments of distribution
            let mut sum: f64 = 0.;
            let mut tot: f64 = 0.;
            let centroid: f64;
            let mut sum4: f64;
            for ind in 0..num_boxes[ti] {
                sum += (ind as f32 * (self.m_proj_means[ind as usize] - proj_min)) as f64;
                tot += (self.m_proj_means[ind as usize] - proj_min) as f64;
            }
            centroid = sum / tot;
            sum = 0.;
            sum4 = 0.;
            for ind in 0..num_boxes[ti] {
                let d = (self.m_proj_means[ind as usize] - proj_min) as f64;
                sum += (ind as f64 - centroid).powf(2.) * d;
                sum4 += (ind as f64 - centroid).powf(4.) * d;
            }
            sum = (sum / tot).sqrt();
            sum4 = (sum4 / tot).powf(0.25);
            if self.m_use_proj_for_spread > 1 {
                *spread = sum4 as f32;
            } else {
                *spread = sum as f32;
            }
        }

        // Now do beads.  If the percentile is 0, just compute min/max
        if self.m_num_beads != 0 {
            self.compute_bead_limits(self.m_bead_excl_pctl_spread);
        }

        // First time, initialize the normalization factors to the current values
        if self.m_simplex_iter == 0 {
            self.m_min_spread = 1.0e10;
            self.m_spread_norms[0] = madns[0];
            self.m_spread_norms[1] = madns[1];
            if self.m_debug_output != 0 {
                printf!(
                    "mSpreadNorms[0] = %g,  mSpreadNorms[1] = %g\n",
                    CArg::Dbl(self.m_spread_norms[0] as f64),
                    CArg::Dbl(self.m_spread_norms[1] as f64)
                );
            }
            if self.m_num_beads != 0 {
                self.m_spread_norms[2] = self.m_bead_high - self.m_bead_low;

                // Measure area covered by structure and beads to base a weighting on that
                self.m_struct_area = self.convex_area_covered(self.m_num_high_sd, 0);
                self.m_bead_area = self.convex_area_covered(self.m_num_beads, self.m_num_high_sd);
                if self.m_debug_output != 0 {
                    printf!(
                        "mSpreadNorms[2] = %g,  mStructArea = %g,  mBeadArea = %g\n",
                        CArg::Dbl(self.m_spread_norms[2] as f64),
                        CArg::Dbl(self.m_struct_area as f64),
                        CArg::Dbl(self.m_bead_area as f64)
                    );
                }
            }
        }

        // Compute spread as mean of normalized values, then take weighted average with
        // normalized thickness based on beads
        if self.m_use_proj_for_spread == 0 {
            *spread = ((madns[0] / self.m_spread_norms[0] + madns[1] / self.m_spread_norms[1])
                as f64
                / 2.) as f32;
        }
        if self.m_num_beads != 0 {
            let r = (self.m_bead_area / self.m_struct_area) as f64;
            weight = (self.m_bead_weight_fac as f64 * if 1. < r { 1. } else { r }) as f32;
            {
                let w = weight as f64;
                let minned = if 1. < w { 1. } else { w };
                weight = (if 0. > minned { 0. } else { minned }) as f32;
            }
            *spread = (*spread as f64 * (1. - weight as f64)
                + (weight * (self.m_bead_high - self.m_bead_low) / self.m_spread_norms[2]) as f64)
                as f32;
        }
        zero_dist = ((yvec[0] * yvec[0] + yvec[1] * yvec[1]) as f64).sqrt() as f32;
        if zero_dist < zero_boost_lim {
            *spread = (*spread as f64
                * (1. + ((zero_boost_lim - zero_dist) / zero_boost_lim) as f64))
                as f32;
        }
        self.m_simplex_iter += 1;
        if self.m_debug_output > 1 {
            printf!(
                "%d %f %f %f %s",
                CArg::Int(self.m_simplex_iter as i64),
                CArg::Dbl(yvec[0] as f64),
                CArg::Dbl(yvec[1] as f64),
                CArg::Dbl(*spread as f64),
                CArg::Str(if *spread < self.m_min_spread {
                    "*\n"
                } else {
                    "\n"
                })
            );
        }
        self.m_min_spread = if self.m_min_spread < *spread {
            self.m_min_spread
        } else {
            *spread
        };
    }

    /// `FindSect::convexAreaCovered` (`findsection.cpp:2874`): compute the area
    /// occupied by either beads or structure from center positions.
    fn convex_area_covered(&mut self, num_pts: i32, offset: i32) -> f32 {
        let mut num_vert: i32 = 0;
        let mut cb_xcen: f32 = 0.;
        let mut cb_ycen: f32 = 0.;
        let n = num_pts as usize;
        let off = offset as usize;
        for ind in 0..n {
            self.m_convex_xtmp[ind] = self.m_block_centers[2 * (off + ind)];
            self.m_convex_ytmp[ind] = self.m_block_centers[2 * (off + ind) + 1];
        }
        {
            let (xin, xout) = self.m_convex_xtmp.split_at_mut(n);
            let (yin, yout) = self.m_convex_ytmp.split_at_mut(n);
            convex_bound(
                xin,
                yin,
                0.01_f64 as f32,
                0.,
                &mut xout[..n],
                &mut yout[..n],
                &mut num_vert,
                &mut cb_xcen,
                &mut cb_ycen,
            );
        }
        let cont = self.m_convex_cont.as_mut().unwrap();
        cont.pts.truncate(0);
        for ind in 0..num_vert as usize {
            cont.pts.push(Ipoint {
                x: self.m_convex_xtmp[n + ind],
                y: self.m_convex_ytmp[n + ind],
                z: 0.,
            });
        }
        imod_contour_area(Some(cont))
    }

    /// `FindSect::computeBeadLimits` (`findsection.cpp:2896`): get mBeadLow and
    /// mBeadHigh limits with the given fraction excluded.
    fn compute_bead_limits(&mut self, exclude_pctl: f32) {
        let bead_bounds = 2 * self.m_num_high_sd as usize;
        let mut ind: i32;

        // If no exclusion, find min/max
        if exclude_pctl <= 0. {
            self.m_bead_low = 1.0e10;
            self.m_bead_high = -1.0e10;
            for ind in 0..self.m_num_beads as usize {
                let v = self.m_bound_rot[bead_bounds + ind];
                self.m_bead_low = if self.m_bead_low < v {
                    self.m_bead_low
                } else {
                    v
                };
                self.m_bead_high = if self.m_bead_high > v {
                    self.m_bead_high
                } else {
                    v
                };
            }
            self.m_bead_low = (self.m_bead_low as f64 - self.m_bead_diameter as f64 / 2.) as f32;
            self.m_bead_high = (self.m_bead_high as f64 + self.m_bead_diameter as f64 / 2.) as f32;
        } else {
            // Otherwise compute the percentiles
            let nb = self.m_num_beads;
            ind = b3dnint!(nb as f32 * exclude_pctl);
            ind = if nb < ind { nb } else { ind };
            ind = if 1 > ind { 1 } else { ind };
            self.m_bead_low = (percentile_float(ind, &mut self.m_bound_rot[bead_bounds..], nb)
                as f64
                - self.m_bead_diameter as f64 / 2.) as f32;
            ind = b3dnint!(nb as f64 * (1. - exclude_pctl as f64));
            ind = if nb < ind { nb } else { ind };
            ind = if 1 > ind { 1 } else { ind };
            self.m_bead_high = (percentile_float(ind, &mut self.m_bound_rot[bead_bounds..], nb)
                as f64
                + self.m_bead_diameter as f64 / 2.) as f32;
        }
    }
}
