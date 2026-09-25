//! Translation of `IMOD/imodutil/tiltxcorr.cpp` with its class header
//! `IMOD/imodutil/tiltxcorr.h` merged in.
//!
//! The C++ `TiltXCorr` class becomes the `TiltXCorr` struct, one method per
//! member function, and `TiltXCorr::main` keeps the source's locals as locals.
//! The source's global Fortran-model arrays (`fortmodel.h`, `fmod*`) are the
//! owned [`FortModel`] in `m_fm`.
//!
//! Shapes that differ from the source, each for a stated reason:
//!
//! * `main`'s `ubArray`/`ubBrray` are the members `m_ub_array`/`m_ub_brray`,
//!   and `correlateAndFindPeaks` takes `ub_main` instead of the two pointers:
//!   every call passes either those two arrays or `mArray`/`mBrray`
//!   (`tiltxcorr.cpp:2011-2013,2054-2056,2089-2095`), and with `ub_main` the
//!   unbinned correlation reads into, and so overwrites, `mArray`/`mBrray` as
//!   the aliased C pointers do.
//! * `mFullLinePtrs` is never assigned anywhere in the source, so the
//!   antialiased patch extraction (`tiltxcorr.cpp:3215`) hands
//!   `zoomWithFilter` an indeterminate pointer and the reference binary dies
//!   with SIGSEGV (`-size ... -binning 2 -antialias 3`).  The translation
//!   forms the line pointers the member is evidently meant to hold -- the
//!   `makeLinePointers` view of the cached full image -- at the point of use.
//!   Recorded in `BUGS.md`.
//! * `rotScanPeaks` gets one extra, zeroed element: `tiltxcorr.cpp:2063`
//!   tests `indBestRot <= numRotSteps` and reads `rotScanPeaks[indBestRot + 1]`,
//!   one past the allocation when the best angle is the last step.  Recorded
//!   in `BUGS.md`.
//! * `mPairMat` is allocated with `mMatCols` rather than `mPairCols` columns:
//!   `tiltxcorr.cpp:4058-4059` store into it with the wrong stride
//!   (`ind * mMatCols`), past the end of the source's allocation for the later
//!   points.  The extra space changes no value the source reads.  Recorded in
//!   `BUGS.md`.
//!
//! OpenMP: the one region in the unit, `maskOutsideBoundaries`
//! (`tiltxcorr.cpp:3051`), writes each pixel independently, so its result
//! cannot depend on the thread count; it runs serially here.

use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::amat_to_rotmagstr::rotmagstr_to_amat;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_get_error, b3d_i_max, b3d_i_min, c_format_bytes, imod_backup_file,
    imod_usage_header, number_in_list, wall_time,
};
use crate::imod::libcfshr::cubinterp::cubinterp;
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, cc_coefficient_two_pads, conjugate_product, get_peak_find_test_limits, nice_frame,
    parabolic_fit_position, set_peak_find_angle, set_peak_find_limits, xcorr_filter_part,
    xcorr_mean_zero, xcorr_peak_find_width, xcorr_set_ctf,
};
use crate::imod::libcfshr::findtransform::find_transform;
use crate::imod::libcfshr::gettiltangles::get_tilt_angles;
use crate::imod::libcfshr::histogram::{kernel_histogram, scan_histogram};
use crate::imod::libcfshr::insidecontour::inside_contour;
use crate::imod::libcfshr::islice::MrcData;
use crate::imod::libcfshr::linearxforms::{
    exit_from_xf_read_error, read_all_xforms, write_xform, xf_apply, xf_copy, xf_invert, xf_unit,
};
use crate::imod::libcfshr::minimize1d::minimize1d;
use crate::imod::libcfshr::multibinstat::make_standard_dev_map;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_three_floats, pip_get_two_floats, pip_get_two_integers,
    pip_number_of_entries, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::{ParseListError, parselist};
use crate::imod::libcfshr::percentile::percentile_float;
use crate::imod::libcfshr::piecefuncs::{check_piece_list, fill_list_of_piece_z, read_piece_list};
use crate::imod::libcfshr::readlinevalues::exit_from_value_read_error;
use crate::imod::libcfshr::reduce_by_binning::{extract_with_binning, repack_float_image};
use crate::imod::libcfshr::robuststat::{rs_fast_madn, rs_sort_indexed_floats, rs_sort_ints};
use crate::imod::libcfshr::samplemeansd::get_sample_of_array;
use crate::imod::libcfshr::simplestat::{
    array_min_max_mean, avg_sd, image_subarea_mean, ls_fit, ls_fit2, scale_array_for_mode,
};
use crate::imod::libcfshr::statfuncs::err_func;
use crate::imod::libcfshr::taperatfill::taper_at_fill;
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad};
use crate::imod::libcfshr::zoomdown::{ZoomLines, ZoomOut, select_zoom_filter, zoom_with_filter};
use crate::imod::libfft::{nice_fft_limit, todfft_c};
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_MODE_FLOAT, mrc_fill_label_string};
use crate::imod::libiimod::mrcslice::{
    SLICE_MODE_FLOAT, full_array_min_max_mean, mrc_write_image_to_file,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_open, iiu_read_section, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_size_samp_cell, iiu_print_header, iiu_ret_basic_head, iiu_ret_delta, iiu_ret_origin,
    iiu_ret_tilt, iiu_trans_header, iiu_write_header_str,
};
use crate::imod::libiimod::unit_reduced::{iiu_read_binned, iiu_read_reduced};
use crate::imod::libimod::fortmodel::{
    allocate_fort_model, fort_mod_obj_to_cont, fort_mod_open_error, read_fort_model,
    scale_fort_mod_to_image, scale_fort_model, write_fort_model,
};
use crate::imod::libimod::imodel_fwrap::{
    getimodflags, getimodobjsize, newimod, putimageref, putimodflag, putimodmaxes, putimodzscale,
    putlinewidth, putmodelname, putobjcolor, putsymsize, putsymtype,
};
use crate::imod::libwarp::warpfiles::{
    new_warp_file, read_warp_file, separate_linear_transform, set_linear_transform,
    set_warp_points, write_warp_file,
};
use crate::imod::libwarp::warputils::read_check_warp_file;
use std::io::Write as _;

use super::nogputxc::{MAX_PLAN_ARRAYS, TxcGPU};

/// `tiltxcorr.h:9`.
pub const LIMPEAKS: usize = 50;
/// `tiltxcorr.h:10`.
pub const LIMBOUND: usize = 1000;
/// `tiltxcorr.h:11`.
pub const MAX_FULL_CACHE: usize = 10;
/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `tiltxcorr.cpp:22`: `#define COSD(a) cosf((a) * RADIANS_PER_DEGREE)` --
/// the product is double, narrowed to the float `cosf` takes.
macro_rules! cosd {
    ($a:expr) => {
        ((($a) as f64 * RADIANS_PER_DEGREE) as f32).cos()
    };
}

/// `tiltxcorr.cpp:23`: `#define SIND(a) sinf((a) * RADIANS_PER_DEGREE)`.
macro_rules! sind {
    ($a:expr) => {
        ((($a) as f64 * RADIANS_PER_DEGREE) as f32).sin()
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

/// A float array seen as the raw bytes a C `void *` parameter takes.
macro_rules! f32_bytes {
    ($v:expr) => {{
        let v: &[f32] = &$v;
        // SAFETY: any `f32` storage is valid, aligned `u8` storage of four
        // times the length, and the view borrows `v` for its lifetime.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
    }};
}

/// A mutable float array seen as raw bytes; every bit pattern is a valid
/// `f32`, so writing bytes through it cannot make an invalid value.
macro_rules! f32_bytes_mut {
    ($v:expr) => {{
        let v: &mut [f32] = &mut $v;
        // SAFETY: as for `f32_bytes`, with the unique borrow of `v`.
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) }
    }};
}

/// Lends a float `Vec` to a routine that takes the typed `MrcData` union, as
/// `full_array_min_max_mean` lends a caller's array to a stack slice.
macro_rules! with_mrc_data {
    ($v:expr, |$d:ident| $body:expr) => {{
        let mut $d = MrcData::F(std::mem::take(&mut $v));
        let result = $body;
        if let MrcData::F(back) = $d {
            $v = back;
        }
        result
    }};
}

/// `rotmagstrToAmat(theta, smag, str, phi, &f[0], &f[2], &f[1], &f[3])`: the
/// source's argument order stores the matrix in the transform's layout, which
/// is the order `rotmagstr_to_amat` fills.
macro_rules! rotmagstr_to_fs {
    ($dst:expr, $theta:expr, $smag:expr, $str_:expr, $phi:expr) => {{
        let mut a = [0f32; 4];
        rotmagstr_to_amat($theta, $smag, $str_, $phi, &mut a);
        $dst[..4].copy_from_slice(&a);
    }};
}

/// `cppdefs.h:21-24`: `PRINT1`..`PRINT4` go through `cout`, whose default
/// float output is `%g`.
macro_rules! print_vals {
    ($($name:expr => $val:expr),+ $(,)?) => {{
        let parts: Vec<Vec<u8>> = vec![$(
            {
                let mut p = $name.as_bytes().to_vec();
                p.extend_from_slice(b" = ");
                p.extend_from_slice(&$val);
                p
            }
        ),+];
        let mut line = parts.join(&b",  "[..]);
        line.push(b'\n');
        let _ = ImodFile::Stdout.write_all(&line);
    }};
}

/// `cout << int`.
fn cout_i(v: i32) -> Vec<u8> {
    c_format_bytes("%d", &[CArg::Int(v as i64)])
}

/// `cout << float` / `cout << double`, default precision 6: `%g`.
fn cout_g(v: f64) -> Vec<u8> {
    c_format_bytes("%g", &[CArg::Dbl(v)])
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// C++ `main` (`tiltxcorr.cpp:25`).
pub fn tiltxcorr(arguments: &[String]) -> i32 {
    let mut txc = Box::new(TiltXCorr::new());
    let status = txc.main(arguments);
    let _ = ImodFile::Stdout.flush();
    status
}

/// `class TiltXCorr` (`tiltxcorr.h:16-262`).
pub struct TiltXCorr {
    m_txc_gpu: Option<TxcGPU>,
    m_use_gpu: i32,
    m_nice_limit: i32,
    m_array: Vec<f32>,
    m_arr_copy: Vec<f32>,
    m_main_array_size: i32,
    m_sd_arr: Vec<f32>,
    m_sum_arr: Vec<f32>,
    m_sqr_arr: Vec<f32>,
    m_sd_box_reduced: i32,
    m_sigmoid_power: f32,
    m_sigmoid_half_rise: f32,
    m_full_cache: Vec<Vec<f32>>,
    m_cache_size: i32,
    m_iz_loaded: [i32; MAX_FULL_CACHE],
    m_load_last_used: [i32; MAX_FULL_CACHE],
    m_loaded_fill_vals: [f32; MAX_FULL_CACHE],
    m_cache_use_count: i32,
    m_bin_width_ratio_crit: f32,
    m_breaking: bool,
    m_brray: Vec<f32>,
    m_central_peak_max_width: f32,
    m_rot_angle: f32,
    m_cos_phi: f32,
    m_cos_rot_angle: f32,
    m_crit_inside: f32,
    m_crit_non_blank: f32,
    m_crray: Vec<f32>,
    m_ctfp: Vec<f32>,
    m_ctf_ub: Vec<f32>,
    m_delta_ctf: f32,
    m_delta_ub_ctf: f32,
    m_dmax: f32,
    m_dmax2: f32,
    m_dmean3: f32,
    m_dmean_sum: f32,
    m_dmin: f32,
    m_dmin2: f32,
    m_if_read_xfs: i32,
    m_dx_preali: Vec<f32>,
    m_dy_preali: Vec<f32>,
    m_fs: [f32; 6],
    m_f_unit: [f32; 6],
    m_sd_xstart: i32,
    m_sd_xend: i32,
    m_sd_ystart: i32,
    m_sd_yend: i32,
    m_i_anti_filt_type: i32,
    m_if_abs_stretch: i32,
    m_if_cumulate: i32,
    m_if_ellipse: i32,
    m_if_exclude: i32,
    m_if_reuse_prev: i32,
    m_if_ub_peak_is_sharp: i32,
    m_in_file: String,
    m_iobj_flags: Vec<i32>,
    m_iter: i32,
    m_iunit_ref: i32,
    m_ix_box_cur: i32,
    m_ix_box_ref: i32,
    m_ix_cen_start: i32,
    m_iy_box_cur: i32,
    m_iy_box_ref: i32,
    m_iy_cen_start: i32,
    m_iz_cur: i32,
    m_iz_end: i32,
    m_iz_last: i32,
    m_iz_start: i32,
    m_len_temp: i64,
    m_limiting_shift: i32,
    m_limit_shift_x: i32,
    m_limit_shift_y: i32,
    m_limit_ub_shift_x: i32,
    m_limit_ub_shift_y: i32,
    m_list_skip: Vec<i32>,
    m_list_string: Vec<u8>,
    m_max_xcorr_peaks: i32,
    m_min_tilt: i32,
    m_mode: i32,
    m_nbinning: i32,
    m_images_binned: i32,
    m_unbinned_pixel: f32,
    m_n_fill_taper: i32,
    m_num_bound: i32,
    m_num_inside: i32,
    m_num_patches: i32,
    m_num_patches_all: i32,
    m_num_skip: i32,
    m_num_xcorr_peaks: i32,
    m_nx: i32,
    m_nx_pad: i32,
    m_nx_patch: i32,
    m_nx_taper: i32,
    m_nx_ub_pad: i32,
    m_nx_ub_taper: i32,
    m_nx_unali: i32,
    m_nx_use: i32,
    m_nx_use_bin: i32,
    m_ny: i32,
    m_ny_pad: i32,
    m_ny_patch: i32,
    m_ny_taper: i32,
    m_ny_ub_pad: i32,
    m_ny_ub_taper: i32,
    m_ny_unali: i32,
    m_ny_use: i32,
    m_ny_use_bin: i32,
    m_xcen: i32,
    m_ycen: i32,
    m_nz_out: i32,
    m_overlap_crit: f32,
    m_overlap_power: f32,
    m_patch_cen_x: Vec<f32>,
    m_patch_cen_xall: Vec<f32>,
    m_patch_cen_y: Vec<f32>,
    m_patch_cen_yall: Vec<f32>,
    m_patch_nx: Vec<i32>,
    m_patch_ny: Vec<i32>,
    m_single_test_patch: i32,
    m_single_nx_pad: i32,
    m_single_ny_pad: i32,
    m_min_dev_for_elim: f32,
    m_num_xdomains: i32,
    m_num_ydomains: i32,
    m_domain_xstarts: Vec<i32>,
    m_domain_ystarts: Vec<i32>,
    #[allow(dead_code)]
    m_domainstarts: Vec<i32>,
    m_domain_xsize: i32,
    m_domain_ysize: i32,
    m_pred_crit_prob: f32,
    m_abs_prob_crit: f32,
    m_patch_xinds: Vec<i32>,
    m_patch_yinds: Vec<i32>,
    m_patch_domains: Vec<i32>,
    m_peak2_to_peak3_crit: f32,
    m_peak_val: f32,
    m_rad_exclude: f32,
    m_reverse_order: i32,
    m_searched_for_mag: bool,
    m_sin_phi: f32,
    m_sin_rot_angle: f32,
    m_stretch: f32,
    m_sum_array: Vec<f32>,
    m_taper_cur: bool,
    m_taper_ref: bool,
    m_tilt: Vec<f32>,
    m_tilt_at_min: f32,
    m_tmp_array: Vec<f32>,
    m_tracking: bool,
    m_ub_width_ratio_crit: f32,
    m_unstretch_dx: f32,
    m_unstretch_dy: f32,
    m_use_max: f32,
    m_use_mean: f32,
    m_use_min: f32,
    m_verbose: i32,
    m_wallfft: f64,
    m_wall_interp: f64,
    m_wall_mask: f64,
    m_wall_peak: f64,
    m_wall_load: f64,
    m_wall_read: f64,
    m_wall_ccc: f64,
    m_wall_sd_stat: f64,
    m_xbound: Vec<f32>,
    m_xpeak: f32,
    m_xpeak_tmp: f32,
    m_xpeak_frac: f32,
    m_xtfs_bound: Vec<f32>,
    m_ybound: Vec<f32>,
    m_ypeak: f32,
    m_ypeak_tmp: f32,
    m_ypeak_frac: f32,
    m_ytfs_bound: Vec<f32>,
    m_ind_bound: [i32; LIMBOUND],
    m_num_in_bound: [i32; LIMBOUND],
    m_xtfs_bmin: [f32; LIMBOUND],
    m_xtfs_bmax: [f32; LIMBOUND],
    m_ytfs_bmin: [f32; LIMBOUND],
    m_ytfs_bmax: [f32; LIMBOUND],
    m_xpeak_list: [f32; LIMPEAKS],
    m_ypeak_list: [f32; LIMPEAKS],
    m_peak_list: [f32; LIMPEAKS],
    m_ind_peak_sort: [i32; LIMPEAKS],
    m_mat_cols: i32,
    m_xmat: Vec<f32>,
    m_pair_mat: Vec<f32>,
    m_pair_cols: i32,
    m_pred_min_fit: i32,
    m_pred_max_fit: i32,
    m_pred_min_tilt_range: i32,
    m_pf_trans_min_pts: i32,
    m_pf_rot_trans_min_pts: i32,
    m_pf_lin_xf_min_pts: i32,
    m_model_xvecs: Vec<Vec<f32>>,
    m_model_yvecs: Vec<Vec<f32>>,
    m_peak_vecs: Vec<Vec<f32>>,
    m_pf_better_dev_crit: f32,
    m_pf_min_closer_peak_ratio: f32,
    m_good_xstart: [i32; 2],
    m_good_xend: [i32; 2],
    m_good_ystart: [i32; 2],
    m_good_yend: [i32; 2],
    /// The `fortmodel.h` globals (`fmod*`).
    m_fm: FortModel,
    /// `main`'s `ubArray`/`ubBrray` (`tiltxcorr.cpp:121`); see the module
    /// comment.
    m_ub_array: Vec<f32>,
    m_ub_brray: Vec<f32>,
    /// `adjustCoord`'s function-level statics (`tiltxcorr.cpp:2975`).
    s_tilt_to_last: f32,
    s_tilt_from_last: f32,
    s_cos_to_last: f32,
    s_cos_from_last: f32,
}

impl TiltXCorr {
    /// `TiltXCorr::TiltXCorr` (`tiltxcorr.cpp:34`).  Members the constructor
    /// does not set are indeterminate in the source (the instance is a local
    /// of `main`); they start at zero or empty here.
    pub fn new() -> Self {
        let mut t = TiltXCorr {
            m_txc_gpu: None,
            m_use_gpu: 0,
            m_nice_limit: 0,
            m_array: Vec::new(),
            m_arr_copy: Vec::new(),
            m_main_array_size: 0,
            m_sd_arr: Vec::new(),
            m_sum_arr: Vec::new(),
            m_sqr_arr: Vec::new(),
            m_sd_box_reduced: 0,
            m_sigmoid_power: 0.,
            m_sigmoid_half_rise: 0.,
            m_full_cache: vec![Vec::new(); MAX_FULL_CACHE],
            m_cache_size: 0,
            m_iz_loaded: [0; MAX_FULL_CACHE],
            m_load_last_used: [0; MAX_FULL_CACHE],
            m_loaded_fill_vals: [0.; MAX_FULL_CACHE],
            m_cache_use_count: 0,
            m_bin_width_ratio_crit: 0.,
            m_breaking: false,
            m_brray: Vec::new(),
            m_central_peak_max_width: 0.,
            m_rot_angle: 0.,
            m_cos_phi: 0.,
            m_cos_rot_angle: 0.,
            m_crit_inside: 0.,
            m_crit_non_blank: 0.,
            m_crray: Vec::new(),
            m_ctfp: vec![0.; 8193],
            m_ctf_ub: vec![0.; 8193],
            m_delta_ctf: 0.,
            m_delta_ub_ctf: 0.,
            m_dmax: 0.,
            m_dmax2: 0.,
            m_dmean3: 0.,
            m_dmean_sum: 0.,
            m_dmin: 0.,
            m_dmin2: 0.,
            m_if_read_xfs: 0,
            m_dx_preali: Vec::new(),
            m_dy_preali: Vec::new(),
            m_fs: [0.; 6],
            m_f_unit: [0.; 6],
            m_sd_xstart: 0,
            m_sd_xend: 0,
            m_sd_ystart: 0,
            m_sd_yend: 0,
            m_i_anti_filt_type: 0,
            m_if_abs_stretch: 0,
            m_if_cumulate: 0,
            m_if_ellipse: 0,
            m_if_exclude: 0,
            m_if_reuse_prev: 0,
            m_if_ub_peak_is_sharp: 0,
            m_in_file: String::new(),
            m_iobj_flags: Vec::new(),
            m_iter: 0,
            m_iunit_ref: 0,
            m_ix_box_cur: 0,
            m_ix_box_ref: 0,
            m_ix_cen_start: 0,
            m_iy_box_cur: 0,
            m_iy_box_ref: 0,
            m_iy_cen_start: 0,
            m_iz_cur: 0,
            m_iz_end: 0,
            m_iz_last: 0,
            m_iz_start: 0,
            m_len_temp: 0,
            m_limiting_shift: 0,
            m_limit_shift_x: 0,
            m_limit_shift_y: 0,
            m_limit_ub_shift_x: 0,
            m_limit_ub_shift_y: 0,
            m_list_skip: Vec::new(),
            m_list_string: Vec::new(),
            m_max_xcorr_peaks: 0,
            m_min_tilt: 0,
            m_mode: 0,
            m_nbinning: 0,
            m_images_binned: 0,
            m_unbinned_pixel: 0.,
            m_n_fill_taper: 0,
            m_num_bound: 0,
            m_num_inside: 0,
            m_num_patches: 0,
            m_num_patches_all: 0,
            m_num_skip: 0,
            m_num_xcorr_peaks: 0,
            m_nx: 0,
            m_nx_pad: 0,
            m_nx_patch: 0,
            m_nx_taper: 0,
            m_nx_ub_pad: 0,
            m_nx_ub_taper: 0,
            m_nx_unali: 0,
            m_nx_use: 0,
            m_nx_use_bin: 0,
            m_ny: 0,
            m_ny_pad: 0,
            m_ny_patch: 0,
            m_ny_taper: 0,
            m_ny_ub_pad: 0,
            m_ny_ub_taper: 0,
            m_ny_unali: 0,
            m_ny_use: 0,
            m_ny_use_bin: 0,
            m_xcen: 0,
            m_ycen: 0,
            m_nz_out: 0,
            m_overlap_crit: 0.,
            m_overlap_power: 0.,
            m_patch_cen_x: Vec::new(),
            m_patch_cen_xall: Vec::new(),
            m_patch_cen_y: Vec::new(),
            m_patch_cen_yall: Vec::new(),
            m_patch_nx: Vec::new(),
            m_patch_ny: Vec::new(),
            m_single_test_patch: 0,
            m_single_nx_pad: 0,
            m_single_ny_pad: 0,
            m_min_dev_for_elim: 0.,
            m_num_xdomains: 0,
            m_num_ydomains: 0,
            m_domain_xstarts: Vec::new(),
            m_domain_ystarts: Vec::new(),
            m_domainstarts: Vec::new(),
            m_domain_xsize: 0,
            m_domain_ysize: 0,
            m_pred_crit_prob: 0.,
            m_abs_prob_crit: 0.,
            m_patch_xinds: Vec::new(),
            m_patch_yinds: Vec::new(),
            m_patch_domains: Vec::new(),
            m_peak2_to_peak3_crit: 0.,
            m_peak_val: 0.,
            m_rad_exclude: 0.,
            m_reverse_order: 0,
            m_searched_for_mag: false,
            m_sin_phi: 0.,
            m_sin_rot_angle: 0.,
            m_stretch: 0.,
            m_sum_array: Vec::new(),
            m_taper_cur: false,
            m_taper_ref: false,
            m_tilt: Vec::new(),
            m_tilt_at_min: 0.,
            m_tmp_array: Vec::new(),
            m_tracking: false,
            m_ub_width_ratio_crit: 0.,
            m_unstretch_dx: 0.,
            m_unstretch_dy: 0.,
            m_use_max: 0.,
            m_use_mean: 0.,
            m_use_min: 0.,
            m_verbose: 0,
            m_wallfft: 0.,
            m_wall_interp: 0.,
            m_wall_mask: 0.,
            m_wall_peak: 0.,
            m_wall_load: 0.,
            m_wall_read: 0.,
            m_wall_ccc: 0.,
            m_wall_sd_stat: 0.,
            m_xbound: Vec::new(),
            m_xpeak: 0.,
            m_xpeak_tmp: 0.,
            m_xpeak_frac: 0.,
            m_xtfs_bound: Vec::new(),
            m_ybound: Vec::new(),
            m_ypeak: 0.,
            m_ypeak_tmp: 0.,
            m_ypeak_frac: 0.,
            m_ytfs_bound: Vec::new(),
            m_ind_bound: [0; LIMBOUND],
            m_num_in_bound: [0; LIMBOUND],
            m_xtfs_bmin: [0.; LIMBOUND],
            m_xtfs_bmax: [0.; LIMBOUND],
            m_ytfs_bmin: [0.; LIMBOUND],
            m_ytfs_bmax: [0.; LIMBOUND],
            m_xpeak_list: [0.; LIMPEAKS],
            m_ypeak_list: [0.; LIMPEAKS],
            m_peak_list: [0.; LIMPEAKS],
            m_ind_peak_sort: [0; LIMPEAKS],
            m_mat_cols: 0,
            m_xmat: Vec::new(),
            m_pair_mat: Vec::new(),
            m_pair_cols: 0,
            m_pred_min_fit: 0,
            m_pred_max_fit: 0,
            m_pred_min_tilt_range: 0,
            m_pf_trans_min_pts: 0,
            m_pf_rot_trans_min_pts: 0,
            m_pf_lin_xf_min_pts: 0,
            m_model_xvecs: Vec::new(),
            m_model_yvecs: Vec::new(),
            m_peak_vecs: Vec::new(),
            m_pf_better_dev_crit: 0.,
            m_pf_min_closer_peak_ratio: 0.,
            m_good_xstart: [0; 2],
            m_good_xend: [0; 2],
            m_good_ystart: [0; 2],
            m_good_yend: [0; 2],
            m_fm: FortModel::default(),
            m_ub_array: Vec::new(),
            m_ub_brray: Vec::new(),
            s_tilt_to_last: 0.,
            s_tilt_from_last: 0.,
            s_cos_to_last: 1.,
            s_cos_from_last: 1.,
        };
        t.m_if_exclude = 0;
        t.m_if_cumulate = 0;
        t.m_if_abs_stretch = 0;
        t.m_nbinning = 0;
        t.m_tracking = false;
        t.m_verbose = 0;
        t.m_crit_inside = 0.75;
        t.m_limit_shift_x = 1000000;
        t.m_limit_shift_y = 1000000;
        t.m_limiting_shift = 0;
        t.m_wall_mask = 0.;
        t.m_wall_interp = 0.;
        t.m_wallfft = 0.;
        t.m_wall_peak = 0.;
        t.m_wall_load = 0.;
        t.m_wall_read = 0.;
        t.m_wall_ccc = 0.;
        t.m_wall_sd_stat = 0.;
        t.m_num_skip = 0;
        t.m_breaking = false;
        t.m_crit_non_blank = 0.7;
        t.m_len_temp = 1000000;
        t.m_i_anti_filt_type = 0;
        t.m_reverse_order = 0;
        t.m_max_xcorr_peaks = 10;
        t.m_bin_width_ratio_crit = 1.05;
        t.m_peak2_to_peak3_crit = 3.;
        t.m_central_peak_max_width = 3.0;
        t.m_ub_width_ratio_crit = 1.6;
        t.m_rad_exclude = 0.3;
        t.m_overlap_crit = 0.125;
        t.m_overlap_power = 6.;
        t.m_if_ellipse = 1;
        t.m_iunit_ref = 1;
        t.m_main_array_size = 0;
        t.m_mat_cols = 14;
        t.m_pair_cols = 6;
        t.m_pred_min_fit = 3;
        t.m_pred_max_fit = 7;
        // `mPredMinTiltRange = 8.;` into an `int`.
        t.m_pred_min_tilt_range = 8;
        t.m_pf_trans_min_pts = 4;
        t.m_pf_rot_trans_min_pts = 5;
        t.m_pf_lin_xf_min_pts = 7;
        t.m_pf_better_dev_crit = 0.67;
        t.m_pf_min_closer_peak_ratio = 0.4;
        t.m_min_dev_for_elim = 2.;
        t.m_domain_xsize = 6;
        t.m_domain_ysize = 6;
        t.m_cache_use_count = 0;
        t.m_sigmoid_power = 1.;
        t.m_sigmoid_half_rise = 0.5;
        t.m_images_binned = 1;
        t.m_unbinned_pixel = 0.;
        t.m_sd_box_reduced = 6;
        t.m_pred_crit_prob = 0.01;
        t.m_abs_prob_crit = 0.002;
        t.m_use_gpu = -1;
        t.m_single_test_patch = -1;
        t.m_single_nx_pad = 0;
        for ind in 0..MAX_FULL_CACHE {
            t.m_iz_loaded[ind] = -1;
            t.m_loaded_fill_vals[ind] = 0.;
        }
        for ind in 0..2 {
            t.m_good_xstart[ind] = 0;
            t.m_good_xend[ind] = 0;
            t.m_good_ystart[ind] = 0;
            t.m_good_yend[ind] = 0;
        }
        t
    }

    /// `TiltXCorr::main` (`tiltxcorr.cpp:114`).  Returns the source's `return
    /// 0`; every failure exits through `exitError`.
    pub fn main(&mut self, argv: &[String]) -> i32 {
        let mut nz: i32;
        let mut nxyz = [0i32; 3];
        let mut mxyz = [0i32; 3];
        let mut nxyz_ref = [0i32; 3];
        let mut delta: [f32; 3];
        let origin: [f32; 3];
        let cur_tilt: [f32; 3];

        let mut pl_file: Vec<u8> = Vec::new();
        let mut xf_file_out: Vec<u8> = Vec::new();
        let mut ref_file: Vec<u8> = Vec::new();
        let mut im_file_out: Option<Vec<u8>> = None;
        let mut fs_inv = [0f32; 6];
        let mut prexf_inv = [0f32; 6];

        let mut f: Vec<f32>;
        let mut f_preali: Vec<f32> = Vec::new();
        let mut ix_pc_list: Vec<i32>;
        let mut iy_pc_list: Vec<i32>;
        let mut iz_pc_list: Vec<i32>;
        let mut listz: Vec<i32>;
        let mut list_mag_views: Vec<i32>;
        let mut x_control: Vec<f32> = Vec::new();
        let mut y_control: Vec<f32> = Vec::new();
        let mut x_vector: Vec<f32> = Vec::new();
        let mut y_vector: Vec<f32> = Vec::new();
        let mut xmodel: Vec<f32> = Vec::new();
        let mut ymodel: Vec<f32> = Vec::new();
        let mut rot_scan_peaks: Vec<f32> = Vec::new();
        let mut if_bound_on_view: Vec<i32> = Vec::new();
        let mut iobj_bound = [0i32; LIMBOUND];
        let mut dmean2: f32 = 0.;
        // `tiltxcorr.cpp:138` declares a local `mDmean3` that shadows the
        // member inside `main`.
        let mut dmean3_local: f32 = 0.;
        let cos_str_max_tilt: f32;
        let mut num_pc_list: i32 = 0;
        let mut num_views: usize = 0;
        let mut min_xpiece = 0;
        let mut num_xpieces = 0;
        let mut nx_overlap = 0;
        let mut min_ypiece = 0;
        let mut num_ypieces = 0;
        let mut ny_overlap = 0;
        let mut num_best: i32;
        let mut if_im_out: i32;
        let mut ind_best: i32 = 0;
        let mut nx_trim: i32;
        let mut ny_trim: i32;
        let mut nx_border: i32 = 0;
        let mut ny_border: i32 = 0;
        let mut idir: i32;
        let mut iz_tmp: i32;
        let dmean: f32;
        let mut radius1: f32;
        let mut radius2: f32;
        let mut sigma1: f32;
        let mut sigma2: f32;
        let mut cos_view: f32;
        let mut iv: i32 = 0;
        let mut ierr: i32;
        let mut iv_start: i32;
        let mut iv_end: i32;
        let mut loop_dir: i32;
        let num_loops: i32;
        let mut iv_ref: i32;
        let mut if_leave_axis: i32;
        let mut iv_ref_base: i32;
        let max_bin_size: i32;
        let mut if_no_stretch: i32;
        let max_track_gap: i32;
        let mut ix_start: i32;
        let mut ix_end: i32;
        let mut iy_start: i32;
        let mut iy_end: i32;
        let mut iv_cur: i32;
        let max_user_binning: i32;
        let mut iz: i32;
        let mut iv_skip: i32 = 0;
        let mut len_contour: i32;
        let mut min_cont_overlap: i32;
        let mut ix_cen_end: i32 = 0;
        let mut iy_cen_end: i32 = 0;
        let mut num_iter: i32;
        let mut iv_bound: i32 = 0;
        let mut lap_total: i32;
        let mut lap_remainder: i32;
        let mut iv_base: i32;
        let mut lap_base: i32;
        let mut num_cont: i32;
        let mut x_box_ofs: f32;
        let y_box_offset: f32;
        let mut x0: f32 = 0.;
        let mut y0: f32 = 0.;
        let mut xshift: f32 = 0.;
        let mut yshift: f32 = 0.;
        let mut pad_frac: f32;
        let taper_frac: f32;
        let mut cum_xshift: f32 = 0.;
        let mut cum_yshift: f32 = 0.;
        let mut cum_xrot: f32;
        let mut x_adjust: f32;
        let mut angle_offset: f32;
        let mut cum_xcenter: f32;
        let mut cum_ycenter: f32;
        let mut x_mod_offset: f32;
        let mut y_mod_offset: f32;
        let mut xrot: f32;
        let mut yrot: f32;
        let mut xpeak_cum: f32 = 0.;
        let mut ypeak_cum: f32 = 0.;
        let mut y_overlap: f32;
        let peak_frac_tol: f32;
        let mut x_from_cen: f32;
        let mut y_from_cen: f32;
        let mut cenx: f32;
        let mut ceny: f32;
        let mut base_tilt: f32;
        let mut x_overlap: f32;
        let mut yval: f32;
        let mut num_xpatch: i32 = 0;
        let mut num_ypatch: i32 = 0;
        let mut ind: i32;
        let mut ix_box_start: i32 = 0;
        let mut iy_box_start: i32 = 0;
        let mut max_nx_patch: i32 = 0;
        let mut max_ny_patch: i32 = 0;
        let mut ix_box_for_adj: i32 = 0;
        let mut iy_box_for_adj: i32 = 0;
        let mut ipatch: i32;
        let mut num_points: i32 = 0;
        let mut iobj: i32;
        let if_patch_num: i32;
        let mut ipnt: i32;
        let mut iobj_seed: i32;
        let mut imod_obj: i32 = 0;
        let mut imod_cont: i32 = 0;
        let mut ix: i32;
        let mut iy: i32 = 0;
        let min_patch_for_pred: i32;
        let mut last_not_skipped: i32;
        let mut len_conts: i32;
        let mut num_control: i32;
        let num_bound_all: i32;
        let idim2: i32;
        let mut iv_first_not_skipped: i32;
        let mut num_all_views: i32;
        let mut iv_pair_offset: i32;
        let mut if_find_warp: i32;
        let limited_bin_size: i32;
        let mut mag_view: i32 = 0;
        let max_auto_binning: i32;
        let mut num_peak_to_proc: i32;
        let mut cos_ratio: f32 = 0.;
        let mut peak_last: f32 = 0.;
        let mut xpeak_last: f32 = 0.;
        let mut y_peak_last: f32 = 0.;
        let mut x_frac_last: f32 = 0.;
        let mut y_frac_last: f32 = 0.;
        let mut x_frac_to_add: f32 = 0.;
        let mut bound_xmin: f32 = 0.;
        let mut bound_xmax: f32 = 0.;
        let mut bound_ymin: f32 = 0.;
        let mut bound_ymax: f32 = 0.;
        let mut frac_xover: f32;
        let mut frac_yover: f32;
        let mut y_frac_to_add: f32 = 0.;
        let frac_over_max: f32;
        let fill_taper_frac: f32;
        let mut cos_max: f32;
        let mut max_frac_to_elim: f32 = 0.33;
        let mut domain_xassigns: Vec<i32> = Vec::new();
        let mut domain_yassigns: Vec<i32> = Vec::new();
        let nice_gpu_limit = 5;
        let mut num_cuts: i32;
        let mut num_mag_views: i32;
        let mut if_scan_rotation: i32;
        let mut num_rot_steps: i32;
        let mut ind_best_rot: i32;
        let mut iz_in_ref_file: i32 = 0;
        let mut nx_sec_peak_box: i32;
        let mut ny_sec_peak_box: i32;
        let mut found_mag: f32 = 0.;
        let mut step_mag: f32;
        let mut search_min: f32;
        let mut search_max: f32;
        let mut brackets = [0f32; 14];
        let mut scan_rot_max: f32 = 0.;
        let mut scan_rot_interval: f32 = 0.;
        let mut angle: f32 = 0.;
        let mut best_angle: f32 = 0.;
        let mut ref_tilt: f32 = 0.;
        let mut sec_xpeak: f32;
        let mut sec_ypeak: f32;
        let mut unstretch_sec_dx: f32 = 0.;
        let mut unstretch_sec_dy: f32 = 0.;
        let mut sec_xshift: f32;
        let mut sec_yshift: f32;

        let mut eval_ccc: i32;
        let mut search_mag: i32;
        let mut add_to_warps: i32;
        let mut ref_view_out: bool;
        let mut raw_aligned_pair: bool;
        let mut cur_view_out: bool = false;
        let reuse_prev_arrays: bool;
        let use_ref_file: bool;
        let mut sec_peak_in_box: bool;
        let mut input_pass0 = false;
        let mut input_pass90 = false;
        let mut input_pass_min90 = false;
        let mut inv_pass0 = false;
        let mut inv_pass90 = false;
        let mut inv_pass_min90 = false;

        const MAX_BINNINGS: usize = 20;
        let mut num_sd_bins: i32 = 0;
        let mut nx_sd: i32 = 0;
        let mut ny_sd: i32 = 0;
        let mut sd_bin_list: Option<Vec<i32>> = None;
        let mut min_sd = [0f32; MAX_BINNINGS];
        let mut pct1_sd = [0f32; MAX_BINNINGS];
        let mut mean_avg: f32 = 0.;
        let mut mean_sd: f32 = 0.;
        let mut temp: f32 = 0.;
        let mut cumul_binning: f32;
        let mut sd_means: Vec<Vec<f32>> = vec![Vec::new(); MAX_BINNINGS];
        let mut expand_vec: Vec<f32> = Vec::new();
        let cen_xsave: Vec<f32>;
        let cen_ysave: Vec<f32>;
        let mut sd_sum_vec: Vec<f32> = Vec::new();
        let mut sd_sum_tmp: Vec<f32>;
        let mut sd_num_pix: Vec<Vec<i32>> = vec![Vec::new(); MAX_BINNINGS];
        let mut full_ind: usize = 0;
        let mut sd_xoff: i32 = 0;
        let mut sd_yoff: i32 = 0;
        let mut ibin: usize = 0;
        let mut nx_patch_orig: i32 = 0;
        let mut ny_patch_orig: i32 = 0;
        let mut sd_bin: i32 = 0;
        let mut px_start: i32;
        let mut py_start: i32;
        let mut px_end: i32;
        let mut py_end: i32;
        let mut min_expand_border: i32 = 0;
        let mut final_bin: i32;
        let mut nx_temp: i32;
        let mut ny_temp: i32;
        let mut num_drop: i32;
        let mut expand_x: f32;
        let mut expand_y: f32;
        let mut sum_mode: f32 = 0.;
        let elim_crit: f32;
        let sum_thresh: f32;
        let mut too_big: bool;
        let mut blocked_x: bool;
        let mut blocked_y: bool;
        let mut min_sd_target_pixel: f32 = 0.2;
        let mut max_sd_target_pixel: f32 = 5.;
        let mut max_cv: f32;
        let mut min_cv: f32;
        let mut summed_sd_crit: f32 = 0.;
        let mut sd_sum: f32;
        let mut expand_fac: f32;
        let mut vec_min: f32;
        let mut vec_max: f32;
        let mut expand_step: f32 = 0.1;
        let mut frac_variation: f32 = 0.85;
        let mut max_patch_expand: f32 = 0.;
        let mut max_tilted_expand: f32 = 0.;
        let mut expand_drop_frac: f32 = 0.7;
        let mut min_mode_frac: f32 = 0.;
        let mut expand_to_pctl: f32 = 50.;
        let sd_sum_lower_ratio: f32 = 0.5;
        let gpu_mem_frac: f32 = 0.7;
        let mut gpu_memory: f32 = 0.;
        let avail_gpu_mem: f32;
        let mut base_mem: f32;
        let mut if_gpu_by_env = 0;
        let mut act_gpu_fail_option = 0;
        let mut act_gpu_fail_environ = 0;
        let max_gpu_plans: i32;

        let mut num_opt_arg = 0;
        let mut num_non_opt_arg = 0;
        //
        // fallbacks from ../manpages/autodoc2man 2 1  tiltxcorr
        //
        let num_options = 84;
        let options: [&[u8]; 84] = [
            b"input:InputFile:FN:",
            b"piece:PieceListFile:FN:",
            b"reference:ReferenceFile:FN:",
            b"rview:ReferenceView:I:",
            b"output:OutputFile:FN:",
            b"rotation:RotationAngle:F:",
            b"first:FirstTiltAngle:F:",
            b"increment:TiltIncrement:F:",
            b"tiltfile:TiltFile:FN:",
            b"angles:TiltAngles:FAM:",
            b"offset:AngleOffset:F:",
            b"reverse:ReverseOrder:B:",
            b"prexf:PrealignmentTransformFile:FN:",
            b"imagebinned:ImagesAreBinned:I:",
            b"pixel:PixelSize:F:",
            b"unali:UnalignedSizeXandY:IP:",
            b"binning:BinningToApply:I:",
            b"antialias:AntialiasFilter:I:",
            b"radius1:FilterRadius1:F:",
            b"radius2:FilterRadius2:F:",
            b"nmrad2:Radius2InvNanometers:F:",
            b"sigma1:FilterSigma1:F:",
            b"nmsig1:Sigma1InvNanometers:F:",
            b"sigma2:FilterSigma2:F:",
            b"nmsig2:Sigma2InvNanometers:F:",
            b"border:BordersInXandY:IP:",
            b"ubborder:UnbinnedBordersXY:IP:",
            b"xminmax:XMinAndMax:IP:",
            b"yminmax:YMinAndMax:IP:",
            b"ubxmm:UnbinnedXMinAndMax:IP:",
            b"ubymm:UnbinnedYMinAndMax:IP:",
            b"pad:PadsInXandY:IP:",
            b"taper:TapersInXandY:IP:",
            b"ccc:CorrelationCoefficient:B:",
            b"iterate:IterateCorrelations:I:",
            b"shift:ShiftLimitsXandY:IP:",
            b"axial:AxialShiftLimits:IP:",
            b"rect:RectangularLimits:B:",
            b"boundary:BoundaryModel:FN:",
            b"objbound:BoundaryObject:I:",
            b"skip:SkipViews:LI:",
            b"break:BreakAtViews:LI:",
            b"views:StartingEndingViews:IP:",
            b"exclude:ExcludeCentralPeak:B:",
            b"central:CentralPeakExclusionCriteria:FT:",
            b"leaveaxis:LeaveTiltAxisShifted:B:",
            b"cumulative:CumulativeCorrelation:B:",
            b"absstretch:AbsoluteCosineStretch:B:",
            b"nostretch:NoCosineStretch:B:",
            b"search:SearchMagChanges:B:",
            b"changes:ViewsWithMagChanges:LI:",
            b"mag:MagnificationLimits:FP:",
            b"scan:ScanRotationMaxAndStep:FP:",
            b"second:SecondPeakBoxSize:IP:",
            b"size:SizeOfPatchesXandY:IP:",
            b"ubsize:UnbinnedPatchSizeXY:IP:",
            b"varying:VaryingToPatchSizeXY:IP:",
            b"number:NumberOfPatchesXandY:IP:",
            b"overlap:OverlapOfPatchesXandY:IP:",
            b"seed:SeedModel:FN:",
            b"objseed:SeedObject:I:",
            b"length:LengthAndOverlap:IP:",
            b"struct:MinStructureModeFrac:F:",
            b"elim:MaxFracOfPatchesToElim:F:",
            b"expand:MaxPatchExpansions:FP:",
            b"percentile:PercentileToMatch:F:",
            b"drop:TiltedExpandDropFrac:F:",
            b"target:TargetPixelMinAndMax:FP:",
            b"variation:FractionOfMaxVariation:F:",
            b"scurve:SigmoidPowerAndHalfRise:FP:",
            b"box:SdReducedBoxSize:I:",
            b"sdbin:SdBinList:LI:",
            b"domains:LocalDomainSizesXandY:IP:",
            b"prob:CriterionProbabilities:FP:",
            b"warp:FindWarpTransforms:I:",
            b"pair:RawAndAlignedPair:IP:",
            b"append:AppendToWarpFile:B:",
            b"gpu:UseGPU:I:",
            b"action:ActionIfGPUFails:IP:",
            b"test:TestOutput:FN:",
            b"single:SingleTestPatch:I:",
            b"verbose:VerboseOutput:I:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
        ];
        //
        // set defaults here where not dependent on image size
        //
        if_im_out = 0;
        nx_trim = 0;
        ny_trim = 0;
        sigma1 = 0.;
        sigma2 = 0.;
        radius1 = 0.;
        radius2 = 0.;
        self.m_rot_angle = 0.;
        if_no_stretch = 0;
        if_leave_axis = 0;
        angle_offset = 0.;
        max_bin_size = 1250;
        max_user_binning = 32;
        max_auto_binning = 16;
        cos_str_max_tilt = 82.;
        num_iter = 1;
        peak_frac_tol = 0.015;
        len_contour = 0;
        min_cont_overlap = 0;
        frac_xover = 0.33;
        frac_yover = 0.33;
        frac_over_max = 0.8;
        self.m_images_binned = 1;
        self.m_if_read_xfs = 0;
        fill_taper_frac = 0.1;
        max_track_gap = 5;
        if_find_warp = 0;
        raw_aligned_pair = false;
        add_to_warps = 0;
        iv_pair_offset = 0;
        eval_ccc = 0;
        search_mag = 0;
        search_min = 0.9;
        search_max = 1.1;
        num_rot_steps = 0;
        limited_bin_size = 4300 * 4300;
        ny_sec_peak_box = 0;
        nx_sec_peak_box = 0;
        min_patch_for_pred = 8;
        self.m_fm.fm_mod_size_type = 2;
        self.m_cache_size = 7;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            b"tiltxcorr",
            3,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
            Some(imod_usage_header_for_pip),
        );
        allocate_fort_model(&mut self.m_fm);

        // Get and open the image file
        let mut in_file: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"InputFile", 0, &mut in_file) != 0 {
            exit_error(b"No input file specified");
        }
        self.m_in_file = String::from_utf8_lossy(&in_file).into_owned();
        unsafe { iiu_open(1, &self.m_in_file, "RO") };
        iiu_print_header(1, Some("Input volume"));
        printf!("\n");
        unsafe {
            iiu_ret_basic_head(
                1,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut self.m_mode,
                &mut self.m_dmin2,
                &mut self.m_dmax2,
                &mut dmean2,
            );
        }

        // Get sizes and center and Z (really view) limits
        self.m_nx = nxyz[0];
        self.m_ny = nxyz[1];
        nz = nxyz[2];
        self.m_iz_start = 1;
        self.m_iz_end = nz;
        // `mXcen = mNx / 2.;` into an `int`.
        self.m_xcen = (self.m_nx as f64 / 2.) as i32;
        self.m_ycen = (self.m_ny as f64 / 2.) as i32;

        // Allocate nz-dependent items
        let nzu = nz.max(0) as usize;
        f = vec![0.; 6 * nzu];
        self.m_tilt = vec![0.; nzu];
        ix_pc_list = vec![0; nzu];
        iy_pc_list = vec![0; nzu];
        iz_pc_list = vec![0; nzu];
        listz = vec![0; nzu];
        self.m_list_skip = vec![0; nzu];
        list_mag_views = vec![0; nzu];

        // Set default for views to find mags at
        for i in 0..nz {
            list_mag_views[i as usize] = i;
        }
        num_mag_views = nz;
        //
        let pl_found = pip_get_string(b"PieceListFile", &mut pl_file) == 0;
        pip_get_integer(b"VerboseOutput", &mut self.m_verbose);
        pip_get_integer(b"FindWarpTransforms", &mut if_find_warp);
        let pl_name = String::from_utf8_lossy(&pl_file).into_owned();
        ierr = read_piece_list(
            if pl_found { Some(&pl_name) } else { None },
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
            nzu,
        );
        if ierr > 0 {
            exit_error_fmt!("Opening piece list file %s", CArg::Str(&pl_name));
        }
        if ierr < 0 {
            exit_from_value_read_error(ierr, "piece list file");
        }
        //
        // if no pieces, set up mocklist.  Montages can't be processed
        //
        if num_pc_list == 0 {
            for i in 0..nzu {
                ix_pc_list[i] = 0;
                iy_pc_list[i] = 0;
                iz_pc_list[i] = i as i32;
            }
            num_pc_list = nz;
        }
        if num_pc_list != nz {
            exit_error_fmt!(
                "Piece list should have an entry for each image; nz = %d, # in piece list = %d",
                CArg::Int(nz as i64),
                CArg::Int(num_pc_list as i64)
            );
        }
        fill_list_of_piece_z(
            &iz_pc_list[..num_pc_list as usize],
            &mut listz,
            &mut num_views,
        );
        if check_piece_list(
            &ix_pc_list,
            1,
            num_pc_list as usize,
            1,
            self.m_nx,
            &mut min_xpiece,
            &mut num_xpieces,
            &mut nx_overlap,
        ) + check_piece_list(
            &iy_pc_list,
            1,
            num_pc_list as usize,
            1,
            self.m_ny,
            &mut min_ypiece,
            &mut num_ypieces,
            &mut ny_overlap,
        ) != 0
        {
            exit_error(b"Piece list (which cannot be used anyway) does not fit requirements");
        }
        if num_xpieces * num_ypieces > 1 {
            exit_error(
                b"Program will not work with montages; blend images into single frames first",
            );
        }
        if num_views as i32 != nz || listz[0] != 0 || listz[num_views - 1] != nz - 1 {
            exit_error_fmt!(
                "The piece list should specify all Z values from 0 to %d",
                CArg::Int(nz as i64 - 1)
            );
        }
        let mut num_views = num_views as i32;
        //
        // Get the output file
        if pip_get_in_out_file(b"OutputFile", 2, &mut xf_file_out) != 0 {
            exit_error(b"No output file specified");
        }
        let xf_file_out = String::from_utf8_lossy(&xf_file_out).into_owned();
        //
        num_all_views = num_views;
        if if_find_warp != 0 {
            //
            // Make sure warping has no tilt angles
            ierr = pip_number_of_entries(b"TiltAngles", &mut nz);
            if nz > 0
                || pip_get_float(b"FirstTiltAngle", &mut self.m_xpeak) == 0
                || pip_get_float(b"TiltIncrement", &mut self.m_xpeak) == 0
                || pip_get_string(b"TiltFile", &mut self.m_list_string) == 0
            {
                exit_error(b"You cannot enter tilt angles when finding warp transforms");
            }
            nz = num_views;
            for t in self.m_tilt.iter_mut().take(nz as usize) {
                *t = 0.;
            }
            //
            // See if doing an aligned pair.  Make gray area criterion less because only the
            // current image will lose area
            raw_aligned_pair =
                pip_get_two_integers(b"RawAndAlignedPair", &mut iv, &mut num_all_views) == 0;
            if raw_aligned_pair {
                if num_all_views < 2 || iv < 2 || iv > num_all_views {
                    exit_error(
                        b"In RawAndAlignedPair entry, total views is less than 2 or view number is out of range",
                    );
                }
                iv_pair_offset = iv - 2;
                ierr = pip_get_boolean(b"AppendToWarpFile", &mut add_to_warps);
                self.m_crit_non_blank = 0.5;
            }
        } else {
            get_tilt_angles(&mut num_views, &mut self.m_tilt[..nzu]);
        }
        let _ = ierr;

        //
        //
        let mut test_name: Vec<u8> = Vec::new();
        if pip_get_string(b"TestOutput", &mut test_name) == 0 {
            unsafe { iiu_open(3, &String::from_utf8_lossy(&test_name), "NEW") };
            iiu_trans_header(3, 1);
            if_im_out = 1;
            im_file_out = Some(test_name);
        }
        if pip_get_integer(b"SingleTestPatch", &mut self.m_single_test_patch) == 0 {
            self.m_single_test_patch -= 1;
        }

        // Get lots of options
        // Need to know about both binnings, incoming and correlation, to interpret
        // filter entries
        if pip_get_integer(b"BinningToApply", &mut self.m_nbinning) == 0 {
            if self.m_nbinning <= 0 || self.m_nbinning > max_user_binning {
                exit_error(b"The entered value for binning is out of range");
            }
        }
        pip_get_integer(b"ImagesAreBinned", &mut self.m_images_binned);
        pip_get_float(b"PixelSize", &mut self.m_unbinned_pixel);
        ierr = pip_get_float(b"FilterRadius1", &mut radius1);
        self.get_filter_value_as_pix_or_nm("FilterRadius2", "Radius2InvNanometers", &mut radius2);
        self.get_filter_value_as_pix_or_nm("FilterSigma1", "Sigma1InvNanometers", &mut sigma1);
        self.get_filter_value_as_pix_or_nm("FilterSigma2", "Sigma2InvNanometers", &mut sigma2);

        pip_get_float(b"RotationAngle", &mut self.m_rot_angle);
        pip_get_boolean(b"ExcludeCentralPeak", &mut self.m_if_exclude);
        pip_get_two_integers(
            b"SecondPeakBoxSize",
            &mut nx_sec_peak_box,
            &mut ny_sec_peak_box,
        );

        {
            let (mut lx, mut ly) = (self.m_limit_shift_x, self.m_limit_shift_y);
            self.m_limiting_shift = self.get_two_ints_binned_or_unbinned(
                "ShiftLimitsXandY",
                "AxialShiftLimits",
                &mut lx,
                &mut ly,
            );
            self.m_limit_shift_x = lx;
            self.m_limit_shift_y = ly;
        }
        if self.m_limiting_shift > 1 {
            self.m_limiting_shift = -1;
        }

        pip_get_integer(b"IterateCorrelations", &mut num_iter);
        num_iter = 1.max(6.min(num_iter));
        pip_get_two_integers(b"LengthAndOverlap", &mut len_contour, &mut min_cont_overlap);
        if len_contour > 0 {
            len_contour = 3.max(len_contour);
        }
        min_cont_overlap = {
            let a = if 1 > min_cont_overlap {
                1
            } else {
                min_cont_overlap
            };
            if a < len_contour - 2 {
                a
            } else {
                len_contour - 2
            }
        };

        // Get border/trimming options specified as binned or unbinned
        self.get_two_ints_binned_or_unbinned(
            "BordersInXandY",
            "UnbinnedBordersXY",
            &mut nx_trim,
            &mut ny_trim,
        );
        ix_start = nx_trim;
        ix_end = self.m_nx - 1 - nx_trim;
        iy_start = ny_trim;
        iy_end = self.m_ny - 1 - ny_trim;
        self.get_two_ints_binned_or_unbinned(
            "XMinAndMax",
            "UnbinnedXMinAndMax",
            &mut ix_start,
            &mut ix_end,
        );
        self.get_two_ints_binned_or_unbinned(
            "YMinAndMax",
            "UnbinnedYMinAndMax",
            &mut iy_start,
            &mut iy_end,
        );
        if ix_start < 0
            || iy_start < 0
            || ix_end >= self.m_nx
            || iy_end >= self.m_ny
            || ix_end - ix_start < 24
            || iy_end - iy_start < 24
        {
            exit_error(b"Impossible amount to trim by or incorrect coordinates");
        }

        // Get antialiasing options
        ierr = pip_get_integer(b"AntialiasFilter", &mut self.m_i_anti_filt_type);
        if self.m_i_anti_filt_type > 6 {
            exit_error_fmt!(
                "Antialias filter type %d is out of range",
                CArg::Int(self.m_i_anti_filt_type as i64)
            );
        }
        if self.m_nbinning == 1 || self.m_i_anti_filt_type == 1 {
            self.m_i_anti_filt_type = 0;
        }
        {
            let v = (max_user_binning * self.m_nx) as i64;
            self.m_len_temp = if self.m_len_temp > v {
                self.m_len_temp
            } else {
                v
            };
        }
        if self.m_i_anti_filt_type > 0 {
            let v = ((16 * max_user_binning + 20) * self.m_nx) as i64;
            self.m_len_temp = if self.m_len_temp > v {
                self.m_len_temp
            } else {
                v
            };
        }

        // Set up temp array
        {
            let v = (self.m_nx as i64) * self.m_ny as i64;
            self.m_len_temp = if self.m_len_temp < v {
                self.m_len_temp
            } else {
                v
            };
        }
        self.m_tmp_array = vec![0.; self.m_len_temp.max(0) as usize];

        // More options
        pip_get_boolean(b"CorrelationCoefficient", &mut eval_ccc);
        pip_get_boolean(b"ReverseOrder", &mut self.m_reverse_order);
        pip_get_boolean(b"CumulativeCorrelation", &mut self.m_if_cumulate);
        pip_get_boolean(b"NoCosineStretch", &mut if_no_stretch);
        pip_get_boolean(b"AbsoluteCosineStretch", &mut self.m_if_abs_stretch);
        pip_get_boolean(b"LeaveTiltAxisShifted", &mut if_leave_axis);
        pip_get_three_floats(
            b"CentralPeakExclusionCriteria",
            &mut self.m_peak2_to_peak3_crit,
            &mut self.m_central_peak_max_width,
            &mut self.m_ub_width_ratio_crit,
        );
        if pip_get_boolean(b"RectangularLimits", &mut iv) == 0 {
            self.m_if_ellipse = 1 - iv;
        }
        ierr = pip_get_boolean(b"SearchMagChanges", &mut search_mag);

        // Process list of views with mags to search
        if search_mag != 0 && pip_get_string(b"ViewsWithMagChanges", &mut self.m_list_string) == 0 {
            match parselist(&String::from_utf8_lossy(&self.m_list_string)) {
                Ok(list) if !list.is_empty() => {
                    num_mag_views = list.len() as i32;
                    list_mag_views = list;
                }
                Ok(_) | Err(ParseListError::LeadingSlash) => {}
                Err(ParseListError::InvalidCharacter) => {
                    exit_error(b"Invalid list entry for ViewsWithMagChanges")
                }
            }
        }

        // Options related to mag search and rotation scan
        pip_get_two_floats(b"MagnificationLimits", &mut search_min, &mut search_max);
        if search_min as f64 >= search_max as f64 - 0.01 {
            exit_error(
                b"Limits for magnification search are out of order or too close to each other",
            );
        }
        if search_mag != 0 && (self.m_if_cumulate > 0 || if_leave_axis > 0 || if_find_warp != 0) {
            exit_error(b"Mag change cannot be searched with -cumulative, -leave, or -warp options");
        }
        if_scan_rotation = 1 - pip_get_two_floats(
            b"ScanRotationMaxAndStep",
            &mut scan_rot_max,
            &mut scan_rot_interval,
        );
        if if_scan_rotation > 0 {
            if search_mag != 0 || self.m_if_cumulate > 0 || if_leave_axis > 0 || if_find_warp != 0 {
                exit_error(
                    b"Rotation cannot be scanned with -search, -cumulative, -leave, or -warp options",
                );
            }
            if scan_rot_interval == 0. {
                num_rot_steps = 0;
            } else {
                if scan_rot_max <= 0.
                    || scan_rot_interval <= 0.
                    || scan_rot_interval as f64 > 2.7 * scan_rot_max as f64
                {
                    exit_error(
                        b"Scan rotation range and interval must be positive and interval should be less than 3 times the range",
                    );
                }
                num_rot_steps = b3dnint!(2. * scan_rot_max as f64 / scan_rot_interval as f64) + 1;
                scan_rot_interval = (2. * scan_rot_max as f64 / (num_rot_steps - 1) as f64) as f32;
                // One element more than the source allocates; see the module
                // comment.
                rot_scan_peaks = vec![0.; num_rot_steps as usize + 1];
            }
        }

        // Angle offset
        pip_get_float(b"AngleOffset", &mut angle_offset);
        for ivw in 0..num_views as usize {
            self.m_tilt[ivw] += angle_offset;
        }

        // Get skip or breaking list
        ierr = pip_get_string(b"SkipViews", &mut self.m_list_string);
        iz = pip_get_string(b"BreakAtViews", &mut self.m_list_string);
        if iz + ierr == 0 {
            exit_error(b"You cannot both skip views and break at views");
        }
        if ierr + iz == 1 {
            match parselist(&String::from_utf8_lossy(&self.m_list_string)) {
                Ok(list) if !list.is_empty() => {
                    self.m_num_skip = list.len() as i32;
                    self.m_list_skip = list;
                }
                _ => exit_error(b"Invalid entry for list of views to skip or break at"),
            }
            self.m_breaking = iz == 0;
        }

        // Reference file
        if pip_get_string(b"ReferenceFile", &mut ref_file) == 0 {
            self.m_iunit_ref = 2;
            unsafe { iiu_open(2, &String::from_utf8_lossy(&ref_file), "ro") };
            unsafe {
                iiu_ret_basic_head(
                    2,
                    nxyz_ref.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut self.m_mode,
                    &mut self.m_dmin2,
                    &mut self.m_dmax2,
                    &mut dmean2,
                );
            }
            if nxyz_ref[0] != self.m_nx || nxyz_ref[1] != self.m_ny {
                exit_error(
                    b"The reference file must be the same size in X and Y as the image file",
                );
            }
            iz_in_ref_file = 0;
            if pip_get_integer(b"ReferenceView", &mut ierr) == 0 {
                iz_in_ref_file = ierr - 1;
            }
            if iz_in_ref_file < 0 || iz_in_ref_file >= nxyz_ref[2] {
                exit_error(b"Reference view number is out of range");
            }
            if self.m_if_cumulate > 0 || if_leave_axis > 0 || self.m_breaking || if_find_warp != 0 {
                exit_error(
                    b"A reference file cannot be used with -break, -cumulative, -leave or -warp options",
                );
            }
        }

        reuse_prev_arrays =
            (num_iter > 1 || search_mag != 0 || num_rot_steps > 0) && if_im_out == 0;
        use_ref_file = self.m_iunit_ref == 2;
        self.m_nx_use = ix_end + 1 - ix_start;
        self.m_ny_use = iy_end + 1 - iy_start;
        self.m_cos_phi = cosd!(self.m_rot_angle);
        self.m_sin_phi = sind!(self.m_rot_angle);
        self.m_num_bound = 0;
        //
        // Now check if boundary model and load it in
        let mut bound_name: Vec<u8> = Vec::new();
        if pip_get_string(b"BoundaryModel", &mut bound_name) == 0 {
            self.m_in_file = String::from_utf8_lossy(&bound_name).into_owned();
            iobj_seed = 0;
            pip_get_integer(b"BoundaryObject", &mut iobj_seed);
            self.get_model_and_flags("Opening boundary model file");
            num_points = 0;
            iobj = 1;
            while iobj <= self.m_fm.max_mod_obj {
                fort_mod_obj_to_cont(iobj, &self.m_fm.obj_color, &mut imod_obj, &mut imod_cont);
                //
                // Use specified object, or any object with closed contours
                let io = (iobj - 1) as usize;
                if ((iobj_seed > 0 && imod_obj == iobj_seed)
                    || (iobj_seed == 0 && self.m_iobj_flags[(imod_obj - 1) as usize] == 0))
                    && self.m_fm.npt_in_obj[io] > 2
                {
                    ipnt = self.m_fm.object[self.m_fm.ibase_obj[io] as usize].abs();
                    iv = b3dnint!(self.m_fm.p_coord[(ipnt - 1) as usize][2]) + 1;
                    if iv >= 1 && iv <= num_all_views {
                        self.m_num_bound += 1;
                        if self.m_num_bound > LIMBOUND as i32 {
                            exit_error(b"Too many boundary contours for arrays");
                        }
                        let nb = (self.m_num_bound - 1) as usize;
                        iobj_bound[nb] = iobj;
                        self.m_num_in_bound[nb] = self.m_fm.npt_in_obj[io];
                        self.m_ind_bound[nb] = num_points;
                        num_points += self.m_fm.npt_in_obj[io];
                    }
                }
                iobj += 1;
            }
            if self.m_num_bound == 0 {
                exit_error(b"No qualifying boundary contours found in model");
            }
            //
            // Allocate point array and copy points to arrays
            self.m_xbound = vec![0.; num_points as usize];
            self.m_ybound = vec![0.; num_points as usize];
            if_bound_on_view = vec![0; num_all_views.max(0) as usize];
            bound_xmin = 1.0e10;
            bound_xmax = -1.0e10;
            bound_ymin = 1.0e10;
            bound_ymax = -1.0e10;
            for ixb in 0..self.m_num_bound as usize {
                iobj = iobj_bound[ixb];
                let io = (iobj - 1) as usize;
                ipnt = self.m_fm.object[(1 + self.m_fm.ibase_obj[io] - 1) as usize].abs() - 1;
                iv = b3dnint!(self.m_fm.p_coord[ipnt as usize][2]) + 1;
                if iv >= 1 && iv <= num_all_views {
                    if_bound_on_view[(iv - 1) as usize] = 1;
                    for ipt in 0..self.m_fm.npt_in_obj[io] {
                        ipnt = self.m_fm.object[(ipt + self.m_fm.ibase_obj[io]) as usize].abs() - 1;
                        x0 = self.m_fm.p_coord[ipnt as usize][0];
                        y0 = self.m_fm.p_coord[ipnt as usize][1];
                        if if_find_warp == 0 {
                            let (mut xo, mut yo) = (0., 0.);
                            self.adjust_coord(
                                self.m_tilt[(iv - 1) as usize],
                                0.,
                                x0 - self.m_xcen as f32,
                                y0 - self.m_ycen as f32,
                                &mut xo,
                                &mut yo,
                                false,
                            );
                            x0 = xo + self.m_xcen as f32;
                            y0 = yo + self.m_ycen as f32;
                        }
                        let bi = (self.m_ind_bound[ixb] + ipt) as usize;
                        self.m_xbound[bi] = x0;
                        self.m_ybound[bi] = y0;
                        bound_xmin = if bound_xmin < x0 { bound_xmin } else { x0 };
                        bound_xmax = if bound_xmax > x0 { bound_xmax } else { x0 };
                        bound_ymin = if bound_ymin < y0 { bound_ymin } else { y0 };
                        bound_ymax = if bound_ymax > y0 { bound_ymax } else { y0 };
                    }
                }
            }

            // This is the way it works for some reason...
            self.m_iobj_flags = Vec::new();
        }
        num_bound_all = self.m_num_bound;
        //
        // Get the view range and the minimum tilt view - needed for evaluating patches
        pip_get_two_integers(
            b"StartingEndingViews",
            &mut self.m_iz_start,
            &mut self.m_iz_end,
        );
        self.find_minimum_tilt_view(nz);
        //
        // Now check if doing patches and set up regular grid of them
        {
            let (mut px, mut py) = (self.m_nx_patch, self.m_ny_patch);
            ierr = self.get_two_ints_binned_or_unbinned(
                "SizeOfPatchesXandY",
                "UnbinnedPatchSizeXY",
                &mut px,
                &mut py,
            );
            self.m_nx_patch = px;
            self.m_ny_patch = py;
        }
        if ierr != 0 {
            if self.m_if_cumulate != 0 {
                exit_error(b"You cannot use cumulative correlation with patch tracking");
            }
            if search_mag != 0 || if_scan_rotation > 0 {
                exit_error(
                    b"You cannot search for a mag change or scan rotation with patch tracking",
                );
            }
            if self.m_breaking && if_find_warp == 0 {
                exit_error(b"You cannot break at views with patch tracking");
            }
            if use_ref_file {
                exit_error(b"You cannot do patch tracking with a reference image");
            }
            self.m_tracking = true;
            if if_find_warp != 0 && self.m_reverse_order != 0 {
                exit_error(b"You cannot find warp transforms in reverse order");
            }
            if self.m_nx_patch > self.m_nx_use || self.m_ny_patch > self.m_ny_use {
                exit_error(b"Patches do not fit within trimmed area of image");
            }
            if pip_get_two_integers(
                b"VaryingToPatchSizeXY",
                &mut max_nx_patch,
                &mut max_ny_patch,
            ) == 0
            {
                if max_nx_patch < self.m_nx_patch || max_ny_patch < self.m_ny_patch {
                    exit_error_fmt!(
                        "The unbinned patch size, %dx%d, is bigger than the size being tested to, %dx%d",
                        CArg::Int((self.m_nx_patch * self.m_images_binned) as i64),
                        CArg::Int((self.m_ny_patch * self.m_images_binned) as i64),
                        CArg::Int(max_nx_patch as i64),
                        CArg::Int(max_ny_patch as i64)
                    );
                }
                max_nx_patch /= self.m_images_binned;
                max_ny_patch /= self.m_images_binned;
                if max_nx_patch > self.m_nx_use || max_ny_patch > self.m_ny_use {
                    exit_error(
                        b"Maximum patch size to be tested will not fit within trimmed area of image",
                    );
                }
            } else {
                max_nx_patch = self.m_nx_patch;
                max_ny_patch = self.m_ny_patch;
            }

            if_patch_num =
                1 - pip_get_two_integers(b"NumberOfPatchesXandY", &mut num_xpatch, &mut num_ypatch);
            ix = pip_get_two_floats(b"OverlapOfPatchesXandY", &mut frac_xover, &mut frac_yover);
            let mut seed_name: Vec<u8> = Vec::new();
            iobj_seed = pip_get_string(b"SeedModel", &mut seed_name);
            if iobj_seed == 0 {
                self.m_in_file = String::from_utf8_lossy(&seed_name).into_owned();
            }
            if (1 - if_patch_num) + iobj_seed + ix < 2 {
                exit_error(b"You must enter only one of the -number, -overlap, or -seed options");
            }

            // Determine GPU usage
            self.m_use_gpu = crate::imod::libcfshr::b3dutil::get_standard_gpu_options(
                &mut if_gpu_by_env,
                Some(&mut act_gpu_fail_option),
                Some(&mut act_gpu_fail_environ),
            );
            if self.m_use_gpu >= 0 {
                if if_im_out != 0 {
                    exit_error(b"You cannot get test output with processing on the GPU");
                }
                if self.m_i_anti_filt_type > 0 {
                    exit_error(b"You cannot use antialiased reduction with processing on the GPU");
                }
                if self.m_nbinning > 4 {
                    exit_error(b"You cannot bin by more than 4 with processing on the GPU");
                }
                self.m_txc_gpu = Some(TxcGPU::new());
                if self.m_txc_gpu.as_mut().unwrap().gpu_available(
                    self.m_use_gpu,
                    &mut gpu_memory,
                    if self.m_verbose != 0 { 1 } else { 0 },
                ) == 0
                {
                    ierr = if if_gpu_by_env != 0 {
                        act_gpu_fail_environ
                    } else {
                        act_gpu_fail_option
                    };
                    if ierr > 1 {
                        exit_error(b"Use of a GPU was requested but none is available");
                    }
                    printf!(
                        "%sUse of a GPU was requested but none is available; using the CPU\n",
                        CArg::Str(if ierr != 0 {
                            "MESSAGE: Tiltxcorr - "
                        } else {
                            ""
                        })
                    );
                    self.m_use_gpu = -1;
                }
            }

            // Full array cache is needed, allocate starter pair of images here
            self.allocate_full_cache(0, 2);

            if iobj_seed != 0 {
                //
                // Specify regular array of patches, either by number
                if if_patch_num != 0 {
                    if num_xpatch < 1 || num_ypatch < 1 {
                        exit_error(b"Number of patches must be positive");
                    }
                } else {
                    //
                    // Or by overlap factors
                    if frac_xover > frac_over_max || frac_yover > frac_over_max {
                        exit_error(b"Fractional overlap between patches is too high");
                    }
                    x_overlap = frac_xover * self.m_nx_patch as f32;
                    y_overlap = frac_yover * self.m_ny_patch as f32;
                    num_xpatch = 1.max(b3dnint!(
                        ((self.m_nx_use - (max_nx_patch - self.m_nx_patch)) as f32 - x_overlap)
                            / (self.m_nx_patch as f32 - x_overlap)
                    ));
                    num_ypatch = 1.max(b3dnint!(
                        ((self.m_ny_use - (max_ny_patch - self.m_ny_patch)) as f32 - y_overlap)
                            / (self.m_ny_patch as f32 - y_overlap)
                    ));
                }
                self.m_num_patches_all = num_xpatch * num_ypatch;
                self.allocate_patch_arrays();
                x_overlap = ((num_xpatch * max_nx_patch - self.m_nx_use) as f64
                    / (if 1. > num_xpatch as f64 - 1. {
                        1.
                    } else {
                        num_xpatch as f64 - 1.
                    })) as f32;
                y_overlap = ((num_ypatch * max_ny_patch - self.m_ny_use) as f64
                    / (if 1. > num_ypatch as f64 - 1. {
                        1.
                    } else {
                        num_ypatch as f64 - 1.
                    })) as f32;

                // Determine domains for local outlier analysis
                if if_find_warp == 0 {
                    pip_get_two_integers(
                        b"LocalDomainSizesXandY",
                        &mut self.m_domain_xsize,
                        &mut self.m_domain_ysize,
                    );
                    pip_get_two_floats(
                        b"CriterionProbabilities",
                        &mut self.m_pred_crit_prob,
                        &mut self.m_abs_prob_crit,
                    );
                    if (self.m_domain_ysize == 0 && self.m_domain_xsize != 0)
                        || (self.m_domain_xsize == 0 && self.m_domain_ysize != 0)
                        || self.m_domain_xsize < 0
                        || self.m_domain_ysize < 0
                    {
                        exit_error(b"Inappropriate entry for domain sizes");
                    }
                    if self.m_domain_xsize != 0 {
                        let mut nd = 0;
                        let mut starts = std::mem::take(&mut self.m_domain_xstarts);
                        self.setup_domains(
                            num_xpatch,
                            self.m_domain_xsize,
                            &mut nd,
                            &mut starts,
                            &mut domain_xassigns,
                        );
                        self.m_num_xdomains = nd;
                        self.m_domain_xstarts = starts;
                        let mut starts = std::mem::take(&mut self.m_domain_ystarts);
                        self.setup_domains(
                            num_ypatch,
                            self.m_domain_ysize,
                            &mut nd,
                            &mut starts,
                            &mut domain_yassigns,
                        );
                        self.m_num_ydomains = nd;
                        self.m_domain_ystarts = starts;
                        for iyd in 0..self.m_num_ydomains as usize {
                            for ixd in 0..self.m_num_xdomains as usize {
                                for j in self.m_domain_ystarts[iyd]..self.m_domain_ystarts[iyd + 1]
                                {
                                    for i in
                                        self.m_domain_xstarts[ixd]..self.m_domain_xstarts[ixd + 1]
                                    {
                                        let indp = (i + j * num_xpatch) as usize;
                                        self.m_patch_domains[indp] = domain_xassigns[i as usize]
                                            + self.m_num_xdomains * domain_yassigns[j as usize];
                                    }
                                }
                            }
                        }
                    }
                }

                // Set up patch centers
                for j in 0..num_ypatch {
                    yval = ((iy_start as f32 + j as f32 * (max_ny_patch as f32 - y_overlap)) as f64
                        + 0.5 * max_ny_patch as f64) as f32;
                    if num_ypatch == 1 {
                        yval = ((iy_end + 1 + iy_start) as f64 / 2.) as f32;
                    }
                    for i in 0..num_xpatch {
                        let indp = (i + j * num_xpatch) as usize;
                        self.m_patch_cen_xall[indp] =
                            ((ix_start as f32 + i as f32 * (max_nx_patch as f32 - x_overlap))
                                as f64
                                + 0.5 * max_nx_patch as f64) as f32;
                        if num_xpatch == 1 {
                            self.m_patch_cen_xall[indp] =
                                ((ix_end + 1 + ix_start) as f64 / 2.) as f32;
                        }
                        self.m_patch_cen_yall[indp] = yval;
                        self.m_patch_xinds[indp] = i;
                        self.m_patch_yinds[indp] = j;
                        if self.m_verbose != 0
                            && (self.m_single_test_patch < 0
                                || self.m_single_test_patch == indp as i32)
                        {
                            printf!(
                                "%d %d at %.0f, %.0f  domain %d\n",
                                CArg::Int(i as i64),
                                CArg::Int(j as i64),
                                CArg::Dbl(self.m_patch_cen_xall[indp] as f64),
                                CArg::Dbl(self.m_patch_cen_yall[indp] as f64),
                                CArg::Int(self.m_patch_domains[indp] as i64)
                            );
                        }
                    }
                }
            } else {
                //
                // Or get a seed model to specify patches
                pip_get_integer(b"SeedObject", &mut iobj_seed);
                self.get_model_and_flags("Opening seed model file");

                // Loop twice, first time to get count and allocate, second time to save
                for iloop in 1..=2 {
                    self.m_num_patches_all = 0;
                    iobj = 1;
                    while iobj <= self.m_fm.max_mod_obj {
                        fort_mod_obj_to_cont(
                            iobj,
                            &self.m_fm.obj_color,
                            &mut imod_obj,
                            &mut imod_cont,
                        );
                        let io = (iobj - 1) as usize;
                        //
                        // Use specified object, or any object with scattered points
                        if (iobj_seed > 0 && imod_obj == iobj_seed)
                            || (iobj_seed == 0 && self.m_iobj_flags[(imod_obj - 1) as usize] == 2)
                        {
                            for ipt in 0..self.m_fm.npt_in_obj[io] {
                                ipnt = self.m_fm.object[(ipt + self.m_fm.ibase_obj[io]) as usize]
                                    .abs()
                                    - 1;
                                iv = b3dnint!(self.m_fm.p_coord[ipnt as usize][2]) + 1;
                                //
                                // Get point down to zero degrees and make sure patch fits
                                if iv >= 1 && iv <= num_all_views {
                                    x0 = self.m_fm.p_coord[ipnt as usize][0];
                                    y0 = self.m_fm.p_coord[ipnt as usize][1];
                                    if if_find_warp == 0 {
                                        let (mut xo, mut yo) = (0., 0.);
                                        self.adjust_coord(
                                            self.m_tilt[(iv - 1) as usize],
                                            0.,
                                            x0 - self.m_xcen as f32,
                                            y0 - self.m_ycen as f32,
                                            &mut xo,
                                            &mut yo,
                                            false,
                                        );
                                        x0 = xo + self.m_xcen as f32;
                                        y0 = yo + self.m_ycen as f32;
                                    }
                                    if b3dnint!(x0) - self.m_nx_patch / 2 >= ix_start
                                        && b3dnint!(x0) + self.m_nx_patch / 2 <= ix_end
                                        && b3dnint!(y0) - self.m_ny_patch / 2 >= iy_start
                                        && b3dnint!(y0) + self.m_ny_patch / 2 <= iy_end
                                    {
                                        if iloop > 1 {
                                            self.m_patch_cen_xall
                                                [self.m_num_patches_all as usize] = x0;
                                            self.m_patch_cen_yall
                                                [self.m_num_patches_all as usize] = y0;
                                        }
                                        self.m_num_patches_all += 1;
                                    }
                                }
                            }
                        }
                        iobj += 1;
                    }
                    if self.m_num_patches_all == 0 {
                        exit_error(
                            b"No qualifying points found in seed model; specify object or make it scattered points",
                        );
                    }
                    if iloop == 1 {
                        self.allocate_patch_arrays();
                    }
                }
                self.m_iobj_flags = Vec::new();
            }

            // Set up for patch expansion
            if self.m_tracking
                && (pip_get_two_floats(
                    b"MaxPatchExpansion",
                    &mut max_patch_expand,
                    &mut max_tilted_expand,
                ) + pip_get_float(b"MinStructureModeFrac", &mut min_mode_frac)
                    < 2)
            {
                if im_file_out.is_some()
                    && ((max_patch_expand > 1. && self.m_single_test_patch < 0)
                        || max_tilted_expand > 1.)
                {
                    exit_error(b"You cannot get test output with patch expansion");
                }
                pip_get_float(b"MaxFracOfPatchesToElim", &mut max_frac_to_elim);
                pip_get_float(b"TiltedExpandDropFrac", &mut expand_drop_frac);
                pip_get_integer(b"SdReducedBoxSize", &mut self.m_sd_box_reduced);
                if self.m_sd_box_reduced < 5 || self.m_sd_box_reduced > 100 {
                    exit_error(b"Box size in reduced image must be between 5 and 100");
                }
                pip_get_float(b"PercentileToMatch", &mut expand_to_pctl);
                if expand_to_pctl < 1. || expand_to_pctl > 99. {
                    exit_error(b"Percentile SD level to match must be between 1 and 99");
                }
                pip_get_float(b"FractionOfMaxVariation", &mut frac_variation);
                if (frac_variation as f64) < 0.1 || frac_variation as f64 > 0.99 {
                    exit_error(
                        b"Fraction of max variation to use to pick reduction must be between 0.1 and 0.99",
                    );
                }

                ierr = pip_get_string(b"SdBinList", &mut self.m_list_string);
                ind = pip_get_two_floats(
                    b"TargetPixelMinAndMax",
                    &mut min_sd_target_pixel,
                    &mut max_sd_target_pixel,
                );
                if ind == 0 && ierr == 0 {
                    exit_error(b"You cannot enter both -SdBinList and -TargetPixelMinAndMax");
                }
                if ierr == 0 {
                    let mut list = match parselist(&String::from_utf8_lossy(&self.m_list_string)) {
                        Ok(list) if !list.is_empty() => list,
                        _ => exit_error(b"Getting bin list"),
                    };
                    num_sd_bins = list.len() as i32;
                    if num_sd_bins > MAX_BINNINGS as i32 {
                        exit_error(b"Too many binnings in list for SD analysis");
                    }
                    rs_sort_ints(&mut list, num_sd_bins);
                    if list[0] <= 0 {
                        exit_error(b"Binning for SD analysis must be at least 1");
                    }
                    sd_bin_list = Some(list);
                } else {
                    if self.m_unbinned_pixel <= 0. {
                        exit_error(
                            b"You must enter an unbinned pixel size to use target pixel sizes to set reductions for SD analysis",
                        );
                    }
                    if min_sd_target_pixel <= 0.
                        || max_sd_target_pixel <= 0.
                        || max_sd_target_pixel < min_sd_target_pixel
                    {
                        exit_error(
                            b"Target pixel sizes for SD analysis out of range or out of order",
                        );
                    }
                    let mut list = vec![0i32; MAX_BINNINGS];
                    cumul_binning =
                        min_sd_target_pixel / (self.m_unbinned_pixel * self.m_images_binned as f32);
                    final_bin = 1.max(b3dnint!(
                        max_sd_target_pixel / (self.m_unbinned_pixel * self.m_images_binned as f32)
                    ));
                    num_sd_bins = 0;
                    while num_sd_bins == 0 || list[(num_sd_bins - 1) as usize] < final_bin {
                        ix = 1.max(b3dnint!(cumul_binning));
                        if num_sd_bins == 0 || ix > list[(num_sd_bins - 1) as usize] {
                            if num_sd_bins >= MAX_BINNINGS as i32 {
                                exit_error(
                                    b"The range of target pixel sizes gives too many binnings",
                                );
                            }
                            list[num_sd_bins as usize] = ix;
                            num_sd_bins += 1;
                        }
                        cumul_binning = (cumul_binning as f64 * 1.2) as f32;
                    }
                    sd_bin_list = Some(list);
                }
                for indb in 0..num_sd_bins as usize {
                    sd_means[indb].resize(self.m_num_patches_all as usize, 0.);
                    sd_num_pix[indb].resize(self.m_num_patches_all as usize, 0);
                    min_sd[indb] = 1.0e30;
                    pct1_sd[indb] = 1.0e30;
                }

                // Sigmoid curve
                if pip_get_two_floats(
                    b"SigmoidPowerAndHalfRise",
                    &mut self.m_sigmoid_power,
                    &mut self.m_sigmoid_half_rise,
                ) == 0
                {
                    if min_mode_frac > 0. {
                        self.m_sigmoid_power = 4.;
                        self.m_sigmoid_half_rise = -1.;
                    }
                } else {
                    if (self.m_sigmoid_half_rise >= 0. && (self.m_sigmoid_half_rise as f64) < 0.1)
                        || self.m_sigmoid_half_rise as f64 > 0.9
                    {
                        exit_error(
                            b"The half-rise point of the sigmoid curve must be between 0.1 and 0.9",
                        );
                    }
                    if self.m_sigmoid_power as f64 <= 0.01 {
                        exit_error(b"The power of the sigmoid curve must be positive");
                    }
                }
            }

            // Get warping arrays
            if if_find_warp != 0 {
                let n = (self.m_num_patches_all + 10) as usize;
                x_control = vec![0.; n];
                y_control = vec![0.; n];
                x_vector = vec![0.; n];
                y_vector = vec![0.; n];
            }
            //
            // Now eliminate patches outside boundary model or just copy centers
            // This is redone on every view for warping if there is more than one contour
            self.make_patch_list_inside_boundary();
            if if_find_warp != 0 && self.m_num_patches < 3 {
                exit_error(
                    b"There are too few patches inside the boundary contour(s) to define control points",
                );
            }
            if self.m_num_patches == 0 {
                exit_error(b"No patches are sufficiently inside the boundary contour(s)");
            }
            //
            // Get prealign transforms, then eliminate patches that have too much blank area
            // on starting view
            pl_file.clear();
            self.m_if_read_xfs = 1 - pip_get_string(b"PrealignmentTransformFile", &mut pl_file);
            if raw_aligned_pair && self.m_if_read_xfs == 0 {
                exit_error(b"Prealignment F transforms must be entered to do raw-aligned pair");
            }
            self.m_nx_unali = self.m_nx;
            self.m_ny_unali = self.m_ny;
            min_expand_border = {
                let a = 16 / self.m_images_binned;
                let b = (nx_trim * ny_trim) / 2;
                if a < b { a } else { b }
            };
            if self.m_if_read_xfs != 0 {
                if raw_aligned_pair && self.m_images_binned > 1 {
                    exit_error(b"You cannot do a raw and aligned pair with binned input");
                }
                let nav = num_all_views.max(0) as usize;
                f_preali = vec![0.; nav * 6];
                self.m_dx_preali = vec![0.; nav];
                self.m_dy_preali = vec![0.; nav];
                let pl_name = String::from_utf8_lossy(&pl_file).into_owned();
                self.check_for_warp_file(&pl_name);
                let fp_xf = match std::fs::File::open(&pl_name) {
                    Ok(file) => file,
                    Err(_) => exit_error_fmt!(
                        "Opening file of prealign transforms: %s",
                        CArg::Str(&pl_name)
                    ),
                };
                let mut reader = std::io::BufReader::new(fp_xf);
                ierr = read_all_xforms(&mut reader, &mut f_preali, num_all_views, &mut iv);
                if ierr != 0 {
                    exit_from_xf_read_error(ierr, "prealign");
                }
                drop(reader);
                if iv != num_all_views {
                    exit_error(b"Not enough transforms in prealign transform file");
                }
                for ivw in 0..nav {
                    self.m_dx_preali[ivw] = f_preali[6 * ivw + 4] / self.m_images_binned as f32;
                    self.m_dy_preali[ivw] = f_preali[6 * ivw + 5] / self.m_images_binned as f32;
                }
                //
                // Get unaligned size if it differs, adjust for binning
                if pip_get_two_integers(
                    b"UnalignedSizeXandY",
                    &mut self.m_nx_unali,
                    &mut self.m_ny_unali,
                ) == 0
                {
                    self.m_nx_unali /= self.m_images_binned;
                    self.m_ny_unali /= self.m_images_binned;
                    if raw_aligned_pair {
                        exit_error(
                            b"Input file with raw and aligned pair must be same size as unaligned stack",
                        );
                    }
                }
                if if_find_warp == 0 {
                    //
                    // Scan around the minimum tilt for the view with the most patches left
                    num_best = 0;
                    iv = self.m_min_tilt - 2;
                    while iv <= self.m_min_tilt + 2 {
                        if iv >= self.m_iz_start
                            && iv <= self.m_iz_end
                            && (self.m_breaking
                                || number_in_list(iv, Some(&self.m_list_skip), self.m_num_skip, 0)
                                    == 0)
                        {
                            ind = 0;
                            self.mark_usable_patches(iv, &mut ind);
                            if self.m_verbose != 0 {
                                printf!(
                                    "Usable patches for view: %d  %d\n",
                                    CArg::Int(iv as i64),
                                    CArg::Int(ind as i64)
                                );
                            }
                            if ind > num_best
                                || (ind == num_best
                                    && (iv - self.m_min_tilt).abs()
                                        < (ind_best - self.m_min_tilt).abs())
                            {
                                num_best = ind;
                                ind_best = iv;
                            }
                        }
                        iv += 1;
                    }
                    //
                    // Mark them for real and copy the usable ones down
                    self.m_min_tilt = ind_best;
                    ind = 0;
                    self.mark_usable_patches(self.m_min_tilt, &mut ind);
                    ind = 0;
                    for ip in 0..self.m_num_patches as usize {
                        if self.m_tmp_array[ip] > 0. {
                            let d = ind as usize;
                            self.m_patch_xinds[d] = self.m_patch_xinds[ip];
                            self.m_patch_yinds[d] = self.m_patch_yinds[ip];
                            self.m_patch_domains[d] = self.m_patch_domains[ip];
                            self.m_patch_cen_x[d] = self.m_patch_cen_x[ip];
                            self.m_patch_cen_y[d] = self.m_patch_cen_y[ip];
                            ind += 1;
                        }
                    }
                    self.m_num_patches = ind;
                    if ind == 0 {
                        exit_error(
                            b"No patches have sufficient image data near the minimum tilt view",
                        );
                    }
                }
            }

            // Set up patch sizes if no expansion
            // On the GPU, it screws up if the binned extract isn't even so round up for that
            if self.m_use_gpu >= 0 && self.m_nbinning > 1 {
                if self.m_nx_patch % (2 * self.m_nbinning) != 0 {
                    self.m_nx_patch =
                        2 * self.m_nbinning * (self.m_nx_patch / (2 * self.m_nbinning) + 1);
                }
                if self.m_ny_patch % (2 * self.m_nbinning) != 0 {
                    self.m_ny_patch =
                        2 * self.m_nbinning * (self.m_ny_patch / (2 * self.m_nbinning) + 1);
                }
            }

            nx_patch_orig = self.m_nx_patch;
            self.m_nx_use = nx_patch_orig;
            ny_patch_orig = self.m_ny_patch;
            self.m_ny_use = ny_patch_orig;
            for ip in 0..self.m_num_patches as usize {
                self.m_patch_nx[ip] = self.m_nx_patch;
                self.m_patch_ny[ip] = self.m_ny_patch;
            }

            /*
             * Analyze SDs at various binnings for patch expansion
             */
            if let Some(sd_bin_list) = sd_bin_list.as_ref() {
                idim2 = (self.m_nx / sd_bin_list[0] + 2) * (self.m_ny / sd_bin_list[0] + 2);
                self.m_sd_arr = vec![0.; idim2 as usize];
                self.m_sum_arr = vec![0.; idim2 as usize];
                self.m_sqr_arr = vec![0.; idim2 as usize];

                // Look up minimum tilt Z value and load into last buffer
                for i in 0..nz {
                    if iz_pc_list[i as usize] + 1 == self.m_min_tilt {
                        self.m_iz_last = i;
                    }
                }
                self.m_iz_cur = self.m_iz_last - 1;
                let mut load_ind = 0;
                self.get_full_array_and_lines(1, self.m_iz_last, &mut load_ind);
                full_ind = load_ind as usize;

                // Mark as not loaded because it wasn't copied to GPU
                if self.m_use_gpu >= 0 {
                    self.m_iz_loaded[load_ind as usize] = -1;
                }

                // get limits for the SD map
                self.set_sd_map_limits(min_expand_border, self.m_min_tilt - 1);

                // Loop on binnings
                max_cv = 0.;
                min_cv = 1.0e30;
                for ib in 0..num_sd_bins as usize {
                    sd_bin = sd_bin_list[ib];
                    self.get_patch_sd_stats(
                        sd_bin,
                        ib as i32,
                        full_ind,
                        &mut sd_num_pix[ib],
                        &mut sd_means[ib],
                        &mut sd_xoff,
                        &mut sd_yoff,
                        &mut min_sd[ib],
                        &mut pct1_sd[ib],
                    );
                    // `VEC_MINIMUM`/`VEC_MAXIMUM`: `std::min_element` and
                    // `std::max_element` over the whole vector, keeping the
                    // first extreme.
                    vec_min = sd_means[ib][0];
                    vec_max = sd_means[ib][0];
                    for &v in &sd_means[ib][1..] {
                        if v < vec_min {
                            vec_min = v;
                        }
                        if vec_max < v {
                            vec_max = v;
                        }
                    }
                    avg_sd(
                        &sd_means[ib],
                        self.m_num_patches,
                        &mut mean_avg,
                        &mut mean_sd,
                        &mut temp,
                    );
                    printf!(
                        "red %d min %.3f max %.3f mean %.3f sd %.3f pix min %.3f pct1 %.3f  cv %.2f %.2f  mmr %.2f %.2f\n",
                        CArg::Int(sd_bin_list[ib] as i64),
                        CArg::Dbl(100. * vec_min as f64),
                        CArg::Dbl(100. * vec_max as f64),
                        CArg::Dbl(100. * mean_avg as f64),
                        CArg::Dbl(100. * mean_sd as f64),
                        CArg::Dbl(100. * min_sd[ib] as f64),
                        CArg::Dbl(100. * pct1_sd[ib] as f64),
                        CArg::Dbl(100. * mean_sd as f64 / mean_avg as f64),
                        CArg::Dbl(100. * mean_sd as f64 / (mean_avg - pct1_sd[ib]) as f64),
                        CArg::Dbl((vec_max / vec_min) as f64),
                        CArg::Dbl(((vec_max - pct1_sd[ib]) / (vec_min - pct1_sd[ib])) as f64)
                    );
                    {
                        let v = mean_sd / (mean_avg - pct1_sd[ib]);
                        max_cv = if max_cv > v { max_cv } else { v };
                        min_cv = if min_cv < v { min_cv } else { v };
                    }
                    nx_sd = (self.m_sd_xend + 1 - self.m_sd_xstart) / sd_bin;
                    ny_sd = (self.m_sd_yend + 1 - self.m_sd_ystart) / sd_bin;
                    if self.m_verbose > 1 {
                        let buffer = format!("sdmap-b{}.mrc", sd_bin);
                        with_mrc_data!(self.m_sd_arr, |d| mrc_write_image_to_file(
                            &buffer,
                            &mut d,
                            MRC_MODE_FLOAT,
                            nx_sd,
                            ny_sd
                        ));
                    }
                }

                // Get optimal binning at the point where it exceeds the criterion
                ind = num_sd_bins - 1;
                sd_bin = sd_bin_list[ind as usize];
                for indb in 0..num_sd_bins as usize {
                    avg_sd(
                        &sd_means[indb],
                        self.m_num_patches,
                        &mut mean_avg,
                        &mut mean_sd,
                        &mut temp,
                    );
                    if mean_sd / (mean_avg - pct1_sd[indb]) - min_cv
                        >= frac_variation * (max_cv - min_cv)
                    {
                        sd_bin = sd_bin_list[indb];
                        ibin = indb;
                        break;
                    }
                }
                printf!(
                    "Minimum and maximum variations (x 100): %.2f  %.2f; chosen reduction %d\n",
                    CArg::Dbl(100. * min_cv as f64),
                    CArg::Dbl(100. * max_cv as f64),
                    CArg::Int(sd_bin as i64)
                );
                expand_vec.resize(self.m_num_patches as usize, 1.);

                // Get the summed SD and the percentile
                for ip in 0..self.m_num_patches as usize {
                    self.m_tmp_array[ip] =
                        (sd_means[ibin][ip] - pct1_sd[ibin]) * sd_num_pix[ibin][ip] as f32;
                    if self.m_verbose > 1 {
                        printf!(
                            "%d  %.2f\n",
                            CArg::Int(ip as i64),
                            CArg::Dbl(self.m_tmp_array[ip] as f64)
                        );
                    }
                }
                self.find_histogram_mode(
                    &self.m_tmp_array[..self.m_num_patches as usize],
                    self.m_num_patches,
                    0.1,
                    sd_sum_lower_ratio,
                    &mut sum_mode,
                    None,
                );
                elim_crit = min_mode_frac * sum_mode;

                summed_sd_crit = percentile_float(
                    b3dnint!(0.01 * expand_to_pctl as f64 * self.m_num_patches as f64),
                    &mut self.m_tmp_array,
                    self.m_num_patches,
                );
                printf!(
                    "Summed SD to match: %.2f  mode of summed SD: %.2f  elimination criterion: %.2f\n",
                    CArg::Dbl(summed_sd_crit as f64),
                    CArg::Dbl(sum_mode as f64),
                    CArg::Dbl(elim_crit as f64)
                );

                // Get the map for this binning
                nx_sd = (self.m_sd_xend + 1 - self.m_sd_xstart) / sd_bin;
                ny_sd = (self.m_sd_yend + 1 - self.m_sd_ystart) / sd_bin;
                make_standard_dev_map(
                    &self.m_full_cache[full_ind],
                    self.m_nx,
                    self.m_sd_xstart,
                    -self.m_sd_xend,
                    self.m_sd_ystart,
                    self.m_sd_yend,
                    -sd_bin,
                    self.m_sd_box_reduced,
                    &mut self.m_sd_arr,
                    &mut self.m_sum_arr,
                    &mut self.m_sqr_arr,
                    &mut sd_xoff,
                    &mut sd_yoff,
                );
                self.sigmoid_scale_sd_map(sd_bin);

                // Expand patches, moving away from edges as needed
                cen_xsave = self.m_patch_cen_x.clone();
                cen_ysave = self.m_patch_cen_y.clone();
                num_drop = 0;
                if max_patch_expand > 1. {
                    let v = (max_patch_expand as f64 - 1.) / 2.01;
                    expand_step = if (expand_step as f64) < v {
                        expand_step
                    } else {
                        v as f32
                    };
                }
                for ip in 0..self.m_num_patches as usize {
                    sd_sum = (sd_means[ibin][ip] - pct1_sd[ibin]) * sd_num_pix[ibin][ip] as f32;
                    self.m_tmp_array[ip] = 0.;
                    expand_fac = 1.;
                    nx_temp = self.m_nx_patch;
                    ny_temp = self.m_ny_patch;

                    // Loop until criterion is reached
                    while sd_sum < summed_sd_crit {
                        if (expand_fac + expand_step) as f64
                            > max_patch_expand as f64 + 0.25 * expand_step as f64
                        {
                            if sd_sum < elim_crit {
                                self.m_tmp_array[ip] = 1.;
                                num_drop += 1;
                            }
                            break;
                        }
                        expand_fac += expand_step;

                        // Get trial expanded size, move X patch center if necessary, get X limits
                        nx_temp = 2 * b3dnint!((self.m_nx_patch as f32 * expand_fac) as f64 / 2.);
                        ny_temp = 2 * b3dnint!((self.m_ny_patch as f32 * expand_fac) as f64 / 2.);
                        if self.m_use_gpu >= 0 && self.m_nbinning > 1 {
                            if nx_temp % (2 * self.m_nbinning) != 0 {
                                nx_temp =
                                    2 * self.m_nbinning * (nx_temp / (2 * self.m_nbinning) + 1);
                            }
                            if ny_temp % (2 * self.m_nbinning) != 0 {
                                ny_temp =
                                    2 * self.m_nbinning * (ny_temp / (2 * self.m_nbinning) + 1);
                            }
                        }
                        if self.m_patch_cen_x[ip] + (nx_temp / 2) as f32
                            > (self.m_nx - min_expand_border) as f32
                        {
                            self.m_patch_cen_x[ip] =
                                ((self.m_nx - min_expand_border) - nx_temp / 2) as f32;
                        }
                        if self.m_patch_cen_x[ip] - ((nx_temp / 2) as f32)
                            < min_expand_border as f32
                        {
                            self.m_patch_cen_x[ip] = (min_expand_border + nx_temp / 2) as f32;
                        }
                        px_start = b3dnint!(self.m_patch_cen_x[ip]) - nx_temp / 2;
                        px_end = (px_start + nx_temp - 1).min(self.m_sd_xend);
                        px_start = px_start.max(self.m_sd_xstart);

                        // Move Y center if needed, get Y limits
                        if self.m_patch_cen_y[ip] + (ny_temp / 2) as f32
                            > (self.m_ny - min_expand_border) as f32
                        {
                            self.m_patch_cen_y[ip] =
                                ((self.m_ny - min_expand_border) - ny_temp / 2) as f32;
                        }
                        if self.m_patch_cen_y[ip] - ((ny_temp / 2) as f32)
                            < min_expand_border as f32
                        {
                            self.m_patch_cen_y[ip] = (min_expand_border + ny_temp / 2) as f32;
                        }
                        py_start = b3dnint!(self.m_patch_cen_y[ip]) - ny_temp / 2;
                        py_end = (py_start + ny_temp - 1).min(self.m_sd_yend);
                        py_start = py_start.max(self.m_sd_ystart);
                        ix = (px_end + 1 - px_start) / sd_bin;
                        iy = (py_end + 1 - py_start) / sd_bin;

                        // Get new sd value
                        array_min_max_mean(
                            &self.m_sd_arr,
                            nx_sd,
                            ny_sd,
                            px_start / sd_bin + sd_xoff,
                            px_end / sd_bin + sd_xoff,
                            py_start / sd_bin + sd_yoff,
                            py_end / sd_bin + sd_yoff,
                            &mut self.m_dmin2,
                            &mut self.m_dmax2,
                            &mut sd_sum,
                        );
                        sd_sum = (sd_sum - pct1_sd[ibin]) * ix as f32 * iy as f32;
                    }

                    // Save patch size, expand factor and sdSum in case have to retain some of
                    // the ones that are being dropped
                    self.m_patch_nx[ip] = nx_temp;
                    self.m_patch_ny[ip] = ny_temp;
                    expand_vec[ip] = expand_fac;
                    sd_sum_vec.push(sd_sum);
                }

                // If too many would be dropped, limit it
                nx_temp = (max_frac_to_elim * self.m_num_patches as f32) as i32;
                if num_drop > nx_temp {
                    sd_sum_tmp = sd_sum_vec.clone();
                    let count = sd_sum_tmp.len() as i32;
                    sum_thresh = percentile_float(nx_temp, &mut sd_sum_tmp, count);
                    printf!(
                        "%d patches are below criterion for dropping; dropping only %d below %.3f\n",
                        CArg::Int(num_drop as i64),
                        CArg::Int(nx_temp as i64),
                        CArg::Dbl(sum_thresh as f64)
                    );
                    num_drop = 0;
                    for ip in 0..self.m_num_patches as usize {
                        if self.m_tmp_array[ip] != 0. && sd_sum_vec[ip] > sum_thresh {
                            self.m_tmp_array[ip] = 0.;
                        }
                        if self.m_tmp_array[ip] != 0. {
                            num_drop += 1;
                        }
                    }
                }
                let _ = num_drop;

                for ip in 0..self.m_num_patches as usize {
                    if self.m_tmp_array[ip] != 0. {
                        printf!(
                            "Dropping patch %d at %.0f %.0f; it reached only %.3f of the summed SD mode, not %.3f\n",
                            CArg::Int(ip as i64 + 1),
                            CArg::Dbl(self.m_patch_cen_x[ip] as f64),
                            CArg::Dbl(self.m_patch_cen_y[ip] as f64),
                            CArg::Dbl((sd_sum_vec[ip] / sum_mode) as f64),
                            CArg::Dbl(min_mode_frac as f64)
                        );
                    } else if expand_vec[ip] > 1. {
                        printf!(
                            "Expanded patch %d at %.0f %.0f to %d x %d (%.2f of criterion)",
                            CArg::Int(ip as i64 + 1),
                            CArg::Dbl(self.m_patch_cen_x[ip] as f64),
                            CArg::Dbl(self.m_patch_cen_y[ip] as f64),
                            CArg::Int(self.m_patch_nx[ip] as i64),
                            CArg::Int(self.m_patch_ny[ip] as i64),
                            CArg::Dbl((sd_sum_vec[ip] / summed_sd_crit) as f64)
                        );
                        if ((cen_xsave[ip] - self.m_patch_cen_x[ip]) as f64).abs() > 0.1
                            || ((cen_ysave[ip] - self.m_patch_cen_y[ip]) as f64).abs() > 0.1
                        {
                            printf!(
                                "; moved from %.0f %.0f",
                                CArg::Dbl(cen_xsave[ip] as f64),
                                CArg::Dbl(cen_ysave[ip] as f64)
                            );
                        }
                        printf!("\n");

                        // Set Use values to largest patch for memory allocation
                        self.m_nx_use = if self.m_nx_use > self.m_patch_nx[ip] {
                            self.m_nx_use
                        } else {
                            self.m_patch_nx[ip]
                        };
                        self.m_ny_use = if self.m_ny_use > self.m_patch_ny[ip] {
                            self.m_ny_use
                        } else {
                            self.m_patch_ny[ip]
                        };
                    }
                }

                // Copy retained patches down
                ind = 0;
                for ip in 0..self.m_num_patches as usize {
                    if self.m_tmp_array[ip] == 0. {
                        let d = ind as usize;
                        expand_vec[d] = expand_vec[ip];
                        self.m_patch_cen_x[d] = self.m_patch_cen_x[ip];
                        self.m_patch_cen_y[d] = self.m_patch_cen_y[ip];
                        self.m_patch_nx[d] = self.m_patch_nx[ip];
                        self.m_patch_ny[d] = self.m_patch_ny[ip];
                        ind += 1;
                    }
                }
                self.m_num_patches = ind;
                if self.m_nbinning == 0 {
                    self.m_nbinning = 1;
                }
            }

            // Start the warping file
            if if_find_warp != 0 {
                delta = iiu_ret_delta(1);
                if add_to_warps != 0 {
                    let (mut wnx, mut wny, mut wnz, mut wbin, mut wver, mut wflags) =
                        (0, 0, 0, 0, 0, 0);
                    ierr = read_warp_file(
                        &xf_file_out,
                        &mut wnx,
                        &mut wny,
                        &mut wnz,
                        &mut wbin,
                        &mut self.m_xpeak_tmp,
                        &mut wver,
                        &mut wflags,
                    );
                    if ierr < 0 {
                        exit_error_fmt!(
                            "Reading output file as existing warp file (error %d)",
                            CArg::Int(ierr as i64)
                        );
                    }
                    if wnx != self.m_nx
                        || wny != self.m_ny
                        || ((self.m_xpeak_tmp - delta[0]).abs() as f64) > 1.0e-4 * delta[0] as f64
                    {
                        exit_error(
                            b"Existing warp file does not match in X or Y image size or pixel size",
                        );
                    }
                    if wbin != 1 || wflags != 3 {
                        exit_error(b"Binning entry or flags in existing warp file are invalid");
                    }
                } else {
                    ierr = new_warp_file(self.m_nx, self.m_ny, 1, delta[0], 3);
                    if ierr < 0 {
                        exit_error(b"Failure to allocate new warp structure in library");
                    }
                }
            }
        } else if self.m_num_bound > 0 {
            //
            // For ordinary correlation with boundary model, adjust ixst etc
            if bound_xmin as f64 - 2. > ix_start as f64 {
                ix_start = (bound_xmin as f64 - 2.) as i32;
            }
            if (bound_xmax as f64 + 2.) < ix_end as f64 {
                ix_end = (bound_xmax as f64 + 2.).ceil() as i32;
            }
            if bound_ymin as f64 - 2. > iy_start as f64 {
                iy_start = (bound_ymin as f64 - 2.) as i32;
            }
            if (bound_ymax as f64 + 2.) < iy_end as f64 {
                iy_end = (bound_ymax as f64 + 2.).ceil() as i32;
            }
            self.m_nx_use = ix_end + 1 - ix_start;
            self.m_ny_use = iy_end + 1 - iy_start;
            if self.m_nx_use < 24 || self.m_ny_use < 24 {
                exit_error(b"Region inside boundary is too small");
            }
            self.m_xtfs_bound = vec![0.; num_points as usize];
            self.m_ytfs_bound = vec![0.; num_points as usize];
            printf!(
                "The area loaded will be X: %d  %d  Y: %d  %d\n",
                CArg::Int(ix_start as i64),
                CArg::Int(ix_end as i64),
                CArg::Int(iy_start as i64),
                CArg::Int(iy_end as i64)
            );
        }

        //
        // Set up one patch if no tracking
        // Also set up pad fraction: and if border was entered for that, compute padFrac
        // that will give this border for "use" size or original patch size
        // Do the same for taper fraction
        ierr = pip_get_two_integers(b"PadsInXandY", &mut nx_border, &mut ny_border);
        ix = pip_get_two_integers(b"TapersInXandY", &mut self.m_nx_taper, &mut self.m_ny_taper);
        if self.m_tracking {
            printf!(
                "%d patches will be tracked  [TXC1]\n",
                CArg::Int(self.m_num_patches as i64)
            );
            if ierr == 0 {
                pad_frac = (0.5
                    * (nx_border as f32 / self.m_nx_patch as f32
                        + ny_border as f32 / self.m_ny_patch as f32) as f64)
                    as f32;
            } else {
                pad_frac = 0.05;
            }
            if ix == 0 {
                taper_frac = (0.5
                    * (self.m_nx_taper as f32 / self.m_nx_patch as f32
                        + self.m_ny_taper as f32 / self.m_ny_patch as f32)
                        as f64) as f32;
            } else {
                taper_frac = 0.1;
            }
        } else {
            if if_find_warp != 0 {
                exit_error(b"You must specify a patch size to find warp transforms");
            }
            self.m_num_patches = 1;
            self.m_patch_cen_x
                .push(((ix_end + 1 + ix_start) as f64 / 2.) as f32);
            self.m_patch_cen_y
                .push(((iy_end + 1 + iy_start) as f64 / 2.) as f32);
            self.m_patch_nx.push(self.m_nx_patch);
            self.m_patch_ny.push(self.m_ny_patch);
            if ierr == 0 {
                pad_frac = (0.5
                    * (nx_border as f32 / self.m_nx_use as f32
                        + ny_border as f32 / self.m_ny_use as f32) as f64)
                    as f32;
            } else {
                pad_frac = 0.1;
            }
            if ix == 0 {
                taper_frac = (0.5
                    * (self.m_nx_taper as f32 / self.m_nx_use as f32
                        + self.m_ny_taper as f32 / self.m_ny_use as f32)
                        as f64) as f32;
            } else {
                taper_frac = 0.1;
            }
        }
        //
        // OK, back to main image operations
        // determine padding - needed here to set binning and unbinned correlations
        //
        nx_border = 5.max(b3dnint!(pad_frac * self.m_nx_use as f32));
        ny_border = 5.max(b3dnint!(pad_frac * self.m_ny_use as f32));
        //
        // get a binning based on the padded size so that large padding is
        // possible
        //
        self.m_nice_limit = nice_fft_limit();
        if self.m_use_gpu >= 0 {
            self.m_nice_limit = nice_gpu_limit;
        }
        if self.m_nbinning == 0 {
            self.m_nbinning = ((self.m_nx_use + 2 * nx_border).max(self.m_ny_use + 2 * ny_border)
                + max_bin_size
                - 1)
                / max_bin_size;
            //
            // If the binning is bigger than 4, find the minimum binning needed to keep the
            // used image within advisable bounds, up to a maximum binning, then stick with that
            if self.m_nbinning > 4 {
                self.m_nbinning = 0;
                let mut i = 4;
                while i <= max_auto_binning && self.m_nbinning == 0 {
                    if (nice_frame((self.m_nx_use + 2 * nx_border) / i, 2, self.m_nice_limit) + 2)
                        * nice_frame((self.m_ny_use + 2 * ny_border) / i, 2, self.m_nice_limit)
                        < limited_bin_size
                    {
                        self.m_nbinning = i;
                    }
                    i += 1;
                }
                if self.m_nbinning == 0 {
                    self.m_nbinning = max_auto_binning;
                }
            }
        }

        self.set_binned_sizes(pad_frac, taper_frac, &mut ix_cen_end, &mut iy_cen_end);

        // If using GPU, now determine number of cached full views and plans can be kept
        // with the available memory
        if self.m_use_gpu >= 0 {
            // Take fraction of total memory and divide by 4 to work with floats
            // The base amount is 2 full images plus 2-5 patch arrays plus 1 FFT plan pairs
            avail_gpu_mem = (gpu_mem_frac as f64 * gpu_memory as f64 / 4.) as f32;
            base_mem = (2 * self.m_nx * self.m_ny
                + self.m_nx_pad
                    * self.m_ny_pad
                    * (4 + (if num_iter > 0 { 2 } else { 0 })
                        + (if eval_ccc != 0 { 1 } else { 0 })
                        + (if self.m_limiting_shift != 0 { 1 } else { 0 })))
                as f32;
            ierr = 0;
            if base_mem <= avail_gpu_mem {
                // Allow one extra plan pair per two full images IF patches are dynamic
                ind = ((avail_gpu_mem - base_mem)
                    / (self.m_nx * self.m_ny
                        + if sd_bin_list.is_some() {
                            self.m_nx_pad * self.m_ny_pad
                        } else {
                            0
                        }) as f32) as i32;
                self.m_cache_size = 2 + max_track_gap.min(ind);
                base_mem += ((self.m_cache_size - 2) * self.m_nx * self.m_ny) as f32;
                let mgp = 1 + if sd_bin_list.is_some() {
                    ((avail_gpu_mem - base_mem) / (2 * self.m_nx_pad + self.m_ny_pad) as f32) as i32
                } else {
                    0
                };
                max_gpu_plans = mgp.min(MAX_PLAN_ARRAYS);

                // If there are no sigmas, it will think there is no filter, so set a mild one
                // past Nyquist
                if sigma1 == 0. && sigma2 == 0. {
                    radius1 = 0.;
                    radius2 = 0.5;
                    sigma2 = 0.05;
                }
                xcorr_set_ctf(
                    sigma1,
                    sigma2,
                    radius1,
                    radius2,
                    &mut self.m_ctfp,
                    5000,
                    5000,
                    &mut self.m_delta_ctf,
                );
                if eval_ccc != 0 {
                    for c in self.m_ctfp.iter_mut().take(8192) {
                        *c = (*c as f64).sqrt() as f32;
                    }
                }
                ierr = self.m_txc_gpu.as_mut().unwrap().initialize(
                    self.m_nx,
                    self.m_ny,
                    (self.m_nx_use / self.m_nbinning) * (self.m_ny_use / self.m_nbinning),
                    self.m_nx_pad * self.m_ny_pad,
                    max_gpu_plans,
                    self.m_cache_size,
                    &self.m_ctfp,
                    8192,
                    self.m_delta_ctf,
                    self.m_nbinning,
                    eval_ccc,
                    num_iter - 1,
                    self.m_limiting_shift != 0,
                );
            }
            if base_mem > avail_gpu_mem || ierr != 0 {
                ix = if if_gpu_by_env != 0 {
                    act_gpu_fail_environ
                } else {
                    act_gpu_fail_option
                };
                if ix > 1 {
                    exit_error(if ierr != 0 {
                        &b"Failure in initial call to GPU"[..]
                    } else {
                        &b"Insufficient memory to use GPU"[..]
                    });
                }
                printf!(
                    "%s%s; falling back to GPU\n",
                    CArg::Str(if ierr != 0 {
                        "Failure in initial call to GPU"
                    } else {
                        "Insufficient memory to use GPU"
                    }),
                    CArg::Str(if ix != 0 { "MESSAGE: Tiltxcorr - " } else { "" })
                );
                self.m_use_gpu = -1;
                self.m_nice_limit = nice_fft_limit();
            }
        }

        // Get image arrays (based on largest size for expanded patches)
        self.allocate_main_arrays(reuse_prev_arrays);
        if self.m_tracking {
            self.allocate_full_cache(2, self.m_cache_size);
        }
        iz = 0;
        let _ = iz;

        printf!(
            "\n%s is %d; %spadded, reduced size is %d by %d\n",
            CArg::Str(if self.m_i_anti_filt_type > 0 {
                "Reduction"
            } else {
                "Binning"
            }),
            CArg::Int(self.m_nbinning as i64),
            CArg::Str(if max_patch_expand > 0. {
                "maximum "
            } else {
                ""
            }),
            CArg::Int(self.m_nx_pad as i64),
            CArg::Int(self.m_ny_pad as i64)
        );
        //
        // Now that padded size exists, get the filter ctf.
        //
        self.set_main_ctf(sigma1, sigma2, radius1, radius2, eval_ccc);

        // Save unbinned shift limit and convert to binned value
        self.m_limit_ub_shift_x = self.m_limit_shift_x;
        self.m_limit_ub_shift_y = self.m_limit_shift_y;
        self.m_limit_shift_x = (self.m_limit_shift_x + self.m_nbinning / 2) / self.m_nbinning;
        self.m_limit_shift_y = (self.m_limit_shift_y + self.m_nbinning / 2) / self.m_nbinning;
        if self.m_limit_shift_x <= 0 || self.m_limit_shift_y <= 0 {
            exit_error(
                b"Shift limits must be positive and at least 1 pixel when divided by binning",
            );
        }
        //
        // If excluding central peak, set up parameters for that
        if self.m_if_exclude > 0 {
            self.m_nx_ub_pad = nice_frame(self.m_nx_use + 2 * nx_border, 2, self.m_nice_limit);
            self.m_ny_ub_pad = nice_frame(self.m_ny_use + 2 * ny_border, 2, self.m_nice_limit);
            xcorr_set_ctf(
                sigma1 / self.m_nbinning as f32,
                sigma2 / self.m_nbinning as f32,
                radius1 / self.m_nbinning as f32,
                0.75,
                &mut self.m_ctf_ub,
                self.m_nx_ub_pad,
                self.m_ny_ub_pad,
                &mut self.m_delta_ub_ctf,
            );
            if self.m_nbinning > 1 {
                let kk = (self.m_ny_ub_pad * (self.m_nx_ub_pad + 2) + 16) as usize;
                self.m_ub_array = vec![0.; kk];
                self.m_ub_brray = vec![0.; kk];
            }
        }
        self.m_cos_rot_angle = cosd!(-self.m_rot_angle);
        self.m_sin_rot_angle = sind!(-self.m_rot_angle);
        //
        // Get max tilt angle and check for appropriateness of cosine stretch
        self.m_use_max = 0.;
        for ivw in self.m_iz_start..=self.m_iz_end {
            let a = self.m_tilt[(ivw - 1) as usize].abs();
            self.m_use_max = if self.m_use_max > a {
                self.m_use_max
            } else {
                a
            };
        }
        if if_no_stretch == 0 && self.m_use_max > cos_str_max_tilt {
            if pip_get_float(b"FirstTiltAngle", &mut self.m_xpeak) == 0
                && pip_get_float(b"TiltIncrement", &mut self.m_ypeak) == 0
            {
                //
                // Check if they screwed up the increment and give a better message
                self.angles_pass_through(
                    self.m_xpeak,
                    self.m_ypeak,
                    nz,
                    &mut input_pass0,
                    &mut input_pass90,
                    &mut input_pass_min90,
                );
                self.angles_pass_through(
                    self.m_xpeak,
                    -self.m_ypeak,
                    nz,
                    &mut inv_pass0,
                    &mut inv_pass90,
                    &mut inv_pass_min90,
                );
                if inv_pass0
                    && !inv_pass90
                    && !inv_pass_min90
                    && ((input_pass90 && !input_pass0 && !input_pass_min90)
                        || (input_pass_min90 && !input_pass0 && !input_pass90))
                {
                    exit_error(
                        b"Tilt angles pass though 90 or -90 degrees (too high for cosine stretching) and not through 0; does the starting angle or increment have the wrong sign?",
                    );
                }
            }
            exit_error(b"Maximum tilt angle is too high to use cosine stretching");
        }
        //
        for kk in 0..nzu {
            xf_unit(&mut f[6 * kk..], 1.0, 2);
        }
        //
        if self.m_tracking {
            self.m_n_fill_taper = b3d_i_min(&[
                self.m_nx_patch / (4 * self.m_nbinning),
                self.m_ny_patch / (4 * self.m_nbinning),
                100,
                10.max(
                    b3dnint!(
                        (fill_taper_frac * (self.m_nx_patch + self.m_ny_patch) as f32) as f64 / 2.
                    ) / self.m_nbinning,
                ),
            ]);
            let nmod = (self.m_num_patches_all * self.m_iz_end) as usize;
            xmodel = vec![-1.0e10; nmod];
            ymodel = vec![-1.0e10; nmod];
            self.m_xmat = vec![0.; (self.m_mat_cols * self.m_num_patches_all) as usize];
            // `mMatCols` rather than `mPairCols` columns; see the module comment.
            self.m_pair_mat =
                vec![0.; (self.m_mat_cols.max(self.m_pair_cols) * self.m_num_patches_all) as usize];
            if len_contour <= 0 {
                len_contour = self.m_iz_end + 1 - self.m_iz_start;
            }
            num_cont = (self.m_iz_end - self.m_iz_start) / (len_contour - min_cont_overlap) + 1;
            if num_cont > self.m_fm.max_obj_num {
                exit_error(b"Too many contours for model arrays");
            }
            if num_cont * len_contour > self.m_fm.max_pt {
                exit_error(b"Too many total points for model arrays");
            }
        }
        //
        if im_file_out.is_some() {
            self.m_nz_out = self.m_iz_end - self.m_iz_start;
            if if_im_out != 0 {
                self.m_nz_out *= 3;
            }
            iiu_alt_size_samp_cell(3, self.m_nx_pad, self.m_ny_pad, self.m_nz_out);
            self.m_dmean_sum = 0.;
            self.m_dmax = -1.0e10;
            self.m_dmin = 1.0e10;
            self.m_nz_out = 0;
        }
        //
        // Report axis offset if leaving axis at box
        x_box_ofs = ((ix_end + 1 + ix_start - self.m_nx) as f64 / 2.) as f32;
        y_box_offset = ((iy_end + 1 + iy_start - self.m_ny) as f64 / 2.) as f32;
        if if_leave_axis != 0 {
            printf!(
                " The tilt axis is being left at a shift of %8.1f pixels from center\n",
                CArg::Dbl((x_box_ofs * self.m_cos_phi + y_box_offset * self.m_sin_phi) as f64)
            );
        }

        //
        // set up for one forward loop through data - modified by case below
        //
        let mut num_loops_v = 1;
        loop_dir = 1;
        //
        // set up for first or only loop
        //
        let mut start_low_for_track = 0;
        if use_ref_file {
            iv_start = self.m_iz_start;
            iv_end = self.m_iz_end;
            ref_tilt = 0.;
        } else if self.m_min_tilt >= self.m_iz_end {
            iv_start = self.m_iz_end - 1;
            iv_end = self.m_iz_start;
            loop_dir = -1;
        } else if self.m_min_tilt < self.m_iz_end && self.m_min_tilt > self.m_iz_start {
            iv_start = self.m_min_tilt + 1;
            iv_end = self.m_iz_end;
            num_loops_v = 2;
        } else {
            iv_start = self.m_iz_start + 1;
            iv_end = self.m_iz_end;
        }
        num_loops = num_loops_v;

        if num_loops == 2 && self.m_tracking && if_find_warp == 0 && iv_start - 2 > self.m_iz_start
        {
            iv_start -= 1;
            start_low_for_track = 1;
        }

        let nall = self.m_num_patches_all as usize;
        // `xmodel[ipatch + mNumPatchesAll * (iview - 1)]`, formed as the source
        // forms it; a negative view index is a C out-of-bounds read.
        let mind = |ip: i32, ivw: i32| -> usize {
            (ip as isize + nall as isize * (ivw as isize - 1)) as usize
        };

        /*
         * Do the loops
         */
        for iloop in 1..=num_loops {
            for i in 0..(self.m_nx_use_bin * self.m_ny_use_bin) as usize {
                self.m_sum_array[i] = 0.;
            }
            self.m_unstretch_dx = 0.;
            self.m_unstretch_dy = 0.;
            self.m_xpeak = 0.;
            self.m_ypeak = 0.;
            cum_xshift = 0.;
            cum_yshift = 0.;
            last_not_skipped = self.m_min_tilt - start_low_for_track;
            iv_first_not_skipped = -1;
            xf_unit(&mut self.m_f_unit, 1.0, 2);

            let mut iview = iv_start;
            while loop_dir * (iv_end - iview) >= 0 {
                iv_cur = iview;
                iv_ref = iview - loop_dir;

                // Look up Z of current view and dump anything in cache past maximum gap
                for i in 0..nz {
                    if iz_pc_list[i as usize] + 1 == iv_cur {
                        for indc in 0..self.m_cache_size as usize {
                            if loop_dir * (i - self.m_iz_loaded[indc]) >= max_track_gap + 2 {
                                self.m_iz_loaded[indc] = -1;
                            }
                        }
                    }
                }
                //
                // Test for skipping of this view, or of skipping of previous view when going
                // backwards and breaking
                iv_skip = iview;
                if self.m_breaking && loop_dir < 0 {
                    iv_skip = iview + 1;
                }
                //
                // If view is on skip/break list, copy the previous cumulative transform
                if number_in_list(iv_skip, Some(&self.m_list_skip), self.m_num_skip, 0) != 0 {
                    if use_ref_file {
                        xf_unit(&mut f[6 * (iv_cur - 1) as usize..], 1.0, 2);
                    } else {
                        f[(6 * (iv_cur - 1) + 4) as usize] = f[(6 * (iv_ref - 1) + 4) as usize];
                        f[(6 * (iv_cur - 1) + 5) as usize] = f[(6 * (iv_ref - 1) + 5) as usize];
                    }
                    iview += loop_dir;
                    continue;
                }
                if iv_first_not_skipped < 0 {
                    iv_first_not_skipped = iview;
                }
                //
                // Align to last one not skipped unless breaking alignment, then revise last
                if self.m_num_skip > 0 && !self.m_breaking {
                    iv_ref = last_not_skipped;
                }
                last_not_skipped = iview;
                cos_view = cosd!(self.m_tilt[(iview - 1) as usize]);
                //
                // get the stretch - if its less than 1., invert everything
                // unless doing cumulative or tracking, where it has to do in order
                // DNM 9/24/09: It seems like all kinds of things may not work if this
                // inversion ever happens...  9/23/11: basic correlation works but not tracking
                idir = 1;
                self.m_stretch = 1.;
                if if_no_stretch == 0 && cos_view.abs() as f64 > 0.01 {
                    if use_ref_file {
                        self.m_stretch = (1. / cos_view as f64) as f32;
                    } else {
                        self.m_stretch = cosd!(self.m_tilt[(iv_ref - 1) as usize]) / cos_view;
                    }
                    if self.m_if_cumulate != 0 && self.m_if_abs_stretch != 0 {
                        self.m_stretch =
                            cosd!(self.m_tilt[(self.m_min_tilt - 1) as usize]) / cos_view;
                    }
                }
                if (self.m_stretch as f64) < 1. && self.m_if_cumulate == 0 && !self.m_tracking {
                    idir = -1;
                    self.m_stretch = (1. / self.m_stretch as f64) as f32;
                    iz_tmp = iv_cur;
                    iv_cur = iv_ref;
                    iv_ref = iz_tmp;
                    cos_view = cosd!(self.m_tilt[(iv_cur - 1) as usize]);
                }
                if !use_ref_file {
                    ref_tilt = self.m_tilt[(iv_ref - 1) as usize];
                }
                if self.m_verbose != 0 {
                    print_vals!(
                        "idir" => cout_i(idir),
                        "mStretch" => cout_g(self.m_stretch as f64),
                        "ivRef" => cout_i(iv_ref),
                        "ivCur" => cout_i(iv_cur)
                    );
                }
                //
                // If doing warp and there is more than one boundary contour or the contour
                // needs to be transformed, get contour(s) from nearest Z to this and get new
                // set of patch centers
                if if_find_warp != 0
                    && (num_bound_all > 1 || (num_bound_all > 0 && raw_aligned_pair))
                {
                    //
                    // Find nearest view with boundaries
                    ind = num_all_views + 10;
                    for ivw in 1..=num_all_views {
                        if if_bound_on_view[(ivw - 1) as usize] > 0
                            && (iv_cur + iv_pair_offset - ivw).abs() < ind
                        {
                            ind = (iv_cur + iv_pair_offset - ivw).abs();
                            iv_bound = ivw;
                        }
                    }
                    //
                    // Redo the indices and repack the points into boundary arrays,
                    // transforming them if needed
                    self.m_num_bound = 0;
                    num_points = 0;
                    for ixb in 0..num_bound_all as usize {
                        iobj = iobj_bound[ixb];
                        let io = (iobj - 1) as usize;
                        ipnt = self.m_fm.object[self.m_fm.ibase_obj[io] as usize].abs() - 1;
                        if b3dnint!(self.m_fm.p_coord[ipnt as usize][2]) + 1 == iv_bound {
                            self.m_ind_bound[self.m_num_bound as usize] = num_points;
                            self.m_num_in_bound[self.m_num_bound as usize] =
                                self.m_fm.npt_in_obj[io];
                            self.m_num_bound += 1;
                            for ipt in 0..self.m_fm.npt_in_obj[io] {
                                ipnt = self.m_fm.object[(ipt + self.m_fm.ibase_obj[io]) as usize]
                                    .abs()
                                    - 1;
                                let np = num_points as usize;
                                self.m_xbound[np] = self.m_fm.p_coord[ipnt as usize][0];
                                self.m_ybound[np] = self.m_fm.p_coord[ipnt as usize][1];
                                if raw_aligned_pair {
                                    let (xb, yb) = xf_apply(
                                        &f_preali[(6 * (iv_cur + iv_pair_offset - 1)) as usize..],
                                        self.m_xcen as f32,
                                        self.m_ycen as f32,
                                        self.m_fm.p_coord[ipnt as usize][0],
                                        self.m_fm.p_coord[ipnt as usize][1],
                                        2,
                                    );
                                    self.m_xbound[np] = xb;
                                    self.m_ybound[np] = yb;
                                }
                                num_points += 1;
                            }
                        }
                    }
                    //
                    // Get the patch centers based on the boundaries
                    self.make_patch_list_inside_boundary();
                }
                //
                // Get inverse of prexf transform when doing pairs, for evaluating taper
                if raw_aligned_pair {
                    xf_invert(
                        &f_preali[(6 * (iv_cur + iv_pair_offset - 1)) as usize..],
                        &mut prexf_inv,
                        2,
                    );
                }

                if (iloop != 1 || iv_cur != iv_start) && max_tilted_expand > 1. {
                    let mut load_ind = 0;
                    self.get_full_array_and_lines(1, self.m_iz_cur, &mut load_ind);
                    full_ind = load_ind as usize;

                    // get limits for the SD map and get the map and its stats
                    self.m_nx_patch = nx_patch_orig;
                    self.m_nx_use = nx_patch_orig;
                    self.m_ny_patch = ny_patch_orig;
                    self.m_ny_use = ny_patch_orig;
                    self.set_sd_map_limits(min_expand_border, iv_cur - 1);

                    self.get_patch_sd_stats(
                        sd_bin,
                        ibin as i32,
                        full_ind,
                        &mut sd_num_pix[ibin],
                        &mut sd_means[ibin],
                        &mut sd_xoff,
                        &mut sd_yoff,
                        &mut min_sd[ibin],
                        &mut pct1_sd[ibin],
                    );
                    avg_sd(
                        &sd_means[ibin],
                        self.m_num_patches,
                        &mut mean_avg,
                        &mut mean_sd,
                        &mut temp,
                    );
                    if self.m_verbose != 0 {
                        print_vals!("meanAvg" => cout_g(mean_avg as f64));
                    }

                    // Try to retain the same summed SD crit but with the new pct1SD base
                    nx_sd = (self.m_sd_xend + 1 - self.m_sd_xstart) / sd_bin;
                    ny_sd = (self.m_sd_yend + 1 - self.m_sd_ystart) / sd_bin;
                }

                /*
                 * LOOP ON THE PATCHES
                 */
                iv_ref_base = iv_ref;
                num_control = 0;
                if self.m_tracking && if_find_warp == 0 {
                    self.m_model_xvecs = vec![Vec::new(); nall];
                    self.m_model_yvecs = vec![Vec::new(); nall];
                    self.m_peak_vecs = vec![Vec::new(); nall];
                }
                ipatch = 0;
                while ipatch < self.m_num_patches {
                    let ipu = ipatch as usize;
                    iv_ref = iv_ref_base;

                    // For dynamic patch sizes, set the controlling sizes for this patch and
                    // set up CTF again if size has changed
                    if max_patch_expand > 0. {
                        self.m_nx_patch = self.m_patch_nx[ipu];
                        self.m_nx_use = self.m_nx_patch;
                        self.m_ny_patch = self.m_patch_ny[ipu];
                        self.m_ny_use = self.m_ny_patch;
                        ix = self.m_nx_pad;
                        iy = self.m_ny_pad;
                        self.set_binned_sizes(
                            pad_frac,
                            taper_frac,
                            &mut ix_cen_end,
                            &mut iy_cen_end,
                        );
                        if self.m_nx_pad != ix || self.m_ny_pad != iy {
                            self.set_main_ctf(sigma1, sigma2, radius1, radius2, eval_ccc);
                        }
                    }
                    //
                    // Get box offset, first by tilt-foreshortened center position
                    // rounded to nearest integer
                    cenx = self.m_patch_cen_x[ipu] - self.m_xcen as f32;
                    ceny = self.m_patch_cen_y[ipu] - self.m_ycen as f32;
                    self.adjust_coord(0., ref_tilt, cenx, ceny, &mut x0, &mut y0, false);
                    self.m_ix_box_ref =
                        (-self.m_ix_cen_start).max((self.m_nx - 1 - ix_cen_end).min(b3dnint!(x0)));
                    self.m_iy_box_ref =
                        (-self.m_iy_cen_start).max((self.m_ny - 1 - iy_cen_end).min(b3dnint!(y0)));
                    base_tilt = 0.;
                    if iview == iv_first_not_skipped {
                        ix_box_start = self.m_ix_box_ref;
                        iy_box_start = self.m_iy_box_ref;
                    }
                    //
                    // If tracking, initialize model position on first time or get  reference
                    // box from model point
                    if self.m_tracking {
                        if (iloop == 1 && iview == iv_first_not_skipped) || if_find_warp != 0 {
                            let indm = mind(ipatch, iv_ref);
                            xmodel[indm] = ((self.m_ix_box_ref + self.m_ix_cen_start) as f64
                                + self.m_nx_use as f64 / 2.)
                                as f32;
                            ymodel[indm] = ((self.m_iy_box_ref + self.m_iy_cen_start) as f64
                                + self.m_ny_use as f64 / 2.)
                                as f32;
                            if if_find_warp != 0 {
                                xmodel[mind(ipatch, iv_cur)] = xmodel[indm];
                                ymodel[mind(ipatch, iv_cur)] = ymodel[indm];
                            }
                        } else {
                            //
                            // Back up reference to last point that was tracked unless it is too
                            // far back
                            let _ = ImodFile::Stdout.flush();
                            while xmodel[mind(ipatch, iv_ref)] < -1.0e9
                                && (iv_ref - iv_cur).abs() <= max_track_gap
                            {
                                iv_ref -= loop_dir;
                            }
                            if xmodel[mind(ipatch, iv_ref)] < -1.0e9 {
                                if self.m_verbose != 0
                                    && (self.m_single_test_patch < 0
                                        || self.m_single_test_patch == ipatch)
                                {
                                    printf!(
                                        "Stopping tracking of patch # %d\n",
                                        CArg::Int(ipatch as i64 + 1)
                                    );
                                }
                                ipatch += 1;
                                continue;
                            }
                            if iv_ref != iv_ref_base {
                                if self.m_verbose != 0
                                    && (self.m_single_test_patch < 0
                                        || self.m_single_test_patch == ipatch)
                                {
                                    printf!(
                                        "Patch %d tracking to reference %d\n",
                                        CArg::Int(ipatch as i64 + 1),
                                        CArg::Int(iv_ref as i64)
                                    );
                                }
                                self.m_stretch = 1.;
                                if if_no_stretch == 0 && cos_view.abs() as f64 > 0.01 {
                                    self.m_stretch =
                                        cosd!(self.m_tilt[(iv_ref - 1) as usize]) / cos_view;
                                }
                            }
                            //
                            // Center the reference box on the model point
                            let indm = mind(ipatch, iv_ref);
                            x0 = (xmodel[indm] as f64
                                - ((self.m_ix_cen_start as f64) + self.m_nx_use as f64 / 2.))
                                as f32;
                            y0 = (ymodel[indm] as f64
                                - ((self.m_iy_cen_start as f64) + self.m_ny_use as f64 / 2.))
                                as f32;
                            base_tilt = self.m_tilt[(iv_ref - 1) as usize];

                            // INTERVENE HERE!
                            if max_tilted_expand > 1. {
                                cenx = x0;
                                ceny = y0;
                                self.adjust_coord(
                                    base_tilt,
                                    self.m_tilt[(iv_cur - 1) as usize],
                                    cenx,
                                    ceny,
                                    &mut x0,
                                    &mut y0,
                                    false,
                                );

                                let cxt = ((cenx + self.m_ix_cen_start as f32) as f64
                                    + self.m_nx_use as f64 / 2.)
                                    as f32;
                                let cyt = ((ceny + self.m_iy_cen_start as f32) as f64
                                    + self.m_ny_use as f64 / 2.)
                                    as f32;
                                let x0t = ((cenx + self.m_ix_cen_start as f32) as f64
                                    + self.m_nx_use as f64 / 2.)
                                    as f32;
                                let y0t = ((ceny + self.m_iy_cen_start as f32) as f64
                                    + self.m_ny_use as f64 / 2.)
                                    as f32;

                                // Start with the patch-specific size
                                nx_temp = self.m_nx_patch;
                                ny_temp = self.m_ny_patch;
                                expand_x = expand_vec[ipu];
                                expand_y = expand_x;
                                sd_sum = 0.;
                                too_big = false;
                                let mnx = self.m_nx;
                                let mny = self.m_ny;
                                let blk_x = |nxt: i32| -> bool {
                                    cxt + (nxt / 2) as f32 > (mnx - min_expand_border) as f32
                                        || cxt - ((nxt / 2) as f32) < min_expand_border as f32
                                        || x0t + (nxt / 2) as f32 > (mnx - min_expand_border) as f32
                                        || x0t - ((nxt / 2) as f32) < min_expand_border as f32
                                };
                                let blk_y = |nyt: i32| -> bool {
                                    cyt + (nyt / 2) as f32 > (mny - min_expand_border) as f32
                                        || cyt - ((nyt / 2) as f32) < min_expand_border as f32
                                        || y0t + (nyt / 2) as f32 > (mny - min_expand_border) as f32
                                        || y0t - ((nyt / 2) as f32) < min_expand_border as f32
                                };
                                while sd_sum < summed_sd_crit {
                                    // Compute the SD measure at the current size
                                    px_start = b3dnint!(x0 + self.m_xcen as f32) - nx_temp / 2;
                                    px_end = (px_start + nx_temp - 1).min(self.m_sd_xend);
                                    px_start = px_start.max(self.m_sd_xstart);
                                    py_start = b3dnint!(y0 + self.m_ycen as f32) - ny_temp / 2;
                                    py_end = (py_start + ny_temp - 1).min(self.m_sd_yend);
                                    py_start = py_start.max(self.m_sd_ystart);
                                    ix = (px_end + 1 - px_start) / sd_bin;
                                    iy = (py_end + 1 - py_start) / sd_bin;
                                    array_min_max_mean(
                                        &self.m_sd_arr,
                                        nx_sd,
                                        ny_sd,
                                        px_start / sd_bin + sd_xoff,
                                        px_end / sd_bin + sd_xoff,
                                        py_start / sd_bin + sd_yoff,
                                        py_end / sd_bin + sd_yoff,
                                        &mut self.m_dmin2,
                                        &mut self.m_dmax2,
                                        &mut sd_sum,
                                    );
                                    sd_sum = (sd_sum - pct1_sd[ibin]) * ix as f32 * iy as f32;
                                    if sd_sum >= summed_sd_crit {
                                        break;
                                    }

                                    // If not good enough, try to expand equally and see if X
                                    // or Y can't
                                    nx_temp = 2 * b3dnint!(
                                        (nx_patch_orig as f32 * (expand_x + expand_step)) as f64
                                            / 2.
                                    );
                                    ny_temp = 2 * b3dnint!(
                                        (ny_patch_orig as f32 * (expand_y + expand_step)) as f64
                                            / 2.
                                    );
                                    blocked_x = blk_x(nx_temp);
                                    blocked_y = blk_y(ny_temp);

                                    // If one is blocked, compute expansion on other axis and
                                    // evaluate block
                                    if blocked_x && !blocked_y {
                                        expand_y =
                                            (expand_y as f64 + 2. * expand_step as f64) as f32;
                                        ny_temp = 2 * b3dnint!(
                                            (ny_patch_orig as f32 * expand_y) as f64 / 2.
                                        );
                                        blocked_y = blk_y(ny_temp);
                                    } else if blocked_y && !blocked_x {
                                        expand_x =
                                            (expand_x as f64 + 2. * expand_step as f64) as f32;
                                        nx_temp = 2 * b3dnint!(
                                            (nx_patch_orig as f32 * expand_x) as f64 / 2.
                                        );
                                        blocked_x = blk_x(nx_temp);
                                    } else {
                                        // Otherwise expanding equally or not at all
                                        expand_x += expand_step;
                                        expand_y += expand_step;
                                    }

                                    // Now if both blocked or we have expanded too far, drop
                                    // the patch
                                    if (blocked_x && blocked_y)
                                        || ((expand_x * expand_y) as f64).sqrt()
                                            > max_tilted_expand as f64 + 0.25 * expand_step as f64
                                    {
                                        if sd_sum < expand_drop_frac * summed_sd_crit {
                                            too_big = true;
                                        }
                                        break;
                                    }

                                    // All set to iterate with new patch size
                                }

                                if too_big {
                                    printf!(
                                        "Skipping patch # %d; it cannot expand enough\n",
                                        CArg::Int(ipatch as i64 + 1)
                                    );
                                    ipatch += 1;
                                    continue;
                                }

                                // Keep the stored, base size, set the variables for new size
                                self.m_nx_patch = nx_temp;
                                self.m_nx_use = nx_temp;
                                self.m_ny_patch = ny_temp;
                                self.m_ny_use = ny_temp;
                                ix = self.m_nx_pad;
                                iy = self.m_ny_pad;
                                self.set_binned_sizes(
                                    pad_frac,
                                    taper_frac,
                                    &mut ix_cen_end,
                                    &mut iy_cen_end,
                                );
                                if self.m_nx_pad != ix || self.m_ny_pad != iy {
                                    self.set_main_ctf(sigma1, sigma2, radius1, radius2, eval_ccc);
                                }
                                self.allocate_main_arrays(reuse_prev_arrays);
                            }

                            self.m_ix_box_ref = (-self.m_ix_cen_start)
                                .max((self.m_nx - 1 - ix_cen_end).min(b3dnint!(x0)));
                            self.m_iy_box_ref = (-self.m_iy_cen_start)
                                .max((self.m_ny - 1 - iy_cen_end).min(b3dnint!(y0)));
                            cenx = self.m_ix_box_ref as f32;
                            ceny = self.m_iy_box_ref as f32;
                        }
                    }
                    //
                    // Now get box offset of current view by adjusting either the zero
                    // degree position (so this will match old behavior of program)
                    // or the box offset on reference view
                    self.adjust_coord(
                        base_tilt,
                        self.m_tilt[(iv_cur - 1) as usize],
                        cenx,
                        ceny,
                        &mut x0,
                        &mut y0,
                        false,
                    );
                    self.m_ix_box_cur =
                        (-self.m_ix_cen_start).max((self.m_nx - 1 - ix_cen_end).min(b3dnint!(x0)));
                    self.m_iy_box_cur =
                        (-self.m_iy_cen_start).max((self.m_ny - 1 - iy_cen_end).min(b3dnint!(y0)));
                    if self.m_verbose > 1
                        || (self.m_verbose != 0 && self.m_single_test_patch == ipatch)
                    {
                        printf!(
                            "Box offsets %d %d %d %d %d\n",
                            CArg::Int(ipatch as i64),
                            CArg::Int(self.m_ix_box_ref as i64),
                            CArg::Int(self.m_iy_box_ref as i64),
                            CArg::Int(self.m_ix_box_cur as i64),
                            CArg::Int(self.m_iy_box_cur as i64)
                        );
                    }
                    //
                    // Now that reference and stretch is fixed, get a limited cosine ratio to
                    // use below, get the transforms and look up the Z's in file
                    cos_max = {
                        let a = cos_view.abs();
                        let b = cosd!(ref_tilt).abs();
                        if a > b { a } else { b }
                    };
                    cos_max = if cos_max as f64 > 1.0e-6 {
                        cos_max
                    } else {
                        1.0e-6
                    };
                    cos_ratio = cos_view.abs() / cos_max;
                    rotmagstr_to_fs!(self.m_fs, 0., 1., self.m_stretch, self.m_rot_angle);
                    self.m_fs[4] = 0.;
                    self.m_fs[5] = 0.;
                    xf_invert(&self.m_fs, &mut fs_inv, 2);
                    for i in 0..nz {
                        if iz_pc_list[i as usize] + 1 == iv_ref {
                            self.m_iz_last = i;
                        }
                        if iz_pc_list[i as usize] + 1 == iv_cur {
                            self.m_iz_cur = i;
                        }
                    }
                    if use_ref_file {
                        self.m_iz_last = iz_in_ref_file;
                    }
                    //
                    // If tracking and prexg's are available, evaluate need for tapering and
                    // skip the patch if it has below criterion image data
                    self.m_taper_ref = false;
                    self.m_taper_cur = false;
                    if self.m_if_read_xfs != 0 {
                        if raw_aligned_pair {
                            self.m_taper_ref = false;
                            ref_view_out = false;
                            self.evaluate_pair_patch(
                                self.m_ix_cen_start + self.m_ix_box_cur,
                                self.m_iy_cen_start + self.m_iy_box_cur,
                                &prexf_inv,
                                &mut cur_view_out,
                            );
                        } else {
                            let mut g0 = 0;
                            let mut g1 = 0;
                            self.usable_patch_extent(
                                self.m_nx,
                                self.m_nx_unali,
                                self.m_nx_patch,
                                self.m_dx_preali[(iv_ref - 1) as usize],
                                self.m_ix_cen_start + self.m_ix_box_ref,
                                &mut ix,
                                Some(&mut g0),
                                Some(&mut g1),
                            );
                            self.m_good_xstart[0] = g0;
                            self.m_good_xend[0] = g1;
                            self.usable_patch_extent(
                                self.m_ny,
                                self.m_ny_unali,
                                self.m_ny_patch,
                                self.m_dy_preali[(iv_ref - 1) as usize],
                                self.m_iy_cen_start + self.m_iy_box_ref,
                                &mut iy,
                                Some(&mut g0),
                                Some(&mut g1),
                            );
                            self.m_good_ystart[0] = g0;
                            self.m_good_yend[0] = g1;
                            self.m_taper_ref = ix < self.m_nx_patch || iy < self.m_ny_patch;
                            ref_view_out = ix < 0
                                || iy < 0
                                || ((ix * iy) as f32)
                                    < self.m_crit_non_blank
                                        * self.m_nx_patch as f32
                                        * self.m_ny_patch as f32;
                            self.usable_patch_extent(
                                self.m_nx,
                                self.m_nx_unali,
                                self.m_nx_patch,
                                self.m_dx_preali[(iv_cur - 1) as usize],
                                self.m_ix_cen_start + self.m_ix_box_cur,
                                &mut ix,
                                Some(&mut g0),
                                Some(&mut g1),
                            );
                            self.m_good_xstart[1] = g0;
                            self.m_good_xend[1] = g1;
                            self.usable_patch_extent(
                                self.m_ny,
                                self.m_ny_unali,
                                self.m_ny_patch,
                                self.m_dy_preali[(iv_cur - 1) as usize],
                                self.m_iy_cen_start + self.m_iy_box_cur,
                                &mut iy,
                                Some(&mut g0),
                                Some(&mut g1),
                            );
                            self.m_good_ystart[1] = g0;
                            self.m_good_yend[1] = g1;
                            self.m_taper_cur = ix < self.m_nx_patch || iy < self.m_ny_patch;
                            cur_view_out = ix < 0
                                || iy < 0
                                || ((ix * iy) as f32)
                                    < self.m_crit_non_blank
                                        * self.m_nx_patch as f32
                                        * self.m_ny_patch as f32;
                        }
                        if cur_view_out {
                            if self.m_verbose != 0 {
                                printf!(
                                    "Skipping tracking of patch %d\n",
                                    CArg::Int(ipatch as i64 + 1)
                                );
                            }
                            if if_find_warp != 0 {
                                xmodel[mind(ipatch, iv_ref)] = -1.0e10;
                            }
                            ipatch += 1;
                            continue;
                        }

                        // If finding warp, skip if no reference view
                        if if_find_warp != 0 && ref_view_out {
                            if self.m_verbose != 0 {
                                printf!(
                                    "Skipping tracking of patch %d; no reference\n",
                                    CArg::Int(ipatch as i64 + 1)
                                );
                            }
                            xmodel[mind(ipatch, iv_ref)] = -1.0e10;
                            ipatch += 1;
                            continue;
                        }

                        if self.m_verbose != 0
                            && self.m_taper_ref
                            && (self.m_single_test_patch < 0 || self.m_single_test_patch == ipatch)
                        {
                            printf!(
                                "Tapering reference patch # %d\n",
                                CArg::Int(ipatch as i64 + 1)
                            );
                        }
                        if self.m_verbose != 0
                            && self.m_taper_cur
                            && (self.m_single_test_patch < 0 || self.m_single_test_patch == ipatch)
                        {
                            printf!(
                                "Tapering current patch # %d\n",
                                CArg::Int(ipatch as i64 + 1)
                            );
                        }
                    }
                    self.m_xpeak_frac = 0.;
                    self.m_ypeak_frac = 0.;
                    self.m_if_reuse_prev = 0;
                    //
                    // If correlating inside boundary contour, transform contours
                    if !self.m_tracking && self.m_num_bound > 0 {
                        for ixb in 0..self.m_num_bound as usize {
                            self.m_xtfs_bmin[ixb] = 1.0e10;
                            self.m_xtfs_bmax[ixb] = -1.0e10;
                            self.m_ytfs_bmin[ixb] = 1.0e10;
                            self.m_ytfs_bmax[ixb] = -1.0e10;
                            for j in 0..self.m_num_in_bound[ixb] {
                                let indb = (j + self.m_ind_bound[ixb]) as usize;
                                self.adjust_coord(
                                    0.,
                                    self.m_tilt[(iv_cur - 1) as usize],
                                    self.m_xbound[indb] - self.m_xcen as f32,
                                    self.m_ybound[indb] - self.m_ycen as f32,
                                    &mut x0,
                                    &mut y0,
                                    false,
                                );
                                self.m_xtfs_bound[indb] = (x0 + self.m_xcen as f32
                                    - (self.m_ix_cen_start + self.m_ix_box_cur) as f32)
                                    / self.m_nbinning as f32;
                                self.m_ytfs_bound[indb] = (y0 + self.m_ycen as f32
                                    - (self.m_iy_cen_start + self.m_iy_box_cur) as f32)
                                    / self.m_nbinning as f32;
                                let (xt, yt) = (self.m_xtfs_bound[indb], self.m_ytfs_bound[indb]);
                                self.m_xtfs_bmin[ixb] = if self.m_xtfs_bmin[ixb] < xt {
                                    self.m_xtfs_bmin[ixb]
                                } else {
                                    xt
                                };
                                self.m_ytfs_bmin[ixb] = if self.m_ytfs_bmin[ixb] < yt {
                                    self.m_ytfs_bmin[ixb]
                                } else {
                                    yt
                                };
                                self.m_xtfs_bmax[ixb] = if self.m_xtfs_bmax[ixb] > xt {
                                    self.m_xtfs_bmax[ixb]
                                } else {
                                    xt
                                };
                                self.m_ytfs_bmax[ixb] = if self.m_ytfs_bmax[ixb] > yt {
                                    self.m_ytfs_bmax[ixb]
                                } else {
                                    yt
                                };
                            }
                        }
                    }
                    //
                    self.m_if_ub_peak_is_sharp = -1;
                    //
                    // Is it time to search for the mag change?
                    mag_view = iv_cur.max(iv_ref);
                    self.m_searched_for_mag = search_mag != 0
                        && number_in_list(mag_view, Some(&list_mag_views), num_mag_views, 0) > 0;
                    if self.m_searched_for_mag {
                        // Swap the search limits if the reference is the mag view; we are
                        // finding the inverse of its mag change
                        if iv_ref == mag_view {
                            self.m_peak_val = (1. / search_min as f64) as f32;
                            search_min = (1. / search_max as f64) as f32;
                            search_max = self.m_peak_val;
                        }

                        // Set up the step size and initial mag
                        step_mag = {
                            let v = (search_max - search_min) as f64 / 3.;
                            (if 0.03 < v { 0.03 } else { v }) as f32
                        };
                        found_mag = search_min - step_mag;
                        num_cuts = -1;
                        if reuse_prev_arrays {
                            self.m_if_reuse_prev = -1;
                        }
                        if self.m_verbose != 0 {
                            printf!("Searching for best magnification change\n");
                        }
                        loop {
                            // Get the transformation incorporating the mag change, correlate
                            // with CCCs
                            rotmagstr_to_fs!(
                                self.m_fs,
                                0.,
                                found_mag,
                                self.m_stretch,
                                self.m_rot_angle
                            );
                            if self.m_nbinning > 1 {
                                self.correlate_and_find_peaks(true, false, 0, -1);
                            } else {
                                self.correlate_and_find_peaks(true, true, 0, -1);
                            }
                            if self.m_verbose != 0 {
                                printf!(
                                    "Mag = %f  CCC = %f\n",
                                    CArg::Dbl(found_mag as f64),
                                    CArg::Dbl(self.m_peak_val as f64)
                                );
                            }

                            // Find out the next step; check if out of range or if step size is
                            // small
                            let mut next = found_mag;
                            ierr = minimize1d(
                                found_mag,
                                -self.m_peak_val,
                                step_mag,
                                b3dnint!((search_max - search_min) / step_mag) + 2,
                                &mut num_cuts,
                                &mut brackets,
                                &mut next,
                            );
                            found_mag = next;
                            if ierr > 0 {
                                printf!(
                                    "Search for magnification change at view %d reached allowed limits\n  without finding a minimum - no mag change will be introduced\n",
                                    CArg::Int(mag_view as i64)
                                );
                                found_mag = 1.;
                                self.m_searched_for_mag = false;
                                break;
                            }
                            if step_mag as f64 / 2f64.powf(num_cuts as f64) < 0.00025 {
                                // Done: get the minimum
                                found_mag = brackets[1];
                                break;
                            }
                        }

                        // Revise the transform again and go on
                        rotmagstr_to_fs!(
                            self.m_fs,
                            0.,
                            found_mag,
                            self.m_stretch,
                            self.m_rot_angle
                        );
                    }

                    // Or, scan rotations if selected
                    ind_best_rot = 0;
                    if if_scan_rotation > 0 {
                        if num_rot_steps > 0 {
                            if reuse_prev_arrays && self.m_if_reuse_prev == 0 {
                                self.m_if_reuse_prev = -1;
                            }
                            self.m_iter = 0;
                            while self.m_iter < num_rot_steps {
                                angle = self.m_iter as f32 * scan_rot_interval - scan_rot_max;
                                rotmagstr_to_fs!(
                                    self.m_fs,
                                    angle,
                                    1.,
                                    self.m_stretch,
                                    self.m_rot_angle
                                );
                                if self.m_nbinning > 1 && self.m_if_exclude > 0 {
                                    self.correlate_and_find_peaks(eval_ccc != 0, false, 0, -1);
                                } else {
                                    self.correlate_and_find_peaks(eval_ccc != 0, true, 0, -1);
                                }
                                rot_scan_peaks[self.m_iter as usize] = self.m_peak_val;
                                if self.m_peak_val > rot_scan_peaks[ind_best_rot as usize] {
                                    ind_best_rot = self.m_iter;
                                }
                                self.m_iter += 1;
                            }
                            angle = ind_best_rot as f32 * scan_rot_interval - scan_rot_max;
                            if ind_best_rot > 0 && ind_best_rot <= num_rot_steps {
                                best_angle = angle;
                                angle = ((parabolic_fit_position(
                                    rot_scan_peaks[(ind_best_rot - 1) as usize],
                                    rot_scan_peaks[ind_best_rot as usize],
                                    rot_scan_peaks[(ind_best_rot + 1) as usize],
                                ) + ind_best_rot as f64)
                                    * scan_rot_interval as f64
                                    - scan_rot_max as f64)
                                    as f32;
                            }
                        } else {
                            angle = idir as f32 * scan_rot_max;
                        }
                        rotmagstr_to_fs!(self.m_fs, angle, 1., self.m_stretch, self.m_rot_angle);
                    }

                    // Now start iterations of regular correlation alignment
                    if reuse_prev_arrays && num_iter > 1 && self.m_if_reuse_prev == 0 {
                        self.m_if_reuse_prev = -1;
                    }
                    if self.m_use_gpu >= 0 {
                        if ipatch == self.m_single_test_patch {
                            print_vals!(
                                "mNxPatch" => cout_i(self.m_nx_patch),
                                "mNyPatch" => cout_i(self.m_ny_patch),
                                "mNxPad" => cout_i(self.m_nx_pad),
                                "mNyPad" => cout_i(self.m_ny_pad)
                            );
                        }
                        if self.m_txc_gpu.as_mut().unwrap().setup_for_patch(
                            self.m_nx_patch,
                            self.m_ny_patch,
                            self.m_nx_pad,
                            self.m_ny_pad,
                            self.m_nx_taper,
                            self.m_ny_taper,
                        ) != 0
                        {
                            exit_error(b"Setting up to do the current patch on GPU");
                        }
                    }
                    x_frac_to_add = 0.;
                    y_frac_to_add = 0.;
                    self.m_iter = 1;
                    while self.m_iter <= num_iter {
                        let do_im = if_im_out != 0
                            && (self.m_single_test_patch < 0 || self.m_single_test_patch == ipatch);
                        if self.m_nbinning > 1 && self.m_if_exclude > 0 {
                            self.correlate_and_find_peaks(
                                eval_ccc != 0,
                                false,
                                do_im as i32,
                                ipatch,
                            );
                        } else {
                            self.correlate_and_find_peaks(
                                eval_ccc != 0,
                                true,
                                do_im as i32,
                                ipatch,
                            );
                        }
                        x_frac_to_add = self.m_xpeak_frac;
                        y_frac_to_add = self.m_ypeak_frac;
                        xpeak_cum = self.m_xpeak_tmp + self.m_xpeak_frac;
                        self.m_xpeak_frac = xpeak_cum - b3dnint!(xpeak_cum) as f32;
                        ypeak_cum = self.m_ypeak_tmp + self.m_ypeak_frac;
                        self.m_ypeak_frac = ypeak_cum - b3dnint!(ypeak_cum) as f32;
                        if self.m_verbose != 0
                            && (self.m_single_test_patch < 0 || self.m_single_test_patch == ipatch)
                        {
                            printf!(
                                "%3d %7.2f %7.2f %14.6g %7.2f %7.2f %7.2f %7.2f\n",
                                CArg::Int(self.m_iter as i64),
                                CArg::Dbl(xpeak_cum as f64),
                                CArg::Dbl(ypeak_cum as f64),
                                CArg::Dbl(self.m_peak_val as f64),
                                CArg::Dbl(
                                    (self.m_xpeak_tmp - b3dnint!(self.m_xpeak_tmp) as f32) as f64
                                ),
                                CArg::Dbl(
                                    (self.m_ypeak_tmp - b3dnint!(self.m_ypeak_tmp) as f32) as f64
                                ),
                                CArg::Dbl(self.m_xpeak_frac as f64),
                                CArg::Dbl(self.m_ypeak_frac as f64)
                            );
                        }

                        // If accumulating and the correlation failed, assign it the last shift
                        if (self.m_peak_val as f64) < -1.0e29 && self.m_if_cumulate != 0 {
                            xpeak_cum = self.m_xpeak;
                            ypeak_cum = self.m_ypeak;
                        }
                        //
                        // Skip out and restore last peak if the peak strength is less
                        if self.m_iter > 1 && self.m_peak_val < peak_last {
                            xpeak_cum = xpeak_last;
                            ypeak_cum = y_peak_last;
                            self.m_peak_val = peak_last;
                            x_frac_to_add = x_frac_last;
                            // `tiltxcorr.cpp:2120` assigns `xFracToAdd` twice; the Y
                            // fraction is never restored (unobservable while
                            // `numPeakToProc` is 1).
                            x_frac_to_add = y_frac_last;
                            break;
                        }
                        xpeak_last = xpeak_cum;
                        y_peak_last = ypeak_cum;
                        peak_last = self.m_peak_val;
                        x_frac_last = x_frac_to_add;
                        y_frac_last = y_frac_to_add;
                        let _ = x_frac_last;
                        //
                        // Skip out if we are close to 0 interpolation
                        let dxf = (self.m_xpeak_tmp - b3dnint!(self.m_xpeak_tmp) as f32) as f64;
                        let dyf = (self.m_ypeak_tmp - b3dnint!(self.m_ypeak_tmp) as f32) as f64;
                        if (dxf.powf(2.) + dyf.powf(2.)).sqrt() < peak_frac_tol as f64 {
                            break;
                        }
                        self.m_iter += 1;
                    }
                    self.m_xpeak = xpeak_cum;
                    self.m_ypeak = ypeak_cum;
                    sec_xpeak = 0.;
                    sec_ypeak = 0.;
                    sec_peak_in_box = false;
                    if ny_sec_peak_box > 0 && self.m_num_xcorr_peaks > 1 {
                        let s1 = self.m_ind_peak_sort[1] as usize;
                        self.m_xpeak_tmp = self.m_xpeak_list[s1] - self.m_xpeak;
                        self.m_ypeak_tmp = self.m_ypeak_list[s1] - self.m_ypeak;
                        xrot = self.m_xpeak_tmp * self.m_cos_rot_angle
                            - self.m_ypeak_tmp * self.m_sin_rot_angle;
                        yrot = self.m_xpeak_tmp * self.m_sin_rot_angle
                            + self.m_ypeak_tmp * self.m_cos_rot_angle;
                        if xrot * self.m_nbinning as f32 <= nx_sec_peak_box as f32
                            && yrot * self.m_nbinning as f32 <= ny_sec_peak_box as f32
                        {
                            sec_xpeak = self.m_xpeak_list[s1];
                            sec_ypeak = self.m_ypeak_list[s1];
                            sec_peak_in_box = true;
                        }
                    }

                    num_peak_to_proc = 1;

                    // Process all the peaks for patch tracking
                    for ipeak in 0..num_peak_to_proc {
                        if ipeak != 0 {
                            let sp = self.m_ind_peak_sort[ipeak as usize] as usize;
                            self.m_xpeak = self.m_xpeak_list[sp] + x_frac_to_add;
                            self.m_ypeak = self.m_ypeak_list[sp] + y_frac_to_add;
                        }
                        //
                        // DNM 5/2/02: only destretch the shift if the current view was
                        // stretched.  Also, put the right sign out in the standard output
                        //
                        if idir > 0 {
                            let (ux, uy) = xf_apply(&fs_inv, 0., 0., self.m_xpeak, self.m_ypeak, 2);
                            self.m_unstretch_dx = ux;
                            self.m_unstretch_dy = uy;
                            let (sx, sy) = xf_apply(&fs_inv, 0., 0., sec_xpeak, sec_ypeak, 2);
                            unstretch_sec_dx = sx;
                            unstretch_sec_dy = sy;
                        } else {
                            self.m_unstretch_dx = self.m_xpeak;
                            self.m_unstretch_dy = self.m_ypeak;
                            unstretch_sec_dx = sec_xpeak;
                            unstretch_sec_dy = sec_ypeak;
                        }
                        //
                        // After this possible destretch, unconditionally adjust for a mag
                        // change and invert the mag change and scale shift if it applies to
                        // reference
                        if self.m_searched_for_mag {
                            self.m_unstretch_dx = (self.m_unstretch_dx as f64
                                - ((found_mag as f64 - 1.) * self.m_ix_box_ref as f64)
                                    / self.m_nbinning as f64)
                                as f32;
                            self.m_unstretch_dy = (self.m_unstretch_dy as f64
                                - ((found_mag as f64 - 1.) * self.m_iy_box_ref as f64)
                                    / self.m_nbinning as f64)
                                as f32;
                            if mag_view == iv_ref {
                                found_mag = (1. / found_mag as f64) as f32;
                                self.m_unstretch_dx *= found_mag;
                                self.m_unstretch_dy *= found_mag;
                                unstretch_sec_dx *= found_mag;
                                unstretch_sec_dy *= found_mag;
                            }
                            f[(6 * (mag_view - 1)) as usize] = found_mag;
                            f[(6 * (mag_view - 1) + 3) as usize] = found_mag;
                        }
                        if self.m_verbose != 0
                            && (self.m_single_test_patch < 0 || ipatch == self.m_single_test_patch)
                        {
                            printf!(
                                "peak usdx,usdy,xpeak,ypeak %f %f %f %f\n",
                                CArg::Dbl(self.m_unstretch_dx as f64),
                                CArg::Dbl(self.m_unstretch_dy as f64),
                                CArg::Dbl(self.m_xpeak as f64),
                                CArg::Dbl(self.m_ypeak as f64)
                            );
                        }
                        //
                        // compensate for the box offsets of reference and current view,
                        // where the reference is the starting view if accumulating
                        //
                        ix_box_for_adj = self.m_ix_box_ref;
                        iy_box_for_adj = self.m_iy_box_ref;
                        if self.m_if_cumulate != 0 {
                            ix_box_for_adj = ix_box_start;
                            iy_box_for_adj = iy_box_start;
                        }

                        xshift = (idir * self.m_nbinning) as f32 * self.m_unstretch_dx
                            + ix_box_for_adj as f32
                            - self.m_ix_box_cur as f32;
                        yshift = (idir * self.m_nbinning) as f32 * self.m_unstretch_dy
                            + iy_box_for_adj as f32
                            - self.m_iy_box_cur as f32;
                        sec_xshift = (idir * self.m_nbinning) as f32 * unstretch_sec_dx
                            + ix_box_for_adj as f32
                            - self.m_ix_box_cur as f32;
                        sec_yshift = (idir * self.m_nbinning) as f32 * unstretch_sec_dy
                            + iy_box_for_adj as f32
                            - self.m_iy_box_cur as f32;
                        //
                        // It is time to print these peaks if dual output is wanted
                        if ny_sec_peak_box > 0 {
                            if sec_peak_in_box {
                                printf!(
                                    "%s %3d%s %9.2f %9.2f%s %14.6g %9.2f %9.2f %14.6g\n",
                                    CArg::Str("View"),
                                    CArg::Int(iview as i64),
                                    CArg::Str(", shifts"),
                                    CArg::Dbl(xshift as f64),
                                    CArg::Dbl(yshift as f64),
                                    CArg::Str("      peak"),
                                    CArg::Dbl(self.m_peak_val as f64),
                                    CArg::Dbl(sec_xshift as f64),
                                    CArg::Dbl(sec_yshift as f64),
                                    CArg::Dbl(
                                        self.m_peak_list[self.m_ind_peak_sort[1] as usize] as f64
                                    )
                                );
                            } else {
                                printf!(
                                    "%s %3d%s %9.2f %9.2f%s %14.6g\n",
                                    CArg::Str("View"),
                                    CArg::Int(iview as i64),
                                    CArg::Str(", shifts"),
                                    CArg::Dbl(xshift as f64),
                                    CArg::Dbl(yshift as f64),
                                    CArg::Str("      peak"),
                                    CArg::Dbl(self.m_peak_val as f64)
                                );
                            }
                        }

                        if self.m_tracking {
                            if if_find_warp != 0 {
                                //
                                // Only add control point if there was a peak found within limits
                                if self.m_peak_val as f64 > -1.0e29 {
                                    let nc = num_control as usize;
                                    x_control[nc] = xmodel[mind(ipatch, iv_cur)];
                                    y_control[nc] = ymodel[mind(ipatch, iv_cur)];
                                    x_vector[nc] = (-self.m_nbinning) as f32 * self.m_xpeak;
                                    y_vector[nc] = (-self.m_nbinning) as f32 * self.m_ypeak;
                                    num_control += 1;
                                }
                            } else {
                                //
                                // compensate the shift that is to be accumulated for the
                                // difference between the center of the box and the model point
                                // on reference view
                                cum_xcenter = xmodel[mind(ipatch, iv_ref)];
                                cum_ycenter = ymodel[mind(ipatch, iv_ref)];
                                x_mod_offset = (cum_xcenter as f64
                                    - ((self.m_ix_box_ref + self.m_ix_cen_start) as f64
                                        + self.m_nx_use as f64 / 2.))
                                    as f32;
                                y_mod_offset = (cum_ycenter as f64
                                    - ((self.m_iy_box_ref + self.m_iy_cen_start) as f64
                                        + self.m_ny_use as f64 / 2.))
                                    as f32;
                                cum_xrot =
                                    x_mod_offset * self.m_cos_phi + y_mod_offset * self.m_sin_phi;
                                x_adjust = (cum_xrot as f64 * (1. - cos_ratio as f64)) as f32;
                                //
                                // Subtract adjusted shift to get new model point position
                                self.m_xpeak_tmp =
                                    cum_xcenter - (xshift + x_adjust * self.m_cos_phi);
                                self.m_ypeak_tmp =
                                    cum_ycenter - (yshift + x_adjust * self.m_sin_phi);
                                self.m_model_xvecs[ipu].push(self.m_xpeak_tmp);
                                self.m_model_yvecs[ipu].push(self.m_ypeak_tmp);
                                self.m_peak_vecs[ipu].push(
                                    self.m_peak_list[self.m_ind_peak_sort[ipeak as usize] as usize],
                                );
                                if ipeak == 0 {
                                    xmodel[mind(ipatch, iview)] = self.m_xpeak_tmp;
                                    ymodel[mind(ipatch, iview)] = self.m_ypeak_tmp;
                                }
                            }
                        }
                    }
                    //
                    // if not leaving axis at center of box, compute amount that
                    // current view must be shifted across tilt axis to be lined up at
                    // center, then rotate that to get shifts in X and Y
                    // This is not due to cosine stretch, it is due to difference in
                    // the box offsets
                    if if_leave_axis == 0 {
                        x_box_ofs = (self.m_ix_box_cur - ix_box_for_adj) as f32 * self.m_cos_phi
                            + (self.m_iy_box_cur - iy_box_for_adj) as f32 * self.m_sin_phi;
                        xshift += x_box_ofs * self.m_cos_phi;
                        yshift += x_box_ofs * self.m_sin_phi;
                    }
                    //
                    // If not cumulative, adjust the shift to bring the tilt axis
                    // to the true center from the center of last view
                    // If last view was shifted to right to line up with center,
                    // then the true center is to the left of the center of that view
                    // and the tilt axis (and this view) must be shifted to the left
                    // But base this on the cumulative shift of the center of the image
                    // not of the box if axis not at center: so add on the cumulative
                    // amount that would have been added due to box offsets in block
                    // above
                    if self.m_if_cumulate == 0 && if_no_stretch == 0 && !use_ref_file {
                        x_from_cen = cum_xshift;
                        y_from_cen = cum_yshift;
                        if if_leave_axis != 0 {
                            x_from_cen += (self.m_ix_box_ref - ix_box_start) as f32;
                            y_from_cen += (self.m_iy_box_ref - iy_box_start) as f32;
                        }
                        cum_xrot = x_from_cen * self.m_cos_phi + y_from_cen * self.m_sin_phi;
                        x_adjust = (cum_xrot as f64 * (cos_ratio as f64 - 1.)) as f32;
                        xshift += x_adjust * self.m_cos_phi;
                        yshift += x_adjust * self.m_sin_phi;
                        //
                        // Add to cumulative shift and report and save the absolute shift
                        //
                        cum_xshift += xshift;
                        cum_yshift += yshift;
                        xshift = cum_xshift;
                        yshift = cum_yshift;
                    }

                    if !self.m_tracking && ny_sec_peak_box <= 0 {
                        printf!(
                            if eval_ccc != 0 {
                                "%s %3d%s %9.2f %9.2f%s %14.6f\n"
                            } else {
                                "%s %3d%s %9.2f %9.2f%s %14.6g\n"
                            },
                            CArg::Str("View"),
                            CArg::Int(iview as i64),
                            CArg::Str(", shifts"),
                            CArg::Dbl(xshift as f64),
                            CArg::Dbl(yshift as f64),
                            CArg::Str("      peak"),
                            CArg::Dbl(self.m_peak_val as f64)
                        );
                    }
                    if if_scan_rotation > 0 {
                        if num_rot_steps > 0 {
                            //
                            // DEPENDENCY: transferfid is looking for 'Best angle in' and reads
                            // the number after the last =
                            if ind_best_rot > 1 && ind_best_rot < num_rot_steps {
                                printf!(
                                    "%s %6.1f%s %7.2f\n",
                                    CArg::Str("  Best angle in rotation scan ="),
                                    CArg::Dbl((idir as f32 * best_angle) as f64),
                                    CArg::Str("   interpolated angle ="),
                                    CArg::Dbl((idir as f32 * angle) as f64)
                                );
                            } else {
                                printf!(
                                    "  Best angle in rotation scan = %6.1f\n",
                                    CArg::Dbl((idir as f32 * angle) as f64)
                                );
                            }
                        }
                        ind = 6 * (mag_view - 1);
                        let mut a = [0f32; 4];
                        rotmagstr_to_amat(idir as f32 * angle, 1., 1., 0., &mut a);
                        f[ind as usize..ind as usize + 4].copy_from_slice(&a);
                    }
                    //
                    // DNM 10/22/03: Only do flush for large stacks because of problem
                    // inside shell scripts in Windows/Intel
                    //
                    if self.m_iz_end - self.m_iz_start > 2 {
                        let _ = ImodFile::Stdout.flush();
                    }
                    f[(6 * (iview - 1) + 4) as usize] = xshift;
                    f[(6 * (iview - 1) + 5) as usize] = yshift;
                    xf_copy(&self.m_fs, 2, &mut self.m_f_unit, 2);
                    if im_file_out.is_some()
                        && (self.m_single_test_patch < 0 || self.m_single_test_patch == ipatch)
                    {
                        if self.m_single_test_patch >= 0 && self.m_single_nx_pad == 0 {
                            self.m_single_nx_pad = self.m_nx_pad;
                            self.m_single_ny_pad = self.m_ny_pad;
                            iiu_alt_size_samp_cell(3, self.m_nx_pad, self.m_ny_pad, self.m_nz_out);
                        }
                        self.pack_corr(self.m_nx_pad + 2);
                        if self.m_mode != 2 {
                            scale_array_for_mode(
                                &mut self.m_crray,
                                self.m_nx_pad,
                                self.m_mode,
                                0,
                                self.m_nx_pad - 1,
                                0,
                                self.m_ny_pad - 1,
                                &mut self.m_dmin2,
                                &mut self.m_dmax2,
                                &mut dmean3_local,
                            );
                        } else {
                            let (nxp, nyp) = (self.m_nx_pad, self.m_ny_pad);
                            if let Some((a, b, c)) = with_mrc_data!(self.m_crray, |d| {
                                full_array_min_max_mean(&mut d, MRC_MODE_FLOAT, nxp, nyp)
                            }) {
                                self.m_dmin2 = a;
                                self.m_dmax2 = b;
                                dmean3_local = c;
                            }
                        }
                        unsafe { iiu_write_section(3, self.m_crray.as_mut_ptr().cast()) };
                        self.m_nz_out += 1;
                        if self.m_verbose != 0 {
                            printf!(
                                "Correlation output at Z = %d\n",
                                CArg::Int(self.m_nz_out as i64)
                            );
                        }
                        //
                        self.m_dmax = if self.m_dmax > self.m_dmax2 {
                            self.m_dmax
                        } else {
                            self.m_dmax2
                        };
                        self.m_dmin = if self.m_dmin < self.m_dmin2 {
                            self.m_dmin
                        } else {
                            self.m_dmin2
                        };
                        self.m_dmean_sum += dmean3_local;
                    }
                    ipatch += 1;
                }

                if if_find_warp != 0 {
                    if self.m_verbose != 0 {
                        printf!(
                            "View %d  %d warp points\n",
                            CArg::Int(iv_cur as i64),
                            CArg::Int(num_control as i64)
                        );
                    }
                    ierr = 0;
                    iv = iv_cur + iv_pair_offset;
                    if num_control > 2 {
                        ierr = set_warp_points(
                            iv - 1,
                            num_control,
                            &x_control,
                            &y_control,
                            &x_vector,
                            &y_vector,
                        );
                        if ierr != 0 {
                            exit_error(b"Setting warp points for the current view");
                        }
                        if raw_aligned_pair {
                            ierr = set_linear_transform(
                                iv - 1,
                                &f_preali[(6 * (iv - 1)) as usize..],
                                2,
                            );
                        }
                        if ierr != 0 {
                            exit_error(b"Setting linear transform in warp file");
                        }
                        if if_find_warp > 0 {
                            ierr = separate_linear_transform(iv - 1);
                        }
                        if ierr != 0 {
                            exit_error(b"Separating linear component from warp transform");
                        }
                    } else if num_control > 0 {
                        xf_unit(&mut self.m_fs, 1.0, 2);
                        if raw_aligned_pair {
                            xf_copy(&f_preali[(6 * (iv - 1)) as usize..], 2, &mut self.m_fs, 2);
                        }
                        for i in 0..num_control as usize {
                            self.m_fs[4] -= x_vector[i] / num_control as f32;
                            self.m_fs[5] -= y_vector[i] / num_control as f32;
                        }
                        // `tiltxcorr.cpp:2381` passes `iv`, not `iv - 1` as the other
                        // warp calls do.
                        ierr = set_linear_transform(iv, &self.m_fs, 2);
                        if ierr != 0 {
                            exit_error(b"Setting linear transform in warp file");
                        }
                    }
                } else if self.m_tracking
                    && self.m_num_patches > min_patch_for_pred
                    && self.m_domain_xsize != 0
                {
                    self.do_prediction_fits(iv_cur, loop_dir, &mut xmodel, &mut ymodel);
                }
                if self.m_tracking {
                    printf!("View %3d processed\n", CArg::Int(iview as i64));
                }
                iview += loop_dir;
            }
            //
            // set up for second loop
            //
            iv_start = self.m_min_tilt - 1 - start_low_for_track;
            iv_end = self.m_iz_start;
            loop_dir = -1;
        }
        //
        if im_file_out.is_some() {
            iiu_alt_size_samp_cell(
                3,
                if self.m_single_test_patch >= 0 {
                    self.m_single_nx_pad
                } else {
                    self.m_nx_pad
                },
                if self.m_single_test_patch >= 0 {
                    self.m_single_ny_pad
                } else {
                    self.m_ny_pad
                },
                self.m_nz_out,
            );
            dmean = self.m_dmean_sum / self.m_nz_out as f32;
            let mut sstr = String::from("TILTXCORR: stack cosine stretch/correlated");
            if self.m_delta_ctf != 0. {
                sstr.push_str(", filtered");
            }
            let mut title = [0u8; MRC_LABEL_SIZE];
            mrc_fill_label_string(sstr.as_bytes(), &mut title);
            let title_len = title.iter().position(|&b| b == 0).unwrap_or(MRC_LABEL_SIZE);
            iiu_write_header_str(
                3,
                &String::from_utf8_lossy(&title[..title_len]),
                1,
                self.m_dmin,
                self.m_dmax,
                dmean,
            );
            unsafe { iiu_close(3) };
        }
        //
        // Now adjust transforms of skipped ones so they are always the same as the one below
        if !self.m_breaking && !use_ref_file {
            last_not_skipped = -1;
            for ivw in 1..=num_views {
                if number_in_list(ivw, Some(&self.m_list_skip), self.m_num_skip, 0) == 0 {
                    last_not_skipped = ivw;
                } else if last_not_skipped > 0 {
                    f[(6 * (ivw - 1) + 4) as usize] = f[(6 * (last_not_skipped - 1) + 4) as usize];
                    f[(6 * (ivw - 1) + 5) as usize] = f[(6 * (last_not_skipped - 1) + 5) as usize];
                }
            }
        }
        //
        // Normal output
        if !self.m_tracking {
            imod_backup_file(&xf_file_out);
            let fp = match std::fs::File::create(&xf_file_out) {
                Ok(file) => file,
                Err(_) => exit_error_fmt!("Opening output file %s", CArg::Str(&xf_file_out)),
            };
            let mut fp_xf = std::io::BufWriter::new(fp);
            let strerror = |e: i32| {
                let text = std::io::Error::from_raw_os_error(e).to_string();
                text.split(" (os error ")
                    .next()
                    .unwrap_or(&text)
                    .to_string()
            };
            //
            // If leaving axis at an offset box, output G transforms directly so that
            // the material in box at zero tilt will stay in box
            if if_leave_axis != 0 || use_ref_file {
                for ivw in 0..nzu {
                    ierr = write_xform(&mut fp_xf, &f[6 * ivw..]);
                    if ierr != 0 {
                        exit_error_fmt!(
                            "Writing transforms to output file: %s",
                            CArg::Str(&strerror(ierr))
                        );
                    }
                }
            } else {
                //
                // Anticipate what xftoxg will do, namely get to transforms with zero
                // mean, and shift tilt axis to be at middle in that case
                //
                let mut iloop = 1;
                cum_xshift = 10.;
                while iloop <= 10
                    && (cum_xshift as f64 > 0.1 || cum_yshift as f64 > 0.1)
                    && if_no_stretch == 0
                {
                    iloop += 1;
                    cum_xshift = 0.;
                    cum_yshift = 0.;
                    for ivw in 0..nzu {
                        cum_xshift -= f[6 * ivw + 4] / nz as f32;
                        cum_yshift -= f[6 * ivw + 5] / nz as f32;
                    }
                    //
                    // rotate the average shift to tilt axis vertical, adjust the X shift
                    // to keep tilt axis in center and apply shift for all views
                    //
                    cum_xrot = cum_xshift * self.m_cos_phi + cum_yshift * self.m_sin_phi;
                    yshift = -cum_xshift * self.m_sin_phi + cum_yshift * self.m_cos_phi;
                    for ivw in 0..nzu {
                        x_adjust = cum_xrot * cosd!(self.m_tilt[ivw]);
                        f[6 * ivw + 4] += x_adjust * self.m_cos_phi - yshift * self.m_sin_phi;
                        f[6 * ivw + 5] += x_adjust * self.m_sin_phi + yshift * self.m_cos_phi;
                    }
                }
                //
                // convert from g to f transforms by taking differences
                //
                let mut ivw = nz - 1;
                while ivw >= 1 {
                    let iu = ivw as usize;
                    if number_in_list(ivw + 1, Some(&self.m_list_skip), self.m_num_skip, 0) == 0 {
                        f[6 * iu + 4] -= f[6 * (iu - 1) + 4];
                        f[6 * iu + 5] -= f[6 * (iu - 1) + 5];
                    } else {
                        xf_unit(&mut f[6 * iu..], 1., 2);
                    }
                    ivw -= 1;
                }
                xf_unit(&mut f[0..], 1., 2);
                for ivw in 0..nzu {
                    ierr = write_xform(&mut fp_xf, &f[6 * ivw..]);
                    if ierr != 0 {
                        exit_error_fmt!(
                            "Writing transforms to output file: %s",
                            CArg::Str(&strerror(ierr))
                        );
                    }
                }
            }
            let _ = fp_xf.flush();
        } else if if_find_warp != 0 {
            //
            // Warp output
            ierr = write_warp_file(&xf_file_out, 0);
            if ierr < 0 {
                exit_error(b"Writing warp file");
            }
            if ierr > 0 {
                printf!(
                    "WARNING: TILTXCORR - Failed to make backup of existing output warping file\n"
                );
            }
        } else {
            //
            // Write Model of tracked points
            ierr = newimod();
            self.m_fm.max_mod_obj = 0;
            self.m_fm.n_point = 0;
            //
            // Go through patches and put points in model structure
            for ip in 0..self.m_num_patches {
                //
                // Count model points in this patch and divide them up
                ipnt = 0;
                iv_base = -1;
                for izv in self.m_iz_start..=self.m_iz_end {
                    if xmodel[mind(ip, izv)] as f64 > -1.0e9 {
                        ipnt += 1;
                        if iv_base < 0 {
                            iv_base = izv;
                        }
                    }
                }
                len_conts = len_contour.min(ipnt);
                if len_conts > min_cont_overlap {
                    num_cont = (ipnt - 1) / (len_conts - min_cont_overlap) + 1;
                } else {
                    num_cont = 1;
                }
                lap_total = num_cont * len_conts - ipnt;
                lap_base = lap_total / (num_cont - 1).max(1);
                lap_remainder = lap_total % (num_cont - 1).max(1);
                //
                // Loop through each contour starting at base position
                for i in 1..=num_cont {
                    let mo = self.m_fm.max_mod_obj as usize;
                    self.m_fm.obj_color[mo][0] = 1;
                    self.m_fm.obj_color[mo][1] = 255;
                    self.m_fm.ibase_obj[mo] = self.m_fm.n_point;
                    self.m_fm.npt_in_obj[mo] = len_conts;
                    self.m_fm.max_mod_obj += 1;
                    iv = iv_base;
                    ipnt = 0;
                    for _j in 0..len_conts {
                        while (xmodel[mind(ip, iv)] as f64) < -1.0e9 && iv < self.m_iz_end {
                            iv += 1;
                        }
                        let np = self.m_fm.n_point as usize;
                        self.m_fm.p_coord[np][0] = xmodel[mind(ip, iv)];
                        self.m_fm.p_coord[np][1] = ymodel[mind(ip, iv)];
                        for izv in 0..nz {
                            if iz_pc_list[izv as usize] + 1 == iv {
                                self.m_fm.p_coord[np][2] = izv as f32;
                            }
                        }

                        // This is still numbered from 1 so add 1 here
                        self.m_fm.object[np] = self.m_fm.n_point + 1;
                        self.m_fm.pt_label[np] = 0;
                        self.m_fm.n_point += 1;
                        ipnt += 1;
                        //
                        // If hit point for start of overlap, record position
                        if (i > lap_remainder && ipnt == len_conts + 1 - lap_base)
                            || (i <= lap_remainder && ipnt == len_conts - lap_base)
                        {
                            iv_base = iv;
                        }
                        iv += 1;
                    }
                }
            }
            //
            // Set model properties: open contours, thicken current contour, etc.
            ierr = putmodelname("Patch Tracking Model");
            let zscale: f32 = {
                let v = (0.3 * (self.m_nx + self.m_ny) as f64) / nz as f64;
                let m = if 40. < v { 40. } else { v };
                (if 1. > m { 1. } else { m }) as f32
            };
            putimodflag(1, 1);
            putimodflag(1, 10);
            putsymtype(1, 0);
            putsymsize(1, 7);
            putlinewidth(1, 2);
            putobjcolor(1, 255, 0, 255);
            putimodzscale(zscale);
            delta = iiu_ret_delta(1);
            origin = iiu_ret_origin(1);
            cur_tilt = iiu_ret_tilt(1);
            ierr = putimageref(&delta, &origin, &cur_tilt);
            ierr = putimodmaxes(self.m_nx, self.m_ny, nz);
            let _ = ierr;
            let _ = scale_fort_mod_to_image(&mut self.m_fm, 1, 1);
            let _ = write_fort_model(&xf_file_out, &mut self.m_fm);
        }
        unsafe { iiu_close(1) };
        //
        if self.m_verbose != 0 {
            printf!(
                "%s %.3f %s %.3f %s %.3f %s %.3f %s %.3f %s %.3f %s %.3f %s %.3f\n",
                CArg::Str("interp"),
                CArg::Dbl(self.m_wall_interp),
                CArg::Str("  fft"),
                CArg::Dbl(self.m_wallfft),
                CArg::Str("  peak"),
                CArg::Dbl(self.m_wall_peak),
                CArg::Str("  mask"),
                CArg::Dbl(self.m_wall_mask),
                CArg::Str("  load/extract"),
                CArg::Dbl(self.m_wall_load),
                CArg::Str("read/copy"),
                CArg::Dbl(self.m_wall_read),
                CArg::Str("CCC"),
                CArg::Dbl(self.m_wall_ccc),
                CArg::Str("SD"),
                CArg::Dbl(self.m_wall_sd_stat)
            );
            if self.m_use_gpu >= 0 {
                self.m_txc_gpu.as_ref().unwrap().report_times();
            }
        }
        let _ = (
            dmean2,
            mxyz,
            min_xpiece,
            min_ypiece,
            nx_overlap,
            ny_overlap,
            num_opt_arg,
            num_non_opt_arg,
            input_pass0,
            iv_skip,
            unstretch_sec_dx,
            unstretch_sec_dy,
            num_loops,
            x_frac_to_add,
            y_frac_to_add,
            cur_view_out,
        );
        0
    }

    /// `TiltXCorr::correlateAndFindPeaks` (`tiltxcorr.cpp:2594`).  `ub_main`
    /// selects `mArray`/`mBrray` as the source's `ubArray`/`ubBrray`
    /// arguments; otherwise they are `m_ub_array`/`m_ub_brray` (see the module
    /// comment).
    fn correlate_and_find_peaks(
        &mut self,
        use_ccc: bool,
        ub_main: bool,
        if_do_im_out: i32,
        ipatch: i32,
    ) {
        let mut widths = [0f32; LIMPEAKS];
        let mut in_streak = [false; LIMPEAKS];
        let mut width_mins = [0f32; LIMPEAKS];
        let mut xtemp: f32;
        let mut ytemp: f32;
        let mut ub_xpeaks = [0f32; 2];
        let mut ub_ypeaks = [0f32; 2];
        let mut ub_peak_list = [0f32; 2];
        let mut overlap: f32;
        let streak: f32;
        let mut wgt_ccc: f32;
        let mut xrot: f32 = 0.;
        let mut yrot: f32 = 0.;
        let mut limit_xlo: i32;
        let mut limit_xhi: i32;
        let mut limit_ylo: i32;
        let mut limit_yhi: i32;
        let mut ind_peak: i32;
        let mut nsum: i32 = 0;
        let nx_cc_trim_a: i32;
        let ny_cc_trim_a: i32;
        let (mut test_xlo, mut test_xhi, mut test_ylo, mut test_yhi) = (0, 0, 0, 0);
        let mut corr_xdim: i32;
        let mut corr_ypad: i32;
        let mut ind_first_out: i32;
        let mut ind_second_out: i32;
        let mut nx_interp: i32;
        let mut ny_interp: i32;
        let mut ind: i32;
        let mut ierr: i32 = 0;
        let nx_cc_trim_b: i32;
        let ny_cc_trim_b: i32;
        let mut amat = [0f32; 6];
        let mut ccc_max: f64;
        let mut ccc: f64;
        let mut wall_start: f64;
        let skip_ccc_crit: f32 = 0.33f32;
        //
        // get "current" into array, stretch into brray, pad it
        //
        self.read_binned_or_reduced(
            1,
            self.m_iz_cur,
            self.m_ix_cen_start + self.m_ix_box_cur,
            self.m_iy_cen_start + self.m_iy_box_cur,
            self.m_taper_cur,
            ipatch,
        );
        nx_interp = self.m_nx_use_bin;
        ny_interp = self.m_ny_use_bin;
        if self.m_use_gpu < 0 {
            //
            // 7/11/03: We have to feed the interpolation the right mean or
            // it will create a bad edge mean for padding
            //
            let (nxb, nyb) = (self.m_nx_use_bin, self.m_ny_use_bin);
            if let Some((a, b, c)) = with_mrc_data!(self.m_array, |d| full_array_min_max_mean(
                &mut d,
                MRC_MODE_FLOAT,
                nxb,
                nyb
            )) {
                self.m_use_min = a;
                self.m_use_max = b;
                self.m_use_mean = c;
            }
            if !self.m_tracking && self.m_num_bound > 0 {
                wall_start = wall_time();
                self.mask_outside_boundaries(self.m_nx_use_bin, self.m_ny_use_bin);
                self.m_wall_mask += wall_time() - wall_start;
            }
            wall_start = wall_time();
            if self.m_searched_for_mag {
                //
                // Get new limits for interpolated output when there is a mag down, so that
                // tapering happens inside true interpolated image area and correlations take
                // account of this area too.  Use the OUTER of two limits to accommodate
                // stretched area
                let xc = (self.m_nx_use_bin as f64 / 2.) as f32;
                let yc = (self.m_ny_use_bin as f64 / 2.) as f32;
                (xrot, yrot) = xf_apply(&self.m_fs, xc, yc, self.m_nx_use_bin as f32, 0., 2);
                (xtemp, ytemp) = xf_apply(
                    &self.m_fs,
                    xc,
                    yc,
                    self.m_nx_use_bin as f32,
                    self.m_ny_use_bin as f32,
                    2,
                );
                nx_interp = self.m_nx_use_bin.min(
                    2 * ((if xrot > xtemp { xrot } else { xtemp }) as f64
                        - self.m_nx_use_bin as f64 / 2.) as i32,
                );
                let a = self.m_ny_use_bin as f64 / 2. - yrot as f64;
                let b = ytemp as f64 - self.m_ny_use_bin as f64 / 2.;
                ny_interp = self
                    .m_ny_use_bin
                    .min(2 * (if a > b { a } else { b }) as i32);
            }
            xf_copy(&self.m_fs, 2, &mut amat, 2);
            cubinterp(
                &self.m_array,
                &mut self.m_brray,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                &[[amat[0], amat[1]], [amat[2], amat[3]]],
                (self.m_nx_use_bin as f64 / 2.) as f32,
                (self.m_ny_use_bin as f64 / 2.) as f32,
                self.m_xpeak_frac,
                self.m_ypeak_frac,
                1.,
                self.m_use_mean,
                0,
            );

            self.m_wall_interp += wall_time() - wall_start;
            slice_taper_in_pad(
                PadIn::InPlace,
                MRC_MODE_FLOAT,
                self.m_nx_use_bin,
                (self.m_nx_use_bin - nx_interp) / 2,
                (self.m_nx_use_bin + nx_interp) / 2 - 1,
                (self.m_ny_use_bin - ny_interp) / 2,
                (self.m_ny_use_bin + ny_interp) / 2 - 1,
                &mut self.m_brray,
                self.m_nx_pad + 2,
                self.m_nx_pad,
                self.m_ny_pad,
                self.m_nx_taper,
                self.m_ny_taper,
            );
            //
            xcorr_mean_zero(
                &mut self.m_brray,
                self.m_nx_pad + 2,
                self.m_nx_pad,
                self.m_ny_pad,
            );
        }
        limit_xlo = -self.m_limit_shift_x;
        limit_xhi = self.m_limit_shift_x;
        limit_ylo = -self.m_limit_shift_y;
        limit_yhi = self.m_limit_shift_y;
        //
        // get "last" into array, just pad it there
        // All operations involving array happen only if not reusing, or on first round
        if self.m_if_reuse_prev <= 0 || self.m_use_gpu >= 0 {
            self.read_binned_or_reduced(
                self.m_iunit_ref,
                self.m_iz_last,
                self.m_ix_cen_start + self.m_ix_box_ref,
                self.m_iy_cen_start + self.m_iy_box_ref,
                self.m_taper_ref,
                ipatch,
            );
            if self.m_if_cumulate != 0 {
                limit_xlo = (self.m_xpeak - self.m_limit_shift_x as f32) as i32;
                limit_xhi = (self.m_xpeak + self.m_limit_shift_x as f32) as i32;
                limit_ylo = (self.m_ypeak - self.m_limit_shift_y as f32) as i32;
                limit_yhi = (self.m_ypeak + self.m_limit_shift_y as f32) as i32;
                if self.m_iter == 1 {
                    //
                    // if accumulating, transform image by last shift, add it to
                    // sum array, then taper and pad into array
                    //
                    if self.m_if_abs_stretch == 0 {
                        xf_unit(&mut self.m_f_unit, 1.0, 2);
                    } else {
                        self.m_unstretch_dx = self.m_xpeak;
                        self.m_unstretch_dy = self.m_ypeak;
                    }
                    if self.m_verbose != 0 {
                        printf!(
                            "cumulating usdx,usdy,xpeak,ypeak %f %f %f %f\n",
                            CArg::Dbl(self.m_unstretch_dx as f64),
                            CArg::Dbl(self.m_unstretch_dy as f64),
                            CArg::Dbl(self.m_xpeak as f64),
                            CArg::Dbl(self.m_ypeak as f64)
                        );
                    }
                    let (nxb, nyb) = (self.m_nx_use_bin, self.m_ny_use_bin);
                    if let Some((a, b, c)) = with_mrc_data!(self.m_array, |d| {
                        full_array_min_max_mean(&mut d, MRC_MODE_FLOAT, nxb, nyb)
                    }) {
                        self.m_use_min = a;
                        self.m_use_max = b;
                        self.m_use_mean = c;
                    }
                    xf_copy(&self.m_f_unit, 2, &mut amat, 2);
                    cubinterp(
                        &self.m_array,
                        &mut self.m_crray,
                        self.m_nx_use_bin,
                        self.m_ny_use_bin,
                        self.m_nx_use_bin,
                        self.m_ny_use_bin,
                        &[[amat[0], amat[1]], [amat[2], amat[3]]],
                        (self.m_nx_use_bin as f64 / 2.) as f32,
                        (self.m_ny_use_bin as f64 / 2.) as f32,
                        self.m_unstretch_dx,
                        self.m_unstretch_dy,
                        1.,
                        self.m_use_mean,
                        0,
                    );
                    for i in 0..(self.m_nx_use_bin * self.m_ny_use_bin) as usize {
                        self.m_sum_array[i] += self.m_crray[i];
                    }
                }
                slice_taper_in_pad(
                    PadIn::Float(&self.m_sum_array),
                    MRC_MODE_FLOAT,
                    self.m_nx_use_bin,
                    0,
                    self.m_nx_use_bin - 1,
                    0,
                    self.m_ny_use_bin - 1,
                    &mut self.m_array,
                    self.m_nx_pad + 2,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    self.m_nx_taper,
                    self.m_ny_taper,
                );
            } else if self.m_use_gpu < 0 {
                slice_taper_in_pad(
                    PadIn::InPlace,
                    MRC_MODE_FLOAT,
                    self.m_nx_use_bin,
                    0,
                    self.m_nx_use_bin - 1,
                    0,
                    self.m_ny_use_bin - 1,
                    &mut self.m_array,
                    self.m_nx_pad + 2,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    self.m_nx_taper,
                    self.m_ny_taper,
                );
            }

            if if_do_im_out != 0 {
                for is_out in 1..=2 {
                    if is_out == 1 {
                        repack_float_image(
                            f32_bytes_mut!(self.m_crray),
                            f32_bytes!(self.m_array),
                            self.m_nx_pad + 2,
                            0,
                            self.m_nx_pad - 1,
                            0,
                            self.m_ny_pad - 1,
                        );
                    } else {
                        repack_float_image(
                            f32_bytes_mut!(self.m_crray),
                            f32_bytes!(self.m_brray),
                            self.m_nx_pad + 2,
                            0,
                            self.m_nx_pad - 1,
                            0,
                            self.m_ny_pad - 1,
                        );
                    }
                    if self.m_mode != 2 {
                        scale_array_for_mode(
                            &mut self.m_crray,
                            self.m_nx_pad,
                            self.m_mode,
                            0,
                            self.m_nx_pad - 1,
                            0,
                            self.m_ny_pad - 1,
                            &mut self.m_dmin2,
                            &mut self.m_dmax2,
                            &mut self.m_dmean3,
                        );
                    } else {
                        let (nxp, nyp) = (self.m_nx_pad, self.m_ny_pad);
                        if let Some((a, b, c)) = with_mrc_data!(self.m_crray, |d| {
                            full_array_min_max_mean(&mut d, MRC_MODE_FLOAT, nxp, nyp)
                        }) {
                            self.m_dmin2 = a;
                            self.m_dmax2 = b;
                            self.m_dmean3 = c;
                        }
                    }
                    if self.m_single_test_patch >= 0 && self.m_single_nx_pad == 0 {
                        self.m_single_nx_pad = self.m_nx_pad;
                        self.m_single_ny_pad = self.m_ny_pad;
                        iiu_alt_size_samp_cell(3, self.m_nx_pad, self.m_ny_pad, self.m_nz_out);
                    }
                    unsafe { iiu_write_section(3, self.m_crray.as_mut_ptr().cast()) };
                    self.m_nz_out += 1;
                    //
                    self.m_dmax = if self.m_dmax > self.m_dmax2 {
                        self.m_dmax
                    } else {
                        self.m_dmax2
                    };
                    self.m_dmin = if self.m_dmin < self.m_dmin2 {
                        self.m_dmin
                    } else {
                        self.m_dmin2
                    };
                    self.m_dmean_sum += self.m_dmean3;
                }
            }
            if self.m_use_gpu < 0 {
                xcorr_mean_zero(
                    &mut self.m_array,
                    self.m_nx_pad + 2,
                    self.m_nx_pad,
                    self.m_ny_pad,
                );
            }
        }

        //
        wall_start = wall_time();
        corr_xdim = self.m_nx_pad + 2;
        corr_ypad = self.m_ny_pad;

        // Set the limits for peak finding, and angle if axial limits
        if self.m_limiting_shift != 0 {
            set_peak_find_limits(
                limit_xlo,
                limit_xhi,
                limit_ylo,
                limit_yhi,
                self.m_if_ellipse,
            );
            if self.m_limiting_shift < 0 {
                set_peak_find_angle(self.m_rot_angle);
            }
        }
        let npad_len = ((self.m_nx_pad + 2) * self.m_ny_pad) as usize;
        if self.m_use_gpu >= 0 {
            if self.m_limiting_shift != 0 {
                get_peak_find_test_limits(
                    self.m_nx_pad,
                    self.m_ny_pad,
                    &mut test_xlo,
                    &mut test_xhi,
                    &mut test_ylo,
                    &mut test_yhi,
                );
                test_xhi = corr_xdim.min(2 * test_xhi.max(-test_xlo) + 2);
                test_yhi = self.m_ny_pad.min(2 * test_yhi.max(-test_ylo));
                if ((test_xhi * test_yhi) as f64) < 0.75 * (corr_xdim * corr_ypad) as f64 {
                    corr_xdim = test_xhi;
                    corr_ypad = test_yhi;
                }
            }
            let dump = self.m_single_test_patch >= 0 && ipatch == self.m_single_test_patch;
            if self.m_txc_gpu.as_mut().unwrap().get_correlation(
                self.m_iter,
                &mut self.m_array,
                &mut self.m_crray,
                &mut self.m_brray,
                corr_xdim,
                corr_ypad,
                dump,
            ) != 0
            {
                exit_error(b"Doing filtering and correlation on GPU");
            }
        } else {
            if use_ccc && self.m_delta_ctf == 0. && self.m_if_reuse_prev <= 0 {
                self.m_crray[..npad_len].copy_from_slice(&self.m_array[..npad_len]);
            }
            if self.m_if_reuse_prev <= 0 {
                todfft_c(&mut self.m_array, self.m_nx_pad, self.m_ny_pad, 0);
            }
            todfft_c(&mut self.m_brray, self.m_nx_pad, self.m_ny_pad, 0);
            //
            if self.m_delta_ctf != 0. {
                if self.m_if_reuse_prev <= 0 {
                    if use_ccc {
                        xcorr_filter_part(
                            FilterIn::InPlace,
                            &mut self.m_array,
                            self.m_nx_pad,
                            self.m_ny_pad,
                            &self.m_ctfp,
                            self.m_delta_ctf,
                        );
                        self.m_crray[..npad_len].copy_from_slice(&self.m_array[..npad_len]);
                        todfft_c(&mut self.m_crray, self.m_nx_pad, self.m_ny_pad, 1);
                    } else {
                        xcorr_filter_part(
                            FilterIn::InPlace,
                            &mut self.m_array,
                            self.m_nx_pad,
                            self.m_ny_pad,
                            &self.m_ctfp,
                            self.m_delta_ctf,
                        );
                    }
                }
                if use_ccc {
                    xcorr_filter_part(
                        FilterIn::InPlace,
                        &mut self.m_brray,
                        self.m_nx_pad,
                        self.m_ny_pad,
                        &self.m_ctfp,
                        self.m_delta_ctf,
                    );
                }
            }
            //
            // Here the array gets either saved or copied back when reusing
            if self.m_if_reuse_prev < 0 {
                self.m_arr_copy[..npad_len].copy_from_slice(&self.m_array[..npad_len]);
                self.m_if_reuse_prev = 1;
            } else if self.m_if_reuse_prev > 0 {
                self.m_array[..npad_len].copy_from_slice(&self.m_arr_copy[..npad_len]);
            }
            //
            // multiply array by complex conjugate of brray, put back in array
            //
            conjugate_product(
                &mut self.m_array,
                &self.m_brray,
                self.m_nx_pad,
                self.m_ny_pad,
            );
            todfft_c(&mut self.m_array, self.m_nx_pad, self.m_ny_pad, 1);

            if use_ccc {
                todfft_c(&mut self.m_brray, self.m_nx_pad, self.m_ny_pad, 1);
            }
        }

        self.m_wallfft += wall_time() - wall_start;
        wall_start = wall_time();
        xcorr_peak_find_width(
            &self.m_array,
            corr_xdim,
            corr_ypad,
            &mut self.m_xpeak_list,
            &mut self.m_ypeak_list,
            &mut self.m_peak_list,
            Some(&mut widths),
            Some(&mut width_mins),
            self.m_max_xcorr_peaks,
            0.05,
        );
        //
        // Get the true number of peaks and start an index to them
        for i in 0..self.m_max_xcorr_peaks as usize {
            if self.m_peak_list[i] as f64 > -0.9e30 {
                self.m_num_xcorr_peaks = i as i32 + 1;
            }
            self.m_ind_peak_sort[i] = i as i32;
        }
        self.m_wall_peak += wall_time() - wall_start;
        ccc_max = -10.;
        ind_peak = 0;
        if use_ccc {
            //
            // Evaluate real-space correlation coefficient, using half of tapered region
            nx_cc_trim_a = (self.m_nx_pad - nx_interp) / 2 + self.m_nx_taper / 2;
            ny_cc_trim_a = (self.m_ny_pad - ny_interp) / 2 + self.m_ny_taper / 2;
            nx_cc_trim_b = (self.m_nx_pad - self.m_nx_use_bin) / 2 + self.m_nx_taper / 2;
            ny_cc_trim_b = (self.m_ny_pad - self.m_ny_use_bin) / 2 + self.m_ny_taper / 2;
            wall_start = wall_time();
            let mut i = 0;
            while i < self.m_num_xcorr_peaks {
                let iu = i as usize;
                ccc = cc_coefficient_two_pads(
                    &self.m_crray,
                    &self.m_brray,
                    self.m_nx_pad + 2,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    self.m_xpeak_list[iu],
                    self.m_ypeak_list[iu],
                    nx_cc_trim_a,
                    ny_cc_trim_a,
                    nx_cc_trim_b,
                    ny_cc_trim_b,
                    25,
                    &mut nsum,
                );
                //
                // Peaks with less than 1/8 overlap were ignored; this is a steep function
                // to downweight them instead
                overlap = nsum as f32
                    / ((self.m_nx_pad - 2 * nx_cc_trim_a) * (self.m_ny_pad - 2 * ny_cc_trim_a))
                        as f32;
                let ratio = (self.m_overlap_crit / overlap) as f64;
                let lim = if 10. < ratio { 10. } else { ratio };
                let lim = if 0.1 > lim { 0.1 } else { lim };
                wgt_ccc = (ccc * 1. / (1. + lim.powf(self.m_overlap_power as f64))) as f32;
                if self.m_verbose != 0 {
                    printf!(
                        "%3d%s %6.1f %6.1f%s %13.7e%s %5.3f%s %7.5f %7.5f%s %7.2f %7.2f\n",
                        CArg::Int(i as i64 + 1),
                        CArg::Str(" at "),
                        CArg::Dbl(self.m_xpeak_list[iu] as f64),
                        CArg::Dbl(self.m_ypeak_list[iu] as f64),
                        CArg::Str(" peak ="),
                        CArg::Dbl(self.m_peak_list[iu] as f64),
                        CArg::Str(" ov = "),
                        CArg::Dbl(overlap as f64),
                        CArg::Str(" cc ="),
                        CArg::Dbl(ccc),
                        CArg::Dbl(wgt_ccc as f64),
                        CArg::Str(" width&Min ="),
                        CArg::Dbl(widths[iu] as f64),
                        CArg::Dbl(width_mins[iu] as f64)
                    );
                }
                if wgt_ccc as f64 > ccc_max {
                    ccc_max = wgt_ccc as f64;
                    ind_peak = i; // NOW NUMBERED FROM 0
                    if i > 0 && self.m_verbose != 0 {
                        printf!("Highest raw peak superceded!\n");
                    }
                }
                self.m_peak_list[iu] = wgt_ccc;
                if i > self.m_num_xcorr_peaks
                    && (wgt_ccc as f64) < skip_ccc_crit as f64 * ccc_max
                    && self.m_peak_list[iu] < skip_ccc_crit * self.m_peak_list[0]
                {
                    for j in (i + 1) as usize..self.m_num_xcorr_peaks as usize {
                        self.m_peak_list[j] = 0.;
                    }
                    break;
                }
                i += 1;
            }
            self.m_wall_ccc += wall_time() - wall_start;
            //
            // Sort the CCC's and reverse the index for clarity below
            rs_sort_indexed_floats(
                &self.m_peak_list,
                &mut self.m_ind_peak_sort,
                self.m_num_xcorr_peaks,
            );
            for i in 0..(self.m_num_xcorr_peaks / 2) as usize {
                let j = self.m_ind_peak_sort[i];
                let k = (self.m_num_xcorr_peaks as usize) - i - 1;
                self.m_ind_peak_sort[i] = self.m_ind_peak_sort[k];
                self.m_ind_peak_sort[k] = j;
            }
        }
        //
        // If excluding central peaks, first determine if each peak is in the streak
        if self.m_if_exclude > 0 && self.m_num_xcorr_peaks > 1 {
            // The bad peaks could really be anywhere in this extent although it is quite
            // implausible for it to be all the way out at the end
            let cr = self.m_cos_rot_angle.abs() as f64;
            let sr = self.m_sin_rot_angle.abs() as f64;
            let a = self.m_nx_use as f64 / (if 0.01 > cr { 0.01 } else { cr });
            let b = self.m_ny_use as f64 / (if 0.01 > sr { 0.01 } else { sr });
            streak = (0.5 * (self.m_stretch as f64 - 1.0) * (if a < b { a } else { b })) as f32;

            // Determine if each peak is in or out of the streak, adjusting for the center
            // if the box extraction is different, and keep track of the
            // first and second one outside the streak
            ind_first_out = -1;
            ind_second_out = -1;
            for i in 0..self.m_num_xcorr_peaks as usize {
                ind = self.m_ind_peak_sort[i];
                let iu = ind as usize;
                self.m_xpeak_tmp = self.m_xpeak_list[iu]
                    - (self.m_ix_box_cur - self.m_ix_box_ref) as f32 / self.m_nbinning as f32;
                self.m_ypeak_tmp = self.m_ypeak_list[iu]
                    - (self.m_iy_box_cur - self.m_iy_box_ref) as f32 / self.m_nbinning as f32;
                xrot = self.m_xpeak_tmp * self.m_cos_rot_angle
                    - self.m_ypeak_tmp * self.m_sin_rot_angle;
                yrot = self.m_xpeak_tmp * self.m_sin_rot_angle
                    + self.m_ypeak_tmp * self.m_cos_rot_angle;
                in_streak[iu] =
                    yrot.abs() < self.m_rad_exclude && xrot.abs() < streak + self.m_rad_exclude;
                if !in_streak[iu] {
                    if ind_first_out < 0 {
                        ind_first_out = ind;
                    } else if ind_second_out < 0 {
                        ind_second_out = ind;
                    }
                }
            }

            // If the first peak is in the streak, and its minimum peak width is at least
            // less than the mean width of the first non-streak peak, and
            // the third peak is sufficiently weaker than the second, need to get the
            // unbinned, unstretched correlation
            ind = self.m_ind_peak_sort[0];
            let first_out = ind_first_out.max(0) as usize;
            if ind_first_out >= 0
                && in_streak[ind as usize]
                && widths[first_out] / width_mins[ind as usize] > self.m_bin_width_ratio_crit
                && (ind_second_out < 0
                    || self.m_peak_list[first_out]
                        / self.m_peak_list[ind_second_out.max(0) as usize]
                        > self.m_peak2_to_peak3_crit)
            {
                if self.m_if_ub_peak_is_sharp < 0 {
                    if self.m_verbose != 0 {
                        printf!("Evaluating first and second peak with unbinned correlation\n");
                    }

                    // Load both images unbinned from reference limits, taper, and take the
                    // correlation with high-pass filter only
                    let (mut ub_array, mut ub_brray) = if ub_main {
                        (
                            std::mem::take(&mut self.m_array),
                            std::mem::take(&mut self.m_brray),
                        )
                    } else {
                        (
                            std::mem::take(&mut self.m_ub_array),
                            std::mem::take(&mut self.m_ub_brray),
                        )
                    };
                    iiu_read_binned(
                        self.m_iunit_ref,
                        self.m_iz_last,
                        &mut ub_array,
                        self.m_nx_use,
                        self.m_ny_use,
                        self.m_ix_cen_start + self.m_ix_box_ref,
                        self.m_iy_cen_start + self.m_iy_box_ref,
                        1,
                        self.m_nx_use,
                        self.m_ny_use,
                        &mut self.m_tmp_array,
                        self.m_len_temp as i32,
                        &mut ierr,
                    );
                    if ierr != 0 {
                        exit_error(b"Reading image file");
                    }
                    slice_taper_in_pad(
                        PadIn::InPlace,
                        MRC_MODE_FLOAT,
                        self.m_nx_use,
                        0,
                        self.m_nx_use - 1,
                        0,
                        self.m_ny_use - 1,
                        &mut ub_array,
                        self.m_nx_ub_pad + 2,
                        self.m_nx_ub_pad,
                        self.m_ny_ub_pad,
                        self.m_nx_ub_taper,
                        self.m_ny_ub_taper,
                    );
                    xcorr_mean_zero(
                        &mut ub_array,
                        self.m_nx_ub_pad + 2,
                        self.m_nx_ub_pad,
                        self.m_ny_ub_pad,
                    );
                    iiu_read_binned(
                        1,
                        self.m_iz_cur,
                        &mut ub_brray,
                        self.m_nx_use,
                        self.m_ny_use,
                        self.m_ix_cen_start + self.m_ix_box_ref,
                        self.m_iy_cen_start + self.m_iy_box_ref,
                        1,
                        self.m_nx_use,
                        self.m_ny_use,
                        &mut self.m_tmp_array,
                        self.m_len_temp as i32,
                        &mut ierr,
                    );
                    if ierr != 0 {
                        exit_error(b"Reading image file");
                    }
                    slice_taper_in_pad(
                        PadIn::InPlace,
                        MRC_MODE_FLOAT,
                        self.m_nx_use,
                        0,
                        self.m_nx_use - 1,
                        0,
                        self.m_ny_use - 1,
                        &mut ub_brray,
                        self.m_nx_ub_pad + 2,
                        self.m_nx_ub_pad,
                        self.m_ny_ub_pad,
                        self.m_nx_ub_taper,
                        self.m_ny_ub_taper,
                    );
                    xcorr_mean_zero(
                        &mut ub_brray,
                        self.m_nx_ub_pad + 2,
                        self.m_nx_ub_pad,
                        self.m_ny_ub_pad,
                    );
                    todfft_c(&mut ub_array, self.m_nx_ub_pad, self.m_ny_ub_pad, 0);
                    todfft_c(&mut ub_brray, self.m_nx_ub_pad, self.m_ny_ub_pad, 0);
                    conjugate_product(&mut ub_array, &ub_brray, self.m_nx_ub_pad, self.m_ny_ub_pad);
                    if self.m_delta_ub_ctf != 0. {
                        xcorr_filter_part(
                            FilterIn::InPlace,
                            &mut ub_array,
                            self.m_nx_ub_pad,
                            self.m_ny_ub_pad,
                            &self.m_ctf_ub,
                            self.m_delta_ub_ctf,
                        );
                    }
                    todfft_c(&mut ub_array, self.m_nx_ub_pad, self.m_ny_ub_pad, 1);

                    // It's not clear if these limits should be applied...
                    if self.m_limiting_shift != 0 {
                        set_peak_find_limits(
                            -self.m_limit_ub_shift_x,
                            self.m_limit_ub_shift_x,
                            -self.m_limit_ub_shift_y,
                            self.m_limit_ub_shift_y,
                            self.m_if_ellipse,
                        );
                    }
                    xcorr_peak_find_width(
                        &ub_array,
                        self.m_nx_ub_pad + 2,
                        self.m_ny_ub_pad,
                        &mut ub_xpeaks,
                        &mut ub_ypeaks,
                        &mut ub_peak_list,
                        Some(&mut widths),
                        Some(&mut width_mins),
                        2,
                        0.,
                    );
                    if ub_main {
                        self.m_array = ub_array;
                        self.m_brray = ub_brray;
                    } else {
                        self.m_ub_array = ub_array;
                        self.m_ub_brray = ub_brray;
                    }
                    if self.m_verbose != 0 {
                        for i in 0..2 {
                            printf!(
                                "%2d %8.2f %8.2f %14.7e %7.2f\n",
                                CArg::Int(i as i64 + 1),
                                CArg::Dbl(ub_xpeaks[i] as f64),
                                CArg::Dbl(ub_ypeaks[i] as f64),
                                CArg::Dbl(ub_peak_list[i] as f64),
                                CArg::Dbl(widths[i] as f64)
                            );
                        }
                    }

                    // Accept the second peak if the first is still at origin and is narrow
                    // enough and if the width ratio is big enough
                    self.m_if_ub_peak_is_sharp = 0;
                    if ub_peak_list[1] > 0.
                        && ((ub_xpeaks[0] * ub_xpeaks[0] + ub_ypeaks[0] * ub_ypeaks[0]) as f64)
                            .sqrt()
                            < 0.2
                        && widths[0] <= self.m_central_peak_max_width
                        && widths[1] / widths[0] > self.m_ub_width_ratio_crit
                    {
                        if self.m_verbose != 0 {
                            printf!("Rejecting peak at 0,0!\n");
                        }
                        self.m_if_ub_peak_is_sharp = 1;
                    }
                }
                if self.m_if_ub_peak_is_sharp > 0 {
                    ind_peak = ind_first_out;
                }
            }
        }
        //
        // Done with all picking of substitute peaks, now proceed with final values
        let ip = ind_peak as usize;
        self.m_xpeak_tmp = self.m_xpeak_list[ip];
        self.m_ypeak_tmp = self.m_ypeak_list[ip];
        self.m_peak_val = self.m_peak_list[ip];
        let _ = (xrot, yrot, test_xlo, test_ylo);
    }

    /// `TiltXCorr::adjustCoord` (`tiltxcorr.cpp:2970`).  Adjusts centered
    /// coordinates `xfrom`, `yfrom` from an image at `tilt_from` to an image at
    /// `tilt_to` by rotating to the tilt axis vertical, adjusting the X
    /// coordinate by the ratio of cosines, and rotating back.  The function's
    /// statics are the `s_*` members.
    #[allow(clippy::too_many_arguments)]
    fn adjust_coord(
        &mut self,
        tilt_from: f32,
        tilt_to: f32,
        mut xfrom: f32,
        mut yfrom: f32,
        xto: &mut f32,
        yto: &mut f32,
        model_coords: bool,
    ) {
        let mut xrot: f32;
        let yrot: f32;
        let cos_to: f32;
        let cos_from: f32;
        let tmp_ratio: f32;
        //
        if model_coords {
            xfrom -= self.m_xcen as f32;
            yfrom -= self.m_ycen as f32;
        }
        if tilt_from == self.s_tilt_from_last {
            cos_from = self.s_cos_from_last;
        } else {
            cos_from = cosd!(tilt_from);
            self.s_cos_from_last = cos_from;
            self.s_tilt_from_last = tilt_from;
        }
        if tilt_to == self.s_tilt_to_last {
            cos_to = self.s_cos_to_last;
        } else {
            cos_to = cosd!(tilt_to);
            self.s_cos_to_last = cos_to;
            self.s_tilt_to_last = tilt_to;
        }
        {
            let a = cos_from.abs() as f64;
            tmp_ratio = (cos_to.abs() as f64 / (if a > 1.0e-6 { a } else { 1.0e-6 })) as f32;
        }
        //
        xrot = xfrom * self.m_cos_phi + yfrom * self.m_sin_phi;
        yrot = -xfrom * self.m_sin_phi + yfrom * self.m_cos_phi;
        xrot *= tmp_ratio;
        *xto = xrot * self.m_cos_phi - yrot * self.m_sin_phi;
        *yto = xrot * self.m_sin_phi + yrot * self.m_cos_phi;

        if model_coords {
            *xto += self.m_xcen as f32;
            *yto += self.m_ycen as f32;
        }
    }

    /// `TiltXCorr::getModelAndFlags` (`tiltxcorr.cpp:3014`).  Loads a boundary
    /// or seed model, scales it, allocates `mIobjFlags` to the needed size and
    /// gets the flags.  The source never uses `errMess`.
    fn get_model_and_flags(&mut self, _err_mess: &str) {
        if read_fort_model(&self.m_in_file, &mut self.m_fm).is_err() {
            let list_string = fort_mod_open_error();
            exit_error_fmt!(
                "Reading model file %s: %s",
                CArg::Str(&self.m_in_file),
                CArg::Str(&list_string)
            );
        }
        let num_obj_tot = getimodobjsize();
        self.m_iobj_flags = vec![0; num_obj_tot.max(0) as usize];
        getimodflags(&mut self.m_iobj_flags);
        let _ = scale_fort_model(&mut self.m_fm, 0);
    }

    /// `TiltXCorr::maskOutsideBoundaries` (`tiltxcorr.cpp:3034`).  Masks out
    /// the area outside of all boundary contours with the mean value.  The
    /// only caller passes `mArray`, which this operates on.  The source's
    /// `omp parallel for` over rows writes each pixel independently, so the
    /// serial loop gives the same result at any thread count.
    fn mask_outside_boundaries(&mut self, nx_mob: i32, ny_mob: i32) {
        let mut x0: f32;
        let mut y0: f32;
        let mut outside: bool;
        let num_bound = self.m_num_bound as usize;
        //
        for iy in 0..ny_mob {
            y0 = (iy as f64 + 0.5) as f32;
            for ix in 0..nx_mob {
                x0 = (ix as f64 + 0.5) as f32;
                outside = true;
                for j in 0..num_bound {
                    if x0 >= self.m_xtfs_bmin[j]
                        && x0 <= self.m_xtfs_bmax[j]
                        && y0 >= self.m_ytfs_bmin[j]
                        && y0 <= self.m_ytfs_bmax[j]
                    {
                        let b = self.m_ind_bound[j] as usize;
                        if inside_contour(
                            &self.m_xtfs_bound[b..],
                            &self.m_ytfs_bound[b..],
                            self.m_num_in_bound[j],
                            x0,
                            y0,
                        ) != 0
                        {
                            outside = false;
                            break;
                        }
                    }
                }
                if outside {
                    self.m_array[(ix + nx_mob * iy) as usize] = self.m_use_mean;
                }
            }
        }
        //
        // Taper if only one boundary
        if self.m_num_bound == 1 {
            if taper_at_fill(&mut self.m_array, nx_mob, ny_mob, 16, true) != 0 {
                exit_error(b"Getting memory for tapering inside boundary");
            }
        }
    }

    /// `TiltXCorr::findMinimumTiltView` (`tiltxcorr.cpp:3085`).  Finds the
    /// minimum tilt view that is not skipped and is in the range being done.
    fn find_minimum_tilt_view(&mut self, nz: i32) {
        self.m_iz_start = 1.max(nz.min(self.m_iz_start));
        self.m_iz_end = self.m_iz_start.max(nz.min(self.m_iz_end));
        self.m_tilt_at_min = 10000.;
        for iv in self.m_iz_start..=self.m_iz_end {
            let t = self.m_tilt[(iv - 1) as usize];
            if (self.m_breaking
                || number_in_list(iv, Some(&self.m_list_skip), self.m_num_skip, 0) == 0)
                && (t.abs() < self.m_tilt_at_min.abs()
                    || (self.m_reverse_order != 0 && t.abs() <= self.m_tilt_at_min.abs()))
            {
                self.m_min_tilt = iv;
                self.m_tilt_at_min = t;
            }
        }
        if self.m_tilt_at_min > 9999. {
            exit_error(b"All views in the range being aligned are in the list to skip");
        }
    }

    /// `TiltXCorr::markUsablePatches` (`tiltxcorr.cpp:3107`).  Counts patches
    /// that have usable image area on the given view, returns the number and
    /// marks them in `mTmpArray`.
    fn mark_usable_patches(&mut self, ind_tilt: i32, num_usable: &mut i32) {
        let mut x0: f32 = 0.;
        let mut y0: f32 = 0.;
        *num_usable = 0;
        for ipatch in 0..self.m_num_patches as usize {
            let cenx = self.m_patch_cen_x[ipatch] - self.m_xcen as f32;
            let ceny = self.m_patch_cen_y[ipatch] - self.m_ycen as f32;
            self.adjust_coord(
                0.,
                self.m_tilt[(ind_tilt - 1) as usize],
                cenx,
                ceny,
                &mut x0,
                &mut y0,
                false,
            );
            let dx = self.m_dx_preali[(ind_tilt - 1) as usize] as f64;
            let dy = self.m_dy_preali[(ind_tilt - 1) as usize] as f64;
            let ix = b3d_i_min(&[
                self.m_nx - 1,
                ((self.m_nx + self.m_nx_unali) as f64 / 2. + dx) as i32,
                ((x0 + self.m_xcen as f32) as f64 + self.m_nx_patch as f64 / 2.) as i32,
            ]) - b3d_i_max(&[
                0,
                ((self.m_nx - self.m_nx_unali) as f64 / 2. + dx).ceil() as i32,
                ((x0 + self.m_xcen as f32) as f64 - self.m_nx_patch as f64 / 2.).ceil() as i32,
            ]);
            let iy = b3d_i_min(&[
                self.m_ny - 1,
                ((self.m_ny + self.m_ny_unali) as f64 / 2. + dy) as i32,
                ((y0 + self.m_ycen as f32) as f64 + self.m_ny_patch as f64 / 2.) as i32,
            ]) - b3d_i_max(&[
                0,
                ((self.m_ny - self.m_ny_unali) as f64 / 2. + dy).ceil() as i32,
                ((y0 + self.m_ycen as f32) as f64 - self.m_ny_patch as f64 / 2.).ceil() as i32,
            ]);
            self.m_tmp_array[ipatch] = -1.;
            if ix > 0
                && iy > 0
                && ((ix * iy) as f32)
                    > self.m_crit_non_blank * self.m_nx_patch as f32 * self.m_ny_patch as f32
            {
                *num_usable += 1;
                self.m_tmp_array[ipatch] = 1.;
            }
        }
    }

    /// `TiltXCorr::makePatchListInsideBoundary` (`tiltxcorr.cpp:3136`).  Copies
    /// the whole patch list from "all" to working arrays if there are no
    /// boundary contours, or just the ones inside the boundary contours.
    fn make_patch_list_inside_boundary(&mut self) {
        let mut x0: f32;
        let mut y0: f32;
        self.m_num_patches = 0;
        for ipatch in 0..self.m_num_patches_all as usize {
            self.m_num_inside = 0;
            if self.m_num_bound > 0 {
                for ix in 1..=32 {
                    x0 = (self.m_patch_cen_xall[ipatch] as f64
                        + self.m_nx_patch as f64 * (ix as f64 - 16.5) / 32.)
                        as f32;
                    for iy in 1..=32 {
                        y0 = (self.m_patch_cen_yall[ipatch] as f64
                            + self.m_ny_patch as f64 * (iy as f64 - 16.5) / 32.)
                            as f32;
                        for iv in 0..self.m_num_bound as usize {
                            let b = self.m_ind_bound[iv] as usize;
                            if inside_contour(
                                &self.m_xbound[b..],
                                &self.m_ybound[b..],
                                self.m_num_in_bound[iv],
                                x0,
                                y0,
                            ) != 0
                            {
                                self.m_num_inside += 1;
                                break;
                            }
                        }
                    }
                }
            }
            //
            // If there is no boundary or enough points are inside, keep the patch
            if self.m_num_bound == 0
                || self.m_num_inside as f64 / 1024. >= self.m_crit_inside as f64
            {
                let d = self.m_num_patches as usize;
                self.m_patch_xinds[d] = self.m_patch_xinds[ipatch];
                self.m_patch_yinds[d] = self.m_patch_yinds[ipatch];
                self.m_patch_domains[d] = self.m_patch_domains[ipatch];
                self.m_patch_cen_x[d] = self.m_patch_cen_xall[ipatch];
                self.m_patch_cen_y[d] = self.m_patch_cen_yall[ipatch];
                self.m_num_patches += 1;
            }
        }
    }

    /// `TiltXCorr::readBinnedOrReduced` (`tiltxcorr.cpp:3176`).  Reads or
    /// extracts an area with the given starting coordinates by binning or
    /// reduction and tapers it if requested.  Both callers pass `mArray` as
    /// `arrRead`, which this writes.
    fn read_binned_or_reduced(
        &mut self,
        im_unit: i32,
        iz_read: i32,
        ix_start_read: i32,
        iy_start_read: i32,
        taper_read: bool,
        ipatch: i32,
    ) {
        let mut ierr: i32 = 0;
        let mut out_width: i32 = 0;
        let mut nxr: i32 = 0;
        let mut nyr: i32 = 0;
        let mut load_ind: i32 = 0;
        let ix_end_read = ix_start_read + self.m_nbinning * self.m_nx_use_bin - 1;
        let iy_end_read = iy_start_read + self.m_nbinning * self.m_ny_use_bin - 1;
        let good_ind = if iz_read == self.m_iz_last { 0 } else { 1 };
        let dmean: f32;
        let wall_start = wall_time();

        if self.m_tracking {
            // For using patches, get pointers to array at given Z
            self.get_full_array_and_lines(im_unit, iz_read, &mut load_ind);
            let li = load_ind as usize;
            if self.m_use_gpu >= 0 {
                // If GPU, get the mean of the area and tell gpu rotuine to extract it
                let nx = self.m_nx;
                dmean = with_mrc_data!(self.m_full_cache[li], |d| image_subarea_mean(
                    &d,
                    MRC_MODE_FLOAT,
                    nx,
                    ix_start_read,
                    ix_end_read,
                    iy_start_read,
                    iy_end_read
                ));
                let dump = self.m_single_test_patch >= 0 && ipatch == self.m_single_test_patch;
                if self.m_txc_gpu.as_mut().unwrap().extract_patch(
                    iz_read,
                    iz_read == self.m_iz_last,
                    dmean,
                    ix_start_read,
                    iy_start_read,
                    &self.m_fs,
                    self.m_xpeak_frac,
                    self.m_ypeak_frac,
                    if taper_read { self.m_n_fill_taper } else { 0 },
                    self.m_loaded_fill_vals[li],
                    self.m_good_xstart[good_ind],
                    self.m_good_xend[good_ind],
                    self.m_good_ystart[good_ind],
                    self.m_good_yend[good_ind],
                    dump,
                ) != 0
                {
                    exit_error(b"Failure extracting patch in GPU");
                }
                self.m_wall_load += wall_time() - wall_start;
                return;
            } else if self.m_i_anti_filt_type > 0 && self.m_nbinning > 1 {
                // Otherwise, if doing antialiased reduction, extract that
                ierr = select_zoom_filter(
                    self.m_i_anti_filt_type - 1,
                    1. / self.m_nbinning as f64,
                    &mut out_width,
                );
                if ierr != 0 {
                    exit_error(b"Binning out of range for antialiased reduction");
                }
                // `mFullLinePtrs[loadInd]`: never assigned in the source (see the
                // module comment); these are the `makeLinePointers` lines of the
                // cached image, each viewing from its start to the end of the
                // array as a C line pointer does.
                let nx = self.m_nx as usize;
                let cache = &self.m_full_cache[li];
                let lines: Vec<&[f32]> =
                    (0..self.m_ny as usize).map(|i| &cache[i * nx..]).collect();
                ierr = zoom_with_filter(
                    ZoomLines::Float(&lines),
                    self.m_nx,
                    self.m_ny,
                    ix_start_read as f32,
                    iy_start_read as f32,
                    self.m_nx_use_bin,
                    self.m_ny_use_bin,
                    self.m_nx_use_bin,
                    0,
                    SLICE_MODE_FLOAT,
                    &mut ZoomOut::Float(&mut self.m_array),
                    None,
                    None,
                );
                if ierr != 0 {
                    exit_error_fmt!(
                        "Extracting reduced area of %d x %d at %d, %d in %d by %d full image (error code %d",
                        CArg::Int(self.m_nx_use_bin as i64),
                        CArg::Int(self.m_ny_use_bin as i64),
                        CArg::Int(ix_start_read as i64),
                        CArg::Int(iy_start_read as i64),
                        CArg::Int(self.m_nx as i64),
                        CArg::Int(self.m_ny as i64),
                        CArg::Int(ierr as i64)
                    );
                }
            } else {
                // Or extract with binning
                ierr = extract_with_binning(
                    f32_bytes!(self.m_full_cache[li]),
                    SLICE_MODE_FLOAT,
                    self.m_nx,
                    ix_start_read,
                    ix_end_read,
                    iy_start_read,
                    iy_end_read,
                    self.m_nbinning,
                    f32_bytes_mut!(self.m_array),
                    self.m_nx_use_bin,
                    &mut nxr,
                    &mut nyr,
                );
                if nxr != self.m_nx_use_bin || nyr != self.m_ny_use_bin || ierr != 0 {
                    exit_error_fmt!(
                        "Extracting binned area of %d x %d at %d, %d in %d x %d full image (return value %d, size %d by %d",
                        CArg::Int(self.m_nx_use_bin as i64),
                        CArg::Int(self.m_ny_use_bin as i64),
                        CArg::Int(ix_start_read as i64),
                        CArg::Int(iy_start_read as i64),
                        CArg::Int(self.m_nx as i64),
                        CArg::Int(self.m_ny as i64),
                        CArg::Int(ierr as i64),
                        CArg::Int(nxr as i64),
                        CArg::Int(nyr as i64)
                    );
                }
            }
        } else if self.m_i_anti_filt_type > 0 {
            // full-image tracking, read reduced
            iiu_read_reduced(
                im_unit,
                iz_read,
                &mut self.m_array,
                self.m_nx_use_bin,
                ix_start_read as f32,
                iy_start_read as f32,
                self.m_nbinning as f32,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                self.m_i_anti_filt_type - 1,
                &mut self.m_tmp_array,
                self.m_len_temp as i32,
                &mut ierr,
            );
            if ierr > 0 {
                exit_error_fmt!(
                    "Calling irdReduced to read image (error code %d)",
                    CArg::Int(ierr as i64)
                );
            }
        } else {
            // Or read binned
            iiu_read_binned(
                im_unit,
                iz_read,
                &mut self.m_array,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                ix_start_read,
                iy_start_read,
                self.m_nbinning,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                &mut self.m_tmp_array,
                self.m_len_temp as i32,
                &mut ierr,
            );
        }
        if ierr != 0 {
            exit_error(b"Reading image file");
        }

        // Taper fill area if not on GPU
        if taper_read && self.m_use_gpu < 0 {
            if taper_at_fill(
                &mut self.m_array,
                self.m_nx_use_bin,
                self.m_ny_use_bin,
                self.m_n_fill_taper,
                true,
            ) != 0
            {
                exit_error(b"Getting memory for tapering from fill area in patch");
            }
        }
        self.m_wall_load += wall_time() - wall_start;
    }

    /// `TiltXCorr::getFullArrayAndLines` (`tiltxcorr.cpp:3265`).  Finds the
    /// cache slot for `iz_read`, reading the image in (and loading it to the
    /// GPU) if necessary, and returns its index in `load_ind`.  The source
    /// also returns `*fullPtr = mFullCache[loadInd]`, which callers reach as
    /// `m_full_cache[load_ind]`, and `mFullLinePtrs[loadInd]` (see the module
    /// comment).  This is for patches only.
    fn get_full_array_and_lines(&mut self, im_unit: i32, iz_read: i32, load_ind: &mut i32) {
        let ierr: i32;
        let mut first_free: i32 = -1;
        let mut oldest_use: i32 = 2000000000;
        let mut oldest_ind: i32 = -1;
        let wall_start = wall_time();

        // Look up Z in cache and Find oldest item in cache and first free entry if any
        *load_ind = -1;
        for ind in 0..self.m_cache_size as usize {
            if self.m_iz_loaded[ind] == iz_read {
                *load_ind = ind as i32;
            }
            if self.m_iz_loaded[ind] < 0 && first_free < 0 {
                first_free = ind as i32;
            }
            if self.m_iz_loaded[ind] >= 0 && self.m_load_last_used[ind] < oldest_use {
                oldest_use = self.m_load_last_used[ind];
                oldest_ind = ind as i32;
            }
        }

        if *load_ind < 0 {
            // If not in cache, read in to free or oldest slot
            if first_free >= 0 {
                *load_ind = first_free;
            } else if oldest_ind >= 0 {
                *load_ind = oldest_ind;
            } else {
                exit_error(b"Program error: no free slot in cache and no oldest entry");
            }
            let li = *load_ind as usize;
            self.m_iz_loaded[li] = iz_read;
            unsafe { iiu_set_position(im_unit, iz_read, 0) };
            ierr = unsafe { iiu_read_section(im_unit, self.m_full_cache[li].as_mut_ptr().cast()) };
            if ierr != 0 {
                exit_error_fmt!("Reading image file: %s", CArg::Str(&b3d_get_error()));
            }
            if self.m_use_gpu >= 0 && self.m_txc_gpu.as_ref().unwrap().is_initialized() {
                if self.m_txc_gpu.as_mut().unwrap().load_full_array(
                    iz_read,
                    *load_ind,
                    &self.m_full_cache[li],
                ) != 0
                {
                    exit_error_fmt!(
                        "Trying to load full array to GPU at index %d",
                        CArg::Int(*load_ind as i64)
                    );
                }

                // Get a fill value in shifted prealigned image by averaging the most
                // shifted edge
                if self.m_if_read_xfs != 0 {
                    let izr = iz_read as usize;
                    let (nx, ny) = (self.m_nx, self.m_ny);
                    let (x0, x1, y0, y1) = if self.m_dx_preali[izr] > self.m_dy_preali[izr].abs() {
                        (0, 0, 0, ny - 1)
                    } else if -self.m_dx_preali[izr] > self.m_dy_preali[izr].abs() {
                        (nx - 1, nx - 1, 0, ny - 1)
                    } else if self.m_dy_preali[izr] > 0. {
                        (0, nx - 1, 0, 0)
                    } else {
                        (0, nx - 1, ny - 1, ny - 1)
                    };
                    self.m_loaded_fill_vals[li] = with_mrc_data!(self.m_full_cache[li], |d| {
                        image_subarea_mean(&d, MRC_MODE_FLOAT, nx, x0, x1, y0, y1)
                    });
                }
            }
        }
        self.m_wall_read += wall_time() - wall_start;
        let li = *load_ind as usize;
        self.m_load_last_used[li] = self.m_cache_use_count;
        self.m_cache_use_count += 1;
    }

    /// `TiltXCorr::packCorr` (`tiltxcorr.cpp:3329`).  Packs and mirrors the
    /// cross-correlation for saving.
    fn pack_corr(&mut self, nx_pad_plus: i32) {
        let mut iout: usize = 0;
        let mut ix_in = self.m_nx_pad / 2;
        let mut iy_in = self.m_ny_pad / 2;
        for _iy in 0..self.m_ny_pad {
            for _ix in 0..self.m_nx_pad {
                self.m_crray[iout] = self.m_array[(nx_pad_plus * iy_in + ix_in) as usize];
                iout += 1;
                ix_in += 1;
                if ix_in >= self.m_nx_pad {
                    ix_in = 0;
                }
            }
            iy_in += 1;
            if iy_in >= self.m_ny_pad {
                iy_in = 0;
            }
        }
    }

    /// `TiltXCorr::usablePatchExtent` (`tiltxcorr.cpp:3352`).  Determines the
    /// extent of usable area in a patch in one dimension; note this is 1 or 2
    /// pixels more stringent than in `markUsablePatches`.
    #[allow(clippy::too_many_arguments)]
    fn usable_patch_extent(
        &self,
        nx: i32,
        nx_unali: i32,
        nx_patch: i32,
        dx: f32,
        ix_start: i32,
        nx_usable: &mut i32,
        use_start: Option<&mut i32>,
        use_end: Option<&mut i32>,
    ) {
        let end = b3d_i_min(&[
            nx - 2,
            b3dnint!((nx + nx_unali) as f64 / 2. + dx as f64) - 2,
            ix_start + nx_patch,
        ]);
        let start = b3d_i_max(&[
            2,
            b3dnint!((nx - nx_unali) as f64 / 2. + dx as f64) + 2,
            ix_start,
        ]);
        *nx_usable = end - start;
        if let Some(use_start) = use_start {
            *use_start = if start > ix_start { start } else { -1 };
        }
        if let Some(use_end) = use_end {
            *use_end = if end < ix_start + nx_patch {
                end - 1
            } else {
                -1
            };
        }
    }

    /// `TiltXCorr::evaluatePairPatch` (`tiltxcorr.cpp:3369`).  Determines if a
    /// patch is sufficiently in the area with image when there is a general
    /// transformation applied to the image.
    fn evaluate_pair_patch(
        &mut self,
        ix_start: i32,
        iy_start: i32,
        prexf_inv: &[f32],
        cur_view_out: &mut bool,
    ) {
        let xtmp: f32;
        let ytmp: f32;
        let area: f32;
        //
        // back-transform the 4 corners of the patch
        xtmp = ix_start as f32;
        ytmp = iy_start as f32;
        let (xc, yc) = (self.m_xcen as f32, self.m_ycen as f32);
        let (mut x_lo_lf, y_lo_lf) = xf_apply(prexf_inv, xc, yc, xtmp, ytmp, 2);
        let (mut x_lo_rt, mut y_lo_rt) =
            xf_apply(prexf_inv, xc, yc, xtmp + self.m_nx_patch as f32, ytmp, 2);
        let (mut x_up_lf, mut y_up_lf) =
            xf_apply(prexf_inv, xc, yc, xtmp, ytmp + self.m_ny_patch as f32, 2);
        let (mut x_up_rt, mut y_up_rt) = xf_apply(
            prexf_inv,
            xc,
            yc,
            xtmp + self.m_nx_patch as f32,
            ytmp + self.m_ny_patch as f32,
            2,
        );
        let mut y_lo_lf = y_lo_lf;

        // If they are all in, there is no tapering.  If all out, the patch is out
        let nxm1 = self.m_nx as f64 - 1.;
        let nym1 = self.m_ny as f64 - 1.;
        self.m_taper_cur = (x_lo_lf as f64) < 1.
            || (x_up_lf as f64) < 1.
            || x_lo_rt as f64 > nxm1
            || x_up_rt as f64 > nxm1
            || (y_lo_lf as f64) < 1.
            || (y_up_lf as f64) < 1.
            || y_lo_rt as f64 > nym1
            || y_up_rt as f64 > nym1;
        *cur_view_out = false;
        if !self.m_taper_cur {
            return;
        }
        *cur_view_out = (if x_lo_rt > x_up_rt { x_lo_rt } else { x_up_rt }) <= 0.
            || (if x_lo_lf < x_up_lf { x_lo_lf } else { x_up_lf }) >= self.m_nx as f32
            || (if y_lo_rt > y_up_rt { y_lo_rt } else { y_up_rt }) <= 0.
            || (if y_lo_lf < y_up_lf { y_lo_lf } else { y_up_lf }) >= self.m_ny as f32;
        if *cur_view_out {
            return;
        }

        // Get intersection with the raw image area and compute the area from the vertices
        let max1 = |v: f32| (if 1. > v as f64 { 1. } else { v as f64 }) as f32;
        x_lo_lf = max1(x_lo_lf);
        x_up_lf = max1(x_up_lf);
        y_lo_lf = max1(y_lo_lf);
        y_lo_rt = max1(y_lo_rt);
        let minx = |v: f32| (if nxm1 < v as f64 { nxm1 } else { v as f64 }) as f32;
        let miny = |v: f32| (if nym1 < v as f64 { nym1 } else { v as f64 }) as f32;
        x_lo_rt = minx(x_lo_rt);
        x_up_rt = minx(x_up_rt);
        y_up_lf = miny(y_up_lf);
        y_up_rt = miny(y_up_rt);
        area = (((x_lo_lf * y_lo_rt - y_lo_lf * x_lo_rt)
            + (x_lo_rt * y_up_rt - y_lo_rt * x_up_rt)
            + (x_up_rt * y_up_lf - y_up_rt * x_up_lf)
            + (x_up_lf * y_lo_lf - y_up_lf * x_lo_lf)) as f64
            / 2.) as f32;
        *cur_view_out =
            area < self.m_crit_non_blank * self.m_nx_patch as f32 * self.m_ny_patch as f32;
    }

    /// `TiltXCorr::checkForWarpFile` (`tiltxcorr.cpp:3411`).  Checks for a
    /// warp file and exits with an error if so.
    fn check_for_warp_file(&self, xf_in_file: &str) {
        let mut strntmp = String::new();
        let (mut idx, mut idy, mut itmp, mut i, mut jj) = (0, 0, 0, 0, 0);
        let mut deltac: f32 = 0.;
        let ierr = read_check_warp_file(
            xf_in_file,
            0,
            0,
            &mut idx,
            &mut idy,
            &mut itmp,
            &mut i,
            &mut deltac,
            &mut jj,
            &mut strntmp,
        );
        if ierr >= 0 {
            exit_error(b"The prealign transform file contains warping transforms");
        }
        if ierr != -1 {
            exit_error_fmt!(
                "ERROR: A problem occurred testing whether the initial transform file had warpings: %s",
                CArg::Str(&strntmp)
            );
        }
    }

    /// `TiltXCorr::anglesPassThrough` (`tiltxcorr.cpp:3427`).  Determines
    /// whether tilt angles pass through 0, 90, and -90 with the given increment.
    fn angles_pass_through(
        &self,
        first: f32,
        test_inc: f32,
        nz: i32,
        pass0: &mut bool,
        pass90: &mut bool,
        pass_min90: &mut bool,
    ) {
        let mut tilt: f32;
        let mut tilt_last: f32;
        *pass0 = false;
        *pass90 = false;
        *pass_min90 = false;
        tilt_last = first;
        for ind in 1..=nz - 1 {
            tilt = first + ind as f32 * test_inc;
            let lo = if tilt_last < tilt { tilt_last } else { tilt };
            let hi = if tilt > tilt_last { tilt } else { tilt_last };
            if lo <= 0. && hi > 0. {
                *pass0 = true;
            }
            if lo <= 90. && hi > 90. {
                *pass90 = true;
            }
            if lo <= -90. && hi > -90. {
                *pass_min90 = true;
            }
            tilt_last = tilt;
        }
    }

    /// `TiltXCorr::allocatePatchArrays` (`tiltxcorr.cpp:3450`).  Resizes the
    /// arrays for patches for patch tracking or warping.
    fn allocate_patch_arrays(&mut self) {
        let ix = (self.m_num_patches_all + 10) as usize;
        self.m_patch_cen_x.resize(ix, 0.);
        self.m_patch_cen_y.resize(ix, 0.);
        self.m_patch_cen_xall.resize(ix, 0.);
        self.m_patch_cen_yall.resize(ix, 0.);
        self.m_patch_nx.resize(ix, 0);
        self.m_patch_ny.resize(ix, 0);
        self.m_patch_xinds.resize(ix, 0);
        self.m_patch_yinds.resize(ix, 0);
        self.m_patch_domains.resize(ix, 0);
    }

    /// `TiltXCorr::allocateFullCache` (`tiltxcorr.cpp:3466`).  Gets the arrays
    /// for the cache of full images.
    fn allocate_full_cache(&mut self, start_ind: i32, size: i32) {
        for ind in start_ind..size {
            self.m_full_cache[ind as usize] = vec![0.; (self.m_nx * self.m_ny) as usize];
        }
    }

    /// `TiltXCorr::allocateMainArrays` (`tiltxcorr.cpp:3477`).  Gets the arrays
    /// for correlations.
    fn allocate_main_arrays(&mut self, reuse_prev_arrays: bool) {
        let idim2 = (self.m_nx_pad + 2) * self.m_ny_pad + 16;
        if idim2 <= self.m_main_array_size {
            return;
        }
        let n = idim2 as usize;
        self.m_sum_array = vec![0.; n];
        self.m_array = vec![0.; n];
        self.m_crray = vec![0.; n];
        self.m_brray = vec![0.; n];
        if reuse_prev_arrays {
            self.m_arr_copy = vec![0.; n];
        }
        self.m_main_array_size = idim2;
    }

    /// `TiltXCorr::setupDomains` (`tiltxcorr.cpp:3507`).  Gets the limits of
    /// the domains used for analyzing predicted positions within subareas for
    /// one axis.  Given the number of patches and size of domains, returns the
    /// number of domains, the starting patch index of each domain, and how
    /// patches are assigned to domains for potential position replacement or
    /// elimination.
    fn setup_domains(
        &self,
        num_patch: i32,
        domain_size: i32,
        num_domains: &mut i32,
        domain_starts: &mut Vec<i32>,
        patch_assigns: &mut Vec<i32>,
    ) {
        let mut start: i32;
        let mut max_size: i32;
        let mut biggest: usize = 0;
        let mut which_end: usize = 0;
        let mut inner_starts: Vec<i32> = Vec::new();
        let mut inner_ends: Vec<i32> = Vec::new();

        if num_patch <= domain_size {
            *num_domains = 1;
            domain_starts.push(0);
            domain_starts.push(num_patch);
            patch_assigns.resize(num_patch as usize, 0);
            return;
        }

        // Number of domains is the number of patches in the interior (thus- 2) divided by
        // the interior size or the domains, with the usual rounding up if not evenly
        // dividing
        *num_domains = ((num_patch - 2) + domain_size - 3) / (domain_size - 2);

        // Patches in overlap are total # of patches in domains minus # of patched
        let total_overlap = domain_size * *num_domains - num_patch;

        // Figure out how to divide up the overlaps
        let base_overlap = total_overlap / (*num_domains - 1);
        let mut overlap_rem = total_overlap % (*num_domains - 1);
        patch_assigns.resize(num_patch as usize, 0);

        // Set up the domain starts
        start = 0;
        if self.m_verbose != 0 {
            print_vals!(
                "numDomains" => cout_i(*num_domains),
                "totalOverlap" => cout_i(total_overlap),
                "baseOverlap" => cout_i(base_overlap),
                "overlapRem" => cout_i(overlap_rem)
            );
            printf!("domain starts: ");
        }
        for _ind in 0..*num_domains {
            domain_starts.push(start);
            if self.m_verbose != 0 {
                printf!(" %d", CArg::Int(start as i64));
            }
            start += domain_size - base_overlap;
            if overlap_rem > 0 {
                start -= 1;
                overlap_rem -= 1;
            }
        }
        domain_starts.push(num_patch);
        if self.m_verbose != 0 {
            printf!("\n");
        }

        // Set up ideal inner ranges
        start = 0;
        for ind in 0..*num_domains as usize {
            if ind != 0 {
                start = domain_starts[ind] + 1;
            }
            inner_starts.push(start);
            if (ind as i32) < *num_domains - 1 {
                inner_ends.push(domain_starts[ind] + domain_size - 2);
            } else {
                inner_ends.push(num_patch - 1);
            }
            if self.m_verbose != 0 {
                printf!(
                    "%d %d %d\n",
                    CArg::Int(ind as i64),
                    CArg::Int(start as i64),
                    CArg::Int(inner_ends[ind] as i64)
                );
            }
        }

        // Iteratively resolve overlaps by taking away from the biggest domain with overlap
        loop {
            max_size = 0;
            for ind in 0..(*num_domains - 1) as usize {
                if inner_ends[ind] >= inner_starts[ind + 1] {
                    for jnd in 0..=1 {
                        if inner_ends[ind + jnd] + 1 - inner_starts[ind + jnd] > max_size {
                            max_size = inner_ends[ind + jnd] + 1 - inner_starts[ind + jnd];
                            biggest = ind + jnd;
                            which_end = jnd;
                        }
                    }
                }
            }
            if max_size == 0 {
                break;
            }
            if which_end != 0 {
                inner_starts[biggest] += 1;
            } else {
                inner_ends[biggest] -= 1;
            }
        }

        // Now assign
        if self.m_verbose != 0 {
            printf!("assigns: ");
        }
        for ind in 0..*num_domains as usize {
            for jnd in inner_starts[ind]..=inner_ends[ind] {
                patch_assigns[jnd as usize] = ind as i32;
                if self.m_verbose != 0 {
                    printf!(" %d", CArg::Int(ind as i64));
                }
            }
        }
        if self.m_verbose != 0 {
            printf!("\n");
        }
    }

    /// `TiltXCorr::setBinnedSizes` (`tiltxcorr.cpp:3610`).  Determines the
    /// binned area size, padded and tapering sizes given the binning for
    /// correlation, and also gets some unbinned values for fixed noise peak
    /// rejection.
    fn set_binned_sizes(
        &mut self,
        pad_frac: f32,
        taper_frac: f32,
        ix_cen_end: &mut i32,
        iy_cen_end: &mut i32,
    ) {
        let nx_border = 5.max(b3dnint!(pad_frac * self.m_nx_use as f32));
        let ny_border = 5.max(b3dnint!(pad_frac * self.m_ny_use as f32));
        self.m_nx_use_bin = self.m_nx_use / self.m_nbinning;
        self.m_ny_use_bin = self.m_ny_use / self.m_nbinning;

        self.m_nx_pad = nice_frame(
            (self.m_nx_use + 2 * nx_border) / self.m_nbinning,
            2,
            self.m_nice_limit,
        );
        self.m_ny_pad = nice_frame(
            (self.m_ny_use + 2 * ny_border) / self.m_nbinning,
            2,
            self.m_nice_limit,
        );
        //
        // Set up tapering, save unbinned values
        //
        self.m_nx_taper = 5.max(b3dnint!(taper_frac * self.m_nx_use as f32));
        self.m_ny_taper = 5.max(b3dnint!(taper_frac * self.m_ny_use as f32));
        self.m_nx_ub_taper = self.m_nx_taper;
        self.m_ny_ub_taper = self.m_ny_taper;
        self.m_nx_taper /= self.m_nbinning;
        self.m_ny_taper /= self.m_nbinning;
        //
        // get starting and ending coordinates of a centered patch of this size, to which
        // box offsets will be added for loading
        //
        self.m_ix_cen_start = (self.m_nx - self.m_nx_use) / 2;
        *ix_cen_end = self.m_ix_cen_start + self.m_nx_use - 1;
        self.m_iy_cen_start = (self.m_ny - self.m_ny_use) / 2;
        *iy_cen_end = self.m_iy_cen_start + self.m_ny_use - 1;
    }

    /// `TiltXCorr::setMainCTF` (`tiltxcorr.cpp:3642`).  Fills the main CTF
    /// filter array given the filter parameters and padded size.
    fn set_main_ctf(
        &mut self,
        sigma1: f32,
        sigma2: f32,
        radius1: f32,
        radius2: f32,
        eval_ccc: i32,
    ) {
        xcorr_set_ctf(
            sigma1,
            sigma2,
            radius1,
            radius2,
            &mut self.m_ctfp,
            self.m_nx_pad,
            self.m_ny_pad,
            &mut self.m_delta_ctf,
        );

        // If doing CCC's, take square root of filter to apply it to both images
        if eval_ccc != 0 {
            for ix in 0..8193 {
                self.m_ctfp[ix] = (self.m_ctfp[ix] as f64).sqrt() as f32;
            }
        }
    }

    /// `TiltXCorr::getFilterValueAsPixOrNm` (`tiltxcorr.cpp:3657`).  Uses an
    /// option to get a filter parameter in reciprocal pixels, then checks the
    /// option for periodicity in nm and converts the latter to reciprocal
    /// pixels given the binning values.
    fn get_filter_value_as_pix_or_nm(&self, pix_opt: &str, nm_opt: &str, value: &mut f32) {
        let ierr = pip_get_float(pix_opt.as_bytes(), value);
        if pip_get_float(nm_opt.as_bytes(), value) == 0 {
            if ierr == 0 {
                exit_error_fmt!(
                    "You cannot enter both -%s and -%s",
                    CArg::Str(pix_opt),
                    CArg::Str(nm_opt)
                );
            }
            if self.m_unbinned_pixel == 0. {
                exit_error_fmt!(
                    "The unbinned pixel size must be entered to specify a filter value with %s",
                    CArg::Str(nm_opt)
                );
            }
            *value *= (self.m_nbinning * self.m_images_binned) as f32 * self.m_unbinned_pixel;
        }
    }

    /// `TiltXCorr::getTwoIntsBinnedOrUnbinned` (`tiltxcorr.cpp:3675`).  Uses an
    /// option to get two binned size/position parameters and then checks the
    /// option for the unbinned form and converts to binned.  Returns 1 if
    /// binned option, 2 if unbinned option, 0 if no entry.
    fn get_two_ints_binned_or_unbinned(
        &self,
        bin_opt: &str,
        ub_opt: &str,
        val1: &mut i32,
        val2: &mut i32,
    ) -> i32 {
        let mut ierr = 1 - pip_get_two_integers(bin_opt.as_bytes(), val1, val2);
        if pip_get_two_integers(ub_opt.as_bytes(), val1, val2) == 0 {
            if ierr != 0 {
                exit_error_fmt!(
                    "You cannot enter both -%s and -%s",
                    CArg::Str(bin_opt),
                    CArg::Str(ub_opt)
                );
            }
            ierr = 2;
            *val1 /= self.m_images_binned;
            *val2 /= self.m_images_binned;
        }
        ierr
    }

    /// `TiltXCorr::setSdMapLimits` (`tiltxcorr.cpp:3691`).  Sets up the limits
    /// for the SD map.
    fn set_sd_map_limits(&mut self, min_expand_border: i32, dxy_ind: i32) {
        self.m_sd_xstart = min_expand_border;
        self.m_sd_ystart = min_expand_border;
        self.m_sd_xend = self.m_nx - min_expand_border - 1;
        self.m_sd_yend = self.m_ny - min_expand_border - 1;
        if self.m_if_read_xfs != 0 {
            let dx = self.m_dx_preali[dxy_ind as usize];
            let dy = self.m_dy_preali[dxy_ind as usize];
            // `int += double`: the sum is formed in double and truncated.
            if dx > 0. {
                self.m_sd_xstart = (self.m_sd_xstart as f64 + (dx as f64 + 1.)) as i32;
            } else {
                self.m_sd_xend = (self.m_sd_xend as f64 + (dx as f64 - 1.)) as i32;
            }
            if dy > 0. {
                self.m_sd_ystart = (self.m_sd_ystart as f64 + (dy as f64 + 1.)) as i32;
            } else {
                self.m_sd_yend = (self.m_sd_yend as f64 + (dy as f64 - 1.)) as i32;
            }
        }
    }

    /// `TiltXCorr::getPatchSdStats` (`tiltxcorr.cpp:3713`).  Gets an SD map at
    /// the given binning and computes the mean and SD for all patches at the
    /// global size and gets some statistics.  `full_ind` is the cache slot the
    /// source's `fullArr` points into.
    #[allow(clippy::too_many_arguments)]
    fn get_patch_sd_stats(
        &mut self,
        sd_bin: i32,
        _ibin: i32,
        full_ind: usize,
        sd_num_pix: &mut [i32],
        sd_means: &mut [f32],
        sd_xoff: &mut i32,
        sd_yoff: &mut i32,
        min_sd: &mut f32,
        pct1_sd: &mut f32,
    ) {
        let wall_start = wall_time();
        let nx_sd = (self.m_sd_xend + 1 - self.m_sd_xstart) / sd_bin;
        let ny_sd = (self.m_sd_yend + 1 - self.m_sd_ystart) / sd_bin;
        let mut sd_num_samp: i32 = 0;
        make_standard_dev_map(
            &self.m_full_cache[full_ind],
            self.m_nx,
            self.m_sd_xstart,
            -self.m_sd_xend,
            self.m_sd_ystart,
            self.m_sd_yend,
            -sd_bin,
            self.m_sd_box_reduced,
            &mut self.m_sd_arr,
            &mut self.m_sum_arr,
            &mut self.m_sqr_arr,
            sd_xoff,
            sd_yoff,
        );
        self.sigmoid_scale_sd_map(sd_bin);
        for ipatch in 0..self.m_num_patches as usize {
            let mut px_start = b3dnint!(self.m_patch_cen_x[ipatch]) - self.m_nx_patch / 2;
            let px_end = (px_start + self.m_nx_patch - 1).min(self.m_sd_xend);
            px_start = px_start.max(self.m_sd_xstart);
            let mut py_start = b3dnint!(self.m_patch_cen_y[ipatch]) - self.m_ny_patch / 2;
            let py_end = (py_start + self.m_ny_patch - 1).min(self.m_sd_yend);
            py_start = py_start.max(self.m_sd_ystart);
            let ix = (px_end + 1 - px_start) / sd_bin;
            let iy = (py_end + 1 - py_start) / sd_bin;
            sd_num_pix[ipatch] = ix * iy;
            array_min_max_mean(
                &self.m_sd_arr,
                nx_sd,
                ny_sd,
                px_start / sd_bin + *sd_xoff,
                px_end / sd_bin + *sd_xoff,
                py_start / sd_bin + *sd_yoff,
                py_end / sd_bin + *sd_yoff,
                &mut self.m_dmin2,
                &mut self.m_dmax2,
                &mut sd_means[ipatch],
            );
            *min_sd = if *min_sd < self.m_dmin2 {
                *min_sd
            } else {
                self.m_dmin2
            };
            get_sample_of_array(
                f32_bytes!(self.m_sd_arr),
                MRC_MODE_FLOAT,
                nx_sd,
                ny_sd,
                1.,
                px_start / sd_bin + *sd_xoff,
                py_start / sd_bin + *sd_yoff,
                ix,
                iy,
                -1.0e30,
                &mut self.m_sum_arr,
                10000,
                &mut sd_num_samp,
            );
            self.m_dmin2 = percentile_float(
                b3dnint!(0.01 * sd_num_samp as f64),
                &mut self.m_sum_arr,
                sd_num_samp,
            );
            *pct1_sd = if *pct1_sd < self.m_dmin2 {
                *pct1_sd
            } else {
                self.m_dmin2
            };
        }
        self.m_wall_sd_stat += wall_time() - wall_start;
    }

    /// `TiltXCorr::sigmoidScaleSdMap` (`tiltxcorr.cpp:3758`).  Scales an SD map
    /// by a sigmoid; the higher the power above 1, the sharper the rise.
    fn sigmoid_scale_sd_map(&mut self, sd_bin: i32) {
        let nx_sd = (self.m_sd_xend + 1 - self.m_sd_xstart) / sd_bin;
        let ny_sd = (self.m_sd_yend + 1 - self.m_sd_ystart) / sd_bin;
        let mut xx: f32;
        let mut sd_max: f32 = 0.;
        let mut num_samples: i32 = 0;
        let border = self.m_sd_box_reduced / 2 + 1;
        const MAX_SAMPLE: usize = 10000;
        let mut sample = vec![0f32; MAX_SAMPLE];
        let mut peak_above: f32 = 0.;
        let mut use_rise: f32;
        let mut madn_mode: f32 = 0.;
        let lower_peak_min_ratio: f32 = 0.5;

        if self.m_sigmoid_power == 1. && self.m_sigmoid_half_rise == 0.5 {
            return;
        }
        for ind in 0..(nx_sd * ny_sd) as usize {
            sd_max = if sd_max > self.m_sd_arr[ind] {
                sd_max
            } else {
                self.m_sd_arr[ind]
            };
        }
        use_rise = self.m_sigmoid_half_rise;
        if self.m_sigmoid_half_rise < 0. {
            if get_sample_of_array(
                f32_bytes!(self.m_sd_arr),
                SLICE_MODE_FLOAT,
                nx_sd,
                ny_sd,
                1.,
                border,
                border,
                nx_sd - 2 * border,
                ny_sd - 2 * border,
                -1.,
                &mut sample,
                MAX_SAMPLE as i32,
                &mut num_samples,
            ) != 0
            {
                return;
            }

            self.find_histogram_mode(
                &sample[..num_samples.max(0) as usize],
                num_samples,
                0.05,
                lower_peak_min_ratio,
                &mut peak_above,
                Some(&mut madn_mode),
            );
            use_rise = (peak_above + self.m_sigmoid_half_rise * madn_mode) / sd_max;
            if self.m_verbose != 0 {
                print_vals!(
                    "sdMax" => cout_g(sd_max as f64),
                    "peakAbove" => cout_g(peak_above as f64),
                    "MADMode" => cout_g(madn_mode as f64),
                    "useRise" => cout_g(use_rise as f64)
                );
            }
            if (use_rise as f64) < 0.01 {
                return;
            }
        }

        for ind in 0..(nx_sd * ny_sd) as usize {
            xx = self.m_sd_arr[ind] / sd_max;
            if xx as f64 > 1.0e-4 && (xx as f64) < 0.99999 {
                xx = 1.0f32
                    / (1.0f32
                        + ((use_rise / xx) * (1.0f32 - xx) / (1.0f32 - use_rise))
                            .powf(self.m_sigmoid_power));
            }
            self.m_sd_arr[ind] = xx * sd_max;
        }
    }

    /// `TiltXCorr::findHistogramMode` (`tiltxcorr.cpp:3801`).  Finds the mode
    /// of a kernel histogram with the given h value: the highest peak, or the
    /// second highest if it is at a higher position and strong enough.
    fn find_histogram_mode(
        &self,
        values: &[f32],
        num_val: i32,
        h_frac: f32,
        lower_peak_min_ratio: f32,
        mode: &mut f32,
        madn_mode: Option<&mut f32>,
    ) {
        const MAX_BINS: i32 = 500;
        let mut hist = [0f32; MAX_BINS as usize];
        let mut val_min: f32 = 1.0e30;
        let mut val_max: f32 = -1.0e30;
        let mut ind: i32;
        let mut ind_max: i32 = 0;
        let mut temp: Vec<f32> = Vec::new();
        let mut hist_max: f32 = -1.;
        if madn_mode.is_some() {
            temp.resize(num_val as usize, 0.);
        }
        for i in 0..num_val as usize {
            val_min = if val_min < values[i] {
                val_min
            } else {
                values[i]
            };
            val_max = if val_max > values[i] {
                val_max
            } else {
                values[i]
            };
        }

        kernel_histogram(
            &values[..num_val as usize],
            &mut hist,
            val_min,
            val_max,
            h_frac * (val_max - val_min),
            0,
        );
        for i in 0..MAX_BINS {
            if hist[i as usize] > hist_max {
                hist_max = hist[i as usize];
                ind_max = i;
            }
        }

        *mode = val_min;
        if let Some(result) = scan_histogram(&hist, val_min, val_max, val_min, val_max, true) {
            *mode = result.peak_above;
        }
        ind = ((*mode - val_min) * MAX_BINS as f32 / (val_max - val_min)) as i32;
        ind = 0.max((MAX_BINS - 1).min(ind));
        if ind > ind_max && hist[ind as usize] > lower_peak_min_ratio * hist_max {
            ind_max = ind;
        }
        *mode = (val_min as f64
            + (ind_max as f64 + 0.5) * (val_max - val_min) as f64 / MAX_BINS as f64)
            as f32;
        if let Some(madn_mode) = madn_mode {
            rs_fast_madn(values, num_val, *mode, &mut temp, madn_mode);
        }
    }

    /// `TiltXCorr::doPredictionFits` (`tiltxcorr.cpp:3843`).  Once all patches
    /// are tracked, gets a prediction of where patches should be on the
    /// current view and fits the found points to that prediction with
    /// identification of probable outliers.  Alternative peaks are used if
    /// they are better, and otherwise the outliers are eliminated.
    fn do_prediction_fits(
        &mut self,
        iv_cur: i32,
        loop_dir: i32,
        xmodel: &mut [f32],
        ymodel: &mut [f32],
    ) -> i32 {
        let pmf = self.m_pred_max_fit as usize;
        let np = self.m_num_patches as usize;
        let mut xx = vec![0f32; pmf];
        let mut yy = vec![0f32; pmf];
        let mut zz = vec![0f32; pmf];
        let mut xrot = vec![0f32; pmf];
        let mut dev_vec = vec![0f32; np];
        let mut eval_dev = vec![0f32; np];
        let mut yrot = vec![0f32; pmf];
        let mut pred_x: Vec<f32> = Vec::new();
        let mut pred_y: Vec<f32> = Vec::new();
        let mut drop_val = vec![0f32; np];
        let mut work_xf = vec![0f32; 2 * np];
        let mut views = vec![0i32; pmf];
        let mut pred_ind: Vec<i32> = Vec::new();
        let iv_ref = iv_cur - loop_dir;
        let mut del_view: i32;
        let mut ind: i32;
        let mut mfit: usize;
        let mut num_points: i32;
        let mut iview: i32;
        let mut if_trans: i32 = 0;
        let mut if_rot_trans: i32 = 0;
        let mut num_replaced: i32 = 0;
        let mut iter: i32;
        let mut min_peak: i32;
        let mut ipt: usize;
        let mut num_points_all: i32;
        let num_fit_iters = 10;
        let mut xform = [0f32; 6];
        let mut tilt_range: f32;
        let mut theta: f32;
        let mut xpred: f32 = 0.;
        let mut ypred: f32 = 0.;
        let mut xtmp: f32;
        let ytmp: f32;
        let mut slope: f32 = 0.;
        let mut bint: f32 = 0.;
        let mut ro: f32 = 0.;
        let mut aa: f32 = 0.;
        let mut bb: f32 = 0.;
        let mut cons: f32 = 0.;
        let mut dev_avg: f32 = 0.;
        let mut dev_sd: f32 = 0.;
        let mut dev_min: f32;
        let mut dev: f32;
        let mut dev_max: f32 = 0.;
        let mut fit_avg_mean: f32 = 0.;
        let mut fit_max_dev: f32 = 0.;
        let cur_tilt = self.m_tilt[(iv_cur - 1) as usize];
        let max_frac_drop: f32 = 0.17;
        let mut max_drop: i32;
        let min_after_drop = 4;
        let nall = self.m_num_patches_all as isize;
        let mind =
            |ip: usize, ivw: i32| -> usize { (ip as isize + nall * (ivw as isize - 1)) as usize };

        num_points_all = 0;
        pred_ind.resize(np, -1);

        // Loop on patches to get predicted and actual position

        for ipatch in 0..np {
            if xmodel[mind(ipatch, iv_cur)] < -9999. {
                continue;
            }
            mfit = 0;
            del_view = 0;
            tilt_range = 0.;

            // Find positions in reference and put them in arrays
            while mfit < pmf {
                iview = iv_ref + del_view;
                if iview < 1 || iview > self.m_iz_end {
                    break;
                }
                del_view -= loop_dir;
                let indm = mind(ipatch, iview);
                if xmodel[indm] < -9999. {
                    continue;
                }
                let d = ((self.m_tilt[(iview - 1) as usize] - cur_tilt) as f64).abs();
                tilt_range = if tilt_range as f64 > d {
                    tilt_range
                } else {
                    d as f32
                };
                views[mfit] = iview;
                xx[mfit] = xmodel[indm] - self.m_xcen as f32;
                yy[mfit] = ymodel[indm] - self.m_ycen as f32;
                zz[mfit] = del_view as f32;
                mfit += 1;
            }

            if mfit < 2 {
                continue;
            }

            // Follow approach in Beadtrack here, rotate to axis and ft to sines and cosines
            // if the tilt range is big enough
            if mfit as i32 >= self.m_pred_min_fit + 2
                && tilt_range >= self.m_pred_min_tilt_range as f32
            {
                for i in 0..mfit {
                    xrot[i] = self.m_cos_phi * xx[i] + self.m_sin_phi * yy[i];
                    yrot[i] = -self.m_sin_phi * xx[i] + self.m_cos_phi * yy[i];
                    theta = self.m_tilt[(views[i] - 1) as usize];
                    xx[i] = cosd!(theta);
                    yy[i] = sind!(theta);
                }
                ls_fit(&zz, &yrot, mfit as i32, &mut slope, &mut bint, &mut ro);
                let ytmp_l = bint;
                ls_fit2(
                    &xx,
                    &yy,
                    &xrot,
                    mfit as i32,
                    &mut aa,
                    &mut bb,
                    Some(&mut cons),
                );
                let xtmp_l = aa * cosd!(cur_tilt) + bb * sind!(cur_tilt) + cons;
                xpred = self.m_cos_phi * xtmp_l - self.m_sin_phi * ytmp_l;
                ypred = self.m_sin_phi * xtmp_l + self.m_cos_phi * ytmp_l;
            } else {
                // Otherwise operate with adjusted coordinates and either average or do line
                // fit
                for i in 0..mfit {
                    let (xf, yf) = (xx[i], yy[i]);
                    let (mut xo, mut yo) = (0., 0.);
                    self.adjust_coord(
                        self.m_tilt[(views[i] - 1) as usize],
                        cur_tilt,
                        xf,
                        yf,
                        &mut xo,
                        &mut yo,
                        false,
                    );
                    xx[i] = xo;
                    yy[i] = yo;
                }

                if (mfit as i32) < self.m_pred_min_fit {
                    xpred = 0.;
                    ypred = 0.;
                    for i in 0..mfit {
                        xpred += xx[i] / mfit as f32;
                        ypred += yy[i] / mfit as f32;
                    }
                } else {
                    ls_fit(&zz, &xx, mfit as i32, &mut slope, &mut xpred, &mut ro);
                    ls_fit(&zz, &yy, mfit as i32, &mut slope, &mut ypred, &mut ro);
                }
            }
            pred_x.push(xpred);
            pred_y.push(ypred);
            pred_ind[ipatch] = num_points_all;
            num_points_all += 1;
        }

        // Error return if too few points
        if num_points_all < self.m_pf_trans_min_pts {
            return -2;
        }

        // Set up kind of transform
        if num_points_all < self.m_pf_rot_trans_min_pts {
            if_trans = 1;
        } else if num_points_all < self.m_pf_lin_xf_min_pts {
            if_rot_trans = if num_points_all > self.m_pf_rot_trans_min_pts {
                2
            } else {
                1
            };
        }
        let _ = (if_trans, if_rot_trans);

        // Iterate
        iter = 0;
        while iter < num_fit_iters {
            num_replaced = 0;
            fit_avg_mean = 0.;
            fit_max_dev = 0.;

            // Loop on domains within each iteration
            for dom in 0..self.m_num_xdomains * self.m_num_ydomains {
                let iy = (dom / self.m_num_xdomains) as usize;
                let ix = (dom % self.m_num_xdomains) as usize;
                num_points = 0;

                // Load the data matrix with patches within this domain
                for ipatch in 0..np {
                    ipt = mind(ipatch, iv_cur);
                    if xmodel[ipt] > -9999.
                        && self.m_patch_xinds[ipatch] >= self.m_domain_xstarts[ix]
                        && self.m_patch_xinds[ipatch]
                            < self.m_domain_xstarts[ix] + self.m_domain_xsize
                        && self.m_patch_yinds[ipatch] >= self.m_domain_ystarts[iy]
                        && self.m_patch_yinds[ipatch]
                            < self.m_domain_ystarts[iy] + self.m_domain_ysize
                        && pred_ind[ipatch] >= 0
                    {
                        // Load data into matrix, centered coordinates
                        let b = (self.m_pair_cols * num_points) as usize;
                        num_points += 1;
                        self.m_pair_mat[b] = pred_x[pred_ind[ipatch] as usize];
                        self.m_pair_mat[b + 1] = pred_y[pred_ind[ipatch] as usize];
                        self.m_pair_mat[b + 2] = xmodel[ipt] - self.m_xcen as f32;
                        self.m_pair_mat[b + 3] = ymodel[ipt] - self.m_ycen as f32;
                        self.m_pair_mat[b + 4] = 1.;
                        self.m_pair_mat[b + 5] = ipatch as f32;
                    }
                }

                if num_points < self.m_pf_trans_min_pts {
                    continue;
                }
                max_drop = 0.max(num_points - min_after_drop);
                {
                    let v = num_points as f32 * max_frac_drop;
                    max_drop = (if (max_drop as f32) < v {
                        max_drop as f32
                    } else {
                        v
                    }) as i32;
                }

                if_trans = 0;
                if_rot_trans = 0;
                if num_points - max_drop < self.m_pf_rot_trans_min_pts {
                    if_trans = 1;
                } else if num_points - max_drop < self.m_pf_lin_xf_min_pts {
                    if_rot_trans = if num_points - max_drop > self.m_pf_rot_trans_min_pts {
                        2
                    } else {
                        1
                    };
                }
                ind = Self::find_xf_without_outliers(
                    &mut self.m_pair_mat,
                    self.m_pair_cols,
                    &mut self.m_xmat,
                    self.m_mat_cols,
                    num_points,
                    0.,
                    0.,
                    if_trans,
                    if_rot_trans,
                    max_drop,
                    self.m_min_dev_for_elim,
                    self.m_pred_crit_prob,
                    self.m_abs_prob_crit,
                    &mut work_xf,
                    &mut xform,
                    &mut dev_avg,
                    &mut dev_sd,
                    &mut dev_max,
                );
                fit_avg_mean += dev_avg / (self.m_num_xdomains * self.m_num_ydomains) as f32;
                fit_max_dev = if fit_max_dev > dev_max {
                    fit_max_dev
                } else {
                    dev_max
                };

                // Error means break and do elimination round
                if ind != 0 {
                    num_replaced = -1;
                    printf!(
                        "Error %d finding transform between predicted and found positions on iteration %d",
                        CArg::Int(ind as i64),
                        CArg::Int(iter as i64 + 1)
                    );
                    break;
                }

                if self.m_verbose != 0
                    && (self.m_single_test_patch < 0
                        || self.m_patch_domains[self.m_single_test_patch as usize] == dom)
                {
                    printf!(
                        "dom %d n=%d xform %.4f %.4f %.4f %.4f %.1f %.1f dev avg, sd, max %.2f %.2f %.2f\n",
                        CArg::Int(dom as i64),
                        CArg::Int(num_points as i64),
                        CArg::Dbl(xform[0] as f64),
                        CArg::Dbl(xform[1] as f64),
                        CArg::Dbl(xform[2] as f64),
                        CArg::Dbl(xform[3] as f64),
                        CArg::Dbl(xform[4] as f64),
                        CArg::Dbl(xform[5] as f64),
                        CArg::Dbl(dev_avg as f64),
                        CArg::Dbl(dev_sd as f64),
                        CArg::Dbl(dev_max as f64)
                    );
                }

                // Get deviations from the work array
                for i in 0..num_points as usize {
                    dev_vec[i] = work_xf[i];
                }

                // Loop on points, testing for ones above criterion for evaluating
                for indp in 0..num_points as usize {
                    let pc = self.m_pair_cols as usize;
                    let ipatch = b3dnint!(self.m_pair_mat[indp * pc + 5]) as usize;
                    if self.m_patch_domains[ipatch] != dom {
                        continue;
                    }
                    drop_val[ipatch] = self.m_pair_mat[indp * pc + 4];
                    eval_dev[ipatch] = dev_vec[indp];
                    if drop_val[ipatch] < 0.
                        && self.m_verbose != 0
                        && (self.m_single_test_patch < 0
                            || self.m_single_test_patch == ipatch as i32)
                    {
                        printf!(
                            "dropped ind  %d patch %d  dev %.2f\n",
                            CArg::Int(indp as i64),
                            CArg::Int(ipatch as i64),
                            CArg::Dbl(dev_vec[indp] as f64)
                        );

                        // If above criterion, transform predicted position, add center back,
                        // compare with peak in the saved arrays
                        let (xt, yt) = xf_apply(
                            &xform,
                            0.,
                            0.,
                            self.m_pair_mat[indp * pc],
                            self.m_pair_mat[indp * pc + 1],
                            2,
                        );
                        xtmp = xt + self.m_xcen as f32;
                        let ytmp_v = yt + self.m_ycen as f32;
                        dev_min = 1.0e20;
                        min_peak = -1;
                        for ipeak in 0..self.m_model_xvecs[ipatch].len() {
                            dev = ((self.m_model_xvecs[ipatch][ipeak] - xtmp).powf(2.0f32)
                                + (self.m_model_yvecs[ipatch][ipeak] - ytmp_v).powf(2.0f32))
                            .sqrt();
                            if dev / dev_vec[indp] <= self.m_pf_better_dev_crit
                                && self.m_peak_vecs[ipatch][ipeak] / self.m_peak_vecs[ipatch][0]
                                    >= self.m_pf_min_closer_peak_ratio
                                && dev < dev_min
                            {
                                dev_min = dev;
                                min_peak = ipeak as i32;
                            }
                        }

                        // If found a better point, replace in model array and data matrix
                        // (centered)
                        if min_peak >= 0 {
                            let mp = min_peak as usize;
                            ipt = mind(ipatch, iv_cur);
                            if self.m_verbose != 0
                                && (self.m_single_test_patch < 0
                                    || self.m_single_test_patch == ipatch as i32)
                            {
                                printf!(
                                    "%d vec %.2f  min %.2f minPeak %d from %.1f %.1f  %.4f to %.1f %.1f %.4f\n",
                                    CArg::Int(indp as i64),
                                    CArg::Dbl(dev_vec[indp] as f64),
                                    CArg::Dbl(dev_min as f64),
                                    CArg::Int(min_peak as i64),
                                    CArg::Dbl(xmodel[ipt] as f64),
                                    CArg::Dbl(ymodel[ipt] as f64),
                                    CArg::Dbl(self.m_peak_vecs[ipatch][0] as f64),
                                    CArg::Dbl(self.m_model_xvecs[ipatch][mp] as f64),
                                    CArg::Dbl(self.m_model_yvecs[ipatch][mp] as f64),
                                    CArg::Dbl(self.m_peak_vecs[ipatch][mp] as f64)
                                );
                            }
                            xmodel[ipt] = self.m_model_xvecs[ipatch][mp];
                            ymodel[ipt] = self.m_model_yvecs[ipatch][mp];
                            // `tiltxcorr.cpp:4058-4059` index with `mMatCols`, not
                            // `mPairCols` (see the module comment).
                            let mc = self.m_mat_cols as usize;
                            self.m_pair_mat[indp * mc + 2] = xmodel[ipt] - self.m_xcen as f32;
                            self.m_pair_mat[indp * mc + 3] = ymodel[ipt] - self.m_ycen as f32;
                            num_replaced += 1;
                            drop_val[ipatch] = 1.;
                        }
                    }
                }
            }
            if num_replaced < 0 {
                break;
            }

            if num_replaced != 0 {
                printf!(
                    "Replaced %d positions on iteration %d\n",
                    CArg::Int(num_replaced as i64),
                    CArg::Int(iter as i64 + 1)
                );
            } else {
                break;
            }
            iter += 1;
        }

        // Now do an elimination round
        if iter > 0 || num_replaced >= 0 {
            num_replaced = 0;
            for ipatch in 0..np {
                if drop_val[ipatch] < 0. {
                    xmodel[mind(ipatch, iv_cur)] = -1.0e10;
                    num_replaced += 1;
                }
            }
            if num_replaced != 0 {
                printf!(
                    "Eliminated %d outlier points\n",
                    CArg::Int(num_replaced as i64)
                );
            }
        }
        printf!(
            "Prediction fit error: mean %.2f  max %.1f\n",
            CArg::Dbl(fit_avg_mean as f64),
            CArg::Dbl(fit_max_dev as f64)
        );
        let _ = (eval_dev, dev_sd);
        0
    }

    /// `TiltXCorr::findXfWithoutOutliers` (`tiltxcorr.cpp:4119`).  Finds a 2-D
    /// transform and analyzes for outliers using the approach in
    /// `flib/model/solve_wo_outliers.f90`.  The source carves `devInds` out of
    /// `work` behind the `numPts` deviations; it is its own array here, since
    /// no caller reads that part of `work`.  The member uses no member state.
    #[allow(clippy::too_many_arguments)]
    fn find_xf_without_outliers(
        pair_mat: &mut [f32],
        pair_col_dim: i32,
        fit_mat: &mut [f32],
        fit_col_dim: i32,
        num_pts: i32,
        xcen: f32,
        ycen: f32,
        if_trans: i32,
        if_rotrans: i32,
        max_drop: i32,
        elim_min: f32,
        crit_prob: f32,
        abs_prob_crit: f32,
        work: &mut [f32],
        xf: &mut [f32],
        dev_avg: &mut f32,
        dev_sd: &mut f32,
        dev_max: &mut f32,
    ) -> i32 {
        let mut num_drop: i32 = 0;
        let mut ipnt_max: i32 = 0;
        let mut num_keep: i32;
        let mut num_fit: i32;
        let mut err: i32;
        let mut last_drop: i32 = 0;
        let mut dev: f32;
        let prob_per_point: f32;
        let abs_per_point: f32;
        let mut z: f32;
        let mut prob: f32;
        let mut gprob: f32;
        let mut sigma_from_mean: f32;
        let mut sigma_from_sd: f32;
        let mut sigma: f32;
        let mut dev_inds = vec![0i32; num_pts.max(0) as usize];
        let pc = pair_col_dim as usize;
        let fc = fit_col_dim as usize;

        // Load all the data
        num_fit = 0;
        for row in 0..num_pts as usize {
            if pair_mat[row * pc + 4] > 0. {
                let nf = num_fit as usize;
                fit_mat[nf * fc] = pair_mat[row * pc] - xcen;
                fit_mat[nf * fc + 1] = pair_mat[row * pc + 1] - ycen;
                fit_mat[nf * fc + 2] = pair_mat[row * pc + 2] - xcen;
                fit_mat[nf * fc + 3] = pair_mat[row * pc + 3] - ycen;
                fit_mat[nf * fc + 5] = row as f32;
                num_fit += 1;
            }
        }
        err = find_transform(
            fit_mat,
            fit_col_dim,
            3,
            num_fit,
            xcen,
            ycen,
            if_trans,
            if_rotrans,
            1,
            xf,
            dev_avg,
            dev_sd,
            dev_max,
            &mut ipnt_max,
        );
        if err != 0 {
            return err;
        }
        if *dev_max < elim_min {
            return 0;
        }

        // Sort the residuals fom the full fit
        let nfu = num_fit as usize;
        for ind in 0..nfu {
            work[ind] = fit_mat[ind * fc + 12];
            dev_inds[ind] = ind as i32;
        }
        rs_sort_indexed_floats(&work[..], &mut dev_inds, num_fit);
        prob_per_point = (1. - crit_prob as f64).powf(1. / num_fit as f64) as f32;
        abs_per_point = (1. - abs_prob_crit as f64).powf(1. / num_fit as f64) as f32;

        // Reload the data in order, using a different column for cross-index
        for ind in 0..nfu {
            let row = b3dnint!(fit_mat[dev_inds[ind] as usize * fc + 5]) as usize;
            fit_mat[ind * fc] = pair_mat[row * pc] - xcen;
            fit_mat[ind * fc + 1] = pair_mat[row * pc + 1] - ycen;
            fit_mat[ind * fc + 2] = pair_mat[row * pc + 2] - xcen;
            fit_mat[ind * fc + 3] = pair_mat[row * pc + 3] - ycen;
            fit_mat[ind * fc + 6] = row as f32;
        }
        for jdrop in 1..=max_drop + 1 {
            ipnt_max = jdrop;
            err = find_transform(
                fit_mat,
                fit_col_dim,
                3,
                num_fit - jdrop,
                xcen,
                ycen,
                if_trans,
                if_rotrans,
                -1,
                xf,
                dev_avg,
                dev_sd,
                dev_max,
                &mut ipnt_max,
            );
            if err != 0 {
                return err;
            }

            sigma_from_mean = (*dev_avg as f64 / (8. / 3.14159f64).sqrt()) as f32;
            sigma_from_sd = (*dev_sd as f64 / (3. - 8. / 3.14159f64).sqrt()) as f32;
            sigma = if sigma_from_mean > sigma_from_sd {
                sigma_from_mean
            } else {
                sigma_from_sd
            };
            num_keep = 0;
            let base = (num_fit - jdrop) as usize;
            for ind in base..nfu {
                work[ind - base] = fit_mat[ind * fc + 12];
                dev_inds[ind - base] = (ind - base) as i32;
            }
            if jdrop > 1 {
                rs_sort_indexed_floats(&work[..], &mut dev_inds, jdrop);
            }
            for ind in base..nfu {
                dev = fit_mat[(dev_inds[ind - base] as usize + base) * fc + 12];
                if (sigma as f64) < 0.1 * dev as f64 || (sigma as f64) < 1.0e-5 {
                    z = 10.;
                } else {
                    z = dev / sigma;
                }
                gprob = (1. - 0.5 * (1. - err_func(z as f64 / 1.414214))) as f32;

                prob = (2. * (gprob as f64 - 0.5)
                    - (2. / 3.14159f64).sqrt() * z as f64 * ((-z * z) as f64 / 2.).exp())
                    as f32;
                if prob < prob_per_point {
                    num_keep += 1;
                }
                if prob >= abs_per_point {
                    num_drop = max_drop.min(num_drop.max(num_fit - ind as i32));
                }
            }
            /*
             * If all points are outliers, this is a candidate for a set to drop
             * When only the first point is kept, and all the rest of the points
             * were outliers on the previous round, then this is a safe place to
             * draw the line between good data and outliers.  In this case, set
             * ndrop; and at end take the biggest ndrop that fits these criteria
             */
            if num_keep == 0 {
                last_drop = jdrop;
            }
            if num_keep == 1 && last_drop == jdrop - 1 && last_drop > 0 {
                num_drop = last_drop;
            }

            // Reload the top end in the right order
            for ind in 0..jdrop as usize {
                dev_inds[ind] = b3dnint!(fit_mat[(dev_inds[ind] as usize + base) * fc + 6]);
            }
            for ind in base..nfu {
                let row = dev_inds[ind - base] as usize;
                fit_mat[ind * fc] = pair_mat[row * pc] - xcen;
                fit_mat[ind * fc + 1] = pair_mat[row * pc + 1] - ycen;
                fit_mat[ind * fc + 2] = pair_mat[row * pc + 2] - xcen;
                fit_mat[ind * fc + 3] = pair_mat[row * pc + 3] - ycen;
                fit_mat[ind * fc + 6] = row as f32;
            }
        }

        ipnt_max = num_drop;
        find_transform(
            fit_mat,
            fit_col_dim,
            3,
            num_fit - num_drop,
            xcen,
            ycen,
            if_trans,
            if_rotrans,
            -1,
            xf,
            dev_avg,
            dev_sd,
            dev_max,
            &mut ipnt_max,
        );

        for ind in 0..nfu {
            let row = b3dnint!(fit_mat[ind * fc + 6]) as usize;
            if ind as i32 >= num_fit - num_drop {
                pair_mat[row * pc + 4] = -1.;
            }
            work[row] = fit_mat[ind * fc + 12];
        }
        0
    }
}
