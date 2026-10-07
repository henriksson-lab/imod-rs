//! Translation of `IMOD/flib/beadtrack/beadtrack.cpp` with its class header
//! `IMOD/flib/beadtrack/beadtrack.h` merged in.
//!
//! BEADTRACK tracks selected fiducial gold beads through a series of tilted
//! views.  It takes a "seed" model, where each bead of choice is marked with at
//! least a single point on a view near zero tilt, tracks each bead as far as
//! possible and puts out a new model.
//!
//! # Structure
//!
//! The C++ `BeadTrack` class becomes the [`BeadTrack`] struct, one method per
//! member function (`main`, `getBackgroundWsumStats`, `addSequence`,
//! `evaluateCGvsSobelResids`, `swapSobelAndCGcoords`,
//! `redoFitsEvaluateResiduals`, `findNearestSobelPeak`, `shiftAndFillBox`,
//! `loadBoxAndTaper`, `addBestBeadFound`, `lookForOneBead`,
//! `findAllBeadsOnView`, `getWsumCriteria`, `getProjectedPositionsSetupFits`,
//! `countAndPreparePointsToDo`), and the file's `main` is [`beadtrack`].
//!
//! The five file-scope static instances (`cgPixels`, `tltCntrl`, `alignVars`,
//! `arrayMaxes`, `sepGroups`, `beadtrack.cpp:32-41`) and the `fmod*` globals of
//! `fortmodel.h` are owned by the struct as `cp`, `tc`, `av`, `mx`, `sg` and
//! `fm`, and are handed to the other units as the explicit parameters their
//! translations take in place of the `*SetPointers` statics (`alivar.rs`,
//! `tltcntrl.rs`, `cgpixels.rs`).  The `*SetPointers` calls of `main` are kept
//! and are no-ops.
//!
//! # Representation
//!
//! - Every `B3DMALLOC`ed `float *`/`int *`/`bool *` member or local is a `Vec`
//!   (zeroed where `malloc` leaves residue; each array is written before it is
//!   read except where noted).  A pointer the source never allocates in a
//!   configuration (`mEdgeSdSave` without an elongation file, the outer-MAD
//!   arrays, the Sobel arrays with Sobel centering off) is an empty `Vec`.
//! - `mCtf`/`mCtfStat` (`float [8193]`) are `Vec`s of 8193 so that
//!   `loadBoxAndTaper`'s `ctfUse` argument can be lent out of the struct
//!   together with the box buffer it loads, which is also a member; both are
//!   taken with `mem::take` for the call and put back, the same storage.
//! - `shiftAndFillBox` is only ever called with `boxLoaded == boxPadded`
//!   (`beadtrack.cpp:3151`) and is written for that in-place case: one slice.
//! - The `BeadTrack bt` object is a local of the C `main`, so the members the
//!   constructor does not set start as stack residue.  All of them are written
//!   before they are read on every path the differential reaches except two:
//!   `mMinzDelzNearZero` (set to 0 only when `ShiftsNearZeroTilt` is absent,
//!   `beadtrack.cpp:673`) and `mMaxDelzNearZero` (never set unless
//!   `SetIndexedParameter -5` is given).  Fixed in translation (2026-09-26,
//!   `BUGS.md`): they start at 0.
//!
//! # Arithmetic
//!
//! `B3DNINT(a)` is `(int)floor((a) + 0.5)` with a double `0.5`.  `B3DMAX`/
//! `B3DMIN` are the source's ternaries (the second operand wins a NaN
//! comparison), written out.  `pow(float, 2.f)` resolves to libstdc++'s
//! `float` overload and folds to a single-precision square (the reference
//! `beadtrack.o` imports no `pow`/`powf`); `sqrt` of a `float` expression is
//! `sqrtf` (imported) and of a `double` expression `sqrt` (imported).
//!
//! # OpenMP
//!
//! `beadtrack.cpp` has no parallel region of its own.  The library routines it
//! reaches that do (`funct`'s gradient sums, `metroSearch`, `scaledSobel`'s
//! filter) already reproduce native at every thread count.
use std::io::{BufReader, Write};

use super::cgpixels::CGPixels;
use super::proc_vars::proc_vars_set_pointers;
use super::tiltali::{tilt_ali, tilt_ali_set_pointers};
use super::tltcntrl::TiltControl;
use super::tracksubs::{
    add_point, adjust_xyz_in_areas, best_center_for_cg, calc_cg, calc_elongation, calc_outer_mad,
    check_sobel_peak, cosd, count_missing, find_piece, findxf_wo_outliers, next_pos, peak_find,
    qd_shift, rescue, setsiz_sam_cel, sind, split_pack, tracksubs_set_pointers,
    wsum_for_sobel_peak,
};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::tiltalign::alivar::AlignVariables;
use crate::imod::flib::tiltalign::arraymaxes::{ArrayMaxes, MAXGRP};
use crate::imod::flib::tiltalign::evalfunct::EvalFunct;
use crate::imod::flib::tiltalign::map_vars::{
    input_groupings, input_separate_groups, map_vars_set_pointers,
};
use crate::imod::flib::tiltalign::mapsepgroups::MapSepGroups;
use crate::imod::flib::tiltalign::utilfuncs::{
    allocate_alivar, allocate_mapsep, copy_array, formatted_error, memory_error,
};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_open_file, b3drand, b3dsrand, c_format_bytes, imod_prog_name,
    imod_usage_header, number_in_list,
};
use crate::imod::libcfshr::beadutil::make_model_bead;
use crate::imod::libcfshr::convexbound::convex_bound;
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, apply_kernel_filter, conjugate_product, nice_frame, scaled_gaussian_kernel,
    xcorr_filter_part, xcorr_mean_zero, xcorr_peak_find, xcorr_set_ctf,
};
use crate::imod::libcfshr::gettiltangles::get_tilt_angles;
use crate::imod::libcfshr::linearxforms::{exit_from_xf_read_error, read_all_xforms, xf_apply};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_allow_comma_defaults, pip_done, pip_get_boolean, pip_get_float,
    pip_get_float_array, pip_get_in_out_file, pip_get_integer, pip_get_string, pip_get_two_floats,
    pip_get_two_integers, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::percentile::percentile_float;
use crate::imod::libcfshr::piecefuncs::{check_piece_list, fill_list_of_piece_z, read_piece_list};
use crate::imod::libcfshr::readlinevalues::exit_from_value_read_error;
use crate::imod::libcfshr::reduce_by_binning::repack_float_image;
use crate::imod::libcfshr::robuststat::{
    rs_fast_madn, rs_fast_median_in_place, rs_median_of_sorted, rs_percentile_of_sorted,
    rs_sort_floats, rs_sort_indexed_floats,
};
use crate::imod::libcfshr::scaledsobel::scaled_sobel;
use crate::imod::libcfshr::simplestat::{
    array_min_max_mean, avg_sd, ls_fit_pred, scale_array_for_mode, sums_to_avg_sd,
};
use crate::imod::libcfshr::taperatfill::{get_last_taper_fill_value, taper_at_fill};
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad, slice_taper_out_pad};
use crate::imod::libcfshr::writelist::write_list;
use crate::imod::libfft::todfft_c;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, mrc_fill_label_string};
use crate::imod::libiimod::mrcslice::SLICE_MODE_FLOAT;
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_open, iiu_read_sec_part, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_mode, iiu_print_header, iiu_ret_basic_head, iiu_trans_header, iiu_write_header_str,
};
use crate::imod::libimod::fortmodel::{
    fort_mod_obj_to_cont, fort_mod_open_error, read_fort_model, scale_fort_mod_to_image,
    scale_fort_model, write_fort_model,
};
use crate::imod::libimod::imodel_fwrap::{
    getimodobjsize, putimodflag, putimodobjname, putimodzscale, putobjcolor, putsymsize, putsymtype,
};

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.  The conversion is
/// the reference build's `cvttsd2si`, which gives `INT_MIN` for a NaN or
/// out-of-range value where Rust's `as i32` saturates (a NaN position reaches
/// it after a degenerate transform fit; see `tracksubs::find_piece`).
macro_rules! b3dnint {
    ($a:expr) => {{
        let v: f64 = (($a) as f64 + 0.5).floor();
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is part of the x86-64 baseline.
            unsafe { core::arch::x86_64::_mm_cvttsd_si32(core::arch::x86_64::_mm_set_sd(v)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::imod::flib::subrs::compat::gfortran_rt::cvttsd2si(v)
        }
    }};
}

/// `B3DMAX(a,b)` (`b3dutil.h:30`): `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        if a > b { a } else { b }
    }};
}

/// `B3DMIN(a,b)` (`b3dutil.h:29`): `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        if a < b { a } else { b }
    }};
}

/// `B3DABS(a)` (`b3dutil.h:34`): `((a) >= 0 ? (a) : -(a))`.
macro_rules! b3dabs {
    ($a:expr) => {{
        let a = $a;
        if a >= Default::default() { a } else { -a }
    }};
}

/// `#define boxSizeFromDiam(diam) B3DMAX(32., B3DMAX(2. * diam + 20., 3.3 * diam + 2.))`
/// (`beadtrack.cpp:28`), evaluated in double.
macro_rules! box_size_from_diam {
    ($d:expr) => {{
        let d = ($d) as f64;
        b3dmax!(32., b3dmax!(2. * d + 20., 3.3 * d + 2.))
    }};
}

/// The source's `printf`, through the C-format writer on libc-order stdout.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `fprintf` to an `ImodFile`.
macro_rules! fprintf {
    ($fp:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = $fp.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `fflush(stdout)`.
macro_rules! fflush_stdout {
    () => {{
        let _ = ImodFile::Stdout.flush();
    }};
}

/// A float array seen as raw bytes, for the `void *` routines.
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

/// `SET_CONTROL_INT(a,b)` (`cppdefs.h:18`): `case a: b = B3DNINT(yy); cout << #b
/// << " set to " << B3DNINT(yy) << endl; break`.  `cout` is synchronised with
/// C stdio, and `endl` flushes.
macro_rules! set_control_int {
    ($var:expr, $name:expr, $yy:expr) => {{
        $var = b3dnint!($yy);
        printf!(
            "%s set to %d\n",
            CArg::Str($name),
            CArg::Int(b3dnint!($yy) as i64)
        );
        fflush_stdout!();
    }};
}

/// `SET_CONTROL_FLOAT(a,b)` (`cppdefs.h:17`): `cout << yy` on a `float` is the
/// `double` insertion at the default precision 6, i.e. `%g`.
macro_rules! set_control_float {
    ($var:expr, $name:expr, $yy:expr) => {{
        $var = $yy;
        printf!("%s set to %g\n", CArg::Str($name), CArg::Dbl($yy as f64));
        fflush_stdout!();
    }};
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// A `NULL`-able `int *` list with its count, as `numberInList` takes it.
fn list_or_null(v: &[i32]) -> Option<&[i32]> {
    (!v.is_empty()).then_some(v)
}

/// `class BeadTrack` (`beadtrack.h:5-267`), plus the file-scope statics and
/// `fmod*` globals it reaches (see the module docs).
pub struct BeadTrack {
    /// `EvalFunct mEvalFunct` (`beadtrack.h:26`).
    pub m_eval_funct: EvalFunct,
    /// `static CGPixels cgPixels` (`beadtrack.cpp:32`).
    cp: CGPixels,
    /// `static TiltControl tltCntrl` (`beadtrack.cpp:34`).
    tc: TiltControl,
    /// `static AlignVariables alignVars` (`beadtrack.cpp:36`).
    av: AlignVariables,
    /// `static ArrayMaxes arrayMaxes` (`beadtrack.cpp:38`).
    mx: ArrayMaxes,
    /// `static MapSepGroups sepGroups` (`beadtrack.cpp:40`).
    sg: MapSepGroups,
    /// The `fmod*` globals of `fortmodel.h`.
    fm: FortModel,

    m_xmat_size: i32,
    m_max_peaks: i32,
    m_ny_im: i32,
    m_nx_im: i32,
    m_mat_kernel: [f32; 49],
    m_alt_pred_wsum: [f32; 2],
    m_alt_yseek: [f32; 2],
    m_alt_xseek: [f32; 2],
    m_alt_pred_ynext: [f32; 2],
    m_alt_pred_xnext: [f32; 2],
    m_alt_score: [f32; 2],
    m_alt_pred_edge_sd: [f32; 2],
    m_alt_pred_ypeak: [f32; 2],
    m_alt_pred_xpeak: [f32; 2],
    m_y_shift_near_zero: [f32; 2],
    m_x_shift_near_zero: [f32; 2],
    m_alt_ypos: [f32; 2],
    m_alt_xpos: [f32; 2],
    m_sobel_sum: Vec<f32>,
    m_corr_sum: Vec<f32>,
    m_box_tmp: Vec<f32>,
    m_cur_sum: Vec<f32>,
    m_boxes: Vec<f32>,
    m_sobel_edge_sd: Vec<f32>,
    m_sbrray: Vec<f32>,
    m_sarray: Vec<f32>,
    m_brray: Vec<f32>,
    m_array: Vec<f32>,
    m_sobel_wsums: Vec<f32>,
    m_box_sobel: Vec<f32>,
    m_tmp_sobel: Vec<f32>,
    m_ref_sobel: Vec<f32>,
    m_prexf: Vec<f32>,
    m_sobel_peaks: Vec<f32>,
    m_sobel_ypeaks: Vec<f32>,
    m_sobel_xpeaks: Vec<f32>,
    m_bkgd_neigh_wmax: Vec<f32>,
    m_stat_tmp: Vec<f32>,
    m_ind_gap: Vec<i32>,
    m_in_core: Vec<i32>,
    m_iz_pclist: Vec<i32>,
    m_iy_pclist: Vec<i32>,
    m_ix_pclist: Vec<i32>,
    m_iv_snap_list: Vec<i32>,
    m_iz_close: Vec<i32>,
    m_ip_close: Vec<i32>,
    m_list_seq: Vec<i32>,
    m_iv_seq_end: Vec<i32>,
    m_iv_seq_str: Vec<i32>,
    m_elong_save: Vec<f32>,
    m_edge_sd_save: Vec<f32>,
    m_prev_res: Vec<f32>,
    m_wsum_save: Vec<f32>,
    m_bkgd_wmax_save: Vec<f32>,
    m_outer_background: Vec<f32>,
    m_outer_madsave: Vec<f32>,
    m_iflag_cgvs_sobel: Vec<i32>,
    m_saved_cgcoord: Vec<f32>,
    m_wsum_min: Vec<f32>,
    m_xmat: Vec<f32>,
    m_wsum_crit: Vec<f32>,
    m_yseek: Vec<f32>,
    m_xseek: Vec<f32>,
    m_yseek_next_pos: Vec<f32>,
    m_xseek_next_pos: Vec<f32>,
    m_ip_near_save: Vec<i32>,
    m_ip_nearest: Vec<i32>,
    m_if_found: Vec<i32>,
    m_num_in_sobel_sum: Vec<i32>,
    m_idrop: Vec<i32>,
    m_iobj_del: Vec<i32>,
    m_in_sobel_sum: Vec<bool>,
    m_in_corr_sum: Vec<bool>,
    m_res_mean: Vec<f32>,
    m_neighbors_for_wfits: Vec<i32>,
    m_num_wneighbors: Vec<i32>,
    m_iv_gap: Vec<i32>,
    m_ctf_stat: Vec<f32>,
    m_ctf: Vec<f32>,
    m_xform: [f32; 6],
    m_lim_pts_shift: i32,
    m_lim_pts_rot: i32,
    m_lim_pts_mag: i32,
    m_lim_pts_stretch: i32,
    m_max_wavg: i32,
    m_mode_box: i32,
    m_frac_crit: f32,
    m_rot_start: f32,
    m_iobj_do: i32,
    m_last_seq: i32,
    m_max_gap: i32,
    m_if_fill_in: i32,
    m_if_white: i32,
    m_npclist: i32,
    m_min_resid: i32,
    max_resid: i32,
    m_min_fit: i32,
    m_num_fit: i32,
    m_max_sum: i32,
    m_rad_max_fit: f32,
    m_relax_fit: f32,
    m_fit_dist_crit: f32,
    m_relax_dist: f32,
    m_relax_int: f32,
    m_dist_crit: f32,
    m_sd_crit: f32,
    m_cg_radius: f32,
    m_tilt_fit_min: f32,
    m_res_diff_crit: f32,
    m_res_diff_min: f32,
    m_rescue_step_size: f32,
    m_iz_next: i32,
    m_ipass: i32,
    m_if_trace: i32,
    m_nxp_dim: i32,
    m_npix_box: i32,
    m_ny_pad: i32,
    m_nx_pad: i32,
    m_ny_box: i32,
    m_nx_box: i32,
    m_ny_taper: i32,
    m_nx_taper: i32,
    m_nz_out: i32,
    m_corr_min: f32,
    m_corro_sum: f32,
    m_ref_max: f32,
    m_ref_min: f32,
    m_ref_sum: f32,
    m_box_max: f32,
    m_box_min: f32,
    m_box_sum: f32,
    m_dx_cur: f32,
    m_dy_cur: f32,
    m_corr_max: f32,
    m_ipiece_z: i32,
    m_iv_list: i32,
    m_iv_use: i32,
    m_iview: i32,
    m_track_dir: i32,
    m_iview_seq: i32,
    m_num_added: i32,
    m_isequence: i32,
    m_num_seqs: i32,
    m_iy1: i32,
    m_iy0: i32,
    m_ix1: i32,
    m_ix0: i32,
    m_if_did_align: i32,
    m_num_data: i32,
    m_ypeak: f32,
    m_xpeak: f32,
    m_ynext: f32,
    m_xnext: f32,
    m_interp_type: i32,
    m_max_obj_orig: i32,
    m_num_pioneer: i32,
    m_num_del: i32,
    m_max_view_do: i32,
    m_min_view_do: i32,
    m_min_peak_ratio: f32,
    m_tilt_max: f32,
    m_cur_res_min: f32,
    m_outlie_crit_abs: f32,
    m_outlie_crit: f32,
    m_outlie_elim_min: f32,
    m_peak: f32,
    m_edge_sd: f32,
    m_scale_by_interp: f32,
    m_scale_fac_sobel: f32,
    m_sobel_sigma: f32,
    m_diameter: f32,
    m_delta_ctf: f32,
    m_nfill_taper: i32,
    m_max_any_sum: i32,
    m_max_all_real: i32,
    m_max_neigh: i32,
    m_max_sobel_sum: i32,
    m_if_align_done: i32,
    m_ivs_on_align: i32,
    m_if_read_xfs: i32,
    m_kernel_dim: i32,
    m_nys_pad: i32,
    m_nxs_pad: i32,
    m_ny_sobel: i32,
    m_nx_sobel: i32,
    m_outer_sigma: f32,
    m_wcrit_to_local_min_ratio: f32,
    m_dbl_norm_min_crit: f32,
    m_percentile_crit_frac: f32,
    m_cg_edge_width: f32,
    m_sobel_res_sum: f32,
    m_cg_res_sum: f32,
    m_ali_mean_res: f32,
    m_ycgsaved: [f32; 2],
    m_xcgsaved: [f32; 2],
    m_relax_bidir_fac: f32,
    m_try_alt_pred_diam_frac: f32,
    m_cg_gap_width: f32,
    m_num_sobel_cgeval: i32,
    m_num_cgbetter: i32,
    m_max_wneigh: i32,
    m_iv_bidir_part2: i32,
    m_num_bidir_relax_crit: i32,
    m_ind_pred_try: i32,
    m_num_pred_tries: i32,
    m_max_drb_delta_z: i32,
    m_max_bmr_delta_z: i32,
    m_ny_stat_pad: i32,
    m_nx_stat_pad: i32,
    m_ny_stat_box: i32,
    m_nx_stat_box: i32,
    m_num_to_do: i32,
    m_nsnap_list: i32,
    m_max_delz_near_zero: i32,
    m_minz_delz_near_zero: i32,
    m_bmr_upper_elong_lim: f32,
    m_bmr_lower_elong_lim: f32,
    m_bkgd_wsum_madn: f32,
    m_bkgd_wsum_median: f32,
    m_delta_ctf_stat: f32,
    m_bmr_min_wsum_madnratio: f32,
    m_bmr_max_elong_sd: f32,
    m_bmr_high_elong_raise_fac: f32,
    m_bmr_low_elong_raise_fac: f32,
    m_drb_just_accept_crit: f32,
    m_bkgd_wsum_max: f32,
    m_bmr_max_num_sdabove_mean: f32,
    m_drb_reject_max_low_madns: f32,
    m_drb_accept_min_madns: f32,
    m_drb_just_reject_crit: f32,
    m_drb_reject_max_high_madns: f32,
    m_drb_any_type_min_madns: f32,
    m_need_taper: bool,
    m_save_all_points: bool,
    m_need_fill: bool,
    m_need4digits: bool,
    /// `FILE *mCplFP`.
    m_cpl_fp: Option<ImodFile>,
    /// `FILE *mBrplFP`.
    m_brpl_fp: Option<ImodFile>,
}

/// Original: `main` (`beadtrack.cpp:46`).
///
/// Instantiate BeadTrack class, set pointers in other modules, and run the real
/// main.
pub fn beadtrack(arguments: &[String]) -> i32 {
    let mut bt = Box::new(BeadTrack::new());
    let bt_ref = &mut *bt;
    tracksubs_set_pointers(&mut bt_ref.cp, &mut bt_ref.mx);
    tilt_ali_set_pointers(
        &mut bt_ref.tc,
        &mut bt_ref.av,
        &mut bt_ref.mx,
        &mut bt_ref.m_eval_funct,
    );
    map_vars_set_pointers(&mut bt_ref.av, &mut bt_ref.mx, &mut bt_ref.sg);
    proc_vars_set_pointers(
        &mut bt_ref.tc,
        &mut bt_ref.sg,
        &mut bt_ref.av,
        &mut bt_ref.mx,
    );
    bt.main(arguments);
    crate::imod::libcfshr::b3dutil::exit(0);
}

impl BeadTrack {
    /// Original: `BeadTrack::BeadTrack` (`beadtrack.cpp:60`).
    ///
    /// Initialize member variables.  Members the constructor leaves alone start
    /// at zero (see the module docs).
    pub fn new() -> BeadTrack {
        BeadTrack {
            m_eval_funct: EvalFunct::new(),
            cp: CGPixels::default(),
            tc: TiltControl::default(),
            av: AlignVariables::default(),
            mx: ArrayMaxes::default(),
            sg: MapSepGroups::default(),
            fm: FortModel::default(),
            m_xmat_size: 19,
            m_max_peaks: 20,
            m_ny_im: 0,
            m_nx_im: 0,
            m_mat_kernel: [0.; 49],
            m_alt_pred_wsum: [0.; 2],
            m_alt_yseek: [0.; 2],
            m_alt_xseek: [0.; 2],
            m_alt_pred_ynext: [0.; 2],
            m_alt_pred_xnext: [0.; 2],
            m_alt_score: [0.; 2],
            m_alt_pred_edge_sd: [0.; 2],
            m_alt_pred_ypeak: [0.; 2],
            m_alt_pred_xpeak: [0.; 2],
            m_y_shift_near_zero: [0., 0.],
            m_x_shift_near_zero: [0., 0.],
            m_alt_ypos: [0.; 2],
            m_alt_xpos: [0.; 2],
            m_sobel_sum: Vec::new(),
            m_corr_sum: Vec::new(),
            m_box_tmp: Vec::new(),
            m_cur_sum: Vec::new(),
            m_boxes: Vec::new(),
            m_sobel_edge_sd: Vec::new(),
            m_sbrray: Vec::new(),
            m_sarray: Vec::new(),
            m_brray: Vec::new(),
            m_array: Vec::new(),
            m_sobel_wsums: Vec::new(),
            m_box_sobel: Vec::new(),
            m_tmp_sobel: Vec::new(),
            m_ref_sobel: Vec::new(),
            m_prexf: Vec::new(),
            m_sobel_peaks: Vec::new(),
            m_sobel_ypeaks: Vec::new(),
            m_sobel_xpeaks: Vec::new(),
            m_bkgd_neigh_wmax: Vec::new(),
            m_stat_tmp: Vec::new(),
            m_ind_gap: Vec::new(),
            m_in_core: Vec::new(),
            m_iz_pclist: Vec::new(),
            m_iy_pclist: Vec::new(),
            m_ix_pclist: Vec::new(),
            m_iv_snap_list: Vec::new(),
            m_iz_close: Vec::new(),
            m_ip_close: Vec::new(),
            m_list_seq: Vec::new(),
            m_iv_seq_end: Vec::new(),
            m_iv_seq_str: Vec::new(),
            m_elong_save: Vec::new(),
            m_edge_sd_save: Vec::new(),
            m_prev_res: Vec::new(),
            m_wsum_save: Vec::new(),
            m_bkgd_wmax_save: Vec::new(),
            m_outer_background: Vec::new(),
            m_outer_madsave: Vec::new(),
            m_iflag_cgvs_sobel: Vec::new(),
            m_saved_cgcoord: Vec::new(),
            m_wsum_min: Vec::new(),
            m_xmat: Vec::new(),
            m_wsum_crit: Vec::new(),
            m_yseek: Vec::new(),
            m_xseek: Vec::new(),
            m_yseek_next_pos: Vec::new(),
            m_xseek_next_pos: Vec::new(),
            m_ip_near_save: Vec::new(),
            m_ip_nearest: Vec::new(),
            m_if_found: Vec::new(),
            m_num_in_sobel_sum: Vec::new(),
            m_idrop: Vec::new(),
            m_iobj_del: Vec::new(),
            m_in_sobel_sum: Vec::new(),
            m_in_corr_sum: Vec::new(),
            m_res_mean: Vec::new(),
            m_neighbors_for_wfits: Vec::new(),
            m_num_wneighbors: Vec::new(),
            m_iv_gap: Vec::new(),
            m_ctf_stat: vec![0.; 8193],
            m_ctf: vec![0.; 8193],
            m_xform: [0.; 6],
            m_lim_pts_shift: 3,
            m_lim_pts_rot: 4,
            m_lim_pts_mag: 6,
            m_lim_pts_stretch: 16,
            m_max_wavg: 15,
            m_mode_box: 0,
            m_frac_crit: 0.6,
            m_rot_start: 0.,
            m_iobj_do: 0,
            m_last_seq: 0,
            m_max_gap: 5,
            m_if_fill_in: 0,
            m_if_white: 0,
            m_npclist: 0,
            m_min_resid: 5,
            max_resid: 0,
            m_min_fit: 3,
            m_num_fit: 7,
            m_max_sum: 4,
            m_rad_max_fit: 2.5,
            m_relax_fit: 0.9,
            m_fit_dist_crit: 2.5,
            m_relax_dist: 0.9,
            m_relax_int: 0.7,
            m_dist_crit: 10.,
            m_sd_crit: 1.,
            m_cg_radius: 0.,
            // Was 15 until cos/sin fitting improved 7/25/12
            m_tilt_fit_min: 8.,
            m_res_diff_crit: 2.,
            m_res_diff_min: 0.04,
            m_rescue_step_size: 1.,
            m_iz_next: 0,
            m_ipass: 0,
            m_if_trace: 0,
            m_nxp_dim: 0,
            m_npix_box: 0,
            m_ny_pad: 0,
            m_nx_pad: 0,
            m_ny_box: 0,
            m_nx_box: 0,
            m_ny_taper: 0,
            m_nx_taper: 0,
            m_nz_out: 0,
            m_corr_min: 0.,
            m_corro_sum: 0.,
            m_ref_max: 0.,
            m_ref_min: 0.,
            m_ref_sum: 0.,
            m_box_max: 0.,
            m_box_min: 0.,
            m_box_sum: 0.,
            m_dx_cur: 0.,
            m_dy_cur: 0.,
            m_corr_max: 0.,
            m_ipiece_z: 0,
            m_iv_list: 0,
            m_iv_use: 0,
            m_iview: 0,
            m_track_dir: 0,
            m_iview_seq: 0,
            m_num_added: 0,
            m_isequence: 0,
            m_num_seqs: 0,
            m_iy1: 0,
            m_iy0: 0,
            m_ix1: 0,
            m_ix0: 0,
            m_if_did_align: 0,
            m_num_data: 0,
            m_ypeak: 0.,
            m_xpeak: 0.,
            m_ynext: 0.,
            m_xnext: 0.,
            m_interp_type: 0,
            m_max_obj_orig: 0,
            m_num_pioneer: 0,
            m_num_del: 0,
            m_max_view_do: 0,
            m_min_view_do: 0,
            m_min_peak_ratio: 0.2,
            m_tilt_max: 0.,
            m_cur_res_min: 0.5,
            m_outlie_crit_abs: 0.002,
            m_outlie_crit: 0.01,
            m_outlie_elim_min: 2.,
            m_peak: 0.,
            m_edge_sd: 0.,
            m_scale_by_interp: 1.2,
            m_scale_fac_sobel: 0.,
            m_sobel_sigma: 0.5,
            m_diameter: 0.,
            m_delta_ctf: 0.,
            m_nfill_taper: 0,
            m_max_any_sum: 0,
            m_max_all_real: 0,
            m_max_neigh: 0,
            m_max_sobel_sum: 0,
            m_if_align_done: 0,
            m_ivs_on_align: 0,
            m_if_read_xfs: 0,
            m_kernel_dim: 0,
            m_nys_pad: 0,
            m_nxs_pad: 0,
            m_ny_sobel: 0,
            m_nx_sobel: 0,
            // Was 3. when this was wanted
            m_outer_sigma: 0.,
            m_wcrit_to_local_min_ratio: 0.075,
            m_dbl_norm_min_crit: 0.2,
            m_percentile_crit_frac: 0.8,
            m_cg_edge_width: 0.,
            m_sobel_res_sum: 0.,
            m_cg_res_sum: 0.,
            m_ali_mean_res: 0.,
            m_ycgsaved: [0.; 2],
            m_xcgsaved: [0.; 2],
            m_relax_bidir_fac: 1.5,
            m_try_alt_pred_diam_frac: 1.,
            m_cg_gap_width: 0.,
            m_num_sobel_cgeval: 0,
            m_num_cgbetter: 0,
            m_max_wneigh: 0,
            m_iv_bidir_part2: 0,
            m_num_bidir_relax_crit: 0,
            m_ind_pred_try: 0,
            m_num_pred_tries: 0,
            //
            // Density rescue background analysis parameters
            m_max_drb_delta_z: 3,
            //
            // Big mean residual forgiveness parameters
            m_max_bmr_delta_z: 4,
            m_ny_stat_pad: 0,
            m_nx_stat_pad: 0,
            m_ny_stat_box: 0,
            m_nx_stat_box: 0,
            m_num_to_do: 0,
            m_nsnap_list: 0,
            m_max_delz_near_zero: 0,
            m_minz_delz_near_zero: 0,
            m_bmr_upper_elong_lim: 1.35,
            m_bmr_lower_elong_lim: 1.2,
            m_bkgd_wsum_madn: 0.,
            m_bkgd_wsum_median: 0.,
            m_delta_ctf_stat: 0.,
            m_bmr_min_wsum_madnratio: 8.,
            m_bmr_max_elong_sd: 0.2,
            m_bmr_high_elong_raise_fac: 1.3,
            m_bmr_low_elong_raise_fac: 1.5,
            m_drb_just_accept_crit: 8.,
            m_bkgd_wsum_max: 0.,
            m_bmr_max_num_sdabove_mean: 1.,
            m_drb_reject_max_low_madns: 0.,
            m_drb_accept_min_madns: 5.,
            m_drb_just_reject_crit: 3.,
            m_drb_reject_max_high_madns: 2.,
            m_drb_any_type_min_madns: 1.5,
            m_need_taper: false,
            m_save_all_points: false,
            m_need_fill: false,
            m_need4digits: false,
            m_cpl_fp: None,
            m_brpl_fp: None,
        }
    }
}

impl BeadTrack {
    /// Original: `BeadTrack::main` (`beadtrack.cpp:139`).
    ///
    /// The real main.
    pub fn main(&mut self, argv: &[String]) {
        let npad: i32;
        let mut maxarr: i32;
        let lim_pc_list: i32;
        let mut max_area: i32 = 0;
        let lim_gaps: i32;
        let lim_inside: i32;
        let lim_edge: i32;
        let mut max_olist: i32;
        let mut lim_resid: i32;
        let mut titlech = [0u8; MRC_LABEL_SIZE];

        let nz: i32;
        let mut nxyz = [0i32; 3];
        let mut mxyz = [0i32; 3];
        //
        let mut seq_dist: Vec<f32>;
        let mut xxtmp: Vec<f32>;
        let mut yytmp: Vec<f32>;
        let mut xvtmp: Vec<f32>;
        let mut yvtmp: Vec<f32>;
        let mut in_an_area: Vec<bool>;
        //
        let mut listz: Vec<i32>;
        let mut num_res_saved: Vec<i32> = Vec::new();
        let mut filin: Vec<u8> = Vec::new();
        let mut elong_file: Option<Vec<u8>>;
        let mut xyz_file: Option<Vec<u8>>;
        let mut prexf_file: Vec<u8> = Vec::new();
        let mut piece_file: Option<Vec<u8>>;
        let mut iv_list: Vec<i32>;
        let mut missing: Vec<bool>;
        let mut area_obj_str: &str;
        let mut iarea_seq: Vec<i32> = Vec::new();
        let mut nin_obj_list: Vec<i32> = Vec::new();
        let mut ind_obj_list: Vec<i32> = Vec::new();
        let mut area_dist: Vec<f32>;
        let mut resid_lists: Vec<f32> = Vec::new();
        let mut xyz_all_area: Vec<f32> = Vec::new();
        let mut iobj_lists: Vec<i32>;
        let mut iobj_lis_tmp: Vec<i32>;
        let mut iobj_map: Vec<i32>;
        let mut tmp_str: Vec<u8> = Vec::new();
        let mut out_file: Option<Vec<u8>>;
        let mut model_file_bytes: Vec<u8> = Vec::new();
        let obj_names: [&str; 5] = [
            "Model projection",
            "Fitted point",
            "Correlation peak",
            "Centroid",
            "Saved point",
        ];
        //
        let colors: [f32; 30] = [
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.5, 0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.9, 0.6, 0.4, 0.6, 0.4, 0.9,
        ];
        //
        let mut dmin2: f32 = 0.;
        let mut dmax2: f32 = 0.;
        let mut dmean2: f32 = 0.;
        let mut mode: i32 = 0;
        let mut k: i32;
        let mut ip: i32;
        let mut num_rounds: i32;
        let mut i: i32;
        let mut j: i32;
        let mut ierr: i32;
        let mut iv: i32 = 0;
        let mut tilt_min: f32;
        let mut xst: f32 = 0.;
        let mut xnd: f32 = 0.;
        let mut yst: f32 = 0.;
        let mut ynd: f32 = 0.;
        let mut bead_min_diam_for_scaling: f32;
        let mut param_scale: f32;
        let mut angle_offset: f32;
        let mut min_xpiece: i32 = 0;
        let mut nx_pieces: i32 = 0;
        let mut nx_overlap: i32 = 0;
        let mut min_ypiece: i32 = 0;
        let mut ny_pieces: i32 = 0;
        let mut ny_overlap: i32 = 0;
        let nx_tot_pix: i32;
        let ny_tot_pix: i32;
        let mut nx_local: i32 = 0;
        let mut ny_local: i32 = 0;
        let mut min_end_z: i32;
        let mut ind_free: i32;
        let mut iobj: i32;
        let mut ibase: i32;
        let mut num_in_obj: i32;
        let mut ipt: i32;
        let mut iz: i32 = 0;
        let mut jz: i32;
        let mut jpt: i32;
        let mut itmp: i32;
        let mut maxnpt: i32;
        let mut num_area_x: i32;
        let mut num_area_y: i32;
        let mut ix: i32;
        let mut iy: i32 = 0;
        let mut nobj_lists: i32 = 0;
        let mut ind_start: i32;
        let num_obj_tot: i32;
        let mut limcg: i32;
        let mut rad_pix: f32;
        let mut xtmp: f32;
        let mut ytmp: f32;
        let mut wsum: f32;
        let taper_frac: f32;
        let mut yy: f32;
        let mut nv_list: i32;
        let mut if_exclude: i32;
        let mut izv: i32;
        let _idebug_obj: i32;
        let mut itry: i32;
        let mut xpos: f32;
        let mut ypos: f32;
        let mut miss_tot: i32;
        let mut num_list_z: i32 = 0;
        let mut imod_obj: i32 = 0;
        let mut imod_cont: i32 = 0;
        let mut cvbxcen: f32 = 0.;
        let mut cvbycen: f32 = 0.;
        let mut area: f32;
        let mut density: f32 = 0.;
        let elong_sigma: f32;
        let mut num_vert: i32 = 0;
        let mut min_in_area: i32;
        let mut min_bead_overlap: i32;
        let mut if_local_area: i32;
        let mut local_target: i32;
        let mut nv_local_in: i32;
        let mut local_view_pass: i32;
        let lim_outer: i32;
        let mut ignore_objs: i32;
        let mut keep_going: bool;
        let mut done: bool;
        let mut split_first_round: bool;
        let mut in_split_round: bool;
        let mut any_resids: bool;
        let mut did_save_all_init: bool;
        let mut num_new: i32 = 0;
        let mut n_overlap: i32;
        let mut iseq_pass: i32;
        let mut ipass_save: i32;
        let mut i_area_save: i32;
        let sigma1: f32;
        let mut sigma2: f32;
        let mut radius2: f32;
        let radius1: f32;
        let ran_frac: f32;
        let mut num_area_tot: i32 = 0;
        let mut max_in_area: i32 = 0;
        let mut lim_in_area: i32;
        let num_view_do: i32;
        let mut num_bound: i32;
        let mut iseed: i32;
        let mut image_binned: i32;
        let limcx_bound: i32;
        let mut x_off_sobel: f32 = 0.;
        let mut y_off_sobel: f32 = 0.;
        let target_sobel: f32;
        let mut elong_sd: f32;
        let mut sdsum: f32;
        let mut sdsumsq: f32;
        let mut sdmin: f32;
        let mut sdmax: f32;
        // `sdavg` is uninitialised in the source when no edge SD was saved;
        // fixed in translation: it starts at 0 (see `BUGS.md`).
        let mut sdavg: f32 = 0.;
        let mut sdmed: f32;
        let mut sdmean: f32;
        let mut edge_sdsd: f32;
        let mut elong_mean: f32;
        let mut elong_med: f32;
        let mut omad_mean: f32;
        let mut oback_mean: f32;
        let tilt_increment: f32;
        let num_wneigh_want: i32;
        let mut min_filled: i32;
        let mut max_filled: i32;
        let mut initial_bidir_views: i32;
        let mut box_geo_mean: f32;
        let mut pixel_size: f32;
        let progname = imod_prog_name(argv.first().map_or("", String::as_str));
        let max_groups: i32 = MAXGRP;
        let mut elong_fp: Option<ImodFile> = None;
        let mut xyz_fp: Option<ImodFile> = None;
        let mut tilt_inc_min_for_scaling: f32;
        let mut cg_edge_diam_frac: f32;
        let mut cg_gap_diam_frac: f32;
        //
        let mut num_opt_arg: i32 = 0;
        let mut num_non_opt_arg: i32 = 0;
        //
        // fallbacks from ../../manpages/autodoc2man 2 2	beadtrack
        //
        let num_options: i32 = 69;
        let options: [&[u8]; 69] = [
            b":InputSeedModel:FN:",
            b":OutputModel:FN:",
            b":ImageFile:FN:",
            b":PieceListFile:FN:",
            b"prexf:PrealignTransformFile:FN:",
            b":XYZOutputFile:FN:",
            b":ElongationOutputFile:FN:",
            b":ImagesAreBinned:I:",
            b"pixel:PixelSize:F:",
            b":SkipViews:LI:",
            b":RotationAngle:F:",
            b":SeparateGroup:LIM:",
            b"first:FirstTiltAngle:F:",
            b"increment:TiltIncrement:F:",
            b"tiltfile:TiltFile:FN:",
            b"angles:TiltAngles:FAM:",
            b"offset:AngleOffset:F:",
            b":TiltDefaultGrouping:I:",
            b":TiltNondefaultGroup:ITM:",
            b":MagDefaultGrouping:I:",
            b":MagNondefaultGroup:ITM:",
            b":RotDefaultGrouping:I:",
            b":RotNondefaultGroup:ITM:",
            b":MinViewsForTiltalign:I:",
            b":CentroidRadius:F:",
            b":BeadDiameter:F:",
            b":MedianForCentroid:B:",
            b":LightBeads:B:",
            b":FillGaps:B:",
            b":MaxGapSize:I:",
            b":ShiftsNearZeroTilt:FA:",
            b":MinTiltRangeToFindAxis:F:",
            b":MinTiltRangeToFindAngles:F:",
            b":BoxSizeXandY:IP:",
            b":RoundsOfTracking:I:",
            b":MaxViewsInAlign:I:",
            b":RestrictViewsOnRound:I:",
            b":UnsplitFirstRound:B:",
            b":InitialBidirectionalViews:I:",
            b":LocalAreaTracking:B:",
            b":LocalAreaTargetSize:I:",
            b":MinBeadsInArea:I:",
            b":MaxBeadsInArea:I:",
            b":MinOverlapBeads:I:",
            b":TrackObjectsTogether:B:",
            b":MaxBeadsToAverage:I:",
            b":LowPassCutoffInverseNm:F:",
            b":SobelFilterCentering:B:",
            b":KernelSigmaForSobel:F:",
            b":ScalableSigmaForSobel:F:",
            b":AverageBeadsForSobel:I:",
            b":InterpolationType:I:",
            b":PositionTrialDiamFrac:F:",
            b":MinDiamForParamScaling:F:",
            b":PointsToFitMaxAndMin:IP:",
            b":DensityRescueFractionAndSD:FP:",
            b":DistanceRescueCriterion:F:",
            b":RescueRelaxationDensityAndDistance:FP:",
            b":PostFitRescueResidual:F:",
            b":DensityRelaxationPostFit:F:",
            b":MaxRescueDistance:F:",
            b":ResidualsToAnalyzeMaxAndMin:IP:",
            b":DeletionCriterionMinAndSD:FP:",
            b":SetIndexedParameter:FA:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
            b":BoxOutputFile:FN:",
            b":SnapshotViews:LI:",
            b":SaveAllPointsAreaRound:IP:",
        ];

        _idebug_obj = 49;
        self.tc.min_in_view = 4;
        self.tc.max_h = 0;
        self.tc.fac_metro = 0.25;
        self.tc.n_cycle = 1000;
        self.tc.eps = 0.00002; //was .00001 then .00002
        piece_file = None;
        angle_offset = 0.;
        self.tc.min_views_tilt_ali = 4;
        self.tc.range_do_axis = 10.;
        self.tc.range_do_tilt = 20.;
        if_local_area = 0;
        min_in_area = 8;
        min_bead_overlap = 3;
        local_target = 1000;
        num_rounds = 1;
        self.max_resid = 9;
        ipass_save = 0;
        i_area_save = 0;
        sigma1 = 0.00;
        sigma2 = 0.05;
        radius2 = 0.0;
        radius1 = 0.;
        area_obj_str = "object";
        lim_in_area = 1000;
        limcx_bound = 2500;
        image_binned = 1;
        npad = 8;
        target_sobel = 8.;
        elong_sigma = 0.85;
        taper_frac = 0.2;
        self.cp.edge_median = 0;
        self.cp.get_edge_sd = 0;
        num_wneigh_want = 30;
        split_first_round = true;
        bead_min_diam_for_scaling = 0.;
        initial_bidir_views = 6;
        pixel_size = 0.;
        tilt_inc_min_for_scaling = 1.5;
        cg_edge_diam_frac = 0.1;
        cg_gap_diam_frac = 0.;

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

        if num_opt_arg + num_non_opt_arg == 0 {
            exit_error(
                b"Sequential interactive input is no longer supported; use Etomo to convert a com file to PIP input",
            );
        }
        if pip_get_string(b"ImageFile", &mut filin) != 0 {
            exit_error(b"No image input file specified");
        }
        {
            let mut piece = Vec::new();
            ierr = pip_get_string(b"PieceListFile", &mut piece);
            if ierr == 0 {
                piece_file = Some(piece);
            }
        }
        //
        let filin_str = String::from_utf8_lossy(&filin).into_owned();
        unsafe { iiu_open(1, &filin_str, "RO") };
        iiu_print_header(1, Some("\nInput image file:"));
        unsafe {
            iiu_ret_basic_head(
                1,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin2,
                &mut dmax2,
                &mut dmean2,
            )
        };
        self.m_nx_im = nxyz[0];
        self.m_ny_im = nxyz[1];
        nz = nxyz[2];
        //
        lim_pc_list = nz + 10;
        self.m_ix_pclist = vec![0; lim_pc_list as usize];
        self.m_iy_pclist = vec![0; lim_pc_list as usize];
        self.m_iz_pclist = vec![0; lim_pc_list as usize];
        listz = vec![0; lim_pc_list as usize];
        self.m_prexf = vec![0.; (lim_pc_list * 6) as usize];
        memory_error(true, "piece list arrays");

        let piece_file_str = piece_file
            .as_ref()
            .map(|p| String::from_utf8_lossy(p).into_owned());
        ierr = read_piece_list(
            piece_file_str.as_deref(),
            &mut self.m_ix_pclist,
            &mut self.m_iy_pclist,
            &mut self.m_iz_pclist,
            &mut self.m_npclist,
            lim_pc_list as usize,
        );
        if ierr > 0 {
            exit_error_fmt!(
                "Opening piece list file %s",
                CArg::Str(piece_file_str.as_deref().unwrap_or(""))
            );
        } else {
            exit_from_value_read_error(ierr, "piece list file");
        }
        //
        // if no pieces, set up mocklist
        //
        if self.m_npclist == 0 {
            i = 1;
            while i <= nz {
                self.m_ix_pclist[(i - 1) as usize] = 0;
                self.m_iy_pclist[(i - 1) as usize] = 0;
                self.m_iz_pclist[(i - 1) as usize] = i - 1;
                i += 1;
            }
            self.m_npclist = nz;
        }
        {
            let mut nview_all: usize = 0;
            fill_list_of_piece_z(
                &self.m_iz_pclist[..self.m_npclist as usize],
                &mut listz,
                &mut nview_all,
            );
            self.tc.nview_all = nview_all as i32;
        }
        //
        // assign mx->maxView the same as in alivar based on actual maximum views and allocate
        // WHY IS MISSING 0-based in fortran?
        self.mx.max_view = self.tc.nview_all + 4;
        let max_view = self.mx.max_view as usize;
        self.m_ip_close = vec![0; max_view];
        self.m_iz_close = vec![0; max_view];
        iv_list = vec![0; max_view];
        missing = vec![false; max_view];
        self.m_prev_res = vec![0.; max_view];
        self.tc.tilt_orig = vec![0.; max_view];
        self.tc.gmag_orig = vec![0.; max_view];
        self.tc.rot_orig = vec![0.; max_view];
        self.tc.dxy_save = vec![0.; max_view * 2];
        self.tc.tilt_all = vec![0.; max_view];
        self.tc.ivsep_in = vec![0; max_view * MAXGRP as usize];

        memory_error(true, "arrays for views");
        ierr = 0;
        allocate_mapsep(&mut self.sg, &self.mx, &mut ierr);
        memory_error(ierr == 0, "arrays in mapsep");

        check_piece_list(
            &self.m_ix_pclist,
            1,
            self.m_npclist as usize,
            1,
            self.m_nx_im,
            &mut min_xpiece,
            &mut nx_pieces,
            &mut nx_overlap,
        );
        nx_tot_pix = self.m_nx_im + (nx_pieces - 1) * (self.m_nx_im - nx_overlap);
        check_piece_list(
            &self.m_iy_pclist,
            1,
            self.m_npclist as usize,
            1,
            self.m_ny_im,
            &mut min_ypiece,
            &mut ny_pieces,
            &mut ny_overlap,
        );
        ny_tot_pix = self.m_ny_im + (ny_pieces - 1) * (self.m_ny_im - ny_overlap);
        self.tc.xcen = (nx_tot_pix as f64 / 2.) as f32;
        self.tc.ycen = (ny_tot_pix as f64 / 2.) as f32;
        self.tc.scale_xy = (self.tc.xcen * self.tc.xcen + self.tc.ycen * self.tc.ycen).sqrt();
        //
        // Read in indexed arcane parameters
        self.m_num_to_do = 0;
        if pip_get_float_array(
            b"SetIndexedParameter",
            &mut self.tc.tilt_all,
            &mut self.m_num_to_do,
            self.mx.max_view,
        ) == 0
        {
            if (self.m_num_to_do % 2) != 0 {
                exit_error(b"An odd number of values was entered for SetIndexedParameter");
            }
            ix = 0;
            while ix < self.m_num_to_do {
                iy = b3dnint!(self.tc.tilt_all[ix as usize]);
                yy = self.tc.tilt_all[(ix + 1) as usize];
                match iy {
                    -1 => set_control_int!(self.m_max_bmr_delta_z, "mMaxBmrDeltaZ", yy),
                    -2 => set_control_int!(self.m_max_drb_delta_z, "mMaxDrbDeltaZ", yy),
                    -3 => set_control_int!(self.m_num_bidir_relax_crit, "mNumBidirRelaxCrit", yy),
                    -4 => set_control_int!(self.m_minz_delz_near_zero, "mMinzDelzNearZero", yy),
                    -5 => set_control_int!(self.m_max_delz_near_zero, "mMaxDelzNearZero", yy),
                    1 => set_control_float!(self.m_bmr_lower_elong_lim, "mBmrLowerElongLim", yy),
                    2 => set_control_float!(self.m_bmr_upper_elong_lim, "mBmrUpperElongLim", yy),
                    3 => set_control_float!(
                        self.m_bmr_low_elong_raise_fac,
                        "mBmrLowElongRaiseFac",
                        yy
                    ),
                    4 => set_control_float!(
                        self.m_bmr_high_elong_raise_fac,
                        "mBmrHighElongRaiseFac",
                        yy
                    ),
                    5 => set_control_float!(self.m_bmr_max_elong_sd, "mBmrMaxElongSD", yy),
                    6 => set_control_float!(
                        self.m_bmr_min_wsum_madnratio,
                        "mBmrMinWsumMADNratio",
                        yy
                    ),
                    7 => set_control_float!(
                        self.m_bmr_max_num_sdabove_mean,
                        "mBmrMaxNumSDaboveMean",
                        yy
                    ),
                    8 => set_control_float!(self.m_drb_just_accept_crit, "mDrbJustAcceptCrit", yy),
                    9 => set_control_float!(self.m_drb_just_reject_crit, "mDrbJustRejectCrit", yy),
                    10 => set_control_float!(self.m_drb_accept_min_madns, "mDrbAcceptMinMADNS", yy),
                    11 => set_control_float!(
                        self.m_drb_reject_max_low_madns,
                        "mDrbRejectMaxLowMADNS",
                        yy
                    ),
                    12 => {
                        set_control_float!(self.m_drb_any_type_min_madns, "mDrbAnyTypeMinMADNs", yy)
                    }
                    13 => set_control_float!(
                        self.m_drb_reject_max_high_madns,
                        "mDrbRejectMaxHighMADNS",
                        yy
                    ),
                    14 => set_control_float!(tilt_inc_min_for_scaling, "tiltIncMinForScaling", yy),
                    15 => set_control_float!(cg_edge_diam_frac, "cgEdgeDiamFrac", yy),
                    16 => set_control_float!(cg_gap_diam_frac, "cgGapDiamFrac", yy),
                    17 => set_control_float!(self.m_relax_bidir_fac, "mRelaxBidirFac", yy),
                    _ => exit_error(b"Index out of range in entry to SetIndexedParameter"),
                }
                ix += 2;
            }
        }
        //
        if pip_get_in_out_file(b"InputSeedModel", 1, &mut model_file_bytes) != 0 {
            exit_error(b"No input seed model file specified");
        }
        //
        self.fm.fm_mod_size_type = 2;
        self.fm.fm_boost_read_in_by = nz as f32;
        let seed_name = String::from_utf8_lossy(&model_file_bytes).into_owned();
        if read_fort_model(&seed_name, &mut self.fm).is_err() {
            let tmp = fort_mod_open_error();
            // Fixed in translation (2026-09-26, `BUGS.md`): `beadtrack.cpp:404`
            // passes only the error text to a format with two `%s`, so native
            // prints the error where the file name belongs and `(null)` after
            // it.  The file name and then the error are printed here.
            exit_error_fmt!(
                "Reading seed model file %s: %s",
                CArg::Str(&seed_name),
                CArg::Str(if tmp.is_empty() {
                    "no reason returned"
                } else {
                    &tmp
                })
            );
        }
        if self.fm.n_point == 0 || self.fm.max_mod_obj == 0 {
            exit_error(b"Input seed model is empty");
        }
        //
        // Initial allocations based on model size, and way oversized temp array
        // for object lists, which is also needed for gaps
        self.m_max_all_real = self.fm.max_mod_obj + 10;
        max_olist = b3dmax!(40 * self.m_max_all_real, 100000);
        let mar = self.m_max_all_real as usize;
        seq_dist = vec![0.; mar];
        xxtmp = vec![0.; mar];
        yytmp = vec![0.; mar];
        xvtmp = vec![0.; mar];
        yvtmp = vec![0.; mar];
        self.tc.iobj_seq = vec![0; mar];
        self.tc.xyz_save = vec![0.; 3 * mar];
        iobj_lis_tmp = vec![0; max_olist as usize];
        self.m_ind_gap = vec![0; mar];
        in_an_area = vec![false; mar];
        memory_error(true, "arrays for all contours in model");

        //
        // convert to image index coordinates and change the origin and delta
        // for X / Y to reflect this
        //
        let _ = scale_fort_mod_to_image(&mut self.fm, 1, 0);
        self.tc.xorig = 0.;
        self.tc.yorig = 0.;
        self.tc.xdelt = 1.;
        self.tc.ydelt = 1.;
        //
        // Check for multiple points on a view
        iobj = 1;
        while iobj <= self.fm.max_mod_obj {
            num_in_obj = self.fm.npt_in_obj[(iobj - 1) as usize];
            if num_in_obj > 0 {
                ibase = self.fm.ibase_obj[(iobj - 1) as usize];
                iv_list[..max_view].fill(0);
                ipt = 1;
                while ipt <= num_in_obj {
                    iz = b3dnint!(
                        self.fm.p_coord[(self.fm.object[(ipt + ibase - 1) as usize] - 1) as usize]
                            [2] as f64
                            + 1.
                    );
                    if iz > 0 && iz <= self.tc.nview_all {
                        if iv_list[(iz - 1) as usize] > 0 {
                            fort_mod_obj_to_cont(
                                iobj,
                                &self.fm.obj_color,
                                &mut ibase,
                                &mut num_in_obj,
                            );
                            exit_error_fmt!(
                                "Two points (# %d and %d) on view %d in contour %d of object %d\n",
                                CArg::Int(iv_list[(iz - 1) as usize] as i64),
                                CArg::Int(ipt as i64),
                                CArg::Int(iz as i64),
                                CArg::Int(num_in_obj as i64),
                                CArg::Int(ibase as i64)
                            );
                        } else {
                            iv_list[(iz - 1) as usize] = ipt;
                        }
                    }
                    ipt += 1;
                }
            }
            iobj += 1;
        }
        //
        model_file_bytes.clear();
        if pip_get_in_out_file(b"OutputModel", 2, &mut model_file_bytes) != 0 {
            exit_error(b"No output model file specified");
        }
        let model_file = String::from_utf8_lossy(&model_file_bytes).into_owned();
        //
        out_file = None;
        elong_file = None;
        xyz_file = None;
        self.tc.num_exclude = 0;
        {
            let mut s = Vec::new();
            ierr = pip_get_string(b"BoxOutputFile", &mut s);
            if ierr == 0 {
                out_file = Some(s);
            }
            let mut s = Vec::new();
            ierr = pip_get_string(b"ElongationOutputFile", &mut s);
            if ierr == 0 {
                elong_file = Some(s);
            }
            let mut s = Vec::new();
            ierr = pip_get_string(b"XYZOutputFile", &mut s);
            if ierr == 0 {
                xyz_file = Some(s);
            }
        }
        self.cp.get_edge_sd = elong_file.is_some() as i32;
        if pip_get_string(b"SkipViews", &mut tmp_str) == 0 {
            match parselist(&String::from_utf8_lossy(&tmp_str)) {
                Ok(list) if !list.is_empty() => {
                    self.tc.num_exclude = list.len() as i32;
                    self.tc.iz_exclude = list;
                }
                _ => exit_error(b"Parsing list of excluded views"),
            }
        }
        ierr = pip_get_float(b"RotationAngle", &mut self.m_rot_start);
        //
        // nview needs to be set for this routine to check the groups properly
        self.av.nview = self.tc.nview_all;
        input_separate_groups::<true>(
            &self.av,
            &self.mx,
            &mut self.sg.num_separate_groups,
            &mut self.tc.nsep_in_grp_in,
            &mut self.tc.ivsep_in,
        );

        self.m_iv_bidir_part2 = 0;
        if self.sg.num_separate_groups == 1 {
            ix = 1000000;
            iy = -ix;
            iz = 0;
            while iz < self.tc.nsep_in_grp_in[0] {
                ix = b3dmin!(ix, self.tc.ivsep_in[iz as usize]);
                iy = b3dmax!(iy, self.tc.ivsep_in[iz as usize]);
                iz += 1;
            }
            if ix == 1 && iy == self.tc.nsep_in_grp_in[0] {
                self.m_iv_bidir_part2 = iy + 1;
            }
            if iy == self.av.nview && ix == self.tc.nsep_in_grp_in[0] + 1 {
                self.m_iv_bidir_part2 = ix;
            }
        }
        //
        get_tilt_angles(&mut self.tc.nview_all, &mut self.tc.tilt_all[..max_view]);
        ierr = pip_get_float(b"AngleOffset", &mut angle_offset);
        iv = 1;
        while iv <= self.tc.nview_all {
            self.tc.tilt_all[(iv - 1) as usize] += angle_offset;
            iv += 1;
        }
        //
        // DNM 5 / 3 / 02: accommodate changes to tiltalign by setting up the
        // mapping arrays, adding to automap call
        // DNM 3 / 25 / 05: mapping arrays are going to be used for real
        // 4 / 11 / 05: default is to ignore objects for transferfid situations
        //
        self.av.nfile_views = self.tc.nview_all;
        ignore_objs = if self.tc.nview_all <= 2 { 1 } else { 0 };
        //
        input_groupings::<true>(
            self.tc.nview_all,
            1,
            "TiltDefaultGrouping",
            "TiltNondefaultGroup",
            &mut self.tc.nmap_tilt,
            &mut self.tc.iv_spec_str_tilt,
            &mut self.tc.iv_spec_end_tilt,
            &mut self.tc.nmap_spec_tilt,
            &mut self.tc.n_ran_spec_tilt,
            max_groups,
        );
        //
        input_groupings::<true>(
            self.tc.nview_all,
            1,
            "MagDefaultGrouping",
            "MagNondefaultGroup",
            &mut self.tc.nmap_mag,
            &mut self.tc.iv_spec_str_mag,
            &mut self.tc.iv_spec_end_mag,
            &mut self.tc.nmap_spec_mag,
            &mut self.tc.n_ran_spec_mag,
            max_groups,
        );

        self.tc.nmap_rot = 1;
        self.tc.n_ran_spec_rot = 0;
        input_groupings::<true>(
            self.tc.nview_all,
            0,
            "RotDefaultGrouping",
            "RotNondefaultGroup",
            &mut self.tc.nmap_rot,
            &mut self.tc.iv_spec_str_rot,
            &mut self.tc.iv_spec_end_rot,
            &mut self.tc.nmap_spec_rot,
            &mut self.tc.n_ran_spec_rot,
            max_groups,
        );
        //
        nv_local_in = 0;
        local_view_pass = 0;
        //
        ierr = pip_get_integer(b"ImagesAreBinned", &mut image_binned);
        ierr = pip_get_float(b"PixelSize", &mut pixel_size);
        if pip_get_float(b"LowPassCutoffInverseNm", &mut radius2) == 0 {
            if pixel_size <= 0. {
                exit_error(b"You must enter a pixel size to use a low pass filter");
            }
            radius2 *= pixel_size;
            sigma2 = (0.15 * radius2 as f64) as f32;
        }
        ierr = pip_get_float(b"CentroidRadius", &mut self.m_cg_radius);
        j = pip_get_float(b"BeadDiameter", &mut self.m_diameter);
        if ierr + j != 1 {
            exit_error(b"You must enter either BeadDiameter or CentroidRadius but not both");
        }
        if j == 0 && self.m_diameter <= 0. {
            exit_error(b"You must enter a positive value for BeadDiameter");
        }
        //
        // compute diameter from cgRadius by the formula used to get cgRadius in
        // copytomocoms, then divide by binning and get cgRadius back
        // Start to increase margin above bead size of 20
        if ierr == 0 {
            self.m_diameter = b3dmax!(2., 2. * self.m_cg_radius as f64 - 3.) as f32;
        }
        self.m_diameter /= image_binned as f32;
        self.m_cg_radius =
            (0.5 * (self.m_diameter as f64 + b3dmax!(3., 0.15 * self.m_diameter as f64))) as f32; // 0.2
        if pip_get_two_integers(b"BoxSizeXandY", &mut self.m_nx_box, &mut self.m_ny_box) != 0 {
            exit_error(b"You must enter a box size");
        }

        if image_binned > 1 {
            //
            // Adjust box size if binned.	 See if the current box size matches what copytomocoms
            // would assign for the unbinned bead size; if so then use the same function for
            // the binned diameter
            ierr = 2 * (box_size_from_diam!(self.m_diameter * image_binned as f32) / 2.) as i32;
            if ierr == self.m_nx_box && ierr == self.m_ny_box {
                self.m_nx_box = 2 * (box_size_from_diam!(self.m_diameter) / 2.) as i32;
                self.m_ny_box = self.m_nx_box;
            } else {
                //
                // If not, get the geometric mean box size and the two bead sizes that could
                // generate that size given the formula (adding 1 for rounding purposes)
                // Whichever one regenerates the smaller box turns out to be the right bead size
                // (could match values instead, but this works)
                // So then get the size from a binned down diameter and scale it to set x or y
                box_geo_mean = (self.m_nx_box as f32 * self.m_ny_box as f32).sqrt();
                xtmp = ((box_geo_mean as f64 + 1. - 20.) / 2.) as f32;
                ytmp = ((box_geo_mean as f64 + 1. - 2.) / 3.3) as f32;
                if box_size_from_diam!(xtmp) < box_size_from_diam!(ytmp) {
                    xtmp = box_size_from_diam!(xtmp / image_binned as f32) as f32;
                } else {
                    xtmp = box_size_from_diam!(ytmp / image_binned as f32) as f32;
                }
                self.m_nx_box =
                    2 * (((xtmp * self.m_nx_box as f32 / box_geo_mean) as f64 / 2.) as i32);
                self.m_ny_box =
                    2 * (((xtmp * self.m_ny_box as f32 / box_geo_mean) as f64 / 2.) as i32);
            }
            printf!(
                // Fixed in translation (`BUGS.md`): the source has no newline.
                "Box size in X and Y adjusted for binning to: %d %d\n",
                CArg::Int(self.m_nx_box as i64),
                CArg::Int(self.m_ny_box as i64)
            );
        }
        //
        // Adjust filter for binning and report /pixel values
        if radius2 > 0. {
            radius2 *= image_binned as f32;
            sigma2 *= image_binned as f32;
            if radius2 < 0.5 {
                printf!(
                    "Radius and sigma for low-pass filtering:%6.3f%6.3f/pixel\n",
                    CArg::Dbl(radius2 as f64),
                    CArg::Dbl(sigma2 as f64)
                );
            } else {
                printf!(
                    "No low-pass filtering will be done; the radius times the binning exceeds 0.5/pixel\n"
                );
                radius2 = 0.;
            }
        }
        //
        ierr = pip_get_integer(b"MinViewsForTiltalign", &mut self.tc.min_views_tilt_ali);
        ierr = pip_get_boolean(b"LightBeads", &mut self.m_if_white);
        ierr = pip_get_boolean(b"FillGaps", &mut self.m_if_fill_in);
        ierr = pip_get_integer(b"MaxGapSize", &mut self.m_max_gap);
        ierr = pip_get_float(b"MinTiltRangeToFindAxis", &mut self.tc.range_do_axis);
        ierr = pip_get_float(b"MinTiltRangeToFindAngles", &mut self.tc.range_do_tilt);
        ierr = pip_get_integer(b"MaxViewsInAlign", &mut nv_local_in);
        ierr = pip_get_float(b"PositionTrialDiamFrac", &mut self.m_try_alt_pred_diam_frac);
        ierr = pip_get_integer(b"RestrictViewsOnRound", &mut local_view_pass);
        ierr = pip_get_integer(b"LocalAreaTargetSize", &mut local_target);
        ierr = pip_get_integer(b"LocalAreaTracking", &mut if_local_area);
        ierr = pip_get_integer(b"MinBeadsInArea", &mut min_in_area);
        if if_local_area != 0 {
            ierr = pip_get_integer(b"MaxBeadsInArea", &mut lim_in_area);
        }
        // limInArea = min(limInArea, maxreal)
        ierr = pip_get_integer(b"MinOverlapBeads", &mut min_bead_overlap);
        ierr = pip_get_integer(b"RoundsOfTracking", &mut num_rounds);
        ierr = pip_get_integer(b"MaxBeadsToAverage", &mut self.m_max_sum);
        ierr = pip_get_two_integers(
            b"PointsToFitMaxAndMin",
            &mut self.m_num_fit,
            &mut self.m_min_fit,
        );
        ierr = pip_get_two_floats(
            b"DensityRescueFractionAndSD",
            &mut self.m_frac_crit,
            &mut self.m_sd_crit,
        );
        ierr = pip_get_float(b"DistanceRescueCriterion", &mut self.m_dist_crit);
        ierr = pip_get_two_floats(
            b"RescueRelaxationDensityAndDistance",
            &mut self.m_relax_int,
            &mut self.m_relax_dist,
        );
        ierr = pip_get_float(b"PostFitRescueResidual", &mut self.m_fit_dist_crit);
        ierr = pip_get_float(b"DensityRelaxationPostFit", &mut self.m_relax_fit);
        ierr = pip_get_float(b"MaxRescueDistance", &mut self.m_rad_max_fit);
        ierr = pip_get_two_integers(
            b"ResidualsToAnalyzeMaxAndMin",
            &mut self.max_resid,
            &mut self.m_min_resid,
        );
        ierr = pip_get_two_floats(
            b"DeletionCriterionMinAndSD",
            &mut self.m_res_diff_min,
            &mut self.m_res_diff_crit,
        );
        ierr = pip_get_float(b"MinDiamForParamScaling", &mut bead_min_diam_for_scaling);
        ierr = pip_get_integer(b"TrackObjectsTogether", &mut ignore_objs);
        self.m_max_sobel_sum = 50;
        ierr = pip_get_integer(b"AverageBeadsForSobel", &mut self.m_max_sobel_sum);
        i = 0;
        ierr = pip_get_boolean(b"SobelFilterCentering", &mut i);
        if i == 0 {
            self.m_max_sobel_sum = 0;
        } else {
            ierr = pip_get_float(b"KernelSigmaForSobel", &mut self.m_sobel_sigma);
            xtmp = 0.;
            if pip_get_float(b"ScalableSigmaForSobel", &mut xtmp) == 0 {
                if ierr == 0 {
                    exit_error(
                        b"You cannot enter both KernelSigmaForSobel and ScalableSigmaForSobel",
                    );
                }
                self.m_sobel_sigma = xtmp * self.m_diameter;
                printf!(
                    "Scaled kernel sigma for Sobel filter is %7.2f pixels\n",
                    CArg::Dbl(self.m_sobel_sigma as f64)
                );
            }
        }
        ierr = pip_get_boolean(b"MedianForCentroid", &mut self.cp.edge_median);
        i = 0;
        ierr = pip_get_boolean(b"UnsplitFirstRound", &mut i);
        split_first_round = i == 0;
        if self.m_sobel_sigma as f64 > 1.49 {
            self.m_interp_type = 1;
        }
        ierr = pip_get_integer(b"InterpolationType", &mut self.m_interp_type);
        if self.m_iv_bidir_part2 > 0 {
            ierr = pip_get_integer(b"InitialBidirectionalViews", &mut initial_bidir_views);
        }

        self.m_if_read_xfs = 1 - pip_get_string(b"PrealignTransformFile", &mut prexf_file);
        if self.m_if_read_xfs != 0 {
            let prexf_name = String::from_utf8_lossy(&prexf_file).into_owned();
            let fp = b3d_open_file(&prexf_name, "r");
            let mut reader = BufReader::new(fp);
            ierr = read_all_xforms(&mut reader, &mut self.m_prexf, lim_pc_list, &mut iv);
            exit_from_xf_read_error(ierr, "prealign");
            if iv < nz {
                exit_error(b"Not enough transforms in prealign transform file");
            }
            i = 0;
            while i < nz {
                self.m_prexf[(6 * i + 4) as usize] /= image_binned as f32;
                self.m_prexf[(6 * i + 5) as usize] /= image_binned as f32;
                i += 1;
            }
            // `fclose(mCplFP)`: the reader is dropped here.
            drop(reader);
            self.m_nfill_taper = b3dmax!(
                4,
                b3dnint!(0.1 * (self.m_nx_box + self.m_ny_box) as f64 / 2.)
            );
        }

        if pip_get_float_array(b"ShiftsNearZeroTilt", &mut xxtmp, &mut ierr, 4) == 0 {
            if ierr != 2 && ierr != 4 {
                exit_error(b"There must be 2 or 4 values entered for ShiftsNearZeroTilt");
            }
            i = 0;
            while i < ierr / 2 {
                self.m_x_shift_near_zero[i as usize] =
                    xxtmp[(2 * i) as usize] / image_binned as f32;
                self.m_y_shift_near_zero[i as usize] =
                    xxtmp[(2 * i + 1) as usize] / image_binned as f32;
                i += 1;
            }
        } else {
            self.m_minz_delz_near_zero = 0;
        }

        tmp_str.clear();
        if pip_get_string(b"SnapshotViews", &mut tmp_str) == 0 {
            match parselist(&String::from_utf8_lossy(&tmp_str)) {
                Ok(list) if !list.is_empty() => {
                    self.m_nsnap_list = list.len() as i32;
                    self.m_iv_snap_list = list;
                }
                _ => exit_error(b"Parsing list of views to snapshot"),
            }
        }
        ierr = pip_get_two_integers(b"SaveAllPointsAreaRound", &mut i_area_save, &mut ipass_save);
        pip_done();
        //
        self.cp.i_polarity = -1;
        if self.m_if_white != 0 {
            self.cp.i_polarity = 1;
        }
        //

        if bead_min_diam_for_scaling > 0. {
            if image_binned > 1 && self.m_diameter * image_binned as f32 > bead_min_diam_for_scaling
            {
                param_scale = b3dmax!(bead_min_diam_for_scaling, self.m_diameter)
                    / (self.m_diameter * image_binned as f32);
                self.m_dist_crit *= param_scale;
                self.m_fit_dist_crit *= param_scale;
                self.m_rad_max_fit *= param_scale;
                self.m_res_diff_min *= param_scale;
                printf!(
                    // Fixed in translation (`BUGS.md`): the source has `/n` for `\n`.
                    "\nTo compensate for binning, parameters were scaled by%6.3f:\n	Distance rescue criterion -> %6.2f	 Post-fit rescue criterion -> %6.2f\n	Max rescue distance ->%6.2f		Residual criterion for deletion ->%7.3f\n",
                    CArg::Dbl(param_scale as f64),
                    CArg::Dbl(self.m_dist_crit as f64),
                    CArg::Dbl(self.m_fit_dist_crit as f64),
                    CArg::Dbl(self.m_rad_max_fit as f64),
                    CArg::Dbl(self.m_res_diff_min as f64)
                );
            }
            if self.m_diameter > bead_min_diam_for_scaling {
                self.m_cur_res_min *= self.m_diameter / bead_min_diam_for_scaling;
            }
            self.m_rescue_step_size =
                b3dmax!(1., (self.m_diameter / bead_min_diam_for_scaling) as f64) as f32;
        }
        xtmp = 1.0e10;
        ytmp = -xtmp;
        ix = 0;
        while ix < self.tc.nview_all {
            xtmp = b3dmin!(xtmp, self.tc.tilt_all[ix as usize]);
            ytmp = b3dmax!(ytmp, self.tc.tilt_all[ix as usize]);
            ix += 1;
        }
        tilt_increment = (ytmp - xtmp) / b3dmax!(1, self.tc.nview_all) as f32;
        if tilt_inc_min_for_scaling > 0. && tilt_increment > tilt_inc_min_for_scaling {
            param_scale = b3dmin!(
                2.,
                (tilt_increment / tilt_inc_min_for_scaling).sqrt() as f64
            ) as f32;
            self.m_res_diff_min *= param_scale;
            self.m_res_diff_crit *= param_scale;
            printf!(
                "To compensate for larger tilt increment, residual change criteria for deletion\n  were scaled by%6.3f: to%7.3f%7.2f\n",
                CArg::Dbl(param_scale as f64),
                CArg::Dbl(self.m_res_diff_min as f64),
                CArg::Dbl(self.m_res_diff_crit as f64)
            );
        }

        //
        // Set up tapering.	 It is inside, so make box a little bigger but not much
        // Some tracking deterioriated with it larger.
        // keep the box even sized
        self.m_nx_taper = b3dmax!(8, b3dnint!(taper_frac * self.m_nx_box as f32));
        self.m_ny_taper = b3dmax!(8, b3dnint!(taper_frac * self.m_ny_box as f32));
        self.m_nx_box = 2 * ((self.m_nx_box + self.m_nx_taper / 2 + 1) / 2);
        self.m_ny_box = 2 * ((self.m_ny_box + self.m_ny_taper / 2 + 1) / 2);
        self.m_nx_pad = nice_frame(self.m_nx_box + 2 * npad, 2, 19);
        self.m_ny_pad = nice_frame(self.m_ny_box + 2 * npad, 2, 19);
        self.m_npix_box = self.m_nx_box * self.m_ny_box;
        self.m_nxp_dim = self.m_nx_pad + 2;
        self.m_nx_stat_box = self.m_nx_box + 2 * b3dnint!(2. * self.m_cg_radius as f64);
        self.m_ny_stat_box = self.m_ny_box + 2 * b3dnint!(2. * self.m_cg_radius as f64);
        self.m_nx_stat_pad = nice_frame(self.m_nx_stat_box + 2 * npad, 2, 19);
        self.m_ny_stat_pad = nice_frame(self.m_ny_stat_box + 2 * npad, 2, 19);
        maxarr = (self.m_nx_stat_pad + 2) * self.m_ny_stat_pad;
        if sigma1 != 0. || radius2 != 0. {
            xcorr_set_ctf(
                sigma1,
                sigma2,
                radius1,
                radius2,
                &mut self.m_ctf,
                self.m_nx_pad,
                self.m_ny_pad,
                &mut self.m_delta_ctf,
            );
            xcorr_set_ctf(
                sigma1,
                sigma2,
                radius1,
                radius2,
                &mut self.m_ctf_stat,
                self.m_nx_stat_pad,
                self.m_ny_stat_pad,
                &mut self.m_delta_ctf_stat,
            );
        }
        //
        self.m_if_trace = 0;
        if self.m_dist_crit < 0. {
            self.m_if_trace = 1;
            self.m_dist_crit = -self.m_dist_crit;
        }
        //
        self.m_tilt_max = 0.;
        iv = 0;
        while iv < self.tc.nview_all {
            let ivu = iv as usize;
            self.tc.rot_orig[ivu] = self.m_rot_start;
            self.tc.tilt_orig[ivu] = self.tc.tilt_all[ivu];
            self.tc.gmag_orig[ivu] = 1.;
            self.tc.dxy_save[ivu * 2] = 0.;
            self.tc.dxy_save[ivu * 2 + 1] = 0.;
            self.m_tilt_max = b3dmax!(self.m_tilt_max, b3dabs!(self.tc.tilt_all[ivu]));
            iv += 1;
        }
        self.av.map_dum_dmag = 0;
        self.av.map_dmag_start = 1;
        //
        // figure out where to start: section with most points at minimum tilt
        // ivList is the number of points present on each view
        // ivGap has a list of views with gaps for each object (if not filling in)
        // indgap is index to location in ivGap for each object
        // minEndZ is the minimum ending Z of all the objects
        // use iobjLisTmp for ivGap
        //
        min_end_z = self.mx.max_view;
        ind_free = 1;
        iv_list[..max_view].fill(0);
        iobj = 1;
        while iobj <= self.fm.max_mod_obj {
            num_in_obj = self.fm.npt_in_obj[(iobj - 1) as usize];
            self.m_ind_gap[(iobj - 1) as usize] = ind_free;
            if num_in_obj > 0 {
                ibase = self.fm.ibase_obj[(iobj - 1) as usize];
                //
                // first order them by z
                //
                ipt = 1;
                while ipt <= num_in_obj - 1 {
                    jpt = ipt + 1;
                    while jpt <= num_in_obj {
                        iz = b3dnint!(
                            self.fm.p_coord
                                [(self.fm.object[(ipt + ibase - 1) as usize] - 1) as usize][2]
                        ) + 1;
                        jz = b3dnint!(
                            self.fm.p_coord
                                [(self.fm.object[(jpt + ibase - 1) as usize] - 1) as usize][2]
                        ) + 1;
                        if iz > jz {
                            itmp = self.fm.object[(ipt + ibase - 1) as usize];
                            self.fm.object[(ipt + ibase - 1) as usize] =
                                self.fm.object[(jpt + ibase - 1) as usize];
                            self.fm.object[(jpt + ibase - 1) as usize] = itmp;
                        }
                        jpt += 1;
                    }
                    ipt += 1;
                }
                ipt = 1;
                while ipt <= num_in_obj {
                    iz = b3dnint!(
                        self.fm.p_coord[(self.fm.object[(ipt + ibase - 1) as usize] - 1) as usize]
                            [2]
                    ) + 1;
                    iv_list[(iz - 1) as usize] += 1;
                    ipt += 1;
                }
                min_end_z = b3dmin!(min_end_z, iz);
                if self.m_if_fill_in == 0 {
                    //
                    // find gaps and add to list
                    //
                    ipt = 1;
                    while ipt <= num_in_obj - 1 {
                        iz = b3dnint!(
                            self.fm.p_coord
                                [(self.fm.object[(ipt + ibase - 1) as usize] - 1) as usize][2]
                        ) + 1;
                        self.m_iz_next = b3dnint!(
                            self.fm.p_coord
                                [(self.fm.object[(ipt + 1 + ibase - 1) as usize] - 1) as usize][2]
                        ) + 1;
                        if self.m_iz_next > iz + 1 {
                            iv = iz + 1;
                            while iv <= self.m_iz_next - 1 {
                                iobj_lis_tmp[(ind_free - 1) as usize] = iv;
                                ind_free += 1;
                                if ind_free > max_olist {
                                    exit_error(b"Too many gaps in existing model for arrays");
                                }
                                iv += 1;
                            }
                        }
                        ipt += 1;
                    }
                }
            }
            iobj += 1;
        }
        let _ = min_end_z;
        self.m_ind_gap[self.fm.max_mod_obj as usize] = ind_free;
        //
        // allocate ivGap and copy to it
        lim_gaps = ind_free;
        self.m_iv_gap = vec![0; lim_gaps as usize];
        memory_error(true, "array for gaps");
        if ind_free > 1 {
            copy_array(&mut self.m_iv_gap, 1, ind_free - 1, &iobj_lis_tmp, 1);
        }
        //
        // now find section with most points
        //
        maxnpt = 0;
        tilt_min = 200.;
        i = 1;
        while i <= self.tc.nview_all {
            let iu = (i - 1) as usize;
            if iv_list[iu] > maxnpt
                || (iv_list[iu] == maxnpt && b3dabs!(self.tc.tilt_all[iu]) < tilt_min)
            {
                maxnpt = iv_list[iu];
                tilt_min = b3dabs!(self.tc.tilt_all[iu]);
                self.tc.min_tilt_ind = i;
            }
            i += 1;
        }

        // Get range of sections that have more than half of maximum
        min_filled = self.tc.min_tilt_ind;
        max_filled = self.tc.min_tilt_ind;
        i = 1;
        while i <= self.tc.nview_all {
            if iv_list[(i - 1) as usize] >= maxnpt / 2 {
                min_filled = b3dmin!(min_filled, i);
                max_filled = b3dmax!(max_filled, i);
            }
            i += 1;
        }
        //
        // Get minimim and maximum views to do: including minTiltInd even if
        // it is excluded
        self.m_min_view_do = 0;
        iv = 1;
        while iv <= self.tc.nview_all && self.m_min_view_do == 0 {
            self.m_min_view_do = iv;
            if iv != self.tc.min_tilt_ind
                && number_in_list(
                    iv,
                    list_or_null(&self.tc.iz_exclude),
                    self.tc.num_exclude,
                    0,
                ) != 0
            {
                self.m_min_view_do = 0;
            }
            iv += 1;
        }
        self.m_max_view_do = 0;
        iv = self.tc.nview_all;
        while iv >= 1 && self.m_max_view_do == 0 {
            self.m_max_view_do = iv;
            if iv != self.tc.min_tilt_ind
                && number_in_list(
                    iv,
                    list_or_null(&self.tc.iz_exclude),
                    self.tc.num_exclude,
                    0,
                ) != 0
            {
                self.m_max_view_do = 0;
            }
            iv -= 1;
        }
        num_view_do = self.m_max_view_do + 1 - self.m_min_view_do;
        //print *,minTiltInd, mMinViewDo, mMaxViewDo

        // Forget about bidirectional treatment if it is effectively outside the tracking range
        // and clear out the initialBidir if there is no part 2 identified
        if self.m_iv_bidir_part2 <= self.m_min_view_do
            || self.m_iv_bidir_part2 >= self.m_max_view_do
        {
            self.m_iv_bidir_part2 = 0;
        }
        if self.m_iv_bidir_part2 == 0 {
            initial_bidir_views = 0;
        }

        // Determine if first round will be split
        if split_first_round {
            split_first_round = self.m_max_view_do - self.tc.min_tilt_ind > 5
                && self.tc.min_tilt_ind - self.m_min_view_do > 5
                && b3dabs!(self.tc.tilt_all[(self.m_min_view_do - 1) as usize]) as f64 > 30.
                && b3dabs!(self.tc.tilt_all[(self.m_max_view_do - 1) as usize]) as f64 > 30.
                && max_filled - self.tc.min_tilt_ind
                    < (self.m_max_view_do - self.tc.min_tilt_ind) / 3
                && self.tc.min_tilt_ind - min_filled
                    < (self.tc.min_tilt_ind - self.m_min_view_do) / 3;
            if split_first_round {
                printf!(" First round of tracking will be split in two\n");
            } else {
                printf!(" First round of tracking will NOT be split in two; it is not sensible\n");
            }
        }
        //
        // figure out an order for the points: from the center outwards
        // i.e., first find position from center at minimum tilt, save that in
        // xyzSave, and store the square of distance in seqDist
        // iobjSeq is just a list of objects to do in original order
        //
        self.tc.num_obj_do = 0;
        iobj = 0;
        while iobj < self.fm.max_mod_obj {
            let iobu = iobj as usize;
            num_in_obj = self.fm.npt_in_obj[iobu];
            if num_in_obj > 0 {
                self.tc.num_obj_do += 1;
                self.tc.iobj_seq[(self.tc.num_obj_do - 1) as usize] = iobj + 1;
                tilt_min = 200.;
                ibase = self.fm.ibase_obj[iobu];
                ipt = 1;
                while ipt <= num_in_obj {
                    let pt =
                        self.fm.p_coord[(self.fm.object[(ipt + ibase - 1) as usize] - 1) as usize];
                    iz = b3dnint!(pt[2]) + 1;
                    if b3dabs!(self.tc.tilt_all[(iz - 1) as usize]) < tilt_min {
                        tilt_min = b3dabs!(self.tc.tilt_all[(iz - 1) as usize]);
                        self.tc.xyz_save[iobu * 3] = pt[0] - self.tc.xcen;
                        self.tc.xyz_save[3 * iobu + 1] = pt[1] - self.tc.ycen;
                        seq_dist[(self.tc.num_obj_do - 1) as usize] = self.tc.xyz_save[iobu * 3]
                            * self.tc.xyz_save[iobu * 3]
                            + self.tc.xyz_save[3 * iobu + 1] * self.tc.xyz_save[3 * iobu + 1];
                        xxtmp[(self.tc.num_obj_do - 1) as usize] = self.tc.xyz_save[iobu * 3];
                        yytmp[(self.tc.num_obj_do - 1) as usize] = self.tc.xyz_save[3 * iobu + 1];
                    }
                    ipt += 1;
                }
                self.tc.xyz_save[3 * iobu + 2] = 0.;
            }
            iobj += 1;
        }

        if if_local_area != 0 && min_bead_overlap > 0 {
            //
            // determine an average density from the area of the convex bound
            // First get a random subset not to exceed the limiting number
            num_bound = self.tc.num_obj_do;
            if self.tc.num_obj_do > limcx_bound {
                num_bound = 0;
                iseed = 12345679;
                b3dsrand(&iseed);
                ran_frac = limcx_bound as f32 / self.tc.num_obj_do as f32;
                i = 1;
                while i <= self.tc.num_obj_do {
                    // Fixed in translation (2026-09-26, `BUGS.md`): the source's
                    // `rand() / RAND_MAX` is an integer quotient, 0 except for one
                    // draw, so its "random subset" is simply the first
                    // `limcxBound` beads.  The draw is compared as the fraction
                    // it is meant to be, `(float)rand() / RAND_MAX` — `b3drand`,
                    // the same seeded generator — so the subset is the intended
                    // deterministic random one.
                    if b3drand() < ran_frac && num_bound < limcx_bound {
                        num_bound += 1;
                        xxtmp[(num_bound - 1) as usize] = xxtmp[(i - 1) as usize];
                        yytmp[(num_bound - 1) as usize] = yytmp[(i - 1) as usize];
                    }
                    i += 1;
                }
            }
            //
            convex_bound(
                &xxtmp[..num_bound as usize],
                &yytmp[..num_bound as usize],
                0.,
                (2. * self.m_cg_radius as f64) as f32,
                &mut xvtmp,
                &mut yvtmp,
                &mut num_vert,
                &mut cvbxcen,
                &mut cvbycen,
            );
            area = 0.;
            i = 1;
            while i <= num_vert {
                j = i % num_vert + 1;
                area = (area as f64
                    + 0.5
                        * (yvtmp[(j - 1) as usize] + yvtmp[(i - 1) as usize]) as f64
                        * (xvtmp[(j - 1) as usize] - xvtmp[(i - 1) as usize]) as f64)
                    as f32;
                i += 1;
            }
            density = self.tc.num_obj_do as f32 / b3dabs!(area);
        }
        //
        // set up for one area in X and Y, then compute number of areas and
        // overlaps if locals
        //
        num_area_x = 1;
        num_area_y = 1;
        nx_overlap = 0;
        ny_overlap = 0;
        done = false;
        while !done {
            done = true;
            if if_local_area != 0 {
                area_obj_str = "area  ";
                n_overlap = 0;
                //
                // set target overlap so that 1.5 overlap areas at this density
                // will give minimum number of overlap beads
                // 6 / 21 / 08: but constrain it to be smaller than the target itself
                if min_bead_overlap > 0 {
                    n_overlap = b3dmin!(
                        (min_bead_overlap as f32 / (density * local_target as f32)) as f64,
                        0.8 * local_target as f64
                    ) as i32;
                    //print *,area, density, nOverlap
                }
                //
                // get number of areas, round up so areas will start below target
                // Then compute or set overlaps so local sizes can be computed
                //
                num_area_x =
                    (nx_tot_pix + local_target - 2 * n_overlap - 2) / (local_target - n_overlap);
                num_area_y =
                    (ny_tot_pix + local_target - 2 * n_overlap - 2) / (local_target - n_overlap);
                if num_area_x == 1 {
                    if num_area_y > 1 {
                        ny_overlap = n_overlap;
                    }
                } else if num_area_y == 1 {
                    nx_overlap = n_overlap;
                } else {
                    nx_local = b3dmax!(nx_tot_pix / num_area_x, ny_tot_pix / num_area_y) + 1;
                    while (nx_local * num_area_x - nx_tot_pix) / (num_area_x - 1)
                        + (nx_local * num_area_y - ny_tot_pix) / (num_area_y - 1)
                        < 2 * n_overlap
                    {
                        nx_local += 1;
                    }
                    ny_local = nx_local;
                    nx_overlap = (nx_local * num_area_x - nx_tot_pix) / (num_area_x - 1);
                    ny_overlap = (ny_local * num_area_y - ny_tot_pix) / (num_area_y - 1);
                }
            }
            //
            // Get area sizes regardless; compute area distances from center
            //
            nx_local = (nx_tot_pix + nx_overlap * (num_area_x - 1)) / num_area_x + 1;
            ny_local = (ny_tot_pix + ny_overlap * (num_area_y - 1)) / num_area_y + 1;
            num_area_tot = num_area_x * num_area_y;
            if if_local_area == 0 && getimodobjsize() > 0 && ignore_objs == 0 {
                num_area_tot = getimodobjsize();
            }
            max_area = num_area_tot + 10;
            //
            // Get arrays for areas
            ix = 2 * num_rounds * max_area;
            if split_first_round {
                ix += 2 * max_area;
            }
            if initial_bidir_views > 0 {
                ix += 2 * max_area;
            }
            iarea_seq = vec![0; max_area as usize];
            nin_obj_list = vec![0; max_area as usize];
            ind_obj_list = vec![0; max_area as usize];
            area_dist = vec![0.; max_area as usize];
            self.m_iv_seq_str = vec![0; ix as usize];
            self.m_iv_seq_end = vec![0; ix as usize];
            self.m_list_seq = vec![0; ix as usize];
            xxtmp = vec![0.; (2 * max_area) as usize];
            yytmp = vec![0.; (3 * max_area) as usize];
            memory_error(true, "arrays for area data");

            i = 0;
            while i < num_area_tot {
                let iu = i as usize;
                if if_local_area == 0 && getimodobjsize() > 0 && ignore_objs == 0 {
                    //
                    // If there are no local areas but there are multiple objects, set up
                    // one area per object.  No point getting distance, they are independent
                    iarea_seq[iu] = i + 1;
                    area_dist[iu] = 0.;
                } else {
                    ix = i % num_area_x;
                    iy = i / num_area_x;
                    iarea_seq[iu] = i + 1;
                    let dx = (ix * (nx_local - nx_overlap) + nx_local / 2) as f32 - self.tc.xcen;
                    let dy = (iy * (ny_local - ny_overlap) + ny_local / 2) as f32 - self.tc.ycen;
                    area_dist[iu] = dx * dx + dy * dy;
                }
                i += 1;
            }
            //
            // order areas by distance: iareaSeq has sequence of area numbers
            i = 0;
            while i < num_area_tot - 1 {
                j = i + 1;
                while j < num_area_tot {
                    if area_dist[(iarea_seq[i as usize] - 1) as usize]
                        > area_dist[(iarea_seq[j as usize] - 1) as usize]
                    {
                        iarea_seq.swap(i as usize, j as usize);
                    }
                    j += 1;
                }
                i += 1;
            }
            //
            // go through each area finding points within it
            // iobjLists is the list of objects to do for each area
            // indObjList is the starting index in that list for each area
            // ninObjList is the number of objects in the list for each area
            //
            nobj_lists = 0;
            ind_free = 1;
            max_in_area = 0;
            in_an_area[..self.fm.max_mod_obj as usize].fill(false);
            k = 1;
            while k <= num_area_tot {
                if if_local_area != 0 || num_area_tot == 1 {
                    //
                    // get starting coordinates and set up to loop until conditions met
                    //
                    ix = (iarea_seq[(k - 1) as usize] - 1) % num_area_x;
                    iy = (iarea_seq[(k - 1) as usize] - 1) / num_area_x;
                    xst = (ix * (nx_local - nx_overlap)) as f32 - self.tc.xcen;
                    xnd = xst + nx_local as f32;
                    yst = (iy * (ny_local - ny_overlap)) as f32 - self.tc.ycen;
                    ynd = yst + ny_local as f32;
                    ind_start = ind_free;
                    keep_going = true;

                    while keep_going {
                        ind_free = ind_start;
                        num_new = 0;
                        //
                        // Look for objects in the area, count up the new ones
                        //
                        j = 1;
                        while j <= self.tc.num_obj_do {
                            iobj = self.tc.iobj_seq[(j - 1) as usize];
                            let ob = (iobj - 1) as usize;
                            if self.tc.xyz_save[ob * 3] >= xst
                                && self.tc.xyz_save[ob * 3] <= xnd
                                && self.tc.xyz_save[ob * 3 + 1] >= yst
                                && self.tc.xyz_save[ob * 3 + 1] <= ynd
                            {
                                iobj_lis_tmp[(ind_free - 1) as usize] = iobj;
                                ind_free += 1;
                                if ind_free > max_olist {
                                    exit_error(
                                        b"Way too many local areas; each point is in many areas",
                                    );
                                }
                                if !in_an_area[ob] {
                                    num_new += 1;
                                }
                            }
                            j += 1;
                        }
                        //
                        // make area bigger if there are any new points at all in it and
                        // it does not already have all the points, and
                        // either the total in it is too low or this is an area after the
                        // first and the old ones are too low for overlap
                        //
                        keep_going = (num_new > 0
                            && ind_free - ind_start < self.tc.num_obj_do
                            && (ind_free - ind_start < min_in_area
                                || (k > 1 && ind_free - ind_start - num_new < min_bead_overlap)))
                            || (num_new == 0 && k == 1);
                        if keep_going {
                            j = b3dmax!(1, local_target / 200);
                            xst -= j as f32;
                            xnd += j as f32;
                            yst -= j as f32;
                            ynd += j as f32;
                        }
                    }
                } else {
                    //
                    // areas from objects: make list of objects in it
                    //
                    num_new = 0;
                    ind_start = ind_free;
                    j = 1;
                    while j <= self.tc.num_obj_do {
                        iobj = self.tc.iobj_seq[(j - 1) as usize];
                        fort_mod_obj_to_cont(iobj, &self.fm.obj_color, &mut ix, &mut iy);
                        if ix == k {
                            num_new += 1;
                            iobj_lis_tmp[(ind_free - 1) as usize] = iobj;
                            ind_free += 1;
                        }
                        j += 1;
                    }
                }
                //
                if num_new > 0 {
                    //
                    // if the area has new points, order the list by distance from
                    // center
                    // (Should that be center of area?  It is center of whole field)
                    //
                    nobj_lists += 1;
                    ind_obj_list[(nobj_lists - 1) as usize] = ind_start;
                    nin_obj_list[(nobj_lists - 1) as usize] = ind_free - ind_start;
                    i = ind_start;
                    while i <= ind_free - 1 {
                        in_an_area[(iobj_lis_tmp[(i - 1) as usize] - 1) as usize] = true;
                        i += 1;
                    }
                    max_in_area = b3dmax!(max_in_area, nin_obj_list[(nobj_lists - 1) as usize]);
                    i = ind_start;
                    while i <= ind_free - 2 {
                        j = i + 1;
                        while j <= ind_free - 1 {
                            if seq_dist[(iobj_lis_tmp[(i - 1) as usize] - 1) as usize]
                                > seq_dist[(iobj_lis_tmp[(j - 1) as usize] - 1) as usize]
                            {
                                iobj_lis_tmp.swap((i - 1) as usize, (j - 1) as usize);
                            }
                            j += 1;
                        }
                        i += 1;
                    }
                    xxtmp[(2 * nobj_lists - 2) as usize] = xst + self.tc.xcen;
                    xxtmp[(2 * nobj_lists - 1) as usize] = xnd + self.tc.xcen;
                    yytmp[(3 * nobj_lists - 2) as usize] = yst + self.tc.ycen;
                    yytmp[(3 * nobj_lists - 1) as usize] = ynd + self.tc.ycen;
                    yytmp[(3 * nobj_lists - 3) as usize] = num_new as f32;
                } else {
                    ind_free = ind_start;
                }
                k += 1;
            }
            if if_local_area != 0 && max_in_area > lim_in_area && local_target > 100 {
                local_target = (0.98 * local_target as f64) as i32;
                done = false;
            }
        }

        if if_local_area != 0 {
            if max_in_area > lim_in_area {
                exit_error(b"The number of points in some local areas is above the limit");
            }
            printf!(
                " Local area number, size, overlap - X: %d %d %d,  Y: %d %d %d\n",
                CArg::Int(num_area_x as i64),
                CArg::Int(nx_local as i64),
                CArg::Int(nx_overlap as i64),
                CArg::Int(num_area_y as i64),
                CArg::Int(ny_local as i64),
                CArg::Int(ny_overlap as i64)
            );
            i = 1;
            while i <= nobj_lists {
                printf!(
                    "Area%4d, X:%6d to%6d, Y:%6d to%6d,%5d points,%5d new\n",
                    CArg::Int(i as i64),
                    CArg::Int(b3dnint!(xxtmp[(2 * i - 2) as usize]) as i64),
                    CArg::Int(b3dnint!(xxtmp[(2 * i - 1) as usize]) as i64),
                    CArg::Int(b3dnint!(yytmp[(3 * i - 2) as usize]) as i64),
                    CArg::Int(b3dnint!(yytmp[(3 * i - 1) as usize]) as i64),
                    CArg::Int(nin_obj_list[(i - 1) as usize] as i64),
                    CArg::Int(b3dnint!(yytmp[(3 * i - 3) as usize]) as i64)
                );
                i += 1;
            }
            // elseif (maxInArea > maxreal) then
            // call exitError( 'Too many points for arrays - try local tracking')
        }
        //
        // Allocate resMean array based on maximum object # and maximum view sequence #
        // which is # of views for multiple areas, but times # of rounds if only one area
        lim_resid = num_view_do;
        if nobj_lists == 1 {
            lim_resid *= num_rounds;
        }
        self.m_res_mean = vec![0.; (lim_resid * self.m_max_all_real) as usize];
        memory_error(true, "array for mean residuals");
        //
        // Maximum Number of points finally known for tiltalign solutions: allocate
        ix = max_in_area * self.tc.nview_all;
        ierr = 0;
        allocate_alivar(
            &mut self.av,
            &mut self.mx,
            ix,
            self.tc.nview_all,
            max_in_area,
            &mut ierr,
        );
        memory_error(ierr == 0, "arrays in av->ar");
        self.m_eval_funct
            .allocate_funct_vars(&self.av, &self.mx, &mut ierr);
        memory_error(ierr == 0, "arrays for funct");
        self.av.comp[..max_view].fill(1.);
        self.av.map_comp[..max_view].fill(0);
        self.av.skew[..max_view].fill(0.);
        self.av.map_skew[..max_view].fill(0);
        self.av.dmag[..max_view].fill(0.);
        self.av.map_dmag[..max_view].fill(0);
        self.av.alf[..max_view].fill(0.);
        self.av.map_alf[..max_view].fill(0);
        //
        self.m_max_any_sum = b3dmax!(self.m_max_sobel_sum, self.m_max_sum);

        // Get final arrays for the object lists and free temporary stuff
        max_olist = ind_free;
        self.m_max_neigh = b3dmax!(num_wneigh_want, max_in_area);
        iobj_lists = vec![0; max_olist as usize];
        self.m_in_core = vec![0; (max_in_area * self.m_max_any_sum) as usize];
        self.m_num_wneighbors = vec![0; max_area as usize];
        self.m_neighbors_for_wfits = vec![0; (max_area * self.m_max_neigh) as usize];
        iobj_map = vec![0; self.fm.max_mod_obj as usize];
        memory_error(true, "arrays for object lists");

        if elong_file.is_some() || xyz_file.is_some() {
            resid_lists = vec![-1.; max_olist as usize];
            xyz_all_area = vec![0.; (max_olist * 3) as usize];
            num_res_saved = vec![0; self.m_max_all_real as usize];
            memory_error(true, "arrays for residuals/XYZ data");
        }
        copy_array(&mut iobj_lists, 1, ind_free - 1, &iobj_lis_tmp, 1);

        // Make additional lists of neighbors for Wsum fitting and analysis
        self.m_max_wneigh = 0;
        self.m_isequence = 0;
        while self.m_isequence < nobj_lists {
            let isq = self.m_isequence as usize;
            let max_neigh = self.m_max_neigh as usize;
            self.tc.num_obj_do = nin_obj_list[isq];
            self.m_num_wneighbors[isq] = self.tc.num_obj_do;

            // Copy over this area's objects and get their centroid
            xpos = 0.;
            ypos = 0.;
            i = 1;
            while i <= self.tc.num_obj_do {
                iobj = iobj_lists[(ind_obj_list[isq] + i - 2) as usize];
                xpos += self.tc.xyz_save[((iobj - 1) * 3) as usize] / self.tc.num_obj_do as f32;
                ypos += self.tc.xyz_save[((iobj - 1) * 3 + 1) as usize] / self.tc.num_obj_do as f32;
                self.m_neighbors_for_wfits[isq * max_neigh + (i - 1) as usize] = iobj;
                i += 1;
            }
            self.m_max_wneigh = b3dmax!(self.m_max_wneigh, self.tc.num_obj_do);
            if self.tc.num_obj_do >= num_wneigh_want {
                self.m_isequence += 1;
                continue;
            }

            // Make a list of ones outside this area and their distance to the centroid
            self.tc.num_obj_do = 0;
            iobj = 1;
            while iobj <= self.fm.max_mod_obj {
                if self.fm.npt_in_obj[(iobj - 1) as usize] > 0
                    && number_in_list(
                        iobj,
                        Some(&self.m_neighbors_for_wfits[isq * max_neigh..]),
                        nin_obj_list[isq],
                        0,
                    ) == 0
                {
                    iobj_lis_tmp[self.tc.num_obj_do as usize] = self.tc.num_obj_do; // Is now numbered from 0
                    iobj_map[self.tc.num_obj_do as usize] = iobj;
                    self.tc.num_obj_do += 1;
                    // Fixed in translation (2026-09-26, `BUGS.md`): the source reads
                    // `tc->xyzSave[(iobj - 1) * 2]` and `[(iobj - 1) * + 1]`
                    // (`beadtrack.cpp:1255-1256`), stride 2 and 1 in a stride-3
                    // array; the bead's own X and Y, `[(iobj - 1) * 3]` and
                    // `[(iobj - 1) * 3 + 1]`, are read here.
                    let dx = self.tc.xyz_save[((iobj - 1) * 3) as usize] - xpos;
                    let dy = self.tc.xyz_save[((iobj - 1) * 3 + 1) as usize] - ypos;
                    seq_dist[(self.tc.num_obj_do - 1) as usize] = dx * dx + dy * dy;
                }
                iobj += 1;
            }
            if self.tc.num_obj_do == 0 {
                self.m_isequence += 1;
                continue;
            }

            // sort and add to neightbor list
            rs_sort_indexed_floats(&seq_dist, &mut iobj_lis_tmp, self.tc.num_obj_do);
            i = 1;
            while i
                <= b3dmin!(
                    num_wneigh_want - self.m_num_wneighbors[isq],
                    self.tc.num_obj_do
                )
            {
                self.m_num_wneighbors[isq] += 1;
                self.m_neighbors_for_wfits
                    [isq * max_neigh + (self.m_num_wneighbors[isq] - 1) as usize] =
                    iobj_map[iobj_lis_tmp[(i - 1) as usize] as usize];
                i += 1;
            }
            self.m_max_wneigh = b3dmax!(self.m_max_wneigh, self.m_num_wneighbors[isq]);
            self.m_isequence += 1;
        }

        // Finally done with these temp arrays
        drop(iobj_lis_tmp);
        drop(xxtmp);
        drop(yytmp);
        drop(xvtmp);
        drop(yvtmp);
        drop(seq_dist);
        drop(iobj_map);

        //
        // Allocate boxes and other image arrays
        self.m_boxes = vec![0.; (self.m_max_any_sum * max_in_area * self.m_npix_box) as usize];
        self.m_corr_sum = vec![0.; (max_in_area * self.m_npix_box) as usize];
        self.m_cur_sum = vec![0.; self.m_npix_box as usize];
        self.m_box_tmp = vec![0.; maxarr as usize];
        self.m_stat_tmp = vec![0.; maxarr as usize];
        self.m_array = vec![0.; maxarr as usize];
        self.m_brray = vec![0.; maxarr as usize];
        memory_error(true, "arrays for boxes of image data");

        //
        // Allocate arrays for all real points in current solution
        i = max_in_area + 10;
        let nbox = (self.m_nx_box * self.m_ny_box) as usize;
        self.cp.elong_smooth = vec![0.; nbox];
        self.cp.elong_mask = vec![0; nbox];
        self.cp.ix_elong = vec![0; nbox];
        self.cp.iy_elong = vec![0; nbox];

        memory_error(true, "arrays for elongation");

        let n_real_view = (self.m_max_all_real * self.m_max_view_do) as usize;
        if self.cp.get_edge_sd != 0 {
            self.m_edge_sd_save = vec![-1.; n_real_view];
            memory_error(true, "array for edge SDs");
        }
        scaled_gaussian_kernel(
            &mut self.cp.elong_kernel,
            &mut self.cp.kern_dim_elong,
            7,
            b3dmax!(elong_sigma, self.m_sobel_sigma),
        );
        if self.m_outer_sigma > 0. {
            self.m_outer_madsave = vec![-1.; n_real_view];
            self.m_outer_background = vec![0.; n_real_view];
            memory_error(true, "arrays for outer MAD");
            scaled_gaussian_kernel(
                &mut self.cp.outer_kernel,
                &mut self.cp.kern_dim_outer,
                9,
                self.m_outer_sigma,
            );
        }
        //
        let iu = i as usize;
        let max_any_sum = self.m_max_any_sum as usize;
        self.m_xseek = vec![0.; iu];
        self.m_yseek = vec![0.; iu];
        self.m_wsum_crit = vec![0.; iu];
        self.m_xmat = vec![0.; iu * self.m_xmat_size as usize];
        self.m_if_found = vec![0; iu];
        self.m_wsum_min = vec![0.; iu];
        self.m_ip_nearest = vec![0; iu];
        self.m_ip_near_save = vec![0; iu];
        self.m_iobj_del = vec![0; iu];
        self.m_idrop = vec![0; iu];
        self.tc.iobj_ali = vec![0; iu];
        self.m_num_in_sobel_sum = vec![0; iu];
        self.m_xseek_next_pos = vec![0.; iu];
        self.m_yseek_next_pos = vec![0.; iu];
        self.m_wsum_save = vec![-1.; (self.m_max_all_real * self.mx.max_view) as usize];
        self.m_in_corr_sum = vec![false; iu * max_any_sum];
        self.m_in_sobel_sum = vec![false; iu * max_any_sum];
        self.m_sobel_xpeaks = vec![0.; self.m_max_peaks as usize];
        self.m_sobel_ypeaks = vec![0.; self.m_max_peaks as usize];
        self.m_sobel_peaks = vec![0.; self.m_max_peaks as usize];
        self.m_sobel_wsums = vec![0.; self.m_max_peaks as usize];
        self.m_sobel_edge_sd = vec![0.; self.m_max_peaks as usize];
        self.m_bkgd_wmax_save = vec![-1.; n_real_view];
        self.m_elong_save = vec![-1.; n_real_view];
        self.m_bkgd_neigh_wmax =
            vec![0.; (self.m_max_wneigh * (2 * self.m_max_drb_delta_z + 1)) as usize];

        memory_error(true, "arrays for points in area");
        ix = 0;
        while ix < max_in_area {
            self.m_xmat[(ix * self.m_xmat_size + 2) as usize] = 0.;
            ix += 1;
        }
        self.m_num_cgbetter = 0;
        self.m_num_sobel_cgeval = 0;
        self.m_sobel_res_sum = 0.;
        self.m_cg_res_sum = 0.;
        //
        // Get size and offset of sobel filtered
        if self.m_max_sobel_sum > 0 {
            self.m_scale_fac_sobel = self.m_diameter / target_sobel;
            // print *,scaleFacSobel, diameter, targetSobel, scaleByInterp
            ierr = scaled_sobel(
                Some(&self.m_boxes),
                self.m_nx_box,
                self.m_ny_box,
                self.m_scale_fac_sobel,
                self.m_scale_by_interp,
                self.m_interp_type,
                -1.,
                None,
                &mut self.m_nx_sobel,
                &mut self.m_ny_sobel,
                &mut x_off_sobel,
                &mut y_off_sobel,
            );
            self.m_nxs_pad = nice_frame(self.m_nx_sobel + 2 * npad, 2, 19);
            self.m_nys_pad = nice_frame(self.m_ny_sobel + 2 * npad, 2, 19);
            maxarr = (self.m_nxs_pad + 2) * self.m_nys_pad;
            i = self.m_nx_sobel * self.m_ny_sobel;
            ix = 0;
            let _ = ix;
            self.m_sarray = vec![0.; maxarr as usize];
            self.m_sbrray = vec![0.; maxarr as usize];
            self.m_ref_sobel = vec![0.; i as usize];
            self.m_box_sobel = vec![0.; i as usize];
            self.m_sobel_sum = vec![0.; (max_in_area * self.m_npix_box) as usize];
            self.m_iflag_cgvs_sobel = vec![0; self.fm.max_pt as usize];
            self.m_saved_cgcoord = vec![0.; (self.fm.max_pt * 2) as usize];
            memory_error(true, "arrays for Sobel filtering");

            if self.m_sobel_sigma > 0. {
                self.m_tmp_sobel = vec![0.; self.m_npix_box as usize];
                memory_error(true, "array for sobel filtering");
                scaled_gaussian_kernel(
                    &mut self.m_mat_kernel,
                    &mut self.m_kernel_dim,
                    7,
                    self.m_sobel_sigma,
                );
            }
        }

        //
        // set up sequencing for object lists and views - odd passes from
        // middle outward, even passes from ends inward
        //
        num_obj_tot = self.tc.num_obj_do;
        self.m_num_seqs = 0;
        self.m_last_seq = 0;
        self.m_save_all_points = false;
        self.m_ipass = 1;
        while self.m_ipass <= num_rounds {
            if (self.m_ipass % 2) == 1 {
                if self.m_ipass == 1 && (split_first_round || initial_bidir_views > 0) {
                    self.m_ix0 = self.tc.min_tilt_ind - 1;
                    self.m_iy0 = self.tc.min_tilt_ind;
                    if self.m_iv_bidir_part2 > 0 {
                        ix = self.m_iv_bidir_part2 - initial_bidir_views / 2;
                        iy = self.m_iv_bidir_part2
                            + (initial_bidir_views - initial_bidir_views / 2 - 1);
                        i = 1;
                        while i <= nobj_lists {
                            self.add_sequence(i, self.m_ix0, ix);
                            self.add_sequence(i, self.m_iy0, iy);
                            i += 1;
                        }
                        self.m_ix0 = ix - 1;
                        self.m_iy0 = iy + 1;
                    }
                    if split_first_round {
                        ix = (self.m_min_view_do + self.tc.min_tilt_ind) / 2;
                        iy = (self.m_max_view_do + self.tc.min_tilt_ind) / 2;
                        i = 1;
                        while i <= nobj_lists {
                            self.add_sequence(i, self.m_ix0, ix);
                            self.add_sequence(i, self.m_iy0, iy);
                            i += 1;
                        }
                        self.m_ix0 = ix - 1;
                        self.m_iy0 = iy + 1;
                    }
                    i = 1;
                    while i <= nobj_lists {
                        self.add_sequence(i, self.m_ix0, self.m_min_view_do);
                        self.add_sequence(i, self.m_iy0, self.m_max_view_do);
                        i += 1;
                    }
                } else {
                    i = 1;
                    while i <= nobj_lists {
                        self.add_sequence(i, self.tc.min_tilt_ind - 1, self.m_min_view_do);
                        self.add_sequence(i, self.tc.min_tilt_ind, self.m_max_view_do);
                        i += 1;
                    }
                }
            } else {
                i = 1;
                while i <= nobj_lists {
                    self.add_sequence(i, self.m_min_view_do, self.tc.min_tilt_ind - 1);
                    self.add_sequence(i, self.m_max_view_do, self.tc.min_tilt_ind);
                    i += 1;
                }
            }
            self.m_ipass += 1;
        }
        //
        // get list of inside and edge pixels for centroid
        //
        self.m_cg_edge_width =
            b3dmax!(1.5, cg_edge_diam_frac as f64 * self.m_diameter as f64) as f32;
        self.m_cg_gap_width = cg_gap_diam_frac * self.m_diameter;
        lim_inside = (3.5 * self.m_cg_radius as f64 * self.m_cg_radius as f64 + 22.) as i32;
        {
            let r1 = self.m_cg_radius + self.m_cg_gap_width + self.m_cg_edge_width;
            let r2 = self.m_cg_radius + self.m_cg_gap_width;
            lim_edge = (3.5 * (r1 * r1 - r2 * r2) as f64 + 22.) as i32;
        }
        limcg = (self.m_cg_radius + 4.) as i32;
        self.cp.idx_in = vec![0; lim_inside as usize];
        self.cp.idyin = vec![0; lim_inside as usize];
        self.cp.idx_edge = vec![0; lim_edge as usize];
        self.cp.idy_edge = vec![0; lim_edge as usize];
        self.cp.edge_pixels = vec![0.; lim_edge as usize];
        memory_error(true, "arrays for computing centroid");

        self.cp.num_outer = 1;
        lim_outer = ((self.m_nx_box * self.m_ny_box) as f64
            - 3.14159 * self.m_diameter as f64 * self.m_diameter as f64) as i32;
        if self.m_outer_sigma > 0. {
            self.cp.idx_outer = vec![0; lim_outer as usize];
            self.cp.idy_outer = vec![0; lim_outer as usize];
            self.cp.outer_pixels = vec![0.; lim_outer as usize];
            memory_error(true, "arrays for outside MAD");
            limcg = b3dmax!(self.m_nx_box, self.m_ny_box) / 2;
            self.cp.num_outer = 0;
        }

        self.cp.num_inside = 0;
        self.cp.num_edge = 0;
        iy = -limcg;
        while iy <= limcg {
            if b3dabs!(iy) > self.m_ny_box / 2 - 2 {
                iy += 1;
                continue;
            }
            ix = -limcg;
            while ix <= limcg {
                if b3dabs!(ix) > self.m_nx_box / 2 - 2 {
                    ix += 1;
                    continue;
                }
                rad_pix = ((ix as f64 - 0.5) * (ix as f64 - 0.5)
                    + (iy as f64 - 0.5) * (iy as f64 - 0.5))
                    .sqrt() as f32;
                if rad_pix <= self.m_cg_radius {
                    self.cp.idx_in[self.cp.num_inside as usize] = ix;
                    self.cp.idyin[self.cp.num_inside as usize] = iy;
                    self.cp.num_inside += 1;
                } else if rad_pix > self.m_cg_radius + self.m_cg_gap_width
                    && rad_pix <= self.m_cg_radius + self.m_cg_gap_width + self.m_cg_edge_width
                {
                    self.cp.idx_edge[self.cp.num_edge as usize] = ix;
                    self.cp.idy_edge[self.cp.num_edge as usize] = iy;
                    self.cp.num_edge += 1;
                }
                if self.m_outer_sigma > 0. && rad_pix >= self.m_diameter {
                    self.cp.idx_outer[self.cp.num_outer as usize] = ix;
                    self.cp.idy_outer[self.cp.num_outer as usize] = iy;
                    self.cp.num_outer += 1;
                }
                if self.cp.num_inside >= lim_inside
                    || self.cp.num_edge >= lim_edge
                    || self.cp.num_outer >= lim_outer
                {
                    exit_error(b"Programmer error computing size of centroid arrays");
                }
                ix += 1;
            }
            iy += 1;
        }
        //
        self.m_nz_out = 0;
        if let Some(out) = out_file.as_ref() {
            let out_s = String::from_utf8_lossy(out).into_owned();
            // `tmpStr = B3DMALLOC(char, strlen(outFile + 10))` is ten bytes
            // short of what the `sprintf`s below write (see `BUGS.md`); the
            // names are `String`s here.
            let name = format!("{}{}", out_s, ".box");
            unsafe { iiu_open(2, &name, "NEW") };
            let name = format!("{}{}", out_s, ".ref");
            unsafe { iiu_open(3, &name, "NEW") };
            let name = format!("{}{}", out_s, ".cor");
            unsafe { iiu_open(4, &name, "NEW") };
            i = 2;
            while i <= 4 {
                iiu_trans_header(i, 1);
                iiu_alt_mode(i, self.m_mode_box);
                setsiz_sam_cel(
                    i,
                    if i == 4 { self.m_nx_pad } else { self.m_nx_box },
                    if i == 4 { self.m_nx_pad } else { self.m_ny_box },
                    1,
                );
                i += 1;
            }
            //
            mrc_fill_label_string(b"Boxes", &mut titlech);
            self.m_box_sum = 0.;
            self.m_box_min = 1.0e20;
            self.m_box_max = -1.0e20;
            self.m_ref_sum = 0.;
            self.m_ref_min = 1.0e20;
            self.m_ref_max = -1.0e20;
            self.m_corro_sum = 0.;
            self.m_corr_min = 1.0e20;
            self.m_corr_max = -1.0e20;
            let name = format!("{}{}", out_s, ".brpl");
            self.m_brpl_fp = Some(b3d_open_file(&name, "w"));
            let name = format!("{}{}", out_s, ".cpl");
            self.m_cpl_fp = Some(b3d_open_file(&name, "w"));
        }
        //
        // Needed for formats
        self.m_need4digits = num_obj_tot >= 1000;
        //
        // Start looping on the sequences of views
        //
        self.m_max_obj_orig = self.fm.max_mod_obj;
        did_save_all_init = false;
        self.m_isequence = 1;
        while self.m_isequence <= self.m_num_seqs {
            let isq = (self.m_isequence - 1) as usize;
            self.m_num_added = 1;
            self.tc.nview_local = 0;
            self.m_if_align_done = 0;
            iseq_pass = ((self.m_isequence + 1) / 2 - 1) / nobj_lists + 1;
            in_split_round = split_first_round && iseq_pass == 1;
            if split_first_round {
                iseq_pass = b3dmax!(1, iseq_pass - 1);
            }
            if iseq_pass >= local_view_pass {
                self.tc.nview_local = nv_local_in;
            }
            if self.m_save_all_points && i_area_save < 0 && self.m_list_seq[isq] != self.m_last_seq
            {
                break;
            }
            self.m_save_all_points =
                b3dabs!(i_area_save) == self.m_list_seq[isq] && ipass_save == iseq_pass;

            // Initial mean residuals for new set of points or beginning of a full round when
            // there is just one set; doing more than 2 rounds will then work the same as
            // restarting from fid as seed
            if self.m_list_seq[isq] != self.m_last_seq
                || ((nobj_lists == 1 && iseq_pass > 1 && (iseq_pass % 2) == 1)
                    && (self.m_isequence % 2) == 1)
            {
                self.m_iview_seq = 1;
                // `INIT_ARRAY(mResMean, limResid * mMaxAllReal, -1.)`; the whole
                // array, which can have grown past `limResid` rows (below)
                self.m_res_mean.fill(-1.);
            }

            if self.m_list_seq[isq] != self.m_last_seq {
                //
                // initialize if doing a new set of points
                //
                self.tc.init_xyz_done = 0;
                self.m_last_seq = self.m_list_seq[isq];
                self.tc.num_obj_do = nin_obj_list[(self.m_last_seq - 1) as usize];
                i = 0;
                while i < self.tc.num_obj_do {
                    self.tc.iobj_seq[i as usize] =
                        iobj_lists[(i + ind_obj_list[(self.m_last_seq - 1) as usize] - 1) as usize];
                    i += 1;
                }
                //
                // Initialize arrays for boxes and residuals
                let nod = self.tc.num_obj_do as usize;
                self.m_in_core[..max_any_sum * nod].fill(-1);
                self.m_num_in_sobel_sum[..nod].fill(0);
                self.m_in_corr_sum[..max_any_sum * nod].fill(false);
                self.m_in_sobel_sum[..max_any_sum * nod].fill(false);
                self.m_corr_sum[..self.m_npix_box as usize * nod].fill(0.);
                if self.m_max_sobel_sum > 0 {
                    self.m_sobel_sum[..self.m_npix_box as usize * nod].fill(0.);
                }
                printf!(
                    "Starting %s%4d, round%3d,%4d contours\n",
                    CArg::Str(area_obj_str),
                    CArg::Int(self.m_list_seq[isq] as i64),
                    CArg::Int(iseq_pass as i64),
                    CArg::Int(self.tc.num_obj_do as i64)
                );
                if nobj_lists > 1 {
                    i = 1;
                    while i <= self.tc.num_obj_do {
                        // Fixed in translation (2026-09-26, `BUGS.md`): the source's
                        // 4-digit format `"%4d"` drops the separator argument
                        // (`beadtrack.cpp:1588`), so the list runs together.
                        printf!(
                            if self.m_need4digits { "%4d%s" } else { "%3d%s" },
                            CArg::Int(self.tc.iobj_seq[(i - 1) as usize] as i64),
                            CArg::Str(
                                if i == self.tc.num_obj_do
                                    || i % (if self.m_need4digits { 16 } else { 20 }) == 0
                                {
                                    "\n"
                                } else {
                                    " "
                                }
                            )
                        );
                        i += 1;
                    }
                }
            }
            if self.m_save_all_points && !did_save_all_init {
                j = 1;
                while j <= 10 {
                    iobj = getimodobjsize() + j;
                    putimodflag(iobj, 1);
                    putsymtype(iobj, 0);
                    putsymsize(iobj, 5);
                    putimodobjname(iobj, obj_names[((j - 1) % 5) as usize]);
                    ix = (255. * colors[(j * 3 - 3) as usize] as f64) as i32;
                    iy = (255. * colors[(j * 3 - 2) as usize] as f64) as i32;
                    iz = (255. * colors[(j * 3 - 1) as usize] as f64) as i32;
                    putobjcolor(iobj, ix, iy, iz);
                    //
                    // Set up and add a point in every single object to keep the numbering the same
                    i = 1;
                    while i <= self.m_max_obj_orig {
                        iobj = i + j * self.m_max_obj_orig;
                        let ob = (iobj - 1) as usize;
                        self.fm.ibase_obj[ob] = self.fm.ibase_free;
                        self.fm.npt_in_obj[ob] = 0;
                        self.fm.obj_color[ob][0] = 1;
                        self.fm.obj_color[ob][1] = 256 - getimodobjsize() - j;
                        self.fm.max_mod_obj = b3dmax!(self.fm.max_mod_obj, iobj);
                        self.fm.ndx_order[ob] = iobj;
                        self.fm.obj_order[ob] = iobj;
                        if self.fm.npt_in_obj[(i - 1) as usize] > 0 {
                            ip = self.fm.object[self.fm.ibase_obj[(i - 1) as usize] as usize];
                            ix = 0;
                            let pt = self.fm.p_coord[(ip - 1) as usize];
                            add_point(&mut self.fm, iobj, &mut ix, pt[0], pt[1], b3dnint!(pt[2]));
                        }
                        i += 1;
                    }
                    j += 1;
                }
                did_save_all_init = true;
            }
            //
            // Set direction and make list of views to do
            //
            self.m_track_dir = if self.m_iv_seq_end[isq] - self.m_iv_seq_str[isq] < 0 {
                -1
            } else {
                1
            };
            nv_list = 0;
            self.m_iview = self.m_iv_seq_str[isq];
            while self.m_track_dir * (self.m_iview - self.m_iv_seq_end[isq]) <= 0 {
                if_exclude = 0;
                let mut iexcl = 1;
                while iexcl <= self.tc.num_exclude {
                    if self.m_iview == self.tc.iz_exclude[(iexcl - 1) as usize] {
                        if_exclude = 1;
                    }
                    iexcl += 1;
                }
                if if_exclude == 0 {
                    nv_list += 1;
                    iv_list[(nv_list - 1) as usize] = self.m_iview;
                }
                self.m_iview += self.m_track_dir;
            }
            //
            // loop on the views
            //
            self.m_iv_list = 1;
            while self.m_iv_list <= nv_list {
                self.m_iview = iv_list[(self.m_iv_list - 1) as usize];
                self.m_iz_next = self.m_iview - 1;
                self.count_and_prepare_points_to_do();
                self.m_ipass = 1;
                while self.m_ipass <= 2 && self.m_num_to_do != 0 {
                    //
                    // now try to do tiltalign if possible
                    if self.m_num_added != 0 {
                        let off = ((self.m_iview_seq - 1) * self.m_max_all_real) as usize;
                        tilt_ali(
                            &mut self.tc,
                            &mut self.av,
                            &self.mx,
                            &mut self.m_eval_funct,
                            &mut self.sg,
                            &self.fm,
                            &mut self.m_if_did_align,
                            &mut self.m_if_align_done,
                            &mut self.m_res_mean[off..],
                            self.m_iview,
                            &mut self.m_ali_mean_res,
                        );
                        if self.m_if_did_align != 0 {
                            self.m_ivs_on_align = self.m_iview_seq;
                        }
                    }
                    self.get_projected_positions_setup_fits();
                    //
                    // Loop through points, refining projections before search
                    if self.m_ipass == 1 {
                        self.m_num_pioneer = self.m_num_data;
                    }
                    self.m_num_added = 0;
                    self.get_wsum_criteria();
                    self.find_all_beads_on_view(out_file.is_some());
                    //
                    // get a final fit and a new tiltalign, then find a maximum
                    // error for each pt
                    if self.m_ipass == 2 {
                        printf!(
                            "%4d pts added on pass 2, conts:%s",
                            CArg::Int(self.m_num_added as i64),
                            CArg::Str(if self.m_num_added != 0 { "" } else { "\n" })
                        );
                        i = 1;
                        while i <= self.m_num_added {
                            // Fixed in translation (`BUGS.md`): the source's `"%5d"`
                            // drops the line-break argument (`beadtrack.cpp:1672`).
                            printf!(
                                if self.m_need4digits { "%5d%s" } else { "%4d%s" },
                                CArg::Int(self.m_iobj_del[(i - 1) as usize] as i64),
                                CArg::Str(
                                    if i == self.m_num_added
                                        || i % (if self.m_need4digits { 9 } else { 11 }) == 0
                                    {
                                        "\n"
                                    } else {
                                        ""
                                    }
                                )
                            );
                            i += 1;
                        }
                    }
                    self.redo_fits_evaluate_residuals(&model_file);
                    self.m_ipass += 1;
                    fflush_stdout!();
                }
                self.m_iview_seq += 1;
                // Fixed in translation (BUGS.md, beadtrack): `mResMean` holds
                // `limResid` rows of `mMaxAllReal`, but the view sequence number
                // can exceed `limResid` (e.g. when the view range at the ends is
                // cut by SkipViews), and native then reads and writes past the
                // allocation.  Defined: the array grows by rows of -1, the value
                // every row starts at.
                let needed = (self.m_iview_seq * self.m_max_all_real) as usize;
                if self.m_res_mean.len() < needed {
                    self.m_res_mean.resize(needed, -1.);
                }
                fflush_stdout!();
                self.m_iv_list += 1;
            }
            //
            // Report total missing at end of pass
            //
            if (self.m_isequence % 2) == 0 && !in_split_round {
                miss_tot = 0;
                i = 1;
                while i <= self.tc.num_obj_do {
                    count_missing(
                        &self.fm,
                        self.tc.iobj_seq[(i - 1) as usize],
                        self.tc.nview_all,
                        &self.tc.iz_exclude,
                        self.tc.num_exclude,
                        &mut missing,
                        &mut listz,
                        &mut num_list_z,
                    );
                    miss_tot += num_list_z;
                    i += 1;
                }
                printf!(
                    "For %s%3d, round%3d:%3d contours, points missing =%5d\n",
                    CArg::Str(area_obj_str),
                    CArg::Int(self.m_list_seq[isq] as i64),
                    CArg::Int(iseq_pass as i64),
                    CArg::Int(self.tc.num_obj_do as i64),
                    CArg::Int(miss_tot as i64)
                );
                //
                // At end of pass also evaluate whether CG positions are better than Sobel, but only
                // if no local Z used and not doing auto seed output
                if self.m_max_sobel_sum > 0
                    && nv_local_in == 0
                    && elong_file.is_none()
                    && xyz_file.is_none()
                {
                    self.evaluate_cg_vs_sobel_resids(self.m_iview_seq - 1);
                }
            }
            //
            // Save residuals and XYZ values on every round if tiltalign run
            if (elong_file.is_some() || xyz_file.is_some()) && self.m_if_align_done != 0 {
                //
                // Find each object in the real object list and move residual/XYZ data into list
                i = 0;
                while i < self.tc.num_obj_do {
                    j = 0;
                    while j < self.av.nreal_pt {
                        if self.tc.iobj_ali[j as usize] == self.tc.iobj_seq[i as usize] {
                            let ind =
                                (i + ind_obj_list[(self.m_last_seq - 1) as usize] - 1) as usize;
                            resid_lists[ind] =
                                self.m_res_mean[((self.m_ivs_on_align - 1) * self.m_max_all_real
                                    + self.tc.iobj_ali[j as usize]
                                    - 1) as usize];
                            ix = 0;
                            while ix < 3 {
                                xyz_all_area[ind * 3 + ix as usize] =
                                    self.av.xyz[(j * 3 + ix) as usize];
                                ix += 1;
                            }
                            //printf("%4d%4d%12.3f%12.3f%12.3f\n", j + 1, i + indObjList[mLastSeq - 1],
                            //       av->xyz[j * 3], av->xyz[j * 3 + 1], av->xyz[j * 3 + 2]);
                            break;
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }
            self.m_isequence += 1;
        }
        //
        // output lists of missing points, except for excluded sections
        //
        printf!("\nObj cont:  Views on which points are missing\n");
        miss_tot = 0;
        iobj = 1;
        while iobj <= self.m_max_obj_orig {
            count_missing(
                &self.fm,
                iobj,
                self.tc.nview_all,
                &self.tc.iz_exclude,
                self.tc.num_exclude,
                &mut missing,
                &mut listz,
                &mut num_list_z,
            );
            if num_list_z != 0 {
                fort_mod_obj_to_cont(iobj, &self.fm.obj_color, &mut imod_obj, &mut imod_cont);
                printf!(
                    "%2d%4d: ",
                    CArg::Int(imod_obj as i64),
                    CArg::Int(imod_cont as i64)
                );
                write_list(&listz, num_list_z, 80);
            }
            miss_tot += num_list_z;
            iobj += 1;
        }
        printf!(" Total points missing = %d\n", CArg::Int(miss_tot as i64));
        //
        // If CG points gave more alignments with better residual, switch to them
        if self.m_num_sobel_cgeval > 0 {
            // Fixed in translation (2026-09-26, `BUGS.md`): `formattedError`
            // returns its one static buffer, so both `%s` arguments of the
            // source's single `printf` print whichever call ran last — the
            // Sobel value (`beadtrack.cpp:1742-1746`).  Each is printed here.
            let cg_str = formatted_error(
                pixel_size * image_binned as f32 * self.m_cg_res_sum
                    / self.m_num_sobel_cgeval as f32,
                0.3,
                0,
            );
            let sobel_str = formatted_error(
                pixel_size * image_binned as f32 * self.m_sobel_res_sum
                    / self.m_num_sobel_cgeval as f32,
                0.3,
                0,
            );
            printf!(
                "\nMean residual %s nm from Sobel centering, %s nm from centroid centering\n  centroid centering better in %3d of %3d fits\n",
                CArg::Str(&sobel_str),
                CArg::Str(&cg_str),
                CArg::Int(self.m_num_cgbetter as i64),
                CArg::Int(self.m_num_sobel_cgeval as i64)
            );
        }
        if self.m_num_cgbetter > self.m_num_sobel_cgeval / 2 {
            printf!(
                "WARNING: BEADTRACK - Using positions from centroid centering\nbecause they gave a lower mean residual than Sobel centering.\nA different kernel sigma might be needed for Sobel centering.\n\n"
            );
            self.swap_sobel_and_cg_coords();
        }
        //
        // Write out residual file and XYZ file
        if elong_file.is_some() || xyz_file.is_some() {
            if let Some(xyz) = xyz_file.as_ref() {
                self.m_boxes = Vec::new();
                self.m_corr_sum = Vec::new();
                adjust_xyz_in_areas(
                    &iobj_lists,
                    max_olist,
                    &mut ind_obj_list,
                    &mut nin_obj_list,
                    &mut xyz_all_area,
                    nobj_lists,
                    &mut self.tc.xyz_save,
                    self.m_max_obj_orig,
                );
                /*for (iobj = 0; iobj < mMaxObjOrig; iobj++) {
                if (fmodNpt_in_obj[iobj] == 0)
                  continue;
                printf("%6d%12.3f%12.3f%12.3f\n",
                       iobj + 1, tc->xyzSave[iobj * 3], tc->xyzSave[iobj * 3 + 1],
                       tc->xyzSave[iobj * 3 + 2]);
                       }*/
                xyz_fp = Some(b3d_open_file(&String::from_utf8_lossy(xyz), "w"));
            }

            if let Some(elong) = elong_file.as_ref() {
                elong_fp = Some(b3d_open_file(&String::from_utf8_lossy(elong), "w"));
            }

            // Get average of the edge SD values
            //
            // `mEdgeSdSave` is allocated only with an elongation file
            // (`beadtrack.cpp:1305`); with an XYZ file alone the source reads
            // an uninitialised pointer here and segfaults.  Fixed in translation
            // (2026-09-26, `BUGS.md`): it reads as an array of -1, i.e. "no edge
            // SD saved".
            let max_view_do = self.m_max_view_do;
            let edge_sd = |s: &Vec<f32>, ind: i32| -> f32 {
                if s.is_empty() { -1. } else { s[ind as usize] }
            };
            sdsum = 0.;
            iy = 0;
            iobj = 1;
            while iobj <= self.m_max_obj_orig {
                self.m_res_mean[(iobj - 1) as usize] = 0.;
                self.m_iview = self.m_min_view_do;
                while self.m_iview <= self.m_max_view_do {
                    let ind = (iobj - 1) * max_view_do + self.m_iview - 1;
                    if edge_sd(&self.m_edge_sd_save, ind) > 0. {
                        sdsum += edge_sd(&self.m_edge_sd_save, ind);
                        iy += 1;
                    }
                    self.m_iview += 1;
                }
                iobj += 1;
            }
            if iy > 0 {
                sdavg = sdsum / iy as f32;
            }
            //
            // Add up mean residuals for all points that have any
            i = 1;
            while i <= ind_free {
                if resid_lists[(i - 1) as usize] >= 0. {
                    iobj = iobj_lists[(i - 1) as usize];
                    num_res_saved[(iobj - 1) as usize] += 1;
                    self.m_res_mean[(iobj - 1) as usize] += resid_lists[(i - 1) as usize];
                }
                i += 1;
            }
            //
            // For each real point, get the top/bottom number, mean residual, and mean, sd, min
            // and max of the edge sd
            // If there are no residuals because of too few points, set them all to 0.1
            xpos = 0.1;
            any_resids = false;
            iobj = 0;
            while iobj < self.m_max_obj_orig {
                if self.fm.npt_in_obj[iobj as usize] > 0 && num_res_saved[iobj as usize] > 0 {
                    any_resids = true;
                }
                iobj += 1;
            }
            iobj = 0;
            while iobj < self.m_max_obj_orig {
                let ob = iobj as usize;
                if self.fm.npt_in_obj[ob] == 0 {
                    iobj += 1;
                    continue;
                }
                fort_mod_obj_to_cont(iobj + 1, &self.fm.obj_color, &mut imod_obj, &mut imod_cont);
                if any_resids {
                    xpos = -1.;
                    if num_res_saved[ob] > 0 {
                        xpos = self.m_res_mean[ob] / num_res_saved[ob] as f32;
                    }
                }
                sdsum = 0.;
                sdsumsq = 0.;
                sdmin = 1.0e20;
                sdmax = -1.;
                iy = 0;
                ix = 0;
                izv = 0;
                omad_mean = 0.;
                oback_mean = 0.;
                wsum = 0.;
                itry = 0;
                self.m_iview = self.m_min_view_do;
                while self.m_iview <= self.m_max_view_do {
                    let ind = iobj * max_view_do + self.m_iview - 1;
                    xtmp = edge_sd(&self.m_edge_sd_save, ind) / sdavg;
                    if xtmp > 0. {
                        sdsum += xtmp;
                        sdsumsq += xtmp * xtmp;
                        sdmin = b3dmin!(sdmin, xtmp);
                        sdmax = b3dmax!(sdmax, xtmp);
                        iy += 1;
                        self.m_prev_res[(iy - 1) as usize] = xtmp;
                    }
                    if self.m_elong_save[ind as usize] > 0. {
                        ix += 1;
                        self.tc.tilt_all[(ix - 1) as usize] = self.m_elong_save[ind as usize];
                    }
                    if self.m_outer_sigma > 0. {
                        if self.m_outer_madsave[ind as usize] > 0. {
                            izv += 1;
                            omad_mean += self.m_outer_madsave[ind as usize];
                            oback_mean += self.m_outer_background[ind as usize];
                        }
                    }
                    let wind = (iobj * self.mx.max_view + self.m_iview - 1) as usize;
                    if self.m_wsum_save[wind] > 0. {
                        itry += 1;
                        wsum += self.m_wsum_save[wind];
                    }
                    self.m_iview += 1;
                }
                sdmean = -1.;
                edge_sdsd = 0.;
                sdmed = -1.;
                if iy < 1 {
                    sdmin = -1.;
                } else if iy > 1 {
                    sums_to_avg_sd(sdsum, sdsumsq, iy, &mut sdmean, &mut edge_sdsd);
                    rs_fast_median_in_place(&mut self.m_prev_res, iy, &mut sdmed);
                }
                let _ = (sdmin, sdmax);
                elong_mean = 0.;
                elong_med = 0.;
                elong_sd = 0.;
                if ix > 1 {
                    avg_sd(
                        &self.tc.tilt_all,
                        ix,
                        &mut elong_mean,
                        &mut elong_med,
                        &mut elong_sd,
                    );
                    rs_fast_median_in_place(&mut self.tc.tilt_all, ix, &mut elong_med);
                }
                if izv > 0 {
                    omad_mean /= izv as f32;
                    oback_mean /= izv as f32;
                }
                let _ = (omad_mean, oback_mean);
                if itry > 0 {
                    wsum /= itry as f32;
                }
                if let Some(fp) = elong_fp.as_mut() {
                    fprintf!(
                        fp,
                        "%3d%7d%12.4f%11.4f%11.4f%11.4f%11.4f%11.4f%11.4f%11.2f\n",
                        CArg::Int(imod_obj as i64),
                        CArg::Int(imod_cont as i64),
                        CArg::Dbl(xpos as f64),
                        CArg::Dbl(sdmean as f64),
                        CArg::Dbl(sdmed as f64),
                        CArg::Dbl(edge_sdsd as f64),
                        CArg::Dbl(elong_mean as f64),
                        CArg::Dbl(elong_med as f64),
                        CArg::Dbl(elong_sd as f64),
                        CArg::Dbl(wsum as f64)
                    );
                }

                if let Some(fp) = xyz_fp.as_mut() {
                    if xpos >= 0.
                        && (self.tc.xyz_save[ob * 3] != 0.
                            || self.tc.xyz_save[ob * 3 + 1] != 0.
                            || self.tc.xyz_save[ob * 3 + 2] != 0.)
                    {
                        fprintf!(
                            fp,
                            "%6d%12.3f%12.3f%12.3f%12.3f\n",
                            CArg::Int((iobj + 1) as i64),
                            CArg::Dbl(self.tc.xyz_save[ob * 3] as f64),
                            CArg::Dbl(self.tc.xyz_save[ob * 3 + 1] as f64),
                            CArg::Dbl(self.tc.xyz_save[ob * 3 + 2] as f64),
                            CArg::Dbl(xpos as f64)
                        );
                    }
                }
                iobj += 1;
            }
            // `fclose(elongFP)`, `fclose(xyzFP)`.
            if let Some(mut fp) = elong_fp.take() {
                let _ = fp.flush();
            }
            if let Some(mut fp) = xyz_fp.take() {
                let _ = fp.flush();
            }
        }

        //
        // convert index coordinates back to model coordinates
        //
        xtmp = b3dmin!(
            50.,
            0.75 * b3dmax!(self.m_nx_im, self.m_ny_im) as f64 / nz as f64
        ) as f32;
        putimodzscale(xtmp);
        let _ = scale_fort_mod_to_image(&mut self.fm, 1, 1);
        let _ = write_fort_model(&model_file, &mut self.fm);
        if out_file.is_some() {
            setsiz_sam_cel(2, self.m_nx_box, self.m_ny_box, self.m_nz_out);
            setsiz_sam_cel(3, self.m_nx_box, self.m_ny_box, self.m_nz_out);
            setsiz_sam_cel(4, self.m_nx_pad, self.m_ny_pad, self.m_nz_out);
            let title_len = titlech
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(MRC_LABEL_SIZE);
            let title = String::from_utf8_lossy(&titlech[..title_len]).into_owned();
            iiu_write_header_str(
                2,
                &title,
                1,
                self.m_box_min,
                self.m_box_max,
                self.m_box_sum / self.m_nz_out as f32,
            );
            iiu_write_header_str(
                3,
                &title,
                1,
                self.m_ref_min,
                self.m_ref_max,
                self.m_ref_sum / self.m_nz_out as f32,
            );
            iiu_write_header_str(
                4,
                &title,
                1,
                self.m_corr_min,
                self.m_corr_max,
                self.m_corro_sum / self.m_nz_out as f32,
            );
            unsafe {
                iiu_close(2);
                iiu_close(3);
                iiu_close(4);
            }
            if let Some(mut fp) = self.m_brpl_fp.take() {
                let _ = fp.flush();
            }
            if let Some(mut fp) = self.m_cpl_fp.take() {
                let _ = fp.flush();
            }
        }
        let _ = (
            ierr, mode, dmin2, dmax2, dmean2, min_xpiece, min_ypiece, lim_gaps, cvbxcen, cvbycen,
            xst, xnd, yst, ynd, tmp_str,
        );
        crate::imod::libcfshr::b3dutil::exit(0);
    }
}

impl BeadTrack {
    /// Original: `BeadTrack::countAndPreparePointsToDo` (`beadtrack.cpp:1913`).
    ///
    /// countAndPreparePointsToDo evaluates each potential point on a new view to see
    /// if it already done, a gap to be skipped, or has no near enough neighbors
    /// Then it loads necessary boxes, forms averages and evaluates wsum if needed
    pub fn count_and_prepare_points_to_do(&mut self) {
        let mut num_zero_w: i32;
        let mut num_wtot: i32;
        let get_edge_sdsave: i32;
        let mut snr_sum: f32;
        let mut snr_sumb: f32;
        let mut best_sum: f32 = 0.;
        // `ib` is uninitialised in the source; every path that reads it has
        // just assigned it (a free box always exists after the unused ones are
        // released).
        let mut ib: i32 = 0;
        let mut ibase: i32;
        let mut ibox: i32;
        let mut ic: i32;
        let mut igap: i32;
        let mut iobj: i32;
        let mut ip: i32;
        let mut ipt: i32;
        let mut ix: i32 = 0;
        let mut iy: i32 = 0;
        let mut iz_box: i32;
        let mut izv: i32;
        let mut jc: i32;
        let mut max_box_use: i32;
        let mut n_close: i32;
        let mut n_farther: i32;
        let mut near_diff: i32;
        let mut num_in_obj: i32;
        let mut wsum: f32 = 0.;
        let mut xbox: f32;
        let mut xt: f32;
        let mut xtmp: f32;
        let mut ybox: f32;
        let mut yt: f32;
        let mut ytmp: f32;

        let max_any_sum = self.m_max_any_sum;
        let npix = self.m_npix_box as usize;
        //
        // first see how many are done on this view
        //
        self.m_num_to_do = 0;
        num_wtot = 0;
        num_zero_w = 0;
        snr_sum = 0.;
        snr_sumb = 0.;
        get_edge_sdsave = self.cp.get_edge_sd;
        if self.m_isequence == 1 && self.m_iv_list == 1 {
            self.cp.get_edge_sd = 1;
        }
        self.m_iobj_do = 0;
        while self.m_iobj_do < self.tc.num_obj_do {
            let iod = self.m_iobj_do as usize;
            iobj = self.tc.iobj_seq[iod];
            num_in_obj = self.fm.npt_in_obj[(iobj - 1) as usize];
            n_close = 0;
            n_farther = 0;
            self.m_if_found[iod] = 0;
            ibase = self.fm.ibase_obj[(iobj - 1) as usize];
            //
            // find closest ones within gap distance, find out if already
            // done, mark as a 2 to protect them
            // ipNearest holds the point number of the nearest for each object
            // ipClose is the list of points within gap distance for this object
            //
            near_diff = 10000;
            ip = 1;
            while ip <= num_in_obj {
                ipt = self.fm.object[(ibase + ip - 1) as usize];
                izv = b3dnint!(self.fm.p_coord[(ipt - 1) as usize][2]) + 1;
                if b3dabs!(izv - self.m_iview) < near_diff {
                    near_diff = b3dabs!(izv - self.m_iview);
                    self.m_ip_nearest[iod] = ip;
                }
                if izv == self.m_iview {
                    self.m_if_found[iod] = 2;
                } else if b3dabs!(izv - self.m_iview)
                    <= b3dmax!(self.m_max_gap, 2 * self.m_max_sobel_sum)
                {
                    xbox = self.fm.p_coord[(ipt - 1) as usize][0];
                    ybox = self.fm.p_coord[(ipt - 1) as usize][1];
                    iz_box = b3dnint!(self.fm.p_coord[(ipt - 1) as usize][2]);
                    find_piece(
                        &self.m_ix_pclist,
                        &self.m_iy_pclist,
                        &self.m_iz_pclist,
                        self.m_npclist,
                        self.m_nx_im,
                        self.m_ny_im,
                        self.m_nx_box,
                        self.m_ny_box,
                        xbox,
                        ybox,
                        iz_box,
                        &mut self.m_ix0,
                        &mut self.m_ix1,
                        &mut self.m_iy0,
                        &mut self.m_iy1,
                        &mut self.m_ipiece_z,
                        self.m_if_read_xfs,
                        &self.m_prexf,
                        &mut self.m_need_taper,
                        &mut self.m_need_fill,
                    );
                    if self.m_ipiece_z >= 0 {
                        if b3dabs!(izv - self.m_iview) <= self.m_max_gap {
                            n_close += 1;
                        }
                        n_farther += 1;
                        self.m_ip_close[(n_farther - 1) as usize] = ip;
                        self.m_iz_close[(n_farther - 1) as usize] = izv;
                    }
                }
                ip += 1;
            }
            //
            // if not found and none close, mark as -1 not to do
            // If there are close ones, make sure this is not a gap to preserve
            //
            if n_close == 0 && self.m_if_found[iod] == 0 {
                self.m_if_found[iod] = -1;
            }
            if n_close > 0 && self.m_if_found[iod] == 0 {
                if self.m_if_fill_in == 0
                    && self.m_ind_gap[iobj as usize] > self.m_ind_gap[(iobj - 1) as usize]
                {
                    //
                    // if not filling in, look for gap on list
                    //
                    igap = self.m_ind_gap[(iobj - 1) as usize];
                    while igap <= self.m_ind_gap[iobj as usize] - 1 {
                        if self.m_iv_gap[(igap - 1) as usize] == self.m_iview {
                            self.m_if_found[iod] = -1;
                        }
                        igap += 1;
                    }
                }
                if self.m_if_found[iod] == 0 {
                    //
                    // order feasible points by distance
                    //
                    self.m_num_to_do += 1;
                    ic = 0;
                    while ic < n_farther - 1 {
                        jc = ic + 1;
                        while jc < n_farther {
                            if b3dabs!(self.m_iz_close[jc as usize] - self.m_iview)
                                < b3dabs!(self.m_iz_close[ic as usize] - self.m_iview)
                            {
                                self.m_iz_close.swap(ic as usize, jc as usize);
                                self.m_ip_close.swap(ic as usize, jc as usize);
                            }
                            jc += 1;
                        }
                        ic += 1;
                    }
                    max_box_use = b3dmin!(n_farther, max_any_sum);
                    //
                    // see if boxes in memory match, free them if not
                    //
                    ibox = 0;
                    while ibox < max_any_sum {
                        let slot = (self.m_iobj_do * max_any_sum + ibox) as usize;
                        if self.m_in_core[slot] >= 0 {
                            //
                            // First subtract from correlation sum
                            if self.m_in_corr_sum[slot] {
                                if number_in_list(
                                    self.m_in_core[slot],
                                    Some(&self.m_iz_close),
                                    self.m_max_sum,
                                    0,
                                ) == 0
                                {
                                    ix = 0;
                                    while ix < self.m_npix_box {
                                        self.m_corr_sum[iod * npix + ix as usize] -=
                                            self.m_boxes[slot * npix + ix as usize];
                                        ix += 1;
                                    }
                                    self.m_in_corr_sum[slot] = false;
                                }
                            }
                            //
                            // Then subtract from sobel sum
                            if self.m_in_sobel_sum[slot] {
                                if number_in_list(
                                    self.m_in_core[slot],
                                    Some(&self.m_iz_close),
                                    self.m_max_sobel_sum,
                                    0,
                                ) == 0
                                {
                                    ix = 0;
                                    while ix < self.m_npix_box {
                                        self.m_sobel_sum[iod * npix + ix as usize] -=
                                            self.m_boxes[slot * npix + ix as usize];
                                        ix += 1;
                                    }
                                    self.m_in_sobel_sum[slot] = false;
                                    self.m_num_in_sobel_sum[iod] -= 1;
                                }
                            }
                            if number_in_list(
                                self.m_in_core[slot],
                                Some(&self.m_iz_close),
                                max_box_use,
                                0,
                            ) == 0
                            {
                                self.m_in_core[slot] = -1;
                            }
                        }
                        ibox += 1;
                    }
                    //
                    // now load ones that are needed into last free box and form sums
                    //
                    ic = 0;
                    while ic < max_box_use {
                        let core_off = (self.m_iobj_do * max_any_sum) as usize;
                        if number_in_list(
                            self.m_iz_close[ic as usize],
                            Some(&self.m_in_core[core_off..]),
                            max_any_sum,
                            0,
                        ) == 0
                        {
                            //
                            // If not loaded, look up last free box
                            ibox = 0;
                            while ibox < max_any_sum {
                                if self.m_in_core[core_off + ibox as usize] < 0 {
                                    ib = ibox;
                                }
                                ibox += 1;
                            }
                            self.m_in_core[core_off + ib as usize] = self.m_iz_close[ic as usize];
                            ipt = b3dabs!(
                                self.fm.object[(ibase + self.m_ip_close[ic as usize] - 1) as usize]
                            );
                            xbox = self.fm.p_coord[(ipt - 1) as usize][0];
                            ybox = self.fm.p_coord[(ipt - 1) as usize][1];
                            iz_box = b3dnint!(self.fm.p_coord[(ipt - 1) as usize][2]);
                            find_piece(
                                &self.m_ix_pclist,
                                &self.m_iy_pclist,
                                &self.m_iz_pclist,
                                self.m_npclist,
                                self.m_nx_im,
                                self.m_ny_im,
                                self.m_nx_box,
                                self.m_ny_box,
                                xbox,
                                ybox,
                                iz_box,
                                &mut self.m_ix0,
                                &mut self.m_ix1,
                                &mut self.m_iy0,
                                &mut self.m_iy1,
                                &mut self.m_ipiece_z,
                                self.m_if_read_xfs,
                                &self.m_prexf,
                                &mut self.m_need_taper,
                                &mut self.m_need_fill,
                            );
                            {
                                let mut load_tmp = std::mem::take(&mut self.m_box_tmp);
                                let ctf = std::mem::take(&mut self.m_ctf);
                                self.load_box_and_taper(
                                    &mut load_tmp,
                                    self.m_nx_box,
                                    self.m_ny_box,
                                    self.m_nx_pad,
                                    self.m_ny_pad,
                                    &ctf,
                                    self.m_delta_ctf,
                                );
                                self.m_ctf = ctf;
                                self.m_box_tmp = load_tmp;
                            }
                            //
                            // do subpixel shift and calculate CG and wsum value
                            // if it is not already saved
                            //
                            xt = b3dnint!(xbox) as f32 - xbox;
                            yt = b3dnint!(ybox) as f32 - ybox;
                            let box_off = ((self.m_iobj_do * max_any_sum + ib) as usize) * npix;
                            qd_shift(
                                &self.m_box_tmp,
                                &mut self.m_boxes[box_off..],
                                self.m_nx_box,
                                self.m_ny_box,
                                xt,
                                yt,
                            );
                            let wind =
                                ((self.tc.iobj_seq[iod] - 1) * self.mx.max_view + iz_box) as usize;
                            if self.m_wsum_save[wind] < 0. {
                                xtmp = 0.;
                                ytmp = 0.;
                                calc_cg(
                                    &mut self.cp,
                                    &self.m_boxes[box_off..],
                                    self.m_nx_box,
                                    self.m_ny_box,
                                    &mut xtmp,
                                    &mut ytmp,
                                    &mut wsum,
                                    &mut self.m_edge_sd,
                                );
                                self.m_wsum_save[wind] = wsum;
                                if get_edge_sdsave != 0
                                    && iz_box + 1 >= self.m_min_view_do
                                    && iz_box + 1 <= self.m_max_view_do
                                {
                                    let eind = ((self.tc.iobj_seq[iod] - 1) * self.m_max_view_do
                                        + iz_box)
                                        as usize;
                                    self.m_edge_sd_save[eind] = self.m_edge_sd;
                                    calc_elongation(
                                        &mut self.cp,
                                        &self.m_boxes[box_off..],
                                        self.m_nx_box,
                                        self.m_ny_box,
                                        0.,
                                        0.,
                                        &mut self.m_elong_save[eind],
                                    );
                                    if self.m_outer_sigma > 0. {
                                        calc_outer_mad(
                                            &mut self.cp,
                                            &self.m_boxes[box_off..],
                                            self.m_nx_box,
                                            self.m_ny_box,
                                            0.,
                                            0.,
                                            &mut self.m_outer_madsave[eind],
                                            &mut self.m_outer_background[eind],
                                        );
                                    }
                                }

                                // Check the integrals of the seed for light/dark beads setting
                                if self.m_isequence == 1 && self.m_iv_list == 1 {
                                    num_wtot += 1;
                                    if wsum <= 0. {
                                        num_zero_w += 1;
                                    }
                                    xtmp = 0.;
                                    ytmp = 0.;
                                    best_center_for_cg(
                                        &self.cp,
                                        &self.m_box_tmp,
                                        self.m_nx_box,
                                        self.m_ny_box,
                                        xtmp,
                                        ytmp,
                                        &mut ix,
                                        &mut iy,
                                        &mut best_sum,
                                    );
                                    let denom = b3dmax!(
                                        b3dabs!(1.0e-10 * wsum as f64),
                                        self.m_edge_sd as f64
                                    );
                                    snr_sum = (snr_sum as f64
                                        + (wsum as f64 * 4.
                                            / (3.14159
                                                * self.m_diameter as f64
                                                * self.m_diameter as f64))
                                            / denom)
                                        as f32;
                                    snr_sumb = (snr_sumb as f64 + best_sum as f64 / denom) as f32;
                                }
                            }
                        } else {
                            //
                            // Or if it is loaded, look it up
                            ibox = 0;
                            while ibox < max_any_sum {
                                if self.m_in_core[core_off + ibox as usize]
                                    == self.m_iz_close[ic as usize]
                                {
                                    ib = ibox;
                                }
                                ibox += 1;
                            }
                        }
                        let slot = (self.m_iobj_do * max_any_sum + ib) as usize;
                        //
                        // Add to correlation sum if not in there
                        if ic < self.m_max_sum && !self.m_in_corr_sum[slot] {
                            self.m_in_corr_sum[slot] = true;
                            ix = 0;
                            while ix < self.m_npix_box {
                                self.m_corr_sum[iod * npix + ix as usize] +=
                                    self.m_boxes[iod * max_any_sum as usize * npix
                                        + ib as usize * npix
                                        + ix as usize];
                                ix += 1;
                            }
                        }
                        //
                        // Add to sobel sum if not there
                        if ic < self.m_max_sobel_sum && !self.m_in_sobel_sum[slot] {
                            self.m_in_sobel_sum[slot] = true;
                            self.m_num_in_sobel_sum[iod] += 1;
                            ix = 0;
                            while ix < self.m_npix_box {
                                self.m_sobel_sum[iod * npix + ix as usize] +=
                                    self.m_boxes[slot * npix + ix as usize];
                                ix += 1;
                            }
                        }
                        ic += 1;
                    }
                }
            }
            self.m_ip_near_save[iod] = self.m_ip_nearest[iod];
            self.m_iobj_do += 1;
        }

        if num_wtot > 0 && num_zero_w > num_wtot / 2 {
            if self.m_if_white != 0 {
                exit_error(
                    b"Most beads are darker than background; do not use the light beads option",
                );
            }
            exit_error(
                b"Most beads are lighter than background; you need to use the light beads option",
            );
        }
        self.cp.get_edge_sd = get_edge_sdsave;
        //if (numWtot > 0) then
        //  print *, 'SNR is ', snrSum / numWtot, snrSumb / numWtot
        //endif
        let _ = (snr_sum, snr_sumb);
    }

    /// Original: `BeadTrack::getProjectedPositionsSetupFits` (`beadtrack.cpp:2149`).
    pub fn get_projected_positions_setup_fits(&mut self) {
        let mut ind_real_in_ali: Vec<i32>;
        let mut just_average: i32;
        // `izDelToNear` is written by every `nextPos` call that precedes its
        // reads.
        let mut iz_del_to_near: i32 = 0;
        let mut iz_tmp: i32 = 0;
        let mut iz_tmp2: i32;
        // `a` .. `f` are uninitialised in the source unless the alignment
        // branch runs; they are only read when it did.
        let mut a: f32 = 0.;
        let mut b: f32 = 0.;
        let mut c: f32 = 0.;
        let mut cosr: f32;
        let mut cost: f32;
        let mut d: f32 = 0.;
        let mut e: f32 = 0.;
        let mut f: f32 = 0.;
        let gmag_cur: f32;
        let mut ibase: i32;
        let mut indr: i32;
        let mut iobj: i32;
        let mut iobj_save: i32;
        let mut ip: i32;
        let mut ipt: i32;
        let mut iv: i32;
        let mut iv_del: i32;
        let mut ix: i32;
        let rot_cur: f32;
        let mut sinr: f32;
        let mut sint: f32;
        let tilt_cur: f32;
        let mut xpos: f32;
        let mut ypos: f32;

        ind_real_in_ali = vec![0; self.tc.num_obj_do.max(0) as usize];

        // Get the index of points in tiltalign and determine if all points to be done were
        // not in align AND consist of only 1 or 2 points on previous 2 views
        just_average = 1;
        self.m_iobj_do = 0;
        while self.m_iobj_do < self.tc.num_obj_do {
            let iod = self.m_iobj_do as usize;
            iobj = self.tc.iobj_seq[iod];
            ind_real_in_ali[iod] = 0;
            if self.m_if_did_align == 1 {
                ipt = 1;
                while ipt <= self.av.nreal_pt {
                    if self.tc.iobj_seq[iod] == self.tc.iobj_ali[(ipt - 1) as usize] {
                        ind_real_in_ali[iod] = self.tc.iobj_seq[iod];
                    }
                    ipt += 1;
                }
            }
            if self.m_if_found[iod] == 0 {
                let npt = self.fm.npt_in_obj[(iobj - 1) as usize];
                if ind_real_in_ali[iod] != 0 || npt > 2 {
                    just_average = 0;
                }
                if just_average > 0 {
                    just_average = b3dmax!(just_average, npt);
                    let base = self.fm.ibase_obj[(iobj - 1) as usize];
                    iz_tmp = b3dnint!(
                        self.fm.p_coord[(self.fm.object[base as usize] - 1) as usize][2] as f64
                            + 1.
                    );
                    iz_tmp2 = b3dnint!(
                        self.fm.p_coord[(self.fm.object[(base + npt - 1) as usize] - 1) as usize][2]
                            as f64
                            + 1.
                    );
                    if (iz_tmp != self.m_iview - self.m_track_dir
                        && iz_tmp != self.m_iview - 2 * self.m_track_dir)
                        || (iz_tmp2 != self.m_iview - self.m_track_dir
                            && iz_tmp2 != self.m_iview - 2 * self.m_track_dir)
                    {
                        just_average = 0;
                    }
                }
            }
            self.m_iobj_do += 1;
        }

        // If just averaging seems to be in order, now try to do it for all points, but
        // bail out if a previously done point does not have at least one in the range
        if just_average > 0 {
            self.m_iobj_do = 0;
            while self.m_iobj_do < self.tc.num_obj_do {
                let iod = self.m_iobj_do as usize;
                iobj = self.tc.iobj_seq[iod];
                ibase = self.fm.ibase_obj[(iobj - 1) as usize];
                xpos = 0.;
                ypos = 0.;
                ix = 0;
                ipt = 1;
                while ipt <= self.fm.npt_in_obj[(iobj - 1) as usize] {
                    ip = self.fm.object[(ibase + ipt - 1) as usize];
                    let pt = self.fm.p_coord[(ip - 1) as usize];
                    iz_tmp = b3dnint!(pt[2] as f64 + 1.);
                    if iz_tmp == self.m_iview - self.m_track_dir
                        || iz_tmp == self.m_iview - just_average * self.m_track_dir
                    {
                        xpos += pt[0];
                        ypos += pt[1];
                        ix += 1;
                    }
                    ipt += 1;
                }
                if ix == 0 {
                    just_average = 0;
                    break;
                }
                self.m_xseek[iod] = xpos / ix as f32;
                self.m_yseek[iod] = ypos / ix as f32;
                self.m_xseek_next_pos[iod] = self.m_xseek[iod];
                self.m_yseek_next_pos[iod] = self.m_yseek[iod];
                //
                // Adjust the positions by the shifts near zero if appropriate
                // If justAverage stays > 0, then this object will not enter the clause below
                // that does the other adjustment.  If justAverage gets set to 0 after this, then
                // both positions will be replaced by the average in the first section of the
                // clause below and then readjusted if appropriate
                if self.m_minz_delz_near_zero > 0
                    && self.fm.npt_in_obj[(iobj - 1) as usize] == 1
                    && b3dabs!(iz_tmp - self.m_iz_next) <= self.m_max_delz_near_zero
                    && b3dabs!(self.m_iz_next + 1 - self.tc.min_tilt_ind)
                        <= self.m_minz_delz_near_zero
                {
                    iz_del_to_near = self.m_iz_next + 1 - iz_tmp;
                    self.m_xseek_next_pos[iod] +=
                        iz_del_to_near as f32 * self.m_x_shift_near_zero[0];
                    self.m_yseek_next_pos[iod] +=
                        iz_del_to_near as f32 * self.m_y_shift_near_zero[0];
                    self.m_xseek[iod] += iz_del_to_near as f32 * self.m_x_shift_near_zero[1];
                    self.m_yseek[iod] += iz_del_to_near as f32 * self.m_y_shift_near_zero[1];
                    if self.m_if_trace != 0 {
                        printf!(
                            "adjust%4d%8.1f%8.1f%8.1f%8.1f\n",
                            CArg::Int(iobj as i64),
                            CArg::Dbl(self.m_xseek[iod] as f64),
                            CArg::Dbl(self.m_yseek[iod] as f64),
                            CArg::Dbl(self.m_xseek_next_pos[iod] as f64),
                            CArg::Dbl(self.m_yseek_next_pos[iod] as f64)
                        );
                    }
                }
                //if (saveAllPoints) print *,'justAvg', iobj, xseek(iobjDo), yseek(iobjDo)
                self.m_iobj_do += 1;
            }
        }
        //
        // If not just averaging, get tentative tilt, rotation, mag for current view
        //
        if self.m_if_did_align > 0 && just_average == 0 {
            self.m_iv_use = self.m_iview;
            if self.av.map_file_to_view[(self.m_iview - 1) as usize] == 0 {
                iv_del = 200;
                iv = 1;
                while iv <= self.av.nview {
                    let mv = self.av.map_view_to_file[(iv - 1) as usize];
                    if b3dabs!(mv - self.m_iview) < iv_del {
                        iv_del = b3dabs!(mv - self.m_iview);
                        self.m_iv_use = mv;
                    }
                    iv += 1;
                }
            }
            if self.m_save_all_points {
                printf!(
                    "ivuse %d iview %d\n",
                    CArg::Int(self.m_iv_use as i64),
                    CArg::Int(self.m_iview as i64)
                );
            }
            let ivu = (self.m_iv_use - 1) as usize;
            tilt_cur = self.tc.tilt_orig[ivu] + self.tc.tilt_all[(self.m_iview - 1) as usize]
                - self.tc.tilt_all[ivu];
            gmag_cur = self.tc.gmag_orig[ivu];
            rot_cur = self.tc.rot_orig[ivu];
            cosr = cosd(rot_cur);
            sinr = sind(rot_cur);
            cost = cosd(tilt_cur);
            sint = sind(tilt_cur);
            self.m_dx_cur = self.tc.dxy_save[ivu * 2];
            self.m_dy_cur = self.tc.dxy_save[ivu * 2 + 1];
            if self.m_iv_use != self.m_iview && self.m_tilt_max as f64 <= 80. {
                //
                // if going from another view and cosine stretch is appropriate,
                // back - rotate the dxy so tilt axis vertical, adjust the
                // dx by the difference in tilt angle, and rotate back
                a = cosr * self.m_dx_cur + sinr * self.m_dy_cur;
                b = -sinr * self.m_dx_cur + cosr * self.m_dy_cur;
                a *= cost / cosd(self.tc.tilt_orig[ivu]);
                self.m_dx_cur = cosr * a - sinr * b;
                self.m_dy_cur = sinr * a + cosr * b;
            }
            a = gmag_cur * cost * cosr;
            b = -gmag_cur * sinr;
            c = gmag_cur * sint * cosr;
            d = gmag_cur * cost * sinr;
            e = gmag_cur * cosr;
            f = gmag_cur * sint * sinr;
        }
        //
        // now get projected positions otherwise, and save the points
        //
        self.m_iobj_do = 0;
        while self.m_iobj_do < self.tc.num_obj_do {
            let iod = self.m_iobj_do as usize;
            indr = ind_real_in_ali[iod];
            iobj = self.tc.iobj_seq[iod];
            ibase = self.fm.ibase_obj[(iobj - 1) as usize];
            let _ = ibase;
            if indr != 0 && just_average == 0 {
                //
                // there is a 3D point for it: so project it
                //
                let xs = self.tc.xyz_save[(indr * 3 - 3) as usize];
                let ys = self.tc.xyz_save[(indr * 3 - 2) as usize];
                let zs = self.tc.xyz_save[(indr * 3 - 1) as usize];
                self.m_xseek[iod] = a * xs + b * ys + c * zs + self.m_dx_cur + self.tc.xcen;
                self.m_yseek[iod] = d * xs + e * ys + f * zs + self.m_dy_cur + self.tc.ycen;
                //
                // Get alternate position if parameter set; otherwise copy it
                if self.m_try_alt_pred_diam_frac > 0. {
                    next_pos(
                        &self.fm,
                        &self.mx,
                        iobj,
                        self.m_ip_nearest[iod],
                        self.m_track_dir,
                        self.m_iz_next,
                        &self.tc.tilt_all,
                        self.m_num_fit,
                        self.m_min_fit,
                        self.m_rot_start,
                        self.m_tilt_fit_min,
                        &self.tc.iz_exclude,
                        self.tc.num_exclude,
                        &mut self.m_xseek_next_pos[iod],
                        &mut self.m_yseek_next_pos[iod],
                        self.tc.min_tilt_ind,
                        0,
                        0,
                        &mut iz_del_to_near,
                    );
                } else {
                    self.m_xseek_next_pos[iod] = self.m_xseek[iod];
                    self.m_yseek_next_pos[iod] = self.m_yseek[iod];
                }
            } else if just_average == 0 || self.fm.npt_in_obj[(iobj - 1) as usize] > 1 {
                //
                // If no averaging but no 3D pos, get extrapolated position and copy it
                // If it is 1-point value, both will be adjusted by near zero shift if appropriate
                if just_average == 0 {
                    next_pos(
                        &self.fm,
                        &self.mx,
                        iobj,
                        self.m_ip_nearest[iod],
                        self.m_track_dir,
                        self.m_iz_next,
                        &self.tc.tilt_all,
                        self.m_num_fit,
                        self.m_min_fit,
                        self.m_rot_start,
                        self.m_tilt_fit_min,
                        &self.tc.iz_exclude,
                        self.tc.num_exclude,
                        &mut self.m_xseek[iod],
                        &mut self.m_yseek[iod],
                        self.tc.min_tilt_ind,
                        self.m_minz_delz_near_zero,
                        self.m_max_delz_near_zero,
                        &mut iz_del_to_near,
                    );
                    self.m_xseek_next_pos[iod] = self.m_xseek[iod];
                    self.m_yseek_next_pos[iod] = self.m_yseek[iod];
                }
                //
                // If there was averaging or not enough points for a fit, take what nextPos
                // offers as a possible different (extrapolated) position
                // There is currently an average in each spot, but if justAverage > 0 they are
                // both already adjusted positions.  If this fails to do  a fit and calls for
                // adjustment, the NextPos position is replaced here and needs adjustment
                // but the xseek position as already adjusted and should not be adjusted again
                if just_average != 0 || self.fm.npt_in_obj[(iobj - 1) as usize] < self.m_min_fit {
                    next_pos(
                        &self.fm,
                        &self.mx,
                        iobj,
                        self.m_ip_nearest[iod],
                        self.m_track_dir,
                        self.m_iz_next,
                        &self.tc.tilt_all,
                        self.m_num_fit,
                        2,
                        self.m_rot_start,
                        self.m_tilt_fit_min,
                        &self.tc.iz_exclude,
                        self.tc.num_exclude,
                        &mut self.m_xseek_next_pos[iod],
                        &mut self.m_yseek_next_pos[iod],
                        self.tc.min_tilt_ind,
                        self.m_minz_delz_near_zero,
                        self.m_max_delz_near_zero,
                        &mut iz_del_to_near,
                    );
                }
                if iz_del_to_near != 0 {
                    self.m_xseek_next_pos[iod] +=
                        iz_del_to_near as f32 * self.m_x_shift_near_zero[0];
                    self.m_yseek_next_pos[iod] +=
                        iz_del_to_near as f32 * self.m_y_shift_near_zero[0];
                    if just_average == 0 {
                        self.m_xseek[iod] += iz_del_to_near as f32 * self.m_x_shift_near_zero[1];
                        self.m_yseek[iod] += iz_del_to_near as f32 * self.m_y_shift_near_zero[1];
                    }
                }
            }
            if self.m_save_all_points {
                iobj_save = iobj + (5 * self.m_ipass - 4) * self.m_max_obj_orig;
                iz_tmp = 0;
                add_point(
                    &mut self.fm,
                    iobj_save,
                    &mut iz_tmp,
                    self.m_xseek[iod],
                    self.m_yseek[iod],
                    self.m_iz_next,
                );
            }
            self.m_iobj_do += 1;
        }
        //
        // build list of points already found for fitting; put
        // actual positions in 4 and 5 for compatibiity with xfmodel
        //
        self.m_num_data = 0;
        let xms = self.m_xmat_size as usize;
        self.m_iobj_do = 0;
        while self.m_iobj_do < self.tc.num_obj_do {
            let iod = self.m_iobj_do as usize;
            if self.m_if_found[iod] > 0 {
                let row = self.m_num_data as usize * xms;
                self.m_xmat[row] = self.m_xseek[iod] - self.tc.xcen;
                self.m_xmat[row + 1] = self.m_yseek[iod] - self.tc.ycen;
                ipt = self.fm.object[(self.fm.ibase_obj[(self.tc.iobj_seq[iod] - 1) as usize]
                    + self.m_ip_nearest[iod]
                    - 1) as usize];
                self.m_xmat[row + 3] = self.fm.p_coord[(ipt - 1) as usize][0] - self.tc.xcen;
                self.m_xmat[row + 4] = self.fm.p_coord[(ipt - 1) as usize][1] - self.tc.ycen;
                self.m_xmat[row + 5] = self.m_iobj_do as f32;
                self.m_num_data += 1;
            }
            self.m_iobj_do += 1;
        }
        drop(ind_real_in_ali);
    }
}

impl BeadTrack {
    /// Original: `BeadTrack::getWsumCriteria` (`beadtrack.cpp:2368`).
    ///
    /// Analyze wsums in the vicinity and make up a criterion for each bead
    pub fn get_wsum_criteria(&mut self) {
        let mut iv_fit: Vec<f32>;
        let mut ws_fit: Vec<f32>;
        let mut w_local_means: Vec<f32>;
        let mut prederr: f32 = 0.;
        let mut pr_slope: f32 = 0.;
        let mut pr_intcp: f32 = 0.;
        let mut prro: f32 = 0.;
        let mut prsa: f32 = 0.;
        let mut prsb: f32 = 0.;
        let mut prse: f32 = 0.;
        let mut wpred: f32 = 0.;
        let mut wpred_local: f32;
        let mut ws_pctl: Vec<f32>;
        let mut wsums_local: Vec<f32>;
        let mut wlocal_sum: f32;
        let mut dbl_norm_crit: f32;
        let min_wsum_for_pred: i32;
        let mut num_in_wlocal_mean: Vec<i32>;
        let max_local_for_pred: i32;
        let min_usable_num: i32;
        let mut max_num_in_mean: i32;
        let mut num_locals: i32;
        let mut num_neigh_in_local: i32;
        let mut min_dif: i32;
        let mut num_pctl: i32;
        let mut i: i32;
        let mut idif: i32;
        let mut idirw: i32;
        let mut iobj: i32;
        let mut itry: i32;
        let mut ix: i32;
        let mut num_avg: i32;
        let lm_dim: i32;
        let fit_dim: i32;
        let wl_dim: i32;
        let mut wstmp: f32;
        let mut wsum: f32;
        let mut wsum_avg: f32 = 0.;
        let mut wsum_sd: f32 = 0.;
        let mut wsumsq: f32;
        let mut xtmp: f32 = 0.;
        lm_dim = 2 * self.m_max_wavg + 1;
        fit_dim = self.m_max_wavg + 2;
        wl_dim = self.m_max_wneigh * (self.m_max_wavg + 1);
        min_wsum_for_pred = 4;
        max_local_for_pred = 7;
        iv_fit = vec![0.; fit_dim as usize];
        ws_fit = vec![0.; fit_dim as usize];
        ws_pctl = vec![0.; fit_dim as usize];
        w_local_means = vec![0.; lm_dim as usize];
        num_in_wlocal_mean = vec![0; lm_dim as usize];
        wsums_local = vec![0.; wl_dim as usize];

        let max_wavg = self.m_max_wavg;
        let max_view = self.mx.max_view;
        let seq_off = ((self.m_last_seq - 1) * self.m_max_neigh) as usize;

        // Go through the collection of neighbors for wsum and get the mean on views in
        // a large range
        num_locals = 0;
        num_in_wlocal_mean.fill(0);
        w_local_means.fill(0.);
        self.m_iobj_do = 1;
        while self.m_iobj_do <= self.m_num_wneighbors[(self.m_last_seq - 1) as usize] {
            iobj = self.m_neighbors_for_wfits[seq_off + (self.m_iobj_do - 1) as usize];
            itry = b3dmax!(self.m_min_view_do, self.m_iview - max_wavg);
            while itry <= b3dmin!(self.m_max_view_do, self.m_iview + max_wavg) {
                idif = itry - self.m_iview;
                let w = self.m_wsum_save[((iobj - 1) * max_view + itry - 1) as usize];
                if w >= 0. {
                    num_in_wlocal_mean[(max_wavg + idif) as usize] += 1;
                    w_local_means[(max_wavg + idif) as usize] += w;
                }
                itry += 1;
            }
            self.m_iobj_do += 1;
        }

        // Get the mean, and max number in any group, and limit the usable ones to
        // ones with at least 3 and at least 1/5 of the maximum up to 9.
        max_num_in_mean = 0;
        i = 0;
        while i < lm_dim {
            max_num_in_mean = b3dmax!(max_num_in_mean, num_in_wlocal_mean[i as usize]);
            if num_in_wlocal_mean[i as usize] > 0 {
                w_local_means[i as usize] /= num_in_wlocal_mean[i as usize] as f32;
            }
            i += 1;
        }
        min_usable_num = b3dmin!(9, b3dmax!(3, max_num_in_mean / 5));

        // Do a fit over usable means
        num_avg = 0;
        min_dif = 100;
        ix = 0;
        while ix <= max_wavg && num_avg < max_local_for_pred {
            idirw = -1;
            while idirw <= 1 {
                idif = ix * idirw;
                if num_in_wlocal_mean[(max_wavg + idif) as usize] >= min_usable_num {
                    min_dif = b3dmin!(min_dif, b3dabs!(idif));
                    iv_fit[num_avg as usize] = idif as f32;
                    ws_fit[num_avg as usize] = w_local_means[(max_wavg + idif) as usize];
                    num_avg += 1;
                    if num_avg >= max_local_for_pred {
                        break;
                    }
                }
                idirw += 2;
            }
            ix += 1;
        }

        wpred_local = 0.;

        // Get the average and do a predictive line fit if there are enough
        if num_avg > 0 {
            avg_sd(&ws_fit, num_avg, &mut wsum_avg, &mut wsum_sd, &mut xtmp);
            if num_avg > min_wsum_for_pred && min_dif < 3 {
                ls_fit_pred(
                    &iv_fit,
                    &ws_fit,
                    num_avg,
                    &mut pr_slope,
                    &mut pr_intcp,
                    &mut prro,
                    &mut prsa,
                    &mut prsb,
                    &mut prse,
                    0.,
                    &mut wpred_local,
                    &mut prederr,
                );
                if self.m_save_all_points {
                    printf!(
                        "pred%4d%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f\n",
                        CArg::Int(num_avg as i64),
                        CArg::Dbl(wsum_avg as f64),
                        CArg::Dbl(wsum_sd as f64),
                        CArg::Dbl(wpred_local as f64),
                        CArg::Dbl(prederr as f64),
                        CArg::Dbl(pr_slope as f64),
                        CArg::Dbl(prsb as f64)
                    );
                }
            } else {
                wpred_local = wsum_avg;
            }
        }

        // Look at the neighbors and get the mean of values normalized by view mean
        // and collect.  This is done the same way as individuals below, go up to maxWavg
        // views away and get up to maxWavg + 1 closest nviews
        num_neigh_in_local = 0;
        self.m_iobj_do = 1;
        while self.m_iobj_do <= self.m_num_wneighbors[(self.m_last_seq - 1) as usize] {
            iobj = self.m_neighbors_for_wfits[seq_off + (self.m_iobj_do - 1) as usize];
            wsum = 0.;
            num_avg = 0;
            ix = 0;
            while ix <= max_wavg {
                idirw = -1;
                while idirw <= 1 {
                    idif = ix * idirw;
                    if num_in_wlocal_mean[(max_wavg + idif) as usize] >= min_usable_num {
                        itry = self.m_iview + idif;
                        let w = self.m_wsum_save[((iobj - 1) * max_view + itry - 1) as usize];
                        if w >= 0. {
                            num_avg += 1;
                            wstmp = w / w_local_means[(max_wavg + idif) as usize];
                            wsum += wstmp;
                            wsums_local[(num_locals + num_avg - 1) as usize] = wstmp;
                        }
                    }
                    idirw += 2;
                }
                if num_avg >= max_wavg {
                    break;
                }
                ix += 1;
            }

            // Divide the ones just added to the list by the bead mean so the list now has
            // double-normalized values
            if num_avg > 0 && wsum > 0. {
                num_neigh_in_local += 1;
                ix = num_locals;
                while ix < num_locals + num_avg {
                    wsums_local[ix as usize] /= wsum / num_avg as f32;
                    ix += 1;
                }
                num_locals += num_avg;
            }
            self.m_iobj_do += 1;
        }

        // Set the double-normalized criterion by taking what can be considered a 20th
        // percentile point of the minima from all the neighbors, and down-rating that,
        // but don't let it get too low
        dbl_norm_crit = 0.;
        if num_locals >= 5 {
            idif = 1 + num_neigh_in_local / 5;
            dbl_norm_crit =
                self.m_percentile_crit_frac * percentile_float(idif, &mut wsums_local, num_locals);
            if self.m_save_all_points {
                printf!(
                    "%d locals, crit: %d %f\n",
                    CArg::Int(num_locals as i64),
                    CArg::Int(idif as i64),
                    CArg::Dbl(dbl_norm_crit as f64)
                );
            }
            dbl_norm_crit = b3dmax!(dbl_norm_crit, self.m_dbl_norm_min_crit);
        }

        self.m_iobj_do = 1;
        while self.m_iobj_do <= self.tc.num_obj_do {
            let iod = (self.m_iobj_do - 1) as usize;
            if self.m_if_found[iod] == 0 || self.m_if_found[iod] == 1 {
                //
                // For each one to be done, accumulate sums for all the views in range of both
                // the bead itself and of the local means on the same views.  Store the first
                // maxWavg + 1 of those for personalized fitting if needed, and store a normalized
                // value for views with usable local means for percentile finding
                //
                num_avg = 0;
                wlocal_sum = 0.;
                num_pctl = 0;
                wsum = 0.;
                wsumsq = 0.;
                self.m_wsum_min[iod] = -1.;
                ix = 0;
                while ix <= max_wavg {
                    idirw = -1;
                    while idirw <= 1 {
                        idif = ix * idirw;
                        if num_in_wlocal_mean[(max_wavg + idif) as usize] > 0 {
                            itry = self.m_iview + idif;
                            wstmp = self.m_wsum_save
                                [((self.tc.iobj_seq[iod] - 1) * max_view + itry - 1) as usize];
                            if wstmp >= 0. {
                                wsum += wstmp;
                                wsumsq += wstmp * wstmp;
                                wlocal_sum += w_local_means[(max_wavg + idif) as usize];
                                num_avg += 1;
                                if num_avg < max_wavg + 2 {
                                    ws_fit[(num_avg - 1) as usize] = wstmp;
                                    iv_fit[(num_avg - 1) as usize] = idif as f32;
                                }
                                if num_in_wlocal_mean[(max_wavg + idif) as usize] >= min_usable_num
                                {
                                    num_pctl += 1;
                                    ws_pctl[(num_pctl - 1) as usize] =
                                        wstmp / w_local_means[(max_wavg + idif) as usize];
                                }
                                if self.m_wsum_min[iod] < 0. {
                                    self.m_wsum_min[iod] = wstmp;
                                } else {
                                    self.m_wsum_min[iod] = b3dmin!(wstmp, self.m_wsum_min[iod]);
                                }
                            }
                        }
                        idirw += 2;
                    }
                    if num_pctl >= max_wavg {
                        break;
                    }
                    ix += 1;
                }

                sums_to_avg_sd(wsum, wsumsq, num_avg, &mut wsum_avg, &mut wsum_sd);
                wlocal_sum /= num_avg as f32;
                if wpred_local > 0. && num_pctl > 0 {
                    // If there is a good local prediction, then use  a down-rated 20th percentile
                    // value if there is enough data in the distribution, otherwise use the
                    // established criterion.  Do not let the normalized criterion get too low
                    xtmp = dbl_norm_crit;
                    if num_pctl >= 5 || dbl_norm_crit == 0. {
                        idif = 1 + num_pctl / 5;
                        xtmp = b3dmax!(
                            self.m_percentile_crit_frac
                                * percentile_float(idif, &mut ws_pctl, num_pctl),
                            self.m_dbl_norm_min_crit
                        );
                    }

                    // Scale the criterion by the predicted value and the ratio of this bead's mean
                    // to the corresponding local mean.  Limit to a small fraction of the local mean
                    //
                    self.m_wsum_crit[iod] = b3dmax!(
                        xtmp * wpred_local * wsum_avg / wlocal_sum,
                        self.m_wcrit_to_local_min_ratio * wlocal_sum
                    );
                    self.m_wsum_crit[iod] = b3dmin!(
                        self.m_wsum_crit[iod],
                        self.m_percentile_crit_frac * wsum_avg
                    );
                    if self.m_save_all_points {
                        printf!(
                            "crit%4d%4d%4d%8.0f%8.0f%8.4f%8.0f\n",
                            CArg::Int(self.tc.iobj_seq[iod] as i64),
                            CArg::Int(num_pctl as i64),
                            CArg::Int(num_avg as i64),
                            CArg::Dbl(wsum_avg as f64),
                            CArg::Dbl(wlocal_sum as f64),
                            CArg::Dbl(xtmp as f64),
                            CArg::Dbl(self.m_wsum_crit[iod] as f64)
                        );
                    }
                } else {
                    // If no good local prediction, fall back to the old mean/SD based criterion but
                    // Do not allow it to get too low.  Then, if there is enough data for a line fit,
                    // get a predicted value and use it if the error is low enough and the slope is
                    // significant.  Relax the predicted value by less than the mean would be
                    self.m_wsum_crit[iod] = b3dmax!(
                        b3dmin!(
                            self.m_frac_crit * wsum_avg,
                            wsum_avg - self.m_sd_crit * wsum_sd
                        ),
                        wsum_avg * self.m_dbl_norm_min_crit
                    );
                    if num_avg >= b3dnint!(1.5 * min_wsum_for_pred as f64) {
                        ls_fit_pred(
                            &iv_fit,
                            &ws_fit,
                            b3dmin!(num_avg, max_wavg + 1),
                            &mut pr_slope,
                            &mut pr_intcp,
                            &mut prro,
                            &mut prsa,
                            &mut prsb,
                            &mut prse,
                            0.,
                            &mut wpred,
                            &mut prederr,
                        );
                        //write(*,'(a,5f8.0)') 'bead fit', wsumAvg, wsumSD, wpred, prSlope, prsb
                        if prederr < wsum_sd && b3dabs!(pr_slope) as f64 > 2.5 * prsb as f64 {
                            self.m_wsum_crit[iod] = b3dmax!(
                                (1. - 0.6 * (1. - self.m_frac_crit as f64)) * wpred as f64,
                                (wsum_avg * self.m_dbl_norm_min_crit) as f64
                            ) as f32;
                        }
                        //write(*,'(a,3f8.0)') 'replacement', wsumCrit(iobjDo)
                    }
                }
            }
            self.m_iobj_do += 1;
        }
    }

    /// Original: `BeadTrack::findAllBeadsOnView` (`beadtrack.cpp:2605`).
    pub fn find_all_beads_on_view(&mut self, save_boxes: bool) {
        let mut if_ro_trans: i32;
        let mut if_trans: i32;
        let mut iobj: i32;
        let mut iobj_save: i32;
        let mut ipnt_max_dev: i32 = 0;
        let mut ix: i32;
        let mut max_drop: i32;
        let mut ndat_fit: i32;
        let mut ip_near: i32;
        let mut num_drop: i32 = 0;
        let mut dev_avg: f32 = 0.;
        let mut dev_max: f32 = 0.;
        let mut dev_sd: f32 = 0.;
        let mut wsum: f32 = 0.;

        self.m_iobj_do = 1;
        while self.m_iobj_do <= self.tc.num_obj_do {
            let iod = (self.m_iobj_do - 1) as usize;
            if self.m_if_found[iod] == 0 {
                iobj = self.tc.iobj_seq[iod];
                //
                // Save the alternate positions for this bead and set up to try both if they
                // differ enough
                self.m_alt_xseek[0] = self.m_xseek[iod];
                self.m_alt_yseek[0] = self.m_yseek[iod];
                self.m_alt_xseek[1] = self.m_xseek_next_pos[iod];
                self.m_alt_yseek[1] = self.m_yseek_next_pos[iod];
                self.m_num_pred_tries = 1;
                {
                    let dx = self.m_alt_xseek[0] - self.m_alt_xseek[1];
                    let dy = self.m_alt_yseek[0] - self.m_alt_yseek[1];
                    if self.m_try_alt_pred_diam_frac > 0.
                        && self.m_try_alt_pred_diam_frac * self.m_diameter
                            < (dx * dx + dy * dy).sqrt()
                    {
                        self.m_num_pred_tries = 2;
                    }
                }
                self.m_ind_pred_try = 1;
                while self.m_ind_pred_try <= self.m_num_pred_tries {
                    let ipt_u = (self.m_ind_pred_try - 1) as usize;
                    self.m_xnext = self.m_alt_xseek[ipt_u];
                    self.m_ynext = self.m_alt_yseek[ipt_u];
                    //
                    // if there are existing points, find right kind of transform
                    // and modify position
                    //
                    if self.m_num_pioneer > 0 || self.m_num_data >= self.m_lim_pts_shift {
                        if_ro_trans = 0;
                        if_trans = 1;
                        ndat_fit = self.m_num_data;
                        if self.m_num_data < self.m_num_pioneer + self.m_lim_pts_shift {
                            ndat_fit = self.m_num_pioneer;
                        }
                        if ndat_fit >= self.m_lim_pts_stretch {
                            if_trans = 0;
                        } else if ndat_fit >= self.m_lim_pts_mag {
                            if_ro_trans = 2;
                        } else if ndat_fit >= self.m_lim_pts_rot {
                            if_ro_trans = 1;
                        }
                        max_drop = b3dnint!(0.26 * ndat_fit as f64);
                        if ndat_fit < 4 || self.m_if_did_align == 0 {
                            max_drop = 0;
                        }
                        findxf_wo_outliers(
                            &mut self.m_xmat,
                            self.m_xmat_size,
                            ndat_fit,
                            self.tc.xcen,
                            self.tc.ycen,
                            if_trans,
                            if_ro_trans,
                            max_drop,
                            self.m_outlie_crit,
                            self.m_outlie_crit_abs,
                            self.m_outlie_elim_min,
                            &mut self.m_idrop,
                            &mut num_drop,
                            &mut self.m_xform,
                            &mut dev_avg,
                            &mut dev_sd,
                            &mut dev_max,
                            &mut ipnt_max_dev,
                        );
                        (self.m_xnext, self.m_ynext) = xf_apply(
                            &self.m_xform,
                            self.tc.xcen,
                            self.tc.ycen,
                            self.m_alt_xseek[ipt_u],
                            self.m_alt_yseek[ipt_u],
                            2,
                        );
                        if self.m_save_all_points {
                            iobj_save = iobj + (5 * self.m_ipass - 3) * self.m_max_obj_orig;
                            ip_near = 0;
                            add_point(
                                &mut self.fm,
                                iobj_save,
                                &mut ip_near,
                                self.m_xnext,
                                self.m_ynext,
                                self.m_iz_next,
                            );
                        }
                    }
                    //
                    find_piece(
                        &self.m_ix_pclist,
                        &self.m_iy_pclist,
                        &self.m_iz_pclist,
                        self.m_npclist,
                        self.m_nx_im,
                        self.m_ny_im,
                        self.m_nx_box,
                        self.m_ny_box,
                        self.m_xnext,
                        self.m_ynext,
                        self.m_iz_next,
                        &mut self.m_ix0,
                        &mut self.m_ix1,
                        &mut self.m_iy0,
                        &mut self.m_iy1,
                        &mut self.m_ipiece_z,
                        self.m_if_read_xfs,
                        &self.m_prexf,
                        &mut self.m_need_taper,
                        &mut self.m_need_fill,
                    );
                    if self.m_ipiece_z >= 0 {
                        self.look_for_one_bead(iobj, &mut wsum);
                    } else {
                        wsum = 0.;
                    }
                    //
                    // Save values from the tried position
                    self.m_alt_pred_wsum[ipt_u] = wsum;
                    self.m_alt_pred_xpeak[ipt_u] = self.m_xpeak;
                    self.m_alt_pred_ypeak[ipt_u] = self.m_ypeak;
                    self.m_alt_pred_edge_sd[ipt_u] = self.m_edge_sd;
                    self.m_alt_pred_xnext[ipt_u] = self.m_xnext;
                    self.m_alt_pred_ynext[ipt_u] = self.m_ynext;
                    self.m_ind_pred_try += 1;
                }
                //
                // If multiple tries, evaluate which is better; make a score that is the sum of
                // the fractional distance from target and the OTHER try's Wsum value relative to
                // the max wsum.  So smaller is better
                self.m_ind_pred_try = 1;
                if self.m_num_pred_tries > 1 {
                    ix = 0;
                    while ix < 2 {
                        let u = ix as usize;
                        self.m_alt_xpos[u] =
                            b3dnint!(self.m_alt_pred_xnext[u]) as f32 + self.m_alt_pred_xpeak[u];
                        self.m_alt_ypos[u] =
                            b3dnint!(self.m_alt_pred_ynext[u]) as f32 + self.m_alt_pred_ypeak[u];
                        let dx = self.m_alt_xpos[u] - self.m_alt_pred_xnext[u];
                        let dy = self.m_alt_ypos[u] - self.m_alt_pred_ynext[u];
                        // `sqrtf(...) / mDiameter` is a float quotient; the Wsum
                        // ratio divides by a double `B3DMAX(1., ...)`, so the sum
                        // is formed in double and narrowed on the store.
                        self.m_alt_score[u] = (((dx * dx + dy * dy).sqrt() / self.m_diameter)
                            as f64
                            + self.m_alt_pred_wsum[1 - u] as f64
                                / b3dmax!(
                                    1.,
                                    b3dmax!(self.m_alt_pred_wsum[0], self.m_alt_pred_wsum[1])
                                        as f64
                                )) as f32;
                        ix += 1;
                    }
                    if self.m_alt_score[1] < self.m_alt_score[0] {
                        self.m_ind_pred_try = 2;
                    }
                    if self.m_if_trace != 0 {
                        printf!(
                            "Trials:%2d%5d%8.1f%8.1f%8.1f%8.1f%9.1f%8.4f    %8.1f%8.1f%8.1f%8.1f%9.1f%8.4f\n",
                            CArg::Int(self.m_ind_pred_try as i64),
                            CArg::Int(iobj as i64),
                            CArg::Dbl(self.m_alt_pred_xnext[0] as f64),
                            CArg::Dbl(self.m_alt_pred_ynext[0] as f64),
                            CArg::Dbl(self.m_alt_xpos[0] as f64),
                            CArg::Dbl(self.m_alt_ypos[0] as f64),
                            CArg::Dbl(self.m_alt_pred_wsum[0] as f64),
                            CArg::Dbl(self.m_alt_score[0] as f64),
                            CArg::Dbl(self.m_alt_pred_xnext[1] as f64),
                            CArg::Dbl(self.m_alt_pred_ynext[1] as f64),
                            CArg::Dbl(self.m_alt_xpos[1] as f64),
                            CArg::Dbl(self.m_alt_ypos[1] as f64),
                            CArg::Dbl(self.m_alt_pred_wsum[1] as f64),
                            CArg::Dbl(self.m_alt_score[1] as f64)
                        );
                    }
                }
                let ipt_u = (self.m_ind_pred_try - 1) as usize;
                wsum = self.m_alt_pred_wsum[ipt_u];
                self.m_xpeak = self.m_alt_pred_xpeak[ipt_u];
                self.m_ypeak = self.m_alt_pred_ypeak[ipt_u];
                self.m_edge_sd = self.m_alt_pred_edge_sd[ipt_u];
                self.m_xnext = self.m_alt_pred_xnext[ipt_u];
                self.m_ynext = self.m_alt_pred_ynext[ipt_u];
                if wsum > 0. {
                    self.add_best_bead_found(iobj, wsum, save_boxes);
                }
            }
            self.m_iobj_do += 1;
        }
    }
}

impl BeadTrack {
    /// Original: `BeadTrack::lookForOneBead` (`beadtrack.cpp:2715`).
    ///
    /// lookForOneBead with the current trial position
    pub fn look_for_one_bead(&mut self, iobj: i32, wsum: &mut f32) {
        let mut jobj: i32;
        let mut jobj_do: i32;
        let mut num_maxes: i32;
        let mut i: i32;
        let mut ind: i32;
        let mut ip: i32;
        let mut ierr: i32;
        let mut iobj_save: i32;
        let mut ipt: i32;
        let mut itry: i32;
        let mut nin_sobel: i32;
        let xpeak_orig: f32;
        let ypeak_orig: f32;
        let wsum_orig: f32;
        let mut wmax_median: f32 = 0.;
        let mut wmax_madn: f32 = 0.;
        let mut reject_max_crit: f32;
        let dist: f32;
        let mut elongation: f32 = 0.;
        let rad_max: f32;
        let mut relax: f32;
        let mut wsum_bkgd_ratio: f32;
        let mut x_off_sobel: f32 = 0.;
        let mut xtmp: f32 = 0.;
        let mut y_off_sobel: f32 = 0.;
        let mut ytmp: f32 = 0.;
        let dist_rescue: bool;
        let npix = self.m_npix_box as usize;
        let iod = (self.m_iobj_do - 1) as usize;
        //
        // get image area
        //
        {
            let mut load_tmp = std::mem::take(&mut self.m_box_tmp);
            let ctf = std::mem::take(&mut self.m_ctf);
            self.load_box_and_taper(
                &mut load_tmp,
                self.m_nx_box,
                self.m_ny_box,
                self.m_nx_pad,
                self.m_ny_pad,
                &ctf,
                self.m_delta_ctf,
            );
            self.m_ctf = ctf;
            self.m_box_tmp = load_tmp;
        }

        // Do cross-correlation on first pass
        if self.m_ipass == 1 {
            //
            // pad image into array on first pass, pad correlation sum into brray
            slice_taper_in_pad(
                PadIn::Float(&self.m_box_tmp),
                SLICE_MODE_FLOAT,
                self.m_nx_box,
                0,
                self.m_nx_box - 1,
                0,
                self.m_ny_box - 1,
                &mut self.m_array,
                self.m_nxp_dim,
                self.m_nx_pad,
                self.m_ny_pad,
                self.m_nx_taper,
                self.m_ny_taper,
            );
            xcorr_mean_zero(
                &mut self.m_array,
                self.m_nxp_dim,
                self.m_nx_pad,
                self.m_ny_pad,
            );
            slice_taper_in_pad(
                PadIn::Float(&self.m_corr_sum[iod * npix..]),
                SLICE_MODE_FLOAT,
                self.m_nx_box,
                0,
                self.m_nx_box - 1,
                0,
                self.m_ny_box - 1,
                &mut self.m_brray,
                self.m_nxp_dim,
                self.m_nx_pad,
                self.m_ny_pad,
                self.m_nx_taper,
                self.m_ny_taper,
            );
            xcorr_mean_zero(
                &mut self.m_brray,
                self.m_nxp_dim,
                self.m_nx_pad,
                self.m_ny_pad,
            );
            //
            // correlate the two
            //
            todfft_c(&mut self.m_array, self.m_nx_pad, self.m_ny_pad, 0);
            todfft_c(&mut self.m_brray, self.m_nx_pad, self.m_ny_pad, 0);
            conjugate_product(
                &mut self.m_array,
                &self.m_brray,
                self.m_nx_pad,
                self.m_ny_pad,
            );
            //if (deltaCtf .ne. 0) call filterPart(array, array, nxpad, nypad, ctf, deltaCtf)
            todfft_c(&mut self.m_array, self.m_nx_pad, self.m_ny_pad, 1);
            //
            // find peak of correlation
            //
            peak_find(
                &self.m_array,
                self.m_nxp_dim,
                self.m_ny_pad,
                &mut self.m_xpeak,
                &mut self.m_ypeak,
                &mut self.m_peak,
            );
            if self.m_save_all_points {
                iobj_save = iobj + (5 * self.m_ipass - 2) * self.m_max_obj_orig;
                xtmp = b3dnint!(self.m_xnext) as f32 + self.m_xpeak;
                ytmp = b3dnint!(self.m_ynext) as f32 + self.m_ypeak;
                ind = 0;
                add_point(
                    &mut self.fm,
                    iobj_save,
                    &mut ind,
                    xtmp,
                    ytmp,
                    self.m_iz_next,
                );
            }
            calc_cg(
                &mut self.cp,
                &self.m_box_tmp,
                self.m_nx_box,
                self.m_ny_box,
                &mut self.m_xpeak,
                &mut self.m_ypeak,
                wsum,
                &mut self.m_edge_sd,
            );
            self.m_xcgsaved[(self.m_ind_pred_try - 1) as usize] = self.m_xpeak;
            self.m_ycgsaved[(self.m_ind_pred_try - 1) as usize] = self.m_ypeak;
            //call wsumForSobelPeak(boxTmp, nxBox, nyBox, xpeak, ypeak, wsum, edgeSD)
        }
        //
        // Need sobel filter peak positions on both passes
        // First get sobel sum with enough beads in it
        if self.m_max_sobel_sum > 0 {
            copy_array(
                &mut self.m_cur_sum,
                1,
                self.m_npix_box,
                &self.m_sobel_sum[iod * npix..],
                1,
            );
            nin_sobel = self.m_num_in_sobel_sum[iod];
            i = 1;
            while i <= self.tc.num_obj_do {
                if nin_sobel >= self.m_max_sobel_sum {
                    break;
                }
                if i != self.m_iobj_do {
                    ind = 0;
                    while ind < self.m_npix_box {
                        self.m_cur_sum[ind as usize] +=
                            self.m_sobel_sum[(i - 1) as usize * npix + ind as usize];
                        ind += 1;
                    }
                    nin_sobel += self.m_num_in_sobel_sum[(i - 1) as usize];
                }
                i += 1;
            }
            //
            // It seems like a model bead is better for low numbers in the average
            if nin_sobel < self.m_max_sobel_sum / 2 && self.m_nx_box == self.m_ny_box {
                make_model_bead(
                    self.m_nx_box,
                    (1.05 * self.m_diameter as f64) as f32,
                    &mut self.m_cur_sum,
                );
            }
            //
            // sobel filter the sum
            ierr = scaled_sobel(
                Some(&self.m_cur_sum),
                self.m_nx_box,
                self.m_ny_box,
                self.m_scale_fac_sobel,
                self.m_scale_by_interp,
                self.m_interp_type,
                2.,
                Some(&mut self.m_ref_sobel),
                &mut self.m_nx_sobel,
                &mut self.m_ny_sobel,
                &mut x_off_sobel,
                &mut y_off_sobel,
            );
            if ierr != 0 {
                exit_error(b"Doing Sobel filter on summed beads");
            }
            // write(*,'(i4,13i5)') (nint(refSobel(i)), i = 1, 14 * 14)
            //
            // Gaussian filter the box and sobel filter it
            if self.m_sobel_sigma > 0. {
                apply_kernel_filter(
                    &self.m_box_tmp,
                    &mut self.m_tmp_sobel,
                    self.m_nx_box,
                    self.m_nx_box,
                    self.m_ny_box,
                    &self.m_mat_kernel,
                    self.m_kernel_dim,
                );
                ierr = scaled_sobel(
                    Some(&self.m_tmp_sobel),
                    self.m_nx_box,
                    self.m_ny_box,
                    self.m_scale_fac_sobel,
                    self.m_scale_by_interp,
                    self.m_interp_type,
                    2.,
                    Some(&mut self.m_box_sobel),
                    &mut self.m_nx_sobel,
                    &mut self.m_ny_sobel,
                    &mut x_off_sobel,
                    &mut y_off_sobel,
                );
            } else {
                ierr = scaled_sobel(
                    Some(&self.m_box_tmp),
                    self.m_nx_box,
                    self.m_ny_box,
                    self.m_scale_fac_sobel,
                    self.m_scale_by_interp,
                    self.m_interp_type,
                    2.,
                    Some(&mut self.m_box_sobel),
                    &mut self.m_nx_sobel,
                    &mut self.m_ny_sobel,
                    &mut x_off_sobel,
                    &mut y_off_sobel,
                );
            }
            if ierr != 0 {
                exit_error(b"Doing Sobel filter on search box");
            }
            //
            // Taper / pad etc
            slice_taper_in_pad(
                PadIn::Float(&self.m_box_sobel),
                SLICE_MODE_FLOAT,
                self.m_nx_sobel,
                0,
                self.m_nx_sobel - 1,
                0,
                self.m_ny_sobel - 1,
                &mut self.m_sarray,
                self.m_nxs_pad + 2,
                self.m_nxs_pad,
                self.m_nys_pad,
                self.m_nx_taper,
                self.m_ny_taper,
            );
            xcorr_mean_zero(
                &mut self.m_sarray,
                self.m_nxs_pad + 2,
                self.m_nxs_pad,
                self.m_nys_pad,
            );
            slice_taper_in_pad(
                PadIn::Float(&self.m_ref_sobel),
                SLICE_MODE_FLOAT,
                self.m_nx_sobel,
                0,
                self.m_nx_sobel - 1,
                0,
                self.m_ny_sobel - 1,
                &mut self.m_sbrray,
                self.m_nxs_pad + 2,
                self.m_nxs_pad,
                self.m_nys_pad,
                self.m_nx_taper,
                self.m_ny_taper,
            );
            xcorr_mean_zero(
                &mut self.m_sbrray,
                self.m_nxs_pad + 2,
                self.m_nxs_pad,
                self.m_nys_pad,
            );
            //
            // correlate the two
            todfft_c(&mut self.m_sarray, self.m_nxs_pad, self.m_nys_pad, 0);
            todfft_c(&mut self.m_sbrray, self.m_nxs_pad, self.m_nys_pad, 0);
            conjugate_product(
                &mut self.m_sarray,
                &self.m_sbrray,
                self.m_nxs_pad,
                self.m_nys_pad,
            );
            todfft_c(&mut self.m_sarray, self.m_nxs_pad, self.m_nys_pad, 1);
            //
            // get the peaks and adjust them for the sobel scaling
            // get the centroid offset of the average and use it to adjust the positions too
            // the original offsets from scaledSobel are irrelevant to correlation
            xcorr_peak_find(
                &self.m_sarray,
                self.m_nxs_pad + 2,
                self.m_nys_pad,
                &mut self.m_sobel_xpeaks,
                &mut self.m_sobel_ypeaks,
                &mut self.m_sobel_peaks,
                self.m_max_peaks,
            );
            calc_cg(
                &mut self.cp,
                &self.m_cur_sum,
                self.m_nx_box,
                self.m_ny_box,
                &mut x_off_sobel,
                &mut y_off_sobel,
                &mut xtmp,
                &mut ytmp,
            );
            //printf("%5d%5d%s%7.2f%7.2f\n", mIobjDo, ninSobel, "  Sobel average offset", xOffSobel,
            //yOffSobel);
            i = 0;
            while i < self.m_max_peaks {
                self.m_sobel_wsums[i as usize] = -1.;
                // Fixed in translation (2026-09-26, `BUGS.md`): the source writes
                // `*= mScaleFacSobel + xOffSobel` (`beadtrack.cpp:2830-2831`),
                // multiplying by the sum; its comment says the peaks are scaled
                // and then the centroid offset is added, which is what is done
                // (as `imodfindbeads.cpp:1085` and `beadfix.cpp:2045` map a
                // Sobel-scaled coordinate: `coord * scaleFactor + offset`).
                self.m_sobel_xpeaks[i as usize] =
                    self.m_sobel_xpeaks[i as usize] * self.m_scale_fac_sobel + x_off_sobel;
                self.m_sobel_ypeaks[i as usize] =
                    self.m_sobel_ypeaks[i as usize] * self.m_scale_fac_sobel + y_off_sobel;
                i += 1;
            }
            /*for (i = 0; i < mMaxPeaks; i++) {
            if (mSobelPeaks[i] > - 1.e29)
              printf("%7.2f%7.2f%13.2f\n", mSobelXpeaks[i], mSobelYpeaks[i], mSobelPeaks[i]);
              }*/
            // Revise position of current peak on first pass
            if self.m_ipass == 1 {
                self.find_nearest_sobel_peak();
            }
        }
        //printf("%4d%9.3f%9.3f%9.3f%9.3f\n", iobj, mXnext, mYnext, mXpeak, mYpeak);

        dist = (self.m_xpeak * self.m_xpeak + self.m_ypeak * self.m_ypeak).sqrt();
        if self.m_ipass == 1 {
            if self.m_save_all_points {
                iobj_save = iobj + (5 * self.m_ipass - 1) * self.m_max_obj_orig;
                xtmp = b3dnint!(self.m_xnext) as f32 + self.m_xpeak;
                ytmp = b3dnint!(self.m_ynext) as f32 + self.m_ypeak;
                ind = 0;
                add_point(
                    &mut self.fm,
                    iobj_save,
                    &mut ind,
                    xtmp,
                    ytmp,
                    self.m_iz_next,
                );
            }
        }

        // On second pass, get wsum from previous pass or compute it
        if self.m_ipass == 2 {
            *wsum = self.m_wsum_save
                [((self.tc.iobj_seq[iod] - 1) * self.mx.max_view + self.m_iview - 1) as usize];
            if *wsum <= 0. {
                calc_cg(
                    &mut self.cp,
                    &self.m_box_tmp,
                    self.m_nx_box,
                    self.m_ny_box,
                    &mut self.m_xpeak,
                    &mut self.m_ypeak,
                    wsum,
                    &mut ytmp,
                );
            }
        }

        dist_rescue = dist > self.m_dist_crit && *wsum as f64 > 0.9 * self.m_wsum_crit[iod] as f64;
        if *wsum < self.m_wsum_crit[iod] || dist > self.m_dist_crit || self.m_ipass == 2 {
            //
            // rescue attempt; search concentric rings from the
            // center of box and take the first point that goes
            // above the relaxed wsum criterion
            //
            if self.m_ipass == 1 {
                if dist > self.m_dist_crit {
                    relax = self.m_relax_dist;
                    if self.m_if_trace != 0 {
                        printf!(
                            " Rescue-distance, sec %3d,  obj %3d, at %7.1f %7.1f, dist=%5.1f, dens=%9.0f, crit=%9.0f, min=%9.0f\n",
                            CArg::Int(self.m_iz_next as i64),
                            CArg::Int(iobj as i64),
                            CArg::Dbl(self.m_xpeak as f64),
                            CArg::Dbl(self.m_ypeak as f64),
                            CArg::Dbl(dist as f64),
                            CArg::Dbl(*wsum as f64),
                            CArg::Dbl(self.m_wsum_crit[iod] as f64),
                            CArg::Dbl(self.m_wsum_min[iod] as f64)
                        );
                    }
                    //
                    // Early in the tracking, if the bead found was above the minimum bead strength
                    // and the minimum is above the criterion, raise the crit halfway to the minimum
                    // This is a small protection against finding a weak non-bead before the real one
                    if self.fm.npt_in_obj[(iobj - 1) as usize] < 3
                        && *wsum as f64 > 1.25 * self.m_wsum_min[iod] as f64
                        && self.m_wsum_min[iod] > self.m_wsum_crit[iod]
                    {
                        relax = (0.5 * ((self.m_wsum_min[iod] / self.m_wsum_crit[iod]) as f64 + 1.))
                            as f32;
                    }
                } else {
                    relax = self.m_relax_int;
                    if self.m_if_trace != 0 {
                        printf!(
                            " Rescue-density , sec %3d,  obj %3d, at %7.1f %7.1f, dist=%5.1f, dens=%9.0f, crit=%9.0f, min=%9.0f\n",
                            CArg::Int(self.m_iz_next as i64),
                            CArg::Int(iobj as i64),
                            CArg::Dbl(self.m_xpeak as f64),
                            CArg::Dbl(self.m_ypeak as f64),
                            CArg::Dbl(dist as f64),
                            CArg::Dbl(*wsum as f64),
                            CArg::Dbl(self.m_wsum_crit[iod] as f64),
                            CArg::Dbl(self.m_wsum_min[iod] as f64)
                        );
                    }
                }
                rad_max = b3dmax!(self.m_nx_box, self.m_ny_box) as f32;
            } else {
                relax = self.m_relax_fit;
                rad_max = self.m_rad_max_fit;
            }
            // if (maxSobelSum > 0) then
            // call rescueFromSobel(boxTmp, nxBox, nyBox, xpeak, ypeak, sobelXpeaks, &
            // sobelYpeaks, sobelPeaks, sobelWsums, &
            // maxPeaks, radMax, relax * wsumCrit(iobjDo), wsum, edgeSD)
            // else
            if self.m_ipass == 2 {
                wsum_for_sobel_peak(
                    &mut self.cp,
                    &self.m_box_tmp,
                    self.m_nx_box,
                    self.m_ny_box,
                    self.m_xpeak,
                    self.m_ypeak,
                    wsum,
                    &mut self.m_edge_sd,
                );
            }
            wsum_orig = *wsum;
            xpeak_orig = self.m_xpeak;
            ypeak_orig = self.m_ypeak;
            rescue(
                &mut self.cp,
                &self.m_box_tmp,
                self.m_nx_box,
                self.m_ny_box,
                &mut self.m_xpeak,
                &mut self.m_ypeak,
                rad_max,
                relax * self.m_wsum_crit[iod],
                self.m_rescue_step_size,
                wsum,
                &mut self.m_edge_sd,
            );
            //
            // Evaluate any rescue attempt with a low density if any of the background statistics
            // test criteria are non-zero
            if (!dist_rescue
                && *wsum < self.m_wsum_crit[iod]
                && (self.m_drb_just_accept_crit > 0. || self.m_drb_just_reject_crit > 0.))
                || (self.m_drb_any_type_min_madns > 0. && *wsum > 0.)
            {
                xtmp = self.m_xpeak;
                ytmp = self.m_ypeak;
                if *wsum <= 0. {
                    xtmp = xpeak_orig;
                    ytmp = ypeak_orig;
                }
                xtmp += b3dnint!(self.m_xnext) as f32;
                ytmp += b3dnint!(self.m_ynext) as f32;
                self.get_background_wsum_stats(
                    xtmp,
                    ytmp,
                    self.m_iz_next,
                    &mut relax,
                    &mut elongation,
                );
                if self.m_bkgd_wsum_madn > 0. {
                    //
                    // Report stats if got any
                    wsum_bkgd_ratio = (relax - self.m_bkgd_wsum_median) / self.m_bkgd_wsum_madn;
                    if self.m_if_trace != 0 {
                        printf!(
                            "BKGD-DENS:%4d %8.2f %8.2f %7.3f %7.3f\n",
                            CArg::Int(iobj as i64),
                            CArg::Dbl(relax as f64),
                            CArg::Dbl(self.m_bkgd_wsum_median as f64),
                            CArg::Dbl(wsum_bkgd_ratio as f64),
                            CArg::Dbl(elongation as f64)
                        );
                    }
                    //
                    // Initial test for just plain too low above background
                    if *wsum > 0.
                        && self.m_drb_any_type_min_madns > 0.
                        && wsum_bkgd_ratio < self.m_drb_any_type_min_madns
                    {
                        if self.m_if_trace != 0 {
                            printf!("TOO LOW FOR ANY RESCUE\n");
                        }
                        *wsum = 0.;
                    } else if !dist_rescue && *wsum < self.m_wsum_crit[iod] {
                        //
                        // See if the background ratio is within the window for further analysis
                        // based on either of the criteria
                        if (*wsum > 0.
                            && wsum_bkgd_ratio < self.m_drb_just_accept_crit
                            && self.m_drb_just_accept_crit > 0.)
                            || (*wsum <= 0.
                                && self.m_drb_just_reject_crit > 0.
                                && wsum_bkgd_ratio > self.m_drb_just_reject_crit)
                        {
                            //
                            // Interpolate the criterion for rejection based on background max between
                            // the two limits
                            reject_max_crit = (0.5
                                * (self.m_drb_reject_max_low_madns
                                    + self.m_drb_reject_max_high_madns)
                                    as f64) as f32;
                            if self.m_drb_just_accept_crit > self.m_drb_any_type_min_madns {
                                reject_max_crit = (wsum_bkgd_ratio - self.m_drb_any_type_min_madns)
                                    * (self.m_drb_reject_max_low_madns
                                        - self.m_drb_reject_max_high_madns)
                                    / (self.m_drb_just_accept_crit - self.m_drb_any_type_min_madns)
                                    + self.m_drb_reject_max_high_madns;
                            }
                            num_maxes = 0;
                            //
                            // Loop over neighbors and get or use the stored background max
                            let seq_off = ((self.m_last_seq - 1) * self.m_max_neigh) as usize;
                            jobj_do = 1;
                            while jobj_do <= self.m_num_wneighbors[(self.m_last_seq - 1) as usize] {
                                jobj = self.m_neighbors_for_wfits[seq_off + (jobj_do - 1) as usize];
                                itry = b3dmax!(
                                    self.m_min_view_do,
                                    self.m_iview - self.m_max_drb_delta_z
                                );
                                while itry
                                    < b3dmin!(
                                        self.m_max_view_do,
                                        self.m_iview + self.m_max_drb_delta_z
                                    )
                                {
                                    let bind =
                                        ((jobj - 1) * self.m_max_view_do + itry - 1) as usize;
                                    if self.m_wsum_save
                                        [((jobj - 1) * self.mx.max_view + itry - 1) as usize]
                                        >= 0.
                                    {
                                        if self.m_bkgd_wmax_save[bind] < 0. {
                                            ip = 1;
                                            while ip <= self.fm.npt_in_obj[(jobj - 1) as usize] {
                                                ipt = self.fm.object[(self.fm.ibase_obj
                                                    [(jobj - 1) as usize]
                                                    + ip
                                                    - 1)
                                                    as usize];
                                                let pt = self.fm.p_coord[(ipt - 1) as usize];
                                                if b3dnint!(pt[2]) == itry - 1 {
                                                    self.get_background_wsum_stats(
                                                        pt[0],
                                                        pt[1],
                                                        itry - 1,
                                                        &mut relax,
                                                        &mut elongation,
                                                    );
                                                    if self.m_bkgd_wsum_max > 0. {
                                                        self.m_bkgd_wmax_save[bind] =
                                                            self.m_bkgd_wsum_max;
                                                    }
                                                    break;
                                                }
                                                ip += 1;
                                            }
                                        }
                                        if self.m_bkgd_wmax_save[bind] >= 0. {
                                            num_maxes += 1;
                                            self.m_bkgd_neigh_wmax[(num_maxes - 1) as usize] =
                                                self.m_bkgd_wmax_save[bind];
                                        }
                                    }
                                    itry += 1;
                                }
                                jobj_do += 1;
                            }
                            //
                            // If enough maxes found, get the median/MADN
                            if num_maxes >= 10 {
                                rs_fast_median_in_place(
                                    &mut self.m_bkgd_neigh_wmax,
                                    num_maxes,
                                    &mut wmax_median,
                                );
                                rs_fast_madn(
                                    &self.m_bkgd_neigh_wmax,
                                    num_maxes,
                                    wmax_median,
                                    &mut self.m_stat_tmp,
                                    &mut wmax_madn,
                                );
                                if self.m_if_trace != 0 {
                                    printf!(
                                        "NEIGH-BK: %3d %8.2f %8.2f %7.3f crit %6.2f\n",
                                        CArg::Int(num_maxes as i64),
                                        CArg::Dbl(wmax_median as f64),
                                        CArg::Dbl(wmax_madn as f64),
                                        CArg::Dbl(((wsum_orig - wmax_median) / wmax_madn) as f64),
                                        CArg::Dbl(reject_max_crit as f64)
                                    );
                                }
                                if wmax_madn > 0. {
                                    //
                                    // Test against criteria for accepting anyway
                                    if *wsum <= 0.
                                        && self.m_drb_just_reject_crit > 0.
                                        && (wsum_orig - wmax_median) / wmax_madn
                                            > self.m_drb_accept_min_madns
                                    {
                                        *wsum = wsum_orig;
                                        self.m_xpeak = xpeak_orig;
                                        self.m_ypeak = ypeak_orig;
                                        if self.m_if_trace != 0 {
                                            printf!(
                                                "ACCEPT ANYWAY %.1f\n",
                                                CArg::Dbl(wsum_orig as f64)
                                            );
                                        }
                                        //
                                        // And against criterion for rejecting it
                                    } else if *wsum > 0.
                                        && self.m_drb_just_accept_crit > 0.
                                        && (*wsum - wmax_median) / wmax_madn < reject_max_crit
                                    {
                                        if self.m_if_trace != 0 {
                                            printf!(
                                                "REJECT IT %.1f\n",
                                                CArg::Dbl(wsum_orig as f64)
                                            );
                                        }
                                        *wsum = 0.;
                                    }
                                }
                            }
                        } else if self.m_if_trace != 0 {
                            printf!("JUST GO WITH RESCUE %.1f\n", CArg::Dbl(*wsum as f64));
                        }
                    }
                }
            }
            self.m_xcgsaved[(self.m_ind_pred_try - 1) as usize] = self.m_xpeak;
            self.m_ycgsaved[(self.m_ind_pred_try - 1) as usize] = self.m_ypeak;
            if self.m_max_sobel_sum > 0 {
                self.find_nearest_sobel_peak();
            }
        }
        fflush_stdout!();
        // Args assigned to: wsum
    }

    /// Original: `BeadTrack::addBestBeadFound` (`beadtrack.cpp:3028`).
    ///
    /// addBestBeadFound - adds point if it passes wsum criterion
    pub fn add_best_bead_found(&mut self, iobj: i32, wsum: f32, save_boxes: bool) {
        let iobj_save: i32;
        let mut tmean: f32 = 0.;
        let mut tmin: f32 = 0.;
        let ypos: f32;
        let mut tmax: f32 = 0.;
        let xpos: f32;
        let mut ip_near: i32;
        if wsum <= 0. {
            return;
        }
        let iod = (self.m_iobj_do - 1) as usize;
        self.m_wsum_save
            [((self.tc.iobj_seq[iod] - 1) * self.mx.max_view + self.m_iview - 1) as usize] = wsum;
        if self.cp.get_edge_sd != 0 {
            let eind =
                ((self.tc.iobj_seq[iod] - 1) * self.m_max_view_do + self.m_iview - 1) as usize;
            self.m_edge_sd_save[eind] = self.m_edge_sd;
            calc_elongation(
                &mut self.cp,
                &self.m_box_tmp,
                self.m_nx_box,
                self.m_ny_box,
                self.m_xpeak,
                self.m_ypeak,
                &mut self.m_elong_save[eind],
            );
            if self.m_outer_sigma > 0. {
                calc_outer_mad(
                    &mut self.cp,
                    &self.m_box_tmp,
                    self.m_nx_box,
                    self.m_ny_box,
                    self.m_xpeak,
                    self.m_ypeak,
                    &mut self.m_outer_madsave[eind],
                    &mut self.m_outer_background[eind],
                );
            }
        }
        xpos = b3dnint!(self.m_xnext) as f32 + self.m_xpeak;
        ypos = b3dnint!(self.m_ynext) as f32 + self.m_ypeak;
        if self.m_if_trace != 0 {
            printf!(
                "add %4d %4d %4d %6.1f %6.1f %6.1f %6.1f %11.0f %11.0f %11.4f\n",
                CArg::Int(self.m_nz_out as i64),
                CArg::Int(self.m_iz_next as i64),
                CArg::Int(iobj as i64),
                CArg::Dbl(self.m_xnext as f64),
                CArg::Dbl(self.m_ynext as f64),
                CArg::Dbl(xpos as f64),
                CArg::Dbl(ypos as f64),
                CArg::Dbl(self.m_peak as f64),
                CArg::Dbl(wsum as f64),
                CArg::Dbl(self.m_edge_sd as f64)
            );
            fflush_stdout!();
        }
        //
        // add point to model
        //
        add_point(
            &mut self.fm,
            iobj,
            &mut self.m_ip_nearest[iod],
            xpos,
            ypos,
            self.m_iz_next,
        );
        if self.m_max_sobel_sum > 0 {
            let np = self.fm.n_point as usize;
            let ipt = (self.m_ind_pred_try - 1) as usize;
            self.m_saved_cgcoord[np * 2 - 2] = b3dnint!(self.m_xnext) as f32 + self.m_xcgsaved[ipt];
            self.m_saved_cgcoord[np * 2 - 1] = b3dnint!(self.m_ynext) as f32 + self.m_ycgsaved[ipt];
            self.m_iflag_cgvs_sobel[np - 1] = 1;
        }
        if self.m_save_all_points {
            iobj_save = iobj + (5 * self.m_ipass) * self.m_max_obj_orig;
            ip_near = 0;
            add_point(
                &mut self.fm,
                iobj_save,
                &mut ip_near,
                xpos,
                ypos,
                self.m_iz_next,
            );
        }
        self.m_num_added += 1;
        self.m_iobj_del[(self.m_num_added - 1) as usize] = iobj;
        //
        // mark as found, add to data matrix for alignment
        //
        self.m_if_found[iod] = 1;
        let row = (self.m_num_data * self.m_xmat_size) as usize;
        let ipt = (self.m_ind_pred_try - 1) as usize;
        self.m_xmat[row] = self.m_alt_xseek[ipt] - self.tc.xcen;
        self.m_xmat[row + 1] = self.m_alt_yseek[ipt] - self.tc.ycen;
        self.m_xmat[row + 3] = xpos - self.tc.xcen;
        self.m_xmat[row + 4] = ypos - self.tc.ycen;
        self.m_xmat[row + 5] = self.m_iobj_do as f32;
        self.m_num_data += 1;
        //
        if save_boxes {
            let nx_box = self.m_nx_box;
            let ny_box = self.m_ny_box;
            if self.m_mode_box == 2 {
                array_min_max_mean(
                    &self.m_box_tmp,
                    nx_box,
                    ny_box,
                    0,
                    nx_box - 1,
                    0,
                    ny_box - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            } else {
                scale_array_for_mode(
                    &mut self.m_box_tmp,
                    nx_box,
                    self.m_mode_box,
                    0,
                    nx_box - 1,
                    0,
                    ny_box - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            }
            unsafe { iiu_write_section(2, self.m_box_tmp.as_mut_ptr().cast()) };
            self.m_box_min = b3dmin!(self.m_box_min, tmin);
            self.m_box_max = b3dmax!(self.m_box_max, tmax);
            self.m_box_sum += tmean;
            self.m_nz_out += 1;

            copy_array(
                &mut self.m_box_tmp,
                1,
                self.m_npix_box,
                &self.m_corr_sum[iod * self.m_npix_box as usize..],
                1,
            );
            if self.m_mode_box == 2 {
                array_min_max_mean(
                    &self.m_box_tmp,
                    nx_box,
                    ny_box,
                    0,
                    nx_box - 1,
                    0,
                    ny_box - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            } else {
                scale_array_for_mode(
                    &mut self.m_box_tmp,
                    nx_box,
                    self.m_mode_box,
                    0,
                    nx_box - 1,
                    0,
                    ny_box - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            }
            unsafe { iiu_write_section(3, self.m_box_tmp.as_mut_ptr().cast()) };
            self.m_ref_min = b3dmin!(self.m_ref_min, tmin);
            self.m_ref_max = b3dmax!(self.m_ref_max, tmax);
            self.m_ref_sum += tmean;
            split_pack(
                &self.m_array,
                self.m_nxp_dim,
                self.m_nx_pad,
                self.m_ny_pad,
                &mut self.m_box_tmp,
            );
            if self.m_mode_box == 2 {
                array_min_max_mean(
                    &self.m_box_tmp,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    0,
                    self.m_nx_pad - 1,
                    0,
                    self.m_ny_pad - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            } else {
                scale_array_for_mode(
                    &mut self.m_box_tmp,
                    self.m_nx_pad,
                    self.m_mode_box,
                    0,
                    self.m_nx_pad - 1,
                    0,
                    self.m_ny_pad - 1,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            }
            unsafe { iiu_write_section(4, self.m_box_tmp.as_mut_ptr().cast()) };
            self.m_corr_min = b3dmin!(self.m_corr_min, tmin);
            self.m_corr_max = b3dmax!(self.m_corr_max, tmax);
            self.m_corro_sum += tmean;
            if let Some(fp) = self.m_brpl_fp.as_mut() {
                fprintf!(
                    fp,
                    "%6d %6d %6d\n",
                    CArg::Int((b3dnint!(self.m_xnext) - self.m_nx_box / 2) as i64),
                    CArg::Int((b3dnint!(self.m_ynext) - self.m_ny_box / 2) as i64),
                    CArg::Int(self.m_iz_next as i64)
                );
            }
            if let Some(fp) = self.m_cpl_fp.as_mut() {
                fprintf!(
                    fp,
                    "%6d %6d %6d\n",
                    CArg::Int((b3dnint!(self.m_xnext) - self.m_nx_pad / 2) as i64),
                    CArg::Int((b3dnint!(self.m_ynext) - self.m_ny_pad / 2) as i64),
                    CArg::Int(self.m_iz_next as i64)
                );
            }
        }
    }

    /// Original: `BeadTrack::loadBoxAndTaper` (`beadtrack.cpp:3128`).
    ///
    /// loadBoxAndTaper - takes care of loading, filling, tapering, and filtering a box
    /// of the given size.  `mLoadTmp` is a member array in every call; the caller
    /// lends it out of the struct for the call (see the module docs).
    pub fn load_box_and_taper(
        &mut self,
        m_load_tmp: &mut [f32],
        nx_load: i32,
        ny_load: i32,
        nx_filt: i32,
        ny_filt: i32,
        ctf_use: &[f32],
        delta_use: f32,
    ) {
        let mut ix: i32;
        let iy: i32;
        let nx_trunc: i32;
        let ny_trunc: i32;
        // `mIfLast` and `fillVal` are uninitialised in the source when the box
        // needs a fill but no taper, or the trial taper fails.  Fixed in
        // translation (2026-09-26, `BUGS.md`): 0 here, which makes
        // `shiftAndFillBox` fill with the edge mean.
        let mut m_if_last: i32 = 0;
        let mut fill_val: f32 = 0.;

        unsafe { iiu_set_position(1, self.m_ipiece_z, 0) };
        //
        // For a box that needs filling, get its truncated size
        if self.m_need_fill {
            nx_trunc = self.m_ix1 + 1 - self.m_ix0;
            ny_trunc = self.m_iy1 + 1 - self.m_iy0;
            ix = unsafe {
                iiu_read_sec_part(
                    1,
                    m_load_tmp.as_mut_ptr().cast(),
                    nx_trunc,
                    self.m_ix0,
                    self.m_ix1,
                    self.m_iy0,
                    self.m_iy1,
                )
            };
            //
            // If it needs tapering, do a trial taper to get the fill value and reload the box
            if self.m_need_taper {
                if taper_at_fill(m_load_tmp, nx_trunc, ny_trunc, self.m_nfill_taper, true) == 0 {
                    m_if_last = get_last_taper_fill_value(&mut fill_val) as i32;
                }
                unsafe { iiu_set_position(1, self.m_ipiece_z, 0) };
                ix = unsafe {
                    iiu_read_sec_part(
                        1,
                        m_load_tmp.as_mut_ptr().cast(),
                        nx_trunc,
                        self.m_ix0,
                        self.m_ix1,
                        self.m_iy0,
                        self.m_iy1,
                    )
                };
            }
            if ix == 0 {
                self.shift_and_fill_box(
                    m_load_tmp,
                    nx_trunc,
                    ny_trunc,
                    nx_load,
                    ny_load,
                    m_if_last,
                    &mut fill_val,
                );
            }
        } else {
            //
            // otherwise get the full box
            ix = unsafe {
                iiu_read_sec_part(
                    1,
                    m_load_tmp.as_mut_ptr().cast(),
                    nx_load,
                    self.m_ix0,
                    self.m_ix1,
                    self.m_iy0,
                    self.m_iy1,
                )
            };
        }
        if ix != 0 {
            printf!(
                "X/Y dim %d %d Limits of load %d %d %d %d\n",
                CArg::Int(nx_load as i64),
                CArg::Int(ny_load as i64),
                CArg::Int(self.m_ix0 as i64),
                CArg::Int(self.m_ix1 as i64),
                CArg::Int(self.m_iy0 as i64),
                CArg::Int(self.m_iy1 as i64)
            );
            exit_error(b"Reading image file");
        }
        //
        // Taper (for real)
        if self.m_need_taper {
            if taper_at_fill(m_load_tmp, nx_load, ny_load, self.m_nfill_taper, true) != 0 {
                exit_error(b"Getting memory for tapering from fill in box");
            }
        }
        //
        // Filter if delta set
        if self.m_delta_ctf != 0. {
            slice_taper_out_pad(
                PadIn::Float(m_load_tmp),
                SLICE_MODE_FLOAT,
                nx_load,
                ny_load,
                &mut self.m_array,
                nx_filt + 2,
                nx_filt,
                ny_filt,
                0,
                0.,
            );
            todfft_c(&mut self.m_array, nx_filt, ny_filt, 0);
            xcorr_filter_part(
                FilterIn::InPlace,
                &mut self.m_array,
                nx_filt,
                ny_filt,
                ctf_use,
                delta_use,
            );
            todfft_c(&mut self.m_array, nx_filt, ny_filt, 1);
            ix = (nx_filt - nx_load) / 2;
            iy = (ny_filt - ny_load) / 2;
            repack_float_image(
                f32_bytes_mut!(*m_load_tmp),
                f32_bytes!(self.m_array),
                nx_filt + 2,
                ix,
                ix + nx_load - 1,
                iy,
                iy + ny_load - 1,
            );
        }
    }

    /// Original: `BeadTrack::shiftAndFillBox` (`beadtrack.cpp:3188`).
    ///
    /// shiftAndFillBox - shifts a partially loaded box on edge of data to proper position and
    /// fills the rest with the proper fill value to match.  The source's only call
    /// passes the same array as `boxLoaded` and `boxPadded`, and the copy loops
    /// run backwards so the in-place shift is safe; `box_padded` is that array.
    #[allow(clippy::too_many_arguments)]
    pub fn shift_and_fill_box(
        &self,
        box_padded: &mut [f32],
        nx_trunc: i32,
        ny_trunc: i32,
        nx_full: i32,
        ny_full: i32,
        m_if_last: i32,
        fill_val: &mut f32,
    ) {
        let ind_start: i32;
        let ind_end: i32;
        let ind_edge_start: i32;
        let ind_edge_end: i32;
        let idel_ind: i32;
        let mut ix: i32;
        let mut iy: i32;
        if m_if_last == 0 {
            *fill_val = 0.;
        }
        if nx_trunc < nx_full {
            // If it is at right edge of image, set up to copy without shift, fill the right
            // side, and take mean from right edge
            if self.m_ix0 > 0 {
                ind_start = nx_trunc + 1;
                ind_end = nx_full;
                ind_edge_end = nx_trunc;
                ind_edge_start = b3dmax!(1, nx_trunc - 3);
                idel_ind = 0;
            } else {
                //
                // If it is at left edge, copy with a shift, fill the left side, and take mean from
                // left edge AFTER the shift
                ind_start = 1;
                ind_end = nx_full - nx_trunc;
                ind_edge_start = ind_end + 1;
                ind_edge_end = b3dmin!(nx_full, ind_end + 4);
                idel_ind = ind_end;
            }
            //
            // Copy the box
            iy = 0;
            while iy < ny_full {
                ix = nx_trunc - 1;
                while ix >= 0 {
                    box_padded[(iy * nx_full + ix + idel_ind) as usize] =
                        box_padded[(iy * nx_trunc + ix) as usize];
                    ix -= 1;
                }
                iy += 1;
            }
            //
            // Get edge mean if necessary
            if m_if_last == 0 {
                iy = 0;
                while iy < ny_full {
                    ix = ind_edge_start;
                    while ix <= ind_edge_end {
                        *fill_val += box_padded[(iy * nx_full + ix - 1) as usize];
                        ix += 1;
                    }
                    iy += 1;
                }
                *fill_val /= (ny_full * (ind_edge_end + 1 - ind_edge_start)) as f32;
            }
            //
            // Fill past the edge
            iy = 0;
            while iy < ny_full {
                ix = ind_start;
                while ix <= ind_end {
                    box_padded[(iy * nx_full + ix - 1) as usize] = *fill_val;
                    ix += 1;
                }
                iy += 1;
            }
        } else {
            //
            // Same operations for missing on top or bottom
            if self.m_iy0 > 0 {
                ind_start = ny_trunc + 1;
                ind_end = ny_full;
                ind_edge_end = ny_trunc;
                ind_edge_start = b3dmax!(1, ny_trunc - 3);
                idel_ind = 0;
            } else {
                ind_start = 1;
                ind_end = ny_full - ny_trunc;
                ind_edge_start = ind_end + 1;
                ind_edge_end = b3dmin!(nx_full, ind_end + 4);
                idel_ind = ind_end;
            }
            iy = ny_trunc - 1;
            while iy >= 0 {
                ix = 0;
                while ix < nx_full {
                    box_padded[(iy * nx_full + ix + idel_ind) as usize] =
                        box_padded[(iy * nx_trunc + ix) as usize];
                    ix += 1;
                }
                iy -= 1;
            }
            if m_if_last == 0 {
                iy = ind_edge_start;
                while iy <= ind_edge_end {
                    ix = 0;
                    while ix < nx_full {
                        *fill_val += box_padded[((iy - 1) * nx_full + ix) as usize];
                        ix += 1;
                    }
                    iy += 1;
                }
                *fill_val /= (nx_full * (ind_edge_end + 1 - ind_edge_start)) as f32;
            }
            iy = ind_start;
            while iy <= ind_end {
                ix = 0;
                while ix < nx_full {
                    box_padded[((iy - 1) * nx_full + ix) as usize] = *fill_val;
                    ix += 1;
                }
                iy += 1;
            }
        }
    }

    /// Original: `BeadTrack::findNearestSobelPeak` (`beadtrack.cpp:3280`).
    ///
    /// Find the sobel peak nearest to the current position (xpeak, ypeak) that is
    /// valid and still reasonably strong, and substitute for xpeak, ypeak
    pub fn find_nearest_sobel_peak(&mut self) {
        let mut ibest: i32;
        let mut i: i32;
        let mut dist_min: f32;
        let mut peak_max: f32;
        let xpcen: f32;
        let ypcen: f32;
        let mut dist: f32;
        //
        // Tried to moved reference position toward center of expected position (0, 0) by up to
        // half a bead diameter - it was sometimes signinficantly worse
        xpcen = self.m_xpeak;
        ypcen = self.m_ypeak;
        ibest = 0;
        peak_max = -1.0e30;
        dist_min = 1.0e20;
        i = 1;
        while i <= self.m_max_peaks {
            check_sobel_peak(
                &mut self.cp,
                i,
                &self.m_box_tmp,
                self.m_nx_box,
                self.m_ny_box,
                &self.m_sobel_xpeaks,
                &self.m_sobel_ypeaks,
                &self.m_sobel_peaks,
                &mut self.m_sobel_wsums,
                &mut self.m_sobel_edge_sd,
                self.m_max_peaks,
            );
            //if (sobelPeaks(i) > - 1.e29) write(*,'(2f7.2,2f12.2)')  &
            //    sobelXpeaks(i), sobelYpeaks(i), sobelPeaks(i), &
            //    sobelWsums(i)

            let iu = (i - 1) as usize;
            if self.m_sobel_wsums[iu] > 0. {
                if (peak_max as f64) < -1.0e29 {
                    peak_max = self.m_sobel_peaks[iu];
                }
                if self.m_sobel_peaks[iu] < self.m_min_peak_ratio * peak_max {
                    break;
                }
                let dx = self.m_sobel_xpeaks[iu] - xpcen;
                let dy = self.m_sobel_ypeaks[iu] - ypcen;
                dist = dx * dx + dy * dy;

                // Do not allow it more than a diameter away or a lot weaker than the criterion
                if dist < dist_min
                    && dist < self.m_diameter * self.m_diameter
                    && self.m_sobel_wsums[iu] as f64
                        > 0.5 * self.m_wsum_crit[(self.m_iobj_do - 1) as usize] as f64
                {
                    dist_min = dist;
                    ibest = i;
                }
            }
            i += 1;
        }
        if ibest == 0 {
            return;
        }

        // On second pass, only accept a position if it is closer to the expected position (0)
        let ib = (ibest - 1) as usize;
        if self.m_ipass == 2
            && self.m_sobel_xpeaks[ib] * self.m_sobel_xpeaks[ib]
                + self.m_sobel_ypeaks[ib] * self.m_sobel_ypeaks[ib]
                > self.m_xpeak * self.m_xpeak + self.m_ypeak * self.m_ypeak
        {
            return;
        }
        self.m_xpeak = self.m_sobel_xpeaks[ib];
        self.m_ypeak = self.m_sobel_ypeaks[ib];
        // wsum = sobelWsums(ibest)
        // print *,ibest, xpeak, ypeak, wsum
    }
}

impl BeadTrack {
    /// Original: `BeadTrack::redoFitsEvaluateResiduals` (`beadtrack.cpp:3331`).
    pub fn redo_fits_evaluate_residuals(&mut self, model_file: &str) {
        let mut num_same_side: i32;
        let mut num_other_side: i32;
        let mut near_bidir: i32;
        let mut num_elong: i32;
        let mut raise_crit: f32;
        let mut elong_avg: f32 = 0.;
        let mut elong_sd: f32 = 0.;
        let mut elong_max: f32;
        let mut res_avg: f32;
        let mut res_sd: f32;
        let mut res_sem: f32 = 0.;
        let mut wsum: f32 = 0.;
        let mut cur_diff: f32;
        let mut dev_avg: f32 = 0.;
        let mut dev_max: f32 = 0.;
        let mut dev_sd: f32 = 0.;
        let mut elongation: f32 = 0.;
        let mut err_max: f32;
        let mut i: i32;
        let mut ibase: i32;
        let mut ibox: i32;
        let mut if_mean_bad: i32;
        let mut if_ro_trans: i32;
        let mut if_trans: i32;
        let mut indr: i32;
        let mut iobj: i32;
        let mut ip: i32;
        let mut ipnt_max_dev: i32 = 0;
        let mut ipt: i32;
        let mut is_cur: i32;
        let mut iv_look: i32;
        let mut ix: i32;
        let mut iy: i32;
        let mut iz: i32;
        let mut max_drop: i32;
        let mut nprev: i32;
        let num_add_tmp: i32;
        let mut num_drop: i32 = 0;
        if self.m_num_data >= self.m_lim_pts_shift {
            if_ro_trans = 0;
            if_trans = 1;
            if self.m_num_data >= self.m_lim_pts_stretch {
                if_trans = 0;
            } else if self.m_num_data >= self.m_lim_pts_mag {
                if_ro_trans = 2;
            } else if self.m_num_data >= self.m_lim_pts_rot {
                if_ro_trans = 1;
            }
            max_drop = b3dnint!(0.26 * self.m_num_data as f64);
            if self.m_num_data < 4 || self.m_if_did_align == 0 {
                max_drop = 0;
            }
            findxf_wo_outliers(
                &mut self.m_xmat,
                self.m_xmat_size,
                self.m_num_data,
                self.tc.xcen,
                self.tc.ycen,
                if_trans,
                if_ro_trans,
                max_drop,
                self.m_outlie_crit,
                self.m_outlie_crit_abs,
                self.m_outlie_elim_min,
                &mut self.m_idrop,
                &mut num_drop,
                &mut self.m_xform,
                &mut dev_avg,
                &mut dev_sd,
                &mut dev_max,
                &mut ipnt_max_dev,
            );
            printf!(
                "view %3d, pass %d,%4d points (- %2d), mean, sd, max dev: %7.2f %7.2f %7.2f\n",
                CArg::Int(self.m_iview as i64),
                CArg::Int(self.m_ipass as i64),
                CArg::Int(self.m_num_data as i64),
                CArg::Int(num_drop as i64),
                CArg::Dbl(dev_avg as f64),
                CArg::Dbl(dev_sd as f64),
                CArg::Dbl(dev_max as f64)
            );
            fflush_stdout!();
            if self.m_if_did_align != 0 {
                self.m_dx_cur += self.m_xform[4];
                self.m_dy_cur += self.m_xform[5];
            }
        }
        //
        // do tiltAli if anything changed, so can get current residuals
        //
        let res_off = ((self.m_iview_seq - 1) * self.m_max_all_real) as usize;
        if self.m_num_added > 0 {
            if self.m_if_did_align != 0 && self.m_iv_use != self.m_iview {
                self.tc.dxy_save[(2 * (self.m_iview - 1)) as usize] = self.m_dx_cur;
                self.tc.dxy_save[(2 * (self.m_iview - 1) + 1) as usize] = self.m_dy_cur;
            }
            tilt_ali(
                &mut self.tc,
                &mut self.av,
                &self.mx,
                &mut self.m_eval_funct,
                &mut self.sg,
                &self.fm,
                &mut self.m_if_did_align,
                &mut self.m_if_align_done,
                &mut self.m_res_mean[res_off..],
                self.m_iview,
                &mut self.m_ali_mean_res,
            );
            if self.m_if_did_align != 0 {
                self.m_ivs_on_align = self.m_iview_seq;
            }
        }
        if number_in_list(
            self.m_iview,
            list_or_null(&self.m_iv_snap_list),
            self.m_nsnap_list,
            0,
        ) != 0
        {
            let cbuffer = c_format_bytes(
                "%s.%d.%d",
                &[
                    CArg::Str(model_file),
                    CArg::Int(self.m_iview as i64),
                    CArg::Int(self.m_ipass as i64),
                ],
            );
            let _ = scale_fort_model(&mut self.fm, 1);
            let _ = write_fort_model(&String::from_utf8_lossy(&cbuffer), &mut self.fm);
            let _ = scale_fort_model(&mut self.fm, 0);
        }
        num_add_tmp = self.m_num_added;
        self.m_num_added = 0;
        self.m_num_to_do = 0;
        self.m_num_del = 0;
        raise_crit = 1.;
        cur_diff = 0.;
        res_avg = 0.;
        res_sd = 1.;
        let max_all_real = self.m_max_all_real;
        if num_add_tmp > 0 {
            self.m_num_to_do = 0;
            self.m_iobj_do = 1;
            while self.m_iobj_do <= self.tc.num_obj_do {
                let iod = (self.m_iobj_do - 1) as usize;
                if self.m_if_found[iod] == 0 {
                    self.m_num_to_do += 1;
                }
                if self.m_if_found[iod] == 1 {
                    iobj = self.tc.iobj_seq[iod];
                    err_max = 0.;
                    // if (ipass == 1) then
                    indr = 0;
                    //
                    // find point in the tiltalign, get its residual
                    //
                    if self.m_if_did_align == 1 {
                        ipt = 1;
                        while ipt <= self.av.nreal_pt {
                            if iobj == self.tc.iobj_ali[(ipt - 1) as usize] {
                                indr = ipt;
                            }
                            ipt += 1;
                        }
                    }
                    if indr != 0 {
                        iv_look = self.av.map_file_to_view[(self.m_iview - 1) as usize];
                        ipt = self.av.ireal_str[(indr - 1) as usize];
                        while ipt < self.av.ireal_str[indr as usize] && err_max == 0. {
                            if self.av.isec_view[(ipt - 1) as usize] == iv_look {
                                let xr = self.av.xresid[(ipt - 1) as usize];
                                let yr = self.av.yresid[(ipt - 1) as usize];
                                err_max = self.tc.scale_xy * (xr * xr + yr * yr).sqrt();
                            }
                            ipt += 1;
                        }
                    }
                    //
                    // When no tiltalign available, just redo the point with
                    // highest deviation?  No, projections could be lousy.
                    //
                    // if (ifDidAlign == 0 .and. numData >= limPtsShift) then
                    // do i = 1, numData
                    // if (nint(xr(6, i)) == iobjDo .and. xr(13, i) > &
                    // 0.99 * devMax) errMax = xr(13, i)
                    // enddo
                    // endif
                    //
                    // endif
                    if_mean_bad = 0;
                    //
                    // see if mean residual has gotten a lot bigger
                    //
                    if self.m_if_did_align == 1 {
                        is_cur = self.m_iview_seq - 1;
                        nprev = 0;

                        while nprev <= self.max_resid && is_cur > 0 {
                            let r =
                                self.m_res_mean[((is_cur - 1) * max_all_real + iobj - 1) as usize];
                            if r > 0. {
                                nprev += 1;
                                self.m_prev_res[(nprev - 1) as usize] = r;
                            }
                            is_cur -= 1;
                        }
                        if nprev > self.m_min_resid {
                            raise_crit = 1.;
                            //
                            // Possible relaxation of criteria when near bidirectional reversal based
                            // on number of views on each side; not used by default
                            near_bidir = self.m_iview - self.m_iv_bidir_part2;
                            if self.m_iview < self.m_iv_bidir_part2 {
                                near_bidir = (self.m_iv_bidir_part2 - 1) - self.m_iview;
                            }
                            if near_bidir < self.m_num_bidir_relax_crit {
                                num_same_side = 0;
                                num_other_side = 0;
                                ipt = 1;
                                while ipt <= self.fm.npt_in_obj[(iobj - 1) as usize] {
                                    ix = self.fm.object[(self.fm.ibase_obj[(iobj - 1) as usize]
                                        + ipt
                                        - 1)
                                        as usize];
                                    iy = b3dnint!(self.fm.p_coord[(ix - 1) as usize][2]) + 1;
                                    if (iy < self.m_iv_bidir_part2
                                        && self.m_iview < self.m_iv_bidir_part2)
                                        || (iy >= self.m_iv_bidir_part2
                                            && self.m_iview >= self.m_iv_bidir_part2)
                                    {
                                        num_same_side += 1;
                                    } else {
                                        num_other_side += 1;
                                    }
                                    ipt += 1;
                                }
                                if num_other_side >= 3 * num_same_side {
                                    raise_crit = (self.m_relax_bidir_fac
                                        * (self.m_num_bidir_relax_crit - near_bidir) as f32)
                                        / self.m_num_bidir_relax_crit as f32;
                                }
                            }
                            //
                            // Evaluate increase in mean residual
                            cur_diff = self.m_res_mean
                                [((self.m_iview_seq - 1) * max_all_real + iobj - 1) as usize]
                                - self.m_prev_res[0];
                            i = 1;
                            while i <= nprev - 1 {
                                self.m_prev_res[(i - 1) as usize] -= self.m_prev_res[i as usize];
                                i += 1;
                            }
                            avg_sd(
                                &self.m_prev_res,
                                nprev - 1,
                                &mut res_avg,
                                &mut res_sd,
                                &mut res_sem,
                            );
                            if cur_diff > self.m_res_diff_min * raise_crit
                                && (cur_diff as f64 - b3dmax!(0., res_avg as f64)) / res_sd as f64
                                    > (self.m_res_diff_crit * raise_crit) as f64
                                && err_max > self.m_cur_res_min * raise_crit
                            {
                                if_mean_bad = 1;
                            }
                        }
                    }
                    //
                    // On second pass, if it is still a bad mean residual, first analyze for
                    // whether it is clearly a bead above background
                    if self.m_ipass == 2 && if_mean_bad == 1 && self.m_bmr_upper_elong_lim > 0. {
                        ip = self.fm.object[(self.fm.ibase_obj[(iobj - 1) as usize]
                            + self.m_ip_nearest[iod]
                            - 1) as usize];
                        let pt = self.fm.p_coord[(ip - 1) as usize];
                        self.get_background_wsum_stats(
                            pt[0],
                            pt[1],
                            self.m_iz_next,
                            &mut wsum,
                            &mut elongation,
                        );
                        if self.m_bkgd_wsum_median != 0.
                            && self.m_bkgd_wsum_madn > 0.
                            && self.m_if_trace != 0
                        {
                            printf!(
                                "KGD-ELONG: %3d %8.2f %8.2f %7.3f %7.3f\n",
                                CArg::Int(iobj as i64),
                                CArg::Dbl(wsum as f64),
                                CArg::Dbl(self.m_bkgd_wsum_median as f64),
                                CArg::Dbl(
                                    ((wsum - self.m_bkgd_wsum_median) / self.m_bkgd_wsum_madn)
                                        as f64
                                ),
                                CArg::Dbl(elongation as f64)
                            );
                        }
                        if self.m_bkgd_wsum_median != 0.
                            && self.m_bkgd_wsum_madn > 0.
                            && (wsum - self.m_bkgd_wsum_median) / self.m_bkgd_wsum_madn
                                > self.m_bmr_min_wsum_madnratio
                            && elongation >= 0.
                        {
                            //
                            // If it is a strong bead with a low enough elongation, just raise the
                            // criteria for the mean residual test
                            if elongation < self.m_bmr_lower_elong_lim {
                                raise_crit *= self.m_bmr_low_elong_raise_fac;
                            } else if elongation < self.m_bmr_upper_elong_lim {
                                //
                                // For a mid-range elongation, check the elongation on at least 3 nearby
                                // views
                                num_elong = 0;
                                elong_max = 0.;
                                ibox = 1;
                                while ibox <= self.m_max_any_sum {
                                    let slot = ((self.m_iobj_do - 1) * self.m_max_any_sum + ibox
                                        - 1)
                                        as usize;
                                    iz = self.m_in_core[slot] - 1;
                                    if iz >= 0
                                        && iz != self.m_iz_next
                                        && b3dabs!(iz - self.m_iz_next) <= self.m_max_bmr_delta_z
                                    {
                                        let eind = ((iobj - 1) * self.m_max_view_do + iz) as usize;
                                        if self.m_elong_save[eind] < 0. {
                                            let npix = self.m_npix_box as usize;
                                            calc_elongation(
                                                &mut self.cp,
                                                &self.m_boxes[slot * npix..],
                                                self.m_nx_box,
                                                self.m_ny_box,
                                                0.,
                                                0.,
                                                &mut self.m_elong_save[eind],
                                            );
                                        }
                                        if self.m_elong_save[eind] >= 0. {
                                            num_elong += 1;
                                            self.m_array[(num_elong - 1) as usize] =
                                                self.m_elong_save[eind];
                                            elong_max = b3dmax!(elong_max, self.m_elong_save[eind]);
                                        }
                                    }
                                    ibox += 1;
                                }
                                //if (numElong > 0) print *,array(1:numElong)
                                //
                                // If all those beads are not elongated much, the scatter of elongation
                                // is not very big, and elongation is within range on the current view,
                                // it is safe to raise the criteria on the big mean residual here
                                if num_elong >= 3 && elong_max < self.m_bmr_upper_elong_lim {
                                    avg_sd(
                                        &self.m_array,
                                        num_elong,
                                        &mut elong_avg,
                                        &mut elong_sd,
                                        &mut res_sem,
                                    );
                                    if self.m_if_trace != 0 {
                                        printf!(
                                            "elong mean/SD %2d %7.3f %7.3f\n",
                                            CArg::Int(num_elong as i64),
                                            CArg::Dbl(elong_avg as f64),
                                            CArg::Dbl(elong_sd as f64)
                                        );
                                    }
                                    if elong_sd < self.m_bmr_max_elong_sd
                                        && elongation
                                            < elong_avg + self.m_bmr_max_num_sdabove_mean * elong_sd
                                    {
                                        raise_crit *= self.m_bmr_high_elong_raise_fac;
                                    }
                                }
                            }
                            //
                            // Repeat the test with the raised criterion
                            if !(cur_diff > self.m_res_diff_min * raise_crit
                                && (cur_diff as f64 - b3dmax!(0., res_avg as f64)) / res_sd as f64
                                    > (self.m_res_diff_crit * raise_crit) as f64
                                && err_max > self.m_cur_res_min * raise_crit)
                            {
                                if_mean_bad = 0;
                            }
                        }
                    }
                    //
                    // if error greater than criterion after pass 1, or mean
                    // residual has zoomed on either pass, delete point for
                    // next round
                    //
                    if (self.m_ipass == 1 && err_max > self.m_fit_dist_crit) || if_mean_bad == 1 {
                        if self.m_if_trace != 0 {
                            printf!(
                                "big res %2d %6.2f %4.2f %1d %8.4f %7.4f %8.4f %8.4f %8.4f %8.4f %7.4f\n",
                                CArg::Int(iobj as i64),
                                CArg::Dbl(err_max as f64),
                                CArg::Dbl((self.m_cur_res_min * raise_crit) as f64),
                                CArg::Int((if_mean_bad + self.m_ipass) as i64),
                                CArg::Dbl(
                                    self.m_res_mean[((self.m_iview_seq - 1) * max_all_real + iobj
                                        - 1)
                                        as usize] as f64
                                ),
                                CArg::Dbl(cur_diff as f64),
                                CArg::Dbl((self.m_res_diff_min * raise_crit) as f64),
                                CArg::Dbl(res_avg as f64),
                                CArg::Dbl(res_sd as f64),
                                CArg::Dbl(
                                    (cur_diff as f64 - b3dmax!(0., res_avg as f64)) / res_sd as f64
                                ),
                                CArg::Dbl((self.m_res_diff_crit * raise_crit) as f64)
                            );
                        }
                        self.m_wsum_save
                            [((iobj - 1) * self.mx.max_view + self.m_iview - 1) as usize] = -1.;
                        if self.cp.get_edge_sd != 0 {
                            self.m_edge_sd_save
                                [((iobj - 1) * self.m_max_view_do + self.m_iview - 1) as usize] =
                                -1.;
                        }
                        self.m_res_mean
                            [((self.m_iview_seq - 1) * max_all_real + iobj - 1) as usize] = -1.;
                        ibase = self.fm.ibase_obj[(iobj - 1) as usize];
                        ip = ibase + self.m_ip_nearest[iod] + 1;
                        while ip <= ibase + self.fm.npt_in_obj[(iobj - 1) as usize] {
                            self.fm.object[(ip - 1 - 1) as usize] =
                                self.fm.object[(ip - 1) as usize];
                            ip += 1;
                        }
                        self.fm.npt_in_obj[(iobj - 1) as usize] -= 1;
                        self.m_if_found[iod] = 0;
                        self.m_num_to_do += 1;
                        self.m_ip_nearest[iod] = self.m_ip_near_save[iod];
                        self.m_num_del += 1;
                        self.m_iobj_del[(self.m_num_del - 1) as usize] = iobj;
                    }
                }
                self.m_iobj_do += 1;
            }
            self.m_num_added = self.m_num_del;
            if self.m_num_del != 0 {
                printf!(
                    "%4d pts deleted%s, conts:",
                    CArg::Int(self.m_num_del as i64),
                    CArg::Str(if self.m_ipass > 1 {
                        ", big mean residual"
                    } else {
                        " for pass 2"
                    })
                );
                i = 0;
                while i < self.m_num_del {
                    printf!(
                        if self.m_need4digits { " %4d" } else { " %3d" },
                        CArg::Int(self.m_iobj_del[i as usize] as i64)
                    );
                    if i == self.m_num_del - 1
                        || (i + 1) % (if self.m_need4digits { 9 } else { 11 }) == 0
                    {
                        printf!("\n");
                    }
                    i += 1;
                }
                fflush_stdout!();
                if self.m_ipass > 1 {
                    self.m_num_added = 0;
                    tilt_ali(
                        &mut self.tc,
                        &mut self.av,
                        &self.mx,
                        &mut self.m_eval_funct,
                        &mut self.sg,
                        &self.fm,
                        &mut self.m_if_did_align,
                        &mut self.m_if_align_done,
                        &mut self.m_res_mean[res_off..],
                        self.m_iview,
                        &mut self.m_ali_mean_res,
                    );
                    if self.m_if_did_align != 0 {
                        self.m_ivs_on_align = self.m_iview_seq;
                    }
                }
            }
        }
        let _ = (elong_avg, elong_sd, res_sem);
    }

    /// Original: `BeadTrack::getBackgroundWsumStats` (`beadtrack.cpp:3607`).
    ///
    /// getBackgroundWsumStats - gets elongation, background wsum median/MADN/max
    pub fn get_background_wsum_stats(
        &mut self,
        xstat: f32,
        ystat: f32,
        iz_stat: i32,
        wsum_new: &mut f32,
        elongation: &mut f32,
    ) {
        let mut tmp_wsum: f32 = 0.;
        let mut xtmp: f32;
        let mut ytmp: f32;
        let num_stat_div: i32;
        let mut num_saved: i32;
        let indent: i32;
        let mut ix_start: i32;
        let mut ix_end: i32;
        let mut iy_start: i32;
        let mut iy_end: i32;
        let min_stat_num: i32;
        let mut missing: i32;
        let mut ix: i32;
        let mut iy: i32;
        self.m_bkgd_wsum_median = 0.;
        self.m_bkgd_wsum_madn = 0.;
        self.m_bkgd_wsum_max = 0.;
        num_stat_div = 10;
        min_stat_num = 15;
        num_saved = 0;

        find_piece(
            &self.m_ix_pclist,
            &self.m_iy_pclist,
            &self.m_iz_pclist,
            self.m_npclist,
            self.m_nx_im,
            self.m_ny_im,
            self.m_nx_stat_box,
            self.m_ny_stat_box,
            xstat,
            ystat,
            iz_stat,
            &mut self.m_ix0,
            &mut self.m_ix1,
            &mut self.m_iy0,
            &mut self.m_iy1,
            &mut self.m_ipiece_z,
            self.m_if_read_xfs,
            &self.m_prexf,
            &mut self.m_need_taper,
            &mut self.m_need_fill,
        );
        if self.m_ipiece_z < 0 {
            return;
        }
        {
            let mut load_tmp = std::mem::take(&mut self.m_stat_tmp);
            let ctf = std::mem::take(&mut self.m_ctf_stat);
            self.load_box_and_taper(
                &mut load_tmp,
                self.m_nx_stat_box,
                self.m_ny_stat_box,
                self.m_nx_stat_pad,
                self.m_ny_stat_pad,
                &ctf,
                self.m_delta_ctf_stat,
            );
            self.m_ctf_stat = ctf;
            self.m_stat_tmp = load_tmp;
        }

        // Get the elongation first by repacking the center of this over-sized box into
        // the regular box size
        ix_start = (self.m_nx_stat_box - self.m_nx_box) / 2;
        iy_start = (self.m_ny_stat_box - self.m_ny_box) / 2;
        repack_float_image(
            f32_bytes_mut!(self.m_array),
            f32_bytes!(self.m_stat_tmp),
            self.m_nx_stat_box,
            ix_start,
            ix_start + self.m_nx_box - 1,
            iy_start,
            iy_start + self.m_ny_box - 1,
        );
        calc_elongation(
            &mut self.cp,
            &self.m_array,
            self.m_nx_box,
            self.m_ny_box,
            0.,
            0.,
            elongation,
        );

        // Indent the area to use by the extent of the CG computation
        indent = (self.m_cg_radius + self.m_cg_gap_width + self.m_cg_edge_width).ceil() as i32;
        ix_start = 1 + indent;
        ix_end = self.m_nx_stat_box - indent;
        iy_start = 1 + indent;
        iy_end = self.m_ny_stat_box - indent;
        //
        // cut it down if there is a fill
        if self.m_need_fill {
            missing = self.m_nx_stat_box - (self.m_ix1 + 1 - self.m_ix0);
            if missing > 0 {
                if self.m_ix0 > 0 {
                    ix_end -= missing;
                } else {
                    ix_start += missing;
                }
            }
            missing = self.m_ny_stat_box - (self.m_iy1 + 1 - self.m_iy0);
            if missing > 0 {
                if self.m_iy0 > 0 {
                    iy_end -= missing;
                } else {
                    iy_start += missing;
                }
            }
        }
        xtmp = 0.;
        ytmp = 0.;
        calc_cg(
            &mut self.cp,
            &self.m_stat_tmp,
            self.m_nx_stat_box,
            self.m_ny_stat_box,
            &mut xtmp,
            &mut ytmp,
            wsum_new,
            &mut self.m_edge_sd,
        );
        //
        // Loop on the positions and get wsums
        iy = 1;
        while iy <= num_stat_div {
            ix = 1;
            while ix <= num_stat_div {
                xtmp = (ix_start + ((ix_end - ix_start) * (ix - 1)) / (num_stat_div - 1)
                    - self.m_nx_stat_box / 2) as f32;
                ytmp = (iy_start + ((iy_end - iy_start) * (iy - 1)) / (num_stat_div - 1)
                    - self.m_ny_stat_box / 2) as f32;
                if (xtmp * xtmp + ytmp * ytmp).sqrt() as f64 > 1.33 * self.m_diameter as f64 {
                    wsum_for_sobel_peak(
                        &mut self.cp,
                        &self.m_stat_tmp,
                        self.m_nx_stat_box,
                        self.m_ny_stat_box,
                        xtmp,
                        ytmp,
                        &mut tmp_wsum,
                        &mut self.m_edge_sd,
                    );
                    if tmp_wsum > 0. {
                        num_saved += 1;
                        self.m_array[(num_saved - 1) as usize] = tmp_wsum;
                    }
                }
                ix += 1;
            }
            iy += 1;
        }
        //
        // Get the stats if there are enough of them
        if num_saved < min_stat_num {
            return;
        }
        //bkgdWsumMax = maxval(array(1:numSaved))
        //call rsFastMedianInPlace(array, numSaved, bkgdWsumMedian)
        rs_sort_floats(&mut self.m_array, num_saved);
        rs_median_of_sorted(&self.m_array, num_saved, &mut self.m_bkgd_wsum_median);
        rs_percentile_of_sorted(&self.m_array, num_saved, 0.90, &mut self.m_bkgd_wsum_max);
        rs_fast_madn(
            &self.m_array,
            num_saved,
            self.m_bkgd_wsum_median,
            &mut self.m_stat_tmp,
            &mut self.m_bkgd_wsum_madn,
        );
    }

    /// Original: `BeadTrack::addSequence` (`beadtrack.cpp:3697`).
    ///
    /// Add a sequence from ivStart to ivEnd to the sequence list
    pub fn add_sequence(&mut self, ind: i32, iv_start: i32, iv_end: i32) {
        let n = self.m_num_seqs as usize;
        self.m_iv_seq_str[n] = iv_start;
        self.m_iv_seq_end[n] = iv_end;
        self.m_list_seq[n] = ind;
        self.m_num_seqs += 1;
    }

    /// Original: `BeadTrack::evaluateCGvsSobelResids` (`beadtrack.cpp:3708`).
    ///
    /// Do a tiltalign alignment with the current points and then with the positions based on
    /// centroid, keep track of how often CG is better and the mean residuals
    pub fn evaluate_cg_vs_sobel_resids(&mut self, iv_seq_use: i32) {
        let mut sobel_res: f32 = 0.;
        let mut cg_res: f32 = 0.;
        let mut num_new: i32;
        let mut ic: i32;
        //
        // Count number of new points and see if it is a large enough fraction
        num_new = 0;
        ic = 0;
        while ic < self.fm.n_point {
            if self.m_iflag_cgvs_sobel[ic as usize] > 0 {
                num_new += 1;
            }
            ic += 1;
        }
        if (num_new as f64) < 0.25 * self.fm.n_point as f64 {
            return;
        }

        printf!(" Aligning Sobel-centered points for comparison\n");
        fflush_stdout!();
        let off = ((iv_seq_use - 1) * self.m_max_all_real) as usize;
        tilt_ali(
            &mut self.tc,
            &mut self.av,
            &self.mx,
            &mut self.m_eval_funct,
            &mut self.sg,
            &self.fm,
            &mut self.m_if_did_align,
            &mut self.m_if_align_done,
            &mut self.m_res_mean[off..],
            self.m_iview,
            &mut sobel_res,
        );
        if self.m_if_did_align == 0 {
            return;
        }
        self.swap_sobel_and_cg_coords();
        printf!(" Aligning saved centroid-centered points\n");
        fflush_stdout!();
        tilt_ali(
            &mut self.tc,
            &mut self.av,
            &self.mx,
            &mut self.m_eval_funct,
            &mut self.sg,
            &self.fm,
            &mut self.m_if_did_align,
            &mut self.m_if_align_done,
            &mut self.m_res_mean[off..],
            self.m_iview,
            &mut cg_res,
        );
        if cg_res < sobel_res {
            self.m_num_cgbetter += 1;
        }
        self.m_num_sobel_cgeval += 1;
        self.m_cg_res_sum += cg_res;
        self.m_sobel_res_sum += sobel_res;
        self.swap_sobel_and_cg_coords();
    }

    /// Original: `BeadTrack::swapSobelAndCGcoords` (`beadtrack.cpp:3743`).
    ///
    /// Substitute the saved centroid-based coordinates for the sobel-refined ones or vice
    /// versa
    pub fn swap_sobel_and_cg_coords(&mut self) {
        let mut ic: i32;
        let mut ib: i32;
        let mut temp: f32;
        ic = 0;
        while ic < self.fm.n_point {
            if self.m_iflag_cgvs_sobel[ic as usize] > 0 {
                ib = 0;
                while ib < 2 {
                    temp = self.fm.p_coord[ic as usize][ib as usize];
                    self.fm.p_coord[ic as usize][ib as usize] =
                        self.m_saved_cgcoord[(ic * 2 + ib) as usize];
                    self.m_saved_cgcoord[(ic * 2 + ib) as usize] = temp;
                    ib += 1;
                }
            }
            ic += 1;
        }
    }
}
