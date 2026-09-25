//! Translation of `IMOD/flib/tilt/tilt.cpp` with its class header
//! `IMOD/flib/tilt/tilt.h` merged in.
//!
//! The C++ `Tilt` class becomes the [`Tilt`] struct, one method per member
//! function, with the original name in each doc comment.  The source's
//! `float *` / `int *` members become owned `Vec`s; an empty `Vec` is the
//! source's `NULL` (`B3DFREE`, the constructor's `= NULL`, and the
//! `if (!mXRayStart)` test in `reprojOneAngle`).  `B3DMALLOC` is uninitialised
//! memory where these are zero-filled; nothing in the source reads an element
//! before writing it on the paths translated here.  An allocation failure
//! aborts rather than returning `NULL`, so `allocateArray`'s retry loop only
//! ever retries for the `inputNeed + outBufNeed < 2147000000` test.
//!
//! Where a member function takes a `float *` that the caller points into one of
//! the class's own arrays (`transform(&mInputArray[ibase], ...)`,
//! `project(mWorkArray, ...)`, `taperEndToStart(&mInputArray[istart - 1])`),
//! the caller lends the array out of the struct with `std::mem::take` for the
//! call and puts it back afterwards, so the method can read the rest of the
//! instance while writing the lent array.  The source's pointer offsets become
//! subslices at the same offsets.
//!
//! `tilt.cpp` is C++: `log10`, `pow`, `sin`, `cos`, `sqrt`, `atan`, `atan2` of
//! a `float` resolve to the `float` overloads (`log10f`, `powf`, `sinf`,
//! `cosf`, `sqrtf`, `atanf`, `atan2f` are what the reference binary imports),
//! so those calls are `f32` methods here; a `double` argument uses the `double`
//! function.  `B3DNINT` adds a `double` `0.5`.
//!
//! GPU: the reference links `nogpu.cpp`, whose stubs all report failure, so
//! every GPU branch here calls the [`super::nogpu`] stubs and falls through to
//! the CPU path as the reference does.
//!
//! OpenMP: `tilt.cpp` has no OpenMP directive; the Fortran kernels it calls
//! have none either.  Everything is sequential, as in the reference; the
//! library routines it reaches that are parallel (`rotateFlipImage`) produce
//! output independent of the thread count.
//!
//! A `(int)` conversion of a `float`/`double` in the hot loops is
//! [`fortran_int!`](super) (`cvttss2si`/`cvttsd2si`, what the reference's C++
//! compiler emits too); elsewhere `as i32`, which differs only out of range.
//! The cosine-stretched backprojection loops in `project` and the X/Z sampling
//! loops of `reprojOneAngle` take their output run and input window as slices
//! checked once per run or step rather than per element, and the default
//! stretch factor of 2 has its own loop so the compiler can vectorize it as
//! the reference's compiler does; each element is still the source's own
//! expression, so no result changes.

use super::bpsumlocal::bp_sum_local;
use super::bpsumnox::{bp_sum_area_no_x, bp_sum_no_x};
use super::bpsumxtilt::bp_sum_xtilt;
use super::nogpu::{
    gpu_alloc_arrays, gpu_available, gpu_bp_local, gpu_bp_no_x, gpu_bp_xtilt, gpu_done,
    gpu_filter_lines, gpu_filter_raw_image, gpu_load_filter, gpu_load_locals, gpu_load_proj,
    gpu_load_raw_filt_map, gpu_reproj_local, gpu_reproj_one_slice, gpu_reproject, gpu_shift_proj,
};
use super::projsumlocal::{loaded_projecting_point, proj_sum_local};
use crate::imod::flib::subrs::hvem::set_projection_rays::set_projection_rays;
use crate::imod::libcfshr::amat_to_rotmagstr::amat_to_rotmagstr;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_i_min, b3d_lock_file, b3d_output_file_type, c_format_bytes, fgetline,
    get_standard_gpu_options, imod_backup_file, number_in_list, set_float_16_output_mode,
    wall_time, write_16_bit_mode_for_floats,
};
use crate::imod::libcfshr::filtxcorr::{fourier_reduce_image, nice_frame};
use crate::imod::libcfshr::linearxforms::{xf_copy, xf_invert, xf_mult};
use crate::imod::libcfshr::parse_params::{
    PipValueArray, exit_error, pip_done, pip_get_boolean, pip_get_error, pip_get_float,
    pip_get_float_array, pip_get_in_out_file, pip_get_integer, pip_get_integer_array,
    pip_get_line_of_values, pip_get_string, pip_get_three_floats, pip_get_two_floats,
    pip_get_two_integers, pip_number_of_entries, pip_print_help, pip_read_or_parse_options,
    pip_set_special_flags,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::projectpixel::make_ray_area_lookup_table;
use crate::imod::libcfshr::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};
use crate::imod::libcfshr::reduce_by_binning::extract_with_binning;
use crate::imod::libcfshr::robuststat::{rs_sort_floats, rs_sort_ints};
use crate::imod::libcfshr::rotateflip::{RotateFlipData, rotate_flip_image};
use crate::imod::libcfshr::samplemeansd::{sample_mean_sd, type_for_sample_mean};
use crate::imod::libcfshr::simplestat::{array_min_max_mean, array_min_max_mean_sd};
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_out_pad};
use crate::imod::libfft::odfft::{nice_fft_limit, odfft_c};
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::libiimod::iimage::ii_test_if_hdf;
use crate::imod::libiimod::mrcfiles::mrc_fill_label_string;
use crate::imod::libiimod::parallelwrite::{
    iiu_par_wrt_flush_buffers, iiu_par_wrt_initialize, iiu_par_wrt_reclose_hdf,
    iiu_write_dummy_sec_to_hdf, par_wrt_close, par_wrt_lin, par_wrt_posn, par_wrt_properties,
    par_wrt_sec, par_wrt_set_current,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_file_type, iiu_mrc_header, iiu_open, iiu_read_lines, iiu_read_sec_part,
    iiu_read_section, iiu_set_position, iiu_write_lines, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_alt_space_group, iiu_alt_tilt,
    iiu_alt_tilt_orig, iiu_create_header, iiu_print_header, iiu_ret_basic_head, iiu_ret_data_type,
    iiu_ret_delta, iiu_ret_origin, iiu_ret_size, iiu_trans_labels, iiu_write_header,
};
use crate::imod::libimod::icont::imod_contours_new;
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, IMODF_FLIPYZ, IMODF_Z_FROM_MINUSPT5, Imod, Ipoint, imod_flip_yz, imod_new,
    imod_new_object, imod_set_ref_image,
};
use crate::imod::libimod::imodel_files::{imod_file_write, imod_read};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_PNT_ON_SEC, IMOD_OBJFLAG_USE_VALUE, MATFLAGS2_CONSTANT, MATFLAGS2_SKIP_LOW,
};
use crate::imod::libimod::istore::{
    GEN_STORE_FLOAT, GEN_STORE_MINMAX1, GEN_STORE_VALUE1, Istore, istore_find_add_min_max1,
    istore_get_min_max, istore_insert, istore_lookup,
};
use std::io::{BufRead, BufReader, Write as _};

/// `tilt.h:5`.
const MAX_LINE: usize = 1000;
/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;
/// `iimage.h:123`: `#define MAX_HALF_FLOAT 65504.`, a double.
const MAX_HALF_FLOAT: f64 = 65504.;
/// `mrcslice.h`: `SLICE_MODE_FLOAT`.
const SLICE_MODE_FLOAT: i32 = 2;
/// `mrcfiles.h`: `MRC_MODE_FLOAT`.
const MRC_MODE_FLOAT: i32 = 2;

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMIN(a,b) ((a) < (b) ? (a) : (b))`; both operands must already have the
/// type the C's usual arithmetic conversions give them.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        if a < b { a } else { b }
    }};
}

/// `B3DMAX(a,b) ((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        if a > b { a } else { b }
    }};
}

/// `B3DABS(a) ((a) >= 0 ? (a) : -(a))`.
macro_rules! b3dabs {
    ($a:expr) => {{
        let a = $a;
        if a >= 0 as _ { a } else { -a }
    }};
}

/// `B3DSIGN(a,b) ((b) < 0 ? -(a) : (a))`.
macro_rules! b3dsign {
    ($a:expr, $b:expr) => {{
        let a = $a;
        if ($b) < 0 as _ { -a } else { a }
    }};
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

fn ci(v: i32) -> CArg<'static> {
    CArg::Int(v as i64)
}

fn cf(v: f64) -> CArg<'static> {
    CArg::Dbl(v)
}

/// `exit(status)` through the translated `exit`, so libc-order stdout is
/// flushed as the C library's `exit` flushes it.
fn c_exit(status: i32) -> ! {
    let _ = ImodFile::Stdout.flush();
    crate::imod::libcfshr::b3dutil::exit(status)
}

/// C `main` (`tilt.cpp:86`): `Tilt tilt; tilt.main(argc, argv); exit(0);`.
pub fn tilt(arguments: &[String]) -> i32 {
    let mut tilt = Tilt::new();
    tilt.main(arguments);
    c_exit(0);
}

/// `class Tilt` (`tilt.h:7-329`).  Field order and comments follow the header.
pub struct Tilt {
    m_ind_load_end: i32,
    m_max_stack: i64,
    m_ind_needed_base: i32,
    m_num_need_se: i32,
    m_filter_array: Vec<f32>,
    m_out_slice_arr: Vec<f32>,
    m_vert_slice_arr: Vec<f32>,
    m_input_array: Vec<f32>,
    m_read_in_array: Vec<f32>,
    m_work_array: Vec<f32>,
    m_super_out_arr: Vec<f32>,
    m_out_buffer: Vec<f32>,
    m_need_for_filt_arr: i32,
    m_need_for_out_arr: i32,
    m_need_for_vert_arr: i64,
    m_need_for_work_arr: i32,
    m_need_for_read_in_arr: i64,
    m_need_for_super_arr: i64,
    m_load_buffer: Vec<f32>,
    m_rot_buffer: Vec<f32>,
    m_raw_filt_map: Vec<f32>,
    m_reproj_lines: Vec<f32>,
    m_proj_line: Vec<f32>,
    m_orig_lines: Vec<f32>,
    m_super_temp_arr: Vec<f32>,
    m_phase_real: Vec<f32>,
    m_phase_imag: Vec<f32>,
    m_needed_starts: Vec<i32>,
    m_needed_ends: Vec<i32>,
    m_interval_head_save: i32,
    m_iwidth: i32,
    m_ithick_bp: i32,
    m_islice_start: i32,
    m_islice_end: i32,
    m_num_pad: i32,
    m_nx_proj: i32,
    m_ny_proj: i32,
    m_nx_pad_dim: i32,
    m_nx_filt_dim: i32,
    m_load_xoffset: i32,
    m_nx_full_proj: i32,
    m_num_views: i32,
    m_lim_view: i32,
    m_lim_reproj: i32,
    m_ithick_out: i32,
    m_super_samp_border: i32,
    m_nx_super_samp: i32,
    m_ny_super_samp: i32,
    m_nx_out_pad: i32,
    m_ny_out_pad: i32,
    m_clean_super_fft: i32,
    m_nx_gpu_crop_pad: i32,
    m_ny_gpu_crop_pad: i32,
    m_beta_min: f32,
    m_beta_max: f32,
    m_min_cos_alpha: f32,
    /// `int mTitle[20]`: an 80-byte MRC label.
    m_title: [u8; 80],
    /// `char mLine[MAX_LINE]`, the line buffer for reading text files.
    m_line: Vec<u8>,
    m_gpu_err_string: String,
    m_dmin_in: f32,
    m_dmax_in: f32,
    m_dmean_in: f32,
    m_val_min: f32,
    m_mask_edges: i32,
    m_perpendicular: i32,
    m_reproj_bp: i32,
    m_rec_reproj: bool,
    m_debug: i32,
    m_read_base_rec: bool,
    m_rec_subtraction: bool,
    m_proj_subtraction: i32,
    m_vert_sirt_input: i32,
    m_save_vert_slices: bool,
    m_rotate_by90: i32,
    m_last_pre_write_line: i32,
    m_parallel_hdf: i32,
    m_use_gpu: bool,
    m_sirt_from_zero: bool,
    m_axis_xoffset: f32,
    m_xcen_in: f32,
    m_center_slice: f32,
    m_xcen_out: f32,
    m_ycen_out: f32,
    m_base_for_log: f32,
    m_y_offset: f32,
    m_xzfac: Vec<f32>,
    m_yzfac: Vec<f32>,
    m_compress: Vec<f32>,
    m_expose_weight: Vec<f32>,
    m_angles: Vec<f32>,
    m_edge_fill: f32,
    m_zero_weight: f32,
    m_flat_frac: f32,
    m_ycen_mod_proj: f32,
    m_new_mode: i32,
    m_if_log: i32,
    m_in_plane_size: i32,
    m_ip_extra_size: i32,
    m_num_planes: i32,
    m_num_out_buf_slices: i32,
    m_cur_out_buf_slice: i32,
    m_interp_fac_stretch: i32,
    m_interp_ord_stretch: i32,
    m_interp_ord_xtilt: i32,
    m_super_sample_fac: i32,
    m_proj_super_fac: i32,
    m_min_tot_slice: i32,
    m_max_tot_slice: i32,
    m_num_view_base: i32,
    m_num_view_subtract: i32,
    m_num_extra_mask_pix: i32,
    m_num_vert_needed: i32,
    m_islice_size_bp: i64,
    m_nx_stretched: Vec<i32>,
    m_ind_stretch_line: Vec<i32>,
    m_map_used_view: Vec<i32>,
    m_stretch_offset: Vec<f32>,
    m_wgt_angles: Vec<f32>,
    m_exact_table: Vec<f32>,
    m_num_tilt_inc_wgt: i32,
    m_num_wgt_angles: i32,
    m_num_exact_cycles: i32,
    m_num_fake_sirt_iter: i32,
    m_tilt_inc_wgts: [f32; 20],
    m_exact_samples: f32,
    m_exact_obj_size: f32,
    /// `char *mFilterFile`; `None` is `NULL`.
    m_filter_file: Option<Vec<u8>>,
    m_use_intersections: i32,
    m_ix_unmasked_se: Vec<i32>,
    m_iview_subtract: Vec<i32>,
    m_max_ray_pixels: Vec<i32>,
    m_out_add: f32,
    m_out_scale: f32,
    m_adjust_out_add_fac: f32,
    m_base_out_add: f32,
    m_base_out_scale: f32,
    m_effective_scale: f32,
    m_use_raw_stack: i32,
    m_rot_flip_operation: i32,
    m_raw_remaining_rot: f32,
    m_ny_raw_padded: i32,
    m_min_raw_pad_aspect: f32,
    m_do_raw_filter_on_gpu: bool,
    m_sin_alpha: Vec<f32>,
    m_sin_beta: Vec<f32>,
    m_cos_alpha: Vec<f32>,
    m_cos_beta: Vec<f32>,
    m_alpha: Vec<f32>,
    m_if_alpha: i32,
    m_num_warp_pos: i32,
    m_lim_warp: i32,
    m_nx_warp: i32,
    m_ny_warp: i32,
    m_ix_start_warp: i32,
    m_iy_start_warp: i32,
    m_idel_xwarp: i32,
    m_idel_ywarp: i32,
    m_if_del_alpha: i32,
    m_ind_warp: Vec<i32>,
    m_num_pix_in_ray: Vec<i32>,
    m_del_alpha: Vec<f32>,
    m_del_beta: Vec<f32>,
    m_cwarp_alpha: Vec<f32>,
    m_cwarp_beta: Vec<f32>,
    m_swarp_alpha: Vec<f32>,
    m_swarp_beta: Vec<f32>,
    m_fwarp: Vec<f32>,
    m_warp_xzfac: Vec<f32>,
    m_warp_yzfac: Vec<f32>,
    m_warp_delz: Vec<f32>,
    m_xray_start: Vec<f32>,
    m_yray_start: Vec<f32>,
    m_xproj_fs: Vec<f32>,
    m_xproj_zs: Vec<f32>,
    m_yproj_fs: Vec<f32>,
    m_yproj_zs: Vec<f32>,
    m_num_reproj: i32,
    m_max_zreproj: i32,
    m_min_xreproj: i32,
    m_max_xreproj: i32,
    m_min_yreproj: i32,
    m_max_yreproj: i32,
    m_min_zreproj: i32,
    m_ithick_reproj: i32,
    m_min_xload: i32,
    m_max_xload: i32,
    m_num_warp_delz: i32,
    m_num_sirt_iter: i32,
    m_if_out_sirt_proj: i32,
    m_if_out_sirt_rec: i32,
    m_isign_constraint: i32,
    m_iter_for_report: i32,
    m_num_read_need: i32,
    m_xproj_offset: f32,
    m_yproj_offset: f32,
    m_proj_mean: f32,
    m_filter_scale: f32,
    m_dx_warp_delz: f32,
    m_thresh_for_reproj: f32,
    m_thresh_polarity: f32,
    m_thresh_sum_fac: f32,
    m_thresh_mark_val: f32,
    m_thresh_fill_val: f32,
    m_report_vals: Vec<f32>,
    m_cos_reproj: Vec<f32>,
    m_sin_reproj: Vec<f32>,
    m_num_gpu_planes: i32,
    m_load_gpu_start: i32,
    m_load_gpu_end: i32,
    m_if_gpu_by_environ: i32,
    m_iact_gpu_fail_option: i32,
    m_iact_gpu_fail_environ: i32,
    m_skip_unseen_points: i32,
    /// `static bool needMap = true` in `loadAndFilterRawData`
    /// (`tilt.cpp:1077`), kept with the one instance.
    s_need_map: bool,
}

impl Tilt {
    /// `Tilt::Tilt` (`tilt.cpp:94`).  The constructor sets only the array
    /// pointers to `NULL`; the scalar members it leaves alone are the stack
    /// object's indeterminate values in the source and are all assigned in
    /// `inputParameters` or `main` before they are read.  They start at zero
    /// here.
    pub fn new() -> Self {
        Tilt {
            m_ind_load_end: 0,
            m_max_stack: 0,
            m_ind_needed_base: 0,
            m_num_need_se: 0,
            m_filter_array: Vec::new(),
            m_out_slice_arr: Vec::new(),
            m_vert_slice_arr: Vec::new(),
            m_input_array: Vec::new(),
            m_read_in_array: Vec::new(),
            m_work_array: Vec::new(),
            m_super_out_arr: Vec::new(),
            m_out_buffer: Vec::new(),
            m_need_for_filt_arr: 0,
            m_need_for_out_arr: 0,
            m_need_for_vert_arr: 0,
            m_need_for_work_arr: 0,
            m_need_for_read_in_arr: 0,
            m_need_for_super_arr: 0,
            m_load_buffer: Vec::new(),
            m_rot_buffer: Vec::new(),
            m_raw_filt_map: Vec::new(),
            m_reproj_lines: Vec::new(),
            m_proj_line: Vec::new(),
            m_orig_lines: Vec::new(),
            m_super_temp_arr: Vec::new(),
            m_phase_real: Vec::new(),
            m_phase_imag: Vec::new(),
            m_needed_starts: Vec::new(),
            m_needed_ends: Vec::new(),
            m_interval_head_save: 0,
            m_iwidth: 0,
            m_ithick_bp: 0,
            m_islice_start: 0,
            m_islice_end: 0,
            m_num_pad: 0,
            m_nx_proj: 0,
            m_ny_proj: 0,
            m_nx_pad_dim: 0,
            m_nx_filt_dim: 0,
            m_load_xoffset: 0,
            m_nx_full_proj: 0,
            m_num_views: 0,
            m_lim_view: 0,
            m_lim_reproj: 0,
            m_ithick_out: 0,
            m_super_samp_border: 0,
            m_nx_super_samp: 0,
            m_ny_super_samp: 0,
            m_nx_out_pad: 0,
            m_ny_out_pad: 0,
            m_clean_super_fft: 0,
            m_nx_gpu_crop_pad: 0,
            m_ny_gpu_crop_pad: 0,
            m_beta_min: 0.,
            m_beta_max: 0.,
            m_min_cos_alpha: 0.,
            m_title: [0; 80],
            m_line: vec![0; MAX_LINE],
            m_gpu_err_string: String::new(),
            m_dmin_in: 0.,
            m_dmax_in: 0.,
            m_dmean_in: 0.,
            m_val_min: 0.,
            m_mask_edges: 0,
            m_perpendicular: 0,
            m_reproj_bp: 0,
            m_rec_reproj: false,
            m_debug: 0,
            m_read_base_rec: false,
            m_rec_subtraction: false,
            m_proj_subtraction: 0,
            m_vert_sirt_input: 0,
            m_save_vert_slices: false,
            m_rotate_by90: 0,
            m_last_pre_write_line: 0,
            m_parallel_hdf: 0,
            m_use_gpu: false,
            m_sirt_from_zero: false,
            m_axis_xoffset: 0.,
            m_xcen_in: 0.,
            m_center_slice: 0.,
            m_xcen_out: 0.,
            m_ycen_out: 0.,
            m_base_for_log: 0.,
            m_y_offset: 0.,
            m_xzfac: Vec::new(),
            m_yzfac: Vec::new(),
            m_compress: Vec::new(),
            m_expose_weight: Vec::new(),
            m_angles: Vec::new(),
            m_edge_fill: 0.,
            m_zero_weight: 0.,
            m_flat_frac: 0.,
            m_ycen_mod_proj: 0.,
            m_new_mode: 0,
            m_if_log: 0,
            m_in_plane_size: 0,
            m_ip_extra_size: 0,
            m_num_planes: 0,
            m_num_out_buf_slices: 0,
            m_cur_out_buf_slice: 0,
            m_interp_fac_stretch: 0,
            m_interp_ord_stretch: 0,
            m_interp_ord_xtilt: 0,
            m_super_sample_fac: 0,
            m_proj_super_fac: 0,
            m_min_tot_slice: 0,
            m_max_tot_slice: 0,
            m_num_view_base: 0,
            m_num_view_subtract: 0,
            m_num_extra_mask_pix: 0,
            m_num_vert_needed: 0,
            m_islice_size_bp: 0,
            m_nx_stretched: Vec::new(),
            m_ind_stretch_line: Vec::new(),
            m_map_used_view: Vec::new(),
            m_stretch_offset: Vec::new(),
            m_wgt_angles: Vec::new(),
            m_exact_table: Vec::new(),
            m_num_tilt_inc_wgt: 0,
            m_num_wgt_angles: 0,
            m_num_exact_cycles: 0,
            m_num_fake_sirt_iter: 0,
            m_tilt_inc_wgts: [0.; 20],
            m_exact_samples: 0.,
            m_exact_obj_size: 0.,
            m_filter_file: None,
            m_use_intersections: 0,
            m_ix_unmasked_se: Vec::new(),
            m_iview_subtract: Vec::new(),
            m_max_ray_pixels: Vec::new(),
            m_out_add: 0.,
            m_out_scale: 0.,
            m_adjust_out_add_fac: 0.,
            m_base_out_add: 0.,
            m_base_out_scale: 0.,
            m_effective_scale: 0.,
            m_use_raw_stack: 0,
            m_rot_flip_operation: 0,
            m_raw_remaining_rot: 0.,
            m_ny_raw_padded: 0,
            m_min_raw_pad_aspect: 0.,
            m_do_raw_filter_on_gpu: false,
            m_sin_alpha: Vec::new(),
            m_sin_beta: Vec::new(),
            m_cos_alpha: Vec::new(),
            m_cos_beta: Vec::new(),
            m_alpha: Vec::new(),
            m_if_alpha: 0,
            m_num_warp_pos: 0,
            m_lim_warp: 0,
            m_nx_warp: 0,
            m_ny_warp: 0,
            m_ix_start_warp: 0,
            m_iy_start_warp: 0,
            m_idel_xwarp: 0,
            m_idel_ywarp: 0,
            m_if_del_alpha: 0,
            m_ind_warp: Vec::new(),
            m_num_pix_in_ray: Vec::new(),
            m_del_alpha: Vec::new(),
            m_del_beta: Vec::new(),
            m_cwarp_alpha: Vec::new(),
            m_cwarp_beta: Vec::new(),
            m_swarp_alpha: Vec::new(),
            m_swarp_beta: Vec::new(),
            m_fwarp: Vec::new(),
            m_warp_xzfac: Vec::new(),
            m_warp_yzfac: Vec::new(),
            m_warp_delz: Vec::new(),
            m_xray_start: Vec::new(),
            m_yray_start: Vec::new(),
            m_xproj_fs: Vec::new(),
            m_xproj_zs: Vec::new(),
            m_yproj_fs: Vec::new(),
            m_yproj_zs: Vec::new(),
            m_num_reproj: 0,
            m_max_zreproj: 0,
            m_min_xreproj: 0,
            m_max_xreproj: 0,
            m_min_yreproj: 0,
            m_max_yreproj: 0,
            m_min_zreproj: 0,
            m_ithick_reproj: 0,
            m_min_xload: 0,
            m_max_xload: 0,
            m_num_warp_delz: 0,
            m_num_sirt_iter: 0,
            m_if_out_sirt_proj: 0,
            m_if_out_sirt_rec: 0,
            m_isign_constraint: 0,
            m_iter_for_report: 0,
            m_num_read_need: 0,
            m_xproj_offset: 0.,
            m_yproj_offset: 0.,
            m_proj_mean: 0.,
            m_filter_scale: 0.,
            m_dx_warp_delz: 0.,
            m_thresh_for_reproj: 0.,
            m_thresh_polarity: 0.,
            m_thresh_sum_fac: 0.,
            m_thresh_mark_val: 0.,
            m_thresh_fill_val: 0.,
            m_report_vals: Vec::new(),
            m_cos_reproj: Vec::new(),
            m_sin_reproj: Vec::new(),
            m_num_gpu_planes: 0,
            m_load_gpu_start: 0,
            m_load_gpu_end: 0,
            m_if_gpu_by_environ: 0,
            m_iact_gpu_fail_option: 0,
            m_iact_gpu_fail_environ: 0,
            m_skip_unseen_points: 0,
            s_need_map: true,
        }
    }
}

impl Tilt {
    /// `Tilt::main` (`tilt.cpp:116`): runs the sequence of operations.
    pub fn main(&mut self, argv: &[String]) {
        let mut nxyz_tmp: [i32; 3] = [0; 3];
        let nxyzst: [i32; 3] = [0, 0, 0];
        let format920 = "Reading in view %d for slice %d\n";
        let num_slices: i32;
        let mut num_slice_out: i32;
        let mut in_load_start: i32;
        let mut in_load_end: i32;
        let mut last_ready: i32;
        let mut next_free_vert_slice: i32;
        let mut lvert_slice_start: i32;
        let mut lvert_slice_end: i32;
        let mut num_vert_slice_in_ring: i32;
        let ni: i32;
        let load_limit: i32;
        let mut lslice_out: i32;
        let mut lslice_start: i32;
        let mut lslice_end: i32;
        let mut need_start: i32;
        let mut need_end: i32 = 0;
        let mut itry_end: i32;
        let mut ibase_sirt: i32;
        let mut itry: i32;
        let mut if_enough: i32;
        let mut last_start: i32;
        let mut last_end: i32;
        let mut lslice_min: i32;
        let mut lslice_max: i32;
        let mut next_read_free: i32;
        let mut nv: i32 = 0;
        let mut num_already: i32;
        let mut lslice_proj_end: i32;
        let mut l_read_start: i32;
        let mut l_read_end: i32;
        let mut num_read_in_ring: i32;
        let mut dmin: f32;
        let mut dmax: f32;
        let ycen_fix: f32;
        let abs_sin_alf: f32;
        let mut tan_alpha: f32 = 0.;
        let mut dmin4: f32;
        let mut dmax4: f32;
        let mut dmin5: f32;
        let mut dmax5: f32;
        let mut dmin6: f32;
        let mut dmax6: f32;
        let mut vert_slice_cen: f32;
        let mut vert_ycen_fix: f32;
        let mut ri_bot: f32;
        let mut ri_top: f32;
        let mut tmax: f32 = 0.;
        let mut tmean: f32 = 0.;
        let mut tmin: f32 = 0.;
        let mut lri_min: i32;
        let mut lri_max: i32;
        let mut load_start: i32;
        let mut load_end: i32;
        let mut ierr: i32;
        let mut j: i32;
        let mut k: i32 = 0;
        let mut lpos_start: i32;
        let mut lpos_end: i32;
        let mut ibase: i32;
        let mut lread_start: i32;
        let mut istart: i32;
        let mut nl: i32 = 0;
        let mut ind_buf: i32;
        let mut iyload: i32;
        let mut ioffset: i64;
        let mut iring_start: i32;
        let mut need_gpu_start: i32 = 0;
        let mut need_gpu_end: i32 = 0;
        let mut keep_on_gpu: i32 = 0;
        let mut num_load_gpu: i32 = 0;
        let mut unscaled_min: f32 = 0.;
        let mut unscaled_max: f32 = 0.;
        let mut rec_scale: f32;
        let rec_add: f32;
        let dmean: f32;
        let mut pixel_tot: f32;
        let mut reproj_fill: f32;
        let mut cur_mean: f32 = 0.;
        let mut first_mean: f32 = 0.;
        let mut vert_sum: f32;
        let mut num_vert_sum: f32;
        let edge_fill_orig: f32;
        let mut compose_fill: f32;
        let mut iset: i32;
        let mut nz5: i32;
        let mut lfill_start: i32;
        let mut lfill_end: i32 = 0;
        let mut dtot8: f64;
        let mut time_start: f64;
        let mut dsum: f64;
        let mut dpix: f64;
        let mut shifted_gpu_load: bool;
        let mut composed_one: bool;
        let mut truncations: bool;
        let mut extremes: bool;
        //
        num_slice_out = 0;
        dtot8 = 0.;
        dmin = 1.0e30;
        dmax = -1.0e30;
        dmin4 = 1.0e30;
        dmax4 = -1.0e30;
        dmin5 = 1.0e30;
        dmax5 = -1.0e30;
        dmin6 = 1.0e30;
        dmax6 = -1.0e30;
        nz5 = 0;
        self.m_debug = 0;
        self.m_skip_unseen_points = 0;
        //
        // Open files and read control data
        self.input_parameters(argv);
        num_slices = (self.m_islice_end - self.m_islice_start) + 1;
        self.m_interval_head_save = b3dmax!(20, num_slices / 50);
        self.m_val_min = (1.0e-3 * (self.m_dmax_in - self.m_dmin_in) as f64) as f32;
        self.m_edge_fill = self.m_dmean_in;
        if self.m_if_log != 0 {
            self.m_edge_fill =
                b3dmax!(self.m_val_min, self.m_dmean_in + self.m_base_for_log).log10();
        }
        self.m_edge_fill = self.m_edge_fill * self.m_zero_weight;
        edge_fill_orig = self.m_edge_fill;

        if self.m_debug != 0 {
            printf!(
                "iflog= %d scale= %f  edgefill= %f\n",
                ci(self.m_if_log),
                cf(self.m_out_scale as f64),
                cf(self.m_edge_fill as f64)
            );
        }
        compose_fill = self.m_edge_fill * self.m_num_views as f32;
        //
        // initialize variables for loaded slices
        in_load_start = 0;
        in_load_end = 0;
        last_ready = 0;
        //
        // initialize variables for ring buffer of vertical slices: # in the ring, next free
        // position, starting and ending slice number of slices in ring
        num_vert_slice_in_ring = 0;
        next_free_vert_slice = 1;
        lvert_slice_start = -1;
        lvert_slice_end = -1;
        //
        // initialize similar variables for ring buffer of read-in slices
        num_read_in_ring = 0;
        next_read_free = 1;
        l_read_start = -1;
        l_read_end = -1;
        lfill_start = -1;
        ycen_fix = self.m_ycen_out;
        abs_sin_alf = b3dabs!(self.m_sin_alpha[0]);
        composed_one = self.m_if_alpha >= 0;
        num_vert_sum = 0.;
        vert_sum = 0.;
        //
        // Report memory allocations for old-times sake
        ni = self.m_num_planes * self.m_in_plane_size;
        if !self.m_rec_reproj {
            printf!(
                "---------------------------\nTotal floats in main arrays %11ld\n  Radial weighting function  %10d\n  Output slice             %12d\n",
                CArg::Int(self.m_max_stack),
                ci(self.m_need_for_filt_arr),
                ci(self.m_islice_size_bp as i32)
            );
            if self.m_need_for_super_arr != 0 {
                printf!(
                    "  Super-sampled slice      %12d\n",
                    ci(self.m_need_for_super_arr as i32)
                );
            }
            if self.m_if_alpha < 0 {
                printf!(
                    " %4d Untilted slices        %10ld\n",
                    ci(self.m_num_vert_needed),
                    CArg::Int(self.m_need_for_vert_arr)
                );
            }
            if self.m_num_sirt_iter > 0 {
                if self.m_need_for_read_in_arr != 0 {
                    printf!(
                        " %4d Slice(s) for SIRT      %10ld\n",
                        ci(b3dmax!(1, self.m_num_read_need)),
                        CArg::Int(self.m_need_for_read_in_arr)
                    );
                }
                printf!(
                    "  Reprojection lines for SIRT %9d\n",
                    ci(self.m_in_plane_size)
                );
            }
            printf!(
                " %4d Transposed projections %10d\n",
                ci(self.m_num_planes),
                ci(ni)
            );
            if self.m_ip_extra_size != 0 {
                printf!(
                    "  Stretching buffer           %9d\n",
                    ci(self.m_ip_extra_size)
                );
            }
            if self.m_num_out_buf_slices > 0 {
                printf!(
                    " %4d Buffered output slices %10d\n",
                    ci(self.m_num_out_buf_slices),
                    ci(self.m_ithick_out * self.m_iwidth * self.m_num_out_buf_slices)
                );
            }
            let _ = ImodFile::Stdout.flush();
            //
            // Prepare fixed maskEdges
            if self.m_if_alpha == 0 || self.m_mask_edges == 0 {
                self.mask_prep(self.m_islice_start);
            }
        }
        self.m_cur_out_buf_slice = 0;
        self.m_last_pre_write_line = -1;
        if self.m_debug != 0 {
            printf!("slicen %.1f\n", cf(self.m_center_slice as f64));
        }
        if self.m_if_alpha >= 0 {
            load_limit = self.m_islice_end;
        } else {
            load_limit = ((self.m_center_slice
                + (self.m_islice_end as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                + self.m_y_offset * self.m_sin_alpha[0]) as f64
                + 0.5 * self.m_ithick_out as f64 * abs_sin_alf as f64
                + 2.) as i32;
        }
        //
        // Main loop over slices perpendicular to tilt axis
        // ------------------------------------------------
        lslice_out = self.m_islice_start;
        while lslice_out <= self.m_islice_end {
            //
            // get limits for slices that are needed: the slice itself for regular
            // work, or required vertical slices for new-style X tilting
            //
            if self.m_debug != 0 {
                printf!("working on %d\n", ci(lslice_out));
            }
            if self.m_if_alpha >= 0 {
                lslice_start = lslice_out;
                lslice_end = lslice_out;
            } else {
                tan_alpha = self.m_sin_alpha[0] / self.m_cos_alpha[0];
                lslice_min = ((self.m_center_slice
                    + (lslice_out as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                    + self.m_y_offset * self.m_sin_alpha[0]) as f64
                    - 0.5 * self.m_ithick_out as f64 * abs_sin_alf as f64
                    - 1.) as i32;
                lslice_max = ((self.m_center_slice
                    + (lslice_out as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                    + self.m_y_offset * self.m_sin_alpha[0]) as f64
                    + 0.5 * self.m_ithick_out as f64 * abs_sin_alf as f64
                    + 2.) as i32;
                if self.m_debug != 0 {
                    printf!("need slices %d %d\n", ci(lslice_min), ci(lslice_max));
                }
                lslice_min = b3dmax!(1, lslice_min);
                lslice_max = b3dmin!(self.m_ny_proj, lslice_max);
                lslice_start = lslice_min;
                if lslice_min >= lvert_slice_start && lslice_min <= lvert_slice_end {
                    lslice_start = lvert_slice_end + 1;
                }
                lslice_end = lslice_max;
            }
            //
            // loop on needed vertical slices
            //
            if self.m_debug != 0 {
                printf!("looping to get %d %d\n", ci(lslice_start), ci(lslice_end));
            }

            for lslice in lslice_start..=lslice_end {
                shifted_gpu_load = false;
                //
                // Load stack with as many lines from projections as will
                // fit into the remaining space.
                //
                // Enter loading procedures unless the load is already set for the
                // current slice
                //
                if in_load_start == 0 || lslice > last_ready {
                    need_start = 0;
                    need_end = 0;
                    itry_end = 0;
                    last_ready = 0;
                    itry = lslice;
                    if_enough = 0;
                    //
                    // loop on successive output slices to find what input slices are
                    // needed for them; until all slices would be loaded or there
                    // would be no more room
                    while itry > 0
                        && itry <= self.m_ny_proj
                        && if_enough == 0
                        && itry <= load_limit
                        && itry_end - need_start + 1 <= self.m_num_planes
                    {
                        last_start =
                            self.m_needed_starts[(itry - self.m_ind_needed_base - 1) as usize];
                        last_end = self.m_needed_ends[(itry - self.m_ind_needed_base - 1) as usize];
                        //
                        // values here must work in single-slice case too
                        // if this is the first time, set needstart
                        // if this load still fits, set needend and lastready
                        //
                        itry_end = last_end;
                        if need_start == 0 {
                            need_start = last_start;
                        }
                        if itry_end - need_start + 1 <= self.m_num_planes {
                            need_end = itry_end;
                            last_ready = itry;
                        } else {
                            if_enough = 1;
                        }
                        itry = itry + 1;
                    }
                    if self.m_debug != 0 {
                        printf!(
                            "itryend %d, needstart %d, needend %d, lastready %d\n",
                            ci(itry_end),
                            ci(need_start),
                            ci(need_end),
                            ci(last_ready)
                        );
                    }
                    if need_end == 0 {
                        exit_error(b"Insufficient size for input array to do a single slice");
                    }
                    //
                    // if some are already loaded, need to shift them down
                    //
                    num_already = b3dmax!(0, in_load_end - need_start + 1);
                    if in_load_end == 0 {
                        num_already = 0;
                    }
                    ioffset = (need_start - in_load_start) as i64 * self.m_in_plane_size as i64;
                    for i in 0..num_already as i64 {
                        let dst = (i * self.m_in_plane_size as i64) as usize;
                        let src = (i * self.m_in_plane_size as i64 + ioffset) as usize;
                        self.m_input_array
                            .copy_within(src..src + self.m_in_plane_size as usize, dst);
                    }
                    if num_already != 0 && self.m_debug != 0 {
                        printf!("shifting %d %d\n", ci(need_start), ci(in_load_end));
                    }
                    //
                    // If it is also time to load something on GPU, shift existing
                    // data if appropriate and enable copy of filtered data
                    if self.m_num_gpu_planes > 0 {
                        if self.m_load_gpu_start <= 0
                            || self.m_needed_ends
                                [(lslice_out - self.m_ind_needed_base - 1) as usize]
                                > self.m_load_gpu_end
                        {
                            self.shift_gpu_setup_copy(
                                lslice_out,
                                need_end,
                                &mut need_gpu_start,
                                &mut need_gpu_end,
                                &mut keep_on_gpu,
                                &mut num_load_gpu,
                                &mut shifted_gpu_load,
                            );
                        }
                    }
                    //
                    // load the planes in one plane high if stretching
                    //
                    ibase = 1 + num_already * self.m_in_plane_size + self.m_ip_extra_size;
                    lread_start = need_start + num_already;
                    lpos_start = b3dmax!(1, lread_start);
                    lpos_end = b3dmin!(need_end, self.m_ny_proj);
                    //
                    if self.m_debug != 0 {
                        printf!("loading %d %d\n", ci(lread_start), ci(need_end));
                    }
                    if self.m_use_raw_stack != 0 {
                        self.load_and_filter_raw_data(ibase, lread_start, need_end);
                    } else if !self.m_rec_reproj {
                        let mut input = std::mem::take(&mut self.m_input_array);
                        let mut load_buf = std::mem::take(&mut self.m_load_buffer);
                        nv = 1;
                        while nv <= self.m_num_views {
                            istart = ibase;

                            // Read from projection NV, lines lposStart to lposEnd
                            if self.m_load_xoffset >= 0 {
                                unsafe {
                                    iiu_set_position(
                                        1,
                                        self.m_map_used_view[(nv - 1) as usize] - 1,
                                        0,
                                    );
                                    nl = iiu_read_sec_part(
                                        1,
                                        load_buf.as_mut_ptr().cast(),
                                        self.m_nx_proj,
                                        self.m_load_xoffset,
                                        self.m_load_xoffset + self.m_nx_proj - 1,
                                        lpos_start - 1,
                                        lpos_end - 1,
                                    );
                                }
                            } else {
                                unsafe {
                                    iiu_set_position(
                                        1,
                                        self.m_map_used_view[(nv - 1) as usize] - 1,
                                        lpos_start - 1,
                                    );
                                    nl = iiu_read_lines(
                                        1,
                                        load_buf.as_mut_ptr().cast(),
                                        lpos_end + 1 - lpos_start,
                                    );
                                }
                            }
                            if nl != 0 {
                                exit_error_fmt!(
                                    format920,
                                    ci(self.m_map_used_view[(nv - 1) as usize]),
                                    ci(lpos_start)
                                );
                            }

                            // Distribute lines into the planes of the input array
                            nl = lread_start;
                            while nl <= need_end {
                                iyload = b3dmax!(0, b3dmin!(self.m_ny_proj - 1, nl - 1));
                                ind_buf = (iyload - (lpos_start - 1)) * self.m_nx_proj;

                                // Take log if requested
                                self.take_log_or_weight_line(
                                    nv,
                                    Some(&load_buf),
                                    &mut ind_buf,
                                    &mut input,
                                    (istart - 1) as usize,
                                    1,
                                );
                                //
                                // pad with taper between start and end of line
                                self.taper_end_to_start(&mut input[(istart - 1) as usize..]);
                                //
                                istart = istart + self.m_in_plane_size;
                                nl += 1;
                            }
                            ibase = ibase + self.m_nx_pad_dim;
                            nv += 1;
                        }
                        self.m_input_array = input;
                        self.m_load_buffer = load_buf;
                        //
                    } else {
                        //
                        // Load reconstruction for projections
                        nl = lread_start;
                        while nl <= need_end {
                            let err = unsafe {
                                iiu_set_position(3, nl - 1, 0);
                                iiu_read_sec_part(
                                    3,
                                    self.m_input_array[(ibase - 1) as usize..]
                                        .as_mut_ptr()
                                        .cast(),
                                    self.m_max_xload + 1 - self.m_min_xload,
                                    self.m_min_xload - 1,
                                    self.m_max_xload - 1,
                                    self.m_min_yreproj - 1,
                                    self.m_max_yreproj - 1,
                                )
                            };
                            if err != 0 {
                                exit_error_fmt!(
                                    format920,
                                    ci(self
                                        .m_map_used_view
                                        .get((nv - 1).max(0) as usize)
                                        .copied()
                                        .unwrap_or(0)),
                                    ci(nl)
                                );
                            }

                            //
                            // Undo the scaling that was used to write, and apply a
                            // scaling that will make the data close for projecting
                            let mut indv = ibase as i64;
                            while indv <= ibase as i64 + self.m_islice_size_bp - 1 {
                                let v = &mut self.m_input_array[(indv - 1) as usize];
                                *v = (*v / self.m_out_scale - self.m_out_add) / self.m_filter_scale;
                                indv += 1;
                            }
                            ibase = ibase + self.m_in_plane_size;
                            nl += 1;
                        }
                    }
                    if !self.m_rec_reproj && self.m_num_sirt_iter <= 0 && self.m_use_raw_stack == 0
                    {
                        ibase = num_already * self.m_in_plane_size;
                        let mut input = std::mem::take(&mut self.m_input_array);
                        nl = lread_start;
                        while nl <= need_end {
                            self.transform(&mut input[ibase as usize..], nl, 1);
                            ibase = ibase + self.m_in_plane_size;
                            nl += 1;
                        }
                        self.m_input_array = input;
                    }
                    in_load_start = need_start;
                    in_load_end = need_end;
                    self.m_ind_load_end =
                        1 + self.m_in_plane_size * ((in_load_end - in_load_start) + 1) - 1;
                }
                //
                // Stack is full.  Now check if GPU needs to be loaded
                if self.m_num_gpu_planes > 0 {
                    if self.m_load_gpu_start <= 0
                        || self.m_needed_ends[(lslice_out - self.m_ind_needed_base - 1) as usize]
                            > self.m_load_gpu_end
                    {
                        //
                        // Load as much as possible.  If the shift fails the first time
                        // it will set loadGpuStart to 0, call again to recompute for
                        // full load
                        if !shifted_gpu_load {
                            self.shift_gpu_setup_copy(
                                lslice_out,
                                need_end,
                                &mut need_gpu_start,
                                &mut need_gpu_end,
                                &mut keep_on_gpu,
                                &mut num_load_gpu,
                                &mut shifted_gpu_load,
                            );
                        }
                        if !shifted_gpu_load {
                            self.shift_gpu_setup_copy(
                                lslice_out,
                                need_end,
                                &mut need_gpu_start,
                                &mut need_gpu_end,
                                &mut keep_on_gpu,
                                &mut num_load_gpu,
                                &mut shifted_gpu_load,
                            );
                        }
                        ibase = 1
                            + (need_gpu_start + keep_on_gpu - in_load_start) * self.m_in_plane_size;
                        if self.m_debug != 0 {
                            printf!(
                                "Loading GPU, # %d  lstart %d  start pos base %d  %.0f\n",
                                ci(num_load_gpu),
                                ci(need_gpu_start + keep_on_gpu),
                                ci(keep_on_gpu + 1),
                                cf((ibase - 1) as f64)
                            );
                        }
                        time_start = wall_time();
                        j = need_gpu_start + keep_on_gpu;
                        k = keep_on_gpu + 1;
                        if gpu_load_proj(
                            &self.m_input_array[(ibase - 1) as usize..],
                            num_load_gpu,
                            j,
                            k,
                        ) == 0
                        {
                            self.m_load_gpu_start = need_gpu_start;
                            self.m_load_gpu_end = need_gpu_end;
                        } else {
                            self.m_load_gpu_start = 0;
                        }
                        if self.m_debug != 0 {
                            printf!("Loading time %.4f\n", cf(wall_time() - time_start));
                        }
                    }
                }
                //
                istart = 1 + self.m_in_plane_size * (lslice - in_load_start);
                if !self.m_rec_reproj {
                    //
                    // Backprojection: Process all views for current slice
                    // If new-style X tilt, set  the Y center based on the slice
                    // number, and adjust the y offset slightly
                    //
                    if self.m_if_alpha < 0 {
                        self.m_ycen_out = (ycen_fix as f64
                            + (1. / self.m_cos_alpha[0] as f64 - 1.) * self.m_y_offset as f64
                            - b3dnint!(tan_alpha * (lslice as f32 - self.m_center_slice)) as f64)
                            as f32;
                    }
                    if self.m_num_sirt_iter == 0 {
                        //
                        // PRINT3(1, lslice, inLoadStart);
                        // print *,'projecting', lslice, ' at', istart, ', ycen =', ycenOut
                        let input = std::mem::take(&mut self.m_input_array);
                        self.project(&input, istart, lslice);
                        self.m_input_array = input;
                        if lslice == 3000 {
                            let mut data = crate::imod::libcfshr::islice::MrcData::F(
                                std::mem::take(&mut self.m_out_slice_arr),
                            );
                            crate::imod::libiimod::mrcslice::mrc_write_image_to_file(
                                "mask3000.mrc",
                                &mut data,
                                2,
                                self.m_iwidth,
                                self.m_ithick_bp,
                            );
                            if let crate::imod::libcfshr::islice::MrcData::F(v) = data {
                                self.m_out_slice_arr = v;
                            }
                        }
                        //
                        // move vertical slice into ring buffer, adjust ring variables
                        //
                        if self.m_if_alpha < 0 {
                            // print *,'moving slice to ring position', nextfreevs
                            ioffset = (next_free_vert_slice - 1) as i64
                                * self.m_ithick_bp as i64
                                * self.m_iwidth as i64;
                            let n = (self.m_ithick_bp * self.m_iwidth) as usize;
                            let off = ioffset as usize;
                            self.m_vert_slice_arr[off..off + n]
                                .copy_from_slice(&self.m_out_slice_arr[..n]);
                            self.manage_ring(
                                self.m_num_vert_needed,
                                &mut num_vert_slice_in_ring,
                                &mut next_free_vert_slice,
                                &mut lvert_slice_start,
                                &mut lvert_slice_end,
                                lslice,
                            );
                        }
                    } else {
                        //
                        // for SIRT, either do the zero iteration here, load single slice
                        // into read-in slice spot, or decompose vertical slice from read-
                        // in slices into spot in  vertical slice ring buffer.
                        // Descale them by output scaling only
                        // Set up for starting from read-in data
                        rec_scale = self.m_filter_scale * self.m_num_views as f32;
                        iset = 1;
                        reproj_fill = self.m_dmean_in;
                        // `sirtArray` points at `mReadInArray` or `mVertSliceArr`; it
                        // is lent out of the instance for the rest of this slice.
                        let sirt_is_vert: bool;
                        let mut sirt_array: Vec<f32>;
                        if self.m_sirt_from_zero {
                            //
                            // Doing zero iteration: set up location to place slice
                            sirt_is_vert = self.m_if_alpha < 0;
                            ibase_sirt = 1;
                            if self.m_if_alpha < 0 {
                                ibase_sirt = 1
                                    + (next_free_vert_slice - 1) * self.m_ithick_bp * self.m_iwidth;
                                self.manage_ring(
                                    self.m_num_vert_needed,
                                    &mut num_vert_slice_in_ring,
                                    &mut next_free_vert_slice,
                                    &mut lvert_slice_start,
                                    &mut lvert_slice_end,
                                    lslice,
                                );
                            }
                            //
                            // Copy and filter the lines needed by filter set 1
                            for iv in 1..=self.m_num_views {
                                j = (iv - 1) * self.m_nx_pad_dim;
                                k = istart + (iv - 1) * self.m_nx_pad_dim;
                                for i in 0..=self.m_nx_pad_dim - 3 {
                                    self.m_work_array[(j + i) as usize] =
                                        self.m_input_array[(k + i - 1) as usize];
                                }
                            }
                            let mut work = std::mem::take(&mut self.m_work_array);
                            self.transform(&mut work, lslice, 1);
                            self.m_edge_fill = edge_fill_orig;
                            //
                            // Backproject and maskEdges/taper edges
                            self.project(&work, 1, lslice);
                            self.m_work_array = work;
                            if self.m_mask_edges != 0 {
                                let mut out = std::mem::take(&mut self.m_out_slice_arr);
                                self.mask_slice(&mut out, self.m_ithick_bp);
                                self.m_out_slice_arr = out;
                            }
                            sirt_array = if sirt_is_vert {
                                std::mem::take(&mut self.m_vert_slice_arr)
                            } else {
                                std::mem::take(&mut self.m_read_in_array)
                            };
                            for indv in 0..self.m_islice_size_bp as usize {
                                sirt_array[indv + (ibase_sirt - 1) as usize] =
                                    self.m_out_slice_arr[indv] / rec_scale;
                            }
                            iset = 2;
                            if self.m_if_out_sirt_rec == 3 {
                                printf!("writing zero iteration slice %d\n", ci(lslice));
                                self.write_internal_slice(
                                    5,
                                    &sirt_array[(ibase_sirt - 1) as usize..],
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                            }
                        } else if self.m_if_alpha == 0 {
                            //
                            // Read in single slice
                            sirt_is_vert = false;
                            let err = unsafe {
                                iiu_set_position(3, lslice - 1, 0);
                                iiu_read_section(3, self.m_read_in_array.as_mut_ptr().cast())
                            };
                            if err != 0 {
                                exit_error_fmt!(
                                    format920,
                                    ci(self
                                        .m_map_used_view
                                        .get((nv - 1).max(0) as usize)
                                        .copied()
                                        .unwrap_or(0)),
                                    ci(nl)
                                );
                            }
                            dsum = 0.;
                            dpix = 0.;
                            for i in 0..=(self.m_ithick_out * self.m_iwidth - 1) as usize {
                                self.m_read_in_array[i] =
                                    self.m_read_in_array[i] / self.m_out_scale - self.m_out_add;
                                dsum = dsum + self.m_read_in_array[i] as f64;
                                dpix = dpix + 1.;
                            }
                            if self.m_debug != 0 {
                                printf!(
                                    "Loaded mean= %f   pmean= %f\n",
                                    cf(dsum / dpix),
                                    cf(self.m_dmean_in as f64)
                                );
                            }
                            ibase_sirt = 1;
                            sirt_array = std::mem::take(&mut self.m_read_in_array);
                        } else {
                            //
                            // Vertical slices: read file directly if it is vertical slices
                            sirt_is_vert = true;
                            ibase_sirt =
                                1 + (next_free_vert_slice - 1) * self.m_ithick_bp * self.m_iwidth;
                            sirt_array = std::mem::take(&mut self.m_vert_slice_arr);
                            if self.m_vert_sirt_input != 0 {
                                if self.m_debug != 0 {
                                    printf!("loading vertical slice %d\n", ci(lslice));
                                }
                                let err = unsafe {
                                    iiu_set_position(3, lslice - 1, 0);
                                    iiu_read_section(
                                        3,
                                        sirt_array[(ibase_sirt - 1) as usize..].as_mut_ptr().cast(),
                                    )
                                };
                                if err != 0 {
                                    exit_error_fmt!(
                                        format920,
                                        ci(self
                                            .m_map_used_view
                                            .get((nv - 1).max(0) as usize)
                                            .copied()
                                            .unwrap_or(0)),
                                        ci(nl)
                                    );
                                }
                            } else {
                                //
                                // Decompose vertical slice from read-in slices
                                // First see if necessary slices are loaded in ring
                                vert_slice_cen = lslice as f32 - self.m_center_slice;
                                vert_ycen_fix = ((self.m_ithick_bp / 2) as f64 + 0.5
                                    - b3dnint!(tan_alpha * (lslice as f32 - self.m_center_slice))
                                        as f64
                                    + (self.m_y_offset / self.m_cos_alpha[0]) as f64)
                                    as f32;
                                ri_bot = self.m_center_slice
                                    + vert_slice_cen * self.m_cos_alpha[0]
                                    + (1. - vert_ycen_fix) * self.m_sin_alpha[0];
                                ri_top =
                                    ri_bot + (self.m_ithick_bp - 1) as f32 * self.m_sin_alpha[0];
                                lri_min = b3dmax!(1, (b3dmin!(ri_bot, ri_top)).floor() as i32);
                                lri_max = b3dmin!(
                                    self.m_ny_proj,
                                    (b3dmax!(ri_bot, ri_top)).ceil() as i32
                                );

                                load_start = lri_min;
                                if lri_min >= l_read_start && lri_min <= l_read_end {
                                    load_start = l_read_end + 1;
                                }
                                load_end = lri_max;
                                //
                                // Read into ring and manage the ring pointers
                                if self.m_debug != 0 && load_end >= load_start {
                                    printf!(
                                        "reading into ring %d %d\n",
                                        ci(load_start),
                                        ci(load_end)
                                    );
                                }
                                for lri in load_start..=load_end {
                                    ioffset = (next_read_free - 1) as i64
                                        * self.m_ithick_out as i64
                                        * self.m_iwidth as i64;
                                    let err = unsafe {
                                        iiu_set_position(3, lri - 1, 0);
                                        iiu_read_section(
                                            3,
                                            self.m_read_in_array[ioffset as usize..]
                                                .as_mut_ptr()
                                                .cast(),
                                        )
                                    };
                                    if err != 0 {
                                        exit_error_fmt!(
                                            format920,
                                            ci(self
                                                .m_map_used_view
                                                .get((nv - 1).max(0) as usize)
                                                .copied()
                                                .unwrap_or(0)),
                                            ci(nl)
                                        );
                                    }
                                    for i in 0..(self.m_ithick_out * self.m_iwidth) as usize {
                                        let v = &mut self.m_read_in_array[i + ioffset as usize];
                                        *v = *v / self.m_out_scale - self.m_out_add;
                                    }
                                    self.manage_ring(
                                        self.m_num_read_need,
                                        &mut num_read_in_ring,
                                        &mut next_read_free,
                                        &mut l_read_start,
                                        &mut l_read_end,
                                        lri,
                                    );
                                }
                                iring_start = 1;
                                if num_read_in_ring == self.m_num_read_need {
                                    iring_start = next_read_free;
                                }
                                self.decompose(
                                    lslice,
                                    l_read_start,
                                    l_read_end,
                                    iring_start,
                                    &mut sirt_array[(ibase_sirt - 1) as usize..],
                                );
                            }
                            if self.m_if_out_sirt_rec == 4 {
                                printf!("writing decomposed slice %d\n", ci(lslice));
                                self.write_internal_slice(
                                    5,
                                    &sirt_array[(ibase_sirt - 1) as usize..],
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                            }
                            self.manage_ring(
                                self.m_num_vert_needed,
                                &mut num_vert_slice_in_ring,
                                &mut next_free_vert_slice,
                                &mut lvert_slice_start,
                                &mut lvert_slice_end,
                                lslice,
                            );
                        }
                        let sb = (ibase_sirt - 1) as usize;
                        //
                        // Ready to iterate.  First get the starting slice mean
                        array_min_max_mean(
                            &sirt_array[sb..],
                            self.m_iwidth,
                            self.m_ithick_bp,
                            0,
                            self.m_iwidth - 1,
                            0,
                            self.m_ithick_bp - 1,
                            &mut unscaled_min,
                            &mut unscaled_max,
                            &mut first_mean,
                        );
                        // It seems to give less edge artifacts using the mean all the time
                        // if (sirtFromZero) rpfill = firstmean
                        reproj_fill = first_mean;
                        //
                        let mut work = std::mem::take(&mut self.m_work_array);
                        for isirt_iter in 1..=self.m_num_sirt_iter {
                            ierr = 1;
                            time_start = wall_time();

                            if self.m_use_gpu {
                                ierr = gpu_reproj_one_slice(
                                    &sirt_array[sb..],
                                    &mut work,
                                    &self.m_sin_reproj,
                                    &self.m_cos_reproj,
                                    self.m_ycen_out,
                                    self.m_num_views,
                                    reproj_fill,
                                );
                            }
                            if ierr != 0 {
                                for iv in 1..=self.m_num_views {
                                    j = (iv - 1) * self.m_nx_pad_dim;
                                    // call reproject(array(ibaseSIRT), iwidth, ithickBP, iwidth, &
                                    // sinReproj(iv), cosReproj(iv), xRayStart(i), yRayStart(i), &
                                    // numPixInRay(i), maxRayPixels(iv), dmeanIn, array(j), 1, 1)
                                    let cos_r = self.m_cos_reproj[(iv - 1) as usize];
                                    let sin_r = self.m_sin_reproj[(iv - 1) as usize];
                                    self.reproj_one_angle(
                                        &sirt_array[sb..],
                                        &mut work[j as usize..],
                                        1,
                                        1,
                                        1,
                                        cos_r,
                                        -sin_r,
                                        1.,
                                        0.,
                                        cos_r,
                                        self.m_iwidth,
                                        self.m_ithick_bp,
                                        self.m_iwidth * self.m_ithick_bp,
                                        self.m_iwidth,
                                        1,
                                        1,
                                        0.,
                                        0.,
                                        ((self.m_iwidth / 2) as f64 + 0.5) as f32,
                                        self.m_ycen_out,
                                        ((self.m_iwidth / 2) as f64 + 0.5) as f32,
                                        self.m_center_slice,
                                        0,
                                        0.,
                                        0.,
                                        reproj_fill,
                                    );
                                }
                            }
                            if self.m_debug != 0 {
                                printf!("Reproj time = %.4f\n", cf(wall_time() - time_start));
                            }
                            if isirt_iter == self.m_num_sirt_iter && self.m_if_out_sirt_proj == 1 {
                                self.write_internal_slice(
                                    4,
                                    &work,
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                            }
                            //
                            // Subtract input projection lines from result.  No scaling
                            // needed here; these intensities should be in register
                            // Taper ends
                            // dsum = 0.
                            for iv in 1..=self.m_num_views {
                                j = (iv - 1) * self.m_nx_pad_dim;
                                k = istart + (iv - 1) * self.m_nx_pad_dim;
                                for i in 0..self.m_iwidth {
                                    work[(j + i) as usize] -=
                                        self.m_input_array[(k + i - 1) as usize];
                                }
                                self.taper_end_to_start(&mut work[j as usize..]);
                            }
                            if isirt_iter == self.m_num_sirt_iter && self.m_if_out_sirt_proj == 2 {
                                self.write_internal_slice(
                                    4,
                                    &work,
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                            }
                            //
                            // filter the working difference lines and get a rough value for
                            // the edgeFill.  Trying to get this better did not prevent edge
                            // artifacts - the masking was needed
                            self.transform(&mut work, lslice, iset);
                            dsum = 0.;
                            for iv in 1..=self.m_num_views {
                                j = (iv - 1) * self.m_nx_pad_dim;
                                dsum = dsum
                                    + work[(j + self.m_iwidth + self.m_num_pad / 2 - 1) as usize]
                                        as f64;
                            }
                            self.m_edge_fill = (dsum / self.m_num_views as f64) as f32;
                            // print *,'   (for diff bp) edgefill =', edgeFill
                            //
                            // Backproject the difference
                            self.project(&work, 1, lslice);
                            if isirt_iter == self.m_num_sirt_iter && self.m_if_out_sirt_rec == 1 {
                                for indv in 0..self.m_islice_size_bp as usize {
                                    self.m_out_slice_arr[indv] = (self.m_out_slice_arr[indv]
                                        + self.m_out_add * rec_scale)
                                        * (self.m_out_scale / rec_scale);
                                }
                                printf!("writing bp difference %d\n", ci(lslice));
                                self.write_internal_slice(
                                    5,
                                    &self.m_out_slice_arr,
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                                for indv in 0..self.m_islice_size_bp as usize {
                                    self.m_out_slice_arr[indv] = self.m_out_slice_arr[indv]
                                        / (self.m_out_scale / rec_scale)
                                        - self.m_out_add * rec_scale;
                                }
                            }
                            //
                            // Accumulate report values
                            if self.m_iter_for_report > 0 {
                                let out = std::mem::take(&mut self.m_out_slice_arr);
                                self.sample_for_report(
                                    &out,
                                    lslice,
                                    self.m_ithick_bp,
                                    isirt_iter,
                                    self.m_out_scale / rec_scale,
                                    self.m_out_add * rec_scale,
                                );
                                self.m_out_slice_arr = out;
                            }
                            //
                            // Subtract from input slice
                            // scale to account for difference between ordinary scaling
                            // for output by 1 / numViews and the fact that input slice was
                            // descaled by 1/filterScale
                            if self.m_isign_constraint == 0 {
                                for indv in 0..self.m_islice_size_bp as usize {
                                    sirt_array[sb + indv] -= self.m_out_slice_arr[indv] / rec_scale;
                                }
                            } else if self.m_isign_constraint < 0 {
                                for indv in 0..self.m_islice_size_bp as usize {
                                    sirt_array[sb + indv] = b3dmin!(
                                        0.,
                                        (sirt_array[sb + indv]
                                            - self.m_out_slice_arr[indv] / rec_scale)
                                            as f64
                                    )
                                        as f32;
                                }
                            } else {
                                for indv in 0..self.m_islice_size_bp as usize {
                                    sirt_array[sb + indv] = b3dmax!(
                                        0.,
                                        (sirt_array[sb + indv]
                                            - self.m_out_slice_arr[indv] / rec_scale)
                                            as f64
                                    )
                                        as f32;
                                }
                            }
                            if self.m_mask_edges != 0 {
                                self.mask_slice(&mut sirt_array[sb..], self.m_ithick_bp);
                            }
                            if isirt_iter == self.m_num_sirt_iter && self.m_if_out_sirt_rec == 2 {
                                printf!("writing internal slice %d\n", ci(lslice));
                                self.write_internal_slice(
                                    5,
                                    &sirt_array[sb..],
                                    lslice,
                                    self.m_nx_pad_dim,
                                    &mut dmin4,
                                    &mut dmax4,
                                    &mut dmin5,
                                    &mut dmax5,
                                    &mut nz5,
                                );
                            }
                            if isirt_iter == self.m_num_sirt_iter
                                && self.m_save_vert_slices
                                && lslice >= self.m_islice_start
                                && lslice <= self.m_islice_end
                            {
                                if self.m_debug != 0 {
                                    printf!("writing vertical slice %d\n", ci(lslice));
                                }
                                par_wrt_set_current(1);
                                array_min_max_mean(
                                    &sirt_array[sb..],
                                    self.m_iwidth,
                                    self.m_ithick_bp,
                                    0,
                                    self.m_iwidth - 1,
                                    0,
                                    self.m_ithick_bp - 1,
                                    &mut tmin,
                                    &mut tmax,
                                    &mut tmean,
                                );
                                dmin6 = b3dmin!(dmin6, tmin);
                                dmax6 = b3dmax!(dmax6, tmax);
                                unsafe {
                                    par_wrt_sec(6, sirt_array[sb..].as_mut_ptr().cast());
                                }
                                par_wrt_set_current(0);
                            }
                            //
                            // Adjust fill value by change in mean
                            // Accumulate mean of starting vertical slices until one output
                            // slice has been composed, and set composeFill from mean
                            if isirt_iter < self.m_num_sirt_iter || self.m_if_alpha < 0 {
                                array_min_max_mean(
                                    &sirt_array[sb..],
                                    self.m_iwidth,
                                    self.m_ithick_bp,
                                    0,
                                    self.m_iwidth - 1,
                                    0,
                                    self.m_ithick_bp - 1,
                                    &mut unscaled_min,
                                    &mut unscaled_max,
                                    &mut cur_mean,
                                );
                                //
                                // Doing it the same way with restarts as with going from 0 seems
                                // to prevent low frequency artifacts
                                // if (sirtFromZero) then
                                reproj_fill = cur_mean;
                                // else
                                // rpfill = dmeanIn + curmean - firstmean
                                // endif
                                if isirt_iter == self.m_num_sirt_iter && !composed_one {
                                    vert_sum = vert_sum + cur_mean;
                                    num_vert_sum = num_vert_sum + 1.;
                                }
                            }
                            if num_vert_sum > 0. {
                                compose_fill = vert_sum / num_vert_sum;
                            }
                        }
                        self.m_work_array = work;
                        if sirt_is_vert {
                            self.m_vert_slice_arr = sirt_array;
                        } else {
                            self.m_read_in_array = sirt_array;
                        }
                    }
                }
            }
            //
            if !self.m_rec_reproj {
                if self.m_if_alpha < 0 {
                    //
                    // interpolate output slice from vertical slices
                    //
                    iring_start = 1;
                    if num_vert_slice_in_ring == self.m_num_vert_needed {
                        iring_start = next_free_vert_slice;
                    }
                    if num_vert_slice_in_ring > 0 {
                        //
                        // If there is anything in ring to use, then we can compose an
                        // output slice, first dumping any deferred fill slices
                        self.dump_fill_slices(
                            &mut lfill_start,
                            lfill_end,
                            compose_fill,
                            &mut dmin,
                            &mut dmax,
                            &mut dtot8,
                            &mut num_slice_out,
                            &mut nxyz_tmp,
                        );
                        //PRINT4("composing", lsliceOut, lvertSliceStart, lvertSliceEnd);
                        self.compose(
                            lslice_out,
                            lvert_slice_start,
                            lvert_slice_end,
                            1,
                            iring_start,
                            compose_fill,
                        );
                        composed_one = true;
                    } else {
                        //
                        // But if there is nothing there, keep track of a range of slices
                        // that need to be filled - after the composeFill value is set
                        if lfill_start < 0 {
                            lfill_start = lslice_out;
                        }
                        lfill_end = lslice_out;
                    }
                }
                //
                // Dump slice unless there are fill slices being held
                if lfill_start < 0 {
                    self.dump_update_header(
                        lslice_out,
                        &mut dmin,
                        &mut dmax,
                        &mut dtot8,
                        &mut num_slice_out,
                        &mut nxyz_tmp,
                    );
                }
                lslice_out = lslice_out + 1;
            } else {
                //
                // REPROJECT all ready slices to minimize file mangling
                lslice_proj_end = last_ready;
                self.reproject_rec(
                    lslice_out,
                    &mut lslice_proj_end,
                    in_load_start,
                    in_load_end,
                    &mut dmin,
                    &mut dmax,
                    &mut dtot8,
                );
                lslice_out = lslice_proj_end + 1;
            }
        }
        //
        // End of main loop
        //-----------------
        //
        self.dump_fill_slices(
            &mut lfill_start,
            lfill_end,
            compose_fill,
            &mut dmin,
            &mut dmax,
            &mut dtot8,
            &mut num_slice_out,
            &mut nxyz_tmp,
        );
        // Close files
        unsafe { iiu_close(1) };
        pixel_tot = num_slices as f32 * self.m_iwidth as f32 * self.m_ithick_out as f32;
        if self.m_reproj_bp != 0 || self.m_rec_reproj {
            pixel_tot = num_slices as f32 * self.m_iwidth as f32 * self.m_num_reproj as f32;
        }
        dmean = (dtot8 / pixel_tot as f64) as f32;
        //
        // get numbers used to compute scaling report, and then fix min/max if
        // necessary
        unscaled_min = dmin / self.m_out_scale - self.m_out_add;
        unscaled_max = dmax / self.m_out_scale - self.m_out_add;
        truncations = false;
        extremes = false;
        if self.m_new_mode == 0 {
            truncations = dmin < 0. || dmax > 256.;
            dmin = b3dmax!(0., dmin as f64) as f32;
            dmax = b3dmin!(255., dmax as f64) as f32;
        } else if self.m_new_mode == 1 {
            extremes = self.m_use_gpu
                && b3dmax!(b3dabs!(dmin), b3dabs!(dmax)) as f64
                    > self.m_effective_scale as f64 * 3.0e5;
            truncations = (dmin as f64) < -32768. || dmax as f64 > 32768.;
            dmin = b3dmax!(-32768., dmin as f64) as f32;
            dmax = b3dmin!(32767., dmax as f64) as f32;
        } else if self.m_new_mode == 2 && write_16_bit_mode_for_floats() != 0 {
            truncations = (dmin as f64) < -MAX_HALF_FLOAT || dmax as f64 > MAX_HALF_FLOAT;
            dmin = if dmin as f64 > -MAX_HALF_FLOAT {
                dmin
            } else {
                -MAX_HALF_FLOAT as f32
            };
            dmax = if (dmax as f64) < MAX_HALF_FLOAT {
                dmax
            } else {
                MAX_HALF_FLOAT as f32
            };
        }
        if self.m_min_tot_slice <= 0 {
            if self.m_perpendicular != 0
                && self.m_interval_head_save > 0
                && !(self.m_reproj_bp != 0 || self.m_rec_reproj)
            {
                nxyz_tmp[0] = self.m_iwidth;
                nxyz_tmp[1] = self.m_ithick_out;
                nxyz_tmp[2] = num_slices;
                iiu_alt_size(2, &nxyz_tmp, &nxyzst);
            }
            iiu_write_header(2, &self.m_title, 1, dmin, dmax, dmean);
            let _ = iiu_ret_size(2);
            iiu_print_header(
                2,
                Some(if self.m_reproj_bp != 0 || self.m_rec_reproj {
                    "\nFinal reprojection file"
                } else {
                    "\nFinal reconstruction file"
                }),
            );
            if self.m_save_vert_slices {
                iiu_write_header(
                    6,
                    &self.m_title,
                    1,
                    dmin6,
                    dmax6,
                    ((dmin6 + dmax6) as f64 / 2.) as f32,
                );
            }
        } else {
            printf!(
                "Min, max, mean, # pixels= %15.7g %15.7g %15.7g %.0f\n",
                cf(dmin as f64),
                cf(dmax as f64),
                cf(dmean as f64),
                cf(pixel_tot as f64)
            );
        }
        if self.m_parallel_hdf != 0 {
            if unsafe { iiu_par_wrt_flush_buffers(2) } != 0 {
                exit_error(b"Finishing writing to output HDF file");
            }
            if self.m_save_vert_slices {
                par_wrt_set_current(1);
                if unsafe { iiu_par_wrt_flush_buffers(6) } != 0 {
                    exit_error(b"Finishing writing to vertical slice HDF file");
                }
                par_wrt_set_current(0);
            }
            par_wrt_close();
        }
        unsafe { iiu_close(2) };
        if self.m_save_vert_slices {
            unsafe { iiu_close(6) };
        }
        if !(self.m_reproj_bp != 0 || self.m_rec_reproj || self.m_num_sirt_iter > 0) {
            rec_scale =
                (self.m_num_views as f64 * 235. / (unscaled_max - unscaled_min) as f64) as f32;
            let rec_add1 = ((10. * (unscaled_max - unscaled_min) as f64 / 235.
                - unscaled_min as f64)
                / self.m_num_views as f64) as f32;
            printf!(
                "\n To scale output to bytes (10-245), use SCALE to add %.3f,  and scale by %.5g\n",
                cf(rec_add1 as f64),
                cf(rec_scale as f64)
            );
            rec_scale =
                (self.m_num_views as f64 * 30000. / (unscaled_max - unscaled_min) as f64) as f32;
            rec_add = ((-15000. * (unscaled_max - unscaled_min) as f64 / 30000.
                - unscaled_min as f64)
                / self.m_num_views as f64) as f32;
            printf!(
                "\n To scale output to -15000 to 15000, use SCALE to add %.3f,  and scale by %.5g\n",
                cf(rec_add as f64),
                cf(rec_scale as f64)
            );
            printf!("\n Reconstruction of %d slices complete.\n", ci(num_slices));
        }
        if extremes {
            printf!(
                "\nWARNING: Tilt - Extremely large values occurred and values were truncated when output to file; there could be errors in GPU computation.  run gputilttest\n"
            );
        }
        if truncations && !extremes {
            printf!(
                "\n WARNING: Tilt - Some values were truncated when output to the file; check the output density scaling factor\n"
            );
        }
        if self.m_use_gpu {
            gpu_done();
        }
        if self.m_iter_for_report > 0 {
            for i in 1..=b3dmax!(1, self.m_num_sirt_iter) {
                let b = (3 * (i - 1)) as usize;
                if self.m_report_vals[2 + b] > 0. {
                    printf!(
                        "\nIter %4d, slices %6d %6d, diff rec mean&sd: %15.3f %15.3f\n",
                        ci(i + self.m_iter_for_report - 1),
                        ci(self.m_islice_start),
                        ci(self.m_islice_end),
                        cf((self.m_report_vals[b] / self.m_report_vals[2 + b]) as f64),
                        cf((self.m_report_vals[1 + b] / self.m_report_vals[2 + b]) as f64)
                    );
                }
            }
        }
        if self.m_if_out_sirt_proj > 0 {
            iiu_write_header(
                4,
                &self.m_title,
                1,
                dmin4,
                dmax4,
                ((dmin4 + dmax4) as f64 / 2.) as f32,
            );
            unsafe { iiu_close(4) };
        }
        if self.m_if_out_sirt_rec > 0 {
            iiu_write_header(
                5,
                &self.m_title,
                1,
                dmin5,
                dmax5,
                ((dmin5 + dmax5) as f64 / 2.) as f32,
            );
            unsafe { iiu_close(5) };
        }
        c_exit(0);
    }
}

impl Tilt {
    /// `Tilt::shiftGpuSetupCopy` (`tilt.cpp:893`): determine parameters for
    /// loading the GPU and make the call to shift existing data if any is to be
    /// retained.
    fn shift_gpu_setup_copy(
        &mut self,
        lslice_out: i32,
        need_end: i32,
        need_gpu_start: &mut i32,
        need_gpu_end: &mut i32,
        keep_on_gpu: &mut i32,
        num_load_gpu: &mut i32,
        shifted_gpu_load: &mut bool,
    ) {
        let time_start: f64;
        let j: i32;
        let k: i32;
        *need_gpu_start = self.m_needed_starts[(lslice_out - self.m_ind_needed_base - 1) as usize];
        *need_gpu_end = b3dmin!(*need_gpu_start + self.m_num_gpu_planes - 1, need_end);
        *keep_on_gpu = 0;
        if self.m_load_gpu_start > 0 {
            *keep_on_gpu = self.m_load_gpu_end + 1 - *need_gpu_start;
        }
        *num_load_gpu = *need_gpu_end + 1 - *need_gpu_start - *keep_on_gpu;
        if self.m_debug != 0 {
            printf!(
                "Shifting GPU, # %d  lstart %d  start pos %d\n",
                ci(*num_load_gpu),
                ci(*need_gpu_start + *keep_on_gpu),
                ci(*keep_on_gpu + 1)
            );
        }
        time_start = wall_time();
        j = *need_gpu_start + *keep_on_gpu;
        k = *keep_on_gpu + 1;
        if gpu_shift_proj(*num_load_gpu, j, k) == 0 {
            *shifted_gpu_load = true;
        } else {
            self.m_load_gpu_start = 0;
        }
        if self.m_debug != 0 {
            printf!("Shifting time %.4f\n", cf(wall_time() - time_start));
        }
    }

    /// `Tilt::writeInternalSlice` (`tilt.cpp:922`): for writing two kinds of
    /// test output.  The source's parameter shadows `mNxPadDim`.
    fn write_internal_slice(
        &self,
        is_unit: i32,
        array: &[f32],
        lslice: i32,
        m_nx_pad_dim: i32,
        dmin4: &mut f32,
        dmax4: &mut f32,
        dmin5: &mut f32,
        dmax5: &mut f32,
        nz5: &mut i32,
    ) {
        let mut istmin: f32 = 0.;
        let mut istmax: f32 = 0.;
        let mut istmean: f32 = 0.;
        let mut iv: i32;
        let mut j: i32;
        let mut nxyz: [i32; 3] = [0; 3];
        let nxyzst: [i32; 3] = [0, 0, 0];
        let mut cell: [f32; 6] = [0., 0., 0., 90., 90., 90.];
        if is_unit == 4 {
            iv = 1;
            while iv <= self.m_num_views {
                j = (iv - 1) * m_nx_pad_dim;
                unsafe {
                    iiu_set_position(4, iv - 1, lslice - self.m_islice_start);
                    iiu_write_lines(4, array[j as usize..].as_ptr() as *mut _, 1);
                }
                iv += 1;
            }
            array_min_max_mean(
                array,
                m_nx_pad_dim,
                iv,
                0,
                self.m_iwidth - 1,
                0,
                self.m_num_views - 1,
                &mut istmin,
                &mut istmax,
                &mut istmean,
            );
            *dmin4 = b3dmin!(*dmin4, istmin);
            *dmax4 = b3dmax!(*dmax4, istmax);
        } else {
            unsafe {
                iiu_write_section(5, array.as_ptr() as *mut _);
            }
            array_min_max_mean(
                array,
                self.m_iwidth,
                self.m_ithick_bp,
                0,
                self.m_iwidth - 1,
                0,
                self.m_ithick_bp - 1,
                &mut istmin,
                &mut istmax,
                &mut istmean,
            );
            *dmin5 = b3dmin!(*dmin5, istmin);
            *dmax5 = b3dmax!(*dmax5, istmax);
            *nz5 = *nz5 + 1;
            nxyz[0] = self.m_iwidth;
            cell[0] = nxyz[0] as f32;
            nxyz[1] = self.m_ithick_bp;
            cell[1] = nxyz[1] as f32;
            nxyz[2] = *nz5;
            cell[2] = nxyz[2] as f32;
            iiu_alt_size(5, &nxyz, &nxyzst);
            iiu_alt_sample(5, &nxyz);
            iiu_alt_cell(5, &cell);
        }
    }

    /// `Tilt::dumpUpdateHeader` (`tilt.cpp:958`): dump a slice and update the
    /// header periodically if appropriate.
    fn dump_update_header(
        &mut self,
        lslice_dump: i32,
        dmin: &mut f32,
        dmax: &mut f32,
        dtot8: &mut f64,
        num_slice_out: &mut i32,
        nxyz_tmp: &mut [i32; 3],
    ) {
        let nxyzst: [i32; 3] = [0, 0, 0];
        let dmean: f32;
        //
        // Write out current slice
        self.dump_slice(lslice_dump, dmin, dmax, dtot8);
        // DNM 10/22/03:  Can't use flush in Windows/Intel because of sample.com
        // call flush(6)
        //
        // write out header periodically, restore writing position
        if self.m_perpendicular != 0
            && self.m_interval_head_save > 0
            && self.m_reproj_bp == 0
            && self.m_min_tot_slice <= 0
        {
            *num_slice_out = *num_slice_out + 1;
            nxyz_tmp[0] = self.m_iwidth;
            nxyz_tmp[1] = self.m_ithick_out;
            nxyz_tmp[2] = *num_slice_out;
            if (*num_slice_out % self.m_interval_head_save) == 1 {
                iiu_alt_size(2, nxyz_tmp, &nxyzst);
                dmean = (*dtot8
                    / (*num_slice_out as f32 * self.m_iwidth as f32 * self.m_ithick_out as f32)
                        as f64) as f32;
                iiu_write_header(2, &self.m_title, -1, *dmin, *dmax, dmean);
                unsafe { par_wrt_posn(2, *num_slice_out, 0) };
            }
        }
    }

    /// `Tilt::dumpFillSlices` (`tilt.cpp:986`): if there are fill slices that
    /// haven't been output yet, dump them now and turn off signal that they
    /// exist.
    fn dump_fill_slices(
        &mut self,
        lfill_start: &mut i32,
        lfill_end: i32,
        compose_fill: f32,
        dmin: &mut f32,
        dmax: &mut f32,
        dtot8: &mut f64,
        num_slice_out: &mut i32,
        nxyz_tmp: &mut [i32; 3],
    ) {
        if *lfill_start >= 0 {
            for i in *lfill_start..=lfill_end {
                for indv in 0..(self.m_iwidth * self.m_ithick_out) as usize {
                    self.m_out_slice_arr[indv] = compose_fill;
                }
                self.dump_update_header(i, dmin, dmax, dtot8, num_slice_out, nxyz_tmp);
            }
            *lfill_start = -1;
        }
    }

    /// `Tilt::taperEndToStart` (`tilt.cpp:1010`): taper intensities across pad
    /// region from end of line to start.  `numAverage` is fixed at 1, so the
    /// averaging branch never runs; it is kept as the source has it.
    fn taper_end_to_start(&self, array: &mut [f32]) {
        let mut xsum: f32;
        let start_mean: f32;
        let end_mean: f32;
        let mut f: f32;
        let mut nsum: i32;
        let num_average: i32 = 1;
        if num_average > 1 {
            nsum = 0;
            xsum = 0.;
            for ix in 0..=b3dmin!(num_average, self.m_nx_proj - 1) {
                nsum = nsum + 1;
                xsum = xsum + array[ix as usize];
            }
            start_mean = xsum / nsum as f32;
            nsum = 0;
            xsum = 0.;
            for ix in b3dmax!(0, self.m_nx_proj - num_average - 1)..self.m_nx_proj {
                nsum = nsum + 1;
                xsum = xsum + array[ix as usize];
            }
            end_mean = xsum / nsum as f32;
        } else {
            start_mean = array[0];
            end_mean = array[(self.m_nx_proj - 1) as usize];
        }
        for ipad in 1..=self.m_num_pad {
            f = (ipad as f64 / (self.m_num_pad as f64 + 1.)) as f32;
            array[(self.m_nx_proj + ipad - 1) as usize] =
                ((f * start_mean) as f64 + (1. - f as f64) * end_mean as f64) as f32;
        }
    }

    /// `Tilt::takeLogOrWeightLine` (`tilt.cpp:1043`): scale or take log of the
    /// data in a line for the given view (# from 1).  Can run forwards or
    /// backwards, in case copying with expansion into same array.
    ///
    /// `outBuf` is `out_buf[out_off..]`; `loadBuf` is `load_buf`, or `None`
    /// when the source passes the same array for both (`loadAndFilterRawData`
    /// after a rotation), in which case the values are read from `out_buf` in
    /// the same element order the source reads them.
    fn take_log_or_weight_line(
        &self,
        nview: i32,
        load_buf: Option<&[f32]>,
        ind_buf: &mut i32,
        out_buf: &mut [f32],
        out_off: usize,
        dir: i32,
    ) {
        let mut val: f32;
        let mut ix: i32 = if dir < 0 { self.m_nx_proj - 1 } else { 0 };
        let end: i32 = if dir < 0 { -1 } else { self.m_nx_proj };
        let inc: i32 = if dir < 0 { -1 } else { 1 };
        let ib = *ind_buf;
        if self.m_if_log != 0 {
            // 3/31/04: limit values to .001 time dynamic range
            while ix != end {
                let src = match load_buf {
                    Some(buf) => buf[(ib + ix) as usize],
                    None => out_buf[(ib + ix) as usize],
                };
                val = src + self.m_base_for_log;
                out_buf[out_off + ix as usize] = b3dmax!(self.m_val_min, val).log10();
                ix += inc;
            }
        } else {
            let weight = self.m_expose_weight[(nview - 1) as usize];
            while ix != end {
                let src = match load_buf {
                    Some(buf) => buf[(ib + ix) as usize],
                    None => out_buf[(ib + ix) as usize],
                };
                out_buf[out_off + ix as usize] = src * weight;
                ix += inc;
            }
        }
        *ind_buf += self.m_nx_proj;
    }

    /// `Tilt::loadAndFilterRawData` (`tilt.cpp:1069`): given the range of lines
    /// in the possibly rotated raw image data, determines what coordinates this
    /// corresponds to in the raw stack file, loads the subarea, rotates if
    /// necessary, then applies a rotated radial filter, and distributes the
    /// lines from a view into the input array.
    fn load_and_filter_raw_data(&mut self, mut ibase: i32, lread_start: i32, need_end: i32) {
        let mut raw_x0: i32;
        let mut raw_x1: i32;
        let mut raw_y0: i32;
        let mut raw_y1: i32;
        let mut istart: i32;
        let read_base: i32;
        let nx_raw: i32;
        let ny_raw: i32;
        let mut nx_rot: i32;
        let mut ny_rot: i32;
        let mut ix: i32;
        let mut iy: i32;
        let pad: i32;
        let mut ind_buf: i32;
        let mut ind: i32;
        let mut nl_use: i32;
        let format920 = "Reading in view %d for slice %d\n";
        let delx: f32;
        let dely: f32;
        let cos_ang: f32;
        let sin_ang: f32;
        let mut fx: f32;
        let mut fy: f32;
        let mut frac: f32;
        let mut xrot: f32;
        let mut fval: f32;
        let mut fl_ind: f32;
        let mut wall_start: f64;
        let mut wall_cum: f64 = 0.;

        // Get the coordinates in the raw input file by applying the inverse of the
        // operation
        raw_x0 = 0;
        raw_x1 = self.m_nx_proj - 1;
        if self.m_load_xoffset >= 0 {
            raw_x0 += self.m_load_xoffset;
            raw_x1 += self.m_load_xoffset;
        }
        raw_y0 = b3dmax!(0, lread_start - 1);
        raw_y1 = b3dmin!(need_end - 1, self.m_ny_proj - 1);
        read_base = b3dmax!(1, lread_start);
        ny_rot = raw_y1 + 1 - raw_y0;
        pad = (self.m_ny_raw_padded - ny_rot) / 2;
        if self.m_rot_flip_operation % 2 != 0 {
            if self.m_rot_flip_operation == 1 {
                raw_y0 = (self.m_nx_full_proj - 1) - raw_x1;
                raw_y1 = (self.m_nx_full_proj - 1) - raw_x0;
                raw_x0 = lread_start - 1;
                raw_x1 = need_end - 1;
            } else {
                raw_y0 = raw_x0;
                raw_y1 = raw_x1;
                raw_x0 = self.m_ny_proj - need_end;
                raw_x1 = self.m_ny_proj - lread_start;
            }
        } else if self.m_rot_flip_operation != 0 {
            raw_y0 = self.m_ny_proj - need_end;
            raw_y1 = self.m_ny_proj - lread_start;
            ix = (self.m_nx_full_proj - 1) - raw_x1;
            raw_x1 = (self.m_nx_full_proj - 1) - raw_x0;
            raw_x0 = ix;
        }
        nx_raw = raw_x1 + 1 - raw_x0;
        ny_raw = raw_y1 + 1 - raw_y0;

        // Make a map from the FFT to filter indexes: forward rotate by remaining
        // rotation and use the X position
        if self.s_need_map {
            delx = (1. / self.m_nx_pad_dim as f64) as f32;
            dely = (1. / self.m_ny_raw_padded as f64) as f32;
            cos_ang = (RADIANS_PER_DEGREE * self.m_raw_remaining_rot as f64).cos() as f32;
            sin_ang = (RADIANS_PER_DEGREE * self.m_raw_remaining_rot as f64).sin() as f32;
            for iy in 0..self.m_ny_raw_padded {
                fy = iy as f32 * dely;
                if fy as f64 > 0.5 {
                    fy -= 1.0f32;
                }
                for ix in 0..self.m_nx_pad_dim / 2 {
                    fx = ix as f32 * delx;
                    xrot = (fx * cos_ang - fy * sin_ang).abs();
                    self.m_raw_filt_map[(ix + iy * self.m_nx_pad_dim / 2) as usize] = b3dmin!(
                        self.m_nx_filt_dim as f64 - 1.01,
                        (xrot * self.m_nx_pad_dim as f32) as f64
                    )
                        as f32;
                }
            }
            //mrcWriteImageToFile("filtmap.mrc", mRawFiltMap, 2, mNxPadDim / 2, mNyRawPadded);
            self.s_need_map = false;
            if self.m_do_raw_filter_on_gpu && gpu_load_raw_filt_map(&self.m_raw_filt_map) != 0 {
                self.m_do_raw_filter_on_gpu = false;
            }
        }

        if self.m_debug != 0 {
            printf!(
                "reading %d  %d lines to %d rawY1 %d X %d to %d %d\n",
                ci(raw_y0),
                ci(ny_raw),
                ci(raw_y0 + ny_raw - 1),
                ci(raw_y1),
                ci(raw_x0),
                ci(raw_x1),
                ci(nx_raw)
            );
        }
        let mut load_buf = std::mem::take(&mut self.m_load_buffer);
        let mut rot_buf = std::mem::take(&mut self.m_rot_buffer);
        let mut input = std::mem::take(&mut self.m_input_array);
        for nv in 0..self.m_num_views {
            nx_rot = nx_raw;
            ny_rot = ny_raw;

            // Read in the data contiguously
            let err = unsafe {
                iiu_set_position(1, self.m_map_used_view[nv as usize] - 1, 0);
                iiu_read_sec_part(
                    1,
                    load_buf.as_mut_ptr().cast(),
                    nx_raw,
                    raw_x0,
                    raw_x1,
                    raw_y0,
                    raw_y1,
                )
            };
            if err != 0 {
                exit_error_fmt!(
                    format920,
                    ci(self.m_map_used_view[nv as usize]),
                    ci(lread_start)
                );
            }
            //if (nv == 32)
            //mrcWriteImageToFile("loaded.mrc", mLoadBuffer, 2, nxRaw, nyRaw);

            // Do the rotation; `rawBuf` is then `mRotBuffer` itself
            let raw_is_rot = self.m_rot_flip_operation != 0;
            if raw_is_rot {
                rotate_flip_image(
                    RotateFlipData::Float {
                        array: &load_buf,
                        brray: &mut rot_buf,
                    },
                    nx_raw,
                    ny_raw,
                    self.m_rot_flip_operation,
                    0,
                    0,
                    0,
                    &mut nx_rot,
                    &mut ny_rot,
                    0,
                );
                //if (nv == 32)
                //mrcWriteImageToFile("rotated.mrc", mRotBuffer, 2, nxRot, nyRot);
            }

            // Scale into padded array, keep it at left edge, and taper in X
            iy = ny_rot - 1;
            while iy >= 0 {
                ind_buf = iy * self.m_nx_proj;
                let row = (iy * self.m_nx_pad_dim) as usize;
                self.take_log_or_weight_line(
                    nv + 1,
                    if raw_is_rot { None } else { Some(&load_buf) },
                    &mut ind_buf,
                    &mut rot_buf,
                    row,
                    -1,
                );
                self.taper_end_to_start(&mut rot_buf[row..]);
                rot_buf[row + self.m_nx_pad_dim as usize - 1] = rot_buf[row];
                rot_buf[row + self.m_nx_pad_dim as usize - 2] = rot_buf[row];
                iy -= 1;
            }
            //if (nv == 32)
            //mrcWriteImageToFile("tapered.mrc", mRotBuffer, 2, mNxPadDim, nyRot);

            // Taper/pad in Y
            slice_taper_out_pad(
                PadIn::InPlace,
                SLICE_MODE_FLOAT,
                self.m_nx_pad_dim,
                ny_rot,
                &mut rot_buf,
                self.m_nx_pad_dim,
                self.m_nx_pad_dim,
                self.m_ny_raw_padded,
                0,
                0.,
            );
            //if (nv == 32)
            //mrcWriteImageToFile("padded.mrc", mRotBuffer, 2, mNxPadDim, mNyRawPadded);

            // FFT and multiply by filter
            wall_start = wall_time();
            if self.m_do_raw_filter_on_gpu && gpu_filter_raw_image(&mut rot_buf, nv) != 0 {
                self.m_do_raw_filter_on_gpu = false;
            }

            if !self.m_do_raw_filter_on_gpu {
                todfft_c(&mut rot_buf, self.m_nx_pad_dim - 2, self.m_ny_raw_padded, 0);
                let filt_base = (nv * self.m_nx_filt_dim) as usize;
                let mut buf_ptr = 0usize;
                for map_ptr in 0..(self.m_ny_raw_padded * self.m_nx_pad_dim / 2) as usize {
                    fl_ind = self.m_raw_filt_map[map_ptr];
                    ind = fl_ind as i32;
                    frac = fl_ind - ind as f32;
                    fval = ((1. - frac as f64)
                        * self.m_filter_array[filt_base + ind as usize] as f64
                        + (frac * self.m_filter_array[filt_base + ind as usize + 1]) as f64)
                        as f32;
                    rot_buf[buf_ptr] *= fval;
                    buf_ptr += 1;
                    rot_buf[buf_ptr] *= fval;
                    buf_ptr += 1;
                }
                todfft_c(&mut rot_buf, self.m_nx_pad_dim - 2, self.m_ny_raw_padded, 1);
            }
            wall_cum += wall_time() - wall_start;
            //if (nv == 32)
            //mrcWriteImageToFile("filtered.mrc", mRotBuffer, 2, mNxPadDim, mNyRawPadded);

            // Distribute lines in the input array for BP
            istart = ibase;
            for nl in lread_start..=need_end {
                nl_use = b3dmax!(1, b3dmin!(self.m_ny_proj, nl));
                iy = nl_use - read_base;
                let src = ((iy + pad) * self.m_nx_pad_dim) as usize;
                let n = (self.m_nx_pad_dim - 2) as usize;
                input[(istart - 1) as usize..(istart - 1) as usize + n]
                    .copy_from_slice(&rot_buf[src..src + n]);
                istart += self.m_in_plane_size;
            }
            ibase += self.m_nx_pad_dim;
        }
        self.m_load_buffer = load_buf;
        self.m_rot_buffer = rot_buf;
        self.m_input_array = input;
        if self.m_debug != 0 {
            printf!(
                "%sPU raw filter time %.5f\n",
                CArg::Str(if self.m_do_raw_filter_on_gpu {
                    "G"
                } else {
                    "C"
                }),
                cf(wall_cum)
            );
        }
    }

    /// `Tilt::manageRing` (`tilt.cpp:1220`): advance in the ring of vertical
    /// slices and maintain information about slices in ring.
    fn manage_ring(
        &self,
        num_vert_needed: i32,
        num_vert_slice_in_ring: &mut i32,
        next_free_vert_slice: &mut i32,
        lvert_slice_start: &mut i32,
        lvert_slice_end: &mut i32,
        lslice: i32,
    ) {
        if *num_vert_slice_in_ring < num_vert_needed {
            if *num_vert_slice_in_ring == 0 {
                *lvert_slice_start = lslice;
            }
            *num_vert_slice_in_ring = *num_vert_slice_in_ring + 1;
        } else {
            *lvert_slice_start = *lvert_slice_start + 1;
        }
        *lvert_slice_end = lslice;
        *next_free_vert_slice = *next_free_vert_slice + 1;
        if *next_free_vert_slice > num_vert_needed {
            *next_free_vert_slice = 1;
        }
    }
}

impl Tilt {
    /// `Tilt::radialWeights` (`tilt.cpp:1239`): set radial transform weighting.
    /// Default is linear ramp plus Gaussian fall off.
    fn radial_weights(
        &mut self,
        irad_max_in: i32,
        rad_fall_in: f32,
        ifilter_set: i32,
        if_mult_by_gaussian: i32,
    ) {
        let mut irad_end: i32;
        let irad_max: i32;
        let mut ind_base: i32 = 0;
        let mut irad_limit: i32;
        let mut imax: i32 = 0;
        let mut dup: i32;
        let mut iramp_end: i32;
        let mut n_filt_xyz: [i32; 3] = [0; 3];
        let mut mxyz: [i32; 3] = [0; 3];
        let mut mode: i32 = 0;
        let match_start: i32;
        let match_end: i32;
        let stretch: f32;
        let mut avg_interval: f32;
        let mut atten: f32;
        let mut sum_interval: f32;
        let mut wsum: f32;
        let mut z: f32;
        let mut arg: f32;
        let sirt_frac: f32;
        let mut zmax: f32;
        let mut freq: f32;
        let mut diff_min: f32;
        let mut diff: f32;
        let mut atten_sum: f32;
        let mut tab_fac: f32;
        let mut sin_diff: f32;
        let mut fact: f32;
        let mut exact_reach_top_mean: f32;
        let fake_alpha: f32;
        let mut fake_iter_use: f32;
        let fake_match_add: f32;
        let rad_fall: f32;
        let mut tmin: f32 = 0.;
        let mut tmax: f32 = 0.;
        let mut tmean: f32 = 0.;
        let mut read_in_scale: f32 = 0.;
        let mut wgt_atten: Vec<f32>;
        let deg_to_rad: f32 = RADIANS_PER_DEGREE as f32;

        //
        wgt_atten = vec![0.; self.m_lim_view as usize];
        sirt_frac = 0.99;
        //
        // Open a filter file if any
        if let Some(filter_file) = self.m_filter_file.clone() {
            unsafe {
                iiu_open(9, &String::from_utf8_lossy(&filter_file), "OLD");
                iiu_ret_basic_head(
                    9,
                    n_filt_xyz.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
            }
            if n_filt_xyz[1] != 1 || n_filt_xyz[2] != self.m_num_views {
                exit_error_fmt!(
                    "The filter file is %d in Y and %d in Z and it needs to be 1 and %d",
                    ci(n_filt_xyz[1]),
                    ci(n_filt_xyz[2]),
                    ci(self.m_num_views)
                );
            }
            let iw = (self.m_nx_proj + self.m_num_pad) / 2;
            if n_filt_xyz[0] < self.m_nx_pad_dim / 2 || n_filt_xyz[0] % iw > 1 {
                exit_error_fmt!(
                    "The filter file is %d in X and needs to be at least %d and a multiple of %d within one pixel",
                    ci(n_filt_xyz[0]),
                    ci(self.m_nx_pad_dim / 2),
                    ci(iw)
                );
            }

            // Read them all in now to determine a global scaling
            match_start = b3dmin!(3, self.m_nx_pad_dim / 2 - 1);
            match_end = b3dmax!(
                b3dmin!(5, self.m_nx_pad_dim / 2 - 1),
                self.m_nx_pad_dim / 20
            );
            wsum = 0.;
            for iv in 1..=self.m_num_views {
                ind_base = (iv - 1 + (ifilter_set - 1) * self.m_num_views) * self.m_nx_pad_dim;
                let err = unsafe {
                    iiu_read_sec_part(
                        9,
                        self.m_filter_array[ind_base as usize..].as_mut_ptr().cast(),
                        self.m_nx_pad_dim,
                        0,
                        self.m_nx_pad_dim / 2 - 1,
                        0,
                        0,
                    )
                };
                if err != 0 {
                    exit_error_fmt!("Reading section %d of filter file", ci(iv - 1));
                }
                for iw in match_start..=match_end {
                    wsum += iw as f32 / self.m_filter_array[(ind_base + iw) as usize];
                }
            }
            read_in_scale = wsum / ((match_end + 1 - match_start) * self.m_num_views) as f32;
            unsafe { iiu_close(9) };
        }
        //
        // Empirically determined parameters for fake SIRT.  The alpha value is good for
        // low iteration numbers but then iteration needs to be scaled to match SIRT
        fake_match_add = 0.3;
        fake_alpha = 0.00195;
        fake_iter_use = self.m_num_fake_sirt_iter as f32;
        if self.m_num_fake_sirt_iter > 15 {
            fake_iter_use = (15. + 0.8 * (self.m_num_fake_sirt_iter - 15) as f64) as f32;
        }
        if self.m_num_fake_sirt_iter > 30 {
            fake_iter_use = (27. + 0.6 * (self.m_num_fake_sirt_iter - 30) as f64) as f32;
        }

        // Scale index to duplicate elements for real and complex components; go to
        // Nyquist
        dup = 2;
        irad_end = self.m_nx_pad_dim / 2;

        // But if using raw stack, do not duplicate elements; fill array, go above
        // Nyquist
        if self.m_use_raw_stack != 0 {
            dup = 1;
            irad_end = self.m_nx_filt_dim;
        }

        // Adjust the entered indexes for the padding
        stretch = (self.m_nx_proj + self.m_num_pad) as f32 / self.m_nx_proj as f32;
        irad_max = b3dnint!(irad_max_in as f32 * stretch);
        rad_fall = rad_fall_in * stretch;
        //
        // Compute the ramp to the start of the falloff by default, or all the way out
        // if multiplying by Gaussian
        iramp_end = b3dmin!(irad_max, irad_end);
        if if_mult_by_gaussian > 0 {
            iramp_end = irad_end;
        }
        avg_interval = 1.;
        atten_sum = 0.;
        self.m_zero_weight = 0.;
        exact_reach_top_mean = 0.;
        if self.m_num_tilt_inc_wgt > 0 && self.m_num_wgt_angles > 1 {
            avg_interval = (self.m_wgt_angles[(self.m_num_wgt_angles - 1) as usize]
                - self.m_wgt_angles[0])
                / (self.m_num_wgt_angles - 1) as f32;
            if self.m_debug != 0 {
                printf!("\n View  Angle Weighting\n");
            }
        }
        //
        // Set up the attenuations for the weighting angles
        for iv in 1..=self.m_num_wgt_angles {
            atten = 1.;
            if self.m_num_tilt_inc_wgt as f64 > 0. && self.m_num_wgt_angles > 1 {
                sum_interval = 0.;
                wsum = 0.;
                for iw in 1..=self.m_num_tilt_inc_wgt {
                    if iv - iw > 0 {
                        wsum = wsum + self.m_tilt_inc_wgts[(iw - 1) as usize];
                        sum_interval = sum_interval
                            + self.m_tilt_inc_wgts[(iw - 1) as usize]
                                * (self.m_wgt_angles[(iv - iw) as usize]
                                    - self.m_wgt_angles[(iv - iw - 1) as usize]);
                    }
                    if iv + iw <= self.m_num_views {
                        wsum = wsum + self.m_tilt_inc_wgts[(iw - 1) as usize];
                        sum_interval = sum_interval
                            + self.m_tilt_inc_wgts[(iw - 1) as usize]
                                * (self.m_wgt_angles[(iv + iw - 1) as usize]
                                    - self.m_wgt_angles[(iv + iw - 2) as usize]);
                    }
                }
                atten = atten * (sum_interval / wsum) / avg_interval;
            }
            wgt_atten[(iv - 1) as usize] = atten;
        }
        //
        // Set up linear ramp
        if self.m_num_exact_cycles == 0 || self.m_flat_frac > 0. {
            for iv in 1..=self.m_num_views {
                //
                // Get weighting from nearest weighting angle
                atten = 1.;
                if self.m_num_tilt_inc_wgt > 0 && self.m_num_wgt_angles > 1 {
                    diff_min = 1.0e10;
                    for iw in 1..=self.m_num_wgt_angles {
                        diff = b3dabs!(
                            self.m_angles[(iv - 1) as usize] - self.m_wgt_angles[(iw - 1) as usize]
                        );
                        if diff < diff_min {
                            diff_min = diff;
                            atten = wgt_atten[(iw - 1) as usize];
                        }
                    }
                    if self.m_debug != 0 {
                        printf!(
                            "%4d %8.2f %10.5f\n",
                            ci(iv),
                            cf((self.m_angles[(iv - 1) as usize] / deg_to_rad) as f64),
                            cf(atten as f64)
                        );
                    }
                }
                //
                // Take negative if subtracting
                if self.m_num_view_subtract > 0 {
                    for i in 1..=self.m_num_view_subtract {
                        if self.m_iview_subtract[(i - 1) as usize] == 0
                            || self.m_map_used_view[(iv - 1) as usize]
                                == self.m_iview_subtract[(i - 1) as usize]
                        {
                            atten = -atten;
                        }
                    }
                }
                //
                atten_sum = atten_sum + atten;
                ind_base = (iv - 1 + (ifilter_set - 1) * self.m_num_views) * self.m_nx_filt_dim;

                // For filters from file, need to replicate values and scale
                if self.m_filter_file.is_some() {
                    let mut i = self.m_nx_pad_dim - 2;
                    while i >= 0 {
                        self.m_filter_array[(ind_base + i) as usize] = self.m_filter_array
                            [(ind_base + i / 2) as usize]
                            * atten
                            * read_in_scale;
                        self.m_filter_array[(ind_base + i + 1) as usize] =
                            self.m_filter_array[(ind_base + i) as usize];
                        i -= 2;
                    }
                } else {
                    for i in 1..=iramp_end {
                        // This was the basic filter
                        // ARRAY(ibase+2*I-1) =atten*(I-1)
                        // This is the mixture of the basic filter and a flat filter with
                        // a scaling that would give approximately the same output magnitude
                        freq = ((i as f64 - 1.) / self.m_nx_pad_dim as f64) as f32;
                        let ind = (ind_base + dup * i - dup) as usize;
                        if self.m_num_fake_sirt_iter <= 0 || i == 1 || freq < fake_alpha {
                            self.m_filter_array[ind] = (atten as f64
                                * ((1. - self.m_flat_frac as f64) * (i as f64 - 1.)
                                    + (self.m_flat_frac * self.m_filter_scale) as f64))
                                as f32;
                        } else {
                            self.m_filter_array[ind] = (atten as f64
                                * (i as f64 - 1.)
                                * (1.
                                    - (1.0f32 - fake_alpha / freq)
                                        .powf(fake_iter_use + fake_match_add)
                                        as f64))
                                as f32;
                        }
                        //
                        // This 0.2 is what Kak and Slaney's weighting function gives at 0
                        if i == 1 {
                            self.m_filter_array[ind] = (atten as f64 * 0.2) as f32;
                        }
                        //
                        // This is the SIRT filter, which divides the error equally among
                        // the pixels on a ray.
                        if self.m_flat_frac > 1. {
                            self.m_filter_array[ind] = sirt_frac * atten * self.m_filter_scale
                                / (self.m_ithick_bp as f32 / self.m_cos_beta[(iv - 1) as usize]);
                        }
                        //
                        // And just the value and compute the mean zero weighting
                        self.m_filter_array[(ind_base + dup * i - 1) as usize] =
                            self.m_filter_array[ind];
                    }
                }
                self.m_zero_weight = self.m_zero_weight
                    + self.m_filter_array[ind_base as usize] / self.m_num_views as f32;
            }
            if self.m_debug != 0 {
                printf!(
                    "Mean weighting factor %f\n",
                    cf((atten_sum / self.m_num_views as f32) as f64)
                );
            }
        } else {
            //
            // Exact filters
            // For each actual view, loop on weighting angles and compute influence
            // factors
            for i in 0..(self.m_num_views * self.m_nx_filt_dim) as usize {
                self.m_filter_array[i] = 0.;
            }
            for iv in 1..=self.m_num_views {
                ind_base = (iv - 1) * self.m_nx_filt_dim;
                for jv in 1..=self.m_num_wgt_angles {
                    sin_diff = b3dabs!(
                        self.m_wgt_angles[(jv - 1) as usize] - self.m_angles[(iv - 1) as usize]
                    )
                    .sin();
                    fact =
                        sin_diff * self.m_exact_obj_size / (self.m_nx_proj + self.m_num_pad) as f32;
                    irad_limit =
                        (self.m_num_exact_cycles as f64 / b3dmax!(1.0e-6, fact as f64)) as i32;
                    tab_fac = self.m_exact_samples * fact;

                    // mExactTable was indexed from 0 in Fortran so do not subtract 1
                    for i in 1..=b3dmin!(irad_limit, iramp_end) {
                        self.m_filter_array[(ind_base + dup * i - 1) as usize] +=
                            self.m_exact_table[b3dnint!((i - 1) as f32 * tab_fac) as usize];
                    }
                }
            }
            //
            // Invert the sum of influence factors
            for iv in 1..=self.m_num_views {
                ind_base = (iv - 1) * self.m_nx_filt_dim;
                zmax = 0.;
                for i in 1..=iramp_end {
                    let ia = (ind_base + dup * i - dup) as usize;
                    let ib = (ind_base + dup * i - 1) as usize;
                    if self.m_filter_array[ib] == 0. {
                        self.m_filter_array[ib] = 1.;
                    }
                    z = irad_end as f32 / self.m_filter_array[ib];
                    self.m_filter_array[ia] = z;
                    self.m_filter_array[ib] = z;
                    if z as f64 > zmax as f64 + 0.001 {
                        zmax = z;
                        imax = i;
                    }
                    //
                    // Limit the zero frequency weighting as above to avoid different
                    // mean and max
                    if i == 1 {
                        self.m_filter_array[ia] = b3dmin!(0.2, z as f64) as f32;
                        self.m_filter_array[ib] = b3dmin!(0.2, z as f64) as f32;
                        self.m_zero_weight += self.m_filter_array[ia] / self.m_num_views as f32;
                    }
                }
                exact_reach_top_mean = (exact_reach_top_mean as f64
                    + 0.5 * (imax as f64 - 1.) / (irad_end * self.m_num_views) as f64)
                    as f32;
            }
            printf!(
                "\nExact filters reach highest point at mean frequency of %6.3f cycles/pixel\n",
                cf(exact_reach_top_mean as f64)
            );
        }
        //
        // Set up Gaussian
        for i in irad_max + 1..=irad_end {
            atten = 0.;
            arg = ((i - irad_max) as f32 as f64 / b3dmax!(0.01, rad_fall as f64)) as f32;
            if arg < 8. {
                atten = ((-arg * arg) as f64 / 2.).exp() as f32;
            }
            //
            // Scale current value if multiplying, or last value if just falling off
            if if_mult_by_gaussian > 0 {
                imax = i;
            } else {
                imax = irad_max;
            }
            ind_base = (ifilter_set - 1) * self.m_num_views * self.m_nx_filt_dim;
            for _iv in 1..=self.m_num_views {
                z = atten * self.m_filter_array[(ind_base + dup * imax - dup) as usize];
                self.m_filter_array[(ind_base + dup * i - dup) as usize] = z;
                self.m_filter_array[(ind_base + dup * i - 1) as usize] = z;
                ind_base = ind_base + self.m_nx_filt_dim;
            }
        }
        if self.m_debug != 0 {
            //for (iv = 1; iv <= mNumViews; iv++) {
            let iv = self.m_num_views / 2;
            ind_base = (iv - 1 + (ifilter_set - 1) * self.m_num_views) * self.m_nx_filt_dim;
            for i in 1..=19 {
                printf!(
                    "RF: %4d %4d %11.5f\n",
                    ci(iv),
                    ci(i),
                    cf(self.m_filter_array[(ind_base + dup * i - dup) as usize] as f64)
                );
            }
            let mut i = 20;
            while i <= irad_end {
                printf!(
                    "RF: %4d %4d %11.5f\n",
                    ci(iv),
                    ci(i),
                    cf(self.m_filter_array[(ind_base + dup * i - dup) as usize] as f64)
                );
                i += 10;
            }
            //}
        }
    }

    /// `Tilt::maskPrep` (`tilt.cpp:1506`): prepare the limits of the slice width
    /// to be computed if masking is used.
    fn mask_prep(&mut self, lslice: i32) {
        //
        let radius_left: f32;
        let radius_right: f32;
        let mut y: f32;
        let mut yy: f32;
        let mut ycen_use: f32;
        let mut ix_left: i32;
        let mut ix_right: i32;
        //
        // Compute left and right edges of unmasked area
        if self.m_mask_edges != 0 {
            //
            // Adjust the Y center for alpha tilt (already adjusted for ifAlpha < 0)
            ycen_use = self.m_ycen_out;
            if self.m_if_alpha > 0 {
                ycen_use = (self.m_ycen_out as f64
                    + (1. / self.m_cos_alpha[0] as f64 - 1.) * self.m_y_offset as f64
                    - b3dnint!(
                        (lslice as f32 - self.m_center_slice) * self.m_sin_alpha[0]
                            / self.m_cos_alpha[0]
                    ) as f64) as f32;
            }
            //
            // Get square of radius of arcs of edge of input data from input center
            radius_left = (self.m_xcen_in + self.m_axis_xoffset - 1.0f32).powf(2.0f32);
            radius_right =
                (self.m_nx_proj as f32 - self.m_xcen_in - self.m_axis_xoffset).powf(2.0f32);
            for i in 1..=self.m_ithick_bp {
                y = i as f32 - ycen_use;
                yy = b3dmin!(y * y, radius_left);
                //
                // get distance of X coordinate from center, subtract from or add to
                // center and round up on left, down on right, plus added maskEdges pixels
                ix_left = (self.m_xcen_out as f64 + 1. - (radius_left - yy).sqrt() as f64) as i32;
                self.m_ix_unmasked_se[(2 * (i - 1)) as usize] =
                    b3dmax!(1, ix_left + self.m_num_extra_mask_pix);
                yy = b3dmin!(y * y, radius_right);
                ix_right = (self.m_xcen_out + (radius_right - yy).sqrt()) as i32;
                self.m_ix_unmasked_se[(1 + 2 * (i - 1)) as usize] =
                    b3dmin!(self.m_iwidth, ix_right - self.m_num_extra_mask_pix);
            }
            //-------------------------------------------------
            // If no maskEdges
        } else {
            for i in 1..=self.m_ithick_bp {
                self.m_ix_unmasked_se[(2 * (i - 1)) as usize] = 1;
                self.m_ix_unmasked_se[(1 + 2 * (i - 1)) as usize] = self.m_iwidth;
            }
        }
    }

    /// `Tilt::transform` (`tilt.cpp:1551`): applies a one-dimensional Fourier
    /// transform to all views corresponding to a given slice, applies the radial
    /// weighting function and then applies an inverse Fourier transform.
    fn transform(&self, array: &mut [f32], lslice: i32, ifilter_set: i32) {
        let ind_start: i32;
        let mut index: i32;
        let mut ind_rad: i32;
        let mut ib_from: i32;
        let mut ibase_to: i32;
        let mut ixp: i32;
        let mut ixp_p1: i32;
        let mut ixp_m1: i32;
        let mut ixp_p2: i32;
        let expand_pad: i32;
        let mut in_index: i32;
        let mut out_index: i32;
        let mut x: f32;
        let mut xp: f32;
        let mut dx: f32;
        let mut dx_m1: f32;
        let mut v4: f32;
        let mut v5: f32;
        let mut v6: f32;
        let mut a: f32;
        let mut c: f32;
        let mut den_new: f32;
        let mut dx_dx_m1: f32;
        let mut fx1: f32;
        let mut fx2: f32;
        let mut fx3: f32;
        let mut fx4: f32;
        let mut tmpre: f32;
        let tstart: f64;

        expand_pad = self.m_proj_super_fac * (self.m_nx_proj + self.m_num_pad) + 2;
        ind_start = self.m_ip_extra_size;
        tstart = wall_time();
        index = 1;
        if self.m_use_gpu {
            index = gpu_filter_lines(&mut array[ind_start as usize..], lslice, ifilter_set);
        }
        if index != 0 {
            //
            // Apply forward Fourier transform
            odfft_c(
                &mut array[ind_start as usize..],
                self.m_nx_proj + self.m_num_pad,
                self.m_num_views,
                0,
            );
            //
            // Apply Radial weighting
            // Do lines from the end backwards in case they are being spread apart for
            // supersampling
            let mut iv = self.m_num_views - 1;
            while iv >= 0 {
                in_index = ind_start + iv * self.m_nx_pad_dim;
                out_index = ind_start + iv * expand_pad;
                ind_rad = ((ifilter_set - 1) * self.m_num_views + iv) * self.m_nx_pad_dim;
                for _i in 0..self.m_nx_pad_dim {
                    array[out_index as usize] =
                        array[in_index as usize] * self.m_filter_array[ind_rad as usize];
                    out_index += 1;
                    in_index += 1;
                    ind_rad += 1;
                }

                // Phase shift and zero-pad if supersampling
                if self.m_proj_super_fac > 1 {
                    out_index = ind_start + iv * expand_pad;
                    for i in 0..(self.m_nx_pad_dim / 2) as usize {
                        let o = out_index as usize;
                        tmpre = array[o];
                        array[o] =
                            self.m_phase_real[i] * tmpre - self.m_phase_imag[i] * array[o + 1];
                        array[o + 1] =
                            self.m_phase_imag[i] * tmpre + self.m_phase_real[i] * array[o + 1];
                        out_index += 2;
                    }
                    for _i in self.m_nx_pad_dim..expand_pad {
                        array[out_index as usize] = 0.;
                        out_index += 1;
                    }
                }
                iv -= 1;
            }
            //
            // Apply inverse transform
            odfft_c(
                &mut array[ind_start as usize..],
                self.m_proj_super_fac * (self.m_nx_proj + self.m_num_pad),
                self.m_num_views,
                1,
            );
        }
        if self.m_debug != 0 {
            printf!("Filter time %8.4f\n", cf(wall_time() - tstart));
        }
        if self.m_ip_extra_size == 0 {
            return;
        }
        //
        // do cosine stretch and move down one plane
        // Use cubic interpolation a la cubinterp
        //
        // print *,'istart, ibase', istart, ibase
        let nx_proj_f = self.m_nx_proj as f32;
        for iv in 0..self.m_num_views {
            let ivu = iv as usize;
            ib_from = ind_start + iv * self.m_nx_pad_dim; // 0 based index
            ibase_to = self.m_ind_stretch_line[ivu]; // Should be same as it was
            // print *,nv, ibfrom, ibto, nxStretched(nv)
            if self.m_interp_ord_stretch == 1 {
                //
                // linear interpolation
                //
                for i in 1..=self.m_nx_stretched[ivu] {
                    x = i as f32 / self.m_interp_fac_stretch as f32 + self.m_stretch_offset[ivu];
                    xp = b3dmin!(
                        b3dmax!(1., (x * self.m_cos_beta[ivu]) as f64),
                        nx_proj_f as f64
                    ) as f32;
                    ixp = fortran_int!(f32: xp);
                    dx = xp - ixp as f32;
                    ixp = ixp + ib_from;
                    ixp_p1 = b3dmin!(ixp + 1, self.m_nx_proj + ib_from);
                    dx_m1 = (dx as f64 - 1.) as f32;
                    array[(ibase_to + i - 1) as usize] =
                        -dx_m1 * array[(ixp - 1) as usize] + dx * array[(ixp_p1 - 1) as usize];
                }
            } else if self.m_interp_ord_stretch == 2 {
                //
                // quadratic
                //
                for i in 1..=self.m_nx_stretched[ivu] {
                    x = i as f32 / self.m_interp_fac_stretch as f32 + self.m_stretch_offset[ivu];
                    xp = b3dmin!(
                        b3dmax!(1., (x * self.m_cos_beta[ivu]) as f64),
                        nx_proj_f as f64
                    ) as f32;
                    ixp = b3dnint!(xp);
                    dx = xp - ixp as f32;
                    ixp = ixp + ib_from;
                    ixp_p1 = b3dmin!(ixp + 1, self.m_nx_proj + ib_from);
                    ixp_m1 = b3dmax!(ixp - 1, 1 + ib_from);
                    v4 = array[(ixp_m1 - 1) as usize];
                    v5 = array[(ixp - 1) as usize];
                    v6 = array[(ixp_p1 - 1) as usize];
                    //
                    a = ((v6 + v4) as f64 * 0.5 - v5 as f64) as f32;
                    c = ((v6 - v4) as f64 * 0.5) as f32;
                    //
                    den_new = a * dx * dx + c * dx + v5;
                    // dennew=min(dennew, max(v4, v5, v6))
                    // dennew=max(dennew, min(v4, v5, v6))
                    array[(ibase_to + i - 1) as usize] = den_new;
                }
            } else {
                //
                // cubic
                //
                for i in 1..=self.m_nx_stretched[ivu] {
                    x = i as f32 / self.m_interp_fac_stretch as f32 + self.m_stretch_offset[ivu];
                    xp = b3dmin!(
                        b3dmax!(1., (x * self.m_cos_beta[ivu]) as f64),
                        nx_proj_f as f64
                    ) as f32;
                    ixp = fortran_int!(f32: xp);
                    dx = xp - ixp as f32;
                    ixp = ixp + ib_from;
                    ixp_p1 = b3dmin!(ixp + 1, self.m_nx_proj + ib_from);
                    ixp_m1 = b3dmax!(ixp - 1, 1 + ib_from);
                    ixp_p2 = b3dmin!(ixp + 2, self.m_nx_proj + ib_from);

                    dx_m1 = (dx as f64 - 1.) as f32;
                    dx_dx_m1 = dx * dx_m1;
                    fx1 = -dx_m1 * dx_dx_m1;
                    fx4 = dx * dx_dx_m1;
                    fx2 = (1. + (dx * dx) as f64 * (dx as f64 - 2.)) as f32;
                    fx3 = (dx as f64 * (1. - dx_dx_m1 as f64)) as f32;
                    den_new = fx1 * array[(ixp_m1 - 1) as usize]
                        + fx2 * array[(ixp - 1) as usize]
                        + fx3 * array[(ixp_p1 - 1) as usize]
                        + fx4 * array[(ixp_p2 - 1) as usize];
                    // dennew=min(dennew, max(array(ixpm1), array(ixp), array(ixpp1), &
                    // array(ixpp2)))
                    // dennew=max(dennew, min(array(ixpm1), array(ixp), array(ixpp1), &
                    // array(ixpp2)))
                    array[(ibase_to + i - 1) as usize] = den_new;
                }
            }
        }
    }
}

impl Tilt {
    /// `Tilt::project` (`tilt.cpp:1688`): assembles one reconstructed slice
    /// perpendicular to the tilt axis, using a back projection method.
    /// `input_arr` is the full array of lines being used, `ind_start` is a
    /// 1-based index into it.
    fn project(&mut self, input_arr: &[f32], ind_start: i32, lslice: i32) {
        let mut j_start: [i32; 3] = [0; 3];
        let mut j_end: [i32; 3] = [0; 3];
        let mut xproj8: f64;
        let tstart: f64;
        let nx_proj_pad: i32;
        let iproj_delta: i32;
        let mut ipoint: i32;
        let mut index: i32;
        let mut j: i32;
        let mut cbeta: f32;
        let mut sbeta: f32;
        let mut zz: f32;
        let mut z_part: f32 = 0.;
        let mut yy: f32;
        let mut yproj: f32 = 0.;
        let mut yfrac: f32 = 0.;
        let mut one_my_frac: f32 = 0.;
        let mut j_proj: i32 = 0;
        let mut j_left: i32;
        let mut j_right: i32;
        let mut iproj: i32;
        let mut ip1: i32;
        let mut ip2: i32;
        let mut ind: i32;
        let mut iproj_base: i32;
        let mut if_ytest: i32;
        let nx_pad: i32;
        let ny_pad: i32;
        let mut j_test_left: i32;
        let mut j_test_right: i32;
        let mut mask_base: i32;
        let mut unmask_start: i32;
        let mut unmask_end: i32;
        let mut x_left: f32;
        let mut x_right: f32;
        let mut x: f32;
        let mut xfrac: f32;
        let mut one_mx_frac: f32;
        let mut z_bottom: f32;
        let mut z_top: f32;
        let mut xproj: f32;
        let y_end_tol: f32;
        const MAX_DIST: i32 = 1000;
        let mut ind_del_ray = [0i32; (MAX_DIST + 1) as usize];
        let mut num_rays_hit = [0i32; (MAX_DIST + 1) as usize];
        let mut ray_areas = [0f32; (3 * (MAX_DIST + 1)) as usize];
        let super_samp: i32 = self.m_super_sample_fac;
        let mut slice_size: i64 = self.m_islice_size_bp;
        let mut ss_width: i32 = self.m_iwidth;
        let mut ss_thick: i32 = self.m_ithick_bp;
        let mut ss_border: i32 = 0;
        //
        // A note on the ubiquitous ytol: It is needed to keep artifacts from
        // building up at ends of data set through SIRT, from reprojection of the
        // line between real data and fill, or backprojection of an edge in the
        // projection difference.  2.05 was sufficient for X-axis tilt cases but
        // 3.05 was needed for local alignments, thus it is set to 3.05 everywhere
        // (Here, reprojection routines, and in GPU routines)
        y_end_tol = 3.05;
        //
        // Determine maskEdges extent if it is variable
        if self.m_if_alpha != 0 && self.m_mask_edges != 0 {
            self.mask_prep(lslice);
        }
        nx_proj_pad = self.m_proj_super_fac * (self.m_nx_proj + self.m_num_pad) + 2;
        tstart = wall_time();
        //
        // GPU backprojection
        if self.m_use_gpu {
            ind = 1;
            x = self.m_xcen_in + self.m_axis_xoffset;
            if self.m_if_alpha <= 0 && self.m_nx_warp == 0 {
                ind = gpu_bp_no_x(
                    &mut self.m_out_slice_arr,
                    &input_arr[(ind_start - 1) as usize..],
                    &self.m_sin_beta,
                    &self.m_cos_beta,
                    self.m_nx_proj,
                    x,
                    self.m_xcen_out,
                    self.m_ycen_out,
                    self.m_edge_fill,
                );
            } else if self.m_nx_warp == 0 && self.m_load_gpu_start > 0 {
                ind = gpu_bp_xtilt(
                    &mut self.m_out_slice_arr,
                    &self.m_sin_beta,
                    &self.m_cos_beta,
                    &self.m_sin_alpha,
                    &self.m_cos_alpha,
                    &self.m_xzfac,
                    &self.m_yzfac,
                    self.m_nx_proj,
                    self.m_ny_proj,
                    x,
                    self.m_xcen_out,
                    self.m_ycen_out,
                    lslice,
                    self.m_center_slice,
                    self.m_edge_fill,
                );
            } else if self.m_load_gpu_start > 0 {
                ind = gpu_bp_local(
                    &mut self.m_out_slice_arr,
                    lslice,
                    self.m_nx_warp,
                    self.m_ny_warp,
                    self.m_ix_start_warp,
                    self.m_iy_start_warp,
                    self.m_idel_xwarp,
                    self.m_idel_ywarp,
                    self.m_nx_proj,
                    self.m_xcen_out,
                    self.m_xcen_in,
                    self.m_axis_xoffset,
                    self.m_ycen_out,
                    self.m_center_slice,
                    self.m_edge_fill,
                );
            }
            if ind == 0 {
                if self.m_debug != 0 {
                    printf!("GPU backprojection time %9.5f\n", cf(wall_time() - tstart));
                }
                return;
            }
        }

        // Changes in sizes for super-sampling
        let use_super = super_samp > 1;
        if use_super {
            ss_width = self.m_nx_super_samp * super_samp;
            ss_thick = self.m_ny_super_samp * super_samp;
            slice_size = ss_width as i64 * ss_thick as i64;
            ss_border = self.m_super_samp_border * super_samp;
        }
        // `outArr` is `mSuperOutArr` when super-sampling, otherwise `mOutSliceArr`
        let mut out_arr = if use_super {
            std::mem::take(&mut self.m_super_out_arr)
        } else {
            std::mem::take(&mut self.m_out_slice_arr)
        };
        //
        // CPU backprojection: clear out the slice
        for iz in 0..slice_size as usize {
            out_arr[iz] = 0.;
        }
        iproj_delta = self.m_in_plane_size;
        let edge_fill = self.m_edge_fill;
        //
        if self.m_nx_warp == 0 {
            //
            // Loop over all views
            ipoint = ind_start - 1;
            for iv in 1..=self.m_num_views {
                let ivu = (iv - 1) as usize;
                //
                // Set view angle
                cbeta = self.m_cos_beta[ivu];
                sbeta = self.m_sin_beta[ivu];
                if self.m_use_intersections != 0 {
                    let angle = (sbeta.atan2(cbeta) as f64 / RADIANS_PER_DEGREE) as f32;
                    make_ray_area_lookup_table(
                        angle,
                        1,
                        MAX_DIST,
                        0.01,
                        &mut ind_del_ray,
                        &mut num_rays_hit,
                        &mut ray_areas,
                    );
                }
                //
                // Loop over all points in output slice line by line
                index = 1;

                // Loop in Z: use indexes relative to final output slice, so start negative
                // for the super-sampling border
                for iz in 1 - ss_border..=ss_thick - ss_border {
                    zz = (((iz as f64 - 0.5) / super_samp as f64 + 0.5 - self.m_ycen_out as f64)
                        * self.m_compress[ivu] as f64) as f32;
                    if self.m_if_alpha <= 0 {
                        z_part = zz * sbeta + self.m_xcen_in + self.m_axis_xoffset;
                    } else {
                        //
                        // If x-axis tilting, find interpolation factor between the
                        // slices
                        //
                        yy = lslice as f32 - self.m_center_slice;
                        z_part = yy * self.m_sin_alpha[ivu] * sbeta
                            + zz * (self.m_cos_alpha[ivu] * sbeta + self.m_xzfac[ivu])
                            + self.m_xcen_in
                            + self.m_axis_xoffset;
                        yproj = yy * self.m_cos_alpha[ivu]
                            - zz * (self.m_sin_alpha[ivu] - self.m_yzfac[ivu])
                            + self.m_center_slice;
                        //
                        // if inside the tolerance, clamp it to the endpoints
                        if yproj as f64 >= 1. - y_end_tol as f64
                            && yproj <= self.m_ny_proj as f32 + y_end_tol
                        {
                            yproj =
                                b3dmax!(1., b3dmin!(self.m_ny_proj as f32, yproj) as f64) as f32;
                        }
                        j_proj = fortran_int!(f32: yproj);
                        j_proj = b3dmin!(self.m_ny_proj - 1, j_proj);
                        yfrac = yproj - j_proj as f32;
                        one_my_frac = (1. - yfrac as f64) as f32;
                    }
                    //
                    // compute left and right limits that come from legal data
                    //
                    x = cbeta;
                    if (b3dabs!(cbeta) as f64) < 0.001 {
                        x = b3dsign!(0.001f64, cbeta) as f32;
                    }
                    x_left = ((1. - z_part as f64) / x as f64 + self.m_xcen_out as f64) as f32;
                    x_right = (self.m_nx_proj as f32 - z_part) / x + self.m_xcen_out;
                    if x_right < x_left {
                        x = x_left;
                        x_left = x_right;
                        x_right = x;
                    }

                    // Masking is off when super-sampling and arrays have values at the
                    // extremes
                    mask_base = 2 * (1 + (iz - 1) / super_samp);
                    mask_base = b3dmax!(2, b3dmin!(2 * self.m_ithick_bp, mask_base));
                    j_left = (super_samp as f32 * x_left) as i32;
                    if (j_left as f32) < super_samp as f32 * x_left {
                        j_left += 1;
                    }

                    // Allow jLeft to go below 0 and jRight to go past final slice end when
                    // super-sampling, so it can be used to compute projection position
                    j_left = b3dmax!(
                        j_left,
                        super_samp * (self.m_ix_unmasked_se[(mask_base - 2) as usize] - 1) + 1
                            - ss_border
                    );
                    j_right = (super_samp as f32 * x_right) as i32;
                    if j_right as f32 == super_samp as f32 * x_right {
                        j_right -= 1;
                    }
                    j_right = b3dmin!(
                        j_right,
                        super_samp * self.m_ix_unmasked_se[(mask_base - 1) as usize] + ss_border
                    );
                    //
                    // If the limits are now crossed, just skip to full fill at end
                    if j_left <= j_right {
                        //
                        // set up starting projection position and index
                        x = ((j_left as f64 - 0.5) / super_samp as f64 + 0.5
                            - self.m_xcen_out as f64) as f32;

                        // Do fill; here add ssBorder back to jLeft to get correct # of
                        // pixels
                        ind = index
                            + super_samp * (self.m_ix_unmasked_se[(mask_base - 2) as usize] - 1);
                        while ind <= index + j_left + ss_border - 2 {
                            out_arr[(ind - 1) as usize] += edge_fill;
                            ind += 1;
                        }
                        index = index + (j_left + ss_border - 1);
                        if self.m_interp_fac_stretch != 0 {
                            //
                            // Computation with prestretched data.  Super-sampling turns
                            // stretching off, so `outArr` is `mOutSliceArr` here, the
                            // array the source names explicitly.
                            //
                            xproj8 = (self.m_interp_fac_stretch as f32
                                * (z_part / cbeta + x - self.m_stretch_offset[ivu]))
                                as f64;
                            xproj8 = b3dmax!(
                                1.,
                                b3dmin!(self.m_nx_stretched[ivu] as f64 - 0.001, xproj8)
                            );
                            iproj = fortran_int!(f64: xproj8);
                            xfrac = (xproj8 - iproj as f64) as f32;
                            iproj = iproj + ipoint + self.m_ind_stretch_line[ivu];
                            one_mx_frac = (1. - xfrac as f64) as f32;
                            if self.m_if_alpha <= 0 {
                                //
                                // interpolation in simple case of no x-axis tilt
                                //
                                let count = j_right - j_left + 1;
                                if count > 0 && self.m_interp_fac_stretch > 0 {
                                    // The source's loop over `ind`, with the output run
                                    // and the strided input run each bounds-checked once
                                    // by slicing, and the same per-element expression.
                                    let step = self.m_interp_fac_stretch as usize;
                                    let n = count as usize;
                                    let ob = (index - 1) as usize;
                                    let ib = (iproj - 1) as usize;
                                    let out = &mut out_arr[ob..ob + n];
                                    let inp = &input_arr[ib..ib + (n - 1) * step + 2];
                                    // Output k reads input k * step and k * step + 1;
                                    // written as a zip over the strided runs so the
                                    // compiler can use packed arithmetic, as the
                                    // reference's compiler does.  Each element is the
                                    // source's own expression.
                                    if step == 1 {
                                        for (o, w) in out.iter_mut().zip(inp.windows(2)) {
                                            *o += one_mx_frac * w[0] + xfrac * w[1];
                                        }
                                    } else if step == 2 {
                                        // The default sampling factor, with the stride a
                                        // constant the compiler can vectorize over.
                                        for (o, c) in out.iter_mut().zip(inp.chunks_exact(2)) {
                                            *o += one_mx_frac * c[0] + xfrac * c[1];
                                        }
                                    } else {
                                        let (head, last) = out.split_at_mut(n - 1);
                                        for (o, c) in head.iter_mut().zip(inp.chunks_exact(step)) {
                                            *o += one_mx_frac * c[0] + xfrac * c[1];
                                        }
                                        let p = (n - 1) * step;
                                        last[0] += one_mx_frac * inp[p] + xfrac * inp[p + 1];
                                    }
                                } else {
                                    ind = index;
                                    while ind <= index + j_right - j_left {
                                        out_arr[(ind - 1) as usize] += one_mx_frac
                                            * input_arr[(iproj - 1) as usize]
                                            + xfrac * input_arr[iproj as usize];
                                        iproj = iproj + self.m_interp_fac_stretch;
                                        ind += 1;
                                    }
                                }
                                index = index + j_right + 1 - j_left;
                            } else {
                                //
                                // If x-axis tilting, interpolate from two lines
                                //
                                ip1 = iproj + (j_proj - lslice) * iproj_delta;
                                ip2 = ip1 + iproj_delta;
                                if yproj as f64 >= 1.
                                    && yproj <= self.m_ny_proj as f32
                                    && ip1 >= 1
                                    && ip2 >= 1
                                    && ip1 < self.m_ind_load_end
                                    && ip2 < self.m_ind_load_end
                                {
                                    let count = j_right - j_left + 1;
                                    if count > 0 && self.m_interp_fac_stretch > 0 {
                                        // As above: the source's loop, the two strided
                                        // input runs sliced up front.
                                        let step = self.m_interp_fac_stretch as usize;
                                        let n = count as usize;
                                        let ob = (index - 1) as usize;
                                        let len = (n - 1) * step + 2;
                                        let b1 = (ip1 - 1) as usize;
                                        let b2 = (ip2 - 1) as usize;
                                        let out = &mut out_arr[ob..ob + n];
                                        let in1 = &input_arr[b1..b1 + len];
                                        let in2 = &input_arr[b2..b2 + len];
                                        if step == 2 {
                                            // The default sampling factor, with the
                                            // stride a constant the compiler can
                                            // vectorize over.
                                            for ((o, c1), c2) in out
                                                .iter_mut()
                                                .zip(in1.chunks_exact(2))
                                                .zip(in2.chunks_exact(2))
                                            {
                                                *o += one_mx_frac
                                                    * (one_my_frac * c1[0] + yfrac * c2[0])
                                                    + xfrac * (one_my_frac * c1[1] + yfrac * c2[1]);
                                            }
                                        } else {
                                            for (k, o) in out.iter_mut().enumerate() {
                                                let p = k * step;
                                                *o += one_mx_frac
                                                    * (one_my_frac * in1[p] + yfrac * in2[p])
                                                    + xfrac
                                                        * (one_my_frac * in1[p + 1]
                                                            + yfrac * in2[p + 1]);
                                            }
                                        }
                                    } else {
                                        ind = index;
                                        while ind <= index + j_right - j_left {
                                            out_arr[(ind - 1) as usize] += one_mx_frac
                                                * (one_my_frac * input_arr[(ip1 - 1) as usize]
                                                    + yfrac * input_arr[(ip2 - 1) as usize])
                                                + xfrac
                                                    * (one_my_frac * input_arr[ip1 as usize]
                                                        + yfrac * input_arr[ip2 as usize]);
                                            ip1 = ip1 + self.m_interp_fac_stretch;
                                            ip2 = ip2 + self.m_interp_fac_stretch;
                                            ind += 1;
                                        }
                                    }
                                } else {
                                    ind = index;
                                    while ind <= index + j_right + 1 - j_left - 1 {
                                        out_arr[(ind - 1) as usize] += edge_fill;
                                        ind += 1;
                                    }
                                }
                                index = index + j_right + 1 - j_left;
                            }
                        } else {
                            //
                            // Computation direct from projection data
                            //
                            xproj8 = self.m_proj_super_fac as f64
                                * ((z_part + x * cbeta) as f64 - 0.5)
                                + 0.5;
                            xproj8 = b3dmax!(1., b3dmin!(self.m_nx_proj as f64 - 0.001, xproj8));

                            if self.m_use_intersections != 0 {
                                self.bp_sum_area_no_x(
                                    &mut out_arr,
                                    index,
                                    input_arr,
                                    ipoint,
                                    j_right + 1 - j_left,
                                    xproj8,
                                    self.m_proj_super_fac as f32 * cbeta / super_samp as f32,
                                    self.m_nx_proj,
                                    &ind_del_ray,
                                    &num_rays_hit,
                                    &ray_areas,
                                    MAX_DIST,
                                );
                                index += j_right + 1 - j_left;
                            } else if self.m_if_alpha <= 0 {
                                //
                                // interpolation in simple case of no x-axis tilt
                                //
                                self.bp_sum_no_x(
                                    &mut out_arr,
                                    &mut index,
                                    input_arr,
                                    ipoint,
                                    j_right + 1 - j_left,
                                    xproj8,
                                    self.m_proj_super_fac as f32 * cbeta / super_samp as f32,
                                );
                            } else {
                                //
                                // If x-axis tilting
                                //
                                iproj = fortran_int!(f64: xproj8);
                                iproj_base = ipoint + (j_proj - lslice) * iproj_delta;
                                ip1 = iproj_base + iproj;
                                ip2 = ip1 + iproj_delta;
                                if yproj as f64 >= 1.
                                    && yproj <= self.m_ny_proj as f32
                                    && ip1 >= 1
                                    && ip2 >= 1
                                    && ip1 < self.m_ind_load_end
                                    && ip2 < self.m_ind_load_end
                                {
                                    self.bp_sum_xtilt(
                                        &mut out_arr,
                                        &mut index,
                                        input_arr,
                                        iproj_base,
                                        iproj_delta,
                                        j_right + 1 - j_left,
                                        xproj8,
                                        self.m_proj_super_fac as f32 * cbeta / super_samp as f32,
                                        yfrac,
                                        one_my_frac,
                                    );
                                } else {
                                    ind = index;
                                    while ind <= index + j_right + 1 - j_left - 1 {
                                        out_arr[(ind - 1) as usize] += edge_fill;
                                        ind += 1;
                                    }
                                    index = index + j_right + 1 - j_left;
                                }
                            }
                        }
                    } else {
                        j_right = -ss_border;
                    }

                    // Final fill from jRight to end, which is ssWidth - ssBorder relative
                    // to jRight
                    ind = index;
                    while ind <= index + ss_width - j_right - ss_border - 1 {
                        out_arr[(ind - 1) as usize] += edge_fill;
                        ind += 1;
                    }
                    index = index + ss_width - j_right - ss_border;
                }
                //
                //-------------------------------------------
                //
                // End of projection loop
                if self.m_interp_fac_stretch == 0 {
                    ipoint = ipoint + nx_proj_pad;
                }
            }
        } else {
            //
            // LOCAL ALIGNMENTS
            //
            // Loop over all views
            ipoint = ind_start - 1;
            for iv in 1..=self.m_num_views {
                let ivu = (iv - 1) as usize;
                //
                // precompute the factors for getting xproj and yproj all the
                // way across the slice
                // Again, go from negative Z index for super-sampling
                if_ytest = 0;
                z_bottom = (1 - ss_border / super_samp) as f32 - self.m_ycen_out;
                z_bottom = z_bottom * self.m_compress[ivu];
                z_top = (self.m_ithick_bp + ss_border / super_samp) as f32 - self.m_ycen_out;
                z_top = z_top * self.m_compress[ivu];
                j = 1;
                while j <= ss_width {
                    let ju = (j - 1) as usize;
                    //
                    // get the fixed and z-dependent component of the
                    // projection coordinates (for every index in super-sampled slice)
                    // subtract ssBorder from j where it is used as a coordinate
                    let (mut xf, mut xz, mut yf, mut yz) = (0f32, 0f32, 0f32, 0f32);
                    self.local_proj_factors(
                        (((j - ss_border) as f64 - 0.5) / super_samp as f64 + 0.5) as f32,
                        lslice,
                        iv,
                        &mut xf,
                        &mut xz,
                        &mut yf,
                        &mut yz,
                    );
                    self.m_xproj_fs[ju] = xf;
                    self.m_xproj_zs[ju] = xz;
                    self.m_yproj_fs[ju] = yf;
                    self.m_yproj_zs[ju] = yz;
                    //
                    // see if any y testing is needed in the inner loop by checking
                    // yproj at top and bottom in Z
                    //
                    yproj = self.m_yproj_fs[ju] + self.m_yproj_zs[ju] * z_bottom;
                    j_proj = fortran_int!(f32: yproj);
                    ip1 = ipoint + (j_proj - lslice) * iproj_delta + 1;
                    ip2 = ip1 + iproj_delta;
                    if ip1 <= 1
                        || ip2 <= 1
                        || ip1 >= self.m_ind_load_end
                        || ip2 >= self.m_ind_load_end
                        || j_proj < 1
                        || j_proj >= self.m_ny_proj
                    {
                        if_ytest = 1;
                    }
                    yproj = self.m_yproj_fs[ju] + self.m_yproj_zs[ju] * z_top;
                    j_proj = fortran_int!(f32: yproj);
                    ip1 = ipoint + (j_proj - lslice) * iproj_delta + 1;
                    ip2 = ip1 + iproj_delta;
                    if ip1 <= 1
                        || ip2 <= 1
                        || ip1 >= self.m_ind_load_end
                        || ip2 >= self.m_ind_load_end
                        || j_proj < 1
                        || j_proj >= self.m_ny_proj
                    {
                        if_ytest = 1;
                    }
                    j += 1;
                }
                //
                // walk in from each end until xproj is safely within bounds
                // to define region where no x checking is needed
                //
                j_test_left = 0;
                j = 1;
                while j_test_left == 0 && j < ss_width {
                    let ju = (j - 1) as usize;
                    if b3dmin!(
                        self.m_xproj_fs[ju] + z_bottom * self.m_xproj_zs[ju],
                        self.m_xproj_fs[ju] + z_top * self.m_xproj_zs[ju]
                    ) >= 1.
                    {
                        j_test_left = j;
                    }
                    j = j + 1;
                }
                if j_test_left == 0 {
                    j_test_left = ss_width;
                }
                //
                j_test_right = 0;
                j = ss_width;
                while j_test_right == 0 && j > 1 {
                    let ju = (j - 1) as usize;
                    if b3dmax!(
                        self.m_xproj_fs[ju] + z_bottom * self.m_xproj_zs[ju],
                        self.m_xproj_fs[ju] + z_top * self.m_xproj_zs[ju]
                    ) < self.m_nx_proj as f32
                    {
                        j_test_right = j;
                    }
                    j = j - 1;
                }
                if j_test_right == 0 {
                    j_test_right = 1;
                }
                if j_test_right < j_test_left {
                    j_test_right = ss_width / 2;
                    j_test_left = j_test_right + 1;
                }
                //
                index = 1;
                //
                // loop over the slice, outer loop on z levels, start negative for
                // super-sampling
                //
                for iz in 1 - ss_border..=ss_thick - ss_border {
                    zz = (((iz as f64 - 0.5) / super_samp as f64 + 0.5 - self.m_ycen_out as f64)
                        * self.m_compress[ivu] as f64) as f32;

                    // Here just forget the mask values and set start/end directly for
                    // super-sample
                    if ss_border != 0 {
                        unmask_start = 1;
                        unmask_end = ss_width;
                    } else {
                        mask_base = 2 * (1 + (iz - 1) / super_samp);
                        unmask_start =
                            super_samp * (self.m_ix_unmasked_se[(mask_base - 2) as usize] - 1) + 1;
                        unmask_end = super_samp * self.m_ix_unmasked_se[(mask_base - 1) as usize];
                    }
                    j_left = b3dmax!(j_test_left, unmask_start);
                    j_right = b3dmin!(j_test_right, unmask_end);
                    index = index + unmask_start - 1;
                    //
                    // set up to do inner loop in three regions of X
                    //
                    j_start[0] = unmask_start;
                    j_end[0] = j_left - 1;
                    j_start[1] = j_left;
                    j_end[1] = j_right;
                    j_start[2] = j_right + 1;
                    j_end[2] = unmask_end;
                    for jregion in 1..=3usize {
                        if jregion != 2 || if_ytest == 1 {
                            //
                            // loop involving full testing - either left or right
                            // sides needing x testing, or anywhere if y testing
                            // needed
                            //
                            for j in j_start[jregion - 1]..=j_end[jregion - 1] {
                                let ju = (j - 1) as usize;
                                xproj = (self.m_proj_super_fac as f64
                                    * ((self.m_xproj_fs[ju] + zz * self.m_xproj_zs[ju]) as f64
                                        - 0.5)
                                    + 0.5) as f32;
                                yproj = self.m_yproj_fs[ju] + zz * self.m_yproj_zs[ju];
                                if yproj as f64 >= 1. - y_end_tol as f64
                                    && yproj <= self.m_ny_proj as f32 + y_end_tol
                                {
                                    yproj =
                                        b3dmax!(1., b3dmin!(self.m_ny_proj as f32, yproj) as f64)
                                            as f32;
                                }
                                if xproj >= 1.
                                    && xproj <= self.m_nx_proj as f32
                                    && yproj as f64 >= 1.
                                    && yproj <= self.m_ny_proj as f32
                                {
                                    //
                                    iproj = fortran_int!(f32: xproj);
                                    iproj = b3dmin!(self.m_nx_proj - 1, iproj);
                                    xfrac = xproj - iproj as f32;
                                    j_proj = fortran_int!(f32: yproj);
                                    j_proj = b3dmin!(self.m_ny_proj - 1, j_proj);
                                    yfrac = yproj - j_proj as f32;
                                    //
                                    ip1 = ipoint + (j_proj - lslice) * iproj_delta + iproj;
                                    ip2 = ip1 + iproj_delta;
                                    if ip1 >= 1
                                        && ip2 >= 1
                                        && ip1 < self.m_ind_load_end
                                        && ip2 < self.m_ind_load_end
                                    {
                                        let o = &mut out_arr[(index - 1) as usize];
                                        *o = (*o as f64
                                            + (1. - yfrac as f64)
                                                * ((1. - xfrac as f64)
                                                    * input_arr[(ip1 - 1) as usize] as f64
                                                    + (xfrac * input_arr[ip1 as usize]) as f64)
                                            + yfrac as f64
                                                * ((1. - xfrac as f64)
                                                    * input_arr[(ip2 - 1) as usize] as f64
                                                    + (xfrac * input_arr[ip2 as usize]) as f64))
                                            as f32;
                                    } else {
                                        out_arr[(index - 1) as usize] += edge_fill;
                                    }
                                } else {
                                    out_arr[(index - 1) as usize] += edge_fill;
                                }
                                index = index + 1;
                            }
                            //
                            // loop for no x-testing and no y testing
                            //
                        } else {
                            let (fs, zs, yfs, yzs) = (
                                std::mem::take(&mut self.m_xproj_fs),
                                std::mem::take(&mut self.m_xproj_zs),
                                std::mem::take(&mut self.m_yproj_fs),
                                std::mem::take(&mut self.m_yproj_zs),
                            );
                            self.bp_sum_local(
                                &mut out_arr,
                                &mut index,
                                input_arr,
                                zz,
                                &fs,
                                &zs,
                                &yfs,
                                &yzs,
                                ipoint,
                                iproj_delta,
                                lslice,
                                j_start[jregion - 1],
                                j_end[jregion - 1],
                            );
                            self.m_xproj_fs = fs;
                            self.m_xproj_zs = zs;
                            self.m_yproj_fs = yfs;
                            self.m_yproj_zs = yzs;
                        }
                    }
                    index = index + ss_width - unmask_end;
                }
                //-------------------------------------------
                //
                // End of projection loop
                ipoint = ipoint + nx_proj_pad;
            }
        }
        if use_super {
            nx_pad = super_samp * self.m_nx_out_pad;
            ny_pad = super_samp * self.m_ny_out_pad;
            slice_taper_out_pad(
                PadIn::InPlace,
                SLICE_MODE_FLOAT,
                ss_width,
                ss_thick,
                &mut out_arr,
                nx_pad + 2,
                nx_pad,
                ny_pad,
                0,
                0.,
            );
            todfft_c(&mut out_arr, nx_pad, ny_pad, 0);
            fourier_reduce_image(
                &out_arr,
                nx_pad,
                ny_pad,
                &mut self.m_out_slice_arr,
                self.m_nx_out_pad,
                self.m_ny_out_pad,
                0.,
                0.,
                Some(&mut self.m_super_temp_arr),
            );
            self.m_super_out_arr = out_arr;
            self.clean_up_reduced_fft(self.m_beta_min, self.m_beta_max);
            todfft_c(
                &mut self.m_out_slice_arr,
                self.m_nx_out_pad,
                self.m_ny_out_pad,
                1,
            );
            j_left = (self.m_nx_out_pad - self.m_iwidth) / 2;
            j_right = (self.m_ny_out_pad - self.m_ithick_bp) / 2;
            // The source extracts in place, the output rows moving towards the start of
            // the same array; reading from a copy gives the same result.
            let src: Vec<u8> = self
                .m_out_slice_arr
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            let mut dst: Vec<u8> = src.clone();
            let mut nxr = 0;
            let mut nyr = 0;
            let err = extract_with_binning(
                &src,
                SLICE_MODE_FLOAT,
                self.m_nx_out_pad + 2,
                j_left,
                j_left + self.m_iwidth - 1,
                j_right,
                j_right + self.m_ithick_bp - 1,
                1,
                &mut dst,
                0,
                &mut nxr,
                &mut nyr,
            );
            for (v, b) in self.m_out_slice_arr.iter_mut().zip(dst.chunks_exact(4)) {
                *v = f32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
            }
            if err != 0 {
                exit_error(b"Reducing super-sampled slice");
            }
        } else {
            self.m_out_slice_arr = out_arr;
        }
        if self.m_debug != 0 {
            printf!("CPU backprojection time %9.5f\n", cf(wall_time() - tstart));
        }
    }

    /// `Tilt::cleanUpReducedFFT` (`tilt.cpp:2124`): remove corner pixels or
    /// missing wedge pixels from the FFT.
    fn clean_up_reduced_fft(&mut self, beta_min: f32, beta_max: f32) {
        let tilt_inc: f32;
        let angle_lim: f32;
        let mut slope_max: f32 = 0.;
        let mut pix_angle: f32;
        let mut sin_max: f32 = 0.;
        let mut cos_max: f32 = 0.;
        let mut slope_min: f32 = 0.;
        let mut sin_min: f32 = 0.;
        let mut cos_min: f32 = 0.;
        let mut x: f32;
        let mut y: f32;
        let delx: f32;
        let dely: f32;
        let mut ysq: f32;
        let mut dist_sq: f32;
        let mut atten: f32;
        let mut ratio: f32;
        let mut use_alpha: f32;
        let mut ind: usize;
        let mut index: i32;
        let nx_div2p1: i32;
        let pixel_margin: f32 = 4.;
        let alpha_margin: f32 = 1.1;
        let corner_cutoff: f32 = 0.2352f32;
        let corner_falloff: f32 = 0.03f32;
        if self.m_clean_super_fft <= 0 {
            return;
        }
        let clean_corners = (self.m_clean_super_fft & 1) != 0;
        let mut clean_wedge = (self.m_clean_super_fft & 2) != 0;
        //mrcWriteFFT("fft-1.mrc", mOutSliceArr, mNxOutPad, mNyOutPad, 1);
        delx = (1. / self.m_nx_out_pad as f64) as f32;
        dely = (1. / self.m_ny_out_pad as f64) as f32;
        nx_div2p1 = self.m_nx_out_pad / 2 + 1;

        // If cleaning the wedge, determine slope criteria and sin/cos for rotating to
        // horizontal in pixel coordinates for testing the pixel distance
        if clean_wedge {
            angle_lim = (89.5 * RADIANS_PER_DEGREE) as f32;
            tilt_inc = (beta_max - beta_min) / b3dmax!(1, self.m_num_views - 1) as f32;
            if beta_max as f64 + tilt_inc as f64 / 2. > angle_lim as f64
                || (beta_min as f64 - tilt_inc as f64 / 2.) < -angle_lim as f64
            {
                clean_wedge = false;
                if !clean_corners {
                    return;
                }
            } else {
                use_alpha = self.m_min_cos_alpha;
                if use_alpha < 1. {
                    use_alpha =
                        (1. - (1. - self.m_min_cos_alpha as f64) * alpha_margin as f64) as f32;
                }
                slope_max =
                    ((beta_max as f64 + tilt_inc as f64 / 2.).tan() / use_alpha as f64) as f32;
                pix_angle = (slope_max * delx / dely).atan();
                sin_max = pix_angle.sin();
                cos_max = pix_angle.cos();
                //PRINT4(slopeMax, pixAngle, sinMax, cosMax);
                slope_min =
                    ((beta_min as f64 - tilt_inc as f64 / 2.).tan() / use_alpha as f64) as f32;
                pix_angle = (slope_min * delx / dely).atan();
                sin_min = pix_angle.sin();
                cos_min = pix_angle.cos();
                //PRINT4(slopeMin, pixAngle, sinMin, cosMin);
            }
        }

        //int numCorn =0, numAxis = 0, numAbove =0, numBelow = 0;
        for iy in 0..self.m_ny_out_pad {
            y = iy as f32 * dely;
            index = iy * nx_div2p1;
            if y as f64 > 0.5 {
                y = y - 1.0f32;
            }
            ysq = y * y;
            for ix in 0..nx_div2p1 {
                x = ix as f32 * delx;
                ind = (2 * (index + ix)) as usize;
                dist_sq = x * x + ysq;

                // A hard edge is worrisome, so taper it in case there is data out there
                if clean_corners && dist_sq > corner_cutoff {
                    if dist_sq < corner_cutoff + corner_falloff {
                        atten = (corner_cutoff + corner_falloff - dist_sq) / corner_falloff;
                        self.m_out_slice_arr[ind] *= atten;
                        self.m_out_slice_arr[ind + 1] *= atten;
                    } else {
                        self.m_out_slice_arr[ind] = 0.;
                        self.m_out_slice_arr[ind + 1] = 0.;
                    }
                    //numCorn++;
                } else if clean_wedge {
                    if ix != 0 {
                        ratio = y / x;
                        if ratio > slope_max
                            && -sin_max * ix as f32 + cos_max * y / dely > pixel_margin
                        {
                            self.m_out_slice_arr[ind] = 0.;
                            self.m_out_slice_arr[ind + 1] = 0.;
                            //  numAbove++;
                        } else if ratio < slope_min
                            && -sin_min * ix as f32 + cos_min * y / dely < -pixel_margin
                        {
                            self.m_out_slice_arr[ind] = 0.;
                            self.m_out_slice_arr[ind + 1] = 0.;
                            //  numBelow++;
                        }
                    } else if y / dely < pixel_margin || y / dely > pixel_margin {
                        self.m_out_slice_arr[ind] = 0.;
                        self.m_out_slice_arr[ind + 1] = 0.;
                        //numAxis++;
                    }
                }
            }
        }
        //PRINT4(numCorn, numAxis , numAbove , numBelow);
        //mrcWriteFFT("fft-2.mrc", mOutSliceArr, mNxOutPad, mNyOutPad, 1);
    }
}

impl Tilt {
    /// `Tilt::compose` (`tilt.cpp:2222`): interpolate the output slice
    /// `lsliceOut` from vertical slices in the ring buffer, where
    /// `lvertSliceStart` and `lVertSliceEnd` are the starting and ending slices
    /// in the ring buffer, `idir` is the direction of reconstruction, and
    /// `iringStart` is the position of `lvertSliceStart` in the ring buffer.
    fn compose(
        &mut self,
        lslice_out: i32,
        lvert_slice_start: i32,
        l_vert_slice_end: i32,
        idir: i32,
        iring_start: i32,
        compose_fill: f32,
    ) {
        let mut ind1: [i64; 4] = [0; 4];
        let mut ind2: [i64; 4] = [0; 4];
        let mut ind3: [i64; 4] = [0; 4];
        let mut ind4: [i64; 4] = [0; 4];
        let mut ibase_ind: i64;
        let tan_alpha: f32;
        let vert_cen: f32;
        let mut centered_j: f32;
        let mut centered_l: f32;
        let mut v_slice: f32;
        let mut vy_centered: f32;
        let mut fx: f32;
        let mut vy: f32;
        let mut fy: f32;
        let mut f22: f32;
        let mut f23: f32;
        let mut f32_: f32;
        let mut f33: f32;
        let mut fxsq: f32;
        let mut fysq: f32;
        let mut fxcub: f32;
        let mut fycub: f32;
        let mut iv_slice: i32;
        let mut if_miss: i32;
        let mut lv_slice: i32;
        let mut iring: i32;
        let mut ivy: i32;
        let mut ind_cen: i32;
        let mut jnd5: i64;
        let mut jnd2: i64;
        let mut fx1: f32;
        let mut fx2: f32;
        let mut fx3: f32;
        let mut fx4: f32;
        let mut fy1: f32;
        let mut fy2: f32;
        let mut fy3: f32;
        let mut fy4: f32;
        let mut v1: f32;
        let mut v2: f32;
        let mut v3: f32;
        let mut v4: f32;
        let mut f5: f32;
        let mut f2: f32;
        let mut f8: f32;
        let mut f4: f32;
        let mut f6: f32;
        let mut jnd8: i64;
        let mut jnd4: i64;
        let mut jnd6: i64;
        let mut num_fill: i32;
        //
        // 12/12/09: stopped reading base here, read on output; eliminate zeroing
        //
        tan_alpha = self.m_sin_alpha[0] / self.m_cos_alpha[0];
        vert_cen = ((self.m_ithick_bp / 2) as f64 + 0.5) as f32;
        fx = 0.;
        fy = 0.;
        num_fill = 0;
        let vsa = &self.m_vert_slice_arr;
        let osa = &mut self.m_out_slice_arr;
        //
        // loop on lines of data
        //
        for j in 1..=self.m_ithick_out {
            centered_j =
                (j as f64 - ((self.m_ithick_out / 2) as f64 + 0.5) - self.m_y_offset as f64) as f32;
            centered_l = lslice_out as f32 - self.m_center_slice;
            //
            // calculate slice number and y position in vertical slices
            //
            v_slice = centered_l * self.m_cos_alpha[0] - centered_j * self.m_sin_alpha[0]
                + self.m_center_slice;
            vy_centered = centered_l * self.m_sin_alpha[0] + centered_j * self.m_cos_alpha[0];
            iv_slice = v_slice as i32;
            fx = v_slice - iv_slice as f32;
            if_miss = 0;
            //
            // for each of 4 slices needed for cubic interpolation, initialize
            // data indexes at zero then see if slice exists in ring
            //
            for i in 1..=4usize {
                ind1[i - 1] = 0;
                ind2[i - 1] = 0;
                ind3[i - 1] = 0;
                ind4[i - 1] = 0;
                lv_slice = iv_slice + i as i32 - 2;
                if idir * (lv_slice - lvert_slice_start) >= 0
                    && idir * (l_vert_slice_end - lv_slice) >= 0
                {
                    //
                    // if slice exists, get base index for the slice, compute the
                    // y index in the slice, then set the 4 data indexes if they
                    // are within the slice
                    //
                    iring = idir * (lv_slice - lvert_slice_start) + iring_start;
                    if iring > self.m_num_vert_needed {
                        iring = iring - self.m_num_vert_needed;
                    }
                    ibase_ind =
                        1 + (iring - 1) as i64 * self.m_ithick_bp as i64 * self.m_iwidth as i64;
                    vy = vy_centered + vert_cen
                        - b3dnint!(tan_alpha * (lv_slice as f32 - self.m_center_slice)) as f32
                        + self.m_y_offset / self.m_cos_alpha[0];
                    ivy = vy as i32;
                    fy = vy - ivy as f32;
                    if ivy - 1 >= 1 && ivy - 1 <= self.m_ithick_bp {
                        ind1[i - 1] = ibase_ind + self.m_iwidth as i64 * (ivy - 2) as i64;
                    }
                    if ivy >= 1 && ivy <= self.m_ithick_bp {
                        ind2[i - 1] = ibase_ind + self.m_iwidth as i64 * (ivy - 1) as i64;
                    }
                    if ivy + 1 >= 1 && ivy + 1 <= self.m_ithick_bp {
                        ind3[i - 1] = ibase_ind + self.m_iwidth as i64 * ivy as i64;
                    }
                    if ivy + 2 >= 1 && ivy + 2 <= self.m_ithick_bp {
                        ind4[i - 1] = ibase_ind + self.m_iwidth as i64 * (ivy + 1) as i64;
                    }
                }
                if ind1[i - 1] == 0 || ind2[i - 1] == 0 || ind3[i - 1] == 0 || ind4[i - 1] == 0 {
                    if_miss = 1;
                }
            }
            ibase_ind = (j - 1) as i64 * self.m_iwidth as i64;
            if self.m_interp_ord_xtilt > 2 && if_miss == 0 {
                //
                // cubic interpolation if selected, and no data missing
                //
                fxsq = fx * fx;
                fxcub = fxsq * fx;
                fysq = fy * fy;
                fycub = fysq * fy;
                fx1 = (2. * fxsq as f64 - fxcub as f64 - fx as f64) as f32;
                fx2 = (fxcub as f64 - 2. * fxsq as f64 + 1.) as f32;
                fx3 = fxsq + fx - fxcub;
                fx4 = fxcub - fxsq;
                fy1 = (2. * fysq as f64 - fycub as f64 - fy as f64) as f32;
                fy2 = (fycub as f64 - 2. * fysq as f64 + 1.) as f32;
                fy3 = fysq + fy - fycub;
                fy4 = fycub - fysq;
                let v = |k: i64| vsa[(k - 1) as usize];
                for i in 1..=self.m_iwidth as i64 {
                    v1 = fx1 * v(ind1[0]) + fx2 * v(ind1[1]) + fx3 * v(ind1[2]) + fx4 * v(ind1[3]);
                    v2 = fx1 * v(ind2[0]) + fx2 * v(ind2[1]) + fx3 * v(ind2[2]) + fx4 * v(ind2[3]);
                    v3 = fx1 * v(ind3[0]) + fx2 * v(ind3[1]) + fx3 * v(ind3[2]) + fx4 * v(ind3[3]);
                    v4 = fx1 * v(ind4[0]) + fx2 * v(ind4[1]) + fx3 * v(ind4[2]) + fx4 * v(ind4[3]);
                    osa[(ibase_ind + i - 1) as usize] = fy1 * v1 + fy2 * v2 + fy3 * v3 + fy4 * v4;
                    for k in 1..=4usize {
                        ind1[k - 1] = ind1[k - 1] + 1;
                        ind2[k - 1] = ind2[k - 1] + 1;
                        ind3[k - 1] = ind3[k - 1] + 1;
                        ind4[k - 1] = ind4[k - 1] + 1;
                    }
                }
            } else if self.m_interp_ord_xtilt == 2 && if_miss == 0 {
                //
                // quadratic interpolation if selected, and no data missing
                // shift to next column or row if fractions > 0.5
                //
                ind_cen = 2;
                if fx as f64 > 0.5 {
                    ind_cen = 3;
                    fx = (fx as f64 - 1.) as f32;
                }
                let ic = ind_cen as usize;
                if fy as f64 <= 0.5 {
                    jnd5 = ind2[ic - 1];
                    jnd2 = ind1[ic - 1];
                    jnd8 = ind3[ic - 1];
                    jnd4 = ind2[ic - 2];
                    jnd6 = ind2[ic];
                } else {
                    fy = (fy as f64 - 1.) as f32;
                    jnd5 = ind3[ic - 1];
                    jnd2 = ind2[ic - 1];
                    jnd8 = ind4[ic - 1];
                    jnd4 = ind3[ic - 2];
                    jnd6 = ind3[ic];
                }
                //
                // get coefficients and do the interpolation
                //
                fxsq = fx * fx;
                fysq = fy * fy;
                f5 = (1. - fxsq as f64 - fysq as f64) as f32;
                f2 = ((fysq - fy) as f64 / 2.) as f32;
                f8 = f2 + fy;
                f4 = ((fxsq - fx) as f64 / 2.) as f32;
                f6 = f4 + fx;
                for i in 1..=self.m_iwidth as i64 {
                    osa[(ibase_ind + i - 1) as usize] = f5 * vsa[(jnd5 - 1) as usize]
                        + f2 * vsa[(jnd2 - 1) as usize]
                        + f4 * vsa[(jnd4 - 1) as usize]
                        + f6 * vsa[(jnd6 - 1) as usize]
                        + f8 * vsa[(jnd8 - 1) as usize];
                    jnd5 = jnd5 + 1;
                    jnd2 = jnd2 + 1;
                    jnd4 = jnd4 + 1;
                    jnd6 = jnd6 + 1;
                    jnd8 = jnd8 + 1;
                }
            } else {
                //
                // linear interpolation
                //
                // print *,j, ind2[1], ind2[2], ind3[1], ind3[2]
                if ind2[1] == 0 || ind2[2] == 0 || ind3[1] == 0 || ind3[2] == 0 {
                    //
                    // if there is a problem, see if it can be rescued by shifting
                    // center back to left or below
                    //
                    if (fx as f64) < 0.02
                        && ind2[0] != 0
                        && ind3[0] != 0
                        && ind2[1] != 0
                        && ind3[1] != 0
                    {
                        fx = fx + 1.;
                        ind2[2] = ind2[1];
                        ind2[1] = ind2[0];
                        ind3[2] = ind3[1];
                        ind3[1] = ind3[0];
                    } else if (fy as f64) < 0.02
                        && ind1[1] != 0
                        && ind1[2] != 0
                        && ind2[1] != 0
                        && ind3[1] != 0
                    {
                        fy = fy + 1.;
                        ind3[1] = ind2[1];
                        ind2[1] = ind1[1];
                        ind3[2] = ind2[2];
                        ind2[2] = ind1[2];
                    }
                }
                //
                // do linear interpolation if conditions are right, otherwise fill
                //
                if ind2[1] != 0 && ind2[2] != 0 && ind3[1] != 0 && ind3[2] != 0 {
                    f22 = ((1. - fy as f64) * (1. - fx as f64)) as f32;
                    f23 = ((1. - fy as f64) * fx as f64) as f32;
                    f32_ = (fy as f64 * (1. - fx as f64)) as f32;
                    f33 = fy * fx;
                    for i in 1..=self.m_iwidth as i64 {
                        osa[(ibase_ind + i - 1) as usize] = f22 * vsa[(ind2[1] - 1) as usize]
                            + f23 * vsa[(ind2[2] - 1) as usize]
                            + f32_ * vsa[(ind3[1] - 1) as usize]
                            + f33 * vsa[(ind3[2] - 1) as usize];
                        ind2[1] = ind2[1] + 1;
                        ind2[2] = ind2[2] + 1;
                        ind3[1] = ind3[1] + 1;
                        ind3[2] = ind3[2] + 1;
                    }
                } else {
                    // print *,'filling', j
                    for i in 1..=self.m_iwidth as i64 {
                        osa[(i + ibase_ind - 1) as usize] = compose_fill;
                    }
                    num_fill = num_fill + 1;
                }
            }
        }
        // if (nfill .ne. 0) print *,nfill, ' lines filled, edgefill =', composeFill
        let _ = num_fill;
    }

    /// `Tilt::decompose` (`tilt.cpp:2433`): interpolate a vertical slice
    /// `lslice` from input slices in the read-in ring buffer, where
    /// `lReadStart` and `lreadEnd` are the starting and ending slices in the
    /// ring buffer, `iringStart` is the position of `lReadStart` in the ring
    /// buffer, and `vertArr` is where to place the slice.
    fn decompose(
        &self,
        lslice: i32,
        l_read_start: i32,
        lread_end: i32,
        iring_start: i32,
        vert_arr: &mut [f32],
    ) {
        let tan_alpha: f32;
        let out_cen: f32;
        let centered_l: f32;
        let v_slice_centered: f32;
        let mut vy_centered: f32;
        let mut out_slice: f32;
        let mut out_ypos: f32;
        let mut f11: f32;
        let mut f12: f32;
        let mut f21: f32;
        let mut f22: f32;
        let mut fx: f32;
        let mut fy: f32;
        let mut ibase_vert: i32;
        let mut ibase1: i32;
        let mut ibase2: i32;
        let mut iout_slice: i32;
        let mut j_out: i32;
        let mut iring: i32;

        tan_alpha = self.m_sin_alpha[0] / self.m_cos_alpha[0];
        out_cen = ((self.m_ithick_out / 2) as f64 + 0.5) as f32;
        centered_l = lslice as f32 - self.m_center_slice;
        v_slice_centered = centered_l;
        let ria = &self.m_read_in_array;
        //
        // loop on lines of data
        for j in 1..=self.m_ithick_bp {
            ibase_vert = (j - 1) * self.m_iwidth;
            //
            // calculate slice number and y position in input slices
            //
            vy_centered = (j as f64
                - ((self.m_ithick_bp / 2) as f64 + 0.5 - b3dnint!(tan_alpha * centered_l) as f64
                    + (self.m_y_offset / self.m_cos_alpha[0]) as f64))
                as f32;
            out_slice = self.m_center_slice
                + v_slice_centered * self.m_cos_alpha[0]
                + vy_centered * self.m_sin_alpha[0];
            out_ypos = out_cen + self.m_y_offset - v_slice_centered * self.m_sin_alpha[0]
                + vy_centered * self.m_cos_alpha[0];
            // print *,j, vycen, outsl, outj
            // if (outsl >= lreadStart - 0.5 .and. outsl <= lreadEnd + 0.5 &
            // .and. outj >= 0.5 .and.outj <= ithickOut + 0.5) then
            //
            // For a legal position, get interpolation integers and fractions,
            // adjust if within half pixel of end
            iout_slice = out_slice as i32;
            fx = out_slice - iout_slice as f32;
            if iout_slice < l_read_start {
                iout_slice = l_read_start;
                fx = 0.;
            } else if iout_slice > lread_end - 1 {
                iout_slice = lread_end - 1;
                fx = 1.;
            }
            j_out = out_ypos as i32;
            fy = out_ypos - j_out as f32;
            if j_out < 1 {
                j_out = 1;
                fy = 0.;
            } else if j_out > self.m_ithick_out - 1 {
                j_out = self.m_ithick_out - 1;
                fy = 1.;
            }
            //
            // Get slice indexes in ring
            iring = iout_slice - l_read_start + iring_start;
            if iring > self.m_num_read_need {
                iring = iring - self.m_num_read_need;
            }
            ibase1 = (iring - 1) * self.m_ithick_out * self.m_iwidth + (j_out - 1) * self.m_iwidth;
            iring = iout_slice + 1 - l_read_start + iring_start;
            if iring > self.m_num_read_need {
                iring = iring - self.m_num_read_need;
            }
            ibase2 = (iring - 1) * self.m_ithick_out * self.m_iwidth + (j_out - 1) * self.m_iwidth;
            //
            // Interpolate line
            f11 = ((1. - fy as f64) * (1. - fx as f64)) as f32;
            f12 = ((1. - fy as f64) * fx as f64) as f32;
            f21 = (fy as f64 * (1. - fx as f64)) as f32;
            f22 = fy * fx;
            let w = self.m_iwidth as usize;
            let (b1, b2, bv) = (ibase1 as usize, ibase2 as usize, ibase_vert as usize);
            for i in 0..w {
                vert_arr[bv + i] = f11 * ria[b1 + i]
                    + f12 * ria[b2 + i]
                    + f21 * ria[b1 + i + w]
                    + f22 * ria[b2 + i + w];
            }
            // else
            //
            // Otherwise fill line
            // do i = 0, iwidth - 1
            // array(ibasev+i) = dmeanIn
            // enddo
            // endif
        }
    }

    /// `Tilt::dumpSlice` (`tilt.cpp:2514`): write the output to a file.
    fn dump_slice(&mut self, lslice: i32, dmin: &mut f32, dmax: &mut f32, dtot8: &mut f64) {
        let mut num_par_extra_lines: i32;
        let iend: i32;
        let mut index: i32;
        let mut ind_del: i32;
        let mut out_ind: i32;
        let mut first_slice: i32;
        let num_write: i32;
        let mut num_pre_write: i32;
        let mut num_pre_chunk: i32;
        let mut per_chunk: i32;
        let mut num_done: i32;
        let mut num_do: i32;
        let fill: f32;
        let mut dtmp8: f64;
        //
        // If adding to a base rec, read in each line and add scaled values
        let out_is_read_in = self.m_num_sirt_iter > 0 && self.m_if_alpha >= 0;
        let mut out_arr = if out_is_read_in {
            std::mem::take(&mut self.m_read_in_array)
        } else {
            std::mem::take(&mut self.m_out_slice_arr)
        };
        if self.m_read_base_rec {
            // A base rec is read only without SIRT, so `outArr` is `mOutSliceArr`,
            // the array this block names explicitly.
            index = 0;
            unsafe { iiu_set_position(3, lslice - 1, 0) };
            if self.m_iter_for_report > 0 {
                self.sample_for_report(
                    &out_arr,
                    lslice,
                    self.m_ithick_out,
                    1,
                    self.m_out_scale,
                    self.m_out_add,
                );
            }
            for _j in 1..=self.m_ithick_out {
                if unsafe { iiu_read_lines(3, self.m_proj_line.as_mut_ptr().cast(), 1) } != 0 {
                    exit_error(b"Reading line for subtraction");
                }
                let pl = &self.m_proj_line;
                let b = index as usize;
                if self.m_rec_subtraction {
                    //
                    // SIRT subtraction from base with possible sign constraints
                    if self.m_isign_constraint == 0 {
                        for i in 0..self.m_iwidth as usize {
                            out_arr[b + i] =
                                pl[i] / self.m_out_scale - self.m_out_add - out_arr[b + i];
                        }
                    } else if self.m_isign_constraint < 0 {
                        for i in 0..self.m_iwidth as usize {
                            out_arr[b + i] = b3dmin!(
                                0.,
                                (pl[i] / self.m_out_scale - self.m_out_add - out_arr[b + i]) as f64
                            ) as f32;
                        }
                    } else {
                        for i in 0..self.m_iwidth as usize {
                            out_arr[b + i] = b3dmax!(
                                0.,
                                (pl[i] / self.m_out_scale - self.m_out_add - out_arr[b + i]) as f64
                            ) as f32;
                        }
                    }
                } else {
                    //
                    // Generic addition of the scaled base data
                    for i in 0..self.m_iwidth as usize {
                        out_arr[b + i] += pl[i] / self.m_base_out_scale - self.m_base_out_add;
                    }
                }
                index = index + self.m_iwidth;
            }
        }
        //
        num_par_extra_lines = 100;
        iend = self.m_ithick_out * self.m_iwidth;
        //
        // scale
        // DNM simplified and fixed bug in getting min/max/mean
        dtmp8 = 0.;
        //
        // DNM 9/23/04: incorporate reprojBP option
        //
        if self.m_reproj_bp != 0 {
            //--------------scale
            for i in 0..iend as usize {
                out_arr[i] = (out_arr[i] + self.m_out_add) * self.m_out_scale;
            }
            //
            // Fill value assumes edge fill value
            //
            fill = (self.m_edge_fill + self.m_out_add) * self.m_out_scale;
            let mut proj_line = std::mem::take(&mut self.m_proj_line);
            for j in 1..=self.m_num_reproj {
                let i = ((j - 1) * self.m_iwidth + 1) as usize;
                self.re_project(
                    &out_arr,
                    self.m_iwidth,
                    self.m_ithick_bp,
                    self.m_iwidth,
                    self.m_sin_reproj[(j - 1) as usize],
                    self.m_cos_reproj[(j - 1) as usize],
                    &self.m_xray_start[i - 1..],
                    &self.m_yray_start[i - 1..],
                    &self.m_num_pix_in_ray[i - 1..],
                    self.m_max_ray_pixels[(j - 1) as usize],
                    fill,
                    &mut proj_line,
                    0,
                    0,
                );
                for i in 0..self.m_iwidth as usize {
                    *dmin = if *dmin < proj_line[i] {
                        *dmin
                    } else {
                        proj_line[i]
                    };
                    *dmax = if *dmax > proj_line[i] {
                        *dmax
                    } else {
                        proj_line[i]
                    };
                    dtmp8 = dtmp8 + proj_line[i] as f64;
                }
                let mut iy = lslice - self.m_islice_start;
                if self.m_min_tot_slice > 0 {
                    iy = lslice - self.m_min_tot_slice;
                }
                unsafe {
                    par_wrt_posn(2, j - 1, iy);
                    par_wrt_lin(2, proj_line.as_mut_ptr().cast());
                }
            }
            self.m_proj_line = proj_line;
            *dtot8 = *dtot8 + dtmp8;
            if out_is_read_in {
                self.m_read_in_array = out_arr;
            } else {
                self.m_out_slice_arr = out_arr;
            }
            return;
        }
        //
        // If there is an adjustment factor, get the mean and do a one-time adjustment
        // to mOutAdd that will reduce the scaled mean by that factor
        if self.m_adjust_out_add_fac as f64 > 0. {
            for i in 0..iend as usize {
                dtmp8 = dtmp8 + out_arr[i] as f64;
            }
            dtmp8 /= iend as f64;
            self.m_out_add =
                ((dtmp8 + self.m_out_add as f64) / self.m_adjust_out_add_fac as f64 - dtmp8) as f32;
            dtmp8 = 0.;
            self.m_adjust_out_add_fac = 0.;
        }
        //
        //--------------scale and get min / max / sum
        for i in 0..iend as usize {
            out_arr[i] = (out_arr[i] + self.m_out_add) * self.m_out_scale;
            *dmin = if *dmin < out_arr[i] {
                *dmin
            } else {
                out_arr[i]
            };
            *dmax = if *dmax > out_arr[i] {
                *dmax
            } else {
                out_arr[i]
            };
            dtmp8 = dtmp8 + out_arr[i] as f64;
            //
        }
        *dtot8 = *dtot8 + dtmp8;
        //
        // Dump slice
        if self.m_perpendicular != 0 {
            // ....slices correspond to sections of map
            unsafe { par_wrt_sec(2, out_arr.as_mut_ptr().cast()) };
        } else {
            // ....slices must be properly stored
            // Take each line of array and place it in the correct section of the map.
            index = 0;
            ind_del = 1;
            if self.m_rotate_by90 != 0 {
                index = (self.m_ithick_out - 1) * self.m_iwidth;
                ind_del = -1;
            }
            first_slice = lslice;
            num_do = self.m_islice_end + 1 - self.m_islice_start;
            if num_do / 10 == 0 {
                // `lslice % (numDo / 10)` is an integer division by zero for fewer
                // than 10 slices: the reference dies with SIGFPE here, and so does
                // this translation.
                let _ = ImodFile::Stdout.flush();
                unsafe { libc::raise(libc::SIGFPE) };
            }
            if lslice % (num_do / 10) == 0 {
                printf!(
                    "Finished slice %d of %d\n",
                    ci(lslice + 1 - self.m_islice_start),
                    ci(num_do)
                );
                let _ = ImodFile::Stdout.flush();
            }

            // Put lines in buffer and return if not full or done
            if self.m_num_out_buf_slices > 0 {
                out_ind = self.m_cur_out_buf_slice * self.m_iwidth;
                let w = self.m_iwidth as usize;
                for _j in 0..self.m_ithick_out {
                    self.m_out_buffer[out_ind as usize..out_ind as usize + w]
                        .copy_from_slice(&out_arr[index as usize..index as usize + w]);
                    out_ind += self.m_num_out_buf_slices * self.m_iwidth;
                    index += ind_del * self.m_iwidth;
                }
                self.m_cur_out_buf_slice += 1;
                if self.m_cur_out_buf_slice < self.m_num_out_buf_slices
                    && lslice != self.m_islice_end
                {
                    if out_is_read_in {
                        self.m_read_in_array = out_arr;
                    } else {
                        self.m_out_slice_arr = out_arr;
                    }
                    return;
                }

                // Write the buffer: set up increment and starting slice
                index = 0;
                ind_del = self.m_num_out_buf_slices;
                first_slice = lslice + 1 - self.m_cur_out_buf_slice;
            }

            num_par_extra_lines = (self.m_islice_end + 1 - self.m_islice_start) / 10;
            num_par_extra_lines = b3dmax!(100, b3dmin!(400, num_par_extra_lines));
            num_write = lslice + 1 - first_slice;

            // Loop on Z and write line(s) at each Z
            for j in 1..=self.m_ithick_out {
                let err = unsafe {
                    iiu_set_position(2, j - 1, first_slice - self.m_islice_start);
                    iiu_write_lines(
                        2,
                        if self.m_num_out_buf_slices > 0 {
                            self.m_out_buffer[index as usize..].as_mut_ptr().cast()
                        } else {
                            out_arr[index as usize..].as_mut_ptr().cast()
                        },
                        num_write,
                    )
                };
                if err != 0 {
                    exit_error(b"Writing data");
                }
                //
                // DNM 2/29/01: partially demangle the parallel output by writing
                // up to 100-400 lines at a time in this plane, subtracting the ones
                // already written from the buffer if any
                //
                if lslice > self.m_last_pre_write_line
                    && lslice != self.m_islice_end
                    && num_par_extra_lines - num_write > 0
                {
                    num_pre_write =
                        b3dmin!(num_par_extra_lines - num_write, self.m_islice_end - lslice);
                    num_pre_chunk = (num_pre_write + self.m_ithick_out - 1) / self.m_ithick_out;
                    per_chunk = num_pre_write / num_pre_chunk;
                    num_done = 0;
                    while num_done < num_pre_write {
                        num_do = b3dmin!(per_chunk, num_pre_write - num_done);
                        if unsafe { iiu_write_lines(2, out_arr.as_mut_ptr().cast(), num_do) } != 0 {
                            exit_error(b"Pre-writing lines");
                        }
                        num_done += num_do;
                    }
                    self.m_last_pre_write_line = lslice + num_pre_write;
                }
                index = index + ind_del * self.m_iwidth;
            }
            self.m_cur_out_buf_slice = 0;
        }
        if out_is_read_in {
            self.m_read_in_array = out_arr;
        } else {
            self.m_out_slice_arr = out_arr;
        }
    }

    /// `Tilt::maskSlice` (`tilt.cpp:2695`): mask out (blur) the edges of the
    /// slice in `array`.
    fn mask_slice(&self, array: &mut [f32], ithick_mask: i32) {
        let mut idir: i32;
        let mut limit: i32;
        let mut nsum: i32;
        let num_smooth: i32;
        let num_taper: i32;
        let mut index: i32;
        let mut ibase: i32;
        let mut sum: f32;
        let mut edge_mean: f32;
        let mut frac: f32;
        let se = &self.m_ix_unmasked_se;
        let w = self.m_iwidth;
        //
        num_smooth = 10;
        num_taper = 10;
        idir = -1;
        limit = 1;
        for lr in 0..2 {
            //
            // find mean along edge
            sum = 0.;
            for j in 0..ithick_mask {
                sum = sum + array[(j * w + se[(lr + 2 * j) as usize] - 1) as usize];
            }
            edge_mean = sum / ithick_mask as f32;
            //
            // For each line, sum progressively more pixels along edge out to a limit
            for j in 1..=ithick_mask {
                // This is "-1-based" to take care of SE and index being 1-based
                ibase = (j - 1) * w - 1;
                index = se[(lr + 2 * (j - 1)) as usize] + idir;
                sum = array[(ibase + se[(lr + 2 * (j - 1)) as usize]) as usize];
                nsum = 1;
                for i in 1..=num_smooth {
                    if idir * (limit - index) < 0 {
                        break;
                    }
                    if j + i <= ithick_mask {
                        nsum = nsum + 1;
                        sum = sum
                            + array[(ibase + i * w + se[(lr + 2 * (j + i - 1)) as usize]) as usize];
                    }
                    if j - i >= 1 {
                        nsum = nsum + 1;
                        sum = sum
                            + array[(ibase - i * w + se[(lr + 2 * (j - i - 1)) as usize]) as usize];
                    }
                    //
                    // Taper partway to mean over this smoothing distance
                    frac = (i as f64 / ((num_taper + num_smooth) as f64 + 1.)) as f32;
                    array[(ibase + index) as usize] =
                        ((1. - frac as f64) * sum as f64 / nsum as f64 + (frac * edge_mean) as f64)
                            as f32;
                    index = index + idir;
                }
                //
                // Then taper rest of way down to mean over more pixels, then fill
                // with mean
                for i in 1..=num_taper {
                    if idir * (limit - index) < 0 {
                        break;
                    }
                    frac =
                        ((i + num_smooth) as f64 / ((num_taper + num_smooth) as f64 + 1.)) as f32;
                    array[(ibase + index) as usize] =
                        ((1. - frac as f64) * sum as f64 / nsum as f64 + (frac * edge_mean) as f64)
                            as f32;
                    index = index + idir;
                }

                // Beware of loops that can run both ways
                let mut i = index;
                while (limit - i) * idir >= 0 {
                    array[(ibase + i) as usize] = edge_mean;
                    i += idir;
                }
            }
            //
            // Set up for other direction
            limit = self.m_iwidth;
            idir = 1;
        }
    }
}

impl Tilt {
    /// `Tilt::inputParameters` (`tilt.cpp:2765`): the giant input routine, gets
    /// input and sets up almost all parameters.
    fn input_parameters(&mut self, argv: &[String]) {
        const LIMNUM: usize = 100;
        //
        let mut mpxyz: [i32; 3] = [0; 3];
        let mut nout_xyz: [i32; 3] = [0; 3];
        let mut nrec_xyz: [i32; 3] = [0; 3];
        let mut nvs_xyz: [i32; 3] = [0; 3];
        let mut max_tex_2d: [i32; 2] = [0; 2];
        let mut max_tex_layer: [i32; 3] = [0; 3];
        let mut max_tex_3d: [i32; 3] = [0; 3];
        let mut out_hdr_tilt: [f32; 3] = [90., 0., 0.];
        let mut cell: [f32; 6] = [0., 0., 0., 90., 90., 90.];
        let deg_to_rad: f32 = RADIANS_PER_DEGREE as f32;
        let delta: [f32; 3];
        let mut imod_to_project: Option<Imod> = None;
        //
        let mut card: Vec<u8> = Vec::new();
        let mut input_file: Vec<u8> = Vec::new();
        let mut output_file: Vec<u8> = Vec::new();
        let mut rec_file: Vec<u8> = Vec::new();
        let mut base_file: Vec<u8> = Vec::new();
        let mut bound_file: Option<Vec<u8>>;
        let mut vert_bound_file: Option<Vec<u8>>;
        let mut vert_out_file: Option<Vec<u8>>;
        let mut angle_output: Option<Vec<u8>>;
        let mut transform_file: Option<Vec<u8>>;
        let mut defocus_file: Option<Vec<u8>>;
        let mut nfields: i32;
        let mut inum = [0i32; LIMNUM];
        let mut nproj_xyz: [i32; 3] = [0; 3];
        let mut list_temp: Vec<i32>;
        let mut max_needs: Vec<i32>;
        let mut xnum = [0f32; LIMNUM];
        let mut raw_xfs: Vec<f32> = Vec::new();
        //
        let mut iv_exclude: Vec<i32>;
        let mut iv_reproj: Vec<i32> = Vec::new();
        let mut pack_local: Vec<f32>;
        let mut ang_reproj: Vec<f32>;
        let mut temp_arr: Vec<f32>;
        let mut mode: i32 = 0;
        let mut new_angles: i32;
        let mut if_tilt_file: i32;
        let mut num_view_use: i32;
        let mut num_view_exclude: i32;
        let mut num_need_eval: i32;
        let mut del_angle: f32;
        let mut comp_factor: f32;
        let mut global_alpha: f32;
        let mut x_offset: f32;
        let mut scale_local: f32;
        let mut rad_max: f32 = 0.;
        let mut rad_fall: f32;
        let mut xoff_adj: f32 = 0.;
        let mut irad_fall: i32;
        let mut irad_max: i32;
        let mut num_compress: i32;
        let mut nx_full: i32;
        let mut ny_full: i32;
        let mut ix_subset: i32;
        let mut iy_subset: i32;
        let mut ix_subset_in: i32;
        let mut kti: i32;
        let mut ind_base: i32 = 0;
        let mut ipos: i32 = 0;
        let id_type: i32;
        let _lens: i32;
        let mut nx_full_in: i32;
        let mut load_len: i32;
        let nd1: i32;
        let _nd2: i32;
        let mut num_slices: i32;
        let mut indi: i32;
        let mut i: i32;
        let mut iex: i32;
        let num_view_orig: i32;
        let mut iv: i32;
        let vd1: f32;
        let vd2: f32;
        let del_theta: f32;
        let mut theta: f32;
        let mut theta_view: f32;
        let mut stretch: f32 = 0.;
        let mut str_phi: f32 = 0.;
        let mut smag: f32 = 0.;
        let mut subset_load_ratio: f32;
        let mut num_input: i32;
        let mut num_exclude_list: i32;
        let mut j: i32;
        let mut ind: i32;
        let mut num_ent: i32 = 0;
        let if_radial: i32;
        let mut if_true_sigma: i32;
        let nproj_pad: i32;
        let mut if_zfactors: i32;
        let mut local_zfacs: i32 = 0;
        let mut adjust_origin: i32;
        let mut slice_adj: i32;
        let mut if_thick_in: i32;
        let mut if_slice_in: i32;
        let mut if_width_in: i32;
        let mut image_binned: i32;
        let mut if_subset_in: i32;
        let mut ierr: i32;
        let use_xproj: i32;
        let use_yproj: i32;
        let mut pixel_local: f32 = 0.;
        let mut dmin_tmp: f32 = 0.;
        let mut dmax_tmp: f32 = 0.;
        let mut dmean_tmp: f32 = 0.;
        let mut frac: f32 = 0.;
        let mut origin_x: f32;
        let mut origin_y: f32;
        let mut origin_z: f32;
        let gpu_memory_frac: f32;
        let mut gpu_memory: f32 = 0.;
        // Uninitialised in the source unless `-PixelForDefocus` is entered.
        let mut pix_for_defocus: f32 = 0.;
        let mut focus_invert: f32;
        let mut freq: f32;
        let mut arg: f32;
        let dx_line: f32;
        let mut xproj_min: f32;
        let mut xproj_max: f32;
        let mut xproj: f32 = 0.;
        let border: f32;
        let mut expand_factor: f32;
        let expanded_binning: f32;
        let floor_fac: f32;
        let mut reference_sd: f32 = 0.;
        let mut sub_frac: f32;
        let mut sd_tmp: f32 = 0.;
        let mut sum_dbl: f64 = 0.;
        let mut sum_sq_dbl: f64 = 0.;
        let mut n_views_reproj: i32;
        let iwide_reproj: i32;
        let mut k: i32 = 0;
        let mut ind1: i32 = 0;
        let mut ind2: i32 = 0;
        let mut if_exp_weight: i32;
        let mut width_for_bp: i32;
        let mut thick_for_bp: i32;
        let mut min_memory: i32;
        let mut ind_gpu: i32;
        let max_super_fac: i32;
        let mut ss_border: i32;
        let mut ind_delta: i32 = 0;
        let mut if_exit: i32;
        let mut if_mult_by_gaussian: i32;
        let if_hamming_like: i32;
        let mut if_3d_texture: i32;
        let proj_model: bool;
        let real_chunk_run: bool;
        let mut use_exp: bool;
        let mut wall_start: f64;
        let pi: f32 = 3.141593;
        let mut local_fp: Option<ImodFile> = None;
        //
        let mut num_opt_arg: i32 = 0;
        let mut num_non_opt_arg: i32 = 0;
        //
        // fallbacks from ../../manpages/autodoc2man 2 2  tilt
        //
        let num_options: i32 = 87;
        let options: [&[u8]; 87] = [
            b"input:InputProjections:FN:",
            b"output:OutputFile:FN:",
            b":TILTFILE:FN:",
            b":XTILTFILE:FN:",
            b":ZFACTORFILE:FN:",
            b":LOCALFILE:FN:",
            b":BoundaryInfoFile:FN:",
            b":WeightAngleFile:FN:",
            b":WeightFile:FN:",
            b":WIDTH:I:",
            b":SLICE:IA:",
            b":TOTALSLICES:IP:",
            b":THICKNESS:I:",
            b":OFFSET:FA:",
            b":SHIFT:FA:",
            b":ANGLES:FAM:",
            b":XSubsetLoadRatio:F:",
            b":XAXISTILT:F:",
            b":COMPFRACTION:F:",
            b":COMPRESS:FAM:",
            b":FULLIMAGE:IP:",
            b":SUBSETSTART:IP:",
            b":IMAGEBINNED:I:",
            b":ExpandedByFactor:F:",
            b":ReferenceSDofScaling:F:",
            b":LOCALSCALE:F:",
            b":LOG:F:",
            b":RADIAL:FP:",
            b":FalloffIsTrueSigma:B:",
            b":MultiplyByGaussian:B:",
            b":HammingLikeFilter:F:",
            b":DENSWEIGHT:FA:",
            b":ExactFilterSize:I:",
            b":FakeSIRTiterations:I:",
            b":FiltersInFile:FN:",
            b"ray:UseRayIntersections:B:",
            b":INCLUDE:LIM:",
            b":EXCLUDELIST2:LIM:",
            b":COSINTERP:IA:",
            b":XTILTINTERP:I:",
            b":SuperSampleFactor:I:",
            b":ExpandInputLines:B:",
            b":UseUnalignedImages:B:",
            b":UseGPU:I:",
            b":ActionIfGPUFails:IP:",
            b":MODE:I:",
            b":SCALE:FP:",
            b":MASK:I:",
            b":PERPENDICULAR:B:",
            b":PARALLEL:B:",
            b":RotateBy90:B:",
            b":AdjustOrigin:B:",
            b":TITLE:CH:",
            b":BaseRecFile:FN:",
            b":BaseNumViews:I:",
            b":SubtractFromBase:LI:",
            b":MinMaxMean:IT:",
            b":REPROJECT:FAM:",
            b":ViewsToReproject:LI:",
            b"recfile:RecFileToReproject:FN:",
            b"xminmax:XMinAndMaxReproj:IP:",
            b"yminmax:YMinAndMaxReproj:IP:",
            b"zminmax:ZMinAndMaxReproj:IP:",
            b"threshold:ThresholdedReproj:FT:",
            b":FlatFilterFraction:F:",
            b":SIRTIterations:I:",
            b":SIRTSubtraction:B:",
            b":StartingIteration:I:",
            b":VertBoundaryFile:FN:",
            b":VertSliceOutputFile:FN:",
            b":VertForSIRTInput:B:",
            b":ConstrainSign:I:",
            b":ProjectModel:FN:",
            b":SkipTurnedOffPoints:B:",
            b":AngleOutputFile:FN:",
            b":AlignTransformFile:FN:",
            b":DefocusFile:FN:",
            b":PixelForDefocus:FP:",
            b":DONE:B:",
            b":FBPINTERP:I:",
            b":REPLICATE:FPM:",
            b"debug:DebugOutput:B:",
            b"internal:InternalSIRTSlices:IP:",
            b"texture:TextureType:I:",
            b"border:BorderForSuperSample:F:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
        ];
        //
        self.m_rec_reproj = false;
        n_views_reproj = 0;
        self.m_use_gpu = false;
        ind_gpu = -1;
        self.m_do_raw_filter_on_gpu = false;
        self.m_num_gpu_planes = 0;
        self.m_num_sirt_iter = 0;
        self.m_sirt_from_zero = false;
        self.m_read_base_rec = false;
        self.m_rec_subtraction = false;
        self.m_proj_subtraction = 0;
        self.m_vert_sirt_input = 0;
        self.m_save_vert_slices = false;
        angle_output = None;
        transform_file = None;
        self.m_filter_file = None;
        self.m_flat_frac = 0.;
        self.m_iter_for_report = 0;
        self.m_thresh_polarity = 0.;
        focus_invert = 0.;
        defocus_file = None;
        self.m_need_for_filt_arr = 0;
        self.m_need_for_out_arr = 0;
        self.m_need_for_vert_arr = 0;
        self.m_need_for_read_in_arr = 0;
        self.m_need_for_work_arr = 0;
        self.m_need_for_super_arr = 0;
        self.m_use_raw_stack = 0;
        self.m_min_raw_pad_aspect = 0.3; // Low frequency artifacts were seen with 0.26
        max_super_fac = 8;
        self.m_load_xoffset = -1;
        subset_load_ratio = 0.;
        self.m_adjust_out_add_fac = 0.;
        //
        // Minimum array size to allocate, desired number of slices to allocate
        // for if it exceeds that minimum size (increased both by 4, 12/23/20, x2 4/30/24)
        min_memory = 160000000;
        num_need_eval = 80;
        gpu_memory_frac = 0.8;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        pip_set_special_flags(1, 1, 2, 2, 0);
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            b"tilt",
            0,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
            None,
        );

        // Handle help here, this program still allows no arguments
        ind = 0;
        pip_get_boolean(b"usage", &mut ind);
        if ind != 0 {
            pip_print_help(b"tilt", 0, 0, 0);
            c_exit(0);
        }

        if pip_get_in_out_file(b"InputProjections", 0, &mut input_file) != 0 {
            exit_error(b"No input file with projections specified");
        }
        if pip_get_in_out_file(b"OutputFile", 1, &mut output_file) != 0 {
            exit_error(b"No output file specified");
        }
        //
        // Allocate array a little bit for temp use
        self.m_lim_reproj = 1000000;
        temp_arr = vec![0.; self.m_lim_reproj as usize];
        ang_reproj = vec![0.; self.m_lim_reproj as usize];
        pip_get_boolean(b"DebugOutput", &mut self.m_debug);
        //
        // Open input projection file
        unsafe {
            iiu_open(1, &String::from_utf8_lossy(&input_file), "RO");
        }
        iiu_print_header(1, Some("\nInput projection file:"));
        unsafe {
            iiu_ret_basic_head(
                1,
                nproj_xyz.as_mut_ptr(),
                mpxyz.as_mut_ptr(),
                &mut mode,
                &mut self.m_dmin_in,
                &mut self.m_dmax_in,
                &mut self.m_dmean_in,
            );
        }
        (id_type, _lens, nd1, _nd2, vd1, vd2) = iiu_ret_data_type(1);
        self.m_nx_proj = nproj_xyz[0];
        self.m_ny_proj = nproj_xyz[1];
        self.m_num_views = nproj_xyz[2];
        self.m_lim_view = self.m_num_views + 10;
        let lv = self.m_lim_view as usize;
        self.m_sin_beta = vec![0.; lv];
        self.m_cos_beta = vec![0.; lv];
        self.m_sin_alpha = vec![0.; lv];
        self.m_cos_alpha = vec![0.; lv];
        self.m_alpha = vec![0.; lv];
        self.m_angles = vec![0.; lv];
        self.m_xzfac = vec![0.; lv];
        self.m_yzfac = vec![0.; lv];
        self.m_compress = vec![0.; lv];
        self.m_nx_stretched = vec![0; lv];
        self.m_ind_stretch_line = vec![0; lv];
        self.m_stretch_offset = vec![0.; lv];
        self.m_expose_weight = vec![0.; lv];
        self.m_map_used_view = vec![0; lv];
        self.m_wgt_angles = vec![0.; lv];
        iv_exclude = vec![0; lv];
        //
        // The approximate implicit scaling caused by the default radial filter
        self.m_filter_scale = (self.m_nx_proj as f64 / 2.2) as f32;
        //
        new_angles = 0;
        if_tilt_file = 0;
        //
        // Get model file to project and other optional files
        proj_model = pip_get_string(b"ProjectModel", &mut rec_file) == 0;
        {
            let mut tf = Vec::new();
            if pip_get_string(b"AlignTransformFile", &mut tf) == 0 {
                transform_file = Some(tf);
            }
        }
        if proj_model {
            imod_to_project = imod_read(String::from_utf8_lossy(&rec_file).as_ref()).ok();
            if imod_to_project.is_none() {
                exit_error(b"Reading model file to reproject");
            }
            let mut s = Vec::new();
            if pip_get_string(b"AngleOutputFile", &mut s) == 0 {
                angle_output = Some(s);
            }
            let mut s = Vec::new();
            if pip_get_string(b"DefocusFile", &mut s) == 0 {
                defocus_file = Some(s);
            }
            pip_get_two_floats(b"PixelForDefocus", &mut pix_for_defocus, &mut focus_invert);
            pip_get_boolean(b"SkipTurnedOffPoints", &mut self.m_skip_unseen_points);
        }
        //
        // Get entries for reprojection from rec file
        pip_get_integer(b"SIRTIterations", &mut self.m_num_sirt_iter);
        pip_get_three_floats(
            b"ThresholdedReproj",
            &mut self.m_thresh_for_reproj,
            &mut self.m_thresh_polarity,
            &mut self.m_thresh_sum_fac,
        );
        if pip_get_string(b"RecFileToReproject", &mut rec_file) == 0 {
            if proj_model {
                exit_error(b"You cannot use -RecFileToReproject with -ProjectModel");
            }
            self.m_proj_mean = self.m_dmean_in;
            unsafe {
                iiu_open(3, &String::from_utf8_lossy(&rec_file), "RO");
            }
            iiu_print_header(3, Some("\nFile to reproject:"));
            unsafe {
                iiu_ret_basic_head(
                    3,
                    nrec_xyz.as_mut_ptr(),
                    mpxyz.as_mut_ptr(),
                    &mut mode,
                    &mut self.m_dmin_in,
                    &mut self.m_dmax_in,
                    &mut self.m_dmean_in,
                );
            }
            if self.m_num_sirt_iter <= 0 {
                self.m_rec_reproj = true;
                self.m_min_xreproj = 0;
                self.m_min_yreproj = 0;
                self.m_min_zreproj = 0;
                self.m_max_xreproj = nrec_xyz[0] - 1;
                self.m_max_yreproj = nrec_xyz[1] - 1;
                self.m_max_zreproj = nrec_xyz[2] - 1;
                pip_get_two_integers(
                    b"XMinAndMaxReproj",
                    &mut self.m_min_xreproj,
                    &mut self.m_max_xreproj,
                );
                pip_get_two_integers(
                    b"YMinAndMaxReproj",
                    &mut self.m_min_yreproj,
                    &mut self.m_max_yreproj,
                );
                pip_get_two_integers(
                    b"ZMinAndMaxReproj",
                    &mut self.m_min_zreproj,
                    &mut self.m_max_zreproj,
                );
                if self.m_min_xreproj < 0
                    || self.m_min_yreproj < 0
                    || self.m_max_xreproj >= nrec_xyz[0]
                    || self.m_max_yreproj >= nrec_xyz[1]
                {
                    exit_error(b"Min or Max X, or Y coordinate to project is out of range");
                }
                if pip_get_string(b"ViewsToReproj", &mut card) == 0 {
                    iv_reproj =
                        self.parse_card(&card, &mut n_views_reproj, "list of views to reproject");
                }
                pip_get_boolean(b"SIRTSubtraction", &mut self.m_proj_subtraction);
            }
            self.m_min_xreproj = self.m_min_xreproj + 1;
            self.m_max_xreproj = self.m_max_xreproj + 1;
            self.m_min_yreproj = self.m_min_yreproj + 1;
            self.m_max_yreproj = self.m_max_yreproj + 1;
            self.m_min_zreproj = self.m_min_zreproj + 1;
            self.m_max_zreproj = self.m_max_zreproj + 1;
            //
            // If not reading from a rec and doing sirt, then must be doing from 0
        } else if self.m_num_sirt_iter > 0 {
            self.m_sirt_from_zero = true;
            self.m_flat_frac = 1.;
        }
        if self.m_thresh_polarity != 0. && !self.m_rec_reproj {
            exit_error(
                b"Thresholded reprojection can be used only with the -RecFileToReproject option",
            );
        }
        if self.m_thresh_polarity != 0.
            && (self.m_num_sirt_iter > 0 || self.m_proj_subtraction != 0)
        {
            exit_error(
                b"Thresholded reprojection cannot be used with -SIRTSubtraction or -SIRTIterations",
            );
        }

        if !self.m_rec_reproj
            && !proj_model
            && self.m_num_sirt_iter <= 0
            && pip_get_string(b"BaseRecFile", &mut base_file) == 0
        {
            self.m_read_base_rec = true;
            if pip_get_string(b"SubtractFromBase", &mut card) == 0 {
                let mut nsub = 0;
                self.m_iview_subtract =
                    self.parse_card(&card, &mut nsub, "list of views to subtract");
                self.m_num_view_subtract = nsub;
                if self.m_num_view_subtract == 1 && self.m_iview_subtract[0] < 0 {
                    self.m_rec_subtraction = true;
                    self.m_num_view_subtract = 0;
                }
            }
            if !self.m_rec_subtraction
                && pip_get_integer(b"BaseNumViews", &mut self.m_num_view_base) != 0
            {
                exit_error(b"You must enter -BaseNumViews with -BaseRecFile");
            }
            unsafe {
                iiu_open(3, &String::from_utf8_lossy(&base_file), "RO");
                iiu_ret_basic_head(
                    3,
                    nrec_xyz.as_mut_ptr(),
                    mpxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin_tmp,
                    &mut dmax_tmp,
                    &mut dmean_tmp,
                );
            }
        }
        //
        //-------------------------------------------------------------
        // Set up defaults:
        //
        //...... Default is no maskEdges and extra pixels to maskEdges is 0
        self.m_mask_edges = 0;
        self.m_num_extra_mask_pix = 0;
        //...... Default is no scaling of output map
        self.m_out_add = 0.;
        self.m_out_scale = 1.;
        //...... Default is no offset or rotation
        del_angle = 0.;
        //...... Default is output mode 2
        self.m_new_mode = 2;
        //...... Start with no list of views to use or exclude
        num_view_use = 0;
        num_view_exclude = 0;
        //...... Default is no logarithms
        self.m_if_log = 0;
        if_mult_by_gaussian = 0;
        //...... Default overall and individual compression of 1; no alpha tilt
        num_compress = 0;
        comp_factor = 1.;
        for nv in 0..self.m_num_views as usize {
            self.m_compress[nv] = 1.;
            self.m_alpha[nv] = 0.;
            self.m_xzfac[nv] = 0.;
            self.m_yzfac[nv] = 0.;
            self.m_expose_weight[nv] = 1.;
        }
        self.m_if_alpha = 0;
        global_alpha = 0.;
        if_zfactors = 0;
        //
        //...... Default weighting by density of adjacent views
        self.m_num_tilt_inc_wgt = 2;
        for i in 1..=self.m_num_tilt_inc_wgt {
            self.m_tilt_inc_wgts[(i - 1) as usize] = (1. / (i as f64 - 0.5)) as f32;
        }
        self.m_num_wgt_angles = 0;
        //
        x_offset = 0.;
        self.m_y_offset = 0.;
        self.m_axis_xoffset = 0.;
        self.m_nx_warp = 0;
        self.m_ny_warp = 0;
        scale_local = 0.;
        nx_full_in = 0;
        ny_full = 0;
        ix_subset_in = 0;
        iy_subset = 0;
        self.m_ithick_bp = 10;
        image_binned = 1;
        expand_factor = 1.;
        if_thick_in = 0;
        if_width_in = 0;
        if_slice_in = 0;
        if_subset_in = 0;
        if_exp_weight = 0;
        self.m_min_tot_slice = -1;
        self.m_max_tot_slice = -1;
        self.m_exact_samples = 100.;
        self.m_num_exact_cycles = 0;
        self.m_num_fake_sirt_iter = 0;
        self.m_use_intersections = 0;
        //
        //...... Default double - width linear interpolation in cosine stretching
        self.m_interp_fac_stretch = 2;
        self.m_interp_ord_stretch = 1;
        self.m_interp_ord_xtilt = 1;
        self.m_super_sample_fac = 1;
        self.m_super_samp_border = 40;
        self.m_clean_super_fft = 1;
        self.m_proj_super_fac = 1;
        self.m_perpendicular = 1;
        self.m_rotate_by90 = 0;
        self.m_reproj_bp = 0;
        self.m_num_reproj = 0;
        adjust_origin = 0;
        self.m_num_view_subtract = 0;
        self.m_iact_gpu_fail_option = 0;
        self.m_iact_gpu_fail_environ = 0;
        self.m_if_out_sirt_proj = 0;
        self.m_if_out_sirt_rec = 0;
        self.m_isign_constraint = 0;
        vert_out_file = None;
        vert_bound_file = None;
        //
        //...... Default title
        mrc_fill_label_string(
            if self.m_rec_reproj {
                &b"TILT: Reprojection from tomogram"[..]
            } else {
                &b"TILT: Tomographic reconstruction"[..]
            },
            &mut self.m_title,
        );
        if pip_get_string(b"TITLE", &mut card) == 0 {
            mrc_fill_label_string(&card, &mut self.m_title);
        }
        //
        if pip_get_integer(b"IMAGEBINNED", &mut image_binned) == 0 {
            image_binned = b3dmax!(1, image_binned);
            if image_binned > 1 {
                printf!(
                    "\n Dimensions and coordinates will be scaled down by a factor of %d\n",
                    ci(image_binned)
                );
            }
        }
        if pip_get_float(b"ExpandedByFactor", &mut expand_factor) == 0 {
            if (expand_factor as f64) < 0.01 {
                exit_error(b"The entry for ExpandedByFactor must be positive");
            }
            if self.m_rec_reproj && expand_factor != 1. {
                exit_error(b"Reprojection cannot be done with an ExpandedByFactor entry");
            }
            if expand_factor != 1. {
                printf!(
                    "\n Dimensions and coordinates will be adjusted by the expansion factor %g\n",
                    cf(expand_factor as f64)
                );
            }
        }
        expanded_binning = image_binned as f32 / expand_factor;

        // Use raw images
        pip_get_boolean(b"UseUnalignedImages", &mut self.m_use_raw_stack);
        if self.m_use_raw_stack != 0 {
            let Some(transform_name) = transform_file.clone() else {
                exit_error(b"AlignTransformFile must be entered to use unaligned images");
            };
            if expand_factor != 1. {
                exit_error(b"Unaligned images cannot be used if ExpandedByFactor is entered");
            }
            let mut fp_xf =
                match ImodFile::open(String::from_utf8_lossy(&transform_name).as_ref(), "r") {
                    Some(f) => f,
                    None => exit_error_fmt!(
                        "Opening file of raw stack transforms: %s",
                        CArg::Bytes(&transform_name)
                    ),
                };

            // Get XFs into an array, swap middle terms (see below) and divide by image
            // binning so they are already scaled to actual input data.
            raw_xfs = vec![0.; 6 * lv];
            nfields = 6 * self.m_num_views;
            self.get_values_from_lines(
                Some(&mut fp_xf),
                None,
                &mut raw_xfs[0..],
                None,
                false,
                &mut nfields,
                "raw stack transforms",
            );
            for kti in 0..self.m_num_views as usize {
                raw_xfs.swap(6 * kti + 1, 6 * kti + 2);
                raw_xfs[6 * kti + 4] /= expanded_binning;
                raw_xfs[6 * kti + 5] /= expanded_binning;
            }

            // Need to analyze for large rotations, use the middle transform
            let kti = (6 * (self.m_num_views / 2)) as usize;
            (theta, smag, stretch, str_phi) = amat_to_rotmagstr(
                raw_xfs[kti],
                raw_xfs[kti + 2],
                raw_xfs[kti + 1],
                raw_xfs[kti + 3],
            );

            // Get angle between -45 and 315 to take nearest integer of division by 90.
            if theta as f64 <= -45. {
                theta = (theta as f64 + 360.) as f32;
            }
            self.m_rot_flip_operation = b3dnint!(theta as f64 / 90.);
            self.m_rot_flip_operation = b3dmax!(0, b3dmin!(3, self.m_rot_flip_operation));
            self.m_raw_remaining_rot =
                (theta as f64 - 90. * self.m_rot_flip_operation as f64) as f32;

            // Compose the inverse of this operation to take it out of the transforms
            if self.m_rot_flip_operation != 0 {
                temp_arr[0] =
                    (-90. * self.m_rot_flip_operation as f64 * deg_to_rad as f64).cos() as f32;
                temp_arr[2] =
                    (90. * self.m_rot_flip_operation as f64 * deg_to_rad as f64).sin() as f32;
                temp_arr[1] = -temp_arr[2];
                temp_arr[3] = temp_arr[0];
            }

            // Invert the transforms, taking out the rotflip operation
            for kti in 0..self.m_num_views as usize {
                if self.m_rot_flip_operation != 0 {
                    let second: [f32; 6] = raw_xfs[6 * kti..6 * kti + 6].try_into().unwrap();
                    xf_mult(&temp_arr[..6], &second, &mut raw_xfs[6 * kti..], 2);
                }
                let matrix: [f32; 6] = raw_xfs[6 * kti..6 * kti + 6].try_into().unwrap();
                xf_invert(&matrix, &mut raw_xfs[6 * kti..], 2);
            }

            // Swap nx and ny for 90/270 rotations!
            // Cancel cosine stretching
            // Set up to evaluate loading close to what is needed for minimum aspect ratio
            if self.m_rot_flip_operation % 2 != 0 {
                std::mem::swap(&mut self.m_nx_proj, &mut self.m_ny_proj);
            }
            self.m_interp_fac_stretch = 0;
            self.m_interp_ord_stretch = 0;
            num_need_eval = b3dmax!(
                num_need_eval as f32,
                self.m_min_raw_pad_aspect * self.m_nx_proj as f32
            ) as i32;
        }

        // finish defaults with nx/ny finalized
        //...... Default slice is all rows in a projection plane, and all columns
        self.m_islice_start = 1;
        self.m_islice_end = self.m_ny_proj;
        self.m_iwidth = self.m_nx_proj;
        self.m_nx_full_proj = self.m_nx_proj;
        max_needs = vec![0; num_need_eval as usize];
        //
        nfields = 0;
        if pip_get_integer_array(b"SLICE", &mut inum, &mut nfields, LIMNUM as i32) == 0 {
            if nfields / 2 != 1 {
                exit_error(b"Wrong number of fields on SLICE line");
            }
            self.m_islice_start = inum[0] + 1;
            self.m_islice_end = inum[1] + 1;
            if nfields > 2 && inum[2] != 1 {
                exit_error(b"A slice increment other than 1 is no longer supported");
            }
            if_slice_in = 1;
        }
        //
        if pip_get_integer(b"THICKNESS", &mut self.m_ithick_bp) == 0 {
            if_thick_in = 1;
        }
        //
        self.m_mask_edges = (pip_get_integer(b"MASK", &mut self.m_num_extra_mask_pix) == 0) as i32;
        pip_get_float(b"XSubsetLoadRatio", &mut subset_load_ratio);

        if_3d_texture = -999;
        pip_get_integer(b"TextureType", &mut if_3d_texture);
        //
        //...... Default radial weighting parameters - no filtering
        irad_max = self.m_nx_proj / 2 + 1;
        rad_fall = 0.;
        //
        // Hamming and RADIAL entry are mutually exclusive
        if_hamming_like = 1 - pip_get_float(b"HammingLikeFilter", &mut rad_max);
        if_radial = 1 - pip_get_two_floats(b"RADIAL", &mut rad_max, &mut rad_fall);
        if if_hamming_like > 0 && if_radial > 0 {
            exit_error(b"You cannot enter HammingLikeFilter with RADIAL");
        }
        if_true_sigma = 0;
        pip_get_boolean(b"FalloffIsTrueSigma", &mut if_true_sigma);
        //
        if pip_get_boolean(b"MultiplyByGaussian", &mut if_mult_by_gaussian) == 0 {
            if if_hamming_like > 0 && if_mult_by_gaussian == 0 {
                exit_error(b"You cannot enter HammingLikeFilter and MultiplyByGaussian 0");
            }
            if if_hamming_like == 0 {
                printf!("\n Multiplying radial function by Gaussian\n");
            }
        }
        //
        nfields = 0;
        if pip_get_float_array(b"OFFSET", &mut xnum, &mut nfields, LIMNUM as i32) == 0 {
            if nfields == 0 || nfields >= 3 {
                exit_error(b"Wrong number of fields on OFFSET line");
            }
            if nfields == 2 {
                self.m_axis_xoffset = xnum[1];
            }
            del_angle = xnum[0];
        }
        //
        if pip_get_two_floats(b"SCALE", &mut self.m_out_add, &mut self.m_out_scale) == 0 {
            printf!(
                "\n Output map densities incremented by %.2f and then multiplied by %.2f\n",
                cf(self.m_out_add as f64),
                cf(self.m_out_scale as f64)
            );
        }
        //
        pip_get_boolean(b"UseRayIntersections", &mut self.m_use_intersections);
        iv = pip_get_boolean(b"PERPENDICULAR", &mut self.m_perpendicular);
        pip_get_boolean(b"RotateBy90", &mut self.m_rotate_by90);
        i = 0;
        pip_get_boolean(b"PARALLEL", &mut i);
        if (iv == 0 && self.m_perpendicular != 0 && (i > 0 || self.m_rotate_by90 != 0))
            || (i > 0 && self.m_rotate_by90 != 0)
        {
            exit_error(b"You can select only one of PERPENDICULAR, PARALLEL, and RotateBy90");
        }
        if i > 0 || self.m_rotate_by90 != 0 {
            self.m_perpendicular = 0;
        }
        if self.m_perpendicular != 0 {
            printf!("\n Output map is sectioned perpendicular to the tilt axis\n");
        } else {
            printf!(
                " Output map is sectioned parallel to the zero tilt projection, %s handedness\n",
                CArg::Str(if self.m_rotate_by90 != 0 {
                    "retaining"
                } else {
                    "inverting"
                })
            );
            min_memory += min_memory / 2;
        }
        //
        if pip_get_integer(b"MODE", &mut self.m_new_mode) == 0 {
            if self.m_new_mode < 0
                || self.m_new_mode > 12
                || (self.m_new_mode > 2 && self.m_new_mode < 12)
            {
                exit_error(b"Illegal output mode");
            }
            set_float_16_output_mode(if self.m_new_mode == 12 { 1 } else { 0 }, 1);
            printf!("\n Data mode of output file is %d\n", ci(self.m_new_mode));
            if self.m_new_mode == 12 {
                self.m_new_mode = 2;
            }
        }
        //
        // Set up map to views if there are INCLUDE entries
        pip_number_of_entries(b"INCLUDE", &mut num_ent);
        for _j in 1..=num_ent {
            pip_get_string(b"INCLUDE", &mut card);
            num_exclude_list = 0;
            list_temp = self.parse_card(&card, &mut num_exclude_list, "list of included views");
            if num_view_use + num_exclude_list > self.m_lim_view {
                exit_error(b"More included views then views in the input file");
            }
            for i in 0..num_exclude_list as usize {
                if list_temp[i] < 1 || list_temp[i] > self.m_num_views {
                    exit_error(b"Illegal view number in INCLUDE list");
                }
                self.m_map_used_view[num_view_use as usize] = list_temp[i];
                num_view_use += 1;
            }
        }
        //
        // Get EXCLUDE entries into a list
        pip_number_of_entries(b"EXCLUDELIST2", &mut num_ent);
        if num_view_use > 0 && num_ent > 0 {
            exit_error(b"Illegal to have both INCLUDE and EXCLUDE entries");
        }

        for _j in 1..=num_ent {
            pip_get_string(b"EXCLUDELIST2", &mut card);
            num_exclude_list = 0;
            list_temp = self.parse_card(&card, &mut num_exclude_list, "list of excluded views");
            if num_view_exclude + num_exclude_list > self.m_lim_view {
                exit_error(b"More excluded views then views in the input file");
            }
            for i in 0..num_exclude_list as usize {
                if list_temp[i] < 1 || list_temp[i] > self.m_num_views {
                    exit_error(b"Illegal view number in EXCLUDE list");
                }
                iv_exclude[num_view_exclude as usize] = list_temp[i];
                num_view_exclude += 1;
            }
        }
        //
        if pip_get_float(b"LOG", &mut self.m_base_for_log) == 0 {
            self.m_if_log = 1;
            printf!(
                "\n Taking logarithm of input data plus %.3f\n",
                cf(self.m_base_for_log as f64)
            );
        }
        //
        pip_number_of_entries(b"ANGLES", &mut num_ent);
        for _j in 1..=num_ent {
            nfields = 0;
            pip_get_float_array(
                b"ANGLES",
                &mut self.m_angles[new_angles as usize..],
                &mut nfields,
                self.m_lim_view - new_angles,
            );
            new_angles = new_angles + nfields;
        }
        //
        pip_number_of_entries(b"COMPRESS", &mut num_ent);
        for _j in 1..=num_ent {
            nfields = 0;
            pip_get_float_array(
                b"COMPRESS",
                &mut self.m_compress[num_compress as usize..],
                &mut nfields,
                self.m_lim_view - num_compress,
            );
            num_compress = num_compress + nfields;
        }
        //
        if pip_get_float(b"COMPFRACTION", &mut comp_factor) == 0 {
            printf!(
                "\n Compression was confined to %.3f of the distance over which it was measured\n",
                cf(comp_factor as f64)
            );
        }
        //
        nfields = 0;
        if pip_get_float_array(b"DENSWEIGHT", &mut xnum, &mut nfields, LIMNUM as i32) == 0 {
            self.m_num_tilt_inc_wgt = b3dnint!(xnum[0]);
            if self.m_num_tilt_inc_wgt > 0 {
                for i in 1..=self.m_num_tilt_inc_wgt {
                    self.m_tilt_inc_wgts[(i - 1) as usize] = (1. / (i as f64 - 0.5)) as f32;
                }
                if nfields == self.m_num_tilt_inc_wgt + 1 {
                    for i in 1..=self.m_num_tilt_inc_wgt as usize {
                        self.m_tilt_inc_wgts[i - 1] = xnum[i];
                    }
                } else if nfields != 1 {
                    exit_error(b"Wrong number of fields on DENSWEIGHT line");
                }
                printf!(
                    "\n Weighting by tilt density computed to distance of %d views\n  weighting factors:",
                    ci(self.m_num_tilt_inc_wgt)
                );
                for i in 1..=self.m_num_tilt_inc_wgt {
                    printf!("%6.3f", cf(self.m_tilt_inc_wgts[(i - 1) as usize] as f64));
                    if i % 10 == 0 || i == self.m_num_tilt_inc_wgt {
                        printf!("\n");
                    }
                }
            } else {
                printf!("\n No weighting by tilt density\n");
            }
        }
        //
        // Keep this with the above entry so that nfields indicates if it was entered
        if pip_get_integer(b"ExactFilterSize", &mut num_ent) == 0 {
            if nfields > 0 {
                exit_error(b"You cannot enter DENSWEIGHT with ExactFilterSize");
            }
            self.m_exact_obj_size = num_ent as f32;
            self.m_exact_table = vec![0.; (b3dnint!(self.m_exact_samples) + 1) as usize];
            self.m_num_exact_cycles = 1;

            // This array was indexed from 0 in Fortran
            for i in 0..=b3dnint!(self.m_exact_samples) {
                self.m_exact_table[i as usize] =
                    (1. - (i as f32 / self.m_exact_samples) as f64) as f32;
            }
        }
        //
        if pip_get_integer(b"FakeSIRTiterations", &mut self.m_num_fake_sirt_iter) == 0
            && self.m_num_exact_cycles > 0
        {
            exit_error(b"You cannot enter FakeSIRTiterations with ExactFilterSize");
        }

        {
            let mut ff = Vec::new();
            if pip_get_string(b"FiltersInFile", &mut ff) == 0 {
                self.m_filter_file = Some(ff.clone());
                if self.m_num_fake_sirt_iter > 0
                    || self.m_num_exact_cycles > 0
                    || if_hamming_like > 0
                    || if_mult_by_gaussian > 0
                {
                    exit_error(
                        b"You cannot enter FiltersInFile with FakeSIRTiterations, ExactFilterSize, HammingLikeFilter, or MultiplyByGaussian",
                    );
                }
                if self.m_num_sirt_iter > 0 || self.m_rec_reproj {
                    exit_error(
                        b"You cannot enter FiltersInFile when doing SIRT iterations or reprojecting from a reconstruction file",
                    );
                }
                if self.m_use_raw_stack != 0 {
                    exit_error(
                        b"You cannot enter FiltersInFile when reconstructing from the raw stack",
                    );
                }
                printf!(
                    "\n Radial filters are being read from file: %s",
                    CArg::Bytes(&ff)
                );
            }
        }
        //
        if pip_get_string(b"TILTFILE", &mut card) == 0 {
            let mut nv = self.m_num_views;
            let mut angles = std::mem::take(&mut self.m_angles);
            self.get_values_from_lines(
                None,
                Some(&card),
                &mut angles,
                None,
                false,
                &mut nv,
                "tilt angles",
            );
            self.m_angles = angles;
            if_tilt_file = 1;
        }
        //
        // Get size and location of reconstructed region
        if pip_get_integer(b"WIDTH", &mut self.m_iwidth) == 0 {
            if_width_in = 1;
        }
        //
        nfields = 0;
        if pip_get_float_array(b"SHIFT", &mut xnum, &mut nfields, LIMNUM as i32) == 0 {
            if (nfields + 1) / 2 != 1 {
                exit_error(b"Wrong number of fields on SHIFT line");
            }
            x_offset = xnum[0];
            if nfields == 2 {
                self.m_y_offset = xnum[1];
            }
        }
        //
        if pip_get_string(b"XTILTFILE", &mut card) == 0 {
            let mut nv = self.m_num_views;
            let mut alpha = std::mem::take(&mut self.m_alpha);
            self.get_values_from_lines(
                None,
                Some(&card),
                &mut alpha,
                None,
                false,
                &mut nv,
                "X-axis tilt angles",
            );
            self.m_alpha = alpha;
            self.m_if_alpha = 1;
            for i in 0..self.m_num_views as usize {
                if (b3dabs!(self.m_alpha[i] - self.m_alpha[0]) as f64) > 1.0e-5 {
                    self.m_if_alpha = 2;
                }
            }
            if self.m_if_alpha == 2 {
                printf!("\n Alpha tilting to be applied with angles from file\n");
            }
            if self.m_if_alpha == 1 {
                printf!(
                    "\n Constant alpha tilt of %.1f  to be applied based on angles from file\n",
                    cf(-self.m_alpha[0] as f64)
                );
            }
        }
        //
        if pip_get_string(b"WeightFile", &mut card) == 0 {
            let mut nv = self.m_num_views;
            let mut weights = std::mem::take(&mut self.m_expose_weight);
            self.get_values_from_lines(
                None,
                Some(&card),
                &mut weights,
                None,
                false,
                &mut nv,
                "weighting factors",
            );
            self.m_expose_weight = weights;
            if_exp_weight = 1;
        }
        //
        bound_file = None;
        {
            let mut bf = Vec::new();
            if pip_get_string(b"BoundaryInfoFile", &mut bf) == 0 {
                bound_file = Some(bf);
            }
        }
        //
        if pip_get_float(b"XAXISTILT", &mut global_alpha) == 0 {
            printf!(
                "\n Global alpha tilt of %.1f will be applied\n",
                cf(global_alpha as f64)
            );
            if (b3dabs!(global_alpha) as f64) > 1.0e-5 && self.m_if_alpha == 0 {
                self.m_if_alpha = 1;
            }
        }
        //
        // REPROJECT entry must be read in before local alignments
        // violates original unless a blank entry is allowed
        pip_number_of_entries(b"REPROJECT", &mut num_ent);
        for j in 1..=num_ent {
            nfields = 0;
            pip_get_float_array(b"REPROJECT", &mut xnum, &mut nfields, LIMNUM as i32);
            if nfields == 0 {
                nfields = 1;
                xnum[0] = 0.;
            }
            if nfields + self.m_num_reproj > self.m_lim_reproj {
                exit_error(b"Too many reprojection angles for arrays");
            }
            for i in 0..nfields as usize {
                ang_reproj[self.m_num_reproj as usize] = xnum[i];
                self.m_num_reproj += 1;
            }
            if j == 1 {
                printf!("\n Output will be one or more reprojections\n");
            }
            self.m_reproj_bp = (!self.m_rec_reproj) as i32;
        }
        //
        if pip_get_string(b"ZFACTORFILE", &mut card) == 0 {
            let mut nv = self.m_num_views;
            let mut xz = std::mem::take(&mut self.m_xzfac);
            let mut yz = std::mem::take(&mut self.m_yzfac);
            self.get_values_from_lines(
                None,
                Some(&card),
                &mut xz,
                Some(&mut yz),
                false,
                &mut nv,
                "Z factors",
            );
            self.m_xzfac = xz;
            self.m_yzfac = yz;
            if_zfactors = 1;
            printf!("\n Z-dependent shifts to be applied with factors from file\n");
        }
        //
        if pip_get_string(b"LOCALFILE", &mut card) == 0 {
            // The source `fopen`s without a test and hands a NULL `fp` to
            // `fgetline`, which reports it and returns -1 (`b3dutil.c:1079-1082`),
            // so the failed first read below is what ends the program.
            let mut fp = match ImodFile::open(String::from_utf8_lossy(&card).as_ref(), "r") {
                Some(fp) => fp,
                None => {
                    crate::imod::libcfshr::b3dutil::b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!("fgetline: file pointer not valid\n"),
                    );
                    exit_error_fmt!(
                        "Reading top line of local alignment file %s",
                        CArg::Bytes(&card)
                    )
                }
            };
            num_input = -1;
            ierr = fgetline(&mut fp, &mut self.m_line, MAX_LINE as i32);
            if ierr < 0 {
                exit_error_fmt!(
                    "Reading top line of local alignment file %s",
                    CArg::Bytes(&card)
                );
            }
            let line_len = self.m_line.iter().position(|&c| c == 0).unwrap_or(0);
            let line = self.m_line[..line_len].to_vec();
            if pip_get_line_of_values(
                &line,
                &line,
                PipValueArray::Float(&mut xnum),
                crate::imod::libcfshr::parse_params::PIP_FLOAT,
                &mut num_input,
                LIMNUM as i32,
            )
            .is_err()
            {
                let mut err = Vec::new();
                pip_get_error(&mut err);
                exit_error_fmt!(
                    "Getting values from top line of local alignment file: %s",
                    CArg::Bytes(&err)
                );
            }
            local_fp = Some(fp);
            self.m_if_del_alpha = 0;
            if num_input > 6 {
                self.m_if_del_alpha = b3dnint!(xnum[6]);
            }
            pixel_local = 0.;
            if num_input > 7 {
                pixel_local = xnum[7];
            }
            local_zfacs = 0;
            if num_input > 8 {
                local_zfacs = xnum[8] as i32;
            }
            self.m_nx_warp = b3dnint!(xnum[0]);
            self.m_ny_warp = b3dnint!(xnum[1]);
            self.m_ix_start_warp = b3dnint!(xnum[2]);
            self.m_iy_start_warp = b3dnint!(xnum[3]);
            self.m_idel_xwarp = b3dnint!(xnum[4]);
            self.m_idel_ywarp = b3dnint!(xnum[5]);
            self.m_num_warp_pos = self.m_nx_warp * self.m_ny_warp;
            self.m_lim_warp = self.m_nx_warp * self.m_ny_warp * self.m_num_views;
            ipos = self.m_lim_warp;
            ind_delta = self.m_num_views;
            //
            // If reprojecting rec, make sure arrays are big enough for all the
            // reprojections and  allocate extra space at top of some arrays for
            // temporary use
            if self.m_rec_reproj {
                ind_delta = b3dmax!(self.m_num_views, self.m_num_reproj);
                self.m_lim_warp = self.m_nx_warp * self.m_ny_warp * ind_delta;
                ipos = self.m_lim_warp + ind_delta;
            }
            if self.m_nx_warp < 2 || self.m_ny_warp < 2 {
                exit_error(b"There must be at least two local alignment areas in X and in Y");
            }
        }

        // Allocate warp arrays
        if self.m_nx_warp > 0 || self.m_use_raw_stack != 0 {
            // Set up for minimal warp allocations if there is none for raw stack
            // without locals
            if self.m_nx_warp == 0 {
                self.m_num_warp_pos = 4;
                self.m_lim_warp = 4 * self.m_num_views;
                ipos = self.m_lim_warp;
                ind_delta = self.m_num_views;
            }

            self.m_ind_warp = vec![0; self.m_num_warp_pos as usize];
            self.m_del_alpha = vec![0.; ipos as usize];
            self.m_cwarp_beta = vec![0.; self.m_lim_warp as usize];
            self.m_swarp_beta = vec![0.; self.m_lim_warp as usize];
            self.m_cwarp_alpha = vec![0.; self.m_lim_warp as usize];
            self.m_swarp_alpha = vec![0.; self.m_lim_warp as usize];
            self.m_fwarp = vec![0.; 6 * ipos as usize];
            self.m_del_beta = vec![0.; ipos as usize];
            self.m_warp_xzfac = vec![0.; ipos as usize];
            self.m_warp_yzfac = vec![0.; ipos as usize];
        }

        // Read local alignments
        if self.m_nx_warp > 0 {
            let fp = local_fp.as_mut().unwrap();
            ind_base = 0;
            for ipos in 1..=self.m_nx_warp * self.m_ny_warp {
                self.m_ind_warp[(ipos - 1) as usize] = ind_base;
                let ib = ind_base as usize;
                let nvu = self.m_num_views as usize;
                let mut nv = self.m_num_views;
                let mut arr = std::mem::take(&mut self.m_del_beta);
                self.get_values_from_lines(
                    Some(fp),
                    None,
                    &mut arr[ib..ib + nvu],
                    None,
                    false,
                    &mut nv,
                    "local tilt alignment data",
                );
                self.m_del_beta = arr;
                if self.m_if_del_alpha > 0 {
                    let mut nv = self.m_num_views;
                    let mut arr = std::mem::take(&mut self.m_del_alpha);
                    self.get_values_from_lines(
                        Some(fp),
                        None,
                        &mut arr[ib..ib + nvu],
                        None,
                        false,
                        &mut nv,
                        "local tilt alignment data",
                    );
                    self.m_del_alpha = arr;
                } else {
                    for i in ib..ib + nvu {
                        self.m_del_alpha[i] = 0.;
                    }
                }
                //
                // Set z factors to zero, read in if supplied, then negate them
                //
                for i in ib..ib + nvu {
                    self.m_warp_xzfac[i] = 0.;
                    self.m_warp_yzfac[i] = 0.;
                }
                if local_zfacs > 0 {
                    let mut nv = self.m_num_views;
                    let mut xz = std::mem::take(&mut self.m_warp_xzfac);
                    let mut yz = std::mem::take(&mut self.m_warp_yzfac);
                    self.get_values_from_lines(
                        Some(fp),
                        None,
                        &mut xz[ib..ib + nvu],
                        Some(&mut yz[ib..ib + nvu]),
                        false,
                        &mut nv,
                        "local tilt alignment data",
                    );
                    self.m_warp_xzfac = xz;
                    self.m_warp_yzfac = yz;
                }
                for i in ib..ib + nvu {
                    self.m_warp_xzfac[i] = -self.m_warp_xzfac[i];
                    self.m_warp_yzfac[i] = -self.m_warp_yzfac[i];
                }
                nfields = 6 * self.m_num_views;
                let mut arr = std::mem::take(&mut self.m_fwarp);
                self.get_values_from_lines(
                    Some(fp),
                    None,
                    &mut arr[6 * ib..6 * ib + 6 * nvu],
                    None,
                    false,
                    &mut nfields,
                    "local tilt alignment data",
                );
                self.m_fwarp = arr;
                // Transforms are stored as a11 a12 a21 a22
                // In fortran they are placed into (1,1) (1,2) (2,1) (2,2) of a 2-D array
                // xf(2,3)
                // So in memory they are supposed to be (1,1) (2,1) (1,2) (2,2)
                // Since we just read in a11 a12 a21 a22 we need to swap a12 and a21
                for kti in 0..nvu {
                    self.m_fwarp.swap(6 * (ib + kti) + 1, 6 * (ib + kti) + 2);
                }
                ind_base = ind_base + ind_delta;
            }
            printf!("\n Local tilt alignment information read from file\n");
        } else if self.m_use_raw_stack != 0 {
            // Set up warping if it is not being done to work with raw stack
            // Increment flag to indicate no native warping
            self.m_use_raw_stack += 1;
            scale_local = 1.;
            for i in 0..self.m_lim_warp as usize {
                self.m_del_beta[i] = 0.;
                self.m_del_alpha[i] = 0.;
            }
            self.m_nx_warp = 2;
            self.m_ny_warp = 2;
            self.m_ix_start_warp = self.m_nx_proj / 4;
            self.m_idel_xwarp = self.m_nx_proj / 2;
            self.m_iy_start_warp = self.m_ny_proj / 4;
            self.m_idel_ywarp = self.m_ny_proj / 2;
            ind_base = 0;
            for ipos in 1..=self.m_nx_warp * self.m_ny_warp {
                self.m_ind_warp[(ipos - 1) as usize] = ind_base;

                // Copy Z factors, taking negative
                for i in 0..self.m_num_views as usize {
                    self.m_warp_xzfac[i + ind_base as usize] = -self.m_xzfac[i];
                    self.m_warp_yzfac[i + ind_base as usize] = -self.m_yzfac[i];
                }

                // Copy the transforms that were already inverted
                for kti in 0..self.m_num_views as usize {
                    let to = 6 * (ind_base as usize + kti);
                    xf_copy(&raw_xfs[6 * kti..], 2, &mut self.m_fwarp[to..], 2);
                }
                ind_base = ind_base + ind_delta;
            }
        }
        //
        if pip_get_float(b"LOCALSCALE", &mut scale_local) == 0 {
            printf!(
                "\n Local alignment positions and shifts reduced by %.4f\n",
                cf(scale_local as f64)
            );
        }
        //
        pip_get_two_integers(b"FULLIMAGE", &mut nx_full_in, &mut ny_full);
        if pip_get_two_integers(b"SUBSETSTART", &mut ix_subset_in, &mut iy_subset) == 0 {
            if_subset_in = 1;
        }
        //
        if !(self.m_rec_reproj || self.m_num_sirt_iter > 0) {
            pip_get_integer(b"SuperSampleFactor", &mut self.m_super_sample_fac);
            if self.m_super_sample_fac < 1 || self.m_super_sample_fac > max_super_fac {
                exit_error_fmt!(
                    "Super-sampling factor must be between 1 and %",
                    ci(max_super_fac)
                );
            }
            if self.m_super_sample_fac > 1 {
                pip_get_integer(b"BorderForSuperSample", &mut self.m_super_samp_border);
                if self.m_super_samp_border < 0 {
                    exit_error(b"Border for super-sampling must be non-negative");
                }
                ierr = 0;
                pip_get_boolean(b"ExpandInputLines", &mut ierr);
                if ierr != 0 {
                    if self.m_use_raw_stack != 0 {
                        exit_error(
                            b"You cannot expand input lines when using raw projection images",
                        );
                    }
                    self.m_proj_super_fac = self.m_super_sample_fac;
                }
                self.m_interp_ord_stretch = 0;
                self.m_interp_fac_stretch = 0;
            }
        } else if pip_get_integer(b"SuperSampleFactor", &mut nfields) == 0 {
            exit_error(
                b"You cannot use supersampling with SIRT iterations or reprojection from a reconstruction",
            );
        }
        //
        nfields = 0;
        if self.m_super_sample_fac < 2
            && self.m_use_raw_stack == 0
            && pip_get_integer_array(b"COSINTERP", &mut inum, &mut nfields, LIMNUM as i32) == 0
        {
            self.m_interp_ord_stretch = inum[0];
            if nfields > 1 {
                self.m_interp_fac_stretch = inum[1];
            }
            self.m_interp_ord_stretch = b3dmax!(0, b3dmin!(3, self.m_interp_ord_stretch));
            if self.m_interp_ord_stretch == 0 {
                self.m_interp_fac_stretch = 0;
            }
            if self.m_interp_fac_stretch == 0 {
                printf!("Cosine stretching is disabled\n");
            } else {
                printf!(
                    "\n Cosine stretching, if any, will have interpolation order %d, sampling factor %d\n",
                    ci(self.m_interp_ord_stretch),
                    ci(self.m_interp_fac_stretch)
                );
            }
        }
        //
        if pip_get_integer(b"XTILTINTERP", &mut self.m_interp_ord_xtilt) == 0 {
            if self.m_interp_ord_xtilt > 2 {
                self.m_interp_ord_xtilt = 3;
            }
            if self.m_interp_ord_xtilt <= 0 {
                printf!("New-style X-tilting with vertical slices is disabled\n");
            } else {
                printf!(
                    "\n X-tilting with vertical slices, if any, will have interpolation order %d\n",
                    ci(self.m_interp_ord_xtilt)
                );
            }
        }
        //
        // Get the GPU option or specification from environment
        ind_gpu = get_standard_gpu_options(
            &mut self.m_if_gpu_by_environ,
            Some(&mut self.m_iact_gpu_fail_option),
            Some(&mut self.m_iact_gpu_fail_environ),
        );
        self.m_use_gpu = ind_gpu >= 0 && self.m_thresh_polarity == 0.;
        //
        let (inum0, inum1) = inum.split_at_mut(1);
        if pip_get_two_integers(b"TOTALSLICES", &mut inum0[0], &mut inum1[0]) == 0 {
            self.m_min_tot_slice = inum[0] + 1;
            self.m_max_tot_slice = inum[1] + 1;
        }
        //
        pip_get_float(b"FlatFilterFraction", &mut self.m_flat_frac);
        self.m_flat_frac = b3dmax!(0., self.m_flat_frac as f64) as f32;
        if self.m_flat_frac > 0. && (self.m_num_fake_sirt_iter > 0 || self.m_filter_file.is_some())
        {
            exit_error(
                b"You cannot enter FlatFilterFraction with FakeSIRTiterations or FiltersFromFile",
            );
        }
        //
        pip_get_boolean(b"AdjustOrigin", &mut adjust_origin);
        //
        if !self.m_rec_reproj {
            pip_get_three_floats(
                b"MinMaxMean",
                &mut self.m_dmin_in,
                &mut self.m_dmax_in,
                &mut self.m_dmean_in,
            );
        }
        //
        if pip_get_string(b"WeightAngleFile", &mut card) == 0 {
            let mut fp = match ImodFile::open(String::from_utf8_lossy(&card).as_ref(), "r") {
                Some(f) => f,
                None => exit_error_fmt!("Opening weight angle file %s", CArg::Bytes(&card)),
            };
            self.m_num_wgt_angles = -self.m_lim_view;
            let mut n = self.m_num_wgt_angles;
            let mut arr = std::mem::take(&mut self.m_wgt_angles);
            self.get_values_from_lines(
                Some(&mut fp),
                None,
                &mut arr,
                None,
                false,
                &mut n,
                "angles for weighting",
            );
            self.m_wgt_angles = arr;
            self.m_num_wgt_angles = n;
            //
            // Sort the angles
            rs_sort_floats(&mut self.m_wgt_angles, self.m_num_wgt_angles);
        }
        //
        // SIRT-related options
        pip_get_integer(b"ConstrainSign", &mut self.m_isign_constraint);
        //
        pip_get_two_integers(
            b"InternalSIRTSlices",
            &mut self.m_if_out_sirt_proj,
            &mut self.m_if_out_sirt_rec,
        );
        if self.m_num_sirt_iter > 0 || self.m_rec_subtraction {
            pip_get_integer(b"StartingIteration", &mut self.m_iter_for_report);
        }
        pip_get_boolean(b"VertForSIRTInput", &mut self.m_vert_sirt_input);
        {
            let mut s = Vec::new();
            if pip_get_string(b"VertSliceOutputFile", &mut s) == 0 {
                vert_out_file = Some(s);
            }
            let mut s = Vec::new();
            if pip_get_string(b"VertBoundaryFile", &mut s) == 0 {
                vert_bound_file = Some(s);
            }
        }
        pip_get_float(b"ReferenceSDofScaling", &mut reference_sd);
        //
        pip_done();
        //
        // END OF OPTION READING
        //
        if nx_full_in == 0 && ny_full == 0 && (ix_subset_in != 0 || iy_subset != 0) {
            exit_error(b"You must enter the full image size if you have a subset");
        }
        if self.m_perpendicular == 0 && self.m_min_tot_slice > 0 {
            exit_error(b"Cannot do chunk writing with parallel slices");
        }
        if self.m_num_reproj > 0 && n_views_reproj > 0 {
            exit_error(b"You cannot enter both views and angles to reproject");
        }
        if proj_model && (self.m_num_reproj > 0 || n_views_reproj > 0) {
            exit_error(b"You cannot do projection from a model with image reprojection");
        }
        if self.m_num_sirt_iter > 0 && (self.m_num_reproj > 0 || n_views_reproj > 0) {
            exit_error(b"You cannot do SIRT with entries for angles/views to reproject");
        }
        if self.m_use_raw_stack != 0 && (self.m_num_sirt_iter > 0 || self.m_rec_reproj) {
            exit_error(
                b"You cannot use raw input when reprojecting from a reconstruction or doing SIRT iterations",
            );
        }

        // If there is a reference SD, linear scaling, and binning, get SD of middle
        // projection image and adjust scaling
        if self.m_if_log == 0
            && image_binned > 1
            && reference_sd > 0.
            && self.m_nx_proj > 10
            && self.m_ny_proj > 10
        {
            use_xproj = if self.m_use_raw_stack != 0 {
                nproj_xyz[0]
            } else {
                self.m_nx_proj
            };
            use_yproj = if self.m_use_raw_stack != 0 {
                nproj_xyz[1]
            } else {
                self.m_ny_proj
            };
            sub_frac = self.m_lim_reproj as f32 / (use_xproj * use_yproj) as f32;
            sub_frac = if sub_frac < 1. { sub_frac } else { 1. };
            ind1 = (2. * ((use_xproj as f32 * sub_frac - 2.) / 2.)) as i32;
            ind2 = (2. * ((use_yproj as f32 * sub_frac - 2.) / 2.)) as i32;
            let err = unsafe {
                iiu_set_position(1, self.m_num_views / 2, 0);
                iiu_read_sec_part(
                    1,
                    temp_arr.as_mut_ptr().cast(),
                    ind1,
                    use_xproj / 2 - ind1 / 2,
                    use_xproj / 2 + ind1 / 2 - 1,
                    use_yproj / 2 - ind2 / 2,
                    use_yproj / 2 + ind2 / 2 - 1,
                )
            };
            if err != 0 {
                exit_error(b"Reading middle section of input stack to determine SD");
            }
            array_min_max_mean_sd(
                &temp_arr,
                ind1,
                ind2,
                0,
                ind1 - 1,
                0,
                ind2 - 1,
                &mut dmin_tmp,
                &mut dmax_tmp,
                &mut sum_dbl,
                &mut sum_sq_dbl,
                &mut dmean_tmp,
                &mut sd_tmp,
            );
            self.m_out_scale *= reference_sd / sd_tmp;
            printf!(
                " Output scaling adjusted by %.2f for change in input standard deviation\n",
                cf((reference_sd / sd_tmp) as f64)
            );
        }
        //
        // Scale dimensions down by binning then report them
        //
        ix_subset = ix_subset_in;
        nx_full = nx_full_in;
        if image_binned > 1 || expand_factor != 1. {
            // Splittilt should have adjusted slices for real slice numbers times binning
            if if_slice_in != 0 && (self.m_min_tot_slice <= 0 || self.m_islice_start > 0) {
                use_exp = self.m_min_tot_slice <= 0 && expand_factor != 1.;
                slice_adj = if use_exp {
                    b3dnint!(self.m_islice_start as f32 / expanded_binning)
                } else {
                    (self.m_islice_start + image_binned - 1) / image_binned
                };
                self.m_islice_start = b3dmax!(1, b3dmin!(self.m_ny_proj, slice_adj));
                slice_adj = if use_exp {
                    b3dnint!(self.m_islice_end as f32 / expanded_binning)
                } else {
                    (self.m_islice_end + image_binned - 1) / image_binned
                };
                self.m_islice_end = b3dmax!(1, b3dmin!(self.m_ny_proj, slice_adj));
            }
            if if_thick_in != 0 {
                self.m_ithick_bp = (self.m_ithick_bp as f32 / expanded_binning) as i32;
            }

            // Set up a floorFac so it does the same as integer arithmetic if no expand,
            // or does a nearest int otherwise
            use_exp = expand_factor != 1.;
            floor_fac = if use_exp { 0.5 } else { 0. };
            self.m_axis_xoffset = self.m_axis_xoffset / expanded_binning;
            nx_full = if use_exp {
                b3dnint!(nx_full_in as f32 / expanded_binning)
            } else {
                (nx_full_in + image_binned - 1) / image_binned
            };
            ny_full = if use_exp {
                b3dnint!(ny_full as f32 / expanded_binning)
            } else {
                (ny_full + image_binned - 1) / image_binned
            };
            ix_subset = (ix_subset_in as f32 / expanded_binning + floor_fac).floor() as i32;
            iy_subset = (iy_subset as f32 / expanded_binning + floor_fac).floor() as i32;
            x_offset = x_offset / expanded_binning;
            self.m_y_offset = self.m_y_offset / expanded_binning;
            if if_width_in != 0 {
                self.m_iwidth =
                    (self.m_iwidth as f32 / expanded_binning + floor_fac).floor() as i32;
            }
            if self.m_min_tot_slice > 0 && !self.m_rec_reproj {
                self.m_min_tot_slice = b3dmax!(
                    1,
                    b3dmin!(
                        self.m_ny_proj,
                        (self.m_min_tot_slice + image_binned - 1) / image_binned
                    )
                );
                self.m_max_tot_slice = b3dmax!(
                    1,
                    b3dmin!(
                        self.m_ny_proj,
                        (self.m_max_tot_slice + image_binned - 1) / image_binned
                    )
                );
            }
            if self.m_num_exact_cycles > 0 {
                self.m_exact_obj_size = self.m_exact_obj_size / expanded_binning;
            }
        }

        //
        if self.m_rec_reproj {
            if self.m_debug != 0 {
                printf!(
                    "%d %d %d %d %d %d %d\n",
                    ci(self.m_min_tot_slice),
                    ci(self.m_max_tot_slice),
                    ci(self.m_min_zreproj),
                    ci(self.m_max_zreproj),
                    ci(nrec_xyz[0]),
                    ci(nrec_xyz[1]),
                    ci(nrec_xyz[2])
                );
            }
            if (self.m_min_tot_slice <= 0
                && (self.m_min_zreproj <= 0 || self.m_max_zreproj > nrec_xyz[2]))
                || (self.m_min_tot_slice >= 0 && self.m_max_tot_slice > nrec_xyz[2])
            {
                exit_error(b"Min or Max Z coordinate to project is out of range");
            }

            if self.m_perpendicular == 0 {
                exit_error(b"Cannot reproject from reconstruction output with PARALLEL");
            }
            if self.m_iwidth != nrec_xyz[0]
                || self.m_islice_end + 1 - self.m_islice_start != nrec_xyz[2]
                || self.m_ithick_bp != nrec_xyz[1]
            {
                exit_error(b"Dimensions of rec file do not match expected values");
            }
        } else {
            //
            // Check conditions of SIRT (would slice increment work?)
            if self.m_num_sirt_iter > 0 {
                if self.m_perpendicular == 0 || x_offset != 0. {
                    exit_error(b"Cannot do SIRT with PARALLEL output or X shifts");
                }
                if self.m_nx_warp != 0
                    || if_zfactors != 0
                    || self.m_if_alpha > 1
                    || (self.m_if_alpha == 1 && self.m_interp_ord_xtilt == 0)
                {
                    exit_error(
                        b"Cannot do SIRT with  local alignments, Z factors, or variable or old-style X tilt",
                    );
                }
                if self.m_iwidth != self.m_nx_proj
                    || (!self.m_sirt_from_zero
                        && (self.m_iwidth != nrec_xyz[0]
                            || nrec_xyz[2] != self.m_ny_proj
                            || (self.m_ithick_bp != nrec_xyz[1] && self.m_vert_sirt_input == 0)))
                {
                    exit_error(
                        b"For SIRT, sizes of input projections, rec file, and width/thickness entries must match",
                    );
                }
                printf!(
                    "\n %d iterations of SIRT algorithm will be done\n",
                    ci(self.m_num_sirt_iter)
                );
                self.m_interp_fac_stretch = 0;
            }
            if if_slice_in != 0 {
                printf!(
                    "\n Rows %d to %d of the projection planes will be reconstructed.\n",
                    ci(self.m_islice_start),
                    ci(self.m_islice_end)
                );
            }
            if if_thick_in != 0 {
                printf!(
                    "\n Thickness of reconstructed slice is %d pixels.\n",
                    ci(self.m_ithick_bp)
                );
            }
            if del_angle != 0. || self.m_axis_xoffset != 0. {
                printf!(
                    "\n Output map rotated by %.1f degrees about tilt axis with respect to tilt origin\n Tilt axis displaced by %.2f pixels from centre of projection\n",
                    cf(del_angle as f64),
                    cf(self.m_axis_xoffset as f64)
                );
            }
            if nx_full != 0 || ny_full != 0 {
                printf!(
                    "\n Full aligned stack will be assumed to be %d by %d pixels\n",
                    ci(nx_full),
                    ci(ny_full)
                );
            }
            if if_subset_in != 0 {
                printf!(
                    "\n Aligned stack will be assumed to be a subset starting at %d %d\n",
                    ci(ix_subset),
                    ci(iy_subset)
                );
            }
            if if_width_in != 0 {
                printf!(
                    "\n Width of reconstruction is %d pixels\n",
                    ci(self.m_iwidth)
                );
            }
            if x_offset != 0. || self.m_y_offset != 0. {
                printf!(
                    "\n Output slice shifted up %.1f and to right %.1f pixels\n",
                    cf(self.m_y_offset as f64),
                    cf(x_offset as f64)
                );
            }
            if self.m_min_tot_slice > 0 {
                printf!(
                    "\n Computed slices are part of a total volume from slice %d to %d\n",
                    ci(self.m_min_tot_slice),
                    ci(self.m_max_tot_slice)
                );
            }
        }
        //
        // If NEWANGLES is 0, get angles from file header.  Otherwise check if angles OK
        //
        if new_angles == 0 && if_tilt_file == 0 {
            //
            // Tilt information is stored in stack header. Read into angles
            // array. All sections are assumed to be equally spaced. If not,
            // you need to set things up differently. In such a case, great
            // care should be taken, since missing views may have severe
            // effects on the quality of the reconstruction.
            //
            //
            // call irtdat(1, idtype, lens, nd1, nd2, vd1, vd2)
            //
            if id_type != 1 {
                exit_error(
                    b"There are no tilt angles from ANGLES or TILTFILE entries or from the image file header",
                );
            }
            //
            if nd1 != 2 {
                exit_error(b" Tilt axis not along Y.");
            }
            //
            del_theta = vd1;
            theta = vd2;
            //
            for nv in 1..=self.m_num_views as usize {
                self.m_angles[nv - 1] = theta;
                theta = theta + del_theta;
            }
            //
        } else if if_tilt_file == 1 && new_angles != 0 {
            exit_error(b"Tried to enter angles with both ANGLES and TILTFILE");
        } else if if_tilt_file == 1 {
            printf!(" Tilt angles were entered from a tilt file\n");
        } else if new_angles == self.m_num_views {
            printf!(" Tilt angles were entered with ANGLES card(s)\n");
        } else {
            exit_error(b"If using ANGLES, a value must be entered for each view");
        }
        //
        if num_compress > 0 {
            if num_compress == self.m_num_views {
                printf!(" Compression values were entered with COMPRESS card(s)\n");
            } else {
                exit_error(b"If using COMPRESS, a value must be entered for each view");
            }
            for nv in 0..self.m_num_views as usize {
                self.m_compress[nv] =
                    (1. + (self.m_compress[nv] as f64 - 1.) / comp_factor as f64) as f32;
            }
        }
        //
        if global_alpha != 0. {
            for iv in 0..self.m_num_views as usize {
                self.m_alpha[iv] = self.m_alpha[iv] - global_alpha;
            }
        }
        //
        if if_exp_weight != 0 {
            if self.m_if_log == 0 {
                printf!(" Weighting factors were entered from a file\n");
            } else {
                printf!(
                    " Weighting factors were entered but will be ignored because log is being taken\n"
                );
            }
        }
        //
        // if no INCLUDE entries, set up map to views, excluding any specified by
        // EXCLUDEs
        if num_view_use == 0 {
            for i in 1..=self.m_num_views {
                if number_in_list(i, Some(&iv_exclude), num_view_exclude, 0) == 0 {
                    self.m_map_used_view[num_view_use as usize] = i;
                    num_view_use += 1;
                }
            }
        }
        //
        // Replace angles at +/-90 with 89.95 etc
        for i in 1..=num_view_use as usize {
            j = self.m_map_used_view[i - 1];
            let a = &mut self.m_angles[(j - 1) as usize];
            if (b3dabs!(b3dabs!(*a) as f64 - 90.)) < 0.05 {
                *a = b3dsign!(90. - b3dsign!(0.05f64, 90. - b3dabs!(*a) as f64), *a) as f32;
            }
        }
        //
        // If reprojecting from rec and no angles entered, copy angles in original
        // order
        if self.m_rec_reproj && self.m_num_reproj == 0 {
            if n_views_reproj == 1 && iv_reproj[0] == 0 {
                self.m_num_reproj = self.m_num_views;
                for i in 0..self.m_num_views as usize {
                    ang_reproj[i] = self.m_angles[i];
                }
            } else if n_views_reproj > 0 {
                self.m_num_reproj = n_views_reproj;
                for i in 0..self.m_num_reproj as usize {
                    if iv_reproj[i] < 1 || iv_reproj[i] > self.m_num_views {
                        exit_error(b"View number to reproject is out of range");
                    }
                    ang_reproj[i] = self.m_angles[(iv_reproj[i] - 1) as usize];
                }
            } else {
                //
                // For default set of included views, order them by view number by
                // first ordering the mapUsedView array by view number.
                rs_sort_ints(&mut self.m_map_used_view, num_view_use);
                self.m_num_reproj = num_view_use;
                for i in 0..num_view_use as usize {
                    ang_reproj[i] = self.m_angles[(self.m_map_used_view[i] - 1) as usize];
                }
            }
        }
        //
        // order the MAPUSE array by angle
        for i in 0..(num_view_use - 1).max(0) as usize {
            for j in i + 1..num_view_use as usize {
                if self.m_angles[(self.m_map_used_view[i] - 1) as usize]
                    > self.m_angles[(self.m_map_used_view[j] - 1) as usize]
                {
                    indi = self.m_map_used_view[i];
                    self.m_map_used_view[i] = self.m_map_used_view[j];
                    self.m_map_used_view[j] = indi;
                }
            }
        }
        //
        // For SIRT, now copy the angles in the ordered list; this is the order
        // in which the reprojections are needed internally
        // Also adjust the recon mean for a fill value
        if self.m_num_sirt_iter > 0 {
            self.m_num_reproj = num_view_use;
            for i in 0..num_view_use as usize {
                ang_reproj[i] = self.m_angles[(self.m_map_used_view[i] - 1) as usize];
            }
            self.m_out_add = self.m_out_add / self.m_filter_scale;
            self.m_out_scale = self.m_out_scale * self.m_filter_scale;
            self.m_dmean_in = self.m_dmean_in / self.m_out_scale - self.m_out_add;
        }
        if self.m_num_reproj > 0 {
            self.m_cos_reproj = vec![0.; self.m_num_reproj as usize];
            self.m_sin_reproj = vec![0.; self.m_num_reproj as usize];
            for i in 0..self.m_num_reproj as usize {
                self.m_cos_reproj[i] = (deg_to_rad * ang_reproj[i]).cos();
                self.m_sin_reproj[i] = (deg_to_rad * ang_reproj[i]).sin();
            }
        }

        //
        // Open output map file
        delta = iiu_ret_delta(1);
        [origin_x, origin_y, origin_z] = iiu_ret_origin(1);
        if !self.m_rec_reproj {
            if (self.m_min_tot_slice <= 0 && (self.m_islice_start < 1 || self.m_islice_end < 1))
                || self.m_islice_start > self.m_ny_proj
                || self.m_islice_end > self.m_ny_proj
            {
                exit_error(b"Slice numbers out of range");
            }
            num_slices = (self.m_islice_end - self.m_islice_start) + 1;
            if self.m_min_tot_slice > 0 && self.m_islice_start < 1 {
                num_slices = self.m_max_tot_slice + 1 - self.m_min_tot_slice;
            }
            // print *,'NSLICE', minTotSlice, maxTotSlice, isliceStart, numSlices
            if num_slices <= 0 {
                exit_error(b"Slice numbers reversed");
            }
            if self.m_reproj_bp != 0 || self.m_read_base_rec {
                self.m_proj_line = vec![0.; self.m_iwidth as usize];
            }
            if defocus_file.is_some() && pix_for_defocus == 0. {
                pix_for_defocus = (delta[0] as f64 / 10.) as f32;
            }
            //
            // DNM 7/27/02: transfer pixel sizes depending on orientation of output
            //
            nout_xyz[0] = self.m_iwidth;
            cell[0] = self.m_iwidth as f32 * delta[0];
            if self.m_perpendicular != 0 {
                nout_xyz[1] = self.m_ithick_bp;
                nout_xyz[2] = num_slices;
                cell[1] = self.m_ithick_bp as f32 * delta[0];
                cell[2] = num_slices as f32 * delta[1];
            } else {
                nout_xyz[1] = num_slices;
                nout_xyz[2] = self.m_ithick_bp;
                cell[2] = self.m_ithick_bp as f32 * delta[0];
                cell[1] = num_slices as f32 * delta[1];
            }
            if self.m_reproj_bp != 0 {
                nout_xyz[1] = num_slices;
                nout_xyz[2] = self.m_num_reproj;
                cell[1] = num_slices as f32 * delta[1];
                cell[2] = delta[0] * self.m_num_reproj as f32;
                j = self.m_iwidth * self.m_num_reproj;
                self.m_xray_start = vec![0.; j as usize];
                self.m_yray_start = vec![0.; j as usize];
                self.m_num_pix_in_ray = vec![0; j as usize];
                self.m_max_ray_pixels = vec![0; self.m_num_reproj as usize];
                for i in 0..self.m_num_reproj as usize {
                    let j = i * self.m_iwidth as usize;
                    //
                    // Note that this will set cosReproj to 0 after you carefully kept it
                    // from being 0
                    set_projection_rays(
                        &mut self.m_sin_reproj[i],
                        &mut self.m_cos_reproj[i],
                        self.m_iwidth,
                        self.m_ithick_bp,
                        self.m_iwidth,
                        &mut self.m_xray_start[j..],
                        &mut self.m_yray_start[j..],
                        &mut self.m_num_pix_in_ray[j..],
                        &mut self.m_max_ray_pixels[i],
                    );
                }
            }
        } else {
            //
            // recReproj stuff
            nout_xyz[0] = self.m_max_xreproj + 1 - self.m_min_xreproj;
            nout_xyz[1] = self.m_max_zreproj + 1 - self.m_min_zreproj;
            self.m_ithick_reproj = self.m_max_yreproj + 1 - self.m_min_yreproj;
            nout_xyz[2] = self.m_num_reproj;
            if self.m_min_tot_slice > 0 {
                nout_xyz[1] = self.m_max_tot_slice + 1 - self.m_min_tot_slice;
            }
            if nout_xyz[0] < 1 || nout_xyz[1] < 1 || self.m_ithick_reproj < 1 {
                exit_error(b"Min and max limits for output are reversed for X, Y, or Z");
            }
            cell[0] = nout_xyz[0] as f32 * delta[0];
            cell[1] = nout_xyz[1] as f32 * delta[0];
            cell[2] = delta[0] * self.m_num_reproj as f32 / image_binned as f32;
            if self.m_proj_subtraction != 0
                && (nout_xyz[0] != self.m_nx_proj
                    || self.m_max_zreproj > self.m_ny_proj
                    || nout_xyz[2] != self.m_num_views)
            {
                exit_error(
                    b"Output size must match original projection file size for SIRT subtraction",
                );
            }
        }
        //
        // Check compatibility of base rec file
        if (self.m_read_base_rec && !self.m_rec_reproj)
            && (nrec_xyz[0] != self.m_iwidth
                || nrec_xyz[1] != self.m_ithick_bp
                || nrec_xyz[2] < self.m_islice_end)
        {
            exit_error(b"Base rec file is not the same size as output file");
        }
        //
        // Initialize parallel writing routines if bound file entered
        self.m_parallel_hdf = 0;
        real_chunk_run = self.m_min_tot_slice > 0
            && ((!self.m_rec_reproj && self.m_islice_start > 0)
                || (self.m_rec_reproj && self.m_min_zreproj > 0));
        if !proj_model {
            if real_chunk_run {
                self.m_parallel_hdf =
                    (bound_file.is_some() && ii_test_if_hdf(&output_file) > 0) as i32;
            } else if self.m_min_tot_slice > 0 {
                self.m_parallel_hdf = (b3d_output_file_type() == 5) as i32;
            }
        }
        ind1 = nout_xyz[0];
        if self.m_parallel_hdf != 0 {
            ind1 = -ind1;
        }
        ierr = iiu_par_wrt_initialize(
            &String::from_utf8_lossy(bound_file.as_deref().unwrap_or(b"")),
            7,
            ind1,
            nout_xyz[1],
            nout_xyz[2],
        );
        if ierr != 0 {
            exit_error_fmt!(
                "Initializing parallel write boundary file, error %d",
                ci(ierr)
            );
        }
        //
        // open old file if in chunk mode and there is real starting slice
        // otherwise open new file
        //
        if !proj_model {
            if real_chunk_run {
                if self.m_parallel_hdf != 0 {
                    par_wrt_properties(&mut ind1, &mut ind2, &mut k);
                    if b3d_lock_file(ind1) != 0 {
                        exit_error(b"Could not get lock for opening HDF file");
                    }
                    ind2 = nout_xyz[2];
                }
                unsafe {
                    iiu_open(2, &String::from_utf8_lossy(&output_file), "OLD");
                    iiu_ret_basic_head(
                        2,
                        nout_xyz.as_mut_ptr(),
                        mpxyz.as_mut_ptr(),
                        &mut self.m_new_mode,
                        &mut dmin_tmp,
                        &mut dmax_tmp,
                        &mut dmean_tmp,
                    );
                }
                if self.m_parallel_hdf != 0 {
                    nout_xyz[2] = ind2;
                }
            } else {
                unsafe {
                    iiu_open(2, &String::from_utf8_lossy(&output_file), "NEW");
                }
                iiu_create_header(
                    2,
                    &nout_xyz,
                    &nout_xyz,
                    self.m_new_mode,
                    &[self.m_title; 10],
                    0,
                );
            }
            if self.m_parallel_hdf != 0 && unsafe { iiu_file_type(2) } != 5 {
                exit_error(b"Expected output file to be an HDF file but it is not");
            }
            iiu_trans_labels(2, 1);

            // For HDF file, the cell and sample size were both set in setup, so leave
            // cell alone
            if !(self.m_parallel_hdf != 0 && real_chunk_run) {
                iiu_alt_cell(2, &cell);
            }
        }
        //
        // if doing perpendicular slices, set up header info to make coordinates
        // congruent with those of tilt series
        //
        if self.m_rec_reproj {
            if adjust_origin != 0 {
                origin_x -= delta[0] * (self.m_min_xreproj - 1) as f32;
                origin_y -= delta[2] * (self.m_min_zreproj - 1) as f32;
                iiu_alt_origin(2, &[origin_x, origin_y, origin_z]);
            } else {
                iiu_alt_origin(2, &[0., 0., 0.]);
            }
            out_hdr_tilt[0] = 0.;
            iiu_alt_tilt(2, &out_hdr_tilt);
        } else {
            if self.m_rotate_by90 == 0 {
                out_hdr_tilt[0] = 90.;
            }
            if adjust_origin != 0 {
                //
                // Full adjustment if requested
                origin_x = origin_x
                    - delta[0] * ((self.m_nx_proj / 2 - self.m_iwidth / 2) as f32 - x_offset);
                origin_z = origin_y - delta[0] * b3dmax!(0, self.m_islice_start - 1) as f32;
                if self.m_min_tot_slice > 0 && self.m_islice_start <= 0 {
                    origin_z = origin_y - delta[0] * (self.m_min_tot_slice - 1) as f32;
                }
                origin_y = (delta[0] as f64
                    * (self.m_ithick_bp as f64 / 2. + self.m_y_offset as f64))
                    as f32;

                // Adjust for 90 degree rotation/transposed sizes as newstack would
                if self.m_use_raw_stack != 0 && self.m_rot_flip_operation % 2 != 0 {
                    origin_x = (origin_x as f64
                        + (delta[0] * (self.m_nx_proj - self.m_ny_proj) as f32) as f64 / 2.)
                        as f32;
                    origin_z = (origin_z as f64
                        + (delta[0] * (self.m_ny_proj - self.m_nx_proj) as f32) as f64 / 2.)
                        as f32;
                }

                // Set these and the tilt angles as clip rotx would
                if self.m_rotate_by90 != 0 {
                    std::mem::swap(&mut origin_y, &mut origin_z);
                }
            } else {
                //
                // Legacy origin.  All kinds of wrong.
                origin_x = (cell[0] as f64 / 2. + self.m_axis_xoffset as f64) as f32;
                origin_y = (cell[1] as f64 / 2.) as f32;
                origin_z = -(b3dmax!(0, self.m_islice_start - 1) as f32);
            }

            if !proj_model && !(self.m_parallel_hdf != 0 && real_chunk_run) {
                iiu_alt_origin(2, &[origin_x, origin_y, origin_z]);
                iiu_alt_tilt(2, &out_hdr_tilt);
                iiu_alt_space_group(2, 1);
                if self.m_rotate_by90 != 0 {
                    out_hdr_tilt[0] = 90.;
                    iiu_alt_tilt_orig(2, &out_hdr_tilt);
                    out_hdr_tilt[0] = 0.;
                    iiu_alt_tilt(2, &out_hdr_tilt);
                }
            }
        }
        //
        // chunk mode starter run: write header and exit
        //
        if_exit = 0;
        if !proj_model {
            if self.m_min_tot_slice > 0 && !real_chunk_run {
                if self.m_parallel_hdf != 0 {
                    unsafe { iiu_write_dummy_sec_to_hdf(2) };
                }
                iiu_write_header(
                    2,
                    &self.m_title,
                    1,
                    self.m_dmin_in,
                    self.m_dmax_in,
                    self.m_dmean_in,
                );
                unsafe { iiu_close(2) };
                if_exit = 1;
            } else if self.m_min_tot_slice > 0 {
                if !self.m_rec_reproj {
                    unsafe { par_wrt_posn(2, self.m_islice_start - self.m_min_tot_slice, 0) };
                }
                if self.m_read_base_rec && !self.m_rec_reproj {
                    unsafe { iiu_set_position(3, self.m_islice_start - self.m_min_tot_slice, 0) };
                }
                if self.m_parallel_hdf != 0 {
                    unsafe { iiu_par_wrt_reclose_hdf(2, 1) };
                }
            }
        }
        //
        // If reprojecting, need to look up each angle in full list of angles and
        // find ones to interpolate from, then pack data into arrays that are
        // otherwise used for packing these factors down
        // lookupAngle returns 0-based indices
        if self.m_rec_reproj {
            for i in 0..self.m_num_reproj as usize {
                self.m_sin_beta[i] = ang_reproj[i];
                self.lookup_angle(
                    ang_reproj[i],
                    &self.m_angles,
                    self.m_num_views,
                    &mut ind1,
                    &mut ind2,
                    &mut frac,
                );
                let (i1, i2) = (ind1 as usize, ind2 as usize);
                self.m_cos_beta[i] = ((1. - frac as f64) * self.m_compress[i1] as f64
                    + (frac * self.m_compress[i2]) as f64)
                    as f32;
                self.m_sin_alpha[i] = ((1. - frac as f64) * self.m_alpha[i1] as f64
                    + (frac * self.m_alpha[i2]) as f64)
                    as f32;
                self.m_cos_alpha[i] = ((1. - frac as f64) * self.m_xzfac[i1] as f64
                    + (frac * self.m_xzfac[i2]) as f64)
                    as f32;
                temp_arr[i] = ((1. - frac as f64) * self.m_yzfac[i1] as f64
                    + (frac * self.m_yzfac[i2]) as f64) as f32;
                temp_arr[i + self.m_num_views as usize] =
                    ((1. - frac as f64) * self.m_expose_weight[i1] as f64
                        + (frac * self.m_expose_weight[i2]) as f64) as f32;
            }
            //
            // Do the same thing with all the local data: pack it into the spot at
            // the top of the local data then copy it back into the local area
            if self.m_nx_warp > 0 {
                let ib = ind_base as usize;
                for i in 0..(self.m_nx_warp * self.m_ny_warp) as usize {
                    for iv in 0..self.m_num_reproj as usize {
                        self.lookup_angle(
                            ang_reproj[iv],
                            &self.m_angles,
                            self.m_num_views,
                            &mut ind1,
                            &mut ind2,
                            &mut frac,
                        );
                        ind1 = ind1 + self.m_ind_warp[i];
                        ind2 = ind2 + self.m_ind_warp[i];
                        let (i1, i2) = (ind1 as usize, ind2 as usize);
                        let omf = 1. - frac as f64;
                        self.m_del_beta[ib + iv] = (omf * self.m_del_beta[i1] as f64
                            + (frac * self.m_del_beta[i2]) as f64)
                            as f32;
                        self.m_del_alpha[ib + iv] = (omf * self.m_del_alpha[i1] as f64
                            + (frac * self.m_del_alpha[i2]) as f64)
                            as f32;
                        self.m_warp_xzfac[ib + iv] = (omf * self.m_warp_xzfac[i1] as f64
                            + (frac * self.m_warp_xzfac[i2]) as f64)
                            as f32;
                        self.m_warp_yzfac[ib + iv] = (omf * self.m_warp_yzfac[i1] as f64
                            + (frac * self.m_warp_yzfac[i2]) as f64)
                            as f32;
                        for j in 0..6 {
                            self.m_fwarp[j + 6 * (ib + iv)] = (omf
                                * self.m_fwarp[j + 6 * i1] as f64
                                + (frac * self.m_fwarp[j + 6 * i2]) as f64)
                                as f32;
                        }
                    }
                    for iv in 0..self.m_num_reproj as usize {
                        let i1 = ib + iv;
                        let i2 = self.m_ind_warp[i] as usize + iv;
                        self.m_del_beta[i2] = self.m_del_beta[i1];
                        self.m_del_alpha[i2] = self.m_del_alpha[i1];
                        self.m_warp_xzfac[i2] = self.m_warp_xzfac[i1];
                        self.m_warp_yzfac[i2] = self.m_warp_yzfac[i1];
                        for j in 0..6 {
                            self.m_fwarp[j + 6 * i2] = self.m_fwarp[j + 6 * i1];
                        }
                    }
                }
            }
            //
            // Replace the mapUsedView array
            num_view_use = self.m_num_reproj;
            for i in 1..=num_view_use {
                self.m_map_used_view[(i - 1) as usize] = i;
            }
        } else {
            //
            // pack angles and other data down as specified by MAPUSE
            // Negate the z factors since things are upside down here
            // Note that local data is not packed but always referenced by mapUsedView
            //
            for i in 0..num_view_use as usize {
                let m = (self.m_map_used_view[i] - 1) as usize;
                self.m_sin_beta[i] = self.m_angles[m];
                self.m_cos_beta[i] = self.m_compress[m];
                self.m_sin_alpha[i] = self.m_alpha[m];
                self.m_cos_alpha[i] = self.m_xzfac[m];
                temp_arr[i] = self.m_yzfac[m];
                temp_arr[i + self.m_num_views as usize] = self.m_expose_weight[m];
            }
        }
        for i in 0..num_view_use as usize {
            self.m_angles[i] = self.m_sin_beta[i];
            self.m_compress[i] = self.m_cos_beta[i];
            self.m_alpha[i] = self.m_sin_alpha[i];
            self.m_xzfac[i] = -self.m_cos_alpha[i];
            self.m_yzfac[i] = -temp_arr[i];
            self.m_expose_weight[i] = temp_arr[i + self.m_num_views as usize];
        }
        num_view_orig = self.m_num_views;
        self.m_num_views = num_view_use;
        //
        // Etomo PEET requires the blank line to find the angles
        printf!("\n Projection angles:\n\n");
        for nv in 0..self.m_num_views {
            printf!("%9.2f", cf(self.m_angles[nv as usize] as f64));
            if (nv + 1) % 8 == 0 || nv == self.m_num_views - 1 {
                printf!("\n");
            }
        }
        printf!("\n");
        //
        // Turn off cosine stretch for high angles
        if (self.m_angles[0] as f64) < -80.
            || self.m_angles[(self.m_num_views - 1) as usize] as f64 > 80.
        {
            if self.m_interp_fac_stretch > 0 {
                printf!("\n Tilt angles are too high to use cosine stretching\n");
            }
            self.m_interp_fac_stretch = 0;
        }
        //
        // Set up trig tables -  Then convert angles to radians
        //
        self.m_beta_min = 10.;
        self.m_beta_max = -10.;
        self.m_min_cos_alpha = 10.;
        for iv in 0..self.m_num_views as usize {
            theta_view = self.m_angles[iv] + del_angle;
            if theta_view as f64 > 180. {
                theta_view = (theta_view as f64 - 360.) as f32;
            }
            if theta_view as f64 <= -180. {
                theta_view = (theta_view as f64 + 360.) as f32;
            }
            self.m_cos_beta[iv] = (theta_view * deg_to_rad).cos();
            //
            // Keep cosine from going to zero so it can be divided by
            if (b3dabs!(self.m_cos_beta[iv]) as f64) < 1.0e-6 {
                self.m_cos_beta[iv] = b3dsign!(1.0e-6f64, self.m_cos_beta[iv]) as f32;
            }
            //
            // Take the negative of the sine of tilt angle to account for the fact
            // that all equations are written for rotations in the X/Z plane,
            // viewed from  the negative Y axis
            // Take the negative of alpha because the entered value is the amount
            // that the specimen is tilted and we need to rotate by negative of that
            self.m_sin_beta[iv] = -(theta_view * deg_to_rad).sin();
            self.m_cos_alpha[iv] = (self.m_alpha[iv] * deg_to_rad).cos();
            self.m_sin_alpha[iv] = -(self.m_alpha[iv] * deg_to_rad).sin();
            self.m_angles[iv] = -deg_to_rad * (self.m_angles[iv] + del_angle);
            self.m_beta_min = if self.m_beta_min < self.m_angles[iv] {
                self.m_beta_min
            } else {
                self.m_angles[iv]
            };
            self.m_beta_max = if self.m_beta_max > self.m_angles[iv] {
                self.m_beta_max
            } else {
                self.m_angles[iv]
            };
            self.m_min_cos_alpha = if self.m_min_cos_alpha < self.m_cos_alpha[iv] {
                self.m_min_cos_alpha
            } else {
                self.m_cos_alpha[iv]
            };
        }
        //
        // If there are weighting angles, convert those the same way, otherwise
        // copy the main angles to weighting angles
        if self.m_num_wgt_angles > 0 {
            for iv in 0..self.m_num_wgt_angles as usize {
                self.m_wgt_angles[iv] = -deg_to_rad * (self.m_wgt_angles[iv] + del_angle);
            }
        } else {
            self.m_num_wgt_angles = self.m_num_views;
            for iv in 0..self.m_num_views as usize {
                self.m_wgt_angles[iv] = self.m_angles[iv];
            }
        }
        //
        // if fixed x axis tilt, set up to try to compute vertical planes
        // and interpolate output planes: adjust thickness that needs to
        // be computed, and find number of vertical planes that are needed
        //
        if if_zfactors > 0 && self.m_if_alpha == 0 {
            self.m_if_alpha = 1;
        }
        self.m_ithick_out = self.m_ithick_bp;
        self.m_ycen_mod_proj =
            ((self.m_ithick_bp / 2) as f64 + 0.5 + self.m_y_offset as f64) as f32;
        self.m_num_read_need = 0;
        if self.m_use_intersections != 0
            && (self.m_if_alpha != 0
                || self.m_interp_fac_stretch > 0
                || self.m_use_gpu
                || self.m_nx_warp != 0)
        {
            exit_error(
                b"You cannot use ray intersection areas with X-axis tilt, Z factors, local alignments, GPU, or cosine stretching",
            );
        }
        if self.m_if_alpha == 1
            && self.m_nx_warp == 0
            && self.m_interp_ord_xtilt > 0
            && if_zfactors == 0
            && !self.m_rec_reproj
        {
            self.m_if_alpha = -1;
            self.m_ithick_out = self.m_ithick_bp;
            self.m_ithick_bp =
                ((self.m_ithick_out as f32 / self.m_cos_alpha[0]) as f64 + 4.5) as i32;
            self.m_num_vert_needed =
                ((self.m_ithick_out as f32 * b3dabs!(self.m_sin_alpha[0])) as f64 + 5.) as i32;
            if self.m_num_sirt_iter > 0 {
                self.m_save_vert_slices = vert_out_file.is_some();
                if !self.m_sirt_from_zero && self.m_vert_sirt_input == 0 {
                    self.m_num_read_need = ((self.m_ithick_bp as f32 * b3dabs!(self.m_sin_alpha[0]))
                        as f64
                        + 4.) as i32;
                }
                if !self.m_sirt_from_zero
                    && self.m_vert_sirt_input != 0
                    && self.m_ithick_bp != nrec_xyz[1]
                {
                    exit_error(
                        b"Thickness of vertical slice input file does not match needed thickness",
                    );
                }
            }
        }
        if self.m_if_alpha != -1
            && self.m_num_sirt_iter > 0
            && (self.m_vert_sirt_input != 0 || vert_out_file.is_some())
        {
            exit_error(
                b"VertForSIRTInput or VertSliceOutputFile cannot be entered unless vertical slices are being computed",
            );
        }
        //
        // Now that we know vertical slices, open or set up output file for them under
        // SIRT
        if self.m_num_sirt_iter > 0 && self.m_save_vert_slices {
            //
            // Initialize parallel writing if vertical bound file
            for i in 0..3 {
                nvs_xyz[i] = nout_xyz[i];
            }
            nvs_xyz[1] = self.m_ithick_bp;
            if self.m_parallel_hdf != 0
                && self.m_min_tot_slice > 0
                && self.m_islice_start > 0
                && vert_bound_file.is_none()
            {
                exit_error(
                    b"A boundary file for vertical slices must be entered if there is one for primary output and output file type is HDF",
                );
            }
            ind1 = nvs_xyz[0];
            if self.m_parallel_hdf != 0 {
                ind1 = -ind1;
            }
            ierr = iiu_par_wrt_initialize(
                &String::from_utf8_lossy(vert_bound_file.as_deref().unwrap_or(b"")),
                8,
                ind1,
                nvs_xyz[1],
                nvs_xyz[2],
            );
            if ierr != 0 {
                exit_error_fmt!(
                    "Initializing parallel write boundary file for vertical slices, error",
                    ci(ierr)
                );
            }
            let vert_out = vert_out_file.clone().unwrap_or_default();
            if self.m_min_tot_slice > 0 && self.m_islice_start > 0 {
                if self.m_parallel_hdf != 0 {
                    par_wrt_properties(&mut ind1, &mut ind2, &mut k);
                    if b3d_lock_file(ind1) != 0 {
                        exit_error(b"Could not get lock for opening HDF file");
                    }
                    ind2 = nvs_xyz[2];
                }
                unsafe {
                    iiu_open(6, &String::from_utf8_lossy(&vert_out), "OLD");
                    iiu_ret_basic_head(
                        6,
                        nvs_xyz.as_mut_ptr(),
                        mpxyz.as_mut_ptr(),
                        &mut self.m_new_mode,
                        &mut dmin_tmp,
                        &mut dmax_tmp,
                        &mut dmean_tmp,
                    );
                }
                if self.m_parallel_hdf != 0 {
                    nvs_xyz[2] = ind2;
                }
            } else {
                unsafe {
                    iiu_open(6, &String::from_utf8_lossy(&vert_out), "NEW");
                }
                iiu_create_header(6, &nvs_xyz, &nvs_xyz, 2, &[self.m_title; 10], 0);
            }
            if self.m_parallel_hdf != 0 && unsafe { iiu_file_type(6) } != 5 {
                exit_error(b"Expected vertical slice file to be an HDF file but it is not");
            }
            //
            // chunk mode: either write header, or set up to write correct location
            if if_exit != 0 {
                if self.m_parallel_hdf != 0 {
                    unsafe { iiu_write_dummy_sec_to_hdf(6) };
                }
                iiu_write_header(
                    6,
                    &self.m_title,
                    0,
                    self.m_dmin_in,
                    self.m_dmax_in,
                    self.m_dmean_in,
                );
                unsafe { iiu_close(6) };
            } else if self.m_min_tot_slice > 0 {
                unsafe { par_wrt_posn(6, self.m_islice_start - self.m_min_tot_slice, 0) };
                if self.m_parallel_hdf != 0 {
                    unsafe { iiu_par_wrt_reclose_hdf(6, 1) };
                }
            }
            par_wrt_set_current(0);
        }

        if if_exit != 0 {
            printf!("Exiting after setting up output file for chunk writing\n");
            c_exit(0);
        }

        if nx_full == 0 {
            nx_full = self.m_nx_proj;
            nx_full_in = b3dnint!(self.m_nx_proj as f32 * expanded_binning);
        }
        if ny_full == 0 {
            ny_full = self.m_ny_proj;
        }

        //
        // if doing warping, convert the angles to radians and set sign
        // Also cancel the z factors if global entry was not made
        //
        if self.m_nx_warp > 0 {
            for i in 0..(num_view_orig * self.m_nx_warp * self.m_ny_warp) as usize {
                self.m_del_beta[i] = -deg_to_rad * self.m_del_beta[i];
            }
            for iv in 0..self.m_num_views as usize {
                for i in 0..(self.m_nx_warp * self.m_ny_warp) as usize {
                    ind = self.m_ind_warp[i] + self.m_map_used_view[iv] - 1;
                    let iu = ind as usize;
                    self.m_cwarp_beta[iu] = (self.m_angles[iv] + self.m_del_beta[iu]).cos();
                    self.m_swarp_beta[iu] = (self.m_angles[iv] + self.m_del_beta[iu]).sin();
                    self.m_cwarp_alpha[iu] =
                        (deg_to_rad * (self.m_alpha[iv] + self.m_del_alpha[iu])).cos();
                    self.m_swarp_alpha[iu] =
                        -(deg_to_rad * (self.m_alpha[iv] + self.m_del_alpha[iu])).sin();
                    if if_zfactors == 0 {
                        self.m_warp_xzfac[iu] = 0.;
                        self.m_warp_yzfac[iu] = 0.;
                    }
                }
            }
            //
            // See if local scale was entered; if not see if it can be set from
            // pixel size and local align pixel size
            if scale_local as f64 <= 0. {
                scale_local = 1.;
                if pixel_local > 0. {
                    scale_local = pixel_local / delta[0];
                    if (b3dabs!(scale_local as f64 - 1.)) > 0.001 {
                        printf!(
                            "\nScaling of local alignments by %.3f determined from pixel sizes\n",
                            cf(scale_local as f64)
                        );
                    }
                }
            }
            //
            // scale the x and y dimensions and shifts if aligned data were
            // shrunk relative to the local alignment solution
            // 10/16/04: fixed to use mapUsedView to scale used views properly
            //
            if scale_local != 1. {
                self.m_ix_start_warp = b3dnint!(self.m_ix_start_warp as f32 * scale_local);
                self.m_iy_start_warp = b3dnint!(self.m_iy_start_warp as f32 * scale_local);
                self.m_idel_xwarp = b3dnint!(self.m_idel_xwarp as f32 * scale_local);
                self.m_idel_ywarp = b3dnint!(self.m_idel_ywarp as f32 * scale_local);
                for iv in 0..self.m_num_views as usize {
                    for i in 0..(self.m_nx_warp * self.m_ny_warp) as usize {
                        ind = self.m_ind_warp[i] + self.m_map_used_view[iv] - 1;
                        self.m_fwarp[4 + 6 * ind as usize] *= scale_local;
                        self.m_fwarp[5 + 6 * ind as usize] *= scale_local;
                    }
                }
            }
            //
            // If using raw stack in addition to warping, now multiply these transforms
            // by the inverse of align xfs
            if self.m_use_raw_stack == 1 {
                for iv in 0..self.m_num_views as usize {
                    for i in 0..(self.m_nx_warp * self.m_ny_warp) as usize {
                        kti = self.m_map_used_view[iv] - 1;
                        ind = self.m_ind_warp[i] + kti;
                        let first: [f32; 6] = self.m_fwarp[6 * ind as usize..6 * ind as usize + 6]
                            .try_into()
                            .unwrap();
                        xf_mult(
                            &first,
                            &raw_xfs[6 * kti as usize..],
                            &mut self.m_fwarp[6 * ind as usize..],
                            2,
                        );
                    }
                }
            }
            //
            // if the input data is a subset in X or Y, subtract starting
            // coordinates from ixStartWarp and iyStartWarp
            //
            self.m_ix_start_warp = self.m_ix_start_warp - ix_subset;
            self.m_iy_start_warp = self.m_iy_start_warp - iy_subset;
        }

        // Set up a thickness and width to use for various steps below that corresponds
        // to the pixels in supersampled slice plus border
        thick_for_bp = self.m_ithick_bp;
        width_for_bp = self.m_iwidth;
        ss_border = 0;
        if self.m_super_sample_fac > 1 {
            self.m_mask_edges = 0;
            if self.m_num_sirt_iter != 0 {
                self.m_super_samp_border = 0;
            }
            if self.m_iwidth + 2 * self.m_super_samp_border > nx_full {
                self.m_super_samp_border = b3dmax!(0, (nx_full - self.m_iwidth) / 2);
            }
            width_for_bp = self.m_iwidth + 2 * self.m_super_samp_border;
            thick_for_bp = self.m_ithick_bp + 2 * self.m_super_samp_border;
            ss_border = self.m_super_samp_border;
        }
        let _ = thick_for_bp;

        if subset_load_ratio > 0. && (self.m_num_sirt_iter != 0 || self.m_rec_reproj) {
            exit_error(
                b"You cannot load subsets in X when doing SIRT or reprojecting from a reconstruction",
            );
        }
        //
        // If subset loading is allowed, see if it is less than size to load, and if so
        // set up the offset and change the mNxProj to this size
        if subset_load_ratio > 1.
            && ((width_for_bp as f32 * subset_load_ratio) as f64) < 0.9 * self.m_nx_proj as f64
        {
            // Set up center coordinates and compute projection of the needed box
            self.set_center_coords(
                nx_full,
                nx_full_in,
                ny_full,
                expanded_binning,
                ix_subset,
                ix_subset_in,
                iy_subset,
                x_offset,
                &mut xoff_adj,
            );
            xproj_min = 1.0e10;
            xproj_max = -1.0e10;

            // Loop on coordinates that span the whole supersample reconstructed slice
            // but are relative to the original centers, so they start negative with
            // border subtracted
            for iv in 0..num_view_use {
                let mut kti = self.m_islice_start;
                while kti <= self.m_islice_end {
                    let mut i = 1 - ss_border;
                    while i <= self.m_ithick_bp + ss_border {
                        let mut j = 1 - ss_border;
                        while j <= self.m_iwidth + ss_border {
                            // The source passes `smag`, `nd1`, `nd2`, `vd1`, `vd2`,
                            // `stretch` and `strPhi` as scratch; none is read again.
                            let mut nd1_scratch: i32 = 0;
                            let mut nd2_scratch: i32 = 0;
                            let mut vd1_scratch: f32 = 0.;
                            let mut vd2_scratch: f32 = 0.;
                            self.projection_position(
                                iv + 1,
                                j as f32,
                                i as f32,
                                kti as f32,
                                self.m_ycen_out,
                                &mut xproj,
                                &mut smag,
                                &mut nd1_scratch,
                                &mut nd2_scratch,
                                &mut vd1_scratch,
                                &mut vd2_scratch,
                                &mut stretch,
                                &mut str_phi,
                            );
                            xproj = b3dmax!(0., b3dmin!(self.m_nx_proj as f32, xproj));
                            xproj_min = if xproj_min < xproj { xproj_min } else { xproj };
                            xproj_max = if xproj_max > xproj { xproj_max } else { xproj };
                            j += self.m_iwidth + 2 * ss_border - 1;
                        }
                        i += self.m_ithick_bp + 2 * ss_border - 1;
                    }
                    kti += self.m_islice_end + 1 - self.m_islice_start;
                }
            }

            load_len = 2 * b3dnint!(((xproj_max - xproj_min) * subset_load_ratio) as f64 / 2.);
            if (load_len as f64) < 0.9 * self.m_nx_proj as f64 {
                border = ((load_len as f32 - (xproj_max - xproj_min)) as f64 / 2.) as f32;
                self.m_load_xoffset = b3dmax!(0., xproj_min - border) as i32;
                if self.m_load_xoffset + load_len > self.m_nx_proj {
                    self.m_load_xoffset = self.m_nx_proj - load_len;
                }
                ix_subset += self.m_load_xoffset;
                ix_subset_in += b3dnint!(self.m_load_xoffset as f32 * expanded_binning);
                x_offset = (x_offset as f64
                    + ((self.m_load_xoffset + load_len / 2) as f64 - self.m_nx_proj as f64 / 2.))
                    as f32;
                self.m_nx_proj = load_len;
            }
            if self.m_debug != 0 {
                printf!(
                    "xprojMin = %g,  xprojMax = %g,  loadLen = %d,  mLoadXoffset = %d\n",
                    cf(xproj_min as f64),
                    cf(xproj_max as f64),
                    ci(load_len),
                    ci(self.m_load_xoffset)
                );
            }
        }

        // Adjust scaling for any kind of sub-line being filtered, either aligned stack
        // or from restricted load
        if !self.m_rec_reproj && self.m_num_sirt_iter == 0 {
            self.m_adjust_out_add_fac = self.get_padded_input_size(nx_full) as f32
                / self.get_padded_input_size(self.m_nx_proj) as f32;
            self.m_out_scale *= self.m_adjust_out_add_fac;
        }

        if if_hamming_like > 0 {
            irad_max = (self.m_nx_proj as f32 * rad_max) as i32;
            rad_fall = (0.438 * (0.5 * self.m_nx_proj as f64 - irad_max as f64)) as f32;
            if_mult_by_gaussian = 1;
            printf!(
                "\n Doing Hamming-like filter by multiplying by Gaussian with cutoff =%7.4f  sigma = %7.4f\n",
                cf((irad_max as f32 / self.m_nx_proj as f32) as f64),
                cf((rad_fall / self.m_nx_proj as f32) as f64)
            );
        }

        if if_radial != 0 {
            irad_max = rad_max as i32;
            irad_fall = rad_fall as i32;
            if irad_max == 0 {
                irad_max = (self.m_nx_proj as f32 * rad_max) as i32;
            }
            if irad_fall == 0 {
                rad_fall = self.m_nx_proj as f32 * rad_fall;
            }
            //
            // Adjust the falloff if it is NOT true sigma, the function now uses value
            // correctly
            if if_true_sigma == 0 {
                rad_fall = (0.5f64.sqrt() * rad_fall as f64) as f32;
            }
            printf!(
                "\n Radial weighting function Gaussian starts at cutoff =%7.4f  sigma =%7.4f\n",
                cf((irad_max as f32 / self.m_nx_proj as f32) as f64),
                cf((rad_fall / self.m_nx_proj as f32) as f64)
            );
        }

        if self.m_mask_edges != 0 {
            self.m_num_extra_mask_pix = b3dmax!(
                -self.m_nx_proj / 50,
                b3dmin!(self.m_nx_proj / 50, self.m_num_extra_mask_pix)
            );
            printf!(
                "\n Mask applied to edges of output slices with %d extra pixels masked\n",
                ci(self.m_num_extra_mask_pix)
            );
        }
        //
        // Set up an effective scaling factor for non-log scaling to test output
        // The equation in ( ) is based on fitting to the amplification of the range
        // with linear scaling for different input sizes
        self.m_effective_scale = 1.;
        if self.m_if_log == 0 {
            self.m_effective_scale =
                (self.m_out_scale as f64 * (0.011 * self.m_nx_proj as f64 + 6.) / 2.) as f32;
        }
        //
        // Set up input/output center coordinates for real
        self.set_center_coords(
            nx_full,
            nx_full_in,
            ny_full,
            expanded_binning,
            ix_subset,
            ix_subset_in,
            iy_subset,
            x_offset,
            &mut xoff_adj,
        );
        if self.m_num_sirt_iter > 0 && (b3dabs!(xoff_adj + self.m_axis_xoffset) as f64) > 0.1 {
            if self.m_nx_proj % 2 > 0 {
                exit_error(b"Cannot do internal SIRT with an odd input size in X");
            }
            exit_error(
                b"Cannot do internal SIRT with a tilt axis offset from center of input images",
            );
        }
        //
        // Done with array in its small form and with angReproj
        drop(temp_arr);
        drop(ang_reproj);
        drop(iv_exclude);
        drop(iv_reproj);
        //
        // Here is the place to project model points and exit
        if proj_model {
            self.project_model(
                imod_to_project.take().unwrap(),
                &output_file,
                angle_output.as_deref(),
                transform_file.as_deref(),
                num_view_orig,
                defocus_file.as_deref(),
                pix_for_defocus,
                focus_invert,
            );
        }
        drop(raw_xfs);
        //
        // If reprojecting, set the pointers and return
        if self.m_rec_reproj {
            self.m_min_xload = self.m_min_xreproj;
            self.m_max_xload = self.m_max_xreproj;
            if self.m_nx_warp != 0 {
                self.m_min_xload = b3dmax!(1, self.m_min_xload - 100);
                self.m_max_xload = b3dmin!(nrec_xyz[0], self.m_max_xload + 100);
            }
            iwide_reproj = self.m_max_xload + 1 - self.m_min_xload;
            self.m_islice_size_bp = iwide_reproj as i64 * self.m_ithick_reproj as i64;
            self.m_in_plane_size = self.m_islice_size_bp as i32;
            if self.m_nx_warp != 0 {
                self.m_dx_warp_delz = (self.m_idel_xwarp as f64 / 2.) as f32;
                self.m_num_warp_delz =
                    (b3dmax!(2., (iwide_reproj - 1) as f64 / self.m_dx_warp_delz as f64) + 1.)
                        as i32;
                self.m_dx_warp_delz =
                    ((iwide_reproj as f64 - 1.) / (self.m_num_warp_delz as f64 - 1.)) as f32;
            }
            //
            // Get projection offsets for getting from coordinate in reprojection
            // to coordinate in original projections.  The X coordinate must account
            // for the original offset in building the reconstruction plus any
            // additional offset.  But the line number here is the line # in the
            // reconstruction so we only need to adjust Y by the original starting
            // line.  Also replace the slice limits.
            self.m_xproj_offset =
                (self.m_min_xreproj - 1 + self.m_nx_proj / 2 - self.m_iwidth / 2) as f32 - x_offset;
            self.m_yproj_offset = (self.m_islice_start - 1) as f32;
            self.m_islice_start = self.m_min_zreproj;
            self.m_islice_end = self.m_max_zreproj;
            self.m_iwidth = nout_xyz[0];
            self.m_ny_proj = nrec_xyz[2];
            self.m_dmean_in =
                (self.m_dmean_in / self.m_out_scale - self.m_out_add) / self.m_filter_scale;
            if self.m_thresh_polarity != 0. {
                self.m_thresh_for_reproj = (self.m_thresh_for_reproj / self.m_out_scale
                    - self.m_out_add)
                    / self.m_filter_scale;
                self.m_thresh_mark_val = self.m_thresh_for_reproj;
                self.m_thresh_fill_val = self.m_dmean_in;
                if self.m_thresh_sum_fac < 1. {
                    self.m_thresh_mark_val = self.m_thresh_mark_val * self.m_ithick_reproj as f32;
                }
            }
            self.m_ip_extra_size = 0;
            self.m_num_pad = 0;
            if self.m_debug != 0 {
                printf!(
                    "scale: %f %f",
                    cf(self.m_out_scale as f64),
                    cf(self.m_out_add as f64)
                );
            }

            num_need_eval = b3dmin!(num_need_eval, self.m_islice_end + 1 - self.m_islice_start);
            self.set_needed_slices(&mut max_needs, num_need_eval);
            if self.allocate_array(&max_needs, num_need_eval, 1, min_memory) == 0 {
                exit_error(
                    b"The main array cannot be allocated large enough to reproject a single Y value",
                );
            }

            self.m_reproj_lines = vec![0.; (self.m_iwidth * self.m_num_planes) as usize];
            //
            if self.m_proj_subtraction != 0 {
                self.m_orig_lines = vec![0.; (self.m_iwidth * self.m_num_planes) as usize];
            }
            if self.m_use_gpu {
                ind = 0;
                if self.m_debug != 0 {
                    ind = 1;
                }
                self.m_use_gpu = gpu_available(
                    ind_gpu,
                    &mut gpu_memory,
                    &mut max_tex_2d,
                    &mut max_tex_layer,
                    &mut max_tex_3d,
                    ind,
                ) != 0;
                if !self.m_use_gpu {
                    self.m_gpu_err_string = "No GPU is available".to_string();
                }
                if !self.m_use_gpu && self.m_debug == 0 {
                    self.m_gpu_err_string += ", run gputilttest for more details";
                }
                ind = max_needs[0] * self.m_in_plane_size + self.m_iwidth * self.m_num_planes;
                iex = self.m_iwidth * self.m_num_planes;
                kti = 0;
                pack_local = Vec::new();
                if self.m_use_gpu && self.m_nx_warp > 0 {
                    ind = ind
                        + max_needs[0] * (8 * self.m_iwidth + self.m_num_warp_delz)
                        + 12 * self.m_num_warp_pos * self.m_num_views;
                    iex = iex + 12 * self.m_num_warp_pos * self.m_num_views;
                    kti = self.m_num_warp_delz;
                    pack_local = self.pack_local_data();
                }
                if self.m_use_gpu {
                    self.m_use_gpu = (4 * ind) as f64 <= (gpu_memory_frac * gpu_memory) as f64;
                    if !self.m_use_gpu {
                        self.m_gpu_err_string = "GPU is available but it has insufficient memory to reproject with current parameters".to_string();
                    }
                }
                if self.m_use_gpu {
                    self.allocate_gpu_planes(
                        iex,
                        self.m_nx_warp * self.m_ny_warp,
                        kti,
                        0,
                        self.m_num_planes,
                        iwide_reproj,
                        self.m_ithick_reproj,
                        &max_needs,
                        if_zfactors,
                        gpu_memory,
                        gpu_memory_frac,
                        &mut if_3d_texture,
                        &max_tex_2d,
                        &max_tex_3d,
                        &max_tex_layer,
                    );
                }
                if self.m_use_gpu && self.m_nx_warp != 0 {
                    self.m_use_gpu =
                        gpu_load_locals(&pack_local, self.m_nx_warp * self.m_ny_warp) == 0;
                    if !self.m_use_gpu {
                        self.m_gpu_err_string =
                            "Failed to load local alignment data into GPU".to_string();
                    }
                }
                if self.m_use_gpu {
                    printf!("Using the GPU for reprojection\n");
                }
                drop(pack_local);
                self.warn_or_exit_if_no_gpu();
                if !self.m_use_gpu {
                    printf!("The GPU cannot be used, using the CPU for reprojection\n");
                }
            }
            //
            // Finally allocate the warpDelz now that number of lines is known,
            // and projecton factors now that number of planes is known
            if self.m_nx_warp != 0 {
                kti = iwide_reproj * self.m_num_planes;
                self.m_warp_delz =
                    vec![0.; (self.m_num_warp_delz * b3dmax!(1, self.m_num_gpu_planes)) as usize];
                self.m_xproj_fs = vec![0.; kti as usize];
                self.m_xproj_zs = vec![0.; kti as usize];
                self.m_yproj_fs = vec![0.; kti as usize];
                self.m_yproj_zs = vec![0.; kti as usize];
            }
            return;
        }
        //
        // BACKPROJECTION ONLY.  First allocate the projection factor array, which has
        // to be big enough for every X index on supersample slice with border
        if self.m_nx_warp != 0 {
            kti = width_for_bp * self.m_super_sample_fac;
            self.m_xproj_fs = vec![0.; kti as usize];
            self.m_xproj_zs = vec![0.; kti as usize];
            self.m_yproj_fs = vec![0.; kti as usize];
            self.m_yproj_zs = vec![0.; kti as usize];
        }
        //
        // If reading base, figure out total views being added and adjust scales
        if self.m_read_base_rec && !self.m_rec_subtraction {
            iv = self.m_num_views;
            for j in 0..self.m_num_views as usize {
                k = 0;
                for i in 0..self.m_num_view_subtract as usize {
                    if self.m_iview_subtract[i] == 0
                        || self.m_map_used_view[j] == self.m_iview_subtract[i]
                    {
                        k = 1;
                    }
                }
                iv = iv - 2 * k;
            }
            self.m_base_out_scale = self.m_out_scale / self.m_num_view_base as f32;
            self.m_base_out_add = self.m_out_add * self.m_num_view_base as f32;
            self.m_out_scale = self.m_out_scale / (iv + self.m_num_view_base) as f32;
            self.m_out_add = self.m_out_add * (iv + self.m_num_view_base) as f32;
            if self.m_debug != 0 {
                printf!(
                    "base: %f  %f\n",
                    cf(self.m_base_out_scale as f64),
                    cf(self.m_base_out_add as f64)
                );
            }
        } else if self.m_num_sirt_iter <= 0 {
            self.m_out_scale = self.m_out_scale / self.m_num_views as f32;
            self.m_out_add = self.m_out_add * self.m_num_views as f32;
        }
        if self.m_debug != 0 {
            printf!(
                "scale: %f %f\n",
                cf(self.m_out_scale as f64),
                cf(self.m_out_add as f64)
            );
        }
        num_need_eval = b3dmin!(num_need_eval, self.m_islice_end + 1 - self.m_islice_start);
        self.set_needed_slices(&mut max_needs, num_need_eval);
        if self.m_debug != 0 {
            for i in 0..num_need_eval as usize {
                printf!("  %d", ci(max_needs[i]));
            }
            printf!("\n");
        }
        if self.m_iter_for_report > 0 {
            kti = 3 * b3dmax!(1, self.m_num_sirt_iter);
            self.m_report_vals = vec![0.; kti as usize];
        }
        //
        // 12/13/09: removed fast backprojection code
        //
        nproj_pad = self.get_padded_input_size(self.m_nx_proj);
        self.m_num_pad = nproj_pad - self.m_nx_proj;
        //
        // Set up defaults for plane size and start of planes of input data
        self.m_islice_size_bp = self.m_iwidth as i64 * self.m_ithick_bp as i64;
        self.m_ip_extra_size = 0;
        self.m_nx_pad_dim = self.m_nx_proj + 2 + self.m_num_pad;
        self.m_in_plane_size = self.m_nx_pad_dim * self.m_num_views;
        self.m_nx_filt_dim = self.m_nx_pad_dim;
        if self.m_use_raw_stack != 0 {
            self.m_nx_filt_dim = (((deg_to_rad * self.m_raw_remaining_rot).cos()
                + (deg_to_rad * self.m_raw_remaining_rot.abs()).sin())
                * (self.m_nx_pad_dim / 2) as f32
                + 4.) as i32;
        }

        self.m_need_for_filt_arr = self.m_nx_filt_dim * self.m_num_views;
        self.m_need_for_out_arr = self.m_islice_size_bp as i32;
        self.setup_sizes_for_supersampling();
        if self.m_num_sirt_iter > 0 {
            if self.m_sirt_from_zero {
                self.m_need_for_filt_arr *= 2;
            }
            self.m_need_for_read_in_arr = self.m_islice_size_bp;
            self.m_need_for_work_arr = self.m_in_plane_size;
            nvs_xyz[0] = self.m_iwidth;
            nvs_xyz[1] = self.m_islice_end + 1 - self.m_islice_start;
            nvs_xyz[2] = self.m_num_views;
            if self.m_if_out_sirt_proj > 0 {
                unsafe { iiu_open(4, "sirttst.prj", "NEW") };
                iiu_create_header(4, &nvs_xyz, &nvs_xyz, 2, &[self.m_title; 10], 0);
                iiu_write_header(4, &self.m_title, 0, -1.0e6, 1.0e6, 0.);
            }
            nvs_xyz[1] = self.m_ithick_bp;
            nvs_xyz[2] = self.m_islice_end + 1 - self.m_islice_start;
            if self.m_if_out_sirt_rec > 0 {
                unsafe { iiu_open(5, "sirttst.drec", "NEW") };
                iiu_create_header(5, &nvs_xyz, &nvs_xyz, 2, &[self.m_title; 10], 0);
                iiu_write_header(5, &self.m_title, 0, -1.0e6, 1.0e6, 0.);
            }
        }
        self.m_max_stack = 0;
        //
        // Determine if GPU can be used, but don't try to allocate yet
        if self.m_use_gpu {
            ind = 0;
            if self.m_debug != 0 {
                ind = 1;
            }
            wall_start = wall_time();
            self.m_use_gpu = gpu_available(
                ind_gpu,
                &mut gpu_memory,
                &mut max_tex_2d,
                &mut max_tex_layer,
                &mut max_tex_3d,
                ind,
            ) != 0;
            if self.m_debug != 0 {
                printf!(
                    "Time to test if GPU available: %.4f\n",
                    cf(wall_time() - wall_start)
                );
            }
            if !self.m_use_gpu {
                self.m_gpu_err_string = "No GPU is available".to_string();
            }
            if !self.m_use_gpu && self.m_debug == 0 {
                self.m_gpu_err_string += ", run gputilttest for more details";
            }
            if self.m_use_gpu {
                //
                // Basic need is input planes for reconstructing one slice plus 2
                // slices for radial filter and planes being filtered, plus output
                // slice. Local alignment adds 4 arrays for local proj factors
                iv = (max_needs[0] + 2) * self.m_in_plane_size + self.m_islice_size_bp as i32;
                if self.m_nx_warp != 0 {
                    iv = iv
                        + 4 * (self.m_iwidth * self.m_num_views
                            + 12 * self.m_num_warp_pos * self.m_num_views);
                }
                if self.m_num_sirt_iter > 0 {
                    iv = iv + self.m_islice_size_bp as i32 + self.m_in_plane_size;
                }
                if self.m_sirt_from_zero {
                    iv = iv + self.m_in_plane_size;
                }
                self.m_use_gpu = (4 * iv) as f64 <= (gpu_memory_frac * gpu_memory) as f64;
                if self.m_use_gpu {
                    self.m_interp_fac_stretch = 0;
                } else {
                    self.m_gpu_err_string = "GPU is available but it has insufficient memory to backproject with current parameters".to_string();
                }
            }
            self.warn_or_exit_if_no_gpu();
            if !self.m_use_gpu {
                printf!("The GPU cannot be used; using CPU for backprojection\n");
            }
        }
        //
        // next evaluate cosine stretch and new-style tilt for memory
        //
        if self.m_if_alpha < 0 {
            //
            // new-style X-axis tilt
            //
            self.m_need_for_vert_arr = self.m_islice_size_bp * self.m_num_vert_needed as i64;
            if self.m_num_sirt_iter > 0 {
                self.m_need_for_read_in_arr =
                    self.m_num_read_need as i64 * self.m_ithick_out as i64 * self.m_iwidth as i64;
                self.m_need_for_work_arr = self.m_in_plane_size;
            }
            //
            // find out what cosine stretch adds if called for  (not allowed if SIRT)
            //
            if self.m_interp_fac_stretch > 0 {
                self.set_cos_stretch();
                self.m_in_plane_size = b3dmax!(
                    self.m_in_plane_size,
                    self.m_ind_stretch_line[self.m_num_views as usize]
                );
                self.m_ip_extra_size = self.m_in_plane_size;
                self.m_need_for_filt_arr = self.m_in_plane_size;
            }
            //
            // Does everything fit?  If not, drop back to old style tilting
            //
            if self.allocate_array(&max_needs, num_need_eval, 4, min_memory) == 0 {
                if self.m_num_sirt_iter > 0 {
                    exit_error(b"Allocating arrays needed for in-memory SIRT iterations");
                }
                self.m_if_alpha = 1;
                self.m_ithick_bp = self.m_ithick_out;
                self.m_islice_size_bp = self.m_iwidth as i64 * self.m_ithick_bp as i64;
                self.m_ycen_out =
                    ((self.m_ithick_bp / 2) as f64 + 0.5 + self.m_y_offset as f64) as f32;
                self.m_ip_extra_size = 0;
                self.m_in_plane_size = self.m_nx_pad_dim * self.m_num_views;
                self.m_need_for_filt_arr = self.m_nx_filt_dim * self.m_num_views;
                self.m_need_for_out_arr = self.m_islice_size_bp as i32;
                self.m_need_for_vert_arr = 0;
                self.setup_sizes_for_supersampling();
                printf!(
                    "\n Failed to allocate an array big enough to use new-style X-axis tilting\n"
                );
                self.set_needed_slices(&mut max_needs, num_need_eval);
            }
        }
        //
        // If not allocated yet and not warping, try cosine stretch here
        //
        if self.m_max_stack == 0 && self.m_nx_warp == 0 && self.m_interp_fac_stretch > 0 {
            self.set_cos_stretch();
            //
            // set size of plane as max of loading size and stretched size
            // also set that an extra plane is needed
            // if there is not enough space for the planes needed, then
            // disable stretching and drop back to regular code
            //
            self.m_in_plane_size = b3dmax!(
                self.m_in_plane_size,
                self.m_ind_stretch_line[self.m_num_views as usize]
            );
            self.m_ip_extra_size = self.m_in_plane_size;
            self.m_need_for_filt_arr =
                (if self.m_sirt_from_zero { 2 } else { 1 }) * self.m_in_plane_size;
            if self.allocate_array(&max_needs, num_need_eval, 1, min_memory) == 0 {
                self.m_ip_extra_size = 0;
                self.m_in_plane_size = self.m_nx_pad_dim * self.m_num_views;
                self.m_need_for_filt_arr =
                    (if self.m_sirt_from_zero { 2 } else { 1 }) * self.m_in_plane_size;
                self.m_interp_fac_stretch = 0;
                printf!("\n Failed to allocate an array big enough to use cosine stretching\n");
            }
        }
        //
        // If array still not allocated (failure, or local alignments), do it now
        if self.m_max_stack == 0 {
            if self.allocate_array(&max_needs, num_need_eval, 1, min_memory) == 0 {
                exit_error(
                    b"Could not allocate main array large enough to reconstruct a single slice",
                );
            }
        }
        if self.m_super_sample_fac > 1 {
            self.m_super_temp_arr =
                vec![0.; (self.m_nx_out_pad * self.m_super_sample_fac + 2) as usize];
            if self.m_proj_super_fac > 1 {
                // Set up the phase shift for the supersampled line and incorporate
                // scaling
                self.m_phase_real = vec![0.; (self.m_nx_pad_dim / 2) as usize];
                self.m_phase_imag = vec![0.; (self.m_nx_pad_dim / 2) as usize];
                dx_line = ((self.m_proj_super_fac as f64 - 1.)
                    / (2. * self.m_proj_super_fac as f64)) as f32;
                for ind in 0..self.m_nx_pad_dim / 2 {
                    freq = (0.5 * ind as f64 / ((self.m_nx_pad_dim / 2) as f64 - 1.)) as f32;
                    arg = (-2. * pi as f64 * freq as f64 * dx_line as f64) as f32;
                    self.m_phase_real[ind as usize] =
                        (arg.cos() as f64 * (self.m_proj_super_fac as f64).sqrt()) as f32;
                    self.m_phase_imag[ind as usize] =
                        (arg.sin() as f64 * (self.m_proj_super_fac as f64).sqrt()) as f32;
                }
            }
        }
        //
        // Set up radial weighting
        if self.m_num_sirt_iter > 0 && !self.m_sirt_from_zero {
            self.m_flat_frac = 2.;
        }
        self.radial_weights(irad_max, rad_fall, 1, if_mult_by_gaussian);
        if self.m_sirt_from_zero {
            frac = self.m_zero_weight;
            self.m_flat_frac = 2.;
            self.radial_weights(irad_max, rad_fall, 2, if_mult_by_gaussian);
            self.m_zero_weight = frac;
        }
        //
        // If Using GPU, make sure memory is OK now and allocate and load things
        // Add up "non-plane" memory needed there: output slice, filters and FFT array
        ind = self.m_islice_size_bp as i32 + 2 * self.m_need_for_filt_arr;
        pack_local = Vec::new();
        if self.m_use_gpu && self.m_nx_warp != 0 {
            ind = ind
                + 4 * self.m_num_views * self.m_iwidth
                + 12 * self.m_num_warp_pos * self.m_num_views;
            pack_local = self.pack_local_data();
        }

        // Add supersample needs: two super-sized and another regular size, plus large
        // FFT for input if expanding
        if self.m_super_sample_fac > 1 {
            ind += (2 * self.m_super_sample_fac * self.m_super_sample_fac + 1)
                * self.m_islice_size_bp as i32;
            if self.m_proj_super_fac > 1 {
                ind += self.m_in_plane_size;
            }
        }

        // Add arrays that could be needed for filtering raw input
        if self.m_use_raw_stack != 0 {
            ind += (3 * self.m_nx_pad_dim + self.m_nx_pad_dim / 2) * self.m_ny_raw_padded;
        }

        wall_start = wall_time();
        if self.m_use_gpu {
            if self.m_if_alpha <= 0 && self.m_nx_warp == 0 {
                iv = 0;
                j = 1;
                if self.m_num_sirt_iter > 0 {
                    iv = self.m_num_views;
                }
                if self.m_sirt_from_zero {
                    j = 2;
                }
                self.m_use_gpu = gpu_alloc_arrays(
                    self.m_iwidth,
                    self.m_ithick_bp,
                    self.m_nx_pad_dim,
                    self.m_num_views,
                    1,
                    self.m_num_views,
                    0,
                    0,
                    j,
                    iv,
                    0,
                    self.m_nx_filt_dim,
                    self.m_super_sample_fac * (if self.m_proj_super_fac > 1 { -1 } else { 1 }),
                    self.m_super_samp_border,
                    self.m_nx_out_pad,
                    self.m_ny_out_pad,
                    self.m_clean_super_fft,
                    1,
                    1,
                    0,
                ) == 0;
            } else {
                self.allocate_gpu_planes(
                    ind,
                    self.m_nx_warp * self.m_ny_warp,
                    0,
                    1,
                    self.m_ithick_bp,
                    self.m_nx_pad_dim,
                    self.m_num_views,
                    &max_needs,
                    if_zfactors,
                    gpu_memory,
                    gpu_memory_frac,
                    &mut if_3d_texture,
                    &max_tex_2d,
                    &max_tex_3d,
                    &max_tex_layer,
                );
            }
            if self.m_debug != 0 && self.m_use_gpu {
                printf!(
                    "Time to allocate on GPU: %.4f\n",
                    cf(wall_time() - wall_start)
                );
            }

            // print *,useGPU
            wall_start = wall_time();
            if self.m_use_gpu {
                self.m_use_gpu = gpu_load_filter(&self.m_filter_array) == 0;
                if !self.m_use_gpu {
                    self.m_gpu_err_string = "Failed to load filter array into GPU".to_string();
                }
            }
            // print *,useGPU
            if self.m_use_gpu && self.m_nx_warp != 0 {
                self.m_use_gpu = gpu_load_locals(&pack_local, self.m_nx_warp * self.m_ny_warp) == 0;
                if !self.m_use_gpu {
                    self.m_gpu_err_string =
                        "Failed to load local alignment data into GPU".to_string();
                }
            }
            // print *,useGPU
            if self.m_debug != 0 && self.m_use_gpu {
                printf!(
                    "Time to load filter/locals on GPU: %.4f\n",
                    cf(wall_time() - wall_start)
                );
            }
            drop(pack_local);
            self.warn_or_exit_if_no_gpu();
            if self.m_use_gpu {
                printf!("Using GPU for backprojection\n");
            } else {
                printf!("The GPU cannot be used - using CPU for backprojection\n");
            }
        }

        // Allocate maskEdges array
        if !self.m_rec_reproj {
            self.m_ix_unmasked_se = vec![0; (2 * self.m_ithick_bp) as usize];
        }
    }
}

impl Tilt {
    /// `Tilt::allocateGpuPlanes` (`tilt.cpp:5055`): allocate as many planes as
    /// possible on the GPU up to the number allowed in the main array.
    fn allocate_gpu_planes(
        &mut self,
        non_plane: i32,
        num_warps: i32,
        num_delz: i32,
        num_filts: i32,
        nygout: i32,
        nx_gplane: i32,
        ny_gplane: i32,
        max_needs: &[i32],
        if_zfactors: i32,
        gpu_memory: f32,
        gpu_memory_frac: f32,
        if_3d_texture: &mut i32,
        max_tex_2d: &[i32],
        max_tex_3d: &[i32],
        max_tex_layer: &[i32],
    ) {
        let mut max_gpu_plane: i32;
        let mut iper_plane: i32;
        let ind_eval_3d: i32;
        let mut err: i32 = 0;

        // Evaluate texture types in this order: 3D, layered, 2D.
        // But for no X tilt/Z-factors or locals, only 2D textures are allowed
        // And for no locals or no reprojection, only 2D layers or 2D textures are
        // allowed (Unimplemented options had no advantage)
        if self.m_if_alpha <= 0 && if_zfactors == 0 && self.m_nx_warp == 0 {
            ind_eval_3d = 2;
        } else if self.m_nx_warp == 0 && !self.m_rec_reproj {
            ind_eval_3d = 1;
        } else {
            ind_eval_3d = 0;
        }
        if self.m_debug != 0 {
            printf!(
                "In allocateGpuPlanes: %d %d %d %d %d\n",
                ci(self.m_if_alpha),
                ci(if_zfactors),
                ci(self.m_nx_warp),
                ci(self.m_rec_reproj as i32),
                ci(ind_eval_3d)
            );
            printf!(
                "%d %d %d %d %d %d \n",
                ci(nx_gplane),
                ci(max_tex_3d[0]),
                ci(ny_gplane),
                ci(max_tex_3d[1]),
                ci(max_needs[0]),
                ci(max_tex_3d[2])
            );
        }
        //
        // Check that the test entry was not for an unimplemented type
        if (ind_eval_3d > 0 && *if_3d_texture > 0)
            || (ind_eval_3d > 1 && *if_3d_texture > -999 && *if_3d_texture != 0)
        {
            exit_error(b"Entry for TextureType not legal with current type of computation");
        }
        //
        // If 3D is allowed and fits, take it; then if Layered is allowed and fits,
        // take it then fall back to 2D
        if *if_3d_texture == -999
            && ind_eval_3d == 0
            && self.m_proj_super_fac * nx_gplane < max_tex_3d[0]
            && ny_gplane < max_tex_3d[1]
            && max_needs[0] < max_tex_3d[2]
        {
            *if_3d_texture = 1;
        }
        if *if_3d_texture == -999
            && ind_eval_3d <= 1
            && self.m_proj_super_fac * nx_gplane < max_tex_layer[0]
            && ny_gplane < max_tex_layer[1]
            && max_needs[0] < max_tex_layer[2]
        {
            *if_3d_texture = -1;
        }
        if *if_3d_texture == -999 {
            *if_3d_texture = 0;
        }
        if self.m_debug != 0 {
            printf!(
                "Proceeding to allocate on GPU with texture type %d\n",
                ci(*if_3d_texture)
            );
        }

        //
        // Start with as many planes as possible but no more than in array
        // and if doing 2D textures, no more lines for array on GPU than the Y limit for
        // 2D textures (32767 before Fermi, typically 65536)
        iper_plane = self.m_in_plane_size;
        if num_delz > 0 {
            iper_plane = self.m_in_plane_size + 8 * self.m_iwidth + self.m_num_warp_delz;
        }
        max_gpu_plane = (((gpu_memory_frac * gpu_memory) as f64 / 4. - non_plane as f64)
            / iper_plane as f64) as i32;
        if max_gpu_plane < max_needs[0] {
            self.m_gpu_err_string = String::from_utf8_lossy(&c_format_bytes(
                "The GPU only has enough memory to load %d planes of data and, with current parameters, up to %d input planes are required to reconstruct a single plane\n",
                &[ci(max_gpu_plane), ci(max_needs[0])],
            ))
            .into_owned();
        } else if *if_3d_texture == 0 && max_tex_2d[1] / ny_gplane < max_needs[0] {
            self.m_gpu_err_string = String::from_utf8_lossy(&c_format_bytes(
                "With current parameters, up to %d input planes are required to reconstruct a single plane, while the allowed number of lines in GPU texture memory will fit only %d input planes\n",
                &[ci(max_needs[0]), ci(max_tex_2d[1] / ny_gplane)],
            ))
            .into_owned();
        }
        //
        // Limit maximum planes by the type allowed for the texture
        if *if_3d_texture > 0 {
            max_gpu_plane = b3d_i_min(&[max_gpu_plane, self.m_num_planes, max_tex_3d[2]]);
        } else if *if_3d_texture < 0 {
            max_gpu_plane = b3d_i_min(&[max_gpu_plane, self.m_num_planes, max_tex_layer[2]]);
        } else {
            max_gpu_plane =
                b3d_i_min(&[max_gpu_plane, self.m_num_planes, max_tex_2d[1] / ny_gplane]);
        }

        // FOR TESTING MEMORY SHIFTING ETC
        // ind = max(maxNeeds(1), min(ind, numPlanes / 3))
        self.m_num_gpu_planes = 0;
        // print *, ind, numPlanes, maxNeeds(1), maxGpuPlane
        let mut i = max_gpu_plane;
        while i >= max_needs[0] {
            err = gpu_alloc_arrays(
                self.m_iwidth,
                nygout,
                nx_gplane,
                ny_gplane,
                i,
                self.m_num_views,
                num_warps,
                num_delz,
                num_filts,
                0,
                if self.m_use_raw_stack != 0 {
                    self.m_ny_raw_padded
                } else {
                    0
                },
                if num_filts != 0 {
                    self.m_nx_filt_dim
                } else {
                    1
                },
                self.m_super_sample_fac * (if self.m_proj_super_fac > 1 { -1 } else { 1 }),
                self.m_super_samp_border,
                self.m_nx_out_pad,
                self.m_ny_out_pad,
                self.m_clean_super_fft,
                max_gpu_plane,
                max_needs[0],
                *if_3d_texture,
            );
            if err <= 0 {
                self.m_num_gpu_planes = i;
                self.m_load_gpu_start = 0;
                self.m_load_gpu_end = 0;
                break;
            }
            i -= 1;
        }
        self.m_use_gpu = self.m_num_gpu_planes > 0;
        self.m_do_raw_filter_on_gpu = err == 0;
        if !self.m_use_gpu && max_gpu_plane >= max_needs[0] {
            self.m_gpu_err_string =
                "Failed to allocate enough memory on the GPU to reconstruct even a single plane"
                    .to_string();
        }
    }

    /// `Tilt::warnOrExitIfNoGPU` (`tilt.cpp:5157`): issue desired warning or exit
    /// on error if GPU not available.
    fn warn_or_exit_if_no_gpu(&self) {
        if self.m_use_gpu {
            return;
        }
        if (self.m_if_gpu_by_environ != 0 && self.m_iact_gpu_fail_environ == 2)
            || (self.m_if_gpu_by_environ == 0 && self.m_iact_gpu_fail_option == 2)
        {
            printf!("\nERROR: TILT - %s\n", CArg::Str(&self.m_gpu_err_string));
            exit_error(b"Use of the GPU was requested but a GPU cannot be used");
        }
        printf!("%s\n", CArg::Str(&self.m_gpu_err_string));
        if (self.m_if_gpu_by_environ != 0 && self.m_iact_gpu_fail_environ == 1)
            || (self.m_if_gpu_by_environ == 0 && self.m_iact_gpu_fail_option == 1)
        {
            printf!("MESSAGE: Use of the GPU was requested but a GPU will not be used\n");
        }
        let _ = ImodFile::Stdout.flush();
    }

    /// `Tilt::packLocalData` (`tilt.cpp:5177`): load local data into a single
    /// array for upload to the GPU.  The allocation cannot fail here, so the
    /// source's failure branch is unreachable.
    fn pack_local_data(&mut self) -> Vec<f32> {
        let stride = (self.m_num_warp_pos * self.m_num_views) as usize;
        let mut pack_local = vec![0f32; 12 * stride];
        //
        // Pack data into one array
        for ipos in 0..self.m_num_warp_pos as usize {
            for iv in 0..self.m_num_views as usize {
                let i = (self.m_ind_warp[ipos] + self.m_map_used_view[iv] - 1) as usize;
                let j = ipos * self.m_num_views as usize + iv;
                pack_local[j] = self.m_fwarp[6 * i];
                pack_local[j + stride] = self.m_fwarp[1 + 6 * i];
                pack_local[j + stride * 2] = self.m_fwarp[2 + 6 * i];
                pack_local[j + stride * 3] = self.m_fwarp[3 + 6 * i];
                pack_local[j + stride * 4] = self.m_fwarp[4 + 6 * i];
                pack_local[j + stride * 5] = self.m_fwarp[5 + 6 * i];
                pack_local[j + stride * 6] = self.m_cwarp_alpha[i];
                pack_local[j + stride * 7] = self.m_swarp_alpha[i];
                pack_local[j + stride * 8] = self.m_cwarp_beta[i];
                pack_local[j + stride * 9] = self.m_swarp_beta[i];
                pack_local[j + stride * 10] = self.m_warp_xzfac[i];
                pack_local[j + stride * 11] = self.m_warp_yzfac[i];
            }
        }
        pack_local
    }

    /// `Tilt::parseCard` (`tilt.cpp:5213`): get a list from an input line,
    /// handling errors.
    fn parse_card(&self, card: &[u8], n_views: &mut i32, descrip: &str) -> Vec<i32> {
        let list = match parselist(&String::from_utf8_lossy(card)) {
            Ok(list) => list,
            Err(_) => exit_error_fmt!("Illegal entry for %s", CArg::Str(descrip)),
        };
        *n_views = list.len() as i32;
        if *n_views > self.m_lim_view {
            exit_error_fmt!("More views in %s than in input file", CArg::Str(descrip));
        }
        list
    }

    /// `Tilt::getValuesFromLines` (`tilt.cpp:5232`): read one or two values into
    /// array(s) using the general value reader function.  A `filename` opens its
    /// own stream; otherwise `fp` is read from where it stands.
    fn get_values_from_lines(
        &mut self,
        fp: Option<&mut ImodFile>,
        filename: Option<&[u8]>,
        values1: &mut [f32],
        values2: Option<&mut [f32]>,
        separate_lines: bool,
        num_values: &mut i32,
        descrip: &str,
    ) {
        let ierr: i32;
        let num_to_get = b3dabs!(*num_values);
        let mut own_fp: Option<ImodFile> = None;
        if let Some(name) = filename {
            match ImodFile::open(String::from_utf8_lossy(name).as_ref(), "r") {
                Some(f) => own_fp = Some(f),
                None => exit_error_fmt!(
                    "Opening file of %s, %s",
                    CArg::Str(descrip),
                    CArg::Bytes(name)
                ),
            }
        }
        let fp: &mut ImodFile = match own_fp.as_mut() {
            Some(f) => f,
            None => fp.expect("getValuesFromLines needs a stream or a file name"),
        };
        let flags = if separate_lines {
            RLFV_SEPARATE_LINES
        } else {
            0
        };
        if let Some(values2) = values2 {
            ierr = read_lines_for_values(
                fp,
                num_values,
                num_to_get,
                &mut self.m_line,
                MAX_LINE as i32,
                flags,
                "ff",
                &mut [
                    ReadValueArray::Floats(values1),
                    ReadValueArray::Floats(values2),
                ],
            );
        } else {
            ierr = read_lines_for_values(
                fp,
                num_values,
                num_to_get,
                &mut self.m_line,
                MAX_LINE as i32,
                flags,
                "f",
                &mut [ReadValueArray::Floats(values1)],
            );
        }
        if ierr != 0 {
            exit_from_value_read_error(ierr, descrip);
        }
    }

    /// `Tilt::setCenterCoords` (`tilt.cpp:5262`): set center of output plane and
    /// center of input for transformations.  Allow the full size to be less than
    /// the aligned stack, with a negative subset start.
    fn set_center_coords(
        &mut self,
        nx_full: i32,
        nx_full_in: i32,
        ny_full: i32,
        expanded_binning: f32,
        ix_subset: i32,
        ix_subset_in: i32,
        iy_subset: i32,
        x_offset: f32,
        xoff_adj: &mut f32,
    ) {
        self.m_xcen_in = (nx_full as f64 / 2. + 0.5 - ix_subset as f64) as f32;
        self.m_center_slice = (ny_full as f64 / 2. + 0.5 - iy_subset as f64) as f32;
        *xoff_adj = x_offset
            - ((b3dnint!(self.m_nx_proj as f32 * expanded_binning) - nx_full_in) / 2 + ix_subset_in)
                as f32
                / expanded_binning;
        self.m_xcen_out =
            ((self.m_iwidth / 2) as f64 + 0.5 + self.m_axis_xoffset as f64 + *xoff_adj as f64)
                as f32;
        self.m_ycen_out = ((self.m_ithick_bp / 2) as f64 + 0.5 + self.m_y_offset as f64) as f32;
    }

    /// `Tilt::getPaddedInputSize` (`tilt.cpp:5275`): get a padded size for given
    /// input size: 10% of X size or minimum of 16, max of 50.
    fn get_padded_input_size(&self, nx_proj: i32) -> i32 {
        let num_pad_tmp = b3dmin!(50, 2 * b3dmax!(8, nx_proj / 20));
        nice_frame(2 * ((nx_proj + num_pad_tmp) / 2), 2, nice_fft_limit())
    }

    /// `Tilt::lookupAngle` (`tilt.cpp:5285`): finds two nearest angles to
    /// `projAngle` and returns their indices `ind1` and `ind2` and an
    /// interpolation fraction `frac`.  THESE INDICES ARE 0-BASED.
    fn lookup_angle(
        &self,
        proj_angle: f32,
        angles: &[f32],
        num_views: i32,
        ind1: &mut i32,
        ind2: &mut i32,
        frac: &mut f32,
    ) {
        *ind1 = -1;
        *ind2 = -1;
        for i in 0..num_views {
            let a = angles[i as usize];
            if proj_angle >= a {
                if *ind1 < 0 {
                    *ind1 = i;
                }
                if a > angles[*ind1 as usize] {
                    *ind1 = i;
                }
            } else {
                if *ind2 < 0 {
                    *ind2 = i;
                }
                if a < angles[*ind2 as usize] {
                    *ind2 = i;
                }
            }
        }
        *frac = 0.;
        if *ind1 < 0 {
            *ind1 = *ind2;
        } else if *ind2 < 0 {
            *ind2 = *ind1;
        } else {
            *frac = (proj_angle - angles[*ind1 as usize])
                / (angles[*ind2 as usize] - angles[*ind1 as usize]);
        }
    }

    /// `Tilt::sampleForReport` (`tilt.cpp:5317`): compute mean and SD of interior
    /// of a slice for SIRT report.
    fn sample_for_report(
        &mut self,
        slice: &[f32],
        lslice: i32,
        kthick: i32,
        iteration: i32,
        samp_scale: f32,
        samp_add: f32,
    ) {
        let mut avg: f32 = 0.;
        let mut sd: f32 = 0.;
        let iskip: i32;
        let ix_low: i32;
        let iy_low: i32;
        let ierr: i32;
        //
        iskip = (self.m_islice_end - self.m_islice_start) / 10;
        if lslice < self.m_islice_start + iskip || lslice > self.m_islice_end - iskip {
            return;
        }
        ix_low = self.m_iwidth / 10;
        iy_low = kthick / 4;
        // `makeLinePointers(slice, mIwidth, kthick, sizeof(float))`
        let row = (self.m_iwidth * 4) as usize;
        let bytes: Vec<u8> = slice[..(self.m_iwidth * kthick) as usize]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let line_ptrs: Vec<&[u8]> = (0..kthick as usize)
            .map(|i| &bytes[i * row..(i + 1) * row])
            .collect();
        ierr = sample_mean_sd(
            Some(&line_ptrs),
            type_for_sample_mean(MRC_MODE_FLOAT),
            self.m_iwidth,
            kthick,
            0.02,
            ix_low,
            iy_low,
            self.m_iwidth - 2 * ix_low,
            kthick - 2 * iy_low,
            Some(&mut avg),
            Some(&mut sd),
        );
        if ierr != 0 {
            return;
        }
        let b = (3 * (iteration - 1)) as usize;
        self.m_report_vals[b] += (avg + samp_add) * samp_scale;
        self.m_report_vals[1 + b] += sd * samp_scale;
        self.m_report_vals[2 + b] += 1.;
    }

    /// `Tilt::localFactors` (`tilt.cpp:5347`): return indices (1-based) to four
    /// local areas, and fractions to apply for each at location `x`, `iy` in view
    /// `iv`, where `x` and `iy` are indexes in the reconstruction adjusted to
    /// match coordinates of projections.
    fn local_factors(
        &self,
        x: f32,
        iy: i32,
        iv: i32,
        ind1: &mut i32,
        ind2: &mut i32,
        ind3: &mut i32,
        ind4: &mut i32,
        f1: &mut f32,
        f2: &mut f32,
        f3: &mut f32,
        f4: &mut f32,
    ) {
        //
        let ix_pos: i32;
        let iyt: i32;
        let iy_pos: i32;
        let xt: f32;
        let fx: f32;
        let fy: f32;
        //
        xt = b3dmin!(
            b3dmax!((x - self.m_ix_start_warp as f32) as f64, 0.),
            ((self.m_nx_warp - 1) * self.m_idel_xwarp) as f64
        ) as f32;
        ix_pos = b3dmin!(
            xt / self.m_idel_xwarp as f32 + 1.,
            (self.m_nx_warp - 1) as f32
        ) as i32;
        fx = (xt - ((ix_pos - 1) * self.m_idel_xwarp) as f32) / self.m_idel_xwarp as f32;
        iyt = b3dmin!(
            b3dmax!(iy - self.m_iy_start_warp, 0),
            (self.m_ny_warp - 1) * self.m_idel_ywarp
        );
        iy_pos = b3dmin!(iyt / self.m_idel_ywarp + 1, self.m_ny_warp - 1);
        fy = (iyt - (iy_pos - 1) * self.m_idel_ywarp) as f32 / self.m_idel_ywarp as f32;

        *ind1 = self.m_ind_warp[(self.m_nx_warp * (iy_pos - 1) + ix_pos - 1) as usize] + iv;
        *ind2 = self.m_ind_warp[(self.m_nx_warp * (iy_pos - 1) + ix_pos) as usize] + iv;
        *ind3 = self.m_ind_warp[(self.m_nx_warp * iy_pos + ix_pos - 1) as usize] + iv;
        *ind4 = self.m_ind_warp[(self.m_nx_warp * iy_pos + ix_pos) as usize] + iv;
        *f1 = ((1. - fy as f64) * (1. - fx as f64)) as f32;
        *f2 = ((1. - fy as f64) * fx as f64) as f32;
        *f3 = (fy as f64 * (1. - fx as f64)) as f32;
        *f4 = fy * fx;
    }

    /// `Tilt::localProjFactors` (`tilt.cpp:5375`): compute local projection
    /// factors at a position in a column for view `iv`: `x` is the X index in the
    /// reconstruction, `lslice` is slice # in aligned stack.
    fn local_proj_factors(
        &self,
        x: f32,
        lslice: i32,
        iv: i32,
        xproj_fix: &mut f32,
        xproj_z: &mut f32,
        yproj_fix: &mut f32,
        yproj_z: &mut f32,
    ) {
        let (mut ind1, mut ind2, mut ind3, mut ind4) = (0i32, 0i32, 0i32, 0i32);
        let (mut f1, mut f2, mut f3, mut f4) = (0f32, 0f32, 0f32, 0f32);
        //
        // get transform and angle adjustment
        //
        let xc = x - self.m_xcen_out + self.m_xcen_in + self.m_axis_xoffset;
        self.local_factors(
            xc,
            lslice,
            self.m_map_used_view[(iv - 1) as usize],
            &mut ind1,
            &mut ind2,
            &mut ind3,
            &mut ind4,
            &mut f1,
            &mut f2,
            &mut f3,
            &mut f4,
        );
        //
        // get all the factors needed to compute a projection position
        // from the four local transforms
        //
        let ind1 = (ind1 - 1) as usize;
        let ind2 = (ind2 - 1) as usize;
        let ind3 = (ind3 - 1) as usize;
        let ind4 = (ind4 - 1) as usize;
        let fw_ind1 = 6 * ind1;
        let fw_ind2 = 6 * ind2;
        let fw_ind3 = 6 * ind3;
        let fw_ind4 = 6 * ind4;
        let fw = &self.m_fwarp;
        let xcen_in = self.m_xcen_in;
        let center_slice = self.m_center_slice;
        let cos_bet = self.m_cwarp_beta[ind1];
        let sin_bet = self.m_swarp_beta[ind1];
        let cos_alph = self.m_cwarp_alpha[ind1];
        let sin_alph = self.m_swarp_alpha[ind1];
        let a11 = fw[fw_ind1];
        let a12 = fw[2 + fw_ind1];
        let a21 = fw[1 + fw_ind1];
        let a22 = fw[3 + fw_ind1];
        let x_add = fw[4 + fw_ind1] + xcen_in - xcen_in * a11 - center_slice * a12;
        let y_add = fw[5 + fw_ind1] + center_slice - xcen_in * a21 - center_slice * a22;
        //
        let cos_bet2 = self.m_cwarp_beta[ind2];
        let sin_bet2 = self.m_swarp_beta[ind2];
        let cos_alph2 = self.m_cwarp_alpha[ind2];
        let sin_alph2 = self.m_swarp_alpha[ind2];
        let a112 = fw[fw_ind2];
        let a122 = fw[2 + fw_ind2];
        let a212 = fw[1 + fw_ind2];
        let a222 = fw[3 + fw_ind2];
        let x_add2 = fw[4 + fw_ind2] + xcen_in - xcen_in * a112 - center_slice * a122;
        let y_add2 = fw[5 + fw_ind2] + center_slice - xcen_in * a212 - center_slice * a222;
        //
        let cos_bet3 = self.m_cwarp_beta[ind3];
        let sin_bet3 = self.m_swarp_beta[ind3];
        let cos_alph3 = self.m_cwarp_alpha[ind3];
        let sin_alph3 = self.m_swarp_alpha[ind3];
        let a113 = fw[fw_ind3];
        let a123 = fw[2 + fw_ind3];
        let a213 = fw[1 + fw_ind3];
        let a223 = fw[3 + fw_ind3];
        let x_add3 = fw[4 + fw_ind3] + xcen_in - xcen_in * a113 - center_slice * a123;
        let y_add3 = fw[5 + fw_ind3] + center_slice - xcen_in * a213 - center_slice * a223;
        //
        let cos_bet4 = self.m_cwarp_beta[ind4];
        let sin_bet4 = self.m_swarp_beta[ind4];
        let cos_alph4 = self.m_cwarp_alpha[ind4];
        let sin_alph4 = self.m_swarp_alpha[ind4];
        let a114 = fw[fw_ind4];
        let a124 = fw[2 + fw_ind4];
        let a214 = fw[1 + fw_ind4];
        let a224 = fw[3 + fw_ind4];
        let x_add4 = fw[4 + fw_ind4] + xcen_in - xcen_in * a114 - center_slice * a124;
        let y_add4 = fw[5 + fw_ind4] + center_slice - xcen_in * a214 - center_slice * a224;
        //
        let f1x = f1 * a11;
        let f2x = f2 * a112;
        let f3x = f3 * a113;
        let f4x = f4 * a114;
        let f1xy = f1 * a12;
        let f2xy = f2 * a122;
        let f3xy = f3 * a123;
        let f4xy = f4 * a124;
        // fxfromy=f1*a12+f2*a122+f3*a123+f4*a124
        let f1y = f1 * a21;
        let f2y = f2 * a212;
        let f3y = f3 * a213;
        let f4y = f4 * a214;
        let f1yy = f1 * a22;
        let f2yy = f2 * a222;
        let f3yy = f3 * a223;
        let f4yy = f4 * a224;
        // fyfromy=f1*a22+f2*a222+f3*a223+f4*a224
        let x_all_add = f1 * x_add + f2 * x_add2 + f3 * x_add3 + f4 * x_add4;
        let y_all_add = f1 * y_add + f2 * y_add2 + f3 * y_add3 + f4 * y_add4;
        //
        // Each projection position is a sum of a fixed factor ("..f")
        // and a factor that multiplies z ("..z")
        //
        let xx = x - self.m_xcen_out;
        let yy = lslice as f32 - center_slice;
        let xa = self.m_axis_xoffset;
        let wxz = &self.m_warp_xzfac;
        let wyz = &self.m_warp_yzfac;
        let xp1f = xx * cos_bet + yy * sin_alph * sin_bet + xcen_in + xa;
        let xp1z = cos_alph * sin_bet + wxz[ind1];
        let xp2f = xx * cos_bet2 + yy * sin_alph2 * sin_bet2 + xcen_in + xa;
        let xp2z = cos_alph2 * sin_bet2 + wxz[ind2];
        let xp3f = xx * cos_bet3 + yy * sin_alph3 * sin_bet3 + xcen_in + xa;
        let xp3z = cos_alph3 * sin_bet3 + wxz[ind3];
        let xp4f = xx * cos_bet4 + yy * sin_alph4 * sin_bet4 + xcen_in + xa;
        let xp4z = cos_alph4 * sin_bet4 + wxz[ind4];

        let yp1f = yy * cos_alph + center_slice;
        let yp2f = yy * cos_alph2 + center_slice;
        let yp3f = yy * cos_alph3 + center_slice;
        let yp4f = yy * cos_alph4 + center_slice;
        //
        // store the fixed and z-dependent component of the
        // projection coordinates
        //
        *xproj_fix = f1x * xp1f
            + f2x * xp2f
            + f3x * xp3f
            + f4x * xp4f
            + f1xy * yp1f
            + f2xy * yp2f
            + f3xy * yp3f
            + f4xy * yp4f
            + x_all_add;
        *xproj_z = f1x * xp1z + f2x * xp2z + f3x * xp3z + f4x * xp4z
            - (f1xy * (sin_alph - wyz[ind1])
                + f2xy * (sin_alph2 - wyz[ind2])
                + f3xy * (sin_alph3 - wyz[ind3])
                + f4xy * (sin_alph4 - wyz[ind4]));
        *yproj_fix = f1y * xp1f
            + f2y * xp2f
            + f3y * xp3f
            + f4y * xp4f
            + f1yy * yp1f
            + f2yy * yp2f
            + f3yy * yp3f
            + f4yy * yp4f
            + y_all_add;
        *yproj_z = f1y * xp1z + f2y * xp2z + f3y * xp3z + f4y * xp4z
            - (f1yy * (sin_alph - wyz[ind1])
                + f2yy * (sin_alph2 - wyz[ind2])
                + f3yy * (sin_alph3 - wyz[ind3])
                + f4yy * (sin_alph4 - wyz[ind4]));
    }

    /// `Tilt::findProjectingPoint` (`tilt.cpp:5510`): finds the point at centered
    /// Z coordinate `zz` projecting to `xproj`, `yproj` in view `iv` of original
    /// projections.  `xx` is X index in reconstruction, `yy` is slice number in
    /// original projections.
    fn find_projecting_point(
        &self,
        xproj: f32,
        yproj: f32,
        zz: f32,
        iv: &mut i32,
        xx: &mut f32,
        yy: &mut f32,
    ) {
        let mut iter: i32;
        let mut if_done: i32;
        let mut ix_assay: i32;
        let mut iy_assay: i32;
        let (mut xproj_fix11, mut xproj_z11, mut yproj_fix11, mut yproj_z11) =
            (0f32, 0f32, 0f32, 0f32);
        let (mut xproj_fix21, mut xproj_z21, mut yproj_fix21, mut yproj_z21) =
            (0f32, 0f32, 0f32, 0f32);
        let (mut xproj_fix12, mut xproj_z12, mut yproj_fix12, mut yproj_z12) =
            (0f32, 0f32, 0f32, 0f32);
        let (mut xp11, mut yp11, mut xp12, mut yp12, mut xp21, mut yp21): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (mut xerr, mut yerr, mut dxpx, mut dxpy, mut dypx, mut dypy): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (mut fx, mut fy, mut den): (f32, f32, f32);

        iter = 1;
        if_done = 0;
        while if_done == 0 && iter <= 5 {
            ix_assay = xx.floor() as i32;
            iy_assay = yy.floor() as i32;
            self.local_proj_factors(
                ix_assay as f32,
                iy_assay,
                *iv,
                &mut xproj_fix11,
                &mut xproj_z11,
                &mut yproj_fix11,
                &mut yproj_z11,
            );
            self.local_proj_factors(
                (ix_assay + 1) as f32,
                iy_assay,
                *iv,
                &mut xproj_fix21,
                &mut xproj_z21,
                &mut yproj_fix21,
                &mut yproj_z21,
            );
            self.local_proj_factors(
                ix_assay as f32,
                iy_assay + 1,
                *iv,
                &mut xproj_fix12,
                &mut xproj_z12,
                &mut yproj_fix12,
                &mut yproj_z12,
            );
            xp11 = xproj_fix11 + xproj_z11 * zz;
            yp11 = yproj_fix11 + yproj_z11 * zz;
            xp21 = xproj_fix21 + xproj_z21 * zz;
            yp21 = yproj_fix21 + yproj_z21 * zz;
            xp12 = xproj_fix12 + xproj_z12 * zz;
            yp12 = yproj_fix12 + yproj_z12 * zz;
            xerr = xproj - xp11;
            yerr = yproj - yp11;
            dxpx = xp21 - xp11;
            dxpy = xp12 - xp11;
            dypx = yp21 - yp11;
            dypy = yp12 - yp11;
            den = dxpx * dypy - dxpy * dypx;
            fx = (xerr * dypy - yerr * dxpy) / den;
            fy = (dxpx * yerr - dypx * xerr) / den;
            *xx = ix_assay as f32 + fx;
            *yy = iy_assay as f32 + fy;
            if fx as f64 > -0.1 && (fx as f64) < 1.1 && fy as f64 > -0.1 && (fy as f64) < 1.1 {
                if_done = 1;
            }
            iter = iter + 1;
        }
    }

    /// `Tilt::setCosStretch` (`tilt.cpp:5555`): compute space needed for cosine
    /// stretched data.
    fn set_cos_stretch(&mut self) {
        let mut lslice_min: i32;
        let mut lslice_max: i32;
        let mut tan_alph: f32 = 0.;
        let mut xp_max: f32;
        let mut xp_min: f32;
        let mut zz: f32;
        let mut z_part: f32;
        let mut yy: f32;
        let mut xproj: f32;
        // make the indexes be bases, numbered from 0
        //
        self.m_ind_stretch_line[0] = 0;
        lslice_min = b3dmin!(self.m_islice_end, self.m_islice_start);
        lslice_max = b3dmax!(self.m_islice_end, self.m_islice_start);
        if self.m_if_alpha < 0 {
            //
            // New-style X tilting: SET MINIMUM NUMBER OF INPUT SLICES HERE
            //
            let abs_sin = b3dabs!(self.m_sin_alpha[0]);
            lslice_min = ((self.m_center_slice
                + (lslice_min as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                + self.m_y_offset * self.m_sin_alpha[0]) as f64
                - 0.5 * self.m_ithick_out as f64 * abs_sin as f64
                - 1.) as i32;
            lslice_max = ((self.m_center_slice
                + (lslice_max as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                + self.m_y_offset * self.m_sin_alpha[0]) as f64
                + 0.5 * self.m_ithick_out as f64 * abs_sin as f64
                + 2.) as i32;
            tan_alph = self.m_sin_alpha[0] / self.m_cos_alpha[0];
            lslice_min = b3dmax!(1, lslice_min);
            lslice_max = b3dmin!(lslice_max, self.m_ny_proj);
        }

        // iv can be indexed from zero, it is used only as a subscript
        for iv in 0..self.m_num_views as usize {
            xp_max = 1.;
            xp_min = self.m_nx_proj as f32;
            //
            // find min and max position of 8 corners of reconstruction
            //
            let mut ix = 1;
            while ix <= self.m_iwidth {
                let mut iy = 1;
                while iy <= self.m_ithick_bp {
                    let mut lslice = lslice_min;
                    while lslice <= lslice_max {
                        zz = (iy as f32 - self.m_ycen_out) * self.m_compress[iv];
                        if self.m_if_alpha < 0 {
                            zz = self.m_compress[iv]
                                * (iy as f32
                                    - (self.m_ycen_out
                                        - b3dnint!(tan_alph * (lslice as f32 - self.m_center_slice))
                                            as f32));
                        }
                        if self.m_if_alpha <= 0 {
                            z_part =
                                zz * self.m_sin_beta[iv] + self.m_xcen_in + self.m_axis_xoffset;
                        } else {
                            yy = lslice as f32 - self.m_center_slice;
                            z_part = yy * self.m_sin_alpha[iv] * self.m_sin_beta[iv]
                                + zz * (self.m_cos_alpha[iv] * self.m_sin_beta[iv]
                                    + self.m_xzfac[iv])
                                + self.m_xcen_in
                                + self.m_axis_xoffset;
                        }
                        xproj = z_part + (ix as f32 - self.m_xcen_out) * self.m_cos_beta[iv];
                        xp_min = b3dmax!(1., b3dmin!(xp_min, xproj) as f64) as f32;
                        xp_max = b3dmin!(self.m_nx_proj as f32, b3dmax!(xp_max, xproj));
                        lslice += b3dmax!(1, lslice_max - lslice_min);
                    }
                    iy += self.m_ithick_bp - 1;
                }
                ix += self.m_iwidth - 1;
            }
            // print *,iv, xpmin, xpmax
            //
            // set up extent and offset of stretches
            //
            self.m_stretch_offset[iv] = ((xp_min / self.m_cos_beta[iv]) as f64
                - 1. / self.m_interp_fac_stretch as f64)
                as f32;
            self.m_nx_stretched[iv] = ((self.m_interp_fac_stretch as f32 * (xp_max - xp_min)
                / self.m_cos_beta[iv]) as f64
                + 2.) as i32;
            self.m_ind_stretch_line[iv + 1] = self.m_ind_stretch_line[iv] + self.m_nx_stretched[iv];
            // print *,iv, xpmin, xpmax, stretchOffset[iv], nxStretched[iv], indStretchLine[iv]
        }
    }

    /// `Tilt::setNeededSlices` (`tilt.cpp:5621`): determine starting and ending
    /// input slice needed to reconstruct each output slice, as well as the maximum
    /// needed over all slices for a series of numbers of output slices up to
    /// `numEval`.
    fn set_needed_slices(&mut self, max_needs: &mut [i32], num_eval: i32) {
        let mut lslice_min: i32;
        let mut lslice_max: i32;
        let mut nx_assay: i32;
        let mut min_slice: i32;
        let mut ix_assay: i32;
        let mut max_slice: i32;
        let mut ix_sample: i32;
        let mut iyp: i32;
        let mut dx_assay: f32;
        let mut dx_temp: f32;
        let mut xx: f32;
        let mut yy: f32;
        let mut zz: f32;
        let mut xp: f32;
        let mut yp: f32;
        let (mut xproj_fix, mut xproj_z, mut yproj_fix, mut yproj_z) = (0f32, 0f32, 0f32, 0f32);
        let mut xproj: f32;
        let mut yproj: f32;
        lslice_min = self.m_islice_start;
        lslice_max = self.m_islice_end;
        if self.m_if_alpha < 0 {
            let abs_sin = b3dabs!(self.m_sin_alpha[0]);
            lslice_min = ((self.m_center_slice
                + (self.m_islice_start as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                + self.m_y_offset * self.m_sin_alpha[0]) as f64
                - 0.5 * self.m_ithick_out as f64 * abs_sin as f64
                - 1.) as i32;
            lslice_max = ((self.m_center_slice
                + (self.m_islice_end as f32 - self.m_center_slice) * self.m_cos_alpha[0]
                + self.m_y_offset * self.m_sin_alpha[0]) as f64
                + 0.5 * self.m_ithick_out as f64 * abs_sin as f64
                + 2.) as i32;
            lslice_min = b3dmax!(1, lslice_min);
            lslice_max = b3dmin!(lslice_max, self.m_ny_proj);
        }
        self.m_ind_needed_base = lslice_min - 1;
        self.m_num_need_se = lslice_max - self.m_ind_needed_base;
        if self.m_needed_starts.is_empty() {
            self.m_needed_starts = vec![0; self.m_num_need_se as usize];
            self.m_needed_ends = vec![0; self.m_num_need_se as usize];
        }

        for itry in lslice_min..=lslice_max {
            let idx = (itry - self.m_ind_needed_base - 1) as usize;
            if self.m_if_alpha <= 0 && self.m_nx_warp == 0 {
                //
                // regular case is simple: just need the current slice
                //
                self.m_needed_starts[idx] = itry;
                self.m_needed_ends[idx] = itry;
            } else {
                //
                // for old-style X-tilt or local alignment, determine what
                // slices are needed by sampling
                // set up sample points: left and right if no warp,
                // or half the warp spacing
                //
                if self.m_nx_warp == 0 {
                    nx_assay = 2;
                    dx_assay = (self.m_iwidth - 1) as f32;
                } else {
                    dx_temp = (self.m_idel_xwarp / 2) as f32;
                    nx_assay = b3dmax!(2., (self.m_iwidth as f32 / dx_temp) as f64 + 1.) as i32;
                    dx_assay = ((self.m_iwidth as f64 - 1.) / (nx_assay as f64 - 1.)) as f32;
                }
                //
                // sample top and bottom at each position
                //
                min_slice = self.m_ny_proj + 1;
                max_slice = 0;
                for iassay in 1..=nx_assay {
                    ix_assay = b3dnint!(1. + (iassay - 1) as f32 * dx_assay);
                    for iv in 1..=self.m_num_views {
                        let ivm1 = (iv - 1) as usize;
                        if !self.m_rec_reproj {
                            ix_sample = b3dnint!(
                                ix_assay as f32 - self.m_xcen_out
                                    + self.m_xcen_in
                                    + self.m_axis_xoffset
                            );
                            if self.m_nx_warp != 0 {
                                self.local_proj_factors(
                                    ix_assay as f32,
                                    itry,
                                    iv,
                                    &mut xproj_fix,
                                    &mut xproj_z,
                                    &mut yproj_fix,
                                    &mut yproj_z,
                                );
                            }
                            let mut iy = 1;
                            while iy <= self.m_ithick_bp {
                                //
                                // for each position, find back-projection location
                                // transform if necessary, and use to get min and
                                // max slices needed to get this position
                                //
                                xx = ix_sample as f32 - self.m_xcen_out;
                                yy = itry as f32 - self.m_center_slice;
                                zz = iy as f32 - self.m_ycen_out;
                                xp = xx * self.m_cos_beta[ivm1]
                                    + yy * self.m_sin_alpha[ivm1] * self.m_sin_beta[ivm1]
                                    + zz * (self.m_cos_alpha[ivm1] * self.m_sin_beta[ivm1]
                                        + self.m_xzfac[ivm1])
                                    + self.m_xcen_in
                                    + self.m_axis_xoffset;
                                yp = yy * self.m_cos_alpha[ivm1]
                                    - zz * (self.m_sin_alpha[ivm1] - self.m_yzfac[ivm1])
                                    + self.m_center_slice;
                                if self.m_nx_warp != 0 {
                                    xp = xproj_fix + xproj_z * zz;
                                    yp = yproj_fix + yproj_z * zz;
                                }
                                let _ = xp;
                                iyp = b3dmax!(1., yp as f64) as i32;
                                min_slice = b3dmin!(min_slice, iyp);
                                max_slice = b3dmax!(max_slice, b3dmin!(self.m_ny_proj, iyp + 1));
                                // if (debug) print *,xx, yy, zz, iyp, minslice, maxslice
                                iy += self.m_ithick_bp - 1;
                            }
                        } else {
                            //
                            // Projections: get Y coordinate in original projection
                            // if local, get the X coordinate in reconstruction too
                            // then get the refinement
                            xproj = ix_assay as f32 + self.m_xproj_offset;
                            yproj = itry as f32 + self.m_yproj_offset;
                            let mut iy = 1;
                            while iy <= self.m_ithick_reproj {
                                zz = (iy + self.m_min_yreproj - 1) as f32 - self.m_ycen_out;
                                yy = (yproj + zz * (self.m_sin_alpha[ivm1] - self.m_yzfac[ivm1])
                                    - self.m_center_slice)
                                    / self.m_cos_alpha[ivm1]
                                    + self.m_center_slice;
                                if self.m_nx_warp != 0 {
                                    xx = (xproj
                                        - yy * self.m_sin_alpha[ivm1] * self.m_sin_beta[ivm1]
                                        - zz * (self.m_cos_alpha[ivm1] * self.m_sin_beta[ivm1]
                                            + self.m_xzfac[ivm1])
                                        - self.m_xcen_in
                                        - self.m_axis_xoffset)
                                        / self.m_cos_beta[ivm1]
                                        + self.m_xcen_out;
                                    let mut ivr = iv;
                                    self.find_projecting_point(
                                        xproj, yproj, zz, &mut ivr, &mut xx, &mut yy,
                                    );
                                }
                                iyp = b3dmax!(1., (yy - self.m_yproj_offset) as f64) as i32;
                                min_slice = b3dmin!(min_slice, iyp);
                                max_slice = b3dmax!(max_slice, b3dmin!(self.m_ny_proj, iyp + 1));
                                iy += self.m_ithick_reproj - 1;
                            }
                        }
                    }
                }
                //
                // set up starts and ends
                //
                self.m_needed_starts[idx] = b3dmax!(1, min_slice);
                self.m_needed_ends[idx] = b3dmin!(self.m_ny_proj, max_slice);
            }
        }
        //
        // Count maximum # of slices needed for number of slices to be computed
        for iv in 0..num_eval {
            max_needs[iv as usize] = 0;
            for iy in lslice_min..=lslice_max - iv {
                max_slice = self.m_needed_ends[(iy + iv - self.m_ind_needed_base - 1) as usize] + 1
                    - self.m_needed_starts[(iy - self.m_ind_needed_base - 1) as usize];
                max_needs[iv as usize] = b3dmax!(max_needs[iv as usize], max_slice);
            }
        }
    }

    /// `Tilt::allocateArray` (`tilt.cpp:5764`): allocate main data array, trying
    /// to get enough to do `numEval` slices without reloading any data, based on
    /// the numbers in `maxNeeds`, and trying fewer slices down to `minLoad` if that
    /// fails.  `minMemory` is the minimum amount it will allocate.  The
    /// `ALLOC_IF_NEEDED` macro (`tilt.cpp:5751`) is expanded in place.
    fn allocate_array(
        &mut self,
        max_needs: &[i32],
        num_eval: i32,
        min_load: i32,
        min_memory: i32,
    ) -> i32 {
        let ierr: i32;
        let mut pad: i32;
        let mut ny_pad: i32;
        let mut num_planes: i32;
        let mut num_out_buf: i32 = 0;
        let max_in_planes: i32 =
            self.m_needed_ends[(self.m_num_need_se - 1) as usize] + 1 - self.m_needed_starts[0];
        let max_num_out_buf: i32 = self.m_islice_end + 1 - self.m_islice_start;
        let mut max_out_buf_need: i64 = 0;
        let max_input_need: i64;
        let mut mem_need: i64;
        let min_need: i64;
        let mut base_need: i64;
        let mut input_need: i64;
        let whole_need: i64;
        let mut out_buf_need: i64 = 0;
        let alloc_load = !self.m_rec_reproj;
        let make_out_buf = self.m_perpendicular == 0 && self.m_reproj_bp == 0;
        let out_slice: i64 = (self.m_ithick_out * self.m_iwidth) as i64;

        // Basic need for defined planes, plus estimate for load buffer
        base_need = self.m_need_for_filt_arr as i64
            + self.m_need_for_out_arr as i64
            + self.m_need_for_vert_arr
            + self.m_need_for_read_in_arr
            + self.m_need_for_work_arr as i64
            + self.m_need_for_super_arr
            + 32;
        if alloc_load {
            base_need += (max_needs[(num_eval - 1) as usize] * self.m_nx_proj) as i64;
        }
        if self.m_use_raw_stack != 0 {
            base_need += (2 * self.m_nx_pad_dim * max_in_planes) as i64;
        }

        // Maximum possible needs for full buffering
        max_input_need =
            self.m_in_plane_size as i64 * max_in_planes as i64 + self.m_ip_extra_size as i64;
        if make_out_buf {
            max_out_buf_need = out_slice * max_num_out_buf as i64;
        }
        whole_need = base_need + max_input_need + max_out_buf_need;

        // Limit that by the allowed minimum memory, and allocate base
        min_need = b3dmin!(whole_need, min_memory as i64);
        ierr = 0;
        if ierr == 0 && self.m_need_for_filt_arr != 0 {
            self.m_filter_array = vec![0.; self.m_need_for_filt_arr as usize];
        }
        if ierr == 0 && self.m_need_for_out_arr != 0 {
            self.m_out_slice_arr = vec![0.; self.m_need_for_out_arr as usize];
        }
        if ierr == 0 && self.m_need_for_vert_arr != 0 {
            self.m_vert_slice_arr = vec![0.; self.m_need_for_vert_arr as usize];
        }
        if ierr == 0 && self.m_need_for_read_in_arr != 0 {
            self.m_read_in_array = vec![0.; self.m_need_for_read_in_arr as usize];
        }
        if ierr == 0 && self.m_need_for_work_arr != 0 {
            self.m_work_array = vec![0.; self.m_need_for_work_arr as usize];
        }
        if ierr == 0 && self.m_need_for_super_arr != 0 {
            self.m_super_out_arr = vec![0.; self.m_need_for_super_arr as usize];
        }
        if ierr == 0 {
            // Loop on trials of number sof loaded slices
            let mut eval = num_eval;
            while eval >= b3dmin!(num_eval, min_load) {
                // Input need with that number of slices, plus half as much for line
                // output buffer
                input_need = self.m_in_plane_size as i64 * max_needs[(eval - 1) as usize] as i64
                    + self.m_ip_extra_size as i64;
                if make_out_buf {
                    out_buf_need = b3dmin!(input_need / 2, max_out_buf_need);
                }
                mem_need = base_need + input_need + out_buf_need;

                // But then allow it to go higher to use the minimum memory
                mem_need = b3dmax!(mem_need, min_need);
                input_need = mem_need - base_need;
                if make_out_buf {
                    // Now if doing output buffer, get the number of planes that allows,
                    // and set up output buffer to use either a third of that or all of
                    // what is left
                    num_planes = b3dmin!(
                        max_in_planes as i64,
                        (input_need - self.m_ip_extra_size as i64) / self.m_in_plane_size as i64
                    ) as i32;
                    out_buf_need = b3dmax!(
                        self.m_in_plane_size as i64
                            * (num_planes - max_needs[min_load as usize]) as i64
                            / 3,
                        input_need - self.m_in_plane_size as i64 * num_planes as i64
                    );

                    // Limit it, forget about having only one line, set true out buf need
                    out_buf_need = if out_buf_need < max_out_buf_need {
                        out_buf_need
                    } else {
                        max_out_buf_need
                    };
                    num_out_buf = (out_buf_need / out_slice) as i32;
                    if num_out_buf <= 1 {
                        num_out_buf = 0;
                    }
                    out_buf_need = out_slice * num_out_buf as i64;

                    // Adjust input need down by that loss
                    input_need -= out_buf_need;
                }

                // Limit input need here then see if it works
                input_need = if input_need < max_input_need {
                    input_need
                } else {
                    max_input_need
                };
                if input_need + out_buf_need < 2147000000 {
                    self.m_input_array = vec![0.; input_need as usize];
                    if out_buf_need != 0 {
                        self.m_out_buffer = vec![0.; out_buf_need as usize];
                    }
                    self.m_max_stack = base_need + input_need + out_buf_need;
                    self.m_num_planes = ((input_need - self.m_ip_extra_size as i64)
                        / self.m_in_plane_size as i64)
                        as i32;
                    self.m_num_out_buf_slices = num_out_buf;
                    if alloc_load {
                        self.m_load_buffer =
                            vec![0.; (self.m_num_planes * self.m_nx_proj) as usize];
                    }

                    // Compute needed size for raw padded image, applying a minimum
                    // aspect ratio
                    if self.m_use_raw_stack != 0 {
                        pad = b3dmax!(8., 0.05 * self.m_num_planes as f64) as i32;
                        ny_pad = b3dmax!(
                            (self.m_num_planes + 2 * pad) as f32,
                            b3dmin!(
                                (self.m_ny_proj + pad) as f32,
                                self.m_min_raw_pad_aspect * self.m_nx_pad_dim as f32
                            )
                        ) as i32;
                        self.m_ny_raw_padded = nice_frame(ny_pad, 2, nice_fft_limit());
                        self.m_raw_filt_map =
                            vec![0.; (self.m_ny_raw_padded * self.m_nx_pad_dim / 2) as usize];
                        self.m_rot_buffer =
                            vec![0.; (self.m_ny_raw_padded * self.m_nx_pad_dim) as usize];
                    }
                    printf!(
                        "\nAllocated %d MB for main arrays\n",
                        ci(b3dnint!(self.m_max_stack as f64 / (1024. * 256.)))
                    );
                    return eval;
                }
                eval -= 1;
            }
        }
        self.m_filter_array = Vec::new();
        self.m_out_slice_arr = Vec::new();
        self.m_vert_slice_arr = Vec::new();
        self.m_read_in_array = Vec::new();
        self.m_work_array = Vec::new();
        self.m_super_out_arr = Vec::new();
        self.m_input_array = Vec::new();
        0
    }

    /// `Tilt::setupSizesForSupersampling` (`tilt.cpp:5884`): determines the sizes
    /// for the super-sampled slice array and the regular slice array when
    /// super-sampling, including the size that would be needed on the GPU.
    fn setup_sizes_for_supersampling(&mut self) {
        let nice_gpu_limit = 5;
        if self.m_super_sample_fac > 1 {
            self.m_nx_super_samp = self.m_iwidth + 2 * self.m_super_samp_border;

            // This padding makes no difference, no tapering is good enough to get rid
            // of the ringing on the edge after reduction, hence the border is added
            self.m_nx_out_pad = (2.
                * ((self.m_nx_super_samp as f64 + b3dmax!(16., 0.01 * self.m_iwidth as f64) + 1.)
                    / 2.)) as i32;
            self.m_nx_gpu_crop_pad = nice_frame(self.m_nx_out_pad, 2, nice_gpu_limit);
            self.m_nx_out_pad = nice_frame(self.m_nx_out_pad, 2, nice_fft_limit());
            self.m_ny_super_samp = self.m_ithick_bp + 2 * self.m_super_samp_border;
            self.m_ny_out_pad = (2.
                * ((self.m_ny_super_samp as f64
                    + b3dmax!(16., 0.01 * self.m_ithick_bp as f64)
                    + 1.)
                    / 2.)) as i32;
            self.m_ny_gpu_crop_pad = nice_frame(self.m_ny_out_pad, 2, nice_gpu_limit);
            self.m_ny_out_pad = nice_frame(self.m_ny_out_pad, 2, nice_fft_limit());
            self.m_need_for_out_arr = (self.m_nx_out_pad + 2) * self.m_ny_out_pad;
            self.m_need_for_super_arr = (self.m_nx_out_pad * self.m_super_sample_fac + 2) as i64
                * self.m_ny_out_pad as i64
                * self.m_super_sample_fac as i64;
            self.m_in_plane_size =
                (self.m_proj_super_fac * (self.m_nx_proj + self.m_num_pad) + 2) * self.m_num_views;
        }
    }
}

impl Tilt {
    /// `Tilt::reProject` (`tilt.cpp:5909`): the old reprojection from a single
    /// slice that matches xyzproj output.
    fn re_project(
        &self,
        array: &[f32],
        nxs: i32,
        _nys: i32,
        nx_out: i32,
        sin_angle: f32,
        cos_angle: f32,
        xray_start: &[f32],
        yray_start: &[f32],
        num_pix_in_ray: &[i32],
        max_ray_pixels: i32,
        fill: f32,
        proj_line: &mut [f32],
        linear: i32,
        no_scale: i32,
    ) {
        let mut ixr: i32;
        let mut iyr: i32;
        let mut num_ray_pts: i32;
        let mut idir: i32;
        let mut ray_fac: f32;
        let mut ray_add: f32;
        let mut x_ray: f32;
        let mut y_ray: f32;
        let mut pix_temp: f32;
        let mut full_fill: f32;
        let (mut dx, mut dy, mut v2, mut v4, mut v6, mut v8, mut v5): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (mut a, mut b, mut c, mut d): (f32, f32, f32, f32);
        let at = |i: i32| array[i as usize];
        //
        ray_fac = (1. / max_ray_pixels as f64) as f32;
        full_fill = fill;
        if no_scale != 0 {
            ray_fac = 1.;
            full_fill = fill * max_ray_pixels as f32;
        }

        // ixOut used only as an index: loop from 0
        for ix_out in 0..nx_out as usize {
            proj_line[ix_out] = full_fill;
            num_ray_pts = num_pix_in_ray[ix_out];
            if num_ray_pts > 0 {
                pix_temp = 0.;
                if sin_angle != 0. {
                    if linear == 0 {
                        for iray in 0..=num_ray_pts - 1 {
                            x_ray = xray_start[ix_out] + iray as f32 * sin_angle;
                            y_ray = yray_start[ix_out] + iray as f32 * cos_angle;
                            ixr = b3dnint!(x_ray);
                            iyr = b3dnint!(y_ray);
                            dx = x_ray - ixr as f32;
                            dy = y_ray - iyr as f32;
                            v2 = at(ixr - 1 + nxs * (iyr - 2));
                            v4 = at(ixr - 2 + nxs * (iyr - 1));
                            v5 = at(ixr - 1 + nxs * (iyr - 1));
                            v6 = at(ixr + nxs * (iyr - 1));
                            v8 = at(ixr - 1 + nxs * iyr);
                            //
                            a = ((v6 + v4) as f64 * 0.5 - v5 as f64) as f32;
                            b = ((v8 + v2) as f64 * 0.5 - v5 as f64) as f32;
                            c = ((v6 - v4) as f64 * 0.5) as f32;
                            d = ((v8 - v2) as f64 * 0.5) as f32;
                            pix_temp = pix_temp + a * dx * dx + b * dy * dy + c * dx + d * dy + v5;
                        }
                    } else {
                        for iray in 0..=num_ray_pts - 1 {
                            x_ray = xray_start[ix_out] + iray as f32 * sin_angle;
                            y_ray = yray_start[ix_out] + iray as f32 * cos_angle;
                            ixr = x_ray as i32;
                            iyr = y_ray as i32;
                            dx = x_ray - ixr as f32;
                            dy = y_ray - iyr as f32;
                            pix_temp = (pix_temp as f64
                                + (1. - dy) as f64
                                    * ((1. - dx as f64) * at(ixr - 1 + nxs * (iyr - 1)) as f64
                                        + (dx * at(ixr + nxs * (iyr - 1))) as f64)
                                + dy as f64
                                    * ((1. - dx as f64) * at(ixr - 1 + nxs * iyr) as f64
                                        + (dx * at(ixr + nxs * iyr)) as f64))
                                as f32;
                        }
                    }
                } else {
                    //
                    // vertical projection
                    //
                    ixr = b3dnint!(xray_start[ix_out]);
                    iyr = b3dnint!(yray_start[ix_out]);
                    idir = b3dsign!(1f64, cos_angle) as i32;
                    for iray in 0..=num_ray_pts - 1 {
                        pix_temp = pix_temp + at(ixr - 1 + nxs * (iyr + idir * iray - 1));
                    }
                }

                ray_add = ray_fac * (max_ray_pixels - num_ray_pts) as f32 * fill;
                proj_line[ix_out] = ray_fac * pix_temp + ray_add;
            }
        }
    }

    /// `Tilt::reprojDelZ` (`tilt.cpp:5988`): computes the change in Z that moves
    /// by 1 pixel along a projection ray given the sines and cosines of alpha and
    /// beta and the z factors.  The parameters shadow `mXZfac`/`mYZfac`.
    fn reproj_del_z(
        &self,
        sin_bet: f32,
        cos_bet: f32,
        sin_alph: f32,
        cos_alph: f32,
        m_xzfac: f32,
        m_yzfac: f32,
    ) -> f32 {
        let dy_fac = (sin_alph - m_yzfac) / cos_alph;
        let big_fac = (dy_fac * sin_alph * sin_bet + cos_alph * sin_bet + m_xzfac) / cos_bet;
        1.0f32 / ((1. + (dy_fac * dy_fac) as f64 + (big_fac * big_fac) as f64) as f32).sqrt()
    }

    /// `Tilt::reprojectRec` (`tilt.cpp:6003`): reprojects slices from
    /// `lsliceStart` to `lsliceEnd` and writes the reprojections.  Fewer slices may
    /// be done if on the GPU, and the ending slice done is returned in
    /// `lsliceEnd`.
    fn reproject_rec(
        &mut self,
        lslice_start: i32,
        lslice_end: &mut i32,
        in_load_start: i32,
        in_load_end: i32,
        dmin: &mut f32,
        dmax: &mut f32,
        dtot8: &mut f64,
    ) {
        let mut ix: i32;
        let mut iy: i32;
        let mut iz: i32;
        let mut iys: i32;
        let mut ind: i32;
        let mut cos_alph: f32;
        let mut sin_alph: f32;
        let mut cos_bet: f32;
        let mut sin_bet: f32;
        let mut del_z: f32;
        let mut fz: f32;
        let mut one_mfz: f32;
        let mut zz: f32;
        let mut xx: f32;
        let mut fx: f32;
        let mut one_mfx: f32;
        let mut yy: f32;
        let mut fy: f32;
        let mut one_mfy: f32;
        let mut xproj: f32;
        let mut yproj: f32;
        let (mut d11, mut d12, mut d21, mut d22): (f32, f32, f32, f32);
        let y_end_tol: f32;
        let mut xproj_min: f32;
        let mut xproj_max: f32;
        let x_jump: f32;
        let mut z_jump: f32;
        let mut ind_base: i32 = 0;
        let nx_load: i32;
        let mut last_zdone: i32;
        let mut ind_jump: i32;
        let mut lgpu_end: i32;
        let mut line_base: i32;
        let ycen_adj: f32;
        let mut tmp_cen_in: f32;
        let mut sum: f64;
        let mut wall_start: f64;
        let mut wall_cumul: f64;
        let mut try_jump: bool;

        y_end_tol = 3.05;
        x_jump = 5.0;
        nx_load = self.m_max_xload + 1 - self.m_min_xload;
        wall_start = wall_time();
        wall_cumul = 0.;
        ycen_adj = self.m_ycen_out - (self.m_min_yreproj - 1) as f32;
        //
        if self.m_use_gpu && self.m_load_gpu_start > 0 {
            ind_jump = 1;
            //
            // GPU REPROJECTION: Find last slice that can be done
            lgpu_end = *lslice_end;
            while lgpu_end >= lslice_start {
                if self.m_needed_ends[(lgpu_end - self.m_ind_needed_base - 1) as usize]
                    <= self.m_load_gpu_end
                {
                    ind_jump = 0;
                    break;
                }
                lgpu_end -= 1;
            }
            if ind_jump == 0 {
                //
                // Loop on views; do non-local case first
                for iv in 1..=self.m_num_views {
                    let ivu = (iv - 1) as usize;
                    if self.m_nx_warp == 0 {
                        del_z = self.reproj_del_z(
                            self.m_sin_beta[ivu],
                            self.m_cos_beta[ivu],
                            self.m_sin_alpha[ivu],
                            self.m_cos_alpha[ivu],
                            self.m_xzfac[ivu],
                            self.m_yzfac[ivu],
                        );
                        tmp_cen_in = self.m_xcen_in + self.m_axis_xoffset;
                        ind_jump = gpu_reproject(
                            &mut self.m_reproj_lines,
                            self.m_sin_beta[ivu],
                            self.m_cos_beta[ivu],
                            self.m_sin_alpha[ivu],
                            self.m_cos_alpha[ivu],
                            self.m_xzfac[ivu],
                            self.m_yzfac[ivu],
                            del_z,
                            lslice_start,
                            lgpu_end,
                            self.m_ithick_reproj,
                            self.m_xcen_out,
                            tmp_cen_in,
                            self.m_min_xreproj,
                            self.m_xproj_offset,
                            self.m_ycen_out,
                            self.m_min_yreproj,
                            self.m_yproj_offset,
                            self.m_center_slice,
                            self.m_if_alpha,
                            self.m_dmean_in,
                        );
                    } else {
                        //
                        // GPU with local alignments: fill warpDelz array for all lines
                        let mut wd = std::mem::take(&mut self.m_warp_delz);
                        for line in lslice_start..=lgpu_end {
                            let off = ((line - lslice_start) * self.m_num_warp_delz) as usize;
                            self.fill_warp_delz(&mut wd[off..], iv, line);
                        }
                        self.m_warp_delz = wd;
                        //
                        // Get the xprojmin and max adjusted by 5
                        xproj_min = 10000000.;
                        xproj_max = 0.;
                        for load in in_load_start..=in_load_end {
                            iys = b3dnint!(load as f32 + self.m_yproj_offset);
                            ix = 1;
                            while ix <= nx_load {
                                let (mut sb, mut cb, mut sa, mut ca) = (0f32, 0f32, 0f32, 0f32);
                                self.local_proj_factors(
                                    (ix + self.m_min_xload - 1) as f32,
                                    iys,
                                    iv,
                                    &mut sb,
                                    &mut cb,
                                    &mut sa,
                                    &mut ca,
                                );
                                sin_alph = sb + (1. - ycen_adj) * cb;
                                cos_alph = sb + (self.m_ithick_reproj as f32 - ycen_adj) * cb;

                                xproj_min = b3dmin!(xproj_min as f64, sin_alph as f64 - 5.) as f32;
                                xproj_min = b3dmin!(xproj_min as f64, cos_alph as f64 - 5.) as f32;
                                xproj_max = b3dmax!(xproj_max as f64, sin_alph as f64 + 5.) as f32;
                                xproj_max = b3dmax!(xproj_max as f64, cos_alph as f64 + 5.) as f32;
                                ix += nx_load - 1;
                            }
                        }
                        // print *,'xprojmin, max', xprojMin, xprojMax
                        //
                        // Do it
                        ind_jump = gpu_reproj_local(
                            &mut self.m_reproj_lines,
                            self.m_sin_beta[ivu],
                            self.m_cos_beta[ivu],
                            self.m_sin_alpha[ivu],
                            self.m_cos_alpha[ivu],
                            self.m_xzfac[ivu],
                            self.m_yzfac[ivu],
                            self.m_nx_warp,
                            self.m_ny_warp,
                            self.m_ix_start_warp,
                            self.m_iy_start_warp,
                            self.m_idel_xwarp,
                            self.m_idel_ywarp,
                            &self.m_warp_delz,
                            self.m_num_warp_delz,
                            self.m_dx_warp_delz,
                            xproj_min,
                            xproj_max,
                            lslice_start,
                            lgpu_end,
                            self.m_ithick_reproj,
                            iv,
                            self.m_xcen_out,
                            self.m_xcen_in,
                            self.m_axis_xoffset,
                            self.m_min_xload,
                            self.m_xproj_offset,
                            ycen_adj,
                            self.m_yproj_offset,
                            self.m_center_slice,
                            self.m_dmean_in,
                        );
                    }
                    if ind_jump != 0 {
                        break;
                    }
                    wall_cumul = wall_cumul + wall_time() - wall_start;
                    self.write_reproj_lines(iv, lslice_start, lgpu_end, dmin, dmax, dtot8);
                    wall_start = wall_time();
                }
                if ind_jump == 0 {
                    if self.m_debug != 0 {
                        printf!("GPU reprojection time %.5f\n", cf(wall_cumul));
                    }
                    *lslice_end = lgpu_end;
                    return;
                }
            }
        }
        //
        // CPU REPROJECTION: loop on views; first handle non-local alignments
        for iv in 1..=self.m_num_views {
            let ivu = (iv - 1) as usize;
            if self.m_thresh_polarity != 0. {
                self.thresholded_reproj(iv, lslice_start, *lslice_end, in_load_start, in_load_end);
            } else if self.m_nx_warp == 0 {
                //
                // Get the delta z for this view
                cos_alph = self.m_cos_alpha[ivu];
                sin_alph = self.m_sin_alpha[ivu];
                cos_bet = self.m_cos_beta[ivu];
                sin_bet = self.m_sin_beta[ivu];
                del_z = self.reproj_del_z(
                    sin_bet,
                    cos_bet,
                    sin_alph,
                    cos_alph,
                    self.m_xzfac[ivu],
                    self.m_yzfac[ivu],
                );
                // print *,sbeta, cbeta, salf, calf, xzfac(iv), yzfac(iv)
                // print *,delx, delz
                //
                // Loop on the output lines to be done
                let input = std::mem::take(&mut self.m_input_array);
                let mut lines = std::mem::take(&mut self.m_reproj_lines);
                for line in lslice_start..=*lslice_end {
                    line_base = (line - lslice_start) * self.m_iwidth + 1;
                    self.reproj_one_angle(
                        &input,
                        &mut lines[(line_base - 1) as usize..],
                        in_load_start,
                        in_load_end,
                        line,
                        cos_bet,
                        sin_bet,
                        cos_alph,
                        sin_alph,
                        del_z,
                        self.m_iwidth,
                        self.m_ithick_reproj,
                        self.m_in_plane_size,
                        nx_load,
                        self.m_min_xreproj,
                        self.m_min_yreproj,
                        self.m_xproj_offset,
                        self.m_yproj_offset,
                        self.m_xcen_out,
                        self.m_ycen_out,
                        self.m_xcen_in + self.m_axis_xoffset,
                        self.m_center_slice,
                        self.m_if_alpha,
                        self.m_xzfac[ivu],
                        self.m_yzfac[ivu],
                        self.m_dmean_in,
                    );
                }
                self.m_input_array = input;
                self.m_reproj_lines = lines;
            } else {
                //
                // LOCAL ALIGNMENTS
                //
                // first step: precompute all the x/yprojf/z  for all slices
                // general BUG ycenAdj replaces ycenOut - minYreproj (off by 1)
                xproj_min = 10000000.;
                xproj_max = 0.;
                for load in in_load_start..=in_load_end {
                    ind_base = nx_load * (load - in_load_start);
                    iys = b3dnint!(load as f32 + self.m_yproj_offset);
                    for ix in 1..=nx_load {
                        ind = ind_base + ix;
                        let iu = (ind - 1) as usize;
                        let (mut xf, mut xz, mut yf, mut yz) = (0f32, 0f32, 0f32, 0f32);
                        self.local_proj_factors(
                            (ix + self.m_min_xload - 1) as f32,
                            iys,
                            iv,
                            &mut xf,
                            &mut xz,
                            &mut yf,
                            &mut yz,
                        );
                        self.m_xproj_fs[iu] = xf;
                        self.m_xproj_zs[iu] = xz;
                        self.m_yproj_fs[iu] = yf;
                        self.m_yproj_zs[iu] = yz;
                        if ix == 1 {
                            let v = xf + (1. - ycen_adj) * xz;
                            xproj_min = if xproj_min < v { xproj_min } else { v };
                            let v = xf + (self.m_ithick_reproj as f32 - ycen_adj) * xz;
                            xproj_min = if xproj_min < v { xproj_min } else { v };
                        }
                        if ix == nx_load {
                            let v = xf + (1. - ycen_adj) * xz;
                            xproj_max = if xproj_max > v { xproj_max } else { v };
                            let v = xf + (self.m_ithick_reproj as f32 - ycen_adj) * xz;
                            xproj_max = if xproj_max > v { xproj_max } else { v };
                        }
                    }
                }
                // print *,'xprojmin, max', xprojMin, xprojMax
                //
                // loop on lines to be done
                for line in lslice_start..=*lslice_end {
                    line_base = (line - lslice_start) * self.m_iwidth;
                    let mut wd = std::mem::take(&mut self.m_warp_delz);
                    self.fill_warp_delz(&mut wd, iv, line);
                    self.m_warp_delz = wd;
                    // print *,iv, line, inloadstr, inloadend
                    //
                    // loop on pixels across line
                    yproj = line as f32 + self.m_yproj_offset;
                    for ixp in 1..=self.m_iwidth {
                        //
                        // Get x projection coord, starting centered Z coordinate, and
                        // approximate x and y coordinates
                        // xproj, yproj are coordinates in original projections
                        // Equations relate them to coordinates in reconstruction
                        // and then X coordinate is adjusted to be a loaded X index
                        // and Y coordinate is adjusted to be a slice of reconstruction
                        xproj = ixp as f32 + self.m_xproj_offset;
                        zz = (1. - ycen_adj as f64) as f32;
                        sum = 0.;
                        // print *,ixp, xproj, yproj, xx, yy
                        // BUG these lines needed to be swapped and yprojOffset deferred
                        yy = (yproj + zz * (self.m_sin_alpha[ivu] - self.m_yzfac[ivu])
                            - self.m_center_slice)
                            / self.m_cos_alpha[ivu]
                            + self.m_center_slice;
                        xx = (xproj
                            - yy * self.m_sin_alpha[ivu] * self.m_sin_beta[ivu]
                            - zz * (self.m_cos_alpha[ivu] * self.m_sin_beta[ivu]
                                + self.m_xzfac[ivu])
                            - self.m_xcen_in
                            - self.m_axis_xoffset)
                            / self.m_cos_beta[ivu]
                            + self.m_xcen_out
                            - (self.m_min_xload - 1) as f32;
                        yy = yy - self.m_yproj_offset;
                        //
                        // Move on ray up in Z
                        last_zdone = 0;
                        try_jump = true;

                        z_jump = ((x_jump * self.m_cos_beta[ivu]) as f64
                            / b3dmax!(0.2, b3dabs!(self.m_sin_beta[ivu]) as f64))
                            as f32;
                        while zz < (self.m_ithick_reproj + 1) as f32 - ycen_adj && last_zdone == 0 {
                            if (xproj as f64) < xproj_min as f64 - 5.
                                || xproj as f64 > xproj_max as f64 + 5.
                            {
                                sum = sum + self.m_dmean_in as f64;
                            } else {
                                self.loaded_projecting_point(
                                    xproj,
                                    yproj,
                                    zz,
                                    nx_load,
                                    in_load_start,
                                    in_load_end,
                                    &mut xx,
                                    &mut yy,
                                );
                                //
                                // If X or Y is out of bounds, fill with mean
                                if yy < in_load_start as f32 - y_end_tol
                                    || yy > in_load_end as f32 + y_end_tol
                                    || (xx as f64) < 1.
                                    || xx >= nx_load as f32
                                {
                                    sum = sum + self.m_dmean_in as f64;
                                } else {
                                    //
                                    // otherwise, get x, y, z indexes, clamp y to limits,
                                    // allow a fractional Z pixel at top of volume
                                    ix = fortran_int!(f32: xx);
                                    fx = xx - ix as f32;
                                    one_mfx = (1. - fx as f64) as f32;
                                    yy = b3dmax!(
                                        in_load_start as f32 as f64,
                                        b3dmin!(in_load_end as f64 - 0.01, yy as f64)
                                    ) as f32;
                                    iy = fortran_int!(f32: yy);
                                    fy = yy - iy as f32;
                                    one_mfy = (1. - fy as f64) as f32;
                                    // BUG ????  Shouldn't this be + ycenOut - minYreproj?
                                    iz = b3dmax!(1., (zz + ycen_adj) as f64) as i32;
                                    fz = zz + ycen_adj - iz as f32;
                                    one_mfz = (1. - fz as f64) as f32;
                                    if iz == self.m_ithick_reproj {
                                        iz = iz - 1;
                                        fz = one_mfz;
                                        one_mfz = 0.;
                                        last_zdone = 1;
                                    }
                                    //
                                    // Do the interpolation
                                    d11 = one_mfx * one_mfy;
                                    d12 = one_mfx * fy;
                                    d21 = fx * one_mfy;
                                    d22 = fx * fy;
                                    ind = self.m_in_plane_size * (iy - in_load_start)
                                        + (iz - 1) * nx_load
                                        + ix;
                                    let ia = &self.m_input_array;
                                    let ips = self.m_in_plane_size;
                                    let g = |k: i32| ia[k as usize];
                                    sum = sum
                                        + (one_mfz
                                            * (d11 * g(ind - 1)
                                                + d12 * g(ind + ips - 1)
                                                + d21 * g(ind)
                                                + d22 * g(ind + ips)))
                                            as f64
                                        + (fz
                                            * (d11 * g(ind + nx_load - 1)
                                                + d12 * g(ind + ips + nx_load - 1)
                                                + d21 * g(ind + nx_load)
                                                + d22 * g(ind + ips + nx_load)))
                                            as f64;
                                    //
                                    if try_jump {
                                        self.proj_sum_local(
                                            &mut xx,
                                            &mut yy,
                                            &mut zz,
                                            &mut sum,
                                            xproj,
                                            yproj,
                                            self.m_sin_beta[ivu],
                                            nx_load,
                                            in_load_start,
                                            in_load_end,
                                            z_jump,
                                            ycen_adj,
                                        );
                                    }
                                }
                            }
                            //
                            // Adjust Z by local factor, move X approximately for next pixel
                            ind = b3dmax!(
                                1.,
                                b3dmin!(self.m_num_warp_delz as f32, xx / self.m_dx_warp_delz)
                                    as f64
                            ) as i32;
                            zz = zz + self.m_warp_delz[(ind - 1) as usize];
                            xx = xx + self.m_sin_beta[ivu];
                        }
                        self.m_reproj_lines[(line_base + ixp - 1) as usize] = sum as f32;
                    }
                }
            }
            wall_cumul = wall_cumul + wall_time() - wall_start;
            self.write_reproj_lines(iv, lslice_start, *lslice_end, dmin, dmax, dtot8);
            wall_start = wall_time();
        }
        if self.m_debug != 0 {
            printf!("CPU reprojection time %9.5f\n", cf(wall_cumul));
        }
    }

    /// `Tilt::fillWarpDelz` (`tilt.cpp:6260`): compute delta z as function of X
    /// across the loaded slice.
    fn fill_warp_delz(&self, warp_delz: &mut [f32], iv: i32, line: i32) {
        let mut xx: f32;
        let (mut f1, mut f2, mut f3, mut f4) = (0f32, 0f32, 0f32, 0f32);
        let mut ixc: i32;
        let iys: i32;
        let (mut ind1, mut ind2, mut ind3, mut ind4) = (0i32, 0i32, 0i32, 0i32);

        iys = b3dnint!(line as f32 + self.m_yproj_offset);
        for i in 1..=self.m_num_warp_delz {
            xx = 1. + self.m_dx_warp_delz * (i - 1) as f32;
            ixc = b3dnint!(
                xx + self.m_min_xload as f32 - 1. - self.m_xcen_out
                    + self.m_xcen_in
                    + self.m_axis_xoffset
            );
            self.local_factors(
                ixc as f32, iys, iv, &mut ind1, &mut ind2, &mut ind3, &mut ind4, &mut f1, &mut f2,
                &mut f3, &mut f4,
            );
            let dz = |k: i32| {
                let k = (k - 1) as usize;
                self.reproj_del_z(
                    self.m_swarp_beta[k],
                    self.m_cwarp_beta[k],
                    self.m_swarp_alpha[k],
                    self.m_cwarp_alpha[k],
                    self.m_warp_xzfac[k],
                    self.m_warp_yzfac[k],
                )
            };
            warp_delz[(i - 1) as usize] =
                f1 * dz(ind1) + f2 * dz(ind2) + f3 * dz(ind3) + f4 * dz(ind4);
        }
        // print *,'got delz eg:', wrpdlz(1), wrpdlz(numWarpDelz/2), &
        // wrpdlz(numWarpDelz)
    }

    /// `Tilt::thresholdedReproj` (`tilt.cpp:6287`): does a reprojection only of
    /// discrete points beyond a threshold.
    fn thresholded_reproj(
        &mut self,
        iv: i32,
        lslice_start: i32,
        lslice_end: i32,
        in_load_start: i32,
        in_load_end: i32,
    ) {
        let polarity: f32;
        let (mut f11, mut f12, mut f21, mut f22) = (0f32, 0f32, 0f32, 0f32);
        let mut rl_x: f32;
        let mut rl_slice: f32;
        let mut rl_z: f32;
        let num_lines: i32;
        let mut iyp: i32;
        let mut ind: i32;
        let mut ind1: i32 = 0;
        let mut ind2: i32 = 0;
        let mut ixp: i32;
        let mut xproj: f32 = 0.;
        let mut yproj: f32 = 0.;
        let mut fx: f32;
        let mut fy: f32;
        let nx_load = self.m_max_xload + 1 - self.m_min_xload;
        num_lines = lslice_end + 1 - lslice_start;
        for ind in 0..(num_lines * self.m_iwidth) as usize {
            self.m_reproj_lines[ind] = self.m_thresh_fill_val * self.m_ithick_reproj as f32;
        }
        polarity = b3dsign!(1f64, self.m_thresh_polarity) as f32;
        for loaded_slice in in_load_start..=in_load_end {
            ind = (loaded_slice - in_load_start) * self.m_in_plane_size;
            for iy in 1..=self.m_ithick_reproj {
                for ix in 1..=nx_load {
                    if polarity * (self.m_input_array[ind as usize] - self.m_thresh_for_reproj)
                        >= 0.
                    {
                        //
                        // Get real coordinate position within full reconstruction and
                        // projection position in full aligned stack
                        rl_x = (ix + self.m_min_xreproj - 1) as f32;
                        rl_slice = loaded_slice as f32;
                        rl_z = (iy + self.m_min_yreproj - 1) as f32;
                        self.projection_position(
                            iv,
                            rl_x,
                            rl_z,
                            rl_slice,
                            self.m_ycen_mod_proj,
                            &mut xproj,
                            &mut yproj,
                            &mut ind1,
                            &mut ind2,
                            &mut f11,
                            &mut f12,
                            &mut f21,
                            &mut f22,
                        );
                        //
                        // Adjust for position in reprojection being produced then adjust
                        // Y to be an index in the lines being produced
                        xproj = xproj - self.m_xproj_offset;
                        yproj = yproj - ((self.m_min_zreproj - 1) as f32 + self.m_yproj_offset);
                        yproj = yproj + self.m_islice_start as f32 - lslice_start as f32;
                        ixp = fortran_int!(f32: xproj);
                        iyp = fortran_int!(f32: yproj);
                        if ixp >= 0 && ixp <= self.m_iwidth && iyp >= 0 && iyp <= num_lines {
                            //
                            // If any of the 4 actual pixels is in range, get the
                            // interpolation factors for those 4 surrounding pixels
                            fx = xproj - ixp as f32;
                            fy = yproj - iyp as f32;
                            f11 = ((1. - fx as f64) * (1. - fy as f64)) as f32;
                            f12 = ((1. - fx as f64) * fy as f64) as f32;
                            f21 = (fx as f64 * (1. - fy as f64)) as f32;
                            f22 = fx * fy;
                            ind1 = ixp + self.m_iwidth * (iyp - 1);
                            let w = self.m_iwidth;
                            let rl = &mut self.m_reproj_lines;
                            let mark = self.m_thresh_mark_val;
                            let fillv = self.m_thresh_fill_val;
                            //
                            // If NOT summing, just mark any pixel with fraction above
                            // threshold Otherwise add the fraction times the value
                            if self.m_thresh_sum_fac < 1. {
                                if f11 >= self.m_thresh_sum_fac && ixp > 0 && iyp > 0 {
                                    rl[(ind1 - 1) as usize] = mark;
                                }
                                if f12 >= self.m_thresh_sum_fac && ixp > 0 && iyp < num_lines {
                                    rl[(ind1 + w - 1) as usize] = mark;
                                }
                                if f21 >= self.m_thresh_sum_fac && ixp < w && iyp > 0 {
                                    rl[(ind1 + 1 - 1) as usize] = mark;
                                }
                                if f22 >= self.m_thresh_sum_fac && ixp < w && iyp < num_lines {
                                    rl[(ind1 + 1 + w - 1) as usize] = mark;
                                }
                            } else {
                                if ixp > 0 && iyp > 0 {
                                    let k = (ind1 - 1) as usize;
                                    rl[k] = rl[k] + f11 * mark - fillv;
                                }
                                if ixp > 0 && iyp < num_lines {
                                    let k = (ind1 + w - 1) as usize;
                                    rl[k] = rl[k] + f12 * mark - fillv;
                                }
                                if ixp < w && iyp > 0 {
                                    let k = (ind1 + 1 - 1) as usize;
                                    rl[k] = rl[k] + f21 * mark - fillv;
                                }
                                if ixp < w && iyp < num_lines {
                                    let k = (ind1 + 1 + w - 1) as usize;
                                    rl[k] = rl[k] + f22 * mark - fillv;
                                }
                            }
                        }
                    }
                    ind = ind + 1;
                }
            }
        }
    }
}

impl Tilt {
    /// `Tilt::reprojOneAngle` (`tilt.cpp:6381`): reprojects a line at one angle
    /// from projection data in `array` into `reprojLines`.  `line` is the Y value
    /// in projections, Z value in reconstructed slices.  `array` is loaded with
    /// slices from `inLoadStart` to `inLoadEnd`, with a slice size of
    /// `inPlaneSize` and `nxLoad` values on each line in X.
    fn reproj_one_angle(
        &mut self,
        array: &[f32],
        reproj_lines: &mut [f32],
        in_load_start: i32,
        in_load_end: i32,
        line: i32,
        cos_bet: f32,
        sin_bet: f32,
        cos_alph: f32,
        sin_alph: f32,
        delz_in: f32,
        iwidth: i32,
        ithick_reproj: i32,
        in_plane_size: i32,
        nx_load: i32,
        min_xreproj: i32,
        min_yreproj: i32,
        xproj_offset: f32,
        yproj_offset: f32,
        xcen_out: f32,
        ycen_out: f32,
        xcen_pdelxx: f32,
        center_slice: f32,
        if_alpha: i32,
        xz_fac_view: f32,
        yz_fac_view: f32,
        dmean_in: f32,
    ) {
        let mut del_x: f32;
        //
        let mut ix: i32;
        let mut iz: i32;
        let num_z: i32;
        let mut iys: i32 = 0;
        let mut ix_end: i32;
        let mut ix_start: i32;
        let mut ind: i32;
        let mut ind_base: i32 = 0;
        let z_num: f32;
        let mut fz: f32;
        let mut one_mfz: f32;
        let mut zz: f32;
        let mut xx: f32;
        let mut fx: f32;
        let y_end_tol: f32;
        let mut pfill: f32;
        let mut salf_sbet_over_calf: f32;
        let mut xcen_adj: f32;
        let mut yslc: f32;
        let mut one_mfx: f32;
        let mut yy: f32;
        let mut fy: f32 = 0.;
        let mut one_mfy: f32 = 0.;
        let mut xproj: f32;
        let mut yproj: f32;
        let mut y_slice: f32;
        let (mut d11, mut d12, mut d21, mut d22): (f32, f32, f32, f32);
        let mut del_y: f32;
        let mut xx8: f64;
        let mut yy8: f64;
        let mut zz8: f64;
        let num_x: i32;
        let mut ixy_ok_start: i32;
        let mut ixy_ok_end: i32;
        let mut ifix_start: i32;
        let mut ifix_end: i32;
        let mut iray: i32;
        let mut lut_ind: i32;
        let mut del_z: f32;
        let x_num: f32;
        let eps: f32;
        let mut angle: f32;
        let mut x_left: f32;
        let mut x_right: f32;
        let mut zpart: f32;
        let mut max_area_sum: f32;
        let mut area: f32;
        const MAX_DIST: i32 = 1000;
        let mut ind_del_ray = [0i32; (MAX_DIST + 1) as usize];
        let mut num_rays_hit = [0i32; (MAX_DIST + 1) as usize];
        let mut ray_areas = [0f32; (3 * (MAX_DIST + 1)) as usize];
        let at = |k: i32| array[k as usize];
        let ipsz = in_plane_size;

        y_end_tol = 3.05;
        eps = 0.01;
        for indv in 0..iwidth as usize {
            reproj_lines[indv] = 0.;
        }
        if b3dabs!(sin_bet * ithick_reproj as f32) <= b3dabs!(cos_bet * iwidth as f32)
            || self.m_use_intersections != 0
        {
            //
            del_z = delz_in;
            if self.m_use_intersections != 0 {
                del_z = 1.;
                angle = (sin_bet.atan2(cos_bet) as f64 / RADIANS_PER_DEGREE) as f32;
                make_ray_area_lookup_table(
                    angle,
                    1,
                    MAX_DIST,
                    0.01,
                    &mut ind_del_ray,
                    &mut num_rays_hit,
                    &mut ray_areas,
                );

                // Use this array for summing projection areas
                if self.m_xray_start.is_empty() {
                    self.m_xray_start = vec![0.; iwidth as usize];
                }
                for indv in 0..iwidth as usize {
                    self.m_xray_start[indv] = 0.;
                }
            }
            del_x = (1. / cos_bet as f64) as f32;
            z_num = (1. + ((ithick_reproj - 1) as f32 / del_z) as f64) as f32;
            let mut nz = z_num as i32;
            if (z_num - nz as f32) as f64 >= 0.1 {
                nz = nz + 1;
            }
            num_z = nz;
            //
            // Loop up in Z through slices, adding in lines of data to the
            // output line
            for kz in 1..=num_z {
                zz = 1. + (kz - 1) as f32 * del_z;
                iz = fortran_int!(f32: zz);
                fz = zz - iz as f32;
                one_mfz = (1. - fz as f64) as f32;
                pfill = dmean_in;
                //
                // If Z is past the top, drop back one line and set up fractions
                // to take just a fraction of the top line
                if zz >= ithick_reproj as f32 {
                    zz = ithick_reproj as f32;
                    iz = ithick_reproj - 1;
                    fz = one_mfz;
                    one_mfz = 0.;
                    pfill = dmean_in * fz;
                }
                zz = zz + min_yreproj as f32 - 1. - ycen_out;
                //
                // Get y slice for this z value
                yproj = line as f32 + yproj_offset;
                yy = (yproj + zz * (sin_alph - yz_fac_view) - center_slice) / cos_alph;
                y_slice = yy + center_slice - yproj_offset;
                if if_alpha == 0 {
                    y_slice = line as f32;
                }
                // if (line==591) print *,kz, zz, iz, fz, omfz, yproj, yy, yslice
                if y_slice < in_load_start as f32 - y_end_tol
                    || y_slice > in_load_end as f32 + y_end_tol
                {
                    //
                    // Really out of bounds, do fill
                    // if (line==591) print *,'Out of bounds, view, line, zz', line, zz
                    for indv in 0..iwidth as usize {
                        reproj_lines[indv] += pfill;
                    }
                } else if self.m_use_intersections != 0 {
                    // For intersections, switch to looping on reconstruction pixels, as
                    // that works at all angles with a maximum of 3 rays intersecting each
                    ix_start = 1;
                    ix_end = iwidth;
                    zpart = zz * sin_bet + xcen_pdelxx - xproj_offset;
                    if cos_bet != 0. {
                        x_left = ((1. - zpart as f64) / cos_bet as f64 + xcen_out as f64
                            - (min_xreproj - 1) as f64) as f32;
                        x_right =
                            (iwidth as f32 - zpart) / cos_bet + xcen_out - (min_xreproj - 1) as f32;
                        ix_start = (b3dmin!(x_left, x_right)).floor() as i32;
                        ix_end = (b3dmax!(x_left, x_right)).ceil() as i32;
                    }
                    ix_start = b3dmax!(1, ix_start);
                    ix_end = b3dmin!(ix_end, iwidth);
                    ind_base = 1 + ipsz * (line - in_load_start) + (iz - 1) * nx_load;

                    // Subtract 0.5 to turn it from an output index centered on first ray
                    // at 1 to a value that ranges from 0 to iwidth
                    if cos_bet != 0. {
                        xx8 = (ix_start as f64 + (min_xreproj as f64 - 1.) - xcen_out as f64)
                            * cos_bet as f64
                            + zpart as f64
                            - 0.5;
                    } else {
                        xx8 = kz as f64 + iwidth as f64 / 2. - ((ithick_reproj + 1) / 2) as f64;
                    }

                    // Loop across line and on rays for each pixel, testing rays for
                    // being with range
                    for i in ix_start..=ix_end {
                        ix = fortran_int!(f64: xx8 + 1.);
                        fx = (ix as f64 - xx8) as f32;
                        lut_ind = (fx * MAX_DIST as f32) as i32;
                        lut_ind = b3dmax!(0, b3dmin!(MAX_DIST - 1, lut_ind));
                        ind = ind_base + i - 1;
                        for ray_ind in 0..num_rays_hit[lut_ind as usize] {
                            iray = ix + ind_del_ray[lut_ind as usize] + ray_ind - 1;
                            if iray >= 0 && iray < iwidth {
                                area = ray_areas[(3 * lut_ind + ray_ind) as usize];
                                reproj_lines[iray as usize] += area * at(ind - 1);
                                self.m_xray_start[iray as usize] += area;
                            }
                        }
                        xx8 += cos_bet as f64;
                    }
                } else {
                    //
                    // otherwise set up iy and interpolation factors
                    iys = y_slice.floor() as i32;
                    if if_alpha != 0 {
                        if iys < in_load_start {
                            iys = in_load_start;
                            fy = 0.;
                        } else if iys >= in_load_end {
                            iys = in_load_end - 1;
                            fy = 1.;
                        } else {
                            fy = y_slice - iys as f32;
                        }
                        one_mfy = (1. - fy as f64) as f32;
                    }
                    //
                    // Now get starting X coordinate, fill to left
                    xproj = 1. + xproj_offset;
                    xx = (xproj
                        - (yy * sin_alph * sin_bet
                            + zz * (cos_alph * sin_bet + xz_fac_view)
                            + xcen_pdelxx))
                        / cos_bet
                        + xcen_out
                        - (min_xreproj - 1) as f32;
                    ix_start = 1;
                    if xx < 1. {
                        ix_start = ((1. - xx as f64) / del_x as f64 + 1.).ceil() as i32;
                    } else if xx >= iwidth as f32 {
                        ix_start = (((iwidth as f32 - xx) / del_x) as f64 + 1.).ceil() as i32;
                    }
                    xx = xx + (ix_start - 1) as f32 * del_x;
                    if xx < 1. || xx >= iwidth as f32 {
                        ix_start = ix_start + 1;
                        xx = xx + del_x;
                    }
                    if ix_start > 1 {
                        for indv in 0..(ix_start - 1) as usize {
                            reproj_lines[indv] += pfill;
                        }
                    }
                    //
                    // get ending X coordinate, fill to right
                    ix_end = iwidth;
                    if xx + (ix_end - ix_start) as f32 * del_x >= iwidth as f32 - eps {
                        ix_end = ((iwidth as f32 - xx) / del_x + ix_start as f32) as i32;
                        if xx + (ix_end - ix_start) as f32 * del_x >= iwidth as f32 - eps {
                            ix_end = ix_end - 1;
                        }
                    } else if ((xx + (ix_end - ix_start) as f32 * del_x) as f64) < 1. + eps as f64 {
                        ix_end = ((1. - xx as f64) / del_x as f64 + ix_start as f64) as i32;
                        if ((xx + (ix_end - ix_start) as f32 * del_x) as f64) < 1. + eps as f64 {
                            ix_end = ix_end - 1;
                        }
                    }
                    if ix_end < iwidth {
                        for indv in ix_end as usize..iwidth as usize {
                            reproj_lines[indv] += pfill;
                        }
                    }

                    //
                    // Add the line in: do simple 2x2 interpolation if no alpha
                    ind_base = 1 + ipsz * (iys - in_load_start) + (iz - 1) * nx_load;
                    // if (line==591) print *,ixst, ixnd
                    xx8 = xx as f64;
                    if if_alpha == 0 {
                        // The source's loop over `i`; the four samples of each step are
                        // taken from one bounds-checked window of `array` starting at
                        // `ind - 1`, and the output element from the output run.
                        let nxl = nx_load as usize;
                        if ix_end >= ix_start {
                            for r in
                                reproj_lines[(ix_start - 1) as usize..ix_end as usize].iter_mut()
                            {
                                ix = fortran_int!(f64: xx8);
                                fx = (xx8 - ix as f64) as f32;
                                one_mfx = (1. - fx as f64) as f32;
                                ind = ind_base + ix - 1;
                                let q = &array[(ind - 1) as usize..(ind - 1) as usize + nxl + 2];
                                *r += one_mfz * one_mfx * q[0]
                                    + one_mfz * fx * q[1]
                                    + fz * one_mfx * q[nxl]
                                    + fz * fx * q[nxl + 1];

                                // if (line==591.and.i==164) print *,reprojLines(i), array(ind), &
                                // array(ind + 1), array(ind + nxload), array(ind + nxload + 1)
                                xx8 = xx8 + del_x as f64;
                            }
                        }
                    } else {
                        //
                        // Or do the full 3D interpolation if any variation in Y
                        for i in ix_start..=ix_end {
                            ix = fortran_int!(f64: xx8);
                            fx = (xx8 - ix as f64) as f32;
                            one_mfx = (1. - fx as f64) as f32;
                            d11 = one_mfx * one_mfy;
                            d12 = one_mfx * fy;
                            d21 = fx * one_mfy;
                            d22 = fx * fy;
                            ind = ind_base + ix - 1;
                            reproj_lines[(i - 1) as usize] += one_mfz
                                * (d11 * at(ind - 1)
                                    + d12 * at(ind + ipsz - 1)
                                    + d21 * at(ind)
                                    + d22 * at(ind + ipsz))
                                + fz * (d11 * at(ind + nx_load - 1)
                                    + d12 * at(ind + ipsz + nx_load - 1)
                                    + d21 * at(ind + nx_load)
                                    + d22 * at(ind + ipsz + nx_load));
                            xx8 = xx8 + del_x as f64;
                        }
                    }
                }
            }

            // For ray intersections, find the maximum area along a ray and add fill for
            // rays having less than the maximum
            if self.m_use_intersections != 0 {
                max_area_sum = 0.;
                for indv in 0..iwidth as usize {
                    max_area_sum = if max_area_sum > self.m_xray_start[indv] {
                        max_area_sum
                    } else {
                        self.m_xray_start[indv]
                    };
                }
                for indv in 0..iwidth as usize {
                    reproj_lines[indv] += dmean_in * (max_area_sum - self.m_xray_start[indv]);
                }
            }
        } else {
            //
            // angles higher than the corner angle need to be done in vertical lines,
            // outer loop on X instead of z
            // Spacing between vertical lines is now sine beta
            // The step between pixels along a line is 1/sin beta with no alpha tilt,
            // The alpha tilt compresses it by the delta Z factor divided by cosine
            // beta, the amount that delta Z factor is compressed from cosine beta.
            del_x = b3dabs!(sin_bet);
            x_num = (1. + ((iwidth - 1) as f32 / del_x) as f64) as f32;
            let mut nx = x_num as i32;
            if (x_num - nx as f32) as f64 >= 0.1 {
                nx = nx + 1;
            }
            num_x = nx;
            del_z = delz_in / (sin_bet * b3dabs!(cos_bet));
            del_y = del_z * (sin_alph - yz_fac_view) / cos_alph;
            if (b3dabs!(del_y) as f64) < 1.0e-10 {
                del_y = 0.;
            }
            // print *,'delx, dely, delz', delx, dely, delz
            //
            // Loop in X across slices, adding in vertical lines of data to the output
            // line
            for kx in 1..=num_x {
                xx = 1. + (kx - 1) as f32 * del_x;
                ix = fortran_int!(f32: xx);
                fx = xx - ix as f32;
                one_mfx = (1. - fx as f64) as f32;
                pfill = dmean_in;
                //
                // If X is past the end, drop back one line and set up fractions
                // to take just a fraction of the right column
                if xx >= iwidth as f32 {
                    xx = iwidth as f32;
                    ix = iwidth - 1;
                    fx = one_mfx;
                    one_mfx = 0.;
                    pfill = dmean_in * fx;
                }

                // get starting Z coordinate
                salf_sbet_over_calf = sin_alph * sin_bet / cos_alph;
                xcen_adj = xcen_out - (min_xreproj - 1) as f32;
                xproj = 1. + xproj_offset;
                yproj = line as f32 + yproj_offset;

                zz = (xproj
                    - (yproj - center_slice) * salf_sbet_over_calf
                    - xcen_pdelxx
                    - (xx - xcen_adj) * cos_bet)
                    / ((sin_alph - yz_fac_view) * salf_sbet_over_calf
                        + cos_alph * sin_bet
                        + xz_fac_view);
                //
                // Get y slice for this z value, then convert Z to be index coordinates
                // in slice
                yy = (yproj + zz * (sin_alph - yz_fac_view) - center_slice) / cos_alph;
                y_slice = yy + center_slice - yproj_offset;
                if if_alpha == 0 {
                    y_slice = line as f32;
                }
                zz = zz - ((min_yreproj - 1) as f32 - ycen_out);
                //
                // Get starting X proj limit based on Z
                ix_start = 1;
                if (zz as f64) < 1. {
                    ix_start = ((1. - zz as f64) / del_z as f64 + 1.).ceil() as i32;
                } else if zz >= ithick_reproj as f32 {
                    ix_start = (((ithick_reproj as f32 - zz) / del_z) as f64 + 1.).ceil() as i32;
                }
                //
                // Revise starting limit for Y
                if if_alpha != 0 && del_y != 0. {
                    yslc = (y_slice as f64 + (ix_start as f64 - 1.) * del_y as f64) as f32;
                    if yslc < in_load_start as f32 - y_end_tol {
                        ix_start = (((in_load_start as f32 - y_end_tol - y_slice) / del_y) as f64
                            + 1.)
                            .ceil() as i32;
                    } else if yslc > in_load_end as f32 + y_end_tol {
                        ix_start = (((in_load_end as f32 + y_end_tol - y_slice) / del_y) as f64
                            + 1.)
                            .ceil() as i32;
                    }
                }
                //
                // Adjust Z start for final start and make sure it works, adjust Y also
                zz = (zz as f64 + (ix_start as f64 - 1.) * del_z as f64) as f32;
                if (zz as f64) < 1. || zz >= ithick_reproj as f32 {
                    zz = zz + del_z;
                    ix_start = ix_start + 1;
                }
                y_slice = (y_slice as f64 + (ix_start as f64 - 1.) * del_y as f64) as f32;
                if if_alpha == 0 {
                    y_slice = line as f32;
                }
                //
                // get ending coordinate based on limits in Z and Y
                ix_end = iwidth;
                if zz + (ix_end - ix_start) as f32 * del_z >= ithick_reproj as f32 - eps {
                    ix_end = ((ithick_reproj as f32 - zz) / del_z + ix_start as f32) as i32;
                    if zz + (ix_end - ix_start) as f32 * del_z >= ithick_reproj as f32 - eps {
                        ix_end = ix_end - 1;
                    }
                } else if ((zz + (ix_end - ix_start) as f32 * del_z) as f64) < 1. + eps as f64 {
                    ix_end = ((1. - zz as f64) / del_z as f64 + ix_start as f64) as i32;
                    if ((zz + (ix_end - ix_start) as f32 * del_z) as f64) < 1. + eps as f64 {
                        ix_end = ix_end - 1;
                    }
                }
                if if_alpha != 0 && del_y != 0. {
                    yslc = y_slice + (ix_end - ix_start) as f32 * del_y;
                    if yslc < in_load_start as f32 - y_end_tol {
                        ix_end = ((in_load_start as f32 - y_end_tol - y_slice) / del_y
                            + ix_start as f32) as i32;
                    } else if yslc > in_load_end as f32 + y_end_tol {
                        ix_end = ((in_load_end as f32 + y_end_tol - y_slice) / del_y
                            + ix_start as f32) as i32;
                    }
                }
                //
                // Now get X indexes within which Y can safely be varied
                ixy_ok_start = ix_start;
                ixy_ok_end = ix_end;
                if if_alpha != 0 && del_y != 0. {
                    if y_slice < in_load_start as f32 {
                        ixy_ok_start = ((in_load_start as f32 - y_slice) / del_y + ix_start as f32)
                            .ceil() as i32;
                        if y_slice + (ixy_ok_start - ix_start) as f32 * del_y
                            < in_load_start as f32 + eps
                        {
                            ixy_ok_start = ixy_ok_start + 1;
                        }
                    } else if y_slice >= in_load_end as f32 {
                        ixy_ok_start = ((in_load_end as f32 - y_slice) / del_y + ix_start as f32)
                            .ceil() as i32;
                        if y_slice + (ixy_ok_start - ix_start) as f32 * del_y
                            >= in_load_end as f32 - eps
                        {
                            ixy_ok_start = ixy_ok_start + 1;
                        }
                    }
                    y_slice = y_slice + (ixy_ok_start - ix_start) as f32 * del_y;
                    //
                    yslc = y_slice + (ix_end - ixy_ok_start) as f32 * del_y;
                    if yslc < in_load_start as f32 {
                        ixy_ok_end =
                            ((in_load_start as f32 - y_slice) / del_y + ixy_ok_start as f32) as i32;
                        if y_slice + (ixy_ok_end - ixy_ok_start) as f32 * del_y
                            < in_load_start as f32 + eps
                        {
                            ixy_ok_end = ixy_ok_end - 1;
                        }
                    } else if yslc >= in_load_end as f32 {
                        ixy_ok_end =
                            ((in_load_end as f32 - y_slice) / del_y + ixy_ok_start as f32) as i32;
                        if y_slice + (ixy_ok_end - ixy_ok_start) as f32 * del_y
                            >= in_load_end as f32 - eps
                        {
                            ixy_ok_end = ixy_ok_end - 1;
                        }
                    }
                }
                // write( *,'(i5,f7.1,4i5,2f7.1)') kx, xx, ixst, ixyOKst, ixyOKnd, ixnd, zz, &
                // zz+(ixnd-ixst)*delz
                //
                // Do the fills
                if ix_start > 1 {
                    for indv in 0..(ix_start - 1) as usize {
                        reproj_lines[indv] += pfill;
                    }
                }
                if ix_end < iwidth {
                    for indv in ix_end as usize..iwidth as usize {
                        reproj_lines[indv] += pfill;
                    }
                }
                //
                // Add the line in: do simple 2x2 interpolation if no alpha
                // if (line==591) print *,ixst, ixnd
                if if_alpha == 0 {
                    zz8 = zz as f64;
                    ind_base = 1 + ipsz * (line - in_load_start) + ix - 1;
                    // As in the Z loop above: one bounds-checked window per step.
                    let nxl = nx_load as usize;
                    if ix_end >= ix_start {
                        for r in reproj_lines[(ix_start - 1) as usize..ix_end as usize].iter_mut() {
                            iz = fortran_int!(f64: zz8);
                            fz = (zz8 - iz as f64) as f32;
                            one_mfz = (1. - fz as f64) as f32;
                            ind = ind_base + (iz - 1) * nx_load;
                            let q = &array[(ind - 1) as usize..(ind - 1) as usize + nxl + 2];
                            *r += one_mfz * one_mfx * q[0]
                                + one_mfz * fx * q[1]
                                + fz * one_mfx * q[nxl]
                                + fz * fx * q[nxl + 1];
                            // if (i==70) print *,reprojLines(i)
                            zz8 = zz8 + del_z as f64;
                        }
                    }
                } else {
                    //
                    // Or do the full 3D interpolation if any variation in Y, starting
                    // with the loop where Y varies
                    yy8 = y_slice as f64;
                    zz8 = (zz + (ixy_ok_start - ix_start) as f32 * del_z) as f64;
                    ind_base = 1 - ipsz * in_load_start + ix - 1;
                    for i in ixy_ok_start..=ixy_ok_end {
                        iz = fortran_int!(f64: zz8);
                        fz = (zz8 - iz as f64) as f32;
                        one_mfz = (1. - fz as f64) as f32;
                        iys = fortran_int!(f64: yy8);
                        fy = (yy8 - iys as f64) as f32;
                        one_mfy = (1. - fy as f64) as f32;
                        d11 = one_mfx * one_mfy;
                        d12 = one_mfx * fy;
                        d21 = fx * one_mfy;
                        d22 = fx * fy;
                        ind = ind_base + ipsz * iys + (iz - 1) * nx_load;
                        reproj_lines[(i - 1) as usize] += one_mfz
                            * (d11 * at(ind - 1)
                                + d12 * at(ind + ipsz - 1)
                                + d21 * at(ind)
                                + d22 * at(ind + ipsz))
                            + fz * (d11 * at(ind + nx_load - 1)
                                + d12 * at(ind + ipsz + nx_load - 1)
                                + d21 * at(ind + nx_load)
                                + d22 * at(ind + ipsz + nx_load));
                        zz8 = zz8 + del_z as f64;
                        yy8 = yy8 + del_y as f64;
                    }
                    //
                    // Now do special loops with Y fixed - do the one at the end first
                    // since Y and Z are all set for that
                    ifix_start = ixy_ok_end + 1;
                    ifix_end = ix_end;
                    for _iy_fix in 1..=2 {
                        for i in ifix_start..=ifix_end {
                            iz = fortran_int!(f64: zz8);
                            fz = (zz8 - iz as f64) as f32;
                            one_mfz = (1. - fz as f64) as f32;
                            d11 = one_mfx * one_mfy;
                            d12 = one_mfx * fy;
                            d21 = fx * one_mfy;
                            d22 = fx * fy;
                            ind = ind_base + ipsz * iys + (iz - 1) * nx_load;
                            reproj_lines[(i - 1) as usize] += one_mfz
                                * (d11 * at(ind - 1)
                                    + d12 * at(ind + ipsz - 1)
                                    + d21 * at(ind)
                                    + d22 * at(ind + ipsz))
                                + fz * (d11 * at(ind + nx_load - 1)
                                    + d12 * at(ind + ipsz + nx_load - 1)
                                    + d21 * at(ind + nx_load)
                                    + d22 * at(ind + ipsz + nx_load));
                            zz8 = zz8 + del_z as f64;
                        }
                        //
                        // Set up for loop with Y fixed at start, reset y and z
                        yy8 = y_slice as f64;
                        zz8 = zz as f64;
                        iys = fortran_int!(f64: yy8);
                        fy = (yy8 - iys as f64) as f32;
                        one_mfy = (1. - fy as f64) as f32;
                        ifix_start = ix_start;
                        ifix_end = ixy_ok_start - 1;
                    }
                }
            }
        }
    }

    /// `Tilt::writeReprojLines` (`tilt.cpp:6851`): writes lines `lineStart` to
    /// `lineEnd` for view `iv` of a reprojection.
    fn write_reproj_lines(
        &mut self,
        iv: i32,
        line_start: i32,
        line_end: i32,
        dmin: &mut f32,
        dmax: &mut f32,
        dtot8: &mut f64,
    ) {
        let mut iy_out: i32;
        let num_vals: i32;
        let mut val: f32;
        let ivu = (iv - 1) as usize;
        //
        // Write the line after scaling.  Scale log data to give approximately
        // constant mean levels.  Descale non-log data by exposure weights
        num_vals = self.m_iwidth * (line_end + 1 - line_start);
        if self.m_if_log != 0 {
            // Hopefully this works for local as well
            let base = (self.m_proj_mean + self.m_base_for_log).log10();
            if self.m_thresh_polarity != 0. {
                val = base - self.m_ithick_reproj as f32 * self.m_dmean_in;
            } else if b3dabs!(self.m_sin_beta[ivu] * self.m_ithick_reproj as f32)
                <= b3dabs!(self.m_cos_beta[ivu] * self.m_iwidth as f32)
            {
                val = base
                    - self.m_ithick_reproj as f32 * self.m_dmean_in / b3dabs!(self.m_cos_beta[ivu]);
            } else {
                val = base - self.m_iwidth as f32 * self.m_dmean_in / b3dabs!(self.m_sin_beta[ivu]);
            }
            if self.m_debug != 0 {
                printf!(
                    "iv = %d,  lineStart = %d,  lineEnd = %d,  val = %g\n",
                    ci(iv),
                    ci(line_start),
                    ci(line_end),
                    cf(val as f64)
                );
            }
            for i in 0..num_vals as usize {
                self.m_reproj_lines[i] =
                    10.0f32.powf(self.m_reproj_lines[i] + val) - self.m_base_for_log;
            }
        } else {
            val = self.m_expose_weight[ivu];
            if self.m_thresh_polarity != 0. {
                val = self.m_expose_weight[((self.m_num_views + 1) / 2 - 1) as usize];
            }
            for i in 0..num_vals as usize {
                self.m_reproj_lines[i] = self.m_reproj_lines[i] / val;
            }
        }
        if self.m_proj_subtraction != 0 {
            let err = unsafe {
                iiu_set_position(1, iv - 1, line_start - 1);
                iiu_read_lines(
                    1,
                    self.m_orig_lines.as_mut_ptr().cast(),
                    line_end + 1 - line_start,
                )
            };
            if err != 0 {
                exit_error(b"Reading from original projection file");
            }
            for i in 0..num_vals as usize {
                self.m_reproj_lines[i] -= self.m_orig_lines[i];
            }
        }
        for i in 0..num_vals as usize {
            val = self.m_reproj_lines[i];
            // if (debug .and. val < dmin) print *,'min:', i, val
            // if (debug .and. val > dmax) print *,'max:', i, val
            *dmin = b3dmin!(*dmin, val);
            *dmax = b3dmax!(*dmax, val);
            *dtot8 = *dtot8 + val as f64;
        }
        for line in line_start..=line_end {
            iy_out = line - self.m_islice_start;
            if self.m_min_tot_slice > 0 {
                iy_out = line - self.m_min_tot_slice;
            }
            let off = ((line - line_start) * self.m_iwidth) as usize;
            unsafe {
                par_wrt_posn(2, iv - 1, iy_out);
                par_wrt_lin(2, self.m_reproj_lines[off..].as_mut_ptr().cast());
            }
        }
    }

    /// `Tilt::projectModel` (`tilt.cpp:6911`): projects model points onto the
    /// included views, writes the model, and exits.
    fn project_model(
        &mut self,
        mut imod: Imod,
        out_model: &[u8],
        out_angles: Option<&[u8]>,
        transform_file: Option<&[u8]>,
        num_views_orig: i32,
        defocus_file: Option<&[u8]>,
        pix_for_defocus: f32,
        mut focus_invert: f32,
    ) -> ! {
        let mut num_points: i32;
        let mut fw_base: usize;
        let mut iv: i32;
        let mut num_xfs: i32;
        let mut one_val: f32;
        let mut rl_x: f32;
        let mut rl_z: f32;
        let mut rl_slice: f32;
        let mut yy: f32;
        let mut zz: f32;
        let mut xproj: f32 = 0.;
        let mut yproj: f32 = 0.;
        let z_offset: f32;
        let mut j: i32 = 0;
        let mut lslice: i32 = 0;
        let size: i32;
        let mut after: usize;
        let mut skip_below: i32;
        let deg_to_rad: f32;
        let mut pt_defocus: f32;
        let mut beta_inv: f32;
        let mut zzp: f32;
        let mut zzpp: f32;
        let mut fjl_mat: [[f32; 2]; 2] = [[0.; 2]; 2];
        let mut f1234: [f32; 4] = [0.; 4];
        let mut gamma: f32;
        let (mut gamma_sum, mut beta_sum, mut alpha_sum): (f32, f32, f32);
        let no_value: f32 = -1.0e30;
        let value_test: f32 = -0.9e30;
        let mut obj1_thresh: f32 = no_value;
        let mut thresh: f32 = 0.;
        let mut val_min: f32 = 0.;
        let mut val_max: f32 = 0.;
        let mut ind1234: [i32; 4] = [0; 4];
        let mut values: Vec<f32>;
        let mut coords: Vec<f32>;
        let mut alixf: Vec<f32>;
        let mut ali_rot: Vec<f32>;
        let mut defocus: Vec<f32>;
        let mut sizes: Vec<f32>;
        let mut has_sizes: bool;
        let mut any_sizes = false;
        let mut map_file_to_view: Vec<i32>;
        let mut angle_fp: Option<ImodFile> = None;
        let mut new_mod: Imod;
        deg_to_rad = RADIANS_PER_DEGREE as f32;

        if imod.flags & IMODF_FLIPYZ != 0 {
            imod_flip_yz(&mut imod);
        }

        //  add up the points
        num_points = 0;
        for obj in &imod.obj {
            for cont in &obj.cont {
                num_points += cont.pts.len() as i32;
            }
        }

        size = if imod.obj[0].pdrawsize != 0 {
            imod.obj[0].pdrawsize
        } else {
            5
        };
        //
        // Models are defined as having Z coordinates ranging from -0.5 to NZ - 0.5 so Z
        // will need to be shifted up by 1 to get to pixel index coordinates
        // But in old beadtrack models, Z started at 0 so the offset needs to be only
        // 0.5 Recognize old beadtrack model by lack of flag AND original object name so
        // that other old models will work
        let name0 = &imod.obj[0].name;
        let name_len = name0.iter().position(|&c| c == 0).unwrap_or(name0.len());
        z_offset = if imod.flags & IMODF_Z_FROM_MINUSPT5 == 0
            && name0[..name_len].windows(8).any(|w| w == b"Wimp no.")
        {
            0.5
        } else {
            1.0
        };
        let jn = (num_views_orig + 10) as usize;
        values = vec![0.; (num_points + 10) as usize];
        sizes = vec![0.; (num_points + 10) as usize];
        coords = vec![0.; (3 * num_points + 30) as usize];
        alixf = vec![0.; 6 * jn];
        ali_rot = vec![0.; jn];
        defocus = vec![0.; jn];
        map_file_to_view = vec![0; self.m_lim_view as usize];
        if let Some(out_angles) = out_angles {
            let name = String::from_utf8_lossy(out_angles).into_owned();
            imod_backup_file(&name);
            angle_fp = ImodFile::open(&name, "w");
            if angle_fp.is_none() {
                exit_error_fmt!("Opening angle output file %s", CArg::Bytes(out_angles));
            }
            if let Some(transform_file) = transform_file {
                num_xfs = -6 * (num_views_orig + 10);
                self.get_values_from_lines(
                    None,
                    Some(transform_file),
                    &mut alixf,
                    None,
                    false,
                    &mut num_xfs,
                    "transforms",
                );
                num_xfs /= 6;
                if num_xfs > num_views_orig {
                    printf!(
                        "\nWARNING: TILT - More alignment transforms than views in input stack\n"
                    );
                }
                if num_xfs < num_views_orig {
                    exit_error(b"Fewer alignment transforms than views in input stack");
                }
                for j in 0..num_views_orig as usize {
                    (ali_rot[j], _, _, _) = amat_to_rotmagstr(
                        alixf[6 * j],
                        alixf[6 * j + 1],
                        alixf[6 * j + 2],
                        alixf[6 * j + 3],
                    );
                }
            }
            //
            if let Some(defocus_file) = defocus_file {
                let mut n = num_views_orig;
                self.get_values_from_lines(
                    None,
                    Some(defocus_file),
                    &mut defocus,
                    None,
                    true,
                    &mut n,
                    "defocus values",
                );
                if focus_invert == 0. {
                    focus_invert = 1.;
                } else {
                    focus_invert = -1.;
                }
            }
        }
        //
        // get each point and its contour value into the arrays
        num_points = 0;
        for ob in 0..imod.obj.len() {
            let obj = &imod.obj[ob];
            skip_below = 0;
            if obj.flags & IMOD_OBJFLAG_USE_VALUE != 0
                && istore_get_min_max(
                    &obj.store,
                    obj.cont.len() as i32,
                    GEN_STORE_MINMAX1,
                    &mut val_min,
                    &mut val_max,
                ) != 0
            {
                thresh = (obj.valblack as f32 * (val_max - val_min)) as f64 as f32;
                thresh = (thresh as f64 / 255. + val_min as f64) as f32;
                if ob == 0 {
                    obj1_thresh = thresh;
                }
                if obj.matflags2 as u32 & MATFLAGS2_SKIP_LOW != 0 {
                    skip_below = self.m_skip_unseen_points;
                }
            }
            for co in 0..obj.cont.len() {
                let cont = &obj.cont[co];
                has_sizes = !cont.sizes.is_empty();
                if has_sizes {
                    any_sizes = true;
                }
                one_val = no_value;
                let (index, after_l) = istore_lookup(&obj.store, co as i32);
                after = after_l;
                if let Some(index) = index {
                    for j in index..after {
                        let item = &obj.store[j];
                        if item.type_ == GEN_STORE_VALUE1 {
                            one_val = item.value.f();
                            break;
                        }
                    }
                }
                if one_val > value_test && skip_below != 0 && one_val < thresh {
                    continue;
                }
                for pt in 0..cont.pts.len() {
                    let np = num_points as usize;
                    coords[3 * np] = cont.pts[pt].x;
                    coords[1 + 3 * np] = cont.pts[pt].y;
                    coords[2 + 3 * np] = cont.pts[pt].z;
                    sizes[np] = if has_sizes { cont.sizes[pt] } else { 0. };
                    values[np] = one_val;
                    num_points += 1;
                }
            }
        }
        //
        // Start a new model
        new_mod = match imod_new() {
            Some(m) => m,
            None => exit_error(b"Allocating a new model structure"),
        };
        new_mod.xmax = self.m_nx_proj;
        new_mod.ymax = self.m_ny_proj;
        new_mod.zmax = self.m_num_views;

        let input_head = unsafe { iiu_mrc_header(1, "projectModel", 1, 0) };
        if imod_set_ref_image(&mut new_mod, unsafe { &*input_head }) != 0 {
            exit_error(b"Putting image reference information into output model");
        }
        //
        // Build a map from views in file to ordered views in program
        for nfv in 1..=num_views_orig as usize {
            map_file_to_view[nfv - 1] = 0;
        }
        for iv in 1..=self.m_num_views {
            map_file_to_view[(self.m_map_used_view[(iv - 1) as usize] - 1) as usize] = iv;
        }
        //
        // Loop on the points, start new contour for each
        if imod_new_object(&mut new_mod) != 0 {
            exit_error(b"Allocating new object in model");
        }
        let mut conts = match imod_contours_new(num_points) {
            Some(c) => c,
            None => exit_error(b"Failure to allocate contour array in model"),
        };
        let mut obj_store: Vec<Istore> = std::mem::take(&mut new_mod.obj[0].store);
        let mut store = Istore::default();
        store.flags = GEN_STORE_FLOAT << 2;
        store.type_ = GEN_STORE_VALUE1;

        for ipt in 0..num_points as usize {
            store.value.set_f(values[ipt]);
            store.index.set_i(ipt as i32);
            if values[ipt] > value_test && istore_insert(&mut obj_store, store) != 0 {
                exit_error(b"Inserting value information into object");
            }
            //
            // Get real pixel coordinates in tomogram file
            rl_x = (coords[3 * ipt] as f64 + 0.5) as f32;
            rl_z = (coords[1 + 3 * ipt] as f64 + 0.5) as f32;
            rl_slice = coords[2 + 3 * ipt] + z_offset;
            //
            // This may never be tested but seems simple enough
            if self.m_perpendicular == 0 {
                rl_z = coords[2 + 3 * ipt] + z_offset;
                rl_slice = (coords[1 + 3 * ipt] as f64 + 0.5) as f32;
            }

            // Get point array
            let cont = &mut conts[ipt];
            cont.pts = Vec::with_capacity(num_views_orig as usize);
            if any_sizes {
                cont.sizes = vec![0.; num_views_orig as usize];
            }

            //
            // Loop on the views in the file
            for nfv in 1..=num_views_orig {
                iv = map_file_to_view[(nfv - 1) as usize];
                if iv > 0 {
                    {
                        let [[f00, f01], [f10, f11]] = &mut fjl_mat;
                        self.projection_position(
                            iv,
                            rl_x,
                            rl_z,
                            rl_slice,
                            self.m_ycen_mod_proj,
                            &mut xproj,
                            &mut yproj,
                            &mut j,
                            &mut lslice,
                            f00,
                            f01,
                            f10,
                            f11,
                        );
                    }
                    if let Some(fp) = angle_fp.as_mut() {
                        beta_sum = 0.;
                        alpha_sum = 0.;
                        gamma_sum = 0.;
                        if self.m_nx_warp == 0 {
                            alpha_sum = self.m_alpha[(iv - 1) as usize];
                            beta_sum = -self.m_angles[(iv - 1) as usize] / deg_to_rad;
                        } else {
                            for j_inc in 0..2 {
                                for ls_inc in 0..2 {
                                    let [i0, i1, i2, i3] = &mut ind1234;
                                    let [g0, g1, g2, g3] = &mut f1234;
                                    self.local_factors(
                                        b3dnint!(
                                            (j + j_inc as i32) as f32 - self.m_xcen_out
                                                + self.m_xcen_in
                                                + self.m_axis_xoffset
                                        ) as f32,
                                        lslice + ls_inc as i32,
                                        nfv,
                                        i0,
                                        i1,
                                        i2,
                                        i3,
                                        g0,
                                        g1,
                                        g2,
                                        g3,
                                    );
                                    for ic in 0..4 {
                                        f1234[ic] *= fjl_mat[j_inc][ls_inc];
                                    }
                                    for ic in 0..4 {
                                        fw_base = (6 * (ind1234[ic] - 1)) as usize;

                                        // These arguments are a11, a12, a21, a22 and in
                                        // memory they are (1,1) (2,1) (1,2) (2,2), so swap
                                        // middle terms
                                        (gamma, _, _, _) = amat_to_rotmagstr(
                                            self.m_fwarp[fw_base],
                                            self.m_fwarp[fw_base + 2],
                                            self.m_fwarp[fw_base + 1],
                                            self.m_fwarp[fw_base + 3],
                                        );
                                        //
                                        // alpha was left as degrees and with its native
                                        // sign, with the negative taken when taking the
                                        // sin; but tilt angle and delBeta were converted to
                                        // radians and made negative The transform has the
                                        // amount image needs to be rotated; need negative
                                        // for rotation of specimen
                                        let k = (ind1234[ic] - 1) as usize;
                                        alpha_sum = alpha_sum
                                            + f1234[ic]
                                                * (self.m_alpha[(iv - 1) as usize]
                                                    + self.m_del_alpha[k]);
                                        beta_sum = beta_sum
                                            - f1234[ic]
                                                * (self.m_angles[(iv - 1) as usize]
                                                    + self.m_del_beta[k])
                                                / deg_to_rad;
                                        gamma_sum = gamma_sum - f1234[ic] * gamma;
                                    }
                                }
                            }
                        }
                        if transform_file.is_some() {
                            gamma_sum = gamma_sum - ali_rot[(nfv - 1) as usize];
                        }
                        pt_defocus = 0.;
                        if defocus_file.is_some() {
                            // Start with the centered Y position used in projectionPosition
                            // but invert the centered Z position because the Z axis is
                            // pointing down in the modeled X/Z planes and we want a correct
                            // Z beta was inverted above from the stored (inverted) angles
                            // and is now correct for working with true Z coordinates
                            yy = rl_slice - self.m_center_slice;
                            zz =
                                -(rl_z - self.m_ycen_mod_proj) * self.m_compress[(iv - 1) as usize];
                            beta_inv = focus_invert * beta_sum * deg_to_rad;

                            // Now get the Z' after the alpha tilt from Y and Z
                            // then get the Z'' after the beta tilt from Z' and centered X
                            // The right side swings to negative Z at positive tilt and
                            // this corresponds to more underfocus (per Ctfplotter), so
                            // subtract the Z from defocus
                            zzp = yy * (alpha_sum * deg_to_rad).sin()
                                + zz * (alpha_sum * deg_to_rad).cos();
                            zzpp = zzp * beta_inv.cos() - (rl_x - self.m_xcen_out) * beta_inv.sin();
                            pt_defocus = defocus[(nfv - 1) as usize] - pix_for_defocus * zzpp;
                        }
                        let _ = fp.write_all(&c_format_bytes(
                            "%6d %5d %5d %8.3f %8.3f %8.3f %8.0f\n",
                            &[
                                ci(ipt as i32 + 1),
                                ci(iv),
                                ci(nfv),
                                cf(alpha_sum as f64),
                                cf(beta_sum as f64),
                                cf(gamma_sum as f64),
                                cf(pt_defocus as f64),
                            ],
                        ));
                    }
                    //
                    // Store model coordinates
                    cont.pts.push(Ipoint {
                        x: (xproj as f64 - 0.5) as f32,
                        y: (yproj as f64 - 0.5) as f32,
                        z: (nfv as f64 - 1.) as f32,
                    });
                    if any_sizes {
                        cont.sizes[(nfv - 1) as usize] = sizes[ipt];
                    }
                }
            }
            // `imodel_write` writes `psize` sizes, the first entries of the
            // `numViewsOrig`-long array the source allocates.
            if any_sizes {
                let psize = cont.pts.len();
                cont.sizes.truncate(psize);
            }
        }
        //
        // Save model
        let obj = &mut new_mod.obj[0];
        obj.cont = conts;
        obj.store = obj_store;

        //
        // Set to open contour, show values etc., and show sphere on section only
        obj.flags |= IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_USE_VALUE | IMOD_OBJFLAG_PNT_ON_SEC;
        obj.matflags2 |= (MATFLAGS2_CONSTANT | MATFLAGS2_SKIP_LOW) as u8;
        obj.pdrawsize = size;
        istore_find_add_min_max1(obj);

        // Transfer the threshold from the first object using the new min/max
        if obj1_thresh > value_test
            && istore_get_min_max(
                &obj.store,
                obj.cont.len() as i32,
                GEN_STORE_MINMAX1,
                &mut val_min,
                &mut val_max,
            ) != 0
        {
            let mut ipt =
                b3dnint!((obj1_thresh - val_min) as f64 * 255. / (val_max - val_min) as f64);
            ipt = b3dmax!(0, b3dmin!(255, ipt));
            obj.valblack = ipt as u8;
        }

        drop(angle_fp);
        let out_name = String::from_utf8_lossy(out_model).into_owned();
        imod_backup_file(&out_name);
        if imod_file_write(&new_mod, &out_name).is_err() {
            exit_error_fmt!("Writing new model to file %s", CArg::Bytes(out_model));
        }
        printf!(
            "%d points written to output model\n",
            ci(num_points * num_views_orig)
        );
        c_exit(0);
    }

    /// `Tilt::projectionPosition` (`tilt.cpp:7211`): computes the projection
    /// position `xproj`, `yproj` on view `iv` (#'d from 1) of position `rlX`,
    /// `rlZ`, `rlSlice` in the reconstruction, where these are real pixel
    /// coordinates, equal to 1 in the middle of the first pixel.  Other returned
    /// values are the rounded down X and slice numbers in `j` and `lslice`, and the
    /// four interpolation factors for the real pixel position.
    fn projection_position(
        &self,
        iv: i32,
        rl_x: f32,
        rl_z: f32,
        rl_slice: f32,
        ycen_out: f32,
        xproj: &mut f32,
        yproj: &mut f32,
        j: &mut i32,
        lslice: &mut i32,
        f11: &mut f32,
        f12: &mut f32,
        f21: &mut f32,
        f22: &mut f32,
    ) {
        let zz: f32;
        let yy: f32;
        let zpart: f32;
        let fj: f32;
        let fls: f32;
        let (mut xf11, mut xz11, mut yf11, mut yz11) = (0f32, 0f32, 0f32, 0f32);
        let (mut xf21, mut xz21, mut yf21, mut yz21) = (0f32, 0f32, 0f32, 0f32);
        let (mut xf12, mut xz12, mut yf12, mut yz12) = (0f32, 0f32, 0f32, 0f32);
        let (mut xf22, mut xz22, mut yf22, mut yz22) = (0f32, 0f32, 0f32, 0f32);
        let xprojf: f32;
        let xprojz: f32;
        let yprojf: f32;
        let yprojz: f32;
        let ivu = (iv - 1) as usize;

        zz = (rl_z - ycen_out) * self.m_compress[ivu];
        yy = rl_slice - self.m_center_slice;
        if self.m_nx_warp == 0 {
            zpart = yy * self.m_sin_alpha[ivu] * self.m_sin_beta[ivu]
                + zz * (self.m_cos_alpha[ivu] * self.m_sin_beta[ivu] + self.m_xzfac[ivu])
                + self.m_xcen_in
                + self.m_axis_xoffset;
            *yproj = yy * self.m_cos_alpha[ivu] - zz * (self.m_sin_alpha[ivu] - self.m_yzfac[ivu])
                + self.m_center_slice;
            *xproj = zpart + (rl_x - self.m_xcen_out) * self.m_cos_beta[ivu];
        } else {
            //
            // local alignments
            *j = rl_x as i32;
            fj = rl_x - *j as f32;
            *lslice = rl_slice as i32;
            fls = rl_slice - *lslice as f32;
            *f11 = ((1. - fj as f64) * (1. - fls as f64)) as f32;
            *f12 = ((1. - fj as f64) * fls as f64) as f32;
            *f21 = (fj as f64 * (1. - fls as f64)) as f32;
            *f22 = fj * fls;
            self.local_proj_factors(
                *j as f32, *lslice, iv, &mut xf11, &mut xz11, &mut yf11, &mut yz11,
            );
            self.local_proj_factors(
                (*j + 1) as f32,
                *lslice,
                iv,
                &mut xf21,
                &mut xz21,
                &mut yf21,
                &mut yz21,
            );
            self.local_proj_factors(
                *j as f32,
                *lslice + 1,
                iv,
                &mut xf12,
                &mut xz12,
                &mut yf12,
                &mut yz12,
            );
            self.local_proj_factors(
                (*j + 1) as f32,
                *lslice + 1,
                iv,
                &mut xf22,
                &mut xz22,
                &mut yf22,
                &mut yz22,
            );
            xprojf = *f11 * xf11 + *f12 * xf12 + *f21 * xf21 + *f22 * xf22;
            xprojz = *f11 * xz11 + *f12 * xz12 + *f21 * xz21 + *f22 * xz22;
            yprojf = *f11 * yf11 + *f12 * yf12 + *f21 * yf21 + *f22 * yf22;
            yprojz = *f11 * yz11 + *f12 * yz12 + *f21 * yz21 + *f22 * yz22;
            *xproj = xprojf + zz * xprojz;
            *yproj = yprojf + zz * yprojz;
            //
        }
    }

    /// `Tilt::memoryError` (`tilt.cpp:7256`), replacement for Fortran
    /// `memoryError`.  Allocations here abort rather than returning `NULL`, so no
    /// caller needs it; kept as the source has it.
    #[allow(dead_code)]
    fn memory_error(&self, ok: bool, mess: &str) {
        if !ok {
            exit_error_fmt!("Failure to allocate %s", CArg::Str(mess));
        }
    }

    //
    // Wrappers to Fortran routines
    //

    /// `Tilt::bpSumNoX` (`tilt.cpp:7265`).
    fn bp_sum_no_x(
        &self,
        array: &mut [f32],
        ind1: &mut i32,
        input: &[f32],
        ipoint: i32,
        n: i32,
        xproj1: f64,
        cbeta: f32,
    ) {
        let mut xproj1 = xproj1;
        bp_sum_no_x(array, ind1, input, ipoint, n, &mut xproj1, cbeta);
    }

    /// `Tilt::bpSumAreaNoX` (`tilt.cpp:7271`).
    fn bp_sum_area_no_x(
        &self,
        array: &mut [f32],
        ind1: i32,
        input: &[f32],
        ipoint: i32,
        n: i32,
        xproj1: f64,
        cbeta: f32,
        nx_out: i32,
        ind_del_ray: &[i32],
        num_rays_hit: &[i32],
        ray_areas: &[f32],
        max_dist: i32,
    ) {
        let mut xproj1 = xproj1;
        bp_sum_area_no_x(
            array,
            ind1,
            input,
            ipoint,
            n,
            &mut xproj1,
            cbeta,
            nx_out,
            ind_del_ray,
            num_rays_hit,
            ray_areas,
            max_dist,
        );
    }

    /// `Tilt::bpSumXtilt` (`tilt.cpp:7279`).
    fn bp_sum_xtilt(
        &self,
        array: &mut [f32],
        ind1: &mut i32,
        input: &[f32],
        ipbase: i32,
        ipdel: i32,
        n: i32,
        xproj1: f64,
        cbeta: f32,
        yfrac: f32,
        omyfrac: f32,
    ) {
        let mut xproj1 = xproj1;
        bp_sum_xtilt(
            array,
            ind1,
            input,
            ipbase,
            ipdel,
            n,
            &mut xproj1,
            cbeta,
            yfrac,
            omyfrac,
        );
    }

    /// `Tilt::bpSumLocal` (`tilt.cpp:7285`).
    fn bp_sum_local(
        &self,
        array: &mut [f32],
        index: &mut i32,
        input: &[f32],
        zz: f32,
        xprojf: &[f32],
        xprojz: &[f32],
        yprojf: &[f32],
        yprojz: &[f32],
        ipoint: i32,
        ipdel: i32,
        lslice: i32,
        jstrt: i32,
        jend: i32,
    ) {
        bp_sum_local(
            array,
            index,
            input,
            zz,
            xprojf,
            xprojz,
            yprojf,
            yprojz,
            ipoint,
            ipdel,
            lslice,
            jstrt,
            jend,
            self.m_proj_super_fac,
        );
    }

    /// `Tilt::projSumLocal` (`tilt.cpp:7293`).
    fn proj_sum_local(
        &self,
        xx: &mut f32,
        yy: &mut f32,
        zz: &mut f32,
        sum: &mut f64,
        xproj: f32,
        yproj: f32,
        sin_beta: f32,
        nx_load: i32,
        in_load_start: i32,
        in_load_end: i32,
        z_jump: f32,
        ycen_adj: f32,
    ) {
        proj_sum_local(
            xx,
            yy,
            zz,
            sum,
            xproj,
            yproj,
            &self.m_input_array,
            &self.m_xproj_fs,
            &self.m_xproj_zs,
            &self.m_yproj_fs,
            &self.m_yproj_zs,
            self.m_num_warp_delz,
            self.m_dx_warp_delz,
            &self.m_warp_delz,
            self.m_ithick_reproj,
            sin_beta,
            nx_load,
            in_load_start,
            in_load_end,
            self.m_in_plane_size,
            z_jump,
            ycen_adj,
        );
    }

    /// `Tilt::loadedProjectingPoint` (`tilt.cpp:7303`).
    fn loaded_projecting_point(
        &self,
        xproj: f32,
        yproj: f32,
        zz: f32,
        nx_load: i32,
        in_load_start: i32,
        in_load_end: i32,
        xx: &mut f32,
        yy: &mut f32,
    ) {
        loaded_projecting_point(
            xproj,
            yproj,
            zz,
            nx_load,
            in_load_start,
            in_load_end,
            &self.m_xproj_fs,
            &self.m_xproj_zs,
            &self.m_yproj_fs,
            &self.m_yproj_zs,
            xx,
            yy,
        );
    }
}
