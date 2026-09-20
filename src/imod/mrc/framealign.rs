//! Translation of `IMOD/mrc/framealign.cpp` and `IMOD/mrc/framealign.h` —
//! the module for aligning movie frames passed in sequentially.
//!
//! Ownership notes, all of them forced by the C's pointer aliasing:
//!
//! * Every `float *` member is a `Vec<f32>`; an empty `Vec` is the C's `NULL`.
//! * The source's `float *fullArr` / `float *binArr` locals move between the
//!   work arrays, the saved stacks and the caller's frame.  They are
//!   [`FullRef`] / [`BinRef`] here — an index of *which* owned buffer the C
//!   pointer is currently pointing at — and the buffer is reached with
//!   [`FrameAlign::take_full`] / [`FrameAlign::put_full`], which move the
//!   `Vec` out of `self` and back so two buffers can be borrowed at once.
//! * `mGainRef` and `mCamDefects` are C pointers to arrays the *caller* owns
//!   across the whole run; they are `Rc` here, so storing one is a refcount
//!   bump rather than a copy of a 64 MB gain reference.
//! * `void *frame` plus `int type` is [`FrameData`], the same shape this tree
//!   already uses for a typed `void *` (`taperpad::PadIn`,
//!   `zoomdown::ZoomLines`, `islice::MrcData`).
//! * The GPU entry points are `nogpuframe.rs`, which is what the reference
//!   build links (it is configured without CUDA).  Every `fgpu*` call there
//!   fails, so the C's GPU branches are translated but always take the
//!   fallback path.

use crate::imod::clip::clip::CameraDefects;
use crate::imod::clip::correct_defects::{
    cor_def_correct_defects, cor_def_fill_defect_array, cor_def_scale_defects_for_k2,
    cor_def_surrounding_mean,
};
use crate::imod::libcfshr::b3dutil::{CArg, balanced_group_limits, data_size_for_mode};
use crate::imod::libcfshr::coresprocsthreads::{num_omp_threads, wall_time};
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, conjugate_product, dose_weight_filter, fourier_reduce_image, fourier_ring_corr,
    fourier_shift_image, nice_frame, set_peak_find_limits, xcorr_filter_part,
    xcorr_peak_find_width, xcorr_set_ctf,
};
use crate::imod::libcfshr::gcvspl::{gcvspl, splder};
use crate::imod::libcfshr::reduce_by_binning::extract_with_binning;
use crate::imod::libcfshr::regression::{mult_regress, robust_regress};
use crate::imod::libcfshr::samplemeansd::{sample_mean_sd, type_for_sample_mean};
use crate::imod::libcfshr::simplestat::{avg_sd, ls_fit_pred, ls_fit2_pred, sums_to_avg_sd};
use crate::imod::libcfshr::taperpad::{
    PadIn, slice_noise_taper_pad, slice_taper_in_pad, slice_taper_out_pad,
};
use crate::imod::libcfshr::zoomdown::{ZoomLines, ZoomOut, select_zoom_filter, zoom_with_filter};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::mrc::frameutil::{
    CharArgType, util_dump_fft, util_dump_image, util_print, util_roll_saved_frames,
    util_set_print_func,
};
use crate::imod::mrc::nogpuframe::*;
use std::collections::BTreeSet;
use std::rc::Rc;

/// C `MAX_ALL_VS_ALL` (`framealign.h:20`).
pub const MAX_ALL_VS_ALL: usize = 100;
/// C `MAX_FILTERS` (`framealign.h:21`).
pub const MAX_FILTERS: usize = 6;

pub const GPU_FOR_SUMMING: i32 = 1;
pub const GPU_DO_EVEN_ODD: i32 = 1 << 1;
pub const GPU_FOR_ALIGNING: i32 = 1 << 2;
pub const GPU_DO_NOISE_TAPER: i32 = 1 << 3;
pub const GPU_DO_BIN_PAD: i32 = 1 << 4;
pub const STACK_FULL_ON_GPU: i32 = 1 << 5;
pub const GPU_DO_GAIN_NORM: i32 = 1 << 6;
pub const GPU_CORRECT_DEFECTS: i32 = 1 << 7;
pub const GPU_DO_PREPROCESS: i32 = 1 << 8;
pub const GPU_STACK_LIMITED: i32 = 1 << 9;
pub const GPU_DO_UNWGT_SUM: i32 = 1 << 10;
pub const GPU_AVG_SUPER_2X: i32 = 1 << 11;
pub const GPU_AVG_SUPER_4X: i32 = 1 << 12;
pub const GPU_RUN_SHRMEMFRAME: i32 = 1 << 15;
pub const GPU_STACK_LIM_SHIFT: i32 = 20;
pub const GPU_STACK_LIM_MASK: i32 = 0xFFF;
pub const GPU_NUMBER_SHIFT: i32 = 16;
pub const GPU_NUMBER_MASK: i32 = 0x7;

const MRC_MODE_BYTE: i32 = 0;
const MRC_MODE_SHORT: i32 = 1;
const MRC_MODE_FLOAT: i32 = 2;
const MRC_MODE_USHORT: i32 = 6;

/// The C's `void *frame` plus its MRC mode.
#[derive(Clone, Copy)]
pub enum FrameData<'a> {
    Byte(&'a [u8]),
    Short(&'a [i16]),
    UShort(&'a [u16]),
    Float(&'a [f32]),
}

impl FrameData<'_> {
    /// The raw `unsigned char *` view the C hands `CorDefSurroundingMean` and
    /// `memcpy`.  Widening a typed slice to bytes never violates alignment.
    pub fn bytes(&self) -> &[u8] {
        match *self {
            FrameData::Byte(v) => v,
            FrameData::Short(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
            },
            FrameData::UShort(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
            },
            FrameData::Float(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
            },
        }
    }

    /// The `PadIn` for handing this frame to the taper/pad routines.
    pub fn pad_in(&self) -> PadIn<'_> {
        match *self {
            FrameData::Byte(v) => PadIn::Byte(v),
            FrameData::Short(v) => PadIn::Short(v),
            FrameData::UShort(v) => PadIn::UShort(v),
            FrameData::Float(v) => PadIn::Float(v),
        }
    }
}

/// The `reduce_by_binning` and `CorrectDefects` translations keep the C's
/// `void *` as a byte view (`reduce_by_binning.rs:658`,
/// `correct_defects.rs:487`); this is the same reinterpretation
/// `correct_defects.rs:563` performs in the other direction.  It is a view of
/// the caller's floats, not a copy, exactly as the C passes `float *`.
fn float_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn float_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    let len = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), len) }
}

/// Which owned buffer the source's `float *fullArr` / `float *useFrame` is
/// pointing at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FullRef {
    /// The C's `NULL`.
    Null,
    /// `mWorkFullSize`
    Work,
    /// `mSavedFullSize[i]`
    Saved(usize),
    /// The caller's `frame` — read-only in the C on every path that sets it.
    Frame,
}

/// Which owned buffer the source's `float *binArr` / `float *refArr` /
/// `float *groupArr` is pointing at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinRef {
    Null,
    /// `mWorkBinPad`
    Work,
    /// `mCorrBinPad`
    Corr,
    /// `mAlignSum`
    AlignSum,
    /// `mSavedBinPad[i]`
    Saved(usize),
    /// `mSavedGroups[i]`
    Group(usize),
}

/// C class `FrameAlign` (`framealign.h:42`).
pub struct FrameAlign {
    pub m_print_func: CharArgType,
    pub m_full_even_sum: Vec<f32>,
    pub m_full_odd_sum: Vec<f32>,
    pub m_unweight_sum: Vec<f32>,
    pub m_align_sum: Vec<f32>,
    pub m_work_full_size: Vec<f32>,
    /// `std::vector<float *> mSavedFullSize`
    pub m_saved_full_size: Vec<Vec<f32>>,
    /// `IntVec mSavedFullFrameNum`
    pub m_saved_full_frame_num: Vec<i32>,
    /// `std::vector<float *> mSavedBinPad`
    pub m_saved_bin_pad: Vec<Vec<f32>>,
    /// `std::vector<float *> mSavedGroups`
    pub m_saved_groups: Vec<Vec<f32>>,
    pub m_work_bin_pad: Vec<f32>,
    pub m_corr_bin_pad: Vec<f32>,
    pub m_corr_filt_temp: Vec<f32>,
    pub m_reduce_temp: Vec<f32>,
    pub m_shift_temp: Vec<f32>,
    /// C `unsigned char **mLinePtrs`, allocated `fullYpad` long.  The line
    /// pointers themselves are built where they are used, so only the count
    /// the C allocates is kept.
    pub m_line_ptrs: i32,
    pub m_fit_mat: Vec<f32>,
    pub m_fit_work: Vec<f32>,
    pub m_sub_filt_mask: [Vec<f32>; MAX_FILTERS],
    pub m_full_filt_mask: Vec<f32>,
    pub m_temp_sub_filt: Vec<f32>,
    pub m_wrap_temp: Vec<f32>,
    pub m_xshifts: [Vec<f32>; MAX_FILTERS + 1],
    pub m_yshifts: [Vec<f32>; MAX_FILTERS + 1],
    pub m_xall_shifts: [Vec<f32>; MAX_FILTERS],
    pub m_yall_shifts: [Vec<f32>; MAX_FILTERS],
    pub m_xfit_shifts: [[f32; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
    pub m_yfit_shifts: [[f32; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
    pub m_xnear_shifts: [f32; MAX_ALL_VS_ALL],
    pub m_ynear_shifts: [f32; MAX_ALL_VS_ALL],
    pub m_last_xfit: [[f32; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
    pub m_last_yfit: [[f32; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
    pub m_cumul_xdiff: [f32; MAX_FILTERS + 1],
    pub m_cumul_ydiff: [f32; MAX_FILTERS + 1],
    pub m_num_as_best_filt: [i32; MAX_FILTERS],
    pub m_radius2: [f32; MAX_FILTERS],
    pub m_sigma2: [f32; MAX_FILTERS],
    pub m_gpu_flags: i32,
    pub m_flags_for_unpad_call: i32,
    pub m_even_odd_for_sum_setup: i32,
    pub m_nx_gain: i32,
    pub m_ny_gain: i32,
    pub m_gain_ref: Option<Rc<Vec<f32>>>,
    pub m_cam_size_x: i32,
    pub m_cam_size_y: i32,
    pub m_cam_defects: Option<Rc<CameraDefects>>,
    pub m_trunc_limit: f32,
    pub m_defect_bin: i32,
    pub m_noise_length: i32,
    pub m_frame_type: i32,
    pub m_stack_type: i32,
    pub m_nx: i32,
    pub m_ny: i32,
    pub m_anti_filt_type: i32,
    pub m_xstart: i32,
    pub m_xend: i32,
    pub m_ystart: i32,
    pub m_yend: i32,
    pub m_full_xpad: i32,
    pub m_full_ypad: i32,
    pub m_sum_xpad: i32,
    pub m_sum_ypad: i32,
    pub m_align_xpad: i32,
    pub m_align_ypad: i32,
    pub m_align_pix: i32,
    pub m_align_bytes: i32,
    pub m_xreduce_temp: i32,
    pub m_yreduce_temp: i32,
    pub m_ali_filt_size: i32,
    pub m_trim_frac: f32,
    pub m_taper_frac: f32,
    pub m_summing_mode: i32,
    pub m_use_hybrid: i32,
    pub m_num_frames: i32,
    pub m_num_full_saved: i32,
    pub m_bin_sum: i32,
    pub m_bin_align: i32,
    pub m_num_all_vs_all: i32,
    pub m_num_filters: i32,
    pub m_best_filt: i32,
    pub m_max_max_weight: f32,
    pub m_max_shift: i32,
    pub m_dump_ind: i32,
    pub m_wall_full_fft: f64,
    pub m_wall_bin_pad: f64,
    pub m_wall_bin_fft: f64,
    pub m_wall_reduce: f64,
    pub m_wall_shift: f64,
    pub m_wall_start: f64,
    pub m_wall_filter: f64,
    pub m_wall_conj_prod: f64,
    pub m_wall_pre_proc: f64,
    pub m_wall_noise: f64,
    pub m_debug: i32,
    pub m_make_unwgt_sum: i32,
    pub m_unwgt_on_gpu: bool,
    pub m_dump_corrs: bool,
    pub m_dump_ref_corrs: bool,
    pub m_dump_even_odd: bool,
    pub m_picked_best_filt: bool,
    pub m_defer_summing: bool,
    pub m_gpu_aligning: bool,
    pub m_gpu_summing: bool,
    pub m_noise_pad_on_gpu: bool,
    pub m_bin_pad_on_gpu: bool,
    pub m_stack_unpad_on_gpu: bool,
    pub m_gpu_stack_limit: i32,
    pub m_num_stacked_on_gpu: i32,
    pub m_group_size: i32,
    pub m_group_size_initial: i32,
    pub m_kfactor: f32,
    pub m_cum_align_at_end: i32,
    pub m_pick_ratio_crit: f32,
    pub m_pick_diff_crit: f32,
    pub m_failed_often_crit: i32,
    pub m_num_fits: i32,
    pub m_report_times: bool,
    pub m_res_mean_sum: [f32; MAX_FILTERS + 1],
    pub m_res_sdsum: [f32; MAX_FILTERS + 1],
    pub m_res_max_sum: [f32; MAX_FILTERS + 1],
    pub m_max_res_max: [f32; MAX_FILTERS + 1],
    pub m_max_raw_max: [f32; MAX_FILTERS + 1],
    pub m_raw_max_sum: [f32; MAX_FILTERS + 1],
    pub m_pred_mean_sum: [f32; MAX_FILTERS + 1],
    pub m_filt_func: [f32; 8193],
    pub m_filt_delta: f32,
    pub m_gpu_lib_loaded: i32,
    pub m_num_expected_frames: i32,
    pub m_doing_dose_weighting: bool,
    pub m_frame_doses: Vec<f32>,
    pub m_dose_wgt_filter: Vec<f32>,
    pub m_prior_dose_cum: f32,
    pub m_pixel_size: f32,
    pub m_crit_dose_scale: f32,
    pub m_crit_dose_afac: f32,
    pub m_crit_dose_bfac: f32,
    pub m_crit_dose_cfac: f32,
    pub m_dwfdelta: f32,
    pub m_reweight_filt: Vec<f32>,
}

impl Default for FrameAlign {
    fn default() -> Self {
        FrameAlign::new()
    }
}

impl FrameAlign {
    /// C `FrameAlign::FrameAlign()` (`framealign.cpp:128`).
    ///
    /// Initialize all the pointers and call cleanup routine for other
    /// variables.
    pub fn new() -> FrameAlign {
        let mut fa = FrameAlign {
            m_print_func: None,
            m_full_even_sum: Vec::new(),
            m_full_odd_sum: Vec::new(),
            m_unweight_sum: Vec::new(),
            m_align_sum: Vec::new(),
            m_work_full_size: Vec::new(),
            m_saved_full_size: Vec::new(),
            m_saved_full_frame_num: Vec::new(),
            m_saved_bin_pad: Vec::new(),
            m_saved_groups: Vec::new(),
            m_work_bin_pad: Vec::new(),
            m_corr_bin_pad: Vec::new(),
            m_corr_filt_temp: Vec::new(),
            m_reduce_temp: Vec::new(),
            m_shift_temp: Vec::new(),
            m_line_ptrs: 0,
            m_fit_mat: Vec::new(),
            m_fit_work: Vec::new(),
            m_sub_filt_mask: Default::default(),
            m_full_filt_mask: Vec::new(),
            m_temp_sub_filt: Vec::new(),
            m_wrap_temp: Vec::new(),
            m_xshifts: Default::default(),
            m_yshifts: Default::default(),
            m_xall_shifts: Default::default(),
            m_yall_shifts: Default::default(),
            m_xfit_shifts: [[0.; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
            m_yfit_shifts: [[0.; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
            m_xnear_shifts: [0.; MAX_ALL_VS_ALL],
            m_ynear_shifts: [0.; MAX_ALL_VS_ALL],
            m_last_xfit: [[0.; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
            m_last_yfit: [[0.; MAX_ALL_VS_ALL]; MAX_FILTERS + 1],
            m_cumul_xdiff: [0.; MAX_FILTERS + 1],
            m_cumul_ydiff: [0.; MAX_FILTERS + 1],
            m_num_as_best_filt: [0; MAX_FILTERS],
            m_radius2: [0.; MAX_FILTERS],
            m_sigma2: [0.; MAX_FILTERS],
            m_gpu_flags: 0,
            m_flags_for_unpad_call: 0,
            m_even_odd_for_sum_setup: 0,
            m_nx_gain: 0,
            m_ny_gain: 0,
            m_gain_ref: None,
            m_cam_size_x: 0,
            m_cam_size_y: 0,
            m_cam_defects: None,
            m_trunc_limit: 0.,
            m_defect_bin: 0,
            m_noise_length: 0,
            m_frame_type: 0,
            m_stack_type: 0,
            m_nx: 0,
            m_ny: 0,
            m_anti_filt_type: 0,
            m_xstart: 0,
            m_xend: 0,
            m_ystart: 0,
            m_yend: 0,
            m_full_xpad: 0,
            m_full_ypad: 0,
            m_sum_xpad: 0,
            m_sum_ypad: 0,
            m_align_xpad: 0,
            m_align_ypad: 0,
            m_align_pix: 0,
            m_align_bytes: 0,
            m_xreduce_temp: 0,
            m_yreduce_temp: 0,
            m_ali_filt_size: 0,
            m_trim_frac: 0.,
            m_taper_frac: 0.,
            m_summing_mode: 0,
            m_use_hybrid: 0,
            m_num_frames: 0,
            m_num_full_saved: 0,
            m_bin_sum: 0,
            m_bin_align: 0,
            m_num_all_vs_all: 0,
            m_num_filters: 0,
            m_best_filt: 0,
            m_max_max_weight: 0.,
            m_max_shift: 0,
            m_dump_ind: 0,
            m_wall_full_fft: 0.,
            m_wall_bin_pad: 0.,
            m_wall_bin_fft: 0.,
            m_wall_reduce: 0.,
            m_wall_shift: 0.,
            m_wall_start: 0.,
            m_wall_filter: 0.,
            m_wall_conj_prod: 0.,
            m_wall_pre_proc: 0.,
            m_wall_noise: 0.,
            m_debug: 0,
            m_make_unwgt_sum: 0,
            m_unwgt_on_gpu: false,
            m_dump_corrs: false,
            m_dump_ref_corrs: false,
            m_dump_even_odd: false,
            m_picked_best_filt: false,
            m_defer_summing: false,
            m_gpu_aligning: false,
            m_gpu_summing: false,
            m_noise_pad_on_gpu: false,
            m_bin_pad_on_gpu: false,
            m_stack_unpad_on_gpu: false,
            m_gpu_stack_limit: 0,
            m_num_stacked_on_gpu: 0,
            m_group_size: 0,
            m_group_size_initial: 1,
            m_kfactor: 0.,
            m_cum_align_at_end: 0,
            m_pick_ratio_crit: 3.,
            m_pick_diff_crit: 5.,
            m_failed_often_crit: 3,
            m_num_fits: 0,
            m_report_times: false,
            m_res_mean_sum: [0.; MAX_FILTERS + 1],
            m_res_sdsum: [0.; MAX_FILTERS + 1],
            m_res_max_sum: [0.; MAX_FILTERS + 1],
            m_max_res_max: [0.; MAX_FILTERS + 1],
            m_max_raw_max: [0.; MAX_FILTERS + 1],
            m_raw_max_sum: [0.; MAX_FILTERS + 1],
            m_pred_mean_sum: [0.; MAX_FILTERS + 1],
            m_filt_func: [0.; 8193],
            m_filt_delta: 0.,
            m_gpu_lib_loaded: -1,
            m_num_expected_frames: 0,
            m_doing_dose_weighting: false,
            m_frame_doses: Vec::new(),
            m_dose_wgt_filter: Vec::new(),
            m_prior_dose_cum: 0.,
            m_pixel_size: 0.,
            m_crit_dose_scale: 0.,
            m_crit_dose_afac: 0.,
            m_crit_dose_bfac: 0.,
            m_crit_dose_cfac: 0.,
            m_dwfdelta: 0.,
            m_reweight_filt: Vec::new(),
        };
        fa.cleanup();
        fa
    }

    /// C `FrameAlign::setPrintFunc` (`framealign.cpp:170`).
    pub fn set_print_func(&mut self, func: CharArgType) {
        self.m_print_func = func;
        util_set_print_func(func);
        if self.m_gpu_lib_loaded > 0 {
            fgpu_set_print_func(func);
        }
    }

    /// C `getFullWorkArray` (`framealign.h:80`).
    pub fn get_full_work_array(&mut self) -> &mut Vec<f32> {
        &mut self.m_work_full_size
    }

    /// C `getPaddedSumSize` (`framealign.h:97`).
    pub fn get_padded_sum_size(&self) -> i32 {
        (self.m_sum_xpad + 2) * self.m_sum_ypad
    }

    /// Move the buffer a [`FullRef`] names out of `self`, leaving an empty
    /// `Vec` behind, so the caller can borrow a second buffer at the same
    /// time.  [`FrameAlign::put_full`] puts it back.
    fn take_full(&mut self, r: FullRef) -> Vec<f32> {
        match r {
            FullRef::Null | FullRef::Frame => Vec::new(),
            FullRef::Work => std::mem::take(&mut self.m_work_full_size),
            FullRef::Saved(i) => std::mem::take(&mut self.m_saved_full_size[i]),
        }
    }

    fn put_full(&mut self, r: FullRef, v: Vec<f32>) {
        match r {
            FullRef::Null | FullRef::Frame => (),
            FullRef::Work => self.m_work_full_size = v,
            FullRef::Saved(i) => self.m_saved_full_size[i] = v,
        }
    }

    fn take_bin(&mut self, r: BinRef) -> Vec<f32> {
        match r {
            BinRef::Null => Vec::new(),
            BinRef::Work => std::mem::take(&mut self.m_work_bin_pad),
            BinRef::Corr => std::mem::take(&mut self.m_corr_bin_pad),
            BinRef::AlignSum => std::mem::take(&mut self.m_align_sum),
            BinRef::Saved(i) => std::mem::take(&mut self.m_saved_bin_pad[i]),
            BinRef::Group(i) => std::mem::take(&mut self.m_saved_groups[i]),
        }
    }

    fn put_bin(&mut self, r: BinRef, v: Vec<f32>) {
        match r {
            BinRef::Null => (),
            BinRef::Work => self.m_work_bin_pad = v,
            BinRef::Corr => self.m_corr_bin_pad = v,
            BinRef::AlignSum => self.m_align_sum = v,
            BinRef::Saved(i) => self.m_saved_bin_pad[i] = v,
            BinRef::Group(i) => self.m_saved_groups[i] = v,
        }
    }

    /// C `FrameAlign::testAndCleanup` (`framealign.cpp:614`).
    ///
    /// Cleanup on failure of the given test.
    pub fn test_and_cleanup(&mut self, failed: bool) -> i32 {
        if !failed {
            return 0;
        }
        self.cleanup();
        2
    }

    /// C `FrameAlign::cleanup` (`framealign.cpp:625`).
    ///
    /// Free all memory and reset sizes etc.
    pub fn cleanup(&mut self) {
        self.m_full_odd_sum = Vec::new();
        self.m_full_even_sum = Vec::new();
        self.m_unweight_sum = Vec::new();
        self.m_align_sum = Vec::new();
        self.m_work_bin_pad = Vec::new();
        self.m_work_full_size = Vec::new();
        self.m_corr_bin_pad = Vec::new();
        self.m_corr_filt_temp = Vec::new();
        self.m_shift_temp = Vec::new();
        self.m_line_ptrs = 0;
        self.m_reduce_temp = Vec::new();
        self.m_full_filt_mask = Vec::new();
        self.m_temp_sub_filt = Vec::new();
        self.m_wrap_temp = Vec::new();
        self.m_fit_mat = Vec::new();
        self.m_fit_work = Vec::new();
        self.m_saved_bin_pad = Vec::new();
        self.m_saved_full_size = Vec::new();
        self.m_saved_groups = Vec::new();
        for ind in 0..MAX_FILTERS {
            self.m_sub_filt_mask[ind] = Vec::new();
        }
        self.m_num_filters = 0;
        self.m_trim_frac = 0.;
        self.m_taper_frac = 0.;
        self.m_cum_align_at_end = 0;
        self.m_ali_filt_size = 0;
        self.m_use_hybrid = 0;
        self.m_full_xpad = 0;
        self.m_full_ypad = 0;
        // `framealign.cpp:723` is `mAlignXpad = mAlignYpad - 0;` -- a typo for
        // `mAlignXpad = mAlignYpad = 0`, which leaves mAlignXpad holding the
        // previous mAlignYpad and never zeroes either.  Reproduced as written.
        self.m_align_xpad = self.m_align_ypad;
        self.m_sum_xpad = 0;
        self.m_sum_ypad = 0;
        self.m_xreduce_temp = 0;
        self.m_yreduce_temp = 0;
        self.m_num_all_vs_all = -1;
        for ind in 0..=MAX_FILTERS {
            self.m_xshifts[ind] = Vec::new();
            self.m_yshifts[ind] = Vec::new();
            if ind < MAX_FILTERS {
                self.m_xall_shifts[ind] = Vec::new();
                self.m_yall_shifts[ind] = Vec::new();
            }
        }
        if self.m_gpu_flags != 0 {
            fgpu_cleanup();
        }
        self.m_frame_doses = Vec::new();
        self.m_dose_wgt_filter = Vec::new();
        self.m_gpu_flags = 0;
    }

    /// C `FrameAlign::gpuAvailable` (`framealign.cpp:684`).
    ///
    /// Find out if GPU is available.  The whole `_WIN32 && DELAY_LOAD_FGPU`
    /// half of this function loads `FrameGPU.dll`; on this platform the
    /// `fgpu*` symbols are linked directly from `nogpuframe.cpp`.
    pub fn gpu_available(&mut self, n_gpu: i32, memory: &mut f32, debug: i32) -> i32 {
        let err = fgpu_gpu_available(n_gpu, memory, debug);
        if err == 0 {
            util_print(
                "GPU is not available%s\n",
                &[CArg::Str(if debug != 0 {
                    ""
                } else {
                    "; run with debugging output for details"
                })],
            );
        }
        err
    }

    /// C `FrameAlign::leastCommonMultiple` (`framealign.cpp:2925`).
    ///
    /// Return smallest multiple of the two numbers that includes all their
    /// divisors.
    pub fn least_common_multiple(&self, num1: i32, mut num2: i32) -> i32 {
        let mut fac = 64;
        while fac > 1 {
            if num2 % fac == 0 && num1 % fac == 0 {
                num2 /= fac;
            }
            fac -= 1;
        }
        num1 * num2
    }

    /// C `FrameAlign::getPadSizesBytes` (`framealign.cpp:3230`).
    ///
    /// Get sizes in bytes for full, sum, and align images.
    pub fn get_pad_sizes_bytes(
        nx: i32,
        ny: i32,
        full_taper_frac: f32,
        sum_bin: i32,
        align_bin: i32,
        full_pad_size: &mut f32,
        sum_pad_size: &mut f32,
        align_pad_size: &mut f32,
    ) {
        *full_pad_size =
            4. * (1. + 2. * full_taper_frac) * (1. + 2. * full_taper_frac) * nx as f32 * ny as f32;
        *sum_pad_size = *full_pad_size / (sum_bin * sum_bin) as f32;
        *align_pad_size =
            (4. * nx as f64 * ny as f64) as f32 / (align_bin as f32 * align_bin as f32);
    }

    /// C `FrameAlign::gpuMemoryNeeds` (`framealign.cpp:3243`).
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_memory_needs(
        full_pad_size: f32,
        sum_pad_size: f32,
        align_pad_size: f32,
        num_all_vs_all: i32,
        nz_align: i32,
        refine_at_end: i32,
        group_size: i32,
        need_for_gpu_sum: &mut f32,
        need_for_gpu_ali: &mut f32,
    ) {
        let num_hold_align: i32;
        *need_for_gpu_sum = full_pad_size
            + sum_pad_size
            + 2. * if full_pad_size < sum_pad_size {
                sum_pad_size
            } else {
                full_pad_size
            };
        if num_all_vs_all == 0 && refine_at_end == 0 {
            num_hold_align = 6;
        } else if refine_at_end != 0 {
            num_hold_align = 6 + nz_align;
        } else {
            num_hold_align = 4 + if num_all_vs_all < nz_align {
                num_all_vs_all
            } else {
                nz_align
            };
        }
        let mut num_hold_align = num_hold_align;
        if group_size > 1 {
            if refine_at_end != 0 {
                num_hold_align += if num_all_vs_all < nz_align {
                    num_all_vs_all
                } else {
                    nz_align
                };
            } else {
                num_hold_align += group_size;
            }
        }
        *need_for_gpu_ali = num_hold_align as f32 * align_pad_size;
    }

    /// C `FrameAlign::totalMemoryNeeds` (`framealign.cpp:3279`).
    ///
    /// Return the computer memory needed based on the conditions, in
    /// gigabytes, set the flag for whether sums can be made in one pass, and
    /// return the maximum number of full frames that need to be held in memory
    /// somewhere.
    #[allow(clippy::too_many_arguments)]
    pub fn total_memory_needs(
        full_pad_size: f32,
        full_data_size: i32,
        sum_pad_size: f32,
        align_pad_size: f32,
        num_all_vs_all: i32,
        nz_align: i32,
        refine_at_end: i32,
        num_bin_tests: i32,
        num_filt_tests: i32,
        hybrid_shifts: i32,
        group_size: i32,
        do_spline: i32,
        gpu_flags: i32,
        defer_sum: i32,
        test_mode: i32,
        start_assess: i32,
        sum_in_one_pass: &mut bool,
        num_hold_full: &mut i32,
    ) -> f32 {
        let stack_limit: i32;
        let mem_tot: f32;

        *sum_in_one_pass = num_bin_tests == 1 && test_mode == 0 && start_assess < 0;

        *num_hold_full = if num_all_vs_all < nz_align {
            num_all_vs_all
        } else {
            nz_align
        };
        let mut num_hold_align = *num_hold_full;
        if num_all_vs_all != 0 {
            if refine_at_end != 0 {
                num_hold_align = nz_align;
            }
            if group_size > 1 && refine_at_end != 0 {
                num_hold_align += if num_all_vs_all < nz_align {
                    num_all_vs_all
                } else {
                    nz_align
                };
            } else if group_size > 1 {
                num_hold_align += group_size;
            }
        } else if refine_at_end != 0 {
            num_hold_align = nz_align;
        }
        if gpu_flags & GPU_FOR_ALIGNING != 0 {
            num_hold_align = 0;
        }

        if (hybrid_shifts == 0 && num_filt_tests > 1 && *sum_in_one_pass && num_all_vs_all != 0)
            || refine_at_end != 0
            || defer_sum != 0
            || do_spline != 0
        {
            *num_hold_full = nz_align;
        }
        if gpu_flags & STACK_FULL_ON_GPU != 0 {
            if gpu_flags & GPU_STACK_LIMITED != 0 {
                stack_limit = (gpu_flags >> GPU_STACK_LIM_SHIFT) & GPU_STACK_LIM_MASK;
                *num_hold_full = if 0 > *num_hold_full - stack_limit {
                    0
                } else {
                    *num_hold_full - stack_limit
                };
            } else {
                *num_hold_full = 0;
            }
        }
        if (num_bin_tests > 1 || test_mode != 0) && num_all_vs_all != 0 && start_assess < 0 {
            *num_hold_full = 0;
        }
        mem_tot = ((2. * sum_pad_size as f64
            + (num_hold_align as f64 + 4.) * align_pad_size as f64
            + full_pad_size as f64
            + *num_hold_full as f64 * full_data_size as f64 * full_pad_size as f64 / 4.)
            / (1024. * 1024. * 1024.)) as f32;
        mem_tot
    }

    /// C `FrameAlign::preprocPadGpuMemoryFits` (`framealign.cpp:3435`).
    ///
    /// For one set of possible operations on GPU, determine if the needed
    /// memory fits within `freeMem`; return the needed amount in bytes in
    /// `needsMem`.
    #[allow(clippy::too_many_arguments)]
    pub fn preproc_pad_gpu_memory_fits(
        unpadded_x: i32,
        unpadded_y: i32,
        data_size: i32,
        binning: i32,
        has_gain: bool,
        has_defect: bool,
        has_trunc: bool,
        do_pre_proc: bool,
        do_noise: bool,
        do_bin_pad: bool,
        free_mem: f32,
        needs_mem: &mut f32,
    ) -> bool {
        let needs_proc = has_gain || has_defect || has_trunc;
        let pass_size = if needs_proc && !do_pre_proc {
            std::mem::size_of::<f32>() as i32
        } else {
            data_size
        };
        let unpad_bytes = (unpadded_x * unpadded_y) as f32;
        *needs_mem = 0.;
        if do_pre_proc && has_gain {
            *needs_mem += 4. * unpad_bytes;
        }
        if do_pre_proc && has_defect {
            *needs_mem += unpad_bytes;
        }
        if do_noise || do_bin_pad {
            *needs_mem += pass_size as f32 * unpad_bytes;
        }
        if do_bin_pad && binning > 1 {
            *needs_mem += 4. * unpad_bytes / binning as f32;
        }
        *needs_mem < free_mem
    }

    /// C `FrameAlign::findPreprocPadGpuFlags` (`framealign.cpp:3332`).
    #[allow(clippy::too_many_arguments)]
    pub fn find_preproc_pad_gpu_flags(
        unpadded_x: i32,
        unpadded_y: i32,
        data_size: i32,
        binning: i32,
        has_gain: bool,
        has_defect: bool,
        has_trunc: bool,
        num_expected: i32,
        free_mem: f32,
        stack_margin: f32,
        in_flags: i32,
        out_flags: &mut i32,
    ) -> f32 {
        let needs_proc = has_gain || has_defect || has_trunc;
        let do_align = (in_flags & GPU_FOR_ALIGNING) != 0;
        let do_sum = (in_flags & GPU_FOR_SUMMING) != 0;
        let pass_size: i32;
        let max_added: i32;
        let stack_limit: i32;
        let mut temp_needs = 0.0f32;
        let unpad_bytes: f32;
        let preproc_flags = GPU_DO_PREPROCESS
            | if has_gain { GPU_DO_GAIN_NORM } else { 0 }
            | if has_defect { GPU_CORRECT_DEFECTS } else { 0 };
        *out_flags = in_flags;

        if do_align && !do_sum {
            if !needs_proc
                || !Self::preproc_pad_gpu_memory_fits(
                    unpadded_x,
                    unpadded_y,
                    data_size,
                    binning,
                    has_gain,
                    has_defect,
                    has_trunc,
                    true,
                    false,
                    true,
                    free_mem,
                    &mut temp_needs,
                )
            {
                if Self::preproc_pad_gpu_memory_fits(
                    unpadded_x,
                    unpadded_y,
                    data_size,
                    binning,
                    has_gain,
                    has_defect,
                    has_trunc,
                    false,
                    false,
                    true,
                    free_mem,
                    &mut temp_needs,
                ) {
                    *out_flags |= GPU_DO_BIN_PAD;
                    return temp_needs;
                }
                return 0.;
            } else {
                *out_flags |= GPU_DO_BIN_PAD | preproc_flags;
                return temp_needs;
            }
        }

        if !needs_proc
            || !Self::preproc_pad_gpu_memory_fits(
                unpadded_x,
                unpadded_y,
                data_size,
                binning,
                has_gain,
                has_defect,
                has_trunc,
                true,
                true,
                false,
                free_mem,
                &mut temp_needs,
            )
        {
            if Self::preproc_pad_gpu_memory_fits(
                unpadded_x,
                unpadded_y,
                data_size,
                binning,
                has_gain,
                has_defect,
                has_trunc,
                false,
                true,
                false,
                free_mem,
                &mut temp_needs,
            ) {
                if do_align
                    && Self::preproc_pad_gpu_memory_fits(
                        unpadded_x,
                        unpadded_y,
                        data_size,
                        binning,
                        has_gain,
                        has_defect,
                        has_trunc,
                        false,
                        true,
                        true,
                        free_mem,
                        &mut temp_needs,
                    )
                {
                    *out_flags |= GPU_DO_BIN_PAD | GPU_DO_NOISE_TAPER;
                } else {
                    *out_flags |= GPU_DO_NOISE_TAPER;
                    return temp_needs;
                }
            } else {
                return 0.;
            }
        } else if do_align
            && Self::preproc_pad_gpu_memory_fits(
                unpadded_x,
                unpadded_y,
                data_size,
                binning,
                has_gain,
                has_defect,
                has_trunc,
                true,
                true,
                true,
                free_mem,
                &mut temp_needs,
            )
        {
            *out_flags |= GPU_DO_BIN_PAD | GPU_DO_NOISE_TAPER | preproc_flags;
        } else {
            *out_flags |= GPU_DO_NOISE_TAPER | preproc_flags;
            return temp_needs;
        }

        pass_size = if needs_proc && (*out_flags & preproc_flags) == 0 {
            std::mem::size_of::<f32>() as i32
        } else {
            data_size
        };
        unpad_bytes = (pass_size * unpadded_x * unpadded_y) as f32;
        max_added = ((free_mem - temp_needs - stack_margin) / unpad_bytes) as i32;
        if max_added >= num_expected - 1 {
            temp_needs += (num_expected - 1) as f32 * unpad_bytes;
            *out_flags |= STACK_FULL_ON_GPU;
        } else if max_added > 0 {
            stack_limit = max_added + 1;
            temp_needs += max_added as f32 * unpad_bytes;
            *out_flags |=
                STACK_FULL_ON_GPU | GPU_STACK_LIMITED | (stack_limit << GPU_STACK_LIM_SHIFT);
        }
        temp_needs
    }

    /// C `FrameAlign::setTruncationLimit` (`framealign.cpp:3463`).
    ///
    /// Set the truncation limit to use: pass it on if >= 0, or find mean/SD
    /// and get a threshold if it is a negative number of SDs.  Returns 1 for
    /// error making line pointers, or 2 for error getting sample arrays.
    pub fn set_truncation_limit(
        &self,
        array: &[u8],
        nx: i32,
        ny: i32,
        use_mode: i32,
        trunc_limit: f32,
        trunc_use: &mut f32,
    ) -> i32 {
        let err: i32;
        let typ = type_for_sample_mean(use_mode);
        let mut data_size = 0;
        let mut temp = 0;
        let num_sample = 20000;
        let sample: f32;
        let mut mean = 0.;
        let mut sd = 0.;
        let trim = if nx < ny { nx } else { ny } / 20;
        let nx_use = nx - 2 * trim;
        let ny_use = ny - 2 * trim;
        *trunc_use = trunc_limit;
        if trunc_limit >= 0. {
            return 0;
        }
        data_size_for_mode(use_mode, &mut data_size, &mut temp);
        // C `makeLinePointers(array, nx, ny, dataSize)`: a row-pointer array,
        // which is a slice of row slices here.
        let row = (nx * data_size) as usize;
        if array.len() < row * ny as usize {
            return 1;
        }
        let line_ptrs: Vec<&[u8]> = (0..ny as usize)
            .map(|i| &array[i * row..(i + 1) * row])
            .collect();
        sample = (if num_sample < nx_use * ny_use {
            num_sample
        } else {
            nx_use * ny_use
        }) as f32
            / (nx_use * ny_use) as f32;
        err = sample_mean_sd(
            Some(&line_ptrs),
            typ,
            nx,
            ny,
            sample,
            trim,
            trim,
            nx_use,
            ny_use,
            Some(&mut mean),
            Some(&mut sd),
        );
        if err != 0 {
            return 2;
        }
        *trunc_use = mean - trunc_limit * sd;
        0
    }
}

impl FrameAlign {
    /// C `FrameAlign::initialize` (`framealign.cpp:182`).
    ///
    /// Initialize before sending a set of frames.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        &mut self,
        bin_sum: i32,
        bin_align: i32,
        trim_frac: f32,
        mut num_all_vs_all: i32,
        cum_align_at_end: i32,
        use_hybrid: i32,
        defer_sum: i32,
        group_size: i32,
        nx: i32,
        ny: i32,
        pad_frac: f32,
        taper_frac: f32,
        anti_filt_type: i32,
        radius1: f32,
        radius2: &[f32],
        sigma1: f32,
        sigma2: &[f32],
        num_filters: i32,
        max_shift: i32,
        k_factor: f32,
        max_max_weight: f32,
        summing_mode: i32,
        expected_z: i32,
        make_unwgt_sum: i32,
        mut gpu_flags: i32,
        debug: i32,
    ) -> i32 {
        let align_xpad: i32;
        let align_ypad: i32;
        let sum_xpad: i32;
        let sum_ypad: i32;
        let full_xpad: i32;
        let full_ypad: i32;
        let mut ind: i32;
        let mut filt: i32;
        let mut divisor: i32;
        let nice_limit: i32;
        let mut nx_pad: i32;
        let mut ny_pad: i32;
        let nx_trim: i32;
        let ny_trim: i32;
        let nx_use: i32;
        let ny_use: i32;
        let expect_stack: i32;
        let gpu_stack_limit: i32;
        let mut ali_filt_size: i32;
        let min_filt_size: i32 = 128;
        let mut x_reduce_temp = 0;
        let mut y_reduce_temp = 0;
        let do_bin_pad = trim_frac != 0. || taper_frac != 0.;

        // Make just a full filter mask with high-frequency included if only one
        // filter, no GPU, and not refining at end
        let just_full_filt =
            num_filters == 1 && cum_align_at_end == 0 && (gpu_flags & GPU_FOR_ALIGNING) == 0;
        let nice_gpu_limit = 5;
        let noise_pad_on_gpu: bool;
        let bin_pad_on_gpu: bool;
        let stack_on_gpu: bool;

        self.m_debug = debug % 10;
        self.m_report_times = (debug / 10) % 10 != 0;
        self.m_dump_corrs = (debug / 100) % 10 != 0;
        self.m_dump_ref_corrs = (debug / 1000) % 10 != 0;
        self.m_dump_even_odd = (debug / 10000) % 10 != 0;
        if num_all_vs_all < 2 + group_size {
            num_all_vs_all = 0;
        }
        if num_all_vs_all > MAX_ALL_VS_ALL as i32
            || num_filters > MAX_FILTERS as i32
            || (num_all_vs_all == 0 && num_filters > 1)
            || (gpu_flags != 0 && !do_bin_pad)
        {
            return 1;
        }

        filt = 0;
        while filt < num_filters {
            let f = filt as usize;
            self.m_xall_shifts[f] = vec![0.; (num_all_vs_all * num_all_vs_all) as usize];
            self.m_yall_shifts[f] = vec![0.; (num_all_vs_all * num_all_vs_all) as usize];
            self.m_radius2[f] = radius2[f];
            self.m_sigma2[f] = sigma2[f];
            filt += 1;
        }
        filt = 0;
        while filt <= num_filters {
            let f = filt as usize;
            self.m_xshifts[f] = Vec::new();
            self.m_yshifts[f] = Vec::new();
            self.m_res_mean_sum[f] = 0.;
            self.m_res_sdsum[f] = 0.;
            self.m_max_res_max[f] = 0.;
            self.m_res_max_sum[f] = 0.;
            self.m_max_raw_max[f] = 0.;
            self.m_raw_max_sum[f] = 0.;
            self.m_cumul_xdiff[f] = 0.;
            self.m_cumul_ydiff[f] = 0.;
            self.m_pred_mean_sum[f] = 0.;
            filt += 1;
        }

        divisor = bin_sum;
        if !do_bin_pad {
            divisor = self.least_common_multiple(bin_sum, bin_align);
        }
        divisor *= 2;
        nice_limit = nice_fft_limit();
        let nice_limit = if gpu_flags != 0 {
            nice_gpu_limit
        } else {
            nice_limit
        };

        // Get size of full as a multiple of the necessary divisor
        nx_pad = (if 32. > pad_frac * nx as f32 {
            32.
        } else {
            pad_frac * nx as f32
        }) as i32;
        ny_pad = (if 32. > pad_frac * ny as f32 {
            32.
        } else {
            pad_frac * ny as f32
        }) as i32;
        full_xpad = divisor * ((nx + 2 * nx_pad + divisor - 1) / divisor);
        full_ypad = divisor * ((ny + 2 * ny_pad + divisor - 1) / divisor);
        let full_xpad = nice_frame(full_xpad, divisor, nice_limit);
        let full_ypad = nice_frame(full_ypad, divisor, nice_limit);

        self.m_anti_filt_type = anti_filt_type;
        if do_bin_pad && bin_align > 1 {
            let mut width = 0;
            if select_zoom_filter(anti_filt_type, 1. / bin_align as f64, &mut width) != 0 {
                return 3;
            }
        }

        // Manage things when there is change in number of all-vs-all
        if num_all_vs_all != self.m_num_all_vs_all
            || summing_mode != self.m_summing_mode
            || cum_align_at_end != self.m_cum_align_at_end
            || use_hybrid != self.m_use_hybrid
            || group_size != self.m_group_size_initial
        {
            self.m_saved_bin_pad.clear();
            self.m_saved_full_size.clear();
            self.m_saved_full_frame_num.clear();
            self.m_saved_groups.clear();
            self.m_fit_mat = Vec::new();
            self.m_fit_work = Vec::new();
            if num_all_vs_all != 0 {
                self.m_fit_mat = vec![
                    0.;
                    (((num_all_vs_all + 3) * (num_all_vs_all - 1) * num_all_vs_all) / 2)
                        as usize
                ];
                self.m_fit_work = vec![0.; ((num_all_vs_all + 2) * (num_all_vs_all + 2)) as usize];
            }
        }

        if full_xpad != self.m_full_xpad
            || full_ypad != self.m_full_ypad
            || num_all_vs_all != self.m_num_all_vs_all
            || summing_mode != self.m_summing_mode
        {
            self.m_shift_temp = Vec::new();
            self.m_work_full_size = vec![0.; ((full_xpad + 2) * full_ypad) as usize];
            self.m_shift_temp = vec![
                0.;
                (2 * (if nx > ny { nx } else { ny }) + (full_xpad - nx) + (full_ypad - ny))
                    as usize
            ];
            self.m_line_ptrs = full_ypad;
            self.m_full_xpad = full_xpad;
            self.m_full_ypad = full_ypad;
        }

        // And size of final sum(s)
        sum_xpad = full_xpad / bin_sum;
        sum_ypad = full_ypad / bin_sum;
        if sum_xpad != self.m_sum_xpad
            || sum_ypad != self.m_sum_ypad
            || summing_mode != self.m_summing_mode
        {
            self.m_full_even_sum = Vec::new();
            self.m_full_odd_sum = Vec::new();
            if summing_mode <= 0 {
                self.m_full_even_sum = vec![0.; ((sum_xpad + 2) * sum_ypad) as usize];
                self.m_full_odd_sum = vec![0.; ((sum_xpad + 2) * sum_ypad) as usize];
            }
        }

        if sum_xpad != self.m_sum_xpad
            || sum_ypad != self.m_sum_ypad
            || make_unwgt_sum != self.m_make_unwgt_sum
        {
            self.m_unweight_sum = Vec::new();
            if make_unwgt_sum != 0 {
                self.m_unweight_sum = vec![0.; ((sum_xpad + 2) * sum_ypad) as usize];
            }
        }
        self.m_sum_xpad = sum_xpad;
        self.m_sum_ypad = sum_ypad;

        // And size of align sum
        if !do_bin_pad {
            align_xpad = full_xpad / bin_align;
            align_ypad = full_ypad / bin_align;
        } else {
            nx_trim = (trim_frac * nx as f32) as i32;
            ny_trim = (trim_frac * ny as f32) as i32;
            nx_use = 2 * bin_align * ((nx - 2 * nx_trim) / (2 * bin_align));
            ny_use = 2 * bin_align * ((ny - 2 * ny_trim) / (2 * bin_align));
            if self.m_debug != 0 {
                util_print(
                    "nxTrim = %d  nyTrim = %d nxUse = %d  nyUse = %d\n",
                    &[
                        CArg::Int(nx_trim as i64),
                        CArg::Int(ny_trim as i64),
                        CArg::Int(nx_use as i64),
                        CArg::Int(ny_use as i64),
                    ],
                );
            }
            self.m_xstart = (nx - nx_use) / 2;
            self.m_xend = self.m_xstart + nx_use - 1;
            self.m_ystart = (ny - ny_use) / 2;
            self.m_yend = self.m_ystart + ny_use - 1;
            nx_pad = (if 16. > pad_frac * nx_use as f32 / bin_align as f32 {
                16.
            } else {
                pad_frac * nx_use as f32 / bin_align as f32
            }) as i32;
            ny_pad = (if 16. > pad_frac * ny_use as f32 / bin_align as f32 {
                16.
            } else {
                pad_frac * ny_use as f32 / bin_align as f32
            }) as i32;
            let _ = (nx_pad, ny_pad);
            divisor = 2;
            let mut ax = nx_use / bin_align;
            let mut ay = ny_use / bin_align;
            if gpu_flags & GPU_FOR_ALIGNING != 0 {
                divisor = NICE_GPU_DIVISOR;
                ax = divisor * ((ax + divisor - 1) / divisor);
                ay = divisor * ((ay + divisor - 1) / divisor);
            }
            align_xpad = nice_frame(ax, divisor, nice_limit);
            align_ypad = nice_frame(ay, divisor, nice_limit);
        }
        if self.m_debug != 0 {
            util_print(
                "fullXpad = %d  fullYpad = %d  alignXpad = %d  alignYpad = %d\n",
                &[
                    CArg::Int(full_xpad as i64),
                    CArg::Int(full_ypad as i64),
                    CArg::Int(align_xpad as i64),
                    CArg::Int(align_ypad as i64),
                ],
            );
        }

        // Clean up extra arrays if no trimming now and there was previously
        if (self.m_trim_frac != 0. || self.m_taper_frac != 0.) && !do_bin_pad {
            self.m_saved_bin_pad.clear();
            self.m_corr_bin_pad = Vec::new();
        }

        // Allocate the align arrays if needed
        self.m_align_pix = (align_xpad + 2) * align_ypad;
        self.m_align_bytes = self.m_align_pix * 4;
        if align_xpad != self.m_align_xpad
            || align_ypad != self.m_align_ypad
            || num_all_vs_all != self.m_num_all_vs_all
        {
            self.m_align_sum = vec![0.; self.m_align_pix as usize];
            self.m_work_bin_pad = vec![0.; self.m_align_pix as usize];
            self.m_corr_bin_pad = vec![0.; self.m_align_pix as usize];
            self.m_saved_bin_pad.clear();
        }
        ali_filt_size = if min_filt_size > 4 * (max_shift / bin_align) {
            min_filt_size
        } else {
            4 * (max_shift / bin_align)
        };
        ali_filt_size = if ali_filt_size < align_xpad {
            ali_filt_size
        } else {
            align_xpad
        };
        ali_filt_size = if ali_filt_size < align_ypad {
            ali_filt_size
        } else {
            align_ypad
        };
        ali_filt_size = nice_frame(ali_filt_size, 2, nice_limit);

        // Manage filter mask arrays
        if num_filters != self.m_num_filters
            || align_xpad != self.m_align_xpad
            || align_ypad != self.m_align_ypad
            || ali_filt_size != self.m_ali_filt_size
        {
            self.m_full_filt_mask = vec![0.; self.m_align_pix as usize];
            self.m_corr_filt_temp = Vec::new();
            self.m_temp_sub_filt = Vec::new();
            self.m_wrap_temp = Vec::new();
            ind = 0;
            while ind <= self.m_num_filters {
                if (ind as usize) < MAX_FILTERS {
                    self.m_sub_filt_mask[ind as usize] = Vec::new();
                }
                ind += 1;
            }
            if !just_full_filt {
                self.m_corr_filt_temp = vec![0.; self.m_align_pix as usize];
                self.m_temp_sub_filt = vec![0.; ((ali_filt_size + 2) * ali_filt_size) as usize];
                self.m_wrap_temp = vec![0.; ((ali_filt_size + 2) * ali_filt_size) as usize];
                ind = 0;
                while ind < num_filters {
                    self.m_sub_filt_mask[ind as usize] =
                        vec![0.; ((ali_filt_size + 2) * ali_filt_size) as usize];
                    ind += 1;
                }
            }
        }
        self.m_align_xpad = align_xpad;
        self.m_align_ypad = align_ypad;

        // And construct the filter masks.  Take square root of the full filter
        // as it just gets applied to each stored FFT before correlation
        let mut filt_delta = 0.;
        xcorr_set_ctf(
            sigma1,
            sigma2[0] * bin_align as f32,
            radius1,
            if !just_full_filt {
                0.71f32
            } else {
                radius2[0] * bin_align as f32
            },
            &mut self.m_filt_func,
            self.m_align_xpad,
            self.m_align_ypad,
            &mut filt_delta,
        );
        self.m_filt_delta = filt_delta;
        for ind in 0..8193 {
            self.m_filt_func[ind] = self.m_filt_func[ind].sqrt();
        }
        for ind in 0..self.m_align_pix as usize {
            self.m_full_filt_mask[ind] = 1.;
        }
        let ctf = self.m_filt_func;
        xcorr_filter_part(
            FilterIn::InPlace,
            &mut self.m_full_filt_mask,
            align_xpad,
            align_ypad,
            &ctf,
            self.m_filt_delta,
        );
        self.m_full_filt_mask[0] = 0.;

        filt = 0;
        while filt < num_filters && !just_full_filt {
            let f = filt as usize;
            let mut delta = 0.;
            xcorr_set_ctf(
                0.,
                sigma2[f] * bin_align as f32,
                0.,
                radius2[f] * bin_align as f32,
                &mut self.m_filt_func,
                ali_filt_size,
                ali_filt_size,
                &mut delta,
            );
            self.m_filt_delta = delta;
            for ind in 0..((ali_filt_size + 2) * ali_filt_size) as usize {
                self.m_sub_filt_mask[f][ind] = 1.;
            }
            let ctf = self.m_filt_func;
            xcorr_filter_part(
                FilterIn::InPlace,
                &mut self.m_sub_filt_mask[f],
                ali_filt_size,
                ali_filt_size,
                &ctf,
                self.m_filt_delta,
            );
            filt += 1;
        }

        //  Now manage the reduction temp array
        if bin_sum > 1 {
            x_reduce_temp = if sum_xpad > x_reduce_temp {
                sum_xpad
            } else {
                x_reduce_temp
            };
            y_reduce_temp = if sum_ypad > y_reduce_temp {
                sum_ypad
            } else {
                y_reduce_temp
            };
        }
        if !do_bin_pad {
            x_reduce_temp = if align_xpad > x_reduce_temp {
                align_xpad
            } else {
                x_reduce_temp
            };
            y_reduce_temp = if align_ypad > y_reduce_temp {
                align_ypad
            } else {
                y_reduce_temp
            };
        }
        if x_reduce_temp != self.m_xreduce_temp || y_reduce_temp != self.m_yreduce_temp {
            self.m_reduce_temp = Vec::new();
            if x_reduce_temp != 0 {
                self.m_reduce_temp = vec![0.; ((x_reduce_temp + 2) * y_reduce_temp) as usize];
            }
        }
        self.m_xreduce_temp = x_reduce_temp;
        self.m_yreduce_temp = y_reduce_temp;
        self.m_defer_summing =
            (cum_align_at_end != 0 || (use_hybrid == 0 && num_filters > 1) || defer_sum != 0)
                && summing_mode == 0;

        // Do not use GPU for group size above the limit
        // If GPU was used and is not going to be, clean it up
        if gpu_flags != 0 && group_size > 5 && (gpu_flags & GPU_FOR_ALIGNING) != 0 {
            gpu_flags = 0;
        }
        if self.m_gpu_flags != 0 && gpu_flags == 0 {
            fgpu_cleanup();
        }

        // See which GPU components are not being used, and clear the component
        // that is not needed immediately
        self.m_gpu_aligning = (gpu_flags & GPU_FOR_ALIGNING) != 0 && summing_mode >= 0;
        self.m_gpu_summing = (gpu_flags & GPU_FOR_SUMMING) != 0 && summing_mode <= 0;
        noise_pad_on_gpu =
            do_bin_pad && self.m_gpu_summing && (gpu_flags & GPU_DO_NOISE_TAPER) != 0;
        bin_pad_on_gpu = do_bin_pad && self.m_gpu_aligning && (gpu_flags & GPU_DO_BIN_PAD) != 0;
        stack_on_gpu = noise_pad_on_gpu && bin_pad_on_gpu && (gpu_flags & STACK_FULL_ON_GPU) != 0;
        gpu_stack_limit = (gpu_flags >> GPU_STACK_LIM_SHIFT) & GPU_STACK_LIM_MASK;

        self.m_num_expected_frames = expected_z;
        expect_stack = if num_all_vs_all < expected_z {
            num_all_vs_all
        } else {
            expected_z
        };
        let mut expect_stack = expect_stack;
        if cum_align_at_end != 0 {
            expect_stack = expected_z;
        }
        if gpu_flags != 0 {
            self.m_flags_for_unpad_call = (if noise_pad_on_gpu {
                GPU_DO_NOISE_TAPER
            } else {
                0
            }) | (if bin_pad_on_gpu { GPU_DO_BIN_PAD } else { 0 })
                | (if stack_on_gpu { STACK_FULL_ON_GPU } else { 0 })
                | (if stack_on_gpu && gpu_stack_limit > 0 {
                    GPU_STACK_LIMITED
                } else {
                    0
                })
                | (gpu_flags & (GPU_AVG_SUPER_2X | GPU_AVG_SUPER_4X));
            fgpu_set_unpadded_size(
                nx,
                ny,
                self.m_flags_for_unpad_call,
                (if self.m_debug != 0 { 1 } else { 0 })
                    + (if self.m_report_times { 10 } else { 0 }),
            );
            if !self.m_gpu_summing || self.m_defer_summing {
                fgpu_clean_sum_items();
            }
            if !self.m_gpu_aligning {
                fgpu_clean_align_items();
            }

            // Set up aligning unconditionally
            if self.m_gpu_aligning
                && fgpu_setup_aligning(
                    align_xpad,
                    align_ypad,
                    if self.m_gpu_summing { sum_xpad } else { 0 },
                    if self.m_gpu_summing { sum_ypad } else { 0 },
                    &self.m_full_filt_mask,
                    ali_filt_size,
                    group_size,
                    expect_stack,
                    cum_align_at_end,
                ) != 0
            {
                gpu_flags = 0;
            }

            // Set up summing unless it is deferred
            self.m_even_odd_for_sum_setup =
                (if gpu_flags & GPU_DO_EVEN_ODD != 0 {
                    1
                } else {
                    0
                }) + (if (gpu_flags & GPU_DO_UNWGT_SUM) != 0 && make_unwgt_sum != 0 {
                    2
                } else {
                    0
                });
            if gpu_flags != 0
                && self.m_gpu_summing
                && !self.m_defer_summing
                && fgpu_setup_summing(
                    full_xpad,
                    full_ypad,
                    sum_xpad,
                    sum_ypad,
                    self.m_even_odd_for_sum_setup,
                ) != 0
            {
                fgpu_cleanup();
                gpu_flags = 0;
            }
        }

        // Save all members for current state
        self.m_gpu_flags = gpu_flags;
        self.m_gpu_aligning = (gpu_flags & GPU_FOR_ALIGNING) != 0 && summing_mode >= 0;
        self.m_gpu_summing = (gpu_flags & GPU_FOR_SUMMING) != 0 && summing_mode <= 0;
        self.m_noise_pad_on_gpu = gpu_flags != 0 && noise_pad_on_gpu;
        self.m_bin_pad_on_gpu = gpu_flags != 0 && bin_pad_on_gpu;
        self.m_stack_unpad_on_gpu = gpu_flags != 0 && stack_on_gpu;
        self.m_unwgt_on_gpu =
            self.m_gpu_summing && (gpu_flags & GPU_DO_UNWGT_SUM) != 0 && make_unwgt_sum != 0;

        // Set this to 0 if no stacking, but it also has to be 0 to indicate no limit
        self.m_gpu_stack_limit = if self.m_stack_unpad_on_gpu {
            gpu_stack_limit
        } else {
            0
        };
        self.m_num_stacked_on_gpu = 0;
        self.m_num_full_saved = 0;
        if gpu_flags != 0 {
            fgpu_zero_timers();
        }
        self.m_group_size = group_size;
        self.m_group_size_initial = group_size;
        self.m_trim_frac = trim_frac;
        self.m_taper_frac = taper_frac;
        self.m_num_frames = 0;
        self.m_bin_sum = bin_sum;
        self.m_bin_align = bin_align;
        self.m_num_all_vs_all = num_all_vs_all;
        self.m_max_shift = max_shift;
        self.m_num_filters = num_filters;
        self.m_summing_mode = summing_mode;
        self.m_num_fits = 0;
        self.m_kfactor = k_factor;
        self.m_max_max_weight = max_max_weight;
        self.m_best_filt = 0;
        self.m_picked_best_filt = false;
        self.m_use_hybrid = use_hybrid;
        for filt in 0..num_filters as usize {
            self.m_num_as_best_filt[filt] = 0;
        }
        self.m_max_shift = max_shift;
        self.m_cum_align_at_end = cum_align_at_end;
        self.m_ali_filt_size = ali_filt_size;
        self.m_nx = nx;
        self.m_ny = ny;
        self.m_align_sum[..self.m_align_pix as usize].fill(0.);
        if summing_mode <= 0 {
            self.m_full_even_sum[..((sum_xpad + 2) * sum_ypad) as usize].fill(0.);
            self.m_full_odd_sum[..((sum_xpad + 2) * sum_ypad) as usize].fill(0.);
        }
        if !self.m_unweight_sum.is_empty() {
            self.m_unweight_sum[..((sum_xpad + 2) * sum_ypad) as usize].fill(0.);
        }
        self.m_wall_full_fft = 0.;
        self.m_wall_bin_pad = 0.;
        self.m_wall_bin_fft = 0.;
        self.m_wall_reduce = 0.;
        self.m_wall_shift = 0.;
        self.m_wall_conj_prod = 0.;
        self.m_wall_filter = 0.;
        self.m_wall_pre_proc = 0.;
        self.m_wall_noise = 0.;
        self.m_doing_dose_weighting = false;
        self.m_dose_wgt_filter = Vec::new();
        self.m_reweight_filt = Vec::new();
        self.m_dwfdelta = 0.;
        self.m_make_unwgt_sum = make_unwgt_sum;
        0
    }

    /// C `FrameAlign::setupDoseWeighting` (`framealign.cpp:563`).
    ///
    /// Store parameters and resize arrays for dose weighting, also save a
    /// reweighting filter at this time.  Pass `None` for `reweightFilt` if no
    /// reweighting is to be done; pass an array with all 1's to have a
    /// normalizing reweighting computed here.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_dose_weighting(
        &mut self,
        prior_dose: f32,
        frame_doses: &[f32],
        pixel_size: f32,
        crit_scale: f32,
        a_fac: f32,
        b_fac: f32,
        c_fac: f32,
        reweight_filt: Option<&[f32]>,
        filt_size: &mut i32,
    ) -> i32 {
        let mut all_ones = true;
        self.m_doing_dose_weighting = true;
        self.m_prior_dose_cum = prior_dose;
        self.m_frame_doses = vec![0.; self.m_num_expected_frames as usize];
        for ind in 0..self.m_num_expected_frames as usize {
            self.m_frame_doses[ind] = frame_doses[ind];
        }
        self.m_pixel_size = pixel_size;
        self.m_crit_dose_scale = crit_scale;
        self.m_crit_dose_afac = a_fac;
        self.m_crit_dose_bfac = b_fac;
        self.m_crit_dose_cfac = c_fac;
        *filt_size = 2 * if self.m_full_xpad > self.m_full_ypad {
            self.m_full_xpad
        } else {
            self.m_full_ypad
        };
        // B3DCLAMP(filtSize, 1024, 8193) is MAX(1024, MIN(8193, filtSize))
        *filt_size = 1024.max(8193.min(*filt_size));
        self.m_dose_wgt_filter = vec![0.; *filt_size as usize];
        if let Some(reweight_filt) = reweight_filt {
            self.m_reweight_filt = vec![0.; *filt_size as usize];
            for ind in 0..*filt_size as usize {
                self.m_reweight_filt[ind] = reweight_filt[ind];
                if reweight_filt[ind] != 1.0 {
                    all_ones = false;
                }
            }

            // If the reweight filter is all ones, compute a normalizing filter
            // here from the inverse of the sum of filters to be used
            if all_ones {
                for ind in 0..*filt_size as usize {
                    self.m_reweight_filt[ind] = 0.;
                }
                let mut prior_dose = prior_dose;
                for frame in 0..self.m_num_expected_frames as usize {
                    let n = self.m_dose_wgt_filter.len() as i32;
                    let mut delta = 0.;
                    dose_weight_filter(
                        prior_dose,
                        prior_dose + self.m_frame_doses[frame],
                        self.m_pixel_size,
                        self.m_crit_dose_afac,
                        self.m_crit_dose_bfac,
                        self.m_crit_dose_cfac,
                        self.m_crit_dose_scale,
                        &mut self.m_dose_wgt_filter,
                        n,
                        0.71f32,
                        &mut delta,
                    );
                    self.m_dwfdelta = delta;
                    prior_dose += self.m_frame_doses[frame];
                    for ind in 0..*filt_size as usize {
                        self.m_reweight_filt[ind] += self.m_dose_wgt_filter[ind];
                    }
                }
                for ind in 0..*filt_size as usize {
                    if self.m_reweight_filt[ind] > 0. {
                        self.m_reweight_filt[ind] =
                            self.m_num_expected_frames as f32 / self.m_reweight_filt[ind];
                    }
                }
            }
        }
        0
    }
}

impl FrameAlign {
    /// C `FrameAlign::preProcessFrame` (`framealign.cpp:833`).
    ///
    /// Preprocess an image with gain normalization, truncation and defect
    /// correction.  The four `NORM_TRUNC` / `NORM_ONLY` / `TRUNC_ONLY` /
    /// `JUST_COPY` macro expansions are written out per mode, as the macros
    /// expand them.  The `#pragma omp parallel for` over `iy` is run
    /// sequentially: every iteration writes a disjoint row of `fOut`.
    pub fn pre_process_frame(
        &mut self,
        frame: FrameData,
        dark_ref: Option<&[i16]>,
        def_bin: i32,
        f_out: FullRef,
    ) {
        let nx_gain = self.m_nx_gain;
        let trunc_limit = self.m_trunc_limit;
        let typ = self.m_frame_type;
        let gain_ref = self.m_gain_ref.clone();
        let mut gain_xoff = 0;
        let mut gain_yoff = 0;
        let max_threads;
        if gain_ref.is_some() {
            gain_xoff = (self.m_nx_gain - self.m_nx) / 2;
            gain_yoff = (self.m_ny_gain - self.m_ny) / 2;
        }

        // All processing converts input to a float array: gain/truncation
        // place it in float, otherwise it gets copied to float; then defect
        // operates float->float
        if trunc_limit > 0. && gain_ref.is_some() {
            max_threads = 6;
        } else if trunc_limit > 0. || gain_ref.is_some() {
            max_threads = 3;
        } else {
            max_threads = 1;
        }
        let _num_threads = num_omp_threads(max_threads);
        let nxt = self.m_nx;
        let nyt = self.m_ny;
        let frame_bytes = frame.bytes();

        let mut out = self.take_full(f_out);
        for iy in 0..nyt {
            let base = (iy * nxt) as usize;
            if let Some(gain) = gain_ref.as_ref() {
                let gbase = ((iy + gain_yoff) * nx_gain + gain_xoff) as usize;
                if dark_ref.is_some() && trunc_limit > 0. {
                    // Dark and gain with truncation
                    let dark = dark_ref.unwrap();
                    for ix in 0..nxt as usize {
                        let raw = match (typ, frame) {
                            (MRC_MODE_BYTE, FrameData::Byte(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as i32) as f32
                            }
                            (MRC_MODE_SHORT, FrameData::Short(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as i32) as f32
                            }
                            (MRC_MODE_USHORT, FrameData::UShort(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as u16 as i32) as f32
                            }
                            (MRC_MODE_FLOAT, FrameData::Float(f)) => {
                                f[base + ix] - dark[base + ix] as f32
                            }
                            _ => continue,
                        };
                        let mut val = raw * gain[gbase + ix];
                        if val > trunc_limit {
                            val = cor_def_surrounding_mean(
                                frame_bytes,
                                typ,
                                nxt,
                                nyt,
                                trunc_limit,
                                ix as i32,
                                iy,
                            );
                        }
                        out[base + ix] = val;
                    }
                } else if trunc_limit > 0. {
                    // Gain norm with truncation
                    for ix in 0..nxt as usize {
                        let raw = match (typ, frame) {
                            (MRC_MODE_BYTE, FrameData::Byte(f)) => f[base + ix] as f32,
                            (MRC_MODE_SHORT, FrameData::Short(f)) => f[base + ix] as f32,
                            (MRC_MODE_USHORT, FrameData::UShort(f)) => f[base + ix] as f32,
                            (MRC_MODE_FLOAT, FrameData::Float(f)) => f[base + ix],
                            _ => continue,
                        };
                        let mut val = raw * gain[gbase + ix];
                        if val > trunc_limit {
                            val = cor_def_surrounding_mean(
                                frame_bytes,
                                typ,
                                nxt,
                                nyt,
                                trunc_limit,
                                ix as i32,
                                iy,
                            );
                        }
                        out[base + ix] = val;
                    }
                } else if let Some(dark) = dark_ref {
                    // Dark and gain without truncation
                    for ix in 0..nxt as usize {
                        let raw = match (typ, frame) {
                            (MRC_MODE_BYTE, FrameData::Byte(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as i32) as f32
                            }
                            (MRC_MODE_SHORT, FrameData::Short(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as i32) as f32
                            }
                            (MRC_MODE_USHORT, FrameData::UShort(f)) => {
                                (f[base + ix] as i32 - dark[base + ix] as u16 as i32) as f32
                            }
                            (MRC_MODE_FLOAT, FrameData::Float(f)) => {
                                f[base + ix] - dark[base + ix] as f32
                            }
                            _ => continue,
                        };
                        out[base + ix] = raw * gain[gbase + ix];
                    }
                } else {
                    // Gain norm without truncation
                    for ix in 0..nxt as usize {
                        let raw = match (typ, frame) {
                            (MRC_MODE_BYTE, FrameData::Byte(f)) => f[base + ix] as f32,
                            (MRC_MODE_SHORT, FrameData::Short(f)) => f[base + ix] as f32,
                            (MRC_MODE_USHORT, FrameData::UShort(f)) => f[base + ix] as f32,
                            (MRC_MODE_FLOAT, FrameData::Float(f)) => f[base + ix],
                            _ => continue,
                        };
                        out[base + ix] = raw * gain[gbase + ix];
                    }
                }
            } else if trunc_limit > 0. {
                // Truncation only
                for ix in 0..nxt as usize {
                    let mut val = match (typ, frame) {
                        (MRC_MODE_BYTE, FrameData::Byte(f)) => f[base + ix] as f32,
                        (MRC_MODE_SHORT, FrameData::Short(f)) => f[base + ix] as f32,
                        (MRC_MODE_USHORT, FrameData::UShort(f)) => f[base + ix] as f32,
                        (MRC_MODE_FLOAT, FrameData::Float(f)) => f[base + ix],
                        _ => continue,
                    };
                    if val > trunc_limit {
                        val = cor_def_surrounding_mean(
                            frame_bytes,
                            typ,
                            nxt,
                            nyt,
                            trunc_limit,
                            ix as i32,
                            iy,
                        );
                    }
                    out[base + ix] = val;
                }
            } else {
                // Or copying to the float array for defect correction
                for ix in 0..nxt as usize {
                    out[base + ix] = match (typ, frame) {
                        (MRC_MODE_BYTE, FrameData::Byte(f)) => f[base + ix] as f32,
                        (MRC_MODE_SHORT, FrameData::Short(f)) => f[base + ix] as f32,
                        (MRC_MODE_USHORT, FrameData::UShort(f)) => f[base + ix] as f32,
                        (MRC_MODE_FLOAT, FrameData::Float(f)) => f[base + ix],
                        _ => continue,
                    };
                }
            }
        }

        // Defect correction: pass one past edge on right and bottom
        if self.m_cam_size_x > 0 {
            let left = (self.m_cam_size_x / self.m_defect_bin - self.m_nx) / 2;
            let top = (self.m_cam_size_y / self.m_defect_bin - self.m_ny) / 2;
            let right = left + self.m_nx;
            let bottom = top + self.m_ny;
            let defects = self.m_cam_defects.clone();
            if let Some(defects) = defects {
                cor_def_correct_defects(
                    &defects,
                    float_bytes_mut(&mut out),
                    MRC_MODE_FLOAT,
                    def_bin,
                    top,
                    left,
                    bottom,
                    right,
                );
            }
        }
        self.put_full(f_out, out);
    }

    /// C `FrameAlign::wrapImage` (`framealign.cpp:2938`).
    ///
    /// Wrap an image to go between a correlation with the origin in the corner
    /// and one with the origin in the middle.
    #[allow(clippy::too_many_arguments)]
    pub fn wrap_image(
        &self,
        buf_from: &[f32],
        nx_dim_from: i32,
        nx_from: i32,
        ny_from: i32,
        buf_to: &mut [f32],
        nx_dim_to: i32,
        nx_to: i32,
        ny_to: i32,
        x_offset: i32,
        y_offset: i32,
    ) {
        let mut ix_from0 = [0i32; 4];
        let mut ix_to0 = [0i32; 4];
        let mut iy_from0 = [0i32; 4];
        let mut iy_to0 = [0i32; 4];
        let mut ix_from1 = [0i32; 4];
        let mut ix_to1 = [0i32; 4];
        let mut iy_from1 = [0i32; 4];
        let mut iy_to1 = [0i32; 4];
        crate::imod::mrc::frameutil::util_coords_for_wrap(
            nx_from,
            ny_from,
            nx_to,
            ny_to,
            x_offset,
            y_offset,
            &mut ix_from0,
            &mut ix_to0,
            &mut iy_from0,
            &mut iy_to0,
            &mut ix_from1,
            &mut ix_to1,
            &mut iy_from1,
            &mut iy_to1,
        );

        for quad in 0..4 {
            let ynum = iy_from1[quad] + 1 - iy_from0[quad];
            let xnum = ix_from1[quad] + 1 - ix_from0[quad];
            if xnum > 0 && ynum > 0 {
                for iy in 0..ynum {
                    let to = ((iy + iy_to0[quad]) * nx_dim_to + ix_to0[quad]) as usize;
                    let from = ((iy + iy_from0[quad]) * nx_dim_from + ix_from0[quad]) as usize;
                    buf_to[to..to + xnum as usize]
                        .copy_from_slice(&buf_from[from..from + xnum as usize]);
                }
            }
        }
    }

    /// C `FrameAlign::smoothedTotalDistance` (`framealign.cpp:2961`).
    ///
    /// Smooth the trajectory of shifts and compute the total length of it.
    #[allow(clippy::too_many_arguments)]
    pub fn smoothed_total_distance(
        &self,
        x_shifts: &[f32],
        y_shifts: &[f32],
        num_shifts: i32,
        raw_total: &mut f32,
        mut x_smoothed: Option<&mut [f32]>,
        mut y_smoothed: Option<&mut [f32]>,
        variance: Option<&mut f64>,
    ) -> f32 {
        let num_fit = if 7 < num_shifts { 7 } else { num_shifts };
        let order = if num_shifts > 4 { 2 } else { 1 };
        let num_before = num_fit / 2;
        let mut delx: f32;
        let mut dely: f32;
        let mut intcpt = 0.;
        let mut slopes = [0.0f32; 2];
        let mut ro = 0.;
        let mut sa = 0.;
        let mut sb = 0.;
        let mut se = 0.;
        let mut xpred = 0.;
        let mut prederr = 0.;
        let mut ypred = 0.;
        let mut last_xpred = 0.;
        let mut last_ypred = 0.;
        let mut dist: f32;
        let mut var_sum = 0.0f32;
        let frame: [f32; 7] = [0., 1., 2., 3., 4., 5., 6.];
        let frame_sq: [f32; 7] = [0., 1., 4., 9., 16., 25., 36.];
        let mut fit_start: i32;
        let mut fit_end: i32;
        *raw_total = 0.;
        if num_shifts < 2 {
            return 0.;
        }
        if num_shifts == 2 {
            delx = x_shifts[1] - x_shifts[0];
            dely = y_shifts[1] - y_shifts[0];
            dist = (delx * delx + dely * dely).sqrt();
            *raw_total = dist;
            return dist;
        }

        dist = 0.;
        for ind in 0..num_shifts {
            if ind != 0 {
                delx = x_shifts[ind as usize] - x_shifts[(ind - 1) as usize];
                dely = y_shifts[ind as usize] - y_shifts[(ind - 1) as usize];
                *raw_total += (delx * delx + dely * dely).sqrt();
            }

            fit_start = if 0 > ind - num_before {
                0
            } else {
                ind - num_before
            };
            fit_end = if fit_start + num_fit - 1 < num_shifts - 1 {
                fit_start + num_fit - 1
            } else {
                num_shifts - 1
            };
            fit_start = fit_end + 1 - num_fit;

            // Get predicted value from fit
            let s = fit_start as usize;
            let n = num_fit;
            if order == 1 {
                ls_fit_pred(
                    &frame,
                    &x_shifts[s..],
                    n,
                    &mut slopes[0],
                    &mut intcpt,
                    &mut ro,
                    &mut sa,
                    &mut sb,
                    &mut se,
                    frame[(ind - fit_start) as usize],
                    &mut xpred,
                    &mut prederr,
                );
                ls_fit_pred(
                    &frame,
                    &y_shifts[s..],
                    n,
                    &mut slopes[0],
                    &mut intcpt,
                    &mut ro,
                    &mut sa,
                    &mut sb,
                    &mut se,
                    frame[(ind - fit_start) as usize],
                    &mut ypred,
                    &mut prederr,
                );
            } else {
                let (s0, s1) = slopes.split_at_mut(1);
                ls_fit2_pred(
                    &frame,
                    &frame_sq,
                    &x_shifts[s..],
                    n,
                    &mut s0[0],
                    &mut s1[0],
                    Some(&mut intcpt),
                    frame[(ind - fit_start) as usize],
                    frame_sq[(ind - fit_start) as usize],
                    &mut xpred,
                    &mut prederr,
                );
                ls_fit2_pred(
                    &frame,
                    &frame_sq,
                    &y_shifts[s..],
                    n,
                    &mut s0[0],
                    &mut s1[0],
                    Some(&mut intcpt),
                    frame[(ind - fit_start) as usize],
                    frame_sq[(ind - fit_start) as usize],
                    &mut ypred,
                    &mut prederr,
                );
            }
            if ind != 0 {
                delx = xpred - last_xpred;
                dely = ypred - last_ypred;
                dist += (delx * delx + dely * dely).sqrt();
            }
            last_xpred = xpred;
            last_ypred = ypred;
            if let Some(xs) = x_smoothed.as_deref_mut() {
                xs[ind as usize] = xpred;
            }
            if let Some(ys) = y_smoothed.as_deref_mut() {
                ys[ind as usize] = ypred;
            }
            // `framealign.cpp:3022-3023` computes delx/dely here and never
            // uses them; the variance accumulates the predictions themselves.
            delx = xpred - x_shifts[ind as usize];
            dely = ypred - y_shifts[ind as usize];
            let _ = (delx, dely);
            var_sum += xpred * xpred + ypred * ypred;
        }
        if let Some(v) = variance {
            *v = var_sum as f64 / (2. * num_shifts as f64);
        }
        dist
    }

    /// C `FrameAlign::frameShiftFromGroups` (`framealign.cpp:3033`).
    ///
    /// Get a shift for one frame from shifts that may be for groups.
    pub fn frame_shift_from_groups(
        &self,
        frame: i32,
        filt: i32,
        shift_x: &mut f32,
        shift_y: &mut f32,
    ) {
        // Spacing between frames is 1 and first frame of group is at
        // -(group size - 1) / 2 frames relative to group center
        let real_ind = (frame as f64 - (self.m_group_size as f64 - 1.) / 2.) as f32;
        let frac: f32;
        let mut ind: i32;
        let f = filt as usize;
        if self.m_group_size == 1 || (self.m_xshifts[f].len() as i32) < 2 {
            *shift_x = self.m_xshifts[f][frame as usize];
            *shift_y = self.m_yshifts[f][frame as usize];
        } else if real_ind < -1. {
            *shift_x = self.m_xshifts[f][0];
            *shift_y = self.m_yshifts[f][0];
        } else if real_ind > self.m_xshifts[f].len() as i32 as f32 {
            *shift_x = *self.m_xshifts[f].last().unwrap();
            *shift_y = *self.m_yshifts[f].last().unwrap();
        } else {
            // Allow a bit of extrapolation.  The (int)mXshifts is crucial here
            // for comparisons
            ind = real_ind.floor() as i32;
            let hi = self.m_xshifts[f].len() as i32 - 2;
            ind = 0.max(hi.min(ind));
            frac = real_ind - ind as f32;
            *shift_x = (1. - frac) * self.m_xshifts[f][ind as usize]
                + frac * self.m_xshifts[f][(ind + 1) as usize];
            *shift_y = (1. - frac) * self.m_yshifts[f][ind as usize]
                + frac * self.m_yshifts[f][(ind + 1) as usize];
        }
    }

    /// C `FrameAlign::getAllFrameShifts` (`framealign.cpp:3060`).
    pub fn get_all_frame_shifts(
        &self,
        frame_xshift: &mut Vec<f32>,
        frame_yshift: &mut Vec<f32>,
        use_filt: i32,
    ) {
        frame_xshift.resize(self.m_num_frames as usize, 0.);
        frame_yshift.resize(self.m_num_frames as usize, 0.);
        for ind in 0..self.m_num_frames {
            let mut x = 0.;
            let mut y = 0.;
            self.frame_shift_from_groups(ind, use_filt, &mut x, &mut y);
            frame_xshift[ind as usize] = x;
            frame_yshift[ind as usize] = y;
        }
    }

    /// C `FrameAlign::adjustAndPushShifts` (`framealign.cpp:2658`).
    ///
    /// Adjust shifts by the cumulative difference from the first set of shifts
    /// used, and store them.
    pub fn adjust_and_push_shifts(&mut self, top_ind: i32, filt: i32, use_filt: i32) {
        let mut x_diff = 0.;
        let mut y_diff = 0.;
        let mut x_sd = 0.;
        let mut y_sd = 0.;
        let mut sem = 0.;
        let f = filt as usize;
        let uf = use_filt as usize;

        // Get mean difference between last and this set of shifts
        for ind in 0..(self.m_num_all_vs_all - 1) as usize {
            self.m_last_xfit[f][ind + 1] =
                self.m_xfit_shifts[uf][ind] - self.m_last_xfit[f][ind + 1];
            self.m_last_yfit[f][ind + 1] =
                self.m_yfit_shifts[uf][ind] - self.m_last_yfit[f][ind + 1];
        }
        avg_sd(
            &self.m_last_xfit[f][1..],
            self.m_num_all_vs_all - 1,
            &mut x_diff,
            &mut x_sd,
            &mut sem,
        );
        avg_sd(
            &self.m_last_yfit[f][1..],
            self.m_num_all_vs_all - 1,
            &mut y_diff,
            &mut y_sd,
            &mut sem,
        );

        // Add to cumulative difference and push new shifts adjusted by this diff
        self.m_cumul_xdiff[f] += x_diff;
        self.m_cumul_ydiff[f] += y_diff;
        for ind in 1..=top_ind as usize {
            let x = self.m_xfit_shifts[uf][ind] - self.m_cumul_xdiff[f];
            let y = self.m_yfit_shifts[uf][ind] - self.m_cumul_ydiff[f];
            self.m_xshifts[f].push(x);
            self.m_yshifts[f].push(y);
        }
    }

    /// C `FrameAlign::splineSmooth` (`framealign.cpp:3072`).
    ///
    /// Do a spline smoothing of the shifts and return the smoothed coordinates
    /// and a distance.
    pub fn spline_smooth(
        &self,
        x_shifts: &[f32],
        y_shifts: &[f32],
        num_shifts: i32,
        smoothed_x: &mut [f32],
        smoothed_y: &mut [f32],
        spline_dist: &mut f32,
    ) -> i32 {
        let mut ierr = 0;
        let m_order = 2;
        let mode = 2;
        let x_var: f64;
        let y_var: f64;
        let variance = 0.5f64;
        let mut delx: f32;
        let mut dely: f32;
        let n = num_shifts as usize;
        // The C carves six windows out of one `allWork` block of
        // `6 * (numShifts * mOrder + 1) + numShifts + 5 * numShifts + 10`
        // doubles; the first five are `numShifts` long and `work` is the rest.
        let mut d_xshifts = vec![0.0f64; n];
        let mut d_yshifts = vec![0.0f64; n];
        let mut ordered_x = vec![0.0f64; n];
        let mut weights = vec![0.0f64; n];
        let mut coeff = vec![0.0f64; n];
        let mut work = vec![0.0f64; (6 * (num_shifts * m_order + 1) + num_shifts + 10) as usize];
        *spline_dist = 0.;
        for ind in 0..n {
            d_xshifts[ind] = x_shifts[ind] as f64;
            d_yshifts[ind] = y_shifts[ind] as f64;
            ordered_x[ind] = ind as f64;
            weights[ind] = 1.;
        }

        // Fit to X then get spline values
        gcvspl(
            &ordered_x, &d_xshifts, num_shifts, &weights, &weights, m_order, num_shifts, 1, mode,
            variance, &mut coeff, num_shifts, &mut work, &mut ierr,
        );
        if ierr != 0 {
            return ierr;
        }
        x_var = work[4];
        for ind in 0..n {
            let mut near = ind as i32;
            smoothed_x[ind] = splder(
                0, m_order, num_shifts, ind as f64, &ordered_x, &coeff, &mut near, &mut work,
            ) as f32;
        }

        // Fit to Y
        gcvspl(
            &ordered_x, &d_yshifts, num_shifts, &weights, &weights, m_order, num_shifts, 1, mode,
            variance, &mut coeff, num_shifts, &mut work, &mut ierr,
        );
        if ierr != 0 {
            return ierr;
        }
        y_var = work[4];
        for ind in 0..n {
            let mut near = ind as i32;
            smoothed_y[ind] = splder(
                0, m_order, num_shifts, ind as f64, &ordered_x, &coeff, &mut near, &mut work,
            ) as f32;
            if ind != 0 {
                delx = smoothed_x[ind] - smoothed_x[ind - 1];
                dely = smoothed_y[ind] - smoothed_y[ind - 1];
                *spline_dist += (delx * delx + dely * dely).sqrt();
            }
        }
        if self.m_debug != 0 {
            util_print(
                "GCV variance estimates = %f  %f\n",
                &[CArg::Dbl(x_var), CArg::Dbl(y_var)],
            );
        }
        0
    }

    /// C `FrameAlign::analyzeFRCcrossings` (`framealign.cpp:3136`).
    ///
    /// Find some crossings of the FRC and the level at half-nyquist.
    pub fn analyze_frc_crossings(
        &self,
        ring_corrs: &[f32],
        frc_delta_r: f32,
        half_cross: &mut f32,
        quart_cross: &mut f32,
        eighth_cross: &mut f32,
        half_nyq: &mut f32,
    ) {
        let cen_bin: i32;
        let num_bins: i32;
        *half_cross = 0.;
        *quart_cross = 0.;
        *eighth_cross = 0.;
        let top = (0.5 / frc_delta_r as f64).floor() as i32;
        for ind in 1..top {
            let i = ind as usize;
            if *half_cross == 0. && ring_corrs[i - 1] >= 0.5 && ring_corrs[i] <= 0.5 {
                *half_cross = (frc_delta_r as f64
                    * (ind as f64 - 0.5
                        + (ring_corrs[i - 1] as f64 - 0.5)
                            / (ring_corrs[i - 1] as f64 - ring_corrs[i] as f64)))
                    as f32;
            }
            if *quart_cross == 0. && ring_corrs[i - 1] >= 0.25 && ring_corrs[i] <= 0.25 {
                *quart_cross = (frc_delta_r as f64
                    * (ind as f64 - 0.5
                        + (ring_corrs[i - 1] as f64 - 0.25)
                            / (ring_corrs[i - 1] as f64 - ring_corrs[i] as f64)))
                    as f32;
            }
            if *eighth_cross == 0. && ring_corrs[i - 1] >= 0.125 && ring_corrs[i] <= 0.125 {
                *eighth_cross = (frc_delta_r as f64
                    * (ind as f64 - 0.5
                        + (ring_corrs[i - 1] as f64 - 0.125)
                            / (ring_corrs[i - 1] as f64 - ring_corrs[i] as f64)))
                    as f32;
            }
        }
        cen_bin = (0.25 / frc_delta_r as f64 - 0.5 + 0.5).floor() as i32;
        let nb = (0.075 / frc_delta_r as f64 + 0.5).floor() as i32;
        num_bins = if 1 > nb { 1 } else { nb };
        *half_nyq = 0.;
        for ind in 0..num_bins {
            *half_nyq += (ring_corrs[(ind + cen_bin - num_bins / 2) as usize] as f64
                / num_bins as f64) as f32;
        }
    }

    /// C `FrameAlign::filterAndAddToSum` (`framealign.cpp:3167`).
    ///
    /// Apply filter to image and add it to the sum; a simplified form of
    /// `XcorrFilterPart` that saves the time of zeroing regions where the
    /// filter is zero, plus saving the second pass through to add to sum.
    #[allow(clippy::too_many_arguments)]
    pub fn filter_and_add_to_sum(
        &self,
        fft: &[f32],
        array: &mut [f32],
        nx: i32,
        ny: i32,
        ctf: &[f32],
        delta: f32,
    ) {
        let mut x: f32;
        let delx: f32;
        let dely: f32;
        let mut y: f32;
        let mut s: f32;
        let max_freq: f32;
        let mut ysq: f64;
        let mut ix: i32;
        let mut index: i32;
        let mut ind: i32;
        let mut indp1: i32;
        let mut indf: i32;
        let nx_div2: i32;
        let nx_div2p1: i32;
        let ny_minus1: i32;
        let mut nx_max: i32;
        let mut num_threads: i32;
        let max_threads = 16;

        nx_div2 = nx / 2;
        nx_div2p1 = nx_div2 + 1;
        ny_minus1 = ny - 1;
        delx = (1.0 / nx as f64) as f32;
        dely = (1.0 / ny as f64) as f32;

        /* Find last non-zero filter value in range that matters */
        ix = (0.707 / delta) as i32;
        while ix > 1 {
            if ctf[ix as usize] != 0. {
                break;
            }
            ix -= 1;
        }

        /* Get a frequency limit to apply in Y and a limit to X indexes */
        max_freq = (ix + 1) as f32 * delta;
        nx_max = (max_freq / delx) as i32 + 1;
        nx_max = 1.max(nx_div2.min(nx_max));

        /* This formula gives 1.5+ at 128 and 11.5+ at 4096 */
        num_threads = (3.33 * ((nx as f64 * ny as f64).log10() - 3.75) + 0.5).floor() as i32;
        num_threads = 1.max(max_threads.min(num_threads));
        let _num_threads = num_omp_threads(num_threads);

        /*   apply filter function on fft, put result in array */
        for iy in 0..=ny_minus1 {
            y = iy as f32 * dely;
            index = iy * nx_div2p1;
            if y > 0.5 {
                y = 1.0 - y;
            }
            if y > max_freq {
                continue;
            }
            ysq = (y * y) as f64;
            x = 0.0;
            for ix in 0..=nx_max {
                ind = 2 * (index + ix);
                indp1 = ind + 1;
                s = ((x * x) as f64 + ysq).sqrt() as f32;
                indf = (s / delta + 0.5f32) as i32;
                array[ind as usize] += fft[ind as usize] * ctf[indf as usize];
                array[indp1 as usize] += fft[indp1 as usize] * ctf[indf as usize];
                x += delx;
            }
        }
    }
}

impl FrameAlign {
    /// C `FrameAlign::nextFrame` (`framealign.cpp:953`).
    ///
    /// Operate on the next frame.
    #[allow(clippy::too_many_arguments)]
    pub fn next_frame(
        &mut self,
        frame: FrameData,
        typ: i32,
        gain_ref: Option<Rc<Vec<f32>>>,
        nx_gain: i32,
        ny_gain: i32,
        dark_ref: Option<&[i16]>,
        trunc_limit: f32,
        defects: Option<Rc<CameraDefects>>,
        cam_size_x: i32,
        cam_size_y: i32,
        def_bin: i32,
        shift_x: f32,
        shift_y: f32,
    ) -> i32 {
        let saving =
            (self.m_num_all_vs_all > 0 || self.m_cum_align_at_end != 0 || self.m_defer_summing)
                && self.m_summing_mode >= 0;
        let saving_full = (saving || (self.m_num_all_vs_all == 0 && self.m_stack_unpad_on_gpu))
            && self.m_summing_mode <= 0;
        let do_bin_pad = self.m_taper_frac != 0. || self.m_trim_frac != 0.;
        let need_preprocess = gain_ref.is_some() || trunc_limit > 0. || cam_size_x > 0;
        let mut preproc_on_gpu =
            (self.m_gpu_flags & GPU_DO_PREPROCESS) != 0 && need_preprocess && dark_ref.is_none();
        let mut preproc_here = need_preprocess
            && ((!self.m_noise_pad_on_gpu && self.m_summing_mode <= 0)
                || (!self.m_bin_pad_on_gpu && self.m_summing_mode >= 0)
                || !preproc_on_gpu);
        let mut stack_on_gpu: bool;
        let mut bin_arr = BinRef::Work;
        let mut full_arr = FullRef::Work;
        let mut nx_bin = 0;
        let mut ny_bin = 0;
        let mut nx_taper = 0;
        let mut ny_taper = 0;
        let mut ind: i32;
        let mut err: i32;
        let mut x_offset: i32;
        let mut y_offset: i32;
        let mut use_ind: i32;
        let mut ix: i32;
        let mut filt: i32;
        let mut use_filt: i32;
        let mut stack_byte_size = 0;
        let mut x_shift = 0.;
        let mut y_shift = 0.;
        let mut near_xshift = 0.;
        let mut near_yshift = 0.;
        let mut need_extract = true;
        let mut did_noise_pad = false;
        let add_to_full =
            self.m_summing_mode < 0 || (self.m_summing_mode == 0 && !self.m_defer_summing);
        let filter_subarea =
            self.m_cum_align_at_end != 0 || (self.m_gpu_flags & GPU_FOR_ALIGNING) != 0;
        let mut use_type = typ;
        let num_frame_for_ava = self.m_num_all_vs_all + self.m_group_size - 1;
        let num_bin_pad_for_ava = if self.m_group_size > 1 {
            self.m_group_size
        } else {
            self.m_num_all_vs_all
        };
        let _sum_frame_ind = if self.m_defer_summing {
            self.m_num_frames
        } else if self.m_num_frames < num_frame_for_ava - 1 {
            self.m_num_frames
        } else {
            num_frame_for_ava - 1
        };
        let ali_frame_ind = if self.m_cum_align_at_end != 0 {
            self.m_num_frames
        } else if self.m_num_frames < num_bin_pad_for_ava - 1 {
            self.m_num_frames
        } else {
            num_bin_pad_for_ava - 1
        };
        let group_ind = if self.m_num_frames + 1 - self.m_group_size < self.m_num_all_vs_all - 1 {
            self.m_num_frames + 1 - self.m_group_size
        } else {
            self.m_num_all_vs_all - 1
        };
        let mut nx_dim_for_bp = self.m_nx;
        let mut nx_for_bp = self.m_nx;
        let mut ny_for_bp = self.m_ny;
        let mut use_frame = FullRef::Frame;
        let mut camera_size_x_for_gpu: i32;
        let mut camera_size_y_for_gpu: i32;

        // Save these as member variables to allow frame recovery from GPU
        self.m_noise_length = if self.m_nx > self.m_ny {
            self.m_nx
        } else {
            self.m_ny
        } / 50;
        self.m_noise_length = 20.max(120.min(self.m_noise_length));
        self.m_defect_bin = def_bin;
        self.m_frame_type = typ;
        self.m_nx_gain = nx_gain;
        self.m_ny_gain = ny_gain;
        self.m_gain_ref = gain_ref.clone();
        self.m_cam_defects = defects.clone();
        self.m_cam_size_x = cam_size_x;
        self.m_cam_size_y = cam_size_y;
        self.m_trunc_limit = trunc_limit;

        // PROCESS ALL-VS-ALL RESULT FIRST
        if self.m_num_all_vs_all != 0
            && self.m_num_frames > self.m_group_size - 1
            && self.m_summing_mode >= 0
        {
            self.find_all_vs_all_alignment(self.m_num_frames < num_frame_for_ava);
        }
        if self.m_num_all_vs_all != 0
            && self.m_num_frames >= num_frame_for_ava
            && self.m_summing_mode >= 0
        {
            filt = 0;
            while filt <= self.m_num_filters {
                let f = filt as usize;
                use_filt = if filt == self.m_num_filters {
                    self.m_best_filt
                } else {
                    filt
                };
                let uf = use_filt as usize;

                // First time, just set the first two shifts and add in 0
                if self.m_num_frames == num_frame_for_ava {
                    let (a, b, c, d) = (
                        self.m_xfit_shifts[uf][0],
                        self.m_yfit_shifts[uf][0],
                        self.m_xfit_shifts[uf][1],
                        self.m_yfit_shifts[uf][1],
                    );
                    self.m_xshifts[f].push(a);
                    self.m_yshifts[f].push(b);
                    self.m_xshifts[f].push(c);
                    self.m_yshifts[f].push(d);
                    if add_to_full
                        && filt == self.m_num_filters
                        && self.add_to_sums(FullRef::Null, 0, -9, 0, -1) != 0
                    {
                        self.cleanup();
                        return 2;
                    }
                } else {
                    // Later, get mean difference between last and this set of
                    // shifts and adjust
                    self.adjust_and_push_shifts(1, filt, use_filt);
                }
                if add_to_full
                    && filt == self.m_num_filters
                    && self.add_to_sums(
                        FullRef::Null,
                        0,
                        -9,
                        self.m_num_frames + 1 - num_frame_for_ava,
                        -1,
                    ) != 0
                {
                    self.cleanup();
                    return 2;
                }

                // Save the shifts
                for ind in 0..self.m_num_all_vs_all as usize {
                    self.m_last_xfit[f][ind] = self.m_xfit_shifts[uf][ind];
                    self.m_last_yfit[f][ind] = self.m_yfit_shifts[uf][ind];
                }

                // Shift the all-vs-all matrix down
                if filt < self.m_num_filters {
                    for reference in 1..self.m_num_all_vs_all - 1 {
                        for ind in reference + 1..self.m_num_all_vs_all {
                            let src = (reference * self.m_num_all_vs_all + ind) as usize;
                            let dst = ((reference - 1) * self.m_num_all_vs_all + ind - 1) as usize;
                            self.m_xall_shifts[f][dst] = self.m_xall_shifts[f][src];
                            self.m_yall_shifts[f][dst] = self.m_yall_shifts[f][src];
                        }
                    }
                }
                filt += 1;
            }
        }

        // Roll the saved align array if no cumulative alignment, roll groups
        // unconditionally
        if self.m_num_all_vs_all != 0
            && self.m_num_frames >= num_bin_pad_for_ava
            && self.m_summing_mode >= 0
        {
            if self.m_gpu_aligning {
                if self.m_cum_align_at_end == 0 {
                    fgpu_roll_align_stack();
                }
                if self.m_group_size > 1
                    && self.m_num_frames >= self.m_num_all_vs_all + self.m_group_size - 1
                {
                    fgpu_roll_group_stack();
                }
            } else {
                if self.m_cum_align_at_end == 0 {
                    util_roll_saved_frames(&mut self.m_saved_bin_pad, num_bin_pad_for_ava);
                }
                if self.m_group_size > 1
                    && self.m_num_frames >= self.m_num_all_vs_all + self.m_group_size - 1
                {
                    util_roll_saved_frames(&mut self.m_saved_groups, self.m_num_all_vs_all);
                }
            }
        }
        stack_on_gpu = self.m_stack_unpad_on_gpu
            && (self.m_gpu_stack_limit == 0 || self.m_num_stacked_on_gpu < self.m_gpu_stack_limit);

        // For first frame, set up pre-processing params on GPU; do this here so
        // it is easy to fall back to CPU entirely if failure
        if self.m_num_frames == 0 && self.m_gpu_flags > 0 {
            err = 0;
            let mut defect_map: Vec<u8> = Vec::new();
            if cam_size_x > 0 && preproc_on_gpu {
                camera_size_x_for_gpu = cam_size_x;
                camera_size_y_for_gpu = cam_size_y;
                let mut gpu_defects: CameraDefects = (*defects.clone().unwrap()).clone();
                defect_map = vec![0u8; (self.m_nx * self.m_ny) as usize];

                // Scale the defects down if they are scaled for K2 and defect
                // binning value is 2
                if err == 0 && gpu_defects.k2_type > 0 && gpu_defects.was_scaled > 0 && def_bin > 1
                {
                    cor_def_scale_defects_for_k2(&mut gpu_defects, true);
                    camera_size_x_for_gpu /= 2;
                    camera_size_y_for_gpu /= 2;
                }
                cor_def_fill_defect_array(
                    &gpu_defects,
                    camera_size_x_for_gpu,
                    camera_size_y_for_gpu,
                    &mut defect_map,
                    self.m_nx,
                    self.m_ny,
                    true,
                );
            }

            // It tests for both defectMap and camSizeX, so no need to make it 0
            // if not preproc
            if fgpu_set_pre_proc_params(
                if preproc_on_gpu {
                    gain_ref.as_deref().map(|v| v.as_slice())
                } else {
                    None
                },
                nx_gain,
                ny_gain,
                if preproc_on_gpu { trunc_limit } else { 0. },
                if defect_map.is_empty() {
                    None
                } else {
                    Some(&defect_map)
                },
                cam_size_x,
                cam_size_y,
            ) != 0
            {
                err = 1;
            }

            // Fallback to doing all preproc and prep steps on CPU if there was
            // an error
            if err != 0 {
                stack_on_gpu = false;
                preproc_on_gpu = false;
                preproc_here = need_preprocess;
                self.cancel_initial_steps_on_gpu();
            }
        }

        // If noise padding is done here, floats will be saved, but if it is done
        // on GPU and either no preprocessing happens or it happens there, then
        // save raw images
        self.m_stack_type = MRC_MODE_FLOAT;
        if self.m_noise_pad_on_gpu && (preproc_on_gpu || !need_preprocess) {
            self.m_stack_type = typ;
        }
        ix = 0;
        data_size_for_mode(self.m_stack_type, &mut stack_byte_size, &mut ix);

        // Substitute the save arrays for the working ones; create new and push
        // if needed
        if saving {
            if self.m_num_all_vs_all != 0 && !self.m_gpu_aligning {
                if ali_frame_ind < self.m_saved_bin_pad.len() as i32 {
                    bin_arr = BinRef::Saved(ali_frame_ind as usize);
                } else {
                    self.m_saved_bin_pad
                        .push(vec![0.; self.m_align_pix as usize]);
                    bin_arr = BinRef::Saved(self.m_saved_bin_pad.len() - 1);
                }
            }
            if saving_full && !stack_on_gpu {
                if self.m_num_full_saved < self.m_saved_full_size.len() as i32 {
                    full_arr = FullRef::Saved(self.m_num_full_saved as usize);
                    self.m_saved_full_frame_num[self.m_num_full_saved as usize] = self.m_num_frames;
                } else {
                    let floats = ((stack_byte_size as i64
                        * (self.m_full_xpad + 2) as i64
                        * self.m_full_ypad as i64) as usize
                        + 3)
                        / 4;
                    self.m_saved_full_size.push(vec![0.; floats]);
                    self.m_saved_full_frame_num.push(self.m_num_frames);
                    full_arr = FullRef::Saved(self.m_saved_full_size.len() - 1);
                }
                self.m_num_full_saved += 1;
            }
        }

        // Now set flags for all the data-flow cases as needed
        let mut need_taper_out = false;
        let mut set_full_to_use = false;
        let mut copy_to_stack_here = false;
        let mut set_full_to_work = false;
        let mut copy_raw_input = false;
        let mut send_raw_to_bin_pad = false;
        let mut preproc_into_work = false;
        if self.m_summing_mode > 0 {
            // Aligning only
            if do_bin_pad {
                if self.m_bin_pad_on_gpu {
                    send_raw_to_bin_pad = preproc_on_gpu;
                } else {
                    set_full_to_use = need_preprocess || typ == MRC_MODE_FLOAT;
                    need_taper_out = !set_full_to_use;
                }
            }
        } else if self.m_summing_mode < 0 {
            // Summing only
            set_full_to_use = self.m_noise_pad_on_gpu;
        } else {
            // Align and sum
            if !self.m_bin_pad_on_gpu && !self.m_noise_pad_on_gpu {
                copy_to_stack_here = !need_preprocess && typ == MRC_MODE_FLOAT;
            } else if !self.m_bin_pad_on_gpu && self.m_noise_pad_on_gpu {
                if !need_preprocess {
                    copy_to_stack_here = true;
                    set_full_to_work = true;
                    need_taper_out = true;
                } else if preproc_on_gpu {
                    copy_to_stack_here = true;
                    copy_raw_input = true;
                    set_full_to_work = true;
                }
            } else if self.m_bin_pad_on_gpu && !self.m_noise_pad_on_gpu {
                send_raw_to_bin_pad = preproc_on_gpu || !need_preprocess;
                preproc_into_work = preproc_here && !preproc_on_gpu;
            } else if self.m_bin_pad_on_gpu && self.m_noise_pad_on_gpu && !stack_on_gpu {
                if !need_preprocess {
                    copy_to_stack_here = true;
                } else if preproc_on_gpu {
                    copy_to_stack_here = true;
                }
            }
        }

        if do_bin_pad {
            nx_bin = (self.m_xend + 1 - self.m_xstart) / self.m_bin_align;
            ny_bin = (self.m_yend + 1 - self.m_ystart) / self.m_bin_align;
            nx_taper = (self.m_taper_frac * nx_bin as f32) as i32;
            ny_taper = (self.m_taper_frac * ny_bin as f32) as i32;
        }

        // PRE-PROCESS IF ANY: namely if there is gain reference, truncation, or
        // defect correction.  If using GPU, set parameters to do the pre-proc,
        // bin-pad and regular processing, and possible stacking there.
        if self.m_gpu_flags != 0 {
            fgpu_set_bin_pad_params(
                self.m_xstart,
                self.m_xend,
                self.m_ystart,
                self.m_yend,
                self.m_bin_align,
                nx_taper,
                ny_taper,
                if preproc_here && !send_raw_to_bin_pad {
                    MRC_MODE_FLOAT
                } else {
                    typ
                },
                self.m_anti_filt_type,
                self.m_noise_length,
            );
        }

        // Copy into stack now if it is to be raw
        if copy_to_stack_here && copy_raw_input {
            ix = 0;
            use_ind = 0;
            data_size_for_mode(typ, &mut use_ind, &mut ix);
            let n = (use_ind * self.m_nx * self.m_ny) as usize;
            let mut dst = self.take_full(full_arr);
            float_bytes_mut(&mut dst)[..n].copy_from_slice(&frame.bytes()[..n]);
            self.put_full(full_arr, dst);
            if set_full_to_work {
                full_arr = FullRef::Work;
            }
        }

        // Otherwise process here if needed
        if preproc_here {
            // Substitute source pointer and type
            use_frame = if preproc_into_work {
                FullRef::Work
            } else {
                full_arr
            };
            use_type = MRC_MODE_FLOAT;
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            self.pre_process_frame(frame, dark_ref, def_bin, use_frame);
            if self.m_report_times {
                self.m_wall_pre_proc += wall_time() - self.m_wall_start;
            }
        }
        // `binArr = sendRawToBinPad ? (float *)frame : useFrame;` under
        // mBinPadOnGpu: the buffer only ever reaches `fgpuProcessAlignImage`,
        // which is the no-GPU stub here and ignores it.

        // Now copy data into stack if flag is set
        if copy_to_stack_here && !copy_raw_input {
            ix = 0;
            use_ind = 0;
            data_size_for_mode(use_type, &mut use_ind, &mut ix);
            let n = (use_ind * self.m_nx * self.m_ny) as usize;
            if use_frame != full_arr {
                let src: Vec<f32> = match use_frame {
                    FullRef::Frame => Vec::new(),
                    other => self.take_full(other),
                };
                let mut dst = self.take_full(full_arr);
                if matches!(use_frame, FullRef::Frame) {
                    float_bytes_mut(&mut dst)[..n].copy_from_slice(&frame.bytes()[..n]);
                } else {
                    float_bytes_mut(&mut dst)[..n].copy_from_slice(&float_bytes(&src)[..n]);
                    self.put_full(use_frame, src);
                }
                self.put_full(full_arr, dst);
            }
            if set_full_to_work {
                full_arr = FullRef::Work;
            }
        }

        // PROCESS THE CURRENT IMAGE: Get the padded full image
        if self.m_bin_pad_on_gpu {
            err = fgpu_process_align_image(
                &[],
                if saving { ali_frame_ind } else { -1 },
                group_ind,
                if saving_full && stack_on_gpu { 1 } else { 0 },
            );
            if err != 0 {
                // If doing noise pad on GPU then need to possibly get the stack
                // back from GPU, and also preprocess and noise pad it to be like
                // a normal CPU stack here
                if self.m_noise_pad_on_gpu {
                    let mut recovered = FullRef::Null;
                    if self
                        .recover_gpu_full_stack(saving_full && stack_on_gpu, Some(&mut recovered))
                        != 0
                    {
                        self.cleanup();
                        return 3;
                    }
                    did_noise_pad = true;
                    full_arr = recovered;
                    bin_arr = BinRef::Work;
                    stack_on_gpu = false;
                }
                self.cancel_initial_steps_on_gpu();
                set_full_to_use = false;

                // But if it was a downstream error, now cancel aligning as well
                if err == 1 {
                    if self.recover_gpu_align_ffts(
                        saving,
                        ali_frame_ind,
                        if self.m_num_all_vs_all != 0 {
                            BinRef::Null
                        } else {
                            BinRef::AlignSum
                        },
                        if saving { BinRef::Null } else { BinRef::Work },
                        None,
                        false,
                    ) != 0
                    {
                        return 3;
                    }
                    bin_arr = BinRef::Saved(ali_frame_ind as usize);
                }
            } else if stack_on_gpu {
                self.m_num_stacked_on_gpu += 1;
            }
        }

        if (self.m_summing_mode <= 0 && !self.m_noise_pad_on_gpu) || !do_bin_pad {
            // This produced floats into the full array
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            if !did_noise_pad {
                let mut dst = self.take_full(full_arr);
                let mut temp = std::mem::take(&mut self.m_shift_temp);
                if use_frame == full_arr {
                    slice_noise_taper_pad(
                        PadIn::InPlace,
                        use_type,
                        self.m_nx,
                        self.m_ny,
                        &mut dst,
                        self.m_full_xpad + 2,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        self.m_noise_length,
                        4,
                        &mut temp,
                    );
                } else if matches!(use_frame, FullRef::Frame) {
                    slice_noise_taper_pad(
                        frame.pad_in(),
                        use_type,
                        self.m_nx,
                        self.m_ny,
                        &mut dst,
                        self.m_full_xpad + 2,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        self.m_noise_length,
                        4,
                        &mut temp,
                    );
                } else {
                    let src = self.take_full(use_frame);
                    slice_noise_taper_pad(
                        PadIn::Float(&src),
                        use_type,
                        self.m_nx,
                        self.m_ny,
                        &mut dst,
                        self.m_full_xpad + 2,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        self.m_noise_length,
                        4,
                        &mut temp,
                    );
                    self.put_full(use_frame, src);
                }
                self.m_shift_temp = temp;
                self.put_full(full_arr, dst);
            }
            if self.m_report_times {
                self.m_wall_noise += wall_time() - self.m_wall_start;
            }

            // Set larger size for bin/pad operation to come from this array
            nx_dim_for_bp = self.m_full_xpad + 2;
            nx_for_bp = self.m_full_xpad;
            ny_for_bp = self.m_full_ypad;
        } else if need_taper_out {
            // This simply converts to a float array with no taper/pad
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let mut dst = self.take_full(full_arr);
            if use_frame == full_arr {
                slice_taper_out_pad(
                    PadIn::InPlace,
                    use_type,
                    self.m_nx,
                    self.m_ny,
                    &mut dst,
                    self.m_nx,
                    self.m_nx,
                    self.m_ny,
                    0,
                    0.,
                );
            } else {
                slice_taper_out_pad(
                    frame.pad_in(),
                    use_type,
                    self.m_nx,
                    self.m_ny,
                    &mut dst,
                    self.m_nx,
                    self.m_nx,
                    self.m_ny,
                    0,
                    0.,
                );
            }
            self.put_full(full_arr, dst);
            if self.m_report_times {
                self.m_wall_noise += wall_time() - self.m_wall_start;
            }
        } else if set_full_to_use {
            // And if it is float, just assign it to replace the work array
            full_arr = use_frame;
        }

        // If doing trim or taper, bin the subarea and taper inside, take the FFT
        if do_bin_pad && self.m_summing_mode >= 0 && !self.m_bin_pad_on_gpu {
            x_offset = (nx_for_bp - self.m_nx) / 2;
            y_offset = (ny_for_bp - self.m_ny) / 2;

            // Use zoomdown routine for binning, it is a lot faster.  Have to
            // select the filter again because reading EER selects filter too
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let full_owned: Vec<f32> = match full_arr {
                FullRef::Frame => Vec::new(),
                other => self.take_full(other),
            };
            let full_slice: &[f32] = if matches!(full_arr, FullRef::Frame) {
                match frame {
                    FrameData::Float(f) => f,
                    _ => &[],
                }
            } else {
                &full_owned
            };
            let mut bin_owned = self.take_bin(bin_arr);
            if self.m_bin_align > 1 {
                let rows: Vec<&[f32]> = (0..ny_for_bp as usize)
                    .map(|i| {
                        &full_slice[i * nx_dim_for_bp as usize..(i + 1) * nx_dim_for_bp as usize]
                    })
                    .collect();
                let mut width = 0;
                if select_zoom_filter(
                    self.m_anti_filt_type,
                    1. / self.m_bin_align as f64,
                    &mut width,
                ) != 0
                    || zoom_with_filter(
                        ZoomLines::Float(&rows),
                        nx_for_bp,
                        ny_for_bp,
                        (self.m_xstart + x_offset) as f32,
                        (self.m_ystart + y_offset) as f32,
                        nx_bin,
                        ny_bin,
                        nx_bin,
                        0,
                        MRC_MODE_FLOAT,
                        &mut ZoomOut::Float(&mut bin_owned),
                        None,
                        None,
                    ) == 0
                {
                    need_extract = false;
                }
            }
            if need_extract {
                extract_with_binning(
                    float_bytes(full_slice),
                    MRC_MODE_FLOAT,
                    nx_dim_for_bp,
                    self.m_xstart + x_offset,
                    self.m_xend + x_offset,
                    self.m_ystart + y_offset,
                    self.m_yend + y_offset,
                    self.m_bin_align,
                    float_bytes_mut(&mut bin_owned),
                    0,
                    &mut nx_bin,
                    &mut ny_bin,
                );
            }
            if !matches!(full_arr, FullRef::Frame) {
                self.put_full(full_arr, full_owned);
            }

            slice_taper_in_pad(
                PadIn::InPlace,
                MRC_MODE_FLOAT,
                nx_bin,
                0,
                nx_bin - 1,
                0,
                ny_bin - 1,
                &mut bin_owned,
                self.m_align_xpad + 2,
                self.m_align_xpad,
                self.m_align_ypad,
                nx_taper,
                ny_taper,
            );
            self.put_bin(bin_arr, bin_owned);
            if self.m_report_times {
                self.m_wall_bin_pad += wall_time() - self.m_wall_start;
            }
        }

        if self.m_gpu_aligning && do_bin_pad && !self.m_bin_pad_on_gpu {
            // Take FFT and save in stack on GPU if needed; if there is an error
            // here, try to recover the stack or the sum and just turn off GPU
            // aligning
            let bin_owned = self.take_bin(bin_arr);
            let rc = fgpu_process_align_image(
                &bin_owned,
                if saving { ali_frame_ind } else { -1 },
                group_ind,
                if saving_full && stack_on_gpu { 1 } else { 0 },
            );
            self.put_bin(bin_arr, bin_owned);
            if rc != 0 {
                let mut recovered = bin_arr;
                if self.recover_gpu_align_ffts(
                    saving,
                    ali_frame_ind,
                    if self.m_num_all_vs_all != 0 {
                        BinRef::Null
                    } else {
                        BinRef::AlignSum
                    },
                    if saving { BinRef::Null } else { BinRef::Work },
                    Some(&mut recovered),
                    saving_full && stack_on_gpu,
                ) != 0
                {
                    return 3;
                }
                bin_arr = recovered;
            } else if stack_on_gpu {
                self.m_num_stacked_on_gpu += 1;
            }
        }

        if !self.m_gpu_aligning {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let mut bin_owned = self.take_bin(bin_arr);
            todfft_c(&mut bin_owned, self.m_align_xpad, self.m_align_ypad, 0);
            self.put_bin(bin_arr, bin_owned);
            if self.m_report_times {
                self.m_wall_bin_fft += wall_time() - self.m_wall_start;
            }
        }

        // Take the full FFT
        if !do_bin_pad || (self.m_summing_mode <= 0 && !self.m_gpu_summing) {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let mut full_owned = self.take_full(full_arr);
            todfft_c(&mut full_owned, self.m_full_xpad, self.m_full_ypad, 0);
            self.put_full(full_arr, full_owned);
            if self.m_report_times {
                self.m_wall_full_fft += wall_time() - self.m_wall_start;
            }
        }

        // If just summing, add into sum and return
        if self.m_summing_mode < 0 {
            self.m_xshifts[0].push(shift_x);
            self.m_yshifts[0].push(shift_y);
            self.m_xshifts[1].push(shift_x);
            self.m_yshifts[1].push(shift_y);
            if self.add_to_sums(full_arr, -1, -9, self.m_num_frames, -1) != 0 {
                self.cleanup();
                return 2;
            }
            self.m_num_frames += 1;
            return 0;
        }

        // Now if not doing taper/pad, reduce the FFT into the align array
        if !do_bin_pad {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let full_owned = self.take_full(full_arr);
            let mut bin_owned = self.take_bin(bin_arr);
            let mut temp = std::mem::take(&mut self.m_shift_temp);
            fourier_reduce_image(
                &full_owned,
                self.m_full_xpad,
                self.m_full_ypad,
                &mut bin_owned,
                self.m_align_xpad,
                self.m_align_ypad,
                0.,
                0.,
                Some(&mut temp),
            );
            self.m_shift_temp = temp;
            self.put_bin(bin_arr, bin_owned);
            self.put_full(full_arr, full_owned);
            if self.m_report_times {
                self.m_wall_reduce += wall_time() - self.m_wall_start;
            }
        }

        // Apply full filter to the align array unless on GPU
        if !self.m_gpu_aligning {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let mut bin_owned = self.take_bin(bin_arr);
            for ind in 0..self.m_align_pix as usize {
                bin_owned[ind] *= self.m_full_filt_mask[ind];
            }
            self.put_bin(bin_arr, bin_owned);
            if self.m_report_times {
                self.m_wall_filter += wall_time() - self.m_wall_start;
            }
        }

        // Make a new group if ready
        use_ind = ali_frame_ind;
        if self.m_group_size > 1 && self.m_num_frames >= self.m_group_size - 1 {
            if !self.m_gpu_aligning {
                let group_ref: BinRef;
                if group_ind < self.m_saved_groups.len() as i32 {
                    group_ref = BinRef::Group(group_ind as usize);
                } else {
                    self.m_saved_groups
                        .push(vec![0.; self.m_align_pix as usize]);
                    group_ref = BinRef::Group(self.m_saved_groups.len() - 1);
                }
                let mut group_owned = self.take_bin(group_ref);
                group_owned[..self.m_align_pix as usize].fill(0.);
                for ind in (ali_frame_ind + 1 - self.m_group_size)..=ali_frame_ind {
                    let temp_bin = &self.m_saved_bin_pad[ind as usize];
                    for ix in 0..self.m_align_pix as usize {
                        group_owned[ix] += temp_bin[ix];
                    }
                }
                self.put_bin(group_ref, group_owned);
            }
            use_ind = group_ind;
        }

        if self.m_num_all_vs_all == 0 && !self.m_xshifts[0].is_empty() {
            near_xshift = *self.m_xshifts[0].last().unwrap();
            near_yshift = *self.m_yshifts[0].last().unwrap();
        }
        if self.m_num_all_vs_all != 0 && self.m_num_frames + self.m_group_size - 1 > 0 {
            // Align this frame with each previous frame, or nonoverlapping group
            ind = if self.m_num_frames + 1 - self.m_group_size < self.m_num_all_vs_all - 1 {
                self.m_num_frames + 1 - self.m_group_size
            } else {
                self.m_num_all_vs_all - 1
            };
            for reference in 0..=(ind - self.m_group_size) {
                for filt in 0..self.m_num_filters {
                    near_xshift = self.m_xnear_shifts[(ind - 1) as usize]
                        - self.m_xnear_shifts[reference as usize];
                    near_yshift = self.m_ynear_shifts[(ind - 1) as usize]
                        - self.m_ynear_shifts[reference as usize];
                    if self.align_two_frames(
                        use_ind + reference - ind,
                        use_ind,
                        near_xshift,
                        near_yshift,
                        filt,
                        &mut x_shift,
                        &mut y_shift,
                        filter_subarea || self.m_num_filters > 1,
                        self.m_dump_corrs,
                    ) != 0
                    {
                        self.cleanup();
                        return 2;
                    }
                    let slot = (reference * self.m_num_all_vs_all + ind) as usize;
                    self.m_xall_shifts[filt as usize][slot] = x_shift;
                    self.m_yall_shifts[filt as usize][slot] = y_shift;
                    if self.m_debug > 1 {
                        util_print(
                            "%d to %d  %.2f  %.2f   near %.2f  %.2f\n",
                            &[
                                CArg::Int(use_ind as i64),
                                CArg::Int((use_ind + reference - ind) as i64),
                                CArg::Dbl(x_shift as f64),
                                CArg::Dbl(y_shift as f64),
                                CArg::Dbl(near_xshift as f64),
                                CArg::Dbl(near_yshift as f64),
                            ],
                        );
                    }
                }
            }
        } else if self.m_num_all_vs_all == 0 {
            // Or align with the sum and add to the sums
            x_shift = 0.;
            y_shift = 0.;
            if self.m_cum_align_at_end == 0 {
                use_ind = -1;
            }
            if self.m_num_frames != 0
                && self.align_two_frames(
                    -1,
                    use_ind,
                    near_xshift,
                    near_yshift,
                    0,
                    &mut x_shift,
                    &mut y_shift,
                    filter_subarea,
                    self.m_dump_corrs,
                ) != 0
            {
                self.cleanup();
                return 2;
            }
            self.m_xshifts[0].push(x_shift);
            self.m_yshifts[0].push(y_shift);
            self.m_xshifts[1].push(x_shift);
            self.m_yshifts[1].push(y_shift);
            if self.m_summing_mode <= 0 && !self.m_defer_summing && stack_on_gpu {
                ix = 0;
            } else {
                ix = if self.m_summing_mode <= 0 && !self.m_defer_summing {
                    -1
                } else {
                    -9
                };
            }
            let pass = if self.m_summing_mode <= 0 && !self.m_defer_summing && !stack_on_gpu {
                full_arr
            } else {
                FullRef::Null
            };
            if self.add_to_sums(pass, ix, use_ind, self.m_num_frames, -1) != 0 {
                self.cleanup();
                return 2;
            }
        }

        // Increase the frame count after processing a frame
        self.m_num_frames += 1;
        0
    }
}

impl FrameAlign {
    /// C `FrameAlign::doRegression` (`framealign.cpp:1841`).
    ///
    /// Fill matrix for regression, optionally dropping the ones in the set, do
    /// robust or regular regression, using values for `filt` but putting the
    /// solution in `fitInd`.
    pub fn do_regression(
        &mut self,
        mut do_robust: bool,
        filt: i32,
        fit_ind: i32,
        drop_set: &BTreeSet<i32>,
    ) {
        let mut row: i32;
        let mut col: i32;
        let num_data: i32;
        let num_col: i32;
        let num_in_col: i32;
        let mut all_ind: i32;
        let mut num_iter = 0;
        let max_zero_wgt: i32;
        let mut sol_mat = [0.0f32; 2 * MAX_ALL_VS_ALL];
        let mut x_mean = [0.0f32; MAX_ALL_VS_ALL];
        let mut x_sd = [0.0f32; MAX_ALL_VS_ALL];
        let max_change = 0.02f32;
        let max_oscill = 0.05f32;
        let max_iter = 50;
        let num_frames = if self.m_num_frames < self.m_num_all_vs_all + self.m_group_size - 1 {
            self.m_num_frames
        } else {
            self.m_num_all_vs_all + self.m_group_size - 1
        };
        let num_groups = num_frames + 1 - self.m_group_size;
        let mut final_x = 0.0f32;
        let mut final_y = 0.0f32;

        // Load the data matrix with the correlations
        row = 0;
        num_col = num_groups + 3;
        num_in_col = num_groups - 1;
        all_ind = 0;
        for ind in 1..num_groups {
            for reference in 0..=(ind - self.m_group_size) {
                let this_ind = all_ind;
                all_ind += 1;
                if drop_set.contains(&this_ind) {
                    continue;
                }
                let base = (num_col * row) as usize;
                self.m_fit_mat[base + num_in_col as usize] = self.m_xall_shifts[filt as usize]
                    [(reference * self.m_num_all_vs_all + ind) as usize];
                self.m_fit_mat[base + num_in_col as usize + 1] = self.m_yall_shifts[filt as usize]
                    [(reference * self.m_num_all_vs_all + ind) as usize];
                col = 0;
                while col < num_in_col {
                    self.m_fit_mat[base + col as usize] =
                        if ind == num_groups - 1 { -1. } else { 0. };
                    col += 1;
                }
                self.m_fit_mat[base + reference as usize] += -1.;
                if ind < num_groups - 1 {
                    self.m_fit_mat[base + ind as usize] += 1.;
                }
                row += 1;
            }
        }
        num_data = row;

        // Do robust fitting if enough data, fall back to regular fit on error
        max_zero_wgt = (0.1 * num_data as f64).min((num_frames - 3) as f64) as i32;
        if do_robust {
            let mut work = std::mem::take(&mut self.m_fit_work);
            row = robust_regress(
                &mut self.m_fit_mat,
                num_col,
                1,
                num_in_col,
                num_data,
                2,
                &mut sol_mat,
                num_in_col,
                None,
                &mut x_mean,
                &mut x_sd,
                &mut work,
                self.m_kfactor,
                &mut num_iter,
                max_iter,
                max_zero_wgt,
                max_change,
                max_oscill,
            );
            self.m_fit_work = work;
            if row != 0 {
                if self.m_debug != 0 {
                    util_print(
                        "robustRegress%s failed with error %d\n",
                        &[
                            CArg::Str(if filt != fit_ind { " for CV" } else { "" }),
                            CArg::Int(row as i64),
                        ],
                    );
                }
                do_robust = false;
            }
        }
        if !do_robust {
            let mut work = std::mem::take(&mut self.m_fit_work);
            mult_regress(
                &self.m_fit_mat,
                num_col,
                1,
                num_in_col,
                num_data,
                2,
                0,
                &mut sol_mat,
                num_in_col,
                None,
                &mut x_mean,
                &mut x_sd,
                &mut work,
            );
            self.m_fit_work = work;
        }

        // Copy to the appropriate fitShifts
        for ind in 0..num_in_col as usize {
            final_x -= sol_mat[ind];
            final_y -= sol_mat[ind + num_in_col as usize];
            self.m_xfit_shifts[fit_ind as usize][ind] = sol_mat[ind];
            self.m_yfit_shifts[fit_ind as usize][ind] = sol_mat[ind + num_in_col as usize];
        }
        self.m_xfit_shifts[fit_ind as usize][num_in_col as usize] = final_x;
        self.m_yfit_shifts[fit_ind as usize][num_in_col as usize] = final_y;
    }

    /// C `FrameAlign::findAllVsAllAlignment` (`framealign.cpp:1551`).
    ///
    /// Solve for the alignment of the current group of frames.
    pub fn find_all_vs_all_alignment(&mut self, just_for_limits: bool) {
        let mut row: i32;
        let mut num_data: i32;
        let num_col: i32;
        let num_in_col: i32;
        let mut all_ind: usize;
        let mut max_as_best: i32;
        let mut ind_of_max: i32 = -1;
        let mut max_drop: i32;
        let mut num_sets: i32;
        let mut num_pred: i32;
        let mut ind_max_dist: i32 = 0;
        let mut pickable: bool;
        let mut failed = [false; MAX_FILTERS];
        let num_frames = if self.m_num_frames < self.m_num_all_vs_all + self.m_group_size - 1 {
            self.m_num_frames
        } else {
            self.m_num_all_vs_all + self.m_group_size - 1
        };
        let num_groups = num_frames + 1 - self.m_group_size;
        let mut res_mean = [0.0f32; MAX_FILTERS];
        let mut res_sd = [0.0f32; MAX_FILTERS];
        let mut pred_mean = [0.0f32; MAX_FILTERS];
        let mut max_wgt_res = [0.0f32; MAX_FILTERS];
        let mut max_raw = [0.0f32; MAX_FILTERS];
        let mut num_failed = [0i32; MAX_FILTERS];

        let mut do_robust = num_frames >= 5 && self.m_kfactor > 0.;
        let mut frac_drop: f32;
        let mut pred_sum: f32;
        let mut errx: f32;
        let mut erry: f32;
        let mut resid: f32;
        let mut wgt_resid: f32;
        let mut res_sum: f32;
        let mut res_sum_sq: f32;
        let mut dist_filt: f32;
        let mut dist0: f32 = 0.;
        let mut weight: f32;
        let mut min_wgt = 2.0f32;
        let mut min_error = 1.0e30f32;
        let mut max_fit_dist = 0.0f32;
        let mut fit_dist = [0.0f32; MAX_FILTERS];
        let mut err_measure = [0.0f32; MAX_FILTERS];
        let mut smooth_dist = [0.0f32; MAX_FILTERS];
        let mut max_smooth_dist = 0.0f32;
        let dist_crit = 0.75f32;
        let not_zero_crit = 4.0f32;
        let abs_zero_crit = (0.15 * self.m_bin_align as f64) as f32;
        let rel_zero_crit = (0.5 * self.m_bin_align as f64) as f32;
        let closer_ratio = 0.2f32;
        let mut full_weights: Vec<f32> = Vec::new();
        let mut drop_set: BTreeSet<i32> = BTreeSet::new();
        let mut ind_res_min = 0usize;
        let mut ind_pred_min = 0usize;
        num_col = num_groups + 3;
        num_in_col = num_groups - 1;

        // Evaluate failures of higher filters relative to lower ones
        if self.m_num_filters > 1 {
            for ind in 1..num_groups {
                for reference in 0..=(ind - self.m_group_size) {
                    all_ind = (reference * self.m_num_all_vs_all + ind) as usize;
                    dist0 = (self.m_xall_shifts[0][all_ind] * self.m_xall_shifts[0][all_ind]
                        + self.m_yall_shifts[0][all_ind] * self.m_yall_shifts[0][all_ind])
                        .sqrt();
                    if dist0 > not_zero_crit {
                        for filt in 1..self.m_num_filters as usize {
                            dist_filt = (self.m_xall_shifts[filt][all_ind]
                                * self.m_xall_shifts[filt][all_ind]
                                + self.m_yall_shifts[filt][all_ind]
                                    * self.m_yall_shifts[filt][all_ind])
                                .sqrt();
                            if dist_filt < abs_zero_crit
                                || (dist_filt < rel_zero_crit && dist_filt < closer_ratio * dist0)
                            {
                                num_failed[filt] += 1;
                            }
                        }
                    }
                }
            }
            if self.m_debug > 1 {
                util_print(
                    "numFailed %d %d %d %d %d\n",
                    &[
                        CArg::Int(num_failed[1] as i64),
                        CArg::Int(num_failed[2] as i64),
                        CArg::Int(num_failed[3] as i64),
                        CArg::Int(num_failed[4] as i64),
                        CArg::Int(num_failed[5] as i64),
                    ],
                );
            }
        }
        max_as_best = 0;
        if !self.m_picked_best_filt {
            self.m_best_filt = 0;
        }
        num_data = (num_groups + 1 - self.m_group_size) * (num_groups - self.m_group_size) / 2;
        do_robust = num_data >= 2 * num_groups && self.m_kfactor > 0.;
        for filt in 0..self.m_num_filters as usize {
            max_raw[filt] = 0.;
            max_wgt_res[filt] = 0.;
            res_sum = 0.;
            res_sum_sq = 0.;
            if self.m_group_size > 1 && (num_groups < self.m_group_size || num_data < num_groups) {
                for ind in 0..num_groups as usize {
                    self.m_xnear_shifts[ind] = 0.;
                    self.m_ynear_shifts[ind] = 0.;
                }
                for ind in 1..num_groups {
                    for reference in 0..=(ind - self.m_group_size) {
                        if self.m_xnear_shifts[ind as usize] == 0. {
                            self.m_xnear_shifts[ind as usize] = self.m_xnear_shifts
                                [reference as usize]
                                + self.m_xall_shifts[0]
                                    [(reference * self.m_num_all_vs_all + ind) as usize];
                            self.m_ynear_shifts[ind as usize] = self.m_ynear_shifts
                                [reference as usize]
                                + self.m_yall_shifts[0]
                                    [(reference * self.m_num_all_vs_all + ind) as usize];
                        }
                    }
                }
                continue;
            } else if num_groups == 1 {
                // Deal with having only 1 or 2 frames
                self.m_xfit_shifts[filt][0] = 0.;
                self.m_yfit_shifts[filt][0] = 0.;
                self.m_xnear_shifts[0] = 0.;
                self.m_ynear_shifts[0] = 0.;
                continue;
            } else if num_groups == 2 {
                self.m_xfit_shifts[filt][0] = -self.m_xall_shifts[filt][1] / 2.;
                self.m_yfit_shifts[filt][0] = -self.m_yall_shifts[filt][1] / 2.;
                self.m_xfit_shifts[filt][1] = self.m_xall_shifts[filt][1] / 2.;
                self.m_yfit_shifts[filt][1] = self.m_yall_shifts[filt][1] / 2.;
                if filt == 0 {
                    self.m_xnear_shifts[0] = -self.m_xall_shifts[0][1] / 2.;
                    self.m_ynear_shifts[0] = -self.m_yall_shifts[0][1] / 2.;
                    self.m_xnear_shifts[1] = self.m_xall_shifts[0][1] / 2.;
                    self.m_ynear_shifts[1] = self.m_yall_shifts[0][1] / 2.;
                }
                continue;
            }

            // Otherwise, do the fitting
            drop_set.clear();
            self.do_regression(do_robust, filt as i32, filt as i32, &drop_set);

            // For first filter, copy to the shifts used for predictions
            if filt == 0 {
                for ind in 0..num_groups as usize {
                    self.m_xnear_shifts[ind] = self.m_xfit_shifts[0][ind];
                    self.m_ynear_shifts[ind] = self.m_yfit_shifts[0][ind];
                }
            }
            if just_for_limits {
                continue;
            }

            // Compute residuals
            row = 0;
            fit_dist[filt] = 0.;
            full_weights.clear();
            for ind in 1..num_groups {
                for reference in 0..=(ind - self.m_group_size) {
                    all_ind = (reference * self.m_num_all_vs_all + ind) as usize;
                    errx = (self.m_xfit_shifts[filt][ind as usize]
                        - self.m_xfit_shifts[filt][reference as usize])
                        - self.m_xall_shifts[filt][all_ind];
                    erry = (self.m_yfit_shifts[filt][ind as usize]
                        - self.m_yfit_shifts[filt][reference as usize])
                        - self.m_yall_shifts[filt][all_ind];
                    resid = (errx * errx + erry * erry).sqrt();
                    weight = 1.;
                    if do_robust {
                        weight = self.m_fit_mat[(num_col * row + num_in_col + 2) as usize];
                    }
                    wgt_resid = resid * weight;
                    min_wgt = if min_wgt < weight { min_wgt } else { weight };
                    full_weights.push(weight);
                    res_sum += wgt_resid;
                    res_sum_sq += wgt_resid * wgt_resid;
                    max_raw[filt] = if max_raw[filt] > resid {
                        max_raw[filt]
                    } else {
                        resid
                    };
                    max_wgt_res[filt] = if max_wgt_res[filt] > wgt_resid {
                        max_wgt_res[filt]
                    } else {
                        wgt_resid
                    };
                    row += 1;
                    if ind == reference + 1 {
                        fit_dist[filt] += (((self.m_xfit_shifts[filt][ind as usize]
                            - self.m_xfit_shifts[filt][reference as usize])
                            as f64)
                            .powf(2.)
                            + ((self.m_yfit_shifts[filt][ind as usize]
                                - self.m_yfit_shifts[filt][reference as usize])
                                as f64)
                                .powf(2.))
                        .sqrt() as f32;
                    }
                }
            }

            sums_to_avg_sd(
                res_sum,
                res_sum_sq,
                num_data,
                &mut res_mean[filt],
                &mut res_sd[filt],
            );
            pred_mean[filt] = res_mean[filt];

            // Cross-validation for >= 4 groups
            if num_groups >= 4 {
                frac_drop = ((0.01 * num_data as f64) / num_groups as f64) as f32;
                frac_drop = 0.05f32.max(0.1f32.min(frac_drop));
                max_drop = 1.max((frac_drop * num_data as f32) as i32);
                num_sets = (num_data + max_drop - 1) / max_drop;
                num_pred = 0;
                pred_sum = 0.;

                // Loop on the runs, set up the drop set for a run, and do
                // regression
                for drop_run in 0..num_sets {
                    drop_set.clear();
                    let mut ind = drop_run;
                    while ind < num_data {
                        drop_set.insert(ind);
                        ind += num_sets;
                    }
                    do_robust =
                        num_data - drop_set.len() as i32 >= 2 * num_groups && self.m_kfactor > 0.;
                    self.do_regression(do_robust, filt as i32, MAX_FILTERS as i32, &drop_set);

                    // Compute the leave-out error
                    row = 0;
                    for ind in 1..num_groups {
                        for reference in 0..=(ind - self.m_group_size) {
                            if drop_set.contains(&row) {
                                all_ind = (reference * self.m_num_all_vs_all + ind) as usize;
                                errx = (self.m_xfit_shifts[MAX_FILTERS][ind as usize]
                                    - self.m_xfit_shifts[MAX_FILTERS][reference as usize])
                                    - self.m_xall_shifts[filt][all_ind];
                                erry = (self.m_yfit_shifts[MAX_FILTERS][ind as usize]
                                    - self.m_yfit_shifts[MAX_FILTERS][reference as usize])
                                    - self.m_yall_shifts[filt][all_ind];
                                resid = (errx * errx + erry * erry).sqrt();
                                pred_sum += (if do_robust {
                                    full_weights[row as usize]
                                } else {
                                    1.
                                }) * resid;
                                num_pred += 1;
                            }
                            row += 1;
                        }
                    }
                }
                pred_mean[filt] = pred_sum / num_pred as f32;
            }

            // Maintain stats for this filter
            if res_mean[filt] < res_mean[ind_res_min] {
                ind_res_min = filt;
            }
            if pred_mean[filt] < pred_mean[ind_pred_min] {
                ind_pred_min = filt;
            }

            self.m_pred_mean_sum[filt] += pred_mean[filt];
            self.m_res_mean_sum[filt] += res_mean[filt];
            self.m_res_sdsum[filt] += res_sd[filt];
            self.m_res_max_sum[filt] += max_wgt_res[filt];
            self.m_raw_max_sum[filt] += max_raw[filt];
            self.m_max_res_max[filt] = if self.m_max_res_max[filt] > max_wgt_res[filt] {
                self.m_max_res_max[filt]
            } else {
                max_wgt_res[filt]
            };
            self.m_max_raw_max[filt] = if self.m_max_raw_max[filt] > max_raw[filt] {
                self.m_max_raw_max[filt]
            } else {
                max_raw[filt]
            };
            err_measure[filt] = (1. - self.m_max_max_weight) * pred_mean[filt]
                + self.m_max_max_weight * max_wgt_res[filt];
            failed[filt] = num_failed[filt] >= 1.max(num_frames - 2);
            if failed[filt] {
                self.m_num_as_best_filt[filt] -= 1;
            }
            smooth_dist[filt] = 0.;
            if self.m_xshifts[filt].len() >= 3 {
                let xs = std::mem::take(&mut self.m_xshifts[filt]);
                let ys = std::mem::take(&mut self.m_yshifts[filt]);
                smooth_dist[filt] = self.smoothed_total_distance(
                    &xs,
                    &ys,
                    xs.len() as i32,
                    &mut dist0,
                    None,
                    None,
                    None,
                );
                self.m_xshifts[filt] = xs;
                self.m_yshifts[filt] = ys;
            }
            if max_fit_dist < fit_dist[filt] {
                ind_max_dist = filt as i32;
                max_fit_dist = fit_dist[filt];
            }
            max_smooth_dist = if max_smooth_dist > smooth_dist[filt] {
                max_smooth_dist
            } else {
                smooth_dist[filt]
            };
            let _ = max_smooth_dist;
            if max_as_best < self.m_num_as_best_filt[filt] {
                ind_of_max = filt as i32;
                max_as_best = self.m_num_as_best_filt[filt];
            }

            // On the last fit, we now know the best filter and can manage the
            // hybrid values
            if filt == (self.m_num_filters - 1) as usize {
                // Determine the best filter, discounting ones that have a much
                // lower distance than the maximum distance.
                for ind in 0..self.m_num_filters as usize {
                    if !self.m_picked_best_filt
                        && self.m_num_as_best_filt[ind] > -self.m_failed_often_crit
                        && !failed[ind]
                        && err_measure[ind] < min_error
                        && !(ind as i32 >= ind_max_dist && fit_dist[ind] < dist_crit * max_fit_dist)
                        && !(max_as_best
                            >= self.m_num_as_best_filt[ind] + self.m_pick_diff_crit as i32
                            && max_as_best as f32
                                >= self.m_pick_ratio_crit * self.m_num_as_best_filt[ind] as f32)
                    {
                        min_error = err_measure[ind];
                        self.m_best_filt = ind as i32;
                    }
                }

                let bf = self.m_best_filt as usize;
                let nf = self.m_num_filters as usize;
                self.m_res_mean_sum[nf] += res_mean[bf];
                self.m_pred_mean_sum[nf] += pred_mean[bf];
                self.m_res_sdsum[nf] += res_sd[bf];
                self.m_res_max_sum[nf] += max_wgt_res[bf];
                self.m_raw_max_sum[nf] += max_raw[bf];
                self.m_max_res_max[nf] = if self.m_max_res_max[nf] > max_wgt_res[bf] {
                    self.m_max_res_max[nf]
                } else {
                    max_wgt_res[bf]
                };
                self.m_max_raw_max[nf] = if self.m_max_raw_max[nf] > max_raw[bf] {
                    self.m_max_raw_max[nf]
                } else {
                    max_raw[bf]
                };
                self.m_num_fits += 1;
                if !self.m_picked_best_filt
                    && self.m_num_filters > 1
                    && self.m_num_frames >= self.m_num_all_vs_all + self.m_group_size - 1
                {
                    self.m_num_as_best_filt[self.m_best_filt as usize] += 1;

                    // 3/15/22: Do not pick a filter at this point unless the
                    // hybrid solution is actually going to be used
                    pickable = self.m_use_hybrid != 0;
                    for ind in 0..self.m_num_filters as usize {
                        if ind as i32 != ind_of_max
                            && (max_as_best
                                < self.m_num_as_best_filt[ind] + self.m_pick_diff_crit as i32
                                || (max_as_best as f32)
                                    < self.m_pick_ratio_crit * self.m_num_as_best_filt[ind] as f32)
                        {
                            pickable = false;
                        }
                    }
                    if pickable {
                        if self.m_debug != 0 {
                            util_print(
                                "After %d frames, picking filter %d as best\n",
                                &[
                                    CArg::Int(self.m_num_frames as i64),
                                    CArg::Int((ind_of_max + 1) as i64),
                                ],
                            );
                        }
                        self.m_best_filt = ind_of_max;
                        self.m_picked_best_filt = true;
                    }
                }
                if num_groups >= 4 && self.m_debug != 0 {
                    for ind in 0..self.m_num_filters as usize {
                        if (ind == ind_pred_min || ind == ind_res_min)
                            && ind_res_min != ind_pred_min
                        {
                            util_print(
                                "filt %d res %.3f%s  pred %.3f%s\n",
                                &[
                                    CArg::Int(ind as i64),
                                    CArg::Dbl(res_mean[ind] as f64),
                                    CArg::Str(if ind == ind_res_min { "*" } else { " " }),
                                    CArg::Dbl(pred_mean[ind] as f64),
                                    CArg::Str(if ind == ind_pred_min { "*" } else { " " }),
                                ],
                            );
                        }
                    }
                }
            }
            if self.m_debug > 1 {
                util_print(
                    "%sresidual: mean = %.2f, SD = %.2f, max = %.2f,  n = %d\n",
                    &[
                        CArg::Str(if do_robust { "weighted " } else { "" }),
                        CArg::Dbl(res_mean[filt] as f64),
                        CArg::Dbl(res_sd[filt] as f64),
                        CArg::Dbl(max_wgt_res[filt] as f64),
                        CArg::Int(num_groups as i64),
                    ],
                );
                if do_robust {
                    util_print(
                        "    unweighted max residual = %.2f, min weight = %.3f\n",
                        &[CArg::Dbl(max_raw[filt] as f64), CArg::Dbl(min_wgt as f64)],
                    );
                }
            }
        }
        let _ = num_data;
        num_data = 0;
        let _ = num_data;
    }
}

impl FrameAlign {
    /// C `FrameAlign::alignTwoFrames` (`framealign.cpp:2335`).
    ///
    /// Align the frame in `binArr` to the one in `refArr` and return the
    /// shifts.  Reference `refInd >= 0` for a saved frame, -1 for `mAlignSum`,
    /// -2 for `mWorkBinPad`.  Image to align: `aliInd >= 0` for a saved frame,
    /// -1 for `mWorkBinPad`.
    #[allow(clippy::too_many_arguments)]
    pub fn align_two_frames(
        &mut self,
        ref_ind: i32,
        ali_ind: i32,
        near_xshift: f32,
        near_yshift: f32,
        filt_ind: i32,
        x_shift: &mut f32,
        y_shift: &mut f32,
        filter_subarea: bool,
        dump: bool,
    ) -> i32 {
        let lim_xlo: i32;
        let lim_xhi: i32;
        let lim_ylo: i32;
        let lim_yhi: i32;
        let mut ind_peak: i32;
        // `float peaks[3]` etc. are uninitialised in the C and only two entries
        // are filled by `XCorrPeakFindWidth(..., 2, 0)`; `peaks[2]` is read in
        // the rejection test below.  See the progress note: native reads stack
        // residue there, this reads zero.
        let mut peaks = [0.0f32; 3];
        let mut xpeaks = [0.0f32; 3];
        let mut ypeaks = [0.0f32; 3];
        let mut widths = [0.0f32; 3];
        let mut min_widths = [0.0f32; 3];
        let mut x_temp = [0.0f32; 2];
        let mut y_temp = [0.0f32; 2];
        let mut exp_dist = [0.0f32; 2];
        let at_zero_crit = (0.1 / self.m_bin_align as f64) as f32;
        let peak_ratio_crit = 2.0f32;
        let third_peak_crit = 3.0f32;
        let exp_dist_ratio_crit = 2.0f32;
        let min_exp_dist = 4.0f32;
        let width_ratio_crit = 0.8f32;
        let use_subarea = filter_subarea || self.m_gpu_aligning;
        let sub_xoffset = if use_subarea {
            (((-near_xshift / self.m_bin_align as f32) as f64) + 0.5).floor() as i32
        } else {
            0
        };
        let sub_yoffset = if use_subarea {
            (((-near_yshift / self.m_bin_align as f32) as f64) + 0.5).floor() as i32
        } else {
            0
        };
        let ali_xsize = if use_subarea {
            self.m_ali_filt_size
        } else {
            self.m_align_xpad
        };
        let ali_ysize = if use_subarea {
            self.m_ali_filt_size
        } else {
            self.m_align_ypad
        };
        let use_groups = self.m_group_size > 1;

        // Going to store shifts but they will be negative because we are
        // getting shift to align reference to frame.  So take negative shift.
        lim_xlo = ((-near_xshift - self.m_max_shift as f32) / self.m_bin_align as f32) as i32
            - sub_xoffset;
        lim_xhi = (((-near_xshift + self.m_max_shift as f32) / self.m_bin_align as f32) as f64)
            .ceil() as i32
            - sub_xoffset;
        lim_ylo = ((-near_yshift - self.m_max_shift as f32) / self.m_bin_align as f32) as i32
            - sub_yoffset;
        lim_yhi = (((-near_yshift + self.m_max_shift as f32) / self.m_bin_align as f32) as f64)
            .ceil() as i32
            - sub_yoffset;

        if filt_ind == 0 {
            if self.m_gpu_aligning {
                // For GPU alignment, it extracts the wrapped image with origin
                // in center which is ready for filtering the subarea
                let mut temp_sub = std::mem::take(&mut self.m_temp_sub_filt);
                let rc = fgpu_cross_correlate(ali_ind, ref_ind, &mut temp_sub, sub_xoffset,
                                              sub_yoffset);
                self.m_temp_sub_filt = temp_sub;
                if rc != 0 {
                    if self.recover_gpu_align_ffts(
                        false,
                        -1,
                        if ref_ind == -1 {
                            BinRef::AlignSum
                        } else {
                            BinRef::Null
                        },
                        if ref_ind < -1 || ali_ind < 0 {
                            BinRef::Work
                        } else {
                            BinRef::Null
                        },
                        None,
                        false,
                    ) != 0
                    {
                        return 3;
                    }
                } else if !filter_subarea {
                    // But if we are not filtering, need to wrap back into corr array
                    let src = std::mem::take(&mut self.m_temp_sub_filt);
                    let mut dst = if use_subarea {
                        std::mem::take(&mut self.m_corr_filt_temp)
                    } else {
                        std::mem::take(&mut self.m_corr_bin_pad)
                    };
                    self.wrap_image(
                        &src,
                        ali_xsize + 2,
                        ali_xsize,
                        ali_ysize,
                        &mut dst,
                        ali_xsize + 2,
                        ali_xsize,
                        ali_ysize,
                        0,
                        0,
                    );
                    if use_subarea {
                        self.m_corr_filt_temp = dst;
                    } else {
                        self.m_corr_bin_pad = dst;
                    }
                    self.m_temp_sub_filt = src;
                }
            }

            if !self.m_gpu_aligning {
                // Assign arrays from indexes if not aligning on GPU
                let saved_len = if use_groups {
                    self.m_saved_groups.len() as i32
                } else {
                    self.m_saved_bin_pad.len() as i32
                };
                if ref_ind >= saved_len || ali_ind >= saved_len {
                    return 2;
                }
                let ref_ref = if ref_ind >= 0 {
                    if use_groups {
                        BinRef::Group(ref_ind as usize)
                    } else {
                        BinRef::Saved(ref_ind as usize)
                    }
                } else if ref_ind == -1 {
                    BinRef::AlignSum
                } else {
                    BinRef::Work
                };
                let bin_ref = if ali_ind >= 0 {
                    if use_groups {
                        BinRef::Group(ali_ind as usize)
                    } else {
                        BinRef::Saved(ali_ind as usize)
                    }
                } else {
                    BinRef::Work
                };

                // Copy into the correlation array
                let mut corr = std::mem::take(&mut self.m_corr_bin_pad);
                {
                    let bin = self.take_bin(bin_ref);
                    corr[..(self.m_align_bytes / 4) as usize]
                        .copy_from_slice(&bin[..(self.m_align_bytes / 4) as usize]);
                    self.put_bin(bin_ref, bin);
                }

                // Get product
                if self.m_report_times {
                    self.m_wall_start = wall_time();
                }
                {
                    let refa = self.take_bin(ref_ref);
                    conjugate_product(&mut corr, &refa, self.m_align_xpad, self.m_align_ypad);
                    self.put_bin(ref_ref, refa);
                }
                if self.m_report_times {
                    self.m_wall_conj_prod += wall_time() - self.m_wall_start;
                }

                // Inverse FFT
                if self.m_report_times {
                    self.m_wall_start = wall_time();
                }
                todfft_c(&mut corr, self.m_align_xpad, self.m_align_ypad, 1);
                if self.m_report_times {
                    self.m_wall_bin_fft += wall_time() - self.m_wall_start;
                }
                if dump && filter_subarea {
                    util_dump_image(
                        &corr,
                        self.m_align_xpad + 2,
                        self.m_align_xpad,
                        self.m_align_ypad,
                        1,
                        "lf correlation",
                        self.m_num_frames,
                    );
                }

                // If high frequency filter being applied to subarea, extract it
                if filter_subarea {
                    let mut temp_sub = std::mem::take(&mut self.m_temp_sub_filt);
                    self.wrap_image(
                        &corr,
                        self.m_align_xpad + 2,
                        self.m_align_xpad,
                        self.m_align_ypad,
                        &mut temp_sub,
                        self.m_ali_filt_size + 2,
                        self.m_ali_filt_size,
                        self.m_ali_filt_size,
                        sub_xoffset,
                        sub_yoffset,
                    );
                    self.m_temp_sub_filt = temp_sub;
                }
                self.m_corr_bin_pad = corr;
            }

            if filter_subarea {
                let mut temp_sub = std::mem::take(&mut self.m_temp_sub_filt);
                slice_taper_in_pad(
                    PadIn::InPlace,
                    MRC_MODE_FLOAT,
                    self.m_ali_filt_size + 2,
                    0,
                    self.m_ali_filt_size - 1,
                    0,
                    self.m_ali_filt_size - 1,
                    &mut temp_sub,
                    self.m_ali_filt_size + 2,
                    self.m_ali_filt_size,
                    self.m_ali_filt_size,
                    8,
                    8,
                );
                self.m_temp_sub_filt = temp_sub;
            }
        }

        // Filter subarea to temp array if doing that
        if filter_subarea {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let n = ((ali_xsize + 2) * ali_ysize) as usize;
            let mut wrap = std::mem::take(&mut self.m_wrap_temp);
            wrap[..n].copy_from_slice(&self.m_temp_sub_filt[..n]);
            todfft_c(&mut wrap, ali_xsize, ali_ysize, 0);
            for ind in 0..n {
                wrap[ind] *= self.m_sub_filt_mask[filt_ind as usize][ind];
            }
            todfft_c(&mut wrap, ali_xsize, ali_ysize, 1);
            let mut dst = if use_subarea {
                std::mem::take(&mut self.m_corr_filt_temp)
            } else {
                std::mem::take(&mut self.m_corr_bin_pad)
            };
            self.wrap_image(
                &wrap,
                ali_xsize + 2,
                ali_xsize,
                ali_ysize,
                &mut dst,
                ali_xsize + 2,
                ali_xsize,
                ali_ysize,
                0,
                0,
            );
            if use_subarea {
                self.m_corr_filt_temp = dst;
            } else {
                self.m_corr_bin_pad = dst;
            }
            self.m_wrap_temp = wrap;
            if self.m_report_times {
                self.m_wall_filter += wall_time() - self.m_wall_start;
            }
        }

        let corr_temp = if use_subarea {
            std::mem::take(&mut self.m_corr_filt_temp)
        } else {
            std::mem::take(&mut self.m_corr_bin_pad)
        };
        if dump {
            util_dump_image(
                &corr_temp,
                ali_xsize + 2,
                ali_xsize,
                ali_ysize,
                1,
                "correlation",
                self.m_num_frames,
            );
        }

        set_peak_find_limits(lim_xlo, lim_xhi, lim_ylo, lim_yhi, 1);
        xcorr_peak_find_width(
            &corr_temp,
            ali_xsize + 2,
            ali_ysize,
            &mut xpeaks,
            &mut ypeaks,
            &mut peaks,
            Some(&mut widths),
            Some(&mut min_widths),
            2,
            0.,
        );
        if use_subarea {
            self.m_corr_filt_temp = corr_temp;
        } else {
            self.m_corr_bin_pad = corr_temp;
        }
        ind_peak = 0;
        for ind in 0..2usize {
            if peaks[ind] > -1.0e29 {
                x_temp[ind] = -(sub_xoffset as f32 + xpeaks[ind]) * self.m_bin_align as f32;
                y_temp[ind] = -(sub_yoffset as f32 + ypeaks[ind]) * self.m_bin_align as f32;
                exp_dist[ind] = (((x_temp[ind] - near_xshift) as f64).powf(2.)
                    + ((y_temp[ind] - near_yshift) as f64).powf(2.))
                .sqrt() as f32;
                if ind != 0
                    && x_temp[0].abs() < at_zero_crit
                    && y_temp[0].abs() < at_zero_crit
                    && peaks[1] > peak_ratio_crit * peaks[0]
                    && widths[1] < width_ratio_crit * widths[0]
                    && peaks[2] < third_peak_crit * peaks[1]
                    && (exp_dist[1] < exp_dist_ratio_crit * exp_dist[0]
                        || (exp_dist[0] < min_exp_dist && exp_dist[1] < min_exp_dist))
                {
                    ind_peak = 1;
                    if self.m_debug != 0 {
                        util_print(
                            "reject peak at %.2f %.2f for %.2f %.2f\npeaks: %g %g %g  widths %.2f %.2f  expDist  %.2f %.2f\n",
                            &[
                                CArg::Dbl(x_temp[0] as f64),
                                CArg::Dbl(y_temp[0] as f64),
                                CArg::Dbl(x_temp[1] as f64),
                                CArg::Dbl(y_temp[1] as f64),
                                CArg::Dbl(peaks[0] as f64),
                                CArg::Dbl(peaks[1] as f64),
                                CArg::Dbl(peaks[2] as f64),
                                CArg::Dbl(widths[0] as f64),
                                CArg::Dbl(widths[1] as f64),
                                CArg::Dbl(exp_dist[0] as f64),
                                CArg::Dbl(exp_dist[1] as f64),
                            ],
                        );
                    }
                }
            }
        }

        *x_shift = x_temp[ind_peak as usize];
        *y_shift = y_temp[ind_peak as usize];
        ind_peak = 0;
        let _ = ind_peak;
        0
    }

    /// C `FrameAlign::addToSums` (`framealign.cpp:2498`).
    ///
    /// Shift and add image to full sum, and bin/pad image to cumulative
    /// alignment sum if `binInd >= -1`.
    pub fn add_to_sums(
        &mut self,
        mut full_arr: FullRef,
        sum_ind: i32,
        bin_ind: i32,
        frame_num: i32,
        mut filt_ind: i32,
    ) -> i32 {
        let mut ind: i32;
        if filt_ind < 0 {
            filt_ind = self.m_num_filters;
        }
        let mut x_shift = 0.;
        let mut y_shift = 0.;
        let use_odd_sum = frame_num % 2 != 0;
        self.frame_shift_from_groups(frame_num, filt_ind, &mut x_shift, &mut y_shift);
        if sum_ind == -1 && matches!(full_arr, FullRef::Null) {
            util_print(
                "Program error in addToSums: sumInd = -1 and fullArr is NULL\n",
                &[],
            );
            return 1;
        }

        // Shift full image and add into final sum if one is passed
        if sum_ind >= -1 {
            // Get a full-sized dose-weight filter regardless of binning because
            // that is needed for the GPU case
            if self.m_doing_dose_weighting {
                let n = self.m_dose_wgt_filter.len() as i32;
                let mut delta = 0.;
                dose_weight_filter(
                    self.m_prior_dose_cum,
                    self.m_prior_dose_cum + self.m_frame_doses[frame_num as usize],
                    self.m_pixel_size,
                    self.m_crit_dose_afac,
                    self.m_crit_dose_bfac,
                    self.m_crit_dose_cfac,
                    self.m_crit_dose_scale,
                    &mut self.m_dose_wgt_filter,
                    n,
                    0.71f32,
                    &mut delta,
                );
                self.m_dwfdelta = delta;
                if !self.m_reweight_filt.is_empty() {
                    for ind in 0..self.m_dose_wgt_filter.len() {
                        self.m_dose_wgt_filter[ind] *= self.m_reweight_filt[ind];
                    }
                }
                if self.m_debug > 1 {
                    util_print(
                        "1/pixel  Attenuation   Dose weight filter for frame %d:\n",
                        &[CArg::Int(frame_num as i64)],
                    );
                    let step = self.m_dose_wgt_filter.len() / 35;
                    let mut ind = 0usize;
                    while ind < self.m_dose_wgt_filter.len() {
                        // `framealign.cpp:2530` passes three arguments to a
                        // two-conversion format: the leading `ind` is an int
                        // and travels in a general register, so both `%.4f`
                        // conversions read the two doubles that follow.
                        util_print(
                            "%.4f  %.4f\n",
                            &[
                                CArg::Dbl((self.m_dwfdelta * ind as f32) as f64),
                                CArg::Dbl(self.m_dose_wgt_filter[ind] as f64),
                            ],
                        );
                        if self.m_dose_wgt_filter[ind] == 0. {
                            break;
                        }
                        ind += step;
                    }
                }
                self.m_prior_dose_cum += self.m_frame_doses[frame_num as usize];
            }

            // Replace fullArr if it is indeed the first one on the stack here
            if sum_ind >= 0
                && !self.m_saved_full_frame_num.is_empty()
                && (!self.m_gpu_summing
                    || !self.m_stack_unpad_on_gpu
                    || (self.m_stack_unpad_on_gpu && self.m_gpu_stack_limit > 0))
            {
                if self.m_saved_full_frame_num[0] == frame_num {
                    full_arr = FullRef::Saved(0);
                } else if !self.m_gpu_summing || !self.m_stack_unpad_on_gpu {
                    util_print(
                        "Next frame to be summed is not the first on the saved memory stack\n",
                        &[],
                    );
                    return 1;
                }
            }

            // Try to do sum on GPU if flag set
            if self.m_gpu_summing {
                ind = self.m_dose_wgt_filter.len() as i32;
                let full_owned = self.take_full(full_arr);
                let rc = fgpu_setup_dose_weighting(
                    if ind > 0 {
                        Some(&self.m_dose_wgt_filter)
                    } else {
                        None
                    },
                    ind,
                    self.m_dwfdelta,
                ) != 0
                    || fgpu_add_to_full_sum(&full_owned, x_shift, y_shift) != 0;
                self.put_full(full_arr, full_owned);
                if rc {
                    let mut passed = full_arr;
                    if self.recover_from_summing_failure(Some(&mut passed), frame_num, sum_ind)
                        != 0
                    {
                        return 3;
                    }
                    full_arr = passed;
                }
            }

            // Do sum into arrays here
            if self.m_bin_sum > 1 && !self.m_gpu_summing {
                if self.m_report_times {
                    self.m_wall_start = wall_time();
                }
                {
                    let full_owned = self.take_full(full_arr);
                    let mut reduce = std::mem::take(&mut self.m_reduce_temp);
                    let mut temp = std::mem::take(&mut self.m_shift_temp);
                    fourier_reduce_image(
                        &full_owned,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        &mut reduce,
                        self.m_sum_xpad,
                        self.m_sum_ypad,
                        x_shift,
                        y_shift,
                        Some(&mut temp),
                    );
                    self.m_shift_temp = temp;
                    self.m_reduce_temp = reduce;
                    self.put_full(full_arr, full_owned);
                }
                if self.m_report_times {
                    self.m_wall_reduce += wall_time() - self.m_wall_start;
                    self.m_wall_start = wall_time();
                }

                // Just scale the delta by the binning to use the initial part
                // of the filter on already-reduced images
                if self.m_doing_dose_weighting {
                    let reduce = std::mem::take(&mut self.m_reduce_temp);
                    let mut sum = if use_odd_sum {
                        std::mem::take(&mut self.m_full_odd_sum)
                    } else {
                        std::mem::take(&mut self.m_full_even_sum)
                    };
                    self.filter_and_add_to_sum(
                        &reduce,
                        &mut sum,
                        self.m_sum_xpad,
                        self.m_sum_ypad,
                        &self.m_dose_wgt_filter,
                        self.m_dwfdelta * self.m_bin_sum as f32,
                    );
                    if use_odd_sum {
                        self.m_full_odd_sum = sum;
                    } else {
                        self.m_full_even_sum = sum;
                    }
                    if self.m_make_unwgt_sum != 0 && !self.m_unwgt_on_gpu {
                        for ind in 0..((self.m_sum_xpad + 2) * self.m_sum_ypad) as usize {
                            self.m_unweight_sum[ind] += reduce[ind];
                        }
                    }
                    self.m_reduce_temp = reduce;
                } else {
                    let reduce = std::mem::take(&mut self.m_reduce_temp);
                    let sum = if use_odd_sum {
                        &mut self.m_full_odd_sum
                    } else {
                        &mut self.m_full_even_sum
                    };
                    for ind in 0..((self.m_sum_xpad + 2) * self.m_sum_ypad) as usize {
                        sum[ind] += reduce[ind];
                    }
                    self.m_reduce_temp = reduce;
                }
                if self.m_report_times {
                    self.m_wall_filter += wall_time() - self.m_wall_start;
                }
            } else if !self.m_gpu_summing {
                if self.m_report_times {
                    self.m_wall_start = wall_time();
                }
                {
                    let mut full_owned = self.take_full(full_arr);
                    let mut temp = std::mem::take(&mut self.m_shift_temp);
                    fourier_shift_image(
                        &mut full_owned,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        x_shift,
                        y_shift,
                        &mut temp,
                    );
                    self.m_shift_temp = temp;
                    self.put_full(full_arr, full_owned);
                }
                if self.m_report_times {
                    self.m_wall_shift += wall_time() - self.m_wall_start;
                    self.m_wall_start = wall_time();
                }
                if self.m_doing_dose_weighting {
                    let full_owned = self.take_full(full_arr);
                    let mut sum = if use_odd_sum {
                        std::mem::take(&mut self.m_full_odd_sum)
                    } else {
                        std::mem::take(&mut self.m_full_even_sum)
                    };
                    self.filter_and_add_to_sum(
                        &full_owned,
                        &mut sum,
                        self.m_full_xpad,
                        self.m_full_ypad,
                        &self.m_dose_wgt_filter,
                        self.m_dwfdelta,
                    );
                    if use_odd_sum {
                        self.m_full_odd_sum = sum;
                    } else {
                        self.m_full_even_sum = sum;
                    }
                    if self.m_make_unwgt_sum != 0 && !self.m_unwgt_on_gpu {
                        for ind in 0..((self.m_full_xpad + 2) * self.m_full_ypad) as usize {
                            self.m_unweight_sum[ind] += full_owned[ind];
                        }
                    }
                    self.put_full(full_arr, full_owned);
                } else {
                    let full_owned = self.take_full(full_arr);
                    let sum = if use_odd_sum {
                        &mut self.m_full_odd_sum
                    } else {
                        &mut self.m_full_even_sum
                    };
                    for ind in 0..((self.m_full_xpad + 2) * self.m_full_ypad) as usize {
                        sum[ind] += full_owned[ind];
                    }
                    self.put_full(full_arr, full_owned);
                }
                if self.m_report_times {
                    self.m_wall_filter += wall_time() - self.m_wall_start;
                }
            }
        }

        // Roll the frame buffer and reduce the number saved
        if sum_ind >= -1 && (self.m_num_all_vs_all != 0 || self.m_defer_summing) {
            if !matches!(full_arr, FullRef::Null) && self.m_num_full_saved > 0 {
                util_roll_saved_frames(&mut self.m_saved_full_size, self.m_num_full_saved);
                self.m_num_full_saved -= 1;
                for ind in 0..self.m_num_full_saved as usize {
                    self.m_saved_full_frame_num[ind] = self.m_saved_full_frame_num[ind + 1];
                }
            } else if matches!(full_arr, FullRef::Null) && self.m_num_stacked_on_gpu > 0 {
                self.m_num_stacked_on_gpu -= 1;
            }
        }

        // If there is a legal binInd, shift it and add to align sum
        if bin_ind < -1 {
            return 0;
        }

        if self.m_gpu_aligning {
            // Shift and add: but don't bother shifting the source when doing
            // simple cum corr
            if fgpu_shift_add_to_align_sum(
                bin_ind,
                x_shift / self.m_bin_align as f32,
                y_shift / self.m_bin_align as f32,
                if bin_ind < 0 { 0 } else { 1 },
            ) != 0
            {
                return 3;
            }
        } else {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            if bin_ind >= self.m_saved_bin_pad.len() as i32 {
                return 2;
            }
            let bin_ref = if bin_ind < 0 {
                BinRef::Work
            } else {
                BinRef::Saved(bin_ind as usize)
            };
            let mut bin_owned = self.take_bin(bin_ref);
            let mut temp = std::mem::take(&mut self.m_shift_temp);
            fourier_shift_image(
                &mut bin_owned,
                self.m_align_xpad,
                self.m_align_ypad,
                x_shift / self.m_bin_align as f32,
                y_shift / self.m_bin_align as f32,
                &mut temp,
            );
            self.m_shift_temp = temp;
            for ind in 0..self.m_align_pix as usize {
                self.m_align_sum[ind] += bin_owned[ind];
            }
            self.put_bin(bin_ref, bin_owned);
            if self.m_report_times {
                self.m_wall_shift += wall_time() - self.m_wall_start;
            }
        }
        0
    }
}

impl FrameAlign {
    /// C `FrameAlign::prepareToFetchAlignFFTs` (`framealign.cpp:2689`).
    ///
    /// Determine how many frames need to be copied from GPU after a failure and
    /// allocate arrays here.
    pub fn prepare_to_fetch_align_ffts(&mut self, ali_frame_ind: i32) -> i32 {
        let mut num_bin_pads = 0;
        let mut num_groups = 0;
        fgpu_number_of_align_ffts(&mut num_bin_pads, &mut num_groups);
        if ali_frame_ind >= num_bin_pads {
            num_bin_pads = ali_frame_ind + 1;
        }
        let mut ind = self.m_saved_bin_pad.len() as i32;
        while ind < num_bin_pads {
            self.m_saved_bin_pad.push(vec![0.; self.m_align_pix as usize]);
            ind += 1;
        }
        let mut ind = self.m_saved_groups.len() as i32;
        while ind < num_groups {
            self.m_saved_groups.push(vec![0.; self.m_align_pix as usize]);
            ind += 1;
        }
        0
    }

    /// C `FrameAlign::recoverGpuAlignFFTs` (`framealign.cpp:2716`).
    #[allow(clippy::too_many_arguments)]
    pub fn recover_gpu_align_ffts(
        &mut self,
        saving: bool,
        ali_frame_ind: i32,
        align_sum: BinRef,
        work_arr: BinRef,
        bin_arr: Option<&mut BinRef>,
        stacking: bool,
    ) -> i32 {
        // Fix up the full stack if doing noise pad on GPU and cancel all initial
        // steps there
        if self.m_noise_pad_on_gpu && self.recover_gpu_full_stack(stacking, None) != 0 {
            self.cleanup();
            return 3;
        }
        self.cancel_initial_steps_on_gpu();

        // Get the arrays made up
        if self.prepare_to_fetch_align_ffts(if saving { ali_frame_ind } else { -1 }) != 0 {
            return 3;
        }

        // Get the data back
        let mut saved = std::mem::take(&mut self.m_saved_bin_pad);
        let mut groups = std::mem::take(&mut self.m_saved_groups);
        let mut sum_owned = self.take_bin(align_sum);
        let mut work_owned = self.take_bin(work_arr);
        let rc = fgpu_return_align_ffts(
            &mut saved,
            &mut groups,
            if matches!(align_sum, BinRef::Null) {
                None
            } else {
                Some(&mut sum_owned)
            },
            if matches!(work_arr, BinRef::Null) {
                None
            } else {
                Some(&mut work_owned)
            },
        );
        self.m_saved_bin_pad = saved;
        self.m_saved_groups = groups;
        self.put_bin(align_sum, sum_owned);
        self.put_bin(work_arr, work_owned);
        if rc != 0 {
            self.cleanup();
            return 3;
        }
        self.m_gpu_aligning = false;

        // Grouping with no refine at end was saved as real space, so need to
        // take FFTs
        if self.m_group_size > 1 && self.m_cum_align_at_end == 0 {
            let top = self.m_saved_bin_pad.len() as i32 - if saving { 1 } else { 0 };
            for ind in 0..top.max(0) as usize {
                let mut bin = std::mem::take(&mut self.m_saved_bin_pad[ind]);
                todfft_c(&mut bin, self.m_align_xpad, self.m_align_ypad, 0);
                for ix in 0..self.m_align_pix as usize {
                    bin[ix] *= self.m_full_filt_mask[ix];
                }
                self.m_saved_bin_pad[ind] = bin;
            }
        }

        // Take care of current align array if provided: put it on the stack
        if saving {
            if let Some(bin_arr) = bin_arr {
                let src = self.take_bin(*bin_arr);
                let n = (self.m_align_bytes / 4) as usize;
                self.m_saved_bin_pad[ali_frame_ind as usize][..n].copy_from_slice(&src[..n]);
                self.put_bin(*bin_arr, src);
                *bin_arr = BinRef::Saved(ali_frame_ind as usize);
            }
        }
        util_print("Switching to aligning with the CPU\n", &[]);
        0
    }

    /// C `FrameAlign::recoverGpuFullStack` (`framealign.cpp:2767`).
    ///
    /// Get back the stack of full-sized images from the GPU and get larger
    /// arrays for ones on the CPU stack if necessary; preprocess if needed, and
    /// noise pad them.
    pub fn recover_gpu_full_stack(
        &mut self,
        stacking: bool,
        mut bin_arr: Option<&mut FullRef>,
    ) -> i32 {
        let num_saved_start = self.m_num_full_saved;
        let mut frame_num: i32;
        let mut source_type: i32;
        let need_preprocess = (self.m_gain_ref.is_some()
            || self.m_trunc_limit > 0.
            || self.m_cam_size_x > 0)
            && (self.m_gpu_flags & GPU_DO_PREPROCESS) != 0;

        let total = num_saved_start + self.m_num_stacked_on_gpu + if stacking { 1 } else { 0 };
        for ind in 0..total {
            let mut alloced: Option<Vec<f32>> = None;
            if self.m_stack_type != MRC_MODE_FLOAT || ind >= self.m_saved_full_size.len() as i32 {
                alloced = Some(vec![0.; ((self.m_full_xpad + 2) * self.m_full_ypad) as usize]);
            }

            // Set source as stack if on CPU and save frame number
            let mut source: Vec<f32>;
            if ind < num_saved_start {
                source = std::mem::take(&mut self.m_saved_full_size[ind as usize]);
                frame_num = self.m_saved_full_frame_num[ind as usize];
            } else if ind < num_saved_start + self.m_num_stacked_on_gpu {
                source = alloced.clone().unwrap_or_default();
                frame_num = 0;
                if fgpu_return_stacked_frame(&mut source, &mut frame_num) != 0 {
                    return 1;
                }
                self.m_num_full_saved += 1;
            } else {
                let cur = bin_arr.as_deref().copied().unwrap_or(FullRef::Null);
                source = self.take_full(cur);
                frame_num = self.m_num_frames;
                self.m_num_full_saved += 1;
                if let Some(b) = bin_arr.as_deref_mut() {
                    *b = FullRef::Work;
                }
            }
            source_type = self.m_stack_type;

            // Preprocess into work array if needed
            if need_preprocess {
                let src = FrameData::Float(&source);
                let mut work = std::mem::take(&mut self.m_work_full_size);
                let _ = &mut work;
                self.m_work_full_size = work;
                let saved_gain = self.m_gain_ref.clone();
                let _ = saved_gain;
                self.pre_process_frame(src, None, self.m_defect_bin, FullRef::Work);
                source_type = MRC_MODE_FLOAT;
                source = std::mem::take(&mut self.m_work_full_size);
            }

            // Noise taper pad into the destination array
            let mut dest = match alloced {
                Some(v) => v,
                None => std::mem::take(&mut self.m_saved_full_size[ind as usize]),
            };
            let mut temp = std::mem::take(&mut self.m_shift_temp);
            slice_noise_taper_pad(
                PadIn::Float(&source),
                source_type,
                self.m_nx,
                self.m_ny,
                &mut dest,
                self.m_full_xpad + 2,
                self.m_full_xpad,
                self.m_full_ypad,
                self.m_noise_length,
                4,
                &mut temp,
            );
            self.m_shift_temp = temp;
            if need_preprocess {
                self.m_work_full_size = source;
            }

            if ind < self.m_saved_full_size.len() as i32 {
                self.m_saved_full_size[ind as usize] = dest;
                self.m_saved_full_frame_num[ind as usize] = frame_num;
            } else {
                self.m_saved_full_size.push(dest);
                self.m_saved_full_frame_num.push(frame_num);
            }
        }

        // Sort the stack by frame number
        for ind in 0..(self.m_num_full_saved - 1).max(0) as usize {
            for jnd in (ind + 1)..self.m_num_full_saved as usize {
                if self.m_saved_full_frame_num[jnd] < self.m_saved_full_frame_num[ind] {
                    self.m_saved_full_size.swap(ind, jnd);
                    self.m_saved_full_frame_num.swap(ind, jnd);
                }
            }
        }
        self.m_num_stacked_on_gpu = 0;
        0
    }

    /// C `FrameAlign::cancelInitialStepsOnGPU` (`framealign.cpp:2862`).
    pub fn cancel_initial_steps_on_gpu(&mut self) {
        if self.m_noise_pad_on_gpu || self.m_bin_pad_on_gpu {
            util_print(
                "Switching to preprocessing and other initial steps on CPU\n",
                &[],
            );
        }
        self.m_noise_pad_on_gpu = false;
        self.m_bin_pad_on_gpu = false;
        self.m_stack_unpad_on_gpu = false;
        self.m_gpu_flags &=
            !(GPU_DO_NOISE_TAPER | GPU_DO_BIN_PAD | STACK_FULL_ON_GPU | GPU_DO_PREPROCESS);
        self.m_flags_for_unpad_call &=
            !(GPU_DO_NOISE_TAPER | GPU_DO_BIN_PAD | STACK_FULL_ON_GPU | GPU_DO_PREPROCESS);
        fgpu_set_unpadded_size(
            self.m_nx,
            self.m_ny,
            0,
            (if self.m_debug != 0 { 1 } else { 0 })
                + (if self.m_report_times { 10 } else { 0 }),
        );
    }

    /// C `FrameAlign::recoverFromSummingFailure` (`framealign.cpp:2881`).
    pub fn recover_from_summing_failure(
        &mut self,
        mut full_arr: Option<&mut FullRef>,
        frame_num: i32,
        sum_ind: i32,
    ) -> i32 {
        // Recover by getting existing sum back and taking FFT of current array
        util_print("Switching to summing on CPU\n", &[]);
        if self.m_noise_pad_on_gpu {
            if self.recover_gpu_full_stack(false, None) != 0 {
                self.cleanup();
                return 3;
            }
            if self.m_saved_full_frame_num[0] == frame_num {
                if let Some(f) = full_arr.as_deref_mut() {
                    *f = FullRef::Saved(0);
                }
            } else {
                util_print(
                    "After recovering from GPU failure, frame to be summed is not first in stack\n",
                    &[],
                );
                self.cleanup();
                return 3;
            }
        }
        self.cancel_initial_steps_on_gpu();
        if frame_num > 0 {
            let mut even = std::mem::take(&mut self.m_full_even_sum);
            let mut odd = std::mem::take(&mut self.m_full_odd_sum);
            // The C passes `mFullEvenSum` for both the sum and the even array.
            let mut scratch = even.clone();
            let rc = fgpu_return_sums(&mut even, &mut scratch, &mut odd, 1);
            self.m_full_even_sum = even;
            self.m_full_odd_sum = odd;
            if rc != 0 {
                self.cleanup();
                return 3;
            }
        } else {
            // `framealign.cpp:2905-2906` omits the `* sizeof(float)` on both
            // memsets, so only the first quarter of each sum is zeroed.
            let n = ((self.m_sum_xpad + 2) * self.m_sum_ypad / 4) as usize;
            self.m_full_even_sum[..n].fill(0.);
            self.m_full_odd_sum[..n].fill(0.);
        }
        self.m_gpu_summing = false;
        if sum_ind < 0 {
            if let Some(f) = full_arr.as_deref_mut() {
                let target = *f;
                let mut owned = self.take_full(target);
                todfft_c(&mut owned, self.m_full_xpad, self.m_full_ypad, 0);
                self.put_full(target, owned);
            }
        }
        for ind in 0..self.m_num_full_saved as usize {
            let mut owned = std::mem::take(&mut self.m_saved_full_size[ind]);
            todfft_c(&mut owned, self.m_full_xpad, self.m_full_ypad, 0);
            self.m_saved_full_size[ind] = owned;
        }
        0
    }

    /// C `FrameAlign::getUnweightedSum` (`framealign.cpp:2303`).
    ///
    /// Get the unweighted sum that was made in tandem with the dose-weighted one.
    pub fn get_unweighted_sum(&mut self, non_dwsum: &mut [f32]) -> i32 {
        let mut nx_bin = self.m_nx / self.m_bin_sum;
        let mut ny_bin = self.m_ny / self.m_bin_sum;
        let x_start = (self.m_sum_xpad - nx_bin) / 2;
        let x_end = x_start + nx_bin - 1;
        let y_start = (self.m_sum_ypad - ny_bin) / 2;
        let y_end = y_start + ny_bin - 1;
        if self.m_make_unwgt_sum == 0
            || (self.m_make_unwgt_sum != 0 && self.m_unwgt_on_gpu && !self.m_gpu_summing)
        {
            util_print("There is no unweighted sum available", &[]);
            return 1;
        }
        if self.m_unwgt_on_gpu {
            if fgpu_return_unweighted_sum(&mut self.m_unweight_sum) != 0 {
                return 1;
            }
        } else {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            let mut sum = std::mem::take(&mut self.m_unweight_sum);
            todfft_c(&mut sum, self.m_sum_xpad, self.m_sum_ypad, 1);
            self.m_unweight_sum = sum;
            if self.m_report_times {
                self.m_wall_full_fft += wall_time() - self.m_wall_start;
            }
        }

        extract_with_binning(
            float_bytes(&self.m_unweight_sum),
            MRC_MODE_FLOAT,
            self.m_sum_xpad + 2,
            x_start,
            x_end,
            y_start,
            y_end,
            1,
            float_bytes_mut(non_dwsum),
            0,
            &mut nx_bin,
            &mut ny_bin,
        );
        0
    }
}

impl FrameAlign {
    /// C `FrameAlign::finishAlignAndSum` (`framealign.cpp:1911`).
    ///
    /// Finish aligning and averaging the remaining frames and return results.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_align_and_sum(
        &mut self,
        mut refine_radius2: f32,
        mut refine_sigma2: f32,
        iter_crit: f32,
        mut group_refine: i32,
        do_spline: i32,
        alisum: &mut [f32],
        x_shifts: &mut [f32],
        y_shifts: &mut [f32],
        raw_xshifts: &mut [f32],
        raw_yshifts: &mut [f32],
        ring_corrs: Option<&mut [f32]>,
        delta_r: f32,
        best_filt: &mut i32,
        smooth_dist: &mut [f32],
        raw_dist: &mut [f32],
        res_mean: &mut [f32],
        pred_mean: &mut [f32],
        mean_res_max: &mut [f32],
        max_res_max: &mut [f32],
        mean_raw_max: &mut [f32],
        max_raw_max: &mut [f32],
        mut even_sum: Option<&mut [f32]>,
        mut odd_sum: Option<&mut [f32]>,
    ) -> i32 {
        let mut ind: i32;
        let num_pix: i32;
        let mut max_as_best: i32;
        let mut use_filt: i32;
        let mut ierr: i32;
        let mut use_frame: i32;
        let num_align: i32;
        let mut shift_x = 0.;
        let mut shift_y = 0.;
        let mut error: f32;
        let mut min_error: f32;
        let mut max_refine: f32;
        let mut ref_xshift: Vec<f32> = Vec::new();
        let mut ref_yshift: Vec<f32> = Vec::new();
        let mut cum_xshift: Vec<f32>;
        let mut cum_yshift: Vec<f32>;
        let mut bin_ref: BinRef;
        let mut nx_bin = self.m_nx / self.m_bin_sum;
        let mut ny_bin = self.m_ny / self.m_bin_sum;
        let x_start = (self.m_sum_xpad - nx_bin) / 2;
        let x_end = x_start + nx_bin - 1;
        let y_start = (self.m_sum_ypad - ny_bin) / 2;
        let y_end = y_start + ny_bin - 1;
        let process_full = self.m_summing_mode <= 0 && !self.m_defer_summing;
        let mut wall_refine = 0.0f64;
        let num_ava_for_frames = self.m_num_all_vs_all + self.m_group_size - 1;
        let mut ring_corrs = ring_corrs;
        // `float *realSum = mFullEvenSum;` -- false until the GPU hands back a
        // real sum in mWorkFullSize.
        let mut real_sum_is_work = false;
        let mut even_source_is_output = false;

        // If nothing is aligned, it is an error
        if self.m_num_frames == 0 {
            return 1;
        }

        // Finish up with all-vs-all
        if self.m_num_all_vs_all != 0 && self.m_summing_mode >= 0 {
            self.find_all_vs_all_alignment(false);

            // pick a best filter if haven't got one yet
            if self.m_num_filters > 1 && !self.m_picked_best_filt {
                max_as_best = 0;
                for ind in 0..self.m_num_filters as usize {
                    max_as_best = if max_as_best > self.m_num_as_best_filt[ind] {
                        max_as_best
                    } else {
                        self.m_num_as_best_filt[ind]
                    };
                }
                self.m_best_filt = 0;
                min_error = 1.0e30;
                for ind in 0..self.m_num_filters as usize {
                    error = (1. - self.m_max_max_weight) * self.m_pred_mean_sum[ind]
                        / 1.max(self.m_num_fits) as f32
                        + self.m_max_max_weight * self.m_max_res_max[ind];
                    if self.m_num_as_best_filt[ind] > -self.m_failed_often_crit
                        && error < min_error
                        && !(max_as_best
                            >= self.m_num_as_best_filt[ind] + self.m_pick_diff_crit as i32
                            && max_as_best as f32
                                >= self.m_pick_ratio_crit * self.m_num_as_best_filt[ind] as f32)
                    {
                        min_error = error;
                        self.m_best_filt = ind as i32;
                    }
                }
            }

            // Then take care of shifts
            if self.m_num_frames <= num_ava_for_frames {
                // Take shifts as is if never got any before and add these images
                for ind in 0..self.m_num_frames {
                    let mut filt = 0;
                    while filt <= self.m_num_filters
                        && ind < self.m_num_frames + 1 - self.m_group_size
                    {
                        use_filt = if filt == self.m_num_filters {
                            self.m_best_filt
                        } else {
                            filt
                        };
                        let x = self.m_xfit_shifts[use_filt as usize][ind as usize];
                        let y = self.m_yfit_shifts[use_filt as usize][ind as usize];
                        self.m_xshifts[filt as usize].push(x);
                        self.m_yshifts[filt as usize].push(y);
                        filt += 1;
                    }
                    if process_full && self.add_to_sums(FullRef::Null, ind, -9, ind, -1) != 0 {
                        return 3;
                    }
                }
            } else {
                // Or adjust and add in ALL the shifts this time and add the images
                let mut filt = 0;
                while filt <= self.m_num_filters {
                    use_filt = if filt == self.m_num_filters {
                        self.m_best_filt
                    } else {
                        filt
                    };
                    self.adjust_and_push_shifts(self.m_num_all_vs_all - 1, filt, use_filt);
                    filt += 1;
                }
                for ind in 1..num_ava_for_frames {
                    if process_full
                        && self.add_to_sums(
                            FullRef::Null,
                            ind - 1,
                            -9,
                            self.m_num_frames + ind - num_ava_for_frames,
                            -1,
                        ) != 0
                    {
                        return 3;
                    }
                }
            }
        }
        *best_filt = self.m_best_filt;
        use_filt = if self.m_use_hybrid != 0 {
            self.m_num_filters
        } else {
            self.m_best_filt
        };
        let uf = use_filt as usize;

        // Convert the shifts from group to frame
        let _group_xshift = self.m_xshifts[uf].clone();
        let _group_yshift = self.m_yshifts[uf].clone();
        self.get_all_frame_shifts(&mut ref_xshift, &mut ref_yshift, use_filt);
        self.m_xshifts[uf] = ref_xshift.clone();
        self.m_yshifts[uf] = ref_yshift.clone();

        // Now do a refinement with alignment to leave-one-out reference
        if self.m_cum_align_at_end != 0 && self.m_summing_mode >= 0 {
            if self.m_report_times {
                self.m_wall_start = wall_time();
            }
            if self.m_debug != 0 {
                let xs = std::mem::take(&mut self.m_xshifts[uf]);
                let ys = std::mem::take(&mut self.m_yshifts[uf]);
                smooth_dist[uf] = self.smoothed_total_distance(
                    &xs,
                    &ys,
                    xs.len() as i32,
                    &mut raw_dist[uf],
                    None,
                    None,
                    None,
                );
                self.m_xshifts[uf] = xs;
                self.m_yshifts[uf] = ys;
                util_print(
                    "Original distance raw %.2f  smoothed %.2f\n",
                    &[
                        CArg::Dbl(raw_dist[uf] as f64),
                        CArg::Dbl(smooth_dist[uf] as f64),
                    ],
                );
            }

            // Get a new high-frequency filter mask
            num_pix = self.m_align_pix;
            if refine_radius2 == 0. {
                refine_radius2 = self.m_radius2[self.m_best_filt as usize];
                refine_sigma2 = self.m_sigma2[self.m_best_filt as usize];
            }
            let mut delta = 0.;
            xcorr_set_ctf(
                0.,
                refine_sigma2 * self.m_bin_align as f32,
                0.,
                refine_radius2 * self.m_bin_align as f32,
                &mut self.m_filt_func,
                self.m_align_xpad,
                self.m_align_ypad,
                &mut delta,
            );
            self.m_filt_delta = delta;
            for ind in 0..num_pix as usize {
                self.m_full_filt_mask[ind] = 1.;
            }
            let ctf = self.m_filt_func;
            xcorr_filter_part(
                FilterIn::InPlace,
                &mut self.m_full_filt_mask,
                self.m_align_xpad,
                self.m_align_ypad,
                &ctf,
                self.m_filt_delta,
            );
            if self.m_gpu_aligning && fgpu_new_filter_mask(&self.m_full_filt_mask) != 0 {
                if self.recover_gpu_align_ffts(false, -1, BinRef::Null, BinRef::Null, None, false)
                    != 0
                {
                    return 3;
                }
            }

            // The real accumulated shifts are kept in cumXYshift; the mXYshifts
            // are the ones that get applied to frames on each iteration
            if self.m_group_size > 1 && group_refine != 0 {
                num_align = self.m_num_frames + 1 - self.m_group_size;
            } else {
                group_refine = 0;
                num_align = self.m_num_frames;
            }
            cum_xshift = self.m_xshifts[uf].clone();
            cum_yshift = self.m_yshifts[uf].clone();

            for iter in 0..self.m_cum_align_at_end {
                self.m_group_size = 1;
                if self.m_gpu_aligning {
                    fgpu_set_group_size(1);
                }
                ref_xshift.clear();
                ref_yshift.clear();
                if iter != 0 {
                    if self.m_gpu_aligning && fgpu_clear_align_sum() != 0 {
                        return 3;
                    } else if !self.m_gpu_aligning {
                        let n = (self.m_align_bytes / 4) as usize;
                        self.m_align_sum[..n].fill(0.);
                    }
                }

                // Loop on frames to shift the bin pad image into alignment and
                // add to sum.  Skip this on the first iteration for cumulative
                // alignment
                if iter != 0 || self.m_num_all_vs_all != 0 {
                    for frame in 0..self.m_num_frames {
                        ierr = self.add_to_sums(FullRef::Null, -9, frame, frame, use_filt);
                        if ierr != 0 {
                            return ierr;
                        }
                    }
                }

                // Loop on frames to align
                max_refine = 0.;
                for frame in 0..num_align {
                    use_frame = frame;
                    bin_ref = BinRef::Saved(frame as usize);

                    // Subtract this frame from the align sum and filter it
                    if group_refine != 0 {
                        // For group refine, add up the frames in group
                        self.m_group_size = self.m_group_size_initial;
                        fgpu_set_group_size(self.m_group_size);
                        use_frame = 0;

                        if self.m_gpu_aligning
                            && fgpu_sum_into_group(frame + self.m_group_size - 1, 0) != 0
                            && self.recover_gpu_align_ffts(
                                false,
                                -1,
                                BinRef::AlignSum,
                                BinRef::Null,
                                None,
                                false,
                            ) != 0
                        {
                            return 3;
                        }
                        if !self.m_gpu_aligning {
                            bin_ref = BinRef::Group(0);
                            let mut group = self.take_bin(bin_ref);
                            let n = (self.m_align_bytes / 4) as usize;
                            group[..n].fill(0.);
                            for bin_ind in frame..frame + self.m_group_size {
                                for ind in 0..self.m_align_pix as usize {
                                    group[ind] += self.m_saved_bin_pad[bin_ind as usize][ind];
                                }
                            }
                            self.put_bin(bin_ref, group);
                        }
                    }

                    // If on GPU, that now needs subtracting and filtering
                    if self.m_gpu_aligning
                        && fgpu_subtract_and_filter_align_sum(use_frame, group_refine) != 0
                    {
                        if self.recover_gpu_align_ffts(
                            false,
                            -1,
                            BinRef::AlignSum,
                            BinRef::Null,
                            None,
                            false,
                        ) != 0
                        {
                            return 3;
                        }
                        bin_ref = if group_refine != 0 {
                            BinRef::Group(0)
                        } else {
                            BinRef::Saved(frame as usize)
                        };
                    }

                    // Or, subtract and filter on CPU
                    if !self.m_gpu_aligning {
                        let bin = self.take_bin(bin_ref);
                        for ind in 0..num_pix as usize {
                            self.m_work_bin_pad[ind] =
                                (self.m_align_sum[ind] - bin[ind]) * self.m_full_filt_mask[ind];
                        }
                        self.put_bin(bin_ref, bin);
                    }

                    // Align it to LOO sum.  May want to pass a smaller max shift
                    ierr = self.align_two_frames(
                        -2,
                        use_frame,
                        0.,
                        0.,
                        0,
                        &mut shift_x,
                        &mut shift_y,
                        false,
                        self.m_dump_ref_corrs,
                    );
                    if ierr != 0 {
                        return ierr;
                    }
                    ref_xshift.push(shift_x);
                    ref_yshift.push(shift_y);
                    error = (shift_x * shift_x + shift_y * shift_y).sqrt();
                    max_refine = if max_refine > error { max_refine } else { error };
                }

                if group_refine != 0 {
                    for frame in 0..self.m_num_frames as usize {
                        self.m_xshifts[uf][frame] = 0.;
                        self.m_yshifts[uf][frame] = 0.;
                    }
                    for frame in 0..num_align {
                        for ind in frame..frame + self.m_group_size {
                            self.m_xshifts[uf][ind as usize] +=
                                ref_xshift[frame as usize] / self.m_group_size as f32;
                            self.m_yshifts[uf][ind as usize] +=
                                ref_yshift[frame as usize] / self.m_group_size as f32;
                        }
                    }
                    ref_xshift = self.m_xshifts[uf].clone();
                    ref_yshift = self.m_yshifts[uf].clone();
                } else {
                    // And copy the refineshift over to be applied next time
                    self.m_xshifts[uf] = ref_xshift.clone();
                    self.m_yshifts[uf] = ref_yshift.clone();
                }

                // Adjust the shifts: add them to cumulative shift
                for frame in 0..self.m_num_frames as usize {
                    cum_xshift[frame] += ref_xshift[frame];
                    cum_yshift[frame] += ref_yshift[frame];
                    if self.m_debug != 0 {
                        util_print(
                            "%d %2d %.2f  %.2f   %.2f  %.2f\n",
                            &[
                                CArg::Int(iter as i64),
                                CArg::Int(frame as i64),
                                CArg::Dbl(ref_xshift[frame] as f64),
                                CArg::Dbl(ref_yshift[frame] as f64),
                                CArg::Dbl(cum_xshift[frame] as f64),
                                CArg::Dbl(cum_yshift[frame] as f64),
                            ],
                        );
                    }
                }

                if max_refine < iter_crit {
                    break;
                }
            }

            // At end, put the full shifts back
            self.m_xshifts[uf] = cum_xshift;
            self.m_yshifts[uf] = cum_yshift;
            if self.m_report_times {
                wall_refine += wall_time() - self.m_wall_start;
            }
        }

        // adjust shifts for initial cumulative alignment to have a mean of 0
        if self.m_num_all_vs_all == 0 && (self.m_defer_summing || self.m_summing_mode > 0) {
            shift_x = 0.;
            shift_y = 0.;
            for ind in 0..self.m_num_frames as usize {
                shift_x += self.m_xshifts[uf][ind] / self.m_num_frames as f32;
                shift_y += self.m_yshifts[uf][ind] / self.m_num_frames as f32;
            }
            for ind in 0..self.m_num_frames as usize {
                self.m_xshifts[uf][ind] -= shift_x;
                self.m_yshifts[uf][ind] -= shift_y;
            }
        }

        // Save to raw shifts and apply spline smoothing now before shifts get used
        for ind in 0..self.m_num_frames as usize {
            raw_xshifts[ind] = self.m_xshifts[uf][ind];
            raw_yshifts[ind] = self.m_yshifts[uf][ind];
        }

        // Spline smoothing: Get the raw distance first, use the spline as
        // smoothed distance
        if do_spline != 0 && self.m_summing_mode >= 0 {
            let xs = std::mem::take(&mut self.m_xshifts[uf]);
            let ys = std::mem::take(&mut self.m_yshifts[uf]);
            self.smoothed_total_distance(
                &xs,
                &ys,
                xs.len() as i32,
                &mut raw_dist[uf],
                None,
                None,
                None,
            );
            self.m_xshifts[uf] = xs;
            self.m_yshifts[uf] = ys;
            let mut xs = std::mem::take(&mut self.m_xshifts[uf]);
            let mut ys = std::mem::take(&mut self.m_yshifts[uf]);
            ind = self.spline_smooth(
                raw_xshifts,
                raw_yshifts,
                self.m_num_frames,
                &mut xs,
                &mut ys,
                &mut smooth_dist[uf],
            );
            self.m_xshifts[uf] = xs;
            self.m_yshifts[uf] = ys;
            if ind != 0 {
                util_print(
                    "Spline smoothing of shifts failed with return value %d",
                    &[CArg::Int(ind as i64)],
                );
                return 1;
            }
        }

        // Sum now if needed, do FRC, then inverse transform and extract the area
        self.m_group_size = 1;
        if self.m_summing_mode <= 0 {
            if self.m_defer_summing {
                // Make the sum
                if self.m_gpu_summing && (self.m_gpu_flags & GPU_FOR_ALIGNING) != 0 {
                    fgpu_clean_align_items();
                    fgpu_set_unpadded_size(
                        self.m_nx,
                        self.m_ny,
                        self.m_flags_for_unpad_call,
                        (if self.m_debug != 0 { 1 } else { 0 })
                            + (if self.m_report_times { 10 } else { 0 }),
                    );
                    if fgpu_setup_summing(
                        self.m_full_xpad,
                        self.m_full_ypad,
                        self.m_sum_xpad,
                        self.m_sum_ypad,
                        self.m_even_odd_for_sum_setup,
                    ) != 0
                        && self.recover_from_summing_failure(None, 0, 0) != 0
                    {
                        return 3;
                    }
                }
                for frame in 0..self.m_num_frames {
                    if self.add_to_sums(FullRef::Null, frame, -9, frame, use_filt) != 0 {
                        return 3;
                    }
                }
            }

            if self.m_gpu_summing {
                let mut work = std::mem::take(&mut self.m_work_full_size);
                let mut even = std::mem::take(&mut self.m_full_even_sum);
                let mut odd = std::mem::take(&mut self.m_full_odd_sum);
                ind = fgpu_return_sums(&mut work, &mut even, &mut odd, 0);
                self.m_work_full_size = work;
                self.m_full_even_sum = even;
                self.m_full_odd_sum = odd;

                // If there is no real sum and no even sum, we can't do anything
                if ind == 3 {
                    return 3;
                }
                // If there is an error getting the real sum but even/odd is
                // there, just cancel flags to process the FFT(s)
                if ind & 2 != 0 {
                    self.m_gpu_flags = 0;
                }
                if ind == 0 {
                    real_sum_is_work = true;
                }
                if (ind & 1) != 0 || (self.m_gpu_flags & GPU_DO_EVEN_ODD) == 0 {
                    ring_corrs = None;
                }
            }

            if let Some(rc) = ring_corrs.as_deref_mut() {
                let mut work = std::mem::take(&mut self.m_work_full_size);
                fourier_ring_corr(
                    &self.m_full_even_sum,
                    &self.m_full_odd_sum,
                    self.m_sum_xpad,
                    self.m_sum_ypad,
                    rc,
                    (0.5 / delta_r as f64).floor() as i32,
                    delta_r,
                    &mut work,
                );
                self.m_work_full_size = work;
            }
            if self.m_dump_even_odd {
                let mut even = std::mem::take(&mut self.m_full_even_sum);
                util_dump_fft(
                    &mut even,
                    self.m_sum_xpad,
                    self.m_sum_ypad,
                    "even sum",
                    1,
                    0,
                    0,
                );
                self.m_full_even_sum = even;
                let mut odd = std::mem::take(&mut self.m_full_odd_sum);
                util_dump_fft(
                    &mut odd,
                    self.m_sum_xpad,
                    self.m_sum_ypad,
                    "odd sum",
                    1,
                    0,
                    0,
                );
                self.m_full_odd_sum = odd;
            }
            if !self.m_gpu_summing || self.m_gpu_flags == 0 {
                // If doing even and odd, copy the even FFT to output buffer
                // before it is added to
                if even_sum.is_some() && odd_sum.is_some() {
                    even_source_is_output = true;
                    let es = even_sum.as_deref_mut().unwrap();
                    for ind in 0..((self.m_sum_xpad + 2) * self.m_sum_ypad) as usize {
                        es[ind] = self.m_full_even_sum[ind];
                    }
                }
                for ind in 0..((self.m_sum_xpad + 2) * self.m_sum_ypad) as usize {
                    self.m_full_even_sum[ind] += self.m_full_odd_sum[ind];
                }
                if self.m_report_times {
                    self.m_wall_start = wall_time();
                }
                let mut even = std::mem::take(&mut self.m_full_even_sum);
                todfft_c(&mut even, self.m_sum_xpad, self.m_sum_ypad, 1);
                self.m_full_even_sum = even;
                if self.m_report_times {
                    self.m_wall_full_fft += wall_time() - self.m_wall_start;
                }
            }
            {
                let real_sum: &[f32] = if real_sum_is_work {
                    &self.m_work_full_size
                } else {
                    &self.m_full_even_sum
                };
                extract_with_binning(
                    float_bytes(real_sum),
                    MRC_MODE_FLOAT,
                    self.m_sum_xpad + 2,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    1,
                    float_bytes_mut(alisum),
                    0,
                    &mut nx_bin,
                    &mut ny_bin,
                );
            }

            // Do the even and odd sums if requested
            if even_sum.is_some() && odd_sum.is_some() {
                let es = even_sum.as_deref_mut().unwrap();
                if even_source_is_output {
                    todfft_c(es, self.m_sum_xpad, self.m_sum_ypad, 1);
                } else {
                    let mut even = std::mem::take(&mut self.m_full_even_sum);
                    todfft_c(&mut even, self.m_sum_xpad, self.m_sum_ypad, 1);
                    self.m_full_even_sum = even;
                }
                let mut odd = std::mem::take(&mut self.m_full_odd_sum);
                todfft_c(&mut odd, self.m_sum_xpad, self.m_sum_ypad, 1);
                self.m_full_odd_sum = odd;
                {
                    let src: Vec<f32> = if even_source_is_output {
                        es.to_vec()
                    } else {
                        self.m_full_even_sum.clone()
                    };
                    extract_with_binning(
                        float_bytes(&src),
                        MRC_MODE_FLOAT,
                        self.m_sum_xpad + 2,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        1,
                        float_bytes_mut(es),
                        0,
                        &mut nx_bin,
                        &mut ny_bin,
                    );
                }
                let os = odd_sum.as_deref_mut().unwrap();
                extract_with_binning(
                    float_bytes(&self.m_full_odd_sum),
                    MRC_MODE_FLOAT,
                    self.m_sum_xpad + 2,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    1,
                    float_bytes_mut(os),
                    0,
                    &mut nx_bin,
                    &mut ny_bin,
                );
            }
        }

        // return best shifts
        for ind in 0..self.m_num_frames as usize {
            x_shifts[ind] = self.m_xshifts[uf][ind];
            y_shifts[ind] = self.m_yshifts[uf][ind];
        }

        // Return all the results
        let mut filt = 0;
        while filt <= self.m_num_filters {
            let f = filt as usize;
            if do_spline == 0 || filt != use_filt {
                let xs = std::mem::take(&mut self.m_xshifts[f]);
                let ys = std::mem::take(&mut self.m_yshifts[f]);
                smooth_dist[f] = self.smoothed_total_distance(
                    &xs,
                    &ys,
                    xs.len() as i32,
                    &mut raw_dist[f],
                    Some(&mut ref_xshift),
                    Some(&mut ref_yshift),
                    None,
                );
                self.m_xshifts[f] = xs;
                self.m_yshifts[f] = ys;
            }
            ind = 1.max(self.m_num_fits);
            res_mean[f] = self.m_res_mean_sum[f] / ind as f32;
            pred_mean[f] = self.m_pred_mean_sum[f] / ind as f32;
            mean_res_max[f] = self.m_res_max_sum[f] / ind as f32;
            max_res_max[f] = self.m_max_res_max[f];
            mean_raw_max[f] = self.m_raw_max_sum[f] / ind as f32;
            max_raw_max[f] = self.m_max_raw_max[f];
            self.m_pred_mean_sum[f] /= ind as f32;
            filt += 1;
        }
        if self.m_report_times {
            util_print(
                "FullFFT %.3f  BinPad %.3f  BinFFT %.3f  Reduce %.3f  Shift %.3f Filt %.3f\nConjProd %.3f   PreProc %.3f  Noise %.3f  Sum of those %.3f  Refine %.3f\n",
                &[
                    CArg::Dbl(self.m_wall_full_fft),
                    CArg::Dbl(self.m_wall_bin_pad),
                    CArg::Dbl(self.m_wall_bin_fft),
                    CArg::Dbl(self.m_wall_reduce),
                    CArg::Dbl(self.m_wall_shift),
                    CArg::Dbl(self.m_wall_filter),
                    CArg::Dbl(self.m_wall_conj_prod),
                    CArg::Dbl(self.m_wall_pre_proc),
                    CArg::Dbl(self.m_wall_noise),
                    CArg::Dbl(
                        self.m_wall_full_fft
                            + self.m_wall_bin_pad
                            + self.m_wall_bin_fft
                            + self.m_wall_reduce
                            + self.m_wall_shift
                            + self.m_wall_filter
                            + self.m_wall_conj_prod
                            + self.m_wall_pre_proc
                            + self.m_wall_noise,
                    ),
                    CArg::Dbl(wall_refine),
                ],
            );
        }
        if self.m_gpu_flags != 0 && self.m_report_times {
            fgpu_print_timers();
        }
        0
    }
}
