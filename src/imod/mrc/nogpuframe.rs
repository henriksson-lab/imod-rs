//! Translation of `IMOD/mrc/nogpuframe.cpp`, the no-GPU implementation of the
//! `fgpu*` entry points declared in `IMOD/mrc/gpuframe.h`.
//!
//! This is what the build links when it is configured without CUDA, which is
//! how the reference build at `/tmp/imod-reference-build` is configured.  Every
//! function is a stub: `fgpuGpuAvailable` reports no GPU and zero memory, and
//! every other `int`-returning entry point returns 1, which `framealign`
//! reads as failure and falls back to the CPU path.

use crate::imod::mrc::frameutil::CharArgType;

/// C `GPUFRAME_VERSION` (`gpuframe.h:20`).
pub const GPUFRAME_VERSION: i32 = 102;
/// C `NICE_GPU_DIVISOR` (`gpuframe.h:21`).
pub const NICE_GPU_DIVISOR: i32 = 32;
/// C `MAX_GPU_GROUP_SIZE` (`gpuframe.h:22`).
pub const MAX_GPU_GROUP_SIZE: i32 = 5;

/// C `fgpuGpuAvailable` (`nogpuframe.cpp:4`).
pub fn fgpu_gpu_available(_n_gpu: i32, memory: &mut f32, _debug: i32) -> i32 {
    *memory = 0.;
    0
}

/// C `fgpuSetUnpaddedSize` (`nogpuframe.cpp:10`).
pub fn fgpu_set_unpadded_size(_unpad_x: i32, _unpad_y: i32, _flags: i32, _debug: i32) {}

/// C `fgpuSetPreProcParams` (`nogpuframe.cpp:11`).
pub fn fgpu_set_pre_proc_params(
    _gain_ref: Option<&[f32]>,
    _nx_gain: i32,
    _ny_gain: i32,
    _trunc_limit: f32,
    _defect_map: Option<&[u8]>,
    _cam_size_x: i32,
    _cam_size_y: i32,
) -> i32 {
    1
}

/// C `fgpuSetBinPadParams` (`nogpuframe.cpp:14`).
#[allow(clippy::too_many_arguments)]
pub fn fgpu_set_bin_pad_params(
    _xstart: i32,
    _xend: i32,
    _ystart: i32,
    _yend: i32,
    _binning: i32,
    _nx_taper: i32,
    _ny_taper: i32,
    _type_0: i32,
    _filt_type: i32,
    _noise_len: i32,
) {
}

/// C `fgpuSetupSumming` (`nogpuframe.cpp:17`).
pub fn fgpu_setup_summing(
    _full_xpad: i32,
    _full_ypad: i32,
    _sum_xpad: i32,
    _sum_ypad: i32,
    _even_odd: i32,
) -> i32 {
    1
}

/// C `fgpuSetupAligning` (`nogpuframe.cpp:18`).
#[allow(clippy::too_many_arguments)]
pub fn fgpu_setup_aligning(
    _align_xpad: i32,
    _align_ypad: i32,
    _sum_xpad: i32,
    _sum_ypad: i32,
    _align_mask: &[f32],
    _ali_filt_size: i32,
    _group_size: i32,
    _expect_stack_size: i32,
    _do_align_sum: i32,
) -> i32 {
    1
}

/// C `fgpuSetupDoseWeighting` (`nogpuframe.cpp:21`).
pub fn fgpu_setup_dose_weighting(_filter: Option<&[f32]>, _filt_size: i32, _delta: f32) -> i32 {
    1
}

/// C `fgpuAddToFullSum` (`nogpuframe.cpp:22`).
pub fn fgpu_add_to_full_sum(_full_arr: &[f32], _shift_x: f32, _shift_y: f32) -> i32 {
    1
}

/// C `fgpuReturnSums` (`nogpuframe.cpp:23`).
pub fn fgpu_return_sums(
    _sum_arr: &mut [f32],
    _even_arr: &mut [f32],
    _odd_arr: &mut [f32],
    _even_odd_only: i32,
) -> i32 {
    1
}

/// C `fgpuReturnUnweightedSum` (`nogpuframe.cpp:24`).
pub fn fgpu_return_unweighted_sum(_sum_arr: &mut [f32]) -> i32 {
    1
}

/// C `fgpuCleanup` (`nogpuframe.cpp:25`).
pub fn fgpu_cleanup() {}

/// C `fgpuRollAlignStack` (`nogpuframe.cpp:26`).
pub fn fgpu_roll_align_stack() {}

/// C `fgpuRollGroupStack` (`nogpuframe.cpp:27`).
pub fn fgpu_roll_group_stack() {}

/// C `fgpuSubtractAndFilterAlignSum` (`nogpuframe.cpp:28`).
pub fn fgpu_subtract_and_filter_align_sum(_stack_ind: i32, _group_refine: i32) -> i32 {
    1
}

/// C `fgpuNewFilterMask` (`nogpuframe.cpp:29`).
pub fn fgpu_new_filter_mask(_align_mask: &[f32]) -> i32 {
    1
}

/// C `fgpuShiftAddToAlignSum` (`nogpuframe.cpp:30`).
pub fn fgpu_shift_add_to_align_sum(
    _stack_ind: i32,
    _shift_x: f32,
    _shift_y: f32,
    _shift_source: i32,
) -> i32 {
    1
}

/// C `fgpuCrossCorrelate` (`nogpuframe.cpp:31`).
pub fn fgpu_cross_correlate(
    _ali_ind: i32,
    _ref_ind: i32,
    _subarea: &mut [f32],
    _sub_xoffset: i32,
    _sub_yoffset: i32,
) -> i32 {
    1
}

/// C `fgpuProcessAlignImage` (`nogpuframe.cpp:33`).
pub fn fgpu_process_align_image(
    _bin_arr: &[f32],
    _stack_ind: i32,
    _group_ind: i32,
    _stack_on_gpu: i32,
) -> i32 {
    1
}

/// C `fgpuNumberOfAlignFFTs` (`nogpuframe.cpp:35`).
pub fn fgpu_number_of_align_ffts(_num_bin_pad: &mut i32, _num_groups: &mut i32) {}

/// C `fgpuReturnAlignFFTs` (`nogpuframe.cpp:36`).
///
/// The C takes `float **saved, float **groups` — the address of the first
/// element of `framealign`'s two stack vectors.  Nothing is written here, so
/// the stacks are passed as owned slices of owned buffers.
pub fn fgpu_return_align_ffts(
    _saved: &mut [Vec<f32>],
    _groups: &mut [Vec<f32>],
    _align_sum: Option<&mut [f32]>,
    _work_arr: Option<&mut [f32]>,
) -> i32 {
    1
}

/// C `fgpuReturnStackedFrame` (`nogpuframe.cpp:38`).
pub fn fgpu_return_stacked_frame(_array: &mut [f32], _frame_num: &mut i32) -> i32 {
    1
}

/// C `fgpuCleanSumItems` (`nogpuframe.cpp:39`).
pub fn fgpu_clean_sum_items() {}

/// C `fgpuCleanAlignItems` (`nogpuframe.cpp:40`).
pub fn fgpu_clean_align_items() {}

/// C `fgpuZeroTimers` (`nogpuframe.cpp:41`).
pub fn fgpu_zero_timers() {}

/// C `fgpuPrintTimers` (`nogpuframe.cpp:42`).
pub fn fgpu_print_timers() {}

/// C `fgpuClearAlignSum` (`nogpuframe.cpp:43`).
pub fn fgpu_clear_align_sum() -> i32 {
    1
}

/// C `fgpuSumIntoGroup` (`nogpuframe.cpp:44`).
pub fn fgpu_sum_into_group(_stack_ind: i32, _group_ind: i32) -> i32 {
    1
}

/// C `fgpuSetGroupSize` (`nogpuframe.cpp:45`).
pub fn fgpu_set_group_size(_in_val: i32) {}

/// C `fgpuGetVersion` (`nogpuframe.cpp:46`).
pub fn fgpu_get_version() -> i32 {
    1
}

/// C `fgpuSetPrintFunc` (`nogpuframe.cpp:47`).
pub fn fgpu_set_print_func(_func: CharArgType) {}
