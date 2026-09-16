//! Translation of the intentional no-GPU fallback `IMOD/mrc/nogpuframe.cpp`.

pub fn fgpu_gpu_available(_gpus: i32, memory: &mut f32, _debug: i32) -> i32 {
    *memory = 0.;
    0
}
pub fn fgpu_set_unpadded_size(_x: i32, _y: i32, _flags: i32, _debug: i32) {}
pub fn fgpu_set_pre_proc_params(
    _gain: &[f32],
    _nx: i32,
    _ny: i32,
    _trunc: f32,
    _defects: &[u8],
    _camera_x: i32,
    _camera_y: i32,
) -> i32 {
    1
}
#[allow(clippy::too_many_arguments)]
pub fn fgpu_set_bin_pad_params(
    _xs: i32,
    _xe: i32,
    _ys: i32,
    _ye: i32,
    _bin: i32,
    _xt: i32,
    _yt: i32,
    _kind: i32,
    _filter: i32,
    _noise: i32,
) {
}
pub fn fgpu_setup_summing(_x: i32, _y: i32, _sum_x: i32, _sum_y: i32, _evenodd: i32) -> i32 {
    1
}
#[allow(clippy::too_many_arguments)]
pub fn fgpu_setup_aligning(
    _x: i32,
    _y: i32,
    _sum_x: i32,
    _sum_y: i32,
    _mask: &[f32],
    _filter: i32,
    _group: i32,
    _stack: i32,
    _align_sum: i32,
) -> i32 {
    1
}
pub fn fgpu_setup_dose_weighting(_filter: &[f32], _size: i32, _delta: f32) -> i32 {
    1
}
pub fn fgpu_add_to_full_sum(_array: &[f32], _x: f32, _y: f32) -> i32 {
    1
}
pub fn fgpu_return_sums(_sum: &mut [f32], _even: &mut [f32], _odd: &mut [f32], _only: i32) -> i32 {
    1
}
pub fn fgpu_return_unweighted_sum(_sum: &mut [f32]) -> i32 {
    1
}
pub fn fgpu_cleanup() {}
pub fn fgpu_roll_align_stack() {}
pub fn fgpu_roll_group_stack() {}
pub fn fgpu_subtract_and_filter_align_sum(_stack: i32, _refine: i32) -> i32 {
    1
}
pub fn fgpu_new_filter_mask(_mask: &[f32]) -> i32 {
    1
}
pub fn fgpu_shift_add_to_align_sum(_stack: i32, _x: f32, _y: f32, _source: i32) -> i32 {
    1
}
pub fn fgpu_cross_correlate(
    _alignment: i32,
    _reference: i32,
    _subarea: &[f32],
    _x: i32,
    _y: i32,
) -> i32 {
    1
}
pub fn fgpu_process_align_image(_array: &[f32], _stack: i32, _group: i32, _on_gpu: i32) -> i32 {
    1
}
pub fn fgpu_number_of_align_ffts(_bin_pad: &mut i32, _groups: &mut i32) {}
pub fn fgpu_return_align_ffts(
    _saved: &mut [Vec<f32>],
    _groups: &mut [Vec<f32>],
    _sum: &mut [f32],
    _work: &mut [f32],
) -> i32 {
    1
}
pub fn fgpu_return_stacked_frame(_array: &mut [f32], _frame: &mut i32) -> i32 {
    1
}
pub fn fgpu_clean_sum_items() {}
pub fn fgpu_clean_align_items() {}
pub fn fgpu_zero_timers() {}
pub fn fgpu_print_timers() {}
pub fn fgpu_clear_align_sum() -> i32 {
    1
}
pub fn fgpu_sum_into_group(_stack: i32, _group: i32) -> i32 {
    1
}
pub fn fgpu_set_group_size(_value: i32) {}
pub fn fgpu_get_version() -> i32 {
    1
}
pub fn fgpu_set_print_func(_function: Option<fn(&str)>) {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fallback_is_explicit() {
        let mut memory = 1.;
        assert_eq!(fgpu_gpu_available(1, &mut memory, 0), 0);
        assert_eq!(memory, 0.);
        assert_eq!(fgpu_clear_align_sum(), 1);
    }
}
