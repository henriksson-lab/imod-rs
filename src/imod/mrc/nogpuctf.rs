//! Translation of the intentional no-GPU fallback `IMOD/mrc/nogpuctf.cpp`.

/// `gpuAvailable`: no GPU is available in this build.
pub fn gpu_available(_gpu_count: i32, memory: &mut f32, _debug: i32) -> i32 {
    *memory = 0.0;
    0
}

/// `gpuInitializeSlice`.
pub fn gpu_initialize_slice(
    _slice: &[f32],
    _nx_file: i32,
    _ny_file: i32,
    _strip_xdim: i32,
    _nx_pad: i32,
    _ny_pad: i32,
    _full: bool,
) -> i32 {
    1
}
/// `gpuExtractAndTransform`.
pub fn gpu_extract_and_transform(
    _strip: i32,
    _begin: i32,
    _end: i32,
    _nx_taper: i32,
    _ny_taper: i32,
) -> i32 {
    1
}
/// `gpuCorrectCTF`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_correct_ctf(
    _strip: i32,
    _frequency_x: f32,
    _frequency_y: f32,
    _defocus: f32,
    _cos_astig: f32,
    _sin_astig: f32,
    _focus_sum: f32,
    _focus_difference: f32,
    _cuton: f32,
    _phase_fraction: f32,
    _phase_shift: f32,
    _amplitude_angle: f32,
    _c1: f32,
    _c2: f32,
    _power: f32,
    _power_half: bool,
    _general_power: bool,
    _first_zero: f32,
    _atten_start: f32,
    _minimum_attenuation: f32,
) -> i32 {
    1
}
/// `gpuInterpolateColumns`.
pub fn gpu_interpolate_columns(
    _strip: i32,
    _y_offset: i32,
    _stride: i32,
    _middle: i32,
    _half: i32,
    _offset: i32,
    _last: i32,
) -> i32 {
    1
}
/// `gpuCopyColumns`.
pub fn gpu_copy_columns(
    _strip: i32,
    _x_offset: i32,
    _y_offset: i32,
    _start: i32,
    _end: i32,
) -> i32 {
    1
}
/// `gpuInterpDiagonals`.
pub fn gpu_interp_diagonals(
    _strip: i32,
    _x_offset: i32,
    _y_offset: i32,
    _stride: i32,
    _sin_axis: f32,
    _cos_axis: f32,
    _last_distance: f32,
    _current_distance: f32,
) -> i32 {
    1
}
/// `gpuCopyDiagonals`.
pub fn gpu_copy_diagonals(
    _strip: i32,
    _x_offset: i32,
    _y_offset: i32,
    _sin_axis: f32,
    _cos_axis: f32,
    _lower: f32,
    _upper: f32,
) -> i32 {
    1
}
/// `gpuReturnImage`.
pub fn gpu_return_image(_image: &mut [f32]) -> i32 {
    1
}
/// `gpuGetTimes`.
pub fn gpu_get_times(
    copy: &mut f64,
    prep: &mut f64,
    fft: &mut f64,
    correct: &mut f64,
    interpolate: &mut f64,
) {
    let _ = (copy, prep, fft, correct, interpolate);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fallback_reports_unavailable_and_rejects_gpu_operations() {
        let mut memory = 10.;
        assert_eq!(gpu_available(1, &mut memory, 0), 0);
        assert_eq!(memory, 0.);
        assert_eq!(gpu_return_image(&mut []), 1);
    }
}
