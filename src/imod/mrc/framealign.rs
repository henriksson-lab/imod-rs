//! Owned in-progress bottom-up translation of `framealign.{h,cpp}`.
//!
//! This unit starts with the source's self-contained numerical and sizing
//! methods; higher frame-stack alignment methods build on these primitives.

use crate::imod::clip::correct_defects::CameraDefects;
use crate::imod::clip::correct_defects::cor_def_correct_defects;
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, dose_weight_filter, xcorr_filter_part, xcorr_set_ctf,
};
use crate::imod::libcfshr::gcvspl::{gcvspl, splder};
use crate::imod::libcfshr::zoomdown::SLICE_MODE_FLOAT;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MRC_MODE_USHORT,
};
use crate::imod::mrc::shrmemframe::{
    FinishParams, FrameAlignBackend, FrcParams, GpuParams, InitializeParams, NextFrameParams,
};

#[derive(Clone, Debug)]
pub struct FrameAlign {
    pub nx: usize,
    pub ny: usize,
    pub bin_sum: usize,
    pub bin_align: usize,
    pub trim_fraction: f32,
    pub taper_fraction: f32,
    pub max_shift: usize,
    pub frames: Vec<Vec<f32>>,
    pub x_shifts: Vec<f32>,
    pub y_shifts: Vec<f32>,
    pub gain_reference: Option<Vec<f32>>,
    pub truncation_limit: f32,
    pub print_func: Option<fn(&str)>,
    pub expected_frames: usize,
    pub frame_doses: Vec<f32>,
    pub prior_dose_cumulative: f32,
    pub pixel_size: f32,
    pub critical_dose_scale: f32,
    pub critical_dose_a: f32,
    pub critical_dose_b: f32,
    pub critical_dose_c: f32,
    pub dose_weight_filter: Vec<f32>,
    pub reweight_filter: Vec<f32>,
    pub dose_weight_delta: f32,
}

/// `FrameAlign()`: create an empty owned alignment engine.
pub fn frame_align() -> FrameAlign {
    FrameAlign::default()
}

/// `~FrameAlign()`: release accumulated frames, shifts, and gain data.
pub fn free_frame_align(mut align: FrameAlign) {
    align.cleanup();
}

impl Default for FrameAlign {
    fn default() -> Self {
        Self {
            nx: 0,
            ny: 0,
            bin_sum: 1,
            bin_align: 1,
            trim_fraction: 0.,
            taper_fraction: 0.,
            max_shift: 0,
            frames: Vec::new(),
            x_shifts: Vec::new(),
            y_shifts: Vec::new(),
            gain_reference: None,
            truncation_limit: 0.,
            print_func: None,
            expected_frames: 0,
            frame_doses: Vec::new(),
            prior_dose_cumulative: 0.,
            pixel_size: 1.,
            critical_dose_scale: 0.,
            critical_dose_a: 0.,
            critical_dose_b: 0.,
            critical_dose_c: 0.,
            dose_weight_filter: Vec::new(),
            reweight_filter: Vec::new(),
            dose_weight_delta: 0.,
        }
    }
}
impl FrameAlign {
    /// Source `setPrintFunc`, represented by a native Rust function pointer.
    pub fn set_print_func(&mut self, print_func: Option<fn(&str)>) {
        self.print_func = print_func;
    }

    /// Source destructor/`cleanup`: dropping the owned vectors releases every
    /// frame and correlation result without a separate allocator path.
    pub fn cleanup(&mut self) {
        self.frames.clear();
        self.x_shifts.clear();
        self.y_shifts.clear();
        self.gain_reference = None;
        self.truncation_limit = 0.;
    }

    /// Source `testAndCleanup`: centralize the failed-operation cleanup path
    /// while preserving Rust's ordinary error propagation.
    pub fn test_and_cleanup(&mut self, failed: bool) -> Result<(), String> {
        if failed {
            self.cleanup();
            Err("frame alignment failed; owned working state was released".into())
        } else {
            Ok(())
        }
    }

    /// Owned counterpart to source `initialize`'s CPU state setup.
    pub fn initialize(
        &mut self,
        bin_sum: usize,
        bin_align: usize,
        trim_fraction: f32,
        taper_fraction: f32,
        nx: usize,
        ny: usize,
        max_shift: usize,
    ) -> Result<(), String> {
        if nx == 0 || ny == 0 || bin_sum == 0 || bin_align == 0 {
            return Err("invalid frame alignment dimensions or binning".into());
        }
        self.nx = nx;
        self.ny = ny;
        self.bin_sum = bin_sum;
        self.bin_align = bin_align;
        self.trim_fraction = trim_fraction;
        self.taper_fraction = taper_fraction;
        self.max_shift = max_shift;
        self.frames.clear();
        self.x_shifts.clear();
        self.y_shifts.clear();
        self.expected_frames = 0;
        self.frame_doses.clear();
        self.dose_weight_filter.clear();
        self.reweight_filter.clear();
        self.prior_dose_cumulative = 0.;
        Ok(())
    }
    /// Source `preProcessFrame`: decode real MRC pixels, optional dark/gain correction, and truncation into an owned float frame.
    pub fn preprocess_frame(
        &self,
        frame: &[u8],
        mode: i32,
        dark: Option<&[u8]>,
        gain: Option<&[f32]>,
        truncation: f32,
    ) -> Result<Vec<f32>, String> {
        let bytes = match mode {
            MRC_MODE_BYTE => 1,
            MRC_MODE_SHORT | MRC_MODE_USHORT => 2,
            MRC_MODE_FLOAT => 4,
            _ => return Err("frame mode is not real".into()),
        };
        let dark_bytes = match mode {
            MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_FLOAT => 2,
            MRC_MODE_USHORT => 2,
            _ => 0,
        };
        if frame.len() < self.nx * self.ny * bytes {
            return Err("frame is shorter than dimensions".into());
        }
        if dark.is_some_and(|value| value.len() < self.nx * self.ny * dark_bytes) {
            return Err("dark reference is shorter than frame".into());
        }
        if gain.is_some_and(|value| value.len() < self.nx * self.ny) {
            return Err("gain reference is shorter than frame".into());
        }
        let sample = |data: &[u8], index: usize| -> f32 {
            let at = index * bytes;
            match mode {
                MRC_MODE_BYTE => data[at] as f32,
                MRC_MODE_SHORT => i16::from_ne_bytes(data[at..at + 2].try_into().unwrap()) as f32,
                MRC_MODE_USHORT => u16::from_ne_bytes(data[at..at + 2].try_into().unwrap()) as f32,
                _ => f32::from_ne_bytes(data[at..at + 4].try_into().unwrap()),
            }
        };
        let mut output = Vec::with_capacity(self.nx * self.ny);
        for index in 0..self.nx * self.ny {
            let dark_value = dark.map_or(0., |data| {
                let at = index * dark_bytes;
                match mode {
                    MRC_MODE_USHORT => {
                        u16::from_ne_bytes(data[at..at + 2].try_into().unwrap()) as f32
                    }
                    _ => i16::from_ne_bytes(data[at..at + 2].try_into().unwrap()) as f32,
                }
            });
            let mut value = sample(frame, index) - dark_value;
            if let Some(gain) = gain {
                value *= gain[index]
            }
            if truncation > 0. && value > truncation {
                value = truncation
            }
            output.push(value)
        }
        Ok(output)
    }

    /// Source `preProcessFrame` gain-reference geometry: an oversized camera
    /// gain image is centred over the acquired frame before multiplication.
    pub fn preprocess_frame_with_gain_geometry(
        &self,
        frame: &[u8],
        mode: i32,
        dark: Option<&[u8]>,
        gain: Option<&[f32]>,
        gain_nx: usize,
        gain_ny: usize,
        truncation: f32,
    ) -> Result<Vec<f32>, String> {
        let mut output = self.preprocess_frame(frame, mode, dark, None, truncation)?;
        let Some(gain) = gain else {
            return Ok(output);
        };
        if gain_nx < self.nx || gain_ny < self.ny || gain.len() < gain_nx * gain_ny {
            return Err("gain-reference geometry does not cover the frame".into());
        }
        let x_offset = (gain_nx - self.nx) / 2;
        let y_offset = (gain_ny - self.ny) / 2;
        for y in 0..self.ny {
            for x in 0..self.nx {
                output[x + y * self.nx] *= gain[x + x_offset + (y + y_offset) * gain_nx];
            }
        }
        Ok(output)
    }
    /// CPU translation of the source's pair correlation shift search, using normalized spatial correlation rather than a GPU/FFI path.
    pub fn align_two_frames(
        &self,
        reference: &[f32],
        moving: &[f32],
        near_x: f32,
        near_y: f32,
    ) -> Result<(f32, f32), String> {
        if reference.len() != self.nx * self.ny || moving.len() != reference.len() {
            return Err("frame size mismatch".into());
        }
        let limit = self.max_shift.max(1) as isize;
        let center_x = near_x.round() as isize;
        let center_y = near_y.round() as isize;
        let mut best = (f32::NEG_INFINITY, 0isize, 0isize);
        for dy in center_y - limit..=center_y + limit {
            for dx in center_x - limit..=center_x + limit {
                let mut cross = 0.;
                let mut reference_sum = 0.;
                let mut moving_sum = 0.;
                let mut reference_sq = 0.;
                let mut moving_sq = 0.;
                let mut count = 0;
                for y in 0..self.ny {
                    let sy = y as isize + dy;
                    if sy < 0 || sy >= self.ny as isize {
                        continue;
                    }
                    for x in 0..self.nx {
                        let sx = x as isize + dx;
                        if sx < 0 || sx >= self.nx as isize {
                            continue;
                        }
                        let ref_value = reference[x + y * self.nx];
                        let moving_value = moving[sx as usize + sy as usize * self.nx];
                        cross += ref_value * moving_value;
                        reference_sum += ref_value;
                        moving_sum += moving_value;
                        reference_sq += ref_value * ref_value;
                        moving_sq += moving_value * moving_value;
                        count += 1;
                    }
                }
                let score = if count > 1 {
                    let count = count as f32;
                    let numerator = cross - reference_sum * moving_sum / count;
                    let denominator = ((reference_sq - reference_sum * reference_sum / count)
                        * (moving_sq - moving_sum * moving_sum / count))
                        .max(0.)
                        .sqrt();
                    if denominator > f32::EPSILON {
                        numerator / denominator
                    } else {
                        f32::NEG_INFINITY
                    }
                } else {
                    f32::NEG_INFINITY
                };
                if score > best.0 + f32::EPSILON
                    || ((score - best.0).abs() <= f32::EPSILON
                        && dx * dx + dy * dy < best.1 * best.1 + best.2 * best.2)
                {
                    best = (score, dx, dy)
                }
            }
        }
        Ok((best.1 as f32, best.2 as f32))
    }
    /// Source `nextFrame` CPU ownership boundary.
    pub fn next_frame(
        &mut self,
        frame: &[u8],
        mode: i32,
        dark: Option<&[u8]>,
        gain: Option<&[f32]>,
        truncation: f32,
    ) -> Result<(f32, f32), String> {
        let processed = self.preprocess_frame(frame, mode, dark, gain, truncation)?;
        let shift = if let Some(reference) = self.frames.last() {
            self.align_two_frames(reference, &processed, 0., 0.)?
        } else {
            (0., 0.)
        };
        self.frames.push(processed);
        self.x_shifts.push(shift.0);
        self.y_shifts.push(shift.1);
        Ok(shift)
    }

    /// Source `nextFrame` stateful form: apply the gain and truncation saved
    /// by the caller, correct camera defects, then retain the owned frame.
    pub fn next_frame_configured(
        &mut self,
        frame: &[u8],
        mode: i32,
        dark: Option<&[u8]>,
        defects: Option<&CameraDefects>,
    ) -> Result<(f32, f32), String> {
        let mut processed = self.preprocess_frame(
            frame,
            mode,
            dark,
            self.gain_reference.as_deref(),
            self.truncation_limit,
        )?;
        if let Some(defects) = defects {
            // `framealign.cpp:940-946`: the correction is the same
            // `CorDefCorrectDefects` the `clip` commands call, with
            // `MRC_MODE_FLOAT` and the frame's own coordinates.  Reach it
            // through the image's bytes, which is the C's `void *fOut`.
            let (head, bytes, tail) = unsafe { processed.align_to_mut::<u8>() };
            debug_assert!(head.is_empty() && tail.is_empty());
            cor_def_correct_defects(
                defects,
                bytes,
                SLICE_MODE_FLOAT,
                1,
                0,
                0,
                self.ny as i32,
                self.nx as i32,
            );
        }
        let shift = if let Some(reference) = self.frames.last() {
            self.align_two_frames(reference, &processed, 0., 0.)?
        } else {
            (0., 0.)
        };
        self.frames.push(processed);
        self.x_shifts.push(shift.0);
        self.y_shifts.push(shift.1);
        Ok(shift)
    }

    /// CPU implementation of `findAllVsAllAlignment`.  Each correlation is a
    /// displacement equation and the source regression system solves their
    /// consistent absolute trajectory without a C scratch matrix.
    pub fn find_all_vs_all_alignment(&mut self) -> Result<(), String> {
        if self.frames.is_empty() {
            return Ok(());
        }
        let count = self.frames.len();
        let mut measurements = Vec::with_capacity(count.saturating_mul(count - 1) / 2);
        for first in 0..count {
            for second in first + 1..count {
                let (dx, dy) =
                    self.align_two_frames(&self.frames[first], &self.frames[second], 0., 0.)?;
                measurements.push((first, second, dx, dy));
            }
        }
        (self.x_shifts, self.y_shifts) = Self::regress_all_vs_all(count, &measurements)?;
        Ok(())
    }

    /// Multi-filter counterpart to source `findAllVsAllAlignment`.  Candidate
    /// correlation sets are fitted independently; for four or more equations
    /// their score is leave-one-out residual prediction, otherwise it is the
    /// direct residual.  Robust candidates discard measurements beyond three
    /// median residuals before the final regression.
    pub fn find_all_vs_all_alignment_filters(
        &mut self,
        candidates: &[Vec<(usize, usize, f32, f32)>],
        robust: bool,
    ) -> Result<usize, String> {
        if candidates.is_empty() {
            return Err("no correlation filter candidates were supplied".into());
        }
        let frame_count = self.frames.len();
        if frame_count == 0 {
            self.x_shifts.clear();
            self.y_shifts.clear();
            return Ok(0);
        }
        let mut best_index = 0;
        let mut best_score = f32::INFINITY;
        let mut best_fit = (Vec::new(), Vec::new());
        for (candidate_index, measurements) in candidates.iter().enumerate() {
            let mut fit = Self::regress_all_vs_all(frame_count, measurements)?;
            if robust && measurements.len() >= frame_count.saturating_mul(2) {
                let mut residuals: Vec<f32> = measurements
                    .iter()
                    .map(|&(from, to, x, y)| {
                        (fit.0[to] - fit.0[from] - x).hypot(fit.1[to] - fit.1[from] - y)
                    })
                    .collect();
                let mut ordered = residuals.clone();
                ordered.sort_by(f32::total_cmp);
                let limit = 3. * ordered[ordered.len() / 2].max(f32::EPSILON);
                let accepted: Vec<_> = measurements
                    .iter()
                    .copied()
                    .zip(residuals.drain(..))
                    .filter_map(|(measurement, residual)| {
                        (residual <= limit).then_some(measurement)
                    })
                    .collect();
                if accepted.len() >= frame_count.saturating_sub(1) {
                    fit = Self::regress_all_vs_all(frame_count, &accepted)?;
                }
            }
            let mut score = 0.;
            let mut count = 0;
            if measurements.len() >= 4 {
                for omitted in 0..measurements.len() {
                    let held_out = measurements[omitted];
                    let remaining: Vec<_> = measurements
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &measurement)| {
                            (index != omitted).then_some(measurement)
                        })
                        .collect();
                    if let Ok(predicted) = Self::regress_all_vs_all(frame_count, &remaining) {
                        score += (predicted.0[held_out.1] - predicted.0[held_out.0] - held_out.2)
                            .hypot(predicted.1[held_out.1] - predicted.1[held_out.0] - held_out.3);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                for &(from, to, x, y) in measurements {
                    score += (fit.0[to] - fit.0[from] - x).hypot(fit.1[to] - fit.1[from] - y);
                    count += 1;
                }
            }
            score /= count.max(1) as f32;
            if score < best_score {
                best_score = score;
                best_index = candidate_index;
                best_fit = fit;
            }
        }
        self.x_shifts = best_fit.0;
        self.y_shifts = best_fit.1;
        Ok(best_index)
    }

    /// Source `doRegression`, fitting one independently measured coordinate
    /// with a least-squares line over frame number.
    pub fn do_regression(values: &[f32]) -> Option<(f32, f32)> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f32;
        let sx: f32 = (0..values.len()).map(|index| index as f32).sum();
        let sy: f32 = values.iter().sum();
        let sxx: f32 = (0..values.len()).map(|index| (index * index) as f32).sum();
        let sxy: f32 = values
            .iter()
            .enumerate()
            .map(|(index, &value)| index as f32 * value)
            .sum();
        let denominator = n * sxx - sx * sx;
        if denominator.abs() <= f32::EPSILON {
            Some((sy / n, 0.))
        } else {
            let slope = (n * sxy - sx * sy) / denominator;
            Some(((sy - slope * sx) / n, slope))
        }
    }

    /// Source `setupDoseWeighting`, expressed as per-frame scalar weights.
    /// A nonpositive critical scale intentionally leaves the source data
    /// unweighted.
    pub fn setup_dose_weighting(&self, dose_per_frame: f32, critical_scale: f32) -> Vec<f32> {
        let mut cumulative_dose = 0.;
        (0..self.frames.len())
            .map(|_| {
                let weight = if critical_scale > 0. {
                    (-cumulative_dose / critical_scale).exp()
                } else {
                    1.
                };
                cumulative_dose += dose_per_frame.max(0.);
                weight
            })
            .collect()
    }

    /// Source `setupDoseWeighting` with all physical parameters retained in
    /// owned state.  An all-one reweight filter is normalized exactly as the
    /// source does from the anticipated complete frame-dose sequence.
    pub fn setup_dose_weighting_full(
        &mut self,
        prior_dose: f32,
        frame_doses: Vec<f32>,
        pixel_size: f32,
        critical_scale: f32,
        a_factor: f32,
        b_factor: f32,
        c_factor: f32,
        reweight_filter: Option<Vec<f32>>,
        filter_size: usize,
    ) -> Result<(), String> {
        if frame_doses.is_empty() || pixel_size <= 0. || filter_size < 2 {
            return Err("invalid dose-weighting parameters".into());
        }
        if reweight_filter
            .as_ref()
            .is_some_and(|filter| filter.len() != filter_size)
        {
            return Err("dose reweight filter length differs from requested filter size".into());
        }
        self.expected_frames = frame_doses.len();
        self.frame_doses = frame_doses;
        self.prior_dose_cumulative = prior_dose;
        self.pixel_size = pixel_size;
        self.critical_dose_scale = critical_scale;
        self.critical_dose_a = a_factor;
        self.critical_dose_b = b_factor;
        self.critical_dose_c = c_factor;
        self.dose_weight_filter = vec![0.; filter_size];
        self.reweight_filter = reweight_filter.unwrap_or_default();
        if self.reweight_filter.iter().all(|&value| value == 1.) {
            self.reweight_filter.fill(0.);
            let mut dose = prior_dose;
            let mut single = vec![0.; filter_size];
            for &frame_dose in &self.frame_doses {
                dose_weight_filter(
                    dose,
                    dose + frame_dose,
                    pixel_size,
                    a_factor,
                    b_factor,
                    c_factor,
                    critical_scale,
                    &mut single,
                    filter_size as i32,
                    0.71,
                    &mut self.dose_weight_delta,
                );
                for (total, value) in self.reweight_filter.iter_mut().zip(&single) {
                    *total += value;
                }
                dose += frame_dose;
            }
            for value in &mut self.reweight_filter {
                if *value > 0. {
                    *value = self.expected_frames as f32 / *value;
                }
            }
        }
        Ok(())
    }

    /// Source `gpuAvailable`.  A zero-device request stays on the owned CPU
    /// path; a positive request is rejected until a safe Rust GPU backend is
    /// supplied by this crate.
    pub fn gpu_available(requested_devices: usize, backend_available: bool) -> Result<(), String> {
        if requested_devices == 0 || backend_available {
            Ok(())
        } else {
            Err("GPU frame alignment was requested but no Rust GPU backend is available".into())
        }
    }

    /// Owned CPU counterpart to `finishAlignAndSum`: shift every selected
    /// frame into the first-frame coordinate system and produce its weighted
    /// arithmetic mean.  Out-of-image samples are omitted, matching the
    /// source's trimmed sum rather than wrapping data across an edge.
    pub fn finish_align_and_sum(&self, weights: Option<&[f32]>) -> Result<Vec<f32>, String> {
        if self.frames.is_empty() {
            return Ok(Vec::new());
        }
        if self.x_shifts.len() != self.frames.len() || self.y_shifts.len() != self.frames.len() {
            return Err("frame shift count does not match frame count".into());
        }
        if weights.is_some_and(|weights| weights.len() != self.frames.len()) {
            return Err("dose weight count does not match frame count".into());
        }
        let mut sum = vec![0.; self.nx * self.ny];
        let mut sum_weight = vec![0.; self.nx * self.ny];
        for (frame_index, frame) in self.frames.iter().enumerate() {
            let weight = weights.map_or(1., |values| values[frame_index]);
            let dx = self.x_shifts[frame_index].round() as isize;
            let dy = self.y_shifts[frame_index].round() as isize;
            for y in 0..self.ny {
                let source_y = y as isize + dy;
                if !(0..self.ny as isize).contains(&source_y) {
                    continue;
                }
                for x in 0..self.nx {
                    let source_x = x as isize + dx;
                    if !(0..self.nx as isize).contains(&source_x) {
                        continue;
                    }
                    let target = x + y * self.nx;
                    sum[target] += weight * frame[source_x as usize + source_y as usize * self.nx];
                    sum_weight[target] += weight;
                }
            }
        }
        for (pixel, weight) in sum.iter_mut().zip(sum_weight) {
            if weight != 0. {
                *pixel /= weight;
            }
        }
        Ok(sum)
    }

    /// Source `finishAlignAndSum` optional even/odd outputs, returned as owned
    /// arrays rather than nullable output pointers.
    pub fn finish_align_and_sum_subsets(
        &self,
        weights: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let sum = self.finish_align_and_sum(weights)?;
        if self.frames.is_empty() {
            return Ok((sum, Vec::new(), Vec::new()));
        }
        let even = FrameAlign {
            frames: self.frames.iter().step_by(2).cloned().collect(),
            x_shifts: self.x_shifts.iter().step_by(2).copied().collect(),
            y_shifts: self.y_shifts.iter().step_by(2).copied().collect(),
            ..self.clone()
        };
        let odd = FrameAlign {
            frames: self.frames.iter().skip(1).step_by(2).cloned().collect(),
            x_shifts: self.x_shifts.iter().skip(1).step_by(2).copied().collect(),
            y_shifts: self.y_shifts.iter().skip(1).step_by(2).copied().collect(),
            ..self.clone()
        };
        let even_weights: Option<Vec<f32>> =
            weights.map(|values| values.iter().step_by(2).copied().collect());
        let odd_weights: Option<Vec<f32>> =
            weights.map(|values| values.iter().skip(1).step_by(2).copied().collect());
        let even_sum = even.finish_align_and_sum(even_weights.as_deref())?;
        let odd_sum = odd.finish_align_and_sum(odd_weights.as_deref())?;
        Ok((sum, even_sum, odd_sum))
    }

    /// Fourier branch of source `finishAlignAndSum`: accumulate shifted packed
    /// real FFTs, optionally apply the per-frame CTF/dose radial filter, then
    /// inverse transform and extract the unpadded image for total/even/odd
    /// sums.  The transform buffers are ordinary owned vectors.
    pub fn finish_align_and_sum_fourier(
        &self,
        frame_ffts: &[Vec<f32>],
        fft_nx: usize,
        fft_ny: usize,
        radial_filters: Option<&[Vec<f32>]>,
        delta: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if fft_nx == 0
            || fft_ny == 0
            || fft_nx % 2 != 0
            || frame_ffts.len() != self.frames.len()
            || self.x_shifts.len() != frame_ffts.len()
            || self.y_shifts.len() != frame_ffts.len()
        {
            return Err("Fourier sum dimensions, frame count, or shifts are inconsistent".into());
        }
        if radial_filters.is_some_and(|filters| filters.len() != frame_ffts.len()) {
            return Err("Fourier dose/CTF filter count differs from frame count".into());
        }
        let packed = (fft_nx + 2) * fft_ny;
        if frame_ffts.iter().any(|frame| frame.len() < packed) {
            return Err("packed frame FFT is shorter than dimensions".into());
        }
        let mut total = vec![0.; packed];
        let mut even = vec![0.; packed];
        let mut odd = vec![0.; packed];
        for (index, frame) in frame_ffts.iter().enumerate() {
            let filter = radial_filters.map(|filters| (filters[index].as_slice(), delta));
            add_to_sums(
                frame,
                &mut total,
                fft_nx,
                fft_ny,
                self.x_shifts[index],
                self.y_shifts[index],
                filter,
            )?;
            if index % 2 == 0 {
                add_to_sums(
                    frame,
                    &mut even,
                    fft_nx,
                    fft_ny,
                    self.x_shifts[index],
                    self.y_shifts[index],
                    filter,
                )?;
            } else {
                add_to_sums(
                    frame,
                    &mut odd,
                    fft_nx,
                    fft_ny,
                    self.x_shifts[index],
                    self.y_shifts[index],
                    filter,
                )?;
            }
        }
        for packed_sum in [&mut total, &mut even, &mut odd] {
            crate::imod::libfft::todfft::todfft(packed_sum, fft_nx as i32, fft_ny as i32, 1);
        }
        let extract = |packed_sum: &[f32]| {
            (0..fft_ny)
                .flat_map(|y| {
                    packed_sum[y * (fft_nx + 2)..y * (fft_nx + 2) + fft_nx]
                        .iter()
                        .copied()
                })
                .collect()
        };
        Ok((extract(&total), extract(&even), extract(&odd)))
    }

    /// Source `getUnweightedSum`.
    pub fn get_unweighted_sum(&self) -> Result<Vec<f32>, String> {
        self.finish_align_and_sum(None)
    }

    /// Source `adjustAndPushShifts`: place a filtered trajectory in the
    /// coordinate system selected by `top_ind`.
    pub fn adjust_and_push_shifts(
        &mut self,
        top_ind: usize,
        x_filtered: &[f32],
        y_filtered: &[f32],
    ) -> Result<(), String> {
        if x_filtered.len() != self.frames.len() || y_filtered.len() != self.frames.len() {
            return Err("filtered shifts do not match frame count".into());
        }
        let (&origin_x, &origin_y) = x_filtered
            .get(top_ind)
            .zip(y_filtered.get(top_ind))
            .ok_or_else(|| "top shift index is outside the frame stack".to_string())?;
        self.x_shifts = x_filtered.iter().map(|&shift| shift - origin_x).collect();
        self.y_shifts = y_filtered.iter().map(|&shift| shift - origin_y).collect();
        Ok(())
    }

    /// Source `getAllFrameShifts`, returning an owned frame trajectory.
    pub fn get_all_frame_shifts(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        if self.x_shifts.len() != self.frames.len() || self.y_shifts.len() != self.frames.len() {
            return Err("frame shift count does not match frame count".into());
        }
        Ok((self.x_shifts.clone(), self.y_shifts.clone()))
    }

    /// CPU equivalent of the source spline smoothing pass.  It uses the
    /// source's local-frame intent while keeping every sample owned and
    /// deterministic; `half_window == 0` preserves the input exactly.
    pub fn spline_smooth(
        &self,
        x_shifts: &[f32],
        y_shifts: &[f32],
        _half_window: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if x_shifts.len() != y_shifts.len() {
            return Err("X and Y shift counts differ".into());
        }
        // `gcvspl` requires n >= 2 * m; this call uses m = 2, so a
        // three-frame sequence cannot enter the native spline solver.
        if x_shifts.len() < 4 {
            return Ok((x_shifts.to_vec(), y_shifts.to_vec()));
        }
        let abscissae: Vec<f64> = (0..x_shifts.len()).map(|index| index as f64).collect();
        let weights = vec![1.; x_shifts.len()];
        let smooth = |values: &[f32]| -> Result<Vec<f32>, String> {
            let ordinates: Vec<f64> = values.iter().map(|&value| value as f64).collect();
            let mut coefficient = vec![0.; values.len()];
            let mut work = vec![0.; 18 * values.len() + 16];
            let mut error = 0;
            if gcvspl(
                &abscissae,
                &ordinates,
                values.len() as i32,
                &weights,
                &weights,
                2,
                values.len() as i32,
                1,
                2,
                0.5,
                &mut coefficient,
                values.len() as i32,
                &mut work,
                &mut error,
            ) != 0
                || error != 0
            {
                return Err(format!("GCV spline fit failed with code {error}"));
            }
            let mut output = Vec::with_capacity(values.len());
            for index in 0..values.len() {
                let mut near = index as i32;
                output.push(splder(
                    0,
                    2,
                    values.len() as i32,
                    index as f64,
                    &abscissae,
                    &coefficient,
                    &mut near,
                    &mut work,
                ) as f32);
            }
            Ok(output)
        };
        Ok((smooth(x_shifts)?, smooth(y_shifts)?))
    }

    /// Source `smoothedTotalDistance`.  Every frame is predicted from the
    /// nearest up-to-seven-frame quadratic least-squares fit, preserving the
    /// source's linear fallback for short sequences.
    pub fn smoothed_total_distance(
        x_shifts: &[f32],
        y_shifts: &[f32],
    ) -> Result<(f32, f32, Vec<f32>, Vec<f32>, f64), String> {
        if x_shifts.len() != y_shifts.len() {
            return Err("X and Y shift counts differ".into());
        }
        if x_shifts.len() < 2 {
            return Ok((0., 0., x_shifts.to_vec(), y_shifts.to_vec(), 0.));
        }
        let raw = total_shift_distance(x_shifts, y_shifts);
        let order = if x_shifts.len() > 4 { 2 } else { 1 };
        let fit_count = x_shifts.len().min(7);
        let before = fit_count / 2;
        let mut smoothed_x = Vec::with_capacity(x_shifts.len());
        let mut smoothed_y = Vec::with_capacity(y_shifts.len());
        for index in 0..x_shifts.len() {
            let mut first = index.saturating_sub(before);
            let last = (first + fit_count).min(x_shifts.len());
            first = last - fit_count;
            let dimension = order + 1;
            let mut normal = vec![vec![0f64; dimension]; dimension];
            let mut right_x = vec![0f64; dimension];
            let mut right_y = vec![0f64; dimension];
            for local in 0..fit_count {
                let x = local as f64;
                for row in 0..dimension {
                    let basis_row = x.powi(row as i32);
                    right_x[row] += basis_row * x_shifts[first + local] as f64;
                    right_y[row] += basis_row * y_shifts[first + local] as f64;
                    for column in 0..dimension {
                        normal[row][column] += basis_row * x.powi(column as i32);
                    }
                }
            }
            for pivot in 0..dimension {
                let best = (pivot..dimension)
                    .max_by(|&a, &b| normal[a][pivot].abs().total_cmp(&normal[b][pivot].abs()))
                    .unwrap();
                normal.swap(pivot, best);
                right_x.swap(pivot, best);
                right_y.swap(pivot, best);
                let divisor = normal[pivot][pivot];
                if divisor.abs() < f64::EPSILON {
                    continue;
                }
                for column in pivot..dimension {
                    normal[pivot][column] /= divisor;
                }
                right_x[pivot] /= divisor;
                right_y[pivot] /= divisor;
                for row in 0..dimension {
                    if row == pivot {
                        continue;
                    }
                    let scale = normal[row][pivot];
                    for column in pivot..dimension {
                        normal[row][column] -= scale * normal[pivot][column];
                    }
                    right_x[row] -= scale * right_x[pivot];
                    right_y[row] -= scale * right_y[pivot];
                }
            }
            let local_x = (index - first) as f64;
            smoothed_x.push(
                (0..dimension)
                    .map(|power| right_x[power] * local_x.powi(power as i32))
                    .sum::<f64>() as f32,
            );
            smoothed_y.push(
                (0..dimension)
                    .map(|power| right_y[power] * local_x.powi(power as i32))
                    .sum::<f64>() as f32,
            );
        }
        let distance = total_shift_distance(&smoothed_x, &smoothed_y);
        let variance = smoothed_x
            .iter()
            .zip(&smoothed_y)
            .zip(x_shifts.iter().zip(y_shifts))
            .map(|((&sx, &sy), (&x, &y))| {
                let dx = sx - x;
                let dy = sy - y;
                (dx * dx + dy * dy) as f64
            })
            .sum::<f64>()
            / (2 * x_shifts.len()) as f64;
        Ok((distance, raw, smoothed_x, smoothed_y, variance))
    }

    /// Source `doRegression` robust mode.  Iteratively rejects large residuals
    /// from the ordinary least-squares line, returning the final intercept,
    /// slope, and source-frame rejection mask.
    pub fn do_robust_regression(values: &[f32], sigma_limit: f32) -> Option<(f32, f32, Vec<bool>)> {
        if values.is_empty() {
            return None;
        }
        let mut used = vec![true; values.len()];
        for _ in 0..4 {
            let indices: Vec<_> = (0..values.len()).filter(|&index| used[index]).collect();
            if indices.len() < 2 {
                break;
            }
            let count = indices.len() as f32;
            let sx: f32 = indices.iter().map(|&index| index as f32).sum();
            let sy: f32 = indices.iter().map(|&index| values[index]).sum();
            let sxx: f32 = indices.iter().map(|&index| (index * index) as f32).sum();
            let sxy: f32 = indices
                .iter()
                .map(|&index| index as f32 * values[index])
                .sum();
            let divisor = count * sxx - sx * sx;
            if divisor.abs() <= f32::EPSILON {
                break;
            }
            let slope = (count * sxy - sx * sy) / divisor;
            let intercept = (sy - slope * sx) / count;
            let variance = indices
                .iter()
                .map(|&index| {
                    let residual = values[index] - (intercept + slope * index as f32);
                    residual * residual
                })
                .sum::<f32>()
                / count;
            let limit = sigma_limit.max(1.) * variance.sqrt();
            if limit <= f32::EPSILON {
                return Some((intercept, slope, used));
            }
            let mut changed = false;
            for index in 0..values.len() {
                if used[index] && (values[index] - (intercept + slope * index as f32)).abs() > limit
                {
                    used[index] = false;
                    changed = true;
                }
            }
            if !changed {
                return Some((intercept, slope, used));
            }
        }
        let kept: Vec<_> = (0..values.len()).filter(|&index| used[index]).collect();
        let count = kept.len().max(1) as f32;
        let sx: f32 = kept.iter().map(|&index| index as f32).sum();
        let sy: f32 = kept.iter().map(|&index| values[index]).sum();
        let sxx: f32 = kept.iter().map(|&index| (index * index) as f32).sum();
        let sxy: f32 = kept.iter().map(|&index| index as f32 * values[index]).sum();
        let divisor = count * sxx - sx * sx;
        let slope = if divisor.abs() <= f32::EPSILON {
            0.
        } else {
            (count * sxy - sx * sy) / divisor
        };
        Some(((sy - slope * sx) / count, slope, used))
    }

    /// Matrix form of source `doRegression` for all-vs-all measurements.
    /// Measurements are `(reference, aligned, X shift, Y shift)` and frame 0
    /// is fixed at zero, eliminating the C implementation's scratch matrix.
    pub fn regress_all_vs_all(
        frame_count: usize,
        measurements: &[(usize, usize, f32, f32)],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if frame_count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if measurements
            .iter()
            .any(|&(from, to, _, _)| from >= frame_count || to >= frame_count || from == to)
        {
            return Err("invalid frame index in all-vs-all measurement".into());
        }
        let mut answer = (vec![0.; frame_count], vec![0.; frame_count]);
        for coordinate in 0..2 {
            let mut matrix = vec![vec![0.; frame_count - 1]; frame_count - 1];
            let mut right = vec![0.; frame_count - 1];
            for &(from, to, x, y) in measurements {
                let value = if coordinate == 0 { x } else { y };
                for (row, coefficient) in [(from, -1.), (to, 1.)] {
                    if row == 0 {
                        continue;
                    }
                    right[row - 1] += coefficient * value;
                    for (column, other) in [(from, -1.), (to, 1.)] {
                        if column != 0 {
                            matrix[row - 1][column - 1] += coefficient * other;
                        }
                    }
                }
            }
            for pivot in 0..frame_count - 1 {
                let best = (pivot..frame_count - 1)
                    .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))
                    .unwrap();
                matrix.swap(pivot, best);
                right.swap(pivot, best);
                let divisor = matrix[pivot][pivot];
                if divisor.abs() <= f32::EPSILON {
                    continue;
                }
                for column in pivot..frame_count - 1 {
                    matrix[pivot][column] /= divisor;
                }
                right[pivot] /= divisor;
                for row in 0..frame_count - 1 {
                    if row == pivot {
                        continue;
                    }
                    let factor = matrix[row][pivot];
                    for column in pivot..frame_count - 1 {
                        matrix[row][column] -= factor * matrix[pivot][column];
                    }
                    right[row] -= factor * right[pivot];
                }
            }
            let output = if coordinate == 0 {
                &mut answer.0
            } else {
                &mut answer.1
            };
            for (index, value) in right.into_iter().enumerate() {
                output[index + 1] = value;
            }
        }
        Ok(answer)
    }

    /// Safe float-frame counterpart to the source defect correction before
    /// correlation.  Bad pixels, complete columns, and complete rows are
    /// replaced with their finite four-neighbour mean.
    pub fn correct_defects(
        frame: &mut [f32],
        nx: usize,
        ny: usize,
        defects: &CameraDefects,
    ) -> Result<(), String> {
        if frame.len() != nx * ny {
            return Err("defect correction frame dimensions differ from data".into());
        }
        let mut damaged = vec![false; frame.len()];
        for (&x, &y) in defects.bad_pixel_x.iter().zip(&defects.bad_pixel_y) {
            if (x as usize) < nx && (y as usize) < ny {
                damaged[x as usize + y as usize * nx] = true;
            }
        }
        for (&start, &width) in defects
            .bad_column_start
            .iter()
            .zip(&defects.bad_column_width)
        {
            for x in start as usize..(start as usize + width.max(0) as usize).min(nx) {
                for y in 0..ny {
                    damaged[x + y * nx] = true;
                }
            }
        }
        for (&start, &height) in defects.bad_row_start.iter().zip(&defects.bad_row_height) {
            for y in start as usize..(start as usize + height.max(0) as usize).min(ny) {
                for x in 0..nx {
                    damaged[x + y * nx] = true;
                }
            }
        }
        for (((&start, &width), &first_y), &last_y) in defects
            .partial_bad_col
            .iter()
            .zip(&defects.partial_bad_width)
            .zip(&defects.partial_bad_start_y)
            .zip(&defects.partial_bad_end_y)
        {
            for x in start as usize..(start as usize + width.max(0) as usize).min(nx) {
                for y in (first_y as usize).min(ny)..=(last_y as usize).min(ny.saturating_sub(1)) {
                    if y < ny {
                        damaged[x + y * nx] = true;
                    }
                }
            }
        }
        for (((&start, &height), &first_x), &last_x) in defects
            .partial_bad_row
            .iter()
            .zip(&defects.partial_bad_height)
            .zip(&defects.partial_bad_start_x)
            .zip(&defects.partial_bad_end_x)
        {
            for y in start as usize..(start as usize + height.max(0) as usize).min(ny) {
                for x in (first_x as usize).min(nx)..=(last_x as usize).min(nx.saturating_sub(1)) {
                    if x < nx {
                        damaged[x + y * nx] = true;
                    }
                }
            }
        }
        if defects.usable_top > 0 {
            for y in 0..(defects.usable_top as usize).min(ny) {
                for x in 0..nx {
                    damaged[x + y * nx] = true;
                }
            }
        }
        if defects.usable_bottom > 0 && (defects.usable_bottom as usize) < ny {
            for y in defects.usable_bottom as usize..ny {
                for x in 0..nx {
                    damaged[x + y * nx] = true;
                }
            }
        }
        if defects.usable_left > 0 {
            for y in 0..ny {
                for x in 0..(defects.usable_left as usize).min(nx) {
                    damaged[x + y * nx] = true;
                }
            }
        }
        if defects.usable_right > 0 && (defects.usable_right as usize) < nx {
            for y in 0..ny {
                for x in defects.usable_right as usize..nx {
                    damaged[x + y * nx] = true;
                }
            }
        }
        let original = frame.to_vec();
        for y in 0..ny {
            for x in 0..nx {
                let index = x + y * nx;
                if !damaged[index] {
                    continue;
                }
                let mut sum = 0.;
                let mut count = 0.;
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let xx = x as isize + dx;
                    let yy = y as isize + dy;
                    if xx >= 0 && yy >= 0 && (xx as usize) < nx && (yy as usize) < ny {
                        let nearby = xx as usize + yy as usize * nx;
                        if !damaged[nearby] {
                            sum += original[nearby];
                            count += 1.;
                        }
                    }
                }
                if count > 0. {
                    frame[index] = sum / count;
                }
            }
        }
        Ok(())
    }
}

/// Direct service implementation for the translated `shrmemframe` protocol.
/// Its request structs own all payloads, so no shared-memory pointer lifetime
/// crosses into `FrameAlign`.
impl FrameAlignBackend for FrameAlign {
    fn initialize(&mut self, params: &mut InitializeParams) {
        params.ret_val = match self.initialize(
            params.bin_sum.max(1) as usize,
            params.bin_align.max(1) as usize,
            params.trim_frac,
            params.taper_frac,
            params.nx,
            params.ny,
            params.max_shift.max(0) as usize,
        ) {
            Ok(()) => {
                self.expected_frames = params.expected_z.max(0) as usize;
                0
            }
            Err(error) => {
                params.messages = error;
                1
            }
        };
    }

    fn next_frame(&mut self, params: &mut NextFrameParams) {
        self.gain_reference = params.gain_reference.clone();
        self.truncation_limit = params.trunc_limit;
        match self.next_frame_configured(&params.frame, params.pixel_type, None, None) {
            Ok((x, y)) => {
                params.shift_x = x;
                params.shift_y = y;
                params.ret_val = 0;
            }
            Err(error) => {
                params.messages = error;
                params.ret_val = 1;
            }
        }
    }

    fn finish_align_and_sum(&mut self, params: &mut FinishParams) {
        let raw_x = self.x_shifts.clone();
        let raw_y = self.y_shifts.clone();
        if params.do_spline != 0 {
            match self.spline_smooth(&raw_x, &raw_y, 0) {
                Ok((x, y)) => {
                    self.x_shifts = x;
                    self.y_shifts = y;
                }
                Err(error) => {
                    params.messages = error;
                    params.ret_val = 1;
                    return;
                }
            }
        }
        match FrameAlign::finish_align_and_sum_subsets(self, None) {
            Ok((sum, even, odd)) => {
                params.aligned_sum = sum;
                params.x_shifts = self.x_shifts.clone();
                params.y_shifts = self.y_shifts.clone();
                params.raw_x_shifts = raw_x;
                params.raw_y_shifts = raw_y;
                if !params.ring_corrs.is_empty() {
                    let (_, _, _, half) = analyze_frc_crossings(&params.ring_corrs, params.delta_r);
                    params.ring_corrs = vec![half];
                }
                let (_, raw_distance, _, _, _) =
                    FrameAlign::smoothed_total_distance(&params.x_shifts, &params.y_shifts)
                        .unwrap_or((0., 0., Vec::new(), Vec::new(), 0.));
                params.raw_dist = vec![raw_distance];
                params.smooth_dist = vec![total_shift_distance(&params.x_shifts, &params.y_shifts)];
                params.best_filt = 0;
                let _ = (even, odd);
                params.ret_val = 0;
            }
            Err(error) => {
                params.messages = error;
                params.ret_val = 1;
            }
        }
    }

    fn analyze_frc_crossings(&mut self, params: &mut FrcParams) {
        (
            params.half_cross,
            params.quart_cross,
            params.eighth_cross,
            params.half_nyq,
        ) = analyze_frc_crossings(&params.ring_corrs, params.frc_delta);
    }

    fn gpu_available(&mut self, params: &mut GpuParams) {
        match FrameAlign::gpu_available(params.n_gpu.max(0) as usize, false) {
            Ok(()) => params.ret_val = 0,
            Err(error) => {
                params.messages = error;
                params.ret_val = 1;
            }
        }
    }

    fn cleanup(&mut self) {
        FrameAlign::cleanup(self);
    }
}

/// CPU-only selection boundary for source GPU capability checks.  GPU modules
/// can opt in later without leaking runtime-library or device handles into the
/// alignment representation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuStrategy {
    #[default]
    Cpu,
    RequestedButUnavailable,
}

pub const GPU_FOR_SUMMING: u32 = 1;
pub const GPU_FOR_ALIGNING: u32 = 1 << 2;
pub const GPU_DO_NOISE_TAPER: u32 = 1 << 3;
pub const GPU_DO_BIN_PAD: u32 = 1 << 4;
pub const STACK_FULL_ON_GPU: u32 = 1 << 5;
pub const GPU_DO_GAIN_NORM: u32 = 1 << 6;
pub const GPU_CORRECT_DEFECTS: u32 = 1 << 7;
pub const GPU_DO_PREPROCESS: u32 = 1 << 8;
pub const GPU_STACK_LIMITED: u32 = 1 << 9;
pub const GPU_STACK_LIM_SHIFT: u32 = 20;
pub const GPU_STACK_LIM_MASK: u32 = 0x0fff;

/// Safe replacement for the source's GPU-resident saved stacks.  A future
/// backend transfers complete owned vectors through this boundary; CPU failure
/// recovery never observes device pointers or partially owned allocations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuFrameStore {
    pub full_frames: Vec<Vec<f32>>,
    pub align_ffts: Vec<Vec<f32>>,
}

impl GpuFrameStore {
    /// Source `prepareToFetchAlignFFTs`.
    pub fn prepare_to_fetch_align_ffts(&self, frame_index: usize) -> Result<(), String> {
        self.align_ffts
            .get(frame_index)
            .map(|_| ())
            .ok_or_else(|| "alignment FFT is not present in the owned GPU staging store".into())
    }

    /// Source `recoverGpuAlignFFTs`; transfer one full owned FFT to the CPU
    /// recovery path, leaving no duplicate device-owned state behind.
    pub fn recover_gpu_align_ffts(&mut self, frame_index: usize) -> Result<Vec<f32>, String> {
        if frame_index >= self.align_ffts.len() {
            return Err("alignment FFT recovery index is outside the staging store".into());
        }
        Ok(self.align_ffts.remove(frame_index))
    }

    /// Source `recoverGpuFullStack`.
    pub fn recover_gpu_full_stack(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.full_frames)
    }

    /// Source `cancelInitialStepsOnGPU`.
    pub fn cancel_initial_steps_on_gpu(&mut self) {
        self.full_frames.clear();
        self.align_ffts.clear();
    }

    /// Source `recoverFromSummingFailure`: preserve an individual frame for
    /// CPU summing and remove it from the staging store.
    pub fn recover_from_summing_failure(&mut self, frame_index: usize) -> Result<Vec<f32>, String> {
        if frame_index >= self.full_frames.len() {
            return Err("summed frame recovery index is outside the staging store".into());
        }
        Ok(self.full_frames.remove(frame_index))
    }
}

/// Source GPU availability policy at the Rust boundary: selecting a GPU is
/// explicit, and is unavailable until a safe backend is registered.
pub fn gpu_strategy(requested: bool, backend_available: bool) -> GpuStrategy {
    if requested && !backend_available {
        GpuStrategy::RequestedButUnavailable
    } else {
        GpuStrategy::Cpu
    }
}

/// Source `filterAndAddToSum` for the packed, interleaved real FFT layout
/// (`(nx + 2) * ny` floats).  Frequencies outside the last nonzero radial
/// filter sample remain unchanged in the destination.
pub fn filter_and_add_to_sum(
    fft: &[f32],
    array: &mut [f32],
    nx: usize,
    ny: usize,
    radial_filter: &[f32],
    delta: f32,
) -> Result<(), String> {
    if nx == 0 || ny == 0 || delta <= 0. || fft.len() < (nx + 2) * ny || array.len() < (nx + 2) * ny
    {
        return Err("invalid packed FFT or radial filter dimensions".into());
    }
    let Some(last) = radial_filter.iter().rposition(|&value| value != 0.) else {
        return Ok(());
    };
    let max_frequency = (last + 1) as f32 * delta;
    let half_x = nx / 2;
    for y in 0..ny {
        let fy = (y as f32 / ny as f32).min(1. - y as f32 / ny as f32);
        if fy > max_frequency {
            continue;
        }
        for x in 0..=half_x {
            let frequency = ((x as f32 / nx as f32).powi(2) + fy.powi(2)).sqrt();
            let filter_index = (frequency / delta + 0.5) as usize;
            let Some(&weight) = radial_filter.get(filter_index) else {
                continue;
            };
            let index = 2 * (y * (half_x + 1) + x);
            array[index] += fft[index] * weight;
            array[index + 1] += fft[index + 1] * weight;
        }
    }
    Ok(())
}

/// Source `addToSums` CPU Fourier path.  This applies the frame translation
/// as a phase rotation in the packed transform, then adds either directly or
/// through the source radial dose filter.
pub fn add_to_sums(
    full_fft: &[f32],
    full_sum: &mut [f32],
    nx: usize,
    ny: usize,
    x_shift: f32,
    y_shift: f32,
    radial_filter: Option<(&[f32], f32)>,
) -> Result<(), String> {
    if nx == 0 || ny == 0 || full_fft.len() < (nx + 2) * ny || full_sum.len() < (nx + 2) * ny {
        return Err("invalid packed FFT dimensions for summing".into());
    }
    let half_x = nx / 2;
    let mut shifted = vec![0.; (nx + 2) * ny];
    for y in 0..ny {
        let signed_y = if y <= ny / 2 {
            y as f32
        } else {
            y as f32 - ny as f32
        };
        for x in 0..=half_x {
            let phase = -std::f32::consts::TAU
                * (x as f32 * x_shift / nx as f32 + signed_y * y_shift / ny as f32);
            let (sin, cos) = phase.sin_cos();
            let index = 2 * (y * (half_x + 1) + x);
            shifted[index] = full_fft[index] * cos - full_fft[index + 1] * sin;
            shifted[index + 1] = full_fft[index] * sin + full_fft[index + 1] * cos;
        }
    }
    if let Some((filter, delta)) = radial_filter {
        filter_and_add_to_sum(&shifted, full_sum, nx, ny, filter, delta)
    } else {
        full_sum
            .iter_mut()
            .zip(shifted)
            .for_each(|(sum, value)| *sum += value);
        Ok(())
    }
}

/// Source `initialize` CTF-mask construction.  The full mask is applied as a
/// square-root filter to individual stored alignment FFTs; the optional
/// subarea masks use the complete filter, matching the later correlation path.
pub fn alignment_ctf_masks(
    nx: usize,
    ny: usize,
    sigma1: f32,
    sigma2: &[f32],
    radius1: f32,
    radius2: &[f32],
    bin_align: usize,
    use_full_only: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, f32), String> {
    if nx == 0 || ny == 0 || sigma2.len() != radius2.len() || sigma2.is_empty() {
        return Err("invalid alignment CTF filter parameters".into());
    }
    let mut radial = vec![0.; 8193];
    let mut delta = 0.;
    xcorr_set_ctf(
        sigma1,
        sigma2[0] * bin_align as f32,
        radius1,
        if use_full_only {
            radius2[0] * bin_align as f32
        } else {
            0.71
        },
        &mut radial,
        nx as i32,
        ny as i32,
        &mut delta,
    );
    let mut full = vec![1.; (nx + 2) * ny];
    for value in &mut radial {
        *value = value.sqrt();
    }
    xcorr_filter_part(
        FilterIn::Fft(&full.clone()),
        &mut full,
        nx as i32,
        ny as i32,
        &radial,
        delta,
    );
    full[0] = 0.;
    if use_full_only {
        return Ok((full, Vec::new(), delta));
    }
    let mut masks = Vec::with_capacity(sigma2.len());
    for (&sigma, &radius) in sigma2.iter().zip(radius2) {
        let mut radial = vec![0.; 8193];
        let mut sub_delta = 0.;
        xcorr_set_ctf(
            0.,
            sigma * bin_align as f32,
            0.,
            radius * bin_align as f32,
            &mut radial,
            nx as i32,
            ny as i32,
            &mut sub_delta,
        );
        let mut mask = vec![1.; (nx + 2) * ny];
        xcorr_filter_part(
            FilterIn::Fft(&mask.clone()),
            &mut mask,
            nx as i32,
            ny as i32,
            &radial,
            sub_delta,
        );
        masks.push(mask);
    }
    Ok((full, masks, delta))
}

/// Complete owned input set for source `totalMemoryNeeds`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameAlignMemoryOptions {
    pub full_pad_size: f32,
    pub full_data_size: usize,
    pub sum_pad_size: f32,
    pub align_pad_size: f32,
    pub num_all_vs_all: usize,
    pub nz_align: usize,
    pub refine_at_end: bool,
    pub num_bin_tests: usize,
    pub num_filter_tests: usize,
    pub hybrid_shifts: bool,
    pub group_size: usize,
    pub do_spline: bool,
    pub gpu_flags: u32,
    pub defer_sum: bool,
    pub test_mode: bool,
    pub start_assess: i32,
}
/// Source `totalMemoryNeeds`, returning its GiB requirement together with the
/// source output parameters rather than writing through pointers.
pub fn total_memory_needs(options: FrameAlignMemoryOptions) -> (f32, bool, usize) {
    let sum_in_one_pass =
        options.num_bin_tests == 1 && !options.test_mode && options.start_assess < 0;
    let mut hold_full = options.num_all_vs_all.min(options.nz_align);
    let mut hold_align = hold_full;
    if options.num_all_vs_all > 0 {
        if options.refine_at_end {
            hold_align = options.nz_align;
        }
        if options.group_size > 1 {
            hold_align += if options.refine_at_end {
                options.num_all_vs_all.min(options.nz_align)
            } else {
                options.group_size
            };
        }
    } else if options.refine_at_end {
        hold_align = options.nz_align;
    }
    if options.gpu_flags & GPU_FOR_ALIGNING != 0 {
        hold_align = 0;
    }
    if (!options.hybrid_shifts
        && options.num_filter_tests > 1
        && sum_in_one_pass
        && options.num_all_vs_all > 0)
        || options.refine_at_end
        || options.defer_sum
        || options.do_spline
    {
        hold_full = options.nz_align;
    }
    if options.gpu_flags & STACK_FULL_ON_GPU != 0 {
        if options.gpu_flags & GPU_STACK_LIMITED != 0 {
            let limit = ((options.gpu_flags >> GPU_STACK_LIM_SHIFT) & GPU_STACK_LIM_MASK) as usize;
            hold_full = hold_full.saturating_sub(limit);
        } else {
            hold_full = 0;
        }
    }
    if (options.num_bin_tests > 1 || options.test_mode)
        && options.num_all_vs_all > 0
        && options.start_assess < 0
    {
        hold_full = 0;
    }
    let bytes = 2. * options.sum_pad_size
        + (hold_align as f32 + 4.) * options.align_pad_size
        + options.full_pad_size
        + hold_full as f32 * options.full_data_size as f32 * options.full_pad_size / 4.;
    (bytes / (1024. * 1024. * 1024.), sum_in_one_pass, hold_full)
}

/// Source `preprocPadGpuMemoryFits`, retaining its byte accounting while
/// leaving device allocation to an opt-in backend.
pub fn preproc_pad_gpu_memory_fits(
    unpadded_x: usize,
    unpadded_y: usize,
    data_size: usize,
    binning: usize,
    has_gain: bool,
    has_defect: bool,
    has_truncation: bool,
    preprocess: bool,
    noise_taper: bool,
    bin_pad: bool,
    free_memory: f32,
) -> (bool, f32) {
    let pixels = unpadded_x.saturating_mul(unpadded_y) as f32;
    let processing_needed = has_gain || has_defect || has_truncation;
    let pass_size = if processing_needed && !preprocess {
        4
    } else {
        data_size
    };
    let mut needed = 0.;
    if preprocess && has_gain {
        needed += 4. * pixels;
    }
    if preprocess && has_defect {
        needed += pixels;
    }
    if noise_taper || bin_pad {
        needed += pass_size as f32 * pixels;
    }
    if bin_pad && binning > 1 {
        needed += 4. * pixels / binning as f32;
    }
    (needed < free_memory, needed)
}

/// Source `findPreprocPadGpuFlags` capability calculation.  It reports the
/// preferred safe-backend operation set and its byte requirement; callers can
/// then keep the CPU path when no backend is registered.
pub fn find_preproc_pad_gpu_flags(
    unpadded_x: usize,
    unpadded_y: usize,
    data_size: usize,
    binning: usize,
    has_gain: bool,
    has_defect: bool,
    has_truncation: bool,
    expected_frames: usize,
    free_memory: f32,
    stack_margin: f32,
    input_flags: u32,
) -> (f32, u32) {
    let align = input_flags & GPU_FOR_ALIGNING != 0;
    let sum = input_flags & GPU_FOR_SUMMING != 0;
    let preprocessing = GPU_DO_PREPROCESS
        | if has_gain { GPU_DO_GAIN_NORM } else { 0 }
        | if has_defect { GPU_CORRECT_DEFECTS } else { 0 };
    let (fits_all, all_need) = preproc_pad_gpu_memory_fits(
        unpadded_x,
        unpadded_y,
        data_size,
        binning,
        has_gain,
        has_defect,
        has_truncation,
        true,
        !(!sum && align),
        align,
        free_memory,
    );
    let (fits_raw, raw_need) = preproc_pad_gpu_memory_fits(
        unpadded_x,
        unpadded_y,
        data_size,
        binning,
        has_gain,
        has_defect,
        has_truncation,
        false,
        !(!sum && align),
        align,
        free_memory,
    );
    if !fits_all && !fits_raw {
        return (0., input_flags);
    }
    let mut flags = input_flags;
    let mut needed = if fits_all {
        flags |= preprocessing;
        all_need
    } else {
        raw_need
    };
    if align {
        flags |= GPU_DO_BIN_PAD;
    }
    if sum {
        flags |= GPU_DO_NOISE_TAPER;
    }
    let pixels = unpadded_x.saturating_mul(unpadded_y) as f32;
    let stack_bytes = data_size as f32 * pixels;
    let additional = ((free_memory - needed - stack_margin) / stack_bytes)
        .floor()
        .max(0.) as usize;
    if additional >= expected_frames.saturating_sub(1) {
        needed += expected_frames.saturating_sub(1) as f32 * stack_bytes;
        flags |= STACK_FULL_ON_GPU;
    } else if additional > 0 {
        needed += additional as f32 * stack_bytes;
        let limit = (additional + 1).min(GPU_STACK_LIM_MASK as usize) as u32;
        flags |= STACK_FULL_ON_GPU | GPU_STACK_LIMITED | (limit << GPU_STACK_LIM_SHIFT);
    }
    (needed, flags)
}

/// Source `setTruncationLimit`: calculate the upper quantile used by frame
/// preprocessing without altering caller-owned pixels.
pub fn set_truncation_limit(values: &[f32], fraction_to_truncate: f32) -> Option<f32> {
    if values.is_empty() || !(0. ..1.).contains(&fraction_to_truncate) {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f32::total_cmp);
    let index = ((1. - fraction_to_truncate) * ordered.len() as f32)
        .floor()
        .min((ordered.len() - 1) as f32) as usize;
    ordered.get(index).copied()
}

/// Negative source truncation limits mean a number of standard deviations
/// above the sampled mean.  Positive values are already explicit limits.
pub fn truncation_limit_from_sigma(values: &[f32], requested_limit: f32) -> Option<f32> {
    if requested_limit >= 0. {
        return Some(requested_limit);
    }
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let deviation = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32)
        .sqrt();
    Some(mean - requested_limit * deviation)
}

/// Source `FrameAlign::leastCommonMultiple`.
pub fn least_common_multiple(mut first: usize, mut second: usize) -> usize {
    let product = first.saturating_mul(second);
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    if first == 0 { 0 } else { product / first }
}
/// Source `FrameAlign::wrapImage`, with ordinary owned slices in place of raw arrays.
pub fn wrap_image(
    from: &[f32],
    from_stride: usize,
    from_width: usize,
    from_height: usize,
    to: &mut [f32],
    to_stride: usize,
    to_width: usize,
    to_height: usize,
    x_offset: isize,
    y_offset: isize,
) {
    assert!(from.len() >= from_stride * from_height && to.len() >= to_stride * to_height);
    for y in 0..to_height {
        let source_y = (y as isize - y_offset).rem_euclid(from_height as isize) as usize;
        for x in 0..to_width {
            let source_x = (x as isize - x_offset).rem_euclid(from_width as isize) as usize;
            to[x + y * to_stride] = from[source_x + source_y * from_stride];
        }
    }
}
/// Source `FrameAlign::analyzeFRCcrossings`.
pub fn analyze_frc_crossings(ring_corrs: &[f32], delta: f32) -> (f32, f32, f32, f32) {
    fn crossing(values: &[f32], delta: f32, level: f32) -> f32 {
        for index in 1..values.len() {
            if values[index] <= level && values[index - 1] > level {
                return (index as f32 - 1.) * delta
                    + delta * (values[index - 1] - level) / (values[index - 1] - values[index]);
            }
        }
        0.
    }
    let center = (0.25 / delta - 0.5).round() as isize;
    let width = (0.075 / delta).round().max(1.) as isize;
    let mut half_nyquist = 0.;
    let mut count = 0;
    for index in center - width / 2..center - width / 2 + width {
        if let Some(&value) = ring_corrs.get(index.max(0) as usize) {
            half_nyquist += value;
            count += 1;
        }
    }
    (
        crossing(ring_corrs, delta, 0.5),
        crossing(ring_corrs, delta, 0.25),
        crossing(ring_corrs, delta, 0.125),
        if count == 0 {
            0.
        } else {
            half_nyquist / count as f32
        },
    )
}
/// Source `FrameAlign::smoothedTotalDistance`'s total-distance measurement.
pub fn total_shift_distance(x: &[f32], y: &[f32]) -> f32 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xx, yy)| (xx[1] - xx[0]).hypot(yy[1] - yy[0]))
        .sum()
}
/// Source `FrameAlign::frameShiftFromGroups` for an owned group shift table.
pub fn frame_shift_from_groups(
    frame: usize,
    group_size: usize,
    x_groups: &[f32],
    y_groups: &[f32],
) -> Option<(f32, f32)> {
    if group_size == 0 {
        return None;
    }
    if x_groups.len() != y_groups.len() || x_groups.is_empty() {
        return None;
    }
    if group_size == 1 || x_groups.len() == 1 {
        return x_groups
            .get(frame)
            .zip(y_groups.get(frame))
            .map(|(&x, &y)| (x, y));
    }
    let real_index = frame as f32 - (group_size as f32 - 1.) / 2.;
    if real_index <= 0. {
        return Some((x_groups[0], y_groups[0]));
    }
    let last = x_groups.len() - 1;
    if real_index >= last as f32 {
        return Some((x_groups[last], y_groups[last]));
    }
    let index = real_index.floor() as usize;
    let fraction = real_index - index as f32;
    Some((
        (1. - fraction) * x_groups[index] + fraction * x_groups[index + 1],
        (1. - fraction) * y_groups[index] + fraction * y_groups[index + 1],
    ))
}
/// Source `FrameAlign::getPadSizesBytes`; values are bytes for padded float FFT arrays.
pub fn get_pad_sizes_bytes(
    nx: usize,
    ny: usize,
    full_taper: f32,
    sum_bin: usize,
    align_bin: usize,
) -> (f32, f32, f32) {
    let full = 4. * (1. + 2. * full_taper).powi(2) * nx as f32 * ny as f32;
    let sum = full / sum_bin.max(1).pow(2) as f32;
    let align = 4. * nx as f32 * ny as f32 / align_bin.max(1).pow(2) as f32;
    (full, sum, align)
}
/// Source `FrameAlign::gpuMemoryNeeds`, returned instead of mutating output pointers.
pub fn gpu_memory_needs(
    full: f32,
    sum: f32,
    align: f32,
    num_all_vs_all: usize,
    nz_align: usize,
    refine: bool,
    group: usize,
) -> (f32, f32) {
    let sum_need = full + sum + 2. * full.max(sum);
    let mut held = if num_all_vs_all == 0 && !refine {
        6
    } else if refine {
        6 + nz_align
    } else {
        4 + num_all_vs_all.min(nz_align)
    };
    if group > 1 {
        held += if refine {
            num_all_vs_all.min(nz_align)
        } else {
            group
        };
    }
    let ali_need = held as f32 * align;
    (sum_need, ali_need)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_facades_own_and_release_alignment_workspace() {
        let mut align = frame_align();
        align.initialize(1, 1, 0., 0., 1, 1, 1).unwrap();
        align.frames.push(vec![3.]);
        free_frame_align(align);
    }
    #[test]
    fn source_utilities_preserve_wrap_and_lcm() {
        assert_eq!(least_common_multiple(6, 8), 24);
        let mut out = vec![0.; 4];
        wrap_image(&[1., 2., 3., 4.], 2, 2, 2, &mut out, 2, 2, 2, 1, 0);
        assert_eq!(out, vec![2., 1., 4., 3.]);
    }
    #[test]
    fn crossings_are_interpolated() {
        let (half, quarter, _, _) = analyze_frc_crossings(&[1., 0.75, 0.25], 0.1);
        assert!((half - 0.15).abs() < 1.0e-6);
        assert!((quarter - 0.2).abs() < 1.0e-6);
    }
    #[test]
    fn source_dark_type_and_centered_gain_geometry_are_preserved() {
        let mut align = FrameAlign::default();
        align.initialize(1, 1, 0., 0., 2, 1, 1).unwrap();
        let frame = [10u8, 20];
        let dark = [2u8, 0, 3, 0];
        let gain = [1., 2., 3., 4.];
        assert_eq!(
            align
                .preprocess_frame_with_gain_geometry(
                    &frame,
                    MRC_MODE_BYTE,
                    Some(&dark),
                    Some(&gain),
                    4,
                    1,
                    0.
                )
                .unwrap(),
            vec![16., 51.]
        );
    }
    #[test]
    fn source_ctf_masks_and_memory_plan_are_owned() {
        let (full, sub, delta) =
            alignment_ctf_masks(4, 4, 0.1, &[0.2], 0.1, &[0.3], 1, false).unwrap();
        assert_eq!(full.len(), 24);
        assert_eq!(full[0], 0.);
        assert_eq!(sub.len(), 1);
        assert!(delta > 0.);
        let (_, one_pass, held) = total_memory_needs(FrameAlignMemoryOptions {
            full_pad_size: 100.,
            full_data_size: 4,
            sum_pad_size: 30.,
            align_pad_size: 20.,
            num_all_vs_all: 3,
            nz_align: 5,
            num_bin_tests: 1,
            start_assess: -1,
            ..Default::default()
        });
        assert!(one_pass);
        assert_eq!(held, 3);
    }
    #[test]
    fn all_pairs_regression_and_weighted_sum_are_owned() {
        let mut align = FrameAlign::default();
        align.initialize(1, 1, 0., 0., 2, 2, 1).unwrap();
        align.frames = vec![vec![1., 2., 3., 4.], vec![1., 2., 3., 4.]];
        align.x_shifts = vec![0., 0.];
        align.y_shifts = vec![0., 0.];
        align.find_all_vs_all_alignment().unwrap();
        assert_eq!(align.x_shifts, vec![0., 0.]);
        assert_eq!(align.setup_dose_weighting(2., 2.), vec![1., (-1f32).exp()]);
        assert_eq!(
            align.finish_align_and_sum(None).unwrap(),
            vec![1., 2., 3., 4.]
        );
        assert_eq!(FrameAlign::do_regression(&[2., 4., 6.]), Some((2., 2.)));
        let (_, slope, used) =
            FrameAlign::do_robust_regression(&[0., 1., 2., 50., 4.], 1.5).unwrap();
        assert!(!used[3]);
        assert!((slope - 1.).abs() < 0.2);
        assert_eq!(
            FrameAlign::regress_all_vs_all(3, &[(0, 1, 2., 1.), (1, 2, 3., 4.), (0, 2, 5., 5.)])
                .unwrap(),
            (vec![0., 2., 5.], vec![0., 1., 5.])
        );
        let (smooth, raw, sx, sy, _) =
            FrameAlign::smoothed_total_distance(&[0., 1., 2., 3., 4.], &[0.; 5]).unwrap();
        assert_eq!(
            (smooth, raw, sx, sy),
            (4., 4., vec![0., 1., 2., 3., 4.], vec![0.; 5])
        );
        align
            .adjust_and_push_shifts(1, &[3., 5.], &[1., 2.])
            .unwrap();
        assert_eq!(
            align.get_all_frame_shifts().unwrap(),
            (vec![-2., 0.], vec![-1., 0.])
        );
        let (spline_x, spline_y) = align
            .spline_smooth(&[0., 3., 0.], &[0., 0., 0.], 1)
            .unwrap();
        assert_eq!(spline_x.len(), 3);
        assert_eq!(spline_y, vec![0.; 3]);
        assert_eq!(set_truncation_limit(&[1., 3., 2., 4.], 0.25), Some(4.));
        align
            .setup_dose_weighting_full(0., vec![1., 1.], 1., 1., 1., 1., 1., Some(vec![1.; 8]), 8)
            .unwrap();
        assert_eq!(align.reweight_filter.len(), 8);
        assert!(align.dose_weight_delta > 0.);
    }
    #[test]
    fn unavailable_gpu_never_creates_a_foreign_handle() {
        assert_eq!(gpu_strategy(false, false), GpuStrategy::Cpu);
        assert_eq!(
            gpu_strategy(true, false),
            GpuStrategy::RequestedButUnavailable
        );
        assert!(FrameAlign::gpu_available(0, false).is_ok());
        assert!(FrameAlign::gpu_available(1, false).is_err());
    }
    #[test]
    fn packed_fft_filter_shift_and_defect_paths_are_owned() {
        let fft = vec![1., 0., 2., 0., 3., 0., 4., 0.];
        let mut sum = vec![0.; 8];
        filter_and_add_to_sum(&fft, &mut sum, 2, 2, &[1., 1.], 0.5).unwrap();
        assert_eq!(sum, fft);
        let mut shifted_sum = vec![0.; 8];
        add_to_sums(&fft, &mut shifted_sum, 2, 2, 0., 0., None).unwrap();
        assert_eq!(shifted_sum, fft);
        let mut defects = CameraDefects::default();
        defects.bad_pixel_x.push(1);
        defects.bad_pixel_y.push(1);
        let mut frame = vec![1., 2., 3., 99.];
        FrameAlign::correct_defects(&mut frame, 2, 2, &defects).unwrap();
        assert_eq!(frame[3], 2.5);
        let mut partial = CameraDefects::default();
        partial.partial_bad_col.push(1);
        partial.partial_bad_width.push(1);
        partial.partial_bad_start_y.push(0);
        partial.partial_bad_end_y.push(0);
        let mut partial_frame = vec![1., 90., 3., 4.];
        FrameAlign::correct_defects(&mut partial_frame, 2, 2, &partial).unwrap();
        assert_eq!(partial_frame[1], 2.5);
        let mut store = GpuFrameStore {
            full_frames: vec![vec![1.]],
            align_ffts: vec![vec![2.]],
        };
        assert_eq!(store.recover_gpu_align_ffts(0).unwrap(), vec![2.]);
        assert_eq!(store.recover_gpu_full_stack(), vec![vec![1.]]);
        assert_eq!(
            preproc_pad_gpu_memory_fits(4, 4, 2, 2, true, false, false, true, true, true, 1_000.),
            (true, 128.)
        );
    }
    #[test]
    fn shrmem_backend_drives_the_concrete_owned_aligner() {
        let mut align = FrameAlign::default();
        let mut initialize = InitializeParams {
            ret_val: -1,
            bin_sum: 1,
            bin_align: 1,
            trim_frac: 0.,
            num_all_vs_all: 0,
            cum_align_at_end: 0,
            use_hybrid: 0,
            defer_sum: 0,
            group_size: 1,
            nx: 2,
            ny: 2,
            pad_frac: 0.,
            taper_frac: 0.,
            anti_filt_type: 0,
            radius1: 0.,
            radius2: vec![],
            sigma1: 0.,
            sigma2: vec![],
            num_filters: 1,
            max_shift: 1,
            k_factor: 0.,
            max_max_weight: 0.,
            summing_mode: 0,
            expected_z: 1,
            make_unwgt_sum: 0,
            gpu_flags: 0,
            debug: 0,
            messages: String::new(),
        };
        FrameAlignBackend::initialize(&mut align, &mut initialize);
        assert_eq!(initialize.ret_val, 0);
        let mut next = NextFrameParams {
            ret_val: -1,
            frame: vec![1, 2, 3, 4],
            pixel_type: MRC_MODE_BYTE,
            gain_reference: None,
            defects: None,
            trunc_limit: 0.,
            camera_size_x: 0,
            camera_size_y: 0,
            defect_binning: 1,
            shift_x: 1.,
            shift_y: 1.,
            messages: String::new(),
        };
        FrameAlignBackend::next_frame(&mut align, &mut next);
        assert_eq!((next.ret_val, next.shift_x, next.shift_y), (0, 0., 0.));
    }
}
