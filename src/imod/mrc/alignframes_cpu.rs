//! Owned CPU alignment for the MRC stack route in `alignframes`.
//!
//! This keeps the existing small-stack path available while `FrameAlign` is
//! translated with the full native C interface.

use super::AliFrameResult;
use crate::imod::libiimod::iimage::OwnedImageStack;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MRC_MODE_USHORT,
};

pub(super) fn align_image_stack(
    stack: &OwnedImageStack,
    max_shift: usize,
    group_size: i32,
    gain: Option<&[f32]>,
    dark: Option<&[u8]>,
    truncation: f32,
    dose_per_frame: Option<f32>,
    critical_dose: f32,
) -> Result<AliFrameResult, String> {
    let pixels = stack
        .nx
        .checked_mul(stack.ny)
        .ok_or("frame dimensions overflow")?;
    let bytes_per_pixel = match stack.mode {
        MRC_MODE_BYTE => 1,
        MRC_MODE_SHORT | MRC_MODE_USHORT => 2,
        MRC_MODE_FLOAT => 4,
        _ => return Err("frame mode is not real".into()),
    };
    if gain.is_some_and(|values| values.len() < pixels) {
        return Err("gain reference is shorter than frame".into());
    }
    if dark.is_some_and(|values| values.len() < pixels * 2) {
        return Err("dark reference is shorter than frame".into());
    }
    let mut frames = Vec::with_capacity(stack.frames.len());
    for raw in &stack.frames {
        if raw.len() < pixels * bytes_per_pixel {
            return Err("frame is shorter than dimensions".into());
        }
        let mut frame = Vec::with_capacity(pixels);
        for index in 0..pixels {
            let offset = index * bytes_per_pixel;
            let sample = match stack.mode {
                MRC_MODE_BYTE => raw[offset] as f32,
                MRC_MODE_SHORT => {
                    i16::from_ne_bytes(raw[offset..offset + 2].try_into().unwrap()) as f32
                }
                MRC_MODE_USHORT => {
                    u16::from_ne_bytes(raw[offset..offset + 2].try_into().unwrap()) as f32
                }
                _ => f32::from_ne_bytes(raw[offset..offset + 4].try_into().unwrap()),
            };
            let dark_value = dark.map_or(0., |values| {
                let offset = index * 2;
                if stack.mode == MRC_MODE_USHORT {
                    u16::from_ne_bytes(values[offset..offset + 2].try_into().unwrap()) as f32
                } else {
                    i16::from_ne_bytes(values[offset..offset + 2].try_into().unwrap()) as f32
                }
            });
            let mut value = sample - dark_value;
            if let Some(gain) = gain {
                value *= gain[index];
            }
            if truncation > 0. && value > truncation {
                value = truncation;
            }
            frame.push(value);
        }
        frames.push(frame);
    }
    let count = frames.len();
    let mut x_shifts = vec![0.; count];
    let mut y_shifts = vec![0.; count];
    if group_size > 1 {
        let mut measurements =
            Vec::with_capacity(count.saturating_mul(count.saturating_sub(1)) / 2);
        for first in 0..count {
            for second in first + 1..count {
                let (dx, dy) = shift_between(
                    &frames[first],
                    &frames[second],
                    stack.nx,
                    stack.ny,
                    max_shift,
                );
                measurements.push((first, second, dx, dy));
            }
        }
        (x_shifts, y_shifts) = regress_all_vs_all(count, &measurements);
    } else {
        for index in 1..count {
            let (dx, dy) = shift_between(
                &frames[index - 1],
                &frames[index],
                stack.nx,
                stack.ny,
                max_shift,
            );
            x_shifts[index] = dx;
            y_shifts[index] = dy;
        }
    }
    let unweighted_sum = sum_frames(&frames, &x_shifts, &y_shifts, stack.nx, stack.ny, None);
    let weights = dose_per_frame.map(|dose| {
        (0..count)
            .map(|index| {
                if critical_dose > 0. {
                    (-(index as f32) * dose.max(0.) / critical_dose).exp()
                } else {
                    1.
                }
            })
            .collect::<Vec<_>>()
    });
    let weighted_sum = sum_frames(
        &frames,
        &x_shifts,
        &y_shifts,
        stack.nx,
        stack.ny,
        weights.as_deref(),
    );
    Ok(AliFrameResult {
        weighted_sum,
        unweighted_sum,
        x_shifts,
        y_shifts,
    })
}

fn shift_between(
    reference: &[f32],
    moving: &[f32],
    nx: usize,
    ny: usize,
    max_shift: usize,
) -> (f32, f32) {
    let limit = max_shift.max(1) as isize;
    let mut best = (f32::NEG_INFINITY, 0isize, 0isize);
    for dy in -limit..=limit {
        for dx in -limit..=limit {
            let (mut cross, mut ref_sum, mut mov_sum, mut ref_sq, mut mov_sq, mut count) =
                (0., 0., 0., 0., 0., 0);
            for y in 0..ny {
                let sy = y as isize + dy;
                if !(0..ny as isize).contains(&sy) {
                    continue;
                }
                for x in 0..nx {
                    let sx = x as isize + dx;
                    if !(0..nx as isize).contains(&sx) {
                        continue;
                    }
                    let a = reference[x + y * nx];
                    let b = moving[sx as usize + sy as usize * nx];
                    cross += a * b;
                    ref_sum += a;
                    mov_sum += b;
                    ref_sq += a * a;
                    mov_sq += b * b;
                    count += 1;
                }
            }
            let score = if count > 1 {
                let count = count as f32;
                let numerator = cross - ref_sum * mov_sum / count;
                let denominator = ((ref_sq - ref_sum * ref_sum / count)
                    * (mov_sq - mov_sum * mov_sum / count))
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
                best = (score, dx, dy);
            }
        }
    }
    (best.1 as f32, best.2 as f32)
}

fn regress_all_vs_all(
    count: usize,
    measurements: &[(usize, usize, f32, f32)],
) -> (Vec<f32>, Vec<f32>) {
    let mut answer = (vec![0.; count], vec![0.; count]);
    for coordinate in 0..2 {
        let mut matrix = vec![vec![0.; count.saturating_sub(1)]; count.saturating_sub(1)];
        let mut right = vec![0.; count.saturating_sub(1)];
        for &(from, to, x, y) in measurements {
            let value = if coordinate == 0 { x } else { y };
            for (row, sign) in [(from, -1.), (to, 1.)] {
                if row == 0 {
                    continue;
                }
                right[row - 1] += sign * value;
                for (column, other) in [(from, -1.), (to, 1.)] {
                    if column != 0 {
                        matrix[row - 1][column - 1] += sign * other;
                    }
                }
            }
        }
        for pivot in 0..count.saturating_sub(1) {
            let best = (pivot..count - 1)
                .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))
                .unwrap();
            matrix.swap(pivot, best);
            right.swap(pivot, best);
            let divisor = matrix[pivot][pivot];
            if divisor.abs() <= f32::EPSILON {
                continue;
            }
            for column in pivot..count - 1 {
                matrix[pivot][column] /= divisor;
            }
            right[pivot] /= divisor;
            for row in 0..count - 1 {
                if row == pivot {
                    continue;
                }
                let factor = matrix[row][pivot];
                for column in pivot..count - 1 {
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
    answer
}

fn sum_frames(
    frames: &[Vec<f32>],
    x_shifts: &[f32],
    y_shifts: &[f32],
    nx: usize,
    ny: usize,
    weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut sum = vec![0.; nx * ny];
    let mut totals = vec![0.; nx * ny];
    for (index, frame) in frames.iter().enumerate() {
        let weight = weights.map_or(1., |values| values[index]);
        let dx = x_shifts[index].round() as isize;
        let dy = y_shifts[index].round() as isize;
        for y in 0..ny {
            let source_y = y as isize + dy;
            if !(0..ny as isize).contains(&source_y) {
                continue;
            }
            for x in 0..nx {
                let source_x = x as isize + dx;
                if !(0..nx as isize).contains(&source_x) {
                    continue;
                }
                let target = x + y * nx;
                sum[target] += weight * frame[source_x as usize + source_y as usize * nx];
                totals[target] += weight;
            }
        }
    }
    for (pixel, total) in sum.iter_mut().zip(totals) {
        if total != 0. {
            *pixel /= total;
        }
    }
    sum
}
