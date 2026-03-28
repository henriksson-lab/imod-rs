//! Bead detection via template matching with normalized cross-correlation.

use imod_fft::cross_correlate_2d;

/// A detected bead in one frame.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub frame: usize,
}

/// Create a synthetic circular template for bead matching.
///
/// Returns a flat `diameter x diameter` image where pixels inside the circle
/// are 1.0 (bright) and outside are -1.0 (dark), normalised to zero mean.
pub fn create_template(diameter: usize) -> Vec<f32> {
    let r = diameter as f32 / 2.0;
    let cx = r;
    let cy = r;
    let mut tpl = vec![0.0f32; diameter * diameter];
    let mut sum = 0.0f32;

    for y in 0..diameter {
        for x in 0..diameter {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = if dist <= r { 1.0 } else { -1.0 };
            tpl[y * diameter + x] = val;
            sum += val;
        }
    }

    // Subtract mean so template is zero-mean
    let mean = sum / (diameter * diameter) as f32;
    for v in &mut tpl {
        *v -= mean;
    }

    tpl
}

/// Next power of two >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Detect beads in a single frame using cross-correlation with the template.
///
/// The frame is cross-correlated with the template (both zero-padded to a
/// common FFT size).  Peaks above `threshold` (relative to the maximum CC
/// value) are returned as detections after sub-pixel refinement and
/// non-maximum suppression.
pub fn detect_beads(
    frame: &[f32],
    nx: usize,
    ny: usize,
    template: &[f32],
    tsize: usize,
    threshold: f32,
) -> Vec<Detection> {
    // Determine FFT size (must be power-of-two and fit both image and template)
    let fft_nx = next_pow2(nx);
    let fft_ny = next_pow2(ny);

    // Zero-pad frame
    let mut padded_frame = vec![0.0f32; fft_nx * fft_ny];
    for y in 0..ny {
        for x in 0..nx {
            padded_frame[y * fft_nx + x] = frame[y * nx + x];
        }
    }

    // Zero-pad template (centred at origin for zero-phase)
    let mut padded_tpl = vec![0.0f32; fft_nx * fft_ny];
    let half = tsize / 2;
    for ty in 0..tsize {
        for tx in 0..tsize {
            // Wrap-around placement so centre of template is at (0,0)
            let dy = (ty as isize - half as isize).rem_euclid(fft_ny as isize) as usize;
            let dx = (tx as isize - half as isize).rem_euclid(fft_nx as isize) as usize;
            padded_tpl[dy * fft_nx + dx] = template[ty * tsize + tx];
        }
    }

    // Cross-correlate
    let cc = cross_correlate_2d(&padded_frame, &padded_tpl, fft_nx, fft_ny);

    // Compute normalisation: local energy in the frame under the template.
    // For speed we use a global normalisation based on image std-dev rather
    // than per-pixel sliding-window energy (good enough for thresholding).
    let n_pixels = (nx * ny) as f32;
    let frame_mean: f32 = frame.iter().sum::<f32>() / n_pixels;
    let frame_var: f32 =
        frame.iter().map(|&v| (v - frame_mean) * (v - frame_mean)).sum::<f32>() / n_pixels;
    let frame_std = frame_var.sqrt().max(1e-12);

    let tpl_energy: f32 = template.iter().map(|v| v * v).sum::<f32>();
    let norm = frame_std * tpl_energy.sqrt() * (tsize * tsize) as f32;

    // Find peaks above threshold (only within original image bounds)
    let abs_threshold = threshold * norm;
    let mut dets = Vec::new();

    let margin = half + 1;
    if nx <= 2 * margin || ny <= 2 * margin {
        return dets;
    }

    for y in margin..(ny - margin) {
        for x in margin..(nx - margin) {
            let val = cc[y * fft_nx + x];
            if val > abs_threshold {
                let (sx, sy) = refine_subpixel(&cc, fft_nx, x, y, 2);
                dets.push(Detection {
                    x: sx,
                    y: sy,
                    score: val / norm,
                    frame: 0, // caller sets this
                });
            }
        }
    }

    // Sort by descending score for NMS
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    non_max_suppression(&mut dets, tsize as f32 * 0.75);

    dets
}

/// Sub-pixel refinement via centre-of-mass around a peak.
fn refine_subpixel(cc: &[f32], stride: usize, px: usize, py: usize, radius: usize) -> (f32, f32) {
    let mut sum_w = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;

    let min_val = cc[py * stride + px] * 0.5; // only use pixels above half-max

    let y_lo = py.saturating_sub(radius);
    let y_hi = (py + radius + 1).min(cc.len() / stride);
    let x_lo = px.saturating_sub(radius);
    let x_hi = (px + radius + 1).min(stride);

    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let v = cc[y * stride + x];
            if v > min_val {
                sum_w += v;
                sum_x += v * x as f32;
                sum_y += v * y as f32;
            }
        }
    }

    if sum_w > 0.0 {
        (sum_x / sum_w, sum_y / sum_w)
    } else {
        (px as f32, py as f32)
    }
}

/// Non-maximum suppression: remove detections too close to a stronger one.
fn non_max_suppression(dets: &mut Vec<Detection>, min_dist: f32) {
    let min_dist_sq = min_dist * min_dist;
    let mut keep = vec![true; dets.len()];

    for i in 0..dets.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..dets.len() {
            if !keep[j] {
                continue;
            }
            let dx = dets[i].x - dets[j].x;
            let dy = dets[i].y - dets[j].y;
            if dx * dx + dy * dy < min_dist_sq {
                keep[j] = false; // j has lower score (sorted descending)
            }
        }
    }

    let mut idx = 0;
    dets.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_is_zero_mean() {
        let t = create_template(12);
        let sum: f32 = t.iter().sum();
        assert!(sum.abs() < 1e-3, "template mean should be ~0, got sum={sum}");
    }

    #[test]
    fn detect_synthetic_bead() {
        let nx = 64;
        let ny = 64;
        let diameter = 8;

        // Create an image with a bright disk at (32, 32)
        let mut img = vec![0.0f32; nx * ny];
        let cx = 32.0f32;
        let cy = 32.0f32;
        let r = diameter as f32 / 2.0;
        for y in 0..ny {
            for x in 0..nx {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                if (dx * dx + dy * dy).sqrt() <= r {
                    img[y * nx + x] = 10.0;
                }
            }
        }

        let tpl = create_template(diameter);
        let dets = detect_beads(&img, nx, ny, &tpl, diameter, 0.1);
        assert!(!dets.is_empty(), "should detect at least one bead");

        // The strongest detection should be near (32, 32)
        let best = &dets[0];
        assert!(
            (best.x - cx).abs() < 3.0 && (best.y - cy).abs() < 3.0,
            "best detection at ({}, {}) should be near ({cx}, {cy})",
            best.x,
            best.y
        );
    }
}
