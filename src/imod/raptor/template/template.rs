//! Owned scalar foundations from `IMOD/raptor/template/template.cpp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimplePoint2d {
    pub x: i32,
    pub y: i32,
    pub score: f64,
}
pub fn diff_vector(values: &[f32]) -> Vec<f32> {
    if values.len() < 2 {
        return vec![];
    }
    values.windows(2).map(|v| v[1] - v[0]).collect()
}
pub fn nnz(values: &[f32]) -> usize {
    values.iter().filter(|&&v| v != 0.).count()
}
pub fn within_diameter(x1: f32, y1: f32, x2: f32, y2: f32, diameter: i32) -> bool {
    let dx = x1 - x2;
    let dy = y1 - y2;
    dx * dx + dy * dy < (diameter * diameter) as f32
}
pub fn find_min(points: &mut Vec<SimplePoint2d>) {
    if let Some((index, _)) = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.score.partial_cmp(&b.score).unwrap())
    {
        points.remove(index);
    }
}
pub fn process_simple_point_2d(
    points: &mut Vec<SimplePoint2d>,
    point: SimplePoint2d,
    max_markers: usize,
    diameter: i32,
) -> bool {
    if points.iter().any(|p| {
        within_diameter(
            p.x as f32,
            p.y as f32,
            point.x as f32,
            point.y as f32,
            diameter,
        )
    }) {
        return false;
    }
    points.push(point);
    if points.len() > max_markers {
        find_min(points);
    }
    true
}
pub fn crop_borders(image: &[f32], width: usize, height: usize, template_side: usize) -> Vec<f32> {
    let mut result = image.to_vec();
    for y in 0..height {
        for x in 0..width {
            if x < template_side
                || y < template_side
                || x + template_side >= width
                || y + template_side >= height
            {
                result[y * width + x] = 0.;
            }
        }
    }
    result
}
/// Owned `findPeaks`: thresholded local maxima, ordered by score and capped.
pub fn find_peaks(
    scores: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
    max_markers: usize,
    min_markers: usize,
    template_side: usize,
    frame_id: i32,
    marker_type: i32,
) -> Vec<SimplePoint2d> {
    let mut peaks = Vec::new();
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let value = scores[y * width + x];
            if value < threshold {
                continue;
            }
            let mut maximum = true;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    if scores[yy * width + xx] > value {
                        maximum = false;
                    }
                }
            }
            if maximum {
                process_simple_point_2d(
                    &mut peaks,
                    SimplePoint2d {
                        x: x as i32,
                        y: y as i32,
                        score: value as f64,
                    },
                    max_markers,
                    template_side as i32,
                );
            }
        }
    }
    peaks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    if peaks.len() < min_markers {
        Vec::new()
    } else {
        let _ = (frame_id, marker_type);
        peaks
    }
}
/// Source `computeNCC` numerical core: valid normalized cross correlation.
pub fn compute_ncc(
    image: &[f32],
    width: usize,
    height: usize,
    template: &[f32],
    side: usize,
) -> Vec<f32> {
    if side == 0 || template.len() != side * side || width < side || height < side {
        return Vec::new();
    }
    let mean = template.iter().sum::<f32>() / template.len() as f32;
    let norm = template
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f32>()
        .sqrt();
    let mut out = vec![0.; (width - side + 1) * (height - side + 1)];
    for y in 0..=height - side {
        for x in 0..=width - side {
            let mut avg = 0.;
            for yy in 0..side {
                for xx in 0..side {
                    avg += image[(y + yy) * width + x + xx];
                }
            }
            avg /= template.len() as f32;
            let (mut n, mut d) = (0., 0.);
            for yy in 0..side {
                for xx in 0..side {
                    let a = image[(y + yy) * width + x + xx] - avg;
                    let b = template[yy * side + xx] - mean;
                    n += a * b;
                    d += a * a;
                }
            }
            out[y * (width - side + 1) + x] = if norm * d.sqrt() > f32::EPSILON {
                n / (norm * d.sqrt())
            } else {
                0.
            };
        }
    }
    out
}
pub fn create_synthetic_template(diameter: usize, white: bool) -> Vec<f32> {
    let side = diameter * 2 + 1;
    let mut result = vec![0.; side * side];
    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - diameter as f32;
            let dy = y as f32 - diameter as f32;
            let inside = dx * dx + dy * dy <= diameter as f32 * diameter as f32;
            result[y * side + x] = if inside == white { 1. } else { 0. };
        }
    }
    result
}
pub fn remove_lines_xray(image: &mut [f32], mask: &[f32]) {
    for (pixel, &weight) in image.iter_mut().zip(mask) {
        if weight == 0. {
            *pixel = 0.;
        }
    }
}
pub fn is_peak_in_frame(points: &[SimplePoint2d], point: SimplePoint2d) -> bool {
    points.iter().any(|p| p.x == point.x && p.y == point.y)
}
pub fn merge_peaks(
    mut first: Vec<SimplePoint2d>,
    second: Vec<SimplePoint2d>,
    maximum: usize,
) -> Vec<SimplePoint2d> {
    for p in second {
        if !is_peak_in_frame(&first, p) {
            first.push(p);
        }
    }
    first.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    first.truncate(maximum);
    first
}
pub fn simple_point_cmp(a: SimplePoint2d, b: SimplePoint2d) -> bool {
    a.score > b.score
}
pub fn point_cmp(a: SimplePoint2d, b: SimplePoint2d) -> bool {
    a.score > b.score
}
pub fn centroid(points: &[SimplePoint2d]) -> Option<(f32, f32)> {
    let total = points.iter().map(|p| p.score.max(0.)).sum::<f64>();
    if total == 0. {
        return None;
    }
    Some((
        points
            .iter()
            .map(|p| p.x as f64 * p.score.max(0.))
            .sum::<f64>() as f32
            / total as f32,
        points
            .iter()
            .map(|p| p.y as f64 * p.score.max(0.))
            .sum::<f64>() as f32
            / total as f32,
    ))
}
/// Source matching pipeline: NCC followed by source-style local peak selection.
pub fn match_template_peaks(
    image: &[f32],
    width: usize,
    height: usize,
    template: &[f32],
    side: usize,
    threshold: f32,
    max_markers: usize,
    min_markers: usize,
    frame_id: i32,
    marker_type: i32,
) -> Vec<SimplePoint2d> {
    let scores = compute_ncc(image, width, height, template, side);
    find_peaks(
        &scores,
        width.saturating_sub(side).saturating_add(1),
        height.saturating_sub(side).saturating_add(1),
        threshold,
        max_markers,
        min_markers,
        side,
        frame_id,
        marker_type,
    )
}
/// Owned frame payload used by source `estimateNumberOfMarkers` and `findAllPeaks`.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateFrame {
    pub image: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub frame_id: i32,
    pub discard: bool,
    pub peaks: Vec<SimplePoint2d>,
}
/// C `estimateNumberOfMarkers`: evaluate every template size and retain the largest peak count.
pub fn estimate_number_of_markers(
    frames: &[TemplateFrame],
    templates: &[Vec<f32>],
    sides: &[usize],
    threshold: f32,
    max_markers: usize,
) -> usize {
    frames
        .iter()
        .filter(|f| !f.discard)
        .flat_map(|f| {
            templates.iter().zip(sides).map(move |(t, &side)| {
                match_template_peaks(
                    &f.image,
                    f.width,
                    f.height,
                    t,
                    side,
                    threshold,
                    max_markers,
                    0,
                    f.frame_id,
                    0,
                )
                .len()
            })
        })
        .max()
        .unwrap_or(0)
}
/// C `findAllPeaks`: source orchestration expressed with owned frame images and template variants.
pub fn find_all_peaks(
    frames: &mut [TemplateFrame],
    templates: &[Vec<f32>],
    sides: &[usize],
    threshold: f32,
    max_markers: usize,
    min_markers: usize,
) {
    for frame in frames {
        if frame.discard {
            continue;
        }
        frame.peaks.clear();
        for (marker_type, (template, &side)) in templates.iter().zip(sides).enumerate() {
            let peaks = match_template_peaks(
                &frame.image,
                frame.width,
                frame.height,
                template,
                side,
                threshold,
                max_markers,
                min_markers,
                frame.frame_id,
                marker_type as i32,
            );
            frame.peaks = merge_peaks(frame.peaks.clone(), peaks, max_markers);
        }
    }
}
/// Source `mldivide`, using owned pivoted Gaussian elimination for A\\B.
pub fn mldivide(
    a: &[f64],
    rows: usize,
    cols: usize,
    b: &[f64],
    bcols: usize,
) -> Result<Vec<f64>, String> {
    if rows != cols || a.len() != rows * cols || b.len() != rows * bcols {
        return Err("bad matrix size".into());
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for k in 0..rows {
        let p = (k..rows)
            .max_by(|&i, &j| {
                a[i * cols + k]
                    .abs()
                    .partial_cmp(&a[j * cols + k].abs())
                    .unwrap()
            })
            .unwrap();
        if a[p * cols + k] == 0. {
            return Err("singular matrix".into());
        }
        for j in 0..cols {
            a.swap(k * cols + j, p * cols + j);
        }
        for j in 0..bcols {
            b.swap(k * bcols + j, p * bcols + j);
        }
        for i in k + 1..rows {
            let f = a[i * cols + k] / a[k * cols + k];
            for j in k..cols {
                a[i * cols + j] -= f * a[k * cols + j];
            }
            for j in 0..bcols {
                b[i * bcols + j] -= f * b[k * bcols + j];
            }
        }
    }
    for i in (0..rows).rev() {
        for j in 0..bcols {
            for k in i + 1..cols {
                b[i * bcols + j] -= a[i * cols + k] * b[k * bcols + j];
            }
            b[i * bcols + j] /= a[i * cols + i];
        }
    }
    Ok(b)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_helpers() {
        assert_eq!(diff_vector(&[1., 3., 2.]), vec![2., -1.]);
        assert_eq!(nnz(&[0., 1., 0.]), 1);
        assert!(within_diameter(0., 0., 1., 1., 2));
        assert_eq!(
            mldivide(&[2., 0., 0., 2.], 2, 2, &[2., 4.], 1).unwrap(),
            vec![1., 2.]
        );
    }
}
