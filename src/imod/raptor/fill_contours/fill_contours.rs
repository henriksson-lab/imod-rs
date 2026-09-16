//! Owned translation of `IMOD/raptor/fillContours/fillContours.{h,cpp}`.

use crate::imod::raptor::main_classes::constants::PEAK_THRESHOLD_FILL_CONTOURS;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::opencv::cvtemplmatch::{CV_TM_CCOEFF_NORMED, cv_match_template};
use crate::imod::raptor::opencv::cxconvert::CvChannelMatrix;
use crate::imod::raptor::optimization::contour::Contour;
use crate::imod::raptor::optimization::sfm_data::SfmData;

/// C++ `centroid`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Centroid {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

/// Source precondition failures formerly handled through process termination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FillContoursError {
    InvalidData,
    InvalidTemplate,
}

/// C++ `distanceTrajectory`.
pub fn distance_trajectory(x1: &[f64], y1: &[f64], x2: &[f64], y2: &[f64]) -> f64 {
    let mut distances: Vec<f64> = x1
        .iter()
        .zip(y1)
        .zip(x2.iter().zip(y2))
        .filter_map(|((&left_x, &left_y), (&right_x, &right_y))| {
            (left_x > 1.0e-6 && right_x > 1.0e-6)
                .then_some((left_x - right_x).abs() + (left_y - right_y).abs())
        })
        .collect();
    if distances.is_empty() {
        return 1.0e20;
    }
    distances.sort_by(|left, right| left.total_cmp(right));
    distances[distances.len() / 2]
}

/// C++ `print(IplImage *, ostream &)`, returned as owned stream text.
pub fn print(image: &CvChannelMatrix<f32>) -> String {
    let mut text = String::new();
    for row in image.data.chunks(image.cols * image.channels) {
        for value in row {
            text.push_str(&format!("{value} "));
        }
        text.push('\n');
    }
    text
}

/// C++ `findMatchingTranslation` over owned one-channel f32 images.
pub fn find_matching_translation(
    image: &CvChannelMatrix<f32>,
    template: &CvChannelMatrix<f32>,
    _marker_type: i32,
) -> Option<Centroid> {
    if image.channels != 1
        || template.channels != 1
        || template.rows > image.rows
        || template.cols > image.cols
    {
        return None;
    }
    let rows = image.rows - template.rows + 1;
    let cols = image.cols - template.cols + 1;
    let mut result = CvChannelMatrix::new(rows, cols, 1, vec![0.0; rows * cols]).ok()?;
    cv_match_template(image, template, &mut result, CV_TM_CCOEFF_NORMED).ok()?;
    result
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, &score)| {
            (score as f64 >= PEAK_THRESHOLD_FILL_CONTOURS).then_some(Centroid {
                x: (index % cols) as f64,
                y: (index / cols) as f64,
                score: score as f64,
            })
        })
        .max_by(|left, right| left.score.total_cmp(&right.score))
}

/// C++ `beadCenter`; adjusts a proposed displacement to the nearest strong NCC peak.
pub fn bead_center(
    patch: &CvChannelMatrix<f32>,
    perfect_marker_template: &CvChannelMatrix<f32>,
    xx: &mut f64,
    yy: &mut f64,
    marker_type: i32,
) {
    if patch.channels != 1
        || perfect_marker_template.channels != 1
        || perfect_marker_template.rows > patch.rows
        || perfect_marker_template.cols > patch.cols
    {
        return;
    }
    let rows = patch.rows - perfect_marker_template.rows + 1;
    let cols = patch.cols - perfect_marker_template.cols + 1;
    let mut result = match CvChannelMatrix::new(rows, cols, 1, vec![0.0; rows * cols]) {
        Ok(result) => result,
        Err(_) => return,
    };
    if cv_match_template(
        patch,
        perfect_marker_template,
        &mut result,
        CV_TM_CCOEFF_NORMED,
    )
    .is_err()
    {
        return;
    }
    let center_x = 0.5 * (cols.saturating_sub(1)) as f64;
    let center_y = 0.5 * (rows.saturating_sub(1)) as f64;
    let radius = 0.25 * perfect_marker_template.cols.saturating_sub(1) as f64;
    let _ = marker_type;
    let mut peaks: Vec<Centroid> = result
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, &score)| {
            (score as f64 >= PEAK_THRESHOLD_FILL_CONTOURS).then_some(Centroid {
                x: (index % cols) as f64,
                y: (index / cols) as f64,
                score: score as f64,
            })
        })
        .collect();
    peaks.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut best_distance = f64::INFINITY;
    for peak in peaks.into_iter().take(5) {
        let candidate_x = peak.x - center_x;
        let candidate_y = peak.y - center_y;
        let distance = ((*xx - candidate_x).powi(2) + (*yy - candidate_y).powi(2)).sqrt();
        if distance < best_distance {
            best_distance = distance;
            if distance < radius {
                *xx = candidate_x;
                *yy = candidate_y;
            }
        }
    }
}

/// C++ `joinSimilarContours`; consuming `sfm` replaces its source delete/new cycle.
pub fn join_similar_contours(mut sfm: SfmData) -> Result<SfmData, FillContoursError> {
    let (mut contour_x, mut contour_y, reproj_x, reproj_y, marker_type) = match (
        sfm.contour_x.take(),
        sfm.contour_y.take(),
        sfm.reproj_x.take(),
        sfm.reproj_y.take(),
        sfm.marker_type.take(),
    ) {
        (Some(x), Some(y), Some(rx), Some(ry), Some(types)) => (x, y, rx, ry, types),
        _ => return Err(FillContoursError::InvalidData),
    };
    let rows = contour_x.num_trajectories;
    let columns = contour_x.num_frames;
    if contour_y.num_trajectories != rows
        || contour_y.num_frames != columns
        || reproj_x.num_trajectories != rows
        || reproj_y.num_trajectories != rows
        || marker_type.len() != rows
    {
        return Err(FillContoursError::InvalidData);
    }
    let mut erased = vec![false; rows];
    for first in 0..rows.saturating_sub(1) {
        if erased[first] {
            continue;
        }
        for second in first + 1..rows {
            if erased[second] {
                continue;
            }
            let a = first * columns;
            let b = second * columns;
            if distance_trajectory(
                &contour_x.scores[a..a + columns],
                &contour_y.scores[a..a + columns],
                &contour_x.scores[b..b + columns],
                &contour_y.scores[b..b + columns],
            ) < 3.0
            {
                for column in 0..columns {
                    let left = a + column;
                    let right = b + column;
                    if contour_x.scores[left] < 1.0e-6 && contour_x.scores[right] > 1.0e-6 {
                        contour_x.scores[left] = contour_x.scores[right];
                        contour_y.scores[left] = contour_y.scores[right];
                    }
                    contour_x.scores[right] = 0.0;
                    contour_y.scores[right] = 0.0;
                }
                erased[second] = true;
            }
        }
    }
    let kept: Vec<usize> = (0..rows).filter(|&row| !erased[row]).collect();
    let copy = |source: &Contour, coordinate| Contour {
        x_or_y: coordinate,
        num_trajectories: kept.len(),
        num_frames: columns,
        scores: kept
            .iter()
            .flat_map(|&row| {
                source.scores[row * columns..(row + 1) * columns]
                    .iter()
                    .copied()
            })
            .collect(),
    };
    Ok(SfmData::with_reprojections(
        copy(&contour_x, 1),
        copy(&contour_y, 2),
        copy(&reproj_x, 1),
        copy(&reproj_y, 2),
        kept.iter().map(|&row| marker_type[row]).collect(),
    ))
}

/// C++ `fillContours`: expands absent points from three valid neighbors in
/// each direction using normalized cross-correlation and bead centering.
pub fn fill_contours(
    sfm: &mut SfmData,
    volume: &IoMrc,
    templates: &[Vec<f32>],
    template_sizes: &[usize],
    _debug_mode: bool,
) -> Result<(), FillContoursError> {
    let (contour_x, contour_y, reproj_x, reproj_y, marker_types) = match (
        &mut sfm.contour_x,
        &mut sfm.contour_y,
        &sfm.reproj_x,
        &sfm.reproj_y,
        &sfm.marker_type,
    ) {
        (Some(x), Some(y), Some(rx), Some(ry), Some(types)) => (x, y, rx, ry, types),
        _ => return Err(FillContoursError::InvalidData),
    };
    let rows = contour_x.num_trajectories;
    let columns = contour_x.num_frames;
    if contour_y.num_trajectories != rows
        || contour_y.num_frames != columns
        || reproj_x.scores.len() != rows * columns
        || reproj_y.scores.len() != rows * columns
        || marker_types.len() != rows
    {
        return Err(FillContoursError::InvalidData);
    }
    for row in 0..rows {
        let type_index =
            usize::try_from(marker_types[row]).map_err(|_| FillContoursError::InvalidTemplate)?;
        let size = *template_sizes
            .get(type_index)
            .ok_or(FillContoursError::InvalidTemplate)?;
        let template_values = templates
            .get(type_index)
            .ok_or(FillContoursError::InvalidTemplate)?;
        if size == 0 || template_values.len() != size * size {
            return Err(FillContoursError::InvalidTemplate);
        }
        let perfect = CvChannelMatrix::new(size, size, 1, template_values.clone())
            .map_err(|_| FillContoursError::InvalidTemplate)?;
        for direction in [1_isize, -1] {
            for offset in 0..columns.saturating_sub(3) {
                let frame = if direction > 0 {
                    offset + 3
                } else {
                    columns - 4 - offset
                };
                let position = row * columns + frame;
                if contour_x.scores[position] > 1.0e-6 {
                    continue;
                }
                let history: Vec<usize> = (1..=3)
                    .map(|offset| (frame as isize - direction * offset) as usize)
                    .collect();
                if history
                    .iter()
                    .any(|&previous| contour_x.scores[row * columns + previous] < 1.0e-6)
                {
                    continue;
                }
                let mut patch_values = vec![0.0; 4 * size * size];
                if !volume.read_mrc_patch(
                    frame as i32,
                    reproj_x.scores[position].ceil() as i32,
                    reproj_y.scores[position].ceil() as i32,
                    size as i32,
                    size as i32,
                    &mut patch_values,
                ) {
                    continue;
                }
                let patch = CvChannelMatrix::new(2 * size, 2 * size, 1, patch_values)
                    .map_err(|_| FillContoursError::InvalidData)?;
                let mut peaks = Vec::new();
                for previous in history {
                    let old = row * columns + previous;
                    let half = size / 2;
                    let mut values = vec![0.0; 4 * half * half];
                    if !volume.read_mrc_patch(
                        previous as i32,
                        contour_x.scores[old].ceil() as i32,
                        contour_y.scores[old].ceil() as i32,
                        half as i32,
                        half as i32,
                        &mut values,
                    ) {
                        continue;
                    }
                    let small = CvChannelMatrix::new(2 * half, 2 * half, 1, values)
                        .map_err(|_| FillContoursError::InvalidData)?;
                    if let Some(peak) = find_matching_translation(&patch, &small, marker_types[row])
                    {
                        peaks.push(peak);
                    }
                }
                if peaks.is_empty() {
                    continue;
                }
                let mean_x = peaks.iter().map(|peak| peak.x).sum::<f64>() / peaks.len() as f64;
                if peaks.iter().any(|peak| (peak.x - mean_x).abs() > 3.0) {
                    continue;
                }
                let score_sum: f64 = peaks.iter().map(|peak| peak.score).sum();
                if score_sum == 0.0 {
                    continue;
                }
                let mut xx = peaks.iter().map(|peak| peak.score * peak.x).sum::<f64>() / score_sum
                    - 0.5 * (patch.cols - size) as f64;
                let mut yy = peaks.iter().map(|peak| peak.score * peak.y).sum::<f64>() / score_sum
                    - 0.5 * (patch.rows - size) as f64;
                bead_center(&patch, &perfect, &mut xx, &mut yy, marker_types[row]);
                contour_x.scores[position] = yy + reproj_x.scores[position].ceil();
                contour_y.scores[position] = xx + reproj_y.scores[position].ceil();
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trajectory_distance_uses_source_median_and_absent_sentinel() {
        assert_eq!(
            distance_trajectory(&[1., 2., 0.], &[4., 5., 0.], &[2., 5., 3.], &[5., 7., 1.]),
            5.0
        );
        assert_eq!(distance_trajectory(&[0.], &[0.], &[2.], &[3.]), 1.0e20);
    }
    #[test]
    fn translation_finds_exact_embedded_template() {
        let image =
            CvChannelMatrix::new(3, 3, 1, vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
        let template = CvChannelMatrix::new(2, 2, 1, vec![4., 5., 7., 8.]).unwrap();
        let peak = find_matching_translation(&image, &template, 0).unwrap();
        assert_eq!((peak.x, peak.y), (1., 1.));
    }
}
