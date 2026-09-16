//! Owned translation of `IMOD/raptor/correspondence/correspondence.{h,cpp}`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::imod::raptor::main_classes::constants::INITIAL_PAIRWISE_SCORE;
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::main_classes::pair_correspondence::PairCorrespondence;
use crate::imod::raptor::main_classes::point2d::Point2d;
use crate::imod::raptor::main_classes::stat::{average, max, mean_and_variance};

/// Failure while materializing or consuming the external MarkersCorrespond protocol.
#[derive(Debug)]
pub enum CorrespondenceError {
    Io(std::io::Error),
    InvalidInput(&'static str),
    InferenceFailed,
    MalformedBeliefs,
}

impl From<std::io::Error> for CorrespondenceError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Parameters following the two `frame *` and output vector arguments of C++
/// `correspondenceRegions`.  Fields retained solely for ABI compatibility are
/// marked by the leading underscore at their use site.
#[derive(Clone, Debug)]
pub struct CorrespondenceParameters {
    pub n_template: i32,
    pub max_previous_markers: usize,
    pub max_next_markers: usize,
    pub max_cliques: i32,
    pub dmax: i32,
    pub q1: i32,
    pub w1: i32,
    pub max_pairwise_table: i32,
    pub min_pairwise_table: i32,
    pub index: String,
    pub diameter: i32,
    pub output_dir: PathBuf,
    pub bin_path: PathBuf,
}

/// C++ `singlePotentials`, retaining its normalized-patch and texture scoring.
pub fn single_potentials<T, P>(
    potentials: &mut [f32],
    frame1: &Frame<T, P>,
    frame2: &Frame<T, P>,
    window: i32,
    dmax: i32,
    mode: i32,
    option: i32,
    volume: &IoMrc,
    reference_indices: &[usize],
    next_count: usize,
) -> Result<(), CorrespondenceError> {
    if window < 0
        || frame1.width <= 0
        || frame1.height <= 0
        || frame2.width <= 0
        || frame2.height <= 0
    {
        return Err(CorrespondenceError::InvalidInput(
            "negative window or empty frame",
        ));
    }
    let previous_count = reference_indices.len();
    if potentials.len()
        != previous_count
            .checked_mul(next_count)
            .ok_or(CorrespondenceError::InvalidInput(
                "potential dimensions overflow",
            ))?
    {
        return Err(CorrespondenceError::InvalidInput(
            "potential dimensions do not match marker counts",
        ));
    }
    if next_count > frame2.points.len()
        || reference_indices
            .iter()
            .any(|&index| index >= frame1.points.len())
    {
        return Err(CorrespondenceError::InvalidInput(
            "marker index is out of range",
        ));
    }
    if previous_count == 0 || next_count == 0 {
        return Ok(());
    }
    let image_len1 = (frame1.width as usize)
        .checked_mul(frame1.height as usize)
        .ok_or(CorrespondenceError::InvalidInput("frame size overflow"))?;
    let image_len2 = (frame2.width as usize)
        .checked_mul(frame2.height as usize)
        .ok_or(CorrespondenceError::InvalidInput("frame size overflow"))?;
    let mut image1 = vec![0.0; image_len1];
    let mut image2 = vec![0.0; image_len2];
    if !volume.read_mrc_slice_float(frame1.frame_id, &mut image1)
        || !volume.read_mrc_slice_float(frame2.frame_id, &mut image2)
    {
        return Err(CorrespondenceError::InvalidInput(
            "frame does not exist in MRC volume",
        ));
    }
    let (_, variance1) = mean_and_variance(&image1);
    let (_, variance2) = mean_and_variance(&image2);
    let side = (2 * window + 1) as usize;
    let patch_len = side
        .checked_mul(side)
        .ok_or(CorrespondenceError::InvalidInput("patch size overflow"))?;
    let threshold = 0.01_f32;
    potentials.fill(threshold);
    let mut patches1 = vec![Vec::new(); previous_count];
    let mut patches2 = vec![Vec::new(); next_count];
    let mut variances1 = vec![0.0; previous_count];
    let mut variances2 = vec![0.0; next_count];
    let mut norms1 = vec![0.0; previous_count];
    let mut norms2 = vec![0.0; next_count];
    let patch = |image: &[f32], width: i32, height: i32, point: &Point2d<P>| -> Option<Vec<f32>> {
        let x = (point.x - window as f32) as i32;
        let y = (point.y - window as f32) as i32;
        if (point.x - (window as f32) < 1.5)
            || (point.y - (window as f32) < 1.5)
            || (point.x + (window as f32) > width as f32 - 1.0)
            || (point.y + (window as f32) > height as f32 - 1.0)
        {
            return None;
        }
        let mut values = Vec::with_capacity(patch_len);
        for yy in 0..side {
            for xx in 0..side {
                values.push(image[(y as usize + yy) * width as usize + x as usize + xx]);
            }
        }
        Some(values)
    };
    for (row, &point_index) in reference_indices.iter().enumerate() {
        if let Some(mut values) = patch(
            &image1,
            frame1.width,
            frame1.height,
            &frame1.points[point_index],
        ) {
            let (mean, variance) = mean_and_variance(&values);
            for value in &mut values {
                *value -= mean;
            }
            variances1[row] = variance;
            norms1[row] = (variance * (patch_len - 1) as f32).sqrt();
            patches1[row] = values;
        }
    }
    for column in 0..next_count {
        if let Some(mut values) =
            patch(&image2, frame2.width, frame2.height, &frame2.points[column])
        {
            let (mean, variance) = mean_and_variance(&values);
            for value in &mut values {
                *value -= mean;
            }
            variances2[column] = variance;
            norms2[column] = (variance * (patch_len - 1) as f32).sqrt();
            patches2[column] = values;
        }
    }
    for row in 0..previous_count {
        if patches1[row].is_empty() {
            continue;
        }
        for column in 0..next_count {
            if patches2[column].is_empty() {
                continue;
            }
            let first = &frame1.points[reference_indices[row]];
            let second = &frame2.points[column];
            let dx = first.x - second.x;
            let dy = first.y - second.y;
            let distance = (dx * dx + dy * dy).sqrt();
            let limit = if option == 1 || option == 3 {
                2.0 * dmax as f32
            } else {
                dmax as f32
            };
            if distance > limit || first.marker_type != second.marker_type {
                continue;
            }
            if mode == 1 {
                return Err(CorrespondenceError::InvalidInput(
                    "mutual-information potentials are not implemented by the source",
                ));
            }
            let dot: f32 = patches1[row]
                .iter()
                .zip(&patches2[column])
                .map(|(left, right)| left * right)
                .sum();
            let score = dot / (norms1[row] * norms2[column]);
            potentials[row * next_count + column] = score.max(1.0e-10);
        }
    }
    let texture_baseline = if max(potentials) > threshold {
        threshold
    } else {
        let accepted: Vec<f32> = potentials
            .iter()
            .copied()
            .filter(|value| *value > threshold)
            .collect();
        if accepted.is_empty() {
            threshold
        } else {
            average(&accepted) as f32
        }
    };
    for row in 0..previous_count {
        if average(&potentials[row * next_count..(row + 1) * next_count]) == threshold as f64 {
            continue;
        }
        for column in 0..next_count {
            let position = row * next_count + column;
            if potentials[position] == threshold {
                continue;
            }
            let first = &frame1.points[reference_indices[row]];
            let second = &frame2.points[column];
            let dx = first.x - second.x;
            let dy = first.y - second.y;
            let distance = (dx * dx + dy * dy).sqrt();
            let limit = if option == 1 || option == 3 {
                2.0 * dmax as f32
            } else {
                dmax as f32
            };
            if distance > limit {
                potentials[position] = 1.0e-10;
                continue;
            }
            let texture = 0.5 * (variances1[row] / variance1 + variances2[column] / variance2);
            potentials[position] += (1.0 - texture) * texture_baseline;
            if option == 1 {
                potentials[position] *= (-(distance / dmax as f32).powi(2)).exp();
            }
        }
    }
    let maximum = max(potentials);
    if maximum != 0.0 {
        for value in potentials {
            *value /= maximum;
        }
    }
    Ok(())
}

/// C++ `correspondenceRegions`: writes the solver protocol, executes the owned
/// process invocation, reads beliefs, and records accepted marker pairs.
pub fn correspondence_regions<T>(
    frame1: &mut Frame<T, PairCorrespondence>,
    frame2: &mut Frame<T, PairCorrespondence>,
    correspondences: &mut Vec<PairCorrespondence>,
    parameters: &CorrespondenceParameters,
    volume: &IoMrc,
    previous_pair: Option<&[PairCorrespondence]>,
) -> Result<bool, CorrespondenceError> {
    let _ = (
        parameters.n_template,
        parameters.q1,
        parameters.w1,
        parameters.diameter,
        previous_pair,
    );
    let frame1_count = frame1.points.len().min(parameters.max_previous_markers);
    let frame2_count = frame2.points.len().min(parameters.max_next_markers);
    let mut reference_indices: Vec<usize> = if previous_pair.is_none() {
        (0..frame1_count).collect()
    } else {
        let mut indices: Vec<usize> = frame1
            .points
            .iter()
            .enumerate()
            .filter_map(|(index, point)| point.used_pair_correspondence.then_some(index))
            .take(parameters.max_previous_markers)
            .collect();
        if indices.len() < parameters.max_previous_markers {
            indices.extend(
                frame1
                    .points
                    .iter()
                    .enumerate()
                    .filter_map(|(index, point)| (!point.used_pair_correspondence).then_some(index))
                    .take(parameters.max_previous_markers - indices.len()),
            );
        }
        indices
    };
    reference_indices.truncate(frame1_count);
    let previous_count = reference_indices.len();
    if previous_count == 0 || frame2_count == 0 {
        return Ok(false);
    }
    let temp = parameters.output_dir.join("temp");
    if !temp.is_dir() {
        return Err(CorrespondenceError::InvalidInput(
            "output temp directory does not exist",
        ));
    }
    let protocol_name = format!("{}FFcorrespondence", parameters.index);
    let mut distance =
        ((frame1.width as f64).powi(2) + (frame1.height as f64).powi(2)).sqrt() + 50.0;
    let mut cliques = parameters.max_cliques + 10;
    while cliques > parameters.max_cliques {
        distance -= 50.0;
        cliques = 0;
        for first in 0..previous_count.saturating_sub(1) {
            for second in first + 1..previous_count {
                let a = &frame1.points[reference_indices[first]];
                let b = &frame1.points[reference_indices[second]];
                if ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() < distance as f32 {
                    cliques += 1;
                }
            }
        }
    }
    fs::write(
        temp.join(format!("{protocol_name}.cfg")),
        format!(
            "{}\n 3\n 8\n 7\n {}\n 0.0001\n 5\n {}\n {}\n {}\n{}",
            parameters.max_previous_markers,
            distance,
            if previous_count <= 8 {
                previous_count - 1
            } else {
                8
            },
            parameters.max_pairwise_table,
            parameters.min_pairwise_table,
            (0..previous_count)
                .map(|_| format!("{}\n{}", -parameters.dmax, parameters.dmax))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )?;
    let coordinates = |points: Vec<&Point2d<PairCorrespondence>>| -> String {
        format!(
            "{}\n{}\n{}\n",
            points.len(),
            points
                .iter()
                .map(|point| point.x.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            points
                .iter()
                .map(|point| point.y.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    let references: Vec<_> = reference_indices
        .iter()
        .map(|&index| &frame1.points[index])
        .collect();
    fs::write(
        temp.join(format!("{protocol_name}.mrkr")),
        coordinates(references.clone()),
    )?;
    fs::write(
        temp.join(format!("{protocol_name}.mrkr_trans")),
        coordinates(references),
    )?;
    fs::write(
        temp.join(format!("{protocol_name}.cand")),
        coordinates(frame2.points[..frame2_count].iter().collect()),
    )?;
    let mut potentials = vec![0.0; previous_count * frame2_count];
    single_potentials(
        &mut potentials,
        frame1,
        frame2,
        15,
        parameters.dmax,
        2,
        1,
        volume,
        &reference_indices,
        frame2_count,
    )?;
    let potential_rows = potentials
        .chunks(frame2_count)
        .map(|row| {
            row.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(
        temp.join(format!("{protocol_name}.sp")),
        format!("{}\n{}\n", frame2_count, potential_rows),
    )?;
    let program = parameters.bin_path.join("MarkersCorrespond");
    let status = Command::new(program)
        .arg(temp.join(&protocol_name))
        .status()?;
    if !status.success() {
        return Err(CorrespondenceError::InferenceFailed);
    }
    let belief = fs::read_to_string(temp.join(format!("{protocol_name}_final_beliefs_scr.m")))?;
    let mut rows = belief.lines();
    rows.next();
    let beliefs: Result<Vec<Vec<f64>>, _> = rows
        .take(previous_count)
        .map(|row| {
            row.split_whitespace()
                .take(frame2_count)
                .map(str::parse::<f64>)
                .collect()
        })
        .collect();
    let beliefs = beliefs.map_err(|_| CorrespondenceError::MalformedBeliefs)?;
    if beliefs.len() != previous_count || beliefs.iter().any(|row| row.len() != frame2_count) {
        return Err(CorrespondenceError::MalformedBeliefs);
    }
    let mut accepted = Vec::new();
    for (row, belief_row) in beliefs.iter().enumerate() {
        let (best_index, &best) = belief_row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Less)
            })
            .ok_or(CorrespondenceError::MalformedBeliefs)?;
        let runner_up = belief_row
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (index != best_index).then_some(*value))
            .fold(-1.0e37, f64::max);
        if frame1.points[reference_indices[row]].pairwise_score == INITIAL_PAIRWISE_SCORE {
            frame1.points[reference_indices[row]].pairwise_score = best;
        }
        if best - runner_up > 1.0 && best > -2.3026 {
            accepted.push((reference_indices[row], best_index, best));
        }
    }
    for (first, second, score) in &accepted {
        let pair = PairCorrespondence::new(*first as i32, *second as i32, (), (), *score);
        frame1.points[*first].index0.push(pair.clone());
        frame1.points[*first].used_pair_correspondence = true;
        frame2.points[*second].pairwise_score = *score;
        frame2.points[*second].used_pair_correspondence = true;
        frame2.points[*second].index1.push(pair.clone());
        correspondences.push(pair);
    }
    Ok(!(accepted.len() < 4 && (frame1.frame_id - frame2.frame_id).abs() < 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::raptor::main_classes::io_mrc_vol::MrcVolume;

    #[test]
    fn single_potentials_prefers_identical_local_patch() {
        let image: Vec<f32> = (0..25).map(|value| value as f32).collect();
        let volume = IoMrc::new(
            MrcVolume::Float([image.clone(), image].concat()),
            5,
            5,
            2,
            2,
        );
        let point = Point2d::<PairCorrespondence>::with_marker_type(2, 2, 0, 1);
        let first = Frame::<(), PairCorrespondence> {
            points: vec![point.clone()],
            frame_id: 0,
            width: 5,
            height: 5,
            ..Frame::default()
        };
        let second = Frame::<(), PairCorrespondence> {
            points: vec![Point2d {
                frame_id: 1,
                ..point
            }],
            frame_id: 1,
            width: 5,
            height: 5,
            ..Frame::default()
        };
        let mut values = [0.0];
        single_potentials(&mut values, &first, &second, 1, 10, 2, 1, &volume, &[0], 1).unwrap();
        assert_eq!(values, [1.0]);
    }
}
