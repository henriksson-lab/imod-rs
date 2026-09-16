//! Safe translation of `IMOD/raptor/optimization/estimation3d.{h,cpp}`.

use super::contour::Contour;
use super::estimation3d_data::Estimation3dData;
use super::prob_data::{ProbData, SparseMatrix};
use super::std_qp_data::{Matrix, StdQpData};
use crate::imod::raptor::main_classes::constants::{MIN_WEIGHT, PI};
use std::collections::BTreeMap;

/// C++ `CreateRandMat`, with caller-provided randomness instead of C's global
/// `rand` state.
pub fn create_rand_mat(
    rows: usize,
    columns: usize,
    scale: f64,
    shift: f64,
    random: &mut impl FnMut() -> f64,
) -> Matrix<f64> {
    Matrix::new(
        rows,
        columns,
        (0..rows * columns)
            .map(|_| scale * random() + shift)
            .collect(),
    )
}

/// C++ `repMat`.
pub fn rep_mat(
    matrix: &Matrix<f64>,
    row_repetitions: usize,
    column_repetitions: usize,
) -> Matrix<f64> {
    let rows = matrix.rows * row_repetitions;
    let columns = matrix.columns * column_repetitions;
    let mut values = vec![0.0; rows * columns];
    for repeated_row in 0..row_repetitions {
        for repeated_column in 0..column_repetitions {
            for source_row in 0..matrix.rows {
                for source_column in 0..matrix.columns {
                    values[(repeated_row * matrix.rows + source_row) * columns
                        + repeated_column * matrix.columns
                        + source_column] =
                        matrix.values[source_row * matrix.columns + source_column];
                }
            }
        }
    }
    Matrix::new(rows, columns, values)
}

/// C++ `estimation3D`.
///
/// `solve` is the native replacement for the not-yet-translated `std_qp`
/// source unit.  It receives the same fully populated problem state that the
/// C++ function passes to `std_qp`; its answer is then processed exactly here.
/// `None` represents malformed dimensions or missing source allocations.
pub fn estimation_3d(
    contour_x: &Contour,
    contour_y: &Contour,
    alpha: f64,
    tilt_angles: &[f64],
    _width: i32,
    _height: i32,
    percentile: f64,
    probability: &mut ProbData,
    weights: &Matrix<f64>,
    measured: &Matrix<f64>,
    random: &mut impl FnMut() -> f64,
    solve: impl FnOnce(&ProbData, usize, usize) -> StdQpData<f64>,
) -> Option<Estimation3dData> {
    let views = contour_x.num_frames;
    let markers = contour_x.num_trajectories;
    if contour_y.num_frames != views
        || contour_y.num_trajectories != markers
        || tilt_angles.len() < views
        || weights.rows != 2 * views
        || weights.columns != markers
        || measured.rows != 2 * views
        || measured.columns != markers
    {
        return None;
    }

    let mut estimate = Estimation3dData::with_dimensions(views, markers);
    let g = estimate.g.as_mut()?;
    let alpha_radians = alpha * PI / 180.0;
    for view in 0..views {
        let tilt_radians = tilt_angles[view] * PI / 180.0;
        let cosine_alpha = alpha_radians.cos();
        let sine_alpha = alpha_radians.sin();
        let cosine_tilt = tilt_radians.cos();
        let sine_tilt = tilt_radians.sin();
        g.values[2 * view * 3] = cosine_alpha;
        g.values[2 * view * 3 + 1] = -sine_alpha * cosine_tilt;
        g.values[2 * view * 3 + 2] = -sine_alpha * sine_tilt;
        g.values[(2 * view + 1) * 3] = sine_alpha;
        g.values[(2 * view + 1) * 3 + 1] = cosine_alpha * cosine_tilt;
        g.values[(2 * view + 1) * 3 + 2] = cosine_alpha * sine_tilt;
    }

    let a = probability.a.as_mut()?;
    if a.nz != -1
        || a.column_or_triplet_offsets.len() != a.columns + 1
        || a.columns < 2 * views + 3 * markers
    {
        return None;
    }
    for column in 2 * views..2 * views + 3 * markers {
        let g_column = (column - 2 * views) % 3;
        for entry in a.column_or_triplet_offsets[column]..a.column_or_triplet_offsets[column + 1] {
            a.values[entry] *= g.values[(a.row_indices[entry] % (2 * views)) * 3 + g_column];
        }
    }

    let x0_length = 4 * views * markers + 2 * views + 3 * markers;
    probability.x0 = Some(create_rand_mat(x0_length, 1, 2.0, -1.0, random));
    let bounds = probability.buc.as_ref()?;
    if bounds.columns != 1 || bounds.rows < 2 * views * markers {
        return None;
    }
    let x0 = probability.x0.as_mut()?;
    for value in &mut x0.values[..2 * views] {
        *value *= 10.0;
    }
    for value in
        &mut x0.values[2 * views + 3 * markers..2 * views + 3 * markers + 2 * views * markers]
    {
        *value = random() * 3.0 + 100.0;
    }
    for (index, value) in x0.values[2 * views * markers + 3 * markers + 2 * views..]
        .iter_mut()
        .enumerate()
    {
        *value = bounds.values[index] + 2.0 * random() - 1.0;
    }

    let q = probability.q.as_ref()?;
    if q.nz != -1 || q.column_or_triplet_offsets.len() != q.columns + 1 || q.rows != q.columns {
        return None;
    }
    let mut columns: Vec<BTreeMap<usize, f64>> = (0..q.columns).map(|_| BTreeMap::new()).collect();
    for (column, destination) in columns.iter_mut().enumerate() {
        for entry in q.column_or_triplet_offsets[column]..q.column_or_triplet_offsets[column + 1] {
            *destination.entry(q.row_indices[entry]).or_default() += q.values[entry];
        }
    }
    for diagonal in 0..2 * views {
        *columns[diagonal].entry(diagonal).or_default() += 1.0e-6;
    }
    let mut offsets = Vec::with_capacity(q.columns + 1);
    let mut rows = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for column in columns {
        for (row, value) in column {
            rows.push(row);
            values.push(value);
        }
        offsets.push(rows.len());
    }
    probability.q = Some(SparseMatrix::new(
        q.rows,
        q.columns,
        rows.len(),
        -1,
        offsets,
        rows,
        values,
    ));

    let result = solve(probability, markers, views);
    if result.num_itrs > 200 {
        estimate.resid_mean_perc = 1.0e6;
        estimate.resid_mean = 1.0e6;
        return Some(estimate);
    }
    let answer = result.answer_mat?;
    if answer.columns != 1 || answer.rows < 2 * views + 3 * markers {
        return None;
    }
    let t = estimate.t.as_mut()?;
    t.values.copy_from_slice(&answer.values[..2 * views]);
    let p = estimate.p.as_mut()?;
    for marker in 0..markers {
        p.values[marker] = answer.values[2 * views + 3 * marker];
        p.values[markers + marker] = answer.values[2 * views + 3 * marker + 1];
        p.values[2 * markers + marker] = answer.values[2 * views + 3 * marker + 2];
    }

    let mut residuals = Vec::new();
    for view in 0..views {
        for marker in 0..markers {
            if weights.values[(2 * view) * markers + marker] > MIN_WEIGHT * 1.1 {
                let projected_x = g.values[2 * view * 3] * p.values[marker]
                    + g.values[2 * view * 3 + 1] * p.values[markers + marker]
                    + g.values[2 * view * 3 + 2] * p.values[2 * markers + marker]
                    + t.values[2 * view];
                let projected_y = g.values[(2 * view + 1) * 3] * p.values[marker]
                    + g.values[(2 * view + 1) * 3 + 1] * p.values[markers + marker]
                    + g.values[(2 * view + 1) * 3 + 2] * p.values[2 * markers + marker]
                    + t.values[2 * view + 1];
                let dx = measured.values[(2 * view) * markers + marker] - projected_x;
                let dy = measured.values[(2 * view + 1) * markers + marker] - projected_y;
                residuals.push((dx * dx + dy * dy).sqrt());
            }
        }
    }
    residuals.sort_by(f64::total_cmp);
    if residuals.is_empty() {
        return Some(estimate);
    }
    let percentile_count =
        ((residuals.len() as f64 * percentile).ceil() as usize).clamp(1, residuals.len());
    let sum: f64 = residuals.iter().sum();
    estimate.resid_mean_perc =
        residuals[..percentile_count].iter().sum::<f64>() / percentile_count as f64;
    estimate.resid_mean = sum / residuals.len() as f64;
    Some(estimate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_and_repeat_matrix_functions_preserve_source_layout() {
        let mut random = || 0.25;
        assert_eq!(
            create_rand_mat(1, 2, 2.0, -1.0, &mut random).values,
            vec![-0.5; 2]
        );
        assert_eq!(
            rep_mat(&Matrix::new(1, 2, vec![1.0, 2.0]), 2, 2),
            Matrix::new(2, 4, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        );
    }

    #[test]
    fn projection_fit_builds_g_regularizes_q_and_calculates_residuals() {
        let empty = || SparseMatrix::new(9, 9, 0, -1, vec![0; 10], vec![], vec![]);
        let mut probability = ProbData {
            buc: Some(Matrix::new(2, 1, vec![0.0; 2])),
            a: Some(SparseMatrix::new(2, 5, 0, -1, vec![0; 6], vec![], vec![])),
            c: Some(empty()),
            q: Some(empty()),
            x0: None,
        };
        let mut random = || 0.5;
        let estimate = estimation_3d(
            &Contour::with_dimensions(1, 1, 1),
            &Contour::with_dimensions(1, 1, 2),
            0.0,
            &[0.0],
            0,
            0,
            0.7,
            &mut probability,
            &Matrix::new(2, 1, vec![1.0, 1.0]),
            &Matrix::new(2, 1, vec![12.0, 25.0]),
            &mut random,
            |problem, markers, views| {
                assert_eq!((markers, views), (1, 1));
                assert_eq!(problem.x0.as_ref().unwrap().rows, 9);
                assert_eq!(problem.q.as_ref().unwrap().values, vec![1.0e-6, 1.0e-6]);
                StdQpData {
                    answer_mat: Some(Matrix::new(
                        9,
                        1,
                        vec![10.0, 20.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    )),
                    num_itrs: 1,
                    exit_flag: 0,
                    gap: 0.0,
                }
            },
        )
        .unwrap();
        assert_eq!(
            estimate.g.unwrap().values,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(estimate.p.unwrap().values, vec![1.0, 2.0, 3.0]);
        assert_eq!(
            (estimate.resid_mean_perc, estimate.resid_mean),
            (10.0_f64.sqrt(), 10.0_f64.sqrt())
        );
    }

    #[test]
    fn excessive_qp_iterations_preserve_source_failure_sentinel() {
        let empty = || SparseMatrix::new(9, 9, 0, -1, vec![0; 10], vec![], vec![]);
        let mut probability = ProbData {
            buc: Some(Matrix::new(2, 1, vec![0.0; 2])),
            a: Some(SparseMatrix::new(2, 5, 0, -1, vec![0; 6], vec![], vec![])),
            c: Some(empty()),
            q: Some(empty()),
            x0: None,
        };
        let mut random = || 0.5;
        let estimate = estimation_3d(
            &Contour::with_dimensions(1, 1, 1),
            &Contour::with_dimensions(1, 1, 2),
            0.0,
            &[0.0],
            0,
            0,
            0.7,
            &mut probability,
            &Matrix::new(2, 1, vec![1.0; 2]),
            &Matrix::new(2, 1, vec![0.0; 2]),
            &mut random,
            |_, _, _| StdQpData {
                answer_mat: None,
                num_itrs: 201,
                exit_flag: 0,
                gap: 0.0,
            },
        )
        .unwrap();
        assert_eq!(
            (estimate.resid_mean_perc, estimate.resid_mean),
            (1.0e6, 1.0e6)
        );
    }
}
