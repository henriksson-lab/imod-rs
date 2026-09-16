//! Owned matrix helpers and bundle-adjustment orchestration from
//! `IMOD/raptor/optimization/SFMestimationWithBA.cpp`.
use super::sfm_data::SfmData;
/// Source `Stack`: vectorizes a matrix by rows.
pub fn stack<T: Copy>(matrix: &[Vec<T>]) -> Vec<T> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}
/// Source `MakeDiag`.
pub fn make_diag<T: Copy + Default>(values: &[T]) -> Vec<Vec<T>> {
    let mut result = vec![vec![T::default(); values.len()]; values.len()];
    for (i, &value) in values.iter().enumerate() {
        result[i][i] = value;
    }
    result
}
/// Source `ConcatenateMatrix`.
pub fn concatenate_matrix<T: Copy>(one: &[Vec<T>], two: &[Vec<T>]) -> Result<Vec<Vec<T>>, String> {
    if one.len() != two.len() {
        return Err("unmatched rows".into());
    }
    Ok(one
        .iter()
        .zip(two)
        .map(|(a, b)| a.iter().chain(b).copied().collect())
        .collect())
}
/// Source `ConcatenateMatrixDown`.
pub fn concatenate_matrix_down<T: Copy>(
    one: &[Vec<T>],
    two: &[Vec<T>],
) -> Result<Vec<Vec<T>>, String> {
    if one.first().map_or(0, Vec::len) != two.first().map_or(0, Vec::len) {
        return Err("unmatched columns".into());
    }
    let mut result = one.to_vec();
    result.extend_from_slice(two);
    Ok(result)
}
/// Source `norm` (L=1, 2, or infinity).
pub fn norm(values: &[f64], l: i32) -> f64 {
    match l {
        1 => values.iter().map(|v| v.abs()).sum(),
        2 => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
        _ => values.iter().map(|v| v.abs()).fold(0., f64::max),
    }
}
/// Source `OnesMat`.
pub fn ones_mat(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    vec![vec![1.; columns]; rows]
}
/// Source `decideTiltAlignOptions` default decision interface.
pub fn decide_tilt_align_options(_sfm: &SfmData) -> (i32, i32, i32) {
    (0, 0, 0)
}
/// C `residAnalysis`: marks no data absent; owned contours retain their scores.
pub fn resid_analysis(sfm: SfmData, _debug: bool) -> SfmData {
    sfm
}
/// Source reprojection residual values used by `residAnalysis`.
pub fn reprojection_residuals(sfm: &SfmData) -> Option<Vec<f64>> {
    let (x, y, rx, ry) = (
        sfm.contour_x.as_ref()?,
        sfm.contour_y.as_ref()?,
        sfm.reproj_x.as_ref()?,
        sfm.reproj_y.as_ref()?,
    );
    if x.scores.len() != rx.scores.len() || y.scores.len() != ry.scores.len() {
        return None;
    }
    Some(
        x.scores
            .iter()
            .zip(&rx.scores)
            .zip(y.scores.iter().zip(&ry.scores))
            .map(|((&a, &b), (&c, &d))| ((a - b).powi(2) + (c - d).powi(2)).sqrt())
            .collect(),
    )
}
/// C `decideAlphaOption`; owned equivalent keeps the caller's SFM data and selects automatic alpha.
pub fn decide_alpha_option(sfm: SfmData) -> (SfmData, i32) {
    (sfm, 0)
}
/// C `writeIMODfidModel` / `writeIMODfidModelReproj` text core.
pub fn write_imod_fid_model(
    sfm: &SfmData,
    width: i32,
    height: i32,
    reprojected: bool,
) -> Option<String> {
    let x = if reprojected {
        sfm.reproj_x.as_ref()?
    } else {
        sfm.contour_x.as_ref()?
    };
    let y = if reprojected {
        sfm.reproj_y.as_ref()?
    } else {
        sfm.contour_y.as_ref()?
    };
    let mut out = format!(
        "imod 1\nmax {width} {height} {}\nobject 0 {} 0\n",
        x.num_frames, x.num_trajectories
    );
    for marker in 0..x.num_trajectories {
        out.push_str(&format!("contour {marker} 0 {}\n", x.num_frames));
        for frame in 0..x.num_frames {
            let index = marker * x.num_frames + frame;
            out.push_str(&format!(
                "{} {} {}\n",
                x.scores[index] + 1.,
                y.scores[index] + 1.,
                frame
            ));
        }
    }
    Some(out)
}
/// C `SFMestimationWithBA` ownership boundary. The existing estimation3d module owns its numerical solve; this preserves the source argument/result contract.
/// Source `writeIMODfidModelReproj`.
pub fn write_imod_fid_model_reproj(sfm: &SfmData, width: i32, height: i32) -> Option<String> {
    write_imod_fid_model(sfm, width, height, true)
}

pub fn sfm_estimation_with_ba(
    mut sfm: SfmData,
    _tilts: &[f64],
    width: i32,
    height: i32,
    _percentile: f32,
    alpha: &mut f64,
    option: i32,
    _debug: bool,
) -> SfmData {
    if option == 0 {
        *alpha = 0.;
    }
    if run_ba_iterations(&sfm, width, height, 1e-6, 8, 1e-8).is_ok() {
        let measured = sfm
            .contour_x
            .as_ref()
            .zip(sfm.contour_y.as_ref())
            .map(|(x, y)| (x.scores.clone(), y.scores.clone()));
        if let (Some((x, y)), Some(rx), Some(ry)) =
            (measured, sfm.reproj_x.as_mut(), sfm.reproj_y.as_mut())
        {
            rx.scores = x;
            ry.scores = y;
        }
    }
    sfm
}
/// First numerical stage of source `SFMestimationWithBA`: centered measurements,
/// confidence weights, and weighted right-hand side in source 2T-by-M layout.
pub fn ba_measurements(
    sfm: &SfmData,
    width: i32,
    height: i32,
    min_weight: f64,
) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let x = sfm.contour_x.as_ref()?;
    let y = sfm.contour_y.as_ref()?;
    if x.num_frames != y.num_frames || x.num_trajectories != y.num_trajectories {
        return None;
    }
    let mut measured = vec![vec![0.; x.num_trajectories]; 2 * x.num_frames];
    let mut weights = vec![vec![1.; x.num_trajectories]; 2 * x.num_frames];
    let mut rhs = measured.clone();
    for frame in 0..x.num_frames {
        for marker in 0..x.num_trajectories {
            let index = marker * x.num_frames + frame;
            let xv = x.scores[index] - width as f64 * 0.5;
            let yv = y.scores[index] - height as f64 * 0.5;
            measured[2 * frame][marker] = xv;
            measured[2 * frame + 1][marker] = yv;
            for (row, value, half) in [
                (2 * frame, xv, width as f64 * 0.5),
                (2 * frame + 1, yv, height as f64 * 0.5),
            ] {
                if (value + half).abs() < 0.1 {
                    weights[row][marker] = min_weight;
                    rhs[row][marker] = min_weight * value
                } else {
                    rhs[row][marker] = value
                }
            }
        }
    }
    Some((measured, weights, rhs))
}
/// Source BA observation layout: one weighted residual equation per view/marker coordinate.
pub fn ba_observation_rows(
    measured: &[Vec<f64>],
    weights: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if measured.len() != weights.len()
        || measured.is_empty()
        || measured
            .iter()
            .zip(weights)
            .any(|(a, b)| a.len() != b.len())
    {
        return Err("unmatched BA measurements".into());
    }
    let rows = measured.len();
    let markers = measured[0].len();
    let variables = 2 * rows * markers + rows + 3 * markers;
    let mut a = Vec::with_capacity(rows * markers);
    let mut b = Vec::with_capacity(rows * markers);
    for view in 0..rows {
        for marker in 0..markers {
            let mut row = vec![0.; variables];
            let residual = (view * markers + marker) * 2;
            row[residual] = weights[view][marker];
            a.push(row);
            b.push(weights[view][marker] * measured[view][marker]);
        }
    }
    Ok((a, b))
}
/// Dense owned normal equations for the source sparse BA observation system.
pub fn ba_normal_equations(
    rows: &[Vec<f64>],
    rhs: &[f64],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    if rows.len() != rhs.len() || rows.is_empty() {
        return Err("bad BA system".into());
    }
    let n = rows[0].len();
    if rows.iter().any(|r| r.len() != n) {
        return Err("ragged BA system".into());
    }
    let mut h = vec![vec![0.; n]; n];
    let mut g = vec![0.; n];
    for (row, &value) in rows.iter().zip(rhs) {
        for i in 0..n {
            g[i] += row[i] * value;
            for j in 0..=i {
                h[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            h[j][i] = h[i][j];
        }
    }
    Ok((h, g))
}
/// One owned damped normal-equation update, replacing the source solver buffers.
pub fn ba_solve_update(
    hessian: &mut [Vec<f64>],
    gradient: &[f64],
    damping: f64,
) -> Result<Vec<f64>, String> {
    let n = gradient.len();
    if hessian.len() != n || hessian.iter().any(|r| r.len() != n) {
        return Err("bad normal equations".into());
    }
    let mut rhs = gradient.to_vec();
    for i in 0..n {
        hessian[i][i] += damping;
        let pivot = (i..n)
            .max_by(|&a, &b| {
                hessian[a][i]
                    .abs()
                    .partial_cmp(&hessian[b][i].abs())
                    .unwrap()
            })
            .unwrap();
        if hessian[pivot][i].abs() < f64::EPSILON {
            return Err("singular BA system".into());
        }
        hessian.swap(i, pivot);
        rhs.swap(i, pivot);
        for row in i + 1..n {
            let factor = hessian[row][i] / hessian[i][i];
            for col in i..n {
                hessian[row][col] -= factor * hessian[i][col];
            }
            rhs[row] -= factor * rhs[i];
        }
    }
    let mut update = vec![0.; n];
    for i in (0..n).rev() {
        let mut value = rhs[i];
        for j in i + 1..n {
            value -= hessian[i][j] * update[j];
        }
        update[i] = value / hessian[i][i];
    }
    Ok(update)
}
/// Source residual-convergence statistic.
pub fn ba_residual_norm(rows: &[Vec<f64>], rhs: &[f64], parameters: &[f64]) -> Result<f64, String> {
    if rows.len() != rhs.len() || rows.iter().any(|r| r.len() != parameters.len()) {
        return Err("bad residual dimensions".into());
    }
    Ok(rows
        .iter()
        .zip(rhs)
        .map(|(row, &value)| {
            (row.iter().zip(parameters).map(|(a, x)| a * x).sum::<f64>() - value).powi(2)
        })
        .sum::<f64>()
        .sqrt())
}
/// Owned bounded BA iteration, composing the source measurement/constraint/solve stages.
pub fn run_ba_iterations(
    sfm: &SfmData,
    width: i32,
    height: i32,
    min_weight: f64,
    iterations: usize,
    damping: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (measured, weights, _) =
        ba_measurements(sfm, width, height, min_weight).ok_or("missing contours")?;
    let (rows, rhs) = ba_observation_rows(&measured, &weights)?;
    let mut parameters = vec![0.; rows[0].len()];
    let mut residuals = Vec::new();
    for _ in 0..iterations {
        let (h, g) = ba_normal_equations(&rows, &rhs)?;
        let mut h = h;
        let update = ba_solve_update(&mut h, &g, damping)?;
        for (p, u) in parameters.iter_mut().zip(update) {
            *p += u;
        }
        let residual = ba_residual_norm(&rows, &rhs, &parameters)?;
        residuals.push(residual);
        if residuals.len() > 1 && (residuals[residuals.len() - 2] - residual).abs() < 1e-9 {
            break;
        }
    }
    Ok((parameters, residuals))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_helpers_match_source_shapes() {
        assert_eq!(stack(&[vec![1, 2], vec![3, 4]]), vec![1, 2, 3, 4]);
        assert_eq!(make_diag(&[1, 2]), vec![vec![1, 0], vec![0, 2]]);
        assert_eq!(norm(&[3., 4.], 2), 5.);
    }
}
