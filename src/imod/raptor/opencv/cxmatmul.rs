//! Matrix product and vector-transform functions from `cxmatmul.cpp`.

use super::cxutils::CvMatrix;

pub const CV_GEMM_A_T: i32 = 1;
pub const CV_GEMM_B_T: i32 = 2;
pub const CV_GEMM_C_T: i32 = 4;
pub const CV_COVAR_NORMAL: i32 = 1;
pub const CV_COVAR_USE_AVG: i32 = 2;
pub const CV_COVAR_SCALE: i32 = 4;
pub const CV_COVAR_ROWS: i32 = 8;
pub const CV_COVAR_COLS: i32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvMatMulError {
    BadArgument,
    UnmatchedSizes,
}

/// Owned `cvGEMM`: `D = alpha * op(A) * op(B) + beta * op(C)`.
pub fn cv_gemm(
    a: &CvMatrix<f64>,
    b: &CvMatrix<f64>,
    alpha: f64,
    c: Option<&CvMatrix<f64>>,
    beta: f64,
    d: &mut CvMatrix<f64>,
    flags: i32,
) -> Result<(), CvMatMulError> {
    let (ar, ac) = if flags & CV_GEMM_A_T != 0 {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let (br, bc) = if flags & CV_GEMM_B_T != 0 {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    if ac != br || d.rows != ar || d.cols != bc {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    if let Some(c) = c {
        let (cr, cc) = if flags & CV_GEMM_C_T != 0 {
            (c.cols, c.rows)
        } else {
            (c.rows, c.cols)
        };
        if cr != ar || cc != bc {
            return Err(CvMatMulError::UnmatchedSizes);
        }
    }
    for row in 0..ar {
        for column in 0..bc {
            let mut value = 0.0;
            for inner in 0..ac {
                let av = if flags & CV_GEMM_A_T != 0 {
                    a.data[inner * a.cols + row]
                } else {
                    a.data[row * a.cols + inner]
                };
                let bv = if flags & CV_GEMM_B_T != 0 {
                    b.data[column * b.cols + inner]
                } else {
                    b.data[inner * b.cols + column]
                };
                value += av * bv;
            }
            if let Some(c) = c {
                value = alpha * value
                    + beta
                        * if flags & CV_GEMM_C_T != 0 {
                            c.data[column * c.cols + row]
                        } else {
                            c.data[row * c.cols + column]
                        };
            } else {
                value *= alpha;
            }
            d.data[row * d.cols + column] = value;
        }
    }
    Ok(())
}

/// Owned `cvTransform` for row vectors, with an optional per-output shift.
pub fn cv_transform(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
    transform: &CvMatrix<f64>,
    shift: Option<&[f64]>,
) -> Result<(), CvMatMulError> {
    if source.rows != destination.rows
        || source.cols != transform.cols
        || destination.cols != transform.rows
        || shift.map_or(false, |value| value.len() != destination.cols)
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    for row in 0..source.rows {
        for output in 0..destination.cols {
            let mut value = shift.map_or(0.0, |v| v[output]);
            for input in 0..source.cols {
                value += source.data[row * source.cols + input]
                    * transform.data[output * transform.cols + input];
            }
            destination.data[row * destination.cols + output] = value;
        }
    }
    Ok(())
}

/// Owned `cvPerspectiveTransform` for 2D/3D row-vector points.
pub fn cv_perspective_transform(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
    transform: &CvMatrix<f64>,
) -> Result<(), CvMatMulError> {
    if (source.cols != 2 && source.cols != 3)
        || destination.rows != source.rows
        || destination.cols != source.cols
        || transform.rows != source.cols + 1
        || transform.cols != source.cols + 1
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    for row in 0..source.rows {
        let mut homogeneous = vec![0.0; source.cols + 1];
        for input in 0..source.cols {
            homogeneous[input] = source.data[row * source.cols + input];
        }
        homogeneous[source.cols] = 1.0;
        let mut projected = vec![0.0; source.cols + 1];
        for output in 0..=source.cols {
            for input in 0..=source.cols {
                projected[output] +=
                    transform.data[output * transform.cols + input] * homogeneous[input];
            }
        }
        for output in 0..source.cols {
            destination.data[row * destination.cols + output] =
                projected[output] / projected[source.cols];
        }
    }
    Ok(())
}

/// Owned `cvScaleAdd`.
pub fn cv_scale_add(
    first: &CvMatrix<f64>,
    scale: &[f64],
    second: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
) -> Result<(), CvMatMulError> {
    if first.rows != second.rows
        || first.cols != second.cols
        || destination.rows != first.rows
        || destination.cols != first.cols
        || (scale.len() != 1 && scale.len() != first.cols)
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    for row in 0..first.rows {
        for column in 0..first.cols {
            destination.data[row * destination.cols + column] = first.data
                [row * first.cols + column]
                * scale[if scale.len() == 1 { 0 } else { column }]
                + second.data[row * second.cols + column];
        }
    }
    Ok(())
}

/// Owned `cvMulTransposed` (`order == 0` gives `AᵀA`; `order != 0` gives `AAᵀ`).
pub fn cv_mul_transposed(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
    order: i32,
    delta: Option<&[f64]>,
    scale: f64,
) -> Result<(), CvMatMulError> {
    let n = if order == 0 { source.cols } else { source.rows };
    if destination.rows != n
        || destination.cols != n
        || delta.map_or(false, |value| {
            value.len() != if order == 0 { source.cols } else { source.rows }
        })
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    for row in 0..n {
        for column in 0..n {
            let mut value = 0.0;
            let count = if order == 0 { source.rows } else { source.cols };
            for index in 0..count {
                let left = if order == 0 {
                    source.data[index * source.cols + row] - delta.map_or(0.0, |d| d[row])
                } else {
                    source.data[row * source.cols + index] - delta.map_or(0.0, |d| d[row])
                };
                let right = if order == 0 {
                    source.data[index * source.cols + column] - delta.map_or(0.0, |d| d[column])
                } else {
                    source.data[column * source.cols + index] - delta.map_or(0.0, |d| d[column])
                };
                value += left * right;
            }
            destination.data[row * n + column] = value * scale;
        }
    }
    Ok(())
}

/// Owned `cvMahalanobis`.
pub fn cv_mahalanobis(
    first: &[f64],
    second: &[f64],
    inverse_covariance: &CvMatrix<f64>,
) -> Result<f64, CvMatMulError> {
    if first.len() != second.len()
        || inverse_covariance.rows != first.len()
        || inverse_covariance.cols != first.len()
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    let mut value = 0.0;
    for row in 0..first.len() {
        for column in 0..first.len() {
            value += (first[row] - second[row])
                * inverse_covariance.data[row * inverse_covariance.cols + column]
                * (first[column] - second[column]);
        }
    }
    Ok(value.sqrt())
}

/// Owned `cvCalcCovarMatrix` for observations stored as matrix rows.
pub fn cv_calc_covar_matrix(
    samples: &CvMatrix<f64>,
    covariance: &mut CvMatrix<f64>,
    average: &mut [f64],
    flags: i32,
) -> Result<(), CvMatMulError> {
    if samples.rows == 0
        || samples.cols == 0
        || average.len() != samples.cols
        || covariance.rows != samples.cols
        || covariance.cols != samples.cols
    {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    if flags & CV_COVAR_USE_AVG == 0 {
        average.fill(0.0);
        for row in 0..samples.rows {
            for column in 0..samples.cols {
                average[column] += samples.data[row * samples.cols + column];
            }
        }
        for value in average.iter_mut() {
            *value /= samples.rows as f64;
        }
    }
    for row in 0..samples.cols {
        for column in 0..samples.cols {
            let mut value = 0.0;
            for sample in 0..samples.rows {
                value += (samples.data[sample * samples.cols + row] - average[row])
                    * (samples.data[sample * samples.cols + column] - average[column]);
            }
            covariance.data[row * covariance.cols + column] = if flags & CV_COVAR_SCALE != 0 {
                value / samples.rows as f64
            } else {
                value
            };
        }
    }
    Ok(())
}

/// Owned `cvDotProduct`.
pub fn cv_dot_product(first: &[f64], second: &[f64]) -> Result<f64, CvMatMulError> {
    if first.len() != second.len() {
        return Err(CvMatMulError::UnmatchedSizes);
    }
    Ok(first.iter().zip(second).map(|(a, b)| a * b).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemm_transform_and_distance_follow_matrix_contracts() {
        let a = CvMatrix::new(2, 2, vec![1., 2., 3., 4.]).unwrap();
        let b = CvMatrix::new(2, 2, vec![5., 6., 7., 8.]).unwrap();
        let mut d = CvMatrix::new(2, 2, vec![0.; 4]).unwrap();
        cv_gemm(&a, &b, 1., None, 0., &mut d, 0).unwrap();
        assert_eq!(d.data, [19., 22., 43., 50.]);
        assert!(
            (cv_mahalanobis(
                &[1., 2.],
                &[0., 0.],
                &CvMatrix::new(2, 2, vec![1., 0., 0., 1.]).unwrap()
            )
            .unwrap()
                - 5_f64.sqrt())
            .abs()
                < 1e-12
        );
    }
}
