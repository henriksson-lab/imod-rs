//! Safe numerical core for `IMOD/raptor/opencv/cxmatrix.cpp`.
use super::cxutils::{CvMatrix, CvUtilsError};

/// C `cvSetIdentity` for an owned scalar matrix.
pub fn cv_set_identity(matrix: &mut CvMatrix<f64>, value: f64) {
    matrix.data.fill(0.);
    for i in 0..matrix.rows.min(matrix.cols) {
        matrix.data[i * matrix.cols + i] = value;
    }
}
/// C `cvTrace`.
pub fn cv_trace(matrix: &CvMatrix<f64>) -> f64 {
    (0..matrix.rows.min(matrix.cols))
        .map(|i| matrix.data[i * matrix.cols + i])
        .sum()
}
/// C `cvTranspose`; source permits in-place square transpose.
pub fn cv_transpose(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
) -> Result<(), CvUtilsError> {
    if destination.rows != source.cols || destination.cols != source.rows {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for y in 0..source.rows {
        for x in 0..source.cols {
            destination.data[x * destination.cols + y] = source.data[y * source.cols + x];
        }
    }
    Ok(())
}
/// C LU determinant path in `cvDet`.
pub fn cv_det(matrix: &CvMatrix<f64>) -> Result<f64, CvUtilsError> {
    if matrix.rows != matrix.cols {
        return Err(CvUtilsError::BadSize);
    }
    let n = matrix.rows;
    let mut a = matrix.data.clone();
    let mut sign = 1.;
    let mut determinant = 1.;
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().partial_cmp(&a[j * n + k].abs()).unwrap())
            .unwrap();
        if a[pivot * n + k] == 0. {
            return Ok(0.);
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            sign = -sign;
        }
        let p = a[k * n + k];
        determinant *= p;
        for i in k + 1..n {
            let scale = a[i * n + k] / p;
            for j in k + 1..n {
                a[i * n + j] -= scale * a[k * n + j];
            }
        }
    }
    Ok(sign * determinant)
}
/// C LU `cvSolve` path, replacing source in-place buffers with owned pivots.
pub fn cv_solve(
    a: &CvMatrix<f64>,
    b: &CvMatrix<f64>,
    x: &mut CvMatrix<f64>,
) -> Result<(), CvUtilsError> {
    if a.rows != a.cols || b.rows != a.rows || x.rows != b.rows || x.cols != b.cols {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let n = a.rows;
    let mut lu = a.data.clone();
    let mut rhs = b.data.clone();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| {
                lu[i * n + k]
                    .abs()
                    .partial_cmp(&lu[j * n + k].abs())
                    .unwrap()
            })
            .unwrap();
        if lu[p * n + k] == 0. {
            return Err(CvUtilsError::BadArgument);
        }
        if p != k {
            for j in 0..n {
                lu.swap(k * n + j, p * n + j);
            }
            for j in 0..b.cols {
                rhs.swap(k * b.cols + j, p * b.cols + j);
            }
        }
        for i in k + 1..n {
            let f = lu[i * n + k] / lu[k * n + k];
            lu[i * n + k] = f;
            for j in k + 1..n {
                lu[i * n + j] -= f * lu[k * n + j];
            }
            for j in 0..b.cols {
                rhs[i * b.cols + j] -= f * rhs[k * b.cols + j];
            }
        }
    }
    for i in (0..n).rev() {
        for j in 0..b.cols {
            let mut v = rhs[i * b.cols + j];
            for k in i + 1..n {
                v -= lu[i * n + k] * x.data[k * x.cols + j];
            }
            x.data[i * x.cols + j] = v / lu[i * n + i];
        }
    }
    Ok(())
}
/// C `cvInvert` LU route.
pub fn cv_invert(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
) -> Result<f64, CvUtilsError> {
    if source.rows != source.cols
        || destination.rows != source.rows
        || destination.cols != source.cols
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let n = source.rows;
    let mut identity = CvMatrix::new(
        n,
        n,
        (0..n * n)
            .map(|i| if i / n == i % n { 1. } else { 0. })
            .collect(),
    )?;
    cv_solve(source, &identity, destination)?;
    let d = cv_det(source)?;
    identity.data.clear();
    Ok(d)
}
/// C `cvCrossProduct`.
pub fn cv_cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// C `CV_SVD` / `CV_SVD_SYM` pseudoinverse route of `cvInvert`.
pub fn cv_invert_svd(
    source: &CvMatrix<f64>,
    destination: &mut CvMatrix<f64>,
) -> Result<f64, CvUtilsError> {
    if destination.rows != source.cols || destination.cols != source.rows {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let mut gram = CvMatrix::new(
        source.cols,
        source.cols,
        vec![0.; source.cols * source.cols],
    )?;
    for i in 0..source.cols {
        for j in 0..source.cols {
            for k in 0..source.rows {
                gram.data[i * gram.cols + j] +=
                    source.data[k * source.cols + i] * source.data[k * source.cols + j];
            }
        }
    }
    let (mut values, vectors) = icv_symmetric_eigen(&gram)?;
    let threshold = values.iter().copied().fold(0_f64, f64::max)
        * f64::EPSILON
        * source.rows.max(source.cols) as f64;
    for value in &mut values {
        *value = if *value > threshold { 1. / *value } else { 0. };
    }
    for y in 0..destination.rows {
        for x in 0..destination.cols {
            let mut result = 0.;
            for i in 0..source.cols {
                for j in 0..source.cols {
                    result += vectors.data[i * vectors.cols + j]
                        * values[j]
                        * vectors.data[x * vectors.cols + j]
                        * source.data[y * source.cols + i];
                }
            }
            destination.data[y * destination.cols + x] = result;
        }
    }
    Ok(threshold)
}
/// C PCA flags.
pub const CV_PCA_DATA_AS_ROW: i32 = 0;
pub const CV_PCA_DATA_AS_COL: i32 = 1;
pub const CV_PCA_USE_AVG: i32 = 2;
/// Owned result of C `cvCalcPCA`.
#[derive(Clone, Debug, PartialEq)]
pub struct CvPca {
    pub average: Vec<f64>,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: CvMatrix<f64>,
    pub data_as_columns: bool,
}
/// C `cvCalcPCA`; covariance is scaled by the number of input vectors as in the source.
pub fn cv_calc_pca(
    data: &CvMatrix<f64>,
    component_count: usize,
    flags: i32,
    provided_average: Option<&[f64]>,
) -> Result<CvPca, CvUtilsError> {
    let columns = flags & CV_PCA_DATA_AS_COL != 0;
    let (len, count) = if columns {
        (data.rows, data.cols)
    } else {
        (data.cols, data.rows)
    };
    if component_count > len.min(count) {
        return Err(CvUtilsError::BadSize);
    }
    let mut average = provided_average.map_or(vec![0.; len], ToOwned::to_owned);
    if average.len() != len {
        return Err(CvUtilsError::BadSize);
    }
    if flags & CV_PCA_USE_AVG == 0 {
        for vector in 0..count {
            for feature in 0..len {
                average[feature] += if columns {
                    data.data[feature * data.cols + vector]
                } else {
                    data.data[vector * data.cols + feature]
                };
            }
        }
        for value in &mut average {
            *value /= count as f64;
        }
    }
    let mut covariance = CvMatrix::new(len, len, vec![0.; len * len])?;
    for i in 0..len {
        for j in 0..len {
            for vector in 0..count {
                let a = if columns {
                    data.data[i * data.cols + vector]
                } else {
                    data.data[vector * data.cols + i]
                } - average[i];
                let b = if columns {
                    data.data[j * data.cols + vector]
                } else {
                    data.data[vector * data.cols + j]
                } - average[j];
                covariance.data[i * len + j] += a * b / count as f64;
            }
        }
    }
    let (mut eigenvalues, all_vectors) = icv_symmetric_eigen(&covariance)?;
    let mut order: (Vec<_>) = (0..len).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let mut vectors = CvMatrix::new(component_count, len, vec![0.; component_count * len])?;
    let mut values = Vec::with_capacity(component_count);
    for (row, &index) in order.iter().take(component_count).enumerate() {
        values.push(eigenvalues[index]);
        for col in 0..len {
            vectors.data[row * len + col] = all_vectors.data[col * len + index];
        }
    }
    eigenvalues.clear();
    Ok(CvPca {
        average,
        eigenvalues: values,
        eigenvectors: vectors,
        data_as_columns: columns,
    })
}
/// C `cvProjectPCA` for row-oriented vectors.
pub fn cv_project_pca(
    data: &CvMatrix<f64>,
    pca: &CvPca,
    result: &mut CvMatrix<f64>,
) -> Result<(), CvUtilsError> {
    if pca.data_as_columns
        || data.cols != pca.average.len()
        || result.rows != data.rows
        || result.cols > pca.eigenvectors.rows
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for row in 0..data.rows {
        for component in 0..result.cols {
            let mut value = 0.;
            for col in 0..data.cols {
                value += (data.data[row * data.cols + col] - pca.average[col])
                    * pca.eigenvectors.data[component * pca.eigenvectors.cols + col];
            }
            result.data[row * result.cols + component] = value;
        }
    }
    Ok(())
}
/// C `cvBackProjectPCA` for row-oriented vectors.
pub fn cv_back_project_pca(
    projection: &CvMatrix<f64>,
    pca: &CvPca,
    result: &mut CvMatrix<f64>,
) -> Result<(), CvUtilsError> {
    if pca.data_as_columns
        || projection.cols > pca.eigenvectors.rows
        || result.rows != projection.rows
        || result.cols != pca.average.len()
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for row in 0..result.rows {
        for col in 0..result.cols {
            let mut value = pca.average[col];
            for component in 0..projection.cols {
                value += projection.data[row * projection.cols + component]
                    * pca.eigenvectors.data[component * pca.eigenvectors.cols + col];
            }
            result.data[row * result.cols + col] = value;
        }
    }
    Ok(())
}
/// Source SVD/covariance calls reduce to a symmetric eigensolve. Eigenvectors are columns.
fn icv_symmetric_eigen(matrix: &CvMatrix<f64>) -> Result<(Vec<f64>, CvMatrix<f64>), CvUtilsError> {
    if matrix.rows != matrix.cols {
        return Err(CvUtilsError::BadSize);
    }
    let n = matrix.rows;
    let mut a = matrix.data.clone();
    let mut vectors = CvMatrix::new(
        n,
        n,
        (0..n * n)
            .map(|i| if i / n == i % n { 1. } else { 0. })
            .collect(),
    )?;
    for _ in 0..n * n * 32 {
        let (mut p, mut q, mut largest) = (0, 0, 0.);
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j].abs() > largest {
                    largest = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= f64::EPSILON {
            break;
        }
        let angle = 0.5 * (2. * a[p * n + q]).atan2(a[q * n + q] - a[p * n + p]);
        let (c, s) = (angle.cos(), angle.sin());
        for i in 0..n {
            let ap = a[i * n + p];
            let aq = a[i * n + q];
            a[i * n + p] = c * ap - s * aq;
            a[i * n + q] = s * ap + c * aq;
        }
        for i in 0..n {
            let ap = a[p * n + i];
            let aq = a[q * n + i];
            a[p * n + i] = c * ap - s * aq;
            a[q * n + i] = s * ap + c * aq;
            let vp = vectors.data[i * n + p];
            let vq = vectors.data[i * n + q];
            vectors.data[i * n + p] = c * vp - s * vq;
            vectors.data[i * n + q] = s * vp + c * vq;
        }
    }
    Ok(((0..n).map(|i| a[i * n + i]).collect(), vectors))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn det_solve_transpose_and_cross() {
        let a = CvMatrix::new(2, 2, vec![4., 7., 2., 6.]).unwrap();
        assert_eq!(cv_det(&a), Ok(10.));
        let b = CvMatrix::new(2, 1, vec![1., 0.]).unwrap();
        let mut x = CvMatrix::new(2, 1, vec![0.; 2]).unwrap();
        cv_solve(&a, &b, &mut x).unwrap();
        assert_eq!(x.data, vec![0.6, -0.2]);
        assert_eq!(cv_cross_product([1., 0., 0.], [0., 1., 0.]), [0., 0., 1.]);
    }
    #[test]
    fn pca_projects_and_backprojects_rows() {
        let data = CvMatrix::new(3, 2, vec![1., 0., 2., 0., 3., 0.]).unwrap();
        let pca = cv_calc_pca(&data, 1, CV_PCA_DATA_AS_ROW, None).unwrap();
        assert!(pca.eigenvalues[0] > 0.);
        let mut projection = CvMatrix::new(3, 1, vec![0.; 3]).unwrap();
        cv_project_pca(&data, &pca, &mut projection).unwrap();
        let mut restored = CvMatrix::new(3, 2, vec![0.; 6]).unwrap();
        cv_back_project_pca(&projection, &pca, &mut restored).unwrap();
        for (i, v) in data.data.iter().enumerate() {
            assert!((restored.data[i] - v).abs() < 1e-8);
        }
    }
}
