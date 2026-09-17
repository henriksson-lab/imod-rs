//! Owned SVD and back-substitution from `IMOD/raptor/opencv/cxsvd.cpp`.

use super::cxutils::CvMatrix;

/// C SVD flag values retained for callers that record source-compatible mode.
pub const CV_SVD_MODIFY_A: i32 = 1;
pub const CV_SVD_U_T: i32 = 2;
pub const CV_SVD_V_T: i32 = 4;

/// Safe owned SVD result.  `u` is thin (`rows(A) × columns(A)`), and `vt` is
/// square (`columns(A) × columns(A)`), matching the source's primary route.
#[derive(Clone, Debug, PartialEq)]
pub struct CvSvd {
    pub singular_values: Vec<f64>,
    pub u: CvMatrix<f64>,
    pub vt: CvMatrix<f64>,
}

/// Single-precision counterpart of [`CvSvd`], used by the source's `_32f`
/// entry points without exposing pointer-based storage.
#[derive(Clone, Debug, PartialEq)]
pub struct CvSvd32 {
    pub singular_values: Vec<f32>,
    pub u: CvMatrix<f32>,
    pub vt: CvMatrix<f32>,
}

/// SVD failures that map to the source's size/format checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvSvdError {
    BadArgument,
    UnmatchedSizes,
}

/// `pythag` (`cxsvd.cpp:208`): avoid intermediate overflow/underflow while
/// computing a Euclidean norm.
pub fn pythag(a: f64, b: f64) -> f64 {
    let (large, small) = if a.abs() > b.abs() {
        (a.abs(), b.abs())
    } else {
        (b.abs(), a.abs())
    };
    if large == 0. {
        0.
    } else {
        large * (1. + (small / large).powi(2)).sqrt()
    }
}

/// `icvMatrAXPY_64f` / `icvMatrAXPY_32f`: `y[i] += a[row] * x[i]`
/// over strided rows.  The slice API makes the native pointer increments
/// checked and explicit.
pub fn icv_matr_axpy_64f(
    m: usize,
    n: usize,
    x: &[f64],
    x_stride: usize,
    a: &[f64],
    y: &mut [f64],
    y_stride: usize,
) -> Result<(), CvSvdError> {
    if a.len() < m
        || (m > 0 && (x.len() < (m - 1) * x_stride + n || y.len() < (m - 1) * y_stride + n))
    {
        return Err(CvSvdError::BadArgument);
    }
    for row in 0..m {
        for column in 0..n {
            y[row * y_stride + column] += a[row] * x[row * x_stride + column];
        }
    }
    Ok(())
}
pub fn icv_matr_axpy_32f(
    m: usize,
    n: usize,
    x: &[f32],
    x_stride: usize,
    a: &[f32],
    y: &mut [f32],
    y_stride: usize,
) -> Result<(), CvSvdError> {
    if a.len() < m
        || (m > 0 && (x.len() < (m - 1) * x_stride + n || y.len() < (m - 1) * y_stride + n))
    {
        return Err(CvSvdError::BadArgument);
    }
    for row in 0..m {
        for column in 0..n {
            y[row * y_stride + column] = (y[row * y_stride + column] as f64
                + a[row] as f64 * x[row * x_stride + column] as f64)
                as f32;
        }
    }
    Ok(())
}

/// `icvMatrAXPY3_64f` / `icvMatrAXPY3_32f`.  `householder[0]` is the C
/// function's otherwise out-of-bounds `x[-1]`, followed by its `n` active
/// vector elements; rows after the first receive the rank-one update.
pub fn icv_matr_axpy3_64f(
    m: usize,
    n: usize,
    householder: &[f64],
    row_stride: usize,
    y: &mut [f64],
    h: f64,
) -> Result<(), CvSvdError> {
    if householder.len() < n + 1 || row_stride < n || (m > 1 && y.len() < (m - 1) * row_stride + n)
    {
        return Err(CvSvdError::BadArgument);
    }
    for row in 1..m {
        let base = row * row_stride;
        let scale = h
            * (0..n)
                .map(|column| householder[column + 1] * y[base + column])
                .sum::<f64>();
        y[base - 1] = scale * householder[0];
        for column in 0..n {
            y[base + column] += scale * householder[column + 1];
        }
    }
    Ok(())
}
pub fn icv_matr_axpy3_32f(
    m: usize,
    n: usize,
    householder: &[f32],
    row_stride: usize,
    y: &mut [f32],
    h: f64,
) -> Result<(), CvSvdError> {
    if householder.len() < n + 1 || row_stride < n || (m > 1 && y.len() < (m - 1) * row_stride + n)
    {
        return Err(CvSvdError::BadArgument);
    }
    for row in 1..m {
        let base = row * row_stride;
        let scale = h
            * (0..n)
                .map(|column| householder[column + 1] as f64 * y[base + column] as f64)
                .sum::<f64>();
        y[base - 1] = (scale * householder[0] as f64) as f32;
        for column in 0..n {
            y[base + column] =
                (y[base + column] as f64 + scale * householder[column + 1] as f64) as f32;
        }
    }
    Ok(())
}

/// Owned `cvSVD`, using deterministic Jacobi rotations of `AᵀA`.
pub fn cv_svd(a: &CvMatrix<f64>) -> Result<CvSvd, CvSvdError> {
    if a.rows == 0 || a.cols == 0 || a.data.len() != a.rows * a.cols {
        return Err(CvSvdError::BadArgument);
    }
    let n = a.cols;
    let mut normal = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            for sample in 0..a.rows {
                normal[row * n + column] += a.data[sample * n + row] * a.data[sample * n + column];
            }
        }
    }
    let mut vectors = vec![0.0; n * n];
    for diagonal in 0..n {
        vectors[diagonal * n + diagonal] = 1.0;
    }
    for _ in 0..(n * n * 32).max(1) {
        let mut largest = 0.0;
        let mut p = 0;
        let mut q = 0;
        for row in 0..n {
            for column in row + 1..n {
                if normal[row * n + column].abs() > largest {
                    largest = normal[row * n + column].abs();
                    p = row;
                    q = column;
                }
            }
        }
        if largest
            <= f64::EPSILON
                * normal
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0)
        {
            break;
        }
        let phi = 0.5 * (2.0 * normal[p * n + q]).atan2(normal[q * n + q] - normal[p * n + p]);
        let cosine = phi.cos();
        let sine = phi.sin();
        for row in 0..n {
            let left = normal[row * n + p];
            let right = normal[row * n + q];
            normal[row * n + p] = cosine * left - sine * right;
            normal[row * n + q] = sine * left + cosine * right;
        }
        for column in 0..n {
            let left = normal[p * n + column];
            let right = normal[q * n + column];
            normal[p * n + column] = cosine * left - sine * right;
            normal[q * n + column] = sine * left + cosine * right;
        }
        for row in 0..n {
            let left = vectors[row * n + p];
            let right = vectors[row * n + q];
            vectors[row * n + p] = cosine * left - sine * right;
            vectors[row * n + q] = sine * left + cosine * right;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        normal[right * n + right]
            .partial_cmp(&normal[left * n + left])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut singular_values = vec![0.0; n];
    let mut v = vec![0.0; n * n];
    for target in 0..n {
        let source = order[target];
        singular_values[target] = normal[source * n + source].max(0.0).sqrt();
        for row in 0..n {
            v[row * n + target] = vectors[row * n + source];
        }
    }
    let mut u =
        CvMatrix::new(a.rows, n, vec![0.0; a.rows * n]).map_err(|_| CvSvdError::BadArgument)?;
    let tolerance =
        singular_values.first().copied().unwrap_or(0.0) * f64::EPSILON * a.rows.max(n) as f64;
    for row in 0..a.rows {
        for column in 0..n {
            if singular_values[column] > tolerance {
                for source in 0..n {
                    u.data[row * n + column] +=
                        a.data[row * n + source] * v[source * n + column] / singular_values[column];
                }
            }
        }
    }
    let mut vt = CvMatrix::new(n, n, vec![0.0; n * n]).map_err(|_| CvSvdError::BadArgument)?;
    for row in 0..n {
        for column in 0..n {
            vt.data[row * n + column] = v[column * n + row];
        }
    }
    Ok(CvSvd {
        singular_values,
        u,
        vt,
    })
}

/// Checked, owned replacement for `icvSVD_64f` (`cxsvd.cpp:232`).
/// Row strides and the scratch workspace of the C implementation are absorbed
/// by [`CvMatrix`] and the returned factorization.
pub fn icv_svd_64f(a: &CvMatrix<f64>) -> Result<CvSvd, CvSvdError> {
    cv_svd(a)
}

/// Checked single-precision replacement for `icvSVD_32f` (`cxsvd.cpp:627`).
/// Calculating rotations in f64 matches the source's double-precision
/// intermediates; results are converted back at the f32 API boundary.
pub fn icv_svd_32f(a: &CvMatrix<f32>) -> Result<CvSvd32, CvSvdError> {
    let wide = CvMatrix::new(
        a.rows,
        a.cols,
        a.data.iter().map(|&value| value as f64).collect(),
    )
    .map_err(|_| CvSvdError::BadArgument)?;
    let svd = cv_svd(&wide)?;
    Ok(CvSvd32 {
        singular_values: svd
            .singular_values
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        u: CvMatrix::new(
            svd.u.rows,
            svd.u.cols,
            svd.u.data.into_iter().map(|value| value as f32).collect(),
        )
        .map_err(|_| CvSvdError::BadArgument)?,
        vt: CvMatrix::new(
            svd.vt.rows,
            svd.vt.cols,
            svd.vt.data.into_iter().map(|value| value as f32).collect(),
        )
        .map_err(|_| CvSvdError::BadArgument)?,
    })
}

/// Owned `cvSVBkSb`: solve `A x = b` from a decomposition returned by [`cv_svd`].
pub fn cv_svbksb(svd: &CvSvd, b: &CvMatrix<f64>, x: &mut CvMatrix<f64>) -> Result<(), CvSvdError> {
    let m = svd.u.rows;
    let n = svd.singular_values.len();
    if b.rows != m
        || x.rows != n
        || x.cols != b.cols
        || svd.u.cols != n
        || svd.vt.rows != n
        || svd.vt.cols != n
    {
        return Err(CvSvdError::UnmatchedSizes);
    }
    // Native `cvSVBkSb` writes each output coefficient; it does not treat the
    // caller's destination as an accumulator.
    x.data.fill(0.0);
    let threshold =
        svd.singular_values.first().copied().unwrap_or(0.0) * f64::EPSILON * m.max(n) as f64;
    for right_hand_side in 0..b.cols {
        let mut weighted = vec![0.0; n];
        for singular in 0..n {
            if svd.singular_values[singular] > threshold {
                for row in 0..m {
                    weighted[singular] +=
                        svd.u.data[row * n + singular] * b.data[row * b.cols + right_hand_side];
                }
                weighted[singular] /= svd.singular_values[singular];
            }
        }
        for row in 0..n {
            for singular in 0..n {
                x.data[row * x.cols + right_hand_side] +=
                    svd.vt.data[singular * n + row] * weighted[singular];
            }
        }
    }
    Ok(())
}

/// Checked, owned replacement for `icvSVBkSb_64f` (`cxsvd.cpp:1023`).
pub fn icv_svbksb_64f(
    svd: &CvSvd,
    b: &CvMatrix<f64>,
    x: &mut CvMatrix<f64>,
) -> Result<(), CvSvdError> {
    cv_svbksb(svd, b, x)
}

/// Checked single-precision replacement for `icvSVBkSb_32f` (`cxsvd.cpp:1117`).
pub fn icv_svbksb_32f(
    svd: &CvSvd32,
    b: &CvMatrix<f32>,
    x: &mut CvMatrix<f32>,
) -> Result<(), CvSvdError> {
    let wide_svd = CvSvd {
        singular_values: svd
            .singular_values
            .iter()
            .map(|&value| value as f64)
            .collect(),
        u: CvMatrix::new(
            svd.u.rows,
            svd.u.cols,
            svd.u.data.iter().map(|&value| value as f64).collect(),
        )
        .map_err(|_| CvSvdError::BadArgument)?,
        vt: CvMatrix::new(
            svd.vt.rows,
            svd.vt.cols,
            svd.vt.data.iter().map(|&value| value as f64).collect(),
        )
        .map_err(|_| CvSvdError::BadArgument)?,
    };
    let wide_b = CvMatrix::new(
        b.rows,
        b.cols,
        b.data.iter().map(|&value| value as f64).collect(),
    )
    .map_err(|_| CvSvdError::BadArgument)?;
    let mut wide_x = CvMatrix::new(x.rows, x.cols, vec![0.0; x.data.len()])
        .map_err(|_| CvSvdError::BadArgument)?;
    icv_svbksb_64f(&wide_svd, &wide_b, &mut wide_x)?;
    x.data
        .iter_mut()
        .zip(wide_x.data)
        .for_each(|(output, value)| *output = value as f32);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn svd_reconstructs_and_solves_a_well_conditioned_matrix() {
        let a = CvMatrix::new(3, 2, vec![3., 1., 1., 3., 2., 2.]).unwrap();
        let svd = cv_svd(&a).unwrap();
        assert!(svd.singular_values[0] >= svd.singular_values[1]);
        let b = CvMatrix::new(3, 1, vec![4., 4., 4.]).unwrap();
        let mut x = CvMatrix::new(2, 1, vec![99.; 2]).unwrap();
        cv_svbksb(&svd, &b, &mut x).unwrap();
        assert!((x.data[0] - 1.).abs() < 1e-8 && (x.data[1] - 1.).abs() < 1e-8);
    }

    #[test]
    fn native_helpers_honor_strides_and_precision() {
        let mut y = [0., 0., 7., 2., 3.];
        icv_matr_axpy3_64f(2, 2, &[0.5, 1., 2.], 3, &mut y, 1.).unwrap();
        assert_eq!(y, [0., 0., 4., 10., 19.]);

        let mut add = [0_f32; 6];
        icv_matr_axpy_32f(2, 2, &[1., 2., 0., 3., 4., 0.], 3, &[2., 3.], &mut add, 3).unwrap();
        assert_eq!(add, [2., 4., 0., 9., 12., 0.]);

        let a = CvMatrix::new(2, 2, vec![3_f32, 1., 1., 3.]).unwrap();
        let svd = icv_svd_32f(&a).unwrap();
        let b = CvMatrix::new(2, 1, vec![4_f32, 4.]).unwrap();
        let mut x = CvMatrix::new(2, 1, vec![99_f32; 2]).unwrap();
        icv_svbksb_32f(&svd, &b, &mut x).unwrap();
        assert!((x.data[0] - 1.).abs() < 1e-5 && (x.data[1] - 1.).abs() < 1e-5);
    }
}
