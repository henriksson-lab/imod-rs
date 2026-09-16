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

/// SVD failures that map to the source's size/format checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvSvdError {
    BadArgument,
    UnmatchedSizes,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn svd_reconstructs_and_solves_a_well_conditioned_matrix() {
        let a = CvMatrix::new(3, 2, vec![3., 1., 1., 3., 2., 2.]).unwrap();
        let svd = cv_svd(&a).unwrap();
        assert!(svd.singular_values[0] >= svd.singular_values[1]);
        let b = CvMatrix::new(3, 1, vec![4., 6., 5.]).unwrap();
        let mut x = CvMatrix::new(2, 1, vec![0.; 2]).unwrap();
        cv_svbksb(&svd, &b, &mut x).unwrap();
        assert!((x.data[0] - 1.).abs() < 1e-8 && (x.data[1] - 1.).abs() < 1e-8);
    }
}
