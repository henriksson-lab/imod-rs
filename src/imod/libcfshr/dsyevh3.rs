//! Translation of `IMOD/libcfshr/dsyevh3.c`.

use crate::imod::libcfshr::dsyevc3::dsyevc3;
use crate::imod::libcfshr::dsyevq3::dsyevq3;

/// Original `dsyevh3` (`dsyevh3.c:35`).
pub fn dsyevh3(
    matrix: &[[f64; 3]; 3],
    orthogonal: &mut [[f64; 3]; 3],
    eigenvalues: &mut [f64; 3],
) -> i32 {
    dsyevc3(matrix, eigenvalues);
    let mut maximum = eigenvalues[0].abs();
    let mut temporary = eigenvalues[1].abs();
    if temporary > maximum {
        maximum = temporary;
    }
    temporary = eigenvalues[2].abs();
    if temporary > maximum {
        maximum = temporary;
    }
    let scale = if maximum < 1.0 {
        maximum
    } else {
        maximum * maximum
    };
    let error = 256.0 * f64::EPSILON * scale * scale;
    orthogonal[0][1] = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1];
    orthogonal[1][1] = matrix[0][2] * matrix[0][1] - matrix[1][2] * matrix[0][0];
    orthogonal[2][1] = matrix[0][1] * matrix[0][1];
    orthogonal[0][0] = orthogonal[0][1] + matrix[0][2] * eigenvalues[0];
    orthogonal[1][0] = orthogonal[1][1] + matrix[1][2] * eigenvalues[0];
    orthogonal[2][0] =
        (matrix[0][0] - eigenvalues[0]) * (matrix[1][1] - eigenvalues[0]) - orthogonal[2][1];
    let mut norm = orthogonal[0][0] * orthogonal[0][0]
        + orthogonal[1][0] * orthogonal[1][0]
        + orthogonal[2][0] * orthogonal[2][0];
    if norm <= error {
        return dsyevq3(matrix, orthogonal, eigenvalues);
    }
    norm = (1.0 / norm).sqrt();
    for row in 0..3 {
        orthogonal[row][0] *= norm;
    }
    orthogonal[0][1] += matrix[0][2] * eigenvalues[1];
    orthogonal[1][1] += matrix[1][2] * eigenvalues[1];
    orthogonal[2][1] =
        (matrix[0][0] - eigenvalues[1]) * (matrix[1][1] - eigenvalues[1]) - orthogonal[2][1];
    norm = orthogonal[0][1] * orthogonal[0][1]
        + orthogonal[1][1] * orthogonal[1][1]
        + orthogonal[2][1] * orthogonal[2][1];
    if norm <= error {
        return dsyevq3(matrix, orthogonal, eigenvalues);
    }
    norm = (1.0 / norm).sqrt();
    for row in 0..3 {
        orthogonal[row][1] *= norm;
    }
    orthogonal[0][2] = orthogonal[1][0] * orthogonal[2][1] - orthogonal[2][0] * orthogonal[1][1];
    orthogonal[1][2] = orthogonal[2][0] * orthogonal[0][1] - orthogonal[0][0] * orthogonal[2][1];
    orthogonal[2][2] = orthogonal[0][0] * orthogonal[1][1] - orthogonal[1][0] * orthogonal[0][1];
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_and_ql_fallback_paths_produce_orthonormal_eigenvectors() {
        let matrix = [[4.0, 1.0, 2.0], [1.0, 3.0, 5.0], [2.0, 5.0, 6.0]];
        let mut q = [[0.0; 3]; 3];
        let mut values = [0.0; 3];
        assert_eq!(dsyevh3(&matrix, &mut q, &mut values), 0);
        for column in 0..3 {
            let norm = q[0][column] * q[0][column]
                + q[1][column] * q[1][column]
                + q[2][column] * q[2][column];
            assert!((norm - 1.0).abs() < 1.0e-10);
        }
        let repeated = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        assert_eq!(dsyevh3(&repeated, &mut q, &mut values), 0);
        assert_eq!(values, [2.0, 2.0, 2.0]);
    }
}
