//! Translation of `IMOD/libcfshr/dsyevv3.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::dsyevc3::dsyevc3;

/// Original `dsyevv3` (`dsyevv3.c:35`).
pub fn dsyevv3(
    matrix: &[[f64; 3]; 3],
    orthogonal: &mut [[f64; 3]; 3],
    eigenvalues: &mut [f64; 3],
) -> i32 {
    let mut matrix = *matrix;
    dsyevc3(&matrix, eigenvalues);
    let mut maximum_eigenvalue = eigenvalues[0].abs();
    let mut temporary = eigenvalues[1].abs();
    if temporary > maximum_eigenvalue {
        maximum_eigenvalue = temporary;
    }
    temporary = eigenvalues[2].abs();
    if temporary > maximum_eigenvalue {
        maximum_eigenvalue = temporary;
    }
    let threshold = (8.0 * f64::EPSILON * maximum_eigenvalue).powi(2);
    let first_template = matrix[0][1] * matrix[0][1] + matrix[0][2] * matrix[0][2];
    let second_template = matrix[0][1] * matrix[0][1] + matrix[1][2] * matrix[1][2];
    orthogonal[0][1] = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1];
    orthogonal[1][1] = matrix[0][2] * matrix[0][1] - matrix[1][2] * matrix[0][0];
    orthogonal[2][1] = matrix[0][1] * matrix[0][1];
    matrix[0][0] -= eigenvalues[0];
    matrix[1][1] -= eigenvalues[0];
    orthogonal[0][0] = orthogonal[0][1] + matrix[0][2] * eigenvalues[0];
    orthogonal[1][0] = orthogonal[1][1] + matrix[1][2] * eigenvalues[0];
    orthogonal[2][0] = matrix[0][0] * matrix[1][1] - orthogonal[2][1];
    let mut norm = orthogonal[0][0] * orthogonal[0][0]
        + orthogonal[1][0] * orthogonal[1][0]
        + orthogonal[2][0] * orthogonal[2][0];
    let mut first_norm = first_template + matrix[0][0] * matrix[0][0];
    let mut second_norm = second_template + matrix[1][1] * matrix[1][1];
    let mut error = first_norm * second_norm;
    if first_norm <= threshold {
        orthogonal[0][0] = 1.0;
        orthogonal[1][0] = 0.0;
        orthogonal[2][0] = 0.0;
    } else if second_norm <= threshold {
        orthogonal[0][0] = 0.0;
        orthogonal[1][0] = 1.0;
        orthogonal[2][0] = 0.0;
    } else if norm < (64.0 * f64::EPSILON).powi(2) * error {
        temporary = matrix[0][1] * matrix[0][1];
        let mut factor = -matrix[0][0] / matrix[0][1];
        if matrix[1][1] * matrix[1][1] > temporary {
            temporary = matrix[1][1] * matrix[1][1];
            factor = -matrix[0][1] / matrix[1][1];
        }
        if matrix[1][2] * matrix[1][2] > temporary {
            factor = -matrix[0][2] / matrix[1][2];
        }
        norm = 1.0 / (1.0 + factor * factor).sqrt();
        orthogonal[0][0] = norm;
        orthogonal[1][0] = factor * norm;
        orthogonal[2][0] = 0.0;
    } else {
        norm = (1.0 / norm).sqrt();
        for row in 0..3 {
            orthogonal[row][0] *= norm;
        }
    }
    temporary = eigenvalues[0] - eigenvalues[1];
    if temporary.abs() > 8.0 * f64::EPSILON * maximum_eigenvalue {
        matrix[0][0] += temporary;
        matrix[1][1] += temporary;
        orthogonal[0][1] += matrix[0][2] * eigenvalues[1];
        orthogonal[1][1] += matrix[1][2] * eigenvalues[1];
        orthogonal[2][1] = matrix[0][0] * matrix[1][1] - orthogonal[2][1];
        norm = orthogonal[0][1] * orthogonal[0][1]
            + orthogonal[1][1] * orthogonal[1][1]
            + orthogonal[2][1] * orthogonal[2][1];
        first_norm = first_template + matrix[0][0] * matrix[0][0];
        second_norm = second_template + matrix[1][1] * matrix[1][1];
        error = first_norm * second_norm;
        if first_norm <= threshold {
            orthogonal[0][1] = 1.0;
            orthogonal[1][1] = 0.0;
            orthogonal[2][1] = 0.0;
        } else if second_norm <= threshold {
            orthogonal[0][1] = 0.0;
            orthogonal[1][1] = 1.0;
            orthogonal[2][1] = 0.0;
        } else if norm < (64.0 * f64::EPSILON).powi(2) * error {
            temporary = matrix[0][1] * matrix[0][1];
            let mut factor = -matrix[0][0] / matrix[0][1];
            if matrix[1][1] * matrix[1][1] > temporary {
                temporary = matrix[1][1] * matrix[1][1];
                factor = -matrix[0][1] / matrix[1][1];
            }
            if matrix[1][2] * matrix[1][2] > temporary {
                factor = -matrix[0][2] / matrix[1][2];
            }
            norm = 1.0 / (1.0 + factor * factor).sqrt();
            orthogonal[0][1] = norm;
            orthogonal[1][1] = factor * norm;
            orthogonal[2][1] = 0.0;
        } else {
            norm = (1.0 / norm).sqrt();
            for row in 0..3 {
                orthogonal[row][1] *= norm;
            }
        }
    } else {
        matrix[1][0] = matrix[0][1];
        matrix[2][0] = matrix[0][2];
        matrix[2][1] = matrix[1][2];
        matrix[0][0] += eigenvalues[0];
        matrix[1][1] += eigenvalues[0];
        let mut index = 0;
        while index < 3 {
            matrix[index][index] -= eigenvalues[1];
            first_norm = matrix[0][index] * matrix[0][index]
                + matrix[1][index] * matrix[1][index]
                + matrix[2][index] * matrix[2][index];
            if first_norm > threshold {
                orthogonal[0][1] =
                    orthogonal[1][0] * matrix[2][index] - orthogonal[2][0] * matrix[1][index];
                orthogonal[1][1] =
                    orthogonal[2][0] * matrix[0][index] - orthogonal[0][0] * matrix[2][index];
                orthogonal[2][1] =
                    orthogonal[0][0] * matrix[1][index] - orthogonal[1][0] * matrix[0][index];
                norm = orthogonal[0][1] * orthogonal[0][1]
                    + orthogonal[1][1] * orthogonal[1][1]
                    + orthogonal[2][1] * orthogonal[2][1];
                if norm > (256.0 * f64::EPSILON).powi(2) * first_norm {
                    norm = (1.0 / norm).sqrt();
                    for row in 0..3 {
                        orthogonal[row][1] *= norm;
                    }
                    break;
                }
            }
            index += 1;
        }
        if index == 3 {
            for row in 0..3 {
                if orthogonal[row][0] != 0.0 {
                    norm = 1.0
                        / (orthogonal[row][0] * orthogonal[row][0]
                            + orthogonal[(row + 1) % 3][0] * orthogonal[(row + 1) % 3][0])
                            .sqrt();
                    orthogonal[row][1] = orthogonal[(row + 1) % 3][0] * norm;
                    orthogonal[(row + 1) % 3][1] = -orthogonal[row][0] * norm;
                    orthogonal[(row + 2) % 3][1] = 0.0;
                    break;
                }
            }
        }
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
    fn handles_general_and_degenerate_source_branches() {
        let general = [[4.0, 1.0, 2.0], [1.0, 3.0, 5.0], [2.0, 5.0, 6.0]];
        let mut q = [[0.0; 3]; 3];
        let mut w = [0.0; 3];
        assert_eq!(dsyevv3(&general, &mut q, &mut w), 0);
        assert!(q.iter().flatten().all(|value| value.is_finite()));
        let diagonal = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        assert_eq!(dsyevv3(&diagonal, &mut q, &mut w), 0);
        assert_eq!(w, [2.0, 2.0, 2.0]);
    }
}
