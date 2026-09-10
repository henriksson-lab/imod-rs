//! Translation of `IMOD/libcfshr/dsytrd3.c`.
#![allow(dead_code)]

/// Original `dsytrd3` (`dsytrd3.c:33`).
pub fn dsytrd3(
    matrix: &[[f64; 3]; 3],
    orthogonal: &mut [[f64; 3]; 3],
    diagonal: &mut [f64; 3],
    off_diagonal: &mut [f64; 2],
) {
    for row in 0..3 {
        orthogonal[row][row] = 1.0;
        for column in 0..row {
            orthogonal[row][column] = 0.0;
            orthogonal[column][row] = 0.0;
        }
    }
    let h = matrix[0][1] * matrix[0][1] + matrix[0][2] * matrix[0][2];
    let g = if matrix[0][1] > 0.0 {
        -h.sqrt()
    } else {
        h.sqrt()
    };
    off_diagonal[0] = g;
    let mut f = g * matrix[0][1];
    let mut u = [0.0; 3];
    let mut q = [0.0; 3];
    u[1] = matrix[0][1] - g;
    u[2] = matrix[0][2];
    let mut omega = h - f;
    if omega > 0.0 {
        omega = 1.0 / omega;
        let mut k = 0.0;
        for index in 1..3 {
            f = matrix[1][index] * u[1] + matrix[index][2] * u[2];
            q[index] = omega * f;
            k += u[index] * f;
        }
        k *= 0.5 * omega * omega;
        for index in 1..3 {
            q[index] -= k * u[index];
        }
        diagonal[0] = matrix[0][0];
        diagonal[1] = matrix[1][1] - 2.0 * q[1] * u[1];
        diagonal[2] = matrix[2][2] - 2.0 * q[2] * u[2];
        for column in 1..3 {
            f = omega * u[column];
            for row in 1..3 {
                orthogonal[row][column] -= f * u[row];
            }
        }
        off_diagonal[1] = matrix[1][2] - q[1] * u[2] - u[1] * q[2];
    } else {
        for index in 0..3 {
            diagonal[index] = matrix[index][index];
        }
        off_diagonal[1] = matrix[1][2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_diagonal_and_general_symmetric_matrices_with_source_layout() {
        let diagonal_input = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let mut q = [[-1.0; 3]; 3];
        let mut d = [0.0; 3];
        let mut e = [0.0; 2];
        dsytrd3(&diagonal_input, &mut q, &mut d, &mut e);
        assert_eq!(d, [2.0, 3.0, 5.0]);
        assert_eq!(e, [0.0, 0.0]);
        let input = [[4.0, 1.0, 2.0], [1.0, 3.0, 5.0], [2.0, 5.0, 6.0]];
        dsytrd3(&input, &mut q, &mut d, &mut e);
        assert!(e[0].is_finite() && e[1].is_finite());
        assert_eq!(q[0], [1.0, 0.0, 0.0]);
    }
}
