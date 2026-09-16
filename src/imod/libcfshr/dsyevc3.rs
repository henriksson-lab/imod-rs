//! Translation of `IMOD/libcfshr/dsyevc3.c`.
#![allow(dead_code)]

/// Complete function inventory for `dsyevc3.c`.
pub const DSYEVC3_SOURCE_FUNCTIONS: &[&str] = &["dsyevc3"];

/// Original `dsyevc3` (`dsyevc3.c:35`).
pub fn dsyevc3(matrix: &[[f64; 3]; 3], eigenvalues: &mut [f64; 3]) -> i32 {
    let diagonal_product = matrix[0][1] * matrix[1][2];
    let first_off_diagonal_square = matrix[0][1] * matrix[0][1];
    let second_off_diagonal_square = matrix[1][2] * matrix[1][2];
    let third_off_diagonal_square = matrix[0][2] * matrix[0][2];
    let mean_sum = matrix[0][0] + matrix[1][1] + matrix[2][2];
    let coefficient_one =
        matrix[0][0] * matrix[1][1] + matrix[0][0] * matrix[2][2] + matrix[1][1] * matrix[2][2]
            - (first_off_diagonal_square + second_off_diagonal_square + third_off_diagonal_square);
    let coefficient_zero = matrix[2][2] * first_off_diagonal_square
        + matrix[0][0] * second_off_diagonal_square
        + matrix[1][1] * third_off_diagonal_square
        - matrix[0][0] * matrix[1][1] * matrix[2][2]
        - 2.0 * matrix[0][2] * diagonal_product;
    let p = mean_sum * mean_sum - 3.0 * coefficient_one;
    let q = mean_sum * (p - 1.5 * coefficient_one) - 13.5 * coefficient_zero;
    let square_root_p = p.abs().sqrt();
    let mut phi = 27.0
        * (0.25 * coefficient_one * coefficient_one * (p - coefficient_one)
            + coefficient_zero * (q + 6.75 * coefficient_zero));
    phi = (1.0 / 3.0) * phi.abs().sqrt().atan2(q);
    let cosine_component = square_root_p * phi.cos();
    let sine_component = square_root_p * phi.sin() / 1.73205080756887729352744634151;
    eigenvalues[1] = (mean_sum - cosine_component) / 3.0;
    eigenvalues[2] = eigenvalues[1] + sine_component;
    eigenvalues[0] = eigenvalues[1] + cosine_component;
    eigenvalues[1] -= sine_component;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_source_ordered_analytic_eigenvalues() {
        let matrix = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let mut values = [0.0; 3];
        assert_eq!(dsyevc3(&matrix, &mut values), 0);
        assert!((values[0] - 5.0).abs() < 1.0e-12);
        assert!((values[1] - 2.0).abs() < 1.0e-12);
        assert!((values[2] - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn preserves_trace_and_determinant_for_a_dense_symmetric_matrix() {
        let matrix = [[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]];
        let mut values = [0.0; 3];
        assert_eq!(dsyevc3(&matrix, &mut values), 0);
        assert!((values.iter().sum::<f64>() - 12.0).abs() < 1.0e-12);
        let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2].powi(2))
            - matrix[0][1] * (matrix[0][1] * matrix[2][2] - matrix[1][2] * matrix[0][2])
            + matrix[0][2] * (matrix[0][1] * matrix[1][2] - matrix[1][1] * matrix[0][2]);
        assert!((values.iter().product::<f64>() - determinant).abs() < 1.0e-10);
        assert_eq!(DSYEVC3_SOURCE_FUNCTIONS, ["dsyevc3"]);
    }
}
