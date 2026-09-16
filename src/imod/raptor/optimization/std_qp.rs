//! Safe, owned translation of the matrix-facing groups in
//! `IMOD/raptor/optimization/std_qp.{h,cpp}`.

use super::prob_data::SparseMatrix;
use super::std_qp_data::{Matrix, StdQpData};

/// Safe owned translation of `std_qp`.
///
/// The source's `prob*` branches use block elimination solely to factor the
/// Newton system efficiently.  This version builds that same positive-definite
/// system densely, so it retains the barrier objective and line-search logic
/// without SuiteSparse or OpenCV ownership.
pub fn std_qp(
    quadratic: &SparseMatrix,
    cost: &SparseMatrix,
    constraints: &SparseMatrix,
    bounds: &Matrix<f64>,
    initial: &Matrix<f64>,
    _m: usize,
    _t: usize,
    option: &str,
) -> StdQpData<f64> {
    let mut result = StdQpData {
        answer_mat: Some(initial.clone()),
        ..StdQpData::new()
    };
    if !matches!(option, "prob" | "probV" | "probU")
        || quadratic.nz != -1
        || cost.nz != -1
        || constraints.nz != -1
        || initial.columns != 1
        || bounds.columns != 1
        || quadratic.rows != initial.rows
        || quadratic.columns != initial.rows
        || cost.rows != initial.rows
        || cost.columns != 1
        || constraints.columns != initial.rows
        || constraints.rows != bounds.rows
    {
        return result;
    }
    let n = initial.rows;
    let mut x = initial.values.clone();
    let multiply = |matrix: &SparseMatrix, vector: &[f64]| -> Vec<f64> {
        let mut answer = vec![0.; matrix.rows];
        for column in 0..matrix.columns {
            for entry in matrix.column_or_triplet_offsets[column]
                ..matrix.column_or_triplet_offsets[column + 1]
            {
                answer[matrix.row_indices[entry]] += matrix.values[entry] * vector[column];
            }
        }
        answer
    };
    let mut slack = bounds
        .values
        .iter()
        .zip(multiply(constraints, &x))
        .map(|(b, ax)| b - ax)
        .collect::<Vec<_>>();
    if let Some(minimum) = slack.iter().copied().reduce(f64::min) {
        if minimum < 0. {
            result.exit_flag = 2;
            result.gap = -minimum;
            return result;
        }
    }
    let tolerance = if option == "prob" { 1e-3 } else { 1e-4 };
    let mut barrier = 0.1;
    let mut iterations = 1;
    let mut q_dense = vec![0.; n * n];
    for column in 0..n {
        for entry in quadratic.column_or_triplet_offsets[column]
            ..quadratic.column_or_triplet_offsets[column + 1]
        {
            q_dense[quadratic.row_indices[entry] * n + column] += quadratic.values[entry];
        }
    }
    let mut c = vec![0.; n];
    for entry in cost.column_or_triplet_offsets[0]..cost.column_or_triplet_offsets[1] {
        c[cost.row_indices[entry]] += cost.values[entry];
    }
    let mut exit = false;
    while n as f64 / barrier > tolerance && !exit && iterations < 200 {
        barrier *= 10.;
        for _ in 0..20 {
            iterations += 1;
            result.num_itrs = iterations;
            let qx = multiply(quadratic, &x);
            let mut gradient = (0..n).map(|i| barrier * (qx[i] + c[i])).collect::<Vec<_>>();
            let mut hessian = q_dense.iter().map(|v| v * barrier).collect::<Vec<_>>();
            for row in 0..constraints.rows {
                let inverse = 1. / slack[row];
                let mut a = vec![0.; n];
                for column in 0..n {
                    for entry in constraints.column_or_triplet_offsets[column]
                        ..constraints.column_or_triplet_offsets[column + 1]
                    {
                        if constraints.row_indices[entry] == row {
                            a[column] += constraints.values[entry];
                        }
                    }
                }
                for i in 0..n {
                    gradient[i] += a[i] * inverse;
                    for j in 0..n {
                        hessian[i * n + j] += a[i] * a[j] * inverse * inverse;
                    }
                }
            }
            let mut system = vec![0.; n * (n + 1)];
            for row in 0..n {
                system[row * (n + 1)..row * (n + 1) + n]
                    .copy_from_slice(&hessian[row * n..(row + 1) * n]);
                system[row * (n + 1) + n] = -gradient[row];
            }
            let mut singular = false;
            for pivot in 0..n {
                let best = (pivot..n)
                    .max_by(|&a, &b| {
                        system[a * (n + 1) + pivot]
                            .abs()
                            .total_cmp(&system[b * (n + 1) + pivot].abs())
                    })
                    .unwrap();
                if system[best * (n + 1) + pivot].abs() <= f64::MIN_POSITIVE {
                    singular = true;
                    break;
                }
                for column in pivot..=n {
                    system.swap(pivot * (n + 1) + column, best * (n + 1) + column);
                }
                let divisor = system[pivot * (n + 1) + pivot];
                for column in pivot..=n {
                    system[pivot * (n + 1) + column] /= divisor;
                }
                for row in 0..n {
                    if row != pivot {
                        let factor = system[row * (n + 1) + pivot];
                        for column in pivot..=n {
                            system[row * (n + 1) + column] -=
                                factor * system[pivot * (n + 1) + column];
                        }
                    }
                }
            }
            if singular {
                exit = true;
                break;
            }
            let direction = (0..n)
                .map(|row| system[row * (n + 1) + n])
                .collect::<Vec<_>>();
            let decrement = direction
                .iter()
                .zip(&gradient)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if decrement.abs() < 1e-3 {
                break;
            }
            let objective = |point: &[f64], point_slack: &[f64]| -> f64 {
                let qpoint = multiply(quadratic, point);
                0.5 * barrier * point.iter().zip(qpoint).map(|(a, b)| a * b).sum::<f64>()
                    + barrier * point.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>()
                    - point_slack.iter().map(|v| v.ln()).sum::<f64>()
            };
            let current_objective = objective(&x, &slack);
            let mut step = 1.;
            loop {
                let candidate = x
                    .iter()
                    .zip(&direction)
                    .map(|(v, d)| v + step * d)
                    .collect::<Vec<_>>();
                let candidate_slack = bounds
                    .values
                    .iter()
                    .zip(multiply(constraints, &candidate))
                    .map(|(b, ax)| b - ax)
                    .collect::<Vec<_>>();
                if candidate_slack.iter().all(|v| *v > 0.)
                    && objective(&candidate, &candidate_slack)
                        <= current_objective + 0.25 * step * decrement
                {
                    x = candidate;
                    slack = candidate_slack;
                    break;
                }
                step *= 0.5;
                if step < 1e-11 {
                    exit = true;
                    break;
                }
            }
            if exit {
                break;
            }
            let lambda2 = direction
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    a * hessian[i * n..(i + 1) * n]
                        .iter()
                        .zip(&direction)
                        .map(|(b, c)| b * c)
                        .sum::<f64>()
                })
                .sum::<f64>();
            if lambda2 < 1e-3 {
                break;
            }
        }
    }
    result.answer_mat = Some(Matrix::new(n, 1, x));
    result
}

/// C++ `ZeroMat`.
pub fn zero_mat(rows: usize, columns: usize) -> Matrix<f64> {
    Matrix::new(rows, columns, vec![0.0; rows * columns])
}

/// C++ `MinElem`.
pub fn min_elem(matrix: &Matrix<f64>) -> Option<f64> {
    matrix.values.iter().copied().reduce(f64::min)
}

/// C++ `Combinedx1dx2`.
pub fn combined_x1_dx2(first: &Matrix<f64>, second: &Matrix<f64>) -> Option<Matrix<f64>> {
    if first.columns != second.columns {
        return None;
    }
    let mut values = first.values.clone();
    values.extend_from_slice(&second.values);
    Some(Matrix::new(first.rows + second.rows, first.columns, values))
}

/// C++ `sumLog`.
pub fn sum_log(matrix: &Matrix<f64>) -> Option<f64> {
    (matrix.columns == 1).then(|| matrix.values.iter().map(|value| value.ln()).sum())
}

/// C++ `GetSubMatrix(const cs *, ...)` with inclusive source bounds.
pub fn get_sub_matrix(
    matrix: &SparseMatrix,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
) -> Option<SparseMatrix> {
    if matrix.nz != -1
        || row_start > row_end
        || column_start > column_end
        || row_end >= matrix.rows
        || column_end >= matrix.columns
    {
        return None;
    }
    let rows = row_end - row_start + 1;
    let columns = column_end - column_start + 1;
    let mut offsets = Vec::with_capacity(columns + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for source_column in column_start..=column_end {
        for entry in matrix.column_or_triplet_offsets[source_column]
            ..matrix.column_or_triplet_offsets[source_column + 1]
        {
            let row = matrix.row_indices[entry];
            if (row_start..=row_end).contains(&row) {
                row_indices.push(row - row_start);
                values.push(matrix.values[entry]);
            }
        }
        offsets.push(row_indices.len());
    }
    Some(SparseMatrix::new(
        rows,
        columns,
        row_indices.len(),
        -1,
        offsets,
        row_indices,
        values,
    ))
}

/// C++ `GetSubMatrixDiag(const cs *, ...)` with inclusive source bounds.
pub fn get_sub_matrix_diag(
    matrix: &SparseMatrix,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
) -> Option<Vec<f64>> {
    let submatrix = get_sub_matrix(matrix, row_start, row_end, column_start, column_end)?;
    if submatrix.rows != submatrix.columns {
        return None;
    }
    let mut diagonal = vec![0.0; submatrix.rows];
    for column in 0..submatrix.columns {
        for entry in submatrix.column_or_triplet_offsets[column]
            ..submatrix.column_or_triplet_offsets[column + 1]
        {
            if submatrix.row_indices[entry] == column {
                diagonal[column] = submatrix.values[entry];
            }
        }
    }
    Some(diagonal)
}

/// C++ `GetSubMatrix(..., CvMat *)` with inclusive source bounds.
pub fn get_sub_matrix_dense(
    matrix: &Matrix<f64>,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
) -> Option<Matrix<f64>> {
    if row_start > row_end
        || column_start > column_end
        || row_end >= matrix.rows
        || column_end >= matrix.columns
    {
        return None;
    }
    let rows = row_end - row_start + 1;
    let columns = column_end - column_start + 1;
    let mut values = Vec::with_capacity(rows * columns);
    for row in row_start..=row_end {
        values.extend_from_slice(
            &matrix.values
                [row * matrix.columns + column_start..row * matrix.columns + column_end + 1],
        );
    }
    Some(Matrix::new(rows, columns, values))
}

/// C++ `SparseDenseMult`.
pub fn sparse_dense_mult(matrix: &SparseMatrix, dense: &Matrix<f64>) -> Option<Matrix<f64>> {
    if matrix.nz != -1 || matrix.columns != dense.rows {
        return None;
    }
    let mut result = zero_mat(matrix.rows, dense.columns);
    for column in 0..matrix.columns {
        for entry in
            matrix.column_or_triplet_offsets[column]..matrix.column_or_triplet_offsets[column + 1]
        {
            for dense_column in 0..dense.columns {
                result.values[matrix.row_indices[entry] * dense.columns + dense_column] +=
                    matrix.values[entry] * dense.values[column * dense.columns + dense_column];
            }
        }
    }
    Some(result)
}

/// C++ `dotProduct`.
pub fn dot_product(dense: &Matrix<f64>, sparse: &SparseMatrix) -> Option<f64> {
    if dense.columns != 1 || sparse.nz != -1 || sparse.columns != 1 || sparse.rows != dense.rows {
        return None;
    }
    Some(
        (sparse.column_or_triplet_offsets[0]..sparse.column_or_triplet_offsets[1])
            .map(|entry| dense.values[sparse.row_indices[entry]] * sparse.values[entry])
            .sum(),
    )
}

/// C++ `print(CvMat *, ostream &)`, returned as owned text.
pub fn print_matrix(matrix: &Matrix<f64>) -> String {
    let mut output = String::new();
    for row in matrix.values.chunks(matrix.columns) {
        for value in row {
            output.push_str(&format!("{value}  "));
        }
        output.push('\n');
    }
    output.push('\n');
    output
}

/// C++ `print(cs *, ostream &)`, returned as owned text.
pub fn print_sparse(matrix: &SparseMatrix) -> Option<String> {
    if matrix.nz != -1 {
        return None;
    }
    let mut output = String::new();
    for column in 0..matrix.columns {
        for entry in
            matrix.column_or_triplet_offsets[column]..matrix.column_or_triplet_offsets[column + 1]
        {
            output.push_str(&format!(
                "{} {column} {}\n",
                matrix.row_indices[entry], matrix.values[entry]
            ));
        }
    }
    Some(output)
}

/// C++ `LPsolver`, using owned dense Newton systems in place of SuiteSparse's
/// temporary factorization objects.  `x0` is updated in place as in source.
pub fn lp_solver(
    cost: &SparseMatrix,
    constraints: &SparseMatrix,
    bounds: &Matrix<f64>,
    x0: &mut Matrix<f64>,
) -> Option<()> {
    if cost.nz != -1
        || constraints.nz != -1
        || bounds.columns != 1
        || x0.columns != 1
        || cost.columns != 1
        || cost.rows != x0.rows
        || constraints.columns != x0.rows
        || constraints.rows != bounds.rows
    {
        return None;
    }
    let mut slack = bounds.values.clone();
    let product = sparse_dense_mult(constraints, x0)?;
    for (value, product) in slack.iter_mut().zip(product.values) {
        *value -= product;
    }
    if min_elem(&Matrix::new(slack.len(), 1, slack.clone()))? < 0.0 {
        return None;
    }
    let mut barrier = 0.1;
    let mut iterations = 1;
    while x0.rows as f64 / barrier > 1.0e-3 && iterations < 200 {
        barrier *= 10.0;
        for _ in 0..20 {
            iterations += 1;
            let mut gradient = vec![0.0; x0.rows];
            for entry in cost.column_or_triplet_offsets[0]..cost.column_or_triplet_offsets[1] {
                gradient[cost.row_indices[entry]] = cost.values[entry] * barrier;
            }
            let mut hessian = vec![0.0; x0.rows * x0.rows];
            for constraint_row in 0..constraints.rows {
                let inverse = 1.0 / slack[constraint_row];
                let mut row = vec![0.0; x0.rows];
                for column in 0..constraints.columns {
                    for entry in constraints.column_or_triplet_offsets[column]
                        ..constraints.column_or_triplet_offsets[column + 1]
                    {
                        if constraints.row_indices[entry] == constraint_row {
                            row[column] += constraints.values[entry];
                            gradient[column] += constraints.values[entry] * inverse;
                        }
                    }
                }
                for left in 0..x0.rows {
                    for right in 0..x0.rows {
                        hessian[left * x0.rows + right] +=
                            row[left] * row[right] * inverse * inverse;
                    }
                }
            }
            let mut augmented = vec![0.0; x0.rows * (x0.rows + 1)];
            for row in 0..x0.rows {
                augmented[row * (x0.rows + 1)..row * (x0.rows + 1) + x0.rows]
                    .copy_from_slice(&hessian[row * x0.rows..(row + 1) * x0.rows]);
                augmented[row * (x0.rows + 1) + x0.rows] = -gradient[row];
            }
            for pivot in 0..x0.rows {
                let best = (pivot..x0.rows).max_by(|&left, &right| {
                    augmented[left * (x0.rows + 1) + pivot]
                        .abs()
                        .total_cmp(&augmented[right * (x0.rows + 1) + pivot].abs())
                })?;
                if augmented[best * (x0.rows + 1) + pivot].abs() <= f64::MIN_POSITIVE {
                    return None;
                }
                for column in pivot..=x0.rows {
                    augmented.swap(
                        pivot * (x0.rows + 1) + column,
                        best * (x0.rows + 1) + column,
                    );
                }
                let divisor = augmented[pivot * (x0.rows + 1) + pivot];
                for column in pivot..=x0.rows {
                    augmented[pivot * (x0.rows + 1) + column] /= divisor;
                }
                for row in 0..x0.rows {
                    if row != pivot {
                        let factor = augmented[row * (x0.rows + 1) + pivot];
                        for column in pivot..=x0.rows {
                            augmented[row * (x0.rows + 1) + column] -=
                                factor * augmented[pivot * (x0.rows + 1) + column];
                        }
                    }
                }
            }
            let direction: Vec<f64> = (0..x0.rows)
                .map(|row| augmented[row * (x0.rows + 1) + x0.rows])
                .collect();
            let decrement: f64 = direction.iter().zip(&gradient).map(|(a, b)| a * b).sum();
            if decrement.abs() < 1.0e-3 {
                break;
            }
            let mut step = 1.0;
            loop {
                let candidate: Vec<f64> = x0
                    .values
                    .iter()
                    .zip(&direction)
                    .map(|(x, dx)| x + step * dx)
                    .collect();
                let constraint_value =
                    sparse_dense_mult(constraints, &Matrix::new(x0.rows, 1, candidate.clone()))?;
                let candidate_slack: Vec<f64> = bounds
                    .values
                    .iter()
                    .zip(constraint_value.values)
                    .map(|(b, ax)| b - ax)
                    .collect();
                if candidate_slack.iter().all(|value| *value > 0.0) {
                    let candidate_objective = barrier
                        * dot_product(&Matrix::new(x0.rows, 1, candidate), cost)?
                        - sum_log(&Matrix::new(candidate_slack.len(), 1, candidate_slack))?;
                    let current_objective = barrier * dot_product(x0, cost)?
                        - sum_log(&Matrix::new(slack.len(), 1, slack.clone()))?;
                    if candidate_objective <= current_objective + 0.25 * step * decrement {
                        break;
                    }
                }
                step *= 0.5;
                if step < 1.0e-11 {
                    return Some(());
                }
            }
            for (value, direction) in x0.values.iter_mut().zip(direction) {
                *value += step * direction;
            }
            let updated = sparse_dense_mult(constraints, x0)?;
            for ((value, bound), product) in
                slack.iter_mut().zip(&bounds.values).zip(updated.values)
            {
                *value = bound - product;
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_source_operations_preserve_row_major_values() {
        let first = Matrix::new(1, 2, vec![1.0, 2.0]);
        let second = Matrix::new(2, 2, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            combined_x1_dx2(&first, &second).unwrap().values,
            vec![1., 2., 3., 4., 5., 6.]
        );
        assert_eq!(
            get_sub_matrix_dense(&second, 0, 1, 1, 1).unwrap().values,
            vec![4., 6.]
        );
        assert_eq!(
            sum_log(&Matrix::new(2, 1, vec![1.0, std::f64::consts::E])),
            Some(1.0)
        );
    }

    #[test]
    fn sparse_source_operations_use_compressed_columns() {
        let sparse = SparseMatrix::new(3, 2, 3, -1, vec![0, 2, 3], vec![0, 2, 1], vec![2., 4., 3.]);
        assert_eq!(
            sparse_dense_mult(&sparse, &Matrix::new(2, 1, vec![5., 7.]))
                .unwrap()
                .values,
            vec![10., 21., 20.]
        );
        assert_eq!(
            get_sub_matrix_diag(&sparse, 0, 1, 0, 1).unwrap(),
            vec![2., 3.]
        );
        assert_eq!(
            dot_product(
                &Matrix::new(3, 1, vec![2., 3., 4.]),
                &SparseMatrix::new(3, 1, 2, -1, vec![0, 2], vec![0, 2], vec![5., 6.])
            ),
            Some(34.)
        );
        assert_eq!(print_sparse(&sparse).unwrap(), "0 0 2\n2 0 4\n1 1 3\n");
    }

    #[test]
    fn lp_solver_finds_the_barrier_solution_for_a_bounded_linear_cost() {
        let cost = SparseMatrix::new(1, 1, 1, -1, vec![0, 1], vec![0], vec![-1.0]);
        let constraints = SparseMatrix::new(2, 1, 2, -1, vec![0, 2], vec![0, 1], vec![1.0, -1.0]);
        let mut point = Matrix::new(1, 1, vec![0.0]);
        lp_solver(
            &cost,
            &constraints,
            &Matrix::new(2, 1, vec![1.0, 1.0]),
            &mut point,
        )
        .unwrap();
        assert!((point.values[0] - 1.0).abs() < 2.0e-3, "{point:?}");
    }

    #[test]
    fn std_qp_finds_the_constrained_quadratic_minimum() {
        // min 1/2 x^2 - 2x, subject to -1 < x < 1: x tends to 1.
        let quadratic = SparseMatrix::new(1, 1, 1, -1, vec![0, 1], vec![0], vec![1.]);
        let cost = SparseMatrix::new(1, 1, 1, -1, vec![0, 1], vec![0], vec![-2.]);
        let constraints = SparseMatrix::new(2, 1, 2, -1, vec![0, 2], vec![0, 1], vec![1., -1.]);
        let result = std_qp(
            &quadratic,
            &cost,
            &constraints,
            &Matrix::new(2, 1, vec![1., 1.]),
            &Matrix::new(1, 1, vec![0.]),
            0,
            0,
            "probU",
        );
        assert_eq!(result.exit_flag, 0);
        assert!((result.answer_mat.unwrap().values[0] - 1.).abs() < 2e-3);
    }
}
