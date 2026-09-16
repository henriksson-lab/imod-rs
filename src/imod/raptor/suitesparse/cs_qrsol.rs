//! Translation of `IMOD/raptor/suitesparse/cs_qrsol.c`.

use super::{
    Cs, cs_happly::cs_happly, cs_qr::cs_qr, cs_sqr::cs_sqr, cs_transpose::cs_transpose,
    cs_usolve::cs_usolve, cs_utsolve::cs_utsolve,
};

/// C `cs_qrsol`: overwrites `right_hand_side` with the QR solution of `A x=b`.
///
/// For an overdetermined system the first `A.columns` entries receive the
/// least-squares solution.  For an underdetermined system all `A.columns`
/// entries receive the minimum-norm solution.
pub fn cs_qrsol(order: i32, matrix: &Cs, right_hand_side: &mut [f64]) -> bool {
    if !matrix.is_csc() || !(0..=3).contains(&order) {
        return false;
    }
    let rows = matrix.rows;
    let columns = matrix.columns;
    if rows >= columns {
        if right_hand_side.len() < rows {
            return false;
        }
        let Some(symbolic) = cs_sqr(order, matrix, true) else {
            return false;
        };
        let Some(numeric) = cs_qr(matrix, &symbolic) else {
            return false;
        };
        let (Some(inverse_rows), Some(reflectors), Some(upper), Some(beta)) = (
            symbolic.pinv.as_deref(),
            numeric.l.as_ref(),
            numeric.u.as_ref(),
            numeric.beta.as_deref(),
        ) else {
            return false;
        };
        let mut workspace = vec![0.0; symbolic.m2];
        if inverse_rows.len() < rows {
            return false;
        }
        for row in 0..rows {
            let destination = inverse_rows[row];
            if destination >= workspace.len() {
                return false;
            }
            workspace[destination] = right_hand_side[row];
        }
        for column in 0..columns {
            if !cs_happly(reflectors, column, beta[column], &mut workspace) {
                return false;
            }
        }
        if !cs_usolve(upper, &mut workspace) {
            return false;
        }
        if let Some(permutation) = symbolic.q.as_deref() {
            for column in 0..columns {
                right_hand_side[permutation[column]] = workspace[column];
            }
        } else {
            right_hand_side[..columns].copy_from_slice(&workspace[..columns]);
        }
        true
    } else {
        if right_hand_side.len() < columns {
            return false;
        }
        let Some(transpose) = cs_transpose(matrix, true) else {
            return false;
        };
        let Some(symbolic) = cs_sqr(order, &transpose, true) else {
            return false;
        };
        let Some(numeric) = cs_qr(&transpose, &symbolic) else {
            return false;
        };
        let (Some(inverse_rows), Some(reflectors), Some(upper), Some(beta)) = (
            symbolic.pinv.as_deref(),
            numeric.l.as_ref(),
            numeric.u.as_ref(),
            numeric.beta.as_deref(),
        ) else {
            return false;
        };
        let mut workspace = vec![0.0; symbolic.m2];
        if let Some(permutation) = symbolic.q.as_deref() {
            for row in 0..rows {
                workspace[row] = right_hand_side[permutation[row]];
            }
        } else {
            workspace[..rows].copy_from_slice(&right_hand_side[..rows]);
        }
        if !cs_utsolve(upper, &mut workspace) {
            return false;
        }
        for column in (0..rows).rev() {
            if !cs_happly(reflectors, column, beta[column], &mut workspace) {
                return false;
            }
        }
        for row in 0..columns {
            let source = inverse_rows[row];
            if source >= workspace.len() {
                return false;
            }
            right_hand_side[row] = workspace[source];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::cs_qrsol;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn qrsol_solves_tall_full_rank_systems() {
        let matrix = Cs {
            nzmax: 4,
            rows: 3,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 1, 2],
            values: vec![1.0, 1.0, 1.0, 1.0],
            nz: -1,
        };
        let mut right_hand_side = vec![1.0, 3.0, 2.0];
        assert!(cs_qrsol(0, &matrix, &mut right_hand_side));
        assert!((right_hand_side[0] - 1.0).abs() < 1.0e-12);
        assert!((right_hand_side[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn qrsol_solves_wide_systems_with_minimum_norm_solution() {
        let matrix = Cs {
            nzmax: 2,
            rows: 1,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![1.0, 1.0],
            nz: -1,
        };
        let mut right_hand_side = vec![2.0, 0.0];
        assert!(cs_qrsol(0, &matrix, &mut right_hand_side));
        assert!((right_hand_side[0] - 1.0).abs() < 1.0e-12);
        assert!((right_hand_side[1] - 1.0).abs() < 1.0e-12);
    }
}
