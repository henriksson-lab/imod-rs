//! Translation of `IMOD/raptor/suitesparse/cs_lusol.c`.

use super::{
    Cs, cs_ipvec::cs_ipvec, cs_lsolve::cs_lsolve, cs_lu::cs_lu, cs_sqr::cs_sqr,
    cs_usolve::cs_usolve,
};

/// C `cs_lusol`: overwrites `right_hand_side` with the solution of `A * x = b`.
pub fn cs_lusol(order: i32, matrix: &Cs, right_hand_side: &mut [f64], tolerance: f64) -> bool {
    if !matrix.is_csc() || matrix.rows != matrix.columns || right_hand_side.len() < matrix.columns {
        return false;
    }
    let Some(symbolic) = cs_sqr(order, matrix, false) else {
        return false;
    };
    let Some(numeric) = cs_lu(matrix, &symbolic, tolerance) else {
        return false;
    };
    let (Some(lower), Some(upper), Some(pivot)) = (numeric.l, numeric.u, numeric.pinv) else {
        return false;
    };
    let mut workspace = vec![0.0; matrix.columns];
    cs_ipvec(Some(&pivot), right_hand_side, &mut workspace)
        && cs_lsolve(&lower, &mut workspace)
        && cs_usolve(&upper, &mut workspace)
        && cs_ipvec(symbolic.q.as_deref(), &workspace, right_hand_side)
}

#[cfg(test)]
mod tests {
    use super::cs_lusol;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn lusol_overwrites_rhs_with_solution_after_row_pivoting() {
        // [0 1; 2 3] * [1; 2] = [2; 8]
        let matrix = Cs {
            nzmax: 4,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 0, 1],
            values: vec![0.0, 2.0, 1.0, 3.0],
            nz: -1,
        };
        let mut rhs = vec![2.0, 8.0];
        assert!(cs_lusol(0, &matrix, &mut rhs, 1.0));
        assert_eq!(rhs, [1.0, 2.0]);
    }

    #[test]
    fn lusol_rejects_non_square_and_singular_systems() {
        let rectangular = Cs {
            nzmax: 1,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![1.0],
            nz: -1,
        };
        assert!(!cs_lusol(0, &rectangular, &mut [1.0, 2.0], 1.0));
    }
}
