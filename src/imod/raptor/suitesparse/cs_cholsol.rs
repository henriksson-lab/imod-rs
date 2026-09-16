//! Translation of `IMOD/raptor/suitesparse/cs_cholsol.c`.

use super::Cs;
use super::cs_chol::cs_chol;
use super::cs_ipvec::cs_ipvec;
use super::cs_lsolve::cs_lsolve;
use super::cs_ltsolve::cs_ltsolve;
use super::cs_pvec::cs_pvec;
use super::cs_schol::cs_schol;

/// C `cs_cholsol`: overwrites `right_hand_side` with the SPD solution `A\b`.
pub fn cs_cholsol(order: i32, matrix: &Cs, right_hand_side: &mut [f64]) -> bool {
    if !matrix.is_csc() || right_hand_side.len() < matrix.columns {
        return false;
    }
    let Some(symbolic) = cs_schol(order, matrix) else {
        return false;
    };
    let Some(numeric) = cs_chol(matrix, &symbolic) else {
        return false;
    };
    let Some(factor) = numeric.l.as_ref() else {
        return false;
    };
    let mut workspace = vec![0.0; matrix.columns];
    if !cs_ipvec(symbolic.pinv.as_deref(), right_hand_side, &mut workspace)
        || !cs_lsolve(factor, &mut workspace)
        || !cs_ltsolve(factor, &mut workspace)
        || !cs_pvec(symbolic.pinv.as_deref(), &workspace, right_hand_side)
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::cs_cholsol;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn cholsol_solves_a_positive_definite_system() {
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 3],
            row_indices: vec![0, 0, 1],
            values: vec![4., 1., 3.],
            nz: -1,
        };
        let mut values = [6., 7.];
        assert!(cs_cholsol(0, &matrix, &mut values));
        assert!((values[0] - 1.0).abs() < 1.0e-12);
        assert!((values[1] - 2.0).abs() < 1.0e-12);
    }
}
