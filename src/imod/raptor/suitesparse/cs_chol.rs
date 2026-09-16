//! Translation of `IMOD/raptor/suitesparse/cs_chol.c`.

use super::cs_ereach::cs_ereach;
use super::cs_symperm::cs_symperm;
use super::{Cs, Csn, Css};

/// C `cs_chol`: computes `L` such that `L * L' = A` from symbolic analysis.
///
/// The input stores its upper triangle, as required by CSparse.  A failed
/// positive-definiteness test returns `None`, matching C's null result.
pub fn cs_chol(matrix: &Cs, symbolic: &Css) -> Option<Csn> {
    if !matrix.is_csc()
        || matrix.rows != matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return None;
    }
    let column_pointers = symbolic.cp.as_ref()?;
    let parent = symbolic.parent.as_ref()?;
    let n = matrix.columns;
    if column_pointers.len() != n + 1
        || parent.len() != n
        || column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return None;
    }
    let c = if let Some(inverse_permutation) = symbolic.pinv.as_deref() {
        cs_symperm(matrix, Some(inverse_permutation), true)?
    } else {
        matrix.clone()
    };
    let entries = c.column_pointers[n];
    if entries > c.row_indices.len() || entries > c.values.len() {
        return None;
    }
    let capacity = column_pointers[n];
    let mut factor = Cs {
        nzmax: capacity,
        rows: n,
        columns: n,
        column_pointers: column_pointers.clone(),
        row_indices: vec![0; capacity],
        values: vec![0.0; capacity],
        nz: -1,
    };
    let mut next = column_pointers[..n].to_vec();
    let mut workspace = vec![0.0; n];
    for column in 0..n {
        let pattern = cs_ereach(&c, column, parent)?;
        workspace[column] = 0.0;
        for entry in c.column_pointers[column]..c.column_pointers[column + 1] {
            let row = c.row_indices[entry];
            if row >= n {
                return None;
            }
            if row <= column {
                workspace[row] = c.values[entry];
            }
        }
        let mut diagonal = workspace[column];
        workspace[column] = 0.0;
        for row in pattern {
            if row >= column || factor.column_pointers[row] >= factor.values.len() {
                return None;
            }
            let diagonal_index = factor.column_pointers[row];
            let lki = workspace[row] / factor.values[diagonal_index];
            workspace[row] = 0.0;
            for entry in diagonal_index + 1..next[row] {
                let target = factor.row_indices[entry];
                if target >= n {
                    return None;
                }
                workspace[target] -= factor.values[entry] * lki;
            }
            diagonal -= lki * lki;
            let entry = next[row];
            if entry >= capacity {
                return None;
            }
            factor.row_indices[entry] = column;
            factor.values[entry] = lki;
            next[row] += 1;
        }
        if diagonal <= 0.0 {
            return None;
        }
        let entry = next[column];
        if entry >= capacity {
            return None;
        }
        factor.row_indices[entry] = column;
        factor.values[entry] = diagonal.sqrt();
        next[column] += 1;
    }
    factor.column_pointers[n] = capacity;
    Some(Csn {
        l: Some(factor),
        ..Csn::default()
    })
}

#[cfg(test)]
mod tests {
    use super::cs_chol;
    use crate::imod::raptor::suitesparse::{Cs, cs_schol::cs_schol};

    #[test]
    fn numeric_cholesky_matches_known_lower_factor() {
        // Upper triangle of [[4, 2], [2, 10]], whose L is [[2, 0], [1, 3]].
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 3],
            row_indices: vec![0, 0, 1],
            values: vec![4.0, 2.0, 10.0],
            nz: -1,
        };
        let symbolic = cs_schol(0, &matrix).unwrap();
        let factor = cs_chol(&matrix, &symbolic).unwrap().l.unwrap();
        assert_eq!(factor.column_pointers, vec![0, 2, 3]);
        assert_eq!(factor.row_indices, vec![0, 1, 1]);
        assert_eq!(factor.values, vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn numeric_cholesky_rejects_indefinite_matrix() {
        let matrix = Cs {
            nzmax: 2,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![0, 1],
            values: vec![1.0, -1.0],
            nz: -1,
        };
        let symbolic = cs_schol(0, &matrix).unwrap();
        assert_eq!(cs_chol(&matrix, &symbolic), None);
    }
}
