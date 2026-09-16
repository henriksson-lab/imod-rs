//! Translation of `IMOD/raptor/suitesparse/cs_schol.c`.

use super::cs_amd::cs_amd;
use super::cs_counts::cs_counts;
use super::cs_etree::cs_etree;
use super::cs_pinv::cs_pinv;
use super::cs_post::cs_post;
use super::cs_symperm::cs_symperm;
use super::{Cs, Css};

/// C `cs_schol`: ordering and symbolic analysis for a Cholesky factorization.
///
/// `order == 0` retains the matrix's natural ordering; orders 1 through 3 use
/// the source's approximate-minimum-degree ordering for their respective
/// Cholesky, LU, and QR graph constructions.
pub fn cs_schol(order: i32, matrix: &Cs) -> Option<Css> {
    if !matrix.is_csc() || matrix.rows != matrix.columns || order < 0 || order > 3 {
        return None;
    }
    let permutation = if order == 0 {
        None
    } else {
        Some(cs_amd(order, matrix)?)
    };
    let inverse_permutation = cs_pinv(permutation.as_deref());
    let symmetric = cs_symperm(matrix, inverse_permutation.as_deref(), false)?;
    let parent = cs_etree(&symmetric, false)?;
    let post = cs_post(&parent)?;
    let mut counts = cs_counts(&symmetric, &parent, &post, false)?;
    let mut column_pointers = vec![0; matrix.columns + 1];
    let mut total = 0_usize;
    for column in 0..matrix.columns {
        column_pointers[column] = total;
        total = total.checked_add(counts[column])?;
        counts[column] = column_pointers[column];
    }
    column_pointers[matrix.columns] = total;
    let nonzeros = total as f64;
    Some(Css {
        pinv: inverse_permutation,
        parent: Some(parent),
        cp: Some(column_pointers),
        lnz: nonzeros,
        unz: nonzeros,
        ..Css::default()
    })
}

#[cfg(test)]
mod tests {
    use super::cs_schol;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn symbolic_cholesky_uses_upper_triangle_and_natural_order() {
        // triu(A) has columns [0], [0, 1], [1, 2], whose etree is 0 -> 1 -> 2.
        let matrix = Cs {
            nzmax: 5,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 1, 3, 5],
            row_indices: vec![0, 0, 1, 1, 2],
            values: vec![4., 1., 3., 1., 2.],
            nz: -1,
        };
        let symbolic = cs_schol(0, &matrix).unwrap();
        assert_eq!(symbolic.pinv, None);
        assert_eq!(symbolic.parent, Some(vec![Some(1), Some(2), None]));
        assert_eq!(symbolic.cp, Some(vec![0, 2, 4, 5]));
        assert_eq!(symbolic.lnz, 5.0);
        assert_eq!(symbolic.unz, 5.0);
    }

    #[test]
    fn symbolic_cholesky_rejects_non_square_inputs() {
        let rectangular = Cs {
            nzmax: 1,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![1.],
            nz: -1,
        };
        assert_eq!(cs_schol(0, &rectangular), None);
        assert_eq!(cs_schol(1, &rectangular), None);
    }
}
