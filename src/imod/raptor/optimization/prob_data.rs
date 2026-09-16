//! Translation of `IMOD/raptor/optimization/probData.{h,cpp}`.

use super::std_qp_data::Matrix;

/// Owned replacement for SuiteSparse's `cs` sparse matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub rows: usize,
    pub columns: usize,
    pub nzmax: usize,
    /// `-1` denotes compressed-column storage, as in SuiteSparse.
    pub nz: isize,
    pub column_or_triplet_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        nzmax: usize,
        nz: isize,
        column_or_triplet_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        assert_eq!(row_indices.len(), nzmax);
        assert_eq!(values.len(), nzmax);
        assert_eq!(
            column_or_triplet_offsets.len(),
            if nz == -1 { columns + 1 } else { nzmax }
        );
        Self {
            rows,
            columns,
            nzmax,
            nz,
            column_or_triplet_offsets,
            row_indices,
            values,
        }
    }
}

/// C++ `probData`, represented entirely by owned native data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProbData {
    pub buc: Option<Matrix<f64>>,
    pub a: Option<SparseMatrix>,
    pub c: Option<SparseMatrix>,
    pub q: Option<SparseMatrix>,
    pub x0: Option<Matrix<f64>>,
}

impl ProbData {
    /// C++ `probData::probData()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `probData::CopyCS`; `Clone` performs the source's deep copy.
    pub fn copy_sparse(matrix: Option<&SparseMatrix>) -> Option<SparseMatrix> {
        matrix.cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::{ProbData, SparseMatrix};
    use crate::imod::raptor::optimization::std_qp_data::Matrix;

    #[test]
    fn default_has_no_owned_payloads() {
        let data = ProbData::new();
        assert!(
            data.buc.is_none()
                && data.a.is_none()
                && data.c.is_none()
                && data.q.is_none()
                && data.x0.is_none()
        );
    }

    #[test]
    fn sparse_copy_is_a_deep_native_copy() {
        let matrix = SparseMatrix::new(2, 1, 2, -1, vec![0, 2], vec![0, 1], vec![3.0, 4.0]);
        let mut copied = ProbData::copy_sparse(Some(&matrix)).unwrap();
        copied.values[0] = 9.0;
        assert_eq!(matrix.values, vec![3.0, 4.0]);
        let data = ProbData {
            buc: Some(Matrix::new(1, 1, vec![2.0])),
            a: Some(matrix),
            ..ProbData::new()
        };
        let mut cloned = data.clone();
        cloned.buc.as_mut().unwrap().values[0] = 7.0;
        assert_eq!(data.buc.unwrap().values[0], 2.0);
    }
}
