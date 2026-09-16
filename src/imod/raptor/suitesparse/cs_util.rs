//! Translation of `IMOD/raptor/suitesparse/cs_util.c`.

use super::{Cs, Csd, Csn, Css};

/// C `cs_spalloc`: allocates a sparse matrix in triplet or CSC form.
pub fn cs_spalloc(rows: usize, columns: usize, nzmax: usize, values: bool, triplet: bool) -> Cs {
    let nzmax = nzmax.max(1);
    Cs {
        nzmax,
        rows,
        columns,
        column_pointers: vec![0; if triplet { nzmax } else { columns + 1 }],
        row_indices: vec![0; nzmax],
        values: values.then(|| vec![0.0; nzmax]).unwrap_or_default(),
        nz: if triplet { 0 } else { -1 },
    }
}

/// C `cs_sprealloc`: changes a matrix's allocated entry capacity.
pub fn cs_sprealloc(matrix: &mut Cs, requested_capacity: usize) -> bool {
    let capacity = if requested_capacity == 0 {
        if matrix.is_csc() {
            let Some(&entries) = matrix.column_pointers.get(matrix.columns) else {
                return false;
            };
            entries
        } else if matrix.nz >= 0 {
            matrix.nz as usize
        } else {
            return false;
        }
    } else {
        requested_capacity
    };
    let allocated_capacity = capacity.max(1);
    matrix.row_indices.resize(allocated_capacity, 0);
    if matrix.is_triplet() {
        matrix.column_pointers.resize(allocated_capacity, 0);
    }
    if !matrix.values.is_empty() {
        matrix.values.resize(allocated_capacity, 0.0);
    }
    matrix.nzmax = capacity;
    true
}

/// C `cs_spfree`: drops a sparse matrix and yields null.
pub fn cs_spfree(matrix: Option<Cs>) -> Option<Cs> {
    drop(matrix);
    None
}

/// C `cs_nfree`: drops a numeric factorization and yields null.
pub fn cs_nfree(numeric: Option<Csn>) -> Option<Csn> {
    drop(numeric);
    None
}

/// C `cs_sfree`: drops a symbolic factorization and yields null.
pub fn cs_sfree(symbolic: Option<Css>) -> Option<Css> {
    drop(symbolic);
    None
}

/// C `cs_dalloc`: creates a zeroed DMperm/SCC result allocation.
pub fn cs_dalloc(rows: usize, columns: usize) -> Csd {
    Csd {
        p: vec![0; rows],
        q: vec![0; columns],
        r: vec![0; rows + 6],
        s: vec![0; columns + 6],
        nb: 0,
        rr: [0; 5],
        cc: [0; 5],
    }
}

/// C `cs_dfree`: drops a DMperm/SCC result and yields null.
pub fn cs_dfree(result: Option<Csd>) -> Option<Csd> {
    drop(result);
    None
}

/// C `cs_done`: returns a sparse result only when its operation succeeded.
/// Workspace needs no explicit handling because Rust drops it at scope exit.
pub fn cs_done(matrix: Option<Cs>, ok: bool) -> Option<Cs> {
    ok.then_some(matrix).flatten()
}

/// C `cs_idone`: returns an index-vector result only when its operation
/// succeeded.  The temporary matrix is consumed independently of `ok`.
pub fn cs_idone(
    permutation: Option<Vec<usize>>,
    temporary_matrix: Option<Cs>,
    ok: bool,
) -> Option<Vec<usize>> {
    drop(temporary_matrix);
    ok.then_some(permutation).flatten()
}

/// C `cs_ndone`: returns a numeric factorization only when its operation
/// succeeded.  Temporary owned values are naturally dropped.
pub fn cs_ndone(numeric: Option<Csn>, temporary_matrix: Option<Cs>, ok: bool) -> Option<Csn> {
    drop(temporary_matrix);
    ok.then_some(numeric).flatten()
}

/// C `cs_ddone`: returns a DMperm/SCC result only when its operation
/// succeeded.  The temporary matrix is naturally dropped.
pub fn cs_ddone(result: Option<Csd>, temporary_matrix: Option<Cs>, ok: bool) -> Option<Csd> {
    drop(temporary_matrix);
    ok.then_some(result).flatten()
}

#[cfg(test)]
mod tests {
    use super::{cs_dalloc, cs_done, cs_idone, cs_ndone, cs_spalloc, cs_sprealloc};
    use crate::imod::raptor::suitesparse::Csn;

    #[test]
    fn sparse_allocation_preserves_c_sparse_layouts() {
        let triplet = cs_spalloc(2, 3, 0, true, true);
        assert_eq!(triplet.nzmax, 1);
        assert_eq!(triplet.nz, 0);
        assert_eq!(triplet.column_pointers, [0]);
        assert_eq!(triplet.row_indices, [0]);
        assert_eq!(triplet.values, [0.0]);

        let csc = cs_spalloc(2, 3, 4, false, false);
        assert_eq!(csc.nz, -1);
        assert_eq!(csc.column_pointers, [0, 0, 0, 0]);
        assert_eq!(csc.row_indices, [0, 0, 0, 0]);
        assert!(csc.values.is_empty());
    }

    #[test]
    fn sparse_reallocation_uses_current_entry_count_for_zero_request() {
        let mut matrix = cs_spalloc(3, 2, 5, true, false);
        matrix.column_pointers.copy_from_slice(&[0, 1, 3]);
        assert!(cs_sprealloc(&mut matrix, 0));
        assert_eq!(matrix.nzmax, 3);
        assert_eq!(matrix.row_indices.len(), 3);
        assert_eq!(matrix.values.len(), 3);
        assert_eq!(matrix.column_pointers, [0, 1, 3]);
    }

    #[test]
    fn utility_results_follow_c_success_and_failure_ownership() {
        let matrix = cs_spalloc(0, 0, 0, false, false);
        assert!(cs_done(Some(matrix.clone()), true).is_some());
        assert_eq!(cs_done(Some(matrix), false), None);
        assert_eq!(cs_idone(Some(vec![2, 0]), None, true), Some(vec![2, 0]));
        assert_eq!(cs_ndone(Some(Csn::default()), None, false), None);
        let result = cs_dalloc(2, 3);
        assert_eq!(result.p.len(), 2);
        assert_eq!(result.q.len(), 3);
        assert_eq!(result.r.len(), 8);
        assert_eq!(result.s.len(), 9);
    }
}
