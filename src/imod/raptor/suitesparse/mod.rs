//! Safe owned translation of Raptor's bundled `suitesparse/cs.h` API.

pub mod cs_add;
pub mod cs_amd;
pub mod cs_chol;
pub mod cs_cholsol;
pub mod cs_compress;
pub mod cs_counts;
pub mod cs_cumsum;
pub mod cs_dfs;
pub mod cs_dmperm;
pub mod cs_droptol;
pub mod cs_dropzeros;
pub mod cs_dupl;
pub mod cs_entry;
pub mod cs_ereach;
pub mod cs_etree;
pub mod cs_fkeep;
pub mod cs_gaxpy;
pub mod cs_happly;
pub mod cs_house;
pub mod cs_ipvec;
pub mod cs_leaf;
pub mod cs_load;
pub mod cs_lsolve;
pub mod cs_ltsolve;
pub mod cs_lu;
pub mod cs_lusol;
pub mod cs_malloc;
pub mod cs_maxtrans;
pub mod cs_multiply;
pub mod cs_norm;
pub mod cs_permute;
pub mod cs_pinv;
pub mod cs_post;
pub mod cs_print;
pub mod cs_pvec;
pub mod cs_qr;
pub mod cs_qrsol;
pub mod cs_randperm;
pub mod cs_reach;
pub mod cs_scatter;
pub mod cs_scc;
pub mod cs_schol;
pub mod cs_spsolve;
pub mod cs_sqr;
pub mod cs_symperm;
pub mod cs_tdfs;
pub mod cs_transpose;
pub mod cs_updown;
pub mod cs_usolve;
pub mod cs_util;
pub mod cs_utsolve;

/// `cs_sparse` from `IMOD/raptor/suitesparse/cs.h`.
///
/// `column_pointers` is either CSC column offsets (`nz == -1`) or triplet
/// column indices (`nz >= 0`); all allocations are represented by vectors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cs {
    pub nzmax: usize,
    pub rows: usize,
    pub columns: usize,
    pub column_pointers: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub nz: isize,
}

/// `cs_symbolic` from `IMOD/raptor/suitesparse/cs.h`.
///
/// Symbolic analysis owns every permutation and factor-column offset.  Fields
/// not used by a particular factorization are absent rather than represented
/// by C null pointers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Css {
    pub pinv: Option<Vec<usize>>,
    pub q: Option<Vec<usize>>,
    pub parent: Option<Vec<Option<usize>>>,
    pub cp: Option<Vec<usize>>,
    pub leftmost: Option<Vec<usize>>,
    pub m2: usize,
    pub lnz: f64,
    pub unz: f64,
}

/// `cs_numeric` from `IMOD/raptor/suitesparse/cs.h`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csn {
    pub l: Option<Cs>,
    pub u: Option<Cs>,
    pub pinv: Option<Vec<usize>>,
    pub beta: Option<Vec<f64>>,
}

/// `cs_dmperm_results` from `IMOD/raptor/suitesparse/cs.h`.
///
/// `p` and `q` are the row and column permutations.  Consecutive entries in
/// `r` and `s` delimit the fine blocks; `rr` and `cc` retain the four coarse
/// Dulmage--Mendelsohn sets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csd {
    pub p: Vec<usize>,
    pub q: Vec<usize>,
    pub r: Vec<usize>,
    pub s: Vec<usize>,
    pub nb: usize,
    pub rr: [usize; 5],
    pub cc: [usize; 5],
}

impl Cs {
    pub fn is_csc(&self) -> bool {
        self.nz == -1
    }

    pub fn is_triplet(&self) -> bool {
        self.nz >= 0
    }
}

#[cfg(test)]
mod tests {
    use super::Cs;
    use super::cs_compress::cs_compress;
    use super::cs_cumsum::cs_cumsum;
    use super::cs_droptol::cs_droptol;
    use super::cs_dropzeros::cs_dropzeros;
    use super::cs_gaxpy::cs_gaxpy;
    use super::cs_happly::cs_happly;
    use super::cs_house::cs_house;
    use super::cs_lsolve::cs_lsolve;
    use super::cs_ltsolve::cs_ltsolve;
    use super::cs_norm::cs_norm;
    use super::cs_pinv::cs_pinv;
    use super::cs_transpose::cs_transpose;
    use super::cs_usolve::cs_usolve;
    use super::cs_utsolve::cs_utsolve;

    #[test]
    fn sparse_triangular_units_follow_csparse_column_order() {
        let lower = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![2.0, 3.0, 4.0],
            nz: -1,
        };
        let mut forward = vec![2.0, 11.0];
        assert!(cs_lsolve(&lower, &mut forward));
        assert_eq!(forward, [1.0, 2.0]);
        let mut transpose = vec![7.0, 8.0];
        assert!(cs_ltsolve(&lower, &mut transpose));
        assert_eq!(transpose, [0.5, 2.0]);

        let upper = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 3],
            row_indices: vec![0, 0, 1],
            values: vec![2.0, 3.0, 4.0],
            nz: -1,
        };
        let mut backward = vec![8.0, 8.0];
        assert!(cs_usolve(&upper, &mut backward));
        assert_eq!(backward, [1.0, 2.0]);
        let mut upper_transpose = vec![2.0, 11.0];
        assert!(cs_utsolve(&upper, &mut upper_transpose));
        assert_eq!(upper_transpose, [1.0, 2.0]);
        let mut product = vec![1.0, 1.0];
        assert!(cs_gaxpy(&upper, &[2.0, 3.0], &mut product));
        assert_eq!(product, [14.0, 13.0]);
        assert_eq!(cs_norm(&upper), Some(7.0));
    }

    #[test]
    fn sparse_utility_units_match_csparse_algorithms() {
        let mut offsets = [0; 4];
        let mut counts = [2, 0, 3];
        assert_eq!(cs_cumsum(&mut offsets, &mut counts), Some(5.0));
        assert_eq!(offsets, [0, 2, 2, 5]);
        assert_eq!(counts, [0, 2, 2]);
        assert_eq!(cs_pinv(Some(&[2, 0, 1])), Some(vec![1, 2, 0]));
        let mut house = [3.0, 4.0];
        let (length, beta) = cs_house(&mut house).unwrap();
        assert_eq!(length, 5.0);
        assert!((beta - 0.1).abs() < 1.0e-14);
        let reflector = Cs {
            nzmax: 2,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 2],
            row_indices: vec![0, 1],
            values: vec![1.0, 0.0],
            nz: -1,
        };
        let mut values = [3.0, 4.0];
        assert!(cs_happly(&reflector, 0, 2.0, &mut values));
        assert_eq!(values, [-3.0, 4.0]);
    }

    #[test]
    fn triplet_compression_and_transpose_preserve_csc_order() {
        let triplet = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![1, 0, 1],
            row_indices: vec![0, 1, 1],
            values: vec![4.0, 2.0, 3.0],
            nz: 3,
        };
        let compressed = cs_compress(&triplet).unwrap();
        assert_eq!(compressed.column_pointers, [0, 1, 3]);
        assert_eq!(compressed.row_indices, [1, 0, 1]);
        assert_eq!(compressed.values, [2.0, 4.0, 3.0]);
        let transposed = cs_transpose(&compressed, true).unwrap();
        assert_eq!(transposed.column_pointers, [0, 1, 3]);
        assert_eq!(transposed.row_indices, [1, 0, 1]);
        assert_eq!(transposed.values, [4.0, 2.0, 3.0]);
    }

    #[test]
    fn sparse_filtering_updates_csc_offsets_and_owned_storage() {
        let mut matrix = Cs {
            nzmax: 4,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 4],
            row_indices: vec![0, 1, 0, 1],
            values: vec![0., 2., -0.5, 3.],
            nz: -1,
        };
        assert_eq!(cs_dropzeros(&mut matrix), Some(3));
        assert_eq!(matrix.column_pointers, [0, 1, 3]);
        assert_eq!(matrix.values, [2., -0.5, 3.]);
        assert_eq!(cs_droptol(&mut matrix, 1.0), Some(2));
        assert_eq!(matrix.column_pointers, [0, 1, 2]);
        assert_eq!(matrix.values, [2., 3.]);
    }
}
