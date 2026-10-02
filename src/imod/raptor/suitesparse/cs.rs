//! Translation of `IMOD/raptor/suitesparse/cs.h`: the CSparse data
//! structures and macros RAPTOR reaches.

/// `cs` (`struct cs_sparse`): a matrix in compressed-column or triplet form.
#[derive(Clone, Debug)]
pub struct Cs {
    /// Maximum number of entries.
    pub nzmax: i32,
    /// Number of rows.
    pub m: i32,
    /// Number of columns.
    pub n: i32,
    /// Column pointers (size n+1) or col indices (size nzmax).
    pub p: Vec<i32>,
    /// Row indices, size nzmax.
    pub i: Vec<i32>,
    /// Numerical values, size nzmax (`NULL` for a pattern-only matrix).
    pub x: Option<Vec<f64>>,
    /// # of entries in triplet matrix, -1 for compressed-col.
    pub nz: i32,
}

/// `css` (`struct cs_symbolic`): symbolic Cholesky, LU, or QR analysis.
#[derive(Clone, Debug, Default)]
pub struct Css {
    /// Inverse row perm. for QR, fill red. perm for Chol.
    pub pinv: Option<Vec<i32>>,
    /// Fill-reducing column permutation for LU and QR.
    pub q: Option<Vec<i32>>,
    /// Elimination tree for Cholesky and QR.
    pub parent: Option<Vec<i32>>,
    /// Column pointers for Cholesky, row counts for QR.
    pub cp: Option<Vec<i32>>,
    /// leftmost[i] = min(find(A(i,:))), for QR.
    pub leftmost: Option<Vec<i32>>,
    /// # of rows for QR, after adding fictitious rows.
    pub m2: i32,
    /// # entries in L for LU or Cholesky; in V for QR.
    pub lnz: f64,
    /// # entries in U for LU; in R for QR.
    pub unz: f64,
}

/// `csn` (`struct cs_numeric`): numeric Cholesky, LU, or QR factorization.
#[derive(Clone, Debug, Default)]
pub struct Csn {
    /// L for LU and Cholesky, V for QR.
    pub l: Option<Cs>,
    /// U for LU, R for QR, not used for Cholesky.
    pub u: Option<Cs>,
    /// Partial pivoting for LU.
    pub pinv: Option<Vec<i32>>,
    /// beta [0..n-1] for QR.
    pub b: Option<Vec<f64>>,
}

/// `CS_MAX(a,b)`.
pub fn cs_max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// `CS_MIN(a,b)`.
pub fn cs_min(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

/// `CS_FLIP(i)`.
pub fn cs_flip(i: i32) -> i32 {
    -i - 2
}

/// `CS_UNFLIP(i)`.
pub fn cs_unflip(i: i32) -> i32 {
    if i < 0 { cs_flip(i) } else { i }
}

/// `CS_MARKED(w,j)`.
pub fn cs_marked(w: &[i32], j: i32) -> bool {
    w[j as usize] < 0
}

/// `CS_MARK(w,j)`.
pub fn cs_mark(w: &mut [i32], j: i32) {
    w[j as usize] = cs_flip(w[j as usize]);
}

/// `CS_CSC(A)`.
pub fn cs_csc(a: &Cs) -> bool {
    a.nz == -1
}

/// `CS_TRIPLET(A)`.
pub fn cs_triplet(a: &Cs) -> bool {
    a.nz >= 0
}
