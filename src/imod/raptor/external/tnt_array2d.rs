//! Translation of `IMOD/raptor/external/tnt_array2d.h` (Template Numerical
//! Toolkit `Array2D<T>`), as `MarkersCorrespond` instantiates it:
//! `TNT::Array2D<double>`, the `gglMatrix` of
//! `svlMarkerCorrespondenceLBModel.h`.
//!
//! TNT's `Array2D` is a reference-counted *view*: copying one shares the
//! data (`tnt_i_refvec.h`).  `MarkersCorrespond` never writes through a
//! copy after making it, so owned storage with `Clone` behaves the same.
//! `Array2D(m, n)` leaves the doubles uninitialised (`new T[m*n]` in
//! `i_refvec`); here they start at zero.  Every element the program reads
//! is written first, except where an input read fails at the end of a file,
//! and there the C++ value is heap residue.

/// `TNT::Array2D<double>`, row-major (`v_[i]` points at row `i`).
#[derive(Clone, Debug, Default)]
pub struct Array2D {
    data: Vec<f64>,
    m: i32,
    n: i32,
}

impl Array2D {
    /// `Array2D(int m, int n)` (`tnt_array2d.h:92`).
    pub fn new(m: i32, n: i32) -> Array2D {
        let size = if m > 0 && n > 0 { (m * n) as usize } else { 0 };
        Array2D {
            data: vec![0.0; size],
            m,
            n,
        }
    }

    /// `dim1()`: number of rows.
    pub fn dim1(&self) -> i32 {
        self.m
    }

    /// `dim2()`: number of columns.
    pub fn dim2(&self) -> i32 {
        self.n
    }
}

/// `operator[](int i)`: row `i`.
impl std::ops::Index<usize> for Array2D {
    type Output = [f64];
    fn index(&self, i: usize) -> &[f64] {
        let n = self.n as usize;
        &self.data[i * n..(i + 1) * n]
    }
}

impl std::ops::IndexMut<usize> for Array2D {
    fn index_mut(&mut self, i: usize) -> &mut [f64] {
        let n = self.n as usize;
        &mut self.data[i * n..(i + 1) * n]
    }
}
