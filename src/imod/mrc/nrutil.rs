//! Translation of `IMOD/mrc/nrutil.{c,h}`.
//!
//! The Numerical Recipes allocators hand back a pointer biased by the caller's
//! lower subscript bound, so that `m[nrl][ncl]` addresses the first element of
//! a `malloc`ed block.  The bias is pointer arithmetic, not behaviour: what a
//! caller observes is a rectangular block with subscripts running `nl..=nh`.
//! Each allocator therefore becomes an owned value of the same extent, keeping
//! the lower bounds as fields so the subscripts a translated caller writes are
//! the subscripts the source writes.  The blocks are contiguous and row-major,
//! exactly as `matrix()`/`f3tensor()` lay them out.
//!
//! One deliberate difference, per `NATIVE.md` §4: the source's `malloc` leaves
//! the block uninitialised and these constructors zero it.  Every call site in
//! `preNAD.cpp`, `preNID.cpp` and `nad_eed_3d.c` writes an element before
//! reading it, with the one exception noted in `pre_nid.rs` at
//! `CreateMaskedLocalSmooth`'s `mask`, where the zero fill is what a fresh
//! allocation gives the reference binary too.

use std::ops::{Index, IndexMut};

/// `nrerror`.
///
/// The source writes three lines on stderr and calls `exit(1)`.
pub fn nrerror(error_text: &str) -> ! {
    eprintln!("Numerical Recipes run-time error...");
    eprintln!("{error_text}");
    eprintln!("...now exiting to system...");
    std::process::exit(1)
}

/// A Numerical Recipes vector `v[nl..nh]`.
#[derive(Clone, Debug, PartialEq)]
pub struct NrVector<T> {
    pub lower: isize,
    pub values: Vec<T>,
}

impl<T: Default + Clone> NrVector<T> {
    pub fn new(nl: isize, nh: isize) -> Self {
        Self {
            lower: nl,
            values: vec![T::default(); (nh - nl + 1) as usize],
        }
    }
}

impl<T> NrVector<T> {
    pub fn get(&self, index: isize) -> &T {
        &self.values[(index - self.lower) as usize]
    }
    pub fn get_mut(&mut self, index: isize) -> &mut T {
        &mut self.values[(index - self.lower) as usize]
    }
}

impl<T> Index<usize> for NrVector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.values[(index as isize - self.lower) as usize]
    }
}

impl<T> IndexMut<usize> for NrVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let at = (index as isize - self.lower) as usize;
        &mut self.values[at]
    }
}

/// `vector`: a float vector with subscript range `v[nl..nh]`.
pub fn vector(nl: isize, nh: isize) -> NrVector<f32> {
    NrVector::new(nl, nh)
}

/// `ivector`: an int vector with subscript range `v[nl..nh]`.
pub fn ivector(nl: isize, nh: isize) -> NrVector<i32> {
    NrVector::new(nl, nh)
}

/// `cvector`: an unsigned char vector with subscript range `v[nl..nh]`.
pub fn cvector(nl: isize, nh: isize) -> NrVector<u8> {
    NrVector::new(nl, nh)
}

/// `lvector`: an unsigned long vector with subscript range `v[nl..nh]`.
pub fn lvector(nl: isize, nh: isize) -> NrVector<u64> {
    NrVector::new(nl, nh)
}

/// `dvector`: a double vector with subscript range `v[nl..nh]`.
pub fn dvector(nl: isize, nh: isize) -> NrVector<f64> {
    NrVector::new(nl, nh)
}

/// `free_vector`; dropping the owned value releases the source allocation.
pub fn free_vector(_v: NrVector<f32>, _nl: isize, _nh: isize) {}

/// `free_ivector`; dropping the owned value releases the source allocation.
pub fn free_ivector(_v: NrVector<i32>, _nl: isize, _nh: isize) {}

/// `free_cvector`; dropping the owned value releases the source allocation.
pub fn free_cvector(_v: NrVector<u8>, _nl: isize, _nh: isize) {}

/// `free_lvector`; dropping the owned value releases the source allocation.
pub fn free_lvector(_v: NrVector<u64>, _nl: isize, _nh: isize) {}

/// `free_dvector`; dropping the owned value releases the source allocation.
pub fn free_dvector(_v: NrVector<f64>, _nl: isize, _nh: isize) {}

/// A Numerical Recipes matrix `m[nrl..nrh][ncl..nch]`, one contiguous
/// row-major block as `matrix()` allocates it.
#[derive(Clone, Debug, PartialEq)]
pub struct NrMatrix<T> {
    pub row_lower: isize,
    pub column_lower: isize,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<T>,
}

impl<T: Default + Clone> NrMatrix<T> {
    pub fn new(nrl: isize, nrh: isize, ncl: isize, nch: isize) -> Self {
        let rows = (nrh - nrl + 1) as usize;
        let columns = (nch - ncl + 1) as usize;
        Self {
            row_lower: nrl,
            column_lower: ncl,
            rows,
            columns,
            values: vec![T::default(); rows * columns],
        }
    }
}

impl<T> NrMatrix<T> {
    pub fn get(&self, row: isize, column: isize) -> &T {
        &self.values
            [(row - self.row_lower) as usize * self.columns + (column - self.column_lower) as usize]
    }
    pub fn get_mut(&mut self, row: isize, column: isize) -> &mut T {
        let at =
            (row - self.row_lower) as usize * self.columns + (column - self.column_lower) as usize;
        &mut self.values[at]
    }
}

impl<T> Index<(usize, usize)> for NrMatrix<T> {
    type Output = T;
    fn index(&self, (row, column): (usize, usize)) -> &T {
        &self.values[(row as isize - self.row_lower) as usize * self.columns
            + (column as isize - self.column_lower) as usize]
    }
}

impl<T> IndexMut<(usize, usize)> for NrMatrix<T> {
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut T {
        let at = (row as isize - self.row_lower) as usize * self.columns
            + (column as isize - self.column_lower) as usize;
        &mut self.values[at]
    }
}

/// `matrix`: a float matrix with subscript range `m[nrl..nrh][ncl..nch]`.
pub fn matrix(nrl: isize, nrh: isize, ncl: isize, nch: isize) -> NrMatrix<f32> {
    NrMatrix::new(nrl, nrh, ncl, nch)
}

/// `dmatrix`: a double matrix with subscript range `m[nrl..nrh][ncl..nch]`.
pub fn dmatrix(nrl: isize, nrh: isize, ncl: isize, nch: isize) -> NrMatrix<f64> {
    NrMatrix::new(nrl, nrh, ncl, nch)
}

/// `imatrix`: an int matrix with subscript range `m[nrl..nrh][ncl..nch]`.
pub fn imatrix(nrl: isize, nrh: isize, ncl: isize, nch: isize) -> NrMatrix<i32> {
    NrMatrix::new(nrl, nrh, ncl, nch)
}

/// `free_matrix`; dropping the owned value releases the source allocation.
pub fn free_matrix(_m: NrMatrix<f32>, _nrl: isize, _nrh: isize, _ncl: isize, _nch: isize) {}

/// `free_dmatrix`; dropping the owned value releases the source allocation.
pub fn free_dmatrix(_m: NrMatrix<f64>, _nrl: isize, _nrh: isize, _ncl: isize, _nch: isize) {}

/// `free_imatrix`; dropping the owned value releases the source allocation.
pub fn free_imatrix(_m: NrMatrix<i32>, _nrl: isize, _nrh: isize, _ncl: isize, _nch: isize) {}

/// `submatrix`: point a submatrix `[newrl..][newcl..]` at
/// `a[oldrl..oldrh][oldcl..oldch]`.
///
/// The source returns a fresh row-pointer array aliasing `a`'s rows; an owned
/// translation copies the addressed region instead, so the element values and
/// subscript range a caller sees are the same.
pub fn submatrix(
    a: &NrMatrix<f32>,
    oldrl: isize,
    oldrh: isize,
    oldcl: isize,
    oldch: isize,
    newrl: isize,
    newcl: isize,
) -> NrMatrix<f32> {
    let mut m = NrMatrix::new(newrl, newrl + oldrh - oldrl, newcl, newcl + oldch - oldcl);
    for row in oldrl..=oldrh {
        for column in oldcl..=oldch {
            *m.get_mut(newrl + row - oldrl, newcl + column - oldcl) = *a.get(row, column);
        }
    }
    m
}

/// `free_submatrix`; the owned copy is released with its value.
pub fn free_submatrix(_b: NrMatrix<f32>, _nrl: isize, _nrh: isize, _ncl: isize, _nch: isize) {}

/// `convert_matrix`: view a C-declared `a[nrow][ncol]` as a matrix with
/// subscript range `m[nrl..nrh][ncl..nch]`.
pub fn convert_matrix(a: &[f32], nrl: isize, nrh: isize, ncl: isize, nch: isize) -> NrMatrix<f32> {
    let nrow = (nrh - nrl + 1) as usize;
    let ncol = (nch - ncl + 1) as usize;
    let mut m = NrMatrix::new(nrl, nrh, ncl, nch);
    m.values[..nrow * ncol].copy_from_slice(&a[..nrow * ncol]);
    m
}

/// `free_convert_matrix`; the owned copy is released with its value.
pub fn free_convert_matrix(_b: NrMatrix<f32>, _nrl: isize, _nrh: isize, _ncl: isize, _nch: isize) {}

/// A Numerical Recipes 3-tensor `t[nrl..nrh][ncl..nch][ndl..ndh]`, one
/// contiguous block ordered as `f3tensor()` lays it out.
#[derive(Clone, Debug, PartialEq)]
pub struct NrTensor3<T> {
    pub row_lower: isize,
    pub column_lower: isize,
    pub depth_lower: isize,
    pub rows: usize,
    pub columns: usize,
    pub depths: usize,
    pub values: Vec<T>,
}

impl<T> NrTensor3<T> {
    pub fn get(&self, row: isize, column: isize, depth: isize) -> &T {
        &self.values[((row - self.row_lower) as usize * self.columns
            + (column - self.column_lower) as usize)
            * self.depths
            + (depth - self.depth_lower) as usize]
    }
    pub fn get_mut(&mut self, row: isize, column: isize, depth: isize) -> &mut T {
        let at = ((row - self.row_lower) as usize * self.columns
            + (column - self.column_lower) as usize)
            * self.depths
            + (depth - self.depth_lower) as usize;
        &mut self.values[at]
    }
}

impl<T> Index<(usize, usize, usize)> for NrTensor3<T> {
    type Output = T;
    fn index(&self, (row, column, depth): (usize, usize, usize)) -> &T {
        &self.values[((row as isize - self.row_lower) as usize * self.columns
            + (column as isize - self.column_lower) as usize)
            * self.depths
            + (depth as isize - self.depth_lower) as usize]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for NrTensor3<T> {
    fn index_mut(&mut self, (row, column, depth): (usize, usize, usize)) -> &mut T {
        let at = ((row as isize - self.row_lower) as usize * self.columns
            + (column as isize - self.column_lower) as usize)
            * self.depths
            + (depth as isize - self.depth_lower) as usize;
        &mut self.values[at]
    }
}

/// `f3tensor`: a float 3-tensor with range `t[nrl..nrh][ncl..nch][ndl..ndh]`.
pub fn f3tensor(
    nrl: isize,
    nrh: isize,
    ncl: isize,
    nch: isize,
    ndl: isize,
    ndh: isize,
) -> NrTensor3<f32> {
    let rows = (nrh - nrl + 1) as usize;
    let columns = (nch - ncl + 1) as usize;
    let depths = (ndh - ndl + 1) as usize;
    NrTensor3 {
        row_lower: nrl,
        column_lower: ncl,
        depth_lower: ndl,
        rows,
        columns,
        depths,
        values: vec![0.; rows * columns * depths],
    }
}

/// `free_f3tensor`; dropping the owned value releases every source allocation.
pub fn free_f3tensor(
    _t: NrTensor3<f32>,
    _nrl: isize,
    _nrh: isize,
    _ncl: isize,
    _nch: isize,
    _ndl: isize,
    _ndh: isize,
) {
}

/// `nrutil.h`'s `SQR` macro.
pub fn sqr(a: f32) -> f32 {
    if a == 0.0 { 0.0 } else { a * a }
}

/// `nrutil.h`'s `DSQR` macro.
pub fn dsqr(a: f64) -> f64 {
    if a == 0.0 { 0.0 } else { a * a }
}

/// `nrutil.h`'s `SIGN` macro.
pub fn sign(a: f32, b: f32) -> f32 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_vector_matrix_and_tensor_keep_source_subscripts() {
        let mut v = vector(-2, 1);
        *v.get_mut(-2) = 7.;
        assert_eq!(*v.get(-2), 7.);
        let mut m = matrix(0, 2, 0, 1);
        m[(2, 1)] = 3.;
        assert_eq!(m[(2, 1)], 3.);
        assert_eq!(m.values.len(), 6);
        let mut t = f3tensor(0, 1, 0, 1, 0, 1);
        t[(1, 1, 1)] = 9.;
        assert_eq!(t.values[7], 9.);
    }

    #[test]
    fn integer_and_double_allocators_exist() {
        let mut d = dmatrix(0, 1, 0, 1);
        d[(1, 1)] = 2.5;
        assert_eq!(d[(1, 1)], 2.5);
        let mut i = imatrix(0, 1, 0, 1);
        i[(0, 1)] = -4;
        assert_eq!(i[(0, 1)], -4);
        let mut c = cvector(0, 3);
        c[2] = 255;
        assert_eq!(c[2], 255);
        let mut l = lvector(0, 1);
        l[1] = 1 << 40;
        assert_eq!(l[1], 1 << 40);
        let mut dv = dvector(0, 1);
        dv[0] = 1.5;
        assert_eq!(dv[0], 1.5);
    }

    #[test]
    fn submatrix_and_convert_matrix_address_the_same_elements() {
        let mut a = matrix(1, 2, 1, 2);
        a.values.copy_from_slice(&[1., 2., 3., 4.]);
        let b = submatrix(&a, 1, 2, 1, 2, 4, 5);
        assert_eq!(*b.get(5, 6), 4.);
        let c = convert_matrix(&[1., 2., 3., 4.], 0, 1, 0, 1);
        assert_eq!(c[(1, 0)], 3.);
    }
}
