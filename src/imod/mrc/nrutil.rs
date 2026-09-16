//! Safe owned replacement for `IMOD/mrc/nrutil.{c,h}`.
//!
//! Numerical Recipes' shifted pointer allocations become zero-based `Vec`
//! storage; callers retain the original lower bounds as metadata when needed.

#[derive(Clone, Debug, PartialEq)]
pub struct NrVector<T> {
    pub lower: isize,
    pub values: Vec<T>,
}
impl<T: Default + Clone> NrVector<T> {
    pub fn new(lower: isize, upper: isize) -> Self {
        assert!(upper >= lower);
        Self {
            lower,
            values: vec![T::default(); (upper - lower + 1) as usize],
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
pub fn vector(nl: isize, nh: isize) -> NrVector<f32> {
    NrVector::new(nl, nh)
}
pub fn ivector(nl: isize, nh: isize) -> NrVector<i32> {
    NrVector::new(nl, nh)
}
pub fn cvector(nl: isize, nh: isize) -> NrVector<u8> {
    NrVector::new(nl, nh)
}
pub fn lvector(nl: isize, nh: isize) -> NrVector<u64> {
    NrVector::new(nl, nh)
}
pub fn dvector(nl: isize, nh: isize) -> NrVector<f64> {
    NrVector::new(nl, nh)
}
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
        assert!(nrh >= nrl && nch >= ncl);
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
pub fn matrix(a: isize, b: isize, c: isize, d: isize) -> NrMatrix<f32> {
    NrMatrix::new(a, b, c, d)
}
pub fn dmatrix(a: isize, b: isize, c: isize, d: isize) -> NrMatrix<f64> {
    NrMatrix::new(a, b, c, d)
}
pub fn imatrix(a: isize, b: isize, c: isize, d: isize) -> NrMatrix<i32> {
    NrMatrix::new(a, b, c, d)
}
pub fn submatrix(
    a: &NrMatrix<f32>,
    oldrl: isize,
    oldrh: isize,
    oldcl: isize,
    oldch: isize,
    newrl: isize,
    newcl: isize,
) -> NrMatrix<f32> {
    let mut out = NrMatrix::new(newrl, newrl + oldrh - oldrl, newcl, newcl + oldch - oldcl);
    for row in oldrl..=oldrh {
        for col in oldcl..=oldch {
            *out.get_mut(newrl + row - oldrl, newcl + col - oldcl) = *a.get(row, col)
        }
    }
    out
}
pub fn convert_matrix(a: &[f32], nrl: isize, nrh: isize, ncl: isize, nch: isize) -> NrMatrix<f32> {
    let mut out = NrMatrix::new(nrl, nrh, ncl, nch);
    assert_eq!(a.len(), out.values.len());
    out.values.copy_from_slice(a);
    out
}
#[derive(Clone, Debug, PartialEq)]
pub struct NrTensor3 {
    pub row_lower: isize,
    pub column_lower: isize,
    pub depth_lower: isize,
    pub rows: usize,
    pub columns: usize,
    pub depths: usize,
    pub values: Vec<f32>,
}
impl NrTensor3 {
    pub fn get_mut(&mut self, row: isize, column: isize, depth: isize) -> &mut f32 {
        let at = (((row - self.row_lower) as usize * self.columns
            + (column - self.column_lower) as usize)
            * self.depths)
            + (depth - self.depth_lower) as usize;
        &mut self.values[at]
    }
}
pub fn f3tensor(
    nrl: isize,
    nrh: isize,
    ncl: isize,
    nch: isize,
    ndl: isize,
    ndh: isize,
) -> NrTensor3 {
    assert!(nrh >= nrl && nch >= ncl && ndh >= ndl);
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
pub fn sqr(value: f32) -> f32 {
    value * value
}
pub fn dsqr(value: f64) -> f64 {
    value * value
}
pub fn sign(value: f32, sign: f32) -> f32 {
    if sign >= 0. {
        value.abs()
    } else {
        -value.abs()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_vector_and_matrix_are_owned() {
        let mut v = vector(-2, 1);
        *v.get_mut(-2) = 7.;
        assert_eq!(*v.get(-2), 7.);
        let mut m = matrix(1, 2, -1, 0);
        *m.get_mut(2, 0) = 3.;
        assert_eq!(*m.get(2, 0), 3.);
        let mut t = f3tensor(1, 1, 1, 1, -1, 0);
        *t.get_mut(1, 1, -1) = 9.;
        assert_eq!(t.values[0], 9.);
    }
    #[test]
    fn converts_and_copies_submatrix() {
        let a = convert_matrix(&[1., 2., 3., 4.], 1, 2, 1, 2);
        let b = submatrix(&a, 1, 2, 1, 2, 4, 5);
        assert_eq!(*b.get(5, 6), 4.);
    }
}
