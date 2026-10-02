//! Translation of `IMOD/raptor/suitesparse/cs_gaxpy.c`.

use super::cs::{Cs, cs_csc};

/// `cs_gaxpy(A, x, y)`: y = A*x+y.
pub fn cs_gaxpy(a: &Cs, x: &[f64], y: &mut [f64]) -> i32 {
    if !cs_csc(a) {
        return 0;
    }
    let n = a.n;
    let ap = &a.p;
    let ai = &a.i;
    let ax = a.x.as_ref().expect("cs_gaxpy values");
    // `y` must hold every row: checked once here instead of per element.
    assert!(y.len() >= a.m as usize, "cs_gaxpy: y shorter than A's rows");
    for j in 0..n as usize {
        let (start, end) = (ap[j] as usize, ap[j + 1] as usize);
        let xj = x[j];
        for (&i, &v) in ai[start..end].iter().zip(&ax[start..end]) {
            debug_assert!((i as u32) < a.m as u32);
            // SAFETY: `0 <= i < A->m <= y.len()`.  Every row index a `Cs`
            // holds is below its `m`: the only code that stores row indices
            // is this module's constructors (`cs_entry` grows `m` to cover
            // the index it stores; `cs_compress`, `cs_transpose`,
            // `cs_multiply`, `cs_add`, `cs_symperm`, `cs_chol`, `cs_fkeep`
            // and `probData::CopyCS` copy or permute indices of a matrix
            // with the same or the transposed dimension), and `y.len() >= m`
            // is asserted above.
            unsafe { *y.get_unchecked_mut(i as usize) += v * xj };
        }
    }
    1
}
