//! Translation of `IMOD/raptor/suitesparse/cs_scatter.c`.

use super::cs::{Cs, cs_csc};

/// `cs_scatter(A, j, beta, w, x, mark, C, nz)`: x = x + beta * A(:,j), where
/// x is a dense vector and A(:,j) is sparse.  `ci` is `C->i`.
#[allow(clippy::too_many_arguments)]
pub fn cs_scatter(
    a: &Cs,
    j: i32,
    beta: f64,
    w: &mut [i32],
    mut x: Option<&mut [f64]>,
    mark: i32,
    ci: &mut [i32],
    mut nz: i32,
) -> i32 {
    if !cs_csc(a) {
        return -1;
    }
    let ap = &a.p;
    let ai = &a.i;
    assert!(
        w.len() >= a.m as usize,
        "cs_scatter: w shorter than A's rows"
    );
    let (start, end) = (ap[j as usize] as usize, ap[j as usize + 1] as usize);
    match x.as_deref_mut() {
        Some(x) => {
            let ax = &a.x.as_ref().expect("cs_scatter values")[start..end];
            assert!(
                x.len() >= a.m as usize,
                "cs_scatter: x shorter than A's rows"
            );
            for (&i, &axp) in ai[start..end].iter().zip(ax) {
                let i = i as usize; // A(i,j) is nonzero
                debug_assert!(i < a.m as usize);
                // SAFETY: `i < A->m` (every row index a `Cs` holds is below
                // its `m`; see `cs_gaxpy`), and `w.len()` and `x.len()` are
                // at least `A->m`, asserted at the top of the function and
                // here.
                unsafe {
                    if *w.get_unchecked(i) < mark {
                        *w.get_unchecked_mut(i) = mark; // i is new entry in column j
                        ci[nz as usize] = i as i32; // add i to pattern of C(:,j)
                        nz += 1;
                        *x.get_unchecked_mut(i) = beta * axp; // x(i) = beta*A(i,j)
                    } else {
                        *x.get_unchecked_mut(i) += beta * axp; // i exists in C(:,j) already
                    }
                }
            }
        }
        None => {
            for &i in &ai[start..end] {
                let i = i as usize;
                if w[i] < mark {
                    w[i] = mark;
                    ci[nz as usize] = i as i32;
                    nz += 1;
                }
            }
        }
    }
    nz
}
