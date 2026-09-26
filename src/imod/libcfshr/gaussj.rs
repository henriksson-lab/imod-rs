//! Translation of `IMOD/libcfshr/gaussj.c`.

/// C `MSIZ` (`gaussj.c:26`).
const MSIZ: i32 = 2000;

/// The two failures `gaussj`/`gaussjDet` distinguish (`gaussj.c:36-38`):
/// "The routine returns -1 if [n] exceeds this value and 1 if the A matrix is
/// singular."  Both codes are load-bearing -- `bsubs.f90:2371-2375` keeps the
/// status and prints a different message for each -- so the failure is named
/// rather than collapsed, and the Fortran wrappers below map it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussjError {
    /// C's `-1`: [n] exceeds `MSIZ`.
    TooManyVariables,
    /// C's `1`: the A matrix is singular.
    Singular,
}

/// Original `gaussj` (`gaussj.c:40`).
///
/// Solves the linear matrix equation A X = B by Gauss-Jordan elimination.
/// A is a square matrix of size [n] by [n] in array [a], dimensioned to [np]
/// columns.  B is a matrix with one row per row of A and [m] columns in array
/// [b], dimensioned to [mp] columns.  The columns of [b] are replaced by the
/// [m] solution vectors while [a] is reduced to a unit matrix.  The matrices
/// must be in row-major order.  The routine fails with
/// [`GaussjError::TooManyVariables`] if [n] exceeds 2000 and
/// [`GaussjError::Singular`] if the A matrix is singular.
pub fn gaussj(
    a: &mut [f32],
    n: i32,
    np: i32,
    b: &mut [f32],
    m: i32,
    mp: i32,
) -> Result<(), GaussjError> {
    let mut determ = 0.0f32;
    gaussj_det(a, n, np, b, m, mp, &mut determ)
}

/// Original `gaussjDet` (`gaussj.c:49`).
///
/// Version of `gaussj` that returns a determinant value.
pub fn gaussj_det(
    a: &mut [f32],
    n: i32,
    np: i32,
    b: &mut [f32],
    m: i32,
    mp: i32,
    determ: &mut f32,
) -> Result<(), GaussjError> {
    // The source checks the fixed work-array limit before it dereferences any
    // matrix dimensions; callers rely on this failing even with dummy
    // buffers (the Fortran-facing contract).
    if n > MSIZ {
        return Err(GaussjError::TooManyVariables);
    }
    if n < 0
        || np < n
        || m < 0
        || mp < m
        || a.len() < n as usize * np as usize
        || b.len() < n as usize * mp as usize
    {
        return Err(GaussjError::Singular);
    }
    let mut index = [[0i16; 2]; MSIZ as usize];
    let mut pivot = [0f32; MSIZ as usize];
    let mut ipivot = [0i16; MSIZ as usize];
    // `irow` and `icolum` are uninitialised in the source until the pivot
    // search sets them, and an all-zero (or all-NaN) matrix leaves them so on
    // the first pass, where the C then indexes `ipivot`, `a` and `b` with
    // stack garbage (the `polyfit` one-point abort in `BUGS.md`).  Fixed in
    // translation (2026-09-26): a first pass that finds no pivot reports the
    // matrix singular, which is what the routine's contract says a matrix
    // with no usable pivot is.  Later passes keep the C's (defined) reuse of
    // the previous pass's indices.
    let mut irow: i32 = 0;
    let mut icolum: i32 = 0;
    let mut pivot_found = false;

    *determ = 1.;
    for j in 0..n {
        ipivot[j as usize] = 0;
    }
    for i in 0..n {
        let mut amax = 0f32;
        for j in 0..n {
            if ipivot[j as usize] != 1 {
                for k in 0..n {
                    if ipivot[k as usize] == 0 {
                        let mut abstmp = a[(j * np + k) as usize];
                        if abstmp < 0. {
                            abstmp = -abstmp;
                        }
                        if amax < abstmp {
                            pivot_found = true;
                            irow = j;
                            icolum = k;
                            amax = abstmp;
                        }
                    } else if ipivot[k as usize] > 1 {
                        /* write(*,*) 'Singular matrix' */
                        return Err(GaussjError::Singular);
                    }
                }
            }
        }
        if !pivot_found {
            return Err(GaussjError::Singular);
        }
        ipivot[icolum as usize] += 1;
        if irow != icolum {
            *determ = -*determ;
            for l in 0..n {
                let t = a[(irow * np + l) as usize];
                a[(irow * np + l) as usize] = a[(icolum * np + l) as usize];
                a[(icolum * np + l) as usize] = t;
            }
            for l in 0..m {
                let t = b[(irow * mp + l) as usize];
                b[(irow * mp + l) as usize] = b[(icolum * mp + l) as usize];
                b[(icolum * mp + l) as usize] = t;
            }
        }
        index[i as usize][0] = irow as i16;
        index[i as usize][1] = icolum as i16;
        let pivotmp = a[(icolum * np + icolum) as usize];
        /*    if(abs(pivotmp) < 1.e-30) write(*,*) 'small pivot',pivotmp */
        pivot[i as usize] = pivotmp;
        *determ *= pivotmp;
        a[(icolum * np + icolum) as usize] = 1.;
        /*      worried about that step! */
        for l in 0..n {
            a[(icolum * np + l) as usize] = a[(icolum * np + l) as usize] / pivotmp;
        }
        for l in 0..m {
            b[(icolum * mp + l) as usize] = b[(icolum * mp + l) as usize] / pivotmp;
        }
        for l1 in 0..n {
            let t = a[(l1 * np + icolum) as usize];
            if t != 0. && l1 != icolum {
                a[(l1 * np + icolum) as usize] = 0.;
                for l in 0..n {
                    a[(l1 * np + l) as usize] =
                        a[(l1 * np + l) as usize] - a[(icolum * np + l) as usize] * t;
                }
                for l in 0..m {
                    b[(l1 * mp + l) as usize] =
                        b[(l1 * mp + l) as usize] - b[(icolum * mp + l) as usize] * t;
                }
            }
        }
    }
    for i in 0..n {
        let l = n - 1 - i;
        if index[l as usize][0] != index[l as usize][1] {
            irow = index[l as usize][0] as i32;
            icolum = index[l as usize][1] as i32;
            for k in 0..n {
                let t = a[(k * np + irow) as usize];
                a[(k * np + irow) as usize] = a[(k * np + icolum) as usize];
                a[(k * np + icolum) as usize] = t;
            }
        }
    }
    Ok(())
}

/// Original Fortran wrapper `gaussjfw` (`gaussj.c:137`).
///
/// Fortran callers test the integer status itself (`bsubs.f90:2371`), so the
/// wrapper is where the named failure goes back to the source's codes.
pub fn gaussjfw(a: &mut [f32], n: &i32, np: &i32, b: &mut [f32], m: &i32, mp: &i32) -> i32 {
    match gaussj(a, *n, *np, b, *m, *mp) {
        Ok(()) => 0,
        Err(GaussjError::TooManyVariables) => -1,
        Err(GaussjError::Singular) => 1,
    }
}

/// Original Fortran wrapper `gaussjdet` (`gaussj.c:142`).
pub fn gaussjdet(
    a: &mut [f32],
    n: &i32,
    np: &i32,
    b: &mut [f32],
    m: &i32,
    mp: &i32,
    determ: &mut f32,
) -> i32 {
    match gaussj_det(a, *n, *np, b, *m, *mp, determ) {
        Ok(()) => 0,
        Err(GaussjError::TooManyVariables) => -1,
        Err(GaussjError::Singular) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_row_major_system_and_preserves_source_determinant() {
        let mut matrix = [2.0_f32, 1.0, 5.0, 7.0];
        let mut right_hand = [11.0_f32, 13.0];
        let mut determinant = 0.0;
        assert_eq!(
            gaussj_det(&mut matrix, 2, 2, &mut right_hand, 1, 1, &mut determinant),
            Ok(())
        );
        for (actual, expected) in matrix
            .iter()
            .zip([7.0 / 9.0, -1.0 / 9.0, -5.0 / 9.0, 2.0 / 9.0])
        {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!((right_hand[0] - 64.0 / 9.0).abs() < 1.0e-5);
        assert!((right_hand[1] + 29.0 / 9.0).abs() < 1.0e-5);
        assert_eq!(determinant, 9.0);
        assert_eq!(
            gaussj(&mut [], 2001, 0, &mut [], 0, 0),
            Err(GaussjError::TooManyVariables)
        );
        assert_eq!(
            gaussj(&mut [], 2, 2, &mut [], 1, 1),
            Err(GaussjError::Singular)
        );
        assert_eq!(gaussjfw(&mut [], &2001, &0, &mut [], &0, &0), -1);
        assert_eq!(gaussjfw(&mut [], &2, &2, &mut [], &1, &1), 1);
    }

    /// `BUGS.md` "polyfit with one point and order 1": the C indexes with
    /// uninitialised `irow`/`icolum` when the first pivot search finds
    /// nothing.  Defined behaviour: singular.
    #[test]
    fn no_first_pivot_is_singular() {
        assert_eq!(
            gaussj(&mut [0.0], 1, 1, &mut [1.0], 1, 1),
            Err(GaussjError::Singular)
        );
        assert_eq!(
            gaussj(
                &mut [f32::NAN, f32::NAN, 0.0, f32::NAN],
                2,
                2,
                &mut [1.0, 2.0],
                1,
                1
            ),
            Err(GaussjError::Singular)
        );
        let (x, y) = ([3.0_f32], [5.0_f32]);
        let (mut slopes, mut bint) = ([0.0_f32; 2], 0.0_f32);
        assert_eq!(
            crate::imod::flib::subrs::statsubs::polyfit::polyfit(
                &x,
                &y,
                1,
                1,
                &mut slopes,
                &mut bint
            ),
            3
        );
    }
}
