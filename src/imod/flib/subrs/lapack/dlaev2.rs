//! Translation of `IMOD/flib/subrs/lapack/dlaev2.f` — LAPACK auxiliary
//! routine version 3.0.

/// Fortran `DLAEV2`, reached from C through `IMOD/include/lapackc.h`'s
/// `dlaev2_` declaration.
///
/// Computes the eigendecomposition of a 2-by-2 symmetric matrix
/// `[ A B ; B C ]`.  On return `rt1` is the eigenvalue of larger absolute
/// value, `rt2` the one of smaller absolute value, and `(cs1, sn1)` the unit
/// right eigenvector for `rt1`, giving
///
/// ```text
/// [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
/// [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
/// ```
pub fn dlaev2(a: f64, b: f64, c: f64, rt1: &mut f64, rt2: &mut f64, cs1: &mut f64, sn1: &mut f64) {
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const ZERO: f64 = 0.0;
    const HALF: f64 = 0.5;

    let sgn1: i32;
    let sgn2: i32;
    let (ab, acmn, acmx, acs, adf, cs, ct, df, rt, sm, tb, mut tn);

    /*     Compute the eigenvalues */

    sm = a + c;
    df = a - c;
    adf = df.abs();
    tb = b + b;
    ab = tb.abs();
    if a.abs() > c.abs() {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }
    if adf > ab {
        rt = adf * (ONE + (ab / adf) * (ab / adf)).sqrt();
    } else if adf < ab {
        rt = ab * (ONE + (adf / ab) * (adf / ab)).sqrt();
    } else {
        /*        Includes case AB=ADF=0 */
        rt = ab * TWO.sqrt();
    }
    if sm < ZERO {
        *rt1 = HALF * (sm - rt);
        sgn1 = -1;
        /*        Order of execution important.
        To get fully accurate smaller eigenvalue,
        next line needs to be executed in higher precision. */
        *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
    } else if sm > ZERO {
        *rt1 = HALF * (sm + rt);
        sgn1 = 1;
        /*        Order of execution important. */
        *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
    } else {
        /*        Includes case RT1 = RT2 = 0 */
        *rt1 = HALF * rt;
        *rt2 = -HALF * rt;
        sgn1 = 1;
    }

    /*     Compute the eigenvector */

    if df >= ZERO {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }
    acs = cs.abs();
    if acs > ab {
        ct = -tb / cs;
        *sn1 = ONE / (ONE + ct * ct).sqrt();
        *cs1 = ct * *sn1;
    } else if ab == ZERO {
        *cs1 = ONE;
        *sn1 = ZERO;
    } else {
        tn = -cs / tb;
        *cs1 = ONE / (ONE + tn * tn).sqrt();
        *sn1 = tn * *cs1;
    }
    if sgn1 == sgn2 {
        tn = *cs1;
        *cs1 = -*sn1;
        *sn1 = tn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_returns_its_own_eigenvalues() {
        let (mut rt1, mut rt2, mut cs1, mut sn1) = (0., 0., 0., 0.);
        dlaev2(3.0, 0.0, 1.0, &mut rt1, &mut rt2, &mut cs1, &mut sn1);
        assert_eq!(rt1, 3.0);
        assert!((rt2 - 1.0).abs() < 1.0e-15);
        assert_eq!((cs1.abs(), sn1.abs()), (1.0, 0.0));
    }

    #[test]
    fn eigenvector_diagonalises_the_matrix() {
        let (a, b, c) = (2.0_f64, -1.5_f64, 0.5_f64);
        let (mut rt1, mut rt2, mut cs1, mut sn1) = (0., 0., 0., 0.);
        dlaev2(a, b, c, &mut rt1, &mut rt2, &mut cs1, &mut sn1);
        /* [cs1 sn1] A [cs1; sn1] == rt1 */
        let q = cs1 * (a * cs1 + b * sn1) + sn1 * (b * cs1 + c * sn1);
        assert!((q - rt1).abs() < 1.0e-12, "{q} != {rt1}");
        assert!(rt1.abs() >= rt2.abs());
    }
}
