//! Translation of `IMOD/libcfshr/sparselsqr.c` and its local `sparselsqr.h` interface.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::lsqr::lsqr;

/// `lsqrfw` (`sparselsqr.c:17`).
///
/// The source wrapper turns Fortran's scalar-pointer arguments into a call to
/// `lsqr`, sends a positive `nout` to stdout, and only exposes `se` when
/// `ifse` is positive.  Rust closures capture the source `UsrWrk` state rather
/// than passing an untyped pointer through the solver.
#[allow(clippy::too_many_arguments)]
pub fn lsqrfw<F>(
    m: i32,
    n: i32,
    mut aprod: F,
    damp: f64,
    ifse: i32,
    u: &mut [f64],
    v: &mut [f64],
    w: &mut [f64],
    x: &mut [f64],
    se: &mut [f64],
    atol: f64,
    btol: f64,
    conlim: f64,
    itnlim: i32,
    nout: i32,
    istop_out: &mut i32,
    itn_out: &mut i32,
    anorm_out: &mut f64,
    acond_out: &mut f64,
    rnorm_out: &mut f64,
    arnorm_out: &mut f64,
    xnorm_out: &mut f64,
) where
    F: FnMut(i32, &mut [f64], &mut [f64]),
{
    assert!(m >= 0 && n >= 0);
    let mut stdout = ImodFile::Stdout;
    let output = if nout > 0 { Some(&mut stdout) } else { None };
    if ifse > 0 {
        lsqr(
            m as usize,
            n as usize,
            &mut aprod,
            damp,
            u,
            v,
            w,
            x,
            Some(se),
            atol,
            btol,
            conlim,
            itnlim,
            output,
            istop_out,
            itn_out,
            anorm_out,
            acond_out,
            rnorm_out,
            arnorm_out,
            xnorm_out,
        );
    } else {
        lsqr(
            m as usize, n as usize, &mut aprod, damp, u, v, w, x, None, atol, btol, conlim, itnlim,
            output, istop_out, itn_out, anorm_out, acond_out, rnorm_out, arnorm_out, xnorm_out,
        );
    }
}

/// `sparseProd` (`sparselsqr.c:64`).
///
/// C overlays `UsrWrk` with integer row/column offsets and float values.  The
/// three typed slices below are that same packed sparse representation without
/// exposing an aliasing byte buffer: `ia` retains its source one-based row
/// starts and `ja` retains its source one-based column numbers.
pub fn sparse_prod(
    mode: i32,
    m: i32,
    n: i32,
    x: &mut [f64],
    y: &mut [f64],
    ia: &[i32],
    ja: &[i32],
    rwrk: &[f32],
) {
    assert!(m >= 0 && n >= 0);
    assert!(ia.len() > m as usize);
    assert!(x.len() >= n as usize);
    assert!(y.len() >= m as usize);
    for irow in 0..m as usize {
        let start = usize::try_from(ia[irow] - 1).expect("source IA is one-based");
        let end = usize::try_from(ia[irow + 1] - 1).expect("source IA is one-based");
        assert!(end <= ja.len() && end <= rwrk.len());
        for ind in start..end {
            let icol = usize::try_from(ja[ind] - 1).expect("source JA is one-based");
            assert!(icol < n as usize);
            if mode == 1 {
                y[irow] += f64::from(rwrk[ind]) * x[icol];
            } else {
                x[icol] += f64::from(rwrk[ind]) * y[irow];
            }
        }
    }
}

/// Original `addValueToRow` (`sparselsqr.c:100`).
pub fn add_value_to_row(
    val: f32,
    icol: i32,
    val_row: &mut [f32],
    icol_row: &mut [i32],
    num_in_row: &mut usize,
) {
    val_row[*num_in_row] = val;
    icol_row[*num_in_row] = icol;
    *num_in_row += 1;
}

/// Original `addvaluetorow` (`sparselsqr.c:109`).
pub fn addvaluetorow(
    val: f32,
    icol: i32,
    val_row: &mut [f32],
    icol_row: &mut [i32],
    num_in_row: &mut i32,
) {
    val_row[*num_in_row as usize] = val;
    icol_row[*num_in_row as usize] = icol;
    *num_in_row += 1;
}

/// Original `addRowToMatrix` (`sparselsqr.c:125`).
pub fn add_row_to_matrix(
    val_row: &[f32],
    icol_row: &[i32],
    num_in_row: usize,
    rwrk: &mut [f32],
    ia: &mut [i32],
    ja: &mut [i32],
    num_rows: &mut usize,
    max_rows: usize,
    max_vals: usize,
) -> i32 {
    let mut ind = (ia[*num_rows] - 1) as usize;
    *num_rows += 1;
    if *num_rows >= max_rows {
        return 1;
    }
    for i in 0..num_in_row {
        rwrk[ind] = val_row[i];
        ja[ind] = icol_row[i];
        ind += 1;
        if ind >= max_vals {
            return 2;
        }
    }
    ia[*num_rows] = (ind + 1) as i32;
    0
}

/// Original `addrowtomatrix` (`sparselsqr.c:144`).
#[allow(clippy::too_many_arguments)]
pub fn addrowtomatrix(
    val_row: &[f32],
    icol_row: &[i32],
    num_in_row: i32,
    rwrk: &mut [f32],
    ia: &mut [i32],
    ja: &mut [i32],
    num_rows: &mut i32,
    max_rows: i32,
    max_vals: i32,
) -> i32 {
    let max_rows = max_rows as usize;
    let max_vals = max_vals as usize;
    let mut rows = *num_rows as usize;
    let status = add_row_to_matrix(
        &val_row[..num_in_row as usize],
        &icol_row[..num_in_row as usize],
        num_in_row as usize,
        &mut rwrk[..max_vals],
        &mut ia[..max_rows],
        &mut ja[..max_vals],
        &mut rows,
        max_rows,
        max_vals,
    );
    *num_rows = rows as i32;
    status
}

/// Original `normalizeColumns` (`sparselsqr.c:162`).
pub fn normalize_columns(
    rwrk: &mut [f32],
    ia: &[i32],
    ja: &[i32],
    num_vars: usize,
    num_rows: usize,
    sum_entries: &mut [f32],
) {
    sum_entries[..num_vars].fill(0.);
    let end = (ia[num_rows] - 1) as usize;
    for j in 0..end {
        let icol = (ja[j] - 1) as usize;
        sum_entries[icol] += rwrk[j] * rwrk[j];
    }
    for value in &mut sum_entries[..num_vars] {
        *value = (*value as f64).sqrt() as f32;
    }
    for j in 0..end {
        let icol = (ja[j] - 1) as usize;
        rwrk[j] /= sum_entries[icol];
    }
}

/// Original `normalizecolumns` (`sparselsqr.c:184`).
pub fn normalizecolumns(
    rwrk: &mut [f32],
    ia: &[i32],
    ja: &[i32],
    num_vars: i32,
    num_rows: i32,
    sum_entries: &mut [f32],
) {
    let end = ia[num_rows as usize] as usize - 1;
    normalize_columns(
        &mut rwrk[..end],
        &ia[..num_rows as usize + 1],
        &ja[..end],
        num_vars as usize,
        num_rows as usize,
        &mut sum_entries[..num_vars as usize],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_product_preserves_source_one_based_row_and_column_indices() {
        let ia = [1, 3, 4];
        let ja = [1, 2, 2];
        let values = [2., 1., 3.];
        let mut x = [4., 5.];
        let mut y = [0., 0.];
        sparse_prod(1, 2, 2, &mut x, &mut y, &ia, &ja, &values);
        assert_eq!(y, [13., 15.]);

        let mut x = [0., 0.];
        let mut y = [7., 11.];
        sparse_prod(2, 2, 2, &mut x, &mut y, &ia, &ja, &values);
        assert_eq!(x, [14., 40.]);
    }

    #[test]
    fn fortran_lsqr_wrapper_preserves_the_optional_standard_error_switch() {
        let mut u = [6.];
        let mut v = [0.];
        let mut w = [0.];
        let mut x = [0.];
        let mut se = [17.];
        let (mut stop, mut iterations) = (0, 0);
        let (mut anorm, mut acond, mut rnorm, mut arnorm, mut xnorm) = (0., 0., 0., 0., 0.);
        lsqrfw(
            1,
            1,
            |mode, x, y| {
                if mode == 1 {
                    y[0] += 2. * x[0];
                } else {
                    x[0] += 2. * y[0];
                }
            },
            0.,
            0,
            &mut u,
            &mut v,
            &mut w,
            &mut x,
            &mut se,
            1.0e-12,
            1.0e-12,
            1.0e12,
            20,
            0,
            &mut stop,
            &mut iterations,
            &mut anorm,
            &mut acond,
            &mut rnorm,
            &mut arnorm,
            &mut xnorm,
        );
        assert_eq!(stop, 1);
        assert!((x[0] - 3.).abs() < 1.0e-10);
        assert_eq!(se, [17.]);
    }

    #[test]
    fn builds_and_normalizes_a_sparse_matrix() {
        let mut values = [0.; 3];
        let mut columns = [0; 3];
        let mut rows = [1, 1, 1];
        let mut count = 0_i32;
        addvaluetorow(3., 1, &mut values, &mut columns, &mut count);
        addvaluetorow(4., 2, &mut values, &mut columns, &mut count);
        let mut matrix = [0.; 3];
        let mut ja = [0; 3];
        let mut num_rows = 0;
        assert_eq!(
            addrowtomatrix(
                &values,
                &columns,
                count,
                &mut matrix,
                &mut rows,
                &mut ja,
                &mut num_rows,
                3,
                3,
            ),
            0
        );
        let mut norms = [0.; 2];
        normalizecolumns(&mut matrix, &rows, &ja, 2, num_rows, &mut norms);
        assert_eq!(norms, [3., 4.]);
        assert_eq!(&matrix[..2], &[1., 1.]);
    }
}
