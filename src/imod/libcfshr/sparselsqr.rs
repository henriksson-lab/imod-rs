//! Translation of `IMOD/libcfshr/sparselsqr.c` and its local `sparselsqr.h` interface.

use core::ffi::c_void;

use super::b3dutil::ImodFile;
use super::lsqr::{Aprod, lsqr};

/// Original `lsqrfw` (`sparselsqr.c:18`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn lsqrfw(
    m: *mut i32,
    n: *mut i32,
    aprod: Aprod,
    damp: *mut f64,
    ifse: *mut i32,
    user_work: *mut c_void,
    u: *mut f64,
    v: *mut f64,
    w: *mut f64,
    x: *mut f64,
    se: *mut f64,
    atol: *mut f64,
    btol: *mut f64,
    conlim: *mut f64,
    itnlim: *mut i32,
    nout: *mut i32,
    istop_out: *mut i32,
    itn_out: *mut i32,
    anorm_out: *mut f64,
    acond_out: *mut f64,
    rnorm_out: *mut f64,
    arnorm_out: *mut f64,
    xnorm_out: *mut f64,
) {
    let mut output = if unsafe { *nout > 0 } {
        Some(ImodFile::Stdout)
    } else {
        None
    };
    let m = unsafe { *m as usize };
    let n = unsafe { *n as usize };
    let u = unsafe { core::slice::from_raw_parts_mut(u, m) };
    let v = unsafe { core::slice::from_raw_parts_mut(v, n) };
    let w = unsafe { core::slice::from_raw_parts_mut(w, n) };
    let x = unsafe { core::slice::from_raw_parts_mut(x, n) };
    let se_call = (unsafe { *ifse > 0 }).then(|| unsafe { core::slice::from_raw_parts_mut(se, n) });
    unsafe {
        lsqr(
            m,
            n,
            |mode, x, y| {
                aprod(
                    mode,
                    m as i32,
                    n as i32,
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    user_work,
                )
            },
            *damp,
            u,
            v,
            w,
            x,
            se_call,
            *atol,
            *btol,
            *conlim,
            *itnlim,
            output.as_mut(),
            &mut *istop_out,
            &mut *itn_out,
            &mut *anorm_out,
            &mut *acond_out,
            &mut *rnorm_out,
            &mut *arnorm_out,
            &mut *xnorm_out,
        );
    }
}

/// Original `sparseProd` (`sparselsqr.c:66`).
pub unsafe extern "C" fn sparse_prod(
    mode: i32,
    m: i32,
    _n: i32,
    x: *mut f64,
    y: *mut f64,
    user_work: *mut c_void,
) {
    let iw = user_work.cast::<i32>();
    let jaofs = unsafe { *iw };
    let rwofs = unsafe { *iw.add(1) };
    let rw = unsafe { user_work.cast::<f32>().add(rwofs as usize) };
    for irow in 0..m as usize {
        let begin = unsafe { *iw.add(irow + 2) - 1 };
        let end = unsafe { *iw.add(irow + 3) - 1 };
        for ind in begin..end {
            let icol = unsafe { *iw.add(jaofs as usize + ind as usize) - 1 } as usize;
            unsafe {
                if mode == 1 {
                    *y.add(irow) += *rw.add(ind as usize) as f64 * *x.add(icol);
                } else {
                    *x.add(icol) += *rw.add(ind as usize) as f64 * *y.add(irow);
                }
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
