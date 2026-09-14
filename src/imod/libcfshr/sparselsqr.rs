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
    let se_call = if unsafe { *ifse > 0 } {
        se
    } else {
        core::ptr::null_mut()
    };
    unsafe {
        lsqr(
            *m,
            *n,
            aprod,
            *damp,
            user_work,
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
            istop_out,
            itn_out,
            anorm_out,
            acond_out,
            rnorm_out,
            arnorm_out,
            xnorm_out,
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
pub unsafe fn add_value_to_row(
    val: f32,
    icol: i32,
    val_row: *mut f32,
    icol_row: *mut i32,
    num_in_row: *mut i32,
) {
    unsafe {
        *val_row.add(*num_in_row as usize) = val;
        *icol_row.add(*num_in_row as usize) = icol;
        *num_in_row += 1;
    }
}

/// Original `addvaluetorow` (`sparselsqr.c:109`).
pub unsafe fn addvaluetorow(
    val: *mut f32,
    icol: *mut i32,
    val_row: *mut f32,
    icol_row: *mut i32,
    num_in_row: *mut i32,
) {
    unsafe {
        add_value_to_row(*val, *icol, val_row, icol_row, num_in_row);
    }
}

/// Original `addRowToMatrix` (`sparselsqr.c:125`).
pub unsafe fn add_row_to_matrix(
    val_row: *mut f32,
    icol_row: *mut i32,
    num_in_row: i32,
    rwrk: *mut f32,
    ia: *mut i32,
    ja: *mut i32,
    num_rows: *mut i32,
    max_rows: i32,
    max_vals: i32,
) -> i32 {
    let mut ind = unsafe { *ia.add(*num_rows as usize) - 1 };
    unsafe {
        *num_rows += 1;
    }
    if unsafe { *num_rows >= max_rows } {
        return 1;
    }
    for i in 0..num_in_row as usize {
        unsafe {
            *rwrk.add(ind as usize) = *val_row.add(i);
            *ja.add(ind as usize) = *icol_row.add(i);
        }
        ind += 1;
        if ind >= max_vals {
            return 2;
        }
    }
    unsafe {
        *ia.add(*num_rows as usize) = ind + 1;
    }
    0
}

/// Original `addrowtomatrix` (`sparselsqr.c:144`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn addrowtomatrix(
    val_row: *mut f32,
    icol_row: *mut i32,
    num_in_row: *mut i32,
    rwrk: *mut f32,
    ia: *mut i32,
    ja: *mut i32,
    num_rows: *mut i32,
    max_rows: *mut i32,
    max_vals: *mut i32,
) -> i32 {
    unsafe {
        add_row_to_matrix(
            val_row,
            icol_row,
            *num_in_row,
            rwrk,
            ia,
            ja,
            num_rows,
            *max_rows,
            *max_vals,
        )
    }
}

/// Original `normalizeColumns` (`sparselsqr.c:162`).
pub unsafe fn normalize_columns(
    rwrk: *mut f32,
    ia: *mut i32,
    ja: *mut i32,
    num_vars: i32,
    num_rows: i32,
    sum_entries: *mut f32,
) {
    for icol in 0..num_vars as usize {
        unsafe {
            *sum_entries.add(icol) = 0.;
        }
    }
    let end = unsafe { *ia.add(num_rows as usize) - 1 };
    for j in 0..end as usize {
        let icol = unsafe { *ja.add(j) - 1 } as usize;
        unsafe {
            *sum_entries.add(icol) += *rwrk.add(j) * *rwrk.add(j);
        }
    }
    for icol in 0..num_vars as usize {
        unsafe {
            *sum_entries.add(icol) = (*sum_entries.add(icol) as f64).sqrt() as f32;
        }
    }
    for j in 0..end as usize {
        let icol = unsafe { *ja.add(j) - 1 } as usize;
        unsafe {
            *rwrk.add(j) /= *sum_entries.add(icol);
        }
    }
}

/// Original `normalizecolumns` (`sparselsqr.c:184`).
pub unsafe fn normalizecolumns(
    rwrk: *mut f32,
    ia: *mut i32,
    ja: *mut i32,
    num_vars: *mut i32,
    num_rows: *mut i32,
    sum_entries: *mut f32,
) {
    unsafe {
        normalize_columns(rwrk, ia, ja, *num_vars, *num_rows, sum_entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_normalizes_a_sparse_matrix() {
        let mut values = [0.; 3];
        let mut columns = [0; 3];
        let mut rows = [1, 1, 1];
        let mut count = 0;
        unsafe {
            add_value_to_row(3., 1, values.as_mut_ptr(), columns.as_mut_ptr(), &mut count);
            add_value_to_row(4., 2, values.as_mut_ptr(), columns.as_mut_ptr(), &mut count);
        }
        let mut matrix = [0.; 3];
        let mut ja = [0; 3];
        let mut num_rows = 0;
        assert_eq!(
            unsafe {
                add_row_to_matrix(
                    values.as_mut_ptr(),
                    columns.as_mut_ptr(),
                    count,
                    matrix.as_mut_ptr(),
                    rows.as_mut_ptr(),
                    ja.as_mut_ptr(),
                    &mut num_rows,
                    3,
                    3,
                )
            },
            0
        );
        let mut norms = [0.; 2];
        unsafe {
            normalize_columns(
                matrix.as_mut_ptr(),
                rows.as_mut_ptr(),
                ja.as_mut_ptr(),
                2,
                num_rows,
                norms.as_mut_ptr(),
            );
        }
        assert_eq!(norms, [3., 4.]);
        assert_eq!(&matrix[..2], &[1., 1.]);
    }
}
