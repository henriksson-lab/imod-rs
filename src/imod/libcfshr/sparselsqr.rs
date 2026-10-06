//! Translation of `IMOD/libcfshr/sparselsqr.c` (with `IMOD/include/sparselsqr.h`):
//! functions for using `lsqr` with sparse matrices.
//!
//! The sparse matrix lives in one `int` work array, as in the C: `UsrWrk[0]`
//! is the offset of the column numbers (`JA`), `UsrWrk[1]` the offset of the
//! data values, and the row starts (`IA`) begin at index 2.  The data values
//! are `float`s stored in that `int` array (`(float *)UsrWrk + rwofs`), so
//! here they are the `i32` elements read and written through
//! `f32::from_bits`/`to_bits` -- the same four bytes the C reads through its
//! cast.  Callers hand [`add_row_to_matrix`] and [`normalize_columns`] the
//! three disjoint parts of that array (`IA`, `JA`, values).
//!
//! The Fortran wrappers (`lsqrfw`, `addvaluetorow`, `addrowtomatrix`,
//! `normalizecolumns`) are not translated: no translated Fortran calls them.

/// Original `sparseProd` (`sparselsqr.c:56`).
///
/// Compute products in a sparse matrix for `lsqr`.  The matrix has `m` data
/// rows times `n` columns, one for each variable.  If `mode` = 1, compute
/// `y = y + A*x`; if `mode` = 2, compute `x = x + A(transpose)*y`.
pub fn sparse_prod(mode: i32, m: i32, _n: i32, x: &mut [f64], y: &mut [f64], usr_wrk: &[i32]) {
    let iw = usr_wrk;
    let jaofs = iw[0] as usize;
    let rwofs = iw[1] as usize;
    let rw = |ind: usize| f32::from_bits(iw[rwofs + ind] as u32);
    if mode == 1 {
        for irow in 0..m as usize {
            for ind in (iw[irow + 2] - 1) as usize..(iw[irow + 3] - 1) as usize {
                let icol = (iw[jaofs + ind] - 1) as usize;
                y[irow] += rw(ind) as f64 * x[icol];
            }
        }
    } else {
        for irow in 0..m as usize {
            for ind in (iw[irow + 2] - 1) as usize..(iw[irow + 3] - 1) as usize {
                let icol = (iw[jaofs + ind] - 1) as usize;
                x[icol] += rw(ind) as f64 * y[irow];
            }
        }
    }
}

/// Original `addValueToRow` (`sparselsqr.c:87`).
///
/// Adds one value `val` and its data column `icol` to arrays `val_row` and
/// `icol_row` for the current row, and maintains the number of values in
/// `num_in_row`.
pub fn add_value_to_row(
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

/// Original `addRowToMatrix` (`sparselsqr.c:111`).
///
/// Adds one data row to a sparse matrix.  Values are in `val_row`, column
/// numbers (numbered from 1) in `icol_row`, number of values in
/// `num_in_row`.  `rwrk` is the sparse matrix array of data values (`float`
/// bits, module comment), `ja` the corresponding array of column numbers,
/// `ia` an array of starting indexes into those arrays for each row (indexes
/// numbered from 1).  The number of rows is maintained in `num_rows`, and
/// `max_rows` and `max_vals` are the maximum number of rows and values
/// allowed.
#[allow(clippy::too_many_arguments)]
pub fn add_row_to_matrix(
    val_row: &[f32],
    icol_row: &[i32],
    num_in_row: i32,
    rwrk: &mut [i32],
    ia: &mut [i32],
    ja: &mut [i32],
    num_rows: &mut i32,
    max_rows: i32,
    max_vals: i32,
) -> i32 {
    let mut ind = (ia[*num_rows as usize] - 1) as usize;
    *num_rows += 1;
    if *num_rows >= max_rows {
        return 1;
    }
    for i in 0..num_in_row as usize {
        rwrk[ind] = val_row[i].to_bits() as i32;
        let icol = icol_row[i];
        ja[ind] = icol;
        ind += 1;
        if ind as i32 >= max_vals {
            return 2;
        }
    }
    ia[*num_rows as usize] = ind as i32 + 1;
    0
}

/// Original `normalizeColumns` (`sparselsqr.c:146`).
///
/// Normalizes each column of data in the sparse matrix by dividing it by the
/// square root of the sum of squares of values in that column, as recommended
/// for running lsqr.  The normalizing factor for each column is returned in
/// `sum_entries`; the solution obtained from normalized data needs to be
/// divided by it.
pub fn normalize_columns(
    rwrk: &mut [i32],
    ia: &[i32],
    ja: &[i32],
    num_vars: i32,
    num_rows: i32,
    sum_entries: &mut [f32],
) {
    for icol in 0..num_vars as usize {
        sum_entries[icol] = 0.;
    }
    for j in 0..(ia[num_rows as usize] - 1).max(0) as usize {
        let icol = (ja[j] - 1) as usize;
        let value = f32::from_bits(rwrk[j] as u32);
        sum_entries[icol] += value * value;
    }
    for icol in 0..num_vars as usize {
        sum_entries[icol] = (sum_entries[icol] as f64).sqrt() as f32;
    }
    for j in 0..(ia[num_rows as usize] - 1).max(0) as usize {
        let icol = (ja[j] - 1) as usize;
        let value = f32::from_bits(rwrk[j] as u32) / sum_entries[icol];
        rwrk[j] = value.to_bits() as i32;
    }
}
