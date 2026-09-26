//! Translation of `IMOD/flib/subrs/hvem/readnumpatches.f90`.

use crate::imod::flib::subrs::hvem::frefor::frefor2;
use std::io::BufRead;

/// Original `readNumPatches` (`readnumpatches.f90:8`).
///
/// Reads in the first line of a patch file opened on the unit [iunit], and
/// returns the first value on the line in [numPatch].  It looks for further
/// numeric values on the line, up to [limVals] of them, and returns them in
/// [values] and the number found in [numVals].  The return value is 1 for
/// error reading the file, or 2 if the first item on the line is not an
/// integer.
///
/// The Fortran unit is the `BufRead` it is connected to.  `read(iunit, '(a)',
/// iostat = ierr) line` takes one record into the `character*1024` variable:
/// longer records are cut at 1024, shorter ones blank padded, and end of
/// file or a read error is a nonzero `ierr`.
pub fn read_num_patches<R: BufRead>(
    iunit: &mut R,
    num_patch: &mut i32,
    num_vals: &mut i32,
    id_values: &mut [i32],
    lim_vals: i32,
) -> i32 {
    let mut xnum = vec![0.0_f32; (2 * lim_vals + 9) as usize];
    let mut numeric = vec![0_i32; (2 * lim_vals + 9) as usize];
    let mut num_fields = 0_i32;
    let mut record: Vec<u8> = Vec::new();
    let read_num_patches = 1;
    let ierr = match iunit.read_until(b'\n', &mut record) {
        Ok(0) | Err(_) => 1,
        Ok(_) => 0,
    };
    if ierr != 0 {
        return read_num_patches;
    }
    if record.last() == Some(&b'\n') {
        record.pop();
    }
    record.resize(1024, b' ');
    let line = String::from_utf8_lossy(&record).into_owned();
    frefor2(
        &line,
        &mut xnum,
        &mut numeric,
        &mut num_fields,
        2 * lim_vals + 9,
    );
    let read_num_patches = 2;
    if record[0] == b'/' || num_fields < 1 || numeric[0] == 0 {
        return read_num_patches;
    }
    *num_patch = xnum[0].round() as i32;
    if (*num_patch as f32 - xnum[0]).abs() > 1.0e3 {
        return read_num_patches;
    }
    *num_vals = 0;
    for ind in 2..=num_fields {
        if numeric[(ind - 1) as usize] > 0 && *num_vals < lim_vals {
            *num_vals += 1;
            id_values[(*num_vals - 1) as usize] = xnum[(ind - 1) as usize].round() as i32;
        }
    }
    0
}
