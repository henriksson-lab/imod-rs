//! Translation of `IMOD/raptor/suitesparse/cs_entry.c`.

use super::cs::{Cs, cs_max, cs_triplet};
use super::cs_util::cs_sprealloc;

/// `cs_entry(T, i, j, x)`: add an entry to a triplet matrix; return 1 if ok,
/// 0 otherwise.
pub fn cs_entry(t: &mut Cs, i: i32, j: i32, x: f64) -> i32 {
    if !cs_triplet(t) || i < 0 || j < 0 {
        return 0;
    }
    if t.nz >= t.nzmax && cs_sprealloc(t, 2 * t.nzmax) == 0 {
        return 0;
    }
    let nz = t.nz as usize;
    if let Some(tx) = t.x.as_mut() {
        tx[nz] = x;
    }
    t.i[nz] = i;
    t.p[nz] = j;
    t.nz += 1;
    t.m = cs_max(t.m, i + 1);
    t.n = cs_max(t.n, j + 1);
    1
}
