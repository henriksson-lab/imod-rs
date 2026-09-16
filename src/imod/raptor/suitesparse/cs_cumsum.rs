//! Translation of `IMOD/raptor/suitesparse/cs_cumsum.c`.

/// C `cs_cumsum`: cumulative counts, also replacing `counts` by the offsets.
pub fn cs_cumsum(offsets: &mut [usize], counts: &mut [usize]) -> Option<f64> {
    if offsets.len() != counts.len() + 1 {
        return None;
    }
    let mut total = 0_usize;
    let mut as_float = 0.0;
    for index in 0..counts.len() {
        offsets[index] = total;
        total = total.checked_add(counts[index])?;
        as_float += counts[index] as f64;
        counts[index] = offsets[index];
    }
    offsets[counts.len()] = total;
    Some(as_float)
}
