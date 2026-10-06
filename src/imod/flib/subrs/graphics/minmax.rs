//! Translation of `IMOD/flib/subrs/graphics/minmax.f`.

/// Original `minmax` (`minmax.f:2`): the minimum and maximum of the first
/// `n` elements of `x`.
pub fn minmax(x: &[f32], n: i32, xmin: &mut f32, xmax: &mut f32) {
    *xmin = x.first().copied().unwrap_or(0.);
    *xmax = *xmin;
    for i in 2..=n.max(0) as usize {
        let xx = x[i - 1];
        if xx > *xmax {
            *xmax = xx;
        }
        if xx < *xmin {
            *xmin = xx;
        }
    }
}
