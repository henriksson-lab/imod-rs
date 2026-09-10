//! Translation of `IMOD/flib/subrs/imsubs/iclcdn.f`.

/// Original `ICLCDN` (`iclcdn.f:14`).
///
/// `array` retains Fortran `COMPLEX` storage layout: real and imaginary
/// `f32` values are adjacent, and coordinates are one-based like the source.
pub fn iclcdn(
    array: &[f32],
    mx: i32,
    _my: i32,
    nx1: i32,
    nx2: i32,
    ny1: i32,
    ny2: i32,
) -> (f32, f32, f32) {
    let (mut dmin, mut dmax, mut dmean) = (1.0e10_f32, -1.0e10_f32, 0.0_f32);
    for iy in ny1..=ny2 {
        for ix in nx1..=nx2 {
            let index = 2 * (((iy - 1) * mx + (ix - 1)) as usize);
            let val = (array[index] * array[index] + array[index + 1] * array[index + 1]).sqrt();
            dmean += val;
            dmin = dmin.min(val);
            dmax = dmax.max(val);
        }
    }
    dmean /= ((nx2 - nx1 + 1) * (ny2 - ny1 + 1)) as f32;
    (dmin, dmax, dmean)
}

#[cfg(test)]
mod tests {
    use super::iclcdn;

    #[test]
    fn complex_modulus_subset_matches_fortran_columns_and_rows() {
        // Column-major Fortran complex array: (3,4), (5,12), (8,15), (0,7).
        let array = [3., 4., 5., 12., 8., 15., 0., 7.];
        assert_eq!(iclcdn(&array, 2, 2, 1, 2, 1, 2), (5., 17., 10.5));
        assert_eq!(iclcdn(&array, 2, 2, 2, 2, 1, 2), (7., 13., 10.));
    }
}
