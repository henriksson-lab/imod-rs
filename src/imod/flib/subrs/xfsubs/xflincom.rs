//! Translation of `IMOD/flib/subrs/xfsubs/xflincom.f`.

/// Original `xflincom` (`xflincom.f:2`).
pub fn xflincom(f: &[f32; 6], a: f32, g: &[f32; 6], b: f32, h: &mut [f32; 6]) {
    h[4] = a * f[4] + b * g[4];
    h[5] = a * f[5] + b * g[5];
    for i in 0..2 {
        for j in 0..2 {
            let index = i + 2 * j;
            h[index] = a * f[index] + b * g[index];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::xflincom;

    #[test]
    fn combines_all_affine_entries_in_fortran_column_order() {
        let (f, g) = ([1., 2., 3., 4., 5., 6.], [6., 5., 4., 3., 2., 1.]);
        let mut h = [0.; 6];
        xflincom(&f, 2., &g, -0.5, &mut h);
        assert_eq!(h, [-1., 1.5, 4., 6.5, 9., 11.5]);
    }
}
