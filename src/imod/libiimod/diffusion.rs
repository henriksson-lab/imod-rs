//! Translation of `IMOD/libiimod/diffusion.c` and `include/sliceproc.h`.
#![allow(dead_code)]

/// Matches C `updateMatrix(float **, float **, int, int, int, double, double)`
/// (`diffusion.c:18`).  Both image matrices use the source's one-based
/// indexing convention: rows `1..=m` and columns `1..=n` are read and written.
///
/// The source's `float **` values are rows owned by the caller, so slices
/// express the actual contract without manufacturing a temporary pointer table.
pub fn update_matrix(
    image: &mut [Vec<f32>],
    image_old: &[Vec<f32>],
    m: i32,
    n: i32,
    cc: i32,
    k: f64,
    lambda: f64,
) {
    let ksq = k * k;
    for i in 1..m + 1 {
        let ip1 = if i == m { m } else { i + 1 };
        let im1 = if i == 1 { 1 } else { i - 1 };
        for j in 1..n + 1 {
            let jp1 = if j == n { n } else { j + 1 };
            let jm1 = if j == 1 { 1 } else { j - 1 };
            let diff_n =
                (image_old[im1 as usize][j as usize] - image_old[i as usize][j as usize]) as f64;
            let diff_s =
                (image_old[ip1 as usize][j as usize] - image_old[i as usize][j as usize]) as f64;
            let diff_e =
                (image_old[i as usize][jp1 as usize] - image_old[i as usize][j as usize]) as f64;
            let diff_w =
                (image_old[i as usize][jm1 as usize] - image_old[i as usize][j as usize]) as f64;
            let grad_n = diff_n;
            let grad_s = diff_s;
            let grad_e = diff_e;
            let grad_w = diff_w;
            let (c_n, c_s, c_e, c_w) = if cc == 1 {
                (
                    (-grad_n * grad_n / ksq).exp(),
                    (-grad_s * grad_s / ksq).exp(),
                    (-grad_e * grad_e / ksq).exp(),
                    (-grad_w * grad_w / ksq).exp(),
                )
            } else if cc == 2 {
                (
                    1. / (1. + grad_n * grad_n / ksq),
                    1. / (1. + grad_s * grad_s / ksq),
                    1. / (1. + grad_e * grad_e / ksq),
                    1. / (1. + grad_w * grad_w / ksq),
                )
            } else {
                let c_n = if diff_n.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_n * grad_n / ksq) * (1. - grad_n * grad_n / ksq)
                };
                let c_s = if diff_s.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_s * grad_s / ksq) * (1. - grad_s * grad_s / ksq)
                };
                let c_e = if diff_e.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_e * grad_e / ksq) * (1. - grad_e * grad_e / ksq)
                };
                let c_w = if diff_w.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_w * grad_w / ksq) * (1. - grad_w * grad_w / ksq)
                };
                (c_n, c_s, c_e, c_w)
            };
            image[i as usize][j as usize] = (image_old[i as usize][j as usize] as f64
                + lambda * (c_n * diff_n + c_s * diff_s + c_e * diff_e + c_w * diff_w))
                as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::update_matrix;

    #[test]
    fn update_matrix_retains_one_based_boundary_and_conduction_branches() {
        let old_rows = vec![
            vec![0.; 4],
            vec![0., 1., 2., 3.],
            vec![0., 4., 5., 6.],
            vec![0., 7., 8., 9.],
        ];
        let mut output_rows = vec![vec![-1.; 4]; 4];
        update_matrix(&mut output_rows, &old_rows, 3, 3, 2, 2., 0.25);
        assert_eq!(output_rows[0], vec![-1.; 4]);
        assert!((output_rows[1][1] - (1.0 + 0.25 * (12.0 / 13.0 + 0.8))).abs() < 1.0e-6);
        let cc_two_corner = output_rows[1][1];
        update_matrix(&mut output_rows, &old_rows, 3, 3, 1, 2., 0.25);
        assert_ne!(output_rows[1][1], cc_two_corner);
        update_matrix(&mut output_rows, &old_rows, 3, 3, 3, 0.5, 0.25);
        assert_eq!(output_rows[2][2], 5.0);
    }

    #[test]
    fn update_matrix_tukey_cutoff_and_corner_coefficients_match_source_formula() {
        // The active 2 by 2 matrix starts at [1][1], as in C.  At its upper
        // left corner north and west are clamped to the corner itself; the
        // south difference (4) is beyond k and must contribute zero.
        let old_rows = vec![vec![91.; 3], vec![92., 0., 2.], vec![93., 4., 6.]];
        let mut output_rows = vec![vec![-7.; 3]; 3];
        update_matrix(&mut output_rows, &old_rows, 2, 2, 3, 3., 0.25);

        // cE = 0.5 * (1 - 2^2 / 3^2)^2 = 25 / 162; cS = 0.
        assert!((output_rows[1][1] - 25. / 324.).abs() < 1.0e-6);
        assert_eq!(output_rows[0], vec![-7.; 3]);
        assert_eq!(output_rows[1][0], -7.);
        assert_eq!(output_rows[2][0], -7.);
    }
}
