//! Translation of `IMOD/libiimod/diffusion.c`.

/// C `updateMatrix`, with the source's clamped (no-flux) edge neighbours.
/// `image` and `image_old` are row-major `(m + 2) × (n + 2)` images.  The
/// source allocates this border but only writes indices `1..=m, 1..=n`; keeping
/// that representation makes the source's pixel coordinates exact without a
/// pointer matrix or manual allocation.
pub fn update_matrix(
    image: &mut [f32],
    image_old: &[f32],
    m: usize,
    n: usize,
    conduction: i32,
    k: f64,
    lambda: f64,
) -> Result<(), String> {
    let stride = n
        .checked_add(2)
        .ok_or_else(|| "diffusion width overflows usize".to_owned())?;
    let buffer_len = m
        .checked_add(2)
        .and_then(|rows| rows.checked_mul(stride))
        .ok_or_else(|| "diffusion dimensions overflow usize".to_owned())?;
    if m == 0 || n == 0 || image.len() != buffer_len || image_old.len() != buffer_len {
        return Err("image dimensions do not match diffusion buffers".into());
    }
    if k == 0. {
        return Err("diffusion k must not be zero".into());
    }
    // `diffusion.c:34-90` writes only `image[1..=m][1..=n]` and reads
    // `imageOld` only at those same clamped indices -- `ip1`/`im1` and
    // `jp1`/`jm1` never leave `1..=m` / `1..=n`.  The padded border of both
    // matrices is therefore never read and never written by the source, which
    // allocates it with `malloc` in `allocate2D_float` (`sliceproc.c:646`) and
    // leaves it uninitialised.  A whole-buffer `image.copy_from_slice(image_old)`
    // here would be a full extra pass over `(m + 2) * (n + 2)` floats per
    // iteration that the source does not make, and it can touch nothing the
    // callers read back: both `sliceAnisoDiff` and `sliceByteAnisoDiff` copy
    // out only rows `1..=m`, columns `1..=n`.
    let ksq = k * k;
    for i in 1..=m {
        let north = if i == 1 { 1 } else { i - 1 };
        let south = if i == m { m } else { i + 1 };
        for j in 1..=n {
            let east = if j == n { n } else { j + 1 };
            let west = if j == 1 { 1 } else { j - 1 };
            // `diffusion.c:43-46`: `imageOld` is `float **`, so each
            // difference is a *single-precision* subtraction, widened to
            // double only on assignment to `diffN` etc.  Widening the two
            // operands first would skip the float rounding whenever the exact
            // difference is not representable in `f32`.
            let at = |row: usize, col: usize| image_old[row * stride + col];
            let center = at(i, j);
            let diff_n = (at(north, j) - center) as f64;
            let diff_s = (at(south, j) - center) as f64;
            let diff_e = (at(i, east) - center) as f64;
            let diff_w = (at(i, west) - center) as f64;
            let (grad_n, grad_s, grad_e, grad_w) = (diff_n, diff_s, diff_e, diff_w);
            // `diffusion.c:59-88`: one branch on `CC` per pixel, all four
            // coefficients inside the taken arm.
            let c_n: f64;
            let c_s: f64;
            let c_e: f64;
            let c_w: f64;
            if conduction == 1 {
                c_n = (-grad_n * grad_n / ksq).exp();
                c_s = (-grad_s * grad_s / ksq).exp();
                c_e = (-grad_e * grad_e / ksq).exp();
                c_w = (-grad_w * grad_w / ksq).exp();
            } else if conduction == 2 {
                c_n = 1. / (1. + grad_n * grad_n / ksq);
                c_s = 1. / (1. + grad_s * grad_s / ksq);
                c_e = 1. / (1. + grad_e * grad_e / ksq);
                c_w = 1. / (1. + grad_w * grad_w / ksq);
            } else {
                /* use Tukey Biweight.  C evaluates `0.5 * a * a` left to
                right as `(0.5 * a) * a`. */
                c_n = if diff_n.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_n * grad_n / ksq) * (1. - grad_n * grad_n / ksq)
                };
                c_s = if diff_s.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_s * grad_s / ksq) * (1. - grad_s * grad_s / ksq)
                };
                c_e = if diff_e.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_e * grad_e / ksq) * (1. - grad_e * grad_e / ksq)
                };
                c_w = if diff_w.abs() > k {
                    0.
                } else {
                    0.5 * (1. - grad_w * grad_w / ksq) * (1. - grad_w * grad_w / ksq)
                };
            }
            image[i * stride + j] = (center as f64
                + lambda * (c_n * diff_n + c_s * diff_s + c_e * diff_e + c_w * diff_w))
                as f32;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_and_clamped_edges_follow_source() {
        let old = vec![5.; 25];
        let mut new = vec![0.; 25];
        update_matrix(&mut new, &old, 3, 3, 1, 1., 0.2).unwrap();
        // Only the interior is defined: the source writes `image[1..=m][1..=n]`
        // and leaves the padded border of its `malloc`ed matrix untouched.
        for row in 1..=3 {
            for column in 1..=3 {
                assert_eq!(new[row * 5 + column], 5.);
            }
        }
        let mut old = vec![0.; 25];
        old[2 * 5 + 2] = 1.;
        update_matrix(&mut new, &old, 3, 3, 2, 1., 0.2).unwrap();
        assert!(new[2 * 5 + 2] < 1. && new[2 * 5 + 2] > 0.);
    }
}
