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
            let at = |row: usize, col: usize| image_old[row * stride + col] as f64;
            let center = at(i, j);
            let differences = [
                at(north, j) - center,
                at(south, j) - center,
                at(i, east) - center,
                at(i, west) - center,
            ];
            let coefficient = |difference: f64| match conduction {
                1 => (-difference * difference / ksq).exp(),
                2 => 1. / (1. + difference * difference / ksq),
                _ => {
                    if difference.abs() > k {
                        0.
                    } else {
                        0.5 * (1. - difference * difference / ksq).powi(2)
                    }
                }
            };
            image[i * stride + j] = (center
                + lambda
                    * (coefficient(differences[0]) * differences[0]
                        + coefficient(differences[1]) * differences[1]
                        + coefficient(differences[2]) * differences[2]
                        + coefficient(differences[3]) * differences[3]))
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
