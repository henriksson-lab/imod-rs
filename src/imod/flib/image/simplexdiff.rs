//! Translation of `IMOD/flib/image/simplexdiff.c` - forms sums for the
//! difference measure in xfsimplex.
//!
//! [`simplex_diff`] is `simplexDiff`; the Fortran wrapper `simplexdiff`
//! (pointer arguments dereferenced) is folded into it, since its only caller
//! is the Rust `xfsimplex`.  The six component macros (`NEAREST_IX_IY`,
//! `LINEAR_IX_IY`, `NEAREST_ARRAY_DEN`, `LINEAR_ARRAY_DEN`, `PATCH_SUM`,
//! `PATCH_CCC_SUM`) are expanded at their sites as the preprocessor does.
//! `(int)` of a `float` is x86 truncation ([`cvttss2si`]).  `fx1 = 1. - fx`
//! is a double subtraction rounded to `float` in the source; with both
//! operands `float` that equals the `float` subtraction (double has more than
//! 2 * 24 + 2 bits, so the double rounding is innocuous), which is also what
//! gcc emits for it.

use crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si;

/// Original `simplexDiff` (`simplexdiff.c:68`), with its Fortran wrapper
/// `simplexdiff` (`simplexdiff.c:222`).
///
/// `amat` is the Fortran `amat(2,2)` in column order; `sxa` .. `num_pix_a`
/// are the `(numXpatch, numYpatch)` patch arrays in column order.
#[allow(clippy::too_many_arguments)]
pub fn simplex_diff(
    crray: &[f32],
    drray: &[f32],
    nx: i32,
    ny: i32,
    nx1: i32,
    nx2: i32,
    ny1: i32,
    ny2: i32,
    amat: &[f32; 4],
    xcen: f32,
    ycen: f32,
    x_add: f32,
    y_add: f32,
    if_interp: i32,
    old_diff: i32,
    if_ccc: i32,
    sx: &mut f64,
    num_pix: &mut i32,
    num_xpatch: i32,
    nxy_patch: i32,
    sxa: &mut [f64],
    sya: &mut [f64],
    sxya: &mut [f64],
    sx_sqa: &mut [f64],
    sy_sqa: &mut [f64],
    num_pix_a: &mut [i32],
) {
    let (mut x, mut y, mut fx, mut fx1, mut fy, mut fy1): (f32, f32, f32, f32, f32, f32);
    let (mut den, mut dend): (f32, f32);
    let (mut ix, mut iy, mut ix1, mut iy1, mut ixp, mut ind): (i32, i32, i32, i32, i32, i32);
    let mut fi: f32;

    // NEAREST_IX_IY
    macro_rules! nearest_ix_iy {
        ($i:expr, $fj:expr) => {
            fi = $i as f32 - xcen;
            ix = cvttss2si(amat[0] * fi + amat[2] * $fj + x_add);
            iy = cvttss2si(amat[1] * fi + amat[3] * $fj + y_add);
        };
    }
    // LINEAR_IX_IY
    macro_rules! linear_ix_iy {
        ($i:expr, $fj:expr) => {
            fi = $i as f32 - xcen;
            x = amat[0] * fi + amat[2] * $fj + x_add;
            y = amat[1] * fi + amat[3] * $fj + y_add;
            ix = cvttss2si(x);
            iy = cvttss2si(y);
        };
    }
    // NEAREST_ARRAY_DEN
    macro_rules! nearest_array_den {
        ($i:expr, $j:expr) => {
            den = crray[(ix - 1 + (iy - 1) * nx) as usize];
            dend = drray[($i - 1 + ($j - 1) * nx) as usize];
        };
    }
    // LINEAR_ARRAY_DEN
    macro_rules! linear_array_den {
        ($i:expr, $j:expr) => {
            ix1 = ix + 1;
            iy1 = iy + 1;
            fx = (1 + ix) as f32 - x;
            fx1 = 1. - fx;
            fy = (1 + iy) as f32 - y;
            fy1 = 1. - fy;
            ind = ix + (iy - 1) * nx;
            den = crray[(ind - 1) as usize] * fx * fy
                + crray[ind as usize] * fx1 * fy
                + crray[(ind + nx - 1) as usize] * fx * fy1
                + crray[(ind + nx) as usize] * fx1 * fy1;
            dend = drray[($i - 1 + ($j - 1) * nx) as usize];
        };
    }
    // NEAREST_ARRAY_DEN and LINEAR_ARRAY_DEN for the "NO RANGE TESTS" branch.
    // SAFETY (both): that branch is entered only when the source pixel of
    // both line ends satisfies 1 < ix < nx and 1 < iy < ny.  `ix`/`iy` are
    // truncations of `amat * fi + const`, which is monotone in `fi` (a
    // correctly rounded product and sum are monotone in each operand), so
    // every pixel between the ends lies in the same range, and the four
    // neighbours `ind - 1 .. ind + nx` of the linear case are inside
    // `nx * ny <= crray.len()`.  `dend`'s subscript has `nx1 <= i <= nx2 <=
    // nx` and `ny1 <= j <= ny2 <= ny`, inside `nx * ny <= drray.len()`
    // (all four asserted on entry).
    macro_rules! nearest_array_den_u {
        ($i:expr, $j:expr) => {
            den = unsafe { *crray.get_unchecked((ix - 1 + (iy - 1) * nx) as usize) };
            dend = unsafe { *drray.get_unchecked(($i - 1 + ($j - 1) * nx) as usize) };
        };
    }
    macro_rules! linear_array_den_u {
        ($i:expr, $j:expr) => {
            ix1 = ix + 1;
            iy1 = iy + 1;
            fx = (1 + ix) as f32 - x;
            fx1 = 1. - fx;
            fy = (1 + iy) as f32 - y;
            fy1 = 1. - fy;
            ind = ix + (iy - 1) * nx;
            den = unsafe {
                *crray.get_unchecked((ind - 1) as usize) * fx * fy
                    + *crray.get_unchecked(ind as usize) * fx1 * fy
                    + *crray.get_unchecked((ind + nx - 1) as usize) * fx * fy1
                    + *crray.get_unchecked((ind + nx) as usize) * fx1 * fy1
            };
            dend = unsafe { *drray.get_unchecked(($i - 1 + ($j - 1) * nx) as usize) };
        };
    }
    // PATCH_SUM
    macro_rules! patch_sum {
        ($i:expr, $iyp:expr) => {
            ixp = ($i - nx1) / nxy_patch;
            ind = ixp + $iyp * num_xpatch;
            sxa[ind as usize] += (den - dend) as f64;
            sx_sqa[ind as usize] += ((den - dend) * (den - dend)) as f64;
            num_pix_a[ind as usize] += 1;
        };
    }
    // PATCH_CCC_SUM
    macro_rules! patch_ccc_sum {
        ($i:expr, $iyp:expr) => {
            ixp = ($i - nx1) / nxy_patch;
            ind = ixp + $iyp * num_xpatch;
            sxa[ind as usize] += den as f64;
            sya[ind as usize] += dend as f64;
            sxya[ind as usize] += (den * dend) as f64;
            sx_sqa[ind as usize] += (den * den) as f64;
            sy_sqa[ind as usize] += (dend * dend) as f64;
            num_pix_a[ind as usize] += 1;
        };
    }

    let area = nx as i64 * ny as i64;
    assert!(
        nx1 >= 1
            && ny1 >= 1
            && nx2 <= nx
            && ny2 <= ny
            && crray.len() as i64 >= area
            && drray.len() as i64 >= area,
        "simplexDiff: arrays or limits out of range"
    );
    for j in ny1..=ny2 {
        /* Top of Y loop: get some constants for Y and compute source pixels for ends of
        line */
        let fj = j as f32 - ycen;
        let iyp = (j - ny1) / nxy_patch;
        fi = nx1 as f32 - xcen;
        ix1 = cvttss2si(amat[0] * fi + amat[2] * fj + x_add);
        iy1 = cvttss2si(amat[1] * fi + amat[3] * fj + y_add);
        fi = nx2 as f32 - xcen;
        ix = cvttss2si(amat[0] * fi + amat[2] * fj + x_add);
        iy = cvttss2si(amat[1] * fi + amat[3] * fj + y_add);
        if ix > 1 && ix < nx && iy > 1 && iy < ny && ix1 > 1 && ix1 < nx && iy1 > 1 && iy1 < ny {
            /* NO RANGE TESTS */
            if if_interp == 0 {
                /* Nearest simple difference */
                if old_diff != 0 {
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        nearest_array_den_u!(i, j);
                        *sx += ((den - dend) as f64).abs();
                    }
                    *num_pix += nx2 + 1 - nx1;
                } else if if_ccc == 0 {
                    /* Nearest patch difference */
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        nearest_array_den_u!(i, j);
                        patch_sum!(i, iyp);
                    }
                } else {
                    /* Nearest CCC */
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        nearest_array_den_u!(i, j);
                        patch_ccc_sum!(i, iyp);
                    }
                }
            } else if old_diff != 0 {
                /* Linear simple difference */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    linear_array_den_u!(i, j);
                    *sx += ((den - dend) as f64).abs();
                }
                *num_pix += nx2 + 1 - nx1;
            } else if if_ccc == 0 {
                /* Linear patch difference */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    linear_array_den_u!(i, j);
                    patch_sum!(i, iyp);
                }
            } else {
                /* Linear CCC */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    linear_array_den_u!(i, j);
                    patch_ccc_sum!(i, iyp);
                }
            }
        } else {
            /* RANGE TESTS REQUIRED */
            if if_interp == 0 {
                /* Nearest simple difference */
                if old_diff != 0 {
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        if ix >= 1 && ix <= nx && iy >= 1 && iy <= ny {
                            nearest_array_den!(i, j);
                            *sx += ((den - dend) as f64).abs();
                            *num_pix += 1;
                        }
                    }
                } else if if_ccc == 0 {
                    /* Nearest patch difference */
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        if ix >= 1 && ix <= nx && iy >= 1 && iy <= ny {
                            nearest_array_den!(i, j);
                            patch_sum!(i, iyp);
                        }
                    }
                } else {
                    /* Nearest CCC */
                    for i in nx1..=nx2 {
                        nearest_ix_iy!(i, fj);
                        if ix >= 1 && ix <= nx && iy >= 1 && iy <= ny {
                            nearest_array_den!(i, j);
                            patch_ccc_sum!(i, iyp);
                        }
                    }
                }
            } else if old_diff != 0 {
                /* Linear simple difference */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    if ix >= 1 && ix < nx && iy >= 1 && iy < ny {
                        linear_array_den!(i, j);
                        *sx += ((den - dend) as f64).abs();
                        *num_pix += 1;
                    }
                }
            } else if if_ccc == 0 {
                /* Linear patch difference */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    if ix >= 1 && ix < nx && iy >= 1 && iy < ny {
                        linear_array_den!(i, j);
                        patch_sum!(i, iyp);
                    }
                }
            } else {
                /* Linear CCC */
                for i in nx1..=nx2 {
                    linear_ix_iy!(i, fj);
                    if ix >= 1 && ix < nx && iy >= 1 && iy < ny {
                        linear_array_den!(i, j);
                        patch_ccc_sum!(i, iyp);
                    }
                }
            }
        }
    }
}
