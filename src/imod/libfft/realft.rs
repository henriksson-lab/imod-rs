//! Translation of `IMOD/libfft/realft.c`.

use super::cmplft;

/// C `realft`.
/// The C takes `even`/`x` and `odd`/`y` pointers; `odd` is the bias of the
/// second stream into `data` -- 1 for interleaved storage, `ny` for
/// `todfft`'s transposed layout.
pub fn realft(data: &mut [f32], odd: usize, n: i32, dim: &mut [i32; 6]) {
    assert!(data.len() >= dim[1] as usize);
    let twopi = 6.2831853_f32;
    let two_n = (2 * n) as f32;
    cmplft(data, odd, n, dim);
    let len = data.len();
    let dp = data.as_mut_ptr();
    let total = dim[1];
    let along = dim[2];
    let limit = dim[3];
    let extent = dim[4] - 1;
    let between = dim[5];
    let over_two = n / 2 + 1;
    if over_two >= 2 {
        for index in 2..=over_two {
            let angle = twopi * (index - 1) as f32 / two_n;
            let co = (angle as f64).cos() as f32;
            let si = (angle as f64).sin() as f32;
            let first = (index - 1) * along + 1;
            let delta = (n + 2 - 2 * index) * along;
            let mut row = first;
            while row <= total {
                let row_end = row + extent;
                let mut point = row - 1;
                // Unchecked access (see `mdftkd.rs`'s module comment for the pattern):
                // `point` runs from `row - 1` in steps of `between > 0` up to its last value below
                // `row_end`, and every access is `point`, `point + delta` or either plus `odd`, so
                // checking the first point (and its partner) is non-negative and the last
                // point plus `max(delta, 0)` plus `odd` is below `len` bounds them all.
                if point < row_end {
                    assert!(
                        point >= 0
                            && point + delta >= 0
                            && between > 0
                            && ((point + (row_end - 1 - point) / between * between + delta.max(0))
                                as usize)
                                .checked_add(odd)
                                .is_some_and(|last| last < len)
                    );
                }
                if between == 1 && point < row_end {
                    // Unit stride, as in `todfft`'s transposed layout.  The
                    // same element operations in the same order, over a
                    // counted loop: gcc versions the source's loop for unit
                    // stride and vectorizes it; a counted loop over the four
                    // walkers lets LLVM do the same (with its own run-time
                    // overlap checks).  Lanes compute exactly the scalar
                    // IEEE operations; nothing is reassociated or contracted.
                    // The walkers are in bounds by the assertion above.
                    unsafe {
                        let even_k = dp.add(point as usize);
                        let even_l = dp.add((point + delta) as usize);
                        let odd_k = even_k.add(odd);
                        let odd_l = even_l.add(odd);
                        for i in 0..(row_end - point) as usize {
                            let a = (*even_l.add(i) + *even_k.add(i)) / 2.0;
                            let c = (*even_l.add(i) - *even_k.add(i)) / 2.0;
                            let b = (*odd_l.add(i) + *odd_k.add(i)) / 2.0;
                            let d = (*odd_l.add(i) - *odd_k.add(i)) / 2.0;
                            let e = c * si + b * co;
                            let f = c * co - b * si;
                            *even_k.add(i) = a + e;
                            *even_l.add(i) = a - e;
                            *odd_k.add(i) = f - d;
                            *odd_l.add(i) = f + d;
                        }
                    }
                    point = row_end;
                }
                while point < row_end {
                    unsafe {
                        let paired = point + delta;
                        let a = (*dp.add(paired as usize) + *dp.add(point as usize)) / 2.0;
                        let c = (*dp.add(paired as usize) - *dp.add(point as usize)) / 2.0;
                        let b =
                            (*dp.add(paired as usize + odd) + *dp.add(point as usize + odd)) / 2.0;
                        let d =
                            (*dp.add(paired as usize + odd) - *dp.add(point as usize + odd)) / 2.0;
                        let e = c * si + b * co;
                        let f = c * co - b * si;
                        *dp.add(point as usize) = a + e;
                        *dp.add(paired as usize) = a - e;
                        *dp.add(point as usize + odd) = f - d;
                        *dp.add(paired as usize + odd) = f + d;
                        point += between;
                    }
                }
                row += limit;
            }
        }
    }
    if n < 1 {
        return;
    }
    let delta = n * along;
    let mut row = 1;
    while row <= total {
        let row_end = row + extent;
        let mut point = row - 1;
        if point < row_end {
            assert!(
                point >= 0
                    && point + delta >= 0
                    && between > 0
                    && ((point + (row_end - 1 - point) / between * between + delta.max(0))
                        as usize)
                        .checked_add(odd)
                        .is_some_and(|last| last < len)
            );
        }
        while point < row_end {
            unsafe {
                let paired = point + delta;
                *dp.add(paired as usize) = *dp.add(point as usize) - *dp.add(point as usize + odd);
                *dp.add(paired as usize + odd) = 0.0;
                *dp.add(point as usize) += *dp.add(point as usize + odd);
                *dp.add(point as usize + odd) = 0.0;
                point += between;
            }
        }
        row += limit;
    }
}
