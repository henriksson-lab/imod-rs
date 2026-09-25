//! Translation of `IMOD/libfft/hermft.c`.

use super::cmplft;

/// C `hermft` over IMOD's interleaved real/imaginary float storage.
/// The C takes `even`/`x` and `odd`/`y` pointers; `odd` is the bias of the
/// second stream into `data` -- 1 for interleaved storage, `ny` for
/// `todfft`'s transposed layout.
pub fn hermft(data: &mut [f32], odd: usize, n: i32, dim: &mut [i32; 6]) {
    assert!(data.len() >= dim[1] as usize);

    let len = data.len();
    let dp = data.as_mut_ptr();
    let twopi = 6.2831853_f32;
    let two_n = (2 * n) as f32;
    let total = dim[1];
    let along = dim[2];
    let limit = dim[3];
    let extent = dim[4] - 1;
    let between = dim[5];
    let mut first = 1;
    while first <= total {
        let end = first + extent;
        let mut index = first - 1;
        if index < end {
            assert!(
                index >= 0
                    && index + 0_i32 >= 0
                    && between > 0
                    && ((index + (end - 1 - index) / between * between + 0_i32.max(0)) as usize)
                        .checked_add(odd)
                        .is_some_and(|last| last < len)
            );
        }
        while index < end {
            unsafe {
                let a = *dp.add(index as usize);
                let b = *dp.add(index as usize + odd);
                *dp.add(index as usize) = a + b;
                *dp.add(index as usize + odd) = a - b;
                index += between;
            }
        }
        first += limit;
    }
    let over_two = n / 2 + 1;
    if over_two < 2 {
        return;
    }
    for source in 2..=over_two {
        let angle = twopi * (source - 1) as f32 / two_n;
        let co = (angle as f64).cos() as f32;
        let si = (angle as f64).sin() as f32;
        let delta = (n + 2 - 2 * source) * along;
        let mut row = (source - 1) * along + 1;
        while row <= total {
            let end = row + extent;
            let mut index = row - 1;
            // Unchecked access (see `mdftkd.rs`'s module comment for the pattern):
            // `index` runs from `row - 1` in steps of `between > 0` up to its last value below
            // `end`, and every access is `index`, `index + delta` or either plus `odd`, so
            // checking the first point (and its partner) is non-negative and the last
            // point plus `max(delta, 0)` plus `odd` is below `len` bounds them all.
            if index < end {
                assert!(
                    index >= 0
                        && index + delta >= 0
                        && between > 0
                        && ((index + (end - 1 - index) / between * between + delta.max(0))
                            as usize)
                            .checked_add(odd)
                            .is_some_and(|last| last < len)
                );
            }
            if between == 1 && index < end {
                // Unit stride (`todfft`'s transposed layout): the same
                // element operations over a counted loop so LLVM can
                // vectorize it, as gcc does the source's; see `realft.rs`.
                unsafe {
                    let x_k = dp.add(index as usize);
                    let x_l = dp.add((index + delta) as usize);
                    let y_k = x_k.add(odd);
                    let y_l = x_l.add(odd);
                    for i in 0..(end - index) as usize {
                        let a = *x_k.add(i) + *x_l.add(i);
                        let b = *x_k.add(i) - *x_l.add(i);
                        let c = *y_k.add(i) + *y_l.add(i);
                        let d = *y_k.add(i) - *y_l.add(i);
                        let e = b * co + c * si;
                        let f = b * si - c * co;
                        *x_k.add(i) = a + f;
                        *x_l.add(i) = a - f;
                        *y_k.add(i) = e + d;
                        *y_l.add(i) = e - d;
                    }
                }
                index = end;
            }
            while index < end {
                unsafe {
                    let paired = index + delta;
                    let a = *dp.add(index as usize) + *dp.add(paired as usize);
                    let b = *dp.add(index as usize) - *dp.add(paired as usize);
                    let c = *dp.add(index as usize + odd) + *dp.add(paired as usize + odd);
                    let d = *dp.add(index as usize + odd) - *dp.add(paired as usize + odd);
                    let e = b * co + c * si;
                    let f = b * si - c * co;
                    *dp.add(index as usize) = a + f;
                    *dp.add(paired as usize) = a - f;
                    *dp.add(index as usize + odd) = e + d;
                    *dp.add(paired as usize + odd) = e - d;
                    index += between;
                }
            }
            row += limit;
        }
    }
    cmplft(data, odd, n, dim);
}

#[cfg(test)]
mod tests {
    use super::hermft;

    #[test]
    fn length_two_hermitian_pair_is_combined_in_interleaved_storage() {
        let mut values = [3.5_f32, -1.25];
        let mut dimensions = [0_i32, 2, 1, 2, 1, 2];

        hermft(&mut values, 1, 1, &mut dimensions);

        assert_eq!(values, [2.25, 4.75]);
    }
}
