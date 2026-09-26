//! Translation of `IMOD/libcfshr/percentile.c`.
//!
//! # Why the partition loops use `get_unchecked`
//!
//! `percentileFloat`/`percentileInt` are called **once per pixel** from
//! `sliceMedianFilter`'s general path (`sliceproc.rs:366,369,412,415`), and
//! `clip median -3 -n 3` on a 671 MB volume is the slowest case in
//! `BENCHMARKS.md` relative to native.  The loops below are a Hoare partition
//! whose indices move under data-dependent comparisons, so LLVM cannot prove
//! them in range and emits a bounds check on every one of the eight hot
//! accesses -- roughly doubling the instructions in the tightest loop against
//! the C's `r[j]`.  The user authorised `get_unchecked` where it is needed for
//! performance (2026-09-24).
//!
//! **The soundness argument, which is what makes this legitimate rather than
//! merely fast.**  One checked precondition is hoisted out of the loops:
//! `count as usize <= values.len()`.  Given it, every index stays inside
//! `[low, high]`, and `[low, high]` never leaves `[0, count - 1]`:
//!
//! * `low` starts at `0`, `high` at `count - 1`, and they only ever move to
//!   `left + 1` / `left - 1` / `right + 1` / `right - 1` for a `left`/`right`
//!   already shown to be in `[low, high]`.  The loop runs only while
//!   `low <= selected <= high`, so the interval is non-empty on entry.
//! * `selected` is clamped to `[0, count - 1]` by the `B3DCLAMP` above.
//! * `left` starts at `low` and only increments under `left < right`;
//!   `right` starts at `high` and, in the guarded scans, only decrements under
//!   `left < right`.  So both stay in `[low, high]`.
//! * The two **unguarded** inner scans (`while values[right] > temporary` and
//!   `while values[left] < temporary`) terminate on a sentinel rather than a
//!   bound: the element swapped into `values[low]` (resp. `values[high]`) is
//!   `temporary` itself, and `temporary > temporary` is false, so the scan
//!   stops there at the latest.  After the first `values[left] = values[right]`
//!   overwrites that sentinel it is replaced by a value the preceding scan just
//!   proved to be `<= temporary` (resp. `>= temporary`), which stops the scan
//!   equally.  A NaN makes every comparison false, which stops the scan
//!   immediately -- it cannot run past the end.
//!
//! The algorithm itself is untouched, deliberately: the **array permutation is
//! observable**, because `sliceproc.rs:369` calls this a second time on the
//! same buffer for an even-sized window.  Removing the bounds checks keeps the
//! permutation bit-identical; a different selection algorithm would not.

/// Original `percentileFloat` (`percentile.c:34`).
pub fn percentile_float(mut selected: i32, values: &mut [f32], count: i32) -> f32 {
    let mut low = 0;
    let mut high = count - 1;
    if count <= 0 {
        return 0.0;
    }
    if count == 1 {
        return values[0];
    }
    // `B3DCLAMP` is `B3DMAX(minv, B3DMIN(maxv, val))`, not `max` then `min`.
    selected = 1.max(count.min(selected)) - 1;
    // The one checked precondition the `get_unchecked` calls below rest on;
    // see the module header for the rest of the argument.  Hoisted out of
    // both loops, so it costs one comparison per call, not one per access.
    assert!(
        count as usize <= values.len(),
        "percentile: count {count} exceeds the {} values given",
        values.len()
    );
    while high >= selected && selected >= low {
        // SAFETY: every index below is in `[low, high]` which is inside
        // `[0, count - 1]`, and `count <= values.len()` was just checked.
        unsafe {
            let mut left = low;
            let mut right = high;
            let temporary = *values.get_unchecked(selected as usize);
            if selected > count / 2 {
                *values.get_unchecked_mut(selected as usize) = *values.get_unchecked(low as usize);
                *values.get_unchecked_mut(low as usize) = temporary;
                while left < right {
                    let (entry_left, entry_right) = (left, right);
                    while *values.get_unchecked(right as usize) > temporary {
                        right -= 1;
                    }
                    *values.get_unchecked_mut(left as usize) =
                        *values.get_unchecked(right as usize);
                    while left < right && *values.get_unchecked(left as usize) <= temporary {
                        left += 1;
                    }
                    *values.get_unchecked_mut(right as usize) =
                        *values.get_unchecked(left as usize);
                    if left == entry_left && right == entry_right {
                        // BUGS.md "percentileFloat / percentileInt do not
                        // terminate on NaN input": only a NaN stops both
                        // scans; the C spins here forever.
                        return percentile_float_nan_fallback(selected, values, count);
                    }
                }
                *values.get_unchecked_mut(left as usize) = temporary;
                if selected < left {
                    high = left - 1;
                } else {
                    low = left + 1;
                }
            } else {
                *values.get_unchecked_mut(selected as usize) = *values.get_unchecked(high as usize);
                *values.get_unchecked_mut(high as usize) = temporary;
                while left < right {
                    let (entry_left, entry_right) = (left, right);
                    while *values.get_unchecked(left as usize) < temporary {
                        left += 1;
                    }
                    *values.get_unchecked_mut(right as usize) =
                        *values.get_unchecked(left as usize);
                    while left < right && *values.get_unchecked(right as usize) >= temporary {
                        right -= 1;
                    }
                    *values.get_unchecked_mut(left as usize) =
                        *values.get_unchecked(right as usize);
                    if left == entry_left && right == entry_right {
                        // As above: a NaN stalled both scans.
                        return percentile_float_nan_fallback(selected, values, count);
                    }
                }
                *values.get_unchecked_mut(right as usize) = temporary;
                if selected > right {
                    low = right + 1;
                } else {
                    high = right - 1;
                }
            }
        }
    }
    values[selected as usize]
}

/// Defined behaviour for the non-terminating NaN case of `percentileFloat`
/// (fixed in translation, 2026-09-26; `BUGS.md`).  Not in the source.
///
/// A partition pass in which *neither* scan moves is impossible for ordered
/// data: the first scan stops on an element that the second scan's
/// comparison then accepts, so one index always advances.  Only a NaN makes
/// both comparisons false, and the C then loops forever with the array in a
/// fixed state.  Rather than spin, the selection is finished by sorting the
/// first `count` values with every NaN ordered above `+inf` and returning
/// item `selected` (0-based) of that order.  Input without a NaN never gets
/// here, so its selection and its array permutation are unchanged; input with
/// a NaN on which the C does terminate also never gets here and gives the C's
/// answer.
fn percentile_float_nan_fallback(selected: i32, values: &mut [f32], count: i32) -> f32 {
    values[..count as usize].sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap(),
        (false, true) => std::cmp::Ordering::Less,
        (true, false) => std::cmp::Ordering::Greater,
        (true, true) => std::cmp::Ordering::Equal,
    });
    values[selected as usize]
}

/// Original Fortran wrapper `percentilefloat` (`percentile.c:95`).
pub fn percentilefloat(selected: &i32, values: &mut [f32], count: &i32) -> f64 {
    percentile_float(*selected, values, *count) as f64
}

/// Original `percentileInt` (`percentile.c:103`).
pub fn percentile_int(mut selected: i32, values: &mut [i32], count: i32) -> i32 {
    let mut low = 0;
    let mut high = count - 1;
    if count <= 0 {
        return 0;
    }
    if count == 1 {
        return values[0];
    }
    // `B3DCLAMP` is `B3DMAX(minv, B3DMIN(maxv, val))`, not `max` then `min`.
    selected = 1.max(count.min(selected)) - 1;
    // The one checked precondition the `get_unchecked` calls below rest on;
    // see the module header for the rest of the argument.  Hoisted out of
    // both loops, so it costs one comparison per call, not one per access.
    assert!(
        count as usize <= values.len(),
        "percentile: count {count} exceeds the {} values given",
        values.len()
    );
    while high >= selected && selected >= low {
        // SAFETY: every index below is in `[low, high]` which is inside
        // `[0, count - 1]`, and `count <= values.len()` was just checked.
        unsafe {
            let mut left = low;
            let mut right = high;
            let temporary = *values.get_unchecked(selected as usize);
            if selected > count / 2 {
                *values.get_unchecked_mut(selected as usize) = *values.get_unchecked(low as usize);
                *values.get_unchecked_mut(low as usize) = temporary;
                while left < right {
                    while *values.get_unchecked(right as usize) > temporary {
                        right -= 1;
                    }
                    *values.get_unchecked_mut(left as usize) =
                        *values.get_unchecked(right as usize);
                    while left < right && *values.get_unchecked(left as usize) <= temporary {
                        left += 1;
                    }
                    *values.get_unchecked_mut(right as usize) =
                        *values.get_unchecked(left as usize);
                }
                *values.get_unchecked_mut(left as usize) = temporary;
                if selected < left {
                    high = left - 1;
                } else {
                    low = left + 1;
                }
            } else {
                *values.get_unchecked_mut(selected as usize) = *values.get_unchecked(high as usize);
                *values.get_unchecked_mut(high as usize) = temporary;
                while left < right {
                    while *values.get_unchecked(left as usize) < temporary {
                        left += 1;
                    }
                    *values.get_unchecked_mut(right as usize) =
                        *values.get_unchecked(left as usize);
                    while left < right && *values.get_unchecked(right as usize) >= temporary {
                        right -= 1;
                    }
                    *values.get_unchecked_mut(left as usize) =
                        *values.get_unchecked(right as usize);
                }
                *values.get_unchecked_mut(right as usize) = temporary;
                if selected > right {
                    low = right + 1;
                } else {
                    high = right - 1;
                }
            }
        }
    }
    values[selected as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_selection_clamps_ranks_and_handles_extreme_duplicate_values() {
        let mut floats = [9.0_f32, 1.0, 1.0, 1.0, 4.0, 8.0, 9.0, 9.0];
        assert_eq!(percentile_float(1, &mut floats, 8), 1.0);
        let mut floats = [9.0_f32, 1.0, 1.0, 1.0, 4.0, 8.0, 9.0, 9.0];
        assert_eq!(percentile_float(100, &mut floats, 8), 9.0);
        let mut integers = [7_i32, 2, 2, 4, 9, 1];
        assert_eq!(percentile_int(3, &mut integers, 6), 2);
        assert_eq!(percentile_float(1, &mut [], 0), 0.0);
    }

    /// `BUGS.md`: the C never returns on these.  Defined behaviour: NaN
    /// orders above every number, so the median of {3,1,NaN,2,5} is 3 and the
    /// top item is NaN.
    #[test]
    fn nan_input_terminates_with_nan_ordered_highest() {
        let mut v = [3.0_f32, 1.0, f32::NAN, 2.0, 5.0];
        assert_eq!(percentile_float(3, &mut v, 5), 3.0);
        let mut v = [3.0_f32, 1.0, f32::NAN, 2.0, 5.0];
        assert!(percentile_float(5, &mut v, 5).is_nan());
        let mut v = [3.0_f32, 1.0, f32::NAN, 2.0, 5.0];
        assert_eq!(percentile_float(1, &mut v, 5), 1.0);
        let mut v = [f32::NAN; 9];
        assert!(percentile_float(5, &mut v, 9).is_nan());
        for s in 1..=9 {
            let mut v = [4.0_f32, f32::NAN, 7.0, -1.0, f32::NAN, 0.5, 9.0, 2.0, 3.0];
            let _ = percentile_float(s, &mut v, 9);
        }
    }
}
