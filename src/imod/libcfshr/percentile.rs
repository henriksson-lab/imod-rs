//! Translation of `IMOD/libcfshr/percentile.c`.
#![allow(dead_code)]

/// Original `percentileFloat` (`percentile.c:34`).
pub unsafe fn percentile_float(mut selected: i32, values: *mut f32, count: i32) -> f32 {
    unsafe {
        let mut low = 0;
        let mut high = count - 1;
        if count <= 0 {
            return 0.0;
        }
        if count == 1 {
            return *values;
        }
        selected = selected.clamp(1, count) - 1;
        while high >= selected && selected >= low {
            let mut left = low;
            let mut right = high;
            let temporary = *values.add(selected as usize);
            if selected > count / 2 {
                *values.add(selected as usize) = *values.add(low as usize);
                *values.add(low as usize) = temporary;
                while left < right {
                    while *values.add(right as usize) > temporary {
                        right -= 1;
                    }
                    *values.add(left as usize) = *values.add(right as usize);
                    while left < right && *values.add(left as usize) <= temporary {
                        left += 1;
                    }
                    *values.add(right as usize) = *values.add(left as usize);
                }
                *values.add(left as usize) = temporary;
                if selected < left {
                    high = left - 1;
                } else {
                    low = left + 1;
                }
            } else {
                *values.add(selected as usize) = *values.add(high as usize);
                *values.add(high as usize) = temporary;
                while left < right {
                    while *values.add(left as usize) < temporary {
                        left += 1;
                    }
                    *values.add(right as usize) = *values.add(left as usize);
                    while left < right && *values.add(right as usize) >= temporary {
                        right -= 1;
                    }
                    *values.add(left as usize) = *values.add(right as usize);
                }
                *values.add(right as usize) = temporary;
                if selected > right {
                    low = right + 1;
                } else {
                    high = right - 1;
                }
            }
        }
        *values.add(selected as usize)
    }
}

/// Original Fortran wrapper `percentilefloat` (`percentile.c:95`).
pub unsafe fn percentilefloat(selected: *mut i32, values: *mut f32, count: *mut i32) -> f64 {
    unsafe { percentile_float(*selected, values, *count) as f64 }
}

/// Original `percentileInt` (`percentile.c:103`).
pub unsafe fn percentile_int(mut selected: i32, values: *mut i32, count: i32) -> i32 {
    unsafe {
        let mut low = 0;
        let mut high = count - 1;
        if count <= 0 {
            return 0;
        }
        if count == 1 {
            return *values;
        }
        selected = selected.clamp(1, count) - 1;
        while high >= selected && selected >= low {
            let mut left = low;
            let mut right = high;
            let temporary = *values.add(selected as usize);
            if selected > count / 2 {
                *values.add(selected as usize) = *values.add(low as usize);
                *values.add(low as usize) = temporary;
                while left < right {
                    while *values.add(right as usize) > temporary {
                        right -= 1;
                    }
                    *values.add(left as usize) = *values.add(right as usize);
                    while left < right && *values.add(left as usize) <= temporary {
                        left += 1;
                    }
                    *values.add(right as usize) = *values.add(left as usize);
                }
                *values.add(left as usize) = temporary;
                if selected < left {
                    high = left - 1;
                } else {
                    low = left + 1;
                }
            } else {
                *values.add(selected as usize) = *values.add(high as usize);
                *values.add(high as usize) = temporary;
                while left < right {
                    while *values.add(left as usize) < temporary {
                        left += 1;
                    }
                    *values.add(right as usize) = *values.add(left as usize);
                    while left < right && *values.add(right as usize) >= temporary {
                        right -= 1;
                    }
                    *values.add(left as usize) = *values.add(right as usize);
                }
                *values.add(right as usize) = temporary;
                if selected > right {
                    low = right + 1;
                } else {
                    high = right - 1;
                }
            }
        }
        *values.add(selected as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_selection_clamps_ranks_and_handles_extreme_duplicate_values() {
        unsafe {
            let mut floats = [9.0_f32, 1.0, 1.0, 1.0, 4.0, 8.0, 9.0, 9.0];
            assert_eq!(percentile_float(1, floats.as_mut_ptr(), 8), 1.0);
            let mut floats = [9.0_f32, 1.0, 1.0, 1.0, 4.0, 8.0, 9.0, 9.0];
            assert_eq!(percentile_float(100, floats.as_mut_ptr(), 8), 9.0);
            let mut integers = [7_i32, 2, 2, 4, 9, 1];
            assert_eq!(percentile_int(3, integers.as_mut_ptr(), 6), 2);
            assert_eq!(percentile_float(1, core::ptr::null_mut(), 0), 0.0);
        }
    }
}
