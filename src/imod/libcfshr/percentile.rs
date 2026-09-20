//! Translation of `IMOD/libcfshr/percentile.c`.

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
    while high >= selected && selected >= low {
        let mut left = low;
        let mut right = high;
        let temporary = values[selected as usize];
        if selected > count / 2 {
            values[selected as usize] = values[low as usize];
            values[low as usize] = temporary;
            while left < right {
                while values[right as usize] > temporary {
                    right -= 1;
                }
                values[left as usize] = values[right as usize];
                while left < right && values[left as usize] <= temporary {
                    left += 1;
                }
                values[right as usize] = values[left as usize];
            }
            values[left as usize] = temporary;
            if selected < left {
                high = left - 1;
            } else {
                low = left + 1;
            }
        } else {
            values[selected as usize] = values[high as usize];
            values[high as usize] = temporary;
            while left < right {
                while values[left as usize] < temporary {
                    left += 1;
                }
                values[right as usize] = values[left as usize];
                while left < right && values[right as usize] >= temporary {
                    right -= 1;
                }
                values[left as usize] = values[right as usize];
            }
            values[right as usize] = temporary;
            if selected > right {
                low = right + 1;
            } else {
                high = right - 1;
            }
        }
    }
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
    while high >= selected && selected >= low {
        let mut left = low;
        let mut right = high;
        let temporary = values[selected as usize];
        if selected > count / 2 {
            values[selected as usize] = values[low as usize];
            values[low as usize] = temporary;
            while left < right {
                while values[right as usize] > temporary {
                    right -= 1;
                }
                values[left as usize] = values[right as usize];
                while left < right && values[left as usize] <= temporary {
                    left += 1;
                }
                values[right as usize] = values[left as usize];
            }
            values[left as usize] = temporary;
            if selected < left {
                high = left - 1;
            } else {
                low = left + 1;
            }
        } else {
            values[selected as usize] = values[high as usize];
            values[high as usize] = temporary;
            while left < right {
                while values[left as usize] < temporary {
                    left += 1;
                }
                values[right as usize] = values[left as usize];
                while left < right && values[right as usize] >= temporary {
                    right -= 1;
                }
                values[left as usize] = values[right as usize];
            }
            values[right as usize] = temporary;
            if selected > right {
                low = right + 1;
            } else {
                high = right - 1;
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
}
