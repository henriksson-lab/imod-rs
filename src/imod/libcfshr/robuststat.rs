//! Translation of `IMOD/libcfshr/robuststat.c` and its `cfsemshare.h` APIs.

use super::percentile::percentile_float;
use core::cmp::Ordering;

static mut VAL_ARRAY: *mut f32 = core::ptr::null_mut();
static mut INDEX_OFFSET: i32 = 0;

/// Original `intCompar` (`robuststat.c:34`).
unsafe fn int_compar(val1: *const i32, val2: *const i32) -> i32 {
    unsafe {
        if *val1 < *val2 {
            -1
        } else if *val1 > *val2 {
            1
        } else {
            0
        }
    }
}
/// Original `rsSortInts` (`robuststat.c:48`).
pub unsafe fn rs_sort_ints(x: *mut i32, n: i32) {
    unsafe {
        core::slice::from_raw_parts_mut(x, n as usize).sort_unstable_by(|a, b| {
            if int_compar(a, b) < 0 {
                Ordering::Less
            } else if int_compar(a, b) > 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    }
}
/// Original `rssortints` (`robuststat.c:56`).
pub unsafe fn rssortints(x: *mut i32, n: *mut i32) {
    unsafe { rs_sort_ints(x, *n) }
}

/// Original `floatCompar` (`robuststat.c:60`).
unsafe fn float_compar(val1: *const f32, val2: *const f32) -> i32 {
    unsafe {
        if *val1 < *val2 {
            -1
        } else if *val1 > *val2 {
            1
        } else {
            0
        }
    }
}
/// Original `rsSortFloats` (`robuststat.c:74`).
pub unsafe fn rs_sort_floats(x: *mut f32, n: i32) {
    unsafe {
        core::slice::from_raw_parts_mut(x, n as usize).sort_unstable_by(|a, b| {
            if float_compar(a, b) < 0 {
                Ordering::Less
            } else if float_compar(a, b) > 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    }
}
/// Original `rssortfloats` (`robuststat.c:82`).
pub unsafe fn rssortfloats(x: *mut f32, n: *mut i32) {
    unsafe { rs_sort_floats(x, *n) }
}

/// Original `indexedFloatCompar` (`robuststat.c:89`).
unsafe fn indexed_float_compar(val1: *const i32, val2: *const i32) -> i32 {
    unsafe {
        let i1 = *val1 - INDEX_OFFSET;
        let i2 = *val2 - INDEX_OFFSET;
        if *VAL_ARRAY.add(i1 as usize) < *VAL_ARRAY.add(i2 as usize) {
            -1
        } else if *VAL_ARRAY.add(i1 as usize) > *VAL_ARRAY.add(i2 as usize) {
            1
        } else {
            0
        }
    }
}
/// Original `rsSortIndexedFloats` (`robuststat.c:106`).
pub unsafe fn rs_sort_indexed_floats(x: *mut f32, index: *mut i32, n: i32) {
    unsafe {
        VAL_ARRAY = x;
        core::slice::from_raw_parts_mut(index, n as usize).sort_unstable_by(|a, b| {
            if indexed_float_compar(a, b) < 0 {
                Ordering::Less
            } else if indexed_float_compar(a, b) > 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        INDEX_OFFSET = 0;
    }
}
/// Original `rssortindexedfloats` (`robuststat.c:119`).
pub unsafe fn rssortindexedfloats(x: *mut f32, index: *mut i32, n: *mut i32) {
    unsafe {
        INDEX_OFFSET = 1;
        rs_sort_indexed_floats(x, index, *n);
    }
}
/// Original `rsSetSortIndexOffset` (`robuststat.c:129`).
pub unsafe fn rs_set_sort_index_offset(offset: i32) {
    unsafe {
        INDEX_OFFSET = offset;
    }
}

/// Original `rsMedianOfSorted` (`robuststat.c:136`).
pub unsafe fn rs_median_of_sorted(x: *mut f32, n: i32, median: *mut f32) {
    unsafe {
        *median = if n % 2 != 0 {
            *x.add((n / 2) as usize)
        } else {
            0.5 * (*x.add((n / 2 - 1) as usize) + *x.add((n / 2) as usize))
        };
    }
}
/// Original `rsmedianofsorted` (`robuststat.c:147`).
pub unsafe fn rsmedianofsorted(x: *mut f32, n: *mut i32, median: *mut f32) {
    unsafe {
        rs_median_of_sorted(x, *n, median);
    }
}
/// Original `rsMedian` (`robuststat.c:156`).
pub unsafe fn rs_median(x: *mut f32, n: i32, xsort: *mut f32, median: *mut f32) {
    unsafe {
        core::ptr::copy_nonoverlapping(x, xsort, n as usize);
        rs_sort_floats(xsort, n);
        rs_median_of_sorted(xsort, n, median);
    }
}
/// Original `rsmedian` (`robuststat.c:166`).
pub unsafe fn rsmedian(x: *mut f32, n: *mut i32, xsort: *mut f32, median: *mut f32) {
    unsafe {
        rs_median(x, *n, xsort, median);
    }
}

/// Original `rsFastMedianInPlace` (`robuststat.c:176`).
pub unsafe fn rs_fast_median_in_place(x: *mut f32, n: i32, median: *mut f32) {
    unsafe {
        *median = percentile_float((n + 1) / 2, x, n);
        if n % 2 == 0 {
            *median = (*median + percentile_float(n / 2 + 1, x, n)) / 2.;
        }
    }
}
/// Original `rsfastmedianinplace` (`robuststat.c:184`).
pub unsafe fn rsfastmedianinplace(x: *mut f32, n: *mut i32, median: *mut f32) {
    unsafe {
        rs_fast_median_in_place(x, *n, median);
    }
}
/// Original `rsFastMedian` (`robuststat.c:192`).
pub unsafe fn rs_fast_median(x: *mut f32, n: i32, xjumble: *mut f32, median: *mut f32) {
    unsafe {
        core::ptr::copy_nonoverlapping(x, xjumble, n as usize);
        rs_fast_median_in_place(xjumble, n, median);
    }
}
/// Original `rsfastmedian` (`robuststat.c:200`).
pub unsafe fn rsfastmedian(x: *mut f32, n: *mut i32, xjumble: *mut f32, median: *mut f32) {
    unsafe {
        rs_fast_median(x, *n, xjumble, median);
    }
}

/// Original `rsPercentileOfSorted` (`robuststat.c:209`).
pub unsafe fn rs_percentile_of_sorted(x: *mut f32, n: i32, fraction: f32, pctile: *mut f32) {
    let real_ind = n as f32 * fraction - 0.5;
    let lower_ind = real_ind.floor() as i32;
    let f = real_ind - lower_ind as f32;
    unsafe {
        *pctile = if lower_ind < 0 {
            *x
        } else if lower_ind >= n - 1 {
            *x.add((n - 1) as usize)
        } else {
            (1. - f) * *x.add(lower_ind as usize) + f * *x.add((lower_ind + 1) as usize)
        };
    }
}
/// Original `rspercentileofsorted` (`robuststat.c:228`).
pub unsafe fn rspercentileofsorted(x: *mut f32, n: *mut i32, fraction: *mut f32, pctile: *mut f32) {
    unsafe {
        rs_percentile_of_sorted(x, *n, *fraction, pctile);
    }
}

/// Original `rsMADN` (`robuststat.c:240`).
pub unsafe fn rs_madn(x: *mut f32, n: i32, median: f32, tmp: *mut f32, madn: *mut f32) {
    unsafe {
        for i in 0..n as usize {
            *tmp.add(i) = (*x.add(i) - median).abs();
        }
        rs_sort_floats(tmp, n);
        rs_median_of_sorted(tmp, n, madn);
        *madn /= 0.6745;
    }
}
/// Original `rsmadn` (`robuststat.c:254`).
pub unsafe fn rsmadn(x: *mut f32, n: *mut i32, median: *mut f32, tmp: *mut f32, madn: *mut f32) {
    unsafe {
        rs_madn(x, *n, *median, tmp, madn);
    }
}
/// Original `rsFastMADN` (`robuststat.c:264`).
pub unsafe fn rs_fast_madn(x: *mut f32, n: i32, median: f32, tmp: *mut f32, madn: *mut f32) {
    unsafe {
        for i in 0..n as usize {
            *tmp.add(i) = (*x.add(i) - median).abs();
        }
        rs_fast_median_in_place(tmp, n, madn);
        *madn /= 0.6745;
    }
}
/// Original `rsfastmadn` (`robuststat.c:276`).
pub unsafe fn rsfastmadn(
    x: *mut f32,
    n: *mut i32,
    median: *mut f32,
    tmp: *mut f32,
    madn: *mut f32,
) {
    unsafe {
        rs_fast_madn(x, *n, *median, tmp, madn);
    }
}

/// Original `rsMadMedianOutliers` (`robuststat.c:290`).
pub unsafe fn rs_mad_median_outliers(x: *mut f32, n: i32, kcrit: f32, out: *mut f32) {
    let (mut median, mut madn) = (0., 0.);
    unsafe {
        rs_fast_median(x, n, out, &mut median);
        rs_fast_madn(x, n, median, out, &mut madn);
        for i in 0..n as usize {
            *out.add(i) = if madn != 0. && (*x.add(i) - median).abs() / madn > kcrit {
                if *x.add(i) > median { 1. } else { -1. }
            } else {
                0.
            };
        }
    }
}
/// Original `rsmadmedianoutliers` (`robuststat.c:310`).
pub unsafe fn rsmadmedianoutliers(x: *mut f32, n: *mut i32, kcrit: *mut f32, out: *mut f32) {
    unsafe {
        rs_mad_median_outliers(x, *n, *kcrit, out);
    }
}

/// Original `rsTrimmedMeanOfSorted` (`robuststat.c:321`).
pub unsafe fn rs_trimmed_mean_of_sorted(x: *mut f32, n: i32, gamma: f32, trmean: *mut f32) {
    let cut = (gamma * n as f32) as i32;
    let mut sum = 0.;
    unsafe {
        *trmean = 0.;
        for i in cut..n - cut {
            sum += *x.add(i as usize) as f64;
        }
        if sum != 0. {
            *trmean = (sum / (n - 2 * cut) as f64) as f32;
        }
    }
}
/// Original `rstrimmedmeanofsorted` (`robuststat.c:336`).
pub unsafe fn rstrimmedmeanofsorted(x: *mut f32, n: *mut i32, gamma: *mut f32, median: *mut f32) {
    unsafe {
        rs_trimmed_mean_of_sorted(x, *n, *gamma, median);
    }
}
/// Original `rsTrimmedMean` (`robuststat.c:347`).
pub unsafe fn rs_trimmed_mean(x: *mut f32, n: i32, gamma: f32, xsort: *mut f32, trmean: *mut f32) {
    unsafe {
        core::ptr::copy_nonoverlapping(x, xsort, n as usize);
        rs_sort_floats(xsort, n);
        rs_trimmed_mean_of_sorted(xsort, n, gamma, trmean);
    }
}
/// Original `rstrimmedmean` (`robuststat.c:358`).
pub unsafe fn rstrimmedmean(
    x: *mut f32,
    n: *mut i32,
    gamma: *mut f32,
    xsort: *mut f32,
    median: *mut f32,
) {
    unsafe {
        rs_trimmed_mean(x, *n, *gamma, xsort, median);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn robust_statistics_reject_extreme_outliers() {
        unsafe {
            let mut values = [-1000., 1., 2., 3., 4., 1000.];
            let mut work = [0.; 6];
            let (mut median, mut madn) = (0., 0.);
            rs_fast_median(values.as_mut_ptr(), 6, work.as_mut_ptr(), &mut median);
            rs_fast_madn(values.as_mut_ptr(), 6, median, work.as_mut_ptr(), &mut madn);
            let mut out = [0.; 6];
            rs_mad_median_outliers(values.as_mut_ptr(), 6, 2.24, out.as_mut_ptr());
            assert_eq!(median, 2.5);
            assert!(madn > 2. && madn < 3.);
            assert_eq!(out, [-1., 0., 0., 0., 0., 1.]);
        }
    }
    #[test]
    fn sorting_percentile_and_trimmed_mean_cover_edges() {
        unsafe {
            let mut values = [5., 1., 4., 2., 3.];
            rs_sort_floats(values.as_mut_ptr(), 5);
            assert_eq!(values, [1., 2., 3., 4., 5.]);
            let mut pct = 0.;
            rs_percentile_of_sorted(values.as_mut_ptr(), 5, 0., &mut pct);
            assert_eq!(pct, 1.);
            rs_percentile_of_sorted(values.as_mut_ptr(), 5, 1., &mut pct);
            assert_eq!(pct, 5.);
            let mut mean = 0.;
            rs_trimmed_mean_of_sorted(values.as_mut_ptr(), 5, 0.2, &mut mean);
            assert_eq!(mean, 3.);
        }
    }
}
