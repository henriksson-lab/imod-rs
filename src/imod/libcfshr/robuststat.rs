//! Translation of `IMOD/libcfshr/robuststat.c` and its `cfsemshare.h` APIs.

use super::percentile::percentile_float;
use core::cell::Cell;
use core::cmp::Ordering;

thread_local! {
    /// Original `indexOffset` (`robuststat.c:97`).
    static INDEX_OFFSET: Cell<i32> = const { Cell::new(0) };
}

/// Original `intCompar` (`robuststat.c:42`).
fn int_compar(val1: &i32, val2: &i32) -> i32 {
    if *val1 < *val2 {
        -1
    } else if *val1 > *val2 {
        1
    } else {
        0
    }
}
/// Original `rsSortInts` (`robuststat.c:56`).
///
/// `qsort` makes no ordering promise for elements the comparison calls equal,
/// but the glibc this tree is verified against orders them stably, and
/// `rsSortIndexedFloats` makes that order observable (a C-versus-C
/// differential over 40 trials matches only with a stable sort), so all three
/// sorts here are stable.
pub fn rs_sort_ints(x: &mut [i32], n: i32) {
    x[..n as usize].sort_by(|a, b| {
        if int_compar(a, b) < 0 {
            Ordering::Less
        } else if int_compar(a, b) > 0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}
/// Original `rssortints` (`robuststat.c:64`).
pub fn rssortints(x: &mut [i32], n: &i32) {
    rs_sort_ints(x, *n)
}

/// Original `floatCompar` (`robuststat.c:69`).
fn float_compar(val1: &f32, val2: &f32) -> i32 {
    if *val1 < *val2 {
        -1
    } else if *val1 > *val2 {
        1
    } else {
        0
    }
}
/// Original `rsSortFloats` (`robuststat.c:83`).
pub fn rs_sort_floats(x: &mut [f32], n: i32) {
    x[..n as usize].sort_by(|a, b| {
        if float_compar(a, b) < 0 {
            Ordering::Less
        } else if float_compar(a, b) > 0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}
/// Original `rssortfloats` (`robuststat.c:91`).
pub fn rssortfloats(x: &mut [f32], n: &i32) {
    rs_sort_floats(x, *n)
}

/// Original `indexedFloatCompar` (`robuststat.c:98`).
///
/// The source's `valArray` static (`robuststat.c:96`) exists only to hand the
/// value array to `qsort`'s comparison callback; it is written and read within
/// a single `rsSortIndexedFloats` call and never observed outside one, so it
/// is a captured argument here.  `indexOffset` genuinely persists between
/// calls (`rsSetSortIndexOffset`) and stays a static.
fn indexed_float_compar(val_array: &[f32], val1: &i32, val2: &i32) -> i32 {
    let index_offset = INDEX_OFFSET.with(|c| c.get());
    let i1 = *val1 - index_offset;
    let i2 = *val2 - index_offset;
    if val_array[i1 as usize] < val_array[i2 as usize] {
        -1
    } else if val_array[i1 as usize] > val_array[i2 as usize] {
        1
    } else {
        0
    }
}
/// Original `rsSortIndexedFloats` (`robuststat.c:115`).
pub fn rs_sort_indexed_floats(x: &[f32], index: &mut [i32], n: i32) {
    let val_array = x;
    index[..n as usize].sort_by(|a, b| {
        if indexed_float_compar(val_array, a, b) < 0 {
            Ordering::Less
        } else if indexed_float_compar(val_array, a, b) > 0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    INDEX_OFFSET.with(|c| c.set(0));
}
/// Original `rssortindexedfloats` (`robuststat.c:126`).
pub fn rssortindexedfloats(x: &[f32], index: &mut [i32], n: &i32) {
    INDEX_OFFSET.with(|c| c.set(1));
    rs_sort_indexed_floats(x, index, *n);
}
/// Original `rsSetSortIndexOffset` (`robuststat.c:137`).
pub fn rs_set_sort_index_offset(offset: i32) {
    INDEX_OFFSET.with(|c| c.set(offset));
}

/// Original `rsMedianOfSorted` (`robuststat.c:146`).
pub fn rs_median_of_sorted(x: &[f32], n: i32, median: &mut f32) {
    *median = if n % 2 != 0 {
        x[(n / 2) as usize]
    } else {
        0.5f32 * (x[(n / 2 - 1) as usize] + x[(n / 2) as usize])
    };
}
/// Original `rsmedianofsorted` (`robuststat.c:157`).
pub fn rsmedianofsorted(x: &[f32], n: &i32, median: &mut f32) {
    rs_median_of_sorted(x, *n, median);
}
/// Original `rsMedian` (`robuststat.c:167`).
pub fn rs_median(x: &[f32], n: i32, xsort: &mut [f32], median: &mut f32) {
    xsort[..n as usize].copy_from_slice(&x[..n as usize]);
    rs_sort_floats(xsort, n);
    rs_median_of_sorted(xsort, n, median);
}
/// Original `rsmedian` (`robuststat.c:177`).
pub fn rsmedian(x: &[f32], n: &i32, xsort: &mut [f32], median: &mut f32) {
    rs_median(x, *n, xsort, median);
}

/// Original `rsFastMedianInPlace` (`robuststat.c:187`).
pub fn rs_fast_median_in_place(x: &mut [f32], n: i32, median: &mut f32) {
    *median = percentile_float((n + 1) / 2, x, n);
    if n % 2 == 0 {
        // C: `(float)((*median + percentileFloat(...)) / 2.)` -- the sum is a
        // single-precision add, the division by the double literal `2.` is not.
        *median = ((*median + percentile_float(n / 2 + 1, x, n)) as f64 / 2.) as f32;
    }
}
/// Original `rsfastmedianinplace` (`robuststat.c:195`).
pub fn rsfastmedianinplace(x: &mut [f32], n: &i32, median: &mut f32) {
    rs_fast_median_in_place(x, *n, median);
}
/// Original `rsFastMedian` (`robuststat.c:204`).
pub fn rs_fast_median(x: &[f32], n: i32, xjumble: &mut [f32], median: &mut f32) {
    xjumble[..n as usize].copy_from_slice(&x[..n as usize]);
    rs_fast_median_in_place(xjumble, n, median);
}
/// Original `rsfastmedian` (`robuststat.c:211`).
pub fn rsfastmedian(x: &[f32], n: &i32, xjumble: &mut [f32], median: &mut f32) {
    rs_fast_median(x, *n, xjumble, median);
}

/// Original `rsPercentileOfSorted` (`robuststat.c:220`).
pub fn rs_percentile_of_sorted(x: &[f32], n: i32, fraction: f32, pctile: &mut f32) {
    /* If each value is though to occupy a "bin" of space along a range of values,
    the center of each value represents a fractional position along the full range
    of (index + 0.5)/n.  At least, this works for median */
    // C: `float realInd = n * fraction - 0.5;` -- `n * fraction` is a single
    // precision product, the subtraction of the double literal `0.5` is not.
    let real_ind: f32 = ((n as f32 * fraction) as f64 - 0.5) as f32;
    let lower_ind = (real_ind as f64).floor() as i32;
    let f: f32 = real_ind - lower_ind as f32;
    *pctile = if lower_ind < 0 {
        x[0]
    } else if lower_ind >= n - 1 {
        x[(n - 1) as usize]
    } else {
        // C: `(1. - f) * x[lowerInd] + f * x[lowerInd + 1]` -- `1. - f` is a
        // double, so its product is a double; `f * x[lowerInd + 1]` is a
        // single-precision product that only then widens for the addition.
        ((1. - f as f64) * x[lower_ind as usize] as f64 + (f * x[(lower_ind + 1) as usize]) as f64)
            as f32
    };
}
/// Original `rspercentileofsorted` (`robuststat.c:238`).
pub fn rspercentileofsorted(x: &[f32], n: &i32, fraction: &f32, pctile: &mut f32) {
    rs_percentile_of_sorted(x, *n, *fraction, pctile);
}

/// Original `rsMADN` (`robuststat.c:251`).
pub fn rs_madn(x: &[f32], n: i32, median: f32, tmp: &mut [f32], madn: &mut f32) {
    for i in 0..n as usize {
        tmp[i] = ((x[i] - median) as f64).abs() as f32;
    }
    rs_sort_floats(tmp, n);
    rs_median_of_sorted(tmp, n, madn);
    // C: `(*MADN) /= 0.6745;` on a `float *` -- the divisor is a double
    // literal, so the division is done in double and rounded once on store.
    *madn = (*madn as f64 / 0.6745) as f32;
}
/// Original `rsmadn` (`robuststat.c:262`).
pub fn rsmadn(x: &[f32], n: &i32, median: &f32, tmp: &mut [f32], madn: &mut f32) {
    rs_madn(x, *n, *median, tmp, madn);
}
/// Original `rsFastMADN` (`robuststat.c:273`).
pub fn rs_fast_madn(x: &[f32], n: i32, median: f32, tmp: &mut [f32], madn: &mut f32) {
    for i in 0..n as usize {
        tmp[i] = ((x[i] - median) as f64).abs() as f32;
    }
    rs_fast_median_in_place(tmp, n, madn);
    *madn = (*madn as f64 / 0.6745) as f32;
}
/// Original `rsfastmadn` (`robuststat.c:283`).
pub fn rsfastmadn(x: &[f32], n: &i32, median: &f32, tmp: &mut [f32], madn: &mut f32) {
    rs_fast_madn(x, *n, *median, tmp, madn);
}

/// Original `rsMadMedianOutliers` (`robuststat.c:297`).
pub fn rs_mad_median_outliers(x: &[f32], n: i32, kcrit: f32, out: &mut [f32]) {
    let (mut median, mut madn) = (0., 0.);
    rs_fast_median(x, n, out, &mut median);
    rs_fast_madn(x, n, median, out, &mut madn);
    for i in 0..n as usize {
        // C: `fabs((double)(x[i] - median)) / madn > kcrit` -- the quotient and
        // the comparison are both done in double.
        out[i] = if madn != 0. && ((x[i] - median) as f64).abs() / madn as f64 > kcrit as f64 {
            if x[i] > median { 1. } else { -1. }
        } else {
            0.
        };
    }
}
/// Original `rsmadmedianoutliers` (`robuststat.c:314`).
pub fn rsmadmedianoutliers(x: &[f32], n: &i32, kcrit: &f32, out: &mut [f32]) {
    rs_mad_median_outliers(x, *n, *kcrit, out);
}

/// Original `rsTrimmedMeanOfSorted` (`robuststat.c:324`).
pub fn rs_trimmed_mean_of_sorted(x: &[f32], n: i32, gamma: f32, trmean: &mut f32) {
    let cut = (gamma * n as f32) as i32;
    let mut sum = 0.;
    *trmean = 0.;
    for i in cut..n - cut {
        sum += x[i as usize] as f64;
    }
    if sum != 0. {
        *trmean = (sum / (n - 2 * cut) as f64) as f32;
    }
}
/// Original `rstrimmedmeanofsorted` (`robuststat.c:339`).
pub fn rstrimmedmeanofsorted(x: &[f32], n: &i32, gamma: &f32, median: &mut f32) {
    rs_trimmed_mean_of_sorted(x, *n, *gamma, median);
}
/// Original `rsTrimmedMean` (`robuststat.c:350`).
pub fn rs_trimmed_mean(x: &[f32], n: i32, gamma: f32, xsort: &mut [f32], trmean: &mut f32) {
    xsort[..n as usize].copy_from_slice(&x[..n as usize]);
    rs_sort_floats(xsort, n);
    rs_trimmed_mean_of_sorted(xsort, n, gamma, trmean);
}
/// Original `rstrimmedmean` (`robuststat.c:360`).
pub fn rstrimmedmean(x: &[f32], n: &i32, gamma: &f32, xsort: &mut [f32], median: &mut f32) {
    rs_trimmed_mean(x, *n, *gamma, xsort, median);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn robust_statistics_reject_extreme_outliers() {
        let values = [-1000., 1., 2., 3., 4., 1000.];
        let mut work = [0.; 6];
        let (mut median, mut madn) = (0., 0.);
        rs_fast_median(&values, 6, &mut work, &mut median);
        rs_fast_madn(&values, 6, median, &mut work, &mut madn);
        let mut out = [0.; 6];
        rs_mad_median_outliers(&values, 6, 2.24, &mut out);
        assert_eq!(median, 2.5);
        assert!(madn > 2. && madn < 3.);
        assert_eq!(out, [-1., 0., 0., 0., 0., 1.]);
    }
    #[test]
    fn sorting_percentile_and_trimmed_mean_cover_edges() {
        let mut values = [5., 1., 4., 2., 3.];
        rs_sort_floats(&mut values, 5);
        assert_eq!(values, [1., 2., 3., 4., 5.]);
        let mut pct = 0.;
        rs_percentile_of_sorted(&values, 5, 0., &mut pct);
        assert_eq!(pct, 1.);
        rs_percentile_of_sorted(&values, 5, 1., &mut pct);
        assert_eq!(pct, 5.);
        let mut mean = 0.;
        rs_trimmed_mean_of_sorted(&values, 5, 0.2, &mut mean);
        assert_eq!(mean, 3.);
    }
}
