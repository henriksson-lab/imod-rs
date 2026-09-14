//! Translation of `IMOD/libcfshr/histogram.c` and its `cfsemshare.h` APIs.

use std::io::Write;

use super::b3dutil::num_omp_threads;
use super::b3dutil::{CArg, ImodFile, c_format};

/// Original `kernelHistogram` (`histogram.c:31`).
pub unsafe fn kernel_histogram(
    values: *mut f32,
    num_vals: i32,
    bins: *mut f32,
    num_bins: i32,
    first_val: f32,
    last_val: f32,
    h: f32,
    verbose: i32,
) {
    let mut temp_bins = None;
    if h != 0. {
        let optimal = (num_vals / 10_000).clamp(1, 8);
        let _num_threads = num_omp_threads(optimal).min(16);
        if optimal > 1 {
            let mut temporary = Vec::new();
            if temporary.try_reserve_exact(num_bins as usize).is_ok() {
                temporary.resize(num_bins as usize, 0.);
                temp_bins = Some(temporary);
            }
        }
    }
    let dxbin = (last_val - first_val) / num_bins as f32;
    for index in 0..num_bins as usize {
        unsafe {
            *bins.add(index) = 0.;
        }
    }
    for value_index in 0..num_vals as usize {
        let val = unsafe { *values.add(value_index) };
        if verbose == 1 {
            let _ = ImodFile::Stdout
                .write_all(c_format("Value: %.4f\n", &[CArg::Dbl(val as f64)]).as_bytes());
        }
        if h != 0. {
            let mut ist = ((val - h - first_val) as f64 / dxbin as f64).ceil() as i32;
            let mut ind = ((val + h - first_val) as f64 / dxbin as f64).floor() as i32;
            ist = ist.max(0);
            ind = ind.min(num_bins - 1);
            for bin in ist..=ind {
                let delta = (val - first_val - bin as f32 * dxbin) / h;
                if let Some(temporary) = temp_bins.as_mut() {
                    temporary[bin as usize] += (1. - delta * delta).powi(3) as f64;
                } else {
                    unsafe {
                        *bins.add(bin as usize) += (1. - delta * delta).powi(3);
                    }
                }
            }
        } else {
            let ist = ((val - first_val) as f64 / dxbin as f64).floor() as i32;
            if ist >= 0 && ist < num_bins {
                unsafe {
                    *bins.add(ist as usize) += 1.;
                }
            } else if ist == num_bins && val - last_val < 0.001 * dxbin {
                unsafe {
                    *bins.add((num_bins - 1) as usize) += 1.;
                }
            }
        }
    }
    if let Some(temporary) = temp_bins {
        for index in 0..num_bins as usize {
            unsafe {
                *bins.add(index) += temporary[index] as f32;
            }
        }
    }
    if h != 0. {
        let scale = 1. / (h * (1.2 - 2. / 7.));
        for index in 0..num_bins as usize {
            unsafe {
                *bins.add(index) *= scale;
            }
        }
    }
    if verbose == 2 {
        for index in 0..num_bins as usize {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "bin: %.4f %f\n",
                    &[
                        CArg::Dbl((first_val + index as f32 * dxbin) as f64),
                        CArg::Dbl(unsafe { *bins.add(index) } as f64),
                    ],
                )
                .as_bytes(),
            );
        }
    }
}

/// Original `kernelhistogram` (`histogram.c:134`).
pub unsafe fn kernelhistogram(
    values: *mut f32,
    num_vals: *mut i32,
    bins: *mut f32,
    num_bins: *mut i32,
    first_val: *mut f32,
    last_val: *mut f32,
    h: *mut f32,
    verbose: *mut i32,
) {
    unsafe {
        kernel_histogram(
            values, *num_vals, bins, *num_bins, *first_val, *last_val, *h, *verbose,
        );
    }
}

/// Original `scanHistogram` (`histogram.c:147`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn scan_histogram(
    bins: *mut f32,
    num_bins: i32,
    first_val: f32,
    last_val: f32,
    scan_bot: f32,
    scan_top: f32,
    find_peaks: i32,
    dip: *mut f32,
    peak_below: *mut f32,
    peak_above: *mut f32,
) -> i32 {
    let dxbin = (last_val - first_val) / num_bins as f32;
    let mut indstr = ((scan_top - first_val) as f64 / dxbin as f64).floor() as i32;
    indstr = indstr.min(num_bins - 1);
    let mut indend = ((scan_bot - first_val) as f64 / dxbin as f64).ceil() as i32;
    indend = indend.max(0);
    if find_peaks != 0 {
        let mut first = -1.;
        let mut second = -1.;
        let mut rising = true;
        let mut last_rank = 0;
        let mut last_ind = num_bins;
        let mut last_peak = 0.;
        let (mut ind_first, mut ind_second) = (0, 0);
        for ind in (indend..indstr).rev() {
            unsafe {
                if rising {
                    if *bins.add(ind as usize) < *bins.add((ind + 1) as usize) || ind == indend {
                        if last_rank != 0
                            && dxbin * ((last_ind - ind - 1) as f32)
                                < 0.005 * (last_val - first_val)
                            && (last_peak - *bins.add((ind + 1) as usize)).abs() < 0.001 * last_peak
                        {
                            if last_peak < *bins.add((ind + 1) as usize) {
                                last_peak = *bins.add((ind + 1) as usize);
                                last_ind = ind + 1;
                                if last_rank == 1 {
                                    first = last_peak;
                                    ind_first = ind + 1;
                                } else {
                                    second = last_peak;
                                    ind_second = ind + 1;
                                }
                            }
                        } else if *bins.add((ind + 1) as usize) > first {
                            second = first;
                            ind_second = ind_first;
                            first = *bins.add((ind + 1) as usize);
                            ind_first = ind + 1;
                            last_peak = first;
                            last_ind = ind_first;
                            last_rank = 1;
                        } else if *bins.add((ind + 1) as usize) > second {
                            ind_second = ind + 1;
                            second = *bins.add((ind + 1) as usize);
                            last_peak = second;
                            last_ind = ind_second;
                            last_rank = 2;
                        }
                        rising = false;
                    }
                } else if *bins.add(ind as usize) > *bins.add((ind + 1) as usize) {
                    rising = true;
                }
            }
        }
        if second < 0. {
            return 1;
        }
        indstr = ind_first.max(ind_second);
        indend = ind_first.min(ind_second);
        unsafe {
            *peak_above = first_val + indstr as f32 * dxbin;
            *peak_below = first_val + indend as f32 * dxbin;
        }
    }
    let mut indmin = indstr;
    let mut valmin = unsafe { *bins.add(indstr as usize) };
    for ind in (indend..=indstr).rev() {
        let value = unsafe { *bins.add(ind as usize) };
        if value < valmin {
            valmin = value;
            indmin = ind;
        }
    }
    unsafe {
        *dip = first_val + indmin as f32 * dxbin;
    }
    0
}

/// Original `scanhistogram` (`histogram.c:246`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn scanhistogram(
    bins: *mut f32,
    num_bins: *mut i32,
    first_val: *mut f32,
    last_val: *mut f32,
    scan_bot: *mut f32,
    scan_top: *mut f32,
    find_peaks: *mut i32,
    dip: *mut f32,
    peak_below: *mut f32,
    peak_above: *mut f32,
) -> i32 {
    unsafe {
        scan_histogram(
            bins,
            *num_bins,
            *first_val,
            *last_val,
            *scan_bot,
            *scan_top,
            *find_peaks,
            dip,
            peak_below,
            peak_above,
        )
    }
}

/// Original `findHistogramDip` (`histogram.c:274`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn find_histogram_dip(
    values: *mut f32,
    num_vals: i32,
    min_guess: i32,
    bins: *mut f32,
    num_bins: i32,
    first_val: f32,
    last_val: f32,
    hist_dip: *mut f32,
    peak_below: *mut f32,
    peak_above: *mut f32,
    verbose: i32,
) -> i32 {
    let range = last_val - first_val;
    let mut coarse_h = 0.2 * range;
    let fine_h = 0.05 * range;
    let mut upper_lim = last_val;
    unsafe {
        kernel_histogram(
            values, num_vals, bins, num_bins, first_val, last_val, 0., verbose,
        );
    }
    if min_guess != 0 {
        let num_crit = ((min_guess as f32 * 0.5).round() as i32).max(1);
        let mut ncum = 0;
        let mut index = num_bins - 1;
        while index > 10 {
            ncum += unsafe { (*bins.add(index as usize)).round() as i32 };
            if ncum >= num_crit {
                break;
            }
            index -= 1;
        }
        upper_lim = first_val + range * (index as f32 / (num_bins - 1) as f32).min(1.);
    }
    let mut cut = 0;
    while cut < 4 {
        unsafe {
            kernel_histogram(
                values, num_vals, bins, num_bins, first_val, last_val, coarse_h, verbose,
            );
        }
        if unsafe {
            scan_histogram(
                bins, num_bins, first_val, last_val, first_val, upper_lim, 1, hist_dip, peak_below,
                peak_above,
            )
        } == 0
            && unsafe { *peak_above < first_val + 0.999 * range }
        {
            break;
        }
        coarse_h *= 0.707;
        cut += 1;
    }
    if cut == 4 {
        return 1;
    }
    if verbose >= 0 {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Histogram smoothed with H = %.3f has dip at %g, peaks at %g and %g\n",
                &[
                    CArg::Dbl(coarse_h as f64),
                    CArg::Dbl(unsafe { *hist_dip } as f64),
                    CArg::Dbl(unsafe { *peak_below } as f64),
                    CArg::Dbl(unsafe { *peak_above } as f64),
                ],
            )
            .as_bytes(),
        );
    }
    unsafe {
        kernel_histogram(
            values, num_vals, bins, num_bins, first_val, last_val, fine_h, verbose,
        );
        scan_histogram(
            bins,
            num_bins,
            first_val,
            last_val,
            0.5 * (*hist_dip + *peak_below),
            0.5 * (*hist_dip + *peak_above),
            0,
            hist_dip,
            peak_below,
            peak_above,
        );
    }
    if verbose >= 0 {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Histogram smoothed with H = %g has lowest dip at %g\n",
                &[
                    CArg::Dbl(fine_h as f64),
                    CArg::Dbl(unsafe { *hist_dip } as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.flush();
    }
    0
}

/// Original `findhistogramdip` (`histogram.c:338`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn findhistogramdip(
    values: *mut f32,
    num_vals: *mut i32,
    min_guess: *mut i32,
    bins: *mut f32,
    num_bins: *mut i32,
    first_val: *mut f32,
    last_val: *mut f32,
    hist_dip: *mut f32,
    peak_below: *mut f32,
    peak_above: *mut f32,
    verbose: *mut i32,
) -> i32 {
    unsafe {
        find_histogram_dip(
            values, *num_vals, *min_guess, bins, *num_bins, *first_val, *last_val, hist_dip,
            peak_below, peak_above, *verbose,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_histogram_keeps_last_edge_in_last_bin() {
        let mut values = [0., 0.99, 1., 1.0001, -0.1];
        let mut bins = [9.; 4];
        unsafe {
            kernel_histogram(
                values.as_mut_ptr(),
                values.len() as i32,
                bins.as_mut_ptr(),
                4,
                0.,
                1.,
                0.,
                0,
            );
        }
        assert_eq!(bins, [1., 0., 0., 3.]);
    }
    #[test]
    fn scan_finds_two_peaks_and_the_dip_between_them() {
        let mut bins = [0., 2., 5., 2., 1., 2., 4., 2., 0.];
        let (mut dip, mut below, mut above) = (0., 0., 0.);
        assert_eq!(
            unsafe {
                scan_histogram(
                    bins.as_mut_ptr(),
                    bins.len() as i32,
                    0.,
                    9.,
                    0.,
                    9.,
                    1,
                    &mut dip,
                    &mut below,
                    &mut above,
                )
            },
            0
        );
        assert_eq!((below, above, dip), (2., 6., 4.));
    }
}
