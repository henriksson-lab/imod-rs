//! Translation of `IMOD/libcfshr/histogram.c` and its `cfsemshare.h` APIs.

use std::io::Write;

use super::b3dutil::ImodFile;
use super::b3dutil::num_omp_threads;

/// Locations of a histogram dip and the peaks bracketing it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistogramDip {
    pub dip: f32,
    pub peak_below: f32,
    pub peak_above: f32,
}

/// Original `kernelHistogram` (`histogram.c:31`).
pub fn kernel_histogram(
    values: &[f32],
    bins: &mut [f32],
    first_val: f32,
    last_val: f32,
    h: f32,
    verbose: i32,
) {
    let mut temp_bins = None;
    if h != 0. {
        let optimal = ((values.len() as i32) / 10_000).clamp(1, 8);
        let _num_threads = num_omp_threads(optimal).min(16);
        if optimal > 1 {
            let mut temporary = Vec::new();
            if temporary.try_reserve_exact(bins.len()).is_ok() {
                temporary.resize(bins.len(), 0.);
                temp_bins = Some(temporary);
            }
        }
    }
    let num_bins = bins.len() as i32;
    let dxbin = (last_val - first_val) / num_bins as f32;
    bins.fill(0.);
    for &val in values {
        if verbose == 1 {
            let _ = writeln!(ImodFile::Stdout, "Value: {val:.4}");
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
                    bins[bin as usize] += (1. - delta * delta).powi(3);
                }
            }
        } else {
            let ist = ((val - first_val) as f64 / dxbin as f64).floor() as i32;
            if ist >= 0 && ist < num_bins {
                bins[ist as usize] += 1.;
            } else if ist == num_bins && val - last_val < 0.001 * dxbin {
                bins[(num_bins - 1) as usize] += 1.;
            }
        }
    }
    if let Some(temporary) = temp_bins {
        for (bin, temporary) in bins.iter_mut().zip(temporary) {
            *bin += temporary as f32;
        }
    }
    if h != 0. {
        let scale = 1. / (h * (1.2 - 2. / 7.));
        for bin in bins.iter_mut() {
            *bin *= scale;
        }
    }
    if verbose == 2 {
        for (index, bin) in bins.iter().enumerate() {
            let _ = writeln!(
                ImodFile::Stdout,
                "bin: {:.4} {bin}",
                first_val + index as f32 * dxbin
            );
        }
    }
}

/// Original `kernelhistogram` (`histogram.c:134`).
pub fn kernelhistogram(
    values: &[f32],
    bins: &mut [f32],
    first_val: f32,
    last_val: f32,
    h: f32,
    verbose: i32,
) {
    kernel_histogram(values, bins, first_val, last_val, h, verbose);
}

/// Original `scanHistogram` (`histogram.c:147`).
#[allow(clippy::too_many_arguments)]
pub fn scan_histogram(
    bins: &[f32],
    first_val: f32,
    last_val: f32,
    scan_bot: f32,
    scan_top: f32,
    find_peaks: bool,
) -> Option<HistogramDip> {
    let num_bins = bins.len() as i32;
    let dxbin = (last_val - first_val) / num_bins as f32;
    let mut indstr = ((scan_top - first_val) as f64 / dxbin as f64).floor() as i32;
    indstr = indstr.min(num_bins - 1);
    let mut indend = ((scan_bot - first_val) as f64 / dxbin as f64).ceil() as i32;
    indend = indend.max(0);
    let peaks = if find_peaks {
        let mut first = -1.;
        let mut second = -1.;
        let mut rising = true;
        let mut last_rank = 0;
        let mut last_ind = num_bins;
        let mut last_peak = 0.;
        let (mut ind_first, mut ind_second) = (0, 0);
        for ind in (indend..indstr).rev() {
            if rising {
                if bins[ind as usize] < bins[(ind + 1) as usize] || ind == indend {
                    if last_rank != 0
                        && dxbin * ((last_ind - ind - 1) as f32) < 0.005 * (last_val - first_val)
                        && (last_peak - bins[(ind + 1) as usize]).abs() < 0.001 * last_peak
                    {
                        if last_peak < bins[(ind + 1) as usize] {
                            last_peak = bins[(ind + 1) as usize];
                            last_ind = ind + 1;
                            if last_rank == 1 {
                                first = last_peak;
                                ind_first = ind + 1;
                            } else {
                                second = last_peak;
                                ind_second = ind + 1;
                            }
                        }
                    } else if bins[(ind + 1) as usize] > first {
                        second = first;
                        ind_second = ind_first;
                        first = bins[(ind + 1) as usize];
                        ind_first = ind + 1;
                        last_peak = first;
                        last_ind = ind_first;
                        last_rank = 1;
                    } else if bins[(ind + 1) as usize] > second {
                        ind_second = ind + 1;
                        second = bins[(ind + 1) as usize];
                        last_peak = second;
                        last_ind = ind_second;
                        last_rank = 2;
                    }
                    rising = false;
                }
            } else if bins[ind as usize] > bins[(ind + 1) as usize] {
                rising = true;
            }
        }
        if second < 0. {
            return None;
        }
        indstr = ind_first.max(ind_second);
        indend = ind_first.min(ind_second);
        Some((
            first_val + indend as f32 * dxbin,
            first_val + indstr as f32 * dxbin,
        ))
    } else {
        None
    };
    let mut indmin = indstr;
    let mut valmin = bins[indstr as usize];
    for ind in (indend..=indstr).rev() {
        let value = bins[ind as usize];
        if value < valmin {
            valmin = value;
            indmin = ind;
        }
    }
    Some(HistogramDip {
        dip: first_val + indmin as f32 * dxbin,
        peak_below: peaks.map_or(0., |(below, _)| below),
        peak_above: peaks.map_or(0., |(_, above)| above),
    })
}

/// Original `scanhistogram` (`histogram.c:246`).
#[allow(clippy::too_many_arguments)]
pub fn scanhistogram(
    bins: &[f32],
    first_val: f32,
    last_val: f32,
    scan_bot: f32,
    scan_top: f32,
    find_peaks: bool,
    dip: &mut f32,
    peak_below: Option<&mut f32>,
    peak_above: Option<&mut f32>,
) -> i32 {
    let result = scan_histogram(bins, first_val, last_val, scan_bot, scan_top, find_peaks);
    let Some(result) = result else {
        return 1;
    };
    *dip = result.dip;
    if find_peaks {
        if let Some(peak_below) = peak_below {
            *peak_below = result.peak_below;
        }
        if let Some(peak_above) = peak_above {
            *peak_above = result.peak_above;
        }
    }
    0
}

/// Original `findHistogramDip` (`histogram.c:274`).
pub fn find_histogram_dip(
    values: &[f32],
    min_guess: i32,
    bins: &mut [f32],
    first_val: f32,
    last_val: f32,
    verbose: i32,
) -> Option<HistogramDip> {
    let range = last_val - first_val;
    let mut coarse_h = 0.2 * range;
    let fine_h = 0.05 * range;
    let mut upper_lim = last_val;
    kernel_histogram(values, bins, first_val, last_val, 0., verbose);
    if min_guess != 0 {
        let num_crit = ((min_guess as f32 * 0.5).round() as i32).max(1);
        let mut ncum = 0;
        let mut index = bins.len() as i32 - 1;
        while index > 10 {
            ncum += bins[index as usize].round() as i32;
            if ncum >= num_crit {
                break;
            }
            index -= 1;
        }
        upper_lim = first_val + range * (index as f32 / (bins.len() - 1) as f32).min(1.);
    }
    let mut cut = 0;
    let coarse = loop {
        kernel_histogram(values, bins, first_val, last_val, coarse_h, verbose);
        if let Some(result) = scan_histogram(bins, first_val, last_val, first_val, upper_lim, true)
            && result.peak_above < first_val + 0.999 * range
        {
            break result;
        }
        coarse_h *= 0.707;
        cut += 1;
        if cut == 4 {
            return None;
        }
    };
    if verbose >= 0 {
        let _ = writeln!(
            ImodFile::Stdout,
            "Histogram smoothed with H = {coarse_h:.3} has dip at {}, peaks at {} and {}",
            coarse.dip,
            coarse.peak_below,
            coarse.peak_above,
        );
    }
    kernel_histogram(values, bins, first_val, last_val, fine_h, verbose);
    let fine = scan_histogram(
        bins,
        first_val,
        last_val,
        0.5 * (coarse.dip + coarse.peak_below),
        0.5 * (coarse.dip + coarse.peak_above),
        false,
    )?;
    let result = HistogramDip {
        dip: fine.dip,
        ..coarse
    };
    if verbose >= 0 {
        let _ = writeln!(
            ImodFile::Stdout,
            "Histogram smoothed with H = {fine_h} has lowest dip at {}",
            result.dip
        );
        let _ = ImodFile::Stdout.flush();
    }
    Some(result)
}

/// Original `findhistogramdip` (`histogram.c:338`).
#[allow(clippy::too_many_arguments)]
pub fn findhistogramdip(
    values: &[f32],
    min_guess: i32,
    bins: &mut [f32],
    first_val: f32,
    last_val: f32,
    hist_dip: &mut f32,
    peak_below: &mut f32,
    peak_above: &mut f32,
    verbose: i32,
) -> i32 {
    let Some(result) = find_histogram_dip(values, min_guess, bins, first_val, last_val, verbose)
    else {
        return 1;
    };
    *hist_dip = result.dip;
    *peak_below = result.peak_below;
    *peak_above = result.peak_above;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_histogram_keeps_last_edge_in_last_bin() {
        let values = [0., 0.99, 1., 1.0001, -0.1];
        let mut bins = [9.; 4];
        kernel_histogram(&values, &mut bins, 0., 1., 0., 0);
        assert_eq!(bins, [1., 0., 0., 3.]);
    }
    #[test]
    fn scan_finds_two_peaks_and_the_dip_between_them() {
        let bins = [0., 2., 5., 2., 1., 2., 4., 2., 0.];
        assert_eq!(
            scan_histogram(&bins, 0., 9., 0., 9., true),
            Some(HistogramDip {
                dip: 4.,
                peak_below: 2.,
                peak_above: 6.
            }),
        );
    }
}
