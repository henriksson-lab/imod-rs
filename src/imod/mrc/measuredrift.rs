//! Owned translation of `IMOD/mrc/measuredrift.cpp`.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::circlefit::fit_centered_ellipse;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad};
use crate::imod::libfft::{nice_fft_limit, todfft_c};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MrcHeader, mrc_head_read, mrc_read_float_slice,
};
use std::io::Write;
use std::path::Path;

/// Audit inventory for `IMOD/mrc/measuredrift.cpp`: `cleanupDrift` is the
/// source's manual deallocator and is intentionally represented by `Vec` drop.
pub const MEASUREDRIFT_SOURCE_FUNCTIONS: &[&str] = &[
    "main",
    "measureDrift",
    "cleanupDrift",
    "interpolateCurve",
    "areaFunc",
];

pub const MAX_RINGS: usize = 200;

/// `cleanupDrift` (`measuredrift.cpp:446`).  All work arrays are owned vectors
/// here; clearing the source's inclusive set of initialized wedge buffers
/// releases their contents immediately while retaining safely reusable outer
/// allocations.
pub fn cleanup_drift(
    work: &mut Vec<f32>,
    spectra: &mut [Vec<f32>],
    sub_spectra: &mut [Vec<f32>],
    frequency_counts: &mut [Vec<i32>],
    max_index: usize,
) {
    work.clear();
    let count = spectra
        .len()
        .min(sub_spectra.len())
        .min(frequency_counts.len());
    if count == 0 {
        return;
    }
    for index in 0..=max_index.min(count - 1) {
        spectra[index].clear();
        sub_spectra[index].clear();
        frequency_counts[index].clear();
    }
}
#[derive(Clone, Debug)]
pub struct DriftResult {
    pub slice: i32,
    pub fall_ratio: f32,
    pub fall_ratio_sd: f32,
    pub oscillation_ratio: f32,
    pub oscillation_sd: f32,
    pub axis_degrees: f32,
    pub spectra: Vec<(i32, f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredriftOptions {
    pub input: String,
    pub output: Option<String>,
    pub sections: Option<Vec<usize>>,
    pub wedges: usize,
    pub ring_width: f32,
    pub fit: (f32, f32),
    pub oscillation_end: Option<f32>,
}

/// Source `interpolateCurve`.
pub fn interpolate_curve(x: &[f32], y: &[f32], value: f32) -> Option<f32> {
    x.windows(2).zip(y.windows(2)).find_map(|(xx, yy)| {
        ((xx[0] <= value && xx[1] > value) || (xx[0] > value && xx[1] <= value))
            .then(|| yy[0] + (value - xx[0]) * (yy[1] - yy[0]) / (xx[1] - xx[0]))
    })
}
/// Source `areaFunc`, made explicit instead of C static global state.
pub fn area_func(
    scale: f32,
    add: f32,
    spectrum: &[f32],
    all_sum: &[f32],
    freqs: &[f32],
    min_freq: f32,
    max_freq: f32,
) -> f32 {
    let mut area = 0.;
    let step = (max_freq - min_freq) / 99.;
    for index in 0..100 {
        let frequency = min_freq + index as f32 * step;
        let all = interpolate_curve(freqs, all_sum, frequency).unwrap_or(0.);
        let spec = interpolate_curve(freqs, spectrum, frequency / scale).unwrap_or(0.) + add;
        area += (all - spec).abs();
    }
    area * step
}
fn mean_sd(values: &[f32]) -> (f32, f32) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt();
    (mean, sd)
}
fn fit_scale(spectrum: &[f32], all: &[f32], freq: &[f32], start: f32, end: f32) -> f32 {
    let mut best = (1., f32::INFINITY);
    for step in -100..=100 {
        let scale = 1. + step as f32 * 0.002;
        let error = area_func(scale, 0., spectrum, all, freq, start, end);
        if error < best.1 {
            best = (scale, error);
        }
    }
    best.0
}
/// Entire numerical body of source `measureDrift`, accepting float pixels
/// after `mrc_read_float_slice` has converted the C-supported real modes.
pub fn measure_drift(
    array: &[f32],
    nx: usize,
    ny: usize,
    wedges: usize,
    ring_width: f32,
    start_freq: f32,
    end_freq: f32,
    amp_end_freq: Option<f32>,
    slice: i32,
) -> Result<DriftResult, String> {
    let rings = (0.5 / ring_width) as usize;
    if !(5..=MAX_RINGS).contains(&rings)
        || wedges < 2
        || wedges >= 36
        || start_freq < ring_width
        || end_freq > 0.5
        || start_freq > end_freq - 0.005
    {
        return Err("parameter out of range in call to measureDrift".into());
    }
    let amp_end = amp_end_freq.unwrap_or(end_freq);
    if amp_end < end_freq {
        return Err("oscillation ending frequency cannot be less than fitting end".into());
    }
    let nx_pad = nice_frame(nx as i32, 2, nice_fft_limit()) as usize;
    let ny_pad = nice_frame(ny as i32, 2, nice_fft_limit()) as usize;
    let mut work = vec![0.; (nx_pad + 2) * ny_pad];
    slice_taper_in_pad(
        PadIn::Float(array),
        2,
        nx as i32,
        0,
        nx as i32 - 1,
        0,
        ny as i32 - 1,
        &mut work,
        (nx_pad + 2) as i32,
        nx_pad as i32,
        ny_pad as i32,
        (0.02 * nx_pad as f32).max(8.) as i32,
        (0.02 * ny_pad as f32).max(8.) as i32,
    );
    todfft_c(&mut work, nx_pad as i32, ny_pad as i32, 0);
    let mut ratios = Vec::new();
    let mut osc = Vec::new();
    let mut axis = 0.;
    let mut saved = Vec::new();
    let nxhalf = nx_pad / 2 + 1;
    let del_wedge = std::f32::consts::PI / wedges as f32;
    for center in 0..3 {
        let offset = center as f32 * del_wedge / 3.;
        let mut sums = vec![vec![0.; rings]; wedges];
        let mut counts = vec![vec![0usize; rings]; wedges];
        for y in 0..ny_pad {
            let mut fy = y as f32 / ny_pad as f32;
            if fy > 0.5 {
                fy -= 1.;
            }
            for x in 0..nxhalf {
                if x == 0 && y == 0 {
                    continue;
                }
                let fx = x as f32 / nx_pad as f32;
                let ring = ((fx * fx + fy * fy).sqrt() / ring_width) as usize;
                if ring < rings {
                    let mut angle = fy.atan2(fx) + std::f32::consts::FRAC_PI_2 - offset;
                    if angle < 0. {
                        angle += std::f32::consts::PI;
                    }
                    let wedge = (angle / del_wedge) as usize;
                    let ind = 2 * (y * nxhalf + x);
                    sums[wedge][ring] += work[ind].powi(2) + work[ind + 1].powi(2);
                    counts[wedge][ring] += 1;
                }
            }
        }
        let mut spectra = vec![vec![0.; rings]; wedges];
        for w in 0..wedges {
            for r in 0..rings {
                let next = (w + 1) % wedges;
                let n = counts[w][r] + counts[next][r];
                spectra[w][r] = ((sums[w][r] + sums[next][r]) / n.max(1) as f32 + 1.0e-30).ln();
            }
        }
        let all: Vec<f32> = (0..rings)
            .map(|r| spectra.iter().map(|s| s[r]).sum::<f32>() / wedges as f32)
            .collect();
        let freqs: Vec<f32> = (0..rings).map(|r| (r as f32 + 0.5) * ring_width).collect();
        let start = (start_freq / ring_width).round() as usize;
        let end = ((amp_end / ring_width).round() as usize).min(rings - 1);
        let mut xp = Vec::new();
        let mut yp = Vec::new();
        let mut diffs = Vec::new();
        for w in 0..wedges {
            let scale = fit_scale(&spectra[w], &all, &freqs, start_freq, end_freq);
            let theta = (w as f32 + 0.5) * del_wedge - std::f32::consts::FRAC_PI_2;
            xp.push(scale * theta.cos());
            yp.push(scale * theta.sin());
            let diff = (start..=end)
                .map(|r| (spectra[w][r] - all[r]).abs())
                .sum::<f32>();
            diffs.push(diff);
            if center == 0 {
                for r in 0..rings {
                    saved.push((w as i32 + 1000 * slice, freqs[r], spectra[w][r]));
                }
            }
        }
        let mut xr = 0.;
        let mut yr = 0.;
        let mut theta = 0.;
        let mut rms = 0.;
        let mut ellipse_work = vec![0.; 3 * wedges];
        fit_centered_ellipse(
            &xp,
            &yp,
            wedges as i32,
            &mut xr,
            &mut yr,
            &mut theta,
            &mut rms,
            &mut ellipse_work,
        );
        ratios.push((xr / yr).abs().max((yr / xr).abs()));
        let mut ox = Vec::new();
        let mut oy = Vec::new();
        let avg = diffs.iter().sum::<f32>() / wedges as f32;
        for w in 0..wedges {
            let angle = (w as f32 + 0.5) * del_wedge - std::f32::consts::FRAC_PI_2;
            ox.push(diffs[w] / avg * angle.cos());
            oy.push(diffs[w] / avg * angle.sin());
        }
        fit_centered_ellipse(
            &ox,
            &oy,
            wedges as i32,
            &mut xr,
            &mut yr,
            &mut theta,
            &mut rms,
            &mut ellipse_work,
        );
        osc.push((xr / yr).abs().max((yr / xr).abs()));
        axis = theta + offset;
    }
    let (fall, fall_sd) = mean_sd(&ratios);
    let (oscillation, oscillation_sd) = mean_sd(&osc);
    Ok(DriftResult {
        slice,
        fall_ratio: fall,
        fall_ratio_sd: fall_sd,
        oscillation_ratio: oscillation,
        oscillation_sd,
        axis_degrees: axis.to_degrees(),
        spectra: saved,
    })
}
/// Source `main` MRC command body, with parsed arguments supplied as values.
pub fn measure_drift_file(
    input: impl AsRef<Path>,
    sections: Option<&[usize]>,
    wedges: usize,
    ring_width: f32,
    fit: (f32, f32),
    amp_end: Option<f32>,
) -> Result<Vec<DriftResult>, String> {
    let mut file = ImodFile::open(input, "rb").ok_or("opening input image")?;
    let mut head = MrcHeader::default();
    if mrc_head_read(&mut file, &mut head) != 0 {
        return Err("reading header".into());
    }
    if !matches!(head.mode, MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_FLOAT) {
        return Err(format!(
            "file mode is {}; only byte, short, float allowed",
            head.mode
        ));
    }
    let list: Vec<_> = sections.map_or_else(|| (0..head.nz as usize).collect(), |s| s.to_vec());
    let mut result = Vec::new();
    for z in list {
        if z >= head.nz as usize {
            return Err("section out of range".into());
        }
        let mut pixels = vec![0.; head.nx as usize * head.ny as usize];
        if mrc_read_float_slice(&mut pixels, &mut head, z as i32) != 0 {
            return Err(format!("reading slice {z}"));
        }
        result.push(measure_drift(
            &pixels,
            head.nx as usize,
            head.ny as usize,
            wedges,
            ring_width,
            fit.0,
            fit.1,
            amp_end,
            z as i32,
        )?);
    }
    Ok(result)
}

/// C `main` command parser.  It accepts the source long and short option
/// names; spectra can be consumed from the returned per-section results.
pub fn measuredrift(arguments: &[String]) -> Result<Vec<DriftResult>, String> {
    let options = measuredrift_options(arguments)?;
    let result = measure_drift_file(
        &options.input,
        options.sections.as_deref(),
        options.wedges,
        options.ring_width,
        options.fit,
        options.oscillation_end,
    )?;
    if let Some(output) = options.output {
        write_drift_spectra(&output, &result)?;
    }
    Ok(result)
}

/// Owned equivalent of source `main`'s Pip option collection.
pub fn measuredrift_options(arguments: &[String]) -> Result<MeasuredriftOptions, String> {
    let mut input = None;
    let mut output = None;
    let mut sections = None;
    let mut wedges = 8usize;
    let mut ring = 0.01f32;
    let mut fit = (0.025f32, 0.25f32);
    let mut oscillation = None;
    let mut index = 1;
    while index < arguments.len() {
        let option = &arguments[index];
        let take = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            arguments
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {option}"))
        };
        match option.as_str() {
            "-input" | "--input" | "-InputFile" | "--InputFile" => input = Some(take(&mut index)?),
            "-output" | "--output" | "-OutputFile" | "--OutputFile" => {
                output = Some(take(&mut index)?)
            }
            "-sections" | "--sections" | "-SectionsToDo" | "--SectionsToDo" => {
                let text = take(&mut index)?;
                let mut listed = Vec::new();
                for part in text.split(',') {
                    let part = part.trim();
                    if let Some((start, end)) = part.split_once('-') {
                        let start: usize = start
                            .trim()
                            .parse()
                            .map_err(|_| "bad entry in list of sections to do")?;
                        let end: usize = end
                            .trim()
                            .parse()
                            .map_err(|_| "bad entry in list of sections to do")?;
                        if end < start {
                            return Err("bad entry in list of sections to do".into());
                        }
                        listed.extend(start..=end);
                    } else {
                        listed.push(
                            part.parse()
                                .map_err(|_| "bad entry in list of sections to do")?,
                        );
                    }
                }
                sections = Some(listed);
            }
            "-wedges" | "--wedges" | "-NumberOfWedges" | "--NumberOfWedges" => {
                wedges = take(&mut index)?
                    .parse()
                    .map_err(|_| "invalid number of wedges")?
            }
            "-ring" | "--ring" | "-FrequencyRingWidth" | "--FrequencyRingWidth" => {
                ring = take(&mut index)?
                    .parse()
                    .map_err(|_| "invalid ring width")?
            }
            "-fit" | "--fit" | "-FrequencyRangeToFit" | "--FrequencyRangeToFit" => {
                let values: Vec<f32> = take(&mut index)?
                    .split(',')
                    .map(str::trim)
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| "invalid fit range")?;
                if values.len() != 2 {
                    return Err("fit range requires two values".into());
                }
                fit = (values[0], values[1]);
            }
            "-oscil" | "--oscil" | "-OscillationEndFreq" | "--OscillationEndFreq" => {
                oscillation = Some(
                    take(&mut index)?
                        .parse()
                        .map_err(|_| "invalid oscillation end frequency")?,
                )
            }
            "-help" | "--help" => return Err("usage: measuredrift -input image [options]".into()),
            value if value.starts_with('-') => return Err(format!("unknown option {value}")),
            value => return Err(format!("unexpected non-option argument {value}")),
        }
        index += 1;
    }
    Ok(MeasuredriftOptions {
        input: input.ok_or("No input image file specified")?,
        output,
        sections,
        wedges,
        ring_width: ring,
        fit,
        oscillation_end: oscillation,
    })
}

/// Source `main`'s optional text spectra output.  Each saved wedge spectrum
/// uses the native `typeNum frequency logPower` line representation.
pub fn write_drift_spectra(path: impl AsRef<Path>, results: &[DriftResult]) -> Result<(), String> {
    let mut file = std::fs::File::create(path)
        .map_err(|error| format!("failed to open file for spectra: {error}"))?;
    for result in results {
        for &(type_number, frequency, power) in &result.spectra {
            writeln!(file, "{type_number:4} {frequency:.3} {power:8.5}")
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_matches_source() {
        assert_eq!(interpolate_curve(&[0., 2.], &[1., 5.], 1.), Some(3.));
    }
    #[test]
    fn rejects_source_parameter_ranges() {
        assert!(measure_drift(&vec![0.; 16], 4, 4, 1, 0.01, 0.03, 0.2, None, 0).is_err());
    }
    #[test]
    fn source_cli_sections_and_output_are_retained() {
        let args = vec![
            "measuredrift".into(),
            "-input".into(),
            "in.mrc".into(),
            "-output".into(),
            "out.txt".into(),
            "-sections".into(),
            "1-2,4".into(),
        ];
        let options = measuredrift_options(&args).unwrap();
        assert_eq!(options.output.as_deref(), Some("out.txt"));
        assert_eq!(options.sections, Some(vec![1, 2, 4]));
        assert_eq!(MEASUREDRIFT_SOURCE_FUNCTIONS.len(), 5);
    }
}
