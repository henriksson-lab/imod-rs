//! Combining strip spectra into one rotationally averaged spectrum at the
//! final resolution: each strip's hyper-resolution spectrum is divided by
//! the noise spectrum for its mean intensity, its frequency axis is scaled
//! so that its CTF zeros land where the zeros of the reference defocus are,
//! and it is added into the average with per-bin component counts.
//!
//! This follows `MyApp::doComputePS`/`scaleAndAddStrip`.  With known
//! astigmatism the angular sectors are grouped into bands of similar defocus
//! and each band is scaled by its mean defocus, as native does (native also
//! requires the sectors of a band to be contiguous, which does not matter for
//! a rotational average and is not done here).

use super::ctf::{CtfModel, RADIANS_PER_DEGREE};
use super::spectra::Binning;

/// Noise spectra measured from noise images, sorted by image mean.
#[derive(Clone, Debug, Default)]
pub struct NoiseModel {
    /// Resolution of the stored noise spectra.
    pub dim: usize,
    pub means: Vec<f64>,
    pub spectra: Vec<Vec<f64>>,
}

impl NoiseModel {
    /// The noise spectrum interpolated (or extrapolated) linearly for an
    /// image mean.
    pub fn for_mean(&self, mean: f64) -> Vec<f64> {
        let n = self.means.len();
        let mut high = n - 1;
        for i in 1..n {
            if self.means[i] > mean {
                high = i;
                break;
            }
        }
        let (lm, hm) = (self.means[high - 1], self.means[high]);
        let (lps, hps) = (&self.spectra[high - 1], &self.spectra[high]);
        let mut out: Vec<f64> = (0..self.dim)
            .map(|i| lps[i] + (hps[i] - lps[i]) * (mean - lm) / (hm - lm))
            .collect();
        if out.iter().any(|&v| v <= 0.0) {
            out = if mean < lm { lps.clone() } else { hps.clone() };
        }
        out
    }
}

/// How frequencies are scaled between defocus values: by the ratio of zero
/// `low` (when `high` is 0) or by the linear map taking zeros `low` and
/// `high` of one defocus onto those of the other.
#[derive(Clone, Copy, Debug)]
pub struct ZeroScaling {
    pub low: i32,
    pub high: i32,
}

/// A wedge of sectors: center angle and full angular range, degrees.
#[derive(Clone, Copy, Debug)]
pub struct Wedge {
    pub center: f64,
    pub range: f64,
}

impl Wedge {
    /// Whether the sector centered at `angle` (in -90..90) is in the wedge.
    pub fn contains(&self, angle: f64) -> bool {
        let plus_min = self.center - self.range / 2.0;
        let plus_max = self.center + self.range / 2.0;
        let shift = if self.center > 0.0 { -180.0 } else { 180.0 };
        (angle >= plus_min && angle < plus_max)
            || (angle >= plus_min + shift && angle < plus_max + shift)
    }
}

/// Center angle of sector `s`.
pub fn sector_angle(s: usize, num_sectors: usize) -> f64 {
    (s as f64 + 0.5) * 180.0 / num_sectors as f64 - 90.0
}

/// Running sums for an average spectrum.
pub struct SpectrumSum {
    pub sums: Vec<f64>,
    pub counts: Vec<i64>,
}

impl SpectrumSum {
    pub fn new(dim: usize) -> Self {
        SpectrumSum {
            sums: vec![0.0; dim],
            counts: vec![0; dim],
        }
    }

    pub fn average(&self) -> Vec<f64> {
        self.sums
            .iter()
            .zip(&self.counts)
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect()
    }
}

/// Everything needed to scale and add one strip.
pub struct ScaleContext<'a> {
    pub model: &'a CtfModel,
    pub bin: &'a Binning,
    pub scaling: ZeroScaling,
    pub phase: Option<f64>,
    pub cuton: Option<f64>,
    /// Maps a hyper index to a noise-spectrum index (0 when no noise).
    pub nps_scale: f64,
}

/// Scales one strip spectrum from `cur_def` to `ref_def` and adds it in.
/// `ps` holds summed power per hyper bin for `tiles` tiles, `count` the
/// components per bin in one tile.
#[allow(clippy::too_many_arguments)]
pub fn scale_and_add(
    ctx: &ScaleContext,
    ps: &[f32],
    count: &[i32],
    tiles: usize,
    noise: Option<&[f64]>,
    ref_def: f64,
    cur_def: f64,
    out: &mut SpectrumSum,
) {
    if tiles == 0 {
        return;
    }
    let dim = ctx.bin.dim;
    let hyper = ctx.bin.hyper;
    let m = ctx.model;
    let hyper_inc = (1.0 / (dim as f64 - 1.0)) / hyper as f64;
    let freq_inc = 1.0 / (dim as f64 - 1.0);
    let ref_low = m.zero(ref_def, ctx.scaling.low, ctx.phase, ctx.cuton);
    let cur_low = m.zero(cur_def, ctx.scaling.low, ctx.phase, ctx.cuton);
    let (scale, offset) = if ctx.scaling.high <= 0 {
        (ref_low / cur_low, 0.0)
    } else {
        let ref_high = m.zero(ref_def, ctx.scaling.high, ctx.phase, ctx.cuton);
        let cur_high = m.zero(cur_def, ctx.scaling.high, ctx.phase, ctx.cuton);
        let s = (ref_high - ref_low) / (cur_high - cur_low);
        (s, ref_low - s * cur_low)
    };
    let nint = |v: f64| (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let limit = dim * hyper - hyper / 2;
    for ii in 0..limit {
        let nps = match noise {
            Some(n) => n[(nint(ctx.nps_scale * ii as f64) as usize).min(n.len() - 1)],
            None => 1.0,
        };
        let f0 = ii as f64 * hyper_inc;
        let f1 = scale * (f0 + hyper_inc) + offset;
        let f0 = scale * f0 + offset;
        let left = nint(f0 / freq_inc);
        let right = nint(f1 / freq_inc);
        if right >= dim as i64 || left < 0 {
            continue;
        }
        let val = ps[ii] as f64 / nps;
        let c = count[ii] as i64;
        if left == right {
            out.sums[right as usize] += val;
            out.counts[right as usize] += c * tiles as i64;
        } else if c > 0 {
            let freq = (right as f64 - 0.5) * freq_inc;
            let lfrac = ((freq - f0) / hyper_inc).clamp(0.0, 1.0);
            let lnum = nint(lfrac * c as f64);
            let lfrac = lnum as f64 / c as f64;
            out.sums[left as usize] += lfrac * val;
            out.counts[left as usize] += lnum * tiles as i64;
            out.sums[right as usize] += (1.0 - lfrac) * val;
            out.counts[right as usize] += (c - lnum) * tiles as i64;
        }
    }
}

/// Defocus offsets for each sector relative to the reference, given the
/// astigmatism (microns, angle in degrees) and an optional wedge; `None` for
/// sectors outside the wedge.  Offsets within a wedge are relative to the
/// wedge's mean defocus.
pub fn sector_offsets(
    num_sectors: usize,
    astig: Option<(f64, f64)>,
    wedge: Option<Wedge>,
) -> Vec<Option<f64>> {
    let mut offs: Vec<Option<f64>> = (0..num_sectors)
        .map(|s| {
            let ang = sector_angle(s, num_sectors);
            if wedge.is_some_and(|w| !w.contains(ang)) {
                return None;
            }
            Some(astig.map_or(0.0, |(a, axis)| {
                0.5 * a * (2.0 * (ang - axis) * RADIANS_PER_DEGREE).cos()
            }))
        })
        .collect();
    if wedge.is_some() {
        let inc: Vec<f64> = offs.iter().flatten().copied().collect();
        if !inc.is_empty() {
            let mean = inc.iter().sum::<f64>() / inc.len() as f64;
            offs.iter_mut().flatten().for_each(|v| *v -= mean);
        }
    }
    offs
}
