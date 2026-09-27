//! The analysis driver: choosing views for a range of tilt angles, building
//! the combined spectrum, fitting it, finding astigmatism and phase shift,
//! scanning for a starting defocus, and autofitting over the tilt series.
//!
//! The control flow follows native `ctfplotter`'s batch path (`main.cpp`
//! and `MyApp::angleChanged`, `multipleSpectraAndFits`, `fitPsFindZero`,
//! `autoFitToRanges`, `scanDefocusAndCrop`), restructured around one
//! `Analyzer` value instead of a GUI application object.

use std::io::Write as _;

use rayon::prelude::*;

use crate::imod::ctfplotter::ctfutils::{
    DEF_FILE_HAS_ASTIG, DEF_FILE_HAS_CUT_ON, DEF_FILE_HAS_PHASE, SavedDefocus,
    add_item_to_defocus_list,
};
use crate::imod::libcfshr::amoeba::dual_amoeba;
use crate::imod::libcfshr::b3dutil::{ImodFile, imod_backup_file};
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::minimize1d::minimize1d;
use crate::imod::libcfshr::regression::polynomial_fit;
use crate::imod::libcfshr::robuststat::rs_mad_median_outliers;
use crate::imod::libfft::odfft::nice_fft_limit;

use super::ctf::{CtfModel, RADIANS_PER_DEGREE};
use super::fit::{
    CtfFitResult, CtfFitter, FitStart, NO_PHASE, PHASE_TEST, WEIGHT_TRUNCATE, opt_phase,
    ps_minus_baseline,
};
use super::spectra::{Binning, SpectrumCache, TileParams, ViewSpectra};
use super::spectrum::{
    NoiseModel, ScaleContext, SpectrumSum, Wedge, ZeroScaling, scale_and_add, sector_offsets,
};

pub fn say(text: &str) {
    let _ = ImodFile::Stdout.write_all(text.as_bytes());
}

fn nint(v: f64) -> i64 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i64
}

/// Settings fixed for a run.
#[derive(Clone, Debug)]
pub struct RunSettings {
    pub dim: usize,
    pub hyper: usize,
    pub num_sectors: usize,
    pub tile: usize,
    pub pixel_size: f64,
    pub def_tol: f64,
    pub left_tol: f64,
    pub right_tol: f64,
    pub axis_angle: f64,
    pub tilt_offset: f64,
    pub base_order: usize,
    pub two_line: bool,
    pub vary_power: bool,
    pub weighting: i32,
    pub num_zeros_to_fit: f64,
    pub find_astig: bool,
    pub find_phase: bool,
    pub find_cuton: bool,
    pub min_views_astig: usize,
    pub min_views_phase: usize,
    pub wedge_range: f64,
    pub wedge_interval: f64,
    pub max_astig: f64,
    pub phase_range: f64,
    pub max_cuton: f64,
    pub skip_list: Vec<i32>,
    pub skip_only_astig_phase: bool,
    pub bidir_view: usize,
    pub fit_more_views: bool,
    pub num_more_views: usize,
    pub more_views_angle: f64,
    pub debug: i32,
    pub defocus_file: String,
    pub no_angles: bool,
    /// `DumpWedgeSpectra -2`: print the final spectrum and fit of each
    /// autofit range.
    pub dump_final: bool,
}

/// What a fit needs from the analyzer, shareable across threads.
#[derive(Clone, Copy)]
struct FitEnv<'a> {
    model: &'a CtfModel,
    s: &'a RunSettings,
    x1: usize,
    x2: usize,
    raw_tile: usize,
}

pub struct Analyzer {
    pub s: RunSettings,
    pub model: CtfModel,
    pub cache: SpectrumCache,
    pub noise: Option<NoiseModel>,
    pub fitter: CtfFitter,
    pub raw_tile: usize,
    pub crop_pixel: f64,
    pub crop: bool,
    pub x1: usize,
    pub x2: usize,
    pub angles: Vec<f32>,
    pub sorted: Vec<f32>,
    pub exclude_skip: bool,
    pub low_angle: f64,
    pub high_angle: f64,
    pub needed: Vec<usize>,
    pub num_views_in_range: usize,
    pub view_range_step: usize,
    pub auto_from: f64,
    pub auto_to: f64,
    pub use_cur_defocus: bool,
    pub use_cur_phase: bool,
    pub last_phase: f64,
    pub last_cuton: f64,
    pub next_fit_phase: f64,
    pub next_fit_cuton: f64,
    pub next_fit_know_phase: bool,
    pub autofit_iteration: usize,
    pub current_astig: f64,
    pub current_astig_angle: f64,
    pub next_spec_astig: f64,
    pub next_spec_astig_angle: f64,
    pub last_spec_astig: f64,
    pub last_spec_astig_angle: f64,
    pub all_views_astig: f64,
    pub all_views_astig_angle: f64,
    wedge: Option<Wedge>,
    thread_start_defocus: f64,
    pub avg: Vec<f64>,
    pub saved: Vec<SavedDefocus>,
    pub stack_mean: f64,
    last_curve: Vec<f64>,
    dump_count: usize,
}

impl Analyzer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: RunSettings,
        model: CtfModel,
        cache: SpectrumCache,
        noise: Option<NoiseModel>,
        angles: Vec<f32>,
        saved: Vec<SavedDefocus>,
    ) -> Self {
        let dim = s.dim;
        let tile = s.tile;
        let pixel = s.pixel_size;
        let exclude = !s.skip_only_astig_phase;
        let mut a = Analyzer {
            fitter: CtfFitter::new(dim),
            raw_tile: tile,
            crop_pixel: pixel,
            crop: false,
            x1: 1,
            x2: dim - 1,
            angles,
            sorted: Vec::new(),
            exclude_skip: exclude,
            low_angle: 0.0,
            high_angle: 0.0,
            needed: Vec::new(),
            num_views_in_range: 1,
            view_range_step: 1,
            auto_from: 0.0,
            auto_to: 0.0,
            use_cur_defocus: false,
            use_cur_phase: false,
            last_phase: NO_PHASE,
            last_cuton: NO_PHASE,
            next_fit_phase: NO_PHASE,
            next_fit_cuton: NO_PHASE,
            next_fit_know_phase: false,
            autofit_iteration: 0,
            current_astig: 0.0,
            current_astig_angle: 0.0,
            next_spec_astig: 0.0,
            next_spec_astig_angle: 0.0,
            last_spec_astig: 0.0,
            last_spec_astig_angle: 0.0,
            all_views_astig: 0.0,
            all_views_astig_angle: 0.0,
            wedge: None,
            thread_start_defocus: 0.0,
            avg: vec![0.0; dim],
            saved,
            stack_mean: 0.0,
            last_curve: Vec::new(),
            dump_count: 0,
            s,
            model,
            cache,
            noise,
        };
        a.sort_angles(0);
        a
    }

    pub fn nz(&self) -> usize {
        self.angles.len()
    }

    fn in_skip(&self, view: usize) -> bool {
        self.s.skip_list.contains(&(view as i32 + 1))
    }

    /// `getSortedAngles`: angles of views not skipped, sorted.  `which_way`
    /// > 0 always excludes the skip list, < 0 never, 0 follows the flag.
    pub fn sort_angles(&mut self, which_way: i32) -> usize {
        let exclude = if which_way != 0 {
            which_way > 0
        } else {
            self.exclude_skip
        };
        let mut v: Vec<f32> = (0..self.nz())
            .filter(|&i| !(exclude && self.in_skip(i)))
            .map(|i| self.angles[i])
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.sorted = v;
        self.sorted.len()
    }

    fn nearest_index(angle: f64, list: &[f32]) -> usize {
        let mut near = 0;
        for (i, &a) in list.iter().enumerate() {
            if (angle - a as f64).abs() < (angle - list[near] as f64).abs() {
                near = i;
            }
        }
        near
    }

    /// `getMidAngleAndRangeIndices`.
    pub fn mid_angle_and_range(&mut self, low: f64, high: f64) -> (f64, usize, usize, usize) {
        let (lo, hi) = (low.min(high), low.max(high));
        self.sort_angles(0);
        let ln = Self::nearest_index(lo, &self.sorted);
        let hn = Self::nearest_index(hi, &self.sorted);
        let num = (hn as i64 - ln as i64).unsigned_abs() as usize + 1;
        (self.sorted[hn - num / 2] as f64, ln, hn, num)
    }

    /// `getRangesFromMidAngle` with step 0: the range of `num_views` views
    /// around `mid`, returning (mid, low angle, high angle, low sorted index,
    /// high sorted index, low view, high view).
    fn ranges_from_mid(
        &mut self,
        num_views: usize,
        mid: f64,
    ) -> (f64, f64, f64, usize, usize, usize, usize) {
        self.sort_angles(0);
        let n = self.sorted.len() as i64;
        let new_mid = Self::nearest_index(mid, &self.sorted) as i64;
        let (mut lo_lim, mut hi_lim) = (0i64, n - 1);
        let nv = num_views as i64;
        if self.s.bidir_view > 0 && self.exclude_skip {
            let first =
                Self::nearest_index(self.angles[self.s.bidir_view - 1] as f64, &self.sorted) as i64;
            let second =
                Self::nearest_index(self.angles[self.s.bidir_view] as f64, &self.sorted) as i64;
            let brk = first.min(second);
            if brk + 1 >= nv && n - (brk + 1) >= nv {
                if new_mid <= brk {
                    hi_lim = brk;
                } else {
                    lo_lim = brk + 1;
                }
            }
        }
        let mut hs = (new_mid + nv / 2).clamp(lo_lim, hi_lim);
        let ls = (hs + 1 - nv).clamp(lo_lim, hi_lim);
        hs = (ls + nv - 1).clamp(lo_lim, hi_lim);
        let low = self.sorted[ls as usize] as f64;
        let high = self.sorted[hs as usize] as f64;
        let mid_ind = hs - (hs + 1 - ls) / 2;
        let mid_out = self.sorted[mid_ind as usize] as f64;
        let lv = Self::nearest_index(low, &self.angles);
        let hv = Self::nearest_index(high, &self.angles);
        (mid_out, low, high, ls as usize, hs as usize, lv, hv)
    }

    /// `whatIsNeeded`: views with angles in the range.
    pub fn set_needed(&mut self, low: f64, high: f64) {
        let eps = if self.s.no_angles { 0.002 } else { 0.02 };
        self.needed = (0..self.nz())
            .filter(|&k| {
                let a = if self.s.no_angles {
                    0.0
                } else {
                    self.angles[k] as f64
                };
                !(a < low - eps || a > high + eps)
                    && !(!self.s.no_angles && self.exclude_skip && self.in_skip(k))
            })
            .collect();
        if self.s.no_angles {
            // Without angles all views are at the single angle; the range
            // selects by the 0.01-degree bookkeeping increments instead.
            self.needed = (0..self.nz())
                .filter(|&k| {
                    let a = self.angles[k] as f64;
                    !(a < low - eps || a > high + eps)
                })
                .collect();
        }
    }

    /// Initial number of views in range and range step (`setSlice`).
    pub fn setup_ranges(&mut self, auto_step_angle: f64, auto_step_range: f64) {
        self.sort_angles(1);
        self.sort_angles(0);
        self.auto_from = self.sorted[0] as f64;
        self.auto_to = *self.sorted.last().unwrap() as f64;
        let (min_a, max_a) = self
            .angles
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &a| {
                (lo.min(a as f64), hi.max(a as f64))
            });
        let nz = self.nz();
        let (mut ln, mut hn) = (0usize, 0usize);
        if auto_step_angle > 0.0 {
            let (mut num_sum, mut num_in) = (0usize, 0usize);
            let mut min_angle = 1.0e10f64;
            let n = self.sorted.len();
            let mut ind = 0;
            while ind < n
                && (ind == 0
                    || (self.sorted[ind] as f64 + auto_step_range) < self.sorted[n - 1] as f64)
            {
                let lo = self.sorted[ind] as f64;
                let (mid, lt, ht, nt) = self.mid_angle_and_range(lo, lo + auto_step_range);
                num_in += 1;
                num_sum += nt;
                if mid.abs() < min_angle {
                    min_angle = mid.abs();
                    ln = lt;
                    hn = ht;
                }
                ind += 1;
            }
            self.num_views_in_range = nint(num_sum as f64 / num_in as f64) as usize;
            self.view_range_step = if max_a > min_a {
                ((nz - 1) as f64 * (auto_step_angle / (max_a - min_a))) as usize
            } else {
                1
            };
        } else {
            let (_, lt, ht, mut nv) = self.mid_angle_and_range(self.low_angle, self.high_angle);
            ln = lt;
            hn = ht;
            if nv > 1 {
                let inc = (self.sorted[hn] - self.sorted[ln]) as f64 / (nv as f64 - 1.0);
                nv = nint((self.high_angle - self.low_angle) / inc) as usize + 1;
                if nv - 1 < hn - ln {
                    if (self.low_angle - self.sorted[ln] as f64).abs()
                        < (self.high_angle - self.sorted[hn] as f64).abs()
                    {
                        hn -= 1;
                    } else {
                        ln += 1;
                    }
                }
            }
            self.num_views_in_range = nv;
            self.view_range_step = nv / 2;
        }
        self.view_range_step = self.view_range_step.clamp(1, (nz / 2).max(1));
        self.low_angle = self.sorted[ln] as f64;
        self.high_angle = self.sorted[hn] as f64;
        let (lo, hi) = (self.low_angle, self.high_angle);
        self.set_needed(lo, hi);
    }

    /// Fitting range indexes from the expected or current defocus
    /// (`getFittingRangeFromDefocus`): (first zero, first zero index,
    /// far zero index, back from zero).
    pub fn fit_range_from_defocus(&self, current: bool) -> (f64, usize, usize, usize) {
        let dim = self.s.dim;
        let min_points = 7usize;
        let far_num = self.s.num_zeros_to_fit as i32;
        let frac_far = self.s.num_zeros_to_fit - far_num as f64;
        let def = if current {
            self.model.defocus
        } else {
            self.model.exp_defocus
        };
        let (z1, z2) = self.model.two_zeros(def, None, None);
        let far = self.model.zero(def, far_num, None, None);
        let next = self.model.zero(def, far_num + 1, None, None);
        let mut sec_ind = (nint((far + frac_far * (next - far)) * (dim as f64 - 1.0)).max(0)
            as usize)
            .min(dim - 1);
        let first_ind = (sec_ind as i64 - 3)
            .min(nint(z1 * (dim as f64 - 1.0)))
            .max(0) as usize;
        let mut back = nint(0.55 * (z2 - z1) * (dim as f64 - 1.0)).clamp(4, 12) as usize;
        if first_ind < back + 3 {
            back = first_ind.saturating_sub(3);
        }
        sec_ind = sec_ind.max(first_ind - back + min_points - 1);
        (z1, first_ind, sec_ind, back)
    }

    /// `setInitialFittingRange`: `start`/`end` are fractions of Nyquist
    /// (already converted), or None for the defocus-based defaults.
    pub fn set_initial_fit_range(&mut self, start: Option<f64>, end: Option<f64>) {
        let dim = self.s.dim;
        let (_, first, sec, back) = self.fit_range_from_defocus(false);
        self.x1 = match start {
            Some(x) => nint(2.0 * x * (dim as f64 - 1.0)) as usize,
            None => first - back,
        };
        self.x2 = match end {
            Some(x) => nint(2.0 * x * (dim as f64 - 1.0)) as usize,
            None => sec,
        };
        self.clamp_range();
    }

    fn clamp_range(&mut self) {
        let dim = self.s.dim;
        self.x1 = self.x1.clamp(1, dim - 1);
        self.x2 = self.x2.clamp(1, dim - 1);
    }

    /// `adjustCachesForTileParams` for a crop pixel size (the true pixel
    /// size for no cropping).
    pub fn set_crop_pixel(&mut self, crop_pixel: f64) {
        let tile = self.s.tile;
        let mut raw = tile;
        if self.crop {
            raw = 2 * nint(0.5 * crop_pixel * tile as f64 / self.s.pixel_size) as usize;
            raw = raw.max(tile);
            raw = nice_frame(raw as i32, 2, nice_fft_limit()) as usize;
        }
        self.crop_pixel = crop_pixel;
        if raw != self.raw_tile {
            let ratio = raw as f64 / self.raw_tile as f64;
            self.model.pixel_size = if raw <= tile {
                self.s.pixel_size
            } else {
                raw as f64 * self.s.pixel_size / tile as f64
            };
            self.x1 = nint(ratio * self.x1 as f64) as usize;
            self.x2 = nint(ratio * self.x2 as f64) as usize;
            self.clamp_range();
            self.raw_tile = raw;
            let mut params = self.cache.params;
            params.raw = raw;
            let bin = Binning::new(self.s.dim, self.s.hyper, self.s.num_sectors, tile, raw);
            self.cache.reset(params, bin);
        }
    }

    /// Starting defocus and expected zeros for a fit, plus all zeros out to
    /// a little past Nyquist for baseline fitting.
    fn starting_defocus_and_zeros(&self) -> (f64, f64, f64, Vec<f64>) {
        let start = if self.thread_start_defocus > 0.0 {
            self.thread_start_defocus
        } else if self.use_cur_defocus {
            self.model.defocus
        } else {
            self.model.exp_defocus
        };
        let mut phase = self.next_fit_phase;
        if phase < PHASE_TEST && self.use_cur_phase {
            phase = self.last_phase;
        }
        let mut cuton = self.next_fit_cuton;
        if cuton < PHASE_TEST && self.use_cur_phase {
            cuton = self.last_cuton;
        }
        let (z1, z2) = self
            .model
            .two_zeros(start, opt_phase(phase), opt_phase(cuton));
        let mut all = vec![z1, z2];
        let mut past = 0;
        for ind in 2..self.s.dim.saturating_sub(1) {
            let z = self
                .model
                .zero(start, ind as i32, opt_phase(phase), opt_phase(cuton));
            all.push(z);
            if z > 1.0 {
                past += 1;
            }
            if past > 2 {
                break;
            }
        }
        (start, z1, z2, all)
    }

    fn phase_for_scaling(&self) -> (Option<f64>, Option<f64>) {
        if self.use_cur_phase || self.wedge.is_some() {
            (opt_phase(self.last_phase), opt_phase(self.last_cuton))
        } else {
            (None, None)
        }
    }

    /// `doComputePS`: the combined spectrum of the needed views into
    /// `self.avg`.  `initial` uses only the central strip, unshifted.
    pub fn compute_ps(&mut self, initial: bool) -> Result<(), String> {
        let dim = self.s.dim;
        let effective = if !initial && self.use_cur_defocus {
            self.model.defocus
        } else {
            self.model.exp_defocus
        };
        let (phase, cuton) = self.phase_for_scaling();
        let mut scaling = ZeroScaling { low: 1, high: 0 };
        let k =
            self.model
                .zeros_in_range(effective, self.x2 as f64 / (dim as f64 - 1.0), phase, cuton);
        if k > 1 {
            if k < 5 {
                scaling.high = k;
            } else {
                scaling.low = 2;
                scaling.high = k - 1;
            }
        }
        let astig = (self.next_spec_astig != 0.0)
            .then_some((self.next_spec_astig, self.next_spec_astig_angle));
        self.current_astig = astig.map_or(0.0, |a| a.0);
        self.current_astig_angle = astig.map_or(0.0, |a| a.1);
        let wedge = self.wedge;
        let use_sectors = astig.is_some() || wedge.is_some();
        let offsets = sector_offsets(self.s.num_sectors, astig, wedge);
        if self.s.debug >= 1 && wedge.is_none() {
            say(&format!(
                "Making spectrum for iter {} from {} slices with astigmatism {} known\n",
                self.autofit_iteration,
                self.needed.len(),
                if astig.is_some() { "" } else { "NOT" }
            ));
            if self.needed.len() == 1 {
                say(&format!(
                    "Slice {} is included, tilt angle is {:.2} degrees. \n",
                    self.needed[0],
                    self.view_angle(self.needed[0])
                ));
            } else if let (Some(&a), Some(&b)) =
                (self.needed.iter().min(), self.needed.iter().max())
            {
                say(&format!(
                    "Slices {} to {} are included, tilt angles are {:.2} to {:.2} degrees. \n",
                    a,
                    b,
                    self.view_angle(a),
                    self.view_angle(b)
                ));
            }
        }
        let needed = self.needed.clone();
        let tile = self.s.tile;
        let raw = self.raw_tile;
        let nps_scale = match &self.noise {
            Some(n) => {
                (tile as f64 * n.dim as f64) / (raw as f64 * self.s.hyper as f64 * dim as f64)
            }
            None => 0.0,
        };
        let mut sum = SpectrumSum::new(dim);
        let pixel = self.s.pixel_size;
        let offset_deg = self.s.tilt_offset;
        // Sectors are scaled in groups of similar defocus: a wedge without
        // astigmatism is one group; with astigmatism the sectors are divided
        // into bands narrower than the defocus tolerance for the fitting
        // range, as native does
        let groups: Vec<(Vec<usize>, f64)> = if !use_sectors {
            Vec::new()
        } else if astig.is_none() {
            vec![(
                (0..offsets.len())
                    .filter(|&i| offsets[i].is_some())
                    .collect(),
                0.0,
            )]
        } else {
            let inc: Vec<(usize, f64)> = offsets
                .iter()
                .enumerate()
                .filter_map(|(i, o)| o.map(|v| (i, v)))
                .collect();
            let (dmin, dmax) = inc
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), &(_, v)| (a.min(v), b.max(v)));
            let mut tol = self.model.tolerance(
                self.x2 as f64 / (dim as f64 - 1.0),
                0.2,
                effective,
                phase,
                cuton,
            );
            let range = dmax - dmin;
            if range <= tol && range > 0.4 && wedge.is_none() {
                tol = 0.6 * range;
            }
            let num_bands = if range > tol {
                (range / tol) as usize + 1
            } else {
                1
            };
            let width = (range + 0.001) / num_bands as f64;
            let mut bands: Vec<(Vec<usize>, f64)> = vec![(Vec::new(), 0.0); num_bands];
            for &(i, v) in &inc {
                let b = (((v - dmin) / width) as usize).min(num_bands - 1);
                bands[b].0.push(i);
                bands[b].1 += v;
            }
            bands
                .into_iter()
                .filter(|b| !b.0.is_empty())
                .map(|(secs, sum)| {
                    let n = secs.len() as f64;
                    (secs, sum / n)
                })
                .collect()
        };
        for &view in &needed {
            let angle = (self.view_angle(view) + offset_deg) * RADIANS_PER_DEGREE;
            self.cache
                .ensure_view(view, use_sectors || self.s.find_astig)?;
            let (spectra, bin) = self.cache.get(view);
            let ctx = ScaleContext {
                model: &self.model,
                bin,
                scaling,
                phase,
                cuton,
                nps_scale,
            };
            let nstrips = if initial { 1 } else { spectra.strips.len() };
            let one = bin.one_ps;
            let group_counts: Vec<Vec<i32>> = groups
                .iter()
                .map(|(secs, _)| {
                    let mut c = vec![0i32; one];
                    for &sc in secs {
                        c.iter_mut()
                            .zip(&bin.sector_count[sc * one..(sc + 1) * one])
                            .for_each(|(a, b)| *a += b);
                    }
                    c
                })
                .collect();
            let mut jobs: Vec<(usize, usize, f64, Option<usize>)> = Vec::new();
            for (si, strip) in spectra.strips.iter().take(nstrips).enumerate() {
                for (side, sign) in [(1usize, 1.0f64), (0usize, -1.0f64)] {
                    if strip[side].count == 0 {
                        continue;
                    }
                    if use_sectors {
                        for g in 0..groups.len() {
                            jobs.push((si, side, sign, Some(g)));
                        }
                    } else {
                        jobs.push((si, side, sign, None));
                    }
                }
            }
            let noise = &self.noise;
            let parts: Vec<SpectrumSum> = jobs
                .par_iter()
                .map(|&(si, side, sign, sec)| {
                    let sp = &spectra.strips[si][side];
                    let mut part = SpectrumSum::new(dim);
                    let delta_z = if initial {
                        0.0
                    } else {
                        sign * sp.axis_dist * pixel * angle.tan() / 1000.0
                    };
                    let nps = noise.as_ref().map(|n| n.for_mean(sp.mean));
                    let strip_def = effective + delta_z;
                    match sec {
                        Some(g) => {
                            let (secs, off) = &groups[g];
                            let mut ps = vec![0f32; one];
                            for &sc in secs {
                                ps.iter_mut()
                                    .zip(&sp.sectors[sc * one..(sc + 1) * one])
                                    .for_each(|(a, b)| *a += b);
                            }
                            scale_and_add(
                                &ctx,
                                &ps,
                                &group_counts[g],
                                sp.count,
                                nps.as_deref(),
                                effective,
                                strip_def + off,
                                &mut part,
                            );
                        }
                        None => scale_and_add(
                            &ctx,
                            &sp.ps,
                            &bin.count,
                            sp.count,
                            nps.as_deref(),
                            effective,
                            strip_def,
                            &mut part,
                        ),
                    }
                    part
                })
                .collect();
            for part in parts {
                sum.sums
                    .iter_mut()
                    .zip(&part.sums)
                    .for_each(|(a, b)| *a += b);
                sum.counts
                    .iter_mut()
                    .zip(&part.counts)
                    .for_each(|(a, b)| *a += b);
            }
        }
        self.avg = sum.average();
        self.next_fit_know_phase = false;
        self.next_spec_astig = 0.0;
        self.next_spec_astig_angle = 0.0;
        Ok(())
    }

    fn view_angle(&self, view: usize) -> f64 {
        if self.s.no_angles {
            self.angles[0] as f64
        } else {
            self.angles[view] as f64
        }
    }

    /// Fit order for the CTF-like curve (`startingDefocusZerosFitOrder`).
    fn fit_order(&self, z1: f64, z2: f64) -> usize {
        let dim = self.s.dim;
        let mut order = if self.s.vary_power { 5 } else { 4 };
        if self.x2 as f64 / (dim as f64 - 1.0) >= z1 + 1.5 * (z2 - z1)
            && (self.x2 + 1 - self.x1) > 2 * (order + 2)
        {
            order += 2;
        }
        order
    }

    fn fit(
        &mut self,
        nvar: usize,
        phase: f64,
        cuton: f64,
        fixed: f64,
        start: FitStart,
    ) -> CtfFitResult {
        self.fitter.fit_ctf(
            &self.model,
            nvar,
            phase,
            cuton,
            fixed,
            start,
            self.s.max_cuton,
            self.s.phase_range,
            self.s.weighting,
        )
    }

    fn report_fit(&self, r: &CtfFitResult) {
        if self.s.debug >= 1 {
            say(&format!(
                "CTF fitting parameters for range {} to {} are:\n{}\n",
                self.x1,
                self.x2,
                self.fitter.describe(r.error)
            ));
        }
    }

    /// Fits a given spectrum with a given starting defocus and fixed phase,
    /// without touching the analyzer state (used for parallel wedge fits).
    fn fit_given(
        env: &FitEnv,
        avg: &[f64],
        fitter: &mut CtfFitter,
        start_def: f64,
        phase: f64,
        cuton: f64,
    ) -> CtfFitResult {
        let FitEnv {
            model,
            s,
            x1,
            x2,
            raw_tile,
        } = *env;
        let ratio = raw_tile.max(s.tile) as f64 / s.tile as f64;
        let (z1, z2) = model.two_zeros(start_def, opt_phase(phase), opt_phase(cuton));
        let mut all = vec![z1, z2];
        let mut past = 0;
        for ind in 2..s.dim.saturating_sub(1) {
            let z = model.zero(start_def, ind as i32, opt_phase(phase), opt_phase(cuton));
            all.push(z);
            if z > 1.0 {
                past += 1;
            }
            if past > 2 {
                break;
            }
        }
        fitter.set_range(x1, x2);
        fitter.raw = ps_minus_baseline(avg, s.two_line, s.base_order, ratio, x1, &all);
        let mut order = if s.vary_power { 5 } else { 4 };
        if x2 as f64 / (s.dim as f64 - 1.0) >= z1 + 1.5 * (z2 - z1)
            && (x2 + 1 - x1) > 2 * (order + 2)
        {
            order += 2;
        }
        let start = FitStart {
            defocus: start_def,
            zero1: z1,
            zero2: z2,
        };
        fitter.fit_ctf(
            model,
            order,
            phase,
            cuton,
            0.0,
            start,
            s.max_cuton,
            s.phase_range,
            s.weighting,
        )
    }

    /// `plotFitPS` + `fitPsFindZero`: subtracts the background from the
    /// current spectrum and fits it, setting the current defocus.
    pub fn fit_spectrum(&mut self) {
        if !self.s.find_phase {
            self.last_phase = NO_PHASE;
        }
        if !self.s.find_phase || !self.s.find_cuton {
            self.last_cuton = NO_PHASE;
        }
        let ratio = self.raw_tile.max(self.s.tile) as f64 / self.s.tile as f64;
        let (start_def, z1, z2, all_zeros) = self.starting_defocus_and_zeros();
        self.fitter.set_range(self.x1, self.x2);
        self.fitter.raw = ps_minus_baseline(
            &self.avg,
            self.s.two_line,
            self.s.base_order,
            ratio,
            self.x1,
            &all_zeros,
        );
        let order = self.fit_order(z1, z2);
        let start = FitStart {
            defocus: start_def,
            zero1: z1,
            zero2: z2,
        };
        let mut result: Option<CtfFitResult> = None;

        if self.s.find_phase
            && !self.next_fit_know_phase
            && self.wedge.is_none()
            && self.autofit_iteration == 0
        {
            result = self.search_phase(order, start);
        }
        if result.is_none() {
            let r = self.fit(order, self.next_fit_phase, self.next_fit_cuton, 0.0, start);
            self.report_fit(&r);
            let mut r = r;
            if self.autofit_iteration > 0
                && self.s.find_phase
                && self.s.min_views_phase <= self.num_views_in_range
            {
                r = self.fit(2, self.next_fit_phase, self.next_fit_cuton, 0.0, start);
                self.report_fit(&r);
                if let Some(p) = r.phase {
                    self.last_phase = p;
                }
            }
            result = Some(r);
        }
        let r = result.unwrap();
        self.last_curve = r.curve.clone();
        self.next_fit_phase = NO_PHASE;
        self.next_fit_know_phase = false;
        self.next_fit_cuton = NO_PHASE;
        self.model.defocus = if r.focus.is_finite() && r.focus > 0.0 {
            r.focus
        } else {
            -2000.0
        };
    }

    /// Brute-force search for phase (and cut-on) by fitting the spectrum at
    /// a series of fixed phases, then refining (`fitPsFindZero`).
    fn search_phase(&mut self, order: usize, start: FitStart) -> Option<CtfFitResult> {
        let find_cuton_setting = self.s.find_cuton;
        let num_steps = 10;
        let max_cuts = 6;
        for fallback in 0..(if find_cuton_setting { 2 } else { 1 }) {
            let mut find_cuton = find_cuton_setting;
            let mut fallback_cuton = NO_PHASE;
            if fallback > 0 {
                fallback_cuton = if self.use_cur_phase {
                    self.last_cuton
                } else {
                    NO_PHASE
                };
                if fallback_cuton < PHASE_TEST {
                    fallback_cuton = self.model.cut_on;
                }
                find_cuton = false;
                if self.s.debug >= 1 {
                    say(&format!(
                        "Searching for phase only with fallback cuton={:.6}\n",
                        fallback_cuton
                    ));
                }
            }
            let cuton_step = (self.s.max_cuton / num_steps as f64) as f32;
            let phase_step = (self.s.phase_range / num_steps as f64) as f32;
            let mut saw_min = false;
            let mut error = 0;
            let mut min_err = 1.0e20f32;
            let mut cur_cuton: f32 = if find_cuton_setting {
                0.0
            } else {
                NO_PHASE as f32
            };
            if fallback > 0 {
                cur_cuton = fallback_cuton as f32;
            }
            let mut cuton_cuts = -1;
            let mut cuton_brack = [0f32; 16];
            let mut best_phase = 0f32;
            let mut next = 0f32;
            while error == 0 {
                let mut cur_phase = (self.model.plate_phase - self.s.phase_range / 2.0) as f32;
                let mut phase_cuts = -1;
                let mut min_for_phase = 1.0e20f64;
                let mut best_foc_for_phase = start.defocus;
                let mut phase_brack = [0f32; 16];
                loop {
                    let r = self.fit(order, cur_phase as f64, cur_cuton as f64, 0.0, start);
                    if r.error < min_for_phase {
                        min_for_phase = r.error;
                        best_foc_for_phase = r.focus;
                    }
                    error = minimize1d(
                        cur_phase,
                        r.error as f32,
                        phase_step,
                        num_steps,
                        &mut phase_cuts,
                        &mut phase_brack,
                        &mut next,
                    );
                    if error != 0 || phase_cuts > max_cuts {
                        break;
                    }
                    cur_phase = next;
                }
                if self.s.debug > 1 {
                    say(&format!(
                        "Cuton= {:.3}   best phase={:6.1}  focus={:7.3}  error={:.9}\n",
                        cur_cuton,
                        phase_brack[1] as f64 / RADIANS_PER_DEGREE,
                        best_foc_for_phase,
                        phase_brack[8]
                    ));
                }
                if error == 0 && phase_brack[8] < min_err {
                    min_err = phase_brack[8];
                    best_phase = phase_brack[1];
                    if cuton_cuts <= 0 {
                        saw_min = false;
                    }
                }
                if (error == 0 && phase_brack[8] / min_err > 1.01) || phase_brack[8] / min_err > 1.1
                {
                    saw_min = true;
                }
                if error != 0 && !saw_min {
                    break;
                }
                if find_cuton {
                    let val = if error != 0 {
                        2.0 * min_err
                    } else {
                        phase_brack[8]
                    };
                    error = minimize1d(
                        cur_cuton,
                        val,
                        cuton_step,
                        num_steps,
                        &mut cuton_cuts,
                        &mut cuton_brack,
                        &mut next,
                    );
                    if error != 0 || cuton_cuts > max_cuts {
                        break;
                    }
                    cur_cuton = next;
                } else {
                    break;
                }
            }
            if saw_min || (!find_cuton && error == 0) {
                let best_cuton = if find_cuton {
                    cuton_brack[1]
                } else {
                    cur_cuton
                };
                if self.s.debug >= 1 {
                    let mut line = format!(
                        "Fitting to spectrum gave best phase= {:.1}",
                        best_phase as f64 / RADIANS_PER_DEGREE
                    );
                    if find_cuton {
                        line += &format!("  cuton= {:.3}", best_cuton);
                    }
                    say(&(line + "\n"));
                }
                let r = self.fit(order, best_phase as f64, best_cuton as f64, 0.0, start);
                self.report_fit(&r);
                let r = self.fit(2, best_phase as f64, best_cuton as f64, 0.0, start);
                self.report_fit(&r);
                if let Some(p) = r.phase {
                    self.last_phase = p;
                }
                if self.s.find_cuton {
                    self.last_cuton = best_cuton as f64;
                }
                return Some(r);
            } else if self.s.debug >= 1 {
                say(&format!(
                    "Fitting to spectrum failed to find phase{}\n",
                    if find_cuton { " and cuton" } else { "" }
                ));
            }
        }
        None
    }

    /// Fits the wedge defocuses to get mean defocus, astigmatism and angle.
    fn fit_wedges(angles: &[f64], defocus: &[f64], range: f64) -> (f64, f64, f64, f64) {
        let n = angles.len();
        let mut weights = vec![1.0f32; n];
        let focus = defocus.iter().sum::<f64>() / n as f64;
        let resid = |aa: &[f32], w: &[f32], out: Option<&mut Vec<f32>>| -> f32 {
            let (df1, df2, axis) = (aa[0], aa[1], aa[2]);
            let sin_fac = (2.0 * RADIANS_PER_DEGREE) as f32;
            let rng = (RADIANS_PER_DEGREE * range) as f32;
            let mut err = 0.0f64;
            let mut res = Vec::with_capacity(n);
            for i in 0..n {
                let ang = angles[i] as f32;
                let pred = 0.5
                    * (df1
                        + df2
                        + (df1 - df2)
                            * 0.5
                            * ((sin_fac * (ang + range as f32 / 2.0 - axis)).sin()
                                - (sin_fac * (ang - range as f32 / 2.0 - axis)).sin())
                            / rng);
                let r = pred - defocus[i] as f32;
                res.push(r);
                err += w[i] as f64 * r as f64 * r as f64;
            }
            if let Some(o) = out {
                *o = res;
            }
            (err / n as f64).sqrt() as f32
        };
        let mut aa = [focus as f32, focus as f32, 0.0];
        let mut err_min = resid(&aa, &weights, None);
        for ind in 1..=10 {
            let mut ang = -90;
            while ang < 90 {
                let t = [
                    focus as f32 + 0.1 * ind as f32,
                    focus as f32 - 0.1 * ind as f32,
                    ang as f32,
                ];
                let e = resid(&t, &weights, None);
                if e < err_min {
                    err_min = e;
                    aa = t;
                }
                ang += 10;
            }
        }
        let da = [0.5f32, 0.5, 10.0];
        let mut run = |aa: &mut [f32; 3], w: &[f32]| {
            let mut yy = [0f32; 4];
            let mut it = 0;
            let mut f = |p: &[f32]| resid(p, w, None);
            dual_amoeba(
                &mut yy,
                3,
                2.0,
                &[5.0e-4, 1.0e-5],
                &[5.0e-4, 1.0e-5],
                aa,
                &da,
                &mut f,
                &mut it,
            );
        };
        run(&mut aa, &weights);
        let mut res = Vec::new();
        let mut error = resid(&aa, &weights, Some(&mut res));

        // Outliers from local quadratic fits to the residuals
        let num_test = 7usize.max(n / 4).min(n);
        let mut num_out = 0;
        let mut xx = vec![0f32; num_test];
        let mut yy = vec![0f32; num_test];
        let mut work = vec![0f32; 3 * (5 + num_test) + 64];
        for i in 0..n {
            for k in 0..num_test {
                yy[k] = res[(i + k) % n];
                xx[k] = k as f32;
            }
            let mut slopes = [0f32; 4];
            let mut intcp = [0f32; 1];
            if polynomial_fit(
                &xx,
                &yy,
                num_test as i32,
                2,
                &mut slopes,
                &mut intcp,
                &mut work,
            ) == 0
            {
                for k in 0..num_test {
                    yy[k] -= k as f32 * slopes[0] + (k * k) as f32 * slopes[1] + intcp[0];
                }
                let mut out = vec![0f32; num_test.max(2) * 2];
                rs_mad_median_outliers(&yy, num_test as i32, 3.0, &mut out);
                if out[num_test / 2] != 0.0 {
                    weights[(i + num_test / 2) % n] = 0.0;
                    num_out += 1;
                }
            }
        }
        if num_out > 0 && num_out < n / 4 {
            run(&mut aa, &weights);
            error = resid(&aa, &weights, None);
        }
        (
            ((aa[0] + aa[1]) / 2.0) as f64,
            (aa[0] - aa[1]) as f64,
            aa[2] as f64,
            error as f64,
        )
    }

    /// `multipleSpectraAndFits`: finds astigmatism and/or phase from wider
    /// ranges of views, then computes and fits the final spectrum.  Returns
    /// true when it did the final fit.
    pub fn multiple_spectra_and_fits(&mut self) -> Result<bool, String> {
        if self.autofit_iteration > 0 {
            return Ok(false);
        }
        let exclude_save = self.exclude_skip;
        let (mid0, _, _, num_for_def) = self.mid_angle_and_range(self.low_angle, self.high_angle);
        let (mid, lo, hi, dls, dhs, dlv, dhv) = self.ranges_from_mid(num_for_def, mid0);
        self.low_angle = lo;
        self.high_angle = hi;
        let (_, _, _, als, ahs, alv, ahv) =
            self.ranges_from_mid(num_for_def.max(self.s.min_views_astig), mid);
        let (_, _, _, pls, phs, plv, phv) =
            self.ranges_from_mid(num_for_def.max(self.s.min_views_phase), mid);
        self.exclude_skip = exclude_save;
        let need_astig_range = dls != als
            || dlv != alv
            || dhs != ahs
            || dhv != ahv
            || num_for_def < self.s.min_views_astig;
        let need_phase_range = dls != pls
            || dlv != plv
            || dhs != phs
            || dhv != phv
            || num_for_def < self.s.min_views_phase;
        let find_astig = self.s.find_astig;
        let find_phase = self.s.find_phase && need_phase_range;
        if !find_astig && !find_phase {
            return Ok(false);
        }
        let mut wider = 0usize;
        let mut next_fit_phase = NO_PHASE;
        let mut next_fit_cuton = NO_PHASE;
        let (mut spec_astig, mut spec_angle) = (0.0f64, 0.0f64);

        if find_astig {
            if self.s.min_views_astig >= self.nz() && self.all_views_astig != 0.0 {
                spec_astig = self.all_views_astig;
                spec_angle = self.all_views_astig_angle;
                self.last_spec_astig = spec_astig;
                self.last_spec_astig_angle = spec_angle;
            } else {
                if self.s.find_phase {
                    let min_pre = self.s.min_views_astig.max(self.s.min_views_phase);
                    let ast = self.s.min_views_phase < self.s.min_views_astig;
                    let differs = if ast {
                        dls != als || dlv != alv || dhs != ahs || dhv != ahv
                    } else {
                        dls != pls || dlv != plv || dhs != phs || dhv != phv
                    };
                    if num_for_def < min_pre || differs {
                        self.set_wider_range(mid, min_pre);
                        wider = min_pre;
                    }
                    self.compute_ps(false)?;
                    self.fit_spectrum();
                    next_fit_phase = self.last_phase;
                    next_fit_cuton = self.last_cuton;
                    if !need_astig_range && wider > 0 {
                        self.exclude_skip = exclude_save;
                        let (lo, hi) = (self.low_angle, self.high_angle);
                        self.set_needed(lo, hi);
                        wider = 0;
                    }
                }
                if need_astig_range {
                    self.set_wider_range(mid, self.s.min_views_astig);
                    wider = self.s.min_views_astig;
                }

                // Initial fit sets the center of the defocus range
                self.compute_ps(false)?;
                self.next_fit_phase = next_fit_phase;
                self.next_fit_cuton = next_fit_cuton;
                self.fit_spectrum();
                let center_def = self.model.defocus;
                let angle_inc = 180.0 / self.s.num_sectors as f64;
                let sect_in_step = nint(self.s.wedge_interval / angle_inc).max(1) as usize;
                let num_in_wedge = nint(self.s.wedge_range / angle_inc) as usize;
                let angle_step = angle_inc * sect_in_step as f64;
                let num_steps = nint(180.0 / angle_step) as usize;
                let mut astigs: Vec<f64> = Vec::new();
                for astig_loop in 0..7 {
                    self.fitter.defocus_range = self.s.max_astig;
                    let mut centers = Vec::with_capacity(num_steps);
                    let mut fit_def = Vec::with_capacity(num_steps);
                    let mut wedge_specs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(num_steps);
                    for w in 0..num_steps {
                        let center = w as f64 * angle_step
                            + if num_in_wedge % 2 == 1 {
                                0.5 * angle_inc
                            } else {
                                0.0
                            }
                            - 90.0;
                        centers.push(center);
                        let mut start_def = center_def;
                        if astig_loop > 0 {
                            self.next_spec_astig = self.last_spec_astig;
                            self.next_spec_astig_angle = self.last_spec_astig_angle;
                            for div in -10..10 {
                                start_def += 0.5
                                    * self.next_spec_astig
                                    * (2.0
                                        * (center
                                            + (div as f64 + 0.5) * self.s.wedge_range / 20.0
                                            - self.next_spec_astig_angle)
                                        * RADIANS_PER_DEGREE)
                                        .cos()
                                    / 20.0;
                            }
                        }
                        self.wedge = Some(Wedge {
                            center,
                            range: self.s.wedge_range,
                        });
                        let save_def = self.model.defocus;
                        self.model.defocus = center_def;
                        let debug = self.s.debug;
                        self.s.debug = debug - 2;
                        self.compute_ps(false)?;
                        self.s.debug = debug;
                        self.model.defocus = save_def;
                        self.wedge = None;
                        wedge_specs.push((start_def, std::mem::take(&mut self.avg)));
                    }

                    // Fit the wedge spectra in parallel, each with its own
                    // copy of the fitter as native does with its threads
                    let env = FitEnv {
                        model: &self.model,
                        s: &self.s,
                        x1: self.x1,
                        x2: self.x2,
                        raw_tile: self.raw_tile,
                    };
                    let base_fitter = &self.fitter;
                    let results: Vec<(f64, f64, f64)> = wedge_specs
                        .par_iter()
                        .map(|(start_def, avg)| {
                            let mut fitter = base_fitter.clone();
                            let r = Self::fit_given(
                                &env,
                                avg,
                                &mut fitter,
                                *start_def,
                                next_fit_phase,
                                next_fit_cuton,
                            );
                            (r.focus, fitter.last_error, fitter.last_nonzero_freq)
                        })
                        .collect();
                    for (w, &(focus, err, nz)) in results.iter().enumerate() {
                        fit_def.push(focus);
                        if self.s.debug >= 2 {
                            say(&format!(
                                "Fitting angle {}  defocus {}   error {}   fit to {:.3}\n",
                                centers[w], focus, err, nz
                            ));
                        }
                    }
                    let (mean, mut astig, mut ang, err) =
                        Self::fit_wedges(&centers, &fit_def, self.s.wedge_range);
                    if astig < 0.0 {
                        astig = -astig;
                        ang += if ang < 0.0 { 90.0 } else { -90.0 };
                    }
                    if ang <= -90.0 {
                        ang += 180.0;
                    }
                    if ang > 90.0 {
                        ang -= 180.0;
                    }
                    if self.s.debug >= 1 {
                        say(&format!(
                            "Wedgefit f1={:.3}  f2={:.3}  mean={:.3}  astig={:.3}  ang={:.1}  error={:.4}\n",
                            mean + astig / 2.0,
                            mean - astig / 2.0,
                            mean,
                            astig,
                            ang,
                            err
                        ));
                    }
                    if astig == 0.0 {
                        astig = 0.00001;
                    }
                    spec_astig = astig;
                    spec_angle = ang;
                    astigs.push(astig);
                    self.last_spec_astig = astig;
                    self.last_spec_astig_angle = ang;
                    if astig_loop == 0 {
                        let mut max_range = -1.0f64;
                        let offs_all = sector_offsets(self.s.num_sectors, Some((astig, ang)), None);
                        for &c in &centers {
                            let w = Wedge {
                                center: c,
                                range: self.s.wedge_range,
                            };
                            let (mut dmin, mut dmax) = (1.0e20f64, -1.0e20f64);
                            for (s, o) in offs_all.iter().enumerate() {
                                if w.contains(super::spectrum::sector_angle(s, self.s.num_sectors))
                                {
                                    let d = center_def + o.unwrap_or(0.0);
                                    dmin = dmin.min(d);
                                    dmax = dmax.max(d);
                                }
                            }
                            max_range = max_range.max(dmax - dmin);
                        }
                        let tol = self.model.tolerance(
                            self.x2 as f64 / (self.s.dim as f64 - 1.0),
                            0.2,
                            center_def,
                            opt_phase(next_fit_phase),
                            opt_phase(next_fit_cuton),
                        );
                        if max_range < tol
                            && !(max_range > 0.05 * center_def && self.s.base_order > 0)
                        {
                            break;
                        }
                        self.last_spec_astig = astig.min(1.0);
                    } else {
                        let l = astigs.len();
                        if (astigs[l - 1] - astigs[l - 2]).abs() < 0.01 {
                            break;
                        }
                        if l > 2
                            && (astigs[l - 1] - astigs[l - 2]).abs()
                                > (astigs[l - 2] - astigs[l - 3]).abs()
                        {
                            break;
                        }
                    }
                }
                self.fitter.defocus_range = 0.0;
                if self.s.min_views_astig >= self.nz() && self.nz() > 1 {
                    self.all_views_astig = spec_astig;
                    self.all_views_astig_angle = spec_angle;
                }
            }
        }

        if find_phase {
            if wider != self.s.min_views_phase {
                self.set_wider_range(mid, self.s.min_views_phase);
                wider = self.s.min_views_phase;
            }
            self.next_spec_astig = spec_astig;
            self.next_spec_astig_angle = spec_angle;
            self.compute_ps(false)?;
            self.fit_spectrum();
            next_fit_phase = self.last_phase;
            next_fit_cuton = self.last_cuton;
        }

        self.exclude_skip = exclude_save;
        if wider > 0 {
            if self.s.debug > 0 {
                say(&format!("restoring range to {}\n", num_for_def));
            }
            let (lo, hi) = (self.low_angle, self.high_angle);
            self.set_needed(lo, hi);
        }
        self.next_spec_astig = spec_astig;
        self.next_spec_astig_angle = spec_angle;
        self.compute_ps(false)?;
        self.next_fit_know_phase = find_phase;
        self.next_fit_phase = if find_phase || find_astig {
            next_fit_phase
        } else {
            NO_PHASE
        };
        self.next_fit_cuton = if find_phase || find_astig {
            next_fit_cuton
        } else {
            NO_PHASE
        };
        self.fit_spectrum();
        Ok(true)
    }

    fn set_wider_range(&mut self, mid: f64, min_views: usize) {
        self.exclude_skip = true;
        if self.s.debug > 0 {
            say(&format!("setting wider range to {}\n", min_views));
        }
        let (_, lo, hi, ..) = self.ranges_from_mid(min_views, mid);
        self.set_needed(lo, hi);
    }

    /// `angleChanged` for batch use: select views, compute and fit.
    pub fn angle_changed(&mut self, low: f64, high: f64) -> Result<(), String> {
        if low != self.low_angle || high != self.high_angle {
            self.low_angle = low;
            self.high_angle = high;
            self.set_needed(low, high);
        }
        if self.needed.is_empty() {
            return Ok(());
        }
        if self.multiple_spectra_and_fits()? {
            return Ok(());
        }
        if self.autofit_iteration > 0 && self.s.find_astig {
            self.next_spec_astig = self.last_spec_astig;
            self.next_spec_astig_angle = self.last_spec_astig_angle;
        }
        if self.autofit_iteration > 0 && self.s.find_phase {
            self.next_fit_phase = self.last_phase;
            self.next_fit_cuton = self.last_cuton;
        }
        self.compute_ps(false)?;
        self.fit_spectrum();
        Ok(())
    }

    /// `saveCurrentDefocus`.
    pub fn save_current(&mut self) {
        let (Some(&first), Some(&last)) = (self.needed.first(), self.needed.last()) else {
            return;
        };
        let mut item = SavedDefocus {
            starting_slice: first as i32,
            ending_slice: last as i32,
            l_angle: self.low_angle,
            h_angle: self.high_angle,
            defocus: self.model.defocus,
            defocus2: 0.0,
            astig_angle: 0.0,
            plate_phase: self.model.plate_phase,
            cut_on_freq: self.model.cut_on,
        };
        if self.current_astig != 0.0 {
            item.defocus += self.current_astig / 2.0;
            item.defocus2 = item.defocus - self.current_astig;
            item.astig_angle = self.current_astig_angle;
        }
        if self.last_phase > PHASE_TEST {
            item.plate_phase = self.last_phase;
        }
        if self.last_cuton > PHASE_TEST {
            item.cut_on_freq = self.last_cuton;
        }
        add_item_to_defocus_list(&mut self.saved, item);
    }

    /// `autoFitToRanges`: fits ranges of views stepping through the series,
    /// iterating each fit `num_iter` times with the current defocus.
    pub fn autofit(
        &mut self,
        min_angle: f64,
        max_angle: f64,
        range_step_in: usize,
        num_iter: usize,
    ) -> Result<(), String> {
        let eps = if self.s.no_angles { 0.002 } else { 0.02 };
        let (mid_angle, ..) = self.mid_angle_and_range(self.low_angle, self.high_angle);
        let (_, mut low_near, mut high_near, _) = self.mid_angle_and_range(min_angle, max_angle);
        let range_size = self.num_views_in_range;
        let range_step = range_step_in.max(1);
        let n = self.sorted.len();
        if low_near != high_near {
            let dir: i64 = if high_near > low_near { 1 } else { -1 };
            if (self.sorted[low_near] as f64) + 0.2 < min_angle {
                let v = low_near as i64 + dir;
                if v >= 0 && (v as usize) < n {
                    low_near = v as usize;
                }
            }
            if (self.sorted[high_near] as f64) - 0.2 > max_angle {
                let v = high_near as i64 - dir;
                if v >= 0 && (v as usize) < n {
                    high_near = v as usize;
                }
            }
        }
        let more = self.s.fit_more_views && self.s.num_more_views > range_size;
        let sorted = self.sorted.clone();
        let wide = |lo: usize, hi: usize| -> bool {
            more && ((sorted[lo] + sorted[hi.min(n - 1)]) as f64 / 2.0).abs()
                >= self.s.more_views_angle
        };
        let mut ind_lo = low_near;
        let mut ind_hi = (ind_lo + range_size - 1).min(n - 1);
        let mut low_mid = ind_hi as i64 - (range_size / 2) as i64;
        if wide(ind_lo, ind_hi) {
            ind_hi = (ind_lo + self.s.num_more_views - 1).min(n - 1);
            low_mid = ind_hi as i64 - (self.s.num_more_views / 2) as i64;
        }
        ind_hi = high_near;
        ind_lo = ind_hi.saturating_sub(range_size - 1);
        let mut high_mid = ind_hi as i64 - (range_size / 2) as i64;
        if wide(ind_lo, ind_hi) {
            high_mid = ind_hi as i64 - (self.s.num_more_views / 2) as i64;
        }
        let num_steps = ((high_mid - low_mid) + range_step as i64 - 1) / range_step as i64 + 1;
        if self.s.debug >= 1 {
            say(&format!(
                "rangeStep = {},  rangeSize = {},  numSteps = {}\n",
                range_step, range_size, num_steps
            ));
        }
        if num_steps < 1 {
            return Err("No ranges to fit in the autofit angle range".to_string());
        }
        let num_steps = num_steps as usize;

        // Remove existing entries whose midpoints are in the range
        let min_del_ind = (low_near + range_size / 4).min(n - 1);
        let max_del_ind = high_near.saturating_sub(range_size / 4).min(n - 1);
        let (mut min_del, mut max_del) = (
            sorted[min_del_ind] as f64 - eps,
            sorted[max_del_ind] as f64 + eps,
        );
        if min_del_ind == 0 && max_del_ind == n - 1 {
            let (lo, hi) = self.angles.iter().fold((f64::MAX, f64::MIN), |(l, h), &a| {
                (l.min(a as f64), h.max(a as f64))
            });
            min_del = lo - eps;
            max_del = hi + eps;
        }
        self.saved.retain(|it| {
            let m = (it.h_angle + it.l_angle) / 2.0;
            !(m >= min_del && m <= max_del)
        });

        let interval = (high_mid - low_mid) / (num_steps as i64 - 1).max(1);
        let remainder = (high_mid - low_mid) % (num_steps as i64 - 1).max(1);
        let mut ind_mid = low_mid;
        let mut best_start = 0.0;
        if mid_angle >= min_angle - eps && mid_angle <= max_angle + eps {
            best_start = mid_angle;
        }
        let mut lows = Vec::new();
        let mut highs = Vec::new();
        let mut min_mid = 1000.0f64;
        let mut mid_step = 0usize;
        for step in 0..num_steps {
            let mut hi = ind_mid + (range_size / 2) as i64;
            let mut lo = hi - (range_size as i64 - 1);
            let (lc, hc) = (
                lo.clamp(0, n as i64 - 1) as usize,
                hi.clamp(0, n as i64 - 1) as usize,
            );
            if wide(lc, hc) {
                hi = ind_mid + (self.s.num_more_views / 2) as i64;
                lo = hi - (self.s.num_more_views as i64 - 1);
            }
            let lo = lo.clamp(0, n as i64 - 1) as usize;
            let hi = hi.clamp(0, n as i64 - 1) as usize;
            lows.push(sorted[lo] as f64);
            highs.push(sorted[hi] as f64);
            let m = ((sorted[lo] + sorted[hi]) as f64 / 2.0 - best_start).abs();
            if m < min_mid {
                min_mid = m;
                mid_step = step;
            }
            ind_mid += interval + if (step as i64) < remainder { 1 } else { 0 };
        }

        let save_use_def = self.use_cur_defocus;
        let save_use_phase = self.use_cur_phase;
        let (mut num_loops, mut start_step, mut end_step, mut dir) =
            (1, 0i64, num_steps as i64 - 1, 1i64);
        if (self.use_cur_defocus || self.use_cur_phase) && num_steps > 1 {
            num_loops = 2;
            start_step = mid_step as i64;
            end_step = 0;
            dir = -1;
        }
        let (mut first_def, mut first_phase, mut first_cuton) =
            (self.model.defocus, self.last_phase, self.last_cuton);
        for lp in 0..num_loops {
            let mut step = start_step;
            while dir * step <= dir * end_step {
                let st = step as usize;
                for it in 0..num_iter {
                    self.autofit_iteration = it;
                    self.angle_changed(lows[st], highs[st])?;
                    self.use_cur_defocus = true;
                    self.use_cur_phase = true;
                }
                self.autofit_iteration = 0;
                self.use_cur_defocus = save_use_def;
                self.use_cur_phase = save_use_phase;
                self.save_current();
                if self.s.dump_final {
                    self.dump_count += 1;
                    say(&format!(
                        "===  Nyquist fraction, PS, fit curve for {:.2} to {:.2} (# {}  {} points) ===\n",
                        lows[st], highs[st], self.dump_count, self.s.dim
                    ));
                    for i in 0..self.s.dim {
                        say(&format!(
                            "{}  {:.4}  {:15.7}  {:15.7}\n",
                            self.dump_count,
                            i as f64 / (self.s.dim as f64 - 1.0),
                            self.fitter.raw[i],
                            self.last_curve.get(i).copied().unwrap_or(0.0)
                        ));
                    }
                    say("=============================================\n");
                }
                if lp == 0 && step == start_step {
                    first_def = self.model.defocus;
                    first_phase = self.last_phase;
                    first_cuton = self.last_cuton;
                }
                step += dir;
            }
            start_step = mid_step as i64 + 1;
            end_step = num_steps as i64 - 1;
            dir = 1;
            if save_use_def {
                self.model.defocus = first_def;
            }
            if save_use_phase {
                self.last_phase = first_phase;
                self.last_cuton = first_cuton;
            }
        }
        Ok(())
    }

    /// Scans a range of expected defocus values for the most consistent
    /// fit (`scanDefocusAndCrop` without autotuning of cropping/resolution),
    /// optionally extending the fitting range as autotuning does.
    pub fn scan_defocus(
        &mut self,
        scan_start: f64,
        scan_end: f64,
        tune: bool,
        low_entered: bool,
        high_entered: bool,
        crop_entered: bool,
    ) -> Result<(), String> {
        let dim = self.s.dim;
        let scan_fit_end = 0.6;
        let min_scan_pixel = 0.27;
        let close_frac = 0.05;
        let save_weighting = self.s.weighting;
        let initial_crop = self.crop;
        let initial_crop_pixel = self.crop_pixel;
        self.s.weighting = 1;
        let pix_for_crop = if self.crop && !crop_entered {
            self.crop_pixel
        } else {
            self.s.pixel_size
        };
        let crop_for_scan = (!self.crop || !crop_entered) && pix_for_crop < min_scan_pixel;
        let mut fallback_crop = self.crop_pixel;
        if crop_for_scan {
            let cp = 0.15f64.max((2.0 * pix_for_crop).min(min_scan_pixel));
            self.crop = true;
            self.set_crop_pixel(cp);
            fallback_crop = cp;
        }
        let (x1_save, x2_save) = (self.x1, self.x2);
        self.compute_ps(false)?;
        let save_use_cur = self.use_cur_defocus;
        self.use_cur_defocus = false;
        let orig_exp = self.model.exp_defocus;
        let (scan_start, scan_end) = if scan_start > 0.0 {
            (scan_start, scan_end)
        } else {
            (orig_exp, orig_exp)
        };
        let mut trial = scan_start;
        let mut trials: Vec<(f64, f64, f64)> = Vec::new();
        let debug = self.s.debug;
        while trial <= scan_end {
            self.model.exp_defocus = trial;
            if !low_entered {
                let (_, first, _, back) = self.fit_range_from_defocus(false);
                self.x1 = first - back;
            }
            if !high_entered {
                self.x2 = nint(scan_fit_end * (dim as f64 - 1.0)) as usize;
            }
            self.thread_start_defocus = trial;
            self.fit_spectrum();
            trials.push((trial, self.model.defocus, self.fitter.last_error));
            let inc = 0.125 * trial;
            trial += inc;
            if trial > scan_end && trial - scan_end < 0.67 * inc {
                trial = scan_end;
            }
        }
        self.thread_start_defocus = 0.0;
        self.s.debug = debug;
        let min_ind = (0..trials.len())
            .min_by(|&a, &b| trials[a].2.partial_cmp(&trials[b].2).unwrap())
            .unwrap_or(0);
        if debug >= 1 {
            say("Trial defocus    Found   Error\n");
            for (i, t) in trials.iter().enumerate() {
                say(&format!(
                    "{:9.2}    {:8.2}  {:8.4} {}\n",
                    t.0,
                    t.1,
                    t.2,
                    if i == min_ind { "*" } else { "" }
                ));
            }
        }
        // Order by error after dropping failures, then group consecutive
        // (in error order) values that agree within 5%
        let mut order: Vec<usize> = (0..trials.len()).filter(|&i| trials[i].1 >= 0.01).collect();
        order.sort_by(|&a, &b| trials[a].2.partial_cmp(&trials[b].2).unwrap());
        let found: Vec<f64> = order.iter().map(|&i| trials[i].1).collect();
        let mut groups: Vec<(usize, f64)> = Vec::new();
        let mut gstart = 0usize;
        let (mut max_in_group, mut max_group) = (0usize, 0usize);
        for ind in 1..found.len() {
            let is_close = (gstart..ind).all(|j| {
                (found[ind] - found[j]).abs() < (found[ind] + found[j]) * 0.5 * close_frac
            });
            if !is_close || ind == found.len() - 1 {
                let num = ind - gstart + usize::from(is_close);
                if num > max_in_group {
                    max_in_group = num;
                    max_group = groups.len();
                }
                let def = found[gstart..gstart + num].iter().sum::<f64>() / num as f64;
                groups.push((num, def));
                gstart += num;
            }
        }
        if !found.is_empty() && gstart == found.len() - 1 {
            if max_in_group == 0 {
                max_group = 0;
                max_in_group = 1;
            }
            groups.push((1, found[gstart]));
        }
        let mut best: Option<f64> = None;
        let mut got_big = false;
        if !groups.is_empty() {
            let other = groups
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != max_group)
                .map(|(_, g)| g.0)
                .max()
                .unwrap_or(0);
            if other + 2 <= groups[max_group].0 || other == 0 {
                best = Some(groups[max_group].1);
                got_big = true;
                say(&format!(
                    "Best focus with most ({}) adjacent close values: {:.2}  (biggest other group has {} close values)\n",
                    groups[max_group].0, groups[max_group].1, other
                ));
            }
        }
        if !got_big {
            let (mut num_apart, mut num_close) = (0usize, 0usize);
            for ind in 1..found.len() {
                let is_close = (found[ind] - found[ind - 1]).abs()
                    < (found[ind] + found[ind - 1]) * 0.5 * close_frac;
                if is_close {
                    num_close += 1;
                } else if num_close <= num_apart {
                    num_apart += 1;
                }
                if (!is_close && num_close > num_apart) || ind == found.len() - 1 {
                    best = Some(found[ind - num_close]);
                    say(&format!(
                        "Best focus with enough adjacent close values: {:.2}  ({} close, {} apart with lower error)\n",
                        found[ind - num_close],
                        num_close,
                        num_apart
                    ));
                    break;
                }
            }
        }
        let best_focus = best.unwrap_or(orig_exp);
        self.model.exp_defocus = best_focus;
        let fallback_x2 = self.x2;
        if !low_entered {
            let (_, first, _, back) = self.fit_range_from_defocus(false);
            self.x1 = first - back;
        }

        // Tuning: extend the fitting range while the weighted fit reaches
        // its end (the cropping and resolution changes of native autotuning
        // are not done)
        let mut tuned = false;
        if tune {
            let third = self.model.zero(best_focus, 3, None, None);
            if third >= scan_fit_end - 0.01 && third < 0.66 && !high_entered {
                self.x2 = (0.66 * (dim as f64 - 1.0)).ceil() as usize;
            }
            self.fit_spectrum();
            let mut last_end = 0.0;
            let mut fit_end = 0.0;
            for _ in 0..8 {
                if self.fitter.last_nonzero_freq <= 0.0 {
                    break;
                }
                fit_end = (self.fitter.last_nonzero_freq + 0.02).min(0.92);
                if fit_end <= self.x2 as f64 / (dim as f64 - 1.0) - 0.02
                    || fit_end >= 0.92 - 0.001
                    || (fit_end - last_end).abs() < 0.3 / dim as f64
                {
                    break;
                }
                let next = (fit_end + 0.1).min(0.92);
                self.x2 = nint(next * (dim as f64 - 1.0)) as usize;
                self.fit_spectrum();
                last_end = fit_end;
            }
            if fit_end > 0.0 && self.fitter.last_nonzero_freq > 0.0 {
                self.x2 = nint(fit_end * (dim as f64 - 1.0)) as usize;
                let ratio = self.model.defocus / best_focus;
                if ratio.max(1.0 / ratio) <= 1.1 {
                    tuned = true;
                    say(&format!(
                        "Final settings from tuning: fitting to {:.3}/pixel,  cropping to {:.3} nm, PS resolution {}\n",
                        0.5 * self.x2 as f64 / (dim as f64 - 1.0),
                        self.crop_pixel,
                        dim
                    ));
                }
            }
            if !tuned {
                say(
                    "WARNING: Autotuning not done: truncation/weighting failed or gave deviant defocus\n",
                );
            }
        }
        self.s.weighting = save_weighting;
        self.use_cur_defocus = save_use_cur;
        if tuned {
            self.s.weighting |= WEIGHT_TRUNCATE;
            self.use_cur_defocus = true;
        }
        if crop_for_scan && !tuned {
            self.crop = initial_crop;
            self.set_crop_pixel(if initial_crop {
                initial_crop_pixel
            } else {
                self.s.pixel_size
            });
        }
        if !tuned {
            let (_, first, sec, back) = self.fit_range_from_defocus(false);
            self.x1 = if low_entered { x1_save } else { first - back };
            self.x2 = if high_entered { x2_save } else { sec };
        }
        // Native refits the spectrum still in memory here, which after the
        // scan's cropping was computed at the cropped pixel size; the
        // resulting deviant defocus is what usually sends it back to the scan
        // settings below.  Kept so the same settings are chosen.
        self.fit_spectrum();
        if !tuned && got_big {
            let r = self.model.defocus / best_focus;
            if r.max(1.0 / r) > 1.15 {
                say(&format!(
                    "WARNING: Defocus value far off after returning to original parameters, falling back to settings\n in defocus scan: fitting to {:.3}/pixel,  cropping to {:.3} nm, PS resolution {}\n",
                    0.5 * fallback_x2 as f64 / (dim as f64 - 1.0),
                    fallback_crop,
                    dim
                ));
                self.model.defocus = best_focus;
                self.crop = fallback_crop > self.s.pixel_size;
                self.set_crop_pixel(if self.crop {
                    fallback_crop
                } else {
                    self.s.pixel_size
                });
                self.x2 = fallback_x2;
                if !low_entered {
                    let (_, first, _, back) = self.fit_range_from_defocus(false);
                    self.x1 = first - back;
                }
                self.compute_ps(false)?;
                self.fit_spectrum();
            }
        }
        Ok(())
    }

    /// Writes the defocus file (`writeDefocusOrTempFile`).
    pub fn write_defocus_file(&self) -> Result<(), String> {
        if self.saved.is_empty() {
            return Err("There are no defocus values to save".into());
        }
        let (mut has_astig, mut has_phase, mut has_cuton) = (false, false, false);
        for it in &self.saved {
            has_astig |= it.defocus2 != 0.0;
            has_phase |= it.plate_phase != 0.0;
            has_cuton |= it.cut_on_freq != 0.0;
        }
        if !has_phase {
            has_phase = self.model.plate_phase != 0.0;
        }
        if !has_cuton && has_phase {
            has_cuton = self.model.cut_on != 0.0;
        }
        imod_backup_file(&self.s.defocus_file);
        let mut out = String::new();
        if has_phase || has_astig {
            let mut flags = if has_astig { DEF_FILE_HAS_ASTIG } else { 0 };
            if has_phase {
                flags += DEF_FILE_HAS_PHASE;
                if has_cuton {
                    flags += DEF_FILE_HAS_CUT_ON;
                }
            }
            out += &format!("{}\t0\t0.0\t0.0\t0.0\t3\n", flags);
            for it in &self.saved {
                let (s, e, la, ha) = (
                    it.starting_slice + 1,
                    it.ending_slice + 1,
                    it.l_angle,
                    it.h_angle,
                );
                let ph = it.plate_phase / RADIANS_PER_DEGREE;
                out += &if has_phase && has_astig && has_cuton {
                    format!(
                        "{}\t{}\t{:5.2}\t{:5.2}\t{:7.1}\t{:7.1}\t{:7.2}\t{:9.2}\t{:9.4}\n",
                        s,
                        e,
                        la,
                        ha,
                        it.defocus * 1000.0,
                        it.defocus2 * 1000.0,
                        it.astig_angle,
                        ph,
                        it.cut_on_freq
                    )
                } else if has_phase && has_astig {
                    format!(
                        "{}\t{}\t{:5.2}\t{:5.2}\t{:7.1}\t{:7.1}\t{:7.2}\t{:9.2}\n",
                        s,
                        e,
                        la,
                        ha,
                        it.defocus * 1000.0,
                        it.defocus2 * 1000.0,
                        it.astig_angle,
                        ph
                    )
                } else if has_phase && has_cuton {
                    format!(
                        "{}\t{}\t{:5.2}\t{:5.2}\t{:7.1}\t{:9.2}\t{:9.4}\n",
                        s,
                        e,
                        la,
                        ha,
                        it.defocus * 1000.0,
                        ph,
                        it.cut_on_freq
                    )
                } else if has_phase {
                    format!(
                        "{}\t{}\t{:5.2}\t{:5.2}\t{:7.1}\t{:9.2}\n",
                        s,
                        e,
                        la,
                        ha,
                        it.defocus * 1000.0,
                        ph
                    )
                } else {
                    format!(
                        "{}\t{}\t{:5.2}\t{:5.2}\t{:7.1}\t{:7.1}\t{:7.2}\n",
                        s,
                        e,
                        la,
                        ha,
                        it.defocus * 1000.0,
                        it.defocus2 * 1000.0,
                        it.astig_angle
                    )
                };
            }
        } else {
            for (i, it) in self.saved.iter().enumerate() {
                out += &format!(
                    "{}\t{}\t{:5.2}\t{:5.2}\t{:6.0}",
                    it.starting_slice + 1,
                    it.ending_slice + 1,
                    it.l_angle,
                    it.h_angle,
                    it.defocus * 1000.0
                );
                out += if i > 0 { "\n" } else { "   2\n" };
            }
        }
        std::fs::write(&self.s.defocus_file, out)
            .map_err(|e| format!("Cannot open output file for saving defocus values: {e}"))
    }

    /// Noise spectrum of one noise image: central-tile average, smoothed.
    pub fn noise_spectrum(
        spectra: &ViewSpectra,
        bin: &Binning,
        dim: usize,
        model: &CtfModel,
    ) -> (Vec<f64>, f64) {
        let mut sum = SpectrumSum::new(dim);
        let ctx = ScaleContext {
            model,
            bin,
            scaling: ZeroScaling { low: 1, high: 0 },
            phase: None,
            cuton: None,
            nps_scale: 0.0,
        };
        let strip = &spectra.strips[0];
        let (mut msum, mut mcount) = (0.0f64, 0usize);
        for sp in strip.iter() {
            msum += sp.mean * sp.count as f64;
            mcount += sp.count;
            scale_and_add(
                &ctx,
                &sp.ps,
                &bin.count,
                sp.count,
                None,
                model.exp_defocus,
                model.exp_defocus,
                &mut sum,
            );
        }
        let mut avg = sum.average();
        // Smooth with local quadratic fits in log space
        let nfit = 9.min(dim);
        let first = 2usize;
        let mut smooth = vec![0f64; dim];
        let mut xx = vec![0f32; nfit];
        let mut yy = vec![0f32; nfit];
        let mut work = vec![0f32; 3 * (5 + nfit) + 16];
        let mut ok = true;
        for ifit in first..dim {
            let mut is = first.max(ifit.saturating_sub(nfit / 2));
            let ie = (dim - 1).min(is + nfit - 1);
            is = ie + 1 - nfit;
            for ind in is..=ie {
                xx[ind - is] = (ind - is) as f32;
                yy[ind - is] = avg[ind].ln() as f32;
            }
            let mut slopes = [0f32; 3];
            let mut intcp = [0f32; 1];
            if polynomial_fit(&xx, &yy, nfit as i32, 2, &mut slopes, &mut intcp, &mut work) != 0 {
                say("WARNING: Computational error smoothing noise power spectrum\n");
                ok = false;
                break;
            }
            let d = (ifit - is) as f64;
            smooth[ifit] = intcp[0] as f64 + slopes[0] as f64 * d + slopes[1] as f64 * d * d;
        }
        if ok {
            for ifit in first..dim {
                avg[ifit] = smooth[ifit].exp();
            }
        }
        (
            avg,
            if mcount > 0 {
                msum / mcount as f64
            } else {
                0.0
            },
        )
    }
}

/// Builds tile parameters for a cache.
pub fn tile_params(s: &RunSettings, raw: usize) -> TileParams {
    TileParams {
        tile: s.tile,
        raw,
        def_tol: s.def_tol,
        left_tol: s.left_tol,
        right_tol: s.right_tol,
        axis_angle: s.axis_angle,
        tilt_offset: s.tilt_offset,
        pixel_size: s.pixel_size,
    }
}
