//! Fitting a CTF-like curve to a background-subtracted log power spectrum,
//! and fitting the baseline that is subtracted.
//!
//! The model and its penalties are those of native `ctfplotter`
//! (`simplexfitting.cpp`, `fitCTF`/`funkCTF`/`makeWeights`/`fitBaseline`),
//! ported closely because the owner's goal for this reimplementation is
//! results close to native: the curve is
//! `base + scale * exp(-decay * dx) * |CTF|^power` (with a second
//! exponential term when fitting past about 1.5 inter-zero intervals),
//! minimized by the translated `dualAmoeba` simplex search, with penalty
//! factors that keep the zeros near the expected ones, the power near 1-2,
//! and scales, decays and defocus positive.  Optional correlation-based
//! weighting truncates the fit where the spectrum stops following the curve.

use crate::imod::libcfshr::amoeba::dual_amoeba;
use crate::imod::libcfshr::regression::weighted_poly_fit;
use crate::imod::libcfshr::simplestat::ls_fit;

use super::ctf::{CtfModel, RADIANS_PER_DEGREE};

pub const NO_PHASE: f64 = -9999.0;
pub const PHASE_TEST: f64 = -9990.0;
pub const WEIGHT_TRUNCATE: i32 = 1;
pub const WEIGHT_NORMALIZE: i32 = 2;
const VARNUM: usize = 9;

pub fn opt_phase(v: f64) -> Option<f64> {
    (v > PHASE_TEST).then_some(v)
}

fn nint(v: f64) -> i64 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i64
}

/// Starting defocus and the expected first and second zeros for a fit.
#[derive(Clone, Copy, Debug)]
pub struct FitStart {
    pub defocus: f64,
    pub zero1: f64,
    pub zero2: f64,
}

/// Result of one CTF fit.
#[derive(Clone, Debug)]
pub struct CtfFitResult {
    pub error: f64,
    pub focus: f64,
    pub phase: Option<f64>,
    pub cuton: Option<f64>,
    pub curve: Vec<f64>,
}

/// State of the CTF-like curve fitter; parameters persist between fits
/// as they do natively, because restricted refits reuse them.
#[derive(Clone, Debug)]
pub struct CtfFitter {
    pub dim: usize,
    pub index1: usize,
    pub index2: usize,
    /// Background-subtracted log spectrum being fit.
    pub raw: Vec<f64>,
    delx: Vec<f64>,
    phi_fixed: Vec<f64>,
    phi_vary: Vec<f64>,
    corr_weight: Vec<f64>,
    use_weights: bool,
    a: [f32; 10],
    power_ind: usize,
    fit_phase: bool,
    phase_inc: usize,
    num_var: usize,
    phase_for_fit: f64,
    cuton_for_fit: f64,
    exp_zero: f64,
    exp_second_zero: f64,
    min_def: f32,
    max_def: f32,
    min_phase: f32,
    max_phase: f32,
    max_cuton: f32,
    /// Allowed defocus range around the start (0 for unrestricted).
    pub defocus_range: f64,
    last_num_vars: usize,
    pub last_error: f64,
    /// Highest frequency with nonzero weight after weighting.
    pub last_nonzero_freq: f64,
    num_nonzero: usize,
    first_amplitude: f32,
    first_max: f32,
}

/// Immutable view of the fitter used while evaluating the objective.
struct Eval<'a> {
    f: &'a CtfFitter,
    m: &'a CtfModel,
    tail: [f32; 10],
}

impl CtfFitter {
    pub fn new(dim: usize) -> Self {
        CtfFitter {
            dim,
            index1: 0,
            index2: 0,
            raw: vec![0.0; dim],
            delx: vec![0.0; dim],
            phi_fixed: vec![0.0; dim],
            phi_vary: vec![0.0; dim],
            corr_weight: vec![1.0; dim],
            use_weights: false,
            a: [0.0; 10],
            power_ind: 4,
            fit_phase: false,
            phase_inc: 0,
            num_var: 0,
            phase_for_fit: NO_PHASE,
            cuton_for_fit: NO_PHASE,
            exp_zero: 0.0,
            exp_second_zero: 0.0,
            min_def: 0.0,
            max_def: 0.0,
            min_phase: 0.0,
            max_phase: 0.0,
            max_cuton: 0.0,
            defocus_range: 0.0,
            last_num_vars: 0,
            last_error: 0.0,
            last_nonzero_freq: 0.0,
            num_nonzero: 0,
            first_amplitude: 0.0,
            first_max: 0.0,
        }
    }

    pub fn set_range(&mut self, i1: usize, i2: usize) {
        self.index1 = i1;
        self.index2 = i2.min(self.dim - 1);
    }

    fn ctf_at_index(&self, m: &CtfModel, ind: usize, p: &[f32]) -> f64 {
        let ph = self.phase_inc;
        let (ctf, delx) = if self.fit_phase {
            let x = ind as f64 / (self.dim as f64 - 1.0);
            let delx = (ind as f64 - self.index1 as f64) / (self.dim as f64 - 1.0);
            (
                m.ctf_value(
                    x,
                    p[0] as f64,
                    opt_phase(p[1] as f64),
                    opt_phase(p[2] as f64),
                ),
                delx,
            )
        } else {
            (
                -2.0 * (self.phi_fixed[ind] + p[0] as f64 * self.phi_vary[ind]).sin(),
                self.delx[ind],
            )
        };
        if self.power_ind == 4 + ph {
            p[1 + ph] as f64
                + p[2 + ph] as f64
                    * (-(p[3 + ph] as f64) * delx).exp()
                    * ctf.abs().powf(p[4 + ph] as f64)
        } else {
            p[1 + ph] as f64
                + (p[2 + ph] as f64 * (-(p[3 + ph] as f64) * delx).exp()
                    + p[4 + ph] as f64 * (-(p[5 + ph] as f64) * delx).exp())
                    * ctf.abs().powf(p[6 + ph] as f64)
        }
    }

    /// The fitted curve at every index for the current parameters.
    fn curve(&self, m: &CtfModel) -> Vec<f64> {
        (0..self.dim)
            .map(|i| self.ctf_at_index(m, i, &self.a))
            .collect()
    }

    fn funk(ev: &Eval, param: &[f32]) -> f32 {
        let f = ev.f;
        let mut par = [0f32; 10];
        let mut use_inc = 0;
        if f.min_def > 0.0 && (f.max_def - f.min_def).abs() < 1.0e-6 {
            par[0] = f.min_def;
            use_inc = 1;
        }
        for i in use_inc..f.num_var {
            par[i] = param[i - use_inc];
        }
        par[f.num_var..VARNUM].copy_from_slice(&ev.tail[f.num_var..VARNUM]);
        let mut keep = [0usize, 2, 3, 4, 5];
        let num_keep = if f.power_ind == 4 + f.phase_inc { 3 } else { 5 };
        if f.fit_phase {
            for k in keep.iter_mut().take(num_keep).skip(1) {
                *k += f.phase_inc;
            }
        }
        let mut neg = false;
        if par[0] < 0.0 {
            par[0] = 0.01;
            neg = true;
        }
        let mut err = 0.0f64;
        for i in f.index1..=f.index2 {
            let y = f.ctf_at_index(ev.m, i, &par) - f.raw[i];
            err += if f.use_weights {
                y * y * f.corr_weight[i]
            } else {
                y * y
            };
        }
        if neg {
            err *= 5.0;
        }
        let pw = par[f.power_ind] as f64;
        if !(0.5..=2.0).contains(&pw) {
            let x = (pw - 2.0).max(0.5 - pw);
            err *= 5.0 * (1.0 + 5.0 * x);
        }
        let (z1, z2) = ev.m.two_zeros(
            par[0] as f64,
            opt_phase(f.phase_for_fit),
            opt_phase(f.cuton_for_fit),
        );
        let (ez, ez2) = (f.exp_zero, f.exp_second_zero);
        if (ez - z2).abs() < 0.33 * (ez - z1).abs() {
            err *= 5.0 * (3.0 - (ez - z2).abs() / ez);
        } else if ez > z2 {
            err *= 5.0 * (1.0 + 5.0 * (ez - z2));
        } else if (z1 - ez2).abs() < 0.33 * (z1 - ez).abs() {
            err *= 5.0 * (3.0 - (z1 - ez2).abs() / z1);
        } else if z1 > ez2 {
            err *= 5.0 * (1.0 + 5.0 * (z1 - ez2));
        }
        if z1 > 0.9 {
            err *= 5.0 * (1.0 + 5.0 * (z1 - 0.9));
        }
        if use_inc == 0 && f.max_def > 0.0 {
            if par[0] > f.max_def {
                err *= 5.0 * (1.0 + (par[0] - f.max_def) as f64);
            }
            if par[0] < f.min_def {
                err *= 5.0 * (1.0 + (f.min_def - par[0]) as f64);
            }
        }
        for &k in keep.iter().take(num_keep) {
            if par[k] <= 1.0e-5 {
                err *= 5.0 * (1.0 - 5.0 * par[k] as f64);
            }
        }
        if f.fit_phase && (par[1] < f.min_phase || par[1] > f.max_phase) {
            err *= 5.0;
        }
        if f.fit_phase && f.num_var > 2 && (par[2] < 0.0 || par[2] > f.max_cuton) {
            err *= 5.0;
        }
        err as f32
    }

    fn run_amoeba(&mut self, m: &CtfModel, nvar: usize, da: &[f32; 10]) {
        let mut yy = [0f32; VARNUM + 1];
        let mut iter = 0;
        let mut start = self.a;
        {
            let ev = Eval {
                f: self,
                m,
                tail: self.a,
            };
            let mut func = |p: &[f32]| Self::funk(&ev, p);
            dual_amoeba(
                &mut yy,
                nvar,
                2.0,
                &[5.0e-4, 1.0e-5],
                &[5.0e-4, 1.0e-5],
                &mut start,
                da,
                &mut func,
                &mut iter,
            );
        }
        self.a[..nvar].copy_from_slice(&start[..nvar]);
    }

    fn eval_current(&self, m: &CtfModel, nvar: usize) -> f32 {
        let ev = Eval {
            f: self,
            m,
            tail: self.a,
        };
        Self::funk(&ev, &self.a[..nvar])
    }

    /// Fits the CTF-like curve with `nvar` variables (4 or 5 for one
    /// exponential without/with power, 6 or 7 for two; 2 or 3 to fit only
    /// defocus and phase, and cut-on, with the rest from the last fit).
    #[allow(clippy::too_many_arguments)]
    pub fn fit_ctf(
        &mut self,
        m: &CtfModel,
        nvar_in: usize,
        initial_phase: f64,
        initial_cuton: f64,
        fixed_focus: f64,
        start: FitStart,
        max_cuton: f64,
        phase_range: f64,
        weighting: i32,
    ) -> CtfFitResult {
        let mut nvar = nvar_in;
        let mut da = [2.0f32, 2.0, 2.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let last_fit_phase = self.fit_phase;
        self.phase_inc = 0;
        self.num_nonzero = self.index2 + 1 - self.index1;
        self.phase_for_fit = initial_phase;
        self.cuton_for_fit = initial_cuton;
        let mut start_def = start.defocus;
        self.exp_zero = start.zero1;
        self.exp_second_zero = start.zero2;
        if fixed_focus > 0.0 {
            start_def = fixed_focus;
            let (z1, z2) = m.two_zeros(
                fixed_focus,
                opt_phase(initial_phase),
                opt_phase(initial_cuton),
            );
            self.exp_zero = z1;
            self.exp_second_zero = z2;
            self.min_def = start_def as f32;
            self.max_def = start_def as f32;
        } else if self.defocus_range > 0.0 {
            self.min_def = (start_def - self.defocus_range / 2.0) as f32;
            self.max_def = (start_def + self.defocus_range / 2.0) as f32;
        }

        self.fit_phase = initial_phase > PHASE_TEST && nvar < 4;
        if self.fit_phase {
            self.phase_inc = 2;
            if !last_fit_phase {
                for i in (2..VARNUM).rev() {
                    self.a[i] = self.a[i - 2];
                }
                self.power_ind += 2;
            }
            self.a[1] = initial_phase as f32;
            da[1] = 0.1;
            self.a[2] = initial_cuton as f32;
            self.max_cuton = max_cuton as f32;
            da[2] = self.max_cuton / 4.0;
            self.min_phase = (initial_phase - phase_range / 2.0) as f32;
            self.max_phase = (initial_phase + phase_range / 2.0) as f32;
        }
        self.num_var = nvar;
        let ph = self.phase_inc;

        // First maximum after the start determines the initial scale
        let (mut raw_min, mut raw_max) = (1.0e30f64, -1.0e30f64);
        let mut first_max: Option<usize> = None;
        for i in self.index1..=self.index2 {
            raw_min = raw_min.min(self.raw[i]);
            raw_max = raw_max.max(self.raw[i]);
            if first_max.is_none()
                && self.raw[i] > self.raw[i.saturating_sub(1)]
                && self.raw[i] > self.raw[(i + 1).min(self.index2)]
            {
                first_max = Some(i);
            }
        }
        let mut ind_for_scale = self.index1;
        let scale_denom: f32 = match first_max {
            Some(fm) if self.raw[fm] > self.raw[self.index1] => {
                ind_for_scale = fm;
                2.0
            }
            _ => {
                let v = m
                    .ctf_value(
                        self.index1 as f64 / (self.dim as f64 - 1.0),
                        start_def,
                        opt_phase(initial_phase),
                        opt_phase(initial_cuton),
                    )
                    .abs() as f32;
                v.max(0.1)
            }
        };
        self.a[0] = start_def as f32;
        if nvar > 1 + ph {
            self.a[1 + ph] = (raw_min - 0.1 * (raw_max - raw_min)) as f32;
            da[1 + ph] = (0.1 * (raw_max - raw_min)) as f32;
        }
        if nvar > 2 + ph {
            self.a[2 + ph] = ((self.raw[ind_for_scale] - raw_min) as f32) / scale_denom;
            da[2 + ph] = 0.2 * self.a[2 + ph];
        }
        if nvar > 3 + ph {
            self.a[3 + ph] = 10.0;
            self.power_ind = 4 + ph;
            if nvar > 5 + ph {
                self.power_ind = 6 + ph;
                self.a[4 + ph] = self.a[2 + ph] / 10.0;
                self.a[5 + ph] = 1.0;
                da[4 + ph] = 0.2 * self.a[4 + ph];
                da[5 + ph] = 0.2;
            }
            if !(self.fit_phase && !last_fit_phase && self.last_num_vars == nvar) {
                self.a[self.power_ind] = 1.5;
                da[self.power_ind] = 0.1;
            }
        }
        self.last_num_vars = nvar;

        if fixed_focus > 0.0 {
            for i in 1..nvar {
                self.a[i - 1] = self.a[i];
                da[i - 1] = da[i];
            }
            nvar -= 1;
        }

        if !self.fit_phase {
            for i in 0..self.dim {
                self.delx[i] = (i as f64 - self.index1 as f64) / (self.dim as f64 - 1.0);
                let (pf, pv) = m.phi_factors(
                    i as f64 / (self.dim as f64 - 1.0),
                    opt_phase(initial_phase),
                    opt_phase(initial_cuton),
                );
                self.phi_fixed[i] = pf;
                self.phi_vary[i] = pv;
            }
        }

        if !((weighting & WEIGHT_TRUNCATE) != 0 && (self.fit_phase && self.num_var <= 3)) {
            self.use_weights = false;
            self.last_nonzero_freq = 0.0;
        }

        self.run_amoeba(m, nvar, &da);
        let mut errmin = self.eval_current(m, nvar);
        let roll_up = |a: &mut [f32; 10], nvar: usize| {
            for i in (1..=nvar).rev() {
                a[i] = a[i - 1];
            }
            a[0] = fixed_focus as f32;
        };
        if fixed_focus > 0.0 {
            roll_up(&mut self.a, nvar);
        }

        if (weighting & WEIGHT_TRUNCATE) != 0
            && !(self.fit_phase && self.num_var <= 3)
            && self.a[0] > 0.0
            && !self.a[0].is_nan()
        {
            let fitted = self.curve(m);
            if self.make_weights(
                m,
                self.a[0] as f64,
                &fitted,
                (weighting & WEIGHT_NORMALIZE) != 0,
            ) {
                if fixed_focus > 0.0 {
                    for i in 1..=nvar {
                        self.a[i - 1] = self.a[i];
                    }
                }
                self.use_weights = true;
                let first_focus = self.a[0];
                self.run_amoeba(m, nvar, &da);
                if fixed_focus <= 0.0 && ((self.a[0] - first_focus) / first_focus).abs() > 0.1 {
                    let fitted = self.curve(m);
                    if self.make_weights(
                        m,
                        self.a[0] as f64,
                        &fitted,
                        (weighting & WEIGHT_NORMALIZE) != 0,
                    ) {
                        self.run_amoeba(m, nvar, &da);
                    }
                }
                errmin = self.eval_current(m, nvar);
                if fixed_focus > 0.0 {
                    roll_up(&mut self.a, nvar);
                }
            }
        }

        let err = (errmin as f64 / self.num_nonzero as f64).sqrt();
        let mut phase = None;
        let mut cuton = None;
        if self.fit_phase {
            phase = Some(self.a[1] as f64);
            if self.num_var == 3 {
                cuton = Some(self.a[2] as f64);
            }
        }
        let curve = self.curve(m);
        self.last_error = err;
        self.min_def = 0.0;
        self.max_def = 0.0;
        CtfFitResult {
            error: err,
            focus: self.a[0] as f64,
            phase,
            cuton,
            curve,
        }
    }

    /// Current fit parameters in a printable form.
    pub fn describe(&self, err: f64) -> String {
        let a = &self.a;
        let ph = self.phase_inc;
        let mut s = format!("Fit error={:.6}\t def={:.6}", err, a[0]);
        if self.fit_phase {
            s += &format!("\t pha={:.1}", a[1] as f64 / RADIANS_PER_DEGREE);
            if self.num_var == 3 {
                s += &format!("\t cuton={:.4}", a[2]);
            }
        }
        s += &format!(
            "\t base={:.6}\t scale={}\t decay={:.6}",
            a[1 + ph],
            a[2 + ph],
            a[3 + ph]
        );
        if self.power_ind == 4 + ph {
            s += &format!("\t pow={:.6}", a[4 + ph]);
        } else {
            s += &format!(
                "\t pow={:.6}\t scale2={}\t decay2={:.6}",
                a[6 + ph],
                a[4 + ph],
                a[5 + ph]
            );
        }
        s
    }

    /// Local correlation between spectrum and fitted curve, used to weight
    /// and truncate the fit.  Returns false when there are too few zeros.
    fn make_weights(
        &mut self,
        m: &CtfModel,
        defocus: f64,
        fitted: &[f64],
        normalize: bool,
    ) -> bool {
        let dim = self.dim;
        let (max_freq, thresh) = (0.98f32, 0.4f32);
        let (min_fit, min_zeros_in_range, num_after, max_rescue) = (9usize, 3usize, 2usize, 8usize);
        let high_frac_map_to_one = 0.75f32;
        let mut low_thresh_for_zero = 0.1f32;
        let mut first_corr = 1.0f32;
        let (mut num_below, mut num_rescue, mut gave_up) = (0usize, 0usize, 0usize);
        let max_weight = 2.0f32;
        let mut ampl_wgt = 1.0f32;
        let initial_ampl_frac = 0.5f32;
        self.first_amplitude = 0.0;
        let mut ps_vals = vec![0f32; dim];
        let mut fit_vals = vec![0f32; dim];
        self.corr_weight.iter_mut().for_each(|w| *w = 1.0);

        let mut zeros: Vec<f32> = Vec::new();
        let mut num_in_range = 0usize;
        for num in 1..dim as i32 {
            let zero = (m.zero(
                defocus,
                num,
                opt_phase(self.phase_for_fit),
                opt_phase(self.cuton_for_fit),
            ) * (dim as f64 - 1.0)) as f32;
            if zero > dim as f32 {
                break;
            }
            if zero <= self.index2 as f32 {
                num_in_range += 1;
            }
            zeros.push(zero);
        }
        if num_in_range < min_zeros_in_range {
            return false;
        }
        let base_num = zeros[0] as usize;
        let mut cur_zero = 0usize;
        let mut ind_wgt: Option<usize> = None;
        let mut interval = 0f32;
        let (mut aa, mut bb, mut corr) = (0f32, 0f32, 0f32);
        for num in base_num..dim {
            if cur_zero + 2 < zeros.len() {
                interval = (zeros[cur_zero + 2] - zeros[cur_zero]) / 2.0;
            } else if cur_zero + 1 < zeros.len() {
                interval = zeros[cur_zero + 1] - zeros[cur_zero];
            }
            let num_fit = min_fit.max(nint(2.0 * interval as f64) as usize);
            if (num + num_fit) as f32 > max_freq * (dim as f32 - 1.0)
                || (num_in_range + num_after < zeros.len()
                    && (num + num_fit) as f32 > zeros[num_in_range + num_after])
            {
                break;
            }
            let (mut ps_min, mut ps_max) = (1.0e10f32, -1.0e10f32);
            for ii in 0..num_fit {
                ps_vals[ii] = self.raw[ii + num] as f32;
                fit_vals[ii] = fitted[ii + num] as f32;
                ps_min = ps_min.min(ps_vals[ii]);
                ps_max = ps_max.max(ps_vals[ii]);
            }
            ls_fit(
                &ps_vals,
                &fit_vals,
                num_fit as i32,
                &mut aa,
                &mut bb,
                &mut corr,
            );
            if num as f32 >= zeros[0] && self.first_amplitude == 0.0 {
                self.first_amplitude = ps_max - ps_min;
                self.first_max = ps_max;
            }
            if self.first_amplitude != 0.0 && ps_max > ps_min && normalize {
                ampl_wgt = max_weight.min(self.first_amplitude / (ps_max - ps_min));
            }
            if num as f32 > zeros[cur_zero] {
                cur_zero += 1;
            }
            if corr < thresh * first_corr {
                num_below += 1;
                num_rescue = 0;
                if num_below > max_rescue && gave_up == 0 {
                    gave_up = num;
                }
            } else if gave_up == 0 && num_below > 0 && num_below <= max_rescue {
                num_rescue += 1;
                if num_rescue > num_below {
                    num_below = 0;
                }
            }
            let iw = (num + num_fit / 2).min(dim - 1);
            ind_wgt = Some(iw);
            if num == base_num {
                first_corr = corr;
                self.corr_weight[iw] = ampl_wgt as f64;
                low_thresh_for_zero *= first_corr;
            } else if corr < low_thresh_for_zero || gave_up > 0 {
                self.corr_weight[iw] = 0.0;
            } else if corr >= high_frac_map_to_one * first_corr {
                self.corr_weight[iw] = ampl_wgt as f64;
            } else {
                self.corr_weight[iw] = (ampl_wgt
                    * (low_thresh_for_zero
                        + (corr - low_thresh_for_zero) * (1.0 - low_thresh_for_zero)
                            / (high_frac_map_to_one * first_corr - low_thresh_for_zero)))
                    as f64;
            }
        }
        match ind_wgt {
            None => {
                self.corr_weight.iter_mut().for_each(|w| *w = 0.0);
            }
            Some(iw) => {
                let last = self.corr_weight[iw];
                for w in self.corr_weight.iter_mut().skip(iw + 1) {
                    *w = if gave_up > 0 { 0.0 } else { last };
                }
            }
        }
        if self.first_amplitude > 0.0 {
            for ii in self.index1..base_num.min(dim) {
                if self.raw[ii] as f32 <= self.first_max + initial_ampl_frac * self.first_amplitude
                {
                    break;
                }
                self.corr_weight[ii] = 0.0;
            }
        }
        self.num_nonzero = 0;
        let mut wsum = 0.0f64;
        for ii in self.index1..=self.index2 {
            if self.corr_weight[ii] > 0.0 {
                self.num_nonzero += 1;
                wsum += self.corr_weight[ii];
                self.last_nonzero_freq = ii as f64 / (dim as f64 - 1.0);
            }
        }
        if wsum > 0.0 {
            let fac = self.num_nonzero as f64 / wsum;
            self.corr_weight.iter_mut().for_each(|w| *w *= fac);
        }
        if self.num_nonzero == 0 {
            self.num_nonzero = 1;
        }
        true
    }
}

/// Fits the line after the best break in slope near the first zero; used as
/// the background when there are no noise files.
pub fn baseline_past_break(average: &[f64], zero1: f64) -> Vec<f64> {
    let dim = average.len();
    let xx: Vec<f32> = (0..dim)
        .map(|i| (i as f64 / (dim as f64 - 1.0)) as f32)
        .collect();
    let yy: Vec<f32> = average.iter().map(|&v| v as f32).collect();
    let num_left = 4usize;
    let (fit_start, fit_end) = (3usize, dim - 4);
    let search_start = (nint(0.67 * zero1 * (dim as f64 - 1.0)) as usize).max(fit_start + 3);
    let search_end = (nint(1.33 * zero1 * (dim as f64 - 1.0)).max(0) as usize).min(fit_end - 3);
    let (mut best_slope, mut best_intcp, mut max_diff) = (0f32, 0f32, f32::MIN);
    let (mut sl, mut il, mut sr, mut ir, mut ro) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for center in search_start..=search_end.max(search_start) {
        let start = fit_start.max((center + 1).saturating_sub(num_left));
        ls_fit(
            &xx[start..],
            &yy[start..],
            (center + 1 - start) as i32,
            &mut sl,
            &mut il,
            &mut ro,
        );
        ls_fit(
            &xx[center..],
            &yy[center..],
            (fit_end + 1 - center) as i32,
            &mut sr,
            &mut ir,
            &mut ro,
        );
        let diff = (sr.atan() as f64 - sl.atan() as f64) / RADIANS_PER_DEGREE;
        if center == search_start || diff as f32 > max_diff {
            max_diff = diff as f32;
            best_slope = sr;
            best_intcp = ir;
        }
    }
    xx.iter()
        .map(|&x| (best_slope * x + best_intcp) as f64)
        .collect()
}

/// Fits a smooth polynomial baseline through minima of the smoothed log
/// spectrum (penalizing inflections inside the fitted interval, as native
/// does) and adds it to `baseline` (or replaces it when `add` is false).
/// `index1` is the start of the fitting range, `zero1`/`zero2` and
/// `all_zeros` the expected zeros.
pub fn fit_baseline(
    average: &[f64],
    baseline: &mut [f64],
    order_in: usize,
    add: bool,
    index1: usize,
    all_zeros: &[f64],
) {
    let dim = average.len();
    let mut order = order_in;
    if !add {
        baseline.iter_mut().for_each(|b| *b = 0.0);
    }
    if order == 0 {
        let (s, e) = (dim / 2, (0.95 * dim as f64) as usize);
        let err: f64 = (s..e)
            .map(|i| (average[i] - baseline[i]) / (e - s) as f64)
            .sum();
        baseline.iter_mut().for_each(|b| *b += err);
        return;
    }
    let box_size = 5usize;
    let left = box_size / 2;
    let x0 = all_zeros[0];
    let mut sigma = 0.1 * (all_zeros[1] - x0) * (dim as f64 - 1.0);
    sigma = sigma.clamp(1.0, 2.5);
    let mut wts = [0f32; 5];
    let mut wsum = 0f32;
    for (i, w) in wts.iter_mut().enumerate() {
        *w = (-0.5 * ((i as f64 - left as f64) / sigma).powi(2)).exp() as f32;
        wsum += *w;
    }
    wts.iter_mut().for_each(|w| *w /= wsum);
    let box_avg: Vec<f64> = (0..dim)
        .map(|b| {
            (0..box_size)
                .map(|k| {
                    let ind = (b as i64 + k as i64 - left as i64).clamp(0, dim as i64 - 1) as usize;
                    wts[k] as f64 * (average[ind] - baseline[ind])
                })
                .sum()
        })
        .collect();

    let mut base_freq: Vec<f32> = Vec::new();
    let mut base_avg: Vec<f32> = Vec::new();
    let (mut raw_min, mut raw_max) = (1.0e30f64, -1.0e30f64);
    let mut ind_last = dim;
    let mut add_point = |ind: usize, freq: &mut Vec<f32>, avg: &mut Vec<f32>, last: &mut usize| {
        freq.push((ind as f64 / (dim as f64 - 1.0)) as f32);
        avg.push(box_avg[ind] as f32);
        *last = ind;
        raw_min = raw_min.min(box_avg[ind]);
        raw_max = raw_max.max(box_avg[ind]);
    };

    let back = 3i64.max(nint(0.12 * x0 * (dim as f64 - 1.0)));
    let mut ind_start = ((x0 * (dim as f64 - 1.0)).floor() as i64 - back)
        .max(index1 as i64)
        .max(3) as usize;
    let ind_end = ind_start + nint(0.5 * (dim - ind_start) as f64) as usize;
    let ind_end = ind_end.min(dim - 4);
    let (mut last_good_min, mut num_at_last_good) = (0usize, 0usize);
    let b = &box_avg;
    for lp in 0..2 {
        for ind in ind_start..=ind_end {
            if b[ind] < b[ind - 1] && b[ind] < b[ind + 1] {
                let ok = lp == 1
                    || ((b[ind - 1] < b[ind - 2]
                        && (b[ind + 1] < b[ind + 2] || b[ind + 1] < b[ind + 3]))
                        || (b[ind + 1] < b[ind + 2]
                            && (b[ind - 1] < b[ind - 2] || b[ind - 1] < b[ind.saturating_sub(3)])));
                if ok {
                    add_point(ind, &mut base_freq, &mut base_avg, &mut ind_last);
                    last_good_min = ind_last;
                    num_at_last_good = base_freq.len();
                }
            }
        }
        if !base_freq.is_empty() {
            break;
        }
    }
    if !base_freq.is_empty() {
        ind_start = ind_last.min(ind_end) + 1;
    }

    // Inter-zero distances from here out
    let mut inter_zeros = vec![0f32; dim];
    let mut ind_zero = 0usize;
    let mut check_good_min = true;
    for ind in ind_start..dim - 1 {
        while ind as f64 / (dim as f64 - 1.0) > all_zeros[ind_zero]
            && ind_zero + 2 < all_zeros.len()
        {
            ind_zero += 1;
        }
        inter_zeros[ind] = ((all_zeros[(ind_zero + 1).min(all_zeros.len() - 1)]
            - all_zeros[ind_zero])
            * (dim as f64 - 1.0)) as f32;
        if nint(0.75 * inter_zeros[ind] as f64) > 9 {
            check_good_min = false;
        }
    }
    let tail_interval = 3usize;
    for ind in ind_start..((0.95 * dim as f64) as usize).min(dim - 3) {
        let local_min = b[ind] < b[ind - 1] && b[ind] < b[ind + 1];
        let n = base_freq.len();
        if local_min
            || (n > 0
                && ind - ind_last >= tail_interval
                && (!check_good_min
                    || ind as i64 - last_good_min as i64 >= nint(0.75 * inter_zeros[ind] as f64)))
        {
            if local_min && n > 0 && ind - ind_last < 2 && ind_last != last_good_min {
                base_freq.pop();
                base_avg.pop();
            }
            add_point(ind, &mut base_freq, &mut base_avg, &mut ind_last);
            if local_min && b[ind - 1] < b[ind - 2] && b[ind + 1] < b[ind + 2] {
                let n = base_freq.len();
                if num_at_last_good > 0
                    && ind as i64 - last_good_min as i64 <= nint(1.25 * inter_zeros[ind] as f64)
                    && num_at_last_good + 1 < n
                {
                    base_freq[num_at_last_good] = base_freq[n - 1];
                    base_avg[num_at_last_good] = base_avg[n - 1];
                    base_freq.truncate(num_at_last_good + 1);
                    base_avg.truncate(num_at_last_good + 1);
                } else {
                    last_good_min = ind_last;
                }
                // Native's `else` (followed by commented-out lines) covers only
                // the assignment above; this one always happens
                num_at_last_good = base_freq.len();
            }
        }
    }

    let num_base = base_freq.len();
    order = order.min((num_base / 2).saturating_sub(1));
    let mut wgt = vec![0f32; num_base];
    if order > 0 {
        let mut s = 0f32;
        for ind in 0..num_base {
            wgt[ind] = if ind == 0 {
                (base_freq[1] - base_freq[0]).sqrt()
            } else if ind == num_base - 1 {
                (base_freq[ind] - base_freq[ind - 1]).sqrt()
            } else {
                0.5 * ((base_freq[ind + 1] - base_freq[ind]).sqrt()
                    + (base_freq[ind] - base_freq[ind - 1]).sqrt())
            };
            s += wgt[ind];
        }
        wgt.iter_mut().for_each(|w| *w /= s / num_base as f32);
        if std::env::var_os("IMOD_CTFPLOTTER_DEBUG_PS").is_some() {
            for i in 0..num_base {
                super::analyzer::say(&format!(
                    "1  {:6.3}  {}   {}\n",
                    base_freq[i] / 2.0,
                    base_avg[i],
                    wgt[i]
                ));
            }
        }
    }

    let mut var = [0f32; VARNUM];
    if order > 0 {
        let mut work = vec![0f32; (order + 4) * (num_base + 8) + 64];
        let (v0, vrest) = var.split_at_mut(1);
        if weighted_poly_fit(
            &base_freq,
            &base_avg,
            &wgt,
            num_base as i32,
            order.min(2) as i32,
            vrest,
            v0,
            &mut work,
        ) != 0
        {
            order = 0;
        }
    }
    let base_order = order;
    let funk_base = |p: &[f32]| -> f32 {
        let mut err = 0.0f64;
        for ind in 0..num_base {
            let f = base_freq[ind] as f64;
            let mut y = p[0] as f64 + p[1] as f64 * f + p[2] as f64 * f * f - base_avg[ind] as f64;
            if base_order > 2 {
                y += p[3] as f64 * f * f * f;
            }
            if base_order > 3 {
                y += p[4] as f64 * f * f * f * f;
            }
            err += wgt[ind] as f64 * y * y;
        }
        let (mut delx, mut delx2) = (0.0f64, 0.0f64);
        let (lo, hi) = (base_freq[0] as f64, base_freq[num_base - 1] as f64);
        if base_order < 4 {
            if base_order > 2 && (p[3] as f64).abs() > 1.0e-10 {
                let inflect = p[2] as f64 / (3.0 * p[3] as f64);
                if inflect > lo && inflect < hi {
                    delx = (inflect - lo).min(hi - inflect);
                }
            }
        } else {
            let root = 9.0 * p[3] as f64 - 24.0 * p[2] as f64 * p[4] as f64;
            if root >= 0.0 {
                let inflect = (-3.0 * p[3] as f64 + root.sqrt()) / (6.0 * p[4] as f64);
                if inflect > lo && inflect < hi {
                    delx = (inflect - lo).min(hi - inflect);
                }
                let inflect = (-3.0 * p[3] as f64 - root.sqrt()) / (6.0 * p[4] as f64);
                if inflect > lo && inflect < hi {
                    delx2 = (inflect - lo).min(hi - inflect);
                }
                delx = delx.max(delx2);
            }
        }
        if delx > 0.0 {
            err *= 5.0 * (1.0 + 5.0 * delx);
        }
        err as f32
    };
    if order > 2 {
        let nvar = order + 1;
        let mut da = [0f32; VARNUM];
        da[0] = (0.2 * (raw_max - raw_min)) as f32;
        da[1] = da[0];
        da[2] = da[0];
        da[3] = 0.5 * da[2];
        da[4] = 0.5 * da[2];
        let mut errmin = 1.0e20f32;
        let mut best: Option<[f32; VARNUM]> = None;
        let mut func = funk_base;

        for iter in 0..5 {
            let mut yy = [0f32; VARNUM + 1];
            let mut it = 0;
            dual_amoeba(
                &mut yy,
                nvar,
                2.0,
                &[5.0e-4, 1.0e-5],
                &[5.0e-4, 1.0e-5],
                &mut var,
                &da,
                &mut func,
                &mut it,
            );
            let e = func(&var[..nvar]);

            let saved = var;
            for ind in 0..nvar {
                da[ind] = ((0.2 * (raw_max - raw_min)) as f32).max(0.5 * var[ind]);
                var[ind] *= 0.7 + 0.2 * iter as f32;
            }
            if e < errmin {
                errmin = e;
                best = Some(saved);
            }
        }
        match best {
            None => order = 0,
            Some(v) => var = v,
        }
    }
    if order == 0 {
        return;
    }
    if std::env::var_os("IMOD_CTFPLOTTER_DEBUG_PS").is_some() {
        super::analyzer::say(&format!(
            "Base function fit error={:.6}  const={:.6}  coeff={:.6}  {:.6}  {:.6}  {:.6}\n",
            (funk_base(&var[..order + 1]) as f64).sqrt(),
            var[0],
            var[1],
            var[2],
            var[3],
            var[4]
        ));
    }
    let mut newb = vec![0f64; dim];
    let is = nint(base_freq[0] as f64 * (dim as f64 - 1.0)) as usize;
    let ie = (nint(base_freq[num_base - 1] as f64 * (dim as f64 - 1.0)) as usize).min(dim - 1);
    for (ind, nb) in newb.iter_mut().enumerate().take(ie + 1).skip(is) {
        let f = ind as f64 / (dim as f64 - 1.0);
        *nb = var[0] as f64
            + var[1] as f64 * f
            + var[2] as f64 * f * f
            + var[3] as f64 * f * f * f
            + var[4] as f64 * f * f * f * f;
    }
    for ind in ie + 1..dim {
        newb[ind] = 2.0 * newb[ind - 1] - newb[ind - 2];
    }
    for ind in (0..is).rev() {
        newb[ind] = 2.0 * newb[ind + 1] - newb[ind + 2];
    }
    for (b, n) in baseline.iter_mut().zip(&newb) {
        if add {
            *b += n;
        } else {
            *b = *n;
        }
    }
}

/// Log of the spectrum minus the fitted background, as fit by the CTF
/// fitter.  `all_zeros` are the expected zeros at the starting defocus.
pub fn ps_minus_baseline(
    average: &[f64],
    two_line: bool,
    base_order: usize,
    tile_ratio: f64,
    index1: usize,
    all_zeros: &[f64],
) -> Vec<f64> {
    let dim = average.len();
    let non_zero_min = average
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .fold(1.0e36f64, f64::min);
    let scale = tile_ratio * tile_ratio;
    let mut ps: Vec<f64> = average
        .iter()
        .map(|&v| (scale * v.max(non_zero_min)).ln())
        .collect();
    let mut baseline = vec![0.0f64; dim];
    if two_line {
        baseline = baseline_past_break(&ps, all_zeros[0]);
    }
    fit_baseline(&ps, &mut baseline, base_order, two_line, index1, all_zeros);
    if (baseline[2] - baseline[3]).abs() * (dim as f64 - 1.0) > 0.25 {
        fit_baseline(&ps, &mut baseline, base_order, true, index1, all_zeros);
    }
    if std::env::var_os("IMOD_CTFPLOTTER_DEBUG_PS").is_some() {
        for (i, (p, b)) in ps.iter().zip(&baseline).enumerate() {
            super::analyzer::say(&format!("DBG {} {:.6} {:.6}\n", i, p, b));
        }
    }
    for (p, b) in ps.iter_mut().zip(&baseline) {
        *p -= b;
    }
    ps
}
