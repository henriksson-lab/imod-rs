//! CTF physics for the `ctfplotter` reimplementation: zero positions, CTF
//! values and the inverse relations, all with frequency expressed as a
//! fraction of Nyquist of the (possibly cropped) spectrum and defocus in
//! microns, underfocus positive.
//!
//! The formulas are the ones `IMOD/ctfplotter/defocusfinder.cpp` uses (they
//! are the standard weak-phase CTF with a fixed amplitude-contrast phase, a
//! phase-plate shift and an optional exponential cut-on of that shift), kept
//! in the same algebraic form so the zero positions agree with native
//! `ctfplotter` to rounding.

use crate::imod::ctfplotter::ctfutils::FREQ_FOR_PHASE;

/// The `MY_PI` constant `ctfplotter` uses (`myapp.h:46`); kept so zero
/// positions agree with the native program to rounding.
pub const MY_PI: f64 = 3.1415926;
/// `b3dutil.h`'s `RADIANS_PER_DEGREE`.
pub const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// Microscope and CTF state: the constants derived from voltage, Cs and
/// amplitude contrast, the effective pixel size, and the current expected
/// and fitted defocus, phase shift and cut-on frequency.
#[derive(Clone, Debug)]
pub struct CtfModel {
    /// Effective pixel size in nm (the cropped size when the spectrum is
    /// cropped).
    pub pixel_size: f64,
    wavelength: f64,
    cs_one: f64,
    cs_two: f64,
    amp_angle: f64,
    amp_phase_factor: f64,
    /// Expected defocus in microns.
    pub exp_defocus: f64,
    /// Current (last fitted) defocus in microns; negative when unset.
    pub defocus: f64,
    /// Fixed phase-plate shift in radians.
    pub plate_phase: f64,
    /// Fixed cut-on frequency in 1/nm (0 for none).
    pub cut_on: f64,
}

impl CtfModel {
    pub fn new(voltage: i32, pixel_size: f64, amp_contrast: f64, cs: f64, exp_def_nm: f64) -> Self {
        let volt = voltage as f64;
        let wavelength = 1.23984 / (volt * (volt + 1022.0)).sqrt();
        let root = (1.0 - amp_contrast * amp_contrast).sqrt();
        CtfModel {
            pixel_size,
            wavelength,
            cs_one: (cs * wavelength).sqrt(),
            cs_two: (1_000_000.0 * cs / wavelength).sqrt().sqrt(),
            amp_angle: 2.0 * (amp_contrast / root).atan() / MY_PI,
            amp_phase_factor: (-amp_contrast / root).atan(),
            exp_defocus: exp_def_nm / 1000.0,
            defocus: -1000.0,
            plate_phase: 0.0,
            cut_on: 0.0,
        }
    }

    fn phase_or_plate(&self, phase: Option<f64>) -> f64 {
        phase.unwrap_or(self.plate_phase)
    }

    fn cuton_or_fixed(&self, cuton: Option<f64>) -> f64 {
        cuton.unwrap_or(self.cut_on)
    }

    fn one_zero_raw(&self, focus: f64, zero_num: i32, phase: f64) -> f64 {
        let delz = focus / self.cs_one;
        let inner =
            (delz * delz + self.amp_angle + 2.0 * phase / MY_PI - 2.0 * zero_num as f64).max(0.0);
        let theta = (delz - inner.sqrt()).sqrt();
        theta * self.pixel_size * 2.0 / (self.wavelength * self.cs_two)
    }

    /// Frequency (fraction of Nyquist) of zero number `zero_num` (from 1) at
    /// `focus` microns.  With a cut-on frequency the effective phase depends
    /// on frequency and the zero is found by fixed-point iteration.
    pub fn zero(&self, focus: f64, zero_num: i32, phase: Option<f64>, cuton: Option<f64>) -> f64 {
        let phase = self.phase_or_plate(phase);
        let cuton = self.cuton_or_fixed(cuton);
        let mut zero = self.one_zero_raw(focus, zero_num, phase);
        if phase != 0.0 && cuton != 0.0 {
            let frac_at_ref = 1.0 / (1.0 - (-FREQ_FOR_PHASE / cuton).exp());
            let mut last_zero = -1.0;
            let mut last_diff = 0.0;
            let mut iter = 0;
            while iter < 10 && (last_zero - zero).abs() > 1.0e-4 {
                last_zero = zero;
                zero = self.one_zero_raw(
                    focus,
                    zero_num,
                    phase * frac_at_ref * (1.0 - (-zero / (2.0 * self.pixel_size * cuton)).exp()),
                );
                let diff = (zero - last_zero).abs();
                if (iter > 0 && diff > last_diff) || zero.is_nan() {
                    return self.one_zero_raw(focus, zero_num, phase);
                }
                last_diff = diff;
                iter += 1;
            }
        }
        zero
    }

    /// First and second zeros.
    pub fn two_zeros(&self, focus: f64, phase: Option<f64>, cuton: Option<f64>) -> (f64, f64) {
        (
            self.zero(focus, 1, phase, cuton),
            self.zero(focus, 2, phase, cuton),
        )
    }

    /// Number of zeros at or below `nyquist_frac`.
    pub fn zeros_in_range(
        &self,
        focus: f64,
        nyquist_frac: f64,
        phase: Option<f64>,
        cuton: Option<f64>,
    ) -> i32 {
        let mut num = 0;
        while num < 1000 {
            if self.zero(focus, num + 1, phase, cuton) > nyquist_frac {
                break;
            }
            num += 1;
        }
        num
    }

    fn phase_fraction(&self, freq: f64, cuton: f64) -> f64 {
        if cuton > 0.0 {
            (1.0 - (-freq / (2.0 * self.pixel_size * cuton)).exp())
                / (1.0 - (-FREQ_FOR_PHASE / cuton).exp())
        } else {
            1.0
        }
    }

    /// CTF value (-2 sin chi) at `freq` for defocus `def`.
    pub fn ctf_value(&self, freq: f64, def: f64, phase: Option<f64>, cuton: Option<f64>) -> f64 {
        let theta = (freq * self.wavelength * self.cs_two) * 0.5 / self.pixel_size;
        let delz = def / self.cs_one;
        let frac = self.phase_fraction(freq, self.cuton_or_fixed(cuton));
        let phi = 0.5 * MY_PI * (theta.powi(4) - 2.0 * theta * theta * delz)
            + self.amp_phase_factor
            - frac * self.phase_or_plate(phase);
        -2.0 * phi.sin()
    }

    /// The two factors from which the CTF at `freq` can be computed for any
    /// defocus: `phi = fixed + def * vary`.
    pub fn phi_factors(&self, freq: f64, phase: Option<f64>, cuton: Option<f64>) -> (f64, f64) {
        let theta = (freq * self.wavelength * self.cs_two) * 0.5 / self.pixel_size;
        let frac = self.phase_fraction(freq, self.cuton_or_fixed(cuton));
        (
            0.5 * MY_PI * theta.powi(4) + self.amp_phase_factor - frac * self.phase_or_plate(phase),
            -0.5 * MY_PI * 2.0 * theta * theta / self.cs_one,
        )
    }

    /// Largest defocus change around `focus` for which the last zero below
    /// `freq_lim` moves by less than `crit_shift` of the inter-zero spacing.
    pub fn tolerance(
        &self,
        freq_lim: f64,
        crit_shift: f64,
        focus: f64,
        phase: Option<f64>,
        cuton: Option<f64>,
    ) -> f64 {
        let num = self.zeros_in_range(focus, freq_lim, phase, cuton);
        let last = self.zero(focus, num, phase, cuton);
        let inter = self.zero(focus, num + 1, phase, cuton) - last;
        let mut good = [0.005f64, 0.005];
        for good_shift in &mut good {
            for del in 0..1000 {
                let shift = (2 * del - 1) as f64 * 0.005;
                let shifted = self.zero(focus + shift, num, phase, cuton);
                if (shifted - last).abs() > crit_shift * inter {
                    break;
                }
                *good_shift = shift.abs();
            }
        }
        2.0 * good[0].min(good[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctf_vanishes_at_its_zeros_and_inverts_to_the_defocus() {
        let mut m = CtfModel::new(300, 0.2, 0.07, 2.7, 3000.0);
        for (phase, cuton) in [(0.0, 0.0), (1.2, 0.0), (1.2, 0.05)] {
            m.plate_phase = phase;
            m.cut_on = cuton;
            for n in 1..6 {
                let z = m.zero(3.0, n, None, None);
                assert!(z > 0.0 && z < 1.0);
                assert!(
                    m.ctf_value(z, 3.0, None, None).abs() < 1.0e-3,
                    "zero {n} phase {phase} cuton {cuton}"
                );
            }
            // The phi factors reproduce the CTF for any defocus
            let (f, v) = m.phi_factors(0.3, None, None);
            assert!(
                (-2.0 * (f + 2.5 * v).sin() - m.ctf_value(0.3, 2.5, None, None)).abs() < 1.0e-12
            );
        }
        m.plate_phase = 0.0;
        let z = m.zero(2.0, 1, None, None);
        assert_eq!(m.zeros_in_range(2.0, z + 1.0e-9, None, None), 1);
    }
}
