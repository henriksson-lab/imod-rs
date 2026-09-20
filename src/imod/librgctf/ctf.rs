//! Translation of `IMOD/librgctf/ctf.{h,cpp}`.
//!
//! Every expression below keeps the source's mixed `float`/`double`
//! evaluation.  `PI` is a `double` macro and the `1.0`, `0.5`, `2.0` and
//! `4.0` literals are `double`s, so most of these formulas are computed in
//! double precision and narrowed only where the source stores or returns a
//! `float`; `sinf`/`cosf`/`sqrtf`/`powf`/`fmodf` calls and their operands stay
//! single precision, exactly as the C++ overload resolution picks them.

use super::defines::PI;

/// C++ `CTF` (`ctf.h:1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ctf {
    spherical_aberration: f32,
    wavelength: f32,
    amplitude_contrast: f32,
    defocus_1: f32,
    defocus_2: f32,
    defocus_half_range: f32,
    astigmatism_azimuth: f32,
    additional_phase_shift: f32,
    // Fitting parameters
    lowest_frequency_for_fitting: f32,
    highest_frequency_for_fitting: f32,
    astigmatism_tolerance: f32,
    // Precomputed terms to make evaluations faster
    precomputed_amplitude_contrast_term: f32,
    squared_wavelength: f32,
    cubed_wavelength: f32,
}

impl Default for Ctf {
    fn default() -> Self {
        Self::new()
    }
}

impl Ctf {
    /// C++ `CTF::CTF()` (`ctf.cpp:4`).
    pub fn new() -> Self {
        Self {
            spherical_aberration: 0.0,
            wavelength: 0.0,
            amplitude_contrast: 0.0,
            defocus_1: 0.0,
            defocus_2: 0.0,
            defocus_half_range: 0.0,
            astigmatism_azimuth: 0.0,
            additional_phase_shift: 0.0,
            lowest_frequency_for_fitting: 0.0,
            highest_frequency_for_fitting: 0.0,
            astigmatism_tolerance: 0.0,
            precomputed_amplitude_contrast_term: 0.0,
            squared_wavelength: 0.0,
            cubed_wavelength: 0.0,
        }
    }

    /// The eleven-argument C++ `CTF::CTF` (`ctf.cpp:24`).
    #[allow(clippy::too_many_arguments)]
    pub fn with_fitting_parameters(
        wanted_acceleration_voltage: f32,
        wanted_spherical_aberration: f32,
        wanted_amplitude_contrast: f32,
        wanted_defocus_1_in_angstroms: f32,
        wanted_defocus_2_in_angstroms: f32,
        wanted_astigmatism_azimuth: f32,
        wanted_lowest_frequency_for_fitting: f32,
        wanted_highest_frequency_for_fitting: f32,
        wanted_astigmatism_tolerance: f32,
        pixel_size: f32,
        wanted_additional_phase_shift_in_radians: f32,
    ) -> Self {
        let mut this = Self::new();
        this.init_with_fitting_parameters(
            wanted_acceleration_voltage,
            wanted_spherical_aberration,
            wanted_amplitude_contrast,
            wanted_defocus_1_in_angstroms,
            wanted_defocus_2_in_angstroms,
            wanted_astigmatism_azimuth,
            wanted_lowest_frequency_for_fitting,
            wanted_highest_frequency_for_fitting,
            wanted_astigmatism_tolerance,
            pixel_size,
            wanted_additional_phase_shift_in_radians,
        );
        this
    }

    /// The eight-argument C++ `CTF::CTF` (`ctf.cpp:39`).
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        wanted_acceleration_voltage: f32,
        wanted_spherical_aberration: f32,
        wanted_amplitude_contrast: f32,
        wanted_defocus_1_in_angstroms: f32,
        wanted_defocus_2_in_angstroms: f32,
        wanted_astigmatism_azimuth: f32,
        pixel_size: f32,
        wanted_additional_phase_shift_in_radians: f32,
    ) -> Self {
        let mut this = Self::new();
        this.init_with_fitting_parameters(
            wanted_acceleration_voltage,
            wanted_spherical_aberration,
            wanted_amplitude_contrast,
            wanted_defocus_1_in_angstroms,
            wanted_defocus_2_in_angstroms,
            wanted_astigmatism_azimuth,
            0.0,
            (1.0 / (2.0 * f64::from(pixel_size))) as f32,
            -10.0,
            pixel_size,
            wanted_additional_phase_shift_in_radians,
        );
        this
    }

    /// The eight-argument C++ `CTF::Init` (`ctf.cpp:58`).
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        wanted_acceleration_voltage_in_kv: f32,
        wanted_spherical_aberration_in_mm: f32,
        wanted_amplitude_contrast: f32,
        wanted_defocus_1_in_angstroms: f32,
        wanted_defocus_2_in_angstroms: f32,
        wanted_astigmatism_azimuth_in_degrees: f32,
        pixel_size_in_angstroms: f32,
        wanted_additional_phase_shift_in_radians: f32,
    ) {
        self.init_with_fitting_parameters(
            wanted_acceleration_voltage_in_kv,
            wanted_spherical_aberration_in_mm,
            wanted_amplitude_contrast,
            wanted_defocus_1_in_angstroms,
            wanted_defocus_2_in_angstroms,
            wanted_astigmatism_azimuth_in_degrees,
            0.0,
            (1.0 / (2.0 * f64::from(pixel_size_in_angstroms))) as f32,
            -10.0,
            pixel_size_in_angstroms,
            wanted_additional_phase_shift_in_radians,
        );
    }

    /// The eleven-argument C++ `CTF::Init` (`ctf.cpp:72`).
    #[allow(clippy::too_many_arguments)]
    pub fn init_with_fitting_parameters(
        &mut self,
        wanted_acceleration_voltage_in_kv: f32,
        wanted_spherical_aberration_in_mm: f32,
        wanted_amplitude_contrast: f32,
        wanted_defocus_1_in_angstroms: f32,
        wanted_defocus_2_in_angstroms: f32,
        wanted_astigmatism_azimuth_in_degrees: f32,
        wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms: f32,
        wanted_highest_frequency_for_fitting_in_reciprocal_angstroms: f32,
        wanted_astigmatism_tolerance_in_angstroms: f32,
        pixel_size_in_angstroms: f32,
        wanted_additional_phase_shift_in_radians: f32,
    ) {
        self.wavelength = self
            .wavelength_given_acceleration_voltage(wanted_acceleration_voltage_in_kv)
            / pixel_size_in_angstroms;
        self.squared_wavelength = f64::from(self.wavelength).powi(2) as f32;
        self.cubed_wavelength = f64::from(self.wavelength).powi(3) as f32;
        self.spherical_aberration = (f64::from(wanted_spherical_aberration_in_mm) * 10000000.0
            / f64::from(pixel_size_in_angstroms)) as f32;
        self.amplitude_contrast = wanted_amplitude_contrast;
        self.defocus_1 = wanted_defocus_1_in_angstroms / pixel_size_in_angstroms;
        self.defocus_2 = wanted_defocus_2_in_angstroms / pixel_size_in_angstroms;
        self.astigmatism_azimuth =
            (f64::from(wanted_astigmatism_azimuth_in_degrees) / 180.0 * PI) as f32;
        self.additional_phase_shift = wanted_additional_phase_shift_in_radians;
        self.lowest_frequency_for_fitting =
            wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
        self.highest_frequency_for_fitting =
            wanted_highest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
        self.astigmatism_tolerance =
            wanted_astigmatism_tolerance_in_angstroms / pixel_size_in_angstroms;

        // DNM 7/24/23: fixed to use square of amplitude contrast
        self.precomputed_amplitude_contrast_term = (f64::from(self.amplitude_contrast)
            / (1.0 - f64::from(self.amplitude_contrast).powi(2)).sqrt())
        .atan() as f32;
    }

    /// C++ `CTF::ReturnNumberOfExtremaBeforeSquaredSpatialFrequency` (`ctf.cpp:103`).
    ///
    /// Eq 11 of Rohou & Grigorieff (2015).
    pub fn return_number_of_extrema_before_squared_spatial_frequency(
        &self,
        squared_spatial_frequency: f32,
        azimuth: f32,
    ) -> i32 {
        let number_of_extrema: i32 = (1.0 / PI
            * f64::from(
                self.phase_shift_given_squared_spatial_frequency_and_azimuth(
                    squared_spatial_frequency,
                    azimuth,
                ),
            )
            + 0.5)
            .floor() as i32;
        number_of_extrema.abs()
    }

    /// C++ `CTF::ReturnSquaredSpatialFrequencyOfAZero` (`ctf.cpp:111`).
    pub fn return_squared_spatial_frequency_of_a_zero(&self, which_zero: i32, azimuth: f32) -> f32 {
        let phase_shift = (f64::from(which_zero) * PI) as f32;
        self.return_squared_spatial_frequency_given_phase_shift_and_azimuth(phase_shift, azimuth)
    }

    /// C++ `CTF::ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth` (`ctf.cpp:119`).
    pub fn return_squared_spatial_frequency_given_phase_shift_and_azimuth(
        &self,
        phase_shift: f32,
        azimuth: f32,
    ) -> f32 {
        let a: f32 =
            (-0.5 * PI * f64::from(self.cubed_wavelength) * f64::from(self.spherical_aberration))
                as f32;
        let b: f32 = (PI
            * f64::from(self.wavelength)
            * f64::from(self.defocus_given_azimuth(azimuth))) as f32;
        let c: f32 = self.additional_phase_shift + self.precomputed_amplitude_contrast_term;
        let det: f32 =
            (f64::from(b.powf(2.0)) - 4.0 * f64::from(a) * f64::from(c - phase_shift)) as f32;

        if self.spherical_aberration == 0.0 {
            (phase_shift - c) / b
        } else if det < 0.0 {
            0.0
        } else {
            let solution_one: f32 = (f64::from(-b + det.sqrt()) / (2.0 * f64::from(a))) as f32;
            let solution_two: f32 = (f64::from(-b - det.sqrt()) / (2.0 * f64::from(a))) as f32;

            if solution_one > 0.0 && solution_two > 0.0 {
                solution_one
            } else if solution_one > 0.0 {
                solution_one
            } else if solution_two > 0.0 {
                solution_two
            } else {
                0.0
            }
        }
    }

    /// C++ `CTF::SetDefocus` (`ctf.cpp:176`): defocus in pixels, angle in radians.
    pub fn set_defocus(
        &mut self,
        wanted_defocus_1_pixels: f32,
        wanted_defocus_2_pixels: f32,
        wanted_astigmatism_angle_radians: f32,
    ) {
        self.defocus_1 = wanted_defocus_1_pixels;
        self.defocus_2 = wanted_defocus_2_pixels;
        self.astigmatism_azimuth = wanted_astigmatism_angle_radians;
    }

    /// C++ `CTF::SetAdditionalPhaseShift` (`ctf.cpp:184`).
    ///
    /// IMOD's `(float)PI` cast makes both operands `float`, so this is `fmodf`.
    pub fn set_additional_phase_shift(&mut self, wanted_additional_phase_shift_radians: f32) {
        self.additional_phase_shift = wanted_additional_phase_shift_radians % (PI as f32);
    }

    /// C++ `CTF::Evaluate` (`ctf.cpp:191`).
    pub fn evaluate(&self, squared_spatial_frequency: f32, azimuth: f32) -> f32 {
        -(self
            .phase_shift_given_squared_spatial_frequency_and_azimuth(
                squared_spatial_frequency,
                azimuth,
            )
            .sin())
    }

    /// C++ `CTF::PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth` (`ctf.cpp:203`).
    pub fn phase_shift_given_squared_spatial_frequency_and_azimuth(
        &self,
        squared_spatial_frequency: f32,
        azimuth: f32,
    ) -> f32 {
        (PI * f64::from(self.wavelength)
            * f64::from(squared_spatial_frequency)
            * (f64::from(self.defocus_given_azimuth(azimuth))
                - 0.5
                    * f64::from(self.squared_wavelength)
                    * f64::from(squared_spatial_frequency)
                    * f64::from(self.spherical_aberration))
            + f64::from(self.additional_phase_shift)
            + f64::from(self.precomputed_amplitude_contrast_term)) as f32
    }

    /// C++ `CTF::DefocusGivenAzimuth` (`ctf.cpp:210`).
    pub fn defocus_given_azimuth(&self, azimuth: f32) -> f32 {
        (0.5 * (f64::from(self.defocus_1 + self.defocus_2)
            + (2.0 * f64::from(azimuth - self.astigmatism_azimuth)).cos()
                * f64::from(self.defocus_1 - self.defocus_2))) as f32
    }

    /// C++ `CTF::WavelengthGivenAccelerationVoltage` (`ctf.cpp:216`).
    pub fn wavelength_given_acceleration_voltage(&self, acceleration_voltage: f32) -> f32 {
        // DNM 7/24/23: Added digits to agree with other sources
        (12.2642
            / (1000.0 * f64::from(acceleration_voltage)
                + 0.9784 * (1000.0 * f64::from(acceleration_voltage)).powi(2) / 10.0f64.powi(6))
            .sqrt()) as f32
    }

    /// C++ `CTF::IsAlmostEqualTo` (`ctf.cpp:225`).
    pub fn is_almost_equal_to(&self, wanted_ctf: &Ctf, delta_defocus: f32) -> bool {
        let mut delta: f32;

        if (self.spherical_aberration - wanted_ctf.spherical_aberration).abs() > 0.01 {
            return false;
        }
        if (self.wavelength - wanted_ctf.wavelength).abs() > 0.0001 {
            return false;
        }
        if (self.amplitude_contrast - wanted_ctf.amplitude_contrast).abs() > 0.0001 {
            return false;
        }
        if (self.defocus_1 - wanted_ctf.defocus_1).abs() > delta_defocus {
            return false;
        }
        if (self.defocus_2 - wanted_ctf.defocus_2).abs() > delta_defocus {
            return false;
        }

        delta = (self.additional_phase_shift - wanted_ctf.additional_phase_shift).abs();
        delta %= 2.0f32 * (PI as f32);
        // 0.0277 = 5/180 (5 deg tolerance)
        if delta > 0.0277 {
            return false;
        }

        delta = (self.astigmatism_azimuth - wanted_ctf.astigmatism_azimuth).abs();
        delta %= PI as f32;
        // 0.0277 = 5/180 (5 deg tolerance)
        if delta > 0.0277 {
            return false;
        }

        true
    }

    /// C++ `CTF::EnforceConvention` (`ctf.cpp:250`).
    ///
    /// Enforce the convention that df1 > df2 and -90 < angast < 90.
    pub fn enforce_convention(&mut self) {
        let defocus_tmp: f32;

        if self.defocus_1 < self.defocus_2 {
            defocus_tmp = self.defocus_2;
            self.defocus_2 = self.defocus_1;
            self.defocus_1 = defocus_tmp;
            self.astigmatism_azimuth = (f64::from(self.astigmatism_azimuth) + PI * 0.5) as f32;
        }
        // IMOD icl 11 use floor(x + .5) instead of round
        self.astigmatism_azimuth = (f64::from(self.astigmatism_azimuth)
            - PI * (f64::from(self.astigmatism_azimuth) / PI + 0.5).floor())
            as f32;
    }

    /// C++ inline `CTF::GetLowestFrequencyForFitting` (`ctf.h:71`).
    pub fn get_lowest_frequency_for_fitting(&self) -> f32 {
        self.lowest_frequency_for_fitting
    }
    /// C++ inline `CTF::GetHighestFrequencyForFitting` (`ctf.h:72`).
    pub fn get_highest_frequency_for_fitting(&self) -> f32 {
        self.highest_frequency_for_fitting
    }
    /// C++ inline `CTF::GetAstigmatismTolerance` (`ctf.h:73`).
    pub fn get_astigmatism_tolerance(&self) -> f32 {
        self.astigmatism_tolerance
    }
    /// C++ inline `CTF::GetAstigmatism` (`ctf.h:74`).
    pub fn get_astigmatism(&self) -> f32 {
        self.defocus_1 - self.defocus_2
    }
    /// C++ inline `CTF::GetDefocus1` (`ctf.h:77`).
    pub fn get_defocus_1(&self) -> f32 {
        self.defocus_1
    }
    /// C++ inline `CTF::GetDefocus2` (`ctf.h:78`).
    pub fn get_defocus_2(&self) -> f32 {
        self.defocus_2
    }
    /// C++ inline `CTF::GetAstigmatismAzimuth` (`ctf.h:79`).
    pub fn get_astigmatism_azimuth(&self) -> f32 {
        self.astigmatism_azimuth
    }
    /// C++ inline `CTF::GetAdditionalPhaseShift` (`ctf.h:80`).
    pub fn get_additional_phase_shift(&self) -> f32 {
        self.additional_phase_shift
    }
    /// C++ inline `CTF::GetWavelength` (`ctf.h:81`).
    pub fn get_wavelength(&self) -> f32 {
        self.wavelength
    }

    /// The private `defocus_half_range` member (`ctf.h:9`), which the source
    /// initialises and never reads.
    pub fn defocus_half_range(&self) -> f32 {
        self.defocus_half_range
    }
}

#[cfg(test)]
mod tests {
    use super::{Ctf, PI};

    #[test]
    fn convention_and_zero_frequency_follow_source_equations() {
        let mut ctf = Ctf::with_parameters(300.0, 2.7, 0.07, 10_000.0, 20_000.0, 0.0, 1.0, 0.0);
        ctf.enforce_convention();
        assert_eq!(ctf.get_defocus_1(), 20_000.0);
        assert_eq!(ctf.get_defocus_2(), 10_000.0);
        assert!((f64::from(ctf.get_astigmatism_azimuth()) + PI * 0.5).abs() < 0.000_001);
        assert!(ctf.return_squared_spatial_frequency_of_a_zero(1, 0.0) > 0.0);
    }

    #[test]
    fn phase_normalization_uses_the_float_modulus() {
        let mut ctf = Ctf::with_parameters(300.0, 2.7, 0.07, 10_000.0, 10_000.0, 0.0, 1.0, 0.0);
        ctf.set_additional_phase_shift((PI * 3.5) as f32);
        assert!((f64::from(ctf.get_additional_phase_shift()) - PI * 0.5).abs() < 0.000_01);
        assert!(ctf.is_almost_equal_to(&ctf.clone(), 100.0));
    }
}
