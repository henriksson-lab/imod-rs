//! Translation of `IMOD/librgctf/ctf.{h,cpp}`.

const PI: f32 = 3.141_592_653_59;

/// Contrast-transfer-function parameters and their precomputed terms.
#[derive(Clone, Debug, PartialEq)]
pub struct Ctf {
    spherical_aberration: f32,
    wavelength: f32,
    amplitude_contrast: f32,
    defocus_1: f32,
    defocus_2: f32,
    defocus_half_range: f32,
    astigmatism_azimuth: f32,
    additional_phase_shift: f32,
    lowest_frequency_for_fitting: f32,
    highest_frequency_for_fitting: f32,
    astigmatism_tolerance: f32,
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
    /// C++ `CTF::CTF()`.
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

    /// The eleven-argument C++ `CTF::CTF` overload.
    #[allow(clippy::too_many_arguments)]
    pub fn with_fitting_parameters(
        acceleration_voltage: f32,
        spherical_aberration: f32,
        amplitude_contrast: f32,
        defocus_1: f32,
        defocus_2: f32,
        astigmatism_azimuth: f32,
        lowest_frequency_for_fitting: f32,
        highest_frequency_for_fitting: f32,
        astigmatism_tolerance: f32,
        pixel_size: f32,
        additional_phase_shift: f32,
    ) -> Self {
        let mut ctf = Self::new();
        ctf.init_with_fitting_parameters(
            acceleration_voltage,
            spherical_aberration,
            amplitude_contrast,
            defocus_1,
            defocus_2,
            astigmatism_azimuth,
            lowest_frequency_for_fitting,
            highest_frequency_for_fitting,
            astigmatism_tolerance,
            pixel_size,
            additional_phase_shift,
        );
        ctf
    }

    /// The eight-argument C++ `CTF::CTF` overload.
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        acceleration_voltage: f32,
        spherical_aberration: f32,
        amplitude_contrast: f32,
        defocus_1: f32,
        defocus_2: f32,
        astigmatism_azimuth: f32,
        pixel_size: f32,
        additional_phase_shift: f32,
    ) -> Self {
        let mut ctf = Self::new();
        ctf.init(
            acceleration_voltage,
            spherical_aberration,
            amplitude_contrast,
            defocus_1,
            defocus_2,
            astigmatism_azimuth,
            pixel_size,
            additional_phase_shift,
        );
        ctf
    }

    /// The eight-argument C++ `CTF::Init` overload.
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        acceleration_voltage: f32,
        spherical_aberration: f32,
        amplitude_contrast: f32,
        defocus_1: f32,
        defocus_2: f32,
        astigmatism_azimuth: f32,
        pixel_size: f32,
        additional_phase_shift: f32,
    ) {
        self.init_with_fitting_parameters(
            acceleration_voltage,
            spherical_aberration,
            amplitude_contrast,
            defocus_1,
            defocus_2,
            astigmatism_azimuth,
            0.0,
            1.0 / (2.0 * pixel_size),
            -10.0,
            pixel_size,
            additional_phase_shift,
        );
    }

    /// The eleven-argument C++ `CTF::Init` overload.
    #[allow(clippy::too_many_arguments)]
    pub fn init_with_fitting_parameters(
        &mut self,
        acceleration_voltage: f32,
        spherical_aberration: f32,
        amplitude_contrast: f32,
        defocus_1: f32,
        defocus_2: f32,
        astigmatism_azimuth: f32,
        lowest_frequency_for_fitting: f32,
        highest_frequency_for_fitting: f32,
        astigmatism_tolerance: f32,
        pixel_size: f32,
        additional_phase_shift: f32,
    ) {
        self.wavelength =
            self.wavelength_given_acceleration_voltage(acceleration_voltage) / pixel_size;
        self.squared_wavelength = self.wavelength.powi(2);
        self.cubed_wavelength = self.wavelength.powi(3);
        self.spherical_aberration = spherical_aberration * 10_000_000.0 / pixel_size;
        self.amplitude_contrast = amplitude_contrast;
        self.defocus_1 = defocus_1 / pixel_size;
        self.defocus_2 = defocus_2 / pixel_size;
        self.astigmatism_azimuth = astigmatism_azimuth / 180.0 * PI;
        self.additional_phase_shift = additional_phase_shift;
        self.lowest_frequency_for_fitting = lowest_frequency_for_fitting * pixel_size;
        self.highest_frequency_for_fitting = highest_frequency_for_fitting * pixel_size;
        self.astigmatism_tolerance = astigmatism_tolerance / pixel_size;
        self.precomputed_amplitude_contrast_term =
            (amplitude_contrast / (1.0 - amplitude_contrast.powi(2)).sqrt()).atan();
    }

    /// C++ `CTF::SetDefocus`.
    pub fn set_defocus(&mut self, defocus_1: f32, defocus_2: f32, astigmatism_angle: f32) {
        self.defocus_1 = defocus_1;
        self.defocus_2 = defocus_2;
        self.astigmatism_azimuth = astigmatism_angle;
    }

    /// C++ `CTF::SetAdditionalPhaseShift`.
    pub fn set_additional_phase_shift(&mut self, phase_shift: f32) {
        self.additional_phase_shift = phase_shift % PI;
    }

    /// C++ `CTF::Evaluate`.
    pub fn evaluate(&self, squared_spatial_frequency: f32, azimuth: f32) -> f32 {
        -self
            .phase_shift_given_squared_spatial_frequency_and_azimuth(
                squared_spatial_frequency,
                azimuth,
            )
            .sin()
    }

    /// C++ `CTF::PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth`.
    pub fn phase_shift_given_squared_spatial_frequency_and_azimuth(
        &self,
        squared_spatial_frequency: f32,
        azimuth: f32,
    ) -> f32 {
        PI * self.wavelength
            * squared_spatial_frequency
            * (self.defocus_given_azimuth(azimuth)
                - 0.5
                    * self.squared_wavelength
                    * squared_spatial_frequency
                    * self.spherical_aberration)
            + self.additional_phase_shift
            + self.precomputed_amplitude_contrast_term
    }

    /// C++ `CTF::DefocusGivenAzimuth`.
    pub fn defocus_given_azimuth(&self, azimuth: f32) -> f32 {
        0.5 * (self.defocus_1
            + self.defocus_2
            + (2.0 * (azimuth - self.astigmatism_azimuth)).cos()
                * (self.defocus_1 - self.defocus_2))
    }

    /// C++ `CTF::WavelengthGivenAccelerationVoltage`.
    pub fn wavelength_given_acceleration_voltage(&self, acceleration_voltage: f32) -> f32 {
        12.2642
            / (1000.0 * acceleration_voltage
                + 0.9784 * (1000.0 * acceleration_voltage).powi(2) / 1_000_000.0)
                .sqrt()
    }

    /// `ReturnNumberOfExtremaBeforeSquaredSpatialFrequency`.
    pub fn number_of_extrema_before_squared_spatial_frequency(
        &self,
        frequency: f32,
        azimuth: f32,
    ) -> i32 {
        ((self.phase_shift_given_squared_spatial_frequency_and_azimuth(frequency, azimuth) / PI
            + 0.5)
            .floor() as i32)
            .abs()
    }

    /// `ReturnSquaredSpatialFrequencyOfAZero`.
    pub fn squared_spatial_frequency_of_a_zero(&self, which_zero: i32, azimuth: f32) -> f32 {
        self.squared_spatial_frequency_given_phase_shift_and_azimuth(
            which_zero as f32 * PI,
            azimuth,
        )
    }

    /// `ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth`.
    pub fn squared_spatial_frequency_given_phase_shift_and_azimuth(
        &self,
        phase_shift: f32,
        azimuth: f32,
    ) -> f32 {
        let a = -0.5 * PI * self.cubed_wavelength * self.spherical_aberration;
        let b = PI * self.wavelength * self.defocus_given_azimuth(azimuth);
        let c = self.additional_phase_shift + self.precomputed_amplitude_contrast_term;
        let determinant = b.powi(2) - 4.0 * a * (c - phase_shift);
        if self.spherical_aberration == 0.0 {
            return (phase_shift - c) / b;
        }
        if determinant < 0.0 {
            return 0.0;
        }
        let solution_one = (-b + determinant.sqrt()) / (2.0 * a);
        let solution_two = (-b - determinant.sqrt()) / (2.0 * a);
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

    /// C++ `CTF::EnforceConvention`.
    pub fn enforce_convention(&mut self) {
        if self.defocus_1 < self.defocus_2 {
            (self.defocus_1, self.defocus_2) = (self.defocus_2, self.defocus_1);
            self.astigmatism_azimuth += PI * 0.5;
        }
        self.astigmatism_azimuth -= PI * (self.astigmatism_azimuth / PI + 0.5).floor();
    }

    /// `GetLowestFrequencyForFitting`.
    pub fn lowest_frequency_for_fitting(&self) -> f32 {
        self.lowest_frequency_for_fitting
    }
    /// `GetHighestFrequencyForFitting`.
    pub fn highest_frequency_for_fitting(&self) -> f32 {
        self.highest_frequency_for_fitting
    }
    /// `GetAstigmatismTolerance`.
    pub fn astigmatism_tolerance(&self) -> f32 {
        self.astigmatism_tolerance
    }
    /// `GetAstigmatism`.
    pub fn astigmatism(&self) -> f32 {
        self.defocus_1 - self.defocus_2
    }
    /// `GetDefocus1`.
    pub fn defocus_1(&self) -> f32 {
        self.defocus_1
    }
    /// `GetDefocus2`.
    pub fn defocus_2(&self) -> f32 {
        self.defocus_2
    }
    /// `GetAstigmatismAzimuth`.
    pub fn astigmatism_azimuth(&self) -> f32 {
        self.astigmatism_azimuth
    }
    /// `GetAdditionalPhaseShift`.
    pub fn additional_phase_shift(&self) -> f32 {
        self.additional_phase_shift
    }
    /// `GetWavelength`.
    pub fn wavelength(&self) -> f32 {
        self.wavelength
    }
}

#[cfg(test)]
mod tests {
    use super::Ctf;

    #[test]
    fn convention_and_zero_frequency_follow_source_equations() {
        let mut ctf = Ctf::with_parameters(300.0, 2.7, 0.07, 10_000.0, 20_000.0, 0.0, 1.0, 0.0);
        ctf.enforce_convention();
        assert_eq!(ctf.defocus_1(), 20_000.0);
        assert_eq!(ctf.defocus_2(), 10_000.0);
        assert!((ctf.astigmatism_azimuth() + super::PI * 0.5).abs() < 0.000_001);
        assert!(ctf.squared_spatial_frequency_of_a_zero(1, 0.0) > 0.0);
    }

    #[test]
    fn phase_normalization_and_comparison_use_source_tolerances() {
        let mut ctf = Ctf::with_parameters(300.0, 2.7, 0.07, 10_000.0, 10_000.0, 0.0, 1.0, 0.0);
        ctf.set_additional_phase_shift(super::PI * 3.5);
        assert!((ctf.additional_phase_shift() - super::PI * 0.5).abs() < 0.000_001);
    }
}
