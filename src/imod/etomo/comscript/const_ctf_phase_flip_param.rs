//! `IMOD/Etomo/src/etomo/comscript/ConstCtfPhaseFlipParam.java`.

use super::command::Command;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;

/// Java `ConstCtfPhaseFlipParam extends Command`.
pub trait ConstCtfPhaseFlipParam: Command {
    /// Java `getVoltage`.
    fn get_voltage(&self) -> &ConstEtomoNumber;

    /// Java `getSphericalAberration`.
    fn get_spherical_aberration(&self) -> &ConstEtomoNumber;

    /// Java `getInvertTiltAngles`.
    fn get_invert_tilt_angles(&self) -> bool;

    /// Java `getAmplitudeContrast`.
    fn get_amplitude_contrast(&self) -> &ConstEtomoNumber;

    /// Java `getDefocusFile`.
    fn get_defocus_file(&self) -> &dyn ConstStringParameter;

    /// Java `getInterpolationWidth`.
    fn get_interpolation_width(&self) -> &ConstEtomoNumber;

    /// Java `getDefocusTol`.
    fn get_defocus_tol(&self) -> &ConstEtomoNumber;

    /// Java `isUseGpu`.
    fn is_use_gpu(&self) -> bool;

    /// Java `getXAxisTilt`.
    fn get_x_axis_tilt(&self) -> String;

    /// Java `getScaleByCtfPower`.
    fn get_scale_by_ctf_power(&self) -> String;

    /// Java `getMinimumZeroSpacing`.
    fn get_minimum_zero_spacing(&self) -> String;
}
