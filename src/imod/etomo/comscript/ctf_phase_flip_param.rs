//! `IMOD/Etomo/src/etomo/comscript/CtfPhaseFlipParam.java`.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use super::shared_constants;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `COMMAND`.
pub const COMMAND: &str = "ctfphaseflip";
/// Java `VOLTAGE_OPTION`.
pub const VOLTAGE_OPTION: &str = "Voltage";
/// Java `SPHERICAL_ABERRATION_OPTION`.
pub const SPHERICAL_ABERRATION_OPTION: &str = "SphericalAberration";
/// Java `INVERT_TILT_ANGLES_OPTION`.
pub const INVERT_TILT_ANGLES_OPTION: &str = "InvertTiltAngles";
/// Java `AMPLITUDE_CONTRAST_OPTION`.
pub const AMPLITUDE_CONTRAST_OPTION: &str = "AmplitudeContrast";
/// Java `INTERPOLATION_WIDTH_OPTION`.
pub const INTERPOLATION_WIDTH_OPTION: &str = "InterpolationWidth";
/// Java `DEFOCUS_TOL_OPTION`.
pub const DEFOCUS_TOL_OPTION: &str = "DefocusTol";
/// Java private `PIXEL_SIZE_OPTION`.
const PIXEL_SIZE_OPTION: &str = "PixelSize";
/// Java `X_AXIS_TILT_OPTION`.
pub const X_AXIS_TILT_OPTION: &str = "XAxisTilt";
/// Java `SCALE_BY_CTF_POWER_OPTION`.
pub const SCALE_BY_CTF_POWER_OPTION: &str = "ScaleByCtfPower";
/// Java `MINIMUM_ZERO_SPACING_OPTION`.
pub const MINIMUM_ZERO_SPACING_OPTION: &str = "MinimumZeroSpacing";
/// Java private `UNBINNED_PIXEL_SIZE_OPTION`.
const UNBINNED_PIXEL_SIZE_OPTION: &str = "UnbinnedPixelSize";

/// Java final `CtfPhaseFlipParam`.
pub struct CtfPhaseFlipParam {
    voltage: ScriptParameter,
    spherical_aberration: ScriptParameter,
    invert_tilt_angles: EtomoBoolean2,
    amplitude_contrast: ScriptParameter,
    defocus_file: StringParameter,
    interpolation_width: ScriptParameter,
    defocus_tol: ScriptParameter,
    output_file_name: StringParameter,
    pixel_size: ScriptParameter,
    use_gpu: ScriptParameter,
    action_if_gpu_fails: StringParameter,
    x_axis_tilt: ScriptParameter,
    scale_by_ctf_power: ScriptParameter,
    minimum_zero_spacing: ScriptParameter,
    unbinned_pixel_size: ScriptParameter,

    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl CtfPhaseFlipParam {
    /// Java package-private `CtfPhaseFlipParam(BaseManager, AxisID)`.
    pub(crate) fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> CtfPhaseFlipParam {
        CtfPhaseFlipParam {
            voltage: ScriptParameter::new_with_name(VOLTAGE_OPTION),
            spherical_aberration: ScriptParameter::new_with_type_and_name(
                Type::Double,
                SPHERICAL_ABERRATION_OPTION,
            ),
            invert_tilt_angles: EtomoBoolean2::new_with_name(INVERT_TILT_ANGLES_OPTION),
            amplitude_contrast: ScriptParameter::new_with_type_and_name(
                Type::Double,
                AMPLITUDE_CONTRAST_OPTION,
            ),
            defocus_file: StringParameter::new("DefocusFile"),
            interpolation_width: ScriptParameter::new_with_name(INTERPOLATION_WIDTH_OPTION),
            defocus_tol: ScriptParameter::new_with_name(DEFOCUS_TOL_OPTION),
            output_file_name: StringParameter::new("OutputFileName"),
            pixel_size: ScriptParameter::new_with_type_and_name(Type::Double, PIXEL_SIZE_OPTION),
            use_gpu: ScriptParameter::new_with_name("UseGPU"),
            action_if_gpu_fails: StringParameter::new("ActionIfGPUFails"),
            x_axis_tilt: ScriptParameter::new_with_type_and_name(Type::Double, X_AXIS_TILT_OPTION),
            scale_by_ctf_power: ScriptParameter::new_with_type_and_name(
                Type::Double,
                SCALE_BY_CTF_POWER_OPTION,
            ),
            minimum_zero_spacing: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MINIMUM_ZERO_SPACING_OPTION,
            ),
            unbinned_pixel_size: ScriptParameter::new_with_type_and_name(
                Type::Double,
                UNBINNED_PIXEL_SIZE_OPTION,
            ),
            manager,
            axis_id,
        }
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.voltage.reset();
        self.spherical_aberration.reset();
        self.invert_tilt_angles.reset();
        self.amplitude_contrast.reset();
        self.defocus_file.reset();
        self.interpolation_width.reset();
        self.defocus_tol.reset();
        self.use_gpu.reset();
        self.action_if_gpu_fails
            .set(Some(shared_constants::ACTION_IF_GPU_FAILS_DEFAULT));
        self.x_axis_tilt.reset();
        self.scale_by_ctf_power.reset();
        self.minimum_zero_spacing.reset();
    }

    /// Java `setVoltage`.
    pub fn set_voltage(&mut self, input: Option<&str>) {
        self.voltage.set_string(input);
    }

    /// Java `setOutputFileName`.
    pub fn set_output_file_name(&mut self, input: Option<&str>) {
        self.output_file_name.set(input);
    }

    /// Java `setSphericalAberration`.
    pub fn set_spherical_aberration(&mut self, input: Option<&str>) {
        self.spherical_aberration.set_string(input);
    }

    /// Java `setInvertTiltAngles`.
    pub fn set_invert_tilt_angles(&mut self, input: bool) {
        self.invert_tilt_angles.set_boolean(input);
    }

    /// Java `setXAxisTilt`.
    pub fn set_x_axis_tilt(&mut self, input: Option<&str>) {
        self.x_axis_tilt.set_string(input);
    }

    /// Java `setScaleByCtfPower`.
    pub fn set_scale_by_ctf_power(&mut self, input: Option<&str>) {
        self.scale_by_ctf_power.set_string(input);
    }

    /// Java `setMinimumZeroSpacing`.
    pub fn set_minimum_zero_spacing(&mut self, input: Option<&str>) {
        self.minimum_zero_spacing.set_string(input);
    }

    /// Java `setUseGpu`.
    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        if use_gpu {
            self.use_gpu.set_int(shared_constants::USE_GPU_BEST);
        } else {
            self.use_gpu.reset();
        }
    }

    /// Java `setAmplitudeContrast`.
    pub fn set_amplitude_contrast(&mut self, input: Option<&str>) {
        self.amplitude_contrast.set_string(input);
    }

    /// Java `setDefocusFile`.
    pub fn set_defocus_file(&mut self, input: Option<&str>) {
        self.defocus_file.set(input);
    }

    /// Java `setInterpolationWidth`.
    pub fn set_interpolation_width(&mut self, input: Option<&str>) {
        self.interpolation_width.set_string(input);
    }

    /// Java `setDefocusTol`.
    pub fn set_defocus_tol(&mut self, input: Option<&str>) {
        self.defocus_tol.set_string(input);
    }

    /// Java `setPixelSize`.
    pub fn set_pixel_size(&mut self, input: f64) {
        self.pixel_size.set_double(input);
    }

    /// Java `updateUnbinnedPixelSize(double)`.  The parameter is unused, as in the
    /// source.
    pub fn update_unbinned_pixel_size(&mut self, _setup_pixel_size: f64) {
        if self.unbinned_pixel_size.is_null() {
            self.unbinned_pixel_size
                .set_const_etomo_number(Some(&self.pixel_size.base.base));
        }
    }
}

impl CommandParam for CtfPhaseFlipParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        self.voltage.parse(script_command)?;
        self.spherical_aberration.parse(script_command)?;
        self.invert_tilt_angles.parse(script_command)?;
        self.amplitude_contrast.parse(script_command)?;
        self.defocus_file.parse(script_command)?;
        self.interpolation_width.parse(script_command)?;
        self.defocus_tol.parse(script_command)?;
        self.use_gpu.parse(script_command)?;
        self.action_if_gpu_fails.parse(script_command)?;
        self.x_axis_tilt.parse(script_command)?;
        self.scale_by_ctf_power.parse(script_command)?;
        self.minimum_zero_spacing.parse(script_command)?;
        self.unbinned_pixel_size.parse(script_command)?;
        if self.action_if_gpu_fails.is_empty() {
            self.action_if_gpu_fails
                .set(Some(shared_constants::ACTION_IF_GPU_FAILS_DEFAULT));
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        self.voltage.update_com_script(script_command);
        self.spherical_aberration.update_com_script(script_command);
        self.invert_tilt_angles.update_com_script(script_command);
        self.amplitude_contrast.update_com_script(script_command);
        self.defocus_file.update_com_script(script_command);
        self.interpolation_width.update_com_script(script_command);
        self.defocus_tol.update_com_script(script_command);
        self.output_file_name.update_com_script(script_command);
        self.pixel_size.update_com_script(script_command);
        self.use_gpu.update_com_script(script_command);
        self.action_if_gpu_fails.update_com_script(script_command);
        self.x_axis_tilt.update_com_script(script_command);
        self.scale_by_ctf_power.update_com_script(script_command);
        self.minimum_zero_spacing.update_com_script(script_command);
        self.unbinned_pixel_size.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl ConstCtfPhaseFlipParam for CtfPhaseFlipParam {
    fn get_voltage(&self) -> &ConstEtomoNumber {
        &self.voltage.base.base
    }

    fn get_spherical_aberration(&self) -> &ConstEtomoNumber {
        &self.spherical_aberration.base.base
    }

    fn get_invert_tilt_angles(&self) -> bool {
        self.invert_tilt_angles.is()
    }

    fn get_x_axis_tilt(&self) -> String {
        self.x_axis_tilt.to_string()
    }

    fn get_scale_by_ctf_power(&self) -> String {
        self.scale_by_ctf_power.to_string()
    }

    fn get_minimum_zero_spacing(&self) -> String {
        self.minimum_zero_spacing.to_string()
    }

    fn is_use_gpu(&self) -> bool {
        !self.use_gpu.is_null()
    }

    fn get_amplitude_contrast(&self) -> &ConstEtomoNumber {
        &self.amplitude_contrast.base.base
    }

    fn get_defocus_file(&self) -> &dyn ConstStringParameter {
        &self.defocus_file
    }

    fn get_interpolation_width(&self) -> &ConstEtomoNumber {
        &self.interpolation_width.base.base
    }

    fn get_defocus_tol(&self) -> &ConstEtomoNumber {
        &self.defocus_tol.base.base
    }
}

impl Command for CtfPhaseFlipParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::CTF_CORRECTION)
    }

    fn get_command(&self) -> Option<String> {
        file_type::CLASS
            .ctf_correction_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::CTF_CORRECTION.to_string())
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `{ getCommandLine() }`: a one-element array whose element may be null.
    /// A null element is dropped here.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.get_command_line().into_iter().collect())
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        Some(file_type::CLASS.ctf_corrected_stack.clone())
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.ctf_corrected_stack))
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        ConstCtfPhaseFlipParam::is_use_gpu(self)
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}
