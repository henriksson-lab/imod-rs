//! `IMOD/Etomo/src/etomo/comscript/CtfPlotterParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_ctf_plotter_param::ConstCtfPlotterParam;
use super::ctf_phase_flip_param;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::util::utilities;

/// Java `COMMAND`.
pub const COMMAND: &str = "ctfplotter";
/// Java `CONFIG_FILE_OPTION`.
pub const CONFIG_FILE_OPTION: &str = "ConfigFile";
/// Java `SCAN_DEFOCUS_RANGE_OPTION`.
pub const SCAN_DEFOCUS_RANGE_OPTION: &str = "ScanDefocusRange";
/// Java `EXPECTED_DEFOCUS_OPTION`.
pub const EXPECTED_DEFOCUS_OPTION: &str = "ExpectedDefocus";
/// Java `PHASE_SHIFT_IN_DEGREES_OPTION`.
pub const PHASE_SHIFT_IN_DEGREES_OPTION: &str = "PhaseShiftInDegrees";
/// Java `OFFSET_TO_ADD_OPTION`.
pub const OFFSET_TO_ADD_OPTION: &str = "OffsetToAdd";
/// Java `TUNE_FITTING_AND_SAMPLING_OPTION`.
pub const TUNE_FITTING_AND_SAMPLING_OPTION: &str = "TuneFittingAndSampling";

/// Java final `CtfPlotterParam`.
pub struct CtfPlotterParam {
    voltage: ScriptParameter,
    spherical_aberration: ScriptParameter,
    invert_tilt_angles: EtomoBoolean2,
    amplitude_contrast: ScriptParameter,
    config_file: StringParameter,
    /// Java package-private field `scanDefocusRange`.
    pub(crate) scan_defocus_range: FortranInputString,
    expected_defocus: ScriptParameter,
    phase_shift_in_degrees: ScriptParameter,
    phase_plate_shift: ScriptParameter,
    offset_to_add: ScriptParameter,
    /// Java package-private field `autoFitRangeAndStep`.
    pub(crate) auto_fit_range_and_step: FortranInputString,
}

impl CtfPlotterParam {
    /// Java implicit `CtfPlotterParam()`.
    pub fn new() -> CtfPlotterParam {
        CtfPlotterParam {
            voltage: ScriptParameter::new_with_name(ctf_phase_flip_param::VOLTAGE_OPTION),
            spherical_aberration: ScriptParameter::new_with_type_and_name(
                Type::Double,
                ctf_phase_flip_param::SPHERICAL_ABERRATION_OPTION,
            ),
            invert_tilt_angles: EtomoBoolean2::new_with_name(
                ctf_phase_flip_param::INVERT_TILT_ANGLES_OPTION,
            ),
            amplitude_contrast: ScriptParameter::new_with_type_and_name(
                Type::Double,
                ctf_phase_flip_param::AMPLITUDE_CONTRAST_OPTION,
            ),
            config_file: StringParameter::new(CONFIG_FILE_OPTION),
            scan_defocus_range: FortranInputString::new_with_key(Some("ScanDefocusRange"), 2),
            expected_defocus: ScriptParameter::new_with_type_and_name(
                Type::Double,
                EXPECTED_DEFOCUS_OPTION,
            ),
            phase_shift_in_degrees: ScriptParameter::new_with_type_and_name(
                Type::Double,
                PHASE_SHIFT_IN_DEGREES_OPTION,
            ),
            phase_plate_shift: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "PhasePlateShift",
            ),
            offset_to_add: ScriptParameter::new_with_type_and_name(
                Type::Double,
                OFFSET_TO_ADD_OPTION,
            ),
            auto_fit_range_and_step: FortranInputString::new_with_key(
                Some("AutoFitRangeAndStep"),
                2,
            ),
        }
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.voltage.reset();
        self.spherical_aberration.reset();
        self.invert_tilt_angles.reset();
        self.amplitude_contrast.reset();
        self.scan_defocus_range.reset();
        self.expected_defocus.reset();
        self.phase_shift_in_degrees.reset();
        self.phase_plate_shift.reset();
        self.offset_to_add.reset();
        self.auto_fit_range_and_step.reset();
    }

    /// Java `setVoltage`.
    pub fn set_voltage(&mut self, input: Option<&str>) {
        self.voltage.set_string(input);
    }

    /// Java `setSphericalAberration`.
    pub fn set_spherical_aberration(&mut self, input: Option<&str>) {
        self.spherical_aberration.set_string(input);
    }

    /// Java `setInvertTiltAngles`.
    pub fn set_invert_tilt_angles(&mut self, input: bool) {
        self.invert_tilt_angles.set_boolean(input);
    }

    /// Java `setAmplitudeContrast`.
    pub fn set_amplitude_contrast(&mut self, input: Option<&str>) {
        self.amplitude_contrast.set_string(input);
    }

    /// Java `setAutoFitRangeAndStep`.
    pub fn set_auto_fit_range_and_step(&mut self, input: &FortranInputString) {
        self.auto_fit_range_and_step.set_fortran_input_string(input);
    }

    /// Java `setConfigFile`.
    pub fn set_config_file(&mut self, input: Option<&str>) {
        self.config_file.set(input);
    }

    /// Java `setScanDefocusRange`.
    pub fn set_scan_defocus_range(
        &mut self,
        input1: Option<&str>,
        input2: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.scan_defocus_range.validate_and_set_two(input1, input2)
    }

    /// Java `setExpectedDefocus`.
    pub fn set_expected_defocus(&mut self, input: Option<&str>) {
        self.expected_defocus.set_string(input);
    }

    /// Java `setPhaseShiftInDegrees`.
    pub fn set_phase_shift_in_degrees(&mut self, input: Option<&str>) {
        self.phase_shift_in_degrees.set_string(input);
    }

    /// Java `setOffsetToAdd`.
    pub fn set_offset_to_add(&mut self, input: Option<&str>) {
        self.offset_to_add.set_string(input);
    }

    /// Java `resetScanDefocusRange`.
    pub fn reset_scan_defocus_range(&mut self) {
        self.scan_defocus_range.reset();
    }
}

impl Default for CtfPlotterParam {
    fn default() -> CtfPlotterParam {
        CtfPlotterParam::new()
    }
}

impl CommandParam for CtfPlotterParam {
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
        self.config_file.parse(script_command)?;
        self.scan_defocus_range
            .validate_and_set_com_script(script_command)?;
        self.expected_defocus.parse(script_command)?;
        // PhaseShiftInDegrees and PhasePlateShift are the same parameter - one in degrees
        // and the other in radians. PhaseShiftInDegrees takes precedence - fill it with
        // PhasePlateShift when it is has no value. Don't keep both.
        self.phase_shift_in_degrees.parse(script_command)?;
        self.phase_plate_shift.parse(script_command)?;
        if self.phase_shift_in_degrees.is_null() && !self.phase_plate_shift.is_null() {
            // Round converted phaseShiftInDegrees to xx.x.
            // Fixed in translation: CtfPlotterParam.java:88-89 is
            // `Math.round(Math.toDegrees(x) * 10) / 10` - a long divided by the int 10,
            // so the value is truncated to whole degrees, contradicting the comment
            // above.  The division here is in double, which gives the xx.x the source
            // intends.
            let rounded = utilities::java_lang_math_round(
                self.phase_plate_shift.get_double().to_degrees() * 10.0,
            );
            self.phase_shift_in_degrees
                .set_double(rounded as f64 / 10.0);
            eprintln!(
                "CtfPlotterParam: Converted the value of PhasePlateShift ({}) to degrees and placed it in .PhaseShiftInDegrees.",
                self.phase_plate_shift
            );
        }
        self.phase_plate_shift.reset();
        //
        self.offset_to_add.parse(script_command)?;
        self.auto_fit_range_and_step
            .validate_and_set_com_script(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Takes `&self` as the trait does; the source's
    /// `phasePlateShift.reset()` only clears a value `parseComScriptCommand` has
    /// already reset, so the parameter is removed from the script either way.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        self.voltage.update_com_script(script_command);
        self.spherical_aberration.update_com_script(script_command);
        self.invert_tilt_angles.update_com_script(script_command);
        self.amplitude_contrast.update_com_script(script_command);
        self.config_file.update_com_script(script_command);
        self.scan_defocus_range
            .update_script_parameter(script_command);
        self.expected_defocus.update_com_script(script_command);
        // PhaseShiftInDegrees and PhasePlateShift: see parseComScriptCommand.
        self.phase_shift_in_degrees
            .update_com_script(script_command);
        if !self.phase_plate_shift.is_null() {
            eprintln!(
                "CtfPlotterParam: PhasePlateShift has been deleted - was {}.",
                self.phase_plate_shift
            );
        }
        // `phasePlateShift.reset(); phasePlateShift.updateComScript(scriptCommand);`:
        // a reset (null) ScriptParameter's updateComScript deletes its key.
        self.phase_plate_shift
            .delete_from_com_script(script_command);
        //
        self.offset_to_add.update_com_script(script_command);
        self.auto_fit_range_and_step
            .update_script_parameter(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl ConstCtfPlotterParam for CtfPlotterParam {
    fn get_config_file(&self) -> String {
        self.config_file.to_string()
    }

    fn get_phase_shift_in_degrees(&self) -> String {
        self.phase_shift_in_degrees.to_string()
    }

    fn is_scan_defocus_range(&self) -> bool {
        !self.scan_defocus_range.is_null()
    }

    fn is_scan_defocus_range_low(&self) -> bool {
        !self.scan_defocus_range.is_null_index(0)
    }

    fn is_scan_defocus_range_high(&self) -> bool {
        !self.scan_defocus_range.is_null_index(1)
    }

    fn get_scan_defocus_range(&self) -> Option<String> {
        if self.scan_defocus_range.is_null() {
            return None;
        }
        Some(self.scan_defocus_range.to_string())
    }

    fn get_scan_defocus_range_low(&self) -> Option<f64> {
        if self.scan_defocus_range.is_null_index(0) {
            return None;
        }
        Some(self.scan_defocus_range.get_double_index(0))
    }

    fn get_scan_defocus_range_high(&self) -> Option<f64> {
        if self.scan_defocus_range.is_null_index(1) {
            return None;
        }
        Some(self.scan_defocus_range.get_double_index(1))
    }

    fn get_expected_defocus(&self) -> Option<&ConstEtomoNumber> {
        if !self.expected_defocus.is() {
            return None;
        }
        Some(&self.expected_defocus.base.base)
    }

    fn get_offset_to_add(&self) -> &ConstEtomoNumber {
        &self.offset_to_add.base.base
    }
}
