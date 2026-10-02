//! `IMOD/Etomo/src/etomo/comscript/ReduceFiltVolParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::FortranInputString;
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::REDUCE_FILT_VOL;
/// Java `NOT_ENOUGH_MEMORY_MSG`.
pub const NOT_ENOUGH_MEMORY_MSG: &str = "The whole volume is too large to \
filter at once but is ready to be filtered in parallel.\n\
Select File - New - Generic Parallel Process then select rfvfilter-001.com for \
the Process name.";
/// Java `LOW_PASS_RADIUS_SIGMA_NPARAMS`.
pub const LOW_PASS_RADIUS_SIGMA_NPARAMS: i32 = 2;
/// Java `LOW_PASS_CUTOFF_INDEX`.
pub const LOW_PASS_CUTOFF_INDEX: i32 = 0;
/// Java `LOW_PASS_CUTOFF_MIN`.
pub const LOW_PASS_CUTOFF_MIN: f64 = 0.0;
/// Java `LOW_PASS_CUTOFF_MAX`.
pub const LOW_PASS_CUTOFF_MAX: f64 = 0.5;
/// Java `LOW_PASS_SIGMA_INDEX`.
pub const LOW_PASS_SIGMA_INDEX: i32 = 1;
/// Java `LOW_PASS_SIGMA_MIN`.
pub const LOW_PASS_SIGMA_MIN: f64 = 0.0;
/// Java `HIGH_PASS_NYQUIST_MIN`.
pub const HIGH_PASS_NYQUIST_MIN: f64 = 0.0;
/// Java `HIGH_PASS_NYQUIST_MAX`.
pub const HIGH_PASS_NYQUIST_MAX: f64 = 1.0;
/// Java `DEFAULT_REDUCTION_FACTOR_FOR_OUTPUT_FILE`.
pub const DEFAULT_REDUCTION_FACTOR_FOR_OUTPUT_FILE: f64 = 1.0;
/// Java `TRIM_VOL_OUTPUT_MIN_PIXEL_AREA`.
pub const TRIM_VOL_OUTPUT_MIN_PIXEL_AREA: f64 = 1960000.0;
/// Java `INPUT_FILE`.
pub const INPUT_FILE: &str = "InputFile";
/// Java `OUTPUT_FILE`.
pub const OUTPUT_FILE: &str = "OutputFile";
/// Java `REDUCTION_FACTOR`.
pub const REDUCTION_FACTOR: &str = "ReductionFactor";
/// Java `Z_REDUCTION_FACTOR`.
pub const Z_REDUCTION_FACTOR: &str = "ZReductionFactor";
/// Java `LOW_PASS_RADIUS_SIGMA`.
pub const LOW_PASS_RADIUS_SIGMA: &str = "LowPassRadiusSigma";
/// Java `DECONVOLUTION_STRENGTH`.
pub const DECONVOLUTION_STRENGTH: &str = "DeconvolutionStrength";
/// Java `SNR_FALLOFF`.
pub const SNR_FALLOFF: &str = "SNRFalloff";
/// Java `HIGH_PASS_NYQUIST`.
pub const HIGH_PASS_NYQUIST: &str = "HighPassNyquist";
/// Java `DEFOCUS_IN_MICRONS`.
pub const DEFOCUS_IN_MICRONS: &str = "DefocusInMicrons";
/// Java `PHASE_SHIFT`.
pub const PHASE_SHIFT: &str = "PhaseShift";
/// Java `MODE_TO_OUTPUT`.
pub const MODE_TO_OUTPUT: &str = "ModeToOutput";
/// Java `SETUP_CHUNKS_IF_MEMORY_ERROR`.
pub const SETUP_CHUNKS_IF_MEMORY_ERROR: &str = "SetupChunksIfMemoryError";

/// Java nested class `ReduceFiltVolParam.Field`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `Field.IS_REDUCE_FILT_VOL_FLIPPED`.
    IsReduceFiltVolFlipped,
}

impl FieldInterface for Field {}

/// Java final `ReduceFiltVolParam`.
pub struct ReduceFiltVolParam {
    input_file: StringParameter,
    output_file: StringParameter,
    reduction_factor: ScriptParameter,
    z_reduction_factor: ScriptParameter,
    low_pass_radius_sigma: FortranInputString,
    deconvolution_strength: ScriptParameter,
    snr_falloff: ScriptParameter,
    high_pass_nyquist: ScriptParameter,
    defocus_in_microns: ScriptParameter,
    phase_shift: ScriptParameter,
    mode_to_output: ScriptParameter,
    setup_chunks_if_memory_error: EtomoBoolean2,
    /// Java `manager`, read only by the constructor.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    reduce_filt_vol_comscript_file: Option<std::path::PathBuf>,
    is_reduce_filt_vol_flipped: bool,
}

impl ReduceFiltVolParam {
    /// Java `ReduceFiltVolParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ReduceFiltVolParam {
        let mut param = ReduceFiltVolParam {
            input_file: StringParameter::new(INPUT_FILE),
            output_file: StringParameter::new(OUTPUT_FILE),
            reduction_factor: ScriptParameter::new_with_type_and_name(
                Type::Double,
                REDUCTION_FACTOR,
            ),
            z_reduction_factor: ScriptParameter::new_with_type_and_name(
                Type::Double,
                Z_REDUCTION_FACTOR,
            ),
            low_pass_radius_sigma: FortranInputString::new_with_key(
                Some(LOW_PASS_RADIUS_SIGMA),
                LOW_PASS_RADIUS_SIGMA_NPARAMS,
            ),
            deconvolution_strength: ScriptParameter::new_with_type_and_name(
                Type::Double,
                DECONVOLUTION_STRENGTH,
            ),
            snr_falloff: ScriptParameter::new_with_type_and_name(Type::Double, SNR_FALLOFF),
            high_pass_nyquist: ScriptParameter::new_with_type_and_name(
                Type::Double,
                HIGH_PASS_NYQUIST,
            ),
            defocus_in_microns: ScriptParameter::new_with_type_and_name(
                Type::Double,
                DEFOCUS_IN_MICRONS,
            ),
            phase_shift: ScriptParameter::new_with_type_and_name(Type::Double, PHASE_SHIFT),
            mode_to_output: ScriptParameter::new_with_type_and_name(Type::Integer, MODE_TO_OUTPUT),
            setup_chunks_if_memory_error: EtomoBoolean2::new_with_name(
                SETUP_CHUNKS_IF_MEMORY_ERROR,
            ),
            manager,
            axis_id,
            reduce_filt_vol_comscript_file: file_type::CLASS
                .reduce_filt_vol_comscript
                .get_file(Some(manager), Some(axis_id)),
            is_reduce_filt_vol_flipped: false,
        };
        param.low_pass_radius_sigma.set_range_by_index(
            LOW_PASS_CUTOFF_INDEX,
            LOW_PASS_CUTOFF_MIN,
            LOW_PASS_CUTOFF_MAX,
        );
        param.low_pass_radius_sigma.set_range_by_index(
            LOW_PASS_SIGMA_INDEX,
            LOW_PASS_SIGMA_MIN,
            f64::INFINITY,
        );
        param
    }

    /// Java `isReductionFactor`.
    pub fn is_reduction_factor(&self) -> bool {
        self.reduction_factor.is()
    }

    /// Java `isZReductionFactor`.
    pub fn is_z_reduction_factor(&self) -> bool {
        self.z_reduction_factor.is()
    }

    /// Java `isLowPassRadiusSigma`.
    pub fn is_low_pass_radius_sigma(&self) -> bool {
        !self.low_pass_radius_sigma.is_empty()
    }

    /// Java `isDeconvolutionStrength`.
    pub fn is_deconvolution_strength(&self) -> bool {
        self.deconvolution_strength.is()
    }

    /// Java `isSNRFalloff`.
    pub fn is_snr_falloff(&self) -> bool {
        self.snr_falloff.is()
    }

    /// Java `isHighPassNyquist`.
    pub fn is_high_pass_nyquist(&self) -> bool {
        self.high_pass_nyquist.is()
    }

    /// Java `isDefocusInMicrons`.
    pub fn is_defocus_in_microns(&self) -> bool {
        self.defocus_in_microns.is()
    }

    /// Java `isPhaseShift`.
    pub fn is_phase_shift(&self) -> bool {
        self.phase_shift.is()
    }

    /// Java `isModeToOutput`.
    pub fn is_mode_to_output(&self) -> bool {
        self.mode_to_output.is()
    }

    /// Java `getInputFile`.
    pub fn get_input_file(&self) -> String {
        self.input_file.to_string()
    }

    /// Java `getOutputFile`.
    pub fn get_output_file(&self) -> String {
        self.output_file.to_string()
    }

    /// Java `getReductionFactor`.
    pub fn get_reduction_factor(&self) -> String {
        self.reduction_factor.to_string()
    }

    /// Java `getZReductionFactor`.
    pub fn get_z_reduction_factor(&self) -> String {
        self.z_reduction_factor.to_string()
    }

    /// Java `getLowPassRadiusSigma`.
    pub fn get_low_pass_radius_sigma(&self) -> String {
        self.low_pass_radius_sigma.to_string()
    }

    /// Java `getDeconvolutionStrength`.
    pub fn get_deconvolution_strength(&self) -> String {
        self.deconvolution_strength.to_string()
    }

    /// Java `getSNRFalloff`.
    pub fn get_snr_falloff(&self) -> String {
        self.snr_falloff.to_string()
    }

    /// Java `getHighPassNyquist`.
    pub fn get_high_pass_nyquist(&self) -> String {
        self.high_pass_nyquist.to_string()
    }

    /// Java `getDefocusInMicrons`.
    pub fn get_defocus_in_microns(&self) -> String {
        self.defocus_in_microns.to_string()
    }

    /// Java `getPhaseShift`.
    pub fn get_phase_shift(&self) -> String {
        self.phase_shift.to_string()
    }

    /// Java `getModeToOutput`.
    pub fn get_mode_to_output(&self) -> String {
        self.mode_to_output.to_string()
    }

    /// Java `isSetupChunksIfMemoryError`.
    pub fn is_setup_chunks_if_memory_error(&self) -> bool {
        self.setup_chunks_if_memory_error.is()
    }

    /// Java `setInputFile(String, boolean)`.
    pub fn set_input_file(&mut self, input: Option<&str>, is_flipped: bool) {
        self.input_file.set(input);
        self.is_reduce_filt_vol_flipped = is_flipped;
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, input: Option<&str>) {
        self.output_file.set(input);
    }

    /// Java `setReductionFactor`.
    pub fn set_reduction_factor(&mut self, input: Option<&str>) {
        self.reduction_factor.set_string(input);
    }

    /// Java `resetReductionFactor`.
    pub fn reset_reduction_factor(&mut self) {
        self.reduction_factor.reset();
    }

    /// Java `setZReductionFactor`.
    pub fn set_z_reduction_factor(&mut self, input: Option<&str>) {
        self.z_reduction_factor.set_string(input);
    }

    /// Java `resetZReductionFactor`.
    pub fn reset_z_reduction_factor(&mut self) {
        self.z_reduction_factor.reset();
    }

    /// Java `setLowPassRadiusSigma(String, boolean)`.  Returns the error message when
    /// validating, else `None`.
    pub fn set_low_pass_radius_sigma(
        &mut self,
        input: Option<&str>,
        do_validation: bool,
    ) -> Option<String> {
        if let Err(except) = self.low_pass_radius_sigma.validate_and_set(input) {
            if do_validation {
                return except.get_message().map(str::to_owned);
            }
            eprintln!(
                "Warning: unable to set lowPassRadiusSigma.  {}",
                except.get_message().unwrap_or("null")
            );
        }
        None
    }

    /// Java `setLowPassCutoffRadius`.
    pub fn set_low_pass_cutoff_radius(&mut self, input: Option<&str>) {
        self.low_pass_radius_sigma.set_index_string(0, input);
    }

    /// Java `setLowPassSigma`.
    pub fn set_low_pass_sigma(&mut self, input: Option<&str>) {
        self.low_pass_radius_sigma.set_index_string(1, input);
    }

    /// Java `resetLowPassRadiusSigma`.
    pub fn reset_low_pass_radius_sigma(&mut self) {
        self.low_pass_radius_sigma.reset();
    }

    /// Java `setDeconvolutionStrength`.
    pub fn set_deconvolution_strength(&mut self, input: Option<&str>) {
        self.deconvolution_strength.set_string(input);
    }

    /// Java `resetDeconvolutionStrength`.
    pub fn reset_deconvolution_strength(&mut self) {
        self.deconvolution_strength.reset();
    }

    /// Java `setSNRFalloff`.
    pub fn set_snr_falloff(&mut self, input: Option<&str>) {
        self.snr_falloff.set_string(input);
    }

    /// Java `resetSNRFalloff`.
    pub fn reset_snr_falloff(&mut self) {
        self.snr_falloff.reset();
    }

    /// Java `setHighPassNyquist`.
    pub fn set_high_pass_nyquist(&mut self, input: Option<&str>) {
        self.high_pass_nyquist.set_string(input);
    }

    /// Java `resetHighPassNyquist`.
    pub fn reset_high_pass_nyquist(&mut self) {
        self.high_pass_nyquist.reset();
    }

    /// Java `setDefocusInMicrons`.
    pub fn set_defocus_in_microns(&mut self, input: Option<&str>) {
        self.defocus_in_microns.set_string(input);
    }

    /// Java `resetDefocusInMicrons`.
    pub fn reset_defocus_in_microns(&mut self) {
        self.defocus_in_microns.reset();
    }

    /// Java `setPhaseShift`.
    pub fn set_phase_shift(&mut self, input: Option<&str>) {
        self.phase_shift.set_string(input);
    }

    /// Java `resetPhaseShift`.
    pub fn reset_phase_shift(&mut self) {
        self.phase_shift.reset();
    }

    /// Java `setModeToOutput`.
    pub fn set_mode_to_output(&mut self, input: Option<&str>) {
        self.mode_to_output.set_string(input);
    }

    /// Java `setSetupChunksIfMemoryError`.
    pub fn set_setup_chunks_if_memory_error(&mut self, input: bool) {
        self.setup_chunks_if_memory_error.set_boolean(input);
    }
}

impl Command for ReduceFiltVolParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(PROCESS_NAME.get_comscript_array(self.axis_id))
    }

    /// Java `new File(reduceFiltVolComscriptFile.getAbsolutePath())`.
    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        self.reduce_filt_vol_comscript_file
            .as_ref()
            .map(|file| std::path::absolute(file).unwrap_or_else(|_| file.clone()))
    }

    /// Java `new File(reduceFiltVolComscriptFile.getAbsolutePath())`.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        self.reduce_filt_vol_comscript_file
            .as_ref()
            .map(|file| std::path::absolute(file).unwrap_or_else(|_| file.clone()))
    }

    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}

impl CommandParam for ReduceFiltVolParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Java calls `scriptCommand.useKeywordValue()` on the command it parses,
        // converting an old-style command in place.  The trait lends the command
        // immutably, so the conversion is made on a copy, which is what is parsed.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        self.input_file.parse(script_command)?;
        self.output_file.parse(script_command)?;
        self.reduction_factor.parse(script_command)?;
        self.z_reduction_factor.parse(script_command)?;
        self.low_pass_radius_sigma
            .validate_and_set_com_script(script_command)?;
        self.deconvolution_strength.parse(script_command)?;
        self.snr_falloff.parse(script_command)?;
        self.high_pass_nyquist.parse(script_command)?;
        self.defocus_in_microns.parse(script_command)?;
        self.phase_shift.parse(script_command)?;
        self.mode_to_output.parse(script_command)?;
        self.setup_chunks_if_memory_error.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.input_file.update_com_script(script_command);
        self.output_file.update_com_script(script_command);
        self.reduction_factor.update_com_script(script_command);
        self.z_reduction_factor.update_com_script(script_command);
        self.low_pass_radius_sigma
            .update_script_parameter(script_command);
        self.deconvolution_strength
            .update_com_script(script_command);
        self.snr_falloff.update_com_script(script_command);
        self.high_pass_nyquist.update_com_script(script_command);
        self.defocus_in_microns.update_com_script(script_command);
        self.phase_shift.update_com_script(script_command);
        self.mode_to_output.update_com_script(script_command);
        self.setup_chunks_if_memory_error
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.input_file.reset();
        self.output_file.reset();
        self.reduction_factor.reset();
        self.z_reduction_factor.reset();
        self.low_pass_radius_sigma.reset();
        self.deconvolution_strength.reset();
        self.snr_falloff.reset();
        self.high_pass_nyquist.reset();
        self.defocus_in_microns.reset();
        self.phase_shift.reset();
        self.mode_to_output.reset();
        self.setup_chunks_if_memory_error.reset();
    }
}

impl Loggable for ReduceFiltVolParam {
    /// Java `getName`, which returns null; the trait's `String` makes that empty.
    fn get_name(&self) -> String {
        String::new()
    }

    /// Java `getLogMessage`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

impl ProcessDetails for ReduceFiltVolParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        Some(0)
    }

    /// A field Java does not recognise throws `IllegalArgumentException`; here that is
    /// `None`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Field>(field) == Some(&Field::IsReduceFiltVolFlipped) {
            return Some(self.is_reduce_filt_vol_flipped);
        }
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        Some(0.0)
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_enough_memory_message_is_the_java_concatenation() {
        assert_eq!(
            NOT_ENOUGH_MEMORY_MSG,
            "The whole volume is too large to filter at once but is ready to be \
             filtered in parallel.\nSelect File - New - Generic Parallel Process then \
             select rfvfilter-001.com for the Process name."
        );
    }
}
