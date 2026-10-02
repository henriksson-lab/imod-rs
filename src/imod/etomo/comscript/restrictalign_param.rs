//! `IMOD/Etomo/src/etomo/comscript/RestrictalignParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::RESTRICTALIGN;
/// Java `TARGET_MEASUREMENT_RATIO_KEY`.
pub const TARGET_MEASUREMENT_RATIO_KEY: &str = "TargetMeasurementRatio";
/// Java `MIN_MEASUREMENT_RATIO_KEY`.
pub const MIN_MEASUREMENT_RATIO_KEY: &str = "MinMeasurementRatio";
/// Java `ORDER_OF_RESTRICTIONS_KEY`.
pub const ORDER_OF_RESTRICTIONS_KEY: &str = "OrderOfRestrictions";
/// Java `SKIP_BEAM_TILT_WITH_ONE_ROT_KEY`.
pub const SKIP_BEAM_TILT_WITH_ONE_ROT_KEY: &str = "SkipBeamTiltWithOneRot";
/// Java `LOCAL_ALIGN_VALIDATION_KEY`.
pub const LOCAL_ALIGN_VALIDATION_KEY: &str = "LocalAlignValidation";

/// Java final `RestrictalignParam`.
pub struct RestrictalignParam {
    target_measurement_ratio: ScriptParameter,
    min_measurement_ratio: ScriptParameter,
    skip_beam_tilt_with_one_rot: EtomoBoolean2,
    local_align_validation: ScriptParameter,
    /// Java `manager`, read only by the constructor.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    align_comscript_file: Option<std::path::PathBuf>,
    /// Java `commandArray`, initialised to null and never assigned by the source.
    #[allow(dead_code)]
    command_array: Option<Vec<String>>,
}

impl RestrictalignParam {
    /// Java `RestrictalignParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> RestrictalignParam {
        RestrictalignParam {
            target_measurement_ratio: ScriptParameter::new_with_type_and_name(
                Type::Double,
                TARGET_MEASUREMENT_RATIO_KEY,
            ),
            min_measurement_ratio: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MIN_MEASUREMENT_RATIO_KEY,
            ),
            skip_beam_tilt_with_one_rot: EtomoBoolean2::new_with_name(
                SKIP_BEAM_TILT_WITH_ONE_ROT_KEY,
            ),
            local_align_validation: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                LOCAL_ALIGN_VALIDATION_KEY,
            ),
            manager,
            axis_id,
            align_comscript_file: file_type::CLASS
                .align_comscript
                .get_file(Some(manager), Some(axis_id)),
            command_array: None,
        }
    }

    /// Java `setTargetMeasurementRatio`.
    pub fn set_target_measurement_ratio(&mut self, input: Option<&str>) {
        self.target_measurement_ratio.set_string(input);
    }

    /// Java `resetTargetMeasurementRatio`.
    pub fn reset_target_measurement_ratio(&mut self) {
        self.target_measurement_ratio.reset();
    }

    /// Java `resetMinMeasurementRatio`.
    pub fn reset_min_measurement_ratio(&mut self) {
        self.min_measurement_ratio.reset();
    }

    /// Java `setMinMeasurementRatio`.
    pub fn set_min_measurement_ratio(&mut self, input: Option<&str>) {
        self.min_measurement_ratio.set_string(input);
    }

    /// Java `setLocalAlignValidation(ConstEtomoNumber)`.
    pub fn set_local_align_validation(&mut self, input: Option<&ConstEtomoNumber>) {
        self.local_align_validation.set_const_etomo_number(input);
    }

    /// Java `resetLocalAlignValidation`.
    pub fn reset_local_align_validation(&mut self) {
        self.local_align_validation.reset();
    }

    /// Java `setSkipBeamTiltWithOneRot(String)`.
    pub fn set_skip_beam_tilt_with_one_rot(&mut self, input: Option<&str>) {
        self.skip_beam_tilt_with_one_rot.set_string(input);
    }

    /// Java `isTargetMeasurementRatio`.
    pub fn is_target_measurement_ratio(&self) -> bool {
        self.target_measurement_ratio.is()
    }

    /// Java `getTargetMeasurementRatio`.
    pub fn get_target_measurement_ratio(&self) -> String {
        self.target_measurement_ratio.to_string()
    }

    /// Java `isMinMeasurementRatio`.
    pub fn is_min_measurement_ratio(&self) -> bool {
        self.min_measurement_ratio.is()
    }

    /// Java `getMinMeasurementRatio`.
    pub fn get_min_measurement_ratio(&self) -> String {
        self.min_measurement_ratio.to_string()
    }

    /// Java `isSkipBeamTiltWithOneRot`.
    pub fn is_skip_beam_tilt_with_one_rot(&self) -> bool {
        self.skip_beam_tilt_with_one_rot.is()
    }

    /// Java `isLocalAlignValidation`.
    pub fn is_local_align_validation(&self) -> bool {
        self.local_align_validation.is()
    }

    /// Java `getLocalAlignValidation`.
    pub fn get_local_align_validation(&self) -> String {
        self.local_align_validation.to_string()
    }
}

impl CommandParam for RestrictalignParam {
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
        self.target_measurement_ratio.parse(script_command)?;
        self.min_measurement_ratio.parse(script_command)?;
        self.skip_beam_tilt_with_one_rot.parse(script_command)?;
        self.local_align_validation.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.target_measurement_ratio
            .update_com_script(script_command);
        self.min_measurement_ratio.update_com_script(script_command);
        self.skip_beam_tilt_with_one_rot
            .update_com_script(script_command);
        self.local_align_validation
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.target_measurement_ratio.reset();
        self.min_measurement_ratio.reset();
        self.skip_beam_tilt_with_one_rot.reset();
        self.local_align_validation.reset();
    }
}

impl Command for RestrictalignParam {
    /// Creates and returns the command array.  Command array will not be changed after
    /// this call.  Not thread safe.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(PROCESS_NAME.get_comscript_array(self.axis_id))
    }

    /// Java returns `AxisID.ONLY` regardless of the constructor's axis.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    /// Java `new File(alignComscriptFile.getAbsolutePath())`.
    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        self.align_comscript_file
            .as_ref()
            .map(|file| std::path::absolute(file).unwrap_or_else(|_| file.clone()))
    }

    /// Not for running the command.  Use getCommandArray.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `new File(alignComscriptFile.getAbsolutePath())`.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        self.align_comscript_file
            .as_ref()
            .map(|file| std::path::absolute(file).unwrap_or_else(|_| file.clone()))
    }

    /// Deprecated 3/15/2019.
    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Deprecated 3/15/2019.
    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }
}
