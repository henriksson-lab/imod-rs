//! `IMOD/Etomo/src/etomo/comscript/MultifiltSetupParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::MULTIFILT_SETUP;
/// Java `FAKE_SIRT_ITERATIONS`.
pub const FAKE_SIRT_ITERATIONS: &str = "FakeSIRTiterations";
/// Java `EXACT_OBJECT_SIZES`.
pub const EXACT_OBJECT_SIZES: &str = "ExactObjectSizes";
/// Java `GAUSSIAN_CUTOFFS`.
pub const GAUSSIAN_CUTOFFS: &str = "GaussianCutoffs";
/// Java `GAUSSIAN_FALLOFFS`.
pub const GAUSSIAN_FALLOFFS: &str = "GaussianFalloffs";
/// Java `HAMMING_LIKE_STARTS`.
pub const HAMMING_LIKE_STARTS: &str = "HammingLikeStarts";
/// Java `WIDTH_IN_X`.
pub const WIDTH_IN_X: &str = "WidthInX";
/// Java `SHIFT_IN_X`.
pub const SHIFT_IN_X: &str = "ShiftInX";
/// Java `SIZE_IN_Y`.
pub const SIZE_IN_Y: &str = "SizeInY";
/// Java `SHIFT_IN_Y`.
pub const SHIFT_IN_Y: &str = "ShiftInY";
/// Java `THICKNESS_IN_Z`.
pub const THICKNESS_IN_Z: &str = "ThicknessInZ";
/// Java `SHIFT_IN_DEPTH`.
pub const SHIFT_IN_DEPTH: &str = "ShiftInDepth";

/// Java final `MultifiltSetupParam`.
pub struct MultifiltSetupParam {
    command_file: StringParameter,
    fake_sirt_iterations: StringParameter,
    exact_object_sizes: StringParameter,
    gaussian_cutoffs: StringParameter,
    gaussian_falloffs: StringParameter,
    hamming_like_starts: StringParameter,
    width_in_x: ScriptParameter,
    shift_in_x: ScriptParameter,
    size_in_y: ScriptParameter,
    shift_in_y: ScriptParameter,
    thickness_in_z: ScriptParameter,
    shift_in_depth: ScriptParameter,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl MultifiltSetupParam {
    /// Java `MultifiltSetupParam(BaseManager, AxisID)`.  The source tests
    /// `axisID != null`; every Rust caller passes an axis, so the non-null branch is
    /// the only one.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> MultifiltSetupParam {
        let mut param = MultifiltSetupParam {
            command_file: StringParameter::new("CommandFile"),
            fake_sirt_iterations: StringParameter::new(FAKE_SIRT_ITERATIONS),
            exact_object_sizes: StringParameter::new(EXACT_OBJECT_SIZES),
            gaussian_cutoffs: StringParameter::new(GAUSSIAN_CUTOFFS),
            gaussian_falloffs: StringParameter::new(GAUSSIAN_FALLOFFS),
            hamming_like_starts: StringParameter::new(HAMMING_LIKE_STARTS),
            width_in_x: ScriptParameter::new_with_type_and_name(Type::Integer, WIDTH_IN_X),
            shift_in_x: ScriptParameter::new_with_type_and_name(Type::Integer, SHIFT_IN_X),
            size_in_y: ScriptParameter::new_with_type_and_name(Type::Integer, SIZE_IN_Y),
            shift_in_y: ScriptParameter::new_with_type_and_name(Type::Integer, SHIFT_IN_Y),
            thickness_in_z: ScriptParameter::new_with_type_and_name(Type::Integer, THICKNESS_IN_Z),
            shift_in_depth: ScriptParameter::new_with_type_and_name(Type::Integer, SHIFT_IN_DEPTH),
            manager,
            axis_id,
        };
        // `ProcessName.TILT.getText() + axisID.getExtension()`: Java concatenation
        // writes a null text as "null"
        let command_file = format!(
            "{}{}",
            ProcessName::TILT.get_text().unwrap_or("null"),
            axis_id.get_extension()
        );
        param.command_file.set(Some(&command_file));
        param
    }

    /// Java `isFakeSIRTiterations`.
    pub fn is_fake_sirt_iterations(&self) -> bool {
        !self.fake_sirt_iterations.is_empty()
    }

    /// Java `isExactObjectSizes`.
    pub fn is_exact_object_sizes(&self) -> bool {
        !self.exact_object_sizes.is_empty()
    }

    /// Java `isGaussianCutoffs`.
    pub fn is_gaussian_cutoffs(&self) -> bool {
        !self.gaussian_cutoffs.is_empty()
    }

    /// Java `isGaussianFalloffs`.
    pub fn is_gaussian_falloffs(&self) -> bool {
        !self.gaussian_falloffs.is_empty()
    }

    /// Java `isHammingLikeStarts`.
    pub fn is_hamming_like_starts(&self) -> bool {
        !self.hamming_like_starts.is_empty()
    }

    /// Java `getFakeSIRTiterations`.
    pub fn get_fake_sirt_iterations(&self) -> String {
        self.fake_sirt_iterations.to_string()
    }

    /// Java `getExactObjectSizes`.
    pub fn get_exact_object_sizes(&self) -> String {
        self.exact_object_sizes.to_string()
    }

    /// Java `getGaussianCutoffs`.
    pub fn get_gaussian_cutoffs(&self) -> String {
        self.gaussian_cutoffs.to_string()
    }

    /// Java `getGaussianFalloffs`.
    pub fn get_gaussian_falloffs(&self) -> String {
        self.gaussian_falloffs.to_string()
    }

    /// Java `getHammingLikeStarts`.
    pub fn get_hamming_like_starts(&self) -> String {
        self.hamming_like_starts.to_string()
    }

    /// Java `getWidthInX`.
    pub fn get_width_in_x(&self) -> String {
        self.width_in_x.to_string()
    }

    /// Java `getShiftInX`.
    pub fn get_shift_in_x(&self) -> String {
        self.shift_in_x.to_string()
    }

    /// Java `getSizeInY`.
    pub fn get_size_in_y(&self) -> String {
        self.size_in_y.to_string()
    }

    /// Java `getShiftInY`.
    pub fn get_shift_in_y(&self) -> String {
        self.shift_in_y.to_string()
    }

    /// Java `getThicknessInZ`.
    pub fn get_thickness_in_z(&self) -> String {
        self.thickness_in_z.to_string()
    }

    /// Java `getShiftInDepth`.
    pub fn get_shift_in_depth(&self) -> String {
        self.shift_in_depth.to_string()
    }

    /// Java `setFakeSIRTiterations`.
    pub fn set_fake_sirt_iterations(&mut self, input: Option<&str>) {
        self.fake_sirt_iterations.set(input);
    }

    /// Java `resetFakeSIRTiterations`.
    pub fn reset_fake_sirt_iterations(&mut self) {
        self.fake_sirt_iterations.reset();
    }

    /// Java `setExactObjectSizes`.
    pub fn set_exact_object_sizes(&mut self, input: Option<&str>) {
        self.exact_object_sizes.set(input);
    }

    /// Java `resetExactObjectSizes`.
    pub fn reset_exact_object_sizes(&mut self) {
        self.exact_object_sizes.reset();
    }

    /// Java `setGaussianCutoffs`.
    pub fn set_gaussian_cutoffs(&mut self, input: Option<&str>) {
        self.gaussian_cutoffs.set(input);
    }

    /// Java `resetGaussianCutoffs`.
    pub fn reset_gaussian_cutoffs(&mut self) {
        self.gaussian_cutoffs.reset();
    }

    /// Java `setGaussianFalloffs`.
    pub fn set_gaussian_falloffs(&mut self, input: Option<&str>) {
        self.gaussian_falloffs.set(input);
    }

    /// Java `resetGaussianFalloffs`.
    pub fn reset_gaussian_falloffs(&mut self) {
        self.gaussian_falloffs.reset();
    }

    /// Java `setHammingLikeStarts`.
    pub fn set_hamming_like_starts(&mut self, input: Option<&str>) {
        self.hamming_like_starts.set(input);
    }

    /// Java `resetHammingLikeStarts`.
    pub fn reset_hamming_like_starts(&mut self) {
        self.hamming_like_starts.reset();
    }

    /// Java `setWidthInX`.
    pub fn set_width_in_x(&mut self, input: Option<&str>) {
        self.width_in_x.set_string(input);
    }

    /// Java `setShiftInX`.
    pub fn set_shift_in_x(&mut self, input: Option<&str>) {
        self.shift_in_x.set_string(input);
    }

    /// Java `setSizeInY`.
    pub fn set_size_in_y(&mut self, input: Option<&str>) {
        self.size_in_y.set_string(input);
    }

    /// Java `setShiftInY`.
    pub fn set_shift_in_y(&mut self, input: Option<&str>) {
        self.shift_in_y.set_string(input);
    }

    /// Java `setThicknessInZ`.
    pub fn set_thickness_in_z(&mut self, input: Option<&str>) {
        self.thickness_in_z.set_string(input);
    }

    /// Java `setShiftInDepth`.
    pub fn set_shift_in_depth(&mut self, input: Option<&str>) {
        self.shift_in_depth.set_string(input);
    }
}

impl CommandParam for MultifiltSetupParam {
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
        // commandFile is read-only
        self.fake_sirt_iterations.parse(script_command)?;
        self.exact_object_sizes.parse(script_command)?;
        self.gaussian_cutoffs.parse(script_command)?;
        self.gaussian_falloffs.parse(script_command)?;
        self.hamming_like_starts.parse(script_command)?;
        self.width_in_x.parse(script_command)?;
        self.shift_in_x.parse(script_command)?;
        self.size_in_y.parse(script_command)?;
        self.shift_in_y.parse(script_command)?;
        self.thickness_in_z.parse(script_command)?;
        self.shift_in_depth.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.command_file.update_com_script(script_command);
        self.fake_sirt_iterations.update_com_script(script_command);
        self.exact_object_sizes.update_com_script(script_command);
        self.gaussian_cutoffs.update_com_script(script_command);
        self.gaussian_falloffs.update_com_script(script_command);
        self.hamming_like_starts.update_com_script(script_command);
        self.width_in_x.update_com_script(script_command);
        self.shift_in_x.update_com_script(script_command);
        self.size_in_y.update_com_script(script_command);
        self.shift_in_y.update_com_script(script_command);
        self.thickness_in_z.update_com_script(script_command);
        self.shift_in_depth.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        // commandFile is read-only
        self.fake_sirt_iterations.reset();
        self.exact_object_sizes.reset();
        self.gaussian_cutoffs.reset();
        self.gaussian_falloffs.reset();
        self.hamming_like_starts.reset();
        self.width_in_x.reset();
        self.shift_in_x.reset();
        self.size_in_y.reset();
        self.shift_in_y.reset();
        self.thickness_in_z.reset();
        self.shift_in_depth.reset();
    }
}

impl Command for MultifiltSetupParam {
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
        PROCESS_NAME.get_text().map(str::to_owned)
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(PROCESS_NAME.get_comscript_array(self.axis_id))
    }

    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        file_type::CLASS
            .tilt_comscript
            .get_file(Some(self.manager), Some(self.axis_id))
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Returning null because there are multiple types of files.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        None
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
