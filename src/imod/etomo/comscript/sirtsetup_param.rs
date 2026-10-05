//! `IMOD/Etomo/src/etomo/comscript/SirtsetupParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `CLEAN_UP_PAST_START_KEY`.
pub const CLEAN_UP_PAST_START_KEY: &str = "CleanUpPastStart";
/// Java `LEAVE_ITERATIONS_KEY`.
pub const LEAVE_ITERATIONS_KEY: &str = "LeaveIterations";
/// Java `RADIUS_AND_SIGMA_KEY`.
pub const RADIUS_AND_SIGMA_KEY: &str = "RadiusAndSigma";
/// Java `RESUME_FROM_ITERATION_KEY`.
pub const RESUME_FROM_ITERATION_KEY: &str = "ResumeFromIteration";
/// Java `SCALE_TO_INTEGER_KEY`.
pub const SCALE_TO_INTEGER_KEY: &str = "ScaleToInteger";
/// Java `START_FROM_ZERO_KEY`.
pub const START_FROM_ZERO_KEY: &str = "StartFromZero";
/// Java `SUBAREA_SIZE_KEY`.
pub const SUBAREA_SIZE_KEY: &str = "SubareaSize";
/// Java `Y_OFFSET_OF_SUBAREA_KEY`.
pub const Y_OFFSET_OF_SUBAREA_KEY: &str = "YOffsetOfSubarea";
/// Java `FLAT_FILTER_FRACTION_KEY`.
pub const FLAT_FILTER_FRACTION_KEY: &str = "FlatFilterFraction";
/// Java `SKIP_VERT_SLICE_OUTPUT_KEY`.
pub const SKIP_VERT_SLICE_OUTPUT_KEY: &str = "SkipVertSliceOutput";

/// Java nested class `SirtsetupParam.Field`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `Field.SUBAREA`.
    Subarea,
    /// Java `Field.SUBAREA_SIZE`.
    SubareaSize,
    /// Java `Field.Y_OFFSET_OF_SUBSET`.
    YOffsetOfSubset,
}

impl FieldInterface for Field {}

/// Java final `SirtsetupParam`.
pub struct SirtsetupParam {
    command_file: StringParameter,
    number_of_processors: ScriptParameter,
    start_from_zero: EtomoBoolean2,
    resume_from_iteration: ScriptParameter,
    leave_iterations: StringParameter,
    radius_and_sigma: FortranInputString,
    subarea_size: FortranInputString,
    y_offset_of_subarea: ScriptParameter,
    scale_to_integer: FortranInputString,
    clean_up_past_start: EtomoBoolean2,
    flat_filter_fraction: ScriptParameter,
    skip_vert_slice_output: EtomoBoolean2,
    falloff_is_true_sigma: EtomoBoolean2,
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
}

impl SirtsetupParam {
    /// Java `SirtsetupParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> SirtsetupParam {
        let mut param = SirtsetupParam {
            command_file: StringParameter::new("CommandFile"),
            number_of_processors: ScriptParameter::new_with_name("NumberOfProcessors"),
            start_from_zero: EtomoBoolean2::new_with_name(START_FROM_ZERO_KEY),
            resume_from_iteration: ScriptParameter::new_with_name(RESUME_FROM_ITERATION_KEY),
            leave_iterations: StringParameter::new(LEAVE_ITERATIONS_KEY),
            radius_and_sigma: FortranInputString::new_with_key(Some(RADIUS_AND_SIGMA_KEY), 2),
            subarea_size: FortranInputString::new_with_key(Some(SUBAREA_SIZE_KEY), 2),
            y_offset_of_subarea: ScriptParameter::new_with_name(Y_OFFSET_OF_SUBAREA_KEY),
            scale_to_integer: FortranInputString::new_with_key(Some(SCALE_TO_INTEGER_KEY), 2),
            clean_up_past_start: EtomoBoolean2::new_with_name(CLEAN_UP_PAST_START_KEY),
            flat_filter_fraction: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FLAT_FILTER_FRACTION_KEY,
            ),
            skip_vert_slice_output: EtomoBoolean2::new_with_name(SKIP_VERT_SLICE_OUTPUT_KEY),
            falloff_is_true_sigma: EtomoBoolean2::new_with_name("FalloffIsTrueSigma"),
            axis_id,
            manager,
        };
        param.subarea_size.set_integer_type(true);
        param.scale_to_integer.set_integer_type(false);
        param
    }

    /// Java `setNumberOfProcessors`.
    pub fn set_number_of_processors(&mut self, input: Option<&str>) {
        self.number_of_processors.set_string(input);
    }

    /// Java `setStartFromZero`.
    pub fn set_start_from_zero(&mut self, input: bool) {
        self.start_from_zero.set_boolean(input);
    }

    /// Java `setSkipVertSliceOutput`.
    pub fn set_skip_vert_slice_output(&mut self, input: bool) {
        self.skip_vert_slice_output.set_boolean(input);
    }

    /// Java `isStartFromZero`.
    pub fn is_start_from_zero(&self) -> bool {
        self.start_from_zero.is()
    }

    /// Java `isSkipVertSliceOutput`.
    pub fn is_skip_vert_slice_output(&self) -> bool {
        self.skip_vert_slice_output.is()
    }

    /// Java `resetResumeFromIteration`.
    pub fn reset_resume_from_iteration(&mut self) {
        self.resume_from_iteration.reset();
    }

    /// Java `isResumeFromIterationNull`.
    pub fn is_resume_from_iteration_null(&self) -> bool {
        self.resume_from_iteration.is_null()
    }

    /// Java `getLeaveIterations`.
    pub fn get_leave_iterations(&self) -> String {
        self.leave_iterations.to_string()
    }

    /// Java `getRadiusAndSigma(int)`.
    pub fn get_radius_and_sigma(&self, index: i32) -> String {
        self.radius_and_sigma
            .to_string_index_default_is_blank(index, true)
    }

    // Updates done

    /// Java `setRadiusAndSigma(int, String)`.
    pub fn set_radius_and_sigma(&mut self, index: i32, input: Option<&str>) {
        self.radius_and_sigma.set_index_string(index, input);
    }

    /// Java `getSubareaSize`.
    pub fn get_subarea_size(&self) -> String {
        self.subarea_size.to_string()
    }

    /// Java `getYOffsetOfSubarea`.
    pub fn get_y_offset_of_subarea(&self) -> String {
        self.y_offset_of_subarea.to_string()
    }

    /// Java `isYOffsetOfSubareaNull`.
    pub fn is_y_offset_of_subarea_null(&self) -> bool {
        self.y_offset_of_subarea.is_null()
    }

    /// Java `isScaleToIntegerNull`.
    pub fn is_scale_to_integer_null(&self) -> bool {
        self.scale_to_integer.is_null()
    }

    /// Java `isSubareaSizeNull`.
    pub fn is_subarea_size_null(&self) -> bool {
        self.subarea_size.is_null()
    }

    /// Java `isCleanUpPastStart`.
    pub fn is_clean_up_past_start(&self) -> bool {
        self.clean_up_past_start.is()
    }

    /// Java `isFalloffIsTrueSigma`.
    pub fn is_falloff_is_true_sigma(&self) -> bool {
        self.falloff_is_true_sigma.is()
    }

    /// Java `resetSubareaSize`.
    pub fn reset_subarea_size(&mut self) {
        self.subarea_size.reset();
    }

    /// Java `resetYOffsetOfSubarea`.
    pub fn reset_y_offset_of_subarea(&mut self) {
        self.y_offset_of_subarea.reset();
    }

    /// Java `setCleanUpPastStart`.
    pub fn set_clean_up_past_start(&mut self, input: bool) {
        self.clean_up_past_start.set_boolean(input);
    }

    /// Java `setLeaveIterations`.
    pub fn set_leave_iterations(&mut self, input: Option<&str>) {
        self.leave_iterations.set(input);
    }

    /// Java `setResume`.
    pub fn set_resume(&mut self, resume: bool) {
        if !resume {
            let name = file_type::CLASS
                .tilt_comscript
                .get_file_name(Some(self.manager), Some(self.axis_id));
            self.command_file.set(name.as_deref());
        } else {
            let name = file_type::CLASS
                .tilt_for_sirt_comscript
                .get_file_name(Some(self.manager), Some(self.axis_id));
            self.command_file.set(name.as_deref());
        }
    }

    /// Java `setResumeFromIteration`.
    pub fn set_resume_from_iteration(&mut self, input: Option<&ConstEtomoNumber>) {
        self.resume_from_iteration.set_const_etomo_number(input);
    }

    /// Java `setScaleToInteger`.
    pub fn set_scale_to_integer(&mut self, input: bool) {
        if input {
            self.scale_to_integer.set_index_double(0, -20000.0);
            self.scale_to_integer.set_index_double(1, 20000.0);
        } else {
            self.scale_to_integer.reset();
        }
    }

    /// Java `setSubareaSize`.
    pub fn set_subarea_size(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.subarea_size.validate_and_set(input)
    }

    /// Java `setYOffsetOfSubarea`.
    pub fn set_y_offset_of_subarea(&mut self, input: Option<&str>) {
        self.y_offset_of_subarea.set_string(input);
    }

    /// Java `setFlatFilterFraction`.
    pub fn set_flat_filter_fraction(&mut self, input: Option<&str>) {
        self.flat_filter_fraction.set_string(input);
    }

    /// Java `getFlatFilterFraction`.
    pub fn get_flat_filter_fraction(&self) -> String {
        self.flat_filter_fraction.to_string()
    }
}

impl CommandParam for SirtsetupParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // reset
        // commandFile is read-only
        self.start_from_zero.reset();
        self.resume_from_iteration.reset();
        self.leave_iterations.reset();
        self.subarea_size.reset();
        self.y_offset_of_subarea.reset();
        self.scale_to_integer.reset();
        self.clean_up_past_start.reset();
        self.flat_filter_fraction.reset();
        self.skip_vert_slice_output.reset();
        self.falloff_is_true_sigma.reset();
        // parse
        // commandFile is read-only
        self.start_from_zero.parse(script_command)?;
        self.resume_from_iteration.parse(script_command)?;
        self.leave_iterations.parse(script_command)?;
        self.radius_and_sigma
            .validate_and_set_com_script(script_command)?;
        self.subarea_size
            .validate_and_set_com_script(script_command)?;
        self.y_offset_of_subarea.parse(script_command)?;
        self.scale_to_integer
            .validate_and_set_com_script(script_command)?;
        self.clean_up_past_start.parse(script_command)?;
        self.flat_filter_fraction.parse(script_command)?;
        self.skip_vert_slice_output.parse(script_command)?;
        self.falloff_is_true_sigma.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.command_file.update_com_script(script_command);
        self.number_of_processors.update_com_script(script_command);
        self.start_from_zero.update_com_script(script_command);
        self.resume_from_iteration.update_com_script(script_command);
        self.leave_iterations.update_com_script(script_command);
        self.radius_and_sigma
            .update_script_parameter(script_command);
        self.subarea_size.update_script_parameter(script_command);
        self.y_offset_of_subarea.update_com_script(script_command);
        self.scale_to_integer
            .update_script_parameter(script_command);
        self.clean_up_past_start.update_com_script(script_command);
        self.flat_filter_fraction.update_com_script(script_command);
        self.skip_vert_slice_output
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl Command for SirtsetupParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command(&self) -> Option<String> {
        Some(ProcessName::SIRTSETUP.get_comscript(self.axis_id))
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(ProcessName::SIRTSETUP.get_comscript_array(self.axis_id))
    }

    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        file_type::CLASS
            .tilt_comscript
            .get_file(Some(self.manager), Some(self.axis_id))
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_command_name(&self) -> Option<String> {
        // In this case the .com file name and the command in the .com file are the same.
        Some(ProcessName::SIRTSETUP.to_string())
    }

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

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::SIRTSETUP)
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

    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for SirtsetupParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        ProcessName::SIRTSETUP.to_string()
    }

    /// Java `getLogMessage`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Every field Java does not recognise throws `IllegalArgumentException("field=" +
/// field)`; here that is `None`.
impl ProcessDetails for SirtsetupParam {
    fn get_boolean_value(&self, field_interface: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Field>(field_interface) == Some(&Field::Subarea) {
            return Some(!self.subarea_size.is_null());
        }
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }

    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Field>(field) == Some(&Field::YOffsetOfSubset) {
            return Some(self.y_offset_of_subarea.get_int());
        }
        None
    }

    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }

    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        if field_interface::as_field::<Field>(field) == Some(&Field::SubareaSize) {
            return Some(self.subarea_size.to_string_default_is_blank(true));
        }
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }
}
