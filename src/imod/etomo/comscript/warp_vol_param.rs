//! `IMOD/Etomo/src/etomo/comscript/WarpVolParam.java`.
//!
//! flattenwarp.  Could be made more general by allowing the input and output
//! files to be set.

use std::path::Path;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_warp_vol_param::ConstWarpVolParam;
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::FortranInputString;
use super::param_utilities;
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_value_of, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `INPUT_FILE_OPTION`.
pub const INPUT_FILE_OPTION: &str = "InputFile";
/// Java `TEMPORARY_DIRECTORY_OPTION`.
pub const TEMPORARY_DIRECTORY_OPTION: &str = "TemporaryDirectory";
/// Java `OUTPUT_SIZE_X_Y_Z_OPTION`.
pub const OUTPUT_SIZE_X_Y_Z_OPTION: &str = "OutputSizeXYZ";
/// Java `INTERPOLATION_ORDER_OPTION`.
pub const INTERPOLATION_ORDER_OPTION: &str = "InterpolationOrder";
/// Java package-private `COMMAND`.
pub(crate) const COMMAND: &str = "warpvol";

/// Java package-private nested class `WarpVolParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `Mode.POST_PROCESSING`.
    PostProcessing,
    /// Java `Mode.TOOLS`.
    Tools,
}

/// The Java class declares no `toString`, so it prints `Object.toString()`
/// (class name and identity hash); the variant name stands in for it.
impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::PostProcessing => f.write_str("POST_PROCESSING"),
            Mode::Tools => f.write_str("TOOLS"),
        }
    }
}

impl CommandMode for Mode {}

/// Java nested class `WarpVolParam.Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `Field.IS_FLATTEN_FLIPPED`.
    IsFlattenFlipped,
}

impl FieldInterface for Field {}

/// Java final `WarpVolParam implements ConstWarpVolParam, CommandParam,
/// ProcessDetails`.
pub struct WarpVolParam {
    /// Java `command`: declared and never used.
    command: Vec<String>,
    input_file: StringParameter,
    // Updates done
    pub(crate) output_file: StringParameter,
    /// optional
    pub(crate) temporary_directory: StringParameter,
    /// optional
    pub(crate) output_size_xyz: FortranInputString,
    /// optional
    pub(crate) interpolation_order: ScriptParameter,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    mode: Option<Mode>,
    is_flatten_flipped: bool,
}

impl WarpVolParam {
    /// Java package-private `WarpVolParam(BaseManager, AxisID, CommandMode)`.
    /// The Java compares the mode with `Mode`'s two instances only, so any other
    /// `CommandMode` behaves as `None`.
    pub(crate) fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        mode: Option<Mode>,
    ) -> WarpVolParam {
        let mut instance = WarpVolParam {
            command: Vec::new(),
            input_file: StringParameter::new(INPUT_FILE_OPTION),
            output_file: StringParameter::new("OutputFile"),
            temporary_directory: StringParameter::new(TEMPORARY_DIRECTORY_OPTION),
            output_size_xyz: FortranInputString::new_with_key(Some(OUTPUT_SIZE_X_Y_Z_OPTION), 3),
            interpolation_order: ScriptParameter::new_with_name(INTERPOLATION_ORDER_OPTION),
            manager,
            axis_id,
            mode,
            is_flatten_flipped: false,
        };
        instance.output_size_xyz.set_integer_type(true);
        instance.reset();
        instance.is_flatten_flipped = false;
        instance
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.output_size_xyz.set_default();
        self.interpolation_order.reset();
    }

    /// Java `setTemporaryDirectory`.
    pub fn set_temporary_directory(&mut self, input: Option<&str>) {
        self.temporary_directory.set(input);
    }

    /// Java `setInputFile(ImageFileType)`.
    pub fn set_input_file(
        &mut self,
        image_file_type: &crate::imod::etomo::r#type::image_file_type::ImageFileType,
    ) {
        self.input_file
            .set(image_file_type.get_file_name(self.manager).as_deref());
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, input: Option<&str>) {
        self.output_file.set(input);
    }

    /// Java `setInputFile(File)`.
    pub fn set_input_file_file(&mut self, file: &Path) {
        let absolute_path = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
        self.input_file.set(Some(&absolute_path));
    }

    /// Java `setOutputSizeZ`.  Assign number to outputSizeXYZ at index 2.
    /// Returns an error message if number is not blank or a number.
    pub fn set_output_size_z(&mut self, number: Option<&str>) -> Option<String> {
        // `outputSizeXYZ.set(2, number)` throws `NumberFormatException` from
        // `Double.valueOf`, which is caught here; the value is checked the same way
        // before setting so the error comes back as the Java message.
        let blank = match number {
            None => true,
            Some(number) => java_lang_string_matches_whitespace(number),
        };
        if !blank {
            if let Err(message) = java_lang_double_value_of(number.unwrap()) {
                return Some(message);
            }
        }
        self.output_size_xyz.set_index_string(2, number);
        None
    }

    /// Java `setInterpolationOrderLinear`.  If input is true, set
    /// interpolationOrder to 1, otherwise default it.
    pub fn set_interpolation_order_linear(&mut self, input: bool) {
        if input {
            self.interpolation_order.set_int(1);
        } else {
            self.interpolation_order.reset();
        }
    }

    /// Java private `getComscriptFileType`.
    fn get_comscript_file_type(&self) -> Option<Arc<FileType>> {
        if self.mode == Some(Mode::PostProcessing) {
            return Some(file_type::CLASS.flatten_comscript.clone());
        }
        if self.mode == Some(Mode::Tools) {
            return Some(file_type::CLASS.flatten_tool_comscript.clone());
        }
        None
    }
}

impl CommandParam for WarpVolParam {
    /// Java `parseComScriptCommand`.  Get parameter values that may be displayed
    /// on the screen.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        self.temporary_directory.parse(script_command)?;
        self.output_size_xyz
            .validate_and_set_com_script(script_command)?;
        self.interpolation_order.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update all parameter values.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.input_file.update_com_script(script_command);
        self.temporary_directory.update_com_script(script_command);
        self.output_file.update_com_script(script_command);
        param_utilities::update_script_parameter_string_required(
            script_command,
            Some("TransformFile"),
            Some(&dataset_files::get_flatten_warp_output_name(self.manager)),
            true,
        )?;
        self.output_size_xyz.update_script_parameter(script_command);
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some("SameSizeAsInput"),
            true,
        );
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.reset();
    }
}

impl ConstWarpVolParam for WarpVolParam {
    /// Java `getTemporaryDirectory`.
    fn get_temporary_directory(&self) -> String {
        self.temporary_directory.to_string()
    }

    /// Java `getOutputSizeZ`.
    fn get_output_size_z(&self) -> String {
        if self.output_size_xyz.is_null_index(2) {
            return String::new();
        }
        self.output_size_xyz.to_string_index(2)
    }

    /// Java `isInterpolationOrderLinear`.
    fn is_interpolation_order_linear(&self) -> bool {
        self.interpolation_order.equals_int(1)
    }
}

impl Command for WarpVolParam {
    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandMode`: null.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::FLATTEN)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        let comscript_file_type = self.get_comscript_file_type();
        if let Some(comscript_file_type) = comscript_file_type {
            return comscript_file_type.get_file_name(Some(self.manager), Some(self.axis_id));
        }
        None
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::FLATTEN.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `getCommandArray`: `{ getCommandLine() }`.  A null command line is a
    /// one-element array holding null in the Java; it is dropped here.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let array: Vec<String> = self.get_command_line().into_iter().collect();
        Some(array)
    }

    /// Java `getCommandInputFile`: null.
    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Java `getCommandOutputFile`: null.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        if self.mode == Some(Mode::PostProcessing) {
            return Some(file_type::CLASS.flatten_output.clone());
        }
        if self.mode == Some(Mode::Tools) {
            return Some(file_type::CLASS.flatten_tool_output.clone());
        }
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if self.mode == Some(Mode::PostProcessing) {
            return Some(FileKey::clone(&file_type::CLASS.flatten_output));
        }
        if self.mode == Some(Mode::Tools) {
            return Some(FileKey::clone(&file_type::CLASS.flatten_tool_output));
        }
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019): null.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`: null.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName`: null.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails`: null.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// The `ProcessDetails` half of this class.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for WarpVolParam {
    /// Java `getName`: an auto-generated stub returning null; the empty string
    /// stands in for it.
    fn get_name(&self) -> String {
        String::new()
    }

    /// Java `getLogMessage`: an auto-generated stub returning null (no message).
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

impl ProcessDetails for WarpVolParam {
    /// Java `getIntValue`: an auto-generated stub returning 0.
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        Some(0)
    }

    /// Java `getBooleanValue`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Field>(field) == Some(&Field::IsFlattenFlipped) {
            return Some(self.is_flatten_flipped);
        }
        // Java: IllegalArgumentException("field=" + field)
        None
    }

    /// Java `getDoubleValue`: an auto-generated stub returning 0.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        Some(0.0)
    }

    /// Java `getHashtable`: stub, null.
    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    /// Java `getEtomoNumber`: stub, null.
    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    /// Java `getIntKeyList`: stub, null.
    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }

    /// Java `getString`: stub, null.
    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    /// Java `getStringArray`: stub, null.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getIteratorElementList`: stub, null.
    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}
