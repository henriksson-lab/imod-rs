//! `IMOD/Etomo/src/etomo/comscript/ExcludeViewsParam.java`.
//!
//! Parameters and run command for excludeviews batch process.

use std::path::PathBuf;
use std::sync::Mutex;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::EXCLUDE_VIEWS;
/// Java private static `MAX_EXCLUSION` (unused in the source).
#[allow(dead_code)]
const MAX_EXCLUSION: &str = "9";

/// Java final `ExcludeViewsParam implements Command`.
pub struct ExcludeViewsParam {
    dataset_dir: Option<String>,
    stack_name: StringParameter,
    views_to_exclude: StringParameter,
    montaged_images: EtomoBoolean2,
    delete_old_files: EtomoBoolean2,
    axis_id: AxisID,
    /// Java field `commandArray`, rebuilt by every `getCommandArray`/`getCommandLine`
    /// call; a `Mutex` because the `Command` methods take `&self`.
    command_array: Mutex<Option<Vec<String>>>,
}

impl std::fmt::Display for ExcludeViewsParam {
    /// Java `toString()`.  `super.toString()` is `Object.toString`, the class name and
    /// identity hash; the address stands in for the hash.  Note that the source never
    /// closes the opening bracket.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.comscript.ExcludeViewsParam@{:x}:[axisID:{},stackName:{},viewsToExclude:{},montagedImages:{},deleteOldFiles:{}",
            self as *const ExcludeViewsParam as usize,
            self.axis_id,
            self.stack_name,
            self.views_to_exclude,
            self.montaged_images,
            self.delete_old_files
        )
    }
}

impl ExcludeViewsParam {
    /// Java `ExcludeViewsParam(AxisID, String)`.
    pub fn new(axis_id: AxisID, dataset_dir: Option<&str>) -> ExcludeViewsParam {
        ExcludeViewsParam {
            dataset_dir: dataset_dir.map(|s| s.to_string()),
            stack_name: StringParameter::new("StackName"),
            views_to_exclude: StringParameter::new("ViewsToExclude"),
            montaged_images: EtomoBoolean2::new_with_name("MontagedImages"),
            delete_old_files: EtomoBoolean2::new_with_name("DeleteOldFiles"),
            axis_id,
            command_array: Mutex::new(None),
        }
    }

    /// Java `setStackName(String)`.
    pub fn set_stack_name(&mut self, input: Option<&str>) {
        self.stack_name.set(input);
    }

    /// Java `setViewsToExclude(String)`.
    pub fn set_views_to_exclude(&mut self, input: Option<&str>) {
        self.views_to_exclude.set(input);
    }

    /// Java `setMontagedImages(boolean)`.
    pub fn set_montaged_images(&mut self, input: bool) {
        self.montaged_images.set_boolean(input);
    }

    /// Java `setDeleteOldFiles(boolean)`.
    pub fn set_delete_old_files(&mut self, input: bool) {
        self.delete_old_files.set_boolean(input);
    }

    /// Java private `createCommand()`.
    fn create_command(&self) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_string());
        command.push("-u".to_string());
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{PROCESS_NAME}"));
        command.push("-PID".to_string());
        command.push(format!("-{}", self.stack_name.get_name()));
        command.push(self.stack_name.to_string());
        command.push(format!("-{}", self.views_to_exclude.get_name()));
        command.push(self.views_to_exclude.to_string());
        if self.montaged_images.is() {
            command.push(format!("-{}", self.montaged_images.get_name()));
        }
        if self.delete_old_files.is() {
            command.push(format!("-{}", self.delete_old_files.get_name()));
        }
        *self.command_array.lock().unwrap() = Some(command);
    }
}

impl Command for ExcludeViewsParam {
    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.create_command();
        self.command_array.lock().unwrap().clone()
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        self.create_command();
        let command_array = self.command_array.lock().unwrap();
        let command_array = match &*command_array {
            None => return Some(String::new()),
            Some(command_array) => command_array,
        };
        let mut buffer = String::new();
        for i in 0..command_array.len() {
            buffer.push_str(&format!("{} ", command_array[i]));
        }
        Some(buffer)
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        if let Some(dataset_dir) = &self.dataset_dir {
            return Some(PathBuf::from(dataset_dir).join(self.stack_name.to_string()));
        }
        Some(PathBuf::from(self.stack_name.to_string()))
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.get_command_input_file()
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }
}
