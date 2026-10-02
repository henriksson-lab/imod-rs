//! `IMOD/Etomo/src/etomo/comscript/ArchiveorigParam.java`.

use std::path::PathBuf;
use std::sync::Arc;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::ARCHIVEORIG;

/// Java `COMMAND_NAME` (`PROCESS_NAME.toString()`).
pub fn command_name() -> String {
    PROCESS_NAME.to_string()
}

/// Java `ArchiveorigParam`.
pub struct ArchiveorigParam {
    command_array: Option<Vec<String>>,
    mode: Mode,
    output_file: Option<PathBuf>,
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
}

impl ArchiveorigParam {
    /// Java `ArchiveorigParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ArchiveorigParam {
        let mut mode = Mode::AxisOnly;
        if axis_id == AxisID::First {
            mode = Mode::AxisA;
        } else if axis_id == AxisID::Second {
            mode = Mode::AxisB;
        }
        let stack = utilities::get_file_must_exist_file_type(
            manager,
            false,
            axis_id,
            &file_type::CLASS.raw_stack,
            "",
        );
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        let mut command_array = vec![
            "python".to_owned(),
            "-u".to_owned(),
            format!("{script_path}{}", command_name()),
            "-PID".to_owned(),
        ];
        // `stack.getName()`: with mustExist false the file is null only when the
        // FileType cannot build a name, which Java would dereference
        // (NullPointerException); the name is left out here instead.
        if let Some(name) = stack
            .as_ref()
            .and_then(|stack| stack.file_name())
            .map(|name| name.to_string_lossy().into_owned())
        {
            command_array.push(name);
        }
        let output_file =
            utilities::get_file_must_exist_extension(manager, false, axis_id, "_xray.st.gz", "");
        ArchiveorigParam {
            command_array: Some(command_array),
            mode,
            output_file,
            manager,
        }
    }
}

impl Command for ArchiveorigParam {
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.command_array.clone()
    }

    fn get_command_name(&self) -> Option<String> {
        Some(command_name())
    }

    fn get_command(&self) -> Option<String> {
        Some(command_name())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command_line(&self) -> Option<String> {
        let command_array = match self.command_array.as_ref() {
            None => return Some(String::new()),
            Some(command_array) => command_array,
        };
        let mut buffer = String::new();
        for element in command_array {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
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

    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.output_file.clone()
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }
}

/// Java final static nested class `ArchiveorigParam.Mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `AXIS_A`.
    AxisA,
    /// Java `AXIS_B`.
    AxisB,
    /// Java `AXIS_ONLY`.
    AxisOnly,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::AxisA => "AxisA",
            Mode::AxisB => "AxisB",
            Mode::AxisOnly => "AxisOnly",
        })
    }
}

impl CommandMode for Mode {}
