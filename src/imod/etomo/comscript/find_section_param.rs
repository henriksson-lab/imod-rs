//! `IMOD/Etomo/src/etomo/comscript/FindSectionParam.java`.
//!
//! The findsection command line for the tomogram positioning samples or the whole
//! tomogram.

use std::path::PathBuf;
use std::sync::Mutex;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::FIND_SECTION;
/// Java private static `TOMOGRAM_FILE_TAG`.
const TOMOGRAM_FILE_TAG: &str = "-tomo";

/// Java nested `FindSectionParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java private `SAMPLE`, "Sample".
    Sample,
    /// Java private `WHOLE_TOMOGRAM`, "WholeTomogram".
    WholeTomogram,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Sample => "Sample",
            Mode::WholeTomogram => "WholeTomogram",
        })
    }
}

impl CommandMode for Mode {}

/// Java final `FindSectionParam implements Command`.
pub struct FindSectionParam {
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    mode: Mode,
    /// Java field `commandArray`, built once; the source guards the null test with
    /// `synchronized (this)`.
    command_array: Mutex<Option<Vec<String>>>,
}

impl FindSectionParam {
    /// Java `FindSectionParam(BaseManager, AxisID, boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        whole_tomogram: bool,
    ) -> FindSectionParam {
        let mode = if !whole_tomogram {
            Mode::Sample
        } else {
            Mode::WholeTomogram
        };
        FindSectionParam {
            axis_id,
            manager,
            mode,
            command_array: Mutex::new(None),
        }
    }

    /// Java private `buildCommandArray()`.
    fn build_command_array(&self) {
        let mut array: Vec<String>;
        {
            let command_array = self.command_array.lock().unwrap();
            if command_array.is_some() {
                return;
            }
            array = Vec::new();
        }
        let manager = Some(self.manager);
        let axis_id = Some(self.axis_id);
        // A null file name is a null element in Java's list; "null" here.
        let null = || "null".to_string();
        array.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(null),
            PROCESS_NAME
        ));
        array.push("-scales".to_string());
        if self.mode == Mode::Sample {
            array.push("4".to_string());
        } else {
            array.push("2".to_string());
        }
        array.push("-pitch".to_string());
        array.push(
            file_type::CLASS
                .tomopitch_model
                .get_file_name(manager, axis_id)
                .unwrap_or_else(null),
        );
        array.push("-size".to_string());
        if self.mode == Mode::Sample {
            array.push("50,1,20".to_string());
        } else {
            array.push("16,1,16".to_string());
        }
        if self.mode == Mode::WholeTomogram {
            array.push("-block".to_string());
            array.push("48".to_string());
            array.push("-samples".to_string());
            array.push("5".to_string());
        }
        array.push(TOMOGRAM_FILE_TAG.to_string());
        if self.mode == Mode::Sample {
            array.push(
                file_type::CLASS
                    .top_sample
                    .get_file_name(manager, axis_id)
                    .unwrap_or_else(null),
            );
            array.push(TOMOGRAM_FILE_TAG.to_string());
            array.push(
                file_type::CLASS
                    .middle_sample
                    .get_file_name(manager, axis_id)
                    .unwrap_or_else(null),
            );
            array.push(TOMOGRAM_FILE_TAG.to_string());
            array.push(
                file_type::CLASS
                    .bottom_sample
                    .get_file_name(manager, axis_id)
                    .unwrap_or_else(null),
            );
        } else {
            array.push(
                file_type::CLASS
                    .tilt_output
                    .get_file_name(manager, axis_id)
                    .unwrap_or_else(null),
            );
        }
        *self.command_array.lock().unwrap() = Some(array);
    }
}

impl Command for FindSectionParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.build_command_array();
        self.command_array.lock().unwrap().clone()
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        self.build_command_array();
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

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
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

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
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
