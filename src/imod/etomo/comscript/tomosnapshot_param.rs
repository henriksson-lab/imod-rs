//! `IMOD/Etomo/src/etomo/comscript/TomosnapshotParam.java`.
//!
//! `getCommandArray` builds the array on first use through the `Command`
//! trait's `&self`, so the cached array sits behind a `Mutex`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::TOMOSNAPSHOT;
/// Java `OUTPUT_LINE`.
pub const OUTPUT_LINE: &str = "Snapshot done";

/// Java private static `COMMAND_NAME` (`PROCESS_NAME.toString()`).
fn command_name() -> String {
    PROCESS_NAME.to_string()
}

/// Java final `TomosnapshotParam`.
pub struct TomosnapshotParam {
    manager: Option<&'static dyn BaseManager>,
    axis_id: AxisID,
    command_array: Mutex<Option<Vec<String>>>,
    debug: bool,
    thumbnail: bool,
}

impl TomosnapshotParam {
    /// Java `TomosnapshotParam(BaseManager, AxisID)`.
    pub fn new(manager: Option<&'static dyn BaseManager>, axis_id: AxisID) -> TomosnapshotParam {
        TomosnapshotParam {
            manager,
            axis_id,
            command_array: Mutex::new(None),
            debug: false,
            thumbnail: false,
        }
    }

    /// Java `setThumbnail(boolean)`.
    pub fn set_thumbnail(&mut self, input: bool) {
        self.thumbnail = input;
    }

    /// Java private final `buildCommand`.
    fn build_command(&self, command_array: &mut Option<Vec<String>>) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{}", command_name()));
        if self.thumbnail {
            command.push("-t".to_owned());
        }
        if let Some(manager) = self.manager {
            // The source dereferences `getBaseMetaData()` without a null check; with no
            // meta data the name is left out here.
            if let Some(meta_data) = manager.get_base_meta_data() {
                // Java string concatenation writes null parts as "null"
                command.push(format!(
                    "{}{}",
                    meta_data.get_dataset_name().as_deref().unwrap_or("null"),
                    meta_data
                        .base()
                        .get_file_extension()
                        .as_deref()
                        .unwrap_or("null")
                ));
            }
        }
        // command.add("-e");
        // command.add(manager.getBaseMetaData().getMetaDataFileName());
        let command_size = command.len();
        let mut array: Vec<String> = Vec::with_capacity(command_size);
        for element in command.into_iter().take(command_size) {
            array.push(element);
        }
        if self.debug {
            eprintln!(
                "Running tomosnapshot in {}",
                std::env::current_dir()
                    .map(|dir| dir.display().to_string())
                    .unwrap_or_else(|_| "null".to_owned())
            );
            for element in &array {
                eprint!("{element} ");
            }
            if !array.is_empty() {
                eprintln!();
            }
        }
        *command_array = Some(array);
    }
}

impl Command for TomosnapshotParam {
    fn get_command(&self) -> Option<String> {
        Some(command_name())
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
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
        self.axis_id
    }

    fn get_command_name(&self) -> Option<String> {
        Some(command_name())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array()?;
        let mut buffer = String::new();
        for element in &command_array {
            buffer.push_str(element);
            buffer.push(' ');
        }
        Some(buffer)
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command_array = self.command_array.lock().unwrap();
        if command_array.is_none() {
            self.build_command(&mut command_array);
        }
        command_array.clone()
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }
}
