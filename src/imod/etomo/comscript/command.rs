//! `IMOD/Etomo/src/etomo/comscript/Command.java`.

use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `Command` interface.  The optional file and subcommand values retain
/// Java's nullable parameter behavior while every command spelling remains
/// separate from `SystemProgram`'s executable child description.
pub trait Command {
    fn get_axis_id(&self) -> AxisID;
    fn get_command_mode(&self) -> Option<&dyn CommandMode>;
    fn get_process_name(&self) -> Option<ProcessName>;
    fn get_command(&self) -> Option<String>;
    fn get_command_name(&self) -> Option<String>;
    fn get_command_line(&self) -> Option<String>;
    fn get_command_array(&self) -> Option<Vec<String>>;
    fn get_command_input_file(&self) -> Option<std::path::PathBuf>;
    fn get_command_output_file(&self) -> Option<std::path::PathBuf>;
    fn get_output_image_file_type(&self) -> Option<FileType>;
    fn get_output_image_file_key(&self) -> Option<FileKey>;
    fn get_output_image_file_type2(&self) -> Option<FileType>;
    fn get_output_image_file_key2(&self) -> Option<FileKey>;
    fn is_message_reporter(&self) -> bool;
    fn get_subcommand_process_name(&self) -> Option<String>;
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails>;
}
