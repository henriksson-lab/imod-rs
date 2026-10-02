//! `IMOD/Etomo/src/etomo/comscript/RemapmodelParam.java`.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_join_state::ConstJoinState;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::int_key_list::Walker;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::REMAPMODEL;
/// Java `COMMAND_NAME`: `PROCESS_NAME.toString()`.
pub static COMMAND_NAME: LazyLock<String> = LazyLock::new(|| PROCESS_NAME.to_string());
/// Java private static `COMMAND_SIZE`.
const COMMAND_SIZE: usize = 1;
/// Java private static `debug`.
const DEBUG: bool = false;

/// Java final `RemapmodelParam implements Command`.
pub struct RemapmodelParam {
    command_array: Vec<String>,
    manager: &'static JoinManager,
}

impl RemapmodelParam {
    /// Java `RemapmodelParam(JoinManager)`.
    pub fn new(manager: &'static JoinManager) -> RemapmodelParam {
        let mut param = RemapmodelParam {
            command_array: Vec::new(),
            manager,
        };
        let options = param.gen_options();
        let mut command_array: Vec<String> = Vec::with_capacity(options.len() + COMMAND_SIZE);
        // Java string concatenation writes a null path as "null"
        command_array.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_owned()),
            *COMMAND_NAME
        ));
        for option in options {
            command_array.push(option);
        }
        if DEBUG {
            eprintln!("{}", command_array.join(" "));
        }
        param.command_array = command_array;
        param
    }

    /// Java private `genOptions`.
    fn gen_options(&self) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        let state = self.manager.get_state();
        let trial = state.get_refine_trial().is();
        options.push("-FromChunkLimits".to_owned());
        options.push(self.build_start_end_string(
            state.get_join_start_list_walker(trial),
            state.get_join_end_list_walker(trial),
        ));
        options.push("-ToChunkLimits".to_owned());
        options.push(self.build_start_end_string(
            state.get_refine_start_list_walker(),
            state.get_refine_end_list_walker(),
        ));
        let output_file = state.get_xf_model_output_file();
        match output_file {
            None => {
                options.push(dataset_files::get_refine_aligned_model_file_name(
                    self.manager,
                ));
                options.push(dataset_files::get_refine_aligned_model_file_name(
                    self.manager,
                ));
            }
            Some(output_file) => {
                options.push(output_file.clone());
                options.push(output_file);
            }
        }
        options
    }

    /// Java private `buildStartEndString(IntKeyList.Walker, IntKeyList.Walker)`.
    fn build_start_end_string(
        &self,
        mut start_list_walker: Walker,
        mut end_list_walker: Walker,
    ) -> String {
        if start_list_walker.size() != end_list_walker.size() {
            return String::new();
        }
        let mut buffer = String::new();
        while start_list_walker.has_next() {
            // Java string concatenation writes a null value as "null".
            let start = start_list_walker
                .next_etomo_number()
                .map(|number| number.to_string())
                .unwrap_or_else(|| "null".to_owned());
            let end = end_list_walker
                .next_etomo_number()
                .map(|number| number.to_string())
                .unwrap_or_else(|| "null".to_owned());
            buffer.push_str(&format!("{start},{end}"));
            if start_list_walker.has_next() {
                buffer.push(',');
            }
        }
        buffer
    }
}

impl Command for RemapmodelParam {
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.clone())
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.command_array.clone())
    }

    /// Java `getCommandLine`.  Java starts the buffer with `commandArray[0]` and then
    /// appends every element from index 0, so the command is written twice; the loop
    /// starts at 1 here.
    fn get_command_line(&self) -> Option<String> {
        if self.command_array.is_empty() {
            return Some(String::new());
        }
        let mut buffer = self.command_array[0].clone();
        for element in self.command_array.iter().skip(1) {
            buffer.push(' ');
            buffer.push_str(element);
        }
        Some(buffer)
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.clone())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        Some(dataset_files::get_refine_aligned_model_file(self.manager))
    }

    /// Java `getOutputImageFileType`, deprecated.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2`, deprecated.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }
}
