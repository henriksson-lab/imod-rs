//! `IMOD/Etomo/src/etomo/comscript/CombineComscriptState.java`.
//!
//! combine.com.  Contains information about which commands will be run.  Also
//! knows about watched files like patch.out.

use std::sync::Arc;

use super::com_script_manager::ComScriptManager;
use super::comscript_state::ComscriptState;
use super::echo_param::{self, EchoParam};
use super::exit_param::{self, ExitParam};
use super::goto_param::GotoParam;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::dataset_files;

/// Java `COMSCRIPT_NAME`.
pub const COMSCRIPT_NAME: &str = "combine";
/// Java `COMSCRIPT_WATCHED_FILE`.
pub const COMSCRIPT_WATCHED_FILE: &str = "combine.out";
/// Java private static `WATCHED_FILES`.
const WATCHED_FILES: [Option<&str>; 5] = [None, None, Some(dataset_files::PATCH_OUT), None, None];
/// Java private `NULL_INDEX`.
const NULL_INDEX: i32 = -1;
/// Java private `SOLVEMATCH_DUALVOLMATCH_INDEX`.
const SOLVEMATCH_DUALVOLMATCH_INDEX: i32 = 0;
/// Java private `MATCHVOL1_INDEX`.
#[allow(dead_code)]
const MATCHVOL1_INDEX: i32 = 1;
/// Java private `PATCHCORR_INDEX`.
const PATCHCORR_INDEX: i32 = 2;
/// Java private `MATCHORWARP_INDEX`.
const MATCHORWARP_INDEX: i32 = 3;
/// Java private `VOLCOMBINE_INDEX`.
const VOLCOMBINE_INDEX: i32 = 4;
/// Java private `LABEL_DELIMITER`.
const LABEL_DELIMITER: char = ':';
/// Java private `SUCCESS_TEXT`.
const SUCCESS_TEXT: &str = "COMBINE SUCCESSFULLY COMPLETED";
/// Java private `THROUGH_TEXT`.
const THROUGH_TEXT: &str = " THROUGH ";
/// Java private `CONSTRUCTED_STATE` (unused in the source).
#[allow(dead_code)]
const CONSTRUCTED_STATE: i32 = 1;
/// Java private `INITIALIZED_STATE` (unused in the source).
#[allow(dead_code)]
const INITIALIZED_STATE: i32 = 2;
/// Java private `START_COMMAND_SET_STATE` (unused in the source).
#[allow(dead_code)]
const START_COMMAND_SET_STATE: i32 = 3;
/// Java private `END_COMMAND_SET_STATE` (unused in the source).
#[allow(dead_code)]
const END_COMMAND_SET_STATE: i32 = 4;

/// Java final `CombineComscriptState implements ComscriptState`.
pub struct CombineComscriptState {
    commands: [String; 5],
    start_command: i32,
    end_command: i32,
    output_image_file_type: Option<Arc<FileType>>,
    output_image_file_type2: Option<Arc<FileType>>,
    output_image_file_type_external: Option<Arc<FileType>>,
    // testing variables
    not_equals_reason: Option<String>,
    #[allow(dead_code)]
    self_test: bool,
}

impl CombineComscriptState {
    /// Java `CombineComscriptState(boolean)`.
    pub fn new(initial_volume_matching: bool) -> CombineComscriptState {
        let mut instance = CombineComscriptState {
            commands: [
                ProcessName::SOLVEMATCH.to_string(),
                ProcessName::MATCHVOL1.to_string(),
                ProcessName::PATCHCORR.to_string(),
                ProcessName::MATCHORWARP.to_string(),
                ProcessName::VOLCOMBINE.to_string(),
            ],
            start_command: NULL_INDEX,
            end_command: NULL_INDEX,
            output_image_file_type: None,
            output_image_file_type2: None,
            output_image_file_type_external: None,
            not_equals_reason: None,
            self_test: false,
        };
        if initial_volume_matching {
            instance.commands[SOLVEMATCH_DUALVOLMATCH_INDEX as usize] =
                ProcessName::DUALVOLMATCH.to_string();
        }
        instance
    }

    /// Java package-private `initialize`.  Initialize instance from combine.com.
    pub(crate) fn initialize(&mut self, com_script_manager: &ComScriptManager) -> bool {
        if !self.load_start_command(com_script_manager) {
            return false;
        }
        self.load_end_command(com_script_manager);
        true
    }

    /// Java `setStartCommand`.  Sets startCommand in combine.com.  An index out of
    /// range is the Java's `IndexOutOfBoundsException`, a caller error.
    pub fn set_start_command(&mut self, start_command: i32, com_script_manager: &ComScriptManager) {
        if start_command < 0 || start_command >= self.commands.len() as i32 {
            panic!("java.lang.IndexOutOfBoundsException");
        }
        self.start_command = start_command;
        if start_command <= PATCHCORR_INDEX {
            self.output_image_file_type = Some(Arc::clone(&file_type::CLASS.patch_vector_model));
            self.output_image_file_type2 =
                Some(Arc::clone(&file_type::CLASS.patch_vector_ccc_model));
        } else {
            self.output_image_file_type = None;
            self.output_image_file_type2 = None;
        }
        let mut goto_param = GotoParam::new();
        goto_param.set_label(Some(&self.commands[start_command as usize]));
        com_script_manager.save_combine_goto(&goto_param);
    }

    /// Java `setEndCommand`.  Sets endCommand in combine.com.  An index out of
    /// range, or an end command other than volcombine or matchorwarp, is the
    /// Java's unchecked exception, a caller error.
    pub fn set_end_command(&mut self, end_command: i32, com_script_manager: &ComScriptManager) {
        if end_command < 0 || end_command >= self.commands.len() as i32 {
            panic!("java.lang.IndexOutOfBoundsException");
        }
        self.end_command = end_command;
        let command_label = self.to_label(VOLCOMBINE_INDEX);
        // if the endCommand is the last command (volcombine), remove the exit
        // success commands from after the volcombine label, if they are there
        if end_command == VOLCOMBINE_INDEX {
            // look for the success echo. Delete it and the exit command if it is
            // found
            let echo_param_in_comscript =
                com_script_manager.get_echo_param_from_combine(&command_label);
            if let Some(echo_param_in_comscript) = echo_param_in_comscript
                && echo_param_in_comscript
                    .get_string()
                    .starts_with(SUCCESS_TEXT)
            {
                com_script_manager.delete_from_combine(echo_param::COMMAND_NAME, &command_label);
                com_script_manager.delete_from_combine(exit_param::COMMAND_NAME, &command_label);
            }
        } else if end_command == MATCHORWARP_INDEX {
            // if the endCommand is not the last command (must be matchorwarp), add
            // or update exit success commands after the volcombine label
            // insert echo param if it is not there, otherwise update it
            let mut echo_param = EchoParam::new();
            echo_param.set_string(&format!(
                "{SUCCESS_TEXT}{THROUGH_TEXT}{}",
                self.commands[MATCHORWARP_INDEX as usize].to_uppercase()
            ));
            let echo_index = com_script_manager.save_combine_echo(&echo_param, &command_label);
            // insert exit param if it is not there, otherwise update it
            let mut exit_param = ExitParam::new();
            exit_param.set_result_value(0);
            com_script_manager.save_combine_exit(&exit_param, echo_index);
        } else {
            panic!(
                "java.lang.IllegalStateException: EndCommand can only be volcombine or matchorwarp.  endCommand={end_command}"
            );
        }
    }

    /// Java `getMatchingCommand`.
    pub fn get_matching_command(&self, line: Option<&str>) -> Option<String> {
        let line = line?;
        for i in 0..self.commands.len() {
            if line.contains(self.commands[i].as_str()) {
                return Some(self.commands[i].clone());
            }
        }
        None
    }

    /// Java `resetOutputImageFileType`.
    pub fn reset_output_image_file_type(&mut self) {
        self.output_image_file_type_external = None;
    }

    /// Java `setOutputImageFileType`.
    pub fn set_output_image_file_type(&mut self, input: Option<Arc<FileType>>) {
        self.output_image_file_type_external = input;
    }

    /// Java `isRunVolcombine`.  Returns true if volcombine will run.
    pub fn is_run_volcombine(&self) -> bool {
        self.end_command >= VOLCOMBINE_INDEX
    }

    /// Java `getInitialProcessName`.
    pub fn get_initial_process_name(&self) -> Option<ProcessName> {
        ProcessName::get_instance(Some(&self.commands[SOLVEMATCH_DUALVOLMATCH_INDEX as usize]))
    }

    /// Java static `getSuccessText`.
    pub fn get_success_text() -> &'static str {
        SUCCESS_TEXT
    }

    /// Java private `getCommandIndex`.  Convert a command name to a command index.
    fn get_command_index(&self, command_name: &str) -> i32 {
        for i in 0..self.commands.len() {
            if command_name == self.commands[i] {
                return i as i32;
            }
        }
        NULL_INDEX
    }

    /// Java private `toLabel`.  Convert a command index to a label string.
    fn to_label(&self, command_index: i32) -> String {
        format!(
            "{}{}",
            self.commands[command_index as usize], LABEL_DELIMITER
        )
    }

    /// Java private `loadStartCommand`.  Load startCommand from combine.com.
    fn load_start_command(&mut self, com_script_manager: &ComScriptManager) -> bool {
        // check the first goto to see which command will be run first
        let goto_param = com_script_manager.get_goto_param_from_combine();
        // backward compatibility - old combine.com did not have this goto
        let goto_param = match goto_param {
            None => return false,
            Some(goto_param) => goto_param,
        };
        // A null label is a NullPointerException in the Java's `getCommandIndex`
        // (CombineComscriptState.java:303); it matches no command here.
        self.start_command = match goto_param.get_label() {
            None => NULL_INDEX,
            Some(label) => self.get_command_index(label),
        };
        true
    }

    /// Java private `loadEndCommand`.  Load end command from combine.com.  Check
    /// to see if combine.com is exiting before running volcombine.
    fn load_end_command(&mut self, com_script_manager: &ComScriptManager) {
        let echo_param =
            com_script_manager.get_echo_param_from_combine(&self.to_label(VOLCOMBINE_INDEX));
        if echo_param.is_some_and(|echo_param| echo_param.get_string().starts_with(SUCCESS_TEXT)) {
            self.end_command = VOLCOMBINE_INDEX - 1;
        } else {
            self.end_command = VOLCOMBINE_INDEX;
        }
    }

    /// Java `isDualvolmatchPresent`.
    pub fn is_dualvolmatch_present(&self, com_script_manager: &ComScriptManager) -> bool {
        com_script_manager.is_dualvolmatch_label_in_combine()
    }

    /// Java `equals(CombineComscriptState)`.
    pub fn equals(&mut self, that: &CombineComscriptState) -> bool {
        if self.start_command != that.start_command {
            self.not_equals_reason = Some(format!(
                "StartCommand is not equal.  this.startCommand={},that.startCommand={}",
                self.start_command, that.start_command
            ));
            return false;
        }
        if self.end_command != that.end_command {
            self.not_equals_reason = Some(format!(
                "EndCommand is not equal.  this.endCommand={},that.endCommand={}",
                self.end_command, that.end_command
            ));
            return false;
        }
        self.not_equals_reason = None;
        true
    }
}

impl ComscriptState for CombineComscriptState {
    /// Java `getStartCommand`.
    fn get_start_command(&self) -> i32 {
        self.start_command
    }

    /// Java `getEndCommand`.
    fn get_end_command(&self) -> i32 {
        self.end_command
    }

    /// Java `getCommand(int)`.
    fn get_command(&self, command_index: i32) -> Option<String> {
        if command_index == NULL_INDEX {
            return None;
        }
        Some(self.commands[command_index as usize].clone())
    }

    /// Java `getWatchedFile(int)`.
    fn get_watched_file(&self, command_index: i32) -> Option<String> {
        if command_index == NULL_INDEX {
            return None;
        }
        WATCHED_FILES[command_index as usize].map(str::to_string)
    }

    /// Java `getComscriptName`.
    fn get_comscript_name(&self) -> String {
        COMSCRIPT_NAME.to_string()
    }

    /// Java `getComscriptWatchedFile`.
    fn get_comscript_watched_file(&self) -> String {
        COMSCRIPT_WATCHED_FILE.to_string()
    }

    /// Java `getOutputImageFileType` (deprecated 6/18/19).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        self.output_image_file_type.clone()
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.output_image_file_type
            .as_ref()
            .map(|file_type| FileKey::clone(file_type))
    }

    /// Java `getOutputImageFileType2` (deprecated 6/18/19).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        self.output_image_file_type2.clone()
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        self.output_image_file_type2
            .as_ref()
            .map(|file_type| FileKey::clone(file_type))
    }

    /// Java `getOutputImageFileType3` (deprecated 6/18/19).
    fn get_output_image_file_type3(&self) -> Option<Arc<FileType>> {
        self.output_image_file_type_external.clone()
    }

    /// Java `getOutputImageFileKey3`.
    fn get_output_image_file_key3(&self) -> Option<FileKey> {
        self.output_image_file_type_external
            .as_ref()
            .map(|file_type| FileKey::clone(file_type))
    }
}
