//! `IMOD/Etomo/src/etomo/comscript/XfmodelParam.java`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::const_join_state::ConstJoinState;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::XFMODEL;
/// Java `COMMAND_NAME`: `PROCESS_NAME.toString()`.
pub static COMMAND_NAME: LazyLock<String> = LazyLock::new(|| PROCESS_NAME.to_string());
/// Java private static `debug`.
const DEBUG: bool = false;
/// Java private static `COMMAND_SIZE`.
const COMMAND_SIZE: usize = 1;

/// Java final `XfmodelParam implements CommandDetails`.
pub struct XfmodelParam {
    manager: &'static dyn BaseManager,
    /// The target of Java's `(JoinManager) manager` casts: set by the `JoinManager`
    /// constructor, the only one that makes `join` true.
    join_manager: Option<&'static JoinManager>,
    axis_id: AxisID,
    /// Java `commandArray`, built lazily by `getCommandArray`.
    command_array: Mutex<Option<Vec<String>>>,
    input_file: Option<String>,
    output_file: Option<String>,
    join: bool,
}

/// Java nested `Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fields {
    /// Java `OUTPUT_FILE`.
    OutputFile,
}

impl FieldInterface for Fields {}

impl XfmodelParam {
    /// Java `XfmodelParam(JoinManager)`.
    pub fn new_join(manager: &'static JoinManager) -> XfmodelParam {
        XfmodelParam {
            manager,
            join_manager: Some(manager),
            axis_id: AxisID::Only,
            command_array: Mutex::new(None),
            input_file: None,
            output_file: None,
            join: true,
        }
    }

    // Updates done

    /// Java `XfmodelParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> XfmodelParam {
        XfmodelParam {
            manager,
            join_manager: None,
            axis_id,
            command_array: Mutex::new(None),
            input_file: None,
            output_file: None,
            join: false,
        }
    }

    /// Java private `genReconOptions`.
    fn gen_recon_options(&self) -> Option<Vec<String>> {
        if self.join {
            eprintln!("ERROR:  calling genReconOptions when join is true.");
            return None;
        }
        let mut options: Vec<String> = Vec::new();
        options.push("-XformsToApply".to_owned());
        options.push(dataset_files::get_transform_file_name(
            self.manager,
            Some(self.axis_id),
        ));
        let file = file_type::CLASS
            .fiducial_no_gaps_model
            .get_file(Some(self.manager), Some(self.axis_id));
        // A null file is a NullPointerException in Java (`file.exists()`); here it is a
        // file that does not exist.
        match file.filter(|file| file.exists()) {
            Some(file) => {
                options.push(utilities::java_io_file_get_name(&file.to_string_lossy()));
            }
            None => {
                options.push(dataset_files::get_fiducial_model_name(
                    self.manager,
                    Some(self.axis_id),
                ));
            }
        }
        // A null file name is a null element in Java, which `ProcessBuilder` rejects;
        // here it is left out.
        if let Some(file_name) = file_type::CLASS
            .ccd_eraser_beads_input_model
            .get_file_name(Some(self.manager), Some(self.axis_id))
        {
            options.push(file_name);
        }
        Some(options)
    }

    /// Java private `genJoinOptions`.
    fn gen_join_options(&self) -> Option<Vec<String>> {
        if !self.join {
            eprintln!("ERROR:  calling genJoinOptions when join is false.");
            return None;
        }
        let join_manager = self.join_manager?;
        let mut options: Vec<String> = Vec::new();
        options.push("-XformsToApply".to_owned());
        options.push(dataset_files::get_refine_xg_file_name(self.manager));
        let state = join_manager.get_state();
        let trial = state.get_refine_trial().is();
        let binning: &ConstEtomoNumber = &state.get_join_trial_binning();
        if trial && !binning.is_null() && binning.gt_int(1) {
            options.push("-ScaleShifts".to_owned());
            let mut scale_shifts = EtomoNumber::new_with_type(Some(Type::Double));
            scale_shifts.set_double(1.0 / binning.get_int() as f64);
            options.push(scale_shifts.to_string());
        }
        options.push("-ChunkSizes".to_owned());
        let mut start_list_walker = state.get_join_start_list_walker(trial);
        let mut end_list_walker = state.get_join_end_list_walker(trial);
        // check for valid lists
        if start_list_walker.size() == end_list_walker.size() {
            let mut buffer = String::new();
            while start_list_walker.has_next() {
                // Java calls `getInt()` on each value without a null check; a missing
                // value ends the list here instead of throwing.
                let (Some(start), Some(end)) = (
                    start_list_walker.next_etomo_number(),
                    end_list_walker.next_etomo_number(),
                ) else {
                    break;
                };
                let start = start.get_int();
                let end = end.get_int();
                if end >= start {
                    buffer.push_str(&(end - start + 1).to_string());
                } else {
                    buffer.push_str(&(start - end + 1).to_string());
                }
                if start_list_walker.has_next() {
                    buffer.push(',');
                }
            }
            options.push(buffer);
        }
        match &self.input_file {
            None => options.push(dataset_files::get_refine_model_file_name(self.manager)),
            Some(input_file) => options.push(input_file.clone()),
        }
        match &self.output_file {
            None => options.push(dataset_files::get_refine_aligned_model_file_name(
                self.manager,
            )),
            Some(output_file) => options.push(output_file.clone()),
        }
        Some(options)
    }

    /// Java `isValid`.
    pub fn is_valid(&self) -> bool {
        if self.join {
            let in_file: PathBuf = match &self.input_file {
                None => match self.join_manager {
                    Some(join_manager) => dataset_files::get_refine_model_file(join_manager),
                    None => return true,
                },
                Some(input_file) => {
                    let mut in_file = PathBuf::from(input_file);
                    if !in_file.is_absolute() {
                        in_file = match self.manager.get_property_user_dir() {
                            Some(dir) => {
                                PathBuf::from(utilities::java_io_file_new(&dir, input_file))
                            }
                            None => PathBuf::from(input_file),
                        };
                    }
                    in_file
                }
            };
            let out_file: PathBuf = match &self.output_file {
                None => dataset_files::get_refine_aligned_model_file(self.manager),
                Some(output_file) => {
                    let mut out_file = PathBuf::from(output_file);
                    if !out_file.is_absolute() {
                        out_file = match self.manager.get_property_user_dir() {
                            Some(dir) => {
                                PathBuf::from(utilities::java_io_file_new(&dir, output_file))
                            }
                            None => PathBuf::from(output_file),
                        };
                    }
                    out_file
                }
            };
            if Path::new(&in_file) == Path::new(&out_file) {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!(
                        "Cannot overwrite xfmodel input file, {} with output file, {}.",
                        in_file.display(),
                        out_file.display()
                    ),
                    "XfmodelParam Error".to_owned(),
                    None,
                );
                return false;
            }
        }
        true
    }

    /// Java `setInputFile(String)`.
    pub fn set_input_file(&mut self, input_file: Option<&str>) {
        self.input_file = input_file.map(str::to_owned);
    }

    /// Java `setOutputFile(String)`.
    pub fn set_output_file(&mut self, output_file: Option<&str>) {
        self.output_file = output_file.map(str::to_owned);
    }
}

impl Command for XfmodelParam {
    /// Java `command instanceof ProcessDetails`: this class is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.clone())
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command_array = self.command_array.lock().unwrap();
        if command_array.is_none() {
            let options = if self.join {
                self.gen_join_options()
            } else {
                self.gen_recon_options()
            };
            let options = match options {
                None => return Some(Vec::new()),
                Some(options) => options,
            };
            let mut array: Vec<String> = Vec::with_capacity(options.len() + COMMAND_SIZE);
            // Java string concatenation writes a null path as "null"
            array.push(format!(
                "{}{}",
                base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_owned()),
                *COMMAND_NAME
            ));
            for option in options {
                array.push(option);
            }
            if DEBUG {
                eprintln!("{}", array.join(" "));
            }
            *command_array = Some(array);
        }
        command_array.clone()
    }

    /// Java `getCommandLine`.  Java starts the buffer with `commandArray[0]` and then
    /// appends every element from index 0, so the command is written twice; the loop
    /// starts at 1 here.  Java also dereferences `commandArray` when `getCommandArray`
    /// found no options and left it null; that is an empty line here.
    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array().unwrap_or_default();
        if command_array.is_empty() {
            return Some(String::new());
        }
        let mut buffer = command_array[0].clone();
        for element in command_array.iter().skip(1) {
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

    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.clone())
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

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        Some(dataset_files::get_refine_aligned_model_file(self.manager))
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }
}

impl Loggable for XfmodelParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage`, which returns null: no log message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Java throws `IllegalArgumentException("field=" + field)` for every field it does not
/// handle; those return `None`.
impl ProcessDetails for XfmodelParam {
    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }

    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::OutputFile) {
            return self.output_file.clone();
        }
        None
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }
}
