//! `IMOD/Etomo/src/etomo/comscript/ClipParam.java`.
//!
//! The `clip rotx` and `clip stats` command lines.

use std::path::{Path, PathBuf};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::FieldInterface;
use super::process_details::ProcessDetails;
use super::utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `PROCESS_NAME`.
pub const PROCESS_NAME: ProcessName = ProcessName::CLIP;
/// Java private static `commandSize`.
const COMMAND_SIZE: usize = 1;

/// Java nested `ClipParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java private `ROTX`, process "rotx".
    Rotx,
    /// Java `STATS`, process "stats".
    Stats,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`: the process.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Rotx => "rotx",
            Mode::Stats => "stats",
        })
    }
}

impl CommandMode for Mode {}

/// Java final `ClipParam implements CommandDetails`.
pub struct ClipParam {
    output_file: Option<PathBuf>,
    command_array: Vec<String>,
    debug: bool,
    application_manager: Option<&'static ApplicationManager>,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    mode: Mode,
    input_file: PathBuf,
}

impl ClipParam {
    /// Java private `ClipParam(BaseManager, ApplicationManager, AxisID, File, File, Mode)`.
    fn new(
        manager: &'static dyn BaseManager,
        application_manager: Option<&'static ApplicationManager>,
        axis_id: AxisID,
        input_file: &Path,
        working_dir: &Path,
        mode: Mode,
    ) -> ClipParam {
        let mut param = ClipParam {
            output_file: None,
            command_array: Vec::new(),
            debug: false,
            application_manager,
            manager,
            axis_id,
            mode,
            input_file: input_file.to_path_buf(),
        };
        let options = param.gen_options(input_file, working_dir);
        let mut command_array = vec![String::new(); options.len() + COMMAND_SIZE];
        command_array[0] = format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string()),
            PROCESS_NAME
        );
        for i in 0..options.len() {
            command_array[i + COMMAND_SIZE] = options[i].clone();
        }
        param.command_array = command_array;
        if param.debug {
            for i in 0..param.command_array.len() {
                eprint!("{} ", param.command_array[i]);
            }
            if !param.command_array.is_empty() {
                eprintln!();
            }
        }
        param
    }

    /// Java `getRotxInstance(BaseManager, AxisID, File, File)`.
    pub fn get_rotx_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        input_file: &Path,
        working_dir: &Path,
    ) -> ClipParam {
        ClipParam::new(manager, None, axis_id, input_file, working_dir, Mode::Rotx)
    }

    /// Java `getStatsInstance(ApplicationManager, AxisID, File, File)`.
    pub fn get_stats_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        input_file: &Path,
        working_dir: &Path,
    ) -> ClipParam {
        ClipParam::new(
            manager,
            Some(manager),
            axis_id,
            input_file,
            working_dir,
            Mode::Stats,
        )
    }

    /// Java private `genOptions(File, File)`.
    fn gen_options(&mut self, input_file: &Path, working_dir: &Path) -> Vec<String> {
        let mut options: Vec<String> = Vec::with_capacity(3);
        // Add process.
        options.push(self.mode.to_string());
        // Add options.
        if self.mode == Mode::Stats {
            match self.application_manager {
                None => {
                    eprintln!(
                        "Warning: Unable to get the view type.  Coordinates may be incorrect if this is a montage."
                    );
                }
                Some(application_manager) => {
                    if application_manager.get_const_meta_data().get_view_type()
                        == ViewType::Montage
                    {
                        options.push("-PID".to_string());
                        let dataset_name = self
                            .manager
                            .get_base_meta_data()
                            .and_then(|meta_data| meta_data.get_dataset_name())
                            .unwrap_or_else(|| "null".to_string());
                        options.push(format!(
                            "{}{}.pl",
                            dataset_name,
                            self.axis_id.get_extension()
                        ));
                        options.push("-O".to_string());
                        options.push(format!(
                            "{},{}",
                            utilities::MONTAGE_SEPARATION,
                            utilities::MONTAGE_SEPARATION
                        ));
                    }
                }
            }
            // Put a * on the outliers.
            options.push("-n".to_string());
            options.push("2.5".to_string());
            // The length should be 1/4 of Z, but between 15 and 30.
            options.push("-l".to_string());
            let mut length: i32;
            let min = 15;
            let max = 30;
            let header = MRCHeader::get_instance_from_file_name(
                self.manager,
                Some(self.axis_id),
                input_file.file_name().and_then(|name| name.to_str()),
            );
            // `header.read(manager)`: an InvalidParameterException or IOException picks a
            // midrange number if the header can't be read.
            match header {
                Some(header) => {
                    let read = header.borrow_mut().read_with_manager(self.manager);
                    match read {
                        Ok(_) => length = header.borrow().get_n_sections(),
                        Err(_) => length = max * 2,
                    }
                }
                None => length = max * 2,
            }
            length /= 4;
            if length < min {
                length = min;
            }
            if length > max {
                length = max;
            }
            options.push(length.to_string());
            // Display the views starting from 1 instead of 0.
            options.push("-1".to_string());
        }
        // Add input files.
        options.push(java_io_file_get_absolute_path(
            &input_file.to_string_lossy(),
        ));
        // Add output files.
        if self.mode == Mode::Rotx {
            let name = input_file
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default();
            let index = name.rfind('.');
            let mut clip_file_name = String::new();
            match index {
                None => clip_file_name.push_str(&name),
                Some(index) => clip_file_name.push_str(&name[..index]),
            }
            // Still using .flip for the output name for clip rotx, since we used to flip
            // instead of rotate.
            let output_file = working_dir.join(format!("{clip_file_name}.flip"));
            options.push(java_io_file_get_absolute_path(
                &output_file.to_string_lossy(),
            ));
            self.output_file = Some(output_file);
        }
        options
    }
}

impl Command for ClipParam {
    /// Java `getAxisID()`: always `AxisID.ONLY`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        let mut buffer = String::new();
        for i in 0..self.command_array.len() {
            buffer.push_str(&format!("{} ", self.command_array[i]));
        }
        Some(buffer)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
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

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.command_array.clone())
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.output_file.clone()
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        Some(self.input_file.clone())
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for ClipParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Every ProcessDetails getter in the source throws
/// `IllegalArgumentException("field=" + field)` (ClipParam.java:236-279), an uncaught
/// exception for any caller.  Fixed in translation: the value is unavailable (`None`).
impl ProcessDetails for ClipParam {
    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    /// Java `getBooleanValue(FieldInterface)`.
    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    /// Java `getEtomoNumber(FieldInterface)`.
    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    /// Java `getIntKeyList(FieldInterface)`.
    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    /// Java `getStringArray(FieldInterface)`.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getIteratorElementList(FieldInterface)`.
    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}
