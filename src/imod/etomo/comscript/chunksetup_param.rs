//! `IMOD/Etomo/src/etomo/comscript/ChunksetupParam.java`.
//!
//! Parameters for `chunksetup`, which splits a volume operation into chunk
//! command files for processchunks (anisotropic diffusion, or a one-line
//! command run over a volume in parallel).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::util::utilities;

/// Java `MEMORY_TO_VOXEL`.
pub const MEMORY_TO_VOXEL: i32 = 36;
/// Java private static `OVERLAP_MIN`.
const OVERLAP_MIN: i32 = 8;
/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::CHUNKSETUP;
/// Java private static `ONE_LINE_COMMAND_PROGRAM_INDEX`.
const ONE_LINE_COMMAND_PROGRAM_INDEX: usize = 0;
/// Java private static `ONE_LINE_COMMAND_ARGUMENTS_INDEX`.
const ONE_LINE_COMMAND_ARGUMENTS_INDEX: usize = 1;
/// Java private static `ONE_LINE_COMMAND_SIZE`.
const ONE_LINE_COMMAND_SIZE: usize = ONE_LINE_COMMAND_ARGUMENTS_INDEX + 1;
/// Java `RESULT_SUFFIX`.
pub const RESULT_SUFFIX: &str = "-cs";

/// Java public final static nested class `Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `Mode.NAD`.
    Nad,
    /// Java `Mode.PARALLEL`.
    Parallel,
}

impl std::fmt::Display for Mode {
    /// Java `toString`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Nad => "NAD",
            Mode::Parallel => "Parallel",
        })
    }
}

impl CommandMode for Mode {}

/// Java public static final nested class `Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `Field.ONE_LINE_COMMAND_PROGRAM`.
    OneLineCommandProgram,
}

impl FieldInterface for Field {}

/// Java public final `ChunksetupParam implements CommandDetails`.
///
/// The setters take `&mut self`: they run before the param is handed to a
/// process.  `command` is built lazily by the `Command` getters, which take
/// `&self`, so it sits in a `Mutex`.
pub struct ChunksetupParam {
    /// Java private final field `command`.
    command: Mutex<Vec<String>>,
    /// Java private final field `megavoxelMaximum`.
    megavoxel_maximum: EtomoNumber,
    /// Java private final field `overlapPixels`.
    overlap_pixels: EtomoNumber,
    /// Java private final field `mode`.
    mode: Mode,
    /// Java private field `subdirName`.
    subdir_name: Option<String>,
    /// Java private field `commandFile`.
    command_file: Option<String>,
    /// Java private field `inputFile`.
    input_file: Option<String>,
    /// Java private field `outputFile`.
    output_file: Option<String>,
    /// Java private field `debug`.
    debug: bool,
    /// Java private field `overlap`.
    overlap: i32,
    /// Java private field `overlapTimesFour`.
    overlap_times_four: bool,
    /// Java private field `oneLineCommand`.
    one_line_command: Option<[Option<String>; ONE_LINE_COMMAND_SIZE]>,
    /// Java private field `inputImageFile`.
    input_image_file: Option<String>,
    /// Java private field `suffixForOutputName`.
    suffix_for_output_name: Option<String>,
    /// Java private field `formatOfOutputFile`: "FormatOfOutputFile is an
    /// ImageOutputFormat instance."  Java declares it `EnumeratedType` and only
    /// calls `toString` on it.
    format_of_output_file: Option<ImageOutputFormat>,
}

impl ChunksetupParam {
    /// Java `ChunksetupParam(DialogType)`.
    pub fn new(dialog_type: Option<DialogType>) -> ChunksetupParam {
        let mode = if dialog_type == Some(DialogType::AnisotropicDiffusion) {
            Mode::Nad
        } else {
            Mode::Parallel
        };
        ChunksetupParam {
            command: Mutex::new(Vec::new()),
            megavoxel_maximum: EtomoNumber::new_with_name("-m"),
            overlap_pixels: EtomoNumber::new_with_name("-o"),
            mode,
            subdir_name: None,
            command_file: None,
            input_file: Some(String::new()),
            output_file: Some(String::new()),
            debug: false,
            overlap: OVERLAP_MIN,
            overlap_times_four: false,
            one_line_command: None,
            input_image_file: None,
            suffix_for_output_name: None,
            format_of_output_file: None,
        }
    }

    /// Java `setOverlapPixels(Number)`.
    pub fn set_overlap_pixels(&mut self, overlap_pixels: Option<Number>) {
        self.overlap_pixels.set_number(overlap_pixels);
    }

    /// Java `setMemoryPerChunk(Number)`.
    pub fn set_memory_per_chunk(&mut self, memory: Number) {
        self.megavoxel_maximum
            .set_int(memory.int_value() / MEMORY_TO_VOXEL);
    }

    /// Java `setMegavoxelMaximum(Number)`.
    pub fn set_megavoxel_maximum(&mut self, megavoxel_maximum: Option<Number>) {
        self.megavoxel_maximum.set_number(megavoxel_maximum);
    }

    /// Java `setOneLineCommand(String, String)`.
    pub fn set_one_line_command(&mut self, program: Option<&str>, arguments: Option<&str>) {
        if utilities::is_empty(program) && utilities::is_empty(arguments) {
            self.one_line_command = None;
            return;
        }
        let one_line_command = self.one_line_command.get_or_insert_with(|| [None, None]);
        one_line_command[ONE_LINE_COMMAND_PROGRAM_INDEX] = program.map(str::to_string);
        one_line_command[ONE_LINE_COMMAND_ARGUMENTS_INDEX] = arguments.map(str::to_string);
    }

    /// Java `setInputImageFile(String)`.
    pub fn set_input_image_file(&mut self, input_image_file: Option<&str>) {
        self.input_image_file = input_image_file.map(str::to_string);
    }

    /// Java `setSuffixForOutputName(String)`.
    pub fn set_suffix_for_output_name(&mut self, suffix_for_output_name: Option<&str>) {
        self.suffix_for_output_name = suffix_for_output_name.map(str::to_string);
    }

    /// Java `setFormatOfOutputFile(EnumeratedType)`.
    pub fn set_format_of_output_file(&mut self, format_of_output_file: Option<ImageOutputFormat>) {
        self.format_of_output_file = format_of_output_file;
    }

    /// Java `setSubdirName(String)`.
    pub fn set_subdir_name(&mut self, input: Option<&str>) {
        self.subdir_name = input.map(str::to_string);
    }

    /// Java `setCommandFile(String)`.
    pub fn set_command_file(&mut self, input: Option<&str>) {
        self.command_file = input.map(str::to_string);
    }

    /// Java `setInputFile(String)`.
    pub fn set_input_file(&mut self, input: Option<&str>) {
        self.input_file = input.map(str::to_string);
    }

    /// Java `setOutputFile(String)`.
    pub fn set_output_file(&mut self, input: Option<&str>) {
        self.output_file = input.map(str::to_string);
    }

    /// Java `setOverlap(Number)`; the input should be an Integer.
    pub fn set_overlap(&mut self, input: Number) {
        self.overlap = input.int_value();
    }

    /// Java `setOverlapTimesFour(boolean)`.
    pub fn set_overlap_times_four(&mut self, input: bool) {
        self.overlap_times_four = input;
    }

    /// Java private `buildCommand`.
    fn build_command(&self, command: &mut Vec<String>) {
        command.clear();
        command.push("python".to_string());
        command.push("-u".to_string());
        let python_script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_string();
        command.push(format!("{python_script_path}{PROCESS_NAME}"));
        if self.mode == Mode::Nad {
            command.push("-p".to_string());
            command.push("0".to_string());
            command.push("-o".to_string());
            // Set -o to overlap or 4 times overlap. Minimum is OVERLAP_MIN.
            let mut calc_overlap = self.overlap;
            if self.overlap_times_four {
                calc_overlap = calc_overlap.wrapping_mul(4);
            }
            if calc_overlap < OVERLAP_MIN {
                command.push(OVERLAP_MIN.to_string());
            } else {
                command.push(calc_overlap.to_string());
            }
            command.push(self.megavoxel_maximum.get_name().to_string());
            command.push(self.megavoxel_maximum.to_string());
            command.push("-no".to_string());
            // ChunksetupParam.java:196-197: `new File(subdirName)` and
            // `new File(subdir, commandFile)` throw NullPointerException when either
            // is null.  Fixed in translation: a null subdirectory is no parent, and
            // a null command file is Java's "null" name.
            let command_file = self.command_file.as_deref().unwrap_or("null");
            let path = match &self.subdir_name {
                None => PathBuf::from(command_file),
                Some(subdir_name) => Path::new(subdir_name).join(command_file),
            };
            command.push(path.to_string_lossy().into_owned());
            command.push(format!(
                "..{}{}",
                std::path::MAIN_SEPARATOR,
                self.input_file.as_deref().unwrap_or("null")
            ));
            command.push(format!(
                "..{}{}",
                std::path::MAIN_SEPARATOR,
                self.output_file.as_deref().unwrap_or("null")
            ));
        } else {
            // ChunksetupParam.java:202 indexes `oneLineCommand` without a null test,
            // so a parallel param with no one-line command throws
            // NullPointerException.  Fixed in translation: no command means no -c.
            let (program, arguments) = match &self.one_line_command {
                None => (None, None),
                Some(one_line_command) => (
                    one_line_command[ONE_LINE_COMMAND_PROGRAM_INDEX].as_deref(),
                    one_line_command[ONE_LINE_COMMAND_ARGUMENTS_INDEX].as_deref(),
                ),
            };
            if !utilities::is_empty(program) {
                command.push("-c".to_string());
                command.push(format!(
                    "{}{}",
                    program.unwrap(),
                    if !utilities::is_empty(arguments) {
                        format!(" {}", arguments.unwrap())
                    } else {
                        String::new()
                    }
                ));
            }
            if !utilities::is_empty(self.input_image_file.as_deref()) {
                command.push("-i".to_string());
                command.push(self.input_image_file.clone().unwrap());
            }
            if !utilities::is_empty(self.suffix_for_output_name.as_deref()) {
                command.push("-s".to_string());
                command.push(self.suffix_for_output_name.clone().unwrap());
            }
            if let Some(format_of_output_file) = self.format_of_output_file {
                command.push("-f".to_string());
                command.push(format_of_output_file.to_string());
            }
            if !self.overlap_pixels.is_null() {
                command.push(self.overlap_pixels.get_name().to_string());
                command.push(self.overlap_pixels.to_string());
            }
            if !self.megavoxel_maximum.is_null() {
                command.push(self.megavoxel_maximum.get_name().to_string());
                command.push(self.megavoxel_maximum.to_string());
            }
        }
        if self.debug {
            for element in command.iter() {
                eprint!("{element} ");
            }
            if !command.is_empty() {
                eprintln!();
            }
        }
    }
}

impl Command for ChunksetupParam {
    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command = self.command.lock().unwrap();
        if command.is_empty() {
            self.build_command(&mut command);
        }
        if command.is_empty() {
            return Some(Vec::new());
        }
        if command.len() == 1 {
            return Some(vec![command[0].clone()]);
        }
        Some(command.clone())
    }

    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandInputFile`.  `new File(null)` throws
    /// NullPointerException in Java when the file was never set; `None` here.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        if self.mode == Mode::Nad {
            return self.input_file.as_ref().map(PathBuf::from);
        }
        self.input_image_file.as_ref().map(PathBuf::from)
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        let mut command = self.command.lock().unwrap();
        if command.is_empty() {
            self.build_command(&mut command);
        }
        let mut command_line = String::new();
        for element in command.iter() {
            command_line.push_str(&format!("{element} "));
        }
        Some(command_line)
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        if self.mode == Mode::Nad {
            return self.output_file.as_ref().map(PathBuf::from);
        }
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java deprecated `getOutputImageFileType`.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java deprecated `getOutputImageFileType2`.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// The `CommandDetails` view: this class implements `ProcessDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for ChunksetupParam {
    /// Java `getLogMessage`, which returns null: nothing to log.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }

    /// Java `getName`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }
}

/// Every Java `ProcessDetails` getter except `getString` throws
/// `IllegalArgumentException("field=" + field)`; that is `None` here.
impl ProcessDetails for ChunksetupParam {
    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
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

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        if field_interface::as_field::<Field>(field) == Some(&Field::OneLineCommandProgram) {
            return match &self.one_line_command {
                Some(one_line_command) => one_line_command[ONE_LINE_COMMAND_PROGRAM_INDEX].clone(),
                None => None,
            };
        }
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }
}
