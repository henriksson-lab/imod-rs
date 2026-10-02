//! `IMOD/Etomo/src/etomo/comscript/MidasParam.java`.
//!
//! `getCommandArray` builds the array on first use through the `Command`
//! trait's `&self`, so the cached array sits behind a `Mutex`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Number, Type, java_lang_double_to_string};
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::section_table_row_data::SectionTableRowData;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static `commandSize`.
const COMMAND_SIZE: usize = 1;
/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::MIDAS;
/// Java private static `commandName`.
const COMMAND_NAME: &str = "midas";
/// Java private static `outputFileExtension`.
const OUTPUT_FILE_EXTENSION: &str = "_midas.xf";

/// Java final `MidasParam`.
pub struct MidasParam {
    binning: EtomoNumber,
    #[allow(dead_code)]
    working_dir: Option<String>,
    output_file: PathBuf,
    root_name: Option<String>,
    output_file_name: String,
    axis_id: AxisID,
    mode: Mode,
    manager: &'static dyn BaseManager,
    section_table_row_data: Option<Vec<Arc<SectionTableRowData>>>,
    input_file_name: Option<String>,
    command_array: Mutex<Option<Vec<String>>>,
    image_rotation: ScriptParameter,
}

impl MidasParam {
    /// Java `MidasParam(BaseManager, AxisID, Mode)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID, mode: Mode) -> MidasParam {
        let working_dir = manager.get_property_user_dir();
        let root_name = manager.get_name();
        // Java string concatenation writes a null root name as "null"
        let output_file_name = format!(
            "{}{OUTPUT_FILE_EXTENSION}",
            root_name.as_deref().unwrap_or("null")
        );
        // `new File(workingDir, outputFileName)`: a null parent resolves the child alone
        let output_file = match working_dir.as_deref() {
            None => PathBuf::from(&output_file_name),
            Some(working_dir) => {
                PathBuf::from(utilities::java_io_file_new(working_dir, &output_file_name))
            }
        };
        MidasParam {
            binning: EtomoNumber::new(),
            working_dir,
            output_file,
            root_name,
            output_file_name,
            axis_id,
            mode,
            manager,
            section_table_row_data: None,
            input_file_name: None,
            command_array: Mutex::new(None),
            image_rotation: ScriptParameter::new_with_type_and_name(Type::Double, "-a"),
        }
    }

    /// Java `getMode`.
    pub fn get_mode(&self) -> Mode {
        self.mode
    }

    /// Java private `genOptions`.  A null file name is a null option in Java, which
    /// `ProcessBuilder` rejects; it is kept as `None` here and dropped by
    /// `getCommandArray`.
    fn gen_options(&self) -> Vec<Option<String>> {
        let mut options: Vec<Option<String>> = Vec::new();
        // Midas must not fork.
        options.push(Some("-D".to_owned()));
        if !self.binning.is_null() && self.binning.gt_int(1) {
            options.push(Some("-B".to_owned()));
            options.push(Some(self.binning.to_string()));
        }
        if self.mode == Mode::RawStack {
            if !self.image_rotation.is_null() {
                options.push(Some("-a".to_owned()));
                options.push(Some(java_lang_double_to_string(
                    -1.0 * self.image_rotation.get_double(),
                )));
            }
            options.push(Some("-t".to_owned()));
            options.push(
                file_type::CLASS
                    .raw_tilt_angles
                    .get_file_name(Some(self.manager), Some(self.axis_id)),
            );
            options.push(self.input_file_name.clone());
            options.push(
                file_type::CLASS
                    .pre_transformation_list
                    .get_file_name(Some(self.manager), Some(self.axis_id)),
            );
        }
        if self.mode == Mode::Sample {
            // If the section table is not available, don't use the chunks option.
            if let Some(section_table_row_data) = self.section_table_row_data.as_ref() {
                let section_data_size = section_table_row_data.len() as i32;
                let mut chunk_size = String::new();
                options.push(Some("-cs".to_owned()));
                let mut n_slices: i32;
                for i in 0..section_data_size {
                    let data: &dyn ConstSectionTableRowData =
                        section_table_row_data[i as usize].as_ref();
                    // Order: section 1 - top, section 2 - bottom, section 2 - top, section 3 -
                    // bottom.
                    n_slices = data.get_sample_bottom_number_slices(section_data_size);
                    if n_slices != -1 {
                        chunk_size.push_str(&n_slices.to_string());
                        if i < section_data_size - 1 {
                            chunk_size.push(',');
                        }
                    }
                    n_slices = data.get_sample_top_number_slices(section_data_size);
                    if n_slices != -1 {
                        chunk_size.push_str(&format!("{n_slices},"));
                    }
                }
                options.push(Some(chunk_size));
            }
        }
        if self.mode == Mode::FixEdges {
            options.push(Some("-p".to_owned()));
            options.push(
                file_type::CLASS
                    .piece_list
                    .get_file_name(Some(self.manager), Some(self.axis_id)),
            );
        }
        if self.mode != Mode::RawStack {
            options.push(Some("-b".to_owned()));
            options.push(Some("0".to_owned()));
        }
        if self.mode == Mode::Sample {
            // options.add("-D");
            options.push(Some("-o".to_owned()));
            options.push(Some(self.output_file_name.clone()));
            options.push(self.input_file_name.clone());
            // Java string concatenation writes a null root name as "null"
            options.push(Some(format!(
                "{}.xf",
                self.root_name.as_deref().unwrap_or("null")
            )));
        } else if self.mode == Mode::FixEdges {
            options.push(Some("-q".to_owned()));
            options.push(self.input_file_name.clone());
            options.push(
                file_type::CLASS
                    .piece_shifts
                    .get_file_name(Some(self.manager), Some(self.axis_id)),
            );
        }
        options
    }

    /// Java `setSectionTableRowData(ArrayList)`.
    pub fn set_section_table_row_data(&mut self, input: Option<Vec<Arc<SectionTableRowData>>>) {
        self.section_table_row_data = input;
    }

    /// Java `setImageRotation(double)`.
    pub fn set_image_rotation(&mut self, input: f64) {
        self.image_rotation.set_double(input);
    }

    /// Java `setInputFileName(String)`.
    pub fn set_input_file_name(&mut self, input: Option<&str>) {
        self.input_file_name = input.map(str::to_owned);
    }

    /// Java `getInputFileName`.
    pub fn get_input_file_name(&self) -> Option<String> {
        self.input_file_name.clone()
    }

    /// Java `setBinning(Number)`.
    pub fn set_binning(&mut self, input: Option<Number>) {
        self.binning.set_number(input);
    }

    /// Java static `getName`.
    pub fn get_name() -> &'static str {
        COMMAND_NAME
    }

    /// Java static `getOutputFileExtension`.
    pub fn get_output_file_extension() -> &'static str {
        OUTPUT_FILE_EXTENSION
    }
}

impl Command for MidasParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command_array = self.command_array.lock().unwrap();
        if command_array.is_none() {
            let options = self.gen_options();
            let mut array: Vec<String> = Vec::with_capacity(options.len() + COMMAND_SIZE);
            // Java string concatenation writes a null path as "null"
            array.push(format!(
                "{}{COMMAND_NAME}",
                base_manager::get_imod_bin_path()
                    .as_deref()
                    .unwrap_or("null")
            ));
            for option in options.into_iter().flatten() {
                array.push(option);
            }
            *command_array = Some(array);
        }
        command_array.clone()
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array().unwrap_or_default();
        let mut buffer = String::new();
        for element in &command_array {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        Some(self.output_file.clone())
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

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }
}

/// Java static final nested class `MidasParam.Mode` (not a `CommandMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `SAMPLE`.
    Sample,
    /// Java `FIX_EDGES`.
    FixEdges,
    /// Java `RAW_STACK`.
    RawStack,
}
