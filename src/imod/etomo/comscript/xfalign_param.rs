//! `IMOD/Etomo/src/etomo/comscript/XfalignParam.java`.
//!
//! The `python -u xfalign -PID ...` command line for the join and serial sections
//! auto-alignment.
//!

use std::path::PathBuf;
use std::sync::Mutex;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::fortran_input_string::FortranInputString;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Number, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::transform::Transform;

pub const PRE_CROSS_CORRELATION_KEY: &str = "PreCrossCorrelation";
pub const EDGE_TO_IGNORE_KEY: &str = "EdgeToIgnore";
pub const REDUCE_BY_BINNING_KEY: &str = "ReduceByBinning";
pub const SKIP_SECTIONS_KEY: &str = "SkipSections";
pub const SECTIONS_NUMBERED_FROM_ONE_KEY: &str = "SectionsNumberedFromOne";
/// Java private static `commandSize`.
const COMMAND_SIZE: usize = 4;
/// Java private static `commandName`.
const COMMAND_NAME: &str = "xfalign";
/// Java private static `outputFileExtension`.
const OUTPUT_FILE_EXTENSION: &str = "_auto.xf";
/// Java private static `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java nested `XfalignParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `INITIAL`, "Initial".
    Initial,
    /// Java `REFINE`, "Refine".
    Refine,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Initial => "Initial",
            Mode::Refine => "Refine",
        })
    }
}

impl CommandMode for Mode {}

/// Java `XfalignParam implements Command`.
pub struct XfalignParam {
    reduce_by_binning: EtomoNumber,
    edge_to_ignore: EtomoNumber,
    warp_patch_size: FortranInputString,
    shift_limits_for_warp: FortranInputString,
    auto_alignment_meta_data: &'static AutoAlignmentMetaData,
    root_name: Option<String>,
    output_file_name: String,
    output_file: PathBuf,
    tomogram_averages: bool,
    mode: Mode,
    manager: &'static dyn BaseManager,
    input_file_name: Option<String>,
    /// Java field `commandArray`, built once by `getCommandArray`; a `Mutex` because
    /// the `Command` methods take `&self`.
    command_array: Mutex<Option<Vec<String>>>,
    pre_cross_correlation: bool,
    skip_sections: Option<String>,
    sections_numbered_from_one: bool,
    boundary_model: bool,
    sobel_filter: bool,
}

impl XfalignParam {
    /// Java `XfalignParam(BaseManager, AutoAlignmentMetaData, Mode, boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        auto_alignment_meta_data: &'static AutoAlignmentMetaData,
        mode: Mode,
        tomogram_averages: bool,
    ) -> XfalignParam {
        let root_name = manager.get_name();
        // `rootName + outputFileExtension`: a null name concatenates as "null"
        let output_file_name = format!(
            "{}{}",
            root_name.as_deref().unwrap_or("null"),
            OUTPUT_FILE_EXTENSION
        );
        // `new File(manager.getPropertyUserDir(), outputFileName)`
        let output_file = match manager.get_property_user_dir() {
            Some(dir) => PathBuf::from(dir).join(&output_file_name),
            None => PathBuf::from(&output_file_name),
        };
        let mut warp_patch_size = FortranInputString::new(2);
        let mut shift_limits_for_warp = FortranInputString::new(2);
        warp_patch_size.set_integer_type(true);
        shift_limits_for_warp.set_integer_type(true);
        XfalignParam {
            reduce_by_binning: EtomoNumber::new(),
            edge_to_ignore: EtomoNumber::new_with_type(Some(Type::Double)),
            warp_patch_size,
            shift_limits_for_warp,
            auto_alignment_meta_data,
            root_name,
            output_file_name,
            output_file,
            tomogram_averages,
            mode,
            manager,
            input_file_name: None,
            command_array: Mutex::new(None),
            pre_cross_correlation: false,
            skip_sections: None,
            sections_numbered_from_one: false,
            boundary_model: false,
            sobel_filter: false,
        }
    }

    /// Java static `getName()`.
    pub fn get_name() -> String {
        COMMAND_NAME.to_string()
    }

    /// Java static `getOutputFileExtension()`.
    pub fn get_output_file_extension() -> String {
        OUTPUT_FILE_EXTENSION.to_string()
    }

    /// Java private `genOptions()`.
    fn gen_options(&self) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        if self.tomogram_averages {
            options.push("-tomo".to_string());
        }
        // The source's final `else` throws IllegalArgumentException("Unknown mode");
        // `Mode` has only these two values, so it cannot be reached here.
        if self.mode == Mode::Initial {
            options.push("-pre".to_string());
        } else if self.mode == Mode::Refine {
            options.push("-ini".to_string());
            options.push(format!(
                "{}.xf",
                self.root_name.as_deref().unwrap_or("null")
            ));
        }
        self.gen_filter_options(&mut options);
        // Java `if (transform == null) transform = Transform.DEFAULT`: the Rust
        // `AutoAlignmentMetaData` holds no null transform, so the test is always false.
        let transform = self.auto_alignment_meta_data.get_align_transform();
        if self.sobel_filter {
            options.push("-sobel".to_string());
        }
        options.push("-par".to_string());
        options.push(transform.get_value().to_string());
        if !self.edge_to_ignore.is_null() {
            options.push("-matt".to_string());
            options.push(self.edge_to_ignore.to_string());
        }
        if !self.reduce_by_binning.is_null() {
            options.push("-reduce".to_string());
            options.push(self.reduce_by_binning.to_string());
        }
        if self.pre_cross_correlation && self.mode != Mode::Refine {
            options.push("-prexcorr".to_string());
        }
        if let Some(skip_sections) = &self.skip_sections {
            options.push("-skip".to_string());
            options.push(skip_sections.clone());
        }
        if self.sections_numbered_from_one {
            options.push("-one".to_string());
        }
        if !self.warp_patch_size.is_null() {
            options.push("-warp".to_string());
            options.push(self.warp_patch_size.to_string());
        }
        if self.boundary_model {
            options.push("-boundary".to_string());
            options.push(
                file_type::CLASS
                    .auto_align_boundary_model
                    .get_file_name(Some(self.manager), Some(AXIS_ID))
                    .unwrap_or_else(|| "null".to_string()),
            );
        }
        if !self.shift_limits_for_warp.is_null() {
            options.push("-shift".to_string());
            options.push(self.shift_limits_for_warp.to_string());
        }
        // A null input file name is a null element in Java's list, which
        // `ProcessBuilder` rejects; "null" here.
        options.push(
            self.input_file_name
                .clone()
                .unwrap_or_else(|| "null".to_string()),
        );
        options.push(self.output_file_name.clone());
        options
    }

    /// Java private `genFilterOptions(ArrayList)`.
    fn gen_filter_options(&self, options: &mut Vec<String>) {
        let sigma_low_frequency = self
            .auto_alignment_meta_data
            .get_sigma_low_frequency_parameter();
        let cutoff_high_frequency = self
            .auto_alignment_meta_data
            .get_cutoff_high_frequency_parameter();
        let sigma_high_frequency = self
            .auto_alignment_meta_data
            .get_sigma_high_frequency_parameter();
        // optional
        if (self
            .auto_alignment_meta_data
            .is_sigma_low_frequency_enabled()
            && sigma_low_frequency.is_not_null_and_not_default())
            || (self
                .auto_alignment_meta_data
                .is_cutoff_high_frequency_enabled()
                && cutoff_high_frequency.is_not_null_and_not_default())
            || (self
                .auto_alignment_meta_data
                .is_sigma_high_frequency_enabled()
                && sigma_high_frequency.is_not_null_and_not_default())
        {
            options.push("-fil".to_string());
            // all three numbers must exist
            options.push(format!(
                "{},{},0,{}",
                sigma_low_frequency.to_defaulted_string(),
                sigma_high_frequency.to_defaulted_string(),
                cutoff_high_frequency.to_defaulted_string()
            ));
        }
    }

    /// Java `resetSobelFilter()`.
    pub fn reset_sobel_filter(&mut self) {
        self.sobel_filter = false;
    }

    /// Java `setSobelFilter(boolean)`.
    pub fn set_sobel_filter(&mut self, input: bool) {
        self.sobel_filter = input;
    }

    /// Java `setReduceByBinning(Number)`.
    pub fn set_reduce_by_binning(&mut self, input: Option<Number>) {
        self.reduce_by_binning.set_number(input);
    }

    /// Java `resetReduceByBinning()`.
    pub fn reset_reduce_by_binning(&mut self) {
        self.reduce_by_binning.reset();
    }

    /// Java `setBoundaryModel(boolean)`.
    pub fn set_boundary_model(&mut self, input: bool) {
        self.boundary_model = input;
    }

    /// Java `resetBoundaryModel()`.
    pub fn reset_boundary_model(&mut self) {
        self.boundary_model = false;
    }

    /// Java `setEdgeToIgnore(String)`.
    pub fn set_edge_to_ignore(&mut self, input: Option<&str>) {
        self.edge_to_ignore.set_string(input);
    }

    /// Java `resetEdgeToIgnore()`.
    pub fn reset_edge_to_ignore(&mut self) {
        self.edge_to_ignore.reset();
    }

    /// Java `setSkipSectionsFrom1(String)`.
    pub fn set_skip_sections_from1(&mut self, input: Option<&str>) {
        // `input.matches("\\s*")`
        if input.is_none_or(|input| {
            input
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        }) {
            self.sections_numbered_from_one = false;
            self.skip_sections = None;
        } else {
            self.sections_numbered_from_one = true;
            self.skip_sections = input.map(|input| input.to_string());
        }
    }

    /// Java `resetSkipSectionsFrom1()`.
    pub fn reset_skip_sections_from1(&mut self) {
        self.sections_numbered_from_one = false;
        self.skip_sections = None;
    }

    /// Java `setPreCrossCorrelation(boolean)`.
    pub fn set_pre_cross_correlation(&mut self, input: bool) {
        self.pre_cross_correlation = input;
    }

    /// Java `setWarpPatchSize(String, String)`.
    pub fn set_warp_patch_size(&mut self, x: Option<&str>, y: Option<&str>) {
        self.warp_patch_size.set_index_string(0, x);
        self.warp_patch_size.set_index_string(1, y);
    }

    /// Java `resetWarpPatchSize()`.
    pub fn reset_warp_patch_size(&mut self) {
        self.warp_patch_size.reset();
    }

    /// Java `setShiftLimitsForWarp(String, String)`.
    pub fn set_shift_limits_for_warp(&mut self, x: Option<&str>, y: Option<&str>) {
        self.shift_limits_for_warp.set_index_string(0, x);
        self.shift_limits_for_warp.set_index_string(1, y);
    }

    /// Java `resetShiftLimitsForWarp()`.
    pub fn reset_shift_limits_for_warp(&mut self) {
        self.shift_limits_for_warp.reset();
    }

    /// Java `setInputFileName(String)`.
    pub fn set_input_file_name(&mut self, input: Option<&str>) {
        self.input_file_name = input.map(|input| input.to_string());
    }
}

impl Command for XfalignParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        AXIS_ID
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command_array = self.command_array.lock().unwrap();
        if command_array.is_none() {
            let options = self.gen_options();
            let mut array = vec![String::new(); options.len() + COMMAND_SIZE];
            array[0] = "python".to_string();
            array[1] = "-u".to_string();
            let script_path = etomo_director::INSTANCE
                .get_python_script_path()
                .as_deref()
                .unwrap_or("null")
                .to_owned();
            array[2] = format!("{script_path}{COMMAND_NAME}");
            array[3] = "-PID".to_string();
            for i in 0..options.len() {
                array[i + COMMAND_SIZE] = options[i].clone();
            }
            *command_array = Some(array);
        }
        command_array.clone()
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array().unwrap_or_default();
        let mut buffer = String::new();
        for i in 0..command_array.len() {
            buffer.push_str(&format!("{} ", command_array[i]));
        }
        Some(buffer)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::XFALIGN)
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        Some(self.output_file.clone())
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

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
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
}
