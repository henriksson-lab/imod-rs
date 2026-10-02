//! `IMOD/Etomo/src/etomo/comscript/XftoxgParam.java`.
//!
//! **Java `EnumeratedType` parameters.**  `setHybridFits(EnumeratedType)` and
//! `setNumberToFit(EnumeratedType)` read only `getValue()`; the only enumerated types
//! a caller passes are this unit's own `HybridFits` and `NumberToFit` (the
//! SerialSectionsDialog radio buttons), and there is no Rust `EnumeratedType` unit, so
//! the Rust parameters are those two types.
//!
//! **The cached command array.**  `getCommandArray` builds `commandArray` once and
//! keeps it; the `Command` interface is reached through a shared reference here, so
//! the cache sits behind a `Mutex`.

use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities::java_io_file_new;

/// Java `PROCESS_NAME`.
pub const PROCESS_NAME: ProcessName = ProcessName::XFTOXG;
/// Java `NUMBER_TO_FIT_KEY`.
pub const NUMBER_TO_FIT_KEY: &str = "NumberToFit";
/// Java `HYBRID_FITS_KEY`.
pub const HYBRID_FITS_KEY: &str = "HybridFits";
/// Java `REFERENCE_SECTION`.
pub const REFERENCE_SECTION: &str = "ReferenceSection";
/// Java private `debug`.
const DEBUG: bool = false;
/// Java private `COMMAND_SIZE`.
const COMMAND_SIZE: usize = 1;

/// Java `COMMAND_NAME = PROCESS_NAME.toString()`.
pub fn command_name() -> String {
    PROCESS_NAME.to_string()
}

/// Java final `XftoxgParam implements Command`.
pub struct XftoxgParam {
    reference_section: EtomoNumber,
    hybrid_fits: EtomoNumber,
    number_to_fit: EtomoNumber,
    manager: &'static dyn BaseManager,
    xf_file_name: String,
    xg_file_name: String,
    command_array: Mutex<Option<Vec<String>>>,
}

impl XftoxgParam {
    /// Java `XftoxgParam(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> XftoxgParam {
        XftoxgParam {
            reference_section: EtomoNumber::new(),
            hybrid_fits: EtomoNumber::new(),
            number_to_fit: EtomoNumber::new(),
            manager,
            xf_file_name: String::new(),
            xg_file_name: String::new(),
            command_array: Mutex::new(None),
        }
    }

    /// Java private `genOptions`.
    fn gen_options(&self) -> Vec<String> {
        let mut options = Vec::new();
        if !self.number_to_fit.is_null() {
            options.push(format!("-{NUMBER_TO_FIT_KEY}"));
            options.push(self.number_to_fit.to_string());
        }
        if !self.reference_section.is_null() {
            options.push(format!("-{REFERENCE_SECTION}"));
            options.push(self.reference_section.to_string());
        }
        if !self.hybrid_fits.is_null() {
            options.push(format!("-{HYBRID_FITS_KEY}"));
            options.push(self.hybrid_fits.to_string());
        }
        options.push(self.xf_file_name.clone());
        options.push(self.xg_file_name.clone());
        options
    }

    /// Java `setReferenceSection(ConstEtomoNumber)`.
    pub fn set_reference_section_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        self.reference_section.set_const_etomo_number(input);
    }

    /// Java `setReferenceSection(Number)`.
    pub fn set_reference_section_number(&mut self, input: Option<Number>) {
        self.reference_section.set_number(input);
    }

    /// Java `resetReferenceSection`.
    pub fn reset_reference_section(&mut self) {
        self.reference_section.reset();
    }

    /// Java `setXfFileName`.
    pub fn set_xf_file_name(&mut self, input: &str) {
        self.xf_file_name = input.to_string();
    }

    /// Java `setXgFileName`.
    pub fn set_xg_file_name(&mut self, input: &str) {
        self.xg_file_name = input.to_string();
    }

    /// Java `setHybridFits(EnumeratedType)`.
    pub fn set_hybrid_fits(&mut self, enum_type: HybridFits) {
        self.hybrid_fits
            .set_const_etomo_number(Some(&enum_type.get_value()));
    }

    /// Java `resetHybridFits`.
    pub fn reset_hybrid_fits(&mut self) {
        self.hybrid_fits.reset();
    }

    /// Java `setNumberToFit(int)`.
    pub fn set_number_to_fit_int(&mut self, input: i32) {
        self.number_to_fit.set_int(input);
    }

    /// Java `setNumberToFit(EnumeratedType)`.
    pub fn set_number_to_fit(&mut self, enum_type: NumberToFit) {
        self.number_to_fit
            .set_const_etomo_number(Some(&enum_type.get_value()));
    }

    /// Java `resetNumberToFit`.
    pub fn reset_number_to_fit(&mut self) {
        self.number_to_fit.reset();
    }

    /// Java `getHybridFits`.
    pub fn get_hybrid_fits(&self) -> i32 {
        self.hybrid_fits.get_int()
    }

    /// Java `isHybridFitsEmpty`.
    pub fn is_hybrid_fits_empty(&self) -> bool {
        self.hybrid_fits.is_null()
    }

    /// Java `getReferenceSection`.
    pub fn get_reference_section(&self) -> i32 {
        self.reference_section.get_int()
    }

    /// Java `isReferenceSectionEmpty`.
    pub fn is_reference_section_empty(&self) -> bool {
        self.reference_section.is_null()
    }

    /// Java `getNumberToFit`.
    pub fn get_number_to_fit(&self) -> i32 {
        self.number_to_fit.get_int()
    }

    /// Java `isNumberToFitEmpty`.
    pub fn is_number_to_fit_empty(&self) -> bool {
        self.number_to_fit.is_null()
    }
}

impl Command for XftoxgParam {
    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(command_name())
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command_array = self.command_array.lock().unwrap();
        if command_array.is_none() {
            let options = self.gen_options();
            let mut array = vec![String::new(); options.len() + COMMAND_SIZE];
            // `BaseManager.getIMODBinPath() + COMMAND_NAME`: a null path concatenates
            // as "null".
            array[0] = format!(
                "{}{}",
                base_manager::get_imod_bin_path().unwrap_or("null".to_string()),
                command_name()
            );
            for i in 0..options.len() {
                array[i + COMMAND_SIZE] = options[i].clone();
            }
            if DEBUG {
                let mut buffer = String::new();
                for i in 0..array.len() {
                    buffer.push_str(&array[i]);
                    if i < array.len() - 1 {
                        buffer.push(' ');
                    }
                }
                eprintln!("{}", buffer);
            }
            *command_array = Some(array);
        }
        command_array.clone()
    }

    /// Java `getCommandLine`.
    ///
    /// XftoxgParam.java:175-184 has two defects, both fixed here.  It reads
    /// `commandArray.length` without building the array, so a call before
    /// `getCommandArray` throws NullPointerException; here the array is built first.
    /// And its loop starts at 0 after the buffer was already seeded with
    /// `commandArray[0]`, so the executable is written twice (`xftoxg xftoxg -...`);
    /// here the loop starts at 1, so the line is the command array joined by spaces.
    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array().unwrap_or_default();
        if command_array.is_empty() {
            return Some(String::new());
        }
        let mut buffer = command_array[0].clone();
        for i in 1..command_array.len() {
            buffer.push_str(&format!(" {}", command_array[i]));
        }
        Some(buffer)
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(command_name())
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        if self.xg_file_name.is_empty() {
            return None;
        }
        // `new File(manager.getPropertyUserDir(), xgFileName)`: a null parent is
        // `new File(xgFileName)`.
        Some(std::path::PathBuf::from(
            match self.manager.get_property_user_dir() {
                None => self.xg_file_name.clone(),
                Some(dir) => java_io_file_new(&dir, &self.xg_file_name),
            },
        ))
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        Some(file_type::CLASS.transformed_refining_model.clone())
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        let file_key: &FileKey = &file_type::CLASS.transformed_refining_model;
        Some(file_key.clone())
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        None
    }
}

/// Java nested final `HybridFits implements EnumeratedType`.  The three singletons
/// are constants; `value` is the `EtomoNumber` the source holds, built on demand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HybridFits {
    value: i32,
}

impl HybridFits {
    /// Java `ROTATION`.
    pub const ROTATION: HybridFits = HybridFits { value: 1 };
    /// Java `TRANSLATIONS`.
    pub const TRANSLATIONS: HybridFits = HybridFits { value: 2 };
    /// Java `TRANSLATIONS_ROTATIONS`.
    pub const TRANSLATIONS_ROTATIONS: HybridFits = HybridFits { value: 3 };

    /// Java `getValue`.
    pub fn get_value(&self) -> ConstEtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(self.value);
        (*value).clone()
    }

    /// Java `isDefault`.
    pub fn is_default(&self) -> bool {
        false
    }

    /// Java `getLabel`.
    pub fn get_label(&self) -> Option<String> {
        None
    }

    /// Java `equals(int)`.
    pub fn equals_int(&self, input: i32) -> bool {
        self.get_value().equals_int(input)
    }
}

/// Java `toString`.
impl std::fmt::Display for HybridFits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_value())
    }
}

/// Java nested final `NumberToFit implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumberToFit {
    value: i32,
}

impl NumberToFit {
    /// Java `GLOBAL_ALIGNMENT`.
    pub const GLOBAL_ALIGNMENT: NumberToFit = NumberToFit { value: 0 };

    /// Java `getValue`.
    pub fn get_value(&self) -> ConstEtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(self.value);
        (*value).clone()
    }

    /// Java `isDefault`.
    pub fn is_default(&self) -> bool {
        false
    }

    /// Java `getLabel`.
    pub fn get_label(&self) -> Option<String> {
        None
    }

    /// Java `equals(int)`.
    pub fn equals_int(&self, input: i32) -> bool {
        self.get_value().equals_int(input)
    }
}

/// Java `toString`.
impl std::fmt::Display for NumberToFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_value())
    }
}
