//! `IMOD/Etomo/src/etomo/comscript/AnisotropicDiffusionParam.java`.
//!
//! Parameters of `nad_eed_3d` for the anisotropic diffusion interface: the test runs
//! with different K values (`nad_eed_3d-NNN.com` files run by processchunks), the
//! test run with different iterations (one `nad_eed_3d -i` process), and the
//! full-volume command file (`nad_eed_3d.com`) chunksetup splits.
//!
//! The setters take `&mut self`: they run before the param is handed to a process.
//! `command` is built lazily by the `Command` getters, which take `&self`, so it sits
//! in a `Mutex`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::storage::test_nad_file_filter::{self, TestNADFileFilter};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number, Type};
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_key::{self, FileKey};
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::r#type::iterator_element_list::IteratorElementList;
use crate::imod::etomo::r#type::parsed_array::ParsedArray;
use crate::imod::etomo::r#type::parsed_element::ParsedElement;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::anisotropic_diffusion_dialog;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java private static final `K_VALUE_TAG`.
const K_VALUE_TAG: &str = "-k";
/// Java private static final `ITERATION_TAG`.
const ITERATION_TAG: &str = "-n";
/// Java private static final `COMMAND_CHAR`.
const COMMAND_CHAR: &str = "$";
/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::ANISOTROPIC_DIFFUSION;

/// Java public static final nested class `Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `K_VALUE_LIST`.
    KValueList,
    /// Java `ITERATION`.
    Iteration,
    /// Java `K_VALUE`.
    KValue,
    /// Java `ITERATION_LIST`.
    IterationList,
}

impl FieldInterface for Field {}

/// Java public final static nested class `Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `VARYING_K = new Mode("VaryingK")`.
    VaryingK,
    /// Java `VARYING_ITERATIONS = new Mode("VaryingIterations")`.
    VaryingIterations,
    /// Java `FULL = new Mode("Full")`.
    Full,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::VaryingK => "VaryingK",
            Mode::VaryingIterations => "VaryingIterations",
            Mode::Full => "Full",
        })
    }
}

impl CommandMode for Mode {}

/// Java `public final class AnisotropicDiffusionParam implements CommandDetails`.
pub struct AnisotropicDiffusionParam {
    /// Java private final `kValueList = ParsedArray.getInstance(EtomoNumber.Type.DOUBLE,
    /// null, "K value")`.
    k_value_list: ParsedArray,
    /// Java private final `iteration = new EtomoNumber()`.
    iteration: EtomoNumber,
    /// Java private final `kValue = new EtomoNumber(EtomoNumber.Type.DOUBLE)`.
    k_value: EtomoNumber,
    /// Java private final `command = new ArrayList()`.
    command: Mutex<Vec<String>>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `iterationList`.  IterationList may contain array
    /// descriptors in the form start-end.  Example: "2,4 - 9,10".
    iteration_list: IteratorElementList,
    /// Java private final `setEnvCommand`.
    set_env_command: Option<String>,
    /// Java private final `mode`.
    mode: Mode,
    /// Java private `subdirName`, initially "".
    subdir_name: Option<String>,
    /// Java private `inputFileName`, initially "".
    input_file_name: Option<String>,
    /// Java private `debugLevel`, initially `DebugLevel.LOW`.
    debug_level: DebugLevel,
    /// Java private `imageOutputFormat`, initially `ImageOutputFormat.MRC`.
    image_output_format: Option<ImageOutputFormat>,
}

impl AnisotropicDiffusionParam {
    /// Java `AnisotropicDiffusionParam(BaseManager, CommandMode, String)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        mode: Mode,
        set_env_command: Option<&str>,
    ) -> AnisotropicDiffusionParam {
        AnisotropicDiffusionParam {
            k_value_list: ParsedArray::get_instance(Some(Type::Double), None, Some("K value")),
            iteration: EtomoNumber::new(),
            k_value: EtomoNumber::new_with_type(Some(Type::Double)),
            command: Mutex::new(Vec::new()),
            manager,
            iteration_list: IteratorElementList::new(
                Some(manager),
                Some(AxisID::Only),
                Some(anisotropic_diffusion_dialog::ITERATION_LIST_LABEL),
            ),
            set_env_command: set_env_command.map(str::to_owned),
            mode,
            subdir_name: Some(String::new()),
            input_file_name: Some(String::new()),
            debug_level: DebugLevel::LOW,
            image_output_format: Some(ImageOutputFormat::Mrc),
        }
    }

    /// Java `setKValueList(String)`.  Returns the error message if invalid.
    pub fn set_k_value_list(&mut self, input: Option<&str>) -> Option<String> {
        self.k_value_list.set_raw_string_string(input);
        if self.debug_level == DebugLevel::HIGH {
            println!(
                "AnisotropicDiffusionParam.setKValueList:kValueList={}",
                self.k_value_list
            );
        }
        self.k_value_list.validate()
    }

    /// Java `setKValue(String)`.
    pub fn set_k_value(&mut self, input: Option<&str>) {
        self.k_value.set_string(input);
    }

    /// Java `setIteration(Number)`.
    pub fn set_iteration(&mut self, input: Option<Number>) {
        self.iteration.set_number(input);
    }

    /// Java `setIterationList(String)`.
    pub fn set_iteration_list(&mut self, input: Option<&str>) -> bool {
        self.iteration_list.set_list_string(input);
        self.iteration_list.is_valid()
    }

    /// Java `setFormat(ImageOutputFormat)`.
    pub fn set_format(&mut self, image_output_format: Option<ImageOutputFormat>) {
        self.image_output_format = image_output_format;
    }

    /// Java `setDebugLevel(DebugLevel)`.
    pub fn set_debug_level(&mut self, input: DebugLevel) {
        self.debug_level = input;
    }

    /// Java `setSubdirName(String)`.
    pub fn set_subdir_name(&mut self, input: Option<&str>) {
        self.subdir_name = input.map(str::to_owned);
    }

    /// Java `getSubdirName()`.
    pub fn get_subdir_name(&self) -> Option<String> {
        self.subdir_name.clone()
    }

    /// Java `setInputFileName(String)`.
    pub fn set_input_file_name(&mut self, input: Option<&str>) {
        self.input_file_name = input.map(str::to_owned);
    }

    /// Java `getInputFileName()`.
    pub fn get_input_file_name(&self) -> Option<String> {
        self.input_file_name.clone()
    }

    /// `new File(manager.getPropertyUserDir(), subdirName)`.
    fn subdir(&self) -> String {
        let subdir_name = self.subdir_name.as_deref().unwrap_or("null");
        match self.manager.get_property_user_dir() {
            Some(property_user_dir) => utilities::java_io_file_new(&property_user_dir, subdir_name),
            None => utilities::java_io_file_normalize(subdir_name),
        }
    }

    /// Java `deleteTestFiles()`.
    pub fn delete_test_files(&self) {
        let filter = TestNADFileFilter::new();
        // Fixed in translation: Java's `listFiles` returns null for a missing
        // directory and `.length` throws NullPointerException; a missing directory
        // has no test files.
        let Ok(entries) = std::fs::read_dir(self.subdir()) else {
            return;
        };
        let mut test_file_list: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| filter.accept(path))
            .collect();
        test_file_list.sort();
        for test_file in test_file_list {
            let _ = std::fs::remove_file(test_file);
        }
    }

    /// Java `createFilterFullFile() throws LogFileException, IOException,
    /// LockException`.  Creates nad_eed_3d.com.
    pub fn create_filter_full_file(&self) -> Result<(), LogFileError> {
        let subdir = self.subdir();
        let filter_full_file = LogFile::get_instance_file(
            Some(std::path::Path::new(&utilities::java_io_file_new(
                &subdir,
                &get_filter_full_file_name(),
            ))),
            Some(self.manager.get_emergency_monitor(None)),
        )?;
        filter_full_file.create()?;
        let writer_id = filter_full_file.open_writer()?;
        if let Some(set_env_command) = &self.set_env_command {
            filter_full_file.write(
                Some(&format!("{COMMAND_CHAR}{set_env_command}")),
                &writer_id,
            )?;
            filter_full_file.new_line(&writer_id)?;
        }
        filter_full_file.write(
            Some(&format!(
                "{COMMAND_CHAR}{PROCESS_NAME} {K_VALUE_TAG} {} {ITERATION_TAG} {} INPUTFILE OUTPUTFILE",
                self.k_value, self.iteration
            )),
            &writer_id,
        )?;
        filter_full_file.new_line(&writer_id)?;
        filter_full_file.close_id(Some(&writer_id));
        Ok(())
    }

    /// Java `createTestFiles() throws LogFileException, IOException, LockException`.
    pub fn create_test_files(&self) -> Result<(), LogFileError> {
        let subdir = self.subdir();
        let mut index = EtomoNumber::new();
        let mut k = EtomoNumber::new_with_type(Some(Type::Double));
        // $nad_eed_3d -k 1234.56789 -n 3 test_input.mrc test.K1234.56789-003
        // $echo CHUNK DONE
        for i in 0..self.k_value_list.size() {
            index.set_int(i + 1);
            k.set_string(self.k_value_list.get_raw_string_int(i).as_deref());
            let test_file = LogFile::get_instance_file(
                Some(std::path::Path::new(&utilities::java_io_file_new(
                    &subdir,
                    &format!(
                        "{}{}{}",
                        test_nad_file_filter::FILE_NAME_BODY,
                        add_leading_zeros(&index.to_string(), 3),
                        test_nad_file_filter::FILE_NAME_EXT
                    ),
                ))),
                Some(self.manager.get_emergency_monitor(Some(AxisID::Only))),
            )?;
            test_file.create()?;
            let writer_id = test_file.open_writer()?;
            if let Some(set_env_command) = &self.set_env_command {
                test_file.write(
                    Some(&format!("{COMMAND_CHAR}{set_env_command}")),
                    &writer_id,
                )?;
                test_file.new_line(&writer_id)?;
            }
            test_file.write(
                Some(&format!(
                    "{COMMAND_CHAR}{PROCESS_NAME} {K_VALUE_TAG} {} {ITERATION_TAG} {} {} {}",
                    self.k_value_list
                        .get_raw_string_int(i)
                        .unwrap_or_else(|| "null".to_owned()),
                    self.iteration,
                    self.input_file_name.as_deref().unwrap_or("null"),
                    get_test_file_name(&k, &self.iteration.to_string())
                        .unwrap_or_else(|| "null".to_owned())
                )),
                &writer_id,
            )?;
            test_file.new_line(&writer_id)?;
            test_file.write(Some(&format!("{COMMAND_CHAR}echo CHUNK DONE")), &writer_id)?;
            test_file.new_line(&writer_id)?;
            test_file.close_id(Some(&writer_id));
        }
        Ok(())
    }

    /// Java private `buildCommand()`.
    fn build_command(&self, command: &mut Vec<String>) {
        // 2452 Need nad_eed_3d to add .mrc/.hdf to output files.
        let subdir =
            utilities::java_io_file_normalize(self.subdir_name.as_deref().unwrap_or("null"));
        command.push(format!(
            "{}{PROCESS_NAME}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_owned())
        ));
        command.push(K_VALUE_TAG.to_owned());
        command.push(self.k_value.to_string());
        command.push("-i".to_owned());
        command.push(self.iteration_list.to_string());
        if let Some(image_output_format) = self.image_output_format {
            command.push("-F".to_owned());
            command.push(image_output_format.to_string());
        }
        command.push("-e".to_owned());
        if self.image_output_format == Some(ImageOutputFormat::Hdf) {
            command.push(extension::CLASS.hdf.to_string());
        } else {
            command.push(extension::CLASS.mrc.to_string());
        }
        command.push("-PID".to_owned());
        command.push(utilities::java_io_file_new(
            &subdir,
            self.input_file_name.as_deref().unwrap_or("null"),
        ));
        command.push(utilities::java_io_file_new(
            &subdir,
            &get_test_file_root(&self.k_value),
        ));
        if self.debug_level.ge(DebugLevel::LOW) {
            for element in command.iter() {
                eprint!("{element} ");
            }
            if !command.is_empty() {
                eprintln!();
            }
        }
    }

    /// Java `getIteratorElementList(FieldInterface)` on this class (the
    /// `ProcessDetails` getter returns a copy).
    pub fn get_iterator_element_list_field(
        &self,
        field: &dyn FieldInterface,
    ) -> Option<&IteratorElementList> {
        if field_interface::as_field::<Field>(field) == Some(&Field::IterationList) {
            return Some(&self.iteration_list);
        }
        None
    }
}

/// Java static `getFilterFullFileName()`.
pub fn get_filter_full_file_name() -> String {
    format!("{PROCESS_NAME}{}", dataset_files::COMSCRIPT_EXT)
}

/// Java static `getTestFileNameList(BaseManager, ParsedArray, ConstEtomoNumber,
/// String)`.  The test volume and the output volumes.
pub fn get_test_file_name_list_k(
    _manager: Option<&'static dyn BaseManager>,
    k_value_list: &ParsedArray,
    iteration: &ConstEtomoNumber,
    test_volume_name: Option<&str>,
) -> Vec<Option<String>> {
    let mut k_value = EtomoNumber::new_with_type(Some(Type::Double));
    let mut list = Vec::new();
    list.push(test_volume_name.map(str::to_owned));
    for i in 0..k_value_list.size() {
        k_value.set_string(k_value_list.get_raw_string_int(i).as_deref());
        list.push(get_test_file_name(&k_value, &iteration.to_string()));
    }
    list
}

/// Java static `getTestFileNameList(ConstEtomoNumber, IteratorElementList, String)`.
pub fn get_test_file_name_list_iteration(
    k_value: &ConstEtomoNumber,
    iteration_list: &IteratorElementList,
    test_volume_name: Option<&str>,
) -> Vec<Option<String>> {
    let mut list = Vec::new();
    list.push(test_volume_name.map(str::to_owned));
    for iteration in iteration_list.get_expanded_list() {
        list.push(get_test_file_name(
            k_value,
            iteration.as_deref().unwrap_or("null"),
        ));
    }
    list
}

/// Java private static `getTestFileName(ConstEtomoNumber, String)`.
fn get_test_file_name(k: &ConstEtomoNumber, iteration: &str) -> Option<String> {
    let iteration = converter::to_integer(Some(iteration));
    file_type::CLASS.test_nad.get_file_name_numeric(
        None,
        None,
        Some(&k.get_number().to_string()),
        iteration.map(|iteration| iteration.to_string()).as_deref(),
    )
    // return getTestFileRoot(k) + "-" + addLeadingZeros(iteration, 3);
}

/// Java private static `getTestFileRoot(ConstEtomoNumber)`.
fn get_test_file_root(k: &ConstEtomoNumber) -> String {
    format!("test.K{}", add_leading_zeros(&k.to_string(), 1))
}

/// Java private static `addLeadingZeros(String, int)`.  Add leading zeros to the
/// number up to maxZeros.  If the size of the number (or the size of the number left
/// of the decimal if the type is float or double) is greater or equal to maxZeros
/// then nothing is done.
fn add_leading_zeros(number: &str, max_zeros: usize) -> String {
    let mut digits = number;
    if let Some(decimal_place) = digits.find('.') {
        digits = &digits[..decimal_place];
    }
    let length = digits.chars().count();
    if length >= max_zeros {
        return number.to_owned();
    }
    let mut retval = String::new();
    for _ in 0..max_zeros - length {
        retval.push('0');
    }
    retval.push_str(number);
    retval
}

impl Command for AnisotropicDiffusionParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandLine()`.
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

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command = self.command.lock().unwrap();
        if command.is_empty() {
            self.build_command(&mut command);
        }
        Some(command.clone())
    }

    /// Java `getCommandInputFile()`: null.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandOutputFile()`: null.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java deprecated `getOutputImageFileType()`.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        if self.mode == Mode::Full {
            return Some(file_type::CLASS.anisotropic_diffusion_output.clone());
        }
        None
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if self.mode == Mode::VaryingK {
            return Some((*file_key::NAD_TEST_VARYING_K).clone());
        }
        if self.mode == Mode::VaryingIterations {
            return Some((*file_key::NAD_TEST_VARYING_ITERATIONS).clone());
        }
        if self.mode == Mode::Full {
            return Some(FileKey::clone(
                &file_type::CLASS.anisotropic_diffusion_output,
            ));
        }
        None
    }

    /// Java deprecated `getOutputImageFileType2()`: null.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2()`: null.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName()`: null.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails()`: null.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// The `CommandDetails` view: this class implements `ProcessDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for AnisotropicDiffusionParam {
    /// Java `getLogMessage()`, which returns null: nothing to log.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }

    /// Java `getName()`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }
}

/// The Java getters throw `IllegalArgumentException("field=" + field)` for a field
/// they do not know; that is `None` here.
impl ProcessDetails for AnisotropicDiffusionParam {
    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::KValueList) => self.k_value_list.get_raw_string_void(),
            Some(Field::IterationList) => Some(self.iteration_list.to_string()),
            _ => None,
        }
    }

    /// Java `getIteratorElementList(FieldInterface)`.
    fn get_iterator_element_list(&self, field: &dyn FieldInterface) -> Option<IteratorElementList> {
        self.get_iterator_element_list_field(field).cloned()
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64> {
        if field_interface::as_field::<Field>(field) == Some(&Field::KValue) {
            return Some(self.k_value.get_double());
        }
        None
    }

    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Field>(field) == Some(&Field::Iteration) {
            return Some(self.iteration.get_int());
        }
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_zeros_pad_the_integer_part() {
        assert_eq!(add_leading_zeros("1", 3), "001");
        assert_eq!(add_leading_zeros(".08", 1), "0.08");
        assert_eq!(add_leading_zeros("1234", 3), "1234");
        assert_eq!(get_filter_full_file_name(), "nad_eed_3d.com");
    }
}
