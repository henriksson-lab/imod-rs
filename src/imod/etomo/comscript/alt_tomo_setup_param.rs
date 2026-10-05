//! `IMOD/Etomo/src/etomo/comscript/AltTomoSetupParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::ALT_TOMO_SETUP;
/// Java `EVEN_AND_ODD_PAIRS`.
pub const EVEN_AND_ODD_PAIRS: &str = "EvenAndOddPairs";
/// Java `ROOTNAME_TO_PROCESS`.
pub const ROOTNAME_TO_PROCESS: &str = "RootnameToProcess";
/// Java `AXIS_TO_PROCESS`.
pub const AXIS_TO_PROCESS: &str = "AxisToProcess";
/// Java `PREPROCESS_FOR_EXTREMES`.
pub const PREPROCESS_FOR_EXTREMES: &str = "PreprocessForExtremes";
/// Java `CORRECT_CTF`.
pub const CORRECT_CTF: &str = "CorrectCTF";
/// Java `ERASE_FIDUCIALS`.
pub const ERASE_FIDUCIALS: &str = "EraseFiducials";
/// Java `FILTER_IN_2D`.
pub const FILTER_IN_2D: &str = "FilterIn2D";
/// Java `TRIM_VOLUME`.
pub const TRIM_VOLUME: &str = "TrimVolume";
/// Java `CLEAN_UP_INTERMEDIATES`.
pub const CLEAN_UP_INTERMEDIATES: &str = "CleanUpIntermediates";
/// Java `NUMBER_OF_PROCESSORS`.
pub const NUMBER_OF_PROCESSORS: &str = "NumberOfProcessors";
/// Java `JUST_RESTORE_INITIAL_SET`.
pub const JUST_RESTORE_INITIAL_SET: &str = "JustRestoreInitialSet";

/// Java nested class `AltTomoSetupParam.Field`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `Field.ROOTNAME_TO_PROCESS`.
    RootnameToProcess,
    /// Java `Field.EVEN_AND_ODD_PAIRS`.
    EvenAndOddPairs,
    /// Java `Field.AXIS_TO_PROCESS`.
    AxisToProcess,
    /// Java `Field.PREPROCESS_FOR_EXTREMES`.
    PreprocessForExtremes,
    /// Java `Field.CORRECT_CTF`.
    CorrectCtf,
    /// Java `Field.ERASE_FIDUCIALS`.
    EraseFiducials,
    /// Java `Field.FILTER_IN_2D`.
    FilterIn2d,
    /// Java `Field.TRIM_VOLUME`.
    TrimVolume,
}

impl FieldInterface for Field {}

/// Java nested class `AltTomoSetupParam.Mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `Mode.JUST_RESTORE_INITIAL_SET`.
    JustRestoreInitialSet,
    /// Java `Mode.ALT_TOMO_SETUP`.
    AltTomoSetup,
}

impl Mode {
    /// Java private static `Mode.DEFAULT`.
    const DEFAULT: Mode = Mode::AltTomoSetup;
}

/// Java `Mode.toString`.
impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::JustRestoreInitialSet => "JUST_RESTORE_INITIAL_SET",
            Mode::AltTomoSetup => "ALT_TOMO_SETUP",
        })
    }
}

impl CommandMode for Mode {}

/// Java final `AltTomoSetupParam`.
pub struct AltTomoSetupParam {
    even_and_odd_pairs: EtomoBoolean2,
    rootname_to_process: StringParameter,
    axis_to_process: StringParameter,
    preprocess_for_extremes: ScriptParameter,
    correct_ctf: EtomoBoolean2,
    erase_fiducials: EtomoBoolean2,
    filter_in_2d: EtomoBoolean2,
    trim_volume: EtomoBoolean2,
    clean_up_intermediates: EtomoBoolean2,
    number_of_processors: ScriptParameter,
    just_restore_initial_set: EtomoBoolean2,
    command_mode: Mode,
    /// Java `manager`, which the source never reads.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl AltTomoSetupParam {
    /// Java `AltTomoSetupParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> AltTomoSetupParam {
        AltTomoSetupParam {
            even_and_odd_pairs: EtomoBoolean2::new_with_name(EVEN_AND_ODD_PAIRS),
            rootname_to_process: StringParameter::new(ROOTNAME_TO_PROCESS),
            axis_to_process: StringParameter::new(AXIS_TO_PROCESS),
            preprocess_for_extremes: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                PREPROCESS_FOR_EXTREMES,
            ),
            correct_ctf: EtomoBoolean2::new_with_name(CORRECT_CTF),
            erase_fiducials: EtomoBoolean2::new_with_name(ERASE_FIDUCIALS),
            filter_in_2d: EtomoBoolean2::new_with_name(FILTER_IN_2D),
            trim_volume: EtomoBoolean2::new_with_name(TRIM_VOLUME),
            clean_up_intermediates: EtomoBoolean2::new_with_name(CLEAN_UP_INTERMEDIATES),
            number_of_processors: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                NUMBER_OF_PROCESSORS,
            ),
            just_restore_initial_set: EtomoBoolean2::new_with_name(JUST_RESTORE_INITIAL_SET),
            command_mode: Mode::DEFAULT,
            manager,
            axis_id,
        }
    }

    /// Java `resetAxisToProcess`.
    pub fn reset_axis_to_process(&mut self) {
        self.axis_to_process.reset();
    }

    /// Java `isEvenAndOddPairs`.
    pub fn is_even_and_odd_pairs(&self) -> bool {
        self.even_and_odd_pairs.is()
    }

    /// Java `isRootnameToProcess`.
    pub fn is_rootname_to_process(&self) -> bool {
        !self.rootname_to_process.is_empty()
    }

    /// Java `getRootnameToProcess`.
    pub fn get_rootname_to_process(&self) -> String {
        self.rootname_to_process.to_string()
    }

    /// Java `isAxisToProcess`.
    pub fn is_axis_to_process(&self) -> bool {
        !self.axis_to_process.is_empty()
    }

    /// Java `getAxisToProcess`.
    pub fn get_axis_to_process(&self) -> String {
        self.axis_to_process.to_string()
    }

    /// Java `isPreprocessForExtremes`.
    pub fn is_preprocess_for_extremes(&self) -> bool {
        self.preprocess_for_extremes.is()
    }

    /// Java `getPreprocessForExtremes`.
    pub fn get_preprocess_for_extremes(&self) -> i32 {
        self.preprocess_for_extremes.get_int()
    }

    /// Java `isCorrectCTF`.
    pub fn is_correct_ctf(&self) -> bool {
        self.correct_ctf.is()
    }

    /// Java `isEraseFiducials`.
    pub fn is_erase_fiducials(&self) -> bool {
        self.erase_fiducials.is()
    }

    /// Java `isFilterIn2D`.
    pub fn is_filter_in_2d(&self) -> bool {
        self.filter_in_2d.is()
    }

    /// Java `isTrimVolume`.
    pub fn is_trim_volume(&self) -> bool {
        self.trim_volume.is()
    }

    /// Java `isCleanUpIntermediates`.
    pub fn is_clean_up_intermediates(&self) -> bool {
        self.clean_up_intermediates.is()
    }

    /// Java `isNumberOfProcessors`.
    pub fn is_number_of_processors(&self) -> bool {
        self.number_of_processors.is()
    }

    /// Java `getNumberOfProcessors`.
    pub fn get_number_of_processors(&self) -> i32 {
        self.number_of_processors.get_int()
    }

    /// Java `isJustRestoreInitialSet`.
    pub fn is_just_restore_initial_set(&self) -> bool {
        self.just_restore_initial_set.is()
    }

    /// Java `resetEvenAndOddPairs`.
    pub fn reset_even_and_odd_pairs(&mut self) {
        self.even_and_odd_pairs.reset();
    }

    /// Java `setEvenAndOddPairs`.
    pub fn set_even_and_odd_pairs(&mut self, input: bool) {
        self.even_and_odd_pairs.set_boolean(input);
    }

    /// Java `resetRootnameToProcess`.
    pub fn reset_rootname_to_process(&mut self) {
        self.rootname_to_process.reset();
    }

    /// Java `setRootnameToProcess`.
    pub fn set_rootname_to_process(&mut self, input: Option<&str>) {
        self.rootname_to_process.set(input);
    }

    /// Java `setAxisToProcess`.
    pub fn set_axis_to_process(&mut self, input: Option<&str>) {
        self.axis_to_process.set(input);
    }

    /// Java `setPreprocessForExtremes(int)`.
    pub fn set_preprocess_for_extremes(&mut self, input: i32) {
        self.preprocess_for_extremes.set_int(input);
    }

    /// Java `setCorrectCTF`.
    pub fn set_correct_ctf(&mut self, input: bool) {
        self.correct_ctf.set_boolean(input);
    }

    /// Java `setEraseFiducials`.
    pub fn set_erase_fiducials(&mut self, input: bool) {
        self.erase_fiducials.set_boolean(input);
    }

    /// Java `setFilterIn2D`.
    pub fn set_filter_in_2d(&mut self, input: bool) {
        self.filter_in_2d.set_boolean(input);
    }

    /// Java `setTrimVolume`.
    pub fn set_trim_volume(&mut self, input: bool) {
        self.trim_volume.set_boolean(input);
    }

    /// Java `setCleanUpIntermediates`.
    pub fn set_clean_up_intermediates(&mut self, input: bool) {
        self.clean_up_intermediates.set_boolean(input);
    }

    /// Java `setNumberOfProcessors(int)`.
    pub fn set_number_of_processors(&mut self, input: i32) {
        self.number_of_processors.set_int(input);
    }

    /// Java `setJustRestoreInitialSet`.
    pub fn set_just_restore_initial_set(&mut self, input: bool) {
        self.just_restore_initial_set.set_boolean(input);
    }

    /// Java `resetJustRestoreInitialSet`.
    pub fn reset_just_restore_initial_set(&mut self) {
        self.just_restore_initial_set.reset();
    }

    /// Java `setCommandMode`.
    pub fn set_command_mode(&mut self, input: Mode) {
        self.command_mode = input;
    }
}

impl Command for AltTomoSetupParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.command_mode)
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::ALT_TOMO_SETUP)
    }

    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(AxisID::Only))
    }

    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::ALT_TOMO_SETUP.to_string())
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(PROCESS_NAME.get_comscript_array(AxisID::Only))
    }

    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}

impl CommandParam for AltTomoSetupParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Java calls `scriptCommand.useKeywordValue()` on the command it parses,
        // converting an old-style command in place.  The trait lends the command
        // immutably, so the conversion is made on a copy, which is what is parsed.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        // public void setCommandMode(Mode input) {
        // commandMode = input;
        // }
        self.even_and_odd_pairs.parse(script_command)?;
        self.rootname_to_process.parse(script_command)?;
        self.axis_to_process.parse(script_command)?;
        self.preprocess_for_extremes.parse(script_command)?;
        self.correct_ctf.parse(script_command)?;
        self.erase_fiducials.parse(script_command)?;
        self.filter_in_2d.parse(script_command)?;
        self.trim_volume.parse(script_command)?;
        self.clean_up_intermediates.parse(script_command)?;
        self.number_of_processors.parse(script_command)?;
        self.just_restore_initial_set.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.even_and_odd_pairs.update_com_script(script_command);
        self.rootname_to_process.update_com_script(script_command);
        self.axis_to_process.update_com_script(script_command);
        self.preprocess_for_extremes
            .update_com_script(script_command);
        self.correct_ctf.update_com_script(script_command);
        self.erase_fiducials.update_com_script(script_command);
        self.filter_in_2d.update_com_script(script_command);
        self.trim_volume.update_com_script(script_command);
        self.clean_up_intermediates
            .update_com_script(script_command);
        self.number_of_processors.update_com_script(script_command);
        self.just_restore_initial_set
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.even_and_odd_pairs.reset();
        self.rootname_to_process.reset();
        self.axis_to_process.reset();
        self.preprocess_for_extremes.reset();
        self.correct_ctf.reset();
        self.erase_fiducials.reset();
        self.filter_in_2d.reset();
        self.trim_volume.reset();
        self.clean_up_intermediates.reset();
        self.number_of_processors.reset();
        self.just_restore_initial_set.reset();
    }
}

impl Loggable for AltTomoSetupParam {
    /// Java `getName`, which returns null; the trait's `String` makes that empty.
    fn get_name(&self) -> String {
        String::new()
    }

    /// Java `getLogMessage`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

impl ProcessDetails for AltTomoSetupParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        Some(0)
    }

    // <p>Updates done</p>

    /// A field Java does not recognise throws `IllegalArgumentException`; here that is
    /// `None`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::EvenAndOddPairs {
            return Some(self.is_even_and_odd_pairs());
        }
        if *field == Field::PreprocessForExtremes {
            return Some(self.is_preprocess_for_extremes());
        }
        if *field == Field::CorrectCtf {
            return Some(self.is_correct_ctf());
        }
        if *field == Field::EraseFiducials {
            return Some(self.is_erase_fiducials());
        }
        if *field == Field::FilterIn2d {
            return Some(self.is_filter_in_2d());
        }
        if *field == Field::TrimVolume {
            return Some(self.is_trim_volume());
        }
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        Some(0.0)
    }

    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
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

    /// A field Java does not recognise throws `IllegalArgumentException`; here that is
    /// `None`.
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::RootnameToProcess {
            return Some(self.rootname_to_process.to_string());
        }
        if *field == Field::AxisToProcess {
            return Some(self.axis_to_process.to_string());
        }
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_strings_are_the_java_to_string() {
        assert_eq!(Mode::DEFAULT, Mode::AltTomoSetup);
        assert_eq!(Mode::AltTomoSetup.to_string(), "ALT_TOMO_SETUP");
        assert_eq!(
            Mode::JustRestoreInitialSet.to_string(),
            "JUST_RESTORE_INITIAL_SET"
        );
    }
}
