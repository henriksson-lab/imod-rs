//! `IMOD/Etomo/src/etomo/plugin/demo/EtomoPluginDemoParam.java`.
//!
//! Param for the parameters of etomoPluginDemo.  Runs from a com file with a different
//! name (`demo.com`).

use std::path::PathBuf;
use std::sync::Arc;

use super::demo_process_name;
use super::sleep_time::SleepTime;
use crate::imod::etomo::comscript::bad_com_script_exception::BadComScriptException;
use crate::imod::etomo::comscript::com_script_command::ComScriptCommand;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::command_mode::CommandMode;
use crate::imod::etomo::comscript::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::comscript::field_interface::{self, FieldInterface};
use crate::imod::etomo::comscript::process_details::{Hashtable, ProcessDetails};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::iterator_element_list::IteratorElementList;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java package-private static final `COMMAND_PROCESS_NAME =
/// DemoProcessName.ETOMO_PLUGIN_DEMO`.
pub fn command_process_name() -> ProcessName {
    *demo_process_name::ETOMO_PLUGIN_DEMO
}

/// Java private static final `SCRIPT_PROCESS_NAME = DemoProcessName.DEMO`.
fn script_process_name() -> ProcessName {
    *demo_process_name::DEMO
}

/// Java package-private static final `SLEEP_TIME_KEY`.
pub const SLEEP_TIME_KEY: &str = "SleepTime";
/// Java package-private static final `MESSAGE_KEY`.
pub const MESSAGE_KEY: &str = "Message";

/// Java `final class EtomoPluginDemoParam implements CommandParam, CommandDetails`.
#[derive(Clone)]
pub struct EtomoPluginDemoParam {
    /// Java private final `sleepTime = new ScriptParameter(SLEEP_TIME_KEY)`.
    sleep_time: ScriptParameter,
    /// Java private final `message = new StringParameter("Message")`.
    message: StringParameter,
    /// Java private final `axisID`.
    axis_id: AxisID,
}

impl EtomoPluginDemoParam {
    /// Java `EtomoPluginDemoParam(AxisID)`.
    pub fn new(axis_id: AxisID) -> EtomoPluginDemoParam {
        let mut sleep_time = ScriptParameter::new_with_name(SLEEP_TIME_KEY);
        // `SleepTime.DEFAULT.getValue().getInt()`: DEFAULT is ONE, whose value is 1.
        sleep_time.set_display_value_int(
            SleepTime::DEFAULT
                .value()
                .map(|value| value.get_int())
                .unwrap_or_default(),
        );
        EtomoPluginDemoParam {
            sleep_time,
            message: StringParameter::new("Message"),
            axis_id,
        }
    }

    /// Java package-private `setSleepTime(String)`.
    pub fn set_sleep_time_string(&mut self, input: Option<&str>) {
        self.sleep_time.set_string(input);
    }

    /// Java package-private `resetSleepTime()`.
    pub fn reset_sleep_time(&mut self) {
        self.sleep_time.reset();
    }

    /// Java package-private `setSleepTime(int)`.
    pub fn set_sleep_time_int(&mut self, mut input: i32) {
        if input < 1 {
            input = SleepTime::DEFAULT
                .value()
                .map(|value| value.get_int())
                .unwrap_or_default();
        }
        self.sleep_time.set_int(input);
    }

    /// Java package-private `isSleepTimeGt(int)`.
    pub fn is_sleep_time_gt(&self, input: i32) -> bool {
        self.sleep_time.gt_int(input)
    }

    /// Java package-private `setSleepTime(ConstEtomoNumber)`.
    pub fn set_sleep_time_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        self.sleep_time.set_const_etomo_number(input);
    }

    /// Java package-private `getSleepTime()`.
    pub fn get_sleep_time(&self) -> SleepTime {
        SleepTime::get_instance(Some(&self.sleep_time))
    }

    /// Java package-private `getSleepTimeValue()`.
    pub fn get_sleep_time_value(&self) -> i32 {
        self.sleep_time.get_int()
    }

    /// Java package-private `setMessage(String)`.
    pub fn set_message(&mut self, input: Option<&str>) {
        self.message.set(input);
    }

    /// Java package-private `resetMessage()`.
    pub fn reset_message(&mut self) {
        self.message.reset();
    }

    /// Java package-private `isMessageSet()`.
    pub fn is_message_set(&self) -> bool {
        !self.message.is_empty()
    }

    /// Java package-private `getMessage()`.
    pub fn get_message(&self) -> String {
        self.message.to_string()
    }
}

//
// Implement CommandParam
//

impl CommandParam for EtomoPluginDemoParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.  Initialize the parameter object
    /// from the ComScriptCommand object.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // reset
        self.initialize_defaults();
        // parse
        self.sleep_time.parse(script_command)?;
        self.message.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.  Replace the parameters of the
    /// ComScriptCommand with the current CommandParameter object's parameters.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.sleep_time.update_com_script(script_command);
        self.message.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {
        self.sleep_time.reset();
        self.message.reset();
    }
}

//
// Implementing Command
//

impl Command for EtomoPluginDemoParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandMode()`.  For params that can be run for different purposes with
    /// different parameters, or run from different comscripts.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getProcessName()`.  For a comscript, the process name of the comscript.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(script_process_name())
    }

    /// Java `getCommand()`.  For a comscript, the name of the comscript file
    /// (`ProcessName.getComscript(axisID)`).
    fn get_command(&self) -> Option<String> {
        Some(script_process_name().get_comscript(self.axis_id))
    }

    /// Java `getCommandName()`.  For a comscript, the name of the comscript
    /// (`ProcessName.toString`).
    fn get_command_name(&self) -> Option<String> {
        Some(script_process_name().to_string())
    }

    /// Java `getCommandLine()`.  For a comscript, the name of the comscript.
    fn get_command_line(&self) -> Option<String> {
        Some(script_process_name().get_comscript(self.axis_id))
    }

    /// Java `getCommandArray()`.  For a comscript, the comscript file (without a path).
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(vec![script_process_name().get_comscript(self.axis_id)])
    }

    /// Java `getCommandInputFile()`: the input file parameter if available.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandOutputFile()`: the output file parameter if available.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter()`.  Return yes if a message from the process needs to be
    /// popped up while the process is running, rather then reported after the process
    /// is finished.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName()`.  For a command that runs another command
    /// (like processchunks).
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails()`.  For a command that runs another command (like
    /// processchunks).
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

//
// Implement Loggable
//

impl Loggable for EtomoPluginDemoParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        script_process_name().to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

//
// Implement ProcessDetails
//
// Every getter but `getIntValue(Field.SLEEP_TIME)` throws
// `IllegalArgumentException("field=" + field)`, an uncaught exception for any caller.
// Fixed in translation (as in the other params): the value is unavailable (`None`).

impl ProcessDetails for EtomoPluginDemoParam {
    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Field>(field) == Some(&Field::SleepTime) {
            return Some(self.sleep_time.get_int());
        }
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
    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Hashtable> {
        None
    }

    /// Java `getEtomoNumber(FieldInterface)`.
    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    /// Java `getIntKeyList(FieldInterface)`.
    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<IntKeyList> {
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
    ) -> Option<IteratorElementList> {
        None
    }
}

/// Java `static final class Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `SLEEP_TIME = new Field("SLEEP_TIME")`.
    SleepTime,
}

impl FieldInterface for Field {}
