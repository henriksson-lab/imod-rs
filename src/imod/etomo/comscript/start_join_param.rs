//! `IMOD/Etomo/src/etomo/comscript/StartJoinParam.java`.
//!
//! The `startjoin.com` run.  It holds the rotation state makejoincom reported (handed
//! over by `JoinManager.postProcess`) so that `JoinProcessManager.postProcess` can
//! save it in the `JoinState` when startjoin succeeds.

use std::path::PathBuf;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::{Hashtable, ProcessDetails};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::STARTJOIN;
/// Java private static final `debug`.
const DEBUG: bool = false;

/// Java nested `Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fields {
    /// Java `ROTATION_ANGLES_LIST`.
    RotationAnglesList,
    /// Java `TOTAL_ROWS`.
    TotalRows,
    /// Java `ROTATE`.
    Rotate,
}

impl FieldInterface for Fields {}

/// Java `public final class StartJoinParam implements CommandDetails`.
pub struct StartJoinParam {
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private `rotationAnglesList`, initially null.
    rotation_angles_list: Option<Hashtable>,
    /// Java private `rotate`, initially false.
    rotate: bool,
    /// Java private `totalRows`, initially 0.
    total_rows: i32,
}

impl StartJoinParam {
    /// Java `StartJoinParam(AxisID)`.
    pub fn new(axis_id: AxisID) -> StartJoinParam {
        StartJoinParam {
            axis_id,
            rotation_angles_list: None,
            rotate: false,
            total_rows: 0,
        }
    }

    /// Java `setRotate(boolean)`.
    pub fn set_rotate(&mut self, rotate: bool) {
        self.rotate = rotate;
    }

    /// Java `setRotationAnglesList(Hashtable)`.
    pub fn set_rotation_angles_list(&mut self, rotation_angles_list: Option<Hashtable>) {
        self.rotation_angles_list = rotation_angles_list;
    }

    /// Java `setTotalRows(int)`.
    pub fn set_total_rows(&mut self, total_rows: i32) {
        self.total_rows = total_rows;
    }
}

impl Command for StartJoinParam {
    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        let command = PROCESS_NAME.get_comscript(self.axis_id);
        if DEBUG {
            eprintln!("{command}");
        }
        Some(command)
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        None
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        None
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        Some(std::sync::Arc::clone(&file_type::CLASS.join_sample))
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.join_sample))
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        Some(std::sync::Arc::clone(
            &file_type::CLASS.join_sample_averages,
        ))
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.join_sample_averages))
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for StartJoinParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Every getter the source does not answer throws `IllegalArgumentException("field="
/// + field)`.  Fixed in translation: the value is unavailable (`None`).
impl ProcessDetails for StartJoinParam {
    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(&self, field: &dyn FieldInterface) -> Option<Hashtable> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::RotationAnglesList) {
            return self.rotation_angles_list.clone();
        }
        None
    }

    /// Java `getBooleanValue(FieldInterface)`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::Rotate) {
            return Some(self.rotate);
        }
        None
    }

    /// Java `getStringArray(FieldInterface)`.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
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

    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::TotalRows) {
            return Some(self.total_rows);
        }
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
