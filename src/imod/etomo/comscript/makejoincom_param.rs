//! `IMOD/Etomo/src/etomo/comscript/MakejoincomParam.java`.
//!
//! The `makejoincom` command line (`python -u <scripts>makejoincom ...`), built from
//! the join meta data's section table when the param is constructed, together with
//! the rotation state (`ROTATE`, `TOTAL_ROWS`, `ROTATION_ANGLES_LIST`) that
//! `JoinManager.postProcess` hands on to `StartJoinParam`.

use std::path::PathBuf;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::{Hashtable, ProcessDetails};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_to_string,
};
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::join_state::JoinState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::slicer_angles::SlicerAngles;
use crate::imod::etomo::util::utilities;

/// Java `public static final int MIDAS_LIMIT_DEFAULT`.
pub const MIDAS_LIMIT_DEFAULT: i32 = 1024;

/// Java private static final `commandSize`.
const COMMAND_SIZE: usize = 3;
/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::MAKEJOINCOM;
/// Java `public static final String commandName`.
pub const COMMAND_NAME: &str = "makejoincom";
/// Java private static final `debug`.
const DEBUG: bool = false;

/// Java nested `Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fields {
    /// Java `ROTATE`.
    Rotate,
    /// Java `ROTATION_ANGLES_LIST`.
    RotationAnglesList,
    /// Java `TOTAL_ROWS`.
    TotalRows,
}

impl FieldInterface for Fields {}

/// Java `public final class MakejoincomParam implements CommandDetails`.
pub struct MakejoincomParam {
    /// Java private final `commandArray`.
    command_array: Vec<String>,
    /// Java private `rotationAnglesList`, initially null.
    rotation_angles_list: Option<Hashtable>,
    /// Java private `rotate`, initially false.
    rotate: bool,
    /// Java private `totalRows`, initially 0.
    total_rows: i32,
}

impl MakejoincomParam {
    /// Java `MakejoincomParam(ConstJoinMetaData, JoinState, BaseManager)`.
    pub fn new(
        meta_data: &dyn ConstJoinMetaData,
        state: &JoinState,
        manager: &'static dyn BaseManager,
    ) -> MakejoincomParam {
        let mut param = MakejoincomParam {
            command_array: Vec::new(),
            rotation_angles_list: None,
            rotate: false,
            total_rows: 0,
        };
        let options = param.gen_options(meta_data, state, manager);
        let mut command_array = vec![String::new(); options.len() + COMMAND_SIZE];
        command_array[0] = "python".to_string();
        command_array[1] = "-u".to_string();
        command_array[2] = format!(
            "{}{}",
            etomo_director::INSTANCE
                .get_python_script_path()
                .unwrap_or_else(|| "null".to_string()),
            COMMAND_NAME
        );
        for i in 0..options.len() {
            command_array[i + COMMAND_SIZE] = options[i].clone();
        }
        if DEBUG {
            let mut buffer = String::new();
            for i in 0..command_array.len() {
                buffer.push_str(&command_array[i]);
                if i < command_array.len() - 1 {
                    buffer.push(' ');
                }
            }
            eprintln!("{buffer}");
        }
        param.command_array = command_array;
        param
    }

    /// Java private `genOptions()`.
    fn gen_options(
        &mut self,
        meta_data: &dyn ConstJoinMetaData,
        state: &JoinState,
        manager: &'static dyn BaseManager,
    ) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        let section_data = meta_data.get_section_table_data();
        if let Some(section_data) = section_data {
            self.total_rows = section_data.len() as i32;
            for i in 0..self.total_rows {
                let screen: &dyn ConstSectionTableRowData = &*section_data[i as usize];
                let section = screen
                    .get_setup_section()
                    .map(|section| section.to_path_buf());
                if i < self.total_rows - 1 {
                    options.push("-top".to_string());
                    // both numbers must exist
                    options.push(format!(
                        "{},{}",
                        screen.get_sample_top_start(),
                        screen.get_sample_top_end()
                    ));
                }
                if i != 0 {
                    options.push("-bot".to_string());
                    // both numbers must exist
                    options.push(format!(
                        "{},{}",
                        screen.get_sample_bottom_start(),
                        screen.get_sample_bottom_end()
                    ));
                }
                // Is the volume rotated?
                if screen.is_rotated() {
                    self.rotate = true;
                    // Get the rotation angles from the screen.  Use 0 in place of null.
                    let rotation_angle_x: &ConstEtomoNumber = screen.get_rotation_angle_x();
                    let rotation_angle_y: &ConstEtomoNumber = screen.get_rotation_angle_y();
                    let rotation_angle_z: &ConstEtomoNumber = screen.get_rotation_angle_z();
                    // Save the angles, so they can be saved in the state object.
                    let mut rotation_angles = SlicerAngles::new();
                    rotation_angles.set_x(Some(rotation_angle_x));
                    rotation_angles.set_y(Some(rotation_angle_y));
                    rotation_angles.set_z(Some(rotation_angle_z));
                    if self.rotation_angles_list.is_none() {
                        self.rotation_angles_list = Some(Hashtable::new());
                    }
                    let hash_key = i;
                    self.rotation_angles_list
                        .as_mut()
                        .unwrap()
                        .insert(hash_key, rotation_angles);
                    // The rotation file does not exist or the angles have changed.
                    // Add the -rot option to run rotatevol and save the angles.
                    options.push("-rot".to_string());
                    let mut buffer = String::new();
                    buffer.push_str(&java_lang_double_to_string(
                        rotation_angle_x.get_defaulted_double(),
                    ));
                    buffer.push(',');
                    buffer.push_str(&java_lang_double_to_string(
                        rotation_angle_y.get_defaulted_double(),
                    ));
                    buffer.push(',');
                    buffer.push_str(&java_lang_double_to_string(
                        rotation_angle_z.get_defaulted_double(),
                    ));
                    options.push(buffer);
                    options.push("-maxxysize".to_string());
                    // Get the .rot file
                    let section_name = section.as_ref().map(|section| {
                        utilities::java_io_file_get_name(&section.to_string_lossy())
                    });
                    let rot_file_name = dataset_tool::substitute_extension(
                        section_name.as_deref(),
                        Some(&extension::CLASS.rot),
                    );
                    // `new File(manager.getPropertyUserDir(), rotFileName)`
                    let rot_file = PathBuf::from(utilities::java_io_file_new(
                        &manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_string()),
                        rot_file_name.as_deref().unwrap_or("null"),
                    ));
                    // See if the rotation file exists and has the same angles
                    let cur_rotation_angles = state.get_rotation_angles(hash_key);
                    if rot_file.exists()
                        && cur_rotation_angles
                            .as_ref()
                            .is_some_and(|cur_rotation_angles| {
                                rotation_angle_x
                                    .equals_const_etomo_number(Some(cur_rotation_angles.get_x()))
                                    && rotation_angle_y.equals_const_etomo_number(Some(
                                        cur_rotation_angles.get_y(),
                                    ))
                                    && rotation_angle_z.equals_const_etomo_number(Some(
                                        cur_rotation_angles.get_z(),
                                    ))
                            })
                    {
                        // Use the existing .rot file, since the angles haven't changed.
                        options.push("-already".to_string());
                    }
                }
                // Fixed in translation: a row without a setup section makes the source
                // throw NullPointerException (`section.getAbsolutePath()`); "null" is
                // added here.
                options.push(
                    section
                        .as_ref()
                        .map(|section| {
                            utilities::java_io_file_get_absolute_path(&section.to_string_lossy())
                        })
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
        }
        options.push("-tmpext".to_string());
        options.push(extension::CLASS.rot.to_string());
        let density_ref_section = meta_data.get_density_ref_section_parameter();
        if density_ref_section.is_not_null_and_not_default() {
            options.push("-ref".to_string());
            options.push(density_ref_section.to_string());
        }
        // Java's unused local `number`.
        let _number = meta_data.get_midas_limit();
        options.push("-midaslim".to_string());
        options.push(meta_data.get_midas_limit().to_string());
        options.push(meta_data.get_name());
        options.push("-NamingStyle ".to_string());
        options.push(meta_data.get_image_filename_style().to_string());
        options
    }
}

impl Command for MakejoincomParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.command_array.clone())
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        let mut buffer = String::new();
        for element in &self.command_array {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
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

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for MakejoincomParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        COMMAND_NAME.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Every getter the source does not answer throws `IllegalArgumentException("field="
/// + field)`, an uncaught exception for any caller.  Fixed in translation: the value
/// is unavailable (`None`).
impl ProcessDetails for MakejoincomParam {
    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::TotalRows) {
            return Some(self.total_rows);
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

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(&self, field: &dyn FieldInterface) -> Option<Hashtable> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::RotationAnglesList) {
            return self.rotation_angles_list.clone();
        }
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
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}
