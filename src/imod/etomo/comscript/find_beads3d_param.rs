//! `IMOD/Etomo/src/etomo/comscript/FindBeads3dParam.java`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_find_beads3d_param::ConstFindBeads3dParam;
use super::field_interface::FieldInterface;
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::util::utilities;

/// Java `BEAD_SIZE_TAG`.
pub const BEAD_SIZE_TAG: &str = "BeadSize";
/// Java `LIGHT_BEADS_TAG`.
pub const LIGHT_BEADS_TAG: &str = "LightBeads";
/// Java `MIN_RELATIVE_STRENGTH_TAG`.
pub const MIN_RELATIVE_STRENGTH_TAG: &str = "MinRelativeStrength";
/// Java `THRESHOLD_FOR_AVERAGING_TAG`.
pub const THRESHOLD_FOR_AVERAGING_TAG: &str = "ThresholdForAveraging";
/// Java `STORAGE_THRESHOLD_TAG`.
pub const STORAGE_THRESHOLD_TAG: &str = "StorageThreshold";
/// Java `MIN_SPACING_TAG`.
pub const MIN_SPACING_TAG: &str = "MinSpacing";
/// Java `GUESS_NUM_BEADS_TAG`.
pub const GUESS_NUM_BEADS_TAG: &str = "GuessNumBeads";
/// Java `MAX_NUM_BEADS_TAG`.
pub const MAX_NUM_BEADS_TAG: &str = "MaxNumBeads";
/// Java `Y_AXIS_ELONGATED_TAG`.
pub const Y_AXIS_ELONGATED_TAG: &str = "YAxisElongated";

/// Java final `FindBeads3dParam implements ConstFindBeads3dParam, CommandParam`.
pub struct FindBeads3dParam {
    input_file: StringParameter,
    output_file: StringParameter,
    bead_size: ScriptParameter,
    light_beads: EtomoBoolean2,
    min_relative_strength: ScriptParameter,
    threshold_for_averaging: ScriptParameter,
    storage_threshold: ScriptParameter,
    min_spacing: ScriptParameter,
    guess_num_beads: ScriptParameter,
    max_num_beads: ScriptParameter,
    binning_of_volume: ScriptParameter,
    y_axis_elongated: EtomoBoolean2,
    /// Written by `updateComScriptCommand`, which the `CommandParam` trait gives only
    /// `&self`.
    tilt_file: Mutex<StringParameter>,

    axis_id: AxisID,
    manager: &'static dyn BaseManager,
}

impl FindBeads3dParam {
    /// Java `FindBeads3dParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> FindBeads3dParam {
        let mut param = FindBeads3dParam {
            input_file: StringParameter::new("InputFile"),
            output_file: StringParameter::new("OutputFile"),
            bead_size: ScriptParameter::new_with_type_and_name(Type::Double, BEAD_SIZE_TAG),
            light_beads: EtomoBoolean2::new_with_name(LIGHT_BEADS_TAG),
            min_relative_strength: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MIN_RELATIVE_STRENGTH_TAG,
            ),
            threshold_for_averaging: ScriptParameter::new_with_type_and_name(
                Type::Double,
                THRESHOLD_FOR_AVERAGING_TAG,
            ),
            storage_threshold: ScriptParameter::new_with_type_and_name(
                Type::Double,
                STORAGE_THRESHOLD_TAG,
            ),
            min_spacing: ScriptParameter::new_with_type_and_name(Type::Double, MIN_SPACING_TAG),
            guess_num_beads: ScriptParameter::new_with_name(GUESS_NUM_BEADS_TAG),
            max_num_beads: ScriptParameter::new_with_name(MAX_NUM_BEADS_TAG),
            binning_of_volume: ScriptParameter::new_with_name("BinningOfVolume"),
            y_axis_elongated: EtomoBoolean2::new_with_name(Y_AXIS_ELONGATED_TAG),
            tilt_file: Mutex::new(StringParameter::new("TiltFile")),
            axis_id,
            manager,
        };
        param.y_axis_elongated.set_display_value_boolean(true);
        param.reset();
        param
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.input_file.reset();
        self.output_file.reset();
        self.bead_size.reset();
        self.light_beads.reset();
        self.min_relative_strength.reset();
        self.threshold_for_averaging.reset();
        self.storage_threshold.reset();
        self.min_spacing.reset();
        self.guess_num_beads.reset();
        self.max_num_beads.reset();
        self.binning_of_volume.reset();
        self.y_axis_elongated.reset();
        self.tilt_file.get_mut().unwrap().reset();
    }

    /// Java `setInputFile(FileType)`.
    pub fn set_input_file(&mut self, input: &Arc<FileType>) {
        self.input_file.set(
            input
                .get_file_name(Some(self.manager), Some(self.axis_id))
                .as_deref(),
        );
        self.binning_of_volume
            .set_int(utilities::get_stack_binning_for_file_type(
                self.manager,
                self.axis_id,
                input,
            ));
    }

    // Updates done

    /// Java `setOutputFile(String)`.
    pub fn set_output_file(&mut self, input: Option<&str>) {
        self.output_file.set(input);
    }

    /// Java `setBeadSize(String)`.
    pub fn set_bead_size(&mut self, input: Option<&str>) {
        self.bead_size.set_string(input);
    }

    /// Java `setLightBeads(boolean)`.
    pub fn set_light_beads(&mut self, input: bool) {
        self.light_beads.set_boolean(input);
    }

    /// Java `setMinRelativeStrength(String)`.
    pub fn set_min_relative_strength(&mut self, input: Option<&str>) {
        self.min_relative_strength.set_string(input);
    }

    /// Java `setThresholdForAveraging(String)`.
    pub fn set_threshold_for_averaging(&mut self, input: Option<&str>) {
        self.threshold_for_averaging.set_string(input);
    }

    /// Java `setStorageThreshold(ConstEtomoNumber)`.
    pub fn set_storage_threshold_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        self.storage_threshold.set_const_etomo_number(input);
    }

    /// Java `setStorageThreshold(String)`.
    pub fn set_storage_threshold_string(&mut self, input: Option<&str>) {
        self.storage_threshold.set_string(input);
    }

    /// Java `setMinSpacing(String)`.
    pub fn set_min_spacing(&mut self, input: Option<&str>) {
        self.min_spacing.set_string(input);
    }

    /// Java `setGuessNumBeads(String)`.
    pub fn set_guess_num_beads(&mut self, input: Option<&str>) {
        self.guess_num_beads.set_string(input);
    }

    /// Java `setMaxNumBeads(String)`.
    pub fn set_max_num_beads(&mut self, input: Option<&str>) {
        self.max_num_beads.set_string(input);
    }
}

impl CommandParam for FindBeads3dParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        self.bead_size.parse(script_command)?;
        self.min_relative_strength.parse(script_command)?;
        self.threshold_for_averaging.parse(script_command)?;
        self.storage_threshold.parse(script_command)?;
        self.min_spacing.parse(script_command)?;
        self.guess_num_beads.parse(script_command)?;
        self.max_num_beads.parse(script_command)?;
        // binningOfVolume is not displayed and it is always derived from the
        // inputFile.
        self.y_axis_elongated.parse(script_command)?;
        self.tilt_file.get_mut().unwrap().parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.input_file.update_com_script(script_command);
        self.output_file.update_com_script(script_command);
        self.bead_size.update_com_script(script_command);
        self.light_beads.update_com_script(script_command);
        self.min_relative_strength.update_com_script(script_command);
        self.threshold_for_averaging
            .update_com_script(script_command);
        self.storage_threshold.update_com_script(script_command);
        self.min_spacing.update_com_script(script_command);
        self.guess_num_beads.update_com_script(script_command);
        self.max_num_beads.update_com_script(script_command);
        self.binning_of_volume.update_com_script(script_command);
        self.y_axis_elongated.update_com_script(script_command);
        // tiltFile
        let mut angle_range_set = false;
        if let Ok(has_keyword) = script_command.has_keyword(Some("AngleRange")) {
            angle_range_set = has_keyword;
        }
        let mut tilt_file = self.tilt_file.lock().unwrap();
        if !angle_range_set {
            tilt_file.set(
                file_type::CLASS
                    .tilt_angles
                    .get_file_name(Some(self.manager), Some(self.axis_id))
                    .as_deref(),
            );
            tilt_file.update_com_script(script_command);
        } else {
            tilt_file.delete_from_com_script(script_command);
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl ConstFindBeads3dParam for FindBeads3dParam {
    fn get_bead_size(&self) -> String {
        self.bead_size.to_string()
    }

    fn get_min_relative_strength(&self) -> String {
        self.min_relative_strength.to_string()
    }

    fn get_threshold_for_averaging(&self) -> String {
        self.threshold_for_averaging.to_string()
    }

    fn get_storage_threshold(&self) -> &ConstEtomoNumber {
        &self.storage_threshold
    }

    fn get_min_spacing(&self) -> String {
        self.min_spacing.to_string()
    }

    fn get_guess_num_beads(&self) -> String {
        self.guess_num_beads.to_string()
    }

    fn get_max_num_beads(&self) -> String {
        self.max_num_beads.to_string()
    }
}

impl Loggable for FindBeads3dParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        ProcessName::FIND_BEADS_3D.to_string()
    }

    /// Java `getLogMessage`, which returns null: no log message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

impl Command for FindBeads3dParam {
    /// Java `command instanceof ProcessDetails`: this class is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::FIND_BEADS_3D)
    }

    fn get_command(&self) -> Option<String> {
        file_type::CLASS
            .find_beads_3d_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    /// Java `getCommandName`.  The axis type it looks up is unused, as in the source.
    fn get_command_name(&self) -> Option<String> {
        let mut axis_type: Option<AxisType> = None;
        if let Some(meta_data) = self.manager.get_base_meta_data() {
            axis_type = Some(meta_data.base().get_axis_type());
        }
        let _ = axis_type;
        ProcessName::FIND_BEADS_3D.get_text().map(str::to_owned)
    }

    fn get_command_line(&self) -> Option<String> {
        file_type::CLASS
            .find_beads_3d_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    /// Java `getCommandArray`: `{ getCommandLine() }`.  A null command line is a null
    /// element in Java; here it is the string Java concatenation would make of it.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let array = vec![self.get_command_line().unwrap_or_else(|| "null".to_owned())];
        Some(array)
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType`, deprecated 3/15/2019.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2`, deprecated 3/15/2019.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}

/// Every Java getter here throws `IllegalArgumentException("field=" + field)`: no field
/// is handled, so each returns `None`.
impl ProcessDetails for FindBeads3dParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }
}
