//! `IMOD/Etomo/src/etomo/comscript/CryoPositionParam.java`.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::CRYO_POSITION;
/// Java private static `LEAVE_TEMP_FILES_DEFAULT`.
const LEAVE_TEMP_FILES_DEFAULT: i32 = -1;

/// Java final `CryoPositionParam`.
pub struct CryoPositionParam {
    axis_id: AxisID,
    bead_size: ScriptParameter,
    find_beads_in_volume: ScriptParameter,
    leave_temp_files: ScriptParameter,
    thickness_of_tomograms: ScriptParameter,
    tilt_param: Option<Arc<dyn ConstTiltParam + Send + Sync>>,
}

impl CryoPositionParam {
    /// Java package-private `CryoPositionParam(AxisID)`.
    pub fn new(axis_id: AxisID) -> CryoPositionParam {
        let mut leave_temp_files = ScriptParameter::new_with_name("LeaveTempFiles");
        leave_temp_files.set_display_value_int(LEAVE_TEMP_FILES_DEFAULT);
        CryoPositionParam {
            axis_id,
            bead_size: ScriptParameter::new_with_type_and_name(Type::Double, "BeadSize"),
            find_beads_in_volume: ScriptParameter::new_with_name("FindBeadsInVolume"),
            leave_temp_files,
            thickness_of_tomograms: ScriptParameter::new_with_name("ThicknessOfTomograms"),
            tilt_param: None,
        }
    }

    /// Java `setThicknessOfTomograms`.
    pub fn set_thickness_of_tomograms(&mut self, input: Option<&str>) {
        self.thickness_of_tomograms.set_string(input);
    }

    /// Java `resetBeadSize`.
    pub fn reset_bead_size(&mut self) {
        self.bead_size.reset();
    }

    /// Java `setBeadSize`.
    pub fn set_bead_size(&mut self, input: Option<&str>) {
        self.bead_size.set_string(input);
    }

    /// Java `isBeadSizeSet`.
    pub fn is_bead_size_set(&self) -> bool {
        !self.bead_size.is_null()
    }

    /// Java `getBeadSize`.
    pub fn get_bead_size(&self) -> String {
        self.bead_size.to_string()
    }

    /// Java `setFindBeadsInVolume`.
    pub fn set_find_beads_in_volume(&mut self, input: bool) {
        if input {
            self.find_beads_in_volume.set_int(2);
        } else {
            self.find_beads_in_volume.reset();
        }
    }

    /// Java `setTiltParam`.
    pub fn set_tilt_param(&mut self, param: Option<Arc<dyn ConstTiltParam + Send + Sync>>) {
        self.tilt_param = param;
    }

    /// Java `getSubcommandDetails`, as the shared object itself: the process
    /// manager hands it to the event dispatch thread
    /// (`ProcessManager.postProcess(ComScriptProcess)`, CRYO_POSITION).
    pub fn get_subcommand_details_shared(&self) -> Option<Arc<dyn ConstTiltParam + Send + Sync>> {
        // Cryoposition uses tilt.com. Information from tilt.com is required for post
        // processing
        self.tilt_param.clone()
    }
}

impl CommandParam for CryoPositionParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.bead_size.parse(script_command)?;
        self.find_beads_in_volume.parse(script_command)?;
        self.leave_temp_files.parse(script_command)?;
        self.thickness_of_tomograms.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.bead_size.update_com_script(script_command);
        self.find_beads_in_volume.update_com_script(script_command);
        self.leave_temp_files.update_com_script(script_command);
        self.thickness_of_tomograms
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl Command for CryoPositionParam {
    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(vec![PROCESS_NAME.get_comscript(self.axis_id)])
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        // Cryoposition uses tilt.com. Information from tilt.com is required for post
        // processing
        match &self.tilt_param {
            None => None,
            Some(tilt_param) => {
                let tilt_param: &(dyn ConstTiltParam + Send + Sync) = tilt_param.as_ref();
                Some(tilt_param as &dyn CommandDetails)
            }
        }
    }
}
