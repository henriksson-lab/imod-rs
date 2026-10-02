//! `IMOD/Etomo/src/etomo/comscript/SortTiltFramesParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::SORT_TILT_FRAMES;

/// Java package-private `COMMAND`.
pub(crate) const COMMAND: &str = "sorttiltframes";

/// Java `DELIMITER_OPEN_DEFAULT`.
pub const DELIMITER_OPEN_DEFAULT: &str = "[";
/// Java `DELIMITER_CLOSE_DEFAULT`.
pub const DELIMITER_CLOSE_DEFAULT: &str = "]";

/// Java `INPUT_FILE`.
pub const INPUT_FILE: &str = "InputFile";
/// Java `LIST_OF_INPUT_FILES`.
pub const LIST_OF_INPUT_FILES: &str = "ListOfInputFiles";
/// Java `TILT_SERIES_FILE`.
pub const TILT_SERIES_FILE: &str = "TiltSeriesFile";
/// Java `TILT_ANGLE_FILE`.
pub const TILT_ANGLE_FILE: &str = "TiltAngleFile";
/// Java `OUTPUT_FILE_LIST`.
pub const OUTPUT_FILE_LIST: &str = "OutputFileList";
/// Java `OUTPUT_TILT_ANGLE_FILE`.
pub const OUTPUT_TILT_ANGLE_FILE: &str = "OutputTiltAngleFile";
/// Java `REVERSE_ORDER`.
pub const REVERSE_ORDER: &str = "ReverseOrder";
/// Java `UNSORTED_OUTPUT`.
pub const UNSORTED_OUTPUT: &str = "UnsortedOutput";
/// Java `DELIMITERS`.
pub const DELIMITERS: &str = "Delimiters";
/// Java `FIXED_IMAGE_DOSE`.
pub const FIXED_IMAGE_DOSE: &str = "FixedImageDose";
/// Java `DOSE_OUTPUT_FILE`.
pub const DOSE_OUTPUT_FILE: &str = "DoseOutputFile";

/// Java `OUTPUT_FILE_LIST_INLIST`.
pub const OUTPUT_FILE_LIST_INLIST: &str = "_inlist";
/// Java `OUTPUT_FILE_LIST_TEMPLIST`.
pub const OUTPUT_FILE_LIST_TEMPLIST: &str = "_templist";
/// Java `OUTPUT_TILT_ANGLE_FILE_MATCHING`.
pub const OUTPUT_TILT_ANGLE_FILE_MATCHING: &str = "_matching";
/// Java `DOSE_OUTPUT_FILE_DOSE`.
pub const DOSE_OUTPUT_FILE_DOSE: &str = "_dose";

/// Java `SortTiltFramesParam`.
pub struct SortTiltFramesParam {
    command_array: Option<Vec<String>>,
    input_file: StringParameter,
    list_of_input_files: StringParameter,
    tilt_series_file: StringParameter,
    tilt_angle_file: StringParameter,
    output_file_list: StringParameter,
    output_tilt_angle_file: StringParameter,
    reverse_order: EtomoBoolean2,
    unsorted_output: EtomoBoolean2,
    delimiters: StringParameter,
    fixed_image_dose: ScriptParameter,
    dose_output_file: StringParameter,
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    /// `ToolsManager` constructs the param with a null axis.
    #[allow(dead_code)]
    axis_id: Option<AxisID>,
    #[allow(dead_code)]
    align_frames_com_filename: Option<String>,
}

impl SortTiltFramesParam {
    /// Java `SortTiltFramesParam(BaseManager, AxisID, String)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        align_frames_com_filename: Option<&str>,
    ) -> SortTiltFramesParam {
        SortTiltFramesParam {
            command_array: None,
            input_file: StringParameter::new(INPUT_FILE),
            list_of_input_files: StringParameter::new(LIST_OF_INPUT_FILES),
            tilt_series_file: StringParameter::new(TILT_SERIES_FILE),
            tilt_angle_file: StringParameter::new(TILT_ANGLE_FILE),
            output_file_list: StringParameter::new(OUTPUT_FILE_LIST),
            output_tilt_angle_file: StringParameter::new(OUTPUT_TILT_ANGLE_FILE),
            reverse_order: EtomoBoolean2::new_with_name(REVERSE_ORDER),
            unsorted_output: EtomoBoolean2::new_with_name(UNSORTED_OUTPUT),
            delimiters: StringParameter::new(DELIMITERS),
            fixed_image_dose: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FIXED_IMAGE_DOSE,
            ),
            dose_output_file: StringParameter::new(DOSE_OUTPUT_FILE),
            manager,
            axis_id,
            align_frames_com_filename: align_frames_com_filename.map(str::to_owned),
        }
    }

    /// Java private `createCommand`.
    fn create_command(&mut self) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes null parts as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!(
            "{script_path}{}",
            PROCESS_NAME.get_text().unwrap_or("null")
        ));
        command.push(format!("-{}", self.list_of_input_files.get_name()));
        command.push(self.list_of_input_files.to_string());
        // AlignFramesParam alignFramesParam =
        // new AlignFramesParam(manager, alignFramesComFilename);
        if self.is_tilt_series_file() {
            command.push(format!("-{}", self.tilt_series_file.get_name()));
            command.push(self.tilt_series_file.to_string());
        }
        if self.is_tilt_angle_file() {
            command.push(format!("-{}", self.tilt_angle_file.get_name()));
            command.push(self.tilt_angle_file.to_string());
        }
        if self.is_tilt_series_file() || self.is_tilt_angle_file() {
            command.push(format!("-{}", self.output_file_list.get_name()));
            command.push(self.output_file_list.to_string());
        } else {
            command.push(format!("-{}", self.output_tilt_angle_file.get_name()));
            command.push(self.output_tilt_angle_file.to_string());
            if self.unsorted_output.is() {
                command.push(format!("-{}", self.unsorted_output.get_name()));
            }
            if self.is_fixed_image_dose() {
                command.push(format!("-{}", self.fixed_image_dose.get_name()));
                command.push(self.fixed_image_dose.to_string());
                command.push(format!("-{}", self.dose_output_file.get_name()));
                command.push(self.dose_output_file.to_string());
            }
        }
        command.push(format!("-{}", self.delimiters.get_name()));
        command.push(self.delimiters.to_string());

        // LEFT AS EXAMPLES FOR BOOLEAN TYPES (TO BE DELETED)
        // if (montagedImages.is()) {
        // command.add("-" + montagedImages.getName());
        // }
        // if (deleteOldFiles.is()) {
        // command.add("-" + deleteOldFiles.getName());
        // }

        self.command_array = Some(command);
    }

    /// Java `getCommandArray`.
    pub fn get_command_array(&mut self) -> Vec<String> {
        self.create_command();
        self.command_array.clone().unwrap_or_default()
    }

    /// Java `isTiltSeriesFile`.
    pub fn is_tilt_series_file(&self) -> bool {
        !self.tilt_series_file.is_empty()
    }

    /// Java `isTiltAngleFile`.
    pub fn is_tilt_angle_file(&self) -> bool {
        !self.tilt_angle_file.is_empty()
    }

    /// Java `isFixedImageDose`.
    pub fn is_fixed_image_dose(&self) -> bool {
        self.fixed_image_dose.is()
    }

    /// Java `getDoseOutputFile`.
    pub fn get_dose_output_file(&self) -> String {
        self.dose_output_file.to_string()
    }

    /// Java `setTiltSeriesFile(String)`.
    pub fn set_tilt_series_file(&mut self, input: Option<&str>) {
        self.tilt_series_file.set(input);
    }

    /// Java `setTiltAngleFile(String)`.
    pub fn set_tilt_angle_file(&mut self, input: Option<&str>) {
        self.tilt_angle_file.set(input);
    }

    /// Java `setDelimiters(String)`.
    pub fn set_delimiters(&mut self, input: Option<&str>) {
        self.delimiters.set(input);
    }

    /// Java `setListOfInputFiles(String)`.
    pub fn set_list_of_input_files(&mut self, input: Option<&str>) {
        self.list_of_input_files.set(input);
    }

    /// Java `setOutputFileList(String)`.
    pub fn set_output_file_list(&mut self, input: Option<&str>) {
        self.output_file_list.set(input);
    }

    /// Java `setOutputTiltAngleFile(String)`.
    pub fn set_output_tilt_angle_file(&mut self, input: Option<&str>) {
        self.output_tilt_angle_file.set(input);
    }

    /// Java `setUnsortedOutput(boolean)`.
    pub fn set_unsorted_output(&mut self, input: bool) {
        self.unsorted_output.set_boolean(input);
    }

    /// Java `setFixedImageDose(String)`.
    pub fn set_fixed_image_dose(&mut self, input: Option<&str>) {
        self.fixed_image_dose.set_string(input);
    }

    /// Java `setDoseOutputFile(String)`.
    pub fn set_dose_output_file(&mut self, input: Option<&str>) {
        self.dose_output_file.set(input);
    }
}

impl CommandParam for SortTiltFramesParam {
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // `scriptCommand.useKeywordValue()` mutates the command; the trait hands the
        // command in by shared reference, so it is applied to a copy that the parse
        // reads from.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        self.input_file.parse(script_command)?;
        self.list_of_input_files.parse(script_command)?;
        self.tilt_series_file.parse(script_command)?;
        self.tilt_angle_file.parse(script_command)?;
        self.output_file_list.parse(script_command)?;
        self.output_tilt_angle_file.parse(script_command)?;
        self.reverse_order.parse(script_command)?;
        self.unsorted_output.parse(script_command)?;
        self.delimiters.parse(script_command)?;
        self.fixed_image_dose.parse(script_command)?;
        self.dose_output_file.parse(script_command)?;
        Ok(())
    }

    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.input_file.update_com_script(script_command);
        self.list_of_input_files.update_com_script(script_command);
        self.tilt_series_file.update_com_script(script_command);
        self.tilt_angle_file.update_com_script(script_command);
        self.output_file_list.update_com_script(script_command);
        self.output_tilt_angle_file
            .update_com_script(script_command);
        self.reverse_order.update_com_script(script_command);
        self.unsorted_output.update_com_script(script_command);
        self.delimiters.update_com_script(script_command);
        self.fixed_image_dose.update_com_script(script_command);
        self.dose_output_file.update_com_script(script_command);
        Ok(())
    }

    fn initialize_defaults(&mut self) {
        self.input_file.reset();
        self.list_of_input_files.reset();
        self.tilt_series_file.reset();
        self.tilt_angle_file.reset();
        self.output_file_list.reset();
        self.output_tilt_angle_file.reset();
        self.reverse_order.reset();
        self.unsorted_output.reset();
        self.delimiters.reset();
        self.fixed_image_dose.reset();
        self.dose_output_file.reset();
    }
}
