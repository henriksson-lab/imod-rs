//! `IMOD/Etomo/src/etomo/comscript/MTFFilterParam.java`.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_mtf_filter_param::ConstMTFFilterParam;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;
use super::param_utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Type, java_lang_double_value_of};
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java `FIXED_IMAGE_DOSE_KEY`.
pub const FIXED_IMAGE_DOSE_KEY: &str = "FixedImageDose";
/// Java `DOSE_WEIGHTING_FILE_KEY`.
pub const DOSE_WEIGHTING_FILE_KEY: &str = "DoseWeightingFile";
/// Java `TYPE_OF_DOSE_FILE_KEY`.
pub const TYPE_OF_DOSE_FILE_KEY: &str = "TypeOfDoseFile";
/// Java `VOLTAGE_200`.
pub const VOLTAGE_200: i32 = 200;
/// Java `OPTIMAL_DOSE_SCALING_KEY`.
pub const OPTIMAL_DOSE_SCALING_KEY: &str = "OptimalDoseScaling";
/// Java `BIDIRECTIONAL_NUM_VIEWS_KEY`.
pub const BIDIRECTIONAL_NUM_VIEWS_KEY: &str = "BidirectionalNumViews";
/// Java `PIXEL_SIZE_OPTION`.
pub const PIXEL_SIZE_OPTION: &str = "PixelSize";

/// Java final `MTFFilterParam`.
pub struct MTFFilterParam {
    low_pass_radius_sigma: FortranInputString,
    inverse_rolloff_radius_sigma: FortranInputString,
    starting_and_ending_z: FortranInputString,
    fixed_image_dose: ScriptParameter,
    dose_weighting_file: StringParameter,
    type_of_dose_file: ScriptParameter,
    voltage: ScriptParameter,
    optimal_dose_scaling: ScriptParameter,
    bidirectional_num_views: ScriptParameter,
    pixel_size: ScriptParameter,

    manager: &'static dyn BaseManager,
    axis_id: AxisID,

    /// Java `inputFile`; `scriptCommand.getValue` can make it null.
    input_file: Option<String>,
    output_file: Option<String>,
    mtf_file: Option<String>,
    maximum_inverse: f64,
}

impl MTFFilterParam {
    /// Java package-private `MTFFilterParam(BaseManager, AxisID)`.
    pub(crate) fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> MTFFilterParam {
        let mut instance = MTFFilterParam {
            low_pass_radius_sigma: FortranInputString::new(2),
            inverse_rolloff_radius_sigma: FortranInputString::new(2),
            starting_and_ending_z: FortranInputString::new(2),
            fixed_image_dose: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FIXED_IMAGE_DOSE_KEY,
            ),
            dose_weighting_file: StringParameter::new(DOSE_WEIGHTING_FILE_KEY),
            type_of_dose_file: ScriptParameter::new_with_name(TYPE_OF_DOSE_FILE_KEY),
            voltage: ScriptParameter::new_with_name("Voltage"),
            optimal_dose_scaling: ScriptParameter::new_with_type_and_name(
                Type::Double,
                OPTIMAL_DOSE_SCALING_KEY,
            ),
            bidirectional_num_views: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                BIDIRECTIONAL_NUM_VIEWS_KEY,
            ),
            pixel_size: ScriptParameter::new_with_type_and_name(Type::Double, PIXEL_SIZE_OPTION),
            manager,
            axis_id,
            input_file: Some(String::new()),
            output_file: Some(String::new()),
            mtf_file: Some(String::new()),
            maximum_inverse: f64::NAN,
        };
        instance
            .starting_and_ending_z
            .set_integer_type_index(0, true);
        instance
            .starting_and_ending_z
            .set_integer_type_index(1, true);
        instance.maximum_inverse = f64::NAN;
        instance
    }

    /// Java `resetOptimalDoseScaling`.
    pub fn reset_optimal_dose_scaling(&mut self) {
        self.optimal_dose_scaling.reset();
    }

    /// Java `resetBidirectionalNumViews`.
    pub fn reset_bidirectional_num_views(&mut self) {
        self.bidirectional_num_views.reset();
    }

    /// Java `setOptimalDoseScaling`.
    pub fn set_optimal_dose_scaling(&mut self, input: Option<&str>) {
        self.optimal_dose_scaling.set_string(input);
    }

    /// Java `setBidirectionalNumViews`.
    pub fn set_bidirectional_num_views(&mut self, input: Option<&str>) {
        self.bidirectional_num_views.set_string(input);
    }

    /// Java `setPixelSize`.
    pub fn set_pixel_size(&mut self, input: f64) {
        self.pixel_size.set_double(input);
    }

    /// Java `resetVoltage`.
    pub fn reset_voltage(&mut self) {
        self.voltage.reset();
    }

    /// Java `setVoltage200`.
    pub fn set_voltage200(&mut self, set: bool) {
        if set {
            self.voltage.set_int(VOLTAGE_200);
        } else {
            self.voltage.reset();
        }
    }

    /// Java `resetTypeOfDoseFile`.
    pub fn reset_type_of_dose_file(&mut self) {
        self.type_of_dose_file.reset();
    }

    /// Java `setTypeOfDoseFile(EnumeratedType)`.
    pub fn set_type_of_dose_file(&mut self, enumerated_type: Option<&dyn EnumeratedType>) {
        match enumerated_type {
            None => {
                self.type_of_dose_file.reset();
            }
            Some(enumerated_type) => {
                self.type_of_dose_file
                    .set_const_etomo_number(Some(&enumerated_type.get_value()));
            }
        }
    }

    /// Java `setInputFile`.
    pub fn set_input_file(&mut self, input_file: Option<&str>) {
        self.input_file = input_file.map(str::to_owned);
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, output_file: Option<&str>) {
        self.output_file = output_file.map(str::to_owned);
    }

    /// Java `setMtfFile`.
    pub fn set_mtf_file(&mut self, mtf_file: Option<&str>) {
        self.mtf_file = mtf_file.map(str::to_owned);
    }

    /// Java `setFixedImageDose`.
    pub fn set_fixed_image_dose(&mut self, input: Option<&str>) {
        self.fixed_image_dose.set_string(input);
    }

    /// Java `setDoseWeightingFile`.
    pub fn set_dose_weighting_file(&mut self, input: Option<&str>) {
        self.dose_weighting_file.set(input);
    }

    /// Java `resetDoseWeightingFile`.
    pub fn reset_dose_weighting_file(&mut self) {
        self.dose_weighting_file.reset();
    }

    /// Java `resetFixedImageDose`.
    pub fn reset_fixed_image_dose(&mut self) {
        self.fixed_image_dose.reset();
    }

    /// Java `resetMtfFile`.
    pub fn reset_mtf_file(&mut self) {
        self.mtf_file = Some(String::new());
    }

    /// Java `setMaximumInverse`.  `ParamUtilities.parseDouble` throws an unchecked
    /// NumberFormatException for a non-numeric entry; it comes back as `Err` (the
    /// field is left unchanged, as the throw leaves it).
    pub fn set_maximum_inverse(&mut self, maximum_inverse: Option<&str>) -> Result<(), String> {
        self.maximum_inverse = param_utilities::parse_double(maximum_inverse)?;
        Ok(())
    }

    /// Java `resetMaximumInverse`.
    pub fn reset_maximum_inverse(&mut self) {
        self.maximum_inverse = f64::NAN;
    }

    /// Java `setLowPassRadiusSigma`.
    pub fn set_low_pass_radius_sigma(
        &mut self,
        low_pass_radius_sigma: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(
            low_pass_radius_sigma,
            &mut self.low_pass_radius_sigma,
        )
    }

    /// Java `resetLowPassRadiusSigma`.
    pub fn reset_low_pass_radius_sigma(&mut self) {
        self.low_pass_radius_sigma.set_default();
    }

    /// Java `setInverseRolloffRadiusSigma`.
    pub fn set_inverse_rolloff_radius_sigma(
        &mut self,
        inverse_rolloff_radius_sigma: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(
            inverse_rolloff_radius_sigma,
            &mut self.inverse_rolloff_radius_sigma,
        )
    }

    /// Java `resetInverseRolloffRadiusSigma`.
    pub fn reset_inverse_rolloff_radius_sigma(&mut self) {
        self.inverse_rolloff_radius_sigma.set_default();
    }

    /// Java `setStartingAndEndingZ`.
    pub fn set_starting_and_ending_z(
        &mut self,
        starting_and_ending_z: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(
            starting_and_ending_z,
            &mut self.starting_and_ending_z,
        )
    }
}

impl CommandParam for MTFFilterParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // The source reads the command line arguments into a local it never uses.
        let _cmd_line_args = script_command.get_command_line_args();
        if script_command.is_keyword_value_pairs() {
            if script_command.has_keyword(Some("InputFile"))? {
                self.input_file = script_command.get_value(Some("InputFile"))?;
            }
            if script_command.has_keyword(Some("OutputFile"))? {
                self.output_file = script_command.get_value(Some("OutputFile"))?;
            }
            if script_command.has_keyword(Some("MtfFile"))? {
                self.mtf_file = script_command.get_value(Some("MtfFile"))?;
            }
            if script_command.has_keyword(Some("MaximumInverse"))? {
                // `Double.parseDouble` throws the unchecked NullPointerException /
                // NumberFormatException, which nothing here catches.
                let value = script_command
                    .get_value(Some("MaximumInverse"))?
                    .expect("java.lang.NullPointerException");
                self.maximum_inverse =
                    java_lang_double_value_of(&value).unwrap_or_else(|message| {
                        panic!("java.lang.NumberFormatException: {}", message)
                    });
            }
            if script_command.has_keyword(Some("LowPassRadiusSigma"))? {
                let value = script_command.get_value(Some("LowPassRadiusSigma"))?;
                self.low_pass_radius_sigma
                    .validate_and_set(value.as_deref())?;
            }
            if script_command.has_keyword(Some("InverseRolloffRadiusSigma"))? {
                let value = script_command.get_value(Some("InverseRolloffRadiusSigma"))?;
                self.inverse_rolloff_radius_sigma
                    .validate_and_set(value.as_deref())?;
            }
            if script_command.has_keyword(Some("StartingAndEndingZ"))? {
                let value = script_command.get_value(Some("StartingAndEndingZ"))?;
                self.starting_and_ending_z
                    .validate_and_set(value.as_deref())?;
            }
            self.fixed_image_dose.parse(script_command)?;
            self.dose_weighting_file.parse(script_command)?;
            self.type_of_dose_file.parse(script_command)?;
            self.voltage.parse(script_command)?;
            self.optimal_dose_scaling.parse(script_command)?;
            self.bidirectional_num_views.parse(script_command)?;
        } else {
            return Err(InvalidParameterException::new(Some(
                "MTF Filter:  Missing parameter, -StandardInput.  Use Etomo to create .com file.",
            ))
            .into());
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        param_utilities::update_script_parameter_string_required(
            script_command,
            Some("InputFile"),
            self.input_file.as_deref(),
            true,
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some("OutputFile"),
            self.output_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some("MtfFile"),
            self.mtf_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_double(
            script_command,
            Some("MaximumInverse"),
            self.maximum_inverse,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("LowPassRadiusSigma"),
            &self.low_pass_radius_sigma,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("InverseRolloffRadiusSigma"),
            &self.inverse_rolloff_radius_sigma,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("StartingAndEndingZ"),
            &self.starting_and_ending_z,
        );
        self.fixed_image_dose.update_com_script(script_command);
        self.dose_weighting_file.update_com_script(script_command);
        self.type_of_dose_file.update_com_script(script_command);
        self.voltage.update_com_script(script_command);
        self.optimal_dose_scaling.update_com_script(script_command);
        self.bidirectional_num_views
            .update_com_script(script_command);
        self.pixel_size.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl ConstMTFFilterParam for MTFFilterParam {
    fn get_optimal_dose_scaling(&self) -> String {
        self.optimal_dose_scaling.to_string()
    }

    fn get_bidirectional_num_views(&self) -> String {
        self.bidirectional_num_views.to_string()
    }

    fn is_optimal_dose_scaling_set(&self) -> bool {
        self.optimal_dose_scaling.is()
    }

    fn is_bidirectional_num_views_set(&self) -> bool {
        self.bidirectional_num_views.is()
    }

    fn is_voltage200(&self) -> bool {
        self.voltage.equals_int(VOLTAGE_200)
    }

    fn get_type_of_dose_file(&self) -> String {
        self.type_of_dose_file.to_string()
    }

    fn get_dose_weighting_file(&self) -> String {
        self.dose_weighting_file.to_string()
    }

    fn is_dose_weighting_file_set(&self) -> bool {
        !self.dose_weighting_file.is_empty()
    }

    fn is_type_of_dose_file_set(&self) -> bool {
        !self.type_of_dose_file.is_null()
    }

    fn is_inverse_rolloff_radius_sigma_set(&self) -> bool {
        !self.inverse_rolloff_radius_sigma.is_null()
    }

    fn get_fixed_image_dose(&self) -> String {
        self.fixed_image_dose.to_string()
    }

    fn is_fixed_image_dose_set(&self) -> bool {
        self.fixed_image_dose.is()
    }

    fn get_mtf_file(&self) -> Option<String> {
        self.mtf_file.clone()
    }

    fn is_mtf_file_set(&self) -> bool {
        self.mtf_file
            .as_deref()
            .is_some_and(|mtf_file| mtf_file != "")
    }

    fn get_maximum_inverse_string(&self) -> String {
        param_utilities::value_of_double(self.maximum_inverse)
    }

    /// Fixed in translation: MTFFilterParam.java:293 is
    /// `maximumInverse != Double.NaN`, which is true for every value (NaN compares
    /// unequal to everything, itself included), so the source reports an unset
    /// maximum inverse as set.  The evident intent is a NaN test, done here.
    fn is_maximum_inverse_set(&self) -> bool {
        !self.maximum_inverse.is_nan()
    }

    fn get_low_pass_radius_sigma_string(&self) -> String {
        self.low_pass_radius_sigma.to_string_default_is_blank(true)
    }

    fn is_low_pass_radius_sigma_set(&self) -> bool {
        !self.low_pass_radius_sigma.is_null()
    }

    fn get_starting_and_ending_z_string(&self) -> String {
        self.starting_and_ending_z.to_string_default_is_blank(true)
    }

    fn is_starting_z_set(&self) -> bool {
        !self.starting_and_ending_z.is_null_index(0) && !self.starting_and_ending_z.is_null_index(0)
    }

    fn is_ending_z_set(&self) -> bool {
        !self.starting_and_ending_z.is_null_index(1) && !self.starting_and_ending_z.is_null_index(1)
    }

    fn get_starting_z(&self) -> i32 {
        self.starting_and_ending_z.get_int(0)
    }

    fn get_ending_z(&self) -> i32 {
        self.starting_and_ending_z.get_int(1)
    }

    fn get_inverse_rolloff_radius_sigma_string(&self) -> String {
        self.inverse_rolloff_radius_sigma
            .to_string_default_is_blank(true)
    }

    fn get_output_file(&self) -> Option<String> {
        self.output_file.clone()
    }
}

impl Command for MTFFilterParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::MTFFILTER)
    }

    fn get_command(&self) -> Option<String> {
        file_type::CLASS
            .mtf_filter_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::MTFFILTER.to_string())
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `{ getCommandLine() }`; a null element is dropped here.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.get_command_line().into_iter().collect())
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        Some(file_type::CLASS.mtf_filtered_stack.clone())
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.mtf_filtered_stack))
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
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
