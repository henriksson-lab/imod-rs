//! `IMOD/Etomo/src/etomo/comscript/TomopitchParam.java`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_tomopitch_param::{
    COMMAND, ConstTomopitchParam, MODEL_FILE, PARAMETER_FILE, SCALE_FACTOR, SPACING_IN_Y,
};
use super::invalid_parameter_exception::InvalidParameterException;
use super::param_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_value_of, java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::pos_sample_type::PosSampleType;
use crate::imod::etomo::util::utilities;

/// Java `TomopitchParam extends ConstTomopitchParam implements CommandParam`.
pub struct TomopitchParam {
    /// Java superclass `ConstTomopitchParam` state.
    pub base: ConstTomopitchParam,
}

/// Java inheritance: every `ConstTomopitchParam` member is reachable on a
/// `TomopitchParam`.
impl std::ops::Deref for TomopitchParam {
    type Target = ConstTomopitchParam;

    fn deref(&self) -> &ConstTomopitchParam {
        &self.base
    }
}

impl std::ops::DerefMut for TomopitchParam {
    fn deref_mut(&mut self) -> &mut ConstTomopitchParam {
        &mut self.base
    }
}

/// `Double.parseDouble(String)`.  Its unchecked `NumberFormatException` (and the
/// `NullPointerException` of a null value) would escape `parseComScriptCommand` as a
/// crash in Java; here it is reported as an `InvalidParameterException` carrying the
/// Java exception text.
fn parse_double(value: Option<&str>) -> Result<f64, InvalidParameterException> {
    match value {
        None => Err(InvalidParameterException::new(Some(
            "java.lang.NullPointerException",
        ))),
        Some(value) => java_lang_double_value_of(value).map_err(|message| {
            InvalidParameterException::new(Some(&format!(
                "java.lang.NumberFormatException: {message}"
            )))
        }),
    }
}

impl TomopitchParam {
    /// Java `TomopitchParam(ApplicationManager, AxisID)`.
    pub fn new(manager: &'static ApplicationManager, axis_id: AxisID) -> TomopitchParam {
        TomopitchParam {
            base: ConstTomopitchParam::new(manager, axis_id),
        }
    }

    /// Java `resetModelFiles`.
    pub fn reset_model_files(&mut self) {
        self.base.model_files = Vec::new();
    }

    /// Java `setModelFile(String)`.
    pub fn set_model_file(&mut self, model_file: Option<&str>) {
        self.base.model_files.push(model_file.map(str::to_owned));
    }

    /// Java `setExtraThickness(String)`.
    pub fn set_extra_thickness(&mut self, extra_thickness: Option<&str>) {
        self.base.extra_thickness.set_string(extra_thickness);
    }

    /// Java `setNoXAxisTilt(boolean)`.
    pub fn set_no_x_axis_tilt(&mut self, input: bool) {
        self.base.no_x_axis_tilt.set_boolean(input);
    }

    /// Java `setSpacingInY(String)`.  `ParamUtilities.parseDouble` throws an unchecked
    /// NumberFormatException for a non-numeric entry; it comes back as `Err` (the
    /// field is left unchanged, as the throw leaves it).
    pub fn set_spacing_in_y(&mut self, spacing_in_y: Option<&str>) -> Result<(), String> {
        self.base.spacing_in_y = param_utilities::parse_double(spacing_in_y)?;
        Ok(())
    }

    /// Java `setScaleFactor(boolean)`.
    pub fn set_scale_factor(&mut self, dialog_whole_tomogram: bool) {
        // Java tests `manager != null` and `state != null`; neither can be null here.
        let state = self.base.manager.get_state();
        let pos_sample_type: Option<PosSampleType> =
            PosSampleType::get_instance(state.get_pos_sample_type(self.base.axis_id));
        // Use the dialog whole tomogram as a backup. The state contains what was last run.
        let mut image_file: &Arc<FileType> = &file_type::CLASS.tilt_output;
        if pos_sample_type == Some(PosSampleType::SAMPLES)
            || (pos_sample_type.is_none() && !dialog_whole_tomogram)
        {
            image_file = &file_type::CLASS.top_sample;
        }
        self.base.scale_factor = utilities::get_stack_binning_for_file_type(
            self.base.manager,
            self.base.axis_id,
            image_file,
        ) as f64;
    }

    /// Java `setParameterFile(String)`.
    pub fn set_parameter_file(&mut self, parameter_file: Option<&str>) {
        self.base.parameter_file = parameter_file.map(str::to_owned);
    }

    /// Java `setAngleOffsetOld(ConstEtomoNumber)`.
    pub fn set_angle_offset_old_const_etomo_number(
        &mut self,
        angle_offset_old: Option<&ConstEtomoNumber>,
    ) {
        self.base
            .angle_offset_old
            .set_const_etomo_number(angle_offset_old);
    }

    /// Java `setAngleOffsetOld(String)`.
    pub fn set_angle_offset_old_string(&mut self, angle_offset_old: Option<&str>) {
        self.base.angle_offset_old.set_string(angle_offset_old);
    }

    /// Java `setZShiftOld(ConstEtomoNumber)`.
    pub fn set_z_shift_old_const_etomo_number(&mut self, z_shift_old: Option<&ConstEtomoNumber>) {
        self.base.z_shift_old.set_const_etomo_number(z_shift_old);
    }

    /// Java `setZShiftOld(String)`.
    pub fn set_z_shift_old_string(&mut self, z_shift_old: Option<&str>) {
        self.base.z_shift_old.set_string(z_shift_old);
    }

    /// Java `setXAxisTiltOld(ConstEtomoNumber)`.
    pub fn set_x_axis_tilt_old(&mut self, x_axis_tilt_old: Option<&ConstEtomoNumber>) {
        self.base
            .x_axis_tilt_old
            .set_const_etomo_number(x_axis_tilt_old);
    }

    /// Java private `getInputArguments(ComScriptCommand)`.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        // Check to be sure that it is a tiltxcorr xommand.  A null command is a
        // NullPointerException in Java; here it is not a tomopitch command.
        if script_command.get_command() != Some(COMMAND) {
            return Err(BadComScriptException::new(&format!(
                "Not a {COMMAND} command"
            )));
        }
        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        Ok(input_args)
    }
}

impl CommandParam for TomopitchParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // get the input arguments from the command
        let input_args = self.get_input_arguments(script_command)?;
        let _cmd_line_args = script_command.get_command_line_args();
        if script_command.is_keyword_value_pairs() {
            if script_command.has_keyword(Some(MODEL_FILE))? {
                let model_files = script_command.get_values(Some(MODEL_FILE));
                for model_file in model_files {
                    self.base.model_files.push(model_file);
                }
            }
            let name = self.base.extra_thickness.get_name().to_owned();
            if script_command.has_keyword(Some(&name))? {
                let value = script_command.get_value(Some(&name))?;
                self.base.extra_thickness.set_string(value.as_deref());
            }
            let name = self.base.no_x_axis_tilt.get_name().to_owned();
            if script_command.has_keyword(Some(&name))? {
                self.base.no_x_axis_tilt.parse(script_command)?;
            }
            if script_command.has_keyword(Some(SPACING_IN_Y))? {
                self.base.spacing_in_y =
                    parse_double(script_command.get_value(Some(SPACING_IN_Y))?.as_deref())?;
            }
            if script_command.has_keyword(Some(SCALE_FACTOR))? {
                self.base.scale_factor =
                    parse_double(script_command.get_value(Some(SCALE_FACTOR))?.as_deref())?;
            }
            if script_command.has_keyword(Some(PARAMETER_FILE))? {
                self.base.parameter_file = script_command.get_value(Some(PARAMETER_FILE))?;
            }
            let name = self.base.angle_offset_old.get_name().to_owned();
            if script_command.has_keyword(Some(&name))? {
                let value = script_command.get_value(Some(&name))?;
                self.base.angle_offset_old.set_string(value.as_deref());
            }
            let name = self.base.z_shift_old.get_name().to_owned();
            if script_command.has_keyword(Some(&name))? {
                let value = script_command.get_value(Some(&name))?;
                self.base.z_shift_old.set_string(value.as_deref());
            }
            // The source repeats the zShiftOld test.
            if script_command.has_keyword(Some(&name))? {
                let value = script_command.get_value(Some(&name))?;
                self.base.z_shift_old.set_string(value.as_deref());
            }
            let name = self.base.x_axis_tilt_old.get_name().to_owned();
            if script_command.has_keyword(Some(&name))? {
                let value = script_command.get_value(Some(&name))?;
                self.base.x_axis_tilt_old.set_string(value.as_deref());
            }
            return Ok(());
        }
        // `inputArgs[inputLine++]` throws ArrayIndexOutOfBoundsException in Java when the
        // script has too few input lines, and `Integer.parseInt` a NumberFormatException;
        // both are reported as an InvalidParameterException here.
        let argument = |index: usize| -> Result<Option<String>, InvalidParameterException> {
            match input_args.get(index) {
                None => Err(InvalidParameterException::new(Some(&format!(
                    "java.lang.ArrayIndexOutOfBoundsException: {index}"
                )))),
                Some(input_arg) => Ok(input_arg.borrow().get_argument().map(str::to_owned)),
            }
        };
        let mut input_line: usize = 0;
        let value = argument(input_line)?;
        input_line += 1;
        self.base.extra_thickness.set_string(value.as_deref());
        let value = argument(input_line)?;
        input_line += 1;
        self.base.spacing_in_y = parse_double(value.as_deref())?;
        let value = argument(input_line)?;
        input_line += 1;
        let number_of_model_files = match value.as_deref() {
            None => {
                return Err(InvalidParameterException::new(Some(
                    "java.lang.NumberFormatException: null",
                ))
                .into());
            }
            Some(value) => java_lang_integer_parse_int(value).map_err(|message| {
                InvalidParameterException::new(Some(&format!(
                    "java.lang.NumberFormatException: {message}"
                )))
            })?,
        };
        for _ in 0..number_of_model_files {
            let value = argument(input_line)?;
            input_line += 1;
            self.base.model_files.push(value);
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // get the input arguments from the command
        let _input_args = self.get_input_arguments(script_command)?;
        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        param_utilities::update_script_parameter_strings_required(
            script_command,
            Some(MODEL_FILE),
            Some(&self.base.model_files),
            true,
        )?;
        param_utilities::update_script_parameter_double(
            script_command,
            Some(SPACING_IN_Y),
            self.base.spacing_in_y,
        );
        param_utilities::update_script_parameter_double(
            script_command,
            Some(SCALE_FACTOR),
            self.base.scale_factor,
        );
        param_utilities::update_script_parameter_string(
            script_command,
            Some(PARAMETER_FILE),
            self.base.parameter_file.as_deref(),
        )?;
        self.base.extra_thickness.update_com_script(script_command);
        self.base.no_x_axis_tilt.update_com_script(script_command);
        self.base.angle_offset_old.update_com_script(script_command);
        self.base.z_shift_old.update_com_script(script_command);
        self.base.x_axis_tilt_old.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
