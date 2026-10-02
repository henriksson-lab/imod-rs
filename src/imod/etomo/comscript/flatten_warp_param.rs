//! `IMOD/Etomo/src/etomo/comscript/FlattenWarpParam.java`.
//!
//! Represents the flattenwarp process interface.

use super::fortran_input_string::FortranInputString;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    Type, java_lang_double_value_of, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ONE_SURFACE_OPTION`.
pub const ONE_SURFACE_OPTION: &str = "OneSurface";
/// Java `WARP_SPACING_X_AND_Y_OPTION`.
pub const WARP_SPACING_X_AND_Y_OPTION: &str = "WarpSpacingXandY";
/// Java `LAMBDA_FOR_SMOOTHING_OPTION`.
pub const LAMBDA_FOR_SMOOTHING_OPTION: &str = "LambdaForSmoothing";
/// Java `LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT`.
pub const LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT: &str = "1,1.5,2,2.5,3";

/// Java final `FlattenWarpParam`.
pub struct FlattenWarpParam {
    /// Built on first use by `getCommandArray`, which the process manager
    /// calls through a shared reference.
    command: std::sync::Mutex<Vec<String>>,
    /// optional
    warp_spacing_x_and_y: FortranInputString,
    manager: &'static dyn BaseManager,
    /// optional
    lambda_for_smoothing: Option<FortranInputString>,
    one_surface: bool,
    middle_contour_file: Option<String>,
    criterion_for_outliers: ScriptParameter,
}

impl FlattenWarpParam {
    /// Java `FlattenWarpParam(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> FlattenWarpParam {
        let mut criterion_for_outliers = ScriptParameter::new_with_type(Some(Type::Double));
        criterion_for_outliers.set_display_value_double(3.0);
        FlattenWarpParam {
            command: std::sync::Mutex::new(Vec::new()),
            warp_spacing_x_and_y: FortranInputString::new(2),
            manager,
            lambda_for_smoothing: None,
            one_surface: false,
            middle_contour_file: None,
            criterion_for_outliers,
        }
    }

    /// Java private `buildCommand`.
    fn build_command(&self) {
        let mut command = self.command.lock().unwrap();
        // Java string concatenation writes a null path as "null"
        command.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_owned()),
            ProcessName::FLATTEN_WARP
        ));
        command.push("-PID".to_owned());
        command.push("-InputFile".to_owned());
        // FlattenWarpParam.java:98 adds the name even when it is null, and the
        // null element then makes ProcessBuilder throw a NullPointerException.
        // Fixed in translation: a null name is passed as an empty argument, so
        // flattenwarp reports the missing file itself.
        command.push(
            file_type::CLASS
                .flatten_warp_input_model
                .get_file_name(Some(self.manager), Some(AxisID::Only))
                .unwrap_or_default(),
        );
        command.push("-OutputFile".to_owned());
        command.push(dataset_files::get_flatten_warp_output_name(self.manager));
        if self.one_surface {
            command.push(format!("-{ONE_SURFACE_OPTION}"));
        }
        if !self.warp_spacing_x_and_y.is_null() {
            command.push(format!("-{WARP_SPACING_X_AND_Y_OPTION}"));
            command.push(self.warp_spacing_x_and_y.to_string_default_is_blank(true));
        }
        if let Some(middle_contour_file) = &self.middle_contour_file {
            command.push("-MiddleContourFile".to_owned());
            command.push(middle_contour_file.clone());
        }
        if let Some(lambda_for_smoothing) = &self.lambda_for_smoothing
            && !lambda_for_smoothing.is_null()
        {
            command.push(format!("-{LAMBDA_FOR_SMOOTHING_OPTION}"));
            command.push(lambda_for_smoothing.to_string_default_is_blank(true));
        }
        command.push("-CriterionForOutliers".to_owned());
        command.push(self.criterion_for_outliers.to_string());
    }

    /// Java `setOneSurface`.
    pub fn set_one_surface(&mut self, input: bool) {
        self.one_surface = input;
    }

    /// Java `setWarpSpacingX`.  Returns an error message if number is not a
    /// number and not blank, otherwise `None`.
    pub fn set_warp_spacing_x(&mut self, number: Option<&str>) -> Option<String> {
        // `try { warpSpacingXandY.set(0, number); } catch (NumberFormatException e)`:
        // `set(int, String)` converts a non-blank string with `Double.valueOf`, which
        // is what throws; the conversion is checked first so the exception's message
        // can be returned instead of propagated.
        if let Some(number) = number
            && !java_lang_string_matches_whitespace(number)
            && let Err(message) = java_lang_double_value_of(number)
        {
            return Some(message);
        }
        self.warp_spacing_x_and_y.set_index_string(0, number);
        None
    }

    /// Java `setWarpSpacingY`.  Returns an error message if number is not a
    /// number and not blank, otherwise `None`.
    pub fn set_warp_spacing_y(&mut self, number: Option<&str>) -> Option<String> {
        // See `set_warp_spacing_x`.
        if let Some(number) = number
            && !java_lang_string_matches_whitespace(number)
            && let Err(message) = java_lang_double_value_of(number)
        {
            return Some(message);
        }
        self.warp_spacing_x_and_y.set_index_string(1, number);
        None
    }

    /// Java `setLambdaForSmoothing`.  Returns an error message if number is not
    /// a number and not blank, otherwise `None`.
    pub fn set_lambda_for_smoothing(&mut self, input: Option<&str>) -> Option<String> {
        match FortranInputString::get_instance_from_list(input) {
            Ok(lambda_for_smoothing) => {
                self.lambda_for_smoothing = Some(lambda_for_smoothing);
            }
            Err(e) => {
                // Java `e.getMessage()`, which may be null.
                return Some(e.get_message().unwrap_or("null").to_owned());
            }
        }
        None
    }

    /// Java `setMiddleContourFile`.
    pub fn set_middle_contour_file(&mut self, input: Option<&str>) {
        self.middle_contour_file = input.map(str::to_owned);
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> ProcessName {
        ProcessName::FLATTEN_WARP
    }

    /// Java `getCommandArray`.  Echoes the command on standard error, as the
    /// source does.
    pub fn get_command_array(&self) -> Vec<String> {
        if self.command.lock().unwrap().is_empty() {
            self.build_command();
        }
        // FlattenWarpParam.java:168-170 builds a one-element array when the command
        // has one element and then immediately overwrites it with the general
        // conversion; only the general conversion is kept.
        let command_array = self.command.lock().unwrap().clone();
        for element in &command_array {
            eprint!("{element} ");
        }
        if !command_array.is_empty() {
            eprintln!();
        }
        command_array
    }
}
