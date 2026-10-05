//! `IMOD/Etomo/src/etomo/comscript/SolvematchParam.java`.
//!
//! `SolvematchParam extends ConstSolvematchParam`: the superclass state is the
//! `base` field, reached through `Deref`/`DerefMut`.

use std::sync::LazyLock;

use regex::Regex;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_solvematch_param::*;
use super::const_solvematchmod_param::ConstSolvematchmodParam;
use super::const_solvematchshift_param::ConstSolvematchshiftParam;
use super::fortran_input_string::FortranInputString;
use super::param_utilities;
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::util::dataset_files;

/// Java private static `CENTER_SHIFT_LIMIT_DEFAULT`.
const CENTER_SHIFT_LIMIT_DEFAULT: f64 = 10.0;

/// Java `"^\\s*\\S+?afid.xyz\\s*$"` with Java's `\s` (`[ \t\n\x0B\f\r]`) and `.`
/// (any character but a line terminator).
pub(crate) static AFID_XYZ: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\A(?:[ \t\n\x0B\x0C\r]*[^ \t\n\x0B\x0C\r]+?afid[^\n\r\x{85}\x{2028}\x{2029}]xyz[ \t\n\x0B\x0C\r]*)\z",
    )
    .unwrap()
});
/// Java `"^\\s*\\S+?bfid.xyz\\s*$"`.
pub(crate) static BFID_XYZ: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\A(?:[ \t\n\x0B\x0C\r]*[^ \t\n\x0B\x0C\r]+?bfid[^\n\r\x{85}\x{2028}\x{2029}]xyz[ \t\n\x0B\x0C\r]*)\z",
    )
    .unwrap()
});

/// Java `SolvematchParam`.
pub struct SolvematchParam {
    /// Java superclass `ConstSolvematchParam` state.
    pub base: ConstSolvematchParam,
    /// Java `manager`.
    manager: &'static dyn BaseManager,
}

impl std::ops::Deref for SolvematchParam {
    type Target = ConstSolvematchParam;

    fn deref(&self) -> &ConstSolvematchParam {
        &self.base
    }
}

impl std::ops::DerefMut for SolvematchParam {
    fn deref_mut(&mut self) -> &mut ConstSolvematchParam {
        &mut self.base
    }
}

impl SolvematchParam {
    /// Java `SolvematchParam(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> SolvematchParam {
        SolvematchParam {
            base: ConstSolvematchParam::new(),
            manager,
        }
    }

    /// Java `mergeSolvematchshift`.
    ///
    /// Java assigns the source's `StringList` and `FortranInputString` objects, so
    /// both params then share them; here they are copied.
    pub fn merge_solvematchshift(
        &mut self,
        solvematchshift: &ConstSolvematchshiftParam,
        model_based: bool,
    ) {
        if !model_based {
            self.base.to_fiducial_file = solvematchshift
                .get_to_fiducial_coordinates_file()
                .map(str::to_owned);
            let to_fiducial_file = self.base.to_fiducial_file.clone();
            self.set_match_b_to_a(to_fiducial_file.as_deref());
            self.base.from_fiducial_file = solvematchshift
                .get_from_fiducial_coordinates_file()
                .map(str::to_owned);
            self.base.to_correspondence_list = solvematchshift.get_fiducial_match_list_a().clone();
            self.base.from_correspondence_list =
                solvematchshift.get_fiducial_match_list_b().clone();
            self.base.x_axist_tilt = solvematchshift.get_x_axist_tilt().clone();
            self.base.surfaces_or_model = solvematchshift.get_n_surfaces();
        }
        self.base.output_file = solvematchshift
            .get_output_transformation_file()
            .map(str::to_owned);
        self.base.maximum_residual = solvematchshift.get_residual_threshold();
        // older version, so coordinate file would not exist
        self.base.transfer_coordinate_file = None;
        self.base
            .center_shift_limit
            .set_double(CENTER_SHIFT_LIMIT_DEFAULT);
    }

    /// Java `mergeSolvematchmod`.
    ///
    /// Java assigns the source's `StringList` and `FortranInputString` objects, so
    /// both params then share them; here they are copied.
    pub fn merge_solvematchmod(
        &mut self,
        solvematchmod: &ConstSolvematchmodParam,
        model_based: bool,
    ) {
        if model_based {
            self.base.to_fiducial_file = solvematchmod
                .get_to_fiducial_coordinates_file()
                .map(str::to_owned);
            let to_fiducial_file = self.base.to_fiducial_file.clone();
            self.set_match_b_to_a(to_fiducial_file.as_deref());
            self.base.from_fiducial_file = solvematchmod
                .get_from_fiducial_coordinates_file()
                .map(str::to_owned);
            self.base.to_correspondence_list = solvematchmod.get_fiducial_match_list_a().clone();
            self.base.from_correspondence_list = solvematchmod.get_fiducial_match_list_b().clone();
            self.base.x_axist_tilt = solvematchmod.get_x_axist_tilt().clone();
            self.base.surfaces_or_model = solvematchmod.get_n_surfaces();
        }
        self.base.to_matching_model = solvematchmod.get_to_matching_model().map(str::to_owned);
        self.base.from_matching_model = solvematchmod.get_from_matching_model().map(str::to_owned);
        self.base.to_tomogram_or_size_xyz = solvematchmod
            .get_to_reconstruction_file()
            .map(str::to_owned);
        self.base.from_tomogram_or_size_xyz = solvematchmod
            .get_from_reconstruction_file()
            .map(str::to_owned);
        // older version so coordinate file would not exist
        self.base.transfer_coordinate_file = None;
        self.base
            .center_shift_limit
            .set_double(CENTER_SHIFT_LIMIT_DEFAULT);
    }

    /// Java package-private `setMatchBToA(String)`.
    pub(crate) fn set_match_b_to_a(&mut self, a_fiducial_filename: Option<&str>) {
        let a_fiducial_filename = match a_fiducial_filename {
            None => return,
            Some(name) if java_lang_string_matches_whitespace(name) => return,
            Some(name) => name,
        };
        self.base.a_fiducial_model = None;
        self.base.b_fiducial_model = None;
        if AFID_XYZ.is_match(a_fiducial_filename) {
            self.base.match_b_to_a = true;
            self.base.a_fiducial_model = Some(dataset_files::get_fiducial_model_name(
                self.manager,
                Some(AxisID::First),
            ));
            self.base.b_fiducial_model = Some(dataset_files::get_fiducial_model_name(
                self.manager,
                Some(AxisID::Second),
            ));
        } else if BFID_XYZ.is_match(a_fiducial_filename) {
            self.base.match_b_to_a = false;
            self.base.b_fiducial_model = Some(dataset_files::get_fiducial_model_name(
                self.manager,
                Some(AxisID::First),
            ));
            self.base.a_fiducial_model = Some(dataset_files::get_fiducial_model_name(
                self.manager,
                Some(AxisID::Second),
            ));
        }
    }

    /// Java `setUsePoints`.
    pub fn set_use_points(&mut self, use_points: Option<&str>) {
        self.base.use_points.parse_string(use_points);
    }

    /// Java `setFromCorrespondenceList`.
    pub fn set_from_correspondence_list(&mut self, string: Option<&str>) {
        self.base.from_correspondence_list.parse_string(string);
    }

    /// Java `setTransferCoordinateFile`.
    pub fn set_transfer_coordinate_file(&mut self, use_correspondence_list: bool) {
        if use_correspondence_list {
            self.base.transfer_coordinate_file = None;
        } else {
            self.base.transfer_coordinate_file =
                Some(dataset_files::get_transfer_fid_coord_file_name());
        }
    }

    /// Java `setFromFiducialFile`.
    pub fn set_from_fiducial_file(&mut self, from_fiducial_file: Option<&str>) {
        self.base.from_fiducial_file = from_fiducial_file.map(str::to_owned);
    }

    /// Java `setFromMatchingModel`.
    pub fn set_from_matching_model(&mut self, from_matching_model: Option<&str>) {
        self.base.from_matching_model = from_matching_model.map(str::to_owned);
    }

    /// Java `setFromTomogramOrSizeXYZ`.
    pub fn set_from_tomogram_or_size_xyz(&mut self, from_tomogram_or_size_xyz: Option<&str>) {
        self.base.from_tomogram_or_size_xyz = from_tomogram_or_size_xyz.map(str::to_owned);
    }

    /// Java `setMaximumResidual(double)`.
    pub fn set_maximum_residual_double(&mut self, maximum_residual: f64) {
        self.base.maximum_residual = maximum_residual;
    }

    /// Java `setMaximumResidual(String)`.  `Err` carries the unchecked
    /// NumberFormatException's message.
    pub fn set_maximum_residual_string(&mut self, value: Option<&str>) -> Result<(), String> {
        self.base.maximum_residual = param_utilities::parse_double(value)?;
        Ok(())
    }

    /// Java `setCenterShiftLimit`.
    pub fn set_center_shift_limit(&mut self, center_shift_limit: Option<&str>) {
        self.base.center_shift_limit.set_string(center_shift_limit);
    }

    /// Java `setSurfacesOrModel`.
    pub fn set_surfaces_or_model(&mut self, value: FiducialMatch) {
        if value == FiducialMatch::UseModelOnly {
            self.base.surfaces_or_model = USE_MODEL_ONLY_OPTION;
            return;
        }
        if value == FiducialMatch::OneSideInverted {
            self.base.surfaces_or_model = ONE_SIDE_INVERTED_OPTION;
            return;
        }
        if value == FiducialMatch::UseModel {
            self.base.surfaces_or_model = USE_MODEL_OPTION;
            return;
        }
        if value == FiducialMatch::OneSide {
            self.base.surfaces_or_model = ONE_SIDE_OPTION;
            return;
        }
        if value == FiducialMatch::BothSides {
            self.base.surfaces_or_model = BOTH_SIDES_OPTION;
        }
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, output_transformation_file: Option<&str>) {
        self.base.output_file = output_transformation_file.map(str::to_owned);
    }

    /// Java `setScaleFactors`.
    pub fn set_scale_factors(&mut self, scale_factors: FortranInputString) {
        self.base.scale_factors = scale_factors;
    }

    /// Java `setToCorrespondenceList`.
    pub fn set_to_correspondence_list(&mut self, string: Option<&str>) {
        self.base.to_correspondence_list.parse_string(string);
    }

    /// Java `setToFiducialFile`.
    pub fn set_to_fiducial_file(&mut self, to_fiducial_file: Option<&str>) {
        self.base.to_fiducial_file = to_fiducial_file.map(str::to_owned);
    }

    /// Java `setToMatchingModel`.
    pub fn set_to_matching_model(&mut self, to_matching_model: Option<&str>) {
        self.base.to_matching_model = to_matching_model.map(str::to_owned);
    }

    /// Java `setToTomogramOrSizeXYZ`.
    pub fn set_to_tomogram_or_size_xyz(&mut self, to_tomogram_or_size_xyz: Option<&str>) {
        self.base.to_tomogram_or_size_xyz = to_tomogram_or_size_xyz.map(str::to_owned);
    }

    /// Java `setXAxistTilt`.
    pub fn set_x_axist_tilt(&mut self, axist_tilt: FortranInputString) {
        self.base.x_axist_tilt = axist_tilt;
    }
}

impl CommandParam for SolvematchParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}

    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Check to be sure that it is a solvematch command.  (A null command is a
        // NullPointerException in Java; here it is not a solvematch command.)
        if script_command.get_command() != Some("solvematch") {
            return Err(BadComScriptException::new("Not a solvematch command").into());
        }
        // Set any of this class's attributes that are present in the script command
        // object
        let base = &mut self.base;
        base.output_file = param_utilities::set_param_if_present_string(
            script_command,
            OUTPUT_FILE,
            base.output_file.as_deref(),
        )?;
        base.to_fiducial_file = param_utilities::set_param_if_present_string(
            script_command,
            TO_FIDUCIAL_FILE,
            base.to_fiducial_file.as_deref(),
        )?;
        base.from_fiducial_file = param_utilities::set_param_if_present_string(
            script_command,
            FROM_FIDUCIAL_FILE,
            base.from_fiducial_file.as_deref(),
        )?;
        base.to_correspondence_list = param_utilities::set_param_if_present_string_list(
            script_command,
            TO_CORRESPONDENCE_LIST,
            Some(base.to_correspondence_list.clone()),
        )?
        .unwrap_or_else(StringList::new);
        base.from_correspondence_list = param_utilities::set_param_if_present_string_list(
            script_command,
            FROM_CORRESPONDENCE_LIST,
            Some(base.from_correspondence_list.clone()),
        )?
        .unwrap_or_else(StringList::new);
        // SolvematchParam.java:103-106 call the String overload of
        // setParamIfPresent for these three and discard the returned value, so the
        // fields are never read from the script and `aFiducialModel` is always null
        // below: the matching direction comes from `setMatchBToA(toFiducialFile)`.
        // Kept native (BUGS.md, eTomo "SolvematchParam"): assigning the values
        // instead routes every current script into the `MatchingAtoB` branch,
        // whose presence-only boolean test reads setupcombine's
        // `MatchingAtoB 0` as A-to-B and so rewrites solvematch.com with the
        // matching direction reversed.
        let _ = param_utilities::set_param_if_present_string(
            script_command,
            TRANSFER_COORDINATE_FILE,
            base.transfer_coordinate_file.as_deref(),
        )?;
        let _ = param_utilities::set_param_if_present_string(
            script_command,
            A_FIDUCIAL_MODEL,
            base.a_fiducial_model.as_deref(),
        )?;
        let _ = param_utilities::set_param_if_present_string(
            script_command,
            B_FIDUCIAL_MODEL,
            base.b_fiducial_model.as_deref(),
        )?;
        base.use_points = param_utilities::set_param_if_present_string_list(
            script_command,
            USE_POINTS,
            Some(base.use_points.clone()),
        )?
        .unwrap_or_else(StringList::new);
        param_utilities::set_param_if_present_fortran_input_string(
            script_command,
            XAXIS_TILTS,
            &mut base.x_axist_tilt,
        )?;
        base.surfaces_or_model = param_utilities::set_param_if_present_int(
            script_command,
            SURFACE_OR_USE_MODELS,
            base.surfaces_or_model,
        )?;
        base.maximum_residual = param_utilities::set_param_if_present_double(
            script_command,
            MAXIMUM_RESIDUAL,
            base.maximum_residual,
        )?;
        base.center_shift_limit.parse(script_command)?;
        base.to_matching_model = param_utilities::set_param_if_present_string(
            script_command,
            TO_MATCHING_MODEL,
            base.to_matching_model.as_deref(),
        )?;
        base.from_matching_model = param_utilities::set_param_if_present_string(
            script_command,
            FROM_MATCHING_MODEL,
            base.from_matching_model.as_deref(),
        )?;
        base.to_tomogram_or_size_xyz = param_utilities::set_param_if_present_string(
            script_command,
            TO_TOMOGRAM_OR_SIZE_XYZ,
            base.to_tomogram_or_size_xyz.as_deref(),
        )?;
        base.from_tomogram_or_size_xyz = param_utilities::set_param_if_present_string(
            script_command,
            FROM_TOMOGRAM_OR_SIZE_XYZ,
            base.from_tomogram_or_size_xyz.as_deref(),
        )?;
        // set matchBToA
        // fiducial model parameters can be added regardless of the transfer
        // coordinate file mode, so they can be used to check the version of the
        // script
        if self.base.a_fiducial_model.is_none() {
            // backwards compatibility - set transferCoordinateFile, aFiducialModel,
            // bFiducialModel
            // Set the matching state based on the toFiducialFile name
            let to_fiducial_file = self.base.to_fiducial_file.clone();
            self.set_match_b_to_a(to_fiducial_file.as_deref());
            self.base.transfer_coordinate_file = None;
        } else {
            // scripts contains a parameter for A to B, not B to A
            self.base.match_b_to_a = !param_utilities::set_param_if_present_boolean(
                script_command,
                MATCHING_A_TO_B,
                !self.base.match_b_to_a,
            )?;
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        if script_command.get_command() != Some("solvematch") {
            return Err(BadComScriptException::new("Not a solvematch command"));
        }
        // Make sure the script is in keyword / value pairs
        script_command.use_keyword_value();
        // Update the values in the comscript command object
        let base = &self.base;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_FILE),
            base.output_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(TO_FIDUCIAL_FILE),
            base.to_fiducial_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(FROM_FIDUCIAL_FILE),
            base.from_fiducial_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(TO_CORRESPONDENCE_LIST),
            Some(&base.to_correspondence_list.to_string()),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(FROM_CORRESPONDENCE_LIST),
            Some(&base.from_correspondence_list.to_string()),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(TRANSFER_COORDINATE_FILE),
            base.transfer_coordinate_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(A_FIDUCIAL_MODEL),
            base.a_fiducial_model.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(B_FIDUCIAL_MODEL),
            base.b_fiducial_model.as_deref(),
        )?;
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some(MATCHING_A_TO_B),
            !base.match_b_to_a,
        );
        param_utilities::update_script_parameter_string_list(
            script_command,
            Some(USE_POINTS),
            Some(&base.use_points),
        )?;
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(XAXIS_TILTS),
            &base.x_axist_tilt,
        );
        param_utilities::update_script_parameter_int(
            script_command,
            Some(SURFACE_OR_USE_MODELS),
            base.surfaces_or_model,
        );
        param_utilities::update_script_parameter_double(
            script_command,
            Some(MAXIMUM_RESIDUAL),
            base.maximum_residual,
        );
        base.center_shift_limit.update_com_script(script_command);
        param_utilities::update_script_parameter_string(
            script_command,
            Some(TO_MATCHING_MODEL),
            base.to_matching_model.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(FROM_MATCHING_MODEL),
            base.from_matching_model.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(TO_TOMOGRAM_OR_SIZE_XYZ),
            base.to_tomogram_or_size_xyz.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(FROM_TOMOGRAM_OR_SIZE_XYZ),
            base.from_tomogram_or_size_xyz.as_deref(),
        )?;
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(SCALE_FACTORS),
            &base.scale_factors,
        );
        Ok(())
    }
}
