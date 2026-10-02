//! `IMOD/Etomo/src/etomo/comscript/OldBeadtrackParam.java`.
//!
//! Was BeadtrackParam.  The pre-PIP beadtrack command, which reads and writes
//! the sequential standard-input form of `track.com`.
//!
//! Java's `OldBeadtrackParam extends OldConstBeadtrackParam`, and
//! `BeadtrackParam extends OldBeadtrackParam`.  Rust has no inheritance, so -
//! as `etomo/type/script_parameter.rs` does for its own superclass - the
//! superclass state is held in the `base` field and reached through
//! `Deref`/`DerefMut`; every inherited member is therefore callable on the
//! subclass exactly as in Java.

use std::cell::RefCell;
use std::rc::Rc;

use regex::Regex;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command_param::{CommandParam, ParseComScriptError};
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::old_const_beadtrack_param::{
    NONDEFAULT_GROUP_INTEGER_TYPE, NONDEFAULT_GROUP_SIZE, OldConstBeadtrackParam,
};
use super::param_utilities;
use super::string_list::StringList;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_double_value_of, java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private class `OldBeadtrackParam`.
#[derive(Clone, Debug)]
pub struct OldBeadtrackParam {
    /// Java superclass `OldConstBeadtrackParam` state.
    pub base: OldConstBeadtrackParam,
}

/// Java inheritance: every `OldConstBeadtrackParam` member is reachable on an
/// `OldBeadtrackParam`.
impl std::ops::Deref for OldBeadtrackParam {
    type Target = OldConstBeadtrackParam;

    fn deref(&self) -> &OldConstBeadtrackParam {
        &self.base
    }
}

impl std::ops::DerefMut for OldBeadtrackParam {
    fn deref_mut(&mut self) -> &mut OldConstBeadtrackParam {
        &mut self.base
    }
}

impl OldBeadtrackParam {
    /// Java implicit `OldBeadtrackParam()`.
    pub(crate) fn new() -> OldBeadtrackParam {
        OldBeadtrackParam {
            base: OldConstBeadtrackParam::new(),
        }
    }

    /// Java `parseComScriptCommand`.  Get the parameters from the
    /// ComScriptCommand.
    ///
    /// OldBeadtrackParam.java:43-151 indexes `inputArgs[inputLine++]` past the
    /// 26 arguments `getInputArguments` guarantees whenever the view-set or group
    /// counts ask for more (ArrayIndexOutOfBoundsException), passes a null
    /// argument to `parseInt`/`parseDouble` (NullPointerException), and lets
    /// the unchecked NumberFormatException of a non-numeric argument escape.
    /// All three crash the caller.  Here each is a `BadComScriptException`
    /// naming the argument, which is the checked exception the method already
    /// uses for a malformed command.
    pub(crate) fn parse_old_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // get the input arguments from the command
        let input_args = self.get_input_arguments(script_command)?;

        // `inputArgs[inputLine].getArgument()`, with the out-of-range index
        // reported instead of thrown.
        let argument = |input_line: usize| -> Result<Option<String>, BadComScriptException> {
            match input_args.get(input_line) {
                None => Err(BadComScriptException::new(&format!(
                    "Incorrect number of input arguments to beadtrack command\nGot {} expected more than {}.",
                    input_args.len(),
                    input_line
                ))),
                Some(input_arg) => Ok(input_arg.borrow().get_argument().map(str::to_owned)),
            }
        };
        let parse_int = |input_line: usize,
                         value: Option<String>|
         -> Result<i32, BadComScriptException> {
            match value {
                None => Err(BadComScriptException::new(&format!(
                    "Missing integer in beadtrack command, standard input argument: {input_line}"
                ))),
                Some(value) => java_lang_integer_parse_int(&value).map_err(|message| {
                    BadComScriptException::new(&format!(
                        "Parse error in beadtrack command, standard input argument: {input_line}\n{message}"
                    ))
                }),
            }
        };
        let parse_double = |input_line: usize,
                            value: Option<String>|
         -> Result<f64, BadComScriptException> {
            match value {
                None => Err(BadComScriptException::new(&format!(
                    "Missing number in beadtrack command, standard input argument: {input_line}"
                ))),
                Some(value) => java_lang_double_value_of(&value).map_err(|message| {
                    BadComScriptException::new(&format!(
                        "Parse error in beadtrack command, standard input argument: {input_line}\n{message}"
                    ))
                }),
            }
        };

        let mut input_line: usize = 0;
        self.input_file = argument(input_line)?;
        input_line += 1;
        self.piece_list_file = argument(input_line)?;
        input_line += 1;
        self.seed_model_file = argument(input_line)?;
        input_line += 1;
        self.output_model_file = argument(input_line)?;
        input_line += 1;
        self.view_skip_list = argument(input_line)?;
        input_line += 1;
        self.image_rotation = parse_double(input_line, argument(input_line)?)?;
        input_line += 1;

        self.n_additional_view_sets = parse_int(input_line, argument(input_line)?)?;
        input_line += 1;
        if self.n_additional_view_sets > 0 {
            self.additional_view_groups =
                StringList::new_with_n_elements(self.n_additional_view_sets);
            for i in 0..self.n_additional_view_sets {
                let value = argument(input_line)?;
                input_line += 1;
                self.additional_view_groups.set(i, value.as_deref());
            }
        }

        let type_spec = parse_int(input_line, argument(input_line)?)?;
        input_line += 1;
        // Java `tiltAngleSpec.setType(TiltAngleType.parseInt(typeSpec))` stores null for
        // an unknown type, which then falls to the error branch below.  The Rust spec
        // holds no null type, so an unknown type is not stored; the branch is taken from
        // the parsed value.
        let tilt_angle_type = TiltAngleType::parse_int(type_spec);
        if let Some(tilt_angle_type) = tilt_angle_type {
            self.tilt_angle_spec.set_type(tilt_angle_type);
        }
        if tilt_angle_type == Some(TiltAngleType::File) {
            let value = argument(input_line)?;
            input_line += 1;
            self.tilt_angle_spec
                .set_tilt_angle_filename(value.as_deref().unwrap_or(""));
        } else if tilt_angle_type == Some(TiltAngleType::Range) {
            let pair = argument(input_line)?;
            input_line += 1;
            // `pair.split(",")` on a null argument is a NullPointerException in
            // Java; an absent pair is the same malformed specification here.
            let values = match &pair {
                None => Vec::new(),
                Some(pair) => java_lang_string_split(pair, &Regex::new(",").unwrap()),
            };
            if values.len() != 2 {
                return Err(
                    BadComScriptException::new("Incorrect tilt angle specification type").into(),
                );
            }
            self.tilt_angle_spec
                .set_range_min_double(parse_double(input_line - 1, Some(values[0].clone()))?);
            self.tilt_angle_spec
                .set_range_step_double(parse_double(input_line - 1, Some(values[1].clone()))?);
        } else if self.tilt_angle_spec.get_type() == TiltAngleType::List {
            return Err(
                BadComScriptException::new("Unimplemented tilt angle specification type").into(),
            );
        } else {
            return Err(
                BadComScriptException::new("Incorrect tilt angle specification type").into(),
            );
        }

        // The Java try block: a FortranInputSyntaxException from any of the
        // validateAndSet calls is rethrown with the argument number prepended.
        let result = (|| -> Result<(), ParseComScriptError> {
            let value = argument(input_line)?;
            input_line += 1;
            self.tilt_angle_group_params
                .validate_and_set(value.as_deref())?;
            let mut n_groups = self.tilt_angle_group_params.get_int(1);
            if n_groups > 0 {
                let mut tilt_angle_groups_list = StringList::new_with_n_elements(n_groups);
                for i in 0..n_groups {
                    let value = argument(input_line)?;
                    input_line += 1;
                    tilt_angle_groups_list.set(i, value.as_deref());
                }
                self.tilt_angle_groups = param_utilities::parse_string_list(
                    Some(&tilt_angle_groups_list),
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                    NONDEFAULT_GROUP_SIZE,
                )?;
            }
            let value = argument(input_line)?;
            input_line += 1;
            self.magnification_group_params
                .validate_and_set(value.as_deref())?;
            n_groups = self.magnification_group_params.get_int(1);
            if n_groups > 0 {
                let mut magnification_groups_list = StringList::new_with_n_elements(n_groups);
                for i in 0..n_groups {
                    let value = argument(input_line)?;
                    input_line += 1;
                    magnification_groups_list.set(i, value.as_deref());
                }
                self.magnification_groups = param_utilities::parse_string_list(
                    Some(&magnification_groups_list),
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                    NONDEFAULT_GROUP_SIZE,
                )?;
            }
            self.n_min_views = parse_int(input_line, argument(input_line)?)?;
            input_line += 1;
            let value = argument(input_line)?;
            input_line += 1;
            self.fiducial_params.validate_and_set(value.as_deref())?;
            let fill_gaps = parse_int(input_line, argument(input_line)?)?;
            input_line += 1;
            if fill_gaps > 0 {
                self.fill_gaps = true;
            } else {
                self.fill_gaps = false;
            }
            self.max_gap = parse_int(input_line, argument(input_line)?)?;
            input_line += 1;
            let value = argument(input_line)?;
            input_line += 1;
            self.tilt_angle_min_range
                .validate_and_set(value.as_deref())?;
            let value = argument(input_line)?;
            input_line += 1;
            self.search_box_pixels.validate_and_set(value.as_deref())?;
            self.max_fiducials_avg = parse_int(input_line, argument(input_line)?)?;
            input_line += 1;
            let value = argument(input_line)?;
            input_line += 1;
            self.fiducial_extrapolation_params
                .validate_and_set(value.as_deref())?;
            let value = argument(input_line)?;
            input_line += 1;
            self.rescue_attempt_params
                .validate_and_set(value.as_deref())?;
            self.min_rescue_distance = parse_int(input_line, argument(input_line)?)?;
            input_line += 1;
            let value = argument(input_line)?;
            input_line += 1;
            self.rescue_relaxation_params
                .validate_and_set(value.as_deref())?;
            self.residual_distance_limit = parse_double(input_line, argument(input_line)?)?;
            input_line += 1;
            let value = argument(input_line)?;
            input_line += 1;
            self.second_pass_params.validate_and_set(value.as_deref())?;
            let value = argument(input_line)?;
            input_line += 1;
            self.mean_resid_change_limits
                .validate_and_set(value.as_deref())?;
            let value = argument(input_line)?;
            input_line += 1;
            self.deletion_params.validate_and_set(value.as_deref())?;
            Ok(())
        })();
        match result {
            Err(ParseComScriptError::FortranInputSyntax(except)) => {
                // `inputLine` has already been incremented past the failing
                // argument, as in Java.
                let message = format!(
                    "Parse error in beadtrack command, standard input argument: {}\n{}",
                    input_line,
                    except.get_message().unwrap_or("null")
                );
                Err(FortranInputSyntaxException::new_with_new_values(
                    &message,
                    except.get_new_string(),
                )
                .into())
            }
            other => other,
        }
    }

    /// Java `initializeDefaults`.
    pub(crate) fn initialize_old_defaults(&mut self) {
        self.tilt_angle_groups = None;
        self.magnification_groups = None;
    }

    /// Java `updateComScriptCommand` (deprecated).  Update the supplied
    /// ComScriptCommand with the parameters of this object.
    pub(crate) fn update_old_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Get the existing input arguments from the command, this is so the
        // comments are preserved.
        let mut input_args = self.get_input_arguments(script_command)?;

        input_args = self.update_input_args(input_args)?;
        script_command.set_input_arguments(&input_args);
        Ok(())
    }

    /// Java `setInputFile`.
    pub(crate) fn set_input_file(&mut self, input_file: Option<&str>) {
        self.input_file = input_file.map(str::to_owned);
    }

    /// Java `setPieceListFile`.
    pub(crate) fn set_piece_list_file(&mut self, piece_list_file: Option<&str>) {
        self.piece_list_file = piece_list_file.map(str::to_owned);
    }

    /// Java `setSeedModelFile`.
    pub(crate) fn set_seed_model_file(&mut self, seed_model_file: Option<&str>) {
        self.seed_model_file = seed_model_file.map(str::to_owned);
    }

    /// Java `setOutputModelFile`.
    pub(crate) fn set_output_model_file(&mut self, output_model_file: Option<&str>) {
        self.output_model_file = output_model_file.map(str::to_owned);
    }

    /// Java `setViewSkipList` (deprecated).
    pub(crate) fn set_view_skip_list(&mut self, view_skip_list: Option<&str>) {
        self.view_skip_list = view_skip_list.map(str::to_owned);
    }

    /// Java `setImageRotation` (deprecated).
    pub(crate) fn set_image_rotation(&mut self, image_rotation: f64) {
        self.image_rotation = image_rotation;
    }

    /// Java public `setAdditionalViewGroups`.
    pub fn set_additional_view_groups(&mut self, new_additional_view_groups: Option<&str>) {
        self.additional_view_groups
            .parse_string(new_additional_view_groups);
        self.n_additional_view_sets = self.additional_view_groups.get_n_elements();
    }

    /// Java `setTiltAngleGroupSize` (deprecated).
    pub(crate) fn set_tilt_angle_group_size(&mut self, group_size: i32) {
        self.tilt_angle_group_params
            .set_index_double(0, group_size as f64);
    }

    /// Java `setTiltAngleGroups(String)` (deprecated).  `BeadtrackParam`
    /// overrides it; see `BeadtrackParam::set_tilt_angle_groups`.
    pub(crate) fn set_old_tilt_angle_groups(
        &mut self,
        new_tilt_angle_groups: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.tilt_angle_groups =
            param_utilities::parse_string(new_tilt_angle_groups, true, NONDEFAULT_GROUP_SIZE)?;
        match &self.base.tilt_angle_groups {
            None => self.base.tilt_angle_group_params.set_default_index(1),
            Some(tilt_angle_groups) => {
                let length = tilt_angle_groups.len();
                self.base
                    .tilt_angle_group_params
                    .set_index_double(1, length as f64)
            }
        }
        Ok(())
    }

    /// Java `setMagnificationGroupSize` (deprecated).
    pub(crate) fn set_magnification_group_size(&mut self, group_size: i32) {
        self.magnification_group_params
            .set_index_double(0, group_size as f64);
    }

    /// Java `setMagnificationGroups(String)` (deprecated).  `BeadtrackParam`
    /// overrides it; see `BeadtrackParam::set_magnification_groups`.
    ///
    /// OldBeadtrackParam.java:230 tests `tiltAngleGroups == null` where it means
    /// `magnificationGroups` (a copy-paste typo from `setTiltAngleGroups`): with
    /// tilt groups set and no magnification groups it dereferences the null
    /// `magnificationGroups.length`, and otherwise it sets the magnification
    /// group count from the wrong array's nullity.  Fixed in translation: the
    /// test is on `magnificationGroups`.
    pub(crate) fn set_old_magnification_groups(
        &mut self,
        new_magnification_groups: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.magnification_groups =
            param_utilities::parse_string(new_magnification_groups, true, NONDEFAULT_GROUP_SIZE)?;
        match &self.base.magnification_groups {
            None => self.base.magnification_group_params.set_default_index(1),
            Some(magnification_groups) => {
                let length = magnification_groups.len();
                self.base
                    .magnification_group_params
                    .set_index_double(1, length as f64)
            }
        }
        Ok(())
    }

    /// Java `setNMinViews` (deprecated).
    pub(crate) fn set_n_min_views(&mut self, n_min_views: i32) {
        self.n_min_views = n_min_views;
    }

    /// Java `setFiducialParams` (deprecated).
    pub(crate) fn set_fiducial_params(
        &mut self,
        fiducial_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.fiducial_params.validate_and_set(fiducial_params)
    }

    /// Java public `setFillGaps`.
    pub fn set_fill_gaps(&mut self, fill_gaps: bool) {
        self.fill_gaps = fill_gaps;
    }

    /// Java `setMaxGap` (deprecated).
    pub(crate) fn set_max_gap(&mut self, max_gap: i32) {
        self.max_gap = max_gap;
    }

    /// Java `setTiltAngleMinRange` (deprecated).
    pub(crate) fn set_tilt_angle_min_range(
        &mut self,
        tilt_angle_min_range: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.tilt_angle_min_range
            .validate_and_set(tilt_angle_min_range)
    }

    /// Java public `setSearchBoxPixels`.
    pub fn set_search_box_pixels(
        &mut self,
        search_box_pixels: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.search_box_pixels.validate_and_set(search_box_pixels)
    }

    /// Java `setMaxFiducialsAvg` (deprecated).
    pub(crate) fn set_max_fiducials_avg(&mut self, max_fiducials_avg: i32) {
        self.max_fiducials_avg = max_fiducials_avg;
    }

    /// Java public `setFiducialExtrapolationParams`.
    pub fn set_fiducial_extrapolation_params(
        &mut self,
        fiducial_extrapolation_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.fiducial_extrapolation_params
            .validate_and_set(fiducial_extrapolation_params)
    }

    /// Java public `setRescueAttemptParams`.
    pub fn set_rescue_attempt_params(
        &mut self,
        rescue_attempt_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.rescue_attempt_params
            .validate_and_set(rescue_attempt_params)
    }

    /// Java `setMinRescueDistance` (deprecated).
    pub(crate) fn set_min_rescue_distance(&mut self, min_rescue_distance: i32) {
        self.min_rescue_distance = min_rescue_distance;
    }

    /// Java public `setRescueRelaxationParams`.
    pub fn set_rescue_relaxation_params(
        &mut self,
        rescue_relaxation_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.rescue_relaxation_params
            .validate_and_set(rescue_relaxation_params)
    }

    /// Java `setResidualDistanceLimit` (deprecated).
    pub(crate) fn set_residual_distance_limit(&mut self, residual_distance_limit: f64) {
        self.residual_distance_limit = residual_distance_limit;
    }

    /// Java `setSecondPassParams` (deprecated).
    pub(crate) fn set_second_pass_params(
        &mut self,
        second_pass_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.second_pass_params.validate_and_set(second_pass_params)
    }

    /// Java public `setMeanResidChangeLimits`.
    pub fn set_mean_resid_change_limits(
        &mut self,
        mean_resid_change_limits: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.mean_resid_change_limits
            .validate_and_set(mean_resid_change_limits)
    }

    /// Java public `setDeletionParams`.
    pub fn set_deletion_params(
        &mut self,
        deletion_params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.deletion_params.validate_and_set(deletion_params)
    }

    /// Java private `getInputArguments`.  Get the standand input arguments from
    /// the ComScriptCommand validating the name of the command and the
    /// appropriate number of input arguments.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        // Check to be sure that it is a beadtrack command
        // `scriptCommand.getCommand().equals(...)` throws on a null command;
        // a null command is "Not a beadtrack command" here.
        if script_command.get_command() != Some("beadtrack") {
            return Err(BadComScriptException::new("Not a beadtrack command"));
        }

        // Extract the parameters
        let input_args = script_command.get_input_arguments();
        if input_args.len() < 26 {
            return Err(BadComScriptException::new(&format!(
                "Incorrect number of input arguments to beadtrack command\nGot {} expected at least 26.",
                input_args.len()
            )));
        }

        Ok(input_args)
    }

    /// Java private `updateInputArgs` (deprecated).  Update the inputArguments
    /// array with new parameters.  Returns a newly allocated array since the
    /// number of elements may be different than the input parameter.
    ///
    /// OldBeadtrackParam.java:397-548 indexes `inputArgs[srcListCount++]` past
    /// the end whenever the old command's group counts skip beyond it, and
    /// dereferences `tiltAngleGroups[i]`/`magnificationGroups[i]` for as many
    /// groups as the parameter count says, whether or not the array exists or
    /// is that long.  Both crash.  Here an index past the arguments is a
    /// `BadComScriptException` (the method's caller already declares it), and
    /// a missing group writes no argument for that group.
    fn update_input_args(
        &self,
        input_args: Vec<Rc<RefCell<ComScriptInputArg>>>,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        let mut input_arg_list: Vec<Rc<RefCell<ComScriptInputArg>>> = Vec::new();
        let at = |index: usize| -> Result<Rc<RefCell<ComScriptInputArg>>, BadComScriptException> {
            match input_args.get(index) {
                None => Err(BadComScriptException::new(&format!(
                    "Incorrect number of input arguments to beadtrack command\nGot {} expected more than {}.",
                    input_args.len(),
                    index
                ))),
                Some(input_arg) => Ok(Rc::clone(input_arg)),
            }
        };

        // Fill in the input argument sequence
        let mut src_list_count: usize = 0;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(self.input_file.as_deref());
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(self.piece_list_file.as_deref());
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(self.seed_model_file.as_deref());
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(self.output_model_file.as_deref());
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(self.view_skip_list.as_deref());
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        // tilt axis angle of rotation parameter [5],
        // this is a system/mag/camera variable
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        // Increment the source list counter to skip the old view groups, the
        // comments for those entries are most likely not applicable
        let argument = at(src_list_count)?
            .borrow()
            .get_argument()
            .map(str::to_owned);
        // `Integer.parseInt` of a null or non-numeric argument throws; that is
        // the same malformed command here.
        let mut n_src_sets = match argument.as_deref().map(java_lang_integer_parse_int) {
            Some(Ok(n_src_sets)) => n_src_sets,
            _ => {
                return Err(BadComScriptException::new(&format!(
                    "Parse error in beadtrack command, standard input argument: {src_list_count}"
                )));
            }
        };
        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.n_additional_view_sets.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;
        src_list_count = (src_list_count as i64 + n_src_sets as i64) as usize;
        for i in 0..self.n_additional_view_sets {
            let mut view_set = ComScriptInputArg::new();
            view_set.set_argument(self.additional_view_groups.get(i).as_deref());
            input_arg_list.push(Rc::new(RefCell::new(view_set)));
        }

        // Tilt angle source and filenames/ranges are not modified by this class
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        // Tilt angle groups
        // The number of non-standard tilt angle sets is the second parameter
        // in the comma separated list.
        let mut fis = FortranInputString::new(2);
        fis.set_integer_type_index(0, true);
        fis.set_integer_type_index(1, true);
        n_src_sets = 0;
        let argument = at(src_list_count)?
            .borrow()
            .get_argument()
            .map(str::to_owned);
        match fis.validate_and_set(argument.as_deref()) {
            Ok(()) => n_src_sets = fis.get_int(1),
            Err(except) => {
                // TODO throw exception so calling object can catch and display a
                // message
                let error_message = [
                    "BeadtrackParam Error".to_owned(),
                    "Existing beadtrack tilt angle parameter was incorrect".to_owned(),
                    "Don't know how many non-default tilt angle groups, assuming 0".to_owned(),
                    format!("Input string: {}", argument.as_deref().unwrap_or("null")),
                    except.get_message().unwrap_or("null").to_owned(),
                ];
                // `JOptionPane.showMessageDialog(null, errorMessage, ...,
                // ERROR_MESSAGE)`: one line per array element.
                ui_harness::post_message_dialog(
                    None,
                    error_message.join("\n"),
                    "BeadtrackParam Error".to_owned(),
                    None,
                );
            }
        }

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.tilt_angle_group_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        src_list_count = (src_list_count as i64 + n_src_sets as i64) as usize;
        for i in 0..self.tilt_angle_group_params.get_int(1) {
            if let Some(group) = self
                .tilt_angle_groups
                .as_ref()
                .and_then(|groups| groups.get(i as usize))
            {
                let mut tilt_angle_group = ComScriptInputArg::new();
                tilt_angle_group.set_argument(Some(&group.to_string()));
                input_arg_list.push(Rc::new(RefCell::new(tilt_angle_group)));
            }
        }

        // Magnification groups
        // The number of non-standard magnification sets is the second parameter
        // in the comma separated list.
        fis.set_integer_type_index(0, true);
        fis.set_integer_type_index(1, true);
        n_src_sets = 0;
        let argument = at(src_list_count)?
            .borrow()
            .get_argument()
            .map(str::to_owned);
        match fis.validate_and_set(argument.as_deref()) {
            Ok(()) => n_src_sets = fis.get_int(1),
            Err(except) => {
                // TODO throw exception so calling object can catch and display a
                // message
                let error_message = [
                    "BeadtrackParam Error".to_owned(),
                    "Existing beadtrack magnification parameter was incorrect".to_owned(),
                    "Don't know how many non-default magnification groups, assuming 0".to_owned(),
                    format!("Input string: {}", argument.as_deref().unwrap_or("null")),
                    except.get_message().unwrap_or("null").to_owned(),
                ];
                ui_harness::post_message_dialog(
                    None,
                    error_message.join("\n"),
                    "BeadtrackParam Error".to_owned(),
                    None,
                );
            }
        }
        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.magnification_group_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;
        src_list_count = (src_list_count as i64 + n_src_sets as i64) as usize;
        for i in 0..self.magnification_group_params.get_int(1) {
            if let Some(group) = self
                .magnification_groups
                .as_ref()
                .and_then(|groups| groups.get(i as usize))
            {
                let mut magnification_group = ComScriptInputArg::new();
                magnification_group.set_argument(Some(&group.to_string()));
                input_arg_list.push(Rc::new(RefCell::new(magnification_group)));
            }
        }

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.n_min_views.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.fiducial_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        if self.fill_gaps {
            at(src_list_count)?.borrow_mut().set_argument(Some("1"));
        } else {
            at(src_list_count)?.borrow_mut().set_argument(Some("0"));
        }
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.max_gap.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.tilt_angle_min_range.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.search_box_pixels.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.max_fiducials_avg.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.fiducial_extrapolation_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.rescue_attempt_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.min_rescue_distance.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.rescue_relaxation_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&java_lang_double_to_string(
                self.residual_distance_limit,
            )));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.second_pass_params.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.mean_resid_change_limits.to_string()));
        input_arg_list.push(at(src_list_count)?);
        src_list_count += 1;

        at(src_list_count)?
            .borrow_mut()
            .set_argument(Some(&self.deletion_params.to_string()));
        input_arg_list.push(at(src_list_count)?);

        Ok(input_arg_list)
    }
}

impl CommandParam for OldBeadtrackParam {
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.parse_old_com_script_command(script_command)
    }

    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        self.update_old_com_script_command(script_command)
    }

    fn initialize_defaults(&mut self) {
        self.initialize_old_defaults();
    }
}
