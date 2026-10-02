//! `IMOD/Etomo/src/etomo/comscript/OldTiltalignParam.java`.
//!
//! The pre-PIP tiltalign param.  Necessary for backwards compatibility: reads (and
//! writes) the old sequential standard-input form of `align*.com`.
//!
//! **Errors.**  The Java reads the input arguments with `inputArgs[inputLine++]`,
//! `Integer.parseInt` and `Double.parseDouble`, whose unchecked
//! `ArrayIndexOutOfBoundsException`/`NumberFormatException` escape the declared
//! exceptions and end the calling thread.  Here (upstream bug fixed in translation) each
//! becomes a `ParseComScriptError::NumberFormat` naming the Java exception (a
//! `BadComScriptException` from `updateComScriptCommand`), returned through the
//! declared error channel.  An argument whose value is null is read as an empty
//! string where the Java would throw `NullPointerException` on `matches`.
//!
//! **Associated functions.**  `parseGroup`, `skipExistingSolnParams`, `updateSolution`,
//! `replaceTiltAngleParameters`, `replaceCompressionParameters`,
//! `replaceDistortionParameters` and `getNextNonBlankArgIndex` are private instance
//! methods in Java that read no instance state; they are associated functions here so
//! a solution field can be passed to them while the arguments are borrowed.

use std::cell::RefCell;
use std::rc::Rc;

use regex::Regex;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command_param::ParseComScriptError;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;
use super::string_list::StringList;
use super::tiltalign_solution::TiltalignSolution;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_double_value_of, java_lang_integer_parse_int,
    java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::java_lang_string_split;

pub const RCSID: &str = "$Id$";

type InputArgs = Vec<Rc<RefCell<ComScriptInputArg>>>;

/// Java package-private `OldTiltalignParam`.
pub struct OldTiltalignParam {
    pub(crate) model_file: Option<String>,
    pub(crate) image_file: Option<String>,
    pub(crate) image_parameters: FortranInputString,
    pub(crate) imod_fiducial_pos_file: Option<String>,
    pub(crate) ascii_fiducial_pos_file: Option<String>,
    pub(crate) tilt_angle_solution_file: Option<String>,
    pub(crate) transform_solution_file: Option<String>,

    pub(crate) solution_type: i32,
    pub(crate) include_exclude_type: i32,
    pub(crate) include_exclude_list: StringList,
    // what is a better name for this parameter
    // projected image rotation?
    pub(crate) initial_image_rotation: f64,
    pub(crate) rotation_angle_solution_type: i32,

    pub(crate) n_separate_view_groups: i32,
    pub(crate) separate_view_groups: StringList,

    pub(crate) tilt_angle_spec: TiltAngleSpec,
    pub(crate) tilt_angle_offset: f64,

    pub(crate) tilt_angle_solution: TiltalignSolution,
    pub(crate) magnification_solution: TiltalignSolution,
    pub(crate) compression_solution: TiltalignSolution,

    pub(crate) distortion_solution_type: i32,
    pub(crate) xstretch_solution: TiltalignSolution,
    pub(crate) skew_solution: TiltalignSolution,

    pub(crate) residual_threshold: f64,
    pub(crate) n_surface_analysis: i32,
    pub(crate) minimization_params: FortranInputString,

    pub(crate) tilt_axis_z_shift: f64,
    pub(crate) tilt_axis_x_shift: f64,

    // Local alignment parameters
    pub(crate) local_alignments: bool,
    pub(crate) local_transform_file: Option<String>,
    pub(crate) n_local_patches: FortranInputString,
    pub(crate) min_local_patch_size: FortranInputString,
    pub(crate) min_local_fiducials: FortranInputString,
    pub(crate) fix_local_fiducial_coodinates: bool,
    pub(crate) local_output_selection: FortranInputString,

    pub(crate) local_rotation_solution: TiltalignSolution,
    pub(crate) local_tilt_solution: TiltalignSolution,
    pub(crate) local_magnification_solution: TiltalignSolution,

    pub(crate) local_distortion_solution_type: i32,
    pub(crate) local_xstretch_solution: TiltalignSolution,
    pub(crate) local_skew_solution: TiltalignSolution,
}

/// `java.lang.ArrayIndexOutOfBoundsException` from `inputArgs[index]`, returned
/// instead of thrown (see the module notes).
fn index_error(index: usize, length: usize) -> ParseComScriptError {
    ParseComScriptError::NumberFormat(format!(
        "java.lang.ArrayIndexOutOfBoundsException: Index {index} out of bounds for length {length}"
    ))
}

/// `java.lang.NumberFormatException`, returned instead of thrown (see the module
/// notes).
fn number_format_error(message: String) -> ParseComScriptError {
    ParseComScriptError::NumberFormat(format!("java.lang.NumberFormatException: {message}"))
}

/// `inputArgs[index].getArgument()`.
fn argument(
    input_args: &[Rc<RefCell<ComScriptInputArg>>],
    index: usize,
) -> Result<Option<String>, ParseComScriptError> {
    match input_args.get(index) {
        None => Err(index_error(index, input_args.len())),
        Some(input_arg) => Ok(input_arg.borrow().get_argument().map(str::to_owned)),
    }
}

/// `Integer.parseInt(inputArgs[index].getArgument())`.
fn int_argument(
    input_args: &[Rc<RefCell<ComScriptInputArg>>],
    index: usize,
) -> Result<i32, ParseComScriptError> {
    match argument(input_args, index)? {
        None => Err(number_format_error(
            "Cannot parse null string: null".to_owned(),
        )),
        Some(value) => java_lang_integer_parse_int(&value).map_err(number_format_error),
    }
}

/// `Double.parseDouble(inputArgs[index].getArgument())`.
fn double_argument(
    input_args: &[Rc<RefCell<ComScriptInputArg>>],
    index: usize,
) -> Result<f64, ParseComScriptError> {
    match argument(input_args, index)? {
        None => Err(number_format_error("null".to_owned())),
        Some(value) => java_lang_double_value_of(&value).map_err(number_format_error),
    }
}

/// `inputArgs[index]` for a write.
fn input_arg(
    input_args: &[Rc<RefCell<ComScriptInputArg>>],
    index: usize,
) -> Result<Rc<RefCell<ComScriptInputArg>>, ParseComScriptError> {
    match input_args.get(index) {
        None => Err(index_error(index, input_args.len())),
        Some(input_arg) => Ok(Rc::clone(input_arg)),
    }
}

impl OldTiltalignParam {
    /// Java `OldTiltalignParam()`.
    pub(crate) fn new() -> OldTiltalignParam {
        let mut image_parameters = FortranInputString::new(6);
        let temp = [true, true, true, true, true, true];
        image_parameters.set_integer_type_array(&temp);

        let include_exclude_list = StringList::new_with_n_elements(0);
        let separate_view_groups = StringList::new_with_n_elements(0);

        let tilt_angle_spec = TiltAngleSpec::new();

        let tilt_angle_solution = TiltalignSolution::new();
        let magnification_solution = TiltalignSolution::new();
        let compression_solution = TiltalignSolution::new();
        let xstretch_solution = TiltalignSolution::new();
        let skew_solution = TiltalignSolution::new();

        let mut minimization_params = FortranInputString::new(2);
        minimization_params.set_integer_type_index(1, true);

        let mut n_local_patches = FortranInputString::new(2);
        n_local_patches.set_integer_type_index(0, true);
        n_local_patches.set_integer_type_index(1, true);

        let min_local_patch_size = FortranInputString::new(2);

        let mut min_local_fiducials = FortranInputString::new(2);
        min_local_fiducials.set_integer_type_index(0, true);
        min_local_fiducials.set_integer_type_index(1, true);

        let mut local_output_selection = FortranInputString::new(3);
        local_output_selection.set_integer_type_index(0, true);
        local_output_selection.set_integer_type_index(1, true);
        local_output_selection.set_integer_type_index(2, true);

        OldTiltalignParam {
            model_file: None,
            image_file: None,
            image_parameters,
            imod_fiducial_pos_file: None,
            ascii_fiducial_pos_file: None,
            tilt_angle_solution_file: None,
            transform_solution_file: None,
            solution_type: 0,
            include_exclude_type: 0,
            include_exclude_list,
            initial_image_rotation: 0.0,
            rotation_angle_solution_type: 0,
            n_separate_view_groups: 0,
            separate_view_groups,
            tilt_angle_spec,
            tilt_angle_offset: 0.0,
            tilt_angle_solution,
            magnification_solution,
            compression_solution,
            distortion_solution_type: 0,
            xstretch_solution,
            skew_solution,
            residual_threshold: 0.0,
            n_surface_analysis: 0,
            minimization_params,
            tilt_axis_z_shift: 0.0,
            tilt_axis_x_shift: 0.0,
            local_alignments: false,
            local_transform_file: None,
            n_local_patches,
            min_local_patch_size,
            min_local_fiducials,
            fix_local_fiducial_coodinates: false,
            local_output_selection,
            local_rotation_solution: TiltalignSolution::new(),
            local_tilt_solution: TiltalignSolution::new(),
            local_magnification_solution: TiltalignSolution::new(),
            local_distortion_solution_type: 0,
            local_xstretch_solution: TiltalignSolution::new(),
            local_skew_solution: TiltalignSolution::new(),
        }
    }

    /// Java `parseComScriptCommand`.  Get the parameters from the ComScriptCommand.
    pub(crate) fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // get the input arguments from the command
        let input_args = self.get_com_script_arguments(script_command)?;
        let input_args = &input_args[..];

        // Read in inputArgs
        let mut input_line: usize = 0;
        self.model_file = argument(input_args, input_line)?;
        input_line += 1;
        self.image_file = argument(input_args, input_line)?;
        input_line += 1;
        if java_lang_string_matches_whitespace(self.image_file.as_deref().unwrap_or("")) {
            self.image_parameters
                .validate_and_set(argument(input_args, input_line)?.as_deref())?;
            input_line += 1;
        }
        self.imod_fiducial_pos_file = argument(input_args, input_line)?;
        input_line += 1;
        self.ascii_fiducial_pos_file = argument(input_args, input_line)?;
        input_line += 1;
        self.tilt_angle_solution_file = argument(input_args, input_line)?;
        input_line += 1;
        self.transform_solution_file = argument(input_args, input_line)?;
        input_line += 1;
        self.solution_type = int_argument(input_args, input_line)?;
        input_line += 1;

        self.include_exclude_type = int_argument(input_args, input_line)?;
        input_line += 1;
        if self.include_exclude_type > 0 {
            self.include_exclude_list
                .parse_string(argument(input_args, input_line)?.as_deref());
            input_line += 1;
        }
        self.initial_image_rotation = double_argument(input_args, input_line)?;
        input_line += 1;
        self.rotation_angle_solution_type = int_argument(input_args, input_line)?;
        input_line += 1;

        self.n_separate_view_groups = int_argument(input_args, input_line)?;
        input_line += 1;
        if self.n_separate_view_groups > 0 {
            self.separate_view_groups =
                StringList::new_with_n_elements(self.n_separate_view_groups);
            for i in 0..self.n_separate_view_groups {
                self.separate_view_groups
                    .set(i, argument(input_args, input_line)?.as_deref());
                input_line += 1;
            }
        }

        // Tilt angle specification
        let type_spec = int_argument(input_args, input_line)?;
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
            self.tilt_angle_spec.set_tilt_angle_filename(
                argument(input_args, input_line)?.as_deref().unwrap_or(""),
            );
            input_line += 1;
        } else if tilt_angle_type == Some(TiltAngleType::Range) {
            let pair = argument(input_args, input_line)?;
            input_line += 1;
            // `pair.split(",")`; a null pair is a NullPointerException in the Java, and
            // is read as an empty string here (see the module notes).
            let values =
                java_lang_string_split(pair.as_deref().unwrap_or(""), &Regex::new(",").unwrap());
            if values.len() != 2 {
                return Err(
                    BadComScriptException::new("Incorrect tilt angle specification type").into(),
                );
            }
            self.tilt_angle_spec.set_range_min_double(
                java_lang_double_value_of(&values[0]).map_err(number_format_error)?,
            );
            self.tilt_angle_spec.set_range_step_double(
                java_lang_double_value_of(&values[1]).map_err(number_format_error)?,
            );
        } else if self.tilt_angle_spec.get_type() == TiltAngleType::List {
            return Err(
                BadComScriptException::new("Unimplemented tilt angle specification type").into(),
            );
        } else {
            return Err(
                BadComScriptException::new("Incorrect tilt angle specification type").into(),
            );
        }

        self.tilt_angle_offset = double_argument(input_args, input_line)?;
        input_line += 1;

        let result = (|| -> Result<(), ParseComScriptError> {
            // Tilt angle solution parameters
            self.tilt_angle_solution.r#type = int_argument(input_args, input_line)?;
            input_line += 1;

            // NOTE shouldn't be a specific integer
            // what about others
            if !(self.tilt_angle_solution.r#type == 0
                || self.tilt_angle_solution.r#type == 2
                || self.tilt_angle_solution.r#type == 5)
            {
                let message = "Don't know how to handle arbitrary tilt views yet!!!";
                return Err(InvalidParameterException::new(Some(message)).into());
            }
            if self.tilt_angle_solution.r#type == 5 {
                input_line = OldTiltalignParam::parse_group(
                    &mut self.tilt_angle_solution,
                    input_args,
                    input_line,
                )?;
            }

            // Magnification solution parameters
            self.magnification_solution
                .reference_view
                .validate_and_set(argument(input_args, input_line)?.as_deref())?;
            input_line += 1;
            self.magnification_solution.r#type = int_argument(input_args, input_line)?;
            input_line += 1;
            // NOTE shouldn't be a specific integer, what about others
            if self.magnification_solution.r#type == 2 {
                let message = "Don't know how to handle arbitrary magnification views yet!!!";
                return Err(InvalidParameterException::new(Some(message)).into());
            }
            if self.magnification_solution.r#type > 2 {
                input_line = OldTiltalignParam::parse_group(
                    &mut self.magnification_solution,
                    input_args,
                    input_line,
                )?;
            }

            // Compression solution parameters
            self.compression_solution
                .reference_view
                .validate_and_set(argument(input_args, input_line)?.as_deref())?;
            input_line += 1;
            if self.compression_solution.reference_view.get_int(0) > 0 {
                self.compression_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;
                // NOTE shouldn't be a specific integer, what about others
                if self.compression_solution.r#type == 2 {
                    let message = "Don't know how to handle arbitrary compression views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }
                if self.compression_solution.r#type > 2 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.compression_solution,
                        input_args,
                        input_line,
                    )?;
                }
            }

            self.distortion_solution_type = int_argument(input_args, input_line)?;
            input_line += 1;
            // If the distortion solution type is 1 then both the xstretch and
            // and skew parameters are stored in the xstretch solution parameters
            // If the distortion solution type is 2 then the xstretch solution
            // parameters are loaded now and the skew parameters are loaded in the
            // next statement block
            if self.distortion_solution_type > 0 {
                self.xstretch_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;

                if self.xstretch_solution.r#type == 2 {
                    let message = "Don't know how to handle arbitrary distortion views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }

                if self.xstretch_solution.r#type > 2 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.xstretch_solution,
                        input_args,
                        input_line,
                    )?;
                }
            }

            if self.distortion_solution_type == 2 {
                self.skew_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;

                if self.skew_solution.r#type == 2 {
                    let message = "Don't know how to handle arbitrary distortion views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }

                if self.skew_solution.r#type > 2 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.skew_solution,
                        input_args,
                        input_line,
                    )?;
                }
            }

            self.residual_threshold = double_argument(input_args, input_line)?;
            input_line += 1;
            self.n_surface_analysis = int_argument(input_args, input_line)?;
            input_line += 1;
            self.minimization_params
                .validate_and_set(argument(input_args, input_line)?.as_deref())?;
            input_line += 1;

            // Check to see if tranformations are being computed
            // FIXME is the test variable correct?
            if self.solution_type > 0 {
                self.tilt_axis_z_shift = double_argument(input_args, input_line)?;
                input_line += 1;
                self.tilt_axis_x_shift = double_argument(input_args, input_line)?;
                input_line += 1;
            }

            // Local alignment parsing
            let local_alignment_state = int_argument(input_args, input_line)?;
            input_line += 1;
            if local_alignment_state == 1 {
                self.local_alignments = true;
            } else {
                self.local_alignments = false;
            }
            // NOTE do we always want to do this?
            if input_args.len() > input_line {
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_transform_file = argument(input_args, input_line)?;
                input_line += 1;
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.n_local_patches
                    .validate_and_set(argument(input_args, input_line)?.as_deref())?;
                input_line += 1;
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.min_local_patch_size
                    .validate_and_set(argument(input_args, input_line)?.as_deref())?;
                input_line += 1;
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.min_local_fiducials
                    .validate_and_set(argument(input_args, input_line)?.as_deref())?;
                input_line += 1;
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                let fix_fiducial_state = int_argument(input_args, input_line)?;
                input_line += 1;
                self.fix_local_fiducial_coodinates = false;
                if fix_fiducial_state == 1 {
                    self.fix_local_fiducial_coodinates = true;
                }
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_output_selection
                    .validate_and_set(argument(input_args, input_line)?.as_deref())?;
                input_line += 1;

                // local rotation solution parameters

                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_rotation_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;
                if self.local_rotation_solution.r#type == 2 {
                    let message = "Don't know how to handle arbitrary local rotation views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }
                if self.local_rotation_solution.r#type > 2 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.local_rotation_solution,
                        input_args,
                        input_line,
                    )?;
                }

                // local tilt solution parameters
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_tilt_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;

                if !(self.local_tilt_solution.r#type == 0
                    || self.local_tilt_solution.r#type == 2
                    || self.local_tilt_solution.r#type == 5)
                {
                    let message = "Don't know how to handle arbitrary local tilt views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }
                if self.local_tilt_solution.r#type == 5 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.local_tilt_solution,
                        input_args,
                        input_line,
                    )?;
                }

                // local magnification solution parameters
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_magnification_solution
                    .reference_view
                    .validate_and_set(argument(input_args, input_line)?.as_deref())?;
                input_line += 1;
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_magnification_solution.r#type = int_argument(input_args, input_line)?;
                input_line += 1;
                // NOTE shouldn't be a specific integer, what about others
                if self.local_magnification_solution.r#type == 2 {
                    let message =
                        "Don't know how to handle arbitrary local magnification views yet!!!";
                    return Err(InvalidParameterException::new(Some(message)).into());
                }
                if self.local_magnification_solution.r#type > 2 {
                    input_line = OldTiltalignParam::parse_group(
                        &mut self.local_magnification_solution,
                        input_args,
                        input_line,
                    )?;
                }

                // local distortion solution parameters
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                self.local_distortion_solution_type = int_argument(input_args, input_line)?;
                input_line += 1;
                // Duplicate the distortion solution type for both xstretch and skew
                if self.local_distortion_solution_type == 1 {
                    input_line =
                        OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                    self.local_xstretch_solution.r#type = int_argument(input_args, input_line)?;
                    input_line += 1;
                    // OldTiltalignParam.java:385 copies `xstretchSolution.type` - the
                    // *global* x-stretch type - into the local skew type, where the
                    // comment above says the local distortion type is duplicated.
                    // Upstream bug fixed in translation: the local x-stretch type just
                    // read is duplicated.
                    self.local_skew_solution.r#type = self.local_xstretch_solution.r#type;
                    if self.local_xstretch_solution.r#type == 2 {
                        let message =
                            "Don't know how to handle arbitrary local distortion views yet!!!";
                        return Err(InvalidParameterException::new(Some(message)).into());
                    }
                    if self.local_xstretch_solution.r#type > 2 {
                        input_line = OldTiltalignParam::parse_group(
                            &mut self.local_xstretch_solution,
                            input_args,
                            input_line,
                        )?;
                    }
                }
                if self.local_distortion_solution_type == 2 {
                    input_line =
                        OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                    self.local_xstretch_solution.r#type = int_argument(input_args, input_line)?;
                    input_line += 1;
                    if self.local_xstretch_solution.r#type == 2 {
                        let message =
                            "Don't know how to handle arbitrary local distortion views yet!!!";
                        return Err(InvalidParameterException::new(Some(message)).into());
                    }
                    if self.local_xstretch_solution.r#type > 2 {
                        input_line = OldTiltalignParam::parse_group(
                            &mut self.local_xstretch_solution,
                            input_args,
                            input_line,
                        )?;
                    }
                    input_line =
                        OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                    self.local_skew_solution.r#type = int_argument(input_args, input_line)?;
                    input_line += 1;
                    if self.local_skew_solution.r#type == 2 {
                        let message =
                            "Don't know how to handle arbitrary local distortion views yet!!!";
                        return Err(InvalidParameterException::new(Some(message)).into());
                    }
                    if self.local_skew_solution.r#type > 2 {
                        input_line = OldTiltalignParam::parse_group(
                            &mut self.local_skew_solution,
                            input_args,
                            input_line,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        match result {
            Err(ParseComScriptError::FortranInputSyntax(except)) => {
                let message = format!(
                    "Parse error in tiltalign command, standard input argument: {}\n{}",
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

    /// Java `updateComScriptCommand`.  Update the script command with the current
    /// values of this object.
    pub(crate) fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // get the input arguments from the command
        let input_args = self.get_com_script_arguments(script_command)?;

        let input_args = self.put_com_script_arguments(input_args)?;
        script_command.set_input_arguments(&input_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    pub(crate) fn initialize_defaults(&mut self) {}

    //
    // Set the values of the tiltalign parameters
    //
    /// Java `setModelFile`.
    pub(crate) fn set_model_file(&mut self, filename: Option<&str>) {
        self.model_file = filename.map(str::to_owned);
    }

    /// Java `setImageFile`.
    pub(crate) fn set_image_file(&mut self, filename: Option<&str>) {
        self.image_file = filename.map(str::to_owned);
    }

    /// Java `setImageParameters`.
    pub(crate) fn set_image_parameters(
        &mut self,
        new_image_parameters: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.image_parameters.validate_and_set(new_image_parameters)
    }

    /// Java `setImodFiducialPosFile`.
    pub(crate) fn set_imod_fiducial_pos_file(&mut self, filename: Option<&str>) {
        self.imod_fiducial_pos_file = filename.map(str::to_owned);
    }

    /// Java `setAsciiFiducialPosFile`.
    pub(crate) fn set_ascii_fiducial_pos_file(&mut self, filename: Option<&str>) {
        self.ascii_fiducial_pos_file = filename.map(str::to_owned);
    }

    /// Java `setTiltAngleSolutionFile`.
    pub(crate) fn set_tilt_angle_solution_file(&mut self, filename: Option<&str>) {
        self.tilt_angle_solution_file = filename.map(str::to_owned);
    }

    /// Java `setTransformSolutionFile`.
    pub(crate) fn set_transform_solution_file(&mut self, filename: Option<&str>) {
        self.transform_solution_file = filename.map(str::to_owned);
    }

    // TODO validation
    /// Java `setSolutionType`.
    pub(crate) fn set_solution_type(&mut self, r#type: i32) {
        self.solution_type = r#type;
    }

    /// Java `setIncludeExcludeType`.
    pub(crate) fn set_include_exclude_type(&mut self, code: i32) {
        self.include_exclude_type = code;
    }

    // TODO validation
    /// Java `setIncludeExcludeList`.
    pub(crate) fn set_include_exclude_list(&mut self, z_list: Option<&str>) {
        self.include_exclude_list.parse_string(z_list);
    }

    /// Java `setInitialImageRoation`.
    pub(crate) fn set_initial_image_roation(&mut self, angle: f64) {
        self.initial_image_rotation = angle;
    }

    /// Java `setRotationAngleSolutionType`.
    pub(crate) fn set_rotation_angle_solution_type(&mut self, r#type: i32) {
        self.rotation_angle_solution_type = r#type;
    }

    /// Java `setSeparateViewGroups`.
    pub(crate) fn set_separate_view_groups(&mut self, new_list: Option<&str>) {
        self.separate_view_groups.parse_string(new_list);
        self.n_separate_view_groups = self.separate_view_groups.get_n_elements();
    }

    /// Java `setTiltAngleOffset`.  `Err` is the `NumberFormatException` message.
    pub(crate) fn set_tilt_angle_offset(
        &mut self,
        new_tilt_angle_offset: &str,
    ) -> Result<(), String> {
        self.tilt_angle_offset = java_lang_double_value_of(new_tilt_angle_offset)?;
        Ok(())
    }

    /// Java `setTiltAngleSolutionType`.
    pub(crate) fn set_tilt_angle_solution_type(&mut self, r#type: i32) {
        self.tilt_angle_solution.r#type = r#type;
    }

    /// Java `setTiltAngleSolutionGroupSize(int)`.
    pub(crate) fn set_tilt_angle_solution_group_size_int(&mut self, size: i32) {
        self.tilt_angle_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setTiltAngleSolutionGroupSize(String)`.  `Err` is the
    /// `NumberFormatException` message.
    pub(crate) fn set_tilt_angle_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.tilt_angle_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setTiltAngleSolutionAdditionalGroups`.
    pub(crate) fn set_tilt_angle_solution_additional_groups(&mut self, list: Option<&str>) {
        self.tilt_angle_solution
            .additional_groups
            .parse_string(list);
        let n = self.tilt_angle_solution.additional_groups.get_n_elements();
        self.tilt_angle_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setMagnificationReferenceView`.
    pub(crate) fn set_magnification_reference_view(
        &mut self,
        new_reference_view: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.magnification_solution
            .reference_view
            .validate_and_set(new_reference_view)
    }

    /// Java `setMagnificationType`.
    pub(crate) fn set_magnification_type(&mut self, r#type: i32) {
        self.magnification_solution.r#type = r#type;
    }

    /// Java `setMagnificationSolutionGroupSize(int)`.
    pub(crate) fn set_magnification_solution_group_size_int(&mut self, size: i32) {
        self.magnification_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setMagnificationSolutionGroupSize(String)`.
    pub(crate) fn set_magnification_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.magnification_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setMagnificationSolutionAdditionalGroups`.
    pub(crate) fn set_magnification_solution_additional_groups(&mut self, list: Option<&str>) {
        self.magnification_solution
            .additional_groups
            .parse_string(list);
        let n = self
            .magnification_solution
            .additional_groups
            .get_n_elements();
        self.magnification_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setCompressionReferenceView`.
    pub(crate) fn set_compression_reference_view(
        &mut self,
        new_reference_view: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.compression_solution
            .reference_view
            .validate_and_set(new_reference_view)
    }

    /// Java `setCompressionType`.
    pub(crate) fn set_compression_type(&mut self, r#type: i32) {
        self.compression_solution.r#type = r#type;
    }

    /// Java `setCompressionSolutionGroupSize(int)`.
    pub(crate) fn set_compression_solution_group_size_int(&mut self, size: i32) {
        self.compression_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setCompressionSolutionGroupSize(String)`.
    pub(crate) fn set_compression_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.compression_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setCompressionSolutionAdditionalGroups`.
    pub(crate) fn set_compression_solution_additional_groups(&mut self, list: Option<&str>) {
        self.compression_solution
            .additional_groups
            .parse_string(list);
        let n = self.compression_solution.additional_groups.get_n_elements();
        self.compression_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setDistortionSolutionType`.
    pub(crate) fn set_distortion_solution_type(&mut self, r#type: i32) {
        self.distortion_solution_type = r#type;
    }

    /// Java `setXstretchType`.
    pub(crate) fn set_xstretch_type(&mut self, r#type: i32) {
        self.xstretch_solution.r#type = r#type;
    }

    /// Java `setXstretchSolutionGroupSize(int)`.
    pub(crate) fn set_xstretch_solution_group_size_int(&mut self, size: i32) {
        self.xstretch_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setXstretchSolutionGroupSize(String)`.
    pub(crate) fn set_xstretch_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.xstretch_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setXstretchSolutionAdditionalGroups`.
    pub(crate) fn set_xstretch_solution_additional_groups(&mut self, list: Option<&str>) {
        self.xstretch_solution.additional_groups.parse_string(list);
        let n = self.xstretch_solution.additional_groups.get_n_elements();
        self.xstretch_solution.params.set_index_double(1, n as f64);
    }

    /// Java `setSkewType`.
    pub(crate) fn set_skew_type(&mut self, r#type: i32) {
        self.skew_solution.r#type = r#type;
    }

    /// Java `setSkewSolutionGroupSize(int)`.
    pub(crate) fn set_skew_solution_group_size_int(&mut self, size: i32) {
        self.skew_solution.params.set_index_double(0, size as f64);
    }

    /// Java `setSkewSolutionGroupSize(String)`.
    pub(crate) fn set_skew_solution_group_size_string(&mut self, size: &str) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.skew_solution.params.set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setSkewSolutionAdditionalGroups`.
    pub(crate) fn set_skew_solution_additional_groups(&mut self, list: Option<&str>) {
        self.skew_solution.additional_groups.parse_string(list);
        let n = self.skew_solution.additional_groups.get_n_elements();
        self.skew_solution.params.set_index_double(1, n as f64);
    }

    /// Java `setResidualThreshold`.
    pub(crate) fn set_residual_threshold(&mut self, threshold: f64) {
        self.residual_threshold = threshold;
    }

    /// Java `setNSurfaceAnalysis(String)`.
    pub(crate) fn set_n_surface_analysis_string(
        &mut self,
        new_n_surface_analysis: &str,
    ) -> Result<(), String> {
        self.n_surface_analysis = java_lang_integer_parse_int(new_n_surface_analysis)?;
        Ok(())
    }

    /// Java `setNSurfaceAnalysis(int)`.
    pub(crate) fn set_n_surface_analysis_int(&mut self, n: i32) {
        self.n_surface_analysis = n;
    }

    /// Java `setMinimizationParams`.
    pub(crate) fn set_minimization_params(
        &mut self,
        params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.minimization_params.validate_and_set(params)
    }

    /// Java `setTiltAxisZShift(double)`.
    pub(crate) fn set_tilt_axis_z_shift_double(&mut self, shift: f64) {
        self.tilt_axis_z_shift = shift;
    }

    /// Java `setTiltAxisZShift(String)`.
    pub(crate) fn set_tilt_axis_z_shift_string(&mut self, shift: &str) -> Result<(), String> {
        self.tilt_axis_z_shift = java_lang_double_value_of(shift)?;
        Ok(())
    }

    /// Java `setTiltAxisXShift(double)`.
    pub(crate) fn set_tilt_axis_x_shift_double(&mut self, shift: f64) {
        self.tilt_axis_x_shift = shift;
    }

    /// Java `setTiltAxisXShift(String)`.
    pub(crate) fn set_tilt_axis_x_shift_string(&mut self, shift: &str) -> Result<(), String> {
        self.tilt_axis_x_shift = java_lang_double_value_of(shift)?;
        Ok(())
    }

    /// Java `setLocalAlignments`.
    pub(crate) fn set_local_alignments(&mut self, state: bool) {
        self.local_alignments = state;
    }

    /// Java `setLocalTransformFile`.
    pub(crate) fn set_local_transform_file(&mut self, filename: Option<&str>) {
        self.local_transform_file = filename.map(str::to_owned);
    }

    /// Java `setMetroFactor`.
    pub(crate) fn set_metro_factor(&mut self, factor: &str) -> Result<(), String> {
        let factor = java_lang_double_value_of(factor)?;
        self.minimization_params.set_index_double(0, factor);
        Ok(())
    }

    /// Java `setCycleLimit`.
    pub(crate) fn set_cycle_limit(&mut self, limit: &str) -> Result<(), String> {
        let limit = java_lang_integer_parse_int(limit)?;
        self.minimization_params.set_index_double(1, limit as f64);
        Ok(())
    }

    /// Java `setNLocalPatches`.
    pub(crate) fn set_n_local_patches(
        &mut self,
        params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.n_local_patches.validate_and_set(params)
    }

    /// Java `setMinLocalPatchSize`.
    pub(crate) fn set_min_local_patch_size(
        &mut self,
        params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.min_local_patch_size.validate_and_set(params)
    }

    /// Java `setMinLocalFiducials`.
    pub(crate) fn set_min_local_fiducials(
        &mut self,
        params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.min_local_fiducials.validate_and_set(params)
    }

    /// Java `setFixLocalFiducialCoodinates`.
    pub(crate) fn set_fix_local_fiducial_coodinates(&mut self, state: bool) {
        self.fix_local_fiducial_coodinates = state;
    }

    /// Java `setLocalOutputSelection`.
    pub(crate) fn set_local_output_selection(
        &mut self,
        params: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.local_output_selection.validate_and_set(params)
    }

    /// Java `setLocalRotationSolutionType`.
    pub(crate) fn set_local_rotation_solution_type(&mut self, r#type: i32) {
        self.local_rotation_solution.r#type = r#type;
    }

    /// Java `setLocalRotationSolutionGroupSize(int)`.
    pub(crate) fn set_local_rotation_solution_group_size_int(&mut self, size: i32) {
        self.local_rotation_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setLocalRotationSolutionGroupSize(String)`.
    pub(crate) fn set_local_rotation_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.local_rotation_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setLocalRotationSolutionAdditionalGroups`.
    pub(crate) fn set_local_rotation_solution_additional_groups(&mut self, list: Option<&str>) {
        self.local_rotation_solution
            .additional_groups
            .parse_string(list);
        let n = self
            .local_rotation_solution
            .additional_groups
            .get_n_elements();
        self.local_rotation_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setLocalTiltSolutionType`.
    pub(crate) fn set_local_tilt_solution_type(&mut self, r#type: i32) {
        self.local_tilt_solution.r#type = r#type;
    }

    /// Java `setLocalTiltSolutionGroupSize(int)`.
    pub(crate) fn set_local_tilt_solution_group_size_int(&mut self, size: i32) {
        self.local_tilt_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setLocalTiltSolutionGroupSize(String)`.
    pub(crate) fn set_local_tilt_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.local_tilt_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setLocalTiltSolutionAdditionalGroups`.
    pub(crate) fn set_local_tilt_solution_additional_groups(&mut self, list: Option<&str>) {
        self.local_tilt_solution
            .additional_groups
            .parse_string(list);
        let n = self.local_tilt_solution.additional_groups.get_n_elements();
        self.local_tilt_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setLocalMagnificationReferenceView`.
    pub(crate) fn set_local_magnification_reference_view(
        &mut self,
        new_reference_view: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.local_magnification_solution
            .reference_view
            .validate_and_set(new_reference_view)
    }

    /// Java `setLocalMagnificationType`.
    pub(crate) fn set_local_magnification_type(&mut self, r#type: i32) {
        self.local_magnification_solution.r#type = r#type;
    }

    /// Java `setLocalMagnificationSolutionGroupSize(int)`.
    pub(crate) fn set_local_magnification_solution_group_size_int(&mut self, size: i32) {
        self.local_magnification_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setLocalMagnificationSolutionGroupSize(String)`.
    pub(crate) fn set_local_magnification_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.local_magnification_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setLocalMagnificationSolutionAdditionalGroups`.
    pub(crate) fn set_local_magnification_solution_additional_groups(
        &mut self,
        list: Option<&str>,
    ) {
        self.local_magnification_solution
            .additional_groups
            .parse_string(list);
        let n = self
            .local_magnification_solution
            .additional_groups
            .get_n_elements();
        self.local_magnification_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setLocalDistortionSolutionType`.
    pub(crate) fn set_local_distortion_solution_type(&mut self, r#type: i32) {
        self.local_distortion_solution_type = r#type;
    }

    /// Java `setLocalXstretchType`.
    pub(crate) fn set_local_xstretch_type(&mut self, r#type: i32) {
        self.local_xstretch_solution.r#type = r#type;
    }

    /// Java `setLocalXstretchSolutionGroupSize(int)`.
    pub(crate) fn set_local_xstretch_solution_group_size_int(&mut self, size: i32) {
        self.local_xstretch_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setLocalXstretchSolutionGroupSize(String)`.
    pub(crate) fn set_local_xstretch_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.local_xstretch_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setLocalXstretchSolutionAdditionalGroups`.
    pub(crate) fn set_local_xstretch_solution_additional_groups(&mut self, list: Option<&str>) {
        self.local_xstretch_solution
            .additional_groups
            .parse_string(list);
        let n = self
            .local_xstretch_solution
            .additional_groups
            .get_n_elements();
        self.local_xstretch_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java `setLocalSkewType`.
    pub(crate) fn set_local_skew_type(&mut self, r#type: i32) {
        self.local_skew_solution.r#type = r#type;
    }

    /// Java `setLocalSkewSolutionGroupSize(int)`.
    pub(crate) fn set_local_skew_solution_group_size_int(&mut self, size: i32) {
        self.local_skew_solution
            .params
            .set_index_double(0, size as f64);
    }

    /// Java `setLocalSkewSolutionGroupSize(String)`.
    pub(crate) fn set_local_skew_solution_group_size_string(
        &mut self,
        size: &str,
    ) -> Result<(), String> {
        let size = java_lang_integer_parse_int(size)?;
        self.local_skew_solution
            .params
            .set_index_double(0, size as f64);
        Ok(())
    }

    /// Java `setLocalSkewSolutionAdditionalGroups`.
    pub(crate) fn set_local_skew_solution_additional_groups(&mut self, list: Option<&str>) {
        self.local_skew_solution
            .additional_groups
            .parse_string(list);
        let n = self.local_skew_solution.additional_groups.get_n_elements();
        self.local_skew_solution
            .params
            .set_index_double(1, n as f64);
    }

    /// Java private `parseGroup`.  Parse a FortranInputString and StringList group.
    fn parse_group(
        solution: &mut TiltalignSolution,
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        input_line: usize,
    ) -> Result<usize, ParseComScriptError> {
        let mut input_line =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
        solution
            .params
            .validate_and_set(argument(input_args, input_line)?.as_deref())?;
        input_line += 1;

        let n_groups = solution.params.get_int(1);
        if n_groups > 0 {
            solution.additional_groups = StringList::new_with_n_elements(n_groups);
            for i in 0..n_groups {
                input_line =
                    OldTiltalignParam::get_next_non_blank_arg_index(input_args, input_line);
                solution
                    .additional_groups
                    .set(i, argument(input_args, input_line)?.as_deref());
                input_line += 1;
            }
        }
        Ok(input_line)
    }

    /// Java private `putComScriptArguments`.  Update the inputArguments array with new
    /// parameters.  Returns the new array of ComScriptInputArgs; a newly allocated
    /// array since the number of elements may be different than the input parameter.
    fn put_com_script_arguments(
        &self,
        input_args: InputArgs,
    ) -> Result<InputArgs, BadComScriptException> {
        let input_args = &input_args[..];
        let mut input_arg_list: InputArgs = Vec::new();

        // The unchecked exceptions of the part before the Java `try` become
        // `BadComScriptException`s (see the module notes).
        let to_bad = |e: ParseComScriptError| BadComScriptException::new(&e.to_string());

        // Fill in the input argument sequence, the srcListCount variable
        // acts as an index into the existing input argument array
        let mut src_list_count: usize = 0;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut().set_argument(self.model_file.as_deref());
        input_arg_list.push(arg);
        src_list_count += 1;

        // Sync the existing and new image file and image parameters
        let image_file_blank =
            java_lang_string_matches_whitespace(self.image_file.as_deref().unwrap_or(""));
        if java_lang_string_matches_whitespace(
            argument(input_args, src_list_count)
                .map_err(to_bad)?
                .as_deref()
                .unwrap_or(""),
        ) {
            let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
            arg.borrow_mut().set_argument(Some(""));
            input_arg_list.push(arg);
            src_list_count += 1;

            // Both blank followed by image parameter
            if image_file_blank {
                let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
                arg.borrow_mut()
                    .set_argument_fortran_input_string(&self.image_parameters);
                input_arg_list.push(arg);
            }
            src_list_count += 1;
        }
        // Only one existing ouput argument
        else {
            let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
            arg.borrow_mut().set_argument(self.image_file.as_deref());
            input_arg_list.push(arg);
            src_list_count += 1;
            if image_file_blank {
                // Kept as the Java writes it (OldTiltalignParam.java:1020-1021): the new
                // argument is built and never added to the list.  This method is not
                // reachable from TiltalignParam, which only parses the old format.
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument_fortran_input_string(&self.image_parameters);
            }
        }

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument(self.imod_fiducial_pos_file.as_deref());
        input_arg_list.push(arg);
        src_list_count += 1;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument(self.ascii_fiducial_pos_file.as_deref());
        input_arg_list.push(arg);
        src_list_count += 1;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument(self.tilt_angle_solution_file.as_deref());
        input_arg_list.push(arg);
        src_list_count += 1;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument(self.transform_solution_file.as_deref());
        input_arg_list.push(arg);
        src_list_count += 1;

        let existing_solution_type = int_argument(input_args, src_list_count).map_err(to_bad)?;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut().set_argument_int(self.solution_type);
        input_arg_list.push(arg);
        src_list_count += 1;

        // Sync the existing and new include points parameters
        let existing_include_points = int_argument(input_args, src_list_count).map_err(to_bad)?;
        // Existing input sequence is single line
        if existing_include_points == 0 {
            let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
            arg.borrow_mut().set_argument_int(self.include_exclude_type);
            input_arg_list.push(arg);
            src_list_count += 1;

            // New include points is multiline
            if self.include_exclude_type > 0 {
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument(Some(&self.include_exclude_list.to_string()));
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            }
        }
        // Existing multiline input sequence
        else {
            let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
            arg.borrow_mut().set_argument_int(self.include_exclude_type);
            input_arg_list.push(arg);
            src_list_count += 1;

            // New include points is multiline
            if self.include_exclude_type > 0 {
                let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
                arg.borrow_mut()
                    .set_argument(Some(&self.include_exclude_list.to_string()));
                input_arg_list.push(arg);
            }
            src_list_count += 1;
        }

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument_double(self.initial_image_rotation);
        input_arg_list.push(arg);
        src_list_count += 1;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument_int(self.rotation_angle_solution_type);
        input_arg_list.push(arg);
        src_list_count += 1;

        // Increment the source list counter to skip the old view groups, the
        // comments for those entries are most likely not applicable
        let n_src_sets = int_argument(input_args, src_list_count).map_err(to_bad)?;
        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut()
            .set_argument_int(self.n_separate_view_groups);
        input_arg_list.push(arg);
        src_list_count += 1;
        // A negative count moves the Java index backwards.
        src_list_count = (src_list_count as i64 + n_src_sets as i64).max(0) as usize;
        for i in 0..self.n_separate_view_groups {
            let mut view_set = ComScriptInputArg::new();
            view_set.set_argument(self.separate_view_groups.get(i).as_deref());
            input_arg_list.push(Rc::new(RefCell::new(view_set)));
        }

        // Tilt angle source and filenames are not modified by this class
        input_arg_list.push(input_arg(input_args, src_list_count).map_err(to_bad)?);
        src_list_count += 1;
        input_arg_list.push(input_arg(input_args, src_list_count).map_err(to_bad)?);
        src_list_count += 1;

        let arg = input_arg(input_args, src_list_count).map_err(to_bad)?;
        arg.borrow_mut().set_argument_double(self.tilt_angle_offset);
        input_arg_list.push(arg);
        src_list_count += 1;

        let result = (|| -> Result<(), ParseComScriptError> {
            src_list_count = OldTiltalignParam::replace_tilt_angle_parameters(
                &self.tilt_angle_solution,
                input_args,
                src_list_count,
                &mut input_arg_list,
            )?;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.magnification_solution.reference_view);
            input_arg_list.push(arg);
            src_list_count += 1;

            // FIXME: this looses the comments to the magnification selection
            src_list_count =
                OldTiltalignParam::skip_existing_soln_params(input_args, src_list_count)?;
            OldTiltalignParam::update_solution(&self.magnification_solution, &mut input_arg_list);

            src_list_count = OldTiltalignParam::replace_compression_parameters(
                &self.compression_solution,
                input_args,
                src_list_count,
                &mut input_arg_list,
            )?;

            src_list_count = OldTiltalignParam::replace_distortion_parameters(
                self.distortion_solution_type,
                &self.xstretch_solution,
                &self.skew_solution,
                input_args,
                src_list_count,
                &mut input_arg_list,
            )?;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_double(self.residual_threshold);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut().set_argument_int(self.n_surface_analysis);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.minimization_params);
            input_arg_list.push(arg);
            src_list_count += 1;

            // Axis shift parameters
            if existing_solution_type > -1 {
                if self.solution_type > -1 {
                    let arg = input_arg(input_args, src_list_count)?;
                    arg.borrow_mut().set_argument_double(self.tilt_axis_z_shift);
                    input_arg_list.push(arg);
                    src_list_count += 1;
                    let arg = input_arg(input_args, src_list_count)?;
                    arg.borrow_mut().set_argument_double(self.tilt_axis_x_shift);
                    input_arg_list.push(arg);
                    src_list_count += 1;
                } else {
                    src_list_count += 2;
                }
            } else if self.solution_type > -1 {
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument_double(self.tilt_axis_z_shift);
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument_double(self.tilt_axis_x_shift);
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            }

            // Local alignments
            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut().set_argument_boolean(self.local_alignments);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument(self.local_transform_file.as_deref());
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.n_local_patches);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.min_local_patch_size);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.min_local_fiducials);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_boolean(self.fix_local_fiducial_coodinates);
            input_arg_list.push(arg);
            src_list_count += 1;

            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut()
                .set_argument_fortran_input_string(&self.local_output_selection);
            input_arg_list.push(arg);
            src_list_count += 1;

            src_list_count =
                OldTiltalignParam::skip_existing_soln_params(input_args, src_list_count)?;
            OldTiltalignParam::update_solution(&self.local_rotation_solution, &mut input_arg_list);

            src_list_count = OldTiltalignParam::replace_tilt_angle_parameters(
                &self.local_tilt_solution,
                input_args,
                src_list_count,
                &mut input_arg_list,
            )?;

            src_list_count =
                OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
            let arg = input_arg(input_args, src_list_count)?;
            arg.borrow_mut().set_argument_fortran_input_string(
                &self.local_magnification_solution.reference_view,
            );
            input_arg_list.push(arg);
            src_list_count += 1;

            src_list_count =
                OldTiltalignParam::skip_existing_soln_params(input_args, src_list_count)?;
            OldTiltalignParam::update_solution(
                &self.local_magnification_solution,
                &mut input_arg_list,
            );

            src_list_count = OldTiltalignParam::replace_distortion_parameters(
                self.local_distortion_solution_type,
                &self.local_xstretch_solution,
                &self.local_skew_solution,
                input_args,
                src_list_count,
                &mut input_arg_list,
            )?;
            Ok(())
        })();
        if let Err(except) = result {
            // `except.printStackTrace()`
            eprintln!("{except}");
            // TODO this should probably throw an excpetion or set a state so that
            // some other code above a handles it
            // `inputArgs[srcListCount].getArgument()` can itself be out of range here,
            // which would throw from the Java catch block; "null" is printed instead.
            let input_string = input_args
                .get(src_list_count)
                .and_then(|arg| arg.borrow().get_argument().map(str::to_owned));
            let error_message = [
                "TiltalignParam Error".to_owned(),
                "Existing Tiltalign parameter was incorrect".to_owned(),
                "The align*.com file appears to have changed inappropriately on disk".to_owned(),
                format!(
                    "Input string: {}",
                    input_string.as_deref().unwrap_or("null")
                ),
                format!("Argument #: {src_list_count}"),
                except.to_string(),
            ];
            // `JOptionPane.showMessageDialog(null, errorMessage, "TiltalignParam Error",
            // ERROR_MESSAGE)`: posted to the event dispatch thread, one line per array
            // element as the option pane lays out a String[] message.
            ui_harness::post_message_dialog(
                None,
                error_message.join("\n"),
                "TiltalignParam Error".to_owned(),
                None,
            );
        }

        Ok(input_arg_list)
    }

    /// Java private `getComScriptArguments`.  Get the standard input arguments from
    /// the ComScriptCommand validating the name of the command and the appropriate
    /// number of input arguments.
    fn get_com_script_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<InputArgs, BadComScriptException> {
        // Check to be sure that it is a tiltxcorr xommand
        if script_command.get_command() != Some("tiltalign") {
            return Err(BadComScriptException::new("Not a tiltalign command"));
        }

        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        if input_args.len() < 17 {
            return Err(BadComScriptException::new(&format!(
                "Incorrect number of input arguments to tiltalign command\nGot {} expected at least 24.",
                input_args.len()
            )));
        }

        Ok(input_args)
    }

    /// Java private `skipExistingSolnParams`.  Generic solution parameter replacement
    /// method.
    fn skip_existing_soln_params(
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        src_list_count: usize,
    ) -> Result<usize, ParseComScriptError> {
        // Solution parameters, need to figure out how many
        // lines are in the existing input subsequence
        let mut src_list_count =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
        let existing_solution_type = int_argument(input_args, src_list_count)?;

        if existing_solution_type == 2 {
            let message = "Don't know how to handle arbitrary views yet!!!";
            return Err(InvalidParameterException::new(Some(message)).into());
        }

        // Skip the type argument
        src_list_count += 1;

        // Skip the corrent number of existing arguments
        if existing_solution_type > 2 {
            let mut existing_params = FortranInputString::new(2);
            existing_params.validate_and_set(argument(input_args, src_list_count)?.as_deref())?;
            src_list_count += 1;
            let n_groups = existing_params.get_int(1);
            src_list_count = (src_list_count as i64 + n_groups as i64).max(0) as usize;
        }
        Ok(src_list_count)
    }

    /// Java private `updateSolution`.
    fn update_solution(solution: &TiltalignSolution, input_arg_list: &mut InputArgs) {
        // Add the new solution parameters to the list
        let mut new_arg = ComScriptInputArg::new();
        new_arg.set_argument_int(solution.r#type);
        input_arg_list.push(Rc::new(RefCell::new(new_arg)));

        if solution.r#type > 2 {
            let mut new_arg = ComScriptInputArg::new();
            new_arg.set_argument_fortran_input_string(&solution.params);
            input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            for i in 0..solution.params.get_int(1) {
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument(solution.additional_groups.get(i).as_deref());
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            }
        }
    }

    /// Java private `replaceTiltAngleParameters`.  Replace the tilt angle parameters.
    fn replace_tilt_angle_parameters(
        solution: &TiltalignSolution,
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        src_list_count: usize,
        input_arg_list: &mut InputArgs,
    ) -> Result<usize, ParseComScriptError> {
        // Skip over the correct number of existing tilt angle solution type
        // and parameters, this requires parsing the existing tilt angle solution
        // parameter sequence
        let mut src_list_count =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
        let existing_tilt_soltn_type = int_argument(input_args, src_list_count)?;
        if !(existing_tilt_soltn_type == 0
            || existing_tilt_soltn_type == 2
            || existing_tilt_soltn_type == 5)
        {
            let message = "Don't know how to handle arbitrary tilt views yet!!!";
            return Err(InvalidParameterException::new(Some(message)).into());
        }
        src_list_count =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
        let arg = input_arg(input_args, src_list_count)?;
        arg.borrow_mut().set_argument_int(solution.r#type);
        input_arg_list.push(arg);
        src_list_count += 1;

        // Skip the corrent number of existing arguments
        if existing_tilt_soltn_type == 5 {
            let mut existing_params = FortranInputString::new(2);
            existing_params.validate_and_set(argument(input_args, src_list_count)?.as_deref())?;
            src_list_count += 1;
            let n_groups = existing_params.get_int(1);
            src_list_count = (src_list_count as i64 + n_groups as i64).max(0) as usize;
        }

        // Add the new tilt angle parameters to the list
        if solution.r#type == 5 {
            let mut new_arg = ComScriptInputArg::new();
            new_arg.set_argument_fortran_input_string(&solution.params);
            input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            for i in 0..solution.params.get_int(1) {
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument(solution.additional_groups.get(i).as_deref());
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
            }
        }
        Ok(src_list_count)
    }

    /// Java private `replaceCompressionParameters`.
    fn replace_compression_parameters(
        solution: &TiltalignSolution,
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        src_list_count: usize,
        input_arg_list: &mut InputArgs,
    ) -> Result<usize, ParseComScriptError> {
        // Solution parameters, need to figure out how many
        // lines are in the existing input subsequence
        let mut src_list_count =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
        let existing_compression_reference_view = int_argument(input_args, src_list_count)?;

        let arg = input_arg(input_args, src_list_count)?;
        arg.borrow_mut()
            .set_argument_fortran_input_string(&solution.reference_view);
        input_arg_list.push(arg);
        src_list_count += 1;

        if existing_compression_reference_view > 0 {
            let existing_solution_type = int_argument(input_args, src_list_count)?;
            src_list_count += 1;
            if existing_solution_type == 2 {
                let message = "Don't know how to handle arbitrary views yet!!!";
                return Err(InvalidParameterException::new(Some(message)).into());
            }

            // Skip the corrent number of existing arguments
            if existing_solution_type > 2 {
                let mut existing_params = FortranInputString::new(2);
                existing_params
                    .validate_and_set(argument(input_args, src_list_count)?.as_deref())?;
                src_list_count += 1;
                let n_groups = existing_params.get_int(1);
                src_list_count = (src_list_count as i64 + n_groups as i64).max(0) as usize;
            }
        }

        // Check the current reference view to see if we should include any more
        // parameters
        if solution.reference_view.get_int(0) > 0 {
            let mut new_arg = ComScriptInputArg::new();
            new_arg.set_argument_int(solution.r#type);
            input_arg_list.push(Rc::new(RefCell::new(new_arg)));

            // Add the new solution parameters to the list
            if solution.r#type > 2 {
                let mut new_arg = ComScriptInputArg::new();
                new_arg.set_argument_fortran_input_string(&solution.params);
                input_arg_list.push(Rc::new(RefCell::new(new_arg)));
                for i in 0..solution.params.get_int(1) {
                    let mut new_arg = ComScriptInputArg::new();
                    new_arg.set_argument(solution.additional_groups.get(i).as_deref());
                    input_arg_list.push(Rc::new(RefCell::new(new_arg)));
                }
            }
        }
        Ok(src_list_count)
    }

    /// Java private `replaceDistortionParameters`.  Replace the distortion parameters.
    fn replace_distortion_parameters(
        solution_type: i32,
        xstretch: &TiltalignSolution,
        skew: &TiltalignSolution,
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        src_list_count: usize,
        input_arg_list: &mut InputArgs,
    ) -> Result<usize, ParseComScriptError> {
        let mut src_list_count =
            OldTiltalignParam::get_next_non_blank_arg_index(input_args, src_list_count);
        let existing_distortion_type = int_argument(input_args, src_list_count)?;

        let arg = input_arg(input_args, src_list_count)?;
        arg.borrow_mut().set_argument_int(solution_type);
        input_arg_list.push(arg);
        src_list_count += 1;

        // Skip the existing X stretch solution parameters
        if existing_distortion_type > 0 {
            src_list_count =
                OldTiltalignParam::skip_existing_soln_params(input_args, src_list_count)?;
        }

        // Skip the existing skew solution distortion parameters
        if existing_distortion_type == 2 {
            src_list_count =
                OldTiltalignParam::skip_existing_soln_params(input_args, src_list_count)?;
        }

        // Update the new xstretch solution parameters
        if solution_type > 0 {
            OldTiltalignParam::update_solution(xstretch, input_arg_list);
        }
        // Update the new skew solution parameters
        if solution_type == 2 {
            OldTiltalignParam::update_solution(skew, input_arg_list);
        }
        Ok(src_list_count)
    }

    /// Java private `getNextNonBlankArgIndex`.  The Java loop runs off the end of the
    /// array (ArrayIndexOutOfBoundsException) when every remaining argument is blank;
    /// here it stops at the length and the next read reports the out-of-range index
    /// (see the module notes).
    fn get_next_non_blank_arg_index(
        input_args: &[Rc<RefCell<ComScriptInputArg>>],
        input_line: usize,
    ) -> usize {
        let mut input_line = input_line;
        while input_line < input_args.len()
            && java_lang_string_matches_whitespace(
                input_args[input_line].borrow().get_argument().unwrap_or(""),
            )
        {
            input_line += 1;
        }
        input_line
    }

    /// Java `getModelFile`.
    pub(crate) fn get_model_file(&self) -> Option<&str> {
        self.model_file.as_deref()
    }

    /// Java `getImageFile`.
    pub(crate) fn get_image_file(&self) -> Option<&str> {
        self.image_file.as_deref()
    }

    /// Java `getImageParameters`.
    pub(crate) fn get_image_parameters(&self) -> String {
        self.image_parameters.to_string()
    }

    /// Java `getIMODFiducialPosFile`.
    pub(crate) fn get_imod_fiducial_pos_file(&self) -> Option<&str> {
        self.imod_fiducial_pos_file.as_deref()
    }

    /// Java `getAsciiFiducialPosFile`.
    pub(crate) fn get_ascii_fiducial_pos_file(&self) -> Option<&str> {
        self.ascii_fiducial_pos_file.as_deref()
    }

    /// Java `getTiltAngleSolutionFile`.
    pub(crate) fn get_tilt_angle_solution_file(&self) -> Option<&str> {
        self.tilt_angle_solution_file.as_deref()
    }

    /// Java `getTransformSolutionFile`.
    pub(crate) fn get_transform_solution_file(&self) -> Option<&str> {
        self.transform_solution_file.as_deref()
    }

    /// Java `getSolutionType`.
    pub(crate) fn get_solution_type(&self) -> i32 {
        self.solution_type
    }

    /// Java `getIncludeExcludeType`.
    pub(crate) fn get_include_exclude_type(&self) -> i32 {
        self.include_exclude_type
    }

    /// Java `getIncludeExcludeList`.
    pub(crate) fn get_include_exclude_list(&self) -> &StringList {
        &self.include_exclude_list
    }

    /// Java `getInitialImageRotation`.
    pub(crate) fn get_initial_image_rotation(&self) -> f64 {
        self.initial_image_rotation
    }

    /// Java `getRotationAngleSolutionType`.
    pub(crate) fn get_rotation_angle_solution_type(&self) -> i32 {
        self.rotation_angle_solution_type
    }

    /// Java `getNSeparateViewGroups`.
    pub(crate) fn get_n_separate_view_groups(&self) -> i32 {
        self.n_separate_view_groups
    }

    /// Java `getSeparateViewGroups`.
    pub(crate) fn get_separate_view_groups(&self) -> &StringList {
        &self.separate_view_groups
    }

    /// Java `getTiltAngleSpec`.
    pub(crate) fn get_tilt_angle_spec(&self) -> &TiltAngleSpec {
        &self.tilt_angle_spec
    }

    /// Java `getTiltAngleOffset`.
    pub(crate) fn get_tilt_angle_offset(&self) -> f64 {
        self.tilt_angle_offset
    }

    /// Java `getTiltAngleSolution`.
    pub(crate) fn get_tilt_angle_solution(&self) -> &TiltalignSolution {
        &self.tilt_angle_solution
    }

    /// Java `getTiltAngleSolutionParams`.
    pub(crate) fn get_tilt_angle_solution_params(&self) -> String {
        self.tilt_angle_solution.params.to_string()
    }

    /// Java `getTiltAngleSolutionGroupSize`.
    pub(crate) fn get_tilt_angle_solution_group_size(&self) -> i32 {
        self.tilt_angle_solution.params.get_int(0)
    }

    /// Java `getTiltAngleSolutionAdditionalGroups`.
    pub(crate) fn get_tilt_angle_solution_additional_groups(&self) -> String {
        self.tilt_angle_solution.additional_groups.to_string()
    }

    /// Java `getMagnificationSolutionReferenceView`.
    pub(crate) fn get_magnification_solution_reference_view(&self) -> String {
        self.magnification_solution.reference_view.to_string()
    }

    /// Java `getMagnificationSolution`.
    pub(crate) fn get_magnification_solution(&self) -> &TiltalignSolution {
        &self.magnification_solution
    }

    /// Java `getMagnificationSolutionParams`.
    pub(crate) fn get_magnification_solution_params(&self) -> String {
        self.magnification_solution.params.to_string()
    }

    /// Java `getMagnificationSolutionGroupSize`.
    pub(crate) fn get_magnification_solution_group_size(&self) -> i32 {
        self.magnification_solution.params.get_int(0)
    }

    /// Java `getMagnificationSolutionAdditionalGroups`.
    pub(crate) fn get_magnification_solution_additional_groups(&self) -> String {
        self.magnification_solution.additional_groups.to_string()
    }

    /// Java `getCompressionSolutionReferenceView`.
    pub(crate) fn get_compression_solution_reference_view(&self) -> String {
        self.compression_solution.reference_view.to_string()
    }

    /// Java `getCompressionSolutionType`.
    pub(crate) fn get_compression_solution_type(&self) -> i32 {
        self.compression_solution.r#type
    }

    /// Java `getCompressionSolutionParams`.
    pub(crate) fn get_compression_solution_params(&self) -> String {
        self.compression_solution.params.to_string()
    }

    /// Java `getCompressionSolutionGroupSize`.
    pub(crate) fn get_compression_solution_group_size(&self) -> i32 {
        self.compression_solution.params.get_int(0)
    }

    /// Java `getCompressionSolutionAdditionalGroups`.
    pub(crate) fn get_compression_solution_additional_groups(&self) -> String {
        self.compression_solution.additional_groups.to_string()
    }

    /// Java `getDistortionSolutionType`.
    pub(crate) fn get_distortion_solution_type(&self) -> i32 {
        self.distortion_solution_type
    }

    /// Java `getXstretchSolution`.
    pub(crate) fn get_xstretch_solution(&self) -> &TiltalignSolution {
        &self.xstretch_solution
    }

    /// Java `getXstretchSolutionParams`.
    pub(crate) fn get_xstretch_solution_params(&self) -> String {
        self.xstretch_solution.params.to_string()
    }

    /// Java `getXstretchSolutionGroupSize`.
    pub(crate) fn get_xstretch_solution_group_size(&self) -> i32 {
        self.xstretch_solution.params.get_int(0)
    }

    /// Java `getXstretchSolutionAdditionalGroups`.
    pub(crate) fn get_xstretch_solution_additional_groups(&self) -> String {
        self.xstretch_solution.additional_groups.to_string()
    }

    /// Java `getSkewSolution`.
    pub(crate) fn get_skew_solution(&self) -> &TiltalignSolution {
        &self.skew_solution
    }

    /// Java `getSkewSolutionParams`.
    pub(crate) fn get_skew_solution_params(&self) -> String {
        self.skew_solution.params.to_string()
    }

    /// Java `getSkewSolutionGroupSize`.
    pub(crate) fn get_skew_solution_group_size(&self) -> i32 {
        self.skew_solution.params.get_int(0)
    }

    /// Java `getSkewSolutionAdditionalGroups`.
    pub(crate) fn get_skew_solution_additional_groups(&self) -> String {
        self.skew_solution.additional_groups.to_string()
    }

    /// Java `getResidualThreshold`.
    pub(crate) fn get_residual_threshold(&self) -> f64 {
        self.residual_threshold
    }

    /// Java `getNSurfaceAnalysis`.
    pub(crate) fn get_n_surface_analysis(&self) -> i32 {
        self.n_surface_analysis
    }

    /// Java `getMinimizationParams`.
    pub(crate) fn get_minimization_params(&self) -> String {
        self.minimization_params.to_string()
    }

    /// Java `getMetroFactor`.
    pub(crate) fn get_metro_factor(&self) -> f64 {
        self.minimization_params.get_double_index(0)
    }

    /// Java `getCycleLimit`.
    pub(crate) fn get_cycle_limit(&self) -> i32 {
        self.minimization_params.get_int(1)
    }

    /// Java `getTiltAxisZShift`.
    pub(crate) fn get_tilt_axis_z_shift(&self) -> f64 {
        self.tilt_axis_z_shift
    }

    /// Java `getTiltAxisXShift`.
    pub(crate) fn get_tilt_axis_x_shift(&self) -> f64 {
        self.tilt_axis_x_shift
    }

    /// Java `getLocalAlignments`.
    pub(crate) fn get_local_alignments(&self) -> bool {
        self.local_alignments
    }

    /// Java `getLocalTransformFile`.
    pub(crate) fn get_local_transform_file(&self) -> Option<&str> {
        self.local_transform_file.as_deref()
    }

    /// Java `getNLocalPatches`.
    pub(crate) fn get_n_local_patches(&self) -> &FortranInputString {
        &self.n_local_patches
    }

    /// Java `getMinLocalPatchSize`.
    pub(crate) fn get_min_local_patch_size(&self) -> &FortranInputString {
        &self.min_local_patch_size
    }

    /// Java `getMinLocalFiducials`.
    pub(crate) fn get_min_local_fiducials(&self) -> &FortranInputString {
        &self.min_local_fiducials
    }

    /// Java `getFixLocalFiducialCoodinates`.
    pub(crate) fn get_fix_local_fiducial_coodinates(&self) -> bool {
        self.fix_local_fiducial_coodinates
    }

    /// Java `getLocalOutputSelection`.
    pub(crate) fn get_local_output_selection(&self) -> &FortranInputString {
        &self.local_output_selection
    }

    /// Java `getLocalRotationSolution`.
    pub(crate) fn get_local_rotation_solution(&self) -> &TiltalignSolution {
        &self.local_rotation_solution
    }

    /// Java `getLocalRotationSolutionType`.
    pub(crate) fn get_local_rotation_solution_type(&self) -> i32 {
        self.local_rotation_solution.r#type
    }

    /// Java `getLocalRotationSolutionGroupSize`.
    pub(crate) fn get_local_rotation_solution_group_size(&self) -> i32 {
        self.local_rotation_solution.params.get_int(0)
    }

    /// Java `getLocalRotationSolutionParams`.
    pub(crate) fn get_local_rotation_solution_params(&self) -> &FortranInputString {
        &self.local_rotation_solution.params
    }

    /// Java `getLocalRotationAdditionalGroups`.
    pub(crate) fn get_local_rotation_additional_groups(&self) -> &StringList {
        &self.local_rotation_solution.additional_groups
    }

    /// Java `getLocalTiltSolution`.
    pub(crate) fn get_local_tilt_solution(&self) -> &TiltalignSolution {
        &self.local_tilt_solution
    }

    /// Java `getLocalTiltSolutionGroupSize`.
    pub(crate) fn get_local_tilt_solution_group_size(&self) -> i32 {
        self.local_tilt_solution.params.get_int(0)
    }

    /// Java `getLocalTiltSolutionParams`.
    pub(crate) fn get_local_tilt_solution_params(&self) -> String {
        self.local_tilt_solution.params.to_string()
    }

    /// Java `getLocalTiltAdditionalGroups`.
    pub(crate) fn get_local_tilt_additional_groups(&self) -> String {
        self.local_tilt_solution.additional_groups.to_string()
    }

    /// Java `getLocalMagnificationSolutionReferenceView`.
    pub(crate) fn get_local_magnification_solution_reference_view(&self) -> String {
        self.local_magnification_solution.reference_view.to_string()
    }

    /// Java `getLocalMagnificationSolution`.
    pub(crate) fn get_local_magnification_solution(&self) -> &TiltalignSolution {
        &self.local_magnification_solution
    }

    /// Java `getLocalMagnificationSolutionParams`.
    pub(crate) fn get_local_magnification_solution_params(&self) -> String {
        self.local_magnification_solution.params.to_string()
    }

    /// Java `getLocalMagnificationSolutionGroupSize`.
    pub(crate) fn get_local_magnification_solution_group_size(&self) -> i32 {
        self.local_magnification_solution.params.get_int(0)
    }

    /// Java `getLocalMagnificationSolutionAdditionalGroups`.
    pub(crate) fn get_local_magnification_solution_additional_groups(&self) -> String {
        self.local_magnification_solution
            .additional_groups
            .to_string()
    }

    /// Java `getLocalMagnificationAdditionalGroups`.
    pub(crate) fn get_local_magnification_additional_groups(&self) -> String {
        self.local_magnification_solution
            .additional_groups
            .to_string()
    }

    /// Java `getLocalDistortionSolutionType`.
    pub(crate) fn get_local_distortion_solution_type(&self) -> i32 {
        self.local_distortion_solution_type
    }

    /// Java `getLocalXstretchSolution`.
    pub(crate) fn get_local_xstretch_solution(&self) -> &TiltalignSolution {
        &self.local_xstretch_solution
    }

    /// Java `getLocalXstretchSolutionParams`.
    pub(crate) fn get_local_xstretch_solution_params(&self) -> String {
        self.local_xstretch_solution.params.to_string()
    }

    /// Java `getLocalXstretchSolutionGroupSize`.
    pub(crate) fn get_local_xstretch_solution_group_size(&self) -> i32 {
        self.local_xstretch_solution.params.get_int(0)
    }

    /// Java `getLocalXstretchSolutionAdditionalGroups`.
    pub(crate) fn get_local_xstretch_solution_additional_groups(&self) -> String {
        self.local_xstretch_solution.additional_groups.to_string()
    }

    /// Java `getLocalSkewSolution`.
    pub(crate) fn get_local_skew_solution(&self) -> &TiltalignSolution {
        &self.local_skew_solution
    }

    /// Java `getLocalSkewSolutionParams`.
    pub(crate) fn get_local_skew_solution_params(&self) -> String {
        self.local_skew_solution.params.to_string()
    }

    /// Java `getLocalSkewSolutionGroupSize`.
    pub(crate) fn get_local_skew_solution_group_size(&self) -> i32 {
        self.local_skew_solution.params.get_int(0)
    }

    /// Java `getLocalSkewSolutionAdditionalGroups`.
    pub(crate) fn get_local_skew_solution_additional_groups(&self) -> String {
        self.local_skew_solution.additional_groups.to_string()
    }
}

/// Java `toString`.  Return a string representation of the values in the object.
impl std::fmt::Display for OldTiltalignParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let null = |value: &Option<String>| value.clone().unwrap_or_else(|| "null".to_owned());
        let mut buffer = String::new();

        buffer.push_str("\nModel file: ");
        buffer.push_str(&null(&self.model_file));

        buffer.push_str("\nImage file: ");
        buffer.push_str(&null(&self.image_file));

        buffer.push_str("\nImage parameters: ");
        buffer.push_str(&self.image_parameters.to_string());

        buffer.push_str("\nIMOD Fiducial Pos File: ");
        buffer.push_str(&null(&self.imod_fiducial_pos_file));

        buffer.push_str("\nASCII Fiducial Pos File: ");
        buffer.push_str(&null(&self.ascii_fiducial_pos_file));

        buffer.push_str("\nTilt Angle Solution File: ");
        buffer.push_str(&null(&self.tilt_angle_solution_file));

        buffer.push_str("\n Transform Solution File: ");
        buffer.push_str(&null(&self.transform_solution_file));

        buffer.push_str("\nSolution Type: ");
        buffer.push_str(&self.solution_type.to_string());

        buffer.push_str("\nInclude Points: ");
        buffer.push_str(&self.include_exclude_type.to_string());

        buffer.push_str("\nInclude Points ZRange: ");
        buffer.push_str(&self.include_exclude_list.to_string());

        buffer.push_str("\nInitial Image Rotation: ");
        buffer.push_str(&java_lang_double_to_string(self.initial_image_rotation));

        buffer.push_str("\nRotation Angle Solution Type: ");
        buffer.push_str(&self.rotation_angle_solution_type.to_string());

        buffer.push_str("\nN Additional View Sets: ");
        buffer.push_str(&self.n_separate_view_groups.to_string());

        buffer.push_str("\nadditionalViewGroups: ");
        buffer.push_str(&self.separate_view_groups.to_string());

        buffer.push_str("\ntiltAngleSpec: ");
        buffer.push_str(&self.tilt_angle_spec.to_string());

        buffer.push_str("\ntiltAngleOffset: ");
        buffer.push_str(&java_lang_double_to_string(self.tilt_angle_offset));

        buffer.push_str("\ntiltAngleSolution.type: ");
        buffer.push_str(&self.tilt_angle_solution.r#type.to_string());

        buffer.push_str("\ntiltAngleSolution.params: ");
        buffer.push_str(&self.tilt_angle_solution.params.to_string());

        // OldTiltalignParam.java:825 writes "\tiltAngleSolution..." - a tab and
        // "iltAngleSolution" - where every other label starts with "\n".  Typo fixed in
        // translation.
        buffer.push_str("\ntiltAngleSolution.additionalGroups: ");
        buffer.push_str(&self.tilt_angle_solution.additional_groups.to_string());

        buffer.push_str("\nmagnificationSolution.referenceView: ");
        buffer.push_str(&self.magnification_solution.reference_view.to_string());

        buffer.push_str("\nmagnificationSolution.type: ");
        buffer.push_str(&self.magnification_solution.r#type.to_string());

        buffer.push_str("\nmagnificationSolution.params: ");
        buffer.push_str(&self.magnification_solution.params.to_string());

        buffer.push_str("\nmagnificationSolution.additionalGroups: ");
        buffer.push_str(&self.magnification_solution.additional_groups.to_string());

        buffer.push_str("\ncompressionSolution.referenceView: ");
        buffer.push_str(&self.compression_solution.reference_view.to_string());

        buffer.push_str("\ncompressionSolution.type: ");
        buffer.push_str(&self.compression_solution.r#type.to_string());

        buffer.push_str("\ncompressionSolution.params: ");
        buffer.push_str(&self.compression_solution.params.to_string());

        buffer.push_str("\ncompressionSolution.additionalGroups: ");
        buffer.push_str(&self.compression_solution.additional_groups.to_string());

        buffer.push_str("\ndistortionSolutionType: ");
        buffer.push_str(&self.distortion_solution_type.to_string());

        buffer.push_str("\nxstretchSolution.type: ");
        buffer.push_str(&self.xstretch_solution.r#type.to_string());

        buffer.push_str("\nxstretchSolution.params: ");
        buffer.push_str(&self.xstretch_solution.params.to_string());

        buffer.push_str("\nxstretchSolution.additionalGroups: ");
        buffer.push_str(&self.xstretch_solution.additional_groups.to_string());

        buffer.push_str("\nskewSolution.type: ");
        buffer.push_str(&self.skew_solution.r#type.to_string());

        buffer.push_str("\nskewSolution.params: ");
        buffer.push_str(&self.skew_solution.params.to_string());

        buffer.push_str("\nskewSolution.additionalGroups: ");
        buffer.push_str(&self.skew_solution.additional_groups.to_string());

        buffer.push_str("\nresidualThreshold: ");
        buffer.push_str(&java_lang_double_to_string(self.residual_threshold));

        buffer.push_str("\nnSurfaceAnalysis: ");
        buffer.push_str(&self.n_surface_analysis.to_string());

        buffer.push_str("\nminimizationParams: ");
        buffer.push_str(&self.minimization_params.to_string());

        buffer.push_str("\nlocalAlignments: ");
        buffer.push_str(&self.local_alignments.to_string());

        buffer.push_str("\ntiltAxisZShift: ");
        buffer.push_str(&java_lang_double_to_string(self.tilt_axis_z_shift));

        buffer.push_str("\ntiltAxisXShift: ");
        buffer.push_str(&java_lang_double_to_string(self.tilt_axis_x_shift));

        buffer.push_str("\nlocalTransformFile: ");
        buffer.push_str(&null(&self.local_transform_file));

        buffer.push_str("\nnLocalPatches: ");
        buffer.push_str(&self.n_local_patches.to_string());

        buffer.push_str("\nminLocalPatchSize: ");
        buffer.push_str(&self.min_local_patch_size.to_string());

        buffer.push_str("\nminLocalFiducials: ");
        buffer.push_str(&self.min_local_fiducials.to_string());

        buffer.push_str("\nfixLocalFiducialCoodinates: ");
        buffer.push_str(&self.fix_local_fiducial_coodinates.to_string());

        buffer.push_str("\nlocalOutputSelection: ");
        buffer.push_str(&self.local_output_selection.to_string());

        buffer.push_str("\nlocalRotationSolution.type: ");
        buffer.push_str(&self.local_rotation_solution.r#type.to_string());

        buffer.push_str("\nlocalRotationSolution.params: ");
        buffer.push_str(&self.local_rotation_solution.params.to_string());

        buffer.push_str("\nlocalRotationSolution.additionalGroups: ");
        buffer.push_str(&self.local_rotation_solution.additional_groups.to_string());

        buffer.push_str("\nlocalTiltSolution.type: ");
        buffer.push_str(&self.local_tilt_solution.r#type.to_string());

        buffer.push_str("\nlocalTiltSolution.params: ");
        buffer.push_str(&self.local_tilt_solution.params.to_string());

        buffer.push_str("\nlocalTiltSolution.additionalGroups: ");
        buffer.push_str(&self.local_tilt_solution.additional_groups.to_string());

        buffer.push_str("\nlocalMagnificationReferenceView: ");
        buffer.push_str(&self.local_magnification_solution.reference_view.to_string());

        buffer.push_str("\nlocalMagnificationSolution.type: ");
        buffer.push_str(&self.local_magnification_solution.r#type.to_string());

        buffer.push_str("\nlocalMagnificationSolution.params: ");
        buffer.push_str(&self.local_magnification_solution.params.to_string());

        buffer.push_str("\nlocalMagnificationSolution.additionalGroups: ");
        buffer.push_str(
            &self
                .local_magnification_solution
                .additional_groups
                .to_string(),
        );

        buffer.push_str("\nlocalDistortionSolutionType: ");
        buffer.push_str(&self.local_distortion_solution_type.to_string());

        buffer.push_str("\nlocalXstretchSolution.type: ");
        buffer.push_str(&self.local_xstretch_solution.r#type.to_string());

        buffer.push_str("\nlocalXstretchSolution.params: ");
        buffer.push_str(&self.local_xstretch_solution.params.to_string());

        buffer.push_str("\nlocalXstretchSolution.additionalGroups: ");
        buffer.push_str(&self.local_xstretch_solution.additional_groups.to_string());

        buffer.push_str("\nlocalSkewSolution.type: ");
        buffer.push_str(&self.local_skew_solution.r#type.to_string());

        buffer.push_str("\nlocalSkewSolution.params: ");
        buffer.push_str(&self.local_skew_solution.params.to_string());

        buffer.push_str("\nlocalSkewSolution.additionalGroups: ");
        buffer.push_str(&self.local_skew_solution.additional_groups.to_string());

        buffer.push('\n');
        f.write_str(&buffer)
    }
}
