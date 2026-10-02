//! `IMOD/Etomo/src/etomo/comscript/SolvematchshiftParam.java`.
//!
//! `SolvematchshiftParam extends ConstSolvematchshiftParam`: the superclass state
//! is the `base` field, reached through `Deref`/`DerefMut`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_solvematchshift_param::ConstSolvematchshiftParam;
use super::fortran_input_string::FortranInputString;
use super::solvematch_param::{AFID_XYZ, BFID_XYZ};
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_value_of, java_lang_integer_parse_int, java_lang_string_matches_whitespace,
};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `SolvematchshiftParam`.
#[derive(Clone, Debug)]
pub struct SolvematchshiftParam {
    /// Java superclass `ConstSolvematchshiftParam` state.
    pub base: ConstSolvematchshiftParam,
}

impl std::ops::Deref for SolvematchshiftParam {
    type Target = ConstSolvematchshiftParam;

    fn deref(&self) -> &ConstSolvematchshiftParam {
        &self.base
    }
}

impl std::ops::DerefMut for SolvematchshiftParam {
    fn deref_mut(&mut self) -> &mut ConstSolvematchshiftParam {
        &mut self.base
    }
}

impl Default for SolvematchshiftParam {
    fn default() -> SolvematchshiftParam {
        SolvematchshiftParam::new()
    }
}

impl SolvematchshiftParam {
    /// Java's implicit `SolvematchshiftParam()`.
    pub fn new() -> SolvematchshiftParam {
        SolvematchshiftParam {
            base: ConstSolvematchshiftParam::new(),
        }
    }

    /// Java package-private `setMatchBToA`.
    pub(crate) fn set_match_b_to_a(&mut self, to_file: Option<&str>) {
        let to_file = match to_file {
            None => return,
            Some(to_file) if java_lang_string_matches_whitespace(to_file) => return,
            Some(to_file) => to_file,
        };
        if AFID_XYZ.is_match(to_file) {
            self.base.match_b_to_a = true;
        } else if BFID_XYZ.is_match(to_file) {
            self.base.match_b_to_a = false;
        }
    }

    /// Java `setFiducialMatchListA`.
    pub fn set_fiducial_match_list_a(&mut self, list: Option<&str>) {
        self.base.fiducial_match_list_a.parse_string(list);
    }

    /// Java `setFiducialMatchListB`.
    pub fn set_fiducial_match_list_b(&mut self, list: Option<&str>) {
        self.base.fiducial_match_list_b.parse_string(list);
    }

    /// Java `setFromFiducialCoordinatesFile`.
    pub fn set_from_fiducial_coordinates_file(
        &mut self,
        from_fiducial_coordinates_file: Option<&str>,
    ) {
        self.base.from_fiducial_coordinates_file =
            from_fiducial_coordinates_file.map(str::to_owned);
    }

    /// Java `setNSurfaces`.
    pub fn set_n_surfaces(&mut self, n_surfaces: i32) {
        self.base.n_surfaces = n_surfaces;
    }

    /// Java `setOutputTransformationFile`.
    pub fn set_output_transformation_file(&mut self, output_transformation_file: Option<&str>) {
        self.base.output_transformation_file = output_transformation_file.map(str::to_owned);
    }

    /// Java `setResidualThreshold`.
    pub fn set_residual_threshold(&mut self, residual_threshold: f64) {
        self.base.residual_threshold = residual_threshold;
    }

    /// Java `setToFiducialCoordinatesFile`.
    pub fn set_to_fiducial_coordinates_file(&mut self, to_fiducial_coordinates_file: Option<&str>) {
        self.base.to_fiducial_coordinates_file = to_fiducial_coordinates_file.map(str::to_owned);
    }

    /// Java `setXAxistTilt`.
    pub fn set_x_axist_tilt(&mut self, x_axist_tilt: FortranInputString) {
        self.base.x_axist_tilt = x_axist_tilt;
    }
}

impl CommandParam for SolvematchshiftParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Check to be sure that it is a solvematch command
        if script_command.get_command() != Some("solvematch") {
            return Err(BadComScriptException::new("Not a solvematch command").into());
        }
        // Extract the parameters
        let input_args = script_command.get_input_arguments();
        if input_args.len() != 8 {
            return Err(BadComScriptException::new(&format!(
                "Incorrect number of input arguments to solvematch command\nGot {} expected 8.",
                input_args.len()
            ))
            .into());
        }
        let mut i = 0;
        self.base.to_fiducial_coordinates_file =
            input_args[i].borrow().get_argument().map(str::to_owned);
        i += 1;
        let to_file = self.base.to_fiducial_coordinates_file.clone();
        self.set_match_b_to_a(to_file.as_deref());
        self.base.from_fiducial_coordinates_file =
            input_args[i].borrow().get_argument().map(str::to_owned);
        i += 1;
        if self.base.match_b_to_a {
            self.base
                .fiducial_match_list_a
                .parse_string(input_args[i].borrow().get_argument());
            i += 1;
            self.base
                .fiducial_match_list_b
                .parse_string(input_args[i].borrow().get_argument());
            i += 1;
        } else {
            self.base
                .fiducial_match_list_b
                .parse_string(input_args[i].borrow().get_argument());
            i += 1;
            self.base
                .fiducial_match_list_a
                .parse_string(input_args[i].borrow().get_argument());
            i += 1;
        }
        self.base
            .x_axist_tilt
            .validate_and_set(input_args[i].borrow().get_argument())?;
        i += 1;
        // `Double.parseDouble` and `Integer.parseInt` throw the unchecked
        // NumberFormatException (NullPointerException for a null double), which the
        // callers catch; it is the NumberFormat error here.
        self.base.residual_threshold = match input_args[i].borrow().get_argument() {
            None => {
                return Err(ParseComScriptError::NumberFormat(
                    "java.lang.NullPointerException".to_owned(),
                ));
            }
            Some(argument) => {
                java_lang_double_value_of(argument).map_err(ParseComScriptError::NumberFormat)?
            }
        };
        i += 1;
        self.base.n_surfaces = match input_args[i].borrow().get_argument() {
            None => return Err(ParseComScriptError::NumberFormat("null".to_owned())),
            Some(argument) => {
                java_lang_integer_parse_int(argument).map_err(ParseComScriptError::NumberFormat)?
            }
        };
        i += 1;
        self.base.output_transformation_file =
            input_args[i].borrow().get_argument().map(str::to_owned);
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    ///
    /// Java's `setMatchBToA(inputArgs[0].getArgument())` mutates `matchBToA`; the
    /// value it would compute is used locally here because this method takes
    /// `&self`.  The recomputation reads only the script, so any later call
    /// recomputes the same value before using it.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Check to be sure that it is a solvematch command
        if script_command.get_command() != Some("solvematch") {
            return Err(BadComScriptException::new("Not a solvematch command"));
        }
        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        if input_args.len() != 8 {
            return Err(BadComScriptException::new(&format!(
                "Incorrect number of input arguments to solvematch command\nGot {} expected 8.",
                input_args.len()
            )));
        }
        // matchBToA has to be set correctly. In case parseComScriptCommand() hasn't
        // been called, set is here to.
        let mut this = self.clone();
        let first = input_args[0].borrow().get_argument().map(str::to_owned);
        this.set_match_b_to_a(first.as_deref());
        let match_b_to_a = this.base.match_b_to_a;
        // Fill in the input argument sequence
        input_args[0]
            .borrow_mut()
            .set_argument(self.base.to_fiducial_coordinates_file.as_deref());
        script_command.set_input_argument(0, &input_args[0].borrow());
        input_args[1]
            .borrow_mut()
            .set_argument(self.base.from_fiducial_coordinates_file.as_deref());
        script_command.set_input_argument(1, &input_args[1].borrow());
        if match_b_to_a {
            input_args[2]
                .borrow_mut()
                .set_argument(Some(&self.base.fiducial_match_list_a.to_string()));
            script_command.set_input_argument(2, &input_args[2].borrow());
            input_args[3]
                .borrow_mut()
                .set_argument(Some(&self.base.fiducial_match_list_b.to_string()));
            script_command.set_input_argument(3, &input_args[3].borrow());
        } else {
            input_args[2]
                .borrow_mut()
                .set_argument(Some(&self.base.fiducial_match_list_b.to_string()));
            script_command.set_input_argument(2, &input_args[2].borrow());
            input_args[3]
                .borrow_mut()
                .set_argument(Some(&self.base.fiducial_match_list_a.to_string()));
            script_command.set_input_argument(3, &input_args[3].borrow());
        }
        input_args[4]
            .borrow_mut()
            .set_argument(Some(&self.base.x_axist_tilt.to_string()));
        script_command.set_input_argument(4, &input_args[4].borrow());
        input_args[5]
            .borrow_mut()
            .set_argument_double(self.base.residual_threshold);
        script_command.set_input_argument(5, &input_args[5].borrow());
        input_args[6]
            .borrow_mut()
            .set_argument_int(self.base.n_surfaces);
        script_command.set_input_argument(6, &input_args[6].borrow());
        input_args[7]
            .borrow_mut()
            .set_argument(self.base.output_transformation_file.as_deref());
        script_command.set_input_argument(7, &input_args[7].borrow());
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
