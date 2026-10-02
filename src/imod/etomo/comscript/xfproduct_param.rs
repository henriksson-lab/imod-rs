//! `IMOD/Etomo/src/etomo/comscript/XfproductParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_xfproduct_param::ConstXfproductParam;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `XfproductParam extends ConstXfproductParam implements CommandParam`.
#[derive(Clone, Debug, Default)]
pub struct XfproductParam {
    /// Java superclass `ConstXfproductParam` state.
    pub base: ConstXfproductParam,
}

/// Java inheritance: every `ConstXfproductParam` member is reachable on an
/// `XfproductParam`.
impl std::ops::Deref for XfproductParam {
    type Target = ConstXfproductParam;

    fn deref(&self) -> &ConstXfproductParam {
        &self.base
    }
}

impl std::ops::DerefMut for XfproductParam {
    fn deref_mut(&mut self) -> &mut ConstXfproductParam {
        &mut self.base
    }
}

impl XfproductParam {
    /// Java's implicit `XfproductParam()`.
    pub fn new() -> XfproductParam {
        XfproductParam {
            base: ConstXfproductParam::new(),
        }
    }

    /// Java `setInputFile1`.
    pub fn set_input_file1(&mut self, input_file1: Option<&str>) {
        self.base.input_file1 = input_file1.map(str::to_owned);
    }

    /// Java `setInputFile2`.
    pub fn set_input_file2(&mut self, input_file2: Option<&str>) {
        self.base.input_file2 = input_file2.map(str::to_owned);
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, output_file: Option<&str>) {
        self.base.output_file = output_file.map(str::to_owned);
    }

    /// Java `setScaleShifts(int)`.
    pub fn set_scale_shifts(&mut self, binning: i32) -> Result<(), FortranInputSyntaxException> {
        if binning > 1 {
            self.base
                .scale_shifts
                .validate_and_set(Some(&format!("1,{binning}")))?;
        } else {
            self.base.scale_shifts.validate_and_set(Some("/"))?;
        }
        Ok(())
    }
}

impl CommandParam for XfproductParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Check to be sure that it is a xfproduct command.  A null command is a
        // NullPointerException in Java (`getCommand().equals`); here it is not an
        // xfproduct command.
        if script_command.get_command() != Some("xfproduct") {
            return Err(BadComScriptException::new("Not a xfproduct command").into());
        }
        let input_args = script_command.get_input_arguments();
        let _cmd_line_args = script_command.get_command_line_args();
        if script_command.is_keyword_value_pairs() {
            self.base.input_file1 = script_command.get_value(Some("InputFile1"))?;
            self.base.input_file2 = script_command.get_value(Some("InputFile2"))?;
            self.base.output_file = script_command.get_value(Some("OutputFile"))?;
            let scale_shifts = script_command.get_value(Some("ScaleShifts"))?;
            self.base
                .scale_shifts
                .validate_and_set(scale_shifts.as_deref())?;
        } else {
            // `inputArgs[0..2]` throw ArrayIndexOutOfBoundsException in Java when the
            // script has fewer than three input lines; a missing line is read as a
            // null argument here.
            let argument = |index: usize| {
                input_args
                    .get(index)
                    .and_then(|input_arg| input_arg.borrow().get_argument().map(str::to_owned))
            };
            self.base.input_file1 = argument(0);
            self.base.input_file2 = argument(1);
            self.base.output_file = argument(2);
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Check to be sure that it is a ccderaser xommand
        if script_command.get_command() != Some("xfproduct") {
            return Err(BadComScriptException::new("Not a xfproduct command"));
        }
        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        // A null file name is a NullPointerException in Java (`inputFile1.equals("")`);
        // here it is treated as empty, so the key is deleted.
        match self.base.input_file1.as_deref() {
            Some(input_file1) if input_file1 != "" => {
                script_command.set_value(Some("InputFile1"), Some(input_file1));
            }
            _ => {
                script_command.delete_key(Some("InputFile1"));
            }
        }
        match self.base.input_file2.as_deref() {
            Some(input_file2) if input_file2 != "" => {
                script_command.set_value(Some("InputFile2"), Some(input_file2));
            }
            _ => {
                script_command.delete_key(Some("InputFile2"));
            }
        }
        match self.base.output_file.as_deref() {
            Some(output_file) if output_file != "" => {
                script_command.set_value(Some("OutputFile"), Some(output_file));
            }
            _ => {
                script_command.delete_key(Some("OutputFile"));
            }
        }
        if self.base.scale_shifts.values_set() && !self.base.scale_shifts.is_default() {
            script_command.set_value(
                Some("ScaleShifts"),
                Some(&self.base.scale_shifts.to_string()),
            );
        } else {
            script_command.delete_key(Some("ScaleShifts"));
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
