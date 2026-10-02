//! `IMOD/Etomo/src/etomo/comscript/MatchvolParam.java`.

use std::cell::RefCell;
use std::rc::Rc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command_param::{CommandParam, ParseComScriptError};
use super::fortran_input_string::FortranInputString;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `COMMAND`.
pub const COMMAND: &str = "matchvol";

/// Java `MatchvolParam`.
#[derive(Clone, Debug)]
pub struct MatchvolParam {
    /// Java `outputSizeXYZ`.
    output_size_xyz: FortranInputString,
}

impl Default for MatchvolParam {
    fn default() -> MatchvolParam {
        MatchvolParam::new()
    }
}

impl MatchvolParam {
    /// Java `MatchvolParam()`.
    pub fn new() -> MatchvolParam {
        let mut instance = MatchvolParam {
            output_size_xyz: FortranInputString::new_with_key(Some("OutputSizeXYZ"), 3),
        };
        instance.output_size_xyz.set_integer_type(true);
        instance.reset();
        instance
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.output_size_xyz.reset();
    }

    /// Java private `getInputArguments`.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        let command = script_command.get_command();
        // Check to be sure that it is the right command
        if command != Some(COMMAND) {
            return Err(BadComScriptException::new(&format!(
                "Not a {COMMAND} command"
            )));
        }
        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        Ok(input_args)
    }

    /// Java `setOutputSizeY`.
    pub fn set_output_size_y(&mut self, output_size_y: Option<&str>) {
        self.output_size_xyz.set_index_string(1, output_size_y);
    }

    /// Java `getOutputSizeY`.
    pub fn get_output_size_y(&self) -> i32 {
        self.output_size_xyz.get_int(1)
    }
}

impl CommandParam for MatchvolParam {
    /// Java `parseComScriptCommand`.  Java reads `getCommandLineArgs()` into an
    /// unused local first.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let _cmd_line_args = script_command.get_command_line_args();
        self.reset();
        self.output_size_xyz
            .validate_and_set_com_script(script_command)?;
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
        self.output_size_xyz.update_script_parameter(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
