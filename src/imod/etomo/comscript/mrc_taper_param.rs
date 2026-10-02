//! `IMOD/Etomo/src/etomo/comscript/MrcTaperParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "mrctaper";

/// Java final `MrcTaperParam implements CommandParam`.
pub struct MrcTaperParam {
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    input_file: Option<String>,
}

impl MrcTaperParam {
    /// Java package-private `MrcTaperParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> MrcTaperParam {
        MrcTaperParam {
            manager,
            axis_id,
            input_file: None,
        }
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.input_file = None;
    }

    /// Java `setInputFile`.
    pub fn set_input_file(&mut self, input: Option<&str>) {
        self.input_file = input.map(|input| input.to_string());
    }
}

impl CommandParam for MrcTaperParam {
    /// Java `parseComScriptCommand`.  Get the parameters from the ComScriptCommand.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // MrcTaperParam.java:57-59 dereferences a null argument array and a null
        // element (NullPointerException); a missing array is read here as empty and a
        // null element is skipped.
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.reset();
        for arg in cmd_line_args.iter() {
            let arg = match arg {
                None => continue,
                Some(arg) => arg,
            };
            // Is it an argument or filename
            if arg.starts_with('-') {
            }
            // input and output filename arguments
            else {
                // (commented out in the source: the last argument as the output file)
                self.input_file = Some(arg.clone());
            }
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update the script command with the current
    /// values of this object.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Create a new command line argument array
        let mut cmd_line_args: Vec<Option<String>> = Vec::with_capacity(1);
        if let Some(input_file) = &self.input_file {
            cmd_line_args.push(Some(input_file.clone()));
        }
        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.input_file = file_type::CLASS
            .aligned_stack
            .get_file_name(Some(self.manager), Some(self.axis_id));
    }
}
