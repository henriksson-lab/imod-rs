//! `IMOD/Etomo/src/etomo/comscript/LabelParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java package-private final `LabelParam`.
pub struct LabelParam {
    label: String,
}

impl LabelParam {
    /// Java package-private `LabelParam(ProcessName)`.
    pub fn new(process_name: ProcessName) -> LabelParam {
        LabelParam {
            label: process_name.to_string(),
        }
    }

    /// Java package-private `getLabel`.
    pub fn get_label(&self) -> String {
        format!("{}:", self.label)
    }
}

impl CommandParam for LabelParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        _script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Replace the parameters of the
    /// ComScriptCommand with the current CommandParameter object's parameters.
    fn update_com_script_command(
        &self,
        _script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
