//! `IMOD/Etomo/src/etomo/comscript/DualvolmatchParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java `MAXIMUM_RESIDUAL`.
pub const MAXIMUM_RESIDUAL: &str = "MaximumResidual";
/// Java `CENTER_SHIFT_LIMIT`.
pub const CENTER_SHIFT_LIMIT: &str = "CenterShiftLimit";

/// Java final `DualvolmatchParam`.
#[derive(Clone, Debug)]
pub struct DualvolmatchParam {
    /// Java `maximumResidual`.
    maximum_residual: ScriptParameter,
    /// Java `centerShiftLimit`.
    center_shift_limit: ScriptParameter,
}

impl Default for DualvolmatchParam {
    fn default() -> DualvolmatchParam {
        DualvolmatchParam::new()
    }
}

impl DualvolmatchParam {
    /// Java package-private `DualvolmatchParam()`.
    pub fn new() -> DualvolmatchParam {
        DualvolmatchParam {
            maximum_residual: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MAXIMUM_RESIDUAL,
            ),
            center_shift_limit: ScriptParameter::new_with_type_and_name(
                Type::Double,
                CENTER_SHIFT_LIMIT,
            ),
        }
    }

    /// Java `getMaximumResidual`.
    pub fn get_maximum_residual(&self) -> String {
        self.maximum_residual.to_string()
    }

    /// Java `getCenterShiftLimit`.
    pub fn get_center_shift_limit(&self) -> String {
        self.center_shift_limit.to_string()
    }

    /// Java `setMaximumResidual`.
    pub fn set_maximum_residual(&mut self, input: Option<&str>) {
        self.maximum_residual.set_string(input);
    }

    /// Java `setCenterShiftLimit`.
    pub fn set_center_shift_limit(&mut self, input: Option<&str>) {
        self.center_shift_limit.set_string(input);
    }
}

impl CommandParam for DualvolmatchParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.maximum_residual.reset();
        self.center_shift_limit.reset();
    }

    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.initialize_defaults();
        self.maximum_residual.parse(script_command)?;
        self.center_shift_limit.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.maximum_residual.update_com_script(script_command);
        self.center_shift_limit.update_com_script(script_command);
        Ok(())
    }
}
