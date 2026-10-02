//! `IMOD/Etomo/src/etomo/comscript/MatchshiftsParam.java`.
//!
//! `MatchshiftsParam extends ConstMatchshiftsParam`: the superclass state is the
//! `base` field, reached through `Deref`/`DerefMut`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_matchshifts_param::ConstMatchshiftsParam;
use super::invalid_parameter_exception::InvalidParameterException;
use super::param_utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java `MatchshiftsParam`.
#[derive(Clone, Debug)]
pub struct MatchshiftsParam {
    /// Java superclass `ConstMatchshiftsParam` state.
    pub base: ConstMatchshiftsParam,
}

impl std::ops::Deref for MatchshiftsParam {
    type Target = ConstMatchshiftsParam;

    fn deref(&self) -> &ConstMatchshiftsParam {
        &self.base
    }
}

impl std::ops::DerefMut for MatchshiftsParam {
    fn deref_mut(&mut self) -> &mut ConstMatchshiftsParam {
        &mut self.base
    }
}

impl Default for MatchshiftsParam {
    fn default() -> MatchshiftsParam {
        MatchshiftsParam::new()
    }
}

impl MatchshiftsParam {
    /// Java's implicit `MatchshiftsParam()`.
    pub fn new() -> MatchshiftsParam {
        MatchshiftsParam {
            base: ConstMatchshiftsParam::new(),
        }
    }
}

impl CommandParam for MatchshiftsParam {
    /// Java `parseComScriptCommand`.
    ///
    /// A null argument array (Java NullPointerException) reads as empty here, so it
    /// takes the missing-parameter path.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        if cmd_line_args.len() < 5 {
            return Err(
                InvalidParameterException::new(Some("Matchshifts:  Missing parameter.")).into(),
            );
        }
        let mut i = 0;
        self.base.root_name1 = cmd_line_args[i].clone();
        i += 1;
        self.base.root_name2 = cmd_line_args[i].clone();
        i += 1;
        self.base.x_dim = param_utilities::parse_int(cmd_line_args[i].as_deref())
            .map_err(ParseComScriptError::NumberFormat)?;
        i += 1;
        self.base.y_dim = param_utilities::parse_int(cmd_line_args[i].as_deref())
            .map_err(ParseComScriptError::NumberFormat)?;
        i += 1;
        self.base.z_dim = param_utilities::parse_int(cmd_line_args[i].as_deref())
            .map_err(ParseComScriptError::NumberFormat)?;
        i += 1;
        if cmd_line_args.len() >= 6 {
            self.base.xf_in = cmd_line_args[i].clone();
            i += 1;
        }
        if cmd_line_args.len() >= 7 {
            self.base.xf_out = cmd_line_args[i].clone();
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Create a new command line argument array
        let mut cmd_line_args: Vec<Option<String>> = Vec::with_capacity(20);
        if !param_utilities::is_empty(self.base.root_name1.as_deref()) {
            cmd_line_args.push(self.base.root_name1.clone());
        }
        if !param_utilities::is_empty(self.base.root_name2.as_deref()) {
            cmd_line_args.push(self.base.root_name2.clone());
        }
        if self.base.x_dim != i32::MIN {
            cmd_line_args.push(Some(param_utilities::value_of_int(self.base.x_dim)));
        }
        if self.base.y_dim != i32::MIN {
            cmd_line_args.push(Some(param_utilities::value_of_int(self.base.y_dim)));
        }
        if self.base.z_dim != i32::MIN {
            cmd_line_args.push(Some(param_utilities::value_of_int(self.base.z_dim)));
        }
        if !param_utilities::is_empty(self.base.xf_in.as_deref()) {
            cmd_line_args.push(self.base.xf_in.clone());
        }
        if !param_utilities::is_empty(self.base.xf_out.as_deref()) {
            cmd_line_args.push(self.base.xf_out.clone());
        }
        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
