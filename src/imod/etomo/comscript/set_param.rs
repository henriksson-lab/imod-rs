//! `IMOD/Etomo/src/etomo/comscript/SetParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_set_param::{ConstSetParam, DELIMITER};
use super::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::r#type::const_etomo_number::Type;

pub use super::const_set_param::COMMAND_NAME;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `SetParam extends ConstSetParam implements CommandParam`.
#[derive(Clone, Debug)]
pub struct SetParam {
    /// The inherited `ConstSetParam` state.
    pub base: ConstSetParam,
}

/// Java inheritance: every `ConstSetParam` member is reachable on a `SetParam`.
impl std::ops::Deref for SetParam {
    type Target = ConstSetParam;

    fn deref(&self) -> &ConstSetParam {
        &self.base
    }
}

impl std::ops::DerefMut for SetParam {
    fn deref_mut(&mut self) -> &mut ConstSetParam {
        &mut self.base
    }
}

impl SetParam {
    /// Java `SetParam(String, EtomoNumber.Type)`.
    pub fn new(expected_name: &str, r#type: Type) -> SetParam {
        SetParam {
            base: ConstSetParam::new(expected_name, r#type),
        }
    }

    /// Java `setValue`.
    pub fn set_value(&mut self, value: Option<&str>) {
        if self.numeric {
            self.numeric_value.set_string(value);
        } else {
            self.value = value.map(|value| value.to_string());
        }
    }
}

impl CommandParam for SetParam {
    /// Java `parseComScriptCommand`.
    ///
    /// `cmdLineArgs[0]`/`cmdLineArgs[1]` on a short argument list throw
    /// `ArrayIndexOutOfBoundsException`, which `ComScriptUtil.initialize` catches as an
    /// `Exception`; that case is returned as an `InvalidParameterException` carrying the
    /// JVM's message (only the class name in the resulting dialog differs).  A null
    /// element (`name.equals` NullPointerException, never produced by `ComScript`) is
    /// treated as not matching the expected name.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // TODO error checking - throw exceptions for bad syntax
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.reset();

        if cmd_line_args.is_empty() {
            return Err(
                InvalidParameterException::new(Some("Index 0 out of bounds for length 0")).into(),
            );
        }
        self.name = cmd_line_args[0].clone();
        if self.expected_name.is_some() && self.name != self.expected_name {
            self.valid = false;
        }

        if cmd_line_args.len() < 2 {
            return Err(InvalidParameterException::new(Some(&format!(
                "Index 1 out of bounds for length {}",
                cmd_line_args.len()
            )))
            .into());
        }
        if cmd_line_args[1].as_deref() != Some(DELIMITER) {
            return Err(InvalidParameterException::new(Some(&format!(
                "Expecting {}, not {}",
                DELIMITER,
                cmd_line_args[1].as_deref().unwrap_or("null")
            )))
            .into());
        }
        if cmd_line_args.len() > 2 {
            if self.numeric && self.valid {
                self.numeric_value.set_string(cmd_line_args[2].as_deref());
            } else {
                self.value = cmd_line_args[2].clone();
            }
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

        cmd_line_args.push(self.name.clone());
        cmd_line_args.push(Some(DELIMITER.to_string()));
        if self.numeric {
            cmd_line_args.push(Some(self.numeric_value.to_string()));
        } else {
            cmd_line_args.push(self.value.clone());
        }

        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.reset();
    }
}
