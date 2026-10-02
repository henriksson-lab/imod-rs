//! `IMOD/Etomo/src/etomo/comscript/ExitParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_exit_param::ConstExitParam;
use super::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;

pub use super::const_exit_param::COMMAND_NAME;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java package-private `ExitParam extends ConstExitParam implements CommandParam`.
#[derive(Clone, Debug, Default)]
pub struct ExitParam {
    /// The inherited `ConstExitParam` state.
    pub base: ConstExitParam,
}

/// Java inheritance: every `ConstExitParam` member is reachable on an `ExitParam`.
impl std::ops::Deref for ExitParam {
    type Target = ConstExitParam;

    fn deref(&self) -> &ConstExitParam {
        &self.base
    }
}

impl std::ops::DerefMut for ExitParam {
    fn deref_mut(&mut self) -> &mut ConstExitParam {
        &mut self.base
    }
}

impl ExitParam {
    /// Java implicit `ExitParam()`, which runs `ConstExitParam()`.
    pub fn new() -> ExitParam {
        ExitParam {
            base: ConstExitParam::new(),
        }
    }

    /// Java package-private `setResultValue`.
    pub fn set_result_value(&mut self, result_value: i32) {
        self.result_value = result_value;
    }
}

impl CommandParam for ExitParam {
    /// Java `parseComScriptCommand`.  The unchecked `ArrayIndexOutOfBoundsException` (no
    /// argument) and `NumberFormatException` (non-integer argument) are caught by
    /// `ComScriptUtil.initialize` as an `Exception`; both are returned here as an
    /// `InvalidParameterException` carrying the JVM's message.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.reset();

        if cmd_line_args.is_empty() {
            return Err(
                InvalidParameterException::new(Some("Index 0 out of bounds for length 0")).into(),
            );
        }
        self.result_value = match &cmd_line_args[0] {
            None => {
                return Err(
                    InvalidParameterException::new(Some("Cannot parse null string: null")).into(),
                );
            }
            Some(arg) => match java_lang_integer_parse_int(arg) {
                Ok(value) => value,
                Err(message) => {
                    return Err(InvalidParameterException::new(Some(&message)).into());
                }
            },
        };
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Create a new command line argument array
        let mut cmd_line_args: Vec<Option<String>> = Vec::with_capacity(20);

        cmd_line_args.push(Some(self.result_value.to_string()));

        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.reset();
    }
}
