//! `IMOD/Etomo/src/etomo/comscript/GotoParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_goto_param::ConstGotoParam;
use super::invalid_parameter_exception::InvalidParameterException;

pub use super::const_goto_param::{COMMAND_NAME, DELIMITER};

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java `GotoParam extends ConstGotoParam implements CommandParam`.
#[derive(Clone, Debug, Default)]
pub struct GotoParam {
    /// The inherited `ConstGotoParam` state.
    pub base: ConstGotoParam,
}

/// Java inheritance: every `ConstGotoParam` member is reachable on a `GotoParam`.
impl std::ops::Deref for GotoParam {
    type Target = ConstGotoParam;

    fn deref(&self) -> &ConstGotoParam {
        &self.base
    }
}

impl std::ops::DerefMut for GotoParam {
    fn deref_mut(&mut self) -> &mut ConstGotoParam {
        &mut self.base
    }
}

impl GotoParam {
    /// Java implicit `GotoParam()`, which runs `ConstGotoParam()`.
    pub fn new() -> GotoParam {
        GotoParam {
            base: ConstGotoParam::new(),
        }
    }

    /// Java `setLabel`.
    pub fn set_label(&mut self, label: Option<&str>) {
        self.label = label.map(|label| label.to_string());
    }
}

impl CommandParam for GotoParam {
    /// Java `parseComScriptCommand`.  An empty argument list throws
    /// `ArrayIndexOutOfBoundsException`, which `ComScriptUtil.initialize` catches as an
    /// `Exception`; it is returned as an `InvalidParameterException` carrying the JVM's
    /// message.
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
        self.label = cmd_line_args[0].clone();
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Create a new command line argument array
        let mut cmd_line_args: Vec<Option<String>> = Vec::with_capacity(20);

        cmd_line_args.push(self.label.clone());

        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.reset();
    }
}
