//! `IMOD/Etomo/src/etomo/comscript/EchoParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_echo_param::ConstEchoParam;

pub use super::const_echo_param::COMMAND_NAME;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java package-private `EchoParam extends ConstEchoParam implements CommandParam`.
#[derive(Clone, Debug, Default)]
pub struct EchoParam {
    /// The inherited `ConstEchoParam` state.
    pub base: ConstEchoParam,
}

/// Java inheritance: every `ConstEchoParam` member is reachable on an `EchoParam`.
impl std::ops::Deref for EchoParam {
    type Target = ConstEchoParam;

    fn deref(&self) -> &ConstEchoParam {
        &self.base
    }
}

impl std::ops::DerefMut for EchoParam {
    fn deref_mut(&mut self) -> &mut ConstEchoParam {
        &mut self.base
    }
}

impl EchoParam {
    /// Java implicit `EchoParam()`, which runs `ConstEchoParam()`.
    pub fn new() -> EchoParam {
        EchoParam {
            base: ConstEchoParam::new(),
        }
    }

    /// Java package-private `setString`.
    pub fn set_string(&mut self, string: &str) {
        self.string = string.to_string();
    }
}

impl CommandParam for EchoParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.reset();

        for i in 0..cmd_line_args.len() {
            let arg = cmd_line_args[i].as_deref().unwrap_or("null").to_string() + " ";
            self.string.push_str(&arg);
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.  The list only ever holds the one `StringBuffer`,
    /// so `nArgs` is always 1 and the `toArray` branch (which would throw
    /// `ArrayStoreException` on a `StringBuffer` element) is never taken.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Create a new command line argument array
        let mut cmd_line_args: Vec<String> = Vec::with_capacity(20);

        cmd_line_args.push(self.string.clone());

        let n_args = cmd_line_args.len();
        if n_args == 1 {
            let mut args: Vec<Option<String>> = vec![None; 1];
            args[0] = Some(cmd_line_args[0].to_string());
            script_command.set_command_line_args(&args);
        } else {
            let args: Vec<Option<String>> = cmd_line_args.into_iter().map(Some).collect();
            script_command.set_command_line_args(&args);
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.reset();
    }
}
