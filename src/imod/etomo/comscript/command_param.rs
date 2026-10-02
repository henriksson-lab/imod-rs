//! `IMOD/Etomo/src/etomo/comscript/CommandParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;

/// The three checked exceptions `parseComScriptCommand` declares.
#[derive(Debug)]
pub enum ParseComScriptError {
    BadComScript(BadComScriptException),
    FortranInputSyntax(FortranInputSyntaxException),
    InvalidParameter(InvalidParameterException),
    /// A `NumberFormatException` (or other runtime exception) out of the
    /// parse, which the Java callers catch as `Exception`.
    NumberFormat(String),
}

impl std::fmt::Display for ParseComScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseComScriptError::BadComScript(e) => write!(f, "{e}"),
            ParseComScriptError::FortranInputSyntax(e) => write!(f, "{e}"),
            ParseComScriptError::InvalidParameter(e) => write!(f, "{e}"),
            ParseComScriptError::NumberFormat(message) => f.write_str(message),
        }
    }
}

impl From<BadComScriptException> for ParseComScriptError {
    fn from(e: BadComScriptException) -> ParseComScriptError {
        ParseComScriptError::BadComScript(e)
    }
}

impl From<FortranInputSyntaxException> for ParseComScriptError {
    fn from(e: FortranInputSyntaxException) -> ParseComScriptError {
        ParseComScriptError::FortranInputSyntax(e)
    }
}

impl From<InvalidParameterException> for ParseComScriptError {
    fn from(e: InvalidParameterException) -> ParseComScriptError {
        ParseComScriptError::InvalidParameter(e)
    }
}

/// Java `CommandParam`.
pub trait CommandParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError>;

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException>;

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self);
}
