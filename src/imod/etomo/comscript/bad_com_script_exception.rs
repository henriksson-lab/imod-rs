//! `IMOD/Etomo/src/etomo/comscript/BadComScriptException.java`.

/// Java `BadComScriptException extends Exception`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BadComScriptException {
    message: String,
}

impl BadComScriptException {
    /// Java `BadComScriptException(String)`.
    pub fn new(message: &str) -> BadComScriptException {
        BadComScriptException {
            message: message.to_owned(),
        }
    }

    /// Java `getMessage`.
    pub fn get_message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for BadComScriptException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BadComScriptException {}
