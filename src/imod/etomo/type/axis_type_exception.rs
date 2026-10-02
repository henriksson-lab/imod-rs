//! `IMOD/Etomo/src/etomo/type/AxisTypeException.java`.
//!
//! An exception class to signify incorrect AxisType parameters.

/// Java `AxisTypeException extends Exception`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisTypeException {
    /// `Throwable.detailMessage`.
    message: String,
}

impl AxisTypeException {
    /// Java `AxisTypeException(String)`.
    pub fn new(message: &str) -> AxisTypeException {
        AxisTypeException {
            message: message.to_string(),
        }
    }

    /// `Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.message
    }
}

/// `Throwable.toString()` is the class name and the message; `getMessage()` is what the
/// callers print, so `Display` is the message as the other translated exceptions do.
impl std::fmt::Display for AxisTypeException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AxisTypeException {}
