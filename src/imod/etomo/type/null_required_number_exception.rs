//! `IMOD/Etomo/src/etomo/type/NullRequiredNumberException.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class NullRequiredNumberException extends Exception`.  The value is
/// the exception's message (`getMessage()`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NullRequiredNumberException(pub String);

impl NullRequiredNumberException {
    /// Java `NullRequiredNumberException(String)`.
    pub fn new(message: &str) -> NullRequiredNumberException {
        NullRequiredNumberException(message.to_string())
    }

    /// Java `getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.0
    }
}

/// Java `Throwable.toString()`: the class name and the message.
impl std::fmt::Display for NullRequiredNumberException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.type.NullRequiredNumberException: {}", self.0)
    }
}
