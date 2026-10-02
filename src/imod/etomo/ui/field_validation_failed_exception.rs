//! `IMOD/Etomo/src/etomo/ui/FieldValidationFailedException.java`.

/// Java `FieldValidationFailedException.rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `FieldValidationFailedException extends Exception`: carries only the
/// exception message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldValidationFailedException {
    /// Java `Throwable.detailMessage`.
    message: Option<String>,
}

impl FieldValidationFailedException {
    /// Java `FieldValidationFailedException(String)`.
    pub fn new(message: Option<&str>) -> FieldValidationFailedException {
        FieldValidationFailedException {
            message: message.map(|message| message.to_string()),
        }
    }

    /// Java `Throwable.getMessage()`.
    pub fn get_message(&self) -> Option<&str> {
        self.message.as_deref()
    }
}

/// Java `Throwable.toString()`: the class name, then `": "` and the message when
/// there is one.
impl std::fmt::Display for FieldValidationFailedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.message {
            None => f.write_str("etomo.ui.FieldValidationFailedException"),
            Some(message) => write!(f, "etomo.ui.FieldValidationFailedException: {}", message),
        }
    }
}

impl std::error::Error for FieldValidationFailedException {}
