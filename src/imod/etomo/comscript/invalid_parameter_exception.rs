//! `IMOD/Etomo/src/etomo/comscript/InvalidParameterException.java`.
//!
//! Thrown when an invalid set of parameters is detected.  These can be detected either
//! in a com script or due to user input.
//!
//! University of Colorado
#![allow(dead_code)]

/// Java `InvalidParameterException`, which extends `java.lang.Exception`.
#[derive(Clone, Debug)]
pub struct InvalidParameterException {
    /// `java.lang.Throwable`'s detail message, set by `super(message)`.
    message: Option<String>,
    /// Java package-private field `comScript`, initialised to "unknown".
    com_script: Option<String>,
    /// Java package-private field `command`, initialised to "unknown".
    command: Option<String>,
    /// Java package-private field `parameter`, initialised to "unknown".
    parameter: Option<String>,
    /// Java package-private field `lineNumber`, initialised to 0.
    line_number: i32,
}

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

impl InvalidParameterException {
    /// Java `InvalidParameterException(String)`.
    pub fn new(message: Option<&str>) -> InvalidParameterException {
        InvalidParameterException {
            message: message.map(|message| message.to_string()),
            com_script: Some("unknown".to_string()),
            command: Some("unknown".to_string()),
            parameter: Some("unknown".to_string()),
            line_number: 0,
        }
    }

    /// Java `InvalidParameterException(String, String, String, String, int)`, the
    /// package-private constructor.
    pub fn new_with_location(
        message: Option<&str>,
        com_script: Option<&str>,
        command: Option<&str>,
        parameter: Option<&str>,
        line_number: i32,
    ) -> InvalidParameterException {
        let mut exception = InvalidParameterException::new(message);
        exception.com_script = com_script.map(|com_script| com_script.to_string());
        exception.command = command.map(|command| command.to_string());
        exception.parameter = parameter.map(|parameter| parameter.to_string());
        exception.line_number = line_number;
        exception
    }

    /// `java.lang.Throwable.getMessage()`.
    pub fn get_message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Java `getComScript`.
    pub fn get_com_script(&self) -> Option<&str> {
        self.com_script.as_deref()
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<&str> {
        self.command.as_deref()
    }

    /// Java `getParameter`.
    pub fn get_parameter(&self) -> Option<&str> {
        self.parameter.as_deref()
    }

    /// Java `getLineNumber`.
    pub fn get_line_number(&self) -> i32 {
        self.line_number
    }
}

/// `java.lang.Throwable.toString()`: the class name, then `": "` and the detail message
/// when there is one.
impl std::fmt::Display for InvalidParameterException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.message {
            None => f.write_str("etomo.comscript.InvalidParameterException"),
            Some(message) => write!(f, "etomo.comscript.InvalidParameterException: {}", message),
        }
    }
}

impl std::error::Error for InvalidParameterException {}
