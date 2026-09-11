//! `IMOD/Etomo/src/etomo/comscript/FortranInputSyntaxException.java`.
#![allow(dead_code)]

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `FortranInputSyntaxException extends Exception`.
#[derive(Clone, Debug)]
pub struct FortranInputSyntaxException {
    /// Java field `exception`.  The source stores a whole `Exception`; only its message
    /// is ever read (`getMessage`), and `printStackTrace` forwards to it.
    exception: Option<String>,
    /// Java field `newString`.
    new_string: String,
    /// The `Throwable` detail message passed to `super(message)`.
    message: Option<String>,
}

impl FortranInputSyntaxException {
    /// Java `FortranInputSyntaxException(String)`.
    pub fn new(message: &str) -> FortranInputSyntaxException {
        FortranInputSyntaxException {
            exception: None,
            new_string: String::new(),
            message: Some(message.to_string()),
        }
    }

    /// Java `FortranInputSyntaxException(String, String)`.
    pub(crate) fn new_with_new_values(
        message: &str,
        new_values: &str,
    ) -> FortranInputSyntaxException {
        FortranInputSyntaxException {
            exception: None,
            new_string: new_values.to_string(),
            message: Some(message.to_string()),
        }
    }

    /// Java `FortranInputSyntaxException(Exception)`.  `super()` leaves the detail
    /// message null.
    pub fn new_from_exception(exception: &str) -> FortranInputSyntaxException {
        FortranInputSyntaxException {
            exception: Some(exception.to_string()),
            new_string: String::new(),
            message: None,
        }
    }

    /// Java `getNewString`.
    pub fn get_new_string(&self) -> &str {
        &self.new_string
    }

    /// Java `printStackTrace`.  A Java stack trace is a property of the JVM, not of the
    /// program; only the choice of which throwable is printed is reproduced.
    pub fn print_stack_trace(&self) {
        match &self.exception {
            Some(exception) => eprintln!("{}", exception),
            None => eprintln!(
                "etomo.comscript.FortranInputSyntaxException: {}",
                match &self.message {
                    None => "null",
                    Some(message) => message,
                }
            ),
        }
    }

    /// Java `getMessage`.
    pub fn get_message(&self) -> Option<&str> {
        if let Some(exception) = &self.exception {
            return Some(exception);
        }
        self.message.as_deref()
    }
}

impl std::fmt::Display for FortranInputSyntaxException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_message().unwrap_or("null"))
    }
}

impl std::error::Error for FortranInputSyntaxException {}
