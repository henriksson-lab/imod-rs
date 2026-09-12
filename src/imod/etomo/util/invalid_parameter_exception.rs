//! `IMOD/Etomo/src/etomo/util/InvalidParameterException.java`.
//!
//! This is deliberately separate from the similarly named com-script
//! exception: the Java source defines two distinct package types.
#![allow(dead_code)]

/// Java `etomo.util.InvalidParameterException`, which extends
/// `java.lang.Exception`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidParameterException {
    /// `java.lang.Throwable`'s detail message, supplied to `super(message)`.
    message: String,
}

impl InvalidParameterException {
    /// Java `InvalidParameterException(String)`.
    pub fn new(message: &str) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Java `java.lang.Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for InvalidParameterException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "etomo.util.InvalidParameterException: {}",
            self.message
        )
    }
}

impl std::error::Error for InvalidParameterException {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_keeps_the_java_exception_message() {
        let exception = InvalidParameterException::new("bad parameter");

        assert_eq!(exception.get_message(), "bad parameter");
        assert_eq!(
            exception.to_string(),
            "etomo.util.InvalidParameterException: bad parameter"
        );
    }
}
