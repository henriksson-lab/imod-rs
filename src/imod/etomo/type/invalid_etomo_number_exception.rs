//! `IMOD/Etomo/src/etomo/type/InvalidEtomoNumberException.java`.

#![allow(dead_code)]

/// Java `InvalidEtomoNumberException extends Exception`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidEtomoNumberException {
    message: String,
}

impl InvalidEtomoNumberException {
    /// Java `InvalidEtomoNumberException(String)`.
    pub fn new(message: &str) -> InvalidEtomoNumberException {
        InvalidEtomoNumberException {
            message: message.into(),
        }
    }

    /// The Java exception message supplied to `super(message)`.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for InvalidEtomoNumberException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for InvalidEtomoNumberException {}

#[cfg(test)]
mod tests {
    use super::InvalidEtomoNumberException;

    #[test]
    fn constructor_preserves_the_java_exception_message() {
        let exception = InvalidEtomoNumberException::new("not a number");
        assert_eq!(exception.message(), "not a number");
        assert_eq!(exception.to_string(), "not a number");
    }
}
