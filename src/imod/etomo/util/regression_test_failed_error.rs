//! `IMOD/Etomo/src/etomo/util/RegressionTestFailedError.java`.
#![allow(dead_code)]

/// Java `RegressionTestFailedError extends Error`.
#[derive(Clone, Debug)]
pub struct RegressionTestFailedError {
    /// The `Throwable` detail message, the superclass's only state here.
    message: String,
}

impl RegressionTestFailedError {
    /// Java `RegressionTestFailedError(String)`.
    pub fn new(message: &str) -> RegressionTestFailedError {
        RegressionTestFailedError {
            message: message.to_string(),
        }
    }

    /// Java `Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for RegressionTestFailedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RegressionTestFailedError {}
