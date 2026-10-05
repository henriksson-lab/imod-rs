//! `IMOD/Etomo/src/etomo/type/DuplicateException.java`.

/// Java `public class DuplicateException extends Exception`.
#[derive(Clone, Debug)]
pub struct DuplicateException(pub String);

impl DuplicateException {
    /// Java `DuplicateException(String)`.
    pub fn new(message: &str) -> DuplicateException {
        DuplicateException(message.to_owned())
    }
}

/// Java `Throwable.toString()`.
impl std::fmt::Display for DuplicateException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.type.DuplicateException: {}", self.0)
    }
}
