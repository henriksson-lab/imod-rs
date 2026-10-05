//! `IMOD/Etomo/src/etomo/type/NotLoadedException.java`.

/// Java `public class NotLoadedException extends Exception`.
#[derive(Clone, Debug)]
pub struct NotLoadedException(pub String);

impl NotLoadedException {
    /// Java `NotLoadedException(String)`.
    pub fn new(message: &str) -> NotLoadedException {
        NotLoadedException(message.to_owned())
    }
}

/// Java `Throwable.toString()`.
impl std::fmt::Display for NotLoadedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.type.NotLoadedException: {}", self.0)
    }
}
