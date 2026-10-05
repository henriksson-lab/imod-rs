//! `IMOD/Etomo/src/etomo/storage/Loggable.java`.
//!
//! Something whose current state can be written to the project log.

/// Java `public interface Loggable`.
pub trait Loggable {
    /// Java `getLogMessage() throws LogFileException, IOException, LockException`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException>;

    /// Java `getName()`.
    fn get_name(&self) -> String;
}

/// The three exceptions `Loggable.getLogMessage()` throws
/// (`LogFileException`, `IOException`, `LockException`), each with its message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoggableException {
    LogFile(String),
    Io(String),
    Lock(String),
}
