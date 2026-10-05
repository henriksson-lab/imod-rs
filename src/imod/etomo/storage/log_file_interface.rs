//! `IMOD/Etomo/src/etomo/storage/LogFileInterface.java`.
//!
//! The package-private interface `LogFile.Handle` implements, so `FileReader` and
//! `SimpleDefocusFile` can be given any log file.  A `LogFile.Handle` is held as an
//! `Arc<Handle>` here, so the interface is implemented for that.

use std::sync::Arc;

use super::log_file::{Handle, Id, LogFileError, ReaderId, WriterId};

/// Java `interface LogFileInterface`.  `LogFileException`, `IOException`,
/// `LockException` and `UnlockedException` are the variants of `LogFileError`.
pub(crate) trait LogFileInterface: Send + Sync {
    /// Java `exists()`.
    fn exists(&self) -> bool;

    /// Java `openReader()`.
    fn open_reader(&self) -> Result<Option<ReaderId>, LogFileError>;

    /// Java `openReader(boolean)`.
    fn open_reader_required(&self, required: bool) -> Result<Option<ReaderId>, LogFileError>;

    /// Java `readLine(ReaderId)`.
    fn read_line(&self, read_id: &ReaderId) -> Result<Option<String>, LogFileError>;

    /// Java `closeId(Id)`.
    fn close_id(&self, id: Option<&Arc<Id>>);

    /// Java `backup()`.
    fn backup(&self) -> Result<bool, LogFileError>;

    /// Java `openWriter()`.
    fn open_writer(&self) -> Result<WriterId, LogFileError>;

    /// Java `write(String, WriterId)`.
    fn write(&self, string: Option<&str>, writer_id: &WriterId) -> Result<(), LogFileError>;

    /// Java `getAbsolutePath()`.
    fn get_absolute_path(&self) -> String;
}

/// Java `LogFile.Handle implements LogFileInterface`.
impl LogFileInterface for Arc<Handle> {
    fn exists(&self) -> bool {
        Handle::exists(self)
    }

    fn open_reader(&self) -> Result<Option<ReaderId>, LogFileError> {
        Handle::open_reader(self)
    }

    fn open_reader_required(&self, required: bool) -> Result<Option<ReaderId>, LogFileError> {
        Handle::open_reader_required(self, required)
    }

    fn read_line(&self, read_id: &ReaderId) -> Result<Option<String>, LogFileError> {
        Handle::read_line(self, read_id)
    }

    fn close_id(&self, id: Option<&Arc<Id>>) {
        Handle::close_id(self, id)
    }

    fn backup(&self) -> Result<bool, LogFileError> {
        Handle::backup(self)
    }

    fn open_writer(&self) -> Result<WriterId, LogFileError> {
        Handle::open_writer(self)
    }

    fn write(&self, string: Option<&str>, writer_id: &WriterId) -> Result<(), LogFileError> {
        Handle::write(self, string, writer_id)
    }

    fn get_absolute_path(&self) -> String {
        Handle::get_absolute_path(self)
    }
}
