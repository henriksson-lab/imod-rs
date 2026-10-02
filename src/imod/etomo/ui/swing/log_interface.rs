//! `IMOD/Etomo/src/etomo/ui/swing/LogInterface.java`.
//!
//! An interface for anything that can act as a log display.  Used by classes
//! that have messages to log.  Also used by `EtomoLogger`, which is a utility
//! for `LogInterface` classes.
//!
//! Implementers are EDT objects (`Rc`, `&self` methods): `LogWindow` is a Swing
//! frame, and `EtomoLogger` calls back into it from `AppendLater.run()` on the
//! event dispatch thread.
//!
//! The `etomo.storage.FileReader`, `etomo.storage.FileWriter` and
//! `etomo.storage.Loggable` interfaces this one names, and the
//! `javax.swing.text.BadLocationException` it throws, are declared here as
//! the boundaries the rest of the translation already imports.
// TODO(unit): needs etomo/storage/FileReader.java, etomo/storage/FileWriter.java and
// etomo/storage/Loggable.java - their own modules; the traits below stand for them.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `javax.swing.text.BadLocationException` at the text-component boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BadLocationException {
    pub message: String,
}

/// Declared-type boundary for `etomo.storage.FileReader`.
pub trait FileReader {
    /// Java `isReadable()`.
    fn is_readable(&self) -> bool;
    /// Java `readLine()`.
    fn read_line(&mut self) -> Option<String>;
}

/// Declared-type boundary for `etomo.storage.FileWriter`.
pub trait FileWriter {
    /// Java `append(String)`.
    fn append(&mut self, string: &str) -> bool;
    /// Java `flush()`.
    fn flush(&mut self);
    /// Java `getPrevLineEndOffset()`.
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException>;
}

/// Declared-type boundary for `etomo.storage.Loggable`.
pub trait Loggable {
    /// Java `getName()`.
    fn get_name(&self) -> String;
    /// Java `getLogMessage()`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException>;
}

/// The three exceptions `Loggable.getLogMessage()` throws
/// (`LogFileException`, `IOException`, `LockException`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoggableException {
    LogFile(String),
    Io(String),
    Lock(String),
}

/// A Java reference to a `FileReader`.
pub type FileReaderRef = Rc<RefCell<dyn FileReader>>;
/// A Java reference to a `FileWriter`.
pub type FileWriterRef = Rc<RefCell<dyn FileWriter>>;

/// Java `public interface LogInterface`.
pub trait LogInterface {
    /// Java `getManager()`.
    fn get_manager(&self) -> Option<&'static dyn BaseManager>;

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> Option<AxisID>;

    /// Java `logMessage(String, AxisID, String[], String)`.
    fn log_message_string_axis_id_string_array_string(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
        msg_id: Option<&str>,
    ) -> bool;

    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    fn log_message_string_axis_id_array_list(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
    );

    /// Java `logMessage(AxisID, ArrayList<String>)`.
    fn log_message_axis_id_array_list(&self, axis_id: Option<AxisID>, message: Option<&[String]>);

    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable_axis_id(
        &self,
        loggable: Option<&dyn Loggable>,
        axis_id: Option<AxisID>,
    );

    /// Java `logMessage(String, AxisID)`.
    fn log_message_string_axis_id(&self, title: Option<&str>, axis_id: Option<AxisID>);

    /// Java `logMessage(String)`.
    fn log_message_string(&self, message: Option<&str>);

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    fn log_message_string_boolean_boolean_file_writer(
        &self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    );

    /// Java `logMessage(File, FileWriter)`.
    fn log_message_file_file_writer(
        &self,
        file: Option<&Path>,
        secondary_log: Option<FileWriterRef>,
    );

    /// Java `logMessage(File, boolean, FileWriter)`.
    fn log_message_file_boolean_file_writer(
        &self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    );

    /// Java `logMessagePrimaryLog(FileReader)`.
    fn log_message_primary_log(&self, reader: Option<FileReaderRef>);

    /// Java `save()`.
    fn save(&self);

    /// Java `setAllowPrimaryLogging(boolean)`.
    fn set_allow_primary_logging(&self, input: bool);

    /// Java `isAllowPrimaryLogging()`.
    fn is_allow_primary_logging(&self) -> bool;

    // Functions used by EtomoLogger

    /// Java `append(String)`.
    fn append(&self, line: &str);

    /// Java `msgChanged()`.
    fn msg_changed(&self);

    /// Java `getPrevLineEndOffset() throws BadLocationException`.
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException>;
}
