//! `IMOD/Etomo/src/etomo/ui/swing/LogInterface.java`.
#![allow(dead_code)]

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

/// Direct declared-type boundary for `etomo.storage.FileReader`.
pub trait FileReader {
    /// Java `isReadable()`.
    fn is_readable(&self) -> bool;
    /// Java `readLine()`.
    fn read_line(&mut self) -> Option<String>;
}

/// Direct declared-type boundary for `etomo.storage.FileWriter`.
pub trait FileWriter {
    /// Java `append(String)`.
    fn append(&mut self, string: &str) -> bool;
    /// Java `flush()`.
    fn flush(&mut self);
    /// Java `getPrevLineEndOffset()`.
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException>;
}

/// Direct declared-type boundary for `etomo.storage.Loggable`.
pub trait Loggable {
    /// Java `getName()`.
    fn get_name(&self) -> String;
    /// Java `getLogMessage()`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException>;
}

/// The three exceptions Java's `EtomoLogger.logMessage(Loggable, AxisID)` catches.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoggableException {
    LogFile(String),
    Io(String),
    Lock(String),
}

/// Java reference to a `FileReader` passed through the Swing event queue.
pub type FileReaderRef = Rc<RefCell<dyn FileReader>>;
/// Java reference to a `FileWriter` passed through the Swing event queue.
pub type FileWriterRef = Rc<RefCell<dyn FileWriter>>;

/// Java `LogInterface`.
pub trait LogInterface {
    /// Java `getManager()`.
    fn get_manager(&self) -> Option<&'static dyn BaseManager>;
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> Option<AxisID>;
    /// Java `logMessage(String, AxisID, String[], String)`.
    fn log_message_array(
        &mut self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[Option<String>]>,
        msg_id: Option<&str>,
    ) -> bool;
    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    fn log_message_list(
        &mut self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[Option<String>]>,
    );
    /// Java `logMessage(AxisID, ArrayList<String>)`.
    fn log_message_axis_list(
        &mut self,
        axis_id: Option<AxisID>,
        message: Option<&[Option<String>]>,
    );
    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable(&mut self, loggable: Option<&dyn Loggable>, axis_id: Option<AxisID>);
    /// Java `logMessage(String, AxisID)`.
    fn log_message_title_axis(&mut self, title: Option<&str>, axis_id: Option<AxisID>);
    /// Java `logMessage(String)`.
    fn log_message(&mut self, message: Option<&str>);
    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    fn log_message_secondary(
        &mut self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    );
    /// Java `logMessage(File, FileWriter)`.
    fn log_message_file(&mut self, file: Option<&Path>, secondary_log: Option<FileWriterRef>);
    /// Java `logMessage(File, boolean, FileWriter)`.
    fn log_message_file_newline(
        &mut self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    );
    /// Java `logMessagePrimaryLog(FileReader)`.
    fn log_message_primary_log(&mut self, reader: Option<FileReaderRef>);
    /// Java `save()`.
    fn save(&mut self);
    /// Java `setAllowPrimaryLogging(boolean)`.
    fn set_allow_primary_logging(&mut self, input: bool);
    /// Java `isAllowPrimaryLogging()`.
    fn is_allow_primary_logging(&self) -> bool;
    /// Java `append(String)`.
    fn append(&mut self, line: &str);
    /// Java `msgChanged()`.
    fn msg_changed(&mut self);
    /// Java `getPrevLineEndOffset()`.
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException>;
}
