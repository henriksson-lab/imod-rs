//! `IMOD/Etomo/src/etomo/ui/swing/EtomoLogger.java`.
//!
//! Uses `SwingUtilities.invokeLater` to add timestamps and lines to a
//! `LogInterface`.
//!
//! The logger is an EDT object owned by its `LogInterface` (`LogWindow`), so
//! it holds the primary log as a `Weak` (Java's back reference) and posts each
//! `AppendLater` through `event_queue::invoke_later` wrapped in an `EdtRef`,
//! the way the rest of the UI translation posts EDT work.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::time::Duration;

use super::log_interface::LogInterface;
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::storage::file_reader::FileReaderRef;
use crate::imod::etomo::storage::file_writer::FileWriterRef;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java `SwingUtilities.invokeLater(new AppendLater(...))`.
fn invoke_later(runnable: AppendLater) {
    let runnable = EdtRef::new(Rc::new(runnable));
    event_queue::invoke_later(move || runnable.get().run());
}

/// Java final `EtomoLogger`.
pub struct EtomoLogger {
    /// Java `private final LogInterface primaryLog`.
    primary_log: Weak<dyn LogInterface>,
    /// Java `private boolean allowPrimaryLogging = true`.
    allow_primary_logging: Cell<bool>,
}

impl EtomoLogger {
    /// Java `EtomoLogger(LogInterface)`.
    pub fn new(primary_log: Weak<dyn LogInterface>) -> EtomoLogger {
        // try { Thread.sleep(1); } catch (InterruptedException e) {}
        std::thread::sleep(Duration::from_millis(1));
        EtomoLogger {
            primary_log,
            allow_primary_logging: Cell::new(true),
        }
    }

    /// Java synchronized `loadInitMessages(ArrayList<String>) throws IOException`.
    /// Nothing in the body throws, so there is no error channel.
    pub fn load_init_messages(&self, line_list: Option<Vec<Option<String>>>) {
        invoke_later(AppendLater::new_log_interface_boolean_array_list_boolean(
            self.primary_log.clone(),
            self.allow_primary_logging.get(),
            line_list,
            true,
        ));
    }

    /// Java `logMessage(String, String)`.
    pub fn log_message_string_string(&self, line1: Option<&str>, line2: Option<&str>) {
        invoke_later(AppendLater::new_log_interface_boolean_string_string_string(
            self.primary_log.clone(),
            self.allow_primary_logging.get(),
            Some(utilities::get_date_time_stamp()),
            line1.map(str::to_owned),
            line2.map(str::to_owned),
        ));
    }

    /// Java `logMessage(Loggable, AxisID)`.
    pub fn log_message_loggable_axis_id(
        &self,
        loggable: Option<&dyn Loggable>,
        axis_id: Option<AxisID>,
    ) {
        let Some(loggable) = loggable else {
            return;
        };
        // try {
        let name = loggable.get_name();
        match loggable.get_log_message() {
            Ok(message) => invoke_later(
                AppendLater::new_log_interface_boolean_string_string_array_list_axis_id(
                    self.primary_log.clone(),
                    self.allow_primary_logging.get(),
                    Some(utilities::get_date_time_stamp()),
                    Some(name),
                    Some(message),
                    axis_id,
                ),
            ),
            // catch (final LogFileException | IOException e)
            Err(LoggableException::LogFile(message) | LoggableException::Io(message)) => {
                // e.printStackTrace();
                eprintln!("{message}");
                invoke_later(AppendLater::new_log_interface_boolean_string_string(
                    self.primary_log.clone(),
                    self.allow_primary_logging.get(),
                    Some("Unable to log message:".to_owned()),
                    Some(message),
                ));
            }
            // catch (final LockException e)
            Err(LoggableException::Lock(message)) => {
                invoke_later(AppendLater::new_log_interface_boolean_string_string(
                    self.primary_log.clone(),
                    self.allow_primary_logging.get(),
                    Some("Unable to log message:".to_owned()),
                    Some(message),
                ));
            }
        }
    }

    /// Java `isAllowPrimaryLogging()`.
    pub fn is_allow_primary_logging(&self) -> bool {
        self.allow_primary_logging.get()
    }

    /// Java `setAllowPrimaryLogging(boolean)`.
    pub fn set_allow_primary_logging(&self, input: bool) {
        self.allow_primary_logging.set(input);
    }

    /// Java `logMessage(String, AxisID, String[], String)`.
    pub fn log_message_string_axis_id_string_array_string(
        &self,
        title: Option<&str>,
        _axis_id: Option<AxisID>,
        message: Option<&[String]>,
        msg_id: Option<&str>,
    ) -> bool {
        invoke_later(
            AppendLater::new_log_interface_boolean_string_string_string_array(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                Some(utilities::get_date_time_stamp()),
                title.map(str::to_owned),
                message.map(|message| message.iter().cloned().map(Some).collect()),
            ),
        );
        if let (Some(msg_id), Some(message)) = (msg_id, message) {
            for line in message {
                if line.contains(msg_id) {
                    return true;
                }
            }
        }
        false
    }

    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    pub fn log_message_string_axis_id_array_list(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
    ) {
        invoke_later(
            AppendLater::new_log_interface_boolean_string_string_array_list_axis_id(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                Some(utilities::get_date_time_stamp()),
                title.map(str::to_owned),
                message.map(|message| message.iter().cloned().map(Some).collect()),
                axis_id,
            ),
        );
    }

    /// Java `logMessage(AxisID, ArrayList<String>)`.
    pub fn log_message_axis_id_array_list(
        &self,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
    ) {
        invoke_later(
            AppendLater::new_log_interface_boolean_string_array_list_axis_id(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                Some(utilities::get_date_time_stamp()),
                message.map(|message| message.iter().cloned().map(Some).collect()),
                axis_id,
            ),
        );
    }

    /// Java `logMessage(String, AxisID)`.
    pub fn log_message_string_axis_id(&self, title: Option<&str>, axis_id: Option<AxisID>) {
        invoke_later(
            AppendLater::new_log_interface_boolean_string_string_axis_id(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                Some(utilities::get_date_time_stamp()),
                title.map(str::to_owned),
                axis_id,
            ),
        );
    }

    /// Java `logMessage(String)`.
    pub fn log_message_string(&self, message: Option<&str>) {
        invoke_later(AppendLater::new_log_interface_boolean_string_string(
            self.primary_log.clone(),
            self.allow_primary_logging.get(),
            Some(utilities::get_date_time_stamp()),
            message.map(str::to_owned),
        ));
    }

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    pub fn log_message_string_boolean_boolean_file_writer(
        &self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        invoke_later(
            AppendLater::new_log_interface_boolean_file_writer_string_string_boolean(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                secondary_log,
                if timestamp {
                    Some(utilities::get_date_time_stamp())
                } else {
                    None
                },
                message.map(str::to_owned),
                newline,
            ),
        );
    }

    /// Java `logMessage(File, FileWriter)`.
    pub fn log_message_file_file_writer(
        &self,
        file: Option<&Path>,
        secondary_log: Option<FileWriterRef>,
    ) {
        invoke_later(AppendLater::new_log_interface_boolean_file_writer_file(
            self.primary_log.clone(),
            self.allow_primary_logging.get(),
            secondary_log,
            file.map(Path::to_path_buf),
        ));
    }

    /// Java `logMessage(File, boolean, FileWriter)`.
    pub fn log_message_file_boolean_file_writer(
        &self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        invoke_later(
            AppendLater::new_log_interface_boolean_file_writer_file_boolean(
                self.primary_log.clone(),
                self.allow_primary_logging.get(),
                secondary_log,
                file.map(Path::to_path_buf),
                newline,
            ),
        );
    }

    /// Java `logMessagePrimaryLog(FileReader)`.
    pub fn log_message_primary_log(&self, reader: Option<FileReaderRef>) {
        invoke_later(AppendLater::new_log_interface_boolean_file_reader(
            self.primary_log.clone(),
            true,
            reader,
        ));
    }

    /// Java private `getEmergencyMonitor()`.  `AppendLater` is a Java inner
    /// class calling this on its enclosing `EtomoLogger`, whose only state read
    /// here is `primaryLog`; that reference is passed in, since the inner
    /// object does not hold its logger.
    fn get_emergency_monitor(
        primary_log: &Weak<dyn LogInterface>,
    ) -> Option<Arc<EmergencyMonitor>> {
        let Some(primary_log) = primary_log.upgrade() else {
            return None;
        };
        let Some(manager) = primary_log.get_manager() else {
            return None;
        };
        Some(manager.get_emergency_monitor(primary_log.get_axis_id()))
    }
}

/// Java private final inner `AppendLater implements Runnable`.
pub struct AppendLater {
    primary_log: Weak<dyn LogInterface>,
    allow_primary_logging: bool,
    secondary_log: Option<FileWriterRef>,

    timestamp: Option<String>,
    line1: Option<String>,
    line2: Option<String>,
    string_array: Option<Vec<Option<String>>>,
    line_list: Option<Vec<Option<String>>>,
    file: Option<PathBuf>,
    reader: Option<FileReaderRef>,
    axis_id: Option<AxisID>,
    /// Prevents excessive empty lines.
    newline: bool,
    init: bool,
    /// The enclosing `EtomoLogger.primaryLog` (Java inner-class `this$0`),
    /// read by `getEmergencyMonitor`.
    outer_primary_log: Weak<dyn LogInterface>,
}

impl AppendLater {
    /// The Java field initialisers, shared by every constructor.
    fn with_fields(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
    ) -> AppendLater {
        AppendLater {
            outer_primary_log: primary_log.clone(),
            primary_log,
            allow_primary_logging,
            secondary_log,
            timestamp: None,
            line1: None,
            line2: None,
            string_array: None,
            line_list: None,
            file: None,
            reader: None,
            axis_id: None,
            newline: true,
            init: false,
        }
    }

    /// Java `AppendLater(LogInterface, boolean, FileReader)`.
    fn new_log_interface_boolean_file_reader(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        reader: Option<FileReaderRef>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.reader = reader;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, FileWriter, String, String, boolean)`.
    fn new_log_interface_boolean_file_writer_string_string_boolean(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        timestamp: Option<String>,
        line1: Option<String>,
        newline: bool,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, secondary_log);
        this.timestamp = timestamp;
        this.line1 = line1;
        this.newline = newline;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, String)`.
    fn new_log_interface_boolean_string_string(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line1 = line1;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, String, AxisID)`.
    fn new_log_interface_boolean_string_string_axis_id(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        axis_id: Option<AxisID>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line1 = line1;
        this.axis_id = axis_id;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, String, String)`.
    fn new_log_interface_boolean_string_string_string(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        line2: Option<String>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line1 = line1;
        this.line2 = line2;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, String, String[])`.
    fn new_log_interface_boolean_string_string_string_array(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        string_array: Option<Vec<Option<String>>>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line1 = line1;
        this.string_array = string_array;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, ArrayList<String>, AxisID)`.
    fn new_log_interface_boolean_string_array_list_axis_id(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line_list: Option<Vec<Option<String>>>,
        axis_id: Option<AxisID>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line_list = line_list;
        this.axis_id = axis_id;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, String, String, ArrayList<String>,
    /// AxisID)`.
    fn new_log_interface_boolean_string_string_array_list_axis_id(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        line_list: Option<Vec<Option<String>>>,
        axis_id: Option<AxisID>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.timestamp = timestamp;
        this.line1 = line1;
        this.line_list = line_list;
        this.axis_id = axis_id;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, ArrayList<String>, boolean)`.
    fn new_log_interface_boolean_array_list_boolean(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        line_list: Option<Vec<Option<String>>>,
        init: bool,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, None);
        this.line_list = line_list;
        this.init = init;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, FileWriter, File)`.
    fn new_log_interface_boolean_file_writer_file(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        file: Option<PathBuf>,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, secondary_log);
        this.file = file;
        this
    }

    /// Java `AppendLater(LogInterface, boolean, FileWriter, File, boolean)`.
    fn new_log_interface_boolean_file_writer_file_boolean(
        primary_log: Weak<dyn LogInterface>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        file: Option<PathBuf>,
        newline: bool,
    ) -> AppendLater {
        let mut this = AppendLater::with_fields(primary_log, allow_primary_logging, secondary_log);
        this.file = file;
        this.newline = newline;
        this
    }

    /// Java private `append(String)`.
    fn append(&self, string: &str) {
        if self.allow_primary_logging {
            // Upstream NPE fixed in translation (EtomoLogger.java:271): Java
            // dereferences primaryLog unconditionally; a log window that is
            // gone (or null) receives nothing.
            if let Some(primary_log) = self.primary_log.upgrade() {
                primary_log.append(string);
            }
        }
        let mut secondary_append_success = false;
        if let Some(secondary_log) = &self.secondary_log {
            secondary_append_success = secondary_log.lock().unwrap().append(string);
        }
        // Make sure that string is logged somewhere.
        if !self.allow_primary_logging && !secondary_append_success {
            eprintln!("{string}");
        }
    }

    /// Java `run()`.  Append lines and lineList to textArea.
    pub fn run(&self) {
        if self.newline {
            self.new_line(None);
        }
        if let Some(line1) = &self.line1 {
            self.append(
                &(line1.clone()
                    + &match &self.timestamp {
                        Some(timestamp) => " - ".to_owned() + timestamp,
                        None => String::new(),
                    }),
            );
            self.new_line(Some(line1));
        } else if let Some(timestamp) = &self.timestamp {
            self.append(timestamp);
            self.new_line(Some(timestamp));
        }
        if let Some(line2) = &self.line2 {
            self.append(line2);
            self.new_line(Some(line2));
        }
        if let Some(string_array) = &self.string_array {
            for i in 0..string_array.len() {
                if let Some(string) = &string_array[i] {
                    self.append(string);
                    self.new_line(Some(string));
                }
            }
        }
        if let Some(line_list) = &self.line_list {
            let len = line_list.len();
            for i in 0..len {
                if let Some(line) = &line_list[i] {
                    self.append(line);
                    self.new_line(Some(line));
                }
            }
        }
        if let Some(file) = &self.file
            && file.exists()
            && file.is_file()
            && std::fs::File::open(file).is_ok()
        {
            self.append(&format!(
                "Logging from file: {}",
                utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
            ));
            self.new_line(None);
            // try {
            let result: Result<(), LogFileError> = (|| {
                let log_file = LogFile::get_instance_file(
                    Some(file),
                    EtomoLogger::get_emergency_monitor(&self.outer_primary_log),
                )?;
                let id = log_file.open_reader()?;
                // Upstream NPE fixed in translation (EtomoLogger.java:326-328):
                // openReader can return null, which readLine then dereferences;
                // a null reader logs nothing.
                let Some(id) = id else {
                    return Ok(());
                };
                while let Some(line) = log_file.read_line(&id)? {
                    self.append(&line);
                    self.new_line(Some(&line));
                }
                Ok(())
            })();
            match result {
                Ok(()) => {}
                // catch (final LockException e) {}
                Err(LogFileError::Lock(_)) => {}
                // catch (final LogFileException | IOException e)
                Err(e) => {
                    // e.printStackTrace();
                    eprintln!("{e}");
                    eprintln!("Unable to log from file.  {e}");
                }
            }
        }
        if let Some(reader) = &self.reader
            && reader.lock().unwrap().is_readable()
        {
            loop {
                let line = reader.lock().unwrap().read_line();
                let Some(line) = line else {
                    break;
                };
                self.append(&line);
                self.new_line(Some(&line));
            }
        }
        if !self.init {
            if self.allow_primary_logging {
                if let Some(primary_log) = self.primary_log.upgrade() {
                    primary_log.msg_changed();
                }
            }
            if let Some(secondary_log) = &self.secondary_log {
                secondary_log.lock().unwrap().flush();
            }
        }
        if let Some(axis_id) = self.axis_id
            && (axis_id == AxisID::First || axis_id == AxisID::Second)
        {
            self.append(&(axis_id.to_string() + " axis"));
            self.new_line(None);
        }
    }

    /// Java private `newLine(String)`.  Appends a newline character if the
    /// last line in the text area is not empty.  Put a null in previousLine to
    /// force a new line.
    fn new_line(&self, previous_line: Option<&str>) {
        // try {
        // messages should be alone on a line
        let mut prev_line_end_offset = Ok(0);
        if self.allow_primary_logging {
            if let Some(primary_log) = self.primary_log.upgrade() {
                prev_line_end_offset = primary_log.get_prev_line_end_offset();
            }
        } else if let Some(secondary_log) = &self.secondary_log {
            prev_line_end_offset =
                Ok(secondary_log.lock().unwrap().get_prev_line_end_offset() as usize);
        }
        match prev_line_end_offset {
            Ok(prev_line_end_offset) => {
                if prev_line_end_offset != 0
                    && (previous_line.is_none() || !previous_line.unwrap().ends_with('\n'))
                {
                    self.append("\n");
                }
            }
            // catch (BadLocationException e) { e.printStackTrace(); }
            Err(e) => eprintln!("{}", e.message),
        }
    }
}
