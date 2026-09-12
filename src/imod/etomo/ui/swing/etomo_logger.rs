//! `IMOD/Etomo/src/etomo/ui/swing/EtomoLogger.java`.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use super::log_interface::{
    FileReaderRef, FileWriterRef, LogInterface, Loggable, LoggableException,
};
use crate::imod::etomo::storage::log_file::LogFile;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;

/// Rust boundary for Java `SwingUtilities.invokeLater(Runnable)`.
pub trait SwingUtilities {
    /// Java `invokeLater`.
    fn invoke_later(&mut self, runnable: AppendLater);
}

/// Java final `EtomoLogger`.
pub struct EtomoLogger {
    primary_log: Rc<RefCell<dyn LogInterface>>,
    allow_primary_logging: bool,
    swing_utilities: Rc<RefCell<dyn SwingUtilities>>,
}

impl EtomoLogger {
    /// Java `EtomoLogger(LogInterface)` plus its `SwingUtilities` boundary.
    pub fn new(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        swing_utilities: Rc<RefCell<dyn SwingUtilities>>,
    ) -> Self {
        std::thread::sleep(Duration::from_millis(1));
        Self {
            primary_log,
            allow_primary_logging: true,
            swing_utilities,
        }
    }

    /// Java synchronized `loadInitMessages(ArrayList<String>)`.
    pub fn load_init_messages(&mut self, line_list: Option<Vec<Option<String>>>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_init(
                self.primary_log.clone(),
                self.allow_primary_logging,
                line_list,
                true,
            ));
    }

    /// Java `logMessage(String, String)`.
    pub fn log_message_lines(&mut self, line1: Option<String>, line2: Option<String>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_lines(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                line1,
                line2,
            ));
    }

    /// Java `logMessage(Loggable, AxisID)`.
    pub fn log_message_loggable(
        &mut self,
        loggable: Option<&dyn Loggable>,
        axis_id: Option<AxisID>,
    ) {
        let Some(loggable) = loggable else { return };
        match loggable.get_log_message() {
            Ok(message) => {
                self.swing_utilities
                    .borrow_mut()
                    .invoke_later(AppendLater::new_title_list(
                        self.primary_log.clone(),
                        self.allow_primary_logging,
                        Some(utilities::get_date_time_stamp()),
                        Some(loggable.get_name()),
                        Some(message),
                        axis_id,
                    ))
            }
            Err(LoggableException::LogFile(message) | LoggableException::Io(message)) => {
                eprintln!("{message}");
                self.swing_utilities
                    .borrow_mut()
                    .invoke_later(AppendLater::new_lines(
                        self.primary_log.clone(),
                        self.allow_primary_logging,
                        None,
                        Some("Unable to log message:".to_owned()),
                        Some(message),
                    ));
            }
            Err(LoggableException::Lock(message)) => self
                .swing_utilities
                .borrow_mut()
                .invoke_later(AppendLater::new_lines(
                    self.primary_log.clone(),
                    self.allow_primary_logging,
                    None,
                    Some("Unable to log message:".to_owned()),
                    Some(message),
                )),
        }
    }

    /// Java `isAllowPrimaryLogging()`.
    pub fn is_allow_primary_logging(&self) -> bool {
        self.allow_primary_logging
    }

    /// Java `setAllowPrimaryLogging(boolean)`.
    pub fn set_allow_primary_logging(&mut self, input: bool) {
        self.allow_primary_logging = input;
    }

    /// Java `logMessage(String, AxisID, String[], String)`.
    pub fn log_message_array(
        &mut self,
        title: Option<String>,
        axis_id: Option<AxisID>,
        message: Option<Vec<Option<String>>>,
        msg_id: Option<&str>,
    ) -> bool {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_array(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                title,
                message.clone(),
            ));
        message.as_ref().is_some_and(|message| {
            msg_id.is_some_and(|id| message.iter().flatten().any(|line| line.contains(id)))
        })
    }

    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    pub fn log_message_list(
        &mut self,
        title: Option<String>,
        axis_id: Option<AxisID>,
        message: Option<Vec<Option<String>>>,
    ) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_title_list(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                title,
                message,
                axis_id,
            ));
    }

    /// Java `logMessage(AxisID, ArrayList<String>)`.
    pub fn log_message_axis_list(
        &mut self,
        axis_id: Option<AxisID>,
        message: Option<Vec<Option<String>>>,
    ) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_axis_list(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                message,
                axis_id,
            ));
    }

    /// Java `logMessage(String, AxisID)`.
    pub fn log_message_title_axis(&mut self, title: Option<String>, axis_id: Option<AxisID>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_line_axis(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                title,
                axis_id,
            ));
    }

    /// Java `logMessage(String)`.
    pub fn log_message(&mut self, message: Option<String>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_timestamp_line(
                self.primary_log.clone(),
                self.allow_primary_logging,
                Some(utilities::get_date_time_stamp()),
                message,
            ));
    }

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    pub fn log_message_secondary(
        &mut self,
        message: Option<String>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_secondary_line(
                self.primary_log.clone(),
                self.allow_primary_logging,
                secondary_log,
                timestamp.then(utilities::get_date_time_stamp),
                message,
                newline,
            ));
    }

    /// Java `logMessage(File, FileWriter)`.
    pub fn log_message_file(&mut self, file: Option<&Path>, secondary_log: Option<FileWriterRef>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_file(
                self.primary_log.clone(),
                self.allow_primary_logging,
                secondary_log,
                file.map(Path::to_path_buf),
            ));
    }

    /// Java `logMessage(File, boolean, FileWriter)`.
    pub fn log_message_file_newline(
        &mut self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_file_newline(
                self.primary_log.clone(),
                self.allow_primary_logging,
                secondary_log,
                file.map(Path::to_path_buf),
                newline,
            ));
    }

    /// Java `logMessagePrimaryLog(FileReader)`.
    pub fn log_message_primary_log(&mut self, reader: Option<FileReaderRef>) {
        self.swing_utilities
            .borrow_mut()
            .invoke_later(AppendLater::new_reader(
                self.primary_log.clone(),
                true,
                reader,
            ));
    }
}

/// Java private final inner `AppendLater implements Runnable`.
pub struct AppendLater {
    primary_log: Rc<RefCell<dyn LogInterface>>,
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
    newline: bool,
    init: bool,
}

impl AppendLater {
    /// Java `AppendLater(LogInterface, boolean, FileReader)`.
    pub fn new_reader(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        reader: Option<FileReaderRef>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp: None,
            line1: None,
            line2: None,
            string_array: None,
            line_list: None,
            file: None,
            reader,
            axis_id: None,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, FileWriter, String, String, boolean)`.
    pub fn new_secondary_line(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        timestamp: Option<String>,
        line1: Option<String>,
        newline: bool,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log,
            timestamp,
            line1,
            line2: None,
            string_array: None,
            line_list: None,
            file: None,
            reader: None,
            axis_id: None,
            newline,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, String, String)`.
    pub fn new_timestamp_line(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1,
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
    /// Java `AppendLater(LogInterface, boolean, String, String, AxisID)`.
    pub fn new_line_axis(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1,
            line2: None,
            string_array: None,
            line_list: None,
            file: None,
            reader: None,
            axis_id,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, String, String, String)`.
    pub fn new_lines(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        line2: Option<String>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1,
            line2,
            string_array: None,
            line_list: None,
            file: None,
            reader: None,
            axis_id: None,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, String, String, String[])`.
    pub fn new_array(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        string_array: Option<Vec<Option<String>>>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1,
            line2: None,
            string_array,
            line_list: None,
            file: None,
            reader: None,
            axis_id: None,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, String, ArrayList<String>, AxisID)`.
    pub fn new_axis_list(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line_list: Option<Vec<Option<String>>>,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1: None,
            line2: None,
            string_array: None,
            line_list,
            file: None,
            reader: None,
            axis_id,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, String, String, ArrayList<String>, AxisID)`.
    pub fn new_title_list(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        timestamp: Option<String>,
        line1: Option<String>,
        line_list: Option<Vec<Option<String>>>,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp,
            line1,
            line2: None,
            string_array: None,
            line_list,
            file: None,
            reader: None,
            axis_id,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, ArrayList<String>, boolean)`.
    pub fn new_init(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        line_list: Option<Vec<Option<String>>>,
        init: bool,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log: None,
            timestamp: None,
            line1: None,
            line2: None,
            string_array: None,
            line_list,
            file: None,
            reader: None,
            axis_id: None,
            newline: true,
            init,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, FileWriter, File)`.
    pub fn new_file(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        file: Option<PathBuf>,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log,
            timestamp: None,
            line1: None,
            line2: None,
            string_array: None,
            line_list: None,
            file,
            reader: None,
            axis_id: None,
            newline: true,
            init: false,
        }
    }
    /// Java `AppendLater(LogInterface, boolean, FileWriter, File, boolean)`.
    pub fn new_file_newline(
        primary_log: Rc<RefCell<dyn LogInterface>>,
        allow_primary_logging: bool,
        secondary_log: Option<FileWriterRef>,
        file: Option<PathBuf>,
        newline: bool,
    ) -> Self {
        Self {
            primary_log,
            allow_primary_logging,
            secondary_log,
            timestamp: None,
            line1: None,
            line2: None,
            string_array: None,
            line_list: None,
            file,
            reader: None,
            axis_id: None,
            newline,
            init: false,
        }
    }

    /// Java private `append(String)`.
    fn append(&mut self, string: &str) {
        if self.allow_primary_logging {
            self.primary_log.borrow_mut().append(string);
        }
        let secondary_append_success = self
            .secondary_log
            .as_ref()
            .is_some_and(|secondary_log| secondary_log.borrow_mut().append(string));
        if !self.allow_primary_logging && !secondary_append_success {
            eprintln!("{string}");
        }
    }

    /// Java `run()`.
    pub fn run(&mut self) {
        if self.newline {
            self.new_line(None);
        }
        if let Some(line1) = self.line1.clone() {
            self.append(
                &(line1.clone()
                    + &self
                        .timestamp
                        .as_ref()
                        .map_or(String::new(), |timestamp| " - ".to_owned() + timestamp)),
            );
            self.new_line(Some(&line1));
        } else if let Some(timestamp) = self.timestamp.clone() {
            self.append(&timestamp);
            self.new_line(Some(&timestamp));
        }
        if let Some(line2) = self.line2.clone() {
            self.append(&line2);
            self.new_line(Some(&line2));
        }
        if let Some(string_array) = self.string_array.clone() {
            for string in string_array.into_iter().flatten() {
                self.append(&string);
                self.new_line(Some(&string));
            }
        }
        if let Some(line_list) = self.line_list.clone() {
            for line in line_list.into_iter().flatten() {
                self.append(&line);
                self.new_line(Some(&line));
            }
        }
        if let Some(file) = self
            .file
            .clone()
            .filter(|file| file.exists() && file.is_file() && std::fs::File::open(file).is_ok())
        {
            self.append(&format!(
                "Logging from file: {}",
                crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy()
                )
            ));
            self.new_line(None);
            let emergency_monitor = self.primary_log.borrow().get_manager().map(|manager| {
                manager.get_emergency_monitor(self.primary_log.borrow().get_axis_id())
            });
            match LogFile::get_instance_file(Some(&file), emergency_monitor) {
                Ok(log_file) => match log_file.open_reader() {
                    Ok(Some(id)) => {
                        while let Ok(Some(line)) = log_file.read_line(&id) {
                            self.append(&line);
                            self.new_line(Some(&line));
                        }
                    }
                    Ok(None) | Err(_) => {}
                },
                Err(error) => {
                    eprintln!("{error}");
                    eprintln!("Unable to log from file.  {error}");
                }
            }
        }
        if let Some(reader) = self.reader.clone() {
            while reader.borrow().is_readable() {
                let line = reader.borrow_mut().read_line();
                let Some(line) = line else { break };
                self.append(&line);
                self.new_line(Some(&line));
            }
        }
        if !self.init {
            if self.allow_primary_logging {
                self.primary_log.borrow_mut().msg_changed();
            }
            if let Some(secondary_log) = &self.secondary_log {
                secondary_log.borrow_mut().flush();
            }
        }
        if matches!(self.axis_id, Some(AxisID::First | AxisID::Second)) {
            let axis = self.axis_id.unwrap();
            self.append(&(axis.to_string() + " axis"));
            self.new_line(None);
        }
    }

    /// Java private `newLine(String)`.
    fn new_line(&mut self, previous_line: Option<&str>) {
        let prev_line_end_offset = if self.allow_primary_logging {
            self.primary_log.borrow().get_prev_line_end_offset()
        } else if let Some(secondary_log) = &self.secondary_log {
            secondary_log.borrow().get_prev_line_end_offset()
        } else {
            Ok(0)
        };
        match prev_line_end_offset {
            Ok(offset)
                if offset != 0
                    && (previous_line.is_none() || !previous_line.unwrap().ends_with('\n')) =>
            {
                self.append("\n")
            }
            Ok(_) => {}
            Err(error) => eprintln!("{}", error.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::base_manager::BaseManager;

    #[derive(Default)]
    struct Log {
        text: String,
        changed: usize,
    }
    impl LogInterface for Log {
        fn get_manager(&self) -> Option<&'static dyn BaseManager> {
            None
        }
        fn get_axis_id(&self) -> Option<AxisID> {
            None
        }
        fn log_message_array(
            &mut self,
            _: Option<&str>,
            _: Option<AxisID>,
            _: Option<&[Option<String>]>,
            _: Option<&str>,
        ) -> bool {
            false
        }
        fn log_message_list(
            &mut self,
            _: Option<&str>,
            _: Option<AxisID>,
            _: Option<&[Option<String>]>,
        ) {
        }
        fn log_message_axis_list(&mut self, _: Option<AxisID>, _: Option<&[Option<String>]>) {}
        fn log_message_loggable(&mut self, _: Option<&dyn Loggable>, _: Option<AxisID>) {}
        fn log_message_title_axis(&mut self, _: Option<&str>, _: Option<AxisID>) {}
        fn log_message(&mut self, _: Option<&str>) {}
        fn log_message_secondary(
            &mut self,
            _: Option<&str>,
            _: bool,
            _: bool,
            _: Option<FileWriterRef>,
        ) {
        }
        fn log_message_file(&mut self, _: Option<&Path>, _: Option<FileWriterRef>) {}
        fn log_message_file_newline(
            &mut self,
            _: Option<&Path>,
            _: bool,
            _: Option<FileWriterRef>,
        ) {
        }
        fn log_message_primary_log(&mut self, _: Option<FileReaderRef>) {}
        fn save(&mut self) {}
        fn set_allow_primary_logging(&mut self, _: bool) {}
        fn is_allow_primary_logging(&self) -> bool {
            true
        }
        fn append(&mut self, line: &str) {
            self.text.push_str(line);
        }
        fn msg_changed(&mut self) {
            self.changed += 1;
        }
        fn get_prev_line_end_offset(
            &self,
        ) -> Result<usize, super::super::log_interface::BadLocationException> {
            Ok(self
                .text
                .rfind('\n')
                .map_or(self.text.len(), |offset| offset + 1))
        }
    }
    #[derive(Default)]
    struct Queue {
        tasks: Vec<AppendLater>,
    }
    impl SwingUtilities for Queue {
        fn invoke_later(&mut self, runnable: AppendLater) {
            self.tasks.push(runnable);
        }
    }

    #[test]
    fn queue_captures_flag_and_preserves_append_newline_order() {
        let concrete_log = Rc::new(RefCell::new(Log::default()));
        let log: Rc<RefCell<dyn LogInterface>> = concrete_log.clone();
        let queue = Rc::new(RefCell::new(Queue::default()));
        let mut logger = EtomoLogger::new(log.clone(), queue.clone());
        logger.log_message_lines(Some("first".to_owned()), Some("second".to_owned()));
        logger.set_allow_primary_logging(false);
        let mut task = queue.borrow_mut().tasks.remove(0);
        task.run();
        assert!(concrete_log.borrow().text.starts_with("first - "));
        assert!(concrete_log.borrow().text.contains("\nsecond\n"));
        assert_eq!(concrete_log.borrow().changed, 1);
    }
}
