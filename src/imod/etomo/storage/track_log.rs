//! `IMOD/Etomo/src/etomo/storage/TrackLog.java`.
//!
//! Reads the beadtrack log for the line worth logging in the project log.
//! Built and read on the event dispatch thread; the `lineList` field Java
//! refills on each call is a `RefCell`.

use std::cell::RefCell;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `public final class TrackLog implements Loggable`.
pub struct TrackLog {
    /// Java `private final ArrayList<String> lineList`.
    line_list: RefCell<Vec<String>>,

    /// Java `private final BaseManager manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java `private final AxisID axisID`.
    axis_id: AxisID,
    /// Java `private final String userDir`.
    user_dir: Option<String>,
}

impl TrackLog {
    /// Java private `TrackLog(BaseManager, AxisID, String)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        user_dir: Option<&str>,
    ) -> TrackLog {
        TrackLog {
            line_list: RefCell::new(Vec::new()),
            manager,
            axis_id,
            user_dir: user_dir.map(str::to_owned),
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, String)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        user_dir: Option<&str>,
    ) -> TrackLog {
        TrackLog::new(manager, axis_id, user_dir)
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> String {
        ProcessName::TRACK.to_string()
    }

    /// Java `getLogMessage()`.  Get a message to be logged in the LogPanel.
    pub fn get_log_message(&self) -> Result<Vec<String>, LogFileError> {
        self.line_list.borrow_mut().clear();
        // refresh the log file
        // A null userDir is `new File(null, name)`, the name alone.
        let track_log = LogFile::get_instance_process_name(
            self.user_dir.as_deref().unwrap_or(""),
            self.axis_id,
            ProcessName::TRACK,
            self.manager
                .map(|manager| manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        if track_log.exists() {
            let reader_id = track_log.open_reader()?;
            if let Some(reader_id) = reader_id
                && !reader_id.is_empty()
            {
                let mut line = track_log.read_line(&reader_id)?;
                while let Some(current) = line {
                    if java_lang_string_trim(&current).starts_with("Total points missing") {
                        self.line_list.borrow_mut().push(current);
                        track_log.close_id(Some(&*reader_id));
                        return Ok(self.line_list.borrow().clone());
                    }
                    line = track_log.read_line(&reader_id)?;
                }
                track_log.close_id(Some(&*reader_id));
            }
        }
        Ok(self.line_list.borrow().clone())
    }
}

impl Loggable for TrackLog {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        TrackLog::get_name(self)
    }

    /// Java `getLogMessage() throws LogFileException, IOException, LockException`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        TrackLog::get_log_message(self)
            .map(|list| list.into_iter().map(Some).collect())
            .map_err(|e| match e {
                LogFileError::Lock(_) => LoggableException::Lock(e.get_message()),
                LogFileError::Io(_) => LoggableException::Io(e.get_message()),
                _ => LoggableException::LogFile(e.get_message()),
            })
    }
}
