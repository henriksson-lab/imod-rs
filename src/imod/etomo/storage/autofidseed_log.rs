//! `IMOD/Etomo/src/etomo/storage/AutofidseedLog.java`.
//!
//! Reads the autofidseed log: the lines worth logging, and whether autofidseed
//! adjusted the tracking parameters.  Built and read on the event dispatch
//! thread; the `lineList` field Java refills on each call is a `RefCell`.

use std::cell::RefCell;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `public final class AutofidseedLog implements Loggable`.
pub struct AutofidseedLog {
    /// Java `private final ArrayList<String> lineList`.
    line_list: RefCell<Vec<String>>,

    /// Java `private final BaseManager manager` (never null here).
    manager: &'static dyn BaseManager,
    /// Java `private final AxisID axisID`.
    axis_id: AxisID,
    /// Java `private final String userDir`.
    user_dir: Option<String>,
}

impl AutofidseedLog {
    /// Java private `AutofidseedLog(BaseManager, AxisID, String)`.
    fn new(manager: &'static dyn BaseManager, axis_id: AxisID, user_dir: Option<&str>) -> Self {
        AutofidseedLog {
            line_list: RefCell::new(Vec::new()),
            manager,
            axis_id,
            user_dir: user_dir.map(str::to_owned),
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, String)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        user_dir: Option<&str>,
    ) -> AutofidseedLog {
        AutofidseedLog::new(manager, axis_id, user_dir)
    }

    // <p>Updates done</p>

    /// Java `getName()`.
    pub fn get_name(&self) -> String {
        ProcessName::AUTOFIDSEED.to_string()
    }

    /// Java `getLogMessage()`.  Get a message to be logged in the LogPanel.
    pub fn get_log_message(&self) -> Result<Vec<String>, LogFileError> {
        self.line_list.borrow_mut().clear();
        // refresh the log file
        // `manager != null ? manager.getEmergencyMonitor(axisID) : null`: the
        // manager is never null here.  A null userDir is `new File(null, name)`,
        // the name alone.
        let log = LogFile::get_instance_process_name(
            self.user_dir.as_deref().unwrap_or(""),
            self.axis_id,
            ProcessName::AUTOFIDSEED,
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        if log.exists() {
            let reader_id = log.open_reader()?;
            if let Some(reader_id) = reader_id
                && !reader_id.is_empty()
            {
                let mut line = log.read_line(&reader_id)?;
                while let Some(current) = line {
                    if current.contains("candidate points") {
                        self.line_list.borrow_mut().push(current);
                    } else if java_lang_string_trim(&current).starts_with("Final:") {
                        self.line_list.borrow_mut().push(current);
                    }
                    // Look for "Tracking parameters adjusted for new unbinned bead size"
                    // message
                    // Also look for AFS3.
                    else if current.contains("AFS1") || current.contains("AFS3") {
                        self.line_list.borrow_mut().push(current);
                    }
                    line = log.read_line(&reader_id)?;
                }
                log.close_id(Some(&*reader_id));
                return Ok(self.line_list.borrow().clone());
            }
        }
        Ok(self.line_list.borrow().clone())
    }

    /// Java `isTrackingAdjusted()`.
    pub fn is_tracking_adjusted(&self) -> Result<bool, LogFileError> {
        // refresh the log file
        let log = LogFile::get_instance_process_name(
            self.user_dir.as_deref().unwrap_or(""),
            self.axis_id,
            ProcessName::AUTOFIDSEED,
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        if log.exists() {
            let reader_id = log.open_reader()?;
            if let Some(reader_id) = reader_id
                && !reader_id.is_empty()
            {
                while let Some(line) = log.read_line(&reader_id)? {
                    if line.contains("AFS1")
                        || line.contains("AFS2")
                        || line.contains("AFS3")
                        || line.contains("AFS4")
                    {
                        log.close_id(Some(&*reader_id));
                        return Ok(true);
                    }
                }
                log.close_id(Some(&*reader_id));
            }
        }
        Ok(false)
    }
}

impl Loggable for AutofidseedLog {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        AutofidseedLog::get_name(self)
    }

    /// Java `getLogMessage() throws LogFileException, IOException, LockException`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        AutofidseedLog::get_log_message(self)
            .map(|list| list.into_iter().map(Some).collect())
            .map_err(|e| match e {
                LogFileError::Lock(_) => LoggableException::Lock(e.get_message()),
                LogFileError::Io(_) => LoggableException::Io(e.get_message()),
                _ => LoggableException::LogFile(e.get_message()),
            })
    }
}
