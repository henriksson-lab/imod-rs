//! `IMOD/Etomo/src/etomo/storage/FlattenWarpLog.java`.
//!
//! Represents the flattenwarp log.  Built on the process thread that finished
//! flattenwarp and read on the event dispatch thread; the fields Java assigns
//! after construction are `RefCell`s.

use std::cell::RefCell;

use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};

/// Java `public final class FlattenWarpLog implements Loggable`.
pub struct FlattenWarpLog {
    /// Java `private final ArrayList<String> lineList`.
    line_list: RefCell<Vec<String>>,

    /// Java `private String[] log`.
    log: RefCell<Option<Vec<String>>>,
}

impl Default for FlattenWarpLog {
    fn default() -> Self {
        FlattenWarpLog::new()
    }
}

impl FlattenWarpLog {
    /// Java's implicit `FlattenWarpLog()`.
    pub fn new() -> FlattenWarpLog {
        FlattenWarpLog {
            line_list: RefCell::new(Vec::new()),
            log: RefCell::new(None),
        }
    }

    /// Java `setLog(String[])`.  Set the log and clear lineList.
    pub fn set_log(&self, log: Option<Vec<String>>) {
        *self.log.borrow_mut() = log;
        self.line_list.borrow_mut().clear();
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> String {
        ProcessName::FLATTEN_WARP.to_string()
    }

    /// Java `getLogMessage()`.  Get a message to be logged in the LogPanel.
    /// Only refreshes lineList if it is empty.  It should be emptied when
    /// this.log is set.  If log changes without setLog being called (because
    /// it is a pointer), it would not necessarily be correct anymore.
    pub fn get_log_message(&self) -> Vec<String> {
        let log = self.log.borrow();
        let Some(log) = log.as_ref() else {
            return self.line_list.borrow().clone();
        };
        if !self.line_list.borrow().is_empty() {
            return self.line_list.borrow().clone();
        }
        for line in log {
            let trimmed = java_lang_string_trim(line);
            if trimmed.starts_with("Minimum spacing between contours is") {
                self.line_list.borrow_mut().push(line.clone());
            } else if trimmed.starts_with("Setting target spacings in X and Y to") {
                self.line_list.borrow_mut().push(line.clone());
            } else if trimmed.starts_with("Mean Z height is") {
                self.line_list.borrow_mut().push(line.clone());
            } else if trimmed.contains("warping transformations written") {
                self.line_list.borrow_mut().push(line.clone());
            }
        }
        self.line_list.borrow().clone()
    }
}

impl Loggable for FlattenWarpLog {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        FlattenWarpLog::get_name(self)
    }

    /// Java `getLogMessage() throws FileNotFoundException, IOException`;
    /// nothing in the body throws.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(FlattenWarpLog::get_log_message(self)
            .into_iter()
            .map(Some)
            .collect())
    }
}
