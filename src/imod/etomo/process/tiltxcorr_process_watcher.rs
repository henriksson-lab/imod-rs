//! `IMOD/Etomo/src/etomo/process/TiltxcorrProcessWatcher.java`.
//!
//! A `LogFileProcessMonitor` subclass; see the module comment of
//! `log_file_process_monitor.rs` for how subclasses are represented.

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
    UPDATE_PERIOD,
};
use super::monitor_tool_kit::{self, WHITESPACE};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::util::utilities;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Java `TILTXCORR_TITLE`.
const TILTXCORR_TITLE: &str = "Cross-correlating stack";

/// Java `TiltxcorrProcessWatcher`'s own fields.
pub struct TiltxcorrProcessWatcher {
    /// Java field `lastLineRead`.
    last_line_read: Mutex<Option<String>>,
    /// Java field `blendmontRan`.
    blendmont_ran: AtomicBool,
    /// Java field `title`.
    title: String,
}

impl TiltxcorrProcessWatcher {
    /// Java `TiltxcorrProcessWatcher(BaseManager, AxisID, ProcessName, boolean,
    /// boolean)`.  Construct an xcorr process watcher.
    pub fn new_process_name(
        manager: &'static dyn BaseManager,
        id: AxisID,
        process_name: ProcessName,
        run_tiltxcorr: bool,
        break_contours: bool,
    ) -> Arc<LogFileProcessMonitorOf<TiltxcorrProcessWatcher>> {
        let title = if run_tiltxcorr {
            TILTXCORR_TITLE
        } else if break_contours {
            "Recutting contours"
        } else {
            "Restoring contours"
        };
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            None,
            Some(process_name),
            TiltxcorrProcessWatcher {
                last_line_read: Mutex::new(None),
                blendmont_ran: AtomicBool::new(false),
                title: title.to_string(),
            },
        );
        *monitor.log_file_basename.lock().unwrap() = process_name.get_text().map(str::to_string);
        monitor
    }

    /// Java `TiltxcorrProcessWatcher(BaseManager, AxisID, boolean)`.
    /// Construct an xcorr process watcher.  `blendmont_ran` - True if
    /// blendmont output is in the log file prior to tiltxcorr output.
    pub fn new_blendmont_ran(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        blendmont_ran: bool,
    ) -> Arc<LogFileProcessMonitorOf<TiltxcorrProcessWatcher>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            axis_id,
            None,
            None,
            TiltxcorrProcessWatcher {
                last_line_read: Mutex::new(None),
                blendmont_ran: AtomicBool::new(blendmont_ran),
                title: TILTXCORR_TITLE.to_string(),
            },
        );
        *monitor.log_file_basename.lock().unwrap() = Some("xcorr".to_string());
        monitor
    }
}

impl LogFileProcessMonitorImpl for TiltxcorrProcessWatcher {
    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        self.title.clone()
    }

    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getCurrentSection`.  Returns the last line found.
    fn get_current_section(
        &self,
        base: &LogFileProcessMonitor,
    ) -> Result<Option<String>, MonitorError> {
        let mut last_line_read = self.last_line_read.lock().unwrap();
        while base.is_running() {
            let line = match base.read_log_file_line()? {
                None => break,
                Some(line) => line,
            };
            if line.starts_with("View") {
                base.current_section.fetch_add(1, Ordering::SeqCst);
            }
            *last_line_read = Some(line);
        }

        if last_line_read.as_deref().is_some_and(|last_line_read| {
            java_lang_string_trim(last_line_read).starts_with("PROGRAM EXECUTED TO END.")
        }) {
            base.ending.store(true, Ordering::SeqCst);
        }
        Ok(last_line_read.clone())
    }

    /// Java `findNSections`.  Search the log file for the header section and
    /// extract the number of sections.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        // Search for the number of sections, we should see a header ouput first
        let mut found_n_sections = false;

        base.n_sections.store(-1, Ordering::SeqCst);
        while base.is_running() && !found_n_sections {
            monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
            while base.is_running() {
                let line = match base.read_log_file_line()? {
                    None => break,
                    Some(line) => line,
                };
                let line = java_lang_string_trim(&line);
                if line.starts_with(base.n_sections_header) {
                    let fields = utilities::java_lang_string_split(line, &WHITESPACE);
                    if fields.len() as i32 > base.n_sections_index {
                        // Take the second header output if there is blendmont output in the
                        // log file
                        if self.blendmont_ran.load(Ordering::SeqCst) {
                            self.blendmont_ran.store(false, Ordering::SeqCst);
                        } else {
                            base.n_sections.store(
                                java_lang_integer_parse_int(
                                    &fields[base.n_sections_index as usize],
                                )
                                .map_err(MonitorError::NumberFormat)?,
                                Ordering::SeqCst,
                            );
                            found_n_sections = true;
                        }
                        break;
                    } else {
                        return Err(MonitorError::NumberFormat(
                            "Incomplete size line in header".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
