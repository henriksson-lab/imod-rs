//! `IMOD/Etomo/src/etomo/process/MatchorwarpProcessMonitor.java`.
//!
//! A `LogFileProcessMonitor` subclass; see the module comment of
//! `log_file_process_monitor.rs` for how subclasses are represented.

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
    UPDATE_PERIOD,
};
use super::monitor_tool_kit::{self, WHITESPACE};
use super::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};
use crate::imod::etomo::util::utilities;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `MatchorwarpProcessMonitor`'s own fields.
pub struct MatchorwarpProcessMonitor {
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
    /// Java field `lastLineRead`.
    last_line_read: Mutex<Option<String>>,
}

impl MatchorwarpProcessMonitor {
    /// Java `MatchorwarpProcessMonitor(ApplicationManager, AxisID,
    /// ProcessResultDisplay)`.  Construct a matchvol1 process watcher.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<LogFileProcessMonitorOf<MatchorwarpProcessMonitor>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            axis_id,
            process_result_display,
            None,
            MatchorwarpProcessMonitor {
                application_manager: manager,
                last_line_read: Mutex::new(None),
            },
        );
        monitor.set_debug(true);
        *monitor.log_file_basename.lock().unwrap() = Some("matchorwarp".to_string());
        monitor
    }
}

impl LogFileProcessMonitorImpl for MatchorwarpProcessMonitor {
    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        "Combine: matchorwarp".to_string()
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
            let line = java_lang_string_trim(&line).to_string();
            if line.starts_with("Finished") {
                let strings = utilities::java_lang_string_split(&line, &WHITESPACE);
                // Fixed in translation: MatchorwarpProcessMonitor.java:63 reads
                // `strings[1]` unchecked (ArrayIndexOutOfBoundsException ends the
                // monitor thread); reported as a NumberFormatException instead.
                let string = strings.get(1).ok_or_else(|| {
                    MonitorError::NumberFormat(
                        "Index 1 out of bounds for length ".to_string()
                            + &strings.len().to_string(),
                    )
                })?;
                base.current_section.store(
                    java_lang_integer_parse_int(string).map_err(MonitorError::NumberFormat)?,
                    Ordering::SeqCst,
                );
            }
            *last_line_read = Some(line);
        }
        if base.current_section.load(Ordering::SeqCst) >= base.n_sections.load(Ordering::SeqCst) {
            base.ending.store(true, Ordering::SeqCst);
        }
        Ok(last_line_read.clone())
    }

    /// Java `findNSections`.  Search matchvol1.log.out file for the number of
    /// positions.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        // Search for the number of sections, we should see a header ouput first
        let mut found_n_sections = false;
        base.n_sections.store(-1, Ordering::SeqCst);
        monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
        while base.is_running() && !found_n_sections {
            let line = base.read_log_file_line()?;
            if line.is_none() {
                monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
            }
            if let Some(line) = line {
                if java_lang_string_trim(&line).starts_with("Finished") {
                    let line = java_lang_string_trim(&line);
                    let strings = utilities::java_lang_string_split(line, &WHITESPACE);
                    // Fixed in translation: MatchorwarpProcessMonitor.java:93
                    // reads `strings[3]` unchecked (ArrayIndexOutOfBoundsException
                    // ends the monitor thread); reported as a
                    // NumberFormatException instead.
                    let string = strings.get(3).ok_or_else(|| {
                        MonitorError::NumberFormat(
                            "Index 3 out of bounds for length ".to_string()
                                + &strings.len().to_string(),
                        )
                    })?;
                    base.n_sections.store(
                        java_lang_integer_parse_int(string).map_err(MonitorError::NumberFormat)?,
                        Ordering::SeqCst,
                    );
                    found_n_sections = true;
                } else if java_lang_string_trim(&line)
                    .starts_with("MATCHORWARP: CREATED patch_vector.mod")
                {
                    self.application_manager.msg_patch_vector_created();
                }
            }
        }
        monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
        Ok(())
    }
}
