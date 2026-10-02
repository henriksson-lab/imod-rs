//! `IMOD/Etomo/src/etomo/process/PatchcorrProcessWatcher.java`.
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
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `PatchcorrProcessWatcher`'s own fields.
pub struct PatchcorrProcessWatcher {
    /// Java field `lastLineRead`.
    last_line_read: Mutex<Option<String>>,
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
}

impl PatchcorrProcessWatcher {
    /// Java `PatchcorrProcessWatcher(ApplicationManager, AxisID,
    /// ProcessResultDisplay)`.  Construct a xcorr process watcher.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<LogFileProcessMonitorOf<PatchcorrProcessWatcher>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            axis_id,
            process_result_display,
            Some(ProcessName::PATCHCORR),
            PatchcorrProcessWatcher {
                last_line_read: Mutex::new(None),
                application_manager: manager,
            },
        );
        monitor
            .standard_log_file_name
            .store(false, Ordering::SeqCst);
        *monitor.log_file_basename.lock().unwrap() = Some(dataset_files::PATCH_OUT.to_string());
        monitor
    }
}

impl LogFileProcessMonitorImpl for PatchcorrProcessWatcher {
    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        "Combine: patchcorr".to_string()
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
            if !java_lang_string_trim(&line).contains("positions") {
                base.current_section.fetch_add(1, Ordering::SeqCst);
            }
            *last_line_read = Some(line);
        }
        if base.current_section.load(Ordering::SeqCst) >= base.n_sections.load(Ordering::SeqCst) {
            base.ending.store(true, Ordering::SeqCst);
        }
        Ok(last_line_read.clone())
    }

    /// Java `findNSections`.  Search patch.out file for the number of
    /// positions.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        // Search for the number of sections, we should see a header ouput first
        let mut found_n_sections = false;
        base.n_sections.store(-1, Ordering::SeqCst);
        while base.is_running() && !found_n_sections {
            monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
            let line = base.read_log_file_line()?;
            if let Some(line) = line {
                if java_lang_string_trim(&line).contains("positions") {
                    let line = java_lang_string_trim(&line);
                    let strings = utilities::java_lang_string_split(line, &WHITESPACE);
                    base.n_sections.store(
                        java_lang_integer_parse_int(&strings[0])
                            .map_err(MonitorError::NumberFormat)?,
                        Ordering::SeqCst,
                    );
                    found_n_sections = true;
                }
            }
        }
        Ok(())
    }

    /// Java `postProcess`.
    fn post_process(&self, base: &LogFileProcessMonitor) {
        self.application_manager
            .post_process(base.axis_id, Some(ProcessName::PATCHCORR), None, None);
    }
}
