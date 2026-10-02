//! `IMOD/Etomo/src/etomo/process/CCDEraserProcessMonitor.java`.
//!
//! A `LogFileProcessMonitor` subclass; see the module comment of
//! `log_file_process_monitor.rs` for how subclasses are represented.

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
};
use super::monitor_tool_kit::WHITESPACE;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Java `CCDEraserProcessMonitor` (it adds no fields).
pub struct CCDEraserProcessMonitor;

impl CCDEraserProcessMonitor {
    /// Java `CCDEraserProcessMonitor(BaseManager, AxisID)`.  Default
    /// constructor.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
    ) -> Arc<LogFileProcessMonitorOf<CCDEraserProcessMonitor>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            None,
            Some(ProcessName::CCDERASER),
            CCDEraserProcessMonitor,
        );
        *monitor.log_file_basename.lock().unwrap() = Some("eraser".to_string());
        monitor
    }
}

impl LogFileProcessMonitorImpl for CCDEraserProcessMonitor {
    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        "CCD Eraser".to_string()
    }

    /// Java `getCurrentSection`.  Returns the last line found.
    fn get_current_section(
        &self,
        base: &LogFileProcessMonitor,
    ) -> Result<Option<String>, MonitorError> {
        let mut prev_line: Option<String> = None;
        while base.is_running() {
            let line = match base.read_log_file_line()? {
                None => break,
                Some(line) => line,
            };
            if line.starts_with("Section") {
                let fields = utilities::java_lang_string_split(&line, &WHITESPACE);
                if fields.len() > 1 {
                    let number = &fields[1];
                    base.current_section.store(
                        java_lang_integer_parse_int(number).map_err(MonitorError::NumberFormat)?,
                        Ordering::SeqCst,
                    );
                }
            }
            prev_line = Some(line);
        }
        Ok(prev_line)
    }
}
