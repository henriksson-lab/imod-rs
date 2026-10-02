//! `IMOD/Etomo/src/etomo/process/AlignFramesProcessMonitor.java`.
//!
//! A `LogFileProcessMonitor` subclass for `alignframes`; see the module comment
//! of `log_file_process_monitor.rs` for how subclasses are represented.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
};
use super::monitor_tool_kit::{self, WHITESPACE};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities;

/// Java `public final class AlignFramesProcessMonitor extends
/// LogFileProcessMonitor` (it adds no fields).
pub struct AlignFramesProcessMonitor;

impl AlignFramesProcessMonitor {
    /// Java package-private `AlignFramesProcessMonitor(BaseManager, AxisID,
    /// String)`.  Default constructor.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
        log_filename: Option<String>,
    ) -> Arc<LogFileProcessMonitorOf<AlignFramesProcessMonitor>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            None,
            Some(ProcessName::ALIGN_FRAMES),
            AlignFramesProcessMonitor,
        );
        *monitor.log_file_basename.lock().unwrap() = log_filename;
        monitor
            .standard_log_file_name
            .store(false, Ordering::SeqCst);
        monitor
    }
}

impl LogFileProcessMonitorImpl for AlignFramesProcessMonitor {
    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle()`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        "Align Frames".to_string()
    }

    /// Java `getCurrentSection()`.  Returns the last line found.
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
            if line.starts_with("File") && line.contains("frames") {
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

    /// Java `findNSections()`.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        // Search for the number of sections, we should see a header output first
        let _found_n_sections = false;
        base.n_sections.store(-1, Ordering::SeqCst);
        while base.is_running() && base.is_process_running() && !base.is_stop() {
            if let Some(line) = base.read_log_file_line()? {
                if line.starts_with("ERROR") {
                    base.stop();
                }
                if line.starts_with("Number of sets of frames to align") {
                    let fields = utilities::java_lang_string_split(&line, &WHITESPACE);
                    if fields.len() > 1 {
                        // Upstream bug fixed in translation
                        // (AlignFramesProcessMonitor.java:83): Java tests for more than
                        // one field and then reads fields[8], throwing
                        // ArrayIndexOutOfBoundsException out of the monitor on a short
                        // line.  Here a line without a ninth field is skipped.
                        let Some(number) = fields.get(8) else {
                            continue;
                        };
                        base.n_sections.store(
                            java_lang_integer_parse_int(number)
                                .map_err(MonitorError::NumberFormat)?,
                            Ordering::SeqCst,
                        );
                        return Ok(());
                    }
                }
            } else {
                let sleep = 5;
                // Run, kill, and rerun. Wait for a bit to get a longer timeout.
                // Test with debug to see if the popup works.
                monitor_tool_kit::sleep(&base.interrupted, sleep)?;
            }
        }
        Ok(())
    }
}
