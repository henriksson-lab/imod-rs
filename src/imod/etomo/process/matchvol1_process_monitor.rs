//! `IMOD/Etomo/src/etomo/process/Matchvol1ProcessMonitor.java`.
//!
//! A `LogFileProcessMonitor` subclass; see the module comment of
//! `log_file_process_monitor.rs` for how subclasses are represented.

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
    UPDATE_PERIOD,
};
use super::monitor_tool_kit::{self, WHITESPACE};
use super::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `Matchvol1ProcessMonitor`'s own fields.
pub struct Matchvol1ProcessMonitor {
    /// Java field `calledFromFlatten`.
    called_from_flatten: bool,
    /// Java field `lastLineRead`.
    last_line_read: Mutex<Option<String>>,
}

impl Matchvol1ProcessMonitor {
    /// Java private `Matchvol1ProcessMonitor(BaseManager, AxisID, boolean,
    /// FileType, ProcessResultDisplay)`.  Construct a matchvol1 process
    /// watcher.
    fn new_full(
        manager: &'static dyn BaseManager,
        id: AxisID,
        called_from_flatten: bool,
        file_type: Option<&FileType>,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<LogFileProcessMonitorOf<Matchvol1ProcessMonitor>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            process_result_display,
            Some(ProcessName::FLATTEN),
            Matchvol1ProcessMonitor {
                called_from_flatten,
                last_line_read: Mutex::new(None),
            },
        );
        let log_file_basename = if called_from_flatten {
            match file_type {
                None => ProcessName::FLATTEN.to_string(),
                Some(file_type) => file_type.get_root(Some(manager), Some(id)),
            }
        } else {
            "matchvol1".to_string()
        };
        *monitor.log_file_basename.lock().unwrap() = Some(log_file_basename);
        monitor
    }

    /// Java `Matchvol1ProcessMonitor(BaseManager, AxisID,
    /// ProcessResultDisplay)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<LogFileProcessMonitorOf<Matchvol1ProcessMonitor>> {
        Matchvol1ProcessMonitor::new_full(manager, id, false, None, process_result_display)
    }

    /// Java static `getFlattenInstance`.
    pub fn get_flatten_instance(
        manager: &'static dyn BaseManager,
        id: AxisID,
        file_type: Option<&FileType>,
    ) -> Arc<LogFileProcessMonitorOf<Matchvol1ProcessMonitor>> {
        Matchvol1ProcessMonitor::new_full(manager, id, true, file_type, None)
    }
}

impl LogFileProcessMonitorImpl for Matchvol1ProcessMonitor {
    /// Java `initializeProgressBar()`.  Sets the title and the number of steps
    /// of the progress bar.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        // The source computes this local and then passes getTitle() instead.
        let _title: String = if self.called_from_flatten {
            ProcessName::FLATTEN.to_string()
        } else {
            "Combine: matchvol1".to_string()
        };
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        if self.called_from_flatten {
            return ProcessName::FLATTEN.to_string();
        }
        "Combine: matchvol1".to_string()
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
                // Fixed in translation: Matchvol1ProcessMonitor.java:97 reads
                // `strings[1]` unchecked; a bare "Finished" line throws an
                // ArrayIndexOutOfBoundsException that ends the monitor thread.
                // The translation reports it as a NumberFormatException.
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
                    // Fixed in translation: Matchvol1ProcessMonitor.java:125
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
                }
            }
        }
        monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
        Ok(())
    }
}
