//! `IMOD/Etomo/src/etomo/process/VolcombineProcessMonitor.java`.
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
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::utilities;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Java `VolcombineProcessMonitor`'s own fields.
pub struct VolcombineProcessMonitor {
    /// Java field `subprocess`.
    subprocess: Subprocess,
}

impl VolcombineProcessMonitor {
    /// Java `VolcombineProcessMonitor(BaseManager, AxisID,
    /// ProcessResultDisplay)`.  Default constructor.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<LogFileProcessMonitorOf<VolcombineProcessMonitor>> {
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            process_result_display,
            Some(ProcessName::VOLCOMBINE),
            VolcombineProcessMonitor {
                subprocess: Subprocess::new(),
            },
        );
        *monitor.log_file_basename.lock().unwrap() = Some("volcombine".to_string());
        monitor
    }

    /// Java static `setSubprocess`.  Sets the subprocess.  Returns true if
    /// there is a current subprocess.
    pub fn set_subprocess(line: &str, subprocess: &Subprocess) -> bool {
        if line.contains("REASSEMBLING PIECES") {
            subprocess.set_reassembling();
            return true;
        }
        if line.contains("RUNNING FILLTOMO") {
            subprocess.set_filltomo();
            return true;
        }
        if line.contains("RUNNING DENSMATCH TO MATCH DENSITIES") {
            subprocess.set_densmatch();
            return true;
        }
        subprocess.reset();
        false
    }

    /// Java private `parseFields`.
    fn parse_fields(
        &self,
        fields: &[String],
        location: usize,
        old_value: i32,
    ) -> Result<i32, MonitorError> {
        if fields.len() > location {
            return java_lang_integer_parse_int(&fields[location])
                .map_err(MonitorError::NumberFormat);
        }
        Ok(old_value)
    }
}

impl LogFileProcessMonitorImpl for VolcombineProcessMonitor {
    /// Java protected `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        "Combine: volcombine".to_string()
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
            if line.starts_with("STATUS:")
                && !VolcombineProcessMonitor::set_subprocess(&line, &self.subprocess)
                && line.contains("EXTRACTING AND COMBINING")
            {
                let fields = utilities::java_lang_string_split(&line, &WHITESPACE);
                base.current_section.store(
                    self.parse_fields(&fields, 5, base.current_section.load(Ordering::SeqCst))?,
                    Ordering::SeqCst,
                );
            }
            prev_line = Some(line);
        }
        Ok(prev_line)
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
                if found_n_sections {
                    break;
                }
                if line.starts_with("STATUS:") {
                    if VolcombineProcessMonitor::set_subprocess(&line, &self.subprocess) {
                        self.update_progress_bar(base);
                    } else if line.contains("EXTRACTING AND COMBINING") {
                        let fields = utilities::java_lang_string_split(&line, &WHITESPACE);
                        base.n_sections.store(
                            self.parse_fields(&fields, 7, base.n_sections.load(Ordering::SeqCst))?,
                            Ordering::SeqCst,
                        );
                        if base.n_sections.load(Ordering::SeqCst) != i32::MIN {
                            found_n_sections = true;
                        } else {
                            return Err(MonitorError::NumberFormat(
                                "Unable to read first STATUS: EXTRACTING AND COMBINING line"
                                    .to_string(),
                            ));
                        }
                        base.current_section.store(
                            self.parse_fields(
                                &fields,
                                5,
                                base.current_section.load(Ordering::SeqCst),
                            )?,
                            Ordering::SeqCst,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Java `updateProgressBar`.
    fn update_progress_bar(&self, base: &LogFileProcessMonitor) {
        let axis_id = base.axis_id;
        if !base.ending.load(Ordering::SeqCst) {
            if self.subprocess.is_filltomo() {
                base.manager.post_main_panel(Box::new(move |panel| {
                    panel.set_progress_bar_value_int_string_axis_id(0, Some("Filltomo"), axis_id);
                }));
            } else if self.subprocess.is_reassembling() {
                base.manager.post_main_panel(Box::new(move |panel| {
                    panel.set_progress_bar_value_int_string_axis_id(0, Some("Reassembling"), axis_id);
                }));
            } else if self.subprocess.is_densmatch() {
                base.manager.post_main_panel(Box::new(move |panel| {
                    panel.set_progress_bar_value_int_string_axis_id(0, Some("Densmatch"), axis_id);
                }));
            } else {
                base.update_progress_bar();
            }
        } else {
            base.update_progress_bar();
        }
    }
}

/// Java static nested class `VolcombineProcessMonitor.Subprocess`.  Shared by
/// a monitor thread and its readers, so its flags are atomics.
pub struct Subprocess {
    /// Java field `reassembling`.
    reassembling: AtomicBool,
    /// Java field `filltomo`.
    filltomo: AtomicBool,
    /// Java field `densmatch`.
    densmatch: AtomicBool,
}

impl Default for Subprocess {
    fn default() -> Subprocess {
        Subprocess::new()
    }
}

impl Subprocess {
    /// Java `Subprocess()`.
    pub fn new() -> Subprocess {
        Subprocess {
            reassembling: AtomicBool::new(false),
            filltomo: AtomicBool::new(false),
            densmatch: AtomicBool::new(false),
        }
    }

    /// Java private `setReassembling`.
    fn set_reassembling(&self) {
        self.reassembling.store(true, Ordering::SeqCst);
        self.filltomo.store(false, Ordering::SeqCst);
        self.densmatch.store(false, Ordering::SeqCst);
    }

    /// Java `setFilltomo`.
    pub fn set_filltomo(&self) {
        self.filltomo.store(true, Ordering::SeqCst);
        self.reassembling.store(false, Ordering::SeqCst);
        self.densmatch.store(false, Ordering::SeqCst);
    }

    /// Java `setDensmatch`.
    pub fn set_densmatch(&self) {
        self.densmatch.store(true, Ordering::SeqCst);
        self.filltomo.store(false, Ordering::SeqCst);
        self.reassembling.store(false, Ordering::SeqCst);
    }

    /// Java `reset`.
    pub fn reset(&self) {
        self.densmatch.store(false, Ordering::SeqCst);
        self.filltomo.store(false, Ordering::SeqCst);
        self.reassembling.store(false, Ordering::SeqCst);
    }

    /// Java `isFilltomo`.
    pub fn is_filltomo(&self) -> bool {
        self.filltomo.load(Ordering::SeqCst)
    }

    /// Java `isDensmatch`.
    pub fn is_densmatch(&self) -> bool {
        self.densmatch.load(Ordering::SeqCst)
    }

    /// Java `isReassembling`.
    pub fn is_reassembling(&self) -> bool {
        self.reassembling.load(Ordering::SeqCst)
    }
}
