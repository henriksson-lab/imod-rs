//! `IMOD/Etomo/src/etomo/process/BlendmontProcessMonitor.java`.
//!
//! A `LogFileProcessMonitor` subclass; see the module comment of
//! `log_file_process_monitor.rs` for how subclasses are represented.

use super::log_file_process_monitor::{
    LogFileProcessMonitor, LogFileProcessMonitorImpl, LogFileProcessMonitorOf, MonitorError,
};
use super::monitor_tool_kit::WHITESPACE;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::{BlendmontParam, Mode};
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::montagesize::Montagesize;
use crate::imod::etomo::util::utilities;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Java `BlendmontProcessMonitor`'s own fields.
pub struct BlendmontProcessMonitor {
    /// Java field `title`.
    title: String,
    /// Java field `mode`.
    mode: Mode,
    /// Java field `lastLineFound`.
    last_line_found: AtomicBool,
    /// Java field `doingMrctaper`.
    doing_mrctaper: AtomicBool,
}

impl BlendmontProcessMonitor {
    /// Java `BlendmontProcessMonitor(BaseManager, AxisID, Mode)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
        mode: Mode,
    ) -> Arc<LogFileProcessMonitorOf<BlendmontProcessMonitor>> {
        let title = if mode == Mode::Xcorr {
            "Cross-correlation"
        } else if mode == Mode::Preblend {
            "Coarse alignment"
        } else if mode == Mode::Blend || mode == Mode::Blend3dFind {
            "Full alignment"
        } else if mode == Mode::Undistort {
            "Distortion correction"
        } else if mode == Mode::WholeTomogramSample {
            "Whole tomogram"
        } else if mode == Mode::SerialSectionPreblend {
            "Initial blend"
        } else if mode == Mode::SerialSectionBlend {
            "Blend serial sections"
        } else {
            ""
        };
        let monitor = LogFileProcessMonitorOf::new(
            manager,
            id,
            None,
            Some(BlendmontParam::get_process_name_for_mode(mode)),
            BlendmontProcessMonitor {
                title: title.to_string(),
                mode,
                last_line_found: AtomicBool::new(false),
                doing_mrctaper: AtomicBool::new(false),
            },
        );
        *monitor.log_file_basename.lock().unwrap() =
            Some(BlendmontParam::get_process_name_for_mode(mode).to_string());
        monitor
    }
}

impl LogFileProcessMonitorImpl for BlendmontProcessMonitor {
    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.initialize_progress_bar_label(
            &self.get_title(base),
            base.n_sections.load(Ordering::SeqCst),
        );
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &LogFileProcessMonitor) -> String {
        self.title.clone()
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
            let line = java_lang_string_trim(&line).to_string();
            if line.starts_with("working on section #") {
                let strings = utilities::java_lang_string_split(&line, &WHITESPACE);
                // set currentSection - section in log starts from 0
                // Fixed in translation: BlendmontProcessMonitor.java:89 reads
                // `strings[4]` unchecked; a short line throws an
                // ArrayIndexOutOfBoundsException that ends the monitor thread
                // before `progressBarDone`.  The translation reports it as the
                // NumberFormatException the method declares.
                let string = strings.get(4).ok_or_else(|| {
                    MonitorError::NumberFormat(
                        "Index 4 out of bounds for length ".to_string()
                            + &strings.len().to_string(),
                    )
                })?;
                base.current_section.store(
                    java_lang_integer_parse_int(string)
                        .map_err(MonitorError::NumberFormat)?
                        .wrapping_add(1),
                    Ordering::SeqCst,
                );
            } else if self.mode == Mode::Blend
                || self.mode == Mode::Blend3dFind
                || self.mode == Mode::WholeTomogramSample
            {
                if line.starts_with("Doing section #")
                    && !self.doing_mrctaper.load(Ordering::SeqCst)
                {
                    self.doing_mrctaper.store(true, Ordering::SeqCst);
                    let label = self.get_title(base) + ": mrctaper";
                    let axis_id = base.axis_id;
                    base.manager.post_main_panel(Box::new(move |panel| {
                        panel.set_progress_bar_string_int_boolean_axis_id(Some(&label), 1, false, axis_id);
                    }));
                } else if base.current_section.load(Ordering::SeqCst)
                    >= base.n_sections.load(Ordering::SeqCst)
                    && line.starts_with("Done!")
                {
                    self.last_line_found.store(true, Ordering::SeqCst);
                }
            } else if self.mode == Mode::Xcorr {
                if line.starts_with(process_output_strings::START_PARAMETERS_TAG)
                    && line.contains(&ProcessName::TILT_XCORR.to_string())
                {
                    // It's starting tiltxcorr. Blendmont must be complete.
                    base.ending.store(true, Ordering::SeqCst);
                    base.halt_process(false, Some(ProcessEndState::Done));
                }
            } else if self.mode == Mode::SerialSectionPreblend
                && line.contains(process_output_strings::SUCCESS_TAG)
            {
                base.ending.store(true, Ordering::SeqCst);
                base.halt_process(false, Some(ProcessEndState::Done));
            }
            prev_line = Some(line);
        }
        // Set ending on the last section
        if base.current_section.load(Ordering::SeqCst) >= base.n_sections.load(Ordering::SeqCst)
            && ((self.mode != Mode::Blend
                && self.mode != Mode::Blend3dFind
                && self.mode != Mode::WholeTomogramSample)
                || self.last_line_found.load(Ordering::SeqCst))
        {
            base.ending.store(true, Ordering::SeqCst);
        }
        Ok(prev_line)
    }

    /// Java `findNSections`.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        // Upstream bug fixed in translation (BlendmontProcessMonitor.java:130-132):
        // `Montagesize.getInstance` returns null when the raw stack's file cannot be
        // named, and `read` then throws NullPointerException out of the monitor.  Here
        // the section count is left unchanged.
        let montagesize = match Montagesize::get_instance(
            base.manager,
            base.axis_id,
            &file_type::CLASS.raw_stack,
            false,
        ) {
            None => return Ok(()),
            Some(montagesize) => montagesize,
        };
        montagesize
            .read(base.manager)
            .map_err(|e| MonitorError::LogFile(LogFileError::Io(std::io::Error::other(e))))?;
        base.n_sections
            .store(montagesize.get_z().get_int(), Ordering::SeqCst);
        Ok(())
    }
}
