//! `IMOD/Etomo/src/etomo/process/ProcesschunksVolcombineMonitor.java`.
//!
//! A `ProcesschunksProcessMonitor` subclass; see the module comment of
//! `processchunks_process_monitor.rs` for how subclasses are represented.

use super::processchunks_process_monitor::{
    ProcesschunksProcessMonitor, ProcesschunksProcessMonitorImpl, UpdateStateError,
};
use super::volcombine_process_monitor::{Subprocess, VolcombineProcessMonitor};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::dataset_files;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Java `ProcesschunksVolcombineMonitor`'s own fields.
pub struct ProcesschunksVolcombineMonitor {
    /// Java field `startLog`.
    start_log: Mutex<Option<Arc<Handle>>>,
    /// Java field `finishLog`.
    finish_log: Mutex<Option<Arc<Handle>>>,
    /// Java field `subprocess`.
    subprocess: Subprocess,
    /// Java field `readerIdStart`.
    reader_id_start: Mutex<Option<ReaderId>>,
    /// Java field `readerIdFinish`.
    reader_id_finish: Mutex<Option<ReaderId>>,
}

impl ProcesschunksVolcombineMonitor {
    /// Java `ProcesschunksVolcombineMonitor(BaseManager, AxisID, String, Map)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        computer_map: Option<BTreeMap<String, String>>,
    ) -> Arc<ProcesschunksProcessMonitor<ProcesschunksVolcombineMonitor>> {
        ProcesschunksProcessMonitor::new_subclass(
            manager,
            axis_id,
            root_name,
            computer_map,
            false,
            ProcesschunksVolcombineMonitor {
                start_log: Mutex::new(None),
                finish_log: Mutex::new(None),
                subprocess: Subprocess::new(),
                reader_id_start: Mutex::new(None),
                reader_id_finish: Mutex::new(None),
            },
        )
    }
}

impl ProcesschunksProcessMonitorImpl for ProcesschunksVolcombineMonitor {
    /// Java `updateState`.
    fn update_state(&self, this: &ProcesschunksProcessMonitor) -> Result<bool, UpdateStateError> {
        if this.update_state_super()? {
            return Ok(true);
        }
        if this.is_starting() {
            {
                let mut start_log = self.start_log.lock().unwrap();
                if start_log.is_none() {
                    *start_log = Some(LogFile::get_instance_user_dir(
                        &this.manager.get_property_user_dir().unwrap_or_default(),
                        &dataset_files::VOLCOMBINE_START_LOG,
                        Some(this.manager.get_emergency_monitor(Some(this.axis_id))),
                    )?);
                }
            }
            let start_log = self.start_log.lock().unwrap().clone().unwrap();
            if self
                .reader_id_start
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|reader_id_start| reader_id_start.is_empty())
            {
                match start_log.open_reader() {
                    Ok(reader_id_start) => *self.reader_id_start.lock().unwrap() = reader_id_start,
                    Err(LogFileError::Lock(e)) => {
                        this.handle_lock_exception(Some(&e), true);
                        if !this.is_running() {
                            return Ok(false);
                        }
                    }
                    Err(LogFileError::Unlocked(_)) => return Ok(false),
                    Err(e) => return Err(e.into()),
                }
            }
            let reader_id_start = self.reader_id_start.lock().unwrap().clone();
            if let Some(reader_id_start) = reader_id_start {
                if !reader_id_start.is_empty() {
                    while this.is_running() {
                        let line = match start_log.read_line(&reader_id_start)? {
                            None => break,
                            Some(line) => line,
                        };
                        if VolcombineProcessMonitor::set_subprocess(&line, &self.subprocess) {
                            return Ok(true);
                        }
                    }
                }
            }
        } else if this.is_finishing() {
            {
                let mut finish_log = self.finish_log.lock().unwrap();
                if finish_log.is_none() {
                    *finish_log = Some(LogFile::get_instance_user_dir(
                        &this.manager.get_property_user_dir().unwrap_or_default(),
                        &dataset_files::VOLCOMBINE_FINISH_LOG,
                        Some(this.manager.get_emergency_monitor(Some(this.axis_id))),
                    )?);
                }
            }
            let finish_log = self.finish_log.lock().unwrap().clone().unwrap();
            if self
                .reader_id_finish
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|reader_id_finish| reader_id_finish.is_empty())
            {
                match finish_log.open_reader() {
                    Ok(reader_id_finish) => {
                        *self.reader_id_finish.lock().unwrap() = reader_id_finish
                    }
                    Err(LogFileError::Lock(e)) => {
                        this.handle_lock_exception(Some(&e), true);
                        if !this.is_running() {
                            return Ok(false);
                        }
                    }
                    Err(LogFileError::Unlocked(_)) => return Ok(false),
                    Err(e) => return Err(e.into()),
                }
            }
            let reader_id_finish = self.reader_id_finish.lock().unwrap().clone();
            if let Some(reader_id_finish) = reader_id_finish {
                if !reader_id_finish.is_empty() {
                    while this.is_running() {
                        let line = match finish_log.read_line(&reader_id_finish)? {
                            None => break,
                            Some(line) => line,
                        };
                        if VolcombineProcessMonitor::set_subprocess(&line, &self.subprocess) {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// Java `updateProgressBar`.
    fn update_progress_bar(&self, this: &ProcesschunksProcessMonitor) {
        let axis_id = this.axis_id;
        if self.subprocess.is_filltomo() {
            this.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_string_axis_id(0, Some("Filltomo"), axis_id);
            }));
        } else if self.subprocess.is_reassembling() {
            this.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_string_axis_id(0, Some("Reassembling"), axis_id);
            }));
        } else if self.subprocess.is_densmatch() {
            this.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_string_axis_id(0, Some("Densmatch"), axis_id);
            }));
        } else {
            this.update_progress_bar_super();
        }
    }

    /// Java `closeProcessOutput`.
    fn close_process_output(&self, this: &ProcesschunksProcessMonitor) {
        this.close_process_output_super();
        {
            let mut start_log = self.start_log.lock().unwrap();
            let mut reader_id_start = self.reader_id_start.lock().unwrap();
            if let (Some(log), Some(id)) = (&*start_log, &*reader_id_start) {
                if !id.is_empty() {
                    log.close_id(Some(&**id));
                    *start_log = None;
                    *reader_id_start = None;
                }
            }
        }
        let mut finish_log = self.finish_log.lock().unwrap();
        let mut reader_id_finish = self.reader_id_finish.lock().unwrap();
        if let (Some(log), Some(id)) = (&*finish_log, &*reader_id_finish) {
            if !id.is_empty() {
                log.close_id(Some(&**id));
                *finish_log = None;
                *reader_id_finish = None;
            }
        }
    }
}
