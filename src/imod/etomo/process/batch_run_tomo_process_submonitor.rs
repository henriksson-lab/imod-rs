//! `IMOD/Etomo/src/etomo/process/BatchRunTomoProcessSubmonitor.java`.
//!
//! A `BatchRunTomoProcessMonitor` that watches the batchruntomo log of one dataset
//! started by serieswatcher.  It is `BatchRunTomoProcessMonitor` with
//! `Kind::Submonitor` (see that module); this module holds its constructor and the
//! bodies of its overrides.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::batch_run_tomo_process_monitor::{self, BatchRunTomoProcessMonitor, Kind};
use super::log_feed_monitor::{LogFeedMonitor, MessagesRef};
use super::process_data::ProcessData;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_type::RunType;

/// Java public constructor `BatchRunTomoProcessSubmonitor(BatchRunTomoManager, String,
/// AxisID, ProcessData, ProcessMessages, ProcessingMethod, File)`.
pub fn new(
    manager: &'static BatchRunTomoManager,
    stack_id: Option<&str>,
    axis_id: AxisID,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
    batch_log: PathBuf,
) -> Arc<BatchRunTomoProcessMonitor> {
    let instance = batch_run_tomo_process_monitor::new(
        manager,
        Some(axis_id),
        Some(RunType::Run),
        None,
        process_data,
        messages,
        false,
        processing_method,
        Kind::Submonitor { batch_log },
    );
    if let Some(stack_id) = stack_id {
        instance.erased().set_cur_stack_id(Some(stack_id));
    }
    instance
}

/// Java `buildProcessOutputLogFile()`.
pub fn build_process_output_log_file(
    this: &LogFeedMonitor,
    batch_log: &Path,
) -> Result<Arc<Handle>, LogFileError> {
    LogFile::get_instance_file(
        Some(batch_log),
        Some(this.manager.get_emergency_monitor(this.axis_id)),
    )
}

/// Java `sendMsgStatusChangerStarted()`:
/// `batchRunTomoManager.msgStatusChangerStarted(this, true)`.
pub fn send_msg_status_changer_started(this: &LogFeedMonitor) {
    this.msg_status_changer_started(true);
}
