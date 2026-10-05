//! `IMOD/Etomo/src/etomo/process/BatchRunTomoChunkMonitor.java`.
//!
//! A `BatchRunTomoProcessMonitor` that watches the log of one processchunks chunk of a
//! parallel batchruntomo run.  It is `BatchRunTomoProcessMonitor` with `Kind::Chunk`
//! (see that module); this module holds its constructors and the bodies of its
//! overrides.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::batch_run_tomo_process_monitor::{self, BatchRunTomoProcessMonitor, Kind};
use super::log_feed_monitor::{LogFeedMonitor, MessagesRef};
use super::process_data::ProcessData;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::storage::file_reader::{FileReader, FileReaderRef};
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::util::event_queue;

/// Java private constructor `BatchRunTomoChunkMonitor(BatchRunTomoManager, RunList,
/// int, ProcessData, ProcessMessages, boolean, ProcessingMethod)`.
fn new(
    manager: &'static BatchRunTomoManager,
    run_list: Option<Arc<RunList>>,
    chunk_index: i32,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    reconnect: bool,
    processing_method: Option<ProcessingMethod>,
) -> Arc<BatchRunTomoProcessMonitor> {
    batch_run_tomo_process_monitor::new(
        manager,
        None,
        None,
        run_list,
        process_data,
        messages,
        reconnect,
        processing_method,
        Kind::Chunk {
            chunk_index,
            // Override the parent running boolean so that this monitor has time to
            // finish its run function.
            running: AtomicBool::new(false),
        },
    )
}

/// Java package-private static `getInstance(BatchRunTomoManager, RunList, int,
/// ProcessData, ProcessMessages, ProcessingMethod)`.
pub fn get_instance(
    manager: &'static BatchRunTomoManager,
    run_list: Option<Arc<RunList>>,
    chunk_index: i32,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
) -> Arc<BatchRunTomoProcessMonitor> {
    new(
        manager,
        run_list,
        chunk_index,
        process_data,
        messages,
        false,
        processing_method,
    )
}

/// Java package-private static `getReconnectInstance(BatchRunTomoManager, RunList, int,
/// ProcessData, ProcessMessages)`.
pub fn get_reconnect_instance(
    manager: &'static BatchRunTomoManager,
    run_list: Option<Arc<RunList>>,
    chunk_index: i32,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
) -> Arc<BatchRunTomoProcessMonitor> {
    let processing_method = process_data
        .as_ref()
        .and_then(|process_data| process_data.lock().unwrap().get_processing_method());
    new(
        manager,
        run_list,
        chunk_index,
        process_data,
        messages,
        true,
        processing_method,
    )
}

/// Java public `pause()`.  brt chunks need to be completed not paused on a
/// processchunks pause.  But they need to know that they're paused.
pub fn pause(this: &LogFeedMonitor) -> bool {
    this.set_pausing(true);
    true
}

/// Java public `run()`.
pub fn run(this: &LogFeedMonitor, running: &AtomicBool) {
    // try
    running.store(true, Ordering::SeqCst);
    this.run_super();
    // Copy the most recent log entries from the dataset's project log (the secondary
    // log) to the batchruntomo dataset's project log (the primary log).
    let secondary_log = this
        .messages
        .as_ref()
        .and_then(|messages| messages.lock().unwrap().get_secondary_log());
    if let Some(secondary_log) = secondary_log
        && !secondary_log.lock().unwrap().is_empty()
    {
        let mut reader = FileReader::new();
        if reader.set_file(Some(&mut secondary_log.lock().unwrap()))
            && reader.go_to_last(Some(process_output_strings::BRT_DATASET_MSG_ID))
        {
            let reader: FileReaderRef = Arc::new(Mutex::new(reader));
            let manager = this.manager;
            let edt_reader = reader.clone();
            // `manager.logMessagePrimaryLog(reader)`: the log window is a Swing
            // component, so the call is posted to the event dispatch thread.
            event_queue::invoke_and_wait(move || {
                manager.log_message_primary_log(Some(edt_reader));
            });
            reader.lock().unwrap().close();
        }
    }
    // finally
    running.store(false, Ordering::SeqCst);
}

/// Java `getLineNumber()`.
pub fn get_line_number(this: &LogFeedMonitor, chunk_index: i32) -> i32 {
    this.get_process_data().map_or(0, |process_data| {
        process_data
            .lock()
            .unwrap()
            .get_chunk_line_number(chunk_index)
    })
}

/// Java `resetLineNumber()`.
pub fn reset_line_number(this: &LogFeedMonitor, chunk_index: i32) {
    if let Some(process_data) = this.get_process_data() {
        process_data
            .lock()
            .unwrap()
            .reset_chunk_line_number(chunk_index);
    }
}

/// Java `incrementLineNumber()`.
pub fn increment_line_number(this: &LogFeedMonitor, chunk_index: i32) {
    if let Some(process_data) = this.get_process_data() {
        process_data
            .lock()
            .unwrap()
            .increment_chunk_line_number(chunk_index);
    }
}

/// Java `gtLineNumber()`.
pub fn gt_line_number(this: &LogFeedMonitor, chunk_index: i32) -> bool {
    let processed = this.get_processed_line_number();
    this.get_process_data().is_some_and(|process_data| {
        process_data
            .lock()
            .unwrap()
            .gt_chunk_line_number(chunk_index, processed)
    })
}

/// Java final `buildProcessOutputLogFile()`.
pub fn build_process_output_log_file(
    this: &LogFeedMonitor,
    chunk_index: i32,
) -> Result<Arc<Handle>, LogFileError> {
    let manager: &'static dyn BaseManager = this.manager;
    let number = (chunk_index + 1).to_string();
    LogFile::get_instance_file(
        file_type::CLASS
            .processchunks_log
            .get_file_numeric(Some(manager), this.axis_id, Some(&number), None)
            .as_deref(),
        Some(this.manager.get_emergency_monitor(this.axis_id)),
    )
}
