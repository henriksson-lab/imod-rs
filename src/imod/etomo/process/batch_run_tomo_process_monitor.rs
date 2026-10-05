//! `IMOD/Etomo/src/etomo/process/BatchRunTomoProcessMonitor.java`.
//!
//! Monitor for batchruntomo in BATCH mode.
//!
//! The class extends `LogFeedMonitor` (see that module for the representation):
//! `BatchRunTomoProcessMonitor` is `LogFeedMonitor<BatchRunTomoProcessMonitorImpl>`.
//! Its two subclasses, `BatchRunTomoProcessSubmonitor` and `BatchRunTomoChunkMonitor`,
//! are the same type with a different [`Kind`]: their state lives in the kind, and each
//! override this class's methods dispatches on it to the subclass's module.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::batch_run_tomo_chunk_monitor;
use super::batch_run_tomo_process_submonitor;
use super::log_feed_monitor::{self, LogFeedMonitor, LogFeedMonitorImpl, MessagesRef};
use super::process_data::ProcessData;
use super::process_messages::{MessageType, ProcessMessages};
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::batchruntomo_param;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_status::BatchRunTomoDatasetStatus;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::ending_step::EndingStep;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_status::RunStatus;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::step::Step;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::utilities;

/// Java package-private static final `STARTING_MESSAGE`.
pub const STARTING_MESSAGE: &str = "Running batchruntomo";

/// Java `BatchRunTomoProcessMonitor` (and its subclasses; see [`Kind`]).
pub type BatchRunTomoProcessMonitor = LogFeedMonitor<BatchRunTomoProcessMonitorImpl>;

/// Which class the monitor is, with that subclass's own fields.
pub enum Kind {
    /// `BatchRunTomoProcessMonitor` itself.
    Plain,
    /// `BatchRunTomoProcessSubmonitor`: private final `batchLog`.
    Submonitor { batch_log: std::path::PathBuf },
    /// `BatchRunTomoChunkMonitor`: private final `chunkIndex`, and the volatile
    /// `running` that overrides the parent's.
    Chunk {
        chunk_index: i32,
        running: AtomicBool,
    },
}

/// The class's own fields, and the subclass in [`Kind`].
pub struct BatchRunTomoProcessMonitorImpl {
    /// Java private final `cleanPrint`.
    clean_print: CleanPrint,
    /// Java private `stackIDIterator`, initially null.
    stack_id_iterator: Mutex<Option<VecDeque<String>>>,
    /// The subclass.
    pub kind: Kind,
}

impl BatchRunTomoProcessMonitorImpl {
    /// The class's field initialisers, for the given subclass.
    pub fn new(kind: Kind) -> BatchRunTomoProcessMonitorImpl {
        BatchRunTomoProcessMonitorImpl {
            clean_print: CleanPrint::get_instance_with_label(Some(match kind {
                Kind::Submonitor { .. } => "BatchRunTomoProcessSubmonitor",
                _ => "BatchRunTomoProcessMonitor",
            })),
            stack_id_iterator: Mutex::new(None),
            kind,
        }
    }
}

/// Java package-private constructor `BatchRunTomoProcessMonitor(BatchRunTomoManager,
/// AxisID, RunType, RunList, ProcessData, ProcessMessages, boolean, ProcessingMethod)`,
/// for any [`Kind`].
#[allow(clippy::too_many_arguments)]
pub fn new(
    manager: &'static BatchRunTomoManager,
    axis_id: Option<AxisID>,
    run_type: Option<RunType>,
    run_list: Option<Arc<RunList>>,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    reconnect: bool,
    processing_method: Option<ProcessingMethod>,
    kind: Kind,
) -> Arc<BatchRunTomoProcessMonitor> {
    let instance = LogFeedMonitor::new_subclass(
        manager,
        axis_id,
        run_type,
        run_list,
        process_data,
        messages,
        reconnect,
        processing_method,
        BatchRunTomoProcessMonitorImpl::new(kind),
    );
    log_feed_monitor::register_parallel_this(&instance);
    instance
}

/// Java package-private static `getInstance(BatchRunTomoManager, AxisID, ProcessData,
/// ProcessMessages, ProcessingMethod)`.
pub fn get_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
) -> Arc<BatchRunTomoProcessMonitor> {
    new(
        manager,
        Some(axis_id),
        Some(RunType::Run),
        None,
        process_data,
        messages,
        false,
        processing_method,
        Kind::Plain,
    )
}

/// Java package-private static final `getResumeInstance(...)`.
pub fn get_resume_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
) -> Arc<BatchRunTomoProcessMonitor> {
    new(
        manager,
        Some(axis_id),
        Some(RunType::Resume),
        None,
        process_data,
        messages,
        false,
        processing_method,
        Kind::Plain,
    )
}

/// Java package-private static final `getReconnectInstance(...)`.
pub fn get_reconnect_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
) -> Arc<BatchRunTomoProcessMonitor> {
    new(
        manager,
        Some(axis_id),
        Some(RunType::Reconnect),
        None,
        process_data,
        messages,
        true,
        processing_method,
        Kind::Plain,
    )
}

/// Java public static final `createProcessMessagesInstance(BaseManager)`.
pub fn create_process_messages_instance(manager: &'static dyn BaseManager) -> ProcessMessages {
    // `ProcessMessages.getLoggedInstance(manager, AxisID.ONLY, true, true,
    // BRT_BATCH_RUN_TOMO_ERROR_TAG, BRT_ABORT_TAG, false, true)`.
    ProcessMessages::get_logged_instance(
        Some(manager),
        AxisID::Only,
        true,
        true,
        Some(process_output_strings::BRT_BATCH_RUN_TOMO_ERROR_TAG),
        Some(process_output_strings::BRT_ABORT_TAG),
        false,
        true,
    )
}

/// Java public `getProcessName()`.
pub fn get_process_name() -> ProcessName {
    ProcessName::BATCHRUNTOMO
}

/// Java `line.indexOf(tag) != -1`.
fn has(line: &str, tag: &str) -> bool {
    line.contains(tag)
}

impl BatchRunTomoProcessMonitorImpl {
    /// Java package-private final `readParameters(String, LogFile.Handle,
    /// LogFile.ReaderId)`.  Finds information in the parameter list from the start of
    /// the batchruntomo log.
    pub fn read_parameters(
        &self,
        this: &LogFeedMonitor,
        _line: Option<String>,
        process_output: &Arc<Handle>,
        process_output_reader_id: &ReaderId,
    ) -> Result<Option<String>, LogFileError> {
        let mut root_names: Vec<String> = Vec::new();
        let mut current_locations: Vec<String> = Vec::new();
        // Get data from the parameters in the brt log.
        while this.is_running() {
            let Some(line) = this.read_line_with_wait()? else {
                break;
            };
            if has(&line, process_output_strings::END_PARAMETERS_TAG) {
                break;
            }
            let line = line.trim().to_owned();
            if let Some(messages) = &this.messages {
                messages.lock().unwrap().feed_string(&line);
            }
            if has(&line, process_output_strings::END_PARAMETERS_TAG) {
                break;
            } else if has(&line, process_output_strings::BRT_ENDING_STEP_PARAM_TAG) {
                this.set_ending_step_set(true);
            } else if has(&line, process_output_strings::BRT_STARTING_STEP_PARAM_TAG) {
                this.set_starting_step_set(true);
            } else if has(
                &line,
                process_output_strings::BRT_CURRENT_LOCATION_PARAM_TAG,
            ) {
                let tag = process_output_strings::BRT_CURRENT_LOCATION_PARAM_TAG;
                current_locations.push(
                    line[line.find(tag).unwrap() + tag.len()..]
                        .trim()
                        .to_owned(),
                );
            } else if has(&line, process_output_strings::BRT_ROOT_NAME_PARAM_TAG) {
                let tag = process_output_strings::BRT_ROOT_NAME_PARAM_TAG;
                root_names.push(
                    line[line.find(tag).unwrap() + tag.len()..]
                        .trim()
                        .to_owned(),
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(
                log_feed_monitor::READ_SLEEP,
            ));
        }
        // Create an array of the stackIDs that are in this run.
        let mut current_location_iterator = current_locations.into_iter();
        let mut root_name_iterator = root_names.into_iter();
        let mut stack_ids: Vec<String> = Vec::new();
        while this.is_running() {
            let Some(current_location) = current_location_iterator.next() else {
                break;
            };
            // Java's `rootNameIterator.next()` throws when the lists differ in length;
            // batchruntomo always prints the two together.
            let root_name = root_name_iterator.next();
            let manager = this.manager;
            let (stack_id, stack) = {
                let current_location = current_location.clone();
                let root_name = root_name.clone();
                crate::imod::etomo::util::event_queue::invoke_and_wait(move || {
                    let stack_id = manager.find_row(Some(&current_location), root_name.as_deref());
                    let stack = manager.get_stack(stack_id.as_deref());
                    (stack_id, stack)
                })
            };
            if let Some(stack_id) = &stack_id {
                stack_ids.push(stack_id.clone());
            }
            // Save the rootName and currentLocation to be used to write to dataset
            // project logs.
            if let Some(run_list) = &this.run_list {
                run_list.set_root_name(stack_id.as_deref(), root_name.as_deref());
                if let Some(stack) = stack {
                    run_list.set_stack_location(
                        stack_id.as_deref(),
                        stack
                            .parent()
                            .map(|parent| parent.to_string_lossy())
                            .as_deref(),
                    );
                }
            }
        }
        let num_stack_ids = stack_ids.len() as i32;
        *self.stack_id_iterator.lock().unwrap() = Some(stack_ids.into_iter().collect());
        if let Some(run_list) = &this.run_list
            && this.reconnect
            && num_stack_ids < run_list.size()
        {
            // When reconnecting to a resume the currrent BRT log doesn't show all the
            // datasets from the original run, but the runList contains the original list.
            // StackIDs in this case refers to the rows yet to be run.
            //
            // Go through the rows already processed.
            for i in 0..run_list.size() - num_stack_ids {
                if run_list.equals_run_status(i, Some(RunStatus::Ran)) {
                    this.incr_num_done();
                }
            }
        }
        // Get the next line
        let line = process_output.read_line(process_output_reader_id)?;
        Ok(line.map(|line| line.trim().to_owned()))
    }

    /// Java `processLines` body, returning `Ok(Some(value))` for a `return value` and
    /// `Ok(None)` when the loop ends.
    fn process_lines_loop(
        &self,
        this: &LogFeedMonitor,
        process_output: &Arc<Handle>,
        process_output_reader_id: &ReaderId,
        reconnect: bool,
        last_line: &mut Option<String>,
    ) -> Result<Option<bool>, LogFileError> {
        let messages = this.messages.clone();
        let feed_string = |line: &str| {
            if let Some(messages) = &messages {
                messages.lock().unwrap().feed_string(line);
            }
        };
        let feed_message = |line: &str| {
            if let Some(messages) = &messages {
                messages
                    .lock()
                    .unwrap()
                    .feed_message(Some(MessageType::Log), Some(line));
            }
        };
        while this.is_running() && !this.is_halt() {
            *last_line = process_output.read_line(process_output_reader_id)?;
            let Some(read) = last_line.clone() else {
                break;
            };
            this.increment_line_number();
            // Avoid processing output more then once (for reconnect).
            if reconnect
                && this.gt_line_number()
                && let Some(messages) = &messages
            {
                messages.lock().unwrap().wake();
            }
            let mut line = read.trim().to_owned();
            let mut temp_cur_dataset_location: Option<String>;
            let mut temp_current_dataset: Option<String>;
            // Handle parameter list at the top of the log
            if has(&line, process_output_strings::BRT_START_PARAMETERS_TAG) {
                let next = self.read_parameters(
                    this,
                    Some(line.clone()),
                    process_output,
                    process_output_reader_id,
                )?;
                *last_line = next.clone();
                match next {
                    Some(next) => line = next,
                    // Java dereferences the null line in the next `indexOf` and the
                    // monitor thread dies; end this pass instead.
                    None => return Ok(None),
                }
            } else if let Some(end_index) =
                line.find(process_output_strings::BRT_DATASET_LOCATION_TAG)
            {
                // In:
                // /home/build/Head-Build/build/IMOD/ImodTests/scriptTests/batchruntomo/Head-Build/BB
                // [brt13]
                // String off everything but the directory path.
                let start_index = match line
                    .find(process_output_strings::BRT_DATASET_LOCATION_START_TAG)
                {
                    Some(start_index) => {
                        start_index + process_output_strings::BRT_DATASET_LOCATION_START_TAG.len()
                    }
                    None => 0,
                };
                // Java `substring(startIndex, endIndex)` throws when start > end.
                temp_cur_dataset_location = line
                    .get(start_index..end_index)
                    .map(|location| location.trim().to_owned());
                if let Some(location) = &temp_cur_dataset_location {
                    this.set_dataset(None, Some(location));
                }
            }
            if has(&line, process_output_strings::BRT_CREATED_DATASET_DIRECTORY) {
                // Created dataset directory /home/sueh/NOBACKUP/test
                // datasets/Linux/Development/UITests/batch2/BBa
                temp_cur_dataset_location = line
                    .get(process_output_strings::BRT_CREATED_DATASET_DIRECTORY.len()..)
                    .map(|location| location.trim().to_owned());
                if let Some(location) = &temp_cur_dataset_location {
                    temp_current_dataset = Some(
                        Path::new(location)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    );
                    // This is a new location, so make sure it is used.
                    this.reset_dataset();
                    this.set_dataset(temp_current_dataset.as_deref(), Some(location));
                }
            }
            // Send all output to the ProcessMessages blocking queue, so each line can be
            // processed immediately.
            // Handle messages that need to be logged, and send other lines to
            // ProcessMessages.
            let mut recognized = false;
            if has(&line, process_output_strings::BRT_DATASET_MSG_ID)
                || has(&line, process_output_strings::BRT_DATASET_TAG)
            {
                if let Some(index) = line.find(process_output_strings::BRT_CURRENT_DATASET_TAG) {
                    // Location of the dataset is after the first "set ".
                    let mut dataset = line
                        [index + process_output_strings::BRT_CURRENT_DATASET_TAG.len()..]
                        .trim()
                        .to_owned();
                    // The dataset name does not contain whitespace
                    let words: Vec<&str> = dataset.split_whitespace().collect();
                    if !dataset.is_empty() && words.len() > 1 {
                        dataset = words[0].to_owned();
                    }
                    this.set_dataset(Some(&dataset), None);
                    // New dataset
                    this.set_dataset_failed(false);
                    // Send a linefeed and the dataset start message to the project log.
                    // Use the ProcessMessages string feed so that messages get to the
                    // project log in the right order.
                    if let Some(messages) = &messages {
                        let mut messages = messages.lock().unwrap();
                        messages.feed_newline(Some(MessageType::Log));
                        messages.feed_message(Some(MessageType::Log), Some(&line));
                    }
                    return Ok(Some(true));
                }
            }
            // Handle tagged lines.
            feed_string(&line);
            // Find deprecate logged messages. Now handled by the addition of a log
            // message tag.
            if !MessageType::Log.is_type(Some(&line)) {
                for tag in process_output_strings::BRT_LOG_TAGS {
                    if has(&line, tag) {
                        recognized = true;
                        // Send output that users want to see in the project log.
                        feed_message(&line);
                        break;
                    }
                }
                // Backwards compatibility
                if has(&line, process_output_strings::BRT_AXIS_B_TAG) {
                    recognized = true;
                    feed_message(&line);
                }
                if recognized {
                    continue;
                }
            }
            if has(&line, process_output_strings::BRT_STARTING_DATASET_MSG_ID)
                || has(&line, process_output_strings::BRT_STARTING_DATASET_TAG)
            {
                let next = self
                    .stack_id_iterator
                    .lock()
                    .unwrap()
                    .as_mut()
                    .and_then(VecDeque::pop_front);
                if let Some(new_cur_stack_id) = next {
                    this.set_cur_stack_id(Some(&new_cur_stack_id));
                    this.set_dataset_stack_id(Some(&new_cur_stack_id));
                } else {
                    eprintln!(
                        "WARNING: Unknown row.  Table will not be  updated.   Previous stack ID was{}\nline:{}",
                        this.get_cur_stack_id().unwrap_or_else(|| "null".to_owned()),
                        line
                    );
                    this.reset_cur_stack_id();
                }
                this.set_dataset_running(true);
                this.set_dataset_failed(false);
                this.set_dataset_delivered(false);
                this.set_dataset_renamed(false);
                if this.is_starting_step_set() {
                    this.send_status_changed_status(Some(
                        BatchRunTomoDatasetState::Starting.into(),
                    ));
                } else {
                    this.send_status_changed_status(Some(BatchRunTomoDatasetState::Running.into()));
                }
                this.set_current_step(None);
                return Ok(Some(true));
            }
            // check for the real batchruntomo error message. Everything else will be
            // logged.
            if has(&line, process_output_strings::BRT_BATCH_RUN_TOMO_ERROR_TAG) {
                // Ignore error message if killed message has already been sent.
                if this.get_end_state() == Some(ProcessEndState::Killed) {
                    return Ok(Some(false));
                }
                this.set_current_step(Some("failed"));
                this.set_dataset_running(false);
                if !this.is_dataset_succeeded() {
                    eprintln!(
                        "batchRunTomoProcessMonitor sendStatusChanged BatchRunTomoDatasetState\nBRT_BATCH_RUN_TOMO_ERROR_TAG {}",
                        line
                    );
                    this.send_status_changed_status(Some(BatchRunTomoDatasetState::Failed.into()));
                    this.end_monitor_state(Some(ProcessEndState::Failed));
                } else {
                    this.end_monitor_state(Some(ProcessEndState::Done));
                }
                this.reset_dataset();
                return Ok(Some(true));
            }
            if line == process_output_strings::SUCCESS_TAG {
                this.end_monitor_state(Some(ProcessEndState::Done));
                return Ok(Some(true));
            }
            if let Some(start_index) = line.find(process_output_strings::BRT_START_AXIS_MSG_ID) {
                // An axis is being started.
                let mut message = line[..start_index].to_owned();
                if let Some(start_index) = line.find(process_output_strings::LOG_TAG) {
                    // Java `message.substring(0, startIndex)` with the index from the
                    // whole line.
                    message = message.get(..start_index).unwrap_or(&message).to_owned();
                }
                let message = message.trim();
                if !utilities::is_empty(Some(message)) {
                    let last =
                        &message[message.len() - message.chars().last().unwrap().len_utf8()..];
                    let temp_axis_id = AxisID::get_instance_ignore_case(Some(last));
                    if temp_axis_id == Some(AxisID::First) || temp_axis_id == Some(AxisID::Second) {
                        this.set_cur_axis_id(temp_axis_id);
                    }
                }
            }
            if has(&line, process_output_strings::BRT_KILLING_MSG_ID)
                || line == process_output_strings::BRT_KILLING_TAG
            {
                // A kill is asynchronous, so it could happen just after a dataset is
                // completed. In that case no dataset was killed.
                if this.get_end_state().is_none() && this.is_dataset_running() {
                    this.send_status_changed_status(Some(BatchRunTomoDatasetState::Killed.into()));
                }
                this.set_process_end_state(Some(ProcessEndState::Killed));
                return Ok(Some(true));
            }
            if has(&line, process_output_strings::BRT_PAUSED_MSG_ID)
                || line == process_output_strings::BRT_PAUSED_TAG
            {
                this.end_monitor_state(Some(ProcessEndState::Paused));
                return Ok(Some(true));
            }
            if has(&line, process_output_strings::BRT_ETOMO_TAG)
                || has(&line, process_output_strings::BRT_ETOMO_TAG_OLD)
            {
                this.set_current_step(Some("Etomo setup"));
                return Ok(Some(true));
            }
            if has(&line, process_output_strings::BRT_STEP_SUCCESS_MSG_ID)
                || has(&line, process_output_strings::BRT_STEP_SUCCESS_TAG)
            {
                if let Some(temp_current_step) = this.get_current_step() {
                    let process_name = ProcessName::get_instance(Some(&temp_current_step));
                    if process_name == Some(ProcessName::VOLCOMBINE)
                        || process_name == Some(ProcessName::TRIMVOL)
                    {
                        this.send_status_changed_status(process_name.map(StatusRef::ProcessName));
                    }
                }
                this.append_to_current_step(" - done");
                return Ok(Some(true));
            }
            // first one was BRT_DATASET_SUCCESS_MSG_ID
            if has(&line, process_output_strings::BRT_DATASET_LOG_CLOSED_TAG) {
                // Dataset succeeded
                this.set_current_step(Some("done"));
                this.set_dataset_running(false);
                let state = if this.is_dataset_failed() {
                    BatchRunTomoDatasetState::Failed
                } else {
                    this.set_dataset_succeeded(true);
                    this.incr_num_done();
                    if this.is_ending_step_set() {
                        BatchRunTomoDatasetState::Stopped
                    } else if this.get_cur_axis_id() == Some(AxisID::First) {
                        BatchRunTomoDatasetState::ADone
                    } else {
                        BatchRunTomoDatasetState::Done
                    }
                };
                this.send_status_changed_status(Some(state.into()));
                this.reset_dataset();
                return Ok(Some(true));
            }
            if has(&line, process_output_strings::BRT_ABORT_SET_TAG) {
                // Dataset failed
                this.set_current_step(Some("failed"));
                this.set_dataset_running(false);
                this.set_dataset_failed(true);
                this.send_status_changed_status(Some(BatchRunTomoDatasetState::Failed.into()));
                this.reset_dataset();
                return Ok(Some(true));
            }
            if has(&line, process_output_strings::BRT_ABORT_AXIS_TAG) {
                this.set_dataset_failed(true);
                this.set_current_step(Some("axis failed"));
                this.send_status_changed_status(Some(BatchRunTomoDatasetState::Failing.into()));
                return Ok(Some(true));
            }
            if (has(&line, process_output_strings::BRT_STEP_MSG_ID)
                || has(&line, process_output_strings::BRT_STEP_TAG))
                && let Some(index) = line.rfind(process_output_strings::BRT_STEP_END_TAG)
            {
                // The current step is the name of the com file (which may not contain
                // whitespace).
                let substring = &line[..index];
                // Java `split("\\s+")`: a non-empty all-whitespace prefix gives an
                // empty array (fall through); an empty prefix gives [""]; otherwise
                // the last field is the last word.
                let all_whitespace =
                    !substring.is_empty() && substring.chars().all(char::is_whitespace);
                if !all_whitespace {
                    let last = substring.split_whitespace().last().unwrap_or("");
                    this.set_current_step(Some(last));
                    return Ok(Some(true));
                }
            }
            if has(&line, process_output_strings::BRT_REACHED_STEP_TAG) {
                let step_value = line[line.rfind(' ').unwrap()..].trim().to_owned();
                if let Some(ending_step) =
                    EndingStep::get_instance_from_step_value(Some(&step_value))
                {
                    this.send_status_changed_status(Some(StatusRef::EndingStep(ending_step)));
                    continue;
                }
                if let Some(step) = Step::get_instance(Some(&step_value)) {
                    this.send_status_changed_status(Some(StatusRef::Step(step)));
                    continue;
                }
            }
            let mut file = None;
            if !this.is_dataset_delivered() {
                file = log_feed_monitor::get_file_from_output(
                    &line,
                    process_output_strings::BRT_DELIVERED_MSG_ID,
                    Some(process_output_strings::BRT_DELIVERED_TAG),
                );
            }
            if let Some(file) = file {
                this.set_dataset_delivered(true);
                this.send_status_changed_file_string(
                    Some(BatchRunTomoDatasetStatus::Delivered.into()),
                    &file,
                );
                continue;
            }
            let mut file = None;
            if !this.is_dataset_renamed() {
                file = log_feed_monitor::get_file_from_output(
                    &line,
                    process_output_strings::BRT_RENAMED_MSG_ID,
                    Some(process_output_strings::BRT_RENAMED_TAG),
                );
            }
            if let Some(file) = file {
                this.set_dataset_renamed(true);
                this.send_status_changed_file_string(
                    Some(BatchRunTomoDatasetStatus::Renamed.into()),
                    &file,
                );
                continue;
            }
        }
        Ok(None)
    }
}

impl LogFeedMonitorImpl for BatchRunTomoProcessMonitorImpl {
    /// Java final `getTitle()`.
    fn get_title(&self, _this: &LogFeedMonitor) -> String {
        "Batchruntomo".to_owned()
    }

    /// Java final `getStartingMessage()`.
    fn get_starting_message(&self, _this: &LogFeedMonitor) -> String {
        STARTING_MESSAGE.to_owned()
    }

    /// Java final `getCheckFileType()`.
    fn get_check_file_type(&self, _this: &LogFeedMonitor) -> Arc<FileType> {
        batchruntomo_param::check_file_value()
    }

    /// Java `buildProcessOutputLogFile()`, and the two subclasses' overrides.
    fn build_process_output_log_file(
        &self,
        this: &LogFeedMonitor,
    ) -> Result<Arc<Handle>, LogFileError> {
        match &self.kind {
            Kind::Plain => {
                let manager: &'static dyn BaseManager = this.manager;
                LogFile::get_instance_file(
                    file_type::CLASS
                        .batch_run_tomo_log
                        .get_file(Some(manager), this.axis_id)
                        .as_deref(),
                    Some(this.manager.get_emergency_monitor(this.axis_id)),
                )
            }
            Kind::Submonitor { batch_log } => {
                batch_run_tomo_process_submonitor::build_process_output_log_file(this, batch_log)
            }
            Kind::Chunk { chunk_index, .. } => {
                batch_run_tomo_chunk_monitor::build_process_output_log_file(this, *chunk_index)
            }
        }
    }

    /// Java final `setProgressBarTitle(boolean, boolean, boolean, boolean, String)`.
    /// Returns "values have changed".
    fn set_progress_bar_title(
        &self,
        this: &LogFeedMonitor,
        process_running: bool,
        killing: bool,
        pausing: bool,
        will_resume: bool,
        current_dataset: Option<&str>,
    ) -> bool {
        let mut title = String::new();
        if process_running {
            if killing {
                title.push_str("killing ");
            } else if pausing {
                title.push_str("pausing ");
                if will_resume {
                    title.push_str("(will resume) ");
                }
            }
        } else if killing {
            title.push_str("killed ");
        } else if pausing {
            title.push_str("paused ");
            if will_resume {
                title.push_str("(will resume) ");
            }
        }
        title.push_str(&self.get_title(this));
        if let Some(current_dataset) = current_dataset {
            title.push_str(&format!(": {}", current_dataset));
        }
        if let Some(temp_current_step) = this.get_current_step() {
            title.push_str(&format!(": {}", temp_current_step));
        }
        this.set_progress_bar(&title)
    }

    /// Java final synchronized `processLines(LogFile.Handle, LogFile.ReaderId,
    /// boolean)`.
    fn process_lines(
        &self,
        this: &LogFeedMonitor,
        process_output: &Arc<Handle>,
        process_output_reader_id: &ReaderId,
        reconnect: bool,
    ) -> Result<bool, LogFileError> {
        let mut line = None;
        if let Some(value) = self.process_lines_loop(
            this,
            process_output,
            process_output_reader_id,
            reconnect,
            &mut line,
        )? {
            return Ok(value);
        }
        if this.is_process_running() && line.is_none() {
            if this.is_interrupted() {
                this.end_monitor();
                return Ok(true);
            } else {
                // Waiting for something to complete - must have caught up with the
                // process.
                this.set_live(true);
            }
        }
        Ok(false)
    }

    /// Java final `getStatusString()`.
    fn get_status_string(&self, this: &LogFeedMonitor) -> Option<String> {
        let num_datasets = this.get_num_datasets();
        let runs = if num_datasets == 1 { "run" } else { "runs" };
        Some(format!(
            "{} of {} {} completed",
            this.get_num_done(),
            num_datasets,
            runs
        ))
    }

    /// `BatchRunTomoChunkMonitor.run()` overrides it.
    fn run(&self, this: &LogFeedMonitor) {
        match &self.kind {
            Kind::Chunk { running, .. } => batch_run_tomo_chunk_monitor::run(this, running),
            _ => this.run_super(),
        }
    }

    /// `BatchRunTomoProcessSubmonitor.sendMsgStatusChangerStarted()` overrides it.
    fn send_msg_status_changer_started(&self, this: &LogFeedMonitor) {
        match &self.kind {
            Kind::Submonitor { .. } => {
                batch_run_tomo_process_submonitor::send_msg_status_changer_started(this)
            }
            _ => this.send_msg_status_changer_started_super(),
        }
    }

    /// `BatchRunTomoChunkMonitor.isProcessChunks()` overrides it.
    fn is_process_chunks(&self, _this: &LogFeedMonitor) -> bool {
        matches!(self.kind, Kind::Chunk { .. })
    }

    /// `BatchRunTomoChunkMonitor.getLineNumber()` overrides it.
    fn get_line_number(&self, this: &LogFeedMonitor) -> i32 {
        match &self.kind {
            Kind::Chunk { chunk_index, .. } => {
                batch_run_tomo_chunk_monitor::get_line_number(this, *chunk_index)
            }
            _ => this.get_line_number_super(),
        }
    }

    /// `BatchRunTomoChunkMonitor.resetLineNumber()` overrides it.
    fn reset_line_number(&self, this: &LogFeedMonitor) {
        match &self.kind {
            Kind::Chunk { chunk_index, .. } => {
                batch_run_tomo_chunk_monitor::reset_line_number(this, *chunk_index)
            }
            _ => this.reset_line_number_super(),
        }
    }

    /// `BatchRunTomoChunkMonitor.incrementLineNumber()` overrides it.
    fn increment_line_number(&self, this: &LogFeedMonitor) {
        match &self.kind {
            Kind::Chunk { chunk_index, .. } => {
                batch_run_tomo_chunk_monitor::increment_line_number(this, *chunk_index)
            }
            _ => this.increment_line_number_super(),
        }
    }

    /// `BatchRunTomoChunkMonitor.gtLineNumber()` overrides it.
    fn gt_line_number(&self, this: &LogFeedMonitor) -> bool {
        match &self.kind {
            Kind::Chunk { chunk_index, .. } => {
                batch_run_tomo_chunk_monitor::gt_line_number(this, *chunk_index)
            }
            _ => this.gt_line_number_super(),
        }
    }

    /// `BatchRunTomoChunkMonitor.isRunning()` overrides it.
    fn is_running(&self, this: &LogFeedMonitor) -> bool {
        match &self.kind {
            Kind::Chunk { running, .. } => {
                running.load(Ordering::SeqCst) || this.is_running_super()
            }
            _ => this.is_running_super(),
        }
    }

    /// Both subclasses override `hasProgressBarAccess()` to return false.
    fn has_progress_bar_access(&self, _this: &LogFeedMonitor) -> bool {
        matches!(self.kind, Kind::Plain)
    }
}

impl BatchRunTomoProcessMonitorImpl {
    /// The class's `cleanPrint`.
    pub fn clean_print(&self) -> &CleanPrint {
        &self.clean_print
    }
}

/// The status a monitor reports when it is stopped by a pause or kill (re-exported
/// for the process manager).
pub fn killed_or_paused(monitor: &BatchRunTomoProcessMonitor) -> BatchRunTomoStatus {
    monitor.erased().get_killed_or_paused_status()
}
