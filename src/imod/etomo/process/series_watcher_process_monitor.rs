//! `IMOD/Etomo/src/etomo/process/SeriesWatcherProcessMonitor.java`.
//!
//! Monitor for serieswatcher.  A `LogFeedMonitor` subclass (see that module for the
//! representation): `SeriesWatcherProcessMonitor` is
//! `LogFeedMonitor<SeriesWatcherProcessMonitorImpl>`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::log_feed_monitor::{self, LogFeedMonitor, LogFeedMonitorImpl, MessagesRef};
use super::process_data::ProcessData;
use super::process_messages::ProcessMessages;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::{self, BatchRunTomoManager};
use crate::imod::etomo::comscript::series_watcher_param;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::series_watcher_meta_data::SeriesWatcherMetaData;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::table_reference::TableReference;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = batch_run_tomo_manager::AXIS_ID;
/// Java private static final `AXIS_TYPE`.
const AXIS_TYPE: AxisType = AxisType::SingleAxis;
/// Java package-private static final `STARTING_MESSAGE`.
pub const STARTING_MESSAGE: &str = "Running serieswatcher";

/// Java `SeriesWatcherProcessMonitor`.
pub type SeriesWatcherProcessMonitor = LogFeedMonitor<SeriesWatcherProcessMonitorImpl>;

/// Java `SeriesWatcherProcessMonitor`'s own fields.
pub struct SeriesWatcherProcessMonitorImpl {
    /// Java private final `cleanPrint`.
    clean_print: CleanPrint,
    /// Java private final `metaData`.
    meta_data: Arc<SeriesWatcherMetaData>,
    /// Java private final `dualAxis`.
    dual_axis: bool,
    /// Java private `seriesWatcherProjectFile`, initially null.
    series_watcher_project_file: Mutex<Option<PathBuf>>,
    /// Java private `stackIDSet`, initially null.
    stack_id_set: Mutex<Option<HashSet<String>>>,
    /// Java private `diagnostics`, initially false.
    diagnostics: AtomicBool,
    /// Java private `printMsg`, initially null (unused).
    #[allow(dead_code)]
    print_msg: Option<String>,
}

/// Java private static final `STACK_ID_CODE = Extension.EBT.toString()`.
fn stack_id_code() -> String {
    extension::CLASS.ebt.to_string()
}

/// Java public constructor `SeriesWatcherProcessMonitor(BatchRunTomoManager, RunType,
/// boolean, ProcessData, ProcessMessages, boolean, ProcessingMethod, TableReference)`.
#[allow(clippy::too_many_arguments)]
pub fn new(
    manager: &'static BatchRunTomoManager,
    run_type: Option<RunType>,
    dual_axis: bool,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    reconnect: bool,
    processing_method: Option<ProcessingMethod>,
    table_reference: Arc<TableReference>,
) -> Arc<SeriesWatcherProcessMonitor> {
    let instance = LogFeedMonitor::new_subclass(
        manager,
        Some(AXIS_ID),
        run_type,
        None,
        process_data,
        messages,
        reconnect,
        processing_method,
        SeriesWatcherProcessMonitorImpl {
            clean_print: CleanPrint::get_instance_with_label(Some("SeriesWatcherProcessMonitor")),
            meta_data: Arc::new(SeriesWatcherMetaData::new(table_reference)),
            dual_axis,
            series_watcher_project_file: Mutex::new(None),
            stack_id_set: Mutex::new(None),
            diagnostics: AtomicBool::new(false),
            print_msg: None,
        },
    );
    log_feed_monitor::register_parallel_this(&instance);
    instance
}

/// Java package-private static `getInstance(BatchRunTomoManager, boolean, ProcessData,
/// ProcessMessages, ProcessingMethod, TableReference)`.
pub fn get_instance(
    manager: &'static BatchRunTomoManager,
    dual_axis: bool,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
    table_reference: Arc<TableReference>,
) -> Arc<SeriesWatcherProcessMonitor> {
    new(
        manager,
        Some(RunType::SeriesWatcher),
        dual_axis,
        process_data,
        messages,
        false,
        processing_method,
        table_reference,
    )
}

/// Java package-private static final `getReconnectInstance(BatchRunTomoManager, AxisID,
/// ProcessData, ProcessMessages, ProcessingMethod, TableReference)`.
pub fn get_reconnect_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages: Option<MessagesRef>,
    processing_method: Option<ProcessingMethod>,
    table_reference: Arc<TableReference>,
) -> Arc<SeriesWatcherProcessMonitor> {
    new(
        manager,
        Some(RunType::Reconnect),
        axis_id == AxisID::First || axis_id == AxisID::Second,
        process_data,
        messages,
        true,
        processing_method,
        table_reference,
    )
}

/// Java public `getProcessName()`.
pub fn get_process_name() -> ProcessName {
    ProcessName::SERIES_WATCHER
}

/// Java public static `createProcessMessagesInstance(BaseManager)`.
pub fn create_process_messages_instance(manager: &'static dyn BaseManager) -> ProcessMessages {
    ProcessMessages::get_logged_instance(
        Some(manager),
        AxisID::Only,
        true,
        true,
        Some(process_output_strings::SRW_SERIES_WATCHER_ERROR_TAG),
        Some(process_output_strings::SRW_ABORT_TAG),
        false,
        true,
    )
}

impl SeriesWatcherProcessMonitorImpl {
    /// Java private `saveStackID(String)`.
    fn save_stack_id(&self, stack_id: &str) {
        let mut stack_id_set = self.stack_id_set.lock().unwrap();
        let set = stack_id_set.get_or_insert_with(HashSet::new);
        if !set.contains(stack_id) {
            set.insert(stack_id.to_owned());
        }
    }

    /// Java private `readProjectFile(String)`.
    fn read_project_file(
        &self,
        this: &LogFeedMonitor,
        stack_id: Option<&str>,
    ) -> Result<(), LogFileError> {
        let mut err_msg = format!(
            ": Unable to load {}data from ",
            stack_id
                .map(|stack_id| format!("{} ", stack_id))
                .unwrap_or_default()
        );
        let Some(project_file) = self.series_watcher_project_file.lock().unwrap().clone() else {
            eprintln!("Error{}the serieswatcher project file.", err_msg);
            return Ok(());
        };
        err_msg.push_str(&format!(
            "{}.  ",
            project_file
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        // Wait for the project file to be created if necessary.
        for _ in 0..25 {
            if project_file.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if !project_file.exists() {
            eprintln!("Warning{}File does not exist yet.", err_msg);
            return Ok(());
        }
        // File has been modified - reload it.
        let manager: &'static dyn BaseManager = this.manager;
        let parameter_store = match ParameterStore::get_instance_manager(
            Some(manager),
            this.axis_id,
            Some(project_file),
        ) {
            Ok(parameter_store) => parameter_store,
            Err(LogFileError::Lock(e)) => {
                this.handle_lock_exception(Some(&e), true);
                if !this.is_running() {
                    return Ok(());
                }
                None
            }
            Err(e) => return Err(e),
        };
        let Some(mut parameter_store) = parameter_store else {
            eprintln!("Error{}Unable to create loader.", err_msg);
            return Ok(());
        };
        // the serieswatcher project file has been updated.
        parameter_store.load(&*self.meta_data);
        let Some(stack_id) = stack_id else {
            eprintln!("Error{}Missing stack ID.", err_msg);
            return Ok(());
        };
        // `batchRunTomoManager.setParameters(metaData, stackID, false)`, on the event
        // dispatch thread where the dialog lives.
        let manager = this.manager;
        let meta_data = self.meta_data.clone();
        let stack_id = stack_id.to_owned();
        event_queue::invoke_and_wait(move || {
            manager.set_parameters(&meta_data, &stack_id, false);
        });
        Ok(())
    }
}

impl LogFeedMonitorImpl for SeriesWatcherProcessMonitorImpl {
    /// Java final `getTitle()`.
    fn get_title(&self, _this: &LogFeedMonitor) -> String {
        "Series Watcher".to_owned()
    }

    /// Java `getStartingMessage()`.
    fn get_starting_message(&self, _this: &LogFeedMonitor) -> String {
        STARTING_MESSAGE.to_owned()
    }

    /// Java `getCheckFileType()`.
    fn get_check_file_type(&self, _this: &LogFeedMonitor) -> Arc<FileType> {
        series_watcher_param::check_file_value()
    }

    /// Java `buildProcessOutputLogFile()`.
    fn build_process_output_log_file(
        &self,
        this: &LogFeedMonitor,
    ) -> Result<Arc<Handle>, LogFileError> {
        let manager: &'static dyn BaseManager = this.manager;
        LogFile::get_instance_file(
            file_type::CLASS
                .series_watcher_log
                .get_file(Some(manager), this.axis_id)
                .as_deref(),
            Some(this.manager.get_emergency_monitor(this.axis_id)),
        )
    }

    /// Java `setProgressBarTitle(boolean, boolean, boolean, boolean, String)`.
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
                title.push_str("finishing ");
                if will_resume {
                    title.push_str("(will resume) ");
                }
            }
        } else if killing {
            title.push_str("killed ");
        } else if pausing {
            title.push_str("finished ");
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

    /// Java final `processLines(LogFile.Handle, LogFile.ReaderId, boolean)`.
    fn process_lines(
        &self,
        this: &LogFeedMonitor,
        process_output: &Arc<Handle>,
        process_output_reader_id: &ReaderId,
        reconnect: bool,
    ) -> Result<bool, LogFileError> {
        this.update_progress_bar();
        let mut line: Option<String> = None;
        while this.is_running() && !this.is_halt() {
            line = process_output.read_line(process_output_reader_id)?;
            let Some(read) = line.clone() else {
                break;
            };
            this.increment_line_number();
            // Avoid processing output more then once (for reconnect).
            if reconnect
                && this.gt_line_number()
                && let Some(messages) = &this.messages
            {
                messages.lock().unwrap().wake();
            }
            let trimmed = read.trim().to_owned();
            if let Some(message_id_tag) = utilities::get_message_id_tag(
                Some(&trimmed),
                Some(process_output_strings::SRW_CODE),
            ) {
                let stack_id = utilities::get_stack_id(Some(&trimmed), Some(&stack_id_code()));
                let data = utilities::strip_label(utilities::strip_ids(Some(&trimmed)).as_deref());
                if message_id_tag == process_output_strings::SRW_STARTING_TO_PROCESS_SET {
                    // Starting to process stack: Batch1BBa.st (ebt1) [SRW1]
                    // Starting to process data set: Batch1BBa (ebt1) [SRW1]
                    if let Some(stack_id) = &stack_id {
                        self.save_stack_id(stack_id);
                        this.set_cur_stack_id(Some(stack_id));
                        self.read_project_file(this, Some(stack_id))?;
                    } else {
                        eprintln!(
                            "ERROR:  Unable to read series watcher project log.  Stack ID not found in message: {}",
                            trimmed
                        );
                    }
                } else if message_id_tag == process_output_strings::SRW_LOG_LOCATION {
                    // with log in: full_path/swbrt_Batch1BBa.1032348.log (ebt1) [SRW2]
                    if let Some(stack_id) = &stack_id {
                        let manager = this.manager;
                        // Java `new File(data)` with a null data throws; batchruntomo
                        // always prints the path.
                        let data = data.clone().unwrap_or_default();
                        let stack_id = stack_id.clone();
                        event_queue::invoke_and_wait(move || {
                            manager
                                .start_series_watcher_batch_monitor(PathBuf::from(data), &stack_id);
                        });
                    }
                } else if message_id_tag == process_output_strings::SRW_FINISHED_PROCESSING_SET {
                    // Finished processing stack Batch1BBa.st with successful completion
                    // (ebt2) [SRW3]
                    self.read_project_file(this, stack_id.as_deref())?;
                    this.incr_num_done();
                } else if message_id_tag == process_output_strings::SRW_PROCESS_FINISHED {
                    this.end_monitor_state(Some(ProcessEndState::Paused));
                    return Ok(true);
                } else if message_id_tag == process_output_strings::SRW_PROCESS_KILLED {
                    // All running sets killed [SRW5]
                    if this.get_end_state().is_none() {
                        this.send_status_changed_status(Some(
                            BatchRunTomoDatasetState::Killed.into(),
                        ));
                    }
                    this.end_monitor_state(Some(ProcessEndState::Killed));
                    return Ok(true);
                } else if message_id_tag == process_output_strings::SRW_ROOT_NAME {
                    // New project root name: batchNov10-140224 [SRW6]
                    if let Some(data) = &data {
                        let manager: &'static dyn BaseManager = this.manager;
                        let file = file_type::CLASS
                            .series_watcher_project
                            .get_file_with_axis_type(
                                Some(manager),
                                Some(data),
                                Some(AXIS_TYPE),
                                this.axis_id,
                            );
                        if self.diagnostics.load(Ordering::SeqCst) {
                            eprintln!(
                                "class etomo.process.SeriesWatcherProcessMonitor:processLines:data:{},seriesWatcherProjectFile:{}",
                                data,
                                file.as_ref()
                                    .map(|file| file
                                        .file_name()
                                        .map(|name| name.to_string_lossy().into_owned())
                                        .unwrap_or_default())
                                    .unwrap_or_else(|| "null".to_owned())
                            );
                        }
                        *self.series_watcher_project_file.lock().unwrap() = file;
                    } else {
                        eprintln!(
                            "ERROR: unable to extract the series watcher project root name from message:\n{}",
                            trimmed
                        );
                    }
                } else {
                    eprintln!("Warning: unknown message '{}'", trimmed);
                }
            } else if let Some(messages) = &this.messages {
                messages.lock().unwrap().feed_string(&trimmed);
            }
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

    /// Java public `getStatusString()`.
    fn get_status_string(&self, this: &LogFeedMonitor) -> Option<String> {
        let num_done = this.get_num_done();
        if num_done == 0 {
            return Some("watching".to_owned());
        }
        let mut runs = if self.dual_axis { "run" } else { "run" }.to_owned();
        if num_done > 1 {
            runs.push('s');
        }
        Some(format!("{} {} completed", num_done, runs))
    }

    /// Java `getKilledOrPausedStatus()`.
    fn get_killed_or_paused_status(&self, _this: &LogFeedMonitor) -> BatchRunTomoStatus {
        BatchRunTomoStatus::KilledOrPausedSeriesWatcher
    }

    /// Java final `sendStatusChanged(Status)`.
    fn send_status_changed_status(&self, this: &LogFeedMonitor, status: Option<StatusRef>) {
        if status
            != Some(StatusRef::BatchRunTomoDatasetState(
                BatchRunTomoDatasetState::Killed,
            ))
        {
            this.send_status_changed_status_super(status);
        } else if let Some(stack_id_set) = self.stack_id_set.lock().unwrap().clone() {
            for stack_id in stack_id_set {
                if !this.is_running() {
                    break;
                }
                this.send_status_changed_stack_id(Some(&stack_id), status);
            }
        }
    }

    /// Java `isIndeterminateProgressBarMode()`.
    fn is_indeterminate_progress_bar_mode(&self, _this: &LogFeedMonitor) -> bool {
        true
    }

    /// Java public final `getProcessEndState()`.
    fn get_process_end_state(&self, this: &LogFeedMonitor) -> Option<ProcessEndState> {
        let mut end_state = this.get_process_end_state_super();
        if end_state == Some(ProcessEndState::Paused) {
            end_state = Some(ProcessEndState::Done);
        }
        end_state
    }

    /// Java final `getNumDatasets()`.
    fn get_num_datasets(&self, this: &LogFeedMonitor) -> i32 {
        match &*self.stack_id_set.lock().unwrap() {
            None => this.get_num_datasets_super(),
            Some(set) => set.len() as i32,
        }
    }

    /// Java public final `msgLogFileRenamed()`.
    fn msg_log_file_renamed(&self, this: &LogFeedMonitor) {
        this.get_tool_kit().msg_log_file_renamed(true);
    }
}

impl SeriesWatcherProcessMonitorImpl {
    /// The class's `cleanPrint`.
    pub fn clean_print(&self) -> &CleanPrint {
        &self.clean_print
    }
}
