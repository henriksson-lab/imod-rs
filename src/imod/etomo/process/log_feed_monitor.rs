//! `IMOD/Etomo/src/etomo/process/LogFeedMonitor.java`.
//!
//! For monitors that need to immediately read the log file as it's being updated: the
//! batchruntomo monitors (`BatchRunTomoProcessMonitor`, its submonitor and chunk
//! monitor) and the serieswatcher monitor.
//!
//! # How the abstract class and its subclasses are represented
//!
//! As in `processchunks_process_monitor.rs`: the class is one struct generic over the
//! subclass state, `LogFeedMonitor<S>`, whose last field is the subclass.
//! `LogFeedMonitor` without a parameter is the type with the subclass erased; every
//! Java method is an inherent method of that type.  [`LogFeedMonitorImpl`] is the set
//! of abstract and overridable methods: each takes the erased monitor as `this`;
//! an overridable one defaults to the Java body, the inherent `<name>_super` method.
//!
//! **Threads.**  A monitor runs on its own thread.  Its listeners (the batchruntomo
//! dialog, table, rows and step panel) are event dispatch thread objects: they are kept
//! as `EdtRef`s and called there through `StatusChangeEventSender`, as the Java posts
//! them with `SwingUtilities.invokeLater`.  The calls the Java makes from the monitor
//! thread into the batchruntomo manager's dialog (`msgStatusChangerStarted`,
//! `findRow`, `getStack`, `setParameters`, `startSeriesWatcherBatchMonitor`) are made
//! on the event dispatch thread with `invoke_and_wait`.  The fields the Java reads and
//! writes from both threads without a lock are atomics or behind short-lived locks;
//! the `synchronized` methods share one re-entrant lock.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use super::base_process_manager::BaseProcessManager;
use super::emergency_monitor::EmergencyMonitor;
use super::monitor::{DetachedProcessMonitor, Monitor, OutfileProcessMonitor, ProcessMonitor};
use super::monitor_tool_kit::{self, MonitorToolKit};
use super::parallel_process_monitor::ParallelProcessMonitor;
use super::process_data::ProcessData;
use super::process_interface::SystemProcessInterface;
use super::process_messages::ProcessMessages;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::log_file::{
    Handle, LockException, LogFile, LogFileError, ReaderId, WriterId,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::r#type::status_change_event_sender::{
    StatusChangeEventSender, StatusChangeListeners,
};
use crate::imod::etomo::r#type::status_change_listener::StatusChangeListener;
use crate::imod::etomo::r#type::status_change_row_event::StatusChangeRowEvent;
use crate::imod::etomo::r#type::status_changer::StatusChanger;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::event_queue::{self, EdtRef, ReentrantLock};

/// Java private static final `START_SLEEP`.
const START_SLEEP: u64 = 2000;
/// Java private static final `INIT_SLEEP`.
const INIT_SLEEP: u64 = 1000;
/// Java private static final `UPDATE_SLEEP`.
const UPDATE_SLEEP: u64 = 100;
/// Java private static final `NUM_TRIES`.
const NUM_TRIES: i32 = 1000;
/// Java private static final `TRY_SLEEP`.
const TRY_SLEEP: u64 = 5;
/// Java package-private static final `READ_SLEEP`.
pub const READ_SLEEP: u64 = 5;

/// A shared `ProcessMessages` object, as a Java reference to one held in a
/// manager's message list (`process_messages::MessagesArray`).
pub type MessagesRef = Arc<Mutex<ProcessMessages>>;

/// Is `e` Java's `FileNotFoundException`?
fn is_file_not_found(e: &LogFileError) -> bool {
    matches!(e, LogFileError::Io(io) if io.kind() == std::io::ErrorKind::NotFound)
}

/// The abstract and overridable methods of Java `LogFeedMonitor`; see the module
/// comment.  Each default is the Java body.
pub trait LogFeedMonitorImpl: Send + Sync + 'static {
    /// Java abstract `getTitle()`.
    fn get_title(&self, this: &LogFeedMonitor) -> String;

    /// Java abstract `getStartingMessage()`.
    fn get_starting_message(&self, this: &LogFeedMonitor) -> String;

    /// Java abstract `getCheckFileType()`.
    fn get_check_file_type(&self, this: &LogFeedMonitor) -> Arc<FileType>;

    /// Java abstract `buildProcessOutputLogFile()`.
    fn build_process_output_log_file(
        &self,
        this: &LogFeedMonitor,
    ) -> Result<Arc<Handle>, LogFileError>;

    /// Java abstract `setProgressBarTitle(boolean, boolean, boolean, boolean, String)`.
    fn set_progress_bar_title(
        &self,
        this: &LogFeedMonitor,
        process_running: bool,
        killing: bool,
        pausing: bool,
        will_resume: bool,
        current_dataset: Option<&str>,
    ) -> bool;

    /// Java abstract `processLines(LogFile.Handle, LogFile.ReaderId, boolean)`.
    fn process_lines(
        &self,
        this: &LogFeedMonitor,
        process_output: &Arc<Handle>,
        process_output_reader_id: &ReaderId,
        reconnect: bool,
    ) -> Result<bool, LogFileError>;

    /// Java `ProcessMonitor.getStatusString()`, which the class leaves abstract.
    fn get_status_string(&self, this: &LogFeedMonitor) -> Option<String>;

    /// Java `run()`.
    fn run(&self, this: &LogFeedMonitor) {
        this.run_super();
    }

    /// Java `getNumDatasets()`.
    fn get_num_datasets(&self, this: &LogFeedMonitor) -> i32 {
        this.num_datasets
    }

    /// Java `sendMsgStatusChangerStarted()`.
    fn send_msg_status_changer_started(&self, this: &LogFeedMonitor) {
        this.send_msg_status_changer_started_super();
    }

    /// Java `getKilledOrPausedStatus()`.
    fn get_killed_or_paused_status(&self, this: &LogFeedMonitor) -> BatchRunTomoStatus {
        this.get_killed_or_paused_status_super()
    }

    /// Java `isProcessChunks()`.
    fn is_process_chunks(&self, this: &LogFeedMonitor) -> bool {
        let _ = this;
        false
    }

    /// Java `sendStatusChanged(Status)`.
    fn send_status_changed_status(&self, this: &LogFeedMonitor, status: Option<StatusRef>) {
        this.send_status_changed_status_super(status);
    }

    /// Java `isPausing()`.
    fn is_pausing(&self, this: &LogFeedMonitor) -> bool {
        this.is_pausing_super()
    }

    /// Java `getProcessEndState()`.
    fn get_process_end_state(&self, this: &LogFeedMonitor) -> Option<ProcessEndState> {
        this.get_process_end_state_super()
    }

    /// Java `incrementLineNumber()`.
    fn increment_line_number(&self, this: &LogFeedMonitor) {
        this.increment_line_number_super();
    }

    /// Java `getLineNumber()`.
    fn get_line_number(&self, this: &LogFeedMonitor) -> i32 {
        this.get_line_number_super()
    }

    /// Java `resetLineNumber()`.
    fn reset_line_number(&self, this: &LogFeedMonitor) {
        this.reset_line_number_super();
    }

    /// Java `gtLineNumber()`.
    fn gt_line_number(&self, this: &LogFeedMonitor) -> bool {
        this.gt_line_number_super()
    }

    /// Java `isIndeterminateProgressBarMode()`.
    fn is_indeterminate_progress_bar_mode(&self, this: &LogFeedMonitor) -> bool {
        let _ = this;
        false
    }

    /// Java `isRunning()`.
    fn is_running(&self, this: &LogFeedMonitor) -> bool {
        this.is_running_super()
    }

    /// Java `hasProgressBarAccess()`.
    fn has_progress_bar_access(&self, this: &LogFeedMonitor) -> bool {
        let _ = this;
        true
    }

    /// Java `msgLogFileRenamed()`.
    fn msg_log_file_renamed(&self, this: &LogFeedMonitor) {
        this.tool_kit.msg_log_file_renamed(false);
    }

    /// The monitor's `halt()`; `ProcesschunksBatchRunTomoMonitor`'s chunk monitors are
    /// halted through it.  Default: the class's synchronized `halt`.
    fn halt(&self, this: &LogFeedMonitor) {
        this.halt_super();
    }
}

/// Java `abstract class LogFeedMonitor implements OutfileProcessMonitor,
/// StatusChanger, ParallelProcessMonitor`; see the module comment.
pub struct LogFeedMonitor<S: ?Sized + LogFeedMonitorImpl = dyn LogFeedMonitorImpl> {
    /// Java private final `cleanPrint`.
    clean_print: CleanPrint,

    /// Java package-private final `manager` (also `batchRunTomoManager`).
    pub manager: &'static BatchRunTomoManager,
    /// Java package-private final `axisID` (null for a chunk monitor).
    pub axis_id: Option<AxisID>,
    /// Java private final `emergencyMonitor`.
    emergency_monitor: Arc<EmergencyMonitor>,
    /// Java package-private final `messages`.
    pub messages: Option<MessagesRef>,
    /// Java package-private final `reconnect`.
    pub reconnect: bool,
    /// Java private final `processData`.
    process_data: Option<Arc<Mutex<ProcessData>>>,
    /// Java private final `busyStatusMediator`.
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java private final `mediator`: an event dispatch thread object.
    mediator: Option<Arc<EdtRef<ProcessingMethodMediator>>>,
    /// Java private final `numDatasets`.
    num_datasets: i32,
    /// Java package-private final `runList`.
    pub run_list: Option<Arc<RunList>>,
    /// Java private final `processingMethod`.
    processing_method: Option<ProcessingMethod>,
    /// Java private final `toolKit`.
    tool_kit: MonitorToolKit,

    /// Java private `updateProgressBar` (turn on to change the progress bar title).
    update_progress_bar: AtomicBool,
    /// Java private `endState`, initially null.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java private `commandsPipe`, initially null.
    commands_pipe: Mutex<Option<Arc<Handle>>>,
    /// Java private `commandsPipeWriterId`, initially null.
    commands_pipe_writer_id: Mutex<Option<WriterId>>,
    /// Java private `useCommandsPipe`, initially true.
    use_commands_pipe: AtomicBool,
    /// Java private `processOutput`, initially null.
    process_output: Mutex<Option<Arc<Handle>>>,
    /// Java private `processOutputReaderId`, initially null.
    process_output_reader_id: Mutex<Option<ReaderId>>,
    /// Java private `processRunning`, initially true.
    process_running: AtomicBool,
    /// Java private `pausing`, initially false.
    pausing: AtomicBool,
    /// Java private `killing`, initially false.
    killing: AtomicBool,
    /// Java private `stop`, initially false.
    stop: AtomicBool,
    /// Java private `process`, initially null.
    process: Mutex<Option<Arc<dyn SystemProcessInterface>>>,
    /// Java private `listeners`, initially null.
    listeners: StatusChangeListeners,

    /// Java private `currentStep`, initially null.
    current_step: Mutex<Option<String>>,
    /// Java private `currentDataset`, initially null.
    current_dataset: Mutex<Option<String>>,
    /// Java private `willResume`, initially false.
    will_resume: AtomicBool,
    /// Java private `interrupted`, initially false (set from a caught
    /// `InterruptedException`).
    interrupted: AtomicBool,
    /// Java private `datasetRunning`, initially false.
    dataset_running: AtomicBool,
    /// Java private `endingStepSet`, initially false.
    ending_step_set: AtomicBool,
    /// Java private `startingStepSet`, initially false.
    starting_step_set: AtomicBool,
    /// Java private `datasetFailed`, initially false.
    dataset_failed: AtomicBool,
    /// Java private `datasetDelivered`, initially false.
    dataset_delivered: AtomicBool,
    /// Java private `datasetRenamed`, initially false.
    dataset_renamed: AtomicBool,
    /// Java private `live`, initially true (the program is running).
    live: AtomicBool,
    /// Java private `processedLineNumber`, initially 0.
    processed_line_number: AtomicI32,
    /// Java private `halt`, initially false.
    halt: AtomicBool,
    /// Java private `reconnectStatus`, initially null.
    reconnect_status: Option<BatchRunTomoStatus>,
    /// Java private `curDatasetLocation`, initially null.
    cur_dataset_location: Mutex<Option<String>>,
    /// Java private `datasetSucceeded`, initially false.
    dataset_succeeded: AtomicBool,

    /// Java private `numDone`, initially 0.
    num_done: AtomicI32,
    /// Java private `curStackID`, initially null.
    cur_stack_id: Mutex<Option<String>>,
    /// Java private `curAxisID`, initially null.
    cur_axis_id: Mutex<Option<AxisID>>,
    /// Java private `debug`, initially false.
    debug: AtomicBool,
    /// Java private volatile `running`, initially false.
    running: AtomicBool,
    /// Java private `curProjectLog`, initially null.
    cur_project_log: Mutex<Option<PathBuf>>,

    /// The class's `synchronized` methods.
    synchronized: ReentrantLock,
    /// `Thread.interrupt()` on this monitor's thread.
    thread_interrupted: AtomicBool,
    /// The subclass state and overrides.
    pub subclass: S,
}

// SAFETY: the `ReentrantLock` guard is never held across threads; every other field
// is `Send + Sync`.
unsafe impl<S: ?Sized + LogFeedMonitorImpl> Send for LogFeedMonitor<S> {}
unsafe impl<S: ?Sized + LogFeedMonitorImpl> Sync for LogFeedMonitor<S> {}

impl<S: LogFeedMonitorImpl> LogFeedMonitor<S> {
    /// Java package-private `LogFeedMonitor(BatchRunTomoManager, AxisID, RunType,
    /// RunList, ProcessData, ProcessMessages, boolean, ProcessingMethod)`, with the
    /// subclass's own state.  Called on the event dispatch thread.
    #[allow(clippy::too_many_arguments)]
    pub fn new_subclass(
        manager: &'static BatchRunTomoManager,
        axis_id: Option<AxisID>,
        run_type: Option<RunType>,
        run_list: Option<Arc<RunList>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        messages: Option<MessagesRef>,
        reconnect: bool,
        processing_method: Option<ProcessingMethod>,
        subclass: S,
    ) -> Arc<LogFeedMonitor<S>> {
        let instance = Arc::new_cyclic(|this: &Weak<LogFeedMonitor<S>>| {
            let emergency_monitor = manager.get_emergency_monitor(axis_id);
            let monitor: Weak<dyn Monitor> = this.clone();
            let tool_kit =
                MonitorToolKit::new(manager, axis_id.unwrap_or(AxisID::Only), Some(monitor));
            // When this is run from the processchunks monitor, the run list already
            // exists.
            let run_list = match run_list {
                Some(run_list) => Some(run_list),
                None => manager.create_run_list(run_type),
            };
            let busy_status_mediator = manager.get_busy_status_mediator();
            busy_status_mediator.msg_monitor_constructed(axis_id.unwrap_or(AxisID::Only));
            let mediator = manager
                .get_processing_method_mediator(axis_id)
                .map(|mediator| Arc::new(EdtRef::new(mediator)));
            // Java dereferences the run list unguarded.
            let num_datasets = run_list.as_ref().map_or(0, |run_list| run_list.size());
            let mut reconnect_status = None;
            let mut num_done = 0;
            if !reconnect {
                reconnect_status = manager.get_meta_data().get_status();
            } else {
                num_done = run_list
                    .as_ref()
                    .map_or(0, |run_list| run_list.get_num_done());
            }
            LogFeedMonitor {
                clean_print: CleanPrint::get_instance_with_label(Some("LogFeedMonitor")),
                manager,
                axis_id,
                emergency_monitor,
                messages,
                reconnect,
                process_data,
                busy_status_mediator,
                mediator,
                num_datasets,
                run_list,
                processing_method,
                tool_kit,
                update_progress_bar: AtomicBool::new(false),
                end_state: Mutex::new(None),
                commands_pipe: Mutex::new(None),
                commands_pipe_writer_id: Mutex::new(None),
                use_commands_pipe: AtomicBool::new(true),
                process_output: Mutex::new(None),
                process_output_reader_id: Mutex::new(None),
                process_running: AtomicBool::new(true),
                pausing: AtomicBool::new(false),
                killing: AtomicBool::new(false),
                stop: AtomicBool::new(false),
                process: Mutex::new(None),
                listeners: Arc::new(Mutex::new(None)),
                current_step: Mutex::new(None),
                current_dataset: Mutex::new(None),
                will_resume: AtomicBool::new(false),
                interrupted: AtomicBool::new(false),
                dataset_running: AtomicBool::new(false),
                ending_step_set: AtomicBool::new(false),
                starting_step_set: AtomicBool::new(false),
                dataset_failed: AtomicBool::new(false),
                dataset_delivered: AtomicBool::new(false),
                dataset_renamed: AtomicBool::new(false),
                live: AtomicBool::new(true),
                processed_line_number: AtomicI32::new(0),
                halt: AtomicBool::new(false),
                reconnect_status,
                cur_dataset_location: Mutex::new(None),
                dataset_succeeded: AtomicBool::new(false),
                num_done: AtomicI32::new(num_done),
                cur_stack_id: Mutex::new(None),
                cur_axis_id: Mutex::new(None),
                debug: AtomicBool::new(false),
                running: AtomicBool::new(false),
                cur_project_log: Mutex::new(None),
                synchronized: ReentrantLock::new(),
                thread_interrupted: AtomicBool::new(false),
                subclass,
            }
        });
        if reconnect {
            // Before continue to process and act on brt messages, must first go through
            // all the messages that have already been processed. Processing will
            // activate after processedLineNumber.
            let erased = instance.erased();
            erased.live.store(false, Ordering::SeqCst);
            erased
                .processed_line_number
                .store(erased.get_line_number(), Ordering::SeqCst);
            erased.reset_line_number();
            if erased.processed_line_number.load(Ordering::SeqCst) > 0
                && let Some(messages) = &erased.messages
            {
                messages.lock().unwrap().hibernate();
            }
        }
        instance
    }

    /// This monitor with its subclass erased; see the module comment.
    pub fn erased(&self) -> &LogFeedMonitor {
        self
    }
}

impl LogFeedMonitor {
    /// Java `dumpState()`.
    pub fn dump_state(&self) {
        let opt = |value: Option<String>| value.unwrap_or_else(|| "null".to_owned());
        eprint!(
            "[axisID:{},reconnect:{},numDatasets:{},\nrunList:{},\nprocessingMethod:{},updateProgressBar:{},endState:{},\ncommandsPipe:{},\ncommandsPipeWriterId:{},useCommandsPipe:{},\nprocessOutput:{},\nprocessOutputReaderId:{},processRunning:{},pausing:{},\nkilling:{},stop:{},running:{},\ncurrentStep:{},currentDataset:{},willResume:{},\ninterrupted:{},datasetRunning:{},endingStepSet:{},\nstartingStepSet:{},datasetFailed:{},datasetDelivered:{},\ndatasetRenamed:{},live:{},reconnectStatus:{},\nprocessedLineNumber:{},halt:{},\ncurDatasetLocation:{},\ndatasetSucceeded:{},numDone:{},curStackID:{}]",
            opt(self.axis_id.map(|axis_id| axis_id.to_string())),
            self.reconnect,
            self.num_datasets,
            opt(self.run_list.as_ref().map(|run_list| run_list.to_string())),
            opt(self.processing_method.map(|method| method.to_string())),
            self.update_progress_bar.load(Ordering::SeqCst),
            opt(self
                .end_state
                .lock()
                .unwrap()
                .map(|state| state.to_string())),
            opt(self
                .commands_pipe
                .lock()
                .unwrap()
                .as_ref()
                .map(|pipe| pipe.get_name())),
            opt(self
                .commands_pipe_writer_id
                .lock()
                .unwrap()
                .as_ref()
                .map(|_| "WriterId".to_owned())),
            self.use_commands_pipe.load(Ordering::SeqCst),
            opt(self
                .process_output
                .lock()
                .unwrap()
                .as_ref()
                .map(|output| output.get_name())),
            opt(self
                .process_output_reader_id
                .lock()
                .unwrap()
                .as_ref()
                .map(|_| "ReaderId".to_owned())),
            self.process_running.load(Ordering::SeqCst),
            self.pausing.load(Ordering::SeqCst),
            self.killing.load(Ordering::SeqCst),
            self.stop.load(Ordering::SeqCst),
            self.is_running(),
            opt(self.current_step.lock().unwrap().clone()),
            opt(self.current_dataset.lock().unwrap().clone()),
            self.will_resume.load(Ordering::SeqCst),
            self.interrupted.load(Ordering::SeqCst),
            self.dataset_running.load(Ordering::SeqCst),
            self.ending_step_set.load(Ordering::SeqCst),
            self.starting_step_set.load(Ordering::SeqCst),
            self.dataset_failed.load(Ordering::SeqCst),
            self.dataset_delivered.load(Ordering::SeqCst),
            self.dataset_renamed.load(Ordering::SeqCst),
            self.live.load(Ordering::SeqCst),
            opt(self.reconnect_status.map(|status| status.to_string())),
            self.processed_line_number.load(Ordering::SeqCst),
            self.halt.load(Ordering::SeqCst),
            opt(self.cur_dataset_location.lock().unwrap().clone()),
            self.dataset_succeeded.load(Ordering::SeqCst),
            self.num_done.load(Ordering::SeqCst),
            opt(self.cur_stack_id.lock().unwrap().clone()),
        );
    }

    /// Java package-private `getNumDatasets()` (virtual).
    pub fn get_num_datasets(&self) -> i32 {
        self.subclass.get_num_datasets(self)
    }

    /// Java package-private `getNumDatasets()` (the class's body).
    pub fn get_num_datasets_super(&self) -> i32 {
        self.num_datasets
    }

    /// Java final `setProcess(SystemProcessInterface)`.  Sets the process.
    pub fn set_process(&self, process: Option<Arc<dyn SystemProcessInterface>>) {
        *self.process.lock().unwrap() = process;
    }

    /// Java final `incrNumDone()`.
    pub fn incr_num_done(&self) {
        self.num_done.fetch_add(1, Ordering::SeqCst);
    }

    /// Java final `stop()`.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java final `resetRun()`.  Reset monitor to handle a process that has been
    /// automatically restarted.  Used for chunks restarted by processchunks.
    pub fn reset_run(&self) {
        self.reset_line_number();
        self.stop.store(false, Ordering::SeqCst);
        self.process_running.store(true, Ordering::SeqCst);
    }

    /// Java final `addStatusChangeListener(StatusChangeListener)`.  Called on the event
    /// dispatch thread, where the listener lives.
    pub fn add_status_change_listener_edt(
        &self,
        listener: Option<std::rc::Rc<dyn StatusChangeListener>>,
    ) {
        let Some(listener) = listener else {
            return;
        };
        let mut listeners = self.listeners.lock().unwrap();
        let mut new_collection = false;
        if listeners.is_none() {
            *listeners = Some(Vec::new());
            new_collection = true;
        }
        let list = listeners.as_mut().unwrap();
        if !new_collection
            && list.iter().any(|existing| {
                std::ptr::addr_eq(
                    std::rc::Rc::as_ptr(existing.get()),
                    std::rc::Rc::as_ptr(&listener),
                )
            })
        {
            return;
        }
        list.push(EdtRef::new(listener));
    }

    /// Java final `drop(String)`: empty.
    pub fn drop_computer(&self, _computer: &str) {}

    /// Java synchronized final `halt()` (virtual through the subclass).
    pub fn halt(&self) {
        self.subclass.halt(self);
    }

    /// Java synchronized final `halt()` (the class's body).
    pub fn halt_super(&self) {
        let _synchronized = self.synchronized.lock();
        self.halt.store(true, Ordering::SeqCst);
    }

    /// Java package-private `sendMsgStatusChangerStarted()` (virtual).
    pub fn send_msg_status_changer_started(&self) {
        self.subclass.send_msg_status_changer_started(self);
    }

    /// Java package-private `sendMsgStatusChangerStarted()` (the class's body):
    /// `batchRunTomoManager.msgStatusChangerStarted(this, false)`, on the event
    /// dispatch thread.
    pub fn send_msg_status_changer_started_super(&self) {
        self.msg_status_changer_started(false);
    }

    /// `batchRunTomoManager.msgStatusChangerStarted(this, tableOnly)`, run on the
    /// event dispatch thread (the dialog lives there).
    pub fn msg_status_changer_started(&self, table_only: bool) {
        let manager = self.manager;
        let listeners = self.listeners.clone();
        event_queue::invoke_and_wait(move || {
            let this: std::rc::Rc<dyn StatusChanger> =
                std::rc::Rc::new(LogFeedMonitorChanger { listeners });
            manager.msg_status_changer_started(&this, table_only);
        });
    }

    /// Java synchronized final `handleLockException(LockException, boolean)`.
    pub fn handle_lock_exception(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _synchronized = self.synchronized.lock();
        if let Some(lock_exception) = lock_exception {
            self.emergency_monitor.alert(Some(lock_exception), do_popup);
            self.stop();
            self.process_running.store(false, Ordering::SeqCst);
            self.set_process_end_state(Some(ProcessEndState::FileLockFailure));
            self.end_monitor_state(Some(ProcessEndState::FileLockFailure));
            self.set_running(false);
        }
    }

    /// Java `run()` (virtual).
    pub fn run(&self) {
        self.subclass.run(self);
    }

    /// `mediator.register(this)` / `deregister(this)`, on the event dispatch thread.
    fn mediator_register(&self, register: bool) {
        let Some(mediator) = self.mediator.clone() else {
            return;
        };
        let Some(this) = self.this_parallel() else {
            return;
        };
        event_queue::invoke_and_wait(move || {
            if register {
                mediator
                    .get()
                    .register_parallel_process_monitor(Some(&this));
            } else {
                mediator
                    .get()
                    .deregister_parallel_process_monitor(Some(&this));
            }
        });
    }

    /// This monitor as Java `this` in `mediator.register(this)`.
    fn this_parallel(&self) -> Option<Arc<dyn ParallelProcessMonitor>> {
        PARALLEL_THIS.with_this(self)
    }

    /// Java `run()` (the class's body).
    pub fn run_super(&self) {
        self.set_running(true);
        // try
        'try_block: {
            self.mediator_register(true);
            self.send_msg_status_changer_started();
            if !self.reconnect && etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                let message = self.subclass.get_starting_message(self);
                let manager = self.manager;
                event_queue::invoke_later(move || manager.log_message(Some(&message)));
            }
            /* Wait for processchunks or prochunks to delete .cmds file before enabling the
             * Kill Process button and Pause button. The main loop uses a sleep of 2000
             * millisecs.  This change pushes the first sleep back before the command
             * buttons are turned on.  The monitor starts running before processchunks
             * starts, so its easy to send a command to a file which is not being watched
             * and will be deleted by processchunks. Not allowing commands to be sent for
             * the period of the first sleep also reduces the chance of a collision on
             * Windows - where processchunks cannot delete the command pipe file (.cmds
             * file) because it is in use. */
            let _ = monitor_tool_kit::sleep(&self.thread_interrupted, START_SLEEP);
            // Get ready to respond to the Kill Process button and Pause button.
            self.use_commands_pipe.store(true, Ordering::SeqCst);
            // Turn on the Kill Process button and Pause button. Use the saved status if
            // this is a reconnect.
            self.initialize_progress_bar();
            self.init_status();
            let _ = monitor_tool_kit::sleep(&self.thread_interrupted, INIT_SLEEP);
            let mut failed = false;
            while self.is_running()
                && self.process_running.load(Ordering::SeqCst)
                && !self.stop.load(Ordering::SeqCst)
                && !self.halt.load(Ordering::SeqCst)
                && self.end_state.lock().unwrap().is_none()
            {
                match self.update_state() {
                    Ok(update) => {
                        if update || self.update_progress_bar.load(Ordering::SeqCst) {
                            self.update_progress_bar();
                        }
                        if (!self.reconnect || self.live.load(Ordering::SeqCst))
                            && monitor_tool_kit::sleep(&self.thread_interrupted, UPDATE_SLEEP)
                                .is_err()
                        {
                            // e.printStackTrace()
                            eprintln!("java.lang.InterruptedException: sleep interrupted");
                            self.interrupted.store(true, Ordering::SeqCst);
                        }
                    }
                    Err(e) => {
                        // FileNotFoundException (reader opener failed) or IOException.
                        eprintln!("{}", e);
                        self.end_monitor_state(Some(ProcessEndState::Failed));
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                if self.halt.load(Ordering::SeqCst) {
                    // Can halt since the updateState has returned. But all existing
                    // messages must be processed before state is valid.
                    if let Some(messages) = &self.messages {
                        let mut messages = messages.lock().unwrap();
                        messages.feed_end_message();
                        messages.close_secondary_log();
                    }
                    self.busy_status_mediator
                        .msg_monitor_stopped(self.axis_id.unwrap_or(AxisID::Only));
                    self.mediator_register(false);
                    break 'try_block;
                }
                self.end_monitor();
            }
            // Disable the use of the commands pipe.
            self.use_commands_pipe.store(false, Ordering::SeqCst);
            if self.listeners.lock().unwrap().is_some() {
                let status =
                    if self.pausing.load(Ordering::SeqCst) || self.killing.load(Ordering::SeqCst) {
                        self.get_killed_or_paused_status()
                    } else if self.ending_step_set.load(Ordering::SeqCst) {
                        BatchRunTomoStatus::Stopped
                    } else {
                        BatchRunTomoStatus::Done
                    };
                self.send_status_changed_batch_run_tomo_status(status);
            }
            if let Some(messages) = &self.messages {
                messages.lock().unwrap().close_secondary_log();
            }
            self.busy_status_mediator
                .msg_monitor_stopped(self.axis_id.unwrap_or(AxisID::Only));
            self.mediator_register(false);
        }
        // finally
        if let Some(messages) = &self.messages {
            messages.lock().unwrap().close_secondary_log();
        }
        self.set_running(false);
    }

    /// Java package-private `getCurrentDataset()`.
    pub fn get_current_dataset(&self) -> Option<String> {
        self.current_dataset.lock().unwrap().clone()
    }

    /// Java package-private `getKilledOrPausedStatus()` (virtual).
    pub fn get_killed_or_paused_status(&self) -> BatchRunTomoStatus {
        self.subclass.get_killed_or_paused_status(self)
    }

    /// Java package-private `getKilledOrPausedStatus()` (the class's body).
    pub fn get_killed_or_paused_status_super(&self) -> BatchRunTomoStatus {
        BatchRunTomoStatus::get_killed_or_paused_instance(self.is_process_chunks())
    }

    // <p>Updates done</p>

    /// Java final `initStatus()`.
    pub fn init_status(&self) {
        let mut status = self.reconnect_status;
        if status == Some(BatchRunTomoStatus::Pausing) {
            self.pausing.store(true, Ordering::SeqCst);
        } else if status == Some(BatchRunTomoStatus::Killing) {
            self.killing.store(true, Ordering::SeqCst);
        } else {
            status = Some(BatchRunTomoStatus::Running);
        }
        if let Some(status) = status {
            self.send_status_changed_batch_run_tomo_status(status);
        }
    }

    /// Java final `setPausing(boolean)`.
    pub fn set_pausing(&self, pausing: bool) {
        self.pausing.store(pausing, Ordering::SeqCst);
    }

    /// Java package-private `isProcessChunks()` (virtual).
    pub fn is_process_chunks(&self) -> bool {
        self.subclass.is_process_chunks(self)
    }

    /// Posts `new StatusChangeEventSender(listeners, ...)` with `invokeLater`.
    fn post_sender(&self, sender: StatusChangeEventSender) {
        event_queue::invoke_later(move || sender.run());
    }

    /// Java final `sendStatusChanged(BatchRunTomoStatus)`.
    pub fn send_status_changed_batch_run_tomo_status(&self, status: BatchRunTomoStatus) {
        self.post_sender(StatusChangeEventSender::new_status(
            self.listeners.clone(),
            Some(StatusRef::BatchRunTomoStatus(status)),
        ));
    }

    /// Java package-private `sendStatusChanged(Status)` (virtual).
    pub fn send_status_changed_status(&self, status: Option<StatusRef>) {
        self.subclass.send_status_changed_status(self, status);
    }

    /// Java package-private `sendStatusChanged(Status)` (the class's body).
    pub fn send_status_changed_status_super(&self, status: Option<StatusRef>) {
        let event: Arc<dyn StatusChangeEvent> = Arc::new(StatusChangeRowEvent::new(
            self.cur_stack_id.lock().unwrap().as_deref(),
            *self.cur_axis_id.lock().unwrap(),
            status,
        ));
        self.post_sender(StatusChangeEventSender::new_event(
            self.listeners.clone(),
            Some(event),
        ));
    }

    /// Java final `sendStatusChanged(String, Status)`.
    pub fn send_status_changed_stack_id(&self, stack_id: Option<&str>, status: Option<StatusRef>) {
        let event: Arc<dyn StatusChangeEvent> = Arc::new(StatusChangeRowEvent::new(
            stack_id,
            *self.cur_axis_id.lock().unwrap(),
            status,
        ));
        self.post_sender(StatusChangeEventSender::new_event(
            self.listeners.clone(),
            Some(event),
        ));
    }

    /// Java final `sendStatusChanged(Status, String)`.
    pub fn send_status_changed_file_string(&self, status: Option<StatusRef>, file_string: &str) {
        let event: Arc<dyn StatusChangeEvent> =
            Arc::new(StatusChangeRowEvent::new_with_file_string(
                self.cur_stack_id.lock().unwrap().as_deref(),
                *self.cur_axis_id.lock().unwrap(),
                status,
                Some(file_string),
            ));
        self.post_sender(StatusChangeEventSender::new_event(
            self.listeners.clone(),
            Some(event),
        ));
    }

    /// Java final `endMonitor(ProcessEndState)`.
    pub fn end_monitor_state(&self, end_state: Option<ProcessEndState>) {
        // Output file will use the success tag when it ends from a pause. Don't lose the
        // pause state unless overriding it with something other then Done.
        let current = *self.end_state.lock().unwrap();
        if current.is_none() || (end_state.is_some() && end_state != Some(ProcessEndState::Done)) {
            self.set_process_end_state(end_state);
        }
        self.end_monitor();
    }

    /// Java final `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.store(debug, Ordering::SeqCst);
    }

    /// Java final `endMonitor()`.
    pub fn end_monitor(&self) {
        self.process_running.store(false, Ordering::SeqCst); // the only place that this should be changed
        self.close_process_output();
        if let Some(messages) = &self.messages {
            messages.lock().unwrap().feed_end_message();
        }
        if let Some(messages) = &self.messages {
            messages.lock().unwrap().close_secondary_log();
        }
    }

    /// Java final `getLogFileName()`.
    pub fn get_log_file_name(&self) -> String {
        match self.get_process_output_file_name() {
            Ok(name) => name,
            Err(e) => {
                eprintln!("{}", e);
                String::new()
            }
        }
    }

    /// Java final `getProcessOutputFileName()`.
    pub fn get_process_output_file_name(&self) -> Result<String, LogFileError> {
        self.create_process_output()?;
        Ok(self
            .process_output
            .lock()
            .unwrap()
            .as_ref()
            .map(|output| output.get_name())
            .unwrap_or_default())
    }

    /// Java synchronized final `setProcessEndState(ProcessEndState)`.  Set end state,
    /// but don't override a more informative state.
    pub fn set_process_end_state(&self, end_state: Option<ProcessEndState>) {
        let _synchronized = self.synchronized.lock();
        let mut this_end_state = self.end_state.lock().unwrap();
        *this_end_state = match (*this_end_state, end_state) {
            (Some(existing), Some(new_state)) => {
                Some(ProcessEndState::precedence(existing, new_state))
            }
            (Some(ProcessEndState::Killed), None) | (Some(ProcessEndState::Paused), None) => {
                *this_end_state
            }
            (_, new_state) => new_state,
        };
    }

    /// Java synchronized `getProcessEndState()` (virtual).
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        self.subclass.get_process_end_state(self)
    }

    /// Java synchronized `getProcessEndState()` (the class's body).
    pub fn get_process_end_state_super(&self) -> Option<ProcessEndState> {
        let _synchronized = self.synchronized.lock();
        *self.end_state.lock().unwrap()
    }

    /// Java final `getSubProcessName()`.
    pub fn get_sub_process_name(&self) -> Option<String> {
        None
    }

    /// Java final `setKilling(boolean)`.
    pub fn set_killing(&self, killing: bool) {
        self.killing.store(killing, Ordering::SeqCst);
    }

    /// Java final `getReconnectStatus()`.
    pub fn get_reconnect_status(&self) -> Option<BatchRunTomoStatus> {
        self.reconnect_status
    }

    /// Java final `kill(SystemProcessInterface, AxisID)`.
    pub fn kill(&self) {
        match self.write_command("Q", BatchRunTomoStatus::Killing) {
            Ok(()) => {
                self.set_killing(true);
                self.update_progress_bar.store(true, Ordering::SeqCst);
            }
            Err(e) => eprintln!("{}", e),
        }
    }

    /// Java final `pause(SystemProcessInterface, AxisID)`.
    pub fn pause(&self) -> bool {
        // Use F to finish the current dataset.
        match self.write_command("F", BatchRunTomoStatus::Pausing) {
            Ok(()) => {
                self.pausing.store(true, Ordering::SeqCst);
                self.update_progress_bar.store(true, Ordering::SeqCst);
                true
            }
            Err(e) => {
                eprintln!("{}", e);
                false
            }
        }
    }

    /// Java `isPausing()` (virtual).
    pub fn is_pausing(&self) -> bool {
        self.subclass.is_pausing(self)
    }

    /// Java `isPausing()` (the class's body).
    pub fn is_pausing_super(&self) -> bool {
        self.pausing.load(Ordering::SeqCst) && self.process_running.load(Ordering::SeqCst)
    }

    /// Java final `setWillResume()`.
    pub fn set_will_resume(&self) {
        self.will_resume.store(true, Ordering::SeqCst);
        self.set_progress_bar_title();
    }

    /// Java final `getNumDone()`.
    pub fn get_num_done(&self) -> i32 {
        self.num_done.load(Ordering::SeqCst)
    }

    /// Java final `isProcessRunning()`.
    pub fn is_process_running(&self) -> bool {
        self.process_running.load(Ordering::SeqCst)
    }

    /// Java final `getAxisID()`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }

    /// Java final `getPid()`.
    pub fn get_pid(&self) -> Option<String> {
        None
    }

    /// Java synchronized final `closeProcessOutput()`.
    pub fn close_process_output(&self) {
        let _synchronized = self.synchronized.lock();
        let mut process_output = self.process_output.lock().unwrap();
        let mut reader_id = self.process_output_reader_id.lock().unwrap();
        if let (Some(output), Some(id)) = (process_output.as_ref(), reader_id.as_ref())
            && !id.is_empty()
        {
            output.close_id(Some(&**id));
            *process_output = None;
        }
        let _ = &mut reader_id;
    }

    /// Java private final `openReaderWithWait()`.
    fn open_reader_with_wait(&self) -> Result<(), LogFileError> {
        let Some(process_output) = self.process_output.lock().unwrap().clone() else {
            return Ok(());
        };
        for _ in 0..NUM_TRIES {
            match process_output.open_reader() {
                Ok(id) => *self.process_output_reader_id.lock().unwrap() = id,
                Err(LogFileError::Lock(e)) => {
                    self.handle_lock_exception(Some(&e), true);
                    if !self.is_running() {
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
            let opened = self
                .process_output_reader_id
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|id| !id.is_empty());
            if opened {
                if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                    eprintln!("{} opened.", process_output.get_name());
                }
                return Ok(());
            } else {
                std::thread::sleep(std::time::Duration::from_millis(TRY_SLEEP));
            }
        }
        Ok(())
    }

    /// Java final `readLineWithWait()`.
    pub fn read_line_with_wait(&self) -> Result<Option<String>, LogFileError> {
        let Some(process_output) = self.process_output.lock().unwrap().clone() else {
            return Ok(None);
        };
        let reader_id = self.process_output_reader_id.lock().unwrap().clone();
        let mut line = None;
        for _ in 0..NUM_TRIES {
            if !self.is_running() {
                break;
            }
            line = match &reader_id {
                Some(id) => process_output.read_line(id)?,
                None => None,
            };
            if line.is_some() {
                return Ok(line);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(TRY_SLEEP));
            }
        }
        Ok(line)
    }

    /// Java final `getProcessData()`.
    pub fn get_process_data(&self) -> Option<Arc<Mutex<ProcessData>>> {
        self.process_data.clone()
    }

    /// Java final `getProcessedLineNumber()`.
    pub fn get_processed_line_number(&self) -> i32 {
        self.processed_line_number.load(Ordering::SeqCst)
    }

    /// Java package-private `incrementLineNumber()` (virtual).
    pub fn increment_line_number(&self) {
        self.subclass.increment_line_number(self);
    }

    /// Java package-private `incrementLineNumber()` (the class's body).
    pub fn increment_line_number_super(&self) {
        if let Some(process_data) = &self.process_data {
            process_data.lock().unwrap().increment_line_number();
        }
    }

    /// Java package-private `getLineNumber()` (virtual).
    pub fn get_line_number(&self) -> i32 {
        self.subclass.get_line_number(self)
    }

    /// Java package-private `getLineNumber()` (the class's body).  Java dereferences
    /// `processData` unguarded; every reconnecting monitor has one.
    pub fn get_line_number_super(&self) -> i32 {
        self.process_data.as_ref().map_or(0, |process_data| {
            process_data.lock().unwrap().get_line_number()
        })
    }

    /// Java package-private `resetLineNumber()` (virtual).
    pub fn reset_line_number(&self) {
        self.subclass.reset_line_number(self);
    }

    /// Java package-private `resetLineNumber()` (the class's body).
    pub fn reset_line_number_super(&self) {
        if let Some(process_data) = &self.process_data {
            process_data.lock().unwrap().reset_line_number();
        }
    }

    /// Java package-private `gtLineNumber()` (virtual).
    pub fn gt_line_number(&self) -> bool {
        self.subclass.gt_line_number(self)
    }

    /// Java package-private `gtLineNumber()` (the class's body).
    pub fn gt_line_number_super(&self) -> bool {
        let processed = self.processed_line_number.load(Ordering::SeqCst);
        self.process_data
            .as_ref()
            .is_some_and(|process_data| process_data.lock().unwrap().gt_line_number(processed))
    }

    /// Java final `resetDataset()`.  Resets dataset and dataset location, and secondary
    /// log.
    pub fn reset_dataset(&self) {
        *self.cur_dataset_location.lock().unwrap() = None;
        *self.current_dataset.lock().unwrap() = None;
    }

    /// Java synchronized final `setDataset(String)`.  Calls setDataset(dataset,
    /// location) with data from a row in runList.
    pub fn set_dataset_stack_id(&self, stack_id: Option<&str>) {
        let _synchronized = self.synchronized.lock();
        let Some(run_list) = &self.run_list else {
            return;
        };
        let element = run_list.get(stack_id);
        if let Some(element) = element {
            self.set_dataset(element.get_root_name(), element.get_stack_location());
        }
    }

    /// Java synchronized final `setDataset(String, String)`.  Sets dataset and location,
    /// and secondary log.  Dataset and location are both required to set secondary
    /// log.  They are not available at the same time.  Secondary log also should not
    /// be reset to the same file more then once.  So rely of resetDataset to blank this
    /// stuff out, and set the secondary log very carefully.
    pub fn set_dataset(
        &self,
        temp_current_dataset: Option<&str>,
        temp_cur_dataset_location: Option<&str>,
    ) {
        let _synchronized = self.synchronized.lock();
        let mut current_dataset = self.current_dataset.lock().unwrap();
        let mut cur_dataset_location = self.cur_dataset_location.lock().unwrap();
        if current_dataset.is_some() && cur_dataset_location.is_some() {
            // Current dataset has already been set. Nothing to do.
            return;
        }
        // Set the dataset and/or location if available.
        if temp_current_dataset.is_some() && current_dataset.is_none() {
            *current_dataset = temp_current_dataset.map(str::to_owned);
        }
        if temp_cur_dataset_location.is_some() && cur_dataset_location.is_none() {
            *cur_dataset_location = temp_cur_dataset_location.map(str::to_owned);
        }
        // Set the secondary log the first time both dataset and location are available.
        if let (Some(dataset), Some(location)) =
            (current_dataset.as_deref(), cur_dataset_location.as_deref())
        {
            let file_name = file_type::CLASS.project_log.get_file_name_with_root_name(
                Some(self.manager as &'static dyn BaseManager),
                Some(dataset),
                self.axis_id,
            );
            let cur_project_log = Path::new(location).join(file_name.unwrap_or_default());
            if !cur_project_log.exists() {
                BaseProcessManager::touch(
                    &cur_project_log.to_string_lossy(),
                    Some(self.manager as &'static dyn BaseManager),
                );
            }
            // `messages.setSecondaryLog(axisID, curProjectLog)`.
            if let Some(messages) = &self.messages {
                messages
                    .lock()
                    .unwrap()
                    .set_secondary_log(self.axis_id, Some(&cur_project_log));
            }
            *self.cur_project_log.lock().unwrap() = Some(cur_project_log);
        }
    }

    /// Java synchronized final `updateState()`.
    fn update_state(&self) -> Result<bool, LogFileError> {
        let _synchronized = self.synchronized.lock();
        self.create_process_output()?;
        let reader_open = self
            .process_output_reader_id
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|id| !id.is_empty());
        if self.is_process_running() && !reader_open {
            if let Err(e) = self.open_reader_with_wait() {
                if is_file_not_found(&e) {
                    // Submitting jobs to the cluster is delaying the creation of the log
                    // file to the point that openReaderWithWait treats it as an error.
                    if self.processing_method != Some(ProcessingMethod::Queue) {
                        return Err(e);
                    }
                    return Ok(false);
                }
                return Err(e);
            }
        }
        let reader_id = self.process_output_reader_id.lock().unwrap().clone();
        let Some(reader_id) = reader_id.filter(|id| !id.is_empty()) else {
            return Ok(false);
        };
        let Some(process_output) = self.process_output.lock().unwrap().clone() else {
            return Ok(false);
        };
        self.subclass
            .process_lines(self, &process_output, &reader_id, self.reconnect)
    }

    /// Java final `setEndingStepSet(boolean)`.
    pub fn set_ending_step_set(&self, ending_step_set: bool) {
        self.ending_step_set
            .store(ending_step_set, Ordering::SeqCst);
    }

    /// Java final `appendToCurrentStep(String)`: `currentStep += append` (a null step
    /// becomes "null" + append, as Java concatenation does).
    pub fn append_to_current_step(&self, append: &str) {
        let mut current_step = self.current_step.lock().unwrap();
        let value = format!("{}{}", current_step.as_deref().unwrap_or("null"), append);
        *current_step = Some(value);
    }

    /// Java final `isDatasetRenamed()`.
    pub fn is_dataset_renamed(&self) -> bool {
        self.dataset_renamed.load(Ordering::SeqCst)
    }

    /// Java final `isDatasetDelivered()`.
    pub fn is_dataset_delivered(&self) -> bool {
        self.dataset_delivered.load(Ordering::SeqCst)
    }

    /// Java final `isDatasetFailed()`.
    pub fn is_dataset_failed(&self) -> bool {
        self.dataset_failed.load(Ordering::SeqCst)
    }

    /// Java final `getCurrentStep()`.
    pub fn get_current_step(&self) -> Option<String> {
        self.current_step.lock().unwrap().clone()
    }

    /// Java final `isDatasetRunning()`.
    pub fn is_dataset_running(&self) -> bool {
        self.dataset_running.load(Ordering::SeqCst)
    }

    /// Java final `isDatasetSucceeded()`.
    pub fn is_dataset_succeeded(&self) -> bool {
        self.dataset_succeeded.load(Ordering::SeqCst)
    }

    /// Java final `setDatasetSucceeded(boolean)`.
    pub fn set_dataset_succeeded(&self, dataset_succeeded: bool) {
        self.dataset_succeeded
            .store(dataset_succeeded, Ordering::SeqCst);
    }

    /// Java final `setCurrentStep(String)`.
    pub fn set_current_step(&self, current_step: Option<&str>) {
        *self.current_step.lock().unwrap() = current_step.map(str::to_owned);
    }

    /// Java final `isStartingStepSet()`.
    pub fn is_starting_step_set(&self) -> bool {
        self.starting_step_set.load(Ordering::SeqCst)
    }

    /// Java final `setStartingStepSet(boolean)`.
    pub fn set_starting_step_set(&self, starting_step_set: bool) {
        self.starting_step_set
            .store(starting_step_set, Ordering::SeqCst);
    }

    /// Java final `isHalt()`.
    pub fn is_halt(&self) -> bool {
        self.halt.load(Ordering::SeqCst)
    }

    /// Java final `isEndingStepSet()`.
    pub fn is_ending_step_set(&self) -> bool {
        self.ending_step_set.load(Ordering::SeqCst)
    }

    /// Java final `isInterrupted()`.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    /// Java final `setDatasetRenamed(boolean)`.
    pub fn set_dataset_renamed(&self, dataset_renamed: bool) {
        self.dataset_renamed
            .store(dataset_renamed, Ordering::SeqCst);
    }

    /// Java final `setDatasetDelivered(boolean)`.
    pub fn set_dataset_delivered(&self, dataset_delivered: bool) {
        self.dataset_delivered
            .store(dataset_delivered, Ordering::SeqCst);
    }

    /// Java final `setDatasetRunning(boolean)`.
    pub fn set_dataset_running(&self, dataset_running: bool) {
        self.dataset_running
            .store(dataset_running, Ordering::SeqCst);
    }

    /// Java final `setDatasetFailed(boolean)`.
    pub fn set_dataset_failed(&self, dataset_failed: bool) {
        self.dataset_failed.store(dataset_failed, Ordering::SeqCst);
    }

    /// Java final `setLive(boolean)`.
    pub fn set_live(&self, live: bool) {
        self.live.store(live, Ordering::SeqCst);
    }

    /// Java final `setCurStackID(String)`.
    pub fn set_cur_stack_id(&self, cur_stack_id: Option<&str>) {
        *self.cur_stack_id.lock().unwrap() = cur_stack_id.map(str::to_owned);
    }

    /// Java final `setCurAxisID(AxisID)`.
    pub fn set_cur_axis_id(&self, cur_axis_id: Option<AxisID>) {
        *self.cur_axis_id.lock().unwrap() = cur_axis_id;
    }

    /// Java final `getCurAxisID()`.
    pub fn get_cur_axis_id(&self) -> Option<AxisID> {
        *self.cur_axis_id.lock().unwrap()
    }

    /// Java final `isCurAxisIDNull()`.
    pub fn is_cur_axis_id_null(&self) -> bool {
        self.cur_axis_id.lock().unwrap().is_none()
    }

    /// Java final `getCurStackID()`.
    pub fn get_cur_stack_id(&self) -> Option<String> {
        self.cur_stack_id.lock().unwrap().clone()
    }

    /// Java final `resetCurStackID()`.
    pub fn reset_cur_stack_id(&self) {
        *self.cur_stack_id.lock().unwrap() = None;
    }

    /// Java final `getEndState()`.
    pub fn get_end_state(&self) -> Option<ProcessEndState> {
        *self.end_state.lock().unwrap()
    }

    /// Java final `updateProgressBar()`.
    pub fn update_progress_bar(&self) {
        self.update_progress_bar.store(false, Ordering::SeqCst);
        let _changed = self.set_progress_bar_title();
        let status = self.subclass.get_status_string(self);
        if let Some(status) = status {
            self.set_progress_bar_value(&status);
        }
    }

    /// Java final `setProgressBarTitle()`.
    pub fn set_progress_bar_title(&self) -> bool {
        let current_dataset = self.current_dataset.lock().unwrap().clone();
        self.subclass.set_progress_bar_title(
            self,
            self.process_running.load(Ordering::SeqCst),
            self.killing.load(Ordering::SeqCst),
            self.pausing.load(Ordering::SeqCst),
            self.will_resume.load(Ordering::SeqCst),
            current_dataset.as_deref(),
        )
    }

    /// Java final `initializeProgressBar()`.
    pub fn initialize_progress_bar(&self) {
        self.set_progress_bar_title();
        self.tool_kit.initialize_progress_bar(
            &self.subclass.get_title(self),
            self.is_indeterminate_progress_bar_mode(),
        );
    }

    /// Java final `setProgressBarValue(String)`.  The main panel is an event dispatch
    /// thread object: the call is posted there, and its result (unused by every
    /// caller) is reported as false.
    pub fn set_progress_bar_value(&self, bar_string: &str) -> bool {
        if !self.has_progress_bar_access() {
            return false;
        }
        let value = self.num_done.load(Ordering::SeqCst);
        let bar_string = bar_string.to_owned();
        let axis_id = self.axis_id.unwrap_or(AxisID::Only);
        self.manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_value_int_string_axis_id(value, Some(&bar_string), axis_id);
        }));
        false
    }

    /// Java final `isKilling()`.
    pub fn is_killing(&self) -> bool {
        self.killing.load(Ordering::SeqCst)
    }

    /// Java final `setProgressBar(String)`.  Posted to the event dispatch thread, as
    /// `setProgressBarValue` is; returns false ("values have changed" is unused).
    pub fn set_progress_bar(&self, label: &str) -> bool {
        if !self.has_progress_bar_access() || self.get_num_datasets() <= 0 {
            return false;
        }
        let label = label.to_owned();
        let num_datasets = self.num_datasets;
        let indeterminate = self.is_indeterminate_progress_bar_mode();
        let axis_id = self.axis_id.unwrap_or(AxisID::Only);
        let killing = self.killing.load(Ordering::SeqCst);
        self.manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_string_int_boolean_axis_id_boolean(
                Some(&label),
                num_datasets,
                indeterminate,
                axis_id,
                !killing,
            );
        }));
        false
    }

    /// Java package-private `isIndeterminateProgressBarMode()` (virtual).
    pub fn is_indeterminate_progress_bar_mode(&self) -> bool {
        self.subclass.is_indeterminate_progress_bar_mode(self)
    }

    /// Java final `useMessageReporter()`: empty.
    pub fn use_message_reporter(&self) {}

    /// Java private final `writeCommand(String, BatchRunTomoStatus)`.  Create
    /// commandsWriter if necessary, write command, add newline, flush.
    fn write_command(&self, command: &str, status: BatchRunTomoStatus) -> Result<(), LogFileError> {
        if !self.use_commands_pipe.load(Ordering::SeqCst) {
            return Ok(());
        }
        let commands_pipe = {
            let mut commands_pipe = self.commands_pipe.lock().unwrap();
            if commands_pipe.is_none() {
                let check_file = self
                    .subclass
                    .get_check_file_type(self)
                    .get_file(Some(self.manager as &'static dyn BaseManager), self.axis_id);
                *commands_pipe = Some(LogFile::get_instance_file(
                    check_file.as_deref(),
                    Some(self.manager.get_emergency_monitor(self.axis_id)),
                )?);
            }
            commands_pipe.clone().unwrap()
        };
        let has_writer = self
            .commands_pipe_writer_id
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|id| !id.is_empty());
        if !has_writer {
            match commands_pipe.open_writer_append(true) {
                Ok(id) => *self.commands_pipe_writer_id.lock().unwrap() = Some(id),
                Err(LogFileError::Lock(e)) => {
                    self.handle_lock_exception(Some(&e), true);
                    if !self.is_running() {
                        return Ok(());
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let writer_id = self.commands_pipe_writer_id.lock().unwrap().clone();
        let Some(writer_id) = writer_id.filter(|id| !id.is_empty()) else {
            return Ok(());
        };
        commands_pipe.write(Some(command), &writer_id)?;
        commands_pipe.new_line(&writer_id)?;
        commands_pipe.flush(&writer_id)?;
        // Close writer after each write. If it is kept open, the file would not be
        // writeable from the command line in Windows.
        commands_pipe.close_id(Some(&*writer_id));
        *self.commands_pipe_writer_id.lock().unwrap() = None;
        self.send_status_changed_batch_run_tomo_status(status);
        Ok(())
    }

    /// Java synchronized final `createProcessOutput()`.  Not responsible for backing up
    /// the process output file.
    pub fn create_process_output(&self) -> Result<(), LogFileError> {
        let _synchronized = self.synchronized.lock();
        if self.process_output.lock().unwrap().is_none() {
            let output = self.subclass.build_process_output_log_file(self)?;
            *self.process_output.lock().unwrap() = Some(output);
        }
        Ok(())
    }

    /// Java private synchronized `setRunning(boolean)`.
    fn set_running(&self, running: bool) {
        let _synchronized = self.synchronized.lock();
        self.running.store(running, Ordering::SeqCst);
    }

    /// Java `isRunning()` (virtual).
    pub fn is_running(&self) -> bool {
        self.subclass.is_running(self)
    }

    /// Java `isRunning()` (the class's body).
    pub fn is_running_super(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Java `hasProgressBarAccess()` (virtual).
    pub fn has_progress_bar_access(&self) -> bool {
        self.subclass.has_progress_bar_access(self)
    }

    /// Java `isReconnect()`.
    pub fn is_reconnect(&self) -> bool {
        false
    }

    /// Java final `getToolKit()`.
    pub fn get_tool_kit(&self) -> &MonitorToolKit {
        &self.tool_kit
    }

    /// The monitor thread's `Thread.interrupt()` flag (for `monitor_tool_kit::sleep`).
    pub fn thread_interrupted(&self) -> &AtomicBool {
        &self.thread_interrupted
    }

    /// The `cleanPrint` the class keeps for debugging.
    pub fn clean_print(&self) -> &CleanPrint {
        &self.clean_print
    }

    /// The dataset project log the secondary log was set to (`curProjectLog`).
    pub fn get_cur_project_log(&self) -> Option<PathBuf> {
        self.cur_project_log.lock().unwrap().clone()
    }

    /// The listener list (`StatusChangeEventSender` takes it).
    pub fn get_listeners(&self) -> StatusChangeListeners {
        self.listeners.clone()
    }
}

/// Java static final `getFileFromOutput(String, String, String)`.  Gets a file path
/// from a recognized output line.  Returns null if the status had been previously
/// sent.  Returns the file path if it is found.
pub fn get_file_from_output(
    line: &str,
    msg_id: &str,
    deprecated_tag: Option<&str>,
) -> Option<String> {
    // Attempt to recognize the output line.
    if !line.contains(msg_id) && deprecated_tag.is_none_or(|tag| !line.contains(tag)) {
        return None;
    }
    // Attempt to get the file.
    let index = line.find(process_output_strings::BRT_FILE_LOCATION_TAG)?;
    let mut file = line[index + process_output_strings::BRT_FILE_LOCATION_TAG.len()..]
        .trim()
        .to_owned();
    if file.is_empty() {
        return None;
    }
    // Remove the message ID if it is at the end of the line.
    if let Some(index) = file.find(msg_id) {
        file = file[..index].trim().to_owned();
    }
    if file.is_empty() {
        return None;
    }
    Some(file)
}

/// The `StatusChanger` a monitor hands to the batchruntomo dialog
/// (`msgStatusChangerStarted(this, ...)`): adding a listener adds it to the monitor's
/// list.  Built on the monitor thread, used on the event dispatch thread.
pub struct LogFeedMonitorChanger {
    listeners: StatusChangeListeners,
}

impl StatusChanger for LogFeedMonitorChanger {
    /// Java final `LogFeedMonitor.addStatusChangeListener(StatusChangeListener)`.
    fn add_status_change_listener(&self, listener: Option<std::rc::Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        let mut listeners = self.listeners.lock().unwrap();
        let mut new_collection = false;
        if listeners.is_none() {
            *listeners = Some(Vec::new());
            new_collection = true;
        }
        let list = listeners.as_mut().unwrap();
        if !new_collection
            && list.iter().any(|existing| {
                std::ptr::addr_eq(
                    std::rc::Rc::as_ptr(existing.get()),
                    std::rc::Rc::as_ptr(&listener),
                )
            })
        {
            return;
        }
        list.push(EdtRef::new(listener));
    }
}

/// The trait impls bind the Java interfaces to the inherent methods.
impl<S: LogFeedMonitorImpl> Monitor for LogFeedMonitor<S> {
    fn run(&self) {
        self.erased().run();
    }
    fn is_running(&self) -> bool {
        self.erased().is_running()
    }
    fn is_pausing(&self) -> bool {
        self.erased().is_pausing()
    }
    fn set_will_resume(&self) {
        self.erased().set_will_resume();
    }
    fn halt(&self) {
        self.erased().halt();
    }
    fn has_progress_bar_access(&self) -> bool {
        self.erased().has_progress_bar_access()
    }
    fn is_reconnect(&self) -> bool {
        self.erased().is_reconnect()
    }
    fn get_process_end_state(&self) -> Option<ProcessEndState> {
        self.erased().get_process_end_state()
    }
    fn interrupt(&self) {
        self.thread_interrupted.store(true, Ordering::SeqCst);
    }
}

impl<S: LogFeedMonitorImpl> ProcessMonitor for LogFeedMonitor<S> {
    fn set_process_end_state(&self, end_state: ProcessEndState) {
        self.erased().set_process_end_state(Some(end_state));
    }
    fn kill(&self, _process: &dyn SystemProcessInterface, _axis_id: AxisID) {
        self.erased().kill();
    }
    fn pause(&self, _process: &dyn SystemProcessInterface, _axis_id: AxisID) -> bool {
        self.erased().pause()
    }
    fn get_status_string(&self) -> Option<String> {
        self.subclass.get_status_string(self.erased())
    }
    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        self.messages
            .as_ref()
            .map(|messages| messages.lock().unwrap())
    }
    fn stop(&self) {
        self.erased().stop();
    }
    fn use_message_reporter(&self) {
        self.erased().use_message_reporter();
    }
    fn dump_state(&self) {
        self.erased().dump_state();
    }
    fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        self.tool_kit
            .msg_log_file_renaming_handle(log_file, indeterminate_mode);
    }
    fn msg_log_file_renaming_name(&self, file_name: &str) {
        self.tool_kit.msg_log_file_renaming_name(file_name);
    }
    fn msg_log_file_renaming_file(&self, file: &Path) {
        self.tool_kit.msg_log_file_renaming_file(file);
    }
    fn msg_log_file_renamed(&self) {
        self.subclass.msg_log_file_renamed(self.erased());
    }
    fn msg_log_file_renaming_failed(&self) {
        self.tool_kit.msg_log_file_renaming_failed();
    }
}

impl<S: LogFeedMonitorImpl> DetachedProcessMonitor for LogFeedMonitor<S> {
    fn is_process_running(&self) -> bool {
        self.erased().is_process_running()
    }
    fn get_process_output_file_name(&self) -> Result<String, LogFileError> {
        self.erased().get_process_output_file_name()
    }
    fn set_process(&self, process: Arc<dyn SystemProcessInterface>) {
        self.erased().set_process(Some(process));
    }
}

impl<S: LogFeedMonitorImpl> OutfileProcessMonitor for LogFeedMonitor<S> {
    fn get_pid(&self) -> Option<String> {
        self.erased().get_pid()
    }
    fn end_monitor(&self, end_state: ProcessEndState) {
        self.erased().end_monitor_state(Some(end_state));
    }
    fn get_sub_process_name(&self) -> Option<String> {
        self.erased().get_sub_process_name()
    }
}

impl<S: LogFeedMonitorImpl> ParallelProcessMonitor for LogFeedMonitor<S> {
    /// Java final `drop(String)`.
    fn drop(&self, computer: &str) {
        self.erased().drop_computer(computer);
    }
}

/// How `mediator.register(this)` finds this monitor as an
/// `Arc<dyn ParallelProcessMonitor>`: each concrete monitor type registers a
/// conversion from its erased form when it is built.
struct ParallelThis;
static PARALLEL_THIS: ParallelThis = ParallelThis;

/// The erased monitors that can be handed out as `Arc<dyn ParallelProcessMonitor>`,
/// keyed by address.
static PARALLEL_REGISTRY: Mutex<Vec<(usize, Weak<dyn ParallelProcessMonitor>)>> =
    Mutex::new(Vec::new());

impl ParallelThis {
    fn with_this(&self, monitor: &LogFeedMonitor) -> Option<Arc<dyn ParallelProcessMonitor>> {
        let address = monitor as *const LogFeedMonitor as *const () as usize;
        let registry = PARALLEL_REGISTRY.lock().unwrap();
        registry
            .iter()
            .find(|(key, _)| *key == address)
            .and_then(|(_, weak)| weak.upgrade())
    }
}

/// Registers a built monitor so the erased type can reach its `Arc` (see
/// `ParallelThis`).  Every constructor of a concrete monitor calls it.
pub fn register_parallel_this<S: LogFeedMonitorImpl>(monitor: &Arc<LogFeedMonitor<S>>) {
    let erased: &LogFeedMonitor = monitor.erased();
    let address = erased as *const LogFeedMonitor as *const () as usize;
    let parallel: Arc<dyn ParallelProcessMonitor> = monitor.clone();
    let mut registry = PARALLEL_REGISTRY.lock().unwrap();
    registry.retain(|(_, weak)| weak.strong_count() > 0);
    registry.push((address, Arc::downgrade(&parallel)));
}
