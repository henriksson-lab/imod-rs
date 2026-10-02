//! `IMOD/Etomo/src/etomo/process/ProcesschunksProcessMonitor.java`.
//!
//! Description: Monitor for processchucks.  Follows processchunks.out.
//!
//! # How the class and its subclasses are represented
//!
//! Java `ProcesschunksProcessMonitor` is a concrete class that two others
//! extend (`ProcesschunksVolcombineMonitor`, `ProcesschunksBatchRunTomoMonitor`)
//! by overriding a set of its methods.  Almost every method here calls one of
//! those overridable methods, so the class is one struct generic over the
//! subclass state, `ProcesschunksProcessMonitor<S>`, whose last field is the
//! subclass.  `ProcesschunksProcessMonitor` without a parameter is the type
//! with the subclass erased (`S = dyn ProcesschunksProcessMonitorImpl`); every
//! Java method is an inherent method of that type, and an instance of any
//! subclass reaches it by unsized coercion (`let this:
//! &ProcesschunksProcessMonitor = monitor;`).
//!
//! [`ProcesschunksProcessMonitorImpl`] is the set of overridable methods.  Each
//! takes the erased monitor as `this` and defaults to the Java body, which is
//! the inherent `<name>_super` method (Java's `super.<name>()`); the inherent
//! `<name>` method is the virtual call, `self.subclass.<name>(self)`.  The Java
//! class itself, with no subclass, is `ProcesschunksProcessMonitor<()>`.
//!
//! The monitor interfaces (`OutfileProcessMonitor` and the ones it extends,
//! `ParallelProcessMonitor`) are implemented for every sized
//! `ProcesschunksProcessMonitor<S>`, so an `Arc` of one coerces to
//! `Arc<dyn OutfileProcessMonitor>`.
//!
//! Java's `synchronized` methods take one instance lock.  `createProcessOutput`
//! calls `handleLockException` while holding it, which Java allows because the
//! lock is re-entrant; `std::sync::Mutex` is not, so `createProcessOutput` and
//! `closeProcessOutput` share `process_output_lock`, `handleLockException`
//! takes `synchronized`, and `endState`/`running` are behind their own locks.
//!
//! `ParallelProgressDisplay` is a Swing panel living on the event dispatch
//! thread; calls on it are posted there (`util/event_queue.rs`).

use super::emergency_monitor::EmergencyMonitor;
use super::monitor::{DetachedProcessMonitor, Monitor, OutfileProcessMonitor, ProcessMonitor};
use super::monitor_tool_kit::{self, InterruptedException, MonitorToolKit, WHITESPACE};
use super::parallel_process_monitor::ParallelProcessMonitor;
use super::process_data::ProcessData;
use super::process_interface::SystemProcessInterface;
use super::process_messages::{MessageType, ProcessMessages};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::log_file::{
    Handle, LockException, LogFile, LogFileError, ReaderId, WriterId,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::const_string_property::ConstStringProperty;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// Java `NO_TCSH_ERROR`.
const NO_TCSH_ERROR: i32 = 5;
/// Java `TITLE`.
const TITLE: &str = "Processchunks";
/// Java `START_SLEEP`.
const START_SLEEP: u64 = 2000;
/// Java `INIT_SLEEP`.
const INIT_SLEEP: u64 = 1000;
/// Java `UPDATE_SLEEP`.
const UPDATE_SLEEP: u64 = 2000;
/// Java `KILL_SLEEP`.
const KILL_SLEEP: u64 = 2001;

/// Java `SUCCESS_TAG`.
pub const SUCCESS_TAG: &str = "Finished reassembling";

/// Java private static `debugLevel`, set by every constructor.
static DEBUG_LEVEL: Mutex<Option<DebugLevel>> = Mutex::new(None);

/// A `ParallelProgressDisplay` as a monitor holds it: shared with the monitor
/// thread through the `Arc`, used only on the event dispatch thread.
pub type ParallelProgressDisplayRef = Arc<EdtRef<dyn ParallelProgressDisplay>>;

/// What Java `updateState` throws: its declared `LogFileException` and
/// `IOException`, and the unchecked `IllegalStateException` it raises for a
/// bad command.
#[derive(Debug)]
pub enum UpdateStateError {
    /// `LogFile.LogFileException` and `java.io.IOException`.
    LogFile(LogFileError),
    /// `java.lang.IllegalStateException`, with its message.
    IllegalState(String),
}

impl std::fmt::Display for UpdateStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateStateError::LogFile(e) => write!(f, "{e}"),
            UpdateStateError::IllegalState(message) => {
                write!(f, "java.lang.IllegalStateException: {message}")
            }
        }
    }
}

impl From<LogFileError> for UpdateStateError {
    fn from(e: LogFileError) -> UpdateStateError {
        UpdateStateError::LogFile(e)
    }
}

/// The overridable methods of Java `ProcesschunksProcessMonitor`; see the
/// module comment.  Each default is the Java body.
pub trait ProcesschunksProcessMonitorImpl: Send + Sync + 'static {
    /// Java `run`.
    fn run(&self, this: &ProcesschunksProcessMonitor) {
        this.run_super();
    }

    /// Java `isRunning`.
    fn is_running(&self, this: &ProcesschunksProcessMonitor) -> bool {
        this.is_running_super()
    }

    /// Java `setProcess`.
    fn set_process(
        &self,
        this: &ProcesschunksProcessMonitor,
        process: Option<Arc<dyn SystemProcessInterface>>,
    ) {
        this.set_process_super(process);
    }

    /// Java `setUseCommandsPipe`.
    fn set_use_commands_pipe(&self, this: &ProcesschunksProcessMonitor, use_: bool) {
        this.set_use_commands_pipe_super(use_);
    }

    /// Java `updateParallelProgressDisplay`.
    fn update_parallel_progress_display(
        &self,
        this: &ProcesschunksProcessMonitor,
        display: Option<&ParallelProgressDisplayRef>,
    ) {
        this.update_parallel_progress_display_super(display);
    }

    /// Java `processMessage`.
    fn process_message(&self, this: &ProcesschunksProcessMonitor, line: &str) {
        let _ = (this, line);
    }

    /// Java `writeKillCommand`.
    fn write_kill_command(&self, this: &ProcesschunksProcessMonitor) -> Result<(), LogFileError> {
        this.write_kill_command_super()
    }

    /// Java `isKilling`.
    fn is_killing(&self, this: &ProcesschunksProcessMonitor) -> bool {
        this.is_killing_super()
    }

    /// Java `pause`.
    fn pause(
        &self,
        this: &ProcesschunksProcessMonitor,
        process: &dyn SystemProcessInterface,
        axis_id: AxisID,
    ) -> bool {
        this.pause_super(process, axis_id)
    }

    /// Java `isPausing`.
    fn is_pausing(&self, this: &ProcesschunksProcessMonitor) -> bool {
        this.is_pausing_super()
    }

    /// Java `setWillResume`.
    fn set_will_resume(&self, this: &ProcesschunksProcessMonitor) {
        this.set_will_resume_super();
    }

    /// Java `synchronized closeProcessOutput`.
    fn close_process_output(&self, this: &ProcesschunksProcessMonitor) {
        this.close_process_output_super();
    }

    /// Java `loadParallelProgressDisplay`.
    fn load_parallel_progress_display(&self, this: &ProcesschunksProcessMonitor) {
        this.load_parallel_progress_display_super();
    }

    /// Java `handlePauseMessage`.
    fn handle_pause_message(&self, this: &ProcesschunksProcessMonitor) {
        this.handle_pause_message_super();
    }

    /// Java `updateState`.
    fn update_state(&self, this: &ProcesschunksProcessMonitor) -> Result<bool, UpdateStateError> {
        this.update_state_super()
    }

    /// Java `updateProgressBar`.
    fn update_progress_bar(&self, this: &ProcesschunksProcessMonitor) {
        this.update_progress_bar_super();
    }

    /// Java `useMessageReporter`.
    fn use_message_reporter(&self, this: &ProcesschunksProcessMonitor) {
        let _ = this;
    }

    /// Java `getCheckFile`.
    fn get_check_file(&self, this: &ProcesschunksProcessMonitor) -> String {
        this.get_check_file_super()
    }

    /// Java `halt`.
    fn halt(&self, this: &ProcesschunksProcessMonitor) {
        let _ = this;
    }

    /// Java `hasProgressBarAccess`.
    fn has_progress_bar_access(&self, this: &ProcesschunksProcessMonitor) -> bool {
        let _ = this;
        true
    }
}

/// The Java class itself, with no subclass state or overrides.
impl ProcesschunksProcessMonitorImpl for () {}

/// Java `ProcesschunksProcessMonitor`; see the module comment.
pub struct ProcesschunksProcessMonitor<
    S: ?Sized + ProcesschunksProcessMonitorImpl = dyn ProcesschunksProcessMonitorImpl,
> {
    /// Java field `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    pub axis_id: AxisID,
    /// Java field `emergencyMonitor`.
    emergency_monitor: Arc<EmergencyMonitor>,
    /// Java field `busyStatusMediator`.
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java field `toolKit`.
    tool_kit: MonitorToolKit,

    /// Java field `nChunks`.
    n_chunks: Mutex<EtomoNumber>,
    /// Java field `chunksFinished`.
    chunks_finished: Mutex<EtomoNumber>,
    /// Java field `rootName`.
    root_name: Option<String>,
    /// Java field `messages`.
    messages: Mutex<ProcessMessages>,
    /// Java field `mediator`: an event-dispatch-thread object, called there.
    mediator: Option<Arc<EdtRef<ProcessingMethodMediator>>>,
    /// Java `this` as the `ParallelProcessMonitor` the mediator registers.
    parallel_this: Weak<dyn ParallelProcessMonitor>,
    /// Java field `subdirName`.
    subdir_name: Mutex<Option<String>>,
    /// Java field `setProgressBarTitle`: turn on to changed the progress bar
    /// title.
    set_progress_bar_title: AtomicBool,
    /// Java field `reassembling`.
    reassembling: AtomicBool,
    /// Java field `endState`.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java field `commandsPipe`.
    commands_pipe: Mutex<Option<Arc<Handle>>>,
    /// Java field `commandsPipeWriterId`.
    commands_pipe_writer_id: Mutex<Option<WriterId>>,
    /// Java field `useCommandsPipe`.
    use_commands_pipe: AtomicBool,
    /// Java field `processOutput`.
    process_output: Mutex<Option<Arc<Handle>>>,
    /// Java field `processOutputReaderId`.
    process_output_reader_id: Mutex<Option<ReaderId>>,
    /// Java field `processRunning`.
    process_running: AtomicBool,
    /// Java field `pausing`.
    pausing: AtomicBool,
    /// Java field `killing`.
    killing: AtomicBool,
    /// Java field `pid`.
    pid: Mutex<Option<String>>,
    /// Java field `starting`.
    starting: AtomicBool,
    /// Java field `finishing`.
    finishing: AtomicBool,
    /// Java field `stop`.
    stop: AtomicBool,
    /// Java field `halt`.
    halt: AtomicBool,
    /// Java field `reconnect`.
    reconnect: AtomicBool,
    /// Java field `process`.
    process: Mutex<Option<Arc<dyn SystemProcessInterface>>>,
    /// Java field `computerMap`.
    pub computer_map: Option<BTreeMap<String, String>>,
    /// Java field `multiLineMessages`.
    multi_line_messages: bool,
    /// Java field `tcshErrorCountDown`.
    tcsh_error_count_down: AtomicI32,
    /// Java field `parallelProgressDisplay`.
    parallel_progress_display: Mutex<Option<ParallelProgressDisplayRef>>,
    /// Java field `messageReporter`.
    // TODO(unit): needs etomo/process/MessageReporter.java - created by
    // `createProcessOutput` (`new MessageReporter(axisID, processOutput)`),
    // polled by `updateProgressBar` and closed by `closeProcessOutput`.
    message_reporter: Mutex<Option<Infallible>>,
    /// Java field `willResume`.
    will_resume: AtomicBool,
    /// Java `volatile` field `running`.
    running: AtomicBool,

    /// The `synchronized` of `handleLockException`; see the module comment.
    synchronized: Mutex<()>,
    /// The `synchronized` of `createProcessOutput` and `closeProcessOutput`.
    process_output_lock: Mutex<()>,
    /// `Thread.interrupt()` on this monitor's thread.
    pub interrupted: AtomicBool,

    /// The subclass state and overrides.
    pub subclass: S,
}

impl ProcesschunksProcessMonitor<()> {
    /// Java `ProcesschunksProcessMonitor(BaseManager, AxisID, String, Map,
    /// boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        computer_map: Option<BTreeMap<String, String>>,
        multi_line_messages: bool,
    ) -> Arc<ProcesschunksProcessMonitor<()>> {
        ProcesschunksProcessMonitor::new_subclass(
            manager,
            axis_id,
            root_name,
            computer_map,
            multi_line_messages,
            (),
        )
    }

    /// Java static `getReconnectInstance`.
    pub fn get_reconnect_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_data: &ProcessData,
        multi_line_messages: bool,
    ) -> Arc<ProcesschunksProcessMonitor<()>> {
        let instance = ProcesschunksProcessMonitor::new(
            manager,
            axis_id,
            process_data.get_sub_process_name().as_deref(),
            process_data.get_computer_map().cloned(),
            multi_line_messages,
        );
        instance.reconnect.store(true, Ordering::SeqCst);
        instance
    }
}

impl<S: ProcesschunksProcessMonitorImpl> ProcesschunksProcessMonitor<S> {
    /// The Java constructor as reached from a subclass constructor's
    /// `super(manager, axisID, rootName, computerMap, multiLineMessages)`,
    /// with that subclass's own state.
    pub fn new_subclass(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        computer_map: Option<BTreeMap<String, String>>,
        multi_line_messages: bool,
        subclass: S,
    ) -> Arc<ProcesschunksProcessMonitor<S>> {
        Arc::new_cyclic(|this: &Weak<ProcesschunksProcessMonitor<S>>| {
            let emergency_monitor = manager.get_emergency_monitor(Some(axis_id));
            let monitor: Weak<dyn Monitor> = this.clone();
            let tool_kit = MonitorToolKit::new(manager, axis_id, Some(monitor));
            let busy_status_mediator = manager.get_busy_status_mediator();
            busy_status_mediator.msg_monitor_constructed(axis_id);
            let messages =
                ProcessMessages::get_instance_for_parallel_processing(multi_line_messages);
            let mediator = manager
                .get_processing_method_mediator(Some(axis_id))
                .map(|mediator| Arc::new(EdtRef::new(mediator)));
            let parallel_this: Weak<dyn ParallelProcessMonitor> = this.clone();
            *DEBUG_LEVEL.lock().unwrap() =
                Some(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level());
            ProcesschunksProcessMonitor {
                manager,
                axis_id,
                emergency_monitor,
                busy_status_mediator,
                tool_kit,
                n_chunks: Mutex::new(EtomoNumber::new()),
                chunks_finished: Mutex::new(EtomoNumber::new()),
                root_name: root_name.map(str::to_string),
                messages: Mutex::new(messages),
                mediator,
                parallel_this,
                subdir_name: Mutex::new(None),
                set_progress_bar_title: AtomicBool::new(false),
                reassembling: AtomicBool::new(false),
                end_state: Mutex::new(None),
                commands_pipe: Mutex::new(None),
                commands_pipe_writer_id: Mutex::new(None),
                use_commands_pipe: AtomicBool::new(true),
                process_output: Mutex::new(None),
                process_output_reader_id: Mutex::new(None),
                process_running: AtomicBool::new(true),
                pausing: AtomicBool::new(false),
                killing: AtomicBool::new(false),
                pid: Mutex::new(None),
                starting: AtomicBool::new(true),
                finishing: AtomicBool::new(false),
                stop: AtomicBool::new(false),
                halt: AtomicBool::new(false),
                reconnect: AtomicBool::new(false),
                process: Mutex::new(None),
                computer_map,
                multi_line_messages,
                tcsh_error_count_down: AtomicI32::new(NO_TCSH_ERROR),
                parallel_progress_display: Mutex::new(None),
                message_reporter: Mutex::new(None),
                will_resume: AtomicBool::new(false),
                running: AtomicBool::new(false),
                synchronized: Mutex::new(()),
                process_output_lock: Mutex::new(()),
                interrupted: AtomicBool::new(false),
                subclass,
            }
        })
    }

    /// This monitor with its subclass erased; see the module comment.
    pub fn erased(&self) -> &ProcesschunksProcessMonitor {
        self
    }
}

impl ProcesschunksProcessMonitor {
    /// Java final `dumpState`.
    pub fn dump_state(&self) {
        eprint!(
            "[rootName:{},subdirName:{},\nsetProgressBarTitle:{},useCommandsPipe:{},\nprocessRunning:{},pausing:{},\nkilling:{},pid:{},starting:{},\nfinishing:{},stop:{},running:{},\nreconnect:{},multiLineMessages:{},\ntcshErrorCountDown:{}]",
            self.root_name.as_deref().unwrap_or("null"),
            self.subdir_name
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null"),
            self.set_progress_bar_title.load(Ordering::SeqCst),
            self.use_commands_pipe.load(Ordering::SeqCst),
            self.process_running.load(Ordering::SeqCst),
            self.pausing.load(Ordering::SeqCst),
            self.killing.load(Ordering::SeqCst),
            self.pid.lock().unwrap().as_deref().unwrap_or("null"),
            self.starting.load(Ordering::SeqCst),
            self.finishing.load(Ordering::SeqCst),
            self.stop.load(Ordering::SeqCst),
            self.is_running(),
            self.reconnect.load(Ordering::SeqCst),
            self.multi_line_messages,
            self.tcsh_error_count_down.load(Ordering::SeqCst),
        );
    }

    /// Java final `setReconnect`.
    pub fn set_reconnect(&self, reconnect: bool) {
        self.reconnect.store(reconnect, Ordering::SeqCst);
    }

    /// Java final `isReconnect`.
    pub fn is_reconnect(&self) -> bool {
        self.reconnect.load(Ordering::SeqCst)
    }

    /// Java `setProcess` (virtual).
    pub fn set_process(&self, process: Option<Arc<dyn SystemProcessInterface>>) {
        self.subclass.set_process(self, process);
    }

    /// Java `setProcess` (the class's body).  Sets the process.  Then, because
    /// this is a parallel process monitor, it sets computerMap in the process
    /// if this isn't a reconnect.  This causes the process to set the
    /// computerMap in ProcessData.
    pub fn set_process_super(&self, process: Option<Arc<dyn SystemProcessInterface>>) {
        *self.process.lock().unwrap() = process.clone();
        if !self.reconnect.load(Ordering::SeqCst) {
            if let Some(process) = process {
                process.set_computer_map(self.computer_map.clone());
            }
        }
    }

    /// Java `setHalt`.
    pub fn set_halt(&self, halt: bool) {
        self.halt.store(halt, Ordering::SeqCst);
    }

    /// Java `isHalt`.
    pub fn is_halt(&self) -> bool {
        self.halt.load(Ordering::SeqCst)
    }

    /// Java final `stop`.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java `setUseCommandsPipe` (virtual).
    pub fn set_use_commands_pipe(&self, use_: bool) {
        self.subclass.set_use_commands_pipe(self, use_);
    }

    /// Java `setUseCommandsPipe` (the class's body).
    pub fn set_use_commands_pipe_super(&self, use_: bool) {
        self.use_commands_pipe.store(use_, Ordering::SeqCst);
    }

    /// Java `synchronized final handleLockException`.
    pub fn handle_lock_exception(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _synchronized = self.synchronized.lock().unwrap();
        if let Some(lock_exception) = lock_exception {
            self.emergency_monitor.alert(Some(lock_exception), do_popup);
            self.stop();
            self.process_running.store(false, Ordering::SeqCst);
            self.set_process_end_state(ProcessEndState::FileLockFailure);
            self.end_monitor(ProcessEndState::FileLockFailure);
            self.set_running(false);
        }
    }

    /// Java `run` (virtual).
    pub fn run(&self) {
        self.subclass.run(self);
    }

    /// Java `run` (the class's body).
    pub fn run_super(&self) {
        self.set_running(true);
        'try_block: {
            // `mediator.register(this)`, on the event dispatch thread.
            if let (Some(mediator), Some(this)) =
                (self.mediator.clone(), self.parallel_this.upgrade())
            {
                event_queue::invoke_and_wait(move || {
                    mediator
                        .get()
                        .register_parallel_process_monitor(Some(&this))
                });
            }
            self.messages.lock().unwrap().set_multi_parse(true);
            if self.parallel_progress_display.lock().unwrap().is_none() {
                self.load_parallel_progress_display();
            }
            if self.reconnect.load(Ordering::SeqCst) {
                let display = self.parallel_progress_display.lock().unwrap().clone();
                self.update_parallel_progress_display(display.as_ref());
            }
            if let Some(display) = self.parallel_progress_display.lock().unwrap().clone() {
                event_queue::invoke_later(move || {
                    display.get().msg_starting_process_on_selected_computers();
                });
            }
            self.n_chunks.lock().unwrap().set_int(0);
            self.chunks_finished.lock().unwrap().set_int(0);
            /* Wait for processchunks or prochunks to delete .cmds file before enabling the Kill
             * Process button and Pause button. The main loop uses a sleep of 2000 millisecs.
             * This change pushes the first sleep back before the command buttons are turned on.
             * The monitor starts running before processchunks starts, so its easy to send a
             * command to a file which is not being watched and will be deleted by
             * processchunks. Not allowing commands to be sent for the period of the first sleep
             * also reduces the chance of a collision on Windows - where processchunks cannot
             * delete the command pipe file (.cmds file) because it is in use. */
            let _ = monitor_tool_kit::sleep(&self.interrupted, START_SLEEP);

            // Get ready to respond to the Kill Process button and Pause button.
            self.set_use_commands_pipe(true);
            // Turn on the Kill Process button and Pause button.
            self.initialize_progress_bar();
            let result: Result<(), InterruptedException> = 'inner_try: {
                let _ = monitor_tool_kit::sleep(&self.interrupted, INIT_SLEEP);
                while self.is_running()
                    && self.process_running.load(Ordering::SeqCst)
                    && !self.stop.load(Ordering::SeqCst)
                    && !self.halt.load(Ordering::SeqCst)
                {
                    match self.update_state() {
                        Ok(update) => {
                            if update || self.set_progress_bar_title.load(Ordering::SeqCst) {
                                self.update_progress_bar();
                            }
                            if let Err(e) = monitor_tool_kit::sleep(&self.interrupted, UPDATE_SLEEP)
                            {
                                break 'inner_try Err(e);
                            }
                        }
                        Err(UpdateStateError::LogFile(e)) => {
                            // File creation may be slow, so give this more tries.
                            eprintln!("{e}");
                        }
                        Err(e @ UpdateStateError::IllegalState(_)) => {
                            // Fixed in translation: the unchecked
                            // IllegalStateException `updateState` throws for a
                            // bad command (ProcesschunksProcessMonitor.java:559)
                            // ends the Java monitor thread after the `finally`
                            // block.  The translation reports it and runs the
                            // same `finally` block.
                            eprintln!("{e}");
                            break 'try_block;
                        }
                    }
                }
                Ok(())
            };
            if result.is_err() {
                self.end_monitor(ProcessEndState::Done);
            }
            // Disable the use of the commands pipe.
        }
        // finally
        self.set_use_commands_pipe(false);
        if self.parallel_progress_display.lock().unwrap().is_none() {
            self.load_parallel_progress_display();
        }
        if let Some(display) = self.parallel_progress_display.lock().unwrap().clone() {
            event_queue::invoke_later(move || {
                display.get().msg_ending_process();
            });
        }
        self.messages.lock().unwrap().end_parse();
        // `mediator.deregister(this)`, on the event dispatch thread.
        if let (Some(mediator), Some(this)) = (self.mediator.clone(), self.parallel_this.upgrade())
        {
            event_queue::invoke_and_wait(move || {
                mediator
                    .get()
                    .deregister_parallel_process_monitor(Some(&this))
            });
        }
        self.busy_status_mediator.msg_monitor_stopped(self.axis_id);
        self.close_process_output();
        self.set_running(false);
    }

    /// Java `updateParallelProgressDisplay` (virtual).
    pub fn update_parallel_progress_display(&self, display: Option<&ParallelProgressDisplayRef>) {
        self.subclass
            .update_parallel_progress_display(self, display);
    }

    /// Java `updateParallelProgressDisplay` (the class's body).
    pub fn update_parallel_progress_display_super(
        &self,
        display: Option<&ParallelProgressDisplayRef>,
    ) {
        if let Some(display) = display {
            let display = display.clone();
            let computer_map = self.computer_map.clone();
            event_queue::invoke_later(move || {
                // `display.setComputerMap(computerMap)`; the Swing side takes the
                // map by reference, and a null map is an empty one there.
                display.get().set_computer_map(
                    computer_map
                        .map(|map| map.into_iter().collect::<std::collections::HashMap<_, _>>())
                        .as_ref(),
                );
            });
        }
    }

    /// Java `processMessage` (virtual).
    pub fn process_message(&self, line: &str) {
        self.subclass.process_message(self, line);
    }

    /// Java final `endMonitor`.
    pub fn end_monitor(&self, end_state: ProcessEndState) {
        self.set_process_end_state(end_state);
        self.process_running.store(false, Ordering::SeqCst); // the only place that this should be changed
        self.set_progress_bar_title();
    }

    /// Java final `getPid`.
    pub fn get_pid(&self) -> Option<String> {
        self.pid.lock().unwrap().clone()
    }

    /// Java final `getLogFileName`.
    pub fn get_log_file_name(&self) -> String {
        match self.get_process_output_file_name() {
            Ok(name) => name.unwrap_or_else(|| "null".to_string()),
            Err(e) => {
                eprintln!("{e}");
                String::new()
            }
        }
    }

    /// Java final `getProcessOutputFileName`.
    pub fn get_process_output_file_name(&self) -> Result<Option<String>, LogFileError> {
        self.create_process_output()?;
        Ok(self
            .process_output
            .lock()
            .unwrap()
            .as_ref()
            .map(|process_output| process_output.get_name()))
    }

    /// Java `synchronized final setProcessEndState`.
    pub fn set_process_end_state(&self, end_state: ProcessEndState) {
        let mut this_end_state = self.end_state.lock().unwrap();
        *this_end_state = Some(match *this_end_state {
            None => end_state,
            Some(existing) => ProcessEndState::precedence(existing, end_state),
        });
    }

    /// Java `synchronized final getProcessEndState`.
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        *self.end_state.lock().unwrap()
    }

    /// Java final `getProcessMessages`.
    pub fn get_process_messages(&self) -> MutexGuard<'_, ProcessMessages> {
        self.messages.lock().unwrap()
    }

    /// Java final `getSubProcessName`.
    pub fn get_sub_process_name(&self) -> Option<String> {
        self.root_name.clone()
    }

    /// Java `writeKillCommand` (virtual).
    pub fn write_kill_command(&self) -> Result<(), LogFileError> {
        self.subclass.write_kill_command(self)
    }

    /// Java `writeKillCommand` (the class's body).
    pub fn write_kill_command_super(&self) -> Result<(), LogFileError> {
        self.write_command("Q")
    }

    /// Java `isKilling` (virtual).
    pub fn is_killing(&self) -> bool {
        self.subclass.is_killing(self)
    }

    /// Java `isKilling` (the class's body).
    pub fn is_killing_super(&self) -> bool {
        self.killing.load(Ordering::SeqCst)
    }

    /// Java final `setKilling`.
    pub fn set_killing(&self, killing: bool) {
        self.killing.store(killing, Ordering::SeqCst);
    }

    /// Java final `kill`.
    pub fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) {
        let result: Result<(), LogFileError> = (|| {
            self.write_kill_command()?;
            if self.parallel_progress_display.lock().unwrap().is_none() {
                self.load_parallel_progress_display();
            }
            if let Some(display) = self.parallel_progress_display.lock().unwrap().clone() {
                event_queue::invoke_later(move || {
                    display.get().msg_killing_process();
                });
            }
            self.set_killing(true);
            self.set_progress_bar_title.store(true, Ordering::SeqCst);
            if self.starting.load(Ordering::SeqCst) {
                // wait to see if processchunks is already starting chunks.
                // (`Thread.sleep` on the calling thread, not the monitor's.)
                std::thread::sleep(std::time::Duration::from_millis(KILL_SLEEP));
                if self.starting.load(Ordering::SeqCst) {
                    // processchunks hasn't started chunks and it won't because the "Q" has
                    // been sent. So it is safe to kill it in the usual way.
                    process.signal_kill(axis_id);
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{e}");
        }
    }

    /// Java `pause` (virtual).
    pub fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        self.subclass.pause(self, process, axis_id)
    }

    /// Java `pause` (the class's body).
    pub fn pause_super(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        let _ = (process, axis_id);
        match self.write_command("P") {
            Ok(()) => {
                if self.parallel_progress_display.lock().unwrap().is_none() {
                    self.load_parallel_progress_display();
                }
                if let Some(display) = self.parallel_progress_display.lock().unwrap().clone() {
                    event_queue::invoke_later(move || {
                        display.get().msg_pausing_process();
                    });
                }
                self.pausing.store(true, Ordering::SeqCst);
                self.set_progress_bar_title.store(true, Ordering::SeqCst);
                true
            }
            Err(e) => {
                eprintln!("{e}");
                false
            }
        }
    }

    /// Java `isPausing` (virtual).
    pub fn is_pausing(&self) -> bool {
        self.subclass.is_pausing(self)
    }

    /// Java `isPausing` (the class's body).
    pub fn is_pausing_super(&self) -> bool {
        self.pausing.load(Ordering::SeqCst) && self.process_running.load(Ordering::SeqCst)
    }

    /// Java `setWillResume` (virtual).
    pub fn set_will_resume(&self) {
        self.subclass.set_will_resume(self);
    }

    /// Java `setWillResume` (the class's body).
    pub fn set_will_resume_super(&self) {
        self.will_resume.store(true, Ordering::SeqCst);
        self.set_progress_bar_title();
    }

    /// Java final `getStatusString`.
    pub fn get_status_string(&self) -> String {
        self.chunks_finished.lock().unwrap().to_string()
            + " of "
            + &self.n_chunks.lock().unwrap().to_string()
            + " completed"
    }

    /// Java final `drop`.
    pub fn drop_computer(&self, computer: &str) {
        if self
            .computer_map
            .as_ref()
            .is_none_or(|computer_map| computer_map.contains_key(computer))
        {
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                eprintln!("try to drop {computer}");
            }
            match self.write_command(&("D ".to_string() + computer)) {
                Ok(()) => self.set_progress_bar_title.store(true, Ordering::SeqCst),
                Err(e) => eprintln!("{e}"),
            }
        }
    }

    /// Java final `isProcessRunning`.
    pub fn is_process_running(&self) -> bool {
        if !self.process_running.load(Ordering::SeqCst) {
            return false;
        }
        let process = self.process.lock().unwrap().clone();
        if let Some(process) = process {
            if process.is_done() {
                self.process_running.store(false, Ordering::SeqCst);
            } else {
                let debug = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
                let array = process.get_std_error();
                if let Some(array) = array {
                    for line in &array {
                        if debug {
                            eprintln!("{line}");
                        }
                        if line.starts_with("ERROR:")
                            || line.starts_with("Traceback")
                            || line.contains("Errno")
                        {
                            self.end_monitor(ProcessEndState::Failed);
                            self.process_running.store(false, Ordering::SeqCst);
                        }
                    }
                }
                let array = process.get_std_output();
                if let Some(array) = array {
                    for line in &array {
                        if debug {
                            eprintln!("{line}");
                        }
                        if line.starts_with("ERROR:")
                            || line.starts_with("Traceback")
                            || line.contains("Errno")
                        {
                            self.end_monitor(ProcessEndState::Failed);
                            self.process_running.store(false, Ordering::SeqCst);
                        }
                    }
                }
            }
        }
        self.process_running.load(Ordering::SeqCst)
    }

    /// Java final `setSubdirName(String)`.
    pub fn set_subdir_name(&self, input: Option<&str>) {
        *self.subdir_name.lock().unwrap() = input.map(str::to_string);
    }

    /// Java final `setSubdirName(ConstStringProperty)`.
    pub fn set_subdir_name_property(&self, input: Option<&dyn ConstStringProperty>) {
        *self.subdir_name.lock().unwrap() = match input {
            Some(input) if !input.is_empty() => Some(input.to_string()),
            _ => None,
        };
    }

    /// Java final `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java final `isStarting`.
    pub fn is_starting(&self) -> bool {
        self.starting.load(Ordering::SeqCst)
    }

    /// Java final `isFinishing`.
    pub fn is_finishing(&self) -> bool {
        self.finishing.load(Ordering::SeqCst)
    }

    /// Java `synchronized closeProcessOutput` (virtual).
    pub fn close_process_output(&self) {
        self.subclass.close_process_output(self);
    }

    /// Java `synchronized closeProcessOutput` (the class's body).
    pub fn close_process_output_super(&self) {
        let _synchronized = self.process_output_lock.lock().unwrap();
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `if (messageReporter != null) messageReporter.close()`.
        let _ = &self.message_reporter;
        let process_output = self.process_output.lock().unwrap().clone();
        let reader_id = self.process_output_reader_id.lock().unwrap().clone();
        if let (Some(process_output), Some(reader_id)) = (process_output, reader_id) {
            if !reader_id.is_empty() {
                process_output.close_id(Some(&*reader_id));
            }
        }
        // Not nulling out anything because many functions call the create function.
    }

    /// Java `loadParallelProgressDisplay` (virtual).
    pub fn load_parallel_progress_display(&self) {
        self.subclass.load_parallel_progress_display(self);
    }

    /// Java `loadParallelProgressDisplay` (the class's body).
    pub fn load_parallel_progress_display_super(&self) {
        // `manager.getProcessingMethodMediator(axisID).getParallelProgressDisplay()`.
        // The mediator and its display are Swing-side objects, reached on the event
        // dispatch thread; the monitor keeps a cross-thread reference to the display.
        let manager = self.manager;
        let axis_id = self.axis_id;
        *self.parallel_progress_display.lock().unwrap() =
            crate::imod::etomo::util::event_queue::invoke_and_wait(move || {
                manager
                    .get_processing_method_mediator(Some(axis_id))
                    .and_then(|mediator| mediator.get_parallel_progress_display())
                    .map(|display| Arc::new(EdtRef::new(display)))
            });
    }

    /// Java `handlePauseMessage` (virtual).
    pub fn handle_pause_message(&self) {
        self.subclass.handle_pause_message(self);
    }

    /// Java `handlePauseMessage` (the class's body).
    pub fn handle_pause_message_super(&self) {
        self.end_monitor(ProcessEndState::Paused);
    }

    /// Java `updateState` (virtual).
    pub fn update_state(&self) -> Result<bool, UpdateStateError> {
        self.subclass.update_state(self)
    }

    /// Java `updateState` (the class's body).
    pub fn update_state_super(&self) -> Result<bool, UpdateStateError> {
        if self.stop.load(Ordering::SeqCst) {
            return Ok(false);
        }
        self.create_process_output()?;
        let mut return_value = false;
        let mut failed = false;
        let process_output = self.process_output.lock().unwrap().clone();
        if self.is_process_running()
            && self
                .process_output_reader_id
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|reader_id| reader_id.is_empty())
        {
            // `processOutput` was set by `createProcessOutput`.
            match process_output.as_ref().unwrap().open_reader() {
                Ok(reader_id) => *self.process_output_reader_id.lock().unwrap() = reader_id,
                Err(LogFileError::Lock(e)) => {
                    self.handle_lock_exception(Some(&e), true);
                    if !self.is_running() {
                        return Ok(false);
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        let reader_id = match self.process_output_reader_id.lock().unwrap().clone() {
            Some(reader_id) if !reader_id.is_empty() => reader_id,
            _ => return Ok(return_value),
        };
        let process_output = process_output.unwrap();
        while self.is_running() {
            let line = match process_output.read_line(&reader_id)? {
                None => break,
                Some(line) => line,
            };
            let line = java_lang_string_trim(&line).to_string();
            // get the first pid
            {
                let mut pid = self.pid.lock().unwrap();
                if pid.is_none() && line.starts_with("Shell PID:") {
                    let array = utilities::java_lang_string_split(&line, &WHITESPACE);
                    if array.len() == 3 {
                        *pid = Some(java_lang_string_trim(&array[2]).to_string());
                    }
                }
            }
            if DEBUG_LEVEL
                .lock()
                .unwrap()
                .is_some_and(|debug_level| debug_level.is_on())
            {
                eprintln!("{line}");
            }
            self.process_message(&line);
            if !line.contains("imodkillgroup") {
                self.messages.lock().unwrap().add_process_output(&line);
            }
            if !self.messages.lock().unwrap().is_empty(MessageType::Error) {
                // Set failure boolean but continue to add all the output lines to
                // messages.
                failed = true;
            }
            // If it got an error message, then it seems like the best thing to do is
            // stop processing.
            if failed {
                continue;
            }
            if line.ends_with("to reassemble") {
                // handle all chunks finished, starting the reassemble
                utilities::timestamp_process_container_status(
                    Some(TITLE),
                    self.root_name.as_deref(),
                    Some("reassembling"),
                );
                // all chunks finished, turn off pause
                {
                    let mut end_state = self.end_state.lock().unwrap();
                    if *end_state == Some(ProcessEndState::Paused) {
                        *end_state = None;
                    }
                }
                self.reassembling.store(true, Ordering::SeqCst);
                self.set_progress_bar_title.store(true, Ordering::SeqCst);
                return_value = true;
            } else if line.contains("BAD COMMAND IGNORED") {
                return Err(UpdateStateError::IllegalState(
                    "Bad command sent to processchunks\n".to_string() + &line,
                ));
            } else if line == SUCCESS_TAG {
                self.end_monitor(ProcessEndState::Done);
            } else if line == "When you rerun with a different set of machines, be sure to use" {
                self.end_monitor(ProcessEndState::Killed);
            } else if line == "All previously running chunks are done - exiting as requested" {
                self.handle_pause_message();
            }
            // A tcsh error can cause processchunks to exit immediately without killing
            // chunks.
            // "No match" has been removed. I may not be producing fatal errors.
            else if (line.contains("Syntax Error")
                || line.contains("Subscript error")
                || line.contains("Undefined variable")
                || line.contains("Expression Syntax")
                || line.contains("Subscript out of range")
                || line.contains("Illegal variable name")
                || line.contains("Variable syntax")
                || line.contains("Badly placed (")
                || line.contains("Badly formed number"))
                && self.process.lock().unwrap().as_ref().is_none_or(|process| {
                    process
                        .get_process_data()
                        .is_none_or(|process_data| !process_data.lock().unwrap().is_running())
                })
            {
                // If a tcsh error is found, start a countdown that progresses each time
                // updateState is run. Sometimes a tcsh error will result in an error
                // being generated and processchunks terminating normally, so give this
                // time to happen rather then popping up the scary error message as soon
                // as the tcsh error is found.
                self.tcsh_error_count_down.fetch_sub(1, Ordering::SeqCst);
            } else {
                let strings = utilities::java_lang_string_split(&line, &WHITESPACE);
                // set nChunks and chunksFinished
                if self.parallel_progress_display.lock().unwrap().is_none() {
                    self.load_parallel_progress_display();
                }
                let display = self.parallel_progress_display.lock().unwrap().clone();
                if strings.len() > 2 && line.ends_with("DONE SO FAR") {
                    self.starting.store(false, Ordering::SeqCst);
                    {
                        let mut n_chunks = self.n_chunks.lock().unwrap();
                        if !n_chunks.equals_string(Some(&strings[2])) {
                            n_chunks.set_string(Some(&strings[2]));
                            self.set_progress_bar_title.store(true, Ordering::SeqCst);
                        }
                    }
                    let mut chunks_finished = self.chunks_finished.lock().unwrap();
                    chunks_finished.set_string(Some(&strings[0]));
                    if chunks_finished
                        .equals_const_etomo_number(Some(&self.n_chunks.lock().unwrap().base))
                    {
                        self.finishing.store(true, Ordering::SeqCst);
                    }
                    return_value = true;
                } else if strings.len() > 1 {
                    if line.starts_with("Dropping") {
                        let failure_reason = if line.contains("it cannot cd to") {
                            "cd failed"
                        } else if line.contains("cannot connect") {
                            "connect failed"
                        } else if line.contains("it cannot run IMOD commands") {
                            "run error"
                        } else if line.contains("it failed (with time out)") {
                            "chunk timed out"
                        } else if line.contains("it failed (with chunk error)") {
                            "chunk error"
                        } else {
                            "unknown error"
                        };
                        // handle a dropped CPU
                        // Fixed in translation: ProcesschunksProcessMonitor.java:632
                        // calls `parallelProgressDisplay.msgDropped` without the
                        // null check the neighbouring branches make, so a monitor
                        // with no display (no mediator display) throws a
                        // NullPointerException out of the run loop.  The
                        // translation skips the message, as those branches do.
                        if let Some(display) = display {
                            let computer = strings[1].clone();
                            event_queue::invoke_later(move || {
                                display
                                    .get()
                                    .msg_dropped(Some(&computer), Some(failure_reason));
                            });
                        }
                    }
                    // handle commandsPipeWriteIda finished chunk
                    else if strings[1] == "finished" {
                        // Fixed in translation: `strings[3]` is read unchecked
                        // (ProcesschunksProcessMonitor.java:637 and :642); a short
                        // line throws an ArrayIndexOutOfBoundsException out of the
                        // run loop.  The translation skips a line too short to
                        // name the computer.
                        if let (Some(display), Some(computer)) = (display, strings.get(3)) {
                            let computer = computer.clone();
                            event_queue::invoke_later(move || {
                                display.get().add_success(Some(&computer));
                            });
                        }
                    }
                    // handle a failed chunk
                    else if strings[1] == "failed" {
                        // Fixed in translation: ProcesschunksProcessMonitor.java:642
                        // calls `parallelProgressDisplay.addRestart` without a null
                        // check (see the "Dropping" branch above).
                        if let (Some(display), Some(computer)) = (display, strings.get(3)) {
                            let computer = computer.clone();
                            event_queue::invoke_later(move || {
                                display.get().add_restart(Some(&computer));
                            });
                        }
                    }
                }
            }
        }
        if failed {
            self.end_monitor(ProcessEndState::Failed);
            return_value = false;
        } else if self.tcsh_error_count_down.load(Ordering::SeqCst) < 0 {
            // The tcsh error did not cause processchunks to terminate normally
            // before the countdown ended, so assume that it died without killing
            // chunks. Currently not ending the monitor because we don't want the
            // user to rerun processchunks before all the chunk processes end.
            eprintln!("ERROR: Tcsh error in processchunks log");
            ui_harness::post_message_dialog(
                Some(self.manager),
                "Unrecoverable error in processchunks.  Please contact the ".to_string()
                    + "programmer.  The chunk processes are still running, but "
                    + "they won't show up in the progress bar or appear in the "
                    + "Finished Chunks column.  IMPORTANT:  Let the chunk "
                    + "processes complete before rerunning the parallel process.  "
                    + "The load may decline when your chunks are done.  Also you "
                    + "can ssh to each computer and run top.  After the chunk "
                    + "processes are complete, exit Etomo to clear the parallel "
                    + "processing panel.  To attempt to continue the parallel "
                    + "process, rerun Etomo, and press Resume.",
                "Fatal Error".to_string(),
                None,
            );
            self.tcsh_error_count_down
                .store(NO_TCSH_ERROR, Ordering::SeqCst);
        } else if self.tcsh_error_count_down.load(Ordering::SeqCst) < NO_TCSH_ERROR {
            // Continue the countdown.
            self.tcsh_error_count_down.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(return_value)
    }

    /// Java `updateProgressBar` (virtual).
    pub fn update_progress_bar(&self) {
        self.subclass.update_progress_bar(self);
    }

    /// Java `updateProgressBar` (the class's body).
    pub fn update_progress_bar_super(&self) {
        if self.set_progress_bar_title.load(Ordering::SeqCst) {
            self.set_progress_bar_title.store(false, Ordering::SeqCst);
            self.set_progress_bar_title();
        }
        let value = self.chunks_finished.lock().unwrap().get_int();
        let status_string = self.get_status_string();
        let axis_id = self.axis_id;
        self.manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_value_int_string_axis_id(value, Some(&status_string), axis_id);
        }));
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `if (messageReporter != null) messageReporter.checkForMessages(manager)`.
    }

    /// Java final `initializeProgressBar`.
    pub fn initialize_progress_bar(&self) {
        self.set_progress_bar_title();
        self.tool_kit.initialize_progress_bar(" ", false);
    }

    /// Java `useMessageReporter` (virtual).
    pub fn use_message_reporter(&self) {
        self.subclass.use_message_reporter(self);
    }

    /// Java private final `setProgressBarTitle`.
    fn set_progress_bar_title(&self) {
        let mut title = String::from(TITLE);
        if let Some(root_name) = &self.root_name {
            title.push_str(&(" ".to_string() + root_name));
        }
        let reassembling = self.reassembling.load(Ordering::SeqCst);
        let killing = self.killing.load(Ordering::SeqCst);
        let pausing = self.pausing.load(Ordering::SeqCst);
        let will_resume = self.will_resume.load(Ordering::SeqCst);
        if self.process_running.load(Ordering::SeqCst) {
            if reassembling {
                title.push_str(":  reassembling");
            } else if killing {
                title.push_str(": killing - exiting current chunks");
            } else if pausing {
                title.push_str(": pausing - finishing current chunks");
                if will_resume {
                    title.push_str(" and will resume");
                }
            }
        } else if killing {
            title.push_str(": killed");
        } else if pausing {
            title.push_str(": paused");
            if will_resume {
                title.push_str(" - will resume");
            }
        }
        let n_steps = self.n_chunks.lock().unwrap().get_int();
        let axis_id = self.axis_id;
        let pause_enabled = !reassembling && !killing;
        self.manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_string_int_boolean_axis_id_boolean(
                Some(&title),
                n_steps,
                false,
                axis_id,
                pause_enabled,
            );
        }));
    }

    /// Java `getCheckFile` (virtual).
    pub fn get_check_file(&self) -> String {
        self.subclass.get_check_file(self)
    }

    /// Java `getCheckFile` (the class's body).
    pub fn get_check_file_super(&self) -> String {
        dataset_files::get_commands_file_name(
            self.subdir_name.lock().unwrap().as_deref(),
            self.root_name.as_deref().unwrap_or("null"),
            None,
        )
    }

    /// Java final `getSubdirName`.
    pub fn get_subdir_name(&self) -> Option<String> {
        self.subdir_name.lock().unwrap().clone()
    }

    /// Java final `getRootName`.
    pub fn get_root_name(&self) -> Option<String> {
        self.root_name.clone()
    }

    /// Java final `writeCommand`.  Create commandsWriter if necessary, write
    /// command, add newline, flush.
    pub fn write_command(&self, command: &str) -> Result<(), LogFileError> {
        if !self.use_commands_pipe.load(Ordering::SeqCst) {
            return Ok(());
        }
        let commands_pipe = {
            let mut commands_pipe = self.commands_pipe.lock().unwrap();
            if commands_pipe.is_none() {
                *commands_pipe = Some(LogFile::get_instance_user_dir(
                    &self.manager.get_property_user_dir().unwrap_or_default(),
                    &self.get_check_file(),
                    Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
                )?);
            }
            commands_pipe.clone().unwrap()
        };
        if self
            .commands_pipe_writer_id
            .lock()
            .unwrap()
            .as_ref()
            .is_none_or(|writer_id| writer_id.is_empty())
        {
            match commands_pipe.open_writer_append(true) {
                Ok(writer_id) => *self.commands_pipe_writer_id.lock().unwrap() = Some(writer_id),
                Err(LogFileError::Lock(e)) => {
                    self.handle_lock_exception(Some(&e), true);
                    if !self.is_running() {
                        return Ok(());
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let writer_id = match self.commands_pipe_writer_id.lock().unwrap().clone() {
            Some(writer_id) if !writer_id.is_empty() => writer_id,
            _ => return Ok(()),
        };
        commands_pipe.write(Some(command), &writer_id)?;
        commands_pipe.new_line(&writer_id)?;
        commands_pipe.flush(&writer_id)?;
        // Close writer after each write. If it is kept open, the file would not be writeable
        // from the command line in Windows.
        commands_pipe.close_id(Some(&*writer_id));
        *self.commands_pipe_writer_id.lock().unwrap() = None;
        Ok(())
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> ProcessName {
        ProcessName::PROCESSCHUNKS
    }

    /// Java private final synchronized `createProcessOutput`.  Make sure
    /// process output file is new and set processOutputFile.  This function
    /// should be first run before the process starts.
    fn create_process_output(&self) -> Result<(), LogFileError> {
        let _synchronized = self.process_output_lock.lock().unwrap();
        if self.process_output.lock().unwrap().is_some() {
            return Ok(());
        }
        let process_output = LogFile::get_instance_user_dir(
            &self.manager.get_property_user_dir().unwrap_or_default(),
            &dataset_files::get_out_file_name(
                self.manager,
                self.subdir_name.lock().unwrap().as_deref(),
                Some(&ProcessName::PROCESSCHUNKS.to_string()),
                Some(self.axis_id),
            ),
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        *self.process_output.lock().unwrap() = Some(process_output.clone());
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `if (messageReporter != null) messageReporter.close();` then
        // `messageReporter = new MessageReporter(axisID, processOutput)`:
        // processchunks needs to always report lines tagged with MESSAGE.
        let _ = &self.message_reporter;
        // Attempt to clean up previous reads. This will only work if somthing is wrong.
        if let Some(reader_id) = self.process_output_reader_id.lock().unwrap().take() {
            process_output.close_id(Some(&*reader_id));
        }
        // Don't remove the file if this is a reconnect.
        if !self.reconnect.load(Ordering::SeqCst) {
            // Avoid looking at a file from a previous run.
            self.tool_kit
                .msg_log_file_renaming_handle(Some(&process_output), false);
            let mut succeeded = false;
            match process_output.backup_once_per_handle() {
                Ok(backed_up) => succeeded = backed_up,
                Err(LogFileError::Lock(e)) => {
                    self.handle_lock_exception(Some(&e), true);
                    if !self.is_running() {
                        return Ok(());
                    }
                }
                Err(e) => return Err(e),
            }
            if !succeeded {
                self.tool_kit.msg_log_file_renaming_failed();
            } else {
                self.tool_kit.msg_log_file_renamed(false);
            }
        }
        Ok(())
    }

    /// Java private synchronized `setRunning`.
    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }

    /// Java `isRunning` (virtual).
    pub fn is_running(&self) -> bool {
        self.subclass.is_running(self)
    }

    /// Java `isRunning` (the class's body).
    pub fn is_running_super(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Java `halt` (virtual).
    pub fn halt(&self) {
        self.subclass.halt(self);
    }

    /// Java `hasProgressBarAccess` (virtual).
    pub fn has_progress_bar_access(&self) -> bool {
        self.subclass.has_progress_bar_access(self)
    }
}

impl<S: ProcesschunksProcessMonitorImpl> Monitor for ProcesschunksProcessMonitor<S> {
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
        self.interrupted.store(true, Ordering::SeqCst);
    }
}

impl<S: ProcesschunksProcessMonitorImpl> ProcessMonitor for ProcesschunksProcessMonitor<S> {
    fn set_process_end_state(&self, end_state: ProcessEndState) {
        self.erased().set_process_end_state(end_state);
    }

    fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) {
        self.erased().kill(process, axis_id);
    }

    fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        self.erased().pause(process, axis_id)
    }

    fn get_status_string(&self) -> Option<String> {
        Some(self.erased().get_status_string())
    }

    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        Some(self.messages.lock().unwrap())
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

    /// Java final `msgLogFileRenaming(LogFile.Handle, boolean)` and its
    /// one-argument overload.
    fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        self.tool_kit
            .msg_log_file_renaming_handle(log_file, indeterminate_mode);
    }

    /// Java final `msgLogFileRenaming(String)`.
    fn msg_log_file_renaming_name(&self, file_name: &str) {
        self.tool_kit.msg_log_file_renaming_name(file_name);
    }

    /// Java final `msgLogFileRenaming(File)`.
    fn msg_log_file_renaming_file(&self, file: &Path) {
        self.tool_kit.msg_log_file_renaming_file(file);
    }

    /// Java final `msgLogFileRenamed`.
    fn msg_log_file_renamed(&self) {
        self.tool_kit.msg_log_file_renamed(false);
    }

    /// Java final `msgLogFileRenamingFailed`.
    fn msg_log_file_renaming_failed(&self) {
        self.tool_kit.msg_log_file_renaming_failed();
    }
}

impl<S: ProcesschunksProcessMonitorImpl> DetachedProcessMonitor for ProcesschunksProcessMonitor<S> {
    fn is_process_running(&self) -> bool {
        self.erased().is_process_running()
    }

    fn get_process_output_file_name(&self) -> Result<String, LogFileError> {
        self.erased()
            .get_process_output_file_name()
            .map(|name| name.unwrap_or_else(|| "null".to_string()))
    }

    fn set_process(&self, process: Arc<dyn SystemProcessInterface>) {
        self.erased().set_process(Some(process));
    }
}

impl<S: ProcesschunksProcessMonitorImpl> OutfileProcessMonitor for ProcesschunksProcessMonitor<S> {
    fn get_pid(&self) -> Option<String> {
        self.erased().get_pid()
    }

    fn end_monitor(&self, end_state: ProcessEndState) {
        self.erased().end_monitor(end_state);
    }

    fn get_sub_process_name(&self) -> Option<String> {
        self.erased().get_sub_process_name()
    }
}

impl<S: ProcesschunksProcessMonitorImpl> ParallelProcessMonitor for ProcesschunksProcessMonitor<S> {
    /// Java final `drop`.
    fn drop(&self, computer: &str) {
        self.erased().drop_computer(computer);
    }
}
