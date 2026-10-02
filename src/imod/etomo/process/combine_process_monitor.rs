//! `IMOD/Etomo/src/etomo/process/CombineProcessMonitor.java`.
//!
//! Description: Provides a threadable class to execute IMOD com scripts in the
//! background.  An instance of this class can be run only once.
//!
//! combine.com runs a series of com files; this monitor follows combine.log and
//! runs the `LogFileProcessMonitor` of each monitored child (matchvol1,
//! patchcorr, matchorwarp, volcombine) on its own thread.  A child of any of
//! those classes is held as
//! `Arc<LogFileProcessMonitorOf<dyn LogFileProcessMonitorImpl>>`; Java's
//! `childThread.interrupt()` is the child's `Monitor::interrupt`, and Java's
//! `runThread.interrupt()` (this monitor's own thread) is this monitor's.
//!
//! The process result displays are Swing buttons, so they are reached on the
//! event dispatch thread (`util/event_queue.rs`).

use super::emergency_monitor::EmergencyMonitor;
use super::log_file_process_monitor::{LogFileProcessMonitorImpl, LogFileProcessMonitorOf};
use super::matchorwarp_process_monitor::MatchorwarpProcessMonitor;
use super::matchvol1_process_monitor::Matchvol1ProcessMonitor;
use super::monitor::{DetachedProcessMonitor, Monitor, ProcessMonitor};
use super::monitor_tool_kit::{self, MonitorToolKit};
use super::patchcorr_process_watcher::PatchcorrProcessWatcher;
use super::process_interface::{ProcessResultDisplayRef, SystemProcessInterface};
use super::process_messages::ProcessMessages;
use super::volcombine_process_monitor::VolcombineProcessMonitor;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::combine_comscript_state::{self, CombineComscriptState};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::storage::log_file::{
    Handle, LockException, LogFile, LogFileError, ReaderId, UnlockedException, WritingId,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::combine_process_type::CombineProcessType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;

/// Java `COMBINE_LABEL`.
pub const COMBINE_LABEL: &str = "Combine";
/// Java `SLEEP`.
const SLEEP: u64 = 100;
/// Java `CONSTRUCTED_STATE`.
const CONSTRUCTED_STATE: i32 = 1;
/// Java `WAITED_FOR_LOG_STATE`.
const WAITED_FOR_LOG_STATE: i32 = 2;
/// Java `RAN_STATE`.
const RAN_STATE: i32 = 3;

/// A child monitor of any `LogFileProcessMonitor` subclass.
type ChildMonitor = Arc<LogFileProcessMonitorOf<dyn LogFileProcessMonitorImpl>>;

/// Java `CombineProcessMonitor`.
pub struct CombineProcessMonitor {
    /// Java field `emergencyMonitor`.
    emergency_monitor: Arc<EmergencyMonitor>,
    /// Java field `busyStatusMediator`.
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java field `selfTest`.
    self_test: bool,
    /// Java field `combineComscriptState`.
    combine_comscript_state: Arc<CombineComscriptState>,
    // Java field `displayFactory` (`manager.getProcessResultDisplayFactory(axisID)`).
    // TODO(unit): held once the faithful ProcessResultDisplayFactory is
    // integrated; meanwhile the restart displays below are null.
    /// Java field `manager`.
    manager: &'static ApplicationManager,
    /// Java field `axisID`.
    axis_id: AxisID,
    /// Java field `toolKit`.
    tool_kit: MonitorToolKit,

    /// Java field `logFileReaderId`.
    log_file_reader_id: Mutex<Option<ReaderId>>,
    /// Java field `sleepCount`.
    sleep_count: AtomicI32,
    /// Java field `endState`; the mutex stands for the `synchronized` of
    /// `setProcessEndState`/`getProcessEndState`.
    end_state: Mutex<Option<ProcessEndState>>,

    /// Java field `processRunning`.  If processRunning is false at any time
    /// before the process ends, it can cause wait loops to end prematurely.
    /// This is because the wait loop can start very repidly for a background
    /// process.  See BackgroundSystemProgram.waitForProcess().
    process_running: AtomicBool,

    /// Java field `logFile`.
    log_file: Mutex<Option<Arc<Handle>>>,
    /// Java field `childMonitor`.
    child_monitor: Mutex<Option<ChildMonitor>>,
    /// Java field `childThread`.
    child_thread: Mutex<Option<JoinHandle<()>>>,
    /// Java field `currentCommand`.
    current_command: Mutex<Option<ProcessName>>,

    /// Java field `runThread`: true while it holds this monitor's thread.
    run_thread: AtomicBool,
    /// Java field `processResultDisplay`.
    process_result_display: Mutex<Option<ProcessResultDisplayRef>>,
    /// Java field `process`.
    process: Mutex<Option<Arc<dyn SystemProcessInterface>>>,

    /// Java field `firstChildProcessSet`.
    first_child_process_set: AtomicBool,
    /// Java field `childLog`.
    child_log: Mutex<Option<Arc<Handle>>>,
    /// Java field `childLogWritingId`.
    child_log_writing_id: Mutex<Option<WritingId>>,
    /// Java field `stop`.
    stop: AtomicBool,
    /// Java `volatile` field `running`.
    running: AtomicBool,
    /// Java field `lockException` (never assigned in the source).
    lock_exception: Mutex<Option<LockException>>,

    /// The instance's intrinsic lock, taken by the `synchronized` method
    /// `handleLockException`.
    synchronized: Mutex<()>,
    /// `Thread.interrupt()` on this monitor's thread.
    interrupted: AtomicBool,
}

impl CombineProcessMonitor {
    /// Java `CombineProcessMonitor(ApplicationManager, AxisID,
    /// CombineComscriptState, ProcessResultDisplay)`.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        combine_comscript_state: Arc<CombineComscriptState>,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Arc<CombineProcessMonitor> {
        let monitor = Arc::new_cyclic(|this: &Weak<CombineProcessMonitor>| {
            let base_manager: &'static dyn BaseManager = manager;
            let emergency_monitor = base_manager.get_emergency_monitor(Some(axis_id));
            let weak_monitor: Weak<dyn Monitor> = this.clone();
            let tool_kit = MonitorToolKit::new(manager, axis_id, Some(weak_monitor));
            let busy_status_mediator = manager.get_busy_status_mediator();
            busy_status_mediator.msg_monitor_constructed(axis_id);
            let self_test = etomo_director::ARGUMENTS.lock().unwrap().is_self_test();
            CombineProcessMonitor {
                emergency_monitor,
                busy_status_mediator,
                self_test,
                combine_comscript_state,
                manager,
                axis_id,
                tool_kit,
                log_file_reader_id: Mutex::new(None),
                sleep_count: AtomicI32::new(0),
                end_state: Mutex::new(None),
                process_running: AtomicBool::new(true),
                log_file: Mutex::new(None),
                child_monitor: Mutex::new(None),
                child_thread: Mutex::new(None),
                current_command: Mutex::new(None),
                run_thread: AtomicBool::new(false),
                process_result_display: Mutex::new(process_result_display),
                process: Mutex::new(None),
                first_child_process_set: AtomicBool::new(false),
                child_log: Mutex::new(None),
                child_log_writing_id: Mutex::new(None),
                stop: AtomicBool::new(false),
                running: AtomicBool::new(false),
                lock_exception: Mutex::new(None),
                synchronized: Mutex::new(()),
                interrupted: AtomicBool::new(false),
            }
        });
        monitor.run_self_test(CONSTRUCTED_STATE);
        monitor
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> ProcessName {
        ProcessName::COMBINE
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `initializeProgressBar`.
    pub fn initialize_progress_bar(&self) {
        self.manager.start_progress_bar(
            Some(COMBINE_LABEL),
            Some(self.axis_id),
            self.combine_comscript_state.get_initial_process_name(),
        );
        self.tool_kit.initialize_progress_bar(COMBINE_LABEL, false);
    }

    /// Java private `getCurrentSection`.  Get each .com file run by
    /// combine.com.
    fn get_current_section(&self) -> Result<(), LogFileError> {
        let result: Result<(), LogFileError> = 'try_block: {
            while self.is_running() {
                let line = match self.read_log_file_line() {
                    Ok(None) => break,
                    Ok(Some(line)) => line,
                    Err(e) => break 'try_block Err(e),
                };
                if line.contains("running ") || line.contains("Running ") {
                    let child_command_name = self
                        .combine_comscript_state
                        .get_matching_command(Some(&line));
                    if let Some(child_command_name) = child_command_name {
                        if let Err(e) = self.set_current_child_command(&child_command_name) {
                            break 'try_block Err(e);
                        }
                        self.run_current_child_monitor();
                    }
                } else if line.starts_with("ERROR:") || line.starts_with("Traceback") {
                    self.set_process_result_display_on_process();
                    self.end_monitor(ProcessEndState::Failed);
                } else if line.starts_with(&CombineComscriptState::get_success_text()) {
                    self.set_process_result_display_on_process();
                    self.end_monitor(ProcessEndState::Done);
                }
            }
            Ok(())
        };
        match result {
            Err(LogFileError::Lock(e)) => {
                self.handle_lock_exception(Some(&e), true);
                Ok(())
            }
            result => result,
        }
    }

    /// `logFile.readLine(logFileReaderId)`: an unopened (null) id is not
    /// locked, so `LogFile.readLine` throws an `UnlockedException`.
    fn read_log_file_line(&self) -> Result<Option<String>, LogFileError> {
        let log_file = self.log_file.lock().unwrap().clone();
        let reader_id = self.log_file_reader_id.lock().unwrap().clone();
        match (log_file, reader_id) {
            (Some(log_file), Some(reader_id)) => log_file.read_line(&reader_id),
            _ => Err(LogFileError::Unlocked(UnlockedException::new_id(
                None, None,
            ))),
        }
    }

    /// `process.setProcessResultDisplay(processResultDisplay)`.
    fn set_process_result_display_on_process(&self) {
        let process_result_display = self.process_result_display.lock().unwrap().clone();
        // Fixed in translation: CombineProcessMonitor.java:174 and :178 call
        // `process.setProcessResultDisplay` on a null `process` when the monitor
        // runs before `setProcess`; the NullPointerException ends the monitor
        // thread.  The translation skips the call.
        if let Some(process) = &*self.process.lock().unwrap() {
            process.set_process_result_display(process_result_display);
        }
    }

    /// Java private `setCurrentChildCommand`.  Get current .com file run by
    /// combine.com; run the monitor associated with the current .com file, if
    /// there is one.
    fn set_current_child_command(&self, child_command_name: &str) -> Result<(), LogFileError> {
        let next_process_result_display: Option<ProcessResultDisplayRef>;
        {
            let mut child_log_writing_id = self.child_log_writing_id.lock().unwrap();
            let child_log = self.child_log.lock().unwrap();
            if let (Some(child_log), Some(writing_id)) = (&*child_log, &*child_log_writing_id) {
                if !writing_id.is_empty() {
                    child_log.close_id(Some(&**writing_id));
                    *child_log_writing_id = None;
                }
            }
        }
        self.manager
            .progress_bar_done(Some(self.axis_id), Some(ProcessEndState::Done));
        let current_command = ProcessName::get_instance_with_axis(child_command_name, self.axis_id);
        *self.current_command.lock().unwrap() = current_command;
        if let Some(current_command) = current_command {
            let base_manager: &'static dyn BaseManager = self.manager;
            let child_log = LogFile::get_instance_process_name(
                &self.manager.get_property_user_dir().unwrap_or_default(),
                self.axis_id,
                current_command,
                Some(base_manager.get_emergency_monitor(Some(self.axis_id))),
            )?;
            *self.child_log.lock().unwrap() = Some(child_log.clone());
            *self.child_log_writing_id.lock().unwrap() = Some(child_log.open_for_writing()?);
        }
        if current_command == Some(ProcessName::MATCHVOL1) {
            next_process_result_display = None /* TODO(unit): displayFactory.get_restart_matchvol1() */;
            self.set_next_process_result_display(next_process_result_display.clone());
            self.manager.show_pane(
                combine_comscript_state::COMSCRIPT_NAME,
                CombineProcessType::MATCHVOL1,
            );
            let child: ChildMonitor = Matchvol1ProcessMonitor::new(
                self.manager,
                self.axis_id,
                next_process_result_display,
            );
            *self.child_monitor.lock().unwrap() = Some(child);
        } else if current_command == Some(ProcessName::PATCHCORR) {
            next_process_result_display = None /* TODO(unit): displayFactory.get_restart_patchcorr() */;
            self.set_next_process_result_display(next_process_result_display.clone());
            self.manager.show_pane(
                combine_comscript_state::COMSCRIPT_NAME,
                CombineProcessType::PATCHCORR,
            );
            let child: ChildMonitor = PatchcorrProcessWatcher::new(
                self.manager,
                self.axis_id,
                next_process_result_display,
            );
            *self.child_monitor.lock().unwrap() = Some(child);
        } else if current_command == Some(ProcessName::MATCHORWARP) {
            next_process_result_display = None /* TODO(unit): displayFactory.get_restart_matchorwarp() */;
            self.set_next_process_result_display(next_process_result_display.clone());
            self.manager.show_pane(
                combine_comscript_state::COMSCRIPT_NAME,
                CombineProcessType::MATCHORWARP,
            );
            let child: ChildMonitor = MatchorwarpProcessMonitor::new(
                self.manager,
                self.axis_id,
                next_process_result_display,
            );
            *self.child_monitor.lock().unwrap() = Some(child);
        } else if current_command == Some(ProcessName::VOLCOMBINE) {
            next_process_result_display = None /* TODO(unit): displayFactory.get_restart_volcombine() */;
            self.set_next_process_result_display(next_process_result_display.clone());
            self.manager.show_pane(
                combine_comscript_state::COMSCRIPT_NAME,
                CombineProcessType::VOLCOMBINE,
            );
            let child: ChildMonitor = VolcombineProcessMonitor::new(
                self.manager,
                self.axis_id,
                next_process_result_display,
            );
            *self.child_monitor.lock().unwrap() = Some(child);
        } else {
            self.start_progress_bar(child_command_name, current_command);
        }
        Ok(())
    }

    /// Java private `setNextProcessResultDisplay`.
    fn set_next_process_result_display(
        &self,
        next_process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        if !self.first_child_process_set.load(Ordering::SeqCst) {
            self.first_child_process_set.store(true, Ordering::SeqCst);
            return;
        }
        if let Some(process_result_display) = self.process_result_display.lock().unwrap().clone() {
            event_queue::invoke_later(move || {
                process_result_display.get().msg_process_succeeded();
            });
        }
        self.end_current_child_monitor(None);
        *self.process_result_display.lock().unwrap() = next_process_result_display.clone();
        if let Some(process_result_display) = next_process_result_display {
            event_queue::invoke_later(move || {
                process_result_display.get().msg_process_starting();
            });
        }
    }

    /// Java private `runCurrentChildMonitor`.  Run the monitor associated with
    /// the current .com file run by combine.com.
    fn run_current_child_monitor(&self) {
        let child_monitor = match self.child_monitor.lock().unwrap().clone() {
            None => return,
            Some(child_monitor) => child_monitor,
        };
        child_monitor.set_last_process(false);
        *self.child_thread.lock().unwrap() = Some(std::thread::spawn(move || child_monitor.run()));
    }

    /// Java private `endCurrentChildMonitor`.  Stop the current monitor
    /// associated with the current .com file run by combine.com.
    fn end_current_child_monitor(&self, cur_end_state: Option<ProcessEndState>) {
        let mut child_monitor = self.child_monitor.lock().unwrap();
        let mut child_thread = self.child_thread.lock().unwrap();
        if let Some(child_monitor) = &*child_monitor {
            child_monitor.halt_process(child_thread.is_some(), cur_end_state);
        }
        *child_monitor = None;
        *child_thread = None;
    }

    /// Java private `endMonitor`.  End this monitor.
    fn end_monitor(&self, end_state: ProcessEndState) {
        {
            let mut child_log_writing_id = self.child_log_writing_id.lock().unwrap();
            let child_log = self.child_log.lock().unwrap();
            if let (Some(child_log), Some(writing_id)) = (&*child_log, &*child_log_writing_id) {
                if !writing_id.is_empty() {
                    child_log.close_id(Some(&**writing_id));
                    *child_log_writing_id = None;
                }
            }
        }
        self.set_process_end_state(end_state);
        self.end_current_child_monitor(Some(end_state));
        self.manager
            .progress_bar_done(Some(self.axis_id), Some(end_state));
        if self.run_thread.swap(false, Ordering::SeqCst) {
            self.interrupted.store(true, Ordering::SeqCst);
        }
        self.process_running.store(false, Ordering::SeqCst); // the only place that this should be changed
    }

    /// Java private `startProgressBar`.  Start a progress bar for the current
    /// .com file run by combine.com.  Used when there is no monitor available
    /// for the child process.  Used for the process that runs before
    /// matchvol1.
    fn start_progress_bar(&self, child_command_name: &str, process_name: Option<ProcessName>) {
        let combine_process_type = match CombineProcessType::get_instance(child_command_name) {
            // must be a command that is not monitored
            None => return,
            Some(combine_process_type) => combine_process_type,
        };
        self.set_next_process_result_display(None);
        self.manager.show_pane(
            combine_comscript_state::COMSCRIPT_NAME,
            combine_process_type,
        );
        self.manager.start_progress_bar(
            Some(&(COMBINE_LABEL.to_string() + ": " + child_command_name)),
            Some(self.axis_id),
            process_name,
        );
    }

    /// Java `synchronized final handleLockException`.
    pub fn handle_lock_exception(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _synchronized = self.synchronized.lock().unwrap();
        if let Some(lock_exception) = lock_exception {
            self.emergency_monitor.alert(Some(lock_exception), do_popup);
            self.stop.store(true, Ordering::SeqCst);
            self.process_running.store(false, Ordering::SeqCst);
            self.set_process_end_state(ProcessEndState::FileLockFailure);
            self.end_monitor(ProcessEndState::FileLockFailure);
            self.set_running(false);
        }
    }

    /// The body of Java `run`'s inner `try` block.  `Ok(false)` is its early
    /// `return` (the process stopped while waiting for the log file).
    fn run_combine(&self) -> Result<bool, RunError> {
        // Instantiate the logFile object
        let base_manager: &'static dyn BaseManager = self.manager;
        let log_file = LogFile::get_instance_name(
            &self.manager.get_property_user_dir().unwrap_or_default(),
            self.axis_id,
            combine_comscript_state::COMSCRIPT_NAME,
            Some(base_manager.get_emergency_monitor(Some(self.axis_id))),
        )
        .map_err(RunError::LogFile)?;
        *self.log_file.lock().unwrap() = Some(log_file);

        // Wait for the log file to exist
        self.wait_for_log_file()?;
        if !self.process_running.load(Ordering::SeqCst) {
            self.busy_status_mediator.msg_monitor_stopped(self.axis_id);
            self.set_running(false);
            return Ok(false);
        }
        self.initialize_progress_bar();

        while self.is_running()
            && self.process_running.load(Ordering::SeqCst)
            && !self.stop.load(Ordering::SeqCst)
        {
            monitor_tool_kit::sleep(&self.interrupted, SLEEP).map_err(|_| RunError::Interrupted)?;
            self.get_current_section().map_err(RunError::LogFile)?;
        }
        Ok(true)
    }

    /// Java private `waitForLogFile`.  Wait for the process to start and the
    /// appropriate log file to be created.
    fn wait_for_log_file(&self) -> Result<(), RunError> {
        let log_file = self.log_file.lock().unwrap().clone().unwrap();
        let mut new_log_file = false;
        let debug = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        while self.is_running() && !new_log_file {
            // Check to see if the log file exists that signifies that the process
            // has started
            if log_file.exists() {
                new_log_file = true;
            } else {
                let process = self.process.lock().unwrap().clone();
                if let Some(process) = process {
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
                                return Ok(());
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
                                return Ok(());
                            }
                        }
                    }
                }
                monitor_tool_kit::sleep(&self.interrupted, SLEEP)
                    .map_err(|_| RunError::Interrupted)?;
            }
        }
        // Open the log file
        match log_file.open_reader() {
            Ok(reader_id) => *self.log_file_reader_id.lock().unwrap() = reader_id,
            Err(LogFileError::Lock(e)) => self.handle_lock_exception(Some(&e), true),
            Err(e) => return Err(RunError::LogFile(e)),
        }
        self.run_self_test(WAITED_FOR_LOG_STATE);
        Ok(())
    }

    /// Java private `runSelfTest`.  Runs selfTest(int) when selfTest is set.
    fn run_self_test(&self, state: i32) {
        if !self.self_test {
            return;
        }
        self.self_test(state);
    }

    /// Java `selfTest`.  Test for incorrect member variable settings.
    ///
    /// Fixed in translation: the source throws an unchecked
    /// NullPointerException/IllegalStateException for a failed check, which
    /// ends the constructing or monitor thread (skipping `run`'s cleanup when
    /// thrown from the log wait).  The translation reports the same exception
    /// on standard error and returns.
    pub fn self_test(&self, state: i32) {
        let state_string: &str;
        match state {
            CONSTRUCTED_STATE => {
                state_string = "After construction:  ";
                // `axisID` and `combineComscriptState` cannot be null here.
                if !self.process_running.load(Ordering::SeqCst) {
                    eprintln!(
                        "java.lang.IllegalStateException: {}ProcessRunning must be true",
                        state_string
                    );
                }
            }

            WAITED_FOR_LOG_STATE => {
                state_string = "After waitForLogFile():  ";
                let exists = self
                    .log_file
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|log_file| log_file.exists());
                let sleep_count = self.sleep_count.load(Ordering::SeqCst);
                if exists && sleep_count != 0 {
                    eprintln!(
                        "java.lang.IllegalStateException: {}The sleepCount should be reset when the log file is found.  sleepCount={}",
                        state_string, sleep_count
                    );
                }
            }

            RAN_STATE => {
                state_string = "After run():  ";
                if self.process_running.load(Ordering::SeqCst) {
                    eprintln!(
                        "java.lang.IllegalStateException: {}ProcessRunning should be false.",
                        state_string
                    );
                }
            }

            _ => {
                eprintln!(
                    "java.lang.IllegalStateException: Unknown state.  state={}",
                    state
                );
            }
        }
    }

    /// Java final `getErrorMessage`.
    pub fn get_error_message(&self) -> Option<String> {
        None
    }

    /// Java private synchronized `setRunning`.
    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }
}

/// What escapes Java `run`'s inner `try` block to its catch clauses.
enum RunError {
    Interrupted,
    LogFile(LogFileError),
}

impl Monitor for CombineProcessMonitor {
    /// Java `run`.  Get log file.  Initialize progress bar.  Loop until
    /// processRunning is turned off or there is a timeout.  Call
    /// getCurrentSection for each loop.  After loop, turn off the monitor if
    /// that hasn't been done already.
    fn run(&self) {
        self.set_running(true);
        'outer_try: {
            self.run_thread.store(true, Ordering::SeqCst);
            self.initialize_progress_bar();
            match self.run_combine() {
                Ok(true) => {}
                Ok(false) => break 'outer_try,
                Err(RunError::Interrupted) => self.end_monitor(ProcessEndState::Done),
                Err(RunError::LogFile(e @ (LogFileError::File(_) | LogFileError::Io(_)))) => {
                    self.end_monitor(ProcessEndState::Failed);
                    eprintln!("{e}");
                }
                Err(RunError::LogFile(e)) => {
                    self.end_monitor(ProcessEndState::Failed);
                    eprintln!("{e}");
                    let base_manager: &'static dyn BaseManager = self.manager;
                    ui_harness::post_message_dialog(
                        Some(base_manager),
                        e.get_message(),
                        "Etomo Error".to_string(),
                        Some(self.axis_id),
                    );
                }
            }
            self.run_self_test(RAN_STATE);
        }
        // finally
        // Close the log file reader
        let log_file = self.log_file.lock().unwrap().clone();
        // Fixed in translation: CombineProcessMonitor.java:389 calls
        // `logFile.getAbsolutePath()` unguarded in the `finally` block, so when
        // `LogFile.getInstance` threw, the NullPointerException skipped
        // `msgMonitorStopped` and `setRunning(false)`.  The translation prints
        // "null" there, as string concatenation would.
        utilities::debug_print(
            &("LogFileProcessMonitor: Closing the log file reader for ".to_string()
                + &log_file
                    .as_ref()
                    .map(|log_file| log_file.get_absolute_path())
                    .unwrap_or_else(|| "null".to_string())),
        );
        if let Some(log_file) = log_file {
            let reader_id = self.log_file_reader_id.lock().unwrap().take();
            log_file.close_id(reader_id.as_ref().map(|reader_id| &**reader_id));
        }
        self.busy_status_mediator.msg_monitor_stopped(self.axis_id);
        self.set_running(false);
    }

    /// Java `isRunning`.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Java `isPausing`.
    fn is_pausing(&self) -> bool {
        false
    }

    /// Java `setWillResume`.
    fn set_will_resume(&self) {}

    /// Java `halt`.
    fn halt(&self) {}

    /// Java `hasProgressBarAccess`.
    fn has_progress_bar_access(&self) -> bool {
        true
    }

    /// Java `isReconnect`.
    fn is_reconnect(&self) -> bool {
        false
    }

    /// Java `synchronized final getProcessEndState`.
    fn get_process_end_state(&self) -> Option<ProcessEndState> {
        *self.end_state.lock().unwrap()
    }

    fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
    }
}

impl ProcessMonitor for CombineProcessMonitor {
    /// Java `synchronized final setProcessEndState`.
    fn set_process_end_state(&self, end_state: ProcessEndState) {
        let mut this_end_state = self.end_state.lock().unwrap();
        *this_end_state = Some(match *this_end_state {
            None => end_state,
            Some(existing) => ProcessEndState::precedence(existing, end_state),
        });
    }

    /// Java `kill`.
    fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) {
        let _ = (process, axis_id);
        self.end_monitor(ProcessEndState::Killed);
    }

    /// Java `pause`.
    fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        let _ = (process, axis_id);
        // Fixed in translation: CombineProcessMonitor.java:518 throws an
        // unchecked IllegalStateException ("can't pause a combine process") out
        // of the caller.  The translation reports it and answers "not able to
        // pause".
        eprintln!("java.lang.IllegalStateException: can't pause a combine process");
        false
    }

    /// Java `getStatusString`.
    fn get_status_string(&self) -> Option<String> {
        None
    }

    /// Java `getProcessMessages`.
    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        None
    }

    /// Java `stop`.
    fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java `useMessageReporter`.
    fn use_message_reporter(&self) {}

    /// Java `dumpState`.
    fn dump_state(&self) {}

    /// Java `msgLogFileRenaming(LogFile.Handle)` and its final two-argument
    /// overload.
    fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        self.tool_kit
            .msg_log_file_renaming_handle(log_file, indeterminate_mode);
    }

    /// Java `msgLogFileRenaming(String)`.
    fn msg_log_file_renaming_name(&self, file_name: &str) {
        self.tool_kit.msg_log_file_renaming_name(file_name);
    }

    /// Java final `msgLogFileRenaming(File)`.
    fn msg_log_file_renaming_file(&self, file: &Path) {
        self.tool_kit.msg_log_file_renaming_file(file);
    }

    /// Java `msgLogFileRenamed`.
    fn msg_log_file_renamed(&self) {
        self.tool_kit.msg_log_file_renamed(false);
    }

    /// Java `msgLogFileRenamingFailed`.
    fn msg_log_file_renaming_failed(&self) {
        self.tool_kit.msg_log_file_renaming_failed();
    }
}

impl DetachedProcessMonitor for CombineProcessMonitor {
    /// Java `isProcessRunning`.  Returns false if the process has stopped,
    /// after giving run() a chance to finish.
    fn is_process_running(&self) -> bool {
        if !self.process_running.load(Ordering::SeqCst) {
            // give run a chance to finish
            // (`Thread.sleep` on the calling thread, which is not this
            // monitor's and is never interrupted through it.)
            std::thread::sleep(std::time::Duration::from_millis(SLEEP));
            return false;
        }
        true
    }

    /// Java final `getProcessOutputFileName`.  The source returns null, which a
    /// detached command line would add as a null argument (a NullPointerException in
    /// `ProcessBuilder`); the empty string is returned instead.
    fn get_process_output_file_name(&self) -> Result<String, LogFileError> {
        Ok(String::new())
    }

    /// Java `setProcess`.
    fn set_process(&self, process: Arc<dyn SystemProcessInterface>) {
        *self.process.lock().unwrap() = Some(process);
    }
}
