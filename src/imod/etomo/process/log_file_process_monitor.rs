//! `IMOD/Etomo/src/etomo/process/LogFileProcessMonitor.java`.
//!
//! # How the abstract class is represented
//!
//! Java's `abstract class LogFileProcessMonitor` is split three ways, and
//! `file_size_process_monitor.rs` follows the same pattern:
//!
//! - [`LogFileProcessMonitor`] holds the class's fields and its methods that
//!   no subclass overrides.  Every field a subclass or another thread touches
//!   is an atomic or a mutex, because the monitor runs on its own thread and
//!   is shared with the process it watches.
//! - [`LogFileProcessMonitorImpl`] is the set of abstract and overridable
//!   methods.  Each takes the base state as `base`; a default method is the
//!   base class's body, so a subclass that does not override it gets the
//!   Java inherited behaviour, and one that calls `super.x()` calls the base
//!   method of the same name on `base`.
//! - [`LogFileProcessMonitorOf<S>`] is the concrete object: the base state
//!   plus the subclass state `S` (a concrete monitor such as
//!   `BlendmontProcessMonitor` is the `S`, holding the subclass's own fields).
//!   It implements `Monitor`/`ProcessMonitor`, dereferences to the base (as
//!   Java inheritance makes every base member callable on the subclass), and
//!   runs the base methods that dispatch to overridable ones (`run`,
//!   `processCurrentSection`).  `S` may be unsized, so
//!   `Arc<LogFileProcessMonitorOf<BlendmontProcessMonitor>>` coerces to
//!   `Arc<LogFileProcessMonitorOf<dyn LogFileProcessMonitorImpl>>`, which is
//!   how a parent monitor holds a child of any subclass (Java's
//!   `LogFileProcessMonitor curChildMonitor`); a sized one coerces to
//!   `Arc<dyn ProcessMonitor>`.
//!
//! Each concrete module's `new` returns `Arc<LogFileProcessMonitorOf<Self>>`,
//! built with `Arc::new_cyclic` so the `MonitorToolKit` gets the monitor
//! (Java `new MonitorToolKit(manager, axisID, this)`).
//!
//! Java's `Thread.interrupt()` on the monitor thread is `Monitor::interrupt`,
//! which sets `interrupted`; every `Thread.sleep` here is
//! `monitor_tool_kit::sleep` on that flag.

use super::emergency_monitor::EmergencyMonitor;
use super::monitor::{Monitor, ProcessMonitor};
use super::monitor_tool_kit::{self, InterruptedException, MonitorToolKit, WHITESPACE};
use super::process_interface::{ProcessResultDisplayRef, SystemProcessInterface};
use super::process_messages::ProcessMessages;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::storage::log_file::{
    Handle, LockException, LogFile, LogFileError, ReaderId, UnlockedException,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::util::mrc_header;
use crate::imod::etomo::util::utilities;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// Java `UPDATE_PERIOD`.
pub const UPDATE_PERIOD: u64 = 5; // 500;

/// The checked exceptions the Java methods of this class and its subclasses
/// declare (`NumberFormatException`, `LogFileException`, `IOException`,
/// `InvalidParameterException`, `InterruptedException`), as one `Result`
/// error.
#[derive(Debug)]
pub enum MonitorError {
    /// `java.lang.NumberFormatException`, with its message.
    NumberFormat(String),
    /// `LogFile.LogFileException` and `java.io.IOException`.
    LogFile(LogFileError),
    /// `etomo.util.InvalidParameterException`.
    InvalidParameter(InvalidParameterException),
    /// `java.lang.InterruptedException`.
    Interrupted(InterruptedException),
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitorError::NumberFormat(message) => {
                write!(f, "java.lang.NumberFormatException: {message}")
            }
            MonitorError::LogFile(e) => write!(f, "{e}"),
            MonitorError::InvalidParameter(e) => write!(f, "{e}"),
            MonitorError::Interrupted(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MonitorError {}

impl From<LogFileError> for MonitorError {
    fn from(e: LogFileError) -> MonitorError {
        MonitorError::LogFile(e)
    }
}

impl From<InterruptedException> for MonitorError {
    fn from(e: InterruptedException) -> MonitorError {
        MonitorError::Interrupted(e)
    }
}

impl From<InvalidParameterException> for MonitorError {
    fn from(e: InvalidParameterException) -> MonitorError {
        MonitorError::InvalidParameter(e)
    }
}

/// The abstract and overridable methods of Java `LogFileProcessMonitor`; see
/// the module comment.
pub trait LogFileProcessMonitorImpl: Send + Sync + 'static {
    /// Java abstract `getCurrentSection`.  Returns the last line found.
    fn get_current_section(
        &self,
        base: &LogFileProcessMonitor,
    ) -> Result<Option<String>, MonitorError>;

    /// Java abstract `getTitle`.
    fn get_title(&self, base: &LogFileProcessMonitor) -> String;

    /// Java `initializeProgressBar()`.
    fn initialize_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.tool_kit
            .initialize_progress_bar(&self.get_title(base), false);
    }

    /// Java `findNSections`.  Search the log file for the header section and
    /// extract the number of sections.
    fn find_n_sections(&self, base: &LogFileProcessMonitor) -> Result<(), MonitorError> {
        base.find_n_sections()
    }

    /// Java `updateProgressBar`.
    fn update_progress_bar(&self, base: &LogFileProcessMonitor) {
        base.update_progress_bar();
    }

    /// Java `postProcess`.
    fn post_process(&self, base: &LogFileProcessMonitor) {
        let _ = base;
    }
}

/// Java `LogFileProcessMonitor`'s mutable timing fields, which only the
/// monitor thread touches (`calcRemainingTime`, `updateProgressBar`).
struct Progress {
    /// Java field `averageSectionTime`.
    average_section_time: f64,
    /// Java field `updatedAverageSectionTime`.
    updated_average_section_time: f64,
    /// Java field `timestamp`.
    timestamp: f64,
    /// Java field `waitTime`.
    wait_time: f64,
    /// Java field `doneSections`.
    done_sections: i32,
    /// Java field `percentDone`.
    percent_done: i64,
    /// Java field `etc`.
    etc: f64,
}

/// The state and non-overridable methods of Java `LogFileProcessMonitor`.
pub struct LogFileProcessMonitor {
    /// Java field `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    pub axis_id: AxisID,
    /// Java field `emergencyMonitor`.
    emergency_monitor: Arc<EmergencyMonitor>,
    /// Java field `busyStatusMediator`.
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java field `processResultDisplay`.
    process_result_display: Option<ProcessResultDisplayRef>,
    /// Java field `processName`.
    process_name: Option<ProcessName>,
    /// Java field `toolKit`.
    pub tool_kit: MonitorToolKit,

    /// Java field `processStartTime`.
    process_start_time: AtomicI64,
    /// Java field `nSections`.
    pub n_sections: AtomicI32,
    /// Java field `currentSection`.
    pub current_section: AtomicI32,
    /// Java field `processRunning`.
    process_running: AtomicBool,
    /// Java `volatile` field `done`.
    done: AtomicBool,
    /// Java field `debug`.
    debug: AtomicBool,
    /// Java field `lastProcess`.
    last_process: AtomicBool,
    /// Java field `standardLogFileName`.
    pub standard_log_file_name: AtomicBool,
    /// Java field `endState`; the mutex stands for the `synchronized` of
    /// `setProcessEndState`/`getProcessEndState`.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java field `stop`.
    stop: AtomicBool,
    /// Java field `ending`.
    pub ending: AtomicBool,

    /// Java field `logFileBasename`.  This needs to be set in the concrete
    /// class constructor.
    pub log_file_basename: Mutex<Option<String>>,
    /// Java field `logFile`.
    log_file: Mutex<Option<Arc<Handle>>>,
    /// Java field `logFileReaderId`.
    log_file_reader_id: Mutex<Option<ReaderId>>,
    /// `synchronized (logFile)`: Java locks the handle object itself; only
    /// this monitor synchronizes on it, so the lock is held here.
    log_file_lock: Mutex<()>,

    /// Java field `nSectionsHeader`.
    pub n_sections_header: &'static str,
    /// Java field `nSectionsIndex`.
    pub n_sections_index: i32,

    /// Java fields `averageSectionTime` .. `etc`.
    progress: Mutex<Progress>,
    /// Java `volatile` field `running`.
    running: AtomicBool,

    /// The instance's intrinsic lock, taken by the `synchronized` methods
    /// `processCurrentSection` and `handleLockException`.
    synchronized: Mutex<()>,
    /// `Thread.interrupt()` on this monitor's thread; see the module comment.
    pub interrupted: AtomicBool,
}

impl LogFileProcessMonitor {
    /// Java `LogFileProcessMonitor(BaseManager, AxisID, ProcessResultDisplay,
    /// ProcessName)`.  `monitor` is the Java `this` handed to the tool kit.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        monitor: Weak<dyn Monitor>,
    ) -> LogFileProcessMonitor {
        let emergency_monitor = manager.get_emergency_monitor(Some(axis_id));
        let tool_kit = MonitorToolKit::new(manager, axis_id, Some(monitor));
        let busy_status_mediator = manager.get_busy_status_mediator();
        busy_status_mediator.msg_monitor_constructed(axis_id);
        let (n_sections_header, n_sections_index) = if !etomo_director::INSTANCE.is_imod_brief_header()
        {
            (mrc_header::SIZE_HEADER, mrc_header::N_SECTIONS_INDEX)
        } else {
            (
                mrc_header::SIZE_HEADER_BRIEF,
                mrc_header::N_SECTIONS_INDEX_BRIEF,
            )
        };
        LogFileProcessMonitor {
            manager,
            axis_id,
            emergency_monitor,
            busy_status_mediator,
            process_result_display,
            process_name,
            tool_kit,
            process_start_time: AtomicI64::new(0),
            n_sections: AtomicI32::new(i32::MIN),
            current_section: AtomicI32::new(0),
            process_running: AtomicBool::new(true),
            done: AtomicBool::new(false),
            debug: AtomicBool::new(false),
            last_process: AtomicBool::new(true),
            standard_log_file_name: AtomicBool::new(true),
            end_state: Mutex::new(None),
            stop: AtomicBool::new(false),
            ending: AtomicBool::new(false),
            log_file_basename: Mutex::new(None),
            log_file: Mutex::new(None),
            log_file_reader_id: Mutex::new(None),
            log_file_lock: Mutex::new(()),
            n_sections_header,
            n_sections_index,
            progress: Mutex::new(Progress {
                average_section_time: 0.0,
                updated_average_section_time: 0.0,
                timestamp: 0.0,
                wait_time: 0.0,
                done_sections: 0,
                percent_done: 0,
                etc: -1.0,
            }),
            running: AtomicBool::new(false),
            synchronized: Mutex::new(()),
            interrupted: AtomicBool::new(false),
        }
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        self.process_name
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.store(debug, Ordering::SeqCst);
    }

    /// Java final `stop`.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java `synchronized final handleLockException`.
    pub fn handle_lock_exception(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _synchronized = self.synchronized.lock().unwrap();
        if let Some(lock_exception) = lock_exception {
            self.emergency_monitor.alert(Some(lock_exception), do_popup);
            self.stop();
            self.process_running.store(false, Ordering::SeqCst);
            self.set_process_end_state(ProcessEndState::FileLockFailure);
            self.set_running(false);
        }
    }

    /// Java `initializeProgressBar(String, int)`.
    pub fn initialize_progress_bar_label(&self, label: &str, n_sections: i32) {
        self.tool_kit
            .initialize_progress_bar_n_sections(label, n_sections, false);
    }

    /// Java final `readLogFileLine`.
    pub fn read_log_file_line(&self) -> Result<Option<String>, LogFileError> {
        let log_file = self.log_file.lock().unwrap().clone();
        let reader_id = self.log_file_reader_id.lock().unwrap().clone();
        // Java `logFile.readLine(logFileReaderId)`: a null id is not locked, so
        // `LogFile.readLine` throws an `UnlockedException`.
        match (log_file, reader_id) {
            (Some(log_file), Some(reader_id)) => log_file.read_line(&reader_id),
            (_, reader_id) => Err(LogFileError::Unlocked(UnlockedException::new_id(
                reader_id.as_ref().map(|reader_id| &**reader_id),
                None,
            ))),
        }
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

    /// Java final `isDone`.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }

    /// Java final `setLastProcess`.
    pub fn set_last_process(&self, last_process: bool) {
        self.last_process.store(last_process, Ordering::SeqCst);
    }

    /// Java private final `waitForLogFile`.  Wait for the process to start and
    /// the appropriate log file to be created.
    fn wait_for_log_file(&self) -> Result<(), MonitorError> {
        self.process_start_time.store(
            utilities::java_lang_system_current_time_millis(),
            Ordering::SeqCst,
        );
        let log_file = self.log_file.lock().unwrap().clone().unwrap();
        let mut new_log_file = false;
        while self.is_running() && !new_log_file {
            // Check to see if the log file exists that signifies that the process
            // has started
            if log_file.exists() {
                new_log_file = true;
            } else {
                monitor_tool_kit::sleep(&self.interrupted, UPDATE_PERIOD)?;
            }
        }
        // Open the log file
        if !self.done.load(Ordering::SeqCst) {
            let _synchronized = self.log_file_lock.lock().unwrap();
            if !self.done.load(Ordering::SeqCst) {
                match log_file.open_reader() {
                    Ok(reader_id) => *self.log_file_reader_id.lock().unwrap() = reader_id,
                    Err(LogFileError::Lock(e)) => self.handle_lock_exception(Some(&e), true),
                    Err(e) => return Err(e.into()),
                }
            }
        }
        Ok(())
    }

    /// Java `findNSections` (the base class body).  Search the log file for
    /// the header section and extract the number of sections.  WARNING:  Do
    /// not catch an interruptedException in this function, or the run loop
    /// won't end.
    pub fn find_n_sections(&self) -> Result<(), MonitorError> {
        // Search for the number of sections, we should see a header ouput first
        let mut found_n_sections = false;
        self.n_sections.store(-1, Ordering::SeqCst);
        while self.is_running() && !found_n_sections {
            monitor_tool_kit::sleep(&self.interrupted, UPDATE_PERIOD)?;
            while self.is_running() {
                let line = match self.read_log_file_line()? {
                    None => break,
                    Some(line) => line,
                };
                let line = java_lang_string_trim(&line);
                if line.starts_with(self.n_sections_header) {
                    let fields = utilities::java_lang_string_split(line, &WHITESPACE);
                    if fields.len() as i32 > self.n_sections_index {
                        self.n_sections.store(
                            java_lang_integer_parse_int(&fields[self.n_sections_index as usize])
                                .map_err(MonitorError::NumberFormat)?,
                            Ordering::SeqCst,
                        );
                        found_n_sections = true;
                        break;
                    } else {
                        return Err(MonitorError::NumberFormat(
                            "Incomplete size line in header".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Java private `calcRemainingTime`.  Calculate the amount of time
    /// remaining and percent done.
    fn calc_remaining_time(&self) {
        let current_section = self.current_section.load(Ordering::SeqCst);
        let n_sections = self.n_sections.load(Ordering::SeqCst);
        let mut p = self.progress.lock().unwrap();
        let newtime_stamp = utilities::java_lang_system_current_time_millis() as f64;
        p.wait_time += newtime_stamp - p.timestamp;
        p.timestamp = newtime_stamp;
        // Has the current section been incremented?
        if p.done_sections < current_section.wrapping_sub(1) {
            // At least one section has been completed since the last time this function was
            // called, so recalculate average section completion time: the total time over the
            // total number of sections done.
            p.average_section_time = ((p.done_sections as f64 * p.average_section_time)
                + p.wait_time)
                / current_section.wrapping_sub(1) as f64;
            p.done_sections = current_section.wrapping_sub(1);
            p.wait_time = 0.0;
            p.updated_average_section_time = p.average_section_time;
            p.percent_done =
                utilities::java_lang_math_round(p.done_sections as f64 / n_sections as f64 * 100.0);
        }
        // If the current section has not been incremented and the wait for the current
        // section to be completed exceeds the average section completion time, then
        // recalculate updatedAverageSectionTime.
        else if p.wait_time > p.average_section_time {
            p.updated_average_section_time = ((p.done_sections as f64 * p.average_section_time)
                + p.wait_time)
                / p.done_sections.wrapping_add(1) as f64;
        }
        // Calculate the new estimated time to completion.
        if p.updated_average_section_time > 0.0 {
            p.etc = p.updated_average_section_time
                * n_sections.wrapping_sub(p.done_sections) as f64
                - p.wait_time;
        }
    }

    /// Java `updateProgressBar` (the base class body).  Update the progress
    /// bar with percentage done and estimated time to completion message.
    pub fn update_progress_bar(&self) {
        let axis_id = self.axis_id;
        if self.ending.load(Ordering::SeqCst) {
            self.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_standard_bar_string_axis_id(0, Some(StandardBarString::Ending), axis_id);
            }));
            return;
        }
        let (percent_done, etc, done_sections) = {
            let p = self.progress.lock().unwrap();
            (p.percent_done, p.etc, p.done_sections)
        };
        let message = if etc >= 0.0 {
            // Format the progress bar string
            percent_done.to_string() + "%   ETC: " + &utilities::millis_to_min_and_secs(etc)
        } else {
            percent_done.to_string() + "%"
        };
        if self.process_running.load(Ordering::SeqCst) {
            self.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_string_axis_id(done_sections, Some(&message), axis_id);
            }));
        }
    }

    /// Java final `haltProcess(Thread, ProcessEndState)`.  Java passes the
    /// thread running this monitor, or null; `run_thread` is true when it
    /// passed that thread, whose interruption is this monitor's
    /// `Monitor::interrupt`.
    pub fn halt_process(&self, run_thread: bool, end_state: Option<ProcessEndState>) {
        self.process_running.store(false, Ordering::SeqCst);
        if let (Some(process_result_display), Some(end_state)) =
            (&self.process_result_display, end_state)
        {
            let process_result_display = process_result_display.clone();
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                process_result_display
                    .get()
                    .msg_process_end_state(end_state);
            });
        }
        if run_thread {
            self.interrupted.store(true, Ordering::SeqCst);
        }
    }

    /// Java `isProcessRunning`.
    pub fn is_process_running(&self) -> bool {
        self.process_running.load(Ordering::SeqCst)
    }

    /// Java `isStop`.
    pub fn is_stop(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Java private `synchronized setRunning`.
    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }

    /// Java `isRunning`.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// A Java `LogFileProcessMonitor` subclass instance: the base state plus the
/// subclass's own state; see the module comment.
pub struct LogFileProcessMonitorOf<S: ?Sized + LogFileProcessMonitorImpl> {
    /// The superclass state.
    pub base: LogFileProcessMonitor,
    /// The subclass state and overrides.
    pub subclass: S,
}

/// Java inheritance: every base member is reachable on the subclass.
impl<S: ?Sized + LogFileProcessMonitorImpl> std::ops::Deref for LogFileProcessMonitorOf<S> {
    type Target = LogFileProcessMonitor;

    fn deref(&self) -> &LogFileProcessMonitor {
        &self.base
    }
}

impl<S: LogFileProcessMonitorImpl> LogFileProcessMonitorOf<S> {
    /// Runs the Java superclass constructor `LogFileProcessMonitor(BaseManager,
    /// AxisID, ProcessResultDisplay, ProcessName)` for a subclass instance
    /// whose own fields are `subclass`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        subclass: S,
    ) -> Arc<LogFileProcessMonitorOf<S>> {
        Arc::new_cyclic(|this: &Weak<LogFileProcessMonitorOf<S>>| {
            let monitor: Weak<dyn Monitor> = this.clone();
            LogFileProcessMonitorOf {
                base: LogFileProcessMonitor::new(
                    manager,
                    axis_id,
                    process_result_display,
                    process_name,
                    monitor,
                ),
                subclass,
            }
        })
    }
}

impl<S: ?Sized + LogFileProcessMonitorImpl> LogFileProcessMonitorOf<S> {
    /// Java `synchronized final processCurrentSection`.
    pub fn process_current_section(&self) -> Result<(), MonitorError> {
        let _synchronized = self.base.synchronized.lock().unwrap();
        if self.base.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        let line = self.subclass.get_current_section(&self.base)?;
        if let Some(line) = line {
            if line.contains(process_output_strings::SUCCESS_TAG) {
                self.base.process_running.store(false, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    /// The body of Java `run` inside its two `try` blocks.
    fn run_log_file(&self) -> Result<(), MonitorError> {
        let base = &self.base;
        // Instantiate the logFile object
        let user_dir = base.manager.get_property_user_dir().unwrap_or_default();
        let log_file_basename = base
            .log_file_basename
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "null".to_string());
        let log_file = if base.standard_log_file_name.load(Ordering::SeqCst) {
            // logFileName = logFileBasename + axisID.getExtension() + ".log";
            LogFile::get_instance_name(
                &user_dir,
                base.axis_id,
                &log_file_basename,
                Some(base.manager.get_emergency_monitor(Some(base.axis_id))),
            )?
        } else {
            // logFileName = logFileBasename;
            LogFile::get_instance_user_dir(
                &user_dir,
                &log_file_basename,
                Some(base.manager.get_emergency_monitor(Some(base.axis_id))),
            )?
        };
        *base.log_file.lock().unwrap() = Some(log_file);
        // Wait for the log file to exist
        base.wait_for_log_file()?;
        self.subclass.find_n_sections(base)?;
        self.subclass.initialize_progress_bar(base);
        base.progress.lock().unwrap().timestamp =
            utilities::java_lang_system_current_time_millis() as f64;
        while base.is_running()
            && base.process_running.load(Ordering::SeqCst)
            && !base.stop.load(Ordering::SeqCst)
        {
            monitor_tool_kit::sleep(&base.interrupted, UPDATE_PERIOD)?;
            self.process_current_section()?;
            base.calc_remaining_time();
            self.subclass.update_progress_bar(base);
        }
        base.set_process_end_state(ProcessEndState::Done);
        Ok(())
    }
}

impl<S: ?Sized + LogFileProcessMonitorImpl> Monitor for LogFileProcessMonitorOf<S> {
    /// Java final `run`.
    fn run(&self) {
        let base = &self.base;
        base.set_running(true);
        {
            {
                self.subclass.initialize_progress_bar(base);
                match self.run_log_file() {
                    Ok(()) => {}
                    Err(MonitorError::Interrupted(e)) => {
                        eprintln!("{e}");
                        base.set_process_end_state(ProcessEndState::Done);
                    }
                    Err(e) => {
                        eprintln!("{e}");
                        base.set_process_end_state(ProcessEndState::Failed);
                    }
                }
                // Close the log file reader
                // Fixed in translation: LogFileProcessMonitor.java:193 calls
                // `logFile.getAbsolutePath()` unguarded, so when
                // `LogFile.getInstance` threw, the NullPointerException skipped
                // `progressBarDone` and ended the thread.  The translation
                // prints "null" there, as string concatenation would.
                let absolute_path = base
                    .log_file
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|log_file| log_file.get_absolute_path())
                    .unwrap_or_else(|| "null".to_string());
                utilities::debug_print(
                    &("LogFileProcessMonitor: Closing the log file reader for ".to_string()
                        + &absolute_path),
                );
            }
            // finally
            base.done.store(true, Ordering::SeqCst);
            if !base.last_process.load(Ordering::SeqCst) {
                self.subclass.post_process(base);
            }
            if base.last_process.load(Ordering::SeqCst) {
                base.manager
                    .progress_bar_done(Some(base.axis_id), base.get_process_end_state());
            }
        }
        // finally
        base.process_running.store(false, Ordering::SeqCst);
        base.busy_status_mediator.msg_monitor_stopped(base.axis_id);
        let log_file = base.log_file.lock().unwrap().clone();
        if let Some(log_file) = log_file {
            let _synchronized = base.log_file_lock.lock().unwrap();
            // Test closing an abandoned read lock from another handle.
            let reader_id = base.log_file_reader_id.lock().unwrap().take();
            log_file.close_id(reader_id.as_ref().map(|reader_id| &**reader_id));
        }
        base.set_running(false);
    }

    /// Java `isRunning`.
    fn is_running(&self) -> bool {
        self.base.is_running()
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
        self.base.get_process_end_state()
    }

    fn interrupt(&self) {
        self.base.interrupted.store(true, Ordering::SeqCst);
    }
}

impl<S: ?Sized + LogFileProcessMonitorImpl> ProcessMonitor for LogFileProcessMonitorOf<S> {
    /// Java `synchronized final setProcessEndState`.
    fn set_process_end_state(&self, end_state: ProcessEndState) {
        self.base.set_process_end_state(end_state);
    }

    /// Java final `kill`.
    fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) {
        *self.base.end_state.lock().unwrap() = Some(ProcessEndState::Killed);
        process.signal_kill(axis_id);
    }

    /// Java final `pause`.
    fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        let _ = (process, axis_id);
        // Fixed in translation: LogFileProcessMonitor.java:409 throws an
        // unchecked IllegalStateException ("pause illegal in this monitor") out
        // of the caller.  The translation reports it and answers "not able to
        // pause".
        eprintln!("java.lang.IllegalStateException: pause illegal in this monitor");
        false
    }

    /// Java final `getStatusString`.
    fn get_status_string(&self) -> Option<String> {
        None
    }

    /// Java final `getProcessMessages`.
    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        None
    }

    /// Java final `stop`.
    fn stop(&self) {
        self.base.stop();
    }

    /// Java `useMessageReporter`.
    fn use_message_reporter(&self) {}

    /// Java final `dumpState`.
    fn dump_state(&self) {}

    /// Java `msgLogFileRenaming(LogFile.Handle, boolean)`; the final
    /// one-argument overload passes false.
    fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        self.base
            .tool_kit
            .msg_log_file_renaming_handle(log_file, indeterminate_mode);
    }

    /// Java final `msgLogFileRenaming(String)`.
    fn msg_log_file_renaming_name(&self, file_name: &str) {
        self.base.tool_kit.msg_log_file_renaming_name(file_name);
    }

    /// Java final `msgLogFileRenaming(File)`.
    fn msg_log_file_renaming_file(&self, file: &Path) {
        self.base.tool_kit.msg_log_file_renaming_file(file);
    }

    /// Java final `msgLogFileRenamed`.
    fn msg_log_file_renamed(&self) {
        self.base.tool_kit.msg_log_file_renamed(false);
    }

    /// Java final `msgLogFileRenamingFailed`.
    fn msg_log_file_renaming_failed(&self) {
        self.base.tool_kit.msg_log_file_renaming_failed();
    }
}
