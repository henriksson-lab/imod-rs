//! `IMOD/Etomo/src/etomo/process/FileSizeProcessMonitor.java`.
//!
//! The abstract class is represented exactly as `LogFileProcessMonitor` is
//! (see the module comment of `log_file_process_monitor.rs`):
//! [`FileSizeProcessMonitor`] is the class's state and non-overridable
//! methods, [`FileSizeProcessMonitorImpl`] its abstract and overridable
//! methods (each taking the base state), and [`FileSizeProcessMonitorOf<S>`]
//! the concrete subclass instance, which implements `ProcessMonitor` and runs
//! the base methods that dispatch to overridable ones (`run`, `waitForFile`,
//! `updateProgressBar`, `openChannel`).
//!
//! `java.nio.channels.FileChannel.size()` is the length of the open file, read
//! through its metadata.

use super::emergency_monitor::EmergencyMonitor;
use super::monitor::{Monitor, ProcessMonitor};
use super::monitor_tool_kit::{self, InterruptedException, MonitorToolKit};
use super::process_interface::SystemProcessInterface;
use super::process_messages::ProcessMessages;
use super::process_output_strings;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::log_file::{
    Handle, LockException, LogFile, LogFileError, ReaderId, UnlockedException,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::util::utilities;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// The checked exceptions of Java `calcFileSize`: `InvalidParameterException`
/// and `IOException`.  `MRCHeader.read` reports both as one message
/// (`util/mrc_header.rs`), which arrives as `Io`.
#[derive(Debug)]
pub enum CalcFileSizeError {
    /// `etomo.util.InvalidParameterException`.
    InvalidParameter(InvalidParameterException),
    /// `java.io.IOException`, with its message.
    Io(String),
}

impl std::fmt::Display for CalcFileSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalcFileSizeError::InvalidParameter(e) => write!(f, "{e}"),
            CalcFileSizeError::Io(message) => write!(f, "java.io.IOException: {message}"),
        }
    }
}

impl From<InvalidParameterException> for CalcFileSizeError {
    fn from(e: InvalidParameterException) -> CalcFileSizeError {
        CalcFileSizeError::InvalidParameter(e)
    }
}

/// The abstract and overridable methods of Java `FileSizeProcessMonitor`.
pub trait FileSizeProcessMonitorImpl: Send + Sync + 'static {
    /// Java abstract `getTitle`.
    fn get_title(&self, base: &FileSizeProcessMonitor) -> String;

    /// Java abstract `calcFileSize`.  The derived class must implement this
    /// function to
    /// - set the expected number of bytes in the output file
    /// - initialize the progress bar through the application manager, the
    ///   maximum value should be the expected size of the file in k bytes
    /// - set the watchedFile reference to the output file being monitored.
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError>;

    /// Java abstract `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor);

    /// Java `getStatusFromLog`.  Inheriting classes that override this
    /// function need to call super.getStatusFromLog to make sure the monitor
    /// can exit.
    fn get_status_from_log(
        &self,
        base: &FileSizeProcessMonitor,
    ) -> Result<(), InterruptedException> {
        base.get_status_from_log()
    }
}

/// The state and non-overridable methods of Java `FileSizeProcessMonitor`.
pub struct FileSizeProcessMonitor {
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

    /// Java field `processStartTime`.
    pub process_start_time: AtomicI64,
    /// Java field `scriptStartTime`.
    pub script_start_time: AtomicI64,
    /// Java field `watchedFile`.
    pub watched_file: Mutex<Option<PathBuf>>,
    /// Java field `watchedChannel`.
    watched_channel: Mutex<Option<std::fs::File>>,
    /// Java field `endState`; the mutex stands for the `synchronized` of
    /// `setProcessEndState`/`getProcessEndState`.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java field `ending`.
    ending: AtomicBool,
    /// Java field `reconnect`.
    reconnect: AtomicBool,
    /// Java field `stop`.
    stop: AtomicBool,
    /// Java field `usingLog` (never read in the source).
    using_log: AtomicBool,
    /// Java field `nKBytes`.
    pub n_k_bytes: AtomicI32,
    /// Java field `updatePeriod`.
    pub update_period: AtomicI32,
    /// Java field `lastSize`.
    pub last_size: AtomicI64,
    /// Java field `processName`.
    pub process_name: ProcessName,
    /// Java field `stream`.
    stream: Mutex<Option<std::fs::File>>,
    /// Java field `logFile`.
    log_file: Option<Arc<Handle>>,
    /// Java field `logReaderId`.
    log_reader_id: Mutex<Option<ReaderId>>,
    /// Java field `findWatchedFileName`.
    find_watched_file_name: AtomicBool,
    /// Java field `messageReporter`.
    // TODO(unit): needs etomo/process/MessageReporter.java - the reporter
    // `useMessageReporter` creates (`new MessageReporter(axisID, logFile)`) and
    // the loop's `checkForMessages(manager)`/`close()`; until then it stays null.
    message_reporter: Mutex<Option<Infallible>>,
    /// Java field `gotStatusFromLog`.
    got_status_from_log: AtomicBool,
    /// Java field `fileWriting`.
    file_writing: AtomicBool,
    /// Java field `logIOExceptions`.
    log_io_exceptions: AtomicI32,
    /// Java `volatile` field `running`.
    running: AtomicBool,

    /// The instance's intrinsic lock, taken by the `synchronized` method
    /// `handleLockException`.
    synchronized: Mutex<()>,
    /// `Thread.interrupt()` on this monitor's thread.
    pub interrupted: AtomicBool,
}

impl FileSizeProcessMonitor {
    /// Java `FileSizeProcessMonitor(BaseManager, AxisID, ProcessName)`.
    /// `monitor` is the Java `this` handed to the tool kit.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_name: ProcessName,
        monitor: Weak<dyn Monitor>,
    ) -> FileSizeProcessMonitor {
        let emergency_monitor = manager.get_emergency_monitor(Some(axis_id));
        let tool_kit = MonitorToolKit::new(manager, axis_id, Some(monitor));
        let busy_status_mediator = manager.get_busy_status_mediator();
        busy_status_mediator.msg_monitor_constructed(axis_id);
        let script_start_time = utilities::java_lang_system_current_time_millis();
        let log_file = match LogFile::get_instance_process_name(
            &manager.get_property_user_dir().unwrap_or_default(),
            axis_id,
            process_name,
            Some(manager.get_emergency_monitor(Some(axis_id))),
        ) {
            Ok(log_file) => Some(log_file),
            Err(e) => {
                eprintln!("{e}");
                ui_harness::post_message_dialog(
                    Some(manager),
                    "Unable to create log file.\n".to_string() + &e.get_message(),
                    "File Size Monitor Log File Failure".to_string(),
                    None,
                );
                None
            }
        };
        FileSizeProcessMonitor {
            manager,
            axis_id,
            emergency_monitor,
            busy_status_mediator,
            tool_kit,
            process_start_time: AtomicI64::new(0),
            script_start_time: AtomicI64::new(script_start_time),
            watched_file: Mutex::new(None),
            watched_channel: Mutex::new(None),
            end_state: Mutex::new(None),
            ending: AtomicBool::new(false),
            reconnect: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            using_log: AtomicBool::new(false),
            n_k_bytes: AtomicI32::new(0),
            update_period: AtomicI32::new(5), // 250;
            last_size: AtomicI64::new(0),
            process_name,
            stream: Mutex::new(None),
            log_file,
            log_reader_id: Mutex::new(None),
            find_watched_file_name: AtomicBool::new(true),
            message_reporter: Mutex::new(None),
            got_status_from_log: AtomicBool::new(false),
            file_writing: AtomicBool::new(true),
            log_io_exceptions: AtomicI32::new(0),
            running: AtomicBool::new(false),
            synchronized: Mutex::new(()),
            interrupted: AtomicBool::new(false),
        }
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> ProcessName {
        self.process_name
    }

    /// `updatePeriod` as the millisecond count `Thread.sleep` takes.
    fn update_period(&self) -> u64 {
        self.update_period.load(Ordering::SeqCst) as u64
    }

    /// Java final `setGotStatusFromLog`.
    pub fn set_got_status_from_log(&self, input: bool) {
        self.got_status_from_log.store(input, Ordering::SeqCst);
    }

    /// Java final `isGotStatusFromLog`.
    pub fn is_got_status_from_log(&self) -> bool {
        self.got_status_from_log.load(Ordering::SeqCst)
    }

    /// Java final `getModeBytes`.
    pub fn get_mode_bytes(&self, mode: i32) -> Result<i32, InvalidParameterException> {
        match mode {
            0 => Ok(1),
            1 => Ok(2),
            2 => Ok(4),
            6 => Ok(2),
            _ => Err(InvalidParameterException::new("Unknown mode parameter")),
        }
    }

    /// Java `synchronized final handleLockException`.
    pub fn handle_lock_exception(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _synchronized = self.synchronized.lock().unwrap();
        if let Some(lock_exception) = lock_exception {
            self.emergency_monitor.alert(Some(lock_exception), do_popup);
            self.stop();
            self.set_process_end_state(ProcessEndState::FileLockFailure);
            self.set_running(false);
        }
    }

    /// Java `useMessageReporter`.
    pub fn use_message_reporter(&self) {
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `messageReporter = new MessageReporter(axisID, logFile)`.
        let _ = &self.message_reporter;
    }

    /// Java final `stop`.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java private final `getLogFile`.
    fn get_log_file(&self) -> Option<&Arc<Handle>> {
        self.log_file.as_ref()
    }

    /// Java `readLineFromLog`.
    pub fn read_line_from_log(&self) -> Result<Option<String>, LogFileError> {
        let log_reader_id = self.log_reader_id.lock().unwrap().clone();
        if let Some(log_reader_id) = log_reader_id {
            // `logReaderId` is only ever set from `logFile.openReader()`, so the
            // log file exists here.
            return self.get_log_file().unwrap().read_line(&log_reader_id);
        }
        Ok(None)
    }

    /// Java final `setReconnect`.
    pub fn set_reconnect(&self, reconnect: bool) {
        self.reconnect.store(reconnect, Ordering::SeqCst);
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

    /// Java `setEnding`.
    pub fn set_ending(&self) {
        self.ending.store(true, Ordering::SeqCst);
    }

    /// Java `isEnding`.
    pub fn is_ending(&self) -> bool {
        self.ending.load(Ordering::SeqCst)
    }

    /// Java `isFileWriting`.
    pub fn is_file_writing(&self) -> bool {
        self.file_writing.load(Ordering::SeqCst)
    }

    /// Java `getStatusFromLog` (the base class body).
    pub fn get_status_from_log(&self) -> Result<(), InterruptedException> {
        if self.is_ending() {
            return Ok(());
        }
        let mut exception = false;
        // try
        let result: Result<(), LogFileError> = 'try_block: {
            if self.log_reader_id.lock().unwrap().is_none() {
                // Fixed in translation: FileSizeProcessMonitor.java:383 calls
                // `getLogFile().openReader()` on a null `logFile` when the
                // constructor could not create the log file, and the
                // NullPointerException ends the monitor thread.  The translation
                // treats the missing file as a failed read.
                let log_file = match self.get_log_file() {
                    None => {
                        break 'try_block Err(LogFileError::Unlocked(UnlockedException::new_id(
                            None, None,
                        )));
                    }
                    Some(log_file) => log_file,
                };
                match log_file.open_reader() {
                    Ok(reader_id) => *self.log_reader_id.lock().unwrap() = reader_id,
                    Err(LogFileError::Lock(e)) => {
                        self.handle_lock_exception(Some(&e), true);
                        return Ok(());
                    }
                    Err(e) => break 'try_block Err(e),
                }
            }
            while self.is_running() {
                let line = match self.read_line_from_log() {
                    Ok(None) => break,
                    Ok(Some(line)) => line,
                    Err(e) => break 'try_block Err(e),
                };
                self.log_io_exceptions.store(0, Ordering::SeqCst);
                if line.contains(process_output_strings::SUCCESS_TAG) {
                    self.file_writing.store(false, Ordering::SeqCst);
                    break;
                }
                monitor_tool_kit::sleep(&self.interrupted, 1)?;
            }
            Ok(())
        };
        match result {
            Ok(()) => {}
            Err(LogFileError::Io(e)) if e.kind() != std::io::ErrorKind::NotFound => {
                exception = true;
                if self.log_io_exceptions.fetch_add(1, Ordering::SeqCst) < 3 {
                    // Don't fill up the log with duplicate exceptions.
                    eprintln!("{e}");
                }
            }
            Err(_) => {
                // LogFileException | FileNotFoundException
                exception = true;
            }
        }
        if exception || self.is_ending() || !self.is_file_writing() {
            self.close_log_reader();
        }
        Ok(())
    }

    /// Java `closeLogReader`.
    pub fn close_log_reader(&self) {
        let log_reader_id = self.log_reader_id.lock().unwrap().take();
        if let Some(log_reader_id) = log_reader_id {
            self.get_log_file().unwrap().close_id(Some(&*log_reader_id));
        }
    }

    /// Java final `closeOpenFiles`.  Attempt to close the watched channel
    /// file.
    pub fn close_open_files(&self) {
        self.close_channel();
        self.close_log_file_reader();
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `if (messageReporter != null) messageReporter.close()`.
    }

    /// Java final `setFindWatchedFileName`.
    pub fn set_find_watched_file_name(&self, input: bool) {
        self.find_watched_file_name.store(input, Ordering::SeqCst);
    }

    /// Java private final `openLogFileReader`.
    fn open_log_file_reader(&self) -> Result<(), InterruptedException> {
        if !self.reconnect.load(Ordering::SeqCst) {
            while self.is_running() && !self.tool_kit.is_renamed() {
                monitor_tool_kit::sleep(&self.interrupted, self.update_period())?;
            }
        }
        // Fixed in translation: FileSizeProcessMonitor.java:450 calls
        // `logFile.exists()` on a null `logFile` when the constructor could not
        // create the log file; the NullPointerException ends the monitor thread.
        // The translation opens no reader, so the read loop that follows fails
        // its reads as it does for an unopened reader.
        let log_file = match &self.log_file {
            None => return Ok(()),
            Some(log_file) => log_file,
        };
        let mut log_file_exists = false;
        while self.is_running() && !log_file_exists {
            // Check to see if the log file exists that signifies that the process
            // has started
            if log_file.exists() {
                log_file_exists = true;
            } else {
                monitor_tool_kit::sleep(&self.interrupted, self.update_period())?;
            }
        }
        match log_file.open_reader() {
            Ok(reader_id) => *self.log_reader_id.lock().unwrap() = reader_id,
            Err(LogFileError::Lock(e)) => self.handle_lock_exception(Some(&e), true),
            Err(e) => eprintln!("{e}"),
        }
        Ok(())
    }

    /// Java private final `closeLogFileReader`.
    fn close_log_file_reader(&self) {
        let mut log_reader_id = self.log_reader_id.lock().unwrap();
        if log_reader_id
            .as_ref()
            .is_some_and(|log_reader_id| !log_reader_id.is_empty())
        {
            let id = log_reader_id.take().unwrap();
            self.log_file.as_ref().unwrap().close_id(Some(&*id));
        }
    }

    /// Java private final `closeChannel`.
    fn close_channel(&self) {
        // `FileChannel.close()`/`FileInputStream.close()`: dropping the file
        // closes it; neither close reports an error here.
        self.watched_channel.lock().unwrap().take();
        self.stream.lock().unwrap().take();
    }

    /// `watchedChannel.size()`.
    fn watched_channel_size(&self) -> std::io::Result<i64> {
        // Fixed in translation: FileSizeProcessMonitor.java:267 and :300 call
        // `watchedChannel.size()` when `openChannel` could not open the file
        // (it prints the FileNotFoundException and leaves the channel null), and
        // the NullPointerException ends the monitor thread.  The translation
        // reports it as the IOException the surrounding catch prints.
        match self.watched_channel.lock().unwrap().as_ref() {
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "java.lang.NullPointerException: watchedChannel",
            )),
            Some(channel) => channel.metadata().map(|metadata| metadata.len() as i64),
        }
    }

    /// Java private synchronized `setRunning`.
    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }

    /// Java `isRunning`.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// A Java `FileSizeProcessMonitor` subclass instance.
pub struct FileSizeProcessMonitorOf<S: ?Sized + FileSizeProcessMonitorImpl> {
    /// The superclass state.
    pub base: FileSizeProcessMonitor,
    /// The subclass state and overrides.
    pub subclass: S,
}

/// Java inheritance: every base member is reachable on the subclass.
impl<S: ?Sized + FileSizeProcessMonitorImpl> std::ops::Deref for FileSizeProcessMonitorOf<S> {
    type Target = FileSizeProcessMonitor;

    fn deref(&self) -> &FileSizeProcessMonitor {
        &self.base
    }
}

impl<S: FileSizeProcessMonitorImpl> FileSizeProcessMonitorOf<S> {
    /// Runs the Java superclass constructor `FileSizeProcessMonitor(BaseManager,
    /// AxisID, ProcessName)` for a subclass instance whose own fields are
    /// `subclass`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_name: ProcessName,
        subclass: S,
    ) -> Arc<FileSizeProcessMonitorOf<S>> {
        Arc::new_cyclic(|this: &Weak<FileSizeProcessMonitorOf<S>>| {
            let monitor: Weak<dyn Monitor> = this.clone();
            FileSizeProcessMonitorOf {
                base: FileSizeProcessMonitor::new(manager, axis_id, process_name, monitor),
                subclass,
            }
        })
    }
}

impl<S: ?Sized + FileSizeProcessMonitorImpl> FileSizeProcessMonitorOf<S> {
    /// Java private final `initializeProgressBar`.
    fn initialize_progress_bar(&self) {
        self.base
            .tool_kit
            .initialize_progress_bar(&self.subclass.get_title(&self.base), false);
    }

    /// Java final `waitForFile`.  Wait for the new output file to be created.
    /// Make sure it is current by comparing the modification time of the file
    /// to the start time of this function.  Set the process start time to the
    /// first new file modification time since we don't have access to the
    /// file creation time.
    pub fn wait_for_file(&self) -> Result<(), InterruptedException> {
        let base = &self.base;
        // look for signal that the watched file has been backed up in the log file
        base.open_log_file_reader()?;
        let mut watched_file_backed_up = false;
        while base.is_running() && !watched_file_backed_up {
            let mut line: Option<String> = None;
            let log_reader_id = base.log_reader_id.lock().unwrap().clone();
            // `logFile.readLine(logReaderId)`: an unopened (null) id is not
            // locked, so `LogFile.readLine` throws an `UnlockedException`.
            let read = match (&base.log_file, &log_reader_id) {
                (Some(log_file), Some(log_reader_id)) => log_file.read_line(log_reader_id),
                _ => Err(LogFileError::Unlocked(UnlockedException::new_id(
                    None, None,
                ))),
            };
            match read {
                Ok(read) => line = read,
                Err(e) => eprintln!("{e}"),
            }
            match line {
                None => monitor_tool_kit::sleep(&base.interrupted, base.update_period())?,
                Some(line) => {
                    self.subclass.reload_watched_file(base);
                    let watched_file_name = utilities::java_io_file_get_name(
                        &base
                            .watched_file
                            .lock()
                            .unwrap()
                            .as_deref()
                            .map(|file| file.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    );
                    if !base.find_watched_file_name.load(Ordering::SeqCst)
                        || (line.contains(&watched_file_name)
                            && java_lang_string_trim(&line)
                                .to_lowercase()
                                .starts_with("new image file on unit"))
                    {
                        watched_file_backed_up = true;
                    }
                }
            }
        }
        base.close_log_file_reader();

        // Get the watched file
        let mut new_output_file = false;
        while base.is_running() && !new_output_file {
            self.open_channel()?;
            base.process_start_time.store(
                utilities::java_lang_system_current_time_millis(),
                Ordering::SeqCst,
            );
            match base.watched_channel_size() {
                Ok(current_size) => {
                    // Hang out in this loop until the file size is growing, otherwise the
                    // progress bar update with have a divide by zero
                    // The ~ file won't change, so the correct file should be loaded when
                    // the loop ends, however, the first comparison will probably be
                    // between the initial lastSize value of 0 and the ~ size, so ignore
                    // the first comparison
                    let last_size = base.last_size.load(Ordering::SeqCst);
                    if last_size > 0 && current_size > last_size {
                        new_output_file = true;
                    }
                    base.last_size.store(current_size, Ordering::SeqCst);
                }
                Err(except) => eprintln!("{except}"),
            }
            monitor_tool_kit::sleep(&base.interrupted, base.update_period())?;
        }
        Ok(())
    }

    /// Java final `updateProgressBar`.  Watch the file size, comparing it to
    /// the expected completed file size and update the progress bar.
    pub fn update_progress_bar(&self) -> Result<(), InterruptedException> {
        let base = &self.base;
        let axis_id = base.axis_id;
        while base.is_running()
            && base.file_writing.load(Ordering::SeqCst)
            && !base.stop.load(Ordering::SeqCst)
            && base.end_state.lock().unwrap().is_none()
        {
            self.subclass.get_status_from_log(base)?;
            if !base.is_ending() {
                let mut current_length: i64 = 0;
                match base.watched_channel_size() {
                    Ok(current_file_size) => {
                        if current_file_size > 0 {
                            current_length = current_file_size / 1024;
                        }
                    }
                    Err(e) => eprintln!("{e}"),
                }

                let mut fraction_done: f64 = 0.0;
                if current_length > 0 {
                    fraction_done =
                        current_length as f64 / base.n_k_bytes.load(Ordering::SeqCst) as f64;
                }

                let mut percentage = utilities::java_lang_math_round(fraction_done * 100.0);
                // Catch any wierd values before they get displayed
                if percentage < 0 {
                    percentage = 0;
                }
                if percentage > 99 {
                    percentage = 99;
                }

                // Update display
                let elapsed_time = utilities::java_lang_system_current_time_millis()
                    - base.process_start_time.load(Ordering::SeqCst);
                if fraction_done == 0.0 {
                    // When the file size is 0, remaining time can't be calculated. Use elapsed
                    // time instead.
                    let message = "Elapsed time: ".to_string()
                        + &utilities::millis_to_min_and_secs(elapsed_time as f64);
                    base.manager.post_main_panel(Box::new(move |panel| {
                        panel.set_progress_bar_value_int_string_axis_id(0, Some(&message), axis_id);
                    }));
                } else {
                    // Calculate and display the estimated time to completion
                    let mut remaining_time =
                        (elapsed_time as f64 / fraction_done) - elapsed_time as f64;
                    // Avoid negative ETCs.
                    if remaining_time < 0.0 {
                        remaining_time = 0.0;
                    }
                    let message = percentage.to_string()
                        + "%   ETC: "
                        + &utilities::millis_to_min_and_secs(remaining_time);
                    let mut i_current_length =
                        converter::to_integer_from_long(Some(current_length), true);
                    if i_current_length.is_none() {
                        i_current_length = Some(i32::MAX);
                    }
                    let i_current_length = i_current_length.unwrap();
                    base.manager.post_main_panel(Box::new(move |panel| {
                        panel.set_progress_bar_value_int_string_axis_id(i_current_length, Some(&message), axis_id);
                    }));
                }
            }
            monitor_tool_kit::sleep(&base.interrupted, base.update_period())?;
            // TODO(unit): needs etomo/process/MessageReporter.java -
            // `if (messageReporter != null) messageReporter.checkForMessages(manager)`.
        }
        // TODO(unit): needs etomo/process/MessageReporter.java -
        // `if (messageReporter != null) messageReporter.close()`.
        base.close_open_files();
        Ok(())
    }

    /// Java private final `openChannel`.
    fn open_channel(&self) -> Result<(), InterruptedException> {
        let base = &self.base;
        let mut watched_file_exists = false;
        while base.is_running() && !watched_file_exists {
            self.subclass.reload_watched_file(base);
            if base
                .watched_file
                .lock()
                .unwrap()
                .as_deref()
                .is_some_and(Path::exists)
            {
                watched_file_exists = true;
            } else {
                monitor_tool_kit::sleep(&base.interrupted, base.update_period())?;
            }
        }
        base.close_channel();
        let watched_file = base
            .watched_file
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default();
        match std::fs::File::open(&watched_file) {
            Ok(stream) => {
                *base.watched_channel.lock().unwrap() = stream.try_clone().ok();
                *base.stream.lock().unwrap() = Some(stream);
            }
            Err(e) => eprintln!("{e}"),
        }
        Ok(())
    }

    /// The body of Java `run`'s `try` block, including its early `return`.
    fn run_file_size(&self) -> Result<(), RunError> {
        let base = &self.base;
        // Reset the progressBar
        self.initialize_progress_bar();

        // Calculate the expected file size in bytes, initialize the progress bar
        // and set the File object.
        if !self
            .subclass
            .calc_file_size(base)
            .map_err(RunError::CalcFileSize)?
        {
            base.set_process_end_state(ProcessEndState::Done);
            base.close_open_files();
            base.busy_status_mediator.msg_monitor_stopped(base.axis_id);
            return Ok(());
        }
        // Wait for the output file to be backed up and set the process start time
        self.wait_for_file().map_err(RunError::Interrupted)?;
        // Interrupted ??? kill the thread by exiting
        // Periodically update the process bar by checking the size of the file
        let _ = self.update_progress_bar();
        if let Err(e) = monitor_tool_kit::sleep(&base.interrupted, base.update_period()) {
            eprintln!("{e}");
        }
        base.tool_kit.reset();
        base.set_process_end_state(ProcessEndState::Done);
        Ok(())
    }
}

/// What escapes Java `run`'s `try` block to its catch clauses.
enum RunError {
    Interrupted(InterruptedException),
    CalcFileSize(CalcFileSizeError),
}

impl<S: ?Sized + FileSizeProcessMonitorImpl> Monitor for FileSizeProcessMonitorOf<S> {
    /// Java final `run`.
    fn run(&self) {
        let base = &self.base;
        base.set_running(true);
        match self.run_file_size() {
            Ok(()) => {}
            Err(RunError::Interrupted(_)) => {
                base.set_process_end_state(ProcessEndState::Done);
            }
            Err(RunError::CalcFileSize(CalcFileSizeError::InvalidParameter(except))) => {
                eprintln!("{except}");
                base.set_process_end_state(ProcessEndState::Failed);
            }
            Err(RunError::CalcFileSize(CalcFileSizeError::Io(_))) => {
                base.set_process_end_state(ProcessEndState::Failed);
            }
        }
        // finally
        base.close_open_files();
        base.busy_status_mediator.msg_monitor_stopped(base.axis_id);
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

impl<S: ?Sized + FileSizeProcessMonitorImpl> ProcessMonitor for FileSizeProcessMonitorOf<S> {
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
        // Fixed in translation: FileSizeProcessMonitor.java:535 throws an
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

    /// Java final `useMessageReporter`.
    fn use_message_reporter(&self) {
        self.base.use_message_reporter();
    }

    /// Java final `dumpState`.
    fn dump_state(&self) {}

    /// Java final `msgLogFileRenaming(LogFile.Handle, boolean)`; the
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
