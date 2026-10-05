//! `IMOD/Etomo/src/etomo/process/ReconnectProcess.java` and
//! `IMOD/Etomo/src/etomo/process/LoggedReconnectProcess.java`.
//!
//! Reconnect process: reattaches eTomo to a process that was started the last
//! time eTomo ran (tilt, processchunks), following it through its monitor and
//! its saved `ProcessData` until it ends, then reporting to the process
//! manager as a finished process would.
//!
//! **Shape.**  The Java object is a `Runnable` shared by the thread running it,
//! its monitor, `AxisProcessData` and the processing-method mediator, so it is
//! held in an `Arc` and every method takes `&self`; the fields Java assigns
//! after construction sit behind locks.  [`ReconnectProcess::run`] is the
//! thread's body.
//!
//! **`LoggedReconnectProcess`** extends it, overriding `msgDone` (it calls the
//! other `msgReconnectDone` overload) and `getProcessMessages` (it prefers the
//! monitor's messages); that is the `logged` field, set by
//! [`ReconnectProcess::new_logged`], its constructor.
//!
//! **Threads.**  The processing-method mediator is an event-dispatch-thread
//! object: `run` registers and deregisters with it there
//! (`util/event_queue.rs`), as `ProcesschunksProcessMonitor` does.

use super::base_process_manager::BaseProcessManager;
use super::monitor::ProcessMonitor;
use super::process_data::ProcessData;
use super::process_interface::{
    ProcessInterface, ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface,
};
use super::process_messages::{MessageType, ProcessMessages};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::log_file::{Handle, LockException, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Java `ReconnectProcess implements SystemProcessInterface, Runnable`.
pub struct ReconnectProcess {
    /// Java `this`.
    this: Weak<ReconnectProcess>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,

    /// Java package-private final `processManager`.
    pub(crate) process_manager: &'static BaseProcessManager,
    /// Java package-private final `monitor`.
    pub(crate) monitor: Option<Arc<dyn ProcessMonitor>>,
    /// Java private final `processData`.
    process_data: Option<Arc<Mutex<ProcessData>>>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java package-private final `popupChunkWarnings`.
    pub(crate) popup_chunk_warnings: bool,
    /// Java private final `reconnectWhenNotRunning`.  Use reconnect to update
    /// dialog - clear processData once process is done.
    reconnect_when_not_running: bool,
    /// Java package-private final `noPopupWithoutMessage`.  Prevent the done
    /// handled from popping up an error dialog without an error message.
    pub(crate) no_popup_without_message: bool,

    /// Java private `processSeries`.
    process_series: Option<ProcessSeriesRef>,
    /// Java private `endState`.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java private `processResultDisplay`.
    process_result_display: Mutex<Option<ProcessResultDisplayRef>>,
    /// Java private `messages`.
    messages: Mutex<Option<ProcessMessages>>,
    /// Java private `logFile`.
    log_file: Mutex<Option<Arc<Handle>>>,
    /// Java private `logSuccessTag`.
    log_success_tag: Mutex<Option<String>>,
    /// Java private `monitorControl`.  When true the process is being
    /// controlled by the monitor.
    monitor_control: AtomicBool,
    /// Java private `lockException` (never assigned in the source).
    #[allow(dead_code)]
    lock_exception: Mutex<Option<LockException>>,
    /// True for a `LoggedReconnectProcess`; see the module comment.
    logged: bool,
}

impl ReconnectProcess {
    /// Java package-private `getProcessMonitor`.
    pub fn get_process_monitor(&self) -> Option<Arc<dyn ProcessMonitor>> {
        self.monitor.clone()
    }

    /// Java package-private `dumpState(int)`.  Currently handles levels<=2.
    /// Add levels parameter to dumpState calls to handle levels>2.
    pub fn dump_state(&self, mut levels: i32) {
        eprint!("[");
        levels -= 1;
        if levels > 0 {
            eprintln!("manager:");
            self.manager.dump_state();
            eprintln!(",processManager:");
            self.process_manager.dump_state();
            eprintln!(",monitor:");
            if let Some(monitor) = &self.monitor {
                monitor.dump_state();
            }
            eprintln!(",processData:");
            if let Some(process_data) = &self.process_data {
                process_data.lock().unwrap().dump_state();
            }
            eprintln!(",axisID:");
            self.axis_id.dump_state();
            eprintln!(",processSeries:");
            // The series is an event-dispatch-thread object; it is dumped only
            // when this runs there.
            if let Some(process_series) = &self.process_series
                && process_series.is_owner_thread()
            {
                process_series.get().borrow().dump_state();
            }
            eprintln!(",endState:");
            if let Some(end_state) = *self.end_state.lock().unwrap() {
                end_state.dump_state();
            }
            eprintln!(",processResultDisplay:");
            // Likewise the display (a Swing button).
            if let Some(display) = self.process_result_display.lock().unwrap().as_ref()
                && display.is_owner_thread()
            {
                display.get().dump_state();
            }
            eprintln!(",messages:");
            if let Some(messages) = self.messages.lock().unwrap().as_ref() {
                messages.dump_state();
            }
            eprintln!(",logFile:");
            if let Some(log_file) = self.log_file.lock().unwrap().as_ref() {
                log_file.dump_state();
            }
        }
        eprintln!(
            ",logSuccessTag:{}]",
            self.log_success_tag
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null")
        );
    }

    /// Java package-private `ReconnectProcess(BaseManager, BaseProcessManager,
    /// ProcessMonitor, ProcessData, AxisID, ProcessSeries, boolean, boolean,
    /// boolean)`; `logged` selects the `LoggedReconnectProcess` subclass.
    #[allow(clippy::too_many_arguments)]
    fn construct(
        manager: &'static dyn BaseManager,
        process_manager: &'static BaseProcessManager,
        monitor: Option<Arc<dyn ProcessMonitor>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        reconnect_when_not_running: bool,
        no_popup_without_message: bool,
        logged: bool,
    ) -> Arc<ReconnectProcess> {
        let process = Arc::new_cyclic(|this| ReconnectProcess {
            this: this.clone(),
            manager,
            process_manager,
            monitor,
            process_data,
            axis_id,
            popup_chunk_warnings,
            reconnect_when_not_running,
            no_popup_without_message,
            process_series,
            end_state: Mutex::new(None),
            process_result_display: Mutex::new(None),
            messages: Mutex::new(None),
            log_file: Mutex::new(None),
            log_success_tag: Mutex::new(None),
            monitor_control: AtomicBool::new(false),
            lock_exception: Mutex::new(None),
            logged,
        });
        manager
            .get_busy_status_mediator()
            .msg_process_constructed(axis_id);
        process
    }

    /// Java static `getInstance(BaseManager, BaseProcessManager, ProcessMonitor,
    /// ProcessData, AxisID, ProcessSeries) throws FileException, IOException`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        process_manager: &'static BaseProcessManager,
        monitor: Option<Arc<dyn ProcessMonitor>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<Arc<ReconnectProcess>, LogFileError> {
        let instance = ReconnectProcess::construct(
            manager,
            process_manager,
            monitor,
            process_data.clone(),
            axis_id,
            process_series,
            true,
            false,
            false,
            false,
        );
        match &process_data {
            Some(process_data) => {
                let process_name = process_data.lock().unwrap().get_process_name();
                // Upstream bug fixed in translation (ReconnectProcess.java:136): a
                // saved ProcessData without a process name throws
                // NullPointerException building the log file name; here the
                // reconnect has no log file, as with no ProcessData.
                *instance.log_file.lock().unwrap() = match process_name {
                    Some(process_name) => Some(LogFile::get_instance_process_name(
                        &manager.get_property_user_dir().unwrap_or_default(),
                        axis_id,
                        process_name,
                        Some(manager.get_emergency_monitor(Some(axis_id))),
                    )?),
                    None => None,
                };
            }
            None => {
                *instance.log_file.lock().unwrap() = None;
            }
        }
        Ok(instance)
    }

    /// Java static `getLogInstance(BaseManager, BaseProcessManager,
    /// ProcessMonitor, ProcessData, AxisID, String logFileName, String
    /// logSuccessTag, ConstStringProperty subDirName, ProcessSeries, boolean,
    /// boolean) throws FileException, FileNotFoundException, IOException`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_log_instance(
        manager: &'static dyn BaseManager,
        process_manager: &'static BaseProcessManager,
        monitor: Option<Arc<dyn ProcessMonitor>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: AxisID,
        log_file_name: &str,
        log_success_tag: Option<&str>,
        sub_dir_name: Option<&str>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        reconnect_when_not_running: bool,
    ) -> Result<Arc<ReconnectProcess>, LogFileError> {
        let instance = ReconnectProcess::construct(
            manager,
            process_manager,
            monitor,
            process_data,
            axis_id,
            process_series,
            popup_chunk_warnings,
            reconnect_when_not_running,
            false,
            false,
        );
        instance.init_log_instance(log_file_name, log_success_tag, sub_dir_name)?;
        Ok(instance)
    }

    /// Java `LoggedReconnectProcess(BaseManager, BaseProcessManager,
    /// ProcessMonitor, ProcessData, AxisID, String logFileName, String
    /// logSuccessTag, ConstStringProperty subDirName, ProcessSeries, boolean,
    /// boolean, boolean) throws FileException, FileNotFoundException,
    /// IOException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_logged(
        manager: &'static dyn BaseManager,
        process_manager: &'static BaseProcessManager,
        monitor: Option<Arc<dyn ProcessMonitor>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: AxisID,
        log_file_name: &str,
        log_success_tag: Option<&str>,
        sub_dir_name: Option<&str>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        reconnect_when_not_running: bool,
        no_popup_without_message: bool,
    ) -> Result<Arc<ReconnectProcess>, LogFileError> {
        let instance = ReconnectProcess::construct(
            manager,
            process_manager,
            monitor,
            process_data,
            axis_id,
            process_series,
            popup_chunk_warnings,
            reconnect_when_not_running,
            no_popup_without_message,
            true,
        );
        instance.init_log_instance(log_file_name, log_success_tag, sub_dir_name)?;
        Ok(instance)
    }

    /// Java package-private `initLogInstance(String, String,
    /// ConstStringProperty)`.  `sub_dir_name` is the property's value.
    pub fn init_log_instance(
        &self,
        log_file_name: &str,
        log_success_tag: Option<&str>,
        sub_dir_name: Option<&str>,
    ) -> Result<(), LogFileError> {
        let user_dir = self.manager.get_property_user_dir().unwrap_or_default();
        let log_file = match sub_dir_name {
            Some(sub_dir_name) if !sub_dir_name.is_empty() => LogFile::get_instance_dir(
                &Path::new(&user_dir).join(sub_dir_name),
                log_file_name,
                Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
            )?,
            _ => LogFile::get_instance_user_dir(
                &user_dir,
                log_file_name,
                Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
            )?,
        };
        *self.log_file.lock().unwrap() = Some(log_file);
        *self.log_success_tag.lock().unwrap() = log_success_tag.map(str::to_owned);
        self.monitor_control.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// The process as the trait object `AxisProcessData` stores.
    pub fn as_process(&self) -> Arc<dyn ProcessInterface> {
        self.this.upgrade().expect("ReconnectProcess is alive")
    }

    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        let process_data = self.process_data.as_ref()?;
        process_data.lock().unwrap().get_processing_method()
    }

    /// Java private `continueRun`: decide whether the run() function should
    /// continue.
    fn continue_run(&self) -> bool {
        // The reconnect has to be controlled by either monitoring the actual
        // process via processData, or relying on the monitor. Otherwise there
        // is no reliable way to end the reconnect. And if
        // reconnectWhenNotRunning is set, then it's impossible to know when to
        // end without the monitor.
        let on_different_host = self
            .process_data
            .as_ref()
            .map(|process_data| process_data.lock().unwrap().is_on_different_host());
        if self.monitor.is_none()
            && (self.reconnect_when_not_running
                || self.process_data.is_none()
                || on_different_host == Some(true))
        {
            eprintln!("ERROR: Monitor required.");
            return false;
        }
        // If possible, use the actual process.
        if !self.reconnect_when_not_running && on_different_host == Some(false) {
            return self
                .process_data
                .as_ref()
                .is_some_and(|process_data| process_data.lock().unwrap().is_running());
        }
        // When the reconnect is not dependent on whether the process is still
        // running, or processData.isRunning() can't be used, then rely on the
        // monitor and end state.
        let end_state = self.get_process_end_state();
        if let Some(monitor) = &self.monitor {
            return monitor.is_running() || end_state.is_none();
        }
        end_state.is_none()
    }

    /// Java `run`.
    pub fn run(&self) {
        if !self.continue_run() {
            return;
        }
        if self.process_series.is_none()
            && let Some(process_data) = &self.process_data
        {
            let process_data = process_data.lock().unwrap();
            utilities::timestamp_process_name_subprocess(
                None,
                process_data.get_process_name(),
                process_data.get_sub_process_name().as_deref(),
                Some("started"),
            );
        }
        // `ProcessingMethodMediator mediator =
        // manager.getProcessingMethodMediator(axisID); mediator.register(this)`,
        // on the event dispatch thread.
        let this = self.this.upgrade();
        let manager = self.manager;
        let axis_id = self.axis_id;
        let mediator: Option<Arc<EdtRef<ProcessingMethodMediator>>> =
            event_queue::invoke_and_wait(move || {
                manager
                    .get_processing_method_mediator(Some(axis_id))
                    .map(|mediator| Arc::new(EdtRef::new(mediator)))
            });
        // Upstream bug fixed in translation (ReconnectProcess.java:247): a
        // manager without a mediator throws NullPointerException here; here
        // the reconnect runs without registering.
        if let Some(mediator) = &mediator {
            let mediator = Arc::clone(mediator);
            let this = this.clone();
            event_queue::invoke_and_wait(move || {
                mediator.get().register_reconnect_process(this.as_ref());
            });
        }

        // `new Thread(monitor).start()`; a null monitor makes a thread that
        // does nothing.
        if let Some(monitor) = &self.monitor {
            let monitor = Arc::clone(monitor);
            std::thread::spawn(move || monitor.run());
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        let mut log_writing_id = None;
        let result = (|| -> Result<(), LogFileError> {
            if !self.monitor_control.load(Ordering::SeqCst) {
                // make sure nothing else is writing or backing up the log file
                if let Some(log_file) = self.log_file.lock().unwrap().as_ref() {
                    log_writing_id = Some(log_file.open_for_writing()?);
                }
            }
            // MonitorControl and reconnectWhenNotRunning are always both true or
            // both false. Unnecessary to use both.
            while self.continue_run() {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Ok(())
        })();
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{}", e.get_message()),
        }
        if !self.monitor_control.load(Ordering::SeqCst) {
            // release the log file
            if let Some(log_file) = self.log_file.lock().unwrap().as_ref() {
                log_file.close_id(log_writing_id.as_deref());
            }
            if let Some(monitor) = &self.monitor {
                monitor.stop();
                // Upstream bug fixed in translation (ReconnectProcess.java:284):
                // the Java waits `while (!monitor.isRunning())`, which never
                // ends once the stopped monitor has finished (a monitor sets
                // running to false as it exits), so the reconnect thread hangs
                // and never reports the process done.  Here it waits while the
                // monitor is still running, as the loop below does.
                while monitor.is_running() {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
        let mut messages = match self.log_success_tag.lock().unwrap().clone() {
            None => ProcessMessages::get_instance_success_tags(
                Some(self.manager),
                self.axis_id,
                Some("Reconstruction of"),
                Some("slices complete."),
            ),
            Some(log_success_tag) => ProcessMessages::get_instance_success_tag(
                Some(self.manager),
                self.axis_id,
                Some(&log_success_tag),
            ),
        };
        // `messages.addProcessOutput(logFile)`.
        // Upstream bug fixed in translation (ReconnectProcess.java:300): a null
        // log file throws NullPointerException in addProcessOutput; here no
        // output is parsed.
        if let Some(log_file) = self.log_file.lock().unwrap().clone() {
            match messages.add_process_output_log_file(log_file) {
                Ok(()) => {}
                // catch (final LockException e) {}
                Err(LogFileError::Lock(_)) => {}
                // catch (final LogFileException | IOException e)
                Err(e) => eprintln!("{e}"),
            }
        }
        let mut exit_value = 0;
        if !messages.is_empty(Some(MessageType::Error)) || !messages.is_success() {
            exit_value = 1;
        }
        *self.messages.lock().unwrap() = Some(messages);
        if let Some(monitor) = &self.monitor {
            while monitor.is_running() {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
        self.msg_done(exit_value);
        // `mediator.deregister(this)`, on the event dispatch thread.
        if let Some(mediator) = mediator {
            event_queue::invoke_and_wait(move || {
                mediator.get().deregister_reconnect_process(this.as_ref());
            });
        }
        if self.reconnect_when_not_running {
            // The next reconnect won't check to see if the process is run -
            // remove process data for completed process.
            self.reset_process_data();
        }
    }

    /// Java package-private `msgDone`; `LoggedReconnectProcess` calls the other
    /// `msgReconnectDone` overload.
    fn msg_done(&self, exit_value: i32) {
        if self.logged {
            self.process_manager.msg_reconnect_done_logged(
                self,
                exit_value,
                self.popup_chunk_warnings,
                self.no_popup_without_message,
            );
            return;
        }
        self.process_manager.msg_reconnect_done(
            self.axis_id,
            self,
            exit_value,
            self.popup_chunk_warnings,
        );
    }

    /// Java package-private `getProcessMessages`; `LoggedReconnectProcess`
    /// prefers the monitor's.  A copy: the Java hands the dialogs the object
    /// itself, and the dialogs run on the event dispatch thread.
    pub fn get_process_messages(&self) -> Option<ProcessMessages> {
        if self.logged
            && let Some(monitor) = &self.monitor
        {
            return monitor
                .get_process_messages()
                .map(|messages| messages.clone());
        }
        let messages = self.messages.lock().unwrap();
        let copy = messages.as_ref()?.clone();
        Some(copy)
    }

    /// Java final package-private `getProcessEndState`.
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        match &self.monitor {
            None => *self.end_state.lock().unwrap(),
            Some(monitor) => monitor.get_process_end_state(),
        }
    }

    /// Java final package-private `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java final package-private `getProcessResultDisplay`.
    pub fn get_process_result_display(&self) -> Option<ProcessResultDisplayRef> {
        self.process_result_display.lock().unwrap().clone()
    }

    /// Java `toString`.
    pub fn to_source_string_impl(&self) -> String {
        if let Some(process_data) = &self.process_data
            && let Some(process_name) = process_data.lock().unwrap().get_process_name()
        {
            // `processName.getText()`
            return process_name.to_string();
        }
        if let Some(log_file) = self.log_file.lock().unwrap().as_ref() {
            return log_file.get_name();
        }
        // `super.toString()`
        format!(
            "etomo.process.ReconnectProcess@{:x}",
            self as *const _ as usize
        )
    }
}

impl ProcessInterface for ReconnectProcess {
    fn get_process_series(&self) -> Option<ProcessSeriesRef> {
        self.process_series.clone()
    }

    /// Always true because always true in ComScriptProcess and we are only
    /// doing tilt so far.
    fn is_nohup(&self) -> bool {
        true
    }

    fn get_process_data(&self) -> Option<Arc<Mutex<ProcessData>>> {
        self.process_data.clone()
    }

    fn pause(&self, axis_id: AxisID) -> bool {
        match &self.monitor {
            None => {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    "Unable to pause.",
                    "Function Not Available",
                    None,
                );
                false
            }
            Some(monitor) => monitor.pause(self, axis_id),
        }
    }

    fn kill(&self, axis_id: AxisID) {
        if let Some(monitor) = &self.monitor {
            monitor.kill(self, axis_id);
        }
        // processManager.signalKill(this, axisID);
    }
}

impl SystemProcessInterface for ReconnectProcess {
    fn to_source_string(&self) -> String {
        self.to_source_string_impl()
    }

    /// Return the log file as the standard output.
    fn get_std_output(&self) -> Option<Vec<String>> {
        let log_file = self.log_file.lock().unwrap().clone();
        let mut reader_id = None;
        if let Some(log_file) = &log_file {
            match log_file.open_reader() {
                Ok(id) => reader_id = id,
                Err(LogFileError::Lock(_)) => return None,
                Err(LogFileError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    return None;
                }
                Err(e) => {
                    eprintln!("{}", e.get_message());
                    return None;
                }
            }
        }
        let mut log = Vec::new();
        if let (Some(log_file), Some(reader_id)) = (&log_file, &reader_id) {
            loop {
                match log_file.read_line(reader_id) {
                    Ok(Some(line)) => log.push(line),
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("{}", e.get_message());
                        break;
                    }
                }
            }
        }
        if let Some(log_file) = &log_file {
            log_file.close_id(reader_id.as_deref());
        }
        if log.is_empty() {
            return None;
        }
        Some(log)
    }

    /// No access to standard error.
    fn get_std_error(&self) -> Option<Vec<String>> {
        None
    }

    /// Always true because we reconnecting to an existing process.
    fn is_started(&self) -> bool {
        true
    }

    fn is_done(&self) -> bool {
        match &self.monitor {
            None => false,
            Some(monitor) => !monitor.is_running(),
        }
    }

    /// Upstream bug fixed in translation (ReconnectProcess.java:329): a
    /// reconnect without ProcessData throws NullPointerException; here the
    /// shell process ID is empty.
    fn get_shell_process_id(&self) -> String {
        self.process_data
            .as_ref()
            .and_then(|process_data| process_data.lock().unwrap().get_pid())
            .unwrap_or_default()
    }

    fn notify_killed(&self) {
        self.set_process_end_state(ProcessEndState::Killed);
    }

    fn set_process_end_state(&self, end_state: ProcessEndState) {
        match &self.monitor {
            None => *self.end_state.lock().unwrap() = Some(end_state),
            Some(monitor) => monitor.set_process_end_state(end_state),
        }
    }

    fn signal_kill(&self, axis_id: AxisID) {
        self.process_manager.signal_kill(self, axis_id);
    }

    fn set_process_result_display(&self, process_result_display: Option<ProcessResultDisplayRef>) {
        *self.process_result_display.lock().unwrap() = process_result_display;
    }

    /// In a reconnect the computerMap is loaded from the data file.  Setting
    /// it here would override it.  Should never happen because the process
    /// monitor should not set the computer map in the process during a
    /// reconnect.
    fn set_computer_map(&self, _computer_map: Option<BTreeMap<String, String>>) {}

    fn set_secondary_queue(&self, _secondary_queue: Option<&str>) {}

    fn set_processing_method(&self, _processing_method: Option<ProcessingMethod>) {
        // already have processing method from process data
    }

    fn reset_process_data(&self) {
        if let Some(process_data) = &self.process_data {
            process_data.lock().unwrap().reset();
        }
    }
}
