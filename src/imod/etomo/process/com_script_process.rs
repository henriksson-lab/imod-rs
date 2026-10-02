//! `IMOD/Etomo/src/etomo/process/ComScriptProcess.java` and
//! `IMOD/Etomo/src/etomo/process/OutfileComScriptProcess.java`.
//!
//! Runs an IMOD command file: `vmstopy` converts it to a Python script, and
//! the script is fed to `python -u` on its standard input.  The process
//! thread is the Java `Thread` this class extends; it is started with
//! [`ComScriptProcess::start`] and runs [`ComScriptProcess::run`].
//!
//! Both external steps go through `SystemProgram`, so they are started the
//! way `system_program.rs` describes: `vmstopy` is our translated command and
//! the script runs in our command-file runner (`runcom -P -S`), which prints
//! the `Runcom PID:` line `ParsePID` reads.
//!
//! **Constructors.**  The Java has ten constructor overloads that differ in
//! which of `command`/`commandDetails`/`processResultDisplay`/`fileType`/
//! `processingMethod` they take and in how they decide `resumable`; they are
//! the named constructors below over one [`ComScriptProcessInit`].
//!
//! **`OutfileComScriptProcess`** overrides `buildLogFile`,
//! `runMsgComScriptDone`, `kill`, `signalKill`, `pause` and `parse`; it is
//! the `outfile_monitor` field, set by [`ComScriptProcess::new_outfile`].
//!
//! **`BackgroundComScriptProcess`** overrides `closeOutputImageFile`,
//! `isComScriptBusy`, `renameFiles`, `execPython`, `notifyKilled`, `parse` and
//! `getShellProcessID`; its state is the `background` field, set by
//! `BackgroundComScriptProcess::new` (`background_com_script_process.rs`).

use super::background_com_script_process::BackgroundComScriptProcess;
use super::base_process_manager::BaseProcessManager;
use super::monitor::{DetachedProcessMonitor, ProcessMonitor};
use super::parse_pid::ParsePID;
use super::process_data::ProcessData;
use super::process_interface::{
    ProcessInterface, ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface,
};
use super::process_messages::{MessageType, ProcessMessages};
use super::system_program::{MessagesKind, SystemProgram};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::storage::log_file::{Handle, LockException, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Java `Thread`'s default name, `"Thread-" + threadInitNumber++`: every
/// process object that extends `Thread` takes the next number when it is
/// constructed, and the managers identify a finished process by it.
pub fn next_thread_name() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!("Thread-{}", NEXT.fetch_add(1, Ordering::SeqCst))
}

/// The arguments the ten Java constructors fill in; see the module comment.
pub struct ComScriptProcessInit {
    pub manager: &'static dyn BaseManager,
    pub com_script: String,
    pub process_manager: &'static BaseProcessManager,
    pub axis_id: AxisID,
    pub watched_file_name: Option<String>,
    pub process_monitor: Option<Arc<dyn ProcessMonitor>>,
    pub process_result_display: Option<ProcessResultDisplayRef>,
    pub process_series: Option<ProcessSeriesRef>,
    pub command: Option<Arc<dyn Command + Send + Sync>>,
    /// Java `commandDetails`: set by the `CommandDetails` overloads, where
    /// `command` is the same object.
    pub is_command_details: bool,
    /// `resumable` for the overloads that take it; the others read
    /// `command.getProcessName().resumable`.
    pub resumable: Option<bool>,
    pub file_type: Option<&'static FileType>,
    pub processing_method: Option<ProcessingMethod>,
}

/// Java `ComScriptProcess extends Thread implements SystemProcessInterface`.
pub struct ComScriptProcess {
    /// `Thread.getName()`.
    name: String,
    this: Weak<ComScriptProcess>,
    indeterminate_mode: bool,
    allow_multi_line_log: bool,
    com_script_name: String,
    working_directory: Mutex<Option<PathBuf>>,
    process_manager: &'static BaseProcessManager,
    debug: AtomicBool,
    system_program: Mutex<Option<Arc<SystemProgram>>>,
    vmstopy: Mutex<Option<Arc<SystemProgram>>>,
    csh_process_id: Arc<Mutex<String>>,
    axis_id: AxisID,
    watched_file_name: Option<String>,
    command: Option<Arc<dyn Command + Send + Sync>>,
    is_command_details: bool,
    process_data: Arc<Mutex<ProcessData>>,
    /// Use reconnect to update dialog - clear processData once process is
    /// done.
    reconnect_when_not_running: bool,
    started: AtomicBool,
    error: AtomicBool,
    process_monitor: Option<Arc<dyn ProcessMonitor>>,
    /// Used when processMonitor is null.
    end_state: Mutex<Option<ProcessEndState>>,
    manager: &'static dyn BaseManager,
    emergency_monitor: Arc<EmergencyMonitor>,
    process_messages: Mutex<ProcessMessages>,
    process_series: Option<ProcessSeriesRef>,
    process_result_display: Mutex<Option<ProcessResultDisplayRef>>,
    parse_log_file: AtomicBool,
    non_blocking: AtomicBool,
    process_name: Option<ProcessName>,
    /// Set in initialize.
    log_file: Option<Arc<Handle>>,
    /// `OutfileComScriptProcess.monitor`; `None` for a plain
    /// `ComScriptProcess`.
    outfile_monitor: Option<Arc<dyn DetachedProcessMonitor>>,
    /// The `BackgroundComScriptProcess` subclass state; unset for a plain
    /// `ComScriptProcess`.
    background: OnceLock<BackgroundComScriptProcess>,
}

impl ComScriptProcess {
    /// The common constructor body.
    fn construct(
        init: ComScriptProcessInit,
        process_name: Option<ProcessName>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
        reconnect_when_not_running: bool,
        indeterminate_mode: bool,
        outfile_monitor: Option<Arc<dyn DetachedProcessMonitor>>,
    ) -> Arc<ComScriptProcess> {
        let manager = init.manager;
        let axis_id = init.axis_id;
        let emergency_monitor = manager.get_emergency_monitor(Some(axis_id));
        manager
            .get_busy_status_mediator()
            .msg_process_constructed(axis_id);
        let own_process_name = process_name
            .clone()
            .or_else(|| ProcessName::get_instance_with_axis(&init.com_script, axis_id));
        let process_data = match managed_process_data {
            Some(process_data) => process_data,
            None => Arc::new(Mutex::new(ProcessData::get_managed_instance(
                Some(axis_id),
                Some(manager),
                own_process_name,
            ))),
        };
        {
            let mut data = process_data.lock().unwrap();
            if let Some(display) = &init.process_result_display {
                data.set_display_key(Some(&**display.get()));
            }
            data.set_processing_method(init.processing_method);
            if let Some(process_series) = &init.process_series {
                let process_series = process_series.get().borrow();
                data.set_dialog_type(process_series.get_dialog_type());
                let resumable = match init.resumable {
                    Some(resumable) => resumable,
                    None => init
                        .command
                        .as_ref()
                        .and_then(|command| command.get_process_name())
                        .is_some_and(|name| name.resumable),
                };
                data.set_last_process(&process_series, resumable);
            }
        }
        let file_type = init.file_type;
        Arc::new_cyclic(|this| {
            let mut process = ComScriptProcess {
                name: next_thread_name(),
                this: this.clone(),
                indeterminate_mode,
                allow_multi_line_log: false,
                com_script_name: init.com_script,
                working_directory: Mutex::new(None),
                process_manager: init.process_manager,
                debug: AtomicBool::new(false),
                system_program: Mutex::new(None),
                vmstopy: Mutex::new(None),
                csh_process_id: Arc::new(Mutex::new(String::new())),
                axis_id,
                watched_file_name: init.watched_file_name,
                command: init.command,
                is_command_details: init.is_command_details,
                process_data,
                reconnect_when_not_running,
                started: AtomicBool::new(false),
                error: AtomicBool::new(false),
                process_monitor: init.process_monitor,
                end_state: Mutex::new(None),
                manager,
                emergency_monitor,
                process_messages: Mutex::new(ProcessMessages::get_instance()),
                process_series: init.process_series,
                process_result_display: Mutex::new(init.process_result_display),
                parse_log_file: AtomicBool::new(true),
                non_blocking: AtomicBool::new(false),
                process_name,
                log_file: None,
                outfile_monitor,
                background: OnceLock::new(),
            };
            process.initialize(file_type);
            process
        })
    }

    /// Java `ComScriptProcess(BaseManager, String, BaseProcessManager, AxisID,
    /// String, ProcessMonitor, ProcessResultDisplay, ProcessSeries, boolean)`
    /// and the overloads without a command.
    pub fn new(init: ComScriptProcessInit) -> Arc<ComScriptProcess> {
        ComScriptProcess::construct(init, None, None, false, false, None)
    }

    /// Java `OutfileComScriptProcess(BaseManager, String, BaseProcessManager,
    /// AxisID, DetachedProcessMonitor, Command, FileType, ProcessName, boolean,
    /// ProcessData, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_outfile(
        manager: &'static dyn BaseManager,
        com_script: String,
        process_manager: &'static BaseProcessManager,
        axis_id: AxisID,
        monitor: Arc<dyn DetachedProcessMonitor>,
        command: Option<Arc<dyn Command + Send + Sync>>,
        file_type: Option<&'static FileType>,
        process_name: Option<ProcessName>,
        reconnect_when_not_running: bool,
        managed_process_data: Arc<Mutex<ProcessData>>,
        indeterminate_mode: bool,
    ) -> Arc<ComScriptProcess> {
        let process_monitor: Arc<dyn ProcessMonitor> = monitor.clone();
        ComScriptProcess::construct(
            ComScriptProcessInit {
                manager,
                com_script,
                process_manager,
                axis_id,
                watched_file_name: None,
                process_monitor: Some(process_monitor),
                process_result_display: None,
                process_series: None,
                command,
                is_command_details: false,
                resumable: Some(false),
                file_type,
                processing_method: None,
            },
            process_name,
            Some(managed_process_data),
            reconnect_when_not_running,
            indeterminate_mode,
            Some(monitor),
        )
    }

    /// Makes this process a `BackgroundComScriptProcess` (its constructor,
    /// right after `super(...)`).
    pub(crate) fn set_background(&self, background: BackgroundComScriptProcess) {
        let _ = self.background.set(background);
    }

    /// `Thread.start()`: runs [`ComScriptProcess::run`] on a new thread.
    pub fn start(&self) {
        let this = self.this.upgrade().expect("ComScriptProcess is alive");
        std::thread::Builder::new()
            .name(self.name.clone())
            .spawn(move || this.run())
            .expect("starting the com script thread");
    }

    /// `Thread.getName()`.
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// Java `closeOutputImageFile`.
    pub fn close_output_image_file(&self) {
        if let Some(background) = self.background.get() {
            background.close_output_image_file();
            return;
        }
        let display = self.process_result_display.lock().unwrap().clone();
        if self.command.is_none() && display.is_none() {
            return;
        }
        let mut file_key = None;
        // Get the output file from the button if available.  The display is
        // a Swing button, read on the event dispatch thread.
        if let Some(display) = display.as_ref().filter(|display| display.is_owner_thread()) {
            file_key = display.get().get_output_image_file_key();
        }
        if file_key.is_none()
            && let Some(command) = &self.command
        {
            file_key = command.get_output_image_file_key();
        }
        self.manager.close_stale_file(file_key, Some(self.axis_id));
        if let Some(command) = &self.command {
            self.manager
                .close_stale_file(command.get_output_image_file_key2(), Some(self.axis_id));
        }
    }

    /// Java private `buildLogFile`; `OutfileComScriptProcess` prefers the
    /// file type.
    fn build_log_file(&self, file_type: Option<&'static FileType>) -> Option<Arc<Handle>> {
        if self.outfile_monitor.is_some() {
            if let Some(file_type) = file_type {
                return self.build_log_file_from_file_type(file_type);
            }
            return self.build_log_file_from_process_name();
        }
        if self.get_process_name().is_some() {
            return self.build_log_file_from_process_name();
        }
        file_type.and_then(|file_type| self.build_log_file_from_file_type(file_type))
    }

    /// Java `buildLogFileFromProcessName`.
    fn build_log_file_from_process_name(&self) -> Option<Arc<Handle>> {
        let result = match (
            self.manager.get_property_user_dir(),
            self.get_process_name(),
        ) {
            (Some(user_dir), Some(process_name)) => LogFile::get_instance_process_name(
                &user_dir,
                self.axis_id,
                process_name,
                Some(Arc::clone(&self.emergency_monitor)),
            )
            .map_err(|e| e.get_message()),
            _ => Err("null".to_owned()),
        };
        match result {
            Ok(handle) => Some(handle),
            Err(message) => {
                eprintln!("{message}");
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Unable to create log file.\n{message}"),
                    "Com Script Log Failure".to_owned(),
                    None,
                );
                None
            }
        }
    }

    /// Java `buildLogFileFromFileType`.
    fn build_log_file_from_file_type(&self, file_type: &'static FileType) -> Option<Arc<Handle>> {
        let result = match (
            self.manager.get_property_user_dir(),
            Some(file_type.get_root(Some(self.manager), Some(self.axis_id))),
        ) {
            (Some(user_dir), Some(root)) => LogFile::get_instance_name(
                &user_dir,
                self.axis_id,
                &root,
                Some(Arc::clone(&self.emergency_monitor)),
            )
            .map_err(|e| e.get_message()),
            _ => Err("null".to_owned()),
        };
        match result {
            Ok(handle) => Some(handle),
            Err(message) => {
                eprintln!("{message}");
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Unable to create log file.\n{message}"),
                    "Com Script Log Failure".to_owned(),
                    None,
                );
                None
            }
        }
    }

    /// Java private `initialize`.
    fn initialize(&mut self, file_type: Option<&'static FileType>) {
        self.log_file = self.build_log_file(file_type);
        if let (Some(command), Some(process_monitor)) = (&self.command, &self.process_monitor)
            && command.is_message_reporter()
        {
            process_monitor.use_message_reporter();
        }
    }

    /// Java `setWorkingDirectory`: set the working directory in which the com
    /// script is to be run.
    pub fn set_working_directory(&self, working_directory: PathBuf) {
        *self.working_directory.lock().unwrap() = Some(working_directory);
    }

    /// Java `getProcessResultDisplay`.
    pub fn get_process_result_display(&self) -> Option<ProcessResultDisplayRef> {
        self.process_result_display.lock().unwrap().clone()
    }

    /// Java `getLogFile`.
    pub fn get_log_file(&self) -> Option<Arc<Handle>> {
        self.log_file.clone()
    }

    /// Java `runMsgComScriptDone`; `OutfileComScriptProcess` waits for its
    /// monitor first and calls the `OutfileComScriptProcess` overload.
    fn run_msg_com_script_done(&self, exit_value: i32) {
        if let Some(monitor) = &self.outfile_monitor {
            let wait_for_monitor = 50;
            // Wait for the monitor to complete.
            for i in 0..wait_for_monitor * 2 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if !monitor.is_process_running() {
                    break;
                }
                if exit_value != 0 && i == wait_for_monitor {
                    monitor.set_process_end_state(ProcessEndState::Failed);
                }
            }
            self.process_manager.msg_com_script_done_outfile(
                self,
                exit_value,
                self.get_non_blocking(),
            );
            return;
        }
        if self.debug.load(Ordering::SeqCst) {
            eprintln!(
                "runMsgComScriptDone:exitValue:{exit_value},comScriptName:{}",
                self.com_script_name
            );
        }
        self.process_manager.msg_com_script_done(
            self.axis_id,
            self,
            exit_value,
            self.non_blocking.load(Ordering::SeqCst),
        );
        if self.reconnect_when_not_running {
            // The reconnect won't check to see if the process is run - remove
            // process data for completed process.
            self.reset_process_data();
        }
    }

    /// Java `getNonBlocking`.
    pub fn get_non_blocking(&self) -> bool {
        self.non_blocking.load(Ordering::SeqCst)
    }

    /// Java `run`: execute the specified com script.
    pub fn run(&self) {
        let body = || -> Result<(), LockException> {
            if self.process_series.is_none() {
                let process_data = self.process_data.lock().unwrap();
                utilities::timestamp_process_name_subprocess(
                    None,
                    process_data.get_process_name(),
                    process_data.get_sub_process_name().as_deref(),
                    Some("started"),
                );
            }
            if !self.non_blocking.load(Ordering::SeqCst) && self.is_com_script_busy() {
                self.error.store(true, Ordering::SeqCst);
                self.process_messages.lock().unwrap().add_message(
                    MessageType::Error,
                    format!("{} is already running", self.com_script_name),
                );
                self.run_msg_com_script_done(1);
                return Ok(());
            }
            if !self.rename_files()? {
                // Hopefully this is only a problem for the monitor. Try running
                // the process.
                eprintln!(
                    "Unable to remove previous log file(s).  Process monitor may not work correctly."
                );
            }
            // Convert the com script to a sequence of csh commands
            let commands = match self.vms_to_py() {
                Ok(commands) => commands,
                Err(except) => {
                    self.error.store(true, Ordering::SeqCst);
                    let vmstopy = self.vmstopy.lock().unwrap().clone();
                    if let Some(message) = except {
                        let mut messages = self.process_messages.lock().unwrap();
                        messages.add_message(MessageType::Error, message);
                        if vmstopy.is_none() {
                            messages.add_message(MessageType::Error, "vmstopy is null");
                        }
                    }
                    self.run_msg_com_script_done(
                        vmstopy.map_or(-1, |vmstopy| vmstopy.get_exit_value()),
                    );
                    return Ok(());
                }
            };
            // Execute the csh commands
            self.started.store(true, Ordering::SeqCst);
            if let Err(except) = self.exec_python(commands) {
                if let Some(message) = except {
                    self.process_messages
                        .lock()
                        .unwrap()
                        .add_message(MessageType::Error, message);
                }
            }
            if let Err(except) = self.parse() {
                match except {
                    ParseError::Lock(lock) => return Err(lock),
                    ParseError::Other(message) => self
                        .process_messages
                        .lock()
                        .unwrap()
                        .add_message(MessageType::Error, message),
                }
            }
            Ok(())
        };
        match body() {
            Ok(()) if self.error.load(Ordering::SeqCst) && !self.started.load(Ordering::SeqCst) => {
                // runMsgComScriptDone already ran on an early return
                return;
            }
            Ok(()) => {}
            Err(_lock) => self.set_process_end_state(ProcessEndState::FileLockFailure),
        }
        // Send a message back to the ProcessManager that this thread is done.
        let system_program = self.system_program.lock().unwrap().clone();
        match system_program {
            None => self.run_msg_com_script_done(1),
            Some(system_program) => self.run_msg_com_script_done(system_program.get_exit_value()),
        }
    }

    /// Java `renameFiles()`.
    fn rename_files(&self) -> Result<bool, LockException> {
        if let Some(background) = self.background.get() {
            return background.rename_files(self);
        }
        let working_directory = self.working_directory.lock().unwrap().clone();
        self.rename_files_with(
            self.watched_file_name.as_deref(),
            working_directory.as_deref(),
            self.log_file.as_ref(),
            self.indeterminate_mode,
        )
    }

    /// Java `resetProcessData`.
    pub fn reset_process_data_impl(&self) {
        self.process_data.lock().unwrap().reset();
    }

    /// Java `renameFiles(String, File, LogFile.Handle, boolean)`: true if
    /// backup or delete worked.
    pub(crate) fn rename_files_with(
        &self,
        watched_file_name: Option<&str>,
        working_directory: Option<&Path>,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) -> Result<bool, LockException> {
        if let Some(process_monitor) = &self.process_monitor {
            process_monitor.msg_log_file_renaming_handle(log_file, indeterminate_mode);
        }
        let mut retval = self.rename_file(log_file);
        if let (Some(process_monitor), Some(watched_file_name)) =
            (&self.process_monitor, watched_file_name)
        {
            process_monitor.msg_log_file_renaming_name(watched_file_name);
            match working_directory.map(|directory| {
                LogFile::get_instance_dir(
                    directory,
                    watched_file_name,
                    Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
                )
            }) {
                Some(Ok(handle)) => {
                    retval = self.rename_file(Some(&handle)) || retval;
                }
                _ => {
                    eprintln!("Error: Invalid watched file:{watched_file_name}");
                }
            }
        }
        if let Some(process_monitor) = &self.process_monitor {
            if retval {
                process_monitor.msg_log_file_renamed();
            } else {
                process_monitor.msg_log_file_renaming_failed();
            }
        }
        Ok(retval)
    }

    /// Java `handleLockException`.
    pub(crate) fn handle_lock_exception(&self, lock_exception: &LockException, do_popup: bool) {
        self.emergency_monitor.alert(Some(lock_exception), do_popup);
        self.set_process_end_state(ProcessEndState::FileLockFailure);
    }

    /// Java private `renameFile`.
    fn rename_file(&self, log_file: Option<&Arc<Handle>>) -> bool {
        let Some(log_file) = log_file else {
            return false;
        };
        // Rename the logfile so that any log file monitor does not get confused
        // by an existing log file
        let success = match log_file.backup() {
            Ok(success) => success,
            Err(LogFileError::Lock(e)) => {
                self.handle_lock_exception(&e, true);
                false
            }
            Err(_) => false,
        };
        if success {
            return true;
        }
        eprintln!(
            "Warning: unable to back up {}.  Attempting to delete.",
            log_file.get_name()
        );
        let success = match log_file.delete() {
            Ok(success) => success,
            Err(LogFileError::Lock(e)) => {
                self.handle_lock_exception(&e, false);
                false
            }
            Err(e) => {
                eprintln!("{e}");
                false
            }
        };
        if success {
            return true;
        }
        eprintln!(
            "Warning: unable to back up or delete {}.",
            log_file.get_name()
        );
        false
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref()
    }

    /// Java `getCommandDetails`.
    pub fn get_command_details(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref().filter(|_| self.is_command_details)
    }

    /// Java `getCommandAction`.
    pub fn get_command_action(&self) -> String {
        match self.system_program.lock().unwrap().as_ref() {
            None => self.get_com_script_name(),
            Some(system_program) => system_program.get_command_action(),
        }
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        if self.process_name.is_some() {
            return self.process_name.clone();
        }
        ProcessName::get_instance_with_axis(&self.com_script_name, self.axis_id)
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getProcessMessages`: the messages, if any from the run method.
    pub fn get_process_messages(&self) -> std::sync::MutexGuard<'_, ProcessMessages> {
        self.process_messages.lock().unwrap()
    }

    /// Java `getDebug`/`isDebug`.
    pub fn is_debug(&self) -> bool {
        self.debug.load(Ordering::SeqCst)
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, state: bool) {
        self.debug.store(state, Ordering::SeqCst);
    }

    /// Java static `parseBaseName`: extract the basename from a filename given
    /// the filename and the expected extension.
    pub fn parse_base_name(filename: &str, extension: &str) -> Option<String> {
        match filename.find(extension) {
            Some(idx_extension) if idx_extension > 0 => Some(filename[..idx_extension].to_owned()),
            _ => None,
        }
    }

    /// Java `execPython`: execute the python commands.  `Err(Some(message))`
    /// is the `LogFileException`/`IOException` arm, `Err(None)` the empty
    /// `SystemProcessException`.
    fn exec_python(&self, commands: Option<Vec<String>>) -> Result<(), Option<String>> {
        if let Some(background) = self.background.get() {
            return background.exec_python(self, commands);
        }
        // Do not use the -e flag for tcsh since David's scripts handle the
        // failure of commands and then report appropriately.
        let system_program = Arc::new(SystemProgram::new(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(vec!["python".to_owned(), "-u".to_owned()]),
            self.axis_id,
            MessagesKind::InstanceAllowMultiLineLog(self.allow_multi_line_log),
        ));
        *self.system_program.lock().unwrap() = Some(Arc::clone(&system_program));
        system_program.set_working_directory(self.working_directory.lock().unwrap().clone());
        system_program.set_std_input(commands);
        let parse_pid = ParsePID::new(
            Arc::clone(&system_program),
            Arc::clone(&self.csh_process_id),
            Some(Arc::clone(&self.process_data)),
        );
        std::thread::spawn(move || parse_pid.run());
        // make sure nothing else is writing or backing up the log file
        let log_writing_id = match self
            .log_file
            .as_ref()
            .map(|log_file| log_file.open_for_writing())
        {
            Some(Ok(id)) => Some(id),
            Some(Err(LogFileError::Lock(e))) => {
                self.handle_lock_exception(&e, false);
                return Ok(());
            }
            Some(Err(e)) => return Err(Some(e.get_message())),
            None => None,
        };
        system_program.run();
        // release the log file
        if let (Some(log_file), Some(log_writing_id)) = (&self.log_file, &log_writing_id) {
            log_file.close_id(Some(log_writing_id));
        }
        // Check the exit value, if it is non zero, parse the warnings and
        // errors from the log file.
        if system_program.get_exit_value() != 0 {
            return Err(None);
        }
        Ok(())
    }

    /// Java private `vmsToPy`: convert the com script to a sequence of python
    /// commands.  `Err(Some(message))` is the `IOException` arm,
    /// `Err(None)` the `SystemProcessException` one.
    fn vms_to_py(&self) -> Result<Option<Vec<String>>, Option<String>> {
        let working_directory = self
            .working_directory
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| Some("null".to_owned()))?;
        let log_file = self
            .log_file
            .as_ref()
            .ok_or_else(|| Some("null".to_owned()))?;
        let python_script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .unwrap_or_default();
        // vmstopy doesn't use stdin
        let command = vec![
            "python".to_owned(),
            "-u".to_owned(),
            format!("{python_script_path}vmstopy"),
            format!(
                "{}/{}",
                utilities::java_io_file_get_absolute_path(&working_directory.to_string_lossy()),
                self.com_script_name
            ),
            log_file.get_name(),
        ];
        let vmstopy = Arc::new(SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(command),
            self.axis_id,
        ));
        *self.vmstopy.lock().unwrap() = Some(Arc::clone(&vmstopy));
        vmstopy.set_working_directory(Some(working_directory));
        vmstopy.run();
        if vmstopy.get_exit_value() != 0 {
            self.process_messages.lock().unwrap().add_message(
                MessageType::Error,
                format!("Running vmstopy against {} failed", self.com_script_name),
            );
            return Err(None);
        }
        Ok(vmstopy.get_std_output())
    }

    /// Java `getComScriptName`.
    pub fn get_com_script_name(&self) -> String {
        self.com_script_name.clone()
    }

    /// Java `setNonBlocking`.
    pub fn set_non_blocking(&self) {
        self.non_blocking.store(true, Ordering::SeqCst);
    }

    /// Java `setParseLogFile`.
    pub fn set_parse_log_file(&self, input: bool) {
        self.parse_log_file.store(input, Ordering::SeqCst);
    }

    /// Java `parse()`.
    fn parse(&self) -> Result<(), ParseError> {
        if let Some(background) = self.background.get() {
            return background.parse(self);
        }
        self.parse_named(&self.com_script_name.clone(), true)
    }

    /// Java `parse(String, boolean)`: parse the log file for warnings.
    /// `OutfileComScriptProcess` does not parse the log file.
    pub(crate) fn parse_named(&self, name: &str, must_exist: bool) -> Result<(), ParseError> {
        if self.outfile_monitor.is_some() {
            return Ok(());
        }
        let mut log_file_to_parse = self.log_file.clone();
        if !self.parse_log_file.load(Ordering::SeqCst) {
            let working_directory = self.working_directory.lock().unwrap().clone();
            let file = working_directory.unwrap_or_default().join(format!(
                "{}.log",
                ComScriptProcess::parse_base_name(name, ".com").unwrap_or_else(|| "null".into())
            ));
            log_file_to_parse = Some(
                LogFile::get_instance_file(Some(&file), Some(Arc::clone(&self.emergency_monitor)))
                    .map_err(ParseError::from)?,
            );
        }
        let Some(log_file_to_parse) = log_file_to_parse else {
            return Err(ParseError::Other("null".to_owned()));
        };
        if !log_file_to_parse.exists() && !must_exist {
            return Ok(());
        }
        let mut messages = self.process_messages.lock().unwrap();
        messages
            .add_process_output_file(log_file_to_parse.get_file())
            .map_err(|e| ParseError::Other(e.to_string()))?;
        messages.print_all();
        Ok(())
    }

    /// Java `isError`: true if an error was found before the com script
    /// process started running.
    pub fn is_error(&self) -> bool {
        self.error.load(Ordering::SeqCst)
    }

    /// Java `isComScriptBusy`: always false in `ComScriptProcess`.
    fn is_com_script_busy(&self) -> bool {
        if let Some(background) = self.background.get() {
            return background.is_com_script_busy(self);
        }
        false
    }

    /// Java `getProcessEndState`.
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        match &self.process_monitor {
            None => *self.end_state.lock().unwrap(),
            Some(process_monitor) => process_monitor.get_process_end_state(),
        }
    }

    /// Java `getMonitor`.
    pub fn get_monitor(&self) -> Option<&Arc<dyn ProcessMonitor>> {
        self.process_monitor.as_ref()
    }

    /// Java `getWorkingDirectory`.
    pub fn get_working_directory(&self) -> Option<PathBuf> {
        self.working_directory.lock().unwrap().clone()
    }

    /// Java `getWatchedFileName`.
    pub fn get_watched_file_name(&self) -> Option<String> {
        self.watched_file_name.clone()
    }

    /// Java `setSystemProgram`.
    pub fn set_system_program(&self, system_program: Arc<SystemProgram>) {
        *self.system_program.lock().unwrap() = Some(system_program);
    }

    /// `OutfileComScriptProcess.getMonitorProcessMessages`.
    pub fn get_monitor_process_messages(&self) -> Option<ProcessMessages> {
        let monitor = self.outfile_monitor.as_ref()?;
        let messages = monitor.get_process_messages()?;
        let mut copy = ProcessMessages::get_instance();
        copy.add_process_messages(&messages);
        Some(copy)
    }

    /// `OutfileComScriptProcess.getStatusString`.
    pub fn get_status_string(&self) -> Option<String> {
        self.outfile_monitor.as_ref()?.get_status_string()
    }

    /// The process as the trait object `AxisProcessData` stores.
    pub fn as_process(&self) -> Arc<dyn ProcessInterface> {
        self.this.upgrade().expect("ComScriptProcess is alive")
    }
}

/// The exceptions `parse` declares.
pub(crate) enum ParseError {
    Lock(LockException),
    Other(String),
}

impl From<LogFileError> for ParseError {
    fn from(error: LogFileError) -> ParseError {
        match error {
            LogFileError::Lock(lock) => ParseError::Lock(lock),
            other => ParseError::Other(other.get_message()),
        }
    }
}

impl ProcessInterface for ComScriptProcess {
    fn get_process_series(&self) -> Option<ProcessSeriesRef> {
        self.process_series.clone()
    }

    /// Always returns true because all comscripts a piped to tcsh and do not
    /// disconnect when etomo exits.
    fn is_nohup(&self) -> bool {
        true
    }

    fn get_process_data(&self) -> Option<Arc<Mutex<ProcessData>>> {
        Some(Arc::clone(&self.process_data))
    }

    fn pause(&self, axis_id: AxisID) -> bool {
        if let Some(monitor) = &self.outfile_monitor {
            return monitor.pause(self, axis_id);
        }
        panic!("pause is not used by any ComScriptProcess");
    }

    fn kill(&self, axis_id: AxisID) {
        if let Some(monitor) = &self.outfile_monitor {
            monitor.kill(self, axis_id);
            return;
        }
        self.process_manager.signal_kill(self, axis_id);
    }
}

impl SystemProcessInterface for ComScriptProcess {
    fn to_source_string(&self) -> String {
        match self.system_program.lock().unwrap().as_ref() {
            None => self.get_com_script_name(),
            Some(system_program) => system_program.get_command_line(),
        }
    }

    fn get_std_output(&self) -> Option<Vec<String>> {
        self.system_program
            .lock()
            .unwrap()
            .as_ref()?
            .get_std_output()
    }

    fn get_std_error(&self) -> Option<Vec<String>> {
        self.system_program
            .lock()
            .unwrap()
            .as_ref()?
            .get_std_error()
    }

    /// Returns true if the com script process is running, this does not
    /// include vmstocsh or vmstopy process.
    fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    fn is_done(&self) -> bool {
        match self.system_program.lock().unwrap().as_ref() {
            None => false,
            Some(system_program) => system_program.is_done(),
        }
    }

    fn get_shell_process_id(&self) -> String {
        if let Some(background) = self.background.get() {
            return background.get_shell_process_id();
        }
        self.csh_process_id.lock().unwrap().clone()
    }

    fn notify_killed(&self) {
        if let Some(background) = self.background.get() {
            background.notify_killed(self);
            return;
        }
        self.set_process_end_state(ProcessEndState::Killed);
    }

    fn set_process_end_state(&self, end_state: ProcessEndState) {
        match &self.process_monitor {
            None => *self.end_state.lock().unwrap() = Some(end_state),
            Some(process_monitor) => process_monitor.set_process_end_state(end_state),
        }
    }

    fn signal_kill(&self, axis_id: AxisID) {
        if self.outfile_monitor.is_some() {
            return;
        }
        self.process_manager.signal_kill(self, axis_id);
    }

    fn set_process_result_display(&self, process_result_display: Option<ProcessResultDisplayRef>) {
        *self.process_result_display.lock().unwrap() = process_result_display;
    }

    /// Gets a computerMap and immediately sends it to processData.
    fn set_computer_map(&self, computer_map: Option<BTreeMap<String, String>>) {
        self.process_data
            .lock()
            .unwrap()
            .set_computer_map(computer_map);
    }

    fn set_secondary_queue(&self, secondary_queue: Option<&str>) {
        self.process_data
            .lock()
            .unwrap()
            .set_secondary_queue(secondary_queue);
    }

    fn set_processing_method(&self, processing_method: Option<ProcessingMethod>) {
        self.process_data
            .lock()
            .unwrap()
            .set_processing_method(processing_method);
    }

    fn reset_process_data(&self) {
        self.reset_process_data_impl();
    }
}
