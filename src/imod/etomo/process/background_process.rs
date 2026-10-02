//! `IMOD/Etomo/src/etomo/process/BackgroundProcess.java` and
//! `IMOD/Etomo/src/etomo/process/DetachedProcess.java`.
//!
//! Runs one command in the background and tells the process manager when it
//! is done.  Like `ComScriptProcess`, the Java extends `Thread`; the object is
//! shared (`Arc`) between its thread, `AxisProcessData` and the manager.
//!
//! **`DetachedProcess`** extends it to run a command detached from eTomo
//! through the `startprocess` script (processchunks, detached com files).  Its
//! overrides consult an `OutfileProcessMonitor`; that is the `detached` field,
//! set by [`BackgroundProcess::new_detached`].

use super::base_process_manager::BaseProcessManager;
use super::com_script_process::next_thread_name;
use super::monitor::OutfileProcessMonitor;
use super::parse_pid::ParsePID;
use super::process_data::ProcessData;
use super::process_interface::{
    ProcessInterface, ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface,
};
use super::process_messages::{MessageType, ProcessMessages};
use super::system_program::{BackgroundWait, MessagesKind, SystemProgram};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::detached_command_details::DetachedCommandDetails;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Java `POPUP_CHUNK_WARNINGS_DEFAULT`.
const POPUP_CHUNK_WARNINGS_DEFAULT: bool = true;

/// The `DetachedProcess` subclass state.
struct Detached {
    monitor: Option<Arc<dyn OutfileProcessMonitor>>,
    command: Arc<dyn DetachedCommandDetails + Send + Sync>,
    /// Java `subdirName`.
    subdir_name: Mutex<Option<String>>,
    /// Java `shortCommandName`.
    short_command_name: Mutex<String>,
    /// Java `pausing`.
    pausing: AtomicBool,
}

/// The arguments of the Java package-private constructor.
pub struct BackgroundProcessInit {
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
    pub command_array_list: Option<Vec<String>>,
    /// `commandDetails`; when set, `command` is the same object.
    pub is_command_details: bool,
    pub command: Option<Arc<dyn Command + Send + Sync>>,
    pub command_array: Option<Vec<String>>,
    pub process_manager: &'static BaseProcessManager,
    pub process_result_display: Option<ProcessResultDisplayRef>,
    pub process_name: Option<ProcessName>,
    pub process_series: Option<ProcessSeriesRef>,
    pub force_next_process: bool,
    pub popup_chunk_warnings: bool,
    pub managed_process_data: Option<Arc<Mutex<ProcessData>>>,
    pub allow_multi_line_log: bool,
}

impl BackgroundProcessInit {
    /// The defaults every `getInstance` overload shares.
    pub fn new(
        manager: &'static dyn BaseManager,
        process_manager: &'static BaseProcessManager,
        axis_id: AxisID,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
    ) -> BackgroundProcessInit {
        BackgroundProcessInit {
            manager,
            axis_id,
            command_array_list: None,
            is_command_details: false,
            command: None,
            command_array: None,
            process_manager,
            process_result_display: None,
            process_name,
            process_series,
            force_next_process: false,
            popup_chunk_warnings: POPUP_CHUNK_WARNINGS_DEFAULT,
            managed_process_data: None,
            allow_multi_line_log: false,
        }
    }
}

/// Java `BackgroundProcess extends Thread implements SystemProcessInterface`.
pub struct BackgroundProcess {
    name: String,
    this: Weak<BackgroundProcess>,
    command_array_list: Option<Vec<String>>,
    process_data: Arc<Mutex<ProcessData>>,
    process_series: Option<ProcessSeriesRef>,
    process_manager: &'static BaseProcessManager,
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    command: Option<Arc<dyn Command + Send + Sync>>,
    is_command_details: bool,
    command_process_id: Arc<Mutex<String>>,
    force_next_process: bool,
    popup_chunk_warnings: bool,
    allow_multi_line_log: bool,
    command_line: Mutex<Option<String>>,
    command_array: Option<Vec<String>>,
    working_directory: Mutex<Option<PathBuf>>,
    debug: AtomicBool,
    std_output: Mutex<Option<Vec<String>>>,
    std_error: Mutex<Option<Vec<String>>>,
    started: AtomicBool,
    end_state: Mutex<Option<ProcessEndState>>,
    program: Mutex<Option<Arc<SystemProgram>>>,
    process_result_display: Mutex<Option<ProcessResultDisplayRef>>,
    detached: Option<Detached>,
}

impl BackgroundProcess {
    /// Java `BackgroundProcess(BaseManager, AxisID, List<String>,
    /// CommandDetails, Command, String[], BaseProcessManager,
    /// ProcessResultDisplay, ProcessName, ProcessSeries, boolean, boolean,
    /// ProcessData, boolean)`.
    fn construct(
        init: BackgroundProcessInit,
        detached: Option<Detached>,
    ) -> Arc<BackgroundProcess> {
        let manager = init.manager;
        let axis_id = init.axis_id;
        manager
            .get_busy_status_mediator()
            .msg_process_constructed(axis_id);
        // command: `if (command != null)` tests the constructor's `command`
        // parameter, which the `CommandDetails` overloads pass as null.
        let command_array = match &init.command {
            Some(command) if !init.is_command_details => command.get_command_array(),
            _ => init.command_array.clone(),
        };
        // processData
        let process_data = match init.managed_process_data {
            Some(process_data) => process_data,
            None => Arc::new(Mutex::new(ProcessData::get_managed_instance(
                Some(axis_id),
                Some(manager),
                init.process_name.clone(),
            ))),
        };
        {
            let mut data = process_data.lock().unwrap();
            if let Some(display) = &init.process_result_display {
                data.set_display_key(Some(&**display.get()));
            }
            if let Some(process_series) = &init.process_series {
                let process_series = process_series.get().borrow();
                data.set_dialog_type(process_series.get_dialog_type());
                data.set_last_process(
                    &process_series,
                    init.process_name
                        .as_ref()
                        .is_some_and(|name| name.resumable),
                );
            }
        }
        Arc::new_cyclic(|this| BackgroundProcess {
            name: next_thread_name(),
            this: this.clone(),
            command_array_list: init.command_array_list,
            process_data,
            process_series: init.process_series,
            process_manager: init.process_manager,
            axis_id,
            manager,
            command: init.command,
            is_command_details: init.is_command_details,
            command_process_id: Arc::new(Mutex::new(String::new())),
            force_next_process: init.force_next_process,
            popup_chunk_warnings: init.popup_chunk_warnings,
            allow_multi_line_log: init.allow_multi_line_log,
            command_line: Mutex::new(None),
            command_array,
            working_directory: Mutex::new(None),
            debug: AtomicBool::new(false),
            std_output: Mutex::new(None),
            std_error: Mutex::new(None),
            started: AtomicBool::new(false),
            end_state: Mutex::new(None),
            program: Mutex::new(None),
            process_result_display: Mutex::new(init.process_result_display),
            detached,
        })
    }

    /// The Java `getInstance` overloads, which all call the one constructor.
    pub fn get_instance(init: BackgroundProcessInit) -> Arc<BackgroundProcess> {
        BackgroundProcess::construct(init, None)
    }

    /// Java `DetachedProcess(BaseManager, DetachedCommandDetails,
    /// BaseProcessManager, AxisID, OutfileProcessMonitor, ProcessResultDisplay,
    /// ProcessName, ProcessSeries, boolean, ProcessingMethod, ProcessData)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_detached(
        manager: &'static dyn BaseManager,
        command_details: Arc<dyn DetachedCommandDetails + Send + Sync>,
        process_manager: &'static BaseProcessManager,
        axis_id: AxisID,
        monitor: Option<Arc<dyn OutfileProcessMonitor>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
    ) -> Arc<BackgroundProcess> {
        let command: Arc<dyn Command + Send + Sync> = command_details.clone();
        let process = BackgroundProcess::construct(
            BackgroundProcessInit {
                manager,
                axis_id,
                command_array_list: None,
                is_command_details: true,
                command: Some(command),
                command_array: None,
                process_manager,
                process_result_display: process_result_display.clone(),
                process_name,
                process_series,
                force_next_process: false,
                popup_chunk_warnings,
                managed_process_data,
                allow_multi_line_log: false,
            },
            Some(Detached {
                monitor: monitor.clone(),
                command: command_details,
                subdir_name: Mutex::new(None),
                short_command_name: Mutex::new(String::new()),
                pausing: AtomicBool::new(false),
            }),
        );
        process.set_process_result_display(process_result_display);
        process.set_processing_method(processing_method);
        if let Some(monitor) = &monitor {
            process
                .process_data
                .lock()
                .unwrap()
                .set_sub_process_name(monitor.get_sub_process_name().as_deref());
        }
        process
    }

    /// `Thread.start()`.
    pub fn start(&self) {
        let this = self.this.upgrade().expect("BackgroundProcess is alive");
        std::thread::Builder::new()
            .name(self.name.clone())
            .spawn(move || this.run())
            .expect("starting the background process thread");
    }

    /// `Thread.getName()`.
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// The process as the trait object `AxisProcessData` stores.
    pub fn as_process(&self) -> Arc<dyn ProcessInterface> {
        self.this.upgrade().expect("BackgroundProcess is alive")
    }

    /// `DetachedProcess.setSubdirName`.
    pub fn set_subdir_name(&self, input: Option<&str>) {
        if let Some(detached) = &self.detached {
            *detached.subdir_name.lock().unwrap() = input.map(str::to_owned);
            self.process_data.lock().unwrap().set_sub_dir_name(input);
        }
    }

    /// `DetachedProcess.setShortCommandName`.
    pub fn set_short_command_name(&self, input: &str) {
        if let Some(detached) = &self.detached {
            *detached.short_command_name.lock().unwrap() = input.to_owned();
        }
    }

    /// Java `closeOutputImageFile`.
    pub fn close_output_image_file(&self) {
        let Some(command) = &self.command else {
            return;
        };
        self.manager
            .close_stale_file(command.get_output_image_file_key(), Some(self.axis_id));
        self.manager
            .close_stale_file(command.get_output_image_file_key2(), Some(self.axis_id));
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        self.process_data.lock().unwrap().get_process_name()
    }

    /// Java `getProcessResultDisplay`.
    pub fn get_process_result_display(&self) -> Option<ProcessResultDisplayRef> {
        self.process_result_display.lock().unwrap().clone()
    }

    /// Java `isDebug`.
    pub fn is_debug(&self) -> bool {
        self.debug.load(Ordering::SeqCst)
    }

    /// Java `isForceNextProcess`.
    pub fn is_force_next_process(&self) -> bool {
        self.force_next_process
    }

    /// Java `getWorkingDirectory`.
    pub fn get_working_directory(&self) -> Option<PathBuf> {
        self.working_directory.lock().unwrap().clone()
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref()
    }

    /// Java `getCommandDetails`.
    pub fn get_command_details(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref().filter(|_| self.is_command_details)
    }

    /// Java private `getAbbreviatedCommandLine`.
    fn get_abbreviated_command_line(&self) -> String {
        let command_line = self.get_command_line();
        let Some(process_name) = self.process_data.lock().unwrap().get_process_name() else {
            return command_line;
        };
        // Return the first part of the command, which will hopefully show the
        // process name.
        match command_line.find(&process_name.to_string()) {
            None => self.get_command_line_to(3).unwrap_or_default(),
            Some(index) => command_line[index..].to_owned(),
        }
    }

    /// Java `getCommandLine(int endIndex)`.
    ///
    /// The Java loops to `Math.max(endIndex, commandArray.length)`, so a
    /// command array shorter than 3 elements throws
    /// `ArrayIndexOutOfBoundsException` (`BackgroundProcess.java:380`) where
    /// `Math.min` was meant.  Fixed in translation (`BUGS.md`).
    pub fn get_command_line_to(&self, end_index: usize) -> Option<String> {
        let command_array = self.get_command_array()?;
        let mut buffer = String::new();
        for element in command_array
            .iter()
            .take(end_index.min(command_array.len()))
        {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    /// Java `getCommandArray`.
    pub fn get_command_array(&self) -> Option<Vec<String>> {
        if let Some(command_array) = &self.command_array {
            return Some(command_array.clone());
        }
        if let Some(command) = &self.command {
            return command.get_command_array();
        }
        self.command_array_list.clone()
    }

    /// Java `getCommandLine()`: returns the full command line.
    pub fn get_command_line(&self) -> String {
        let mut command_line = self.command_line.lock().unwrap();
        if command_line.is_none() {
            let mut buffer = String::new();
            if let Some(command_array) = &self.command_array {
                for element in command_array {
                    buffer.push_str(&format!("{element} "));
                }
                *command_line = Some(buffer);
            } else if let Some(command) = &self.command {
                *command_line = command
                    .get_command_line()
                    .map(|line| line.trim().to_owned());
            } else if let Some(list) = &self.command_array_list {
                for element in list {
                    buffer.push_str(&format!("{element} "));
                }
                *command_line = Some(buffer);
            }
        }
        match command_line.as_ref() {
            None => panic!("commandLine is null"),
            Some(line) => line.clone(),
        }
    }

    /// Java `getCommandName`: returns command name of the process.
    pub fn get_command_name(&self) -> Option<String> {
        if let Some(command) = &self.command {
            return command.get_command_name();
        }
        if let Some(command_array) = &self.command_array {
            return command_array.first().cloned();
        }
        self.get_command_line()
            .split(|c: char| c.is_whitespace())
            .next()
            .map(str::to_owned)
    }

    /// Java `getCommandAction`.
    pub fn get_command_action(&self) -> Option<String> {
        match self.program.lock().unwrap().as_ref() {
            Some(program) => Some(program.get_command_action()),
            None => self.get_command_name(),
        }
    }

    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&self, working_directory: PathBuf) {
        *self.working_directory.lock().unwrap() = Some(working_directory);
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.store(debug, Ordering::SeqCst);
    }

    /// Java `newProgram`; `DetachedProcess` runs `startprocess`.
    fn new_program(&self) -> bool {
        if let Some(detached) = &self.detached {
            let run_command = match self.create_run_command(detached) {
                Ok(run_command) => run_command,
                Err(message) => {
                    ui_harness::post_message_dialog(
                        Some(self.manager),
                        message,
                        format!(
                            "Can't Run {}",
                            self.get_command_name().unwrap_or_else(|| "null".into())
                        ),
                        None,
                    );
                    return false;
                }
            };
            if !detached.command.is_valid() {
                self.process_done(1);
                return false;
            }
            let Some(monitor) = detached.monitor.clone() else {
                self.process_done(1);
                return false;
            };
            let program = SystemProgram::background(
                self.manager,
                Some(run_command),
                Arc::new(MonitorWait(monitor)),
                self.axis_id,
            );
            program.set_accept_input_while_running(true);
            self.set_program(Arc::new(program));
            return true;
        }
        let command_array = if let Some(command_array) = &self.command_array {
            Some(command_array.clone())
        } else if let Some(command) = &self.command {
            command.get_command_array()
        } else if let Some(list) = &self.command_array_list {
            Some(list.clone())
        } else {
            self.process_done(1);
            return false;
        };
        let program = SystemProgram::new(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            command_array,
            self.axis_id,
            MessagesKind::InstanceAllowMultiLineLog(self.allow_multi_line_log),
        );
        self.set_program(Arc::new(program));
        true
    }

    /// `DetachedProcess.createRunCommand`.
    fn create_run_command(&self, detached: &Detached) -> Result<Vec<String>, String> {
        let python_script_path = etomo_director::INSTANCE.get_python_script_path()
            .unwrap_or_default();
        let mut run_command = vec![
            "python".to_owned(),
            "-u".to_owned(),
            format!("{python_script_path}startprocess"),
            "-o".to_owned(),
        ];
        let monitor = detached.monitor.as_ref().ok_or_else(|| "null".to_owned())?;
        run_command.push(
            monitor
                .get_process_output_file_name()
                .map_err(|e| e.get_message())?,
        );
        if let Some(subdir_name) = detached.subdir_name.lock().unwrap().as_ref() {
            run_command.push("-d".to_owned());
            run_command.push(subdir_name.clone());
        }
        if let Some(command_array) = detached.command.get_command_array() {
            run_command.extend(command_array);
        }
        Ok(run_command)
    }

    /// Java `getStatusString`; `DetachedProcess` asks its monitor.
    pub fn get_status_string(&self) -> Option<String> {
        match &self.detached {
            Some(detached) => detached.monitor.as_ref()?.get_status_string(),
            None => None,
        }
    }

    /// Java `waitForPid`; `DetachedProcess` waits on its monitor.
    fn wait_for_pid(&self) {
        if let Some(detached) = &self.detached
            && let Some(monitor) = detached.monitor.clone()
        {
            let process_data = Arc::clone(&self.process_data);
            let manager = self.manager;
            let axis_id = self.axis_id;
            std::thread::spawn(move || pid_thread(monitor, process_data, manager, axis_id));
            return;
        }
        let Some(program) = self.get_program() else {
            return;
        };
        let parse_pid = ParsePID::new(
            program,
            Arc::clone(&self.command_process_id),
            Some(Arc::clone(&self.process_data)),
        );
        std::thread::spawn(move || parse_pid.run());
    }

    /// Java `run`: execute the command and notify the ProcessManager when it
    /// is done.
    pub fn run(&self) {
        if self.process_series.is_none() {
            utilities::timestamp_command_status(
                Some(&self.get_abbreviated_command_line()),
                Some("started"),
            );
        }
        self.started.store(true, Ordering::SeqCst);
        if !self.new_program() {
            return;
        }
        let program = self.get_program().expect("newProgram set the program");
        program.set_working_directory(self.working_directory.lock().unwrap().clone());
        // Execute the command
        self.wait_for_pid();
        program.run();
        // Get any output from the command
        *self.std_error.lock().unwrap() = program.get_std_error();
        *self.std_output.lock().unwrap() = program.get_std_output();
        // Send a message back to the ProcessManager that this thread is done.
        self.process_done(program.get_exit_value());
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    /// Java `processDone(int)`.
    fn process_done(&self, exit_value: i32) {
        let process_messages = self.get_process_messages();
        let monitor_messages = self.get_monitor_messages();
        // Check to see if the exit value is non-zero
        let end_state = self.get_process_end_state();
        let mut error_found = false;
        if exit_value == 0 {
            // treate any error message as a failure
            // popup error messages from the process
            if let Some(process_messages) = &process_messages
                && !process_messages.is_empty(MessageType::Error)
            {
                error_found = true;
                ui_harness::post_error_message_dialog(
                    Some(self.manager),
                    copy_messages(process_messages),
                    "Process Error".to_owned(),
                    self.axis_id,
                );
            }
            // popup error messages from the monitor
            if let Some(monitor_messages) = &monitor_messages
                && !monitor_messages.is_empty(MessageType::Error)
            {
                error_found = true;
                ui_harness::post_error_message_dialog(
                    Some(self.manager),
                    copy_messages(monitor_messages),
                    "Process Monitor Error".to_owned(),
                    self.axis_id,
                );
            }
            if self.popup_chunk_warnings
                && !error_found
                && let Some(monitor_messages) = &monitor_messages
            {
                let size = monitor_messages.size(MessageType::ChunkWarning);
                if size > 0 {
                    let mut warning_message = ProcessMessages::get_instance();
                    warning_message.add_empty(MessageType::Warning);
                    warning_message
                        .add_message(MessageType::Warning, "<html><U>Warnings Occurred</U>");
                    warning_message
                        .add_message(MessageType::Warning, "<html><U>Chunk warnings:</U>");
                    for i in 0..size {
                        if let Some(message) = monitor_messages.get(MessageType::ChunkWarning, i) {
                            warning_message.add_message(MessageType::Warning, message);
                        }
                    }
                    ui_harness::post_warning_message_dialog(
                        Some(self.manager),
                        warning_message,
                        format!("{} Warning", display_process_name(self.get_process_name())),
                        self.axis_id,
                    );
                }
            }
        } else if end_state != Some(ProcessEndState::Killed)
            && end_state != Some(ProcessEndState::Paused)
        {
            if self.debug.load(Ordering::SeqCst) {
                eprintln!("processDone:exitValue:{exit_value},endState:{end_state:?}");
            }
            error_found = true;
            let std_error = self.std_error.lock().unwrap().clone();
            let std_output = self.std_output.lock().unwrap().clone();
            let mut error_message = ProcessMessages::get_instance();
            error_message.add_message(
                MessageType::Error,
                format!(
                    "<html>Command failed: {}",
                    self.get_abbreviated_command_line()
                ),
            );
            error_message.add_array(
                MessageType::Error,
                "<html><U>Standard error output:</U>",
                std_error.as_deref(),
            );
            error_message.add_from_type(
                MessageType::Error,
                MessageType::ChunkError,
                "<html><U>Chunk errors:</U>",
                monitor_messages.as_ref(),
            );
            error_message.add_from(
                MessageType::Error,
                "<html><U>Monitor error messages:</U>",
                monitor_messages.as_ref(),
            );
            if let Some(std_output) = std_output {
                error_message.add_process_output_lines(Some("Program output:"), std_output);
            }
            // make sure script knows about failure
            self.set_process_end_state(ProcessEndState::Failed);
            // popup error messages
            ui_harness::post_error_message_dialog(
                Some(self.manager),
                error_message,
                format!(
                    "{} terminated",
                    display_process_name(self.get_process_name())
                ),
                self.axis_id,
            );
        } else if end_state == Some(ProcessEndState::Killed)
            && let Some(process_series) = self.process_series.clone()
        {
            let axis_id = self.axis_id;
            let display = self.get_process_result_display();
            event_queue::invoke_later(move || {
                process_series
                    .get()
                    .borrow_mut()
                    .kill_series(axis_id, display);
            });
        }
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!(
                "BackgroundProcess:processDone:endState:{end_state:?}, exitValue:{exit_value}, errorFound:{error_found}"
            );
        }
        self.process_done_error(exit_value, error_found);
    }

    /// Java `processDone(int, boolean)`; `DetachedProcess` calls the
    /// `DetachedProcess` overload of `msgProcessDone`.
    fn process_done_error(&self, exit_value: i32, error_found: bool) {
        if self.detached.is_some() {
            self.process_manager
                .msg_process_done_detached(self, exit_value, error_found);
            return;
        }
        self.process_manager.msg_process_done_background(
            self,
            exit_value,
            error_found,
            self.popup_chunk_warnings,
        );
    }

    /// Java `getProcessMessages`.
    pub fn get_process_messages(&self) -> Option<ProcessMessages> {
        let program = self.get_program()?;
        let messages = program.get_process_messages();
        let mut copy = ProcessMessages::get_instance();
        copy.add_process_messages(&messages);
        Some(copy)
    }

    /// Java `getMonitorMessages`; `DetachedProcess` asks its monitor.
    fn get_monitor_messages(&self) -> Option<ProcessMessages> {
        let detached = self.detached.as_ref()?;
        let monitor = detached.monitor.as_ref()?;
        let messages = monitor.get_process_messages()?;
        let mut copy = ProcessMessages::get_instance();
        copy.add_process_messages(&messages);
        Some(copy)
    }

    /// Java `getProgram`.
    pub fn get_program(&self) -> Option<Arc<SystemProgram>> {
        self.program.lock().unwrap().clone()
    }

    /// Java `getProcessEndState`; `DetachedProcess` asks its monitor.
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        if let Some(detached) = &self.detached
            && let Some(monitor) = &detached.monitor
        {
            return monitor.get_process_end_state();
        }
        *self.end_state.lock().unwrap()
    }

    /// Java `isPausing`.
    pub fn is_pausing(&self) -> bool {
        self.detached
            .as_ref()
            .is_some_and(|detached| detached.pausing.load(Ordering::SeqCst))
    }

    /// Java `setProgram`.
    fn set_program(&self, program: Arc<SystemProgram>) {
        *self.program.lock().unwrap() = Some(program);
    }

    /// Java `isDetached`-style test used by `BaseProcessManager.postProcess`.
    pub fn is_detached(&self) -> bool {
        self.detached.is_some()
    }
}

/// `DetachedProcess.PidThread.run`.
fn pid_thread(
    monitor: Arc<dyn OutfileProcessMonitor>,
    process_data: Arc<Mutex<ProcessData>>,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
) {
    // wait until monitor is running
    while !monitor.is_process_running() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // wait for pid
    let mut pid = None;
    while pid.is_none() && monitor.is_process_running() {
        pid = monitor.get_pid();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if let Some(pid) = pid {
        let mut data = process_data.lock().unwrap();
        data.set_pid(Some(&pid));
        if !data.is_empty() {
            // The shared `ProcessData` is the `Storable` (its `Mutex`); `store` takes
            // the lock itself, so it is released first.
            drop(data);
            // Save the process data for this process so that it will already
            // be saved if there is a crash.
            manager.save_storable(Some(axis_id), Some(&*process_data));
        }
    }
}

/// `BackgroundSystemProgram`'s view of the `DetachedProcessMonitor`.
struct MonitorWait(Arc<dyn OutfileProcessMonitor>);

impl BackgroundWait for MonitorWait {
    fn is_process_running(&self) -> bool {
        self.0.is_process_running()
    }
    fn is_process_end_state_done(&self) -> bool {
        self.0.get_process_end_state() == Some(ProcessEndState::Done)
    }
}

/// `processName.toString()` in a string concatenation.
fn display_process_name(process_name: Option<ProcessName>) -> String {
    process_name.map_or_else(|| "null".to_owned(), |name| name.to_string())
}

impl ProcessInterface for BackgroundProcess {
    fn get_process_series(&self) -> Option<ProcessSeriesRef> {
        self.process_series.clone()
    }

    /// Returns false if the process will stop if Etomo exits.
    /// `DetachedProcess`: always true, the process is detached.
    fn is_nohup(&self) -> bool {
        self.detached.is_some()
    }

    fn get_process_data(&self) -> Option<Arc<Mutex<ProcessData>>> {
        Some(Arc::clone(&self.process_data))
    }

    fn pause(&self, axis_id: AxisID) -> bool {
        match &self.detached {
            Some(detached) => {
                detached.pausing.store(true, Ordering::SeqCst);
                match &detached.monitor {
                    Some(monitor) => monitor.pause(self, axis_id),
                    None => false,
                }
            }
            None => panic!("pause is not valid in BackgroundProcess"),
        }
    }

    fn kill(&self, axis_id: AxisID) {
        if let Some(detached) = &self.detached {
            if let Some(monitor) = &detached.monitor {
                monitor.kill(self, axis_id);
            }
            return;
        }
        self.process_manager.signal_kill(self, axis_id);
    }
}

impl SystemProcessInterface for BackgroundProcess {
    fn to_source_string(&self) -> String {
        self.get_command_line()
    }

    fn get_std_output(&self) -> Option<Vec<String>> {
        let program = self.get_program()?;
        let std_output = program.get_std_output();
        *self.std_output.lock().unwrap() = std_output.clone();
        std_output
    }

    fn get_std_error(&self) -> Option<Vec<String>> {
        let program = self.get_program()?;
        let std_error = program.get_std_error();
        *self.std_error.lock().unwrap() = std_error.clone();
        std_error
    }

    fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    fn is_done(&self) -> bool {
        match self.get_program() {
            None => false,
            Some(program) => program.is_done(),
        }
    }

    fn get_shell_process_id(&self) -> String {
        if let Some(detached) = &self.detached
            && let Some(monitor) = &detached.monitor
        {
            return monitor.get_pid().unwrap_or_else(|| "null".to_owned());
        }
        self.command_process_id.lock().unwrap().clone()
    }

    fn notify_killed(&self) {
        self.set_process_end_state(ProcessEndState::Killed);
        if let Some(detached) = &self.detached
            && let Some(monitor) = &detached.monitor
            && let Some(end_state) = self.get_process_end_state()
        {
            monitor.end_monitor(end_state);
        }
    }

    fn set_process_end_state(&self, end_state: ProcessEndState) {
        {
            let mut current = self.end_state.lock().unwrap();
            *current = Some(match *current {
                None => end_state,
                Some(existing) => ProcessEndState::precedence(existing, end_state),
            });
        }
        if let Some(detached) = &self.detached
            && let Some(monitor) = &detached.monitor
        {
            monitor.set_process_end_state(end_state);
        }
    }

    fn signal_kill(&self, axis_id: AxisID) {
        self.process_manager.signal_kill(self, axis_id);
    }

    fn set_process_result_display(&self, process_result_display: Option<ProcessResultDisplayRef>) {
        *self.process_result_display.lock().unwrap() = process_result_display;
    }

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
        self.process_data.lock().unwrap().reset();
    }
}

/// A copy of `messages` for the event dispatch thread, where the Java hands
/// the dialog the same object.
fn copy_messages(messages: &ProcessMessages) -> ProcessMessages {
    let mut copy = ProcessMessages::get_instance();
    copy.add_process_messages(messages);
    copy
}
