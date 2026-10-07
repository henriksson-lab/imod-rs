//! `IMOD/Etomo/src/etomo/process/SystemProgram.java` and
//! `IMOD/Etomo/src/etomo/process/BackgroundSystemProgram.java`.
//!
//! SystemProgram provides a class to execute programs under the host
//! operating system.  The class provides access to stdin, stdout and stderr
//! streams and implements the Runnable interface so that it can be threaded.
//!
//! **Shape.**  The Java object is shared between the thread running `run`
//! and the threads polling it (`ParsePID`, the process monitors, the
//! manager), so every mutable field sits behind its own lock and every method
//! takes `&self`; callers hold it in an `Arc`.  The two output buffers are
//! `Arc<Mutex<OutputBufferManager>>`, the reader threads' shared lists.
//!
//! `BackgroundSystemProgram` overrides only `waitForProcess` and
//! `getProcessExitValue`; it is the [`SystemProgram::background`]
//! constructor, which records the `DetachedProcessMonitor` those overrides
//! consult.
//!
//! **The `Runtime.exec` boundary.**  [`runtime_exec`] stands in for the JDK's
//! `Runtime.getRuntime().exec(cmdarray, envp, dir)`, including its `PATH`
//! lookup, and is where this crate's single `imod` binary replaces the
//! per-program executables of an IMOD installation: a command array naming
//! one of our commands (directly, as `$IMOD_DIR/bin/<name>`, or as
//! `python [-u] <path>/<name>` for a translated Python script) runs our
//! binary with `argv[0]` set to the command's link name, which is how the
//! launcher dispatches (`CLAUDE.md`, "Source mirroring").  The `vmstopy`
//! script that `ComScriptProcess` pipes into `python -u` runs through our
//! command-file runner (`runcom -P -S`).  Everything else is started as the
//! Java starts it.

use super::output_buffer_manager::OutputBufferManager;
use super::process_messages::ProcessMessages;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

/// What `BackgroundSystemProgram` consults in its two overrides; the
/// `DetachedProcessMonitor` methods it calls.
pub trait BackgroundWait: Send + Sync {
    /// `DetachedProcessMonitor.isProcessRunning`.
    fn is_process_running(&self) -> bool;
    /// `ProcessMonitor.getProcessEndState` is `ProcessEndState.DONE`.
    fn is_process_end_state_done(&self) -> bool;
}

/// Java `SystemProgram`.
pub struct SystemProgram {
    property_user_dir: Option<String>,
    manager: Option<&'static dyn BaseManager>,
    command_array: Mutex<Option<Vec<String>>>,
    axis_id: AxisID,
    process_messages: Mutex<ProcessMessages>,
    debug: Mutex<DebugLevel>,
    exit_value: AtomicI32,
    std_input: Mutex<Option<Vec<String>>>,
    stdout: Mutex<Option<Arc<Mutex<OutputBufferManager>>>>,
    stderr: Mutex<Option<Arc<Mutex<OutputBufferManager>>>>,
    working_directory: Mutex<Option<PathBuf>>,
    exception_message: Mutex<String>,
    started: AtomicBool,
    done: AtomicBool,
    run_timestamp: Mutex<Option<SystemTime>>,
    /// Java `cmdInputStream` / `cmdInBuffer`.
    cmd_in_buffer: Mutex<Option<std::io::BufWriter<ChildStdin>>>,
    accept_input_while_running: AtomicBool,
    command_line: Mutex<Option<String>>,
    /// Java `process`: the child's id, which `destroy` signals.
    process: Mutex<Option<u32>>,
    collect_output: AtomicBool,
    command_action: Mutex<Option<String>>,
    /// `BackgroundSystemProgram.monitor`; `None` for a plain `SystemProgram`.
    background_monitor: Option<Arc<dyn BackgroundWait>>,
}

/// `SystemProgram(BaseManager, String, List<String>, AxisID)` and the other
/// constructors differ only in which `ProcessMessages` factory they call.
#[derive(Clone, Copy, Debug)]
pub enum MessagesKind {
    /// `ProcessMessages.getInstance(manager, axisID)`.
    Instance,
    /// `ProcessMessages.getInstance(manager, axisID, allowMultiLineLog)`.
    InstanceAllowMultiLineLog(bool),
    /// `ProcessMessages.getMultiLineInstance(manager, axisID)`.
    MultiLine,
    /// `ProcessMessages.getMultiLineInstance(manager, axisID, allowMultiLineLog)`.
    MultiLineAllowMultiLineLog(bool),
    /// `ProcessMessages.getMultiLineInstance(manager, axisID, multilineWarning,
    /// multilineInfo, logInfoMessages)`.
    MultiLineWarningInfo(bool, bool, bool),
}

impl MessagesKind {
    fn build(self, manager: Option<&'static dyn BaseManager>, axis_id: AxisID) -> ProcessMessages {
        match self {
            MessagesKind::Instance => ProcessMessages::get_instance(manager, axis_id),
            MessagesKind::InstanceAllowMultiLineLog(allow) => {
                ProcessMessages::get_instance_allow_multi_line_log(manager, axis_id, allow)
            }
            MessagesKind::MultiLine => ProcessMessages::get_multi_line_instance(manager, axis_id),
            MessagesKind::MultiLineAllowMultiLineLog(allow) => {
                ProcessMessages::get_multi_line_instance_allow_multi_line_log(
                    manager, axis_id, allow,
                )
            }
            MessagesKind::MultiLineWarningInfo(warning, info, log_info) => {
                ProcessMessages::get_multi_line_instance_with_options(
                    manager, axis_id, warning, info, log_info,
                )
            }
        }
    }
}

impl SystemProgram {
    /// The common body of every Java constructor.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        property_user_dir: Option<String>,
        command_array: Option<Vec<String>>,
        axis_id: AxisID,
        messages: MessagesKind,
    ) -> SystemProgram {
        SystemProgram {
            property_user_dir,
            manager,
            command_array: Mutex::new(command_array),
            axis_id,
            process_messages: Mutex::new(messages.build(manager, axis_id)),
            debug: Mutex::new(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level()),
            exit_value: AtomicI32::new(i32::MIN),
            std_input: Mutex::new(None),
            stdout: Mutex::new(None),
            stderr: Mutex::new(None),
            working_directory: Mutex::new(None),
            exception_message: Mutex::new(String::new()),
            started: AtomicBool::new(false),
            done: AtomicBool::new(false),
            run_timestamp: Mutex::new(None),
            cmd_in_buffer: Mutex::new(None),
            accept_input_while_running: AtomicBool::new(false),
            command_line: Mutex::new(None),
            process: Mutex::new(None),
            collect_output: AtomicBool::new(true),
            command_action: Mutex::new(None),
            background_monitor: None,
        }
    }

    /// Java `SystemProgram(BaseManager, String, String[], AxisID)`.
    pub fn new_array(
        manager: Option<&'static dyn BaseManager>,
        property_user_dir: Option<String>,
        cmd_array: Option<Vec<String>>,
        axis_id: AxisID,
    ) -> SystemProgram {
        SystemProgram::new(
            manager,
            property_user_dir,
            cmd_array,
            axis_id,
            MessagesKind::Instance,
        )
    }

    /// Java static `getMultiLineInstance(BaseManager, String, String[], AxisID)`:
    /// `new SystemProgram(manager, propertyUserDir, cmdArray, axisID, true, false)`.
    pub fn get_multi_line_instance(
        manager: Option<&'static dyn BaseManager>,
        property_user_dir: Option<String>,
        cmd_array: Option<Vec<String>>,
        axis_id: AxisID,
    ) -> SystemProgram {
        SystemProgram::new(
            manager,
            property_user_dir,
            cmd_array,
            axis_id,
            MessagesKind::MultiLineAllowMultiLineLog(false),
        )
    }

    /// Java `BackgroundSystemProgram(BaseManager, String[], DetachedProcessMonitor,
    /// AxisID)`: `super(manager, manager.getPropertyUserDir(), command, axisID)`.
    pub fn background(
        manager: &'static dyn BaseManager,
        command: Option<Vec<String>>,
        monitor: Arc<dyn BackgroundWait>,
        axis_id: AxisID,
    ) -> SystemProgram {
        let mut program = SystemProgram::new_array(
            Some(manager),
            manager.get_property_user_dir(),
            command,
            axis_id,
        );
        program.background_monitor = Some(monitor);
        program
    }

    /// Java `changeParameter`.
    pub fn change_parameter(&self, parameter: Option<&str>, index: i32) {
        let mut command_array = self.command_array.lock().unwrap();
        if let (Some(parameter), Some(command_array)) = (parameter, command_array.as_mut())
            && index >= 0
            && command_array.len() as i32 > index
        {
            command_array[index as usize] = parameter.to_owned();
            *self.command_line.lock().unwrap() = None;
        }
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, debug_level: DebugLevel) {
        *self.debug.lock().unwrap() = debug_level;
    }

    /// Java `setStdInput`.
    pub fn set_std_input(&self, program_input: Option<Vec<String>>) {
        *self.std_input.lock().unwrap() = program_input;
    }

    /// Java `getStdInput`.
    pub fn get_std_input(&self) -> Option<Vec<String>> {
        self.std_input.lock().unwrap().clone()
    }

    /// Java `clearStdError`.
    pub fn clear_std_error(&self) {
        if let Some(stderr) = self.stderr.lock().unwrap().as_ref() {
            stderr.lock().unwrap().clear();
        }
    }

    /// Java `getStdError(Object listenerKey)`.
    pub fn get_std_error_listener(&self, listener_key: &str) -> Option<Vec<String>> {
        let stderr = self.stderr.lock().unwrap().clone()?;
        let lines = stderr.lock().unwrap().get_for_listener(listener_key);
        Some(lines)
    }

    /// Java `getStdOutput(Object listenerKey)`.
    pub fn get_std_output_listener(&self, listener_key: &str) -> Option<Vec<String>> {
        let stdout = self.stdout.lock().unwrap().clone()?;
        let lines = stdout.lock().unwrap().get_for_listener(listener_key);
        Some(lines)
    }

    /// Java `dropStdOutputListener`.
    pub fn drop_std_output_listener(&self, listener_key: &str) {
        if let Some(stdout) = self.stdout.lock().unwrap().as_ref() {
            stdout.lock().unwrap().drop_listener(listener_key);
        }
    }

    /// Java `setCurrentStdInput`.
    pub fn set_current_std_input(&self, input: &str) -> std::io::Result<()> {
        let mut cmd_in_buffer = self.cmd_in_buffer.lock().unwrap();
        if let Some(cmd_in_buffer) = cmd_in_buffer.as_mut() {
            cmd_in_buffer.write_all(input.as_bytes())?;
            cmd_in_buffer.write_all(b"\n")?;
            cmd_in_buffer.flush()?;
        }
        Ok(())
    }

    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&self, working_directory: Option<PathBuf>) {
        *self.working_directory.lock().unwrap() = working_directory;
    }

    /// Java `run`: execute the command.
    pub fn run(&self) {
        let debug = *self.debug.lock().unwrap();
        let mut max_debug_print = 0;
        if debug.is_on() {
            max_debug_print = 5;
            if debug.is_extra_verbose() {
                max_debug_print = 1000;
            } else if debug.is_verbose() {
                max_debug_print = 15;
            } else if debug.is_extra() {
                max_debug_print = 10;
            }
        }
        let command_array = self.command_array.lock().unwrap().clone();
        let std_input = self.std_input.lock().unwrap().clone();
        let mut print_command = false;
        if let Some(command_array) = &command_array
            && debug.is_on()
            && !command_array.is_empty()
            && (debug.is_verbose()
                || (command_array[0] != "env"
                    && command_array[0] != "ssh"
                    && command_array[0] != "ps"))
        {
            print_command = true;
            eprintln!();
            for command in command_array {
                eprintln!("  {command}");
            }
        }
        self.started.store(true, Ordering::SeqCst);
        if debug.is_on()
            && let Some(working_directory) = self.working_directory.lock().unwrap().as_ref()
        {
            eprintln!(
                "SystemProgram: working directory: {}",
                utilities::java_io_file_get_absolute_path(&working_directory.to_string_lossy())
            );
        }
        // Setup the Process object and run the command
        *self.process.lock().unwrap() = None;
        let result: std::io::Result<()> = (|| {
            {
                let mut working_directory = self.working_directory.lock().unwrap();
                if working_directory.is_none()
                    && let Some(property_user_dir) = &self.property_user_dir
                    && !property_user_dir.chars().all(char::is_whitespace)
                {
                    *working_directory = Some(PathBuf::from(property_user_dir));
                }
            }
            // timestamp
            let mut timestamp_string = String::new();
            let Some(command_array) = &command_array else {
                self.exit_value.store(1204, Ordering::SeqCst); // bug# 1204
                return Ok(());
            };
            for command in command_array.iter().take(2) {
                timestamp_string.push_str(&format!("{command} "));
            }
            if print_command {
                utilities::timestamp_command_status(
                    Some(&timestamp_string),
                    Some(utilities::STARTED_STATUS),
                );
            }
            *self.run_timestamp.lock().unwrap() = Some(SystemTime::now());

            *self.command_action.lock().unwrap() =
                utilities::get_command_action_array(Some(command_array), std_input.as_deref());
            let working_directory = self.working_directory.lock().unwrap().clone();
            let mut process = runtime_exec(
                command_array,
                std_input.as_deref(),
                working_directory.as_deref(),
            )?;
            *self.process.lock().unwrap() = Some(process.id());
            std::thread::sleep(std::time::Duration::from_millis(100));
            if debug.is_extra() {
                eprintln!("returned, process started");
            }
            // Create a buffered writer to handle the stdin, stdout and stderr
            // streams of the process
            // `new BufferedWriter(new OutputStreamWriter(cmdIn))`: one write per flush.
            let mut cmd_in = process.stdin.take().map(std::io::BufWriter::new);

            // Set up a reader thread to keep the stdout buffers of the process empty
            let stdout = self.new_output_buffer_manager(debug);
            *self.stdout.lock().unwrap() = Some(Arc::clone(&stdout));
            let stdout_reader_thread = spawn_reader(process.stdout.take(), Arc::clone(&stdout));
            // Set up a reader thread to keep the stdout buffers of the process empty
            let stderr = self.new_output_buffer_manager(debug);
            *self.stderr.lock().unwrap() = Some(Arc::clone(&stderr));
            let stderr_reader_thread = spawn_reader(process.stderr.take(), Arc::clone(&stderr));

            // Write out to the program's stdin pipe each line of the
            // stdInput array if it is not null
            if let (Some(std_input), Some(cmd_in)) = (&std_input, cmd_in.as_mut()) {
                for line in std_input {
                    let _ = cmd_in.write_all(line.as_bytes());
                    let _ = cmd_in.write_all(b"\n");
                    let _ = cmd_in.flush();
                }
            }
            if !self.accept_input_while_running.load(Ordering::SeqCst) {
                drop(cmd_in.take());
            } else {
                *self.cmd_in_buffer.lock().unwrap() = cmd_in.take();
            }
            if let Some(std_input) = &std_input
                && !std_input.is_empty()
                && debug.is_on()
            {
                eprintln!("SystemProgram stdin: {} line(s)", std_input.len());
                for line in std_input.iter().take(max_debug_print) {
                    eprintln!("{line}");
                }
                if max_debug_print > 0 && std_input.len() > max_debug_print {
                    eprintln!("...");
                }
            }

            // Wait for the process to exit
            let command_action = self.command_action.lock().unwrap().clone();
            if debug.is_verbose()
                && let Some(command_action) = &command_action
            {
                eprint!("SystemProgram: {command_action}: Waiting for process to end...");
            }
            self.wait_for_process();
            let status = process.wait();
            if print_command {
                utilities::timestamp_command_status(
                    Some(&timestamp_string),
                    Some(utilities::FINISHED_STATUS),
                );
            }
            // Inform the output manager threads that the process is done
            stdout.lock().unwrap().set_process_done(true);
            stderr.lock().unwrap().set_process_done(true);

            let exit_value = self.get_process_exit_value(status);
            self.exit_value.store(exit_value, Ordering::SeqCst);
            if exit_value == 0 {
                if let Some(msg) = utilities::get_command_action_message(command_action.as_deref())
                {
                    eprintln!("{msg}");
                }
            } else if debug.is_on() {
                eprintln!("SystemProgram exit value: {exit_value}");
            }

            // Wait for the manager threads to complete.  Java joins with a
            // one-second limit; the readers end at the child's end of file.
            let _ = stderr_reader_thread.map(|thread| thread.join());
            let _ = stdout_reader_thread.map(|thread| thread.join());

            let size = stdout.lock().unwrap().size();
            if size > 0
                && debug.is_verbose()
                && let Some(command_action) = &command_action
            {
                eprintln!("\nSystemProgram: {command_action}: stdout: {size} line(s):");
                let stdout = stdout.lock().unwrap();
                for i in 0..size.min(max_debug_print) {
                    eprintln!("{}", stdout.get_line(i).unwrap_or(""));
                }
                if max_debug_print > 0 && size > max_debug_print {
                    eprintln!("...");
                }
                eprintln!();
            }
            let size = stderr.lock().unwrap().size();
            if size > 0 {
                let mut printed = false;
                if debug.is_verbose()
                    && let Some(command_action) = &command_action
                {
                    eprintln!("SystemProgram: {command_action}: stderr: {size} line(s):");
                    printed = true;
                    let stderr = stderr.lock().unwrap();
                    for i in 0..size.min(max_debug_print) {
                        eprintln!("{}", stderr.get_line(i).unwrap_or(""));
                    }
                    if max_debug_print > 0 && size > max_debug_print {
                        eprintln!("...");
                    }
                }
                if printed {
                    eprintln!();
                }
            }
            Ok(())
        })();
        if let Err(except) = result {
            eprintln!("{}", self.get_command_line());
            // Java's IOException message from `Runtime.exec`: `Cannot run program
            // "<name>" (in directory "<dir>"): error=<errno>, <strerror>`.
            let command_array = self.command_array.lock().unwrap().clone();
            let program = command_array
                .as_ref()
                .and_then(|array| array.first().cloned())
                .unwrap_or_default();
            let exception_message = format!(
                "Cannot run program \"{program}\": error={}, {}",
                except.raw_os_error().unwrap_or(0),
                except
            );
            eprintln!("{exception_message}");
            *self.exception_message.lock().unwrap() = exception_message.clone();
            let error_tag = "error=";
            if exception_message.contains("Cannot run program \"tcsh\"") {
                ui_harness::post_message_dialog(
                    self.manager,
                    exception_message.clone(),
                    "System Error".to_owned(),
                    None,
                );
            } else if exception_message.contains("Cannot run program \"python\"") {
                ui_harness::post_message_dialog(
                    self.manager,
                    format!(
                        "Unable to run python.  Please see the IMOD Users Guide.\n{exception_message}"
                    ),
                    "System Error".to_owned(),
                    None,
                );
            } else if exception_message.contains("not found") {
                // Unable to pop up an error message. This exception may cause
                // dialog.setVisible to lock up.
                eprintln!("ERROR: Unable to run command.\n{exception_message}");
                self.exit_value.store(-3, Ordering::SeqCst);
                self.done.store(true, Ordering::SeqCst);
                return;
            } else if exception_message.contains(error_tag) {
                self.exit_value.store(1, Ordering::SeqCst);
                // Get the error number from the exception message
                let array: Vec<&str> = exception_message.split_whitespace().collect();
                for word in array.iter().rev() {
                    if word.contains(error_tag) {
                        let error_array: Vec<&str> = word.split('=').map(str::trim).collect();
                        if error_array.len() > 1 {
                            let mut n = EtomoNumber::new();
                            n.set_string(Some(error_array[1].trim_end_matches(',')));
                            if n.is_valid() {
                                self.exit_value.store(n.get_int(), Ordering::SeqCst);
                            }
                        }
                    }
                }
                // Add extra documentation for too many open files (error 24).
                ui_harness::post_message_dialog(
                    self.manager,
                    format!(
                        "Unable to run command{}.\n{exception_message}",
                        if exception_message.contains("Too many open files") {
                            format!(":\n{}", self.get_command_line())
                        } else {
                            String::new()
                        }
                    ),
                    "System Error".to_owned(),
                    None,
                );
            }
        }
        {
            let mut process_messages = self.process_messages.lock().unwrap();
            if let Some(stdout) = self.stdout.lock().unwrap().as_ref() {
                process_messages.add_process_output_output_buffer_manager(&stdout.lock().unwrap());
            }
            if let Some(stderr) = self.stderr.lock().unwrap().as_ref() {
                process_messages.add_process_output_output_buffer_manager(&stderr.lock().unwrap());
            }
            if !debug.is_on() {
                process_messages.print_all();
            }
        }
        // close standard input if it wasn't closed before
        if self.accept_input_while_running.load(Ordering::SeqCst) {
            drop(self.cmd_in_buffer.lock().unwrap().take());
        }
        // Set the done flag for the thread
        self.done.store(true, Ordering::SeqCst);
    }

    /// Java `destroy`: `Process.destroy()` sends `SIGTERM` on Unix and is
    /// `TerminateProcess(handle, 1)` on Windows.
    pub fn destroy(&self) {
        let Some(pid) = *self.process.lock().unwrap() else {
            return;
        };
        // SAFETY: signalling a child this program started.
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        #[cfg(windows)]
        crate::imod::libcfshr::b3dutil::terminate_process(pid as u32, 1);
    }

    /// Java private `newOutputBufferManager` / `newErrorBufferManager`.
    fn new_output_buffer_manager(&self, debug: DebugLevel) -> Arc<Mutex<OutputBufferManager>> {
        let mut buffer_manager = OutputBufferManager::new();
        buffer_manager.set_debug(debug.is_extra_verbose());
        buffer_manager.set_collect_output(self.collect_output.load(Ordering::SeqCst));
        Arc::new(Mutex::new(buffer_manager))
    }

    /// Java `setCollectOutput`.
    pub fn set_collect_output(&self, input: bool) {
        self.collect_output.store(input, Ordering::SeqCst);
    }

    /// Java `waitForProcess`: empty in `SystemProgram`; `BackgroundSystemProgram`
    /// uses the process monitor to wait for a background process to finish.
    fn wait_for_process(&self) {
        if let Some(monitor) = &self.background_monitor {
            // wait until process is finished
            while monitor.is_process_running() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    /// Java `getProcessExitValue`.
    fn get_process_exit_value(&self, status: std::io::Result<std::process::ExitStatus>) -> i32 {
        if let Some(monitor) = &self.background_monitor {
            if monitor.is_process_running() {
                panic!("getExitValue() called while process is running.");
            }
            if monitor.is_process_end_state_done() {
                return 0;
            }
            return 1;
        }
        // `Process.exitValue()`: a child killed by a signal reports 128 + signal.
        match status {
            Ok(status) => status.code().unwrap_or_else(|| {
                128 + crate::imod::libcfshr::b3dutil::exit_signal(&status).unwrap_or(0)
            }),
            Err(_) => 1,
        }
    }

    /// Java `getStdOutput()`.
    pub fn get_std_output(&self) -> Option<Vec<String>> {
        let stdout = self.stdout.lock().unwrap().clone()?;
        let lines = stdout.lock().unwrap().get();
        Some(lines)
    }

    /// Java `getStdError()`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        let stderr = self.stderr.lock().unwrap().clone()?;
        let lines = stderr.lock().unwrap().get();
        Some(lines)
    }

    /// Java `printStdError`.
    pub fn print_std_error(&self) {
        eprintln!("stderr:");
        if let Some(stderr) = self.stderr.lock().unwrap().as_ref() {
            stderr.lock().unwrap().print_to_err();
        }
    }

    /// Java `printStdOutput`.
    pub fn print_std_output(&self) {
        eprintln!("stdout:");
        if let Some(stdout) = self.stdout.lock().unwrap().as_ref() {
            stdout.lock().unwrap().print_to_err();
        }
    }

    /// Java `getStdErrorString`.
    pub fn get_std_error_string(&self) -> Option<String> {
        let std_error_array = self.get_std_error()?;
        if std_error_array.is_empty() {
            return None;
        }
        let mut builder = String::new();
        for line in std_error_array {
            builder.push_str(&(line + "\n"));
        }
        Some(builder)
    }

    /// Java `getStdOutputString`.
    pub fn get_std_output_string(&self) -> Option<String> {
        let array = self.get_std_output()?;
        if array.is_empty() {
            return None;
        }
        let mut builder = String::new();
        for line in array {
            builder.push_str(&(line + "\n"));
        }
        Some(builder)
    }

    /// Java `getWorkingDirectory`.
    pub fn get_working_directory(&self) -> Option<String> {
        match self.working_directory.lock().unwrap().as_ref() {
            None => self.property_user_dir.clone(),
            Some(working_directory) => Some(working_directory.to_string_lossy().into_owned()),
        }
    }

    /// Java `getExitValue`.
    pub fn get_exit_value(&self) -> i32 {
        self.exit_value.load(Ordering::SeqCst)
    }

    /// Java `setExitValue`.
    pub fn set_exit_value(&self, value: i32) {
        self.exit_value.store(value, Ordering::SeqCst);
    }

    /// Java `getCommandLine`.
    pub fn get_command_line(&self) -> String {
        let mut command_line = self.command_line.lock().unwrap();
        if command_line.is_none() {
            let mut buffer = String::new();
            if let Some(command_array) = self.command_array.lock().unwrap().as_ref() {
                for command in command_array {
                    buffer.push_str(&format!("{command} "));
                }
            }
            *command_line = Some(buffer);
        }
        command_line.clone().unwrap()
    }

    /// Java `getCommandAction`.
    pub fn get_command_action(&self) -> String {
        match self.command_action.lock().unwrap().clone() {
            None => self.get_command_line(),
            Some(command_action) => command_action,
        }
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `setMessagePrependTag`.
    pub fn set_message_prepend_tag(&self, tag: Option<&str>) {
        self.process_messages
            .lock()
            .unwrap()
            .set_message_prepend_tag(tag);
    }

    /// Java `isStarted`.
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    /// Java `isDone`.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }

    /// Java `getRunTimestamp`.
    pub fn get_run_timestamp(&self) -> Option<SystemTime> {
        *self.run_timestamp.lock().unwrap()
    }

    /// Java `getProcessMessages`.
    pub fn get_process_messages(&self) -> std::sync::MutexGuard<'_, ProcessMessages> {
        self.process_messages.lock().unwrap()
    }

    /// Java `setAcceptInputWhileRunning`.
    pub fn set_accept_input_while_running(&self, accept_input_while_running: bool) {
        self.accept_input_while_running
            .store(accept_input_while_running, Ordering::SeqCst);
    }

    /// The command array (read by `BackgroundProcess` and the tests).
    pub fn get_command_array(&self) -> Option<Vec<String>> {
        self.command_array.lock().unwrap().clone()
    }
}

/// The reader thread of `OutputBufferManager.run`: until `SystemProgram.run` reports
/// the process done, add every line the stream has and sleep 100 ms; then add what is
/// left.  The sleep after the child's end of file is what delays `SystemProgram`'s
/// `done` (it joins this thread first), so `ParsePID`'s `isDone` poll still sees a
/// process that has just ended as running and reads its PID line.
fn spawn_reader<R: std::io::Read + Send + 'static>(
    reader: Option<R>,
    buffer: Arc<Mutex<OutputBufferManager>>,
) -> Option<std::thread::JoinHandle<()>> {
    let reader = reader?;
    Some(std::thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = Vec::new();
        // `BufferedReader.readLine`: a line without "\n", "\r\n" or "\r", or None at end
        // of file (or on an error, which Java's IOException catch ends the thread on).
        let mut read_line = |line: &mut Vec<u8>| -> Option<String> {
            line.clear();
            match reader.read_until(b'\n', line) {
                Ok(0) | Err(_) => None,
                Ok(_) => {
                    if line.last() == Some(&b'\n') {
                        line.pop();
                    }
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    Some(String::from_utf8_lossy(line).into_owned())
                }
            }
        };
        while !buffer.lock().unwrap().is_process_done() {
            while let Some(text) = read_line(&mut line) {
                buffer.lock().unwrap().add(text);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        while let Some(text) = read_line(&mut line) {
            buffer.lock().unwrap().add(text);
        }
    }))
}

/// The `imod` binary that runs our commands.  Defaults to this process's own
/// executable when that is the `imod` launcher; the tests name the built
/// binary with [`set_imod_executable`].
static IMOD_EXECUTABLE: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Names the `imod` binary [`runtime_exec`] runs our commands with.  Only the
/// first call has an effect.
pub fn set_imod_executable(path: PathBuf) {
    let _ = IMOD_EXECUTABLE.set(Some(path));
}

/// Our `imod` binary, when this process is it.
pub(crate) fn imod_executable() -> Option<&'static Path> {
    IMOD_EXECUTABLE
        .get_or_init(|| {
            std::env::current_exe().ok().filter(|exe| {
                exe.file_name().is_some_and(|base| {
                    base.to_string_lossy() == format!("imod{}", std::env::consts::EXE_SUFFIX)
                })
            })
        })
        .as_deref()
}

/// The command array `runtime_exec` actually starts: our `imod` binary with
/// `argv[0]` naming the command, or the array as given.  See the module
/// comment.  Returns `(program, argv0, arguments)`.
pub fn resolve_command_array(
    command_array: &[String],
    std_input: Option<&[String]>,
) -> (PathBuf, Option<PathBuf>, Vec<String>) {
    let unchanged = || {
        (
            PathBuf::from(&command_array[0]),
            None,
            command_array[1..].to_vec(),
        )
    };
    let Some(imod) = imod_executable() else {
        return unchanged();
    };
    let own = |name: &str, rest: &[String]| {
        (
            imod.to_path_buf(),
            Some(imod.with_file_name(name)),
            rest.to_vec(),
        )
    };
    let is_ours = |path: &str| -> Option<String> {
        let name = Path::new(path).file_name()?.to_str()?.to_owned();
        let in_bin = !path.contains('/')
            || std::env::var_os("IMOD_DIR")
                .is_some_and(|dir| Path::new(path).parent() == Some(&Path::new(&dir).join("bin")));
        (in_bin && crate::imod::commands::find(&name).is_some()).then_some(name)
    };
    let first = command_array[0].as_str();
    if first == "python" || first == "python3" {
        let mut index = 1;
        if command_array.get(index).is_some_and(|arg| arg == "-u") {
            index += 1;
        }
        match command_array.get(index) {
            // `python -u` reading the script `vmstopy` wrote on its standard
            // input (`ComScriptProcess.execPython`)
            None if std_input
                .and_then(|input| input.first())
                .is_some_and(|line| line.starts_with("#!/usr/bin/env python")) =>
            {
                return own("runcom", &["-P".to_owned(), "-S".to_owned()]);
            }
            Some(script) => {
                if let Some(name) = Path::new(script).file_name().and_then(|n| n.to_str())
                    && crate::imod::commands::find(name).is_some()
                {
                    return own(name, &command_array[index + 1..]);
                }
            }
            None => {}
        }
        return unchanged();
    }
    if let Some(name) = is_ours(first) {
        return own(&name, &command_array[1..]);
    }
    unchanged()
}

/// `Runtime.getRuntime().exec(cmdarray, null, dir)`; see the module comment.
pub fn runtime_exec(
    command_array: &[String],
    std_input: Option<&[String]>,
    working_directory: Option<&Path>,
) -> std::io::Result<Child> {
    if command_array.is_empty() {
        return Err(std::io::Error::other("Empty command"));
    }
    let (program, argv0, arguments) = resolve_command_array(command_array, std_input);
    let mut process = std::process::Command::new(&program);
    if let Some(argv0) = argv0 {
        crate::imod::libcfshr::b3dutil::command_arg0(&mut process, argv0);
    }
    process
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(working_directory) = working_directory {
        process.current_dir(working_directory);
    }
    process.spawn()
}
