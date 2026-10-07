//! `IMOD/Etomo/src/etomo/process/InteractiveSystemProgram.java`.
//!
//! Runs a program whose standard input stays open while it runs (3dmod,
//! midas, imodsendevent) and whose output is read line by line as it becomes
//! available.  `BufferedReader.ready()` + `readLine()` is a reader thread per
//! stream feeding a channel that `readStdout`/`readStderr` poll without
//! blocking.
//!
//! The Java keeps the lines it has read in two **static** lists
//! (`STD_OUTPUT`, `STD_ERROR`, `InteractiveSystemProgram.java:65-66`), shared
//! by every instance, so `getStdOutput()` of one 3dmod returns the output of
//! every interactive program run before it.  Fixed in translation
//! (`BUGS.md`): the lists belong to the instance.

use super::base_process_manager::BaseProcessManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ChildStdin;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

/// Java final class `InteractiveSystemProgram implements Runnable`.
pub struct InteractiveSystemProgram {
    /// Java `outputFileLastModified`, milliseconds since the epoch.
    output_file_last_modified: Mutex<Option<i64>>,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    process_manager: Option<&'static BaseProcessManager>,
    thread_name: Mutex<Option<String>>,
    exit_value: Mutex<i32>,
    command_line: Option<String>,
    command_array: Option<Vec<String>>,
    command: Option<Arc<dyn Command + Send + Sync>>,
    input_buffer: Mutex<Option<std::io::BufWriter<ChildStdin>>>,
    output_buffer: Mutex<Option<Receiver<String>>>,
    error_buffer: Mutex<Option<Receiver<String>>>,
    working_directory: Mutex<Option<PathBuf>>,
    exception_message: Mutex<String>,
    command_action: Mutex<Option<String>>,
    print_stderr: Mutex<bool>,
    std_output: Mutex<Vec<String>>,
    std_error: Mutex<Vec<String>>,
}

impl InteractiveSystemProgram {
    fn build(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_manager: Option<&'static BaseProcessManager>,
        command_array: Option<Vec<String>>,
        command_line: Option<String>,
        command: Option<Arc<dyn Command + Send + Sync>>,
    ) -> InteractiveSystemProgram {
        InteractiveSystemProgram {
            output_file_last_modified: Mutex::new(None),
            manager,
            axis_id,
            process_manager,
            thread_name: Mutex::new(None),
            exit_value: Mutex::new(i32::MIN),
            command_line,
            command_array,
            command,
            input_buffer: Mutex::new(None),
            output_buffer: Mutex::new(None),
            error_buffer: Mutex::new(None),
            working_directory: Mutex::new(None),
            exception_message: Mutex::new(String::new()),
            command_action: Mutex::new(None),
            print_stderr: Mutex::new(false),
            std_output: Mutex::new(Vec::new()),
            std_error: Mutex::new(Vec::new()),
        }
    }

    /// Java `InteractiveSystemProgram(BaseManager, List<String>, AxisID)` and
    /// `(BaseManager, String[], AxisID)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        command_array: Option<Vec<String>>,
        axis_id: AxisID,
    ) -> InteractiveSystemProgram {
        InteractiveSystemProgram::build(manager, axis_id, None, command_array, None, None)
    }

    /// Java `InteractiveSystemProgram(BaseManager, Command, BaseProcessManager,
    /// AxisID)`.
    pub fn new_command(
        manager: &'static dyn BaseManager,
        command: Arc<dyn Command + Send + Sync>,
        process_manager: &'static BaseProcessManager,
        axis_id: AxisID,
    ) -> InteractiveSystemProgram {
        let command_array = command.get_command_array();
        let command_line = command.get_command_line();
        InteractiveSystemProgram::build(
            manager,
            axis_id,
            Some(process_manager),
            command_array,
            command_line,
            Some(command),
        )
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

    /// Java `setCurrentStdInput`.
    pub fn set_current_std_input(&self, input: &str) -> std::io::Result<()> {
        let mut input_buffer = self.input_buffer.lock().unwrap();
        if let Some(input_buffer) = input_buffer.as_mut() {
            input_buffer.write_all(input.as_bytes())?;
            input_buffer.write_all(b"\n")?;
            input_buffer.flush()?;
        }
        Ok(())
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `setName`.
    pub fn set_name(&self, thread_name: &str) {
        *self.thread_name.lock().unwrap() = Some(thread_name.to_owned());
    }

    /// Java `setPrintStderr`.
    pub fn set_print_stderr(&self) {
        *self.print_stderr.lock().unwrap() = true;
    }

    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&self, working_directory: Option<PathBuf>) {
        *self.working_directory.lock().unwrap() = working_directory;
    }

    /// Java `getCommandLine`.
    pub fn get_command_line(&self) -> Option<String> {
        self.command_line.clone()
    }

    /// Java `getCommandName`.
    pub fn get_command_name(&self) -> Option<String> {
        self.command.as_ref()?.get_command_name()
    }

    /// Java `getCommandAction`.
    pub fn get_command_action(&self) -> Option<String> {
        match self.command_action.lock().unwrap().clone() {
            None => self.get_command_name(),
            Some(command_action) => Some(command_action),
        }
    }

    /// Java `run`: execute the command.
    pub fn run(&self) {
        // Setup the Process object and run the command
        let mut process: Option<std::process::Child> = None;
        if let Some(command) = &self.command {
            let output_file = command.get_command_output_file();
            // `File.lastModified()` is 0 for a missing file.
            let modified = output_file
                .and_then(|file| std::fs::metadata(file).ok())
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |duration| duration.as_millis() as i64);
            *self.output_file_last_modified.lock().unwrap() = Some(modified);
        }
        let command_array: Option<Vec<String>> = self.command_array.clone().or_else(|| {
            // `Runtime.exec(String)` splits the command line on white space.
            self.command_line
                .as_ref()
                .map(|line| line.split_whitespace().map(str::to_owned).collect())
        });
        let working_directory = self.working_directory.lock().unwrap().clone();
        let directory = match working_directory {
            None => {
                let current_user_directory = self
                    .manager
                    .get_property_user_dir()
                    .map(PathBuf::from)
                    .unwrap_or_default();
                *self.command_action.lock().unwrap() = match &self.command_array {
                    Some(command_array) => {
                        utilities::get_command_action_array(Some(command_array), None)
                    }
                    None => utilities::get_command_action(self.command_line.as_deref()),
                };
                current_user_directory
            }
            Some(working_directory) => working_directory,
        };
        match command_array
            .as_deref()
            .map(|array| super::system_program::runtime_exec(array, None, Some(&directory)))
        {
            Some(Ok(mut child)) => {
                // Create a buffered writer to handle the stdin, stdout and stderr
                // streams of the process
                *self.input_buffer.lock().unwrap() =
                    child.stdin.take().map(std::io::BufWriter::new);
                *self.output_buffer.lock().unwrap() = child.stdout.take().map(line_channel);
                *self.error_buffer.lock().unwrap() = child.stderr.take().map(line_channel);
                process = Some(child);
            }
            Some(Err(except)) => {
                eprintln!("{except}");
                *self.exception_message.lock().unwrap() = except.to_string();
            }
            None => {}
        }
        match process.as_mut() {
            Some(process) => {
                let status = process.wait();
                let exit_value = match status {
                    Ok(status) => status.code().unwrap_or_else(|| {
                        128 + crate::imod::libcfshr::b3dutil::exit_signal(&status).unwrap_or(0)
                    }),
                    Err(_) => i32::MIN,
                };
                *self.exit_value.lock().unwrap() = exit_value;
                let command_action = self.command_action.lock().unwrap().clone();
                if exit_value == 0
                    && let Some(msg) =
                        utilities::get_command_action_message(command_action.as_deref())
                {
                    eprintln!("{msg}");
                }
            }
            None => {
                // `process.waitFor()` on a null process: NullPointerException.
                *self.exception_message.lock().unwrap() = "null".to_owned();
            }
        }
        if let Some(process_manager) = self.process_manager {
            let exit_value = *self.exit_value.lock().unwrap();
            process_manager.msg_interactive_system_program_done(self, exit_value);
        }
        if *self.print_stderr.lock().unwrap() {
            self.get_std_error();
        }
    }

    /// Java `writeStdin`: send text to the program's standard input.
    pub fn write_stdin(&self, line: &str) {
        let mut input_buffer = self.input_buffer.lock().unwrap();
        if let Some(input_buffer) = input_buffer.as_mut() {
            let result = input_buffer
                .write_all(line.as_bytes())
                .and_then(|()| input_buffer.write_all(b"\n"))
                .and_then(|()| input_buffer.flush());
            if let Err(except) = result {
                eprintln!("{except}");
                *self.exception_message.lock().unwrap() = except.to_string();
            }
        }
    }

    /// Java `readStdout`: read one line from the stdout buffer if available,
    /// if one isn't available then null is returned.
    pub fn read_stdout(&self) -> Option<String> {
        let line = read_ready(&self.output_buffer)?;
        self.std_output.lock().unwrap().push(line.clone());
        Some(line)
    }

    /// Java `getStdOutput`: put all of stdout into stdOutput and then return
    /// stdOutput.
    pub fn get_std_output(&self) -> Option<Vec<String>> {
        while self.read_stdout().is_some() {}
        get_string_array(&self.std_output.lock().unwrap())
    }

    /// Java `getStdError`: put all of stderr into stdError and then return
    /// stdError.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        while self.read_stderr().is_some() {}
        get_string_array(&self.std_error.lock().unwrap())
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref()
    }

    /// Java `getOutputFileLastModified`.
    pub fn get_output_file_last_modified(&self) -> Option<i64> {
        *self.output_file_last_modified.lock().unwrap()
    }

    /// Java `readStderr`: read one line from the stderr buffer if available,
    /// if one isn't available then null is returned.
    pub fn read_stderr(&self) -> Option<String> {
        let line = read_ready(&self.error_buffer)?;
        if *self.print_stderr.lock().unwrap() {
            eprintln!("{line}");
        }
        self.std_error.lock().unwrap().push(line.clone());
        Some(line)
    }

    /// Java `getExitValue`.
    pub fn get_exit_value(&self) -> i32 {
        *self.exit_value.lock().unwrap()
    }

    /// Java `getName`.
    pub fn get_name(&self) -> Option<String> {
        self.thread_name.lock().unwrap().clone()
    }

    /// Java `getExceptionMessage`.
    pub fn get_exception_message(&self) -> String {
        self.exception_message.lock().unwrap().clone()
    }
}

/// Java private `getStringArray`.
fn get_string_array(list: &[String]) -> Option<Vec<String>> {
    if list.is_empty() {
        return None;
    }
    Some(list.to_vec())
}

/// `reader.ready() ? reader.readLine() : null`.
fn read_ready(buffer: &Mutex<Option<Receiver<String>>>) -> Option<String> {
    let buffer = buffer.lock().unwrap();
    match buffer.as_ref()?.try_recv() {
        Ok(line) => Some(line),
        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
    }
}

/// A reader thread delivering the lines of `stream` as they arrive.
fn line_channel<R: std::io::Read + Send + 'static>(stream: R) -> Receiver<String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stream).lines() {
            match line {
                Ok(line) => {
                    if sender.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    receiver
}
