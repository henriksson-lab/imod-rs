//! Rust-native execution and line IPC for eTomo's `SystemProgram` family.
//!
//! Java used a collection of `SystemProgram`, `InteractiveSystemProgram`, and
//! reader threads.  This one owner keeps the same important contract: child
//! standard input is a command channel and stdout/stderr are independently
//! observable line streams.  Higher-level managers decide what each command
//! means; this module never shell-interprets an argument.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use super::output_buffer_manager::OutputBufferManager;
use super::process_messages::ProcessMessages;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub working_directory: Option<PathBuf>,
    /// Lines supplied to a COM-style `-StandardInput` process after launch.
    pub stdin: Vec<String>,
    /// Java `acceptInputWhileRunning`.  Batch/COM children receive EOF after
    /// their configured input; interactive children explicitly retain stdin.
    pub accept_input_while_running: bool,
}

impl ProcessCommand {
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            working_directory: None,
            stdin: Vec::new(),
            accept_input_while_running: false,
        }
    }
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }
    pub fn current_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.working_directory = Some(path.as_ref().to_owned());
        self
    }
    pub fn stdin_lines<I, S>(mut self, lines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stdin = lines.into_iter().map(Into::into).collect();
        self
    }
    /// Java `setStdInput`, before this immutable launch description is spawned.
    pub fn set_std_input<I, S>(&mut self, lines: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stdin = lines.into_iter().map(Into::into).collect();
    }
    /// Java `getStdInput`.
    pub fn get_std_input(&self) -> &[String] {
        &self.stdin
    }
    /// Java `setWorkingDirectory`, before spawn.
    pub fn set_working_directory(&mut self, path: impl AsRef<Path>) {
        self.working_directory = Some(path.as_ref().to_owned());
    }
    pub fn keep_stdin_open(mut self) -> Self {
        self.accept_input_while_running = true;
        self
    }
    /// Java `changeParameter`; invalid indices leave the command untouched.
    pub fn change_parameter(&mut self, parameter: impl Into<OsString>, index: usize) -> bool {
        let Some(argument) = self.args.get_mut(index) else {
            return false;
        };
        *argument = parameter.into();
        true
    }
    pub fn command_line(&self) -> String {
        std::iter::once(&self.program)
            .chain(self.args.iter())
            .map(|part| part.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessStream {
    Stdout,
    Stderr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessLine {
    pub stream: ProcessStream,
    pub line: String,
}

/// A running local process plus nonblocking output collection.
pub struct SystemProgram {
    command: ProcessCommand,
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<ProcessLine>,
    exit_status: Option<ExitStatus>,
    stdout: Vec<String>,
    stderr: Vec<String>,
    /// Source `OutputBufferManager` ownership for monitors that consume a
    /// stream independently of the durable transcript below.
    stdout_buffer: OutputBufferManager,
    stderr_buffer: OutputBufferManager,
    collect_output: bool,
    /// Java `SystemProgram.processMessages`: raw streams remain available,
    /// while parsed error/warning/success state travels with the child.
    process_messages: ProcessMessages,
}

fn collect_lines<R: Read + Send + 'static>(
    reader: R,
    stream: ProcessStream,
    sender: mpsc::Sender<ProcessLine>,
) {
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => {
                    if sender
                        .send(ProcessLine {
                            stream: stream.clone(),
                            line,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

impl SystemProgram {
    pub fn spawn(command: &ProcessCommand) -> io::Result<Self> {
        let mut child_command = Command::new(&command.program);
        child_command
            .args(&command.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(path) = &command.working_directory {
            child_command.current_dir(path);
        }
        let mut child = child_command.spawn()?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (sender, lines) = mpsc::channel();
        if let Some(stdout) = stdout {
            collect_lines(stdout, ProcessStream::Stdout, sender.clone());
        }
        if let Some(stderr) = stderr {
            collect_lines(stderr, ProcessStream::Stderr, sender.clone());
        }
        drop(sender);
        let mut program = Self {
            command: command.clone(),
            child,
            stdin,
            lines,
            exit_status: None,
            stdout: vec![],
            stderr: vec![],
            stdout_buffer: Self::new_output_buffer_manager(),
            stderr_buffer: Self::new_error_buffer_manager(),
            collect_output: true,
            process_messages: ProcessMessages::get_instance(),
        };
        for line in &command.stdin {
            program.send_line(line)?;
        }
        if !command.accept_input_while_running {
            program.close_stdin();
        }
        Ok(program)
    }

    /// Java static `getMultiLineInstance`: retain the normal child ownership
    /// but use the parser configuration that joins multi-line process output.
    pub fn get_multi_line_instance(command: &ProcessCommand) -> io::Result<Self> {
        let mut program = Self::spawn(command)?;
        program.process_messages = ProcessMessages::get_multi_line_instance();
        Ok(program)
    }

    /// Java private `newOutputBufferManager`.
    fn new_output_buffer_manager() -> OutputBufferManager {
        OutputBufferManager::new()
    }

    /// Java private `newErrorBufferManager`.
    fn new_error_buffer_manager() -> OutputBufferManager {
        OutputBufferManager::new()
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn command(&self) -> &ProcessCommand {
        &self.command
    }
    pub fn get_working_directory(&self) -> Option<&Path> {
        self.command.working_directory.as_deref()
    }
    /// The launch-time Java `getStdInput` value remains observable after the
    /// child starts; live input is supplied through `send_line`.
    pub fn get_std_input(&self) -> &[String] {
        self.command.get_std_input()
    }
    pub fn get_command_line(&self) -> String {
        self.command.command_line()
    }
    pub fn get_exit_value(&self) -> Option<i32> {
        self.exit_status.and_then(|status| status.code())
    }
    pub fn is_started(&self) -> bool {
        true
    }
    pub fn is_done(&self) -> bool {
        self.exit_status.is_some()
    }
    pub fn send_line(&mut self, line: &str) -> io::Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "process stdin is closed"))?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()
    }
    /// Java `setStdInput(null)`: closing the pipe signals EOF to an interactive child.
    pub fn close_stdin(&mut self) {
        self.stdin = None;
    }
    pub fn drain_lines(&mut self) -> Vec<ProcessLine> {
        let mut output: Vec<_> = self.lines.try_iter().collect();
        // Once the child has exited, let the two short-lived reader threads
        // flush their final partial lines instead of making callers race EOF.
        if self.exit_status.is_some() {
            loop {
                match self.lines.recv_timeout(Duration::from_millis(20)) {
                    Ok(line) => output.push(line),
                    Err(RecvTimeoutError::Disconnected) | Err(RecvTimeoutError::Timeout) => break,
                }
            }
        }
        for line in &output {
            match &line.stream {
                ProcessStream::Stdout => {
                    self.stdout.push(line.line.clone());
                    self.stdout_buffer.add(line.line.clone());
                }
                ProcessStream::Stderr => {
                    self.stderr.push(line.line.clone());
                    self.stderr_buffer.add(line.line.clone());
                }
            }
            self.process_messages.add_process_output(&line.line);
        }
        output
    }
    /// Java `getStdOutput`; callers may drain live messages and still retrieve
    /// the complete process transcript after completion.
    pub fn get_std_output(&self) -> &[String] {
        &self.stdout
    }
    /// Java `getStdError`.
    pub fn get_std_error(&self) -> &[String] {
        &self.stderr
    }
    /// Java `getStdOutputString`.
    pub fn get_std_output_string(&self) -> String {
        self.stdout.join("\n")
    }
    /// Java `getStdErrorString`.
    pub fn get_std_error_string(&self) -> String {
        self.stderr.join("\n")
    }
    /// Output-buffer monitor registration for a source `SystemProgram`
    /// stdout consumer.  The first caller receives output accumulated before
    /// registration; later callers receive only subsequently drained lines.
    pub fn get_std_output_for_listener(&mut self, listener_key: impl Into<String>) -> Vec<String> {
        self.stdout_buffer.get_for_listener(listener_key)
    }
    /// Output-buffer monitor registration for stderr.
    pub fn get_std_error_for_listener(&mut self, listener_key: impl Into<String>) -> Vec<String> {
        self.stderr_buffer.get_for_listener(listener_key)
    }
    /// Java `OutputBufferManager.dropListener` for both owned streams.
    pub fn drop_output_listener(&mut self, listener_key: &str) {
        self.stdout_buffer.drop_listener(listener_key);
        self.stderr_buffer.drop_listener(listener_key);
    }
    /// Intermittent monitors can consume and discard output without changing
    /// the durable `getStdOutput`/`getStdError` transcript.
    pub fn set_collect_stream_output(&mut self, collect_output: bool) {
        self.collect_output = collect_output;
        self.stdout_buffer.set_collect_output(collect_output);
        self.stderr_buffer.set_collect_output(collect_output);
    }
    /// Java `setCollectOutput`.
    pub fn set_collect_output(&mut self, collect_output: bool) {
        self.set_collect_stream_output(collect_output);
    }
    /// Java `clearStdError`.
    pub fn clear_std_error(&mut self) {
        self.stderr_buffer.clear();
    }
    /// Java `getProcessMessages`.
    pub fn get_process_messages(&self) -> &ProcessMessages {
        &self.process_messages
    }
    pub fn get_process_messages_mut(&mut self) -> &mut ProcessMessages {
        &mut self.process_messages
    }
    /// Java `setMessagePrependTag`, forwarded to the child-owned parser before
    /// future stream records are drained.
    pub fn set_message_prepend_tag(&mut self, tag: Option<&str>) {
        self.process_messages.set_message_prepend_tag(tag);
    }
    /// Java `setDebug`; stream collection remains deterministic, while parser
    /// diagnostics gain the configured debug behavior.
    pub fn set_debug(&mut self, debug: bool) {
        self.process_messages.set_debug(debug);
        self.stdout_buffer.set_debug(debug);
        self.stderr_buffer.set_debug(debug);
    }
    /// Java `printStdError`.
    pub fn print_std_error(&self) {
        eprintln!("stderr:");
        for line in &self.stderr {
            eprintln!("{line}");
        }
    }
    /// Java `printStdOutput`.
    pub fn print_std_output(&self) {
        eprintln!("stdout:");
        for line in &self.stdout {
            eprintln!("{line}");
        }
    }
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.exit_status {
            return Ok(Some(status));
        }
        if let Some(status) = self.child.try_wait()? {
            self.exit_status = Some(status);
        }
        Ok(self.exit_status)
    }
    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.exit_status {
            return Ok(status);
        }
        let status = self.child.wait()?;
        self.exit_status = Some(status);
        Ok(status)
    }
    /// Wait and atomically collect the final stream records.  This is the
    /// convenient terminal operation for managers that do not consume live
    /// output, while [`Self::wait`] retains its existing explicit-drain
    /// contract for interactive callers.
    pub fn wait_and_drain(&mut self) -> io::Result<(ExitStatus, Vec<ProcessLine>)> {
        let status = self.wait()?;
        let lines = self.drain_lines();
        Ok((status, lines))
    }
    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }
    /// Send the POSIX stop signal used by eTomo's pausable local-process path.
    /// Windows has no source-equivalent signal operation, so it reports that
    /// the operation is unsupported instead of pretending the child paused.
    pub fn pause(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        unsafe {
            if libc::kill(self.child.id() as i32, libc::SIGSTOP) == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }
        #[cfg(not(unix))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "pausing a child process is unavailable on this platform",
            ))
        }
    }
    /// Resume a child previously paused by [`Self::pause`].
    pub fn resume(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        unsafe {
            if libc::kill(self.child.id() as i32, libc::SIGCONT) == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }
        #[cfg(not(unix))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "resuming a child process is unavailable on this platform",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runs_without_a_shell_and_collects_both_streams() {
        let command = ProcessCommand::new("sh").args(["-c", "printf out; printf err >&2"]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        assert!(program.wait().unwrap().success());
        let lines = program.drain_lines();
        assert!(
            lines
                .iter()
                .any(|line| line.stream == ProcessStream::Stdout && line.line == "out")
        );
        assert!(
            lines
                .iter()
                .any(|line| line.stream == ProcessStream::Stderr && line.line == "err")
        );
        assert_eq!(program.get_std_output(), ["out"]);
        assert_eq!(program.get_std_error_string(), "err");
    }

    #[test]
    fn batch_commands_close_stdin_but_interactive_commands_keep_it() {
        let batch = ProcessCommand::new("sh").args(["-c", "read value || printf eof"]);
        let mut program = SystemProgram::spawn(&batch).unwrap();
        assert!(program.wait().unwrap().success());
        program.drain_lines();
        assert_eq!(program.get_std_output(), ["eof"]);

        let interactive = ProcessCommand::new("sh")
            .args(["-c", "read value; printf '%s' \"$value\""])
            .keep_stdin_open();
        let mut program = SystemProgram::spawn(&interactive).unwrap();
        assert!(program.is_started());
        assert!(!program.is_done());
        program.send_line("live input").unwrap();
        program.close_stdin();
        assert!(program.wait().unwrap().success());
        program.drain_lines();
        assert_eq!(program.get_std_output(), ["live input"]);
    }

    #[test]
    fn command_metadata_is_mutable_and_observable() {
        let mut command = ProcessCommand::new("echo").args(["old"]);
        let working_directory = std::env::temp_dir();
        assert!(command.change_parameter("new", 0));
        assert!(!command.change_parameter("ignored", 2));
        command.set_std_input(["first", "second"]);
        command.set_working_directory(&working_directory);
        assert_eq!(command.command_line(), "echo new");
        assert_eq!(command.get_std_input(), ["first", "second"]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        assert_eq!(program.get_command_line(), "echo new");
        assert_eq!(program.get_std_input(), ["first", "second"]);
        assert_eq!(
            program.get_working_directory(),
            Some(working_directory.as_path())
        );
        assert_eq!(program.get_exit_value(), None);
        assert!(program.wait().unwrap().success());
        assert_eq!(program.get_exit_value(), Some(0));
    }

    #[test]
    fn wait_and_drain_collects_complete_output() {
        let command = ProcessCommand::new("sh").args(["-c", "printf out; printf err >&2"]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        let (status, lines) = program.wait_and_drain().unwrap();
        assert!(status.success());
        assert_eq!(lines.len(), 2);
        assert_eq!(program.get_std_output(), ["out"]);
        assert_eq!(program.get_std_error(), ["err"]);
    }

    #[test]
    fn live_streams_share_the_source_output_buffer_listener_contract() {
        let command = ProcessCommand::new("sh").args(["-c", "printf one; printf two >&2"]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        program.wait_and_drain().unwrap();
        assert_eq!(program.get_std_output_for_listener("primary"), ["one"]);
        assert_eq!(program.get_std_error_for_listener("primary"), ["two"]);
        assert!(program.get_std_output_for_listener("secondary").is_empty());
        assert_eq!(program.get_std_output(), ["one"]);
        assert_eq!(program.get_std_error(), ["two"]);
    }

    #[test]
    fn drained_child_streams_feed_typed_process_messages() {
        let command = ProcessCommand::new("sh").args([
            "-c",
            "printf 'WARNING: recoverable\\n'; printf 'ERROR: failed\\n' >&2",
        ]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        program.wait_and_drain().unwrap();
        assert_eq!(
            program
                .get_process_messages()
                .get(super::super::process_messages::MessageType::Warning, 0),
            Some("WARNING: recoverable")
        );
        assert_eq!(
            program
                .get_process_messages()
                .get(super::super::process_messages::MessageType::Error, 0),
            Some("ERROR: failed")
        );
    }

    #[test]
    fn child_parser_honors_configured_prepend_tag() {
        let command = ProcessCommand::new("sh").args([
            "-c",
            "printf 'context: section 8\\nWARNING: missing data\\n'",
        ]);
        let mut program = SystemProgram::spawn(&command).unwrap();
        program.set_message_prepend_tag(Some("context:"));
        program.wait_and_drain().unwrap();
        assert_eq!(
            program
                .get_process_messages()
                .get(super::super::process_messages::MessageType::Warning, 0),
            Some("context: section 8\nWARNING: missing data")
        );
    }
}
