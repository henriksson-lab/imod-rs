//! Local execution core of `IMOD/Etomo/src/etomo/process/BackgroundProcess.java`.
//!
//! Swing/result-display and command-details ownership stay with their callers; this
//! owner is the source process lifecycle: a command, its local child, stream records,
//! and the terminal end state.

use super::process_data::ProcessData;
use super::process_messages::ProcessMessages;
use super::system_program::{ProcessCommand, ProcessLine, SystemProgram};
use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::process_details::ProcessDetails;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;
use std::rc::Rc;

pub struct BackgroundProcess {
    command: ProcessCommand,
    axis_id: AxisID,
    program: Option<SystemProgram>,
    output: Vec<ProcessLine>,
    end_state: Option<ProcessEndState>,
    process_data: Option<ProcessData>,
    demo_mode: bool,
    debug: bool,
    force_next_process: bool,
    process_series: Option<ProcessSeriesHandle>,
    process_result_display: Option<ProcessResultDisplayHandle>,
    command_details: Option<Rc<dyn CommandDetails>>,
    process_details: Option<Rc<dyn ProcessDetails>>,
}

impl BackgroundProcess {
    /// Java static `getInstance` factory forms collapse to the executable
    /// command and axis at this non-GUI boundary.  Process manager, display,
    /// and series links stay with their concrete Rust owners rather than
    /// becoming invented placeholders here.
    pub fn get_instance(command: ProcessCommand, axis_id: AxisID) -> Self {
        Self::new(command, axis_id)
    }

    /// Source constructor form for a String command array.
    pub fn new(command: ProcessCommand, axis_id: AxisID) -> Self {
        Self {
            command,
            axis_id,
            program: None,
            output: vec![],
            end_state: None,
            process_data: None,
            demo_mode: false,
            debug: false,
            force_next_process: false,
            process_series: None,
            process_result_display: None,
            command_details: None,
            process_details: None,
        }
    }
    /// Java `run` launch phase.
    pub fn start(&mut self) -> Result<(), String> {
        if self.program.is_none() {
            self.program =
                Some(SystemProgram::spawn(&self.command).map_err(|error| error.to_string())?);
        }
        Ok(())
    }
    /// Nonblocking monitor turn.  Output remains available after terminal status.
    pub fn poll(&mut self) -> Result<Option<ProcessEndState>, String> {
        let program = self
            .program
            .as_mut()
            .ok_or_else(|| "background process has not started".to_owned())?;
        self.output.extend(program.drain_lines());
        if self.end_state.is_none() {
            if let Some(status) = program.try_wait().map_err(|error| error.to_string())? {
                self.output.extend(program.drain_lines());
                self.end_state = Some(if status.success() {
                    ProcessEndState::Done
                } else {
                    ProcessEndState::Failed
                });
            }
        }
        Ok(self.end_state)
    }
    /// Java `run`: launch if needed, wait for the owned local child, retain
    /// every final stream record, and set the terminal process state.  The
    /// separate `start`/`poll` API remains available for event-loop callers.
    pub fn run(&mut self) -> Result<ProcessEndState, String> {
        self.start()?;
        let program = self
            .program
            .as_mut()
            .ok_or_else(|| "background process has not started".to_owned())?;
        let (status, lines) = program
            .wait_and_drain()
            .map_err(|error| error.to_string())?;
        self.output.extend(lines);
        let computed = if status.success() {
            ProcessEndState::Done
        } else {
            ProcessEndState::Failed
        };
        self.set_process_end_state(computed);
        Ok(self.end_state.expect("terminal state was just assigned"))
    }
    /// Java `pause`/resume toggling, delegated to the owned OS child.
    pub fn pause(&mut self) -> Result<(), String> {
        self.program
            .as_mut()
            .ok_or_else(|| "background process has not started".to_owned())?
            .pause()
            .map_err(|error| error.to_string())
    }
    pub fn resume(&mut self) -> Result<(), String> {
        self.program
            .as_mut()
            .ok_or_else(|| "background process has not started".to_owned())?
            .resume()
            .map_err(|error| error.to_string())
    }
    /// Java `signalKill`; ownership proves this is the child to terminate.
    pub fn kill(&mut self) -> Result<(), String> {
        self.program
            .as_mut()
            .ok_or_else(|| "background process has not started".to_owned())?
            .kill()
            .map_err(|error| error.to_string())?;
        self.set_process_end_state(ProcessEndState::Killed);
        Ok(())
    }
    /// Java `isNohup`: ordinary background processes are intentionally tied
    /// to eTomo's local child lifecycle.
    pub const fn is_nohup(&self) -> bool {
        false
    }
    /// Java `setComputerMap`.
    pub fn set_computer_map(&mut self, computer_map: Option<BTreeMap<String, String>>) {
        if let Some(data) = &mut self.process_data {
            data.set_computer_map(computer_map);
        }
    }
    /// Java `setSecondaryQueue`.
    pub fn set_secondary_queue(&mut self, secondary_queue: Option<&str>) {
        if let Some(data) = &mut self.process_data {
            data.set_secondary_queue(secondary_queue);
        }
    }
    /// Java `setProcessingMethod`.
    pub fn set_processing_method(&mut self, processing_method: Option<ProcessingMethod>) {
        if let Some(data) = &mut self.process_data {
            data.set_processing_method(processing_method);
        }
    }
    /// Attach Java's concrete persisted process owner when a manager created one.
    pub fn set_process_data(&mut self, process_data: Option<ProcessData>) {
        self.process_data = process_data;
    }
    /// Java `getProcessSeries`; returns the shared series owner rather than a
    /// copied workflow, preserving process-completion sequencing.
    pub fn get_process_series(&self) -> Option<ProcessSeriesHandle> {
        self.process_series.clone()
    }
    /// Attach the Java constructor's `ProcessSeries` reference.
    pub fn set_process_series(&mut self, process_series: Option<ProcessSeriesHandle>) {
        self.process_series = process_series;
    }
    /// Java `getProcessResultDisplay`.
    pub fn get_process_result_display(&self) -> Option<ProcessResultDisplayHandle> {
        self.process_result_display.clone()
    }
    /// Java `setProcessResultDisplay`.
    pub fn set_process_result_display(
        &mut self,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        self.process_result_display = process_result_display;
    }
    /// Java `getProcessData`.
    pub fn get_process_data(&self) -> Option<&ProcessData> {
        self.process_data.as_ref()
    }
    /// Java `resetProcessData`.
    pub fn reset_process_data(&mut self) {
        if let Some(data) = &mut self.process_data {
            data.reset();
        }
    }
    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        self.process_data
            .as_ref()
            .and_then(ProcessData::get_process_name)
    }
    /// Java `isDemoMode`.
    pub const fn is_demo_mode(&self) -> bool {
        self.demo_mode
    }
    /// Java package-private `setDemoMode`.
    pub fn set_demo_mode(&mut self, demo_mode: bool) {
        self.demo_mode = demo_mode;
    }
    /// Java `isDebug`.
    pub const fn is_debug(&self) -> bool {
        self.debug
    }
    /// Java package-private `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    /// Java `isForceNextProcess`.
    pub const fn is_force_next_process(&self) -> bool {
        self.force_next_process
    }
    /// Java `getWorkingDirectory`.
    pub fn get_working_directory(&self) -> Option<&Path> {
        self.command.working_directory.as_deref()
    }
    /// Java `getCommand`.
    pub const fn get_command(&self) -> &ProcessCommand {
        &self.command
    }
    /// Java `getCommandDetails`.  This is intentionally separate from the
    /// executable child description: command details may carry typed comscript
    /// parameters even when the process runs a generated shell array.
    pub fn get_command_details(&self) -> Option<Rc<dyn CommandDetails>> {
        self.command_details.clone()
    }
    /// Attach Java's nullable `commandDetails` constructor field.
    pub fn set_command_details(&mut self, command_details: Option<Rc<dyn CommandDetails>>) {
        self.command_details = command_details;
    }
    /// Java `setWorkingDirectory`, applied to the launch description before
    /// the child exists.  Changing it after start would not affect the source
    /// child either, so that is reported to the caller.
    pub fn set_working_directory(&mut self, directory: &Path) -> Result<(), String> {
        if self.program.is_some() {
            return Err("cannot change a started background process directory".to_owned());
        }
        self.command.set_working_directory(directory);
        Ok(())
    }
    /// Java `getCommandArray`.
    pub fn get_command_array(&self) -> Vec<OsString> {
        std::iter::once(self.command.program.clone())
            .chain(self.command.args.iter().cloned())
            .collect()
    }
    /// Java `getCommandLine` and its abbreviated overload.  The optional
    /// count retains Java's command-token rather than byte-oriented boundary.
    pub fn get_command_line(&self, end_index: Option<usize>) -> String {
        self.get_command_array()
            .into_iter()
            .take(end_index.unwrap_or(usize::MAX))
            .map(|word| word.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    }
    /// Java `getAbbreviatedCommandLine` fallback for the local command form.
    pub fn get_abbreviated_command_line(&self) -> String {
        self.get_command_line(Some(3))
    }
    /// Java `getCommandName`.
    pub fn get_command_name(&self) -> String {
        self.command.program.to_string_lossy().into_owned()
    }
    /// Java `getCommandAction`; a local `SystemProgram` has no separate
    /// monitor-supplied action, so its executable name is the source fallback.
    pub fn get_command_action(&self) -> String {
        self.get_command_name()
    }
    /// Java `getProcessDetails`.  Detached/parallel process paths can retain
    /// post-process field values without also claiming to be a runnable
    /// `CommandDetails` instance.
    pub fn get_process_details(&self) -> Option<Rc<dyn ProcessDetails>> {
        self.process_details.clone()
    }
    /// Attach Java's nullable `processDetails` constructor field.
    pub fn set_process_details(&mut self, process_details: Option<Rc<dyn ProcessDetails>>) {
        self.process_details = process_details;
    }
    /// Java `getProgram`.
    pub fn get_program(&self) -> Option<&SystemProgram> {
        self.program.as_ref()
    }
    /// Java `setProgram`, used by process-manager code that has already
    /// constructed the local child wrapper (for example after a lock retry).
    pub fn set_program(&mut self, program: SystemProgram) {
        self.program = Some(program);
    }
    /// Local portion of Java `processDone`: retain the terminal result after
    /// manager/UI notification has been performed by the caller boundary.
    pub fn process_done(&mut self, exit_value: i32) -> ProcessEndState {
        self.set_process_end_state(if exit_value == 0 {
            ProcessEndState::Done
        } else {
            ProcessEndState::Failed
        });
        self.end_state.expect("terminal state was just assigned")
    }
    /// Java `notifyKilled`.
    pub fn notify_killed(&mut self) {
        self.set_process_end_state(ProcessEndState::Killed);
    }
    /// Java `setProcessEndState`, including source state precedence.
    pub fn set_process_end_state(&mut self, end_state: ProcessEndState) {
        self.end_state = Some(match self.end_state {
            Some(existing) => ProcessEndState::precedence(existing, end_state),
            None => end_state,
        });
    }
    /// Java `getAxisID`.
    pub fn axis_id(&self) -> AxisID {
        self.axis_id
    }
    /// Java `getShellProcessID` for the locally owned child.
    pub fn get_shell_process_id(&self) -> String {
        self.program
            .as_ref()
            .map(|program| program.pid().to_string())
            .unwrap_or_default()
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        self.get_command_line(None)
    }
    pub fn output(&self) -> &[ProcessLine] {
        &self.output
    }
    /// Source `getStdOutput`: stdout records survive polling and do not expose
    /// stderr entries as normal output.
    pub fn get_std_output(&self) -> Vec<&str> {
        self.output
            .iter()
            .filter(|line| line.stream == super::system_program::ProcessStream::Stdout)
            .map(|line| line.line.as_str())
            .collect()
    }
    /// Source `getStdError`.
    pub fn get_std_error(&self) -> Vec<&str> {
        self.output
            .iter()
            .filter(|line| line.stream == super::system_program::ProcessStream::Stderr)
            .map(|line| line.line.as_str())
            .collect()
    }
    /// The source process owns `SystemProgram`, and therefore its parsed
    /// `ProcessMessages`, for the entire child lifetime.
    pub fn get_process_messages(&self) -> Option<&ProcessMessages> {
        self.program
            .as_ref()
            .map(SystemProgram::get_process_messages)
    }
    pub fn is_started(&self) -> bool {
        self.program.is_some()
    }
    pub fn is_done(&self) -> bool {
        self.end_state.is_some()
    }
    pub fn end_state(&self) -> Option<ProcessEndState> {
        self.end_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn child_lifecycle_collects_output_and_terminal_status() {
        let command = ProcessCommand::new("sh").args(["-c", "printf out; printf err >&2"]);
        let mut process = BackgroundProcess::new(command, AxisID::First);
        process.start().unwrap();
        while process.poll().unwrap().is_none() {
            std::thread::yield_now();
        }
        assert_eq!(process.end_state(), Some(ProcessEndState::Done));
        assert!(process.output().iter().any(|line| line.line == "out"));
        assert!(process.output().iter().any(|line| line.line == "err"));
        assert_eq!(process.get_std_output(), ["out"]);
        assert_eq!(process.get_std_error(), ["err"]);
        assert!(process.is_started());
        assert!(process.is_done());
    }

    #[cfg(unix)]
    #[test]
    fn source_run_factory_and_terminal_controls_retain_child_contract() {
        let command = ProcessCommand::new("sh").args(["-c", "printf completed"]);
        let mut process = BackgroundProcess::get_instance(command, AxisID::Second);
        assert!(!process.is_nohup());
        assert_eq!(process.to_source_string(), "sh -c printf completed");
        assert_eq!(process.get_command_name(), "sh");
        assert_eq!(process.get_command_action(), "sh");
        assert_eq!(
            process.get_abbreviated_command_line(),
            "sh -c printf completed"
        );
        assert_eq!(process.get_command_array().len(), 3);
        assert!(process.get_program().is_none());
        assert_eq!(process.run().unwrap(), ProcessEndState::Done);
        assert_eq!(process.get_std_output(), ["completed"]);
        assert!(!process.get_shell_process_id().is_empty());
        process.notify_killed();
        // Java precedence keeps a killed terminal state against later success.
        process.set_process_end_state(ProcessEndState::Done);
        assert_eq!(process.end_state(), Some(ProcessEndState::Killed));
        assert_eq!(process.process_done(0), ProcessEndState::Killed);
    }

    #[test]
    fn process_series_link_is_shared_not_a_copied_workflow() {
        use crate::imod::etomo::process_series::ProcessSeries;
        use std::cell::RefCell;
        use std::rc::Rc;

        let series = Rc::new(RefCell::new(ProcessSeries::new(
            None,
            AxisID::First,
            None,
            Some("background"),
        )));
        let mut process = BackgroundProcess::new(ProcessCommand::new("true"), AxisID::First);
        process.set_process_series(Some(series.clone()));
        assert!(Rc::ptr_eq(&process.get_process_series().unwrap(), &series));
    }
}
