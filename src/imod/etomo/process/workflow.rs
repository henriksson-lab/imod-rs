//! Typed local process workflow used by eTomo managers and COM-script runners.

use super::system_program::{ProcessCommand, ProcessLine, SystemProgram};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use std::collections::VecDeque;
use std::io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowState {
    Idle,
    Running,
    Paused,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowStep {
    pub name: String,
    pub command: ProcessCommand,
}

#[derive(Debug)]
pub struct WorkflowResult {
    pub name: String,
    pub success: bool,
    pub end_state: ProcessEndState,
    pub lines: Vec<ProcessLine>,
}

/// Sequential process-series core.  The GUI can decide how to display this;
/// process ordering, output retention, and terminal states live here.
pub struct ProcessWorkflow {
    pending: VecDeque<WorkflowStep>,
    active: Option<(String, SystemProgram)>,
    results: Vec<WorkflowResult>,
    state: WorkflowState,
}

impl Default for ProcessWorkflow {
    fn default() -> Self {
        Self::new()
    }
}
impl ProcessWorkflow {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            active: None,
            results: Vec::new(),
            state: WorkflowState::Idle,
        }
    }
    pub fn push(&mut self, name: impl Into<String>, command: ProcessCommand) {
        if matches!(
            self.state,
            WorkflowState::Complete | WorkflowState::Failed | WorkflowState::Cancelled
        ) {
            self.state = WorkflowState::Idle;
        }
        self.pending.push_back(WorkflowStep {
            name: name.into(),
            command,
        });
    }
    pub fn state(&self) -> WorkflowState {
        self.state
    }
    pub fn results(&self) -> &[WorkflowResult] {
        &self.results
    }
    pub fn active_pid(&self) -> Option<u32> {
        self.active.as_ref().map(|(_, program)| program.pid())
    }
    pub fn active_name(&self) -> Option<&str> {
        self.active.as_ref().map(|(name, _)| name.as_str())
    }
    pub fn start(&mut self) -> io::Result<bool> {
        if self.active.is_some() {
            return Ok(false);
        }
        let Some(step) = self.pending.pop_front() else {
            self.state = WorkflowState::Complete;
            return Ok(false);
        };
        match SystemProgram::spawn(&step.command) {
            Ok(program) => {
                self.active = Some((step.name, program));
                self.state = WorkflowState::Running;
                Ok(true)
            }
            Err(error) => {
                self.results.push(WorkflowResult {
                    name: step.name,
                    success: false,
                    end_state: ProcessEndState::Failed,
                    lines: Vec::new(),
                });
                self.state = WorkflowState::Failed;
                Err(error)
            }
        }
    }
    /// Advance one child and automatically start the next successful step.
    pub fn poll(&mut self) -> io::Result<WorkflowState> {
        if self.state == WorkflowState::Paused {
            return Ok(self.state);
        }
        let Some((_, program)) = self.active.as_mut() else {
            if self.state == WorkflowState::Idle {
                let _ = self.start()?;
            }
            return Ok(self.state);
        };
        let status = program.try_wait()?;
        if let Some(status) = status {
            let (name, mut program) = self.active.take().expect("active process checked above");
            let lines = program.drain_lines();
            let success = status.success();
            self.results.push(WorkflowResult {
                name,
                success,
                end_state: if success {
                    ProcessEndState::Done
                } else {
                    ProcessEndState::Failed
                },
                lines,
            });
            if success {
                self.state = WorkflowState::Idle;
                let _ = self.start()?;
            } else {
                self.state = WorkflowState::Failed;
            }
        }
        Ok(self.state)
    }
    pub fn cancel(&mut self) -> io::Result<()> {
        if let Some((_, program)) = self.active.as_mut() {
            program.kill()?;
        }
        self.pending.clear();
        self.state = WorkflowState::Cancelled;
        Ok(())
    }
    /// Java `ProcessInterface.pause`: stop the active local child without
    /// advancing the series.  Calling it again resumes the same child.
    pub fn pause(&mut self) -> io::Result<bool> {
        let Some((_, program)) = self.active.as_mut() else {
            return Ok(false);
        };
        if self.state == WorkflowState::Paused {
            program.resume()?;
            self.state = WorkflowState::Running;
        } else {
            program.pause()?;
            self.state = WorkflowState::Paused;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn successful_steps_run_in_order() {
        let mut workflow = ProcessWorkflow::new();
        workflow.push("one", ProcessCommand::new("true"));
        workflow.push("two", ProcessCommand::new("true"));
        workflow.start().unwrap();
        while workflow.state() == WorkflowState::Running || workflow.state() == WorkflowState::Idle
        {
            workflow.poll().unwrap();
        }
        assert_eq!(workflow.state(), WorkflowState::Complete);
        assert_eq!(
            workflow
                .results()
                .iter()
                .map(|result| result.name.as_str())
                .collect::<Vec<_>>(),
            ["one", "two"]
        );
    }
    #[cfg(unix)]
    #[test]
    fn active_child_can_be_paused_resumed_and_cancelled() {
        let mut workflow = ProcessWorkflow::new();
        workflow.push("wait", ProcessCommand::new("sh").args(["-c", "sleep 10"]));
        assert!(workflow.start().unwrap());
        assert!(workflow.pause().unwrap());
        assert_eq!(workflow.state(), WorkflowState::Paused);
        assert_eq!(workflow.poll().unwrap(), WorkflowState::Paused);
        assert!(workflow.pause().unwrap());
        assert_eq!(workflow.state(), WorkflowState::Running);
        workflow.cancel().unwrap();
        assert_eq!(workflow.state(), WorkflowState::Cancelled);
    }
}
