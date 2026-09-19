//! Local-process portion of `IMOD/Etomo/src/etomo/process/ReconnectProcess.java`.
//!
//! The Java unit reconnects a persisted `ProcessData` record to its monitor
//! and log reader.  This Rust owner supplies the part that can be authoritative
//! without Swing or a remote monitor: it watches the recorded local PID and
//! exposes source-shaped terminal state.  Remote/log-backed reconnects remain
//! callers of their translated monitor/log owners instead of being reported as
//! successful local reconnects.

use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReconnectError {
    MissingProcessData,
    RemoteMonitorRequired,
    MonitorRequired,
}

/// Java `ReconnectProcess` for a locally observable process record.
pub struct ReconnectProcess {
    process_data: Option<ProcessData>,
    axis_id: AxisID,
    reconnect_when_not_running: bool,
    end_state: Option<ProcessEndState>,
    started: bool,
    log_file: Option<PathBuf>,
    process_series: Option<ProcessSeriesHandle>,
    process_result_display: Option<ProcessResultDisplayHandle>,
}

impl ReconnectProcess {
    /// Java constructor's local-data form.  The absent Java manager/monitor
    /// references are intentionally not fabricated here.
    pub fn new(
        process_data: Option<ProcessData>,
        axis_id: AxisID,
        reconnect_when_not_running: bool,
    ) -> Self {
        Self {
            process_data,
            axis_id,
            reconnect_when_not_running,
            end_state: None,
            started: false,
            log_file: None,
            process_series: None,
            process_result_display: None,
        }
    }
    /// Local equivalent of Java static `getInstance`.  The Java factory's
    /// manager-derived log handle is passed explicitly at this Rust boundary.
    pub fn get_instance(
        process_data: Option<ProcessData>,
        axis_id: AxisID,
        log_file: Option<&Path>,
    ) -> Self {
        let mut reconnect = Self::new(process_data, axis_id, false);
        reconnect.set_log_file(log_file);
        reconnect
    }

    /// Local equivalent of Java static `getLogInstance`.  A log-controlled
    /// reconnect must continue even after the saved local PID stops, hence
    /// `reconnect_when_not_running` is retained from the source factory.
    pub fn get_log_instance(
        process_data: Option<ProcessData>,
        axis_id: AxisID,
        log_file: Option<&Path>,
        reconnect_when_not_running: bool,
    ) -> Self {
        let mut reconnect = Self::new(process_data, axis_id, reconnect_when_not_running);
        reconnect.set_log_file(log_file);
        reconnect
    }
    /// Source `continueRun` for the local `ProcessData.isRunning()` branch.
    pub fn continue_run(&self) -> Result<bool, ReconnectError> {
        let data = self
            .process_data
            .as_ref()
            .ok_or(ReconnectError::MissingProcessData)?;
        if data.is_on_different_host() {
            return Err(ReconnectError::RemoteMonitorRequired);
        }
        if self.reconnect_when_not_running {
            return Err(ReconnectError::MonitorRequired);
        }
        Ok(data.is_running())
    }
    /// One nonblocking equivalent of the source's reconnect loop.
    pub fn poll(&mut self) -> Result<Option<ProcessEndState>, ReconnectError> {
        self.started = true;
        if self.end_state.is_some() {
            return Ok(self.end_state);
        }
        if !self.continue_run()? {
            self.end_state = Some(ProcessEndState::Done);
        }
        Ok(self.end_state)
    }

    /// Java `run`, represented as one caller-scheduled reconnect turn.  The
    /// native implementation sleeps between monitor turns; Rust callers own
    /// that scheduling while this retains the same terminal state transition.
    pub fn run(&mut self) -> Result<Option<ProcessEndState>, ReconnectError> {
        self.poll()
    }
    /// Java `kill(AxisID)`: the reconnect does not own the original child, so
    /// it records the terminal notification instead of sending a signal to an
    /// unrelated PID.
    pub fn kill(&mut self, axis_id: AxisID) {
        if axis_id == self.axis_id {
            self.end_state = Some(ProcessEndState::Killed);
        }
    }
    pub fn is_started(&self) -> bool {
        self.started
    }
    pub fn is_done(&self) -> bool {
        self.end_state.is_some()
    }
    pub fn get_process_end_state(&self) -> Option<ProcessEndState> {
        self.end_state
    }
    /// Java `setProcessEndState` for a reconnect without a monitor owner.
    pub fn set_process_end_state(&mut self, end_state: Option<ProcessEndState>) {
        self.end_state = end_state;
    }
    /// Java `getAxisID`.
    pub const fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }
    pub fn get_process_data(&self) -> Option<&ProcessData> {
        self.process_data.as_ref()
    }
    /// Java `getProcessSeries`.
    pub fn get_process_series(&self) -> Option<ProcessSeriesHandle> {
        self.process_series.clone()
    }
    /// Attach the constructor's mutable series link.
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
    pub fn get_shell_process_id(&self) -> Option<String> {
        self.process_data.as_ref().and_then(ProcessData::get_pid)
    }
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.process_data
            .as_ref()
            .and_then(ProcessData::get_processing_method)
    }

    /// Java `toString`: prefer the persisted process name, then the attached
    /// reconnect log, before falling back to a stable Rust diagnostic name.
    pub fn to_source_string(&self) -> String {
        if let Some(name) = self
            .process_data
            .as_ref()
            .and_then(ProcessData::get_process_name)
        {
            return name.to_string();
        }
        self.log_file
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "ReconnectProcess".to_owned())
    }

    /// Java `setComputerMap`: reconnect uses the persisted ProcessData map,
    /// so a monitor must not overwrite it.
    pub fn set_computer_map(&mut self, _computer_map: Option<BTreeMap<String, String>>) {}

    /// Java `setSecondaryQueue`: reconnect preserves its persisted queue.
    pub fn set_secondary_queue(&mut self, _secondary_queue: Option<&str>) {}

    /// Java `setProcessingMethod`: reconnect preserves ProcessData's method.
    pub fn set_processing_method(&mut self, _processing_method: Option<ProcessingMethod>) {}

    /// Attach the log read by Java `getStdOutput` after reconnect completion.
    pub fn set_log_file(&mut self, path: Option<&Path>) {
        self.log_file = path.map(Path::to_owned);
    }

    /// Java `getStdError`: reconnect has no independent stderr stream.
    pub const fn get_std_error(&self) -> Option<&[String]> {
        None
    }

    /// Java `getStdOutput`: return nonempty persisted log lines, if readable.
    pub fn get_std_output(&self) -> Option<Vec<String>> {
        let path = self.log_file.as_ref()?;
        let text = std::fs::read_to_string(path).ok()?;
        let lines: Vec<_> = text.lines().map(str::to_owned).collect();
        (!lines.is_empty()).then_some(lines)
    }

    /// Java `isNohup`: reconnecting always refers to an existing detached job.
    pub const fn is_nohup(&self) -> bool {
        true
    }

    /// Java `notifyKilled`.
    pub fn notify_killed(&mut self) {
        self.end_state = Some(ProcessEndState::Killed);
    }

    /// Java `signalKill`.  A local reconnect observes but does not own the
    /// original child, therefore signaling records the same terminal request.
    pub fn signal_kill(&mut self, axis_id: AxisID) {
        self.kill(axis_id);
    }

    /// Java `pause`: without the Java monitor there is no safe remote pause
    /// operation, matching the source's unavailable-function branch.
    pub const fn pause(&mut self, _axis_id: AxisID) -> bool {
        false
    }
    /// Java `resetProcessData` after a monitor-controlled reconnect.
    pub fn reset_process_data(&mut self) {
        if let Some(data) = &mut self.process_data {
            data.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::process::system_program::{ProcessCommand, SystemProgram};

    #[cfg(unix)]
    #[test]
    fn local_pid_reconnect_tracks_running_then_completed_child() {
        let mut child =
            SystemProgram::spawn(&ProcessCommand::new("sh").args(["-c", "sleep 0.05"])).unwrap();
        let mut data = ProcessData::new(Some(AxisID::First), None);
        data.set_local_process(child.pid(), None);
        let mut reconnect = ReconnectProcess::new(Some(data), AxisID::First, false);
        assert_eq!(reconnect.poll().unwrap(), None);
        assert!(reconnect.is_started());
        child.wait().unwrap();
        assert_eq!(reconnect.poll().unwrap(), Some(ProcessEndState::Done));
    }

    #[test]
    fn reconnect_without_observable_pid_needs_a_monitor() {
        let data = ProcessData::new(Some(AxisID::First), None);
        let mut reconnect = ReconnectProcess::new(Some(data), AxisID::First, true);
        assert_eq!(reconnect.poll(), Err(ReconnectError::MonitorRequired));
        reconnect.kill(AxisID::First);
        assert_eq!(
            reconnect.get_process_end_state(),
            Some(ProcessEndState::Killed)
        );
    }

    #[test]
    fn reconnect_log_output_and_nonowning_controls_follow_source_contract() {
        let root = std::env::temp_dir().join(format!(
            "imod-rs-reconnect-log-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&root, "first\nsecond\n").unwrap();
        let mut reconnect = ReconnectProcess::new(None, AxisID::First, false);
        reconnect.set_log_file(Some(&root));
        assert_eq!(
            reconnect.get_std_output(),
            Some(vec!["first".to_owned(), "second".to_owned()])
        );
        assert_eq!(reconnect.get_std_error(), None);
        assert!(reconnect.is_nohup());
        assert!(!reconnect.pause(AxisID::First));
        reconnect.signal_kill(AxisID::First);
        assert_eq!(
            reconnect.get_process_end_state(),
            Some(ProcessEndState::Killed)
        );
        std::fs::remove_file(root).unwrap();
    }

    #[test]
    fn source_factories_and_terminal_state_accessors_retain_local_ownership() {
        let mut reconnect = ReconnectProcess::get_instance(None, AxisID::Second, None);
        assert_eq!(reconnect.get_axis_id(), AxisID::Second);
        reconnect.set_process_end_state(Some(ProcessEndState::Paused));
        assert_eq!(
            reconnect.get_process_end_state(),
            Some(ProcessEndState::Paused)
        );
        let log_reconnect = ReconnectProcess::get_log_instance(None, AxisID::First, None, true);
        assert_eq!(
            log_reconnect.continue_run(),
            Err(ReconnectError::MissingProcessData)
        );
    }

    #[test]
    fn reconnect_retains_the_constructor_series_link() {
        use crate::imod::etomo::process_series::ProcessSeries;
        use std::cell::RefCell;
        use std::rc::Rc;

        let series = Rc::new(RefCell::new(ProcessSeries::new(
            None,
            AxisID::Second,
            None,
            Some("reconnect"),
        )));
        let mut reconnect = ReconnectProcess::new(None, AxisID::Second, false);
        reconnect.set_process_series(Some(series.clone()));
        assert!(Rc::ptr_eq(
            &reconnect.get_process_series().unwrap(),
            &series
        ));
    }
}
