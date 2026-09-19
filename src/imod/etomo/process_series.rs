//! `IMOD/Etomo/src/etomo/ProcessSeries.java`.
//!
//! The next/process-list/last/fail/pause queue remains a concrete state machine.
//! Manager dispatch, Swing displays, deferred-3dmod buttons, commands, targets, and
//! processing methods retain explicit null-equivalent boundaries until their source
//! units are available; this module never reports an unstarted process as started.
#![allow(dead_code)]

use crate::imod::etomo::comscript::com_script_file::ComScriptFile;
use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process::system_program::ProcessCommand;
use crate::imod::etomo::process::workflow::{ProcessWorkflow, WorkflowResult, WorkflowState};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use std::cell::RefCell;
use std::convert::Infallible;
use std::path::Path;
use std::rc::Rc;

/// Shared mutable link used where Java stores a `ProcessSeries` reference in
/// a process object.  eTomo schedules these on its one UI thread, so this is
/// the direct ownership analogue of the Java reference rather than a copied
/// queue snapshot.
pub type ProcessSeriesHandle = Rc<RefCell<ProcessSeries>>;

/// Java final `ProcessSeries implements ConstProcessSeries`.
pub struct ProcessSeries {
    manager: Option<Infallible>,
    dialog_type: Option<DialogType>,
    process_display: Option<Infallible>,
    ui_component: Option<Infallible>,
    busy_status_mediator: Option<Infallible>,
    busy_axis_id: AxisID,
    descr: Option<String>,
    next_process: Option<Box<Process>>,
    process_list: Option<Box<Process>>,
    last_process: Option<Box<Process>>,
    run_3dmod_button: Option<Infallible>,
    run_3dmod_menu_options: Option<Infallible>,
    fail_process: Option<Box<Process>>,
    debug: bool,
    force_next_process: bool,
    pause_process: Option<Box<Process>>,
    /// Concrete local queue used while parameter-specific Java process types
    /// are translated.  It preserves ProcessSeries ordering independently of
    /// the legacy nullable task graph above.
    typed_workflow: ProcessWorkflow,
}
impl ProcessSeries {
    /// Java `(BaseManager,AxisID,DialogType,String)` constructor.
    pub fn new(
        manager: Option<Infallible>,
        busy_axis_id: AxisID,
        dialog_type: Option<DialogType>,
        descr: Option<&str>,
    ) -> Self {
        Self {
            manager,
            dialog_type,
            process_display: None,
            ui_component: None,
            busy_status_mediator: None,
            busy_axis_id,
            descr: descr.map(str::to_owned),
            next_process: None,
            process_list: None,
            last_process: None,
            run_3dmod_button: None,
            run_3dmod_menu_options: None,
            fail_process: None,
            debug: false,
            force_next_process: false,
            pause_process: None,
            typed_workflow: ProcessWorkflow::new(),
        }
    }
    /// Queue a source-named concrete command in this series.
    pub fn queue_local_command(&mut self, name: impl Into<String>, command: ProcessCommand) {
        self.typed_workflow.push(name, command);
    }
    /// Load a translated `.com` file into this series's concrete execution
    /// queue.  The source `ProcessSeries` owns sequencing while the command
    /// parser owns standard-input and continuation reconstruction; keeping the
    /// handoff here makes parsed COM blocks execute in their source order.
    pub fn queue_com_script(&mut self, path: &Path) -> std::io::Result<usize> {
        let script = ComScriptFile::load(path)?;
        let commands = script.process_commands();
        let count = commands.len();
        for (name, command) in commands {
            self.queue_local_command(name, command);
        }
        Ok(count)
    }
    pub fn start_local_workflow(&mut self) -> std::io::Result<bool> {
        self.typed_workflow.start()
    }
    pub fn poll_local_workflow(&mut self) -> std::io::Result<WorkflowState> {
        self.typed_workflow.poll()
    }
    pub fn local_workflow_state(&self) -> WorkflowState {
        self.typed_workflow.state()
    }
    /// Terminal child records for the concrete portion of the series.  This
    /// is the Rust owner that result-display/manager routes consume instead of
    /// reconstructing success from a process name or an absent task object.
    pub fn local_workflow_results(&self) -> &[WorkflowResult] {
        self.typed_workflow.results()
    }
    /// Java UIComponent constructor.
    pub fn new_with_ui_component(
        manager: Option<Infallible>,
        busy_axis_id: AxisID,
        ui_component: Option<Infallible>,
        dialog_type: Option<DialogType>,
        descr: Option<&str>,
    ) -> Self {
        let mut value = Self::new(manager, busy_axis_id, dialog_type, descr);
        value.ui_component = ui_component;
        value
    }
    /// Java ProcessDisplay constructor.
    pub fn new_with_process_display(
        manager: Option<Infallible>,
        busy_axis_id: AxisID,
        dialog_type: Option<DialogType>,
        process_display: Option<Infallible>,
        descr: Option<&str>,
    ) -> Self {
        let mut value = Self::new(manager, busy_axis_id, dialog_type, descr);
        value.process_display = process_display;
        value
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!(
            "ProcessSeries:[forceNextProcess:{},nextProcess:{:?},processList:{:?},lastProcess:{:?},failProcess:{:?},pauseProcess:{:?}]",
            self.force_next_process,
            self.next_process,
            self.process_list,
            self.last_process,
            self.fail_process,
            self.pause_process
        )
    }
    /// Java `dumpState`.
    pub fn dump_state(&self) {
        eprint!("[debug:{}]", self.debug);
    }
    /// Java `startNextProcess(AxisID)`.
    pub fn start_next_process(&mut self, axis_id: AxisID) -> bool {
        self.start_next_process_with_display(axis_id, None)
    }
    /// Java `startNextProcess(AxisID,ProcessResultDisplay)`.  A dequeued process needs
    /// BaseManager/NextProcessTarget dispatch, so it is left queued at that frontier.
    pub fn start_next_process_with_display(
        &mut self,
        axis_id: AxisID,
        display: Option<Infallible>,
    ) -> bool {
        let _ = (axis_id, display);
        if self.typed_workflow.state() == WorkflowState::Idle {
            if let Ok(started) = self.typed_workflow.start() {
                if started {
                    return true;
                }
            }
        }
        if matches!(
            self.typed_workflow.state(),
            WorkflowState::Running | WorkflowState::Paused
        ) {
            return true;
        }
        if self.next_process.is_some() || self.process_list.is_some() || self.last_process.is_some()
        {
            false
        } else if self.run_3dmod_button.is_some() {
            self.start_3dmod_process();
            false
        } else {
            self.clear_processes();
            false
        }
    }
    /// Java `startFailProcess(AxisID)`.
    pub fn start_fail_process(&mut self, axis_id: AxisID) {
        self.start_fail_process_with_display(axis_id, None);
    }
    /// Java `startFailProcess(AxisID,ProcessResultDisplay)`.
    pub fn start_fail_process_with_display(
        &mut self,
        axis_id: AxisID,
        display: Option<Infallible>,
    ) {
        let _ = (axis_id, display);
        if self.force_next_process {
            let _ = self.start_next_process(axis_id);
            return;
        }
        self.clear_processes();
    }
    /// Java `killSeries`.
    pub fn kill_series(&mut self, axis_id: AxisID, display: Option<Infallible>) {
        let _ = (axis_id, display);
        let _ = self.typed_workflow.cancel();
        self.clear_processes();
    }
    /// Java `endSeries`.
    pub fn end_series(&self) {}
    /// Java `startPauseProcess`.
    pub fn start_pause_process(&mut self, axis_id: AxisID, display: Option<Infallible>) -> bool {
        let _ = (axis_id, display);
        if self.typed_workflow.pause().unwrap_or(false) {
            return true;
        }
        let exists = self.pause_process.is_some();
        self.clear_processes();
        exists && false
    }
    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    /// Java `getDialogType`.
    pub fn get_dialog_type(&self) -> Option<DialogType> {
        self.dialog_type
    }
    /// Java `getLastProcess`.
    pub fn get_last_process(&self) -> Option<String> {
        self.last_process.as_ref().and_then(|p| p.get_process())
    }
    /// Java String/ProcessingMethod `setNextProcess`.
    pub fn set_next_process(&mut self, process: Option<&str>, method: Option<ProcessingMethod>) {
        self.next_process = Some(Box::new(Process::new(
            process, None, None, None, method, None, false, None, None, None, None,
        )));
    }
    /// Java `prependNextProcess(TaskInterface)`.
    pub fn prepend_next_process(&mut self, task: Option<Infallible>) {
        let old = self.next_process.take();
        self.prepend_process(old);
        self.next_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, None,
        )));
    }
    /// Java TaskInterface `setNextProcess`.
    pub fn set_next_process_task(&mut self, task: Option<Infallible>) {
        self.next_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, None,
        )));
    }
    /// Java TaskInterface/String overload.
    pub fn set_next_process_task_parameter(
        &mut self,
        task: Option<Infallible>,
        parameter: Option<&str>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, parameter,
        )));
    }
    /// Java NextProcessTarget/TaskInterface overload.
    pub fn set_next_process_target_task(
        &mut self,
        target: Option<Infallible>,
        task: Option<Infallible>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, target, None, None,
        )));
    }
    /// Java TaskInterface/Command overload.
    pub fn set_next_process_task_command(
        &mut self,
        task: Option<Infallible>,
        command: Option<Infallible>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, command, None, None, None,
        )));
    }
    /// Java `setPauseProcess`.
    pub fn set_pause_process(&mut self, task: Option<Infallible>) {
        self.pause_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, None,
        )));
    }
    /// Java `addProcess(TaskInterface)`.
    pub fn add_process(&mut self, task: Option<Infallible>) {
        self.add_process_force(task, false);
    }
    /// Java TaskInterface/bool `addProcess`.
    pub fn add_process_force(&mut self, task: Option<Infallible>, force: bool) {
        self.append_process(Some(Box::new(Process::new(
            None, None, None, None, None, task, force, None, None, None, None,
        ))));
    }
    /// Java TaskInterface/Command/AxisType `addProcess`.
    pub fn add_process_command_axis_type(
        &mut self,
        task: Option<Infallible>,
        command: Option<Infallible>,
        axis_type: Option<AxisType>,
    ) {
        self.append_process(Some(Box::new(Process::new(
            None, None, None, None, None, task, false, command, None, axis_type, None,
        ))));
    }
    /// Java private `prependProcess`.
    fn prepend_process(&mut self, mut process: Option<Box<Process>>) {
        if let Some(mut process) = process.take() {
            process.next = self.process_list.take();
            self.process_list = Some(process);
        }
    }
    /// Java private `appendProcess`.
    fn append_process(&mut self, process: Option<Box<Process>>) {
        let Some(process) = process else { return };
        let mut cursor = &mut self.process_list;
        while let Some(node) = cursor {
            cursor = &mut node.next;
        }
        *cursor = Some(process);
    }
    /// Java String/ProcessName/ProcessingMethod `setNextProcess`.
    pub fn set_next_process_subprocess(
        &mut self,
        process: Option<&str>,
        subprocess: Option<ProcessName>,
        method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process, subprocess, None, None, method, None, false, None, None, None, None,
        )));
    }
    /// Java TaskInterface/ProcessName/ProcessingMethod overload.
    pub fn set_next_process_task_subprocess(
        &mut self,
        task: Option<Infallible>,
        subprocess: Option<ProcessName>,
        method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None, subprocess, None, None, method, task, false, None, None, None, None,
        )));
    }
    /// Java String/ProcessName/FileType/ProcessingMethod overload.
    pub fn set_next_process_output_file_type(
        &mut self,
        process: Option<&str>,
        subprocess: Option<ProcessName>,
        file_type: Option<Infallible>,
        method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process, subprocess, file_type, None, method, None, false, None, None, None, None,
        )));
    }
    /// Java String/ProcessName/FileKey/FileKey/ProcessingMethod overload.
    pub fn set_next_process_output_file_keys(
        &mut self,
        process: Option<&str>,
        subprocess: Option<ProcessName>,
        key: Option<Infallible>,
        key2: Option<Infallible>,
        method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process, subprocess, key, key2, method, None, false, None, None, None, None,
        )));
    }
    /// Java package-private `clearProcesses`.
    pub fn clear_processes(&mut self) {
        self.next_process = None;
        self.process_list = None;
        self.last_process = None;
        self.run_3dmod_button = None;
        self.run_3dmod_menu_options = None;
        self.pause_process = None;
        self.force_next_process = false;
        self.fail_process = None;
    }
    /// Java `setNextProcessParameter`.
    pub fn set_next_process_parameter(&mut self, input: Option<&str>) {
        let process = if self.next_process.is_some() {
            self.next_process.as_deref_mut()
        } else if self.process_list.is_some() {
            self.process_list.as_deref_mut()
        } else {
            self.last_process.as_deref_mut()
        };
        if let Some(process) = process {
            process.set_parameter(input);
        }
    }
    /// Java `willProcessBeDropped(ProcessData)`.
    pub fn will_process_be_dropped(&self, data: Option<&ProcessData>) -> bool {
        self.will_process_be_dropped_inner(data, false)
    }
    /// Java private `willProcessBeDropped(ProcessData,boolean)`.
    fn will_process_be_dropped_inner(&self, data: Option<&ProcessData>, list_only: bool) -> bool {
        if !list_only && self.will_process_be_dropped_process(self.next_process.as_deref()) {
            return true;
        }
        let mut current = self.process_list.as_deref();
        while let Some(process) = current {
            if self.will_process_be_dropped_process(Some(process)) {
                return true;
            }
            current = process.next.as_deref();
        }
        if list_only {
            return false;
        }
        if self.will_process_be_dropped_process(self.last_process.as_deref()) {
            return true;
        }
        if let (Some(data), Some(last)) = (data, self.last_process.as_deref()) {
            return self.dialog_type != data.get_dialog_type()
                || last.get_process() != data.get_last_process();
        }
        false
    }
    /// Java `willProcessListBeDropped`.
    pub fn will_process_list_be_dropped(&self) -> bool {
        self.will_process_be_dropped_inner(None, true)
    }
    /// Java private `willProcessBeDropped(Process)`.
    fn will_process_be_dropped_process(&self, process: Option<&Process>) -> bool {
        process.is_some_and(|process| process.task.is_none())
    }
    /// Java `peekNextProcess`.
    pub fn peek_next_process(&self) -> Option<String> {
        self.peek()
            .and_then(Process::get_process)
            .or_else(|| self.run_3dmod_button.as_ref().map(|_| "3dmod".into()))
    }
    /// Java private `peek`.
    fn peek(&self) -> Option<&Process> {
        self.next_process
            .as_deref()
            .or(self.process_list.as_deref())
            .or(self.last_process.as_deref())
    }
    /// Java String `setLastProcess`.
    pub fn set_last_process(&mut self, process: Option<&str>) {
        self.last_process = Some(Box::new(Process::new(
            process, None, None, None, None, None, false, None, None, None, None,
        )));
    }
    /// Java NextProcessTarget/String `setLastProcess`.
    pub fn set_last_process_target(&mut self, target: Option<Infallible>, process: Option<&str>) {
        self.last_process = Some(Box::new(Process::new(
            process, None, None, None, None, None, false, None, target, None, None,
        )));
    }
    /// Java TaskInterface `setLastProcess`.
    pub fn set_last_process_task(&mut self, task: Option<Infallible>) {
        self.last_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, None,
        )));
    }
    /// Java NextProcessTarget/TaskInterface `setLastProcess`.
    pub fn set_last_process_target_task(
        &mut self,
        target: Option<Infallible>,
        task: Option<Infallible>,
    ) {
        self.last_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, target, None, None,
        )));
    }
    /// Java `setFailProcess`.
    pub fn set_fail_process(&mut self, task: Option<Infallible>) {
        self.fail_process = Some(Box::new(Process::new(
            None, None, None, None, None, task, false, None, None, None, None,
        )));
    }
    /// Java `setRun3dmodDeferred`.
    pub fn set_run_3dmod_deferred(
        &mut self,
        button: Option<Infallible>,
        options: Option<Infallible>,
    ) {
        if button.is_some() {
            self.run_3dmod_button = button;
            self.run_3dmod_menu_options = options;
        }
    }
    /// Java private `sendMsgSecondaryProcess`.
    fn send_msg_secondary_process(&self, display: Option<Infallible>) {
        let _ = display;
    }
    /// Java private `start3dmodProcess`; button action is an explicit UI boundary.
    fn start_3dmod_process(&mut self) {
        self.run_3dmod_button = None;
        self.run_3dmod_menu_options = None;
    }
}
/// Java public static final nested `Process`.
#[derive(Debug)]
pub struct Process {
    process: Option<String>,
    subprocess_name: Option<ProcessName>,
    output_image_file_key: Option<Infallible>,
    output_image_file_key2: Option<Infallible>,
    processing_method: Option<ProcessingMethod>,
    task: Option<Infallible>,
    force_next_process: bool,
    command: Option<Infallible>,
    target: Option<Infallible>,
    axis_type: Option<AxisType>,
    next: Option<Box<Process>>,
    parameter: Option<String>,
}
impl Process {
    /// Java private constructor.
    fn new(
        process: Option<&str>,
        subprocess_name: Option<ProcessName>,
        key: Option<Infallible>,
        key2: Option<Infallible>,
        method: Option<ProcessingMethod>,
        task: Option<Infallible>,
        force: bool,
        command: Option<Infallible>,
        target: Option<Infallible>,
        axis_type: Option<AxisType>,
        parameter: Option<&str>,
    ) -> Self {
        Self {
            process: process.map(str::to_owned),
            subprocess_name,
            output_image_file_key: key,
            output_image_file_key2: key2,
            processing_method: method,
            task,
            force_next_process: force,
            command,
            target,
            axis_type,
            next: None,
            parameter: parameter.map(str::to_owned),
        }
    }
    /// Java `getDescr`.
    pub fn get_descr(&self) -> Option<String> {
        let name = self.process.clone().filter(|p| !p.contains('@'));
        let sub = self.subprocess_name.map(|p| p.to_string());
        match (name, sub) {
            (Some(name), Some(sub)) => Some(format!("{name} {sub}")),
            (Some(name), None) => Some(name),
            (None, Some(sub)) => Some(sub),
            (None, None) => None,
        }
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!(
            "Process:[forceNextProcess:{},process:{:?},task:{:?},target:{:?}]",
            self.force_next_process, self.process, self.task, self.target
        )
    }
    /// Java `setParameter`.
    pub fn set_parameter(&mut self, input: Option<&str>) {
        self.parameter = input.map(str::to_owned);
    }
    /// Java private `getProcess`.
    pub fn get_process(&self) -> Option<String> {
        self.process.clone()
    }
    /// Java `getAxisType`.
    pub fn get_axis_type(&self) -> Option<AxisType> {
        self.axis_type
    }
    /// Java `getParameter`.
    pub fn get_parameter(&self) -> Option<String> {
        self.parameter.clone()
    }
    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<Infallible> {
        self.command
    }
    /// Java `getSubprocessName`.
    pub fn get_subprocess_name(&self) -> Option<ProcessName> {
        self.subprocess_name
    }
    /// Java deprecated `getOutputImageFileType`.
    pub fn get_output_image_file_type(&self) -> Option<Infallible> {
        self.output_image_file_key
    }
    /// Java `getOutputImageFileKey`.
    pub fn get_output_image_file_key(&self) -> Option<Infallible> {
        self.output_image_file_key
    }
    /// Java `equals(TaskInterface)`.
    pub fn equals_task(&self, task: Option<Infallible>) -> bool {
        self.task == task
    }
    /// Java `equals(String)`.
    pub fn equals_string(&self, input: Option<&str>) -> bool {
        self.process.as_deref() == input
    }
    /// Java `getTask`.
    pub fn get_task(&self) -> Option<Infallible> {
        self.task
    }
    /// Java `equals(ProcessName)`.
    pub fn equals_process_name(&self, name: Option<ProcessName>) -> bool {
        self.process.as_deref() == name.map(|n| n.to_string()).as_deref()
    }
    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.processing_method
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queue_order_and_reset_follow_source() {
        let mut s = ProcessSeries::new(None, AxisID::First, None, Some("x"));
        s.set_last_process(Some("last"));
        s.set_next_process(Some("next"), None);
        s.add_process(None);
        assert_eq!(s.peek_next_process().as_deref(), Some("next"));
        assert!(s.will_process_list_be_dropped());
        s.clear_processes();
        assert_eq!(s.peek_next_process(), None);
    }
    #[test]
    fn typed_workflow_is_started_by_series_dispatch() {
        let mut series = ProcessSeries::new(None, AxisID::First, None, None);
        series.queue_local_command("probe", ProcessCommand::new("true"));
        assert!(series.start_next_process(AxisID::First));
        assert_eq!(series.local_workflow_state(), WorkflowState::Running);
        series.kill_series(AxisID::First, None);
        assert_eq!(series.local_workflow_state(), WorkflowState::Cancelled);
    }
    #[test]
    fn process_retains_typed_method_and_axis_metadata() {
        let mut series = ProcessSeries::new(None, AxisID::First, None, None);
        series.set_next_process(Some("tilt"), Some(ProcessingMethod::PpGpu));
        assert_eq!(
            series
                .next_process
                .as_ref()
                .unwrap()
                .get_processing_method(),
            Some(ProcessingMethod::PpGpu)
        );
        let process = Process::new(
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
            Some(AxisType::DualAxis),
            None,
        );
        assert_eq!(process.get_axis_type(), Some(AxisType::DualAxis));
    }

    #[test]
    fn com_script_enters_and_runs_in_the_series_owned_workflow_order() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-process-series-{}-{}.com",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::write(&path, "$true\n$true\n").unwrap();
        let mut series = ProcessSeries::new(None, AxisID::First, None, None);
        assert_eq!(series.queue_com_script(&path).unwrap(), 2);
        assert!(series.start_local_workflow().unwrap());
        loop {
            if !matches!(
                series.poll_local_workflow().unwrap(),
                WorkflowState::Running
            ) {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(series.local_workflow_state(), WorkflowState::Complete);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn series_exposes_failure_result_and_does_not_start_later_com_command() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-process-series-failure-{}-{}.com",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::write(&path, "$sh -c 'exit 7'\n$true\n").unwrap();
        let mut series = ProcessSeries::new(None, AxisID::First, None, None);
        assert_eq!(series.queue_com_script(&path).unwrap(), 2);
        assert!(series.start_local_workflow().unwrap());
        loop {
            if !matches!(
                series.poll_local_workflow().unwrap(),
                WorkflowState::Running
            ) {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(series.local_workflow_state(), WorkflowState::Failed);
        assert_eq!(series.local_workflow_results().len(), 1);
        assert!(series.local_workflow_results()[0].name.starts_with("sh:"));
        assert!(!series.local_workflow_results()[0].success);
        assert_eq!(
            series.local_workflow_results()[0].end_state,
            crate::imod::etomo::r#type::process_end_state::ProcessEndState::Failed
        );
        std::fs::remove_file(path).unwrap();
    }
}
