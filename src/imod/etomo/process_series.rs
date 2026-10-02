//! `IMOD/Etomo/src/etomo/ProcessSeries.java` and
//! `IMOD/Etomo/src/etomo/type/ConstProcessSeries.java`.
//!
//! The ProcessSeries instance is passed as a constant to the run process
//! function in the process manager.  It goes into the process object and
//! eventually gets to a processDone function, where
//! `ProcessSeries.startNextProcess()` is called.  If nextProcess or
//! lastProcess is set, `startNextProcess` calls the manager's
//! `startNextProcess` and passes it a reference to the instance.
//!
//! **Shape.**  A series lives on the event dispatch thread with the dialogs it
//! drives, so it is an `Rc<RefCell<ProcessSeries>>` ([`ProcessSeriesHandle`]);
//! processes carry it as `process_interface::ProcessSeriesRef`.  The methods
//! that hand the series itself to the manager (`startNextProcess`,
//! `startFailProcess`, `startPauseProcess`, `start3dmodProcess`) are
//! associated functions on the handle: they take the next process out of the
//! series, release the borrow, and then call the manager, which may well add
//! more processes to the same series.  Java's `synchronized (busyStatusMediator)`
//! blocks are reentrant and run on the one EDT, so releasing the borrow before
//! the call does not change what runs.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::util::utilities;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// The EDT-owned series; see the module comment.
pub type ProcessSeriesHandle = Rc<RefCell<ProcessSeries>>;

/// Java `etomo.type.NextProcessTarget`.
pub trait NextProcessTarget {
    /// Java `startNextProcess`.
    #[allow(clippy::too_many_arguments)]
    fn start_next_process(
        &self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool;
}

/// Java final `ProcessSeries implements ConstProcessSeries`.
pub struct ProcessSeries {
    manager: &'static dyn BaseManager,
    dialog_type: Option<DialogType>,
    process_display: Option<Rc<dyn ProcessDisplay>>,
    ui_component: Option<Rc<dyn UiComponent>>,
    busy_status_mediator: Arc<BusyStatusMediator>,
    busy_axis_id: AxisID,
    descr: Option<String>,
    next_process: Option<Box<Process>>,
    /// The processes between nextProcess and lastProcess.
    process_list: Option<Box<Process>>,
    last_process: Option<Box<Process>>,
    /// 3dmod is opened after last process.
    run_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
    run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    fail_process: Option<Box<Process>>,
    debug: bool,
    force_next_process: bool,
    pause_process: Option<Box<Process>>,
}

impl ProcessSeries {
    fn construct(
        manager: &'static dyn BaseManager,
        busy_axis_id: AxisID,
        ui_component: Option<Rc<dyn UiComponent>>,
        dialog_type: Option<DialogType>,
        process_display: Option<Rc<dyn ProcessDisplay>>,
        descr: Option<&str>,
    ) -> ProcessSeriesHandle {
        utilities::timestamp_process_container_status(
            descr,
            Some("process series"),
            Some("started"),
        );
        let busy_status_mediator = manager.get_busy_status_mediator();
        busy_status_mediator.msg_process_series_constructed(busy_axis_id);
        Rc::new(RefCell::new(ProcessSeries {
            manager,
            dialog_type,
            process_display,
            ui_component,
            busy_status_mediator,
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
        }))
    }

    /// Java `ProcessSeries(BaseManager, AxisID, DialogType, String)`.
    /// `busyAxisID` is the axis where the processes are running.
    pub fn new(
        manager: &'static dyn BaseManager,
        busy_axis_id: AxisID,
        dialog_type: Option<DialogType>,
        descr: Option<&str>,
    ) -> ProcessSeriesHandle {
        ProcessSeries::construct(manager, busy_axis_id, None, dialog_type, None, descr)
    }

    /// Java `ProcessSeries(BaseManager, AxisID, UIComponent, DialogType, String)`.
    pub fn new_with_ui_component(
        manager: &'static dyn BaseManager,
        busy_axis_id: AxisID,
        ui_component: Option<Rc<dyn UiComponent>>,
        dialog_type: Option<DialogType>,
        descr: Option<&str>,
    ) -> ProcessSeriesHandle {
        ProcessSeries::construct(manager, busy_axis_id, ui_component, dialog_type, None, descr)
    }

    /// Java `ProcessSeries(BaseManager, AxisID, DialogType, ProcessDisplay, String)`.
    pub fn new_with_process_display(
        manager: &'static dyn BaseManager,
        busy_axis_id: AxisID,
        dialog_type: Option<DialogType>,
        process_display: Option<Rc<dyn ProcessDisplay>>,
        descr: Option<&str>,
    ) -> ProcessSeriesHandle {
        ProcessSeries::construct(
            manager,
            busy_axis_id,
            None,
            dialog_type,
            process_display,
            descr,
        )
    }

    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!(
            "ProcessSeries:[forceNextProcess:{},nextProcess:{},processList:{},lastProcess:{},failProcess:{},pauseProcess:{}]",
            self.force_next_process,
            describe(self.next_process.as_deref()),
            describe(self.process_list.as_deref()),
            describe(self.last_process.as_deref()),
            describe(self.fail_process.as_deref()),
            describe(self.pause_process.as_deref())
        )
    }

    /// Java `dumpState`.
    pub fn dump_state(&self) {
        eprint!("[debug:{}]", self.debug);
    }

    /// Java `startNextProcess(AxisID)`.
    pub fn start_next_process(this: &ProcessSeriesHandle, axis_id: AxisID) -> bool {
        ProcessSeries::start_next_process_display(this, axis_id, None)
    }

    /// Java `startNextProcess(AxisID, ProcessResultDisplay)`: start next
    /// process from the start process queue.  If it is empty then start next
    /// process from the end process queue.  If next process and last process
    /// are empty, run 3dmod based on deferred3dmodButton and
    /// run3dmodMenuOptions.  The started process is removed.  If there are no
    /// processes to start, the fail process is removed.  Returns true if a
    /// process is run.
    pub fn start_next_process_display(
        this: &ProcessSeriesHandle,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> bool {
        // Get the next process.
        let process = {
            let mut series = this.borrow_mut();
            if let Some(process) = series.next_process.take() {
                process
            } else if let Some(mut process) = series.process_list.take() {
                series.process_list = process.next.take();
                process
            } else if let Some(process) = series.last_process.take() {
                process
            } else if series.run_3dmod_button.is_some() {
                drop(series);
                ProcessSeries::start_3dmod_process(this);
                return true;
            } else {
                series.clear_processes();
                series
                    .busy_status_mediator
                    .msg_process_series_done(series.busy_axis_id);
                utilities::timestamp_process_container_status(
                    series.descr.as_deref(),
                    Some("process series"),
                    Some("finished"),
                );
                return false;
            }
        };
        let (manager, ui_component, dialog_type, process_display) = {
            let mut series = this.borrow_mut();
            utilities::timestamp_full(
                series.descr.as_deref(),
                process.get_descr().as_deref(),
                Some("next process"),
                Some("started"),
            );
            send_msg_secondary_process(process_result_display.as_ref());
            if series.debug {
                println!(
                    "ProcessSeries.startNextProcess:process={}",
                    describe(Some(&process))
                );
            }
            series.force_next_process = process.force_next_process;
            (
                series.manager,
                series.ui_component.clone(),
                series.dialog_type,
                series.process_display.clone(),
            )
        };
        match &process.target {
            None => manager.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
            Some(target) => target.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
        };
        true
    }

    /// Java `startFailProcess(AxisID)`.
    pub fn start_fail_process(this: &ProcessSeriesHandle, axis_id: AxisID) {
        ProcessSeries::start_fail_process_display(this, axis_id, None);
    }

    /// Java `startFailProcess(AxisID, ProcessResultDisplay)`: all other
    /// processes are deleted and the failprocess is started if it exists.
    pub fn start_fail_process_display(
        this: &ProcessSeriesHandle,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        if this.borrow().force_next_process {
            // Not allowed to fail.
            ProcessSeries::start_next_process_display(this, axis_id, process_result_display);
            return;
        }
        let (process, manager, ui_component, dialog_type, process_display) = {
            let mut series = this.borrow_mut();
            let process = series.fail_process.take();
            series.clear_processes();
            match process {
                None => {
                    series
                        .busy_status_mediator
                        .msg_process_series_done(series.busy_axis_id);
                    utilities::timestamp_process_container_status(
                        series.descr.as_deref(),
                        Some("process series"),
                        Some("finished"),
                    );
                    return;
                }
                Some(process) => {
                    utilities::timestamp_full(
                        series.descr.as_deref(),
                        process.get_descr().as_deref(),
                        Some("fail process"),
                        Some("started"),
                    );
                    (
                        process,
                        series.manager,
                        series.ui_component.clone(),
                        series.dialog_type,
                        series.process_display.clone(),
                    )
                }
            }
        };
        match &process.target {
            None => manager.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
            Some(target) => target.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
        };
    }

    /// Java `killSeries`.
    pub fn kill_series(
        &mut self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        let _ = (axis_id, process_result_display);
        self.clear_processes();
        self.busy_status_mediator
            .msg_process_series_done(self.busy_axis_id);
        utilities::timestamp_process_container_status(
            self.descr.as_deref(),
            Some("process series"),
            Some("killed"),
        );
    }

    /// Java `endSeries`: a bandage function.  It tells BusyStatusMediator that
    /// the series is done without changing ProcessSeries.
    pub fn end_series(&self) {
        self.busy_status_mediator
            .msg_process_series_done(self.busy_axis_id);
        utilities::timestamp_process_container_status(
            self.descr.as_deref(),
            Some("process series"),
            Some("finished"),
        );
    }

    /// Java `startPauseProcess`: all other processes are deleted and the
    /// pauseprocess is started if it exists.  Returns true if a process was
    /// started.
    ///
    /// The Java tests `if (process.target != null)` before calling the
    /// manager and calls `process.target.startNextProcess` in the else arm
    /// (`ProcessSeries.java:382-389`), the reverse of every other dispatch in
    /// the class, so a pause process without a target throws
    /// `NullPointerException`.  Fixed in translation (`BUGS.md`): the
    /// condition is `target == null`, as in `startNextProcess` and
    /// `startFailProcess`.
    pub fn start_pause_process(
        this: &ProcessSeriesHandle,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> bool {
        let (process, manager, ui_component, dialog_type, process_display) = {
            let mut series = this.borrow_mut();
            let process = series.pause_process.take();
            series.clear_processes();
            match process {
                None => {
                    series
                        .busy_status_mediator
                        .msg_process_series_done(series.busy_axis_id);
                    utilities::timestamp_process_container_status(
                        series.descr.as_deref(),
                        Some("process series"),
                        Some("finished"),
                    );
                    return true;
                }
                Some(process) => {
                    utilities::timestamp_full(
                        series.descr.as_deref(),
                        process.get_descr().as_deref(),
                        Some("pause process"),
                        Some("started"),
                    );
                    (
                        process,
                        series.manager,
                        series.ui_component.clone(),
                        series.dialog_type,
                        series.process_display.clone(),
                    )
                }
            }
        };
        match &process.target {
            None => manager.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
            Some(target) => target.start_next_process(
                ui_component,
                axis_id,
                &process,
                process_result_display,
                this,
                dialog_type,
                process_display,
            ),
        };
        true
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
        self.last_process.as_ref()?.get_process()
    }

    /// Java `setNextProcess(String, ProcessingMethod)`.
    pub fn set_next_process(
        &mut self,
        process: Option<&str>,
        processing_method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            None,
            None,
            None,
            processing_method,
            None,
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `prependNextProcess`: works like setNextProcess but does not
    /// overwrite an existing nextProcess; the old one moves to the beginning
    /// of processList.
    pub fn prepend_next_process(&mut self, task: Rc<dyn TaskInterface>) {
        let old = self.next_process.take();
        self.prepend_process(old);
        self.next_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setNextProcess(TaskInterface)`.
    pub fn set_next_process_task(&mut self, task: Rc<dyn TaskInterface>) {
        self.next_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setNextProcess(TaskInterface, String)`.
    pub fn set_next_process_task_parameter(
        &mut self,
        task: Rc<dyn TaskInterface>,
        process_parameter: Option<&str>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            process_parameter.map(str::to_owned),
        )));
    }

    /// Java `setNextProcess(NextProcessTarget, TaskInterface)`.
    pub fn set_next_process_target_task(
        &mut self,
        target: Rc<dyn NextProcessTarget>,
        task: Rc<dyn TaskInterface>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            Some(target),
            None,
            None,
        )));
    }

    /// Java `setNextProcess(TaskInterface, Command)`.
    pub fn set_next_process_task_command(
        &mut self,
        task: Rc<dyn TaskInterface>,
        command: Arc<dyn Command + Send + Sync>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            Some(command),
            None,
            None,
            None,
        )));
    }

    /// Java `setPauseProcess`.
    pub fn set_pause_process(&mut self, task: Rc<dyn TaskInterface>) {
        self.pause_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `addProcess(TaskInterface)`.
    pub fn add_process(&mut self, task: Rc<dyn TaskInterface>) {
        self.add_process_force(task, false);
    }

    /// Java `addProcess(TaskInterface, boolean)`: adds a process to the end of
    /// processList.
    pub fn add_process_force(&mut self, task: Rc<dyn TaskInterface>, force_next_process: bool) {
        self.append_process(Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            force_next_process,
            None,
            None,
            None,
            None,
        ))));
    }

    /// Java `addProcess(TaskInterface, Command, AxisType)`.
    pub fn add_process_command_axis_type(
        &mut self,
        task: Rc<dyn TaskInterface>,
        command: Option<Arc<dyn Command + Send + Sync>>,
        axis_type: Option<AxisType>,
    ) {
        self.append_process(Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            command,
            None,
            axis_type,
            None,
        ))));
    }

    /// Java private `prependProcess`.
    fn prepend_process(&mut self, process: Option<Box<Process>>) {
        let Some(mut process) = process else {
            return;
        };
        // Place process at the beginning of ProcessList
        process.next = self.process_list.take();
        self.process_list = Some(process);
    }

    /// Java private `appendProcess`.
    fn append_process(&mut self, process: Option<Box<Process>>) {
        let Some(process) = process else {
            return;
        };
        // Go to the end of processList.
        let mut slot = &mut self.process_list;
        while let Some(pointer) = slot {
            slot = &mut pointer.next;
        }
        *slot = Some(process);
    }

    /// Java `setNextProcess(String, ProcessName, ProcessingMethod)`.
    pub fn set_next_process_subprocess(
        &mut self,
        process: Option<&str>,
        subprocess_name: Option<ProcessName>,
        processing_method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            subprocess_name,
            None,
            None,
            processing_method,
            None,
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setNextProcess(TaskInterface, ProcessName, ProcessingMethod)`.
    pub fn set_next_process_task_subprocess(
        &mut self,
        task: Rc<dyn TaskInterface>,
        subprocess_name: Option<ProcessName>,
        processing_method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            None,
            subprocess_name,
            None,
            None,
            processing_method,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setNextProcess(String, ProcessName, FileType, ProcessingMethod)`.
    pub fn set_next_process_output_file_type(
        &mut self,
        process: Option<&str>,
        subprocess_name: Option<ProcessName>,
        output_image_file_type: Option<&'static FileType>,
        processing_method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            subprocess_name,
            output_image_file_type.map(|file_type| (**file_type).clone()),
            None,
            processing_method,
            None,
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setNextProcess(String, ProcessName, FileKey, FileKey,
    /// ProcessingMethod)`.
    pub fn set_next_process_output_file_keys(
        &mut self,
        process: Option<&str>,
        subprocess_name: Option<ProcessName>,
        output_image_file_key: Option<FileKey>,
        output_image_file_key2: Option<FileKey>,
        processing_method: Option<ProcessingMethod>,
    ) {
        self.next_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            subprocess_name,
            output_image_file_key,
            output_image_file_key2,
            processing_method,
            None,
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `clearProcesses`.
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
        // `peek().setParameter(input)`: null when the series is empty.
        process
            .expect("setNextProcessParameter on an empty series")
            .set_parameter(input);
    }

    /// Java `willProcessBeDropped(ProcessData)`: true if a next process or
    /// process list exists and is not OK to drop, or lastProcess exists and
    /// hasn't been saved in processData.
    pub fn will_process_be_dropped(&self, process_data: Option<&ProcessData>) -> bool {
        self.will_process_be_dropped_list(process_data, false)
    }

    /// Java private `willProcessBeDropped(ProcessData, boolean)`.
    fn will_process_be_dropped_list(
        &self,
        process_data: Option<&ProcessData>,
        process_list_only: bool,
    ) -> bool {
        if !process_list_only && will_process_be_dropped_process(self.next_process.as_deref()) {
            return true;
        }
        let mut process = self.process_list.as_deref();
        while let Some(current) = process {
            if will_process_be_dropped_process(Some(current)) {
                return true;
            }
            process = current.next.as_deref();
        }
        if process_list_only {
            return false;
        }
        if will_process_be_dropped_process(self.last_process.as_deref()) {
            return true;
        }
        let Some(process_data) = process_data else {
            return false;
        };
        let Some(last_process) = &self.last_process else {
            return false;
        };
        // Return true of the last process information in processData doesn't
        // match the information in this instance.
        self.dialog_type != process_data.get_dialog_type()
            || !last_process.equals_string(process_data.get_last_process().as_deref())
    }

    /// Java `willProcessListBeDropped`.
    pub fn will_process_list_be_dropped(&self) -> bool {
        self.will_process_be_dropped_list(None, true)
    }

    /// Java `peekNextProcess`.
    pub fn peek_next_process(&self) -> Option<String> {
        if let Some(process) = self.peek() {
            return process.get_process();
        }
        if self.run_3dmod_button.is_some() {
            return Some("3dmod".to_owned());
        }
        None
    }

    /// Java private `peek`.
    fn peek(&self) -> Option<&Process> {
        self.next_process
            .as_deref()
            .or(self.process_list.as_deref())
            .or(self.last_process.as_deref())
    }

    /// Java `setLastProcess(String)`.
    pub fn set_last_process(&mut self, process: Option<&str>) {
        self.last_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setLastProcess(NextProcessTarget, String)`.
    pub fn set_last_process_target(
        &mut self,
        target: Rc<dyn NextProcessTarget>,
        process: Option<&str>,
    ) {
        self.last_process = Some(Box::new(Process::new(
            process.map(str::to_owned),
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            Some(target),
            None,
            None,
        )));
    }

    /// Java `setLastProcess(TaskInterface)`.
    pub fn set_last_process_task(&mut self, task: Rc<dyn TaskInterface>) {
        self.last_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setLastProcess(NextProcessTarget, TaskInterface)`.
    pub fn set_last_process_target_task(
        &mut self,
        target: Rc<dyn NextProcessTarget>,
        task: Rc<dyn TaskInterface>,
    ) {
        self.last_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            Some(target),
            None,
            None,
        )));
    }

    /// Java `setFailProcess`: a process to start when one of the process fails
    /// and forceNextProcess is off.
    pub fn set_fail_process(&mut self, task: Rc<dyn TaskInterface>) {
        self.fail_process = Some(Box::new(Process::new(
            None,
            None,
            None,
            None,
            None,
            Some(task),
            false,
            None,
            None,
            None,
            None,
        )));
    }

    /// Java `setRun3dmodDeferred`: sets the option to open a 3dmod after all
    /// the processes are done.  This function cannot be used to reset a 3dmod
    /// process.
    pub fn set_run_3dmod_deferred(
        &mut self,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let Some(deferred_3dmod_button) = deferred_3dmod_button else {
            return;
        };
        self.run_3dmod_button = Some(deferred_3dmod_button);
        self.run_3dmod_menu_options = run_3dmod_menu_options;
    }

    /// Java private `start3dmodProcess`: calls the action function of
    /// run3dmodButton and blanks it out so it can't be run more then once.
    fn start_3dmod_process(this: &ProcessSeriesHandle) {
        let (button, options) = {
            let mut series = this.borrow_mut();
            (
                series.run_3dmod_button.take(),
                series.run_3dmod_menu_options.take(),
            )
        };
        if let Some(button) = button {
            button.action(options.unwrap_or_default());
        }
        let series = this.borrow();
        series
            .busy_status_mediator
            .msg_process_series_done(series.busy_axis_id);
        utilities::timestamp_process_container_status(
            series.descr.as_deref(),
            Some("process series"),
            Some("finished"),
        );
    }
}

/// Java private `sendMsgSecondaryProcess`.
fn send_msg_secondary_process(process_result_display: Option<&ProcessResultDisplayRef>) {
    let Some(process_result_display) = process_result_display else {
        return;
    };
    process_result_display.get().msg_secondary_process();
}

/// Java private `willProcessBeDropped(Process)`: true if the process exists
/// and is not OK to drop.
fn will_process_be_dropped_process(process: Option<&Process>) -> bool {
    let Some(process) = process else {
        return false;
    };
    match &process.task {
        None => true,
        Some(task) => !task.ok_to_drop(),
    }
}

fn describe(process: Option<&Process>) -> String {
    match process {
        None => "null".to_owned(),
        Some(process) => process.to_source_string(),
    }
}

/// Java public static final nested class `ProcessSeries.Process`.
pub struct Process {
    process: Option<String>,
    subprocess_name: Option<ProcessName>,
    output_image_file_key: Option<FileKey>,
    output_image_file_key2: Option<FileKey>,
    processing_method: Option<ProcessingMethod>,
    pub task: Option<Rc<dyn TaskInterface>>,
    force_next_process: bool,
    command: Option<Arc<dyn Command + Send + Sync>>,
    target: Option<Rc<dyn NextProcessTarget>>,
    axis_type: Option<AxisType>,
    next: Option<Box<Process>>,
    /// Information passed from a previous process.
    parameter: Option<String>,
}

impl Process {
    #[allow(clippy::too_many_arguments)]
    fn new(
        process: Option<String>,
        subprocess_name: Option<ProcessName>,
        output_image_file_key: Option<FileKey>,
        output_image_file_key2: Option<FileKey>,
        processing_method: Option<ProcessingMethod>,
        task: Option<Rc<dyn TaskInterface>>,
        force_next_process: bool,
        command: Option<Arc<dyn Command + Send + Sync>>,
        target: Option<Rc<dyn NextProcessTarget>>,
        axis_type: Option<AxisType>,
        parameter: Option<String>,
    ) -> Process {
        Process {
            process,
            subprocess_name,
            output_image_file_key,
            output_image_file_key2,
            processing_method,
            task,
            force_next_process,
            command,
            target,
            axis_type,
            next: None,
            parameter,
        }
    }

    /// Java `getDescr`.
    pub fn get_descr(&self) -> Option<String> {
        let mut name = self.task.as_ref().and_then(|task| task.get_descr());
        let mut subcommand_name = None;
        if name.is_none()
            && let Some(process) = &self.process
            && !process.contains('@')
        {
            name = Some(process.clone());
        }
        if let Some(subprocess_name) = &self.subprocess_name {
            subcommand_name = Some(subprocess_name.to_string());
        }
        if let Some(command) = &self.command {
            if name.is_none() {
                name = command.get_command_name();
            }
            if subcommand_name.is_none() {
                subcommand_name = command.get_subcommand_process_name();
            }
        }
        if name.is_some() || subcommand_name.is_some() {
            let mut builder = String::new();
            if let Some(name) = name {
                builder.push_str(&format!("{name} "));
            }
            if let Some(subcommand_name) = subcommand_name {
                builder.push_str(&subcommand_name);
            }
            return Some(builder);
        }
        None
    }

    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!(
            "Process:[forceNextProcess:{},process:{},task:{},target:{}]",
            self.force_next_process,
            self.process.as_deref().unwrap_or("null"),
            self.task
                .as_ref()
                .and_then(|task| task.get_descr())
                .unwrap_or_else(|| "null".to_owned()),
            if self.target.is_some() { "set" } else { "null" }
        )
    }

    /// Java `setParameter`.
    pub fn set_parameter(&mut self, parameter: Option<&str>) {
        self.parameter = parameter.map(str::to_owned);
    }

    /// Java private `getProcess`: the process string, else the task's
    /// `toString()` (its description).
    fn get_process(&self) -> Option<String> {
        if let Some(process) = &self.process {
            return Some(process.clone());
        }
        self.task.as_ref().and_then(|task| task.get_descr())
    }

    /// Java `getAxisType`.
    pub fn get_axis_type(&self) -> Option<AxisType> {
        self.axis_type
    }

    /// Java `getParameter`.
    pub fn get_parameter(&self) -> Option<&str> {
        self.parameter.as_deref()
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<&Arc<dyn Command + Send + Sync>> {
        self.command.as_ref()
    }

    /// Java `getSubprocessName`.
    pub fn get_subprocess_name(&self) -> Option<ProcessName> {
        self.subprocess_name.clone()
    }

    /// Java `getOutputImageFileKey`.
    pub fn get_output_image_file_key(&self) -> Option<&FileKey> {
        self.output_image_file_key.as_ref()
    }

    /// Java `getOutputImageFileKey2` (the field has no Java getter; kept for
    /// the managers that read it through `process.outputImageFileKey2`).
    pub fn get_output_image_file_key2(&self) -> Option<&FileKey> {
        self.output_image_file_key2.as_ref()
    }

    /// Java `equals(TaskInterface)`: `this.task == task`.  Tasks are enum
    /// singletons in the Java; the translation's tasks are unit values, so
    /// identity is the same concrete type and description.
    pub fn equals_task(&self, task: &dyn TaskInterface) -> bool {
        match &self.task {
            None => false,
            Some(own) => {
                (own.as_ref() as &dyn std::any::Any).type_id()
                    == (task as &dyn std::any::Any).type_id()
                    && own.get_descr() == task.get_descr()
            }
        }
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, string: Option<&str>) -> bool {
        if let (Some(process), Some(string)) = (&self.process, string)
            && process == string
        {
            return true;
        }
        // `task.equals(string)`: an enum constant never equals a String.
        false
    }

    /// Java `getTask`.
    pub fn get_task(&self) -> Option<&Rc<dyn TaskInterface>> {
        self.task.as_ref()
    }

    /// Java `equals(ProcessName)`.
    pub fn equals_process_name(&self, process_name: &ProcessName) -> bool {
        self.process
            .as_deref()
            .is_some_and(|process| process == process_name.to_string())
    }

    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.processing_method
    }
}
