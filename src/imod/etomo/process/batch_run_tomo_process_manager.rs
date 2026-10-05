//! `IMOD/Etomo/src/etomo/process/BatchRunTomoProcessManager.java`.
//!
//! The process manager of the batchruntomo interface (`BatchRunTomoManager`).  It
//! embeds the `BaseProcessManager` superclass as `base` and installs itself as the
//! base's [`BaseProcessManagerHooks`] for its overrides.
//!
//! **Threads.**  The post/error hooks run on process threads.  They report to the
//! batchruntomo dialog, an event dispatch thread object, so the manager calls they make
//! are posted to that thread.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::background_process::BackgroundProcess;
use super::base_process_manager::{AxisBusyException, BaseProcessManager, BaseProcessManagerHooks};
use super::batch_run_tomo_process_monitor::{self, BatchRunTomoProcessMonitor};
use super::batch_run_tomo_process_submonitor;
use super::com_script_process::ComScriptProcess;
use super::log_feed_monitor::{LogFeedMonitor, LogFeedMonitorImpl, MessagesRef};
use super::monitor::{DetachedProcessMonitor, Monitor, ProcessMonitor};
use super::process_data::ProcessData;
use super::process_interface::{ProcessInterface, ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface};
use super::process_messages::MessagesArray;
use super::process_output_strings;
use super::processchunks_batch_run_tomo_monitor;
use super::reconnect_process::ReconnectProcess;
use super::series_watcher_process_monitor;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::{self, BatchRunTomoManager};
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::series_watcher_param::SeriesWatcherParam;
use crate::imod::etomo::comscript::split_batch_param::SplitBatchParam;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_status::RunStatus;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::table_reference::TableReference;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = batch_run_tomo_manager::AXIS_ID;

/// Java `public final class BatchRunTomoProcessManager extends BaseProcessManager`.
pub struct BatchRunTomoProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
}

impl BatchRunTomoProcessManager {
    /// Java public `BatchRunTomoProcessManager(BatchRunTomoManager)`.  The manager keeps
    /// it for the run.
    pub fn new(manager: &'static BatchRunTomoManager) -> &'static BatchRunTomoProcessManager {
        let process_manager: &'static BatchRunTomoProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(manager),
            manager,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// The superclass, at its final address.
    fn base(&'static self) -> &'static BaseProcessManager {
        &self.base
    }

    /// Java public `splitBatch(SplitBatchParam, ProcessSeries) throws
    /// AxisBusyException`.
    pub fn split_batch(
        &'static self,
        param: &mut SplitBatchParam,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base().start_background_process_array_display(
            param.get_command(),
            AXIS_ID,
            None,
            Some(ProcessName::SPLIT_BATCH),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java public `startSeriesWatcherBatchSubmonitor(String, File, ProcessMessages,
    /// ProcessingMethod)`.
    pub fn start_series_watcher_batch_submonitor(
        &'static self,
        stack_id: Option<&str>,
        batch_log: PathBuf,
        messages: Option<MessagesRef>,
        processing_method: Option<ProcessingMethod>,
    ) -> Option<Arc<BatchRunTomoProcessMonitor>> {
        let submonitor = batch_run_tomo_process_submonitor::new(
            self.manager,
            stack_id,
            AXIS_ID,
            None,
            messages,
            processing_method,
            batch_log,
        );
        match submonitor.erased().create_process_output() {
            Ok(()) => {
                self.base
                    .start_monitor(submonitor.clone() as Arc<dyn Monitor>, AXIS_ID, true);
                Some(submonitor)
            }
            Err(e) => {
                eprintln!("ERROR: Unable to start batchruntomo submonitor for serieswatcher.");
                eprintln!("{}", e);
                None
            }
        }
    }

    /// Java public `batchruntomo(BatchruntomoParam, ProcessData, ProcessMessages,
    /// ProcessingMethod) throws AxisBusyException`.
    pub fn batchruntomo(
        &'static self,
        param: Option<Arc<BatchruntomoParam>>,
        process_data: Arc<Mutex<ProcessData>>,
        messages: Option<MessagesRef>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<Option<String>, AxisBusyException> {
        let Some(param) = param else {
            return Ok(None);
        };
        let manager: &'static dyn BaseManager = self.manager;
        let command = file_type::CLASS
            .batch_run_tomo_comscript
            .get_file_name(Some(manager), Some(AXIS_ID))
            .unwrap_or_default();
        let monitor = batch_run_tomo_process_monitor::get_instance(
            self.manager,
            AXIS_ID,
            Some(process_data.clone()),
            messages,
            processing_method,
        );
        // Start the com script in the background
        let com_script_process = self.base().start_outfile_com_script(
            &command,
            monitor,
            AXIS_ID,
            Some(param as Arc<dyn Command + Send + Sync>),
            Some(&file_type::CLASS.batch_run_tomo_comscript),
            Some(ProcessName::BATCHRUNTOMO),
            true,
            process_data,
            false,
        )?;
        Ok(Some(com_script_process.get_name()))
    }

    /// Java public `serieswatcher(SeriesWatcherParam, ProcessData, ProcessMessages,
    /// ProcessingMethod, TableReference) throws AxisBusyException`.
    pub fn serieswatcher(
        &'static self,
        param: Option<Arc<SeriesWatcherParam>>,
        process_data: Arc<Mutex<ProcessData>>,
        messages: Option<MessagesRef>,
        processing_method: Option<ProcessingMethod>,
        table_reference: Arc<TableReference>,
    ) -> Result<Option<String>, AxisBusyException> {
        let Some(param) = param else {
            return Ok(None);
        };
        let manager: &'static dyn BaseManager = self.manager;
        let command = file_type::CLASS
            .series_watcher_comscript
            .get_file_name(Some(manager), Some(AXIS_ID))
            .unwrap_or_default();
        let monitor = series_watcher_process_monitor::get_instance(
            self.manager,
            param.is_dual_axis(),
            Some(process_data.clone()),
            messages,
            processing_method,
            table_reference,
        );
        // Start the com script in the background
        let com_script_process = self.base().start_outfile_com_script(
            &command,
            monitor,
            AXIS_ID,
            Some(param as Arc<dyn Command + Send + Sync>),
            Some(&file_type::CLASS.series_watcher_comscript),
            Some(ProcessName::SERIES_WATCHER),
            true,
            process_data,
            true,
        )?;
        let retval = com_script_process.get_name();
        Ok(Some(retval))
    }

    /// The reconnect shared by `reconnectSerieswatcher` and `reconnectBatchruntomo`:
    /// a `LoggedReconnectProcess` on the monitor's log, run on its own thread.
    fn reconnect_monitor<S: LogFeedMonitorImpl>(
        &'static self,
        monitor: Arc<LogFeedMonitor<S>>,
        log_file_name: &str,
        process_name: &str,
    ) -> bool {
        let process_monitor: Arc<dyn ProcessMonitor> = monitor.clone();
        let process = match ReconnectProcess::new_logged(
            self.manager,
            self.base(),
            Some(process_monitor),
            Some(self.base.axis_process_data.get_saved_process_data(AXIS_ID)),
            AXIS_ID,
            log_file_name,
            Some(process_output_strings::SUCCESS_TAG),
            None,
            None,
            false,
            true,
            true,
        ) {
            Ok(process) => process,
            Err(e) => {
                let message = match &e {
                    LogFileError::Lock(_) => e.to_string(),
                    _ => e.get_message(),
                };
                eprintln!("{}", message);
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("Unable to reconnect to {}.\n{}", process_name, message),
                    "Reconnect Failure",
                    Some(AXIS_ID),
                );
                return false;
            }
        };
        DetachedProcessMonitor::set_process(
            &*monitor,
            Arc::clone(&process) as Arc<dyn SystemProcessInterface>,
        );
        let thread_process = Arc::clone(&process);
        std::thread::spawn(move || thread_process.run());
        self.base
            .axis_process_data
            .map_axis_thread(Some(process.as_process()), AXIS_ID);
        self.base.axis_process_data.map_axis_process_monitor(
            None,
            Some(monitor as Arc<dyn Monitor>),
            AXIS_ID,
        );
        true
    }

    /// Java public `reconnectSerieswatcher(ProcessData, ProcessMessages,
    /// TableReference)`.
    pub fn reconnect_serieswatcher(
        &'static self,
        process_data: Arc<Mutex<ProcessData>>,
        messages: Option<MessagesRef>,
        table_reference: Arc<TableReference>,
    ) -> bool {
        let processing_method = process_data.lock().unwrap().get_processing_method();
        let monitor = series_watcher_process_monitor::get_reconnect_instance(
            self.manager,
            AXIS_ID,
            Some(process_data),
            messages,
            processing_method,
            table_reference,
        );
        let log_file_name = monitor.erased().get_log_file_name();
        self.reconnect_monitor(monitor, &log_file_name, "serieswatcher")
    }

    /// Java public `resumeBatchruntomo(BatchruntomoParam, ProcessData, ProcessMessages,
    /// ProcessingMethod) throws AxisBusyException`.
    pub fn resume_batchruntomo(
        &'static self,
        param: Option<Arc<BatchruntomoParam>>,
        process_data: Arc<Mutex<ProcessData>>,
        messages: Option<MessagesRef>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<Option<String>, AxisBusyException> {
        let Some(param) = param else {
            return Ok(None);
        };
        let manager: &'static dyn BaseManager = self.manager;
        let command = file_type::CLASS
            .batch_run_tomo_comscript
            .get_file_name(Some(manager), Some(AXIS_ID))
            .unwrap_or_default();
        let monitor = batch_run_tomo_process_monitor::get_resume_instance(
            self.manager,
            AXIS_ID,
            Some(process_data.clone()),
            messages,
            processing_method,
        );
        // Start the com script in the background
        let com_script_process = self.base().start_outfile_com_script(
            &command,
            monitor,
            AXIS_ID,
            Some(param as Arc<dyn Command + Send + Sync>),
            Some(&file_type::CLASS.batch_run_tomo_comscript),
            Some(ProcessName::BATCHRUNTOMO),
            true,
            process_data,
            false,
        )?;
        Ok(Some(com_script_process.get_name()))
    }

    /// Java public `reconnectBatchruntomo(ProcessData, ProcessMessages)`.
    pub fn reconnect_batchruntomo(
        &'static self,
        process_data: Arc<Mutex<ProcessData>>,
        messages: Option<MessagesRef>,
    ) -> bool {
        let processing_method = process_data.lock().unwrap().get_processing_method();
        let monitor = batch_run_tomo_process_monitor::get_reconnect_instance(
            self.manager,
            AXIS_ID,
            Some(process_data),
            messages,
            processing_method,
        );
        let log_file_name = monitor.erased().get_log_file_name();
        self.reconnect_monitor(monitor, &log_file_name, "batchruntomo")
    }

    /// Java public `haltMonitorThread()`.
    pub fn halt_monitor_thread(&self) {
        self.base.axis_process_data.halt_monitor_thread(AXIS_ID);
    }

    /// `manager.statusChanged(status)`, on the event dispatch thread.
    fn status_changed(&self, status: BatchRunTomoStatus) {
        let manager = self.manager;
        event_queue::invoke_later(move || manager.status_changed(Some(status)));
    }

    /// `manager.resetCurrentProcesschunks(axisID)`, on the event dispatch thread.
    fn reset_current_processchunks(&self, axis_id: AxisID) {
        let manager = self.manager;
        event_queue::invoke_later(move || manager.reset_current_processchunks(Some(axis_id)));
    }
}

impl BaseProcessManagerHooks for BatchRunTomoProcessManager {
    /// Java override `reconnectProcesschunks(AxisID, ProcessData, ProcessResultDisplay,
    /// ProcessSeries, boolean, boolean, List<ProcessMessages>) throws LockException`.
    fn reconnect_processchunks(
        &self,
        base: &'static BaseProcessManager,
        axis_id: AxisID,
        process_data: Arc<Mutex<ProcessData>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        multi_line_messages: bool,
        popup_chunk_warnings: bool,
        messages_array: Option<MessagesArray>,
    ) -> Result<bool, LogFileError> {
        let monitor = processchunks_batch_run_tomo_monitor::get_reconnect_instance(
            self.manager,
            axis_id,
            process_data.clone(),
            messages_array.unwrap_or_default(),
            multi_line_messages,
        );
        base.reconnect_processchunks_monitor(
            axis_id,
            process_data,
            process_result_display,
            process_series,
            monitor,
            popup_chunk_warnings,
            true,
        )
    }

    /// Java override `processchunks(AxisID, ProcesschunksParam, ParallelProgressDisplay,
    /// ProcessResultDisplay, ProcessSeries, boolean, ProcessingMethod, boolean, RunType,
    /// ProcessData, List<ProcessMessages>) throws AxisBusyException`.
    fn processchunks(
        &self,
        base: &'static BaseProcessManager,
        axis_id: AxisID,
        param: Arc<ProcesschunksParam>,
        parallel_progress_display: &dyn ParallelProgressDisplay,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
        run_type: Option<RunType>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
        messages_array: Option<MessagesArray>,
    ) -> Result<String, AxisBusyException> {
        let run_list = self.manager.create_run_list(run_type);
        if let Some(run_list) = &run_list {
            param.correct_multi_proc(run_list.size_run_status(Some(RunStatus::ToRun)));
        }
        let monitor = processchunks_batch_run_tomo_monitor::get_instance(
            self.manager,
            axis_id,
            param.get_root_name().as_deref(),
            Some(param.get_computer_map().into_iter().collect()),
            param.get_secondary_queue().as_deref(),
            run_type,
            run_list,
            managed_process_data.clone(),
            messages_array.unwrap_or_default(),
            multi_line_messages,
            processing_method,
        );
        base.processchunks_monitor(
            monitor,
            axis_id,
            param,
            parallel_progress_display,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            managed_process_data,
        )
    }

    /// Java override `postProcess(BackgroundProcess)`.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        // try
        let Some(process_name) = process.get_process_name() else {
            return;
        };
        let end_state = process.get_process_end_state();
        if process_name == ProcessName::SPLIT_BATCH {
            let manager = self.manager;
            let process_series = process.get_process_series();
            let std_output = process.get_std_output();
            event_queue::invoke_later(move || {
                manager.set_premade_machine_list(process_series, std_output.as_deref())
            });
        } else if process_name == ProcessName::PROCESSCHUNKS {
            if process.is_pausing() || end_state == Some(ProcessEndState::Killed) {
                self.status_changed(BatchRunTomoStatus::KilledOrPausedProcessChunks);
            } else {
                self.reset_current_processchunks(process.get_axis_id());
            }
        } else if process.is_pausing() || end_state == Some(ProcessEndState::Killed) {
            self.status_changed(BatchRunTomoStatus::KilledOrPaused);
        } else {
            self.reset_current_processchunks(process.get_axis_id());
        }
    }

    /// Java override `postProcess(DetachedProcess)`.
    fn post_process_detached(&self, _base: &BaseProcessManager, process: &BackgroundProcess) {
        let end_state = process.get_process_end_state();
        // `process.getCommand().getCommandName()`: Java catches the
        // NullPointerException of a missing command and does nothing.
        let Some(command_name) = process
            .get_command()
            .and_then(|command| command.get_command_name())
        else {
            return;
        };
        if command_name == ProcessName::PROCESSCHUNKS.to_string() {
            if process.is_pausing() || end_state == Some(ProcessEndState::Killed) {
                self.status_changed(BatchRunTomoStatus::KilledOrPausedProcessChunks);
            } else {
                self.reset_current_processchunks(process.get_axis_id());
            }
        } else if process.is_pausing() || end_state == Some(ProcessEndState::Killed) {
            self.status_changed(BatchRunTomoStatus::KilledOrPaused);
        } else {
            self.reset_current_processchunks(process.get_axis_id());
        }
    }

    /// Java override `errorProcess(DetachedProcess)`.
    fn error_process_detached(&self, _base: &BaseProcessManager, process: &BackgroundProcess) {
        let process_name = process.get_process_name();
        if process_name == Some(ProcessName::PROCESSCHUNKS) {
            if process.is_pausing() {
                self.status_changed(BatchRunTomoStatus::KilledOrPausedProcessChunks);
            } else {
                self.status_changed(BatchRunTomoStatus::Failed);
            }
        } else {
            self.status_changed(BatchRunTomoStatus::Failed);
        }
    }

    /// Java override `errorProcess(ComScriptProcess)`.
    fn error_process_com_script(&self, _base: &BaseProcessManager, process: &ComScriptProcess) {
        eprintln!(
            "BatchRunTomoProcessManager errorProcess ComScriptProcess statusChanged BatchRunTomoStatus\nprocess:{}",
            process.to_source_string()
        );
        self.status_changed(BatchRunTomoStatus::Failed);
    }

    /// Java override `errorProcess(ReconnectProcess)`.
    fn error_process_reconnect(&self, _base: &BaseProcessManager, script: &ReconnectProcess) {
        eprintln!(
            "BatchRunTomoProcessManager errorProcess ReconnectProcess statusChanged BatchRunTomoStatus\nscript:{}",
            script.to_source_string()
        );
        self.status_changed(BatchRunTomoStatus::Failed);
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    fn error_process_background(&self, _base: &BaseProcessManager, process: &BackgroundProcess) {
        eprintln!(
            "BatchRunTomoProcessManager errorProcess BackgroundProcess statusChanged BatchRunTomoStatus\nprocess:{}",
            process.to_source_string()
        );
        self.status_changed(BatchRunTomoStatus::Failed);
    }
}

