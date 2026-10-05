//! `IMOD/Etomo/src/etomo/process/BaseProcessManager.java`.
//!
//! Process manager for processes not associated with one interface such as
//! processchunks.  It also contains axis busy functions, start process
//! functions, and end process functions.  It also handles killing most kinds
//! of processes.  Relationships: one BaseManager to many BaseProcessManager.
//!
//! **Shape.**  The Java class is abstract; its subclasses (`ProcessManager`,
//! `JoinProcessManager`, ...) override the `errorProcess`/`postProcess`
//! hooks.  Here the class is a struct that each subclass embeds and leaks for
//! the program's lifetime (processes hold `&'static BaseProcessManager`), and
//! the overridden hooks are the [`BaseProcessManagerHooks`] a subclass
//! installs with [`BaseProcessManager::set_hooks`].
//!
//! **Threads.**  The start functions run on the event dispatch thread; the
//! `msg*Done` functions run on the finished process's thread, as in the
//! Java.  Where the Java calls the manager's `processDone` or a `UIHarness`
//! dialog from there, the call is posted to the EDT (`util/event_queue.rs`).
//!
//! **Overloads.**  `processchunks` and `reconnectProcesschunks` each have a
//! public overload that builds a `ProcesschunksProcessMonitor` and a
//! package-private one that takes the monitor; the latter are generic over
//! the monitor's subclass (`processchunks_monitor`,
//! `reconnect_processchunks_monitor`).  `ProcessManager` overrides the public
//! `processchunks`, so [`BaseProcessManager::processchunks`] dispatches through
//! the hooks and [`BaseProcessManager::processchunks_base`] is the base body.

use super::axis_process_data::{AxisProcessData, ClearKind};
use super::background_com_script_process::BackgroundComScriptProcess;
use super::background_process::{BackgroundProcess, BackgroundProcessInit};
use super::com_script_process::{ComScriptProcess, ComScriptProcessInit};
use super::imodqtassist_process::ImodqtassistProcess;
use super::interactive_system_program::InteractiveSystemProgram;
use super::intermittent_background_process::IntermittentBackgroundProcess;
use super::intermittent_process_monitor::IntermittentProcessMonitor;
use super::load_monitor::LoadMonitor;
use super::monitor::{DetachedProcessMonitor, Monitor, OutfileProcessMonitor, ProcessMonitor};
use super::process_data::ProcessData;
use super::process_interface::{
    ProcessInterface, ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface,
};
use super::process_messages::{MessageType, ProcessMessages};
use super::processchunks_process_monitor::{
    self, ProcesschunksProcessMonitor, ProcesschunksProcessMonitorImpl,
};
use super::reconnect_process::ReconnectProcess;
use super::system_program::SystemProgram;
use super::tomosetexts_output::TomosetextsOutput;
use super::tomosnapshot_process::TomosnapshotProcess;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::comscript_state::ComscriptState;
use crate::imod::etomo::comscript::detached_command_details::DetachedCommandDetails;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::python_info_param::PythonInfoParam;
use crate::imod::etomo::comscript::tomosnapshot_param;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::process_messages::MessagesArray;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Java `static final boolean DEBUG = EtomoDirector.INSTANCE.getArguments().isDebug()`.
fn debug_flag() -> bool {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug())
}

/// The methods Java subclasses of `BaseProcessManager` override.  The
/// defaults are the base class's bodies.
#[allow(unused_variables)]
pub trait BaseProcessManagerHooks: Send + Sync {
    /// Java `errorProcess(BackgroundProcess)`.
    fn error_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {}
    /// Java `errorProcess(ComScriptProcess)`.
    fn error_process_com_script(&self, base: &BaseProcessManager, process: &ComScriptProcess) {}
    /// Java `postProcess(ComScriptProcess)`.
    fn post_process_com_script(&self, base: &BaseProcessManager, script: &ComScriptProcess) {}
    /// Java `postProcess(InteractiveSystemProgram)`.
    fn post_process_interactive(
        &self,
        base: &BaseProcessManager,
        program: &InteractiveSystemProgram,
    ) {
    }
    /// Java `postProcess(BackgroundProcess)`.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
    }
    /// Java `errorProcess(DetachedProcess)`.
    fn error_process_detached(&self, base: &BaseProcessManager, process: &BackgroundProcess) {}
    /// Java `postProcess(DetachedProcess)`.
    fn post_process_detached(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_detached_base(process);
    }
    /// Java `errorProcess(ReconnectProcess)`.
    fn error_process_reconnect(&self, base: &BaseProcessManager, script: &ReconnectProcess) {}
    /// Java `postProcess(ReconnectProcess)`.
    fn post_process_reconnect(&self, base: &BaseProcessManager, script: &ReconnectProcess) {}
    /// Java public `processchunks(AxisID, ProcesschunksParam,
    /// ParallelProgressDisplay, ProcessResultDisplay, ProcessSeries, boolean,
    /// ProcessingMethod, boolean, RunType, ProcessData, List<ProcessMessages>)`.
    #[allow(clippy::too_many_arguments)]
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
        base.processchunks_base(
            axis_id,
            param,
            parallel_progress_display,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            run_type,
            managed_process_data,
            messages_array,
        )
    }
    /// Java public `reconnectProcesschunks(AxisID, ProcessData, ProcessResultDisplay,
    /// ProcessSeries, boolean, boolean, List<ProcessMessages>) throws LockException`.
    #[allow(clippy::too_many_arguments)]
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
        base.reconnect_processchunks_base(
            axis_id,
            process_data,
            process_result_display,
            process_series,
            multi_line_messages,
            popup_chunk_warnings,
            messages_array,
        )
    }
}

/// The base class's own hooks, for a manager that overrides none.
struct NoHooks;
impl BaseProcessManagerHooks for NoHooks {}

/// Java abstract `BaseProcessManager`.
pub struct BaseProcessManager {
    debug: AtomicBool,
    /// Java protected final `manager`.
    pub(crate) manager: &'static dyn BaseManager,
    /// Java final `axisProcessData`.
    pub(crate) axis_process_data: Arc<AxisProcessData>,
    hooks: OnceLock<&'static dyn BaseProcessManagerHooks>,
}

impl BaseProcessManager {
    /// Java protected `BaseProcessManager(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> BaseProcessManager {
        BaseProcessManager {
            debug: AtomicBool::new(false),
            manager,
            axis_process_data: manager.get_axis_process_data(),
            hooks: OnceLock::new(),
        }
    }

    /// Installs the subclass's overrides (Rust-only; the Java subclass
    /// overrides by inheritance).
    pub fn set_hooks(&self, hooks: &'static dyn BaseProcessManagerHooks) {
        let _ = self.hooks.set(hooks);
    }

    fn hooks(&self) -> &'static dyn BaseProcessManagerHooks {
        static NO_HOOKS: NoHooks = NoHooks;
        self.hooks.get().copied().unwrap_or(&NO_HOOKS)
    }

    /// Java `dumpState`.
    pub fn dump_state(&self) {
        self.axis_process_data.dump_state();
        if debug_flag() {
            eprintln!("[debug:{}]", self.debug.load(Ordering::SeqCst));
        }
    }

    /// Java package-private `setDebug`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.store(debug, Ordering::SeqCst);
    }

    /// Java static `isPython3`.
    pub fn is_python3() -> bool {
        let mut param = PythonInfoParam::new();
        param.set_greater_then_version(2);
        let is_python3 =
            SystemProgram::new_array(None, None, param.get_command_array(), AxisID::Only);
        is_python3.run();
        let std_out = is_python3.get_std_output();
        if let Some(std_out) = std_out
            && std_out.first().is_some_and(|line| line == "True")
        {
            return true;
        }
        false
    }

    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!("BaseProcessManager[{}]", self.param_string())
    }

    /// Java final `paramString`.
    pub fn param_string(&self) -> String {
        format!(
            "axisProcessData:{},uiHarness=UIHarness,",
            self.axis_process_data.to_source_string()
        )
    }

    /// Java final `writeLogFile(BackgroundProcess, AxisID, String)`.
    pub fn write_log_file(&self, process: &BackgroundProcess, axis_id: AxisID, file_name: &str) {
        // Write the standard output to a the log file
        let std_output = process.get_std_output();
        let Some(user_dir) = self.manager.get_property_user_dir() else {
            return;
        };
        let log_file = match LogFile::get_instance_user_dir(
            &user_dir,
            file_name,
            Some(self.manager.get_emergency_monitor(Some(axis_id))),
        ) {
            Ok(log_file) => log_file,
            Err(e) => {
                eprintln!("{}", e.get_message());
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    e.get_message(),
                    "log File Write Error".to_owned(),
                    Some(axis_id),
                );
                return;
            }
        };
        let writer_id = match log_file.open_writer() {
            Ok(writer_id) => writer_id,
            Err(LogFileError::Lock(_)) => return,
            Err(e) => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    e.get_message(),
                    "log File Write Error".to_owned(),
                    Some(axis_id),
                );
                return;
            }
        };
        let result = (|| -> Result<(), LogFileError> {
            if let Some(std_output) = &std_output {
                for line in std_output {
                    log_file.write(Some(line), &writer_id)?;
                    log_file.new_line(&writer_id)?;
                }
            }
            Ok(())
        })();
        log_file.close_id(Some(&writer_id));
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            Err(e) => ui_harness::post_message_dialog(
                Some(self.manager),
                e.get_message(),
                "log File Write Error".to_owned(),
                Some(axis_id),
            ),
        }
    }

    /// Java final `startLoad(IntermittentCommand, LoadMonitor)`.
    ///
    /// Upstream bug fixed in translation (BaseProcessManager.java:184): a null
    /// `param` or `monitor` throws NullPointerException inside
    /// IntermittentBackgroundProcess; here nothing is started.
    pub fn start_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        let (Some(param), Some(monitor)) = (param, monitor) else {
            return;
        };
        let monitor: Arc<dyn IntermittentProcessMonitor> = monitor;
        IntermittentBackgroundProcess::start_instance(self.manager, param, monitor);
    }

    /// Java final `endLoad(IntermittentCommand, LoadMonitor)`.
    ///
    /// Upstream bug fixed in translation (BaseProcessManager.java:188): a null
    /// `param` or `monitor` throws NullPointerException inside
    /// IntermittentBackgroundProcess; here nothing happens.
    pub fn end_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        let (Some(param), Some(monitor)) = (param, monitor) else {
            return;
        };
        IntermittentBackgroundProcess::end_instance(self.manager, &*param, &*monitor);
    }

    /// Java final `stopLoad(IntermittentCommand, LoadMonitor)`.
    ///
    /// Upstream bug fixed in translation (BaseProcessManager.java:192): a null
    /// `param` or `monitor` throws NullPointerException inside
    /// IntermittentBackgroundProcess; here nothing happens.
    pub fn stop_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        let (Some(param), Some(monitor)) = (param, monitor) else {
            return;
        };
        IntermittentBackgroundProcess::stop_instance(self.manager, &*param, &*monitor);
    }

    /// Java `xfmodel(XfmodelParam, AxisID, ProcessResultDisplay, ProcessSeries)`.
    /// `XfmodelParam` is a `CommandDetails`, so Java's overload resolution picks
    /// `startBackgroundProcess(CommandDetails, AxisID, ProcessResultDisplay,
    /// ProcessName, ProcessSeries)`: the process carries the param as its process
    /// details, which `JoinProcessManager.postProcess` reads the output file from.
    pub fn xfmodel(
        &'static self,
        param: Arc<dyn Command + Send + Sync>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.start_background_process_command(
            param,
            true,
            axis_id,
            Some(ProcessName::XFMODEL),
            process_result_display,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `reconnectProcesschunks(AxisID, ProcessData, ProcessResultDisplay,
    /// ProcessSeries, boolean multiLineMessages, boolean popupChunkWarnings,
    /// List<ProcessMessages>) throws LockException`.  Only a `LockException`
    /// comes back as `Err`.
    #[allow(clippy::too_many_arguments)]
    pub fn reconnect_processchunks(
        &'static self,
        axis_id: AxisID,
        process_data: Arc<Mutex<ProcessData>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        multi_line_messages: bool,
        popup_chunk_warnings: bool,
        messages_array: Option<MessagesArray>,
    ) -> Result<bool, LogFileError> {
        self.hooks().reconnect_processchunks(
            self,
            axis_id,
            process_data,
            process_result_display,
            process_series,
            multi_line_messages,
            popup_chunk_warnings,
            messages_array,
        )
    }

    /// The base class's `reconnectProcesschunks(AxisID, ProcessData,
    /// ProcessResultDisplay, ProcessSeries, boolean, boolean, List<ProcessMessages>)`
    /// body (the virtual call is `reconnect_processchunks`).
    #[allow(clippy::too_many_arguments)]
    pub fn reconnect_processchunks_base(
        &'static self,
        axis_id: AxisID,
        process_data: Arc<Mutex<ProcessData>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        multi_line_messages: bool,
        popup_chunk_warnings: bool,
        messages_array: Option<MessagesArray>,
    ) -> Result<bool, LogFileError> {
        let _ = messages_array;
        let monitor = {
            let data = process_data.lock().unwrap();
            ProcesschunksProcessMonitor::get_reconnect_instance(
                self.manager,
                axis_id,
                &data,
                multi_line_messages,
            )
        };
        self.reconnect_processchunks_monitor(
            axis_id,
            process_data,
            process_result_display,
            process_series,
            monitor,
            popup_chunk_warnings,
            false,
        )
    }

    /// Java final `reconnectProcesschunks(AxisID, ProcessData,
    /// ProcessResultDisplay, ProcessSeries, ProcesschunksProcessMonitor, boolean
    /// popupChunkWarnings, boolean reconnectWhenNotRunning) throws
    /// LockException`.  Only a `LockException` comes back as `Err`.
    #[allow(clippy::too_many_arguments)]
    pub fn reconnect_processchunks_monitor<S: ProcesschunksProcessMonitorImpl>(
        &'static self,
        axis_id: AxisID,
        process_data: Arc<Mutex<ProcessData>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        monitor: Arc<ProcesschunksProcessMonitor<S>>,
        popup_chunk_warnings: bool,
        reconnect_when_not_running: bool,
    ) -> Result<bool, LogFileError> {
        // `monitor.setSubdirName(processData.getSubDirName())`, the
        // ConstStringProperty overload: an empty name is null.
        let sub_dir_name = process_data.lock().unwrap().get_sub_dir_name();
        monitor
            .erased()
            .set_subdir_name(sub_dir_name.as_deref().filter(|name| !name.is_empty()));
        let process_monitor: Arc<dyn ProcessMonitor> = monitor.clone();
        let process = match ReconnectProcess::get_log_instance(
            self.manager,
            self,
            Some(process_monitor),
            Some(self.axis_process_data.get_saved_process_data(axis_id)),
            axis_id,
            &monitor.erased().get_log_file_name(),
            Some(processchunks_process_monitor::SUCCESS_TAG),
            sub_dir_name.as_deref(),
            process_series,
            popup_chunk_warnings,
            reconnect_when_not_running,
        ) {
            Ok(process) => process,
            Err(LogFileError::Lock(e)) => return Err(LogFileError::Lock(e)),
            Err(e) => {
                eprintln!("{}", e.get_message());
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("Unable to reconnect to processchunks.\n{}", e.get_message()),
                    "Reconnect Failure",
                    Some(axis_id),
                );
                return Ok(false);
            }
        };
        DetachedProcessMonitor::set_process(
            &*monitor,
            Arc::clone(&process) as Arc<dyn SystemProcessInterface>,
        );
        process.set_process_result_display(process_result_display);
        let thread_process = Arc::clone(&process);
        std::thread::spawn(move || thread_process.run());
        self.axis_process_data
            .map_axis_thread(Some(process.as_process()), axis_id);
        self.axis_process_data.map_axis_process_monitor(
            None,
            Some(monitor as Arc<dyn Monitor>),
            axis_id,
        );
        Ok(true)
    }

    /// Java final `tomodataplots`: `param.getCommandArray(manager, axisID)` is
    /// passed in as `command_array`.
    pub fn tomodataplots(&self, command_array: Vec<String>, axis_id: AxisID) {
        let program = SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(command_array),
            axis_id,
        );
        std::thread::spawn(move || program.run());
    }

    /// Java `midas`: run midas.
    pub fn midas(
        &'static self,
        midas_param: Arc<dyn Command + Send + Sync>,
    ) -> Result<Option<String>, SystemProcessException> {
        let program = self.start_interactive_system_program(midas_param)?;
        Ok(program.get_name())
    }

    /// Java public `processchunks(AxisID, ProcesschunksParam,
    /// ParallelProgressDisplay, ProcessResultDisplay, ProcessSeries, boolean
    /// popupChunkWarnings, ProcessingMethod, boolean multiLineMessages, RunType,
    /// ProcessData managedProcessData, List<ProcessMessages>)`, the virtual
    /// call (`ProcessManager` overrides it); see the module comment.  Runs on
    /// the event dispatch thread, where the progress display lives.
    #[allow(clippy::too_many_arguments)]
    pub fn processchunks(
        &'static self,
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
        self.hooks().processchunks(
            self,
            axis_id,
            param,
            parallel_progress_display,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            run_type,
            managed_process_data,
            messages_array,
        )
    }

    /// Java public `processchunks(AxisID, ProcesschunksParam, ...)`, the base
    /// class's body.
    #[allow(clippy::too_many_arguments)]
    pub fn processchunks_base(
        &'static self,
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
        let _ = (run_type, messages_array);
        self.processchunks_monitor(
            ProcesschunksProcessMonitor::new(
                self.manager,
                axis_id,
                param.get_root_name().as_deref(),
                Some(param.get_computer_map().into_iter().collect()),
                multi_line_messages,
            ),
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

    /// Java final `processchunks(ProcesschunksProcessMonitor, AxisID,
    /// ProcesschunksParam, ParallelProgressDisplay, ProcessResultDisplay,
    /// ProcessSeries, boolean, ProcessingMethod, ProcessData)`: run
    /// processchunks.
    #[allow(clippy::too_many_arguments)]
    pub fn processchunks_monitor<S: ProcesschunksProcessMonitorImpl>(
        &'static self,
        monitor: Arc<ProcesschunksProcessMonitor<S>>,
        axis_id: AxisID,
        param: Arc<ProcesschunksParam>,
        parallel_progress_display: &dyn ParallelProgressDisplay,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
    ) -> Result<String, AxisBusyException> {
        let detached_command_details: Arc<dyn DetachedCommandDetails + Send + Sync> = param.clone();
        let outfile_monitor: Arc<dyn OutfileProcessMonitor> = monitor.clone();
        let process = if param.is_subdir_name_empty() {
            self.start_detached_process(
                detached_command_details,
                axis_id,
                Some(outfile_monitor),
                process_result_display,
                Some(ProcessName::PROCESSCHUNKS),
                None,
                process_series,
                popup_chunk_warnings,
                processing_method,
                managed_process_data,
            )?
        } else {
            let subdir_name = param.get_subdir_name();
            monitor.erased().set_subdir_name(subdir_name.as_deref());
            let subdir_name = subdir_name.unwrap_or_else(|| "null".to_owned());
            let short_command_name = param.get_short_command_name();
            self.start_detached_process(
                detached_command_details,
                axis_id,
                Some(outfile_monitor),
                process_result_display,
                Some(ProcessName::PROCESSCHUNKS),
                Some((&subdir_name, &short_command_name)),
                process_series,
                popup_chunk_warnings,
                processing_method,
                managed_process_data,
            )?
        };
        parallel_progress_display.msg_process_started();
        Ok(process.get_name())
    }

    /// Java final `createNewFile`.
    pub fn create_new_file(&self, absolute_path: &str) {
        let file = Path::new(absolute_path);
        if file.exists() {
            return;
        }
        let dir = file.parent().unwrap_or(Path::new("."));
        if !dir.exists() && std::fs::create_dir_all(dir).is_err() {
            ui_harness::post_message_dialog(
                Some(self.manager),
                format!("Unable to create {}", dir.display()),
                "File Error".to_owned(),
                None,
            );
            return;
        }
        if dir
            .metadata()
            .map(|metadata| metadata.permissions().readonly())
            .unwrap_or(true)
        {
            ui_harness::post_message_dialog(
                Some(self.manager),
                format!("Cannot write to {}", dir.display()),
                "File Error".to_owned(),
                None,
            );
            return;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(file)
        {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Cannot create  {}", file.display()),
                    "File Error".to_owned(),
                    None,
                );
            }
            Err(e) => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Cannot create  {}.\n{e}", file.display()),
                    "File Error".to_owned(),
                    None,
                );
            }
        }
    }

    /// Java static final `touch`: run touch command on file.
    pub fn touch(absolute_path: &str, manager: Option<&'static dyn BaseManager>) {
        let file = Path::new(absolute_path);
        let dir = file.parent().unwrap_or(Path::new("."));
        if !dir.exists() && std::fs::create_dir_all(dir).is_err() {
            ui_harness::open_message_dialog_from_process(
                manager,
                &format!("Unable to create {}", dir.display()),
                "File Error",
                None,
            );
            return;
        }
        // `File.canWrite()`: the access check, not the mode bits.
        if !crate::imod::etomo::util::utilities::java_io_file_can_write(&dir.to_string_lossy()) {
            ui_harness::open_message_dialog_from_process(
                manager,
                &format!("Cannot write to {}", dir.display()),
                "File Error",
                None,
            );
            return;
        }
        let python_script_path = python_script_path();
        let command_array = vec![
            "python".to_owned(),
            format!("{python_script_path}b3dtouch"),
            absolute_path.to_owned(),
        ];
        BaseProcessManager::start_system_program_thread(command_array, AxisID::Only, manager);
        let timeout = 5;
        let mut t = 0;
        while !file.exists() && t < timeout {
            std::thread::sleep(std::time::Duration::from_millis(1020));
            t += 1;
        }
    }

    /// Java static final `tomosetexts`: run b3dtomosetexts command on dir.
    pub fn tomosetexts(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dir: &Path,
    ) -> Option<TomosetextsOutput> {
        if !dir.exists() {
            return None;
        }
        if std::fs::read_dir(dir).is_err() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!("Cannot write to {}", dir.display()),
                "File Error".to_owned(),
                None,
            );
            return None;
        }
        let system_program = SystemProgram::new_array(
            Some(manager),
            manager.get_property_user_dir(),
            Some(vec![
                "python".to_owned(),
                format!("{}b3dtomosetexts", python_script_path()),
                utilities::java_io_file_get_absolute_path(&dir.to_string_lossy()),
            ]),
            axis_id,
        );
        system_program.run();
        let stdout = system_program.get_std_output()?;
        Some(TomosetextsOutput::new(Some(&stdout)))
    }

    /// Java `imodqtassistQuery`: runs `imodqtassist -t` and returns its
    /// standard output.
    pub fn imodqtassist_query(&self, axis_id: AxisID) -> Option<Vec<String>> {
        let process = ImodqtassistProcess::get_query_instance();
        process.run(self.manager, axis_id);
        process.get_standard_output()
    }

    /// Java final `startComScript(String, ProcessMonitor, AxisID,
    /// ProcessResultDisplay, CommandDetails, ProcessSeries)` and
    /// `startComScript(String, ProcessMonitor, AxisID, ProcessResultDisplay,
    /// Command, ProcessSeries)`.
    pub fn start_com_script_command(
        &'static self,
        command_string: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        command: Option<Arc<dyn Command + Send + Sync>>,
        is_command_details: bool,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command_string.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: process_monitor.clone(),
            process_result_display,
            process_series,
            command,
            is_command_details,
            resumable: None,
            file_type: None,
            processing_method: None,
        });
        self.start_com_script_process(process, command_string, process_monitor, axis_id)
    }

    /// Java final `startOutfileComScript`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_outfile_com_script(
        &'static self,
        command_string: &str,
        monitor: Arc<dyn OutfileProcessMonitor>,
        axis_id: AxisID,
        command: Option<Arc<dyn Command + Send + Sync>>,
        file_type: Option<&'static FileType>,
        process_name: Option<ProcessName>,
        reconnect_when_not_running: bool,
        managed_process_data: Arc<Mutex<ProcessData>>,
        indeterminate_mode: bool,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let detached: Arc<dyn DetachedProcessMonitor> = monitor.clone();
        let process = ComScriptProcess::new_outfile(
            self.manager,
            command_string.to_owned(),
            self,
            axis_id,
            detached,
            command,
            file_type,
            process_name,
            reconnect_when_not_running,
            managed_process_data,
            indeterminate_mode,
        );
        monitor.set_process(process.clone());
        let process_monitor: Arc<dyn ProcessMonitor> = monitor;
        self.start_com_script_process(process, command_string, Some(process_monitor), axis_id)
    }

    /// Java final `startComScript(String, ProcessMonitor, AxisID,
    /// ProcessResultDisplay, ProcessSeries, boolean resumable)`.
    pub fn start_com_script_resumable(
        &'static self,
        command: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        resumable: bool,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: process_monitor.clone(),
            process_result_display,
            process_series,
            command: None,
            is_command_details: false,
            resumable: Some(resumable),
            file_type: None,
            processing_method: None,
        });
        self.start_com_script_process(process, command, process_monitor, axis_id)
    }

    /// Java final `startComScript(String, AxisID, ProcessSeries, FileType)`.
    pub fn start_com_script_file_type(
        &'static self,
        command: &str,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
        file_type: Option<&'static FileType>,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: None,
            process_result_display: None,
            process_series,
            command: None,
            is_command_details: false,
            resumable: Some(false),
            file_type,
            processing_method: None,
        });
        self.start_com_script_process(process, command, None, axis_id)
    }

    /// Java final `startComScript(String, ProcessMonitor, AxisID,
    /// ProcessResultDisplay, Command, ProcessSeries, FileType)`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_com_script_command_file_type(
        &'static self,
        command_string: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        command: Option<Arc<dyn Command + Send + Sync>>,
        process_series: Option<ProcessSeriesRef>,
        file_type: Option<&'static FileType>,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command_string.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: process_monitor.clone(),
            process_result_display,
            process_series,
            command,
            is_command_details: false,
            resumable: None,
            file_type,
            processing_method: None,
        });
        self.start_com_script_process(process, command_string, process_monitor, axis_id)
    }

    /// Java final `startNonBlockingComScript(String, AxisID,
    /// ProcessResultDisplay)`.
    pub fn start_non_blocking_com_script(
        &'static self,
        command: &str,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: None,
            process_result_display,
            process_series: None,
            command: None,
            is_command_details: false,
            resumable: Some(false),
            file_type: None,
            processing_method: None,
        });
        self.start_non_blocking_com_script_process(&process, command, axis_id);
    }

    /// Java final `startComScript(String, ProcessMonitor, AxisID, ProcessSeries,
    /// boolean resumable)`.
    pub fn start_com_script_series_resumable(
        &'static self,
        command: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
        resumable: bool,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        self.start_com_script_watched_file(
            command,
            process_monitor,
            axis_id,
            None,
            process_series,
            resumable,
        )
    }

    /// Java final `startComScript(CommandDetails, ProcessMonitor, AxisID,
    /// ProcessSeries)`, `startComScript(CommandDetails, ProcessMonitor, AxisID,
    /// ProcessResultDisplay, ProcessSeries[, ProcessingMethod])` and
    /// `startComScript(Command, ProcessMonitor, AxisID, ProcessResultDisplay,
    /// ProcessSeries[, ProcessingMethod])`: the com script is
    /// `command.getCommand()`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_com_script_param(
        &'static self,
        command: Arc<dyn Command + Send + Sync>,
        is_command_details: bool,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let com_script = command.get_command().unwrap_or_default();
        let command_line = command.get_command_line().unwrap_or_default();
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script,
            process_manager: self,
            axis_id,
            watched_file_name: None,
            process_monitor: process_monitor.clone(),
            process_result_display,
            process_series,
            command: Some(command),
            is_command_details,
            resumable: None,
            file_type: None,
            processing_method,
        });
        self.start_com_script_process(process, &command_line, process_monitor, axis_id)
    }

    /// Java final `startBackgroundComScript(String, DetachedProcessMonitor,
    /// AxisID, ComscriptState, String watchedFileName, ProcessSeries, boolean
    /// resumable)`: start a managed background command script for the
    /// specified axis.
    #[allow(clippy::too_many_arguments)]
    pub fn start_background_com_script(
        &'static self,
        comscript: &str,
        process_monitor: Arc<dyn DetachedProcessMonitor>,
        axis_id: AxisID,
        comscript_state: Option<Arc<dyn ComscriptState + Send + Sync>>,
        watched_file_name: Option<&str>,
        process_series: Option<ProcessSeriesRef>,
        resumable: bool,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = BackgroundComScriptProcess::new(
            self.manager,
            comscript.to_owned(),
            self,
            axis_id,
            watched_file_name.map(str::to_owned),
            Arc::clone(&process_monitor),
            comscript_state,
            process_series,
            resumable,
        );
        process_monitor.set_process(Arc::clone(&process) as Arc<dyn SystemProcessInterface>);
        let monitor: Arc<dyn ProcessMonitor> = process_monitor;
        self.start_com_script_process(process, comscript, Some(monitor), axis_id)
    }

    /// Java final `startComScript(String, ProcessMonitor, AxisID, String
    /// watchedFileName, ProcessSeries, boolean resumable)`.
    pub fn start_com_script_watched_file(
        &'static self,
        command: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
        watched_file_name: Option<&str>,
        process_series: Option<ProcessSeriesRef>,
        resumable: bool,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager: self.manager,
            com_script: command.to_owned(),
            process_manager: self,
            axis_id,
            watched_file_name: watched_file_name.map(str::to_owned),
            process_monitor: process_monitor.clone(),
            process_result_display: None,
            process_series,
            command: None,
            is_command_details: false,
            resumable: Some(resumable),
            file_type: None,
            processing_method: None,
        });
        self.start_com_script_process(process, command, process_monitor, axis_id)
    }

    /// Java final `startMonitor`.
    pub fn start_monitor(
        &self,
        monitor: Arc<dyn Monitor>,
        axis_id: AxisID,
        _reconnect_when_not_running: bool,
    ) {
        let thread_monitor = Arc::clone(&monitor);
        let process_monitor_thread = std::thread::spawn(move || thread_monitor.run());
        self.axis_process_data.map_axis_process_monitor(
            Some(process_monitor_thread),
            Some(monitor),
            axis_id,
        );
    }

    /// Java final `startComScript(ComScriptProcess, String, ProcessMonitor,
    /// AxisID)`: the one every overload ends in.
    pub fn start_com_script_process(
        &'static self,
        com_script_process: Arc<ComScriptProcess>,
        command: &str,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
        axis_id: AxisID,
    ) -> Result<Arc<ComScriptProcess>, AxisBusyException> {
        // Make sure there isn't something going on in the current axis
        self.is_axis_busy(axis_id, com_script_process.get_process_result_display())?;
        com_script_process.close_output_image_file();
        // Run the script as a thread in the background
        com_script_process.set_working_directory(PathBuf::from(
            self.manager.get_property_user_dir().unwrap_or_default(),
        ));
        com_script_process.set_debug(etomo_director::ARGUMENTS.lock().unwrap().is_debug());
        self.manager.save_storables(Some(axis_id));
        com_script_process.start();
        // Map the thread to the correct axis
        self.axis_process_data
            .map_axis_thread(Some(com_script_process.as_process()), axis_id);
        if debug_flag() {
            eprintln!("Started {command} (1)");
            eprintln!("  Name: {}", com_script_process.get_name());
        }
        // Start the process monitor thread if a runnable process is provided
        if let Some(process_monitor) = process_monitor {
            // Wait for the started flag within the comScriptProcess, this
            // ensures that log file has already been moved
            let process = Arc::clone(&com_script_process);
            let axis_process_data = Arc::clone(&self.axis_process_data);
            std::thread::spawn(move || {
                BaseProcessManager::start_com_script_monitor(
                    &axis_process_data,
                    &process,
                    process_monitor,
                    axis_id,
                );
            });
        }
        Ok(com_script_process)
    }

    /// Java final `startComScriptMonitor`.
    fn start_com_script_monitor(
        axis_process_data: &AxisProcessData,
        com_script_process: &ComScriptProcess,
        process_monitor: Arc<dyn ProcessMonitor>,
        axis_id: AxisID,
    ) {
        while !com_script_process.is_started() && !com_script_process.is_error() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let monitor: Arc<dyn Monitor> = process_monitor;
        let thread_monitor = Arc::clone(&monitor);
        let process_monitor_thread = std::thread::spawn(move || thread_monitor.run());
        axis_process_data.map_axis_process_monitor(
            Some(process_monitor_thread),
            Some(monitor),
            axis_id,
        );
    }

    /// Java final `startNonBlockingComScript(ComScriptProcess, String,
    /// AxisID)`: start an unmanaged comscript.
    fn start_non_blocking_com_script_process(
        &self,
        com_script_process: &ComScriptProcess,
        command: &str,
        axis_id: AxisID,
    ) {
        // Run the script as a thread in the background
        com_script_process.close_output_image_file();
        com_script_process.set_working_directory(PathBuf::from(
            self.manager.get_property_user_dir().unwrap_or_default(),
        ));
        com_script_process.set_debug(etomo_director::ARGUMENTS.lock().unwrap().is_debug());
        com_script_process.set_non_blocking();
        self.manager.save_storables(Some(axis_id));
        com_script_process.start();
        if debug_flag() {
            eprintln!("Started {command} (2)");
            eprintln!("  Name: {}", com_script_process.get_name());
        }
    }

    /// Java final `inUse`.
    pub fn in_use(
        &self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        popup_error: bool,
    ) -> bool {
        if let Err(e) = self.is_axis_busy(axis_id, process_result_display) {
            eprintln!("{}", e.0);
            if popup_error {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!(
                        "A process is already executing{}",
                        if self.is_dual_axis() {
                            " in the current axis"
                        } else {
                            ""
                        }
                    ),
                    "Cannot run process".to_owned(),
                    Some(axis_id),
                );
            }
            return true;
        }
        false
    }

    /// Java private `isDualAxis`: returns true if this is a dual axis
    /// dataset.
    fn is_dual_axis(&self) -> bool {
        match self.manager.get_base_meta_data() {
            Some(meta_data) => meta_data.base().get_axis_type() == AxisType::DualAxis,
            None => false,
        }
    }

    /// Java final `isAxisBusy`: check to see if specified axis is busy.
    pub fn is_axis_busy(
        &self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Result<(), AxisBusyException> {
        // Check to make sure there is not another process already running on
        // this axis.
        let busy = if axis_id == AxisID::Second {
            !self.axis_process_data.is_thread_axis_null(AxisID::Second)
        } else {
            !self.axis_process_data.is_thread_axis_null(AxisID::First)
        };
        if busy {
            let mut key = String::new();
            if let Some(display) = &process_result_display {
                let display = display.get();
                display.msg_process_failed_to_start();
                key = format!(
                    "\nprocessResultDisplay:{}",
                    display
                        .get_button_state_key()
                        .unwrap_or_else(|| "null".to_owned())
                );
            }
            return Err(AxisBusyException(format!(
                "A process is already executing{}{key}",
                if self.is_dual_axis() {
                    " in the current axis"
                } else {
                    ""
                }
            )));
        }
        // check for running processes that are not managed by Etomo because the
        // user exited and then reran Etomo
        let saved_process_data = self.axis_process_data.get_saved_process_data(axis_id);
        let running = saved_process_data.lock().unwrap().is_running();
        if running {
            if let Some(display) = &process_result_display {
                display.get().msg_process_failed_to_start();
            }
            let data = saved_process_data.lock().unwrap();
            let mut message = format!(
                "A process is running{}.\nThe process was started the last time Etomo was run",
                if self.is_dual_axis() {
                    " in the current axis"
                } else {
                    ""
                }
            );
            if data.is_on_different_host() {
                message.push_str(&format!(" on {}", data.get_host_name()));
            }
            message.push_str(&format!(
                ".\n\nReferences:\nProcessName={}\n{}\nPID = {}",
                data.get_process_name()
                    .map_or_else(|| "null".to_owned(), |name| name.to_string()),
                self.manager
                    .get_param_file()
                    .map_or_else(|| "null".to_owned(), |file| file.display().to_string()),
                data.get_pid().unwrap_or_else(|| "null".to_owned())
            ));
            return Err(AxisBusyException(message));
        }
        // ensure that out of date process info won't be resaved
        saved_process_data.lock().unwrap().reset();
        self.save_process_data(axis_id, &saved_process_data);
        if axis_id == AxisID::Second {
            if self.axis_process_data.is_block_axis(AxisID::Second) {
                return Err(AxisBusyException(
                    "Process attempting to restart - axis B is blocked.".to_owned(),
                ));
            }
        } else if self.axis_process_data.is_block_axis(AxisID::First) {
            return Err(AxisBusyException(
                "Process attempting to restart - axis A is blocked.".to_owned(),
            ));
        }
        Ok(())
    }

    /// Java final `unblockAxis`.
    pub fn unblock_axis(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.axis_process_data.set_block_axis(AxisID::Second, false);
        } else {
            self.axis_process_data.set_block_axis(AxisID::First, false);
        }
    }

    /// Java private `saveProcessData`.
    fn save_process_data(&self, axis_id: AxisID, process_data: &Mutex<ProcessData>) {
        let _ = axis_id;
        let param_file = self.manager.get_param_file();
        let Ok(Some(mut param_store)) = ParameterStore::get_instance(param_file) else {
            return;
        };
        let _ = param_store.save(Some(process_data));
    }

    /// Java final `getProcessData`.
    pub fn get_process_data(&self, axis_id: AxisID) -> Arc<Mutex<ProcessData>> {
        let Some(thread) = self.axis_process_data.get_thread(axis_id) else {
            return self.axis_process_data.get_saved_process_data(axis_id);
        };
        match thread.get_process_data() {
            None => self.axis_process_data.get_saved_process_data(axis_id),
            Some(process_data) => process_data,
        }
    }

    /// Java final `pause`.
    pub fn pause(&self, axis_id: AxisID) -> bool {
        match self.axis_process_data.get_thread(axis_id) {
            None => false,
            Some(thread) => thread.pause(axis_id),
        }
    }

    /// Java final `kill(AxisID)`.
    pub fn kill(&self, axis_id: AxisID) {
        if let Some(thread) = self.axis_process_data.get_thread(axis_id) {
            thread.kill(axis_id);
        }
    }

    /// Java final `signalKill`: kill the thread for the specified axis.
    pub fn signal_kill(&self, thread: &dyn SystemProcessInterface, axis_id: AxisID) {
        thread.set_process_end_state(ProcessEndState::Killed);
        let process_id = thread.get_shell_process_id();
        self.kill_process_id(&process_id, axis_id);
        std::thread::sleep(std::time::Duration::from_millis(200));
        thread.notify_killed();
    }

    /// Java private `kill(String processID, AxisID)`: `imodkillgroup`.
    fn kill_process_id(&self, process_id: &str, axis_id: AxisID) {
        if debug_flag() {
            eprintln!("killing {process_id}");
        }
        let windows = utilities::is_windows_os();
        let mut command = vec![
            "python".to_owned(),
            format!("{}imodkillgroup", python_script_path()),
        ];
        if !windows {
            command.push("-t".to_owned());
        }
        command.push(process_id.to_owned());
        let kill_shell = SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(command),
            axis_id,
        );
        kill_shell.run();
        utilities::debug_print(&format!(
            "imodkillgroup {process_id} at {:?}",
            kill_shell.get_run_timestamp()
        ));
    }

    /// Java private `getChildProcessList`: the PIDs of child processes for
    /// the specified parent process.  Unused in the Java (no caller).
    pub fn get_child_process_list(&self, process_id: &str, axis_id: AxisID) -> Option<Vec<String>> {
        utilities::debug_print(&format!("in getChildProcessList: processID={process_id}"));
        // ps -l: get user processes on this terminal
        let ps = SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(vec!["ps".to_owned(), "axl".to_owned()]),
            axis_id,
        );
        ps.run();
        // Find the index of the Parent ID and ProcessID
        let stdout = ps.get_std_output()?;
        let header = stdout.first()?.trim().to_owned();
        let labels: Vec<&str> = header.split_whitespace().collect();
        let mut idx_pid: i32 = -1;
        let mut idx_ppid: i32 = -1;
        let mut idx_cmd: i32 = -1;
        let mut found = 0;
        for (i, label) in labels.iter().enumerate() {
            if *label == "PID" {
                idx_pid = i as i32;
                found += 1;
            }
            if *label == "PPID" {
                idx_ppid = i as i32;
                found += 1;
            }
            if *label == "CMD" || *label == "COMMAND" {
                idx_cmd = i as i32;
                found += 1;
            }
            if found >= 3 {
                break;
            }
        }
        // Return null if the PID or PPID fields are not found
        if idx_ppid == -1 || idx_pid == -1 {
            return None;
        }
        // Walk through the process list finding the PID of the children
        let mut children_pid = Vec::new();
        for line in stdout.iter().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.get(idx_ppid as usize) == Some(&process_id)
                && !self
                    .axis_process_data
                    .contains_key_killed_list(fields[idx_pid as usize])
            {
                if idx_cmd != -1 {
                    utilities::debug_print(&format!(
                        "child found:PID={},PPID={},name={}",
                        fields[idx_pid as usize],
                        fields[idx_ppid as usize],
                        fields.get(idx_cmd as usize).unwrap_or(&"")
                    ));
                }
                children_pid.push(fields[idx_pid as usize].to_owned());
            }
        }
        // If there are no children return null
        if children_pid.is_empty() {
            return None;
        }
        Some(children_pid)
    }

    /// Java private `logProcessOutput`: log the end of the stdout and stderr,
    /// unless debug is on.
    fn log_process_output(
        &self,
        command_action: &str,
        std_output: Option<&[String]>,
        std_error: Option<&[String]>,
    ) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            return;
        }
        let print_std_output = std_output.is_some_and(|output| !output.is_empty());
        let print_std_error =
            std_error.is_some_and(|error| !error.is_empty() && !error[0].contains("PID:"));
        if !print_std_output && !print_std_error {
            return;
        }
        if debug_flag() {
            eprintln!("Output from {command_action}:");
        }
        let lines_to_log = 10;
        if debug_flag() {
            let mut printed = false;
            if print_std_output && let Some(std_output) = std_output {
                eprintln!("Standard out:");
                for line in &std_output[std_output.len().saturating_sub(lines_to_log)..] {
                    eprintln!("{line}");
                    printed = true;
                }
            }
            if print_std_error && let Some(std_error) = std_error {
                eprintln!("Standard error:");
                for line in &std_error[std_error.len().saturating_sub(lines_to_log)..] {
                    eprintln!("{line}");
                    printed = true;
                }
            }
            if printed {
                eprintln!();
            }
        }
    }

    /// Java final `msgComScriptDone(OutfileComScriptProcess, int, boolean)`.
    pub fn msg_com_script_done_outfile(
        &self,
        process: &ComScriptProcess,
        exit_value: i32,
        _non_blocking: bool,
    ) {
        let end_state = process.get_process_end_state();
        let failed =
            exit_value != 0 || (end_state.is_some() && end_state != Some(ProcessEndState::Done));
        if failed {
            if let Some(process_messages) = process.get_monitor_process_messages()
                && !process_messages.is_empty(Some(MessageType::Error))
            {
                ui_harness::open_error_message_dialog_and_wait(
                    Some(self.manager),
                    process_messages,
                    "Comscript Terminated (1)".to_owned(),
                    process.get_axis_id(),
                );
            }
            self.hooks().error_process_com_script(self, process);
        } else {
            self.log_process_output(
                &process.get_command_action(),
                process.get_std_output().as_deref(),
                process.get_std_error().as_deref(),
            );
            self.hooks().post_process_com_script(self, process);
        }
        self.manager.save_storables(Some(process.get_axis_id()));
        self.axis_process_data
            .clear_thread(process, ClearKind::ComScript);
        // Inform the manager that this process is complete
        self.post_process_done(
            process.get_name(),
            exit_value,
            process.get_process_name(),
            process.get_axis_id(),
            false,
            process.get_process_end_state(),
            process.get_status_string(),
            failed,
            process.get_process_result_display(),
            process.get_process_series(),
            false,
        );
        if process.get_process_series().is_none() {
            let process_data = process.get_process_data();
            let sub_process_name = process_data
                .as_ref()
                .and_then(|data| data.lock().unwrap().get_sub_process_name());
            utilities::timestamp_process_name_subprocess(
                Some(&process.get_name()),
                process.get_process_name(),
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// Java final `msgComScriptDone(AxisID, ComScriptProcess, int, boolean)`:
    /// a com script has finished execution.
    pub fn msg_com_script_done(
        &self,
        axis_id: AxisID,
        script: &ComScriptProcess,
        exit_value: i32,
        non_blocking: bool,
    ) {
        let process_name = script.get_process_name();
        let end_state = script.get_process_end_state();
        if debug_flag() {
            eprintln!(
                "msgComScriptDone:scriptName={},processName={:?}",
                script.get_com_script_name(),
                process_name
            );
        }
        if end_state != Some(ProcessEndState::FileLockFailure) {
            if exit_value != 0 {
                let std_error = script.get_std_error();
                let mut combined_messages =
                    ProcessMessages::get_instance(Some(self.manager), axis_id);
                // Is the last string "Killed"
                if let Some(std_error) = &std_error
                    && std_error.last().is_some_and(|line| line.trim() == "Killed")
                {
                    combined_messages.add_message(
                        MessageType::Error,
                        &format!("<html>Terminated: {}", script.get_com_script_name()),
                    );
                } else {
                    let messages = script.get_process_messages().clone();
                    combined_messages.add_message(
                        MessageType::Error,
                        &format!("<html>Com script failed: {}", script.get_com_script_name()),
                    );
                    combined_messages.add_from(
                        MessageType::Error,
                        Some("\n<html><U>Log file errors:</U>"),
                        Some(&messages),
                    );
                    combined_messages.add_array(
                        MessageType::Error,
                        Some("\n<html><U>Standard error output:</U>"),
                        std_error.as_deref(),
                    );
                }
                if end_state != Some(ProcessEndState::Killed)
                    && end_state != Some(ProcessEndState::Paused)
                {
                    ui_harness::open_error_message_dialog_and_wait(
                        Some(self.manager),
                        combined_messages,
                        "Comscript Terminated (2)".to_owned(),
                        script.get_axis_id(),
                    );
                    // make sure script knows about failure
                    script.set_process_end_state(ProcessEndState::Failed);
                }
                self.hooks().error_process_com_script(self, script);
            } else {
                self.log_process_output(
                    &script.get_command_action(),
                    script.get_std_output().as_deref(),
                    script.get_std_error().as_deref(),
                );
                self.hooks().post_process_com_script(self, script);
                let mut messages = script.get_process_messages();
                if messages.size(MessageType::Warning) > 0 {
                    messages.add_message(
                        MessageType::Warning,
                        &format!("Com script: {}", script.get_com_script_name()),
                    );
                    let copy = messages.clone();
                    ui_harness::post_warning_message_dialog(
                        Some(self.manager),
                        copy,
                        "Comscript Warnings".to_owned(),
                        script.get_axis_id(),
                    );
                }
            }
        }
        self.manager.save_storables(Some(script.get_axis_id()));
        self.axis_process_data
            .clear_thread(script, ClearKind::ComScript);
        // Inform the app manager that this process is complete
        self.post_process_done(
            script.get_name(),
            exit_value,
            process_name.clone(),
            script.get_axis_id(),
            false,
            end_state,
            None,
            exit_value != 0,
            script.get_process_result_display(),
            script.get_process_series(),
            non_blocking,
        );
        if script.get_process_series().is_none() {
            let process_data = script.get_process_data();
            let sub_process_name = process_data
                .as_ref()
                .and_then(|data| data.lock().unwrap().get_sub_process_name());
            utilities::timestamp_process_name_subprocess(
                Some(&script.get_name()),
                process_name,
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// Java final `msgReconnectDone(AxisID, ReconnectProcess, int, boolean)`.
    ///
    /// Upstream bug fixed in translation (BaseProcessManager.java:1168): a
    /// reconnect without ProcessData (or one without a process name) throws
    /// NullPointerException; here its name is "null".
    pub fn msg_reconnect_done(
        &self,
        axis_id: AxisID,
        script: &ReconnectProcess,
        exit_value: i32,
        popup_chunk_warnings: bool,
    ) {
        let process_data = script.get_process_data();
        let process_name = process_data
            .as_ref()
            .and_then(|process_data| process_data.lock().unwrap().get_process_name());
        let name = process_name
            .as_ref()
            .map_or_else(|| "null".to_owned(), |name| name.to_string());
        // Java also reads `processData.getSubProcessName()` into a local it never
        // uses.
        if debug_flag() {
            eprintln!("msgReconnectDone:processName={name}");
        }
        if exit_value != 0 {
            let std_error = script.get_std_error();
            let mut combined_messages = ProcessMessages::get_instance(Some(self.manager), axis_id);
            // Is the last string "Killed"
            if let Some(std_error) = &std_error
                && std_error.last().is_some_and(|line| line.trim() == "Killed")
            {
                combined_messages
                    .add_message(MessageType::Error, &format!("<html>Terminated: {name}"));
            } else {
                let messages = script.get_process_messages();
                combined_messages.add_message(
                    MessageType::Error,
                    &format!("<html>Com script failed: {name}"),
                );
                combined_messages.add_from(
                    MessageType::Error,
                    Some("\n<html><U>Log file errors:</U>"),
                    messages.as_ref(),
                );
                combined_messages.add_array(
                    MessageType::Error,
                    Some("\n<html><U>Standard error output:</U>"),
                    std_error.as_deref(),
                );
                combined_messages.add_from_type(
                    MessageType::Error,
                    MessageType::ChunkError,
                    Some("<html><U>Chunk errors:</U>"),
                    messages.as_ref(),
                );
            }
            let end_state = script.get_process_end_state();
            if end_state != Some(ProcessEndState::Killed)
                && end_state != Some(ProcessEndState::Paused)
            {
                ui_harness::open_error_message_dialog_and_wait(
                    Some(self.manager),
                    combined_messages,
                    "Reconnect Terminated".to_owned(),
                    script.get_axis_id(),
                );
                if debug_flag() {
                    eprint!(
                        "Reconnect Terminated ({}):exitValue:{exit_value},\nscript:",
                        utilities::get_date_time_stamp()
                    );
                }
                script.dump_state(2);
                // make sure script knows about failure
                script.set_process_end_state(ProcessEndState::Failed);
            }
            self.hooks().error_process_reconnect(self, script);
        } else {
            self.log_process_output(
                &name,
                script.get_std_output().as_deref(),
                script.get_std_error().as_deref(),
            );
            self.hooks().post_process_reconnect(self, script);
            if let Some(mut messages) = script.get_process_messages()
                && popup_chunk_warnings
                && messages.size(MessageType::Warning) > 0
            {
                messages.add_message(MessageType::Warning, &format!("Com script: {name}"));
                ui_harness::post_warning_message_dialog(
                    Some(self.manager),
                    messages,
                    "Reconnect Warnings".to_owned(),
                    script.get_axis_id(),
                );
            }
        }
        self.manager.save_storables(Some(script.get_axis_id()));
        self.axis_process_data
            .clear_thread(script, ClearKind::Reconnect);
        // Inform the app manager that this process is complete
        let end_state = script.get_process_end_state();
        self.post_process_done(
            name.clone(),
            exit_value,
            process_name.clone(),
            script.get_axis_id(),
            false,
            end_state,
            None,
            end_state != Some(ProcessEndState::Done) || exit_value != 0,
            script.get_process_result_display(),
            script.get_process_series(),
            false,
        );
        // Get rid of process data after the process is complete.
        if let Some(process_data) = &process_data {
            process_data.lock().unwrap().reset();
        }
        if script.get_process_series().is_none() {
            // Java reads the names again after the reset.
            let (process_name, sub_process_name) = match &process_data {
                Some(process_data) => {
                    let data = process_data.lock().unwrap();
                    (data.get_process_name(), data.get_sub_process_name())
                }
                None => (None, None),
            };
            utilities::timestamp_process_name_subprocess(
                Some(&name),
                process_name,
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// Java final `msgReconnectDone(LoggedReconnectProcess, int, boolean,
    /// boolean)`.
    ///
    /// Upstream bug fixed in translation (BaseProcessManager.java:1239, :1247):
    /// a reconnect without ProcessData (or a process name), or without
    /// messages, throws NullPointerException; here the name is "null" and
    /// missing messages are empty.
    pub fn msg_reconnect_done_logged(
        &self,
        script: &ReconnectProcess,
        exit_value: i32,
        _popup_chunk_warnings: bool,
        no_popup_without_message: bool,
    ) {
        let process_data = script.get_process_data();
        let process_name = process_data
            .as_ref()
            .and_then(|process_data| process_data.lock().unwrap().get_process_name());
        let name = process_name
            .as_ref()
            .map_or_else(|| "null".to_owned(), |name| name.to_string());
        if debug_flag() {
            eprintln!("msgReconnectDone:processName={name}");
        }
        if exit_value != 0 {
            let messages = script.get_process_messages();
            let end_state = script.get_process_end_state();
            if end_state != Some(ProcessEndState::Killed)
                && end_state != Some(ProcessEndState::Paused)
            {
                match messages {
                    Some(messages) if !messages.is_empty(Some(MessageType::Error)) => {
                        ui_harness::open_error_message_dialog_and_wait(
                            Some(self.manager),
                            messages,
                            "Reconnect Terminated".to_owned(),
                            script.get_axis_id(),
                        );
                    }
                    _ if !no_popup_without_message => {
                        ui_harness::post_message_dialog(
                            Some(self.manager),
                            format!(
                                "Unable to reconnect.\n\nEnd state:{},exit value:{exit_value}",
                                end_state
                                    .map_or_else(|| "null".to_owned(), |state| state.to_string())
                            ),
                            "Reconnect Terminated".to_owned(),
                            Some(script.get_axis_id()),
                        );
                    }
                    _ => {}
                }
                if debug_flag() {
                    eprint!(
                        "Reconnect Terminated ({}):exitValue:{exit_value}",
                        utilities::get_date_time_stamp()
                    );
                }
                // script.dumpState(2);
                // make sure script knows about failure
                script.set_process_end_state(ProcessEndState::Failed);
            }
            self.hooks().error_process_reconnect(self, script);
        } else {
            self.hooks().post_process_reconnect(self, script);
        }
        self.manager.save_storables(Some(script.get_axis_id()));
        self.axis_process_data
            .clear_thread(script, ClearKind::Reconnect);
        // Inform the app manager that this process is complete
        let end_state = script.get_process_end_state();
        self.post_process_done(
            name.clone(),
            exit_value,
            process_name.clone(),
            script.get_axis_id(),
            false,
            end_state,
            None,
            end_state != Some(ProcessEndState::Done) || exit_value != 0,
            script.get_process_result_display(),
            script.get_process_series(),
            false,
        );
        if script.get_process_series().is_none() {
            let sub_process_name = process_data
                .as_ref()
                .and_then(|process_data| process_data.lock().unwrap().get_sub_process_name());
            utilities::timestamp_process_name_subprocess(
                Some(&name),
                process_name,
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// `manager.processDone(...)` from a process thread, posted to the event
    /// dispatch thread.
    #[allow(clippy::too_many_arguments)]
    fn post_process_done(
        &self,
        thread_name: String,
        exit_value: i32,
        process_name: Option<ProcessName>,
        axis_id: AxisID,
        force_next_process: bool,
        end_state: Option<ProcessEndState>,
        status_string: Option<String>,
        failed: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        non_blocking: bool,
    ) {
        let manager = self.manager;
        event_queue::invoke_later(move || {
            manager.process_done(
                Some(&thread_name),
                exit_value,
                process_name,
                Some(axis_id),
                force_next_process,
                end_state,
                status_string.as_deref(),
                failed,
                process_result_display,
                process_series,
                non_blocking,
            );
        });
    }

    /// The common tail of every `startBackgroundProcess` overload that builds
    /// a `BackgroundProcess` from `init`.
    fn start_background_init(
        &'static self,
        init: BackgroundProcessInit,
        command_line: String,
        axis_id: AxisID,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let background_process = BackgroundProcess::get_instance(init);
        self.start_background_process(background_process, &command_line, axis_id, None)
    }

    /// Java final `startBackgroundProcess(List<String>, AxisID,
    /// ProcessResultDisplay, ProcessName, ProcessSeries)` and `(String[], ...)`:
    /// start a managed background process.
    pub fn start_background_process_array_display(
        &'static self,
        command: Vec<String>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let mut init =
            BackgroundProcessInit::new(self.manager, self, axis_id, process_name, process_series);
        // `commandArray.toString()` is the array's identity string in the Java.
        let command_line = format!("[Ljava.lang.String;@{:x}", command.as_ptr() as usize);
        init.command_array = Some(command);
        init.process_result_display = process_result_display;
        self.start_background_init(init, command_line, axis_id)
    }

    /// Java final `startBackgroundProcess(Command, AxisID, ProcessResultDisplay,
    /// ProcessName, ProcessSeries[, boolean allowMultiLineLog])`.
    pub fn start_background_process_command_display(
        &'static self,
        command: Arc<dyn Command + Send + Sync>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        self.start_background_process_command_multi_line(
            command,
            axis_id,
            process_result_display,
            process_name,
            process_series,
            false,
        )
    }

    /// Java final `startBackgroundProcess(Command, AxisID, ProcessResultDisplay,
    /// ProcessName, ProcessSeries, boolean allowMultiLineLog)`.
    pub fn start_background_process_command_multi_line(
        &'static self,
        command: Arc<dyn Command + Send + Sync>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
        allow_multi_line_log: bool,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let command_line = command.get_command_array().unwrap_or_default().join(" ");
        let mut init =
            BackgroundProcessInit::new(self.manager, self, axis_id, process_name, process_series);
        init.command = Some(command);
        init.process_result_display = process_result_display;
        init.allow_multi_line_log = allow_multi_line_log;
        self.start_background_init(init, command_line, axis_id)
    }

    /// Java final `startBackgroundProcess(String[], AxisID, boolean
    /// forceNextProcess, ProcessResultDisplay, ProcessSeries, ProcessName)`.
    pub fn start_background_process_array_force(
        &'static self,
        command_array: Vec<String>,
        axis_id: AxisID,
        force_next_process: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        process_name: Option<ProcessName>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let mut init =
            BackgroundProcessInit::new(self.manager, self, axis_id, process_name, process_series);
        let command_line = command_array.join(" ");
        init.command_array = Some(command_array);
        init.force_next_process = force_next_process;
        init.process_result_display = process_result_display;
        self.start_background_init(init, command_line, axis_id)
    }

    /// Java final `startBackgroundProcess(String[], AxisID, ProcessName,
    /// ProcessSeries)`.
    pub fn start_background_process_array(
        &'static self,
        command_array: Vec<String>,
        axis_id: AxisID,
        process_name: Option<ProcessName>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let mut command_line = String::new();
        for element in &command_array {
            command_line.push_str(&format!("{element} "));
        }
        let mut init =
            BackgroundProcessInit::new(self.manager, self, axis_id, process_name, process_series);
        init.command_array = Some(command_array);
        self.start_background_init(init, command_line, axis_id)
    }

    /// Java final `startDetachedProcess(DetachedCommandDetails, AxisID,
    /// OutfileProcessMonitor, ProcessResultDisplay, ProcessName, ProcessSeries,
    /// boolean, ProcessingMethod, ProcessData)` and the overload that also
    /// takes `subdirName` and `shortCommandName`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_detached_process(
        &'static self,
        detached_command_details: Arc<dyn DetachedCommandDetails + Send + Sync>,
        axis_id: AxisID,
        monitor: Option<Arc<dyn OutfileProcessMonitor>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_name: Option<ProcessName>,
        subdir: Option<(&str, &str)>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let command_line = detached_command_details
            .get_command_line()
            .unwrap_or_default();
        let detached_process = BackgroundProcess::new_detached(
            self.manager,
            detached_command_details,
            self,
            axis_id,
            monitor.clone(),
            process_result_display,
            process_name,
            process_series,
            popup_chunk_warnings,
            processing_method,
            managed_process_data,
        );
        if let Some((subdir_name, short_command_name)) = subdir {
            detached_process.set_subdir_name(Some(subdir_name));
            detached_process.set_short_command_name(short_command_name);
        }
        if let Some(monitor) = &monitor {
            monitor.set_process(detached_process.clone());
        }
        let process_monitor = monitor.map(|monitor| {
            let monitor: Arc<dyn ProcessMonitor> = monitor;
            monitor
        });
        self.start_background_process(detached_process, &command_line, axis_id, process_monitor)
    }

    /// Java final `startBackgroundProcess(CommandDetails, AxisID, ProcessName,
    /// ProcessSeries[, boolean popupChunkWarnings])`,
    /// `(CommandDetails, AxisID, ProcessResultDisplay, ProcessName,
    /// ProcessSeries)`, `(Command, AxisID, ProcessName, ProcessSeries)`,
    /// `(Command, AxisID, ProcessName, ProcessResultDisplay, ProcessSeries)` and
    /// `(Command, AxisID, boolean forceNextProcess, ProcessName,
    /// ProcessSeries)`: the command line is `command.getCommandLine()`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_background_process_command(
        &'static self,
        command: Arc<dyn Command + Send + Sync>,
        is_command_details: bool,
        axis_id: AxisID,
        process_name: Option<ProcessName>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        force_next_process: bool,
        popup_chunk_warnings: bool,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        let command_line = command.get_command_line().unwrap_or_default();
        let mut init =
            BackgroundProcessInit::new(self.manager, self, axis_id, process_name, process_series);
        init.command = Some(command);
        init.is_command_details = is_command_details;
        init.process_result_display = process_result_display;
        init.force_next_process = force_next_process;
        init.popup_chunk_warnings = popup_chunk_warnings;
        self.start_background_init(init, command_line, axis_id)
    }

    /// Java private `startBackgroundProcess(BackgroundProcess, String, AxisID,
    /// ProcessMonitor)`: the one every overload ends in.
    fn start_background_process(
        &'static self,
        background_process: Arc<BackgroundProcess>,
        command_line: &str,
        axis_id: AxisID,
        process_monitor: Option<Arc<dyn ProcessMonitor>>,
    ) -> Result<Arc<BackgroundProcess>, AxisBusyException> {
        background_process.set_working_directory(PathBuf::from(
            self.manager.get_property_user_dir().unwrap_or_default(),
        ));
        background_process.set_debug(etomo_director::ARGUMENTS.lock().unwrap().is_debug());
        self.manager.save_storables(Some(axis_id));
        self.is_axis_busy(axis_id, background_process.get_process_result_display())?;
        background_process.close_output_image_file();
        background_process.start();
        if debug_flag() {
            eprintln!("Started {command_line} (3)");
            eprintln!("  Name: {}", background_process.get_name());
        }
        self.axis_process_data
            .map_axis_thread(Some(background_process.as_process()), axis_id);
        // Start the process monitor thread if a runnable process is provided
        if let Some(process_monitor) = process_monitor {
            let process = Arc::clone(&background_process);
            let axis_process_data = Arc::clone(&self.axis_process_data);
            std::thread::spawn(move || {
                // Wait for the started flag within the backgroundProcess
                while !process.is_started() {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                let monitor: Arc<dyn Monitor> = process_monitor;
                let thread_monitor = Arc::clone(&monitor);
                let process_monitor_thread = std::thread::spawn(move || thread_monitor.run());
                axis_process_data.map_axis_process_monitor(
                    Some(process_monitor_thread),
                    Some(monitor),
                    axis_id,
                );
            });
        }
        Ok(background_process)
    }

    /// Java final `startInteractiveSystemProgram(Command)`.
    pub fn start_interactive_system_program(
        &'static self,
        command: Arc<dyn Command + Send + Sync>,
    ) -> Result<Arc<InteractiveSystemProgram>, SystemProcessException> {
        let axis_id = command.get_axis_id();
        let program = Arc::new(InteractiveSystemProgram::new_command(
            self.manager,
            command,
            self,
            axis_id,
        ));
        program.close_output_image_file();
        program.set_working_directory(Some(PathBuf::from(
            self.manager.get_property_user_dir().unwrap_or_default(),
        )));
        let thread_name = super::com_script_process::next_thread_name();
        self.manager.save_storables(Some(axis_id));
        let thread_program = Arc::clone(&program);
        std::thread::Builder::new()
            .name(thread_name.clone())
            .spawn(move || thread_program.run())
            .map_err(|e| SystemProcessException(e.to_string()))?;
        program.set_name(&thread_name);
        if debug_flag() {
            eprintln!(
                "Started {} (4)",
                program.get_command_line().unwrap_or_default()
            );
            eprintln!("  Name: {thread_name}");
        }
        Ok(program)
    }

    /// Java final `tomosnapshot(AxisID, boolean thumbnail)`.
    pub fn tomosnapshot(&self, axis_id: AxisID, thumbnail: bool) {
        let process = TomosnapshotProcess::new(self.manager, axis_id, thumbnail);
        if let Err(e) = std::thread::Builder::new().spawn(move || process.run()) {
            ui_harness::open_message_dialog_from_process(
                None,
                &e.to_string(),
                "Process Exception",
                Some(axis_id),
            );
        }
    }

    /// Java static `getCommandOutput`.
    pub fn get_command_output(
        command_line: Vec<String>,
        axis_id: AxisID,
        manager: Option<&'static dyn BaseManager>,
    ) -> Option<Vec<String>> {
        let command = SystemProgram::new_array(
            manager,
            manager.and_then(|manager| manager.get_property_user_dir()),
            Some(command_line),
            axis_id,
        );
        command.run();
        command.get_std_output()
    }

    /// Java static final `startSystemProgramThread(String[], AxisID,
    /// BaseManager)`: start an arbitrary command as an unmanaged background
    /// thread.
    pub fn start_system_program_thread(
        command: Vec<String>,
        axis_id: AxisID,
        manager: Option<&'static dyn BaseManager>,
    ) -> Arc<SystemProgram> {
        // Initialize the SystemProgram object
        let sys_program = Arc::new(SystemProgram::new_array(
            manager,
            manager.and_then(|manager| manager.get_property_user_dir()),
            Some(command),
            axis_id,
        ));
        if let Some(manager) = manager {
            sys_program.set_working_directory(Some(PathBuf::from(
                manager.get_property_user_dir().unwrap_or_default(),
            )));
            manager.save_storables(Some(sys_program.get_axis_id()));
        }
        // Start the system program thread
        let thread_program = Arc::clone(&sys_program);
        std::thread::spawn(move || thread_program.run());
        if debug_flag() {
            eprintln!("Started {} (5)", sys_program.get_command_line());
            eprintln!(
                "  working directory: {}",
                manager
                    .and_then(|manager| manager.get_property_user_dir())
                    .unwrap_or_else(|| "null".to_owned())
            );
        }
        sys_program
    }

    /// Java final `msgProcessDone(DetachedProcess, int, boolean)`.
    pub fn msg_process_done_detached(
        &self,
        process: &BackgroundProcess,
        exit_value: i32,
        error_found: bool,
    ) {
        let end_state = process.get_process_end_state();
        if (exit_value != 0 || error_found) && end_state != Some(ProcessEndState::Killed) {
            self.hooks().error_process_detached(self, process);
        } else {
            self.log_process_output(
                &process.get_command_action().unwrap_or_default(),
                process.get_std_output().as_deref(),
                process.get_std_error().as_deref(),
            );
            self.hooks().post_process_detached(self, process);
        }
        self.manager.save_storables(Some(process.get_axis_id()));
        self.axis_process_data
            .clear_thread(process, ClearKind::Other);
        // Inform the manager that this process is complete
        let status_string = if end_state.is_none() || end_state == Some(ProcessEndState::Done) {
            None
        } else {
            process.get_status_string()
        };
        self.post_process_done(
            process.get_name(),
            exit_value,
            process.get_process_name(),
            process.get_axis_id(),
            process.is_force_next_process(),
            process.get_process_end_state(),
            status_string,
            exit_value != 0 || error_found,
            process.get_process_result_display(),
            process.get_process_series(),
            false,
        );
        if process.get_process_series().is_none() {
            let sub_process_name = process
                .get_process_data()
                .and_then(|data| data.lock().unwrap().get_sub_process_name());
            utilities::timestamp_process_name_subprocess(
                Some(&process.get_name()),
                process.get_process_name(),
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// Java final `msgProcessDone(BackgroundProcess, int, boolean, boolean)`:
    /// a background process has finished execution.
    pub fn msg_process_done_background(
        &self,
        process: &BackgroundProcess,
        exit_value: i32,
        error_found: bool,
        popup_chunk_warnings: bool,
    ) {
        let end_state = process.get_process_end_state();
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("BaseProcessManager:msgProcessDone:endState:{end_state:?}");
        }
        if (exit_value != 0 || error_found) && end_state != Some(ProcessEndState::Killed) {
            self.hooks().error_process_background(self, process);
        } else {
            self.log_process_output(
                &process.get_command_action().unwrap_or_default(),
                process.get_std_output().as_deref(),
                process.get_std_error().as_deref(),
            );
            self.hooks().post_process_background(self, process);
            if let Some(messages) = process.get_process_messages()
                && popup_chunk_warnings
                && messages.size(MessageType::Warning) > 0
            {
                ui_harness::post_warning_message_dialog(
                    Some(self.manager),
                    messages,
                    "Process Warnings".to_owned(),
                    process.get_axis_id(),
                );
            }
        }
        self.manager.save_storables(Some(process.get_axis_id()));
        self.axis_process_data
            .clear_thread(process, ClearKind::Other);
        // Inform the manager that this process is complete
        let status_string = if end_state.is_none() || end_state == Some(ProcessEndState::Done) {
            None
        } else {
            process.get_status_string()
        };
        self.post_process_done(
            process.get_name(),
            exit_value,
            process.get_process_name(),
            process.get_axis_id(),
            process.is_force_next_process(),
            process.get_process_end_state(),
            status_string,
            exit_value != 0 || error_found,
            process.get_process_result_display(),
            process.get_process_series(),
            false,
        );
        if process.get_process_series().is_none() {
            let sub_process_name = process
                .get_process_data()
                .and_then(|data| data.lock().unwrap().get_sub_process_name());
            utilities::timestamp_process_name_subprocess(
                Some(&process.get_name()),
                process.get_process_name(),
                sub_process_name.as_deref(),
                Some("done"),
            );
        }
    }

    /// Java final `msgInteractiveSystemProgramDone`.
    pub fn msg_interactive_system_program_done(
        &self,
        program: &InteractiveSystemProgram,
        _exit_value: i32,
    ) {
        self.log_process_output(
            &program.get_command_action().unwrap_or_default(),
            program.get_std_output().as_deref(),
            program.get_std_error().as_deref(),
        );
        self.hooks().post_process_interactive(self, program);
        self.manager.save_storables(Some(program.get_axis_id()));
        utilities::timestamp_command_status(program.get_command_name().as_deref(), Some("done"));
    }

    /// Java `postProcess(BackgroundProcess)`, the base class's body.
    pub fn post_process_background_base(&self, process: &BackgroundProcess) {
        let Some(command_name) = process.get_command_name() else {
            return;
        };
        if ProcessName::TOMOSNAPSHOT.equals(&command_name) {
            let std_output = process.get_std_output();
            utilities::find_message_and_open_dialog(
                Some(self.manager),
                Some(process.get_axis_id()),
                std_output.as_deref(),
                tomosnapshot_param::OUTPUT_LINE,
                "Tomosnapshot Complete",
            );
        }
    }

    /// Java `postProcess(DetachedProcess)`, the base class's body.
    pub fn post_process_detached_base(&self, process: &BackgroundProcess) {
        let end_state = process.get_process_end_state();
        let Some(command) = process.get_command() else {
            return;
        };
        if command.get_command_name().as_deref() == Some(&ProcessName::PROCESSCHUNKS.to_string())
            && !process.is_pausing()
            && end_state != Some(ProcessEndState::Killed)
        {
            let manager = self.manager;
            let axis_id = process.get_axis_id();
            event_queue::invoke_later(move || manager.reset_current_processchunks(Some(axis_id)));
        }
    }
}

/// `EtomoDirector.INSTANCE.getPythonScriptPath()`.
pub fn python_script_path() -> String {
    etomo_director::INSTANCE
        .get_python_script_path()
        .unwrap_or_default()
}

/// Java `etomo.process.AxisBusyException`.
#[derive(Clone, Debug)]
pub struct AxisBusyException(pub String);

impl std::fmt::Display for AxisBusyException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Java `etomo.process.SystemProcessException`.
#[derive(Clone, Debug)]
pub struct SystemProcessException(pub String);

impl std::fmt::Display for SystemProcessException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
