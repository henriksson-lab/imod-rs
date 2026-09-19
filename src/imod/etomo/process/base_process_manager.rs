//! `IMOD/Etomo/src/etomo/process/BaseProcessManager.java`.
//!
//! This is deliberately a source-shaped process boundary.  `BaseProcessManager`
//! coordinates the Java process, monitor, comscript, log-file, and Swing hierarchies.
//! Those units are not translated yet, so their declared reference types use
//! `Option<Infallible>`: exactly Java's currently possible `null` value, without a
//! fabricated process implementation.  Rust cannot overload methods; overloads retain
//! the Java name and gain a stable suffix naming the parameter family.
#![allow(dead_code)]

use std::convert::Infallible;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::path::Path;
use std::process::Command as OsCommand;
use std::sync::Mutex;

use crate::imod::etomo::process::background_process::BackgroundProcess;
use crate::imod::etomo::process::system_program::ProcessCommand;
use crate::imod::etomo::process::tomosetexts_output::TomosetextsOutput;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java abstract `BaseProcessManager`.
///
/// `manager`, `etomoDirector`, `uiHarness`, and `axisProcessData` have source
/// declarations whose concrete translations are still unavailable at this layer.
/// They intentionally remain null-equivalent rather than becoming invented stand-ins.
pub struct BaseProcessManager {
    /// Java private static final `DEBUG`; dynamic director arguments are not yet a
    /// dependency of this module, so this preserves its initial false state.
    debug: Mutex<bool>,
    /// Java protected final `manager`.
    manager: Option<Infallible>,
    /// Java final `uiHarness`.
    ui_harness: Option<Infallible>,
    /// Java private final `etomoDirector`.
    etomo_director: Option<Infallible>,
    /// Java final `axisProcessData`.
    axis_process_data: Option<Infallible>,
}

impl BaseProcessManager {
    /// Java protected `BaseProcessManager(BaseManager)`.
    pub fn new(manager: Option<Infallible>) -> Self {
        Self {
            debug: Mutex::new(false),
            manager,
            ui_harness: None,
            etomo_director: None,
            axis_process_data: None,
        }
    }

    /// Runnable local form of Java's String-array `startBackgroundProcess`.
    /// Source UI/monitor references are optional decorations; local command execution
    /// itself belongs to `BackgroundProcess` and must not be dropped when absent.
    pub fn start_background_process_local(
        &self,
        command: ProcessCommand,
        axis_id: AxisID,
    ) -> Result<BackgroundProcess, String> {
        let mut process = BackgroundProcess::new(command, axis_id);
        process.start()?;
        Ok(process)
    }

    /// Java `dumpState`.
    pub fn dump_state(&self) {
        if *self.debug.lock().unwrap() {
            eprintln!("[debug:true]");
        }
    }
    /// Java package-private `setDebug`.
    pub fn set_debug(&self, debug: bool) {
        *self.debug.lock().unwrap() = debug;
    }
    /// Java static `isPython3`.
    pub fn is_python3() -> bool {
        OsCommand::new("python")
            .args(["-c", "import sys; print(sys.version_info > (2,))"])
            .output()
            .ok()
            .is_some_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "True")
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!("BaseProcessManager[{}]", self.param_string())
    }
    /// Java final `paramString`.
    pub fn param_string(&self) -> String {
        format!(
            "axisProcessData:{:?},uiHarness:{:?}",
            self.axis_process_data, self.ui_harness
        )
    }

    /// Java hook `errorProcess(BackgroundProcess)`.
    pub fn error_process_background(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java hook `errorProcess(ComScriptProcess)`.
    pub fn error_process_com_script(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java hook `errorProcess(ReconnectProcess)`.
    pub fn error_process_reconnect(&self, script: Option<Infallible>) {
        let _ = script;
    }
    /// Java hook `postProcess(ComScriptProcess)`.
    pub fn post_process_com_script(&self, script: Option<Infallible>) {
        let _ = script;
    }
    /// Java hook `postProcess(InteractiveSystemProgram)`.
    pub fn post_process_interactive_system_program(&self, program: Option<Infallible>) {
        let _ = program;
    }
    /// Java hook `postProcess(ReconnectProcess)`.
    pub fn post_process_reconnect(&self, script: Option<Infallible>) {
        let _ = script;
    }
    /// Java final `writeLogFile`.
    pub fn write_log_file(
        &self,
        process: Option<Infallible>,
        axis_id: AxisID,
        file_name: Option<&str>,
    ) {
        let _ = (process, axis_id, file_name);
    }

    /// Runnable form of Java `writeLogFile(BackgroundProcess, AxisID, String)`.
    ///
    /// The source copies only the process standard-output records into the
    /// requested log, one line at a time.  `BackgroundProcess` retains those
    /// records after polling, so this has the same post-process lifetime as
    /// the Java implementation without requiring the untranslated manager
    /// and emergency-monitor ownership graph.
    pub fn write_log_file_local(
        &self,
        process: &BackgroundProcess,
        file_name: &Path,
    ) -> std::io::Result<()> {
        if let Some(parent) = file_name.parent() {
            fs::create_dir_all(parent)?;
        }
        use std::io::Write;
        let mut log = File::create(file_name)?;
        for line in process.get_std_output() {
            writeln!(log, "{line}")?;
        }
        Ok(())
    }
    /// Java final `startLoad`.
    pub fn start_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        let _ = (param, monitor);
    }
    /// Java final `endLoad`.
    pub fn end_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        let _ = (param, monitor);
    }
    /// Java final `stopLoad`.
    pub fn stop_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        let _ = (param, monitor);
    }
    /// Java `xfmodel`.
    pub fn xfmodel(
        &self,
        param: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, result, series);
        None
    }
    /// Java public `reconnectProcesschunks`.
    pub fn reconnect_processchunks(
        &self,
        axis_id: AxisID,
        data: Option<Infallible>,
        result: Option<Infallible>,
        series: Option<Infallible>,
        multi_line: bool,
        popup: bool,
        messages: Option<Infallible>,
    ) -> bool {
        let _ = (axis_id, data, result, series, multi_line, popup, messages);
        false
    }
    /// Java final `reconnectProcesschunks` monitor overload.
    pub fn reconnect_processchunks_monitor(
        &self,
        axis_id: AxisID,
        data: Option<Infallible>,
        result: Option<Infallible>,
        series: Option<Infallible>,
        monitor: Option<Infallible>,
        popup: bool,
        reconnect: bool,
    ) -> bool {
        let _ = (axis_id, data, result, series, monitor, popup, reconnect);
        false
    }
    /// Java final `tomodataplots`.
    pub fn tomodataplots(&self, param: Option<Infallible>, axis_id: AxisID) {
        let _ = (param, axis_id);
    }
    /// Java `midas`.
    pub fn midas(&self, param: Option<Infallible>) -> Option<String> {
        let _ = param;
        None
    }
    /// Java public `processchunks`.
    pub fn processchunks(
        &self,
        axis_id: AxisID,
        param: Option<Infallible>,
        display: Option<Infallible>,
        result: Option<Infallible>,
        series: Option<Infallible>,
        popup: bool,
        method: Option<Infallible>,
        multi_line: bool,
        run_type: Option<Infallible>,
        data: Option<Infallible>,
        messages: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            axis_id, param, display, result, series, popup, method, multi_line, run_type, data,
            messages,
        );
        None
    }
    /// Java final `processchunks` monitor overload.
    pub fn processchunks_monitor(
        &self,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        param: Option<Infallible>,
        display: Option<Infallible>,
        result: Option<Infallible>,
        series: Option<Infallible>,
        popup: bool,
        method: Option<Infallible>,
        data: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            monitor, axis_id, param, display, result, series, popup, method, data,
        );
        None
    }

    /// Java final `createNewFile`; directory and file semantics do not cross an
    /// unavailable Java boundary, so they are retained directly.
    pub fn create_new_file(&self, absolute_path: &Path) -> std::io::Result<()> {
        if absolute_path.exists() {
            return Ok(());
        }
        if let Some(parent) = absolute_path.parent() {
            fs::create_dir_all(parent)?;
        }
        File::create(absolute_path).map(|_| ())
    }
    /// Java static `touch`; `b3dtouch` and its director-owned script path are not
    /// available here, so this only retains the file-existence part of its contract.
    pub fn touch(absolute_path: &Path, manager: Option<Infallible>) -> std::io::Result<()> {
        let _ = manager;
        if let Some(parent) = absolute_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !absolute_path.exists() {
            File::create(absolute_path)?;
        }
        Ok(())
    }
    /// Java static `tomosetexts`.
    pub fn tomosetexts(
        manager: Option<Infallible>,
        axis_id: AxisID,
        dir: &Path,
    ) -> Option<TomosetextsOutput> {
        let _ = (manager, axis_id, dir);
        None
    }

    /// Runnable core of Java static `tomosetexts(BaseManager, AxisID, File)`.
    /// The caller supplies the configured Python interpreter and IMOD script
    /// path, while this source unit owns directory validation, child lifetime,
    /// and conversion of stdout to `TomosetextsOutput`.
    pub fn tomosetexts_local(
        dir: &Path,
        python: &OsStr,
        script: &Path,
    ) -> Option<TomosetextsOutput> {
        if !dir.is_dir() || fs::read_dir(dir).is_err() {
            return None;
        }
        let command = ProcessCommand::new(python)
            .args([script.as_os_str(), dir.as_os_str()])
            .current_dir(dir);
        let mut program = super::system_program::SystemProgram::spawn(&command).ok()?;
        let (_, lines) = program.wait_and_drain().ok()?;
        let stdout: Vec<String> = lines
            .into_iter()
            .filter(|line| line.stream == super::system_program::ProcessStream::Stdout)
            .map(|line| line.line)
            .collect();
        Some(TomosetextsOutput::new(Some(&stdout)))
    }
    /// Java `imodqtassistQuery`.
    pub fn imodqtassist_query(&self, axis_id: AxisID) -> Option<Vec<String>> {
        let _ = axis_id;
        Self::imodqtassist_query_local(OsStr::new("imodqtassist"))
    }

    /// Runnable query mode of `ImodqtassistProcess.getQueryInstance()`.
    /// `imodqtassist` is resolved through PATH exactly as the Java command
    /// list does; callers/tests can provide a configured absolute program.
    pub fn imodqtassist_query_local(program: &OsStr) -> Option<Vec<String>> {
        let command = ProcessCommand::new(program).args(["-t"]);
        let mut child = super::system_program::SystemProgram::spawn(&command).ok()?;
        let (_, lines) = child.wait_and_drain().ok()?;
        Some(
            lines
                .into_iter()
                .filter(|line| line.stream == super::system_program::ProcessStream::Stdout)
                .map(|line| line.line)
                .collect(),
        )
    }

    // The following overload families require process/comscript classes.  Each body is
    // intentionally a hard null boundary until the named Java declared type is ported.
    /// Java `startComScript(String, ProcessMonitor, AxisID, ProcessResultDisplay, CommandDetails, ProcessSeries)`.
    pub fn start_com_script_command_details(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        details: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, result, details, series);
        None
    }
    /// Java `startComScript(String, ProcessMonitor, AxisID, ProcessResultDisplay, Command, ProcessSeries)`.
    pub fn start_com_script_command(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        parameter: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, result, parameter, series);
        None
    }
    /// Java `startOutfileComScript`.
    pub fn start_outfile_com_script(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        parameter: Option<Infallible>,
        file_type: Option<Infallible>,
        process_name: Option<Infallible>,
        reconnect: bool,
        data: Option<Infallible>,
        indeterminate: bool,
    ) -> Option<Infallible> {
        let _ = (
            command,
            monitor,
            axis_id,
            parameter,
            file_type,
            process_name,
            reconnect,
            data,
            indeterminate,
        );
        None
    }
    /// Java `startComScript(String, ProcessMonitor, AxisID, ProcessResultDisplay, ProcessSeries, boolean)`.
    pub fn start_com_script_resumable(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
        resumable: bool,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, result, series, resumable);
        None
    }
    /// Java `startComScript(String, AxisID, ProcessSeries, FileType)`.
    pub fn start_com_script_file_type(
        &self,
        command: Option<&str>,
        axis_id: AxisID,
        series: Option<Infallible>,
        file_type: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, series, file_type);
        None
    }
    /// Java `startComScript(String, ProcessMonitor, AxisID, ProcessResultDisplay, Command, ProcessSeries, FileType)`.
    pub fn start_com_script_command_file_type(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        parameter: Option<Infallible>,
        series: Option<Infallible>,
        file_type: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (
            command, monitor, axis_id, result, parameter, series, file_type,
        );
        None
    }
    /// Java `startNonBlockingComScript(String, AxisID, ProcessResultDisplay)`.
    pub fn start_non_blocking_com_script(
        &self,
        command: Option<&str>,
        axis_id: AxisID,
        result: Option<Infallible>,
    ) {
        let _ = (command, axis_id, result);
    }
    /// Java `startComScript(String, ProcessMonitor, AxisID, ProcessSeries, boolean)`.
    pub fn start_com_script_monitor_resumable(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        series: Option<Infallible>,
        resumable: bool,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, series, resumable);
        None
    }
    /// Java `startComScript(CommandDetails, ProcessMonitor, AxisID, ProcessSeries)`.
    pub fn start_com_script_details(
        &self,
        details: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (details, monitor, axis_id, series);
        None
    }
    /// Java protected `startComScript(CommandDetails,...,ProcessingMethod)`.
    pub fn start_com_script_details_method(
        &self,
        details: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
        method: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (details, monitor, axis_id, result, series, method);
        None
    }
    /// Java protected `startComScript(CommandDetails, ProcessMonitor, AxisID,
    /// ProcessResultDisplay, ProcessSeries)`.
    pub fn start_com_script_details_display(
        &self,
        details: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (details, monitor, axis_id, result, series);
        None
    }
    /// Java protected `startComScript(Command,...)`.
    pub fn start_com_script_parameter(
        &self,
        command: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, result, series);
        None
    }
    /// Java final `startComScript(Command,...,ProcessingMethod)`.
    pub fn start_com_script_parameter_method(
        &self,
        command: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        series: Option<Infallible>,
        method: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, result, series, method);
        None
    }
    /// Java final `startBackgroundComScript`.
    pub fn start_background_com_script(
        &self,
        comscript: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        state: Option<Infallible>,
        watched: Option<&str>,
        series: Option<Infallible>,
        resumable: bool,
    ) -> Option<Infallible> {
        let _ = (
            comscript, monitor, axis_id, state, watched, series, resumable,
        );
        None
    }
    /// Java final `startComScript` watched-file overload.
    pub fn start_com_script_watched_file(
        &self,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        watched: Option<&str>,
        series: Option<Infallible>,
        resumable: bool,
    ) -> Option<Infallible> {
        let _ = (command, monitor, axis_id, watched, series, resumable);
        None
    }
    /// Java final `startMonitor`.
    pub fn start_monitor(
        &self,
        monitor: Option<Infallible>,
        axis_id: AxisID,
        reconnect: bool,
    ) -> Option<Infallible> {
        let _ = (monitor, axis_id, reconnect);
        None
    }
    /// Java final `startComScript(ComScriptProcess,...)`.
    pub fn start_com_script_process(
        &self,
        process: Option<Infallible>,
        command: Option<&str>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
    ) -> Option<Infallible> {
        let _ = (process, command, monitor, axis_id);
        None
    }
    /// Java final `startComScriptMonitor`.
    pub fn start_com_script_monitor(
        &self,
        process: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
    ) {
        let _ = (process, monitor, axis_id);
    }
    /// Java final `startNonBlockingComScript(ComScriptProcess,...)`.
    pub fn start_non_blocking_com_script_process(
        &self,
        process: Option<Infallible>,
        command: Option<&str>,
        axis_id: AxisID,
    ) {
        let _ = (process, command, axis_id);
    }

    /// Java final `inUse`.
    pub fn in_use(&self, axis_id: AxisID, result: Option<Infallible>, popup_error: bool) -> bool {
        let _ = (axis_id, result, popup_error);
        false
    }
    /// Java private `isDualAxis`.
    pub fn is_dual_axis(&self) -> bool {
        false
    }
    /// Java final `isAxisBusy`.
    pub fn is_axis_busy(&self, axis_id: AxisID, result: Option<Infallible>) -> Result<(), String> {
        let _ = (axis_id, result);
        Ok(())
    }
    /// Java final `unblockAxis`.
    pub fn unblock_axis(&self, axis_id: AxisID) {
        let _ = axis_id;
    }
    /// Java private `saveProcessData`.
    pub fn save_process_data(&self, axis_id: AxisID, data: Option<Infallible>) {
        let _ = (axis_id, data);
    }
    /// Java final `mapAxisThread`.
    pub fn map_axis_thread(&self, thread: Option<Infallible>, axis_id: AxisID) {
        let _ = (thread, axis_id);
    }
    /// Java private `mapAxisProcessMonitor`.
    pub fn map_axis_process_monitor(&self, monitor: Option<Infallible>, axis_id: AxisID) {
        let _ = (monitor, axis_id);
    }
    /// Java final `getProcessData`.
    pub fn get_process_data(&self, axis_id: AxisID) -> Option<Infallible> {
        let _ = axis_id;
        None
    }
    /// Java final `pause`.
    pub fn pause(&self, axis_id: AxisID) -> bool {
        let _ = axis_id;
        false
    }
    /// Java final `kill(AxisID)`.
    pub fn kill(&self, axis_id: AxisID) {
        let _ = axis_id;
    }
    /// Java final `signalKill`.
    pub fn signal_kill(&self, thread: Option<Infallible>, axis_id: AxisID) {
        let _ = (thread, axis_id);
    }
    /// Java private `kill(String,AxisID)`.
    pub fn kill_process_id(&self, process_id: Option<&str>, axis_id: AxisID) {
        let _ = (process_id, axis_id);
    }
    /// Java private `getChildProcessList`.
    pub fn get_child_process_list(
        &self,
        process_id: Option<&str>,
        axis_id: AxisID,
    ) -> Option<Vec<String>> {
        let _ = (process_id, axis_id);
        None
    }
    /// Java private `logProcessOutput`.
    pub fn log_process_output(
        &self,
        command_action: Option<&str>,
        stdout: Option<&[String]>,
        stderr: Option<&[String]>,
    ) {
        let _ = (command_action, stdout, stderr);
    }

    /// Java `msgComScriptDone(OutfileComScriptProcess,...)`.
    pub fn msg_com_script_done_outfile(
        &self,
        process: Option<Infallible>,
        exit_value: i32,
        non_blocking: bool,
    ) {
        let _ = (process, exit_value, non_blocking);
    }
    /// Java `msgComScriptDone(AxisID,ComScriptProcess,...)`.
    pub fn msg_com_script_done(
        &self,
        axis_id: AxisID,
        script: Option<Infallible>,
        exit_value: i32,
        non_blocking: bool,
    ) {
        let _ = (axis_id, script, exit_value, non_blocking);
    }
    /// Java `msgReconnectDone(AxisID,ReconnectProcess,...)`.
    pub fn msg_reconnect_done(
        &self,
        axis_id: AxisID,
        script: Option<Infallible>,
        exit_value: i32,
        popup: bool,
    ) {
        let _ = (axis_id, script, exit_value, popup);
    }
    /// Java `msgReconnectDone(LoggedReconnectProcess,...)`.
    pub fn msg_reconnect_done_logged(
        &self,
        script: Option<Infallible>,
        exit_value: i32,
        popup: bool,
        no_popup: bool,
    ) {
        let _ = (script, exit_value, popup, no_popup);
    }

    /// Java `startBackgroundProcess(List<String>,...)`.
    pub fn start_background_process_list(
        &self,
        command: Option<&[String]>,
        axis_id: AxisID,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, result, name, series);
        None
    }
    /// Java `startBackgroundProcess(String[],...)`.
    pub fn start_background_process_array(
        &self,
        command: Option<&[String]>,
        axis_id: AxisID,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, result, name, series);
        None
    }
    /// Java `startBackgroundProcess(Command,...)`.
    pub fn start_background_process_command(
        &self,
        command: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, result, name, series);
        None
    }
    /// Java Command overload with `allowMultiLineLog`.
    pub fn start_background_process_command_multi_line(
        &self,
        command: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
        multi_line: bool,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, result, name, series, multi_line);
        None
    }
    /// Java String-array overload with `forceNextProcess`.
    pub fn start_background_process_array_force(
        &self,
        command: Option<&[String]>,
        axis_id: AxisID,
        force: bool,
        result: Option<Infallible>,
        series: Option<Infallible>,
        name: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, force, result, series, name);
        None
    }
    /// Java String-array overload without display.
    pub fn start_background_process_array_named(
        &self,
        command: Option<&[String]>,
        axis_id: AxisID,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, name, series);
        None
    }
    /// Java `startDetachedProcess`.
    pub fn start_detached_process(
        &self,
        details: Option<Infallible>,
        axis_id: AxisID,
        monitor: Option<Infallible>,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
        popup: bool,
        method: Option<Infallible>,
        data: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (
            details, axis_id, monitor, result, name, series, popup, method, data,
        );
        None
    }
    /// Java detached-process subdirectory overload.
    pub fn start_detached_process_subdir(
        &self,
        details: Option<Infallible>,
        axis_id: AxisID,
        monitor: Option<Infallible>,
        result: Option<Infallible>,
        name: Option<Infallible>,
        subdir: Option<&str>,
        short_name: Option<&str>,
        series: Option<Infallible>,
        popup: bool,
        method: Option<Infallible>,
        data: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (
            details, axis_id, monitor, result, name, subdir, short_name, series, popup, method,
            data,
        );
        None
    }
    /// Java `startBackgroundProcess(CommandDetails, ProcessName, ProcessSeries)`.
    pub fn start_background_process_details(
        &self,
        details: Option<Infallible>,
        axis_id: AxisID,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (details, axis_id, name, series);
        None
    }
    /// Java CommandDetails popup-warnings overload.
    pub fn start_background_process_details_popup(
        &self,
        details: Option<Infallible>,
        axis_id: AxisID,
        name: Option<Infallible>,
        series: Option<Infallible>,
        popup: bool,
    ) -> Option<Infallible> {
        let _ = (details, axis_id, name, series, popup);
        None
    }
    /// Java CommandDetails display overload.
    pub fn start_background_process_details_display(
        &self,
        details: Option<Infallible>,
        axis_id: AxisID,
        result: Option<Infallible>,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (details, axis_id, result, name, series);
        None
    }
    /// Java Command ProcessName overload.
    pub fn start_background_process_command_named(
        &self,
        command: Option<Infallible>,
        axis_id: AxisID,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, name, series);
        None
    }
    /// Java Command display overload.
    pub fn start_background_process_command_display(
        &self,
        command: Option<Infallible>,
        axis_id: AxisID,
        name: Option<Infallible>,
        result: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, name, result, series);
        None
    }
    /// Java Command force-next overload.
    pub fn start_background_process_command_force(
        &self,
        command: Option<Infallible>,
        axis_id: AxisID,
        force: bool,
        name: Option<Infallible>,
        series: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, force, name, series);
        None
    }
    /// Java private `startBackgroundProcess(BackgroundProcess,...)`.
    pub fn start_background_process(
        &self,
        process: Option<Infallible>,
        command_line: Option<&str>,
        axis_id: AxisID,
        monitor: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (process, command_line, axis_id, monitor);
        None
    }
    /// Java private `startBackgroundProcessMonitor`.
    pub fn start_background_process_monitor(
        &self,
        process: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
    ) {
        let _ = (process, monitor, axis_id);
    }
    /// Java `startInteractiveSystemProgram`.
    pub fn start_interactive_system_program(
        &self,
        command: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = command;
        None
    }
    /// Java final `tomosnapshot`.
    pub fn tomosnapshot(&self, axis_id: AxisID, thumbnail: bool) {
        let _ = (axis_id, thumbnail);
    }
    /// Java static `getCommandOutput`.
    pub fn get_command_output(
        command_line: Option<&[String]>,
        axis_id: AxisID,
        manager: Option<Infallible>,
    ) -> Option<Vec<String>> {
        let _ = (command_line, axis_id, manager);
        None
    }
    /// Java static `startSystemProgramThread(String[],...)`.
    pub fn start_system_program_thread(
        command: Option<&[String]>,
        axis_id: AxisID,
        manager: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (command, axis_id, manager);
        None
    }
    /// Java private static `startSystemProgramThread(SystemProgram,...)`.
    pub fn start_system_program_thread_program(
        program: Option<Infallible>,
        manager: Option<Infallible>,
    ) {
        let _ = (program, manager);
    }
    /// Java `msgProcessDone(DetachedProcess,...)`.
    pub fn msg_process_done_detached(
        &self,
        process: Option<Infallible>,
        exit_value: i32,
        error_found: bool,
    ) {
        let _ = (process, exit_value, error_found);
    }
    /// Java `msgProcessDone(BackgroundProcess,...)`.
    pub fn msg_process_done_background(
        &self,
        process: Option<Infallible>,
        exit_value: i32,
        error_found: bool,
        popup: bool,
    ) {
        let _ = (process, exit_value, error_found, popup);
    }
    /// Java `msgInteractiveSystemProgramDone`.
    pub fn msg_interactive_system_program_done(
        &self,
        program: Option<Infallible>,
        exit_value: i32,
    ) {
        let _ = (program, exit_value);
    }
    /// Java hook `postProcess(BackgroundProcess)`.
    pub fn post_process_background(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java hook `errorProcess(DetachedProcess)`.
    pub fn error_process_detached(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java hook `postProcess(DetachedProcess)`.
    pub fn post_process_detached(&self, process: Option<Infallible>) {
        let _ = process;
    }
}

pub struct ComScriptMonitorRunnable;
impl ComScriptMonitorRunnable {
    #[allow(non_snake_case)]
    pub fn run(
        manager: &BaseProcessManager,
        process: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
    ) {
        let _ = manager.start_com_script_monitor(process, monitor, axis_id);
    }
}
pub struct BackgroundProcessMonitorRunnable;
impl BackgroundProcessMonitorRunnable {
    #[allow(non_snake_case)]
    pub fn run(
        manager: &BaseProcessManager,
        process: Option<Infallible>,
        monitor: Option<Infallible>,
        axis_id: AxisID,
    ) {
        manager.start_background_process_monitor(process, monitor, axis_id);
    }
}

#[cfg(test)]
mod tests {
    use super::BaseProcessManager;
    use crate::imod::etomo::process::system_program::ProcessCommand;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
    #[test]
    fn create_new_file_creates_parent_and_preserves_existing_file() {
        let root =
            std::env::temp_dir().join(format!("imod-rs-base-process-{}", std::process::id()));
        let file = root.join("nested/file");
        let manager = BaseProcessManager::new(None);
        manager.create_new_file(&file).unwrap();
        manager.create_new_file(&file).unwrap();
        assert!(file.is_file());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn local_background_entry_point_starts_a_real_child() {
        let manager = BaseProcessManager::new(None);
        let mut process = manager
            .start_background_process_local(
                ProcessCommand::new("sh").args(["-c", "exit 0"]),
                AxisID::First,
            )
            .unwrap();
        while process.poll().unwrap().is_none() {
            std::thread::yield_now();
        }
        assert_eq!(process.end_state(), Some(ProcessEndState::Done));
    }

    #[cfg(unix)]
    #[test]
    fn local_log_writer_keeps_stdout_and_excludes_stderr() {
        let manager = BaseProcessManager::new(None);
        let mut process = manager
            .start_background_process_local(
                ProcessCommand::new("sh")
                    .args(["-c", "printf first; printf bad >&2; printf '\\nsecond'"]),
                AxisID::First,
            )
            .unwrap();
        while process.poll().unwrap().is_none() {
            std::thread::yield_now();
        }
        let path = std::env::temp_dir().join(format!(
            "imod-rs-base-process-log-{}-{}.log",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        manager.write_log_file_local(&process, &path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first\nsecond\n");
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn tomosetexts_runner_parses_the_first_stdout_line() {
        let root = std::env::temp_dir().join(format!(
            "imod-rs-tomosetexts-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("b3dtomosetexts-test.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf 'MRC ignored-second-line\\nextra\\n'\n",
        )
        .unwrap();
        let output =
            BaseProcessManager::tomosetexts_local(&root, std::ffi::OsStr::new("sh"), &script)
                .unwrap();
        assert_eq!(
            output.get_image_filename_style(),
            Some(crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle::Mrc)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn imodqtassist_query_runs_the_source_thread_query_switch() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "imod-rs-imodqtassist-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let program = root.join("imodqtassist-test");
        std::fs::write(
            &program,
            "#!/bin/sh\nprintf 'argument=%s\\n' \"$1\"\nprintf stderr >&2\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&program).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&program, permissions).unwrap();
        assert_eq!(
            BaseProcessManager::imodqtassist_query_local(program.as_os_str()),
            Some(vec!["argument=-t".to_owned()])
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
