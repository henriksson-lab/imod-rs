//! `IMOD/Etomo/src/etomo/process/BackgroundComScriptProcess.java`.
//!
//! Provides a threadable class to execute IMOD com scripts in the background
//! (combine.com): the script is written to `<root>.py` and started detached
//! through the `startprocess` script, and a `DetachedProcessMonitor` follows
//! it.
//!
//! **Shape.**  The Java class extends `ComScriptProcess` and overrides
//! `closeOutputImageFile`, `isComScriptBusy`, `renameFiles`, `execPython`,
//! `notifyKilled`, `parse` and `getShellProcessID`.  As with
//! `OutfileComScriptProcess` (see `com_script_process.rs`), the process object
//! is a `ComScriptProcess`; this struct is the subclass state it carries (its
//! `background` field), and each override here takes the process as
//! `process`, the Java `this`.  [`BackgroundComScriptProcess::new`] is the
//! Java constructor and returns the process.
//!
//! **Our command-file runner.**  `execPython` has `startprocess` run
//! `python -u <root>.py`, the script `vmstopy` wrote.  When this process is
//! our `imod` binary, the detached command is our runner instead,
//! `imod runcom -P <comscript> <log>` (owner rule, `CLAUDE.md`: com files run
//! in our in-process runner), which converts the command file with the same
//! `vmstopy` translation; the `.py` file is still written, as the Java writes
//! it.  `-P` makes the runner print `Runcom PID: <pid>` where the `vmstopy`
//! script printed `Python PID: <pid>`, so [`parse_pid_string`] also accepts
//! that line, as `ParsePID` does.

use super::com_script_process::{ComScriptProcess, ComScriptProcessInit, ParseError};
use super::monitor::{DetachedProcessMonitor, ProcessMonitor};
use super::parse_background_pid::parse_background_pid;
use super::process_interface::{ProcessInterface, ProcessSeriesRef, SystemProcessInterface};
use super::system_program::{self, BackgroundWait, SystemProgram};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::comscript_state::ComscriptState;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::storage::file_location;
use crate::imod::etomo::storage::log_file::{LockException, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::util::utilities;
use std::io::BufRead;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Java `BackgroundComScriptProcess extends ComScriptProcess`: the subclass's
/// own fields.
pub struct BackgroundComScriptProcess {
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `comscriptState`.
    comscript_state: Option<Arc<dyn ComscriptState + Send + Sync>>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `processID` (a `StringBuffer` shared with
    /// `ParseBackgroundPID`).
    process_id: Arc<Mutex<String>>,
    /// The constructor's `monitor`, as `getDetachedMonitor()` returns it
    /// (`(DetachedProcessMonitor) super.getMonitor()`).
    detached_monitor: Arc<dyn DetachedProcessMonitor>,
}

/// `BackgroundSystemProgram`'s view of the `DetachedProcessMonitor`.
struct DetachedMonitorWait(Arc<dyn DetachedProcessMonitor>);

impl BackgroundWait for DetachedMonitorWait {
    fn is_process_running(&self) -> bool {
        self.0.is_process_running()
    }
    fn is_process_end_state_done(&self) -> bool {
        self.0.get_process_end_state() == Some(ProcessEndState::Done)
    }
}

impl BackgroundComScriptProcess {
    /// Java `BackgroundComScriptProcess(BaseManager, String,
    /// BaseProcessManager, AxisID, String, DetachedProcessMonitor,
    /// ComscriptState, ProcessSeries, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manager: &'static dyn BaseManager,
        com_script: String,
        process_manager: &'static BaseProcessManager,
        axis_id: AxisID,
        watched_file_name: Option<String>,
        monitor: Arc<dyn DetachedProcessMonitor>,
        comscript_state: Option<Arc<dyn ComscriptState + Send + Sync>>,
        process_series: Option<ProcessSeriesRef>,
        resumable: bool,
    ) -> Arc<ComScriptProcess> {
        // super(manager, comScript, processManager, axisID, watchedFileName,
        // monitor, processSeries, resumable)
        let process = ComScriptProcess::new(ComScriptProcessInit {
            manager,
            com_script,
            process_manager,
            axis_id,
            watched_file_name,
            process_monitor: Some(monitor.clone() as Arc<dyn ProcessMonitor>),
            process_result_display: None,
            process_series,
            command: None,
            is_command_details: false,
            resumable: Some(resumable),
            file_type: None,
            processing_method: None,
        });
        process.set_background(BackgroundComScriptProcess {
            axis_id,
            comscript_state,
            manager,
            process_id: Arc::new(Mutex::new(String::new())),
            detached_monitor: monitor,
        });
        process.set_parse_log_file(false);
        process
    }

    /// Java `closeOutputImageFile`.
    pub(crate) fn close_output_image_file(&self) {
        let Some(comscript_state) = &self.comscript_state else {
            return;
        };
        self.manager.close_stale_file(
            comscript_state.get_output_image_file_key(),
            Some(self.axis_id),
        );
        self.manager.close_stale_file(
            comscript_state.get_output_image_file_key2(),
            Some(self.axis_id),
        );
        self.manager.close_stale_file(
            comscript_state.get_output_image_file_key3(),
            Some(self.axis_id),
        );
    }

    /// Java `isComScriptBusy`.  Since background processes can run after etomo
    /// has exited, it would be easy to start a second combine that would
    /// interfer and cause file corruption.  Check to see if comscript log is
    /// open by using lsof (list open files).  If it is, stop the monitor and
    /// return false.  If the monitor isn't stopped it reattaches to the
    /// existing combine.log.
    pub(crate) fn is_com_script_busy(&self, process: &ComScriptProcess) -> bool {
        // `System.getProperty("os.name")`
        // lsof does not exist in Windows. In Windows, a busy log file will be
        // detected when the rename fails.
        if !cfg!(target_os = "linux") && !cfg!(target_os = "macos") {
            return false;
        }
        let working_directory = process.get_working_directory().unwrap_or_default();
        let pid_file = working_directory.join(process.get_watched_file_name().unwrap_or_default());
        let mut group_pid = None;
        if pid_file.exists() {
            group_pid = self.parse_pid_string(process, &pid_file);
        }
        if !file_location::LSOF.exists() {
            // if lsof cannot be run assume that com script is not busy
            return false;
        }
        let lsof_path = file_location::LSOF
            .get_absolute_path()
            .unwrap_or_else(|| "null".to_owned());
        let command = match group_pid {
            None => {
                // Upstream bug fixed in translation (BackgroundComScriptProcess.java:95):
                // the Java wraps the directory in literal double quotes
                // (`"\"" + manager.getPropertyUserDir() + "\""`), which
                // Runtime.exec passes to lsof unchanged (there is no shell), so
                // `+D` names a directory that does not exist and the busy check
                // never finds anything.  The directory is passed as itself.
                vec![
                    lsof_path,
                    "-w".to_owned(),
                    "-S".to_owned(),
                    "-l".to_owned(),
                    "-M".to_owned(),
                    "-L".to_owned(),
                    "+D".to_owned(),
                    self.manager
                        .get_property_user_dir()
                        .unwrap_or_else(|| "null".to_owned()),
                ]
            }
            Some(group_pid) => vec![
                lsof_path,
                "-w".to_owned(),
                "-S".to_owned(),
                "-l".to_owned(),
                "-M".to_owned(),
                "-L".to_owned(),
                "-g".to_owned(),
                group_pid,
            ],
        };
        let lsof = SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(command),
            self.axis_id,
        );
        lsof.run();
        let Some(stdout) = lsof.get_std_output() else {
            return false;
        };
        if stdout.is_empty() {
            return false;
        }
        let header =
            crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(&stdout[0]);
        let name_index = header.find(" NAME").map_or(0, |index| index + 1);
        // Return false if the NAME field is not found
        if name_index == 0 {
            return false;
        }
        let comscript_log = match LogFile::get_instance_name(
            &utilities::java_io_file_get_absolute_path(&working_directory.to_string_lossy()),
            self.axis_id,
            &self
                .comscript_state
                .as_ref()
                .map_or_else(|| "null".to_owned(), |state| state.get_comscript_name()),
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        ) {
            Ok(comscript_log) => comscript_log,
            Err(e) => {
                eprintln!("{}", e.get_message());
                return false;
            }
        };
        for (i, line) in stdout.iter().enumerate().skip(1) {
            // check for missing size entry - assume name is last
            // Upstream bug fixed in translation (BackgroundComScriptProcess.java:124):
            // `stdout[i].substring(nameIndex)` throws
            // StringIndexOutOfBoundsException for a line shorter than the
            // header's NAME column, ending the check with an exception; here
            // such a line does not match.
            let Some(name) = line.get(name_index..) else {
                continue;
            };
            if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(name)
                == comscript_log.get_absolute_path()
            {
                eprintln!("\nisComScriptBusy:");
                eprintln!("lsof:stdout[{i}]:{line}");
                eprintln!(
                    "lsof output contains comscriptLog.getAbsolutePath().  Comscript is busy.\n"
                );
                if let Some(monitor) = process.get_monitor() {
                    monitor.kill(process, self.axis_id);
                }
                return true;
            }
        }
        false
    }

    /// Java `renameFiles`.
    pub(crate) fn rename_files(&self, process: &ComScriptProcess) -> Result<bool, LockException> {
        let working_directory = process.get_working_directory();
        let mut retval = process.rename_files_with(
            process.get_watched_file_name().as_deref(),
            working_directory.as_deref(),
            process.get_log_file().as_ref(),
            false,
        )?;
        let Some(comscript_state) = &self.comscript_state else {
            return Ok(retval);
        };
        let start_command = comscript_state.get_start_command();
        let end_command = comscript_state.get_end_command();
        let mut index = start_command;
        while index <= end_command {
            let command = comscript_state
                .get_command(index)
                .unwrap_or_else(|| "null".to_owned());
            match LogFile::get_instance_name(
                &self.manager.get_property_user_dir().unwrap_or_default(),
                process.get_axis_id(),
                &command,
                Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
            ) {
                Ok(log_file) => {
                    if !process.rename_files_with(
                        comscript_state.get_watched_file(index).as_deref(),
                        working_directory.as_deref(),
                        Some(&log_file),
                        false,
                    )? {
                        retval = false;
                    }
                }
                Err(_) => {
                    eprintln!("Error: Invalid log file:{command}");
                    retval = false;
                }
            }
            index += 1;
        }
        Ok(retval)
    }

    /// Java `execPython`.  Places commmands in the .py file.  Creates and runs a
    /// file containing commands to execute the .py file in the background.
    /// `Err(Some(message))` is the `LogFileException`/`IOException` arm,
    /// `Err(None)` the `SystemProcessException` one (which `run` ignores).
    pub(crate) fn exec_python(
        &self,
        process: &ComScriptProcess,
        commands: Option<Vec<String>>,
    ) -> Result<(), Option<String>> {
        let working_directory = process.get_working_directory().unwrap_or_default();
        let run_name = ComScriptProcess::parse_base_name(&process.get_com_script_name(), ".com")
            .unwrap_or_else(|| "null".to_owned());
        let python_file_name = format!("{run_name}.py");
        let python_file = working_directory.join(&python_file_name);
        let watched_file_name = process
            .get_watched_file_name()
            .unwrap_or_else(|| "null".to_owned());
        let out_file = working_directory.join(&watched_file_name);
        utilities::write_file(
            Some(self.manager),
            Some(self.axis_id),
            Some(&python_file),
            commands.as_deref(),
            true,
            false,
        )
        .map_err(|e| Some(e.to_string()))?;
        let python_script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .unwrap_or_default();
        let log_file = process.get_log_file();
        let command = match (system_program::imod_executable(), &log_file) {
            // Our command-file runner in place of `python -u <root>.py`; see the
            // module comment.
            (Some(imod), Some(log_file)) => vec![
                "python".to_owned(),
                "-u".to_owned(),
                format!("{python_script_path}startprocess"),
                "-o".to_owned(),
                watched_file_name.clone(),
                imod.to_string_lossy().into_owned(),
                "runcom".to_owned(),
                "-P".to_owned(),
                process.get_com_script_name(),
                log_file.get_name(),
            ],
            _ => vec![
                "python".to_owned(),
                "-u".to_owned(),
                format!("{python_script_path}startprocess"),
                "-o".to_owned(),
                watched_file_name.clone(),
                "python".to_owned(),
                "-u".to_owned(),
                python_file_name,
            ],
        };
        let program = Arc::new(SystemProgram::background(
            self.manager,
            Some(command.clone()),
            Arc::new(DetachedMonitorWait(Arc::clone(&self.detached_monitor))),
            process.get_axis_id(),
        ));
        process.set_system_program(Arc::clone(&program));
        program.set_working_directory(Some(working_directory));

        let parse_pid = parse_background_pid(
            Arc::clone(&program),
            Arc::clone(&self.process_id),
            out_file,
            process.get_process_data(),
        );
        std::thread::spawn(move || parse_pid.run());

        // make sure nothing else is writing or backing up the log files
        // Upstream bug fixed in translation (BackgroundComScriptProcess.java:185):
        // a null log file (it could not be created) throws NullPointerException
        // here; here the run goes ahead without the write lock.
        let log_writing_id = match log_file
            .as_ref()
            .map(|log_file| log_file.open_for_writing())
        {
            Some(Ok(id)) => Some(id),
            Some(Err(LogFileError::Lock(e))) => {
                process.handle_lock_exception(&e, false);
                return Ok(());
            }
            Some(Err(e)) => return Err(Some(e.get_message())),
            None => None,
        };

        program.run();

        // release the log files
        if let (Some(log_file), Some(log_writing_id)) = (&log_file, &log_writing_id) {
            log_file.close_id(Some(log_writing_id));
        }
        // Check the exit value, if it is non zero, parse the warnings and errors
        // from the log file.
        if program.get_exit_value() != 0 {
            // `throw new SystemProcessException(command[0] + " process died
            // immediately")`: `ComScriptProcess.run` catches it and drops the
            // message.
            return Err(None);
        }
        Ok(())
    }

    /// Java `notifyKilled`: kill monitor when notified that a kill was done.
    pub(crate) fn notify_killed(&self, process: &ComScriptProcess) {
        if let Some(monitor) = process.get_monitor() {
            monitor.kill(process, self.axis_id);
        }
        // super.notifyKilled()
        process.set_process_end_state(ProcessEndState::Killed);
    }

    /// Java `parse`.  Parses errors and warnings from log files.  Parses errors
    /// and warnings from the comscript and all child comscripts found in
    /// comscriptState that may have been executed.
    pub(crate) fn parse(&self, process: &ComScriptProcess) -> Result<(), ParseError> {
        process.parse_named(&process.get_com_script_name(), true)?;
        let Some(comscript_state) = &self.comscript_state else {
            return Ok(());
        };
        let start_command = comscript_state.get_start_command();
        let end_command = comscript_state.get_end_command();
        let mut index = start_command;
        while index <= end_command {
            process.parse_named(
                &format!(
                    "{}.com",
                    comscript_state
                        .get_command(index)
                        .unwrap_or_else(|| "null".to_owned())
                ),
                false,
            )?;
            index += 1;
        }
        Ok(())
    }

    /// Java private `parsePIDString`.  Want to parse the pid file on this
    /// thread, without access to the system program thread so I can't use
    /// ParseBackgroundPID.
    fn parse_pid_string(&self, process: &ComScriptProcess, out_file: &Path) -> Option<String> {
        let mut pid = String::new();
        let buffered_reader = match std::fs::File::open(out_file) {
            Ok(file) => std::io::BufReader::new(file),
            Err(e) => {
                eprintln!("{e}");
                return None;
            }
        };
        let mut lines = buffered_reader.lines();
        match lines.next() {
            Some(Ok(line)) => {
                // "Runcom PID:" is our command-file runner's line; see the module
                // comment.
                if line.starts_with("Shell PID:")
                    || line.contains("Python PID:")
                    || line.starts_with("Runcom PID:")
                    || line.starts_with("Windows PID:")
                    || line.starts_with("Cygwin PID:")
                {
                    // `line.split("\\s+")`: a leading run of whitespace gives
                    // a leading empty token.
                    let mut tokens: Vec<&str> = Vec::new();
                    if line.starts_with(char::is_whitespace) {
                        tokens.push("");
                    }
                    tokens.extend(line.split_whitespace());
                    if tokens.len() > 2 {
                        let mut found = false;
                        for token in &tokens {
                            if found {
                                pid.push_str(token);
                                break;
                            } else if token.ends_with("PID:") {
                                found = true;
                            }
                        }
                    }
                }
                if let Some(process_data) = process.get_process_data() {
                    process_data.lock().unwrap().set_pid(Some(&pid));
                }
            }
            None => {
                if let Some(process_data) = process.get_process_data() {
                    process_data.lock().unwrap().set_pid(Some(&pid));
                }
            }
            Some(Err(e)) => eprintln!("{e}"),
        }
        // closeFile(bufferedReader): dropping the reader closes it.
        Some(pid)
    }

    /// Java `getShellProcessID`: get the csh process ID if it is available.
    pub(crate) fn get_shell_process_id(&self) -> String {
        self.process_id.lock().unwrap().clone()
    }
}
