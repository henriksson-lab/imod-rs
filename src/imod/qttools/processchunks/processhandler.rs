//! `IMOD/qttools/processchunks/processhandler.h` and
//! `IMOD/qttools/processchunks/processhandler.cpp`.

#![allow(dead_code)]

use super::machinehandler::{MachineHandler, ProcessError, ProcessExitStatus};
use super::processchunks::Processchunks;
use super::{CHUNK_DONE, CHUNK_NOT_DONE, CHUNK_TO_SKIP};
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

/// C++ `ProcessHandler`.
pub struct ProcessHandler {
    log_file: Option<PathBuf>,
    job_file: Option<PathBuf>,
    qid_file: Option<PathBuf>,
    log_file_exists: bool,
    valid_job: bool,
    starting_process: bool,
    pausing: i32,
    com_file_job_index: i32,
    stderr: Vec<u8>,
    pid: String,
    escaped_remote_dir_path: String,
    decorated_class_name: String,
    command: String,
    param_list: Vec<String>,
    processchunks: *mut Processchunks,
    process: Option<Child>,
    pause_time: Instant,
    start_time: Instant,
    no_log_start_time: Instant,
    elapsed_time: i32,
    log_last_modified: SystemTime,
    last_size_check_time: SystemTime,
    size_changed_time: SystemTime,
    last_log_size: i64,
    machine: *mut MachineHandler,
    gpu_number: i32,
    num_did_not_see_log: i32,
    sec_to_wait_if_no_log: i32,
    max_num_no_log_seen: i32,
    kill_process: Option<Child>,
    kill_counter: i32,
    kill: bool,
    kill_started: bool,
    ignore_kill: bool,
    killing_one: bool,
    error_signal_received: bool,
    finished_signal_received: bool,
    kill_finished_signal_received: bool,
    log_has_error: bool,
    process_error: Option<ProcessError>,
    exit_code: i32,
    exit_status: ProcessExitStatus,
}

impl ProcessHandler {
    /// C++ `ProcessHandler::ProcessHandler`.
    pub fn new() -> Self {
        let now = Instant::now();
        let system_now = SystemTime::now();
        let mut process_handler = Self {
            log_file: None,
            job_file: None,
            qid_file: None,
            log_file_exists: false,
            valid_job: false,
            starting_process: false,
            pausing: 0,
            com_file_job_index: -1,
            stderr: Vec::new(),
            pid: String::new(),
            escaped_remote_dir_path: String::new(),
            decorated_class_name: "ProcessHandler".to_owned(),
            command: String::new(),
            param_list: Vec::new(),
            processchunks: std::ptr::null_mut(),
            process: None,
            pause_time: now,
            start_time: now,
            no_log_start_time: now,
            elapsed_time: -1,
            log_last_modified: system_now,
            last_size_check_time: system_now,
            size_changed_time: system_now,
            last_log_size: 0,
            machine: std::ptr::null_mut(),
            gpu_number: -1,
            num_did_not_see_log: 0,
            sec_to_wait_if_no_log: 50,
            max_num_no_log_seen: 50,
            kill_process: None,
            kill_counter: 0,
            kill: false,
            kill_started: false,
            ignore_kill: true,
            killing_one: false,
            error_signal_received: false,
            finished_signal_received: false,
            kill_finished_signal_received: false,
            log_has_error: false,
            process_error: None,
            exit_code: -1,
            exit_status: ProcessExitStatus::CrashExit,
        };
        process_handler.reset_fields();
        process_handler
    }

    /// C++ `ProcessHandler::resetFields`.
    pub fn reset_fields(&mut self) {
        self.log_file_exists = false;
        self.error_signal_received = false;
        self.process_error = None;
        self.finished_signal_received = false;
        self.start_time = Instant::now();
        self.elapsed_time = -1;
        self.log_last_modified = SystemTime::now();
        self.last_size_check_time = self.log_last_modified;
        self.size_changed_time = self.log_last_modified;
        self.last_log_size = 0;
        self.starting_process = false;
        self.kill_finished_signal_received = false;
        self.kill = false;
        self.num_did_not_see_log = 0;
        self.reset_signal_values();
        self.pausing = 0;
        self.stderr.clear();
        self.pid.clear();
    }

    /// C++ `ProcessHandler::initProcess`.
    pub fn init_process(&mut self) {
        self.process = None;
    }

    /// C++ `ProcessHandler::setup`.
    pub fn setup(&mut self, processchunks: &mut Processchunks, gpu_num: i32) {
        self.processchunks = processchunks;
        self.escaped_remote_dir_path = processchunks.get_remote_dir().replace(' ', "\\ ");
        self.gpu_number = gpu_num;
        self.command = if processchunks.is_queue() {
            processchunks.get_queue_command().to_owned()
        } else {
            "python".to_owned()
        };
        self.init_process();
    }

    /// C++ `ProcessHandler::setJob`.
    pub fn set_job(&mut self, job_index: i32) {
        if self.valid_job {
            return;
        }
        self.reset_fields();
        self.valid_job = true;
        self.com_file_job_index = job_index;
        unsafe {
            let processchunks = &mut *self.processchunks;
            let jobs = processchunks.get_com_file_jobs();
            let index = job_index as usize;
            self.log_file = Some(PathBuf::from(jobs.get_log_file_name(index)));
            self.param_list.clear();
            if processchunks.is_queue() {
                self.job_file = Some(PathBuf::from(jobs.get_job_file_name(index)));
                self.qid_file = Some(PathBuf::from(jobs.get_qid_file_name(index)));
                self.param_list
                    .extend(processchunks.get_queue_param_list().iter().cloned());
                self.param_list.extend([
                    "-w".to_owned(),
                    self.escaped_remote_dir_path.clone(),
                    "-a".to_owned(),
                    "R".to_owned(),
                    jobs.get_root(index).to_owned(),
                ]);
            } else {
                self.param_list
                    .extend(["-u".to_owned(), jobs.get_py_file_name(index)]);
            }
        }
    }

    /// C++ `ProcessHandler::setFlagNotDone`.
    pub fn set_flag_not_done(&mut self, single_file: bool) {
        if self.com_file_job_index < 0 || self.get_flag() == CHUNK_TO_SKIP {
            return;
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs_mut()
                .set_flag_not_done(self.com_file_job_index as usize, single_file);
        }
    }

    /// C++ `ProcessHandler::resetSignalValues`.
    pub fn reset_signal_values(&mut self) {
        self.finished_signal_received = false;
        self.error_signal_received = false;
        self.exit_code = -1;
        self.exit_status = ProcessExitStatus::CrashExit;
        self.process_error = None;
        self.log_has_error = false;
    }

    /// C++ `ProcessHandler::getFlag`.
    pub fn get_flag(&self) -> i32 {
        if self.com_file_job_index < 0 {
            return CHUNK_DONE;
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_flag(self.com_file_job_index as usize)
        }
    }

    /// C++ `ProcessHandler::logFileExists`.
    pub fn log_file_exists(&mut self, newly_created_file: bool) -> bool {
        if self.com_file_job_index < 0 {
            return false;
        }
        if self.log_file_exists {
            return true;
        }
        let exists = self.log_file.as_ref().is_some_and(|path| path.exists());
        if newly_created_file {
            self.log_file_exists = exists;
        }
        self.log_file_exists || exists
    }

    /// C++ `ProcessHandler::qidFileExists`.
    pub fn qid_file_exists(&self) -> bool {
        if self.com_file_job_index < 0 {
            return false;
        }
        unsafe {
            !(*self.processchunks).is_queue()
                || self.qid_file.as_ref().is_some_and(|path| path.exists())
        }
    }

    /// C++ `ProcessHandler::getPid`.
    pub fn get_pid(&mut self) -> String {
        if self.com_file_job_index < 0 {
            return String::new();
        }
        if !self.pid.is_empty() {
            return self.pid.clone();
        }
        let queue = unsafe { (*self.processchunks).is_queue() };
        let contents = if queue {
            self.qid_file
                .as_ref()
                .and_then(|p| fs::read(p).ok())
                .unwrap_or_default()
        } else {
            self.read_all_standard_error();
            self.stderr.clone()
        };
        self.get_pid_from_bytes(&contents, true);
        self.pid.clone()
    }

    /// C++ private `ProcessHandler::getPid(QTextStream &, bool)`.
    pub fn get_pid_from_bytes(&mut self, stream: &[u8], save: bool) -> bool {
        if save && !self.pid.is_empty() {
            return true;
        }
        let output = String::from_utf8_lossy(stream);
        if let Some(index) = output.rfind("PID:") {
            if let Some(end) = output[index..].find('\n') {
                if save {
                    self.pid = output[index + 4..index + end].trim().to_owned();
                }
                return true;
            }
        }
        false
    }

    /// C++ `ProcessHandler::isPidInStderr`.
    pub fn is_pid_in_stderr(&mut self) -> bool {
        if unsafe { (*self.processchunks).is_queue() } {
            return false;
        }
        self.read_all_standard_error();
        let stderr = self.stderr.clone();
        self.get_pid_from_bytes(&stderr, false)
    }

    /// C++ `ProcessHandler::readAllLogFile`.
    pub fn read_all_log_file(&self) -> Vec<u8> {
        if self.com_file_job_index < 0 {
            return Vec::new();
        }
        self.log_file
            .as_ref()
            .and_then(|path| fs::read(path).ok())
            .unwrap_or_default()
    }

    /// C++ `ProcessHandler::isLogFileEmpty`.
    pub fn is_log_file_empty(&self) -> bool {
        self.read_all_log_file().is_empty()
    }

    /// C++ declaration `ProcessHandler::isJobFileEmpty`.
    ///
    /// The vendored header declares this method, but neither its implementation
    /// nor a call site exists in the IMOD processchunks sources.
    pub fn is_job_file_empty(&self) -> bool {
        unimplemented!("ProcessHandler::isJobFileEmpty has no definition in IMOD source")
    }

    /// C++ `ProcessHandler::isLogFileOlderThan`.
    pub fn is_log_file_older_than(&mut self, timeout_sec: i32) -> bool {
        if timeout_sec <= 0 {
            return false;
        }
        let now = SystemTime::now();
        let Some(path) = self.log_file.as_ref() else {
            return false;
        };
        let Ok(metadata) = fs::metadata(path) else {
            return false;
        };
        let new_size = metadata.len() as i64;
        self.last_size_check_time = now;
        if new_size > self.last_log_size {
            self.last_log_size = new_size;
            self.size_changed_time = now;
        }
        self.log_last_modified = metadata.modified().unwrap_or(now);
        self.log_last_modified
            .elapsed()
            .map_or(false, |age| age >= Duration::from_secs(timeout_sec as u64))
            && self
                .size_changed_time
                .elapsed()
                .map_or(false, |age| age >= Duration::from_secs(timeout_sec as u64))
    }

    /// C++ private `ProcessHandler::readAllStandardError`.
    pub fn read_all_standard_error(&mut self) {
        if let Some(process) = &mut self.process {
            use std::io::Read;
            if let Some(stderr) = &mut process.stderr {
                let mut err = Vec::new();
                let _ = stderr.read_to_end(&mut err);
                self.stderr.extend(err);
            }
        }
    }

    /// C++ `ProcessHandler::getSshError`.
    pub fn get_ssh_error(&mut self, drop_mess: &mut String) -> bool {
        let contents = if unsafe { (*self.processchunks).is_queue() } {
            self.job_file
                .as_ref()
                .and_then(|p| fs::read(p).ok())
                .unwrap_or_default()
        } else {
            self.read_all_standard_error();
            self.stderr.clone()
        };
        self.get_ssh_error_from_bytes(drop_mess, &contents)
    }

    /// C++ private `ProcessHandler::getSshError(QString &, QTextStream &)`.
    pub fn get_ssh_error_from_bytes(&self, drop_mess: &mut String, stream: &[u8]) -> bool {
        for line in String::from_utf8_lossy(stream).lines() {
            if line.contains("cd: ") {
                *drop_mess = format!("it cannot cd to {} ({line})", unsafe {
                    (*self.processchunks).get_remote_dir()
                });
                return true;
            }
            if line.contains("ssh: connect to host") {
                *drop_mess = format!("cannot connect ({line})");
                return true;
            }
        }
        false
    }

    /// C++ `ProcessHandler::isComProcessDone`.
    pub fn is_com_process_done(&mut self) -> bool {
        if self.com_file_job_index < 0 {
            return false;
        }
        // Qt updates the two signal fields from its event loop.  Rust's child
        // handle has no signal dispatcher, so perform the equivalent poll at
        // the same state-check boundary.
        if !self.finished_signal_received {
            if let Some(process) = &mut self.process {
                if let Ok(Some(status)) = process.try_wait() {
                    self.finished_signal_received = true;
                    self.starting_process = false;
                    self.exit_code = status.code().unwrap_or(1);
                    self.exit_status = if status.success() {
                        ProcessExitStatus::NormalExit
                    } else {
                        ProcessExitStatus::CrashExit
                    };
                    self.elapsed_time = self.start_time.elapsed().as_millis() as i32;
                }
            }
        }
        if unsafe { (*self.processchunks).is_queue() } {
            if self.exit_code != 0 || self.exit_status != ProcessExitStatus::NormalExit {
                return true;
            }
            let py_exists = self.py_file_exists();
            if !py_exists && self.log_file_exists(true) {
                self.machine = std::ptr::null_mut();
                return true;
            }
            if !py_exists {
                if self.num_did_not_see_log == 0 {
                    self.no_log_start_time = Instant::now();
                }
                self.num_did_not_see_log += 1;
                return self.no_log_start_time.elapsed()
                    > Duration::from_secs(self.sec_to_wait_if_no_log as u64)
                    && self.num_did_not_see_log > self.max_num_no_log_seen;
            }
            self.num_did_not_see_log = 0;
            return false;
        }
        self.finished_signal_received
            && self.log_file_exists(true)
            && self.get_flag() != CHUNK_NOT_DONE
    }

    /// C++ `ProcessHandler::isChunkDone`.
    pub fn is_chunk_done(&mut self) -> bool {
        if self.com_file_job_index < 0 {
            return false;
        }
        let log = self.read_all_log_file();
        if log.is_empty() {
            return false;
        }
        let start = log.len().saturating_sub(512);
        let last_part = String::from_utf8_lossy(&log[start..]).trim().to_owned();
        self.log_has_error = last_part.contains("ERROR:");
        last_part.ends_with("CHUNK DONE")
    }

    /// C++ `ProcessHandler::isPausing`.
    pub fn is_pausing(&mut self) -> bool {
        if self.pausing != 0 {
            return self.pause_time.elapsed() <= Duration::from_secs(1);
        }
        if self.exit_code > 0 && self.log_has_error {
            return false;
        }
        self.pausing += 1;
        self.pause_time = Instant::now();
        true
    }

    /// C++ `ProcessHandler::getErrorMessageFromLog`.
    pub fn get_error_message_from_log(&self, error_mess: &mut String) {
        let log = String::from_utf8_lossy(&self.read_all_log_file()).to_string();
        if log.is_empty() {
            return;
        }
        let tail = &log[log.len().saturating_sub(1000)..];
        let mut found = false;
        let mut last = "";
        for line in tail.lines() {
            last = line;
            if found || line.contains("ERROR:") {
                found = true;
                error_mess.push_str(line);
                error_mess.push('\n');
            }
        }
        if error_mess.is_empty() {
            error_mess.push_str(&format!("CHUNK ERROR: (last line) - {last}\n"));
        } else {
            error_mess.insert_str(0, "CHUNK ");
        }
        error_mess.push_str("END CHUNK ERROR");
    }

    /// C++ `ProcessHandler::getErrorMessageFromOutput`.
    pub fn get_error_message_from_output(&mut self, error_mess: &mut String) {
        error_mess.push('\n');
        self.read_all_standard_error();
        let output = String::from_utf8_lossy(&self.stderr).trim().to_owned();
        if let Some(line) = output.lines().last() {
            error_mess.push_str(line);
            error_mess.push('\n');
        }
    }

    /// C++ `ProcessHandler::printTooManyErrorsMessage`.
    pub fn print_too_many_errors_message(&self, num_err: i32) {
        unsafe {
            (*self.processchunks).write_out(&format!(
                "ERROR: {} has given processing error {num_err} times - giving up\n",
                self.get_com_file_name()
            ));
        }
    }

    /// C++ `ProcessHandler::printWarnings`.
    pub fn print_warnings(&self, machine_name: &str) {
        let mut warn_list: Vec<String> = Vec::new();
        let mut num_warns: Vec<i32> = Vec::new();
        for line in String::from_utf8_lossy(&self.read_all_log_file()).lines() {
            if line.contains("WARNING:") {
                let mut warning = line.trim().to_owned();
                let mut match_index = None;
                for (index, old_warning) in warn_list.iter().enumerate() {
                    if warning == *old_warning {
                        match_index = Some(index);
                        break;
                    }
                    let line_index = warning.rfind(' ');
                    let warn_index = old_warning.rfind(' ');
                    if let (Some(line_index), Some(warn_index)) = (line_index, warn_index) {
                        if line_index > 0
                            && line_index == warn_index
                            && warning[..line_index] == old_warning[..warn_index]
                        {
                            warning = format!("{} ...", &old_warning[..warn_index]);
                            match_index = Some(index);
                            break;
                        }
                    }
                }
                if let Some(index) = match_index {
                    num_warns[index] += 1;
                    warn_list[index] = warning;
                } else {
                    warn_list.push(warning);
                    num_warns.push(1);
                }
            } else if line.contains("MESSAGE:") {
                unsafe {
                    (*self.processchunks)
                        .write_out(&format!("{} - on {machine_name}\n", line.trim()));
                }
            } else if line.contains("LOGFILE:") {
                unsafe {
                    (*self.processchunks).write_out(&format!("{}\n", line.trim()));
                }
            }
        }
        for (warning, count) in warn_list.into_iter().zip(num_warns) {
            unsafe {
                (*self.processchunks).write_out(&format!(
                    "{}{suffix}\n",
                    warning,
                    suffix = if count > 1 {
                        format!(" ({count} times)")
                    } else {
                        String::new()
                    }
                ));
            }
        }
    }

    /// C++ inline `ProcessHandler::backupLog`.
    pub fn backup_log(&self) {
        let Some(log_file) = &self.log_file else {
            return;
        };
        let backup = PathBuf::from(format!("{}~", log_file.to_string_lossy()));
        let _ = fs::copy(log_file, backup);
    }

    /// C++ `ProcessHandler::incrementNumChunkErr`.
    pub fn increment_num_chunk_err(&mut self) {
        if self.com_file_job_index >= 0 {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs_mut()
                    .increment_num_chunk_err(self.com_file_job_index as usize);
            }
        }
    }

    /// C++ `ProcessHandler::pyFileExists`.
    pub fn py_file_exists(&self) -> bool {
        self.com_file_job_index >= 0
            && PathBuf::from(unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_py_file_name(self.com_file_job_index as usize)
            })
            .exists()
    }

    /// C++ `ProcessHandler::getNumChunkErr`.
    pub fn get_num_chunk_err(&self) -> i32 {
        if self.com_file_job_index < 0 {
            0
        } else {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_num_chunk_err(self.com_file_job_index as usize)
            }
        }
    }

    /// C++ `ProcessHandler::getComFileName`.
    pub fn get_com_file_name(&self) -> String {
        if self.com_file_job_index < 0 {
            String::new()
        } else {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_com_file_name(self.com_file_job_index as usize)
            }
        }
    }

    /// C++ `ProcessHandler::getLogFileName`.
    pub fn get_log_file_name(&self) -> String {
        if self.com_file_job_index < 0 {
            String::new()
        } else {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_log_file_name(self.com_file_job_index as usize)
            }
        }
    }

    /// C++ `ProcessHandler::isStartProcessTimedOut`.
    pub fn is_start_process_timed_out(&mut self, timeout_millisec: i32) -> bool {
        self.starting_process
            && self.start_time.elapsed() > Duration::from_millis(timeout_millisec as u64)
            && !self.log_file_exists(true)
    }

    /// C++ `ProcessHandler::setFlag`.
    pub fn set_flag(&mut self, flag: i32) {
        if self.com_file_job_index >= 0 && self.get_flag() != CHUNK_TO_SKIP {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs_mut()
                    .set_flag(self.com_file_job_index as usize, flag);
            }
        }
    }

    /// C++ `ProcessHandler::removeFiles`.
    pub fn remove_files(&mut self) {
        if self.com_file_job_index < 0 {
            return;
        }
        if let Some(path) = &self.log_file {
            let _ = fs::remove_file(path);
        }
        unsafe {
            let _ = fs::remove_file(
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_py_file_name(self.com_file_job_index as usize),
            );
        }
        self.remove_process_files();
    }

    /// C++ `ProcessHandler::removeProcessFiles`.
    pub fn remove_process_files(&mut self) {
        if unsafe { (*self.processchunks).is_queue() } {
            if let Some(path) = &self.job_file {
                let _ = fs::remove_file(path);
            }
            if let Some(path) = &self.qid_file {
                let _ = fs::remove_file(path);
            }
        } else {
            self.stderr.clear();
            self.pid.clear();
        }
    }

    /// C++ `ProcessHandler::getPyFile`.
    pub fn get_py_file(&self) -> String {
        if self.com_file_job_index < 0 {
            String::new()
        } else {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_py_file_name(self.com_file_job_index as usize)
            }
        }
    }

    /// C++ `ProcessHandler::runProcess`.
    pub fn run_process(&mut self, machine: &mut MachineHandler) {
        if !self.valid_job {
            return;
        }
        self.machine = machine;
        let queue = unsafe { (*self.processchunks).is_queue() };
        let mut command =
            if !queue && !unsafe { (*self.processchunks).name_is_local_host(machine.get_name()) } {
                let mut command = Command::new("ssh");
                command
                    .arg("-x")
                    .arg(machine.get_name())
                    .arg("bash")
                    .arg("--login")
                    .arg("-c")
                    .arg(format!(
                        "\"cd {} && python -u < {}\"",
                        self.escaped_remote_dir_path,
                        self.get_py_file()
                    ));
                command
            } else {
                let mut command = Command::new(&self.command);
                command.args(&self.param_list);
                command
            };
        if !queue {
            if let Ok(file) = fs::File::create(format!("{}.stdout", self.get_py_file())) {
                command.stdout(Stdio::from(file));
            }
            if let Ok(file) = fs::File::open(self.get_py_file()) {
                command.stdin(Stdio::from(file));
            }
        }
        command.stderr(Stdio::piped());
        self.reset_signal_values();
        self.process = command.spawn().ok();
        self.starting_process = true;
        self.start_time = Instant::now();
    }

    /// C++ inline `ProcessHandler::isFinishedSignalReceived`.
    pub fn is_finished_signal_received(&self) -> bool {
        self.finished_signal_received
    }
    /// C++ inline `ProcessHandler::resetPausing`.
    pub fn reset_pausing(&mut self) {
        self.pausing = 0;
    }
    /// C++ inline `ProcessHandler::invalidateJob`.
    pub fn invalidate_job(&mut self) {
        self.valid_job = false;
    }
    /// C++ inline `ProcessHandler::getAssignedJobIndex`.
    pub fn get_assigned_job_index(&self) -> i32 {
        self.com_file_job_index
    }
    /// C++ inline `ProcessHandler::getGpuNumber`.
    pub fn get_gpu_number(&self) -> i32 {
        self.gpu_number
    }
    /// C++ inline `ProcessHandler::getElapsedTime`.
    pub fn get_elapsed_time(&self) -> i32 {
        if self.elapsed_time < 0 {
            self.start_time.elapsed().as_millis() as i32
        } else {
            self.elapsed_time
        }
    }
    /// C++ inline `ProcessHandler::isJobValid`.
    pub fn is_job_valid(&self) -> bool {
        self.valid_job
    }
    /// C++ inline `ProcessHandler::isKillFinished`.
    pub fn is_kill_finished(&self) -> bool {
        self.ignore_kill || (self.kill_finished_signal_received && self.finished_signal_received)
    }

    /// C++ `ProcessHandler::startKill`.
    pub fn start_kill(&mut self, kill_one: bool) {
        self.kill = true;
        self.ignore_kill = !kill_one && !self.is_job_valid();
        self.killing_one = kill_one;
    }
    /// C++ `ProcessHandler::killSignal`.
    pub fn kill_signal(&mut self) {
        if self.ignore_kill
            || (self.kill_finished_signal_received && self.finished_signal_received)
            || !unsafe { (*self.processchunks).is_queue() }
        {
            return;
        }
        if !self.kill_started {
            self.kill_started = true;
            self.set_job_not_done();
            unsafe {
                (*self.processchunks).increment_kills();
            }
        } else {
            self.kill_counter += 1;
            if self.kill_counter > 15 {
                self.kill_process = None;
                unsafe {
                    (*self.processchunks).decrement_kills();
                }
                self.kill_finished_signal_received = true;
                self.finished_signal_received = true;
            }
        }
    }
    /// C++ `ProcessHandler::resetKill`.
    pub fn reset_kill(&mut self) {
        self.kill_finished_signal_received = false;
        self.kill_started = false;
        self.kill_counter = 0;
        self.kill = false;
        self.ignore_kill = true;
    }
    /// C++ `ProcessHandler::isPidEmpty`.
    pub fn is_pid_empty(&mut self) -> bool {
        self.get_pid().is_empty()
    }
    /// C++ `ProcessHandler::setJobNotDone`.
    pub fn set_job_not_done(&mut self) {
        if self.com_file_job_index >= 0 {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs_mut()
                    .set_flag(self.com_file_job_index as usize, CHUNK_NOT_DONE);
            }
        }
    }
    /// C++ `ProcessHandler::killQProcesses`.
    pub fn kill_q_processes(&mut self) {
        if let Some(process) = &mut self.process {
            let _ = process.kill();
        }
        if let Some(process) = &mut self.kill_process {
            let _ = process.kill();
        }
    }
    /// C++ `ProcessHandler::handleFinished`.
    pub fn handle_finished(&mut self, exit_code: i32, exit_status: ProcessExitStatus) {
        if self.com_file_job_index < 0 {
            return;
        }
        self.finished_signal_received = true;
        self.starting_process = false;
        self.exit_code = exit_code;
        self.exit_status = exit_status;
        if self.elapsed_time < 0 {
            self.elapsed_time = self.start_time.elapsed().as_millis() as i32;
        }
        if !unsafe { (*self.processchunks).is_queue() } {
            let _ = fs::remove_file(self.get_py_file());
            let _ = fs::remove_file(format!("{}.stdout", self.get_py_file()));
            if !self.kill {
                self.machine = std::ptr::null_mut();
            }
        }
    }
    /// C++ `ProcessHandler::handleKillFinished`.
    pub fn handle_kill_finished(&mut self, _exit_code: i32, _exit_status: ProcessExitStatus) {
        if self.com_file_job_index < 0 {
            return;
        }
        if !self.kill_finished_signal_received {
            unsafe {
                (*self.processchunks).decrement_kills();
            }
            self.kill_finished_signal_received = true;
        }
    }
    /// C++ `ProcessHandler::handleError`.
    pub fn handle_error(&mut self, process_error: ProcessError) {
        if self.kill || self.com_file_job_index < 0 {
            return;
        }
        self.error_signal_received = true;
        self.process_error = Some(process_error);
    }
    /// C++ `ProcessHandler::closeProcess`.
    pub fn close_process(&mut self) {
        if self.starting_process {
            if let Some(process) = &mut self.process {
                let _ = process.kill();
            }
        }
    }
}
