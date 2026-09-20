//! `IMOD/qttools/processchunks/processhandler.h` and
//! `IMOD/qttools/processchunks/processhandler.cpp`.

use super::machinehandler::{MachineHandler, ProcessError, ProcessExitStatus};
use super::processchunks::Processchunks;
use super::{CHUNK_DONE, CHUNK_NOT_DONE, CHUNK_TO_SKIP};
use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_milli_sleep};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
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
    /// When `QProcess::finished` would have been emitted: a waiter thread
    /// blocks in `waitid(..., WEXITED | WNOWAIT)` (the child stays reapable
    /// for `try_wait`) and records the instant it returned.
    finish_stamp: Option<Arc<Mutex<Option<Instant>>>>,
}

impl ProcessHandler {
    /// `ProcessHandler()` source constructor.  The paired C++ destructor's
    /// explicit resource releases are represented by Rust `Option` ownership.
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
            decorated_class_name: "14ProcessHandler".to_owned(),
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
            finish_stamp: None,
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
        // `secsTo(now)` for the two stamps.
        let modified_secs = now
            .duration_since(self.log_last_modified)
            .map_or(0, |age| age.as_secs() as i64);
        let size_secs = now
            .duration_since(self.size_changed_time)
            .map_or(0, |age| age.as_secs() as i64);
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "isLogFileOlderThan", 1)
            {
                (*self.processchunks).write_out(&format!(
                    "{}:isLogFileOlderThan: new mod {modified_secs}  size time {size_secs}\n",
                    self.decorated_class_name
                ));
            }
        }
        modified_secs >= timeout_sec as i64 && size_secs >= timeout_sec as i64
    }

    /// C++ private `ProcessHandler::readAllStandardError`.
    pub fn read_all_standard_error(&mut self) {
        if let Some(process) = &mut self.process {
            use std::io::Read;
            if let Some(stderr) = &mut process.stderr {
                let mut err = Vec::new();
                let _ = stderr.read_to_end(&mut err);
                if !err.is_empty() {
                    self.stderr.extend(&err);
                    unsafe {
                        if (*self.processchunks).is_verbose(
                            &self.decorated_class_name,
                            "readAllStandardError",
                            1,
                        ) {
                            // `printf("%s\n", err.data())`
                            let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
                            let _ = ImodFile::Stdout.write_all(&err[..end]);
                            let _ = ImodFile::Stdout.write_all(b"\n");
                        }
                    }
                }
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
        // Qt delivers `QProcess::finished` to the `handleFinished` slot from
        // its event loop.  Rust's child handle has no signal dispatcher, so
        // poll at the same state-check boundary and deliver to the same slot,
        // with the elapsed time taken at the instant the waiter thread saw the
        // exit rather than at this poll.
        if !self.finished_signal_received {
            let mut delivered = None;
            if let Some(process) = &mut self.process {
                if let Ok(Some(status)) = process.try_wait() {
                    // `QProcess::exitCode()` is 0 after a crash; `exitStatus()`
                    // is CrashExit only when the process was killed by a signal.
                    let exit_code = status.code().unwrap_or(0);
                    let exit_status = if status.code().is_some() {
                        ProcessExitStatus::NormalExit
                    } else {
                        ProcessExitStatus::CrashExit
                    };
                    delivered = Some((exit_code, exit_status));
                }
            }
            if let Some((exit_code, exit_status)) = delivered {
                if self.elapsed_time < 0 {
                    if let Some(stamp) = self.finish_stamp.as_ref() {
                        if let Some(finish) = stamp.lock().map(|s| *s).unwrap_or(None) {
                            self.elapsed_time =
                                finish.duration_since(self.start_time).as_millis() as i32;
                        }
                    }
                }
                self.handle_finished(exit_code, exit_status);
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
            if (*self.processchunks).is_verbose(
                &self.decorated_class_name,
                "printTooManyErrorsMessage",
                1,
            ) {
                (*self.processchunks).write_out(&format!(
                    "{}:printTooManyErrorsMessage:mExitCode:{},mExitStatus:{}\n",
                    self.decorated_class_name,
                    self.exit_code,
                    match self.exit_status {
                        ProcessExitStatus::NormalExit => 0,
                        ProcessExitStatus::CrashExit => 1,
                    }
                ));
            }
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
        self.finish_stamp = None;
        if let Some(process) = &self.process {
            // The waiter thread stands in for Qt's SIGCHLD notifier: it wakes
            // when the child exits, without reaping it.
            let pid = process.id();
            let stamp = Arc::new(Mutex::new(None));
            let writer = Arc::clone(&stamp);
            std::thread::spawn(move || {
                unsafe {
                    let mut info: libc::siginfo_t = std::mem::zeroed();
                    libc::waitid(
                        libc::P_PID,
                        pid as libc::id_t,
                        &mut info,
                        libc::WEXITED | libc::WNOWAIT,
                    );
                }
                if let Ok(mut slot) = writer.lock() {
                    *slot = Some(Instant::now());
                }
            });
            self.finish_stamp = Some(stamp);
        }
        // `mProcess->closeWriteChannel(); b3dMilliSleep(mMillisecSleep);` and,
        // for a queue, `mProcess->waitForFinished(60000)`.
        b3d_milli_sleep(unsafe { (*self.processchunks).get_millisec_sleep() });
        if queue {
            if let Some(process) = &mut self.process {
                let deadline = Instant::now() + Duration::from_millis(60000);
                while !matches!(process.try_wait(), Ok(Some(_))) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        //Turn on running process boolean and record start time
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
        let status_int = match exit_status {
            ProcessExitStatus::NormalExit => 0,
            ProcessExitStatus::CrashExit => 1,
        };
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "handleFinished", 1) {
                let _ = ImodFile::Stdout.write_all(
                    format!(
                        "{}:handleFinished:{exit_code},exitStatus:{status_int}\n",
                        self.decorated_class_name
                    )
                    .as_bytes(),
                );
            }
        }
        if self.com_file_job_index < 0 {
            let _ = ImodFile::Stdout.write_all(b"Processchunks warning: Job index not set\n");
            return;
        }
        self.finished_signal_received = true;
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "handleFinished", 1) {
                let _ = ImodFile::Stdout.write_all(
                    format!(
                        "{}:handleFinished:{}\n",
                        self.decorated_class_name,
                        (*self.processchunks)
                            .get_com_file_jobs()
                            .get_com_file_name(self.com_file_job_index as usize)
                    )
                    .as_bytes(),
                );
                self.read_all_standard_error();
            }
        }
        self.starting_process = false;
        self.exit_code = exit_code;
        self.exit_status = exit_status;

        // record the actual elapsed time when the process ends
        if self.elapsed_time < 0 {
            self.elapsed_time = self.start_time.elapsed().as_millis() as i32;
        }
        //The queue request just submits the chunk to the queue, or submits a request
        //to kill the chunk to the queue.  Don't use it to figure out the state of the
        //chunk.
        if !unsafe { (*self.processchunks).is_queue() } {
            let _ = fs::remove_file(self.get_py_file());
            let _ = fs::remove_file(format!("{}.stdout", self.get_py_file()));
            if !self.kill {
                self.machine = std::ptr::null_mut();
            }
        } else if self.exit_code != 0 || status_int != 0 {
            // Get error messages out when queue submission fails
            let mut byte_array = Vec::new();
            let mut out_array = Vec::new();
            if let Some(process) = &mut self.process {
                use std::io::Read;
                if let Some(stderr) = &mut process.stderr {
                    let _ = stderr.read_to_end(&mut byte_array);
                }
                if let Some(stdout) = &mut process.stdout {
                    let _ = stdout.read_to_end(&mut out_array);
                }
            }
            unsafe {
                if !byte_array.is_empty() {
                    (*self.processchunks)
                        .write_out(&format!("{}\n", String::from_utf8_lossy(&byte_array)));
                }
                if !out_array.is_empty() {
                    let com_split: Vec<&str> = self.command.split(' ').collect();
                    if !com_split.is_empty() {
                        (*self.processchunks).write_out(&format!("{} ", com_split[0]));
                    }
                    (*self.processchunks)
                        .write_out(&format!("{}\n", String::from_utf8_lossy(&out_array)));
                }
            }
        }
    }
    /// C++ `ProcessHandler::handleKillFinished`.
    pub fn handle_kill_finished(&mut self, exit_code: i32, exit_status: ProcessExitStatus) {
        if self.com_file_job_index < 0 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "handleKillFinished", 1)
            {
                (*self.processchunks).write_out(&format!(
                    "{}:handleKillFinished:{},exitCode:{exit_code},exitStatus:{}\n",
                    self.decorated_class_name,
                    (*self.processchunks)
                        .get_com_file_jobs()
                        .get_com_file_name(self.com_file_job_index as usize),
                    match exit_status {
                        ProcessExitStatus::NormalExit => 0,
                        ProcessExitStatus::CrashExit => 1,
                    }
                ));
            }
        }
        if !self.kill_finished_signal_received {
            unsafe {
                (*self.processchunks).decrement_kills();
            }
            self.kill_finished_signal_received = true;
        }
        if unsafe { (*self.processchunks).is_queue() } {
            //The queue kill request is syncronous with the job.
            //exitCode == 0:  Kill is completed
            //exitCode == 100:  Process finished before it could be killed
            //exitCode == 101:  Unable to pause because the process had already started.
            if exit_code != 0 && exit_code != 100 && exit_code != 101 {
                unsafe {
                    (*self.processchunks)
                        .write_out(&format!("kill process exitCode:{exit_code}\n"));
                }
                let mut byte_array = Vec::new();
                if let Some(kill_process) = &mut self.kill_process {
                    use std::io::Read;
                    if let Some(stderr) = &mut kill_process.stderr {
                        let _ = stderr.read_to_end(&mut byte_array);
                    }
                }
                if !byte_array.is_empty() {
                    unsafe {
                        (*self.processchunks)
                            .write_out(&format!("{}\n", String::from_utf8_lossy(&byte_array)));
                    }
                }
            }
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
