//! `IMOD/qttools/processchunks/processhandler.h` and
//! `IMOD/qttools/processchunks/processhandler.cpp`.
//!
//! Qt is not a dependency of this crate, so the two foreign boundaries the
//! C++ leans on are stood in for: a `QProcess` is a `std::process::Child`
//! whose output channels are drained by reader threads the way Qt's socket
//! notifiers drain them, and the `finished` signal is delivered by
//! `deliver_process_signals` at the state-check boundaries where Qt's event
//! loop would have run the slot.  Every function below is the C++ function of
//! the same name with the C++ body.

use super::machinehandler::{MachineHandler, ProcessError, ProcessExitStatus};
use super::processchunks::Processchunks;
use super::{CHUNK_DONE, CHUNK_NOT_DONE, CHUNK_TO_SKIP};
use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_milli_sleep, imod_backup_file};
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
    /// `QProcess::errorString()`, which `setErrorAndEmit` stores alongside the
    /// error the `error`/`errorOccurred` signal carries.
    error_string: String,
    /// C++ `int mExitCode`.
    exit_code: i32,
    /// C++ `int mExitStatus`, which holds -1 before any `finished` signal, a
    /// value `QProcess::ExitStatus` cannot represent.
    exit_status: i32,
    /// Qt drains a `QProcess`'s output channels into its own buffer as the
    /// child writes; `readAllStandardError`/`readAllStandardOutput` then take
    /// whatever has arrived without blocking.  These are that buffer.
    stderr_buffer: Option<Arc<Mutex<Vec<u8>>>>,
    stdout_buffer: Option<Arc<Mutex<Vec<u8>>>>,
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
            error_string: String::new(),
            exit_code: -1,
            exit_status: -1,
            stderr_buffer: None,
            stdout_buffer: None,
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
        self.stderr_buffer = None;
        self.stdout_buffer = None;
        self.finish_stamp = None;
    }

    /// Qt's event loop, not a source function: `QProcess` emits `finished` to
    /// `handleFinished` as soon as the loop spins after the child exits.  A
    /// `std::process::Child` has no signal dispatcher, so the exit is picked
    /// up here at the state checks where the slot would already have run, and
    /// the elapsed time is taken at the instant the waiter thread saw the exit
    /// rather than at this poll.
    fn deliver_process_signals(&mut self) {
        if self.finished_signal_received {
            return;
        }
        let mut delivered = None;
        if let Some(process) = &mut self.process {
            if let Ok(Some(status)) = process.try_wait() {
                // `QProcess::exitCode()` is 0 after a crash; `exitStatus()` is
                // CrashExit only when the process was killed by a signal.
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
                        self.elapsed_time = finish
                            .saturating_duration_since(self.start_time)
                            .as_millis() as i32;
                    }
                }
            }
            self.handle_finished(exit_code, exit_status);
        }
    }

    /// C++ `ProcessHandler::setup`.
    pub fn setup(&mut self, processchunks: &mut Processchunks, gpu_num: i32) {
        self.processchunks = processchunks;
        self.escaped_remote_dir_path = processchunks.get_remote_dir().replace(' ', "\\ ");
        self.gpu_number = gpu_num;
        if processchunks.is_queue() {
            //Queue command
            //finishes after putting things into the queue
            //$queuecom -w "$curdir" -a R $comname:r
            self.command = processchunks.get_queue_command().to_owned();
        } else {
            //Local host command
            self.command = "python".to_owned();
        }
        self.init_process();
    }

    /// C++ `ProcessHandler::setJob`.
    pub fn set_job(&mut self, job_index: i32) {
        if self.valid_job {
            unsafe {
                (*self.processchunks).write_out(&format!(
                    "ERROR: Unable to set job, process handler already contains a runnable job:{},mComFileJobIndex:{}\n",
                    (*self.processchunks)
                        .get_com_file_jobs()
                        .get_com_file_name(self.com_file_job_index as usize),
                    self.com_file_job_index
                ));
            }
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
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        // Do not reset the flag if chunk marked for skipping
        if self.get_flag() != CHUNK_TO_SKIP {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs_mut()
                    .set_flag_not_done(self.com_file_job_index as usize, single_file);
            }
        }
    }

    /// C++ `ProcessHandler::resetSignalValues`.
    pub fn reset_signal_values(&mut self) {
        self.finished_signal_received = false;
        self.error_signal_received = false;
        self.exit_code = -1;
        self.exit_status = -1;
        self.process_error = None;
        self.log_has_error = false;
    }

    /// C++ `ProcessHandler::getFlag`.
    pub fn get_flag(&self) -> i32 {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
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
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return false;
        }
        if self.log_file_exists {
            return true;
        }
        let Some(log_file) = self.log_file.clone() else {
            return false;
        };
        if !newly_created_file {
            //Don't set mLogFileExists when newlyCreateFile is false because ls may need
            //to be run (with handleFileSystemBug) later - the file may be backed up.
            return log_file.exists();
        }
        //Set mLogFileExists.  Run ls (with handleFileSystemBug) if necessary.
        self.log_file_exists = log_file.exists();
        if !self.log_file_exists && newly_created_file {
            unsafe {
                if (*self.processchunks).is_queue()
                    && (self.exit_code != 0 || self.exit_status != 0)
                {
                    self.log_file_exists = false;
                } else {
                    (*self.processchunks)
                        .handle_file_system_bug(&format!("see {}", log_file.to_string_lossy()));
                    self.log_file_exists = log_file.exists();
                }
            }
        }
        self.log_file_exists
    }

    /// C++ `ProcessHandler::qidFileExists`.
    pub fn qid_file_exists(&self) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return false;
        }
        unsafe {
            if (*self.processchunks).is_queue() {
                return self.qid_file.as_ref().is_some_and(|path| path.exists());
            }
        }
        true
    }

    //Looks for PID in either stderr (non-queue) or in .qid file (queue).
    /// C++ `ProcessHandler::getPid`.
    pub fn get_pid(&mut self) -> String {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return String::new();
        }
        if unsafe { !(*self.processchunks).is_queue() } {
            self.read_all_standard_error();
            let text_stream = self.stderr.clone();
            self.get_pid_from_bytes(&text_stream, true);
        } else {
            let Some(contents) = self.qid_file.as_ref().and_then(|path| fs::read(path).ok()) else {
                return self.pid.clone();
            };
            self.get_pid_from_bytes(&contents, true);
        }
        self.pid.clone()
    }

    /// Returns true if there is a pid in stderr.  Always returns fase if queue
    /// is set.
    /// C++ `ProcessHandler::isPidInStderr`.
    pub fn is_pid_in_stderr(&mut self) -> bool {
        if unsafe { (*self.processchunks).is_queue() } {
            return false;
        }
        self.read_all_standard_error();
        let text_stream = self.stderr.clone();
        self.get_pid_from_bytes(&text_stream, false)
    }

    /// return true if the pid is found in stream.  If save is true, save the
    /// pid to mPid; in this case only check for the pid if mPid is empty.
    /// C++ private `ProcessHandler::getPid(QTextStream &, const bool)`.
    pub fn get_pid_from_bytes(&mut self, stream: &[u8], save: bool) -> bool {
        if save && !self.pid.is_empty() {
            //Don't look for the PID more then once
            return true;
        }
        //Don't set the PID unless the line is conplete (includes an EOL).  Process
        //may not be finished when this function runs.
        let output = String::from_utf8_lossy(stream);
        //Look for a PID entry with an EOL so the the complete PID is collected
        if let Some(index) = output.rfind("PID:") {
            if let Some(offset) = output[index..].find('\n') {
                let end_index = index + offset;
                if save {
                    self.pid = output[index + 4..end_index].trim().to_owned();
                }
                return true;
            }
        }
        false
    }

    /// C++ `ProcessHandler::readAllLogFile`.
    pub fn read_all_log_file(&self) -> Vec<u8> {
        let log = Vec::new();
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return log;
        }
        self.log_file
            .as_ref()
            .and_then(|path| fs::read(path).ok())
            .unwrap_or(log)
    }

    /// C++ `ProcessHandler::isLogFileEmpty`.
    pub fn is_log_file_empty(&self) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return true;
        }
        // `QFile::size()` is 0 for a file that does not exist.
        self.log_file
            .as_ref()
            .and_then(|path| fs::metadata(path).ok())
            .map_or(0, |metadata| metadata.len())
            == 0
    }

    /// Test whether the log file is older than the given timeout valu.
    /// Refreshs the stored last modified time and size only if it is older
    /// than the timeout or if it hasn't been checked for 1/5 of the timeout
    /// interval.
    /// C++ `ProcessHandler::isLogFileOlderThan`.
    pub fn is_log_file_older_than(&mut self, timeout_sec: i32) -> bool {
        if timeout_sec <= 0 {
            return false;
        }
        let now = SystemTime::now();
        // `QDateTime::secsTo(now)`, which is signed.
        let secs_to = |from: SystemTime| -> i64 {
            match now.duration_since(from) {
                Ok(span) => span.as_secs() as i64,
                Err(error) => -(error.duration().as_secs() as i64),
            }
        };
        let modified_ok = secs_to(self.log_last_modified) < timeout_sec as i64;
        if modified_ok && secs_to(self.last_size_check_time) < (timeout_sec / 5) as i64 {
            return false;
        }
        let Some(metadata) = self
            .log_file
            .as_ref()
            .and_then(|path| fs::metadata(path).ok())
        else {
            return false;
        };
        let new_size = metadata.len() as i64;
        self.last_size_check_time = now;
        if new_size > self.last_log_size {
            self.last_log_size = new_size;
            self.size_changed_time = now;
        }
        self.log_last_modified = metadata.modified().unwrap_or(now);
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "isLogFileOlderThan", 1)
            {
                (*self.processchunks).write_out(&format!(
                    "{}:isLogFileOlderThan: new mod {}  size time {}\n",
                    self.decorated_class_name,
                    secs_to(self.log_last_modified),
                    secs_to(self.size_changed_time)
                ));
            }
        }
        secs_to(self.log_last_modified) >= timeout_sec as i64
            && secs_to(self.size_changed_time) >= timeout_sec as i64
    }

    /// C++ private `ProcessHandler::readAllStandardError`.
    pub fn read_all_standard_error(&mut self) {
        let err = match self.stderr_buffer.as_ref() {
            Some(buffer) => match buffer.lock() {
                Ok(mut buffer) => std::mem::take(&mut *buffer),
                Err(_) => Vec::new(),
            },
            None => Vec::new(),
        };
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

    //Looks for cd or ssh error in either stdout/stderr (non-queue) or
    //.job file (queue).
    //Returns true if found
    /// C++ `ProcessHandler::getSshError`.
    pub fn get_ssh_error(&mut self, drop_mess: &mut String) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return false;
        }
        let mut found = false;
        if unsafe { !(*self.processchunks).is_queue() } {
            self.read_all_standard_error();
            let text_stream = self.stderr.clone();
            found = self.get_ssh_error_from_bytes(drop_mess, &text_stream);
        } else {
            let Some(job_file) = self.job_file.clone() else {
                return found;
            };
            if !job_file.exists() {
                return found;
            }
            if fs::metadata(&job_file).map_or(0, |metadata| metadata.len()) == 0 {
                return found;
            }
            let Ok(contents) = fs::read(&job_file) else {
                return found;
            };
            found = self.get_ssh_error_from_bytes(drop_mess, &contents);
        }
        found
    }

    /// C++ private `ProcessHandler::getSshError(QString &, QTextStream &)`.
    pub fn get_ssh_error_from_bytes(&self, drop_mess: &mut String, stream: &[u8]) -> bool {
        //look for cd error & ssh error
        for line in String::from_utf8_lossy(stream).lines() {
            if line.contains("cd: ") {
                *drop_mess = format!("it cannot cd to {} ({line})", unsafe {
                    (*self.processchunks).get_remote_dir()
                });
                return true;
            } else if line.contains("ssh: connect to host") {
                *drop_mess = format!("cannot connect ({line})");
                return true;
            }
        }
        false
    }

    //True when the .com file has been run and it has finished.  Returns true if
    //the log exists and the finished signal has been received
    /// C++ `ProcessHandler::isComProcessDone`.
    pub fn is_com_process_done(&mut self) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return false;
        }
        self.deliver_process_signals();
        if unsafe { (*self.processchunks).is_queue() } {
            if self.exit_code != 0 || self.exit_status != 0 {
                return true;
            }
            let py_exists = self.py_file_exists();
            if !py_exists && self.log_file_exists(true) {
                self.machine = std::ptr::null_mut();
                return true;
            } else {
                if !py_exists {
                    // If the ".py" file is not there, start timer first time this
                    // occurs, increment count, and say the process is done if enough time has
                    // elapsed and the log was checked enough times
                    if self.num_did_not_see_log == 0 {
                        self.no_log_start_time = Instant::now();
                    }
                    self.num_did_not_see_log += 1;
                    if self.no_log_start_time.elapsed().as_millis() as i64
                        > 1000 * self.sec_to_wait_if_no_log as i64
                        && self.num_did_not_see_log > self.max_num_no_log_seen
                    {
                        return true;
                    }

                    // But if the file DOES show up, zero the count, so we can start the timer
                    // again from when it disappears
                } else {
                    self.num_did_not_see_log = 0;
                }
                return false;
            }
        }
        self.finished_signal_received
            && self.log_file_exists(true)
            && self.get_flag() != CHUNK_NOT_DONE
    }

    //Returns true if a last line of the log file starts with "CHUNK DONE"
    /// C++ `ProcessHandler::isChunkDone`.
    pub fn is_chunk_done(&mut self) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("ERROR: Job index not set\n");
            }
            return false;
        }
        let Some(log_file) = self.log_file.clone() else {
            return false;
        };
        let contents = match fs::read(&log_file) {
            Ok(contents) => contents,
            Err(_) => {
                if self.log_file_exists(true) && fs::read(&log_file).is_err() {
                    unsafe {
                        (*self.processchunks).write_out(&format!(
                            "Warning: Unable to open {}\n",
                            log_file.to_string_lossy()
                        ));
                    }
                }
                return false;
            }
        };
        let size = contents.len();
        let size_to_check = 512; // This was 25 before scanning for ERROR
        //Attempt to seek to the last line of the file (if the file is larger
        //then 512 characters).  If the seek fails, the whole file will have to be
        //looked at.
        let start = if size > size_to_check {
            size - size_to_check
        } else {
            0
        };
        let last_part_of_file = String::from_utf8_lossy(&contents[start..])
            .trim_matches(|character: char| character.is_ascii_whitespace())
            .to_owned();
        let done = last_part_of_file.ends_with("CHUNK DONE");
        self.log_has_error = last_part_of_file.contains("ERROR:");
        done
    }

    // Pause occurs if process is done but CHUNK DONE not found, unless the exit code is
    // nonzero and ERROR: was found in the log
    // Start a timer on first call and do not return false until enough time is elapsed
    /// C++ `ProcessHandler::isPausing`.
    pub fn is_pausing(&mut self) -> bool {
        if self.pausing != 0 {
            return self.pause_time.elapsed() <= Duration::from_secs(1);
        } else {
            if self.exit_code > 0 && self.log_has_error {
                return false;
            }
            self.pausing += 1;
            self.pause_time = Instant::now();
        }
        true
    }

    //Reads the last 1000 characters of the file.  Returns all the text between
    //with "ERROR:" and the end of the file.
    /// C++ `ProcessHandler::getErrorMessageFromLog`.
    pub fn get_error_message_from_log(&self, error_mess: &mut String) {
        let eol = "\n";
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        let Some(contents) = self.log_file.as_ref().and_then(|path| fs::read(path).ok()) else {
            return;
        };
        let size = contents.len();
        if size == 0 {
            return;
        }
        let size_to_check = 1000;
        let start = if size > size_to_check {
            size - size_to_check
        } else {
            0
        };
        let tail = String::from_utf8_lossy(&contents[start..]).into_owned();
        let mut line = "";
        let mut error_found = false;
        for next in tail.lines() {
            line = next;
            if error_found || line.contains("ERROR:") {
                error_found = true;
                error_mess.push_str(line);
                error_mess.push_str(eol);
            }
        }
        if error_mess.is_empty() {
            error_mess.push_str("CHUNK ERROR: (last line) - ");
            error_mess.push_str(line);
            error_mess.push_str(eol);
        } else {
            error_mess.insert_str(0, "CHUNK ");
        }
        error_mess.push_str("END CHUNK ERROR");
    }

    //Reads the last lines of and stderr and appends then to errorMess..
    /// C++ `ProcessHandler::getErrorMessageFromOutput`.
    pub fn get_error_message_from_output(&mut self, error_mess: &mut String) {
        //Use the last lines of and stderr as the error message if the log
        //file is empty.
        let eol = "\n";
        error_mess.push_str(eol);
        //stderr
        self.read_all_standard_error();
        let stderr = String::from_utf8_lossy(&self.stderr).into_owned();
        let output = stderr.trim_matches(|character: char| character.is_ascii_whitespace());
        if !output.is_empty() {
            // `QByteArray::mid(lastLineIndex)` keeps the end-of-line itself.
            if let Some(last_line_index) = output.rfind(eol) {
                error_mess.push_str(&output[last_line_index..]);
                error_mess.push_str(eol);
            } else {
                error_mess.push_str(&stderr);
            }
        }
    }

    // imodpy runProcesschunks is looking for this exact text string with a 1
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
                    self.decorated_class_name, self.exit_code, self.exit_status
                ));
            }
        }
    }

    /// C++ `ProcessHandler::incrementNumChunkErr`.
    pub fn increment_num_chunk_err(&mut self) {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs_mut()
                .increment_num_chunk_err(self.com_file_job_index as usize);
        }
    }

    /// C++ `ProcessHandler::printWarnings`.
    pub fn print_warnings(&self, machine_name: &str) {
        let mut warn_list: Vec<String> = Vec::new();
        let mut num_warns: Vec<i32> = Vec::new();
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        let Some(contents) = self.log_file.as_ref().and_then(|path| fs::read(path).ok()) else {
            return;
        };
        let text = String::from_utf8_lossy(&contents).into_owned();
        let mut reader = text.lines();
        // `QString line = mLogFile->readLine();` before the loop: the first
        // line of the log is read and discarded.
        reader.next();
        for next in reader {
            let mut line = next.to_owned();
            if line.contains("WARNING:") {
                // Keep track of matching warnings on the list
                line = line.trim().to_owned();
                let mut matched = false;
                let mut index = 0;
                while index < warn_list.len() {
                    if line == warn_list[index] {
                        matched = true;
                        break;
                    }

                    // Look for a match up to the last space and replace final word with ...
                    let line_ind = line.rfind(' ');
                    let warn_ind = warn_list[index].rfind(' ');
                    matched = match (line_ind, warn_ind) {
                        (Some(line_ind), Some(warn_ind)) => {
                            line_ind > 0
                                && line_ind == warn_ind
                                && line[..line_ind] == warn_list[index][..warn_ind]
                        }
                        _ => false,
                    };
                    if matched {
                        let warn_ind = warn_list[index].rfind(' ').unwrap();
                        warn_list[index] = format!("{} ...", &warn_list[index][..warn_ind]);
                        break;
                    }
                    index += 1;
                }
                if matched {
                    num_warns[index] += 1;
                } else {
                    num_warns.push(1);
                    warn_list.push(line);
                }
            } else if line.contains("MESSAGE:") {
                unsafe {
                    (*self.processchunks)
                        .write_out(&format!("{} - on {machine_name}\n", line.trim()));
                }
            } else if line.contains("LOGFILE:") {
                // Assume files to be logged will not be duplicated by the caller.
                unsafe {
                    (*self.processchunks).write_out(&format!("{}\n", line.trim()));
                }
            }
        }
        for index in 0..warn_list.len() {
            unsafe {
                (*self.processchunks).write_out(&warn_list[index]);
                if num_warns[index] > 1 {
                    (*self.processchunks).write_out(&format!(" ({} times)", num_warns[index]));
                }
                (*self.processchunks).write_out("\n");
            }
        }
    }

    /// C++ inline `ProcessHandler::backupLog`.
    pub fn backup_log(&self) {
        let Some(log_file) = &self.log_file else {
            return;
        };
        imod_backup_file(&log_file.to_string_lossy());
    }

    /// C++ `ProcessHandler::pyFileExists`.
    pub fn py_file_exists(&self) -> bool {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return false;
        }
        PathBuf::from(unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_py_file_name(self.com_file_job_index as usize)
        })
        .exists()
    }

    /// C++ `ProcessHandler::getNumChunkErr`.
    pub fn get_num_chunk_err(&self) -> i32 {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return 0;
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_num_chunk_err(self.com_file_job_index as usize)
        }
    }

    /// C++ `ProcessHandler::getComFileName`.
    pub fn get_com_file_name(&self) -> String {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return String::new();
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_com_file_name(self.com_file_job_index as usize)
        }
    }

    /// C++ `ProcessHandler::getLogFileName`.
    pub fn get_log_file_name(&self) -> String {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return String::new();
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_log_file_name(self.com_file_job_index as usize)
        }
    }

    //Returns true if the process has started, but the log file hasn't been create,
    //and the timeout (milliseconds) has been exceeded.
    /// C++ `ProcessHandler::isStartProcessTimedOut`.
    pub fn is_start_process_timed_out(&mut self, timeout: i32) -> bool {
        //If a process isn't running then there is nothing to timeout
        //If a process is running and the log file has been created then the log file
        //was created before the timeout
        if !self.starting_process {
            return false;
        }
        if self.start_time.elapsed() <= Duration::from_millis(timeout as u64) {
            return false;
        }
        !self.log_file_exists(true)
    }

    /// C++ `ProcessHandler::setFlag`.
    pub fn set_flag(&mut self, flag: i32) {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        // Only set the flag if the chunk is not already marked for skipping
        if self.get_flag() != CHUNK_TO_SKIP {
            unsafe {
                (*self.processchunks)
                    .get_com_file_jobs_mut()
                    .set_flag(self.com_file_job_index as usize, flag);
            }
        }
    }

    /// C++ `ProcessHandler::removeFiles`.
    pub fn remove_files(&mut self) {
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
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
            /* CSH -> PY  make sure there is no .csh file to confuse queuechunk */
            unsafe {
                let _ = fs::remove_file(format!(
                    "{}.csh",
                    (*self.processchunks)
                        .get_com_file_jobs()
                        .get_root(self.com_file_job_index as usize)
                ));
            }
        } else {
            self.stderr.clear();
            self.pid.clear();
        }
    }

    /// C++ `ProcessHandler::getPyFile`.
    pub fn get_py_file(&self) -> String {
        let temp = String::new();
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return temp;
        }
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs()
                .get_py_file_name(self.com_file_job_index as usize)
        }
    }

    //Run the process.
    /// C++ `ProcessHandler::runProcess`.
    pub fn run_process(&mut self, machine: &mut MachineHandler) {
        /*There is no way to undo the functionality preceding the running of a
        process.  The process MUST be run before exiting this function unless there
        is a serious error.*/
        //Don't run a job that has been reset.
        if !self.valid_job {
            unsafe {
                (*self.processchunks).write_out(&format!(
                    "Processchunks warning: Job is not runnable,index={}\n",
                    self.com_file_job_index
                ));
            }
            return;
        }
        self.machine = machine;
        //Build command if necessary
        let mut command: Option<String> = None;
        let mut param_list: Option<Vec<String>> = None;
        let queue = unsafe { (*self.processchunks).is_queue() };
        if !queue {
            //It is not working to run processes using csh on Windows.
            if unsafe { !(*self.processchunks).name_is_local_host(machine.get_name()) } {
                //Original command:
                //ssh -x $sshopts $machname bash --login -c \'"cd $curdir && (csh -ef < $cshname >& $pidname ; \rm -f $cshname)"\' >&! $sshname &
                command = Some("ssh".to_owned());
                //To run remote command: bash --login -c '"command"'
                //Escape spaces in the directory path
                //Escaping the single quote shouldn't be necessary because this is not
                //being run from a shell.
                // DNM Note: the rm prevents the ssh from passing on the exit status of the script
                // So now processchunks removes .py for nonlocal jobs too
                let param = format!(
                    "\"cd {} && python -u < {}\"",
                    self.escaped_remote_dir_path,
                    unsafe {
                        (*self.processchunks)
                            .get_com_file_jobs()
                            .get_py_file_name(self.com_file_job_index as usize)
                    }
                );
                let mut list = vec!["-x".to_owned()];
                let ssh_opts = unsafe { (*self.processchunks).get_ssh_opts() };
                for opt in ssh_opts {
                    list.push(opt.clone());
                }
                list.push(machine.get_name().to_owned());
                list.push("bash".to_owned());
                list.push("--login".to_owned());
                list.push("-c".to_owned());
                list.push(param);
                param_list = Some(list);
            }
        }
        /*Hook stdin to a file to avoid excessive pipes - this is necessary to keep
        the pipes per process down to 4.  If stdout was no going to a file, the
        process pipe count would be 6.  In that case the total number of CPUs
        allowed (Processchunks::setupMachineList::numCpusLimit) would have to be
        reduced.*/
        let mut process = match &command {
            Some(command) => Command::new(command),
            None => Command::new(&self.command),
        };
        match &param_list {
            Some(param_list) => {
                process.args(param_list);
            }
            None => {
                process.args(&self.param_list);
            }
        }
        if !queue {
            if let Ok(file) = fs::File::create(format!("{}.stdout", unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_py_file_name(self.com_file_job_index as usize)
            })) {
                process.stdout(Stdio::from(file));
            }
            if let Ok(file) = fs::File::open(unsafe {
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_py_file_name(self.com_file_job_index as usize)
            }) {
                process.stdin(Stdio::from(file));
            }
        } else {
            process.stdout(Stdio::piped());
        }
        process.stderr(Stdio::piped());
        //Run command
        self.reset_signal_values();
        self.init_process();
        match process.spawn() {
            Ok(child) => self.process = Some(child),
            // Qt's `QProcess` reports a failed start through `error`, not by
            // refusing to start: the slot runs and the object stays valid.
            // The child's own `execvp: <strerror>` text is what reaches
            // `errorString` (`qprocess_unix.cpp`'s startup pipe).
            Err(error) => {
                let text = error.to_string();
                self.error_string = format!(
                    "execvp: {}",
                    match text.find(" (os error ") {
                        Some(index) => &text[..index],
                        None => &text[..],
                    }
                );
                self.handle_error(ProcessError::FailedToStart);
            }
        }
        if let Some(process) = &mut self.process {
            // Qt's socket notifiers drain a QProcess's channels into its own
            // buffer as the child writes, so the child never blocks on a full
            // pipe and `readAllStandardError` never waits.
            if let Some(mut stderr) = process.stderr.take() {
                let buffer = Arc::new(Mutex::new(Vec::new()));
                let writer = Arc::clone(&buffer);
                std::thread::spawn(move || {
                    use std::io::Read;
                    let mut chunk = [0u8; 4096];
                    while let Ok(count) = stderr.read(&mut chunk) {
                        if count == 0 {
                            break;
                        }
                        if let Ok(mut slot) = writer.lock() {
                            slot.extend_from_slice(&chunk[..count]);
                        }
                    }
                });
                self.stderr_buffer = Some(buffer);
            }
            if let Some(mut stdout) = process.stdout.take() {
                let buffer = Arc::new(Mutex::new(Vec::new()));
                let writer = Arc::clone(&buffer);
                std::thread::spawn(move || {
                    use std::io::Read;
                    let mut chunk = [0u8; 4096];
                    while let Ok(count) = stdout.read(&mut chunk) {
                        if count == 0 {
                            break;
                        }
                        if let Ok(mut slot) = writer.lock() {
                            slot.extend_from_slice(&chunk[..count]);
                        }
                    }
                });
                self.stdout_buffer = Some(buffer);
            }
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
        // `mProcess->closeWriteChannel()` is the `Stdio` the child was given.
        b3d_milli_sleep(unsafe { (*self.processchunks).get_millisec_sleep() });
        if queue {
            // `mProcess->waitForFinished(60000)`, which also delivers the
            // `finished` signal before this function returns.
            let deadline = Instant::now() + Duration::from_millis(60000); // 2/9/16: THIS USED TO BE 2000
            loop {
                let Some(process) = &mut self.process else {
                    break;
                };
                if matches!(process.try_wait(), Ok(Some(_))) {
                    break;
                }
                if Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            self.deliver_process_signals();
        }
        //Turn on running process boolean and record start time
        self.starting_process = true;
        self.start_time = Instant::now();
    }

    /// C++ inline `ProcessHandler::isFinishedSignalReceived`.
    pub fn is_finished_signal_received(&mut self) -> bool {
        self.deliver_process_signals();
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

    // Save flage when initiating kill.  Single kill job should still be valid, but don't
    // ignore it regardless
    /// C++ `ProcessHandler::startKill`.
    pub fn start_kill(&mut self, kill_one: bool) {
        self.kill = true;
        self.ignore_kill = !kill_one && !self.is_job_valid();
        self.killing_one = kill_one;
    }

    /*
    Handles the kill signal when imodkillgroup isn't used.  If this a queue and
    the kill request has already been sent, this process waits for it to finish.
    Currently imodkillgroup is used for everything but queues
    */
    /// C++ `ProcessHandler::killSignal`.
    pub fn kill_signal(&mut self) {
        if self.ignore_kill
            || (self.kill_finished_signal_received && self.finished_signal_received)
            || unsafe { !(*self.processchunks).is_queue() }
        {
            return;
        }
        if !self.kill_started {
            self.kill_started = true; //This starts the 15-count timeout
            self.set_job_not_done();
            unsafe {
                (*self.processchunks).increment_kills();
            }
            //Kill the process
            let ans = unsafe { (*self.processchunks).get_ans() };
            if ans != 'P' || !self.log_file_exists(false) {
                let action = if ans == 'P' { "P" } else { "K" };
                //Don't know if this waits until the kill is does
                //$queuecom -w "$curdir" -a $action $comlist[$ind]:r
                //The second to last parameter is the action letter
                let replace_at = self.param_list.len() - 2;
                self.param_list[replace_at] = action.to_owned();
                // `mKillProcess` has `QProcess::ForwardedChannels`, so both of
                // its channels go straight to this program's own.
                let spawned = Command::new(&self.command)
                    .args(&self.param_list)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn();
                match spawned {
                    Ok(child) => self.kill_process = Some(child),
                    Err(_) => self.kill_process = None,
                }
                // `mKillProcess->waitForFinished(1000)`, which also delivers
                // the `finished` signal to `handleKillFinished`.
                let deadline = Instant::now() + Duration::from_millis(1000);
                let mut delivered = None;
                loop {
                    let Some(kill_process) = &mut self.kill_process else {
                        break;
                    };
                    if let Ok(Some(status)) = kill_process.try_wait() {
                        delivered = Some((
                            status.code().unwrap_or(0),
                            if status.code().is_some() {
                                ProcessExitStatus::NormalExit
                            } else {
                                ProcessExitStatus::CrashExit
                            },
                        ));
                        break;
                    }
                    if Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                if let Some((exit_code, exit_status)) = delivered {
                    self.handle_kill_finished(exit_code, exit_status);
                }
                //Put mParamList back to its regular form
                self.param_list[replace_at] = "R".to_owned();
            }
        } else {
            //Waiting for kill to finish
            self.kill_counter += 1;
            if self.kill_counter > 15
                && (!self.kill_finished_signal_received || !self.finished_signal_received)
            {
                if let Some(kill_process) = &mut self.kill_process {
                    let _ = kill_process.kill();
                }
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
        self.get_pid();
        self.pid.is_empty()
    }

    /// C++ `ProcessHandler::setJobNotDone`.
    pub fn set_job_not_done(&mut self) {
        unsafe {
            (*self.processchunks)
                .get_com_file_jobs_mut()
                .set_flag(self.com_file_job_index as usize, CHUNK_NOT_DONE);
        }
    }

    /// C++ `ProcessHandler::killQProcesses`.
    pub fn kill_q_processes(&mut self) {
        if let Some(process) = &mut self.process {
            let _ = process.kill();
        }
        if let Some(kill_process) = &mut self.kill_process {
            let _ = kill_process.kill();
        }
    }

    //Sets signal variables.  For a non-queue removes the .py file.
    //If the process was killed, tell processchunks that its done.
    /// C++ slot `ProcessHandler::handleFinished`.
    pub fn handle_finished(&mut self, exit_code: i32, exit_status: ProcessExitStatus) {
        let exit_status = match exit_status {
            ProcessExitStatus::NormalExit => 0,
            ProcessExitStatus::CrashExit => 1,
        };
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "handleFinished", 1) {
                let _ = ImodFile::Stdout.write_all(
                    format!(
                        "{}:handleFinished:{exit_code},exitStatus:{exit_status}\n",
                        self.decorated_class_name
                    )
                    .as_bytes(),
                );
            }
        }
        if self.com_file_job_index == -1 {
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
        if unsafe { !(*self.processchunks).is_queue() } {
            unsafe {
                let _ = fs::remove_file(
                    (*self.processchunks)
                        .get_com_file_jobs()
                        .get_py_file_name(self.com_file_job_index as usize),
                );
                let _ = fs::remove_file(format!(
                    "{}.stdout",
                    (*self.processchunks)
                        .get_com_file_jobs()
                        .get_py_file_name(self.com_file_job_index as usize)
                ));
            }
            if !self.kill {
                self.machine = std::ptr::null_mut();
            }
        } else if self.exit_code != 0 || self.exit_status != 0 {
            // Get error messages out when queue submission fails
            let byte_array = match self.stderr_buffer.as_ref() {
                Some(buffer) => match buffer.lock() {
                    Ok(mut buffer) => std::mem::take(&mut *buffer),
                    Err(_) => Vec::new(),
                },
                None => Vec::new(),
            };
            if !byte_array.is_empty() {
                unsafe {
                    (*self.processchunks)
                        .write_out(&format!("{}\n", String::from_utf8_lossy(&byte_array)));
                }
            }
            let byte_array = match self.stdout_buffer.as_ref() {
                Some(buffer) => match buffer.lock() {
                    Ok(mut buffer) => std::mem::take(&mut *buffer),
                    Err(_) => Vec::new(),
                },
                None => Vec::new(),
            };
            if !byte_array.is_empty() {
                let com_split: Vec<&str> = self.command.split(' ').collect();
                unsafe {
                    if !com_split.is_empty() {
                        (*self.processchunks).write_out(&format!("{} ", com_split[0]));
                    }
                    (*self.processchunks)
                        .write_out(&format!("{}\n", String::from_utf8_lossy(&byte_array)));
                }
            }
        }
    }

    /// C++ slot `ProcessHandler::handleKillFinished`.
    pub fn handle_kill_finished(&mut self, exit_code: i32, exit_status: ProcessExitStatus) {
        if self.com_file_job_index == -1 {
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
                // `mKillProcess` forwards its channels, so `readAllStandardError`
                // on it is always empty.
                let byte_array: Vec<u8> = Vec::new();
                if !byte_array.is_empty() {
                    unsafe {
                        (*self.processchunks)
                            .write_out(&format!("{}\n", String::from_utf8_lossy(&byte_array)));
                    }
                }
            }
        }
    }

    /// C++ slot `ProcessHandler::handleError`.
    pub fn handle_error(&mut self, process_error: ProcessError) {
        if self.kill {
            // local tree killing was used - causes a return code of 1
            return;
        }
        if self.com_file_job_index == -1 {
            unsafe {
                (*self.processchunks).write_out("Processchunks warning: Job index not set\n");
            }
            return;
        }
        self.error_signal_received = true;
        self.process_error = Some(process_error);
        unsafe {
            (*self.processchunks).write_out(&format!(
                "{}:process error:{},{}\n",
                (*self.processchunks)
                    .get_com_file_jobs()
                    .get_com_file_name(self.com_file_job_index as usize),
                process_error as i32,
                self.error_string
            ));
            // A process that never started reports exit code 0, NormalExit and
            // NotRunning.
            (*self.processchunks).write_out("exitCode:0,QtExitStatus:0,state:0\n");
        }
    }

    // Close the process; used if there is a timeout and no PID to kill it with
    /// C++ `ProcessHandler::closeProcess`.
    pub fn close_process(&mut self) {
        if self.starting_process {
            // `QProcess::close()` kills the child and waits for it.
            if let Some(process) = &mut self.process {
                let _ = process.kill();
                let _ = process.wait();
            }
        }
    }
}

// 6/17/13: Removed killLocalProcessAndDescendents and StopProcess
