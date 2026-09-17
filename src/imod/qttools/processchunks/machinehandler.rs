//! `IMOD/qttools/processchunks/machinehandler.h` and
//! `IMOD/qttools/processchunks/machinehandler.cpp`.
//!
//! A description of one computer or queue and the `ProcessHandler`s assigned
//! to its CPUs.  `processchunks` owns `MachineHandler`, as it does in C++; the
//! raw pointer below is the direct translation of the non-owning
//! `Processchunks *mProcesschunks` member.

#![allow(dead_code)]

use super::processchunks::Processchunks;
use super::processhandler::ProcessHandler;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

/// Qt `QProcess::ExitStatus`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessExitStatus {
    NormalExit,
    CrashExit,
}

/// Qt `QProcess::ProcessError`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessError {
    FailedToStart,
    Crashed,
    Timedout,
    WriteError,
    ReadError,
    UnknownError,
}

/// C++ `MachineHandler`.
pub struct MachineHandler {
    process_handler_array: Vec<ProcessHandler>,
    name: String,
    decorated_class_name: String,
    num_cpus: i32,
    failure_count: i32,
    slowest_time: i32,
    slow_time_count: i32,
    kill: bool,
    chunk_erred: bool,
    dropped: bool,
    internal_dropped: bool,
    processchunks: *mut Processchunks,
    full_num_cpus: i32,
    job_machine_lists: Vec<String>,
    job_thread_limits: Vec<i32>,
    ignore_kill: bool,
    kill_finished_signal_received: bool,
    kill_started: bool,
    pids_available: bool,
    kill_warning: bool,
    killing_one: bool,
    kill_counter: i32,
    pid_wait_counter: i32,
    one_pid_to_kill: String,
    kill_process: Option<Child>,
}

impl MachineHandler {
    /// `MachineHandler()` source constructor.  Its paired C++ destructor is
    /// represented by Rust ownership of `process_handler_array` and `kill_process`.
    /// C++ `MachineHandler::MachineHandler`.
    pub fn new() -> MachineHandler {
        MachineHandler {
            process_handler_array: Vec::new(),
            name: String::new(),
            decorated_class_name: "MachineHandler".to_owned(),
            num_cpus: 0,
            failure_count: 0,
            slowest_time: -1,
            slow_time_count: 0,
            kill: false,
            chunk_erred: false,
            dropped: false,
            internal_dropped: false,
            processchunks: std::ptr::null_mut(),
            full_num_cpus: 0,
            job_machine_lists: Vec::new(),
            job_thread_limits: Vec::new(),
            ignore_kill: true,
            kill_finished_signal_received: false,
            kill_started: false,
            pids_available: false,
            kill_warning: false,
            killing_one: false,
            kill_counter: 0,
            pid_wait_counter: 0,
            one_pid_to_kill: String::new(),
            kill_process: None,
        }
    }

    /// C++ private `MachineHandler::init`.  It is declared in the header but
    /// has no IMOD definition.
    fn init(&mut self) {
        panic!("MachineHandler::init has no definition in IMOD source")
    }

    /// C++ private no-argument `MachineHandler::setup`.  Rust has no overloads;
    /// the suffix records its otherwise identical source name.  It is declared
    /// in the header but has no IMOD definition.
    fn setup_without_arguments(&mut self) {
        panic!("MachineHandler::setup() has no definition in IMOD source")
    }

    /// C++ `MachineHandler::setup(Processchunks &, const QString &, const int,
    /// const IntVec &, const int)`.
    pub fn setup(
        &mut self,
        processchunks: &mut Processchunks,
        machine_name: &str,
        num_cpus: i32,
        gpu_list: &[i32],
        base_index: usize,
    ) {
        let gpu_mode = processchunks.get_gpu_mode();
        self.name = machine_name.to_owned();
        self.num_cpus = num_cpus;
        self.full_num_cpus = num_cpus;
        self.processchunks = processchunks as *mut Processchunks;
        self.process_handler_array.clear();
        for i in 0..num_cpus as usize {
            let mut process_handler = ProcessHandler::new();
            process_handler.setup(
                processchunks,
                if gpu_mode {
                    gpu_list[base_index + i]
                } else {
                    -1
                },
            );
            self.process_handler_array.push(process_handler);
        }
    }

    /// C++ `MachineHandler::setValues`.
    pub fn set_values(&mut self, machine_name: &str, num_cpus: i32) {
        self.name = machine_name.to_owned();
        self.num_cpus = num_cpus;
    }

    /// C++ inline `MachineHandler::nameToLong`.
    pub fn name_to_long(&self, ok: &mut bool) -> i64 {
        match self.name.parse::<i64>() {
            Ok(value) => {
                *ok = true;
                value
            }
            Err(_) => {
                *ok = false;
                0
            }
        }
    }

    /// C++ inline `MachineHandler::incrementNumCpus`.
    pub fn increment_num_cpus(&mut self) {
        self.num_cpus += 1;
    }

    /// C++ inline `MachineHandler::getName`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// C++ inline `MachineHandler::getNumCpus`.
    pub fn get_num_cpus(&self) -> i32 {
        self.num_cpus
    }

    /// C++ inline `MachineHandler::getSlowestTime`.
    pub fn get_slowest_time(&self) -> i32 {
        self.slowest_time
    }

    /// C++ inline `MachineHandler::getSlowTimeCount`.
    pub fn get_slow_time_count(&self) -> i32 {
        self.slow_time_count
    }

    /// C++ `MachineHandler::setSlowestTime`.
    pub fn set_slowest_time(&mut self, new_time: i32) {
        if new_time < 0 || new_time > self.slowest_time {
            self.slowest_time = new_time;
        }
        if new_time < 0 {
            self.slow_time_count = 0;
        } else {
            self.slow_time_count += 1;
        }
    }

    /// C++ inline `MachineHandler::getProcessHandler`.
    pub fn get_process_handler(&mut self, index: usize) -> &mut ProcessHandler {
        &mut self.process_handler_array[index]
    }

    /// C++ `MachineHandler::getFailureCount`.
    pub fn get_failure_count(&self) -> i32 {
        if self.dropped { 0 } else { self.failure_count }
    }

    /// C++ inline `MachineHandler::isChunkErred`.
    pub fn is_chunk_erred(&self) -> bool {
        self.chunk_erred
    }

    /// C++ inline `MachineHandler::setFailureCount`.
    pub fn set_failure_count(&mut self, failure_count: i32) {
        self.failure_count = failure_count;
    }

    /// C++ inline `MachineHandler::setChunkErred`.
    pub fn set_chunk_erred(&mut self, chunk_erred: bool) {
        self.chunk_erred = chunk_erred;
    }

    /// C++ declaration `MachineHandler::isTimedOut`.  The IMOD source declares
    /// this member but provides no definition, so no behavior is invented here.
    pub fn is_timed_out(&self, _index: usize, _timeout_millisec: i32) -> bool {
        panic!("MachineHandler::isTimedOut has no definition in IMOD source")
    }

    /// C++ inline `MachineHandler::incrementFailureCount`.
    pub fn increment_failure_count(&mut self) {
        self.failure_count += 1;
    }

    /// C++ declaration `MachineHandler::msgKillProcessTimeout`.  The IMOD
    /// source declares this member but provides no definition.
    pub fn msg_kill_process_timeout(&mut self) {
        panic!("MachineHandler::msgKillProcessTimeout has no definition in IMOD source")
    }

    /// C++ inline `MachineHandler::isJobValid`.
    pub fn is_job_valid(&self, index: usize) -> bool {
        self.process_handler_array[index].is_job_valid()
    }

    /// C++ `MachineHandler::getFullNumCpus`.
    pub fn get_full_num_cpus(&self) -> i32 {
        self.full_num_cpus
    }

    /// C++ `MachineHandler::getMultiProcInfoForCpu`.
    pub fn get_multi_proc_info_for_cpu(
        &self,
        cpu_index: usize,
        job_machine_list: &mut String,
    ) -> i32 {
        if cpu_index >= self.job_thread_limits.len() {
            return -1;
        }
        *job_machine_list = self.job_machine_lists[cpu_index].clone();
        self.job_thread_limits[cpu_index]
    }

    /// C++ declaration `MachineHandler::isKillNeeded`.  The IMOD source
    /// declares this member but provides no definition.
    pub fn is_kill_needed(&self) -> bool {
        panic!("MachineHandler::isKillNeeded has no definition in IMOD source")
    }

    /// C++ declaration `MachineHandler::isKillSignal`.  The IMOD source
    /// declares this member but provides no definition.
    pub fn is_kill_signal(&self) -> bool {
        panic!("MachineHandler::isKillSignal has no definition in IMOD source")
    }

    /// C++ `MachineHandler::resetKill`.
    pub fn reset_kill(&mut self) {
        unsafe {
            if !self.kill_warning
                && !self.ignore_kill
                && !(*self.processchunks).is_queue()
                && !self.pids_available
            {
                (*self.processchunks)
                    .write_out(&format!("No processes are running on {}\n", self.name));
            }
        }
        self.ignore_kill = true;
        self.kill_finished_signal_received = false;
        self.kill_started = false;
        self.pids_available = false;
        self.kill_counter = 0;
        self.pid_wait_counter = 0;
        self.kill_warning = false;
        for process_handler in &mut self.process_handler_array {
            process_handler.reset_kill();
        }
    }

    /// C++ `MachineHandler::startKill`.
    pub fn start_kill(&mut self, one_pid: &str) {
        self.ignore_kill = false;
        if self.dropped {
            self.ignore_kill = true;
            return;
        }
        self.killing_one = !one_pid.is_empty();
        self.one_pid_to_kill = one_pid.to_owned();
        unsafe {
            if !self.killing_one && (*self.processchunks).get_ans() == 'D' {
                if (*self.processchunks).get_drop_list().is_empty()
                    || !(*self.processchunks).get_drop_list().contains(&self.name)
                {
                    self.ignore_kill = true;
                } else {
                    self.dropped = true;
                }
            }
        }
        if self.ignore_kill {
            return;
        }
        self.ignore_kill = true;
        for process_handler in &mut self.process_handler_array {
            if (process_handler.is_job_valid() && !self.killing_one)
                || (self.killing_one && one_pid == process_handler.get_pid())
            {
                process_handler.start_kill(self.killing_one);
                self.ignore_kill = false;
            }
        }
        unsafe {
            if !self.ignore_kill
                && (*self.processchunks).is_queue()
                && (*self.processchunks).get_ans() == 'Q'
            {
                (*self.processchunks).write_out(&format!("Killing jobs on {}\n", self.name));
            }
        }
    }

    /// C++ `MachineHandler::killSignal`.
    pub fn kill_signal(&mut self) {
        if self.ignore_kill || self.kill_finished_signal_received {
            return;
        }
        unsafe {
            if !(*self.processchunks).is_queue() {
                let remote = self.remote_or_local_kill_type();
                if !self.kill_started {
                    if self.killing_one {
                        self.pids_available = true;
                    }
                    if !self.pids_available {
                        self.pids_available = true;
                        for process_handler in &mut self.process_handler_array {
                            if process_handler.is_job_valid() && process_handler.is_pid_empty() {
                                self.pids_available = false;
                            }
                        }
                    }
                    if self.pids_available || self.pid_wait_counter > 15 {
                        if (*self.processchunks).resources_available_for_kill() {
                            (*self.processchunks).write_out(&format!(
                                "Killing {} on {}\n",
                                if self.killing_one { "one job" } else { "jobs" },
                                self.name
                            ));
                            self.kill_started = true;
                            let mut pid_list = Vec::new();
                            let mut pid_found = false;
                            for process_handler in &mut self.process_handler_array {
                                if (process_handler.is_job_valid()
                                    && !self.killing_one
                                    && !process_handler.is_pid_empty())
                                    || (self.killing_one
                                        && process_handler.get_pid() == self.one_pid_to_kill)
                                {
                                    pid_found = true;
                                    process_handler.set_job_not_done();
                                    pid_list.push(process_handler.get_pid());
                                    process_handler.invalidate_job();
                                }
                            }
                            let mut command = "imodkillgroup".to_owned();
                            #[cfg(windows)]
                            {
                                command.push_str(".cmd");
                            }
                            let mut parameter_list = Vec::new();
                            if remote > 0 {
                                let mut parameter = format!("\"{}", command);
                                for pid in &pid_list {
                                    parameter.push(' ');
                                    parameter.push_str(pid);
                                }
                                parameter.push('"');
                                command = "ssh".to_owned();
                                parameter_list.push("-x".to_owned());
                                parameter_list
                                    .extend((*self.processchunks).get_ssh_opts().iter().cloned());
                                parameter_list.push(self.name.clone());
                                parameter_list.push("bash".to_owned());
                                parameter_list.push("--login".to_owned());
                                parameter_list.push("-c".to_owned());
                                parameter_list.push(parameter);
                            } else {
                                if remote < 0 {
                                    parameter_list.push("-t".to_owned());
                                }
                                parameter_list.extend(pid_list);
                            }
                            if pid_found {
                                (*self.processchunks).increment_kills();
                                match Command::new(&command)
                                    .args(&parameter_list)
                                    .stdout(Stdio::inherit())
                                    .stderr(Stdio::inherit())
                                    .spawn()
                                {
                                    Ok(process) => self.kill_process = Some(process),
                                    Err(_) => self.handle_error(ProcessError::FailedToStart),
                                }
                                thread::sleep(Duration::from_millis(
                                    (*self.processchunks).get_millisec_sleep() as u64,
                                ));
                            } else {
                                self.kill_warning = true;
                                (*self.processchunks).write_out(&format!(
                                    "Unable to kill any processes on {}\n",
                                    self.name
                                ));
                                self.kill_finished_signal_received = true;
                            }
                        }
                    } else {
                        self.pid_wait_counter += 1;
                    }
                } else {
                    self.kill_counter += 1;
                    if self.kill_counter > 15 && !self.kill_finished_signal_received {
                        if let Some(process) = &mut self.kill_process {
                            let _ = process.kill();
                        }
                        (*self.processchunks).decrement_kills();
                        self.kill_finished_signal_received = true;
                    }
                }
            } else {
                for process_handler in &mut self.process_handler_array {
                    if !self.killing_one || process_handler.get_pid() == self.one_pid_to_kill {
                        process_handler.kill_signal();
                    }
                }
            }
        }
    }

    /// C++ slot `MachineHandler::handleFinished`.
    pub fn handle_finished(&mut self, exit_code: i32, exit_status: ProcessExitStatus) {
        unsafe {
            if (*self.processchunks).is_verbose(&self.decorated_class_name, "handleFinished", 1) {
                (*self.processchunks).write_out(&format!(
                    "{}:handleFinished:exitCode:{},exitStatus:{}\n",
                    self.decorated_class_name,
                    exit_code,
                    match exit_status {
                        ProcessExitStatus::NormalExit => 0,
                        ProcessExitStatus::CrashExit => 1,
                    }
                ));
            }
            if !self.kill_finished_signal_received {
                (*self.processchunks).decrement_kills();
                self.kill_finished_signal_received = true;
            }
        }
    }

    /// C++ slot `MachineHandler::handleError`.
    pub fn handle_error(&mut self, error: ProcessError) {
        if error == ProcessError::FailedToStart || error == ProcessError::Crashed {
            unsafe {
                if !self.kill_finished_signal_received {
                    (*self.processchunks).decrement_kills();
                    self.kill_finished_signal_received = true;
                }
            }
        }
    }

    /// C++ `MachineHandler::isKillFinished`.
    pub fn is_kill_finished(&mut self) -> bool {
        if self.ignore_kill {
            return true;
        }
        unsafe {
            if !(*self.processchunks).is_queue() {
                let mut processes_finished = true;
                for process_handler in &mut self.process_handler_array {
                    if (!self.killing_one || process_handler.get_pid() == self.one_pid_to_kill)
                        && !process_handler.is_finished_signal_received()
                    {
                        processes_finished = false;
                    }
                }
                if processes_finished {
                    return true;
                }
                self.kill_finished_signal_received && (!self.killing_one || self.kill_counter > 15)
            } else {
                for process_handler in &mut self.process_handler_array {
                    if (!self.killing_one || process_handler.get_pid() == self.one_pid_to_kill)
                        && !process_handler.is_kill_finished()
                    {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// C++ `MachineHandler::remoteOrLocalKillType`.
    fn remote_or_local_kill_type(&self) -> i32 {
        let mut remote = unsafe {
            if self.name != (*self.processchunks).get_host_root()
                && self.name != "localhost"
                && !(*self.processchunks).is_queue()
            {
                1
            } else {
                0
            }
        };
        #[cfg(not(windows))]
        if remote == 0 {
            remote = -1;
        }
        remote
    }

    /// C++ `MachineHandler::killQProcesses`.
    pub fn kill_q_processes(&mut self) {
        for process_handler in &mut self.process_handler_array {
            process_handler.kill_q_processes();
            if let Some(process) = &mut self.kill_process {
                let _ = process.kill();
            }
        }
    }

    /// C++ `MachineHandler::setMultiProcJobLists`.
    pub fn set_multi_proc_job_lists(&mut self, machine_lists: &[String], thread_limits: &[i32]) {
        self.job_machine_lists = machine_lists.to_vec();
        self.job_thread_limits = thread_limits.to_vec();
        self.num_cpus = thread_limits.len() as i32;
    }

    /// C++ inline `MachineHandler::isDropped`.
    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    /// C++ inline `MachineHandler::isInternalDropped`.
    pub fn is_internal_dropped(&self) -> bool {
        self.internal_dropped
    }

    /// C++ inline `MachineHandler::setInternalDropped`.
    pub fn set_internal_dropped(&mut self) {
        self.internal_dropped = true;
    }
}
