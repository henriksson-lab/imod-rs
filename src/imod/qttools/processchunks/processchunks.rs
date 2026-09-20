//! Translation of `IMOD/qttools/processchunks/processchunks.{h,cpp}`.
//!
//! The C++ program owns the scheduling loop; `ComFileJobs`, `MachineHandler`,
//! and `ProcessHandler` remain separate source units just as they are in IMOD.
//!
//! The Qt event loop is represented directly: `startTimer`/`killTimer` become
//! the `timer_id`/`timer_interval` pair, `exec()` is the sleep-and-call loop in
//! `start_loop`, and the `QTimer::singleShot(0, ...)` in `startTimers` is the
//! immediate first call of `timer_event`.  `QProcess` signals are polled by
//! `ProcessHandler` at the same state-check boundaries the slots served.

use super::comfilejobs::ComFileJobs;
use super::machinehandler::MachineHandler;
use super::processhandler::ProcessHandler;
use super::{CHUNK_ASSIGNED, CHUNK_DONE, CHUNK_NOT_DONE, CHUNK_SYNC, CHUNK_TO_SKIP};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_milli_sleep, c_format, c_format_bytes, imod_backup_file, imod_usage_header,
    pid_to_stderr,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_integer, pip_get_non_option_arg, pip_get_string,
    pip_get_three_floats, pip_get_two_integers, pip_print_help, pip_read_or_parse_options,
};
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

/// C++ `static const int sleepMillisec = 1000;`
pub const SLEEP_MILLISEC: i32 = 1000;
/// C++ `static const int maxLocalByNum = 128;`
pub const MAX_LOCAL_BY_NUM: i32 = 128;
/// C++ `static const int runProcessTimeout = 30 * 2 * 1000;`
pub const RUN_PROCESS_TIMEOUT: i32 = 30 * 2 * 1000;
/// C++ `static const int checkFileReconnectReset = 10;`
pub const CHECK_FILE_RECONNECT_RESET: i32 = 10;
/// C++ `static const int otherProbeInterval = 60;`
pub const OTHER_PROBE_INTERVAL: i32 = 60;
/// C++ `static const char *commandName = "processchunks";`
const COMMAND_NAME: &[u8] = b"processchunks";

/// C++ `static const char *options[]` (autodoc2man fallbacks).
const OPTIONS: [&[u8]; 29] = [
    b":r:B:",
    b":s:B:",
    b":m:B:",
    b":G:B:",
    b":O:I:",
    b":M:I:",
    b":p:CH:",
    b":g:B:",
    b":n:I:",
    b":w:FN:",
    b":d:I:",
    b":e:I:",
    b":C:FT:",
    b":T:IP:",
    b":c:FN:",
    b":L:I:",
    b":q:I:",
    b":Q:CH:",
    b":I:CH:",
    b":D:CH:",
    b":W:I:",
    b":JC:I:",
    b":JG:I:",
    b":SQ:CH:",
    b":SN:I:",
    b":P:B:",
    b":v:B:",
    b":V:CH:",
    b":help:B:",
];
/// C++ `static const char *queueNameDefault = "queue";`
const QUEUE_NAME_DEFAULT: &str = "queue";

/// The C++ `Processchunks` application.  Header and implementation are merged
/// by the Rust source-organization rule.
pub struct Processchunks {
    pub size_job_array: i32,
    pub machine_list_size: i32,
    pub num_machines_dropped: i32,
    pub job_limit_per_cycle: i32,
    pub com_file_jobs: Option<ComFileJobs>,
    /// C++ `mMachineList`; `Vec` supplies the ownership of the former array.
    pub machine_list: Vec<MachineHandler>,
    //parameters
    pub retain: i32,
    pub just_go: i32,
    pub nice: i32,
    pub millisec_sleep: i32,
    pub drop_crit: i32,
    pub queue: i32,
    pub single_file: i32,
    pub max_chunk_err: i32,
    pub verbose: i32,
    pub gpu_mode: i32,
    pub num_threads: i32,
    pub multiple_files: i32,
    pub entered_max_chunk_err: i32,
    pub num_multi_proc_jobs: i32,
    pub max_on_secondary_queue: i32,
    pub skip_probe: bool,
    pub queue_name: String,
    /// C++ `char *mRootName`, NULL until `loadParams`.
    pub root_name: Option<String>,
    pub init_queue: Option<String>,
    pub deinit_queue: Option<String>,
    pub secondary_queue: Option<String>,
    pub wait_for_queue_init: i32,
    pub multi_max_queue_jobs: i32,
    pub cores_per_cluster_job: i32,
    pub gpus_per_cluster_job: i32,
    /// C++ `QFile *mCheckFile`: its name, and the open handle while it is open.
    pub check_file: Option<String>,
    pub check_file_handle: Option<io::BufReader<fs::File>>,
    pub cpu_list: String,
    pub verbose_class: String,
    pub remote_dir: Option<String>,
    pub verbose_function_list: Vec<String>,
    pub gpu_pool_list: Vec<String>,
    pub gpu_only_machines: Vec<String>,
    pub multi_proc_gpu_pool: String,
    //setup
    pub copy_log_index: i32,
    pub num_cpus: i32,
    pub host_root: String,
    pub queue_command: String,
    pub decorated_class_name: String,
    pub escaped_remote_dir_path: String,
    pub com_extension: String,
    pub host_root2: String,
    pub full_host_name: String,
    pub ssh_opts: Vec<String>,
    pub queue_param_list: Vec<String>,
    pub current_dir: PathBuf,
    //loop
    pub num_done: i32,
    pub last_num_done: i32,
    pub hold_crit: i32,
    pub timer_id: i32,
    pub first_undone_index: i32,
    pub next_sync_index: i32,
    pub syncing: i32,
    pub check_file_reconnect: i32,
    pub slowest_time: i32,
    pub slow_time_count: i32,
    pub num_skipped: i32,
    pub pausing: bool,
    pub any_done: bool,
    pub hold_for_multi_proc_drop: bool,
    pub ignore_pausing_errors: bool,
    pub last_other_probe_time: SystemTime,
    pub slow_overall_crit: f32,
    pub slow_machine_crit: f32,
    pub slow_sync_factor: f32,
    pub slow_log_timeout: i32,
    pub slow_sync_log_timeout: i32,
    pub ans: char,
    pub save_check_file_lines: Vec<String>,
    //killing processes
    pub kill: bool,
    pub kill_counter: i32,
    pub num_kills: i32,
    pub max_kills: i32,
    pub drop_list: Vec<String>,
    //running processes
    /// C++ `QProcess *mVmstopy`, reused for `vmstopy` and the periodic probes.
    pub vmstopy: Option<Child>,
    pub ls_param_list: Vec<String>,
    /// The interval of the running Qt timer, the argument of `startTimer`.
    pub timer_interval: i32,
    /// `QCoreApplication` state: whether `exec()` is running, whether
    /// `exit()` has been called inside it, and the code it was given.
    pub exec_running: bool,
    pub quit_now: bool,
    pub return_code: i32,
}

/// C++ `processchunksUsageHeader`.
pub fn processchunks_usage_header(pname: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(pname)));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "\nUsage: %s [Options] machine_list root_name\nWill process multiple command \
         files on multiple processors or machines\nmachine_list is a list of \
         available machines, separated by commas.\nList machine names multiple \
         times or followed by :n to use multiple CPUs on a machine.\nRoot_name is \
         the base name of the command files, omitting -nnn.com\n\n",
        &[CArg::Bytes(pname)],
    ));
    // `PipPrintHelp` writes through Rust's stdout; hand the C stream over first.
    let _ = ImodFile::Stdout.flush();
}

impl Default for Processchunks {
    fn default() -> Self {
        Self::new()
    }
}

impl Processchunks {
    /// C++ `Processchunks::Processchunks`.
    pub fn new() -> Self {
        Self {
            size_job_array: 0,
            machine_list_size: 0,
            num_machines_dropped: 0,
            job_limit_per_cycle: 20,
            com_file_jobs: None,
            machine_list: vec![],
            retain: 0,
            just_go: 0,
            nice: 18,
            millisec_sleep: 50,
            drop_crit: -1,
            queue: 0,
            single_file: 0,
            max_chunk_err: 5,
            verbose: 0,
            gpu_mode: 0,
            num_threads: 1,
            multiple_files: 0,
            entered_max_chunk_err: 0,
            num_multi_proc_jobs: 0,
            max_on_secondary_queue: 0,
            skip_probe: false,
            queue_name: QUEUE_NAME_DEFAULT.to_owned(),
            root_name: None,
            init_queue: None,
            deinit_queue: None,
            secondary_queue: None,
            wait_for_queue_init: 30000,
            multi_max_queue_jobs: 0,
            cores_per_cluster_job: 0,
            gpus_per_cluster_job: 0,
            check_file: None,
            check_file_handle: None,
            cpu_list: String::new(),
            verbose_class: String::new(),
            remote_dir: None,
            verbose_function_list: vec![],
            gpu_pool_list: vec![],
            gpu_only_machines: vec![],
            multi_proc_gpu_pool: String::new(),
            copy_log_index: -1,
            num_cpus: 0,
            host_root: String::new(),
            queue_command: String::new(),
            // `typeid(*this).name()` under the Itanium C++ ABI, which is what
            // the reference build prints in its verbose diagnostics.
            decorated_class_name: "13Processchunks".to_owned(),
            escaped_remote_dir_path: String::new(),
            com_extension: ".com".to_owned(),
            host_root2: String::new(),
            full_host_name: String::new(),
            ssh_opts: vec![
                "-o PreferredAuthentications=publickey".to_owned(),
                "-o StrictHostKeyChecking=no".to_owned(),
            ],
            queue_param_list: vec![],
            current_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            num_done: 0,
            last_num_done: 0,
            hold_crit: 0,
            timer_id: 0,
            first_undone_index: 0,
            next_sync_index: 0,
            syncing: 0,
            check_file_reconnect: CHECK_FILE_RECONNECT_RESET,
            slowest_time: -1,
            slow_time_count: 0,
            num_skipped: 0,
            pausing: false,
            any_done: false,
            hold_for_multi_proc_drop: false,
            ignore_pausing_errors: false,
            last_other_probe_time: SystemTime::now(),
            slow_machine_crit: 4., // This is a true criterion
            slow_overall_crit: 3., // This is a factor here, it will be multiplied by machine crit
            slow_sync_factor: 0.,
            slow_log_timeout: 300,
            slow_sync_log_timeout: 0,
            ans: ' ',
            save_check_file_lines: vec![],
            kill: false,
            kill_counter: 0,
            num_kills: 0,
            max_kills: 0,
            drop_list: vec![],
            vmstopy: None,
            ls_param_list: vec![],
            timer_interval: 0,
            exec_running: false,
            quit_now: false,
            return_code: 0,
        }
    }

    /// `QCoreApplication::exit(int)`: the static member that an unqualified
    /// `exit(...)` inside this class resolves to.  It tells a running event
    /// loop to quit with `returnCode` and returns; before `exec()` it does
    /// nothing.  The C library's `exit` is only reached from `main`.
    pub fn exit(&mut self, return_code: i32) {
        if self.exec_running {
            self.quit_now = true;
            self.return_code = return_code;
        }
    }

    /// C++ `Processchunks::printOsInformation`.
    pub fn print_os_information(&self) {
        let _ = ImodFile::Stdout.write_all(
            b"\nIMPORTANT:  Ctrl-C does not work with this version of processchunks.  Use ",
        );
        #[cfg(not(windows))]
        let _ = ImodFile::Stdout.write_all(b"<Esc> <Enter> or ");
        let _ =
            ImodFile::Stdout.write_all(b"the -c option (-c defaults to processchunks.input).\n\n");
    }

    /// C++ `Processchunks::loadParams`.
    pub fn load_params(&mut self, argv: &[String]) {
        let mut num_opt_args = 0;
        let mut num_non_opt_args = 0;
        let mut if_help = 0;
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &OPTIONS,
            OPTIONS.len() as i32,
            COMMAND_NAME,
            2,
            2,
            0,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(processchunks_usage_header),
        );
        if pip_get_boolean(b"help", &mut if_help) == 0 {
            processchunks_usage_header(COMMAND_NAME);
            pip_print_help(COMMAND_NAME, 0, 1, 1);
            self.exit(0);
        }
        pip_get_boolean(b"r", &mut self.retain);
        pip_get_boolean(b"G", &mut self.gpu_mode);
        pip_get_boolean(b"s", &mut self.single_file);
        pip_get_boolean(b"m", &mut self.multiple_files);
        if self.single_file != 0 && self.multiple_files != 0 {
            exit_error(b"You cannot enter both -s and -m");
        }
        if pip_get_integer(b"M", &mut self.num_multi_proc_jobs) == 0 && self.num_multi_proc_jobs < 2
        {
            exit_error(b"Number of multiprocessor jobs must be at least 2");
        }
        if self.num_multi_proc_jobs != 0 && self.gpu_mode != 0 {
            exit_error(b"You cannot enter both -G and -M");
        }
        if self.num_multi_proc_jobs != 0 && self.single_file != 0 {
            exit_error(b"You cannot enter both -s and -M");
        }
        if self.gpu_mode != 0
            || self.single_file != 0
            || self.multiple_files != 0
            || self.num_multi_proc_jobs != 0
        {
            self.num_threads = 0;
        }
        let mut gpu_pool = Vec::new();
        // Make a string list with the full inital GPU pool
        if pip_get_string(b"p", &mut gpu_pool) == 0 {
            if self.num_multi_proc_jobs == 0 {
                exit_error(b"You can enter -p only when doing multiprocessor jobs");
            }
            self.gpu_pool_list = String::from_utf8_lossy(&gpu_pool)
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect();
        }
        if self.multiple_files != 0 {
            self.slow_machine_crit = 0.;
            self.max_chunk_err = 2;
            self.multiple_files = num_non_opt_args - 1;
        } else if num_non_opt_args > 2 {
            exit_error(
                b"More than two non-option arguments were entered; use the -m option to \
                  enter multiple command files",
            );
        }
        pip_get_integer(b"O", &mut self.num_threads);
        pip_get_boolean(b"g", &mut self.just_go);
        pip_get_integer(b"n", &mut self.nice);
        pip_get_integer(b"L", &mut self.job_limit_per_cycle);
        //PipGetInteger("m", &mMillisecSleep);
        let mut remote_dir = Vec::new();
        if pip_get_string(b"w", &mut remote_dir) == 0 {
            self.remote_dir = Some(String::from_utf8_lossy(&remote_dir).into_owned());
        }
        pip_get_integer(b"d", &mut self.drop_crit);
        self.entered_max_chunk_err = 1 - pip_get_integer(b"e", &mut self.max_chunk_err);
        pip_get_three_floats(
            b"C",
            &mut self.slow_machine_crit,
            &mut self.slow_overall_crit,
            &mut self.slow_sync_factor,
        );
        self.slow_overall_crit *= self.slow_machine_crit;
        pip_get_two_integers(
            b"T",
            &mut self.slow_log_timeout,
            &mut self.slow_sync_log_timeout,
        );
        let mut check_file = Vec::new();
        if pip_get_string(b"c", &mut check_file) == 0 {
            self.check_file = Some(String::from_utf8_lossy(&check_file).into_owned());
        } else {
            self.check_file = Some("processchunks.input".to_owned());
        }
        pip_get_boolean(b"v", &mut self.verbose);
        if self.verbose != 0 {
            let mut verbose_class_functions = Vec::new();
            //Set verbose instructions.
            if pip_get_string(b"V", &mut verbose_class_functions) == 0 {
                let param = String::from_utf8_lossy(&verbose_class_functions).into_owned();
                let mut param_list: Vec<String> = param
                    .trim()
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned)
                    .collect();
                if !param_list.is_empty() {
                    //Set verbosity level.
                    if let Ok(temp) = param_list[param_list.len() - 1].parse::<i32>() {
                        self.verbose = temp;
                        param_list.pop();
                    }
                    let mut help = false;
                    if !param_list.is_empty() {
                        //Set the verbose class.
                        self.verbose_class = param_list.remove(0);
                        if self.verbose_class == "?" {
                            help = true;
                            //If the param is a question mark, print messages from Processchunks::isVerbose.
                            self.verbose_class = "processchunks".to_owned();
                            self.verbose_function_list.push("isverbose".to_owned());
                        }
                    }
                    if !help && !param_list.is_empty() {
                        //Set the verbose function list.
                        self.verbose_function_list = param_list;
                    }
                }
            }
        }
        if pip_get_integer(b"q", &mut self.queue) == 0 {
            self.skip_probe = true;
            self.just_go = 1;
            if !self.gpu_pool_list.is_empty() {
                exit_error(b"You cannot enter a GPU pool list with a queue command");
            }
            if self.num_multi_proc_jobs != 0 {
                self.multi_max_queue_jobs = self.queue;
                self.queue = self.num_multi_proc_jobs;
            }
        }
        if pip_get_integer(b"JC", &mut self.cores_per_cluster_job) == 0 {
            if self.num_multi_proc_jobs == 0 || self.queue == 0 {
                exit_error(
                    b"You can enter cores per node per only when doing multiprocessor jobs on\
                      a cluster queue",
                );
            }
            if self.cores_per_cluster_job <= 0 {
                exit_error(b"Cores per node must be positive");
            }
            pip_get_integer(b"JG", &mut self.gpus_per_cluster_job);
            if self.gpus_per_cluster_job < 0 {
                exit_error(b"GPUs per node must be positive");
            }
        }
        let mut secondary_queue = Vec::new();
        if pip_get_string(b"SQ", &mut secondary_queue) == 0 {
            self.secondary_queue = Some(String::from_utf8_lossy(&secondary_queue).into_owned());
            if self.num_multi_proc_jobs == 0 || self.queue == 0 {
                exit_error(
                    b"You can enter a secondary queue only when doing multiprocessor jobs on\
                      a cluster queue",
                );
            }
            if pip_get_integer(b"SN", &mut self.max_on_secondary_queue) != 0 {
                exit_error(
                    b"You must enter the maximum number of jobs for a secondary queue if \
                      such a queue is entered",
                );
            }
            if self.max_on_secondary_queue <= 0 {
                exit_error(b"The maximum number of jobs for a secondary queue must be positive");
            }
        }
        if self.drop_crit < 1 {
            self.drop_crit = if self.queue != 0 { 10 } else { 5 };
        }
        let mut string = Vec::new();
        if pip_get_string(b"Q", &mut string) == 0 {
            self.queue_name = String::from_utf8_lossy(&string).into_owned();
        }
        let mut string = Vec::new();
        if pip_get_string(b"I", &mut string) == 0 {
            self.init_queue = Some(String::from_utf8_lossy(&string).into_owned());
        }
        let mut string = Vec::new();
        if pip_get_string(b"D", &mut string) == 0 {
            self.deinit_queue = Some(String::from_utf8_lossy(&string).into_owned());
        }
        let mut wait_sec = self.wait_for_queue_init / 1000;
        if pip_get_integer(b"W", &mut wait_sec) != 0 {
            if let Ok(wait_str) = std::env::var("IMOD_QUEUE_INIT_WAIT") {
                // `atoi`: leading whitespace, optional sign, digits, else 0.
                let trimmed = wait_str.trim_start();
                let end = trimmed
                    .char_indices()
                    .take_while(|(i, c)| {
                        c.is_ascii_digit() || (*i == 0 && (*c == '-' || *c == '+'))
                    })
                    .map(|(i, c)| i + c.len_utf8())
                    .last()
                    .unwrap_or(0);
                wait_sec = trimmed[..end].parse::<i32>().unwrap_or(0);
            }
        }
        if wait_sec < -1 || wait_sec == 0 || wait_sec > 2000000 {
            exit_error(
                b"Wait time for queue initialization command must be -1 or a positive \
                  value up to 2000000",
            );
        }
        if wait_sec < 0 {
            self.wait_for_queue_init = -1;
        } else {
            self.wait_for_queue_init = 1000 * wait_sec;
        }
        let mut return_pid = 0;
        pip_get_boolean(b"P", &mut return_pid);
        if return_pid != 0 {
            self.skip_probe = true;
            pid_to_stderr();
        }
        let mut cpu_list = Vec::new();
        if pip_get_non_option_arg(0, &mut cpu_list) == 0 {
            self.cpu_list = String::from_utf8_lossy(&cpu_list).into_owned();
        }
        let mut root_name = Vec::new();
        if pip_get_non_option_arg(1, &mut root_name) == 0 {
            self.root_name = Some(String::from_utf8_lossy(&root_name).into_owned());
        }
        //Error check
        if self.retain != 0 && self.single_file != 0 {
            exit_error(b"You cannot use the retain option with a single command file");
        }
    }

    /// The `mRootName` the C++ dereferences freely after `loadParams`.
    fn root_name(&self) -> &str {
        self.root_name.as_deref().unwrap_or("")
    }

    /// C++ `Processchunks::setup`: mSshOpts, mCpuArray, mProcessArray,
    /// mHostRoot, mRemoteDir and multi-proc jobs.  Probe machines.
    pub fn setup(&mut self) -> bool {
        let mut num_cpus_list: Vec<i32> = Vec::new();
        let mut gpu_list: Vec<i32> = Vec::new();

        self.setup_ssh_opts();
        //Get current directory if the -w option was not used
        if self.remote_dir.is_none() {
            self.remote_dir = Some(
                fs::canonicalize(&self.current_dir)
                    .unwrap_or_else(|_| self.current_dir.clone())
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        let mut machine_name_list: Vec<String> = Vec::new();
        self.init_machine_list(&mut machine_name_list, &mut num_cpus_list, &mut gpu_list);

        // If there is only one machine, there is no point failing chunks 5 times
        if machine_name_list.len() == 1 && self.entered_max_chunk_err == 0 {
            self.max_chunk_err = if self.queue != 0 { 10 } else { 2 };
        }
        self.setup_host_root();
        self.setup_com_file_jobs();
        let retval = self.probe_machines(&mut machine_name_list);
        self.setup_machine_list(&mut machine_name_list, &num_cpus_list, &gpu_list);
        machine_name_list.clear();

        // Modify things for multi-processor jobs
        if self.num_multi_proc_jobs != 0 && self.queue == 0 {
            // Make list of machines in GPU pool that are not in CPU list
            for gpu in 0..self.gpu_pool_list.len() {
                let mut found = false;
                for mach in 0..self.machine_list_size as usize {
                    if self.gpu_pool_list[gpu]
                        .to_lowercase()
                        .find(&self.machine_list[mach].get_name().to_lowercase())
                        == Some(0)
                    {
                        found = true;
                    }
                }
                if !found {
                    let gpu_split: Vec<&str> = self.gpu_pool_list[gpu].split(':').collect();
                    self.gpu_only_machines.push(gpu_split[0].to_owned());
                }
            }

            if self.divide_machines_for_jobs() != 0 {
                exit_error(
                    b"There is only one processor available; cannot run multiprocessor jobs",
                );
            }
            if self.skip_probe || self.just_go != 0 {
                // Better probe the other machines now
                loop {
                    self.probe_other_multi_proc_machines(true);
                    if !self.hold_for_multi_proc_drop {
                        break;
                    }
                    self.hold_for_multi_proc_drop = false;
                    let err = self.divide_machines_for_jobs();
                    if err == 1 {
                        exit_error(
                            b"That leaves only one processor; cannot run multiprocessor jobs",
                        );
                    }
                    if err == 2 {
                        exit_error(b"That leaves no GPUs; cannot run multiprocessor jobs");
                    }
                }
            }
        }
        retval
    }

    /// C++ `Processchunks::startLoop`: Setup mFlags.  Find first not-done log
    /// file.  Delete log files and miscellaneous files.  Listen for ctrl-C.
    /// Run event loop.
    pub fn start_loop(&mut self) -> i32 {
        //Prescan logs for done ones to find first undone one, or back up unfinished
        self.num_done = 0;
        self.num_skipped = 0;
        self.first_undone_index = -1;
        let this = self as *mut Self;
        let mut process = ProcessHandler::new();
        process.setup(unsafe { &mut *this }, -1);
        for i in 0..self.size_job_array {
            process.set_job(i);
            if process.log_file_exists(false) {
                if process.is_chunk_done() {
                    //If it was done and we are resuming, set flag it is done, count
                    if self.retain != 0 {
                        process.set_flag(CHUNK_DONE);
                        self.num_done += 1;
                    }
                } else if self.retain == 0 {
                    //If it was not done and we are restarting, back up the old log
                    process.backup_log();
                }
            }
            //If resuming and this is the first undone one, keep track of that
            if self.retain != 0 && self.first_undone_index == -1 && process.get_flag() != CHUNK_DONE
            {
                self.first_undone_index = i;
            }
            //OLD:remove logs if not restarting
            //remove logs if not resuming
            if self.retain == 0 {
                process.remove_files();
            }
            process.invalidate_job();
        }
        if self.first_undone_index == -1 {
            self.first_undone_index = 0;
        }

        if self.single_file == 0 || self.skip_probe {
            self.write_out(&format!(
                "{} OF {} DONE SO FAR \n",
                self.num_done, self.size_job_array
            ));
        }
        //Initialize variables needed by the timer event function
        self.last_num_done = 0;
        self.pausing = false;
        self.syncing = 0;
        self.any_done = false;
        self.next_sync_index = self.size_job_array + 2 - 1;

        // Make sure at least 2 failures are required to do a hold
        // (Wrong comment: Change from script: base this on number of CPU's not # of machines)
        self.hold_crit = 2.max((self.machine_list_size + 1) / 2);
        //Error messages from inside the event loop must using QApplication functionality
        pip_done();
        self.exec_running = true;
        self.start_timers();

        // 6/11/3: Tried SetConsoleCtrlHandler in Windows for catching Ctrl-C.  It worked
        // fine in a DOS window but not in a mintty, so we are stuck with the current situation
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_IGN);
            #[cfg(not(windows))]
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
        }
        // `exec()`: the timer fires at its interval until `exit()` has been
        // called from inside an event; the single-shot call made by
        // `startTimers` is the first event.
        loop {
            if self.quit_now {
                return self.return_code;
            }
            b3d_milli_sleep(self.timer_interval);
            self.timer_event();
        }
    }

    /// C++ `Processchunks::startTimers`.
    pub fn start_timers(&mut self) {
        //Make sure there isn't already a timer going
        if self.timer_id != 0 {
            self.timer_id = 0;
        }
        //The timer event function should be called immediately and then put on a timer
        if self.queue != 0 {
            //Must look at files instead of stdout/err.  Prevent program from being a hog.
            self.timer_interval = (2 + self.num_cpus / 100) * 1000;
        } else {
            self.timer_interval = SLEEP_MILLISEC;
        }
        self.timer_id = 1;
        self.timer_event();
    }

    /// C++ `Processchunks::timerEvent`.
    pub fn timer_event(&mut self) {
        if self.kill {
            self.kill_signal();
            return;
        }
        //Handle the regular timer.
        if self.escape_entered() != 0 {
            self.handle_interrupt();
            return;
        }
        if self.num_done >= self.size_job_array {
            self.cleanup_and_exit(0);
            return;
        }
        if self.read_check_file() {
            return;
        }
        //Count failures and assignments
        let mut assign_tot = 0;
        let mut fail_tot = 0;
        let mut min_fail = self.drop_crit;
        let mut fail_count;
        let mut chunk_err_tot = 0;
        let mut num_cpus;
        let mut no_chunks = false;
        let mut need_kill = false;
        let mut num_added = 0;
        let this = self as *mut Self;
        for i in 0..self.machine_list_size as usize {
            // Change from script: neither chunkErrTot nor failTot is incremented for each cpu,
            // only per machine
            fail_count = self.machine_list[i].get_failure_count();
            let mut jobs_running = 0;
            if fail_count != 0 {
                fail_tot += 1;
            }
            if fail_count < min_fail {
                min_fail = fail_count;
            }
            if self.machine_list[i].is_chunk_erred() {
                chunk_err_tot += 1;
            }
            num_cpus = self.machine_list[i].get_num_cpus();
            for cpu_index in 0..num_cpus.max(0) as usize {
                if self.machine_list[i].is_job_valid(cpu_index) {
                    assign_tot += 1;
                    jobs_running += 1;
                }
            }

            // If there are no more jobs and the machine had not been dropped yet, now
            // drop it
            if jobs_running == 0
                && self.any_done
                && fail_count >= self.drop_crit
                && !self.machine_list[i].is_internal_dropped()
            {
                let name = self.machine_list[i].get_name().to_owned();
                self.write_out(&format!("Dropping {name}\n"));
                self.machine_list[i].set_internal_dropped();
            }
        }

        // If pausing on a queue and there was nothing running yet, it simply needs to test here
        // that there is nothing and exit
        // But it has to exit with 2 and give the same message for Etomo to enable Resume
        if self.queue != 0 && assign_tot == 0 && self.pausing {
            self.write_out(
                "All previously running chunks are done - exiting as requested\n\
                 Rerun with -r to resume and retain existing results\n",
            );
            self.cleanup_and_exit(2);
            return;
        }

        if self.exit_if_dropped(min_fail, fail_tot, assign_tot) {
            return;
        }

        self.probe_other_multi_proc_machines(false);

        //Loop on machines and CPUs, if they have an assignment check if it is done
        let mut i: i32 = -1;
        let mut loop_done = false;
        while {
            i += 1;
            i < self.machine_list_size && !loop_done
        } {
            let machine: *mut MachineHandler = &mut self.machine_list[i as usize];
            num_cpus = unsafe { (*machine).get_num_cpus() };
            for cpu_index in 0..num_cpus.max(0) as usize {
                let process: *mut ProcessHandler =
                    unsafe { (*machine).get_process_handler(cpu_index) };
                let mut job_index = -1;
                let mut dropout = false;
                if unsafe { (*process).is_job_valid() } {
                    job_index = unsafe { (*process).get_assigned_job_index() };
                    let mut drop_mess = String::new();
                    let mut error_mess = String::new();
                    if unsafe { (*process).is_com_process_done() } {
                        //Handle the comscript ran and finished
                        //OLD:If the log is present and the .csh is gone, it has exited
                        //If the log is present and the process's finished signal has been caught
                        if unsafe { (*process).is_chunk_done() } {
                            //If mSingleFile is true, set loopDone to end outer loop, and break
                            //out of inner loop.
                            loop_done = unsafe {
                                (*this).handle_chunk_done(&mut *machine, &mut *process, job_index)
                            };
                            if loop_done {
                                break;
                            }
                        } else {
                            if self.queue == 0 && unsafe { (*process).is_pausing() } {
                                // DNM: changed from return to continue.  When 30 jobs crash right away,
                                // it can take a long time to get through them if you have to get through
                                // successive pauses on each one
                                continue;
                            }
                            //otherwise set flag to redo it
                            dropout = true;
                            if !unsafe { (*process).is_log_file_empty() } {
                                if !unsafe {
                                    (*this).handle_log_file_error(
                                        &mut error_mess,
                                        &mut *machine,
                                        &mut *process,
                                    )
                                } {
                                    return;
                                }
                            } else if self.queue == 0 {
                                //OLD: If log is zero length, check for something in .pid
                                //If the com script issues a PID to standard error and nothing
                                //to standard out, it can't run the first real command in the
                                //file.
                                if unsafe { (*process).is_pid_in_stderr() }
                                    && !unsafe {
                                        (*this).handle_error(
                                            None,
                                            &mut *machine,
                                            &mut *process,
                                            false,
                                        )
                                    }
                                {
                                    return;
                                }
                            }
                        }
                    } else {
                        unsafe {
                            (*this).handle_com_process_not_done(
                                &mut dropout,
                                &mut drop_mess,
                                &mut *machine,
                                &mut *process,
                                &mut need_kill,
                            )
                        };
                    }

                    //if failed, remove the assignment, mark chunk as to be done,
                    //skip this machine on this round
                    if dropout {
                        unsafe {
                            (*this).handle_drop_out(
                                &mut no_chunks,
                                &mut drop_mess,
                                &mut *machine,
                                &mut *process,
                                &mut error_mess,
                                need_kill,
                            )
                        };
                    }

                    // For a kill, set flag and slow down timer as in killProcesses, but first let
                    // a chunk error kill and exit occur
                    if need_kill {
                        if !unsafe {
                            (*this).handle_error(None, &mut *machine, &mut *process, true)
                        } {
                            return;
                        }
                        if self.timer_id != 0 {
                            self.timer_id = 0;
                        }
                        self.kill = true;
                        self.ans = 'P';
                        let pid = unsafe { (*process).get_pid() };
                        unsafe { (*machine).start_kill(&pid) };
                        self.kill_signal();
                        self.timer_interval = 1000;
                        self.timer_id = 1;
                        return;
                    }

                    //OLD:Clean up .ssh and .pid if no longer assigned
                    //For queue only:  clean up .job and .qid if no longer assigned
                    if !unsafe { (*process).is_job_valid() } && self.queue != 0 {
                        unsafe { (*process).remove_process_files() };
                    }
                }

                //Drop a machine if it has failed more than given number of times
                //Institute hold on any failed machine if no chunks are done and
                //machine failure count is above criterion
                fail_count = unsafe { (*machine).get_failure_count() };
                if fail_count >= self.drop_crit
                    || self.pausing
                    || (fail_count != 0 && !self.any_done && fail_tot >= self.hold_crit)
                    || self.hold_for_multi_proc_drop
                {
                    dropout = true;
                }
                /*If the current machine is unassigned and has not been dropped, find
                next com to do and run it.  Move current log out of way so non-existence
                of log can be sign of nothing having started.  Skip if no chunks are
                available*/
                if !unsafe { (*machine).is_dropped() }
                    && !unsafe { (*process).is_job_valid() }
                    && !dropout
                    && !no_chunks
                    && self.syncing != 2
                {
                    job_index = self.first_undone_index;
                    let mut found_chunks = false;
                    let mut undone_index = -1;
                    while job_index < self.size_job_array && !unsafe { (*process).is_job_valid() } {
                        let mut run_flag = 0;
                        let mut chunk_ok = false;
                        if !unsafe {
                            (*this).check_chunk(
                                &mut run_flag,
                                &mut no_chunks,
                                &mut undone_index,
                                &mut found_chunks,
                                &mut chunk_ok,
                                &mut *machine,
                                job_index,
                                chunk_err_tot,
                            )
                        } {
                            break;
                        }
                        if (run_flag == CHUNK_SYNC || run_flag == CHUNK_NOT_DONE) && chunk_ok {
                            let error = unsafe {
                                (*this).run_process(
                                    &mut *machine,
                                    &mut *process,
                                    job_index,
                                    cpu_index as i32,
                                )
                            };
                            if error == 0 {
                                num_added += 1;
                            }
                            if error > 1 {
                                return;
                            }
                        }
                        job_index += 1;
                    }
                    //If no chunks were found in that loop set the nochunks flag
                    if !found_chunks || num_added >= self.job_limit_per_cycle {
                        no_chunks = true;
                    }
                    if undone_index > self.first_undone_index {
                        self.first_undone_index = undone_index;
                    }
                }
            }
        }
        if self.num_done > self.last_num_done {
            self.write_out(&format!(
                "{} OF {} DONE SO FAR\n",
                self.num_done, self.size_job_array
            ));
        }
        self.last_num_done = self.num_done;
        //Old:  if we have finished up to the sync file, then allow the loop to run it
        if self.num_done - 1 >= self.next_sync_index - 1 {
            let end_com_name = format!("{}-finish.com", self.root_name());
            if self
                .get_com_file_jobs()
                .get_com_file_name(self.next_sync_index as usize)
                == end_com_name
            {
                self.write_out(&format!(
                    "ALL DONE - going to run {end_com_name} to reassemble\n"
                ));
            }
            //Set syncing flag to 1 to get it started
            self.syncing = 1;
            self.first_undone_index = self.next_sync_index;
            self.next_sync_index = self.size_job_array + 2 - 1;
            no_chunks = false;
            let _ = no_chunks;
        }

        // Finish up now if all chunks are done
        if (self.single_file != 0 && self.num_done > 0)
            || self.num_done + self.num_skipped >= self.size_job_array
        {
            self.cleanup_and_exit(0);
        }
    }

    /// C++ `Processchunks::cleanupAndExit`.  Its final `exit(exitCode)` is
    /// `QCoreApplication::exit`, so it returns to its caller and the process
    /// ends when the event loop hands the code back to `main`.
    pub fn cleanup_and_exit(&mut self, exit_code: i32) {
        if self.timer_id != 0 {
            self.timer_id = 0;
        }
        if exit_code == 0 {
            //Etomo is looking for "to reassemble"
            let end_com_name = format!("{}-finish.com", self.root_name());
            if !self.current_dir.join(&end_com_name).exists() {
                self.write_out("ALL DONE - nothing to reassemble\n");
            }
            //Etomo is looking for this line too
            self.write_out("Finished reassembling\n");
        }
        if let Some(check_file) = self.check_file.clone() {
            self.check_file_handle = None;
            if exit_code == 0 && self.current_dir.join(&check_file).exists() {
                let _ = fs::remove_file(self.current_dir.join(&check_file));
            }
        }
        if exit_code != 0 {
            for i in 0..self.machine_list_size as usize {
                self.machine_list[i].kill_q_processes();
            }
        }
        let deinit_queue = self.deinit_queue.clone();
        let i = self.run_generic_queue_command(deinit_queue.as_deref(), 30000);
        if i != 0 {
            self.write_out(&format!(
                "WARNING: Command to de-initialize queue returned with status {i}\n"
            ));
        }
        self.write_out(&format!("exitCode:{exit_code}\n"));
        self.exit(exit_code);
    }

    /// C++ `Processchunks::escapeEntered`.
    pub fn escape_entered(&mut self) -> i32 {
        #[cfg(not(windows))]
        {
            use std::cell::Cell;
            thread_local! {
                static NUM_CHAR: Cell<i32> = const { Cell::new(0) };
                static GOT_ESC: Cell<i32> = const { Cell::new(0) };
            }
            unsafe {
                let mut readfds: libc::fd_set = std::mem::zeroed();
                let mut writefds: libc::fd_set = std::mem::zeroed();
                let mut exceptfds: libc::fd_set = std::mem::zeroed();
                let mut timeout = libc::timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                };
                let mut charin: u8 = 0;
                libc::FD_ZERO(&mut readfds);
                libc::FD_ZERO(&mut writefds);
                libc::FD_ZERO(&mut exceptfds);
                let stdin_fd = libc::STDIN_FILENO;
                libc::FD_SET(stdin_fd, &mut readfds);
                while libc::select(1, &mut readfds, &mut writefds, &mut exceptfds, &mut timeout) > 0
                {
                    if libc::read(stdin_fd, (&mut charin as *mut u8).cast(), 1) == 0 {
                        return 0;
                    }
                    if charin == b'\n' {
                        if NUM_CHAR.get() == 1 && GOT_ESC.get() == 1 {
                            NUM_CHAR.set(0);
                            GOT_ESC.set(0);
                            return 1;
                        }
                        NUM_CHAR.set(0);
                        GOT_ESC.set(0);
                    } else {
                        NUM_CHAR.set(NUM_CHAR.get() + 1);
                        if charin == 27 {
                            GOT_ESC.set(1);
                        }
                    }
                }
            }
        }
        0
    }

    /// C++ `Processchunks::handleInterrupt`.
    pub fn handle_interrupt(&mut self) {
        self.write_out(&format!(
            "{} chunks are still undone\n",
            self.size_job_array - self.num_done
        ));
        let mut command = String::new();
        while self.ans != 'Q'
            && self.ans != 'C'
            && self.ans != 'P'
            && !(self.ans == 'D' && self.queue == 0)
        {
            self.ans = ' ';
            command.clear();
            self.write_out(
                "Enter Q to kill all jobs and quit, P to finish running jobs then exit,\n",
            );
            if self.queue == 0 {
                self.write_out(" D machine_list to kill jobs and drop given machines,\n");
            }
            self.write_out(" or C to continue waiting: \n");
            // `QTextStream(stdin) >> command` reads one whitespace-delimited word.
            let mut line = String::new();
            let _ = io::stdin().lock().read_line(&mut line);
            let mut words = line.split_whitespace();
            command = words.next().unwrap_or("").trim().to_uppercase();
            self.ans = command.chars().next().unwrap_or('\0');
            if self.ans == 'D' && self.queue == 0 {
                command = words.next().unwrap_or("").trim().to_owned();
                if command.is_empty() {
                    self.write_out("\nEntry error: missing machine list\n");
                    self.ans = ' ';
                }
            }
        }
        if self.ans == 'D' {
            let drop_list: Vec<String> = command
                .split(',')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect();
            self.kill_processes(Some(&drop_list));
            self.hold_for_multi_proc_drop = self.num_multi_proc_jobs > 0 && self.queue == 0;
        } else {
            self.kill_processes(None);
        }
    }

    /// C++ `Processchunks::killProcesses`: Handle mAns:  killing, pausing,
    /// dropping machines, and exiting as required.
    pub fn kill_processes(&mut self, drop_list: Option<&[String]>) {
        if let Some(drop_list) = drop_list {
            self.drop_list = drop_list.to_vec();
        }
        self.pausing = false;
        if self.ans == 'P' {
            self.pausing = true;
        }
        if self.ans == 'C'
            || (self.ans == 'P' && self.queue == 0)
            || (self.ans == 'D' && self.queue != 0)
        {
            self.drop_list.clear();
            self.ans = ' ';
            //Continue with timer loop
            return;
        }
        if self.timer_id != 0 {
            self.timer_id = 0;
        }
        //Slow down the timer for killing
        self.kill = true;
        //killProcessOnNextMachine();
        //Run startKill on machines.  Increment mNumMachinesDropped for each matching
        //machine on the drop list.
        for i in 0..self.machine_list_size as usize {
            let active_machine = !self.machine_list[i].is_dropped();
            self.machine_list[i].start_kill("");
            if active_machine && self.machine_list[i].is_dropped() {
                self.num_machines_dropped += 1;
            }
        }
        self.kill_signal();
        self.timer_interval = 1000;
        self.timer_id = 1;
    }

    /// C++ `Processchunks::killSignal`: Loops through the machine list each
    /// time the counter goes off and mKill is on.  Sends kill signals and
    /// timeout instructions, decides when the kill is done, cleans up, and
    /// responds to the mAns request.
    pub fn kill_signal(&mut self) {
        let mut kill_done = true;
        for i in 0..self.machine_list_size as usize {
            //Send the machine a kill signal each time the timer goes off
            self.machine_list[i].kill_signal();
            //See if the kill is done
            if !self.machine_list[i].is_kill_finished() {
                kill_done = false;
            }
        }
        //If killDone was not turned off during the loop, then the kill is done
        if kill_done {
            //clean up kill
            //Kill the timer to clean up.  It will go back on for D and P.
            if self.timer_id != 0 {
                self.timer_id = 0;
            }
            //Reset kill variables and tell the machines to reset their kill variables.
            self.drop_list.clear();
            self.kill = false;
            for i in 0..self.machine_list_size as usize {
                self.machine_list[i].reset_kill();
            }
            //Handle error
            if self.ans == 'E' {
                if self.syncing == 0 {
                    self.write_out(&format!(
                        "ERROR: A CHUNK HAS FAILED {} times\n",
                        self.max_chunk_err
                    ));
                } else {
                    self.write_out("ERROR: A START, FINISH, OR SYNC CHUNK HAS FAILED\n");
                }
                self.cleanup_and_exit(4);
                return;
            }
            //Handle drop and pause by resuming processesing
            if (self.ans == 'D' && self.machine_list_size > self.num_machines_dropped)
                || self.ans == 'P'
            {
                self.ans = ' ';
                self.write_out("Resuming processing\n");
                self.start_timers();
                return;
            }
            //If not returning to the timer loop, then exit the program.
            self.write_out(
                "\nWhen you rerun with a different set of machines, be sure to use\n\
                 the -r flag to retain the existing results\n",
            );
            self.cleanup_and_exit(2);
        }
    }

    /// C++ `Processchunks::askGo`.
    pub fn ask_go(&mut self) -> bool {
        if self.just_go != 0 {
            return true;
        }
        self.write_out("Enter Y to proceed with the current set of machines: ");
        // `QTextStream(stdin) >> answer` skips whitespace and reads one char.
        let mut line = String::new();
        let _ = io::stdin().lock().read_line(&mut line);
        let answer = line.trim_start().chars().next().unwrap_or('\0');
        if answer == 'Y' || answer == 'y' {
            return true;
        }
        false
    }

    /// C++ `Processchunks::setupSshOpts`: Change mSshOpts if version is recent
    /// enough.
    pub fn setup_ssh_opts(&mut self) {
        if let Ok(ssh) = Command::new("ssh").arg("-V").output() {
            let mut version = self.extract_version(&String::from_utf8_lossy(&ssh.stderr));
            if version == -1 {
                version = self.extract_version(&String::from_utf8_lossy(&ssh.stdout));
            }
            //Check if version if >= to ssh version 3.9
            if version >= 309 {
                self.ssh_opts.insert(0, "-o ConnectTimeout=5 ".to_owned());
            }
        }
    }

    /// C++ `Processchunks::initMachineList`: Setup mMachineList with the
    /// values in mCpuList (nothing to do for queue).
    pub fn init_machine_list(
        &mut self,
        machine_name_list: &mut Vec<String>,
        num_cpus_list: &mut Vec<i32>,
        gpu_list: &mut Vec<i32>,
    ) {
        let mut num_cores;
        let mut num_cpus = 0;
        //Not implementing $IMOD_ALL_MACHINES since no one seems to have used it.
        if self.queue != 0 {
            return;
        }
        if self.cpu_list.contains('#') {
            exit_error(
                b"The machine list must contain : instead of # with or without the \
                  -G option",
            );
        }
        //Setup up machine names from mCpuList
        let cpu_array: Vec<String> = self
            .cpu_list
            .split(',')
            .filter(|part| !part.is_empty())
            .map(str::to_owned)
            .collect();
        /*The number of chunk (.com file) processes that this program can run at
        one time is added up below.  The CPU limit is necessary
        because of the OS's application-level 1024 process pipe limit (this
        program uses a pipe limit of 1012 for safety).  There are 4 pipes per
        chunk process because stdout is going to a file (if it wasn't, there would
        be 6 per process). Keeping the number chunk process pipes under the pipe
        limit leaves room for running vmstopy and killing processes.*/
        #[cfg(not(windows))]
        let num_cpus_limit = 240;
        #[cfg(windows)]
        let num_cpus_limit = 56;
        /*Now handling mixed up names (as in bear,shrek,bear) without extra probes
        and MachineHandler instances.  Identical machine names are always
        consolidated into one MachineHandler instance and the MachineHandler
        instance order is based on where a machine name first appeared in the
        list.*/
        //Consolidate list of machines and add up CPUs, allowing for multiple entries with
        // '#' and for multiple GPU specifications with ':'
        for cpu_machine in &cpu_array {
            let machine_split: Vec<&str> = cpu_machine.split(':').collect();
            let machine_name = machine_split[0].to_owned();
            if machine_split.len() == 1 {
                num_cores = 1;
                // If no complex entry, add a 0 to the list for a GPU
                if self.gpu_mode != 0 {
                    gpu_list.push(0);
                }
            } else if self.gpu_mode != 0 {
                // In GPU mode, each component is a positive GPU number
                num_cores = 0;
                for part in &machine_split[1..] {
                    let gpu_num = part.parse::<i32>();
                    if !matches!(gpu_num, Ok(value) if value >= 1) {
                        exit_error(&c_format_bytes(
                            "Incorrect entry for GPU number in: %s",
                            &[CArg::Str(cpu_machine)],
                        ));
                    }
                    num_cores += 1;
                    gpu_list.push(gpu_num.unwrap_or(0));
                }
            } else {
                // For regular machines, insist on one * and convert the number as numCores
                if machine_split.len() > 2 {
                    exit_error(&c_format_bytes(
                        "Multiple : characters in machine specification: %s",
                        &[CArg::Str(cpu_machine)],
                    ));
                }
                let parsed = machine_split[1].parse::<i32>();
                if !matches!(parsed, Ok(value) if value >= 1) {
                    exit_error(&c_format_bytes(
                        "Incorrect entry for number of cores in: %s",
                        &[CArg::Str(cpu_machine)],
                    ));
                }
                num_cores = parsed.unwrap_or(0);
                if num_cores > MAX_LOCAL_BY_NUM {
                    exit_error(&c_format_bytes(
                        "You cannot specify more than %d cores on a machine with \
                         machine:number",
                        &[CArg::Int(MAX_LOCAL_BY_NUM as i64)],
                    ));
                }
            }
            // Now test if there are too many CPUs
            if num_cpus + num_cores > num_cpus_limit {
                self.write_out(&format!(
                    "WARNING:the number of CPUs exceeds limit ({num_cpus_limit}).  CPU list will be truncated.\n"
                ));
                num_cores = num_cpus_limit - num_cpus;
                if num_cores == 0 {
                    break;
                }
            }
            num_cpus += num_cores;
            // Add a machine if the name has not occurred before
            if !machine_name_list
                .iter()
                .any(|name| name.eq_ignore_ascii_case(&machine_name))
            {
                machine_name_list.push(machine_name);
                num_cpus_list.push(num_cores);
            } else {
                //Increment the number of CPUs in an existing machine, unless GPU mode
                if self.gpu_mode != 0 {
                    exit_error(b"You can enter each machine only once with the -G option");
                }
                //Machine names could be mixed up (as in bear,bebop,bear).  Find the
                //correct machine name.
                for j in 0..machine_name_list.len() {
                    if machine_name.eq_ignore_ascii_case(&machine_name_list[j]) {
                        num_cpus_list[j] += num_cores;
                        break;
                    }
                }
            }
        }
        if machine_name_list.is_empty() {
            exit_error(b"No machines specified");
        }
        //OLD:Translate a single number into a list of localhost entries
        //Set the single number as the number of CPUs in the one instance of MachineHandler.
        if machine_name_list.len() == 1 {
            if let Ok(local_by_num) = machine_name_list[0].parse::<i32>() {
                if local_by_num > MAX_LOCAL_BY_NUM {
                    exit_error(&c_format_bytes(
                        "You cannot run more than %d chunks on localhost by \
                         entering a number",
                        &[CArg::Int(MAX_LOCAL_BY_NUM as i64)],
                    ));
                }
                if self.gpu_mode != 0 {
                    exit_error(
                        b"You cannot enter a number for the machine list with the -G option",
                    );
                }
                if local_by_num < 1 {
                    exit_error(b"A number entered for the machine list must be positive");
                }
                machine_name_list[0] = "localhost".to_owned();
                num_cpus_list[0] = local_by_num;
                num_cpus = local_by_num;
            }
        }
        //set max kills allowed to run at the same time, leaving room for misc processes
        #[cfg(not(windows))]
        {
            //1024 pipe limit
            self.max_kills = (1012 - (4 * num_cpus)) / 6;
        }
        #[cfg(windows)]
        {
            //62 whatsit limit
            self.max_kills = 60 - num_cpus;
        }
    }

    /// C++ `Processchunks::setupMachineList`: Setup mMachineList with the
    /// queue name or the values in mCpuList.
    pub fn setup_machine_list(
        &mut self,
        machine_name_list: &mut [String],
        num_cpus_list: &[i32],
        gpu_list: &[i32],
    ) {
        let this = self as *mut Self;
        //Not implementing $IMOD_ALL_MACHINES since no one seems to have used it.
        if self.queue != 0 {
            self.num_cpus = self.queue;
            self.queue_param_list = self
                .cpu_list
                .split_whitespace()
                .map(str::to_owned)
                .collect();
            //Parse mCpuList into mQueueComand and mQueueParamList
            if self.cpu_list.is_empty() {
                exit_error(b"Queue command doesn't exist.");
            }
            self.queue_command = self.queue_param_list.remove(0);
            self.machine_list_size = 1;
            //OLD: For a queue, make a CPU list that is all the same name
            //For a queue, create a single MachineHandler instance.
            self.machine_list = vec![MachineHandler::new()];
            let queue_name = self.queue_name.clone();
            let queue = self.queue;
            let machine: *mut MachineHandler = &mut self.machine_list[0];
            unsafe {
                (*machine).setup(&mut *this, &queue_name, queue, gpu_list, 0);
            }
            // Initialize the queue with entered command
            self.escaped_remote_dir_path = self.get_remote_dir().replace(' ', "\\ ");
            let init_queue = self.init_queue.clone();
            let status =
                self.run_generic_queue_command(init_queue.as_deref(), self.wait_for_queue_init);
            if status != 0 {
                exit_error(&c_format_bytes(
                    "Command to initialize queue returned with error %d",
                    &[CArg::Int(status as i64)],
                ));
            }
        } else {
            self.machine_list_size = 0;
            for name in machine_name_list.iter() {
                //When machines where probed, the failures where removed by setting their
                //name to "".  Ignore removed machines.
                if !name.is_empty() {
                    self.machine_list_size += 1;
                }
            }
            //Setup machine list.  Do this only once after the number of CPUs is finalized.
            if self.is_verbose(&self.decorated_class_name.clone(), "setupMachineList", 1) {
                self.write_out("mMachineList:\n");
            }
            self.num_cpus = 0;
            self.machine_list = (0..self.machine_list_size)
                .map(|_| MachineHandler::new())
                .collect();
            let mut new_index = 0;
            let mut base_index = 0;
            for i in 0..machine_name_list.len() {
                if !machine_name_list[i].is_empty() {
                    let machine: *mut MachineHandler = &mut self.machine_list[new_index];
                    unsafe {
                        (*machine).setup(
                            &mut *this,
                            &machine_name_list[i],
                            num_cpus_list[i],
                            gpu_list,
                            base_index as usize,
                        );
                    }
                    self.num_cpus += num_cpus_list[i];
                    if self.is_verbose(&self.decorated_class_name.clone(), "setupMachineList", 1) {
                        let name = self.machine_list[new_index].get_name().to_owned();
                        self.write_out(&format!("{new_index}:{name}\n"));
                    }
                    new_index += 1;
                }
                base_index += num_cpus_list[i];
            }
            if self.is_verbose(&self.decorated_class_name.clone(), "setupMachineList", 1) {
                self.write_out("\n");
            }
        }
    }

    /// C++ `Processchunks::runGenericQueueCommand`: Run a generic entered
    /// command on the queue with -C:<command>, escaping spaces.
    pub fn run_generic_queue_command(&mut self, command: Option<&str>, wait_msec: i32) -> i32 {
        let Some(command) = command else {
            return 0;
        };
        if command.is_empty() || self.queue == 0 {
            return 0;
        }
        let queue_com = format!("C:{command}");
        let mut params = self.queue_param_list.clone();
        params.push("-w".to_owned());
        params.push(self.escaped_remote_dir_path.clone());
        params.push("-a".to_owned());
        params.push(queue_com);
        let mut output = Vec::new();
        if self.verbose != 0 {
            self.write_out(&format!("queue command: {}\nparams:\n", self.queue_command));
            for (i, param) in params.iter().enumerate() {
                self.write_out(&format!("{i}: {param}\n"));
            }
            self.write_out("\n");
        }
        let queue_command = self.queue_command.clone();
        let num_lines = if self.verbose != 0 { 100 } else { 0 };
        self.run_generic_process(&mut output, &queue_command, &params, num_lines, wait_msec)
    }

    /// C++ `Processchunks::setupHostRoot`.
    pub fn setup_host_root(&mut self) {
        if let Ok(hostname) = Command::new("hostname").output() {
            let temp = String::from_utf8_lossy(&hostname.stdout).into_owned();
            self.full_host_name = temp.to_lowercase();
            let args_none: Vec<&str> = self.full_host_name.split('.').collect();
            self.host_root = args_none[0].to_owned();
            if args_none.len() > 1 {
                self.host_root2 = format!("{}.{}", self.host_root, args_none[1]);
            }
            if self.is_verbose(&self.decorated_class_name.clone(), "setupHostRoot", 1) {
                self.write_out(&format!("mHostRoot:{}\n", self.host_root));
            }
        } else {
            exit_error(b"Unable to run the hostname command");
        }
    }

    /// C++ `Processchunks::nameIsLocalHost`: Test if machine is local host.
    pub fn name_is_local_host(&self, mach_name: &str) -> bool {
        let mach = mach_name.to_lowercase();
        mach == "localhost"
            || mach == self.host_root
            || mach == self.host_root2
            || mach == self.full_host_name
    }

    /// C++ `Processchunks::setupComFileJobs`: Sets up ComFileJobs for single
    /// file or multi-file processing.  This functionality should be scaleable
    /// because the number of chunks can be very large.  The file name
    /// limitations max out the number of chunks at around 100,000.  Coming
    /// close to this limit is not unrealistic, especially with PEET processing.
    pub fn setup_com_file_jobs(&mut self) {
        let mut com_file_array: Vec<String> = Vec::new();
        let reg_filters = ["-???", "-????", "-?????", "-??????"];
        //?- is a special character and gets a warning about -trigraphs
        let sync_filters = ["-???-sync", "-????-sync", "-?????-sync", "-??????-sync"];
        if self.is_verbose(&self.decorated_class_name.clone(), "setupComFileJobs", 1) {
            self.write_out(&format!(
                "current path:{}\n",
                self.current_dir.to_string_lossy()
            ));
        }
        //OLD:Make the list for a single file
        //For a single file or each multiple file, one element is added to comFileArray.
        if self.single_file != 0 || self.multiple_files != 0 {
            for i in 0..1.max(self.multiple_files) {
                let mut arg_name = Vec::new();
                pip_get_non_option_arg(i + 1, &mut arg_name);
                let mut root_name = String::from_utf8_lossy(&arg_name).into_owned();

                // If it already ends in .com or .pcm, check that it exists
                if root_name.ends_with(".com") || root_name.ends_with(".pcm") {
                    if !self.current_dir.join(&root_name).exists() {
                        exit_error(&c_format_bytes(
                            "The %s command file %s does not exist",
                            &[
                                CArg::Str(if self.single_file != 0 { "single" } else { "" }),
                                CArg::Str(&root_name),
                            ],
                        ));
                    }
                } else {
                    // Otherwise look for default first then for other one
                    let mut alt_name = root_name.clone();
                    root_name.push_str(&self.com_extension);
                    alt_name.push_str(if self.com_extension == ".com" {
                        ".pcm"
                    } else {
                        ".com"
                    });
                    if i == 0
                        && self.current_dir.join(&root_name).exists()
                        && self.current_dir.join(&alt_name).exists()
                    {
                        exit_error(&c_format_bytes(
                            "You must enter the full command file name with extension because\
                             both %s and %s exist",
                            &[CArg::Str(&root_name), CArg::Str(&alt_name)],
                        ));
                    }

                    if !self.current_dir.join(&root_name).exists() {
                        if !self.current_dir.join(&alt_name).exists() {
                            exit_error(&c_format_bytes(
                                "The %s command file %s or %s does not exist",
                                &[
                                    CArg::Str(if self.single_file != 0 { "single" } else { "" }),
                                    CArg::Str(&root_name),
                                    CArg::Str(&alt_name),
                                ],
                            ));
                        }
                        root_name = alt_name;
                    }
                }

                // Assign the com extension from the first file found
                let right4 = root_name
                    .get(root_name.len().saturating_sub(4)..)
                    .unwrap_or("")
                    .to_owned();
                if i == 0 {
                    self.com_extension = right4;
                } else if right4 != self.com_extension {
                    exit_error(&c_format_bytes(
                        "All command files must have the same extension; %s does not match \
                         the preceding ones",
                        &[CArg::Str(&root_name)],
                    ));
                }
                com_file_array.push(root_name);
            }
        } else {
            //Build up lists in order -nnn, -nnnn, -nnnnn*, which should work both for
            //lists that are all 5 digits or lists that are 3, 4, 5 digits
            //Put -start.com on front and -finish.com on end
            // Loop twice, first on default extension then on alternate one if there are no
            // numeric coms by the default extension
            let mut num_numeric_command_files;
            let mut com_dir = PathBuf::from(".");
            let mut root_path = String::new();
            // `QDir::cleanPath`: collapse separators and dot components.
            let cleaned = self.root_name().replace("//", "/");
            let mut root_file = cleaned.trim_end_matches('/').to_owned();
            if root_file.is_empty() && cleaned.starts_with('/') {
                root_file = "/".to_owned();
            }
            if let Some(slash) = root_file.rfind('/') {
                root_path = root_file[..slash + 1].to_owned();
                com_dir = PathBuf::from(&root_path);
                root_file = root_file[slash + 1..].to_owned();
            }
            for ext_loop in 0..2 {
                num_numeric_command_files = 0;

                // Compose start and finish names for general use
                let start_com_file = format!("{root_file}-start{}", self.com_extension);
                let finish_com_file = format!("{root_file}-finish{}", self.com_extension);

                //Add start com file
                if com_dir.join(&start_com_file).exists() {
                    com_file_array.push(format!("{root_path}{start_com_file}"));
                }

                //Add numeric com files
                let mut filters: Vec<String> = Vec::new();
                let mut list: Vec<String> = Vec::new();
                //Using QDirIterator directly is a more scaleable solution then using
                //QDir::entryList.

                // Add -nnn, -nnnn, -nnnnn, -nnnnnn
                // Note: 100000 files takes 1.5 sec to scan for 3-5 digits, another 0.5 sec for 6.
                // Per extension...  Almost all the time is intrinsic to iterating to build the list
                for digits in 3..=6 {
                    self.build_filters(
                        &root_file,
                        reg_filters[digits - 3],
                        sync_filters[digits - 3],
                        &mut filters,
                    );
                    if let Ok(entries) = fs::read_dir(&com_dir) {
                        for entry in entries.flatten() {
                            let file_name = entry.file_name().to_string_lossy().into_owned();
                            // QDirIterator name filters: `?` matches exactly one
                            // character, everything else matches itself.
                            let matches_filter = filters.iter().any(|filter| {
                                let f: Vec<char> = filter.chars().collect();
                                let n: Vec<char> = file_name.chars().collect();
                                f.len() == n.len()
                                    && f.iter().zip(&n).all(|(a, b)| *a == '?' || a == b)
                            });
                            if matches_filter
                                && file_name != start_com_file
                                && file_name != finish_com_file
                            {
                                list.push(format!("{root_path}{file_name}"));
                            }
                        }
                    }

                    list.sort();
                    let clean_buf = format!("-\\D{{{digits},{digits}}}(-sync){{0,1}}\\");
                    self.cleanup_list(&clean_buf, &mut list);
                    num_numeric_command_files += list.len() as i32;
                    com_file_array.append(&mut list);
                    filters.clear();
                    list.clear();
                }

                //Add finish com file
                if com_dir.join(&finish_com_file).exists() {
                    com_file_array.push(format!("{root_path}{finish_com_file}"));
                }
                if num_numeric_command_files > 0 {
                    break;
                }

                // If there are no numeric files, either look with other extension or give error
                if ext_loop != 0 {
                    exit_error(&c_format_bytes(
                        "There are no command files matching %s-nnn.com or %s-nnn.pcm",
                        &[CArg::Str(self.root_name()), CArg::Str(self.root_name())],
                    ));
                } else {
                    self.com_extension = if self.com_extension == ".com" {
                        ".pcm".to_owned()
                    } else {
                        ".com".to_owned()
                    };
                    com_file_array.clear();
                }
            }
        }

        // Check for current limit
        if self
            .current_dir
            .join(format!(
                "{}-1000000{}",
                self.root_name(),
                self.com_extension
            ))
            .exists()
            || self
                .current_dir
                .join(format!(
                    "{}-1000000-sync{}",
                    self.root_name(),
                    self.com_extension
                ))
                .exists()
        {
            exit_error(b"Cannot process more than 999999 chunks");
        }
        if self.is_verbose(&self.decorated_class_name.clone(), "setupComFileJobs", 1) {
            self.write_out("comFileArray:\n");
            for (i, name) in com_file_array.iter().enumerate() {
                self.write_out(&format!("{i}:{name}\n"));
            }
            self.write_out("\n");
        }

        //Build mJobArray from comFileArray.
        //set up flag list and set up which chunk to copy the log from, the first
        //non-sync if any, otherwise just the first one.
        self.size_job_array = com_file_array.len() as i32;
        self.com_file_jobs = Some(ComFileJobs::new(
            com_file_array,
            self.single_file != 0,
            self.com_extension.clone(),
        ));
        for i in 0..self.size_job_array {
            if self.get_com_file_jobs().get_flag(i as usize) != CHUNK_SYNC
                && self.copy_log_index == -1
            {
                //Setting mCopyLogIndex to the first non-sync log
                self.copy_log_index = i;
            }
        }
        if self.copy_log_index == -1 {
            self.copy_log_index = 0;
        }
    }

    /// C++ `Processchunks::probeMachines`: Probe machines by running the "w"
    /// command.  Drop machines that don't respond.
    pub fn probe_machines(&mut self, machine_name_list: &mut [String]) -> bool {
        //Remove the old checkfile
        if let Some(check_file) = &self.check_file {
            if self.current_dir.join(check_file).exists() {
                let _ = fs::remove_file(self.current_dir.join(check_file));
            }
        }
        //probe machines and get all the verifications unless etomo is running it or its a queue
        if self.skip_probe || self.just_go != 0 {
            return true;
        }
        let first_name = machine_name_list[0].clone();
        let retval = self.name_is_local_host(&first_name) && machine_name_list.len() == 1;
        //Windows processchunks only runs on the local machine.
        if !retval {
            self.write_out("Probing machine connections and loads...\n");
            #[cfg(not(windows))]
            let local_command = "w";
            #[cfg(windows)]
            let local_command = "imodwincpu";
            let local_params: Vec<String> = Vec::new();
            let remote_command = "ssh";
            let mut remote_params: Vec<String> = vec!["-x".to_owned()];
            let mut remote_win_params: Vec<String> = vec!["-x".to_owned()];
            let mut uname_params: Vec<String> = vec!["-x".to_owned()];
            for opt in &self.ssh_opts {
                remote_params.push(opt.clone());
                remote_win_params.push(opt.clone());
                uname_params.push(opt.clone());
            }
            remote_params.push("placeholder".to_owned());
            remote_params.push("hostname ; w".to_owned());
            remote_win_params.push("placeholder".to_owned());
            remote_win_params.push("bash".to_owned());
            remote_win_params.push("--login".to_owned());
            remote_win_params.push("-c".to_owned());
            remote_win_params.push("\"hostname ; imodwincpu\"".to_owned());
            uname_params.push("placeholder".to_owned());
            uname_params.push("uname -s".to_owned());
            //Probing the machines and building a new cpu array from the ones that
            //respond.
            let mut status;
            if self.is_verbose(&self.decorated_class_name.clone(), "probeMachines", 1) {
                self.write_out(&format!(
                    "machineNameList.size():{}\n",
                    machine_name_list.len()
                ));
            }
            for i in 0..machine_name_list.len() {
                let mach_name = machine_name_list[i].clone();
                let mut output = Vec::new();
                if self.name_is_local_host(&mach_name) {
                    self.write_out(&format!("{mach_name}\n"));
                    status = self.run_generic_process(
                        &mut output,
                        local_command,
                        &local_params,
                        1,
                        30000,
                    );
                } else {
                    //Use uname to find out whether machName is a Windows system.  Use
                    //imodwincpu instead of w for Windows systems.
                    uname_params[self.ssh_opts.len() + 1] = mach_name.clone();
                    let uname_status = self.run_generic_process(
                        &mut output,
                        remote_command,
                        &uname_params,
                        0,
                        30000,
                    );
                    let mut use_win = false;
                    if uname_status == 0 && !output.is_empty() {
                        let uname_output = String::from_utf8_lossy(&output).to_lowercase();
                        if uname_output.contains("cygwin") || uname_output.contains("nt") {
                            use_win = true;
                        }
                    }
                    let params = if use_win {
                        &mut remote_win_params
                    } else {
                        &mut remote_params
                    };
                    params[self.ssh_opts.len() + 1] = mach_name.clone();
                    let params = params.clone();
                    status =
                        self.run_generic_process(&mut output, remote_command, &params, 2, 30000);
                }
                //status can also be set to 1 on the local machine if it times out.
                //No longer testing for 141 because no longer supporting SGI
                if status != 0 {
                    self.write_out(&format!(
                        "Dropping {mach_name} from list because it does not respond\n\n"
                    ));
                    //Drops failed machine from the machine list
                    machine_name_list[i] = String::new();
                }
            }
        }
        retval
    }

    /// C++ `Processchunks::readCheckFile`: Look for commands in mCheckFile.
    /// CheckFile is kept open so already processed commands are not read
    /// twice.  Return true if a valid command is found in the check file.
    /// Handle a deleted check file by closing and reopening the checkFile at
    /// intervals.
    pub fn read_check_file(&mut self) -> bool {
        //Handle mCheckFile
        let Some(check_file) = self.check_file.clone() else {
            return false;
        };
        if !self.current_dir.join(&check_file).exists() {
            return false;
        }
        let mut opened_file = false;
        if self.check_file_handle.is_none() {
            self.check_file_handle = fs::File::open(self.current_dir.join(&check_file))
                .ok()
                .map(io::BufReader::new);
            opened_file = true;
        }
        self.check_file_reconnect -= 1;
        if self.check_file_handle.is_some() {
            // `QTextStream::readLine` returns a null string at end of file.
            let mut com_line: Option<String> = {
                let mut line = String::new();
                match self
                    .check_file_handle
                    .as_mut()
                    .unwrap()
                    .read_line(&mut line)
                {
                    Ok(n) if n > 0 => Some(line.trim_end_matches(['\n', '\r']).to_owned()),
                    _ => None,
                }
            };
            //Go past the lines in the file that have already been read.
            if opened_file && !self.save_check_file_lines.is_empty() {
                let mut i = 0;
                //Get the next line in the file while comLine is the same as the saved line.
                while com_line.is_some()
                    && i < self.save_check_file_lines.len()
                    && com_line.as_deref() == Some(self.save_check_file_lines[i].as_str())
                {
                    let mut line = String::new();
                    com_line = match self
                        .check_file_handle
                        .as_mut()
                        .unwrap()
                        .read_line(&mut line)
                    {
                        Ok(n) if n > 0 => Some(line.trim_end_matches(['\n', '\r']).to_owned()),
                        _ => None,
                    };
                    i += 1;
                }
                //Remove lines that are different - that's where it will start reading the new file.
                if i == 0 {
                    self.save_check_file_lines.clear();
                } else {
                    // `for (j = i; j < size; j++) removeAt(j)` removes every
                    // other element from `i` on, exactly as the source does.
                    let mut j = i;
                    while j < self.save_check_file_lines.len() {
                        self.save_check_file_lines.remove(j);
                        j += 1;
                    }
                }
            }

            //Process the lines in the check file.
            while let Some(line) = com_line.clone() {
                self.save_check_file_lines.push(line.clone());
                self.ans = line
                    .chars()
                    .next()
                    .map(|c| c.to_ascii_uppercase())
                    .unwrap_or('\0');
                if self.ans == 'D' && line.len() > 1 {
                    //machine name(s) are required
                    let drop_list: Vec<String> = line[1..]
                        .trim()
                        .split(',')
                        .filter(|part| !part.is_empty())
                        .map(str::to_owned)
                        .collect();
                    self.kill_processes(Some(&drop_list));
                    return true;
                } else if self.ans == 'P' {
                    // Here is a quick kludge to prevent "need to restart" and chunk error
                    // messages when pausing, since BRT needs to exit with an error to prevent
                    // CHUNK DONE in the log file.
                    if self.num_multi_proc_jobs > 0 {
                        let dflt_cmds = format!("{}.cmds", self.root_name());

                        // Look at the possible top level check file if it is different and
                        // if it has a Q, suppress error messages
                        if !check_file.contains(&dflt_cmds) {
                            if let Ok(dflt_file) = fs::File::open(&dflt_cmds) {
                                let mut dflt_line = String::new();
                                if matches!(io::BufReader::new(dflt_file).read_line(&mut dflt_line), Ok(n) if n > 0)
                                {
                                    let dflt_line = dflt_line.trim_end_matches(['\n', '\r']);
                                    if !dflt_line.is_empty() {
                                        let dflt_ans = dflt_line
                                            .chars()
                                            .next()
                                            .map(|c| c.to_ascii_uppercase())
                                            .unwrap_or('\0');
                                        if dflt_ans == 'Q' {
                                            self.ignore_pausing_errors = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    self.kill_processes(None);
                    return true;
                } else if self.ans == 'Q' {
                    self.kill_processes(None);
                    return true;
                } else {
                    self.write_out(&format!("BAD COMMAND IGNORED: {line}\n"));
                }
                let mut next = String::new();
                com_line = match self
                    .check_file_handle
                    .as_mut()
                    .unwrap()
                    .read_line(&mut next)
                {
                    Ok(n) if n > 0 => Some(next.trim_end_matches(['\n', '\r']).to_owned()),
                    _ => None,
                };
            }
            if self.check_file_reconnect <= 0 {
                self.check_file_handle = None;
                self.check_file_reconnect = CHECK_FILE_RECONNECT_RESET;
            }
        }
        false
    }

    /// C++ `Processchunks::exitIfDropped`: Stop if all have now been dropped
    /// out or all have failed and none done.
    pub fn exit_if_dropped(&mut self, min_fail: i32, fail_tot: i32, assign_tot: i32) -> bool {
        if self.is_verbose(&self.decorated_class_name.clone(), "exitIfDropped", 2) {
            self.write_out(&format!(
                "{}:exitIfDropped:minFail={min_fail},failTot:{fail_tot},assignTot:{assign_tot},mDropCrit:{},mNumCpus:{},mNumDone:{},mPausing:{},mSyncing:{},mQueue:{},mMachineListSize:{}\n",
                self.decorated_class_name,
                self.drop_crit,
                self.num_cpus,
                self.num_done,
                self.pausing as i32,
                self.syncing,
                self.queue,
                self.machine_list_size
            ));
        }
        if min_fail >= self.drop_crit {
            self.write_out("ERROR: ALL MACHINES HAVE BEEN DROPPED DUE TO FAILURES\n");
            if self.queue == 0 {
                // DNM: this function does return, so we need to return with true if exiting
                self.cleanup_and_exit(1);
                return true;
            } else {
                self.ans = 'E';
                // DNM: The kill will eventually exit so return after it also
                self.kill_processes(None);
                return true;
            }
        }

        // If a machine was dropped and running chunks are done, now reallocate the machines
        if self.hold_for_multi_proc_drop && assign_tot == 0 {
            let err = self.divide_machines_for_jobs();
            if err != 0 {
                if err == 1 {
                    self.write_out(
                        "ERROR: THERE IS ONLY ONE CPU LEFT, NOT ENOUGH FOR MULTIPROCESSOR JOBS\n",
                    );
                } else {
                    self.write_out("ERROR: THERE ARE NO GPUs LEFT ON NON_DROPPED MACHINES\n");
                }
                self.cleanup_and_exit(1);
                return true;
            }
            self.hold_for_multi_proc_drop = false;
        }

        if self.pausing && assign_tot == 0 {
            self.write_out(
                "All previously running chunks are done - exiting as requested\n\
                 Rerun with -r to resume and retain existing results\n",
            );
            self.cleanup_and_exit(2);
            return true;
        }
        if assign_tot == 0 && self.num_done == 0 {
            // DNM: needed to compare with mMachineListSize not mNumCpus, but then the tests
            // for failure of first sync were not reached, so those tests are included in this
            // one test, since they all took the same actions
            if fail_tot == self.machine_list_size
                && (self.syncing == 0
                    || self.queue == 0
                    || min_fail == self.queue
                    || (self.syncing != 0 && self.machine_list[0].get_failure_count() > 1))
            {
                self.write_out("ERROR: NO CHUNKS HAVE WORKED AND EVERY MACHINE HAS FAILED\n");
                self.cleanup_and_exit(1);
                return true;
            }
        }
        false
    }

    /// C++ `Processchunks::handleChunkDone`: Handle chunk done: deassign, get
    /// rid of chunk errors.  When it is the first chunk done, issue drop
    /// messages; copy the log for the first non-sync chunk.  Return true if
    /// all chunks are done.
    pub fn handle_chunk_done(
        &mut self,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
        job_index: i32,
    ) -> bool {
        //If it is DONE, then set flag to done and deassign
        //Exonerate the machine from chunk errors if this chunk
        //gave a previous chunk error
        process.set_flag(CHUNK_DONE);
        process.invalidate_job();
        machine.set_failure_count(0);

        // For a non-sync chunk, keep track of the longest time for the machine and overall
        if self.syncing == 0 {
            let time = process.get_elapsed_time();
            machine.set_slowest_time(time);
            if self.slowest_time < 0 || time > self.slowest_time {
                self.slowest_time = time;
            }
            self.slow_time_count += 1;
            if self.is_verbose(&self.decorated_class_name.clone(), "handleChunkDone", 1) {
                self.write_out(&format!(
                    "{}:handleChunkDone:slowest time, machine {} - {}  overall {} - {}\n",
                    self.decorated_class_name,
                    machine.get_slowest_time(),
                    machine.get_slow_time_count(),
                    self.slowest_time,
                    self.slow_time_count
                ));
            }
        }
        // But if we were syncing, reset the longest time data
        else {
            self.slowest_time = -1;
            self.slow_time_count = 0;
            for i in 0..self.machine_list_size as usize {
                self.machine_list[i].set_slowest_time(-1);
            }
        }

        self.syncing = 0;
        if process.get_num_chunk_err() != 0 {
            machine.set_chunk_erred(false);
        }
        self.num_done += 1;
        let elapse_buf = c_format(
            " in %.2f sec",
            &[CArg::Dbl(process.get_elapsed_time() as f64 / 1000.)],
        );
        self.write_out(&format!(
            "{} finished on {}{elapse_buf}\n",
            process.get_com_file_name(),
            machine.get_name()
        ));
        process.print_warnings(machine.get_name());
        if self.single_file != 0 {
            if !self.skip_probe {
                self.cleanup_and_exit(0);
            }
            return true;
        }
        //If this is the first one done, issue drop messages now
        //on ones that chunk errored and exceeded failure count
        if !self.any_done {
            for i in 0..self.machine_list_size as usize {
                if self.machine_list[i].get_failure_count() >= self.drop_crit
                    && self.machine_list[i].is_chunk_erred()
                {
                    let name = self.machine_list[i].get_name().to_owned();
                    self.write_out(&format!("Dropping {name}\n"));
                    self.hold_for_multi_proc_drop = self.num_multi_proc_jobs > 0 && self.queue == 0;
                    self.machine_list[i].set_internal_dropped();
                }
            }
        }
        self.any_done = true;
        //copy the log for the first non-sync chunk
        if job_index == self.copy_log_index {
            let root_log_name = format!("{}.log", self.root_name());
            //Backup the root log if it exists
            imod_backup_file(&root_log_name);
            let mut root_log = fs::File::create(&root_log_name);
            if root_log.is_err() {
                self.handle_file_system_bug(&format!("open {root_log_name}"));
                root_log = fs::File::create(&root_log_name);
                if root_log.is_err() {
                    self.write_out(&format!(
                        "Warning: Unable to write copied chunk log {root_log_name}\n"
                    ));
                    return false;
                }
            }
            let mut root_log = root_log.unwrap();
            let _ = write!(
                root_log,
                "THIS FILE IS JUST THE LOG FOR ONE CHUNK AND WAS COPIED BY PROCESSCHUNKS FROM {}\n",
                process.get_log_file_name()
            );
            let log = process.read_all_log_file();
            if !log.is_empty() {
                // `writeStream << log.data()` writes the bytes up to a NUL.
                let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
                let _ = root_log.write_all(&log[..end]);
            }
            drop(root_log);
        }
        false
    }

    /// C++ `Processchunks::handleLogFileError`: Looks for and print an error
    /// message in log file.  If the chunk has errored too many times, set mAns
    /// to E and kill jobs.  Return false the chunk has errored too many times.
    pub fn handle_log_file_error(
        &mut self,
        error_mess: &mut String,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
    ) -> bool {
        process.get_error_message_from_log(error_mess);
        self.handle_error(Some(error_mess), machine, process, false)
    }

    /// C++ `Processchunks::handleError`: Print an error message.  If the chunk
    /// has errored too many times, set mAns to E and kill jobs.  Return false
    /// the chunk has errored too many times.
    pub fn handle_error(
        &mut self,
        error_mess: Option<&String>,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
        hung_job: bool,
    ) -> bool {
        process.increment_num_chunk_err();
        let num_err = process.get_num_chunk_err();
        machine.set_chunk_erred(true);

        // For multiple files, mark chunk to be skipped if it fails too many times
        if num_err >= self.max_chunk_err && self.multiple_files != 0 {
            process.set_flag(CHUNK_TO_SKIP);
            self.num_skipped += 1;
        }
        //Otherwise give up if the chunk errored too many  times: and
        //for a sync chunk that is twice or once if one machine, or up to 3 times for hung job
        else if num_err >= self.max_chunk_err
            || (self.syncing != 0
                && ((!hung_job && (self.machine_list_size == 1 || num_err >= 2))
                    || (hung_job && num_err >= (self.machine_list_size + 1).min(3))))
        {
            process.print_too_many_errors_message(num_err);
            if let Some(error_mess) = error_mess {
                if !error_mess.is_empty() {
                    self.write_out(&format!("{error_mess}\n"));
                }
            }
            self.ans = 'E';

            // A hung job needs to stay valid to get killed in this process, since it is not
            // a single process kill
            if !hung_job {
                process.invalidate_job();
            }
            self.kill_processes(None);
            return false;
        }
        true
    }

    /// C++ `Processchunks::handleComProcessNotDone`: Handle timeouts and
    /// missing qid files for queues.  Handle com never started and process
    /// ended - drop.  Handle ssh error - drop.  Handle com not started yet.
    /// Handle log doen't exist and timeout - drop.
    pub fn handle_com_process_not_done(
        &mut self,
        dropout: &mut bool,
        drop_mess: &mut String,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
        need_kill: &mut bool,
    ) {
        *need_kill = false;
        if self.queue != 0 && !process.qid_file_exists() {
            //For a queue, the qid file should be there
            *dropout = true;
            *drop_mess = "it failed to be submitted to queue".to_owned();
        } else if self.queue == 0 {
            //Either there is no log file or the .py is still present:
            //OLD:check the ssh file and accumulate timeout
            //OLD:If the ssh file is non empty check for errors there
            //Look for cd or ssh errors in stdout and stderr.  For a queue check the
            //.job file.
            if process.get_ssh_error(drop_mess) {
                //A cd or ssh error is very serious - stop using this machine.
                *dropout = true;
                machine.set_internal_dropped();
                machine.set_failure_count(self.drop_crit);
                self.hold_for_multi_proc_drop = self.num_multi_proc_jobs > 0;
            }
            if !*dropout && process.is_finished_signal_received() {
                *dropout = true;
            } else if !*dropout {
                //OLD:if log file doesn't exist, check the pid
                //OLD:and give up after timeout
                //Check for timeout
                if process.is_start_process_timed_out(RUN_PROCESS_TIMEOUT) {
                    *dropout = true;

                    // Either kill it if there is a PID, or just close it to avoid errors on restart
                    if process.is_pid_empty() {
                        process.close_process();
                    } else {
                        *need_kill = true;
                    }
                }
            }
            if !*dropout && !process.is_pid_empty() {
                // Test the elapsed time relative to other chunks, and then the log file time
                // to see if job looks hung
                let elapsed = process.get_elapsed_time();
                let factor = if self.syncing != 0 {
                    self.slow_sync_factor
                } else {
                    1.
                };
                let too_slow = self.process_too_slow(
                    elapsed,
                    self.slowest_time,
                    self.slow_time_count,
                    self.slow_overall_crit * factor,
                ) || self.process_too_slow(
                    elapsed,
                    machine.get_slowest_time(),
                    machine.get_slow_time_count(),
                    self.slow_machine_crit * factor,
                );

                // Only test for a log file time if it will be conclusive
                if self.syncing != 0 {
                    *need_kill = ((self.slow_sync_factor > 0. && too_slow)
                        || self.slow_sync_factor == 0.)
                        && process.is_log_file_older_than(self.slow_sync_log_timeout);
                } else {
                    *need_kill = too_slow && process.is_log_file_older_than(self.slow_log_timeout);
                }
                *dropout = *need_kill;
            }
        }
    }

    /// C++ `Processchunks::processTooSlow`: Evaluate whether an elapsed time
    /// is too long based on the criterion, current slowest time and number of
    /// times that is based on.  Ignore if the count is not at least 2 or if
    /// there is no criterion; make the criterion even bigger for small count.
    pub fn process_too_slow(
        &self,
        elapsed: i32,
        slowest_time: i32,
        slow_time_count: i32,
        mut slow_crit: f32,
    ) -> bool {
        let max_derate = 4.0f32;
        if slow_time_count < 2 || slow_crit <= 0. {
            return false;
        }
        if (slow_time_count as f32) < max_derate {
            slow_crit *= max_derate / slow_time_count as f32;
        }
        elapsed as f32 > slowest_time as f32 * slow_crit
    }

    /// C++ `Processchunks::handleDropOut`: remove the assignment, mark chunk
    /// as to be done, issue messages including machine drops.
    pub fn handle_drop_out(
        &mut self,
        no_chunks: &mut bool,
        drop_mess: &mut String,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
        error_mess: &mut String,
        need_kill: bool,
    ) {
        let mut num_running = 0;

        if need_kill {
            self.write_out(&format!(
                "Killing {} on {}, it appears to be hung up    [PRC3]\n",
                process.get_com_file_name(),
                machine.get_name()
            ));
        } else if !machine.is_dropped() && !(self.pausing && self.ignore_pausing_errors) {
            self.write_out(&format!(
                "{} failed on {}{}\n",
                process.get_com_file_name(),
                machine.get_name(),
                if process.get_flag() == CHUNK_TO_SKIP {
                    " - giving up and skipping it"
                } else {
                    " - need to restart    [PRC2]"
                }
            ));
            if !error_mess.is_empty() {
                self.write_out(&format!("{error_mess}\n"));
            }
        }
        process.set_flag_not_done(self.single_file != 0);
        if self.syncing != 0 {
            self.syncing = 1;
        }

        // Keep a hung job valid until it is killed
        if !need_kill {
            process.invalidate_job();
        }
        *no_chunks = false;
        machine.increment_failure_count();

        // DNM note: isDropped() returns mDropped which is true only if machine was dropped
        // from an external D list
        // When a dropping message is issued for any reason, set InternalDropped flag
        if !machine.is_dropped() && machine.get_failure_count() >= self.drop_crit {
            if error_mess.is_empty() {
                process.get_error_message_from_output(error_mess);
                if !error_mess.is_empty() {
                    self.write_out(&format!("{error_mess}\n"));
                }
            }

            // Get running job count
            for cpu_index in 0..machine.get_num_cpus().max(0) as usize {
                if machine.is_job_valid(cpu_index) {
                    num_running += 1;
                }
            }

            if drop_mess.is_empty() {
                *drop_mess = "it failed (with ".to_owned();
                if !machine.is_chunk_erred() {
                    drop_mess.push_str("time out or crash");
                } else {
                    drop_mess.push_str("chunk error");
                }
                drop_mess.push_str(") %1 times in a row");
                *drop_mess = drop_mess.replacen("%1", &machine.get_failure_count().to_string(), 1);
            }

            // Hold if no chunks done and they are chunk errors (?), or if anything else is
            // still running on the machine
            if num_running != 0 || (!self.any_done && machine.is_chunk_erred()) {
                self.write_out("Holding off on using ");
            } else {
                self.write_out("Dropping ");
                machine.set_internal_dropped();
            }
            self.write_out(&format!("{} - {drop_mess}\n", machine.get_name()));
        }
    }

    /// C++ `Processchunks::checkChunk`: See if a process can be run by the
    /// current machine.  Return false when need to break out of the loop.
    #[allow(clippy::too_many_arguments)]
    pub fn check_chunk(
        &mut self,
        run_flag: &mut i32,
        no_chunks: &mut bool,
        undone_index: &mut i32,
        found_chunks: &mut bool,
        chunk_ok: &mut bool,
        machine: &mut MachineHandler,
        job_index: i32,
        chunk_err_tot: i32,
    ) -> bool {
        *run_flag = self.get_com_file_jobs().get_flag(job_index as usize);
        if self.is_verbose(&self.decorated_class_name.clone(), "checkChunk", 2) {
            self.write_out(&format!(
                "{}:checkChunk:jobIndex:{job_index},runFlag:{}\n",
                self.decorated_class_name, *run_flag
            ));
        }
        //But if the next com is a sync, record number and break loop
        if *run_flag == CHUNK_SYNC && self.syncing == 0 {
            self.next_sync_index = job_index;
            if !*found_chunks {
                *no_chunks = true;
            }
            return false;
        }
        if *undone_index == -1 && *run_flag != CHUNK_DONE && *run_flag != CHUNK_TO_SKIP {
            *undone_index = job_index;
        }
        //If any chunks found set that flag
        if *run_flag == CHUNK_SYNC || *run_flag == CHUNK_NOT_DONE {
            *found_chunks = true;
            //Skip a chunk if it has errored, if this machine has given chunk
            //error, and not all machines have done so
            // Change from script: chunkErrTot is based on number of machines not # of cpus
            *chunk_ok = true;
            if self
                .get_com_file_jobs()
                .get_num_chunk_err(job_index as usize)
                > 0
                && machine.is_chunk_erred()
                && chunk_err_tot < self.machine_list_size
            {
                *chunk_ok = false;
                if self.syncing != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// C++ `Processchunks::runProcess`: Build the .py file and run the process.
    pub fn run_process(
        &mut self,
        machine: &mut MachineHandler,
        process: &mut ProcessHandler,
        job_index: i32,
        cpu_index: i32,
    ) -> i32 {
        let max_trials = 5;

        //Lock the pipe counter until the process either starts or fails to start.
        //Since there is a delay between
        process.set_job(job_index);
        process.reset_pausing();
        process.set_flag(CHUNK_ASSIGNED);
        process.backup_log();
        process.remove_process_files();

        // Do multiple trials since this is a serious problem.  If it still fails, skip the
        // chunk if it a problem running the process, or give up if there is error in vmstopy
        // Overloaded systems can apparently give an error on running (err = 2) not just
        // timeouts
        let mut err = 0;
        for _trial in 0..max_trials {
            err = self.make_py_file(process, machine, cpu_index);
            if err == 0 {
                break;
            }
            b3d_milli_sleep(24000);
        }
        if err != 0 {
            if err == 1 {
                process.set_flag(CHUNK_NOT_DONE);
                process.invalidate_job();
                return 1;
            }
            self.write_out("Giving up due to failure to convert command file to Python\n");
            self.cleanup_and_exit(2);
            return 2;
        }

        let time_buf = unsafe {
            let cur_time = libc::time(std::ptr::null_mut());
            let mut tms: libc::tm = std::mem::zeroed();
            libc::localtime_r(&cur_time, &mut tms);
            c_format(
                " at %02d:%02d:%02d ...",
                &[
                    CArg::Int(tms.tm_hour as i64),
                    CArg::Int(tms.tm_min as i64),
                    CArg::Int(tms.tm_sec as i64),
                ],
            )
        };
        self.write_out(&format!(
            "Running {} on {}{time_buf}     [PRC1]\n",
            process.get_com_file_name(),
            machine.get_name()
        ));
        //If running a sync, set the syncing flag to 2
        if self.syncing != 0 {
            self.syncing = 2;
        }
        process.run_process(machine);
        0
    }

    /// C++ `Processchunks::extractVersion`: Extracts the first two numbers of
    /// a numeric version.  Multiples the first number by 100 and adds it to
    /// the second number.  Places the result in mVersion.
    pub fn extract_version(&mut self, version_string: &str) -> i32 {
        let mut iversion = -1;
        if self.is_verbose(&self.decorated_class_name.clone(), "extractVersion", 1) {
            self.write_out(&format!("ssh sshOutput:{version_string}\n"));
        }
        // QRegExp "[0-9]+\.[0-9]+": the first match in the string.
        let bytes = version_string.as_bytes();
        let mut found: Option<(usize, usize)> = None;
        let mut i = 0;
        while i < bytes.len() && found.is_none() {
            if bytes[i].is_ascii_digit() {
                let mut j = i;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j < bytes.len()
                    && bytes[j] == b'.'
                    && j + 1 < bytes.len()
                    && bytes[j + 1].is_ascii_digit()
                {
                    let mut k = j + 1;
                    while k < bytes.len() && bytes[k].is_ascii_digit() {
                        k += 1;
                    }
                    found = Some((i, k - i));
                } else {
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        if let Some((i, len)) = found {
            let version = &version_string[i..i + len];
            if !version.is_empty() {
                let array: Vec<&str> = version.split('.').filter(|part| !part.is_empty()).collect();
                if !array.is_empty() {
                    match array[0].parse::<i64>() {
                        Ok(value) => iversion = (value * 100) as i32,
                        Err(_) => {
                            iversion = -1;
                            return iversion;
                        }
                    }
                    if array.len() > 1 {
                        match array[1].parse::<i64>() {
                            Ok(value) => iversion += value as i32,
                            Err(_) => iversion = -1,
                        }
                    }
                }
            }
        }
        if self.is_verbose(&self.decorated_class_name.clone(), "extractVersion", 1) {
            self.write_out(&format!("ssh version:{iversion}\n"));
        }
        iversion
    }

    /// C++ `Processchunks::buildFilters`.
    pub fn build_filters(&self, root_file: &str, reg: &str, sync: &str, filters: &mut Vec<String>) {
        let mut filter1 = root_file.to_owned();
        filter1.push_str(reg);
        filter1.push_str(&self.com_extension);
        filters.push(filter1);
        let mut filter2 = root_file.to_owned();
        filter2.push_str(sync);
        filter2.push_str(&self.com_extension);
        filters.push(filter2);
    }

    /// C++ `Processchunks::cleanupList`.
    pub fn cleanup_list(&self, remove: &str, list: &mut Vec<String>) {
        //Remove files that don't have digits after rootname-
        let mut reg_exp = self.root_name().to_owned();
        reg_exp.push_str(remove);
        reg_exp.push_str(&self.com_extension);
        // `QStringList::indexOf(QRegExp)` matches anywhere in the string.
        let Ok(reg_exp) = regex::Regex::new(&reg_exp) else {
            return;
        };
        while let Some(i) = list.iter().position(|entry| reg_exp.is_match(entry)) {
            list.remove(i);
        }
    }

    /// C++ `Processchunks::runGenericProcess`: Runs process, outputs first
    /// numLinesToPrint lines, and returns the exit code.  If numLinesToPrint to
    /// is zero, no lines with be printed.  Places stdout into the output
    /// parameter.
    pub fn run_generic_process(
        &mut self,
        output: &mut Vec<u8>,
        command: &str,
        params: &[String],
        num_lines_to_print: i32,
        wait_msec: i32,
    ) -> i32 {
        let Ok(mut process) = Command::new(command)
            .args(params)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        else {
            return 1;
        };
        // `waitForFinished(waitMsec)`: poll until exit or timeout.
        let deadline = Instant::now() + Duration::from_millis(wait_msec.max(0) as u64);
        let mut status = None;
        loop {
            if let Ok(Some(exit)) = process.try_wait() {
                status = Some(exit);
                break;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if let Some(status) = status {
            output.clear();
            if let Some(mut stdout) = process.stdout.take() {
                let _ = stdout.read_to_end(output);
            }
            //Output first lines up to numLinesToPrint
            let mut start_index: i64;
            let mut end_index: i64 = -1;
            for _ in 0..num_lines_to_print {
                let temp = end_index;
                end_index = output
                    .iter()
                    .skip((end_index + 1) as usize)
                    .position(|&b| b == b'\n')
                    .map_or(-1, |p| p as i64 + end_index + 1);
                start_index = temp + 1;
                if end_index == -1 {
                    //No more lines
                    self.write_out(&String::from_utf8_lossy(output));
                    break;
                } else {
                    self.write_out(&String::from_utf8_lossy(
                        &output[start_index as usize..=end_index as usize],
                    ));
                }
            }
            return status.code().unwrap_or(0);
        }
        1
    }

    /// C++ `Processchunks::handleFileSystemBug`.
    pub fn handle_file_system_bug(&mut self, string: &str) {
        self.write_out(&format!("running ls to try to {string}\n"));
        // `mLsProcess->start("ls", mLsParamList); waitForFinished(10000)`; the
        // output is captured by the QProcess and never shown.
        if let Ok(mut ls) = Command::new("ls")
            .args(&self.ls_param_list)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            let deadline = Instant::now() + Duration::from_millis(10000);
            while !matches!(ls.try_wait(), Ok(Some(_))) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    /* CSH -> PY
    Deleted old makeCshFile 6/13/13, see earlier versions */

    /// C++ `Processchunks::makePyFile`.
    pub fn make_py_file(
        &mut self,
        process: &mut ProcessHandler,
        machine: &mut MachineHandler,
        cpu_index: i32,
    ) -> i32 {
        let mut job_machine_list = String::new();
        let mut num_error = 0;
        let max_errors = 10;
        let py_file_name = process.get_py_file();
        if py_file_name.is_empty() {
            self.write_out("Warning: no .py file name available \n");
            return 2;
        }
        let com_file_name = process.get_com_file_name();
        let mut command = "vmstopy".to_owned();
        #[cfg(windows)]
        command.push_str(".cmd");
        let mut param_list: Vec<String> = Vec::new();
        param_list.push("-c".to_owned());
        if self.queue == 0 {
            param_list.push("-n".to_owned());
            param_list.push(format!("{}", self.nice));
        }
        if process.get_gpu_number() >= 0 {
            param_list.push("-e".to_owned());
            param_list.push(format!("IMOD_USE_GPU2={}", process.get_gpu_number()));
        }
        if self.num_threads > 0 {
            param_list.push("-e".to_owned());
            param_list.push(format!("OMP_NUM_THREADS={}", self.num_threads));
        }
        let thread_limit =
            machine.get_multi_proc_info_for_cpu(cpu_index as usize, &mut job_machine_list);
        if thread_limit > 0 {
            param_list.push("-e".to_owned());
            param_list.push(format!("MULTI_PROC_THREAD_LIMIT={thread_limit}"));
            param_list.push("-e".to_owned());
            param_list.push(format!("MULTI_PROC_CPU_LIST={job_machine_list}"));
            if !self.gpu_pool_list.is_empty() {
                param_list.push("-e".to_owned());
                param_list.push(format!("MULTI_PROC_GPU_POOL={}", self.multi_proc_gpu_pool));
            } else {
                param_list.push("-e".to_owned());
                param_list.push("MULTI_PROC_GPU_POOL=None".to_owned());
            }
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_QUEUE_COMMAND=None".to_owned());
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_JOB_GPUS=None".to_owned());
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_JOB_CORES=None".to_owned());
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_GPU_QUEUE=None".to_owned());
        } else if self.queue != 0 && self.num_multi_proc_jobs != 0 {
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_GPU_POOL=None".to_owned());
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_THREAD_LIMIT=1".to_owned());
            param_list.push("-e".to_owned());
            param_list.push("MULTI_PROC_CPU_LIST=None".to_owned());
            param_list.push("-e".to_owned());
            param_list.push(format!(
                "MULTI_PROC_QUEUE_COMMAND={}",
                if self.cores_per_cluster_job != 0 {
                    "None"
                } else {
                    &self.cpu_list
                }
            ));
            param_list.push("-e".to_owned());
            param_list.push(format!(
                "MULTI_PROC_MAX_QUEUE_JOBS={}",
                self.multi_max_queue_jobs
            ));
            param_list.push("-e".to_owned());
            param_list.push(format!(
                "MULTI_PROC_GPU_QUEUE={}",
                self.secondary_queue.as_deref().unwrap_or("None")
            ));
            if self.secondary_queue.is_some() {
                param_list.push("-e".to_owned());
                param_list.push(format!(
                    "MULTI_PROC_MAX_GPU_JOBS={}",
                    self.max_on_secondary_queue
                ));
            }
            if self.cores_per_cluster_job != 0 {
                param_list.push("-e".to_owned());
                param_list.push(format!(
                    "MULTI_PROC_JOB_CORES={}",
                    self.cores_per_cluster_job
                ));
            } else {
                param_list.push("-e".to_owned());
                param_list.push("MULTI_PROC_JOB_CORES=None".to_owned());
            }
            if self.gpus_per_cluster_job != 0 {
                param_list.push("-e".to_owned());
                param_list.push(format!("MULTI_PROC_JOB_GPUS={}", self.gpus_per_cluster_job));
            } else {
                param_list.push("-e".to_owned());
                param_list.push("MULTI_PROC_JOB_GPUS=None".to_owned());
            }
        }
        param_list.push(com_file_name.clone());
        param_list.push(process.get_log_file_name());
        param_list.push(py_file_name);
        let spawned = Command::new(&command)
            .args(&param_list)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        self.vmstopy = None;
        let mut failed_to_start = false;
        match spawned {
            Ok(child) => self.vmstopy = Some(child),
            Err(_) => failed_to_start = true,
        }

        // A command not found results in waitForFinished and waitForStarted both returning
        // false, but with 0 forexitStatus and exitCode;  error() seems reliable
        let mut exit_code = 0;
        let mut crashed = false;
        loop {
            // `waitForFinished()` with its 30000 ms default.
            let mut finished = false;
            if let Some(child) = self.vmstopy.as_mut() {
                let deadline = Instant::now() + Duration::from_millis(30000);
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            finished = true;
                            exit_code = status.code().unwrap_or(0);
                            crashed = status.code().is_none();
                            break;
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                    if Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            if finished || num_error >= max_errors {
                break;
            }
            self.write_out(&format!("Warning: vmstopy conversion of {com_file_name}"));
            if failed_to_start {
                self.write_out(" failed to start: ");
            } else if crashed {
                self.write_out(" crashed after starting: ");
            } else if self.vmstopy.is_some() {
                // Loop multiple times on a timeout
                self.write_out(" timed out after 30 seconds: ");
                num_error += 1;
                continue;
            } else {
                self.write_out(" failed with an unknown error: ");
            }

            // Otherwise get out of loop on any other error
            let mut err = Vec::new();
            if let Some(child) = self.vmstopy.as_mut() {
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_end(&mut err);
                }
            }
            let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
            self.write_out(&format!("{}\n", String::from_utf8_lossy(&err[..end])));
            num_error = max_errors + 1;
        }

        // Close after multiple timeouts
        if num_error >= max_errors {
            if num_error == max_errors {
                // `mVmstopy->close()`: kill the process and wait for it.
                if let Some(mut child) = self.vmstopy.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
            return 1;
        }

        // Check the exit code
        if exit_code > 0 {
            let mut out = Vec::new();
            if let Some(child) = self.vmstopy.as_mut() {
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_end(&mut out);
                }
            }
            let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
            self.write_out(&format!(
                "Warning: vmstopy conversion of {com_file_name} exited with error code {exit_code} {}\n",
                String::from_utf8_lossy(&out[..end])
            ));
            return 2;
        }
        0
    }

    /// C++ `Processchunks::divideMachinesForJobs`: When doing multi-processor
    /// jobs, divide up the CPUs for the given # of jobs, keeping as many in
    /// each group as possible.
    pub fn divide_machines_for_jobs(&mut self) -> i32 {
        let mut start_ind = 0;
        let mut end_ind = 0;
        let mut tot_num_cpus = 0;
        let size = self.machine_list_size.max(0) as usize;
        let mut temp_num_cpus: Vec<i32> = vec![0; size];
        let mut mach_cpu_lists: Vec<Vec<String>> = vec![Vec::new(); size];
        let mut need_cpus: Vec<i32> = Vec::new();
        let mut thread_limits: Vec<Vec<i32>> = vec![Vec::new(); size];
        let mut job_cpu_lists: Vec<String> = Vec::new();
        let mut new_gpu_pool: Vec<String> = Vec::new();
        let mut job_mach_inds: Vec<usize> = Vec::new();

        // Initialize vectors and count up the cpus
        for mach in 0..size {
            if !self.machine_list[mach].is_dropped()
                && self.machine_list[mach].get_failure_count() < self.drop_crit
            {
                let full_num = self.machine_list[mach].get_full_num_cpus();
                tot_num_cpus += full_num;
                temp_num_cpus[mach] = full_num;
            }
        }

        // Limit # of jobs, bail out if only 1
        self.num_multi_proc_jobs = self.num_multi_proc_jobs.min(tot_num_cpus);
        if self.num_multi_proc_jobs < 2 {
            return 1;
        }

        // Get # of CPUs to give to each job
        for job in 0..self.num_multi_proc_jobs {
            crate::imod::libcfshr::b3dutil::balanced_group_limits(
                tot_num_cpus,
                self.num_multi_proc_jobs,
                job,
                &mut start_ind,
                &mut end_ind,
            );
            need_cpus.push(end_ind + 1 - start_ind);
            job_cpu_lists.push(String::new());
        }

        // Make assignments in rounds, on each round assign to each job from machine with most
        // remaining cores up to # needed
        let mut tot_left = tot_num_cpus;
        while tot_left > 0 {
            for job in 0..self.num_multi_proc_jobs as usize {
                if need_cpus[job] == 0 {
                    continue;
                }
                let mut best_mach: i32 = -1;
                for mach in 0..size {
                    if !self.machine_list[mach].is_dropped()
                        && self.machine_list[mach].get_failure_count() < self.drop_crit
                        && ((best_mach < 0 && temp_num_cpus[mach] != 0)
                            || (best_mach >= 0
                                && temp_num_cpus[best_mach as usize] < need_cpus[job]
                                && temp_num_cpus[mach] > temp_num_cpus[best_mach as usize]))
                    {
                        best_mach = mach as i32;
                    }
                }
                let best_mach = best_mach as usize;
                let num_assign = need_cpus[job].min(temp_num_cpus[best_mach]);
                // Add to CPU list, with comma after first time, and set the local thread
                // limit with the number assigned the first time
                if job_cpu_lists[job].is_empty() {
                    thread_limits[best_mach].push(num_assign);
                    job_mach_inds.push(best_mach);
                } else {
                    job_cpu_lists[job].push(',');
                }
                job_cpu_lists[job].push_str(&format!(
                    "{}:{}",
                    self.machine_list[best_mach].get_name(),
                    num_assign
                ));
                need_cpus[job] -= num_assign;
                temp_num_cpus[best_mach] -= num_assign;
                tot_left -= num_assign;
            }
        }

        // Loop through the jobs adding information to their machine lists
        for job in 0..self.num_multi_proc_jobs as usize {
            let mach = job_mach_inds[job];
            mach_cpu_lists[mach].push(job_cpu_lists[job].clone());
        }

        // Pass the lists to every machine, zeroing out some.  They will adjust numCpus
        for mach in 0..size {
            self.machine_list[mach]
                .set_multi_proc_job_lists(&mach_cpu_lists[mach], &thread_limits[mach]);
        }

        // Process the initial GPU pool list against the current set of machines
        for gpu in 0..self.gpu_pool_list.len() {
            let mut found = false;
            // Look a machine that is not dropped
            for mach in 0..size {
                if !self.machine_list[mach].is_dropped()
                    && self.machine_list[mach].get_failure_count() < self.drop_crit
                    && self.gpu_pool_list[gpu]
                        .to_lowercase()
                        .find(&self.machine_list[mach].get_name().to_lowercase())
                        == Some(0)
                {
                    found = true;
                }
            }
            // Look in current list of GPU only machines too
            if !found {
                for mach in 0..self.gpu_only_machines.len() {
                    if self.gpu_pool_list[gpu]
                        .to_lowercase()
                        .find(&self.gpu_only_machines[mach].to_lowercase())
                        == Some(0)
                    {
                        found = true;
                    }
                }
            }
            // If it is found, add it to new current pool
            if found {
                new_gpu_pool.push(self.gpu_pool_list[gpu].clone());
            }
        }

        // Error if there are no more GPUs, or make new string for envornment variable
        if !self.gpu_pool_list.is_empty() && new_gpu_pool.is_empty() {
            return 2;
        }
        self.multi_proc_gpu_pool = new_gpu_pool.join(",");
        0
    }

    /// C++ `Processchunks::probeOtherMultiProcMachines`: When doing
    /// multi-processor jobs, make sure each machine not being used directly
    /// (run on here) can be ssh'd to and cd to the directory.
    pub fn probe_other_multi_proc_machines(&mut self, do_it_now: bool) {
        let mut num_check = 1;
        let mut mach_limit = self.machine_list_size.max(0) as usize;
        let mut gpu_failed: Vec<i32> = Vec::new();
        if self.num_multi_proc_jobs <= 0 || self.queue != 0 {
            return;
        }

        // Test at set interval or now if argument says to
        let now = SystemTime::now();
        if !do_it_now
            && now
                .duration_since(self.last_other_probe_time)
                .map_or(0, |d| d.as_secs() as i64)
                < OTHER_PROBE_INTERVAL as i64
        {
            return;
        }
        self.last_other_probe_time = now;
        if !self.gpu_only_machines.is_empty() {
            num_check = 2;
            gpu_failed = vec![0; self.gpu_only_machines.len()];
        }

        // Check on the CPUs then on the GPU-only machines
        for check_gpu in 0..num_check {
            for mach in 0..mach_limit {
                let mach_name = if check_gpu != 0 {
                    self.gpu_only_machines[mach].clone()
                } else {
                    self.machine_list[mach].get_name().to_owned()
                };
                if (check_gpu != 0
                    || (!self.machine_list[mach].is_dropped()
                        && self.machine_list[mach].get_failure_count() < self.drop_crit
                        && self.machine_list[mach].get_num_cpus() == 0))
                    && !self.name_is_local_host(&mach_name)
                {
                    // A machine that needs checking: not dropped, has 0 cpu's officially, or GPU only
                    let mut param_list: Vec<String> = vec!["-x".to_owned()];
                    for opt in &self.ssh_opts {
                        param_list.push(opt.clone());
                    }
                    param_list.push(mach_name.clone());
                    param_list.push("bash".to_owned());
                    param_list.push("--login".to_owned());
                    param_list.push("-c".to_owned());
                    param_list.push(format!("\"cd {}\"", self.escaped_remote_dir_path));
                    let mut failed = false;

                    // Reuse this process for the test
                    self.vmstopy = Command::new("ssh")
                        .args(&param_list)
                        .stdin(Stdio::null())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .ok();
                    let mut stderr_bytes = Vec::new();
                    match self.vmstopy.as_mut() {
                        None => failed = true,
                        Some(child) => {
                            let deadline = Instant::now() + Duration::from_millis(30000);
                            let mut status = None;
                            loop {
                                if let Ok(Some(exit)) = child.try_wait() {
                                    status = Some(exit);
                                    break;
                                }
                                if Instant::now() >= deadline {
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(10));
                            }
                            match status {
                                None => {
                                    // `error() == QProcess::Timedout` -> `close()`
                                    let _ = child.kill();
                                    let _ = child.wait();
                                    failed = true;
                                }
                                Some(exit) => {
                                    if exit.code().unwrap_or(0) > 0 {
                                        failed = true;
                                    }
                                }
                            }
                            if let Some(mut stderr) = child.stderr.take() {
                                let _ = stderr.read_to_end(&mut stderr_bytes);
                            }
                        }
                    }

                    // For either kind of failure, issue dropping message and set flag to hold
                    // running further jobs so allocations can be reset
                    if failed {
                        if check_gpu == 0 {
                            self.write_out(&format!(
                                "Dropping {mach_name} due to failure in periodic probe:\n"
                            ));
                        } else {
                            self.write_out(&format!(
                                "Removing {mach_name} from GPU pool due to failure in periodic probe:\n"
                            ));
                        }
                        let end = stderr_bytes
                            .iter()
                            .position(|&b| b == 0)
                            .unwrap_or(stderr_bytes.len());
                        self.write_out(&format!(
                            "{}\n",
                            String::from_utf8_lossy(&stderr_bytes[..end])
                        ));
                        if check_gpu != 0 {
                            gpu_failed[mach] = 1;
                        } else {
                            let drop_crit = self.drop_crit;
                            self.machine_list[mach].set_failure_count(drop_crit);
                            self.machine_list[mach].set_internal_dropped();
                        }
                        self.hold_for_multi_proc_drop = true;
                    }
                }
            }
            mach_limit = self.gpu_only_machines.len();
        }

        if num_check == 2 {
            for mach in (0..self.gpu_only_machines.len()).rev() {
                if gpu_failed[mach] != 0 {
                    self.gpu_only_machines.remove(mach);
                }
            }
        }
    }

    /// C++ `Processchunks::isVerbose` with `print = true`, the inline header
    /// overload used everywhere but from `isVerbose` itself.
    pub fn is_verbose(
        &mut self,
        verbose_class: &str,
        verbose_function: &str,
        verbosity: i32,
    ) -> bool {
        self.is_verbose_print(verbose_class, verbose_function, verbosity, true)
    }

    /// C++ private `Processchunks::isVerbose(verboseClass, verboseFunction,
    /// verbosity, print)`: Returns true if its parameters match the verbose
    /// member variables.  If print is true, will print this function's verbose
    /// message only if class and function match (uses the verbosity level from
    /// the calling function).
    pub fn is_verbose_print(
        &mut self,
        verbose_class: &str,
        verbose_function: &str,
        verbosity: i32,
        print: bool,
    ) -> bool {
        if self.verbose == 0 {
            return false;
        }
        if verbosity > self.verbose {
            return false;
        }
        if self.verbose_class.is_empty() {
            return true;
        }
        if !self.verbose_function_list.is_empty()
            && print
            && self.is_verbose_print(&self.decorated_class_name.clone(), "isVerbose", 1, false)
        {
            self.write_out(&format!("{verbose_class},{verbose_function},{verbosity}\n"));
        }
        if !verbose_class
            .to_lowercase()
            .ends_with(&self.verbose_class.to_lowercase())
        {
            return false;
        }
        if self.verbose_function_list.is_empty() {
            return true;
        }
        for entry in &self.verbose_function_list {
            if verbose_function
                .to_lowercase()
                .ends_with(&entry.to_lowercase())
            {
                return true;
            }
        }
        false
    }

    // Inline header accessors.

    /// C++ inline `isQueue`.
    pub fn is_queue(&self) -> bool {
        self.queue != 0
    }
    /// C++ inline `getGpuMode`.
    pub fn get_gpu_mode(&self) -> bool {
        self.gpu_mode != 0
    }
    /// C++ inline `getQueueCommand`.
    pub fn get_queue_command(&self) -> &str {
        &self.queue_command
    }
    /// C++ inline `getQueueParamList`.
    pub fn get_queue_param_list(&self) -> &[String] {
        &self.queue_param_list
    }
    /// C++ inline `getSshOpts`.
    pub fn get_ssh_opts(&self) -> &[String] {
        &self.ssh_opts
    }
    /// C++ inline `getHostRoot`.
    pub fn get_host_root(&self) -> &str {
        &self.host_root
    }
    /// C++ inline `getMillisecSleep`.
    pub fn get_millisec_sleep(&self) -> i32 {
        self.millisec_sleep
    }
    /// C++ inline `getAns`.
    pub fn get_ans(&self) -> char {
        self.ans
    }
    /// C++ inline `getDropList`.
    pub fn get_drop_list(&self) -> &[String] {
        &self.drop_list
    }
    /// C++ inline `getRemoteDir`.
    pub fn get_remote_dir(&self) -> &str {
        self.remote_dir.as_deref().unwrap_or("")
    }
    /// C++ inline `getComFileJobs`.
    pub fn get_com_file_jobs(&self) -> &ComFileJobs {
        self.com_file_jobs
            .as_ref()
            .expect("Processchunks::setup_com_file_jobs must precede ProcessHandler setup")
    }
    /// The mutable half of `getComFileJobs`, which returns a pointer in C++.
    pub fn get_com_file_jobs_mut(&mut self) -> &mut ComFileJobs {
        self.com_file_jobs
            .as_mut()
            .expect("Processchunks::setup_com_file_jobs must precede ProcessHandler setup")
    }
    /// C++ inline `resourcesAvailableForKill`.
    pub fn resources_available_for_kill(&self) -> bool {
        self.max_kills > self.num_kills
    }
    /// C++ inline `incrementKills`.
    pub fn increment_kills(&mut self) {
        self.num_kills += 1
    }
    /// C++ inline `decrementKills`.
    pub fn decrement_kills(&mut self) {
        self.num_kills -= 1
    }
    /// C++ `*mOutStream << ...` with `QT_ENDL`'s flush: the stream is a
    /// `QTextStream(stdout)` over the same `FILE`, so order is program order.
    pub fn write_out(&mut self, text: &str) {
        let _ = ImodFile::Stdout.write_all(text.as_bytes());
        let _ = ImodFile::Stdout.flush();
    }
}

/// C++ `main`.
pub fn processchunks(argv: &[String]) -> i32 {
    let mut pc = Processchunks::new();
    pc.load_params(argv);
    pc.print_os_information();
    if !pc.setup() && !pc.ask_go() {
        return 0;
    }
    pc.start_loop()
}
