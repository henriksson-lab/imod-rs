//! Translation of `IMOD/qttools/processchunks/processchunks.{h,cpp}`.
//!
//! The C++ program owns the scheduling loop; `ComFileJobs`, `MachineHandler`,
//! and `ProcessHandler` remain separate source units just as they are in IMOD.

use super::comfilejobs::ComFileJobs;
use super::machinehandler::MachineHandler;
use super::{CHUNK_ASSIGNED, CHUNK_DONE, CHUNK_NOT_DONE, CHUNK_SYNC, CHUNK_TO_SKIP};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const SLEEP_MILLISEC: i32 = 1000;
pub const MAX_LOCAL_BY_NUM: i32 = 128;
pub const RUN_PROCESS_TIMEOUT: i32 = 60_000;
pub const CHECK_FILE_RECONNECT_RESET: i32 = 10;
pub const OTHER_PROBE_INTERVAL: i32 = 60;

/// The C++ `Processchunks` application.  Header and implementation are merged
/// by the Rust source-organization rule.
pub struct Processchunks {
    pub size_job_array: usize,
    pub machine_list_size: usize,
    pub num_machines_dropped: usize,
    pub job_limit_per_cycle: usize,
    pub com_file_jobs: Option<ComFileJobs>,
    /// C++ `mMachineList`; `Vec` supplies the ownership of the former array.
    pub machine_list: Vec<MachineHandler>,
    pub retain: bool,
    pub just_go: bool,
    pub nice: i32,
    pub millisec_sleep: i32,
    pub drop_crit: i32,
    pub queue: i32,
    pub single_file: bool,
    pub max_chunk_err: i32,
    pub verbose: i32,
    pub gpu_mode: bool,
    pub num_threads: i32,
    pub multiple_files: usize,
    pub entered_max_chunk_err: bool,
    pub num_multi_proc_jobs: i32,
    pub max_on_secondary_queue: i32,
    pub skip_probe: bool,
    pub queue_name: String,
    pub root_name: String,
    /// PIP's indexed non-option argument table, retained for `-s` and `-m`.
    pub non_option_args: Vec<String>,
    pub init_queue: Option<String>,
    pub deinit_queue: Option<String>,
    pub secondary_queue: Option<String>,
    pub wait_for_queue_init: i32,
    pub multi_max_queue_jobs: i32,
    pub cores_per_cluster_job: i32,
    pub gpus_per_cluster_job: i32,
    pub check_file: PathBuf,
    pub cpu_list: String,
    pub verbose_class: String,
    pub remote_dir: Option<String>,
    pub verbose_function_list: Vec<String>,
    pub gpu_pool_list: Vec<String>,
    pub gpu_only_machines: Vec<String>,
    pub multi_proc_gpu_pool: String,
    pub copy_log_index: usize,
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
    pub num_done: usize,
    pub last_num_done: usize,
    pub hold_crit: i32,
    pub first_undone_index: usize,
    pub next_sync_index: usize,
    pub syncing: i32,
    pub check_file_reconnect: i32,
    pub slowest_time: i32,
    pub slow_time_count: i32,
    pub num_skipped: usize,
    pub pausing: bool,
    pub any_done: bool,
    pub hold_for_multi_proc_drop: bool,
    pub ignore_pausing_errors: bool,
    pub last_other_probe_time: Instant,
    pub slow_overall_crit: f32,
    pub slow_machine_crit: f32,
    pub slow_sync_factor: f32,
    pub slow_log_timeout: i32,
    pub slow_sync_log_timeout: i32,
    pub ans: char,
    pub save_check_file_lines: Vec<String>,
    pub kill: bool,
    pub kill_counter: i32,
    pub num_kills: i32,
    pub max_kills: i32,
    pub drop_list: Vec<String>,
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
            retain: false,
            just_go: false,
            nice: 18,
            millisec_sleep: 50,
            drop_crit: -1,
            queue: 0,
            single_file: false,
            max_chunk_err: 5,
            verbose: 0,
            gpu_mode: false,
            num_threads: 1,
            multiple_files: 0,
            entered_max_chunk_err: false,
            num_multi_proc_jobs: 0,
            max_on_secondary_queue: 0,
            skip_probe: false,
            queue_name: "queue".into(),
            root_name: String::new(),
            non_option_args: vec![],
            init_queue: None,
            deinit_queue: None,
            secondary_queue: None,
            wait_for_queue_init: 30_000,
            multi_max_queue_jobs: 0,
            cores_per_cluster_job: 0,
            gpus_per_cluster_job: 0,
            check_file: PathBuf::from("processchunks.input"),
            cpu_list: String::new(),
            verbose_class: String::new(),
            remote_dir: None,
            verbose_function_list: vec![],
            gpu_pool_list: vec![],
            gpu_only_machines: vec![],
            multi_proc_gpu_pool: String::new(),
            copy_log_index: usize::MAX,
            num_cpus: 0,
            host_root: String::new(),
            queue_command: String::new(),
            decorated_class_name: "Processchunks".into(),
            escaped_remote_dir_path: String::new(),
            com_extension: ".com".into(),
            host_root2: String::new(),
            full_host_name: String::new(),
            ssh_opts: vec![
                "-o PreferredAuthentications=publickey".into(),
                "-o StrictHostKeyChecking=no".into(),
            ],
            queue_param_list: vec![],
            current_dir: env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            num_done: 0,
            last_num_done: 0,
            hold_crit: 0,
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
            last_other_probe_time: Instant::now(),
            slow_overall_crit: 12.0,
            slow_machine_crit: 4.0,
            slow_sync_factor: 0.0,
            slow_log_timeout: 300,
            slow_sync_log_timeout: 0,
            ans: ' ',
            save_check_file_lines: vec![],
            kill: false,
            kill_counter: 0,
            num_kills: 0,
            max_kills: 0,
            drop_list: vec![],
        }
    }

    /// C++ `Processchunks::printOsInformation`.
    pub fn print_os_information(&self) {
        println!(
            "\nIMPORTANT:  Ctrl-C does not work with this version of processchunks.  Use <Esc> <Enter> or the -c option (-c defaults to processchunks.input).\n"
        );
    }

    /// C++ `processchunksUsageHeader`.
    pub fn processchunks_usage_header(pname: &str) {
        println!(
            "Usage: {pname} [Options] machine_list root_name\nWill process multiple command files on multiple processors or machines\nmachine_list is a list of available machines, separated by commas.\nList machine names multiple times or followed by :n to use multiple CPUs on a machine.\nRoot_name is the base name of the command files, omitting -nnn.com"
        );
    }

    /// C++ `Processchunks::loadParams`.  The option spelling and validation is
    /// retained; PIP's generated parser is replaced by this direct parser.
    pub fn load_params(&mut self, argv: &[String]) -> Result<(), String> {
        let mut non_options = Vec::new();
        let mut index = 1;
        while index < argv.len() {
            let word = &argv[index];
            if !word.starts_with('-') || word == "-" {
                non_options.push(word.clone());
                index += 1;
                continue;
            }
            let (name, attached) = match word.split_once(':') {
                Some((a, b)) => (&a[1..], Some(b)),
                None => (&word[1..], None),
            };
            if name == "help" {
                Self::processchunks_usage_header("processchunks");
                return Err(String::new());
            }
            let takes_value = !matches!(name, "r" | "G" | "s" | "m" | "g" | "P" | "v");
            let value = if takes_value {
                if let Some(value) = attached {
                    value.to_owned()
                } else {
                    index += 1;
                    argv.get(index)
                        .cloned()
                        .ok_or_else(|| format!("Missing value for -{name}"))?
                }
            } else {
                String::new()
            };
            match name {
                "r" => self.retain = true,
                "G" => self.gpu_mode = true,
                "s" => self.single_file = true,
                "m" => self.multiple_files = 1,
                "g" => self.just_go = true,
                "P" => self.skip_probe = true,
                "v" => self.verbose = 1,
                "O" => {
                    self.num_threads = value.parse().map_err(|_| "Invalid -O value".to_owned())?
                }
                "M" => {
                    self.num_multi_proc_jobs =
                        value.parse().map_err(|_| "Invalid -M value".to_owned())?
                }
                "p" => {
                    self.gpu_pool_list = value
                        .split(',')
                        .filter(|x| !x.is_empty())
                        .map(str::to_owned)
                        .collect()
                }
                "n" => self.nice = value.parse().map_err(|_| "Invalid -n value".to_owned())?,
                "L" => {
                    self.job_limit_per_cycle =
                        value.parse().map_err(|_| "Invalid -L value".to_owned())?
                }
                "w" => self.remote_dir = Some(value),
                "d" => self.drop_crit = value.parse().map_err(|_| "Invalid -d value".to_owned())?,
                "e" => {
                    self.max_chunk_err =
                        value.parse().map_err(|_| "Invalid -e value".to_owned())?;
                    self.entered_max_chunk_err = true;
                }
                "C" => {
                    let x: Vec<f32> = value
                        .split(',')
                        .map(str::trim)
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|_| "Invalid -C value".to_owned())?;
                    if x.len() != 3 {
                        return Err("-C requires three values".into());
                    }
                    self.slow_machine_crit = x[0];
                    self.slow_overall_crit = x[0] * x[1];
                    self.slow_sync_factor = x[2];
                }
                "T" => {
                    let x: Vec<i32> = value
                        .split(',')
                        .map(str::trim)
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|_| "Invalid -T value".to_owned())?;
                    if x.len() != 2 {
                        return Err("-T requires two values".into());
                    }
                    self.slow_log_timeout = x[0];
                    self.slow_sync_log_timeout = x[1];
                }
                "c" => self.check_file = PathBuf::from(value),
                "q" => self.queue = value.parse().map_err(|_| "Invalid -q value".to_owned())?,
                "Q" => self.queue_name = value,
                "I" => self.init_queue = Some(value),
                "D" => self.deinit_queue = Some(value),
                "W" => {
                    let n: i32 = value.parse().map_err(|_| "Invalid -W value".to_owned())?;
                    if !(-1..=2_000_000).contains(&n) || n == 0 {
                        return Err("Wait time for queue initialization command must be -1 or a positive value up to 2000000".into());
                    }
                    self.wait_for_queue_init = if n < 0 { -1 } else { n * 1000 };
                }
                "JC" => {
                    self.cores_per_cluster_job =
                        value.parse().map_err(|_| "Invalid -JC value".to_owned())?
                }
                "JG" => {
                    self.gpus_per_cluster_job =
                        value.parse().map_err(|_| "Invalid -JG value".to_owned())?
                }
                "SQ" => self.secondary_queue = Some(value),
                "SN" => {
                    self.max_on_secondary_queue =
                        value.parse().map_err(|_| "Invalid -SN value".to_owned())?
                }
                "V" => {
                    let mut x: Vec<String> = value
                        .split(',')
                        .map(|x| x.trim().to_owned())
                        .filter(|x| !x.is_empty())
                        .collect();
                    if let Some(last) = x.last().and_then(|x| x.parse().ok()) {
                        self.verbose = last;
                        x.pop();
                    }
                    if let Some(class) = x.first() {
                        self.verbose_class = class.clone();
                        x.remove(0);
                    }
                    self.verbose_function_list = x;
                }
                _ => return Err(format!("Unrecognized option -{name}")),
            }
            index += 1;
        }
        if self.single_file && self.multiple_files != 0 {
            return Err("You cannot enter both -s and -m".into());
        }
        if self.num_multi_proc_jobs != 0 && self.num_multi_proc_jobs < 2 {
            return Err("Number of multiprocessor jobs must be at least 2".into());
        }
        if self.num_multi_proc_jobs != 0 && (self.gpu_mode || self.single_file) {
            return Err("You cannot enter -M with -G or -s".into());
        }
        if self.gpu_mode
            || self.single_file
            || self.multiple_files != 0
            || self.num_multi_proc_jobs != 0
        {
            self.num_threads = 0;
        }
        if self.multiple_files != 0 {
            self.slow_machine_crit = 0.;
            self.max_chunk_err = 2;
            self.multiple_files = non_options.len().saturating_sub(1);
        }
        if !self.gpu_pool_list.is_empty() && self.num_multi_proc_jobs == 0 {
            return Err("You can enter -p only when doing multiprocessor jobs".into());
        }
        if self.queue != 0 {
            self.skip_probe = true;
            self.just_go = true;
            if !self.gpu_pool_list.is_empty() {
                return Err("You cannot enter a GPU pool list with a queue command".into());
            }
            if self.num_multi_proc_jobs != 0 {
                self.multi_max_queue_jobs = self.queue;
                self.queue = self.num_multi_proc_jobs;
            }
        }
        if self.drop_crit < 1 {
            self.drop_crit = if self.queue != 0 { 10 } else { 5 };
        }
        if non_options.len() < 2 {
            return Err("Two non-option arguments are required".into());
        }
        if self.multiple_files == 0 && non_options.len() > 2 {
            return Err("More than two non-option arguments were entered; use the -m option to enter multiple command files".into());
        }
        self.cpu_list = non_options[0].clone();
        self.root_name = non_options[1].clone();
        self.non_option_args = non_options;
        if self.retain && self.single_file {
            return Err("You cannot use the retain option with a single command file".into());
        }
        Ok(())
    }

    /// C++ `Processchunks::setupSshOpts`.
    pub fn setup_ssh_opts(&mut self) {
        if let Ok(output) = Command::new("ssh").arg("-V").output() {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            );
            if self.extract_version(&text) >= 309 {
                self.ssh_opts.insert(0, "-o ConnectTimeout=5".into());
            }
        }
    }

    /// C++ `Processchunks::initMachineList`.
    pub fn init_machine_list(
        &mut self,
        machine_name_list: &mut Vec<String>,
        num_cpus_list: &mut Vec<i32>,
        gpu_list: &mut Vec<i32>,
    ) -> Result<(), String> {
        if self.queue != 0 {
            return Ok(());
        }
        if self.cpu_list.contains('#') {
            return Err(
                "The machine list must contain : instead of # with or without the -G option".into(),
            );
        }
        let mut num_cpus = 0;
        for entry in self.cpu_list.split(',').filter(|x| !x.is_empty()) {
            let pieces: Vec<&str> = entry.split(':').collect();
            let name = pieces[0].to_owned();
            let mut cores = 1;
            if pieces.len() == 1 {
                if self.gpu_mode {
                    gpu_list.push(0);
                }
            } else if pieces.len() > 1 && self.gpu_mode {
                cores = 0;
                for item in &pieces[1..] {
                    let gpu: i32 = item
                        .parse()
                        .map_err(|_| format!("Incorrect entry for GPU number in: {entry}"))?;
                    if gpu < 1 {
                        return Err(format!("Incorrect entry for GPU number in: {entry}"));
                    }
                    cores += 1;
                    gpu_list.push(gpu);
                }
            } else if pieces.len() == 2 {
                cores = pieces[1]
                    .parse()
                    .map_err(|_| format!("Incorrect entry for number of cores in: {entry}"))?;
                if !(1..=MAX_LOCAL_BY_NUM).contains(&cores) {
                    return Err(format!("Incorrect entry for number of cores in: {entry}"));
                }
            } else {
                return Err(format!(
                    "Multiple : characters in machine specification: {entry}"
                ));
            }
            if num_cpus + cores > 240 {
                eprintln!(
                    "WARNING:the number of CPUs exceeds limit (240).  CPU list will be truncated."
                );
                cores = 240 - num_cpus;
                if cores == 0 {
                    break;
                }
            }
            num_cpus += cores;
            if let Some(i) = machine_name_list
                .iter()
                .position(|old| old.eq_ignore_ascii_case(&name))
            {
                if self.gpu_mode {
                    return Err("You can enter each machine only once with the -G option".into());
                }
                num_cpus_list[i] += cores;
            } else {
                machine_name_list.push(name);
                num_cpus_list.push(cores);
            }
        }
        if machine_name_list.is_empty() {
            return Err("No machines specified".into());
        }
        if machine_name_list.len() == 1 {
            if let Ok(n) = machine_name_list[0].parse::<i32>() {
                if !(1..=MAX_LOCAL_BY_NUM).contains(&n) {
                    return Err(
                        "A number entered for the machine list must be positive and at most 128"
                            .into(),
                    );
                }
                if self.gpu_mode {
                    return Err(
                        "You cannot enter a number for the machine list with the -G option".into(),
                    );
                }
                machine_name_list[0] = "localhost".into();
                num_cpus_list[0] = n;
                num_cpus = n;
            }
        }
        self.max_kills = (1012 - 4 * num_cpus) / 6;
        Ok(())
    }

    /// C++ `Processchunks::setupHostRoot`.
    pub fn setup_host_root(&mut self) -> Result<(), String> {
        let output = Command::new("hostname")
            .output()
            .map_err(|_| "Unable to run the hostname command".to_owned())?;
        if !output.status.success() {
            return Err("Unable to run the hostname command".into());
        }
        self.full_host_name = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_lowercase();
        let parts: Vec<&str> = self.full_host_name.split('.').collect();
        self.host_root = parts.first().unwrap_or(&"").to_string();
        if parts.len() > 1 {
            self.host_root2 = format!("{}.{}", self.host_root, parts[1]);
        }
        Ok(())
    }

    /// C++ `Processchunks::setup`.
    pub fn setup(&mut self) -> Result<bool, String> {
        self.setup_ssh_opts();
        if self.remote_dir.is_none() {
            self.remote_dir = Some(self.current_dir.to_string_lossy().into_owned());
        }
        let mut machine_names = Vec::new();
        let mut cpu_counts = Vec::new();
        let mut gpus = Vec::new();
        self.init_machine_list(&mut machine_names, &mut cpu_counts, &mut gpus)?;
        if machine_names.len() == 1 && !self.entered_max_chunk_err {
            self.max_chunk_err = if self.queue != 0 { 10 } else { 2 };
        }
        self.setup_host_root()?;
        self.setup_com_file_jobs()?;
        let result = self.probe_machines(&mut machine_names);
        self.setup_machine_list(&machine_names, &cpu_counts, &gpus)?;
        if self.num_multi_proc_jobs != 0 && self.queue == 0 {
            for gpu in &self.gpu_pool_list {
                if !self.machine_list.iter().any(|machine| {
                    gpu.to_lowercase()
                        .starts_with(&machine.get_name().to_lowercase())
                }) {
                    if let Some(name) = gpu.split(':').next() {
                        self.gpu_only_machines.push(name.to_owned());
                    }
                }
            }
            let error = self.divide_machines_for_jobs();
            if error == 1 {
                return Err(
                    "There is only one processor available; cannot run multiprocessor jobs".into(),
                );
            }
            if error == 2 {
                return Err("There are no GPUs available; cannot run multiprocessor jobs".into());
            }
        }
        Ok(result)
    }

    /// C++ `Processchunks::setupMachineList`.
    pub fn setup_machine_list(
        &mut self,
        names: &[String],
        cpu_counts: &[i32],
        gpus: &[i32],
    ) -> Result<(), String> {
        self.machine_list.clear();
        if self.queue != 0 {
            self.num_cpus = self.queue;
            self.queue_param_list = self
                .cpu_list
                .split_whitespace()
                .map(str::to_owned)
                .collect();
            if self.queue_param_list.is_empty() {
                return Err("Queue command doesn't exist.".into());
            }
            self.queue_command = self.queue_param_list.remove(0);
            self.escaped_remote_dir_path = self.get_remote_dir().replace(' ', "\\ ");
            let status = self.run_generic_queue_command(
                self.init_queue.clone().as_deref(),
                self.wait_for_queue_init,
            );
            if status != 0 {
                return Err(format!(
                    "Command to initialize queue returned with error {status}"
                ));
            }
            let mut machine = MachineHandler::new();
            machine.setup(self, &self.queue_name.clone(), self.queue, gpus, 0);
            self.machine_list.push(machine);
        } else {
            let mut base = 0usize;
            for (index, name) in names.iter().enumerate() {
                let count = cpu_counts[index];
                if !name.is_empty() {
                    let mut machine = MachineHandler::new();
                    machine.setup(self, name, count, gpus, base);
                    self.num_cpus += count;
                    self.machine_list.push(machine);
                }
                base += count as usize;
            }
        }
        self.machine_list_size = self.machine_list.len();
        Ok(())
    }

    /// C++ `Processchunks::startLoop`.  Rust has no Qt event dispatcher here,
    /// so the source timer callback is called directly, followed by its timer
    /// interval.  ProcessHandler polls its Child at the same callback point
    /// where Qt delivered `finished` signals.
    pub fn start_loop(&mut self) -> i32 {
        self.num_done = 0;
        self.num_skipped = 0;
        self.first_undone_index = 0;
        for index in 0..self.size_job_array {
            let log = self.get_com_file_jobs().get_log_file_name(index);
            let done = fs::read_to_string(&log)
                .ok()
                .is_some_and(|text| text.trim_end().ends_with("CHUNK DONE"));
            if self.retain && done {
                self.get_com_file_jobs_mut().set_flag(index, CHUNK_DONE);
                self.num_done += 1;
            } else if !self.retain {
                let _ = fs::remove_file(&log);
                let _ = fs::remove_file(format!("{log}~"));
            }
        }
        self.last_num_done = self.num_done;
        self.pausing = false;
        self.syncing = 0;
        self.any_done = false;
        self.next_sync_index = self.size_job_array + 1;
        self.hold_crit = 2.max((self.machine_list_size as i32 + 1) / 2);
        self.start_timers();
        loop {
            if let Some(code) = self.timer_event() {
                return code;
            }
            std::thread::sleep(Duration::from_millis(self.millisec_sleep as u64));
        }
    }

    /// C++ `Processchunks::startTimers`.
    pub fn start_timers(&mut self) {
        self.millisec_sleep = if self.queue != 0 {
            (2 + self.num_cpus / 100) * 1000
        } else {
            SLEEP_MILLISEC
        };
    }

    /// C++ `Processchunks::timerEvent`.
    pub fn timer_event(&mut self) -> Option<i32> {
        if self.kill {
            self.kill_signal();
            return None;
        }
        if self.escape_entered() {
            self.handle_interrupt();
            return None;
        }
        if self.num_done + self.num_skipped >= self.size_job_array {
            return Some(self.cleanup_and_exit(0));
        }
        if self.read_check_file() {
            return None;
        }
        // C++ counts failures and live assignments before scheduling each cycle.
        let mut min_fail = self.drop_crit;
        let mut fail_total = 0;
        let mut assigned_total = 0;
        for machine in &mut self.machine_list {
            let failures = machine.get_failure_count();
            if failures != 0 {
                fail_total += 1;
            }
            min_fail = min_fail.min(failures);
            for cpu in 0..machine.get_num_cpus().max(0) as usize {
                if machine.get_process_handler(cpu).is_job_valid() {
                    assigned_total += 1;
                }
            }
        }
        if let Some(code) = self.exit_if_dropped(min_fail, fail_total, assigned_total) {
            return Some(code);
        }
        self.probe_other_multi_proc_machines(false);
        let mut started = 0usize;
        for machine_index in 0..self.machine_list.len() {
            let num_cpus = self.machine_list[machine_index].get_num_cpus().max(0) as usize;
            for cpu_index in 0..num_cpus {
                let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
                let process = unsafe { (*machine).get_process_handler(cpu_index) };
                if process.is_job_valid() && process.is_com_process_done() {
                    let job_index = process.get_assigned_job_index() as usize;
                    if process.is_chunk_done() {
                        process.set_flag(CHUNK_DONE);
                        process.invalidate_job();
                        unsafe { (*machine).set_failure_count(0) };
                        self.num_done += 1;
                        self.any_done = true;
                        println!(
                            "{} finished on {} in {:.2} sec",
                            process.get_com_file_name(),
                            unsafe { (*machine).get_name() },
                            process.get_elapsed_time() as f64 / 1000.
                        );
                        process.print_warnings(unsafe { (*machine).get_name() });
                    } else {
                        let mut message = String::new();
                        process.get_error_message_from_log(&mut message);
                        if !message.is_empty() {
                            println!("{message}");
                        }
                        process.increment_num_chunk_err();
                        let errors = process.get_num_chunk_err();
                        process.set_flag_not_done(self.single_file);
                        process.invalidate_job();
                        unsafe {
                            (*machine).increment_failure_count();
                            (*machine).set_chunk_erred(true);
                        }
                        if let Some(code) = self.handle_error(job_index, errors, self.syncing != 0)
                        {
                            return Some(code);
                        }
                    }
                }
                if self.pausing
                    || unsafe { (*machine).is_dropped() }
                    || process.is_job_valid()
                    || started >= self.job_limit_per_cycle
                {
                    continue;
                }
                let mut job_to_run = None;
                for job_index in 0..self.size_job_array {
                    let flag = self.get_com_file_jobs().get_flag(job_index);
                    if flag != CHUNK_NOT_DONE && flag != CHUNK_SYNC {
                        continue;
                    }
                    if flag == CHUNK_SYNC
                        && !(0..job_index).all(|prior| {
                            matches!(
                                self.get_com_file_jobs().get_flag(prior),
                                CHUNK_DONE | CHUNK_TO_SKIP
                            )
                        })
                    {
                        continue;
                    }
                    if flag == CHUNK_NOT_DONE
                        && !(0..job_index).all(|prior| {
                            self.get_com_file_jobs().get_flag(prior) != CHUNK_SYNC
                                || self.get_com_file_jobs().get_flag(prior) == CHUNK_DONE
                        })
                    {
                        continue;
                    }
                    job_to_run = Some(job_index);
                    break;
                }
                if let Some(job_index) = job_to_run {
                    let status = self.run_process(machine_index, cpu_index, job_index);
                    if status > 1 {
                        return Some(self.cleanup_and_exit(2));
                    }
                    if status == 0 {
                        started += 1;
                    }
                }
            }
        }
        if self.num_done > self.last_num_done {
            println!("{} OF {} DONE SO FAR", self.num_done, self.size_job_array);
            self.last_num_done = self.num_done;
        }
        if self.pausing
            && self.machine_list.iter_mut().all(|machine| {
                (0..machine.get_num_cpus().max(0) as usize)
                    .all(|cpu| !machine.get_process_handler(cpu).is_job_valid())
            })
        {
            return Some(self.cleanup_and_exit(2));
        }
        None
    }

    /// C++ `Processchunks::runProcess`.
    pub fn run_process(&mut self, machine_index: usize, cpu_index: usize, job_index: usize) -> i32 {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        process.set_job(job_index as i32);
        process.reset_pausing();
        process.set_flag(CHUNK_ASSIGNED);
        process.backup_log();
        process.remove_process_files();
        let mut error = 1;
        for _ in 0..5 {
            error = self.make_py_file(machine_index, cpu_index);
            if error == 0 {
                break;
            }
            std::thread::sleep(Duration::from_secs(24));
        }
        if error != 0 {
            process.set_flag_not_done(self.single_file);
            process.invalidate_job();
            return if error == 1 { 1 } else { 2 };
        }
        println!(
            "Running {} on {} ...     [PRC1]",
            process.get_com_file_name(),
            unsafe { (*machine).get_name() }
        );
        process.run_process(unsafe { &mut *machine });
        0
    }

    /// C++ `Processchunks::makePyFile`.
    pub fn make_py_file(&mut self, machine_index: usize, cpu_index: usize) -> i32 {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        let py_file = process.get_py_file();
        if py_file.is_empty() {
            println!("Warning: no .py file name available");
            return 2;
        }
        let mut params = vec!["-c".to_owned()];
        if self.queue == 0 {
            params.extend(["-n".to_owned(), self.nice.to_string()]);
        }
        if process.get_gpu_number() >= 0 {
            params.extend([
                "-e".to_owned(),
                format!("IMOD_USE_GPU2={}", process.get_gpu_number()),
            ]);
        }
        if self.num_threads > 0 {
            params.extend([
                "-e".to_owned(),
                format!("OMP_NUM_THREADS={}", self.num_threads),
            ]);
        }
        let mut job_machines = String::new();
        let thread_limit =
            unsafe { (*machine).get_multi_proc_info_for_cpu(cpu_index, &mut job_machines) };
        if thread_limit > 0 {
            params.extend([
                "-e".to_owned(),
                format!("MULTI_PROC_THREAD_LIMIT={thread_limit}"),
                "-e".to_owned(),
                format!("MULTI_PROC_CPU_LIST={job_machines}"),
                "-e".to_owned(),
                format!(
                    "MULTI_PROC_GPU_POOL={}",
                    if self.gpu_pool_list.is_empty() {
                        "None"
                    } else {
                        &self.multi_proc_gpu_pool
                    }
                ),
            ]);
        }
        params.extend([
            process.get_com_file_name(),
            process.get_log_file_name(),
            py_file,
        ]);
        match Command::new("vmstopy").args(&params).status() {
            Ok(status) if status.success() => 0,
            Ok(status) => {
                println!(
                    "Warning: vmstopy conversion exited with error code {}",
                    status.code().unwrap_or(1)
                );
                2
            }
            Err(error) => {
                println!("Warning: vmstopy conversion failed to start: {error}");
                1
            }
        }
    }

    /// C++ `Processchunks::readCheckFile`.
    pub fn read_check_file(&mut self) -> bool {
        let Ok(file) = fs::File::open(&self.check_file) else {
            return false;
        };
        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .collect();
        let start = lines
            .iter()
            .zip(&self.save_check_file_lines)
            .take_while(|(a, b)| a == b)
            .count();
        if start == 0 && !self.save_check_file_lines.is_empty() {
            self.save_check_file_lines.clear();
        }
        for line in lines.iter().skip(start) {
            self.save_check_file_lines.push(line.clone());
            let mut letters = line.trim().chars();
            self.ans = letters.next().unwrap_or(' ').to_ascii_uppercase();
            match self.ans {
                'D' if self.queue == 0 => {
                    self.drop_list = line[1..]
                        .trim()
                        .split(',')
                        .filter(|x| !x.is_empty())
                        .map(str::to_owned)
                        .collect();
                    self.kill_processes(Some(self.drop_list.clone()));
                    return true;
                }
                'P' | 'Q' => {
                    self.kill_processes(None);
                    return true;
                }
                _ => println!("BAD COMMAND IGNORED: {line}"),
            }
        }
        false
    }

    /// C++ `Processchunks::escapeEntered`.  Reading a terminal in nonblocking
    /// mode is platform-specific; check-file commands provide the same source
    /// control interface in this non-Qt implementation.
    pub fn escape_entered(&mut self) -> bool {
        false
    }

    /// C++ `Processchunks::handleInterrupt`.
    pub fn handle_interrupt(&mut self) {
        self.ans = 'P';
        self.kill_processes(None);
    }

    /// C++ `Processchunks::killProcesses`.
    pub fn kill_processes(&mut self, drops: Option<Vec<String>>) {
        if let Some(drops) = drops {
            self.drop_list = drops;
        }
        self.pausing = self.ans == 'P';
        if self.ans == 'C' || (self.ans == 'P' && self.queue == 0) {
            self.drop_list.clear();
            self.ans = ' ';
            return;
        }
        self.kill = true;
        for machine in &mut self.machine_list {
            machine.start_kill("");
        }
        self.kill_signal();
    }

    /// C++ `Processchunks::killSignal`.
    pub fn kill_signal(&mut self) {
        let mut done = true;
        for machine in &mut self.machine_list {
            machine.kill_signal();
            if !machine.is_kill_finished() {
                done = false;
            }
        }
        if !done {
            return;
        }
        self.drop_list.clear();
        self.kill = false;
        for machine in &mut self.machine_list {
            machine.reset_kill();
        }
        if self.ans == 'E' || self.ans == 'Q' {
            self.pausing = true;
            return;
        }
        if self.ans == 'D' || self.ans == 'P' {
            self.ans = ' ';
            println!("Resuming processing");
            self.start_timers();
        }
    }

    /// C++ `Processchunks::cleanupAndExit`, returned to `main` rather than
    /// calling C `exit` from a scheduler method.
    pub fn cleanup_and_exit(&mut self, exit_code: i32) -> i32 {
        if exit_code == 0 {
            let finish = format!("{}-finish{}", self.root_name, self.com_extension);
            if !self.current_dir.join(finish).exists() {
                println!("ALL DONE - nothing to reassemble");
            }
            println!("Finished reassembling");
            let _ = fs::remove_file(&self.check_file);
        }
        if exit_code != 0 {
            for machine in &mut self.machine_list {
                machine.kill_q_processes();
            }
        }
        let command = self.deinit_queue.clone();
        let status = self.run_generic_queue_command(command.as_deref(), 30_000);
        if status != 0 {
            println!("WARNING: Command to de-initialize queue returned with status {status}");
        }
        println!("exitCode:{exit_code}");
        exit_code
    }

    /// C++ `Processchunks::exitIfDropped`.
    pub fn exit_if_dropped(
        &mut self,
        min_fail: i32,
        fail_total: i32,
        assigned_total: i32,
    ) -> Option<i32> {
        if min_fail >= self.drop_crit {
            println!("ERROR: ALL MACHINES HAVE BEEN DROPPED DUE TO FAILURES");
            return Some(self.cleanup_and_exit(1));
        }
        if self.hold_for_multi_proc_drop && assigned_total == 0 {
            let result = self.divide_machines_for_jobs();
            if result != 0 {
                println!(
                    "ERROR: THERE IS NOT ENOUGH PROCESSING CAPACITY LEFT FOR MULTIPROCESSOR JOBS"
                );
                return Some(self.cleanup_and_exit(1));
            }
            self.hold_for_multi_proc_drop = false;
        }
        if self.pausing && assigned_total == 0 {
            println!(
                "All previously running chunks are done - exiting as requested\nRerun with -r to resume and retain existing results"
            );
            return Some(self.cleanup_and_exit(2));
        }
        if assigned_total == 0 && self.num_done == 0 && fail_total == self.machine_list_size as i32
        {
            println!("ERROR: NO CHUNKS HAVE WORKED AND EVERY MACHINE HAS FAILED");
            return Some(self.cleanup_and_exit(1));
        }
        None
    }

    /// C++ `Processchunks::handleError`.
    pub fn handle_error(
        &mut self,
        job_index: usize,
        chunk_errors: i32,
        syncing: bool,
    ) -> Option<i32> {
        if chunk_errors >= self.max_chunk_err && self.multiple_files != 0 {
            self.get_com_file_jobs_mut()
                .set_flag(job_index, CHUNK_TO_SKIP);
            self.num_skipped += 1;
            return None;
        }
        let limit = if syncing {
            if self.machine_list_size == 1 { 1 } else { 2 }
        } else {
            self.max_chunk_err
        };
        if chunk_errors >= limit {
            println!("ERROR: A CHUNK HAS FAILED {} times", chunk_errors);
            return Some(self.cleanup_and_exit(4));
        }
        None
    }

    /// C++ `Processchunks::processTooSlow`.
    pub fn process_too_slow(
        &self,
        elapsed: i32,
        slowest_time: i32,
        slow_time_count: i32,
        slow_crit: f32,
    ) -> bool {
        slow_crit > 0.
            && slow_time_count > 0
            && slowest_time > 0
            && elapsed as f32 > slow_crit * slowest_time as f32
    }

    /// C++ `Processchunks::checkChunk`.
    pub fn check_chunk(&mut self, job_index: usize, chunk_error_total: i32) -> (bool, bool) {
        let flag = self.get_com_file_jobs().get_flag(job_index);
        if flag == CHUNK_SYNC && self.syncing == 0 {
            self.next_sync_index = job_index;
            return (false, false);
        }
        let chunk_ok = self.get_com_file_jobs().get_num_chunk_err(job_index) == 0
            || chunk_error_total >= self.machine_list_size as i32;
        (flag == CHUNK_SYNC || flag == CHUNK_NOT_DONE, chunk_ok)
    }

    /// C++ `Processchunks::handleChunkDone`.
    pub fn handle_chunk_done(
        &mut self,
        machine_index: usize,
        cpu_index: usize,
        job_index: usize,
    ) -> bool {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        process.set_flag(CHUNK_DONE);
        process.invalidate_job();
        unsafe { (*machine).set_failure_count(0) };
        self.num_done += 1;
        self.any_done = true;
        if self.syncing != 0 {
            self.syncing = 0;
        }
        println!(
            "{} finished on {} in {:.2} sec",
            process.get_com_file_name(),
            unsafe { (*machine).get_name() },
            process.get_elapsed_time() as f64 / 1000.
        );
        process.print_warnings(unsafe { (*machine).get_name() });
        if job_index == self.copy_log_index {
            let root_log = format!("{}.log", self.root_name);
            if let Ok(log) = fs::read(process.get_log_file_name()) {
                let _ = fs::write(root_log, log);
            }
        }
        self.single_file
    }

    /// C++ `Processchunks::handleLogFileError`.
    pub fn handle_log_file_error(
        &mut self,
        machine_index: usize,
        cpu_index: usize,
        job_index: usize,
    ) -> Option<i32> {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        let mut error = String::new();
        process.get_error_message_from_log(&mut error);
        if !error.is_empty() {
            println!("{error}");
        }
        process.increment_num_chunk_err();
        let count = process.get_num_chunk_err();
        process.set_flag_not_done(self.single_file);
        process.invalidate_job();
        unsafe {
            (*machine).increment_failure_count();
            (*machine).set_chunk_erred(true);
        }
        self.handle_error(job_index, count, self.syncing != 0)
    }

    /// C++ `Processchunks::handleComProcessNotDone`.
    pub fn handle_com_process_not_done(&mut self, machine_index: usize, cpu_index: usize) -> bool {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        let elapsed = process.get_elapsed_time();
        let slow = self.process_too_slow(
            elapsed,
            unsafe { (*machine).get_slowest_time() },
            unsafe { (*machine).get_slow_time_count() },
            self.slow_machine_crit,
        );
        let stale = process.is_log_file_older_than(if self.syncing != 0 {
            self.slow_sync_log_timeout
        } else {
            self.slow_log_timeout
        });
        if slow && stale {
            process.close_process();
            return true;
        }
        false
    }

    /// C++ `Processchunks::handleDropOut`.
    pub fn handle_drop_out(&mut self, machine_index: usize, cpu_index: usize) {
        let machine = unsafe { self.machine_list.as_mut_ptr().add(machine_index) };
        let process = unsafe { (*machine).get_process_handler(cpu_index) };
        process.set_flag_not_done(self.single_file);
        process.invalidate_job();
        unsafe {
            (*machine).increment_failure_count();
        }
        if unsafe { (*machine).get_failure_count() } >= self.drop_crit {
            println!("Dropping {}", unsafe { (*machine).get_name() });
            unsafe {
                (*machine).set_internal_dropped();
            }
            self.hold_for_multi_proc_drop = self.num_multi_proc_jobs > 0 && self.queue == 0;
        }
    }

    /// C++ `Processchunks::probeOtherMultiProcMachines`.
    pub fn probe_other_multi_proc_machines(&mut self, do_it_now: bool) {
        if self.num_multi_proc_jobs <= 0
            || self.queue != 0
            || (!do_it_now
                && self.last_other_probe_time.elapsed()
                    < Duration::from_secs(OTHER_PROBE_INTERVAL as u64))
        {
            return;
        }
        self.last_other_probe_time = Instant::now();
        let names = self.gpu_only_machines.clone();
        self.gpu_only_machines.clear();
        for name in names {
            if self.name_is_local_host(&name)
                || Command::new("ssh")
                    .args(["-x"])
                    .args(&self.ssh_opts)
                    .arg(&name)
                    .arg("bash")
                    .arg("--login")
                    .arg("-c")
                    .arg(format!("cd {}", self.escaped_remote_dir_path))
                    .status()
                    .map(|status| status.success())
                    .unwrap_or(false)
            {
                self.gpu_only_machines.push(name);
            }
        }
    }

    /// C++ `Processchunks::nameIsLocalHost`.
    pub fn name_is_local_host(&self, mach_name: &str) -> bool {
        let n = mach_name.to_lowercase();
        n == "localhost" || n == self.host_root || n == self.host_root2 || n == self.full_host_name
    }

    /// C++ `Processchunks::setupComFileJobs`.
    pub fn setup_com_file_jobs(&mut self) -> Result<(), String> {
        let mut names = Vec::new();
        if self.single_file || self.multiple_files != 0 {
            let number = if self.multiple_files == 0 {
                1
            } else {
                self.multiple_files
            };
            for i in 0..number {
                let supplied = self
                    .non_option_args
                    .get(i + 1)
                    .ok_or_else(|| "Missing command file name".to_owned())?;
                let plain = PathBuf::from(supplied);
                let name = if plain.extension().is_some() {
                    if !plain.exists() {
                        return Err(format!("The command file {supplied} does not exist"));
                    }
                    plain
                } else {
                    let com = plain.with_extension("com");
                    let pcm = plain.with_extension("pcm");
                    if com.exists() && pcm.exists() && i == 0 {
                        return Err(format!(
                            "You must enter the full command file name with extension because both {} and {} exist",
                            com.display(),
                            pcm.display()
                        ));
                    }
                    if com.exists() {
                        com
                    } else if pcm.exists() {
                        pcm
                    } else {
                        return Err(format!(
                            "The command file {supplied}.com or {supplied}.pcm does not exist"
                        ));
                    }
                };
                let extension = name
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| format!(".{x}"))
                    .unwrap_or_default();
                if i == 0 {
                    self.com_extension = extension.clone();
                } else if extension != self.com_extension {
                    return Err(format!(
                        "All command files must have the same extension; {} does not match the preceding ones",
                        name.display()
                    ));
                }
                names.push(name.to_string_lossy().into_owned());
            }
            self.setup_com_file_jobs_from_names(names);
            return Ok(());
        }
        let root_path = Path::new(&self.root_name);
        let dir = root_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let file = root_path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or(&self.root_name);
        for extension in [".com", ".pcm"] {
            names.clear();
            let start = format!("{file}-start{extension}");
            if dir.join(&start).exists() {
                names.push(dir.join(start).to_string_lossy().into_owned());
            }
            let mut numerical: Vec<String> = fs::read_dir(dir)
                .map_err(|e| e.to_string())?
                .filter_map(Result::ok)
                .filter_map(|x| x.file_name().into_string().ok())
                .filter(|x| {
                    let prefix = format!("{file}-");
                    x.starts_with(&prefix) && x.ends_with(extension)
                })
                .filter(|x| {
                    let middle = &x[file.len() + 1..x.len() - extension.len()];
                    let number = middle.strip_suffix("-sync").unwrap_or(middle);
                    (3..=6).contains(&number.len()) && number.chars().all(|x| x.is_ascii_digit())
                })
                .collect();
            numerical.sort();
            names.extend(
                numerical
                    .into_iter()
                    .map(|x| dir.join(x).to_string_lossy().into_owned()),
            );
            let finish = format!("{file}-finish{extension}");
            if dir.join(&finish).exists() {
                names.push(dir.join(finish).to_string_lossy().into_owned());
            }
            if names.iter().any(|x| {
                Path::new(x)
                    .file_name()
                    .and_then(|x| x.to_str())
                    .map(|x| {
                        x.contains('-')
                            && x.split('-')
                                .nth(1)
                                .map(|y| {
                                    y.chars()
                                        .next()
                                        .map(|z| z.is_ascii_digit())
                                        .unwrap_or(false)
                                })
                                .unwrap_or(false)
                    })
                    .unwrap_or(false)
            }) {
                self.com_extension = extension.into();
                break;
            }
        }
        if names.is_empty() {
            return Err(format!(
                "There are no command files matching {}-nnn.com or {}-nnn.pcm",
                self.root_name, self.root_name
            ));
        }
        self.setup_com_file_jobs_from_names(names);
        Ok(())
    }

    /// Rust entry required only because C++ reaches this point through PIP's
    /// non-option argument table; it is the body of the final setup block.
    pub fn setup_com_file_jobs_from_names(&mut self, names: Vec<String>) {
        self.size_job_array = names.len();
        self.com_file_jobs = Some(ComFileJobs::new(
            names,
            self.single_file,
            self.com_extension.clone(),
        ));
        self.copy_log_index = (0..self.size_job_array)
            .find(|&i| self.com_file_jobs.as_ref().unwrap().get_flag(i) != CHUNK_SYNC)
            .unwrap_or(0);
    }

    /// C++ `Processchunks::probeMachines`.
    pub fn probe_machines(&mut self, machine_name_list: &mut [String]) -> bool {
        if self.skip_probe || self.just_go {
            return true;
        }
        let local = machine_name_list.len() == 1 && self.name_is_local_host(&machine_name_list[0]);
        if local {
            return true;
        }
        println!("Probing machine connections and loads...");
        for name in machine_name_list.iter_mut() {
            let status = if self.name_is_local_host(name) {
                Command::new("w").status()
            } else {
                Command::new("ssh")
                    .args(["-x"])
                    .args(&self.ssh_opts)
                    .arg(&*name)
                    .arg("hostname ; w")
                    .status()
            };
            if !status.map(|x| x.success()).unwrap_or(false) {
                println!("Dropping {name} from list because it does not respond\n");
                name.clear();
            }
        }
        false
    }

    /// C++ `Processchunks::askGo`.
    pub fn ask_go(&mut self) -> bool {
        if self.just_go {
            return true;
        }
        print!("Enter Y to proceed with the current set of machines: ");
        let _ = io::stdout().flush();
        let mut answer = String::new();
        io::stdin().read_line(&mut answer).is_ok() && matches!(answer.trim(), "Y" | "y")
    }

    /// C++ `Processchunks::extractVersion`.
    pub fn extract_version(&self, version_string: &str) -> i32 {
        let bytes = version_string.as_bytes();
        for i in 0..bytes.len() {
            if bytes[i].is_ascii_digit() {
                let rest = &version_string[i..];
                if let Some((major, tail)) = rest.split_once('.') {
                    let minor: String = tail.chars().take_while(|x| x.is_ascii_digit()).collect();
                    if !minor.is_empty() {
                        if let (Ok(a), Ok(b)) = (
                            major
                                .chars()
                                .take_while(|x| x.is_ascii_digit())
                                .collect::<String>()
                                .parse::<i32>(),
                            minor.parse::<i32>(),
                        ) {
                            return a * 100 + b;
                        }
                    }
                }
            }
        }
        -1
    }

    /// C++ `Processchunks::buildFilters`.
    pub fn build_filters(&self, root_file: &str, reg: &str, sync: &str, filters: &mut Vec<String>) {
        filters.push(format!("{root_file}{reg}{}", self.com_extension));
        filters.push(format!("{root_file}{sync}{}", self.com_extension));
    }

    /// C++ `Processchunks::cleanupList`.
    pub fn cleanup_list(&self, remove: &str, list: &mut Vec<String>) {
        let root = regex::escape(&self.root_name);
        let suffix = regex::escape(&self.com_extension);
        let expression = regex::Regex::new(&format!("{root}{remove}{suffix}$")).unwrap();
        list.retain(|x| !expression.is_match(x));
    }

    /// C++ `Processchunks::runGenericProcess`.
    pub fn run_generic_process(
        &mut self,
        command: &str,
        params: &[String],
        num_lines_to_print: usize,
        wait_msec: i32,
    ) -> i32 {
        let mut child = match Command::new(command)
            .args(params)
            .stdout(Stdio::piped())
            .spawn()
        {
            Ok(x) => x,
            Err(_) => return 1,
        };
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let text = child
                        .stdout
                        .take()
                        .and_then(|mut x| {
                            let mut b = Vec::new();
                            std::io::Read::read_to_end(&mut x, &mut b).ok()?;
                            Some(String::from_utf8_lossy(&b).into_owned())
                        })
                        .unwrap_or_default();
                    for line in text.lines().take(num_lines_to_print) {
                        println!("{line}");
                    }
                    return status.code().unwrap_or(1);
                }
                Ok(None)
                    if wait_msec < 0
                        || start.elapsed() < Duration::from_millis(wait_msec as u64) =>
                {
                    std::thread::sleep(Duration::from_millis(10))
                }
                _ => {
                    let _ = child.kill();
                    return 1;
                }
            }
        }
    }

    /// C++ `Processchunks::runGenericQueueCommand`.
    pub fn run_generic_queue_command(&mut self, command: Option<&str>, wait_msec: i32) -> i32 {
        if self.queue == 0 || command.is_none_or(str::is_empty) {
            return 0;
        }
        let mut params = self.queue_param_list.clone();
        params.extend([
            "-w".into(),
            self.escaped_remote_dir_path.clone(),
            "-a".into(),
            format!("C:{}", command.unwrap()),
        ]);
        let queue = self.queue_command.clone();
        self.run_generic_process(
            &queue,
            &params,
            if self.verbose > 0 { 100 } else { 0 },
            wait_msec,
        )
    }

    /// C++ `Processchunks::handleFileSystemBug`.
    pub fn handle_file_system_bug(&mut self, what: &str) {
        println!("running ls to try to {what}");
        let _ = Command::new("ls").status();
    }

    /// C++ `Processchunks::divideMachinesForJobs`.
    pub fn divide_machines_for_jobs(&mut self) -> i32 {
        let mut remaining: Vec<i32> = self
            .machine_list
            .iter()
            .map(|m| {
                if !m.is_dropped() && m.get_failure_count() < self.drop_crit {
                    m.get_full_num_cpus()
                } else {
                    0
                }
            })
            .collect();
        let total: i32 = remaining.iter().sum();
        self.num_multi_proc_jobs = self.num_multi_proc_jobs.min(total);
        if self.num_multi_proc_jobs < 2 {
            return 1;
        }
        let jobs = self.num_multi_proc_jobs as usize;
        let mut needed = Vec::with_capacity(jobs);
        for job in 0..jobs {
            let start = (total as i64 * job as i64 / jobs as i64) as i32;
            let end = (total as i64 * (job + 1) as i64 / jobs as i64) as i32;
            needed.push(end - start);
        }
        let mut lists = vec![Vec::<String>::new(); self.machine_list.len()];
        let mut limits = vec![Vec::<i32>::new(); self.machine_list.len()];
        let mut job_lists = vec![String::new(); jobs];
        let mut first_machine = vec![usize::MAX; jobs];
        while remaining.iter().any(|&x| x > 0) {
            for job in 0..jobs {
                if needed[job] == 0 {
                    continue;
                }
                let mut best = None;
                for machine in 0..self.machine_list.len() {
                    if remaining[machine] > 0
                        && !self.machine_list[machine].is_dropped()
                        && self.machine_list[machine].get_failure_count() < self.drop_crit
                    {
                        if best.is_none()
                            || (remaining[best.unwrap()] < needed[job]
                                && remaining[machine] > remaining[best.unwrap()])
                        {
                            best = Some(machine);
                        }
                    }
                }
                let Some(machine) = best else {
                    return 1;
                };
                let number = needed[job].min(remaining[machine]);
                if job_lists[job].is_empty() {
                    limits[machine].push(number);
                    first_machine[job] = machine;
                } else {
                    job_lists[job].push(',');
                }
                job_lists[job].push_str(&format!(
                    "{}:{number}",
                    self.machine_list[machine].get_name()
                ));
                needed[job] -= number;
                remaining[machine] -= number;
            }
        }
        for job in 0..jobs {
            if first_machine[job] != usize::MAX {
                lists[first_machine[job]].push(job_lists[job].clone());
            }
        }
        for machine in 0..self.machine_list.len() {
            self.machine_list[machine].set_multi_proc_job_lists(&lists[machine], &limits[machine]);
        }
        let mut pool = Vec::new();
        for gpu in &self.gpu_pool_list {
            if self.machine_list.iter().any(|machine| {
                !machine.is_dropped()
                    && machine.get_failure_count() < self.drop_crit
                    && gpu
                        .to_lowercase()
                        .starts_with(&machine.get_name().to_lowercase())
            }) || self
                .gpu_only_machines
                .iter()
                .any(|machine| gpu.to_lowercase().starts_with(&machine.to_lowercase()))
            {
                pool.push(gpu.clone());
            }
        }
        if !self.gpu_pool_list.is_empty() && pool.is_empty() {
            return 2;
        }
        self.multi_proc_gpu_pool = pool.join(",");
        0
    }

    /// C++ `Processchunks::isVerbose`.
    pub fn is_verbose(&self, verbose_class: &str, verbose_function: &str, verbosity: i32) -> bool {
        if self.verbose == 0 || verbosity > self.verbose {
            return false;
        }
        if self.verbose_class.is_empty() {
            return true;
        }
        if !verbose_class
            .to_lowercase()
            .ends_with(&self.verbose_class.to_lowercase())
        {
            return false;
        }
        self.verbose_function_list.is_empty()
            || self
                .verbose_function_list
                .iter()
                .any(|x| verbose_function.to_lowercase().ends_with(&x.to_lowercase()))
    }

    // Accessors mapped from inline C++ header members; MachineHandler and
    // ProcessHandler use these rather than reaching into scheduler state.
    pub fn is_queue(&self) -> bool {
        self.queue != 0
    }
    pub fn get_gpu_mode(&self) -> bool {
        self.gpu_mode
    }
    pub fn get_queue_command(&self) -> &str {
        &self.queue_command
    }
    pub fn get_queue_param_list(&self) -> &[String] {
        &self.queue_param_list
    }
    pub fn get_ssh_opts(&self) -> &[String] {
        &self.ssh_opts
    }
    pub fn get_host_root(&self) -> &str {
        &self.host_root
    }
    pub fn get_millisec_sleep(&self) -> i32 {
        self.millisec_sleep
    }
    pub fn get_ans(&self) -> char {
        self.ans
    }
    pub fn get_drop_list(&self) -> &[String] {
        &self.drop_list
    }
    pub fn get_remote_dir(&self) -> &str {
        self.remote_dir.as_deref().unwrap_or("")
    }
    pub fn get_com_file_jobs(&self) -> &ComFileJobs {
        self.com_file_jobs
            .as_ref()
            .expect("Processchunks::setup_com_file_jobs must precede ProcessHandler setup")
    }
    pub fn get_com_file_jobs_mut(&mut self) -> &mut ComFileJobs {
        self.com_file_jobs
            .as_mut()
            .expect("Processchunks::setup_com_file_jobs must precede ProcessHandler setup")
    }
    pub fn resources_available_for_kill(&self) -> bool {
        self.max_kills > self.num_kills
    }
    pub fn increment_kills(&mut self) {
        self.num_kills += 1
    }
    pub fn decrement_kills(&mut self) {
        self.num_kills -= 1
    }
    pub fn write_out(&mut self, text: &str) {
        print!("{text}");
        let _ = io::stdout().flush();
    }
}

/// C++ `main`.
pub fn processchunks(argv: &[String]) -> i32 {
    let mut app = Processchunks::new();
    if let Err(error) = app.load_params(argv) {
        if !error.is_empty() {
            eprintln!("ERROR: {error}");
            return 1;
        }
        return 0;
    }
    app.print_os_information();
    if let Err(error) = app.setup() {
        eprintln!("ERROR: {error}");
        return 1;
    }
    if !app.ask_go() {
        return 0;
    }
    app.start_loop()
}
