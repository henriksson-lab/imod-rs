//! Translation of `IMOD/mrc/manageshrmem.cpp`.
//!
//! The C++ keeps one static `ManageShrMem` so that the signal handler can
//! reach its file list; here that instance lives behind a `Mutex`, which the
//! handler only `try_lock`s so that a signal arriving while `main` holds the
//! lock cannot deadlock.  The image-file handles are the raw `ImodImageFile`
//! pointers that `iiNew` hands out, exactly as in the C++.

use crate::imod::libcfshr::b3dutil::number_in_list;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_addressable_memory, b3d_physical_memory, c_format_bytes, imod_prog_name,
    imod_usage_header,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_integer, pip_get_integer_array, pip_get_string,
    pip_number_of_entries, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::iimage::{ImodImageFile, ii_delete, ii_new};
use crate::imod::libiimod::iishrmem::{
    SHR_MEM_NAME_TAG, ii_shr_mem_check_size, ii_shr_mem_create, ii_shr_mem_remove,
};
use std::io::Write;
use std::sync::Mutex;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    // `PipPrintHelp` writes through Rust's stdout; the banner is on the C
    // stream, so hand it over before the help body follows it.
    let _ = ImodFile::Stdout.flush();
}

/// C++ `class ManageShrMem`: "A simple class".
pub struct ManageShrMem {
    /// C++ `ImodImageFile **mIIfiles`, NULL until allocated.
    ii_files: Option<Vec<*mut ImodImageFile>>,
    num_files: i32,
    /// C++ `int *mKeepList`, NULL until a keep list is entered.
    keep_list: Option<Vec<i32>>,
    num_keep: i32,
}

// The handles are owned by this one static instance, as in the C++.
unsafe impl Send for ManageShrMem {}

/// C++ `static ManageShrMem sManageShrMem;` - Static instance of class so it
/// is accessible from the signal handler.
static S_MANAGE_SHR_MEM: Mutex<ManageShrMem> = Mutex::new(ManageShrMem::new());

impl ManageShrMem {
    /// C++ `ManageShrMem::ManageShrMem`: Initialize.
    pub const fn new() -> Self {
        Self {
            ii_files: None,
            num_files: 0,
            keep_list: None,
            num_keep: 0,
        }
    }

    /// C++ `ManageShrMem::cleanupIIFiles`: Delete up to numFiles files
    /// (default mNumFiles).
    pub fn cleanup_ii_files(&mut self, mut num_files: i32) {
        let Some(ii_files) = self.ii_files.as_mut() else {
            return;
        };
        if num_files < 0 {
            num_files = self.num_files;
        }
        for file in 0..num_files {
            if number_in_list(file + 1, self.keep_list.as_deref(), self.num_keep, 0) == 0 {
                unsafe {
                    ii_delete(ii_files[file as usize]);
                }
                ii_files[file as usize] = std::ptr::null_mut();
            }
        }
    }

    /// C++ `ManageShrMem::memoryError`.
    fn memory_error(&self, all_non_null: bool, descrip: &str) {
        if !all_non_null {
            exit_error(&c_format_bytes("Allocating %s", &[CArg::Str(descrip)]));
        }
    }

    /// C++ `ManageShrMem::testAvailableMemory`: Test whether physical memory
    /// is sufficient.
    fn test_available_memory(&self, cumul_max: f32) {
        let phys_mem = b3d_physical_memory() as f32;
        let usable_mem = b3d_addressable_memory() as f32;
        if phys_mem > 0. && cumul_max as f64 * 1024. > phys_mem as f64 - 1.0e9 {
            exit_error(&c_format_bytes(
                "The maximum memory needed, %.1f MB, is too big for the memory available",
                &[CArg::Dbl(cumul_max as f64 / 1024.)],
            ));
        }

        if usable_mem > 0. && cumul_max as f64 * 1024. > usable_mem as f64 - 1.0e8 {
            exit_error(&c_format_bytes(
                "The maximum memory needed, %.1f MB, is too big for the adressable memory \
                 available",
                &[CArg::Dbl(cumul_max as f64 / 1024.)],
            ));
        }
    }
}

impl Default for ManageShrMem {
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `static void signalHandler(int signal_number)`.
#[cfg(not(windows))]
extern "C" fn signal_handler(_signal_number: libc::c_int) {
    //printf("Received signal: %s\n", strsignal(signal_number));
    if let Ok(mut manager) = S_MANAGE_SHR_MEM.try_lock() {
        manager.cleanup_ii_files(-1);
    }
}

/// C++ `int main` and `ManageShrMem::main`: The main operation.
pub fn manageshrmem(arguments: &[String]) -> i32 {
    const MAX_SIZES: usize = 1000;
    let mut sizes = [0_i32; MAX_SIZES];
    let mut size_vec: Vec<i32> = Vec::new();
    let progname = imod_prog_name(arguments.first().map_or("", String::as_str));
    let mut names: Vec<String> = Vec::new();
    let mut processes: Vec<String> = Vec::new();
    let mut needed_files: Vec<Vec<i32>> = Vec::new();
    let mut earliest_use: Vec<i32>;
    let mut latest_use: Vec<i32>;
    let mut num_procs = 0;
    let mut num_needed = 0;
    let mut try_sizes = 0;
    let mut ind_of_max: i32 = -1;
    let mut num_non_opt_args = 0;
    let mut num_opt_args = 0;
    let mut num_to_get;
    let mut error = 0;
    let mut cumul_mem: f32;
    let mut cumul_max: f32;

    // Fallbacks from    ../manpages/autodoc2man 2 1 manageshrmem
    let num_options = 8;
    let options: [&[u8]; 8] = [
        b"file:FileToCreate:FNM:",
        b"command:CommandToRun:CHM:",
        b"need:NeedFilesForCommand:IAM:",
        b"try:TrySizesFirst:I:",
        b"test:TestSizesInKilobytes:IA:",
        b"keep:ListOfFilesToKeep:LI:",
        b"remove:JustRemoveFiles:B:",
        b"help:Usage:B:",
    ];

    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        num_options,
        progname.as_bytes(),
        1,
        0,
        0,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );

    #[cfg(not(windows))]
    unsafe {
        let handler = signal_handler as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGFPE, handler);
        libc::signal(libc::SIGILL, handler);
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGSEGV, handler);
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGHUP, handler);
        libc::signal(libc::SIGQUIT, handler);
    }

    let mut this = S_MANAGE_SHR_MEM.lock().unwrap_or_else(|e| e.into_inner());

    num_to_get = 0;
    if pip_get_integer_array(
        b"TestSizesInKilobytes",
        &mut sizes,
        &mut num_to_get,
        MAX_SIZES as i32,
    ) == 0
    {
        cumul_max = 0.;
        for file in 0..num_to_get as usize {
            cumul_max += sizes[file] as f32;
        }
        this.test_available_memory(cumul_max);

        // Test for creating sizes
        this.ii_files = Some(vec![std::ptr::null_mut(); num_to_get.max(0) as usize]);
        this.memory_error(true, "array of ImodImageFile structures");
        cumul_mem = 0.;
        for file in 0..num_to_get as usize {
            let ii_file = ii_new();
            this.ii_files.as_mut().unwrap()[file] = ii_file;
            if ii_file.is_null() {
                this.cleanup_ii_files(file as i32);
                std::process::exit(1);
            }
            let buffer = c_format_bytes(
                "%s%d_dummy%d",
                &[
                    CArg::Str(SHR_MEM_NAME_TAG),
                    CArg::Int(sizes[file] as i64),
                    CArg::Int(file as i64),
                ],
            );
            cumul_mem += sizes[file] as f32;
            if ii_shr_mem_create(&String::from_utf8_lossy(&buffer), unsafe { &mut *ii_file }) != 0 {
                this.cleanup_ii_files(file as i32 + 1);
                exit_error(&c_format_bytes(
                    "Could not create shared memory file #%d with cumulative memory \
                     use of %.1f MB",
                    &[
                        CArg::Int(file as i64 + 1),
                        CArg::Dbl(cumul_mem as f64 / 1024.),
                    ],
                ));
            }
        }
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "All %d shared memory files totalling %.1f MB could be created\n",
            &[
                CArg::Int(num_to_get as i64),
                CArg::Dbl(cumul_mem as f64 / 1024.),
            ],
        ));
        this.cleanup_ii_files(num_to_get);
        std::process::exit(0);
    }

    // Get files
    let mut num_files = 0;
    pip_number_of_entries(b"FileToCreate", &mut num_files);
    this.num_files = num_files;
    if this.num_files == 0 {
        exit_error(b"At least one shared memory filename must be entered");
    }
    for _ind in 0..this.num_files {
        let mut filename = Vec::new();
        pip_get_string(b"FileToCreate", &mut filename);
        let filename = String::from_utf8_lossy(&filename).into_owned();
        let check_size = ii_shr_mem_check_size(&filename);
        if check_size == 0 {
            exit_error(&c_format_bytes(
                "%s is not a properly formatted name for a shared memory file",
                &[CArg::Str(&filename)],
            ));
        }
        names.push(filename);
        size_vec.push((check_size as f64 / 1024.) as i32);
    }

    // Remove files and exit
    let mut ind = 0;
    pip_get_boolean(b"JustRemoveFiles", &mut ind);
    if ind != 0 {
        #[cfg(windows)]
        exit_error(b"You cannot remove shared memory areas in Windows; they do not persist");
        for ind in 0..this.num_files as usize {
            ii_shr_mem_remove(&names[ind]);
        }
        std::process::exit(0);
    }

    // Get the processes
    pip_number_of_entries(b"CommandToRun", &mut num_procs);
    if num_procs < 2 {
        exit_error(b"At least two commands to run must be entered");
    }
    for _ind in 0..num_procs {
        let mut filename = Vec::new();
        pip_get_string(b"CommandToRun", &mut filename);
        processes.push(String::from_utf8_lossy(&filename).into_owned());
    }

    // Get the maps from processes to files
    needed_files.resize(num_procs as usize, Vec::new());
    pip_number_of_entries(b"NeedFilesForCommand", &mut num_needed);
    if num_needed == 0 && this.num_files == num_procs - 1 {
        for proc in 0..this.num_files as usize {
            needed_files[proc].push(proc as i32);
            needed_files[proc + 1].push(proc as i32);
        }
    } else if num_needed == 0 {
        exit_error(b"The list of files needed for each command must be entered");
    } else if num_needed != num_procs {
        exit_error(b"The number of lists of files needed must equal the number of commands");
    } else {
        for ind in 0..num_procs as usize {
            num_to_get = 0;
            pip_get_integer_array(
                b"NeedFilesForCommand",
                &mut sizes,
                &mut num_to_get,
                MAX_SIZES as i32,
            );
            for jnd in 0..num_to_get as usize {
                needed_files[ind].push(sizes[jnd] - 1);
                if sizes[jnd] < 0 || sizes[jnd] > this.num_files {
                    exit_error(&c_format_bytes(
                        "Index %d in needed file list %d is out of range",
                        &[CArg::Int(sizes[jnd] as i64), CArg::Int(ind as i64 + 1)],
                    ));
                }
            }
        }
    }

    // Determine when each one is created and destroyed
    earliest_use = vec![num_procs + 1; this.num_files.max(0) as usize];
    latest_use = vec![-1; this.num_files.max(0) as usize];
    for proc in 0..num_procs as usize {
        for jnd in 0..needed_files[proc].len() {
            if needed_files[proc][jnd] >= 0 {
                let file = needed_files[proc][jnd] as usize;
                // ACCUM_MIN / ACCUM_MAX
                if (proc as i32) < earliest_use[file] {
                    earliest_use[file] = proc as i32;
                }
                if (proc as i32) > latest_use[file] {
                    latest_use[file] = proc as i32;
                }
            }
        }
    }

    // Get list of ones to keep
    let mut filename = Vec::new();
    if pip_get_string(b"ListOfFilesToKeep", &mut filename) == 0 {
        let keep_list = parselist(&String::from_utf8_lossy(&filename)).ok();
        let Some(keep_list) = keep_list else {
            exit_error(b"An error occurred parsing the list of files to keep");
        };
        this.num_keep = keep_list.len() as i32;
        this.keep_list = Some(keep_list);
        for ind in 0..this.num_keep as usize {
            let value = this.keep_list.as_ref().unwrap()[ind];
            if value < 1 || value > this.num_files {
                exit_error(&c_format_bytes(
                    "Index %d in list of files to keep is out of range",
                    &[CArg::Int(value as i64)],
                ));
            }
        }

        // Set the latest use to past the end
        for ind in 0..this.num_keep as usize {
            let value = this.keep_list.as_ref().unwrap()[ind];
            latest_use[value as usize - 1] = num_procs + 1;
        }
    }

    // Make sure they were all used
    for ind in 0..this.num_files as usize {
        if latest_use[ind] < 0 {
            exit_error(&c_format_bytes(
                "Input file #%d, %s, is not in any of the input/output maps",
                &[CArg::Int(ind as i64 + 1), CArg::Str(&names[ind])],
            ));
        }
    }

    // Get max memory in use at once
    cumul_mem = 0.;
    cumul_max = 0.;
    for proc in 0..num_procs {
        for file in 0..this.num_files as usize {
            if proc == earliest_use[file] {
                cumul_mem += size_vec[file] as f32;
            }
            if proc == latest_use[file] + 1 {
                cumul_mem -= size_vec[file] as f32;
            }
        }
        if cumul_mem > cumul_max {
            cumul_max = cumul_mem;
            ind_of_max = proc;
        }
    }

    this.test_available_memory(cumul_max);

    this.ii_files = Some(vec![std::ptr::null_mut(); this.num_files.max(0) as usize]);
    this.memory_error(true, "array of ImodImageFile structures");
    pip_get_integer(b"TrySizesFirst", &mut try_sizes);
    if try_sizes != 0 {
        cumul_mem = 0.;
        for file in 0..this.num_files as usize {
            if earliest_use[file] >= ind_of_max && latest_use[file] <= ind_of_max {
                let ii_file = ii_new();
                this.ii_files.as_mut().unwrap()[file] = ii_file;
                cumul_mem += size_vec[file] as f32;
                if ii_shr_mem_create(&names[file], unsafe { &mut *ii_file }) != 0 {
                    this.cleanup_ii_files(-1);
                    exit_error(&c_format_bytes(
                        "Could not create shared memory file #%d with cumulative memory \
                         use of %.1f MB; max use would be %.1f MB",
                        &[
                            CArg::Int(file as i64 + 1),
                            CArg::Dbl(cumul_mem as f64 / 1024.),
                            CArg::Dbl(cumul_max as f64 / 1024.),
                        ],
                    ));
                }
            }
        }
        this.cleanup_ii_files(-1);

        if try_sizes < 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "Shared memory files totalling %.1f MB (the maximum) can be created\n",
                &[CArg::Dbl(cumul_max as f64 / 1024.)],
            ));
            std::process::exit(0);
        }
    }

    // Now we are ready to create files and run processes in sequence
    let mut proc = 0;
    while proc <= num_procs {
        for file in 0..this.num_files as usize {
            // Create file when it is time
            if proc == earliest_use[file] {
                let ii_file = ii_new();
                this.ii_files.as_mut().unwrap()[file] = ii_file;
                if ii_shr_mem_create(&names[file], unsafe { &mut *ii_file }) != 0 {
                    this.cleanup_ii_files(-1);
                    exit_error(&c_format_bytes(
                        "Could not create shared memory file #%d, %s",
                        &[CArg::Int(file as i64 + 1), CArg::Str(&names[file])],
                    ));
                }
            }

            // Remove file when its last use was the last proc
            if proc == latest_use[file] + 1 {
                unsafe {
                    ii_delete(this.ii_files.as_ref().unwrap()[file]);
                }
                this.ii_files.as_mut().unwrap()[file] = std::ptr::null_mut();
            }
        }

        if proc < num_procs {
            // glibc's `system()` is `execl("/bin/sh", "sh", "-c", line, NULL)`,
            // and it returns the wait status.  Release the instance while the
            // command runs so a signal can clean the files up.
            drop(this);
            use std::os::unix::process::CommandExt as _;
            use std::os::unix::process::ExitStatusExt as _;
            let mut shell = std::process::Command::new("/bin/sh");
            shell.arg0("sh").arg("-c").arg(&processes[proc as usize]);
            error = match shell.status() {
                Ok(status) => status.into_raw(),
                Err(_) => -1,
            };
            this = S_MANAGE_SHR_MEM.lock().unwrap_or_else(|e| e.into_inner());
            if error != 0 {
                break;
            }
        }
        proc += 1;
    }

    // This should only be needed on error, but do it just in case
    this.cleanup_ii_files(-1);

    if error != 0 {
        exit_error(&c_format_bytes(
            "Command #%d exited with code %d: %s",
            &[
                CArg::Int(proc as i64 + 1),
                CArg::Int(error as i64),
                CArg::Str(&processes[proc as usize]),
            ],
        ));
    }
    let _ = ImodFile::Stdout.write_all(b"All commands succeeded\n");
    std::process::exit(0);
}
