//! Translation of `IMOD/pysrc/splitbatch`: splits a batch command file for
//! `batchruntomo` into one command file per data set, for running in
//! parallel with `processchunks`.
//!
//! A Python command script with no functions; its top level is
//! [`splitbatch`].  It runs no program.

use super::imodpy::{
    INT_VALUE, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup, clean_chunk_files,
    complete_and_check_com_file, fmtstr, option_value, os_path_splitext, prnstr, py_int,
    read_text_file, write_text_file,
};
use super::pip::{
    exit_error, pip_exit_on_error, pip_get_boolean, pip_get_in_out_file, pip_get_integer,
    pip_parse_input, pip_print_help, python_uncaught,
};
use super::prochunks::get_translation_from_remote_dir;
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`splitbatch:1-223`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn splitbatch(arguments: &[OsString]) -> i32 {
    let progname = "splitbatch";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    let options: Vec<String> = [
        "comfile:CommandFile:FN:Command file for running Batchruntomo (required)",
        "maxgpu:MaxGPUsForOneJob:I:Maximum # of GPUs that one batch run would use (default 4)",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    pip_exit_on_error(0, &prefix);
    let (num_opts, _num_non_opts) = pip_parse_input(&argv, &options).unwrap_or_else(|_| {
        python_uncaught("TypeError: cannot unpack non-iterable NoneType object")
    });

    let if_help = pip_get_boolean("help", 0).unwrap_or(0);
    if num_opts == 0 || if_help != 0 {
        pip_print_help(progname, 0, 0, 0);
        return done(0);
    }

    // Get options
    let comfile = pip_get_in_out_file("CommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (comfile, rootname) = complete_and_check_com_file(&comfile);
    let (_root, com_ext) = os_path_splitext(&comfile);
    let max_gpu = pip_get_integer("MaxGPUsForOneJob", 4).unwrap_or(4);
    if max_gpu < 1 {
        exit_error("The maximum number of GPUs to use must be at least 1");
    }

    // Read command file
    let full_lines = read_text_file(&comfile, None, false, None).unwrap_or_default();

    // Collect the data-set specific lines into separate lists and everything
    // else except what will be added below into the common list
    let mut common_lines: Vec<String> = Vec::new();
    let mut directives: Vec<String> = Vec::new();
    let mut set_roots: Vec<String> = Vec::new();
    let mut current_dirs: Vec<String> = Vec::new();
    let mut deliver_dirs: Vec<String> = Vec::new();

    for line in &full_lines {
        let line = line.trim().to_owned();
        if line.starts_with("DirectiveFile") {
            directives.push(line);
        } else if line.starts_with("RootName") {
            set_roots.push(line);
        } else if line.starts_with("CurrentLocation") {
            current_dirs.push(line);
        } else if line.starts_with("DeliverToDirectory") {
            deliver_dirs.push(line);
        } else if !(line.starts_with("CPUMachineList")
            || line.starts_with("SingleOnFirstCP")
            || line.starts_with("LimitLocalThreads")
            || line.starts_with("GPUMachineList"))
        {
            common_lines.push(line);
        }
    }

    // Get translation options from remote entry
    let (com_dir_root, remote_root) =
        get_translation_from_remote_dir(&full_lines, &common_lines, &comfile);
    if !com_dir_root.is_empty() {
        common_lines.push(format!("TranslatePathsFrom {com_dir_root}"));
        common_lines.push(format!("TranslatePathsTo {remote_root}"));
    }

    let mut num_sets = set_roots.len();
    if num_sets == 0 {
        num_sets = directives.len();
    }

    // Check the validity of these lists
    for (opts, name) in [
        (&directives, "DirectiveFile"),
        (&current_dirs, "CurrentLocation"),
        (&deliver_dirs, "DeliverToDirectory"),
    ] {
        if opts.len() > 1 && opts.len() != num_sets {
            exit_error(&format!(
                "The number of entries for {name} ({}) does not match the number of data sets ({num_sets})",
                opts.len()
            ));
        }
    }

    let string_value = |option: &str| match option_value(
        &full_lines,
        option,
        STRING_VALUE,
        false,
        0,
        None,
        None,
    ) {
        Some(OptionValue::String(value)) => Some(value),
        _ => None,
    };
    let int_value = |option: &str, number_values: usize| match option_value(
        &full_lines,
        option,
        INT_VALUE,
        false,
        number_values,
        None,
        None,
    ) {
        Some(OptionValue::Integers(values)) => Some(values),
        _ => None,
    };

    // Check if queue
    let queue_command = string_value("QueueCommand");
    let queue_set = queue_command
        .as_deref()
        .is_some_and(|value| !value.is_empty());
    if queue_set && int_value("MaxJobsOnQueue", 0).is_none() {
        exit_error("Command file must have a MaxJobsOnQueue entry if it has QueueCommand");
    }

    // Get CPU list and object if none
    let cpu_list = string_value("CPUMachineList");
    let cpu_set = cpu_list.as_deref().is_some_and(|value| !value.is_empty());
    if queue_set && cpu_set {
        exit_error("CPUMachineList cannot be included with QueueCommand");
    }

    //Check validity of some other entries
    let gpu_queue = string_value("GPUQueueCommand");
    let gpu_queue_set = gpu_queue.as_deref().is_some_and(|value| !value.is_empty());
    if gpu_queue_set && int_value("MaxGPUJobsOnQueue", 0).is_none() {
        exit_error("Command file must have a MaxGPUJobsOnQueue entry if it has GPUQueueCommand");
    }

    // `numVal = 1`: the single value, or None
    let cores_per_job = int_value("CoresPerClusterJob", 1).map(|values| values[0]);
    let gpus_per_job = int_value("GPUsPerClusterJob", 1).map(|values| values[0]);
    if let Some(cores) = cores_per_job {
        if queue_set {
            exit_error("The command file has both a CoresPerClusterJob and a QueueCommand");
        }
        if cores <= 0 {
            exit_error("The command file has a non-positive value for CoresPerClusterJob");
        }
        if cpu_set {
            exit_error("CPUMachineList cannot be included with CoresPerClusterJob");
        }
    }

    if let Some(gpus) = gpus_per_job {
        if cores_per_job.unwrap_or(0) == 0 {
            exit_error("The command file has a GPUsPerClusterJob but no CoresPerClusterJob");
        }
        if gpu_queue_set {
            exit_error("The command file has both GPUsPerClusterJob and GPUQueueCommand");
        }
        if gpus <= 0 {
            exit_error("The command file has a non-positive value for GPUsPerClusterJob");
        }
    }

    if !cpu_set && !(queue_set || cores_per_job.is_some()) {
        exit_error("The command file must include a CPUMachineList entry with multiple CPUs");
    }

    // Get GPU list
    // If the list is 1 because local GPU was selected, need to see if there are other machines
    let mut gpu_list = string_value("GPUMachineList");
    if gpu_list.is_some() && (queue_set || cores_per_job.is_some()) {
        exit_error("GPUMachineList cannot be included with QueueCommand or CoresPerClusterJob");
    }
    if gpu_list.as_deref() == Some("1") {
        let mut cpu_array: Vec<String> = Vec::new();
        // `cpuList` is set here: every path without it has exited above
        for machine in cpu_list.as_deref().unwrap_or_default().split(',') {
            if !cpu_array.iter().any(|known| known == machine) {
                cpu_array.push(machine.to_owned());
            }
        }

        // IF there is more than one machine or the name is not localhost, try to find
        // this hostname in the list and use it as specified
        if cpu_array.len() > 1 || cpu_array[0] != "localhost" {
            // `platform.node()`: the kernel's node name
            let local_name = std::fs::read_to_string("/proc/sys/kernel/hostname")
                .map(|name| name.trim_end_matches('\n').to_owned())
                .unwrap_or_default();
            let local_short = local_name.split('.').next().unwrap_or_default().to_owned();
            for machine in &cpu_array {
                if machine.split('.').next().unwrap_or_default() == local_short {
                    if cpu_array.len() > 1 {
                        gpu_list = Some(machine.clone());
                    }
                    break;
                } else {
                    // ELSE ON FOR: Otherwise use the full name (the source's
                    // `else` is on the `if`, which gives the same result)
                    gpu_list = Some(local_name.clone());
                }
            }
        }
    }

    // Add common options
    common_lines.push(format!("ParallelBatchRootName {rootname}"));
    if let Some(gpus) = gpu_list.as_deref().filter(|value| !value.is_empty()) {
        common_lines.push(format!("GPUMachineList {gpus}"));
        common_lines.push(format!("MaxGPUsInParallelBatch {max_gpu}"));
    }

    // Parse the CPU list, make list of machine names and get total count
    if let Some(cpu_list) = cpu_list.as_deref().filter(|value| !value.is_empty()) {
        let mut num_cpu_tot: i64 = 0;
        let mut machines: Vec<(String, i64, i64)> = Vec::new();
        match py_int(cpu_list) {
            Some(num_cpu_by_int) => {
                num_cpu_tot = num_cpu_by_int;
                machines = vec![("localhost".to_owned(), num_cpu_tot, num_cpu_tot)];
            }
            None => {
                for machine in cpu_list.split(',') {
                    let replaced = machine.replace('#', ":");
                    let msplit: Vec<&str> = replaced.split(':').collect();
                    if msplit.len() > 2 {
                        exit_error("A machine name cannot be followed by two : or # signs");
                    }
                    let num_cpu = if msplit.len() < 2 {
                        1
                    } else {
                        match py_int(msplit[1]) {
                            Some(num_cpu) => {
                                if num_cpu < 1 {
                                    exit_error(&format!(
                                        "The value after : or # is less than 1 in {machine}"
                                    ));
                                }
                                num_cpu
                            }
                            None => exit_error(&format!(
                                "Failed to convert value after : or # to integer in {machine}"
                            )),
                        }
                    };
                    num_cpu_tot += num_cpu;
                    machines.push((msplit[0].to_owned(), num_cpu, num_cpu));
                }
            }
        }
        let _ = machines;

        if num_cpu_tot < 2 {
            exit_error("The CPUMachineList entry only contains a single CPU");
        }
    }

    clean_chunk_files(&rootname, false);

    let check_file = string_value("CheckFile");
    common_lines.push("SingleOnFirstCPU".to_owned());

    for dset in 0..num_sets {
        let mut all_com_lines = common_lines.clone();
        for opts in [&directives, &current_dirs, &deliver_dirs, &set_roots] {
            if !opts.is_empty() {
                let ind = dset.min(opts.len() - 1);
                all_com_lines.push(opts[ind].clone());
            }
        }
        let com_name = format!("{rootname}-{:03}{com_ext}", dset + 1);
        let _ = write_text_file(&com_name, &all_com_lines, false);
    }

    let _ = write_text_file(
        &format!("{rootname}-finish{com_ext}"),
        &[fmtstr(
            &format!("$b3dremove -g {{0}}-[0-9][0-9][0-9]*{com_ext}* {{0}}-finish*{com_ext}*"),
            &[rootname.clone()],
        )],
        false,
    );

    prnstr(
        &format!(
            "{} command files created with root name {rootname} to be run with:",
            num_sets + 1
        ),
        "\n",
        false,
    );
    // `fmtstr('...{}...', cpuList, ...)`: `str(None)` when there is no list
    prnstr(
        &format!(
            "   \"processchunks -M # {} {rootname}\"",
            cpu_list.as_deref().unwrap_or("None")
        ),
        "\n",
        false,
    );
    prnstr(
        " where # is the maximum # of files to run in parallel",
        "\n",
        false,
    );
    if let Some(check_file) = check_file.as_deref().filter(|value| !value.is_empty()) {
        prnstr(
            &format!("To stop all processing, use \"echo Q > {check_file}\""),
            "\n",
            false,
        );
        prnstr(
            "   echo F instead of Q to finish current sets before quitting",
            "\n",
            false,
        );
    }
    done(0)
}
