//! Translation of `IMOD/pysrc/etomo`.
//!
//! This launcher intentionally retains `java … etomo.EtomoDirector` as the
//! JVM/UI boundary.  It does not replace Java eTomo with a different Rust UI.
#![allow(dead_code)]

use super::imodpy::{
    bkgd_process, cygwin_path, get_err_strings, imod_nice, make_backup_file, prnstr, run_cmd,
    set_lib_path,
};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Matches `which` (`IMOD/pysrc/etomo:11`).
pub fn which(program: &str) -> Option<String> {
    let program = if cfg!(windows) || cfg!(target_os = "cygwin") {
        format!("{program}.exe")
    } else {
        program.to_owned()
    };
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let full = directory.join(&program);
        if full.exists() && full.is_file() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if full
                    .metadata()
                    .ok()
                    .is_some_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
                {
                    return Some(full.to_string_lossy().into_owned());
                }
            }
            #[cfg(not(unix))]
            return Some(full.to_string_lossy().into_owned());
        }
    }
    None
}

/// Matches `rollLogs` (`IMOD/pysrc/etomo:21`).
pub fn roll_logs() {
    let mut last_error = "etomo_err12.log".to_owned();
    for index in (0..=11).rev() {
        let this_error = if index == 0 {
            "etomo_err.log".to_owned()
        } else {
            format!("etomo_err{index}.log")
        };
        if Path::new(&this_error).exists() {
            if last_error == "etomo_err12.log" && Path::new(&last_error).exists() {
                let _ = fs::remove_file(&last_error);
            }
            if let Err(error) = fs::rename(&this_error, &last_error) {
                prnstr(
                    &format!(
                        "WARNING: an error occurred renaming {this_error} to {last_error} ({error})"
                    ),
                    "\n",
                    false,
                );
            }
        }
        last_error = this_error;
    }
}

/// Original Python top-level program (`IMOD/pysrc/etomo:1`).
pub fn etomo(arguments: &[OsString]) -> i32 {
    let Some(imod_directory) = std::env::var_os("IMOD_DIR") else {
        println!(
            "The IMOD_DIR environment variable has not been set\nSet it to point to the directory where IMOD is installed"
        );
        return 1;
    };
    let imod_directory = cygwin_path(&imod_directory.to_string_lossy());
    let mut path = OsString::from(Path::new(&imod_directory).join("bin"));
    path.push(if cfg!(windows) { ";" } else { ":" });
    path.push(std::env::var_os("PATH").unwrap_or_default());
    unsafe {
        std::env::set_var("PATH", path);
    }
    set_lib_path();
    let arguments = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let _new_stuff = arguments.iter().any(|argument| argument == "--newstuff");
    let memory_limit = std::env::var("ETOMO_MEM_LIM").unwrap_or_else(|_| "512m".to_owned());
    let thread_limit = if let Ok(value) = std::env::var("ETOMO_THREAD_LIM") {
        value
    } else {
        let mut limit = 16;
        if let Ok(Some(lines)) = run_cmd("imodqtassist -t", None, None, Some("stdout"), &[]) {
            for line in lines {
                if line.contains("thread count") {
                    let tokens = line.split_whitespace().collect::<Vec<_>>();
                    if let Some(index) = tokens.iter().position(|token| *token == "=") {
                        limit = limit.min(
                            tokens
                                .get(index + 1)
                                .and_then(|value| value.parse().ok())
                                .unwrap_or(limit),
                        );
                    }
                }
            }
        }
        limit.to_string()
    };
    if let Some(java_directory) = std::env::var_os("IMOD_JAVADIR") {
        let mut path =
            OsString::from(Path::new(&cygwin_path(&java_directory.to_string_lossy())).join("bin"));
        path.push(if cfg!(windows) { ";" } else { ":" });
        path.push(std::env::var_os("PATH").unwrap_or_default());
        unsafe {
            std::env::set_var("PATH", path);
        }
    }
    let version_lines = match run_cmd("java -version", None, None, Some("stdout"), &[]) {
        Ok(Some(lines)) => lines,
        _ => {
            prnstr(
                "ERROR: There is no java runtime in the current search path.  A Java runtime environment needs to be installed and the command search path may need to be defined or IMOD_JAVADIR set to locate the java command.",
                "\n",
                false,
            );
            return 1;
        }
    };
    let mut major = 0i32;
    let mut build = 0i32;
    for line in &version_lines {
        if line.contains("GNU") {
            prnstr(
                "ERROR: Etomo will not work with GNU java.  You should install an OpenJDK version of the Java runtime environment and put it on your command search path",
                "\n",
                false,
            );
            return 1;
        }
        if line.contains("version") && line.contains('"') {
            if let Some(version) = line.split('"').nth(1) {
                let values = version
                    .replace('_', ".")
                    .split('.')
                    .map(str::parse::<i32>)
                    .collect::<Result<Vec<_>, _>>();
                if let Ok(values) = values {
                    if values.len() > 2 && values[0] > 1 {
                        major = values[0];
                        build = *values.last().unwrap_or(&0);
                    } else if values.len() > 2 {
                        major = values[1];
                        build = *values.last().unwrap_or(&0);
                    }
                }
            }
        }
    }
    unsafe {
        std::env::set_var("LC_NUMERIC", "C");
        std::env::set_var("PIP_PRINT_ENTRIES", "1");
        std::env::remove_var("RUNCMD_VERBOSE");
    }
    let directive = arguments.iter().any(|argument| argument == "--directive");
    let foreground = arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "--fg" | "--directive" | "--grabit"));
    let help = arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help" | "--h" | "--grabit"));
    if cfg!(target_os = "linux")
        && !directive
        && !arguments.iter().any(|argument| argument == "--headless")
    {
        if let Ok(Some(lines)) = run_cmd("java -XshowSettings", None, None, Some("stdout"), &[1]) {
            if lines
                .iter()
                .any(|line| line.contains("java.awt.headless") && line.contains("true"))
            {
                prnstr(
                    "ERROR: The installed java is \"headless\"; to open the Etomo interface\n  you need to use a full installation of java that does not have\n  \"headless\" in its package name",
                    "\n",
                    false,
                );
                return 1;
            }
        }
    }
    let separator = if cfg!(windows) || cfg!(target_os = "cygwin") {
        ";"
    } else {
        ":"
    };
    let mut plugin_paths = String::new();
    for plugin in [
        Path::new(&imod_directory).join("Plugins"),
        Path::new(&imod_directory).join("imodplug").join("etomo"),
    ] {
        if plugin.exists() {
            plugin_paths.push_str(&format!("{separator}{}/*", plugin.display()));
        }
    }
    let mut jar_directory = Path::new(&imod_directory)
        .join("bin")
        .to_string_lossy()
        .into_owned();
    if let Some(index) = arguments.iter().position(|argument| argument == "--jardir") {
        if let Some(directory) = arguments.get(index + 1) {
            jar_directory = directory.to_owned();
        }
    }
    let mut command = vec!["java".to_owned(), format!("-Xmx{memory_limit}")];
    for option in ["-XX:ConcGCThreads=", "-XX:ParallelGCThreads="] {
        command.push(format!("{option}{thread_limit}"));
    }
    if major > 8 || (major == 8 && build >= 191) {
        command.push(format!("-XX:ActiveProcessorCount={thread_limit}"));
    }
    command.push("-cp".to_owned());
    command.push(format!("{jar_directory}/etomo.jar{plugin_paths}"));
    command.push("etomo.EtomoDirector".to_owned());
    let mut skip_next = false;
    for argument in arguments.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if argument == "--jardir" {
            skip_next = true;
            continue;
        }
        command.push(if argument == "-h" {
            "--help".to_owned()
        } else {
            argument.to_owned()
        });
        if argument.starts_with('-') && !argument.starts_with("--") {
            prnstr(
                &format!("WARNING: YOU ENTERED AN ARGUMENT WITH A SINGLE DASH: {argument}"),
                "\n",
                false,
            );
        }
    }
    if help {
        match run_cmd(
            &command
                .iter()
                .map(|argument| format!("\"{argument}\""))
                .collect::<Vec<_>>()
                .join(" "),
            None,
            None,
            Some("pipe"),
            &[],
        ) {
            Ok(Some(lines)) => {
                for line in lines {
                    prnstr(line.trim_end_matches(['\r', '\n']), "\n", false);
                }
            }
            _ => prnstr(
                "An error occurred running etomo for help output",
                "\n",
                false,
            ),
        }
        return 0;
    }
    let out_log = "etomo_out.log";
    let mut log_directory = std::env::var("ETOMO_LOG_DIR").unwrap_or_default();
    if log_directory.is_empty()
        && let Some(home) = std::env::var_os("HOME")
    {
        log_directory = Path::new(&home)
            .join(".etomologs")
            .to_string_lossy()
            .into_owned();
        if !Path::new(&log_directory).exists() && fs::create_dir(&log_directory).is_err() {
            prnstr(
                &format!("WARNING: Failed to create logs directory {log_directory}"),
                "\n",
                false,
            );
        }
    }
    let mut error_log = "etomo_err.log".to_owned();
    if !log_directory.is_empty() && Path::new(&log_directory).is_dir() {
        let retain = std::env::var("ETOMO_LOGS_TO_RETAIN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(31);
        let stamp = std::process::Command::new("date")
            .arg("+%b-%d-%H%M%S")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_owned();
        error_log = format!("{log_directory}/etomo_err_{stamp}.log");
        if let Ok(entries) = fs::read_dir(&log_directory) {
            let mut logs = entries
                .flatten()
                .filter_map(|entry| {
                    entry.metadata().ok().and_then(|metadata| {
                        metadata.modified().ok().map(|time| (time, entry.path()))
                    })
                })
                .filter(|(_, path)| {
                    path.file_name().is_some_and(|name| {
                        name.to_string_lossy().starts_with("etomo_")
                            && name.to_string_lossy().ends_with(".log")
                    })
                })
                .collect::<Vec<_>>();
            logs.sort_by_key(|(time, _)| *time);
            for (_, path) in logs.into_iter().rev().skip(retain) {
                if fs::remove_file(&path).is_err() {
                    prnstr(
                        &format!("WARNING: failed to remove old log {}", path.display()),
                        "\n",
                        false,
                    );
                }
            }
        }
        let mut reference = match fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("etomo_err.log")
        {
            Ok(file) => file,
            Err(_) => match fs::File::create("etomo_err.log") {
                Ok(file) => file,
                Err(error) => {
                    prnstr(
                        &format!(
                            "WARNING: An error occurred appending to the etomo_err.log ({error})"
                        ),
                        "\n",
                        false,
                    );
                    return 1;
                }
            },
        };
        use std::io::{Read, Seek, SeekFrom};
        let mut first = String::new();
        let _ = reference.read_to_string(&mut first);
        if !first.contains("Error log") {
            drop(reference);
            roll_logs();
            reference = match fs::File::create("etomo_err.log") {
                Ok(file) => file,
                Err(_) => return 1,
            };
        }
        let _ = reference.seek(SeekFrom::End(0));
        let now = std::process::Command::new("date")
            .arg("+%a %b %d %H:%M:%S %Y")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_owned();
        let _ = writeln!(reference, "Error log for {now} is in {error_log}");
    } else {
        roll_logs();
    }
    make_backup_file(out_log);
    prnstr(
        &format!("Starting Etomo with log in {error_log}"),
        "\n",
        false,
    );
    prnstr(
        "This log may contain personal information, such as your username",
        "\n",
        false,
    );
    let command = command.into_iter().map(OsString::from).collect::<Vec<_>>();
    if !foreground {
        return if bkgd_process(&command, Some(out_log), Some(&error_log), true, false).is_ok() {
            0
        } else {
            1
        };
    }
    let command_text = command
        .iter()
        .map(|argument| format!("\"{}\"", argument.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ");
    match run_cmd(&command_text, None, Some(out_log), Some(&error_log), &[]) {
        Ok(_) => 0,
        Err(_) => {
            prnstr(
                &format!("ERROR: etomo exited with an error status, check: {error_log}"),
                "\n",
                false,
            );
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::which;
    #[test]
    fn finds_a_path_command() {
        assert!(which("sh").is_some());
    }
}
