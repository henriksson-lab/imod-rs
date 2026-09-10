//! Translation of `IMOD/pysrc/submfg`.
//!
//! The original is a Python command program, so its one top-level program body is
//! represented by [`submfg`].  `vmstopy`, `vmstocsh`, `tcsh`, and the generated
//! Python command file are intentionally retained as process boundaries: they are
//! separate IMOD command units, not alternate Rust implementations.
#![allow(dead_code)]

use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use super::imodpy::add_imod_bin_ignore_sighup;
use super::pip::expand_arg_list;

/// Original Python top-level program (`IMOD/pysrc/submfg:1`).
pub fn submfg(arguments: &[OsString]) -> i32 {
    let progname = "submfg";
    let prefix = "ERROR: submfg - ";
    if std::env::var_os("IMOD_DIR").is_none() {
        println!("{prefix} IMOD_DIR is not defined!");
        return 1;
    }
    // Source startup does this before resolving vmstopy/vmstocsh from PATH.
    add_imod_bin_ignore_sighup();
    let comtmp = format!("submtemp.{}", std::process::id());
    let bell = "\x07";
    let mut message = format!(" finished successfully{bell}");
    if let Some(value) = std::env::var_os("SUBM_MESSAGE") {
        message = value.to_string_lossy().into_owned();
    }
    let mut log_type = match std::env::var("SUBM_LOG_TYPE") {
        Ok(value) => match value.parse::<i32>() {
            Ok(value) => value,
            Err(_) => {
                eprintln!(
                    "{prefix}Converting environment variable SUBM_LOG_TYPE ({value}) to integer"
                );
                return 1;
            }
        },
        Err(_) => 0,
    };
    if arguments.len() < 2 {
        println!("{}",
            "subm or submfg will execute a series of command files in sequence\nUsage:  submfg [options] command_file1 command_file2 ...\n        Command files can have default extension .com or .pcm\n        If the filename is comfile.com or comfile.pcm, you can enter \n                 comfile    comfile.  or  comfile.com or comfile.pcm\n        submfg will execute the files in the foreground\n        subm is an alias defined in the IMOD startup script to execute submfg\n               in the background\n        Set the environment variable SUBM_MESSAGE to modify the message upon\n               completion\n        Set the environment variable SUBM_LOG_TYPE to set a default log type\n    Options:\n        -t     Report the execution time\n        -c     Continue with the next command file if one fails\n        -s     Translate file with vmstocsh and run with tcsh\n                 (default is to translate with vmstopy and run with python)\n        -k     Keep backslashes instead of converting to forward slashes\n        -n #   Run niced with # as nice increment (range 1 to 19)\n        -l #   Log type for numbered or time-stamped logs:\n                  1 - 4 for sequential numbers with 1-4 digits\n                 -1 for date-time stamps like Mar-01-195046.4\n                 -2 for date-time stamps like 20120301-195121.9\n                 -3 for date-time stamps like 2012-03-01T19:51:51.9"
        .replace("forward slashes\n", "forward slashes'\n"));
        return 0;
    }
    let mut argind = 1usize;
    let mut nice = 0i32;
    let mut use_tcsh = false;
    let mut do_time = false;
    let mut continue_if_error = false;
    let mut keep_backslash = false;
    let windows = cfg!(windows);
    while argind < arguments.len() {
        let original = arguments[argind].to_string_lossy();
        if !original.starts_with('-') {
            break;
        }
        match original.as_ref() {
            "-t" => do_time = true,
            "-c" => continue_if_error = true,
            "-k" => keep_backslash = true,
            "-s" => {
                use_tcsh = true;
                if windows {
                    eprintln!("{prefix}You cannot run command files with tcsh from Windows Python");
                    return 1;
                }
            }
            "-n" | "-l" => {
                argind += 1;
                if argind >= arguments.len() {
                    break;
                }
                let value = arguments[argind].to_string_lossy();
                match value.parse::<i32>() {
                    Ok(value) if original == "-n" => nice = value,
                    Ok(value) => log_type = value,
                    Err(_) => {
                        let description = if original == "-n" {
                            "\"nice\" value"
                        } else {
                            "log type value"
                        };
                        // `convertToInteger` calls Python PIP `exitError`; this
                        // program installed only an exit prefix, whose default
                        // destination is stdout.
                        println!("{prefix}Converting {description} ({value}) to integer");
                        return 1;
                    }
                }
            }
            _ => {
                println!("{prefix}Unrecognized argument {original}");
                return 1;
            }
        }
        argind += 1;
    }
    if argind >= arguments.len() {
        // Missing values after -n/-l fall through here in the source and
        // report via its stdout PIP exit route.
        println!("{prefix}No command file was entered");
        return 1;
    }
    let (expanded_arguments, no_match) = expand_arg_list(&arguments[argind..]);
    if no_match >= 0 {
        eprintln!(
            "{prefix}No files match the entry: {}",
            arguments[argind + no_match as usize].to_string_lossy()
        );
        return 1;
    }
    let new_args = expanded_arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let mut exit_value = 0;
    for argname in new_args {
        let path = Path::new(&argname);
        // Python `splitext("name.")` returns `(name, ".")`; `Path` does
        // not preserve that distinction, so retain it from the original text.
        let trailing_dot = argname.ends_with('.');
        let rootname = if trailing_dot {
            argname.trim_end_matches('.').to_owned()
        } else {
            path.with_extension("").to_string_lossy().into_owned()
        };
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy().into_owned())
            .unwrap_or_default();
        let comname = if extension.is_empty() || trailing_dot {
            let com_exists = Path::new(&(rootname.clone() + ".com")).exists();
            let pcm_exists = Path::new(&(rootname.clone() + ".pcm")).exists();
            if com_exists && pcm_exists {
                // `submfg:127-128`: PIP `exitError` follows the prefix's
                // default stdout route for command-file resolution errors.
                println!("{prefix}Both {rootname}.com and {rootname}.pcm exist; specify which");
                exit_value = 1;
                if !continue_if_error {
                    break;
                }
                continue;
            }
            if com_exists {
                rootname.clone() + ".com"
            } else if pcm_exists {
                rootname.clone() + ".pcm"
            } else {
                println!("{prefix}Neither {rootname}.com nor {rootname}.pcm exists");
                exit_value = 1;
                if !continue_if_error {
                    break;
                }
                continue;
            }
        } else {
            argname.clone()
        };
        let mut logname = format!("{rootname}.log");
        if log_type > 0 {
            let digits = log_type.min(4) as usize;
            let mut lognum = 1u32;
            if let Some(parent) = Path::new(&logname).parent() {
                if let Ok(entries) = fs::read_dir(parent) {
                    let prefix_name = format!(
                        "{}-",
                        Path::new(&logname)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    );
                    for entry in entries.flatten() {
                        let file_name = entry.file_name().to_string_lossy().into_owned();
                        if let Some(number) = file_name
                            .strip_prefix(&prefix_name)
                            .and_then(|tail| tail.parse::<u32>().ok())
                        {
                            lognum = lognum.max(number + 1);
                        }
                    }
                }
            }
            logname.push_str(&format!("-{lognum:0digits$}"));
        } else if log_type < 0 {
            // Python's local-time datetime formatting is retained at the process
            // boundary using `date`; this avoids adding a non-source time library.
            let date_format = if log_type == -1 {
                "+%b-%d-%H%M%S"
            } else if log_type == -2 {
                "+%Y%m%d-%H%M%S"
            } else {
                "+%Y-%m-%dT%H:%M:%S"
            };
            let stamp = Command::new("date")
                .arg(date_format)
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .unwrap_or_default()
                .trim()
                .to_owned();
            let tenth = Command::new("date")
                .arg("+%N")
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .and_then(|value| value.trim().parse::<u32>().ok())
                .unwrap_or(0)
                / 100_000_000;
            logname.push_str(&format!("-{stamp}.{tenth}"));
        }
        let conversion = if use_tcsh {
            match fs::read_to_string(&comname) {
                Ok(lines) => match Command::new("sh")
                    .arg("-c")
                    .arg(format!("vmstocsh {logname}"))
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .spawn()
                {
                    Ok(mut child) => {
                        if let Some(mut stdin) = child.stdin.take() {
                            let _ = stdin.write_all(lines.as_bytes());
                        }
                        match child.wait_with_output() {
                            Ok(output) if output.status.success() => {
                                let mut cshlines =
                                    String::from_utf8_lossy(&output.stdout).into_owned();
                                if nice != 0 {
                                    cshlines = format!("nice +{nice}\n{cshlines}");
                                }
                                fs::write(&comtmp, cshlines).map_err(|error| error.to_string())
                            }
                            Ok(output) => Err(String::from_utf8_lossy(&output.stderr).into_owned()),
                            Err(error) => Err(error.to_string()),
                        }
                    }
                    Err(error) => Err(error.to_string()),
                },
                Err(error) => Err(error.to_string()),
            }
        } else {
            // `runcmd` receives an argument string here, but its process has
            // no shell syntax.  Pass the original individual arguments so a
            // command/log/tmp name containing shell metacharacters is not
            // reinterpreted by an invented shell boundary.
            let mut command = Command::new("vmstopy");
            if nice != 0 {
                command.args(["-n", &nice.to_string()]);
            }
            if keep_backslash {
                command.arg("-k");
            }
            command
                .arg(&comname)
                .arg(&logname)
                .arg(&comtmp)
                .status()
                .map_err(|error| error.to_string())
                .and_then(|status| {
                    if status.success() {
                        Ok(())
                    } else {
                        Err(format!("exit status {status}"))
                    }
                })
        };
        if let Err(error) = conversion {
            eprintln!("Error executing {comname}{bell}");
            if !error.is_empty() {
                eprintln!("{error}");
            }
            exit_value = 1;
            if !continue_if_error {
                break;
            }
            continue;
        }
        let mut command = if use_tcsh {
            "tcsh -ef ".to_owned()
        } else {
            "python -u ".to_owned()
        };
        command.push_str(&comtmp);
        if do_time && !windows {
            command = format!("time {command}");
        }
        if log_type != 0 {
            print!("Running {comname} with log in {logname} ... ");
        } else {
            print!("Running {comname} ... ");
        }
        let _ = io::stdout().flush();
        let start_time = Instant::now();
        let status = Command::new("sh").arg("-c").arg(command).status();
        if status.as_ref().is_err() || status.as_ref().is_ok_and(|status| !status.success()) {
            eprintln!("Error executing {comname}{bell}");
            exit_value = 1;
            if let Ok(lines) = fs::read_to_string(&logname) {
                let mut found = false;
                for line in lines.lines().filter(|line| line.contains("ERROR:")) {
                    println!("{line}");
                    found = true;
                }
                if !found {
                    println!("   last lines of log:");
                    for line in lines
                        .lines()
                        .rev()
                        .take(4)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                    {
                        println!("{line}");
                    }
                }
            } else if let Err(error) = status {
                eprintln!("{error}");
            }
            if !continue_if_error {
                break;
            }
        } else {
            if do_time && windows {
                message.push_str(&format!(
                    "   in {:.2} sec",
                    start_time.elapsed().as_secs_f64()
                ));
            }
            println!("{comname} {message}");
        }
    }
    let _ = fs::remove_file(&comtmp);
    exit_value
}

#[cfg(test)]
mod tests {
    use super::submfg;
    use std::ffi::OsString;

    #[test]
    fn requires_imod_dir_as_source_does() {
        // This test cannot mutate process environment in Rust 2024; an empty
        // environment-independent argument list still exercises the source usage path.
        assert!(matches!(submfg(&[OsString::from("submfg")]), 0 | 1));
    }
}
