//! `IMOD/Etomo/src/etomo/process/ParsePID.java`.
//!
//! Scans a running program's standard error for the line giving the process
//! ID of the shell or Python job it started, and records it.  The PID buffer
//! is the invoking object's `StringBuffer`, shared here as an
//! `Arc<Mutex<String>>`.
//!
//! `ParseBackgroundPID` overrides only `parsePIDString`, to read the first
//! line of an output file instead; that is the `out_file` field
//! (`parse_background_pid.rs` constructs it).
//!
//! The command-file runner prints `Runcom PID: <pid>` where the `vmstopy`
//! script printed `Python PID: <pid>` (`comrun.rs`, `runcom -P`), so that
//! prefix is accepted beside the four the Java lists.

use super::process_data::ProcessData;
use super::system_program::SystemProgram;
use crate::imod::etomo::etomo_director;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Java `ParsePID implements Runnable`.
pub struct ParsePID {
    process: Arc<SystemProgram>,
    pid: Arc<Mutex<String>>,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    /// `ParseBackgroundPID.outFile`; `None` for a plain `ParsePID`.
    pub(crate) out_file: Option<PathBuf>,
}

impl ParsePID {
    /// Java `ParsePID(SystemProgram, StringBuffer, ProcessData)`.
    pub fn new(
        process: Arc<SystemProgram>,
        buf_pid: Arc<Mutex<String>>,
        process_data: Option<Arc<Mutex<ProcessData>>>,
    ) -> ParsePID {
        ParsePID {
            process,
            pid: buf_pid,
            process_data,
            out_file: None,
        }
    }

    /// Java `run`.
    pub fn run(&self) {
        // Wait for the csh thread to start
        while !self.process.is_started() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        // Once it is started scan the stderr output for the appropriate string
        while self.pid.lock().unwrap().is_empty() && !self.process.is_done() {
            self.parse_pid_string();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let pid = self.pid.lock().unwrap().clone();
        if let Some(process_data) = &self.process_data {
            process_data.lock().unwrap().set_pid(Some(&pid));
        }
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("PID:{pid}");
        }
    }

    /// Java `appendPID`.
    fn append_pid(&self, pid: &str) {
        self.pid.lock().unwrap().push_str(pid);
    }

    /// Java `parsePIDString`: walk the standard error output to parse the PID
    /// string (`ParseBackgroundPID`: the first line of the output file).
    fn parse_pid_string(&self) {
        let lines: Vec<String> = match &self.out_file {
            None => match self.process.get_std_error() {
                None => return,
                Some(stderr) => stderr,
            },
            Some(out_file) => {
                let Ok(file) = std::fs::File::open(out_file) else {
                    return;
                };
                let mut line = String::new();
                match std::io::BufReader::new(file).read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => vec![line.trim_end_matches(['\n', '\r']).to_owned()],
                }
            }
        };
        for line in &lines {
            if line.starts_with("Shell PID:")
                || line.contains("Python PID:")
                || line.starts_with("Runcom PID:")
                || line.starts_with("Windows PID:")
                || line.starts_with("Cygwin PID:")
            {
                let tokens: Vec<&str> = java_split_whitespace(line);
                if tokens.len() > 2 {
                    let mut found = false;
                    for token in tokens {
                        if found {
                            self.append_pid(token);
                            break;
                        } else if token.ends_with("PID:") {
                            found = true;
                        }
                    }
                }
            }
            if self.out_file.is_some() {
                break;
            }
        }
    }
}

/// `String.split("\\s+")`: a leading separator gives a leading empty token.
fn java_split_whitespace(line: &str) -> Vec<&str> {
    let mut tokens: Vec<&str> = line
        .split(|c: char| c.is_ascii_whitespace())
        .filter(|token| !token.is_empty())
        .collect();
    if line.starts_with(|c: char| c.is_ascii_whitespace()) {
        tokens.insert(0, "");
    }
    tokens
}
