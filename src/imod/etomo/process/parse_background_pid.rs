//! `IMOD/Etomo/src/etomo/process/ParseBackgroundPID.java`.
//!
//! The process ID is read from the first line of an output file written by a
//! background job; the scan itself is `ParsePID`'s, with `outFile` set.

use super::parse_pid::ParsePID;
use super::process_data::ProcessData;
use super::system_program::SystemProgram;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Java `ParseBackgroundPID(SystemProgram, StringBuffer, File, ProcessData)`.
pub fn parse_background_pid(
    process: Arc<SystemProgram>,
    buf_pid: Arc<Mutex<String>>,
    out_file: PathBuf,
    process_data: Option<Arc<Mutex<ProcessData>>>,
) -> ParsePID {
    let mut parse = ParsePID::new(process, buf_pid, process_data);
    parse.out_file = Some(out_file);
    parse
}
