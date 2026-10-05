//! `IMOD/Etomo/src/etomo/process/AlignLogGenerator.java`.
//!
//! Runs `alignlog` once per log section and writes each section's output to
//! its own `ta*.log` file.

use super::system_program::SystemProgram;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use std::io::Write;

/// Java `ERROR_LOG_NAME`.
pub const ERROR_LOG_NAME: &str = "taError";
/// Java `ANGLES_LOG_NAME`.
pub const ANGLES_LOG_NAME: &str = "taAngles";
/// Java `ROBUST_LOG_NAME`.
pub const ROBUST_LOG_NAME: &str = "taRobust";

/// Java static nested class `AlignLogGenerator.Mode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Java `TILT_ALIGN_LOGS` (default).
    TiltAlignLogs,
    /// Java `PROJECT_LOG`.
    ProjectLog,
}

/// Java `AlignLogGenerator implements Loggable`.
pub struct AlignLogGenerator {
    /// Java field `axisID`.
    pub axis_id: AxisID,

    // Updates done
    /// Java field `alignLogCommand`.
    pub align_log_command: Option<Vec<Option<String>>>,
    /// Java field `manager`.
    manager: &'static dyn BaseManager,
    /// Java field `mode`.
    mode: Mode,

    /// Java field `loggingName`.
    logging_name: Option<String>,
    /// Java field `stdOutput`.
    std_output: Option<Vec<String>>,
    /// Java field `stdError`.
    std_error: Option<Vec<String>>,
}

impl AlignLogGenerator {
    /// Java `AlignLogGenerator(BaseManager, AxisID, Mode)`.
    pub fn new(manager: &'static dyn BaseManager, id: AxisID, mode: Mode) -> AlignLogGenerator {
        let axis_id = id;
        // Do not use the -e flag for tcsh since David's scripts handle the failure
        // of commands and then report appropriately. The exception to this is the
        // com scripts which require the -e flag. RJG: 2003-11-06
        let mut align_log_command: Vec<Option<String>> = if id == AxisID::Only {
            vec![None; 4]
        } else {
            vec![None; 5]
        };
        align_log_command[0] = Some("python".to_string());
        align_log_command[1] = Some("-u".to_string());
        align_log_command[2] = Some(
            etomo_director::INSTANCE
                .get_python_script_path()
                .as_deref()
                .unwrap_or("null")
                .to_string()
                + "alignlog",
        );
        if id != AxisID::Only {
            align_log_command[4] = Some(axis_id.get_extension());
        }
        AlignLogGenerator {
            axis_id,
            align_log_command: Some(align_log_command),
            manager,
            mode,
            logging_name: None,
            std_output: None,
            std_error: None,
        }
    }

    /// Java `run`.
    pub fn run(&mut self) -> std::io::Result<()> {
        if self.mode == Mode::ProjectLog {
            self.run_argument("-p", &ProcessName::TILT_ALIGN.to_string())?;
        } else {
            self.run_argument("-a", ANGLES_LOG_NAME)?;
            self.run_argument("-c", "taCoordinates")?;
            self.run_argument("-e", ERROR_LOG_NAME)?;
            self.run_argument("-l", "taLocals")?;
            self.run_argument("-m", "taMappings")?;
            self.run_argument("-r", "taResiduals")?;
            self.run_argument("-s", "taSolution")?;
            self.run_argument("-b", "taBeamtilt")?;
            self.run_argument("-w", "taRobust")?;
        }
        Ok(())
    }

    /// Java private `runArgument`.
    fn run_argument(&mut self, argument: &str, log_file: &str) -> std::io::Result<()> {
        self.align_log_command.as_mut().unwrap()[3] = Some(argument.to_string());
        // `String[]` elements are never null here: the constructor fills every
        // slot but [3], which the line above sets.
        let command: Vec<String> = self
            .align_log_command
            .as_ref()
            .unwrap()
            .iter()
            .map(|element| element.clone().unwrap_or_else(|| "null".to_string()))
            .collect();
        let alignlog = SystemProgram::new_array(
            Some(self.manager),
            self.manager.get_property_user_dir(),
            Some(command),
            self.axis_id,
        );

        alignlog.run();

        self.std_output = alignlog.get_std_output();
        self.std_error = alignlog.get_std_error();

        self.logging_name = Some(log_file.to_string());
        let mut file_buffer = std::io::BufWriter::new(std::fs::File::create(
            self.manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string())
                + std::path::MAIN_SEPARATOR_STR
                + log_file
                + &self.axis_id.get_extension()
                + ".log",
        )?);

        match &self.std_output {
            None => match &self.std_error {
                None => file_buffer.write_all(b"alignlog produced no output")?,
                Some(std_error) => {
                    for line in std_error {
                        file_buffer.write_all(line.as_bytes())?;
                        file_buffer.write_all(b"\n")?;
                    }
                }
            },
            Some(std_output) => {
                for line in std_output {
                    file_buffer.write_all(line.as_bytes())?;
                    file_buffer.write_all(b"\n")?;
                }
            }
        }
        file_buffer.flush()?;
        Ok(())
    }

    /// Java `getName` (from `Loggable`).
    pub fn get_name(&self) -> Option<String> {
        self.logging_name.clone()
    }

    /// Java `getLogMessage` (from `Loggable`).
    pub fn get_log_message(&self) -> Result<Vec<String>, LogFileError> {
        let mut list: Vec<String> = Vec::new();
        match &self.std_output {
            None => match &self.std_error {
                None => list.push("alignlog produced no output for project log".to_string()),
                Some(std_error) => {
                    for line in std_error {
                        list.push(line.clone());
                    }
                }
            },
            Some(std_output) => {
                for line in std_output {
                    list.push(line.clone());
                }
            }
        }
        Ok(list)
    }
}

/// Java `AlignLogGenerator implements Loggable`: the interface view of the two
/// methods above, as `BaseManager.logMessage(Loggable, AxisID)` takes it.  A null
/// name prints as "null", as Java string concatenation does.
impl Loggable for AlignLogGenerator {
    /// Java `getName`.
    fn get_name(&self) -> String {
        AlignLogGenerator::get_name(self).unwrap_or_else(|| "null".to_string())
    }

    /// Java `getLogMessage`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        AlignLogGenerator::get_log_message(self)
            .map(|list| list.into_iter().map(Some).collect())
            .map_err(|e| LoggableException::LogFile(e.to_string()))
    }
}
